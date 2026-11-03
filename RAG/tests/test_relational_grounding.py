import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.relational_grounding import (AntecedentGrounder, AuditFailure, DIMENSIONS, anchored_schema,
    audit_claims, audit_input, audit_schema, validate_audit)
from experiments.compare_relational import main, run_controls, run_experiment
from rag_pensiones.grounding import OllamaGrounder
from test_grounding import FakeGrounderGenerator
from test_compare_retrieval_k import _make_valid_index, MockGenerator, MockEmbedder, _make_standard_grounding_responses
from test_validate_reranker import queries


QUOTE = "El coste proyectado del sistema alcanza el 8% del PIB en 2040."


def public():
    return {"status": "answered", "claims": [{"text": "El coste será del 8% del PIB en 2040.",
            "evidence": [{"source_id": "S1", "quote": QUOTE}]}]}


def assessment(cid="A1"):
    return {"claim_id": cid, "dimensions": {d: {"status": "match", "evidence_ids": ["E1"]} for d in DIMENSIONS},
            "claim_relation": "nivel final", "source_relation": "nivel final", "anchors": [{"evidence_id": "E1", "quote": QUOTE}]}


def payload():
    return {"audits": [assessment()]}


def messages(sources):
    return [{"role": "system", "content": "original"}, {"role": "user", "content": json.dumps({"fuentes": sources})}]


class RelationalTests(unittest.TestCase):
    def test_anchor_schema_uses_only_literal_fixed_spans_of_assigned_quote(self):
        source=public(); source['claims'][0]['evidence'][0]['quote']=' '.join(['Texto literal con contexto.']*20)
        items=audit_input(source)
        branch=audit_schema(items)['properties']['audits']['items']['anyOf'][0]
        alternatives=branch['properties']['anchors']['items']['anyOf']
        self.assertEqual(len(alternatives),1)
        self.assertEqual(alternatives[0]['properties']['evidence_id'],{'const':'E1'})
        for span in alternatives[0]['properties']['quote']['enum']:
            self.assertIn(span,source['claims'][0]['evidence'][0]['quote'])
            self.assertLessEqual(len(span),220)

    def test_windows_keep_prior_context_and_decimals_without_crossing_sources(self):
        first = "El régimen está cerrado. Su coste alcanza el 1.2% del PIB."
        second = "Otra fuente contiene información diferente."
        msg = messages([{"source_id": "S1", "text": first}, {"source_id": "S2", "text": second}])
        original = copy.deepcopy(msg)
        branches = anchored_schema(msg)["anyOf"][0]["properties"]["evidence"]["items"]["anyOf"]
        self.assertEqual(branches[0]["properties"]["quote"]["enum"], ["El régimen está cerrado.", first])
        self.assertEqual(branches[1]["properties"]["quote"]["enum"], [second])
        self.assertEqual(msg, original)

    def test_long_sentences_not_truncated_or_spliced(self):
        long = "x" * 701 + "."
        tail = "El segundo dato está disponible."
        msg = messages([{"source_id": "S1", "text": long + " " + tail}])
        enum = anchored_schema(msg)["anyOf"][0]["properties"]["evidence"]["items"]["anyOf"][0]["properties"]["quote"]["enum"]
        self.assertEqual(enum, [tail])
        self.assertEqual(len(anchored_schema(messages([{"source_id": "S1", "text": long}]))["anyOf"]), 1)

    def test_duplicate_source_rejected(self):
        with self.assertRaises(ValueError):
            anchored_schema(messages([{"source_id": "S1", "text": QUOTE}] * 2))

    def test_override_selection_repair_only_without_mutating_messages(self):
        fake = FakeGrounderGenerator(responses=[{}, {}, {}])
        g = AntecedentGrounder(generator=fake)
        msg = messages([{"source_id": "S1", "text": QUOTE}]); original = copy.deepcopy(msg)
        for stage in ("selection", "repair_selection", "generation"):
            g._chat_step(stage, msg, {}, fake.digest)
        self.assertNotEqual(fake.history[0][0][0]["content"], "original")
        self.assertNotEqual(fake.history[1][0][0]["content"], "original")
        self.assertEqual(fake.history[2], (msg, {}))
        self.assertEqual(msg, original)

    def test_audit_supported_and_no_mutation_separate_trace(self):
        source = public(); original = copy.deepcopy(source)
        fake = FakeGrounderGenerator(responses=[payload()])
        audited = audit_claims(source, OllamaGrounder(generator=fake), fake.digest)
        self.assertEqual(audited["effective_status"], "answered")
        self.assertEqual(source, original)
        self.assertEqual(len(audited["calls"]), 1)
        self.assertEqual(audited["calls"][0]["trace"]["model_digest"], fake.digest)
        self.assertNotIn("source_id", fake.history[0][0][1]["content"])

    def test_each_claim_is_physically_isolated(self):
        source = public(); other = copy.deepcopy(source["claims"][0]);other["text"] = "SECRET_OTHER_CLAIM"
        source["claims"].append(other)
        fake = FakeGrounderGenerator(responses=[payload(), {"audits": [assessment("A2")]}])
        result = audit_claims(source, OllamaGrounder(generator=fake), fake.digest)
        self.assertEqual(len(result["calls"]), 2)
        self.assertNotIn("SECRET_OTHER_CLAIM", fake.history[0][0][1]["content"])
        self.assertEqual(len(json.loads(fake.history[1][0][1]["content"])["claims"]), 1)

    def test_conflict_and_missing_derived_fail_closed_not_model_verdict(self):
        for status, verdict in [("conflict", "contradicted"), ("missing", "insufficient_evidence")]:
            p = payload();p["audits"][0]["dimensions"]["relacion"] = {"status": status, "evidence_ids": ["E1"] if status == "conflict" else []}
            fake = FakeGrounderGenerator(responses=[p])
            result = audit_claims(public(), OllamaGrounder(generator=fake), fake.digest)
            self.assertEqual(result["verdicts"][0]["verdict"], verdict)
            self.assertEqual(result["published_claims"], [])
            self.assertEqual(result["effective_status"], "insufficient_evidence")

    def test_reject_malformed_assignment_anchor_status_coverage_and_types(self):
        edits = [lambda p:p.update(verdict="supported"), lambda p:p.update(audits=[]),
            lambda p:p["audits"].append(copy.deepcopy(p["audits"][0])),
            lambda p:p["audits"][0].update(claim_id="A2"),
            lambda p:p["audits"][0].update(anchors=[{"evidence_id":"E1", "quote":"fabricated"}]),
            lambda p:p["audits"][0].update(anchors=[]),
            lambda p:p["audits"][0].update(claim_relation=True),
            lambda p:p["audits"][0]["dimensions"].pop("unidad"),
            lambda p:p["audits"][0]["dimensions"]["relacion"].update(status="not_applicable",evidence_ids=[]),
            lambda p:p["audits"][0]["dimensions"]["valor"].update(status="missing"),
            lambda p:p["audits"][0]["dimensions"]["valor"].update(evidence_ids=["E2"]),
            lambda p:p["audits"][0]["dimensions"]["valor"].update(evidence_ids=["E1","E1"]),
            lambda p:p["audits"][0]["anchors"].append(copy.deepcopy(p["audits"][0]["anchors"][0]))]
        for edit in edits:
            p=payload(); edit(p)
            with self.subTest(p=p), self.assertRaises(ValueError):
                validate_audit(p, audit_input(public()))

    def test_not_applicable_allowed_only_nonasserted_dimensions(self):
        p=payload();p["audits"][0]["dimensions"]["unidad"]={"status":"not_applicable","evidence_ids":[]}
        self.assertEqual(validate_audit(p,audit_input(public()))[0]["verdict"],"supported")

    def test_abstention_skips_and_model_drift_and_budget_fail_before_publication(self):
        fake=FakeGrounderGenerator(responses=[payload()]);g=OllamaGrounder(generator=fake)
        self.assertEqual(audit_claims({"status":"insufficient_evidence","claims":[]},g,fake.digest)["status"],"skipped")
        self.assertEqual(fake.call_count,0)
        with self.assertRaises(AuditFailure):audit_claims(public(),g,"wrong_digest")
        self.assertEqual(fake.call_count,0)
        drift=FakeGrounderGenerator(responses=[payload()],digest_mutations={1:"changed"})
        with self.assertRaises(AuditFailure):audit_claims(public(),OllamaGrounder(generator=drift),drift.digest)

    def test_invalid_audit_retains_current_and_previous_calls_and_runner_diagnostics(self):
        source=public();source['claims'].append(copy.deepcopy(source['claims'][0]))
        bad={'audits':[assessment('A2')]};bad['audits'][0]['anchors'][0]['quote']='inventada'
        fake=FakeGrounderGenerator(responses=[payload(),bad])
        with self.assertRaises(AuditFailure) as caught:
            audit_claims(source,OllamaGrounder(generator=fake),fake.digest)
        calls=caught.exception.calls
        self.assertEqual(len(calls),2)
        self.assertEqual([c['validation_status'] for c in calls],['validated','invalid'])
        self.assertEqual(calls[1]['payload'],bad)
        self.assertIn('prompt_sha256',calls[1]['trace'])
        controls={'schema_version':'relational-controls-v1','cases':[{'id':'c','claim':public()['claims'][0]['text'],'evidence':[QUOTE]}]}
        g=OllamaGrounder(generator=MockGenerator())
        failure=AuditFailure('bad anchor',calls)
        with patch('experiments.compare_relational.audit_claims',side_effect=failure),patch.object(g,'verify',return_value={'verdicts':[{'verdict':'supported'}]}),patch('sys.stderr',io.StringIO()):
            result=run_controls(controls,grounder=g)
        self.assertEqual(result['run_status'],'error')
        self.assertEqual(result['cases'][0]['conditions']['dimensional']['audit_failure_calls'],calls)

    def test_runner_shared_search_native_validation_alternation_and_derived_reuse(self):
        from rag_pensiones.search import search
        base=OllamaGrounder(generator=MockGenerator());ant=AntecedentGrounder(generator=MockGenerator())
        aud=OllamaGrounder(generator=MockGenerator())
        with patch('experiments.compare_relational.search',wraps=search) as searched,patch('sys.stderr',io.StringIO()):
            result=run_experiment(_make_valid_index(),queries(),baseline=base,antecedent=ant,auditor=aud,embedder=MockEmbedder())
        self.assertEqual(searched.call_count,2)
        self.assertTrue(all(c.kwargs['top_k']==4 for c in searched.call_args_list))
        self.assertEqual(result['run_status'],'completed')
        self.assertEqual(result['cases'][1]['condition_order'],['antecedent','baseline'])
        self.assertEqual(aud.generator.call_count,0)
        self.assertTrue(all(r['conditions']['antecedent_plus_audit']['generation_reused_from']=='antecedent' for r in result['cases']))

    def test_runner_failures_keep_checkpoint_running_and_continue_other_arm(self):
        base=OllamaGrounder(generator=MockGenerator());ant=AntecedentGrounder(generator=MockGenerator())
        def run(*args,**kwargs):return run_experiment(_make_valid_index(),queries(),baseline=base,antecedent=ant,auditor=base,embedder=MockEmbedder(),**kwargs)
        with patch.object(ant,'answer',side_effect=RuntimeError('backend')),patch('sys.stderr',io.StringIO()):
            result=run()
        self.assertEqual(result['run_status'],'error')
        self.assertTrue(all(c['conditions']['baseline']['status']=='insufficient_evidence' for c in result['cases']))
        with tempfile.TemporaryDirectory(dir='tmp') as directory:
            path=Path(directory)/'result.json'
            from rag_pensiones.search import search
            counter=[0]
            def interrupt_second(*args,**kwargs):
                counter[0]+=1
                if counter[0]==2:raise KeyboardInterrupt()
                return search(*args,**kwargs)
            with patch('experiments.compare_relational.search',side_effect=interrupt_second),patch('sys.stderr',io.StringIO()):
                with self.assertRaises(KeyboardInterrupt):run(output=path)
            saved=json.loads(path.read_text(encoding='utf-8'))
            self.assertEqual(saved['run_status'],'running')
            self.assertEqual(len(saved['cases'][0]['conditions']),3)
            with self.assertRaises(ValueError):run(output=path)

    def test_real_generated_answer_is_validated_then_audited_without_regeneration(self):
        base=OllamaGrounder(generator=MockGenerator(_make_standard_grounding_responses()*2))
        ant=AntecedentGrounder(generator=MockGenerator(_make_standard_grounding_responses()*2))
        p=payload();p['audits'][0]['anchors'][0]['quote']='16,1%'
        fake=FakeGrounderGenerator(responses=[p,p],digest=base.generator.digest)
        with patch('sys.stderr',io.StringIO()):
            result=run_experiment(_make_valid_index(),queries(),baseline=base,antecedent=ant,
                auditor=OllamaGrounder(generator=fake),embedder=MockEmbedder())
        self.assertEqual(result['run_status'],'completed')
        self.assertEqual(ant.generator.call_count,6)
        self.assertEqual(fake.call_count,2)
        self.assertTrue(all(c['conditions']['antecedent_plus_audit']['status']=='answered' for c in result['cases']))

    def test_retrieval_errors_checkpoint_each_condition_and_invalid_result_never_audited(self):
        g=OllamaGrounder(generator=MockGenerator())
        with patch('experiments.compare_relational.search',side_effect=RuntimeError('offline')),patch('experiments.compare_relational.checkpoint') as cp:
            result=run_experiment(_make_valid_index(),queries(),baseline=g,antecedent=g,auditor=g,embedder=MockEmbedder())
        self.assertEqual(cp.call_count,7)
        self.assertEqual(result['run_status'],'error')
        with patch.object(g,'answer',return_value={'status':'answered'}),patch('experiments.compare_relational.audit_claims') as audit,patch('sys.stderr',io.StringIO()):
            result=run_experiment(_make_valid_index(),queries(),baseline=g,antecedent=g,auditor=g,embedder=MockEmbedder())
        audit.assert_not_called()
        self.assertEqual(result['run_status'],'error')

    def test_cli_and_controls_reject_labels_before_network(self):
        with patch('experiments.compare_relational.OllamaGrounder') as ctor:
            with self.assertRaises(ValueError):run_controls({'schema_version':'relational-controls-v1','cases':[{'id':'c','claim':'x','evidence':[QUOTE],'expected':'supported'}]})
            ctor.assert_not_called()
        with tempfile.TemporaryDirectory(dir='tmp') as directory:
            q=Path(directory)/'bad.json';q.write_text(json.dumps(dict(queries(),gold='secret')))
            with patch('experiments.compare_relational.OllamaGrounder') as ctor,patch('sys.stderr',io.StringIO()):
                self.assertEqual(main(['--queries',str(q),'--output',str(Path(directory)/'out.json')]),1)
                ctor.assert_not_called()


if __name__ == '__main__':
    unittest.main()
