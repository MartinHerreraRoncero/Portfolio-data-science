"""Deterministic policy tests; semantic reliability is measured by separate controls."""
import copy
import json
import unittest

from experiments.answer_alignment import (
    AlignmentChecker, DIMENSIONS, REFUSAL, apply_alignment, decision_from_dimensions,
)
from experiments.compare_alignment import run_controls, validate_controls


def view():
    return dict(status='answered',claims=[dict(text='La cobertura nacional es del 80% en 2036.',
        evidence=[dict(source_id='S1',quote='La cobertura nacional proyectada para 2036 es del 80%.')])],reason='')


def facets():
    return {k:dict(requirement='Requisito solicitado',outcome='match',reason='Coincide con pregunta y cita.') for k in DIMENSIONS}


class Generator:
    model='qwen3.5:9b'
    def __init__(self,payload=None,fail=False,change=False):
        self.payload=payload if payload is not None else facets()
        self.fail=fail;self.change=change;self.count=0;self.calls=[]
    def get_digest(self):return ('b' if self.change and self.count else 'a')*64
    def chat(self,messages,response_schema):
        call=dict(messages=copy.deepcopy(messages),schema=copy.deepcopy(response_schema));self.calls.append(call);self.count+=1
        if self.fail:
            call['error']='Offline transport failure';raise RuntimeError(call['error'])
        raw=dict(model=self.model,done=True,done_reason='stop',message=dict(role='assistant',content=json.dumps(self.payload)),
                 prompt_eval_count=150,eval_count=80,eval_duration=1000,total_duration=2000)
        call['raw_response']=copy.deepcopy(raw)
        return raw


class AlignmentTests(unittest.TestCase):
    def test_any_mismatch_or_uncertain_blocks_and_matching_view_is_unchanged(self):
        for dim in DIMENSIONS:
            for outcome in ('mismatch','uncertain'):
                with self.subTest(dim=dim,outcome=outcome):
                    p=facets();p[dim]['outcome']=outcome
                    checker=AlignmentChecker(Generator(p));original=view();before=copy.deepcopy(original)
                    audit=checker.check('¿Qué cobertura regional habrá en 2036?',original,expected_digest='a'*64)
                    out=apply_alignment(original,audit)
                    self.assertEqual((out['status'],out['claims'],out['reason']),('insufficient_evidence',[],REFUSAL))
                    self.assertEqual(original,before)
        original=view();audit=AlignmentChecker(Generator()).check('Pregunta',original,expected_digest='a'*64)
        out=apply_alignment(original,audit)
        self.assertEqual(out,original);self.assertIsNot(out,original)

    def test_abstention_is_skipped_without_inference_or_reason_rewriting(self):
        original=dict(status='insufficient_evidence',claims=[],reason='No hay dato para ese ámbito.')
        gen=Generator(fail=True);audit=AlignmentChecker(gen).check('Pregunta',original,expected_digest='a'*64)
        self.assertEqual(gen.count,0);self.assertEqual(audit['decision'],'abstained')
        self.assertEqual(apply_alignment(original,audit),original)

    def test_global_boolean_cannot_override_facets_and_missing_dimensions_fail(self):
        p=facets();p['scope']['outcome']='mismatch'
        with self.assertRaisesRegex(ValueError,'disagrees'):
            apply_alignment(view(),dict(decision='accept',dimensions=p))
        p=facets();p.pop('period')
        with self.assertRaisesRegex(ValueError,'dimensions'):
            decision_from_dimensions(p)

    def test_unknown_labels_and_incoherent_not_required_are_rejected(self):
        for replacement in [dict(requirement='',outcome='match',reason='Falta requisito.'),dict(requirement='Ámbito',outcome='not_required',reason='Incoherente.'),dict(requirement='Ámbito',outcome='yes',reason='Etiqueta inválida.'),dict(requirement='Ámbito',outcome='match',reason='')]:
            p=facets();p['scope']=replacement
            with self.assertRaises(ValueError):decision_from_dimensions(p)
        p=facets();p['period']=dict(requirement='',outcome='not_required',reason='No se pregunta por un año.')
        self.assertEqual(decision_from_dimensions(p),'accept')

    def test_digest_change_and_budget_abort_are_not_safe_abstentions(self):
        gen=Generator(change=True)
        with self.assertRaisesRegex(RuntimeError,'digest changed'):
            AlignmentChecker(gen).check('Pregunta',view(),expected_digest='a'*64)
        gen=Generator()
        with self.assertRaisesRegex(ValueError,'budget'):
            AlignmentChecker(gen).check('Pregunta larga '*1000,view(),expected_digest='a'*64)
        self.assertEqual(gen.count,0)

    def test_rejection_clears_context_bindings_but_keeps_original_for_diagnostics(self):
        original=view();original['bindings']=[dict(claim_id='C1',package_evidence_ids=['E1'])]
        p=facets();p['nature']['outcome']='uncertain'
        out=apply_alignment(original,dict(decision='reject',dimensions=p))
        self.assertEqual(out['bindings'],[])
        self.assertEqual(original['bindings'][0]['claim_id'],'C1')

    def test_real_messages_include_question_and_only_published_claims_not_hidden_metadata(self):
        gen=Generator();original=view();original['secret_gold']='not_for_the_judge'
        audit=AlignmentChecker(gen).check('¿Qué cobertura habrá?',original,expected_digest='a'*64)
        user=json.loads(gen.calls[0]['messages'][1]['content'])
        self.assertEqual(set(user),{'pregunta','respuesta'})
        self.assertNotIn('secret_gold',user['respuesta'])
        self.assertEqual(audit['trace']['messages_budget_bytes'],6500)

    def test_control_labels_are_rejected_and_transport_failure_retains_raw_diagnostics(self):
        data=dict(schema_version='answer-alignment-controls-v1',notice='Synthetic',cases=[dict(id='x',query='Pregunta',answer=view())])
        bad=copy.deepcopy(data);bad['cases'][0]['expected']='reject'
        with self.assertRaisesRegex(ValueError,'labels'):validate_controls(bad)
        gen=Generator(fail=True);r=run_controls(data,None,checker=AlignmentChecker(gen))
        self.assertEqual(r['run_status'],'error')
        self.assertEqual(r['cases'][0]['input_view'],view())
        self.assertEqual(r['cases'][0]['calls'][0]['error'],'Offline transport failure')
        self.assertNotIn('output_view',r['cases'][0])

    def test_malformed_model_response_retains_raw_and_is_not_an_abstention(self):
        data=dict(schema_version='answer-alignment-controls-v1',notice='Synthetic',cases=[dict(id='x',query='Pregunta',answer=view())])
        gen=Generator({'scope':'yes'});r=run_controls(data,None,checker=AlignmentChecker(gen))
        row=r['cases'][0];self.assertEqual(row['status'],'error')
        self.assertIn('raw_response',row['calls'][0]);self.assertNotIn('output_view',row)


if __name__=='__main__':unittest.main()
