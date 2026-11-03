import copy
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.validate_reranker import main, run_validation, validate_queries, validate_reranking
from rag_pensiones.grounding import OllamaGrounder
from rag_pensiones.search import SearchHit, search as actual_search
from test_compare_retrieval_k import _make_valid_index, MockGenerator, MockEmbedder


def queries():
    return {"schema_version": "reranker-validation-queries-v1", "cases": [{"id": "q1", "query": "Pregunta uno"}, {"id": "q2", "query": "Pregunta dos"}]}


class FakeReranker:
    def rank(self, query, hits):
        self.last_scores = [{"chunk_id": h.chunk.chunk_id, "score": float(len(hits) - i)} for i, h in enumerate(hits)]
        return list(hits)


def grounder(): return OllamaGrounder(generator=MockGenerator())


class ValidationTests(unittest.TestCase):
    def test_closed_schema_empty_strings_duplicates_size_and_no_gold(self):
        bad = [None, {"schema_version": "bad", "cases": []}, dict(queries(), gold="SECRET")]
        for modify in [lambda q:q["cases"][0].update(query=""), lambda q:q["cases"][1].update(id="q1"), lambda q:q["cases"][0].update(query="x" * 1001), lambda q:q["cases"][0].update(reference_answer="SECRET")]:
            q=queries();modify(q);bad.append(q)
        for q in bad:
            with self.subTest(q=q), self.assertRaises(ValueError):validate_queries(q)

    def test_full_ranking_once_top4_order_and_real_grounder_validation(self):
        g=grounder();e=MockEmbedder()
        with patch("experiments.validate_reranker.search", wraps=actual_search) as search, patch("sys.stderr", io.StringIO()):
            result=run_validation(_make_valid_index(6), queries(), FakeReranker(), g, e)
        self.assertEqual(search.call_count,2)
        self.assertTrue(all(c.kwargs["top_k"]==6 for c in search.call_args_list))
        self.assertEqual(result["cases"][0]["condition_order"],["baseline","reranker"])
        self.assertEqual(result["cases"][1]["condition_order"],["reranker","baseline"])
        self.assertEqual(result["run_status"],"completed")
        self.assertTrue(all(len(c["retrieved"])==4 for r in result["cases"] for c in r["conditions"].values()))
        self.assertEqual([h[0] for h in e.calls], ['task: search result | query: Pregunta uno','task: search result | query: Pregunta dos'])

    def test_rerank_boundary_permutation_metadata_cosine_and_scores(self):
        dense=actual_search(_make_valid_index(),"query",top_k=5,client=MockEmbedder())
        rr=FakeReranker();ranked=rr.rank("q",dense);validate_reranking(dense,ranked,rr.last_scores)
        variants=[ranked[:-1], [ranked[0]]*5, [replace(ranked[0],rank=True)]+ranked[1:], [replace(ranked[0],score=True)]+ranked[1:], [replace(ranked[0],score=float('nan'))]+ranked[1:], [replace(ranked[0],chunk=replace(ranked[0].chunk,text='CHANGED'))]+ranked[1:]]
        for changed in variants:
            with self.assertRaises(ValueError):validate_reranking(dense,changed,rr.last_scores)
        for score in [True,float('inf'),float('nan')]:
            scores=copy.deepcopy(rr.last_scores);scores[0]['score']=score
            with self.assertRaises(ValueError):validate_reranking(dense,ranked,scores)

    def test_reranker_failure_continues_baseline_in_both_orders(self):
        rr=FakeReranker()
        with patch.object(rr,'rank',side_effect=RuntimeError('backend')),patch('sys.stderr',io.StringIO()):
            result=run_validation(_make_valid_index(),queries(),rr,grounder(),MockEmbedder())
        self.assertEqual(result['summary']['baseline']['valid_results'],2)
        self.assertEqual(result['summary']['reranker']['errors'],2)
        self.assertTrue(all('retrieved' not in row['conditions']['reranker'] for row in result['cases']))

    def test_invalid_generation_keeps_retrieval_without_fake_success(self):
        g=grounder()
        with patch.object(g,'answer',return_value={'status':'answered'}),patch('sys.stderr',io.StringIO()):
            result=run_validation(_make_valid_index(),queries(),FakeReranker(),g,MockEmbedder())
        self.assertEqual(result['run_status'],'error')
        self.assertEqual(result['summary']['baseline']['errors'],2)
        self.assertTrue(all('retrieved' in c for row in result['cases'] for c in row['conditions'].values()))

    def test_invalid_model_pin_or_queries_before_search(self):
        for mode in ['model','pin','query']:
            g=grounder();q=queries()
            if mode=='model':g.generator.model='wrong'
            if mode=='pin':g.generator.digest=''
            if mode=='query':q['cases'][0]['extra']='GOLD'
            with patch('experiments.validate_reranker.search') as search:
                with self.assertRaises(ValueError):run_validation(_make_valid_index(),q,FakeReranker(),g,MockEmbedder())
                search.assert_not_called()

    def test_digest_drift_does_not_call_answer(self):
        g=grounder()
        with patch.object(g.generator,'get_digest',side_effect=['first']+['changed']*4),patch.object(g,'answer') as answer,patch('sys.stderr',io.StringIO()):
            result=run_validation(_make_valid_index(),queries(),FakeReranker(),g,MockEmbedder())
        answer.assert_not_called();self.assertEqual(result['run_status'],'error')

    def test_partial_checkpoint_never_completed_and_output_no_clobber(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()/'tmp') as directory:
            output=Path(directory)/'results.json';g=grounder();original=g.answer;calls=[]
            def interrupted(**kw):
                if calls:raise KeyboardInterrupt()
                calls.append(True);return original(**kw)
            with patch.object(g,'answer',side_effect=interrupted),patch('sys.stderr',io.StringIO()):
                with self.assertRaises(KeyboardInterrupt):run_validation(_make_valid_index(),queries(),FakeReranker(),g,MockEmbedder(),output)
            partial=json.loads(output.read_text(encoding='utf-8'));self.assertEqual(partial['run_status'],'running')
            self.assertEqual(len(partial['cases'][0]['conditions']),1)
            with self.assertRaises(ValueError):run_validation(_make_valid_index(),queries(),FakeReranker(),g,MockEmbedder(),output)

    def test_inplace_reranker_cannot_change_original_candidates(self):
        rr=FakeReranker();original=rr.rank
        def mutate(query,hits):
            hits[0]=replace(hits[0],chunk=replace(hits[0].chunk,text='CHANGED'));return original(query,hits)
        with patch.object(rr,'rank',side_effect=mutate),patch('sys.stderr',io.StringIO()):
            result=run_validation(_make_valid_index(),queries(),rr,grounder(),MockEmbedder())
        self.assertEqual(result['summary']['reranker']['errors'],2)
        self.assertEqual(result['summary']['baseline']['valid_results'],2)

    def test_cli_bad_queries_rejected_before_models(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()/'tmp') as directory:
            source=Path(directory)/'q.json';source.write_text('{"gold":"SECRET"}',encoding='utf-8')
            with patch('experiments.validate_reranker.LocalReranker') as model,patch('sys.stderr',io.StringIO()):
                code=main(['--queries',str(source),'--output',str(Path(directory)/'out.json'),'--reranker-path','work/models/mmarco'])
                self.assertEqual(code,1);model.assert_not_called()
