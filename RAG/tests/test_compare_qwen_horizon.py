"""Shared retrieval, model identity, counterbalancing and cleanup failure semantics."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from experiments import compare_qwen_horizon as trial
from rag_pensiones.search import DEFAULT_INDEX_PATH, load_index, SearchHit
from rag_pensiones.qwen_embeddings import validate_index


class Control:
    def __init__(self, fail_cleanup=False):self.releases=[];self.fail_cleanup=fail_cleanup
    def unload(self,model):
        self.releases.append(model)
        if self.fail_cleanup and len(self.releases)>2:raise RuntimeError('cleanup failed')
        return {'done':True,'model':model}
    def ensure_no_other(self,model):return {'models':[]}


class Grounder:
    def __init__(self,model,fail=False):
        self.model=model;self.fail=fail;self.seen=[]
        self.generator=SimpleNamespace(calls=[],get_digest=lambda:model+'digest')
    def answer(self,query,hits,**kwargs):
        self.seen.append((query,list(hits),kwargs['expected_digest']))
        self.generator.calls.append({'fake':True})
        if self.fail:raise RuntimeError('generation failed')
        hits.pop()
        return {'status':'insufficient_evidence','claims':[]}


class ModelComparisonTests(unittest.TestCase):
    def setup_trial(self):
        index=load_index(DEFAULT_INDEX_PATH)
        hits=[SearchHit(i+1,.5,c) for i,(c,v) in enumerate(validate_index(index)[:4])]
        queries={'schema_version':'reranker-validation-queries-v1','cases':[{'id':f'q{i}','query':f'Pregunta {i}'} for i in range(3)]}
        models={k:Grounder(name) for k,name in trial.MODELS.items()}
        return index,hits,queries,models

    def test_both_models_get_identical_sources_one_search_per_query_and_opposite_order_in_replica(self):
        index,hits,queries,models=self.setup_trial(); control=Control()
        with patch.object(trial,'search',return_value=hits) as search,patch.object(trial,'validate_grounded_result') as validate:
            first=trial.run(index,queries,None,embedder=SimpleNamespace(calls=[]),models=models,control=control)
            second=trial.run(index,queries,None,reverse=True,embedder=SimpleNamespace(calls=[]),models=models,control=Control())
        self.assertEqual(search.call_count,6)
        self.assertEqual(validate.call_count,12)
        first_orders={c['id']:c['condition_order'] for c in first['cases']}
        self.assertEqual([c['id'] for c in second['cases']],['q2','q1','q0'])
        for c in second['cases']:self.assertEqual(c['condition_order'],list(reversed(first_orders[c['id']])))
        for g in models.values():
            self.assertTrue(all(seen_hits==hits for query,seen_hits,digest in g.seen))
            self.assertTrue(all(digest==g.model+'digest' for query,seen_hits,digest in g.seen))
        self.assertEqual(len(hits),4)
        self.assertEqual(len(control.releases),8)

    def test_generation_failure_preserves_calls_releases_model_and_runs_other_arm(self):
        index,hits,queries,models=self.setup_trial();models['horizon7b'].fail=True;control=Control()
        with patch.object(trial,'search',return_value=hits),patch.object(trial,'validate_grounded_result'):
            d=trial.run(index,queries,None,embedder=SimpleNamespace(calls=[]),models=models,control=control)
        self.assertEqual(d['run_status'],'error')
        for c in d['cases']:
            self.assertEqual(c['conditions']['horizon7b']['status'],'error')
            self.assertEqual(c['conditions']['horizon7b']['calls'],[{'fake':True}])
            self.assertEqual(c['conditions']['qwen9b']['status'],'insufficient_evidence')
        self.assertEqual(len(control.releases),8)

    def test_shared_retrieval_failure_is_two_errors_and_zero_generation_calls(self):
        index,hits,queries,models=self.setup_trial()
        with patch.object(trial,'search',side_effect=ValueError('bad retrieval')):
            d=trial.run(index,queries,None,embedder=SimpleNamespace(calls=[]),models=models,control=Control())
        self.assertEqual(d['run_status'],'error')
        self.assertTrue(all(o['status']=='error' and not o['calls'] for c in d['cases'] for o in c['conditions'].values()))
        self.assertTrue(all(not g.seen for g in models.values()))

    def test_cleanup_failure_cannot_be_published_as_success(self):
        index,hits,queries,models=self.setup_trial()
        with patch.object(trial,'search',return_value=hits),patch.object(trial,'validate_grounded_result'):
            d=trial.run(index,queries,None,embedder=SimpleNamespace(calls=[]),models=models,control=Control(True))
        self.assertEqual(d['run_status'],'error')
        self.assertTrue(all(o['status']=='error' and o['cleanup_error']=='cleanup failed' for c in d['cases'] for o in c['conditions'].values()))

    def test_recorded_clients_pin_exact_requested_model(self):
        for model in trial.MODELS.values():self.assertEqual(trial.RecordedModel(model).model,model)
        with self.assertRaises(ValueError):trial.LocalControl().unload('unrelated-model')


if __name__=='__main__':unittest.main()
