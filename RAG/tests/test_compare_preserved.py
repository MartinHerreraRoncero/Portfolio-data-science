"""Trial isolates arm failures, shares retrieval and preserves raw diagnostics."""
import copy
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from experiments import compare_preserved as trial
from rag_pensiones.search import load_index,DEFAULT_INDEX_PATH,SearchHit,validate_index


class Grounder:
    model='qwen3.5:9b'
    def __init__(self,fail=False):
        self.fail=fail;self.seen=[]
        self.generator=SimpleNamespace(calls=[],get_digest=lambda:'a'*64)
    def answer(self,query,hits,expected_digest):
        self.seen.append(copy.deepcopy(hits));self.expanded_hits=copy.deepcopy(hits)
        self.generator.calls.append({'recorded':'call'})
        hits.pop()
        if self.fail: raise ValueError('Arm failed')
        return dict(status='insufficient_evidence',claims=[],reason='No hay dato.')


class RunnerTests(unittest.TestCase):
    def setup_trial(self):
        index=load_index(DEFAULT_INDEX_PATH)
        hits=[SearchHit(i+1,.7,c) for i,(c,v) in enumerate(validate_index(index)[:4])]
        queries={'schema_version':'reranker-validation-queries-v1','cases':[dict(id=f'q{i}',query=f'Pregunta {i}') for i in range(6)]}
        models={k:Grounder() for k in ('baseline','reconstructed','preserved')}
        return index,hits,queries,models

    def test_shared_retrieval_rotates_all_orders_without_mutating_hits(self):
        index,hits,queries,models=self.setup_trial()
        with patch.object(trial,'validate_catalog'),patch.object(trial,'search',return_value=hits) as search,patch.object(trial,'validate_grounded_result') as validate:
            r=trial.run(index,queries,{},[],None,models=models,embedder=SimpleNamespace(calls=[]))
        self.assertEqual(search.call_count,6);self.assertEqual(validate.call_count,18)
        self.assertEqual(len({tuple(c['condition_order']) for c in r['cases']}),6)
        self.assertTrue(all(h==hits for g in models.values() for h in g.seen))
        self.assertEqual(len(hits),4)
        self.assertEqual(r['run_status'],'completed')

    def test_failed_arm_retains_calls_and_other_arms_continue(self):
        index,hits,queries,models=self.setup_trial();models['preserved'].fail=True
        with patch.object(trial,'validate_catalog'),patch.object(trial,'search',return_value=hits),patch.object(trial,'validate_grounded_result'):
            r=trial.run(index,queries,{},[],None,models=models,embedder=SimpleNamespace(calls=[]))
        self.assertEqual(r['run_status'],'error')
        for c in r['cases']:
            self.assertEqual(c['conditions']['preserved']['calls'],[{'recorded':'call'}])
            self.assertEqual(c['conditions']['preserved']['status'],'error')
            self.assertEqual(c['conditions']['baseline']['status'],'insufficient_evidence')

    def test_shared_retrieval_failure_is_three_errors_with_zero_generation(self):
        index,hits,queries,models=self.setup_trial()
        with patch.object(trial,'validate_catalog'),patch.object(trial,'search',side_effect=ValueError('Broken retrieval')):
            r=trial.run(index,queries,{},[],None,models=models,embedder=SimpleNamespace(calls=[]))
        self.assertEqual(r['run_status'],'error')
        self.assertTrue(all(not g.seen for g in models.values()))
        self.assertTrue(all(o['status']=='error' and o['calls']==[] for c in r['cases'] for o in c['conditions'].values()))

    def test_validation_failure_keeps_completed_result_and_trace(self):
        index,hits,queries,models=self.setup_trial()
        completed=dict(status='insufficient_evidence',claims=[],reason='No hay dato.',grounding={'trace':[{'stage':'selection','hash':'recorded'}]})
        with patch.object(trial,'validate_catalog'),patch.object(trial,'search',return_value=hits),patch.object(trial,'validate_grounded_result',side_effect=ValueError('Invalid final trace')):
            for g in models.values():g.answer=lambda *args,**kwargs:copy.deepcopy(completed)
            r=trial.run(index,queries,{},[],None,models=models,embedder=SimpleNamespace(calls=[]))
        self.assertEqual(r['run_status'],'error')
        self.assertTrue(all(o['status']=='error' and o['full_result']==completed for c in r['cases'] for o in c['conditions'].values()))


if __name__=='__main__':unittest.main()
