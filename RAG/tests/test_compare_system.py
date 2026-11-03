"""Paired comparisons preserve source outputs, isolate failures and never regenerate filtered answers."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from experiments import compare_system as trial
from experiments.direct_alignment import DirectChecker
from test_answer_alignment import Generator
import test_compare_preserved as preserved_tests


class SystemTests(unittest.TestCase):
    def fixture(self, case_id='n19'):
        source = json.loads(Path('evaluation/preserved_results.json').read_text(encoding='utf-8'))
        source['cases'] = [next(c for c in source['cases'] if c['id'] == case_id)]
        gen = Generator(dict(outcome='reject', reason='No responde al territorio solicitado.'))
        gen.get_digest = lambda: source['metadata']['digests']['baseline']
        return source, gen

    def test_frozen_copies_two_arms_exactly_and_only_filters_preserved(self):
        source, gen = self.fixture(); before = copy.deepcopy(source)
        result = trial.run_frozen(source, None, checker=DirectChecker(gen))
        conditions = result['cases'][0]['conditions']
        for key in ('baseline', 'preserved'):
            self.assertEqual(conditions[key], source['cases'][0]['conditions'][key])
        self.assertEqual(gen.count, 1)
        self.assertEqual(conditions['filtered']['status'], 'insufficient_evidence')
        self.assertEqual(source, before)

    def test_abstention_skips_model_and_preserves_reason(self):
        source, gen = self.fixture('n17')
        result = trial.run_frozen(source, None, checker=DirectChecker(gen))
        arms = result['cases'][0]['conditions']
        self.assertEqual(gen.count, 0)
        self.assertEqual(arms['filtered']['publication'], arms['preserved']['publication'])

    def test_tampered_citation_retained_as_error_without_inference(self):
        source, gen = self.fixture()
        original = source['cases'][0]['conditions']['preserved']
        original['publication']['claims'][0]['evidence'][0]['quote'] = 'Una cita inventada.'
        result = trial.run_frozen(source, None, checker=DirectChecker(gen))
        arms = result['cases'][0]['conditions']
        self.assertEqual(result['run_status'], 'error')
        self.assertEqual(arms['preserved']['publication'], original['publication'])
        self.assertEqual(arms['baseline']['status'], 'insufficient_evidence')
        self.assertEqual(arms['filtered']['status'], 'error')
        self.assertEqual(gen.count, 0)

    def test_digest_mismatch_aborts_before_judging(self):
        source, gen = self.fixture(); gen.get_digest = lambda: 'b'*64
        with self.assertRaisesRegex(ValueError, 'exact frozen generator digest'):
            trial.run_frozen(source, None, checker=DirectChecker(gen))
        self.assertEqual(gen.count, 0)

    def test_filter_error_keeps_inputs_and_raw_without_erasing_other_arms(self):
        source, gen = self.fixture(); gen.payload = dict(outcome='invalid', reason='Bad')
        result = trial.run_frozen(source, None, checker=DirectChecker(gen))
        arms = result['cases'][0]['conditions']
        self.assertEqual(arms['filtered']['status'], 'error')
        self.assertEqual(arms['filtered']['input_view'], arms['preserved']['publication'])
        self.assertIn('raw_response', arms['filtered']['calls'][0])
        self.assertEqual(arms['preserved']['status'], 'answered')

    def setup_new(self):
        index, hits, queries, _ = preserved_tests.RunnerTests().setup_trial()
        models = {k: preserved_tests.Grounder() for k in ('baseline', 'preserved')}
        gen = Generator(dict(outcome='accept', reason='Corresponde.'))
        return index, hits, queries, models, gen

    def test_new_uses_shared_retrieval_alternates_two_arms_and_skips_abstained_filter(self):
        index, hits, queries, models, gen = self.setup_new()
        with patch.object(trial, 'validate_catalog'), patch.object(trial, 'search', return_value=hits) as search, patch.object(trial, 'validate_grounded_result'):
            result = trial.run_new(index, queries, {}, [], None, models=models, embedder=SimpleNamespace(calls=[]), checker=DirectChecker(gen))
        self.assertEqual(search.call_count, len(queries['cases']))
        self.assertEqual(len({tuple(c['condition_order']) for c in result['cases']}), 2)
        self.assertTrue(all(h == hits for g in models.values() for h in g.seen))
        self.assertEqual(gen.count, 0)
        self.assertEqual(result['run_status'], 'completed')

    def test_new_failure_keeps_completed_result_and_unrelated_arm(self):
        index, hits, queries, models, gen = self.setup_new()
        with patch.object(trial, 'validate_catalog'), patch.object(trial, 'search', return_value=hits), patch.object(trial, 'validate_grounded_result', side_effect=ValueError('Bad final result')):
            result = trial.run_new(index, queries, {}, [], None, models=models, embedder=SimpleNamespace(calls=[]), checker=DirectChecker(gen))
        self.assertTrue(all('full_result' in c['conditions']['preserved'] for c in result['cases']))
        self.assertTrue(all(c['conditions']['filtered']['status'] == 'error' for c in result['cases']))
        self.assertEqual(result['run_status'], 'error')
        self.assertEqual(gen.count, 0)


if __name__ == '__main__':
    unittest.main()
