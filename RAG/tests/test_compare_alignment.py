"""Frozen comparison rejects tampering and preserves abstentions without model calls."""
import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from experiments.compare_alignment import run_comparison
from experiments.answer_alignment import AlignmentChecker
from test_answer_alignment import Generator, facets


class FrozenComparisonTests(unittest.TestCase):
    def fixture(self):
        source=json.loads(Path('evaluation/preserved_results.json').read_text(encoding='utf-8'))
        source['cases']=[next(c for c in source['cases'] if c['id']=='n19')]
        digest=source['metadata']['digests']['baseline']
        gen=Generator();gen.get_digest=lambda:digest
        return source,gen

    def test_frozen_baseline_abstains_and_candidate_is_filtered_without_regeneration(self):
        source,gen=self.fixture();before=copy.deepcopy(source)
        p=facets();p['scope']['outcome']='mismatch';gen.payload=p
        r=run_comparison(source,None,checker=AlignmentChecker(gen))
        self.assertEqual(r['run_status'],'completed')
        self.assertEqual(gen.count,1)
        self.assertEqual(r['cases'][0]['conditions']['baseline']['audit']['decision'],'abstained')
        self.assertEqual(r['cases'][0]['conditions']['preserved']['output_view']['status'],'insufficient_evidence')
        self.assertEqual(source,before)

    def test_tampered_publication_is_an_error_without_judging_it(self):
        source,gen=self.fixture()
        source['cases'][0]['conditions']['preserved']['publication']['claims'][0]['evidence'][0]['quote']='Una cita inventada.'
        r=run_comparison(source,None,checker=AlignmentChecker(gen))
        row=r['cases'][0]['conditions']['preserved']
        self.assertEqual(row['status'],'error');self.assertEqual(gen.count,0)
        self.assertEqual(row['input_view'],source['cases'][0]['conditions']['preserved']['publication'])

    def test_mismatched_source_digest_stops_before_any_judge_calls(self):
        source,gen=self.fixture();gen.get_digest=lambda:'b'*64
        with self.assertRaisesRegex(ValueError,'exact generator digest'):
            run_comparison(source,None,checker=AlignmentChecker(gen))
        self.assertEqual(gen.count,0)


if __name__=='__main__':unittest.main()
