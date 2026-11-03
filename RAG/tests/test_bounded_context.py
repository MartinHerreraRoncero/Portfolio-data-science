"""Decoder choices and post-validation enforce the same finite evidence unions."""
import json
import unittest

from experiments.bounded_context import BoundedPreservedGrounder, feasible_package_ids
from experiments.fixed_context import restore_packages
from test_answer_alignment import Generator
import test_fixed_context as fixtures
import test_preserved_evidence as preserved


class BoundedTests(unittest.TestCase):
    def test_choices_include_complementary_evidence_but_exclude_over_budget_unions(self):
        data = fixtures.records()
        data.append(dict(id='P3', evidence=[fixtures.citation('Un cuarto dato literal.', 'S2')]))
        choices = feasible_package_ids(data)
        self.assertIn(['P1', 'P2'], choices)
        self.assertNotIn(['P1', 'P2', 'P3'], choices)
        unions = [restore_packages(fixtures.selected(ids), data)['evidence'] for ids in choices]
        fingerprints = [frozenset((e['source_id'], e['quote']) for e in u) for u in unions]
        self.assertEqual(len(fingerprints), len(set(fingerprints)))
        self.assertTrue(all(len(u) <= 3 for u in unions))

    def checker(self, payload):
        units, pages, _ = preserved.fixture()
        gen = Generator(payload)
        g = BoundedPreservedGrounder(units, pages, generator=gen)
        g.catalog = dict(records=fixtures.records(), omitted=[]); g.selections=[]
        return g, gen

    def test_schema_exposes_only_the_same_feasible_choices_used_by_validator(self):
        g, gen = self.checker(fixtures.selected(['P1', 'P2']))
        out, _ = g._chat_step('selection', [dict(role='user', content='{}')], {}, 'a'*64)
        ids_schema = gen.calls[0]['schema']['anyOf'][0]['properties']['ids']
        self.assertEqual(ids_schema['enum'], feasible_package_ids(g.catalog['records']))
        self.assertEqual(len(out['evidence']), 3)
        self.assertEqual(len(gen.calls), 1)

    def test_transport_ignoring_enum_is_error_and_keeps_raw_selection(self):
        g, gen = self.checker(fixtures.selected(['P2', 'P1']))
        with self.assertRaisesRegex(ValueError, 'feasible schema'):
            g._chat_step('repair_selection', [dict(role='user', content='{}')], {}, 'a'*64)
        self.assertEqual(g.selections[0]['payload']['ids'], ['P2', 'P1'])
        self.assertIn('raw_response', gen.calls[0])
