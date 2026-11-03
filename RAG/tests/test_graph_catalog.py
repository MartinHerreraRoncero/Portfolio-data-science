import copy
from decimal import Decimal
import json
import math
from pathlib import Path
import unittest

from rag_pensiones import graph_catalog as m
from rag_pensiones.graph_types import EMBED_DIMENSION
from experiments import source_recovery_policy as policy


class GraphCatalogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = m.build_catalog()

    def tiny(self):
        c = copy.deepcopy(self.catalog)
        c['cards'] = [next(card for card in c['cards'] if oid in card['observation_ids'])
                      for oid in ['airef:2025:454', 'airef:2026:388', 'airef:2026:4179']]
        ids = {oid for card in c['cards'] for oid in card['observation_ids']}
        c['observations'] = {k: v for k, v in c['observations'].items() if k in ids}
        c['catalog_hash'] = m.canonical_hash(m._content(c))
        return c

    def test_all_primary_198_records_and_public_domain_preserved(self):
        facts = json.loads((Path(__file__).resolve().parents[1] / 'evaluation/structured_rag_2026-11-01/audit/primary_expected_records.json').read_text(encoding='utf-8'))['facts']
        self.assertEqual(len(self.catalog['observations']), 42880)
        for fact in facts:
            row = self.catalog['observations'][fact['observation_id']]
            for field in ['book', 'unit', 'source_sheet', 'source_cell']:
                self.assertEqual(row[field], fact[field])
            self.assertEqual(Decimal(row['value']), Decimal(fact['value_exact']))
            self.assertEqual(row['metric'], fact.get('metric', fact.get('metric_literal')))
            self.assertEqual(row['series'], fact.get('series', fact.get('series_literal')))

    def test_all_real_embedding_inputs_are_bounded_without_changing_cards(self):
        before=m.canonical_hash(m._content(self.catalog));changed=[]
        for card in self.catalog['cards']:
            original=copy.deepcopy(card);text=m.embedding_description(card)
            self.assertLessEqual(len(text.encode()),m.MAX_EMBED_INPUT_BYTES)
            self.assertEqual(text,m.embedding_description(card));self.assertEqual(original,card)
            if text!=card['description']:
                changed.append(card['card_id'])
                self.assertIn(f"{len(card['periods'])} etiquetas",text)
                self.assertIn(card['periods'][0],text);self.assertIn(card['periods'][-1],text)
                for line in card['description'].splitlines():
                    if not line.startswith('Periodos disponibles: '):self.assertIn(line,text)
        self.assertEqual(len(changed),13)
        self.assertEqual(before,self.catalog['catalog_hash'])
        self.assertEqual(before,m.canonical_hash(m._content(self.catalog)))

    def test_overlong_identity_fails_without_truncation(self):
        with self.assertRaisesRegex(ValueError,'identity'):
            m.embedding_description({'description':'Magnitud: '+'x'*6401,'periods':['2050']})

    def test_embedding_representation_hash_cannot_be_relabelled(self):
        c=self.tiny();v=[1.]+[0.]*(EMBED_DIMENSION-1)
        c=m.attach_embeddings(c,[v]*len(c['cards']))
        c['embedding_index']['embedding_text_hash']='forged'
        c['embedding_index']['index_hash']=m.canonical_hash({k:v for k,v in c['embedding_index'].items()if k!='index_hash'})
        with self.assertRaisesRegex(ValueError,'incompatible'):m.retrieve_catalog(c,'pensiones',v)

    def test_cards_partition_series_units_scope_and_period_kind(self):
        m.validate_catalog(self.catalog)
        for card in self.catalog['cards']:
            identities = {(self.catalog['observations'][oid]['book'],
                self.catalog['observations'][oid]['series'], self.catalog['observations'][oid]['unit'],
                self.catalog['observations'][oid]['scope'], self.catalog['observations'][oid]['period_kind'])
                for oid in card['observation_ids']}
            self.assertEqual(len(identities), 1)
        self.assertTrue(any(c['book'] is None and c['source_path'].endswith('.pdf') for c in self.catalog['cards']))

    def test_ids_and_content_are_deterministic(self):
        self.assertEqual(self.catalog['catalog_hash'], m.build_catalog()['catalog_hash'])
        ids = [c['card_id'] for c in self.catalog['cards']]
        self.assertEqual(len(ids), len(set(ids)))

    def test_all16_source_policy_warnings_preserved_per_observation(self):
        flagged = [r for r in self.catalog['observations'].values()
                   if r['numeric_reconciliation_failed'] is True]
        self.assertEqual(len(flagged), 16)
        for row in flagged:
            actual = policy.query_observation(m.DEFAULT_DATABASE, observation_id=row['observation_id'])
            for field in ['warning', 'source_priority', 'global_calculation_prohibition', 'calculation_safe']:
                self.assertEqual(row[field], actual[field])
            self.assertTrue(row['warning'])
            self.assertTrue(row['source_priority'])
            self.assertIs(row['calculation_safe'], False)
            self.assertIs(row['global_calculation_prohibition'], False)
        for card in self.catalog['cards']:
            self.assertNotIn('calculation_safe', card, 'No blanket card/series prohibition')
            unsafe = {w['observation_id'] for w in card['period_warnings']}
            for oid in card['observation_ids']:
                if oid not in unsafe:
                    self.assertFalse(self.catalog['observations'][oid].get('warning'))
                    self.assertIsNot(self.catalog['observations'][oid].get('calculation_safe'), False)
                if self.catalog['observations'][oid]['numeric_reconciliation_failed'] is True:
                    self.assertIn(oid, unsafe)

    def test_unknown_observation_and_stale_hash_rejected(self):
        c = self.tiny()
        c['cards'][0]['observation_ids'].append('unknown')
        with self.assertRaises(ValueError):
            m.validate_catalog(c)
        c['catalog_hash'] = m.canonical_hash(m._content(c))
        with self.assertRaises(ValueError):
            m.validate_catalog(c)

    def test_source_authentication_rejects_rehashed_subset(self):
        with self.assertRaisesRegex(ValueError, 'certified source'):
            m.verify_catalog_source(self.tiny())

    def test_bm25_discovery_does_not_require_figure_number(self):
        found = m.retrieve_catalog(self.tiny(), 'pensiones AIReF 2026 2050', top_k=3)
        self.assertTrue(any('airef:2026:388' in card['observation_ids'] for card in found))
        self.assertTrue(all(c['retrieval']['mode'] == 'bm25_only' for c in found))
        self.assertTrue(all(c['retrieval']['cosine'] is None for c in found))

    def test_hybrid_returns_semantic_candidate_even_without_term_overlap(self):
        c = self.tiny()
        values = [[0.] * EMBED_DIMENSION for _ in c['cards']]
        for i, value in enumerate(values):
            value[i] = 1.
        enriched = m.attach_embeddings(c, values)
        q = [0.] * EMBED_DIMENSION
        q[2] = 1.
        found = m.retrieve_catalog(enriched, 'retirement expenditure outlook', q, top_k=3)
        self.assertEqual(found[0]['card_id'], c['cards'][2]['card_id'])
        self.assertEqual(found[0]['retrieval']['mode'], 'hybrid_rrf')
        self.assertEqual(found[0]['retrieval']['cosine'], 1.)

    def test_embedding_binding_dimensions_nonfinite_and_partial_index_rejected(self):
        c = self.tiny()
        vectors = [[1.] + [0.] * (EMBED_DIMENSION - 1) for _ in c['cards']]
        good = m.attach_embeddings(c, vectors)
        q = vectors[0]
        for field, value in [('catalog_hash', 'bad'), ('model_digest', 'wrong')]:
            bad = copy.deepcopy(good)
            bad['embedding_index'][field] = value
            with self.assertRaises(ValueError):
                m.retrieve_catalog(bad, 'pensiones', q)
        for invalid in [[float('nan')] * EMBED_DIMENSION, [float('inf')] * EMBED_DIMENSION, [0.], [0.] * EMBED_DIMENSION]:
            with self.assertRaises(ValueError):
                m.retrieve_catalog(good, 'pensiones', invalid)
        with self.assertRaises(ValueError):
            m.retrieve_catalog(c, 'pensiones', q)
        with self.assertRaises(ValueError):
            m.retrieve_catalog(good, 'pensiones')


if __name__ == '__main__':
    unittest.main()
