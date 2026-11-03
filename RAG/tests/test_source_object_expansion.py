"""Synthetic unit tests for two-step source-object then row selection expansion.

No network calls, no real LLMs, no embedding models, no retrieval replay.
"""
import copy
from decimal import Decimal
import unittest
from unittest.mock import MagicMock, patch

from rag_pensiones.graph_catalog import (
    _source_object_key,
    expand_source_object_candidates,
)
from rag_pensiones.graph_pipeline import _plan_one_need, _grounding_batch


def _make_card(card_id, book='book_2025', block='Table_1', source_path='data/doc.xlsx',
               metric='Pensions expenditure', series='Baseline', unit='percent_gdp',
               periods=None, quantity_role='', period_kind='year', scope='',
               observation_ids=None):
    if periods is None:
        periods = ['2025', '2030', '2050']
    if observation_ids is None:
        observation_ids = [f'obs:{card_id}:{p}' for p in periods]
    return dict(
        card_id=card_id,
        book=book,
        block=block,
        source_path=source_path,
        source_sha256='sha_dummy',
        metric=metric,
        series=series,
        unit=unit,
        period_kind=period_kind,
        quantity_role=quantity_role,
        scope=scope,
        title=f'Container for {metric}',
        periods=periods,
        observation_ids=observation_ids,
        description=f'Metric: {metric}\nBlock: {block}',
        period_warnings=[],
    )


def _make_catalog(cards):
    obs = {}
    for c in cards:
        for oid, p in zip(c['observation_ids'], c['periods']):
            obs[oid] = dict(
                observation_id=oid,
                book=c['book'],
                block=c['block'],
                metric=c['metric'],
                series=c['series'],
                period=p,
                period_kind=c['period_kind'],
                unit=c['unit'],
                value='12.5',
                quantity_role=c['quantity_role'],
                scope=c['scope'],
                calculation_safe=True,
                source_path=c['source_path'],
                source_sha256=c['source_sha256'],
            )
    return dict(cards=cards, observations=obs)


class SourceObjectExpansionTests(unittest.TestCase):

    def test_existing_rows_gain_other_eligible_needs(self):
        a=_make_card('a',periods=['2025','2030']);b=_make_card('b',periods=['2030'])
        a['matched_need_ids']=['n1'];b['matched_need_ids']=['n2']
        needs=[dict(need_id=n,kind='data',unit='percent_gdp',periods=[p],quantity_request='level')
               for n,p in [('n1','2025'),('n2','2030')]]
        rows=expand_source_object_candidates(_make_catalog([a,b]),[a,b],needs)
        self.assertEqual(rows[0]['matched_need_ids'],['n1','n2'])
        # Discovery defers year availability; capability compilation checks it.
        self.assertEqual(rows[1]['matched_need_ids'],['n2','n1'])

    def test_grounding_chat_preserves_prior_budget_accounting(self):
        from rag_pensiones.graph_pipeline import _chat
        services=MagicMock();services.client.calls=[];services.client.chat.return_value={}
        payload=dict(candidates=[dict(card_id='a',book='x'),dict(card_id='b',book='x')],
                     omitted_candidate_count=4,omitted_source_objects=2,omitted_object_count=2)
        _chat(services,'grounding',payload,dict(type='object',properties={}))
        import json
        wire=json.loads(services.client.chat.call_args.args[0][1]['content'])
        self.assertEqual(wire['omitted_candidate_count'],4)
        self.assertEqual(wire['omitted_object_count'],2)

    def test_empty_candidates_and_catalog(self):
        self.assertEqual(expand_source_object_candidates({}, [], []), [])
        self.assertEqual(expand_source_object_candidates({'cards': []}, [], [{'need_id': 'n1'}]), [])
        card = _make_card('c1')
        self.assertEqual(expand_source_object_candidates({}, [card], []), [card])

    def test_missing_block_or_path_never_groups_unrelated_cards(self):
        # Missing block or missing source_path must isolate cards into their own group
        c1 = _make_card('c1', block=None, source_path='pathA.xlsx')
        c2 = _make_card('c2', block=None, source_path='pathA.xlsx')
        c3 = _make_card('c3', block='', source_path='pathA.xlsx')
        c4 = _make_card('c4', block='Table_1', source_path=None)
        c5 = _make_card('c5', block='Table_1', source_path=None)

        self.assertEqual(_source_object_key(c1), ('card', 'c1'))
        self.assertEqual(_source_object_key(c2), ('card', 'c2'))
        self.assertEqual(_source_object_key(c3), ('card', 'c3'))
        self.assertEqual(_source_object_key(c4), ('card', 'c4'))
        self.assertEqual(_source_object_key(c5), ('card', 'c5'))

        # They must never match as the same source object
        self.assertNotEqual(_source_object_key(c1), _source_object_key(c2))
        self.assertNotEqual(_source_object_key(c4), _source_object_key(c5))

        # In expansion, cards with missing identity never pull sibling rows
        catalog_card = _make_card('c_sib', block=None, source_path='pathA.xlsx')
        catalog = _make_catalog([c1, c2, catalog_card])
        need = dict(need_id='n1', kind='data', unit='percent_gdp', periods=['2025'])
        res = expand_source_object_candidates(catalog, [c1], [need])
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0]['card_id'], 'c1')

    def test_same_block_different_files_or_books_are_isolated(self):
        # Same block, but different files or books
        c1 = _make_card('c1', book='2024', source_path='fileA.xlsx', block='Table_1')
        c2 = _make_card('c2', book='2024', source_path='fileB.xlsx', block='Table_1')
        c3 = _make_card('c3', book='2025', source_path='fileA.xlsx', block='Table_1')

        self.assertNotEqual(_source_object_key(c1), _source_object_key(c2))
        self.assertNotEqual(_source_object_key(c1), _source_object_key(c3))
        self.assertNotEqual(_source_object_key(c2), _source_object_key(c3))

        # Catalog has a sibling strictly for c1's object (book=2024, fileA.xlsx, Table_1)
        sib1 = _make_card('sib1', book='2024', source_path='fileA.xlsx', block='Table_1', series='Alternative')
        # Sibling for c2's object
        sib2 = _make_card('sib2', book='2024', source_path='fileB.xlsx', block='Table_1', series='Alternative')

        catalog = _make_catalog([c1, c2, c3, sib1, sib2])
        need = dict(need_id='n1', kind='data', unit='percent_gdp', periods=['2025'])

        # Candidate only has c1
        res = expand_source_object_candidates(catalog, [c1], [need])
        res_ids = [r['card_id'] for r in res]
        self.assertIn('c1', res_ids)
        self.assertIn('sib1', res_ids)
        self.assertNotIn('sib2', res_ids)
        self.assertNotIn('c2', res_ids)
        self.assertNotIn('c3', res_ids)

    def test_duplicate_representatives_and_id_deduplication(self):
        c1 = _make_card('c1', book='2025', source_path='file.xlsx', block='T1', metric='M1')
        c2 = _make_card('c2', book='2025', source_path='file.xlsx', block='T2', metric='M2')
        # Duplicate of c1 with different need
        c1_dup = copy.deepcopy(c1)
        c1['matched_need_ids'] = ['n1']
        c1_dup['matched_need_ids'] = ['n2']

        # Duplicate rep of c1's object with different card_id
        c1_sibling_in_candidates = _make_card('c1_other', book='2025', source_path='file.xlsx', block='T1', metric='M1_alt')
        c1_sibling_in_candidates['matched_need_ids'] = ['n1']

        # Sibling in catalog
        sib_c1 = _make_card('sib_c1', book='2025', source_path='file.xlsx', block='T1', metric='M1_third')
        sib_c2 = _make_card('sib_c2', book='2025', source_path='file.xlsx', block='T2', metric='M2_alt')

        catalog = _make_catalog([c1, c2, c1_sibling_in_candidates, sib_c1, sib_c2])
        needs = [
            dict(need_id='n1', kind='data', unit='percent_gdp', periods=['2025']),
            dict(need_id='n2', kind='data', unit='percent_gdp', periods=['2025']),
        ]

        # Candidates has [c1, c2, c1_dup, c1_sibling_in_candidates]
        candidates = [c1, c2, c1_dup, c1_sibling_in_candidates]
        res = expand_source_object_candidates(catalog, candidates, needs)

        # 1. c1 should only appear once (deduplicated ID)
        c1_count = sum(1 for r in res if r['card_id'] == 'c1')
        self.assertEqual(c1_count, 1)

        # 2. c1's matched_need_ids should have merged ['n1', 'n2']
        merged_c1 = next(r for r in res if r['card_id'] == 'c1')
        self.assertEqual(set(merged_c1['matched_need_ids']), {'n1', 'n2'})

        # 3. Rows of the same object should be visible together:
        # Object T1 rows (c1, c1_other, sib_c1) should be contiguous!
        card_blocks = [r['block'] for r in res]
        self.assertEqual(card_blocks, ['T1', 'T1', 'T1', 'T2', 'T2'])
        self.assertEqual([r['card_id'] for r in res], ['c1', 'c1_other', 'sib_c1', 'c2', 'sib_c2'])

    def test_multiple_needs_filters_and_no_cross_need_leakage(self):
        # Need 1: percent_gdp
        # Need 2: euros
        need1 = dict(need_id='n1', kind='data', unit='percent_gdp', periods=['2025'], quantity_request='level')
        need2 = dict(need_id='n2', kind='data', unit='euros', periods=['2025'], quantity_request='level')
        needs = [need1, need2]

        card_gdp = _make_card('c_gdp', block='T1', unit='percent_gdp')
        sib_gdp = _make_card('sib_gdp', block='T1', unit='percent_gdp', series='AltGDP')
        sib_eur = _make_card('sib_eur', block='T1', unit='euros', series='AltEUR')
        sib_persons = _make_card('sib_persons', block='T1', unit='persons', series='AltPersons')

        catalog = _make_catalog([card_gdp, sib_gdp, sib_eur, sib_persons])

        # Case A: Candidate only matched to n1 (percent_gdp)
        card_gdp['matched_need_ids'] = ['n1']
        resA = expand_source_object_candidates(catalog, [card_gdp], needs)
        resA_ids = {r['card_id']: r for r in resA}

        self.assertIn('c_gdp', resA_ids)
        self.assertIn('sib_gdp', resA_ids)
        self.assertEqual(resA_ids['sib_gdp']['matched_need_ids'], ['n1'])
        # sib_eur must NOT be exposed because Object T1 was not selected for n2 and sib_eur does not match n1
        self.assertNotIn('sib_eur', resA_ids)
        self.assertNotIn('sib_persons', resA_ids)

        # Case B: Candidate has two cards in T1: one for n1, one for n2
        card_eur = _make_card('c_eur', block='T1', unit='euros')
        card_eur['matched_need_ids'] = ['n2']
        catalogB = _make_catalog([card_gdp, card_eur, sib_gdp, sib_eur, sib_persons])

        resB = expand_source_object_candidates(catalogB, [card_gdp, card_eur], needs)
        resB_ids = {r['card_id']: r for r in resB}

        self.assertIn('sib_gdp', resB_ids)
        self.assertIn('sib_eur', resB_ids)
        self.assertNotIn('sib_persons', resB_ids)

        # No cross-need leakage:
        # sib_gdp only matches n1 (not n2)
        self.assertEqual(resB_ids['sib_gdp']['matched_need_ids'], ['n1'])
        # sib_eur only matches n2 (not n1)
        self.assertEqual(resB_ids['sib_eur']['matched_need_ids'], ['n2'])

    def test_input_immutability(self):
        c1 = _make_card('c1', block='T1')
        sib = _make_card('sib', block='T1')
        catalog = _make_catalog([c1, sib])
        need = dict(need_id='n1', kind='data', unit='percent_gdp', periods=['2025'])

        candidates = [c1]
        candidates_before = copy.deepcopy(candidates)
        catalog_before = copy.deepcopy(catalog)

        res = expand_source_object_candidates(catalog, candidates, [need])
        self.assertEqual(candidates, candidates_before)
        self.assertEqual(catalog, catalog_before)
        self.assertEqual(len(res), 2)

    def test_no_invented_retrieval_scores_and_parent_provenance(self):
        c1 = _make_card('c1', block='T1')
        c1['retrieval'] = dict(mode='hybrid_rrf', bm25=2.45, cosine=0.88, rrf=0.033, rank=1)
        c1['need_retrieval'] = [dict(need_id='n1', rank=1, scores=copy.deepcopy(c1['retrieval']))]
        c1['matched_need_ids'] = ['n1']

        sib = _make_card('sib', block='T1')
        catalog = _make_catalog([c1, sib])
        need = dict(need_id='n1', kind='data', unit='percent_gdp', periods=['2025'])

        res = expand_source_object_candidates(catalog, [c1], [need])
        self.assertEqual(len(res), 2)

        # Original candidate retains original scores
        self.assertEqual(res[0]['card_id'], 'c1')
        self.assertEqual(res[0]['retrieval']['bm25'], 2.45)
        self.assertEqual(res[0]['retrieval']['cosine'], 0.88)

        # Sibling card has parent provenance and NO invented model scores
        s = res[1]
        self.assertEqual(s['card_id'], 'sib')
        self.assertEqual(s['retrieval']['mode'], 'source_object_expansion')
        self.assertEqual(s['retrieval']['parent_card_id'], 'c1')
        self.assertNotIn('bm25', s['retrieval'])
        self.assertNotIn('cosine', s['retrieval'])
        self.assertNotIn('reranker', s['retrieval'])
        # No raw observation values in sibling card
        self.assertNotIn('value', s)
        self.assertNotIn('raw_value', s)

    def test_complete_object_budget_removal_and_accounting_in_planning(self):
        # Test pruning in _plan_one_need
        # Create 2 source objects, each with 2 rows:
        # Object 1 (book 2024, T1): c1, sib1
        # Object 2 (book 2024, T2): c2, sib2
        c1 = _make_card('c1', book='2024', block='T1')
        sib1 = _make_card('sib1', book='2024', block='T1', series='Alt1')
        c2 = _make_card('c2', book='2024', block='T2')
        sib2 = _make_card('sib2', book='2024', block='T2', series='Alt2')

        rows = [c1, sib1, c2, sib2]
        catalog = _make_catalog(rows)

        services = MagicMock()
        services.catalog = catalog
        services.semantic_context = None

        state = dict(query='Expenditure in 2025?', effective_context=None)
        need = dict(need_id='n1', kind='data', unit='percent_gdp', periods=['2025'],
                    metric='Pensions expenditure', series=None, book='2024')

        # Mock compile_capabilities to return dummy capabilities
        caps = [
            dict(capability_id='cap1', selection=dict(card_ids=['c1'], periods=['2025'], operation='lookup')),
            dict(capability_id='cap2', selection=dict(card_ids=['c2'], periods=['2025'], operation='lookup')),
        ]

        # First call does not fit; second call fits after dropping one source object
        fits_calls = [False, True]
        def fake_payload_fits(stage, payload, schema):
            return fits_calls.pop(0) if fits_calls else True

        decision_resp = dict(action='select', capability_id='cap1')

        with patch('rag_pensiones.graph_pipeline.compile_capabilities', side_effect=lambda need,rows,services:[cap for cap in caps if cap['selection']['card_ids'][0] in {r['card_id']for r in rows}]), \
             patch('rag_pensiones.graph_pipeline._payload_fits', side_effect=fake_payload_fits), \
             patch('rag_pensiones.graph_pipeline._chat', return_value=decision_resp), \
             patch('rag_pensiones.graph_pipeline.capability_options', return_value=[]), \
             patch('rag_pensiones.graph_pipeline.subtotal_scope_allowed', return_value=True):

            services.client.calls = [
                dict(request=dict(messages=[{}, dict(content='{"cap_options":[{"capability_id":"cap1","source_metadata":[{"card_id":"c1"}]}]} ')]))
            ]

            decision, selectable, final_rows, summary = _plan_one_need(services, state, need, rows)

            # Object 2 (both c2 and sib2) was completely removed
            final_ids = [r['card_id'] for r in final_rows]
            self.assertIn('c1', final_ids)
            self.assertIn('sib1', final_ids)
            self.assertNotIn('c2', final_ids)
            self.assertNotIn('sib2', final_ids)

            # Summary records exact omitted counts
            self.assertEqual(summary['omitted_row_count'], 2)
            self.assertEqual(summary['omitted_object_count'], 1)

    def test_protected_overflow_raises_explicit_bounded_error(self):
        # When all remaining source objects are protected and cannot fit, raise ValueError
        c1 = _make_card('c1', book='2024', block='T1')
        rows = [c1]
        catalog = _make_catalog(rows)

        services = MagicMock()
        services.catalog = catalog
        services.semantic_context = None

        state = dict(query='Expenditure?', effective_context=None)
        need = dict(need_id='n1', kind='data', unit='percent_gdp', periods=['2025'],
                    metric='Pensions expenditure', series=None, book='2024')

        # c1 is protected because it is the sole representative for book 2024
        with patch('rag_pensiones.graph_pipeline.compile_capabilities', return_value=[]), \
             patch('rag_pensiones.graph_pipeline._payload_fits', return_value=False), \
             patch('rag_pensiones.graph_pipeline.capability_options', return_value=[]), \
             patch('rag_pensiones.graph_pipeline.subtotal_scope_allowed', return_value=True):

            with self.assertRaisesRegex(ValueError, 'Protected capability sources exceed byte budget'):
                _plan_one_need(services, state, need, rows)

    def test_protected_overflow_in_grounding_raises_explicit_bounded_error(self):
        c1 = _make_card('c1', book='2024', block='T1')
        rows = [c1]
        catalog = _make_catalog(rows)

        services = MagicMock()
        services.catalog = catalog

        state = dict(
            candidates=rows,
            interpretation=dict(needs=[dict(need_id='n1', kind='data', unit='percent_gdp', periods=['2025'], metric='M1', series=None, book=None,population='aggregate',scope_terms=[],source_block=None,quantity_request='level')]),
            query='Expenditure?',
            effective_context=None
        )

        with patch('rag_pensiones.graph_pipeline.grounding_inputs', return_value={'n1': dict(before=dict(series=None,periods=['2025']),exact_series=None, sources=[c1], series_labels=[], years=['2025'])}), \
             patch('rag_pensiones.graph_pipeline._payload_fits', return_value=False):

            with self.assertRaisesRegex(ValueError, 'Protected grounding metadata exceed byte budget'):
                _grounding_batch(services, state)


if __name__ == '__main__':
    unittest.main()
