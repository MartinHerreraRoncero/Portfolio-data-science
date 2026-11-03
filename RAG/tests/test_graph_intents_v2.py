"""Typed intent/unit contracts use local mocks, never local inference."""
import copy
import json
import unittest

from rag_pensiones import graph_pipeline as pipeline
from rag_pensiones.graph_models import GraphStageClient,MAX_CHAT_REQUEST_BYTES
from rag_pensiones.graph_planning import (literal_query_spans,interpretation_schema,
    normalize_intent,compact_intent_draft)
from rag_pensiones.graph_units import unit_family,unit_matches
from rag_pensiones.graph_types import CHAT_MODEL,CHAT_DIGEST,EMBED_MODEL,EMBED_DIGEST,PROFILE


def draft(query):
    return dict(resolved_query=query,needs=[dict(need_id='N1',query_span=query,kind='data',
        metric='indicador',population='aggregate',scope_terms=[],book='2024',series=None,
        source_block=None,unit='unspecified',periods=['2060'],quantity_request='level')])


class GraphIntentsV2Tests(unittest.TestCase):
    def test_source_policy_replaces_only_bound_comparison_members_and_keeps_other_measures(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from rag_pensiones.graph_planning import prepare_need_candidates
        cards=[dict(card_id=cid,book='2026',matched_need_ids=['n1'])for cid in ['comparison','unrelated']]
        cards.append(dict(card_id='other_need',book='2026',matched_need_ids=['n2']))
        preferred=dict(card_id='preferred',book='2026')
        services=SimpleNamespace(source_policy={},catalog=dict(cards=cards+[preferred]),semantic_context=None)
        n=dict(need_id='n1',quantity_request='level')
        group=dict(preferred_card_id='preferred',comparison_card_ids=['comparison'])
        before=copy.deepcopy(cards)
        with patch('rag_pensiones.graph_planning.candidate_matches_need',return_value=True),patch(
                'rag_pensiones.graph_source_policy._entries_for_need',return_value=[(group,{})])as entries:
            rows=prepare_need_candidates(n,cards,services)
        self.assertEqual(entries.call_args.kwargs,dict(candidate_card_ids={'comparison','unrelated'}))
        self.assertEqual({c['card_id']for c in rows},{'preferred','unrelated'})
        self.assertEqual(cards,before)

    def test_families_keep_rates_contributions_and_gdp_dimensions_separate(self):
        examples={'percent_growth_as_published':'percent','percent_change_as_published':'percent',
                  'percent_interest_rate_as_published':'percent',
                  'percentage_points_published_growth_contribution':'percentage_points',
                  'percentage_points_annual_pension_expenditure_growth':'percentage_points',
                  'percent_gdp':'percent_gdp','interannual_percentage_points_gdp':'percentage_points_gdp',
                  'percentage_points_GDP_debt_ratio':'percentage_points_gdp'}
        for raw,expected in examples.items():
            with self.subTest(raw=raw):self.assertEqual(unit_family(raw),expected)
        self.assertFalse(unit_matches('percent_gdp','percent_growth_as_published'))
        self.assertFalse(unit_matches('percent','percent_gdp'))
        self.assertFalse(unit_matches('percentage_points','percentage_points_gdp'))
        self.assertFalse(unit_matches('percentage_points_gdp','percent_gdp'))
        self.assertTrue(unit_matches('percentage_points_gdp','percent_gdp',allow_level_operands=True))
        self.assertFalse(unit_matches('percentage_points_gdp','percent_growth_as_published',allow_level_operands=True))
        self.assertIsNone(unit_family('basis_points_risk_spread_as_published'))
        self.assertIsNone(unit_family('thousand_persons_signed_plot'))

    def test_null_normalization_preserves_the_raw_trace_and_never_changes_periods(self):
        query='Según el informe, indica el valor para 2060.';raw=draft(query)
        raw['needs'][0].update(book=' null ',series='None',source_block='   ')
        original=copy.deepcopy(raw);normalized=normalize_intent(raw,query)
        for key in ['book','series','source_block']:self.assertIsNone(normalized['needs'][0][key])
        self.assertEqual(raw,original)
        self.assertEqual(normalized['needs'][0]['periods'],['2060'])
        self.assertEqual(normalized['resolved_query'],query)

    def test_literal_spans_are_question_bound_and_paraphrases_cannot_pass(self):
        query='¿Qué valor tiene A en 2060? Además, explica su definición.'
        spans=literal_query_spans(query)
        self.assertEqual(spans[0],query)
        self.assertTrue(all(span in query for span in spans))
        schema=interpretation_schema(query,pipeline.INTERPRET_SCHEMA)
        self.assertEqual(schema['properties']['needs']['items']['properties']['query_span']['enum'],spans)
        raw=draft(query);raw['needs'][0]['query_span']='Valor de A para el futuro'
        with self.assertRaisesRegex(ValueError,'literal'):normalize_intent(raw,query)

    def test_legacy_compact_draft_helper_retains_four_needs_without_mutation(self):
        query='En el informe de 2024, indica A en 2060, B en 2070, C en 2080 y D en 2090, conservando sus unidades publicadas.'
        raw=draft(query);raw['needs']=[dict(raw['needs'][0],need_id=f'N{i}',metric=f'indicador {i}')for i in range(4)]
        compact=compact_intent_draft(raw,query)
        self.assertEqual(len(compact['needs']),4)
        self.assertTrue(all(n['span_index']==0 and'query_span'not in n for n in compact['needs']))
        self.assertLess(len(json.dumps(compact)),len(json.dumps(raw)))
        self.assertEqual(PROFILE,'langgraph-9b-v8')


if __name__=='__main__':unittest.main()
