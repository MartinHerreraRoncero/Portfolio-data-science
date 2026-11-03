import copy
from decimal import Decimal
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from rag_pensiones import graph_validation as m
from rag_pensiones.graph_catalog import build_catalog
from rag_pensiones.graph_pipeline import format_graph_answer,StrictJSONSerializer
from rag_pensiones.structured_retrieval import DEFAULT_DATABASE


class GraphValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog=build_catalog()

    def test_official_pdf_keeps_authority_real_page_and_locator(self):
        oid='official:tesoro:strategy2022:net-issuance:2021'
        card=next(c for c in self.catalog['cards']if oid in c['observation_ids'])
        plan={'selections':[{'need_id':'N','card_ids':[card['card_id']],'periods':['2021'],'operation':'lookup'}]}
        facts=m.execute_plan(plan,{},[card],SimpleNamespace(catalog=self.catalog,database=DEFAULT_DATABASE))
        self.assertEqual(facts[0]['source_pdf_page'],3)
        self.assertIsNone(facts[0]['book'])
        claim=m.numeric_claim(facts[0]);evidence=claim['evidence'][0]
        self.assertEqual(evidence['source_kind'],'structured_official_pdf')
        self.assertEqual(evidence['page_number'],3)
        self.assertEqual(evidence['source_locator'],'EMISIONES NETAS /2021')
        self.assertEqual(evidence['source_variant'],'authoritative_external_correction_not_airef_original')
        self.assertNotIn('quote',evidence)
        self.assertIsNone(evidence['source_sheet']);self.assertIsNone(evidence['source_cell'])
        output=format_graph_answer({'publication':{'status':'answered','claims':[claim],'missing_parts':[]}})
        self.assertIn('pág. 3',output);self.assertNotIn('None!None',output)
        self.assertNotIn('AIReF',claim['text'])

    def test_difference_has_percentage_point_unit_and_preserves_operands(self):
        original=copy.deepcopy(self.catalog['observations']['airef:2025:454'])
        a=dict(original,observation_id='synthetic:a',period='2040',value='10',calculation_safe=True,numeric_reconciliation_failed=False)
        b=dict(a,observation_id='synthetic:b',period='2050',value='12')
        card=dict(next(c for c in self.catalog['cards']if original['observation_id']in c['observation_ids']),
                  card_id='synthetic:card',observation_ids=['synthetic:a','synthetic:b'])
        service=SimpleNamespace(catalog={'observations':{'synthetic:a':a,'synthetic:b':b}},database=DEFAULT_DATABASE)
        def api(db,*,observation_id):
            r=service.catalog['observations'][observation_id]
            return dict(r,metric_literal=r['metric'],series_literal=r['series'])
        plan={'selections':[{'need_id':'N','card_ids':[card['card_id']],'periods':['2040','2050'],'operation':'difference'}]}
        with patch.object(m.policy,'query_observation',side_effect=api):
            fact=m.execute_plan(plan,{},[card],service)[0]
            self.assertEqual(Decimal(fact['value']),Decimal(2))
            self.assertEqual(fact['unit'],'percentage_points_gdp')
            self.assertTrue(fact['derived'])
            self.assertTrue(all(f['unit']=='percent_gdp'for f in fact['operands']))
            a['calculation_safe']=None
            with self.assertRaisesRegex(m.InvalidPlan,'calculation-safe'):
                m.execute_plan(plan,{},[card],service)

    def test_checkpoint_json_rejects_pickle_nan_unknown_types_and_roundtrips(self):
        serde=StrictJSONSerializer()
        value={'x':[1,True,None],'tuple':('a',2)}
        self.assertEqual(serde.loads_typed(serde.dumps_typed(value)),value)
        for bad in [('pickle',b'evil'),('json',b'{"t":"arbitrary_import","v":"os.system"}'),('json',b'{"t":"scalar","v":NaN}')]:
            with self.assertRaises(ValueError):serde.loads_typed(bad)
        with self.assertRaises(ValueError):serde.dumps_typed({'x':float('nan')})

    def test_published_interval_accepts_its_endpoints_without_substituting_horizon(self):
        card=dict(card_id='interval',book='2026',series='Total',metric='Gasto',unit='percentage_points_gdp',
                  scope='aggregate',title='Cambio publicado',quantity_role='published_change',
                  periods=['2050-2070'],period_kind='interval')
        need=dict(need_id='N',kind='data',book='2026',series='Total',periods=['2050','2070'],
                  unit='percentage_points_gdp',scope_terms=[],population='aggregate')
        plan=dict(status='ready',selections=[dict(need_id='N',card_ids=['interval'],periods=['2050-2070'],operation='published_change')])
        m.validate_plan(plan,{'needs':[need]},[card])
        need['periods']=['2023','2070']
        with self.assertRaisesRegex(m.InvalidPlan,'horizon'):m.validate_plan(plan,{'needs':[need]},[card])

    def test_text_requires_literal_canonical_quote_and_requested_report(self):
        source=dict(source_id='canonical',text='El informe explica la evolución del gasto publicado.',book='2026',
                    doc_id='pdfsha',page_number=20,source_file='report.pdf',source_path='report.pdf',source_sha256='pdfsha')
        need=dict(need_id='N',kind='text',book='2026')
        claim=dict(need_id='N',text='El informe explica la evolución.',evidence=[dict(source_id='canonical',quote=source['text'])])
        actual=m.validate_text_claims([copy.deepcopy(claim)],[need],[source])
        self.assertEqual(actual[0]['evidence'][0]['page_number'],20)
        self.assertEqual(actual[0]['evidence'][0]['source_kind'],'pdf_text')
        bad=copy.deepcopy(claim);bad['evidence'][0]['quote']='Un motivo inventado que no aparece en el documento.'
        with self.assertRaisesRegex(m.InvalidPlan,'canonical'):m.validate_text_claims([bad],[need],[source])
        need['book']='2025'
        with self.assertRaisesRegex(m.InvalidPlan,'different report'):m.validate_text_claims([claim],[need],[source])


if __name__=='__main__':unittest.main()
