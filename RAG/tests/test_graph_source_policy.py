"""The reviewed precedence is scoped data, not a query rule or numeric rewrite."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from rag_pensiones import graph_source_policy as m
from rag_pensiones.graph_catalog import candidate_matches_need
from rag_pensiones.graph_validation import validate_plan,execute_plan,numeric_claim,InvalidPlan
from rag_pensiones.structured_retrieval import DEFAULT_DATABASE,ROOT
from rag_pensiones.graph_planning import prepare_need_candidates,compile_capabilities


class GraphSourcePolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog=json.loads((ROOT/'data/processed/langgraph_9b_assets/catalog.json').read_text(encoding='utf-8'))
        cls.policy=m.load_source_policy(cls.catalog,DEFAULT_DATABASE)
        cls.group=cls.policy['groups'][0]
        cls.cards={c['card_id']:c for c in cls.catalog['cards']}
        cls.preferred=cls.cards[cls.group['preferred_card_id']]
        cls.comparison=cls.cards[cls.group['comparison_card_ids'][0]]
        cls.services=SimpleNamespace(catalog=cls.catalog,database=DEFAULT_DATABASE,source_policy=cls.policy)

    def need(self,**changes):
        result=dict(need_id='N1',kind='data',book='2025',metric='gasto en pensiones',
                    series=None,source_block=None,unit='percent_gdp',periods=['2050'],
                    population='aggregate',scope_terms=[])
        result.update(changes);return result

    def candidates(self,need):
        candidate=copy.deepcopy(self.comparison)
        candidate.update(matched_need_ids=[need['need_id']],need_retrieval=[],retrieval=dict(mode='test'))
        return m.expand_policy_candidates(self.catalog,[candidate],[need],self.policy)

    def plan(self,cid,period='2050',operation='lookup'):
        return dict(status='ready',selections=[dict(need_id='N1',card_ids=[cid],periods=[period],operation=operation)])

    def test_loader_authenticates_ten_records_and_primary_quotes(self):
        self.assertEqual(len(self.group['periods']),5)
        self.assertEqual(self.policy['primary_review']['same_definition_verified'],True)
        self.assertEqual(m.verify_policy_asset(self.policy)['policy_id'],self.policy['policy_id'])

    def test_summary_is_main_and_chart_is_only_a_literal_disclosure(self):
        need=self.need(); candidates=self.candidates(need)
        self.assertEqual({c['card_id']for c in candidates},{self.preferred['card_id'],*self.group['comparison_card_ids']})
        selected=self.plan(self.preferred['card_id'])
        validate_plan(selected,dict(needs=[need]),candidates)
        m.enforce_source_policy(selected,dict(needs=[need]),candidates,self.policy,self.catalog)
        raw=execute_plan(selected,dict(needs=[need]),candidates,self.services)
        facts=m.add_policy_support(raw,dict(needs=[need]),candidates,self.services)
        self.assertEqual(len(facts),3)
        self.assertEqual(facts[0]['value'],raw[0]['value'])
        self.assertEqual(facts[0]['value'],'16.059173455264322000000000')
        self.assertEqual(facts[1]['value'],'16.054187216823770000000000')
        self.assertTrue(facts[1]['supporting_only'])
        claim=numeric_claim(facts[0]);self.assertEqual(len(claim['evidence']),3)
        self.assertNotIn('supporting_only',claim['evidence'][0])
        self.assertTrue(claim['evidence'][1]['supporting_only'])
        self.assertIn('criterio del usuario',claim['text'])
        self.assertIn('Discrepancia no explicada ni conciliada',claim['text'])
        for fact in facts:self.assertIn(fact['value'],claim['text'])
        with self.assertRaises(InvalidPlan):numeric_claim(facts[1])
        with self.assertRaisesRegex(InvalidPlan,'summary as main'):
            m.enforce_source_policy(self.plan(self.comparison['card_id']),dict(needs=[need]),candidates,self.policy,self.catalog)

    def test_summary_priority_handles_descriptive_metric_paraphrases(self):
        for metric in ['gasto en pensiones como proporción del producto interior bruto',
                       'peso futuro de las pensiones sobre el PIB']:
            need=self.need(metric=metric)
            with self.subTest(metric=metric):
                candidates=self.candidates(need)
                self.assertIn(self.preferred['card_id'],{c['card_id']for c in candidates})
                with self.assertRaisesRegex(InvalidPlan,'summary as main'):
                    m.enforce_source_policy(self.plan(self.comparison['card_id']),
                        dict(needs=[need]),candidates,self.policy,self.catalog)

    def test_policy_requires_bound_source_member_not_metric_vocabulary(self):
        need=self.need(metric='Free semantic description')
        self.assertEqual(m._entries_for_need(self.policy,need,self.catalog),[])
        self.assertEqual(m._entries_for_need(self.policy,need,self.catalog,candidate_card_ids=[]),[])
        bound=m._entries_for_need(self.policy,need,self.catalog,candidate_card_ids=[self.comparison['card_id']])
        self.assertEqual({g['group_id']for g,e in bound},{self.group['group_id']})
        self.assertEqual(prepare_need_candidates(need,[],self.services),[])
        other=next(c for c in self.catalog['cards']if c['metric']=='Sanidad'
            and c['book']=='2025'and c['unit']=='percent_gdp'and'2050'in c['periods'])
        self.assertEqual(m._entries_for_need(self.policy,need,self.catalog,candidate_card_ids=[other['card_id']]),[])
        other=copy.deepcopy(other);other['matched_need_ids']=['N1']
        expanded=m.expand_policy_candidates(self.catalog,[other],[need],self.policy)
        self.assertEqual([c['card_id']for c in expanded],[other['card_id']])

    def test_reviewed_re44_is_support_but_explicit_object_and_older_series_are_preserved(self):
        re=next(self.cards[cid]for cid in self.group['comparison_card_ids']if '4.4'in self.cards[cid]['block'])
        need=self.need(quantity_request='level')
        candidate=copy.deepcopy(re);candidate.update(matched_need_ids=['N1'],need_retrieval=[],retrieval=dict(mode='test'))
        candidates=m.expand_policy_candidates(self.catalog,[candidate],[need],self.policy)
        rows=prepare_need_candidates(need,candidates,self.services)
        caps=compile_capabilities(need,rows,self.services)
        self.assertEqual([c['selection']['card_ids']for c in caps],[[self.preferred['card_id']]])
        selected=self.plan(self.preferred['card_id'])
        raw=execute_plan(selected,dict(needs=[need]),candidates,self.services)
        facts=m.add_policy_support(raw,dict(needs=[need]),candidates,self.services)
        supports=[f for f in facts if f.get('supporting_only')]
        self.assertEqual({f['source_cell']for f in supports},{'AM8','AI12'})
        self.assertTrue(all(f['value']=='16.054187216823770000000000'for f in supports))
        main=numeric_claim(facts[0]);self.assertEqual(len(main['evidence']),3)
        self.assertIn('Discrepancia no explicada ni conciliada',main['text'])
        explicit=self.need(source_block=re['block'],series=re['series'],quantity_request='level')
        selected=self.plan(re['card_id']);validate_plan(selected,dict(needs=[explicit]),[re])
        m.enforce_source_policy(selected,dict(needs=[explicit]),[re],self.policy,self.catalog)
        direct=m.add_policy_support(execute_plan(selected,dict(needs=[explicit]),[re],self.services),
            dict(needs=[explicit]),[re],self.services)
        self.assertEqual(len(direct),1);self.assertEqual(direct[0]['source_cell'],'AI12')
        self.assertEqual(direct[0]['value'],'16.054187216823770000000000')
        self.assertNotIn('source_policy_note',direct[0])
        older=next(c for c in self.catalog['cards']if c['book']=='2025'and c['block']==re['block']and c['series']=='Opinión 2023'and c['unit']=='percent_gdp')
        self.assertNotIn(older['card_id'],self.group['comparison_card_ids'])
        old_need=self.need(series=older['series'],quantity_request='level')
        self.assertEqual([c['card_id']for c in prepare_need_candidates(old_need,[older],self.services)],[older['card_id']])
        old_facts=m.add_policy_support(execute_plan(self.plan(older['card_id']),dict(needs=[old_need]),[older],self.services),
            dict(needs=[old_need]),[older],self.services)
        self.assertEqual(len(old_facts),1);self.assertNotIn('source_policy_note',old_facts[0])

    def test_explicit_chart_or_unknown_source_is_never_redirected(self):
        need=self.need(source_block='grafico45');candidates=self.candidates(need)
        self.assertEqual(len(candidates),1)
        self.assertTrue(candidate_matches_need(self.comparison,need))
        self.assertFalse(candidate_matches_need(self.preferred,need))
        selected=self.plan(self.comparison['card_id'])
        validate_plan(selected,dict(needs=[need]),candidates)
        m.enforce_source_policy(selected,dict(needs=[need]),candidates,self.policy,self.catalog)
        raw=execute_plan(selected,dict(needs=[need]),candidates,self.services)
        facts=m.add_policy_support(raw,dict(needs=[need]),candidates,self.services)
        self.assertEqual(len(facts),1);self.assertNotIn('source_policy_note',facts[0])
        with self.assertRaisesRegex(InvalidPlan,'explicitly requested'):
            validate_plan(self.plan(self.preferred['card_id']),dict(needs=[need]),[self.preferred])
        self.assertFalse(candidate_matches_need(self.preferred,self.need(source_block='unknown999')))

    def test_unavailable_summary_period_is_not_interpolated_and_equal_period_has_no_conflict(self):
        for period in ['2030','2023']:
            with self.subTest(period=period):
                need=self.need(periods=[period]);candidates=self.candidates(need)
                self.assertEqual(len(candidates),1)
                raw=execute_plan(self.plan(self.comparison['card_id'],period),dict(needs=[need]),candidates,self.services)
                facts=m.add_policy_support(raw,dict(needs=[need]),candidates,self.services)
                self.assertEqual(len(facts),1);self.assertNotIn('source_policy_note',facts[0])

    def test_other_concepts_scenarios_populations_units_and_books_have_no_policy(self):
        for change in [dict(series='Crecimiento superior'),dict(population='individual'),
                       dict(scope_terms=['mujeres']),dict(unit='percentage_points_gdp'),dict(book='2026')]:
            with self.subTest(change=change):
                need=self.need(**change);self.assertEqual(len(self.candidates(need)),1)
        # Incidental pension candidates cannot hijack an unrelated main selection.
        need=self.need(metric='intereses');candidates=self.candidates(need)
        other=next(c for c in self.catalog['cards']if c['book']=='2025'and c['metric']=='Intereses'and c['unit']=='percent_gdp'and'2050'in c['periods'])
        candidates.append(other)
        m.enforce_source_policy(self.plan(other['card_id']),dict(needs=[need]),candidates,self.policy,self.catalog)

    def test_tampered_policy_bytes_or_in_memory_data_are_rejected(self):
        changed=copy.deepcopy(self.policy);changed['groups'][0]['preferred_card_id']='forged'
        with self.assertRaisesRegex(ValueError,'In-memory'):m.verify_policy_asset(changed)
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp')as directory:
            path=Path(directory)/'policy.json';path.write_text(json.dumps(changed),encoding='utf-8')
            with patch.object(m,'POLICY_PATH',path):
                with self.assertRaisesRegex(ValueError,'bytes changed'):m.verify_policy_asset()

    def test_tampered_original_value_or_cell_is_not_accepted_as_a_new_policy(self):
        for field,value in [('value','99'),('source_cell','X999')]:
            changed=copy.deepcopy(self.catalog)
            oid=self.group['periods'][0]['preferred_observation_id'];changed['observations'][oid][field]=value
            with self.subTest(field=field),self.assertRaisesRegex(ValueError,'Policy'):
                m.load_source_policy(changed,DEFAULT_DATABASE)

    def test_registry_priority_is_per_book_and_preserves_a_second_edition(self):
        need=self.need(book=None,quantity_request='level')
        other=next(c for c in self.catalog['cards']if c['book']=='2026'and c['series']=='AIReF 2026'
                   and c['unit']=='percent_gdp'and c['metric']=='Pensiones'and'2050'in c['periods'])
        rows=prepare_need_candidates(need,[self.comparison,other],self.services)
        self.assertEqual({c['book']for c in rows},{'2025','2026'})
        self.assertIn(self.preferred['card_id'],{c['card_id']for c in rows})
        self.assertNotIn(self.comparison['card_id'],{c['card_id']for c in rows})
        caps=compile_capabilities(need,rows,self.services)
        self.assertEqual({self.cards[c['source_card_id']]['book']for c in caps},{'2025','2026'})

    def test_second_reviewed_concept_uses_its_own_summary_not_the_pension_group(self):
        group=self.policy['groups'][1]
        self.assertEqual(group['book'],'2026')
        need=self.need(book='2026',metric='Intereses',periods=['2030'],quantity_request='level')
        comparison=self.cards[group['comparison_card_ids'][0]]
        rows=prepare_need_candidates(need,[comparison],self.services)
        self.assertEqual([c['card_id']for c in rows],[group['preferred_card_id']])
        caps=compile_capabilities(need,rows,self.services)
        self.assertEqual(caps[0]['selection']['card_ids'],[group['preferred_card_id']])
        unrelated=self.need(book='2026',metric='Sanidad',periods=['2030'],quantity_request='level')
        other=next(c for c in self.catalog['cards']if c['metric']=='Sanidad'
            and c['book']=='2026'and c['unit']=='percent_gdp'and'2030'in c['periods'])
        self.assertEqual([c['card_id']for c in prepare_need_candidates(unrelated,[other],self.services)],
                         [other['card_id']])


if __name__=='__main__':unittest.main()
