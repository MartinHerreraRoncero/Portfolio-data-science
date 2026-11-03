"""Offline execution checks with synthetic facts; no inference or source assets."""
import copy
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from rag_pensiones import graph_pipeline as p
from rag_pensiones.graph_planning import CHECKER_APPROVALS,checker_schema
from rag_pensiones.graph_validation import InvalidPlan
from rag_pensiones.graph_types import stage_chat_settings


class ResultCheckTests(unittest.TestCase):
    def setUp(self):
        self.need=dict(need_id='n1',kind='data',metric='incorrect interpretation',
                       series='invented wanted series',book='2026',unit='percent_gdp')
        self.selection=dict(need_id='n1',card_ids=['c1'],periods=['2050'],operation='lookup',reason='compiled')
        self.cap=dict(capability_id='cap',selection=self.selection,source_card_id='c1')
        self.fact=dict(need_id='n1',card_id='c1',observation_id='o1',book='2026',
            metric='Gasto',series='Central',period='2050',value='12.3',unit='percent_gdp',
            scope='aggregate',source_sheet='Table',source_cell='B4',derived=False)
        self.state=dict(query='¿Qué gasto hay en 2050 y por qué cambia?',effective_context={'prior':'original'},
            interpretation=dict(needs=[self.need]),plan=dict(status='ready',selections=[self.selection]),
            checker_scope=['n1'],planning_decisions={'n1':dict(action='select',capability_id='cap')},
            offered_capabilities={'n1':[self.cap]},planning_card_ids=['c1'],candidates=[],
            ambiguity_summaries={},repairs=1)
        self.services=SimpleNamespace(client=SimpleNamespace(calls=[]),source_policy={},catalog={},
                                      authenticate=lambda:None)
        self.graph=p.build_graph(self.services)

    def node(self,name,state):return self.graph.nodes[name].bound.invoke(state)

    def validated(self,facts=None):
        with patch.object(p,'canonical_decision',return_value=self.selection),patch.object(p,'validate_plan'),\
             patch.object(p,'compile_capabilities',return_value=[self.cap]),\
             patch.object(p,'enforce_source_policy'),patch.object(p,'execute_plan',return_value=[self.fact]if facts is None else facts),\
             patch.object(p,'_chat')as chat:
            result=self.node('validate',self.state)
            chat.assert_not_called()
            return result

    def test_compiled_source_contract_retains_actual_values_without_another_llm_judgment(self):
        result=self.validated()
        self.assertTrue(result['plan_valid'])
        self.assertEqual(result['checked_numeric_results'],{'n1':[self.fact]})
        self.assertEqual(len(result['raw_result_checks']),1)
        record=result['raw_result_checks'][0]
        self.assertEqual(record['validation_method'],'compiled_source_contract')
        self.assertEqual(record['guard_status'],'passed')
        self.assertIn('not certified',record['reason'])

    def test_hard_failure_never_queries_or_asks_the_checker(self):
        with patch.object(p,'canonical_decision',return_value=self.selection),\
             patch.object(p,'compile_capabilities',return_value=[self.cap]),\
             patch.object(p,'validate_plan',side_effect=[InvalidPlan('wrong unit'),None]),\
             patch.object(p,'execute_plan')as sql,patch.object(p,'_chat')as chat:
            result=self.node('validate',self.state)
        sql.assert_not_called();chat.assert_not_called()
        self.assertEqual(result['need_outcomes']['n1'],'source_contract_rejected')
        self.assertEqual(result['checked_numeric_results'],{})

    def test_final_execution_requeries_and_rejects_changed_or_missing_checked_facts(self):
        state=dict(self.state,checked_numeric_results={'n1':[copy.deepcopy(self.fact)]})
        for checked,actual in [({'n1':[self.fact]},dict(self.fact,value='99')),({},self.fact)]:
            state['checked_numeric_results']=checked
            with patch.object(p,'execute_plan',return_value=[actual])as sql,patch.object(p,'add_policy_support')as support:
                result=self.node('execute',state)
            sql.assert_called_once();support.assert_not_called()
            self.assertIn('ValueError: Authenticated execution differs',result['errors'][0])
        with patch.object(p,'execute_plan',return_value=[self.fact])as sql,\
             patch.object(p,'add_policy_support',side_effect=lambda facts,*_:facts):
            state['checked_numeric_results']={'n1':[self.fact]}
            self.assertEqual(self.node('execute',state)['numeric_facts'],[self.fact])
            sql.assert_called_once()

    def test_derived_results_show_actual_formula_and_operands(self):
        other=dict(self.fact,observation_id='o2',period='2070',value='15.1')
        derived=dict(self.fact,derived=True,value='2.8',formula='operand_2 - operand_1',operands=[self.fact,other])
        row=p.result_check_records([derived])[0]
        self.assertEqual(row['formula'],'operand_2 - operand_1')
        self.assertEqual([f['value']for f in row['operands']],['12.3','15.1'])

    def test_resume_discards_checked_facts_and_raw_decisions(self):
        state=dict(self.state,context={},plan=dict(clarification='¿Qué año?',clarification_field='period'))
        state.update(checked_numeric_results={'n1':[self.fact]},raw_result_checks=[{'old':True}])
        with patch.object(p,'interrupt',return_value='2050'):
            result=self.node('clarify',state)
        self.assertEqual(result['checked_numeric_results'],{})
        self.assertEqual(result['raw_result_checks'],[])

    def test_nonselect_actions_use_actual_pool_without_numeric_execution(self):
        card=dict(card_id='c2',book='2025',metric='Other',series='Observed',unit='euros',scope='aggregate',
                  periods=['2030'],matched_need_ids=['n1'])
        self.state['candidates']=[card]
        self.state['planning_decisions']['n1']=dict(action='unavailable',reason='no compatible capability')
        self.selection=dict(self.selection,card_ids=[],periods=[])
        self.state['plan']['selections']=[self.selection]
        result=self.validated()
        row=result['raw_result_checks'][0]
        self.assertEqual(row['action'],'unavailable')
        self.assertEqual(row['scope'],'retrieved_and_offered_capabilities_only')
        self.assertEqual(row['offered_capability_count'],1)
        self.assertEqual(result['checked_numeric_results'],{})

    def test_empty_or_misbound_selected_records_are_rejected(self):
        for facts in [[],[dict(self.fact,period='other')],[dict(self.fact,card_id='other')],[dict(self.fact,need_id='other')]]:
            with self.subTest(facts=facts):
                result=self.validated(facts)
                self.assertEqual(result['need_outcomes']['n1'],'source_contract_rejected')
                self.assertEqual(result['checked_numeric_results'],{})

    def test_forged_offered_capability_fails_recompilation(self):
        forged=dict(self.cap,capability_id='forged')
        self.state['offered_capabilities']['n1']=[forged]
        self.state['planning_decisions']['n1']['capability_id']='forged'
        result=self.validated()
        self.assertEqual(result['need_outcomes']['n1'],'source_contract_rejected')

    def test_four_actual_records_fit_full_request_bound_and_stage_settings_are_isolated(self):
        rows={'n'+str(i):dict(action='select',operation='lookup',results=[dict(self.fact,need_id='n'+str(i))])for i in range(1,5)}
        payload=dict(query=self.state['query'],context=None,needs=rows)
        self.assertLessEqual(p._wire_size('plan_check_batch',payload,checker_schema({nid:'select'for nid in rows})),6400)
        self.assertEqual(stage_chat_settings('plan_check_batch'),stage_chat_settings('generate'))
        default=stage_chat_settings('generate')
        self.assertFalse(default['think']);self.assertEqual(default['options']['num_predict'],1024)
        modified=stage_chat_settings('plan_check_batch');modified['options']['seed']=99
        self.assertEqual(stage_chat_settings('plan_check_batch')['options']['seed'],42)

    def test_nonselect_pool_is_honest_and_bounded_with_many_long_period_lists(self):
        self.state['candidates']=[dict(card_id='c'+str(i),book='2026',metric='Gasto',series='Central',
            unit='percent_gdp',scope='aggregate',periods=[str(year)for year in range(2020,2200)],
            matched_need_ids=['n1'])for i in range(12)]
        pool=p.result_check_pool(self.state,'n1')
        self.assertEqual(pool['total'],12);self.assertEqual(pool['omitted_source_count'],10)
        self.assertEqual(pool['sources'][0]['period_count'],180)
        self.assertEqual(len(pool['sources'][0]['period_preview']),4)
        rows={'n'+str(i):dict(action='unavailable',pool=pool)for i in range(1,5)}
        payload=dict(query=self.state['query'],context=None,needs=rows)
        self.assertLessEqual(p._wire_size('plan_check_batch',payload,checker_schema({nid:'unavailable'for nid in rows})),6400)

    def packed_passages(self,stage,payload):
        captured={};schema=p.obj(dict(ok=dict(type='boolean')))
        def chat(messages,response_schema,called_stage):
            raw=json.loads(messages[1]['content'])
            if called_stage=='review':
                from rag_pensiones.graph_wire import decode_review_payload
                raw=decode_review_payload(raw)
            captured.update(raw)
            self.assertLessEqual(p._wire_size(called_stage,captured,response_schema),p.MAX_CHAT_REQUEST_BYTES)
            from rag_pensiones.graph_types import stage_chat_settings
            options=stage_chat_settings(called_stage)['options']
            self.assertLessEqual(p.prompt_context_bound(messages,options),options['num_ctx'])
            return dict(ok=True)
        services=SimpleNamespace(client=SimpleNamespace(calls=[],chat=chat))
        p._chat(services,stage,payload,schema)
        return captured

    def test_generation_preserves_relevant_passage_tail_and_canonical_inputs(self):
        relevant='neuroarquitectura '+('contexto '*90)+'Respuesta situada al final.'
        passages=[dict(source_id='noise'+str(i),book='synthetic',text='material ajeno '*75)for i in range(5)]
        passages.append(dict(source_id='relevant',book='synthetic',text=relevant))
        payload=dict(query='Explica neuroarquitectura',needs=[dict(metric='neuroarquitectura')],passages=passages)
        original=copy.deepcopy(payload)
        packed=self.packed_passages('generate',payload)
        self.assertEqual(next(r['text']for r in packed['passages']if r['source_id']=='relevant'),relevant)
        self.assertEqual(payload,original)
        self.assertLess(len(json.dumps(packed)),len(json.dumps(payload)))

    def test_generation_protects_one_relevant_passage_per_text_need(self):
        passages=[dict(source_id='noise'+str(i),book='synthetic',text='contexto general '*70)for i in range(5)]
        for term in ['neuroarquitectura','plasticidad']:
            passages.append(dict(source_id=term,book='synthetic',text=term+' '+('material '*90)+'Respuesta final.'))
        payload=dict(query='Explica neuroarquitectura y plasticidad',needs=[dict(metric=term)for term in ['neuroarquitectura','plasticidad']],passages=passages)
        packed=self.packed_passages('generate',payload)
        for original in passages[-2:]:self.assertIn(original,packed['passages'])

    def test_review_preserves_all_distant_quote_spans(self):
        quotes=['La neuroarquitectura conserva esta evidencia.','Esta segunda cita acredita plasticidad.']
        text=('prefacio '*160)+quotes[0]+(' contexto '*160)+quotes[1]+' cierre'
        passages=[dict(source_id='noise'+str(i),book='synthetic',text='material general '*70)for i in range(4)]
        passages.append(dict(source_id='evidence',book='synthetic',text=text))
        payload=dict(query='Explica neuroarquitectura',needs=[dict(metric='neuroarquitectura')],passages=passages,
            claims=[dict(evidence=[dict(source_id='evidence',quote=quote)for quote in quotes])])
        packed=self.packed_passages('review',payload)
        excerpt=next(r['text']for r in packed['passages']if r['source_id']=='evidence')
        for quote in quotes:self.assertIn(quote,excerpt)
        self.assertEqual(payload['passages'][-1]['text'],text)

    def test_oversized_protected_evidence_fails_before_chat(self):
        from unittest.mock import Mock
        text='neuroarquitectura '+('evidencia '*900)
        payload=dict(query='Explica neuroarquitectura',needs=[dict(metric='neuroarquitectura')],
            passages=[dict(source_id='evidence',book='synthetic',text=text)],
            claims=[dict(evidence=[dict(source_id='evidence',quote=text)])])
        client=SimpleNamespace(calls=[],chat=Mock())
        with self.assertRaisesRegex(ValueError,'Protected documentary evidence'):
            p._chat(SimpleNamespace(client=client),'review',payload,p.obj({}))
        client.chat.assert_not_called()


if __name__=='__main__':unittest.main()
