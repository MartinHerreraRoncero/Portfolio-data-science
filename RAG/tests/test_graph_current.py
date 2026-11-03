"""Regression contracts: publication preferences and clarification year roles."""
import copy
import unittest
import json
from types import SimpleNamespace
from unittest.mock import patch
from rag_pensiones.graph_goals import quantitative_goal_schema,publication_policy,explain_alternatives,explanation_need,text_stage_question
from rag_pensiones.graph_planning import decision_schema
from rag_pensiones.graph_validation import check_shape,InvalidPlan


class CurrentQuestionTests(unittest.TestCase):
    def nodes(self,services):
        from rag_pensiones import graph_pipeline as p
        class Graph:
            def __init__(self,*args):self.nodes={}
            def add_node(self,name,fn):self.nodes[name]=fn
            def add_edge(self,*args):pass
            def add_conditional_edges(self,*args):pass
            def compile(self,**kwargs):return self
        with patch.object(p,'StateGraph',Graph):return p.build_graph(services).nodes

    def goal(self):
        return dict(question_fragment='Déficit actual',requested_measure='Déficit actual',
                    answer_kind='quantity',report_publication_year='2026')

    def years(self,query,context):
        schema=quantitative_goal_schema(query,context,self.goal(),publication_years=['2026'])
        return schema['properties']['answer_requests']['items']['properties']['target_data_periods']

    def context(self,field):
        return dict(clarification_question='Pregunta con edición 2026 y periodo 2050',
                    clarification_answer='2026',clarification_field=field)

    def test_edition_answer_cannot_become_data_year(self):
        self.assertEqual(self.years('Déficit actual',self.context('edition'))['maxItems'],0)

    def test_same_year_explicit_in_question_is_still_a_data_choice(self):
        self.assertEqual(self.years('Déficit en 2026',self.context('edition'))['items']['enum'],['2026'])

    def test_period_clarification_preserves_answer_but_not_question_years(self):
        self.assertEqual(self.years('Déficit actual',self.context('period'))['items']['enum'],['2026'])

    def test_nested_editions_excluded_and_assistant_years_ignored(self):
        context=self.context('edition');context['previous_context']=dict(history=[
            dict(role='assistant',content='Dato observado 2024; último año 2050')])
        self.assertEqual(self.years('Déficit actual',context)['maxItems'],0)

    def test_policy_is_explicit_nested_and_never_from_assistant(self):
        self.assertIsNone(publication_policy(dict(role='assistant',publication_policy='latest_supported')))
        self.assertEqual(publication_policy(dict(previous_context=dict(publication_policy='latest_supported'))),'latest_supported')
        self.assertIsNone(publication_policy(dict(history=[dict(role='assistant',content='Usa 2026')])))

    def test_edition_does_not_force_irrelevant_question_over_metric_or_period(self):
        schema=decision_schema([],['edition','metric','period'],edition_choices=['2025','2026'])
        for field in ['metric','period']:
            check_shape(dict(action='clarify',field=field,question='¿Qué definición o periodo necesitas?'),schema)

    def test_no_year_is_invented_for_current_and_no_assistant_preference(self):
        goal=self.goal();goal['report_publication_year']=None
        schema=quantitative_goal_schema('¿Cuál es el dato actual?',dict(history=[dict(role='assistant',content='En 2050')]),goal,publication_years=['2026'])
        self.assertEqual(schema['properties']['answer_requests']['items']['properties']['target_data_periods']['maxItems'],0)

    def test_latest_policy_removes_edition_question_but_retains_temporal_clarification(self):
        from rag_pensiones import graph_pipeline as p
        rows=[dict(card_id='new',book='2026',metric='Financiación publicada',series='Componente',unit='percent_gdp',period_kind='year',periods=['2020','2050'])]
        services=SimpleNamespace(client=SimpleNamespace(calls=[]))
        need=dict(need_id='n1',book=None,query_span='Dato actual')
        def chat(s,stage,payload,schema):
            self.assertNotIn('edition',payload['scope_summary']['material_fields'])
            self.assertIn('metric',payload['scope_summary']['material_fields'])
            self.assertEqual(payload['unresolved_sources'][0]['period_range'],['2020','2050'])
            self.assertEqual(payload['cap_options'],[])
            with self.assertRaises(InvalidPlan):check_shape(dict(action='clarify',field='edition',question='¿2026?'),schema)
            result=dict(action='clarify',field='metric',question='¿Qué definición de la magnitud necesitas?')
            check_shape(result,schema)
            services.client.calls.append(dict(request=dict(messages=p._model_messages(stage,payload,schema))))
            return result
        with patch.object(p,'prepare_need_candidates',return_value=copy.deepcopy(rows)),patch.object(p,'material_axes',return_value=['edition','unit','period']),patch.object(p,'compile_capabilities',return_value=[]),patch.object(p,'capability_options',return_value=[]),patch.object(p,'_chat',side_effect=chat):
            decision,caps,_,_=p._plan_one_need(services,dict(query='Dato actual',effective_context=dict(publication_policy='latest_supported')),need,rows)
        self.assertEqual(decision['field'],'metric');self.assertEqual(caps,[])

    def test_latest_policy_survives_unresolved_scenario_and_metric_only_fallback(self):
        from rag_pensiones import graph_pipeline as p
        rows=[dict(card_id=str(i),book='2026',metric='Indicador',series=str(i),unit='percent_gdp',period_kind='year',periods=['2025','2050'])for i in range(2)]
        services=SimpleNamespace(client=SimpleNamespace(calls=[]));need=dict(need_id='n1',book=None,series='incierto',query_span='Dato actual')
        def chat(s,stage,payload,schema):
            self.assertEqual(payload['scope_summary']['material_fields'],['metric'])
            result=dict(action='clarify',field='metric',question='¿Qué definición necesitas?');check_shape(result,schema)
            services.client.calls.append(dict(request=dict(messages=p._model_messages(stage,payload,schema))))
            return result
        state=dict(query='Dato actual',effective_context=dict(publication_policy='latest_supported'),grounding_records=dict(n1=dict(decision=dict(scenario=dict(action='unresolved')))))
        with patch.object(p,'prepare_need_candidates',return_value=copy.deepcopy(rows)),patch.object(p,'material_axes',return_value=['edition']),patch.object(p,'capability_options',return_value=[]),patch.object(p,'_chat',side_effect=chat):
            decision,caps,_,_=p._plan_one_need(services,state,need,rows)
        self.assertEqual(decision['field'],'metric');self.assertEqual(caps,[])

    def test_latest_policy_does_not_discard_older_capability_with_requested_coverage(self):
        from rag_pensiones import graph_pipeline as p
        rows=[dict(card_id='old',book='2025',unit='euros'),dict(card_id='new',book='2026',unit='euros')]
        cap=dict(capability_id='old_supported',selection=dict(card_ids=['old']))
        services=SimpleNamespace(client=SimpleNamespace(calls=[]));need=dict(need_id='n1',book=None,query_span='Dato en 2070')
        def chat(s,stage,payload,schema):
            result=dict(action='select',capability_id='old_supported');check_shape(result,schema)
            services.client.calls.append(dict(request=dict(messages=p._model_messages(stage,payload,schema))))
            return result
        with patch.object(p,'prepare_need_candidates',return_value=copy.deepcopy(rows)),patch.object(p,'material_axes',return_value=['edition']),patch.object(p,'compile_capabilities',return_value=[cap]),patch.object(p,'capability_options',return_value=[dict(capability_id='old_supported',source_metadata=[rows[0]])]),patch.object(p,'_chat',side_effect=chat):
            decision,caps,_,_=p._plan_one_need(services,dict(query='Dato en2070',effective_context=dict(publication_policy='latest_supported')),need,rows)
        self.assertEqual(decision['capability_id'],'old_supported');self.assertEqual(caps,[cap])

    def test_documentary_retrieval_respects_explicit_publication(self):
        from rag_pensiones.graph_documents import retrieve_documents
        corpus=dict(chunks=[dict(source_id='old',book='2025',text='pensiones déficit pensiones'),dict(source_id='new',book='2026',text='pensiones financiación')])
        with patch('rag_pensiones.graph_documents.validate_document_corpus'):
            self.assertEqual([p['source_id']for p in retrieve_documents(corpus,'pensiones',books={'2026'})],['new'])
            self.assertEqual(retrieve_documents(corpus,'pensiones',books={'2030'}),[])

    def test_abstention_shows_retrieved_documents_without_claiming_numeric_evidence(self):
        from experiments.prototype_presentation import present_report
        report=dict(publication=dict(status='insufficient_evidence',claims=[],reason='Las evidencias no permiten responder'),
                    retrieval=dict(documents=[dict(source_file='Informe.pdf',page_number=53)]))
        result=present_report(report,database=None)
        self.assertIn('no demuestra',result['answer']);self.assertIn('Informe.pdf',result['sources'])
        self.assertEqual(result['calls'],[])

    def test_supported_explanation_is_separate_from_unresolved_numeric_need(self):
        original=dict(need_id='n1',kind='data',metric='Saldo',query_span='Saldo actual',periods=['2026'],book='2026',series='base',unit='euros',quantity_request='level')
        auxiliary=explanation_need(original)
        self.assertEqual((auxiliary['need_id'],auxiliary['kind'],auxiliary['periods']),('n1_explanation','text',['2026']))
        self.assertEqual(set(auxiliary),set(original));self.assertEqual(original['kind'],'data')
        self.assertEqual(original['periods'],['2026']);self.assertEqual(auxiliary['book'],'2026')
        self.assertTrue(explain_alternatives(dict(previous_context=dict(ambiguity_policy='explain_supported_alternatives'))))
        self.assertFalse(explain_alternatives(dict(role='assistant',ambiguity_policy='explain_supported_alternatives')))

    def test_auxiliary_explanation_cannot_publish_original_quantity_as_answered(self):
        from rag_pensiones import graph_pipeline as p
        n=dict(need_id='n1',kind='data',metric='Saldo',query_span='Saldo actual',periods=[],book=None,series=None,unit='unspecified',quantity_request='level')
        seed=dict(source_id='first',doc_id='pdf',page_number=53,text='Inicio auténtico')
        neighbor=dict(source_id='second',doc_id='pdf',page_number=53,text='Continuación auténtica')
        unrelated=dict(source_id='elsewhere',doc_id='pdf',page_number=54,text='Otra página')
        services=SimpleNamespace(client=SimpleNamespace(calls=[]),documents=dict(chunks=[seed,neighbor,unrelated]));nodes=self.nodes(services)
        state=dict(query='Saldo actual',resolved_query='Saldo actual',effective_context=dict(ambiguity_policy='explain_supported_alternatives'),interpretation=dict(needs=[n]),candidates=[],query_vector=[])
        with patch.object(p,'_plan_one_need',return_value=(dict(action='explain',reason='Explicar las definiciones respaldadas'),[],[],dict(material_fields=['metric']))),patch.object(p,'retrieve_documents',return_value=[seed]):
            update=nodes['plan'](state)
        self.assertEqual([d['source_id']for d in update['documents']],['first','second'])
        self.assertEqual(update['documents'][1]['text'],neighbor['text'])
        self.assertNotIn('errors',update);self.assertEqual([x['need_id']for x in update['interpretation']['needs']],['n1','n1_explanation'])
        merged=dict(state,**update);merged.update(plan_valid=True,text_claims=[dict(need_id='n1_explanation',text='Explicación respaldada',evidence=[])],text_coverage=[dict(need_id='n1_explanation',answered=True)])
        published=nodes['publish'](merged)['publication']
        self.assertEqual(published['status'],'partial');self.assertEqual(len(published['missing_parts']),1)
        self.assertIn('No se ha acreditado una cifra',published['missing_parts'][0]['message'])

    def test_no_auxiliary_generation_when_chat_budget_has_no_review_reserve(self):
        from rag_pensiones import graph_pipeline as p
        n=dict(need_id='n1',kind='data',metric='Saldo',query_span='Saldo',periods=[],book=None)
        services=SimpleNamespace(client=SimpleNamespace(calls=[dict(endpoint='/api/chat')]*10));nodes=self.nodes(services)
        state=dict(query='Saldo',effective_context=dict(ambiguity_policy='explain_supported_alternatives'),interpretation=dict(needs=[n]),candidates=[])
        with patch.object(p,'_plan_one_need',return_value=(dict(action='unavailable',reason='Sin capacidad'),[],[],{})):
            update=nodes['plan'](state)
        self.assertNotIn('errors',update);self.assertEqual(update['interpretation']['needs'],[n])

    def test_auxiliary_and_native_text_have_uniform_review_metadata(self):
        from rag_pensiones.graph_wire import encode_review,decode_review_payload
        n=dict(need_id='n1',kind='data',metric='Saldo',query_span='Saldo',periods=[],book=None,series=None,unit='unspecified',quantity_request='level')
        native=dict(n,need_id='n2',kind='text',metric='Explicación')
        payload=dict(needs=[native,explanation_need(n)],claims=[],passages=[])
        self.assertEqual(decode_review_payload(encode_review(payload)[0]),payload)


    def test_auxiliary_text_stage_is_scoped_but_native_question_is_preserved(self):
        self.assertEqual(text_stage_question('Cifra actual',[dict(need_id='n1_explanation')]),'Cifra actual')
        self.assertEqual(text_stage_question('Pregunta mixta',[dict(need_id='n1_explanation'),dict(need_id='n2')]),'Pregunta mixta')



if __name__=='__main__':unittest.main()
