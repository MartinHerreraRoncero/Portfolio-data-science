"""COUNT -> named extraction contracts; all model calls use an offline transport."""
import copy
import json
import unittest
from functools import partial

from rag_pensiones import graph_pipeline as p
from rag_pensiones.graph_planning import COUNT_SCHEMA,named_intent_schema,decode_named_intent,literal_query_spans,UNIT_WIRE_LABELS
from rag_pensiones.graph_validation import InvalidPlan
from rag_pensiones.graph_models import GraphStageClient,MAX_CHAT_REQUEST_BYTES
from rag_pensiones.graph_types import CHAT_MODEL,CHAT_DIGEST,EMBED_MODEL,EMBED_DIGEST
from rag_pensiones.graph_goals import goal_schema,decode_goals,quantitative_goal_schema,extract_goal_needs

# Explicit synthetic source editions for the existing goal contract fixtures.
goal_schema=partial(goal_schema,publication_years=('2026',))
decode_goals=partial(decode_goals,publication_years=('2026',))
quantitative_goal_schema=partial(quantitative_goal_schema,publication_years=('2026',))
extract_goal_needs=partial(extract_goal_needs,publication_years=('2026',))


def request(query,**changes):
    item=dict(report_publication_year=None,target_data_periods=[],requested_unit='no especificada',
        requested_measure='indicador',answer_kind='data',requested_quantity='level',population_type='aggregate',
        explicit_population_group=None,explicit_geographic_area=None,explicit_forecast_scenario=None,
        explicit_table_or_chart_name=None,question_fragment=query)
    item.update(changes)
    item['requested_unit']=UNIT_WIRE_LABELS.get(item['requested_unit'],item['requested_unit'])
    return item


class CountedIntentTests(unittest.TestCase):
    def test_nominal_publication_requires_original_user_provenance_and_audit_locks_null(self):
        from rag_pensiones import graph_goals as g
        goal=dict(question_fragment='Importe del horizonte',requested_measure='importe',answer_kind='quantity',report_publication_year=None)
        for query,context,expected in [
            ('Dato de 2050.',None,[None]),
            ('La publicación más reciente que tenga el dato.',None,[None]),
            ('Dato de 2050.',dict(report_year='2090'),[None]),
            ('Dato de 2050.',[dict(role='assistant',content='Usa informe 2090.')],[None]),
            ('Informe 2090: dato de 2050.',None,['2090',None]),
            ('Dato de 2050.',[dict(role='user',content='Usa informe 2090.')],['2090',None])]:
            with self.subTest(query=query,context=context):
                schema=g.goal_schema(query,context,publication_years=['2090'])
                self.assertEqual(schema['properties']['answer_requests']['items']['properties']['report_publication_year']['enum'],expected)
                numeric=g.quantitative_goal_schema(query,context,goal,publication_years=['2090'])
                self.assertEqual(numeric['properties']['answer_requests']['items']['properties']['report_publication_year']['enum'],[None])
                reply=dict(answer_requests=[request(query,question_fragment=goal['question_fragment'],report_publication_year='2090')])
                with self.assertRaises(InvalidPlan):g.extract_goal_needs(dict(answer_requests=[goal]),query,context,'production_full',lambda *_:reply,publication_years=['2090'])
                nominal=dict(goal,report_publication_year='2090')
                if '2090'in expected:
                    g.decode_goals(dict(answer_requests=[nominal]),query,context,publication_years=['2090'])
                    self.assertEqual(g.quantitative_goal_schema(query,context,nominal,publication_years=['2090'])['properties']['answer_requests']['items']['properties']['report_publication_year']['enum'],['2090'])
                else:
                    with self.assertRaises(InvalidPlan):g.decode_goals(dict(answer_requests=[nominal]),query,context,publication_years=['2090'])
                    with self.assertRaises(InvalidPlan):g.quantitative_goal_schema(query,context,nominal,publication_years=['2090'])

    def test_planner_receives_user_latest_preference_and_can_select_older_supported_capability(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        context=[dict(role='user',content='La publicación más reciente que tenga ese dato')]
        before=copy.deepcopy(context);query='Valor del indicador en el horizonte pedido.'
        rows=[dict(card_id='old',book='2025',unit='euros'),dict(card_id='new',book='2026',unit='euros')]
        cap=dict(capability_id='supported_old',selection=dict(card_ids=['old']))
        n=dict(need_id='n',book=None,query_span=query)
        class Offline:
            def request(self,method,path,payload=None):
                if path=='/api/tags':return dict(models=[dict(name=CHAT_MODEL,digest=CHAT_DIGEST)])
                return dict(model=CHAT_MODEL,done=True,done_reason='stop',prompt_eval_count=64,eval_count=16,
                    message=dict(content=json.dumps(dict(action='select',capability_id='supported_old'))))
        services=SimpleNamespace(client=GraphStageClient(transport=Offline()))
        with patch.object(p,'prepare_need_candidates',return_value=copy.deepcopy(rows)),patch.object(p,'material_axes',return_value=[]),patch.object(p,'compile_capabilities',return_value=[cap]),patch.object(p,'capability_options',return_value=[dict(capability_id='supported_old',source_metadata=[rows[0]])]):
            decision,caps,_,_=p._plan_one_need(services,dict(query=query,effective_context=context),n,rows)
        self.assertEqual(decision['capability_id'],'supported_old');self.assertEqual(caps,[cap])
        request_payload=services.client.calls[-1]['request']
        data=json.loads(request_payload['messages'][1]['content']);self.assertEqual(data['context'],before)
        self.assertEqual(context,before);self.assertEqual(data['scope_summary']['available_books'],['2025','2026'])
        self.assertLessEqual(len(json.dumps(request_payload,ensure_ascii=False,separators=(',',':')).encode()),6400)

    def test_publication_choices_are_from_active_sources_not_data_years(self):
        from types import SimpleNamespace
        from rag_pensiones import graph_goals as g
        services=SimpleNamespace(scope='production_full',catalog=dict(sources={'2025':{},'2026':{}}),
                                 documents=dict(sources=[dict(book='2025')]))
        years=g.source_publication_years(services);query='Dato de 2050 y 2035.'
        self.assertEqual(years,['2025','2026'])
        goal=dict(question_fragment='Importe solicitado',requested_measure='importe',answer_kind='quantity',report_publication_year=None)
        fields=g.goal_schema(query,publication_years=years)['properties']['answer_requests']['items']['properties']
        self.assertEqual(fields['report_publication_year']['enum'],[None])
        raw=dict(answer_requests=[goal]);g.decode_goals(raw,query,publication_years=years)
        schema=g.quantitative_goal_schema(query,None,goal,publication_years=years)
        props=schema['properties']['answer_requests']['items']['properties']
        self.assertEqual(props['target_data_periods']['items']['enum'],['2035','2050'])
        for fake_book in ['2050','2035']:
            with self.assertRaises(InvalidPlan):g.decode_goals(dict(answer_requests=[dict(goal,report_publication_year=fake_book)]),query,publication_years=years)
            reply=dict(answer_requests=[request(query,question_fragment=goal['question_fragment'],report_publication_year=fake_book,target_data_periods=['2050'])])
            with self.assertRaises(InvalidPlan):g.extract_goal_needs(raw,query,None,'production_full',lambda *_:reply,publication_years=years)
        services.scope='legacy_2025_reference'
        self.assertEqual(g.source_publication_years(services),['2025'])

    def test_goal_count_is_derived_and_paraphrases_keep_exact_duplicate_audit(self):
        query='Informe 2026: explica la relación y da el valor del norte en 2040.'
        a=dict(question_fragment=query,requested_measure='relación',answer_kind='explanation',report_publication_year=None)
        b=dict(a,requested_measure='valor del norte',answer_kind='quantity')
        raw=dict(answer_requests=[a,copy.deepcopy(a),b]);before=copy.deepcopy(raw)
        goals,audit=decode_goals(raw,query)
        self.assertEqual([g['need_id']for g in goals],['n1','n3'])
        self.assertEqual(audit,dict(derived_result_count=3,raw_count=3,retained_count=2,
            original_to_retained={'n1':'n1','n2':'n1','n3':'n3'}))
        self.assertEqual(raw,before)
        with self.assertRaises(InvalidPlan):decode_goals(dict(raw,requested_result_count=2),query)
        paraphrase=dict(a,question_fragment='Describe la relación económica en sus propios términos, conservando el ámbito aplicable.')
        self.assertEqual(decode_goals(dict(answer_requests=[paraphrase]),query)[0][0]['question_fragment'],paraphrase['question_fragment'])
        from types import SimpleNamespace
        interpreted=extract_goal_needs(dict(answer_requests=[paraphrase]),query,None,'production_full',lambda *_:None)[0]
        publication=p.build_graph(SimpleNamespace()).nodes['publish'].invoke(dict(query=query,
            interpretation=interpreted,plan_valid=True,numeric_facts=[],text_claims=[],text_coverage=[]))['publication']
        message=publication['missing_parts'][0]['message']
        self.assertIn('Parte interpretada: '+paraphrase['question_fragment'],message)
        self.assertNotIn('«',message)
        self.assertNotIn('requested_result_count',goal_schema(query)['properties'])
        for items in [[],[a]*5]:
            with self.assertRaises(InvalidPlan):decode_goals(dict(answer_requests=items),query)
        for field,value in [('report_publication_year','2026'),('question_fragment','Explica la relación'),('answer_kind','reference')]:
            self.assertEqual(len(decode_goals(dict(answer_requests=[a,dict(a,**{field:value})]),query)[0]),2)

    def test_reference_explanation_and_legacy_quantity_do_not_extract_sql_filters(self):
        query='Informe 2026: ¿referencia histórica y clasificación personal en el sur, y cuánto recibiré en 2040?'
        goals=[dict(question_fragment=query,requested_measure=metric,answer_kind=kind,report_publication_year='2026')
               for metric,kind in [('referencia','reference'),('clasificación','explanation'),('cálculo personal','quantity')]]
        raw=dict(answer_requests=goals);before=copy.deepcopy(raw)
        def forbidden(*args):raise AssertionError('Legacy cannot call numerical extraction')
        intent,replies=extract_goal_needs(raw,query,dict(report_year='2026'),'legacy_2025_reference',forbidden)
        self.assertEqual(replies['by_goal'],{})
        for n in intent['needs']:
            self.assertEqual((n['kind'],n['query_span'],n['book']),('text',query,'2026'))
            self.assertEqual((n['population'],n['scope_terms'],n['periods'],n['unit'],n['quantity_request']),
                             ('unspecified',[],[],'unspecified','unspecified'))
            self.assertIsNone(n['series']);self.assertIsNone(n['source_block'])
        text=dict(answer_requests=goals[:2])
        self.assertEqual(len(extract_goal_needs(text,query,dict(report_year='2026'),'production_full',forbidden)[0]['needs']),2)
        self.assertEqual(raw,before)

    def test_quantitative_attributes_are_isolated_per_goal_and_raw_replies_retained(self):
        query='Informe 2026: explica el pasado y da euros del norte para mujeres en 2040.'
        goals=[dict(question_fragment=query,requested_measure='explicación pasada',answer_kind='explanation',report_publication_year='2026'),
               dict(question_fragment=query,requested_measure='importe mujeres norte',answer_kind='quantity',report_publication_year='2026')]
        raw=dict(answer_requests=goals);before=copy.deepcopy(raw);calls=[]
        reply=dict(answer_requests=[request(query,report_publication_year='2026',target_data_periods=['2040'],requested_unit='euros',
            requested_measure='importe mujeres norte',population_type='subgroup',explicit_population_group='mujeres',
            explicit_geographic_area='norte',explicit_forecast_scenario='inventado')])
        def chat(data,schema):
            calls.append((copy.deepcopy(data),copy.deepcopy(schema)))
            self.assertEqual(data['question'],query);self.assertEqual(data['focus']['requested_measure'],goals[1]['requested_measure'])
            return copy.deepcopy(reply)
        intent,replies=extract_goal_needs(raw,query,None,'production_full',chat)
        self.assertEqual(len(calls),1);self.assertEqual(replies['by_goal'],{'n2':reply})
        text,number=intent['needs']
        self.assertEqual((text['periods'],text['scope_terms'],text['population']),([],[],'unspecified'))
        self.assertEqual((number['need_id'],number['periods'],number['scope_terms'],number['unit']),('n2',['2040'],['mujeres','norte'],'euros'))
        self.assertEqual(intent['discarded_scenario_hints'][0]['need_id'],'n2');self.assertIsNone(number['series'])
        self.assertEqual(raw,before);self.assertEqual(reply,replies['by_goal']['n2'])
        for changes in [dict(answer_kind='text'),dict(question_fragment='explica el pasado'),dict(report_publication_year='2040')]:
            wrong=dict(answer_requests=[dict(reply['answer_requests'][0],**changes)])
            with self.assertRaises(InvalidPlan):extract_goal_needs(raw,query,None,'production_full',lambda *_:wrong)

    def test_numerical_year_pool_uses_user_text_but_publication_pool_keeps_metadata(self):
        query='Da el importe.';context=dict(report_year='2026',history=[dict(role='user',content='En 2040 para el norte.'),
                                                                    dict(role='assistant',content='En 2199.')])
        goal=dict(need_id='n1',question_fragment=query,requested_measure='importe',answer_kind='quantity',report_publication_year=None)
        fields=quantitative_goal_schema(query,context,goal)['properties']['answer_requests']['items']['properties']
        self.assertEqual(fields['target_data_periods']['items']['enum'],['2040'])
        self.assertEqual(fields['report_publication_year']['enum'],[None])
        fields=quantitative_goal_schema(query,dict(report_year='2026'),goal)['properties']['answer_requests']['items']['properties']
        self.assertEqual(fields['target_data_periods']['maxItems'],0)

    def test_four_quantity_budget_and_repeated_context_do_not_reuse_extraction(self):
        query='Valores a, b, c y d en 2040.'
        raw=dict(answer_requests=[dict(question_fragment=query,requested_measure=m,
            answer_kind='quantity',report_publication_year=None)for m in ['a','b','c','d']])
        calls=[]
        def chat(data,schema):
            calls.append(data['focus']['requested_measure'])
            return dict(answer_requests=[request(query,requested_measure=calls[-1],target_data_periods=['2040'])])
        for _ in range(2):self.assertEqual(len(extract_goal_needs(raw,query,None,'production_full',chat)[0]['needs']),4)
        self.assertEqual(calls,['a','b','c','d']*2)
        self.assertEqual(1+4+1+4,10)
        self.assertEqual(1+3+1+3+2,10)

    def test_source_goal_wire_uses_real_client_and_single_goal_locked_schema(self):
        query='Informe 2026: valor en 2040 y explicación.'
        raw=dict(answer_requests=[dict(question_fragment=query,requested_measure=m,
            answer_kind=k,report_publication_year='2026')for m,k in [('valor','quantity'),('explicación','explanation')]])
        class Transport:
            def request(self,method,path,payload=None):
                if path=='/api/tags':return dict(models=[dict(name=CHAT_MODEL,digest=CHAT_DIGEST)])
                kinds=payload['format']['properties']['answer_requests']['items']['properties']['answer_kind']['enum']
                stage='interpret'if 'quantity'in kinds else'intent_audit'
                result=raw if stage=='interpret'else dict(answer_requests=[request(query,report_publication_year='2026',target_data_periods=['2040'])])
                return dict(model=CHAT_MODEL,done=True,done_reason='stop',prompt_eval_count=64,eval_count=64,message=dict(content=json.dumps(result)))
        from types import SimpleNamespace
        services=SimpleNamespace(client=GraphStageClient(transport=Transport()))
        parsed=p._chat(services,'interpret',dict(question=query,context=None),goal_schema(query))
        extract_goal_needs(parsed,query,None,'production_full',lambda data,schema:p._chat(services,'intent_audit',data,schema))
        self.assertEqual([c['stage']for c in services.client.calls],['interpret','intent_audit'])
        for call in services.client.calls:
            self.assertEqual(call['request']['keep_alive'],'0s')
            self.assertLessEqual(len(json.dumps(call['request'],ensure_ascii=False,separators=(',',':')).encode()),6400)

    def test_only_entirely_identical_raw_requests_coalesce_after_count_validation(self):
        query='Explica alcance y límites en 2035 o 2040.'
        item=request(query,requested_measure='alcance y límites')
        raw=dict(answer_requests=[copy.deepcopy(item),copy.deepcopy(item)])
        before=copy.deepcopy(raw);result=decode_named_intent(raw,query,count=2)
        self.assertEqual(len(result['needs']),1)
        self.assertEqual(result['needs'][0]['query_span'],query)
        self.assertEqual(result['request_coalescing'],dict(raw_count=2,retained_count=1,original_to_retained={'n1':'n1','n2':'n1'}))
        self.assertEqual(raw,before)
        with self.assertRaises(InvalidPlan):decode_named_intent(raw,query,count=1)
        for changes in [dict(requested_measure='límites'),dict(requested_unit='euros'),dict(target_data_periods=['2040']),
                        dict(report_publication_year='2035'),dict(explicit_population_group='subgrupo'),
                        dict(explicit_geographic_area='territorio'),dict(explicit_forecast_scenario='actual'),
                        dict(population_type='individual'),dict(answer_kind='text'),dict(requested_quantity='mean')]:
            different=dict(item,**changes)
            intent=decode_named_intent(dict(answer_requests=[item,different]),query,count=2)
            self.assertEqual(len(intent['needs']),2,changes)

    def test_only_original_user_literal_scenario_hints_can_become_filters(self):
        query='Compara publicaciones de 2025 y 2026: dato de 2050.'
        for invented in ['EUROPOP 2025','AIReF 2025']:
            raw=dict(answer_requests=[request(query,explicit_forecast_scenario=invented,
                explicit_population_group='mujeres',explicit_geographic_area='norte')])
            before=copy.deepcopy(raw);intent=decode_named_intent(raw,query,count=1)
            self.assertIsNone(intent['needs'][0]['series'])
            self.assertEqual(intent['needs'][0]['scope_terms'],['mujeres','norte'])
            self.assertEqual(intent['discarded_scenario_hints'][0]['discarded_hint'],invented)
            self.assertEqual(raw,before)
        raw=dict(answer_requests=[request(query,explicit_forecast_scenario='Opinión 2025')])
        assistant=[dict(role='assistant',content='Usa Opinión 2025')]
        self.assertIsNone(decode_named_intent(raw,query,assistant,count=1)['needs'][0]['series'])
        for text,hint in [('La previsión OPINIÓN, 2025','opinión 2025'),
                          ('Usa la actual','actual'),('La anterior','anterior')]:
            context=[dict(role='user',content=text)]
            raw=dict(answer_requests=[request(query,explicit_forecast_scenario=hint)])
            intent=decode_named_intent(raw,query,context,count=1)
            self.assertEqual(intent['needs'][0]['series'],hint)
            self.assertNotIn('discarded_scenario_hints',intent)
        q='Dato según Opinión 2025 en 2050.'
        raw=dict(answer_requests=[request(q,explicit_forecast_scenario='Opinión 2025')])
        self.assertEqual(decode_named_intent(raw,q,count=1)['needs'][0]['series'],'Opinión 2025')

    def test_field_contracts_separate_publication_scenario_horizon_and_unit_dimensions(self):
        query='Informe 2026, escenario Opinión 2025: peso de empleos no financieros en PIB en 2030.'
        props=named_intent_schema(query,None,1)['properties']['answer_requests']['items']['properties']
        for key in ['report_publication_year','target_data_periods','requested_unit','requested_measure','answer_kind','requested_quantity']:
            self.assertTrue(props[key]['description'])
        raw=dict(answer_requests=[request(query,report_publication_year='2026',target_data_periods=['2030'],
            requested_measure='empleos no financieros',requested_unit='percent_gdp',explicit_forecast_scenario='Opinión 2025')])
        n=decode_named_intent(raw,query,count=1)['needs'][0]
        self.assertEqual((n['book'],n['series'],n['periods'],n['unit'],n['metric']),
            ('2026','Opinión 2025',['2030'],'percent_gdp','empleos no financieros'))
        self.assertIn('puntos del PIB',props['requested_unit']['description'])

    def test_spanish_wire_units_decode_exactly_without_changing_raw_or_quantity(self):
        query='Informe 2026: compara datos de 2023 y 2070.'
        for code,label in UNIT_WIRE_LABELS.items():
            raw=dict(answer_requests=[request(query,requested_unit=label,target_data_periods=['2023','2070'],requested_quantity='change')])
            before=copy.deepcopy(raw);n=decode_named_intent(raw,query,count=1)['needs'][0]
            self.assertEqual((n['unit'],n['periods'],n['quantity_request']),(code,['2023','2070'],'change'))
            self.assertEqual(raw,before)
        props=named_intent_schema(query,None,1)['properties']['answer_requests']['items']['properties']
        self.assertEqual(props['requested_unit']['enum'],list(UNIT_WIRE_LABELS.values()))
        self.assertEqual(props['target_data_periods']['items']['enum'],['2023','2026','2070'])
        raw=dict(answer_requests=[request(query,requested_unit='puntos del PIB',target_data_periods=['2070'])])
        self.assertEqual(decode_named_intent(raw,query,count=1)['needs'][0]['periods'],['2070'])
        raw['answer_requests'][0]['requested_unit']='percent_gdp'
        with self.assertRaises(InvalidPlan):decode_named_intent(raw,query,count=1)

    def test_contribution_with_unspecified_unit_preserves_the_published_interval(self):
        from rag_pensiones.graph_catalog import candidate_matches_need
        query='Informe 2026: contribución entre 2023 y 2070.'
        raw=dict(answer_requests=[request(query,report_publication_year='2026',target_data_periods=['2023','2070'],
            requested_quantity='contribution')])
        n=decode_named_intent(raw,query,count=1)['needs'][0]
        self.assertEqual((n['unit'],n['periods']),('unspecified',['2023-2070']))
        source=dict(book='2026',unit='percentage_points',period_kind='interval',periods=['2023-2070'])
        self.assertTrue(candidate_matches_need(source,n))

    def test_count_is_exact_and_publication_horizon_roles_are_model_selected(self):
        query='Informe 2026 frente a 2025: ¿qué cifra se prevé en 2050 y por qué se revisó?'
        raw=dict(answer_requests=[request(query,report_publication_year='2026',target_data_periods=['2050']),
                                  request(query,report_publication_year='2026',answer_kind='text',requested_quantity='unspecified')])
        before=copy.deepcopy(raw);result=decode_named_intent(raw,query,count=2)
        self.assertEqual([n['need_id']for n in result['needs']],['n1','n2'])
        self.assertEqual([n['book']for n in result['needs']],['2026','2026'])
        self.assertEqual(result['needs'][0]['periods'],['2050'])
        self.assertEqual(result['needs'][1]['kind'],'text')
        self.assertEqual(raw,before)
        with self.assertRaises(InvalidPlan):decode_named_intent(raw,query,count=1)

    def test_no_year_in_question_or_context_means_no_foreign_example_year(self):
        query='¿Cuánto supone el indicador y por qué cambia?'
        raw=dict(answer_requests=[request(query)])
        self.assertEqual(decode_named_intent(raw,query,count=1)['needs'][0]['periods'],[])
        for change in [dict(report_publication_year='2038'),dict(target_data_periods=['2038'])]:
            with self.subTest(change=change),self.assertRaises(InvalidPlan):
                decode_named_intent(dict(answer_requests=[request(query,**change)]),query,count=1)
        context=[dict(role='user',content='Hablamos del informe 2026 y horizonte 2050.')]
        raw=dict(answer_requests=[request(query,report_publication_year='2026',target_data_periods=['2050'])])
        self.assertEqual(decode_named_intent(raw,query,context,count=1)['needs'][0]['book'],'2026')

    def test_followup_resolved_search_keeps_the_model_extracted_topic_and_parameters(self):
        query='¿Y en 2050?';context=[dict(role='user',content='Hablamos de pensiones en el informe 2026.')]
        raw=dict(answer_requests=[request(query,report_publication_year='2026',target_data_periods=['2050'],
                                         requested_measure='gasto en pensiones')])
        result=decode_named_intent(raw,query,context,count=1)
        for literal in [query,'pensiones','2026','2050']:self.assertIn(literal,result['resolved_query'])
        self.assertEqual(raw['answer_requests'][0]['requested_measure'],'gasto en pensiones')

    def test_qualifiers_only_population_and_geography_scenario_is_separate(self):
        query='En 2026, para mujeres de Andalucía en el escenario alto, ¿qué valor hay en 2050?'
        raw=dict(answer_requests=[request(query,report_publication_year='2026',target_data_periods=['2050'],
            population_type='subgroup',explicit_population_group='mujeres',explicit_geographic_area='Andalucía',
            explicit_forecast_scenario='alto')])
        result=decode_named_intent(raw,query,count=1)['needs'][0]
        self.assertEqual(result['scope_terms'],['mujeres','Andalucía'])
        self.assertEqual(result['series'],'alto')
        self.assertEqual(result['population'],'subgroup')
        personal=dict(answer_requests=[request(query,population_type='individual')])
        self.assertEqual(decode_named_intent(personal,query,count=1)['needs'][0]['population'],'individual')

    def test_spans_are_full_questions_never_report_prefixes_and_mean_keeps_full_span(self):
        query='Según el informe 2026, ¿cuál es el promedio de 2022 a 2050?'
        spans=literal_query_spans(query)
        self.assertIn(query,spans);self.assertIn('¿cuál es el promedio de 2022 a 2050?',spans)
        self.assertNotIn('Según el informe 2026,',spans)
        raw=dict(answer_requests=[request(query,report_publication_year='2026',target_data_periods=['2050','2022'],requested_quantity='mean')])
        before=copy.deepcopy(raw)
        self.assertEqual(decode_named_intent(raw,query,count=1)['needs'][0]['periods'],['2022-2050'])
        self.assertEqual(raw,before)
        raw['answer_requests'][0]['target_data_periods']=['2022-2038']
        with self.assertRaises(InvalidPlan):decode_named_intent(raw,query,count=1)
        raw['answer_requests'][0]['target_data_periods']=['2022-2050']
        with self.assertRaises(InvalidPlan):decode_named_intent(raw,query,count=1)

    def test_named_four_requests_fit_existing_model_budget(self):
        query='Informe 2024: indica A en 2060, B en 2070, C en 2080 y D en 2090, con las unidades publicadas.'
        raw=dict(answer_requests=[request(query,report_publication_year='2024',target_data_periods=[year],
                                         requested_measure=name)for name,year in zip('ABCD',['2060','2070','2080','2090'])])
        class Offline:
            def request(self,method,path,payload=None):
                if path=='/api/tags':return dict(models=[dict(name=CHAT_MODEL,digest=CHAT_DIGEST),dict(name=EMBED_MODEL,digest=EMBED_DIGEST)])
                result=dict(requested_result_count=4)if 'requested_result_count'in payload['format']['properties']else raw
                return dict(model=CHAT_MODEL,done=True,done_reason='stop',prompt_eval_count=64,eval_count=64,
                            message=dict(content=json.dumps(result,ensure_ascii=False)))
        class Services:client=GraphStageClient(transport=Offline())
        services=Services()
        count=p._chat(services,'interpret',dict(question=query,context=None),COUNT_SCHEMA)
        result=p._chat(services,'intent_audit',dict(question=query,context=None,requested_result_count=count['requested_result_count']),named_intent_schema(query,None,4))
        self.assertEqual(len(decode_named_intent(result,query,count=4)['needs']),4)
        self.assertEqual(len(services.client.calls),2)
        for call in services.client.calls:
            self.assertLessEqual(len(json.dumps(call['request'],ensure_ascii=False,separators=(',',':')).encode()),MAX_CHAT_REQUEST_BYTES)

    def test_full_named_schema_keeps_real_long_questions_and_documentation_within_wire_bound(self):
        queries=[
            'En el estudio de AIReF de 2026, ¿qué porcentaje del PIB prevé la serie AIReF 2026 para los empleos no financieros en 2030?',
            'En la comparación que publica AIReF en 2026, ¿qué peso de los empleos no financieros sobre el PIB asigna la previsión de la Opinión 2025 a 2030?']
        for query in queries:
            schema=named_intent_schema(query,None,1);before=copy.deepcopy(schema)
            class Offline:
                def request(self,method,path,payload=None):
                    if path=='/api/tags':return dict(models=[dict(name=CHAT_MODEL,digest=CHAT_DIGEST)])
                    return dict(model=CHAT_MODEL,done=True,done_reason='stop',prompt_eval_count=64,eval_count=64,
                        message=dict(content=json.dumps(dict(answer_requests=[request(query)]),ensure_ascii=False)))
            class Services:client=GraphStageClient(transport=Offline())
            services=Services();data=dict(question=query,context=None,requested_result_count=1)
            p._chat(services,'intent_audit',data,schema)
            actual=services.client.calls[0]['request']
            self.assertLessEqual(len(json.dumps(actual,ensure_ascii=False,separators=(',',':')).encode()),MAX_CHAT_REQUEST_BYTES)
            self.assertEqual(json.loads(actual['messages'][1]['content']),data)
            self.assertIn(json.dumps(schema,ensure_ascii=False,separators=(',',':')),actual['messages'][0]['content'])
            self.assertEqual(schema,before)


if __name__=='__main__':unittest.main()
