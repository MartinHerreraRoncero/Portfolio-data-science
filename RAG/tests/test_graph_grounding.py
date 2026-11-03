"""Source-grounded interface contracts with explicit offline model replies."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from rag_pensiones import graph_pipeline as p
from rag_pensiones.graph_planning import (grounding_inputs,grounding_schema,apply_grounding,
    checker_schema,decode_checker,CHECKER_APPROVALS,compile_capabilities,unresolved_scenario_token)
from rag_pensiones.graph_validation import check_shape,InvalidPlan
from rag_pensiones.graph_models import GraphStageClient,MAX_CHAT_REQUEST_BYTES
from rag_pensiones.graph_types import CHAT_MODEL,CHAT_DIGEST,EMBED_MODEL,EMBED_DIGEST


def need(nid='n1',**changes):
    result=dict(need_id=nid,kind='data',metric='Pensiones',book='2026',unit='percent_gdp',
        population='aggregate',scope_terms=[],series=None,source_block=None,periods=['2050'],
        quantity_request='level',query_span='Consulta')
    result.update(changes);return result


def source(cid='a',nid='n1',**changes):
    result=dict(card_id=cid,book='2026',block='Objeto sintético',title='Pensiones',metric='Pensiones',
        series='Previsión central',unit='percent_gdp',scope='aggregate',period_kind='year',
        periods=['2025','2050'],quantity_role='level',period_warnings=[],observation_ids=[],
        matched_need_ids=[nid],need_retrieval=[])
    result.update(changes);return result


def services(rows,reply=None):
    class Offline:
        def request(self,method,path,payload=None):
            if path=='/api/tags':return dict(models=[dict(name=CHAT_MODEL,digest=CHAT_DIGEST),dict(name=EMBED_MODEL,digest=EMBED_DIGEST)])
            data=json.loads(payload['messages'][1]['content'])
            result=reply(data,payload['format'])if reply else {nid:dict(scenario=row['locked_series'],target_years=row['years'])
                for nid,row in data['needs'].items()}
            return dict(model=CHAT_MODEL,done=True,done_reason='stop',prompt_eval_count=64,eval_count=64,
                message=dict(content=json.dumps(result,ensure_ascii=False)))
    return SimpleNamespace(catalog=dict(cards=rows,observations={},catalog_hash='synthetic'),
        source_policy=dict(groups=[]),client=GraphStageClient(transport=Offline()))


class GroundingTests(unittest.TestCase):
    def test_requested_horizon_requires_nonempty_offered_years_or_explicit_null(self):
        rows=[source(periods=['2037','2049'])]
        history=[dict(role='user',content='Los periodos pedidos son 2037 y 2049.')]
        for requested in [[],['2037'],['2037','2049'],['2199']]:
            with self.subTest(requested=requested):
                intent=dict(needs=[need(periods=requested)])
                before=copy.deepcopy(intent)
                inputs=grounding_inputs(intent,rows,'Consulta del informe 2026.',history)
                schema=grounding_schema(inputs);target=schema['properties']['n1']['properties']['target_years']
                if requested and not inputs['n1']['years']:
                    self.assertEqual(target,dict(type='null'))
                    for invalid in [[],['2199'],'2199',False]:
                        with self.assertRaises(InvalidPlan):check_shape(dict(n1=dict(scenario=None,target_years=invalid)),schema)
                else:
                    self.assertEqual(target['type'],['array','null'])
                    self.assertEqual(target['minItems'],1 if requested else 0)
                    self.assertEqual(target['maxItems'],len(inputs['n1']['years']))
                check_shape(dict(n1=dict(scenario=None,target_years=None)),schema)
                if requested:
                    with self.assertRaises(InvalidPlan):check_shape(dict(n1=dict(scenario=None,target_years=[])),schema)
                    out,records=apply_grounding(intent,inputs,dict(n1=dict(scenario=None,target_years=None)))
                    self.assertEqual(out['needs'][0]['periods'],requested)
                    self.assertEqual(records['n1']['horizon_status'],'unresolved')
                else:check_shape(dict(n1=dict(scenario=None,target_years=[])),schema)
                if inputs['n1']['years']:
                    out,records=apply_grounding(intent,inputs,dict(n1=dict(scenario=None,target_years=inputs['n1']['years'])))
                    self.assertEqual(out['needs'][0]['periods'],requested)
                    self.assertEqual(records['n1']['horizon_status'],'bound')
                with self.assertRaises(InvalidPlan):check_shape(dict(n1=dict(scenario=None,target_years=['2199'])),schema)
                self.assertEqual(intent,before)

    def test_four_protected_real_source_cards_fit_without_losing_semantic_metadata(self):
        fixture=json.loads((Path(__file__).parent/'fixtures/graph_four_need_grounding_metadata_v6.json').read_text(encoding='utf8'))
        before=copy.deepcopy(fixture);rows=fixture['candidates'];intent=fixture['interpretation'];query=fixture['query']
        self.assertEqual((len(rows),len(intent['needs'])),(4,4))
        s=services(rows)
        out,records,_=p._grounding_batch(s,dict(query=query,effective_context=fixture['effective_context'],
            interpretation=intent,candidates=rows))
        request=s.client.calls[0]['request'];wire=json.loads(request['messages'][1]['content'])
        presented=p.unpack_candidate_table(wire['candidates'])
        self.assertEqual({row['card_id']for row in presented},{row['card_id']for row in rows})
        self.assertEqual(wire['omitted_grounding_sources'],0)
        self.assertEqual(wire['query'],query);self.assertEqual(wire['context'],fixture['effective_context'])
        self.assertEqual(len(records),4);self.assertEqual(out['needs'],intent['needs'])
        expected_schema=grounding_schema(grounding_inputs(intent,rows,query,fixture['effective_context']))
        from rag_pensiones.graph_models import structural_response_schema
        self.assertEqual(request['format'],structural_response_schema(expected_schema))
        predicted=p._wire_size('grounding',dict(wire,candidates=presented),expected_schema)
        actual=len(json.dumps(request,ensure_ascii=False,separators=(',',':')).encode())
        self.assertEqual(predicted,actual);self.assertLessEqual(actual,MAX_CHAT_REQUEST_BYTES)
        originals={row['card_id']:row for row in rows}
        for source in presented:
            original=originals[source['card_id']];context=original['semantic_context']
            for key in ['book','series','unit','scope','block','quantity_role','period_kind']:
                self.assertEqual(source[key],original[key])
            self.assertEqual(source['measure'],context['semantic_label'])
            self.assertEqual(source['series_role'],context['series_role'])
            reconstructed=dict(source.get('semantic_context')or{},semantic_label=source['measure'],series_role=source['series_role'])
            for key in ['source_scope','scope_qualifier','forecast_role','forecast_vintage']:
                if source[key]is not None:reconstructed[key]=source[key]
            self.assertEqual(reconstructed,context)
            self.assertEqual(source['years'],['2030'])
            self.assertNotIn('value',source)
        # Grounding cannot change the requested operation or any raw card.
        self.assertEqual([n['quantity_request']for n in out['needs']],[n['quantity_request']for n in intent['needs']])
        self.assertEqual(fixture,before)

    def test_horizons_are_monotone_and_null_series_cannot_acquire_a_restriction(self):
        q='Informe 2026 frente a 2025: dato en 2050.'
        rows=[source(periods=['2025','2026','2050','2199'])]
        intent=dict(needs=[need(periods=['2050'])]);inputs=grounding_inputs(intent,rows,q)
        self.assertEqual(inputs['n1']['years'],['2050'])
        for years in [['2025'],['2026'],['2050','2025']]:
            with self.subTest(years=years),self.assertRaises(InvalidPlan):
                apply_grounding(intent,inputs,dict(n1=dict(scenario=None,target_years=years)))
        with self.assertRaises(InvalidPlan):apply_grounding(intent,inputs,dict(n1=dict(scenario='Previsión central',target_years=['2050'])))
        for original in [[],['2199']]:
            intent=dict(needs=[need(periods=original)]);inputs=grounding_inputs(intent,rows,q)
            self.assertEqual(inputs['n1']['years'],[])
            with self.assertRaises(InvalidPlan):apply_grounding(intent,inputs,dict(n1=dict(scenario=None,target_years=['2026'])))
        intent=dict(needs=[need(series='BAD_HINT_MUST_NOT_LEAK',periods=['2050'])]);s=services(rows)
        p._grounding_batch(s,dict(query=q,effective_context=None,interpretation=intent,candidates=rows))
        payload=json.loads(s.client.calls[0]['request']['messages'][1]['content'])
        self.assertNotIn('before',payload['needs']['n1'])
        self.assertNotIn('BAD_HINT_MUST_NOT_LEAK',json.dumps(payload))

    def test_source_contract_retains_selected_records_without_sending_grounding_to_a_checker(self):
        rows=[source(series='Selected row'),source('unused',series='DISCARDED_LABEL')]
        rows[0]['observation_ids']=['selected:record']
        s=services(rows);s.catalog['observations']['selected:record']=dict(observation_id='selected:record',period='2050',source_sheet='Sheet',source_cell='A1')
        n=need(series='Selected row');caps=compile_capabilities(n,rows,s)
        decisions=dict(n1=dict(action='select',capability_id=caps[0]['capability_id']));offered=dict(n1=caps)
        state=dict(query='Dato publicado en 2050.',effective_context=None,interpretation=dict(needs=[n]),candidates=rows,
            plan=p._canonical_plan([n],decisions,offered),planning_decisions=decisions,offered_capabilities=offered,
            checker_scope=['n1'],planning_card_ids=['a'],ambiguity_summaries={},repairs=0,
            grounding_records=dict(n1=dict(before=dict(series='BAD_HINT'),offered=dict(sources=[rows[1]]))))
        nodes={};original=p._safe
        def capture(fn):nodes[fn.__name__]=original(fn);return nodes[fn.__name__]
        with patch.object(p,'_safe',side_effect=capture):p.build_graph(s)
        fact=dict(need_id='n1',card_id='a',observation_id='selected:record',book='2026',metric='Pensiones',
            series='Selected row',period='2050',value='12.3',unit='percent_gdp',scope='aggregate',
            source_sheet='Sheet',source_cell='A1',derived=False)
        with patch.object(p,'_chat')as chat,patch.object(p,'execute_plan',return_value=[fact])as sql:
            result=nodes['validate'](state)
        sql.assert_called_once();chat.assert_not_called()
        self.assertTrue(result['plan_valid'],result)
        wire=json.dumps(result['raw_result_checks'])
        for forbidden in ['unused','DISCARDED_LABEL','BAD_HINT','grounding','offered','before']:
            self.assertNotIn(forbidden,wire)
        self.assertEqual(result['checked_numeric_results'],{'n1':[fact]})

    def test_edition_clarification_is_prioritized_without_forcing_it_over_selection(self):
        from rag_pensiones.graph_planning import decision_schema
        cap=dict(capability_id='real-capability')
        schema=decision_schema([cap],['edition','scenario','period'],edition_choices=['2025','2026'])
        check_shape(dict(action='select',capability_id='real-capability'),schema)
        branch=next(b for b in schema['oneOf']if b['properties']['action']['enum']==['clarify'])
        self.assertEqual(branch['properties']['field']['enum'],['edition','scenario','period'])
        self.assertNotIn('enum',branch['properties']['question'])
        question='¿Qué publicación deseas consultar?'
        check_shape(dict(action='clarify',field='edition',question=question),schema)
        check_shape(dict(action='clarify',field='scenario',question='¿Escenario?'),schema)

    def test_series_component_rows_remain_selectable_without_a_forced_scenario_question(self):
        rows=[source(series='Total',metric='Empleos no financieros'),source('component',series='Componente',metric='Empleos no financieros')]
        n=need(metric='Total de empleos no financieros')
        def select_row(data,schema):
            cap=next(c for c in data['cap_options']if c['source_metadata'][0]['series']=='Total')
            return dict(action='select',capability_id=cap['capability_id'])
        s=services(rows,reply=select_row)
        decision,caps,_,_=p._plan_one_need(s,dict(query='Total de empleos no financieros en 2050.'),n,rows)
        self.assertEqual(decision['action'],'select')
        self.assertTrue(caps)

    def test_discovery_keeps_a_real_horizon_despite_an_extra_publication_year(self):
        from test_graph_need_retrieval import catalog,card
        from rag_pensiones.graph_catalog import retrieve_catalog_for_needs
        from rag_pensiones.graph_validation import validate_plan
        n=need(periods=['2026','2050'])
        c=catalog([card('only2050',periods=['2050'],description='Pensiones')])
        self.assertEqual(retrieve_catalog_for_needs(c,'Pensiones',None,[n]),[])
        average=dict(c['cards'][0],period_kind='interval',periods=['2022-2050'])
        from rag_pensiones.graph_catalog import candidate_matches_need
        self.assertFalse(candidate_matches_need(average,n,metadata_discovery=True))
        found=retrieve_catalog_for_needs(c,'Pensiones',None,[n],metadata_discovery=True)
        self.assertEqual([r['card_id']for r in found],['only2050'])
        inputs=grounding_inputs(dict(needs=[n]),found,'Informe 2026: dato en 2050.')
        self.assertEqual(inputs['n1']['years'],['2050'])
        out,records=apply_grounding(dict(needs=[n]),inputs,dict(n1=dict(scenario=None,target_years=['2050'])))
        self.assertEqual(out['needs'][0]['periods'],['2050']);self.assertEqual(records['n1']['horizon_status'],'bound')
        selected=dict(need_id='n1',card_ids=['only2050'],periods=['2050'],operation='lookup')
        validate_plan(dict(status='ready',selections=[selected]),out,found)
        with self.assertRaises(InvalidPlan):validate_plan(dict(status='ready',selections=[selected]),dict(needs=[n]),found)

    def test_missing_explicit_horizon_cannot_be_erased_or_replaced_by_a_source_year(self):
        n=need(periods=['2199']);rows=[source(periods=['2050'])]
        inputs=grounding_inputs(dict(needs=[n]),rows,'Informe 2026: dato en 2199.')
        self.assertEqual(inputs['n1']['years'],[])
        with self.assertRaises(InvalidPlan):apply_grounding(dict(needs=[n]),inputs,dict(n1=dict(scenario=None,target_years=[])))
        raw=dict(n1=dict(scenario=None,target_years=None))
        out,records=apply_grounding(dict(needs=[n]),inputs,raw)
        self.assertEqual(out['needs'][0]['periods'],['2199'])
        self.assertEqual(records['n1']['horizon_status'],'unresolved')
        with patch.object(p,'_chat',return_value=dict(action='unavailable',reason='No existe horizonte compatible')):
            with patch.object(p,'capability_options',return_value=[]):
                s=services(rows);s.client.calls=[dict(request=dict(messages=[{},dict(content='{"cap_options":[]}')]))]
                decision,caps,_,_=p._plan_one_need(s,dict(query='Dato2199',grounding_records=records),out['needs'][0],rows)
        self.assertEqual(decision['action'],'unavailable');self.assertEqual(caps,[])
        with self.assertRaises(InvalidPlan):apply_grounding(dict(needs=[n]),inputs,dict(n1=dict(scenario=None,target_years=['2050'])))
        from rag_pensiones.graph_validation import validate_plan
        selected=dict(need_id='n1',card_ids=['a'],periods=['2050'],operation='lookup')
        with self.assertRaises(InvalidPlan):validate_plan(dict(status='ready',selections=[selected]),dict(needs=[n]),rows)
        n=need(periods=[]);inputs=grounding_inputs(dict(needs=[n]),rows,'Informe 2026.')
        _,records=apply_grounding(dict(needs=[n]),inputs,dict(n1=dict(scenario=None,target_years=[])))
        self.assertEqual(records['n1']['horizon_status'],'not_requested')

    def test_binding_refines_only_series_and_literal_source_horizon_with_full_history(self):
        n=need(series='previsión publicada por la institución',periods=['2025','2050'])
        intent=dict(needs=[n],resolved_query='Original');before=copy.deepcopy(intent)
        inputs=grounding_inputs(intent,[source(periods=['2025','2050','2051'])], 'Informe 2026 frente al 2025: dato en 2050.')
        self.assertEqual(inputs['n1']['years'],['2025','2050'])
        raw=dict(n1=dict(scenario='Previsión central',target_years=['2050']))
        out,records=apply_grounding(intent,inputs,raw)
        self.assertEqual(intent,before)
        self.assertEqual(out['needs'][0],dict(n,series='Previsión central',periods=['2050']))
        self.assertEqual(records['n1']['before'],dict(series=n['series'],periods=['2025','2050']))
        self.assertEqual(records['n1']['decision']['scenario'],dict(action='bind',label='Previsión central'))
        for mutation in [dict(scenario='Not offered',target_years=['2050']),
                         dict(scenario=raw['n1']['scenario'],target_years=['2051']),
                         dict(scenario=raw['n1']['scenario'],target_years=['2050'],sql='SELECT anything'),
                         dict(scenario=raw['n1']['scenario'],target_years=['2050','2050'])]:
            with self.subTest(mutation=mutation),self.assertRaises(InvalidPlan):apply_grounding(intent,inputs,dict(n1=mutation))

    def test_exact_series_is_locked_and_nonrestriction_requires_recorded_model_choice(self):
        intent=dict(needs=[need(series='PREVISIÓN CENTRAL')]);q='Dato en 2050.'
        inputs=grounding_inputs(intent,[source()],q)
        with self.assertRaises(InvalidPlan):apply_grounding(intent,inputs,dict(n1=dict(
            scenario=None,target_years=['2050'])))
        original=dict(needs=[need(series='2050')]);inputs=grounding_inputs(original,[source()],q)
        unresolved=dict(n1=dict(scenario=unresolved_scenario_token(inputs['n1']),target_years=['2050']))
        out,_=apply_grounding(original,inputs,unresolved);self.assertEqual(out['needs'][0]['series'],'2050')
        raw=dict(n1=dict(scenario=None,target_years=['2050']))
        out,records=apply_grounding(original,inputs,raw)
        self.assertEqual(out['needs'][0]['series'],'2050');self.assertEqual(records['n1']['decision']['scenario']['action'],'unresolved')
        n=need(quantity_request='mean',periods=['2025-2050'])
        inputs=grounding_inputs(dict(needs=[n]),[source(period_kind='interval',periods=['2025-2050'])], 'Promedio 2025 a 2050.')
        out,_=apply_grounding(dict(needs=[n]),inputs,dict(n1=dict(scenario=None,target_years=['2050','2025'])))
        self.assertEqual(out['needs'][0]['periods'],['2025-2050'])

    def test_four_needs_share_one_bounded_call_with_only_presented_source_enums(self):
        rows=[source('c'+str(i),'n'+str(i))for i in range(1,5)]
        ns=[need('n'+str(i))for i in range(1,5)];s=services(rows)
        out,records,raw=p._grounding_batch(s,dict(query='Informe 2026: cuatro datos en 2050.',
            effective_context=None,interpretation=dict(needs=ns),candidates=rows))
        self.assertEqual(len(s.client.calls),1);self.assertEqual(len(records),4)
        request=s.client.calls[0]['request'];wire=json.loads(request['messages'][1]['content'])
        self.assertLessEqual(len(json.dumps(request,ensure_ascii=False,separators=(',',':')).encode()),MAX_CHAT_REQUEST_BYTES)
        presented=p.unpack_candidate_table(wire['candidates'])
        self.assertEqual({c['card_id']for c in presented},{c['card_id']for c in rows})
        self.assertTrue(all('value'not in c for c in presented))
        check_shape(raw,request['format']);self.assertEqual([n['periods']for n in out['needs']],[['2050']]*4)
        huge=[source(title='Metadata '*3000)];s=services(huge)
        with self.assertRaisesRegex(ValueError,'Protected grounding'):p._grounding_batch(s,dict(query='2050',effective_context=None,
            interpretation=dict(needs=[need()]),candidates=huge))
        self.assertEqual(s.client.calls,[])

    def test_unresolved_one_label_is_scoped_unavailable_not_forced_clarification(self):
        n=need(series='Proyección no encontrada');rows=[source()];s=services(rows,lambda data,schema:dict(action='unavailable',reason='No hay vínculo respaldado'))
        binding=dict(decision=dict(scenario=dict(action='unresolved',label=None)))
        decision,caps,_,summary=p._plan_one_need(s,dict(query='Dato 2050',grounding_records=dict(n1=binding)),n,rows)
        self.assertEqual(decision['action'],'unavailable');self.assertEqual(caps,[])
        self.assertIn('scenario',summary['material_fields']);self.assertEqual(len(s.client.calls),1)
        empty=services([],lambda data,schema:dict(action='unavailable',reason='No hay fuentes pertinentes'))
        decision,_,_,_=p._plan_one_need(empty,dict(query='Dato 2050',grounding_records=dict(n1=binding)),n,[])
        self.assertEqual(decision['action'],'unavailable');self.assertEqual(len(empty.client.calls),1)

    def test_unresolved_multiple_labels_can_offer_but_never_force_a_material_question(self):
        n=need(series='Referencia ambigua');rows=[source(),source('b',series='Previsión alternativa')]
        s=services(rows,reply=lambda data,schema:dict(action='clarify',field='scenario',question='¿Qué escenario quieres comparar?'))
        binding=dict(decision=dict(scenario=dict(action='unresolved',label=None)))
        decision,caps,_,summary=p._plan_one_need(s,dict(query='Dato 2050',grounding_records=dict(n1=binding)),n,rows)
        self.assertEqual(decision['action'],'clarify');self.assertEqual(caps,[])
        self.assertIn('scenario',summary['material_fields'])
        self.assertEqual(len(s.client.calls),1)

    def test_action_checker_cannot_use_another_action_approval_or_legacy_boolean(self):
        actions=dict(a='select',b='clarify',c='unavailable')
        raw={nid:dict(decision=CHECKER_APPROVALS[action],reason='Contrato de la acción satisfecho')for nid,action in actions.items()}
        self.assertTrue(all(v['valid']for v in decode_checker(raw,actions).values()))
        for nid in actions:
            bad=copy.deepcopy(raw);bad[nid]['decision']=next(v for v in CHECKER_APPROVALS.values()if v!=bad[nid]['decision'])
            with self.assertRaises(InvalidPlan):decode_checker(bad,actions)
        with self.assertRaises(InvalidPlan):decode_checker(dict(a=dict(valid=True,issue='none',reason='Legacy')),dict(a='select'))
        raw['a']['decision']='rechazar';self.assertFalse(decode_checker(raw,actions)['a']['valid'])

    def test_plan_repair_reserves_text_generation_and_review_and_keeps_local_rejection(self):
        rows=[source()];s=services(rows);n=need();text=need('t',kind='text')
        caps=compile_capabilities(n,rows,s);decisions=dict(n1=dict(action='select',capability_id=caps[0]['capability_id']),t=dict(action='text'))
        offered=dict(n1=caps,t=[]);state=dict(query='Dato y explicación en 2050',effective_context=None,
            interpretation=dict(needs=[n,text]),candidates=rows,plan=p._canonical_plan([n,text],decisions,offered),
            planning_decisions=decisions,offered_capabilities=offered,checker_scope=['n1'],
            planning_card_ids=['a'],ambiguity_summaries={},repairs=0)
        nodes={};original=p._safe
        def capture(fn):nodes[fn.__name__]=original(fn);return nodes[fn.__name__]
        with patch.object(p,'_safe',side_effect=capture):p.build_graph(s)
        s.client.calls=[dict(endpoint='/api/chat')for _ in range(8)]
        with patch.object(p,'_chat')as chat,patch.object(p,'execute_plan',side_effect=InvalidPlan('Protected source record unavailable'))as sql:
            out=nodes['validate'](state)
        sql.assert_called_once();chat.assert_not_called()
        self.assertTrue(out['plan_valid']);self.assertEqual(out['need_outcomes']['n1'],'source_contract_rejected')
        self.assertIsNone(out['repair_need_id']);self.assertNotIn('errors',out)


if __name__=='__main__':unittest.main()
