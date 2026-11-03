"""Compiled arguments and dynamic wire schemas, with no model inference."""
import copy
import json
from types import SimpleNamespace
import unittest

from test_graph_need_retrieval import catalog,card,need
from rag_pensiones import graph_pipeline as p
from rag_pensiones.graph_planning import (compile_capabilities,decision_schema,canonical_decision,
                                         prepare_need_candidates,checker_schema)
from rag_pensiones.graph_validation import check_shape,InvalidPlan
from rag_pensiones.graph_models import GraphStageClient,MAX_CHAT_REQUEST_BYTES
from rag_pensiones.graph_types import CHAT_MODEL,CHAT_DIGEST,EMBED_MODEL,EMBED_DIGEST


def intent(**changes):
    result=dict(need(),metric='indicator',population='aggregate',scope_terms=[],series=None,
                source_block=None,quantity_request='level')
    result.update(changes);return result


def services(cards):
    return SimpleNamespace(catalog=catalog(cards),source_policy=dict(groups=[]))


class GraphCapabilitiesV2Tests(unittest.TestCase):
    def test_two_annual_variations_cannot_answer_a_full_span_change_but_one_remains_a_lookup(self):
        from rag_pensiones.graph_catalog import candidate_matches_need,retrieve_catalog_for_needs
        from rag_pensiones.graph_validation import validate_plan
        s=services([card('annual_changes',unit='percentage_points_gdp',periods=['2050','2070'])])
        n=intent(unit='percentage_points_gdp',periods=['2050','2070'],quantity_request='change')
        self.assertFalse(candidate_matches_need(s.catalog['cards'][0],n))
        self.assertEqual(retrieve_catalog_for_needs(s.catalog,'indicator',None,[n]),[])
        self.assertEqual(retrieve_catalog_for_needs(s.catalog,'indicator',None,[n],metadata_discovery=True),[])
        self.assertEqual(compile_capabilities(n,s.catalog['cards'],s),[])
        for operation in ['lookup','published_change','difference']:
            selection=dict(need_id=n['need_id'],card_ids=['annual_changes'],periods=['2050','2070'],operation=operation)
            with self.subTest(operation=operation),self.assertRaisesRegex(InvalidPlan,'annual published variations'):
                validate_plan(dict(status='ready',selections=[selection]),dict(needs=[n]),s.catalog['cards'])
        single=dict(n,periods=['2050'])
        caps=compile_capabilities(single,s.catalog['cards'],s)
        self.assertEqual([(c['selection']['operation'],c['selection']['periods'])for c in caps],[('lookup',['2050'])])

    def test_two_books_allow_material_edition_clarification_without_selected_sources(self):
        s=services([card('old',book='2025'),card('new',book='2026')]);n=intent(book=None)
        class Offline:
            calls=[]
            def chat(self,messages,schema,stage):
                branch=next(b for b in schema['oneOf']if b['properties']['action']['enum']==['clarify'])
                response=dict(action='clarify',field='edition',question='¿Qué edición permite la comparación solicitada?')
                check_shape(response,schema)
                self.calls.append(dict(request=dict(messages=messages)))
                self.schema=schema;return response
        s.client=Offline()
        decision,caps,rows,summary=p._plan_one_need(s,dict(query='Consulta'),n,s.catalog['cards'])
        self.assertEqual(summary['available_books'],['2025','2026'])
        self.assertTrue(caps)
        data=json.loads(s.client.calls[-1]['request']['messages'][1]['content'])
        preview=data['cap_options']
        self.assertNotIn('selection_blocked_reason',summary)
        self.assertEqual({source['book']for option in preview for source in option['source_metadata']},{'2025','2026'})
        for option in preview:
            self.assertIn('capability_id',option)
            self.assertEqual(option['operation'],'lookup');self.assertEqual(option['periods'],['2050'])
            for source in option['source_metadata']:
                self.assertTrue(source['measure']);self.assertEqual(source['series'],'reference')
                self.assertNotIn('value',source)
        with self.assertRaises(InvalidPlan):check_shape(dict(action='select',capability_id='preview-is-not-selectable'),s.client.schema)
        self.assertEqual(canonical_decision(n,decision,caps)['card_ids'],[])
        self.assertIn(decision['field'],summary['material_fields'])
        check_shape(dict(action='clarify',field='scenario',question='¿Escenario?'),s.client.schema)
        exact=intent(book='2026',unit='percent_gdp',periods=['2050'],series='reference')
        from rag_pensiones.graph_planning import material_axes
        self.assertEqual(material_axes(exact,[s.catalog['cards'][1]]),['edition','unit','period','population','metric','scenario'])
        with self.assertRaises(InvalidPlan):check_shape(decision,decision_schema([],[]))

    def test_point_level_has_no_change_operation_to_generate(self):
        s=services([card('year'),dict(card('average',periods=['2022-2050']),period_kind='interval')])
        caps=compile_capabilities(intent(),s.catalog['cards'],s)
        self.assertEqual(len(caps),1)
        self.assertEqual(caps[0]['selection']['operation'],'lookup')
        self.assertEqual(caps[0]['selection']['periods'],['2050'])
        schema=decision_schema(caps,[])
        check_shape(dict(action='select',capability_id=caps[0]['capability_id']),schema)
        for response in [dict(action='select',capability_id='forged'),
                         dict(action='select',capability_id=caps[0]['capability_id'],sql='SELECT anything'),
                         dict(action='clarify',field='edition',question='Edition?'),dict(action='unknown')]:
            with self.subTest(response=response),self.assertRaises(InvalidPlan):check_shape(response,schema)
        selected=canonical_decision(intent(),dict(action='select',capability_id=caps[0]['capability_id']),caps)
        self.assertEqual(selected['card_ids'],['year'])
        with self.assertRaises(ValueError):canonical_decision(intent(),dict(action='select',capability_id='forged'),caps)

    def test_edition_preview_trimming_preserves_both_books_without_forcing_clarification(self):
        cards=[dict(card(str(i),book='2025'if i%2 else'2026'),metric='Descriptive published measure '+str(i)+'x'*430)
               for i in range(14)]
        s=services(cards);n=intent(book=None)
        class Offline:
            def request(self,method,path,payload=None):
                if path=='/api/tags':return dict(models=[dict(name=CHAT_MODEL,digest=CHAT_DIGEST)])
                return dict(model=CHAT_MODEL,done=True,done_reason='stop',prompt_eval_count=64,eval_count=64,
                    message=dict(content=json.dumps(dict(action='unavailable',reason='Potential sources are unrelated to the requested measure'))))
        s.client=GraphStageClient(transport=Offline())
        decision,caps,rows,summary=p._plan_one_need(s,dict(query='A different requested measure'),n,s.catalog['cards'])
        self.assertEqual(decision['action'],'unavailable');self.assertTrue(caps)
        self.assertLess(len(rows),len(cards))
        actual=s.client.calls[-1]['request'];data=json.loads(actual['messages'][1]['content'])
        self.assertLessEqual(len(json.dumps(actual,ensure_ascii=False,separators=(',',':')).encode()),MAX_CHAT_REQUEST_BYTES)
        self.assertEqual({src['book']for option in data['cap_options']for src in option['source_metadata']},{'2025','2026'})
        self.assertNotIn('clarification_candidates',data)
        self.assertEqual([branch['properties']['action']['enum']for branch in actual['format']['oneOf']],
                         [['select'],['clarify'],['unavailable']])

    def test_interval_mean_full_span_and_relative_percent_growth_are_distinct(self):
        s=services([dict(card('mean',periods=['2022-2050']),period_kind='interval'),
                    dict(card('growth',unit='percent_growth_as_published',periods=['2023-2070']),period_kind='interval')])
        mean=compile_capabilities(intent(periods=['2022-2050'],quantity_request='mean'),s.catalog['cards'],s)
        self.assertEqual([(c['source_card_id'],c['selection']['operation'])for c in mean],[('mean','lookup')])
        growth=compile_capabilities(intent(unit='percent',periods=['2023','2070'],quantity_request='change'),s.catalog['cards'],s)
        self.assertEqual([(c['source_card_id'],c['selection']['operation'])for c in growth],[('growth','published_change')])
        self.assertEqual(growth[0]['selection']['periods'],['2023-2070'])

    def test_difference_requires_two_safe_same_identity_point_operands(self):
        s=services([card('levels',periods=['2023','2070'])])
        n=intent(unit='percentage_points_gdp',periods=['2023','2070'],quantity_request='change')
        self.assertEqual(compile_capabilities(n,s.catalog['cards'],s),[])
        for row in s.catalog['observations'].values():row['calculation_safe']=True
        caps=compile_capabilities(n,s.catalog['cards'],s)
        self.assertEqual(len(caps),1);self.assertEqual(caps[0]['selection']['operation'],'difference')
        self.assertEqual(compile_capabilities(dict(n,quantity_request='level'),s.catalog['cards'],s),[])

    def test_requested_level_cannot_select_growth_even_with_the_same_percent_family(self):
        s=services([card('growth',unit='percent_growth_as_published',periods=['2023'])])
        level=intent(unit='percent',periods=['2023'],quantity_request='level')
        self.assertEqual(compile_capabilities(level,s.catalog['cards'],s),[])
        growth=compile_capabilities(dict(level,quantity_request='change'),s.catalog['cards'],s)
        self.assertEqual(growth[0]['selection']['operation'],'lookup')

    def test_need_pool_respects_membership_and_its_own_rank_before_packing(self):
        s=services([card('other'),card('own2'),card('own1')])
        for c,nid,rank in zip(s.catalog['cards'],['N2','N1','N1'],[1,2,1]):
            c.update(matched_need_ids=[nid],need_retrieval=[dict(need_id=nid,rank=rank)])
        selected=prepare_need_candidates(intent(),s.catalog['cards'],s)
        self.assertEqual([c['card_id']for c in selected],['own1','own2'])

    def test_clarification_axis_is_conditional_and_unavailability_is_explicit(self):
        schema=decision_schema([],['edition'])
        check_shape(dict(action='clarify',field='edition',question='¿Qué edición?'),schema)
        check_shape(dict(action='unavailable',reason='No certified source'),schema)
        for response in [dict(action='clarify',field='',question='Question'),
                         dict(action='clarify',field='unit',question='Question'),
                         dict(action='clarify',field='edition',question='')]:
            with self.assertRaises(InvalidPlan):check_shape(response,schema)

    def test_wire_packing_rebuilds_enums_and_never_keeps_omitted_source_choices(self):
        s=services([card(f'c{i}',description='indicator')for i in range(16)])
        for i,c in enumerate(s.catalog['cards']):
            c.update(title='source '+str(i)+' '+'detailed scope '*20,matched_need_ids=['N1'],need_retrieval=[])
        class Offline:
            def request(self,method,path,payload=None):
                if path=='/api/tags':return dict(models=[dict(name=CHAT_MODEL,digest=CHAT_DIGEST),dict(name=EMBED_MODEL,digest=EMBED_DIGEST)])
                data=json.loads(payload['messages'][1]['content']);ids={c['capability_id']for c in data['cap_options']}
                branch=next(b for b in payload['format']['oneOf']if b['properties']['action']['enum']==['select'])
                self_test.assertEqual(ids,set(branch['properties']['capability_id']['enum']))
                presented={c['card_id']for cap in data['cap_options']for c in cap['source_metadata']}
                self_test.assertTrue(presented<={c['card_id']for c in s.catalog['cards']})
                return dict(model=CHAT_MODEL,done=True,done_reason='stop',prompt_eval_count=64,eval_count=64,
                    message=dict(content=json.dumps(dict(action='select',capability_id=data['cap_options'][0]['capability_id']))))
        self_test=self;s.client=GraphStageClient(transport=Offline())
        n=intent();state=dict(query='Indica el indicador de 2050.',interpretation=dict(needs=[n]))
        decision,caps,rows,summary=p._plan_one_need(s,state,n,s.catalog['cards'])
        self.assertLess(len(rows),16)
        self.assertIn(decision['capability_id'],{c['capability_id']for c in caps})
        request=s.client.calls[-1]['request']
        self.assertLessEqual(len(json.dumps(request,ensure_ascii=False,separators=(',',':')).encode()),MAX_CHAT_REQUEST_BYTES)


if __name__=='__main__':unittest.main()
