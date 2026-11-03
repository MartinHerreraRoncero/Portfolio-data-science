"""Closed review identities, flat certified rows and hybrid numeric ownership."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from rag_pensiones import graph_pipeline as p
from rag_pensiones.graph_planning import review_schema,decode_review,checker_schema,grounding_inputs,apply_grounding
from rag_pensiones.graph_validation import (InvalidPlan,contains_financial_quantity,validate_text_claims_isolated,align_literal_quote)
from rag_pensiones.graph_wire import encode_review,decode_review_payload
from rag_pensiones.graph_models import GraphStageClient,MAX_CHAT_REQUEST_BYTES,prompt_context_bound
from rag_pensiones.graph_types import CHAT_MODEL,CHAT_DIGEST
from test_graph_grounding import need,source


def coverage(present=False,scope=False,ids=(),reason='not_in_evaluated_evidence'):
    return dict(requested_result_present=present,requested_scope_supported=scope,claim_ids=list(ids))


class ReviewContractsTests(unittest.TestCase):
    def test_effective_wire_prompts_keep_literal_selection_and_modality_checks(self):
        generated=p._model_messages('generate',dict(query='¿Qué concluye el ensayo?'),p.GENERATE_SCHEMA)[0]['content']
        reviewed=p._model_messages('review',dict(query='¿Qué concluye el ensayo?',needs=[],claims=[],passages=[]),review_schema([],[]))[0]['content']
        self.assertIn('No reescribas citas',generated)
        self.assertIn('Si la pregunta es abierta',generated)
        self.assertIn('preserving modality, strength and causality',reviewed)
        self.assertIn('A limitation is not impossibility',reviewed)
        self.assertIn('Norms/recommendations do not prove necessity or causality',reviewed)
        self.assertIn('Past findings do not establish present facts',reviewed)

    def test_oversized_context_defers_whole_claim_and_preserves_later_supported_goal(self):
        long_text='No extrapolar estas conclusiones. '+('Contexto completo obligatorio. '*600)+'La muestra es local.'
        short_text='El experimento sólo permite describir la muestra observada.'
        documents=[dict(source_id='long',book='2035',text=long_text),dict(source_id='short',book='2035',text=short_text)]
        needs=[dict(need_id=nid,kind='text',metric='alcance',query_span='Explica el alcance')for nid in ['a','b']]
        claims=[dict(claim_id='aC1',need_id='a',text='Conclusión de la fuente extensa.',evidence=[dict(source_id='long',quote='La muestra es local.')]),
                dict(claim_id='bC1',need_id='b',text=short_text,evidence=[dict(source_id='short',quote=short_text)])]
        state=dict(query='Explica el alcance de ambas fuentes',interpretation=dict(needs=needs),documents=documents,
            active_text_need_ids=['a','b'],text_claims=claims,repairs=1)
        service=SimpleNamespace(client=SimpleNamespace(calls=[]))
        def checked_chat(services,stage,payload,schema):
            self.assertEqual(payload['passages'],[documents[1]])
            self.assertEqual([c['claim_id']for c in payload['claims']],['bC1'])
            self.assertTrue(p._payload_fits(stage,payload,schema))
            return dict(judgments=dict(bC1=dict(supported=True,reason='Apoyo directo')),
                coverage=dict(a=coverage(),b=coverage(True,True,['bC1'])))
        with patch.object(p,'_chat',side_effect=checked_chat):
            result=p.build_graph(service).nodes['review'].invoke(state)
        self.assertEqual([c['claim_id']for c in result['text_claims']],['bC1'])
        self.assertTrue(any(r.get('code')=='review_budget_deferred'for r in result['text_rejections']))
        self.assertEqual(state['documents'][0]['text'],long_text)

    def test_literal_but_unsupported_conclusion_is_still_rejected(self):
        quote='La ficha describe la temperatura, sin evaluar la durabilidad.'
        state=dict(query='¿Es duradero el material?',interpretation=dict(needs=[dict(need_id='n',kind='text',metric='durabilidad')]),
            documents=[dict(source_id='s',book='2035',text=quote)],repairs=1,active_text_need_ids=['n'],
            text_claims=[dict(claim_id='nC1',need_id='n',text='El material es duradero.',evidence=[dict(source_id='s',quote=quote)])])
        raw=dict(judgments=dict(nC1=dict(supported=False,reason='La fuente no evalúa durabilidad')),
            coverage=dict(n=coverage(False,False,[])))
        with patch.object(p,'_chat',return_value=raw):
            result=p.build_graph(SimpleNamespace(client=SimpleNamespace(calls=[]))).nodes['review'].invoke(state)
        self.assertEqual(result['text_claims'],[])
        self.assertFalse(result['text_coverage'][0]['answered'])

    def test_extended_transport_preserves_full_protected_passages_within_conservative_context(self):
        quote='The reported value does not fall in 2050.'
        # Leave the configured output reserve available for reasoning + answer.
        passages=[dict(source_id=key,book='synthetic',text=('Full evidence retained. '*70)+quote+(' More context. '*20))for key in ['a','b']]
        claims=[dict(claim_id=key+'C',need_id=key,text='Published statement',evidence=[dict(source_id=key,quote=quote)])for key in ['a','b']]
        needs=[dict(need_id=key,metric='reported value')for key in ['a','b']]
        payload=dict(query='Explain both statements',context=None,needs=needs,claims=claims,passages=passages)
        schema=review_schema(claims,needs);before=copy.deepcopy(payload)
        class Offline:
            def request(self,method,path,data=None):
                if path=='/api/tags':return dict(models=[dict(name=CHAT_MODEL,digest=CHAT_DIGEST)])
                result=dict(judgments={c['claim_id']:dict(supported=True,reason='Mock protocol only')for c in claims},coverage={n['need_id']:coverage(True,True,[n['need_id']+'C'])for n in needs})
                return dict(model=CHAT_MODEL,done=True,done_reason='stop',prompt_eval_count=64,eval_count=64,message=dict(content=json.dumps(result)))
        service=SimpleNamespace(client=GraphStageClient(transport=Offline()))
        p._chat(service,'review',payload,schema);request=service.client.calls[-1]['request']
        self.assertEqual(decode_review_payload(json.loads(request['messages'][1]['content'])),payload)
        size=len(json.dumps(request,ensure_ascii=False,separators=(',',':')).encode())
        self.assertGreater(size,6400);self.assertLessEqual(size,MAX_CHAT_REQUEST_BYTES)
        self.assertLessEqual(prompt_context_bound(request['messages'],request['options']),request['options']['num_ctx'])
        self.assertEqual(payload,before)
        huge=copy.deepcopy(payload)
        for passage in huge['passages']:passage['text']='No evidence may be omitted. '*400+quote
        with self.assertRaisesRegex(ValueError,'Protected documentary evidence'):p._chat(service,'review',huge,schema)
        self.assertEqual(len(service.client.calls),1)

    def test_wire_has_no_coverage_reason_and_negative_codes_are_derived_without_promoting_flags(self):
        claims=[dict(claim_id='a',need_id='n'),dict(claim_id='b',need_id='other')]
        needs=[dict(need_id='n'),dict(need_id='other'),dict(need_id='empty')]
        schema=review_schema(claims,needs)
        self.assertEqual(set(schema['properties']['coverage']['properties']['n']['properties']),
                         {'requested_result_present','requested_scope_supported','claim_ids'})
        for present,scope,reason in [(True,True,'covered'),(False,True,'missing_from_answer'),
                                    (True,False,'scope_mismatch'),(False,False,'not_in_evaluated_evidence')]:
            raw=dict(judgments={c['claim_id']:dict(supported=True,reason='Entailment independent of relevance')for c in claims},
                     coverage=dict(n=coverage(present,scope,['a']),other=coverage(),empty=coverage()))
            before=copy.deepcopy(raw);result=decode_review(raw,claims,needs)['coverage']
            self.assertEqual(result[0]['reason'],reason);self.assertEqual(result[0]['answered'],present and scope)
            self.assertEqual(result[0]['requested_result_present'],present)
            self.assertEqual(result[0]['requested_scope_supported'],scope)
            self.assertEqual(result[2]['reason'],'missing_from_answer');self.assertEqual(raw,before)
        for witness in ['b','unknown']:
            raw['coverage']['n']=coverage(True,True,[witness])
            result=decode_review(raw,claims,needs)['coverage'][0]
            self.assertFalse(result['answered']);self.assertTrue(result['validation_issues'])
        raw['coverage']['n']=coverage(True,True,['a']);raw['judgments']['a']['supported']=False
        self.assertFalse(decode_review(raw,claims,needs)['coverage'][0]['answered'])
        raw['coverage']['n']['requested_result_present']='true'
        with self.assertRaises(InvalidPlan):decode_review(raw,claims,needs)

    def test_real_failed_review_payloads_fit_with_full_context_and_current_schema(self):
        fixtures=json.loads((Path(__file__).parent/'fixtures/review_wire_legacy_metadata_v6.json').read_text(encoding='utf8'))
        for case_id,payload in fixtures.items():
            payload['context']=dict(report_year='2025')
            before=copy.deepcopy(payload);schema=review_schema(payload['claims'],payload['needs'])
            class Offline:
                def request(self,method,path,request=None):
                    if path=='/api/tags':return dict(models=[dict(name=CHAT_MODEL,digest=CHAT_DIGEST)])
                    result=dict(judgments={c['claim_id']:dict(supported=True,reason='Protocol fixture')for c in payload['claims']},coverage={})
                    for n in payload['needs']:
                        ids=[c['claim_id']for c in payload['claims']if c['need_id']==n['need_id']]
                        result['coverage'][n['need_id']]=coverage(bool(ids),bool(ids),ids,'covered'if ids else'not_in_evaluated_evidence')
                    return dict(model=CHAT_MODEL,done=True,done_reason='stop',prompt_eval_count=64,eval_count=64,
                        message=dict(content=json.dumps(result,ensure_ascii=False)))
            service=SimpleNamespace(client=GraphStageClient(transport=Offline()))
            p._chat(service,'review',payload,schema)
            request=service.client.calls[0]['request'];wire=json.loads(request['messages'][1]['content'])
            self.assertEqual(decode_review_payload(wire),payload)
            self.assertEqual(payload,before)
            actual=len(json.dumps(request,ensure_ascii=False,separators=(',',':')).encode())
            self.assertEqual(actual,p._wire_size('review',payload,schema));self.assertLessEqual(actual,MAX_CHAT_REQUEST_BYTES,case_id)
            self.assertEqual(request['format'],p.structural_response_schema(schema))
            self.assertNotIn(json.dumps(schema,ensure_ascii=False,separators=(',',':')),request['messages'][0]['content'])
            if case_id=='b08':self.assertTrue(any('2015'in source['text']for source in decode_review_payload(wire)['passages']))

    def test_review_chat_keeps_full_protected_context_outside_a_late_quote_window(self):
        text='No se autoriza extrapolar a 2015. '+('Contexto documental íntegro. '*50)+'Esta cita está situada al final del pasaje.'
        quote='Esta cita está situada al final del pasaje.'
        self.assertGreater(len(text),1000)
        payload=dict(query='Explica el contexto documental',needs=[dict(need_id='n',metric='contexto documental')],
            claims=[dict(claim_id='nC1',need_id='n',text='Claim',evidence=[dict(source_id='s',quote=quote)])],
            passages=[dict(source_id='s',book='synthetic',text=text)])
        schema=review_schema(payload['claims'],payload['needs']);captured={}
        def chat(messages,response_schema,stage):
            captured.update(decode_review_payload(json.loads(messages[1]['content'])))
            self.assertEqual(response_schema,schema)
            return dict(judgments=dict(nC1=dict(supported=True,reason='Protocol only')),
                coverage=dict(n=coverage(True,True,['nC1'],'covered')))
        service=SimpleNamespace(client=SimpleNamespace(calls=[],chat=chat))
        p._chat(service,'review',payload,schema)
        self.assertEqual(captured,payload)
        self.assertTrue(captured['passages'][0]['text'].startswith('No se autoriza'))

    def test_four_distinct_text_needs_keep_all_claim_and_coverage_identities_in_budget(self):
        query='Explica alcance, método, fuentes y límites.'
        texts=['El alcance es documental.','El método contrasta fuentes.','Las fuentes son públicas.','Los límites quedan expresos.']
        needs=[dict(need_id='n'+str(i),kind='text',metric=metric,query_span=query,book='2035',population='aggregate',
                    scope_terms=[],series=None,source_block=None,unit='unspecified',periods=[],quantity_request='unspecified')
               for i,metric in enumerate(['alcance','método','fuentes','límites'],1)]
        claims=[dict(claim_id=n['need_id']+'C1',need_id=n['need_id'],text=text,evidence=[dict(source_id='s',quote=text)])
                for n,text in zip(needs,texts)]
        raw=dict(query=query,needs=needs,claims=claims,passages=[dict(source_id='s',book='2035',text=' '.join(texts))])
        schema=review_schema(claims,needs);messages=p._model_messages('review',raw,schema)
        self.assertEqual(decode_review_payload(json.loads(messages[1]['content'])),raw)
        self.assertLessEqual(p._wire_size('review',raw,schema),6400)
        self.assertEqual(set(schema['properties']['coverage']['properties']),{n['need_id']for n in needs})

    def test_document_edition_projection_label_and_cell_horizon_are_distinct_fields(self):
        card=source(series='Opinión 2025',metric='Empleos no financieros',periods=['2030'],
            title='Contenedor de varias proyecciones')
        n=need(series='Opinión 2025',metric=card['metric'],periods=['2030'])
        state=dict(candidates=[card],interpretation=dict(needs=[n]),plan=dict(selections=[dict(
            need_id='n1',card_ids=['a'],periods=['2030'],operation='lookup')]))
        row=p.selected_source_views(state,SimpleNamespace(catalog=dict(observations={})),['n1'])[0]
        self.assertEqual((row['edicion_documento'],row['serie_fila'],row['serie_o_version_pedida'],row['horizonte_celda_certificado']),
            ('2026','Opinión 2025','Opinión 2025',['2030']))

    def test_review_only_allows_generated_claim_ids_and_real_need_ids(self):
        claims=[dict(claim_id='n2C1',need_id='n2')];needs=[dict(need_id='n2'),dict(need_id='n3')]
        raw=dict(judgments=dict(n2C1=dict(supported=True,reason='Respaldo')),
            coverage=dict(n2=coverage(True,True,['n2C1'],'covered'),n3=coverage()))
        canonical=decode_review(raw,claims,needs)
        self.assertEqual(canonical['judgments'][0]['claim_id'],'n2C1')
        self.assertEqual([r['need_id']for r in canonical['coverage']],['n2','n3'])
        bads=[]
        bad=copy.deepcopy(raw);bad['judgments']['n2C2']=bad['judgments'].pop('n2C1');bads.append(bad)
        bad=copy.deepcopy(raw);bad['coverage']['other']=bad['coverage'].pop('n3');bads.append(bad)
        bad=copy.deepcopy(raw);bad['judgments']['n2C1']['sql']='SELECT';bads.append(bad)
        wrong=copy.deepcopy(raw);wrong['coverage']['n3']=coverage(True,True,['n2C1'])
        self.assertFalse(decode_review(wrong,claims,needs)['coverage'][1]['answered'])
        for bad in bads:
            with self.subTest(bad=bad),self.assertRaises(InvalidPlan):decode_review(bad,claims,needs)
        empty=decode_review(dict(judgments={},coverage=dict(n2=coverage())),[],needs[:1])
        self.assertEqual(empty['judgments'],[])

    def test_coverage_needs_supported_result_and_scope_witnesses_not_background(self):
        claims=[dict(claim_id='nC1',need_id='n'),dict(claim_id='nC2',need_id='n'),dict(claim_id='otherC1',need_id='other')]
        needs=[dict(need_id='n',query_span='Resultado y comparación completos'),dict(need_id='other')]
        raw=dict(judgments=dict(nC1=dict(supported=False,reason='Requested result unsupported'),
            nC2=dict(supported=True,reason='Background only'),otherC1=dict(supported=True,reason='Independent valid sibling')),
            coverage=dict(n=coverage(True,True,['nC1','nC2'],'covered'),other=coverage(True,True,['otherC1'],'covered')))
        before=copy.deepcopy(raw);result=decode_review(raw,claims,needs)
        self.assertFalse(result['coverage'][0]['answered']);self.assertTrue(result['coverage'][0]['validation_issues'])
        self.assertTrue(result['coverage'][1]['answered']);self.assertEqual(raw,before)
        for changes in [dict(requested_result_present=False,claim_ids=['nC2']),
                        dict(requested_scope_supported=False,claim_ids=['nC2']),
                        dict(claim_ids=[]),dict(claim_ids=['nC2','nC2'])]:
            altered=copy.deepcopy(raw);altered['coverage']['n'].update(changes)
            self.assertFalse(decode_review(altered,claims,needs)['coverage'][0]['answered'])
        altered=copy.deepcopy(raw);altered['coverage']['n']['claim_ids']=['otherC1']
        self.assertFalse(decode_review(altered,claims,needs)['coverage'][0]['answered'])
        # Unknown scope is explicitly false for every dimension, without a
        # business-name heuristic in the validator.
        for filters in [dict(unit='persons'),dict(population='subgroup',scope_terms=['women']),
                        dict(scope_terms=['region']),dict(periods=['2091']),dict(book='2035')]:
            scoped=[dict(needs[0],**filters),needs[1]];altered=copy.deepcopy(raw)
            altered['coverage']['n']=coverage(True,False,['nC2'],'scope_mismatch')
            self.assertFalse(decode_review(altered,claims,scoped)['coverage'][0]['answered'])

    def test_layout_punctuation_alignment_preserves_source_and_rejects_numeric_or_word_changes(self):
        canonical='La economía no crecerá en el escenario .\n Después se explica el 14,5 % completo.'
        quote='La economía no crecerá en el escenario. Después se explica el 14,5% completo.'
        result=align_literal_quote(quote,canonical)
        self.assertEqual(result['quote'],canonical)
        self.assertEqual(canonical[result['start_char']:result['end_char']],result['quote'])
        for invalid in ['La economía crecerá en el escenario.', 'Después se explica el 14.5 % completo.',
                        'Después se explica el 14,6 % completo.', '4,5 % completo.', '5 % completo.']:
            with self.subTest(invalid=invalid),self.assertRaises(InvalidPlan):align_literal_quote(invalid,canonical)
        with self.assertRaises(InvalidPlan):align_literal_quote('El importe es 14,5 %.', 'El importe es 14 , 5 %.')

    def test_wire_roundtrip_is_lossless_for_repeated_nested_overlapping_and_no_quotes(self):
        for text,quotes in [('Negación: no crecerá. Contexto 2015.',['no crecerá']),
                            ('abcabc',['abc','abc']),('abcdef',['abcdef','bcd']),
                            ('abcdef',['abcd','cdef']),('Todo el contexto original.',[])]:
            raw=dict(query='Synthetic',needs=[dict(need_id='n',metric='result')],claims=[dict(claim_id='nC1',need_id='n',
                text='Claim',evidence=[dict(source_id='s',quote=q)for q in quotes])],
                passages=[dict(source_id='s',book='synthetic',text=text)])
            before=copy.deepcopy(raw)
            for fragments in [False,True]:
                wire,_=encode_review(raw,fragments=fragments)
                self.assertEqual(decode_review_payload(wire),raw)
                self.assertEqual(raw,before)

    def test_financial_claims_are_local_only_in_hybrid_mode_and_quotes_years_stay_valid(self):
        for text in ['16,1 % del PIB','0.5 puntos porcentuales','200 millones de euros','€2.000','2026 euros']:
            self.assertTrue(contains_financial_quantity(text),text)
        for text in ['En 2026 se revisaron las bases.','La revisión duró 2 años.','La edad de 65 años cambia en 2026.']:
            self.assertFalse(contains_financial_quantity(text),text)
        doc=dict(source_id='s',book='2026',text='El informe de 2026 publica un 16,1 % del PIB y revisa sus hipótesis.',
            doc_id='d',page_number=1,source_file='synthetic.pdf',source_path='synthetic.pdf',source_sha256='hash')
        def claim(text):return dict(need_id='t',text=text,evidence=[dict(source_id='s',quote=doc['text'])])
        claims=[claim('En 2026 cambian las hipótesis.'),claim('El gasto es 16,1 % del PIB.')]
        accepted,rejections=validate_text_claims_isolated(claims,[dict(need_id='t',kind='text',book='2026')],[doc],hybrid=True)
        self.assertEqual(len(accepted),1);self.assertEqual(rejections[0]['code'],'hybrid_financial_quantity')
        self.assertIn('16,1 %',accepted[0]['evidence'][0]['quote'])
        accepted,rejections=validate_text_claims_isolated(claims,[dict(need_id='t',kind='text',book='2026')],[doc])
        self.assertEqual(len(accepted),2);self.assertEqual(rejections,[])

    def test_unavailable_nonpublication_horizons_cannot_be_deleted(self):
        for periods,book,expected in [(['2026','2199'],'2026','unresolved'),(['2026','2050'],'2026','bound'),
                                     (['2025','2050'],'2026','unresolved')]:
            n=need(periods=periods,book=book);row=source(periods=['2026','2050'])
            intent=dict(needs=[n]);inputs=grounding_inputs(intent,[row],'Informe 2026 frente a 2025: dato en 2050 o 2199.')
            chosen=['2026']if'2199'in periods else['2050']
            out,records=apply_grounding(intent,inputs,dict(n1=dict(scenario=None,target_years=chosen)))
            self.assertEqual(records['n1']['horizon_status'],expected)
            self.assertEqual(out['needs'][0]['periods'],periods if expected=='unresolved'else['2050'])

    def test_four_selected_real_numeric_records_fit_the_checker_budget(self):
        catalog=json.loads((p.ROOT/'data/processed/langgraph_9b_assets/catalog.json').read_text(encoding='utf-8'))
        cards=[next(c for c in catalog['cards']if c['book']=='2026'and c['block'].strip()=='Cuadro 12'
            and c['metric']==metric and c['series']=='AIReF 2026'and c['unit']=='percent_gdp'and'2030'in c['periods'])
            for metric in ['Pensiones','Sanidad','Educación','Intereses']]
        needs=[need('n'+str(i),metric=c['metric'],periods=['2030'])for i,c in enumerate(cards,1)]
        rows={}
        for n,card in zip(needs,cards):
            actual=next(catalog['observations'][oid]for oid in card['observation_ids']
                        if catalog['observations'][oid]['period']=='2030')
            record=p.result_check_records([dict(actual,need_id=n['need_id'],card_id=card['card_id'],derived=False)])[0]
            self.assertEqual(record['value'],actual['value'])
            for field in ['observation_id','book','metric','series','unit','scope','period','source_sheet','source_cell']:
                self.assertEqual(record[field],actual.get(field))
            rows[n['need_id']]=dict(action='select',operation='lookup',results=[record])
        payload=dict(query='Informe 2026: pensiones, sanidad, educación e intereses en 2030, en porcentaje del PIB.',context=None,
            needs=rows)
        schema=checker_schema({n['need_id']:'select'for n in needs})
        messages=p._model_messages('plan_check_batch',payload,schema)
        wire=json.loads(messages[1]['content'])
        self.assertEqual(wire['needs'],rows)
        self.assertNotIn('selected_sources',wire)
        self.assertNotIn(p.CANDIDATE_TABLE_NOTE,messages[0]['content'])
        self.assertLessEqual(p._wire_size('plan_check_batch',payload,schema),6400)


if __name__=='__main__':unittest.main()
