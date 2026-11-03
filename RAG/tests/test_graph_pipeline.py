"""Real graph/checkpoint/replay wiring with an explicit offline transport.

The fake language replies test source-bound mechanics, not model quality.
"""
import copy
from functools import lru_cache
import json
import os
import sqlite3
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from rag_pensiones import graph_pipeline as m
from rag_pensiones.graph_catalog import build_catalog,attach_embeddings
from rag_pensiones.graph_documents import legacy_document_corpus,build_document_corpus,attach_document_embeddings
from rag_pensiones.graph_models import GraphStageClient
from rag_pensiones.graph_types import CHAT_MODEL,CHAT_DIGEST,EMBED_MODEL,EMBED_DIGEST,EMBED_DIMENSION
from rag_pensiones.graph_validation import replay_graph_report,runtime_manifest
from rag_pensiones.search import DEFAULT_INDEX_PATH,load_index
from rag_pensiones.qwen_embeddings import _json
from rag_pensiones.graph_wire import decode_review_payload


def stage_payload(payload,stage):
    data=json.loads(payload['messages'][1]['content'])
    return decode_review_payload(data)if stage=='review'else data


def review_coverage(needs,claims,accepted=None,reason='not_in_evaluated_evidence'):
    result={}
    for need in needs:
        nid=need['need_id'];ids=[c['claim_id']for c in claims if c['need_id']==nid]
        valid=bool(ids)and(accepted is None or nid in accepted)
        result[nid]=dict(requested_result_present=valid,requested_scope_supported=valid,
            claim_ids=ids if valid else[])
    return result


def named_response(needs,question,schema):
    from rag_pensiones.graph_planning import UNIT_WIRE_LABELS
    fields=schema['properties']['answer_requests']['items']['properties']
    allowed=fields['report_publication_year']['enum']
    fragment=fields['question_fragment'].get('enum',[question])[0]
    return dict(answer_requests=[dict(report_publication_year=n.get('book')if n.get('book')in allowed else None,
        target_data_periods=[year for label in n.get('periods',[])for year in label.split('-')],
        requested_unit=UNIT_WIRE_LABELS[n['unit']],requested_measure=n['metric'],
        answer_kind=n['kind'],requested_quantity=n.get('quantity_request','unspecified'),population_type=n['population'],
        explicit_population_group=None,explicit_geographic_area=None,explicit_forecast_scenario=n.get('series'),
        explicit_table_or_chart_name=None,question_fragment=fragment)for n in needs])


def goal_response(measures,question,schema,book='2025'):
    allowed=schema['properties']['answer_requests']['items']['properties']['report_publication_year']['enum']
    return dict(answer_requests=[dict(question_fragment=question,
        requested_measure=metric,answer_kind=kind,report_publication_year=book if book in allowed else None)
        for metric,kind in measures])


def grounding_response(data):
    return {nid:dict(scenario=row['locked_series'],
            target_years=row['years'])
            for nid,row in data['needs'].items()}


def checker_response(data,rejected=()):
    from rag_pensiones.graph_planning import CHECKER_APPROVALS
    return {nid:dict(decision='rechazar'if nid in rejected else CHECKER_APPROVALS[row['action']],
        reason='Deliberate semantic rejection fixture'if nid in rejected else'Approved fixture decision')
        for nid,row in data['needs'].items()}


class FourNeedTransport:
    """Offline semantic choices over real public records; never an accuracy claim."""
    def __init__(self,reject_need=None,metrics=None):
        self.reject_need=reject_need;self.plans={};self.checks=0
        self.metrics=metrics or ['Pensiones','Sanidad','Educación','Intereses']
    def request(self,method,path,payload=None):
        if path=='/api/tags':return {'models':[{'name':CHAT_MODEL,'digest':CHAT_DIGEST},{'name':EMBED_MODEL,'digest':EMBED_DIGEST}]}
        if path=='/api/embed':return {'model':EMBED_MODEL,'embeddings':[[1.]+[0.]*(EMBED_DIMENSION-1)for _ in payload['input']]}
        stage=next(key for key,text in m.PROMPTS.items()if payload['messages'][0]['content'].startswith(text+'\n'))
        data=stage_payload(payload,stage)
        if stage=='interpret':
            result=goal_response([(metric,'quantity')for metric in self.metrics],data['question'],payload['format'],book='2026')
        elif stage=='intent_audit':
            needs=[dict(kind='data',metric=metric,population='aggregate',book='2026',series=None,
                unit='percent_gdp',periods=['2030'],quantity_request='level')
                for metric in self.metrics if metric==data['focus']['requested_measure']]
            result=named_response(needs,data['question'],payload['format'])
        elif stage=='grounding':result=grounding_response(data)
        elif stage=='plan_need':
            nid=data['need']['need_id'];self.plans[nid]=self.plans.get(nid,0)+1
            cards=[c for cap in data['cap_options']for c in cap['source_metadata']];metric=data['need']['metric'].casefold()
            options=[c for c in cards if metric in c.get('published_measure_name',c['measure']).casefold()and c['series']=='AIReF 2026']
            chosen=next(cap for cap in data['cap_options']if cap['source_metadata'][0]['card_id']in{c['card_id']for c in options})
            result=dict(action='select',capability_id=chosen['capability_id'])
        elif stage=='plan_check_batch':
            self.checks+=1
            result=checker_response(data,{self.reject_need})
        else:raise AssertionError('Unexpected stage '+stage)
        return dict(model=CHAT_MODEL,done=True,done_reason='stop',prompt_eval_count=64,eval_count=64,
                    message=dict(content=json.dumps(result,ensure_ascii=False)))


class OfflineTransport:
    def __init__(self):
        self.requests=[]
        self.clarify_once=False
        self.plans=0
        self.bad_id=False
        self.truncate_plan=False

    def request(self,method,path,payload=None):
        self.requests.append((method,path,copy.deepcopy(payload)))
        if path=='/api/tags':return {'models':[{'name':CHAT_MODEL,'digest':CHAT_DIGEST},{'name':EMBED_MODEL,'digest':EMBED_DIGEST}]}
        if path=='/api/embed':return {'model':EMBED_MODEL,'embeddings':[[1.]+[0.]*(EMBED_DIMENSION-1)for _ in payload['input']]}
        stage=next(key for key,text in m.PROMPTS.items()if payload['messages'][0]['content'].startswith(text+'\n'))
        data=stage_payload(payload,stage)
        if stage=='interpret':
            kind='quantity'if self.clarify_once and self.plans==0 else'reference'
            result=goal_response([('generosidad',kind)],data['question'],payload['format'])
        elif stage=='intent_audit':
            need=dict(kind='data'if self.clarify_once and self.plans==0 else'text',metric='generosidad',population='aggregate',
                      book='2025',series=None,unit='unspecified',periods=[])
            result=named_response([need],data['question'],payload['format'])
        elif stage=='plan':
            self.plans+=1
            clarify=self.clarify_once and self.plans==1
            result=dict(status='clarification'if clarify else'ready',selections=[dict(need_id='N1',card_ids=['unknown']if self.bad_id else[],periods=[],operation='lookup',reason='Documentary source')],
                        clarification='¿Qué periodo te interesa?'if clarify else'',clarification_field='period'if clarify else'')
        elif stage=='plan_check':result={'valid':True,'reason':'Offline wiring check'}
        elif stage=='grounding':result=grounding_response(data)
        elif stage=='plan_need':
            self.plans+=1
            if self.clarify_once and self.plans==1:
                branch=next(b for b in payload['format']['oneOf']if b['properties']['action']['enum']==['clarify'])
                field=branch['properties']['field']['enum'][0]
                question=branch['properties']['question'].get('enum',['¿Qué periodo te interesa?'])[0]
                result=dict(action='clarify',field=field,question=question)
            elif self.bad_id:result=dict(action='select',capability_id='unknown')
            elif data['cap_options']:result=dict(action='select',capability_id=data['cap_options'][0]['capability_id'])
            else:result=dict(action='unavailable',reason='No compatible source')
        elif stage=='plan_check_batch':
            result=checker_response(data)
        elif stage=='generate':
            source=next(p for p in data['evidence_options']if '2015'in p['quote']and'generosidad'in p['quote'])
            result={'claims':[dict(need_id='n1',text='La tasa de generosidad volvería a niveles próximos a los alcanzados en 2015.',
                       evidence=[dict(evidence_id=source['evidence_id'])])]}
        elif stage=='document_focus':result=dict(source_ids=[p['source_id']for p in data['passages'][:3]])
        elif stage=='review':result={'judgments':{c['claim_id']:dict(supported=True,reason='Offline literal evidence')for c in data['claims']},
                                     'coverage':review_coverage(data['needs'],data['claims'])}
        if stage in {'plan','plan_need','generate'}and self.truncate_plan:
            return {'model':CHAT_MODEL,'done':True,'done_reason':'length','message':{'content':'{"status":'},'prompt_eval_count':64,'eval_count':1024}
        return {'model':CHAT_MODEL,'done':True,'done_reason':'stop','message':{'content':json.dumps(result,ensure_ascii=False)},
                'prompt_eval_count':64,'eval_count':64}


@lru_cache(maxsize=1)
def base_services():
    catalog=build_catalog()
    v=[1.]+[0.]*(EMBED_DIMENSION-1)
    catalog=attach_embeddings(catalog,[v]*len(catalog['cards']))
    pages=[_json(line)for line in (m.ROOT/'data/processed/airef_2025_pages.jsonl').read_text(encoding='utf-8').splitlines()]
    docs=legacy_document_corpus(load_index(m.ROOT/DEFAULT_INDEX_PATH),pages)
    return m.GraphServices(catalog,docs,GraphStageClient(transport=OfflineTransport()),scope='legacy_2025_reference')


@lru_cache(maxsize=1)
def full_services():
    base=base_services();docs=build_document_corpus();v=[1.]+[0.]*(EMBED_DIMENSION-1)
    docs=attach_document_embeddings(docs,[v]*len(docs['chunks']))
    return m.GraphServices(base.catalog,docs,GraphStageClient(transport=OfflineTransport()))


def make_services(production=False):
    """Reusable offline fixture for evaluator integration; never an accuracy oracle."""
    services=copy.copy(full_services()if production else base_services())
    services.client=GraphStageClient(transport=OfflineTransport())
    return services


def fixture_four_sources(services):
    rows=[]
    for i,metric in enumerate(['Pensiones','Sanidad','Educación','Intereses']):
        source=next(c for c in services.catalog['cards']if c['book']=='2026'and c['block'].strip()=='Cuadro 12'
                    and c['metric']==metric and c['series']=='AIReF 2026'and c['unit']=='percent_gdp'and'2030'in c['periods'])
        source=copy.deepcopy(source);source.update(matched_need_ids=[f'n{i+1}'],need_retrieval=[],retrieval=dict(mode='offline_primary_fixture'))
        rows.append(source)
    return rows


def fixture_narrative_source(services):
    return [copy.deepcopy(next(c for c in services.documents['chunks']if c['book']=='2025'
                        and'2015'in c['text']and'generosidad'in c['text']))]


class GraphPipelineTests(unittest.TestCase):
    def test_hybrid_quantified_text_preserves_data_and_closed_review_replays_partial(self):
        services=make_services(production=True);group=services.source_policy['groups'][0]
        anchor=copy.deepcopy(next(c for c in services.catalog['cards']if c['card_id']==group['preferred_card_id']))
        anchor.update(matched_need_ids=['n1'],need_retrieval=[],retrieval=dict(mode='offline_primary_fixture'))
        class HybridTransport(OfflineTransport):
            def __init__(self):super().__init__();self.generations=0;self.feedbacks=[];self.review_ids=[]
            def request(self,method,path,payload=None):
                if path!='/api/chat':return super().request(method,path,payload)
                stage=next(k for k,text in m.PROMPTS.items()if payload['messages'][0]['content'].startswith(text+'\n'))
                data=stage_payload(payload,stage)
                if stage=='interpret':result=goal_response([('gasto en pensiones','quantity'),('causa de la revisión','explanation')],data['question'],payload['format'])
                elif stage=='intent_audit':
                    ns=[dict(kind='data',metric='gasto en pensiones',population='aggregate',book='2025',series=None,
                        unit='percent_gdp',periods=['2050'],quantity_request='level'),
                        dict(kind='text',metric='causa de la revisión',population='aggregate',book='2025',series=None,
                        unit='unspecified',periods=[],quantity_request='unspecified')]
                    result=named_response(ns[:1],data['question'],payload['format'])
                elif stage=='generate':
                    self.generations+=1;self.feedbacks.append(data['feedback']);s=data['evidence_options'][0]
                    texts=['El gasto asciende al 16,1 % en 2026.']
                    if self.generations==1:texts.append('El cambio corresponde a las hipótesis del informe.')
                    result=dict(claims=[dict(need_id='n2',text=t,evidence=[dict(evidence_id=s['evidence_id'])])for t in texts])
                elif stage=='review':
                    ids=[c['claim_id']for c in data['claims']];self.review_ids.append(ids)
                    if set(payload['format']['properties']['judgments']['properties'])!=set(ids):raise AssertionError('Review IDs not grounded')
                    result=dict(judgments={cid:dict(supported=False,reason='No concrete causal support in this fixture')for cid in ids},
                        coverage=review_coverage(data['needs'],data['claims'],accepted=set()))
                else:return super().request(method,path,payload)
                return dict(model=CHAT_MODEL,done=True,done_reason='stop',prompt_eval_count=64,eval_count=64,
                    message=dict(content=json.dumps(result,ensure_ascii=False)))
        transport=HybridTransport();services.client=GraphStageClient(transport=transport)
        query='Informe 2025: gasto en pensiones en 2050 en porcentaje del PIB y por qué cambia.'
        with patch.object(m,'retrieve_catalog_for_needs',return_value=[anchor]),patch.object(m,'retrieve_documents',return_value=fixture_narrative_source(services)):
            report=m.answer_graph(query,services=services)
            self.assertEqual(report['status'],'partial',report['publication'])
            self.assertEqual(report['numeric_facts'][0]['value'],'16.059173455264322000000000')
            self.assertEqual([c['need_id']for c in report['publication']['claims']],['n1'])
            self.assertEqual(sum(r.get('code')=='hybrid_financial_quantity'for r in report['text_rejections']),2)
            self.assertEqual(transport.review_ids,[['n2C1'],[]]);self.assertIn('SIN cifras',transport.feedbacks[1])
            self.assertEqual(sum(c['endpoint']=='/api/chat'for c in report['stage_calls']),8)
            for call in report['stage_calls']:
                if call['endpoint']=='/api/chat':self.assertLessEqual(len(json.dumps(call['request'],ensure_ascii=False,separators=(',',':')).encode()),6400)
            self.assertEqual(replay_graph_report(report,services=services)['publication'],report['publication'])

    def test_grounded_horizon_and_hint_reexpand_policy_and_replay_without_rewriting_raw(self):
        services=make_services(production=True)
        group=services.source_policy['groups'][0]
        anchor=copy.deepcopy(next(c for c in services.catalog['cards']if c['card_id']==group['preferred_card_id']))
        anchor.update(matched_need_ids=['n1'],need_retrieval=[],retrieval=dict(mode='offline_primary_fixture'))
        hint='previsión del informe de 2025'
        class GroundedTransport(OfflineTransport):
            def request(self,method,path,payload=None):
                if path!='/api/chat':return super().request(method,path,payload)
                stage=next(k for k,text in m.PROMPTS.items()if payload['messages'][0]['content'].startswith(text+'\n'))
                data=stage_payload(payload,stage)
                if stage=='interpret':result=goal_response([('gasto en pensiones','quantity')],data['question'],payload['format'])
                elif stage=='intent_audit':
                    n=dict(kind='data',metric='gasto en pensiones',population='aggregate',book='2025',
                        series=hint,unit='percent_gdp',periods=['2025','2050'],quantity_request='level')
                    result=named_response([n],data['question'],payload['format'])
                elif stage=='grounding':result=dict(n1=dict(scenario=None,target_years=['2050']))
                elif stage=='plan_need':
                    cap=next(c for c in data['cap_options']if [source['card_id']for source in c['source_metadata']]==[group['preferred_card_id']])
                    result=dict(action='select',capability_id=cap['capability_id'])
                else:return super().request(method,path,payload)
                return dict(model=CHAT_MODEL,done=True,done_reason='stop',prompt_eval_count=64,eval_count=64,
                    message=dict(content=json.dumps(result,ensure_ascii=False)))
        services.client=GraphStageClient(transport=GroundedTransport())
        query='Según la previsión del informe de 2025: gasto en pensiones en 2050, en porcentaje del PIB.'
        with patch.object(m,'retrieve_catalog_for_needs',return_value=[anchor]),patch.object(m,'retrieve_documents',return_value=fixture_narrative_source(services)):
            report=m.answer_graph(query,services=services)
            self.assertEqual(report['status'],'answered',report['publication'])
            self.assertEqual(report['raw_intent_audit']['by_goal']['n1']['answer_requests'][0]['explicit_forecast_scenario'],hint)
            record=report['grounding_records']['n1']
            self.assertEqual(record['before'],dict(series=hint,periods=['2025','2050']))
            self.assertEqual(record['after'],dict(series=None,periods=['2050']))
            self.assertEqual({s['card_id']for s in record['offered']['sources']},{anchor['card_id']})
            self.assertNotIn('2025',anchor['periods'])
            self.assertEqual(report['plan']['selections'][0]['card_ids'],[group['preferred_card_id']])
            self.assertEqual(report['numeric_facts'][0]['value'],'16.059173455264322000000000')
            self.assertEqual(len([f for f in report['numeric_facts']if f.get('supporting_only')]),2)
            self.assertEqual(sum(c['stage']=='grounding'for c in report['stage_calls']),1)
            self.assertEqual(replay_graph_report(report,services=services)['grounding_records'],report['grounding_records'])
            forged=copy.deepcopy(report);forged['grounding_records']['n1']['after']['series']='forged'
            with self.assertRaisesRegex(ValueError,'grounding_records'):replay_graph_report(forged,services=services)

    def test_text_rejection_keeps_other_text_and_replays_raw_and_rejections(self):
        class IsolatedTransport(OfflineTransport):
            def __init__(self):super().__init__();self.generated=[]
            def request(self,method,path,payload=None):
                if path!='/api/chat':return super().request(method,path,payload)
                stage=next(k for k,text in m.PROMPTS.items()if payload['messages'][0]['content'].startswith(text+'\n'))
                data=stage_payload(payload,stage)
                if stage=='interpret':result=goal_response([(metric,'explanation')for metric in ['generosidad','causas de la generosidad']],data['question'],payload['format'])
                elif stage=='intent_audit':
                    needs=[dict(kind='text',metric=metric,population='aggregate',book='2025',series=None,
                        unit='unspecified',periods=[])for metric in ['generosidad','causas de la generosidad']]
                    result=named_response(needs,data['question'],payload['format'])
                elif stage=='generate':
                    self.generated.append([n['need_id']for n in data['needs']]);source=data['evidence_options'][0]
                    result=dict(claims=[dict(need_id=n['need_id'],text='Explicación de prueba con evidencia acreditada.',
                        evidence=[dict(evidence_id=source['evidence_id']if n['need_id']=='n1'else'not_offered')])for n in data['needs']])
                elif stage=='review':
                    result=dict(judgments={c['claim_id']:dict(supported=True,reason='Approved fixture')for c in data['claims']},
                        coverage=review_coverage(data['needs'],data['claims'],accepted={'n1'}))
                else:return super().request(method,path,payload)
                return dict(model=CHAT_MODEL,done=True,done_reason='stop',prompt_eval_count=64,eval_count=64,
                    message=dict(content=json.dumps(result,ensure_ascii=False)))
        services=make_services();transport=IsolatedTransport();services.client=GraphStageClient(transport=transport)
        with patch.object(m,'retrieve_documents',return_value=fixture_narrative_source(services)):
            report=m.answer_graph('Informe 2025: explica la generosidad y sus causas.',services=services)
            self.assertEqual(report['status'],'partial',report['publication'])
            self.assertEqual([c['need_id']for c in report['publication']['claims']],['n1'])
            self.assertEqual(transport.generated,[['n1','n2'],['n2']])
            self.assertEqual(len(report['text_rejections']),2)
            self.assertEqual(report['publication']['missing_parts'][0]['reason_code'],'textual_semantic_rejected')
            self.assertEqual(replay_graph_report(report,services=services)['publication'],report['publication'])
            forged=copy.deepcopy(report);forged['text_rejections'][0]['reason']='forged'
            with self.assertRaisesRegex(ValueError,'text_rejections'):replay_graph_report(forged,services=services)

    def test_four_data_needs_fit_typed_planning_without_redundant_checker_within_budget(self):
        services=copy.copy(full_services());transport=FourNeedTransport()
        services.client=GraphStageClient(transport=transport)
        with patch.object(m,'retrieve_catalog_for_needs',return_value=fixture_four_sources(services)):
            report=m.answer_graph('Informe de 2026: indica pensiones, sanidad, educación e intereses en 2030, en porcentaje del PIB.',services=services)
        (m.ROOT/'tmp/graph_v2_four_need_fixture.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        self.assertEqual(report['status'],'answered',report['publication'])
        self.assertEqual(len(report['plan']['selections']),4)
        self.assertEqual(len(report['publication']['claims']),4)
        self.assertEqual(transport.plans,{'n1':1,'n2':1,'n3':1,'n4':1})
        self.assertEqual(transport.checks,0)
        self.assertEqual(sum(c['endpoint']=='/api/chat'for c in report['stage_calls']),10)
        self.assertEqual(sum(c.get('stage')=='grounding'for c in report['stage_calls']),1)
        for call in report['stage_calls']:
            if call['endpoint']=='/api/chat':
                self.assertLessEqual(len(json.dumps(call['request'],ensure_ascii=False,separators=(',',':')).encode()),m.MAX_CHAT_REQUEST_BYTES)
        with patch.object(m,'retrieve_catalog_for_needs',return_value=fixture_four_sources(services)):
            self.assertEqual(replay_graph_report(report,services=services)['publication'],report['publication'])
            for field in ['interpretation','planning_decisions','offered_capabilities','checker_verdicts','need_outcomes','grounding_raw','grounding_records']:
                bad=copy.deepcopy(report);bad[field]={'forged':True}
                with self.subTest(field=field),self.assertRaises(ValueError):replay_graph_report(bad,services=services)

    def test_one_source_contract_repair_preserves_other_valid_needs(self):
        services=copy.copy(full_services());transport=FourNeedTransport(reject_need='n2',metrics=['Pensiones','Sanidad','Educación'])
        services.client=GraphStageClient(transport=transport)
        original_execute=m.execute_plan
        def reject_source(plan,*args,**kwargs):
            if any(s['need_id']=='n2'and s['card_ids']for s in plan['selections']):raise m.InvalidPlan('Offline protected record is unavailable')
            return original_execute(plan,*args,**kwargs)
        with patch.object(m,'retrieve_catalog_for_needs',return_value=fixture_four_sources(services)),patch.object(m,'execute_plan',side_effect=reject_source):
            report=m.answer_graph('Informe de 2026: indica pensiones, sanidad y educación en 2030, en porcentaje del PIB.',services=services)
        (m.ROOT/'tmp/graph_v2_partial_fixture.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        self.assertEqual(report['status'],'partial',report['publication'])
        self.assertEqual(len(report['publication']['claims']),2)
        self.assertEqual(transport.plans,{'n1':1,'n2':2,'n3':1})
        self.assertEqual(transport.checks,0)
        self.assertEqual(sum(c['endpoint']=='/api/chat'for c in report['stage_calls']),9)
        self.assertEqual(sum(c.get('stage')=='grounding'for c in report['stage_calls']),1)
        self.assertEqual(report['need_outcomes']['n2'],'source_contract_rejected')
        self.assertEqual(report['planning_decisions']['n2']['action'],'select')
        self.assertEqual(report['publication']['missing_parts'][0]['reason_code'],'scope_mismatch')
        with patch.object(m,'retrieve_catalog_for_needs',return_value=fixture_four_sources(services)),patch.object(m,'execute_plan',side_effect=reject_source):
            self.assertEqual(replay_graph_report(report,services=services)['publication'],report['publication'])

    def test_truncated_live_style_response_replays_same_technical_error(self):
        services=make_services();services.client.transport.truncate_plan=True
        report=m.answer_graph('¿Qué dice el informe?',services=services)
        self.assertEqual(report['status'],'error');self.assertEqual(report['publication']['claims'],[])
        self.assertIn('Incomplete or truncated',report['publication']['reason'])
        replay=replay_graph_report(report,services=services)
        self.assertEqual(report['publication'],replay['publication'])
        for field,value in [('digest_after',EMBED_DIGEST),('error','ValueError: forged')]:
            bad=copy.deepcopy(report);bad['stage_calls'][-1][field]=value
            with self.assertRaises(ValueError):replay_graph_report(bad,services=services)
        bad=copy.deepcopy(report);bad['stage_calls'][-1]['request']['model']=EMBED_MODEL
        with self.assertRaises(ValueError):replay_graph_report(bad,services=services)

    def test_source_signatures_detect_equal_length_change_with_restored_mtime(self):
        with tempfile.TemporaryDirectory(dir=m.ROOT/'tmp')as directory:
            path=Path(directory)/'data.duckdb';path.write_bytes(b'ABCDEF')
            (path.parent/'validation.json').write_text('{}',encoding='utf-8')
            services=m.GraphServices.__new__(m.GraphServices)
            services.database=path;services.catalog={'sources':{},'observations':{}}
            services.semantic_context=None
            before=services.source_signatures();stat=path.stat()
            path.write_bytes(b'FEDCBA');os.utime(path,ns=(stat.st_atime_ns,stat.st_mtime_ns))
            self.assertEqual(path.stat().st_size,stat.st_size)
            self.assertEqual(path.stat().st_mtime_ns,stat.st_mtime_ns)
            self.assertNotEqual(before,services.source_signatures())

    def test_authenticate_and_direct_manifest_reject_changed_source_with_same_stat(self):
        # Replace reads only: certified originals are never edited by this test.
        services=make_services();target=(m.ROOT/services.documents['sources'][0]['source_path']).resolve()
        original_read=Path.read_bytes;original=original_read(target);stat=target.stat()
        replaced=b'!'+original[1:]
        self.assertEqual(len(original),len(replaced))
        def changed_read(path):return replaced if path.resolve()==target else original_read(path)
        for entry in [lambda:services.authenticate(),lambda:runtime_manifest(services)]:
            with self.subTest(entry=entry),patch.object(Path,'read_bytes',autospec=True,side_effect=changed_read):
                with self.assertRaises(ValueError):entry()
            self.assertEqual(target.stat().st_size,stat.st_size)
            self.assertEqual(target.stat().st_mtime_ns,stat.st_mtime_ns)

    def test_production_numeric_question_uses_both_indices_and_exact_source_record(self):
        base=base_services();oid='airef:2025:454';record=base.catalog['observations'][oid]
        card=next(c for c in base.catalog['cards']if oid in c['observation_ids'])
        class NumericTransport(OfflineTransport):
            def request(self,method,path,payload=None):
                if path=='/api/embed':return {'model':EMBED_MODEL,'embeddings':[[0.,1.]+[0.]*(EMBED_DIMENSION-2)for _ in payload['input']]}
                if path!='/api/chat':return super().request(method,path,payload)
                stage=next(key for key,text in m.PROMPTS.items()if payload['messages'][0]['content'].startswith(text+'\n'))
                if stage=='plan_check_batch':
                    data=stage_payload(payload,stage)
                    self_test.assertNotIn('candidates',data)
                    shown=[fact for row in data['needs'].values()for fact in row['results']]
                    self_test.assertEqual({c['card_id']for c in shown},{card['card_id']})
                    self_test.assertEqual(len(shown),1)
                    for field in ['observation_id','book','metric','series','period','value','unit','scope','source_sheet','source_cell']:
                        self_test.assertEqual(shown[0][field],record.get(field))
                    self_test.assertTrue(all(set(row)=={'action','operation','results'}for row in data['needs'].values()))
                if stage not in{'interpret','intent_audit','plan_need'}:return super().request(method,path,payload)
                data=stage_payload(payload,stage)
                if stage=='interpret':result=goal_response([(record['metric'],'quantity')],data['question'],payload['format'],book=record['book'])
                elif stage=='intent_audit':
                    need=dict(kind='data',metric=record['metric'],population='aggregate',book=record['book'],
                              series=record['series'],unit=record['unit'],periods=[record['period']])
                    result=named_response([need],data['question'],payload['format'])
                else:
                    if card['card_id']not in{source['card_id']for cap in data['cap_options']for source in cap['source_metadata']}:raise AssertionError('Known exact source not offered')
                    choice=next(c for c in data['cap_options']if [source['card_id']for source in c['source_metadata']]==[card['card_id']])
                    result=dict(action='select',capability_id=choice['capability_id'])
                return {'model':CHAT_MODEL,'done':True,'done_reason':'stop','message':{'content':json.dumps(result)},'prompt_eval_count':64,'eval_count':64}
        self_test=self
        docs=build_document_corpus();v=[1.]+[0.]*(EMBED_DIMENSION-1)
        docs=attach_document_embeddings(docs,[v]*len(docs['chunks']))
        vectors=[v]*len(base.catalog['cards'])
        vectors[next(i for i,c in enumerate(base.catalog['cards'])if c['card_id']==card['card_id'])]=[0.,1.]+[0.]*(EMBED_DIMENSION-2)
        catalog=attach_embeddings(base.catalog,vectors)
        services=m.GraphServices(catalog,docs,GraphStageClient(transport=NumericTransport()))
        report=m.answer_graph('¿Cuál es el gasto previsto en pensiones en 2050 de acuerdo con el informe de 2025?',services=services)
        self.assertEqual(report['status'],'answered',report['publication'])
        self.assertFalse(report['manifest']['reference_text_only'])
        self.assertEqual(report['manifest']['source_policy']['sha256'],m.POLICY_SHA256)
        self.assertTrue(report['retrieval']['catalog']);self.assertTrue(report['retrieval']['documents'])
        self.assertGreater(len(report['retrieval']['catalog']),1)
        self.assertEqual(report['plan']['selections'][0]['reason'],'Compiled from certified source metadata')
        fact=report['numeric_facts'][0]
        for field in ['observation_id','value','metric','series','period','unit','source_cell','source_sha256']:
            self.assertEqual(fact[field],record[field])
        support=[f for f in report['numeric_facts']if f.get('supporting_only')]
        self.assertEqual(len(support),2)
        self.assertEqual(len(report['publication']['claims']),1)
        self.assertEqual(len(report['publication']['claims'][0]['evidence']),3)
        self.assertIn('Discrepancia no explicada ni conciliada',report['publication']['claims'][0]['text'])
        self.assertEqual(replay_graph_report(report,services=services)['numeric_facts'],report['numeric_facts'])

    def test_policy_asset_drift_blocks_checkpoint_resume(self):
        services=make_services(production=True);services.client.transport.clarify_once=True
        with tempfile.TemporaryDirectory(dir=m.ROOT/'tmp')as folder,patch.object(m,'retrieve_documents',return_value=fixture_narrative_source(services)):
            path=Path(folder)/'policy_state.sqlite3';query='¿Qué dice el informe sobre la generosidad?'
            first=m.answer_graph(query,services=services,thread_id='policy_drift',checkpoint_path=path)
            self.assertEqual(first['status'],'clarification')
            original_read=Path.read_bytes;target=m.POLICY_PATH.resolve();raw=original_read(target)
            def changed_read(source):return b'!'+raw[1:]if source.resolve()==target else original_read(source)
            with patch.object(Path,'read_bytes',autospec=True,side_effect=changed_read):
                with self.assertRaisesRegex(ValueError,'policy bytes changed'):
                    m.answer_graph(query,services=services,thread_id='policy_drift',checkpoint_path=path,resume='La edición 2025')
            resumed=m.answer_graph(query,services=services,thread_id='policy_drift',checkpoint_path=path,resume='La edición 2025')
            self.assertEqual(resumed['status'],'answered')

    def test_planner_payload_is_packed_with_explicit_omission_count(self):
        services=make_services()
        candidates=[{'card_id':str(i),'metric':'descripción '*200+str(i)}for i in range(10)]
        m._chat(services,'plan',{'query':'Gasto','interpretation':{'needs':[]},'candidates':candidates},m.PLAN_SCHEMA)
        request=services.client.calls[-1]['request'];data=json.loads(request['messages'][1]['content'])
        self.assertIn(json.dumps(m.PLAN_SCHEMA,ensure_ascii=False,separators=(',',':')),request['messages'][0]['content'])
        self.assertGreater(data['omitted_candidate_count'],0)
        self.assertEqual(len(m.unpack_candidate_table(data['candidates']))+data['omitted_candidate_count'],10)
        self.assertLessEqual(len(json.dumps(request,ensure_ascii=False,separators=(',',':')).encode()),m.MAX_CHAT_REQUEST_BYTES)

    def test_retrieval_branches_run_concurrently_before_join(self):
        original=m._safe;barrier=threading.Barrier(2,timeout=10);visited=[]
        def wrapped(fn):
            safe=original(fn)
            if fn.__name__ not in {'catalog','documents'}:return safe
            def node(state):
                visited.append(fn.__name__);barrier.wait();return safe(state)
            return node
        with patch.object(m,'_safe',side_effect=wrapped):
            report=m.answer_graph('¿A qué año se aproxima la generosidad?',services=make_services())
        self.assertEqual(report['status'],'answered',report['publication'])
        self.assertEqual(set(visited),{'catalog','documents'})

    def test_real_graph_reference_has_fanout_join_trace_and_verified_publication(self):
        services=make_services()
        report=m.answer_graph('¿A qué año histórico se aproxima el indicador agregado de generosidad?',services=services)
        self.assertEqual(report['status'],'answered',report['publication'])
        self.assertIn('2015',report['formatted_answer'])
        self.assertEqual(report['retrieval']['catalog'],[])
        self.assertTrue(report['manifest']['reference_text_only'])
        self.assertEqual(report['manifest']['exposed_card_ids'],[])
        self.assertEqual(sum(c['endpoint']=='/api/embed'for c in report['stage_calls']),1)
        self.assertTrue(all(c['model']in{CHAT_MODEL,EMBED_MODEL}for c in report['stage_calls']))
        self.assertEqual(replay_graph_report(report,services=services)['publication'],report['publication'])

    def test_checkpoint_clarification_resume_and_completed_resume_rejected(self):
        services=make_services(production=True);services.client.transport.clarify_once=True
        with tempfile.TemporaryDirectory(dir=m.ROOT/'tmp')as folder,patch.object(m,'retrieve_documents',return_value=fixture_narrative_source(services)):
            path=Path(folder)/'state.sqlite3';query='¿Qué dice el informe sobre la generosidad?'
            first=m.answer_graph(query,services=services,thread_id='clarification_test',checkpoint_path=path)
            self.assertEqual(first['status'],'clarification',first)
            second=m.answer_graph(query,services=services,thread_id='clarification_test',checkpoint_path=path,resume='La edición 2025')
            self.assertEqual(second['status'],'answered',second['publication'])
            self.assertTrue(second['history_stage_calls'])
            self.assertEqual(replay_graph_report(second,services=services)['publication'],second['publication'])
            with self.assertRaisesRegex(ValueError,'Checkpoint'):
                m.answer_graph(query,services=services,thread_id='clarification_test',checkpoint_path=path,resume='another')
            with self.assertRaisesRegex(ValueError,'already exists'):
                m.answer_graph('Otra pregunta',services=services,thread_id='clarification_test',checkpoint_path=path)

    def test_resume_discards_rehashed_checkpoint_numeric_and_publication_values(self):
        services=make_services(production=True);services.client.transport.clarify_once=True
        with tempfile.TemporaryDirectory(dir=m.ROOT/'tmp')as folder,patch.object(m,'retrieve_documents',return_value=fixture_narrative_source(services)):
            path=Path(folder)/'state.sqlite3';query='¿Qué dice el informe sobre la generosidad?'
            first=m.answer_graph(query,services=services,thread_id='tampered',checkpoint_path=path)
            self.assertEqual(first['status'],'clarification')
            serde=m.StrictJSONSerializer()
            connection=sqlite3.connect(path)
            try:
                row=connection.execute('SELECT checkpoint_id,type,checkpoint FROM checkpoints ORDER BY checkpoint_id DESC LIMIT 1').fetchone()
                state=serde.loads_typed((row[1],row[2]))
                state['channel_values']['numeric_facts']=[{'value':'999999','need_id':'N1'}]
                state['channel_values']['publication']={'status':'answered','claims':[{'text':'FORGED999999'}]}
                encoding,raw=serde.dumps_typed(state)
                connection.execute('UPDATE checkpoints SET type=?,checkpoint=? WHERE checkpoint_id=?',(encoding,raw,row[0]))
                connection.commit()
            finally:connection.close()
            report=m.answer_graph(query,services=services,thread_id='tampered',checkpoint_path=path,resume='La edición 2025')
            self.assertEqual(report['status'],'answered',report['publication'])
            self.assertEqual(report['numeric_facts'],[])
            self.assertNotIn('999999',report['formatted_answer'])

    def test_text_needs_do_not_expose_free_numeric_source_choices(self):
        services=make_services();services.client.transport.bad_id=True
        report=m.answer_graph('¿Qué dice el informe?',services=services)
        self.assertEqual(report['status'],'answered')
        self.assertEqual(report['numeric_facts'],[])
        self.assertEqual(services.client.transport.plans,0)

    def test_wrong_digest_swaps_and_publication_tampering_rejected(self):
        services=make_services();report=m.answer_graph('¿A qué año se aproxima la generosidad?',services=services)
        bad=copy.deepcopy(report)
        next(c for c in bad['stage_calls']if c['endpoint']=='/api/chat')['digest_before']=EMBED_DIGEST
        with self.assertRaises(ValueError):replay_graph_report(bad,services=services)
        bad=copy.deepcopy(report);bad['publication']['claims'][0]['text']='Año9999'
        with self.assertRaisesRegex(ValueError,'publication'):replay_graph_report(bad,services=services)
        bad=copy.deepcopy(report);bad['stage_calls'][0]['stage']='tampered_stage'
        with self.assertRaisesRegex(ValueError,'stage'):replay_graph_report(bad,services=services)

    def test_external_tracing_flags_are_disabled_at_api_entry(self):
        services=make_services()
        with patch.dict(os.environ,{'LANGSMITH_TRACING':'true','LANGCHAIN_TRACING_V2':'true','LANGSMITH_API_KEY':'DUMMY_UNIT_TEST'}), \
             patch('urllib.request.urlopen',side_effect=AssertionError('No external request')):
            report=m.answer_graph('¿A qué año se aproxima la generosidad?',services=services)
            self.assertEqual(report['status'],'answered',report['publication'])
            self.assertEqual(os.environ['LANGSMITH_TRACING'],'false')
            self.assertEqual(os.environ['LANGCHAIN_TRACING_V2'],'false')


if __name__=='__main__':unittest.main()
