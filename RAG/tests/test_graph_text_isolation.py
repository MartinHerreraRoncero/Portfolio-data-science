"""Generated-content failures remain local; protocol/integrity failures remain fatal."""
import copy
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from rag_pensiones import graph_pipeline as p
from rag_pensiones import graph_validation as v


def need(nid):return dict(need_id=nid,kind='text',book='2026',query_span=nid)


DOCUMENT=dict(source_id='s',book='2026',text='Un pasaje publicado con evidencia literal suficiente.',
    doc_id='doc',page_number=1,source_file='synthetic.pdf',source_path='synthetic.pdf',source_sha256='hash')


def claim(nid,quote=None):
    return dict(need_id=nid,text='Una explicación acreditada.',evidence=[dict(source_id='s',quote=quote or DOCUMENT['text'])])


class TextIsolationTests(unittest.TestCase):
    def test_user_followup_scope_reaches_real_generate_review_wire_without_becoming_evidence(self):
        from rag_pensiones.graph_models import GraphStageClient
        from rag_pensiones.graph_types import CHAT_MODEL,CHAT_DIGEST
        from rag_pensiones.graph_wire import decode_review_payload
        context=[dict(role='user',content='Sólo Cataluña, en 2050.'),dict(role='assistant',content='España.')]
        original=copy.deepcopy(context);seen={}
        class Offline:
            def request(self,method,path,payload=None):
                if path=='/api/tags':return dict(models=[dict(name=CHAT_MODEL,digest=CHAT_DIGEST)])
                stage=next(k for k in ['generate','review']if payload['messages'][0]['content'].startswith(p.PROMPTS[k]+'\n'))
                data=json.loads(payload['messages'][1]['content'])
                if stage=='review':data=decode_review_payload(data)
                seen[stage]=data
                if stage=='generate':
                    offered=data['evidence_options'][0]['evidence_id']
                    self_schema=payload['format']['properties']['claims']['items']['properties']['evidence']['items']['properties']['evidence_id']
                    assert 'unoffered'not in self_schema['enum']
                    result=dict(claims=[dict(need_id='t1',text='Una explicación acreditada.',evidence=[dict(evidence_id=offered)]),
                        dict(need_id='t1',text='Otra afirmación no acreditada.',evidence=[dict(evidence_id='unoffered')])])
                else:result=dict(judgments={c['claim_id']:dict(supported=True,reason='Mock entailment only')for c in data['claims']},
                    coverage=dict(t1=dict(requested_result_present=True,requested_scope_supported=False,claim_ids=[])))
                return dict(model=CHAT_MODEL,done=True,done_reason='stop',prompt_eval_count=64,eval_count=64,
                            message=dict(content=json.dumps(result,ensure_ascii=False)))
        services=SimpleNamespace(client=GraphStageClient(transport=Offline()));graph=p.build_graph(services)
        state=dict(query='¿Por qué cambió?',effective_context=context,documents=[copy.deepcopy(DOCUMENT)],repairs=1,
            interpretation=dict(needs=[dict(need('t1'),query_span='¿Por qué cambió?',metric='causa del cambio')]))
        state.update(graph.nodes['generate'].invoke(state));state.update(graph.nodes['review'].invoke(state))
        for stage,data in seen.items():
            self.assertEqual(data['context'],original);self.assertEqual(data['query'],state['query'])
            if stage=='review':self.assertEqual(data['passages'],[{k:DOCUMENT[k]for k in ['source_id','book','text']}])
            else:
                self.assertNotIn('passages',data)
                self.assertEqual(data['evidence_options'][0]['quote'],DOCUMENT['text'])
                self.assertNotIn('España',data['evidence_options'][0]['quote'])
        self.assertEqual(set(seen),{'generate','review'});self.assertEqual(context,original)
        self.assertEqual(state['documents'],[DOCUMENT])
        self.assertFalse(state['text_coverage'][0]['answered']) # mock refuses unknown scope; no semantic certification
        self.assertEqual(len(state['text_claims']),1)
        self.assertTrue(any(r.get('code')=='unknown_evidence_id'for r in state['text_rejections']))
        for call in services.client.calls:
            self.assertLessEqual(len(json.dumps(call['request'],ensure_ascii=False,separators=(',',':')).encode()),6400)

    def test_contradictory_coverage_is_audited_locally_and_preserves_numeric_and_text_siblings(self):
        nodes={};original=p._safe;services=SimpleNamespace(client=SimpleNamespace(calls=[]))
        def capture(fn):nodes[fn.__name__]=original(fn);return nodes[fn.__name__]
        with patch.object(p,'_safe',side_effect=capture):graph=p.build_graph(services)
        claims=[dict(claim(nid),claim_id=nid+'C1')for nid in ['t1','t2']]
        raw=dict(judgments={c['claim_id']:dict(supported=c['need_id']=='t2',reason='Fixture entailment')for c in claims},
            coverage={c['need_id']:dict(requested_result_present=True,requested_scope_supported=True,
                claim_ids=[c['claim_id']])for c in claims})
        before=copy.deepcopy(raw)
        state=dict(query='Datos y dos explicaciones',interpretation=dict(needs=[dict(need_id='d',kind='data',query_span='dato'),need('t1'),need('t2')]),
            text_claims=claims,documents=[DOCUMENT],active_text_need_ids=['t1','t2'],numeric_facts=[dict(need_id='d',text='Numeric fact',evidence=[])],
            repairs=0,plan_valid=True)
        with patch.object(p,'_chat',return_value=raw):result=nodes['review'](state)
        self.assertNotIn('errors',result);self.assertTrue(result['text_missing']);self.assertEqual(result['repair_text_need_id'],'t1')
        self.assertTrue(any(r.get('code')=='review_coverage_contradiction'for r in result['text_rejections']))
        self.assertEqual(raw,before);state.update(result)
        with patch.object(p,'numeric_claim',side_effect=lambda fact:fact):publication=graph.nodes['publish'].invoke(state)['publication']
        self.assertEqual(publication['status'],'partial');self.assertEqual({c['need_id']for c in publication['claims']},{'d','t2'})

    def nodes(self):
        result={};original=p._safe
        def capture(fn):result[fn.__name__]=original(fn);return result[fn.__name__]
        with patch.object(p,'_safe',side_effect=capture):p.build_graph(SimpleNamespace())
        return result

    def test_content_and_foreign_need_are_local_but_schema_is_fatal(self):
        original=[claim('t1'),claim('t2','Una cita que no existe en este documento.')]
        before=copy.deepcopy(original)
        accepted,rejected=v.validate_text_claims_isolated(original,[need('t1'),need('t2')],[DOCUMENT])
        self.assertEqual([c['need_id']for c in accepted],['t1'])
        self.assertEqual(rejected[0]['need_id'],'t2');self.assertEqual(original,before)
        accepted,rejected=v.validate_text_claims_isolated([claim('alien')],[need('t1')],[DOCUMENT])
        self.assertEqual(accepted,[]);self.assertEqual(rejected[0]['code'],'unknown_need_id')
        with self.assertRaises(v.InvalidPlan):v.check_shape({'claims':'bad'},p.GENERATE_SCHEMA)
        with self.assertRaises(KeyError):v.validate_text_claims_isolated([dict(need_id='t1')],[need('t1')],[DOCUMENT])
        for error in [OSError('HTTP failed'),ValueError('Source SHA changed'),v.InvalidPlan('JSON schema invalid')]:
            def broken(state):raise error
            self.assertIn(type(error).__name__,p._safe(broken)({})['errors'][0])

    def test_numeric_and_other_text_survive_one_local_retry_and_budget_is_respected(self):
        nodes={};original=p._safe;services=SimpleNamespace(client=SimpleNamespace(calls=[]))
        def capture(fn):nodes[fn.__name__]=original(fn);return nodes[fn.__name__]
        with patch.object(p,'_safe',side_effect=capture):p.build_graph(services)
        def chat(s,stage,payload,schema):
            s.client.calls.append(dict(endpoint='/api/chat'))
            if stage=='generate':
                return dict(claims=[dict(need_id=n['need_id'],text='Explicación válida'if n['need_id']=='t1'else'Conclusión ajena a la cita.',evidence=[dict(evidence_id=payload['evidence_options'][0]['evidence_id'])])for n in payload['needs']])
            return dict(judgments={c['claim_id']:dict(supported=c['need_id']=='t1',reason='Cita exacta, pero sólo t1 está implicada por ella')for c in payload['claims']},
                coverage={n['need_id']:dict(requested_result_present=n['need_id']=='t1',
                    requested_scope_supported=n['need_id']=='t1',
                    claim_ids=[c['claim_id']for c in payload['claims']if c['need_id']==n['need_id']=='t1'])for n in payload['needs']})
        state=dict(query='Consulta mixta',interpretation=dict(needs=[dict(need_id='d',kind='data',query_span='dato'),need('t1'),need('t2')]),
            documents=[DOCUMENT],numeric_facts=[dict(need_id='d',text='12.5',evidence=[])],repairs=0,plan_valid=True)
        with patch.object(p,'_chat',side_effect=chat):
            state.update(nodes['generate'](state));state.update(nodes['review'](state))
            self.assertTrue(state['text_missing']);self.assertEqual(state['repair_text_need_id'],'t2')
            approved=copy.deepcopy(state['text_claims'][0]);state.update(nodes['prepare_repair'](state))
            state.update(nodes['generate'](state));self.assertEqual(state['active_text_need_ids'],['t2'])
            state.update(nodes['review'](state))
        self.assertFalse(state['text_missing']);self.assertEqual(state['text_claims'],[approved])
        self.assertEqual(len(state['text_rejections']),2)
        self.assertEqual(len(services.client.calls),4)
        # The public node is intentionally unwrapped; use its registered runnable.
        graph=p.build_graph(services)
        with patch.object(p,'numeric_claim',side_effect=lambda fact:fact):
            publication=graph.nodes['publish'].invoke(state)['publication']
        self.assertEqual(publication['status'],'partial')
        self.assertEqual({c['need_id']for c in publication['claims']},{'d','t1'})
        self.assertEqual(publication['missing_parts'][0]['reason_code'],'textual_semantic_rejected')
        state['repairs']=0;state['active_text_need_ids']=['t2'];services.client.calls=[dict(endpoint='/api/chat')]*9
        with patch.object(p,'_chat',side_effect=chat):state.update(nodes['review'](state))
        self.assertFalse(state['text_missing']) # one chat left cannot fund generate + review


if __name__=='__main__':unittest.main()
