import copy
import json
import threading
import time
import unittest
from unittest.mock import patch, MagicMock
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from rag_pensiones.graph_models import GraphStageClient, GraphEmbedderAdapter, LocalOllamaTransport, RejectRedirects,structural_response_schema
from rag_pensiones.graph_types import CHAT_MODEL, CHAT_DIGEST, CHAT_OPTIONS, EMBED_MODEL, EMBED_DIGEST, EMBED_DIMENSION,EMBED_KEEP_ALIVE,stage_chat_settings,MAX_THINKING_OUTPUT_TOKENS

SCHEMA = {'type': 'object', 'properties': {'ok': {'type': 'boolean'}}, 'required': ['ok']}
MESSAGES = [{'role': 'user', 'content': 'Control sintético'}]


class FakeTransport:
    def __init__(self):
        self.requests = []
        self.tags = [{'name': CHAT_MODEL, 'digest': CHAT_DIGEST}, {'name': EMBED_MODEL, 'digest': EMBED_DIGEST}]
        self.reply_model = None
        self.failure = None
        self.content = '{"ok":true}'
        self.active = self.max_active = 0
        self.lock = threading.Lock()

    def request(self, method, path, payload=None):
        self.requests.append((method, path, copy.deepcopy(payload)))
        if path == '/api/tags':
            return {'models': copy.deepcopy(self.tags)}
        if self.failure:
            raise self.failure
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        time.sleep(.01)
        with self.lock:
            self.active -= 1
        if path == '/api/embed':
            return {'model': self.reply_model or EMBED_MODEL, 'embeddings': [[1.] + [0.] * (EMBED_DIMENSION - 1) for _ in payload['input']]}
        return {'model': self.reply_model or CHAT_MODEL, 'done': True,
                'done_reason': 'stop', 'message': {'content': self.content}, 'eval_count': 4,'prompt_eval_count':32}


class GraphModelTests(unittest.TestCase):
    def test_reasoning_can_finish_after_old_output_limit_without_growing_context(self):
        transport=FakeTransport();original=transport.request
        def request(method,path,payload=None):
            result=original(method,path,payload)
            if path=='/api/chat':
                result['eval_count']=20000
                result['message']['thinking']='Synthetic reasoning budget test'
            return result
        transport.request=request;client=GraphStageClient(transport=transport)
        self.assertEqual(client.chat(MESSAGES,SCHEMA,'interpret'),dict(ok=True))
        options=client.calls[0]['request']['options']
        self.assertEqual(options['num_ctx'],32768)
        self.assertEqual(options['num_predict'],MAX_THINKING_OUTPUT_TOKENS)
        self.assertEqual(CHAT_OPTIONS['num_predict'],8192)

    def test_dense_input_preserves_minimum_output_and_rejects_context_overflow(self):
        from rag_pensiones.graph_models import prompt_context_bound
        messages=[dict(role='user',content='x'*20000)]
        settings=stage_chat_settings('review',messages)
        self.assertGreaterEqual(settings['options']['num_predict'],8192)
        self.assertLess(settings['options']['num_predict'],MAX_THINKING_OUTPUT_TOKENS)
        self.assertLessEqual(prompt_context_bound(messages,settings['options']),32768)
        transport=FakeTransport();client=GraphStageClient(transport=transport)
        with patch.dict(CHAT_OPTIONS,num_ctx=8192,num_predict=2048):
            with self.assertRaisesRegex(ValueError,'context bound'):
                client.chat([dict(role='user',content='x'*6100)],SCHEMA,'review')
        self.assertEqual(transport.requests,[])

    def test_quality_adapter_requires_exact_context_derived_wire_budget(self):
        from experiments import graph_quality as q
        client=GraphStageClient(transport=FakeTransport())
        client.chat([dict(role='user',content='x'*6000)],SCHEMA,'review')
        q._validate_calls(dict(stage_calls=client.calls))
        changed=copy.deepcopy(client.calls)
        changed[0]['request']['options']['num_predict']+=1
        with self.assertRaisesRegex(ValueError,'stage options'):
            q._validate_calls(dict(stage_calls=changed))

    def test_active_chat_identity_is_the_user_selected_q8_model(self):
        self.assertEqual(CHAT_MODEL,'qwen3.5:9b-q8_0')
        self.assertEqual(CHAT_DIGEST,'a8ee95bc3592ebaa624014ca8b429d25e0ef5f45030dc35786da7ff7e9e1b60a')

    def test_lifecycle_options_are_copied_and_embedding_is_cpu_bounded(self):
        from rag_pensiones.graph_types import EMBED_OPTIONS,model_lifecycle_policy
        policy=model_lifecycle_policy();self.assertEqual(policy['embedding_options'],{'num_gpu':0,'num_ctx':8192,'num_batch':128})
        policy['embedding_options']['num_gpu']=99;self.assertEqual(EMBED_OPTIONS['num_gpu'],0)
        client=GraphStageClient(transport=FakeTransport());client.embed(['Synthetic'])
        self.assertEqual(client.calls[-1]['request']['options'],EMBED_OPTIONS)
        self.assertEqual(client.calls[-1]['request']['keep_alive'],-1)

    def test_stage_lifecycle_keeps_chat_gpu_and_embedding_cpu_resident_with_identical_wire_prediction(self):
        from rag_pensiones import graph_pipeline as p
        transport=FakeTransport();client=GraphStageClient(transport=transport)
        data=dict(query='Identical synthetic input');before=copy.deepcopy(data)
        for stage in ['interpret','intent_audit','grounding']:
            client.chat(p._model_messages(stage,data,SCHEMA),SCHEMA,stage)
            request=client.calls[-1]['request']
            self.assertEqual(p._wire_size(stage,data,SCHEMA),len(json.dumps(request,ensure_ascii=False,separators=(',',':')).encode()))
            self.assertEqual(request['options'],stage_chat_settings(stage,request['messages'])['options']);self.assertFalse(request['think'])
            if stage=='intent_audit':
                vectors=client.embed(['Identical synthetic input'])
                self.assertEqual(vectors,[[1.]+[0.]*(EMBED_DIMENSION-1)])
        self.assertEqual([call['request']['keep_alive']for call in client.calls],[-1,-1,-1,-1])
        for call in client.calls:
            if call['endpoint']=='/api/embed':self.assertEqual(call['request']['options']['num_gpu'],0)
            else:self.assertNotIn('num_gpu',call['request']['options'])
            self.assertEqual(call['request']['options']['num_ctx'],8192 if call['endpoint']=='/api/embed'else 32768)
        self.assertEqual([call['endpoint']for call in client.calls],['/api/chat','/api/chat','/api/embed','/api/chat'])
        for call in client.calls:
            expected=EMBED_DIGEST if call['endpoint']=='/api/embed'else CHAT_DIGEST
            self.assertEqual((call['digest_before'],call['digest_after']),(expected,expected))
        self.assertEqual(data,before)

    def test_structural_format_visits_schema_nodes_and_preserves_literal_description_keys(self):
        from rag_pensiones.graph_validation import check_shape,InvalidPlan
        literal={'description':'literal instance'}
        schema=dict(type='object',description='root documentation',required=['description','choice'],additionalProperties=False,
            properties=dict(description=dict(type='string',description='property documentation',minLength=1,maxLength=8),
                choice=dict(type='object',description='choice documentation',enum=[literal],const=literal)),
            oneOf=[dict(type='object',description='branch documentation')],
            **{'$defs':{'description':dict(type='string',description='definition documentation')},
               'x-literal':literal})
        before=copy.deepcopy(schema);structural=structural_response_schema(schema)
        self.assertEqual(schema,before)
        self.assertNotIn('description',structural)
        self.assertNotIn('description',structural['properties']['description'])
        self.assertIn('description',structural['properties'])
        self.assertEqual(structural['properties']['choice']['enum'],[literal])
        self.assertEqual(structural['properties']['choice']['const'],literal)
        self.assertEqual(structural['x-literal'],literal)
        self.assertEqual(structural['$defs']['description'],{'type':'string'})
        self.assertNotIn('description',structural['oneOf'][0])
        # Test ordinary local validation separately from the branch-only schema.
        for candidate in [dict(description='valid',choice=literal),dict(description='',choice=literal),
                          dict(description='valid',choice={'description':'wrong'}),dict(choice=literal)]:
            outcomes=[]
            for version in [schema,structural]:
                version=copy.deepcopy(version);version.pop('oneOf')
                try:check_shape(candidate,version);outcomes.append(True)
                except InvalidPlan:outcomes.append(False)
            self.assertEqual(outcomes[0],outcomes[1])
        structural['properties']['description']['maxLength']=100
        self.assertEqual(schema,before)

    def test_real_format_and_predicted_wire_use_same_structural_copy(self):
        from rag_pensiones import graph_pipeline as p
        schema=copy.deepcopy(SCHEMA);schema['properties']['ok']['description']='Complete field instructions'
        data=dict(query='Synthetic bounded query')
        transport=FakeTransport();client=GraphStageClient(transport=transport)
        messages=p._model_messages('plan_need',data,schema)
        client.chat(messages,schema,'plan_need')
        request=client.calls[-1]['request']
        self.assertIn('Complete field instructions',request['messages'][0]['content'])
        self.assertNotIn('description',request['format']['properties']['ok'])
        self.assertEqual(p._wire_size('plan_need',data,schema),len(json.dumps(request,ensure_ascii=False,separators=(',',':')).encode()))

    def test_full_wire_byte_budget_before_network_and_embedding_per_input(self):
        transport=FakeTransport();client=GraphStageClient(transport=transport)
        with patch.dict(CHAT_OPTIONS,num_ctx=8192,num_predict=2048):
            with self.assertRaisesRegex(ValueError,'context bound'):
                client.chat([{'role':'user','content':'á'*4000}],SCHEMA,'plan')
        with self.assertRaisesRegex(ValueError,'byte budget'):
            client.chat([{'role':'user','content':'x'*17000}],SCHEMA,'plan')
        self.assertEqual(transport.requests,[])
        # Two independently embedded inputs may exceed the single-input budget.
        self.assertEqual(len(client.embed(['x'*4000,'y'*4000])),2)
        count=len(transport.requests)
        with self.assertRaisesRegex(ValueError,'byte budget'):
            client.embed(['x'*6401])
        self.assertEqual(len(transport.requests),count)

    def test_model_token_statistics_and_stop_reason_are_real_and_strict(self):
        for key,bad in [('eval_count',True),('prompt_eval_count',float('inf')),
                        ('eval_count',MAX_THINKING_OUTPUT_TOKENS+1),('prompt_eval_count',32768),('done_reason','length')]:
            with self.subTest(key=key,bad=bad):
                transport=FakeTransport();original=transport.request
                def request(method,path,payload=None):
                    response=original(method,path,payload)
                    if path=='/api/chat':response[key]=bad
                    return response
                transport.request=request
                client=GraphStageClient(transport=transport)
                with self.assertRaises(ValueError):client.chat(MESSAGES,SCHEMA,'verify')
                self.assertEqual(client.calls[0]['response'][key],bad)
                self.assertTrue(client.calls[0]['error'])

    def test_local_transport_ignores_proxy_environment_and_global_opener(self):
        with patch.dict('os.environ', {'HTTP_PROXY':'http://external.invalid:9999', 'HTTPS_PROXY':'http://external.invalid:9999'}):
            transport = LocalOllamaTransport()
        # Empty ProxyHandler does not install a proxy handler into the opener.
        self.assertFalse(any(isinstance(h, urllib.request.ProxyHandler) for h in transport.opener.handlers))
        response = MagicMock()
        response.__enter__.return_value.read.return_value = b'{"models":[]}'
        with patch.object(transport.opener, 'open', return_value=response) as local_open, \
             patch('urllib.request.urlopen', side_effect=AssertionError('Global opener forbidden')) as global_open:
            self.assertEqual(transport.request('GET', '/api/tags'), {'models':[]})
            self.assertEqual(transport.request('POST', '/api/chat', {'model':CHAT_MODEL}), {'models':[]})
            self.assertEqual(local_open.call_count, 2)
            self.assertIsNone(transport.timeout)
            self.assertTrue(all(call.kwargs['timeout'] is None for call in local_open.call_args_list))
            global_open.assert_not_called()

    def test_all_redirects_rejected_without_following_get_or_post(self):
        redirect = RejectRedirects()
        for method in ['GET', 'POST']:
            for code in [301,302,303,307,308]:
                with self.subTest(method=method, code=code), patch('urllib.request.urlopen') as network:
                    req=urllib.request.Request('http://127.0.0.1:11434/api/tags', method=method)
                    with self.assertRaisesRegex(ValueError, 'redirects'):
                        redirect.redirect_request(req,None,code,'redirect',{},'https://external.invalid/')
                    network.assert_not_called()

    def test_embedding_adapter_has_4b_identity_and_shared_calls(self):
        client = GraphStageClient(transport=FakeTransport())
        adapter = GraphEmbedderAdapter(client)
        self.assertEqual(adapter.get_digest(), EMBED_DIGEST)
        self.assertEqual(len(adapter.embed_texts(['texto'])[0]), EMBED_DIMENSION)
        self.assertEqual(len(adapter.calls), 1)
        self.assertEqual(adapter.calls[0]['model'], EMBED_MODEL)
        request=adapter.calls[0]['request']
        self.assertEqual(request['keep_alive'],EMBED_KEEP_ALIVE)
        self.assertEqual(request['keep_alive'],-1)
        self.assertEqual(request['input'],['texto'])
        self.assertEqual(request['dimensions'],EMBED_DIMENSION);self.assertFalse(request['truncate'])
        self.assertEqual(adapter.calls[0]['digest_before'],EMBED_DIGEST)
        self.assertEqual(adapter.calls[0]['digest_after'],EMBED_DIGEST)

    def test_27b_and_remote_endpoint_rejected_before_network(self):
        transport = FakeTransport()
        with self.assertRaises(ValueError):
            GraphStageClient(model='hf.co/gguf/Qwen3.8-27B-GGUF:Q3_K_M', transport=transport)
        self.assertEqual(transport.requests, [])
        for url in ['https://example.com', 'http://example.com:11434', 'http://token@localhost:11434', 'http://localhost:11434/alternate']:
            with self.assertRaises(ValueError):
                LocalOllamaTransport(url)

    def test_real_options_and_trace_identity(self):
        transport = FakeTransport()
        client = GraphStageClient(transport=transport)
        self.assertEqual(client.chat(MESSAGES, SCHEMA, 'plan'), {'ok': True})
        call = client.calls[0]
        self.assertEqual(call['request']['options'], stage_chat_settings('plan',MESSAGES)['options'])
        self.assertEqual(call['request']['keep_alive'],-1)
        self.assertEqual(call['request']['options']['presence_penalty'],1.5)
        self.assertEqual(call['request']['options']['temperature'],0.7)
        self.assertEqual(call['request']['options']['top_p'],0.8)
        self.assertEqual(call['request']['options']['top_k'],20)
        self.assertEqual(call['request']['options']['min_p'],0.0)
        self.assertEqual(call['request']['options']['repeat_penalty'],1.0)
        self.assertFalse(call['request']['think'])
        self.assertEqual(call['digest_before'], CHAT_DIGEST)
        self.assertEqual(call['digest_after'], CHAT_DIGEST)
        self.assertEqual(call['response']['eval_count'], 4)
        self.assertEqual(call['parsed_response'], {'ok': True})
        self.assertIsNone(call['error'])

    def test_uniform_default_settings_preserve_truncated_raw_response(self):
        transport=FakeTransport();client=GraphStageClient(transport=transport)
        original=transport.request
        def request(method,path,payload=None):
            result=original(method,path,payload)
            if path=='/api/chat':
                result.update(done_reason='length',eval_count=1024)
                result['message']['thinking']='Synthetic incomplete analysis'
            return result
        transport.request=request
        with self.assertRaisesRegex(ValueError,'truncated'):
            client.chat(MESSAGES,SCHEMA,'plan_check_batch')
        call=client.calls[0]
        self.assertFalse(call['request']['think'])
        self.assertEqual(call['request']['options']['num_predict'],MAX_THINKING_OUTPUT_TOKENS)
        self.assertEqual(call['request']['options']['presence_penalty'],1.5)
        self.assertEqual(call['response']['message']['thinking'],'Synthetic incomplete analysis')
        self.assertEqual(call['response']['eval_count'],1024)
        self.assertIsNone(call.get('parsed_response'))
        self.assertIn('truncated',call['error'])

    def test_missing_digest_blocks_inference_without_pull_or_fallback(self):
        transport = FakeTransport()
        transport.tags[0]['digest'] = 'incorrect'
        client = GraphStageClient(transport=transport)
        with self.assertRaises(ValueError):
            client.chat(MESSAGES, SCHEMA, 'plan')
        self.assertTrue(all(method == 'GET' for method, _, _ in transport.requests))
        self.assertFalse(client.calls[0]['attempted'])

    def test_reply_model_and_nonfinite_output_rejected_and_retained(self):
        for bad_model, content in [('some27b', '{"ok":true}'), (None, '{"bad":NaN}')]:
            transport = FakeTransport()
            transport.reply_model, transport.content = bad_model, content
            client = GraphStageClient(transport=transport)
            with self.assertRaises(ValueError):
                client.chat(MESSAGES, SCHEMA, 'verify')
            self.assertIsNotNone(client.calls[0]['response'])
            self.assertTrue(client.calls[0]['error'])

    def test_network_failure_retains_request_without_retry(self):
        transport = FakeTransport()
        transport.failure = TimeoutError('synthetic timeout')
        client = GraphStageClient(transport=transport)
        with self.assertRaises(TimeoutError):
            client.chat(MESSAGES, SCHEMA, 'plan')
        self.assertEqual(sum(method == 'POST' for method, _, _ in transport.requests), 1)
        self.assertIn('TimeoutError', client.calls[0]['error'])

    def test_chat_and_embed_share_single_permit_across_clients(self):
        transport = FakeTransport()
        one, two = GraphStageClient(transport=transport), GraphStageClient(transport=transport)
        barrier = threading.Barrier(2)
        def chat():
            barrier.wait()
            return one.chat(MESSAGES, SCHEMA, 'plan')
        def embed():
            barrier.wait()
            return two.embed(['texto sintético'])
        with ThreadPoolExecutor(max_workers=2) as pool:
            a, b = pool.submit(chat), pool.submit(embed)
            self.assertEqual(a.result(), {'ok': True})
            self.assertEqual(len(b.result()[0]), EMBED_DIMENSION)
        self.assertEqual(transport.max_active, 1)
        self.assertEqual(two.calls[0]['request']['truncate'], False)
        self.assertEqual(two.calls[0]['digest_before'], EMBED_DIGEST)


if __name__ == '__main__':
    unittest.main()
