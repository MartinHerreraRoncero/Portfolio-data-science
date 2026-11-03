import json
import threading
import unittest
from unittest.mock import patch
from experiments import prototype_server as server
from rag_pensiones.graph_types import CHAT_MODEL,EMBED_MODEL


class PrototypeSessionTests(unittest.TestCase):
    def handler(self,headers=None):
        handler=object.__new__(server.Handler);handler.path='/end-session'
        handler.headers=headers or {'Origin':'http://127.0.0.1:8765','Host':'127.0.0.1:8765'}
        handler.respond=lambda status,body:setattr(handler,'result',(status,body))
        return handler

    def test_close_requires_local_origin_and_rejects_busy_without_unloading(self):
        for headers in [{'Host':'127.0.0.1:8765'}, {'Origin':'http://elsewhere:8765','Host':'127.0.0.1:8765'},
                        {'Origin':'http://127.0.0.1:8765','Host':'elsewhere:8765'}]:
            h=self.handler(headers)
            with patch.object(server,'unload_task_models')as unload:h.do_POST();unload.assert_not_called()
            self.assertEqual(h.result[0],403)
        lock=threading.Lock();lock.acquire();h=self.handler()
        with patch.object(server,'LOCK',lock),patch.object(server,'unload_task_models')as unload:
            h.do_POST();unload.assert_not_called();self.assertEqual(h.result[0],409)
        lock.release()

    def test_close_success_releases_lock_and_keeps_http_server_available(self):
        lock=threading.Lock();h=self.handler()
        with patch.object(server,'LOCK',lock),patch.object(server,'unload_task_models',return_value=[dict(model=m,unloaded=True)for m in [CHAT_MODEL,EMBED_MODEL]]):
            h.do_POST();self.assertEqual(h.result[0],200);self.assertFalse(lock.locked())
        self.assertEqual(h.result[1]['status'],'session_closed')
        h.path='/health';h.do_GET()
        self.assertEqual(h.result,(200,dict(status='ok',busy=False,experimental=True)))

    def test_unload_posts_only_the_two_task_models_with_zero_keep_alive(self):
        requests=[]
        class Connection:
            def __init__(self,host,port,timeout):self.model=None;self.asserted=(host,port)
            def request(self,method,path,body,headers):
                payload=json.loads(body);self.model=payload['model'];requests.append((method,path,payload,self.asserted))
            def getresponse(self):return self
            status=200
            def read(self,limit):return json.dumps(dict(model=self.model,done=True,done_reason='unload')).encode()
            def close(self):pass
        with patch.object(server,'HTTPConnection',Connection):results=server.unload_task_models()
        self.assertTrue(all(r['unloaded']for r in results))
        self.assertEqual([r[2]['model']for r in requests],[CHAT_MODEL,EMBED_MODEL])
        for method,path,payload,address in requests:
            self.assertEqual((method,path,address),('POST','/api/generate',('127.0.0.1',11434)))
            self.assertEqual(payload['keep_alive'],0);self.assertNotIn('prompt',payload)

    def test_close_failure_is_visible_and_unlocks(self):
        h=self.handler();lock=threading.Lock()
        with patch.object(server,'LOCK',lock),patch.object(server,'unload_task_models',side_effect=OSError('Local service unavailable')):
            h.do_POST();self.assertEqual(h.result[0],500);self.assertFalse(lock.locked())
            self.assertIn('unavailable',h.result[1]['error'])


if __name__=='__main__':unittest.main()
