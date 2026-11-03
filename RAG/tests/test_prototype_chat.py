import io
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
from experiments.prototype_chat import ConversationStore, combined_context
from experiments import prototype_server as server


class MemoryTests(unittest.TestCase):
    def test_last_ten_messages_and_copy_isolation(self):
        store=ConversationStore(); cid=store.create()
        store.append(cid,[dict(role='user',content=str(i)) for i in range(11)])
        self.assertEqual([m['content'] for m in store.get(cid)],list(map(str,range(1,11))))
        history=store.get(cid);history[0]['content']='changed'
        self.assertEqual(store.get(cid)[0]['content'],'1')
        self.assertEqual(store.get(store.create()),[])

    def test_utf8_and_oversized_message_drop_whole_messages(self):
        store=ConversationStore(max_bytes=8);cid=store.create()
        store.append(cid,[dict(role='user',content='😀'),dict(role='assistant',content='éé')])
        self.assertEqual(len(store.get(cid)),2)
        store.append(cid,[dict(role='user',content='a')])
        self.assertEqual([m['content'] for m in store.get(cid)],['éé','a'])
        store.append(cid,[dict(role='assistant',content='x'*9)])
        self.assertEqual(store.get(cid),[])

    def test_expiry_capacity_and_clear(self):
        clock=[0];store=ConversationStore(max_sessions=2,ttl=10,clock=lambda:clock[0])
        first=store.create();clock[0]=1;second=store.create();clock[0]=2;third=store.create()
        with self.assertRaises(ValueError):store.get(first)
        store.clear(second);self.assertEqual(store.get(third),[])
        clock[0]=12
        with self.assertRaises(ValueError):store.get(third)

    def test_context_keeps_roles_preferences_and_no_mutation(self):
        base=dict(history=[dict(role='user',content='Prefiero 2026')],other='kept')
        context=combined_context(base,[dict(role='assistant',content='Dato no certificado')])
        self.assertEqual(len(base['history']),1)
        self.assertEqual(context['history'][-1]['role'],'assistant')
        self.assertEqual(context['other'],'kept')


class RequestTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=server.ROOT/'tmp')
        self.out=Path(self.temp.name);self.base=self.out/'base.json'
        self.base.write_text(json.dumps(dict(history=[dict(role='user',content='Preferencia')])),'utf-8')
        self.store=ConversationStore();self.lock=threading.Lock();self.contexts=[];self.status='answered'
        self.patches=[patch.object(server,'CONVERSATIONS',self.store),patch.object(server,'LOCK',self.lock),
                      patch.object(server,'OUT',self.out),patch.object(server,'CONTEXT',self.base),
                      patch.object(server.subprocess,'run',side_effect=self.run_cli),
                      patch('experiments.prototype_presentation.present_report',return_value=dict(answer='Respuesta',sources='Fuente'))]
        for p in self.patches:p.start()

    def tearDown(self):
        for p in reversed(self.patches):p.stop()
        self.temp.cleanup()

    def run_cli(self,args,**kwargs):
        context=Path(args[args.index('--context')+1]);self.contexts.append(json.loads(context.read_text('utf-8')))
        Path(args[args.index('--output')+1]).write_text(json.dumps(dict(status=self.status)),'utf-8')
        return type('Completed',(),dict(returncode=0,stdout='',stderr=''))()

    def request(self,body,path='/query',origin=True):
        h=object.__new__(server.Handler);h.path=path
        raw=json.dumps(body).encode();h.rfile=io.BytesIO(raw)
        h.headers={'Content-Length':str(len(raw)),'Host':'127.0.0.1:8765'}
        if origin:h.headers['Origin']='http://127.0.0.1:8765'
        h.respond=lambda status,value:setattr(h,'result',(status,value))
        h.do_POST();return h.result

    def test_followup_receives_prior_turn_and_temporary_file_removed(self):
        status,first=self.request(dict(query='Gasto en 2050'))
        self.assertEqual(status,200);cid=first['conversation_id']
        status,_=self.request(dict(query='¿Y en 2070?',conversation_id=cid))
        self.assertEqual(status,200)
        self.assertEqual([m['content'] for m in self.contexts[-1]['history']],['Preferencia','Gasto en 2050','Respuesta'])
        self.assertEqual(list(self.out.glob('chat_context_*')),[])
        self.assertEqual(len(self.store.get(cid)),4)

    def test_clarification_ownership_resume_and_no_duplicates(self):
        self.status='clarification';_,first=self.request(dict(query='Gasto'))
        cid=first['conversation_id'];self.assertEqual(self.store.get(cid),[])
        other=self.store.create()
        status,_=self.request(dict(query='Gasto',resume='2050',thread_id=first['session_id'],conversation_id=other))
        self.assertEqual(status,400)
        self.status='answered'
        status,_=self.request(dict(query='Gasto',resume='2050',thread_id=first['session_id'],conversation_id=cid))
        self.assertEqual(status,200)
        self.assertEqual([m['content'] for m in self.store.get(cid)],['Gasto','2050','Respuesta'])

    def test_new_conversation_and_close_do_not_clear_other_sessions(self):
        cid=self.store.create();other=self.store.create()
        self.store.append(other,[dict(role='user',content='Conservar')])
        status,data=self.request(dict(conversation_id=cid),'/new-conversation')
        self.assertEqual(status,200);self.assertEqual(self.store.get(data['conversation_id']),[])
        with patch.object(server,'unload_task_models',return_value=[dict(unloaded=True)]):
            self.assertEqual(self.request(dict(conversation_id=data['conversation_id']),'/end-session')[0],200)
        self.assertEqual(self.store.get(other)[0]['content'],'Conservar')

    def test_rejects_invalid_ids_origin_and_busy(self):
        self.assertEqual(self.request(dict(query='Q'),origin=False)[0],403)
        self.assertEqual(self.request(dict(query='Q',conversation_id='bad'))[0],400)
        self.assertEqual(self.request(dict(query='Q',conversation_id='00000000-0000-0000-0000-000000000000'))[0],400)
        self.lock.acquire()
        try:self.assertEqual(self.request(dict(query='Q'))[0],409)
        finally:self.lock.release()
        self.assertEqual(self.contexts,[])

    def test_failed_cli_releases_lock_and_preserves_memory_and_cleans_file(self):
        cid=self.store.create();self.store.append(cid,[dict(role='user',content='previo')])
        with patch.object(server.subprocess,'run',side_effect=OSError('test failure')):
            self.assertEqual(self.request(dict(query='Q',conversation_id=cid))[0],500)
        self.assertFalse(self.lock.locked());self.assertEqual(len(self.store.get(cid)),1)
        self.assertEqual(list(self.out.glob('chat_context_*')),[])

    def test_resume_error_ends_pending_and_allows_new_question(self):
        self.status='clarification';_,first=self.request(dict(query='Gasto'))
        body=dict(query='Gasto',resume='2050',thread_id=first['session_id'],conversation_id=first['conversation_id'])
        self.status='error';status,data=self.request(body)
        self.assertEqual((status,data['status']),(200,'error'))
        self.assertIsNone(self.store.session(first['conversation_id'])['pending'])
        self.assertEqual(self.store.get(first['conversation_id']),[])
        self.status='answered';self.assertEqual(self.request(dict(query='Gasto en 2050',conversation_id=first['conversation_id']))[0],200)

    def test_pending_clarifications_are_bounded(self):
        self.status='clarification';_,first=self.request(dict(query='Gasto'))
        for _ in range(12):
            self.assertEqual(self.request(dict(query='Gasto',resume='é'*800,thread_id=first['session_id'],conversation_id=first['conversation_id']))[0],200)
        replies=self.store.session(first['conversation_id'])['pending']['replies']
        self.assertLessEqual(len(replies),8)
        self.assertLessEqual(sum(len(r.encode('utf-8'))for r in replies),self.store.max_bytes)

    def test_cleanup_failure_still_unlocks(self):
        with patch.object(Path,'unlink',side_effect=OSError('denied')),patch.object(server.Handler,'log_error'):
            self.assertEqual(self.request(dict(query='Q'))[0],200)
        self.assertFalse(self.lock.locked())


if __name__=='__main__':unittest.main()
