import io
import json
from pathlib import Path
import tempfile
import threading
import unittest
import uuid
from http.client import HTTPConnection
from unittest.mock import patch
from typing import TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, interrupt
from experiments.prototype_progress import ProgressStore, observe_nodes, NodeObserver, PREFIX
from experiments import prototype_server as server


class ProgressStoreTests(unittest.TestCase):
    def test_isolation_cursor_copy_and_bounded_events(self):
        store=ProgressStore(max_events=2);one,two=str(uuid.uuid4()),str(uuid.uuid4())
        store.start(one);store.start(two)
        for node in ['interpret','embed','catalog']:store.emit(one,dict(node=node,state='start',secret='discard'))
        result=store.read(one)
        self.assertTrue(result['truncated']);self.assertEqual(result['cursor'],3)
        self.assertEqual(len(result['events']),2);self.assertNotIn('secret',result['events'][0])
        result['events'][0]['node']='changed'
        self.assertEqual(store.read(one,2)['events'][0]['node'],'catalog')
        self.assertEqual(store.read(two)['events'],[])
        store.finish(one);store.emit(one,dict(node='publish',state='end'))
        self.assertEqual(store.read(one)['cursor'],3);self.assertTrue(store.read(one)['done'])

    def test_expiry_capacity_and_active_request_preservation(self):
        now=[0];store=ProgressStore(limit=2,ttl=10,clock=lambda:now[0])
        one,two,three=[str(uuid.uuid4())for _ in range(3)]
        store.start(one);store.start(two)
        with self.assertRaises(ValueError):store.start(three)
        store.finish(two);store.start(three)
        with self.assertRaises(KeyError):store.read(two)
        store.finish(three);now[0]=11
        with self.assertRaises(KeyError):store.read(three)
        self.assertFalse(store.read(one)['done'])
        with self.assertRaises(ValueError):store.start(one)

    def test_invalid_events_ids_and_cursors(self):
        store=ProgressStore();identifier=str(uuid.uuid4());store.start(identifier)
        for event in [None,dict(node='unknown',state='end'),dict(node='embed',state='forged')]:
            with self.assertRaises(ValueError):store.emit(identifier,event)
        with self.assertRaises(ValueError):store.start('bad')
        with self.assertRaises(ValueError):store.read(identifier,-1)


class ObserverTests(unittest.TestCase):
    def test_real_graph_parallel_nodes_observed_before_their_work_and_same_output(self):
        class State(TypedDict,total=False):
            query:str
            left:int
            right:int
        events=[];barrier=threading.Barrier(2);observing=[False]
        def branch(name,key):
            def run(state):
                if observing[0]:self.assertIn(dict(node=name,state='start'),events)
                barrier.wait(timeout=5)
                return {key:len(state['query'])}
            return run
        graph=StateGraph(State)
        graph.add_node('catalog',branch('catalog','left'));graph.add_node('documents',branch('documents','right'))
        for node in ['catalog','documents']:graph.add_edge(START,node);graph.add_edge(node,END)
        compiled=graph.compile()
        ordinary=compiled.invoke(dict(query='test'),{'callbacks':[]})
        events.clear();observing[0]=True
        with observe_nodes(events.append):observed=compiled.invoke(dict(query='test'),{'callbacks':[]})
        self.assertEqual(ordinary,observed)
        self.assertEqual(len(events),4)
        for node in ['catalog','documents']:
            self.assertLess(events.index(dict(node=node,state='start')),events.index(dict(node=node,state='end')))
        events.clear();observing[0]=False;compiled.invoke(dict(query='test'),{'callbacks':[]});self.assertEqual(events,[])

    def test_interruption_and_resume_are_not_errors(self):
        events=[];graph=StateGraph(dict)
        graph.add_node('clarify',lambda state:dict(reply=interrupt('Which year?')))
        graph.add_edge(START,'clarify');graph.add_edge('clarify',END)
        compiled=graph.compile(checkpointer=InMemorySaver());config={'configurable':{'thread_id':'offline'}}
        with observe_nodes(events.append):first=compiled.invoke({},config)
        self.assertIn('__interrupt__',first)
        self.assertEqual(events[-1],dict(node='clarify',state='waiting'))
        with observe_nodes(events.append):last=compiled.invoke(Command(resume='2050'),config)
        self.assertEqual(last['reply'],'2050');self.assertEqual(events[-1],dict(node='clarify',state='end'))

    def test_observer_failure_does_not_change_graph_and_node_errors_are_observed(self):
        graph=StateGraph(dict);graph.add_node('interpret',lambda state:dict(ok=True))
        graph.add_edge(START,'interpret');graph.add_edge('interpret',END)
        def broken(event):raise OSError('closed pipe')
        with observe_nodes(broken):self.assertEqual(graph.compile().invoke({}),dict(ok=True))
        events=[];observer=NodeObserver(events.append);identifier=uuid.uuid4()
        observer.on_chain_start(None,{},run_id=identifier,metadata={'langgraph_node':'interpret'},name='interpret')
        observer.on_chain_end({'errors':['not exposed']},run_id=identifier)
        self.assertEqual(events[-1],dict(node='interpret',state='error'))
        graph=StateGraph(dict)
        def fail(state):raise ValueError('synthetic failure')
        graph.add_node('execute',fail);graph.add_edge(START,'execute');graph.add_edge('execute',END)
        with self.assertRaises(ValueError),observe_nodes(events.append):graph.compile().invoke({})
        self.assertEqual(events[-1],dict(node='execute',state='error'))


class ProgressHTTPTests(unittest.TestCase):
    def get(self,path,host='127.0.0.1:8765'):
        handler=object.__new__(server.Handler);handler.path=path;handler.headers={'Host':host}
        handler.respond=lambda code,value,*args:setattr(handler,'result',(code,value))
        handler.do_GET();return handler.result

    def test_http_cursor_validation_host_and_assets(self):
        identifier=str(uuid.uuid4());store=ProgressStore();store.start(identifier)
        store.emit(identifier,dict(node='catalog',state='start'))
        with patch.object(server,'PROGRESS',store):
            self.assertEqual(self.get('/progress/'+identifier)[1]['events'][0]['node'],'catalog')
            self.assertEqual(self.get('/progress/'+identifier+'?after=1')[1]['events'],[])
            self.assertEqual(self.get('/progress/'+identifier+'?after=-1')[0],400)
            self.assertEqual(self.get('/progress/'+identifier,'foreign.example')[0],403)
            self.assertEqual(self.get('/progress/'+str(uuid.uuid4()))[0],404)
            self.assertEqual(self.get('/progress/bad')[0],400)
        for path in ['/assets/chat.css','/assets/chat.js']:
            status,body=self.get(path);self.assertEqual(status,200);self.assertTrue(body)
        self.assertEqual(self.get('/assets/../../AGENTS.md')[0],404)

    def test_streaming_process_consumes_real_stdout_without_leaking_events(self):
        identifier=str(uuid.uuid4());store=ProgressStore();store.start(identifier)
        payload=PREFIX+'{"node":"embed","state":"start"}\n'+PREFIX+'bad\nanswer\n'
        class Process:
            def __enter__(self):return self
            def __exit__(self,*args):pass
            stdout=io.StringIO(payload)
            def wait(self):return 0
        with patch.object(server,'PROGRESS',store),patch.object(server.subprocess,'Popen',return_value=Process())as popen:
            result=server.run_with_progress(['python','query'],{},identifier)
        self.assertEqual(result.stdout,'answer\n');self.assertEqual(result.returncode,0)
        self.assertIn('--progress',popen.call_args.args[0]);self.assertFalse(popen.call_args.kwargs['shell'])
        self.assertEqual(store.read(identifier)['events'][0]['node'],'embed')

    def test_actual_http_exposes_running_stage_before_query_finishes_and_unlocks(self):
        ready,release=threading.Event(),threading.Event();store=ProgressStore()
        identifier=str(uuid.uuid4());results=[];lock=threading.Lock()
        def run(args,environment,request_id):
            store.emit(request_id,dict(node='interpret',state='start'));ready.set()
            if not release.wait(5):raise TimeoutError('Test did not release the process')
            store.emit(request_id,dict(node='interpret',state='end'))
            Path(args[args.index('--output')+1]).write_text(json.dumps(dict(status='answered')),'utf-8')
            return server.subprocess.CompletedProcess(args,0,'','')
        with tempfile.TemporaryDirectory(dir=server.ROOT/'tmp')as directory:
            out=Path(directory);context=out/'context.json';context.write_text('{}','utf-8')
            patches=[patch.object(server,'PROGRESS',store),patch.object(server,'LOCK',lock),
                patch.object(server,'OUT',out),patch.object(server,'CONTEXT',context),
                patch.object(server,'CONVERSATIONS',server.ConversationStore()),
                patch.object(server,'run_with_progress',side_effect=run),
                patch.object(server.Handler,'log_message'),
                patch('experiments.prototype_presentation.present_report',return_value=dict(answer='Respuesta',sources='Fuente'))]
            from contextlib import ExitStack
            with ExitStack()as stack:
                for item in patches:stack.enter_context(item)
                http=server.ThreadingHTTPServer(('127.0.0.1',0),server.Handler);http.daemon_threads=True
                serving=threading.Thread(target=http.serve_forever,daemon=True);serving.start()
                def request(method,path,body=None):
                    connection=HTTPConnection('127.0.0.1',http.server_port,timeout=8)
                    try:
                        connection.request(method,path,body=json.dumps(body)if body else None,
                            headers={'Host':'127.0.0.1:8765','Origin':'http://127.0.0.1:8765','Content-Type':'application/json'})
                        response=connection.getresponse();return response.status,json.loads(response.read())
                    finally:connection.close()
                posting=threading.Thread(target=lambda:results.append(request('POST','/query',dict(query='Pregunta sintética',request_id=identifier))))
                try:
                    posting.start();self.assertTrue(ready.wait(5))
                    code,progress=request('GET','/progress/'+identifier)
                    self.assertEqual(code,200);self.assertFalse(progress['done'])
                    self.assertEqual(progress['events'][0]['state'],'start');self.assertTrue(posting.is_alive())
                    self.assertEqual(request('POST','/query',dict(query='Otra'))[0],409)
                    release.set();posting.join(5);self.assertFalse(posting.is_alive())
                    self.assertEqual(results[0][0],200)
                    self.assertTrue(lock.acquire(timeout=2));lock.release()
                    final_progress=request('GET','/progress/'+identifier)[1]
                    self.assertTrue(final_progress['done'])
                    self.assertEqual([(e['node'],e['state'])for e in final_progress['events']],[(n,s)for n in ['context','interpret','presentation','persist']for s in ['start','end']])
                    self.assertEqual(list(out.glob('chat_context_*')),[])
                    self.assertEqual(request('POST','/query',dict(query='Duplicada',request_id=identifier))[0],400)
                    self.assertFalse(lock.locked())
                finally:
                    release.set();posting.join(5);http.shutdown();http.server_close();serving.join(2)


class ProgressCLITests(unittest.TestCase):
    def test_progress_flag_preserves_saved_report_and_escapes_model_telemetry_text(self):
        from experiments import query_graph
        report=dict(status='answered',formatted_answer='texto\n'+PREFIX+'{"node":"execute","state":"end"}')
        with tempfile.TemporaryDirectory(dir=server.ROOT/'tmp')as directory:
            paths=[Path(directory)/name for name in ['normal.json','observed.json','compact.json']]
            for progress,destination in enumerate(paths):
                output=io.StringIO()
                with patch.object(query_graph,'load_default_services',return_value=object()),patch.object(query_graph,'answer_graph',return_value=report),patch('sys.stdout',output):
                    result=query_graph.main(['--query','Sintética','--output',str(destination)]+(['--json']if progress<2 else [])+(['--progress']if progress else []))
                self.assertEqual(result,0)
                if progress:
                    lines=output.getvalue().splitlines()
                    self.assertEqual(len([line for line in lines if line.startswith(PREFIX)]),4)
                    self.assertEqual(json.loads(lines[-1]),report if progress==1 else dict(status='answered'))
            self.assertEqual(paths[0].read_bytes(),paths[1].read_bytes());self.assertEqual(paths[0].read_bytes(),paths[2].read_bytes())


if __name__=='__main__':unittest.main()
