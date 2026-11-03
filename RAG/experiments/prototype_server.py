"""Local experimental UI for the existing graph CLI; startup runs no models."""
import argparse
from html import escape
from datetime import datetime,timezone
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from http.client import HTTPConnection
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import tempfile
from experiments.prototype_progress import ProgressStore, PREFIX
from experiments.prototype_chat import ConversationStore, combined_context, report_answer_kind
from urllib.parse import urlsplit,parse_qs
import uuid

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/prototipo'
CONTEXT=ROOT/'prototipo_contexto.json'
LOCK=threading.Lock()
MAX_BODY=64*1024
CONVERSATIONS=ConversationStore()
PROGRESS=ProgressStore()
HTML=(ROOT/'experiments/prototype_chat_ui.html').read_text(encoding='utf-8')


def valid_uuid(value):
    parsed=str(uuid.UUID(value))
    if parsed!=value:raise ValueError('Identificador de sesión inválido')
    return parsed


def save(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def local_origin(headers):
    try:
        origin=urlsplit(headers.get('Origin',''))
        return (origin.scheme=='http'and origin.hostname in {'127.0.0.1','localhost'}and origin.port==8765
                and not origin.username and not origin.password
                and headers.get('Host','').casefold()in {'127.0.0.1:8765','localhost:8765'})
    except ValueError:return False


def unload_task_models():
    from rag_pensiones.graph_types import CHAT_MODEL,EMBED_MODEL
    results=[]
    for model in [CHAT_MODEL,EMBED_MODEL]:
        connection=HTTPConnection('127.0.0.1',11434,timeout=30)
        try:
            connection.request('POST','/api/generate',body=json.dumps(dict(model=model,keep_alive=0,stream=False)).encode('utf-8'),headers={'Content-Type':'application/json'})
            response=connection.getresponse();raw=response.read(1024*1024+1)
            if response.status!=200 or len(raw)>1024*1024:raise RuntimeError('No se pudo descargar '+model)
            data=json.loads(raw)
            if data.get('model')!=model or data.get('done')is not True:raise RuntimeError('Respuesta de descarga inválida')
            results.append(dict(model=model,unloaded=True))
        except Exception as exc:results.append(dict(model=model,unloaded=False,error=str(exc)))
        finally:connection.close()
    return results


def run_with_progress(args,environment,request_id):
    """Drain both pipes without buffering stderr in memory or exposing model content."""
    with tempfile.TemporaryFile(mode='w+',encoding='utf-8')as errors:
        with subprocess.Popen(args+['--progress'],cwd=ROOT,shell=False,stdout=subprocess.PIPE,
                stderr=errors,text=True,encoding='utf-8',errors='replace',env=environment)as process:
            output=[]
            for line in process.stdout:
                if line.startswith(PREFIX):
                    try:PROGRESS.emit(request_id,json.loads(line[len(PREFIX):]))
                    except (ValueError,TypeError,KeyError):pass
                else:output.append(line)
            returncode=process.wait()
        errors.seek(0)
        return subprocess.CompletedProcess(args,returncode,''.join(output),errors.read())


class Handler(BaseHTTPRequestHandler):
    def respond(self,status,value,content_type='application/json; charset=utf-8'):
        data=value.encode('utf-8')if isinstance(value,str)else json.dumps(value,ensure_ascii=False,allow_nan=False).encode('utf-8')
        self.send_response(status);self.send_header('Content-Type',content_type)
        self.send_header('Content-Length',str(len(data)));self.send_header('Cache-Control','no-store')
        self.send_header('X-Content-Type-Options','nosniff');self.end_headers();self.wfile.write(data)

    def do_GET(self):
        path=urlsplit(self.path).path
        if path=='/':
            from rag_pensiones.graph_types import CHAT_MODEL
            return self.respond(200,(ROOT/'experiments/prototype_chat_ui.html').read_text(encoding='utf-8').replace('__CHAT_MODEL__',escape(CHAT_MODEL)),'text/html; charset=utf-8')
        if path=='/health':return self.respond(200,dict(status='ok',busy=LOCK.locked(),experimental=True))
        assets={'/assets/chat.css':('prototype_chat.css','text/css; charset=utf-8'),
                '/assets/chat.js':('prototype_chat.js','text/javascript; charset=utf-8')}
        if path in assets:
            name,content_type=assets[path]
            return self.respond(200,(ROOT/'experiments'/name).read_text(encoding='utf-8'),content_type)
        if path.startswith('/progress/'):
            if self.headers.get('Host','').casefold()not in {'127.0.0.1:8765','localhost:8765'}:
                return self.respond(403,dict(error='Host local requerido'))
            try:
                request_id=valid_uuid(path.removeprefix('/progress/'))
                after=int(parse_qs(urlsplit(self.path).query).get('after',['0'])[0])
                return self.respond(200,PROGRESS.read(request_id,after))
            except (ValueError,TypeError):return self.respond(400,dict(error='Identificador o cursor inválido'))
            except KeyError:return self.respond(404,dict(error='Progreso no encontrado'))
        if path.startswith('/report/'):
            try:
                report_id=valid_uuid(path.removeprefix('/report/'))
                report=json.loads((OUT/(report_id+'.json')).read_text(encoding='utf-8'))
                return self.respond(200,report)
            except (ValueError,OSError):return self.respond(404,dict(error='Traza no encontrada'))
        self.respond(404,dict(error='Ruta no encontrada'))

    def do_POST(self):
        path=urlsplit(self.path).path
        if not local_origin(self.headers):return self.respond(403,dict(error='Origen local requerido'))
        if path in {'/end-session','/new-conversation'}:
            if not local_origin(self.headers):return self.respond(403,dict(error='Origen local requerido'))
            if not LOCK.acquire(blocking=False):return self.respond(409,dict(error='Hay una consulta activa; espera a que termine.'))
            try:
                length=int(self.headers.get('Content-Length','0'))
                if not 0<=length<=MAX_BODY:return self.respond(413,dict(error='Petición demasiado grande'))
                body=json.loads(self.rfile.read(length))if length else {}
                cid=body.get('conversation_id')
                if cid is not None:valid_uuid(cid)
                if path=='/new-conversation':
                    CONVERSATIONS.clear(cid)
                    return self.respond(200,dict(conversation_id=CONVERSATIONS.create(),status='new'))
                results=unload_task_models()
                if not all(r['unloaded']for r in results):return self.respond(502,dict(error='No se pudieron descargar todos los modelos',models=results))
                CONVERSATIONS.clear(cid)
                return self.respond(200,dict(status='session_closed',models=results))
            except (ValueError,TypeError,AttributeError)as exc:return self.respond(400,dict(error=str(exc)))
            except Exception as exc:return self.respond(500,dict(error=str(exc)))
            finally:LOCK.release()
        if urlsplit(self.path).path!='/query':return self.respond(404,dict(error='Ruta no encontrada'))
        try:
            length=int(self.headers.get('Content-Length','0'))
            if not 0<length<=MAX_BODY:return self.respond(413,dict(error='La petición debe ocupar entre 1 y 65536 bytes'))
            body=json.loads(self.rfile.read(length).decode('utf-8'))
            query=body.get('query');resume=body.get('resume')
            cid=valid_uuid(body['conversation_id'])if body.get('conversation_id')is not None else None
            if not isinstance(query,str)or not query.strip()or '\0'in query:raise ValueError('Pregunta inválida')
            if resume is not None and(not isinstance(resume,str)or not resume.strip()or '\0'in resume):raise ValueError('Aclaración inválida')
            thread_id=valid_uuid(body['thread_id'])if resume is not None else str(uuid.uuid4())
            request_id=valid_uuid(body['request_id'])if body.get('request_id')is not None else None
        except (ValueError,TypeError,KeyError,AttributeError)as exc:return self.respond(400,dict(error=str(exc)))
        if not LOCK.acquire(blocking=False):return self.respond(409,dict(error='Ya hay una consulta en curso. Espera a que termine.'))
        context_path=None;progress_started=False
        try:
            if request_id is not None:
                PROGRESS.start(request_id);progress_started=True
                PROGRESS.emit(request_id,dict(node='context',state='start'))
            if cid is None:
                if resume is not None:raise ValueError('La aclaración requiere su conversación')
                cid=CONVERSATIONS.create()
            memory=CONVERSATIONS.session(cid)
            pending=memory['pending']
            if resume is not None:
                if not pending or pending['thread_id']!=thread_id or pending['query']!=query:
                    raise ValueError('La aclaración no pertenece a esta conversación')
            elif pending:raise ValueError('Responde la aclaración o inicia una nueva conversación')
            OUT.mkdir(parents=True,exist_ok=True)
            session_path=OUT/('session_'+thread_id+'.json')
            if resume is not None:
                if not session_path.exists():raise ValueError('Sesión de aclaración no encontrada')
                session=json.loads(session_path.read_text(encoding='utf-8'))
                if session['query']!=query or session.get('status')!='clarification':raise ValueError('La aclaración debe continuar la misma pregunta pendiente')
            else:session=dict(query=query,thread_id=thread_id)
            if not CONTEXT.is_file():raise FileNotFoundError('Falta prototipo_contexto.json en la raíz del proyecto')
            context=combined_context(json.loads(CONTEXT.read_text(encoding='utf-8')),CONVERSATIONS.get(cid),memory.get('answer_kind'))
            with tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',suffix='.json',prefix='chat_context_',dir=OUT,delete=False)as temporary:
                context_path=Path(temporary.name)
                json.dump(context,temporary,ensure_ascii=False,allow_nan=False)
            if request_id:PROGRESS.emit(request_id,dict(node='context',state='end'))
            report_id=str(uuid.uuid4());destination=OUT/(report_id+'.json')
            args=[sys.executable,'-m','experiments.query_graph','--query',query,'--context',str(context_path),
                  '--thread-id',thread_id,'--checkpoint',str(OUT/'checkpoints.sqlite3'),'--output',str(destination)]
            if resume is not None:args.extend(['--resume',resume])
            save(OUT/(report_id+'_input.json'),dict(query=query,resume=resume,thread_id=thread_id,created_at=datetime.now(timezone.utc).isoformat()))
            environment=os.environ.copy();environment['PYTHONIOENCODING']='utf-8'
            completed=(run_with_progress(args,environment,request_id)if request_id else
                subprocess.run(args,cwd=ROOT,shell=False,capture_output=True,text=True,encoding='utf-8',errors='replace',env=environment))
            save(OUT/(report_id+'_process.json'),dict(returncode=completed.returncode,stdout=completed.stdout,stderr=completed.stderr))
            if not destination.is_file():raise RuntimeError(completed.stderr.strip()or completed.stdout.strip()or'La CLI no guardó un informe')
            report=json.loads(destination.read_text(encoding='utf-8'))
            if report.get('status','error')=='error' or completed.returncode!=0:
                if any((call.get('response')or{}).get('done_reason')=='length'for call in report.get('stage_calls',[])):
                    raise RuntimeError('El modelo agotó su presupuesto de salida antes de terminar; no se ha añadido a la memoria.')
                raise RuntimeError('La consulta falló; no se ha añadido a la memoria.')
            session['status']=report['status'];save(session_path,session)
            from experiments.prototype_presentation import present_report
            from rag_pensiones.structured_retrieval import DEFAULT_DATABASE
            if request_id:PROGRESS.emit(request_id,dict(node='presentation',state='start'))
            presentation=present_report(report,database=DEFAULT_DATABASE)
            if request_id:PROGRESS.emit(request_id,dict(node='presentation',state='end'))
            if request_id:PROGRESS.emit(request_id,dict(node='persist',state='start'))
            save(OUT/(report_id+'_presentation.json'),presentation)
            if session['status']=='clarification':
                replies=(pending['replies']if pending else [])+([resume]if resume else [])
                while replies and (len(replies)>8 or sum(len(r.encode('utf-8'))for r in replies)>CONVERSATIONS.max_bytes):replies.pop(0)
                memory['pending']=dict(query=query,thread_id=thread_id,replies=replies)
            elif session['status']!='error' and completed.returncode==0:
                messages=[dict(role='user',content=query)]
                messages.extend(dict(role='user',content=reply)for reply in ((pending['replies']if pending else [])+([resume]if resume else [])))
                messages.append(dict(role='assistant',content=presentation['answer']))
                CONVERSATIONS.append(cid,messages);memory['pending']=None;memory['answer_kind']=report_answer_kind(report)
            if request_id:PROGRESS.emit(request_id,dict(node='persist',state='end'))
            self.respond(200,dict(status=session['status'],formatted_answer=presentation['answer'],sources=presentation['sources'],presentation=presentation,
                conversation_id=cid,session_id=thread_id,trace_url='/report/'+report_id,report=report))
        except ValueError as exc:self.respond(400,dict(error=str(exc)))
        except Exception as exc:
            if resume is not None:
                memory['pending']=None
                self.respond(200,dict(status='error',conversation_id=cid,session_id=thread_id,
                    formatted_answer='No se pudo completar la aclaración. Envía una pregunta nueva incluyendo la aclaración. Se conserva la memoria de las consultas anteriores.',error=type(exc).__name__+': '+str(exc)))
            else:self.respond(500,dict(error=type(exc).__name__+': '+str(exc)))
        finally:
            try:
                if context_path is not None:context_path.unlink(missing_ok=True)
            except OSError as exc:
                self.log_error('No se pudo eliminar contexto temporal: %s',exc)
            finally:
                if progress_started:PROGRESS.finish(request_id)
                LOCK.release()


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    server=ThreadingHTTPServer(('127.0.0.1',8765),Handler);server.daemon_threads=True
    print('Prototipo experimental: http://127.0.0.1:8765 — sin inferencias hasta pulsar Preguntar.',flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()


if __name__=='__main__':main()
