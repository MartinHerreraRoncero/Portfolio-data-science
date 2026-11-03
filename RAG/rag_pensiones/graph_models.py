"""Local-only, pinned chat/embedding transports with a shared inference permit."""
import copy
import json
import math
import threading
import time
from urllib.parse import urlparse
import urllib.request

from rag_pensiones.graph_types import (CHAT_DIGEST, CHAT_MODEL, CHAT_OPTIONS,
                                       EMBED_DIGEST, EMBED_DIMENSION, EMBED_MODEL,EMBED_KEEP_ALIVE,EMBED_OPTIONS,stage_chat_settings,chat_keep_alive,
                                       CHAT_TEMPLATE_TOKEN_RESERVE,chat_input_bound)
from rag_pensiones.qwen_embeddings import _json, vector

INFERENCE_GATE = threading.BoundedSemaphore(1)
MAX_RESPONSE_BYTES = 16 * 1024 * 1024
MAX_CHAT_REQUEST_BYTES = 16 * 1024
MAX_EMBED_INPUT_BYTES = 6400


def prompt_context_bound(messages,options):
    """Byte-level tokenizer upper bound, plus template and reserved output."""
    return chat_input_bound(messages)+options['num_predict']


def structural_response_schema(schema):
    """Remove description annotations only at JSON Schema nodes, on a copy.

    Prompt documentation stays intact. Enum/const/default objects are instance
    data, and names in properties/$defs are not schema keywords.
    """
    if not isinstance(schema,dict):return copy.deepcopy(schema)
    result=copy.deepcopy(schema);result.pop('description',None)
    for key in ['properties','patternProperties','$defs','definitions','dependentSchemas']:
        if isinstance(result.get(key),dict):
            result[key]={name:structural_response_schema(value)for name,value in result[key].items()}
    for key in ['allOf','anyOf','oneOf','prefixItems']:
        if isinstance(result.get(key),list):result[key]=[structural_response_schema(value)for value in result[key]]
    for key in ['items','additionalItems','additionalProperties','unevaluatedItems','unevaluatedProperties',
                'contains','propertyNames','not','if','then','else','contentSchema']:
        if key in result:
            result[key]=([structural_response_schema(value)for value in result[key]]if isinstance(result[key],list)
                         else structural_response_schema(result[key]))
    if isinstance(result.get('dependencies'),dict):
        result['dependencies']={name:structural_response_schema(value)if isinstance(value,dict)else value
                                for name,value in result['dependencies'].items()}
    return result


class RejectRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError('Local Ollama redirects are forbidden')


class LocalOllamaTransport:
    def __init__(self, base_url='http://127.0.0.1:11434', timeout=None):
        parsed = urlparse(base_url)
        if (parsed.scheme != 'http' or parsed.hostname not in {'localhost', '127.0.0.1', '::1'}
                or parsed.username or parsed.password or parsed.path not in {'', '/'}
                or parsed.query or parsed.fragment):
            raise ValueError('Graph profile requires a local Ollama endpoint')
        if timeout is not None and (isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0):
            raise ValueError('Invalid timeout')
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), RejectRedirects())

    def request(self, method, path, payload=None):
        if path not in {'/api/tags', '/api/chat', '/api/embed'}:
            raise ValueError('Unsupported graph transport endpoint')
        request = urllib.request.Request(self.base_url + path, method=method,
                    data=json.dumps(payload, allow_nan=False,ensure_ascii=False,separators=(',',':')).encode() if payload is not None else None,
                    headers={'Content-Type': 'application/json'})
        with self.opener.open(request, timeout=self.timeout) as response:
            raw = response.read(MAX_RESPONSE_BYTES + 1)
        if len(raw) > MAX_RESPONSE_BYTES:
            raise ValueError('Oversized Ollama response')
        return _json(raw)


class GraphStageClient:
    """Every chat uses the pinned model; embeddings use the existing Qwen4B model.

    ``chat`` returns parsed JSON; raw requests and responses remain in ``calls``.
    A missing model is an error, never an instruction to pull or switch models.
    """
    def __init__(self, *, model=CHAT_MODEL, transport=None, semaphore=None):
        if model != CHAT_MODEL:
            raise ValueError('Only the configured chat model is permitted')
        self.model = model
        self.transport = transport if transport is not None else LocalOllamaTransport()
        self.semaphore = semaphore if semaphore is not None else INFERENCE_GATE
        self.calls = []

    @property
    def verification_policy(self):
        return {'model': CHAT_MODEL, 'digest': CHAT_DIGEST, 'parameters': copy.deepcopy(CHAT_OPTIONS)}

    def _digest(self, model):
        if model not in {CHAT_MODEL, EMBED_MODEL}:
            raise ValueError('Unapproved model before network access')
        data = self.transport.request('GET', '/api/tags')
        models = data.get('models') if isinstance(data, dict) else None
        if not isinstance(models, list):
            raise ValueError('Malformed model inventory')
        matches = [m for m in models if isinstance(m, dict) and model in (m.get('name'), m.get('model'))]
        expected = CHAT_DIGEST if model == CHAT_MODEL else EMBED_DIGEST
        if len(matches) != 1 or matches[0].get('digest') != expected:
            raise ValueError('Required installed model/digest mismatch: ' + model)
        return expected

    def get_digest(self):
        with self.semaphore:
            return self._digest(CHAT_MODEL)

    def get_embedding_digest(self):
        with self.semaphore:
            return self._digest(EMBED_MODEL)

    def _execute(self, payload, path, stage):
        if payload.get('model') not in {CHAT_MODEL, EMBED_MODEL}:
            raise ValueError('Unapproved inference model')
        if (path == '/api/chat' and payload['model'] != CHAT_MODEL) or (path == '/api/embed' and payload['model'] != EMBED_MODEL):
            raise ValueError('Model and operation mismatch')
        record = dict(stage=stage, endpoint=path, model=payload['model'], request=copy.deepcopy(payload),
                      attempted=False, response=None, error=None)
        self.calls.append(record)
        started = time.perf_counter()
        try:
            with self.semaphore:
                record['digest_before'] = self._digest(payload['model'])
                record['attempted'] = True
                response = self.transport.request('POST', path, payload)
                record['response'] = copy.deepcopy(response)
                if not isinstance(response, dict) or response.get('model') != payload['model']:
                    raise ValueError('Returned model identity mismatch')
                record['digest_after'] = self._digest(payload['model'])
                return response, record
        except Exception as exc:
            record['error'] = f'{type(exc).__name__}: {exc}'
            raise
        finally:
            record['wall_seconds'] = time.perf_counter() - started

    def chat(self, messages: list[dict], schema: dict, stage: str) -> dict:
        if not isinstance(stage, str) or not stage.strip():
            raise ValueError('Named graph stage required')
        if (not isinstance(messages, list) or not messages or any(
                not isinstance(m, dict) or set(m) != {'role', 'content'} or
                m['role'] not in {'system', 'user', 'assistant'} or
                not isinstance(m['content'], str) or not m['content'].strip() for m in messages)):
            raise ValueError('Invalid chat messages')
        if not isinstance(schema, dict) or schema.get('type') != 'object':
            raise ValueError('Object JSON response schema required')
        settings=stage_chat_settings(stage,messages)
        payload = dict(model=CHAT_MODEL, messages=copy.deepcopy(messages), format=structural_response_schema(schema),
                       stream=False, **settings, keep_alive=chat_keep_alive(stage))
        request_bytes=len(json.dumps(payload,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode())
        if request_bytes>MAX_CHAT_REQUEST_BYTES:
            raise ValueError('Graph chat input exceeds conservative full-request byte budget')
        if prompt_context_bound(messages,settings['options'])>settings['options']['num_ctx']:
            raise ValueError('Graph prompt exceeds conservative context bound')
        response, record = self._execute(payload, '/api/chat', stage)
        try:
            if response.get('done') is not True or response.get('done_reason') != 'stop':
                raise ValueError('Incomplete or truncated graph chat')
            for key in ['prompt_eval_count','eval_count']:
                if type(response.get(key))is not int or response[key]<0:
                    raise ValueError('Missing or invalid real token statistics: '+key)
            for key in ['prompt_eval_cached_count','total_duration','load_duration','prompt_eval_duration','eval_duration']:
                if key in response and(type(response[key])is not int or response[key]<0):
                    raise ValueError('Invalid real model statistics: '+key)
            if response['eval_count']>settings['options']['num_predict']or response['prompt_eval_count']+response['eval_count']>settings['options']['num_ctx']:
                raise ValueError('Actual model token count exceeds declared context/output budget')
            content = response.get('message', {}).get('content')
            if not isinstance(content, str):
                raise ValueError('Missing JSON chat content')
            result = _json(content)
            if not isinstance(result, dict):
                raise ValueError('Expected object chat result')
            record['parsed_response'] = copy.deepcopy(result)
            return result
        except Exception as exc:
            record['error'] = f'{type(exc).__name__}: {exc}'
            raise

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not isinstance(texts, list) or len(texts) > 32 or any(
                not isinstance(t, str) or not t.strip() for t in texts):
            raise ValueError('Expected at most32 nonempty embedding inputs')
        if not texts:
            return []
        if any(len(t.encode()) > MAX_EMBED_INPUT_BYTES for t in texts):
            raise ValueError('Embedding input exceeds byte budget')
        payload = dict(model=EMBED_MODEL, input=list(texts), truncate=False,
                       dimensions=EMBED_DIMENSION, keep_alive=EMBED_KEEP_ALIVE,options=copy.deepcopy(EMBED_OPTIONS))
        response, record = self._execute(payload, '/api/embed', 'embedding')
        try:
            values = response.get('embeddings')
            if not isinstance(values, list) or len(values) != len(texts):
                raise ValueError('Embedding count mismatch')
            return [vector(v) for v in values]
        except Exception as exc:
            record['error'] = f'{type(exc).__name__}: {exc}'
            raise


class GraphEmbedderAdapter:
    """Embedding-only identity for existing search(), using the same inference gate."""
    def __init__(self, client):
        self.client = client

    @property
    def calls(self):
        return [call for call in self.client.calls if call['endpoint'] == '/api/embed']

    def get_digest(self):
        return self.client.get_embedding_digest()

    def embed_texts(self, texts):
        return self.client.embed(texts)
