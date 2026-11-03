"""Bounded ephemeral conversation memory; never a source of RAG evidence."""
from copy import deepcopy
import time
import uuid


class ConversationStore:
    def __init__(self, max_messages=10, max_bytes=1800, max_sessions=64, ttl=7200, clock=time.monotonic):
        self.max_messages, self.max_bytes = max_messages, max_bytes
        self.max_sessions, self.ttl, self.clock = max_sessions, ttl, clock
        self.sessions = {}

    def _expire(self):
        now = self.clock()
        for key in list(self.sessions):
            if now - self.sessions[key]['touched'] >= self.ttl:
                del self.sessions[key]

    def create(self):
        self._expire()
        if len(self.sessions) >= self.max_sessions:
            del self.sessions[min(self.sessions, key=lambda key: self.sessions[key]['touched'])]
        key = str(uuid.uuid4())
        self.sessions[key] = dict(messages=[], touched=self.clock(), pending=None)
        return key

    def session(self, key):
        self._expire()
        if not isinstance(key, str) or key not in self.sessions:
            raise ValueError('Conversación caducada o desconocida. Pulsa Nueva conversación.')
        self.sessions[key]['touched'] = self.clock()
        return self.sessions[key]

    def get(self, conversation_id):
        return deepcopy(self.session(conversation_id)['messages'])

    def append(self, conversation_id, messages):
        for message in messages:
            if (not isinstance(message, dict) or message.get('role') not in {'user', 'assistant'}
                    or not isinstance(message.get('content'), str)):
                raise ValueError('Mensaje de conversación inválido')
        session = self.session(conversation_id)
        history = session['messages'] + deepcopy(messages)
        while history and (len(history) > self.max_messages or
                              sum(len(m['content'].encode('utf-8')) for m in history) > self.max_bytes):
            history.pop(0)
        session['messages'] = history

    def clear(self, conversation_id):
        self.sessions.pop(conversation_id, None)


def report_answer_kind(report):
    needs=report.get('interpretation',{}).get('needs',[])
    kinds={n.get('kind')for n in needs if not n.get('need_id','').endswith('_explanation')}
    if not kinds:return None
    return 'quantity'if kinds=={'data'}else'text'if kinds=={'text'}else'unspecified'


def combined_context(base, history, pending_answer_kind=None):
    """Preserve explicit user preferences and the original message roles."""
    if isinstance(base, list):
        base=dict(history=deepcopy(base))
    if not isinstance(base, dict):
        raise ValueError('El contexto base debe ser objeto o historial')
    result = deepcopy(base)
    result['history'] = result.get('history', []) + deepcopy(history)
    if pending_answer_kind is not None:
        if pending_answer_kind not in {'quantity','text','unspecified'}:raise ValueError('Tipo de petición pendiente inválido')
        result['pending_answer_kind']=pending_answer_kind
    return result
