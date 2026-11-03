"""Local, content-free execution telemetry. Never changes graph state or reports."""
from contextlib import contextmanager
from contextvars import ContextVar
import json
import threading
import time
import uuid

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.tracers.context import register_configure_hook
from langgraph.errors import GraphInterrupt

PREFIX = '@rag-progress '
NODES = frozenset(('interpret', 'intent_audit', 'embed', 'catalog', 'documents',
    'ground', 'plan', 'validate', 'clarify', 'execute', 'generate', 'review',
    'prepare_repair', 'publish', 'load', 'presentation', 'context', 'report', 'persist'))
STATES = frozenset(('start', 'end', 'error', 'waiting'))


def clean_event(event):
    if not isinstance(event, dict) or event.get('node') not in NODES or event.get('state') not in STATES:
        raise ValueError('Invalid progress event')
    return {'node': event['node'], 'state': event['state']}


class ProgressStore:
    """Bounded, request-isolated metadata; UUIDs act as unguessable read tokens."""
    def __init__(self, *, ttl=1800, limit=64, max_events=512, clock=time.monotonic):
        self.ttl, self.limit, self.max_events = ttl, limit, max_events
        self.clock = clock
        self.lock = threading.Lock()
        self.entries = {}

    def _prune(self):
        now = self.clock()
        for key, value in list(self.entries.items()):
            if value['done'] and now - value['updated'] > self.ttl:
                del self.entries[key]

    def start(self, request_id: str) -> None:
        if str(uuid.UUID(request_id)) != request_id:
            raise ValueError('Invalid progress identifier')
        with self.lock:
            self._prune()
            if request_id in self.entries:
                raise ValueError('Progress identifier already used')
            if len(self.entries) >= self.limit:
                completed = [key for key, entry in self.entries.items() if entry['done']]
                if not completed:
                    raise ValueError('Progress capacity reached')
                del self.entries[completed[0]]
            self.entries[request_id] = dict(events=[], sequence=0, done=False, updated=self.clock())

    def emit(self, request_id: str, event: dict) -> None:
        event = clean_event(event)
        with self.lock:
            entry = self.entries[request_id]
            if entry['done']:
                return
            entry['sequence'] += 1
            entry['events'].append(dict(event, sequence=entry['sequence']))
            entry['events'] = entry['events'][-self.max_events:]
            entry['updated'] = self.clock()

    def read(self, request_id: str, after: int = 0) -> dict:
        if not isinstance(after, int) or after < 0:
            raise ValueError('Invalid progress cursor')
        with self.lock:
            self._prune()
            entry = self.entries[request_id]
            return dict(events=[dict(event) for event in entry['events'] if event['sequence'] > after],
                        done=entry['done'], cursor=entry['sequence'],
                        truncated=bool(entry['events'] and after < entry['events'][0]['sequence'] - 1))

    def finish(self, request_id: str) -> None:
        with self.lock:
            entry = self.entries[request_id]
            entry['done'] = True
            entry['updated'] = self.clock()


class NodeObserver(BaseCallbackHandler):
    run_inline = True

    def __init__(self, emit):
        self.emit = emit
        self.runs = {}
        self.lock = threading.RLock()

    def _emit(self, node, state):
        try:
            self.emit(dict(node=node, state=state))
        except Exception:
            pass  # Telemetry is optional; a broken observer cannot break a query.

    def on_chain_start(self, serialized, inputs, *, run_id, metadata=None, **kwargs):
        node = (metadata or {}).get('langgraph_node')
        if node in NODES and kwargs.get('name') == node:
            with self.lock:
                self.runs[run_id] = node
                self._emit(node, 'start')

    def on_chain_end(self, outputs, *, run_id, **kwargs):
        with self.lock:
            node = self.runs.pop(run_id, None)
            if node:
                self._emit(node, 'error' if isinstance(outputs, dict) and outputs.get('errors') else 'end')

    def on_chain_error(self, error, *, run_id, **kwargs):
        with self.lock:
            node = self.runs.pop(run_id, None)
            if node:
                self._emit(node, 'waiting' if isinstance(error, GraphInterrupt) else 'error')


_observer = ContextVar('local_rag_progress_observer', default=None)
register_configure_hook(_observer, inheritable=True)


@contextmanager
def observe_nodes(emit):
    token = _observer.set(NodeObserver(emit))
    try:
        yield
    finally:
        _observer.reset(token)


def stdout_event(event):
    print(PREFIX + json.dumps(clean_event(event), separators=(',', ':')), flush=True)
