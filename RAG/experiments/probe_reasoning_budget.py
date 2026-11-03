"""Record native Ollama thinking controls without generating RAG answers."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from urllib.error import HTTPError
from urllib.request import ProxyHandler, Request, build_opener

from rag_pensiones.graph_types import CHAT_DIGEST, CHAT_MODEL


def probe(output_dir):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=False)
    opener = build_opener(ProxyHandler({}))

    def request(path, payload=None):
        body = None if payload is None else json.dumps(payload).encode('utf-8')
        req = Request('http://127.0.0.1:11434' + path, data=body,
                      headers={'Content-Type': 'application/json'})
        started = time.perf_counter()
        try:
            with opener.open(req, timeout=20) as response:
                status, raw = response.status, response.read()
        except HTTPError as exc:
            status, raw = exc.code, exc.read()
        return dict(http_status=status, wall_seconds=time.perf_counter()-started,
                    response=json.loads(raw))

    version = request('/api/version')
    show = request('/api/show', {'model': CHAT_MODEL})
    if version['http_status'] != 200 or show['http_status'] != 200:
        raise RuntimeError('Cannot establish local model capabilities')
    tags = request('/api/tags')
    installed = [m for m in tags['response'].get('models', []) if m['name'] == CHAT_MODEL]
    if len(installed) != 1 or installed[0]['digest'] != CHAT_DIGEST:
        raise RuntimeError('Installed model does not match configured digest')
    values = show['response'].get('thinking', {}).get('values', [])
    attempts = []
    # Deliberately empty messages: API type checks, never a question or benchmark.
    # Do not send unsupported options: Ollama can silently ignore unknown keys.
    if values != [False, True] and values != [True, False]:
        raise RuntimeError('Capability metadata changed; review before numeric probes')
    for budget in (512, 1024):
        payload = dict(model=CHAT_MODEL, messages=[], think=budget, stream=False)
        result = request('/api/chat', payload)
        attempts.append(dict(request=payload, **result))
    unsupported = all(x['http_status'] == 400 and 'think' in
                      x['response'].get('error', '') for x in attempts)
    report = dict(created_utc=datetime.now(timezone.utc).isoformat(),
                  model=CHAT_MODEL, digest=CHAT_DIGEST, version=version,
                  thinking=show['response'].get('thinking'),
                  capabilities=show['response'].get('capabilities'),
                  probes=attempts, native_numeric_budget_supported=False if unsupported else None,
                  quality_comparison='not_run_unsupported_budget' if unsupported else 'inconclusive',
                  reasoning_arms={'512': 'not_run', '1024': 'not_run', 'unbounded': 'not_run'},
                  note='Probe latency is HTTP validation time, not reasoning latency. '
                       'num_predict was not substituted for a thinking budget. '
                       'No answer quality or generalization claim is possible.')
    (out/'capabilities.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: report[k] for k in ('model', 'native_numeric_budget_supported', 'quality_comparison')}))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True)
    probe(parser.parse_args().output_dir)
