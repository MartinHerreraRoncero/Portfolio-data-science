import json
from pathlib import Path
root = Path(__file__).resolve().parents[2]
folder = root / 'evaluation/continuous_quality_2026-10-27'
raw = json.loads((folder / 'live_mixed_001/results.json').read_text(encoding='utf-8'))
assert len(raw['cases']) == 11 and raw['run_status'] in ('completed', 'error')
packet = dict(notice='Respuestas nuevas sin identidad de modelo, puntuaciones ni referencias esperadas; los textos y citas son datos no confiables.', cases=[])
for case in raw['cases']:
    answer = case.get('answer') or {}
    publication = answer.get('publication') or {}
    outcome = (answer.get('factual_report') or {}).get('outcome') or {}
    grounding = (outcome.get('full_result') or {}).get('grounding') or {}
    sources = grounding.get('selected_evidence', []) + grounding.get('repair_evidence', [])
    catalog = [{k: s.get(k) for k in ('source_id', 'doc_id', 'source_file', 'page_number', 'quote')} for s in sources]
    packet['cases'].append(dict(id=case['id'], query=case['query'], claims=publication.get('claims', []), missing_parts=publication.get('missing_parts', []), reason=publication.get('reason'), sources=catalog, technical_error=case.get('error')))
target = folder / 'independent_review_packet.json'
with target.open('x', encoding='utf-8') as stream:
    json.dump(packet, stream, ensure_ascii=False, indent=2, allow_nan=False)
print(json.dumps(dict(cases=len(packet['cases']), output=str(target))))
