"""First-pass generalization diagnostic. No reference labels are read here."""
import json,hashlib,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.compare_mixed_single27 import (MixedSingle27Transport,ensure_cold_state,
    GENERATOR_9B_MODEL,TEXT_27B_MODEL,EMBEDDING_4B_MODEL,PINNED_9B_DIGEST,PINNED_27B_DIGEST,
    PINNED_EMBEDDING_DIGEST,EFFECTIVE_GENERATION_OPTIONS)
from experiments.compare_layout_answers import build_index
from experiments.table_grounded import execute_table_attempt
from experiments.preserve_evidence import validate_catalog
from experiments.query_coverage import answer_covered,CorrectionRecordedGenerator,format_covered_answer
from experiments.preserved_q3_verifier import Q3VerifierClient,VERIFIER_Q3_PARAMETERS
from experiments.source_bound_trial import checkpoint
from rag_pensiones.qwen_embeddings import QwenEmbedder
from rag_pensiones.table_evidence import build_table_packets
from rag_pensiones.grounding import DEFAULT_PARAMETERS
base=ROOT/'evaluation/generalization_2026-10-28'
out=base/'first_pass'
if out.exists():raise SystemExit('Refusing existing first_pass output directory')
out.mkdir()
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
queries=read(base/'queries.json')['cases']
assert len(queries)==14 and len({q['id'] for q in queries})==14
old=ROOT/'evaluation/table_grounding_2026-10-27/workspace_v2'
old_index,old_units,old_bundle=[read(old/n) for n in ('layout_index.json','layout_table_catalog.json','structured.json')]
old_pages=[json.loads(l) for l in (old/'layout_pages.jsonl').read_text(encoding='utf-8').splitlines()]
legacy_pages=[json.loads(l) for l in (base/'legacy_pages.jsonl').read_text(encoding='utf-8').splitlines()]
legacy_units=read(base/'original_units.json')
validate_catalog(legacy_units,legacy_pages)
native_bundle=read(base/'native_2026/structured.json')
native_pages=[json.loads(l) for l in (base/'native_2026/pages.jsonl').read_text(encoding='utf-8').splitlines()]
try:
 build_table_packets(native_bundle,native_pages)
 native_table_error=None
except Exception as exc:native_table_error=dict(type=type(exc).__name__,message=str(exc))
catalog_error=read(base/'reanchor_error.json') if (base/'reanchor_error.json').exists() else None
# This fixed probe retains both observed admission failures. It cannot silently repair or bypass them.
if native_table_error is None or catalog_error is None:raise RuntimeError('Preparation state differs from declared diagnostic; review before execution')
transport=MixedSingle27Transport()
expected={GENERATOR_9B_MODEL:PINNED_9B_DIGEST,TEXT_27B_MODEL:PINNED_27B_DIGEST,EMBEDDING_4B_MODEL:PINNED_EMBEDDING_DIGEST}
def runtime_check():
 version=transport.get_version()
 if version.get('version')!='0.40.11':raise ValueError('Unexpected server version')
 tags={m['name']:m['digest'] for m in transport.get_tags()['models']}
 if any(tags.get(m)!=d for m,d in expected.items()):raise ValueError('Unexpected model digest')
 return version
version=runtime_check();ensure_cold_state(transport)
legacy_index=build_index(legacy_pages,QwenEmbedder(timeout=180))
checkpoint(legacy_index,out/'legacy_index.json')
ensure_cold_state(transport)
source_paths=[p for d in ('rag_pensiones','experiments') for p in (ROOT/d).rglob('*.py')]+[Path(__file__).resolve()]
input_paths=[base/'queries.json',base/'legacy_pages.jsonl',base/'original_units.json',base/'native_2026/structured.json',base/'native_2026/pages.jsonl',base/'reanchor_error.json',out/'legacy_index.json']
input_paths += [old/n for n in ('layout_index.json','layout_table_catalog.json','structured.json','layout_pages.jsonl')]
input_paths += [ROOT/'Airef/Estudio_Regla_Gasto_Pensiones-2026.pdf',ROOT/'Airef/Opinion sobre la Sostenibilidad de las AAPP a largo plazo Marzo 2025.pdf']
hashes=lambda:{p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths+input_paths}
frozen=hashes()
blocked=[dict(id=c['id'],query=c['query'],status='blocked_before_inference',adequate=False,include_in_native_denominator=True,
             stage='table_admission' if c['route']=='native_table' else 'catalog_binding',
             error=native_table_error if c['route']=='native_table' else catalog_error,calls=[])
         for c in queries if c['route'].startswith('native_')]
schedule=[dict(id=c['id'],route=c['route'],query=c['query']) for c in queries if c['route']=='old_table']
schedule += [dict(id=c['id'],route='legacy_diagnostic',query=c['query']) for c in queries if c['route']=='native_prose']
protocol=dict(schema_version='generalization-first-pass-v1',server_version=version,model_digests=expected,
 frozen_files=frozen,schedule=schedule,native_blockers=dict(table_admission=native_table_error,catalog_binding=catalog_error),
 table_generation_options=EFFECTIVE_GENERATION_OPTIONS,legacy_generation_options=DEFAULT_PARAMETERS,verifier_options=VERIFIER_Q3_PARAMETERS,
 scope='Six unseen questions on known manually reviewed PDF plus eight on four pages of new PDF without regions. Eight native cases blocked by preparation; four legacy cases separate diagnostic, never native successes. No expected answers or labels accessed. No production code changes or retries.')
checkpoint(protocol,out/'protocol.json')
report=dict(run_status='running',blocked_native_cases=blocked,attempts=[])
checkpoint(report,out/'results.json')
for n,c in enumerate(schedule):
 if hashes()!=frozen:raise RuntimeError('Source/input drift')
 runtime_check();ensure_cold_state(transport)
 started=time.perf_counter();raw=None;error=None;generator=None;verifier=None;embedder=None
 try:
  if c['route']=='old_table':
   spec=dict(attempt_index=n+1,case_id=c['id'],query=c['query'],arm='mixed')
   raw=execute_table_attempt(spec,index_data=old_index,units_data=old_units,pages_data=old_pages,
      transport=transport,table_bundle=old_bundle,enabled=True,table_dynamic_schema=True,table_reference_expansion=True)
   if raw.get('error'):error=raw['error']
  else:
   generator=CorrectionRecordedGenerator();verifier=Q3VerifierClient(expected_digest=PINNED_27B_DIGEST,transport=transport);embedder=QwenEmbedder(timeout=180)
   raw=answer_covered(legacy_index,c['query'],legacy_units,legacy_pages,generator=generator,verifier=verifier,
                      embedder=embedder,coverage_correction=True)
   if raw.get('status')=='error':error=raw.get('error')
 except Exception as exc:error=f'{type(exc).__name__}: {exc}'
 entry=dict(id=c['id'],route=c['route'],query=c['query'],result=raw,error=error,wall_seconds=time.perf_counter()-started)
 if c['route']=='legacy_diagnostic':entry['actual_client_calls']=dict(generator=getattr(generator,'calls',[]),verifier=getattr(verifier,'calls',[]),embedder=getattr(embedder,'calls',[]))
 report['attempts'].append(entry)
 checkpoint(report,out/'results.json')
 print(c['id'],c['route'],error,flush=True)
 ensure_cold_state(transport)
report.update(run_status='completed',source_inputs_unchanged=hashes()==frozen,technical_errors=sum(bool(c['error']) for c in report['attempts']))
checkpoint(report,out/'results.json')
