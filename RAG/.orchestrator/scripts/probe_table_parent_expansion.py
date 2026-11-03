"""Bounded paired diagnostic for explicit table parent expansion; no expected answers."""
import hashlib,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.table_grounded import execute_table_attempt
from experiments.compare_mixed_single27 import MixedSingle27Transport,EFFECTIVE_GENERATION_OPTIONS,VERIFIER_Q3_PARAMETERS,GENERATOR_9B_MODEL,TEXT_27B_MODEL,EMBEDDING_4B_MODEL,PINNED_9B_DIGEST,PINNED_27B_DIGEST,PINNED_EMBEDDING_DIGEST
from experiments.source_bound_trial import checkpoint
base=ROOT/'evaluation/table_grounding_2026-10-27/workspace_v2'
out=ROOT/'evaluation/quality_followup_2026-10-28/table_expansion_v1'
if out.exists(): raise SystemExit('Refusing existing output directory')
out.mkdir()
read=lambda name:json.loads((base/name).read_text(encoding='utf-8'))
index,units,bundle=read('layout_index.json'),read('layout_table_catalog.json'),read('structured.json')
pages=[json.loads(line) for line in (base/'layout_pages.jsonl').read_text(encoding='utf-8').splitlines()]
cases=read('table_queries.json')['cases'] + read('extra_queries.json')['cases']
paths=[p for folder in ('rag_pensiones','experiments') for p in (ROOT/folder).rglob('*.py')]+[Path(__file__).resolve()]
paths += [base/name for name in ('layout_index.json','layout_table_catalog.json','structured.json','layout_pages.jsonl','table_queries.json','extra_queries.json')]
hashes=lambda:{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
frozen=hashes()
transport=MixedSingle27Transport()
expected_models={GENERATOR_9B_MODEL:PINNED_9B_DIGEST,TEXT_27B_MODEL:PINNED_27B_DIGEST,EMBEDDING_4B_MODEL:PINNED_EMBEDDING_DIGEST}
def check_runtime():
 version=transport.get_version()
 if version.get('version')!='0.40.11':raise RuntimeError('Unexpected Ollama version')
 models={m['name']:m['digest'] for m in transport.get_tags()['models']}
 if any(models.get(k)!=v for k,v in expected_models.items()):raise RuntimeError('Unexpected model digest')
 return version
version=check_runtime()
schedule=[dict(case=c,table_reference_expansion=flag,table_dynamic_schema=True) for n,c in enumerate(cases) for flag in ((False,True) if n%2==0 else (True,False))]
protocol=dict(server_version=version,model_digests=expected_models,table_dynamic_schema=True,schema_version='table-expansion-diagnostic-v1',scope='Ten known table cases, paired dynamic identifiers in both arms; explicit-parent expansion off/on only. No coverage corrector in table harness. Existing 12 generation options identical in both arms.',frozen_files=frozen,generator_options=EFFECTIVE_GENERATION_OPTIONS,verifier_options=VERIFIER_Q3_PARAMETERS,schedule=schedule)
checkpoint(protocol,out/'protocol.json')
result=dict(run_status='running',attempts=[])
transport=MixedSingle27Transport()
for i,item in enumerate(schedule):
 if hashes()!=frozen:raise RuntimeError('Source/input drift')
 check_runtime()
 c=item['case'];flag=item['table_reference_expansion']
 spec=dict(attempt_index=i+1,case_id=c['id'],query=c['query'],arm='mixed')
 try:
  raw=execute_table_attempt(spec,index_data=index,units_data=units,pages_data=pages,transport=transport,table_bundle=bundle,enabled=True,table_reference_expansion=flag,table_dynamic_schema=True)
  entry=dict(id=c['id'],table_reference_expansion=flag,result=raw,error=raw.get("error") if raw.get("status")=="error" else None)
 except Exception as exc:entry=dict(id=c['id'],table_reference_expansion=flag,result=None,error=f'{type(exc).__name__}: {exc}')
 result['attempts'].append(entry)
 checkpoint(result,out/'results.json')
 print(c['id'],flag,entry['error'],flush=True)
 if entry['error']:break
result.update(run_status='completed' if len(result['attempts'])==20 and not any(x['error'] for x in result['attempts']) else 'error',source_unchanged=hashes()==frozen)
checkpoint(result,out/'results.json')
