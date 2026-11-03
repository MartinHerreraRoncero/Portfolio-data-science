"""Repeated evaluation of the unchanged selected direct preservation profile."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

from experiments.preserve_evidence import validate_catalog
from experiments.query_preserved import answer_preserved
from experiments.source_bound_trial import RecordedGenerator, checkpoint
from experiments.validate_reranker import canonical_hash, output_path, validate_queries
from rag_pensiones.ingest import validate_project_path
from rag_pensiones.qwen_embeddings import MODEL, QwenEmbedder, _json
from rag_pensiones.search import DEFAULT_INDEX_PATH, load_index, validate_index


def validate_schedule(queries, schedule):
    validate_queries(queries)
    ids={c['id'] for c in queries['cases']}
    if not isinstance(schedule,list) or len(schedule)!=3*len(ids):raise ValueError('Require three full repetitions')
    expected={(r,cid) for r in range(1,4) for cid in ids}
    actual=[]
    for item in schedule:
        if not isinstance(item,dict) or set(item)!={'repetition','id'}:
            raise ValueError('Invalid schedule fields')
        if type(item['repetition']) is not int or not isinstance(item['id'],str):
            raise ValueError('Invalid schedule types')
        actual.append((item['repetition'],item['id']))
    if len(set(actual))!=len(actual) or set(actual)!=expected:
        raise ValueError('Schedule repeats, omits, or adds a case/repetition')


def run(index,queries,units,pages,output,*,expected_digest,schedule):
    validate_schedule(queries,schedule)
    validate_index(index)
    if index['model']!=MODEL:raise ValueError('Selected profile requires Qwen4B index')
    validate_catalog(units,pages)
    destination=output_path(output);lookup={c['id']:c['query'] for c in queries['cases']}
    report=dict(schema_version='preserved-stability-v1',run_status='running',metadata=dict(
        expected_digest=expected_digest,repetitions=3,profile='preserved_v1',planner_used=False,
        **{k+'_hash':canonical_hash(v) for k,v in [('index',index),('queries',queries),('units',units),('pages',pages),('schedule',schedule)]},
        gold_access='No labels loaded; frozen reference files only hashed',
        design='One unchanged direct profile. Fresh clients/retrieval/generation for all scheduled repetitions; no historical answer/embedding reuse, no conversation history. Same parameters/seed; server not reset. No extra retries or code tuning.'),cases=[])
    checkpoint(report,destination)
    for item in schedule:
        out=dict(**item,query=lookup[item['id']]);started=time.perf_counter()
        generator=embedder=answer=None
        try:
            generator=RecordedGenerator();embedder=QwenEmbedder(timeout=180)
            if generator.get_digest()!=expected_digest:raise ValueError('Generator differs from pinned protocol')
            answer=answer_preserved(index,out['query'],units,pages,generator=generator,embedder=embedder)
            meta=answer['metadata']
            if meta['digest']!=expected_digest or generator.get_digest()!=expected_digest:
                raise ValueError('Actual answer or generator digest changed')
            if (meta['profile']!='preserved_v1' or meta['planner_used'] is not False
                    or meta['closed_note_tails'] is not False or meta['max_packages']!=1):
                raise ValueError('Selected profile changed')
            out.update(answer=answer,status=answer['outcome']['status'])
            if out['status']=='error':out['error']=answer['outcome']['error']
        except Exception as exc:
            out.update(status='error',error=f'{type(exc).__name__}: {exc}')
            if answer is not None:out['invalid_answer']=copy.deepcopy(answer)
        out.update(calls=copy.deepcopy(getattr(generator,'calls',[])),
                   embedding_calls=copy.deepcopy(getattr(embedder,'calls',[])),wall_seconds=time.perf_counter()-started)
        out['new_llm_stats']={}
        for key in ['prompt_eval_count','eval_count','eval_duration','total_duration']:
            values=[c['raw_response'].get(key) if isinstance(c.get('raw_response'),dict) else None for c in out['calls']]
            out['new_llm_stats'][key]=sum(values) if all(type(v) is int and v>=0 for v in values) else None
        report['cases'].append(out);checkpoint(report,destination)
        print(item['repetition'],item['id'],out['status'],flush=True)
    report['run_status']='error' if any(c['status']=='error' for c in report['cases']) else 'completed'
    checkpoint(report,destination)
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--queries',type=Path,default=Path('evaluation/baseline_stability_queries.json'))
    parser.add_argument('--units',type=Path,default=Path('evaluation/reconstructed_units_catalog.json'))
    parser.add_argument('--pages',type=Path,default=Path('data/processed/airef_2025_pages.jsonl'))
    parser.add_argument('--index',type=Path,default=DEFAULT_INDEX_PATH)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    read=lambda p:_json(validate_project_path(p).read_text(encoding='utf-8'))
    protocol=read(args.protocol)
    for name,digest in protocol['frozen_files'].items():
        if hashlib.sha256(validate_project_path(Path(name)).read_bytes()).hexdigest()!=digest:
            raise ValueError('Frozen file changed: '+name)
    index=load_index(validate_project_path(args.index));queries,units=read(args.queries),read(args.units)
    pages=[json.loads(x) for x in validate_project_path(args.pages).read_text(encoding='utf-8').splitlines()]
    schedule=protocol['schedule']
    for key,data in [('index',index),('queries',queries),('units',units),('pages',pages),('schedule',schedule)]:
        if canonical_hash(data)!=protocol[key+'_hash']:raise ValueError('Input differs from protocol: '+key)
    result=run(index,queries,units,pages,args.output,expected_digest=protocol['generator_digest'],schedule=schedule)
    return int(result['run_status']=='error')


if __name__=='__main__':sys.exit(main())
