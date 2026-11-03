"""Paired query transport under identical saved plans; node outputs are fresh."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

from experiments.compare_retained_notes import FrozenPlanPlanner
from experiments.compare_system import finish
from experiments.planned_preserved import answer_planned_preserved, validate_profile
from experiments.requirements_query import compose_execution_query
from experiments.scoped_planning import build_scoped_plan_messages
from experiments.source_bound_trial import RecordedGenerator, checkpoint
from experiments.validate_reranker import canonical_hash, output_path, validate_queries
from rag_pensiones.ingest import validate_project_path
from rag_pensiones.answer import DEFAULT_GENERATION_MODEL, parse_chat_response
from rag_pensiones.planning import PLAN_JSON_SCHEMA, build_plan_messages
from rag_pensiones.qwen_embeddings import QwenEmbedder, _json
from rag_pensiones.search import DEFAULT_INDEX_PATH, load_index

ARMS = ('baseline', 'carry')


def validated_drivers(drivers, source, stage):
    if set(drivers) != {'schema_version','cases'} or drivers['schema_version'] != 'requirements-query-drivers-v1':
        raise ValueError('Invalid driver schema')
    if source.get('run_status') != 'completed' or stage not in ['gate','primary']:
        raise ValueError('Require completed source and known stage')
    validate_queries(dict(schema_version='reranker-validation-queries-v1',
        cases=[dict(id=c['id'],query=c['query']) for c in drivers['cases']]))
    originals={c['id']:c for c in source['cases']}
    if len(originals) != len(source['cases']):raise ValueError('Duplicate source IDs')
    selected=[]
    for driver in drivers['cases']:
        if set(driver) != {'id','query','source_id','source_arm','stage'}:
            raise ValueError('Invalid driver fields')
        if any(not isinstance(driver[key],str) or not driver[key].strip() for key in ['source_id','source_arm','stage']):
            raise ValueError('Driver source and stage must be nonempty strings')
        if driver['source_arm'] not in ['original','scoped'] or driver['stage'] not in ['gate','primary']:
            raise ValueError('Invalid source arm/stage')
        saved=originals.get(driver['source_id'])
        if saved is None or saved['query'] != driver['query']:
            raise ValueError('Driver/source query mismatch')
        condition=saved['conditions'][driver['source_arm']]
        plan=condition['answer']['semantic']['plan']
        call=condition['calls'][0]
        builder=build_plan_messages if driver['source_arm']=='original' else build_scoped_plan_messages
        prompt_bytes=json.dumps(call['messages'],ensure_ascii=False).encode('utf-8')
        if call['messages']!=builder(driver['query']) or call['schema']!=PLAN_JSON_SCHEMA:
            raise ValueError('Source planning messages/query/schema mismatch')
        parsed=parse_chat_response(call['raw_response'],DEFAULT_GENERATION_MODEL)
        if (condition['status']=='error' or parsed['payload']!={'subquestions':plan['subquestions']}
                or parsed['stats']!=plan['stats'] or plan['model']!=DEFAULT_GENERATION_MODEL
                or hashlib.sha256(prompt_bytes).hexdigest()!=plan['prompt_sha256']
                or plan['parameters']!=dict(temperature=0,seed=42,num_ctx=8192,num_predict=1024)):
            raise ValueError('Source plan metadata differs from recorded call')
        if driver['stage']==stage:selected.append((driver,dict(query=saved['query'],conditions={'planned':condition})))
    if not selected:raise ValueError('No drivers for requested stage')
    return selected


def run(index, drivers, units, pages, source, output, *, expected_digest, stage):
    selected=validated_drivers(drivers,source,stage)
    if any(saved['conditions']['planned']['answer']['semantic']['plan']['model_digest']!=expected_digest for _,saved in selected):
        raise ValueError('Saved plan digest differs from pinned protocol')
    validate_profile(index,units,pages)
    destination=output_path(output)
    report=dict(schema_version='requirements-query-comparison-v1',run_status='running',metadata=dict(
        expected_digest=expected_digest,stage=stage,
        **{k+'_hash':canonical_hash(v) for k,v in [('index',index),('drivers',drivers),('units',units),('pages',pages),('source',source)]},
        gold_access='No labels loaded; reference files only hashed',
        design='Identical saved plan per pair. Fresh retrieval/resolution/grounding; alternating arms. Only query_builder differs. Canonical stats retain reused plan; physical new costs exclude planning. No semantic retries.'),cases=[])
    checkpoint(report,destination)
    for number,(driver,saved) in enumerate(selected):
        order=list(ARMS if number%2==0 else reversed(ARMS))
        row=dict(**driver,condition_order=order,conditions={});report['cases'].append(row)
        for arm in order:
            generator=RecordedGenerator();embedder=QwenEmbedder(timeout=180)
            original=saved['conditions']['planned']
            out=dict(reused_plan=copy.deepcopy(original['answer']['semantic']['plan']),reused_plan_call=copy.deepcopy(original['calls'][0]))
            started=time.perf_counter()
            try:
                if generator.get_digest()!=expected_digest:raise ValueError('Generator differs from pinned digest')
                answer=answer_planned_preserved(index,driver['query'],units,pages,generator=generator,embedder=embedder,
                    planner=FrozenPlanPlanner(saved,generator),prefer_retrieved_notes=True,
                    query_builder=compose_execution_query if arm=='carry' else None)
                if answer['metadata']['digest']!=expected_digest or generator.get_digest()!=expected_digest:
                    raise ValueError('Answer/generator digest differs from pinned digest')
                out.update(answer=answer,status=answer['publication']['status'])
            except Exception as exc:out.update(status='error',error=f'{type(exc).__name__}: {exc}')
            out.update(calls=copy.deepcopy(generator.calls),embedding_calls=copy.deepcopy(embedder.calls),wall_seconds=time.perf_counter()-started)
            out['new_llm_stats']={}
            for key in ['prompt_eval_count','eval_count','eval_duration','total_duration']:
                values=[c['raw_response'].get(key) if isinstance(c.get('raw_response'),dict) else None for c in generator.calls]
                out['new_llm_stats'][key]=sum(values) if all(type(v) is int and v>=0 for v in values) else None
            row['conditions'][arm]=out;checkpoint(report,destination)
            print(driver['id'],arm,out['status'],flush=True)
    return finish(report,destination)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--drivers',type=Path,default=Path('evaluation/carry_requirements_drivers.json'))
    parser.add_argument('--source',type=Path,default=Path('evaluation/scoped_plan_results.json'))
    parser.add_argument('--stage',choices=['gate','primary'],required=True)
    parser.add_argument('--units',type=Path,default=Path('evaluation/linked_note_units.json'))
    parser.add_argument('--pages',type=Path,default=Path('data/processed/airef_2025_pages.jsonl'))
    parser.add_argument('--index',type=Path,default=DEFAULT_INDEX_PATH)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    read=lambda p:_json(validate_project_path(p).read_text(encoding='utf-8'))
    protocol=read(args.protocol)
    for name,digest in protocol['frozen_files'].items():
        if hashlib.sha256(validate_project_path(Path(name)).read_bytes()).hexdigest()!=digest:
            raise ValueError('Frozen file changed: '+name)
    index=load_index(validate_project_path(args.index));drivers,source,units=read(args.drivers),read(args.source),read(args.units)
    pages=[json.loads(x) for x in validate_project_path(args.pages).read_text(encoding='utf-8').splitlines()]
    for key,data in [('index',index),('drivers',drivers),('source',source),('units',units),('pages',pages)]:
        if canonical_hash(data)!=protocol[key+'_hash']:raise ValueError('Input differs from protocol: '+key)
    result=run(index,drivers,units,pages,source,args.output,expected_digest=protocol['generator_digest'],stage=args.stage)
    return int(result['run_status']=='error')


if __name__=='__main__':sys.exit(main())
