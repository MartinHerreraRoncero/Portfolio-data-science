"""Fresh original vs scoped planning; identical downstream code and source profile."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

from experiments.compare_system import finish
from experiments.planned_preserved import answer_planned_preserved, validate_profile
from experiments.scoped_planning import ScopedPlanner
from experiments.source_bound_trial import RecordedGenerator, checkpoint
from experiments.validate_reranker import canonical_hash, output_path, validate_queries
from rag_pensiones.ingest import validate_project_path
from rag_pensiones.planning import OllamaPlanner
from rag_pensiones.qwen_embeddings import QwenEmbedder, _json
from rag_pensiones.search import DEFAULT_INDEX_PATH, load_index

ARMS = ('original', 'scoped')


def run(index, queries, units, pages, output, *, expected_digest):
    validate_queries(queries)
    validate_profile(index, units, pages)
    destination = output_path(output)
    report = dict(schema_version='scoped-planning-comparison-v1', run_status='running',
        metadata=dict(expected_digest=expected_digest,
            **{k+'_hash':canonical_hash(v) for k,v in [('index',index),('queries',queries),('units',units),('pages',pages)]},
            gold_access='No labels loaded; protocol inputs hashed only',
            design='Both plans and all downstream calls fresh. Alternating order, one pass. Only planning system prompt differs; linked retrieved notes enabled in both. Retrieval algorithm/index unchanged; different plan queries can retrieve different chunks. No semantic retries.'), cases=[])
    checkpoint(report, destination)
    for number,case in enumerate(queries['cases']):
        order = list(ARMS if number%2==0 else reversed(ARMS))
        row = dict(id=case['id'],query=case['query'],condition_order=order,conditions={})
        report['cases'].append(row)
        for arm in order:
            generator = RecordedGenerator()
            embedder = QwenEmbedder(timeout=180)
            planner = (OllamaPlanner if arm=='original' else ScopedPlanner)(generator=generator)
            started = time.perf_counter()
            out = {}
            try:
                if generator.get_digest() != expected_digest:
                    raise ValueError('Generator differs from pinned digest')
                answer = answer_planned_preserved(index,case['query'],units,pages,
                    embedder=embedder,generator=generator,planner=planner,prefer_retrieved_notes=True)
                if answer['metadata']['digest'] != expected_digest:
                    raise ValueError('Actual answer digest differs from pinned protocol')
                out.update(answer=answer,status=answer['publication']['status'])
                if generator.get_digest() != expected_digest:
                    raise ValueError('Generator digest changed after query')
            except Exception as exc:
                out=dict(status='error',error=f'{type(exc).__name__}: {exc}')
            out.update(calls=copy.deepcopy(generator.calls),embedding_calls=copy.deepcopy(embedder.calls),
                wall_seconds=time.perf_counter()-started)
            out['new_llm_stats']={}
            for key in ['prompt_eval_count','eval_count','eval_duration','total_duration']:
                values=[c.get('raw_response',{}).get(key) for c in generator.calls]
                out['new_llm_stats'][key]=(sum(values) if all(type(v) is int and v>=0 for v in values) else None)
            row['conditions'][arm]=out
            checkpoint(report,destination)
            print(case['id'],arm,out['status'],flush=True)
    return finish(report,destination)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--queries',type=Path,default=Path('evaluation/planned_preserved_queries.json'))
    parser.add_argument('--units',type=Path,default=Path('evaluation/linked_note_units.json'))
    parser.add_argument('--pages',type=Path,default=Path('data/processed/airef_2025_pages.jsonl'))
    parser.add_argument('--index',type=Path,default=DEFAULT_INDEX_PATH)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    read=lambda p:_json(validate_project_path(p).read_text(encoding='utf-8'))
    protocol=read(args.protocol)
    for name,expected in protocol['frozen_files'].items():
        if hashlib.sha256(validate_project_path(Path(name)).read_bytes()).hexdigest()!=expected:
            raise ValueError('Frozen file changed: '+name)
    index=load_index(validate_project_path(args.index))
    queries,units=read(args.queries),read(args.units)
    pages=[json.loads(x) for x in validate_project_path(args.pages).read_text(encoding='utf-8').splitlines()]
    for key,data in [('index',index),('queries',queries),('units',units),('pages',pages)]:
        if canonical_hash(data)!=protocol[key+'_hash']:
            raise ValueError('Input differs from protocol: '+key)
    result=run(index,queries,units,pages,args.output,expected_digest=protocol['generator_digest'])
    return int(result['run_status']=='error')


if __name__=='__main__':sys.exit(main())
