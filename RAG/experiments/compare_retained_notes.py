"""Note-retention ablation under identical saved plans; all node answers are fresh."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

from experiments.compare_system import finish
from experiments.planned_preserved import answer_planned_preserved, validate_profile
from experiments.source_bound_trial import RecordedGenerator, checkpoint
from experiments.validate_reranker import canonical_hash, output_path, validate_queries
from rag_pensiones.answer import DEFAULT_GENERATION_MODEL
from rag_pensiones.planning import OllamaPlanner
from rag_pensiones.qwen_embeddings import QwenEmbedder, _json
from rag_pensiones.ingest import validate_project_path
from rag_pensiones.search import DEFAULT_INDEX_PATH, load_index

ARMS = ('overlap', 'retain_notes')


class FrozenPlanPlanner:
    def __init__(self, case, generator):
        self.model = DEFAULT_GENERATION_MODEL
        self.generator = generator
        self.query = case['query']
        self.saved = copy.deepcopy(case['conditions']['planned']['answer']['semantic']['plan'])
        self.live = OllamaPlanner(generator=generator)

    def plan(self, query, scope_years=None):
        if query != self.query or self.generator.get_digest() != self.saved['model_digest']:
            raise ValueError('Frozen plan query or digest mismatch')
        if self.saved['model'] != self.model:
            raise ValueError('Frozen plan model mismatch')
        return dict(plan=dict(subquestions=copy.deepcopy(self.saved['subquestions'])),
            **{k:copy.deepcopy(self.saved[k]) for k in ['model','model_digest','prompt_sha256','stats','parameters']})

    def resolve(self, **kwargs):
        return self.live.resolve(**kwargs)


def run(index, queries, units, pages, source, output, *, expected_digest):
    validate_queries(queries)
    validate_profile(index, units, pages)
    if source.get('run_status') != 'completed':
        raise ValueError('Require completed saved-plan source')
    original = {c['id']:c for c in source['cases']}
    if len(original) != len(source['cases']) or any(c['id'] not in original or original[c['id']]['query'] != c['query'] for c in queries['cases']):
        raise ValueError('Frozen plan source and queries differ')
    destination = output_path(output)
    report = dict(schema_version='retained-notes-comparison-v1', run_status='running', metadata=dict(
        expected_digest=expected_digest, index_hash=canonical_hash(index), units_hash=canonical_hash(units),
        pages_hash=canonical_hash(pages), queries_hash=canonical_hash(queries), source_hash=canonical_hash(source),
        gold_access='Labels not loaded; protocol files only hashed for integrity',
        design='Same saved plan in each pair, no planning calls. Fresh retrieval/resolution/grounding, alternating order. Only closed retrieved note priority differs. Canonical plan stats reused, new call costs recorded separately.'), cases=[])
    checkpoint(report, destination)
    for number, case in enumerate(queries['cases']):
        order = list(ARMS if number % 2 == 0 else reversed(ARMS))
        saved = original[case['id']]
        row = dict(id=case['id'], query=case['query'], condition_order=order, conditions={})
        report['cases'].append(row)
        for arm in order:
            generator = RecordedGenerator()
            embedder = QwenEmbedder(timeout=180)
            started = time.perf_counter()
            out = dict(reused_plan=copy.deepcopy(saved['conditions']['planned']['answer']['semantic']['plan']),
                       reused_plan_call=copy.deepcopy(saved['conditions']['planned']['calls'][0]))
            try:
                if generator.get_digest() != expected_digest:
                    raise ValueError('Generator digest differs from pinned protocol')
                result = answer_planned_preserved(index, case['query'], units, pages, generator=generator,
                    embedder=embedder, planner=FrozenPlanPlanner(saved, generator), prefer_retrieved_notes=arm=='retain_notes')
                out.update(answer=result, status=result['publication']['status'])
                if generator.get_digest() != expected_digest:
                    raise ValueError('Generator digest changed after query')
            except Exception as exc:
                out.update(status='error', error=str(exc))
            out.update(calls=copy.deepcopy(generator.calls), embedding_calls=copy.deepcopy(embedder.calls),
                       wall_seconds=time.perf_counter()-started)
            out['new_llm_stats'] = {k:sum(c.get('raw_response',{}).get(k,0) for c in generator.calls)
                                   for k in ['prompt_eval_count','eval_count','eval_duration','total_duration']}
            row['conditions'][arm] = out
            checkpoint(report, destination)
            print(case['id'], arm, out['status'], flush=True)
    return finish(report, destination)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queries', type=Path, default=Path('evaluation/planned_preserved_queries.json'))
    parser.add_argument('--source', type=Path, default=Path('evaluation/planned_preserved_results.json'))
    parser.add_argument('--units', type=Path, default=Path('evaluation/context_fix_units.json'))
    parser.add_argument('--pages', type=Path, default=Path('data/processed/airef_2025_pages.jsonl'))
    parser.add_argument('--index', type=Path, default=DEFAULT_INDEX_PATH)
    parser.add_argument('--protocol', type=Path, default=Path('evaluation/retain_notes_protocol_v2.json'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    read = lambda p: _json(validate_project_path(p).read_text(encoding='utf-8'))
    protocol = read(args.protocol)
    for name, expected in protocol['frozen_files'].items():
        if hashlib.sha256(validate_project_path(Path(name)).read_bytes()).hexdigest() != expected:
            raise ValueError('Frozen file changed: '+name)
    index, queries, units, source = load_index(validate_project_path(args.index)), read(args.queries), read(args.units), read(args.source)
    pages = [json.loads(x) for x in validate_project_path(args.pages).read_text(encoding='utf-8').splitlines()]
    for name, data in [('index',index),('queries',queries),('units',units),('pages',pages),('source',source)]:
        if canonical_hash(data) != protocol[name+'_hash']:
            raise ValueError('Comparison input differs from protocol: '+name)
    result = run(index, queries, units, pages, source, args.output, expected_digest=protocol['generator_digest'])
    return int(result['run_status'] == 'error')


if __name__ == '__main__': sys.exit(main())
