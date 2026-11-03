"""Paired direct versus existing automatic planner, both using corrected notes."""
import argparse
import copy
import json
from pathlib import Path
import sys
import time

from experiments.compare_system import finish
from experiments.planned_preserved import answer_direct_notes, answer_planned_preserved, validate_profile
from experiments.source_bound_trial import RecordedGenerator, checkpoint
from experiments.validate_reranker import canonical_hash, output_path, validate_queries
from rag_pensiones.qwen_embeddings import QwenEmbedder, _json
from rag_pensiones.search import DEFAULT_INDEX_PATH, load_index
from rag_pensiones.ingest import validate_project_path

ARMS = ('direct', 'planned')


def run(index, queries, units, pages, output, *, expected_digest):
    validate_queries(queries)
    validate_profile(index, units, pages)
    destination = output_path(output)
    report = dict(schema_version='planned-preserved-comparison-v1', run_status='running', metadata=dict(
        expected_digest=expected_digest, index_hash=canonical_hash(index), units_hash=canonical_hash(units),
        pages_hash=canonical_hash(pages), queries_hash=canonical_hash(queries), gold_access='none',
        design='Fresh direct/planned per query, alternating order; same notes(true/1),index,models,budgets. Fresh retrieval per executed question; plan+resolution included. No manual split or judge.'), cases=[])
    checkpoint(report, destination)
    for number, case in enumerate(queries['cases']):
        order = list(ARMS if number % 2 == 0 else reversed(ARMS))
        row = dict(id=case['id'], query=case['query'], condition_order=order, conditions={})
        report['cases'].append(row)
        for arm in order:
            generator = RecordedGenerator()
            embedder = QwenEmbedder(timeout=180)
            started = time.perf_counter()
            out = {}
            try:
                if generator.get_digest() != expected_digest:
                    raise ValueError('Comparison generator digest differs from pinned protocol')
                answer = answer_direct_notes if arm == 'direct' else answer_planned_preserved
                result = answer(index, case['query'], units, pages, embedder=embedder, generator=generator)
                out['answer'] = result
                out['status'] = result['outcome']['status'] if arm == 'direct' else result['publication']['status']
                if generator.get_digest() != expected_digest:
                    raise ValueError('Generator digest changed after query')
            except Exception as exc:
                out.update(status='error', error=str(exc))
            out.update(calls=copy.deepcopy(generator.calls), embedding_calls=copy.deepcopy(embedder.calls),
                       wall_seconds=time.perf_counter()-started)
            row['conditions'][arm] = out
            checkpoint(report, destination)
            print(case['id'], arm, out['status'], flush=True)
    return finish(report, destination)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queries', type=Path, default=Path('evaluation/planned_preserved_queries.json'))
    parser.add_argument('--units', type=Path, default=Path('evaluation/context_fix_units.json'))
    parser.add_argument('--pages', type=Path, default=Path('data/processed/airef_2025_pages.jsonl'))
    parser.add_argument('--index', type=Path, default=DEFAULT_INDEX_PATH)
    parser.add_argument('--protocol', type=Path, default=Path('evaluation/planned_preserved_protocol.json'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    read = lambda p: _json(validate_project_path(p).read_text(encoding='utf-8'))
    protocol = read(args.protocol)
    pages = [json.loads(x) for x in validate_project_path(args.pages).read_text(encoding='utf-8').splitlines()]
    result = run(load_index(validate_project_path(args.index)), read(args.queries), read(args.units), pages,
                 args.output, expected_digest=protocol['generator_digest'])
    return int(result['run_status'] == 'error')


if __name__ == '__main__': sys.exit(main())
