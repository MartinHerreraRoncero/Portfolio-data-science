"""Three-arm development trial with shared active Qwen4B retrieval; no gold access."""
import argparse
import copy
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

from experiments.preserve_evidence import PreservedGrounder, publication, validate_catalog, validate_publication
from experiments.reconstructed_units import ReconstructedGrounder
from experiments.source_bound_trial import RecordedGenerator, checkpoint
from experiments.validate_reranker import canonical_hash, output_path, validate_queries
from rag_pensiones.answer import DEFAULT_GENERATION_MODEL
from rag_pensiones.grounding import OllamaGrounder, validate_grounded_result
from rag_pensiones.ingest import validate_project_path
from rag_pensiones.qwen_embeddings import QwenEmbedder, MODEL, _json
from rag_pensiones.search import DEFAULT_INDEX_PATH, load_index, search, validate_index


def run(index, queries, units, pages, output, *, models=None, embedder=None):
    validate_queries(queries)
    validate_index(index)
    if index['model'] != MODEL:
        raise ValueError('Trial requires active Qwen4B embedding')
    validate_catalog(units, pages)
    destination = output_path(output)
    models = models if models is not None else {
        'baseline': OllamaGrounder(generator=RecordedGenerator()),
        'reconstructed': ReconstructedGrounder(units, pages, generator=RecordedGenerator()),
        'preserved': PreservedGrounder(units, pages, generator=RecordedGenerator())}
    if set(models) != {'baseline', 'reconstructed', 'preserved'} or any(g.model != DEFAULT_GENERATION_MODEL for g in models.values()):
        raise ValueError('Wrong arm or generator identity')
    embedder = embedder if embedder is not None else QwenEmbedder(timeout=180)
    digests = {key: g.generator.get_digest() for key, g in models.items()}
    if len(set(digests.values())) != 1:
        raise ValueError('All arms must share the same generator digest')
    report = {'schema_version': 'preserved-citation-trial-v1', 'run_status': 'running',
        'metadata': {'model': DEFAULT_GENERATION_MODEL, 'digests': digests, 'embedding_model': MODEL,
          'embedding_digest': index['model_digest'], 'index_hash': canonical_hash(index),
          'queries_hash': canonical_hash(queries), 'units_hash': canonical_hash(units), 'pages_hash': canonical_hash(pages),
          'gold_access': 'none', 'design': 'Shared top4, three arms with rotating order; no planner. Preserved publication keeps context, not relabeled semantic support.'},
        'cases': []}
    checkpoint(report, destination)
    keys = ['baseline', 'reconstructed', 'preserved']
    for number, case in enumerate(queries['cases']):
        order = keys[number % 3:] + keys[:number % 3]
        if number % 2:
            order = list(reversed(order))
        row = dict(case, condition_order=order, conditions={})
        report['cases'].append(row)
        embedder.calls.clear()
        try:
            hits = search(index, case['query'], top_k=4, client=embedder)
            row['retrieval'] = {'status': 'completed', 'hits': [dict(rank=h.rank, score=h.score, chunk=asdict(h.chunk)) for h in hits],
                                'embedding_calls': copy.deepcopy(embedder.calls)}
        except Exception as exc:
            row['retrieval'] = {'status': 'error', 'error': str(exc), 'embedding_calls': copy.deepcopy(embedder.calls)}
            row['conditions'] = {key: {'status': 'error', 'error': 'Shared retrieval: '+str(exc), 'calls': []} for key in keys}
            checkpoint(report, destination)
            continue
        checkpoint(report, destination)
        for key in order:
            grounder = models[key]
            grounder.generator.calls.clear()
            for name, empty in [('expanded_hits', []), ('catalog', {}), ('expansion', {}), ('selections', []), ('bindings', [])]:
                setattr(grounder, name, copy.deepcopy(empty))
            started = time.perf_counter()
            outcome = {}
            try:
                result = grounder.answer(case['query'], copy.deepcopy(hits), expected_digest=digests[key])
                outcome['full_result'] = result
                evaluated_hits = hits if key == 'baseline' else grounder.expanded_hits
                validate_grounded_result(result, evaluated_hits, expected_digest=digests[key], expected_model=DEFAULT_GENERATION_MODEL)
                view = publication(result, preserve=key == 'preserved')
                outcome['publication'] = view
                validate_publication(view, result, preserve=key == 'preserved')
                outcome['status'] = view['status']
            except Exception as exc:
                outcome.update(status='error', error=str(exc))
            outcome.update(wall_seconds=time.perf_counter()-started, calls=copy.deepcopy(grounder.generator.calls))
            if key != 'baseline':
                outcome.update(expansion=copy.deepcopy(grounder.expansion), catalog=copy.deepcopy(grounder.catalog),
                    selections=copy.deepcopy(grounder.selections), bindings=copy.deepcopy(grounder.bindings),
                    expanded_hits=[dict(rank=h.rank, score=h.score, chunk=asdict(h.chunk)) for h in grounder.expanded_hits])
            row['conditions'][key] = outcome
            checkpoint(report, destination)
            print(case['id'], key, outcome['status'], flush=True)
    report['run_status'] = 'error' if any(o['status'] == 'error' for row in report['cases'] for o in row['conditions'].values()) else 'completed'
    checkpoint(report, destination)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', type=Path, default=DEFAULT_INDEX_PATH)
    parser.add_argument('--queries', type=Path, default=Path('evaluation/reranker_validation_queries.json'))
    parser.add_argument('--units', type=Path, default=Path('evaluation/reconstructed_units_catalog.json'))
    parser.add_argument('--pages', type=Path, default=Path('data/processed/airef_2025_pages.jsonl'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    read = lambda p: _json(validate_project_path(p).read_text(encoding='utf-8'))
    pages = [json.loads(line) for line in validate_project_path(args.pages).read_text(encoding='utf-8').splitlines()]
    result = run(load_index(validate_project_path(args.index)), read(args.queries), read(args.units), pages, args.output)
    return int(result['run_status'] == 'error')


if __name__ == '__main__':
    sys.exit(main())
