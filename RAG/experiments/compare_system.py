"""Compare complete RAG outputs: baseline, preserved context, and the same output filtered."""
import argparse
import copy
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

from experiments.direct_alignment import DirectChecker, apply_direct
from experiments.preserve_evidence import PreservedGrounder, publication, validate_catalog, validate_publication
from experiments.source_bound_trial import RecordedGenerator, checkpoint
from experiments.validate_reranker import canonical_hash, output_path, validate_queries
from rag_pensiones.answer import DEFAULT_GENERATION_MODEL
from rag_pensiones.chunking import ChunkRecord
from rag_pensiones.grounding import OllamaGrounder, validate_grounded_result
from rag_pensiones.ingest import validate_project_path
from rag_pensiones.qwen_embeddings import QwenEmbedder, MODEL, _json
from rag_pensiones.search import DEFAULT_INDEX_PATH, SearchHit, load_index, search, validate_index

ARMS = ('baseline', 'preserved', 'filtered')


def filter_arm(query, original, checker, digest):
    checker.generator.calls.clear()
    started = time.perf_counter()
    out = dict(source_arm='preserved', source_arm_hash=canonical_hash(original))
    if original.get('status') == 'error':
        return dict(out, status='error', error='Preserved source failed', calls=[], wall_seconds=0)
    out['input_view'] = copy.deepcopy(original['publication'])
    try:
        audit = checker.check(query, copy.deepcopy(out['input_view']), expected_digest=digest)
        out['audit'] = copy.deepcopy(audit)
        out['publication'] = apply_direct(out['input_view'], audit)
        out['status'] = out['publication']['status']
    except Exception as exc:
        out.update(status='error', error=str(exc))
    out.update(calls=copy.deepcopy(checker.generator.calls), wall_seconds=time.perf_counter()-started)
    return out


def validated_copy(original, hits, digest, *, preserve):
    out = copy.deepcopy(original)
    try:
        if original['status'] == 'error':
            raise ValueError('Frozen source arm failed')
        evaluated = [SearchHit(h['rank'], h['score'], ChunkRecord(**h['chunk'])) for h in hits]
        validate_grounded_result(original['full_result'], evaluated,
                                expected_digest=digest, expected_model=DEFAULT_GENERATION_MODEL)
        validate_publication(original['publication'], original['full_result'], preserve=preserve)
        if original['status'] != original['publication']['status']:
            raise ValueError('Source status disagrees with publication')
    except Exception as exc:
        out.update(original_status=original.get('status'), status='error', error=str(exc))
    return out


def finish(report, destination):
    report['run_status'] = 'error' if any(o['status'] == 'error'
        for c in report['cases'] for o in c['conditions'].values()) else 'completed'
    checkpoint(report, destination)
    return report


def run_frozen(source, output, *, checker=None):
    if source.get('schema_version') != 'preserved-citation-trial-v1' or source.get('run_status') != 'completed':
        raise ValueError('Require completed frozen preservation trial')
    validate_queries(dict(schema_version='reranker-validation-queries-v1',
                          cases=[dict(id=c['id'], query=c['query']) for c in source['cases']]))
    destination = output_path(output)
    checker = checker if checker is not None else DirectChecker()
    digest = checker.generator.get_digest()
    if source['metadata']['model'] != DEFAULT_GENERATION_MODEL or set(source['metadata']['digests'].values()) != {digest}:
        raise ValueError('Require exact frozen generator digest')
    report = dict(schema_version='rag-system-comparison-v1', run_status='running',
        metadata=dict(stage='known_frozen', model=DEFAULT_GENERATION_MODEL, digest=digest,
                      source_hash=canonical_hash(source), gold_access='none',
                      generation='Reused byte-equivalent saved baseline/preserved objects; only filter calls are new'), cases=[])
    checkpoint(report, destination)
    for case in source['cases']:
        row = dict(id=case['id'], query=case['query'], retrieval=copy.deepcopy(case['retrieval']), conditions={})
        report['cases'].append(row)
        for key in ('baseline', 'preserved'):
            original = case['conditions'][key]
            hits = case['retrieval']['hits'] if key == 'baseline' else original['expanded_hits']
            row['conditions'][key] = validated_copy(original, hits, digest, preserve=key == 'preserved')
        row['conditions']['filtered'] = filter_arm(case['query'], row['conditions']['preserved'], checker, digest)
        checkpoint(report, destination)
        print(case['id'], row['conditions']['filtered']['status'], flush=True)
    return finish(report, destination)


def generate_arm(grounder, query, hits, digest, *, preserve, verification_policy=None, expected_model=DEFAULT_GENERATION_MODEL):
    grounder.generator.calls.clear()
    if hasattr(grounder, 'verifier') and grounder.verifier is not None:
        if hasattr(grounder.verifier, 'calls') and isinstance(grounder.verifier.calls, list):
            grounder.verifier.calls.clear()
    for name, empty in [('expanded_hits', []), ('catalog', {}), ('expansion', {}), ('selections', []), ('bindings', [])]:
        setattr(grounder, name, copy.deepcopy(empty))
    started = time.perf_counter()
    out = {}
    try:
        result = grounder.answer(query, copy.deepcopy(hits), expected_digest=digest)
        out['full_result'] = result
        actual_hits = grounder.expanded_hits if preserve else hits
        if verification_policy is not None:
            validate_grounded_result(
                result,
                actual_hits,
                expected_digest=digest,
                expected_model=expected_model,
                verification_policy=verification_policy,
            )
        else:
            validate_grounded_result(
                result,
                actual_hits,
                expected_digest=digest,
                expected_model=expected_model,
            )
        allow_partial = getattr(grounder, 'partial_generation', False)
        if type(allow_partial) is not bool:
            raise ValueError('partial_generation must be a strict boolean')
        view = publication(result, preserve=preserve, allow_partial=allow_partial)
        out['publication'] = view
        validate_publication(view, result, preserve=preserve, allow_partial=allow_partial)
        out['status'] = view['status']
    except Exception as exc:
        out.update(status='error', error=str(exc))
    out.update(calls=copy.deepcopy(grounder.generator.calls), wall_seconds=time.perf_counter()-started)
    if verification_policy is not None:
        verifier_calls = (
            copy.deepcopy(grounder.verifier.calls)
            if hasattr(grounder, 'verifier') and hasattr(grounder.verifier, 'calls')
            else []
        )
        lifecycle_records = (
            copy.deepcopy(grounder.verifier.lifecycle_records)
            if hasattr(grounder, 'verifier') and hasattr(grounder.verifier, 'lifecycle_records')
            else []
        )
        out['verifier_calls'] = verifier_calls
        out['lifecycle_records'] = lifecycle_records
        out['verification_policy'] = copy.deepcopy(verification_policy)
    if preserve:
        for name in ('expansion', 'catalog', 'selections', 'bindings'):
            out[name] = copy.deepcopy(getattr(grounder, name))
        out['expanded_hits'] = [dict(rank=h.rank, score=h.score, chunk=asdict(h.chunk)) for h in grounder.expanded_hits]
    return out


def run_new(index, queries, units, pages, output, *, models=None, embedder=None, checker=None):
    validate_queries(queries)
    validate_index(index)
    if index['model'] != MODEL:
        raise ValueError('Require active Qwen4B index')
    validate_catalog(units, pages)
    destination = output_path(output)
    models = models if models is not None else {
        'baseline': OllamaGrounder(generator=RecordedGenerator()),
        'preserved': PreservedGrounder(units, pages, generator=RecordedGenerator())}
    if set(models) != {'baseline', 'preserved'} or any(g.model != DEFAULT_GENERATION_MODEL for g in models.values()):
        raise ValueError('Wrong generator arms')
    checker = checker if checker is not None else DirectChecker()
    digests = {k: g.generator.get_digest() for k, g in models.items()}
    digest = checker.generator.get_digest()
    if set(digests.values()) != {digest}:
        raise ValueError('All generators and filter must share exact digest')
    embedder = embedder if embedder is not None else QwenEmbedder(timeout=180)
    report = dict(schema_version='rag-system-comparison-v1', run_status='running',
        metadata=dict(stage='reserved_new_outputs', model=DEFAULT_GENERATION_MODEL, digest=digest,
                      embedding_model=MODEL, embedding_digest=index['model_digest'], index_hash=canonical_hash(index),
                      queries_hash=canonical_hash(queries), units_hash=canonical_hash(units), pages_hash=canonical_hash(pages),
                      gold_access='none', generation='Two arms with shared top4 and alternating order; filtered reuses preserved output'), cases=[])
    checkpoint(report, destination)
    for number, case in enumerate(queries['cases']):
        order = ['baseline', 'preserved'] if number % 2 == 0 else ['preserved', 'baseline']
        row = dict(id=case['id'], query=case['query'], condition_order=order, conditions={})
        report['cases'].append(row)
        embedder.calls.clear()
        try:
            hits = search(index, case['query'], top_k=4, client=embedder)
            row['retrieval'] = dict(status='completed', hits=[dict(rank=h.rank, score=h.score, chunk=asdict(h.chunk)) for h in hits],
                                    embedding_calls=copy.deepcopy(embedder.calls))
        except Exception as exc:
            row['retrieval'] = dict(status='error', error=str(exc), embedding_calls=copy.deepcopy(embedder.calls))
            row['conditions'] = {k: dict(status='error', error='Shared retrieval: '+str(exc), calls=[]) for k in ARMS}
            checkpoint(report, destination)
            continue
        for key in order:
            row['conditions'][key] = generate_arm(models[key], case['query'], hits, digest, preserve=key == 'preserved')
            checkpoint(report, destination)
            print(case['id'], key, row['conditions'][key]['status'], flush=True)
        row['conditions']['filtered'] = filter_arm(case['query'], row['conditions']['preserved'], checker, digest)
        checkpoint(report, destination)
        print(case['id'], 'filtered', row['conditions']['filtered']['status'], flush=True)
    return finish(report, destination)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['frozen', 'new'])
    parser.add_argument('--input', type=Path, default=Path('evaluation/preserved_results.json'))
    parser.add_argument('--queries', type=Path, default=Path('evaluation/preserved_additional_queries.json'))
    parser.add_argument('--index', type=Path, default=DEFAULT_INDEX_PATH)
    parser.add_argument('--units', type=Path, default=Path('evaluation/reconstructed_units_catalog.json'))
    parser.add_argument('--pages', type=Path, default=Path('data/processed/airef_2025_pages.jsonl'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    read = lambda p: _json(validate_project_path(p).read_text(encoding='utf-8'))
    if args.mode == 'frozen':
        result = run_frozen(read(args.input), args.output)
    else:
        pages = [json.loads(line) for line in validate_project_path(args.pages).read_text(encoding='utf-8').splitlines()]
        result = run_new(load_index(validate_project_path(args.index)), read(args.queries), read(args.units), pages, args.output)
    return int(result['run_status'] == 'error')


if __name__ == '__main__':
    sys.exit(main())
