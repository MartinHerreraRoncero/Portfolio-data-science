"""Run synthetic controls or filter frozen grounded outputs without regenerating them."""
import argparse
import copy
from pathlib import Path
import sys
import time

from experiments.answer_alignment import AlignmentChecker, apply_alignment, validate_view
from experiments.preserve_evidence import validate_publication
from experiments.source_bound_trial import checkpoint
from experiments.validate_reranker import canonical_hash, output_path
from rag_pensiones.answer import DEFAULT_GENERATION_MODEL
from rag_pensiones.chunking import ChunkRecord
from rag_pensiones.grounding import validate_grounded_result
from rag_pensiones.ingest import validate_project_path
from rag_pensiones.qwen_embeddings import _json
from rag_pensiones.search import SearchHit


def validate_controls(data):
    if not isinstance(data, dict) or set(data) != {'schema_version', 'notice', 'cases'} or data['schema_version'] != 'answer-alignment-controls-v1':
        raise ValueError('Wrong controls schema')
    if not isinstance(data['notice'], str) or not isinstance(data['cases'], list) or not data['cases']:
        raise ValueError('Invalid controls list')
    seen = set()
    for case in data['cases']:
        if not isinstance(case, dict) or set(case) != {'id', 'query', 'answer'}:
            raise ValueError('Controls contain unapproved fields,possibly labels')
        if not isinstance(case['id'], str) or not case['id'] or case['id'] in seen:
            raise ValueError('Invalid control ID')
        seen.add(case['id'])
        if not isinstance(case['query'], str) or not case['query'].strip():
            raise ValueError('Invalid control query')
        validate_view(case['answer'])


def evaluate_one(checker, query, view, digest):
    checker.generator.calls.clear()
    started = time.perf_counter()
    out = {'input_view': copy.deepcopy(view)}
    try:
        audit = checker.check(query, copy.deepcopy(view), expected_digest=digest)
        out['audit'] = copy.deepcopy(audit)
        out['output_view'] = apply_alignment(view, audit)
        out['status'] = 'completed'
    except Exception as exc:
        out.update(status='error', error=str(exc))
    out.update(calls=copy.deepcopy(checker.generator.calls), wall_seconds=time.perf_counter()-started)
    return out


def run_controls(data, output, *, checker=None):
    validate_controls(data)
    destination = output_path(output)
    checker = checker if checker is not None else AlignmentChecker()
    digest = checker.generator.get_digest()
    report = {'schema_version': 'alignment-controls-run-v1', 'run_status': 'running',
        'metadata': {'model': DEFAULT_GENERATION_MODEL, 'digest': digest, 'controls_hash': canonical_hash(data), 'gold_access': 'none'}, 'cases': []}
    checkpoint(report, destination)
    for case in data['cases']:
        row = dict(id=case['id'], query=case['query'], **evaluate_one(checker, case['query'], case['answer'], digest))
        report['cases'].append(row)
        checkpoint(report, destination)
        print(case['id'], row.get('audit', {}).get('decision', row['status']), flush=True)
    report['run_status'] = 'error' if any(c['status'] == 'error' for c in report['cases']) else 'completed'
    checkpoint(report, destination)
    return report


def run_comparison(source, output, *, checker=None):
    if source.get('schema_version') != 'preserved-citation-trial-v1' or source.get('run_status') != 'completed':
        raise ValueError('Require completed frozen preservation trial')
    destination = output_path(output)
    checker = checker if checker is not None else AlignmentChecker()
    digest = checker.generator.get_digest()
    if source['metadata']['model'] != DEFAULT_GENERATION_MODEL or set(source['metadata']['digests'].values()) != {digest}:
        raise ValueError('Frozen outputs and checker must share exact generator digest')
    report = {'schema_version': 'alignment-frozen-comparison-v1', 'run_status': 'running',
        'metadata': {'model': DEFAULT_GENERATION_MODEL, 'digest': digest, 'source_hash': canonical_hash(source),
            'gold_access': 'none', 'design': 'Frozen baseline and preserved outputs,with separate alignment audit;no new retrieval or generation;existing abstentions unchanged'}, 'cases': []}
    checkpoint(report, destination)
    for case in source['cases']:
        row = dict(id=case['id'], query=case['query'], conditions={})
        report['cases'].append(row)
        for key in ('baseline', 'preserved'):
            original = case['conditions'][key]
            try:
                hits = case['retrieval']['hits'] if key == 'baseline' else original['expanded_hits']
                actual_hits = [SearchHit(h['rank'], h['score'], ChunkRecord(**h['chunk'])) for h in hits]
                validate_grounded_result(original['full_result'], actual_hits, expected_digest=digest, expected_model=DEFAULT_GENERATION_MODEL)
                validate_publication(original['publication'], original['full_result'], preserve=key == 'preserved')
                out = evaluate_one(checker, case['query'], original['publication'], digest)
            except Exception as exc:
                out = {'status': 'error', 'error': str(exc), 'calls': [],
                       'input_view': copy.deepcopy(original.get('publication'))}
            row['conditions'][key] = out
            checkpoint(report, destination)
            print(case['id'], key, out.get('audit', {}).get('decision', out['status']), flush=True)
    report['run_status'] = 'error' if any(o['status'] == 'error' for c in report['cases'] for o in c['conditions'].values()) else 'completed'
    checkpoint(report, destination)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['controls', 'compare'])
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    data = _json(validate_project_path(args.input).read_text(encoding='utf-8'))
    report = (run_controls if args.mode == 'controls' else run_comparison)(data, args.output)
    return int(report['run_status'] == 'error')


if __name__ == '__main__':
    sys.exit(main())
