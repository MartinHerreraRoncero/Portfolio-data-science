"""Two separately gated stages: literal extraction, then comparison of saved pairs."""
import argparse
import copy
from pathlib import Path
import sys

from experiments.answer_alignment import apply_alignment
from experiments.compare_alignment import validate_controls
from experiments.literal_alignment import LiteralChecker, validate_pair
from experiments.source_bound_trial import checkpoint
from experiments.validate_reranker import canonical_hash, output_path
from rag_pensiones.answer import DEFAULT_GENERATION_MODEL
from rag_pensiones.ingest import validate_project_path
from rag_pensiones.qwen_embeddings import _json


def run_extract(data, output, *, checker=None):
    validate_controls(data)
    destination = output_path(output)
    checker = checker if checker is not None else LiteralChecker()
    digest = checker.generator.get_digest()
    report = {'schema_version': 'literal-extraction-run-v1', 'run_status': 'running', 'metadata': {
        'model': DEFAULT_GENERATION_MODEL, 'digest': digest, 'input_hash': canonical_hash(data), 'gold_access': 'none'}, 'cases': []}
    checkpoint(report, destination)
    for case in data['cases']:
        row = dict(id=case['id'], query=case['query'], input_view=copy.deepcopy(case['answer']))
        checker.generator.calls.clear(); checker.diagnostics = []
        try:
            row['pair'] = checker.extract_pair(case['query'], case['answer'], expected_digest=digest)
            row['status'] = 'completed'
        except Exception as exc:
            row.update(status='error', error=str(exc))
        row.update(calls=copy.deepcopy(checker.generator.calls), diagnostics=copy.deepcopy(checker.diagnostics))
        report['cases'].append(row)
        checkpoint(report, destination)
        print(case['id'], row['status'], flush=True)
    report['run_status'] = 'error' if any(c['status'] == 'error' for c in report['cases']) else 'completed'
    checkpoint(report, destination)
    return report


def run_compare(extractions, output, *, checker=None):
    if extractions.get('schema_version') != 'literal-extraction-run-v1' or extractions.get('run_status') != 'completed':
        raise ValueError('Require completed literal extraction run')
    destination = output_path(output)
    checker = checker if checker is not None else LiteralChecker()
    digest = checker.generator.get_digest()
    if extractions['metadata']['digest'] != digest:
        raise ValueError('Generator digest changed between extraction and comparison')
    report = {'schema_version': 'literal-comparison-run-v1', 'run_status': 'running', 'metadata': {
        'model': DEFAULT_GENERATION_MODEL, 'digest': digest, 'source_hash': canonical_hash(extractions), 'gold_access': 'none'}, 'cases': []}
    checkpoint(report, destination)
    for source in extractions['cases']:
        row = dict(id=source['id'], input_view=copy.deepcopy(source['input_view']))
        checker.generator.calls.clear(); checker.diagnostics = []
        try:
            validate_pair(source['pair'], source['query'], source['input_view'])
            row['audit'] = checker.compare_pair(source['pair'], expected_digest=digest)
            row['output_view'] = apply_alignment(source['input_view'], row['audit'])
            row['status'] = 'completed'
        except Exception as exc:
            row.update(status='error', error=str(exc))
        row.update(calls=copy.deepcopy(checker.generator.calls), diagnostics=copy.deepcopy(checker.diagnostics))
        report['cases'].append(row)
        checkpoint(report, destination)
        print(source['id'], row.get('audit', {}).get('decision', row['status']), flush=True)
    report['run_status'] = 'error' if any(c['status'] == 'error' for c in report['cases']) else 'completed'
    checkpoint(report, destination)
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['extract', 'compare'])
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(argv)
    source = _json(validate_project_path(args.input).read_text(encoding='utf-8'))
    report = (run_extract if args.mode == 'extract' else run_compare)(source, args.output)
    return int(report['run_status'] == 'error')


if __name__ == '__main__': sys.exit(main())
