"""Fresh active, legacy-preserved and corrected-preserved runs over frozen shared retrieval."""
import argparse
import copy
import json
import math
from pathlib import Path
import sys

from experiments.compare_system import generate_arm, finish
from experiments.fixed_context import FixedPreservedGrounder
from experiments.preserve_evidence import PreservedGrounder, validate_catalog
from experiments.source_bound_trial import RecordedGenerator, checkpoint
from experiments.validate_reranker import canonical_hash, output_path
from rag_pensiones.answer import DEFAULT_GENERATION_MODEL
from rag_pensiones.chunking import ChunkRecord
from rag_pensiones.grounding import OllamaGrounder
from rag_pensiones.ingest import validate_project_path
from rag_pensiones.qwen_embeddings import MODEL, _json
from rag_pensiones.search import DEFAULT_INDEX_PATH, SearchHit, load_index, validate_index

ARMS = ('baseline', 'legacy', 'fixed')


def validate_inputs(inputs, index):
    chunks = {c.chunk_id: c for c, _ in validate_index(index)}
    if index['model'] != MODEL or not isinstance(inputs, dict) or set(inputs) != {'schema_version', 'index_hash', 'generator_digest', 'sources', 'cases'}:
        raise ValueError('Invalid frozen retrieval input')
    if inputs['schema_version'] != 'context-fix-inputs-v1' or inputs['index_hash'] != canonical_hash(index):
        raise ValueError('Frozen retrieval index mismatch')
    if not isinstance(inputs['generator_digest'], str) or not inputs['generator_digest'].strip():
        raise ValueError('Missing pinned generator digest')
    if not isinstance(inputs['cases'], list) or not inputs['cases']:
        raise ValueError('Empty frozen cases')
    seen = set()
    for c in inputs['cases']:
        if (not isinstance(c, dict) or set(c) != {'id', 'query', 'split', 'hits'}
                or not isinstance(c['id'], str) or not c['id'] or c['id'] in seen
                or not isinstance(c['query'], str) or not c['query'].strip()
                or c['split'] not in ('known', 'reserved') or not isinstance(c['hits'], list) or len(c['hits']) != 4):
            raise ValueError('Invalid case or labels in frozen input')
        seen.add(c['id'])
        ids = []
        for rank, h in enumerate(c['hits'], 1):
            if set(h) != {'rank', 'score', 'chunk'} or h['rank'] != rank or type(h['score']) not in (int, float) or not math.isfinite(h['score']):
                raise ValueError('Invalid frozen rank or score')
            chunk = ChunkRecord(**h['chunk'])
            if chunks.get(chunk.chunk_id) != chunk:
                raise ValueError('Frozen hit does not match index')
            ids.append(chunk.chunk_id)
        if len(set(ids)) != 4:
            raise ValueError('Duplicate frozen hits')


def validate_source_hits(inputs, pages):
    mapping = {(p['doc_id'], p['page_number']): p for p in pages}
    if len(mapping) != len(pages):
        raise ValueError('Duplicate canonical page')
    for c in inputs['cases']:
        for h in c['hits']:
            chunk = h['chunk']
            page = mapping.get((chunk['doc_id'], chunk['page_number']))
            start, end = chunk['start_char'], chunk['end_char']
            if (page is None or any(chunk[k] != page[k] for k in ('source_file', 'published_at'))
                    or not 0 <= start < end <= len(page['text']) or page['text'][start:end] != chunk['text']):
                raise ValueError('Frozen hit lost canonical page provenance')


def run(inputs, index, units, pages, output, *, models=None):
    validate_inputs(inputs, index)
    validate_source_hits(inputs, pages)
    validate_catalog(units, pages)
    destination = output_path(output)
    legacy_units = copy.deepcopy(units)
    for u in legacy_units['units']:
        u.pop('source_boundary', None)
    models = models if models is not None else {
        'baseline': OllamaGrounder(generator=RecordedGenerator()),
        'legacy': PreservedGrounder(legacy_units, pages, generator=RecordedGenerator()),
        'fixed': FixedPreservedGrounder(units, pages, generator=RecordedGenerator())}
    if set(models) != set(ARMS) or any(g.model != DEFAULT_GENERATION_MODEL for g in models.values()):
        raise ValueError('Wrong comparison models')
    digests = {k: g.generator.get_digest() for k, g in models.items()}
    if set(digests.values()) != {inputs['generator_digest']}:
        raise ValueError('Comparison generator digests differ')
    report = dict(schema_version='context-fix-comparison-v1', run_status='running',
        metadata=dict(model=DEFAULT_GENERATION_MODEL, digests=digests, embedding_model=MODEL,
                      embedding_digest=index['model_digest'], input_hash=canonical_hash(inputs),
                      units_hash=canonical_hash(units), pages_hash=canonical_hash(pages), gold_access='none',
                      design='Frozen shared top4; all three generators fresh, rotating order; no judge or semantic retries'), cases=[])
    checkpoint(report, destination)
    for number, c in enumerate(inputs['cases']):
        order = list(ARMS[number % 3:] + ARMS[:number % 3])
        if number % 2:
            order.reverse()
        hits = [SearchHit(h['rank'], h['score'], ChunkRecord(**h['chunk'])) for h in c['hits']]
        row = dict(id=c['id'], query=c['query'], split=c['split'], condition_order=order,
                   retrieval=dict(status='frozen', hits=copy.deepcopy(c['hits'])), conditions={})
        report['cases'].append(row)
        for key in order:
            row['conditions'][key] = generate_arm(models[key], c['query'], hits, digests[key], preserve=key != 'baseline')
            checkpoint(report, destination)
            print(c['id'], key, row['conditions'][key]['status'], flush=True)
    return finish(report, destination)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, default=Path('evaluation/context_fix_inputs.json'))
    parser.add_argument('--units', type=Path, default=Path('evaluation/context_fix_units.json'))
    parser.add_argument('--index', type=Path, default=DEFAULT_INDEX_PATH)
    parser.add_argument('--pages', type=Path, default=Path('data/processed/airef_2025_pages.jsonl'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    read = lambda p: _json(validate_project_path(p).read_text(encoding='utf-8'))
    pages = [json.loads(x) for x in validate_project_path(args.pages).read_text(encoding='utf-8').splitlines()]
    r = run(read(args.inputs), load_index(validate_project_path(args.index)), read(args.units), pages, args.output)
    return int(r['run_status'] == 'error')


if __name__ == '__main__':
    sys.exit(main())
