"""Paired note-tail ablation; the original single-package selector is inherited."""
import argparse
import copy
import json
from pathlib import Path
import sys

from experiments.compare_context_fix import validate_inputs, validate_source_hits
from experiments.compare_system import generate_arm, finish
from experiments.preserve_evidence import PreservedGrounder, validate_catalog
from experiments.source_bound_trial import RecordedGenerator, checkpoint
from experiments.validate_reranker import canonical_hash, output_path
from rag_pensiones.answer import DEFAULT_GENERATION_MODEL
from rag_pensiones.chunking import ChunkRecord
from rag_pensiones.ingest import validate_project_path
from rag_pensiones.qwen_embeddings import MODEL, _json
from rag_pensiones.search import DEFAULT_INDEX_PATH, SearchHit, load_index

ARMS = ('legacy', 'notes_only')


class NotesOnlyGrounder(PreservedGrounder):
    def __init__(self, extracted, pages, **kwargs):
        super().__init__(extracted, pages, **kwargs)
        for unit in extracted['units']:
            if 'source_boundary' in unit and (unit.get('kind') != 'note' or unit['source_boundary'] != 'closed'):
                raise ValueError('Closed boundary certificates are restricted to reviewed notes')
        self.closed_note_tails = True


def build_models(units, pages):
    legacy = copy.deepcopy(units)
    for unit in legacy['units']:
        unit.pop('source_boundary', None)
    return dict(legacy=PreservedGrounder(legacy, pages, generator=RecordedGenerator()),
                notes_only=NotesOnlyGrounder(copy.deepcopy(units), pages, generator=RecordedGenerator()))


def run(inputs, index, units, pages, output, *, models=None):
    validate_inputs(inputs, index)
    validate_source_hits(inputs, pages)
    validate_catalog(units, pages)
    destination = output_path(output)
    models = build_models(units, pages) if models is None else models
    if (set(models) != set(ARMS) or any(g.model != DEFAULT_GENERATION_MODEL or g.max_packages != 1
            or g.closed_note_tails != (key == 'notes_only') for key, g in models.items())):
        raise ValueError('Wrong note-only comparison models or selector flags')
    digests = {key: g.generator.get_digest() for key, g in models.items()}
    if set(digests.values()) != {inputs['generator_digest']}:
        raise ValueError('Comparison generator digests differ')
    report = dict(schema_version='notes-only-comparison-v1', run_status='running', metadata=dict(
        model=DEFAULT_GENERATION_MODEL, digests=digests, embedding_model=MODEL,
        embedding_digest=index['model_digest'], input_hash=canonical_hash(inputs),
        units_hash=canonical_hash(units), pages_hash=canonical_hash(pages), gold_access='none',
        flags={key: dict(closed_note_tails=g.closed_note_tails, max_packages=g.max_packages) for key, g in models.items()},
        design='Two fresh arms, alternating order, frozen shared top4; original selector/verifier/repair/publication; no judge'), cases=[])
    checkpoint(report, destination)
    for number, case in enumerate(inputs['cases']):
        order = list(ARMS if number % 2 == 0 else reversed(ARMS))
        hits = [SearchHit(h['rank'], h['score'], ChunkRecord(**h['chunk'])) for h in case['hits']]
        row = dict(id=case['id'], query=case['query'], split=case['split'], condition_order=order,
                   retrieval=dict(status='frozen', hits=copy.deepcopy(case['hits'])), conditions={})
        report['cases'].append(row)
        for key in order:
            row['conditions'][key] = generate_arm(models[key], case['query'], hits, digests[key], preserve=True)
            checkpoint(report, destination)
            print(case['id'], key, row['conditions'][key]['status'], flush=True)
    return finish(report, destination)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, default=Path('evaluation/context_fix_inputs.json'))
    parser.add_argument('--units', type=Path, default=Path('evaluation/context_fix_units.json'))
    parser.add_argument('--index', type=Path, default=DEFAULT_INDEX_PATH)
    parser.add_argument('--pages', type=Path, default=Path('data/processed/airef_2025_pages.jsonl'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    read = lambda path: _json(validate_project_path(path).read_text(encoding='utf-8'))
    pages = [json.loads(line) for line in validate_project_path(args.pages).read_text(encoding='utf-8').splitlines()]
    result = run(read(args.inputs), load_index(validate_project_path(args.index)), read(args.units), pages, args.output)
    return int(result['run_status'] == 'error')


if __name__ == '__main__':
    sys.exit(main())
