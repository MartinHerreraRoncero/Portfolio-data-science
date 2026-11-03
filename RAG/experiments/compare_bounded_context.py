"""Second engineering trial: same fresh controls, with feasible-combination decoding."""
import argparse
import json
from pathlib import Path
import sys

from experiments.bounded_context import BoundedPreservedGrounder
from experiments.compare_context_fix import run
from experiments.preserve_evidence import PreservedGrounder
from experiments.source_bound_trial import RecordedGenerator, checkpoint
from rag_pensiones.grounding import OllamaGrounder
from rag_pensiones.ingest import validate_project_path
from rag_pensiones.qwen_embeddings import _json
from rag_pensiones.search import DEFAULT_INDEX_PATH, load_index


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, default=Path('evaluation/context_fix_inputs.json'))
    parser.add_argument('--units', type=Path, default=Path('evaluation/context_fix_units.json'))
    parser.add_argument('--index', type=Path, default=DEFAULT_INDEX_PATH)
    parser.add_argument('--pages', type=Path, default=Path('data/processed/airef_2025_pages.jsonl'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    read = lambda p: _json(validate_project_path(p).read_text(encoding='utf-8'))
    units = read(args.units)
    legacy_units = read(args.units)
    for u in legacy_units['units']:
        u.pop('source_boundary', None)
    pages = [json.loads(x) for x in validate_project_path(args.pages).read_text(encoding='utf-8').splitlines()]
    models = dict(baseline=OllamaGrounder(generator=RecordedGenerator()),
                  legacy=PreservedGrounder(legacy_units, pages, generator=RecordedGenerator()),
                  fixed=BoundedPreservedGrounder(units, pages, generator=RecordedGenerator()))
    r = run(read(args.inputs), load_index(validate_project_path(args.index)), units, pages, args.output, models=models)
    r['metadata']['fixed_implementation'] = 'feasible-combination-schema; engineering repair after first trial'
    checkpoint(r, validate_project_path(args.output))
    return int(r['run_status'] == 'error')


if __name__ == '__main__':
    sys.exit(main())
