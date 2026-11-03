"""Query the existing planner with preserved context and corrected footnotes."""
import argparse
import json
from pathlib import Path
import sys

from experiments.planned_preserved import answer_planned_preserved, answer_direct_notes
from experiments.query_preserved import format_answer
from experiments.source_bound_trial import checkpoint
from experiments.validate_reranker import output_path
from rag_pensiones.ingest import validate_project_path
from rag_pensiones.qwen_embeddings import _json
from rag_pensiones.search import DEFAULT_INDEX_PATH, load_index


def format_planned(report):
    sources = [dict(rank=int(s['source_id'][1:]), chunk=s) for s in report['semantic']['sources']]
    return format_answer(dict(outcome=dict(status=report['publication']['status'],
        publication=report['publication'], expanded_hits=sources)))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--query', required=True)
    parser.add_argument('--index', type=Path, default=DEFAULT_INDEX_PATH)
    parser.add_argument('--units', type=Path, default=Path('evaluation/context_fix_units.json'))
    parser.add_argument('--pages', type=Path, default=Path('data/processed/airef_2025_pages.jsonl'))
    parser.add_argument('--output', type=Path)
    parser.add_argument('--direct', action='store_true')
    parser.add_argument('--retain-notes', action='store_true')
    parser.add_argument('--scoped-plan', action='store_true')
    parser.add_argument('--carry-requirements', action='store_true')
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args(argv)
    if args.direct and args.retain_notes:
        parser.error('--retain-notes is available with the planner only')
    if args.direct and args.scoped_plan:
        parser.error('--scoped-plan is available with the planner only')
    if args.direct and args.carry_requirements:
        parser.error('--carry-requirements is available with the planner only')
    destination = output_path(args.output)
    read = lambda p: _json(validate_project_path(p).read_text(encoding='utf-8'))
    pages = [json.loads(x) for x in validate_project_path(args.pages).read_text(encoding='utf-8').splitlines()]
    answer = answer_direct_notes if args.direct else answer_planned_preserved
    options = {} if args.direct else dict(prefer_retrieved_notes=args.retain_notes)
    if args.scoped_plan:
        from experiments.scoped_planning import ScopedPlanner
        from experiments.source_bound_trial import RecordedGenerator
        generator = RecordedGenerator()
        options.update(generator=generator, planner=ScopedPlanner(generator=generator))
    if args.carry_requirements:
        from experiments.requirements_query import compose_execution_query
        options['query_builder'] = compose_execution_query
    report = answer(load_index(validate_project_path(args.index)), args.query, read(args.units), pages, **options)
    checkpoint(report, destination)
    print(json.dumps(report, ensure_ascii=False, indent=2) if args.json else
          format_answer(report) if args.direct else format_planned(report))
    return int(args.direct and report['outcome']['status'] == 'error')


if __name__ == '__main__': sys.exit(main())
