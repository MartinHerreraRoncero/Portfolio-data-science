"""Direct query entry point for the selected original context-preservation profile."""
import argparse
import copy
from dataclasses import asdict
import json
from pathlib import Path
import sys

from experiments.compare_system import generate_arm
from experiments.preserve_evidence import PreservedGrounder, validate_catalog
from experiments.preserved_q3_verifier import Q3PreservedGrounder, Q3VerifierClient
from experiments.source_bound_trial import RecordedGenerator, checkpoint
from experiments.validate_reranker import canonical_hash, output_path
from rag_pensiones.answer import DEFAULT_GENERATION_MODEL
from rag_pensiones.grounding import DEFAULT_PARAMETERS, VERIFIER_Q3_DIGEST, VERIFIER_Q3_MODEL
from rag_pensiones.ingest import validate_project_path
from rag_pensiones.qwen_embeddings import QwenEmbedder, MODEL, _json
from rag_pensiones.search import DEFAULT_INDEX_PATH, load_index, search, validate_index
from rag_pensiones.structured_retrieval import retrieve_parallel, retrieve_for_index


def answer_preserved(index, query, units, pages, *, embedder=None, generator=None, verifier=None, partial_generation: bool = False, structured_retriever=None):
    if type(partial_generation) is not bool:
        raise ValueError('partial_generation must be a strict boolean')
    if partial_generation and verifier is None:
        raise ValueError('partial_generation=True requires an explicit verifier')
    if not isinstance(query, str) or not query.strip():
        raise ValueError('Query must be nonempty text')
    validate_index(index)
    if index['model'] != MODEL:
        raise ValueError('Selected profile requires Qwen3-Embedding4B index')
    validate_catalog(units, pages)

    if verifier is not None:
        # Release any resident Q3 verifier before fresh embedding/search so VRAM is free
        verifier.release_loaded(VERIFIER_Q3_MODEL)
        grounder_kwargs = dict(
            verifier=verifier,
            generator=generator if generator is not None else RecordedGenerator(),
        )
        if partial_generation:
            grounder_kwargs['partial_generation'] = True
        grounder = Q3PreservedGrounder(
            units,
            pages,
            **grounder_kwargs,
        )
        verification_policy = grounder.verification_policy
    else:
        grounder = PreservedGrounder(
            units,
            pages,
            generator=generator if generator is not None else RecordedGenerator(),
        )
        verification_policy = None

    digest = grounder.generator.get_digest()
    embedder = embedder if embedder is not None else QwenEmbedder(timeout=180)
    if structured_retriever is False:
        hits = search(index, query, top_k=4, client=embedder)
        structured = None
    else:
        hits, structured = retrieve_parallel(
            lambda: search(index, query, top_k=4, client=embedder),
            lambda: retrieve_for_index(query, index, structured_retriever),
        )
    outcome = generate_arm(grounder, query, hits, digest, preserve=True, verification_policy=verification_policy)

    if verifier is not None:
        metadata = dict(
            profile='preserved_q3_verifier_v1',
            model=DEFAULT_GENERATION_MODEL,
            digest=digest,
            generator_parameters=copy.deepcopy(DEFAULT_PARAMETERS),
            verifier_model=verifier.model,
            verifier_digest=verifier.expected_digest,
            verifier_parameters=copy.deepcopy(verifier.parameters),
            embedding_model=MODEL,
            embedding_digest=index['model_digest'],
            index_hash=canonical_hash(index),
            units_hash=canonical_hash(units),
            pages_hash=canonical_hash(pages),
            top_k=4,
            planner_used=False,
            closed_note_tails=False,
            max_packages=1,
        )
        if partial_generation:
            metadata['generation_profile'] = 'partial_facts_v1'
            metadata['partial_generation'] = True
    else:
        metadata = dict(
            profile='preserved_v1',
            model=DEFAULT_GENERATION_MODEL,
            digest=digest,
            embedding_model=MODEL,
            embedding_digest=index['model_digest'],
            index_hash=canonical_hash(index),
            units_hash=canonical_hash(units),
            pages_hash=canonical_hash(pages),
            top_k=4,
            planner_used=False,
            closed_note_tails=False,
            max_packages=1,
        )

    return dict(
        schema_version='preserved-direct-query-v1',
        query=query,
        metadata=metadata,
        retrieval=dict(
            hits=[dict(rank=h.rank, score=h.score, chunk=asdict(h.chunk)) for h in hits],
            embedding_calls=copy.deepcopy(getattr(embedder, 'calls', [])),
            structured=structured,
        ),
        outcome=outcome,
    )


def format_answer(report):
    outcome = report['outcome']
    if outcome['status'] == 'error':
        return 'Error: ' + outcome['error']
    view = outcome['publication']
    if view['status'] == 'insufficient_evidence':
        return view['reason']
    sources = {f"S{h['rank']}": h['chunk'] for h in outcome['expanded_hits']}
    lines = []
    for claim in view['claims']:
        lines.append(claim['text'])
        for ev in claim['evidence']:
            source = sources[ev['source_id']]
            lines.append(f"  [{ev['source_id']}, pág. {source['page_number']}] {ev['quote']}")
        lines.append('')
    return '\n'.join(lines).rstrip()


def legacy_main(argv=None):
    parser = argparse.ArgumentParser(
        description="Query pipeline with truthful stage identities and query coverage audit (default: covered pipeline; --factual-only for Q3 factual verification only; --legacy-verifier for original 9B verifier)."
    )
    parser.add_argument('--query', required=True)
    parser.add_argument('--index', type=Path, default=DEFAULT_INDEX_PATH)
    parser.add_argument('--units', type=Path, default=Path('evaluation/reconstructed_units_catalog.json'))
    parser.add_argument('--pages', type=Path, default=Path('data/processed/airef_2025_pages.jsonl'))
    parser.add_argument('--output', type=Path)
    parser.add_argument(
        '--json',
        action='store_true',
        help="Output full JSON report including mixed generator/verifier metadata",
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        '--factual-only',
        action='store_true',
        help="Select existing Q3 factual pipeline without query coverage audit",
    )
    mode_group.add_argument(
        '--legacy-verifier',
        action='store_true',
        help="Use legacy 9B verifier instead of the default selected Qwen 27B Q3_K_M verifier",
    )
    parser.add_argument('--coverage-correction', action=argparse.BooleanOptionalAction, default=None,
                        help='Single coverage correction (default in covered mode); may add up to three model calls.')
    args = parser.parse_args(argv)
    if args.coverage_correction is True and (args.legacy_verifier or args.factual_only):
        parser.error("--coverage-correction requires covered mode")
    destination = output_path(args.output)
    read = lambda p: _json(validate_project_path(p).read_text(encoding='utf-8'))
    pages = [json.loads(x) for x in validate_project_path(args.pages).read_text(encoding='utf-8').splitlines()]
    index_data = load_index(validate_project_path(args.index))
    units_data = read(args.units)

    if args.legacy_verifier:
        report = answer_preserved(index_data, args.query, units_data, pages, verifier=None)
        checkpoint(report, destination)
        print(json.dumps(report, ensure_ascii=False, indent=2) if args.json else format_answer(report))
        return int(report['outcome']['status'] == 'error')
    elif args.factual_only:
        verifier = Q3VerifierClient(expected_digest=VERIFIER_Q3_DIGEST)
        report = answer_preserved(index_data, args.query, units_data, pages, verifier=verifier)
        checkpoint(report, destination)
        print(json.dumps(report, ensure_ascii=False, indent=2) if args.json else format_answer(report))
        return int(report['outcome']['status'] == 'error')
    else:
        from experiments.query_coverage import answer_covered, format_covered_answer

        verifier = Q3VerifierClient(expected_digest=VERIFIER_Q3_DIGEST)
        correction = True if args.coverage_correction is None else args.coverage_correction
        report = answer_covered(index_data, args.query, units_data, pages, verifier=verifier,
                                coverage_correction=correction)
        checkpoint(report, destination)
        print(json.dumps(report, ensure_ascii=False, indent=2) if args.json else format_covered_answer(report))
        return int(report.get('status') == 'error')


def main(argv=None):
    """Active local profile: semantic LangGraph with 9B text stages."""
    from experiments.query_graph import main as graph_main
    return graph_main(argv)


if __name__ == '__main__':
    sys.exit(main())
