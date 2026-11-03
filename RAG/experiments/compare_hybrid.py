"""Ensayo pareado semántica/BM25+RRF, top4 fijo y gold solo en evaluación."""

import argparse
import json
from pathlib import Path
import sys
import time

from experiments.compare_retrieval_k import (
    DEFAULT_DATASET_PATH, compute_annotated_hits, compute_canonical_sha256,
    compute_slot_coverage, validate_dataset,
)
from experiments.hybrid_search import BM25_B, BM25_K1, RRF_CONSTANT, STOPWORDS, fuse_rankings, lexical_ranking
from rag_pensiones.answer import DEFAULT_GENERATION_MODEL, _reject_constant
from rag_pensiones.embeddings import OllamaEmbedder
from rag_pensiones.grounding import OllamaGrounder, validate_grounded_result
from rag_pensiones.ingest import validate_project_path
from rag_pensiones.search import DEFAULT_INDEX_PATH, SearchHit, load_index, search, validate_index


def summarize(cases: list[dict]) -> dict:
    summary = {}
    for method in ("dense", "hybrid"):
        runs = [c["conditions"][method] for c in cases]
        valid = [r for r in runs if r["status"] in ("answered", "insufficient_evidence")]
        summary[method] = {
            "cases": len(runs),
            "retrievals_evaluated": sum("annotated_passages_hit" in r for r in runs),
            "any_annotated_hit_count": sum(r.get("annotated_passages_hit", {}).get("annotated_hits", 0) > 0 for r in runs),
            "answered_count": sum(r["status"] == "answered" for r in runs),
            "insufficient_evidence_count": sum(r["status"] == "insufficient_evidence" for r in runs),
            "error_count": sum(r["status"] == "error" for r in runs),
            "valid_generation_count": len(valid),
            "all_slots_covered_count": sum(r["slot_coverage"]["all_applicable_covered"] for r in valid),
            "returned_trace_chat_count": sum(len(r["full_result"]["grounding"]["trace"]) for r in valid),
        }
    return summary


def run_comparison(index: dict, dataset: dict, grounder=None, embedder=None, generate: bool = False) -> dict:
    chunks = validate_index(index)
    if len(chunks) < 4:
        raise ValueError("Need at least four chunks")
    validate_dataset(dataset, {c.chunk_id for c, _ in chunks})
    if not isinstance(generate, bool):
        raise ValueError("generate must be boolean")
    digest = None
    if generate:
        grounder = grounder if grounder is not None else OllamaGrounder(model=DEFAULT_GENERATION_MODEL)
        if grounder.model != DEFAULT_GENERATION_MODEL or grounder.generator.model != DEFAULT_GENERATION_MODEL:
            raise ValueError("Experiment requires matching Qwen9B grounder and generator")
        digest = grounder.generator.get_digest()
        if not isinstance(digest, str) or not digest.strip():
            raise ValueError("Empty model digest")
    embedder = embedder if embedder is not None else OllamaEmbedder()
    report = {
        "schema_version": "hybrid-comparison-v1", "run_status": "completed",
        "metadata": {
            "dataset_canonical_sha256": compute_canonical_sha256(dataset),
            "index_canonical_sha256": compute_canonical_sha256(index),
            "model": DEFAULT_GENERATION_MODEL if generate else None, "model_digest": digest,
            "embedding_model": index["model"], "embedding_digest": index["model_digest"],
            "generate": generate, "top_k": 4, "bm25_k1": BM25_K1, "bm25_b": BM25_B,
            "rrf_constant": RRF_CONSTANT, "rrf_weights": "equal", "stopwords": sorted(STOPWORDS),
            "tokenizer": "NFKD remove combining marks; casefold; Unicode words and decimal numbers; comma normalized to dot; fixed stopwords; no stemming",
            "statistics_scope": "returned validated generation traces only; failed-condition usage may be unknown",
            "generator_scores": "Original cosine scores in both conditions, to isolate retrieval membership/order from score-scale changes in prompts",
            "hybrid_ranking_scores": "RRF if lexical matches exist; otherwise unchanged cosine fallback",
        }, "cases": [],
    }
    for number, case in enumerate(dataset["cases"]):
        query = case["query"]
        order = ["dense", "hybrid"] if number % 2 == 0 else ["hybrid", "dense"]
        row = {"id": case["id"], "family": case["family"], "query": query, "condition_order": order, "conditions": {}}
        try:
            dense = search(index, query, top_k=len(chunks), client=embedder)
            hybrid = fuse_rankings(dense, query)
            lexical = lexical_ranking(dense, query)
            cosines = {h.chunk.chunk_id: h.score for h in dense}
            row["rankings"] = {
                name: [{"rank": h.rank, "chunk_id": h.chunk.chunk_id, "score": h.score, "page_number": h.chunk.page_number} for h in hits]
                for name, hits in (("dense", dense), ("hybrid", hybrid))
            }
            row["rankings"]["bm25"] = [{"rank": r, "chunk_id": h.chunk.chunk_id, "score": s, "page_number": h.chunk.page_number} for r, (h, s) in enumerate(lexical, 1)]
            row["hybrid_score_type"] = "rrf" if lexical else "cosine_fallback"
        except Exception as exc:
            report["run_status"] = "error"
            row["conditions"] = {m: {"status": "error", "error": f"Retrieval: {exc}"} for m in order}
            report["cases"].append(row)
            continue
        for method in order:
            ranking = dense if method == "dense" else hybrid
            # Keep original cosine in model inputs; ranking score remains separate in report.
            hits = [SearchHit(rank=r, score=cosines[h.chunk.chunk_id], chunk=h.chunk) for r, h in enumerate(ranking[:4], 1)]
            outcome = {"status": "retrieved", "retrieved_chunk_ids": [h.chunk.chunk_id for h in hits],
                       "annotated_passages_hit": compute_annotated_hits(case["reference_chunk_ids"], [h.chunk.chunk_id for h in hits])}
            if generate:
                start = time.perf_counter()
                try:
                    if grounder.generator.get_digest() != digest:
                        raise RuntimeError("Generator digest changed before condition")
                    result = grounder.answer(query=query, hits=hits, expected_digest=digest)
                    result = validate_grounded_result(result, hits, expected_digest=digest, expected_model=DEFAULT_GENERATION_MODEL)
                    if grounder.generator.get_digest() != digest:
                        raise RuntimeError("Generator digest changed after condition")
                    outcome.update(status=result["status"], full_result=result,
                                   slot_coverage=compute_slot_coverage(case["slots"], [e["quote"] for e in result["grounding"]["selected_evidence"]]))
                except Exception as exc:
                    report["run_status"] = "error"
                    outcome.update(status="error", error=str(exc))
                outcome["generation_latency_ms"] = round((time.perf_counter() - start) * 1000, 2)
            row["conditions"][method] = outcome
        report["cases"].append(row)
    report["summary"] = summarize(report["cases"])
    json.dumps(report, allow_nan=False)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--generate", action="store_true", help="Ejecutar además generación y verificación reales")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX_PATH)
    try:
        args = parser.parse_args(argv)
        dataset = json.loads(validate_project_path(args.dataset).read_text(encoding="utf-8-sig"), parse_constant=_reject_constant)
        report = run_comparison(load_index(validate_project_path(args.index)), dataset, generate=args.generate)
        if args.json:
            print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
        else:
            print("COMPARACIÓN SEMÁNTICA / HÍBRIDA — top4 fijo")
            for method, stats in report["summary"].items():
                print(f"{method}: pasaje anotado {stats['any_annotated_hit_count']}/{stats['cases']}; answered {stats['answered_count']}; errores {stats['error_count']}")
            print("answered y cobertura literal no son métricas de corrección semántica.")
        return 0 if report["run_status"] == "completed" else 1
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 1
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
