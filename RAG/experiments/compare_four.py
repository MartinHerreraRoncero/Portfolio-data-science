"""Comparación de cuatro mejoras aisladas frente al grounder actual, top4 fijo."""
import argparse
import json
from pathlib import Path
import sys
import time

from experiments.compare_retrieval_k import (DEFAULT_DATASET_PATH, compute_annotated_hits,
    compute_slot_coverage, compute_canonical_sha256, validate_dataset)
from experiments.four_strategies import (ContextSelectorGrounder, SpecialistVerifierGrounder,
    StructuredGrounder, LocalReranker, extract_records)
from rag_pensiones.grounding import OllamaGrounder, validate_grounded_result
from rag_pensiones.answer import DEFAULT_GENERATION_MODEL, _reject_constant
from rag_pensiones.search import DEFAULT_INDEX_PATH, SearchHit, load_index, search, validate_index
from rag_pensiones.embeddings import OllamaEmbedder
from rag_pensiones.ingest import validate_project_path

METHODS = ("baseline", "selector", "reranker", "structured", "verifier")


def summarize(cases):
    summary = {}
    for method in METHODS:
        runs = [c["conditions"][method] for c in cases if method in c["conditions"]]
        valid = [r for r in runs if r["status"] in ("answered", "insufficient_evidence")]
        summary[method] = {"cases": len(runs), "valid_generation_count": len(valid),
            "answered_count": sum(r["status"] == "answered" for r in runs),
            "abstention_count": sum(r["status"] == "insufficient_evidence" for r in runs),
            "error_count": sum(r["status"] == "error" for r in runs),
            "retrievals_evaluated": sum("annotated_passages_hit" in r for r in runs),
            "annotated_hit_count": sum(r.get("annotated_passages_hit", {}).get("annotated_hits", 0) > 0 for r in runs),
            "all_slots_covered_count": sum(r["slot_coverage"]["all_applicable_covered"] for r in valid),
            "mean_wall_seconds_valid_only": sum(r["wall_seconds"] for r in valid) / len(valid) if valid else None,
            "returned_chat_count": sum(len(r["full_result"]["grounding"]["trace"]) for r in valid)}
    return summary


def save(report, output):
    if output:
        path = validate_project_path(Path(output))
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def run_comparison(index: dict, dataset: dict, grounders: dict, reranker, embedder=None, output=None):
    chunks = validate_index(index)
    validate_dataset(dataset, {c.chunk_id for c, _ in chunks})
    if len(chunks) < 4 or set(grounders) != set(METHODS): raise ValueError("Invalid corpus or method set")
    pins = {}
    for method, grounder in grounders.items():
        if grounder.model != DEFAULT_GENERATION_MODEL or grounder.generator.model != DEFAULT_GENERATION_MODEL:
            raise ValueError("All models must be Qwen9B")
        pins[method] = grounder.generator.get_digest()
        if not isinstance(pins[method], str) or not pins[method].strip(): raise ValueError("Empty digest")
    if len(set(pins.values())) != 1: raise ValueError("Different generator digests across methods")
    if output: validate_project_path(Path(output))
    embedder = embedder if embedder is not None else OllamaEmbedder()
    report = {"schema_version": "four-comparison-v1", "run_status": "running", "cases": [],
              "metadata": {"model": DEFAULT_GENERATION_MODEL, "digest": pins[METHODS[0]],
                "embedding_model": index["model"], "embedding_digest": index["model_digest"],
                "top_k": 4, "order": "cyclic Latin rotation by case index", "methods": list(METHODS),
                "dataset_sha256": compute_canonical_sha256(dataset), "index_sha256": compute_canonical_sha256(index),
                "statistics_scope": "Returned validated traces only; failed-call usage may be unknown; reranker time included in its condition, preprocessing separate"}}
    failed = False
    for case_number, case in enumerate(dataset["cases"]):
        shift = case_number % len(METHODS)
        order = list(METHODS[shift:] + METHODS[:shift])
        row = {"id": case["id"], "query": case["query"], "condition_order": order, "conditions": {}}
        try:
            dense = search(index, case["query"], top_k=len(chunks), client=embedder)
            row["dense_ranking"] = [{"rank": h.rank, "chunk_id": h.chunk.chunk_id, "score": h.score, "page_number": h.chunk.page_number} for h in dense]
        except Exception as exc:
            failed = True
            row["conditions"] = {m: {"status": "error", "error": f"Retrieval: {exc}"} for m in order}
            report["cases"].append(row); save(report, output); continue
        for method in order:
            start = time.perf_counter()
            outcome = {}
            try:
                ranking = reranker.rank(case["query"], dense) if method == "reranker" else dense
                if method == "reranker": outcome["reranker_scores"] = reranker.last_scores
                hits = [SearchHit(i, h.score, h.chunk) for i, h in enumerate(ranking[:4], 1)]
                outcome["retrieved"] = [{"rank": h.rank, "chunk_id": h.chunk.chunk_id, "cosine_score": h.score} for h in hits]
                outcome["annotated_passages_hit"] = compute_annotated_hits(case["reference_chunk_ids"], [h.chunk.chunk_id for h in hits])
                grounder = grounders[method]
                if grounder.generator.get_digest() != pins[method]: raise RuntimeError("Digest changed before condition")
                result = grounder.answer(query=case["query"], hits=hits, expected_digest=pins[method])
                result = validate_grounded_result(result, hits, expected_digest=pins[method], expected_model=DEFAULT_GENERATION_MODEL)
                if grounder.generator.get_digest() != pins[method]: raise RuntimeError("Digest changed after condition")
                outcome.update(status=result["status"], full_result=result,
                    slot_coverage=compute_slot_coverage(case["slots"], [e["quote"] for e in result["grounding"]["selected_evidence"]]))
                if method == "structured": outcome["structured_selection"] = grounder.events[-1:] if grounder.events else []
            except Exception as exc:
                failed = True; outcome.update(status="error", error=str(exc))
            outcome["wall_seconds"] = time.perf_counter() - start
            row["conditions"][method] = outcome
            # Durable checkpoint after each condition, including errors.
            if row not in report["cases"]: report["cases"].append(row)
            report["summary"] = summarize(report["cases"])
            save(report, output)
            print(f"{case['id']} {method}: {outcome['status']}", file=sys.stderr, flush=True)
    report["summary"] = summarize(report["cases"])
    report["run_status"] = "error" if failed else "completed"
    save(report, output)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--extract-records", action="store_true")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX_PATH)
    parser.add_argument("--records", type=Path)
    parser.add_argument("--reranker-path", type=Path)
    parser.add_argument("--reranker-device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    try:
        args = parser.parse_args(argv)
        if args.generate == args.extract_records: raise ValueError("Choose exactly one of --generate / --extract-records")
        validate_project_path(args.output)
        index = load_index(validate_project_path(args.index))
        base = OllamaGrounder()
        if args.extract_records:
            report = extract_records(index, base, base.generator.get_digest())
        else:
            if args.records is None or args.reranker_path is None: raise ValueError("Need --records and --reranker-path")
            rows = json.loads(validate_project_path(args.records).read_text(encoding="utf-8"), parse_constant=_reject_constant)
            if (rows.get("schema_version") != "structured-records-v1" or rows.get("model") != base.model
                or rows.get("digest") != base.generator.get_digest() or rows.get("index_sha256") != compute_canonical_sha256(index)):
                raise ValueError("Structured data model/digest/index provenance mismatch")
            grounders = {"baseline": base, "selector": ContextSelectorGrounder(), "reranker": OllamaGrounder(),
                         "structured": StructuredGrounder(rows["records"]), "verifier": SpecialistVerifierGrounder()}
            reranker = LocalReranker(str(validate_project_path(args.reranker_path)), args.reranker_device)
            dataset = json.loads(validate_project_path(args.dataset).read_text(encoding="utf-8"), parse_constant=_reject_constant)
            report = run_comparison(index, dataset, grounders, reranker, output=args.output)
        save(report, args.output)
        print(json.dumps(report.get("summary", {"records": len(report.get("records", [])), "run_status": report["run_status"]}), ensure_ascii=False, allow_nan=False))
        return 0 if report["run_status"] == "completed" else 1
    except SystemExit as exc: return exc.code if isinstance(exc.code, int) else 1
    except Exception as exc: print(f"Error: {exc}", file=sys.stderr); return 1


if __name__ == "__main__": sys.exit(main())
