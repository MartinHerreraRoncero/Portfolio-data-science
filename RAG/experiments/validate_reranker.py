"""Validación pareada con preguntas nuevas; este módulo nunca carga gold."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
import time

from experiments.four_strategies import LocalReranker
from rag_pensiones.answer import DEFAULT_GENERATION_MODEL, _reject_constant
from rag_pensiones.embeddings import OllamaEmbedder
from rag_pensiones.grounding import OllamaGrounder, validate_grounded_result
from rag_pensiones.ingest import validate_project_path
from rag_pensiones.search import DEFAULT_INDEX_PATH, SearchHit, load_index, search, validate_index


def canonical_hash(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def validate_queries(data: dict) -> None:
    if not isinstance(data, dict) or set(data) != {"schema_version", "cases"}:
        raise ValueError("Queries must contain only schema_version and cases")
    if data["schema_version"] != "reranker-validation-queries-v1": raise ValueError("Wrong query schema")
    if not isinstance(data["cases"], list) or not 1 <= len(data["cases"]) <= 30: raise ValueError("Need 1..30 cases")
    seen = set()
    for case in data["cases"]:
        if not isinstance(case, dict) or set(case) != {"id", "query"}: raise ValueError("Query case fields invalid")
        if any(not isinstance(case[f], str) or not case[f].strip() for f in ["id", "query"]): raise ValueError("Invalid case strings")
        if len(case["query"]) > 1000 or case["id"] in seen: raise ValueError("Oversize query or duplicate ID")
        seen.add(case["id"])
    json.dumps(data, allow_nan=False)


def validate_reranking(dense, ranked, scores):
    if not isinstance(ranked, list) or len(ranked) != len(dense): raise ValueError("Rerank must preserve candidate count")
    originals = {h.chunk.chunk_id: h for h in dense}
    seen = set()
    for position, hit in enumerate(ranked, 1):
        if not isinstance(hit, SearchHit) or type(hit.rank) is not int or hit.rank != position:
            raise ValueError("Rerank ranks must be consecutive integers")
        original = originals.get(hit.chunk.chunk_id)
        if original is None or hit.chunk.chunk_id in seen or hit.chunk != original.chunk:
            raise ValueError("Rerank changed or duplicated source chunks")
        if isinstance(hit.score, bool) or not isinstance(hit.score, (int, float)) or not math.isfinite(hit.score) or hit.score != original.score:
            raise ValueError("Rerank changed original cosine score")
        seen.add(hit.chunk.chunk_id)
    if not isinstance(scores, list) or len(scores) != len(dense): raise ValueError("Missing reranker scores")
    if [s.get("chunk_id") for s in scores if isinstance(s, dict)] != [h.chunk.chunk_id for h in ranked]:
        raise ValueError("Reranker score IDs/order mismatch")
    for score in scores:
        value = score.get("score")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError("Non-finite or invalid reranker score")


def summarize(cases):
    summary = {}
    for method in ["baseline", "reranker"]:
        runs = [r["conditions"][method] for r in cases if method in r["conditions"]]
        valid = [r for r in runs if r["status"] in ["answered", "insufficient_evidence"]]
        summary[method] = {"attempts": len(runs), "valid_results": len(valid),
            "answered": sum(r["status"] == "answered" for r in runs),
            "insufficient_evidence": sum(r["status"] == "insufficient_evidence" for r in runs),
            "errors": sum(r["status"] == "error" for r in runs),
            "returned_chats": sum(len(r["full_result"]["grounding"]["trace"]) for r in valid),
            "mean_wall_seconds_valid_only": sum(r["wall_seconds"] for r in valid) / len(valid) if valid else None}
    return summary


def output_path(path):
    if path is None: return None
    resolved = validate_project_path(Path(path))
    if resolved.exists(): raise ValueError("Output already exists; choose a fresh JSON path")
    if resolved.suffix.lower() != ".json" or not resolved.parent.is_dir(): raise ValueError("Output needs existing parent and .json suffix")
    return resolved


def checkpoint(report, path):
    if path is not None: path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def run_validation(index: dict, queries: dict, reranker, grounder=None, embedder=None, output: Path | None = None):
    chunks = validate_index(index)
    if len(chunks) < 4: raise ValueError("Need at least four source chunks")
    validate_queries(queries)
    destination = output_path(output)
    grounder = grounder if grounder is not None else OllamaGrounder()
    if grounder.model != DEFAULT_GENERATION_MODEL or grounder.generator.model != DEFAULT_GENERATION_MODEL:
        raise ValueError("Grounder/generator must match Qwen9B")
    digest = grounder.generator.get_digest()
    if not isinstance(digest, str) or not digest.strip(): raise ValueError("Empty model digest")
    embedder = embedder if embedder is not None else OllamaEmbedder()
    report = {"schema_version": "reranker-new-validation-v1", "run_status": "running", "cases": [],
        "metadata": {"model": grounder.model, "digest": digest, "embedding_model": index["model"],
            "embedding_digest": index["model_digest"], "top_k": 4, "queries_canonical_sha256": canonical_hash(queries),
            "index_canonical_sha256": canonical_hash(index), "order": "alternating baseline/reranker by case index",
            "gold_access": "none; semantic grading and reference retrieval metrics are separate",
            "statistics_scope": "returned validated results only; failed-call usage may be unknown"}}
    failed = False
    for number, case in enumerate(queries["cases"]):
        order = ["baseline", "reranker"] if number % 2 == 0 else ["reranker", "baseline"]
        row = dict(case, condition_order=order, conditions={})
        report["cases"].append(row)
        try:
            dense = search(index, case["query"], top_k=len(chunks), client=embedder)
            row["dense_ranking"] = [{"rank": h.rank, "chunk_id": h.chunk.chunk_id, "cosine_score": h.score, "page_number": h.chunk.page_number} for h in dense]
        except Exception as exc:
            failed = True
            row["conditions"] = {m: {"status": "error", "error": f"Retrieval: {exc}"} for m in order}
            report["summary"] = summarize(report["cases"]);checkpoint(report, destination);continue
        for method in order:
            started = time.perf_counter()
            outcome = {"rerank_seconds": 0.0}
            try:
                ranked = dense
                if method == "reranker":
                    rerank_start = time.perf_counter()
                    try: ranked = reranker.rank(case["query"], copy.deepcopy(dense))
                    finally: outcome["rerank_seconds"] = time.perf_counter() - rerank_start
                    validate_reranking(dense, ranked, reranker.last_scores)
                    outcome["reranker_scores"] = copy.deepcopy(reranker.last_scores)
                hits = [SearchHit(i, h.score, h.chunk) for i, h in enumerate(ranked[:4], 1)]
                outcome["retrieved"] = [{"rank": h.rank, "chunk_id": h.chunk.chunk_id, "cosine_score": h.score, "page_number": h.chunk.page_number} for h in hits]
                if grounder.generator.get_digest() != digest: raise RuntimeError("Digest changed before answer")
                result = grounder.answer(query=case["query"], hits=hits, expected_digest=digest)
                result = validate_grounded_result(result, hits, expected_digest=digest, expected_model=DEFAULT_GENERATION_MODEL)
                if grounder.generator.get_digest() != digest: raise RuntimeError("Digest changed after answer")
                outcome.update(status=result["status"], full_result=copy.deepcopy(result))
            except Exception as exc:
                failed = True;outcome.update(status="error", error=str(exc))
            outcome["wall_seconds"] = time.perf_counter() - started
            row["conditions"][method] = outcome
            report["summary"] = summarize(report["cases"]);checkpoint(report, destination)
            print(f"{case['id']} {method}: {outcome['status']}", file=sys.stderr, flush=True)
    report["run_status"] = "error" if failed else "completed"
    checkpoint(report, destination)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX_PATH)
    parser.add_argument("--reranker-path", type=Path, required=True)
    parser.add_argument("--reranker-device", choices=["cuda", "cpu"], default="cuda")
    try:
        args = parser.parse_args(argv)
        queries = json.loads(validate_project_path(args.queries).read_text(encoding="utf-8-sig"), parse_constant=_reject_constant)
        validate_queries(queries)
        index = load_index(validate_project_path(args.index))
        if len(validate_index(index)) < 4: raise ValueError("Need at least four source chunks")
        output_path(args.output)
        reranker = LocalReranker(str(validate_project_path(args.reranker_path)), args.reranker_device)
        result = run_validation(index, queries, reranker, output=args.output)
        print(json.dumps(result["summary"], ensure_ascii=False, allow_nan=False))
        return 0 if result["run_status"] == "completed" else 1
    except SystemExit as exc: return exc.code if isinstance(exc.code, int) else 1
    except Exception as exc: print(f"Error: {exc}", file=sys.stderr);return 1


if __name__ == "__main__": sys.exit(main())
