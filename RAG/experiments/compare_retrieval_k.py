"""Comparación de recuperación k=4 vs k=5 sobre consultas de desarrollo de pensiones AIReF."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

from rag_pensiones.answer import DEFAULT_GENERATION_MODEL, _reject_constant
from rag_pensiones.embeddings import OllamaEmbedder
from rag_pensiones.grounding import OllamaGrounder, validate_grounded_result
from rag_pensiones.ingest import validate_project_path
from rag_pensiones.search import DEFAULT_INDEX_PATH, SearchHit, load_index, search, validate_index

DATASET_SCHEMA_VERSION = "k4-k5-development-v1"
EXPERIMENT_SCHEMA_VERSION = "retrieval-k-comparison-v1"
DEFAULT_DATASET_PATH = Path("evaluation/k4_k5_dev.json")
MAX_CASES = 20
MAX_QUERY_CHARS = 1000
REQUIRED_SLOTS = ("entity", "period", "unit", "value")


def compute_canonical_sha256(data: dict) -> str:
    canonical = json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def normalize_for_matching(text: str) -> str:
    return " ".join(text.lower().split())


def compute_slot_coverage(slots: dict, quotes: list[str]) -> dict:
    norm_quotes = " ".join(normalize_for_matching(q) for q in quotes)
    fields: dict[str, bool | None] = {}
    applicable = 0
    covered = 0
    for field in REQUIRED_SLOTS:
        val = slots.get(field)
        if val is None:
            fields[field] = None
        else:
            applicable += 1
            is_cov = any(normalize_for_matching(alt) in norm_quotes for alt in val)
            fields[field] = is_cov
            if is_cov:
                covered += 1
    return {
        "fields": fields,
        "applicable_slots": applicable,
        "covered_slots": covered,
        "all_applicable_covered": (covered == applicable) if applicable > 0 else False,
    }


def compute_annotated_hits(reference_chunk_ids: list[str], retrieved_chunk_ids: list[str]) -> dict:
    hits_found = [ref for ref in reference_chunk_ids if ref in retrieved_chunk_ids]
    return {
        "annotated_total": len(reference_chunk_ids),
        "annotated_hits": len(hits_found),
        "hit_chunk_ids": hits_found,
        "all_annotated_hit": len(hits_found) == len(reference_chunk_ids),
    }


def validate_dataset(dataset: dict, index_chunk_ids: set[str]) -> None:
    if isinstance(dataset, bool) or not isinstance(dataset, dict):
        raise ValueError(f"Dataset must be a dict, got {type(dataset).__name__}")
    json.dumps(dataset, allow_nan=False)
    if set(dataset.keys()) != {"schema_version", "annotation", "cases"}:
        raise ValueError(f"Dataset keys mismatch: {sorted(dataset.keys())}")
    if dataset["schema_version"] != DATASET_SCHEMA_VERSION:
        raise ValueError(f"Invalid schema_version: expected {DATASET_SCHEMA_VERSION!r}")
    if not isinstance(dataset["annotation"], str) or not dataset["annotation"].strip():
        raise ValueError("annotation must be a non-empty string")
    cases = dataset["cases"]
    if isinstance(cases, bool) or not isinstance(cases, list) or not (1 <= len(cases) <= MAX_CASES):
        raise ValueError(f"cases must be a list containing 1..{MAX_CASES} items")
    seen_ids: set[str] = set()
    req_keys = {"id", "family", "query", "reference_chunk_ids", "slots", "reference_answer"}
    for idx, c in enumerate(cases):
        if isinstance(c, bool) or not isinstance(c, dict) or set(c.keys()) != req_keys:
            raise ValueError(f"Case {idx} keys mismatch or invalid type")
        cid = c["id"]
        if isinstance(cid, bool) or not isinstance(cid, str) or not cid.strip() or cid in seen_ids:
            raise ValueError(f"Invalid or duplicate case id at index {idx}: {cid!r}")
        seen_ids.add(cid)
        for sf in ("family", "reference_answer"):
            if isinstance(c[sf], bool) or not isinstance(c[sf], str) or not c[sf].strip():
                raise ValueError(f"Case {cid} invalid non-empty string field: {sf}")
        q = c["query"]
        if isinstance(q, bool) or not isinstance(q, str) or not q.strip() or len(q) > MAX_QUERY_CHARS:
            raise ValueError(f"Case {cid} query must be 1..{MAX_QUERY_CHARS} chars")
        refs = c["reference_chunk_ids"]
        if isinstance(refs, bool) or not isinstance(refs, list) or len(refs) == 0:
            raise ValueError(f"Case {cid} reference_chunk_ids must be non-empty list")
        if len(set(refs)) != len(refs):
            raise ValueError(f"Case {cid} duplicate reference_chunk_ids")
        for ref in refs:
            if isinstance(ref, bool) or not isinstance(ref, str) or not ref.strip():
                raise ValueError(f"Case {cid} reference_chunk_id must be non-empty string")
            if ref not in index_chunk_ids:
                raise ValueError(f"Case {cid} reference_chunk_id {ref!r} not in index")
        slots = c["slots"]
        if isinstance(slots, bool) or not isinstance(slots, dict) or set(slots.keys()) != set(REQUIRED_SLOTS):
            raise ValueError(f"Case {cid} slots keys mismatch: expected {sorted(REQUIRED_SLOTS)}")
        for k in ("entity", "value"):
            v = slots[k]
            if v is None or isinstance(v, bool) or not isinstance(v, list) or len(v) == 0:
                raise ValueError(f"Case {cid} slot {k!r} is mandatory non-empty list")
            if not all(isinstance(s, str) and s.strip() for s in v):
                raise ValueError(f"Case {cid} slot {k!r} elements must be non-empty strings")
        for k in ("period", "unit"):
            v = slots[k]
            if v is not None:
                if isinstance(v, bool) or not isinstance(v, list) or len(v) == 0:
                    raise ValueError(f"Case {cid} slot {k!r} must be None or non-empty list")
                if not all(isinstance(s, str) and s.strip() for s in v):
                    raise ValueError(f"Case {cid} slot {k!r} elements must be non-empty strings")


def _compute_summary(case_results: list[dict]) -> dict:
    summary: dict[str, Any] = {}
    for k_key in ("k4", "k5"):
        k_cases = [c["conditions"][k_key] for c in case_results if k_key in c["conditions"]]
        valid = [c for c in k_cases if c.get("status") in ("answered", "insufficient_evidence")]
        latencies = [c["latency_ms"] for c in valid if "latency_ms" in c]
        slot_rates = {}
        for f in REQUIRED_SLOTS:
            app = sum(1 for c in valid if c.get("slot_coverage", {}).get("fields", {}).get(f) is not None)
            cov = sum(1 for c in valid if c.get("slot_coverage", {}).get("fields", {}).get(f) is True)
            slot_rates[f] = {"applicable": app, "covered": cov, "rate": round(cov / app, 4) if app > 0 else None}
        summary[k_key] = {
            "total_evaluated": len(k_cases),
            "valid_executions": len(valid),
            "answered_count": sum(1 for c in valid if c["status"] == "answered"),
            "insufficient_evidence_count": sum(1 for c in valid if c["status"] == "insufficient_evidence"),
            "error_count": sum(1 for c in k_cases if c.get("status") == "error"),
            "repair_attempts_count": sum(c.get("repair_attempts", 0) for c in valid),
            "all_slots_covered_count": sum(1 for c in valid if c.get("slot_coverage", {}).get("all_applicable_covered")),
            "retrievals_evaluated": sum(1 for c in k_cases if "annotated_passages_hit" in c),
            "any_annotated_hit_count": sum(1 for c in k_cases if c.get("annotated_passages_hit", {}).get("annotated_hits", 0) > 0),
            "all_annotated_hit_count": sum(1 for c in k_cases if c.get("annotated_passages_hit", {}).get("all_annotated_hit")),
            "avg_latency_ms": round(sum(latencies) / len(latencies), 2) if latencies else None,
            "slot_coverage_rates": slot_rates,
        }
    lat4, lat5 = summary["k4"]["avg_latency_ms"], summary["k5"]["avg_latency_ms"]
    summary["comparison"] = {
        "answered_diff_k5_minus_k4": summary["k5"]["answered_count"] - summary["k4"]["answered_count"],
        "all_slots_covered_diff": summary["k5"]["all_slots_covered_count"] - summary["k4"]["all_slots_covered_count"],
        "all_annotated_hit_diff": summary["k5"]["all_annotated_hit_count"] - summary["k4"]["all_annotated_hit_count"],
        "latency_delta_ms": round(lat5 - lat4, 2) if (lat4 is not None and lat5 is not None) else None,
    }
    return summary


def run_comparison(index: dict, dataset: dict, grounder: Any = None, embedder: Any = None) -> dict:
    validate_index(index)
    chunks = index.get("chunks", [])
    if len(chunks) < 5:
        raise ValueError(f"Index must contain at least 5 chunks, found {len(chunks)}")
    index_chunk_ids = {c["chunk_id"] for c in chunks}
    validate_dataset(dataset, index_chunk_ids)

    if embedder is None:
        embedder = OllamaEmbedder()
    if grounder is None:
        grounder = OllamaGrounder(model=DEFAULT_GENERATION_MODEL)
    if getattr(grounder, "model", None) != DEFAULT_GENERATION_MODEL:
        raise ValueError(f"Grounder model must be {DEFAULT_GENERATION_MODEL!r}, got {getattr(grounder, 'model', None)!r}")
    if getattr(grounder.generator, "model", None) != DEFAULT_GENERATION_MODEL:
        raise ValueError("Generator model must match the experiment model")

    pinned_digest = grounder.generator.get_digest()
    if not isinstance(pinned_digest, str) or not pinned_digest.strip():
        raise ValueError("A non-empty model digest is required")
    case_results: list[dict] = []
    has_global_error = False
    total_searches = 0
    total_valid_answers = 0
    total_chat_calls = 0
    valid_stats: list[dict] = []

    for case_idx, case in enumerate(dataset["cases"]):
        cid = case["id"]
        q = case["query"]
        ref_ids = case["reference_chunk_ids"]
        slots = case["slots"]

        try:
            hits5 = search(index=index, query=q, top_k=len(chunks), client=embedder)
            total_searches += 1
        except Exception as exc:
            has_global_error = True
            case_results.append({
                "id": cid, "family": case["family"], "query": q,
                "condition_order": ["k4", "k5"] if case_idx % 2 == 0 else ["k5", "k4"],
                "conditions": {
                    "k4": {"k": 4, "order": 1 if case_idx % 2 == 0 else 2, "status": "error", "error": f"Search: {exc}"},
                    "k5": {"k": 5, "order": 2 if case_idx % 2 == 0 else 1, "status": "error", "error": f"Search: {exc}"},
                },
            })
            continue

        hits_by_k = {
            4: [SearchHit(rank=i, score=h.score, chunk=h.chunk) for i, h in enumerate(hits5[:4], start=1)],
            5: [SearchHit(rank=i, score=h.score, chunk=h.chunk) for i, h in enumerate(hits5[:5], start=1)],
        }
        ranking_meta = [
            {"rank": h.rank, "chunk_id": h.chunk.chunk_id, "score": h.score, "source_file": h.chunk.source_file, "page_number": h.chunk.page_number}
            for h in hits5
        ]
        k_order = [4, 5] if case_idx % 2 == 0 else [5, 4]
        conditions: dict[str, dict] = {}

        for order_idx, k_val in enumerate(k_order, start=1):
            hits_k = hits_by_k[k_val]
            ann_hits = compute_annotated_hits(ref_ids, [h.chunk.chunk_id for h in hits_k])
            t0 = time.perf_counter()
            try:
                curr_digest = grounder.generator.get_digest()
                if curr_digest != pinned_digest:
                    raise RuntimeError(f"Digest changed: {pinned_digest} -> {curr_digest}")
                ans = grounder.answer(query=q, hits=hits_k, expected_digest=pinned_digest)
                lat_ms = (time.perf_counter() - t0) * 1000.0
                val_res = validate_grounded_result(ans, hits_k, expected_digest=pinned_digest, expected_model=DEFAULT_GENERATION_MODEL)
                sel_quotes = [e["quote"] for e in val_res["grounding"]["selected_evidence"]]
                slot_cov = compute_slot_coverage(slots, sel_quotes)
                cits = [{"claim": c.get("text", ""), "source_id": e.get("source_id"), "quote": e.get("quote")}
                        for c in val_res.get("claims", []) for e in c.get("evidence", [])]
                total_valid_answers += 1
                total_chat_calls += len(val_res["grounding"]["trace"])
                valid_stats.append(val_res["stats"])

                conditions[f"k{k_val}"] = {
                    "k": k_val, "order": order_idx, "status": val_res["status"], "reason": val_res["reason"],
                    "latency_ms": round(lat_ms, 2), "repair_attempts": val_res["grounding"]["repair_attempts"],
                    "repairs": val_res["grounding"]["repairs"], "citations": cits, "slot_coverage": slot_cov,
                    "annotated_passages_hit": ann_hits, "prompt_sha256": val_res["prompt_sha256"],
                    "digest": val_res["digest"], "stats": val_res["stats"], "claims": val_res["claims"],
                    "full_result": val_res,
                }
            except Exception as exc:
                lat_ms = (time.perf_counter() - t0) * 1000.0
                has_global_error = True
                conditions[f"k{k_val}"] = {"k": k_val, "order": order_idx, "status": "error", "error": str(exc), "latency_ms": round(lat_ms, 2), "annotated_passages_hit": ann_hits}

        case_results.append({
            "id": cid, "family": case["family"], "query": q, "ranking_metadata": ranking_meta,
            "condition_order": [f"k{k}" for k in k_order], "conditions": conditions,
        })

    agg_stats = {
        "total_searches": total_searches, "total_valid_answers": total_valid_answers, "total_chat_calls": total_chat_calls,
        "prompt_eval_count": sum(s.get("prompt_eval_count", 0) for s in valid_stats),
        "eval_count": sum(s.get("eval_count", 0) for s in valid_stats),
        "eval_duration": sum(s.get("eval_duration", 0) for s in valid_stats),
        "total_duration": sum(s.get("total_duration", 0) for s in valid_stats),
    }
    for key in ("load_duration", "prompt_eval_duration"):
        if any(key in s for s in valid_stats):
            agg_stats[key] = sum(s.get(key, 0) for s in valid_stats)
    return {
        "schema_version": EXPERIMENT_SCHEMA_VERSION,
        "run_status": "error" if has_global_error else "completed",
        "metadata": {
            "model": DEFAULT_GENERATION_MODEL, "model_digest": pinned_digest,
            "embedder_model": index["model"], "embedder_digest": index["model_digest"],
            "dataset_canonical_sha256": compute_canonical_sha256(dataset),
            "index_canonical_sha256": compute_canonical_sha256(index),
            "total_cases": len(dataset["cases"]),
            "statistics_scope": "successful_validated_results_only; failed-condition usage may be unknown",
        },
        "cases": case_results, "summary": _compute_summary(case_results), "stats": agg_stats,
    }


def _format_text_report(report: dict) -> str:
    s4, s5 = report.get("summary", {}).get("k4", {}), report.get("summary", {}).get("k5", {})
    return "\n".join([
        "=" * 68, "REPORTE COMPARATIVO DE RECUPERACIÓN (k=4 vs k=5)", "=" * 68,
        f"Estado global: {report.get('run_status')} | Casos evaluados: {report.get('metadata', {}).get('total_cases')}",
        f"Modelo: {report.get('metadata', {}).get('model')} (digest: {report.get('metadata', {}).get('model_digest', '')[:12]}...)",
        "-" * 68,
        "Métrica                              k=4            k=5",
        "-" * 68,
        f"Válidas / Errores:                   {s4.get('valid_executions')}/{s4.get('error_count')}          {s5.get('valid_executions')}/{s5.get('error_count')}",
        f"Estado answered (no prueba acierto): {s4.get('answered_count')}              {s5.get('answered_count')}",
        f"Abstenciones (insufficient):         {s4.get('insufficient_evidence_count')}              {s5.get('insufficient_evidence_count')}",
        f"Algún pasaje anotado recuperado:     {s4.get('any_annotated_hit_count')}              {s5.get('any_annotated_hit_count')}",
        f"Slots cubiertos (todos aplicables):  {s4.get('all_slots_covered_count')}              {s5.get('all_slots_covered_count')}",
        f"Reparaciones ejecutadas:             {s4.get('repair_attempts_count')}              {s5.get('repair_attempts_count')}",
        f"Latencia promedio (ms):              {s4.get('avg_latency_ms')}         {s5.get('avg_latency_ms')}",
        "=" * 68,
        "Nota metodológica: Coocurrencia en selected_evidence no demuestra entailment semántico.",
    ])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Comparación acotada de k=4 vs k=5.")
    parser.add_argument("--json", action="store_true", help="Emitir reporte en JSON UTF-8.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH, help="Ruta al dataset.")
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX_PATH, help="Ruta al índice.")
    try:
        args = parser.parse_args(argv)
        dataset_path = validate_project_path(args.dataset)
        index_path = validate_project_path(args.index)
        if not dataset_path.is_file():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")
        index_data = load_index(index_path)
        ds_bytes = dataset_path.read_bytes()
        dataset_data = json.loads(ds_bytes.decode("utf-8"), parse_constant=_reject_constant)
        report = run_comparison(index=index_data, dataset=dataset_data)
        if args.json:
            sys.stdout.write(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        else:
            sys.stdout.write(_format_text_report(report) + "\n")
        return 0 if report.get("run_status") == "completed" else 1
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 1
    except Exception as exc:
        sys.stderr.write(f"Error: {exc}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
