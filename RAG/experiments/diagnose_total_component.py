"""Ablación de recuperación y controles de citas para un caso conocido, no un test reservado."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

from rag_pensiones.answer import DEFAULT_GENERATION_MODEL, collapse_spaces, validate_chat_stats
from rag_pensiones.grounding import OllamaGrounder, validate_grounded_result, validate_verification_payload
from rag_pensiones.ingest import validate_project_path
from rag_pensiones.search import SearchHit, load_index, search, validate_index


def _hash(value):
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def validate_case(case, chunks):
    """Valida las referencias manuales sin transmitir etiquetas al modelo."""
    _hash(case)
    keys = {"schema_version", "query", "top_k", "oracle_chunk_id", "additional_reference_chunk_ids",
            "annotation", "verifier_controls"}
    if not isinstance(case, dict) or set(case) != keys:
        raise ValueError("Case keys mismatch")
    if case["schema_version"] != "total-component-diagnostic-v1":
        raise ValueError("Unsupported diagnostic schema")
    if not isinstance(case["query"], str) or not 1 <= len(case["query"].strip()) <= 1000:
        raise ValueError("Invalid query")
    if type(case["top_k"]) is not int or not 1 <= case["top_k"] <= len(chunks):
        raise ValueError("Invalid top_k")
    if not isinstance(case["annotation"], str) or not case["annotation"].strip():
        raise ValueError("Annotation required")
    oracle = case["oracle_chunk_id"]
    if not isinstance(oracle, str) or oracle not in chunks:
        raise ValueError("Unknown oracle chunk")
    refs = case["additional_reference_chunk_ids"]
    if (not isinstance(refs, list) or any(not isinstance(r, str) or r not in chunks for r in refs)
            or len(set(refs)) != len(refs) or oracle in refs):
        raise ValueError("Invalid additional references")
    controls = case["verifier_controls"]
    if not isinstance(controls, list) or not 1 <= len(controls) <= 4:
        raise ValueError("Expected one to four verifier controls")
    seen = set()
    for control in controls:
        if not isinstance(control, dict) or set(control) != {"id", "claim", "chunk_id", "quote", "expected_supported"}:
            raise ValueError("Control keys mismatch")
        cid = control["id"]
        if not isinstance(cid, str) or not cid.strip() or cid in seen:
            raise ValueError("Invalid or duplicate control id")
        seen.add(cid)
        source = control["chunk_id"]
        if not isinstance(source, str) or source not in chunks:
            raise ValueError("Unknown control source")
        if not isinstance(control["claim"], str) or not 1 <= len(control["claim"].strip()) <= 300:
            raise ValueError("Invalid control claim")
        quote = control["quote"]
        if (not isinstance(quote, str) or not 12 <= len(quote) <= 700
                or len(collapse_spaces(quote)) < 12
                or collapse_spaces(quote) not in collapse_spaces(chunks[source].text)):
            raise ValueError("Invalid or non-literal control quote")
        if type(control["expected_supported"]) is not bool:
            raise ValueError("Expected support must be boolean")


def run_diagnosis(index: dict, case: dict, grounder=None, embedder=None) -> dict:
    chunks = {c.chunk_id: c for c, _ in validate_index(index)}
    validate_case(case, chunks)
    grounder = grounder or OllamaGrounder()
    if (grounder.model != DEFAULT_GENERATION_MODEL
            or grounder.generator.model != DEFAULT_GENERATION_MODEL):
        raise ValueError("Diagnostic requires the same Qwen9B generator throughout")
    digest = grounder.generator.get_digest()
    if not isinstance(digest, str) or not digest.strip():
        raise ValueError("Missing generator digest")
    started = time.perf_counter()
    ranked = search(index, case["query"], top_k=len(chunks), client=embedder)
    by_id = {h.chunk.chunk_id: h for h in ranked}
    oracle = by_id[case["oracle_chunk_id"]]
    k = case["top_k"]
    actual = ranked[:k]
    injected = actual if oracle in actual else actual[:-1] + [oracle]
    # Añadir la referencia sin retirar el distractor controla el efecto de la sustitución.
    conditions = [("actual_top_k", actual), ("oracle_only", [oracle]),
                  ("oracle_injected_top_k", injected), ("actual_top_k_plus_one", ranked[:k + 1])]
    outputs, cache, measured = [], {}, []

    def describe(h):
        return {"chunk_id": h.chunk.chunk_id, "original_rank": h.rank, "score": h.score,
                "page_number": h.chunk.page_number, "source_file": h.chunk.source_file}

    for name, selected in conditions:
        source_key = tuple(h.chunk.chunk_id for h in selected)
        hits = [SearchHit(i, h.score, h.chunk) for i, h in enumerate(selected, start=1)]
        if source_key in cache:
            previous = cache[source_key]
            result, latency, reused = previous["result"], 0.0, previous["condition"]
        else:
            tick = time.perf_counter()
            result = grounder.answer(case["query"], hits, expected_digest=digest)
            validate_grounded_result(result, hits, digest, expected_model=grounder.model)
            latency, reused = time.perf_counter() - tick, None
            measured.append(result)
        entry = {"condition": name, "sources": [dict(describe(h), source_id=f"S{i}")
                  for i, h in enumerate(selected, start=1)], "result": result,
                 "latency_seconds": latency, "reused_from": reused}
        outputs.append(entry)
        if reused is None:
            cache[source_key] = entry

    controls = []
    for control in case["verifier_controls"]:
        claims = [{"claim_id": "C1", "text": control["claim"], "evidence_ids": ["E1"]}]
        evidence = [{"evidence_id": "E1", "source_id": "S1", "quote": control["quote"]}]
        tick = time.perf_counter()
        result = grounder.verify(claims, evidence, expected_digest=digest)
        latency = time.perf_counter() - tick
        if not isinstance(result, dict) or set(result) != {"verdicts", "digest", "prompt_sha256", "stats"}:
            raise ValueError("Invalid verify result keys")
        if result["digest"] != digest:
            raise ValueError("Verifier digest changed")
        phash = result["prompt_sha256"]
        if not isinstance(phash, str) or len(phash) != 64 or any(c not in "0123456789abcdefABCDEF" for c in phash):
            raise ValueError("Invalid verifier prompt hash")
        validate_chat_stats(result["stats"])
        validated = validate_verification_payload({"verdicts": result["verdicts"]}, claims, evidence)
        accepted = validated["verdicts"][0]["verdict"] == "supported"
        controls.append({"id": control["id"], "claim": control["claim"], "evidence": evidence,
                         "chunk_id": control["chunk_id"], "expected_supported": control["expected_supported"],
                         "predicted_supported": accepted, "matches_reference": accepted == control["expected_supported"],
                         "latency_seconds": latency, "result": result})
    if grounder.generator.get_digest() != digest:
        raise ValueError("Generator digest changed during experiment")
    all_stats = [r["stats"] for r in measured] + [c["result"]["stats"] for c in controls]
    stats = {key: sum(s.get(key, 0) for s in all_stats) for key in set().union(*all_stats)}
    report = {"schema_version": "total-component-diagnostic-results-v1", "run_status": "completed",
              "query": case["query"], "top_k": k, "model": grounder.model, "model_digest": digest,
              "embedding_model": index["model"], "embedding_digest": index["model_digest"],
              "case_sha256": _hash(case), "index_sha256": _hash(index), "hash_kind": "canonical_json",
              "reference_annotation": case["annotation"], "planning_used": False,
              "oracle_rank": oracle.rank, "oracle_in_top_k": oracle.rank <= k,
              "additional_reference_ranks": {cid: by_id[cid].rank for cid in case["additional_reference_chunk_ids"]},
              "ranking": [describe(h) for h in ranked], "conditions": outputs, "verifier_controls": controls,
              "calls": {"searches": 1, "answers": len(measured), "verifier_controls": len(controls),
                        "chats": sum(len(r["grounding"]["trace"]) for r in measured) + len(controls)},
              "stats": stats, "duration_seconds": time.perf_counter() - started}
    _hash(report)
    return report


def format_text(report):
    lines = ["DIAGNÓSTICO DE DESARROLLO: GASTO TOTAL / COMPONENTES",
             f"Consulta: {report['query']}",
             f"Pasaje de referencia: rango {report['oracle_rank']}; top-{report['top_k']}: {report['oracle_in_top_k']}",
             "Condición | Estado | Afirmaciones (corrección pendiente de revisión manual)"]
    for row in report["conditions"]:
        text = " ; ".join(c["text"] for c in row["result"]["claims"])
        lines.append(f"{row['condition']} | {row['result']['status']} | {text}")
    lines.append("Controles | Aceptación esperada | Aceptación del modelo")
    for row in report["verifier_controls"]:
        lines.append(f"{row['id']} | {row['expected_supported']} | {row['predicted_supported']}")
    lines.append("El oráculo es un control manual, no una mejora productiva ni una estimación de calidad general.")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, default=Path("evaluation/total_component_diagnostic_case.json"))
    parser.add_argument("--index", type=Path, default=Path("data/processed/airef_2025_embeddings.json"))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    try:
        case = json.loads(validate_project_path(args.case).read_text(encoding="utf-8-sig"))
        report = run_diagnosis(load_index(validate_project_path(args.index)), case)
        print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) if args.json else format_text(report))
        return 0
    except Exception as exc:
        print(f"Diagnostic failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
