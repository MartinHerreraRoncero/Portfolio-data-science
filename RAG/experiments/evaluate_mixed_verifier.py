"""Paired 16-run RAG evaluation comparing legacy 9B verification vs selected 27B Q3_K_M verification.

Strictly offline in testing, reproducible, zero live model downloads or settings modifications.
Execution is bound to a frozen protocol document without gold-set reading or evaluation.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

from experiments.compare_verifier_quantization import QuantizationOllamaTransport
from experiments.preserve_evidence import validate_catalog
from experiments.preserved_q3_verifier import (
    Q3VerifierClient,
    VERIFIER_Q3_DIGEST,
    VERIFIER_Q3_MODEL,
    VERIFIER_Q3_PARAMETERS,
)
from experiments.query_preserved import answer_preserved, format_answer
from experiments.source_bound_trial import RecordedGenerator, checkpoint
from experiments.validate_reranker import canonical_hash, output_path, validate_queries
from rag_pensiones.answer import DEFAULT_GENERATION_MODEL
from rag_pensiones.grounding import DEFAULT_PARAMETERS
from rag_pensiones.ingest import PROJECT_ROOT, validate_project_path
from rag_pensiones.qwen_embeddings import MODEL, QwenEmbedder, _json
from rag_pensiones.search import DEFAULT_INDEX_PATH, load_index, validate_index

SCHEMA_VERSION = "mixed-verifier-pipeline-v1"
ARMS = ("legacy", "q3")
EXPECTED_CASE_COUNT = 8
TOTAL_EVENTS = 16

REQUIRED_FROZEN_MODULES = [
    "rag_pensiones/grounding.py",
    "rag_pensiones/answer.py",
    "rag_pensiones/search.py",
    "rag_pensiones/qwen_embeddings.py",
    "rag_pensiones/ingest.py",
    "experiments/preserved_q3_verifier.py",
    "experiments/query_preserved.py",
    "experiments/compare_system.py",
    "experiments/preserve_evidence.py",
    "experiments/source_bound_trial.py",
    "experiments/evaluate_mixed_verifier.py",
    "evaluation/verifier_capacity_2026-10-26/ollama_defaults_v0.40.11.json",
]
REQUIRED_FROZEN_MODULES = sorted(set(REQUIRED_FROZEN_MODULES) | {
    path.relative_to(PROJECT_ROOT).as_posix()
    for folder in ("rag_pensiones", "experiments")
    for path in (PROJECT_ROOT / folder).rglob("*.py")
})


def validate_schedule(queries: dict[str, Any], schedule: list[dict[str, Any]]) -> None:
    """Validates the exact 16-event schedule: 8 legacy then 8 q3 in identical query order."""
    validate_queries(queries)
    cases = queries.get("cases", [])
    if not isinstance(cases, list) or len(cases) != EXPECTED_CASE_COUNT:
        raise ValueError(f"Expected exactly {EXPECTED_CASE_COUNT} query cases, got {len(cases)}")

    expected_ids = [c["id"] for c in cases]
    if expected_ids != [f"b{i:02d}" for i in range(1, 9)]:
        raise ValueError("Require the original ordered b01..b08 cases")
    for cid in expected_ids:
        if isinstance(cid, bool) or not isinstance(cid, str) or not cid.strip():
            raise ValueError(f"Invalid case ID: {cid!r}")

    if not isinstance(schedule, list) or len(schedule) != TOTAL_EVENTS:
        raise ValueError(f"Schedule must contain exactly {TOTAL_EVENTS} items, got {len(schedule)}")

    expected_events = [("legacy", cid) for cid in expected_ids] + [("q3", cid) for cid in expected_ids]
    actual_events = []

    for idx, item in enumerate(schedule):
        if not isinstance(item, dict) or isinstance(item, bool):
            raise ValueError(f"Schedule item at index {idx} must be a dict")
        allowed_keys = {"arm", "id"}
        if set(item.keys()) != allowed_keys:
            raise ValueError(f"Schedule item {idx} keys mismatch: {sorted(item.keys())}")

        arm = item.get("arm")
        cid = item.get("id")

        if isinstance(arm, bool) or arm not in ARMS:
            raise ValueError(f"Schedule item {idx} invalid arm: {arm!r}. Must be in {ARMS}")
        if isinstance(cid, bool) or not isinstance(cid, str) or cid not in expected_ids:
            raise ValueError(f"Schedule item {idx} invalid case ID: {cid!r}")

        actual_events.append((arm, cid))

    if actual_events != expected_events:
        raise ValueError(
            "Schedule ordering mismatch: required all 8 legacy cases then all 8 q3 cases in identical order"
        )


def validate_protocol(protocol: dict[str, Any]) -> None:
    """Strictly validates the protocol freeze document, dependencies, and parameters before execution."""
    if not isinstance(protocol, dict) or isinstance(protocol, bool):
        raise ValueError("Protocol must be a dict")

    required_keys = {
        "schema_version",
        "server_version",
        "generator_digest",
        "verifier_digest",
        "embedding_digest",
        "generator_parameters",
        "verifier_parameters",
        "input_paths",
        "input_hashes",
        "frozen_files",
        "schedule",
    }
    missing_keys = required_keys - set(protocol.keys())
    if missing_keys:
        raise ValueError(f"Protocol missing required keys: {sorted(missing_keys)}")

    if protocol["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"Protocol schema mismatch: expected {SCHEMA_VERSION!r}, got {protocol['schema_version']!r}")

    if protocol["server_version"] != "0.40.11":
        raise ValueError(f"Server version mismatch: expected '0.40.11', got {protocol['server_version']!r}")

    for dig_key in ("generator_digest", "verifier_digest", "embedding_digest"):
        d_val = protocol[dig_key]
        if isinstance(d_val, bool) or not isinstance(d_val, str) or len(d_val) != 64 or any(c not in "0123456789abcdef" for c in d_val.lower()):
            raise ValueError(f"Invalid {dig_key}: expected 64-character hex string, got {d_val!r}")

    if protocol["verifier_digest"] != VERIFIER_Q3_DIGEST:
        raise ValueError(
            f"Verifier digest mismatch against frozen constant: expected {VERIFIER_Q3_DIGEST!r}, "
            f"got {protocol['verifier_digest']!r}"
        )

    if canonical_hash(protocol["generator_parameters"]) != canonical_hash(DEFAULT_PARAMETERS):
        raise ValueError(
            f"Generator parameters mismatch: expected {DEFAULT_PARAMETERS}, got {protocol['generator_parameters']}"
        )

    if canonical_hash(protocol["verifier_parameters"]) != canonical_hash(VERIFIER_Q3_PARAMETERS):
        raise ValueError(
            f"Verifier parameters mismatch: expected {VERIFIER_Q3_PARAMETERS}, got {protocol['verifier_parameters']}"
        )

    input_paths = protocol["input_paths"]
    if not isinstance(input_paths, dict) or set(input_paths.keys()) != {"index", "queries", "units", "pages"}:
        raise ValueError(f"input_paths must have exactly 'index', 'queries', 'units', 'pages', got {sorted(input_paths.keys()) if isinstance(input_paths, dict) else type(input_paths).__name__}")

    # Forbid any gold path or access in runner
    if any("gold" in k.lower() or "gold" in str(v).lower() for k, v in input_paths.items()):
        raise ValueError("Gold paths are forbidden in runner protocol")

    frozen_files = protocol["frozen_files"]
    if not isinstance(frozen_files, dict) or not frozen_files:
        raise ValueError("frozen_files must be a non-empty dict")

    # Check that all 4 input files are present in frozen_files
    for in_key, in_rel in input_paths.items():
        if in_rel not in frozen_files:
            raise ValueError(f"Required input path {in_key!r} ({in_rel!r}) missing from protocol frozen_files")

    # Check required code modules are pinned in frozen_files
    for req_mod in REQUIRED_FROZEN_MODULES:
        if req_mod not in frozen_files:
            raise ValueError(f"Required dependency code pin {req_mod!r} missing from frozen_files")

    # Validate file hashes on disk for all entries in frozen_files
    for rel_path, expected_sha in frozen_files.items():
        if not rel_path.endswith(".py") and "gold" in rel_path.lower():
            raise ValueError("Gold labels must not be read or hashed by the inference runner")
        resolved_path = validate_project_path(PROJECT_ROOT / rel_path)
        if not resolved_path.is_file():
            raise FileNotFoundError(f"Declared frozen file not found: {resolved_path}")
        actual_sha = hashlib.sha256(resolved_path.read_bytes()).hexdigest()
        if actual_sha != expected_sha:
            raise ValueError(
                f"Frozen file integrity mismatch for {rel_path}: expected {expected_sha!r}, got {actual_sha!r}"
            )


def validate_pre_inference(
    transport: Any,
    *,
    server_version: str,
    generator_digest: str,
    verifier_digest: str,
    embedding_digest: str,
) -> None:
    """Verifies Ollama server version and exact installed tags/digests without running inference."""
    v_data = transport.get_version()
    if not isinstance(v_data, dict) or v_data.get("version") != server_version:
        raise RuntimeError(f"Server version mismatch: reported {v_data.get('version')!r}, expected {server_version!r}")

    tags_data = transport.get_tags()
    models = tags_data.get("models", []) if isinstance(tags_data, dict) else []
    digests_by_model: dict[str, str] = {}
    quants_by_model: dict[str, str] = {}

    for m in models:
        if isinstance(m, dict):
            name = m.get("name") or m.get("model")
            digest = m.get("digest")
            details = m.get("details", {}) if isinstance(m.get("details"), dict) else {}
            quant = details.get("quantization_level")
            if isinstance(name, str) and isinstance(digest, str):
                digests_by_model[name] = digest.strip()
            if isinstance(name, str) and isinstance(quant, str):
                quants_by_model[name] = quant.strip()

    if DEFAULT_GENERATION_MODEL not in digests_by_model:
        raise RuntimeError(f"Required generator model {DEFAULT_GENERATION_MODEL!r} not found in tags")
    if digests_by_model[DEFAULT_GENERATION_MODEL] != generator_digest:
        raise RuntimeError(
            f"Generator model digest mismatch: expected {generator_digest!r}, got {digests_by_model[DEFAULT_GENERATION_MODEL]!r}"
        )

    if VERIFIER_Q3_MODEL not in digests_by_model:
        raise RuntimeError(f"Required verifier model {VERIFIER_Q3_MODEL!r} not found in tags")
    if digests_by_model[VERIFIER_Q3_MODEL] != verifier_digest:
        raise RuntimeError(
            f"Verifier model digest mismatch: expected {verifier_digest!r}, got {digests_by_model[VERIFIER_Q3_MODEL]!r}"
        )
    if quants_by_model.get(VERIFIER_Q3_MODEL) != "Q3_K_M":
        raise RuntimeError(
            f"Verifier quantization mismatch: expected 'Q3_K_M', got {quants_by_model.get(VERIFIER_Q3_MODEL)!r}"
        )

    if MODEL not in digests_by_model:
        raise RuntimeError(f"Required embedding model {MODEL!r} not found in tags")
    if digests_by_model[MODEL] != embedding_digest:
        raise RuntimeError(
            f"Embedding model digest mismatch: expected {embedding_digest!r}, got {digests_by_model[MODEL]!r}"
        )


def _extract_role_stats(calls: list[dict[str, Any]]) -> dict[str, int | None]:
    """Extracts numeric stats distinguishing unknown (None) from 0."""
    stats: dict[str, int | None] = {}
    for key in ("prompt_eval_count", "eval_count", "eval_duration", "total_duration"):
        values = [c.get("raw_response", {}).get(key) if isinstance(c.get("raw_response"), dict) else None for c in calls]
        if calls and all(type(v) is int and v >= 0 for v in values):
            stats[key] = sum(values)  # type: ignore[arg-type]
        else:
            stats[key] = None
    return stats


def run(
    index: dict[str, Any],
    queries: dict[str, Any],
    units: dict[str, Any],
    pages: list[dict[str, Any]],
    output: Path | str,
    *,
    generator_digest: str,
    verifier_digest: str,
    embedding_digest: str,
    schedule: list[dict[str, Any]],
    transport: Any | None = None,
) -> dict[str, Any]:
    """Executes the 16-run paired evaluation sequentially with atomic checkpoints."""
    validate_schedule(queries, schedule)
    validate_index(index)
    if index["model"] != MODEL:
        raise ValueError(f"Selected profile requires Qwen4B index ({MODEL}), got {index.get('model')!r}")
    validate_catalog(units, pages)

    destination = output_path(output)
    if destination.exists():
        raise FileExistsError(f"Target output file already exists at {destination}. Refusing to overwrite.")

    lookup = {c["id"]: c["query"] for c in queries["cases"]}
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "run_status": "running",
        "metadata": {
            "generator_model": DEFAULT_GENERATION_MODEL,
            "generator_digest": generator_digest,
            "verifier_model": VERIFIER_Q3_MODEL,
            "verifier_digest": verifier_digest,
            "embedding_model": MODEL,
            "embedding_digest": embedding_digest,
            "arms": list(ARMS),
            "total_events": TOTAL_EVENTS,
            "index_hash": canonical_hash(index),
            "queries_hash": canonical_hash(queries),
            "units_hash": canonical_hash(units),
            "pages_hash": canonical_hash(pages),
            "schedule_hash": canonical_hash(schedule),
        },
        "cases": [],
    }
    checkpoint(report, destination)

    for idx, item in enumerate(schedule):
        arm = item["arm"]
        cid = item["id"]
        query_text = lookup[cid]
        started = time.perf_counter()

        generator = None
        embedder = None
        verifier = None
        answer = None
        invalid_answer = None
        case_error = None
        formatted = None

        try:
            generator = RecordedGenerator()
            embedder = QwenEmbedder(timeout=180)

            if generator.get_digest() != generator_digest:
                raise ValueError(
                    f"Generator digest before run mismatch: expected {generator_digest!r}, got {generator.get_digest()!r}"
                )

            if arm == "q3":
                verifier = Q3VerifierClient(expected_digest=verifier_digest, transport=transport)
                answer = answer_preserved(
                    index,
                    query_text,
                    units,
                    pages,
                    generator=generator,
                    embedder=embedder,
                    verifier=verifier,
                )
            else:
                answer = answer_preserved(
                    index,
                    query_text,
                    units,
                    pages,
                    generator=generator,
                    embedder=embedder,
                    verifier=None,
                )

            if generator.get_digest() != generator_digest:
                raise ValueError(
                    f"Generator digest after run mismatch: expected {generator_digest!r}, got {generator.get_digest()!r}"
                )

            if not isinstance(answer, dict) or "metadata" not in answer or "outcome" not in answer:
                raise ValueError("Malformed answer structure returned from pipeline")

            meta = answer["metadata"]
            if meta.get("digest") != generator_digest:
                raise ValueError(
                    f"Metadata generator digest mismatch: expected {generator_digest!r}, got {meta.get('digest')!r}"
                )
            if (
                meta.get("planner_used") is not False
                or meta.get("closed_note_tails") is not False
                or meta.get("max_packages") != 1
            ):
                raise ValueError("Selected profile invariants violated in metadata")

            if arm == "q3":
                if meta.get("profile") != "preserved_q3_verifier_v1":
                    raise ValueError(f"Profile mismatch for Q3 arm: {meta.get('profile')!r}")
                if meta.get("verifier_model") != VERIFIER_Q3_MODEL:
                    raise ValueError(f"Verifier model mismatch: {meta.get('verifier_model')!r}")
                if meta.get("verifier_digest") != verifier_digest:
                    raise ValueError(f"Verifier digest mismatch: {meta.get('verifier_digest')!r}")
            else:
                if meta.get("profile") != "preserved_v1":
                    raise ValueError(f"Profile mismatch for legacy arm: {meta.get('profile')!r}")
                if meta.get("model") != DEFAULT_GENERATION_MODEL:
                    raise ValueError(f"Generation model mismatch: {meta.get('model')!r}")

            status = answer["outcome"]["status"]
            if status == "error":
                case_error = answer["outcome"].get("error")
            else:
                formatted = format_answer(answer)
        except Exception as exc:
            status = "error"
            case_error = f"{type(exc).__name__}: {exc}"
            if answer is not None:
                invalid_answer = copy.deepcopy(answer)
                answer = None
        finally:
            if verifier is not None:
                try:
                    verifier.release_loaded(VERIFIER_Q3_MODEL)
                except Exception as exc:
                    status = "error"
                    case_error = f"{case_error or ''}; cleanup: {type(exc).__name__}: {exc}"
                    if answer is not None:
                        invalid_answer, answer = copy.deepcopy(answer), None
                    formatted = None

        gen_calls = copy.deepcopy(getattr(generator, "calls", []))
        ver_calls = copy.deepcopy(getattr(verifier, "calls", [])) if verifier is not None else []
        emb_calls = copy.deepcopy(getattr(embedder, "calls", []))
        lifecycle_records = copy.deepcopy(getattr(verifier, "lifecycle_records", [])) if verifier is not None else []

        trace = []
        if isinstance(answer, dict) and isinstance(answer.get("outcome"), dict):
            trace = copy.deepcopy(answer["outcome"].get("full_result", {}).get("grounding", {}).get("trace", []))
        elif isinstance(invalid_answer, dict) and isinstance(invalid_answer.get("outcome"), dict):
            trace = copy.deepcopy(invalid_answer["outcome"].get("full_result", {}).get("grounding", {}).get("trace", []))

        case_record = {
            "schedule_index": idx,
            "arm": arm,
            "id": cid,
            "query": query_text,
            "attempted": True,
            "status": status,
            "error": case_error,
            "formatted_answer": formatted,
            "trace": trace,
            "calls": gen_calls,
            "verifier_calls": ver_calls,
            "embedding_calls": emb_calls,
            "lifecycle_records": lifecycle_records,
            "wall_seconds": time.perf_counter() - started,
            "per_role_stats": {
                "generator": _extract_role_stats(gen_calls),
                "verifier": _extract_role_stats(ver_calls) if arm == "q3" else None,
            },
        }
        if answer is not None:
            case_record["answer"] = answer
        if invalid_answer is not None:
            case_record["invalid_answer"] = invalid_answer

        report["cases"].append(case_record)
        checkpoint(report, destination)

    incomplete = len(report["cases"]) < TOTAL_EVENTS or any(not c.get("attempted") for c in report["cases"])
    has_errors = any(c.get("status") == "error" for c in report["cases"])
    report["run_status"] = "incomplete" if incomplete else ("error" if has_errors else "completed")
    checkpoint(report, destination)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True, help="Path to frozen protocol JSON")
    parser.add_argument("--output", type=Path, required=True, help="Path to destination results JSON")
    args = parser.parse_args(argv)

    protocol_path = validate_project_path(args.protocol)
    read_json = lambda p: _json(validate_project_path(p).read_text(encoding="utf-8"))
    protocol = read_json(protocol_path)

    # 1. Validate protocol structure, dependency pins, and frozen hashes
    validate_protocol(protocol)

    # 2. Resolve and load input files declared in protocol
    input_paths = protocol["input_paths"]
    index_file = validate_project_path(PROJECT_ROOT / input_paths["index"])
    queries_file = validate_project_path(PROJECT_ROOT / input_paths["queries"])
    units_file = validate_project_path(PROJECT_ROOT / input_paths["units"])
    pages_file = validate_project_path(PROJECT_ROOT / input_paths["pages"])

    index_data = load_index(index_file)
    queries_data = read_json(queries_file)
    units_data = read_json(units_file)
    pages_data = [json.loads(x) for x in pages_file.read_text(encoding="utf-8").splitlines()]
    schedule_data = protocol["schedule"]

    # 3. Check canonical hashes against input_hashes
    for k, d in [("index", index_data), ("queries", queries_data), ("units", units_data), ("pages", pages_data), ("schedule", schedule_data)]:
        act_hash = canonical_hash(d)
        exp_hash = protocol["input_hashes"].get(k)
        if act_hash != exp_hash:
            raise ValueError(f"Input canonical hash mismatch for {k}: expected {exp_hash}, got {act_hash}")

    # 4. Verify server and model availability
    transport = QuantizationOllamaTransport()
    transport.allowed_models = (DEFAULT_GENERATION_MODEL, VERIFIER_Q3_MODEL)
    validate_pre_inference(
        transport,
        server_version=protocol["server_version"],
        generator_digest=protocol["generator_digest"],
        verifier_digest=protocol["verifier_digest"],
        embedding_digest=protocol["embedding_digest"],
    )

    # 5. Run inference
    result = run(
        index_data,
        queries_data,
        units_data,
        pages_data,
        args.output,
        generator_digest=protocol["generator_digest"],
        verifier_digest=protocol["verifier_digest"],
        embedding_digest=protocol["embedding_digest"],
        schedule=schedule_data,
        transport=transport,
    )
    return int(result["run_status"] == "error")


if __name__ == "__main__":
    sys.exit(main())
