"""Evaluation runner for the covered RAG pipeline (Phase 2).

Integrates Qwen 9B generation, Qwen 27B Q3_K_M factual verification, and Q3 query coverage
audit across the fixed 11-case benchmark roster (b01..b08, n01..n03).
Strictly offline in testing, reproducible, and bound to a frozen protocol freeze document.
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
from experiments.query_coverage import answer_covered, format_covered_answer, CorrectionRecordedGenerator
from experiments.source_bound_trial import RecordedGenerator, checkpoint
from experiments.validate_reranker import canonical_hash, output_path
from rag_pensiones.answer import DEFAULT_GENERATION_MODEL
from rag_pensiones.grounding import DEFAULT_PARAMETERS
from rag_pensiones.ingest import PROJECT_ROOT, validate_project_path
from rag_pensiones.qwen_embeddings import MODEL, QwenEmbedder, _json
from rag_pensiones.search import DEFAULT_INDEX_PATH, load_index, validate_index

SCHEMA_VERSION = "covered-pipeline-v1"
PROTOCOL_SCHEMA_VERSION = "covered-pipeline-protocol-v1"
REQUIRED_SERVER_VERSION = "0.40.11"
EXPECTED_CASE_IDS = ("b01", "b02", "b03", "b04", "b05", "b06", "b07", "b08", "n01", "n02", "n03")
EXPECTED_CASE_COUNT = len(EXPECTED_CASE_IDS)

REQUIRED_FROZEN_MODULES = sorted({
    p.relative_to(PROJECT_ROOT).as_posix()
    for folder in ("rag_pensiones", "experiments")
    for p in (PROJECT_ROOT / folder).rglob("*.py")
} | {"evaluation/verifier_capacity_2026-10-26/ollama_defaults_v0.40.11.json"})


def validate_queries_roster(queries: dict[str, Any]) -> list[dict[str, Any]]:
    """Validates the exact 11-case roster: b01..b08, n01..n03 once each in order."""
    if not isinstance(queries, dict) or isinstance(queries, bool):
        raise ValueError("Queries data must be a dict")

    allowed_top_keys = {"schema_version", "cases"}
    extra_keys = set(queries.keys()) - allowed_top_keys
    if extra_keys:
        raise ValueError(f"Queries dict contains unexpected keys: {sorted(extra_keys)}")

    cases = queries.get("cases")
    if not isinstance(cases, list) or len(cases) != EXPECTED_CASE_COUNT:
        raise ValueError(f"Queries cases must contain exactly {EXPECTED_CASE_COUNT} items, got {len(cases) if isinstance(cases, list) else type(cases).__name__}")

    for idx, c in enumerate(cases):
        if not isinstance(c, dict) or isinstance(c, bool):
            raise ValueError(f"Case {idx} must be a dict")
        if set(c.keys()) != {"id", "query"}:
            raise ValueError(f"Case {idx} keys mismatch: {sorted(c.keys())}, expected ['id', 'query']")

        cid = c.get("id")
        expected_id = EXPECTED_CASE_IDS[idx]
        if cid != expected_id:
            raise ValueError(f"Case {idx} ID mismatch: expected {expected_id!r}, got {cid!r}")

        q = c.get("query")
        if not isinstance(q, str) or isinstance(q, bool) or not q.strip():
            raise ValueError(f"Case {cid} query must be a non-empty string")
        if len(q) > 1000:
            raise ValueError(f"Case {cid} query exceeds 1000 characters ({len(q)})")

    return cases


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
        "index_path",
        "queries_path",
        "units_path",
        "pages_path",
        "frozen_files",
    }
    missing_keys = required_keys - set(protocol.keys())
    if missing_keys:
        raise ValueError(f"Protocol missing required keys: {sorted(missing_keys)}")

    version = protocol["schema_version"]
    if version == 'covered-pipeline-protocol-v3':
        from rag_pensiones.structured_retrieval import validate_source_manifest
        allowed = required_keys | {'input_hashes', 'coverage_correction', 'structured_retrieval'}
        if set(protocol) - allowed or 'structured_retrieval' not in protocol:
            raise ValueError('Invalid structured v3 protocol shape')
        if 'coverage_correction' in protocol and protocol['coverage_correction'] is not True:
            raise ValueError('Invalid structured correction flag')
        validate_source_manifest(protocol['structured_retrieval'])
        required_pins = set(protocol['structured_retrieval']['files']) | {'rag_pensiones/structured_retrieval.py', 'experiments/source_recovery_policy.py'}
        if required_pins - set(protocol.get('frozen_files', {})):
            raise ValueError('Structured source/code byte pins missing')
    elif version == "covered-pipeline-protocol-v2":
        if protocol.get("coverage_correction") is not True:
            raise ValueError("v2 requires strict coverage_correction=True")
        allowed = required_keys | {"input_hashes", "coverage_correction"}
        if set(protocol) - allowed:
            raise ValueError("Unknown v2 protocol fields")
    elif "coverage_correction" in protocol:
        raise ValueError("v1 forbids coverage_correction")
    if version not in (PROTOCOL_SCHEMA_VERSION, "covered-pipeline-protocol-v2", 'covered-pipeline-protocol-v3'):
        raise ValueError(
            f"Protocol schema mismatch: expected {PROTOCOL_SCHEMA_VERSION!r}, got {protocol['schema_version']!r}"
        )

    if protocol["server_version"] != REQUIRED_SERVER_VERSION:
        raise ValueError(
            f"Server version mismatch: expected {REQUIRED_SERVER_VERSION!r}, got {protocol['server_version']!r}"
        )

    for dig_key in ("generator_digest", "verifier_digest", "embedding_digest"):
        d_val = protocol[dig_key]
        if isinstance(d_val, bool) or not isinstance(d_val, str) or len(d_val) != 64 or any(ch not in "0123456789abcdef" for ch in d_val.lower()):
            raise ValueError(f"Invalid {dig_key}: expected 64-character hex string, got {d_val!r}")

    if protocol["verifier_digest"] != VERIFIER_Q3_DIGEST:
        raise ValueError(
            f"Verifier digest mismatch against frozen constant: expected {VERIFIER_Q3_DIGEST!r}, got {protocol['verifier_digest']!r}"
        )

    if canonical_hash(protocol["generator_parameters"]) != canonical_hash(DEFAULT_PARAMETERS):
        raise ValueError("Generator parameters mismatch against frozen DEFAULT_PARAMETERS")

    if canonical_hash(protocol["verifier_parameters"]) != canonical_hash(VERIFIER_Q3_PARAMETERS):
        raise ValueError("Verifier parameters mismatch against frozen VERIFIER_Q3_PARAMETERS")

    for path_key in ("index_path", "queries_path", "units_path", "pages_path"):
        val = protocol[path_key]
        if not isinstance(val, str) or isinstance(val, bool) or not val.strip():
            raise ValueError(f"Invalid {path_key}: must be a non-empty string, got {val!r}")
        if "gold" in val.lower():
            raise ValueError(f"Gold paths are forbidden in runner protocol: {val!r}")

    frozen_files = protocol["frozen_files"]
    if not isinstance(frozen_files, dict) or not frozen_files:
        raise ValueError("frozen_files must be a non-empty dict")

    for path_key in ("index_path", "queries_path", "units_path", "pages_path"):
        rel_p = protocol[path_key]
        if rel_p not in frozen_files:
            raise ValueError(f"Required input path {path_key!r} ({rel_p!r}) missing from protocol frozen_files")

    missing_mods = set(REQUIRED_FROZEN_MODULES) - set(frozen_files.keys())
    if missing_mods:
        raise ValueError(f"Required dependency code pins missing: {sorted(missing_mods)}")

    for rel_path, expected_sha in frozen_files.items():
        if "gold" in rel_path.lower():
            raise ValueError(f"Gold labels must not be read or hashed by the inference runner: {rel_path!r}")
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
    """Verifies Ollama server version and installed tags/digests without running inference."""
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
    transport: Any | None = None,
    protocol: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Executes the covered pipeline across the 11 ordered benchmark cases with atomic checkpoints."""
    if protocol is None or not isinstance(protocol, dict):
        raise ValueError("protocol is mandatory: execution must be bound to a frozen protocol freeze document")

    validate_protocol(protocol)
    coverage_correction = protocol.get("coverage_correction") is True
    structured_mode = protocol.get('schema_version') == 'covered-pipeline-protocol-v3'

    if protocol["generator_digest"] != generator_digest:
        raise ValueError(
            f"generator_digest mismatch: protocol has {protocol['generator_digest']!r}, kwarg has {generator_digest!r}"
        )
    if protocol["verifier_digest"] != verifier_digest:
        raise ValueError(
            f"verifier_digest mismatch: protocol has {protocol['verifier_digest']!r}, kwarg has {verifier_digest!r}"
        )
    if protocol["embedding_digest"] != embedding_digest:
        raise ValueError(
            f"embedding_digest mismatch: protocol has {protocol['embedding_digest']!r}, kwarg has {embedding_digest!r}"
        )

    pinned_index = load_index(validate_project_path(PROJECT_ROOT / protocol["index_path"]))
    pinned_queries = _json(validate_project_path(PROJECT_ROOT / protocol["queries_path"]).read_text(encoding="utf-8"))
    pinned_units = _json(validate_project_path(PROJECT_ROOT / protocol["units_path"]).read_text(encoding="utf-8"))
    pinned_pages = [json.loads(x) for x in validate_project_path(PROJECT_ROOT / protocol["pages_path"]).read_text(encoding="utf-8").splitlines()]

    if index != pinned_index:
        raise ValueError("In-memory index data does not match decoded pinned index file")
    if queries != pinned_queries:
        raise ValueError("In-memory queries data does not match decoded pinned queries file")
    if units != pinned_units:
        raise ValueError("In-memory units data does not match decoded pinned units file")
    if pages != pinned_pages:
        raise ValueError("In-memory pages data does not match decoded pinned pages file")

    if protocol.get("input_hashes") and isinstance(protocol["input_hashes"], dict):
        for k, d in [("index", index), ("queries", queries), ("units", units), ("pages", pages)]:
            if k in protocol["input_hashes"]:
                act_h = canonical_hash(d)
                exp_h = protocol["input_hashes"][k]
                if act_h != exp_h:
                    raise ValueError(f"Input canonical hash mismatch for {k}: expected {exp_h}, got {act_h}")

    cases = validate_queries_roster(queries)
    validate_index(index)
    if index.get("model") != MODEL:
        raise ValueError(f"Selected profile requires Qwen4B index ({MODEL}), got {index.get('model')!r}")
    if index.get("model_digest") != embedding_digest:
        raise ValueError(f"Index model digest mismatch: expected {embedding_digest!r}, got {index.get('model_digest')!r}")
    validate_catalog(units, pages)

    destination = output_path(output)
    if destination is None:
        raise ValueError("output path must be specified")

    if transport is None:
        transport = QuantizationOllamaTransport()
        transport.allowed_models = (DEFAULT_GENERATION_MODEL, VERIFIER_Q3_MODEL)

    validate_pre_inference(
        transport,
        server_version=REQUIRED_SERVER_VERSION,
        generator_digest=generator_digest,
        verifier_digest=verifier_digest,
        embedding_digest=embedding_digest,
    )

    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "run_status": "running",
        "metadata": {
            "generator_model": DEFAULT_GENERATION_MODEL,
            "generator_digest": generator_digest,
            "generator_parameters": copy.deepcopy(DEFAULT_PARAMETERS),
            "verifier_model": VERIFIER_Q3_MODEL,
            "verifier_digest": verifier_digest,
            "verifier_parameters": copy.deepcopy(VERIFIER_Q3_PARAMETERS),
            "embedding_model": MODEL,
            "embedding_digest": embedding_digest,
            "total_cases": EXPECTED_CASE_COUNT,
            "case_ids": list(EXPECTED_CASE_IDS),
            "index_hash": canonical_hash(index),
            "queries_hash": canonical_hash(queries),
            "units_hash": canonical_hash(units),
            "pages_hash": canonical_hash(pages),
            "protocol_hash": canonical_hash(protocol),
            "protocol_schema_version": protocol.get("schema_version"),
        },
        "cases": [],
    }
    if coverage_correction:
        report["metadata"]["coverage_correction"] = True
    if structured_mode:
        report['metadata']['structured_retrieval'] = copy.deepcopy(protocol['structured_retrieval'])
    checkpoint(report, destination)

    for case in cases:
        cid = case["id"]
        query_text = case["query"]
        started = time.perf_counter()

        generator = None
        embedder = None
        verifier = None
        answer = None
        invalid_answer = None
        case_error = None
        formatted = None

        try:
            generator = CorrectionRecordedGenerator() if coverage_correction else RecordedGenerator()
            embedder = QwenEmbedder(timeout=180)
            verifier = Q3VerifierClient(expected_digest=verifier_digest, transport=transport)

            if generator.get_digest() != generator_digest:
                raise ValueError(
                    f"Generator digest before run mismatch: expected {generator_digest!r}, got {generator.get_digest()!r}"
                )

            answer = answer_covered(
                index,
                query_text,
                units,
                pages,
                generator=generator,
                embedder=embedder,
                verifier=verifier,
                **({"coverage_correction": True} if coverage_correction else {}),
                **({'structured_retrieval': False} if not structured_mode else {}),
            )

            if generator.get_digest() != generator_digest:
                raise ValueError(
                    f"Generator digest after run mismatch: expected {generator_digest!r}, got {generator.get_digest()!r}"
                )

            if not isinstance(answer, dict) or answer.get("schema_version") != ("preserved-query-coverage-v2" if coverage_correction else "preserved-query-coverage-v1"):
                raise ValueError("Malformed answer structure returned from covered pipeline")

            meta = answer.get("metadata", {})
            if structured_mode and meta.get('structured_profile') != 'certified_lookup_parallel_v1':
                raise ValueError('Structured execution missing actual typed retrieval profile')
            if meta.get("profile") != ("preserved_query_coverage_v2" if coverage_correction else "preserved_query_coverage_v1"):
                raise ValueError(f"Profile mismatch for covered pipeline: {meta.get('profile')!r}")

            if coverage_correction and meta.get("coverage_correction") is not True:
                raise ValueError("Missing actual correction flag")
            status = answer.get("status")
            if status == "error":
                case_error = answer.get("error", "Covered answer reported error")
                formatted = None
            else:
                if meta.get("verifier_model") != VERIFIER_Q3_MODEL:
                    raise ValueError(f"Verifier model mismatch: {meta.get('verifier_model')!r}")
                if meta.get("verifier_digest") != verifier_digest:
                    raise ValueError(f"Verifier digest mismatch: {meta.get('verifier_digest')!r}")

                factual_meta = answer.get("factual_report", {}).get("metadata", {})
                if (
                    factual_meta.get("planner_used") is not False
                    or factual_meta.get("closed_note_tails") is not False
                    or factual_meta.get("max_packages") != 1
                ):
                    raise ValueError("Selected profile invariants violated in factual report metadata")
                if factual_meta.get("digest") != generator_digest:
                    raise ValueError(f"Factual report digest mismatch: {factual_meta.get('digest')!r}")
                if factual_meta.get("verifier_model") != VERIFIER_Q3_MODEL or factual_meta.get("verifier_digest") != verifier_digest:
                    raise ValueError("Factual report verifier identity mismatch")

                formatted = format_covered_answer(answer)
        except Exception as exc:
            status = "error"
            case_error = f"{type(exc).__name__}: {exc}"
            if answer is not None:
                invalid_answer, answer = copy.deepcopy(answer), None
            formatted = None
        finally:
            if verifier is not None:
                try:
                    verifier.release_loaded(VERIFIER_Q3_MODEL)
                except Exception as exc:
                    status = "error"
                    case_error = f"{case_error or ''}; cleanup: {type(exc).__name__}: {exc}".lstrip("; ")
                    if answer is not None:
                        invalid_answer, answer = copy.deepcopy(answer), None
                    formatted = None

        gen_calls = copy.deepcopy(getattr(generator, "calls", []))
        all_ver_calls = copy.deepcopy(getattr(verifier, "calls", [])) if verifier is not None else []
        verifier_calls = [c for c in all_ver_calls if c.get("stage") in ("verification", "reverification")]
        coverage_calls = [c for c in all_ver_calls if c.get("stage") == "query_coverage"]
        emb_calls = copy.deepcopy(getattr(embedder, "calls", []))
        lifecycle_records = copy.deepcopy(getattr(verifier, "lifecycle_records", [])) if verifier is not None else []

        trace = []
        source_obj = answer if answer is not None else invalid_answer
        if isinstance(source_obj, dict):
            factual_part = source_obj.get("factual_report", {})
            if isinstance(factual_part, dict):
                trace = copy.deepcopy(factual_part.get("outcome", {}).get("full_result", {}).get("grounding", {}).get("trace", []))

        per_role_stats = {
            "generator": _extract_role_stats(gen_calls),
            "factual_verifier": _extract_role_stats(verifier_calls),
            "coverage": _extract_role_stats(coverage_calls),
        }

        case_record = {
            "id": cid,
            "query": query_text,
            "attempted": True,
            "status": status,
            "error": case_error,
            "formatted_answer": formatted,
            "calls": gen_calls,
            "verifier_calls": verifier_calls,
            "coverage_calls": coverage_calls,
            "embedding_calls": emb_calls,
            "lifecycle_records": lifecycle_records,
            "trace": trace,
            "wall_seconds": time.perf_counter() - started,
            "per_role_stats": per_role_stats,
        }
        if answer is not None:
            case_record["answer"] = answer
        if invalid_answer is not None:
            case_record["invalid_answer"] = invalid_answer

        report["cases"].append(case_record)
        checkpoint(report, destination)

    incomplete = len(report["cases"]) != EXPECTED_CASE_COUNT or any(not c.get("attempted") for c in report["cases"])
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

    validate_protocol(protocol)

    index_file = validate_project_path(PROJECT_ROOT / protocol["index_path"])
    queries_file = validate_project_path(PROJECT_ROOT / protocol["queries_path"])
    units_file = validate_project_path(PROJECT_ROOT / protocol["units_path"])
    pages_file = validate_project_path(PROJECT_ROOT / protocol["pages_path"])

    index_data = load_index(index_file)
    queries_data = read_json(queries_file)
    units_data = read_json(units_file)
    pages_data = [json.loads(x) for x in pages_file.read_text(encoding="utf-8").splitlines()]

    result = run(
        index_data,
        queries_data,
        units_data,
        pages_data,
        args.output,
        generator_digest=protocol["generator_digest"],
        verifier_digest=protocol["verifier_digest"],
        embedding_digest=protocol["embedding_digest"],
        transport=None,
        protocol=protocol,
    )
    return int(result["run_status"] == "error")


if __name__ == "__main__":
    sys.exit(main())
