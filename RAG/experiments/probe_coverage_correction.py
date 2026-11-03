"""Standalone diagnostic probe: bounded query coverage correction on saved inputs.

Evaluates whether correct_coverage_once improves or maintains query coverage across the fixed
11-case benchmark roster (b01..b08, n01..n03) from prior accepted evaluation runs.
Bound strictly to saved inputs without new factual generation, retrieval, embeddings, or gold answers.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

from experiments.preserved_q3_verifier import (
    Q3VerifierClient,
    VERIFIER_Q3_DIGEST,
    VERIFIER_Q3_MODEL,
    VERIFIER_Q3_PARAMETERS,
)
from experiments.query_coverage import (
    PINNED_9B_DIGEST,
    PINNED_9B_MODEL,
    PINNED_9B_PARAMETERS,
    PINNED_27B_DIGEST,
    PINNED_27B_MODEL,
    PINNED_27B_PARAMETERS,
    CorrectionRecordedGenerator,
    _sanitize_diagnostic_data,
    _validate_role_options,
    claim_inventory,
    correct_coverage_once,
)
from experiments.validate_reranker import canonical_hash
from rag_pensiones.ingest import PROJECT_ROOT, validate_project_path

SCHEMA_VERSION = "coverage-correction-probe-v1"
PROTOCOL_SCHEMA_VERSION = "coverage-correction-probe-protocol-v1"
EXPERIMENT_KIND = "correction_on_saved_inputs"

DEFAULT_MANIFEST_PATH = PROJECT_ROOT / "evaluation/quality_followup_2026-10-28/saved_probe_input_manifest.json"
PINNED_MANIFEST_SHA256 = "3cb9077bceff9951ae3a1130203dc173a18c61df0b9f3cb28e2d124d750096a2"
MANIFEST_PATH = DEFAULT_MANIFEST_PATH
MANIFEST_PIN = PINNED_MANIFEST_SHA256

EXPECTED_CASE_IDS = ("b01", "b02", "b03", "b04", "b05", "b06", "b07", "b08", "n01", "n02", "n03")
EXPECTED_CASE_COUNT = len(EXPECTED_CASE_IDS)

SOURCE_MODULES = (
    "rag_pensiones/answer.py",
    "experiments/query_coverage.py",
    "experiments/preserved_q3_verifier.py",
    "experiments/probe_coverage_correction.py",
    "experiments/source_bound_trial.py",
)


def file_sha256(path: Path) -> str:
    """Calculates SHA-256 digest of a file in binary mode."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def compute_source_manifest() -> dict[str, str]:
    """Computes SHA-256 digests for current active probe source modules."""
    manifest = {}
    for rel in SOURCE_MODULES:
        p = PROJECT_ROOT / rel
        if p.is_file():
            manifest[rel] = file_sha256(p)
    return manifest


def checkpoint(report: dict[str, Any], destination: Path) -> None:
    """Replaces a JSON report atomically; destination parent directory must exist."""
    sanitized = _sanitize_diagnostic_data(report)
    serialized = json.dumps(sanitized, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    temp = destination.with_suffix(".json.pending")
    temp.write_text(serialized, encoding="utf-8")
    os.replace(temp, destination)


def validate_input_files(
    results_path: Path,
    protocol_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], Path, Path, list[dict[str, Any]], list[dict[str, Any]], Path, str]:
    """Validates manifest integrity, input results, protocol, roster, pages, and case queries."""
    manifest_p = validate_project_path(MANIFEST_PATH)
    if not manifest_p.is_file():
        raise FileNotFoundError(f"Saved probe input manifest not found: {manifest_p}")

    act_manifest_sha = file_sha256(manifest_p)
    if act_manifest_sha != MANIFEST_PIN:
        raise ValueError(f"Manifest SHA256 mismatch: expected {MANIFEST_PIN!r}, got {act_manifest_sha!r}")

    try:
        manifest_data = json.loads(manifest_p.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"Malformed JSON in saved probe input manifest: {exc}") from exc

    if not isinstance(manifest_data, dict):
        raise ValueError("Manifest must be a JSON object")
    if manifest_data.get("schema_version") != "saved-correction-probe-inputs-v1":
        raise ValueError(f"Manifest schema_version mismatch: {manifest_data.get('schema_version')!r}")
    if manifest_data.get("experiment_kind") != EXPERIMENT_KIND:
        raise ValueError(f"Manifest experiment_kind mismatch: {manifest_data.get('experiment_kind')!r}")

    roles = manifest_data.get("roles")
    if not isinstance(roles, dict) or not {"results", "protocol", "pages", "queries"}.issubset(roles.keys()):
        raise ValueError(f"Manifest roles missing required keys: {roles}")

    files_sha = manifest_data.get("files_sha256")
    if not isinstance(files_sha, dict):
        raise ValueError("Manifest files_sha256 must be a dictionary")

    expected_digests = {
        role: files_sha.get(roles[role]) for role in ("results", "protocol", "pages", "queries")
    }
    if any(digest is None for digest in expected_digests.values()):
        raise ValueError(f"Manifest files_sha256 missing entry for one or more roles: {expected_digests}")

    results_path = validate_project_path(results_path)
    protocol_path = validate_project_path(protocol_path)
    if not results_path.is_file():
        raise FileNotFoundError(f"Input results file not found: {results_path}")
    if not protocol_path.is_file():
        raise FileNotFoundError(f"Input protocol file not found: {protocol_path}")

    act_results_sha = file_sha256(results_path)
    if act_results_sha != expected_digests["results"]:
        raise ValueError(
            f"Results file SHA256 mismatch against manifest: expected {expected_digests['results']!r}, got {act_results_sha!r}"
        )

    act_protocol_sha = file_sha256(protocol_path)
    if act_protocol_sha != expected_digests["protocol"]:
        raise ValueError(
            f"Protocol file SHA256 mismatch against manifest: expected {expected_digests['protocol']!r}, got {act_protocol_sha!r}"
        )

    try:
        results_data = json.loads(results_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"Malformed JSON in input results: {exc}") from exc

    try:
        protocol_data = json.loads(protocol_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"Malformed JSON in input protocol: {exc}") from exc

    if not isinstance(results_data, dict) or not isinstance(protocol_data, dict):
        raise ValueError("Input results and protocol must be JSON objects")

    # 1. Validate protocol model digests and parameters
    gen_digest = protocol_data.get("generator_digest")
    if gen_digest != PINNED_9B_DIGEST:
        raise ValueError(f"Protocol generator_digest mismatch: expected {PINNED_9B_DIGEST!r}, got {gen_digest!r}")

    verif_digest = protocol_data.get("verifier_digest")
    if verif_digest != PINNED_27B_DIGEST:
        raise ValueError(f"Protocol verifier_digest mismatch: expected {PINNED_27B_DIGEST!r}, got {verif_digest!r}")

    gen_params = protocol_data.get("generator_parameters")
    ok_gp, err_gp = _validate_role_options(gen_params, PINNED_9B_PARAMETERS, "protocol generator parameters")
    if not ok_gp:
        raise ValueError(f"Protocol generator_parameters invalid: {err_gp}")

    ver_params = protocol_data.get("verifier_parameters")
    ok_vp, err_vp = _validate_role_options(ver_params, PINNED_27B_PARAMETERS, "protocol verifier parameters")
    if not ok_vp:
        raise ValueError(f"Protocol verifier_parameters invalid: {err_vp}")

    # 2. Validate pages file
    pages_rel = protocol_data.get("pages_path", roles["pages"])
    pages_path = validate_project_path(PROJECT_ROOT / pages_rel)
    if not pages_path.is_file():
        raise FileNotFoundError(f"Pages file not found: {pages_path}")

    act_pages_sha = file_sha256(pages_path)
    if act_pages_sha != expected_digests["pages"]:
        raise ValueError(
            f"Resolved pages file SHA256 mismatch against manifest: expected {expected_digests['pages']!r}, got {act_pages_sha!r}"
        )

    pages_lines = [json.loads(line) for line in pages_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not pages_lines:
        raise ValueError(f"Pages file {pages_path} is empty")

    # 3. Validate queries roster
    queries_rel = protocol_data.get("queries_path", roles["queries"])
    queries_path = validate_project_path(PROJECT_ROOT / queries_rel)
    if not queries_path.is_file():
        raise FileNotFoundError(f"Queries roster file not found: {queries_path}")

    act_queries_sha = file_sha256(queries_path)
    if act_queries_sha != expected_digests["queries"]:
        raise ValueError(
            f"Resolved queries roster SHA256 mismatch against manifest: expected {expected_digests['queries']!r}, got {act_queries_sha!r}"
        )

    queries_data = json.loads(queries_path.read_text(encoding="utf-8"))
    if not isinstance(queries_data, dict) or "cases" not in queries_data:
        raise ValueError("Queries roster must contain a 'cases' list")

    roster_cases = queries_data.get("cases", [])
    if len(roster_cases) != EXPECTED_CASE_COUNT:
        raise ValueError(f"Queries roster case count mismatch: expected {EXPECTED_CASE_COUNT}, got {len(roster_cases)}")

    # 4. Compare original report provenance to its original protocol
    results_meta = results_data.get("metadata", {})
    if not isinstance(results_meta, dict):
        raise ValueError("Results metadata must be a dictionary")

    proto_canonical = canonical_hash(protocol_data)
    exp_proto_canonical = manifest_data.get("original_protocol_canonical_hash")
    if exp_proto_canonical and proto_canonical != exp_proto_canonical:
        raise ValueError(f"Protocol canonical hash mismatch: expected {exp_proto_canonical!r}, got {proto_canonical!r}")

    if "protocol_hash" in results_meta:
        if results_meta["protocol_hash"] != proto_canonical:
            raise ValueError(
                f"Results protocol_hash mismatch: expected {proto_canonical!r}, got {results_meta['protocol_hash']!r}"
            )

    res_gen_digest = results_meta.get("generator_digest") or results_meta.get("digest")
    if res_gen_digest != PINNED_9B_DIGEST:
        raise ValueError(f"Results generator digest mismatch: expected {PINNED_9B_DIGEST!r}, got {res_gen_digest!r}")

    res_ver_digest = results_meta.get("verifier_digest")
    if res_ver_digest != PINNED_27B_DIGEST:
        raise ValueError(f"Results verifier digest mismatch: expected {PINNED_27B_DIGEST!r}, got {res_ver_digest!r}")

    # 5. Validate full case roster and query identity equality
    results_cases = results_data.get("cases")
    if not isinstance(results_cases, list) or len(results_cases) != EXPECTED_CASE_COUNT:
        raise ValueError(
            f"Results cases count mismatch: expected exactly {EXPECTED_CASE_COUNT} cases, got {len(results_cases) if isinstance(results_cases, list) else type(results_cases).__name__}"
        )

    for idx, case in enumerate(results_cases):
        expected_cid = EXPECTED_CASE_IDS[idx]
        roster_c = roster_cases[idx]
        if not isinstance(roster_c, dict) or roster_c.get("id") != expected_cid:
            raise ValueError(f"Queries roster index {idx} case ID mismatch: expected {expected_cid!r}")
        roster_q = roster_c.get("query")
        if not isinstance(roster_q, str) or not roster_q.strip():
            raise ValueError(f"Queries roster case {expected_cid} missing valid query")

        if not isinstance(case, dict) or case.get("id") != expected_cid:
            raise ValueError(f"Results case index {idx} ID mismatch: expected {expected_cid!r}, got {case.get('id')!r}")

        case_q = case.get("query")
        if case_q != roster_q:
            raise ValueError(f"Case {expected_cid} query mismatch against original roster: {case_q!r} != {roster_q!r}")

        ans_obj = case.get("answer") if isinstance(case.get("answer"), dict) else case
        fact_rep = ans_obj.get("factual_report")
        if isinstance(fact_rep, dict):
            fact_q = fact_rep.get("query")
            if fact_q is not None and fact_q != roster_q:
                raise ValueError(f"Case {expected_cid} factual report query mismatch: {fact_q!r} != {roster_q!r}")

    return results_data, protocol_data, pages_path, queries_path, pages_lines, results_cases, manifest_p, act_manifest_sha


def run_probe(
    input_results: Path,
    input_protocol: Path,
    output_dir: Path,
    *,
    generator: Any | None = None,
    verifier: Any | None = None,
) -> dict[str, Any]:
    """Runs bounded query coverage correction across all saved benchmark cases."""
    results_path = validate_project_path(input_results)
    protocol_path = validate_project_path(input_protocol)
    dest_dir = validate_project_path(output_dir)

    if dest_dir.exists():
        raise FileExistsError(f"Output directory {dest_dir} already exists; refusing to overwrite")

    (
        results_data,
        protocol_data,
        pages_path,
        queries_path,
        pages_lines,
        results_cases,
        manifest_p,
        act_manifest_sha,
    ) = validate_input_files(results_path, protocol_path)

    # Freeze input file and source module hashes BEFORE any execution
    input_hashes_before = {
        "manifest_sha256": act_manifest_sha,
        "results_sha256": file_sha256(results_path),
        "protocol_sha256": file_sha256(protocol_path),
        "pages_sha256": file_sha256(pages_path),
        "queries_roster_sha256": file_sha256(queries_path),
    }
    source_manifest_before = compute_source_manifest()

    dest_dir.mkdir(parents=True, exist_ok=False)

    started_protocol: dict[str, Any] = {
        "schema_version": PROTOCOL_SCHEMA_VERSION,
        "experiment_kind": EXPERIMENT_KIND,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_path": str(manifest_p.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "manifest_sha256": act_manifest_sha,
        "input_results_path": str(results_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "input_protocol_path": str(protocol_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "pages_path": str(pages_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "queries_path": str(queries_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "input_hashes_before": copy.deepcopy(input_hashes_before),
        "source_manifest_before": copy.deepcopy(source_manifest_before),
        "pinned_models": {
            "generator": {
                "model": PINNED_9B_MODEL,
                "digest": PINNED_9B_DIGEST,
                "parameters": copy.deepcopy(PINNED_9B_PARAMETERS),
            },
            "verifier": {
                "model": PINNED_27B_MODEL,
                "digest": PINNED_27B_DIGEST,
                "parameters": copy.deepcopy(PINNED_27B_PARAMETERS),
            },
        },
        "case_ids": list(EXPECTED_CASE_IDS),
    }

    protocol_dest = dest_dir / "protocol.json"
    results_dest = dest_dir / "results.json"

    # Preserve started protocol file before any calls
    protocol_dest.write_text(
        json.dumps(_sanitize_diagnostic_data(started_protocol), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "experiment_kind": EXPERIMENT_KIND,
        "run_status": "running",
        "metadata": copy.deepcopy(started_protocol),
        "cases": [],
    }
    checkpoint(report, results_dest)

    if generator is None:
        generator = CorrectionRecordedGenerator()
    if verifier is None:
        verifier = Q3VerifierClient(expected_digest=PINNED_27B_DIGEST)

    for case in results_cases:
        cid = case["id"]
        query = case["query"]

        answer_obj = case.get("answer")
        if isinstance(answer_obj, dict):
            factual_report = answer_obj.get("factual_report")
            coverage_audit = answer_obj.get("coverage_audit")
        else:
            factual_report = case.get("factual_report")
            coverage_audit = case.get("coverage_audit")

        if not isinstance(factual_report, dict):
            factual_report = {}

        factual_fingerprint = canonical_hash(factual_report)
        audit_fingerprint = canonical_hash(coverage_audit) if coverage_audit is not None else None

        # Derive initial inventory using verified 1:1 claim_inventory helper
        try:
            inventory = claim_inventory(factual_report)
        except Exception as exc:
            inventory = []

        gen_calls_before = len(getattr(generator, "calls", []))
        verif_calls_before = len(getattr(verifier, "calls", []))
        t_case_start = time.perf_counter()

        case_error = None
        corr_res: dict[str, Any] | None = None

        try:
            corr_res = correct_coverage_once(
                query,
                factual_report,
                inventory,
                coverage_audit,
                pages=pages_lines,
                generator=generator,
                verifier=verifier,
            )
        except Exception as exc:
            case_error = f"{type(exc).__name__}: {exc}"
            corr_res = {
                "attempted": True,
                "applied": False,
                "inventory": copy.deepcopy(inventory),
                "coverage_audit": copy.deepcopy(coverage_audit),
                "diagnostics": {
                    "attempted": True,
                    "applied": False,
                    "trigger": None,
                    "reason": f"Execution error in correct_coverage_once: {case_error}",
                    "error": case_error,
                },
            }
        finally:
            wall_seconds = time.perf_counter() - t_case_start
            raw_gen_calls = getattr(generator, "calls", [])[gen_calls_before:]
            raw_verif_calls = getattr(verifier, "calls", [])[verif_calls_before:]

        case_record = {
            "id": cid,
            "query": query,
            "original_factual_fingerprint": factual_fingerprint,
            "original_coverage_audit_fingerprint": audit_fingerprint,
            "original_inventory": copy.deepcopy(inventory),
            "original_coverage_audit": copy.deepcopy(coverage_audit),
            "attempted": corr_res.get("attempted", False) if isinstance(corr_res, dict) else False,
            "applied": corr_res.get("applied", False) if isinstance(corr_res, dict) else False,
            "error": case_error,
            "corrected_inventory": copy.deepcopy(corr_res.get("inventory")) if isinstance(corr_res, dict) else copy.deepcopy(inventory),
            "corrected_coverage_audit": copy.deepcopy(corr_res.get("coverage_audit")) if isinstance(corr_res, dict) else copy.deepcopy(coverage_audit),
            "diagnostics": copy.deepcopy(corr_res.get("diagnostics")) if isinstance(corr_res, dict) else {},
            "client_calls": {
                "generator": copy.deepcopy(raw_gen_calls),
                "verifier": copy.deepcopy(raw_verif_calls),
            },
            "call_counts": {
                "generator": len(raw_gen_calls),
                "verifier": len(raw_verif_calls),
            },
            "wall_seconds": wall_seconds,
        }

        report["cases"].append(case_record)
        checkpoint(report, results_dest)

    # Post-run hash and integrity check
    source_manifest_after = compute_source_manifest()
    act_manifest_sha_after = file_sha256(manifest_p)
    input_hashes_after = {
        "manifest_sha256": act_manifest_sha_after,
        "results_sha256": file_sha256(results_path),
        "protocol_sha256": file_sha256(protocol_path),
        "pages_sha256": file_sha256(pages_path),
        "queries_roster_sha256": file_sha256(queries_path),
    }

    hash_integrity_verified = (
        source_manifest_before == source_manifest_after and
        input_hashes_before == input_hashes_after and
        act_manifest_sha_after == MANIFEST_PIN
    )

    summary = {
        "total_cases": len(report["cases"]),
        "attempted_count": sum(1 for c in report["cases"] if c.get("attempted")),
        "applied_count": sum(1 for c in report["cases"] if c.get("applied")),
        "unattempted_count": sum(1 for c in report["cases"] if not c.get("attempted")),
        "error_count": sum(1 for c in report["cases"] if c.get("error") is not None),
    }

    has_errors = any(c.get("error") is not None for c in report["cases"])
    report["run_status"] = "error" if (has_errors or not hash_integrity_verified) else "completed"
    report["metadata"]["source_manifest_after"] = source_manifest_after
    report["metadata"]["input_hashes_after"] = input_hashes_after
    report["metadata"]["hash_integrity_verified"] = hash_integrity_verified
    report["metadata"]["summary"] = summary
    if not hash_integrity_verified:
        report["metadata"]["invalidated_diagnostic"] = True

    checkpoint(report, results_dest)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-results", type=Path, required=True, help="Path to input results.json")
    parser.add_argument("--input-protocol", type=Path, required=True, help="Path to input protocol.json")
    parser.add_argument("--output-dir", type=Path, required=True, help="Path to new output directory")
    args = parser.parse_args(argv)

    try:
        report = run_probe(args.input_results, args.input_protocol, args.output_dir)
        return 0 if report.get("run_status") == "completed" else 1
    except Exception as exc:
        sys.stderr.write(f"probe_coverage_correction fatal error: {exc}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
