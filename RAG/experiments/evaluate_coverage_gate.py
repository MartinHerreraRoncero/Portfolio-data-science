"""Isolated evaluation gate runner for the Q3 query coverage auditor.

Pins server version 0.40.11, exact verifier digest/sampling, and project code hashes.
Executes sequentially over a single hot Q3 client without retries, without gold access,
and with per-case atomic checkpoints and isolated call logging.
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

from experiments.compare_system import checkpoint
from experiments.preserved_q3_verifier import (
    Q3VerifierClient,
    VERIFIER_Q3_DIGEST,
    VERIFIER_Q3_MODEL,
    VERIFIER_Q3_PARAMETERS,
)
from experiments.query_coverage import (
    audit_coverage,
    build_coverage_publication,
)
from experiments.validate_reranker import canonical_hash, output_path
from rag_pensiones.ingest import PROJECT_ROOT, validate_project_path

GATE_PROTOCOL_SCHEMA_VERSION = "query-coverage-gate-v1"
GATE_RESULTS_SCHEMA_VERSION = "query-coverage-gate-results-v1"
REQUIRED_SERVER_VERSION = "0.40.11"
REQUIRED_FROZEN_MODULES = {
    p.relative_to(PROJECT_ROOT).as_posix()
    for folder in ("rag_pensiones", "experiments")
    for p in (PROJECT_ROOT / folder).rglob("*.py")
} | {"evaluation/verifier_capacity_2026-10-26/ollama_defaults_v0.40.11.json"}


def validate_protocol(protocol: dict[str, Any]) -> None:
    """Strictly validates the query coverage gate protocol pins, config, and file hashes."""
    if not isinstance(protocol, dict) or isinstance(protocol, bool):
        raise ValueError("Protocol must be a dict")

    required_keys = {
        "schema_version",
        "server_version",
        "verifier_digest",
        "verifier_parameters",
        "input_path",
        "input_sha256",
        "frozen_files",
    }
    missing_keys = required_keys - set(protocol.keys())
    if missing_keys:
        raise ValueError(f"Protocol missing required keys: {sorted(missing_keys)}")

    if protocol["schema_version"] != GATE_PROTOCOL_SCHEMA_VERSION:
        raise ValueError(
            f"Protocol schema mismatch: expected {GATE_PROTOCOL_SCHEMA_VERSION!r}, got {protocol['schema_version']!r}"
        )

    if protocol["server_version"] != REQUIRED_SERVER_VERSION:
        raise ValueError(
            f"Server version mismatch: expected {REQUIRED_SERVER_VERSION!r}, got {protocol['server_version']!r}"
        )

    if protocol["verifier_digest"] != VERIFIER_Q3_DIGEST:
        raise ValueError(
            f"Verifier digest mismatch: expected {VERIFIER_Q3_DIGEST!r}, got {protocol['verifier_digest']!r}"
        )

    if canonical_hash(protocol["verifier_parameters"]) != canonical_hash(VERIFIER_Q3_PARAMETERS):
        raise ValueError("Verifier parameters mismatch against frozen VERIFIER_Q3_PARAMETERS")

    input_path_str = protocol["input_path"]
    if not isinstance(input_path_str, str) or not input_path_str.strip():
        raise ValueError("input_path must be a non-empty string")
    if "gold" in input_path_str.lower():
        raise ValueError("Gold paths are strictly forbidden in gate runner protocol")

    input_sha = protocol["input_sha256"]
    if not isinstance(input_sha, str) or len(input_sha) != 64 or any(c not in "0123456789abcdef" for c in input_sha.lower()):
        raise ValueError(f"Invalid input_sha256: expected 64-char hex string, got {input_sha!r}")

    frozen_files = protocol["frozen_files"]
    if not isinstance(frozen_files, dict) or not frozen_files:
        raise ValueError("frozen_files must be a non-empty dict")

    if input_path_str not in frozen_files:
        raise ValueError(f"Declared input_path {input_path_str!r} missing from protocol frozen_files")
    if missing := REQUIRED_FROZEN_MODULES - frozen_files.keys():
        raise ValueError(f"Required dependency code pins missing: {sorted(missing)}")

    for rel_path, expected_sha in frozen_files.items():
        if "gold" in rel_path.lower():
            raise ValueError(f"Gold labels must not be read or hashed by gate runner: {rel_path!r}")
        target = validate_project_path(PROJECT_ROOT / rel_path)
        if not target.is_file():
            raise FileNotFoundError(f"Declared frozen file not found: {target}")
        actual_sha = hashlib.sha256(target.read_bytes()).hexdigest()
        if actual_sha != expected_sha:
            raise ValueError(
                f"Frozen file integrity mismatch for {rel_path}: expected {expected_sha!r}, got {actual_sha!r}"
            )


def validate_gate_inputs(inputs_data: dict[str, Any], expected_sha256: str | None = None) -> list[dict[str, Any]]:
    """Validates the exact roster structure of input cases {id, query, inventory}."""
    if not isinstance(inputs_data, dict) or isinstance(inputs_data, bool):
        raise ValueError("Inputs data must be a dict")

    cases = inputs_data.get("cases")
    if not isinstance(cases, list) or len(cases) == 0:
        raise ValueError("Inputs data must contain a non-empty 'cases' list")

    seen_ids: set[str] = set()
    for idx, c in enumerate(cases):
        if not isinstance(c, dict) or isinstance(c, bool):
            raise ValueError(f"Case at index {idx} must be a dict")
        if set(c.keys()) != {"id", "query", "inventory"}:
            raise ValueError(f"Case {idx} keys mismatch: {sorted(c.keys())}, expected ['id', 'inventory', 'query']")

        cid = c.get("id")
        if not isinstance(cid, str) or not cid.strip():
            raise ValueError(f"Case {idx} has invalid id: {cid!r}")
        if cid in seen_ids:
            raise ValueError(f"Duplicate case id in inputs: {cid!r}")
        seen_ids.add(cid)

        q = c.get("query")
        if not isinstance(q, str) or not q.strip():
            raise ValueError(f"Case {cid} has empty query")

        inv = c.get("inventory")
        if not isinstance(inv, list):
            raise ValueError(f"Case {cid} inventory must be a list")
        for r_idx, row in enumerate(inv):
            if not isinstance(row, dict) or isinstance(row, bool):
                raise ValueError(f"Case {cid} inventory row {r_idx} must be a dict")
            req_keys = {"claim_id", "text", "evidence", "publication_claim"}
            if not req_keys.issubset(row.keys()):
                raise ValueError(f"Case {cid} inventory row {r_idx} missing required keys: {sorted(req_keys - set(row.keys()))}")

    return cases


def run(
    inputs_path: Path | str,
    output_path_target: Path | str,
    *,
    verifier: Any | None = None,
    protocol: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Runs the isolated coverage gate sequentially across cases with atomic checkpoints.

    Reuses a single hot Q3 client across the whole gate.
    Captures only newly produced calls per case, avoids duplicate call accumulation,
    and never calls the model for cases with empty inventory.
    """
    inputs_resolved = validate_project_path(Path(inputs_path))
    if "gold" in str(inputs_resolved).lower():
        raise ValueError("Gold paths are forbidden in gate runner")

    input_bytes = inputs_resolved.read_bytes()
    actual_input_sha = hashlib.sha256(input_bytes).hexdigest()

    if protocol is None:
        raise ValueError("Frozen protocol required before model clients")
    if protocol is not None:
        validate_protocol(protocol)
        if actual_input_sha != protocol["input_sha256"]:
            raise ValueError(
                f"Inputs SHA-256 mismatch against protocol: expected {protocol['input_sha256']!r}, got {actual_input_sha!r}"
            )

    destination = output_path(Path(output_path_target))
    if destination.exists():
        raise FileExistsError(f"Target output file already exists at {destination}. Refusing to overwrite.")

    inputs_data = json.loads(input_bytes.decode("utf-8"))
    cases = validate_gate_inputs(inputs_data)

    if verifier is None:
        verifier = Q3VerifierClient(expected_digest=VERIFIER_Q3_DIGEST)

    if hasattr(verifier, "model") and verifier.model != VERIFIER_Q3_MODEL:
        raise ValueError(f"Verifier model mismatch: expected {VERIFIER_Q3_MODEL!r}, got {verifier.model!r}")
    if hasattr(verifier, "expected_digest") and verifier.expected_digest != VERIFIER_Q3_DIGEST:
        raise ValueError(f"Verifier digest mismatch: expected {VERIFIER_Q3_DIGEST!r}, got {verifier.expected_digest!r}")
    if hasattr(verifier, "parameters") and canonical_hash(verifier.parameters) != canonical_hash(VERIFIER_Q3_PARAMETERS):
        raise ValueError("Verifier parameters mismatch against frozen VERIFIER_Q3_PARAMETERS")

    report: dict[str, Any] = {
        "schema_version": GATE_RESULTS_SCHEMA_VERSION,
        "run_status": "running",
        "metadata": {
            "verifier_model": VERIFIER_Q3_MODEL,
            "verifier_digest": VERIFIER_Q3_DIGEST,
            "verifier_parameters": copy.deepcopy(VERIFIER_Q3_PARAMETERS),
            "input_path": str(inputs_resolved),
            "input_sha256": actual_input_sha,
            "total_cases": len(cases),
        },
        "cases": [],
    }
    checkpoint(report, destination)

    for case in cases:
        cid = case["id"]
        query_text = case["query"]
        inv = case["inventory"]

        n_calls_before = len(getattr(verifier, "calls", []))
        started = time.perf_counter()

        audit = audit_coverage(query_text, inv, verifier=verifier)
        wall_time = time.perf_counter() - started

        all_calls = getattr(verifier, "calls", [])
        new_calls = all_calls[n_calls_before:]

        if audit.get("status") == "completed":
            pub = build_coverage_publication(query_text, inv, audit["validated_payload"])
            case_entry = {
                "id": cid,
                "query": query_text,
                "attempted": True,
                "status": pub["status"],
                "publication": pub,
                "audit": copy.deepcopy(audit),
                "calls": copy.deepcopy(new_calls),
                "wall_seconds": wall_time,
                "error": None,
            }
        else:
            case_entry = {
                "id": cid,
                "query": query_text,
                "attempted": True,
                "status": "error",
                "publication": {
                    "schema_version": "coverage-publication-v1",
                    "status": "error",
                    "claims": [],
                    "missing_parts": [],
                    "reason": f"Coverage audit error: {audit.get('error')}",
                    "parts": [],
                },
                "audit": copy.deepcopy(audit),
                "calls": copy.deepcopy(new_calls),
                "wall_seconds": wall_time,
                "error": audit.get("error"),
            }

        report["cases"].append(case_entry)
        checkpoint(report, destination)

    has_errors = any(c.get("status") == "error" for c in report["cases"])
    report["run_status"] = "error" if has_errors else "completed"
    checkpoint(report, destination)

    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True, help="Path to frozen gate protocol JSON")
    parser.add_argument("--output", type=Path, required=True, help="Path to destination gate results JSON")
    args = parser.parse_args(argv)

    protocol_path = validate_project_path(args.protocol)
    protocol_data = json.loads(protocol_path.read_text(encoding="utf-8"))

    validate_protocol(protocol_data)

    input_path = PROJECT_ROOT / protocol_data["input_path"]
    result = run(
        input_path,
        args.output,
        protocol=protocol_data,
    )

    return int(result["run_status"] == "error")


if __name__ == "__main__":
    sys.exit(main())
