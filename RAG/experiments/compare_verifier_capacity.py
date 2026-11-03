"""Verifier capacity comparison (Phase 1, Phase 2, and Phase 3).

Isolates claim/evidence verification across 4 model arms without running live models in tests.
Phase 1: Produces frozen canonical JSON artifacts: inputs.json, gold.json, schedule.json, protocol.json.
Phase 2: Executes frozen schedule sequentially over dedicated loopback Ollama transport with atomic checkpoints.
Phase 3: Pure offline scoring, separating primary 28 controls from b05 diagnostic repeats, validating protocol pins.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Protocol
import urllib.error
import urllib.request

from rag_pensiones.answer import _parse_float, _reject_constant
from rag_pensiones.grounding import (
    MAX_MESSAGES_BYTES_VERIFICATION,
    build_verification_messages,
    build_verification_schema,
    normalize_verification_payload,
    validate_verification_payload,
)
from rag_pensiones.ingest import PROJECT_ROOT, validate_project_path

INPUTS_SCHEMA_VERSION = "verifier-capacity-inputs-v1"
GOLD_SCHEMA_VERSION = "verifier-capacity-gold-v1"
SCHEDULE_SCHEMA_VERSION = "verifier-capacity-schedule-v1"
PROTOCOL_SCHEMA_VERSION = "verifier-capacity-protocol-v1"
RUN_RESULTS_SCHEMA_VERSION = "verifier-capacity-run-results-v1"
SCORE_REPORT_SCHEMA_VERSION = "verifier-capacity-score-report-v1"

ALLOWED_VERDICT_LABELS = ("supported", "contradicted", "insufficient_evidence")

REQUIRED_CODE_MODULES: dict[str, str] = {
    "grounding": "rag_pensiones/grounding.py",
    "answer": "rag_pensiones/answer.py",
    "compare_verifier_capacity": "experiments/compare_verifier_capacity.py",
}


def is_valid_sha256(val: Any) -> bool:
    """Checks whether val is a valid 64-character lowercase hex SHA-256 digest string."""
    return isinstance(val, str) and len(val) == 64 and bool(re.fullmatch(r"[0-9a-f]{64}", val))

DEFAULT_PAIRS_PATH = PROJECT_ROOT / "evaluation" / "grounding_pairs.json"
DEFAULT_COUNTEREXAMPLES_PATH = PROJECT_ROOT / "evaluation" / "baseline_stability_counterexamples.json"
DEFAULT_OLLAMA_DEFAULTS_PATH = (
    PROJECT_ROOT / "evaluation" / "verifier_capacity_2026-10-26" / "ollama_defaults_v0.40.11.json"
)

DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
ALLOWED_OLLAMA_MODELS = ("qwen3.5:9b", "qwen3.8:27b")
DEFAULT_RUN_TIMEOUT = 240.0
MAX_RUN_RESPONSE_BYTES = 4 * 1024 * 1024  # Strict 4 MiB limit
MAX_CONTEXT_WINDOW = 8192
CONTEXT_SAFETY_MARGIN = 256


class OllamaResponseError(RuntimeError):
    """Raised when Ollama returns an invalid, malformed, non-finite, or oversized response body."""

    def __init__(
        self,
        message: str,
        *,
        raw_text: str | None = None,
        is_truncated: bool = False,
        status_code: int = 200,
    ) -> None:
        super().__init__(message)
        self.raw_text = raw_text
        self.is_truncated = is_truncated
        self.status_code = status_code


def parse_show_parameters(parameters_val: Any) -> dict[str, Any]:
    """Parses model parameter text or dict from /api/show into typed numeric/string parameter map."""
    params: dict[str, Any] = {}
    if isinstance(parameters_val, dict):
        for k, v in parameters_val.items():
            params[k] = v
    elif isinstance(parameters_val, str):
        for line in parameters_val.strip().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(None, 1)
            if len(parts) == 2:
                k, v_str = parts[0], parts[1]
                try:
                    if "." in v_str:
                        val: int | float | str = float(v_str)
                    else:
                        val = int(v_str)
                except ValueError:
                    val = v_str
                params[k] = val
    return params


def resolve_effective_sampling(
    server_defaults: dict[str, Any],
    show_params: dict[str, Any],
    request_options: dict[str, Any],
) -> dict[str, Any]:
    """Resolves effective sampling options following precedence: defaults < /api/show parameters < request options."""
    effective = copy.deepcopy(server_defaults)
    effective.update(show_params)
    effective.update(request_options)
    return effective


def validate_sampling_match(
    sampling_b: dict[str, Any],
    sampling_d: dict[str, Any],
    *,
    excluded_fields: set[str] | None = None,
) -> tuple[bool, list[str]]:
    """Validates that sampling fields match between two configurations, returning (matches, mismatches)."""
    if excluded_fields is None:
        excluded_fields = {"model", "think", "num_predict"}
    all_keys = (set(sampling_b.keys()) | set(sampling_d.keys())) - excluded_fields
    mismatches: list[str] = []
    for k in sorted(all_keys):
        v_b = sampling_b.get(k)
        v_d = sampling_d.get(k)
        if isinstance(v_b, (int, float)) and isinstance(v_d, (int, float)):
            if float(v_b) != float(v_d):
                mismatches.append(f"{k}: B={v_b} vs D={v_d}")
        elif v_b != v_d:
            mismatches.append(f"{k}: B={v_b} vs D={v_d}")
    return (len(mismatches) == 0, mismatches)

# Four experimental arms strictly matching contract specifications
ARMS: dict[str, dict[str, Any]] = {
    "A": {
        "arm_id": "A",
        "model": "qwen3.5:9b",
        "think": False,
        "options": {
            "temperature": 0,
            "seed": 42,
            "num_ctx": 8192,
            "num_predict": 1024,
        },
        "description": "Baseline qwen3.5:9b, think=false, num_predict=1024 (recorded baseline defaults, repeat_penalty omitted).",
    },
    "B": {
        "arm_id": "B",
        "model": "qwen3.5:9b",
        "think": False,
        "options": {
            "temperature": 0,
            "seed": 42,
            "num_ctx": 8192,
            "num_predict": 4096,
        },
        "description": "qwen3.5:9b, think=false, num_predict=4096 (extended output budget, identical baseline sampling).",
    },
    "C": {
        "arm_id": "C",
        "model": "qwen3.5:9b",
        "think": True,
        "options": {
            "temperature": 0,
            "seed": 42,
            "num_ctx": 8192,
            "num_predict": 4096,
        },
        "description": "qwen3.5:9b, think=true, num_predict=4096 (reasoning/thinking enabled, extended output budget).",
    },
    "D": {
        "arm_id": "D",
        "model": "qwen3.8:27b",
        "think": False,
        "options": {
            "temperature": 0,
            "seed": 42,
            "num_ctx": 8192,
            "num_predict": 4096,
            "presence_penalty": 1.5,
            "top_k": 20,
            "top_p": 0.95,
            "min_p": 0.0,
        },
        "description": "qwen3.8:27b, think=false, num_predict=4096 (scaled 27B model generation; explicitly aligns 9B defaults presence_penalty=1.5, top_k=20, top_p=0.95, min_p=0.0).",
    },
}


def sanitize_finite(data: Any) -> Any:
    """Recursively converts non-finite floats to string representations for valid JSON serialization."""
    if isinstance(data, float):
        if math.isnan(data):
            return "NaN"
        if math.isinf(data):
            return "Infinity" if data > 0 else "-Infinity"
        return data
    elif isinstance(data, dict):
        return {k: sanitize_finite(v) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        return [sanitize_finite(item) for item in data]
    return data


def assert_finite(data: Any) -> None:
    """Recursively validates that floats are finite numbers (no NaN, +Infinity, -Infinity)."""
    if isinstance(data, float):
        if math.isnan(data) or math.isinf(data):
            raise ValueError(f"Non-finite float value encountered: {data!r}")
    elif isinstance(data, dict):
        for k, v in data.items():
            assert_finite(v)
    elif isinstance(data, (list, tuple)):
        for item in data:
            assert_finite(item)


def canonical_json_bytes(data: Any) -> bytes:
    """Serializes data to canonical finite UTF-8 JSON bytes (sorted keys, compact separators, no NaN)."""
    assert_finite(data)
    return json.dumps(
        data,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(data: Any) -> str:
    """Computes SHA-256 of canonical JSON bytes."""
    return hashlib.sha256(canonical_json_bytes(data)).hexdigest()


def compute_file_sha256(path: Path) -> str:
    """Computes SHA-256 of raw file bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def get_arm_case_order(arm_id: str, case_ids: list[str]) -> list[str]:
    """Applies deterministic case rotation/reversal per arm block to mitigate position biases."""
    n = len(case_ids)
    if arm_id == "A":
        return list(case_ids)
    elif arm_id == "B":
        return list(reversed(case_ids))
    elif arm_id == "C":
        shift = n // 2
        return case_ids[shift:] + case_ids[:shift]
    elif arm_id == "D":
        shift = n // 2
        rotated = case_ids[shift:] + case_ids[:shift]
        return list(reversed(rotated))
    else:
        raise ValueError(f"Unknown arm_id {arm_id!r}")


def build_schedule(case_ids: list[str], dev_status_map: dict[str, str]) -> list[dict[str, Any]]:
    """Builds a deterministic schedule: 4 fixed arm blocks (A, B, C, D).

    Each block runs all 29 cases once (using deterministic rotation/reversal),
    followed by 2 additional correlated repeats of counterexample b05.
    Total runs: 4 * (29 + 2) = 124 scheduled executions.
    """
    items: list[dict[str, Any]] = []
    schedule_idx = 0

    for block_idx, arm_id in enumerate(["A", "B", "C", "D"]):
        ordered_cases = get_arm_case_order(arm_id, case_ids)

        # 1. Primary evaluation of all 29 cases in this arm
        for cid in ordered_cases:
            is_b05 = (cid == "b05")
            item = {
                "schedule_index": schedule_idx,
                "block_index": block_idx,
                "arm_id": arm_id,
                "case_id": cid,
                "repeat_index": 1,
                "is_b05": is_b05,
                "is_correlated_repeat": False,
                "development_status": dev_status_map[cid],
            }
            items.append(item)
            schedule_idx += 1

        # 2. Two additional repeats of b05 in this arm block
        for rep in [2, 3]:
            item = {
                "schedule_index": schedule_idx,
                "block_index": block_idx,
                "arm_id": arm_id,
                "case_id": "b05",
                "repeat_index": rep,
                "is_b05": True,
                "is_correlated_repeat": True,
                "development_status": dev_status_map["b05"],
            }
            items.append(item)
            schedule_idx += 1

    return items


def extract_b05_case(counterexamples_data: dict[str, Any]) -> tuple[dict[str, Any], str]:
    """Extracts initial b05 verification call from counterexamples data preserving messages/schema exactly.

    Validates that claims/evidence match the actual call user message payload, not guessed repair evidence.
    Returns (b05_input_record, expected_verdict).
    """
    cases = counterexamples_data.get("cases", [])
    if not isinstance(cases, list):
        raise ValueError("Malformed counterexamples data: 'cases' must be a list")

    b05_case = next((c for c in cases if isinstance(c, dict) and c.get("id") == "b05"), None)
    if b05_case is None:
        raise ValueError("Case 'b05' not found in counterexamples dataset")

    calls = b05_case.get("calls_for_development", [])
    if not isinstance(calls, list) or len(calls) == 0:
        raise ValueError("Case 'b05' missing 'calls_for_development' list")

    verify_call_item = next(
        (item for item in calls if isinstance(item, dict) and item.get("stage") == "verification"),
        None,
    )
    if verify_call_item is None:
        raise ValueError("No verification stage call found in b05 'calls_for_development'")

    call = verify_call_item.get("call")
    if not isinstance(call, dict):
        raise ValueError("Verification call item in b05 is missing 'call' dictionary")

    messages = call.get("messages")
    schema = call.get("schema")

    if not isinstance(messages, list) or len(messages) < 2:
        raise ValueError("b05 verification call messages must be a list with at least 2 messages")
    if not isinstance(schema, dict) or not schema:
        raise ValueError("b05 verification call schema must be a non-empty dict")

    # Check payload size guard without truncation
    serialized_bytes = json.dumps(messages, ensure_ascii=False).encode("utf-8")
    if len(serialized_bytes) > MAX_MESSAGES_BYTES_VERIFICATION:
        raise ValueError(
            f"b05 verification messages ({len(serialized_bytes)} bytes) exceed maximum "
            f"budget of {MAX_MESSAGES_BYTES_VERIFICATION} bytes"
        )

    # Derive claims and evidence directly from the verification call's user payload
    user_msg = messages[1]
    if not isinstance(user_msg, dict) or user_msg.get("role") != "user":
        raise ValueError("b05 verification messages[1] must have role 'user'")

    user_content = user_msg.get("content", "")
    try:
        user_payload = json.loads(user_content, parse_constant=_reject_constant)
    except Exception as exc:
        raise ValueError(f"b05 user message content is not valid JSON: {exc}") from exc

    claims = user_payload.get("afirmaciones")
    evidence = user_payload.get("catalogo_evidencias")

    if not isinstance(claims, list) or len(claims) == 0:
        raise ValueError("b05 verification payload missing non-empty 'afirmaciones'")
    if not isinstance(evidence, list) or len(evidence) == 0:
        raise ValueError("b05 verification payload missing non-empty 'catalogo_evidencias'")

    # Validate that claims and evidence structure can be processed by validate_verification_payload
    for c in claims:
        if not isinstance(c, dict) or "claim_id" not in c or "text" not in c or "evidence_ids" not in c:
            raise ValueError(f"Malformed claim in b05: {c!r}")
    for ev in evidence:
        if not isinstance(ev, dict) or "evidence_id" not in ev or "source_id" not in ev or "quote" not in ev:
            raise ValueError(f"Malformed evidence in b05: {ev!r}")

    # Build input record preserving call.messages and call.schema exactly
    b05_input = {
        "id": "b05",
        "source_case_id": "b05",
        "kind": "counterexample_initial_draft",
        "claims": copy.deepcopy(claims),
        "evidence": copy.deepcopy(evidence),
        "messages": copy.deepcopy(messages),
        "schema": copy.deepcopy(schema),
    }

    # Expected label: per contract, initial b05 draft is supported by E2 (not its failed repair)
    expected_verdict = "supported"
    return b05_input, expected_verdict


def prepare(
    output_dir: Path | str,
    *,
    pairs_path: Path | str = DEFAULT_PAIRS_PATH,
    counterexamples_path: Path | str = DEFAULT_COUNTEREXAMPLES_PATH,
    defaults_path: Path | str = DEFAULT_OLLAMA_DEFAULTS_PATH,
) -> dict[str, Any]:
    """Prepares and freezes inputs.json, gold.json, schedule.json, and protocol.json in output_dir.

    Refuses to overwrite existing files or non-empty target directories.
    Runs strictly offline with zero network and zero live model calls.
    """
    out_dir = Path(output_dir)
    target_filenames = ["inputs.json", "gold.json", "schedule.json", "protocol.json"]

    # Refuse existing outputs and non-empty output directory
    if out_dir.exists():
        existing_targets = [f for f in target_filenames if (out_dir / f).exists()]
        if existing_targets:
            raise FileExistsError(
                f"Target output files already exist in {out_dir}: {existing_targets}. "
                "Refusing to overwrite existing frozen artifacts."
            )
        if any(out_dir.iterdir()):
            raise FileExistsError(
                f"Output directory {out_dir} is not empty. Refusing to write to non-empty directory."
            )
    else:
        out_dir.mkdir(parents=True, exist_ok=False)

    resolved_pairs_path = Path(pairs_path)
    if not resolved_pairs_path.is_file():
        resolved_pairs_path = (PROJECT_ROOT / pairs_path).resolve()
    if not resolved_pairs_path.is_file():
        raise FileNotFoundError(f"Pairs dataset file not found: {resolved_pairs_path}")

    resolved_ce_path = Path(counterexamples_path)
    if not resolved_ce_path.is_file():
        resolved_ce_path = (PROJECT_ROOT / counterexamples_path).resolve()
    if not resolved_ce_path.is_file():
        raise FileNotFoundError(f"Counterexamples file not found: {resolved_ce_path}")

    resolved_defaults_path = Path(defaults_path)
    if not resolved_defaults_path.is_file():
        resolved_defaults_path = (PROJECT_ROOT / defaults_path).resolve()
    if not resolved_defaults_path.is_file():
        raise FileNotFoundError(f"Ollama defaults reference file not found: {resolved_defaults_path}")
    if not resolved_ce_path.is_file():
        raise FileNotFoundError(f"Counterexamples file not found: {resolved_ce_path}")

    # Read and hash raw fixture files
    pairs_raw_bytes = resolved_pairs_path.read_bytes()
    pairs_raw_sha256 = hashlib.sha256(pairs_raw_bytes).hexdigest()
    try:
        pairs_data = json.loads(pairs_raw_bytes.decode("utf-8"), parse_constant=_reject_constant)
    except Exception as exc:
        raise ValueError(f"Malformed JSON in pairs file {resolved_pairs_path}: {exc}") from exc
    assert_finite(pairs_data)

    ce_raw_bytes = resolved_ce_path.read_bytes()
    ce_raw_sha256 = hashlib.sha256(ce_raw_bytes).hexdigest()
    try:
        ce_data = json.loads(ce_raw_bytes.decode("utf-8"), parse_constant=_reject_constant)
    except Exception as exc:
        raise ValueError(f"Malformed JSON in counterexamples file {resolved_ce_path}: {exc}") from exc
    assert_finite(ce_data)

    pairs_cases = pairs_data.get("cases", [])
    if not isinstance(pairs_cases, list) or len(pairs_cases) != 28:
        raise ValueError(f"Expected exactly 28 cases in pairs dataset, got {len(pairs_cases)}")

    inputs_cases: list[dict[str, Any]] = []
    gold_cases: list[dict[str, Any]] = []
    gold_labels: dict[str, str] = {}
    dev_status_map: dict[str, str] = {}

    # 1. Build 28 historical controls using current prompt/schema builders
    for case in pairs_cases:
        cid = case["id"]
        claims = [
            {
                "claim_id": "C1",
                "text": case["claim"],
                "evidence_ids": [f"E{i + 1}" for i in range(len(case["evidence"]))],
            }
        ]
        evidence = [
            {
                "evidence_id": f"E{i + 1}",
                "source_id": f"S{i + 1}",
                "quote": q,
            }
            for i, q in enumerate(case["evidence"])
        ]

        # Build messages and schema with existing project builders
        messages = build_verification_messages(claims, evidence)
        schema = build_verification_schema(claims)

        inputs_cases.append(
            {
                "id": cid,
                "source_case_id": cid,
                "kind": "historical_control",
                "claims": claims,
                "evidence": evidence,
                "messages": messages,
                "schema": schema,
            }
        )

        gold_labels[cid] = case["expected"]
        gold_cases.append(
            {
                "id": cid,
                "expected": case["expected"],
                "split": case["split"],
                "kind": case["kind"],
                "source_kind": case["source_kind"],
            }
        )

        dev_status_map[cid] = (
            "historical_dev_control" if case["split"] == "dev" else "historical_reserved_control"
        )

    # 2. Extract and preserve initial b05 verification call
    b05_input, b05_expected = extract_b05_case(ce_data)
    inputs_cases.append(b05_input)

    gold_labels["b05"] = b05_expected
    gold_cases.append(
        {
            "id": "b05",
            "expected": b05_expected,
            "split": "dev",
            "kind": "counterexample_initial_draft",
            "source_kind": "extracto_publico",
        }
    )
    dev_status_map["b05"] = "counterexample_development"

    # Enforce strict label and metadata leakage isolation:
    for inp in inputs_cases:
        case_id = inp["id"]
        for msg in inp["messages"]:
            content = msg.get("content", "")
            if case_id in content:
                raise ValueError(f"Leakage detected: case_id {case_id!r} present in model message content")
            if '"split"' in content or '"dev"' in content or '"reserved"' in content:
                raise ValueError(f"Leakage detected: split annotation present in model message content for {case_id}")
            if '"kind"' in content:
                raise ValueError(f"Leakage detected: 'kind' field present in model message content for {case_id}")

    # Build inputs data
    inputs_data = {
        "schema_version": INPUTS_SCHEMA_VERSION,
        "case_count": len(inputs_cases),
        "cases": inputs_cases,
    }
    assert_finite(inputs_data)

    # Build gold data (kept completely separate from inputs)
    gold_data = {
        "schema_version": GOLD_SCHEMA_VERSION,
        "case_count": len(gold_cases),
        "labels": gold_labels,
        "cases": gold_cases,
    }
    assert_finite(gold_data)

    # Build deterministic schedule
    all_case_ids = [c["id"] for c in inputs_cases]
    schedule_items = build_schedule(all_case_ids, dev_status_map)
    schedule_data = {
        "schema_version": SCHEDULE_SCHEMA_VERSION,
        "total_runs": len(schedule_items),
        "runs_per_arm": len(schedule_items) // len(ARMS),
        "arm_blocks": list(ARMS.keys()),
        "ordering_notes": (
            "Fixed arm blocks A, B, C, D. "
            "Arm A: forward base order. Arm B: reversed base order. "
            "Arm C: rotated base order (half-shift). Arm D: reversed of rotated base order. "
            "Each arm executes all 29 cases once, then appends 2 additional correlated b05 repeats."
        ),
        "items": schedule_items,
    }
    assert_finite(schedule_data)

    # Compute canonical finite SHA-256 hashes
    inputs_canonical_hash = canonical_sha256(inputs_data)
    gold_canonical_hash = canonical_sha256(gold_data)
    schedule_canonical_hash = canonical_sha256(schedule_data)

    # Load and hash versioned public defaults reference file
    defaults_bytes = resolved_defaults_path.read_bytes()
    try:
        defaults_data = json.loads(
            defaults_bytes.decode("utf-8"),
            parse_constant=_reject_constant,
            parse_float=_parse_float,
        )
    except Exception as exc:
        raise ValueError(f"Malformed JSON in defaults reference file {resolved_defaults_path}: {exc}") from exc
    assert_finite(defaults_data)
    defaults_canonical_hash = canonical_sha256(defaults_data)
    defaults_raw_sha256 = compute_file_sha256(resolved_defaults_path)

    # Hash relevant implementation files for reproducibility
    code_hashes: dict[str, dict[str, str]] = {}
    for mod_name, rel_str in REQUIRED_CODE_MODULES.items():
        if mod_name == "compare_verifier_capacity":
            p = Path(__file__).resolve()
        else:
            p = (PROJECT_ROOT / rel_str).resolve()
        if not p.is_file():
            raise FileNotFoundError(f"Required source code file for pin {mod_name!r} not found at {p}")
        h = compute_file_sha256(p)
        if not is_valid_sha256(h):
            raise ValueError(f"Computed invalid SHA-256 for {mod_name}: {h!r}")
        code_hashes[mod_name] = {
            "path": rel_str,
            "sha256": h,
        }

    # Build protocol freeze document
    protocol_data = {
        "schema_version": PROTOCOL_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "design_goal": (
            "Isolate verifier capacity using identical claims, evidence, messages and schema "
            "across 4 arms. Not a test of answer completeness or full RAG."
        ),
        "arms": ARMS,
        "case_counts": {
            "historical_controls": 28,
            "counterexamples": 1,
            "total_unique_cases": len(inputs_cases),
            "runs_per_arm": len(schedule_items) // len(ARMS),
            "total_scheduled_runs": len(schedule_items),
            "arms_count": len(ARMS),
        },
        "defaults_reference": {
            "server_version": defaults_data.get("server_version"),
            "source_url": defaults_data.get("source_url"),
            "defaults": defaults_data.get("defaults"),
            "canonical_sha256": defaults_canonical_hash,
            "file_sha256": defaults_raw_sha256,
        },
        "hashes": {
            "inputs_canonical_sha256": inputs_canonical_hash,
            "schedule_canonical_sha256": schedule_canonical_hash,
            "gold_canonical_sha256": gold_canonical_hash,
            "defaults_canonical_sha256": defaults_canonical_hash,
            "fixtures": {
                "grounding_pairs_sha256": pairs_raw_sha256,
                "counterexamples_sha256": ce_raw_sha256,
                "ollama_defaults_sha256": defaults_raw_sha256,
            },
            "code": code_hashes,
        },
        "disclosures": {
            "hardware_and_cache_confounds": (
                "Fixed arm blocks isolate model swapping using keep_alive=0; "
                "deterministic rotation and reversal mitigate position and KV-cache biases."
            ),
            "dataset_status": (
                "20 dev controls + 8 reserved controls + 1 counterexample (b05). "
                "b05 is a known posthoc development counterexample from baseline stability audit, "
                "not a fresh heldout validation case."
            ),
            "correlated_repeats": (
                "b05 is repeated 3 times per arm (1 in main rotation + 2 additional repeats); "
                "repeats are correlated measurements of the same instance, not fresh heldout data."
            ),
            "arm_d_generation_change": (
                "Arm D changes model generation to qwen3.8:27b and aligns sampling defaults "
                "(presence_penalty=1.5, top_k=20, top_p=0.95, min_p=0.0); differences reflect "
                "both parameter scale and model architecture."
            ),
            "scope_limitation": (
                "Verifier capacity comparison only; does not evaluate search recall, chunking, "
                "or full end-to-end generation."
            ),
        },
    }
    assert_finite(protocol_data)
    protocol_canonical_hash = canonical_sha256(protocol_data)

    # Write all artifacts to output_dir
    inputs_path = out_dir / "inputs.json"
    gold_path = out_dir / "gold.json"
    schedule_path = out_dir / "schedule.json"
    protocol_path = out_dir / "protocol.json"

    inputs_path.write_text(
        json.dumps(inputs_data, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    gold_path.write_text(
        json.dumps(gold_data, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    schedule_path.write_text(
        json.dumps(schedule_data, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    protocol_path.write_text(
        json.dumps(protocol_data, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    return {
        "output_dir": str(out_dir),
        "inputs_path": str(inputs_path),
        "gold_path": str(gold_path),
        "schedule_path": str(schedule_path),
        "protocol_path": str(protocol_path),
        "case_counts": protocol_data["case_counts"],
        "hashes": {
            **protocol_data["hashes"],
            "protocol_canonical_sha256": protocol_canonical_hash,
        },
    }


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Blocks any HTTP redirect for strict security isolation."""

    def redirect_request(self, req: Any, fp: Any, code: int, msg: str, headers: Any, newurl: str) -> Any:
        raise urllib.error.HTTPError(
            req.full_url, code, f"HTTP redirects are blocked: {code} to {newurl}", headers, fp
        )


class OllamaTransportProtocol(Protocol):
    """Explicit protocol matching the transport interface expected by run()."""

    def get_version(self) -> dict[str, Any]:
        """Queries /api/version and returns dict containing version string."""
        ...

    def get_tags(self) -> dict[str, Any]:
        """Queries /api/tags and returns dict containing list of available models."""
        ...

    def get_ps(self) -> dict[str, Any]:
        """Queries /api/ps and returns dict containing list of resident models."""
        ...

    def show(self, model: str) -> dict[str, Any]:
        """Queries /api/show for model metadata and supported capabilities."""
        ...

    def unload(self, model: str) -> dict[str, Any]:
        """Unloads model from resident memory using keep_alive=0."""
        ...

    def chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Sends chat payload to /api/chat and returns raw response dict."""
        ...


class OllamaTransport:
    """Dedicated stdlib Ollama adapter strictly adhering to loopback 127.0.0.1:11434 and contract constraints."""

    def __init__(
        self,
        base_url: str = DEFAULT_OLLAMA_URL,
        timeout: float = DEFAULT_RUN_TIMEOUT,
        allowed_models: tuple[str, ...] = ALLOWED_OLLAMA_MODELS,
        opener: urllib.request.OpenerDirector | None = None,
    ) -> None:
        if base_url != DEFAULT_OLLAMA_URL:
            raise ValueError(
                f"Invalid Ollama base_url: {base_url!r}. Only dedicated loopback {DEFAULT_OLLAMA_URL} is permitted."
            )
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0:
            raise ValueError(f"timeout must be a positive number, got {timeout!r}")
        if not isinstance(allowed_models, tuple) or set(allowed_models) != set(ALLOWED_OLLAMA_MODELS):
            raise ValueError(f"allowed_models must be exact allowed models: {ALLOWED_OLLAMA_MODELS}")

        self.base_url = base_url
        self.timeout = float(timeout)
        self.allowed_models = allowed_models
        if opener is not None:
            self._opener = opener
        else:
            self._opener = urllib.request.build_opener(
                urllib.request.ProxyHandler({}),
                _NoRedirectHandler(),
            )

    def _open_request(self, req: urllib.request.Request) -> bytes:
        try:
            with self._opener.open(req, timeout=self.timeout) as resp:
                raw = resp.read(MAX_RUN_RESPONSE_BYTES + 1)
                if len(raw) > MAX_RUN_RESPONSE_BYTES:
                    capped = raw[:MAX_RUN_RESPONSE_BYTES]
                    capped_text = capped.decode("utf-8", errors="replace")
                    raise OllamaResponseError(
                        f"Response from Ollama exceeded 4 MiB limit ({len(raw)} bytes received)",
                        raw_text=capped_text,
                        is_truncated=True,
                        status_code=getattr(resp, "status", 200),
                    )
                return raw
        except urllib.error.HTTPError as exc:
            raw_err = b""
            try:
                try:
                    raw_err = exc.read(MAX_RUN_RESPONSE_BYTES + 1)
                finally:
                    exc.close()
            except Exception:
                raw_err = b""
            is_trunc = len(raw_err) > MAX_RUN_RESPONSE_BYTES
            capped_err = raw_err[:MAX_RUN_RESPONSE_BYTES]
            body = capped_err.decode("utf-8", errors="replace")
            raise OllamaResponseError(
                f"HTTP error {exc.code} from Ollama: {exc.reason}. Response: {body}",
                raw_text=body,
                is_truncated=is_trunc,
                status_code=exc.code,
            ) from exc
        except urllib.error.URLError as exc:
            raise ConnectionError(f"Connection error contacting Ollama at {self.base_url}: {exc.reason}") from exc
        except TimeoutError as exc:
            raise TimeoutError(f"Timeout contacting Ollama after {self.timeout}s") from exc

    def get_version(self) -> dict[str, Any]:
        """Queries /api/version to verify server version against versioned public defaults reference."""
        url = f"{self.base_url}/api/version"
        req = urllib.request.Request(url, method="GET")
        raw = self._open_request(req)
        try:
            data = json.loads(raw.decode("utf-8"), parse_constant=_reject_constant, parse_float=_parse_float)
        except Exception as exc:
            raise ValueError(f"Malformed JSON from /api/version: {exc}") from exc
        assert_finite(data)
        if not isinstance(data, dict) or "version" not in data:
            raise ValueError("Invalid response format from /api/version: expected 'version' field")
        return data

    def get_tags(self) -> dict[str, Any]:
        """Queries /api/tags to list available models and their SHA-256 digests."""
        url = f"{self.base_url}/api/tags"
        req = urllib.request.Request(url, method="GET")
        raw = self._open_request(req)
        try:
            data = json.loads(raw.decode("utf-8"), parse_constant=_reject_constant, parse_float=_parse_float)
        except Exception as exc:
            raise ValueError(f"Malformed JSON from /api/tags: {exc}") from exc
        assert_finite(data)
        if not isinstance(data, dict) or "models" not in data or not isinstance(data["models"], list):
            raise ValueError("Invalid response format from /api/tags: expected 'models' list")
        return data

    def get_ps(self) -> dict[str, Any]:
        """Queries /api/ps to inspect currently loaded resident models and placement."""
        url = f"{self.base_url}/api/ps"
        req = urllib.request.Request(url, method="GET")
        raw = self._open_request(req)
        try:
            data = json.loads(raw.decode("utf-8"), parse_constant=_reject_constant, parse_float=_parse_float)
        except Exception as exc:
            raise ValueError(f"Malformed JSON from /api/ps: {exc}") from exc
        assert_finite(data)
        if not isinstance(data, dict) or "models" not in data or not isinstance(data["models"], list):
            raise ValueError("Invalid response format from /api/ps: expected 'models' list")
        return data

    def show(self, model: str) -> dict[str, Any]:
        """Queries /api/show for model details, parameter boundaries, and capabilities."""
        if model not in self.allowed_models:
            raise ValueError(f"Model {model!r} is not allowed. Allowed models: {self.allowed_models}")
        url = f"{self.base_url}/api/show"
        payload = {"model": model}
        body_bytes = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=body_bytes,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        raw = self._open_request(req)
        try:
            data = json.loads(raw.decode("utf-8"), parse_constant=_reject_constant, parse_float=_parse_float)
        except Exception as exc:
            raise ValueError(f"Malformed JSON from /api/show: {exc}") from exc
        assert_finite(data)
        if not isinstance(data, dict):
            raise ValueError("Response from /api/show must be a JSON object")
        return data

    def unload(self, model: str) -> dict[str, Any]:
        """Unloads resident model from memory using keep_alive=0."""
        if model not in self.allowed_models:
            raise ValueError(f"Model {model!r} is not allowed. Allowed models: {self.allowed_models}")
        url = f"{self.base_url}/api/generate"
        payload = {"model": model, "keep_alive": 0}
        body_bytes = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=body_bytes,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        raw = self._open_request(req)
        try:
            data = json.loads(raw.decode("utf-8"), parse_constant=_reject_constant, parse_float=_parse_float)
        except Exception:
            data = {"status": "unloaded"}
        return data

    def chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Sends chat request to /api/chat with strict payload validation."""
        assert_finite(payload)
        model = payload.get("model")
        if model not in self.allowed_models:
            raise ValueError(f"Model {model!r} is not allowed. Allowed models: {self.allowed_models}")

        url = f"{self.base_url}/api/chat"
        body_bytes = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=body_bytes,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        raw = self._open_request(req)
        raw_text = raw.decode("utf-8", errors="replace")
        try:
            data = json.loads(raw_text, parse_constant=_reject_constant, parse_float=_parse_float)
        except Exception as exc:
            raise OllamaResponseError(
                f"Malformed JSON from /api/chat: {exc}",
                raw_text=raw_text,
                is_truncated=False,
                status_code=200,
            ) from exc
        try:
            assert_finite(data)
        except Exception as exc:
            raise OllamaResponseError(
                f"Non-finite JSON from /api/chat: {exc}",
                raw_text=raw_text,
                is_truncated=False,
                status_code=200,
            ) from exc
        if not isinstance(data, dict):
            raise OllamaResponseError(
                "Response from /api/chat must be a JSON object",
                raw_text=raw_text,
                is_truncated=False,
                status_code=200,
            )
        return data


def unload_resident_models(transport: Any) -> list[str]:
    """Unloads only models currently reported as resident by /api/ps using keep_alive=0."""
    unloaded: list[str] = []
    try:
        ps_data = transport.get_ps()
        models = ps_data.get("models", [])
        if isinstance(models, list):
            for m in models:
                if isinstance(m, dict):
                    name = m.get("name") or m.get("model")
                    if isinstance(name, str) and name:
                        try:
                            transport.unload(name)
                            unloaded.append(name)
                        except Exception:
                            pass
    except Exception:
        pass
    return unloaded


def check_model_supports_thinking(show_data: dict[str, Any]) -> bool:
    """Checks whether model metadata from /api/show supports think values."""
    if not isinstance(show_data, dict):
        return False
    if show_data.get("supports_thinking") is False:
        return False
    caps = show_data.get("capabilities")
    if isinstance(caps, dict) and caps.get("thinking") is False:
        return False
    if isinstance(caps, list) and len(caps) > 0 and "thinking" not in caps and "think" not in caps:
        return False
    unsupported = show_data.get("unsupported_features")
    if isinstance(unsupported, list) and any(f in ("think", "thinking") for f in unsupported):
        return False
    return True


def run(
    protocol_path: Path | str,
    output_path: Path | str,
    *,
    transport: Any = None,
) -> dict[str, Any]:
    """Phase 2 execution runner.

    Executes frozen schedule sequentially over dedicated Ollama transport.
    Captures full exact requests, raw responses, validated verdicts, stats, and thinking metrics.
    Performs atomic checkpoints after every single attempt.
    """
    prot_path = Path(protocol_path)
    if not prot_path.is_file():
        raise FileNotFoundError(f"Protocol file not found: {prot_path}")

    out_path = Path(output_path)
    if out_path.exists():
        raise FileExistsError(f"Target output file already exists: {out_path}. Refusing to overwrite.")

    # 1. Validate frozen protocol and pins BEFORE any network/transport access
    prot_bytes = prot_path.read_bytes()
    try:
        protocol_data = json.loads(prot_bytes.decode("utf-8"), parse_constant=_reject_constant, parse_float=_parse_float)
    except Exception as exc:
        raise ValueError(f"Malformed JSON in protocol file: {exc}") from exc
    assert_finite(protocol_data)

    if protocol_data.get("schema_version") != PROTOCOL_SCHEMA_VERSION:
        raise ValueError(
            f"Unexpected protocol schema_version: {protocol_data.get('schema_version')!r}, "
            f"expected {PROTOCOL_SCHEMA_VERSION!r}"
        )

    parent_dir = prot_path.parent
    inputs_path = parent_dir / "inputs.json"
    schedule_path = parent_dir / "schedule.json"

    if not inputs_path.is_file():
        raise FileNotFoundError(f"Inputs file not found in protocol directory: {inputs_path}")
    if not schedule_path.is_file():
        raise FileNotFoundError(f"Schedule file not found in protocol directory: {schedule_path}")

    # Note: run strictly never opens or loads gold.json!
    inputs_raw = inputs_path.read_bytes()
    try:
        inputs_data = json.loads(inputs_raw.decode("utf-8"), parse_constant=_reject_constant, parse_float=_parse_float)
    except Exception as exc:
        raise ValueError(f"Malformed JSON in inputs file: {exc}") from exc
    assert_finite(inputs_data)

    if inputs_data.get("schema_version") != INPUTS_SCHEMA_VERSION:
        raise ValueError(
            f"Unexpected inputs schema_version: {inputs_data.get('schema_version')!r}, "
            f"expected {INPUTS_SCHEMA_VERSION!r}"
        )

    schedule_raw = schedule_path.read_bytes()
    try:
        schedule_data = json.loads(schedule_raw.decode("utf-8"), parse_constant=_reject_constant, parse_float=_parse_float)
    except Exception as exc:
        raise ValueError(f"Malformed JSON in schedule file: {exc}") from exc
    assert_finite(schedule_data)

    if schedule_data.get("schema_version") != SCHEDULE_SCHEMA_VERSION:
        raise ValueError(
            f"Unexpected schedule schema_version: {schedule_data.get('schema_version')!r}, "
            f"expected {SCHEDULE_SCHEMA_VERSION!r}"
        )

    # Validate finite canonical SHA-256 hashes against frozen protocol pins
    prot_hashes = protocol_data.get("hashes")
    if not isinstance(prot_hashes, dict) or not prot_hashes:
        raise ValueError("protocol.json missing valid 'hashes' dictionary")

    required_hash_keys = {
        "inputs_canonical_sha256",
        "schedule_canonical_sha256",
        "defaults_canonical_sha256",
        "fixtures",
        "code",
    }
    missing_hash_keys = required_hash_keys - set(prot_hashes.keys())
    if missing_hash_keys:
        raise ValueError(f"protocol.json hashes missing required keys: {sorted(missing_hash_keys)}")

    expected_inputs_hash = prot_hashes.get("inputs_canonical_sha256")
    expected_schedule_hash = prot_hashes.get("schedule_canonical_sha256")
    expected_defaults_hash = prot_hashes.get("defaults_canonical_sha256")

    if not is_valid_sha256(expected_inputs_hash):
        raise ValueError(f"Missing or invalid protocol inputs_canonical_sha256 pin: {expected_inputs_hash!r}")
    if not is_valid_sha256(expected_schedule_hash):
        raise ValueError(f"Missing or invalid protocol schedule_canonical_sha256 pin: {expected_schedule_hash!r}")
    if not is_valid_sha256(expected_defaults_hash):
        raise ValueError(f"Missing or invalid protocol defaults_canonical_sha256 pin: {expected_defaults_hash!r}")

    actual_inputs_hash = canonical_sha256(inputs_data)
    actual_schedule_hash = canonical_sha256(schedule_data)

    if actual_inputs_hash != expected_inputs_hash:
        raise ValueError(
            f"Tampering detected: inputs.json canonical SHA-256 ({actual_inputs_hash}) "
            f"does not match protocol pin ({expected_inputs_hash})"
        )
    if actual_schedule_hash != expected_schedule_hash:
        raise ValueError(
            f"Tampering detected: schedule.json canonical SHA-256 ({actual_schedule_hash}) "
            f"does not match protocol pin ({expected_schedule_hash})"
        )

    # Validate defaults file exists and canonical sha256 matches
    defaults_file = DEFAULT_OLLAMA_DEFAULTS_PATH
    if not defaults_file.is_file():
        raise FileNotFoundError(f"Ollama defaults reference file not found at {defaults_file}")
    defaults_raw = defaults_file.read_bytes()
    try:
        defaults_data = json.loads(defaults_raw.decode("utf-8"), parse_constant=_reject_constant, parse_float=_parse_float)
    except Exception as exc:
        raise ValueError(f"Malformed JSON in defaults reference file: {exc}") from exc
    assert_finite(defaults_data)
    actual_defaults_hash = canonical_sha256(defaults_data)
    if actual_defaults_hash != expected_defaults_hash:
        raise ValueError(
            f"Tampering detected: defaults canonical SHA-256 ({actual_defaults_hash}) "
            f"does not match protocol pin ({expected_defaults_hash})"
        )

    # Validate fixtures
    fixtures_pins = prot_hashes.get("fixtures")
    if not isinstance(fixtures_pins, dict):
        raise ValueError("protocol.json missing valid 'hashes.fixtures' dictionary")
    for fix_key in ["grounding_pairs_sha256", "counterexamples_sha256", "ollama_defaults_sha256"]:
        h_val = fixtures_pins.get(fix_key)
        if not is_valid_sha256(h_val):
            raise ValueError(f"Missing or invalid SHA-256 in protocol fixtures for {fix_key!r}: {h_val!r}")

    # Validate declared code pins: all REQUIRED_CODE_MODULES must be present exactly
    code_pins = prot_hashes.get("code")
    if not isinstance(code_pins, dict) or not code_pins:
        raise ValueError("protocol.json missing valid 'hashes.code' dictionary")

    expected_modules = set(REQUIRED_CODE_MODULES.keys())
    actual_modules = set(code_pins.keys())
    if actual_modules != expected_modules:
        missing_mods = expected_modules - actual_modules
        extra_mods = actual_modules - expected_modules
        err_msg = []
        if missing_mods:
            err_msg.append(f"missing required code pins: {sorted(missing_mods)}")
        if extra_mods:
            err_msg.append(f"extra unknown code pins: {sorted(extra_mods)}")
        raise ValueError(f"Code pins mismatch in protocol.json: {'; '.join(err_msg)}")

    for mod_name, pin_info in code_pins.items():
        if isinstance(pin_info, dict):
            rel_path = pin_info.get("path") or REQUIRED_CODE_MODULES.get(mod_name)
            expected_sha = pin_info.get("sha256")
        elif isinstance(pin_info, str):
            rel_path = REQUIRED_CODE_MODULES.get(mod_name)
            expected_sha = pin_info
        else:
            raise ValueError(f"Invalid code pin structure for {mod_name!r}: {pin_info!r}")

        if not rel_path:
            raise ValueError(f"Unknown code pin module {mod_name!r} with no known path")
        if not is_valid_sha256(expected_sha):
            raise ValueError(f"Missing or invalid SHA-256 string for code pin {mod_name!r}: {expected_sha!r}")

        if mod_name == "compare_verifier_capacity":
            p = Path(__file__).resolve()
        else:
            p = (PROJECT_ROOT / rel_path).resolve()

        if not p.is_file():
            raise FileNotFoundError(f"Declared code file for pin {mod_name!r} not found at {p}")

        actual_sha = compute_file_sha256(p)
        if actual_sha != expected_sha:
            raise ValueError(
                f"Tampering/stale code detected for {mod_name}: "
                f"file SHA-256 ({actual_sha}) does not match protocol pin ({expected_sha})"
            )

    # Validate inputs case dictionary map
    cases_list = inputs_data.get("cases", [])
    if not isinstance(cases_list, list) or len(cases_list) != 29:
        raise ValueError(f"Expected 29 cases in inputs.json, got {len(cases_list)}")
    cases_by_id: dict[str, dict[str, Any]] = {c["id"]: c for c in cases_list}

    schedule_items = schedule_data.get("items", [])
    if not isinstance(schedule_items, list) or len(schedule_items) != 124:
        raise ValueError(f"Expected 124 items in schedule.json, got {len(schedule_items)}")

    # 2. Instantiate or validate transport
    if transport is None:
        transport = OllamaTransport()

    # Query and validate server version against versioned public defaults reference
    version_info = transport.get_version()
    if not isinstance(version_info, dict) or "version" not in version_info:
        raise ValueError(f"Invalid response from transport.get_version(): {version_info!r}")
    server_ver = version_info["version"]
    expected_server_version = defaults_data.get("server_version")
    if server_ver != expected_server_version:
        raise ValueError(
            f"Ollama server version mismatch: server reported {server_ver!r}, "
            f"expected {expected_server_version!r} matching versioned defaults reference"
        )

    run_start_time = datetime.now(timezone.utc).isoformat()

    # 3. Unload resident models at start
    initial_unloaded = unload_resident_models(transport)

    # 4. Pin model digests at run START
    tags_data = transport.get_tags()
    tags_models = tags_data.get("models", [])
    pinned_initial_digests: dict[str, str] = {}
    for item in tags_models:
        if isinstance(item, dict):
            name = item.get("name") or item.get("model")
            digest = item.get("digest")
            if isinstance(name, str) and isinstance(digest, str) and digest.strip():
                pinned_initial_digests[name] = digest.strip()

    # 5. Query /api/show and resolve effective sampling options
    models_show: dict[str, Any] = {}
    models_thinking_supported: dict[str, bool] = {}
    models_show_params: dict[str, dict[str, Any]] = {}
    for m_name in ALLOWED_OLLAMA_MODELS:
        try:
            m_show = transport.show(m_name)
            models_show[m_name] = m_show
            models_thinking_supported[m_name] = check_model_supports_thinking(m_show)
            models_show_params[m_name] = parse_show_parameters(m_show.get("parameters"))
        except Exception as exc:
            raise RuntimeError(f"Cannot verify model sampling metadata for {m_name}: {exc}") from exc

    server_defaults = defaults_data.get("defaults", {})
    arm_effective_sampling: dict[str, dict[str, Any]] = {}
    for arm_id, arm_def in ARMS.items():
        m_name = arm_def["model"]
        arm_effective_sampling[arm_id] = resolve_effective_sampling(
            server_defaults,
            models_show_params.get(m_name, {}),
            arm_def["options"],
        )

    # Validate sampling match between Arm B and Arm D
    sampling_matched_b_d, sampling_mismatches_b_d = validate_sampling_match(
        arm_effective_sampling.get("B", {}),
        arm_effective_sampling.get("D", {}),
    )
    block_arm_d = not sampling_matched_b_d

    server_metadata = {
        "server_version": server_ver,
        "tags": tags_data,
        "pinned_initial_digests": pinned_initial_digests,
        "models_show": models_show,
        "thinking_supported": models_thinking_supported,
        "effective_sampling": arm_effective_sampling,
        "sampling_matched_B_D": sampling_matched_b_d,
        "sampling_mismatches_B_D": sampling_mismatches_b_d if not sampling_matched_b_d else None,
        "initial_unloaded_models": initial_unloaded,
    }

    arm_placements: dict[str, Any] = {}
    attempts: list[dict[str, Any]] = []

    def compute_completion_status(completed: bool) -> str:
        if not completed or len(attempts) < len(schedule_items):
            return "incomplete/skipped"
        total_unattempted = sum(
            1 for a in attempts if a.get("status") == "unattempted" or not a.get("attempted")
        )
        total_errors = sum(1 for a in attempts if a.get("status") == "error")
        if total_unattempted > 0:
            return "incomplete/skipped"
        if total_errors > 0:
            return "completed_with_errors"
        return "completed"

    # Helper function for atomic checkpoint write
    def save_checkpoint(completed: bool = False) -> None:
        checkpoint_data = {
            "schema_version": RUN_RESULTS_SCHEMA_VERSION,
            "created_at": run_start_time,
            "completed_at": datetime.now(timezone.utc).isoformat() if completed else None,
            "completion_status": compute_completion_status(completed),
            "protocol_path": str(prot_path),
            "protocol_canonical_sha256": canonical_sha256(protocol_data),
            "inputs_canonical_sha256": actual_inputs_hash,
            "schedule_canonical_sha256": actual_schedule_hash,
            "total_scheduled": len(schedule_items),
            "total_attempted": sum(1 for a in attempts if a.get("attempted") is True),
            "total_successful": sum(1 for a in attempts if a.get("status") == "success"),
            "total_technical_errors": sum(1 for a in attempts if a.get("status") == "error"),
            "total_unattempted": sum(1 for a in attempts if a.get("status") == "unattempted"),
            "server_metadata": server_metadata,
            "arm_placements": arm_placements,
            "attempts": attempts,
        }
        assert_finite(checkpoint_data)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = out_path.parent / f"{out_path.name}.tmp"
        content_bytes = (
            json.dumps(checkpoint_data, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8") + b"\n"
        )
        tmp_path.write_bytes(content_bytes)
        tmp_path.replace(out_path)

    # Initial checkpoint (empty attempts)
    save_checkpoint(completed=False)

    # Group schedule items by arm block
    arm_blocks = ["A", "B", "C", "D"]
    for arm_id in arm_blocks:
        arm_def = ARMS[arm_id]
        arm_model = arm_def["model"]
        arm_think = arm_def["think"]
        arm_options = arm_def["options"]

        arm_items = [it for it in schedule_items if it["arm_id"] == arm_id]

        # 1. Check if Arm D is blocked due to sampling mismatch with Arm B
        if arm_id == "D" and block_arm_d:
            mismatch_err = f"Arm D blocked: effective sampling mismatch with Arm B: {sampling_mismatches_b_d}"
            arm_placements[arm_id] = {
                "error": mismatch_err,
                "attempted": False,
                "effective_sampling": arm_effective_sampling.get("D"),
            }
            for it in arm_items:
                unattempted_record = {
                    "schedule_index": it["schedule_index"],
                    "block_index": it["block_index"],
                    "arm_id": arm_id,
                    "case_id": it["case_id"],
                    "repeat_index": it["repeat_index"],
                    "is_b05": it["is_b05"],
                    "is_correlated_repeat": it["is_correlated_repeat"],
                    "development_status": it["development_status"],
                    "attempted": False,
                    "status": "unattempted",
                    "error_type": "effective_sampling_mismatch",
                    "error_message": mismatch_err,
                    "requested_model": arm_model,
                    "returned_model": None,
                    "model_digest": None,
                    "request": None,
                    "raw_response": None,
                    "parsed_verdict": None,
                    "thinking_length": 0,
                    "thinking_sha256": None,
                    "stats": None,
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                }
                attempts.append(unattempted_record)
                save_checkpoint(completed=False)
            continue

        # 2. Validate model digest availability and require match with initial pinned digest
        initial_pinned = pinned_initial_digests.get(arm_model)
        try:
            arm_tags = transport.get_tags()
            available_digests = {
                (m.get("name") or m.get("model")): m.get("digest")
                for m in arm_tags.get("models", [])
                if isinstance(m, dict) and m.get("digest")
            }
        except Exception:
            available_digests = {}

        arm_digest = available_digests.get(arm_model)
        if not initial_pinned or not arm_digest or arm_digest != initial_pinned:
            err_msg = (
                f"Model {arm_model!r} digest changed or unavailable before arm {arm_id} "
                f"(initial_pinned={initial_pinned!r}, current_available={arm_digest!r})"
            )
            arm_placements[arm_id] = {
                "error": err_msg,
                "attempted": False,
            }
            for it in arm_items:
                unattempted_record = {
                    "schedule_index": it["schedule_index"],
                    "block_index": it["block_index"],
                    "arm_id": arm_id,
                    "case_id": it["case_id"],
                    "repeat_index": it["repeat_index"],
                    "is_b05": it["is_b05"],
                    "is_correlated_repeat": it["is_correlated_repeat"],
                    "development_status": it["development_status"],
                    "attempted": False,
                    "status": "unattempted",
                    "error_type": "model_or_digest_unavailable",
                    "error_message": err_msg,
                    "requested_model": arm_model,
                    "returned_model": None,
                    "model_digest": None,
                    "request": None,
                    "raw_response": None,
                    "parsed_verdict": None,
                    "thinking_length": 0,
                    "thinking_sha256": None,
                    "stats": None,
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                }
                attempts.append(unattempted_record)
                save_checkpoint(completed=False)
            continue

        # Model and digest are available:
        # 1. Unload resident models between arms via keep_alive=0
        unload_resident_models(transport)

        # 2. Record /api/ps placement per arm
        ps_placement = None
        try:
            ps_placement = transport.get_ps()
        except Exception as exc:
            ps_placement = {"error": str(exc)}

        arm_thinking_supported = models_thinking_supported.get(arm_model, True)

        arm_placements[arm_id] = {
            "model": arm_model,
            "model_digest": arm_digest,
            "ps_placement": ps_placement,
            "effective_sampling": arm_effective_sampling.get(arm_id),
            "supports_thinking": arm_thinking_supported,
            "attempted": True,
        }

        # Execute scheduled cases sequentially for this arm
        for it in arm_items:
            cid = it["case_id"]
            case_input = cases_by_id[cid]

            # Enforce 12288 message bytes limit preflight
            messages_bytes = json.dumps(case_input["messages"], ensure_ascii=False).encode("utf-8")
            if len(messages_bytes) > MAX_MESSAGES_BYTES_VERIFICATION:
                unattempted_record = {
                    "schedule_index": it["schedule_index"],
                    "block_index": it["block_index"],
                    "arm_id": arm_id,
                    "case_id": cid,
                    "repeat_index": it["repeat_index"],
                    "is_b05": it["is_b05"],
                    "is_correlated_repeat": it["is_correlated_repeat"],
                    "development_status": it["development_status"],
                    "attempted": False,
                    "status": "unattempted",
                    "error_type": "messages_bytes_exceeded",
                    "error_message": f"Preflight check failed: Messages size {len(messages_bytes)} exceeds {MAX_MESSAGES_BYTES_VERIFICATION} bytes",
                    "requested_model": arm_model,
                    "returned_model": None,
                    "model_digest": arm_digest,
                    "request": None,
                    "raw_response": None,
                    "parsed_verdict": None,
                    "thinking_length": 0,
                    "thinking_sha256": None,
                    "stats": None,
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                }
                attempts.append(unattempted_record)
                save_checkpoint(completed=False)
                continue

            # Check unsupported thinking preflight
            if arm_think and not arm_thinking_supported:
                unattempted_record = {
                    "schedule_index": it["schedule_index"],
                    "block_index": it["block_index"],
                    "arm_id": arm_id,
                    "case_id": cid,
                    "repeat_index": it["repeat_index"],
                    "is_b05": it["is_b05"],
                    "is_correlated_repeat": it["is_correlated_repeat"],
                    "development_status": it["development_status"],
                    "attempted": False,
                    "status": "unattempted",
                    "error_type": "unsupported_think",
                    "error_message": f"Preflight check failed: Thinking requested for arm {arm_id} but model {arm_model} does not support think",
                    "requested_model": arm_model,
                    "returned_model": None,
                    "model_digest": arm_digest,
                    "request": None,
                    "raw_response": None,
                    "parsed_verdict": None,
                    "thinking_length": 0,
                    "thinking_sha256": None,
                    "stats": None,
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                }
                attempts.append(unattempted_record)
                save_checkpoint(completed=False)
                continue

            # Construct exact request payload and enter transport.chat()
            exact_request = {
                "model": arm_model,
                "messages": copy.deepcopy(case_input["messages"]),
                "stream": False,
                "think": arm_think,
                "format": copy.deepcopy(case_input["schema"]),
                "options": copy.deepcopy(arm_options),
            }

            raw_response = None
            transport_error = None
            try:
                raw_response = transport.chat(exact_request)
            except Exception as exc:
                transport_error = exc

            if transport_error is not None:
                if isinstance(transport_error, OllamaResponseError):
                    raw_resp_record: dict[str, Any] | None = {
                        "raw_text": transport_error.raw_text,
                        "is_truncated": transport_error.is_truncated,
                        "status_code": transport_error.status_code,
                        "error": str(transport_error),
                    }
                else:
                    raw_resp_record = None

                attempt_record = {
                    "schedule_index": it["schedule_index"],
                    "block_index": it["block_index"],
                    "arm_id": arm_id,
                    "case_id": cid,
                    "repeat_index": it["repeat_index"],
                    "is_b05": it["is_b05"],
                    "is_correlated_repeat": it["is_correlated_repeat"],
                    "development_status": it["development_status"],
                    "attempted": True,
                    "status": "error",
                    "error_type": type(transport_error).__name__,
                    "error_message": str(transport_error),
                    "requested_model": arm_model,
                    "returned_model": None,
                    "model_digest": arm_digest,
                    "request": exact_request,
                    "raw_response": raw_resp_record,
                    "parsed_verdict": None,
                    "thinking_length": 0,
                    "thinking_sha256": None,
                    "stats": None,
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                }
                attempts.append(attempt_record)
                save_checkpoint(completed=False)
                continue

            # Completed raw response is captured before any validation that can fail
            validation_error_type = None
            validation_error_message = None
            parsed_verdict = None

            try:
                # 1. Finite JSON validation on response
                assert_finite(raw_response)

                # 2. Check done and done_reason
                if raw_response.get("done") is not True:
                    raise ValueError(f"Response done is not True: {raw_response.get('done')!r}")
                done_reason = raw_response.get("done_reason")
                if done_reason != "stop":
                    raise ValueError(f"Response done_reason is {done_reason!r}, expected 'stop'")

                # 3. Model check - require exact returned model
                returned_model = raw_response.get("model")
                if returned_model != arm_model:
                    raise ValueError(f"Returned model {returned_model!r} does not match requested {arm_model!r}")

                # 4. Context budget constraint: require non-bool non-negative integer prompt_eval_count
                prompt_eval_count = raw_response.get("prompt_eval_count")
                if (
                    prompt_eval_count is None
                    or isinstance(prompt_eval_count, bool)
                    or not isinstance(prompt_eval_count, int)
                    or prompt_eval_count < 0
                ):
                    raise ValueError(
                        f"Missing or invalid prompt_eval_count: {prompt_eval_count!r}. "
                        "Must be a non-boolean non-negative integer to verify context budget."
                    )

                num_predict = arm_options["num_predict"]
                if prompt_eval_count + num_predict + CONTEXT_SAFETY_MARGIN > MAX_CONTEXT_WINDOW:
                    raise ValueError(
                        f"Context budget exceeded: prompt_eval_count ({prompt_eval_count}) + "
                        f"num_predict ({num_predict}) + {CONTEXT_SAFETY_MARGIN} = "
                        f"{prompt_eval_count + num_predict + CONTEXT_SAFETY_MARGIN} > {MAX_CONTEXT_WINDOW}"
                    )

                # 5. Extract and parse message content JSON
                msg_obj = raw_response.get("message")
                if not isinstance(msg_obj, dict):
                    raise ValueError("Response missing 'message' object")
                content_str = msg_obj.get("content")
                if not isinstance(content_str, str) or not content_str.strip():
                    raise ValueError("Response message 'content' is empty or not a string")

                try:
                    content_json = json.loads(content_str, parse_constant=_reject_constant, parse_float=_parse_float)
                except Exception as exc:
                    raise ValueError(f"Malformed JSON in response content: {exc}") from exc
                assert_finite(content_json)

                # 6. Normalize and validate payload against frozen claims and evidence
                normalized_payload = normalize_verification_payload(content_json, claims=case_input["claims"])
                validated_payload = validate_verification_payload(
                    normalized_payload,
                    claims=case_input["claims"],
                    evidence=case_input["evidence"],
                )
                parsed_verdict = validated_payload

            except Exception as exc:
                validation_error_type = type(exc).__name__
                validation_error_message = str(exc)

            # Extract thinking length and SHA-256 without printing/logging thinking text
            thinking_str = None
            if isinstance(raw_response.get("message"), dict) and "thinking" in raw_response["message"]:
                t_val = raw_response["message"]["thinking"]
                if isinstance(t_val, str):
                    thinking_str = t_val
            elif "thinking" in raw_response and isinstance(raw_response["thinking"], str):
                thinking_str = raw_response["thinking"]

            if thinking_str is not None:
                thinking_len = len(thinking_str)
                thinking_hash = hashlib.sha256(thinking_str.encode("utf-8")).hexdigest()
            else:
                thinking_len = 0
                thinking_hash = None

            # Extract stats
            stats = {
                "total_duration": raw_response.get("total_duration"),
                "load_duration": raw_response.get("load_duration"),
                "prompt_eval_count": raw_response.get("prompt_eval_count"),
                "prompt_eval_duration": raw_response.get("prompt_eval_duration"),
                "eval_count": raw_response.get("eval_count"),
                "eval_duration": raw_response.get("eval_duration"),
            }

            attempt_record = {
                "schedule_index": it["schedule_index"],
                "block_index": it["block_index"],
                "arm_id": arm_id,
                "case_id": cid,
                "repeat_index": it["repeat_index"],
                "is_b05": it["is_b05"],
                "is_correlated_repeat": it["is_correlated_repeat"],
                "development_status": it["development_status"],
                "attempted": True,
                "status": "success" if validation_error_type is None else "error",
                "error_type": validation_error_type,
                "error_message": validation_error_message,
                "requested_model": arm_model,
                "returned_model": raw_response.get("model"),
                "model_digest": arm_digest,
                "request": exact_request,
                "raw_response": sanitize_finite(raw_response),
                "parsed_verdict": parsed_verdict,
                "thinking_length": thinking_len,
                "thinking_sha256": thinking_hash,
                "stats": sanitize_finite(stats),
                "completed_at": datetime.now(timezone.utc).isoformat(),
            }
            attempts.append(attempt_record)
            save_checkpoint(completed=False)

    # 5. At end of run, unload generator models returned by /api/ps
    final_unloaded = unload_resident_models(transport)
    server_metadata["final_unloaded_models"] = final_unloaded

    # Final checkpoint marked completed
    save_checkpoint(completed=True)

    return {
        "output_path": str(out_path),
        "protocol_path": str(prot_path),
        "completion_status": compute_completion_status(True),
        "total_scheduled": len(schedule_items),
        "total_attempted": sum(1 for a in attempts if a.get("attempted") is True),
        "total_successful": sum(1 for a in attempts if a.get("status") == "success"),
        "total_technical_errors": sum(1 for a in attempts if a.get("status") == "error"),
        "total_unattempted": sum(1 for a in attempts if a.get("status") == "unattempted"),
        "server_metadata": server_metadata,
        "arm_placements": arm_placements,
    }


def extract_verdict_label(parsed: Any) -> str:
    """Extracts verdict string ('supported', 'contradicted', 'insufficient_evidence') from parsed verdict."""
    if isinstance(parsed, str):
        val = parsed.strip()
        if val in ALLOWED_VERDICT_LABELS:
            return val
        raise ValueError(f"Unknown predicted label {parsed!r}")
    if isinstance(parsed, dict):
        if "verdict" in parsed and isinstance(parsed["verdict"], str):
            val = parsed["verdict"].strip()
            if val in ALLOWED_VERDICT_LABELS:
                return val
            raise ValueError(f"Unknown predicted label {val!r}")
        verdicts = parsed.get("verdicts")
        if isinstance(verdicts, list) and len(verdicts) > 0 and isinstance(verdicts[0], dict):
            val = verdicts[0].get("verdict")
            if isinstance(val, str) and val.strip() in ALLOWED_VERDICT_LABELS:
                return val.strip()
            raise ValueError(f"Unknown predicted label in verdicts list: {val!r}")
    raise ValueError(f"Unable to extract valid verdict label from parsed verdict: {parsed!r}")


def compute_numeric_summary(values: list[float | int]) -> dict[str, Any]:
    """Computes count, median, min, max, range for numeric lists without floating non-finites."""
    if not values:
        return {
            "count": 0,
            "median": None,
            "min": None,
            "max": None,
            "range": None,
        }
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    if n % 2 == 1:
        med = float(sorted_vals[n // 2])
    else:
        med = float(sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2.0
    mn = float(sorted_vals[0])
    mx = float(sorted_vals[-1])
    rng = mx - mn
    return {
        "count": n,
        "median": round(med, 6),
        "min": round(mn, 6),
        "max": round(mx, 6),
        "range": round(rng, 6),
    }


def score(
    results_path: Path | str,
    gold_path: Path | str,
) -> dict[str, Any]:
    """Phase 3 scoring evaluator.

    Validates finite JSON/schema, gold hash/pin against original protocol,
    complete expected schedule roster (refusing duplicate or unknown events),
    and strictly separates 28 primary historical controls from b05 correlated repeats.
    Operates strictly offline with zero network and zero live model calls.
    """
    res_path = Path(results_path)
    if not res_path.is_file():
        raise FileNotFoundError(f"Results file not found: {res_path}")

    g_path = Path(gold_path)
    if not g_path.is_file():
        raise FileNotFoundError(f"Gold file not found: {g_path}")

    # Read and validate finite JSON
    res_bytes = res_path.read_bytes()
    try:
        results_data = json.loads(
            res_bytes.decode("utf-8"),
            parse_constant=_reject_constant,
            parse_float=_parse_float,
        )
    except Exception as exc:
        raise ValueError(f"Malformed JSON in results file {res_path}: {exc}") from exc
    assert_finite(results_data)

    gold_bytes = g_path.read_bytes()
    try:
        gold_data = json.loads(
            gold_bytes.decode("utf-8"),
            parse_constant=_reject_constant,
            parse_float=_parse_float,
        )
    except Exception as exc:
        raise ValueError(f"Malformed JSON in gold file {g_path}: {exc}") from exc
    assert_finite(gold_data)

    # Validate schemas
    if results_data.get("schema_version") != RUN_RESULTS_SCHEMA_VERSION:
        raise ValueError(
            f"Unexpected results schema_version: {results_data.get('schema_version')!r}, "
            f"expected {RUN_RESULTS_SCHEMA_VERSION!r}"
        )

    if gold_data.get("schema_version") != GOLD_SCHEMA_VERSION:
        raise ValueError(
            f"Unexpected gold schema_version: {gold_data.get('schema_version')!r}, "
            f"expected {GOLD_SCHEMA_VERSION!r}"
        )

    # Validate gold labels
    gold_labels = gold_data.get("labels")
    if not isinstance(gold_labels, dict) or not gold_labels:
        raise ValueError("Gold data missing valid 'labels' dictionary")

    for cid, exp_label in gold_labels.items():
        if exp_label not in ALLOWED_VERDICT_LABELS:
            raise ValueError(
                f"Unknown gold expected label for case {cid!r}: {exp_label!r}. "
                f"Allowed labels: {sorted(ALLOWED_VERDICT_LABELS)}"
            )

    # Look for protocol.json to verify gold hash pin
    prot_candidates: list[Path] = [
        g_path.parent / "protocol.json",
    ]
    if results_data.get("protocol_path"):
        prot_candidates.append(Path(results_data["protocol_path"]))
    prot_candidates.append(res_path.parent / "protocol.json")
    prot_candidates.append(PROJECT_ROOT / "protocol.json")

    protocol_path = next((p for p in prot_candidates if p.is_file()), None)
    if protocol_path is None:
        raise FileNotFoundError(
            f"Protocol file not found to verify gold pin against protocol (checked: {prot_candidates})"
        )

    prot_bytes = protocol_path.read_bytes()
    try:
        protocol_data = json.loads(
            prot_bytes.decode("utf-8"),
            parse_constant=_reject_constant,
            parse_float=_parse_float,
        )
    except Exception as exc:
        raise ValueError(f"Malformed JSON in protocol file {protocol_path}: {exc}") from exc
    assert_finite(protocol_data)

    if protocol_data.get("schema_version") != PROTOCOL_SCHEMA_VERSION:
        raise ValueError(f"Unexpected protocol schema_version: {protocol_data.get('schema_version')!r}")

    # Validate protocol hash pin presence in results
    expected_prot_hash = results_data.get("protocol_canonical_sha256")
    if not is_valid_sha256(expected_prot_hash):
        raise ValueError(f"Missing or invalid protocol_canonical_sha256 pin in results: {expected_prot_hash!r}")

    # Validate gold pin presence in protocol
    expected_gold_hash = protocol_data.get("hashes", {}).get("gold_canonical_sha256")
    if not is_valid_sha256(expected_gold_hash):
        raise ValueError(f"Missing or invalid gold_canonical_sha256 pin in protocol: {expected_gold_hash!r}")

    # Validate defaults pin presence in protocol
    expected_defaults_hash = protocol_data.get("hashes", {}).get("defaults_canonical_sha256")
    if not is_valid_sha256(expected_defaults_hash):
        raise ValueError(f"Missing or invalid defaults_canonical_sha256 pin in protocol: {expected_defaults_hash!r}")

    # Validate protocol hash against results pin
    actual_prot_hash = canonical_sha256(protocol_data)
    if actual_prot_hash != expected_prot_hash:
        raise ValueError(
            f"Tampering detected: protocol.json canonical SHA-256 ({actual_prot_hash}) "
            f"does not match results pin ({expected_prot_hash})"
        )

    # Validate gold canonical hash against protocol pin
    actual_gold_hash = canonical_sha256(gold_data)
    if actual_gold_hash != expected_gold_hash:
        raise ValueError(
            f"Tampering detected: gold.json canonical SHA-256 ({actual_gold_hash}) "
            f"does not match protocol pin ({expected_gold_hash})"
        )

    # Load and validate frozen schedule against protocol
    sched_candidates = [
        protocol_path.parent / "schedule.json",
        g_path.parent / "schedule.json",
        res_path.parent / "schedule.json",
    ]
    schedule_file = next((p for p in sched_candidates if p.is_file()), None)
    if schedule_file is None:
        raise FileNotFoundError(
            f"Schedule file not found to verify schedule against protocol (checked: {sched_candidates})"
        )

    sched_bytes = schedule_file.read_bytes()
    try:
        schedule_data = json.loads(
            sched_bytes.decode("utf-8"),
            parse_constant=_reject_constant,
            parse_float=_parse_float,
        )
    except Exception as exc:
        raise ValueError(f"Malformed JSON in schedule file {schedule_file}: {exc}") from exc
    assert_finite(schedule_data)

    if schedule_data.get("schema_version") != SCHEDULE_SCHEMA_VERSION:
        raise ValueError(
            f"Unexpected schedule schema_version: {schedule_data.get('schema_version')!r}, "
            f"expected {SCHEDULE_SCHEMA_VERSION!r}"
        )

    actual_sched_hash = canonical_sha256(schedule_data)
    expected_sched_hash = protocol_data.get("hashes", {}).get("schedule_canonical_sha256")
    if not is_valid_sha256(expected_sched_hash):
        raise ValueError(f"Missing or invalid schedule_canonical_sha256 pin in protocol: {expected_sched_hash!r}")
    if actual_sched_hash != expected_sched_hash:
        raise ValueError(
            f"Tampering detected: schedule.json canonical SHA-256 ({actual_sched_hash}) "
            f"does not match protocol pin ({expected_sched_hash})"
        )

    schedule_items = schedule_data.get("items", [])
    TOTAL_PLANNED_RUNS = 124
    if not isinstance(schedule_items, list) or len(schedule_items) != TOTAL_PLANNED_RUNS:
        raise ValueError(f"Expected {TOTAL_PLANNED_RUNS} items in schedule.json, got {len(schedule_items)}")

    expected_by_sidx: dict[int, dict[str, Any]] = {}
    for it in schedule_items:
        s_idx = it.get("schedule_index")
        if not isinstance(s_idx, int):
            raise ValueError(f"Invalid schedule_index in schedule.json item: {s_idx!r}")
        expected_by_sidx[s_idx] = it

    # Validate attempts list
    attempts = results_data.get("attempts")
    if not isinstance(attempts, list):
        raise ValueError("Results data missing 'attempts' list")

    gold_cases_set = set(gold_labels.keys())
    seen_event_keys: set[tuple[str, str, int]] = set()
    seen_schedule_indices: set[int] = set()

    for idx, att in enumerate(attempts):
        if not isinstance(att, dict):
            raise ValueError(f"Attempt at index {idx} must be a dictionary")
        arm_id = att.get("arm_id")
        if arm_id not in ARMS:
            raise ValueError(f"Unknown arm_id {arm_id!r} in attempt {idx}")
        cid = att.get("case_id")
        if cid not in gold_cases_set:
            raise ValueError(f"Unknown case_id {cid!r} in attempt {idx}")
        rep_idx = att.get("repeat_index")
        if not isinstance(rep_idx, int) or rep_idx < 1:
            raise ValueError(f"Invalid repeat_index {rep_idx!r} in attempt {idx}")

        # Check impossible repeat index
        if cid != "b05" and rep_idx != 1:
            raise ValueError(f"Impossible repeat_index {rep_idx} for primary case {cid!r} (expected 1)")
        if cid == "b05" and rep_idx not in (1, 2, 3):
            raise ValueError(f"Impossible repeat_index {rep_idx} for b05 (expected 1, 2, or 3)")

        event_key = (arm_id, cid, rep_idx)
        if event_key in seen_event_keys:
            raise ValueError(f"Duplicate attempt event detected: {event_key}")
        seen_event_keys.add(event_key)

        s_idx = att.get("schedule_index")
        if s_idx is None or not isinstance(s_idx, int):
            raise ValueError(f"Missing or non-integer schedule_index in attempt {idx}: {s_idx!r}")
        if s_idx not in expected_by_sidx:
            raise ValueError(f"Unknown schedule_index {s_idx} in attempt {idx}")
        if s_idx in seen_schedule_indices:
            raise ValueError(f"Duplicate schedule_index detected: {s_idx}")
        seen_schedule_indices.add(s_idx)

        # Require schedule_index to map exactly to expected (arm_id, case_id, repeat_index)
        exp_it = expected_by_sidx[s_idx]
        if (arm_id, cid, rep_idx) != (exp_it["arm_id"], exp_it["case_id"], exp_it["repeat_index"]):
            raise ValueError(
                f"Reassigned schedule_index {s_idx}: attempt has {(arm_id, cid, rep_idx)}, "
                f"expected {(exp_it['arm_id'], exp_it['case_id'], exp_it['repeat_index'])}"
            )

    # Roster completeness check:
    attempted_runs_count = sum(1 for a in attempts if a.get("attempted") is True)
    skipped_runs_count = sum(1 for a in attempts if not a.get("attempted") or a.get("status") == "unattempted")
    missing_runs_count = max(0, TOTAL_PLANNED_RUNS - len(attempts))

    is_completed_evaluation = (
        results_data.get("completed_at") is not None
        and results_data.get("completion_status") in ("completed", "completed_with_errors")
        and len(attempts) == TOTAL_PLANNED_RUNS
        and skipped_runs_count == 0
        and attempted_runs_count == TOTAL_PLANNED_RUNS
        and len(seen_schedule_indices) == TOTAL_PLANNED_RUNS
    )

    if not is_completed_evaluation:
        incomplete_report = {
            "schema_version": SCORE_REPORT_SCHEMA_VERSION,
            "evaluation_status": "incomplete",
            "is_complete": False,
            "completion_status": results_data.get("completion_status", "incomplete/skipped"),
            "results_path": str(res_path),
            "gold_path": str(g_path),
            "protocol_path": str(protocol_path),
            "planned_runs": TOTAL_PLANNED_RUNS,
            "recorded_runs": len(attempts),
            "attempted_runs": attempted_runs_count,
            "skipped_runs": skipped_runs_count,
            "missing_runs": missing_runs_count,
            "message": (
                "Execution is incomplete or contains skipped/unattempted runs. Refusing to score partial execution as completed evaluation. "
                f"Planned {TOTAL_PLANNED_RUNS} runs, recorded {len(attempts)}, attempted {attempted_runs_count}, skipped {skipped_runs_count}, missing {missing_runs_count}."
            ),
        }
        assert_finite(incomplete_report)
        return incomplete_report

    # Completed run: process each arm
    arm_summaries: dict[str, Any] = {}
    for arm_id in ["A", "B", "C", "D"]:
        arm_attempts = [a for a in attempts if a["arm_id"] == arm_id]
        if len(arm_attempts) != 31:
            raise ValueError(f"Arm {arm_id} has {len(arm_attempts)} attempts, expected 31")

        primary_attempts = [a for a in arm_attempts if a["case_id"] != "b05"]
        b05_attempts = [a for a in arm_attempts if a["case_id"] == "b05"]

        if len(primary_attempts) != 28:
            raise ValueError(f"Arm {arm_id} has {len(primary_attempts)} primary attempts, expected 28")
        if len(b05_attempts) != 3:
            raise ValueError(f"Arm {arm_id} has {len(b05_attempts)} b05 repeat attempts, expected 3")

        planned_count = 28
        attempted_count = sum(1 for a in primary_attempts if a.get("attempted") is True)
        valid_count = sum(1 for a in primary_attempts if a.get("status") == "success")
        failed_count = sum(1 for a in primary_attempts if a.get("status") == "error")
        skipped_count = sum(1 for a in primary_attempts if a.get("status") == "unattempted")

        # 3-class confusion matrix
        classes = ["supported", "contradicted", "insufficient_evidence"]
        confusion_matrix = {exp: {pred: 0 for pred in classes} for exp in classes}
        correct_count = 0
        nonsupported_count = 0
        false_accept_count = 0
        supported_count = 0
        false_reject_count = 0

        for a in primary_attempts:
            cid = a["case_id"]
            exp_label = gold_labels[cid]
            if a.get("status") == "success":
                pred_label = extract_verdict_label(a.get("parsed_verdict"))
                confusion_matrix[exp_label][pred_label] += 1
                if pred_label == exp_label:
                    correct_count += 1
                if exp_label != "supported":
                    nonsupported_count += 1
                    if pred_label == "supported":
                        false_accept_count += 1
                if exp_label == "supported":
                    supported_count += 1
                    if pred_label != "supported":
                        false_reject_count += 1

        accuracy_planned = correct_count / planned_count
        accuracy_valid = (correct_count / valid_count) if valid_count > 0 else None
        technical_error_rate = failed_count / planned_count

        false_accepts = {
            "numerator": false_accept_count,
            "denominator": nonsupported_count,
            "rate": (false_accept_count / nonsupported_count) if nonsupported_count > 0 else None,
        }
        false_rejects = {
            "numerator": false_reject_count,
            "denominator": supported_count,
            "rate": (false_reject_count / supported_count) if supported_count > 0 else None,
        }

        # Technical errors list
        technical_errors = [
            {
                "case_id": a["case_id"],
                "error_type": a.get("error_type"),
                "error_message": a.get("error_message"),
            }
            for a in primary_attempts
            if a.get("status") == "error"
        ]

        # b05 diagnostic
        b05_repeats_info = []
        b05_verdicts = []
        b05_correct_count = 0
        b05_valid_count = 0

        for a in sorted(b05_attempts, key=lambda x: x.get("repeat_index", 0)):
            rep_idx = a.get("repeat_index")
            status = a.get("status")
            err_type = a.get("error_type")
            err_msg = a.get("error_message")
            pred_label = None
            is_correct = False
            if status == "success":
                pred_label = extract_verdict_label(a.get("parsed_verdict"))
                b05_verdicts.append(pred_label)
                b05_valid_count += 1
                if pred_label == "supported":
                    is_correct = True
                    b05_correct_count += 1

            b05_repeats_info.append({
                "repeat_index": rep_idx,
                "status": status,
                "error_type": err_type,
                "error_message": err_msg,
                "predicted_verdict": pred_label,
                "expected_verdict": "supported",
                "correct": is_correct,
            })

        b05_diagnostic = {
            "case_id": "b05",
            "expected_verdict": "supported",
            "total_repeats": len(b05_attempts),
            "valid_repeats": b05_valid_count,
            "correct_repeats": b05_correct_count,
            "all_repeats_agreed": (len(set(b05_verdicts)) == 1) if b05_verdicts else False,
            "repeats": b05_repeats_info,
            "disclaimer": (
                "b05 is a single posthoc known counterexample repeated 3 times. "
                "These repeats are correlated diagnostic measurements of the same instance, "
                "not independent test cases, and are strictly separated from primary controls."
            ),
        }

        # Latencies, Tokens, Thinking, Truncations
        durations_s = []
        load_durations_s = []
        in_tokens = []
        out_tokens = []
        thinking_lens = []
        truncation_count = 0

        for a in arm_attempts:
            raw = a.get("raw_response")
            stats = a.get("stats") or {}

            done_reason = raw.get("done_reason") if isinstance(raw, dict) else None
            err_msg = a.get("error_message") or ""
            if done_reason == "length" or "done_reason is 'length'" in err_msg or "truncat" in err_msg.lower():
                truncation_count += 1

            tot_dur = stats.get("total_duration")
            if isinstance(tot_dur, (int, float)) and not math.isnan(tot_dur) and not math.isinf(tot_dur) and tot_dur > 0:
                durations_s.append(tot_dur / 1e9)

            load_dur = stats.get("load_duration")
            if isinstance(load_dur, (int, float)) and not math.isnan(load_dur) and not math.isinf(load_dur) and load_dur > 0:
                load_durations_s.append(load_dur / 1e9)

            p_eval = stats.get("prompt_eval_count")
            if isinstance(p_eval, int) and p_eval >= 0:
                in_tokens.append(p_eval)

            eval_c = stats.get("eval_count")
            if isinstance(eval_c, int) and eval_c >= 0:
                out_tokens.append(eval_c)

            t_len = a.get("thinking_length")
            if isinstance(t_len, int) and t_len >= 0:
                thinking_lens.append(t_len)

        thinking_summary = compute_numeric_summary(thinking_lens)
        thinking_summary["attempts_with_thinking"] = sum(1 for tl in thinking_lens if tl > 0)

        placement_snapshot = results_data.get("arm_placements", {}).get(arm_id)

        arm_summaries[arm_id] = {
            "arm_id": arm_id,
            "model": ARMS[arm_id]["model"],
            "think": ARMS[arm_id]["think"],
            "options": ARMS[arm_id]["options"],
            "counts": {
                "planned_total": 31,
                "primary_planned": planned_count,
                "primary_attempted": attempted_count,
                "primary_valid": valid_count,
                "primary_failed": failed_count,
                "primary_skipped": skipped_count,
            },
            "accuracy": {
                "correct_count": correct_count,
                "correct_planned_ratio": f"{correct_count}/{planned_count}",
                "correct_valid_ratio": f"{correct_count}/{valid_count}" if valid_count > 0 else None,
                "accuracy_planned": accuracy_planned,
                "accuracy_valid": accuracy_valid,
                "technical_error_rate": technical_error_rate,
            },
            "confusion_matrix": confusion_matrix,
            "false_accepts_nonsupported": false_accepts,
            "false_rejects_supported": false_rejects,
            "technical_errors": technical_errors,
            "b05_diagnostic": b05_diagnostic,
            "performance_metrics": {
                "latency_seconds": compute_numeric_summary(durations_s),
                "load_duration_seconds": compute_numeric_summary(load_durations_s),
                "input_tokens": compute_numeric_summary(in_tokens),
                "output_tokens": compute_numeric_summary(out_tokens),
                "thinking_characters": thinking_summary,
                "truncation_count": truncation_count,
            },
            "placement_snapshot": placement_snapshot,
        }

    # Cross-arm comparisons
    arm_a = arm_summaries["A"]
    arm_b = arm_summaries["B"]
    arm_c = arm_summaries["C"]
    arm_d = arm_summaries["D"]

    def _calc_comparison(target_arm: dict[str, Any], base_arm: dict[str, Any]) -> dict[str, Any]:
        planned_diff = target_arm["accuracy"]["accuracy_planned"] - base_arm["accuracy"]["accuracy_planned"]
        valid_diff = None
        if target_arm["accuracy"]["accuracy_valid"] is not None and base_arm["accuracy"]["accuracy_valid"] is not None:
            valid_diff = round(target_arm["accuracy"]["accuracy_valid"] - base_arm["accuracy"]["accuracy_valid"], 6)
        return {
            "accuracy_planned_diff": round(planned_diff, 6),
            "accuracy_valid_diff": valid_diff,
        }

    comparisons = {
        "budget_effect_B_minus_A": {
            **_calc_comparison(arm_b, arm_a),
            "description": "Isolates output token budget expansion (1024 to 4096) on qwen3.5:9b.",
        },
        "thinking_effect_C_minus_B": {
            **_calc_comparison(arm_c, arm_b),
            "description": "Isolates reasoning/thinking activation (think=true) on qwen3.5:9b under 4096 token budget.",
        },
        "model_config_difference_D_minus_B": {
            **_calc_comparison(arm_d, arm_b),
            "description": "Reflects qwen3.8:27b scaled parameters + architecture differences + aligned sampling vs qwen3.5:9b.",
        },
    }

    disclosures = {
        "reused_controls": (
            "All 28 historical controls were previously used during pipeline development (20 dev + 8 reserved). "
            "No general reliability or production precision claims can be made from reused controls."
        ),
        "b05_status": (
            "b05 is a known posthoc development counterexample from the baseline stability audit. "
            "Its 3 repeats are correlated measurements of a known failure mode, not independent test cases."
        ),
        "execution_order_effects": (
            "Arm execution order is fixed (A -> B -> C -> D). Warm-up and KV-cache characteristics may vary."
        ),
        "vram_and_speed_disclaimer": (
            "Placement snapshots from /api/ps represent single instantaneous points in time, not peak VRAM. "
            "Wall-clock durations from local execution do not permit model-independent CPU vs GPU speed comparisons."
        ),
        "scope_limitation": (
            "This isolated verifier experiment tests only claim-evidence consistency classification; "
            "it does not measure retrieval recall, context truncation, or end-to-end RAG answer completeness."
        ),
    }

    score_report = {
        "schema_version": SCORE_REPORT_SCHEMA_VERSION,
        "evaluation_status": "completed",
        "is_complete": True,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "results_path": str(res_path),
        "gold_path": str(g_path),
        "protocol_path": str(protocol_path),
        "arms": arm_summaries,
        "comparisons": comparisons,
        "disclosures": disclosures,
    }
    assert_finite(score_report)
    return score_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verifier capacity comparison: Phase 1 preparation, Phase 2 execution, and Phase 3 scoring runner.",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # prepare subcommand
    prep_parser = subparsers.add_parser(
        "prepare",
        help="Prepare frozen protocol, inputs, gold and schedule.",
    )
    prep_parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Target directory where frozen artifacts will be written (must not exist or be empty).",
    )

    # run subcommand (Phase 2)
    run_parser = subparsers.add_parser(
        "run",
        help="Run model evaluations (Phase 2).",
    )
    run_parser.add_argument(
        "--protocol",
        type=Path,
        required=True,
        help="Path to protocol.json",
    )
    run_parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to output results file (must not exist)",
    )

    # score subcommand (Phase 3)
    score_parser = subparsers.add_parser(
        "score",
        help="Score evaluation results offline (Phase 3).",
    )
    score_parser.add_argument(
        "--results",
        type=Path,
        required=True,
        help="Path to results file",
    )
    score_parser.add_argument(
        "--gold",
        type=Path,
        required=True,
        help="Path to gold.json",
    )
    score_parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to output score report file (refuses existing)",
    )

    args = parser.parse_args(argv)
    if args.command == "prepare":
        res = prepare(args.output_dir)
        print(f"Preparation complete. Outputs written to {args.output_dir}")
        print(
            f"Total unique cases: {res['case_counts']['total_unique_cases']}, "
            f"Total scheduled runs: {res['case_counts']['total_scheduled_runs']}"
        )
        return 0
    elif args.command == "run":
        res = run(args.protocol, args.output)
        print(f"Run complete. Results written to {args.output}")
        print(
            f"Scheduled: {res['total_scheduled']}, Attempted: {res['total_attempted']}, "
            f"Successful: {res['total_successful']}, Errors: {res['total_technical_errors']}, "
            f"Unattempted: {res['total_unattempted']}"
        )
        return 0
    elif args.command == "score":
        out_path = Path(args.output)
        if out_path.exists():
            raise FileExistsError(
                f"Target score report file already exists: {out_path}. Refusing to overwrite."
            )
        report = score(args.results, args.gold)
        assert_finite(report)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        print(f"Scoring complete. Report written to {out_path}")
        if report.get("is_complete"):
            print("Evaluation status: completed")
            for arm_id in ["A", "B", "C", "D"]:
                arm_data = report["arms"][arm_id]
                acc = arm_data["accuracy"]
                print(
                    f"Arm {arm_id}: planned={acc['correct_planned_ratio']} ({acc['accuracy_planned']:.1%}), "
                    f"valid={acc['correct_valid_ratio']}, errors={acc['technical_error_rate']:.1%}"
                )
        else:
            print(f"Evaluation status: incomplete ({report.get('message')})")
        return 0
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
