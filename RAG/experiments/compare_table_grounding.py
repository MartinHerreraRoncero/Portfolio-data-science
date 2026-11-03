"""Phase 3 table-grounding comparison runner & protocol.

Executes a bounded, counterbalanced 31-condition evaluation matrix comparing
control (table grounding bypassed) vs candidate (table grounding enabled via
structured packets and numeric fact guard) under the mixed 9B generation + 27B verification arm.

Primary cohort (11 queries: b01..b08, n01..n03) runs candidate-only (delegates narrative unchanged).
Diagnostic cohorts (6 table queries t01..t06, 4 extra queries u01..u04) run as fresh paired
control/candidate conditions with alternating first arm.
Both arms share the exact same layout pages 95..99, index, and catalogs; only the opt-in flag changes.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Callable

from experiments.compare_mixed_single27 import (
    MixedSingle27Transport,
    validate_queries_payload,
)
from experiments.compare_layout_answers import (
    PINNED_9B_DIGEST,
    PINNED_27B_DIGEST,
    PINNED_EMBEDDING_DIGEST,
    SERVER_VERSION,
    checkpoint_atomic,
    compute_source_closure,
    _json_type_equal,
    _reject_constant,
    _parse_float,
    verify_environment,
)
from experiments.preserve_evidence import validate_catalog
from experiments.preserved_q3_verifier import (
    VERIFIER_Q3_DIGEST,
    VERIFIER_Q3_MODEL,
    VERIFIER_Q3_PARAMETERS,
)
from experiments.table_grounded import execute_table_attempt
from rag_pensiones.answer import DEFAULT_GENERATION_MODEL
from rag_pensiones.grounding import DEFAULT_PARAMETERS
from rag_pensiones.ingest import PROJECT_ROOT, validate_project_path
from rag_pensiones.qwen_embeddings import (
    DIMENSION,
    MODEL as QWEN_EMBEDDING_MODEL,
    validate_index as validate_qwen_index,
)

SCHEMA_VERSION = "table-grounding-effect-v1"
RESULTS_SCHEMA_VERSION = "table-grounding-results-v1"
REVIEW_SCHEMA_VERSION = "table-grounding-review-v1"
REVIEW_KEY_SCHEMA_VERSION = "table-grounding-review-key-v1"
HISTORICAL_PROTOCOL_SHA256 = "68a3f1feb9f7c9dda44e5b3f6d7c49c2239b2a7c327fd1c09188be9983704c98"

DEFAULT_HISTORICAL_PROTOCOL_PATH = Path("evaluation/layout_answer_effect_2026-10-27/prepared_v1/protocol.json")
DEFAULT_LAYOUT_PAGES_PATH = Path("evaluation/layout_answer_effect_2026-10-27/prepared_v1/layout_pages.jsonl")
DEFAULT_LAYOUT_INDEX_PATH = Path("evaluation/layout_answer_effect_2026-10-27/prepared_v1/layout_index.json")
DEFAULT_LAYOUT_PRIMARY_CATALOG_PATH = Path("evaluation/layout_answer_effect_2026-10-27/prepared_v1/layout_primary_catalog.json")
DEFAULT_LAYOUT_TABLE_CATALOG_PATH = Path("evaluation/layout_answer_effect_2026-10-27/prepared_v1/layout_table_catalog.json")
DEFAULT_STRUCTURED_BUNDLE_PATH = Path("evaluation/document_layout_2026-10-27/pilot_v4/structured.json")
DEFAULT_MANIFEST_PATH = Path("evaluation/document_layout_2026-10-27/pilot_v4/manifest.json")
DEFAULT_PRIMARY_QUERIES_PATH = Path("evaluation/scope_coverage_2026-10-26/full_inputs.json")
DEFAULT_TABLE_QUERIES_PATH = Path("evaluation/layout_answer_effect_2026-10-27/secondary_queries.json")
DEFAULT_EXTRA_QUERIES_PATH = Path("evaluation/table_grounding_2026-10-27/extra_queries.json")
DEFAULT_REFERENCE_FREEZE_PATH = Path("evaluation/table_grounding_2026-10-27/reference_freeze.json")
DEFAULT_OLLAMA_DEFAULTS_REL = "evaluation/verifier_capacity_2026-10-26/ollama_defaults_v0.40.11.json"

REQUIRED_INPUT_PATHS = (
    DEFAULT_LAYOUT_PAGES_PATH.as_posix(),
    DEFAULT_LAYOUT_INDEX_PATH.as_posix(),
    DEFAULT_LAYOUT_PRIMARY_CATALOG_PATH.as_posix(),
    DEFAULT_LAYOUT_TABLE_CATALOG_PATH.as_posix(),
    DEFAULT_STRUCTURED_BUNDLE_PATH.as_posix(),
    DEFAULT_MANIFEST_PATH.as_posix(),
    DEFAULT_PRIMARY_QUERIES_PATH.as_posix(),
    DEFAULT_TABLE_QUERIES_PATH.as_posix(),
    DEFAULT_EXTRA_QUERIES_PATH.as_posix(),
    DEFAULT_REFERENCE_FREEZE_PATH.as_posix(),
    DEFAULT_OLLAMA_DEFAULTS_REL,
)

CLOSED_ROLE_FILES = {
    "pages_file": "layout_pages.jsonl",
    "index_file": "layout_index.json",
    "primary_catalog_file": "layout_primary_catalog.json",
    "table_catalog_file": "layout_table_catalog.json",
    "table_bundle_file": "structured.json",
}

ASSET_FILES = (
    "layout_pages.jsonl",
    "layout_index.json",
    "layout_primary_catalog.json",
    "layout_table_catalog.json",
    "structured.json",
    "primary_queries.json",
    "table_queries.json",
    "extra_queries.json",
)


def _is_hex64(val: Any) -> bool:
    """Validate that val is a 64-character hexadecimal string."""
    return isinstance(val, str) and len(val) == 64 and all(c in "0123456789abcdefABCDEF" for c in val)


def compute_canonical_sha256(content: bytes, is_jsonl: bool = False) -> str:
    """Compute type-aware canonical SHA256 of JSON/JSONL content."""
    if is_jsonl:
        lines = [
            json.loads(line, parse_constant=_reject_constant, parse_float=_parse_float)
            for line in content.decode("utf-8").splitlines()
            if line.strip()
        ]
        canon_bytes = "\n".join(
            json.dumps(l, sort_keys=True, ensure_ascii=False, allow_nan=False) for l in lines
        ).encode("utf-8")
    else:
        obj = json.loads(content.decode("utf-8"), parse_constant=_reject_constant, parse_float=_parse_float)
        canon_bytes = json.dumps(obj, sort_keys=True, ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(canon_bytes).hexdigest()


def _validate_query_cases(payload: dict[str, Any], expected_ids: list[str], cohort_name: str) -> list[dict[str, str]]:
    """Validate query cases dictionary strictly, forbidding any answer or gold labels."""
    if not isinstance(payload, dict):
        raise ValueError(f"{cohort_name} queries must be a dict, got {type(payload).__name__}")
    cases = payload.get("cases")
    if not isinstance(cases, list) or len(cases) != len(expected_ids):
        raise ValueError(
            f"{cohort_name} queries must contain exactly {len(expected_ids)} cases, "
            f"got {len(cases) if isinstance(cases, list) else type(cases)}"
        )

    validated: list[dict[str, str]] = []
    for idx, c in enumerate(cases):
        if not isinstance(c, dict):
            raise ValueError(f"Invalid case at index {idx} in {cohort_name} queries")
        keys = set(c.keys())
        if keys != {"id", "query"}:
            extra = keys - {"id", "query"}
            if any(any(token in k.lower() for token in ("gold", "answer", "label", "reference")) for k in extra):
                raise ValueError(f"Forbidden gold label keys in {cohort_name} query {c.get('id')!r}: {sorted(extra)}")
            raise ValueError(f"{cohort_name} case keys must be exactly {{'id', 'query'}}, got {sorted(keys)}")
        cid = c.get("id")
        if cid != expected_ids[idx]:
            raise ValueError(f"{cohort_name} case at index {idx} expected id {expected_ids[idx]!r}, got {cid!r}")
        q = c.get("query")
        if not isinstance(q, str) or not q.strip():
            raise ValueError(f"Query for {cohort_name} case {cid} must be a non-empty string")
        validated.append({"id": cid, "query": q})
    return validated


def build_schedule(
    primary: dict[str, Any],
    table: dict[str, Any],
    extra: dict[str, Any],
) -> list[dict[str, Any]]:
    """Build deterministic 31-condition evaluation schedule.

    - Primary cohort: 11 benchmark queries (b01..b08, n01..n03), CANDIDATE ONLY (table grounding opt-in).
    - Table diagnostic cohort: 6 queries (t01..t06), fresh CONTROL/CANDIDATE pair with alternate first arm.
    - Extra diagnostic cohort: 4 queries (u01..u04), fresh CONTROL/CANDIDATE pair with alternate first arm.

    Pairs alternate first arm across the 10 paired questions:
    Pair 0 (t01): control, candidate
    Pair 1 (t02): candidate, control
    Pair 2 (t03): control, candidate
    ...
    Pair 9 (u04): candidate, control
    """
    primary_cases = validate_queries_payload(primary)
    table_expected_ids = ["t01", "t02", "t03", "t04", "t05", "t06"]
    table_cases = _validate_query_cases(table, table_expected_ids, "table")
    extra_expected_ids = ["u01", "u02", "u03", "u04"]
    extra_cases = _validate_query_cases(extra, extra_expected_ids, "extra")

    schedule: list[dict[str, Any]] = []
    attempt_idx = 1
    seen_keys: set[tuple[str, str, str]] = set()

    # 1. Primary cohort: 11 questions, candidate only
    for q_idx, c in enumerate(primary_cases):
        cid = c["id"]
        cond = "candidate"
        key = ("primary", cid, cond)
        if key in seen_keys:
            raise ValueError(f"Duplicate attempt key detected: {key}")
        seen_keys.add(key)

        schedule.append({
            "attempt_index": attempt_idx,
            "question_index": q_idx,
            "cohort": "primary",
            "case_id": cid,
            "query": c["query"],
            "condition": cond,
            "arm": "mixed",
            "table_enabled": True,
        })
        attempt_idx += 1

    # 2. Paired cohorts (6 table + 4 extra = 10 paired questions)
    paired_items: list[tuple[str, str, str]] = []
    for c in table_cases:
        paired_items.append(("table", c["id"], c["query"]))
    for c in extra_cases:
        paired_items.append(("extra", c["id"], c["query"]))

    for pair_idx, (cohort, cid, query) in enumerate(paired_items):
        q_idx = 11 + pair_idx
        if pair_idx % 2 == 0:
            conditions = ["control", "candidate"]
        else:
            conditions = ["candidate", "control"]

        for cond in conditions:
            key = (cohort, cid, cond)
            if key in seen_keys:
                raise ValueError(f"Duplicate attempt key detected: {key}")
            seen_keys.add(key)

            schedule.append({
                "attempt_index": attempt_idx,
                "question_index": q_idx,
                "cohort": cohort,
                "case_id": cid,
                "query": query,
                "condition": cond,
                "arm": "mixed",
                "table_enabled": (cond == "candidate"),
            })
            attempt_idx += 1

    if len(schedule) != 31:
        raise ValueError(f"Expected exactly 31 attempts in schedule, got {len(schedule)}")
    return schedule


def _validate_attempt_result(
    raw_attempt: Any,
    scheduled: dict[str, Any],
) -> tuple[bool, str | None]:
    """Validates that execute_fn result matches minimum identity, status, flag, and audit shapes."""
    if not isinstance(raw_attempt, dict):
        return False, f"Attempt result must be a dict, got {type(raw_attempt).__name__}"

    if raw_attempt.get("attempt_index") != scheduled["attempt_index"]:
        return False, f"Attempt index mismatch: expected {scheduled['attempt_index']}, got {raw_attempt.get('attempt_index')}"
    if raw_attempt.get("case_id") != scheduled["case_id"]:
        return False, f"Case ID mismatch: expected {scheduled['case_id']!r}, got {raw_attempt.get('case_id')!r}"
    if raw_attempt.get("query") != scheduled["query"]:
        return False, "Query text mismatch against schedule"
    if raw_attempt.get("arm") != "mixed":
        return False, f"Arm mismatch: expected 'mixed', got {raw_attempt.get('arm')!r}"

    status = raw_attempt.get("status")
    if status not in ("completed", "error"):
        return False, f"Invalid attempt status {status!r}; legacy raw status must be 'completed' or 'error'"

    table_audit = raw_attempt.get("table_audit")
    if not isinstance(table_audit, dict):
        return False, f"Attempt missing table_audit dict, got {type(table_audit).__name__}"

    expected_enabled = scheduled["table_enabled"]
    if type(table_audit.get("enabled")) is not bool or table_audit.get("enabled") != expected_enabled:
        return False, f"table_audit enabled mismatch or non-bool: expected {expected_enabled}, got {table_audit.get('enabled')!r}"

    is_table_query = table_audit.get("is_table_query")
    if type(is_table_query) is not bool:
        return False, f"table_audit is_table_query must be bool, got {type(is_table_query).__name__}"

    gate_decision = table_audit.get("gate_decision")
    if not isinstance(gate_decision, str) or not gate_decision.strip():
        return False, f"table_audit gate_decision must be non-empty str, got {gate_decision!r}"

    # Semantic routing validation:
    # When table grounding is disabled (control condition), adapter ALWAYS emits
    # table_audit: {enabled: False, is_table_query: False, gate_decision: "bypassed"}
    if not expected_enabled:
        if is_table_query is not False:
            return False, f"Disabled control table_audit must have is_table_query=False, got {is_table_query!r}"
        if status != "error" and gate_decision != "bypassed":
            return False, f"Disabled control table_audit must have gate_decision='bypassed', got {gate_decision!r}"
    else:
        # Candidate conditions (enabled=True):
        is_candidate_table_cohort = (scheduled["cohort"] in ("table", "extra"))
        if is_candidate_table_cohort:
            # Candidate explicit table query
            # If attempt errored before factory/grounder ran, table_audit may be bypassed or technical_error
            if status != "error" and is_table_query is not True:
                return False, f"Candidate table query must have is_table_query=True, got {is_table_query!r}"
        else:
            # Candidate narrative query (primary cohort): delegates unchanged
            if is_table_query is not False:
                return False, f"Candidate narrative query must have is_table_query=False, got {is_table_query!r}"
            if status != "error" and gate_decision != "bypassed":
                return False, f"Candidate narrative query must have gate_decision='bypassed', got {gate_decision!r}"

    if status == "error":
        err_msg = raw_attempt.get("error")
        if not isinstance(err_msg, str) or not err_msg.strip():
            return False, "Error attempt missing non-empty error message"
        return True, err_msg

    # Shape checks for completed attempts
    formatted_ans = raw_attempt.get("formatted_answer")
    if not isinstance(formatted_ans, str) or not formatted_ans.strip():
        return False, "Completed attempt missing non-empty formatted_answer string"

    factual_outcome = raw_attempt.get("factual_outcome")
    if not isinstance(factual_outcome, dict):
        return False, f"Completed attempt factual_outcome must be a dict, got {type(factual_outcome).__name__}"

    fo_status = factual_outcome.get("status")
    if fo_status not in ("answered", "partial", "insufficient_evidence"):
        return False, f"Completed attempt factual_outcome status must be 'answered', 'partial', or 'insufficient_evidence', got {fo_status!r}"

    facts = factual_outcome.get("facts")
    if facts is not None:
        if not isinstance(facts, list) or any(not isinstance(f, dict) for f in facts):
            return False, "Facts in factual_outcome must be a list of dicts"

    claims = factual_outcome.get("claims")
    if claims is not None:
        if not isinstance(claims, list) or any(not isinstance(c, dict) for c in claims):
            return False, "Claims in factual_outcome must be a list of dicts"

    if not isinstance(raw_attempt.get("coverage_audit"), dict):
        return False, f"Completed attempt missing coverage_audit dict, got {type(raw_attempt.get('coverage_audit')).__name__}"

    coverage_pub = raw_attempt.get("coverage_publication")
    if not isinstance(coverage_pub, dict):
        return False, f"Completed attempt missing coverage_publication dict, got {type(coverage_pub).__name__}"

    cov_status = coverage_pub.get("status")
    if not isinstance(cov_status, str) or not cov_status.strip():
        return False, "coverage_publication missing valid status string"

    return True, None


def validate_protocol(protocol: dict[str, Any], *, protocol_dir: Path | None = None) -> None:
    """Strictly validate protocol schema, parameters, 31-attempt schedule, source integrity, and asset bindings."""
    if not isinstance(protocol, dict):
        raise ValueError(f"Protocol must be a dict, got {type(protocol).__name__}")

    # 1. Schema version & server version
    if protocol.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"Protocol schema mismatch: expected {SCHEMA_VERSION!r}, got {protocol.get('schema_version')!r}"
        )
    if protocol.get("server_version") != SERVER_VERSION:
        raise ValueError(
            f"Server version mismatch: expected {SERVER_VERSION!r}, got {protocol.get('server_version')!r}"
        )

    # 2. Pinned models and digests
    models = protocol.get("models")
    if not isinstance(models, dict):
        raise ValueError("Protocol missing 'models' dict")
    expected_models = {
        "generation_model": DEFAULT_GENERATION_MODEL,
        "generation_digest": PINNED_9B_DIGEST,
        "verifier_model": VERIFIER_Q3_MODEL,
        "verifier_digest": PINNED_27B_DIGEST,
        "embedding_model": QWEN_EMBEDDING_MODEL,
        "embedding_digest": PINNED_EMBEDDING_DIGEST,
    }
    if set(models.keys()) != set(expected_models.keys()):
        raise ValueError(f"Protocol models keys mismatch: {sorted(models.keys())} != {sorted(expected_models.keys())}")
    for k, expected_v in expected_models.items():
        if models[k] != expected_v:
            raise ValueError(f"Protocol model {k} mismatch: expected {expected_v!r}, got {models[k]!r}")

    # 3. Parameters (type-aware check)
    params = protocol.get("parameters")
    if not isinstance(params, dict) or set(params.keys()) != {"generation_parameters", "verifier_parameters"}:
        raise ValueError("Protocol parameters keys mismatch: expected {'generation_parameters', 'verifier_parameters'}")

    gen_params = params["generation_parameters"]
    if not isinstance(gen_params, dict) or set(gen_params.keys()) != set(DEFAULT_PARAMETERS.keys()):
        raise ValueError("Generation parameters keys mismatch against DEFAULT_PARAMETERS")
    for k, exp_val in DEFAULT_PARAMETERS.items():
        val = gen_params[k]
        if type(val) is not type(exp_val):
            raise ValueError(f"Generation parameter {k!r} type mismatch: expected {type(exp_val).__name__}, got {type(val).__name__}")
        if val != exp_val:
            raise ValueError(f"Generation parameter {k!r} value mismatch: expected {exp_val!r}, got {val!r}")

    ver_params = params["verifier_parameters"]
    if not isinstance(ver_params, dict) or set(ver_params.keys()) != set(VERIFIER_Q3_PARAMETERS.keys()):
        raise ValueError("Verifier parameters keys mismatch against VERIFIER_Q3_PARAMETERS")
    for k, exp_val in VERIFIER_Q3_PARAMETERS.items():
        val = ver_params[k]
        if type(val) is not type(exp_val):
            raise ValueError(f"Verifier parameter {k!r} type mismatch: expected {type(exp_val).__name__}, got {type(val).__name__}")
        if val != exp_val:
            raise ValueError(f"Verifier parameter {k!r} value mismatch: expected {exp_val!r}, got {val!r}")

    # 4. Execution options (type-aware check, strict bool)
    exec_options = protocol.get("execution_options")
    expected_options = {
        "top_k": 4,
        "max_packages": 1,
        "closed_note_tails": False,
        "partial_generation": True,
    }
    if not isinstance(exec_options, dict) or set(exec_options.keys()) != set(expected_options.keys()):
        raise ValueError("Execution options keys mismatch")
    for k, exp_val in expected_options.items():
        val = exec_options[k]
        if type(val) is not type(exp_val):
            raise ValueError(f"Execution option {k!r} type mismatch: expected {type(exp_val).__name__}, got {type(val).__name__}")
        if val != exp_val:
            raise ValueError(f"Execution option {k!r} value mismatch: expected {exp_val!r}, got {val!r}")

    # 5. Closed role configuration for both arms
    both_arms = protocol.get("both_arms")
    if not isinstance(both_arms, dict) or set(both_arms.keys()) != {"control", "candidate"}:
        raise ValueError("Protocol both_arms must contain exactly {'control', 'candidate'}")

    for arm_name in ("control", "candidate"):
        arm_data = both_arms[arm_name]
        if not isinstance(arm_data, dict):
            raise ValueError(f"both_arms[{arm_name!r}] must be a dict")
        for role_field, expected_fname in CLOSED_ROLE_FILES.items():
            if arm_data.get(role_field) != expected_fname:
                raise ValueError(
                    f"both_arms[{arm_name!r}][{role_field!r}] mismatch: "
                    f"expected canonical filename {expected_fname!r}, got {arm_data.get(role_field)!r}"
                )
        expected_enabled = (arm_name == "candidate")
        if type(arm_data.get("table_enabled")) is not bool or arm_data.get("table_enabled") != expected_enabled:
            raise ValueError(f"both_arms[{arm_name!r}] table_enabled mismatch or non-bool: expected {expected_enabled}")
        if arm_data.get("pages_count") != 5:
            raise ValueError(f"both_arms[{arm_name!r}] pages_count must be 5, got {arm_data.get('pages_count')}")
        if arm_data.get("chunks_count") != 13:
            raise ValueError(f"both_arms[{arm_name!r}] chunks_count must be 13, got {arm_data.get('chunks_count')}")
        if arm_data.get("primary_units_count") != 13:
            raise ValueError(f"both_arms[{arm_name!r}] primary_units_count must be 13, got {arm_data.get('primary_units_count')}")
        if arm_data.get("table_units_count") != 14:
            raise ValueError(f"both_arms[{arm_name!r}] table_units_count must be 14, got {arm_data.get('table_units_count')}")

    # 6. Schedule check (31 attempts)
    schedule = protocol.get("schedule")
    if not isinstance(schedule, list) or len(schedule) != 31:
        raise ValueError(f"Protocol schedule must contain exactly 31 attempts, got {len(schedule) if isinstance(schedule, list) else type(schedule)}")

    seen_attempts: set[tuple[str, str, str]] = set()
    for idx, att in enumerate(schedule):
        if not isinstance(att, dict):
            raise ValueError(f"Attempt at index {idx} must be a dict")
        if att.get("attempt_index") != idx + 1:
            raise ValueError(f"Attempt at index {idx} has wrong attempt_index: expected {idx+1}, got {att.get('attempt_index')}")
        cohort = att.get("cohort")
        if cohort not in ("primary", "table", "extra"):
            raise ValueError(f"Attempt at index {idx} has invalid cohort: {cohort!r}")
        cond = att.get("condition")
        if cond not in ("control", "candidate"):
            raise ValueError(f"Attempt at index {idx} has invalid condition: {cond!r}")
        if att.get("arm") != "mixed":
            raise ValueError(f"Attempt at index {idx} has invalid arm: {att.get('arm')!r} (must be 'mixed')")
        expected_enabled = (cond == "candidate")
        if type(att.get("table_enabled")) is not bool or att.get("table_enabled") != expected_enabled:
            raise ValueError(f"Attempt at index {idx} table_enabled mismatch or non-bool")
        cid = att.get("case_id")
        if not isinstance(cid, str) or not cid.strip():
            raise ValueError(f"Attempt at index {idx} missing case_id")
        q = att.get("query")
        if not isinstance(q, str) or not q.strip():
            raise ValueError(f"Attempt at index {idx} missing query")

        key = (cohort, cid, cond)
        if key in seen_attempts:
            raise ValueError(f"Duplicate attempt key in protocol schedule: {key}")
        seen_attempts.add(key)

    # 7. Source freeze closure check
    frozen_sources = protocol.get("source_freeze")
    if not isinstance(frozen_sources, dict) or not frozen_sources:
        raise ValueError("Protocol missing or empty 'source_freeze'")
    current_closure = compute_source_closure()
    if set(frozen_sources.keys()) != set(current_closure.keys()):
        missing = set(current_closure.keys()) - set(frozen_sources.keys())
        extra = set(frozen_sources.keys()) - set(current_closure.keys())
        raise ValueError(f"Source closure keys mismatch: missing {sorted(missing)}, extra {sorted(extra)}")
    for rel_path, exp_sha in frozen_sources.items():
        curr_sha = current_closure[rel_path]
        if curr_sha != exp_sha:
            raise ValueError(f"Source code drift detected for {rel_path}: expected {exp_sha}, got {curr_sha}")

    # 8. Manifest and reference freeze hashes (using raw bytes)
    p_manifest = PROJECT_ROOT / DEFAULT_MANIFEST_PATH
    if not p_manifest.is_file():
        raise FileNotFoundError(f"Source artifact manifest not found: {p_manifest}")
    manifest_bytes = p_manifest.read_bytes()
    curr_manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    if protocol.get("source_artifact_manifest_sha256") != curr_manifest_sha:
        raise ValueError(
            f"Source artifact manifest SHA mismatch: expected {protocol.get('source_artifact_manifest_sha256')}, got {curr_manifest_sha}"
        )
    manifest_data = json.loads(manifest_bytes.decode("utf-8"), parse_constant=_reject_constant, parse_float=_parse_float)

    p_ref_freeze = PROJECT_ROOT / DEFAULT_REFERENCE_FREEZE_PATH
    if not p_ref_freeze.is_file():
        raise FileNotFoundError(f"Reference freeze file not found: {p_ref_freeze}")
    curr_ref_freeze_sha = hashlib.sha256(p_ref_freeze.read_bytes()).hexdigest()
    if protocol.get("reference_freeze_sha256") != curr_ref_freeze_sha:
        raise ValueError(
            f"Reference freeze SHA mismatch: expected {protocol.get('reference_freeze_sha256')}, got {curr_ref_freeze_sha}"
        )

    # Verify hashes of reference freeze files without reading their contents
    ref_data = json.loads(p_ref_freeze.read_bytes().decode("utf-8"), parse_constant=_reject_constant, parse_float=_parse_float)
    ref_files = ref_data.get("files", {})
    if not isinstance(ref_files, dict):
        raise ValueError("reference_freeze.json missing 'files' dict")
    for rpath, exp_rhash in ref_files.items():
        rfp = PROJECT_ROOT / rpath
        if not rfp.is_file():
            raise FileNotFoundError(f"Reference freeze target file not found: {rfp}")
        actual_rhash = hashlib.sha256(rfp.read_bytes()).hexdigest()
        if actual_rhash != exp_rhash:
            raise ValueError(f"Reference file drift for {rpath}: expected {exp_rhash}, got {actual_rhash}")

    # 9. Required input file hashes
    input_hashes = protocol.get("input_file_hashes")
    if not isinstance(input_hashes, dict):
        raise ValueError("Protocol missing 'input_file_hashes' dict")

    expected_input_keys = set(Path(p).as_posix() for p in REQUIRED_INPUT_PATHS)
    raw_keys = list(input_hashes.keys())
    if len(raw_keys) != len(REQUIRED_INPUT_PATHS):
        missing = expected_input_keys - set(raw_keys)
        extra = set(raw_keys) - expected_input_keys
        raise ValueError(
            f"input_file_hashes count mismatch: expected {len(REQUIRED_INPUT_PATHS)}, got {len(raw_keys)} "
            f"(missing: {sorted(missing)}, extra: {sorted(extra)})"
        )

    seen_canonical_inputs: set[str] = set()
    for k in raw_keys:
        pk = Path(k)
        if pk.is_absolute():
            try:
                rel_k = pk.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
            except ValueError:
                rel_k = pk.as_posix()
        else:
            rel_k = (PROJECT_ROOT / pk).resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
        if rel_k in seen_canonical_inputs:
            raise ValueError(f"Alias duplicate in input_file_hashes: {k!r} resolves to {rel_k!r}")
        seen_canonical_inputs.add(rel_k)

    if seen_canonical_inputs != expected_input_keys:
        raise ValueError(f"input_file_hashes keyset mismatch: missing {sorted(expected_input_keys - seen_canonical_inputs)}")

    for file_path_str, exp_hash in input_hashes.items():
        fp = PROJECT_ROOT / file_path_str if not Path(file_path_str).is_absolute() else Path(file_path_str)
        if not fp.is_file():
            raise FileNotFoundError(f"Input file not found: {fp}")
        actual_hash = hashlib.sha256(fp.read_bytes()).hexdigest()
        if actual_hash != exp_hash:
            raise ValueError(f"Input file drift for {file_path_str}: expected {exp_hash}, got {actual_hash}")

    # 10. Generated assets validation: recompute and validate both raw and canonical format
    gen_assets = protocol.get("generated_assets")
    if not isinstance(gen_assets, dict):
        raise ValueError("Protocol missing 'generated_assets' dict")
    for af in ASSET_FILES:
        if af not in gen_assets:
            raise ValueError(f"Generated assets missing {af!r}")
        entry = gen_assets[af]
        if not isinstance(entry, dict) or "raw_sha256" not in entry or "canonical_sha256" not in entry:
            raise ValueError(f"Generated asset entry for {af!r} missing raw or canonical sha256")
        if not _is_hex64(entry["raw_sha256"]):
            raise ValueError(f"Generated asset entry for {af!r} has invalid raw_sha256 format: {entry['raw_sha256']!r}")
        if not _is_hex64(entry["canonical_sha256"]):
            raise ValueError(f"Generated asset entry for {af!r} has invalid canonical_sha256 format: {entry['canonical_sha256']!r}")

    # 11. Historical protocol and authoritative bundle bindings
    p_hist = PROJECT_ROOT / DEFAULT_HISTORICAL_PROTOCOL_PATH
    if not p_hist.is_file():
        raise FileNotFoundError(f"Historical protocol not found: {p_hist}")
    hist_raw_bytes = p_hist.read_bytes()
    hist_raw_sha = hashlib.sha256(hist_raw_bytes).hexdigest()
    if hist_raw_sha != HISTORICAL_PROTOCOL_SHA256:
        raise ValueError(f"Historical protocol SHA mismatch: expected {HISTORICAL_PROTOCOL_SHA256}, got {hist_raw_sha}")
    if protocol.get("historical_protocol_sha256") != HISTORICAL_PROTOCOL_SHA256:
        raise ValueError(f"Protocol historical_protocol_sha256 mismatch: expected {HISTORICAL_PROTOCOL_SHA256}, got {protocol.get('historical_protocol_sha256')}")

    hist_proto = json.loads(hist_raw_bytes.decode("utf-8"), parse_constant=_reject_constant, parse_float=_parse_float)

    # Validate historical generated assets binding
    for af in ("layout_pages.jsonl", "layout_index.json", "layout_primary_catalog.json", "layout_table_catalog.json"):
        hist_entry = hist_proto.get("generated_assets", {}).get(af)
        if not hist_entry:
            raise ValueError(f"Historical protocol missing asset {af!r}")
        if gen_assets[af]["canonical_sha256"] != hist_entry["canonical_sha256"]:
            raise ValueError(
                f"Asset {af!r} canonical_sha256 does not match historical asset: "
                f"expected {hist_entry['canonical_sha256']}, got {gen_assets[af]['canonical_sha256']}"
            )

    # Validate authoritative structured bundle binding
    p_struct_auth = PROJECT_ROOT / DEFAULT_STRUCTURED_BUNDLE_PATH
    if not p_struct_auth.is_file():
        raise FileNotFoundError(f"Authoritative structured bundle not found: {p_struct_auth}")
    auth_struct_bytes = p_struct_auth.read_bytes()
    auth_struct_raw = hashlib.sha256(auth_struct_bytes).hexdigest()
    manifest_struct_raw = manifest_data.get("files", {}).get("structured.json", {}).get("sha256")
    if auth_struct_raw != manifest_struct_raw:
        raise ValueError(f"Authoritative structured.json raw SHA mismatch against manifest: expected {manifest_struct_raw}, got {auth_struct_raw}")
    auth_struct_canon = compute_canonical_sha256(auth_struct_bytes, is_jsonl=False)
    if gen_assets["structured.json"]["canonical_sha256"] != auth_struct_canon:
        raise ValueError(
            f"structured.json canonical_sha256 does not match authoritative bundle: "
            f"expected {auth_struct_canon}, got {gen_assets['structured.json']['canonical_sha256']}"
        )

    # Validate authoritative original query files binding
    query_bindings = (
        ("primary_queries.json", DEFAULT_PRIMARY_QUERIES_PATH, "primary"),
        ("table_queries.json", DEFAULT_TABLE_QUERIES_PATH, "table"),
        ("extra_queries.json", DEFAULT_EXTRA_QUERIES_PATH, "extra"),
    )
    auth_query_payloads: dict[str, dict[str, Any]] = {}
    auth_query_canons: dict[str, str] = {}

    for af, rel_path, cohort_name in query_bindings:
        p_q_auth = PROJECT_ROOT / rel_path
        if not p_q_auth.is_file():
            raise FileNotFoundError(f"Authoritative {cohort_name} query file not found: {p_q_auth}")
        q_bytes = p_q_auth.read_bytes()
        q_canon = compute_canonical_sha256(q_bytes, is_jsonl=False)
        auth_query_canons[af] = q_canon
        if gen_assets[af]["canonical_sha256"] != q_canon:
            raise ValueError(
                f"Asset {af!r} canonical_sha256 does not match authoritative original {rel_path.as_posix()}: "
                f"expected {q_canon}, got {gen_assets[af]['canonical_sha256']}"
            )
        auth_query_payloads[af] = json.loads(q_bytes.decode("utf-8"), parse_constant=_reject_constant, parse_float=_parse_float)

    # Validate schedule against authoritative original queries
    auth_schedule = build_schedule(
        auth_query_payloads["primary_queries.json"],
        auth_query_payloads["table_queries.json"],
        auth_query_payloads["extra_queries.json"],
    )
    if schedule != auth_schedule:
        raise ValueError("Protocol schedule does not match schedule built from authoritative original queries")

    # 12. If protocol_dir is provided, recompute and verify BOTH raw and canonical SHA for every asset on disk
    if protocol_dir is not None:
        pdir = Path(protocol_dir)
        for af in ASSET_FILES:
            af_path = pdir / af
            if not af_path.is_file():
                raise FileNotFoundError(f"Generated asset file not found in protocol dir: {af_path}")
            af_bytes = af_path.read_bytes()
            raw_sha = hashlib.sha256(af_bytes).hexdigest()
            if raw_sha != gen_assets[af]["raw_sha256"]:
                raise ValueError(f"Asset raw sha mismatch for {af}: expected {gen_assets[af]['raw_sha256']}, got {raw_sha}")
            canon_sha = compute_canonical_sha256(af_bytes, is_jsonl=af.endswith(".jsonl"))
            if canon_sha != gen_assets[af]["canonical_sha256"]:
                raise ValueError(f"Asset canonical sha mismatch for {af}: expected {gen_assets[af]['canonical_sha256']}, got {canon_sha}")

            # Recompute and bind against historical / authoritative content
            if af in ("layout_pages.jsonl", "layout_index.json", "layout_primary_catalog.json", "layout_table_catalog.json"):
                hist_canon = hist_proto["generated_assets"][af]["canonical_sha256"]
                if canon_sha != hist_canon:
                    raise ValueError(f"Asset {af!r} disk canonical SHA drifted from historical binding: expected {hist_canon}, got {canon_sha}")
            elif af == "structured.json":
                if canon_sha != auth_struct_canon:
                    raise ValueError(f"structured.json disk canonical SHA drifted from authoritative bundle: expected {auth_struct_canon}, got {canon_sha}")
            elif af in auth_query_canons:
                exp_q_canon = auth_query_canons[af]
                if canon_sha != exp_q_canon:
                    raise ValueError(f"Asset {af!r} disk canonical SHA drifted from authoritative original: expected {exp_q_canon}, got {canon_sha}")
                copy_payload = json.loads(af_bytes.decode("utf-8"), parse_constant=_reject_constant, parse_float=_parse_float)
                if not _json_type_equal(copy_payload, auth_query_payloads[af]):
                    raise ValueError(f"Asset {af!r} disk content drifted in type/payload from authoritative original")

        # Validate pages
        pages_data = [
            json.loads(line, parse_constant=_reject_constant, parse_float=_parse_float)
            for line in (pdir / "layout_pages.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(pages_data) != 5:
            raise ValueError(f"Expected 5 pages in layout_pages.jsonl, got {len(pages_data)}")
        page_nums = [p.get("page_number") for p in pages_data]
        if page_nums != [95, 96, 97, 98, 99]:
            raise ValueError(f"Expected pages 95..99, got {page_nums}")

        # Validate catalogs
        prim_cat = json.loads((pdir / "layout_primary_catalog.json").read_text(encoding="utf-8"), parse_constant=_reject_constant, parse_float=_parse_float)
        validate_catalog(prim_cat, pages_data)
        if len(prim_cat.get("units", [])) != 13:
            raise ValueError(f"Primary catalog must contain 13 units, got {len(prim_cat.get('units', []))}")
        if any(u.get("id") == "T95" for u in prim_cat.get("units", [])):
            raise ValueError("Forbidden T95 table unit found in primary catalog")

        table_cat = json.loads((pdir / "layout_table_catalog.json").read_text(encoding="utf-8"), parse_constant=_reject_constant, parse_float=_parse_float)
        validate_catalog(table_cat, pages_data)
        if len(table_cat.get("units", [])) != 14:
            raise ValueError(f"Table catalog must contain 14 units, got {len(table_cat.get('units', []))}")
        if not any(u.get("id") == "T95" for u in table_cat.get("units", [])):
            raise ValueError("Required T95 table unit missing from table catalog")

        # Validate index
        index_data = json.loads((pdir / "layout_index.json").read_text(encoding="utf-8"), parse_constant=_reject_constant, parse_float=_parse_float)
        validate_qwen_index(index_data)
        if len(index_data.get("chunks", [])) != 13:
            raise ValueError(f"Index chunk count mismatch: expected 13, got {len(index_data.get('chunks', []))}")

        # Validate structured bundle
        structured_bundle = json.loads((pdir / "structured.json").read_text(encoding="utf-8"), parse_constant=_reject_constant, parse_float=_parse_float)
        if not isinstance(structured_bundle, dict) or "tables" not in structured_bundle or "pages" not in structured_bundle:
            raise ValueError("structured.json in protocol dir is not a valid structured bundle")
        if structured_bundle.get("metadata", {}).get("doc_id") != manifest_data.get("metadata", {}).get("doc_id"):
            raise ValueError("structured.json metadata doc_id does not match manifest")

        # Validate schedule against frozen query files
        prim_q = json.loads((pdir / "primary_queries.json").read_text(encoding="utf-8"), parse_constant=_reject_constant, parse_float=_parse_float)
        tbl_q = json.loads((pdir / "table_queries.json").read_text(encoding="utf-8"), parse_constant=_reject_constant, parse_float=_parse_float)
        ext_q = json.loads((pdir / "extra_queries.json").read_text(encoding="utf-8"), parse_constant=_reject_constant, parse_float=_parse_float)
        fresh_schedule = build_schedule(prim_q, tbl_q, ext_q)
        if schedule != fresh_schedule:
            raise ValueError("Protocol schedule does not match freshly computed schedule from frozen query files")
        if fresh_schedule != auth_schedule:
            raise ValueError("Freshly computed schedule from frozen query files does not match authoritative schedule")


def prepare(
    output_dir: Path | str,
    *,
    transport: Any | None = None,
) -> dict[str, Any]:
    """Prepares and freezes protocol and assets in a new exclusive output folder."""
    out_p = Path(output_dir)
    if not out_p.is_absolute():
        out_p = (PROJECT_ROOT / out_p).resolve()
    else:
        out_p = out_p.resolve()

    validate_project_path(out_p)

    if out_p.exists() and any(out_p.iterdir()):
        raise FileExistsError(f"Target prepare output directory already exists and is not empty: {out_p}")

    # 1. Environment verification
    if transport is None:
        transport = MixedSingle27Transport(timeout=240.0)
    verify_environment(transport)

    # 2. Check historical protocol and asset bindings before copying
    p_hist = PROJECT_ROOT / DEFAULT_HISTORICAL_PROTOCOL_PATH
    if not p_hist.is_file():
        raise FileNotFoundError(f"Historical protocol not found: {p_hist}")
    hist_raw_bytes = p_hist.read_bytes()
    hist_raw_sha = hashlib.sha256(hist_raw_bytes).hexdigest()
    if hist_raw_sha != HISTORICAL_PROTOCOL_SHA256:
        raise ValueError(f"Historical protocol SHA mismatch: expected {HISTORICAL_PROTOCOL_SHA256}, got {hist_raw_sha}")
    hist_proto = json.loads(hist_raw_bytes.decode("utf-8"), parse_constant=_reject_constant, parse_float=_parse_float)

    for af, p_rel in [
        ("layout_pages.jsonl", DEFAULT_LAYOUT_PAGES_PATH),
        ("layout_index.json", DEFAULT_LAYOUT_INDEX_PATH),
        ("layout_primary_catalog.json", DEFAULT_LAYOUT_PRIMARY_CATALOG_PATH),
        ("layout_table_catalog.json", DEFAULT_LAYOUT_TABLE_CATALOG_PATH),
    ]:
        fp = PROJECT_ROOT / p_rel
        if not fp.is_file():
            raise FileNotFoundError(f"Missing historical asset file: {fp}")
        fb = fp.read_bytes()
        raw_h = hashlib.sha256(fb).hexdigest()
        if raw_h != hist_proto["generated_assets"][af]["raw_sha256"]:
            raise ValueError(f"Historical source raw SHA mismatch for {af}: expected {hist_proto['generated_assets'][af]['raw_sha256']}, got {raw_h}")
        canon_h = compute_canonical_sha256(fb, is_jsonl=af.endswith(".jsonl"))
        if canon_h != hist_proto["generated_assets"][af]["canonical_sha256"]:
            raise ValueError(f"Historical source canonical SHA mismatch for {af}: expected {hist_proto['generated_assets'][af]['canonical_sha256']}, got {canon_h}")

    # Check authoritative manifest and structured.json
    p_manifest = PROJECT_ROOT / DEFAULT_MANIFEST_PATH
    if not p_manifest.is_file():
        raise FileNotFoundError(f"Source artifact manifest not found: {p_manifest}")
    manifest_bytes = p_manifest.read_bytes()
    manifest_data = json.loads(manifest_bytes.decode("utf-8"), parse_constant=_reject_constant, parse_float=_parse_float)
    exp_struct_raw = manifest_data.get("files", {}).get("structured.json", {}).get("sha256")
    p_struct = PROJECT_ROOT / DEFAULT_STRUCTURED_BUNDLE_PATH
    if not p_struct.is_file():
        raise FileNotFoundError(f"Authoritative structured bundle not found: {p_struct}")
    struct_bytes = p_struct.read_bytes()
    struct_raw = hashlib.sha256(struct_bytes).hexdigest()
    if struct_raw != exp_struct_raw:
        raise ValueError(f"Authoritative structured.json raw SHA mismatch against manifest: expected {exp_struct_raw}, got {struct_raw}")
    auth_struct_canon = compute_canonical_sha256(struct_bytes, is_jsonl=False)

    # 3. Check and record input file hashes
    required_inputs = [PROJECT_ROOT / p for p in REQUIRED_INPUT_PATHS]
    for inp in required_inputs:
        if not inp.is_file():
            raise FileNotFoundError(f"Required input file not found: {inp}")

    hashes_before = {p.relative_to(PROJECT_ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in required_inputs}

    # 4. Read and validate source assets
    pages_raw_bytes = (PROJECT_ROOT / DEFAULT_LAYOUT_PAGES_PATH).read_bytes()
    pages_data = [
        json.loads(line, parse_constant=_reject_constant, parse_float=_parse_float)
        for line in pages_raw_bytes.decode("utf-8").splitlines() if line.strip()
    ]
    if len(pages_data) != 5:
        raise ValueError(f"Expected 5 layout pages (95..99), got {len(pages_data)}")
    page_nums = [p.get("page_number") for p in pages_data]
    if page_nums != [95, 96, 97, 98, 99]:
        raise ValueError(f"Expected page numbers 95..99, got {page_nums}")

    index_data = json.loads((PROJECT_ROOT / DEFAULT_LAYOUT_INDEX_PATH).read_text(encoding="utf-8"), parse_constant=_reject_constant, parse_float=_parse_float)
    validate_qwen_index(index_data)
    if len(index_data.get("chunks", [])) != 13:
        raise ValueError(f"Expected 13 chunks in layout index, got {len(index_data.get('chunks', []))}")

    prim_cat_data = json.loads((PROJECT_ROOT / DEFAULT_LAYOUT_PRIMARY_CATALOG_PATH).read_text(encoding="utf-8"), parse_constant=_reject_constant, parse_float=_parse_float)
    validate_catalog(prim_cat_data, pages_data)
    if len(prim_cat_data.get("units", [])) != 13:
        raise ValueError(f"Primary catalog must contain 13 units, got {len(prim_cat_data.get('units', []))}")
    if any(u.get("id") == "T95" for u in prim_cat_data.get("units", [])):
        raise ValueError("Primary catalog cannot contain T95")

    table_cat_data = json.loads((PROJECT_ROOT / DEFAULT_LAYOUT_TABLE_CATALOG_PATH).read_text(encoding="utf-8"), parse_constant=_reject_constant, parse_float=_parse_float)
    validate_catalog(table_cat_data, pages_data)
    if len(table_cat_data.get("units", [])) != 14:
        raise ValueError(f"Table catalog must contain 14 units, got {len(table_cat_data.get('units', []))}")
    if not any(u.get("id") == "T95" for u in table_cat_data.get("units", [])):
        raise ValueError("Table catalog must contain T95")

    structured_data = json.loads(struct_bytes.decode("utf-8"), parse_constant=_reject_constant, parse_float=_parse_float)
    if not isinstance(structured_data, dict):
        raise ValueError("structured.json must be a dict")

    # Verify reference freeze files using raw bytes
    ref_freeze_path = PROJECT_ROOT / DEFAULT_REFERENCE_FREEZE_PATH
    ref_freeze_bytes = ref_freeze_path.read_bytes()
    ref_freeze_sha = hashlib.sha256(ref_freeze_bytes).hexdigest()
    ref_freeze_data = json.loads(ref_freeze_bytes.decode("utf-8"), parse_constant=_reject_constant, parse_float=_parse_float)
    ref_files = ref_freeze_data.get("files", {})
    for rpath, exp_rhash in ref_files.items():
        rfp = PROJECT_ROOT / rpath
        if not rfp.is_file():
            raise FileNotFoundError(f"Reference file not found: {rfp}")
        act_rhash = hashlib.sha256(rfp.read_bytes()).hexdigest()
        if act_rhash != exp_rhash:
            raise ValueError(f"Reference file drift for {rpath}: expected {exp_rhash}, got {act_rhash}")

    # Read queries and build schedule
    prim_queries_data = json.loads((PROJECT_ROOT / DEFAULT_PRIMARY_QUERIES_PATH).read_text(encoding="utf-8"), parse_constant=_reject_constant, parse_float=_parse_float)
    tbl_queries_data = json.loads((PROJECT_ROOT / DEFAULT_TABLE_QUERIES_PATH).read_text(encoding="utf-8"), parse_constant=_reject_constant, parse_float=_parse_float)
    ext_queries_data = json.loads((PROJECT_ROOT / DEFAULT_EXTRA_QUERIES_PATH).read_text(encoding="utf-8"), parse_constant=_reject_constant, parse_float=_parse_float)

    schedule = build_schedule(prim_queries_data, tbl_queries_data, ext_queries_data)

    # Verify input files unchanged during prepare read phase
    hashes_after = {p.relative_to(PROJECT_ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in required_inputs}
    if hashes_before != hashes_after:
        raise RuntimeError("Input files were modified during prepare execution!")

    # Write asset files to destination folder
    out_p.mkdir(parents=True, exist_ok=True)

    def _write_json(name: str, data: Any) -> None:
        (out_p / name).write_text(
            json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )

    (out_p / "layout_pages.jsonl").write_bytes(pages_raw_bytes if pages_raw_bytes.endswith(b"\n") else pages_raw_bytes + b"\n")
    _write_json("layout_index.json", index_data)
    _write_json("layout_primary_catalog.json", prim_cat_data)
    _write_json("layout_table_catalog.json", table_cat_data)
    _write_json("structured.json", structured_data)
    _write_json("primary_queries.json", prim_queries_data)
    _write_json("table_queries.json", tbl_queries_data)
    _write_json("extra_queries.json", ext_queries_data)

    # Compute raw & canonical SHA for all generated asset files
    generated_assets: dict[str, dict[str, str]] = {}
    for af in ASSET_FILES:
        af_path = out_p / af
        af_bytes = af_path.read_bytes()
        raw_sha = hashlib.sha256(af_bytes).hexdigest()
        canon_sha = compute_canonical_sha256(af_bytes, is_jsonl=af.endswith(".jsonl"))

        # Verify against historical / authoritative bindings
        if af in ("layout_pages.jsonl", "layout_index.json", "layout_primary_catalog.json", "layout_table_catalog.json"):
            exp_c = hist_proto["generated_assets"][af]["canonical_sha256"]
            if canon_sha != exp_c:
                raise ValueError(f"Prepared copy {af} canonical SHA mismatch against historical original: expected {exp_c}, got {canon_sha}")
        elif af == "structured.json":
            if canon_sha != auth_struct_canon:
                raise ValueError(f"Prepared copy structured.json canonical SHA mismatch against authoritative bundle: expected {auth_struct_canon}, got {canon_sha}")
        elif af in ("primary_queries.json", "table_queries.json", "extra_queries.json"):
            orig_path = {
                "primary_queries.json": DEFAULT_PRIMARY_QUERIES_PATH,
                "table_queries.json": DEFAULT_TABLE_QUERIES_PATH,
                "extra_queries.json": DEFAULT_EXTRA_QUERIES_PATH,
            }[af]
            exp_c = compute_canonical_sha256((PROJECT_ROOT / orig_path).read_bytes(), is_jsonl=False)
            if canon_sha != exp_c:
                raise ValueError(f"Prepared copy {af} canonical SHA mismatch against authoritative original: expected {exp_c}, got {canon_sha}")

        generated_assets[af] = {
            "raw_sha256": raw_sha,
            "canonical_sha256": canon_sha,
        }

    source_closure = compute_source_closure()
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()

    protocol = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "server_version": SERVER_VERSION,
        "models": {
            "generation_model": DEFAULT_GENERATION_MODEL,
            "generation_digest": PINNED_9B_DIGEST,
            "verifier_model": VERIFIER_Q3_MODEL,
            "verifier_digest": PINNED_27B_DIGEST,
            "embedding_model": QWEN_EMBEDDING_MODEL,
            "embedding_digest": PINNED_EMBEDDING_DIGEST,
        },
        "parameters": {
            "generation_parameters": copy.deepcopy(DEFAULT_PARAMETERS),
            "verifier_parameters": copy.deepcopy(VERIFIER_Q3_PARAMETERS),
        },
        "execution_options": {
            "top_k": 4,
            "max_packages": 1,
            "closed_note_tails": False,
            "partial_generation": True,
        },
        "both_arms": {
            "control": {
                "pages_file": "layout_pages.jsonl",
                "index_file": "layout_index.json",
                "primary_catalog_file": "layout_primary_catalog.json",
                "table_catalog_file": "layout_table_catalog.json",
                "table_bundle_file": "structured.json",
                "table_enabled": False,
                "raw_sha256": generated_assets["layout_pages.jsonl"]["raw_sha256"],
                "canonical_sha256": generated_assets["layout_pages.jsonl"]["canonical_sha256"],
                "pages_count": len(pages_data),
                "chunks_count": len(index_data["chunks"]),
                "primary_units_count": len(prim_cat_data["units"]),
                "table_units_count": len(table_cat_data["units"]),
            },
            "candidate": {
                "pages_file": "layout_pages.jsonl",
                "index_file": "layout_index.json",
                "primary_catalog_file": "layout_primary_catalog.json",
                "table_catalog_file": "layout_table_catalog.json",
                "table_bundle_file": "structured.json",
                "table_enabled": True,
                "raw_sha256": generated_assets["layout_pages.jsonl"]["raw_sha256"],
                "canonical_sha256": generated_assets["layout_pages.jsonl"]["canonical_sha256"],
                "pages_count": len(pages_data),
                "chunks_count": len(index_data["chunks"]),
                "primary_units_count": len(prim_cat_data["units"]),
                "table_units_count": len(table_cat_data["units"]),
            },
        },
        "generated_assets": generated_assets,
        "schedule": schedule,
        "counts": {
            "pages_count": len(pages_data),
            "primary_queries_count": len(prim_queries_data.get("cases", [])),
            "table_queries_count": len(tbl_queries_data.get("cases", [])),
            "extra_queries_count": len(ext_queries_data.get("cases", [])),
            "total_queries_count": 21,
            "attempts_count": len(schedule),
            "primary_units_count": len(prim_cat_data["units"]),
            "table_units_count": len(table_cat_data["units"]),
            "chunks_count": len(index_data["chunks"]),
        },
        "source_freeze": source_closure,
        "source_artifact_manifest_sha256": manifest_sha,
        "reference_freeze_sha256": ref_freeze_sha,
        "reference_freeze_files": ref_files,
        "input_file_hashes": hashes_after,
        "historical_protocol_sha256": HISTORICAL_PROTOCOL_SHA256,
    }

    _write_json("protocol.json", protocol)
    return protocol


def run(
    protocol_path: Path | str,
    output_path: Path | str,
    *,
    transport: Any | None = None,
    execute_fn: Any | None = None,
) -> dict[str, Any]:
    """Execute the bounded 31-attempt evaluation matrix with strict anti-drift guards."""
    p_proto = Path(protocol_path)
    if not p_proto.is_absolute():
        p_proto = (PROJECT_ROOT / p_proto).resolve()
    else:
        p_proto = p_proto.resolve()

    p_out = Path(output_path)
    if not p_out.is_absolute():
        p_out = (PROJECT_ROOT / p_out).resolve()
    else:
        p_out = p_out.resolve()

    validate_project_path(p_proto)
    validate_project_path(p_out)

    if not p_proto.is_file():
        raise FileNotFoundError(f"Protocol file not found: {p_proto}")
    if p_out.exists():
        raise FileExistsError(f"Target run output destination already exists: {p_out}")

    original_protocol_bytes = p_proto.read_bytes()
    original_protocol_raw_sha = hashlib.sha256(original_protocol_bytes).hexdigest()
    protocol = json.loads(
        original_protocol_bytes.decode("utf-8"),
        parse_constant=_reject_constant,
        parse_float=_parse_float,
    )

    def verify_protocol_on_disk(stage_label: str) -> None:
        if not p_proto.is_file():
            raise RuntimeError(f"Protocol file was deleted during {stage_label}: {p_proto}")
        current_bytes = p_proto.read_bytes()
        current_sha = hashlib.sha256(current_bytes).hexdigest()
        if current_sha != original_protocol_raw_sha:
            raise RuntimeError(
                f"Protocol file on disk was modified during {stage_label}: "
                f"expected {original_protocol_raw_sha}, got {current_sha}"
            )

    # Validate protocol completely before any attempt
    verify_protocol_on_disk("pre-run")
    validate_protocol(protocol, protocol_dir=p_proto.parent)

    if transport is None:
        transport = MixedSingle27Transport(timeout=240.0)

    verify_environment(transport)

    proto_dir = p_proto.parent

    # Load frozen assets
    pages_data = [
        json.loads(line, parse_constant=_reject_constant, parse_float=_parse_float)
        for line in (proto_dir / "layout_pages.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    index_data = json.loads((proto_dir / "layout_index.json").read_text(encoding="utf-8"), parse_constant=_reject_constant, parse_float=_parse_float)
    prim_cat = json.loads((proto_dir / "layout_primary_catalog.json").read_text(encoding="utf-8"), parse_constant=_reject_constant, parse_float=_parse_float)
    table_cat = json.loads((proto_dir / "layout_table_catalog.json").read_text(encoding="utf-8"), parse_constant=_reject_constant, parse_float=_parse_float)
    table_bundle = json.loads((proto_dir / "structured.json").read_text(encoding="utf-8"), parse_constant=_reject_constant, parse_float=_parse_float)

    if execute_fn is None:
        execute_fn = execute_table_attempt

    schedule = protocol["schedule"]
    results: dict[str, Any] = {
        "schema_version": RESULTS_SCHEMA_VERSION,
        "protocol_path": str(p_proto),
        "protocol_raw_sha256": original_protocol_raw_sha,
        "run_status": "running",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "finished_at_utc": None,
        "expected_count": 31,
        "attempts": [],
        "summary": {
            "total_scheduled": 31,
            "completed_attempts": 0,
            "error_attempts": 0,
            "total_wall_seconds": 0.0,
        },
    }

    total_wall_seconds = 0.0
    completed_count = 0
    error_count = 0

    p_out.parent.mkdir(parents=True, exist_ok=True)

    for attempt in schedule:
        # Pre-attempt drift check on disk bytes and full protocol
        try:
            verify_protocol_on_disk(f"pre-attempt {attempt['attempt_index']}")
            validate_protocol(protocol, protocol_dir=p_proto.parent)
        except Exception as exc:
            results["run_status"] = "invalid"
            results["error"] = f"Pre-attempt integrity drift at attempt {attempt['attempt_index']}: {exc}"
            results["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
            checkpoint_atomic(results, p_out)
            raise RuntimeError(results["error"]) from exc

        cohort = attempt["cohort"]
        cond = attempt["condition"]
        enabled = (cond == "candidate")

        units_data = prim_cat if cohort == "primary" else table_cat
        cat_file = "layout_primary_catalog.json" if cohort == "primary" else "layout_table_catalog.json"

        input_hashes = {
            "pages_sha256": protocol["generated_assets"]["layout_pages.jsonl"]["raw_sha256"],
            "index_sha256": protocol["generated_assets"]["layout_index.json"]["canonical_sha256"],
            "catalog_sha256": protocol["generated_assets"][cat_file]["canonical_sha256"],
            "table_bundle_sha256": protocol["generated_assets"]["structured.json"]["canonical_sha256"],
        }

        attempt_spec = {
            "attempt_index": attempt["attempt_index"],
            "case_id": attempt["case_id"],
            "query": attempt["query"],
            "arm": "mixed",
        }

        attempt_status = "completed"
        attempt_error = None
        raw_attempt: dict[str, Any] = {}

        try:
            # Call execute_fn EXACTLY ONCE per scheduled condition with real keyword args always.
            # NO retry without table_bundle or enabled. ANY exception is retained as technical error.
            raw_attempt = execute_fn(
                attempt_spec,
                index_data=index_data,
                units_data=units_data,
                pages_data=pages_data,
                transport=transport,
                table_bundle=table_bundle,
                enabled=enabled,
            )

            is_valid_shape, shape_error = _validate_attempt_result(raw_attempt, attempt)
            if not is_valid_shape:
                attempt_status = "error"
                attempt_error = f"Technical invalid attempt result: {shape_error}"
                error_count += 1
                raw_attempt = {
                    "synthetic_exception_record": True,
                    "status": "error",
                    "error": attempt_error,
                    "arm": "mixed",
                    "case_id": attempt["case_id"],
                    "query": attempt["query"],
                    "attempt_index": attempt["attempt_index"],
                    "formatted_answer": "",
                    "coverage_publication": None,
                    "factual_outcome": None,
                    "lifecycle_records": [],
                    "stats_summary": {},
                    "query_wall_seconds": 0.0,
                    "table_audit": {
                        "enabled": enabled,
                        "is_table_query": (cohort in ("table", "extra")) if enabled else False,
                        "gate_decision": "technical_error",
                        "error": attempt_error,
                    },
                }
            elif raw_attempt.get("status") == "error":
                attempt_status = "error"
                attempt_error = str(raw_attempt.get("error", "Table grounding or verification error"))
                error_count += 1
            else:
                attempt_status = "completed"
                attempt_error = None
                completed_count += 1
        except Exception as exc:
            attempt_status = "error"
            attempt_error = str(exc)
            error_count += 1
            raw_attempt = {
                "synthetic_exception_record": True,
                "status": "error",
                "error": str(exc),
                "arm": "mixed",
                "case_id": attempt["case_id"],
                "query": attempt["query"],
                "attempt_index": attempt["attempt_index"],
                "formatted_answer": "",
                "coverage_publication": None,
                "factual_outcome": None,
                "lifecycle_records": [],
                "stats_summary": {},
                "query_wall_seconds": 0.0,
                "table_audit": {
                    "enabled": enabled,
                    "is_table_query": (cohort in ("table", "extra")) if enabled else False,
                    "gate_decision": "technical_error",
                    "error": str(exc),
                },
            }

        query_wall = raw_attempt.get("query_wall_seconds", 0.0) if isinstance(raw_attempt, dict) else 0.0
        total_wall_seconds += query_wall

        raw_canonical_sha = hashlib.sha256(
            json.dumps(raw_attempt, sort_keys=True, ensure_ascii=False, allow_nan=False).encode("utf-8")
        ).hexdigest()

        attempt_record = {
            "attempt_index": attempt["attempt_index"],
            "question_index": attempt["question_index"],
            "cohort": cohort,
            "case_id": attempt["case_id"],
            "query": attempt["query"],
            "condition": cond,
            "arm": "mixed",
            "table_enabled": enabled,
            "status": attempt_status,
            "error": attempt_error,
            "formatted_answer": raw_attempt.get("formatted_answer", "") if isinstance(raw_attempt, dict) else "",
            "coverage_publication": copy.deepcopy(raw_attempt.get("coverage_publication")) if isinstance(raw_attempt, dict) else None,
            "factual_outcome": raw_attempt.get("factual_outcome") if isinstance(raw_attempt, dict) else None,
            "lifecycle_records": raw_attempt.get("lifecycle_records", []) if isinstance(raw_attempt, dict) else [],
            "stats_summary": raw_attempt.get("stats_summary", {}) if isinstance(raw_attempt, dict) else {},
            "table_audit": copy.deepcopy(raw_attempt.get("table_audit", {})) if isinstance(raw_attempt, dict) else {},
            "query_wall_seconds": query_wall,
            "input_hashes": input_hashes,
            "raw_attempt_canonical_sha256": raw_canonical_sha,
            "raw_attempt": raw_attempt,
        }
        results["attempts"].append(attempt_record)
        results["summary"]["completed_attempts"] = completed_count
        results["summary"]["error_attempts"] = error_count
        results["summary"]["total_wall_seconds"] = total_wall_seconds

        checkpoint_atomic(results, p_out)

        # Post-attempt drift check on disk bytes and full protocol
        try:
            verify_protocol_on_disk(f"post-attempt {attempt['attempt_index']}")
            validate_protocol(protocol, protocol_dir=p_proto.parent)
        except Exception as exc:
            results["run_status"] = "invalid"
            results["error"] = f"Post-attempt integrity drift at attempt {attempt['attempt_index']}: {exc}"
            results["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
            checkpoint_atomic(results, p_out)
            raise RuntimeError(results["error"]) from exc

    verify_protocol_on_disk("finalization")
    validate_protocol(protocol, protocol_dir=p_proto.parent)

    results["run_status"] = "completed"
    results["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    checkpoint_atomic(results, p_out)
    return results


def export_review(
    protocol_path: Path | str,
    results_path: Path | str,
    output_path: Path | str,
) -> dict[str, Any]:
    """Export blinded review bundle with neutral labels and sibling review_key.json."""
    p_proto = Path(protocol_path)
    if not p_proto.is_absolute():
        p_proto = (PROJECT_ROOT / p_proto).resolve()
    else:
        p_proto = p_proto.resolve()

    p_res = Path(results_path)
    if not p_res.is_absolute():
        p_res = (PROJECT_ROOT / p_res).resolve()
    else:
        p_res = p_res.resolve()

    p_out = Path(output_path)
    if not p_out.is_absolute():
        p_out = (PROJECT_ROOT / p_out).resolve()
    else:
        p_out = p_out.resolve()

    key_path = p_out.parent / "review_key.json"

    # Reject bundle destination equal to review_key.json (resolved paths equality, case-normalized on Windows)
    # Reject collisions BEFORE creating or writing either file
    p_out_resolved = p_out.resolve()
    key_path_resolved = key_path.resolve()
    if os.name == "nt":
        same_dest = p_out_resolved.as_posix().lower() == key_path_resolved.as_posix().lower()
    else:
        same_dest = p_out_resolved == key_path_resolved
    if same_dest:
        raise ValueError(f"Review bundle destination cannot be named review_key.json: {p_out}")

    validate_project_path(p_proto)
    validate_project_path(p_res)
    validate_project_path(p_out)
    validate_project_path(key_path)

    if not p_proto.is_file():
        raise FileNotFoundError(f"Protocol file not found: {p_proto}")
    if not p_res.is_file():
        raise FileNotFoundError(f"Results file not found: {p_res}")

    # Refuse overwrite on either bundle output or review_key.json
    if p_out.exists():
        raise FileExistsError(f"Review bundle destination already exists: {p_out}")
    if key_path.exists():
        raise FileExistsError(f"Review key destination already exists: {key_path}")

    protocol = json.loads(p_proto.read_text(encoding="utf-8"), parse_constant=_reject_constant, parse_float=_parse_float)
    results = json.loads(p_res.read_text(encoding="utf-8"), parse_constant=_reject_constant, parse_float=_parse_float)

    # Validate CURRENT protocol/source/input/assets before output writes
    validate_protocol(protocol, protocol_dir=p_proto.parent)

    proto_sha = hashlib.sha256(p_proto.read_bytes()).hexdigest()
    if results.get("protocol_raw_sha256") != proto_sha:
        raise ValueError("Results protocol_raw_sha256 does not match protocol file hash")

    if results.get("run_status") != "completed":
        raise ValueError(f"Cannot export review for run with status {results.get('run_status')!r}; must be 'completed'")

    attempts = results.get("attempts", [])
    if len(attempts) != 31:
        raise ValueError(f"Expected exactly 31 attempts in results, got {len(attempts)}")

    schedule = protocol.get("schedule", [])
    if len(schedule) != 31:
        raise ValueError(f"Protocol schedule must contain 31 attempts, got {len(schedule)}")

    seen_tuple_keys: set[tuple[str, str, str]] = set()

    for idx in range(31):
        att = attempts[idx]
        sched = schedule[idx]

        for field in ("attempt_index", "question_index", "cohort", "case_id", "query", "condition", "arm", "table_enabled"):
            if att.get(field) != sched.get(field):
                raise ValueError(
                    f"Attempt at index {idx} field {field!r} mismatch: "
                    f"results has {att.get(field)!r}, schedule has {sched.get(field)!r}"
                )

        tuple_key = (att["cohort"], att["case_id"], att["condition"])
        if tuple_key in seen_tuple_keys:
            raise ValueError(f"Duplicate attempt tuple key in results: {tuple_key}")
        seen_tuple_keys.add(tuple_key)

        cohort = att["cohort"]
        cat_file = "layout_primary_catalog.json" if cohort == "primary" else "layout_table_catalog.json"
        expected_cat_sha = protocol["generated_assets"][cat_file]["canonical_sha256"]
        expected_pages_sha = protocol["generated_assets"]["layout_pages.jsonl"]["raw_sha256"]
        expected_index_sha = protocol["generated_assets"]["layout_index.json"]["canonical_sha256"]
        expected_bundle_sha = protocol["generated_assets"]["structured.json"]["canonical_sha256"]

        att_hashes = att.get("input_hashes")
        if not isinstance(att_hashes, dict):
            raise ValueError(f"Attempt {idx+1} missing or invalid input_hashes dict")

        for req_h in ("pages_sha256", "catalog_sha256", "index_sha256", "table_bundle_sha256"):
            val = att_hashes.get(req_h)
            if not isinstance(val, str) or not val.strip():
                raise ValueError(f"Attempt {idx+1} missing or empty input_hashes field {req_h!r}")

        if att_hashes["pages_sha256"] != expected_pages_sha:
            raise ValueError(f"Attempt {idx+1} input pages_sha256 mismatch against protocol")
        if att_hashes["catalog_sha256"] != expected_cat_sha:
            raise ValueError(f"Attempt {idx+1} input catalog_sha256 mismatch against protocol")
        if att_hashes["index_sha256"] != expected_index_sha:
            raise ValueError(f"Attempt {idx+1} input index_sha256 mismatch against protocol")
        if att_hashes["table_bundle_sha256"] != expected_bundle_sha:
            raise ValueError(f"Attempt {idx+1} input table_bundle_sha256 mismatch against protocol")

        raw_att = att.get("raw_attempt")
        if not isinstance(raw_att, dict):
            raise ValueError(f"Attempt {idx+1} raw_attempt must be a dict")
        if raw_att.get("arm") != "mixed":
            raise ValueError(f"Attempt {idx+1} raw_attempt arm must be 'mixed', got {raw_att.get('arm')!r}")
        if raw_att.get("case_id") != att["case_id"]:
            raise ValueError(f"Attempt {idx+1} raw_attempt case_id mismatch: {raw_att.get('case_id')!r} != {att['case_id']!r}")
        if raw_att.get("query") != att["query"]:
            raise ValueError(f"Attempt {idx+1} raw_attempt query mismatch")
        if raw_att.get("attempt_index") != att["attempt_index"]:
            raise ValueError(f"Attempt {idx+1} raw_attempt attempt_index mismatch")

        # Verify raw_attempt canonical sha against recomputed sha
        raw_sha_field = att.get("raw_attempt_canonical_sha256")
        if not isinstance(raw_sha_field, str) or not raw_sha_field.strip():
            raise ValueError(f"Attempt {idx+1} missing or empty raw_attempt_canonical_sha256")

        raw_sha = hashlib.sha256(
            json.dumps(raw_att, sort_keys=True, ensure_ascii=False, allow_nan=False).encode("utf-8")
        ).hexdigest()
        if raw_sha_field != raw_sha:
            raise ValueError(f"Attempt {idx+1} raw_attempt_canonical_sha256 mismatch: expected {raw_sha}, got {raw_sha_field}")

        # Strict binding of outer fields to raw_attempt
        raw_ans = raw_att.get("formatted_answer", "")
        if not isinstance(raw_ans, str):
            raise ValueError(f"Attempt {idx+1} raw_attempt formatted_answer must be str")
        if att.get("formatted_answer") != raw_ans:
            raise ValueError(f"Attempt {idx+1} formatted_answer altered from raw_attempt")

        att_status = att.get("status")
        if att_status not in ("completed", "error"):
            raise ValueError(f"Attempt {idx+1} invalid status: {att_status!r}")

        raw_status = raw_att.get("status")
        if raw_status != att_status:
            raise ValueError(f"Attempt {idx+1} status altered from raw_attempt")

        raw_err = raw_att.get("error")
        if att_status == "error":
            if not isinstance(att.get("error"), str) or not att["error"].strip():
                raise ValueError(f"Attempt {idx+1} marked as error but missing outer error message")
            if not isinstance(raw_err, str) or not raw_err.strip():
                raise ValueError(f"Attempt {idx+1} marked as error but raw_attempt missing error message")
            if att.get("error") != raw_err:
                raise ValueError(f"Attempt {idx+1} error altered from raw_attempt: {att.get('error')!r} != {raw_err!r}")
        else:
            if att.get("error") is not None:
                raise ValueError(f"Attempt {idx+1} marked completed but has outer error: {att.get('error')!r}")
            if raw_err is not None:
                raise ValueError(f"Attempt {idx+1} marked completed but raw_attempt has error: {raw_err!r}")

        raw_cov = raw_att.get("coverage_publication")
        att_cov = att.get("coverage_publication")
        if not _json_type_equal(att_cov, raw_cov):
            raise ValueError(f"Attempt {idx+1} coverage_publication altered from raw_attempt")

        # Authoritative attempt result validation
        is_valid_att, att_err_reason = _validate_attempt_result(raw_att, sched)
        if not is_valid_att:
            raise ValueError(f"Attempt {idx+1} raw_attempt failed authoritative validation: {att_err_reason}")

    # Group attempts by question_index (0..20)
    cases_by_qidx: dict[int, list[dict[str, Any]]] = {}
    for att in attempts:
        cases_by_qidx.setdefault(att["question_index"], []).append(att)

    if len(cases_by_qidx) != 21:
        raise ValueError(f"Expected 21 unique questions, got {len(cases_by_qidx)}")

    review_cases: list[dict[str, Any]] = []
    mapping_dict: dict[str, dict[str, str]] = {}

    for q_idx in range(21):
        pair = cases_by_qidx.get(q_idx, [])
        if q_idx < 11:
            # Primary cohort: candidate only (single neutral label)
            if len(pair) != 1:
                raise ValueError(f"Expected 1 attempt for primary question index {q_idx}, got {len(pair)}")
            att = pair[0]
            cid = att["case_id"]
            raw_att = att["raw_attempt"]

            arm_single = {
                "formatted_answer": raw_att.get("formatted_answer", ""),
                "coverage_publication": copy.deepcopy(raw_att.get("coverage_publication")),
                "technical_error": raw_att.get("error"),
            }

            review_cases.append({
                "question_index": q_idx,
                "cohort": att["cohort"],
                "case_id": cid,
                "query": att["query"],
                "source_pages": [95, 96, 97, 98, 99],
                "arms": {
                    "single": arm_single,
                },
            })
            mapping_dict[cid] = {
                "single": att["condition"],
            }
        else:
            # Table or extra cohort: neutral A/B pair
            if len(pair) != 2:
                raise ValueError(f"Expected 2 attempts for paired question index {q_idx}, got {len(pair)}")
            att_first, att_second = pair[0], pair[1]
            cid = att_first["case_id"]
            if cid != att_second["case_id"]:
                raise ValueError(f"Case ID mismatch within question index {q_idx}: {cid} != {att_second['case_id']}")
            if att_first["condition"] == att_second["condition"]:
                raise ValueError(f"Question index {q_idx} has duplicate condition {att_first['condition']!r}")

            raw_first = att_first["raw_attempt"]
            raw_second = att_second["raw_attempt"]

            arm_A = {
                "formatted_answer": raw_first.get("formatted_answer", ""),
                "coverage_publication": copy.deepcopy(raw_first.get("coverage_publication")),
                "technical_error": raw_first.get("error"),
            }
            arm_B = {
                "formatted_answer": raw_second.get("formatted_answer", ""),
                "coverage_publication": copy.deepcopy(raw_second.get("coverage_publication")),
                "technical_error": raw_second.get("error"),
            }

            review_cases.append({
                "question_index": q_idx,
                "cohort": att_first["cohort"],
                "case_id": cid,
                "query": att_first["query"],
                "source_pages": [95, 96, 97, 98, 99],
                "arms": {
                    "A": arm_A,
                    "B": arm_B,
                },
            })
            mapping_dict[cid] = {
                "A": att_first["condition"],
                "B": att_second["condition"],
            }

    res_sha = hashlib.sha256(p_res.read_bytes()).hexdigest()

    review_bundle = {
        "schema_version": REVIEW_SCHEMA_VERSION,
        "protocol_raw_sha256": proto_sha,
        "results_raw_sha256": res_sha,
        "cases": review_cases,
    }

    review_key = {
        "schema_version": REVIEW_KEY_SCHEMA_VERSION,
        "protocol_raw_sha256": proto_sha,
        "results_raw_sha256": res_sha,
        "mapping": mapping_dict,
    }

    p_out.parent.mkdir(parents=True, exist_ok=True)
    p_out.write_text(json.dumps(review_bundle, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    key_path.write_text(json.dumps(review_key, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    return review_bundle


def main() -> int:
    """CLI entrypoint for table grounding comparison."""
    parser = argparse.ArgumentParser(description="Phase 3 table-grounding comparison runner")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # prepare --output-dir
    p_prep = subparsers.add_parser("prepare", help="Prepare and freeze protocol and assets")
    p_prep.add_argument("--output-dir", required=True, help="Directory to store frozen protocol and assets")

    # run --protocol --output
    p_run = subparsers.add_parser("run", help="Run 31-condition evaluation matrix")
    p_run.add_argument("--protocol", required=True, help="Path to protocol.json")
    p_run.add_argument("--output", required=True, help="Path to destination results.json")

    # export-review --protocol --results --output
    p_exp = subparsers.add_parser("export-review", help="Export blinded neutral review bundle")
    p_exp.add_argument("--protocol", required=True, help="Path to protocol.json")
    p_exp.add_argument("--results", required=True, help="Path to completed results.json")
    p_exp.add_argument("--output", required=True, help="Path to destination review_bundle.json")

    args = parser.parse_args()

    if args.command == "prepare":
        prepare(args.output_dir)
        print(f"Protocol prepared successfully in {args.output_dir}")
        return 0
    elif args.command == "run":
        run(args.protocol, args.output)
        print(f"Run completed successfully; results written to {args.output}")
        return 0
    elif args.command == "export-review":
        export_review(args.protocol, args.results, args.output)
        print(f"Review exported successfully to {args.output}")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
