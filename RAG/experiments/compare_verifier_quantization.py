"""Verifier quantization comparison (Phase 1, Phase 2, and Phase 3).

Isolates claim/evidence verification across 2 model quantization arms:
- Q8: qwen3.5:9b-q8_0 (9.0B, Q8_0 quantization)
- Q3: hf.co/gguf/Qwen3.8-27B-GGUF:Q3_K_M (27.3B, Q3_K_M quantization)

Aligns all effective sampling with the historical Arm B reference (server defaults < show.parameters < request).
Pins Ollama server version 0.40.11, validates exact installed quantization metadata (Q8_0 / Q3_K_M),
and enforces strict digest matching before each arm.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any
import urllib.request

from experiments.compare_verifier_capacity import (
    ALLOWED_VERDICT_LABELS,
    CONTEXT_SAFETY_MARGIN,
    DEFAULT_OLLAMA_DEFAULTS_PATH,
    DEFAULT_OLLAMA_URL,
    DEFAULT_RUN_TIMEOUT,
    MAX_CONTEXT_WINDOW,
    MAX_RUN_RESPONSE_BYTES,
    OllamaResponseError,
    OllamaTransport,
    OllamaTransportProtocol,
    assert_finite,
    canonical_json_bytes,
    canonical_sha256,
    check_model_supports_thinking,
    compute_file_sha256,
    compute_numeric_summary,
    extract_verdict_label,
    is_valid_sha256,
    parse_show_parameters,
    resolve_effective_sampling,
    sanitize_finite,
    unload_resident_models,
    validate_sampling_match,
)
from rag_pensiones.answer import _parse_float, _reject_constant
from rag_pensiones.grounding import (
    MAX_MESSAGES_BYTES_VERIFICATION,
    normalize_verification_payload,
    validate_verification_payload,
)
from rag_pensiones.ingest import PROJECT_ROOT, validate_project_path

INPUTS_SCHEMA_VERSION = "verifier-capacity-inputs-v1"
GOLD_SCHEMA_VERSION = "verifier-capacity-gold-v1"
SCHEDULE_SCHEMA_VERSION = "verifier-quantization-schedule-v1"
PROTOCOL_SCHEMA_VERSION = "verifier-quantization-protocol-v1"
RUN_RESULTS_SCHEMA_VERSION = "verifier-quantization-run-results-v1"
SCORE_REPORT_SCHEMA_VERSION = "verifier-quantization-score-report-v1"

PRIOR_FROZEN_DIR = PROJECT_ROOT / "evaluation" / "verifier_capacity_2026-10-26" / "frozen_final"
PRIOR_PROTOCOL_PATH = PRIOR_FROZEN_DIR / "protocol.json"
PRIOR_INPUTS_PATH = PRIOR_FROZEN_DIR / "inputs.json"
PRIOR_GOLD_PATH = PRIOR_FROZEN_DIR / "gold.json"
PRIOR_SCHEDULE_PATH = PRIOR_FROZEN_DIR / "schedule.json"

DEFAULT_ENVIRONMENT_PATH = (
    PROJECT_ROOT / "evaluation" / "verifier_quantization_2026-10-26" / "environment.json"
)

REQUIRED_CODE_MODULES: dict[str, str] = {
    "grounding": "rag_pensiones/grounding.py",
    "answer": "rag_pensiones/answer.py",
    "ingest": "rag_pensiones/ingest.py",
    "compare_verifier_capacity": "experiments/compare_verifier_capacity.py",
    "compare_verifier_quantization": "experiments/compare_verifier_quantization.py",
}

QUANTIZATION_ALLOWED_MODELS = (
    "qwen3.5:9b-q8_0",
    "hf.co/gguf/Qwen3.8-27B-GGUF:Q3_K_M",
)

# Reference Arm B show parameters for qwen3.5:9b
REFERENCE_ARM_B_SHOW_PARAMETERS: dict[str, Any] = {
    "presence_penalty": 1.5,
    "temperature": 1,
    "top_k": 20,
    "top_p": 0.95,
}

# Reference Arm B request options
REFERENCE_ARM_B_OPTIONS: dict[str, Any] = {
    "temperature": 0,
    "seed": 42,
    "num_ctx": 8192,
    "num_predict": 4096,
}

# Two experimental arms: Q8 (9B Q8_0) and Q3 (27B Q3_K_M)
QUANTIZATION_ARMS: dict[str, dict[str, Any]] = {
    "Q8": {
        "arm_id": "Q8",
        "model": "qwen3.5:9b-q8_0",
        "think": False,
        "quantization_level": "Q8_0",
        "options": {
            "temperature": 0,
            "seed": 42,
            "num_ctx": 8192,
            "num_predict": 4096,
        },
        "description": "qwen3.5:9b-q8_0, think=false, num_predict=4096 (Q8_0 quantization, 9.0B parameter base, sampling aligned with Arm B reference).",
    },
    "Q3": {
        "arm_id": "Q3",
        "model": "hf.co/gguf/Qwen3.8-27B-GGUF:Q3_K_M",
        "think": False,
        "quantization_level": "Q3_K_M",
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
        "description": "hf.co/gguf/Qwen3.8-27B-GGUF:Q3_K_M, think=false, num_predict=4096 (Q3_K_M quantization, 27.3B parameter base; explicitly aligns 9B defaults presence_penalty=1.5, top_k=20, top_p=0.95, min_p=0.0 given empty show parameters).",
    },
}


class QuantizationOllamaTransport(OllamaTransport):
    """Scoped subclass of OllamaTransport restricted to the quantization comparison model pair."""

    def __init__(
        self,
        base_url: str = DEFAULT_OLLAMA_URL,
        timeout: float = DEFAULT_RUN_TIMEOUT,
        opener: urllib.request.OpenerDirector | None = None,
    ) -> None:
        super().__init__(base_url=base_url, timeout=timeout, opener=opener)
        self.allowed_models = QUANTIZATION_ALLOWED_MODELS


def build_quantization_schedule(
    case_ids: list[str],
    dev_status_map: dict[str, str],
) -> list[dict[str, Any]]:
    """Builds deterministic schedule for 2 arms (Q8, Q3) with identical case order.

    Each arm executes all 29 cases once in base forward order (schedule_index 0..28 for Q8, 31..59 for Q3),
    followed by 2 additional correlated repeats of counterexample b05 (repeat_index 2, 3).
    Total runs: 2 * (29 + 2) = 62 scheduled executions.
    """
    items: list[dict[str, Any]] = []
    schedule_idx = 0

    for block_idx, arm_id in enumerate(["Q8", "Q3"]):
        # 1. Primary evaluation of all 29 cases in base forward order
        for cid in case_ids:
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


def prepare(
    output_dir: Path | str,
    *,
    prior_protocol_path: Path | str = PRIOR_PROTOCOL_PATH,
    prior_inputs_path: Path | str = PRIOR_INPUTS_PATH,
    prior_gold_path: Path | str = PRIOR_GOLD_PATH,
    prior_schedule_path: Path | str = PRIOR_SCHEDULE_PATH,
    defaults_path: Path | str = DEFAULT_OLLAMA_DEFAULTS_PATH,
    environment_path: Path | str = DEFAULT_ENVIRONMENT_PATH,
) -> dict[str, Any]:
    """Prepares and freezes inputs.json, gold.json, schedule.json, and protocol.json in output_dir.

    Copies prior frozen_final inputs and gold byte-for-byte after validating prior protocol pins.
    Refuses to overwrite existing files or non-empty target directories.
    Runs strictly offline with zero network and zero live model calls.
    """
    out_dir = Path(output_dir)
    target_filenames = ["inputs.json", "gold.json", "schedule.json", "protocol.json"]

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

    # 1. Resolve prior source files safely
    res_prior_prot = validate_project_path(Path(prior_protocol_path))
    if not res_prior_prot.is_file():
        raise FileNotFoundError(f"Prior protocol file not found at {res_prior_prot}")

    res_prior_inputs = validate_project_path(Path(prior_inputs_path))
    if not res_prior_inputs.is_file():
        raise FileNotFoundError(f"Prior inputs file not found at {res_prior_inputs}")

    res_prior_gold = validate_project_path(Path(prior_gold_path))
    if not res_prior_gold.is_file():
        raise FileNotFoundError(f"Prior gold file not found at {res_prior_gold}")

    res_prior_sched = validate_project_path(Path(prior_schedule_path))
    if not res_prior_sched.is_file():
        raise FileNotFoundError(f"Prior schedule file not found at {res_prior_sched}")

    res_defaults = validate_project_path(Path(defaults_path))
    if not res_defaults.is_file():
        raise FileNotFoundError(f"Defaults reference file not found at {res_defaults}")

    # 2. Validate prior protocol and verify byte-for-byte / canonical pins
    prior_prot_raw = res_prior_prot.read_bytes()
    try:
        prior_prot_data = json.loads(
            prior_prot_raw.decode("utf-8"),
            parse_constant=_reject_constant,
            parse_float=_parse_float,
        )
    except Exception as exc:
        raise ValueError(f"Malformed JSON in prior protocol {res_prior_prot}: {exc}") from exc
    assert_finite(prior_prot_data)

    prior_hashes = prior_prot_data.get("hashes", {})
    if not isinstance(prior_hashes, dict):
        raise ValueError(f"Prior protocol missing valid 'hashes' dictionary: {res_prior_prot}")

    for req_key in [
        "inputs_canonical_sha256",
        "gold_canonical_sha256",
        "defaults_canonical_sha256",
        "schedule_canonical_sha256",
    ]:
        h_val = prior_hashes.get(req_key)
        if not is_valid_sha256(h_val):
            raise ValueError(f"Prior protocol missing or invalid {req_key} pin: {h_val!r}")

    exp_prior_inputs_hash = prior_hashes["inputs_canonical_sha256"]
    exp_prior_gold_hash = prior_hashes["gold_canonical_sha256"]
    exp_prior_defaults_hash = prior_hashes["defaults_canonical_sha256"]
    exp_prior_sched_hash = prior_hashes["schedule_canonical_sha256"]

    # Read prior inputs
    inputs_bytes = res_prior_inputs.read_bytes()
    try:
        inputs_data = json.loads(
            inputs_bytes.decode("utf-8"),
            parse_constant=_reject_constant,
            parse_float=_parse_float,
        )
    except Exception as exc:
        raise ValueError(f"Malformed JSON in prior inputs {res_prior_inputs}: {exc}") from exc
    assert_finite(inputs_data)
    inputs_canonical_hash = canonical_sha256(inputs_data)
    if inputs_canonical_hash != exp_prior_inputs_hash:
        raise ValueError(
            f"Prior inputs canonical hash ({inputs_canonical_hash}) does not match prior protocol pin ({exp_prior_inputs_hash})"
        )

    # Read prior gold
    gold_bytes = res_prior_gold.read_bytes()
    try:
        gold_data = json.loads(
            gold_bytes.decode("utf-8"),
            parse_constant=_reject_constant,
            parse_float=_parse_float,
        )
    except Exception as exc:
        raise ValueError(f"Malformed JSON in prior gold {res_prior_gold}: {exc}") from exc
    assert_finite(gold_data)
    gold_canonical_hash = canonical_sha256(gold_data)
    if gold_canonical_hash != exp_prior_gold_hash:
        raise ValueError(
            f"Prior gold canonical hash ({gold_canonical_hash}) does not match prior protocol pin ({exp_prior_gold_hash})"
        )

    # Read defaults
    defaults_bytes = res_defaults.read_bytes()
    try:
        defaults_data = json.loads(
            defaults_bytes.decode("utf-8"),
            parse_constant=_reject_constant,
            parse_float=_parse_float,
        )
    except Exception as exc:
        raise ValueError(f"Malformed JSON in defaults file {res_defaults}: {exc}") from exc
    assert_finite(defaults_data)
    defaults_canonical_hash = canonical_sha256(defaults_data)
    if defaults_canonical_hash != exp_prior_defaults_hash:
        raise ValueError(
            f"Defaults canonical hash ({defaults_canonical_hash}) does not match prior protocol pin ({exp_prior_defaults_hash})"
        )

    # Read prior schedule and validate historical schedule pin before consuming development_status
    prior_sched_bytes = res_prior_sched.read_bytes()
    try:
        prior_sched_data = json.loads(
            prior_sched_bytes.decode("utf-8"),
            parse_constant=_reject_constant,
            parse_float=_parse_float,
        )
    except Exception as exc:
        raise ValueError(f"Malformed JSON in prior schedule {res_prior_sched}: {exc}") from exc
    assert_finite(prior_sched_data)

    prior_sched_canonical_hash = canonical_sha256(prior_sched_data)
    if prior_sched_canonical_hash != exp_prior_sched_hash:
        raise ValueError(
            f"Prior schedule canonical hash ({prior_sched_canonical_hash}) does not match prior protocol pin ({exp_prior_sched_hash})"
        )

    dev_status_map: dict[str, str] = {}
    for it in prior_sched_data.get("items", []):
        cid = it.get("case_id")
        stat = it.get("development_status")
        if cid and stat and cid not in dev_status_map:
            dev_status_map[cid] = stat

    cases_list = inputs_data.get("cases", [])
    if not isinstance(cases_list, list) or len(cases_list) != 29:
        raise ValueError(f"Expected 29 cases in inputs.json, got {len(cases_list)}")
    case_ids = [c["id"] for c in cases_list]

    for cid in case_ids:
        if cid not in dev_status_map:
            raise ValueError(f"Case {cid!r} missing from dev_status_map extracted from prior schedule")

    # 3. Build new deterministic schedule (62 items: 31 for Q8, 31 for Q3)
    schedule_items = build_quantization_schedule(case_ids, dev_status_map)
    schedule_data = {
        "schema_version": SCHEDULE_SCHEMA_VERSION,
        "total_runs": len(schedule_items),
        "runs_per_arm": len(schedule_items) // len(QUANTIZATION_ARMS),
        "arm_blocks": list(QUANTIZATION_ARMS.keys()),
        "ordering_notes": (
            "Fixed arm blocks Q8, Q3. Both arms execute identical forward base order for 29 cases, "
            "followed by 2 additional correlated b05 repeats. Exactly 31 runs per arm, 62 runs total."
        ),
        "items": schedule_items,
    }
    assert_finite(schedule_data)
    schedule_canonical_hash = canonical_sha256(schedule_data)

    # 4. Hash code modules
    code_hashes: dict[str, dict[str, str]] = {}
    for mod_name, rel_str in REQUIRED_CODE_MODULES.items():
        p = validate_project_path(PROJECT_ROOT / rel_str)
        if not p.is_file():
            raise FileNotFoundError(f"Required source code file for pin {mod_name!r} not found at {p}")
        h = compute_file_sha256(p)
        if not is_valid_sha256(h):
            raise ValueError(f"Computed invalid SHA-256 for {mod_name}: {h!r}")
        code_hashes[mod_name] = {
            "path": rel_str,
            "sha256": h,
        }

    # Hash raw files
    prior_prot_file_sha256 = compute_file_sha256(res_prior_prot)
    inputs_file_sha256 = compute_file_sha256(res_prior_inputs)
    gold_file_sha256 = compute_file_sha256(res_prior_gold)
    defaults_file_sha256 = compute_file_sha256(res_defaults)

    fixtures_hashes: dict[str, str] = {
        "prior_protocol_sha256": prior_prot_file_sha256,
        "inputs_file_sha256": inputs_file_sha256,
        "gold_file_sha256": gold_file_sha256,
        "ollama_defaults_sha256": defaults_file_sha256,
    }

    res_env = Path(environment_path)
    if not res_env.is_file():
        res_env = (PROJECT_ROOT / environment_path).resolve()
    if res_env.is_file():
        fixtures_hashes["environment_sha256"] = compute_file_sha256(res_env)

    # 5. Build protocol freeze document
    protocol_data = {
        "schema_version": PROTOCOL_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "design_goal": (
            "Isolate claim-evidence verification across 2 quantization arms (Q8 vs Q3) "
            "with aligned sampling, identical prompts, identical schemas, and no live model changes."
        ),
        "arms": QUANTIZATION_ARMS,
        "case_counts": {
            "historical_controls": 28,
            "counterexamples": 1,
            "total_unique_cases": len(case_ids),
            "runs_per_arm": len(schedule_items) // len(QUANTIZATION_ARMS),
            "total_scheduled_runs": len(schedule_items),
            "arms_count": len(QUANTIZATION_ARMS),
        },
        "defaults_reference": {
            "server_version": defaults_data.get("server_version"),
            "source_url": defaults_data.get("source_url"),
            "defaults": defaults_data.get("defaults"),
            "canonical_sha256": defaults_canonical_hash,
            "file_sha256": defaults_file_sha256,
        },
        "hashes": {
            "inputs_canonical_sha256": inputs_canonical_hash,
            "schedule_canonical_sha256": schedule_canonical_hash,
            "gold_canonical_sha256": gold_canonical_hash,
            "defaults_canonical_sha256": defaults_canonical_hash,
            "fixtures": fixtures_hashes,
            "code": code_hashes,
        },
        "disclosures": {
            "quantization_and_scale_confounds": (
                "Q8 uses qwen3.5:9b-q8_0 (9.0B parameter base) while Q3 uses hf.co/gguf/Qwen3.8-27B-GGUF:Q3_K_M "
                "(27.3B parameter base). Comparison reflects both quantization level and model capacity/weights."
            ),
            "reused_controls": (
                "All 28 historical controls were previously used during pipeline development (20 dev + 8 reserved). "
                "b05 is a known posthoc diagnostic counterexample from baseline stability audit, repeated 3 times per arm."
            ),
            "noncontemporaneous_historical_baseline": (
                "Historical comparison against reference Arm B (qwen3.5:9b Q4_K_M) is non-contemporaneous "
                "and confounded by backend runner updates, template differences, and weight calibration."
            ),
            "hardware_and_cache_confounds": (
                "Fixed arm blocks isolate model swapping using keep_alive=0; placement snapshots represent single "
                "instantaneous points in time and do not claim peak VRAM."
            ),
            "scope_limitation": (
                "Verifier capacity comparison only; does not evaluate search recall, chunking, or full end-to-end generation."
            ),
        },
    }
    assert_finite(protocol_data)
    protocol_canonical_hash = canonical_sha256(protocol_data)

    # 6. Write artifacts byte-for-byte for inputs/gold, canonical JSON for schedule/protocol
    inputs_out = out_dir / "inputs.json"
    gold_out = out_dir / "gold.json"
    schedule_out = out_dir / "schedule.json"
    protocol_out = out_dir / "protocol.json"

    # Byte-for-byte copy of prior frozen inputs and gold
    inputs_out.write_bytes(inputs_bytes)
    gold_out.write_bytes(gold_bytes)

    # Canonical formatted JSON for schedule and protocol
    schedule_out.write_text(
        json.dumps(schedule_data, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    protocol_out.write_text(
        json.dumps(protocol_data, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    return {
        "output_dir": str(out_dir),
        "inputs_path": str(inputs_out),
        "gold_path": str(gold_out),
        "schedule_path": str(schedule_out),
        "protocol_path": str(protocol_out),
        "case_counts": protocol_data["case_counts"],
        "hashes": {
            **protocol_data["hashes"],
            "protocol_canonical_sha256": protocol_canonical_hash,
        },
    }


def run(
    protocol_path: Path | str,
    output_path: Path | str,
    *,
    transport: Any = None,
    prior_protocol_path: Path | str = PRIOR_PROTOCOL_PATH,
    defaults_path: Path | str = DEFAULT_OLLAMA_DEFAULTS_PATH,
) -> dict[str, Any]:
    """Phase 2 execution runner.

    Validates all required pins BEFORE any transport access. Never opens gold.json.
    Executes frozen schedule sequentially over dedicated loopback Ollama transport.
    Captures exact requests, raw responses, validated verdicts, stats, and thinking metrics.
    Performs atomic checkpoints after every single attempt.
    """
    prot_path = validate_project_path(Path(protocol_path))
    if not prot_path.is_file():
        raise FileNotFoundError(f"Protocol file not found: {prot_path}")

    out_path = validate_project_path(Path(output_path))
    if out_path.exists():
        raise FileExistsError(f"Target output file already exists: {out_path}. Refusing to overwrite.")

    # 1. Validate frozen protocol and pins BEFORE any network/transport access
    prot_bytes = prot_path.read_bytes()
    try:
        protocol_data = json.loads(
            prot_bytes.decode("utf-8"),
            parse_constant=_reject_constant,
            parse_float=_parse_float,
        )
    except Exception as exc:
        raise ValueError(f"Malformed JSON in protocol file: {exc}") from exc
    assert_finite(protocol_data)

    if protocol_data.get("schema_version") != PROTOCOL_SCHEMA_VERSION:
        raise ValueError(
            f"Unexpected protocol schema_version: {protocol_data.get('schema_version')!r}, "
            f"expected {PROTOCOL_SCHEMA_VERSION!r}"
        )

    parent_dir = prot_path.parent
    inputs_path = validate_project_path(parent_dir / "inputs.json")
    schedule_path = validate_project_path(parent_dir / "schedule.json")

    if not inputs_path.is_file():
        raise FileNotFoundError(f"Inputs file not found in protocol directory: {inputs_path}")
    if not schedule_path.is_file():
        raise FileNotFoundError(f"Schedule file not found in protocol directory: {schedule_path}")

    # Note: run strictly NEVER opens or loads gold.json!
    inputs_raw = inputs_path.read_bytes()
    try:
        inputs_data = json.loads(
            inputs_raw.decode("utf-8"),
            parse_constant=_reject_constant,
            parse_float=_parse_float,
        )
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
        schedule_data = json.loads(
            schedule_raw.decode("utf-8"),
            parse_constant=_reject_constant,
            parse_float=_parse_float,
        )
    except Exception as exc:
        raise ValueError(f"Malformed JSON in schedule file: {exc}") from exc
    assert_finite(schedule_data)

    if schedule_data.get("schema_version") != SCHEDULE_SCHEMA_VERSION:
        raise ValueError(
            f"Unexpected schedule schema_version: {schedule_data.get('schema_version')!r}, "
            f"expected {SCHEDULE_SCHEMA_VERSION!r}"
        )

    prot_hashes = protocol_data.get("hashes")
    if not isinstance(prot_hashes, dict) or not prot_hashes:
        raise ValueError("protocol.json missing valid 'hashes' dictionary")

    required_hash_keys = {
        "inputs_canonical_sha256",
        "schedule_canonical_sha256",
        "gold_canonical_sha256",
        "defaults_canonical_sha256",
        "fixtures",
        "code",
    }
    missing_hash_keys = required_hash_keys - set(prot_hashes.keys())
    if missing_hash_keys:
        raise ValueError(f"protocol.json hashes missing required keys: {sorted(missing_hash_keys)}")

    expected_inputs_hash = prot_hashes.get("inputs_canonical_sha256")
    expected_schedule_hash = prot_hashes.get("schedule_canonical_sha256")
    expected_gold_hash = prot_hashes.get("gold_canonical_sha256")
    expected_defaults_hash = prot_hashes.get("defaults_canonical_sha256")

    if not is_valid_sha256(expected_inputs_hash):
        raise ValueError(f"Missing or invalid protocol inputs_canonical_sha256 pin: {expected_inputs_hash!r}")
    if not is_valid_sha256(expected_schedule_hash):
        raise ValueError(f"Missing or invalid protocol schedule_canonical_sha256 pin: {expected_schedule_hash!r}")
    if not is_valid_sha256(expected_gold_hash):
        raise ValueError(f"Missing or invalid protocol gold_canonical_sha256 pin: {expected_gold_hash!r}")
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
    defaults_file = validate_project_path(Path(defaults_path))
    if not defaults_file.is_file():
        raise FileNotFoundError(f"Ollama defaults reference file not found at {defaults_file}")
    defaults_raw = defaults_file.read_bytes()
    try:
        defaults_data = json.loads(
            defaults_raw.decode("utf-8"),
            parse_constant=_reject_constant,
            parse_float=_parse_float,
        )
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
    for fix_key in ["prior_protocol_sha256", "inputs_file_sha256", "gold_file_sha256", "ollama_defaults_sha256"]:
        h_val = fixtures_pins.get(fix_key)
        if not is_valid_sha256(h_val):
            raise ValueError(f"Missing or invalid SHA-256 in protocol fixtures for {fix_key!r}: {h_val!r}")

    # Validate actual file byte hashes against fixture pins before transport (never read gold in run)
    act_inputs_file_sha = compute_file_sha256(inputs_path)
    if act_inputs_file_sha != fixtures_pins["inputs_file_sha256"]:
        raise ValueError(
            f"Tampering detected: inputs.json file SHA-256 ({act_inputs_file_sha}) "
            f"does not match fixture pin ({fixtures_pins['inputs_file_sha256']})"
        )

    act_defaults_file_sha = compute_file_sha256(defaults_file)
    if act_defaults_file_sha != fixtures_pins["ollama_defaults_sha256"]:
        raise ValueError(
            f"Tampering detected: defaults file SHA-256 ({act_defaults_file_sha}) "
            f"does not match fixture pin ({fixtures_pins['ollama_defaults_sha256']})"
        )

    res_prior_prot = validate_project_path(Path(prior_protocol_path))
    if not res_prior_prot.is_file():
        raise FileNotFoundError(f"Prior protocol file not found at {res_prior_prot}")
    act_prior_prot_file_sha = compute_file_sha256(res_prior_prot)
    if act_prior_prot_file_sha != fixtures_pins["prior_protocol_sha256"]:
        raise ValueError(
            f"Tampering detected: prior protocol file SHA-256 ({act_prior_prot_file_sha}) "
            f"does not match fixture pin ({fixtures_pins['prior_protocol_sha256']})"
        )

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

        p = validate_project_path(PROJECT_ROOT / rel_path)
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
    if not isinstance(schedule_items, list) or len(schedule_items) != 62:
        raise ValueError(f"Expected 62 items in schedule.json, got {len(schedule_items)}")

    # 2. Instantiate or validate transport
    if transport is None:
        transport = QuantizationOllamaTransport()

    # Query and validate server version against versioned public defaults reference (0.40.11)
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

    # 3. Unload resident models before start
    initial_unloaded = unload_resident_models(transport)

    # 4. Pin model digests and verify installed quantization metadata at start
    tags_data = transport.get_tags()
    tags_models = tags_data.get("models", [])
    pinned_initial_digests: dict[str, str] = {}
    installed_quant_levels: dict[str, str] = {}

    for item in tags_models:
        if isinstance(item, dict):
            name = item.get("name") or item.get("model")
            digest = item.get("digest")
            details = item.get("details", {})
            q_level = details.get("quantization_level") if isinstance(details, dict) else None
            if isinstance(name, str) and isinstance(digest, str) and digest.strip():
                pinned_initial_digests[name] = digest.strip()
            if isinstance(name, str) and isinstance(q_level, str):
                installed_quant_levels[name] = q_level

    # Check that required models exist with exact required quantization levels
    for arm_id, arm_def in QUANTIZATION_ARMS.items():
        m_name = arm_def["model"]
        exp_quant = arm_def["quantization_level"]
        if m_name not in pinned_initial_digests:
            raise RuntimeError(f"Required model {m_name!r} not found in Ollama /api/tags at start of run")
        act_quant = installed_quant_levels.get(m_name)
        if act_quant != exp_quant:
            raise RuntimeError(
                f"Model {m_name!r} quantization mismatch: reported {act_quant!r}, expected {exp_quant!r}"
            )

    # 5. Query /api/show and resolve effective sampling options
    models_show: dict[str, Any] = {}
    models_thinking_supported: dict[str, bool] = {}
    models_show_params: dict[str, dict[str, Any]] = {}
    for m_name in QUANTIZATION_ALLOWED_MODELS:
        try:
            m_show = transport.show(m_name)
            if not isinstance(m_show, dict):
                raise RuntimeError(f"Response from /api/show for {m_name!r} must be a JSON object")
            models_show[m_name] = m_show
            models_thinking_supported[m_name] = check_model_supports_thinking(m_show)
            # Check quantization in show details as well if present
            show_details = m_show.get("details", {})
            if isinstance(show_details, dict) and "quantization_level" in show_details:
                show_q = show_details["quantization_level"]
                for arm_def in QUANTIZATION_ARMS.values():
                    if arm_def["model"] == m_name and show_q != arm_def["quantization_level"]:
                        raise RuntimeError(
                            f"Model {m_name!r} show quantization mismatch: {show_q!r} != {arm_def['quantization_level']!r}"
                        )
            params_raw = m_show.get("parameters")
            # Q3 parameters=null is legitimate empty overrides
            models_show_params[m_name] = parse_show_parameters(params_raw) if params_raw else {}
        except Exception as exc:
            raise RuntimeError(f"Cannot verify model metadata from /api/show for {m_name}: {exc}") from exc

    server_defaults = defaults_data.get("defaults", {})

    # Reference Arm B effective sampling: defaults < qwen3.5:9b show parameters < Arm B options
    reference_arm_b_effective = resolve_effective_sampling(
        server_defaults,
        REFERENCE_ARM_B_SHOW_PARAMETERS,
        REFERENCE_ARM_B_OPTIONS,
    )

    arm_effective_sampling: dict[str, dict[str, Any]] = {}
    arm_sampling_mismatches: dict[str, list[str]] = {}
    blocked_arms: set[str] = set()

    for arm_id, arm_def in QUANTIZATION_ARMS.items():
        m_name = arm_def["model"]
        eff_sampling = resolve_effective_sampling(
            server_defaults,
            models_show_params.get(m_name, {}),
            arm_def["options"],
        )
        arm_effective_sampling[arm_id] = eff_sampling
        is_match, mismatches = validate_sampling_match(reference_arm_b_effective, eff_sampling)
        if not is_match:
            arm_sampling_mismatches[arm_id] = mismatches
            blocked_arms.add(arm_id)

    server_metadata = {
        "server_version": server_ver,
        "tags": tags_data,
        "pinned_initial_digests": pinned_initial_digests,
        "installed_quant_levels": installed_quant_levels,
        "models_show": models_show,
        "thinking_supported": models_thinking_supported,
        "effective_sampling": arm_effective_sampling,
        "reference_arm_b_effective": reference_arm_b_effective,
        "sampling_mismatches": arm_sampling_mismatches if arm_sampling_mismatches else None,
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
    arm_blocks = ["Q8", "Q3"]
    for arm_id in arm_blocks:
        arm_def = QUANTIZATION_ARMS[arm_id]
        arm_model = arm_def["model"]
        arm_think = arm_def["think"]
        arm_options = arm_def["options"]
        arm_quant = arm_def["quantization_level"]

        arm_items = [it for it in schedule_items if it["arm_id"] == arm_id]

        # 1. Check if arm is blocked due to sampling mismatch
        if arm_id in blocked_arms:
            mismatch_err = (
                f"Arm {arm_id} blocked: effective sampling mismatch with Arm B reference: "
                f"{arm_sampling_mismatches.get(arm_id)}"
            )
            arm_placements[arm_id] = {
                "error": mismatch_err,
                "error_type": "effective_sampling_mismatch",
                "attempted": False,
                "effective_sampling": arm_effective_sampling.get(arm_id),
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
                    "wall_time_seconds": None,
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
            available_quants = {
                (m.get("name") or m.get("model")): (
                    m.get("details", {}).get("quantization_level") if isinstance(m.get("details"), dict) else None
                )
                for m in arm_tags.get("models", [])
                if isinstance(m, dict)
            }
        except Exception:
            available_digests = {}
            available_quants = {}

        arm_digest = available_digests.get(arm_model)
        current_quant = available_quants.get(arm_model)

        if not initial_pinned or not arm_digest or arm_digest != initial_pinned or current_quant != arm_quant:
            err_msg = (
                f"Model {arm_model!r} digest/quant changed or unavailable before arm {arm_id} "
                f"(initial_pinned={initial_pinned!r}, current_available={arm_digest!r}, "
                f"expected_quant={arm_quant!r}, current_quant={current_quant!r})"
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
                    "wall_time_seconds": None,
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                }
                attempts.append(unattempted_record)
                save_checkpoint(completed=False)
            continue

        # Model and digest are available:
        # 1. Unload resident models between arms via keep_alive=0
        unload_resident_models(transport)

        # 2. Record initial placement
        ps_placement_initial = None
        try:
            ps_placement_initial = transport.get_ps()
        except Exception as exc:
            ps_placement_initial = {"error": str(exc)}

        arm_thinking_supported = models_thinking_supported.get(arm_model, True)

        arm_placements[arm_id] = {
            "model": arm_model,
            "model_digest": arm_digest,
            "quantization_level": arm_quant,
            "ps_placement_initial": ps_placement_initial,
            "loaded_placement": None,
            "effective_sampling": arm_effective_sampling.get(arm_id),
            "supports_thinking": arm_thinking_supported,
            "attempted": True,
        }

        first_chat_completed = False

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
                    "wall_time_seconds": None,
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
            t_start = time.perf_counter()
            try:
                raw_response = transport.chat(exact_request)
                wall_time_seconds = time.perf_counter() - t_start
            except Exception as exc:
                wall_time_seconds = time.perf_counter() - t_start
                transport_error = exc

            # After the arm's first chat, capture resident placement while loaded (excludes placement time from chat wall time)
            if not first_chat_completed:
                try:
                    loaded_ps = transport.get_ps()
                    arm_placements[arm_id]["loaded_placement"] = loaded_ps
                except Exception as exc:
                    arm_placements[arm_id]["loaded_placement"] = {"error": str(exc)}
                first_chat_completed = True

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
                    "wall_time_seconds": round(wall_time_seconds, 6),
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

                # 4. Mandatory non-bool non-negative integer prompt_eval_count and eval_count
                prompt_eval_count = raw_response.get("prompt_eval_count")
                if (
                    prompt_eval_count is None
                    or isinstance(prompt_eval_count, bool)
                    or not isinstance(prompt_eval_count, int)
                    or prompt_eval_count < 0
                ):
                    raise ValueError(
                        f"Missing or invalid prompt_eval_count: {prompt_eval_count!r}. "
                        "Must be a non-boolean non-negative integer."
                    )

                eval_count = raw_response.get("eval_count")
                if (
                    eval_count is None
                    or isinstance(eval_count, bool)
                    or not isinstance(eval_count, int)
                    or eval_count < 0
                ):
                    raise ValueError(
                        f"Missing or invalid eval_count: {eval_count!r}. "
                        "Must be a non-boolean non-negative integer."
                    )

                # Context budget constraint: prompt + 4096 + 256 <= 8192
                num_predict = arm_options["num_predict"]
                if prompt_eval_count + num_predict + CONTEXT_SAFETY_MARGIN > MAX_CONTEXT_WINDOW:
                    raise ValueError(
                        f"Context budget exceeded: prompt_eval_count ({prompt_eval_count}) + "
                        f"num_predict ({num_predict}) + {CONTEXT_SAFETY_MARGIN} = "
                        f"{prompt_eval_count + num_predict + CONTEXT_SAFETY_MARGIN} > {MAX_CONTEXT_WINDOW}"
                    )

                # 5. Extract thinking length without accepting unexpected thinking
                thinking_str = None
                if isinstance(raw_response.get("message"), dict) and "thinking" in raw_response["message"]:
                    t_val = raw_response["message"]["thinking"]
                    if isinstance(t_val, str):
                        thinking_str = t_val
                elif "thinking" in raw_response and isinstance(raw_response["thinking"], str):
                    thinking_str = raw_response["thinking"]

                if thinking_str is not None and len(thinking_str.strip()) > 0:
                    raise ValueError(
                        f"Technical protocol violation: unexpected thinking content ({len(thinking_str)} chars) "
                        "returned when think=false"
                    )

                # 6. Extract and parse message content JSON
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

                # 7. Normalize and validate payload against frozen claims and evidence
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

            # Thinking metrics
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
                "wall_time_seconds": round(wall_time_seconds, 6),
                "completed_at": datetime.now(timezone.utc).isoformat(),
            }
            attempts.append(attempt_record)
            save_checkpoint(completed=False)

    # 5. At end of run, unload generator models
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


def score(
    results_path: Path | str,
    gold_path: Path | str,
    output_path: Path | str | None = None,
) -> dict[str, Any]:
    """Phase 3 scoring evaluator.

    Validates finite JSON/schema, gold hash/pin against original protocol,
    verifies complete expected schedule roster of exactly 62 attempts (refusing duplicate or unknown events),
    and strictly separates 28 primary historical controls from b05 correlated repeats.
    Operates strictly offline with zero network and zero live model calls.
    """
    res_path = validate_project_path(Path(results_path))
    if not res_path.is_file():
        raise FileNotFoundError(f"Results file not found: {res_path}")

    g_path = validate_project_path(Path(gold_path))
    if not g_path.is_file():
        raise FileNotFoundError(f"Gold file not found: {g_path}")

    if output_path is not None:
        out_p = validate_project_path(Path(output_path))
        if out_p.exists():
            raise FileExistsError(f"Target output score report file already exists: {out_p}. Refusing to overwrite.")

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

    # Validate gold file byte hash against protocol fixture pin
    expected_gold_file_sha = protocol_data.get("hashes", {}).get("fixtures", {}).get("gold_file_sha256")
    if not is_valid_sha256(expected_gold_file_sha):
        raise ValueError(f"Missing or invalid gold_file_sha256 pin in protocol fixtures: {expected_gold_file_sha!r}")
    actual_gold_file_sha = compute_file_sha256(g_path)
    if actual_gold_file_sha != expected_gold_file_sha:
        raise ValueError(
            f"Tampering detected: gold.json file SHA-256 ({actual_gold_file_sha}) "
            f"does not match protocol fixture pin ({expected_gold_file_sha})"
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
    TOTAL_PLANNED_RUNS = 62
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
        if arm_id not in QUANTIZATION_ARMS:
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

    # Roster completeness check: exactly 62 actual attempted calls
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
        if output_path is not None:
            Path(output_path).write_text(
                json.dumps(incomplete_report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                encoding="utf-8",
            )
        return incomplete_report

    # Completed run: process each arm (Q8 and Q3)
    arm_summaries: dict[str, Any] = {}
    for arm_id in ["Q8", "Q3"]:
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
        wall_durations_s = []
        load_durations_s = []
        in_tokens = []
        out_tokens = []
        thinking_lens = []
        truncation_count = 0

        for a in arm_attempts:
            # Wall time includes all actual attempted calls (even with technical errors)
            w_time = a.get("wall_time_seconds")
            if isinstance(w_time, (int, float)) and not math.isnan(w_time) and not math.isinf(w_time) and w_time >= 0:
                wall_durations_s.append(float(w_time))

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
            "model": QUANTIZATION_ARMS[arm_id]["model"],
            "think": QUANTIZATION_ARMS[arm_id]["think"],
            "quantization_level": QUANTIZATION_ARMS[arm_id]["quantization_level"],
            "options": QUANTIZATION_ARMS[arm_id]["options"],
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
                "server_total_duration_seconds": compute_numeric_summary(durations_s),
                "measured_chat_wall_time_seconds": compute_numeric_summary(wall_durations_s),
                "load_duration_seconds": compute_numeric_summary(load_durations_s),
                "input_tokens": compute_numeric_summary(in_tokens),
                "output_tokens": compute_numeric_summary(out_tokens),
                "thinking_characters": thinking_summary,
                "truncation_count": truncation_count,
            },
            "placement_snapshot": placement_snapshot,
        }

    # Cross-arm contrast
    arm_q8 = arm_summaries["Q8"]
    arm_q3 = arm_summaries["Q3"]

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
        "quantization_and_scale_contrast_Q3_minus_Q8": {
            **_calc_comparison(arm_q3, arm_q8),
            "description": (
                "Contrast between Q3_K_M (hf.co/gguf/Qwen3.8-27B-GGUF:Q3_K_M, 27.3B) and Q8_0 (qwen3.5:9b-q8_0, 9.0B). "
                "Confounds quantization level with parameter size (27.3B vs 9B), architecture, and calibration imatrix."
            ),
        },
        "historical_reference_contrast_ArmB_Q4": {
            "description": (
                "Historical reference Arm B (qwen3.5:9b Q4_K_M from verifier_capacity_2026-10-26) achieved 22/28 (0.7857). "
                "This comparison is explicitly non-contemporaneous and confounded by Ollama runtime updates, template differences, "
                "and distinct quantization matrices; no pure quantization causality can be asserted."
            ),
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
        "quantization_and_scale_confounding": (
            "Q8 (9.0B Q8_0) and Q3 (27.3B Q3_K_M) differ simultaneously in quantization precision, parameter scale, "
            "base weights, and GGUF imatrix calibration. Observed accuracy or latency differences must not be attributed "
            "purely to quantization."
        ),
        "noncontemporaneous_baseline": (
            "Comparisons with historical Arm B (Q4) from previous capacity experiments are non-contemporaneous and "
            "subject to runner, driver, and environment drift."
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

    if output_path is not None:
        Path(output_path).write_text(
            json.dumps(score_report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )

    return score_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verifier quantization comparison: Phase 1 preparation, Phase 2 execution, and Phase 3 scoring runner.",
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
            f"Status: {res['completion_status']}, "
            f"Attempted: {res['total_attempted']}/{res['total_scheduled']}, "
            f"Successful: {res['total_successful']}, Errors: {res['total_technical_errors']}"
        )
        return 0
    elif args.command == "score":
        res = score(args.results, args.gold, args.output)
        print(f"Scoring complete. Score report written to {args.output}")
        print(f"Evaluation status: {res.get('evaluation_status')}, Complete: {res.get('is_complete')}")
        return 0
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
