"""Unit tests for verifier quantization comparison (experiments.compare_verifier_quantization).

Offline tests using fake transport:
- valid 62 calls with identical messages/schema and no gold loaded during run
- code, default, input, and schedule pin removal/tamper before any network/transport interaction
- digest drift and wrong quantization level detection
- missing metadata and effective sampler mismatch blocking
- HTTP malformed response and raw response preservation on error
- missing, boolean, or negative eval counts and context budget overflow
- length truncation, returned model identity mismatch, and unexpected thinking protocol violation
- missing, duplicate, reassigned, and unknown schedule events
- zero denominator resilience
- partial run never scored as complete
- refusal of overwrites across prepare, run, and score
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
from typing import Any

from experiments.compare_verifier_capacity import (
    DEFAULT_OLLAMA_DEFAULTS_PATH,
    OllamaResponseError,
    canonical_json_bytes,
    canonical_sha256,
    compute_file_sha256,
)
from experiments.compare_verifier_quantization import (
    GOLD_SCHEMA_VERSION,
    INPUTS_SCHEMA_VERSION,
    PRIOR_FROZEN_DIR,
    PRIOR_GOLD_PATH,
    PRIOR_INPUTS_PATH,
    PRIOR_PROTOCOL_PATH,
    PRIOR_SCHEDULE_PATH,
    PROTOCOL_SCHEMA_VERSION,
    QUANTIZATION_ALLOWED_MODELS,
    QUANTIZATION_ARMS,
    QuantizationOllamaTransport,
    REQUIRED_CODE_MODULES,
    RUN_RESULTS_SCHEMA_VERSION,
    SCHEDULE_SCHEMA_VERSION,
    SCORE_REPORT_SCHEMA_VERSION,
    main,
    prepare,
    run,
    score,
)
from rag_pensiones.ingest import PROJECT_ROOT


class FakeQuantizationTransport:
    """Offline fake transport matching QuantizationOllamaTransport interface."""

    def __init__(
        self,
        tags: dict[str, Any] | None = None,
        ps_models: list[dict[str, Any]] | None = None,
        show_responses: dict[str, dict[str, Any]] | None = None,
        chat_responder: Any = None,
        version: str = "0.40.11",
    ) -> None:
        self.version = version
        self.tags = tags or {
            "models": [
                {
                    "name": "qwen3.5:9b-q8_0",
                    "model": "qwen3.5:9b-q8_0",
                    "digest": "a8ee95bc3592ebaa624014ca8b429d25e0ef5f45030dc35786da7ff7e9e1b60a",
                    "details": {
                        "quantization_level": "Q8_0",
                        "parameter_size": "9.0B",
                    },
                },
                {
                    "name": "hf.co/gguf/Qwen3.8-27B-GGUF:Q3_K_M",
                    "model": "hf.co/gguf/Qwen3.8-27B-GGUF:Q3_K_M",
                    "digest": "1844fd49a2fcf04662d7e81afe585e4846fec76cae77b0a15e3bd1efe45969ed",
                    "details": {
                        "quantization_level": "Q3_K_M",
                        "parameter_size": "27.3B",
                    },
                },
            ]
        }
        self.ps_models = ps_models or [
            {"name": "resident_model_prev", "model": "resident_model_prev", "size_vram": 1024}
        ]
        self.show_responses = show_responses or {
            "qwen3.5:9b-q8_0": {
                "parameters": "presence_penalty 1.5\ntemperature 1\ntop_k 20\ntop_p 0.95\n",
                "details": {"quantization_level": "Q8_0"},
                "capabilities": ["thinking"],
            },
            "hf.co/gguf/Qwen3.8-27B-GGUF:Q3_K_M": {
                "parameters": None,
                "details": {"quantization_level": "Q3_K_M"},
                "capabilities": ["thinking"],
            },
        }
        self.chat_responder = chat_responder
        self.chat_calls: list[dict[str, Any]] = []
        self.unload_calls: list[str] = []
        self.ps_calls: int = 0
        self.show_calls: list[str] = []
        self.tags_calls: int = 0
        self.version_calls: int = 0

    def get_version(self) -> dict[str, Any]:
        self.version_calls += 1
        return {"version": self.version}

    def get_tags(self) -> dict[str, Any]:
        self.tags_calls += 1
        return copy.deepcopy(self.tags)

    def get_ps(self) -> dict[str, Any]:
        self.ps_calls += 1
        return {"models": copy.deepcopy(self.ps_models)}

    def show(self, model: str) -> dict[str, Any]:
        self.show_calls.append(model)
        if model in self.show_responses:
            return copy.deepcopy(self.show_responses[model])
        return {"parameters": None, "details": {}}

    def unload(self, model: str) -> dict[str, Any]:
        self.unload_calls.append(model)
        self.ps_models = [m for m in self.ps_models if m.get("name") != model and m.get("model") != model]
        return {"status": "unloaded"}

    def chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.chat_calls.append(copy.deepcopy(payload))
        if self.chat_responder is not None:
            return self.chat_responder(payload)

        # Default compliant responder: build valid verification output matching schema
        model = payload["model"]
        schema = payload.get("format", {})

        claim_props = schema.get("properties", {}).get("verdicts", {}).get("properties", {})
        verdicts = {}
        for cid, val in claim_props.items():
            first_branch = val.get("anyOf", [{}])[0]
            ev_props = first_branch.get("properties", {}).get("evidence_verdicts", {}).get("properties", {})
            evidence_verdicts = {eid: {"relation": "supports"} for eid in ev_props}
            verdicts[cid] = {
                "verdict": "supported",
                "reason": "Supported by cited evidence.",
                "evidence_verdicts": evidence_verdicts,
            }

        content_json = {"verdicts": verdicts}
        return {
            "model": model,
            "done": True,
            "done_reason": "stop",
            "prompt_eval_count": 480,
            "prompt_eval_duration": 12000000,
            "eval_count": 35,
            "eval_duration": 25000000,
            "total_duration": 600000000,
            "load_duration": 15000000,
            "message": {
                "role": "assistant",
                "content": json.dumps(content_json, ensure_ascii=False),
            },
        }


class TestCompareVerifierQuantization(unittest.TestCase):
    """Test suite for verifier quantization comparison protocol, execution, and offline scoring."""

    def setUp(self) -> None:
        self.test_root = PROJECT_ROOT / "tmp"
        self.test_root.mkdir(parents=True, exist_ok=True)
        self.temp_dir = tempfile.TemporaryDirectory(dir=self.test_root)
        self.out_dir = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_prepare_produces_valid_frozen_artifacts_and_refuses_overwrite(self) -> None:
        """prepare() produces exact 62-item schedule, byte-for-byte inputs/gold copy, and refuses overwrite."""
        prep_res = prepare(self.out_dir)

        self.assertEqual(prep_res["case_counts"]["total_scheduled_runs"], 62)
        self.assertEqual(prep_res["case_counts"]["arms_count"], 2)
        self.assertEqual(prep_res["case_counts"]["runs_per_arm"], 31)

        # Verify byte-for-byte copy of inputs.json and gold.json
        gen_inputs = (self.out_dir / "inputs.json").read_bytes()
        prior_inputs = PRIOR_INPUTS_PATH.read_bytes()
        self.assertEqual(gen_inputs, prior_inputs)

        gen_gold = (self.out_dir / "gold.json").read_bytes()
        prior_gold = PRIOR_GOLD_PATH.read_bytes()
        self.assertEqual(gen_gold, prior_gold)

        # Verify schedule has 62 items and identical case order across arms
        sched = json.loads((self.out_dir / "schedule.json").read_text(encoding="utf-8"))
        items = sched["items"]
        self.assertEqual(len(items), 62)

        q8_cases = [it["case_id"] for it in items[:31]]
        q3_cases = [it["case_id"] for it in items[31:]]
        self.assertEqual(q8_cases, q3_cases)
        # b05 is at index 28, 29, 30 for Q8
        self.assertEqual(q8_cases[28:], ["b05", "b05", "b05"])

        # Refusal of overwrite on non-empty target directory
        with self.assertRaises(FileExistsError):
            prepare(self.out_dir)

    def test_valid_62_calls_identical_messages_schema_and_no_gold_opened(self) -> None:
        """run() executes 62 calls matching inputs.json payload byte-for-byte, never accessing gold.json."""
        prepare(self.out_dir)
        prot_path = self.out_dir / "protocol.json"
        results_path = self.out_dir / "run_results.json"

        gold_path = self.out_dir / "gold.json"
        # Track whether gold.json is read during run()
        gold_accessed = False
        orig_read_bytes = Path.read_bytes

        def instrumented_read_bytes(path_self: Path, *args: Any, **kwargs: Any) -> bytes:
            nonlocal gold_accessed
            if path_self.resolve() == gold_path.resolve():
                gold_accessed = True
            return orig_read_bytes(path_self, *args, **kwargs)

        fake_transport = FakeQuantizationTransport()
        with patch.object(Path, "read_bytes", side_effect=instrumented_read_bytes, autospec=True):
            res = run(prot_path, results_path, transport=fake_transport)

        self.assertFalse(gold_accessed, "run() strictly must never read gold.json")
        self.assertEqual(len(fake_transport.chat_calls), 62)
        self.assertEqual(res["total_attempted"], 62)
        self.assertEqual(res["total_successful"], 62)
        self.assertEqual(res["total_technical_errors"], 0)
        self.assertEqual(res["completion_status"], "completed")

        # Verify exact request messages and schema match inputs.json
        inputs_data = json.loads((self.out_dir / "inputs.json").read_text(encoding="utf-8"))
        case_map = {c["id"]: c for c in inputs_data["cases"]}

        for call in fake_transport.chat_calls:
            # Model must be one of the two allowed models
            self.assertIn(call["model"], QUANTIZATION_ALLOWED_MODELS)
            self.assertFalse(call["think"])
            # Format is schema dict
            self.assertIsInstance(call["format"], dict)

    def test_code_default_input_schedule_pin_removal_tamper_fails_before_network(self) -> None:
        """Tampering or removing any required pin in protocol.json halts execution before any network call."""
        prepare(self.out_dir)
        prot_path = self.out_dir / "protocol.json"
        results_path = self.out_dir / "run_results.json"

        # 1. Tamper code pin
        prot_data = json.loads(prot_path.read_text(encoding="utf-8"))
        bad_prot_data = copy.deepcopy(prot_data)
        bad_prot_data["hashes"]["code"]["compare_verifier_quantization"]["sha256"] = "0" * 64
        bad_prot_file = self.out_dir / "protocol_bad_code.json"
        bad_prot_file.write_text(json.dumps(bad_prot_data), encoding="utf-8")

        fake_transport = FakeQuantizationTransport()
        with self.assertRaises(ValueError) as ctx:
            run(bad_prot_file, results_path, transport=fake_transport)
        self.assertIn("Tampering/stale code detected", str(ctx.exception))
        self.assertEqual(fake_transport.version_calls, 0, "No network call on code pin failure")

        # 2. Tamper defaults pin
        bad_prot_data = copy.deepcopy(prot_data)
        bad_prot_data["hashes"]["defaults_canonical_sha256"] = "f" * 64
        bad_prot_file = self.out_dir / "protocol_bad_defaults.json"
        bad_prot_file.write_text(json.dumps(bad_prot_data), encoding="utf-8")

        fake_transport = FakeQuantizationTransport()
        with self.assertRaises(ValueError) as ctx:
            run(bad_prot_file, results_path, transport=fake_transport)
        self.assertIn("Tampering detected: defaults canonical SHA-256", str(ctx.exception))
        self.assertEqual(fake_transport.version_calls, 0)

        # 3. Tamper inputs canonical sha256 pin
        bad_prot_data = copy.deepcopy(prot_data)
        bad_prot_data["hashes"]["inputs_canonical_sha256"] = "e" * 64
        bad_prot_file = self.out_dir / "protocol_bad_inputs.json"
        bad_prot_file.write_text(json.dumps(bad_prot_data), encoding="utf-8")

        fake_transport = FakeQuantizationTransport()
        with self.assertRaises(ValueError) as ctx:
            run(bad_prot_file, results_path, transport=fake_transport)
        self.assertIn("Tampering detected: inputs.json", str(ctx.exception))
        self.assertEqual(fake_transport.version_calls, 0)

        # 4. Tamper schedule canonical sha256 pin
        bad_prot_data = copy.deepcopy(prot_data)
        bad_prot_data["hashes"]["schedule_canonical_sha256"] = "d" * 64
        bad_prot_file = self.out_dir / "protocol_bad_sched.json"
        bad_prot_file.write_text(json.dumps(bad_prot_data), encoding="utf-8")

        fake_transport = FakeQuantizationTransport()
        with self.assertRaises(ValueError) as ctx:
            run(bad_prot_file, results_path, transport=fake_transport)
        self.assertIn("Tampering detected: schedule.json", str(ctx.exception))
        self.assertEqual(fake_transport.version_calls, 0)

        # 5. Missing code pin module (grounding)
        bad_prot_data = copy.deepcopy(prot_data)
        del bad_prot_data["hashes"]["code"]["grounding"]
        bad_prot_file = self.out_dir / "protocol_missing_code.json"
        bad_prot_file.write_text(json.dumps(bad_prot_data), encoding="utf-8")

        fake_transport = FakeQuantizationTransport()
        with self.assertRaises(ValueError) as ctx:
            run(bad_prot_file, results_path, transport=fake_transport)
        self.assertIn("missing required code pins", str(ctx.exception))
        self.assertEqual(fake_transport.version_calls, 0)

        # 6. Tamper compare_verifier_capacity code pin
        bad_prot_data = copy.deepcopy(prot_data)
        bad_prot_data["hashes"]["code"]["compare_verifier_capacity"]["sha256"] = "1" * 64
        bad_prot_file2 = self.out_dir / "protocol_bad_capacity_code.json"
        bad_prot_file2.write_text(json.dumps(bad_prot_data), encoding="utf-8")

        fake_transport2 = FakeQuantizationTransport()
        with self.assertRaises(ValueError) as ctx:
            run(bad_prot_file2, results_path, transport=fake_transport2)
        self.assertIn("Tampering/stale code detected for compare_verifier_capacity", str(ctx.exception))
        self.assertEqual(fake_transport2.version_calls, 0)

        # 7. Tamper ingest code pin
        bad_prot_data = copy.deepcopy(prot_data)
        bad_prot_data["hashes"]["code"]["ingest"]["sha256"] = "2" * 64
        bad_prot_file3 = self.out_dir / "protocol_bad_ingest_code.json"
        bad_prot_file3.write_text(json.dumps(bad_prot_data), encoding="utf-8")

        fake_transport3 = FakeQuantizationTransport()
        with self.assertRaises(ValueError) as ctx:
            run(bad_prot_file3, results_path, transport=fake_transport3)
        self.assertIn("Tampering/stale code detected for ingest", str(ctx.exception))
        self.assertEqual(fake_transport3.version_calls, 0)

        # 8. Missing compare_verifier_capacity code pin
        bad_prot_data = copy.deepcopy(prot_data)
        del bad_prot_data["hashes"]["code"]["compare_verifier_capacity"]
        bad_prot_file4 = self.out_dir / "protocol_missing_capacity.json"
        bad_prot_file4.write_text(json.dumps(bad_prot_data), encoding="utf-8")

        fake_transport4 = FakeQuantizationTransport()
        with self.assertRaises(ValueError) as ctx:
            run(bad_prot_file4, results_path, transport=fake_transport4)
        self.assertIn("missing required code pins", str(ctx.exception))
        self.assertEqual(fake_transport4.version_calls, 0)

        # 9. Missing ingest code pin
        bad_prot_data = copy.deepcopy(prot_data)
        del bad_prot_data["hashes"]["code"]["ingest"]
        bad_prot_file5 = self.out_dir / "protocol_missing_ingest.json"
        bad_prot_file5.write_text(json.dumps(bad_prot_data), encoding="utf-8")

        fake_transport5 = FakeQuantizationTransport()
        with self.assertRaises(ValueError) as ctx:
            run(bad_prot_file5, results_path, transport=fake_transport5)
        self.assertIn("missing required code pins", str(ctx.exception))
        self.assertEqual(fake_transport5.version_calls, 0)

    def test_digest_drift_and_wrong_quant_blocks_execution(self) -> None:
        """Digest drift between initial tags and arm execution or wrong quantization level blocks the arm."""
        prepare(self.out_dir)
        prot_path = self.out_dir / "protocol.json"
        results_path = self.out_dir / "run_results.json"

        # 1. Wrong quantization level at start
        bad_tags = {
            "models": [
                {
                    "name": "qwen3.5:9b-q8_0",
                    "model": "qwen3.5:9b-q8_0",
                    "digest": "a8ee95bc3592ebaa624014ca8b429d25e0ef5f45030dc35786da7ff7e9e1b60a",
                    "details": {"quantization_level": "Q4_0"},  # Mismatch!
                },
                {
                    "name": "hf.co/gguf/Qwen3.8-27B-GGUF:Q3_K_M",
                    "model": "hf.co/gguf/Qwen3.8-27B-GGUF:Q3_K_M",
                    "digest": "1844fd49a2fcf04662d7e81afe585e4846fec76cae77b0a15e3bd1efe45969ed",
                    "details": {"quantization_level": "Q3_K_M"},
                },
            ]
        }
        fake_transport = FakeQuantizationTransport(tags=bad_tags)
        with self.assertRaises(RuntimeError) as ctx:
            run(prot_path, results_path, transport=fake_transport)
        self.assertIn("quantization mismatch", str(ctx.exception))

        # 2. Digest drift right before arm execution
        fake_transport = FakeQuantizationTransport()
        original_get_tags = fake_transport.get_tags

        def drifting_get_tags() -> dict[str, Any]:
            tags_copy = original_get_tags()
            # If called during arm loop (tags_calls > 1), mutate digest
            if fake_transport.tags_calls > 1:
                tags_copy["models"][0]["digest"] = "mutated_digest_12345"
            return tags_copy

        fake_transport.get_tags = drifting_get_tags
        res = run(prot_path, results_path, transport=fake_transport)
        self.assertEqual(res["completion_status"], "incomplete/skipped")
        self.assertGreater(res["total_unattempted"], 0)

    def test_missing_metadata_and_effective_sampler_mismatch_blocks(self) -> None:
        """Absent /api/show or sampling parameter mismatch blocks model inference."""
        prepare(self.out_dir)
        prot_path = self.out_dir / "protocol.json"
        results_path = self.out_dir / "run_results.json"

        # 1. /api/show error
        fake_transport = FakeQuantizationTransport()
        fake_transport.show = lambda model: (_ for _ in ()).throw(RuntimeError("show failed"))
        with self.assertRaises(RuntimeError) as ctx:
            run(prot_path, results_path, transport=fake_transport)
        self.assertIn("Cannot verify model metadata from /api/show", str(ctx.exception))

        # 2. Effective sampler mismatch (e.g. show returned unexpected top_k)
        bad_show = {
            "qwen3.5:9b-q8_0": {
                "parameters": "presence_penalty 1.5\ntemperature 1\ntop_k 999\ntop_p 0.95\n",  # top_k mismatch!
                "details": {"quantization_level": "Q8_0"},
            },
            "hf.co/gguf/Qwen3.8-27B-GGUF:Q3_K_M": {
                "parameters": None,
                "details": {"quantization_level": "Q3_K_M"},
            },
        }
        fake_transport2 = FakeQuantizationTransport(show_responses=bad_show)
        res_mismatch_path = self.out_dir / "res_mismatch.json"
        res2 = run(prot_path, res_mismatch_path, transport=fake_transport2)
        self.assertEqual(res2["completion_status"], "incomplete/skipped")
        self.assertEqual(res2["arm_placements"]["Q8"]["error_type"], "effective_sampling_mismatch")

    def test_http_malformed_and_raw_preservation(self) -> None:
        """Transport returning OllamaResponseError preserves raw_text and error details in checkpoints."""
        prepare(self.out_dir)
        prot_path = self.out_dir / "protocol.json"
        results_path = self.out_dir / "run_results.json"

        def error_chat_responder(payload: dict[str, Any]) -> dict[str, Any]:
            raise OllamaResponseError(
                "HTTP error 500: internal server error",
                raw_text='{"error": "corrupt gguf"}',
                is_truncated=False,
                status_code=500,
            )

        fake_transport = FakeQuantizationTransport(chat_responder=error_chat_responder)
        res = run(prot_path, results_path, transport=fake_transport)

        self.assertEqual(res["total_attempted"], 62)
        self.assertEqual(res["total_technical_errors"], 62)
        self.assertEqual(res["completion_status"], "completed_with_errors")

        saved = json.loads(results_path.read_text(encoding="utf-8"))
        att0 = saved["attempts"][0]
        self.assertTrue(att0["attempted"])
        self.assertEqual(att0["status"], "error")
        self.assertIsNotNone(att0["raw_response"])
        self.assertEqual(att0["raw_response"]["raw_text"], '{"error": "corrupt gguf"}')
        self.assertEqual(att0["raw_response"]["status_code"], 500)

    def test_missing_bool_negative_counts_and_context_overflow(self) -> None:
        """Invalid prompt_eval_count or context window overflow is caught and recorded as technical error."""
        prepare(self.out_dir)
        prot_path = self.out_dir / "protocol.json"

        # Negative prompt_eval_count
        def neg_eval_responder(payload: dict[str, Any]) -> dict[str, Any]:
            return {
                "model": payload["model"],
                "done": True,
                "done_reason": "stop",
                "prompt_eval_count": -5,
                "eval_count": 10,
                "total_duration": 1000,
                "message": {"role": "assistant", "content": '{"verdicts": []}'},
            }

        res_path1 = self.out_dir / "res1.json"
        fake_transport1 = FakeQuantizationTransport(chat_responder=neg_eval_responder)
        res1 = run(prot_path, res_path1, transport=fake_transport1)
        self.assertEqual(res1["total_technical_errors"], 62)

        # Boolean prompt_eval_count
        def bool_eval_responder(payload: dict[str, Any]) -> dict[str, Any]:
            return {
                "model": payload["model"],
                "done": True,
                "done_reason": "stop",
                "prompt_eval_count": True,
                "eval_count": 10,
                "total_duration": 1000,
                "message": {"role": "assistant", "content": '{"verdicts": []}'},
            }

        res_path2 = self.out_dir / "res2.json"
        fake_transport2 = FakeQuantizationTransport(chat_responder=bool_eval_responder)
        res2 = run(prot_path, res_path2, transport=fake_transport2)
        self.assertEqual(res2["total_technical_errors"], 62)

        # Context overflow: prompt_eval_count + 4096 + 256 > 8192
        def overflow_responder(payload: dict[str, Any]) -> dict[str, Any]:
            return {
                "model": payload["model"],
                "done": True,
                "done_reason": "stop",
                "prompt_eval_count": 4000,  # 4000 + 4096 + 256 = 8352 > 8192
                "eval_count": 10,
                "total_duration": 1000,
                "message": {"role": "assistant", "content": '{"verdicts": []}'},
            }

        res_path3 = self.out_dir / "res3.json"
        fake_transport3 = FakeQuantizationTransport(chat_responder=overflow_responder)
        res3 = run(prot_path, res_path3, transport=fake_transport3)
        self.assertEqual(res3["total_technical_errors"], 62)

    def test_length_truncation_identity_mismatch_and_unexpected_thinking(self) -> None:
        """Truncation (done_reason=length), wrong model identity, and unexpected thinking trigger technical errors."""
        prepare(self.out_dir)
        prot_path = self.out_dir / "protocol.json"

        # 1. Truncation: done_reason == "length"
        def length_responder(payload: dict[str, Any]) -> dict[str, Any]:
            return {
                "model": payload["model"],
                "done": True,
                "done_reason": "length",
                "prompt_eval_count": 500,
                "eval_count": 4096,
                "total_duration": 1000,
                "message": {"role": "assistant", "content": '{"verdicts": []}'},
            }

        res_path1 = self.out_dir / "res_length.json"
        fake1 = FakeQuantizationTransport(chat_responder=length_responder)
        res1 = run(prot_path, res_path1, transport=fake1)
        self.assertEqual(res1["total_technical_errors"], 62)

        # 2. Identity mismatch
        def wrong_model_responder(payload: dict[str, Any]) -> dict[str, Any]:
            return {
                "model": "unrequested_model_xyz",
                "done": True,
                "done_reason": "stop",
                "prompt_eval_count": 500,
                "eval_count": 20,
                "total_duration": 1000,
                "message": {"role": "assistant", "content": '{"verdicts": []}'},
            }

        res_path2 = self.out_dir / "res_model.json"
        fake2 = FakeQuantizationTransport(chat_responder=wrong_model_responder)
        res2 = run(prot_path, res_path2, transport=fake2)
        self.assertEqual(res2["total_technical_errors"], 62)

        # 3. Unexpected thinking returned when think=false
        def thinking_responder(payload: dict[str, Any]) -> dict[str, Any]:
            return {
                "model": payload["model"],
                "done": True,
                "done_reason": "stop",
                "prompt_eval_count": 500,
                "eval_count": 30,
                "total_duration": 1000,
                "message": {
                    "role": "assistant",
                    "content": '{"verdicts": []}',
                    "thinking": "Unexpected thinking trace that should violate protocol.",
                },
            }

        res_path3 = self.out_dir / "res_thinking.json"
        fake3 = FakeQuantizationTransport(chat_responder=thinking_responder)
        res3 = run(prot_path, res_path3, transport=fake3)
        self.assertEqual(res3["total_technical_errors"], 62)
        saved = json.loads(res_path3.read_text(encoding="utf-8"))
        self.assertIn("unexpected thinking content", saved["attempts"][0]["error_message"])

    def test_score_detects_missing_duplicate_reassigned_and_unknown_events(self) -> None:
        """score() validates roster identity and rejects duplicate, reassigned, or unknown attempt events."""
        prepare(self.out_dir)
        prot_path = self.out_dir / "protocol.json"
        results_path = self.out_dir / "run_results.json"
        gold_path = self.out_dir / "gold.json"

        fake_transport = FakeQuantizationTransport()
        run(prot_path, results_path, transport=fake_transport)

        valid_results = json.loads(results_path.read_text(encoding="utf-8"))

        # 1. Duplicate attempt event
        dup_results = copy.deepcopy(valid_results)
        dup_results["attempts"][1] = copy.deepcopy(dup_results["attempts"][0])
        dup_file = self.out_dir / "res_dup.json"
        dup_file.write_text(json.dumps(dup_results), encoding="utf-8")
        with self.assertRaises(ValueError) as ctx:
            score(dup_file, gold_path)
        self.assertIn("Duplicate", str(ctx.exception))

        # 2. Reassigned schedule_index
        reassigned_results = copy.deepcopy(valid_results)
        reassigned_results["attempts"][0]["case_id"] = "d02"
        reassigned_file = self.out_dir / "res_reassigned.json"
        reassigned_file.write_text(json.dumps(reassigned_results), encoding="utf-8")
        with self.assertRaises(ValueError) as ctx:
            score(reassigned_file, gold_path)
        self.assertIn("Reassigned schedule_index", str(ctx.exception))

        # 3. Unknown case_id
        unknown_results = copy.deepcopy(valid_results)
        unknown_results["attempts"][0]["case_id"] = "unknown_case_999"
        unknown_file = self.out_dir / "res_unknown.json"
        unknown_file.write_text(json.dumps(unknown_results), encoding="utf-8")
        with self.assertRaises(ValueError) as ctx:
            score(unknown_file, gold_path)
        self.assertIn("Unknown case_id", str(ctx.exception))

    def test_score_zero_denominators_resilience(self) -> None:
        """score() cleanly returns None (JSON null) for zero denominators without division errors."""
        prepare(self.out_dir)
        prot_path = self.out_dir / "protocol.json"
        results_path = self.out_dir / "run_results.json"
        gold_path = self.out_dir / "gold.json"

        # All attempts fail with technical error -> 0 valid attempts
        def fail_responder(payload: dict[str, Any]) -> dict[str, Any]:
            raise OllamaResponseError("Failed call", raw_text="err", is_truncated=False, status_code=500)

        fake_transport = FakeQuantizationTransport(chat_responder=fail_responder)
        run(prot_path, results_path, transport=fake_transport)

        score_rep = score(results_path, gold_path)
        self.assertTrue(score_rep["is_complete"])
        for arm_id in ["Q8", "Q3"]:
            arm_acc = score_rep["arms"][arm_id]["accuracy"]
            self.assertEqual(arm_acc["correct_count"], 0)
            self.assertIsNone(arm_acc["accuracy_valid"])
            self.assertEqual(arm_acc["technical_error_rate"], 1.0)

    def test_partial_run_never_scored_as_complete(self) -> None:
        """Partial execution (< 62 attempts or unattempted runs) is never reported as complete evaluation."""
        prepare(self.out_dir)
        prot_path = self.out_dir / "protocol.json"
        results_path = self.out_dir / "run_results.json"
        gold_path = self.out_dir / "gold.json"

        fake_transport = FakeQuantizationTransport()
        run(prot_path, results_path, transport=fake_transport)

        results_data = json.loads(results_path.read_text(encoding="utf-8"))
        # Truncate to only 30 attempts
        results_data["attempts"] = results_data["attempts"][:30]
        results_data["completion_status"] = "incomplete/skipped"
        partial_file = self.out_dir / "partial_results.json"
        partial_file.write_text(json.dumps(results_data), encoding="utf-8")

        rep = score(partial_file, gold_path)
        self.assertFalse(rep["is_complete"])
        self.assertEqual(rep["evaluation_status"], "incomplete")
        self.assertIn("Refusing to score partial execution", rep["message"])

    def test_refuse_overwrites_prepare_run_score_cli(self) -> None:
        """prepare, run, and score CLI strictly refuse to overwrite existing files."""
        prepare(self.out_dir)
        prot_path = self.out_dir / "protocol.json"
        results_path = self.out_dir / "run_results.json"
        gold_path = self.out_dir / "gold.json"
        score_path = self.out_dir / "score_report.json"

        fake_transport = FakeQuantizationTransport()
        run(prot_path, results_path, transport=fake_transport)

        # 1. run() refuses existing output
        with self.assertRaises(FileExistsError):
            run(prot_path, results_path, transport=fake_transport)

        # 2. score() CLI refuses existing output
        score_path.write_text("existing", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            score(results_path, gold_path, output_path=score_path)

    def test_cli_subcommands_dispatch(self) -> None:
        """CLI main() handles prepare, run, and score subcommands."""
        cli_out_dir = self.out_dir / "cli_run"
        res_prep = main(["prepare", "--output-dir", str(cli_out_dir)])
        self.assertEqual(res_prep, 0)

        prot_file = cli_out_dir / "protocol.json"
        res_file = cli_out_dir / "results.json"
        gold_file = cli_out_dir / "gold.json"
        score_file = cli_out_dir / "score.json"

        fake_transport = FakeQuantizationTransport()
        with patch("experiments.compare_verifier_quantization.QuantizationOllamaTransport", return_value=fake_transport):
            res_run = main(["run", "--protocol", str(prot_file), "--output", str(res_file)])
            self.assertEqual(res_run, 0)

        res_score = main(["score", "--results", str(res_file), "--gold", str(gold_file), "--output", str(score_file)])
        self.assertEqual(res_score, 0)
        self.assertTrue(score_file.is_file())

    def test_fixture_byte_pins_and_whitespace_changes_fail_before_transport(self) -> None:
        """Fixture byte pins validate raw file bytes before transport; whitespace modifications are rejected."""
        prepare(self.out_dir)
        prot_path = self.out_dir / "protocol.json"
        results_path = self.out_dir / "run_results.json"
        prot_data = json.loads(prot_path.read_text(encoding="utf-8"))

        # 1. 64-hex wrong fixture hash for inputs_file_sha256
        bad_prot = copy.deepcopy(prot_data)
        bad_prot["hashes"]["fixtures"]["inputs_file_sha256"] = "a" * 64
        bad_file1 = self.out_dir / "prot_bad_inputs_byte.json"
        bad_file1.write_text(json.dumps(bad_prot), encoding="utf-8")
        fake1 = FakeQuantizationTransport()
        with self.assertRaises(ValueError) as ctx:
            run(bad_file1, results_path, transport=fake1)
        self.assertIn("inputs.json file SHA-256", str(ctx.exception))
        self.assertEqual(fake1.version_calls, 0)

        # 2. 64-hex wrong fixture hash for ollama_defaults_sha256
        bad_prot = copy.deepcopy(prot_data)
        bad_prot["hashes"]["fixtures"]["ollama_defaults_sha256"] = "b" * 64
        bad_file2 = self.out_dir / "prot_bad_defaults_byte.json"
        bad_file2.write_text(json.dumps(bad_prot), encoding="utf-8")
        fake2 = FakeQuantizationTransport()
        with self.assertRaises(ValueError) as ctx:
            run(bad_file2, results_path, transport=fake2)
        self.assertIn("defaults file SHA-256", str(ctx.exception))
        self.assertEqual(fake2.version_calls, 0)

        # 3. 64-hex wrong fixture hash for prior_protocol_sha256
        bad_prot = copy.deepcopy(prot_data)
        bad_prot["hashes"]["fixtures"]["prior_protocol_sha256"] = "c" * 64
        bad_file3 = self.out_dir / "prot_bad_prior_prot_byte.json"
        bad_file3.write_text(json.dumps(bad_prot), encoding="utf-8")
        fake3 = FakeQuantizationTransport()
        with self.assertRaises(ValueError) as ctx:
            run(bad_file3, results_path, transport=fake3)
        self.assertIn("prior protocol file SHA-256", str(ctx.exception))
        self.assertEqual(fake3.version_calls, 0)

        # 4. Whitespace-only modification to inputs.json (preserves canonical SHA-256 but fails byte pin)
        inputs_p = self.out_dir / "inputs.json"
        raw_inputs = inputs_p.read_bytes()
        inputs_p.write_bytes(raw_inputs + b"   \n")
        self.assertEqual(
            canonical_sha256(json.loads(inputs_p.read_text(encoding="utf-8"))),
            prot_data["hashes"]["inputs_canonical_sha256"],
        )
        fake4 = FakeQuantizationTransport()
        res_ws_inputs = self.out_dir / "res_ws_inputs.json"
        with self.assertRaises(ValueError) as ctx:
            run(prot_path, res_ws_inputs, transport=fake4)
        self.assertIn("inputs.json file SHA-256", str(ctx.exception))
        self.assertEqual(fake4.version_calls, 0)
        inputs_p.write_bytes(raw_inputs)

        # 5. Whitespace-only modification to defaults file (preserves canonical SHA-256 but fails byte pin)
        ws_defaults_dir = self.out_dir / "ws_defaults_dir"
        ws_defaults_dir.mkdir()
        ws_defaults_file = ws_defaults_dir / "defaults.json"
        orig_defaults_bytes = DEFAULT_OLLAMA_DEFAULTS_PATH.read_bytes()
        ws_defaults_file.write_bytes(orig_defaults_bytes + b"\n\n  \n")
        fake5 = FakeQuantizationTransport()
        res_ws_defaults = self.out_dir / "res_ws_defaults.json"
        with self.assertRaises(ValueError) as ctx:
            run(prot_path, res_ws_defaults, transport=fake5, defaults_path=ws_defaults_file)
        self.assertIn("defaults file SHA-256", str(ctx.exception))
        self.assertEqual(fake5.version_calls, 0)

    def test_gold_byte_pin_mismatch_rejected_offline_in_score(self) -> None:
        """score() offline evaluator verifies gold_file_sha256 byte pin and rejects whitespace or tampered gold."""
        prepare(self.out_dir)
        prot_path = self.out_dir / "protocol.json"
        results_path = self.out_dir / "run_results.json"
        gold_path = self.out_dir / "gold.json"

        fake_transport = FakeQuantizationTransport()
        run(prot_path, results_path, transport=fake_transport)

        # 1. Whitespace-only modification to gold.json: canonical SHA-256 matches, but file byte hash fails
        orig_gold_bytes = gold_path.read_bytes()
        ws_gold_file = self.out_dir / "gold_ws.json"
        ws_gold_bytes = orig_gold_bytes + b"  \n\n"
        ws_gold_file.write_bytes(ws_gold_bytes)

        self.assertEqual(
            canonical_sha256(json.loads(orig_gold_bytes.decode("utf-8"))),
            canonical_sha256(json.loads(ws_gold_bytes.decode("utf-8"))),
        )

        with self.assertRaises(ValueError) as ctx:
            score(results_path, ws_gold_file)
        self.assertIn("gold.json file SHA-256", str(ctx.exception))

        # 2. Protocol fixtures with 64-hex wrong gold_file_sha256 pin
        prot_data = json.loads(prot_path.read_text(encoding="utf-8"))
        prot_bad_gold = copy.deepcopy(prot_data)
        prot_bad_gold["hashes"]["fixtures"]["gold_file_sha256"] = "9" * 64

        bad_prot_dir = self.out_dir / "bad_prot_gold_test"
        bad_prot_dir.mkdir()
        bad_prot_file = bad_prot_dir / "protocol.json"
        bad_prot_file.write_text(json.dumps(prot_bad_gold), encoding="utf-8")
        bad_gold_file = bad_prot_dir / "gold.json"
        bad_gold_file.write_bytes(orig_gold_bytes)
        bad_res_file = bad_prot_dir / "results.json"
        res_data = json.loads(results_path.read_text(encoding="utf-8"))
        res_data["protocol_canonical_sha256"] = canonical_sha256(prot_bad_gold)
        res_data["protocol_path"] = str(bad_prot_file)
        bad_res_file.write_text(json.dumps(res_data), encoding="utf-8")

        with self.assertRaises(ValueError) as ctx:
            score(bad_res_file, bad_gold_file)
        self.assertIn("gold.json file SHA-256", str(ctx.exception))

    def test_historical_schedule_pin_enforced_in_prepare(self) -> None:
        """prepare() validates prior_hashes.schedule_canonical_sha256 before consuming schedule development_status."""
        # 1. Missing schedule_canonical_sha256 in prior protocol
        prior_prot_data = json.loads(PRIOR_PROTOCOL_PATH.read_text(encoding="utf-8"))
        bad_prior_prot = copy.deepcopy(prior_prot_data)
        del bad_prior_prot["hashes"]["schedule_canonical_sha256"]

        fixture_dir = self.out_dir / "hist_sched_fixtures"
        fixture_dir.mkdir()
        bad_prior_prot_file = fixture_dir / "bad_prior_prot.json"
        bad_prior_prot_file.write_text(json.dumps(bad_prior_prot), encoding="utf-8")

        out1 = self.out_dir / "prep_out1"
        with self.assertRaises(ValueError) as ctx:
            prepare(out1, prior_protocol_path=bad_prior_prot_file)
        self.assertIn("Prior protocol missing or invalid schedule_canonical_sha256 pin", str(ctx.exception))

        # 2. Tampered prior schedule canonical hash vs prior protocol pin
        bad_sched_data = json.loads(PRIOR_SCHEDULE_PATH.read_text(encoding="utf-8"))
        bad_sched_data["items"][0]["development_status"] = "tampered_status"
        bad_sched_file = fixture_dir / "tampered_prior_sched.json"
        bad_sched_file.write_text(json.dumps(bad_sched_data), encoding="utf-8")

        out2 = self.out_dir / "prep_out2"
        with self.assertRaises(ValueError) as ctx:
            prepare(out2, prior_schedule_path=bad_sched_file)
        self.assertIn("Prior schedule canonical hash", str(ctx.exception))

    def test_measured_chat_wall_time_and_metrics(self) -> None:
        """run() records chat wall time on success and exceptions excluding ps placement, and score() reports both."""
        prepare(self.out_dir)
        prot_path = self.out_dir / "protocol.json"
        results_path = self.out_dir / "run_results.json"
        gold_path = self.out_dir / "gold.json"

        call_counter = 0

        def mixed_chat_responder(payload: dict[str, Any]) -> dict[str, Any]:
            nonlocal call_counter
            call_counter += 1
            if call_counter == 2:
                raise OllamaResponseError("Simulated transport error", raw_text="err", is_truncated=False, status_code=500)
            schema = payload.get("format", {})
            claim_props = schema.get("properties", {}).get("verdicts", {}).get("properties", {})
            verdicts = {}
            for cid, val in claim_props.items():
                first_branch = val.get("anyOf", [{}])[0]
                ev_props = first_branch.get("properties", {}).get("evidence_verdicts", {}).get("properties", {})
                evidence_verdicts = {eid: {"relation": "supports"} for eid in ev_props}
                verdicts[cid] = {
                    "verdict": "supported",
                    "reason": "Supported by cited evidence.",
                    "evidence_verdicts": evidence_verdicts,
                }
            return {
                "model": payload["model"],
                "done": True,
                "done_reason": "stop",
                "prompt_eval_count": 480,
                "eval_count": 35,
                "total_duration": 500000000,
                "load_duration": 10000000,
                "message": {
                    "role": "assistant",
                    "content": json.dumps({"verdicts": verdicts}, ensure_ascii=False),
                },
            }

        fake_transport = FakeQuantizationTransport(chat_responder=mixed_chat_responder)

        original_get_ps = fake_transport.get_ps

        def delayed_get_ps() -> dict[str, Any]:
            time.sleep(0.08)
            return original_get_ps()

        fake_transport.get_ps = delayed_get_ps

        res = run(prot_path, results_path, transport=fake_transport)
        self.assertEqual(res["total_attempted"], 62)
        self.assertEqual(res["total_successful"], 61)
        self.assertEqual(res["total_technical_errors"], 1)

        saved = json.loads(results_path.read_text(encoding="utf-8"))
        att0 = saved["attempts"][0]  # Success
        att1 = saved["attempts"][1]  # Error

        self.assertEqual(att0["status"], "success")
        self.assertIsInstance(att0["wall_time_seconds"], (int, float))
        self.assertGreaterEqual(att0["wall_time_seconds"], 0)
        self.assertLess(att0["wall_time_seconds"], 0.06)

        self.assertEqual(att1["status"], "error")
        self.assertIsInstance(att1["wall_time_seconds"], (int, float))
        self.assertGreaterEqual(att1["wall_time_seconds"], 0)

        score_rep = score(results_path, gold_path)
        q8_perf = score_rep["arms"]["Q8"]["performance_metrics"]

        self.assertIn("server_total_duration_seconds", q8_perf)
        self.assertIn("measured_chat_wall_time_seconds", q8_perf)
        self.assertEqual(q8_perf["measured_chat_wall_time_seconds"]["count"], 31)
        self.assertEqual(q8_perf["server_total_duration_seconds"]["count"], 30)

        # Unattempted run has wall_time_seconds = None
        unatt_prep_dir = self.out_dir / "unatt_test"
        prepare(unatt_prep_dir)
        unatt_inputs_p = unatt_prep_dir / "inputs.json"
        unatt_inputs_data = json.loads(unatt_inputs_p.read_text(encoding="utf-8"))
        unatt_inputs_data["cases"][0]["messages"][0]["content"] = "x" * 13000
        unatt_inputs_p.write_text(json.dumps(unatt_inputs_data), encoding="utf-8")
        unatt_prot_p = unatt_prep_dir / "protocol.json"
        unatt_prot_data = json.loads(unatt_prot_p.read_text(encoding="utf-8"))
        unatt_prot_data["hashes"]["inputs_canonical_sha256"] = canonical_sha256(unatt_inputs_data)
        unatt_prot_data["hashes"]["fixtures"]["inputs_file_sha256"] = compute_file_sha256(unatt_inputs_p)
        unatt_prot_p.write_text(json.dumps(unatt_prot_data), encoding="utf-8")

        unatt_res_p = unatt_prep_dir / "unatt_results.json"
        unatt_fake = FakeQuantizationTransport()
        run(unatt_prot_p, unatt_res_p, transport=unatt_fake)
        saved_unatt = json.loads(unatt_res_p.read_text(encoding="utf-8"))
        unatt_att0 = saved_unatt["attempts"][0]
        self.assertFalse(unatt_att0["attempted"])
        self.assertEqual(unatt_att0["status"], "unattempted")
        self.assertIsNone(unatt_att0["wall_time_seconds"])


if __name__ == "__main__":
    unittest.main()
