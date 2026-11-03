"""Unit tests for Phase 1 & Phase 2 of verifier capacity comparison (experiments.compare_verifier_capacity).

Tests offline protocol/case preparation, case counts, label separation, b05 fidelity,
deterministic schedule, fixture integrity, refusal of overwrites and non-finite inputs.
Phase 2 tests: dedicated Ollama transport adapter, mocked full schedule (124 attempts),
budget/think and arm isolation, request equality, label/gold isolation, unload flow,
handling of model/digest unavailability, and technical errors with raw response and checkpoint preservation.
Zero network or live model dependencies.
"""

from __future__ import annotations

import copy
import hashlib
from io import StringIO
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import urllib.error
import urllib.request

from rag_pensiones.ingest import PROJECT_ROOT

from experiments.compare_verifier_capacity import (
    ALLOWED_OLLAMA_MODELS,
    ALLOWED_VERDICT_LABELS,
    ARMS,
    CONTEXT_SAFETY_MARGIN,
    DEFAULT_COUNTEREXAMPLES_PATH,
    DEFAULT_OLLAMA_DEFAULTS_PATH,
    DEFAULT_OLLAMA_URL,
    DEFAULT_PAIRS_PATH,
    DEFAULT_RUN_TIMEOUT,
    MAX_CONTEXT_WINDOW,
    MAX_RUN_RESPONSE_BYTES,
    OllamaResponseError,
    OllamaTransport,
    OllamaTransportProtocol,
    REQUIRED_CODE_MODULES,
    SCORE_REPORT_SCHEMA_VERSION,
    _NoRedirectHandler,
    assert_finite,
    build_schedule,
    canonical_json_bytes,
    canonical_sha256,
    check_model_supports_thinking,
    compute_file_sha256,
    compute_numeric_summary,
    extract_b05_case,
    extract_verdict_label,
    get_arm_case_order,
    is_valid_sha256,
    main,
    parse_show_parameters,
    prepare,
    resolve_effective_sampling,
    run,
    score,
    unload_resident_models,
    validate_sampling_match,
)
from rag_pensiones.grounding import (
    MAX_MESSAGES_BYTES_VERIFICATION,
    validate_verification_payload,
)


class FakeOllamaTransport:
    """Explicit fake transport implementing OllamaTransportProtocol for offline deterministic tests."""

    def __init__(
        self,
        tags: dict[str, Any] | None = None,
        ps_models: list[dict[str, Any]] | None = None,
        show_responses: dict[str, dict[str, Any]] | None = None,
        chat_responder: Any = None,
        version: str = "0.40.11",
    ) -> None:
        self.tags = tags or {
            "models": [
                {
                    "name": "qwen3.5:9b",
                    "model": "qwen3.5:9b",
                    "digest": "sha256:9b_valid_digest_1234567890abcdef",
                },
                {
                    "name": "qwen3.8:27b",
                    "model": "qwen3.8:27b",
                    "digest": "sha256:27b_valid_digest_1234567890abcdef",
                },
            ]
        }
        self.ps_models = ps_models or [
            {"name": "resident_model_prev", "model": "resident_model_prev", "size_vram": 1024}
        ]
        self.show_responses = show_responses or {
            "qwen3.5:9b": {
                "parameters": "num_ctx 8192\ntemperature 0\npresence_penalty 1.5\ntop_k 20\ntop_p 0.95\n",
                "details": {"family": "qwen"},
                "supports_thinking": True,
            },
            "qwen3.8:27b": {
                "parameters": "num_ctx 8192\ntemperature 0\n",
                "details": {"family": "qwen"},
                "supports_thinking": False,
            },
        }
        self.chat_responder = chat_responder
        self.version = version
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
        return {"parameters": "", "details": {}}

    def unload(self, model: str) -> dict[str, Any]:
        self.unload_calls.append(model)
        self.ps_models = [m for m in self.ps_models if m.get("name") != model and m.get("model") != model]
        return {"status": "unloaded"}

    def chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.chat_calls.append(copy.deepcopy(payload))
        if self.chat_responder is not None:
            resp = self.chat_responder(payload)
            if isinstance(resp, (str, bytes)):
                text = resp.decode("utf-8", errors="replace") if isinstance(resp, bytes) else resp
                if len(text) > MAX_RUN_RESPONSE_BYTES:
                    capped = text[:MAX_RUN_RESPONSE_BYTES]
                    raise OllamaResponseError(
                        "Response exceeded 4 MiB limit",
                        raw_text=capped,
                        is_truncated=True,
                    )
                try:
                    data = json.loads(text)
                    assert_finite(data)
                except Exception as exc:
                    raise OllamaResponseError(
                        f"Malformed/non-finite JSON in chat: {exc}",
                        raw_text=text,
                        is_truncated=False,
                    ) from exc
                return data
            return resp

        # Default responder: valid verification output matching schema
        model = payload["model"]
        schema = payload["format"]

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
        thinking_text = "Step-by-step thinking trace for claims." if payload.get("think") else None

        resp = {
            "model": model,
            "created_at": "2026-10-26T12:00:00Z",
            "message": {
                "role": "assistant",
                "content": json.dumps(content_json, ensure_ascii=False),
            },
            "done": True,
            "done_reason": "stop",
            "total_duration": 1200000000,
            "load_duration": 100000000,
            "prompt_eval_count": 250,
            "prompt_eval_duration": 300000000,
            "eval_count": 80,
            "eval_duration": 800000000,
        }
        if thinking_text:
            resp["message"]["thinking"] = thinking_text
        return resp


class TestVerifierCapacityPreparation(unittest.TestCase):
    """Test suite for Phase 1 verifier capacity experiment preparation (preserved)."""

    def setUp(self) -> None:
        self.pairs_path = DEFAULT_PAIRS_PATH
        self.ce_path = DEFAULT_COUNTEREXAMPLES_PATH
        self.initial_pairs_hash = compute_file_sha256(self.pairs_path)
        self.initial_ce_hash = compute_file_sha256(self.ce_path)

    def tearDown(self) -> None:
        current_pairs_hash = compute_file_sha256(self.pairs_path)
        current_ce_hash = compute_file_sha256(self.ce_path)
        self.assertEqual(
            self.initial_pairs_hash,
            current_pairs_hash,
            "evaluation/grounding_pairs.json was modified during testing!",
        )
        self.assertEqual(
            self.initial_ce_hash,
            current_ce_hash,
            "evaluation/baseline_stability_counterexamples.json was modified during testing!",
        )

    def test_case_counts_and_arm_totals(self) -> None:
        """Verifies 29 unique cases (28 historical + 1 b05), and 124 scheduled executions across 4 arms."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir) / "protocol_out"
            res = prepare(out_dir)

            inputs_path = Path(res["inputs_path"])
            gold_path = Path(res["gold_path"])
            schedule_path = Path(res["schedule_path"])
            protocol_path = Path(res["protocol_path"])

            self.assertTrue(inputs_path.is_file())
            self.assertTrue(gold_path.is_file())
            self.assertTrue(schedule_path.is_file())
            self.assertTrue(protocol_path.is_file())

            inputs_data = json.loads(inputs_path.read_text(encoding="utf-8"))
            gold_data = json.loads(gold_path.read_text(encoding="utf-8"))
            schedule_data = json.loads(schedule_path.read_text(encoding="utf-8"))

            self.assertEqual(inputs_data["case_count"], 29)
            self.assertEqual(len(inputs_data["cases"]), 29)

            historical_cases = [c for c in inputs_data["cases"] if c["kind"] == "historical_control"]
            b05_cases = [c for c in inputs_data["cases"] if c["id"] == "b05"]
            self.assertEqual(len(historical_cases), 28)
            self.assertEqual(len(b05_cases), 1)

            self.assertEqual(gold_data["case_count"], 29)
            self.assertEqual(len(gold_data["cases"]), 29)
            self.assertEqual(len(gold_data["labels"]), 29)

            self.assertEqual(schedule_data["total_runs"], 124)
            self.assertEqual(len(schedule_data["items"]), 124)

            for arm_id in ["A", "B", "C", "D"]:
                arm_items = [item for item in schedule_data["items"] if item["arm_id"] == arm_id]
                self.assertEqual(len(arm_items), 31, f"Arm {arm_id} must have exactly 31 executions")
                arm_b05 = [item for item in arm_items if item["case_id"] == "b05"]
                self.assertEqual(len(arm_b05), 3, f"Arm {arm_id} must have exactly 3 b05 executions")
                arm_non_b05 = [item for item in arm_items if item["case_id"] != "b05"]
                self.assertEqual(len(arm_non_b05), 28, f"Arm {arm_id} must have 28 historical runs")

            total_b05_runs = sum(1 for item in schedule_data["items"] if item["case_id"] == "b05")
            total_non_b05_runs = sum(1 for item in schedule_data["items"] if item["case_id"] != "b05")
            self.assertEqual(total_b05_runs, 12)
            self.assertEqual(total_non_b05_runs, 112)

    def test_labels_and_metadata_isolation(self) -> None:
        """Verifies strict separation: model messages contain no labels, IDs, split names, or kinds."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir) / "protocol_out"
            res = prepare(out_dir)

            inputs_data = json.loads(Path(res["inputs_path"]).read_text(encoding="utf-8"))
            gold_data = json.loads(Path(res["gold_path"]).read_text(encoding="utf-8"))

            gold_labels = gold_data["labels"]
            self.assertEqual(len(gold_labels), 29)
            self.assertEqual(gold_labels["b05"], "supported")

            for inp in inputs_data["cases"]:
                cid = inp["id"]
                messages = inp["messages"]

                self.assertEqual(len(messages), 2)
                self.assertEqual(messages[0]["role"], "system")
                self.assertEqual(messages[1]["role"], "user")

                user_content = messages[1]["content"]

                self.assertNotIn(cid, user_content, f"Case ID {cid!r} leaked in model user content")
                self.assertNotIn('"dev"', user_content)
                self.assertNotIn('"reserved"', user_content)
                self.assertNotIn('"split"', user_content)
                self.assertNotIn('"kind"', user_content)
                self.assertIn("C1", user_content)
                self.assertIn("E1", user_content)

    def test_b05_byte_and_json_faithful_messages_and_schema(self) -> None:
        """Verifies b05 preserves verification messages and schema exactly from counterexamples trace."""
        ce_data = json.loads(self.ce_path.read_text(encoding="utf-8"))
        b05_raw = next(c for c in ce_data["cases"] if c["id"] == "b05")
        verify_call_raw = next(
            item for item in b05_raw["calls_for_development"] if item["stage"] == "verification"
        )
        expected_messages = verify_call_raw["call"]["messages"]
        expected_schema = verify_call_raw["call"]["schema"]

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir) / "protocol_out"
            res = prepare(out_dir)

            inputs_data = json.loads(Path(res["inputs_path"]).read_text(encoding="utf-8"))
            b05_input = next(c for c in inputs_data["cases"] if c["id"] == "b05")

            self.assertEqual(b05_input["messages"], expected_messages)
            self.assertEqual(b05_input["schema"], expected_schema)

            self.assertEqual(
                canonical_json_bytes(b05_input["messages"]),
                canonical_json_bytes(expected_messages),
            )
            self.assertEqual(
                canonical_json_bytes(b05_input["schema"]),
                canonical_json_bytes(expected_schema),
            )

            self.assertEqual(len(b05_input["claims"]), 1)
            self.assertEqual(b05_input["claims"][0]["claim_id"], "C1")
            self.assertEqual(b05_input["claims"][0]["evidence_ids"], ["E1", "E2"])
            self.assertEqual(len(b05_input["evidence"]), 2)
            self.assertEqual(b05_input["evidence"][0]["evidence_id"], "E1")
            self.assertEqual(b05_input["evidence"][1]["evidence_id"], "E2")

            dummy_supported_verdict = {
                "verdicts": [
                    {
                        "claim_id": "C1",
                        "verdict": "supported",
                        "reason": "E2 provides sufficient support",
                        "evidence_verdicts": [
                            {"evidence_id": "E1", "relation": "irrelevant"},
                            {"evidence_id": "E2", "relation": "supports"},
                        ],
                    }
                ]
            }
            validated = validate_verification_payload(
                dummy_supported_verdict,
                claims=b05_input["claims"],
                evidence=b05_input["evidence"],
            )
            self.assertEqual(validated["verdicts"][0]["verdict"], "supported")

    def test_deterministic_hashes_and_schedule(self) -> None:
        """Verifies preparation produces identical hashes and identical schedule across runs."""
        with tempfile.TemporaryDirectory() as tmp1, tempfile.TemporaryDirectory() as tmp2:
            dir1 = Path(tmp1) / "prep1"
            dir2 = Path(tmp2) / "prep2"

            res1 = prepare(dir1)
            res2 = prepare(dir2)

            self.assertEqual(
                res1["hashes"]["inputs_canonical_sha256"],
                res2["hashes"]["inputs_canonical_sha256"],
            )
            self.assertEqual(
                res1["hashes"]["schedule_canonical_sha256"],
                res2["hashes"]["schedule_canonical_sha256"],
            )
            self.assertEqual(
                res1["hashes"]["gold_canonical_sha256"],
                res2["hashes"]["gold_canonical_sha256"],
            )

            sched1 = json.loads(Path(res1["schedule_path"]).read_text(encoding="utf-8"))
            sched2 = json.loads(Path(res2["schedule_path"]).read_text(encoding="utf-8"))
            self.assertEqual(sched1["items"], sched2["items"])

            case_ids = [c["id"] for c in json.loads(Path(res1["inputs_path"]).read_text())["cases"]]
            order_A = get_arm_case_order("A", case_ids)
            order_B = get_arm_case_order("B", case_ids)
            order_C = get_arm_case_order("C", case_ids)
            order_D = get_arm_case_order("D", case_ids)

            self.assertEqual(order_A, case_ids)
            self.assertEqual(order_B, list(reversed(case_ids)))
            shift = len(case_ids) // 2
            self.assertEqual(order_C, case_ids[shift:] + case_ids[:shift])
            self.assertEqual(order_D, list(reversed(order_C)))

            arm_sequence = [item["arm_id"] for item in sched1["items"]]
            self.assertEqual(arm_sequence[:31], ["A"] * 31)
            self.assertEqual(arm_sequence[31:62], ["B"] * 31)
            self.assertEqual(arm_sequence[62:93], ["C"] * 31)
            self.assertEqual(arm_sequence[93:124], ["D"] * 31)

    def test_refusal_of_overwrite_and_non_empty_dir(self) -> None:
        """Verifies prepare strictly refuses to overwrite existing files or non-empty directories."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir) / "target"
            prepare(out_dir)

            with self.assertRaises(FileExistsError) as ctx:
                prepare(out_dir)
            self.assertIn("Refusing to overwrite", str(ctx.exception))

            out_dir2 = Path(tmp_dir) / "target2"
            out_dir2.mkdir()
            (out_dir2 / "some_file.txt").write_text("hello", encoding="utf-8")
            with self.assertRaises(FileExistsError) as ctx2:
                prepare(out_dir2)
            self.assertIn("not empty", str(ctx2.exception))

    def test_refusal_of_nonfinite_inputs(self) -> None:
        """Verifies canonical JSON serializer and assert_finite strictly reject NaN and Inf."""
        with self.assertRaises(ValueError):
            assert_finite({"a": float("nan")})
        with self.assertRaises(ValueError):
            assert_finite({"a": [1.0, float("inf")]})
        with self.assertRaises(ValueError):
            assert_finite({"a": [1.0, float("-inf")]})

        with self.assertRaises(ValueError):
            canonical_json_bytes({"a": float("nan")})
        with self.assertRaises(ValueError):
            canonical_json_bytes({"a": float("inf")})

        with tempfile.TemporaryDirectory() as tmp_dir:
            bad_pairs = Path(tmp_dir) / "bad_pairs.json"
            bad_pairs.write_text('{"cases": [{"id": "d01", "val": NaN}]}', encoding="utf-8")
            with self.assertRaises(ValueError):
                prepare(Path(tmp_dir) / "out", pairs_path=bad_pairs)

    def test_malformed_b05_rejected(self) -> None:
        """Verifies prepare rejects malformed b05 cases in counterexamples."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            bad_ce = Path(tmp_dir) / "no_b05.json"
            bad_ce.write_text('{"cases": [{"id": "other"}]}', encoding="utf-8")
            with self.assertRaises(ValueError) as ctx:
                prepare(Path(tmp_dir) / "out", counterexamples_path=bad_ce)
            self.assertIn("Case 'b05' not found", str(ctx.exception))

        with tempfile.TemporaryDirectory() as tmp_dir:
            bad_ce2 = Path(tmp_dir) / "no_verify.json"
            bad_ce2.write_text(
                '{"cases": [{"id": "b05", "calls_for_development": [{"stage": "generation"}]}]}',
                encoding="utf-8",
            )
            with self.assertRaises(ValueError) as ctx:
                prepare(Path(tmp_dir) / "out", counterexamples_path=bad_ce2)
            self.assertIn("No verification stage call found", str(ctx.exception))

        oversized_ce = {
            "cases": [
                {
                    "id": "b05",
                    "calls_for_development": [
                        {
                            "stage": "verification",
                            "call": {
                                "messages": [
                                    {"role": "system", "content": "x" * (MAX_MESSAGES_BYTES_VERIFICATION + 10)},
                                    {"role": "user", "content": '{"afirmaciones": [], "catalogo_evidencias": []}'},
                                ],
                                "schema": {"type": "object"},
                            },
                        }
                    ],
                }
            ]
        }
        with self.assertRaises(ValueError) as ctx:
            extract_b05_case(oversized_ce)
        self.assertIn("exceed maximum budget", str(ctx.exception))

    def test_arms_configuration(self) -> None:
        """Verifies exact contract specification for the four experimental arms."""
        self.assertEqual(set(ARMS.keys()), {"A", "B", "C", "D"})

        arm_a = ARMS["A"]
        self.assertEqual(arm_a["model"], "qwen3.5:9b")
        self.assertIs(arm_a["think"], False)
        self.assertEqual(arm_a["options"]["num_predict"], 1024)
        self.assertEqual(arm_a["options"]["num_ctx"], 8192)
        self.assertEqual(arm_a["options"]["temperature"], 0)
        self.assertEqual(arm_a["options"]["seed"], 42)
        self.assertNotIn("repeat_penalty", arm_a["options"])

        arm_b = ARMS["B"]
        self.assertEqual(arm_b["model"], "qwen3.5:9b")
        self.assertIs(arm_b["think"], False)
        self.assertEqual(arm_b["options"]["num_predict"], 4096)
        self.assertEqual(arm_b["options"]["num_ctx"], 8192)
        self.assertEqual(arm_b["options"]["temperature"], 0)
        self.assertEqual(arm_b["options"]["seed"], 42)

        arm_c = ARMS["C"]
        self.assertEqual(arm_c["model"], "qwen3.5:9b")
        self.assertIs(arm_c["think"], True)
        self.assertEqual(arm_c["options"]["num_predict"], 4096)
        self.assertEqual(arm_c["options"]["num_ctx"], 8192)
        self.assertEqual(arm_c["options"]["temperature"], 0)
        self.assertEqual(arm_c["options"]["seed"], 42)

        arm_d = ARMS["D"]
        self.assertEqual(arm_d["model"], "qwen3.8:27b")
        self.assertIs(arm_d["think"], False)
        self.assertEqual(arm_d["options"]["num_predict"], 4096)
        self.assertEqual(arm_d["options"]["num_ctx"], 8192)
        self.assertEqual(arm_d["options"]["temperature"], 0)
        self.assertEqual(arm_d["options"]["seed"], 42)
        self.assertEqual(arm_d["options"]["presence_penalty"], 1.5)
        self.assertEqual(arm_d["options"]["top_k"], 20)
        self.assertEqual(arm_d["options"]["top_p"], 0.95)
        self.assertEqual(arm_d["options"]["min_p"], 0.0)

    def test_cli_interface_and_deferred_commands(self) -> None:
        """Verifies CLI prepare subcommand and confirms score remains deferred."""
        with patch("sys.stdout", new_callable=StringIO) as mock_out:
            with self.assertRaises(SystemExit) as exit_ctx:
                main(["--help"])
            self.assertEqual(exit_ctx.exception.code, 0)
            self.assertIn("Verifier capacity comparison", mock_out.getvalue())

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir) / "cli_out"
            ret = main(["prepare", "--output-dir", str(out_dir)])
            self.assertEqual(ret, 0)
            self.assertTrue((out_dir / "protocol.json").is_file())
            self.assertTrue((out_dir / "inputs.json").is_file())
            self.assertTrue((out_dir / "gold.json").is_file())
            self.assertTrue((out_dir / "schedule.json").is_file())

        # score CLI requires arguments
        with self.assertRaises(SystemExit):
            main(["score"])

    def test_prepare_via_python_m_in_tempdir_pins_itself(self) -> None:
        """Verifies prepare invoked via python -m from an isolated tempdir pins itself without sentinels."""
        with tempfile.TemporaryDirectory() as tmp_work_dir, tempfile.TemporaryDirectory() as tmp_out_dir:
            out_path = Path(tmp_out_dir) / "prep_out"
            env = dict(os.environ)
            env["PYTHONPATH"] = str(PROJECT_ROOT)
            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "experiments.compare_verifier_capacity",
                    "prepare",
                    "--output-dir",
                    str(out_path),
                ],
                cwd=tmp_work_dir,
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, f"STDOUT: {proc.stdout}\nSTDERR: {proc.stderr}")
            prot_data = json.loads((out_path / "protocol.json").read_text(encoding="utf-8"))
            code_pins = prot_data["hashes"]["code"]
            self.assertIn("compare_verifier_capacity", code_pins)
            pin_val = code_pins["compare_verifier_capacity"]
            pin_sha = pin_val["sha256"] if isinstance(pin_val, dict) else pin_val
            self.assertTrue(is_valid_sha256(pin_sha))
            self.assertNotEqual(pin_sha, "uncommitted_or_not_found")


class TestOllamaTransport(unittest.TestCase):
    """Test suite for dedicated stdlib Ollama adapter validation."""

    def test_transport_defaults_and_loopback_restriction(self) -> None:
        """Verifies loopback 127.0.0.1:11434 requirement, allowed models, and timeouts."""
        transport = OllamaTransport()
        self.assertEqual(transport.base_url, DEFAULT_OLLAMA_URL)
        self.assertEqual(transport.timeout, DEFAULT_RUN_TIMEOUT)
        self.assertEqual(transport.allowed_models, ALLOWED_OLLAMA_MODELS)

        with self.assertRaises(ValueError):
            OllamaTransport(base_url="http://localhost:11434")

        with self.assertRaises(ValueError):
            OllamaTransport(base_url="http://192.168.1.1:11434")

        with self.assertRaises(ValueError):
            OllamaTransport(timeout=0)

        with self.assertRaises(ValueError):
            OllamaTransport(timeout=-10)

        with self.assertRaises(ValueError):
            OllamaTransport(allowed_models=("llama3:8b",))

    def test_redirect_handler_blocks_redirects(self) -> None:
        """Verifies _NoRedirectHandler strictly raises urllib.error.HTTPError on redirect."""
        handler = _NoRedirectHandler()
        req = urllib.request.Request("http://127.0.0.1:11434/api/chat")
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            handler.redirect_request(req, None, 302, "Found", {}, "http://127.0.0.1:11434/redirected")
        self.assertEqual(ctx.exception.code, 302)
        self.assertIn("HTTP redirects are blocked", str(ctx.exception))


class TestVerifierCapacityRun(unittest.TestCase):
    """Test suite for Phase 2 verifier capacity runner."""

    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.protocol_dir = Path(self.tmp_dir.name) / "protocol"
        self.prep_res = prepare(self.protocol_dir)
        self.protocol_path = Path(self.prep_res["protocol_path"])
        self.gold_path = Path(self.prep_res["gold_path"])
        self.output_path = Path(self.tmp_dir.name) / "run_output.json"

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()

    def test_run_rejects_existing_output(self) -> None:
        """Verifies run() strictly refuses to overwrite existing output files."""
        self.output_path.write_text("{}", encoding="utf-8")
        with self.assertRaises(FileExistsError) as ctx:
            run(self.protocol_path, self.output_path, transport=FakeOllamaTransport())
        self.assertIn("already exists", str(ctx.exception))

    def test_run_validates_pins_before_network(self) -> None:
        """Verifies run() validates inputs/schedule/code canonical SHA256 and schemas before any transport access."""
        fake_transport = FakeOllamaTransport()

        # 1. Tamper inputs.json
        inputs_file = self.protocol_dir / "inputs.json"
        inputs_data = json.loads(inputs_file.read_text(encoding="utf-8"))
        inputs_data["case_count"] = 999  # modify canonical hash
        inputs_file.write_text(json.dumps(inputs_data), encoding="utf-8")

        with self.assertRaises(ValueError) as ctx:
            run(self.protocol_path, self.output_path, transport=fake_transport)
        self.assertIn("inputs.json canonical SHA-256", str(ctx.exception))
        self.assertEqual(fake_transport.tags_calls, 0, "No network calls allowed before pin validation")
        self.assertEqual(fake_transport.version_calls, 0)

        # 2. Tamper schedule.json
        prepare(self.protocol_dir / "restored")
        restored_prot = self.protocol_dir / "restored" / "protocol.json"
        sched_file = self.protocol_dir / "restored" / "schedule.json"
        sched_data = json.loads(sched_file.read_text(encoding="utf-8"))
        sched_data["total_runs"] = 999
        sched_file.write_text(json.dumps(sched_data), encoding="utf-8")

        with self.assertRaises(ValueError) as ctx2:
            run(restored_prot, self.output_path, transport=fake_transport)
        self.assertIn("schedule.json canonical SHA-256", str(ctx2.exception))
        self.assertEqual(fake_transport.tags_calls, 0)
        self.assertEqual(fake_transport.version_calls, 0)

        # 3. Tamper code pins in protocol.json
        prepare(self.protocol_dir / "restored_code")
        prot_code = self.protocol_dir / "restored_code" / "protocol.json"
        prot_code_data = json.loads(prot_code.read_text(encoding="utf-8"))
        prot_code_data["hashes"]["code"]["grounding"]["sha256"] = "0" * 64
        prot_code.write_text(json.dumps(prot_code_data), encoding="utf-8")

        with self.assertRaises(ValueError) as ctx3:
            run(prot_code, self.output_path, transport=fake_transport)
        self.assertIn("Tampering/stale code detected", str(ctx3.exception))
        self.assertEqual(fake_transport.tags_calls, 0)
        self.assertEqual(fake_transport.version_calls, 0)

        # 4. Invalid protocol schema_version
        prepare(self.protocol_dir / "restored_schema")
        prot_schema = self.protocol_dir / "restored_schema" / "protocol.json"
        prot_schema_data = json.loads(prot_schema.read_text(encoding="utf-8"))
        prot_schema_data["schema_version"] = "invalid-protocol-v999"
        prot_schema.write_text(json.dumps(prot_schema_data), encoding="utf-8")

        with self.assertRaises(ValueError) as ctx4:
            run(prot_schema, self.output_path, transport=fake_transport)
        self.assertIn("Unexpected protocol schema_version", str(ctx4.exception))
        self.assertEqual(fake_transport.tags_calls, 0)
        self.assertEqual(fake_transport.version_calls, 0)

    def test_run_never_reads_gold(self) -> None:
        """Verifies run() executes completely without ever reading or requiring gold.json."""
        gold_file = self.protocol_dir / "gold.json"
        # Delete gold.json completely
        gold_file.unlink()
        self.assertFalse(gold_file.exists())

        fake_transport = FakeOllamaTransport()
        res = run(self.protocol_path, self.output_path, transport=fake_transport)
        self.assertEqual(res["total_scheduled"], 124)
        self.assertEqual(res["total_successful"], 124)
        self.assertTrue(self.output_path.is_file())

    def test_mocked_full_schedule_124_attempts(self) -> None:
        """Verifies full sequential execution of 124 attempts with atomic checkpoints and metrics."""
        fake_transport = FakeOllamaTransport()
        res = run(self.protocol_path, self.output_path, transport=fake_transport)

        self.assertEqual(res["total_scheduled"], 124)
        self.assertEqual(res["total_attempted"], 124)
        self.assertEqual(res["total_successful"], 124)
        self.assertEqual(res["total_technical_errors"], 0)
        self.assertEqual(res["total_unattempted"], 0)

        # Output JSON validation
        output_data = json.loads(self.output_path.read_text(encoding="utf-8"))
        assert_finite(output_data)
        self.assertEqual(len(output_data["attempts"]), 124)

        # Verify thinking metrics recorded without printing raw thinking text
        arm_c_attempts = [a for a in output_data["attempts"] if a["arm_id"] == "C"]
        self.assertEqual(len(arm_c_attempts), 31)
        for a in arm_c_attempts:
            self.assertGreater(a["thinking_length"], 0)
            self.assertIsNotNone(a["thinking_sha256"])
            self.assertIsInstance(a["thinking_sha256"], str)
            self.assertEqual(len(a["thinking_sha256"]), 64)

        arm_a_attempts = [a for a in output_data["attempts"] if a["arm_id"] == "A"]
        for a in arm_a_attempts:
            self.assertEqual(a["thinking_length"], 0)
            self.assertIsNone(a["thinking_sha256"])

        # Check stats recorded
        for a in output_data["attempts"]:
            self.assertIsNotNone(a["stats"])
            self.assertEqual(a["stats"]["prompt_eval_count"], 250)
            self.assertEqual(a["stats"]["eval_count"], 80)
            self.assertIsNotNone(a["completed_at"])

    def test_budgets_and_options_and_think_isolation(self) -> None:
        """Verifies arm options strictly follow contract (A original 4, B/C num_predict/think, D aligned)."""
        fake_transport = FakeOllamaTransport()
        run(self.protocol_path, self.output_path, transport=fake_transport)

        self.assertEqual(len(fake_transport.chat_calls), 124)

        # Inspect chat calls by arm block
        calls_A = fake_transport.chat_calls[:31]
        calls_B = fake_transport.chat_calls[31:62]
        calls_C = fake_transport.chat_calls[62:93]
        calls_D = fake_transport.chat_calls[93:124]

        for req in calls_A:
            self.assertEqual(req["model"], "qwen3.5:9b")
            self.assertIs(req["think"], False)
            self.assertEqual(
                req["options"],
                {"temperature": 0, "seed": 42, "num_ctx": 8192, "num_predict": 1024},
            )

        for req in calls_B:
            self.assertEqual(req["model"], "qwen3.5:9b")
            self.assertIs(req["think"], False)
            self.assertEqual(
                req["options"],
                {"temperature": 0, "seed": 42, "num_ctx": 8192, "num_predict": 4096},
            )

        for req in calls_C:
            self.assertEqual(req["model"], "qwen3.5:9b")
            self.assertIs(req["think"], True)
            self.assertEqual(
                req["options"],
                {"temperature": 0, "seed": 42, "num_ctx": 8192, "num_predict": 4096},
            )

        for req in calls_D:
            self.assertEqual(req["model"], "qwen3.8:27b")
            self.assertIs(req["think"], False)
            self.assertEqual(
                req["options"],
                {
                    "temperature": 0,
                    "seed": 42,
                    "num_ctx": 8192,
                    "num_predict": 4096,
                    "presence_penalty": 1.5,
                    "top_k": 20,
                    "top_p": 0.95,
                    "min_p": 0.0,
                },
            )

    def test_request_equality_across_arms(self) -> None:
        """Verifies per-case messages and schema are 100% identical across all 4 arms."""
        fake_transport = FakeOllamaTransport()
        run(self.protocol_path, self.output_path, transport=fake_transport)

        # Group chat requests by case_id using schedule items
        schedule_data = json.loads((self.protocol_dir / "schedule.json").read_text(encoding="utf-8"))
        case_requests: dict[str, list[dict]] = {}
        for item, req in zip(schedule_data["items"], fake_transport.chat_calls):
            cid = item["case_id"]
            case_requests.setdefault(cid, []).append(req)

        for cid, reqs in case_requests.items():
            base_messages = reqs[0]["messages"]
            base_format = reqs[0]["format"]
            for r in reqs[1:]:
                self.assertEqual(
                    canonical_json_bytes(r["messages"]),
                    canonical_json_bytes(base_messages),
                    f"Messages mismatch for case {cid}",
                )
                self.assertEqual(
                    canonical_json_bytes(r["format"]),
                    canonical_json_bytes(base_format),
                    f"Format schema mismatch for case {cid}",
                )

    def test_no_gold_or_label_or_case_id_leakage(self) -> None:
        """Verifies no case IDs, labels, splits, or kinds leak into any chat request messages."""
        fake_transport = FakeOllamaTransport()
        run(self.protocol_path, self.output_path, transport=fake_transport)

        inputs_data = json.loads((self.protocol_dir / "inputs.json").read_text(encoding="utf-8"))
        case_ids = {c["id"] for c in inputs_data["cases"]}

        for req in fake_transport.chat_calls:
            for msg in req["messages"]:
                content = msg.get("content", "")
                for cid in case_ids:
                    self.assertNotIn(cid, content, f"Leaked case ID {cid}")
                self.assertNotIn('"dev"', content)
                self.assertNotIn('"reserved"', content)
                self.assertNotIn('"split"', content)
                self.assertNotIn('"kind"', content)

    def test_valid_unload_flow_and_ps_placement(self) -> None:
        """Verifies resident models in /api/ps are unloaded at start, between arms, and at end."""
        fake_transport = FakeOllamaTransport()
        res = run(self.protocol_path, self.output_path, transport=fake_transport)

        # Unloads occurred at start, between arms, and at end
        self.assertGreater(len(fake_transport.unload_calls), 0)
        self.assertIn("resident_model_prev", fake_transport.unload_calls)

        # arm_placements recorded in output
        output_data = json.loads(self.output_path.read_text(encoding="utf-8"))
        arm_placements = output_data["arm_placements"]
        self.assertEqual(set(arm_placements.keys()), {"A", "B", "C", "D"})
        for arm_id in ["A", "B", "C", "D"]:
            self.assertIn("ps_placement", arm_placements[arm_id])
            self.assertIn("model_digest", arm_placements[arm_id])

    def test_model_digest_unavailable_stops_arm(self) -> None:
        """Verifies missing model or digest in /api/tags stops arm, marks cases unattempted (not abstention)."""
        tags_missing_d = {
            "models": [
                {
                    "name": "qwen3.5:9b",
                    "model": "qwen3.5:9b",
                    "digest": "sha256:9b_valid_digest",
                }
                # qwen3.8:27b omitted!
            ]
        }
        fake_transport = FakeOllamaTransport(tags=tags_missing_d)
        res = run(self.protocol_path, self.output_path, transport=fake_transport)

        # Arms A, B, C executed (3 * 31 = 93), Arm D stopped (31 unattempted)
        self.assertEqual(res["total_scheduled"], 124)
        self.assertEqual(res["total_attempted"], 93)
        self.assertEqual(res["total_successful"], 93)
        self.assertEqual(res["total_unattempted"], 31)

        output_data = json.loads(self.output_path.read_text(encoding="utf-8"))
        arm_d_attempts = [a for a in output_data["attempts"] if a["arm_id"] == "D"]
        self.assertEqual(len(arm_d_attempts), 31)
        for a in arm_d_attempts:
            self.assertIs(a["attempted"], False)
            self.assertEqual(a["status"], "unattempted")
            self.assertEqual(a["error_type"], "model_or_digest_unavailable")
            self.assertNotEqual(a.get("parsed_verdict"), "insufficient_evidence")

    def test_technical_errors_preserve_raw_and_checkpoints(self) -> None:
        """Verifies isolated technical errors preserve raw responses, checkpoint atomically, and continue."""
        attempt_counter = 0

        def dynamic_responder(payload: dict) -> dict:
            nonlocal attempt_counter
            attempt_counter += 1

            if attempt_counter == 1:
                # 1. Budget exceeded error
                resp = FakeOllamaTransport().chat(payload)
                resp["prompt_eval_count"] = 7000  # 7000 + 1024 + 256 = 8280 > 8192
                return resp
            elif attempt_counter == 2:
                # 2. Truncation error (done_reason == "length")
                resp = FakeOllamaTransport().chat(payload)
                resp["done_reason"] = "length"
                return resp
            elif attempt_counter == 3:
                # 3. Malformed JSON content
                resp = FakeOllamaTransport().chat(payload)
                resp["message"]["content"] = "This is not json {{"
                return resp
            elif attempt_counter == 4:
                # 4. Non-finite float in response
                resp = FakeOllamaTransport().chat(payload)
                resp["total_duration"] = float("nan")
                return resp
            elif attempt_counter == 5:
                # 5. Transport timeout exception
                raise TimeoutError("Timeout contacting Ollama after 240s")
            elif attempt_counter == 6:
                # 6. Schema / verdict invalidity (missing claim)
                resp = FakeOllamaTransport().chat(payload)
                resp["message"]["content"] = '{"verdicts": {}}'
                return resp
            else:
                return FakeOllamaTransport().chat(payload)

        fake_transport = FakeOllamaTransport(chat_responder=dynamic_responder)
        res = run(self.protocol_path, self.output_path, transport=fake_transport)

        self.assertEqual(res["total_scheduled"], 124)
        self.assertEqual(res["total_attempted"], 124)
        self.assertEqual(res["total_technical_errors"], 6)
        self.assertEqual(res["total_successful"], 118)

        output_data = json.loads(self.output_path.read_text(encoding="utf-8"))
        attempts = output_data["attempts"]

        # Attempt 1: budget exceeded, raw_response preserved
        self.assertEqual(attempts[0]["status"], "error")
        self.assertIn("Context budget exceeded", attempts[0]["error_message"])
        self.assertIsNotNone(attempts[0]["raw_response"])

        # Attempt 2: truncation length, raw_response preserved
        self.assertEqual(attempts[1]["status"], "error")
        self.assertIn("done_reason is 'length'", attempts[1]["error_message"])
        self.assertIsNotNone(attempts[1]["raw_response"])

        # Attempt 3: malformed JSON, raw_response preserved
        self.assertEqual(attempts[2]["status"], "error")
        self.assertIn("Malformed JSON", attempts[2]["error_message"])
        self.assertIsNotNone(attempts[2]["raw_response"])

        # Attempt 4: non-finite float, raw_response preserved
        self.assertEqual(attempts[3]["status"], "error")
        self.assertIn("Non-finite float", attempts[3]["error_message"])
        self.assertIsNotNone(attempts[3]["raw_response"])

        # Attempt 5: transport timeout, raw_response is None
        self.assertEqual(attempts[4]["status"], "error")
        self.assertEqual(attempts[4]["error_type"], "TimeoutError")
        self.assertIsNone(attempts[4]["raw_response"])

        # Attempt 6: invalid verdict, raw_response preserved
        self.assertEqual(attempts[5]["status"], "error")
        self.assertIsNotNone(attempts[5]["raw_response"])

        # Attempt 7+: success
        self.assertEqual(attempts[6]["status"], "success")

    def test_unsupported_think_marks_technical_error(self) -> None:
        """Verifies unsupported thinking metadata marks cases unattempted before chat() and yields incomplete score."""
        show_responses = {
            "qwen3.5:9b": {
                "parameters": "num_ctx 8192\ntemperature 0\npresence_penalty 1.5\ntop_k 20\ntop_p 0.95\n",
                "details": {"family": "qwen"},
                "supports_thinking": False,  # thinking unsupported!
            },
            "qwen3.8:27b": {
                "parameters": "num_ctx 8192\ntemperature 0\n",
                "details": {"family": "qwen"},
                "supports_thinking": False,
            },
        }
        fake_transport = FakeOllamaTransport(show_responses=show_responses)
        res = run(self.protocol_path, self.output_path, transport=fake_transport)

        # Arm C has 31 items, all marked as unattempted (preflight check failed)
        output_data = json.loads(self.output_path.read_text(encoding="utf-8"))
        arm_c_attempts = [a for a in output_data["attempts"] if a["arm_id"] == "C"]
        self.assertEqual(len(arm_c_attempts), 31)
        for a in arm_c_attempts:
            self.assertIs(a["attempted"], False)
            self.assertEqual(a["status"], "unattempted")
            self.assertEqual(a["error_type"], "unsupported_think")

        # Zero chat calls were made with think=True
        think_calls = [c for c in fake_transport.chat_calls if c.get("think") is True]
        self.assertEqual(len(think_calls), 0)

        # Score marks execution incomplete
        report = score(self.output_path, self.gold_path)
        self.assertIs(report["is_complete"], False)
        self.assertEqual(report["evaluation_status"], "incomplete")
        self.assertEqual(report["skipped_runs"], 31)

    def test_digest_changes_between_arms_blocked(self) -> None:
        """Verifies that if a model digest changes after run start, subsequent arms are stopped and marked unattempted."""
        fake_transport = FakeOllamaTransport()
        original_get_tags = fake_transport.get_tags
        call_count = 0

        def dynamic_get_tags() -> dict[str, Any]:
            nonlocal call_count
            call_count += 1
            tags = original_get_tags()
            if call_count > 2:  # After run start and Arm A, alter 9B digest before Arm B
                for m in tags["models"]:
                    if m.get("name") == "qwen3.5:9b":
                        m["digest"] = "sha256:different_digest_9b"
            return tags

        fake_transport.get_tags = dynamic_get_tags
        res = run(self.protocol_path, self.output_path, transport=fake_transport)

        output_data = json.loads(self.output_path.read_text(encoding="utf-8"))
        arm_b_attempts = [a for a in output_data["attempts"] if a["arm_id"] == "B"]
        self.assertEqual(len(arm_b_attempts), 31)
        for a in arm_b_attempts:
            self.assertIs(a["attempted"], False)
            self.assertEqual(a["status"], "unattempted")
            self.assertEqual(a["error_type"], "model_or_digest_unavailable")
            self.assertIn("digest changed", a["error_message"])

        # Score marks execution incomplete
        report = score(self.output_path, self.gold_path)
        self.assertIs(report["is_complete"], False)
        self.assertEqual(report["evaluation_status"], "incomplete")
        self.assertGreater(report["skipped_runs"], 0)

    def test_missing_27b_show_blocks_all_inference(self) -> None:
        fake = FakeOllamaTransport()
        known_9b = fake.show("qwen3.5:9b")
        with patch.object(fake, "show", side_effect=[known_9b, RuntimeError("show unavailable")]):
            with self.assertRaisesRegex(RuntimeError, "Cannot verify model sampling metadata"):
                run(self.protocol_path, self.output_path, transport=fake)
        self.assertEqual(fake.chat_calls, [])
        self.assertFalse(self.output_path.exists())

    def test_effective_sampling_resolution_and_version_default_mismatch(self) -> None:
        """Verifies server version check and sampling field matching between Arm B and Arm D."""
        # 1. Server version mismatch blocks run
        fake_transport_bad_ver = FakeOllamaTransport(version="0.40.10")
        with self.assertRaises(ValueError) as ctx:
            run(self.protocol_path, self.output_path, transport=fake_transport_bad_ver)
        self.assertIn("server version mismatch", str(ctx.exception).lower())

        # 2. Sampling parameters mismatch between B and D blocks Arm D
        show_mismatched = {
            "qwen3.5:9b": {
                "parameters": "num_ctx 8192\ntemperature 0\npresence_penalty 2.0\ntop_k 20\ntop_p 0.95\n",
                "details": {"family": "qwen"},
                "supports_thinking": True,
            },
            "qwen3.8:27b": {
                "parameters": "num_ctx 8192\ntemperature 0\n",
                "details": {"family": "qwen"},
                "supports_thinking": False,
            },
        }
        fake_transport2 = FakeOllamaTransport(show_responses=show_mismatched)
        res = run(self.protocol_path, self.output_path, transport=fake_transport2)

        output_data = json.loads(self.output_path.read_text(encoding="utf-8"))
        arm_d_attempts = [a for a in output_data["attempts"] if a["arm_id"] == "D"]
        self.assertEqual(len(arm_d_attempts), 31)
        for a in arm_d_attempts:
            self.assertIs(a["attempted"], False)
            self.assertEqual(a["status"], "unattempted")
            self.assertEqual(a["error_type"], "effective_sampling_mismatch")

    def test_invalid_http_json_and_nonfinite_raw_survives_checkpoint(self) -> None:
        """Verifies HTTP 200 invalid JSON / non-finite body raises OllamaResponseError and survives checkpoint."""
        def raw_responder(payload: dict) -> str:
            return "This is completely malformed JSON {{"

        fake_transport = FakeOllamaTransport(chat_responder=raw_responder)
        run(self.protocol_path, self.output_path, transport=fake_transport)

        output_data = json.loads(self.output_path.read_text(encoding="utf-8"))
        att = output_data["attempts"][0]
        self.assertIs(att["attempted"], True)
        self.assertEqual(att["status"], "error")
        self.assertEqual(att["error_type"], "OllamaResponseError")
        self.assertIsNotNone(att["raw_response"])
        self.assertIn("raw_text", att["raw_response"])
        self.assertEqual(att["raw_response"]["raw_text"], "This is completely malformed JSON {{")
        self.assertIs(att["raw_response"]["is_truncated"], False)

    def test_missing_bool_negative_prompt_counts_never_success(self) -> None:
        """Verifies missing, boolean, negative, or invalid prompt_eval_count results in technical error with raw preserved."""
        for invalid_val in [None, True, False, -1, "invalid_str"]:
            out_p = self.protocol_dir / f"test_prompt_{invalid_val}.json"

            def responder(payload: dict) -> dict:
                resp = FakeOllamaTransport().chat(payload)
                resp["prompt_eval_count"] = invalid_val
                return resp

            fake_transport = FakeOllamaTransport(chat_responder=responder)
            run(self.protocol_path, out_p, transport=fake_transport)

            output_data = json.loads(out_p.read_text(encoding="utf-8"))
            att = output_data["attempts"][0]
            self.assertIs(att["attempted"], True)
            self.assertEqual(att["status"], "error")
            self.assertIn("prompt_eval_count", att["error_message"])
            self.assertIsNotNone(att["raw_response"])

    def test_removed_code_pin_before_network(self) -> None:
        """Verifies removing any required code pin or the code dict fails loudly before any network calls."""
        fake_transport = FakeOllamaTransport()

        # 1. Remove one required code pin ("grounding")
        prepare(self.protocol_dir / "missing_pin_dir")
        p_path = self.protocol_dir / "missing_pin_dir" / "protocol.json"
        p_data = json.loads(p_path.read_text(encoding="utf-8"))
        del p_data["hashes"]["code"]["grounding"]
        p_path.write_text(json.dumps(p_data), encoding="utf-8")

        with self.assertRaises(ValueError) as ctx:
            run(p_path, self.output_path, transport=fake_transport)
        self.assertIn("Code pins mismatch", str(ctx.exception))
        self.assertEqual(fake_transport.tags_calls, 0)
        self.assertEqual(fake_transport.version_calls, 0)

        # 2. Remove entire code section
        del p_data["hashes"]["code"]
        p_path.write_text(json.dumps(p_data), encoding="utf-8")
        with self.assertRaises(ValueError) as ctx2:
            run(p_path, self.output_path, transport=fake_transport)
        self.assertIn("code", str(ctx2.exception))
        self.assertEqual(fake_transport.tags_calls, 0)
        self.assertEqual(fake_transport.version_calls, 0)

        # 3. Tamper defaults canonical sha256
        prepare(self.protocol_dir / "tampered_defaults_dir")
        p_path3 = self.protocol_dir / "tampered_defaults_dir" / "protocol.json"
        p_data3 = json.loads(p_path3.read_text(encoding="utf-8"))
        p_data3["hashes"]["defaults_canonical_sha256"] = "0" * 64
        p_path3.write_text(json.dumps(p_data3), encoding="utf-8")
        with self.assertRaises(ValueError) as ctx3:
            run(p_path3, self.output_path, transport=fake_transport)
        self.assertIn("defaults canonical SHA-256", str(ctx3.exception))
        self.assertEqual(fake_transport.tags_calls, 0)
        self.assertEqual(fake_transport.version_calls, 0)

    def test_exact_returned_model_required(self) -> None:
        """Verifies startswith alias model acceptance is rejected; exact match required."""
        def alias_responder(payload: dict) -> dict:
            resp = FakeOllamaTransport().chat(payload)
            resp["model"] = f"{payload['model']}:latest"  # alias suffix!
            return resp

        fake_transport = FakeOllamaTransport(chat_responder=alias_responder)
        run(self.protocol_path, self.output_path, transport=fake_transport)

        output_data = json.loads(self.output_path.read_text(encoding="utf-8"))
        att = output_data["attempts"][0]
        self.assertIs(att["attempted"], True)
        self.assertEqual(att["status"], "error")
        self.assertIn("does not match requested", att["error_message"])

    def test_cli_run_interface(self) -> None:
        """Verifies CLI run command parsing and execution."""
        with patch("experiments.compare_verifier_capacity.OllamaTransport") as mock_transport_cls:
            mock_transport_cls.return_value = FakeOllamaTransport()
            ret = main(["run", "--protocol", str(self.protocol_path), "--output", str(self.output_path)])
            self.assertEqual(ret, 0)
            self.assertTrue(self.output_path.is_file())


class TestVerifierCapacityScore(unittest.TestCase):
    """Test suite for Phase 3 verifier capacity offline scoring evaluator."""

    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.protocol_dir = Path(self.tmp_dir.name) / "protocol"
        self.prep_res = prepare(self.protocol_dir)
        self.protocol_path = Path(self.prep_res["protocol_path"])
        self.gold_path = Path(self.prep_res["gold_path"])
        self.results_path = Path(self.tmp_dir.name) / "run_output.json"
        self.score_output_path = Path(self.tmp_dir.name) / "score_report.json"

        # Generate a valid baseline execution results file using FakeOllamaTransport
        self.fake_transport = FakeOllamaTransport()
        self.run_res = run(self.protocol_path, self.results_path, transport=self.fake_transport)

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()

    def test_score_full_mocked_results_complete(self) -> None:
        """Verifies score() processes complete mocked results, outputs valid schema and all arm summaries."""
        report = score(self.results_path, self.gold_path)
        assert_finite(report)

        self.assertEqual(report["schema_version"], SCORE_REPORT_SCHEMA_VERSION)
        self.assertEqual(report["evaluation_status"], "completed")
        self.assertIs(report["is_complete"], True)
        self.assertEqual(set(report["arms"].keys()), {"A", "B", "C", "D"})

        # Verify cross-arm comparisons present
        comparisons = report["comparisons"]
        self.assertIn("budget_effect_B_minus_A", comparisons)
        self.assertIn("thinking_effect_C_minus_B", comparisons)
        self.assertIn("model_config_difference_D_minus_B", comparisons)
        for comp_name in ["budget_effect_B_minus_A", "thinking_effect_C_minus_B", "model_config_difference_D_minus_B"]:
            self.assertIn("accuracy_planned_diff", comparisons[comp_name])
            self.assertIn("accuracy_valid_diff", comparisons[comp_name])
            self.assertIn("description", comparisons[comp_name])

        # Verify disclosures present
        disclosures = report["disclosures"]
        self.assertIn("reused_controls", disclosures)
        self.assertIn("b05_status", disclosures)
        self.assertIn("execution_order_effects", disclosures)
        self.assertIn("vram_and_speed_disclaimer", disclosures)
        self.assertIn("scope_limitation", disclosures)

    def test_score_separates_primary28_and_b05_diagnostic(self) -> None:
        """Verifies primary 28 controls are strictly separated from b05's 3 correlated repeats per arm."""
        report = score(self.results_path, self.gold_path)

        for arm_id in ["A", "B", "C", "D"]:
            arm = report["arms"][arm_id]
            counts = arm["counts"]
            self.assertEqual(counts["planned_total"], 31)
            self.assertEqual(counts["primary_planned"], 28)
            self.assertEqual(counts["primary_attempted"], 28)
            self.assertEqual(counts["primary_valid"], 28)
            self.assertEqual(counts["primary_failed"], 0)
            self.assertEqual(counts["primary_skipped"], 0)

            # Confusion matrix sum must equal 28, not 31 or 29
            cm = arm["confusion_matrix"]
            total_cm_entries = sum(sum(row.values()) for row in cm.values())
            self.assertEqual(total_cm_entries, 28)

            # b05 diagnostic strictly separated
            b05_diag = arm["b05_diagnostic"]
            self.assertEqual(b05_diag["case_id"], "b05")
            self.assertEqual(b05_diag["total_repeats"], 3)
            self.assertEqual(len(b05_diag["repeats"]), 3)
            for rep in b05_diag["repeats"]:
                self.assertIn(rep["repeat_index"], [1, 2, 3])
                self.assertEqual(rep["expected_verdict"], "supported")
                self.assertEqual(rep["status"], "success")

    def test_false_accept_and_false_reject_with_denominator_zeros(self) -> None:
        """Verifies false accept and false reject rates report numerator/denominator, and null when denominator is zero."""
        # Test Case 1: Gold has only "supported" cases (nonsupported denominator == 0)
        gold_data_all_sup = json.loads(self.gold_path.read_text(encoding="utf-8"))
        for cid in gold_data_all_sup["labels"]:
            gold_data_all_sup["labels"][cid] = "supported"
        for c in gold_data_all_sup["cases"]:
            c["expected"] = "supported"

        # Update protocol hash pin to match modified gold
        protocol_data = json.loads(self.protocol_path.read_text(encoding="utf-8"))
        protocol_data["hashes"]["gold_canonical_sha256"] = canonical_sha256(gold_data_all_sup)
        self.protocol_path.write_text(json.dumps(protocol_data, indent=2), encoding="utf-8")

        # Update results protocol hash
        results_data = json.loads(self.results_path.read_text(encoding="utf-8"))
        results_data["protocol_canonical_sha256"] = canonical_sha256(protocol_data)
        self.results_path.write_text(json.dumps(results_data, indent=2), encoding="utf-8")

        sup_gold_path = self.protocol_dir / "gold_all_supported.json"
        sup_gold_path.write_text(json.dumps(gold_data_all_sup, indent=2), encoding="utf-8")

        report_sup = score(self.results_path, sup_gold_path)
        arm_a = report_sup["arms"]["A"]
        # nonsupported count is 0 -> false accepts denominator is 0, rate is None
        self.assertEqual(arm_a["false_accepts_nonsupported"]["denominator"], 0)
        self.assertEqual(arm_a["false_accepts_nonsupported"]["numerator"], 0)
        self.assertIsNone(arm_a["false_accepts_nonsupported"]["rate"])
        # supported count is 28 -> false rejects denominator is 28
        self.assertEqual(arm_a["false_rejects_supported"]["denominator"], 28)

        # Test Case 2: Gold has only "contradicted" cases (supported denominator == 0)
        gold_data_all_contra = json.loads(self.gold_path.read_text(encoding="utf-8"))
        for cid in gold_data_all_contra["labels"]:
            gold_data_all_contra["labels"][cid] = "contradicted"
        for c in gold_data_all_contra["cases"]:
            c["expected"] = "contradicted"

        protocol_data["hashes"]["gold_canonical_sha256"] = canonical_sha256(gold_data_all_contra)
        self.protocol_path.write_text(json.dumps(protocol_data, indent=2), encoding="utf-8")
        results_data["protocol_canonical_sha256"] = canonical_sha256(protocol_data)
        self.results_path.write_text(json.dumps(results_data, indent=2), encoding="utf-8")

        contra_gold_path = self.protocol_dir / "gold_all_contra.json"
        contra_gold_path.write_text(json.dumps(gold_data_all_contra, indent=2), encoding="utf-8")

        report_contra = score(self.results_path, contra_gold_path)
        arm_a_contra = report_contra["arms"]["A"]
        # supported count is 0 -> false rejects denominator is 0, rate is None
        self.assertEqual(arm_a_contra["false_rejects_supported"]["denominator"], 0)
        self.assertEqual(arm_a_contra["false_rejects_supported"]["numerator"], 0)
        self.assertIsNone(arm_a_contra["false_rejects_supported"]["rate"])
        # nonsupported count is 28 -> false accepts denominator is 28
        self.assertEqual(arm_a_contra["false_accepts_nonsupported"]["denominator"], 28)

    def test_score_rejects_malformed_and_nonfinite_json(self) -> None:
        """Verifies score() raises ValueError for malformed JSON and non-finite floats."""
        # 1. Malformed JSON in results
        malformed_results = self.protocol_dir / "malformed_results.json"
        malformed_results.write_text("not json {{{", encoding="utf-8")
        with self.assertRaises(ValueError):
            score(malformed_results, self.gold_path)

        # 2. Malformed JSON in gold
        malformed_gold = self.protocol_dir / "malformed_gold.json"
        malformed_gold.write_text("not json {{{", encoding="utf-8")
        with self.assertRaises(ValueError):
            score(self.results_path, malformed_gold)

        # 3. Non-finite float (NaN) in results
        nan_results_data = json.loads(self.results_path.read_text(encoding="utf-8"))
        nan_results = self.protocol_dir / "nan_results.json"
        nan_results.write_text(
            json.dumps(nan_results_data).replace('"total_scheduled": 124', '"total_scheduled": NaN'),
            encoding="utf-8",
        )
        with self.assertRaises(ValueError):
            score(nan_results, self.gold_path)

        # 4. Non-finite float (Infinity) in gold
        inf_gold_data = json.loads(self.gold_path.read_text(encoding="utf-8"))
        inf_gold = self.protocol_dir / "inf_gold.json"
        inf_gold.write_text(
            json.dumps(inf_gold_data).replace('"case_count": 29', '"case_count": Infinity'),
            encoding="utf-8",
        )
        with self.assertRaises(ValueError):
            score(self.results_path, inf_gold)

    def test_score_rejects_unknown_labels(self) -> None:
        """Verifies unknown labels in gold or predicted verdicts fail loudly."""
        # Unknown label in gold
        bad_gold_data = json.loads(self.gold_path.read_text(encoding="utf-8"))
        bad_gold_data["labels"]["d01"] = "unknown_semantic_label"
        bad_gold = self.protocol_dir / "bad_gold.json"
        bad_gold.write_text(json.dumps(bad_gold_data), encoding="utf-8")
        with self.assertRaises(ValueError) as ctx:
            score(self.results_path, bad_gold)
        self.assertIn("Unknown gold expected label", str(ctx.exception))

        # Unknown label in attempt predicted verdict
        bad_results_data = json.loads(self.results_path.read_text(encoding="utf-8"))
        bad_results_data["attempts"][0]["parsed_verdict"]["verdicts"][0]["verdict"] = "abstain"
        bad_results = self.protocol_dir / "bad_results.json"
        bad_results.write_text(json.dumps(bad_results_data), encoding="utf-8")
        with self.assertRaises(ValueError) as ctx2:
            score(bad_results, self.gold_path)
        self.assertIn("Unknown predicted label", str(ctx2.exception))

    def test_score_rejects_tampered_gold_or_protocol_digest(self) -> None:
        """Verifies tampering of gold or protocol canonical SHA-256 is detected."""
        # Tampered gold
        tampered_gold_data = json.loads(self.gold_path.read_text(encoding="utf-8"))
        tampered_gold_data["labels"]["d01"] = "contradicted"  # modified label changes canonical sha256
        tampered_gold = self.protocol_dir / "tampered_gold.json"
        tampered_gold.write_text(json.dumps(tampered_gold_data), encoding="utf-8")
        with self.assertRaises(ValueError) as ctx:
            score(self.results_path, tampered_gold)
        self.assertIn("gold.json canonical SHA-256", str(ctx.exception))

        # Tampered protocol
        tampered_protocol_data = json.loads(self.protocol_path.read_text(encoding="utf-8"))
        tampered_protocol_data["design_goal"] = "tampered protocol"
        self.protocol_path.write_text(json.dumps(tampered_protocol_data), encoding="utf-8")
        with self.assertRaises(ValueError) as ctx2:
            score(self.results_path, self.gold_path)
        self.assertIn("protocol.json canonical SHA-256", str(ctx2.exception))

    def test_score_rejects_duplicate_and_unknown_events(self) -> None:
        """Verifies duplicate attempts, duplicate schedule indices, or unknown arms/cases raise ValueError."""
        # 1. Duplicate attempt event
        results_data = json.loads(self.results_path.read_text(encoding="utf-8"))
        dup_attempt = copy.deepcopy(results_data["attempts"][0])
        results_data["attempts"].append(dup_attempt)
        dup_results = self.protocol_dir / "dup_results.json"
        dup_results.write_text(json.dumps(results_data), encoding="utf-8")
        with self.assertRaises(ValueError) as ctx:
            score(dup_results, self.gold_path)
        self.assertIn("Duplicate attempt event", str(ctx.exception))

        # 2. Unknown arm_id
        results_unknown_arm = json.loads(self.results_path.read_text(encoding="utf-8"))
        results_unknown_arm["attempts"][0]["arm_id"] = "Z"
        unk_arm_results = self.protocol_dir / "unk_arm_results.json"
        unk_arm_results.write_text(json.dumps(results_unknown_arm), encoding="utf-8")
        with self.assertRaises(ValueError) as ctx2:
            score(unk_arm_results, self.gold_path)
        self.assertIn("Unknown arm_id", str(ctx2.exception))

        # 3. Unknown case_id
        results_unknown_case = json.loads(self.results_path.read_text(encoding="utf-8"))
        results_unknown_case["attempts"][0]["case_id"] = "unknown_case_999"
        unk_case_results = self.protocol_dir / "unk_case_results.json"
        unk_case_results.write_text(json.dumps(results_unknown_case), encoding="utf-8")
        with self.assertRaises(ValueError) as ctx3:
            score(unk_case_results, self.gold_path)
        self.assertIn("Unknown case_id", str(ctx3.exception))

    def test_score_reports_incomplete_execution_never_completed(self) -> None:
        """Verifies incomplete executions report incomplete with planned/attempted/missing counts, never completed."""
        # 1. Truncated execution (e.g. 50 attempts instead of 124)
        trunc_data = json.loads(self.results_path.read_text(encoding="utf-8"))
        trunc_data["attempts"] = trunc_data["attempts"][:50]
        trunc_data["completed_at"] = None
        trunc_results = self.protocol_dir / "trunc_results.json"
        trunc_results.write_text(json.dumps(trunc_data), encoding="utf-8")

        report = score(trunc_results, self.gold_path)
        self.assertIs(report["is_complete"], False)
        self.assertEqual(report["evaluation_status"], "incomplete")
        self.assertEqual(report["planned_runs"], 124)
        self.assertEqual(report["recorded_runs"], 50)
        self.assertEqual(report["missing_runs"], 74)
        self.assertIn("Refusing to score partial execution", report["message"])

        # 2. All 124 attempts present, but completed_at is None (in-flight crash)
        unfin_data = json.loads(self.results_path.read_text(encoding="utf-8"))
        unfin_data["completed_at"] = None
        unfin_results = self.protocol_dir / "unfin_results.json"
        unfin_results.write_text(json.dumps(unfin_data), encoding="utf-8")

        report_unfin = score(unfin_results, self.gold_path)
        self.assertIs(report_unfin["is_complete"], False)
        self.assertEqual(report_unfin["evaluation_status"], "incomplete")

    def test_technical_errors_after_real_attempts_separately_scored_not_abstentions(self) -> None:
        """Verifies technical errors after real attempts are counted explicitly in primary_failed and not mapped to abstention."""
        # Inject technical errors after real attempts
        res_data = json.loads(self.results_path.read_text(encoding="utf-8"))
        # Set 2 attempts in Arm A to error (attempted = True)
        arm_a_attempts = [a for a in res_data["attempts"] if a["arm_id"] == "A" and a["case_id"] != "b05"]
        arm_a_attempts[0]["status"] = "error"
        arm_a_attempts[0]["attempted"] = True
        arm_a_attempts[0]["error_type"] = "TimeoutError"
        arm_a_attempts[0]["error_message"] = "Connection timed out"
        arm_a_attempts[0]["parsed_verdict"] = None

        arm_a_attempts[1]["status"] = "error"
        arm_a_attempts[1]["attempted"] = True
        arm_a_attempts[1]["error_type"] = "ContextBudgetExceeded"
        arm_a_attempts[1]["error_message"] = "Prompt exceeds context budget"
        arm_a_attempts[1]["parsed_verdict"] = None

        err_results = self.protocol_dir / "err_results.json"
        err_results.write_text(json.dumps(res_data), encoding="utf-8")

        report = score(err_results, self.gold_path)
        self.assertIs(report["is_complete"], True)
        arm_a = report["arms"]["A"]

        # Validate explicit count isolation
        counts = arm_a["counts"]
        self.assertEqual(counts["primary_planned"], 28)
        self.assertEqual(counts["primary_attempted"], 28)
        self.assertEqual(counts["primary_valid"], 26)
        self.assertEqual(counts["primary_failed"], 2)
        self.assertEqual(counts["primary_skipped"], 0)

        # Technical errors never counted as abstention ("insufficient_evidence")
        # Sum of confusion matrix rows is exactly primary_valid (26)
        cm_sum = sum(sum(row.values()) for row in arm_a["confusion_matrix"].values())
        self.assertEqual(cm_sum, 26)

        # Accuracy planned evaluates against planned 28, not valid 26
        acc = arm_a["accuracy"]
        self.assertEqual(acc["correct_planned_ratio"], f"{acc['correct_count']}/28")
        self.assertEqual(acc["correct_valid_ratio"], f"{acc['correct_count']}/26")
        self.assertAlmostEqual(acc["accuracy_planned"], acc["correct_count"] / 28)
        self.assertAlmostEqual(acc["accuracy_valid"], acc["correct_count"] / 26)
        self.assertAlmostEqual(acc["technical_error_rate"], 2 / 28)

    def test_full_124_roster_with_skipped_arm_remains_incomplete(self) -> None:
        """Verifies full 124 roster with skipped/unattempted arm remains incomplete without completed metrics."""
        res_data = json.loads(self.results_path.read_text(encoding="utf-8"))
        for a in res_data["attempts"]:
            if a["arm_id"] == "D":
                a["attempted"] = False
                a["status"] = "unattempted"
                a["error_type"] = "model_or_digest_unavailable"
                a["parsed_verdict"] = None

        res_data["completion_status"] = "incomplete/skipped"
        skipped_results = self.protocol_dir / "skipped_results.json"
        skipped_results.write_text(json.dumps(res_data), encoding="utf-8")

        report = score(skipped_results, self.gold_path)
        self.assertIs(report["is_complete"], False)
        self.assertEqual(report["evaluation_status"], "incomplete")
        self.assertEqual(report["completion_status"], "incomplete/skipped")
        self.assertEqual(report["planned_runs"], 124)
        self.assertEqual(report["recorded_runs"], 124)
        self.assertEqual(report["attempted_runs"], 93)
        self.assertEqual(report["skipped_runs"], 31)
        self.assertEqual(report["missing_runs"], 0)
        self.assertNotIn("arms", report)

    def test_score_rejects_missing_pins(self) -> None:
        """Verifies score fails loudly when protocol or results pins are missing or invalid hex."""
        # 1. Missing protocol_canonical_sha256 in results
        res_no_pin = json.loads(self.results_path.read_text(encoding="utf-8"))
        del res_no_pin["protocol_canonical_sha256"]
        p1 = self.protocol_dir / "p1.json"
        p1.write_text(json.dumps(res_no_pin), encoding="utf-8")
        with self.assertRaises(ValueError) as ctx:
            score(p1, self.gold_path)
        self.assertIn("protocol_canonical_sha256", str(ctx.exception))

        # 2. Missing gold_canonical_sha256 in protocol
        prepare(self.protocol_dir / "no_gold_pin")
        prot_no_gold = self.protocol_dir / "no_gold_pin" / "protocol.json"
        p_data = json.loads(prot_no_gold.read_text(encoding="utf-8"))
        del p_data["hashes"]["gold_canonical_sha256"]
        prot_no_gold.write_text(json.dumps(p_data), encoding="utf-8")
        with self.assertRaises(ValueError) as ctx2:
            score(self.results_path, self.protocol_dir / "no_gold_pin" / "gold.json")
        self.assertIn("gold_canonical_sha256", str(ctx2.exception))

    def test_score_rejects_impossible_repeat_index_and_reassigned_schedule_index(self) -> None:
        """Verifies score rejects impossible repeat_index and reassigned schedule indices."""
        # 1. Impossible repeat index for primary case
        res_data1 = json.loads(self.results_path.read_text(encoding="utf-8"))
        res_data1["attempts"][0]["repeat_index"] = 2
        f1 = self.protocol_dir / "f1.json"
        f1.write_text(json.dumps(res_data1), encoding="utf-8")
        with self.assertRaises(ValueError) as ctx:
            score(f1, self.gold_path)
        self.assertIn("Impossible repeat_index", str(ctx.exception))

        # 2. Reassigned schedule_index
        res_data2 = json.loads(self.results_path.read_text(encoding="utf-8"))
        target_case = "d02" if res_data2["attempts"][0]["case_id"] != "d02" else "d03"
        res_data2["attempts"][0]["case_id"] = target_case
        f2 = self.protocol_dir / "f2.json"
        f2.write_text(json.dumps(res_data2), encoding="utf-8")
        with self.assertRaises(ValueError) as ctx2:
            score(f2, self.gold_path)
        self.assertIn("Reassigned schedule_index", str(ctx2.exception))

    def test_cli_score_interface_and_no_overwrite(self) -> None:
        """Verifies CLI score command generates report and strictly refuses to overwrite existing output."""
        # 1. Run CLI score
        ret = main([
            "score",
            "--results", str(self.results_path),
            "--gold", str(self.gold_path),
            "--output", str(self.score_output_path),
        ])
        self.assertEqual(ret, 0)
        self.assertTrue(self.score_output_path.is_file())

        report_data = json.loads(self.score_output_path.read_text(encoding="utf-8"))
        assert_finite(report_data)
        self.assertEqual(report_data["schema_version"], SCORE_REPORT_SCHEMA_VERSION)

        # 2. Refuses existing output file
        with self.assertRaises(FileExistsError) as ctx:
            main([
                "score",
                "--results", str(self.results_path),
                "--gold", str(self.gold_path),
                "--output", str(self.score_output_path),
            ])
        self.assertIn("already exists", str(ctx.exception))

    def test_performance_metrics_and_snapshots(self) -> None:
        """Verifies latency, load duration, tokens, thinking length and placement snapshot metrics."""
        report = score(self.results_path, self.gold_path)
        arm_c = report["arms"]["C"]
        metrics = arm_c["performance_metrics"]

        # Latencies & tokens
        self.assertIsNotNone(metrics["latency_seconds"]["median"])
        self.assertIsNotNone(metrics["latency_seconds"]["range"])
        self.assertIsNotNone(metrics["input_tokens"]["median"])
        self.assertIsNotNone(metrics["output_tokens"]["median"])

        # Thinking characters (Arm C think=True)
        self.assertGreater(metrics["thinking_characters"]["attempts_with_thinking"], 0)
        self.assertGreater(metrics["thinking_characters"]["median"], 0)

        # Arm A think=False has 0 thinking attempts
        arm_a = report["arms"]["A"]
        self.assertEqual(arm_a["performance_metrics"]["thinking_characters"]["attempts_with_thinking"], 0)

        # Placement snapshot captured
        self.assertIsNotNone(arm_c["placement_snapshot"])
        self.assertIn("model", arm_c["placement_snapshot"])
        self.assertIn("ps_placement", arm_c["placement_snapshot"])


if __name__ == "__main__":
    unittest.main()
