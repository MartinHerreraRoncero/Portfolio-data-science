"""Offline unit tests for the isolated query coverage evaluation gate runner."""

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from experiments.evaluate_coverage_gate import (
    GATE_PROTOCOL_SCHEMA_VERSION,
    GATE_RESULTS_SCHEMA_VERSION,
    REQUIRED_SERVER_VERSION,
    REQUIRED_FROZEN_MODULES,
    main as gate_main,
    run,
    validate_gate_inputs,
    validate_protocol,
)
from experiments.preserved_q3_verifier import (
    Q3VerifierClient,
    VERIFIER_Q3_DIGEST,
    VERIFIER_Q3_MODEL,
    VERIFIER_Q3_PARAMETERS,
)
from rag_pensiones.ingest import PROJECT_ROOT


class MockGateTransport:
    """Mock transport for gate runner testing."""

    def __init__(self, chat_responses: list[dict] | None = None) -> None:
        self.chat_responses = list(chat_responses or [])
        self.calls: list[dict] = []
        self.call_idx = 0

    def get_version(self) -> dict:
        return {"version": REQUIRED_SERVER_VERSION}

    def get_tags(self) -> dict:
        return {
            "models": [
                {
                    "name": VERIFIER_Q3_MODEL,
                    "model": VERIFIER_Q3_MODEL,
                    "digest": VERIFIER_Q3_DIGEST,
                    "details": {"quantization_level": "Q3_K_M"},
                }
            ]
        }

    def show(self, model: str) -> dict:
        return {
            "details": {"quantization_level": "Q3_K_M"},
            "parameters": "presence_penalty 1.5\ntop_k 20\ntop_p 0.95\nmin_p 0.0\n",
        }

    def get_ps(self) -> dict:
        return {"models": []}

    def unload(self, model: str) -> dict:
        return {"status": "success"}

    def chat(self, request: dict) -> dict:
        self.calls.append(copy.deepcopy(request))
        if self.call_idx < len(self.chat_responses):
            resp = self.chat_responses[self.call_idx]
            self.call_idx += 1
            if isinstance(resp, Exception):
                raise resp
            return copy.deepcopy(resp)
        raise RuntimeError("No mock response configured for chat call")


def _make_mock_response(content_payload: dict) -> dict:
    return {
        "model": VERIFIER_Q3_MODEL,
        "done": True,
        "done_reason": "stop",
        "prompt_eval_count": 300,
        "eval_count": 70,
        "prompt_eval_duration": 100000000,
        "eval_duration": 200000000,
        "total_duration": 350000000,
        "load_duration": 10000000,
        "message": {
            "role": "assistant",
            "content": json.dumps(content_payload, ensure_ascii=False),
        },
    }


class TestGateProtocolValidation(unittest.TestCase):
    """Verifies strict protocol validation, frozen file hashes, and gold label isolation."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

        # Create dummy input file
        self.input_file = self.temp_path / "inputs.json"
        inputs_content = json.dumps({
            "cases": [
                {
                    "id": "c01",
                    "query": "¿Consulta de prueba?",
                    "inventory": [],
                }
            ]
        }, ensure_ascii=False).encode("utf-8")
        self.input_file.write_bytes(inputs_content)
        self.input_sha = hashlib.sha256(inputs_content).hexdigest()

        # Relative path within project or temp
        self.rel_input_path = self.input_file.relative_to(PROJECT_ROOT).as_posix() if self.input_file.is_relative_to(PROJECT_ROOT) else str(self.input_file)

        # Sample valid frozen file from repo
        ref_file = PROJECT_ROOT / "experiments" / "query_coverage.py"
        ref_sha = hashlib.sha256(ref_file.read_bytes()).hexdigest()

        self.valid_protocol = {
            "schema_version": GATE_PROTOCOL_SCHEMA_VERSION,
            "server_version": REQUIRED_SERVER_VERSION,
            "verifier_digest": VERIFIER_Q3_DIGEST,
            "verifier_parameters": copy.deepcopy(VERIFIER_Q3_PARAMETERS),
            "input_path": "experiments/query_coverage.py",
            "input_sha256": ref_sha,
            "frozen_files": {
                "experiments/query_coverage.py": ref_sha,
            },
        }
        self.valid_protocol["frozen_files"].update({
            rel: hashlib.sha256((PROJECT_ROOT / rel).read_bytes()).hexdigest()
            for rel in REQUIRED_FROZEN_MODULES
        })

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_valid_protocol_passes(self):
        validate_protocol(self.valid_protocol)

    def test_every_required_source_pin_is_mandatory(self):
        for rel in REQUIRED_FROZEN_MODULES - {self.valid_protocol["input_path"]}:
            with self.subTest(rel=rel):
                protocol = copy.deepcopy(self.valid_protocol)
                del protocol["frozen_files"][rel]
                with self.assertRaisesRegex(ValueError, "Required dependency code pins"):
                    validate_protocol(protocol)

    def test_protocol_missing_keys_fails(self):
        bad_protocol = copy.deepcopy(self.valid_protocol)
        del bad_protocol["server_version"]
        with self.assertRaises(ValueError) as ctx:
            validate_protocol(bad_protocol)
        self.assertIn("missing required keys", str(ctx.exception))

    def test_protocol_wrong_server_version_fails(self):
        bad_protocol = copy.deepcopy(self.valid_protocol)
        bad_protocol["server_version"] = "0.39.0"
        with self.assertRaises(ValueError) as ctx:
            validate_protocol(bad_protocol)
        self.assertIn("Server version mismatch", str(ctx.exception))

    def test_protocol_wrong_verifier_digest_fails(self):
        bad_protocol = copy.deepcopy(self.valid_protocol)
        bad_protocol["verifier_digest"] = "0" * 64
        with self.assertRaises(ValueError) as ctx:
            validate_protocol(bad_protocol)
        self.assertIn("Verifier digest mismatch", str(ctx.exception))

    def test_gold_access_forbidden_in_protocol(self):
        bad_protocol = copy.deepcopy(self.valid_protocol)
        bad_protocol["input_path"] = "evaluation/gold.json"
        with self.assertRaises(ValueError) as ctx:
            validate_protocol(bad_protocol)
        self.assertIn("Gold paths are strictly forbidden", str(ctx.exception))

    def test_gold_file_in_frozen_files_forbidden(self):
        bad_protocol = copy.deepcopy(self.valid_protocol)
        bad_protocol["frozen_files"]["evaluation/gold.json"] = "a" * 64
        with self.assertRaises(ValueError) as ctx:
            validate_protocol(bad_protocol)
        self.assertIn("Gold labels must not be read or hashed", str(ctx.exception))

    def test_input_path_missing_from_frozen_files_fails(self):
        bad_protocol = copy.deepcopy(self.valid_protocol)
        bad_protocol["frozen_files"] = {}
        with self.assertRaises(ValueError) as ctx:
            validate_protocol(bad_protocol)
        self.assertIn("frozen_files must be a non-empty dict", str(ctx.exception))


class TestGateInputsValidation(unittest.TestCase):
    """Verifies strict validation of input case roster."""

    def test_valid_case_passes(self):
        data = {
            "cases": [
                {
                    "id": "c01",
                    "query": "¿Año?",
                    "inventory": [
                        {
                            "claim_id": "C1",
                            "text": "2015",
                            "evidence": [{"source_id": "S1", "quote": "Texto"}],
                            "publication_claim": {"text": "2015", "evidence": []},
                        }
                    ],
                }
            ]
        }
        cases = validate_gate_inputs(data)
        self.assertEqual(len(cases), 1)

    def test_case_with_extra_keys_fails(self):
        data = {
            "cases": [
                {
                    "id": "c01",
                    "query": "¿Año?",
                    "inventory": [],
                    "extra_label": "gold_answer",
                }
            ]
        }
        with self.assertRaises(ValueError) as ctx:
            validate_gate_inputs(data)
        self.assertIn("keys mismatch", str(ctx.exception))

    def test_duplicate_case_id_fails(self):
        data = {
            "cases": [
                {"id": "c01", "query": "¿Año?", "inventory": []},
                {"id": "c01", "query": "¿Otro año?", "inventory": []},
            ]
        }
        with self.assertRaises(ValueError) as ctx:
            validate_gate_inputs(data)
        self.assertIn("Duplicate case id", str(ctx.exception))


class TestGateExecution(unittest.TestCase):
    """Verifies execution, hot client reuse, per-case call isolation, and no-overwrite."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

        # Setup 2 cases: one with inventory, one empty
        self.inputs_data = {
            "cases": [
                {
                    "id": "c01",
                    "query": "¿Con qué año se compara?",
                    "inventory": [
                        {
                            "claim_id": "C1",
                            "text": "Se compara con 2015.",
                            "evidence": [{"source_id": "S1", "quote": "Cita"}],
                            "publication_claim": {"text": "Se compara con 2015.", "evidence": [{"source_id": "S1", "quote": "Cita"}]},
                        }
                    ],
                },
                {
                    "id": "c02",
                    "query": "¿Pregunta sin evidencias evaluadas?",
                    "inventory": [],
                },
            ]
        }
        self.inputs_file = self.temp_path / "gate_inputs.json"
        inputs_bytes = json.dumps(self.inputs_data, ensure_ascii=False).encode("utf-8")
        self.inputs_file.write_bytes(inputs_bytes)
        self.inputs_sha = hashlib.sha256(inputs_bytes).hexdigest()

        resp_c01 = _make_mock_response({
            "parts": [
                {
                    "query_span": "¿Con qué año se compara?",
                    "decision": "answered",
                    "claim_ids": ["C1"],
                    "reason_code": "covered",
                }
            ]
        })
        self.transport = MockGateTransport(chat_responses=[resp_c01])
        self.verifier = Q3VerifierClient(expected_digest=VERIFIER_Q3_DIGEST, transport=self.transport)
        rel = self.inputs_file.as_posix()
        self.protocol = {
            "schema_version": GATE_PROTOCOL_SCHEMA_VERSION,
            "server_version": REQUIRED_SERVER_VERSION,
            "verifier_digest": VERIFIER_Q3_DIGEST,
            "verifier_parameters": copy.deepcopy(VERIFIER_Q3_PARAMETERS),
            "input_path": rel, "input_sha256": self.inputs_sha,
            "frozen_files": {p: hashlib.sha256((PROJECT_ROOT / p).read_bytes()).hexdigest()
                             for p in REQUIRED_FROZEN_MODULES},
        }
        self.protocol["frozen_files"][rel] = self.inputs_sha

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_no_overwrite_protection(self):
        out_file = self.temp_path / "results.json"
        out_file.write_text("existing content", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Output already exists"):
            run(self.inputs_file, out_file, verifier=self.verifier, protocol=self.protocol)

    def test_no_protocol_rejected_before_client_creation(self):
        with patch("experiments.evaluate_coverage_gate.Q3VerifierClient") as client:
            with self.assertRaisesRegex(ValueError, "Frozen protocol required"):
                run(self.inputs_file, self.temp_path / "unfrozen.json")
            client.assert_not_called()
        self.assertEqual(self.verifier.calls, [])

    def test_run_executes_hot_and_isolates_calls(self):
        out_file = self.temp_path / "results.json"
        report = run(self.inputs_file, out_file, verifier=self.verifier, protocol=self.protocol)

        self.assertEqual(report["schema_version"], GATE_RESULTS_SCHEMA_VERSION)
        self.assertEqual(report["run_status"], "completed")
        self.assertEqual(len(report["cases"]), 2)

        # Case 1: non-empty inventory -> 1 call made
        case1 = report["cases"][0]
        self.assertEqual(case1["id"], "c01")
        self.assertEqual(case1["status"], "answered")
        self.assertEqual(len(case1["calls"]), 1)
        self.assertEqual(case1["calls"][0]["stage"], "query_coverage")

        # Case 2: empty inventory -> 0 calls made (deterministic skip)
        case2 = report["cases"][1]
        self.assertEqual(case2["id"], "c02")
        self.assertEqual(case2["status"], "insufficient_evidence")
        self.assertEqual(len(case2["calls"]), 0)
        self.assertTrue(case2["audit"]["skipped"])

        # Check total calls on hot verifier
        self.assertEqual(len(self.verifier.calls), 1)

        # Verify output checkpoint exists on disk
        saved = json.loads(out_file.read_text(encoding="utf-8"))
        self.assertEqual(saved["run_status"], "completed")

    def test_run_handles_case_error_without_retries(self):
        # Setup case that causes error in LLM
        err_transport = MockGateTransport(chat_responses=[RuntimeError("Ollama timeout")])
        err_verifier = Q3VerifierClient(expected_digest=VERIFIER_Q3_DIGEST, transport=err_transport)

        out_file = self.temp_path / "results_err.json"
        report = run(self.inputs_file, out_file, verifier=err_verifier, protocol=self.protocol)

        self.assertEqual(report["run_status"], "error")
        self.assertEqual(len(report["cases"]), 2)
        self.assertEqual(report["cases"][0]["status"], "error")
        self.assertIn("Ollama timeout", report["cases"][0]["error"])
        # Exactly 1 call was attempted (no infinite retries)
        self.assertEqual(len(err_verifier.calls), 1)


if __name__ == "__main__":
    unittest.main()
