"""Comprehensive unit tests for Q3 verifier integration, role-pinned validation, and stage truthfulness."""

import copy
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from experiments.compare_verifier_capacity import OllamaResponseError

from experiments.compare_system import generate_arm
from experiments.preserved_q3_verifier import (
    ALLOWED_RELEASE_MODELS,
    Q3PreservedGrounder,
    Q3VerifierClient,
)
from rag_pensiones.answer import (
    DEFAULT_GENERATION_MODEL,
    MAX_MESSAGES_BYTES,
)
from rag_pensiones.chunking import ChunkRecord
from rag_pensiones.grounding import (
    DEFAULT_PARAMETERS,
    MAX_MESSAGES_BYTES_VERIFICATION,
    VERIFIER_Q3_DIGEST,
    VERIFIER_Q3_MODEL,
    VERIFIER_Q3_PARAMETERS,
    validate_grounded_result,
)
from rag_pensiones.search import SearchHit


TEST_9B_DIGEST = "a" * 64
TEST_Q3_DIGEST = VERIFIER_Q3_DIGEST


class FakeTransport:
    """Mock transport simulating Ollama API for unit testing."""

    def __init__(
        self,
        *,
        version: str = "0.40.11",
        q3_digest: str = VERIFIER_Q3_DIGEST,
        quantization_level: str = "Q3_K_M",
        show_quant: str = "Q3_K_M",
        show_parameters: str | None = "presence_penalty 1.5\ntop_k 20\ntop_p 0.95\nmin_p 0.0\n",
        resident_models: list[str] | None = None,
        chat_response: dict | None = None,
        chat_error: Exception | None = None,
        drift_after: str | None = None,
    ) -> None:
        self.version = version
        self.q3_digest = q3_digest
        self.quantization_level = quantization_level
        self.show_quant = show_quant
        self.show_parameters = show_parameters
        self.resident_models = list(resident_models or [])
        self.unloaded_models: list[str] = []
        self.chat_response = chat_response
        self.chat_error = chat_error
        self.drift_after = drift_after
        self.chat_calls: list[dict] = []
        self.chat_count = 0

    def get_version(self) -> dict:
        return {"version": self.version}

    def get_tags(self) -> dict:
        digest = self.drift_after if (self.drift_after and self.chat_count > 0) else self.q3_digest
        return {
            "models": [
                {
                    "name": VERIFIER_Q3_MODEL,
                    "model": VERIFIER_Q3_MODEL,
                    "digest": digest,
                    "details": {"quantization_level": self.quantization_level},
                },
                {
                    "name": DEFAULT_GENERATION_MODEL,
                    "model": DEFAULT_GENERATION_MODEL,
                    "digest": TEST_9B_DIGEST,
                    "details": {"quantization_level": "Q4_K_M"},
                },
            ]
        }

    def show(self, model: str) -> dict:
        return {
            "details": {"quantization_level": self.show_quant},
            "parameters": self.show_parameters,
        }

    def get_ps(self) -> dict:
        return {"models": [{"name": m, "model": m} for m in self.resident_models]}

    def unload(self, model: str) -> dict:
        self.unloaded_models.append(model)
        if model in self.resident_models:
            self.resident_models.remove(model)
        return {"status": "unloaded"}

    def chat(self, request: dict) -> dict:
        self.chat_calls.append(copy.deepcopy(request))
        self.chat_count += 1
        if self.chat_error is not None:
            raise self.chat_error
        if self.chat_response is not None:
            return copy.deepcopy(self.chat_response)
        # Default valid verifier response
        verdicts_payload = {
            "verdicts": [
                {
                    "claim_id": "C1",
                    "verdict": "supported",
                    "reason": "Afirmación respaldada por la evidencia",
                    "evidence_verdicts": [
                        {"evidence_id": "E1", "relation": "supports"}
                    ],
                }
            ]
        }
        return {
            "model": VERIFIER_Q3_MODEL,
            "done": True,
            "done_reason": "stop",
            "message": {
                "role": "assistant",
                "content": json.dumps(verdicts_payload),
            },
            "prompt_eval_count": 200,
            "eval_count": 60,
            "prompt_eval_duration": 50000000,
            "eval_duration": 150000000,
            "total_duration": 250000000,
            "load_duration": 10000000,
        }


class Fake9BGenerator:
    """Mock 9B generator for legacy stages."""

    def __init__(self, responses: list[dict] | None = None) -> None:
        self.model = DEFAULT_GENERATION_MODEL
        self.digest = TEST_9B_DIGEST
        self.calls: list[dict] = []
        self.responses = list(responses or [])
        self.call_idx = 0

    def get_digest(self) -> str:
        return self.digest

    def chat(self, messages: list[dict], response_schema: dict) -> dict:
        call = {"messages": copy.deepcopy(messages), "schema": copy.deepcopy(response_schema)}
        self.calls.append(call)
        if self.call_idx < len(self.responses):
            resp = self.responses[self.call_idx]
            self.call_idx += 1
            if isinstance(resp, Exception):
                call["error"] = str(resp)
                raise resp
            call["raw_response"] = copy.deepcopy(resp)
            return copy.deepcopy(resp)
        raise RuntimeError("Fake9BGenerator responses exhausted")


def load_fixtures():
    units = json.loads(Path("evaluation/reconstructed_units_catalog.json").read_text(encoding="utf-8"))
    pages = [
        json.loads(x)
        for x in Path("data/processed/airef_2025_pages.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    doc_id = pages[1]["doc_id"]
    source_file = pages[1]["source_file"]
    published_at = pages[1]["published_at"]
    page_num = pages[1]["page_number"]
    page_text = pages[1]["text"]
    chunk = ChunkRecord(
        chunk_id="chunk_test_1",
        doc_id=doc_id,
        source_file=source_file,
        published_at=published_at,
        page_number=page_num,
        start_char=630,
        end_char=800,
        text=page_text[630:800],
        strategy="page-local-v1:max_chars=1000:overlap_chars=150",
    )
    hit = SearchHit(1, 0.95, chunk)
    return units, pages, [hit]


def make_valid_raw_response(model, content_obj, prompt_eval=150, eval_count=50):
    return {
        "model": model,
        "done": True,
        "done_reason": "stop",
        "message": {
            "role": "assistant",
            "content": json.dumps(content_obj),
        },
        "prompt_eval_count": prompt_eval,
        "eval_count": eval_count,
        "prompt_eval_duration": 40000000,
        "eval_duration": 120000000,
        "total_duration": 200000000,
        "load_duration": 15000000,
    }


class TestMixedVerifier(unittest.TestCase):
    """Exhaustive tests for Q3 verifier integration, role-pinned validation, and truthful identities."""

    def test_constructor_no_network_and_source_validation(self):
        """Constructor validates config without network and checks expected_digest format."""
        # Non-hex or invalid length digest raises ValueError
        with self.assertRaises(ValueError):
            Q3VerifierClient(expected_digest="invalid_short_digest")
        with self.assertRaises(ValueError):
            Q3VerifierClient(expected_digest=True)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            Q3VerifierClient(expected_digest="Z" * 64)

        # Constructor performs ZERO network calls
        class FailingTransport:
            def get_version(self): raise AssertionError("Network called!")
            def get_tags(self): raise AssertionError("Network called!")
            def show(self, m): raise AssertionError("Network called!")
            def get_ps(self): raise AssertionError("Network called!")
            def unload(self, m): raise AssertionError("Network called!")
            def chat(self, r): raise AssertionError("Network called!")

        client = Q3VerifierClient(transport=FailingTransport())
        self.assertEqual(client.model, VERIFIER_Q3_MODEL)
        self.assertEqual(client.expected_digest, VERIFIER_Q3_DIGEST)
        self.assertEqual(client.parameters, VERIFIER_Q3_PARAMETERS)

        # Verification policy is defensive copy
        pol = client.verification_policy
        pol["parameters"]["temperature"] = 999
        self.assertEqual(client.parameters["temperature"], 0)

    def test_metadata_preflight_and_drift_and_unsupported_quant(self):
        """Preflight validates version 0.40.11, Q3_K_M quantization, sampling; stops on mismatch."""
        # 1. Version mismatch
        t1 = FakeTransport(version="0.40.10")
        c1 = Q3VerifierClient(transport=t1)
        with self.assertRaisesRegex(RuntimeError, "version mismatch"):
            c1.preflight()

        # 2. Model tag missing
        class NoModelTransport(FakeTransport):
            def get_tags(self): return {"models": []}
        c2 = Q3VerifierClient(transport=NoModelTransport())
        with self.assertRaisesRegex(RuntimeError, "not found in Ollama"):
            c2.preflight()

        # 3. Model digest mismatch in tags
        t3 = FakeTransport(q3_digest="b" * 64)
        c3 = Q3VerifierClient(transport=t3)
        with self.assertRaisesRegex(RuntimeError, "digest mismatch"):
            c3.preflight()

        # 4. Quantization level mismatch in tags
        t4 = FakeTransport(quantization_level="Q8_0")
        c4 = Q3VerifierClient(transport=t4)
        with self.assertRaisesRegex(RuntimeError, "quantization mismatch"):
            c4.preflight()

        # 5. Quantization level mismatch in show
        t5 = FakeTransport(show_quant="Q4_K_M")
        c5 = Q3VerifierClient(transport=t5)
        with self.assertRaisesRegex(RuntimeError, "show quantization mismatch"):
            c5.preflight()

        # 6. Sampling mismatch against Arm B reference (unaligned parameter in show)
        t6 = FakeTransport(show_parameters="repeat_penalty 1.5\n")
        c6 = Q3VerifierClient(transport=t6)
        with self.assertRaisesRegex(RuntimeError, "Effective sampling mismatch"):
            c6.preflight()

        # 7. Null show parameters is valid
        t7 = FakeTransport(show_parameters=None)
        # When show_parameters is None, server defaults apply; Q3 parameters explicitly set presence_penalty=1.5 etc.
        # Resolving server_defaults + empty show_params + Q3 params matches Arm B reference
        c7 = Q3VerifierClient(transport=t7)
        res = c7.preflight()
        self.assertEqual(res["quantization_level"], "Q3_K_M")
        self.assertTrue(c7._preflight_done)

        # 8. Digest drift after chat
        t8 = FakeTransport(drift_after="c" * 64)
        c8 = Q3VerifierClient(transport=t8)
        c8._preflight_done = True
        with self.assertRaisesRegex(RuntimeError, "digest mismatch after"):
            c8.chat(
                [{"role": "user", "content": "test"}],
                {"type": "object"},
                stage="verification",
            )

    def test_q3_chat_contracts_and_rejections(self):
        """Strict validation of Q3 chat stage, budget, thinking, stats, and call recording."""
        transport = FakeTransport()
        client = Q3VerifierClient(transport=transport)
        client._preflight_done = True

        # Non-verification stage rejected
        with self.assertRaisesRegex(ValueError, "accepts only 'verification', 'reverification', or 'query_coverage'"):
            client.chat([{"role": "user", "content": "hi"}], {"type": "object"}, stage="selection")

        # Message bytes exceeding 12288 bytes rejected before contacting model
        large_msg = [{"role": "user", "content": "x" * 12300}]
        with self.assertRaisesRegex(ValueError, "exceeds 12288 bytes"):
            client.chat(large_msg, {"type": "object"}, stage="verification")
        self.assertEqual(len(transport.chat_calls), 0)

        # Unexpected thinking rejected
        resp_with_think = make_valid_raw_response(VERIFIER_Q3_MODEL, {"verdicts": []})
        resp_with_think["message"]["thinking"] = "I am reasoning about this claim..."
        t_think = FakeTransport(chat_response=resp_with_think)
        c_think = Q3VerifierClient(transport=t_think)
        c_think._preflight_done = True
        with self.assertRaisesRegex(ValueError, "unexpected thinking content"):
            c_think.chat([{"role": "user", "content": "test"}], {"type": "object"}, stage="verification")
        # Call record preserved even when validation fails
        self.assertEqual(len(c_think.calls), 1)
        self.assertTrue(c_think.calls[0]["attempted"])
        self.assertIn("unexpected thinking content", c_think.calls[0]["error"])
        self.assertIsNotNone(c_think.calls[0]["raw_response"])

        # Context budget exceeded (prompt_eval_count + 4096 + 256 > 8192)
        resp_overflow = make_valid_raw_response(VERIFIER_Q3_MODEL, {"verdicts": []}, prompt_eval=3841)
        t_overflow = FakeTransport(chat_response=resp_overflow)
        c_overflow = Q3VerifierClient(transport=t_overflow)
        c_overflow._preflight_done = True
        with self.assertRaisesRegex(ValueError, "Context budget exceeded"):
            c_overflow.chat([{"role": "user", "content": "test"}], {"type": "object"}, stage="verification")
        self.assertEqual(len(c_overflow.calls), 1)
        self.assertIn("Context budget exceeded", c_overflow.calls[0]["error"])

        # Tool calls rejected
        resp_tool = make_valid_raw_response(VERIFIER_Q3_MODEL, {"verdicts": []})
        resp_tool["message"]["tool_calls"] = [{"name": "search"}]
        t_tool = FakeTransport(chat_response=resp_tool)
        c_tool = Q3VerifierClient(transport=t_tool)
        c_tool._preflight_done = True
        with self.assertRaisesRegex(ValueError, "Unexpected tool_calls"):
            c_tool.chat([{"role": "user", "content": "test"}], {"type": "object"}, stage="verification")

        # Negative prompt_eval_count rejected
        resp_neg = make_valid_raw_response(VERIFIER_Q3_MODEL, {"verdicts": []}, prompt_eval=-5)
        t_neg = FakeTransport(chat_response=resp_neg)
        c_neg = Q3VerifierClient(transport=t_neg)
        c_neg._preflight_done = True
        with self.assertRaisesRegex(ValueError, "invalid prompt_eval_count"):
            c_neg.chat([{"role": "user", "content": "test"}], {"type": "object"}, stage="verification")

        # Transport error preserved with attempted=True and wall time
        t_err = FakeTransport(chat_error=RuntimeError("Connection dropped"))
        c_err = Q3VerifierClient(transport=t_err)
        c_err._preflight_done = True
        with self.assertRaisesRegex(RuntimeError, "Connection dropped"):
            c_err.chat([{"role": "user", "content": "test"}], {"type": "object"}, stage="verification")
        self.assertEqual(len(c_err.calls), 1)
        self.assertTrue(c_err.calls[0]["attempted"])
        self.assertIn("Connection dropped", c_err.calls[0]["error"])
        self.assertGreaterEqual(c_err.calls[0]["wall_time_seconds"], 0.0)

    def test_lifecycle_errors_are_retained_and_propagated(self):
        for operation in ("get_ps", "unload"):
            transport = FakeTransport(resident_models=[DEFAULT_GENERATION_MODEL])
            client = Q3VerifierClient(transport=transport)
            with patch.object(transport, operation, side_effect=RuntimeError("offline")):
                with self.assertRaises(RuntimeError):
                    client.release_loaded(DEFAULT_GENERATION_MODEL)
            self.assertIn("offline", client.lifecycle_records[-1]["error"])
            self.assertEqual(transport.chat_calls, [])

    def test_release_loaded_memory_management(self):
        """release_loaded accepts ONLY the two exact model names and unloads only if resident."""
        t = FakeTransport(resident_models=[DEFAULT_GENERATION_MODEL])
        client = Q3VerifierClient(transport=t)

        # Invalid model name rejected
        for bad in ["qwen3:14b", "all-minilm", "qwen3.5:9b-q8_0", ""]:
            with self.subTest(bad=bad):
                with self.assertRaisesRegex(ValueError, "accepts ONLY"):
                    client.release_loaded(bad)

        # Release resident 9B model
        self.assertTrue(client.release_loaded(DEFAULT_GENERATION_MODEL))
        self.assertIn(DEFAULT_GENERATION_MODEL, t.unloaded_models)
        self.assertEqual(len(client.lifecycle_records), 1)
        self.assertTrue(client.lifecycle_records[0]["was_resident"])
        self.assertTrue(client.lifecycle_records[0]["unloaded"])
        self.assertEqual(len(client.calls), 0)  # Never counted as inference

        # Release non-resident Q3 model
        self.assertFalse(client.release_loaded(VERIFIER_Q3_MODEL))
        self.assertEqual(len(client.lifecycle_records), 2)
        self.assertFalse(client.lifecycle_records[1]["was_resident"])
        self.assertFalse(client.lifecycle_records[1]["unloaded"])
        self.assertEqual(len(client.calls), 0)

    def test_transport_raw_error_body_is_preserved(self):
        error = OllamaResponseError("bad JSON", raw_text="invalid{", status_code=200, is_truncated=True)
        client = Q3VerifierClient(transport=FakeTransport(chat_error=error))
        with self.assertRaises(OllamaResponseError):
            client.chat([{"role": "user", "content": "fixture"}], {"type": "object"}, stage="verification")
        self.assertEqual(client.calls[-1]["raw_response"], {"raw_text": "invalid{", "status_code": 200, "is_truncated": True})
        self.assertTrue(client.calls[-1]["attempted"])

    def test_legacy_default_unchanged(self):
        """Legacy default is preserved when verification_policy is None."""
        units, pages, hits = load_fixtures()
        # Create a valid legacy 9B result
        sel_resp = make_valid_raw_response(DEFAULT_GENERATION_MODEL, {
            "status": "selected",
            "ids": ["P1"],
            "reason": "",
        })
        gen_resp = make_valid_raw_response(DEFAULT_GENERATION_MODEL, {
            "status": "answered",
            "claims": [{"claim_id": "C1", "text": "Claim text.", "evidence_ids": ["E1"]}],
            "reason": "",
        })
        ver_resp = make_valid_raw_response(DEFAULT_GENERATION_MODEL, {
            "verdicts": [{
                "claim_id": "C1",
                "verdict": "supported",
                "reason": "OK",
                "evidence_verdicts": [{"evidence_id": "E1", "relation": "supports"}],
            }]
        })

        legacy_gen = Fake9BGenerator([sel_resp, gen_resp, ver_resp])
        # Use ReconstructedGrounder directly to produce legacy result
        from experiments.preserve_evidence import PreservedGrounder
        grounder = PreservedGrounder(units, pages, generator=legacy_gen)
        legacy_res = grounder.answer("Pregunta de prueba", hits, expected_digest=TEST_9B_DIGEST)

        # Validate with verification_policy=None
        validated = validate_grounded_result(
            legacy_res,
            grounder.expanded_hits,
            expected_digest=TEST_9B_DIGEST,
            expected_model=DEFAULT_GENERATION_MODEL,
            verification_policy=None,
        )
        self.assertEqual(validated["status"], "answered")

        # generate_arm with verification_policy=None produces exact legacy output structure
        legacy_gen2 = Fake9BGenerator([sel_resp, gen_resp, ver_resp])
        grounder2 = PreservedGrounder(units, pages, generator=legacy_gen2)
        arm_out = generate_arm(grounder2, "Pregunta de prueba", hits, TEST_9B_DIGEST, preserve=True, verification_policy=None)
        self.assertEqual(arm_out["status"], "answered")
        self.assertNotIn("verifier_calls", arm_out)
        self.assertNotIn("lifecycle_records", arm_out)
        self.assertNotIn("verification_policy", arm_out)
        self.assertIn("calls", arm_out)
        self.assertEqual(len(arm_out["calls"]), 3)

    def test_routing_initial_and_repair_verification_only(self):
        """Routing routes initial verification and reverification to Q3VerifierClient; selection/generation stay 9B."""
        units, pages, hits = load_fixtures()
        quote = hits[0].chunk.text[:50]

        # 1. Successful run without repair
        sel_resp = make_valid_raw_response(DEFAULT_GENERATION_MODEL, {
            "status": "selected",
            "ids": ["P1"],
            "reason": "",
        })
        gen_resp = make_valid_raw_response(DEFAULT_GENERATION_MODEL, {
            "status": "answered",
            "claims": [{"claim_id": "C1", "text": "Afirmación 1.", "evidence_ids": ["E1"]}],
            "reason": "",
        })
        q3_ver_resp = make_valid_raw_response(VERIFIER_Q3_MODEL, {
            "verdicts": [{
                "claim_id": "C1",
                "verdict": "supported",
                "reason": "Soporte verificado",
                "evidence_verdicts": [{"evidence_id": "E1", "relation": "supports"}],
            }]
        })

        t_q3 = FakeTransport(chat_response=q3_ver_resp)
        verifier = Q3VerifierClient(transport=t_q3)
        verifier._preflight_done = True
        gen9b = Fake9BGenerator([sel_resp, gen_resp])

        q3_grounder = Q3PreservedGrounder(units, pages, verifier=verifier, generator=gen9b)
        result = q3_grounder.answer("Pregunta", hits, expected_digest=TEST_9B_DIGEST)

        # Verify trace models and parameters
        trace = result["grounding"]["trace"]
        self.assertEqual(len(trace), 3)
        self.assertEqual(trace[0]["stage"], "selection")
        self.assertEqual(trace[0]["model"], DEFAULT_GENERATION_MODEL)
        self.assertEqual(trace[0]["parameters"], DEFAULT_PARAMETERS)
        self.assertEqual(trace[0]["model_digest"], TEST_9B_DIGEST)

        self.assertEqual(trace[1]["stage"], "generation")
        self.assertEqual(trace[1]["model"], DEFAULT_GENERATION_MODEL)
        self.assertEqual(trace[1]["parameters"], DEFAULT_PARAMETERS)
        self.assertEqual(trace[1]["model_digest"], TEST_9B_DIGEST)

        self.assertEqual(trace[2]["stage"], "verification")
        self.assertEqual(trace[2]["model"], VERIFIER_Q3_MODEL)
        self.assertEqual(trace[2]["parameters"], VERIFIER_Q3_PARAMETERS)
        self.assertEqual(trace[2]["model_digest"], VERIFIER_Q3_DIGEST)

        self.assertEqual(result["digest"], TEST_9B_DIGEST)
        self.assertEqual(len(gen9b.calls), 2)
        self.assertEqual(len(verifier.calls), 1)

        # Validate with explicit verification_policy
        val = validate_grounded_result(
            result,
            q3_grounder.expanded_hits,
            expected_digest=TEST_9B_DIGEST,
            expected_model=DEFAULT_GENERATION_MODEL,
            verification_policy=q3_grounder.verification_policy,
        )
        self.assertEqual(val["status"], "answered")

        # 2. Run with repair: initial verification fails, repair_selection + repair_generation on 9B, reverification on Q3
        sel_resp2 = make_valid_raw_response(DEFAULT_GENERATION_MODEL, {
            "status": "selected",
            "ids": ["P1"],
            "reason": "",
        })
        gen_resp2 = make_valid_raw_response(DEFAULT_GENERATION_MODEL, {
            "status": "answered",
            "claims": [{"claim_id": "C1", "text": "Afirmación 1.", "evidence_ids": ["E1"]}],
            "reason": "",
        })
        q3_ver_fail = make_valid_raw_response(VERIFIER_Q3_MODEL, {
            "verdicts": [{
                "claim_id": "C1",
                "verdict": "contradicted",
                "reason": "Contradicción detectada",
                "evidence_verdicts": [{"evidence_id": "E1", "relation": "contradicts"}],
            }]
        })
        rep_sel_resp = make_valid_raw_response(DEFAULT_GENERATION_MODEL, {
            "status": "selected",
            "ids": ["P1"],
            "reason": "",
        })
        rep_gen_resp = make_valid_raw_response(DEFAULT_GENERATION_MODEL, {
            "repairs": [{
                "claim_id": "C1",
                "status": "answered",
                "text": "Afirmación reparada.",
                "evidence_ids": ["E2"],
                "reason": "",
            }]
        })
        q3_rever_ok = make_valid_raw_response(VERIFIER_Q3_MODEL, {
            "verdicts": [{
                "claim_id": "C1",
                "verdict": "supported",
                "reason": "Reparación respaldada",
                "evidence_verdicts": [{"evidence_id": "E2", "relation": "supports"}],
            }]
        })

        t_repair = FakeTransport()
        q3_calls = [q3_ver_fail, q3_rever_ok]
        t_repair.chat = lambda req: copy.deepcopy(q3_calls.pop(0))
        verifier_rep = Q3VerifierClient(transport=t_repair)
        verifier_rep._preflight_done = True

        gen9b_rep = Fake9BGenerator([sel_resp2, gen_resp2, rep_sel_resp, rep_gen_resp])
        grounder_rep = Q3PreservedGrounder(units, pages, verifier=verifier_rep, generator=gen9b_rep)
        rep_res = grounder_rep.answer("Pregunta 2", hits, expected_digest=TEST_9B_DIGEST)

        trace_rep = rep_res["grounding"]["trace"]
        self.assertEqual(len(trace_rep), 6)
        expected_models = [
            DEFAULT_GENERATION_MODEL,  # selection
            DEFAULT_GENERATION_MODEL,  # generation
            VERIFIER_Q3_MODEL,         # verification
            DEFAULT_GENERATION_MODEL,  # repair_selection
            DEFAULT_GENERATION_MODEL,  # repair_generation
            VERIFIER_Q3_MODEL,         # reverification
        ]
        self.assertEqual([t["model"] for t in trace_rep], expected_models)

        val_rep = validate_grounded_result(
            rep_res,
            grounder_rep.expanded_hits,
            expected_digest=TEST_9B_DIGEST,
            expected_model=DEFAULT_GENERATION_MODEL,
            verification_policy=grounder_rep.verification_policy,
        )
        self.assertEqual(val_rep["status"], "answered")

    def test_validation_rejects_spoofed_fields_and_tampering(self):
        """Strict boundary rejection for spoofed models, digests, parameters, booleans, and tampering."""
        units, pages, hits = load_fixtures()
        quote = hits[0].chunk.text[:50]

        sel_resp = make_valid_raw_response(DEFAULT_GENERATION_MODEL, {
            "status": "selected",
            "ids": ["P1"],
            "reason": "",
        })
        gen_resp = make_valid_raw_response(DEFAULT_GENERATION_MODEL, {
            "status": "answered",
            "claims": [{"claim_id": "C1", "text": "Claim.", "evidence_ids": ["E1"]}],
            "reason": "",
        })
        q3_ver = make_valid_raw_response(VERIFIER_Q3_MODEL, {
            "verdicts": [{
                "claim_id": "C1",
                "verdict": "supported",
                "reason": "OK",
                "evidence_verdicts": [{"evidence_id": "E1", "relation": "supports"}],
            }]
        })
        t = FakeTransport(chat_response=q3_ver)
        v = Q3VerifierClient(transport=t)
        v._preflight_done = True
        g = Q3PreservedGrounder(units, pages, verifier=v, generator=Fake9BGenerator([sel_resp, gen_resp]))
        valid_res = g.answer("Q", hits, expected_digest=TEST_9B_DIGEST)
        policy = g.verification_policy

        # 1. verification_policy with unknown extra keys rejected
        bad_policy = copy.deepcopy(policy)
        bad_policy["extra_key"] = "forbidden"
        with self.assertRaisesRegex(ValueError, "verification_policy keys mismatch"):
            validate_grounded_result(valid_res, g.expanded_hits, TEST_9B_DIGEST, verification_policy=bad_policy)

        # 2. verification_policy with boolean or NaN parameter rejected
        bad_pol_bool = copy.deepcopy(policy)
        bad_pol_bool["parameters"]["seed"] = True
        with self.assertRaisesRegex(ValueError, "cannot be boolean"):
            validate_grounded_result(valid_res, g.expanded_hits, TEST_9B_DIGEST, verification_policy=bad_pol_bool)

        bad_pol_nan = copy.deepcopy(policy)
        bad_pol_nan["parameters"]["temperature"] = float("nan")
        with self.assertRaisesRegex(ValueError, "non-compliant JSON values"):
            validate_grounded_result(valid_res, g.expanded_hits, TEST_9B_DIGEST, verification_policy=bad_pol_nan)

        # 3. Spoofed model in verification trace (9B instead of Q3) rejected
        spoofed_model = copy.deepcopy(valid_res)
        spoofed_model["grounding"]["trace"][2]["model"] = DEFAULT_GENERATION_MODEL
        with self.assertRaisesRegex(ValueError, "model mismatch for verification"):
            validate_grounded_result(spoofed_model, g.expanded_hits, TEST_9B_DIGEST, verification_policy=policy)

        # 4. Spoofed digest in verification trace rejected
        spoofed_dig = copy.deepcopy(valid_res)
        spoofed_dig["grounding"]["trace"][2]["model_digest"] = "e" * 64
        with self.assertRaisesRegex(ValueError, "model_digest mismatch for verification"):
            validate_grounded_result(spoofed_dig, g.expanded_hits, TEST_9B_DIGEST, verification_policy=policy)

        # 5. Spoofed parameters in verification trace rejected
        spoofed_param = copy.deepcopy(valid_res)
        spoofed_param["grounding"]["trace"][2]["parameters"]["num_predict"] = 1024
        with self.assertRaisesRegex(ValueError, "parameter 'num_predict' mismatch"):
            validate_grounded_result(spoofed_param, g.expanded_hits, TEST_9B_DIGEST, verification_policy=policy)

        # 6. Spoofed model in generation trace (Q3 instead of 9B) rejected
        spoofed_gen = copy.deepcopy(valid_res)
        spoofed_gen["grounding"]["trace"][1]["model"] = VERIFIER_Q3_MODEL
        with self.assertRaisesRegex(ValueError, "model mismatch"):
            validate_grounded_result(spoofed_gen, g.expanded_hits, TEST_9B_DIGEST, verification_policy=policy)

        # 6b. Unknown model not in ALLOWED_MODELS in selection trace rejected
        spoofed_sel = copy.deepcopy(valid_res)
        spoofed_sel["grounding"]["trace"][0]["model"] = "unknown_model:7b"
        with self.assertRaisesRegex(ValueError, "not in ALLOWED_MODELS"):
            validate_grounded_result(spoofed_sel, g.expanded_hits, TEST_9B_DIGEST, verification_policy=policy)

        # 7. Aggregate prompt_sha256 tampering rejected
        tampered_agg = copy.deepcopy(valid_res)
        tampered_agg["prompt_sha256"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "prompt_sha256 aggregate mismatch"):
            validate_grounded_result(tampered_agg, g.expanded_hits, TEST_9B_DIGEST, verification_policy=policy)

        # 8. Aggregate stats sum tampering rejected
        tampered_stats = copy.deepcopy(valid_res)
        tampered_stats["stats"]["eval_count"] += 1
        with self.assertRaisesRegex(ValueError, "Stats eval_count sum mismatch"):
            validate_grounded_result(tampered_stats, g.expanded_hits, TEST_9B_DIGEST, verification_policy=policy)

    def test_pre_embedding_lifecycle_survives_generate_arm(self):
        verifier = SimpleNamespace(calls=[], lifecycle_records=[{"action": "unload", "model": VERIFIER_Q3_MODEL}])
        grounder = SimpleNamespace(generator=SimpleNamespace(calls=[]), verifier=verifier, answer=Mock(side_effect=RuntimeError("fixture")))
        policy = {"model": VERIFIER_Q3_MODEL, "digest": VERIFIER_Q3_DIGEST, "parameters": VERIFIER_Q3_PARAMETERS}
        out = generate_arm(grounder, "fixture", [], TEST_9B_DIGEST, preserve=True, verification_policy=policy)
        self.assertEqual(out["lifecycle_records"], [{"action": "unload", "model": VERIFIER_Q3_MODEL}])

    def test_errors_retain_partial_raw_calls_no_fallback(self):
        """Errors retain attempted calls in verifier.calls; generate_arm preserves distinct call logs."""
        units, pages, hits = load_fixtures()
        quote = hits[0].chunk.text[:50]

        sel_resp = make_valid_raw_response(DEFAULT_GENERATION_MODEL, {
            "status": "selected",
            "ids": ["P1"],
            "reason": "",
        })
        gen_resp = make_valid_raw_response(DEFAULT_GENERATION_MODEL, {
            "status": "answered",
            "claims": [{"claim_id": "C1", "text": "Claim.", "evidence_ids": ["E1"]}],
            "reason": "",
        })

        t_fail = FakeTransport(chat_error=RuntimeError("GPU OOM in Q3 verifier"))
        v_fail = Q3VerifierClient(transport=t_fail)
        v_fail._preflight_done = True
        gen9b = Fake9BGenerator([sel_resp, gen_resp])
        grounder = Q3PreservedGrounder(units, pages, verifier=v_fail, generator=gen9b)

        # Execute generate_arm with verification_policy
        arm_out = generate_arm(
            grounder,
            "Pregunta",
            hits,
            TEST_9B_DIGEST,
            preserve=True,
            verification_policy=grounder.verification_policy,
        )

        self.assertEqual(arm_out["status"], "error")
        self.assertIn("GPU OOM", arm_out["error"])
        # Original generator calls remain genuine 9B
        self.assertEqual(len(arm_out["calls"]), 2)
        # Distinct verifier calls retained with raw error body
        self.assertEqual(len(arm_out["verifier_calls"]), 1)
        self.assertTrue(arm_out["verifier_calls"][0]["attempted"])
        self.assertIn("GPU OOM", arm_out["verifier_calls"][0]["error"])
        # Lifecycle records retained
        self.assertIn("lifecycle_records", arm_out)
        self.assertIn("verification_policy", arm_out)


if __name__ == "__main__":
    unittest.main()
