"""Offline unit tests for the covered RAG pipeline evaluation runner (Phase 2)."""

import copy
import hashlib
import json
from pathlib import Path
import re
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from experiments.evaluate_covered_pipeline import (
    EXPECTED_CASE_COUNT,
    EXPECTED_CASE_IDS,
    PROTOCOL_SCHEMA_VERSION,
    REQUIRED_FROZEN_MODULES,
    REQUIRED_SERVER_VERSION,
    SCHEMA_VERSION,
    _extract_role_stats,
    main as runner_main,
    run,
    validate_pre_inference,
    validate_protocol,
    validate_queries_roster,
)
from experiments.preserved_q3_verifier import (
    VERIFIER_Q3_DIGEST,
    VERIFIER_Q3_MODEL,
    VERIFIER_Q3_PARAMETERS,
)
from rag_pensiones.answer import DEFAULT_GENERATION_MODEL
from rag_pensiones.grounding import DEFAULT_PARAMETERS
from rag_pensiones.ingest import PROJECT_ROOT, validate_project_path
from rag_pensiones.qwen_embeddings import MODEL
from rag_pensiones.search import DEFAULT_INDEX_PATH, load_index


def _sample_inputs():
    index = load_index(DEFAULT_INDEX_PATH)
    units = json.loads(Path("evaluation/reconstructed_units_catalog.json").read_text(encoding="utf-8"))
    pages = [json.loads(x) for x in Path("data/processed/airef_2025_pages.jsonl").read_text(encoding="utf-8").splitlines()]
    queries = json.loads(Path("evaluation/scope_coverage_2026-10-26/full_inputs.json").read_text(encoding="utf-8"))
    return index, units, pages, queries


def _sample_protocol() -> dict:
    frozen_files = {}
    for mod in REQUIRED_FROZEN_MODULES:
        p = validate_project_path(PROJECT_ROOT / mod)
        frozen_files[mod] = hashlib.sha256(p.read_bytes()).hexdigest()

    inputs = {
        "index_path": "data/processed/airef_2025_qwen3_4b_embeddings.json",
        "queries_path": "evaluation/scope_coverage_2026-10-26/full_inputs.json",
        "units_path": "evaluation/reconstructed_units_catalog.json",
        "pages_path": "data/processed/airef_2025_pages.jsonl",
    }
    for path_val in inputs.values():
        p = validate_project_path(PROJECT_ROOT / path_val)
        frozen_files[path_val] = hashlib.sha256(p.read_bytes()).hexdigest()

    return {
        "schema_version": PROTOCOL_SCHEMA_VERSION,
        "server_version": REQUIRED_SERVER_VERSION,
        "generator_digest": "a" * 64,
        "verifier_digest": VERIFIER_Q3_DIGEST,
        "embedding_digest": "b" * 64,
        "generator_parameters": copy.deepcopy(DEFAULT_PARAMETERS),
        "verifier_parameters": copy.deepcopy(VERIFIER_Q3_PARAMETERS),
        "index_path": inputs["index_path"],
        "queries_path": inputs["queries_path"],
        "units_path": inputs["units_path"],
        "pages_path": inputs["pages_path"],
        "frozen_files": frozen_files,
    }


class FakeRunnerTransport:
    """Mock transport for Ollama preflight verification."""

    def __init__(
        self,
        *,
        version: str = REQUIRED_SERVER_VERSION,
        gen_digest: str = "a" * 64,
        ver_digest: str = VERIFIER_Q3_DIGEST,
        emb_digest: str = "b" * 64,
        quant_level: str = "Q3_K_M",
    ) -> None:
        self.version = version
        self.gen_digest = gen_digest
        self.ver_digest = ver_digest
        self.emb_digest = emb_digest
        self.quant_level = quant_level

    def get_version(self) -> dict:
        return {"version": self.version}

    def get_tags(self) -> dict:
        return {
            "models": [
                {
                    "name": DEFAULT_GENERATION_MODEL,
                    "model": DEFAULT_GENERATION_MODEL,
                    "digest": self.gen_digest,
                    "details": {"quantization_level": "Q4_K_M"},
                },
                {
                    "name": VERIFIER_Q3_MODEL,
                    "model": VERIFIER_Q3_MODEL,
                    "digest": self.ver_digest,
                    "details": {"quantization_level": self.quant_level},
                },
                {
                    "name": MODEL,
                    "model": MODEL,
                    "digest": self.emb_digest,
                    "details": {"quantization_level": "F16"},
                },
            ]
        }

    def show(self, model: str) -> dict:
        return {
            "details": {"quantization_level": self.quant_level},
            "parameters": "presence_penalty 1.5\ntop_k 20\ntop_p 0.95\nmin_p 0.0\n",
        }

    def get_ps(self) -> dict:
        return {"models": []}

    def unload(self, model: str) -> dict:
        return {"status": "success"}


class TestMandatoryProtocolAndOrdering(unittest.TestCase):
    """Verifies Fix 1 (mandatory valid protocol before clients) and Fix 2 (ordering/local validation)."""

    def test_direct_run_rejects_missing_protocol_before_clients(self):
        index, units, pages, queries = _sample_inputs()
        with tempfile.TemporaryDirectory(dir="tmp") as tmp_dir:
            out_file = Path(tmp_dir) / "out.json"
            with patch("experiments.evaluate_covered_pipeline.RecordedGenerator") as mock_gen, \
                 patch("experiments.evaluate_covered_pipeline.Q3VerifierClient") as mock_ver, \
                 patch("experiments.evaluate_covered_pipeline.QwenEmbedder") as mock_emb:
                with self.assertRaises(ValueError) as ctx:
                    run(
                        index,
                        queries,
                        units,
                        pages,
                        out_file,
                        generator_digest="a" * 64,
                        verifier_digest=VERIFIER_Q3_DIGEST,
                        embedding_digest=index["model_digest"],
                        protocol=None,
                    )
                self.assertIn("protocol is mandatory", str(ctx.exception))
                mock_gen.assert_not_called()
                mock_ver.assert_not_called()
                mock_emb.assert_not_called()

    def test_in_memory_data_mismatch_vs_pinned_files_fails_before_clients(self):
        index, units, pages, queries = _sample_inputs()
        proto = _sample_protocol()
        proto["embedding_digest"] = index["model_digest"]

        # 1. Modifying queries
        bad_queries = copy.deepcopy(queries)
        bad_queries["cases"][0]["query"] = "Tampered query text"
        with tempfile.TemporaryDirectory(dir="tmp") as tmp_dir:
            out_file = Path(tmp_dir) / "out.json"
            with patch("experiments.evaluate_covered_pipeline.RecordedGenerator") as mock_gen, \
                 patch("experiments.evaluate_covered_pipeline.Q3VerifierClient") as mock_ver:
                with self.assertRaises(ValueError) as ctx:
                    run(
                        index,
                        bad_queries,
                        units,
                        pages,
                        out_file,
                        generator_digest="a" * 64,
                        verifier_digest=VERIFIER_Q3_DIGEST,
                        embedding_digest=index["model_digest"],
                        protocol=proto,
                    )
                self.assertIn("In-memory queries data does not match decoded pinned queries file", str(ctx.exception))
                mock_gen.assert_not_called()
                mock_ver.assert_not_called()

        # 2. Modifying index
        bad_index = copy.deepcopy(index)
        bad_index["model_digest"] = "different_digest" * 2
        with tempfile.TemporaryDirectory(dir="tmp") as tmp_dir:
            out_file = Path(tmp_dir) / "out.json"
            with patch("experiments.evaluate_covered_pipeline.RecordedGenerator") as mock_gen:
                with self.assertRaises(ValueError) as ctx:
                    run(
                        bad_index,
                        queries,
                        units,
                        pages,
                        out_file,
                        generator_digest="a" * 64,
                        verifier_digest=VERIFIER_Q3_DIGEST,
                        embedding_digest=index["model_digest"],
                        protocol=proto,
                    )
                self.assertIn("In-memory index data does not match", str(ctx.exception))
                mock_gen.assert_not_called()

    def test_missing_pin_or_altered_source_fails_before_network(self):
        index, units, pages, queries = _sample_inputs()
        proto = _sample_protocol()
        proto["embedding_digest"] = index["model_digest"]

        # Missing pin
        bad_proto = copy.deepcopy(proto)
        del bad_proto["frozen_files"]["experiments/query_coverage.py"]
        with tempfile.TemporaryDirectory(dir="tmp") as tmp_dir:
            out_file = Path(tmp_dir) / "out.json"
            with patch("experiments.evaluate_covered_pipeline.QuantizationOllamaTransport") as mock_transport_cls:
                with self.assertRaises(ValueError) as ctx:
                    run(
                        index,
                        queries,
                        units,
                        pages,
                        out_file,
                        generator_digest="a" * 64,
                        verifier_digest=VERIFIER_Q3_DIGEST,
                        embedding_digest=index["model_digest"],
                        protocol=bad_proto,
                    )
                self.assertIn("Required dependency code pins missing", str(ctx.exception))
                mock_transport_cls.assert_not_called()

        # Altered pin
        bad_proto2 = copy.deepcopy(proto)
        bad_proto2["frozen_files"]["experiments/query_coverage.py"] = "f" * 64
        with tempfile.TemporaryDirectory(dir="tmp") as tmp_dir:
            out_file = Path(tmp_dir) / "out.json"
            with patch("experiments.evaluate_covered_pipeline.QuantizationOllamaTransport") as mock_transport_cls:
                with self.assertRaises(ValueError) as ctx:
                    run(
                        index,
                        queries,
                        units,
                        pages,
                        out_file,
                        generator_digest="a" * 64,
                        verifier_digest=VERIFIER_Q3_DIGEST,
                        embedding_digest=index["model_digest"],
                        protocol=bad_proto2,
                    )
                self.assertIn("Frozen file integrity mismatch", str(ctx.exception))
                mock_transport_cls.assert_not_called()


class TestNoOverwriteGuard(unittest.TestCase):
    """Verifies Fix 3: output_path consistently raises ValueError and prevents overwrite."""

    def test_refuses_to_overwrite_existing_file_without_clients(self):
        index, units, pages, queries = _sample_inputs()
        proto = _sample_protocol()
        proto["embedding_digest"] = index["model_digest"]

        with tempfile.TemporaryDirectory(dir="tmp") as tmp_dir:
            out_file = Path(tmp_dir) / "covered_results.json"
            out_file.write_text("original content", encoding="utf-8")

            with patch("experiments.evaluate_covered_pipeline.RecordedGenerator") as mock_gen, \
                 patch("experiments.evaluate_covered_pipeline.Q3VerifierClient") as mock_ver, \
                 patch("experiments.evaluate_covered_pipeline.QuantizationOllamaTransport") as mock_transport_cls:
                with self.assertRaises(ValueError) as ctx:
                    run(
                        index,
                        queries,
                        units,
                        pages,
                        out_file,
                        generator_digest="a" * 64,
                        verifier_digest=VERIFIER_Q3_DIGEST,
                        embedding_digest=index["model_digest"],
                        protocol=proto,
                    )
                self.assertRegex(str(ctx.exception), re.compile(r"Output already exists; choose a fresh JSON path"))
                self.assertEqual(out_file.read_text(encoding="utf-8"), "original content")
                mock_gen.assert_not_called()
                mock_ver.assert_not_called()
                mock_transport_cls.assert_not_called()


class TestRealErrorShapesPreserved(unittest.TestCase):
    """Verifies Fix 4: real error report shapes from answer_covered are preserved without relabeling."""

    def test_factual_error_shape_preserves_primary_error_and_continues(self):
        index, units, pages, queries = _sample_inputs()
        proto = _sample_protocol()
        proto["embedding_digest"] = index["model_digest"]
        transport = FakeRunnerTransport(gen_digest="a" * 64, emb_digest=index["model_digest"])

        call_idx = 0

        def answer_covered_with_real_error_shape(*args, **kwargs):
            nonlocal call_idx
            call_idx += 1
            q_text = args[1]
            if call_idx == 1:
                # Actual error shape produced by query_coverage.answer_covered for factual error
                return {
                    "schema_version": "preserved-query-coverage-v1",
                    "status": "error",
                    "query": q_text,
                    "error": "Factual grounding error: CUDA out of memory",
                    "factual_report": {
                        "outcome": {"status": "error", "error": "CUDA out of memory"}
                    },
                    "coverage_audit": None,
                    "publication": {
                        "schema_version": "coverage-publication-v1",
                        "status": "error",
                        "claims": [],
                        "missing_parts": [],
                        "reason": "Factual grounding error: CUDA out of memory",
                        "parts": [],
                    },
                    "metadata": {
                        "profile": "preserved_query_coverage_v1",
                        "factual_status": "error",
                        "coverage_status": "error",
                    },
                }
            elif call_idx == 2:
                # Actual error shape produced by query_coverage.answer_covered for coverage error
                return {
                    "schema_version": "preserved-query-coverage-v1",
                    "status": "error",
                    "query": q_text,
                    "error": "Coverage audit error: JSON parsing truncated",
                    "factual_report": {
                        "metadata": {
                            "planner_used": False,
                            "closed_note_tails": False,
                            "max_packages": 1,
                            "digest": "a" * 64,
                            "verifier_model": VERIFIER_Q3_MODEL,
                            "verifier_digest": VERIFIER_Q3_DIGEST,
                        },
                        "outcome": {"status": "answered", "full_result": {"grounding": {"trace": []}}},
                    },
                    "coverage_audit": {"status": "error", "error": "JSON parsing truncated"},
                    "publication": {
                        "schema_version": "coverage-publication-v1",
                        "status": "error",
                        "claims": [],
                        "missing_parts": [],
                        "reason": "Coverage audit error: JSON parsing truncated",
                        "parts": [],
                    },
                    "metadata": {
                        "profile": "preserved_query_coverage_v1",
                        "factual_status": "answered",
                        "coverage_status": "error",
                    },
                }
            # Successful case
            return {
                "schema_version": "preserved-query-coverage-v1",
                "status": "answered",
                "query": q_text,
                "factual_report": {
                    "metadata": {
                        "planner_used": False,
                        "closed_note_tails": False,
                        "max_packages": 1,
                        "digest": "a" * 64,
                        "verifier_model": VERIFIER_Q3_MODEL,
                        "verifier_digest": VERIFIER_Q3_DIGEST,
                    },
                    "outcome": {"status": "answered", "full_result": {"grounding": {"trace": []}}},
                },
                "coverage_audit": {"status": "completed", "skipped": True, "calls": []},
                "publication": {
                    "schema_version": "coverage-publication-v1",
                    "status": "answered",
                    "claims": [],
                    "missing_parts": [],
                    "reason": "",
                },
                "metadata": {
                    "profile": "preserved_query_coverage_v1",
                    "verifier_model": VERIFIER_Q3_MODEL,
                    "verifier_digest": VERIFIER_Q3_DIGEST,
                },
            }

        with tempfile.TemporaryDirectory(dir="tmp") as tmp_dir:
            out_file = Path(tmp_dir) / "real_errors.json"
            with patch("experiments.evaluate_covered_pipeline.RecordedGenerator") as mock_gen_cls, \
                 patch("experiments.evaluate_covered_pipeline.QwenEmbedder") as mock_emb_cls, \
                 patch("experiments.evaluate_covered_pipeline.answer_covered", side_effect=answer_covered_with_real_error_shape):

                mock_gen_cls.side_effect = lambda: SimpleNamespace(calls=[], get_digest=lambda: "a" * 64)
                mock_emb_cls.side_effect = lambda **k: SimpleNamespace(calls=[])

                report = run(
                    index,
                    queries,
                    units,
                    pages,
                    out_file,
                    generator_digest="a" * 64,
                    verifier_digest=VERIFIER_Q3_DIGEST,
                    embedding_digest=index["model_digest"],
                    transport=transport,
                    protocol=proto,
                )

        self.assertEqual(report["run_status"], "error")
        self.assertEqual(len(report["cases"]), 11)

        # First case preserves primary factual error
        c0 = report["cases"][0]
        self.assertEqual(c0["status"], "error")
        self.assertEqual(c0["error"], "Factual grounding error: CUDA out of memory")
        self.assertIsNone(c0["formatted_answer"])
        self.assertIn("factual_report", c0["answer"])

        # Second case preserves primary coverage error
        c1 = report["cases"][1]
        self.assertEqual(c1["status"], "error")
        self.assertEqual(c1["error"], "Coverage audit error: JSON parsing truncated")
        self.assertIsNone(c1["formatted_answer"])

        # Remaining cases executed normally
        c2 = report["cases"][2]
        self.assertEqual(c2["status"], "answered")

    def test_identity_guard_rejects_explicit_wrong_identity_on_success(self):
        index, units, pages, queries = _sample_inputs()
        proto = _sample_protocol()
        proto["embedding_digest"] = index["model_digest"]
        transport = FakeRunnerTransport(gen_digest="a" * 64, emb_digest=index["model_digest"])

        def wrong_identity_answer(*args, **kwargs):
            return {
                "schema_version": "preserved-query-coverage-v1",
                "status": "answered",
                "query": args[1],
                "factual_report": {
                    "metadata": {
                        "planner_used": False,
                        "closed_note_tails": False,
                        "max_packages": 1,
                        "digest": "a" * 64,
                        "verifier_model": "wrong_verifier_model",
                        "verifier_digest": VERIFIER_Q3_DIGEST,
                    },
                    "outcome": {"status": "answered", "full_result": {"grounding": {"trace": []}}},
                },
                "coverage_audit": {"status": "completed", "skipped": True, "calls": []},
                "publication": {"schema_version": "coverage-publication-v1", "status": "answered", "claims": [], "missing_parts": []},
                "metadata": {
                    "profile": "preserved_query_coverage_v1",
                    "verifier_model": "wrong_verifier_model",
                    "verifier_digest": VERIFIER_Q3_DIGEST,
                },
            }

        with tempfile.TemporaryDirectory(dir="tmp") as tmp_dir:
            out_file = Path(tmp_dir) / "wrong_id.json"
            with patch("experiments.evaluate_covered_pipeline.RecordedGenerator") as mock_gen_cls, \
                 patch("experiments.evaluate_covered_pipeline.QwenEmbedder") as mock_emb_cls, \
                 patch("experiments.evaluate_covered_pipeline.answer_covered", side_effect=wrong_identity_answer):

                mock_gen_cls.side_effect = lambda: SimpleNamespace(calls=[], get_digest=lambda: "a" * 64)
                mock_emb_cls.side_effect = lambda **k: SimpleNamespace(calls=[])

                report = run(
                    index,
                    queries,
                    units,
                    pages,
                    out_file,
                    generator_digest="a" * 64,
                    verifier_digest=VERIFIER_Q3_DIGEST,
                    embedding_digest=index["model_digest"],
                    transport=transport,
                    protocol=proto,
                )

        self.assertEqual(report["run_status"], "error")
        self.assertTrue(all(c["status"] == "error" for c in report["cases"]))
        self.assertTrue(all("Verifier model mismatch" in c["error"] for c in report["cases"]))


class TestCoveredPipelineExecution(unittest.TestCase):
    """Verifies end-to-end execution across 11 cases, separation of roles, and stats extraction."""

    def test_run_executes_exact_11_cases_with_isolated_roles(self):
        index, units, pages, queries = _sample_inputs()
        proto = _sample_protocol()
        proto["embedding_digest"] = index["model_digest"]
        transport = FakeRunnerTransport(gen_digest="a" * 64, emb_digest=index["model_digest"])

        def mock_answer_covered(idx, q_text, u, p, *, embedder=None, generator=None, verifier=None,
                               structured_retrieval=True):
            self.assertIs(structured_retrieval, False)
            cid = None
            for c in queries["cases"]:
                if c["query"] == q_text:
                    cid = c["id"]
                    break

            if cid == "b06":
                # Insufficient evidence with empty inventory -> 0 coverage calls
                return {
                    "schema_version": "preserved-query-coverage-v1",
                    "status": "insufficient_evidence",
                    "query": q_text,
                    "factual_report": {
                        "metadata": {
                            "planner_used": False,
                            "closed_note_tails": False,
                            "max_packages": 1,
                            "digest": "a" * 64,
                            "verifier_model": VERIFIER_Q3_MODEL,
                            "verifier_digest": VERIFIER_Q3_DIGEST,
                        },
                        "outcome": {
                            "status": "insufficient_evidence",
                            "full_result": {"grounding": {"trace": [{"step": "grounding_insufficient"}]}},
                        },
                    },
                    "coverage_audit": {"status": "completed", "skipped": True, "calls": []},
                    "publication": {
                        "schema_version": "coverage-publication-v1",
                        "status": "insufficient_evidence",
                        "claims": [],
                        "missing_parts": [{"query_span": q_text, "reason_code": "not_in_evaluated_evidence", "message": "Sin datos."}],
                        "reason": "Sin datos.",
                    },
                    "metadata": {
                        "profile": "preserved_query_coverage_v1",
                        "verifier_model": VERIFIER_Q3_MODEL,
                        "verifier_digest": VERIFIER_Q3_DIGEST,
                    },
                }

            # Simulating factual verification call and coverage call on verifier
            if verifier is not None:
                verifier.calls.append({
                    "stage": "verification",
                    "model": VERIFIER_Q3_MODEL,
                    "attempted": True,
                    "raw_response": {"prompt_eval_count": 50, "eval_count": 20, "eval_duration": 100, "total_duration": 200},
                })
                verifier.calls.append({
                    "stage": "query_coverage",
                    "model": VERIFIER_Q3_MODEL,
                    "attempted": True,
                    "raw_response": {"prompt_eval_count": 40, "eval_count": 15, "eval_duration": 80, "total_duration": 160},
                })
            if generator is not None:
                generator.calls.append({
                    "prompt": "mock prompt",
                    "raw_response": {"prompt_eval_count": 100, "eval_count": 50, "eval_duration": 500, "total_duration": 600},
                })

            status = "partial" if cid == "b08" else "answered"
            pub_claims = [{"text": "Pensión en 2015", "evidence": [{"source_id": "S1", "quote": "texto"}]}]
            missing = [{"query_span": "¿y qué pensión cobraré?", "reason_code": "not_in_evaluated_evidence", "message": "Falta dato individual."}] if cid == "b08" else []

            return {
                "schema_version": "preserved-query-coverage-v1",
                "status": status,
                "query": q_text,
                "factual_report": {
                    "metadata": {
                        "planner_used": False,
                        "closed_note_tails": False,
                        "max_packages": 1,
                        "digest": "a" * 64,
                        "verifier_model": VERIFIER_Q3_MODEL,
                        "verifier_digest": VERIFIER_Q3_DIGEST,
                    },
                    "outcome": {
                        "status": "answered",
                        "full_result": {"grounding": {"trace": [{"step": "canonical_factual_trace"}]}},
                        "expanded_hits": [{"rank": 1, "chunk": {"page_number": 42}}],
                    },
                },
                "coverage_audit": {
                    "status": "completed",
                    "skipped": False,
                    "calls": [{"mock": "coverage_call"}],
                },
                "publication": {
                    "schema_version": "coverage-publication-v1",
                    "status": status,
                    "claims": pub_claims,
                    "missing_parts": missing,
                    "reason": "",
                },
                "metadata": {
                    "profile": "preserved_query_coverage_v1",
                    "verifier_model": VERIFIER_Q3_MODEL,
                    "verifier_digest": VERIFIER_Q3_DIGEST,
                },
            }

        with tempfile.TemporaryDirectory(dir="tmp") as tmp_dir:
            out_file = Path(tmp_dir) / "covered_results.json"
            with patch("experiments.evaluate_covered_pipeline.RecordedGenerator") as mock_gen_cls, \
                 patch("experiments.evaluate_covered_pipeline.QwenEmbedder") as mock_emb_cls, \
                 patch("experiments.evaluate_covered_pipeline.answer_covered", side_effect=mock_answer_covered):

                def mock_gen_factory():
                    m = MagicMock(calls=[])
                    m.get_digest.return_value = "a" * 64
                    return m

                mock_gen_cls.side_effect = mock_gen_factory
                mock_emb_cls.side_effect = lambda **k: SimpleNamespace(calls=[])

                report = run(
                    index,
                    queries,
                    units,
                    pages,
                    out_file,
                    generator_digest="a" * 64,
                    verifier_digest=VERIFIER_Q3_DIGEST,
                    embedding_digest=index["model_digest"],
                    transport=transport,
                    protocol=proto,
                )

        self.assertEqual(report["schema_version"], SCHEMA_VERSION)
        self.assertEqual(report["run_status"], "completed")
        self.assertEqual(len(report["cases"]), 11)
        self.assertEqual([c["id"] for c in report["cases"]], list(EXPECTED_CASE_IDS))
        self.assertIsNotNone(report["metadata"]["protocol_hash"])

        # Check b06 had empty inventory and 0 coverage calls
        b06_case = next(c for c in report["cases"] if c["id"] == "b06")
        self.assertEqual(b06_case["status"], "insufficient_evidence")
        self.assertEqual(len(b06_case["coverage_calls"]), 0)
        self.assertIsNone(b06_case["per_role_stats"]["coverage"]["eval_count"])

        # Check b08 partial visible format and status coherence
        b08_case = next(c for c in report["cases"] if c["id"] == "b08")
        self.assertEqual(b08_case["status"], "partial")
        self.assertIn("Pensión en 2015", b08_case["formatted_answer"])
        self.assertIn("Partes no respondidas en las fuentes evaluadas", b08_case["formatted_answer"])
        self.assertIn("Falta dato individual", b08_case["formatted_answer"])

        # Check verifier vs coverage call separation without duplication
        b01_case = next(c for c in report["cases"] if c["id"] == "b01")
        self.assertEqual(len(b01_case["verifier_calls"]), 1)
        self.assertEqual(b01_case["verifier_calls"][0]["stage"], "verification")
        self.assertEqual(len(b01_case["coverage_calls"]), 1)
        self.assertEqual(b01_case["coverage_calls"][0]["stage"], "query_coverage")

        # Check trace contains only factual grounding trace
        self.assertEqual(b01_case["trace"], [{"step": "canonical_factual_trace"}])

        # Check per_role_stats separation
        self.assertEqual(b01_case["per_role_stats"]["generator"]["prompt_eval_count"], 100)
        self.assertEqual(b01_case["per_role_stats"]["factual_verifier"]["prompt_eval_count"], 50)
        self.assertEqual(b01_case["per_role_stats"]["coverage"]["prompt_eval_count"], 40)

    def test_cleanup_failure_retains_primary_error_and_continues(self):
        index, units, pages, queries = _sample_inputs()
        proto = _sample_protocol()
        proto["embedding_digest"] = index["model_digest"]
        transport = FakeRunnerTransport(gen_digest="a" * 64, emb_digest=index["model_digest"])

        def ver_with_failing_cleanup(**kwargs):
            v = MagicMock(calls=[], lifecycle_records=[])
            v.release_loaded.side_effect = RuntimeError("VRAM unload dropped")
            return v

        with tempfile.TemporaryDirectory(dir="tmp") as tmp_dir:
            out_file = Path(tmp_dir) / "cleanup_results.json"
            with patch("experiments.evaluate_covered_pipeline.RecordedGenerator") as mock_gen_cls, \
                 patch("experiments.evaluate_covered_pipeline.QwenEmbedder") as mock_emb_cls, \
                 patch("experiments.evaluate_covered_pipeline.Q3VerifierClient", side_effect=ver_with_failing_cleanup), \
                 patch("experiments.evaluate_covered_pipeline.answer_covered", side_effect=ValueError("pipeline error")):

                mock_gen_cls.side_effect = lambda: SimpleNamespace(calls=[], get_digest=lambda: "a" * 64)
                mock_emb_cls.side_effect = lambda **k: SimpleNamespace(calls=[])

                report = run(
                    index,
                    queries,
                    units,
                    pages,
                    out_file,
                    generator_digest="a" * 64,
                    verifier_digest=VERIFIER_Q3_DIGEST,
                    embedding_digest=index["model_digest"],
                    transport=transport,
                    protocol=proto,
                )

        self.assertEqual(report["run_status"], "error")
        self.assertEqual(len(report["cases"]), 11)
        self.assertTrue(all("pipeline error" in c["error"] for c in report["cases"]))
        self.assertTrue(all("cleanup: RuntimeError: VRAM unload dropped" in c["error"] for c in report["cases"]))


class TestQueriesRosterValidation(unittest.TestCase):
    """Verifies fixed 11-case roster constraints (b01..b08, n01..n03 ordered once each)."""

    def test_valid_roster_accepted(self):
        _, _, _, queries = _sample_inputs()
        cases = validate_queries_roster(queries)
        self.assertEqual(len(cases), 11)
        self.assertEqual([c["id"] for c in cases], list(EXPECTED_CASE_IDS))

    def test_rejects_wrong_count_or_order(self):
        _, _, _, queries = _sample_inputs()

        short_queries = copy.deepcopy(queries)
        short_queries["cases"].pop()
        with self.assertRaises(ValueError):
            validate_queries_roster(short_queries)

        long_queries = copy.deepcopy(queries)
        long_queries["cases"].append({"id": "extra", "query": "extra query?"})
        with self.assertRaises(ValueError):
            validate_queries_roster(long_queries)

        reversed_queries = copy.deepcopy(queries)
        reversed_queries["cases"].reverse()
        with self.assertRaises(ValueError):
            validate_queries_roster(reversed_queries)

    def test_rejects_duplicates_unexpected_ids_and_extra_keys(self):
        _, _, _, queries = _sample_inputs()

        dup_queries = copy.deepcopy(queries)
        dup_queries["cases"][1]["id"] = "b01"
        with self.assertRaises(ValueError):
            validate_queries_roster(dup_queries)

        unexpected_queries = copy.deepcopy(queries)
        unexpected_queries["cases"][0]["id"] = "x01"
        with self.assertRaises(ValueError):
            validate_queries_roster(unexpected_queries)

        extra_key_queries = copy.deepcopy(queries)
        extra_key_queries["cases"][0]["gold"] = "forbidden"
        with self.assertRaises(ValueError):
            validate_queries_roster(extra_key_queries)

        bool_queries = copy.deepcopy(queries)
        bool_queries["cases"][0]["id"] = True
        with self.assertRaises(ValueError):
            validate_queries_roster(bool_queries)


class TestRoleStatsHelper(unittest.TestCase):
    """Verifies stats extraction logic distinguishing None from 0."""

    def test_extract_role_stats(self):
        calls_valid = [
            {"raw_response": {"prompt_eval_count": 10, "eval_count": 5, "eval_duration": 100, "total_duration": 200}},
            {"raw_response": {"prompt_eval_count": 20, "eval_count": 10, "eval_duration": 200, "total_duration": 400}},
        ]
        stats = _extract_role_stats(calls_valid)
        self.assertEqual(stats["prompt_eval_count"], 30)
        self.assertEqual(stats["eval_count"], 15)
        self.assertEqual(stats["eval_duration"], 300)
        self.assertEqual(stats["total_duration"], 600)

        calls_missing_field = [{"raw_response": {"prompt_eval_count": 10}}]
        stats_missing = _extract_role_stats(calls_missing_field)
        self.assertEqual(stats_missing["prompt_eval_count"], 10)
        self.assertIsNone(stats_missing["eval_count"])
        self.assertIsNone(stats_missing["eval_duration"])
        self.assertIsNone(stats_missing["total_duration"])

        stats_empty = _extract_role_stats([])
        self.assertTrue(all(v is None for v in stats_empty.values()))


if __name__ == "__main__":
    unittest.main()
