"""Offline unit tests for the mixed-verifier RAG pipeline and evaluation runner."""

import copy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from experiments.compare_system import generate_arm
from experiments.evaluate_mixed_verifier import (
    ARMS,
    EXPECTED_CASE_COUNT,
    REQUIRED_FROZEN_MODULES,
    SCHEMA_VERSION,
    TOTAL_EVENTS,
    _extract_role_stats,
    main as runner_main,
    run,
    validate_pre_inference,
    validate_protocol,
    validate_schedule,
)
from experiments.preserved_q3_verifier import (
    Q3PreservedGrounder,
    Q3VerifierClient,
    VERIFIER_Q3_DIGEST,
    VERIFIER_Q3_MODEL,
    VERIFIER_Q3_PARAMETERS,
)
from experiments.query_preserved import (
    answer_preserved,
    format_answer,
    main as query_main,
)
from rag_pensiones.answer import DEFAULT_GENERATION_MODEL
from rag_pensiones.grounding import DEFAULT_PARAMETERS
from rag_pensiones.ingest import PROJECT_ROOT, validate_project_path
from rag_pensiones.qwen_embeddings import MODEL
from rag_pensiones.search import DEFAULT_INDEX_PATH, SearchHit, load_index, validate_index


def _sample_inputs():
    index = load_index(DEFAULT_INDEX_PATH)
    units = json.loads(Path("evaluation/reconstructed_units_catalog.json").read_text(encoding="utf-8"))
    pages = [json.loads(x) for x in Path("data/processed/airef_2025_pages.jsonl").read_text(encoding="utf-8").splitlines()]
    hits = [SearchHit(i + 1, 0.75, c) for i, (c, _) in enumerate(validate_index(index)[:4])]
    queries = json.loads(Path("evaluation/baseline_stability_queries.json").read_text(encoding="utf-8"))
    schedule = [{"arm": "legacy", "id": c["id"]} for c in queries["cases"]] + [
        {"arm": "q3", "id": c["id"]} for c in queries["cases"]
    ]
    return index, units, pages, hits, queries, schedule


class FakeRunnerTransport:
    """Mock transport for Ollama preflight verification."""

    def __init__(
        self,
        *,
        version: str = "0.40.11",
        gen_digest: str = "a" * 64,
        ver_digest: str = VERIFIER_Q3_DIGEST,
        emb_digest: str = "b" * 64,
        quant_level: str = "Q3_K_M",
        show_parameters: str | None = "presence_penalty 1.5\ntop_k 20\ntop_p 0.95\nmin_p 0.0\n",
    ) -> None:
        self.version = version
        self.gen_digest = gen_digest
        self.ver_digest = ver_digest
        self.emb_digest = emb_digest
        self.quant_level = quant_level
        self.show_parameters = show_parameters
        self.unloaded_models: list[str] = []
        self.resident_models: list[str] = []
        self.chat_calls: list[dict] = []

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
            "parameters": self.show_parameters,
        }

    def get_ps(self) -> dict:
        return {"models": [{"name": m, "model": m} for m in self.resident_models]}

    def unload(self, model: str) -> dict:
        self.unloaded_models.append(model)
        if model in self.resident_models:
            self.resident_models.remove(model)
        return {"status": "success"}

    def chat(self, messages: list[dict], response_schema: dict, options: dict | None = None) -> dict:
        call = {"messages": messages, "response_schema": response_schema, "options": options}
        self.chat_calls.append(call)
        if VERIFIER_Q3_MODEL not in self.resident_models:
            self.resident_models.append(VERIFIER_Q3_MODEL)
        return {
            "message": {
                "role": "assistant",
                "content": json.dumps({"status": "verified", "claims": []}),
            },
            "prompt_eval_count": 40,
            "eval_count": 20,
            "eval_duration": 500,
            "total_duration": 1000,
        }


class TestAnswerPreservedPipeline(unittest.TestCase):
    """Tests for direct query pipeline routing, truthfulness, and lifecycle."""

    def test_answer_preserved_legacy_default_routing(self):
        index, units, pages, hits, _, _ = _sample_inputs()
        gen = MagicMock(calls=[])
        gen.model = DEFAULT_GENERATION_MODEL
        gen.get_digest.return_value = "a" * 64
        outcome = {
            "status": "insufficient_evidence",
            "publication": {"status": "insufficient_evidence", "claims": [], "reason": "Sin datos."},
            "expanded_hits": [],
            "trace": [],
        }

        with patch("experiments.query_preserved.search", return_value=hits) as mock_search, \
             patch("experiments.query_preserved.generate_arm", return_value=outcome) as mock_gen_arm:
            report = answer_preserved(
                index,
                "¿Pregunta legacy?",
                units,
                pages,
                embedder=SimpleNamespace(calls=[]),
                generator=gen,
                verifier=None,
            )

        grounder = mock_gen_arm.call_args.args[0]
        self.assertNotIsInstance(grounder, Q3PreservedGrounder)
        self.assertIsNone(mock_gen_arm.call_args.kwargs.get("verification_policy"))
        self.assertEqual(report["metadata"]["profile"], "preserved_v1")
        self.assertEqual(report["metadata"]["model"], DEFAULT_GENERATION_MODEL)
        self.assertEqual(report["metadata"]["digest"], "a" * 64)
        self.assertNotIn("verifier_model", report["metadata"])
        self.assertFalse(report["metadata"]["planner_used"])
        self.assertEqual(report["metadata"]["max_packages"], 1)

    def test_answer_preserved_explicit_q3_releases_verifier_before_search(self):
        index, units, pages, hits, _, _ = _sample_inputs()
        transport = FakeRunnerTransport()
        client = Q3VerifierClient(expected_digest=VERIFIER_Q3_DIGEST, transport=transport)
        client.resident_models = [VERIFIER_Q3_MODEL]
        gen = MagicMock(calls=[])
        gen.model = DEFAULT_GENERATION_MODEL
        gen.get_digest.return_value = "a" * 64

        call_order = []

        def tracked_release(model):
            call_order.append(("release_loaded", model))
            client.lifecycle_records.append({"action": "release_loaded", "model": model})

        def tracked_search(*args, **kwargs):
            call_order.append(("search", kwargs.get("top_k")))
            return hits

        outcome = {
            "status": "answered",
            "publication": {"status": "answered", "claims": []},
            "expanded_hits": [],
            "trace": [],
        }

        with patch.object(client, "release_loaded", side_effect=tracked_release), \
             patch("experiments.query_preserved.search", side_effect=tracked_search), \
             patch("experiments.query_preserved.generate_arm", return_value=outcome) as mock_gen_arm:
            report = answer_preserved(
                index,
                "¿Pregunta Q3?",
                units,
                pages,
                embedder=SimpleNamespace(calls=[]),
                generator=gen,
                verifier=client,
            )

        self.assertEqual(call_order, [("release_loaded", VERIFIER_Q3_MODEL), ("search", 4)])
        grounder = mock_gen_arm.call_args.args[0]
        self.assertIsInstance(grounder, Q3PreservedGrounder)
        policy = mock_gen_arm.call_args.kwargs.get("verification_policy")
        self.assertIsNotNone(policy)
        self.assertEqual(policy["model"], VERIFIER_Q3_MODEL)
        self.assertEqual(policy["digest"], VERIFIER_Q3_DIGEST)
        self.assertEqual(policy["parameters"], VERIFIER_Q3_PARAMETERS)

        meta = report["metadata"]
        self.assertEqual(meta["profile"], "preserved_q3_verifier_v1")
        self.assertEqual(meta["model"], DEFAULT_GENERATION_MODEL)
        self.assertEqual(meta["digest"], "a" * 64)
        self.assertEqual(meta["verifier_model"], VERIFIER_Q3_MODEL)
        self.assertEqual(meta["verifier_digest"], VERIFIER_Q3_DIGEST)
        self.assertEqual(meta["verifier_parameters"], VERIFIER_Q3_PARAMETERS)
        self.assertEqual(meta["generator_parameters"], DEFAULT_PARAMETERS)

    def test_answer_preserved_input_validation_stops_before_models(self):
        index, units, pages, _, _, _ = _sample_inputs()
        transport = FakeRunnerTransport()
        client = Q3VerifierClient(expected_digest=VERIFIER_Q3_DIGEST, transport=transport)
        client.release_loaded = MagicMock()
        gen = MagicMock(calls=[])

        for bad_query in ["", "   ", None, 123, True]:
            with self.subTest(bad_query=bad_query), \
                 patch("experiments.query_preserved.search") as mock_search:
                with self.assertRaises(ValueError):
                    answer_preserved(index, bad_query, units, pages, generator=gen, verifier=client)
                client.release_loaded.assert_not_called()
                mock_search.assert_not_called()

        bad_index = copy.deepcopy(index)
        bad_index["model"] = "wrong-embedding-model"
        with patch("experiments.query_preserved.search") as mock_search:
            with self.assertRaises(ValueError):
                answer_preserved(bad_index, "Valida?", units, pages, generator=gen, verifier=client)
            client.release_loaded.assert_not_called()
            mock_search.assert_not_called()

    def test_format_answer_representation(self):
        ans_report = {
            "outcome": {
                "status": "answered",
                "publication": {
                    "status": "answered",
                    "claims": [
                        {
                            "text": "La pensión media sube.",
                            "evidence": [{"source_id": "S1", "quote": "Cita textual."}],
                        }
                    ],
                },
                "expanded_hits": [{"rank": 1, "chunk": {"page_number": 96}}],
            }
        }
        text = format_answer(ans_report)
        self.assertIn("La pensión media sube.", text)
        self.assertIn("[S1, pág. 96] Cita textual.", text)

        abstain_report = {
            "outcome": {
                "status": "insufficient_evidence",
                "publication": {"status": "insufficient_evidence", "claims": [], "reason": "No hay desglose."},
            }
        }
        self.assertEqual(format_answer(abstain_report), "No hay desglose.")

        error_report = {"outcome": {"status": "error", "error": "Fallo de conexión."}}
        self.assertEqual(format_answer(error_report), "Error: Fallo de conexión.")


class TestQueryCLI(unittest.TestCase):
    """Tests for ordinary query CLI defaults and flags."""

    def test_cli_default_uses_covered_pipeline(self):
        index, units, pages, _, _, _ = _sample_inputs()
        with patch("experiments.query_preserved.load_index", return_value=index), \
             patch("experiments.query_preserved.Q3VerifierClient") as mock_client_cls, \
             patch("experiments.query_coverage.answer_covered") as mock_answer_covered, \
             patch("experiments.query_preserved.checkpoint"):
            mock_answer_covered.return_value = {
                "schema_version": "preserved-query-coverage-v1",
                "status": "answered",
                "publication": {"schema_version": "coverage-publication-v1", "status": "answered", "claims": [], "missing_parts": []},
            }
            code = query_main(["--query", "¿Pregunta CLI?"])
            self.assertEqual(code, 0)
            mock_client_cls.assert_called_once_with(expected_digest=VERIFIER_Q3_DIGEST)
            self.assertIsNotNone(mock_answer_covered.call_args.kwargs.get("verifier"))

    def test_cli_factual_only_flag_uses_q3_answer_preserved(self):
        index, units, pages, _, _, _ = _sample_inputs()
        with patch("experiments.query_preserved.load_index", return_value=index), \
             patch("experiments.query_preserved.Q3VerifierClient") as mock_client_cls, \
             patch("experiments.query_preserved.answer_preserved") as mock_answer, \
             patch("experiments.query_preserved.checkpoint"):
            mock_answer.return_value = {
                "outcome": {"status": "answered", "publication": {"status": "answered", "claims": []}, "expanded_hits": []}
            }
            code = query_main(["--query", "¿Pregunta CLI?", "--factual-only"])
            self.assertEqual(code, 0)
            mock_client_cls.assert_called_once_with(expected_digest=VERIFIER_Q3_DIGEST)
            self.assertIsNotNone(mock_answer.call_args.kwargs.get("verifier"))

    def test_cli_legacy_flag_uses_none_verifier(self):
        index, units, pages, _, _, _ = _sample_inputs()
        with patch("experiments.query_preserved.load_index", return_value=index), \
             patch("experiments.query_preserved.Q3VerifierClient") as mock_client_cls, \
             patch("experiments.query_preserved.answer_preserved") as mock_answer, \
             patch("experiments.query_preserved.checkpoint"):
            mock_answer.return_value = {
                "outcome": {"status": "answered", "publication": {"status": "answered", "claims": []}, "expanded_hits": []}
            }
            code = query_main(["--query", "¿Pregunta CLI?", "--legacy-verifier"])
            self.assertEqual(code, 0)
            mock_client_cls.assert_not_called()
            self.assertIsNone(mock_answer.call_args.kwargs.get("verifier"))

    def test_cli_flags_mutually_exclusive(self):
        with self.assertRaises(SystemExit):
            query_main(["--query", "¿Pregunta CLI?", "--factual-only", "--legacy-verifier"])

    def test_cli_returns_one_on_error_and_zero_on_partial(self):
        index, units, pages, _, _, _ = _sample_inputs()
        with patch("experiments.query_preserved.load_index", return_value=index), \
             patch("experiments.query_preserved.Q3VerifierClient"), \
             patch("experiments.query_coverage.answer_covered") as mock_answer_covered, \
             patch("experiments.query_preserved.checkpoint"):
            mock_answer_covered.return_value = {
                "schema_version": "preserved-query-coverage-v1",
                "status": "partial",
                "publication": {"schema_version": "coverage-publication-v1", "status": "partial", "claims": [], "missing_parts": []},
            }
            code_partial = query_main(["--query", "¿Pregunta CLI?"])
            self.assertEqual(code_partial, 0)

            mock_answer_covered.return_value = {
                "schema_version": "preserved-query-coverage-v1",
                "status": "error",
                "error": "Pipeline failure",
            }
            code_error = query_main(["--query", "¿Pregunta CLI?"])
            self.assertEqual(code_error, 1)


class TestValidateSchedule(unittest.TestCase):
    """Tests strict 16-run paired schedule validation."""

    def test_valid_schedule_accepted(self):
        _, _, _, _, queries, schedule = _sample_inputs()
        validate_schedule(queries, schedule)

    def test_rejects_wrong_length_or_order(self):
        _, _, _, _, queries, schedule = _sample_inputs()

        with self.assertRaises(ValueError):
            validate_schedule(queries, schedule[:-1])

        with self.assertRaises(ValueError):
            validate_schedule(queries, schedule + [schedule[0]])

        interleaved = []
        cases = queries["cases"]
        for c in cases:
            interleaved.append({"arm": "legacy", "id": c["id"]})
            interleaved.append({"arm": "q3", "id": c["id"]})
        with self.assertRaises(ValueError):
            validate_schedule(queries, interleaved)

        reversed_arms = schedule[8:] + schedule[:8]
        with self.assertRaises(ValueError):
            validate_schedule(queries, reversed_arms)

    def test_rejects_booleans_duplicates_and_invalid_keys(self):
        _, _, _, _, queries, schedule = _sample_inputs()

        bad_bool_id = copy.deepcopy(schedule)
        bad_bool_id[0]["id"] = True
        with self.assertRaises(ValueError):
            validate_schedule(queries, bad_bool_id)

        bad_bool_arm = copy.deepcopy(schedule)
        bad_bool_arm[0]["arm"] = True
        with self.assertRaises(ValueError):
            validate_schedule(queries, bad_bool_arm)

        bad_unknown_id = copy.deepcopy(schedule)
        bad_unknown_id[0]["id"] = "unknown_case"
        with self.assertRaises(ValueError):
            validate_schedule(queries, bad_unknown_id)

        bad_extra_key = copy.deepcopy(schedule)
        bad_extra_key[0]["extra"] = 123
        with self.assertRaises(ValueError):
            validate_schedule(queries, bad_extra_key)


class TestValidateProtocol(unittest.TestCase):
    """Tests protocol validation and dependency freeze integrity."""

    def _sample_protocol(self):
        _, _, _, _, queries, schedule = _sample_inputs()
        frozen_files = {}
        for mod in REQUIRED_FROZEN_MODULES:
            p = validate_project_path(PROJECT_ROOT / mod)
            frozen_files[mod] = hashlib.sha256(p.read_bytes()).hexdigest()

        inputs = {
            "index": "data/processed/airef_2025_qwen3_4b_embeddings.json",
            "queries": "evaluation/baseline_stability_queries.json",
            "units": "evaluation/reconstructed_units_catalog.json",
            "pages": "data/processed/airef_2025_pages.jsonl",
        }
        for path_val in inputs.values():
            p = validate_project_path(PROJECT_ROOT / path_val)
            frozen_files[path_val] = hashlib.sha256(p.read_bytes()).hexdigest()

        return {
            "schema_version": SCHEMA_VERSION,
            "server_version": "0.40.11",
            "generator_digest": "a" * 64,
            "verifier_digest": VERIFIER_Q3_DIGEST,
            "embedding_digest": "b" * 64,
            "generator_parameters": copy.deepcopy(DEFAULT_PARAMETERS),
            "verifier_parameters": copy.deepcopy(VERIFIER_Q3_PARAMETERS),
            "input_paths": inputs,
            "input_hashes": {
                "index": "mock",
                "queries": "mock",
                "units": "mock",
                "pages": "mock",
                "schedule": "mock",
            },
            "frozen_files": frozen_files,
            "schedule": schedule,
        }

    def test_valid_protocol_passes(self):
        proto = self._sample_protocol()
        validate_protocol(proto)

    def test_rejects_schema_server_version_and_params_mismatch(self):
        proto = self._sample_protocol()
        bad_schema = copy.deepcopy(proto)
        bad_schema["schema_version"] = "wrong-schema"
        with self.assertRaises(ValueError):
            validate_protocol(bad_schema)

        bad_version = copy.deepcopy(proto)
        bad_version["server_version"] = "0.40.10"
        with self.assertRaises(ValueError):
            validate_protocol(bad_version)

        bad_ver_digest = copy.deepcopy(proto)
        bad_ver_digest["verifier_digest"] = "f" * 64
        with self.assertRaises(ValueError):
            validate_protocol(bad_ver_digest)

        bad_gen_params = copy.deepcopy(proto)
        bad_gen_params["generator_parameters"]["temperature"] = False
        with self.assertRaises(ValueError):
            validate_protocol(bad_gen_params)

        bad_ver_params = copy.deepcopy(proto)
        bad_ver_params["verifier_parameters"]["presence_penalty"] = 0.0
        with self.assertRaises(ValueError):
            validate_protocol(bad_ver_params)

    def test_rejects_gold_paths_in_protocol(self):
        proto = self._sample_protocol()
        bad_gold = copy.deepcopy(proto)
        bad_gold["input_paths"]["queries"] = "evaluation/baseline_stability_gold.json"
        with self.assertRaises(ValueError):
            validate_protocol(bad_gold)
        gold = "evaluation/baseline_stability_gold.json"
        proto["frozen_files"][gold] = hashlib.sha256(Path(gold).read_bytes()).hexdigest()
        with self.assertRaises(ValueError):
            validate_protocol(proto)

    def test_transitive_dependency_pins_cannot_be_removed(self):
        for module in ("experiments/compare_verifier_capacity.py", "experiments/compare_verifier_quantization.py", "experiments/reconstructed_units.py", "experiments/validate_reranker.py"):
            proto = self._sample_protocol()
            del proto["frozen_files"][module]
            with self.assertRaises(ValueError):
                validate_protocol(proto)

    def test_rejects_missing_code_pins_or_hash_mismatch(self):
        proto = self._sample_protocol()
        missing_pin = copy.deepcopy(proto)
        del missing_pin["frozen_files"]["experiments/preserved_q3_verifier.py"]
        with self.assertRaises(ValueError):
            validate_protocol(missing_pin)

        mismatch_hash = copy.deepcopy(proto)
        mismatch_hash["frozen_files"]["rag_pensiones/grounding.py"] = "0" * 64
        with self.assertRaises(ValueError):
            validate_protocol(mismatch_hash)


class TestValidatePreInference(unittest.TestCase):
    """Tests preflight checking of Ollama tags and version."""

    def test_valid_preflight_passes(self):
        transport = FakeRunnerTransport()
        validate_pre_inference(
            transport,
            server_version="0.40.11",
            generator_digest="a" * 64,
            verifier_digest=VERIFIER_Q3_DIGEST,
            embedding_digest="b" * 64,
        )

    def test_preflight_rejects_mismatches(self):
        transport = FakeRunnerTransport(version="0.40.10")
        with self.assertRaises(RuntimeError):
            validate_pre_inference(
                transport,
                server_version="0.40.11",
                generator_digest="a" * 64,
                verifier_digest=VERIFIER_Q3_DIGEST,
                embedding_digest="b" * 64,
            )

        transport2 = FakeRunnerTransport(ver_digest="0" * 64)
        with self.assertRaises(RuntimeError):
            validate_pre_inference(
                transport2,
                server_version="0.40.11",
                generator_digest="a" * 64,
                verifier_digest=VERIFIER_Q3_DIGEST,
                embedding_digest="b" * 64,
            )

        transport3 = FakeRunnerTransport(quant_level="Q4_K_M")
        with self.assertRaises(RuntimeError):
            validate_pre_inference(
                transport3,
                server_version="0.40.11",
                generator_digest="a" * 64,
                verifier_digest=VERIFIER_Q3_DIGEST,
                embedding_digest="b" * 64,
            )


class TestRunnerExecution(unittest.TestCase):
    """Tests run driver execution, checkpoints, stats distinction, and error tolerance."""

    def test_refuses_to_overwrite_existing_output(self):
        index, units, pages, _, queries, schedule = _sample_inputs()
        with tempfile.TemporaryDirectory(dir="tmp") as tmp_dir:
            out_file = Path(tmp_dir) / "results.json"
            out_file.write_text("existing content", encoding="utf-8")
            with self.assertRaises((ValueError, FileExistsError)):
                run(
                    index,
                    queries,
                    units,
                    pages,
                    out_file,
                    generator_digest="a" * 64,
                    verifier_digest=VERIFIER_Q3_DIGEST,
                    embedding_digest="b" * 64,
                    schedule=schedule,
                )

    def test_run_executes_16_events_releases_q3_in_finally_and_records_stats(self):
        index, units, pages, _, queries, schedule = _sample_inputs()
        transport = FakeRunnerTransport()

        def make_answer(idx, q_text, u, p, *, generator=None, embedder=None, verifier=None):
            arm = "q3" if verifier is not None else "legacy"
            meta = {
                "profile": "preserved_q3_verifier_v1" if arm == "q3" else "preserved_v1",
                "model": DEFAULT_GENERATION_MODEL,
                "digest": "a" * 64,
                "planner_used": False,
                "closed_note_tails": False,
                "max_packages": 1,
            }
            if arm == "q3":
                meta["verifier_model"] = VERIFIER_Q3_MODEL
                meta["verifier_digest"] = VERIFIER_Q3_DIGEST
                meta["verifier_parameters"] = VERIFIER_Q3_PARAMETERS

            if generator is not None:
                generator.calls.append({
                    "messages": [],
                    "raw_response": {
                        "prompt_eval_count": 10,
                        "eval_count": 5,
                        "eval_duration": 100,
                        "total_duration": 200,
                    },
                })
            if verifier is not None:
                verifier.calls.append({
                    "stage": "verification",
                    "raw_response": {
                        "prompt_eval_count": 30,
                        "eval_count": 15,
                        "eval_duration": 300,
                        "total_duration": 600,
                    },
                })
                if VERIFIER_Q3_MODEL not in transport.resident_models:
                    transport.resident_models.append(VERIFIER_Q3_MODEL)

            return {
                "schema_version": "preserved-direct-query-v1",
                "query": q_text,
                "metadata": meta,
                "retrieval": {"hits": [], "embedding_calls": []},
                "outcome": {
                    "status": "answered",
                    "publication": {"status": "answered", "claims": [], "reason": ""},
                    "expanded_hits": [],
                    "full_result": {"grounding": {"trace": [{"stage": "generation", "model": DEFAULT_GENERATION_MODEL}]}},
                },
            }

        with tempfile.TemporaryDirectory(dir="tmp") as tmp_dir:
            out_file = Path(tmp_dir) / "results.json"
            with patch("experiments.evaluate_mixed_verifier.RecordedGenerator") as mock_gen_cls, \
                 patch("experiments.evaluate_mixed_verifier.QwenEmbedder") as mock_emb_cls, \
                 patch("experiments.evaluate_mixed_verifier.answer_preserved", side_effect=make_answer):

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
                    embedding_digest="b" * 64,
                    schedule=schedule,
                    transport=transport,
                )

            self.assertTrue(out_file.exists())
            self.assertEqual(report["run_status"], "completed")
            self.assertEqual(len(report["cases"]), 16)

            self.assertTrue(all(c["trace"][0]["stage"] == "generation" for c in report["cases"]))

            for i in range(8):
                self.assertEqual(report["cases"][i]["arm"], "legacy")
                self.assertIsNone(report["cases"][i]["per_role_stats"]["verifier"])
                self.assertEqual(report["cases"][i]["per_role_stats"]["generator"]["prompt_eval_count"], 10)

            for i in range(8, 16):
                self.assertEqual(report["cases"][i]["arm"], "q3")
                self.assertEqual(report["cases"][i]["per_role_stats"]["verifier"]["prompt_eval_count"], 30)
                self.assertEqual(report["cases"][i]["per_role_stats"]["generator"]["prompt_eval_count"], 10)
                # Verify that release_loaded was called in finally
                self.assertIn(VERIFIER_Q3_MODEL, transport.unloaded_models)

    def test_stats_extraction_distinguishes_unknown_from_zero(self):
        calls_complete = [
            {"raw_response": {"prompt_eval_count": 10, "eval_count": 0, "eval_duration": 5, "total_duration": 15}},
            {"raw_response": {"prompt_eval_count": 20, "eval_count": 5, "eval_duration": 10, "total_duration": 25}},
        ]
        stats = _extract_role_stats(calls_complete)
        self.assertEqual(stats["prompt_eval_count"], 30)
        self.assertEqual(stats["eval_count"], 5)
        self.assertEqual(stats["eval_duration"], 15)
        self.assertEqual(stats["total_duration"], 40)

        calls_missing_key = [{"raw_response": {"prompt_eval_count": 10}}]
        stats_missing = _extract_role_stats(calls_missing_key)
        self.assertEqual(stats_missing["prompt_eval_count"], 10)
        self.assertIsNone(stats_missing["eval_count"])
        self.assertIsNone(stats_missing["eval_duration"])
        self.assertIsNone(stats_missing["total_duration"])

        calls_none = [{"raw_response": None}]
        stats_none = _extract_role_stats(calls_none)
        self.assertTrue(all(v is None for v in stats_none.values()))

    def test_technical_errors_and_digest_drift_continue_without_retry(self):
        index, units, pages, _, queries, schedule = _sample_inputs()
        transport = FakeRunnerTransport()

        call_count = 0

        def dynamic_mock(idx, q_text, u, p, *, generator=None, embedder=None, verifier=None):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise RuntimeError("Transport dropped")
            arm = "q3" if verifier is not None else "legacy"
            meta = {
                "profile": "preserved_q3_verifier_v1" if arm == "q3" else "preserved_v1",
                "model": DEFAULT_GENERATION_MODEL,
                "digest": "a" * 64,
                "planner_used": False,
                "closed_note_tails": False,
                "max_packages": 1,
            }
            if arm == "q3":
                meta["verifier_model"] = VERIFIER_Q3_MODEL
                meta["verifier_digest"] = VERIFIER_Q3_DIGEST
                meta["verifier_parameters"] = VERIFIER_Q3_PARAMETERS
            return {
                "schema_version": "preserved-direct-query-v1",
                "query": q_text,
                "metadata": meta,
                "retrieval": {"hits": [], "embedding_calls": []},
                "outcome": {
                    "status": "answered",
                    "publication": {"status": "answered", "claims": []},
                    "expanded_hits": [],
                },
            }

        with tempfile.TemporaryDirectory(dir="tmp") as tmp_dir:
            out_file = Path(tmp_dir) / "results.json"
            with patch("experiments.evaluate_mixed_verifier.RecordedGenerator") as mock_gen_cls, \
                 patch("experiments.evaluate_mixed_verifier.QwenEmbedder") as mock_emb_cls, \
                 patch("experiments.evaluate_mixed_verifier.answer_preserved", side_effect=dynamic_mock):

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
                    embedding_digest="b" * 64,
                    schedule=schedule,
                    transport=transport,
                )

            self.assertEqual(report["run_status"], "error")
            self.assertEqual(len(report["cases"]), 16)
            self.assertEqual(report["cases"][0]["status"], "error")
            self.assertIn("Transport dropped", report["cases"][0]["error"])
            self.assertEqual(report["cases"][1]["status"], "error")
            self.assertEqual(report["cases"][2]["status"], "answered")
            self.assertEqual(report["cases"][-1]["status"], "answered")


class TestCleanupContinuity(unittest.TestCase):
    def test_cleanup_failure_keeps_primary_error_and_all_events(self):
        index, units, pages, _, queries, schedule = _sample_inputs()
        gen = lambda: SimpleNamespace(calls=[], get_digest=lambda: "a" * 64)
        ver = lambda **kw: SimpleNamespace(calls=[], lifecycle_records=[], release_loaded=MagicMock(side_effect=RuntimeError("cleanup offline")))
        with tempfile.TemporaryDirectory(dir="tmp") as tmp_dir, \
             patch("experiments.evaluate_mixed_verifier.RecordedGenerator", side_effect=gen), \
             patch("experiments.evaluate_mixed_verifier.QwenEmbedder", side_effect=lambda **kw: SimpleNamespace(calls=[])), \
             patch("experiments.evaluate_mixed_verifier.Q3VerifierClient", side_effect=ver), \
             patch("experiments.evaluate_mixed_verifier.answer_preserved", side_effect=ValueError("primary failure")):
            report = run(index, queries, units, pages, Path(tmp_dir) / "out.json",
                         generator_digest="a" * 64, verifier_digest=VERIFIER_Q3_DIGEST,
                         embedding_digest="b" * 64, schedule=schedule, transport=FakeRunnerTransport())
        self.assertEqual(len(report["cases"]), 16)
        self.assertEqual(report["run_status"], "error")
        self.assertTrue(all("primary failure" in c["error"] for c in report["cases"]))
        self.assertTrue(all("cleanup offline" in c["error"] for c in report["cases"] if c["arm"] == "q3"))


if __name__ == "__main__":
    unittest.main()
