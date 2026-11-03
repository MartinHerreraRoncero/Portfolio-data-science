"""Pruebas del experimento; los mocks no evalúan verdad semántica."""

import copy
import contextlib
import io
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from experiments.diagnose_total_component import main, run_diagnosis
from rag_pensiones.answer import DEFAULT_GENERATION_MODEL
from rag_pensiones.grounding import OllamaGrounder
from test_grounding import FakeGrounderGenerator, _make_hit


class TestDiagnosis(unittest.TestCase):
    def setUp(self):
        self.hits = [_make_hit(i, f"Pasaje sintético número {i} con evidencia de prueba.") for i in range(1, 4)]
        self.index = {"model": "embeddinggemma:300m", "model_digest": "embedding_digest"}
        self.case = {"schema_version": "total-component-diagnostic-v1", "query": "¿Qué dice la fuente?",
                     "top_k": 2, "oracle_chunk_id": self.hits[2].chunk.chunk_id,
                     "additional_reference_chunk_ids": [self.hits[0].chunk.chunk_id],
                     "annotation": "Control sintético.", "verifier_controls": [{"id": "positive", "claim": "Dato de prueba.",
                     "chunk_id": self.hits[0].chunk.chunk_id, "quote": self.hits[0].chunk.text, "expected_supported": True}]}
        self.stats = {"prompt_eval_count": 2, "eval_count": 3, "eval_duration": 4,
                      "total_duration": 5, "load_duration": 1}
        self.grounder = Mock(model=DEFAULT_GENERATION_MODEL)
        self.grounder.generator = SimpleNamespace(model=DEFAULT_GENERATION_MODEL, get_digest=Mock(return_value="pinned"))
        self.grounder.answer.side_effect = lambda *a, **k: {"status": "insufficient_evidence", "claims": [],
                         "stats": self.stats.copy(), "grounding": {"trace": [{"stage": "selection"}]}}
        self.grounder.verify.return_value = {"digest": "pinned", "prompt_sha256": "a" * 64, "stats": self.stats.copy(),
                         "verdicts": [{"claim_id": "C1", "verdict": "supported", "reason": "Control.",
                         "evidence_verdicts": [{"evidence_id": "E1", "relation": "supports"}]}]}
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch("experiments.diagnose_total_component.validate_index",
                                      side_effect=lambda _: [(h.chunk, [1.0]) for h in self.hits]))
        self.search = self.stack.enter_context(patch("experiments.diagnose_total_component.search",
                                                   side_effect=lambda *a, **k: self.hits))
        self.boundary = self.stack.enter_context(patch("experiments.diagnose_total_component.validate_grounded_result"))

    def run_case(self, case=None):
        return run_diagnosis(self.index, self.case if case is None else case, grounder=self.grounder, embedder="fake")

    def test_missing_oracle_keeps_query_sources_ranks_and_addition_control(self):
        r = self.run_case()
        self.assertEqual(r["oracle_rank"], 3)
        self.assertFalse(r["oracle_in_top_k"])
        expected = [[1, 2], [3], [1, 3], [1, 2, 3]]
        for call, ranks in zip(self.grounder.answer.call_args_list, expected):
            self.assertEqual(call.args[0], self.case["query"])
            self.assertEqual([h.chunk.page_number // 10 for h in call.args[1]], ranks)
            self.assertEqual([h.rank for h in call.args[1]], list(range(1, len(ranks) + 1)))
            self.assertEqual(call.kwargs, {"expected_digest": "pinned"})
        self.assertEqual(r["calls"], {"searches": 1, "answers": 4, "verifier_controls": 1, "chats": 5})
        self.assertEqual(r["stats"]["load_duration"], 5)
        self.search.assert_called_once_with(self.index, self.case["query"], top_k=3, client="fake")

    def test_present_oracle_and_top_k_one_reuse_identical_inputs(self):
        self.case.update(top_k=1, oracle_chunk_id=self.hits[0].chunk.chunk_id, additional_reference_chunk_ids=[])
        r = self.run_case()
        self.assertEqual(r["calls"]["answers"], 2)
        self.assertEqual([c["reused_from"] for c in r["conditions"]], [None, "actual_top_k", "actual_top_k", None])
        self.assertEqual(r["conditions"][1]["latency_seconds"], 0)

    def test_controls_send_no_reference_labels_or_source_context(self):
        self.run_case()
        args, kwargs = self.grounder.verify.call_args
        self.assertEqual(set(args[0][0]), {"claim_id", "text", "evidence_ids"})
        self.assertEqual(set(args[1][0]), {"evidence_id", "source_id", "quote"})
        self.assertEqual(kwargs, {"expected_digest": "pinned"})

    def test_invalid_cases_fail_before_clients_and_search(self):
        mutations = [lambda c: c.update(top_k=True), lambda c: c.update(top_k=99),
                     lambda c: c.update(oracle_chunk_id="unknown"), lambda c: c.update(extra=float("nan")),
                     lambda c: c["verifier_controls"][0].update(quote="Cita ajena a la fuente original."),
                     lambda c: c["verifier_controls"][0].update(expected_supported=1),
                     lambda c: c["verifier_controls"].append(copy.deepcopy(c["verifier_controls"][0]))]
        with patch("experiments.diagnose_total_component.OllamaGrounder") as constructor:
            for mutate in mutations:
                case = copy.deepcopy(self.case)
                mutate(case)
                with self.subTest(case=case), self.assertRaises(ValueError):
                    run_diagnosis(self.index, case)
            constructor.assert_not_called()
        self.search.assert_not_called()

    def test_invalid_boundary_or_verifier_aborts_without_retry(self):
        self.boundary.side_effect = ValueError("bad node")
        with self.assertRaises(ValueError):
            self.run_case()
        self.assertEqual(self.grounder.answer.call_count, 1)
        self.grounder.verify.assert_not_called()
        self.boundary.side_effect = None
        for field, value in (("digest", "other"), ("prompt_sha256", "short"),
                             ("stats", dict(self.stats, load_duration=float("nan"))),
                             ("verdicts", [{"claim_id": "C2"}])):
            old = copy.deepcopy(self.grounder.verify.return_value)
            self.grounder.verify.return_value[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.run_case()
            self.grounder.verify.return_value = old

    def test_model_or_final_digest_change_aborts(self):
        self.grounder.model = "other"
        with self.assertRaises(ValueError):
            self.run_case()
        self.search.assert_not_called()
        self.grounder.model = DEFAULT_GENERATION_MODEL
        self.grounder.generator.get_digest.side_effect = ["pinned", "changed"]
        with self.assertRaises(ValueError):
            self.run_case()

    def test_real_grounder_fake_transport_and_no_duplicate_inference(self):
        self.hits = self.hits[:1]
        self.case.update(top_k=1, oracle_chunk_id=self.hits[0].chunk.chunk_id, additional_reference_chunk_ids=[])
        quote = self.hits[0].chunk.text
        verdict = {"verdicts": [{"claim_id": "C1", "verdict": "supported", "reason": "Control.",
                               "evidence_verdicts": [{"evidence_id": "E1", "relation": "supports"}]}]}
        generator = FakeGrounderGenerator([
            {"status": "selected", "evidence": [{"source_id": "S1", "quote": quote}], "reason": ""},
            {"status": "answered", "claims": [{"claim_id": "C1", "text": "Dato de prueba.", "evidence_ids": ["E1"]}], "reason": ""},
            verdict, verdict])
        self.boundary.side_effect = __import__("rag_pensiones.grounding", fromlist=["validate_grounded_result"]).validate_grounded_result
        r = run_diagnosis(self.index, self.case, grounder=OllamaGrounder(generator=generator), embedder="fake")
        self.assertEqual(r["calls"]["chats"], 4)
        self.assertEqual(generator.call_count, 4)
        self.assertEqual(r["conditions"][3]["reused_from"], "actual_top_k")

    def test_cli_rejects_external_paths_and_does_not_mask_errors(self):
        with contextlib.redirect_stderr(io.StringIO()) as stderr:
            code = main(["--case", str(__import__("pathlib").Path.cwd().parent / "outside.json")])
        self.assertEqual(code, 1)
        self.assertIn("Diagnostic failed", stderr.getvalue())
