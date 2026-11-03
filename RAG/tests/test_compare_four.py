import copy
import io
import json
import unittest
from unittest.mock import Mock, patch

from experiments.compare_four import METHODS, run_comparison
from experiments.four_strategies import ContextSelectorGrounder, SpecialistVerifierGrounder, StructuredGrounder
from rag_pensiones.grounding import OllamaGrounder
from rag_pensiones.search import search as real_search
from test_compare_retrieval_k import _make_valid_index, _make_valid_dataset, MockGenerator, MockEmbedder


def grounders():
    constructors = [OllamaGrounder, ContextSelectorGrounder, OllamaGrounder, lambda **kw: StructuredGrounder([], **kw), SpecialistVerifierGrounder]
    return {m: c(generator=MockGenerator()) for m, c in zip(METHODS, constructors)}


def reranker():
    r = Mock();r.last_scores = [];r.rank.side_effect = lambda q, h: list(h)
    return r


class CompareFourTests(unittest.TestCase):
    def test_latin_order_full_ranking_once_top4_and_actual_validation(self):
        ds = _make_valid_dataset();first = ds["cases"][0]
        ds["cases"] = [dict(copy.deepcopy(first), id=f"q{i}") for i in range(10)]
        gs = grounders();rr = reranker()
        with patch("experiments.compare_four.search", wraps=real_search) as search, patch("sys.stderr", io.StringIO()):
            report = run_comparison(_make_valid_index(6), ds, gs, rr, MockEmbedder())
        self.assertEqual(search.call_count, 10)
        self.assertTrue(all(c.kwargs["top_k"] == 6 for c in search.call_args_list))
        for m in METHODS:
            self.assertEqual([sum(r["condition_order"][pos] == m for r in report["cases"]) for pos in range(5)], [2] * 5)
            self.assertEqual(report["summary"][m]["valid_generation_count"], 10)
        self.assertTrue(all(len(c["retrieved"]) == 4 for row in report["cases"] for c in row["conditions"].values()))
        self.assertEqual(rr.rank.call_count, 10)

    def test_gold_never_sent_to_model_or_reranker(self):
        ds = _make_valid_dataset();ds["cases"][0]["reference_answer"] = "SECRET_GOLD";ds["cases"][0]["slots"]["value"] = ["SECRET_GOLD"]
        gs = grounders();rr = reranker()
        with patch("sys.stderr", io.StringIO()):run_comparison(_make_valid_index(), ds, gs, rr, MockEmbedder())
        for g in gs.values():self.assertNotIn("SECRET_GOLD", json.dumps(g.generator.history))
        self.assertNotIn("SECRET_GOLD", repr(rr.rank.call_args_list))

    def test_errors_keep_hits_and_all_conditions_continue(self):
        gs = grounders()
        with patch.object(gs["selector"], "answer", side_effect=ValueError("budget")), patch("sys.stderr", io.StringIO()):
            report = run_comparison(_make_valid_index(), _make_valid_dataset(), gs, reranker(), MockEmbedder())
        self.assertEqual(report["run_status"], "error")
        self.assertEqual(report["summary"]["selector"]["error_count"], 2)
        self.assertEqual(report["summary"]["selector"]["annotated_hit_count"], 2)
        self.assertEqual(report["summary"]["baseline"]["valid_generation_count"], 2)

    def test_bad_digest_models_and_schema_rejected(self):
        for bad in ["model", "digest", "empty"]:
            gs = grounders()
            if bad == "model": gs["verifier"].generator.model = "wrong"
            elif bad == "digest": gs["verifier"].generator.digest = "different"
            else: gs["verifier"].generator.digest = ""
            with patch("experiments.compare_four.search") as search:
                with self.assertRaises(ValueError):run_comparison(_make_valid_index(), _make_valid_dataset(), gs, reranker(), MockEmbedder())
                search.assert_not_called()

    def test_reranker_error_no_fake_retrieval_hit_and_invalid_return_rejected(self):
        gs = grounders();rr = reranker();rr.rank.side_effect = RuntimeError("model error")
        with patch.object(gs["selector"], "answer", return_value={"status": "answered"}), patch("sys.stderr", io.StringIO()):
            report = run_comparison(_make_valid_index(), _make_valid_dataset(), gs, rr, MockEmbedder())
        self.assertEqual(report["summary"]["reranker"]["retrievals_evaluated"], 0)
        self.assertEqual(report["summary"]["selector"]["error_count"], 2)
