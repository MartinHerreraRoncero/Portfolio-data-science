import copy
import io
import json
import unittest
from unittest.mock import patch

from experiments.compare_hybrid import main, run_comparison
from rag_pensiones.grounding import OllamaGrounder
from rag_pensiones.search import search as real_search
from test_compare_retrieval_k import _make_valid_index, _make_valid_dataset, MockEmbedder, MockGenerator, _make_standard_grounding_responses


class CompareHybridTests(unittest.TestCase):
    def test_retrieval_only_full_ranking_once_and_no_generator(self):
        index, dataset, embedder = _make_valid_index(6), _make_valid_dataset(), MockEmbedder()
        with patch("experiments.compare_hybrid.OllamaGrounder") as constructor, patch("experiments.compare_hybrid.search", wraps=real_search) as search:
            report = run_comparison(index, dataset, embedder=embedder)
            constructor.assert_not_called()
            self.assertEqual(search.call_count, 2)
            self.assertTrue(all(c.kwargs["top_k"] == 6 for c in search.call_args_list))
        self.assertEqual(len(embedder.calls), 2)
        self.assertEqual(report["cases"][0]["condition_order"], ["dense", "hybrid"])
        self.assertEqual(report["cases"][1]["condition_order"], ["hybrid", "dense"])
        for row in report["cases"]:
            self.assertTrue(all(len(c["retrieved_chunk_ids"]) == 4 for c in row["conditions"].values()))

    def test_gold_excluded_and_actual_grounder_result_validated(self):
        index, dataset = _make_valid_index(), _make_valid_dataset()
        dataset["cases"][0]["reference_answer"] = "GOLD_SECRET_SENTINEL"
        dataset["cases"][0]["slots"]["entity"] = ["GOLD_SECRET_SENTINEL"]
        generator = MockGenerator(_make_standard_grounding_responses() * 4)
        grounder = OllamaGrounder(model=generator.model, generator=generator)
        embedder = MockEmbedder()
        report = run_comparison(index, dataset, grounder, embedder, True)
        self.assertEqual(report["run_status"], "completed")
        self.assertNotIn("GOLD_SECRET_SENTINEL", json.dumps(generator.history))
        self.assertNotIn("GOLD_SECRET_SENTINEL", json.dumps(embedder.calls))
        self.assertTrue(all(c["status"] == "answered" for r in report["cases"] for c in r["conditions"].values()))
        self.assertEqual(report["summary"]["dense"]["returned_trace_chat_count"], 6)

    def test_errors_do_not_erase_retrieval_and_continue(self):
        generator = MockGenerator()
        grounder = OllamaGrounder(model=generator.model, generator=generator)
        with patch.object(grounder, "answer", side_effect=ValueError("budget exceeded")) as answer:
            report = run_comparison(_make_valid_index(), _make_valid_dataset(), grounder, MockEmbedder(), True)
        self.assertEqual(answer.call_count, 4)
        self.assertEqual(report["run_status"], "error")
        self.assertEqual(report["summary"]["dense"]["any_annotated_hit_count"], 2)
        self.assertEqual(report["summary"]["dense"]["error_count"], 2)
        self.assertEqual(report["summary"]["dense"]["valid_generation_count"], 0)

    def test_search_failure_has_no_fake_hits(self):
        with patch("experiments.compare_hybrid.search", side_effect=RuntimeError("network")):
            report = run_comparison(_make_valid_index(), _make_valid_dataset(), embedder=MockEmbedder())
        self.assertEqual(report["summary"]["hybrid"]["retrievals_evaluated"], 0)
        self.assertEqual(report["summary"]["hybrid"]["error_count"], 2)

    def test_pin_change_and_corrupt_grounder_return_fail(self):
        for bad_return in [None, {"status": "answered"}]:
            generator = MockGenerator()
            grounder = OllamaGrounder(model=generator.model, generator=generator)
            with patch.object(grounder, "answer", return_value=bad_return):
                report = run_comparison(_make_valid_index(), _make_valid_dataset(), grounder, MockEmbedder(), True)
            self.assertEqual(report["summary"]["dense"]["error_count"], 2)
        generator = MockGenerator()
        grounder = OllamaGrounder(model=generator.model, generator=generator)
        with patch.object(generator, "get_digest", side_effect=["first", "changed", "changed", "changed", "changed"]), patch.object(grounder, "answer") as answer:
            report = run_comparison(_make_valid_index(), _make_valid_dataset(), grounder, MockEmbedder(), True)
            answer.assert_not_called()
            self.assertEqual(report["run_status"], "error")

    def test_invalid_dataset_and_models_rejected_before_search(self):
        generator = MockGenerator()
        grounder = OllamaGrounder(model=generator.model, generator=generator)
        cases = [(None, None), (_make_valid_dataset(), "bad-model"), (_make_valid_dataset(), "empty-pin")]
        for dataset, bad in cases:
            with self.subTest(bad=bad), patch("experiments.compare_hybrid.search") as search:
                generator.model = "other" if bad == "bad-model" else grounder.model
                generator.digest = "" if bad == "empty-pin" else "valid"
                with self.assertRaises(ValueError): run_comparison(_make_valid_index(), dataset, grounder, MockEmbedder(), True)
                search.assert_not_called()

    def test_cli_json_and_error_exit(self):
        output = io.StringIO()
        with patch("experiments.compare_hybrid.run_comparison", return_value={"run_status": "error", "summary": {}}), patch("experiments.compare_hybrid.load_index", return_value={}), patch("sys.stdout", output):
            self.assertEqual(main(["--json"]), 1)
        self.assertEqual(json.loads(output.getvalue())["run_status"], "error")
        with patch("experiments.compare_hybrid.run_comparison", return_value={"run_status": "completed", "summary": {}}), patch("experiments.compare_hybrid.load_index", return_value={}), patch("sys.stdout", io.StringIO()):
            self.assertEqual(main(["--json"]), 0)
