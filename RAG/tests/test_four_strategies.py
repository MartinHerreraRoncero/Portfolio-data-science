import copy
import json
import unittest
from unittest.mock import Mock, patch

from experiments.four_strategies import (ContextSelectorGrounder, SpecialistVerifierGrounder,
    StructuredGrounder, LocalReranker, context_schema, validate_record, extract_records, FIELDS)
from rag_pensiones.grounding import OllamaGrounder
from rag_pensiones.search import SearchHit, validate_index
from test_compare_retrieval_k import _make_valid_index, MockGenerator


def fact(chunk):
    return dict(entity="pensiones", period="2050", value="16,1", unit="%", scenario=None, quote=chunk.text)


class FourStrategiesTests(unittest.TestCase):
    def test_prompt_override_only_own_stages_without_mutation(self):
        messages = [{"role": "system", "content": "ORIGINAL"}, {"role": "user", "content": json.dumps({"fuentes": [{"source_id": "S1", "text": "Texto primero. Texto segundo."}]})}]
        for cls, stages in [(ContextSelectorGrounder, ["selection", "repair_selection"]), (SpecialistVerifierGrounder, ["verification", "reverification"])]:
            g = cls(generator=MockGenerator())
            for stage in ["selection", "generation", "verification", "repair_selection", "repair_generation", "reverification"]:
                before = copy.deepcopy(messages)
                with patch.object(OllamaGrounder, "_chat_step", return_value=({}, {})) as call:
                    g._chat_step(stage, messages, {}, "digest")
                    changed = call.call_args.args[1][0]["content"] != "ORIGINAL"
                    self.assertEqual(changed, stage in stages)
                self.assertEqual(messages, before)

    def test_all_adjacent_windows_include_late_antecedents(self):
        messages = [{"role": "user", "content": json.dumps({"fuentes": [{"source_id": "S1", "text": "Oracion primera. Oracion segunda. Oracion tercera. Oracion cuarta. Sujeto quinto. Valor sexto."}]})}]
        schema = context_schema(messages)
        choices = schema["anyOf"][0]["properties"]["evidence"]["items"]["anyOf"][0]["properties"]["quote"]["enum"]
        self.assertIn("Sujeto quinto. Valor sexto.", choices)
        self.assertTrue(all(len(q) <= 700 for q in choices))

    def test_records_require_literal_source_and_fields(self):
        chunk = validate_index(_make_valid_index())[0][0]
        good = fact(chunk)
        self.assertEqual(validate_record(good, chunk)["value"], "16,1")
        self.assertEqual(validate_record(dict(good, value="16.1"), chunk)["value"], "16.1")
        for change in [{"quote": "Invented evidence with 16,1 in 2050"}, {"value": "99,9"}, {"entity": "credito"}, {"period": 2050}, {"value": None}, {"unit": "dolares"}]:
            with self.subTest(change=change), self.assertRaises(ValueError): validate_record(dict(good, **change), chunk)
        # Literal cooccurrence is deliberately not an entailment guarantee.
        row = dict(good, period="2050", entity="pensiones")
        self.assertIsInstance(validate_record(row, chunk), dict)

    def test_extraction_keeps_errors_instead_of_silent_success(self):
        g = OllamaGrounder(generator=MockGenerator())
        with patch.object(g, "_chat_step", return_value=({"records": [dict(entity="X", period=None, value="1", unit=None, scenario=None, quote="Invented quote")]}, {"stage": "generation"})):
            result = extract_records(_make_valid_index(1), g, "digest")
        self.assertEqual(result["run_status"], "partial")
        self.assertEqual(result["records"], [])
        self.assertEqual(len(result["chunks"][0]["errors"]), 1)

    def test_extraction_valid_rows_receive_source_metadata(self):
        idx = _make_valid_index(1);chunk = validate_index(idx)[0][0]
        g = OllamaGrounder(generator=MockGenerator())
        with patch.object(g, "_chat_step", return_value=({"records": [fact(chunk)]}, {})):
            result = extract_records(idx, g, "digest")
        self.assertEqual(result["records"][0]["chunk_id"], chunk.chunk_id)
        self.assertEqual(result["run_status"], "completed")

    def test_structured_sql_filters_to_retrieved_sources(self):
        chunks = [c for c, _ in validate_index(_make_valid_index(2))]
        rows = [dict(fact(c), row_id=f"R{i}", chunk_id=c.chunk_id) for i, c in enumerate(chunks)]
        g = StructuredGrounder(rows, generator=MockGenerator());g.source_chunks = {"S1": chunks[0]}
        messages = [{"role": "system", "content": "old"}, {"role": "user", "content": json.dumps({"pregunta": "query", "fuentes": [{"source_id": "S1"}]})}]
        with patch.object(OllamaGrounder, "_chat_step", return_value=({}, {})) as call:
            g._chat_step("selection", messages, {}, "digest")
        data = json.loads(call.call_args.args[1][1]["content"])
        self.assertEqual([r["row_id"] for r in data["registros"]], ["R0"])
        self.assertEqual(g.events[-1]["included_rows"], ["R0"])
        g.db.close()

    def test_structured_budget_omissions_logged_whole_rows(self):
        chunk = validate_index(_make_valid_index(1))[0][0]
        rows = [dict(fact(chunk), row_id=f"R{i:03d}", chunk_id=chunk.chunk_id) for i in range(40)]
        g = StructuredGrounder(rows, generator=MockGenerator());g.source_chunks = {"S1": chunk}
        messages = [{"role": "system", "content": "old"}, {"role": "user", "content": json.dumps({"pregunta": "query", "fuentes": [{"source_id": "S1"}]})}]
        with patch.object(OllamaGrounder, "_chat_step", return_value=({}, {})) as call:g._chat_step("selection", messages, {}, "digest")
        self.assertLessEqual(len(json.dumps(call.call_args.args[1], ensure_ascii=False).encode()), 6500)
        self.assertTrue(g.events[-1]["omitted_rows_budget"])
        self.assertEqual(len(g.events[-1]["included_rows"]) + len(g.events[-1]["omitted_rows_budget"]), 40)
        g.db.close()

    def test_reranker_scores_count_finite_ties_and_cosine_preserved(self):
        chunks = [c for c, _ in validate_index(_make_valid_index(3))]
        hits = [SearchHit(i, .9 - i / 10, c) for i, c in enumerate(chunks, 1)]
        reranker = object.__new__(LocalReranker);reranker.model = Mock();reranker.last_scores = []
        reranker.model.predict.return_value = [1., 3., 3.]
        result = reranker.rank("query", hits)
        self.assertEqual([h.chunk.chunk_id for h in result], [chunks[1].chunk_id, chunks[2].chunk_id, chunks[0].chunk_id])
        self.assertEqual(result[0].score, hits[1].score)
        for scores in [[1.], [1., 2., float("nan")], [float("inf"), 2., 3.]]:
            reranker.model.predict.return_value = scores
            with self.assertRaises(ValueError): reranker.rank("query", hits)
