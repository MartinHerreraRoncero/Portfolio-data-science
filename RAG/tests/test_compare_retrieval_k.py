"""Tests deterministas offline para la comparación de recuperación k=4 vs k=5."""

import io
import json
import unittest
from unittest.mock import patch

from experiments.compare_retrieval_k import (
    DATASET_SCHEMA_VERSION,
    DEFAULT_GENERATION_MODEL,
    compute_annotated_hits,
    compute_canonical_sha256,
    compute_slot_coverage,
    main,
    run_comparison,
    validate_dataset,
)
from rag_pensiones.chunking import ChunkRecord
from rag_pensiones.embeddings import DIMENSION, DOCUMENT_PREFIX, MODEL_NAME, QUERY_PREFIX, SCHEMA_VERSION as EMB_SCHEMA
from rag_pensiones.grounding import OllamaGrounder
from rag_pensiones.search import SearchHit


def _make_chunk(idx: int, text: str = "Texto sobre gasto de pensiones", cid: str | None = None) -> ChunkRecord:
    return ChunkRecord(
        chunk_id=cid or f"chunk_id_{idx}",
        doc_id=f"doc_{idx}",
        source_file=f"file_{idx}.pdf",
        published_at="2025-03-31",
        page_number=idx,
        start_char=0,
        end_char=len(text),
        text=text,
        strategy="page-local-v1:max_chars=1000:overlap_chars=150",
    )


def _make_valid_index(num_chunks: int = 5, digest: str = "fake_embed_digest_sha256") -> dict:
    chunks = []
    for i in range(1, num_chunks + 1):
        vec = [0.0] * DIMENSION
        vec[i % DIMENSION] = 1.0
        chunks.append({
            "chunk_id": f"chunk_id_{i}",
            "doc_id": f"doc_{i}",
            "source_file": f"file_{i}.pdf",
            "published_at": "2025-03-31",
            "page_number": i,
            "start_char": 0,
            "end_char": 100,
            "text": f"Gasto total en pensiones en 2050 alcanzara el 16,1% del PIB para caso {i}.",
            "strategy": "page-local-v1:max_chars=1000:overlap_chars=150",
            "embedding": vec,
        })
    return {
        "schema_version": EMB_SCHEMA,
        "model": MODEL_NAME,
        "model_digest": digest,
        "dimension": DIMENSION,
        "document_prefix": DOCUMENT_PREFIX,
        "query_prefix": QUERY_PREFIX,
        "input_sha256": "4cac2ee74838d6c5caa37efa0f27336689cd1ff12b6cbf9aa1948e1a36314cdc",
        "chunks": chunks,
    }


def _make_valid_dataset(cases: list[dict] | None = None) -> dict:
    if cases is None:
        cases = [
            {
                "id": "c1",
                "family": "f1",
                "query": "¿Cuál es el gasto en pensiones en 2050?",
                "reference_chunk_ids": ["chunk_id_1"],
                "slots": {"entity": ["pensiones"], "period": ["2050"], "unit": ["%"], "value": ["16,1"]},
                "reference_answer": "16,1% en 2050",
            },
            {
                "id": "c2",
                "family": "f2",
                "query": "¿Cuál es el gasto en 2070?",
                "reference_chunk_ids": ["chunk_id_2"],
                "slots": {"entity": ["pensiones"], "period": None, "unit": None, "value": ["14,7"]},
                "reference_answer": "14,7%",
            },
        ]
    return {
        "schema_version": DATASET_SCHEMA_VERSION,
        "annotation": "Anotación de prueba para desarrollo k4 vs k5.",
        "cases": cases,
    }


class MockEmbedder:
    def __init__(self, digest: str = "fake_embed_digest_sha256") -> None:
        self.digest = digest
        self.calls: list[list[str]] = []

    def get_digest(self) -> str:
        return self.digest

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        vec = [0.0] * DIMENSION
        vec[1] = 1.0
        return [vec]


class MockGenerator:
    def __init__(self, responses: list[dict] | None = None, digest: str = "fake_gen_digest_sha256") -> None:
        self.digest = digest
        self.model = DEFAULT_GENERATION_MODEL
        self.responses = list(responses or [])
        self.call_count = 0
        self.history: list[list[dict]] = []

    def get_digest(self) -> str:
        return self.digest

    def chat(self, messages: list[dict], response_schema: dict) -> dict:
        self.history.append(messages)
        if self.call_count >= len(self.responses):
            resp_payload = {"status": "insufficient_evidence", "evidence": [], "reason": "No evidence"}
        else:
            resp_payload = self.responses[self.call_count]
        self.call_count += 1
        return {
            "model": self.model,
            "done": True,
            "done_reason": "stop",
            "message": {"role": "assistant", "content": json.dumps(resp_payload, ensure_ascii=False)},
            "prompt_eval_count": 100,
            "eval_count": 50,
            "eval_duration": 1000000,
            "total_duration": 2000000,
        }


def _make_standard_grounding_responses(quote: str = "Gasto total en pensiones en 2050 alcanzara el 16,1% del PIB") -> list[dict]:
    return [
        {"status": "selected", "evidence": [{"source_id": "S1", "quote": quote}], "reason": ""},
        {"status": "answered", "claims": [{"claim_id": "C1", "text": "El gasto sera 16,1%.", "evidence_ids": ["E1"]}], "reason": ""},
        {"verdicts": [{"claim_id": "C1", "verdict": "supported", "reason": "Apoyo.", "evidence_verdicts": [{"evidence_id": "E1", "relation": "supports"}]}]},
    ]


class TestCompareRetrievalK(unittest.TestCase):
    def test_k4_k5_prefixes_and_single_search(self) -> None:
        index = _make_valid_index(6)
        ds = _make_valid_dataset([_make_valid_dataset()["cases"][0]])
        embedder = MockEmbedder()
        responses = _make_standard_grounding_responses() * 2
        gen = MockGenerator(responses)
        grounder = OllamaGrounder(generator=gen)

        res = run_comparison(index, ds, grounder=grounder, embedder=embedder)
        self.assertEqual(res["run_status"], "completed")
        self.assertEqual(embedder.calls.__len__(), 1)
        self.assertEqual(len(res["cases"][0]["ranking_metadata"]), 6)
        case_conds = res["cases"][0]["conditions"]
        self.assertEqual(case_conds["k4"]["status"], "answered")
        self.assertEqual(case_conds["k5"]["status"], "answered")
        self.assertNotIn("recall_global", json.dumps(res))
        self.assertNotIn("recall", case_conds["k4"]["annotated_passages_hit"])

    def test_counterbalance_order(self) -> None:
        index = _make_valid_index(5)
        ds = _make_valid_dataset()
        embedder = MockEmbedder()
        responses = _make_standard_grounding_responses() * 4
        gen = MockGenerator(responses)
        grounder = OllamaGrounder(generator=gen)

        res = run_comparison(index, ds, grounder=grounder, embedder=embedder)
        c0 = res["cases"][0]
        c1 = res["cases"][1]
        self.assertEqual(c0["condition_order"], ["k4", "k5"])
        self.assertEqual(c0["conditions"]["k4"]["order"], 1)
        self.assertEqual(c0["conditions"]["k5"]["order"], 2)
        self.assertEqual(c1["condition_order"], ["k5", "k4"])
        self.assertEqual(c1["conditions"]["k5"]["order"], 1)
        self.assertEqual(c1["conditions"]["k4"]["order"], 2)

    def test_no_gold_sent_to_clients(self) -> None:
        index = _make_valid_index(5)
        ds = _make_valid_dataset([_make_valid_dataset()["cases"][0]])
        gold_case = ds["cases"][0]
        embedder = MockEmbedder()
        responses = _make_standard_grounding_responses() * 2
        gen = MockGenerator(responses)
        grounder = OllamaGrounder(generator=gen)

        run_comparison(index, ds, grounder=grounder, embedder=embedder)
        gold_strings = [gold_case["reference_answer"], gold_case["id"], gold_case["family"]]
        for texts in embedder.calls:
            for t in texts:
                for gs in gold_strings:
                    self.assertNotIn(gs, t)
        for msgs in gen.history:
            dumped = json.dumps(msgs, ensure_ascii=False)
            for gs in gold_strings:
                self.assertNotIn(gs, dumped)

    def test_slot_coverage_none_and_cooccurrence_no_entailment(self) -> None:
        slots = {
            "entity": ["Gasto en pensiones", "pensiones"],
            "period": None,
            "unit": ["%", "por ciento"],
            "value": ["16,1"],
        }
        quotes = ["El gasto en pensiones alcanzara el 16,1% en el escenario base."]
        cov = compute_slot_coverage(slots, quotes)
        self.assertTrue(cov["fields"]["entity"])
        self.assertIsNone(cov["fields"]["period"])
        self.assertTrue(cov["fields"]["unit"])
        self.assertTrue(cov["fields"]["value"])
        self.assertEqual(cov["applicable_slots"], 3)
        self.assertEqual(cov["covered_slots"], 3)
        self.assertTrue(cov["all_applicable_covered"])

        # Sin coincidencia de valor
        quotes_no_val = ["El gasto en pensiones alcanzara un monto elevado."]
        cov_no_val = compute_slot_coverage(slots, quotes_no_val)
        self.assertFalse(cov_no_val["fields"]["value"])
        self.assertFalse(cov_no_val["all_applicable_covered"])

    def test_invalid_dataset_rejected_before_clients(self) -> None:
        index = _make_valid_index(5)
        embedder = MockEmbedder()
        gen = MockGenerator()
        grounder = OllamaGrounder(generator=gen)

        # 1. Slot obligatorio entity ausente o None
        bad_ds1 = _make_valid_dataset([{"id": "c1", "family": "f", "query": "q", "reference_chunk_ids": ["chunk_id_1"],
                                        "slots": {"entity": None, "period": None, "unit": None, "value": ["1"]}, "reference_answer": "a"}])
        with self.assertRaises(ValueError):
            run_comparison(index, bad_ds1, grounder=grounder, embedder=embedder)

        # 2. Reference chunk_id no presente en índice
        bad_ds2 = _make_valid_dataset([{"id": "c1", "family": "f", "query": "q", "reference_chunk_ids": ["unknown_chunk"],
                                        "slots": {"entity": ["e"], "period": None, "unit": None, "value": ["1"]}, "reference_answer": "a"}])
        with self.assertRaises(ValueError):
            run_comparison(index, bad_ds2, grounder=grounder, embedder=embedder)

        # 3. Índice con menos de 5 fragmentos
        small_index = _make_valid_index(4)
        with self.assertRaises(ValueError):
            run_comparison(small_index, _make_valid_dataset(), grounder=grounder, embedder=embedder)

        # Verificar que ningún cliente fue contactado
        self.assertEqual(len(embedder.calls), 0)
        self.assertEqual(gen.call_count, 0)

    def test_generation_error_continues_and_reports_error_status(self) -> None:
        index = _make_valid_index(5)
        ds = _make_valid_dataset([_make_valid_dataset()["cases"][0]])
        embedder = MockEmbedder()

        class FaultyGenerator(MockGenerator):
            def __init__(self, responses: list[dict], digest: str = "fake_gen_digest_sha256") -> None:
                super().__init__(responses, digest)
                self.failed_once = False

            def chat(self, messages: list[dict], response_schema: dict) -> dict:
                if not self.failed_once:
                    self.failed_once = True
                    raise RuntimeError("Fallo transitorio de Ollama")
                return super().chat(messages, response_schema)

        responses = _make_standard_grounding_responses()
        gen = FaultyGenerator(responses)
        grounder = OllamaGrounder(generator=gen)

        res = run_comparison(index, ds, grounder=grounder, embedder=embedder)
        self.assertEqual(res["run_status"], "error")
        conds = res["cases"][0]["conditions"]
        self.assertEqual(conds["k4"]["status"], "error")
        self.assertIn("Fallo transitorio", conds["k4"]["error"])
        self.assertEqual(conds["k5"]["status"], "answered")

    def test_search_error_marks_both_conditions_without_metrics(self) -> None:
        index = _make_valid_index(5)
        ds = _make_valid_dataset([_make_valid_dataset()["cases"][0]])
        embedder = MockEmbedder()
        embedder.embed_texts = lambda texts: (_ for _ in ()).throw(RuntimeError("Embedding model offline"))  # type: ignore
        gen = MockGenerator()
        grounder = OllamaGrounder(generator=gen)

        res = run_comparison(index, ds, grounder=grounder, embedder=embedder)
        self.assertEqual(res["run_status"], "error")
        conds = res["cases"][0]["conditions"]
        self.assertEqual(conds["k4"]["status"], "error")
        self.assertEqual(conds["k5"]["status"], "error")
        self.assertNotIn("slot_coverage", conds["k4"])
        self.assertNotIn("annotated_passages_hit", conds["k4"])

    def test_pin_changed_aborts_or_records_global_error(self) -> None:
        index = _make_valid_index(5)
        ds = _make_valid_dataset([_make_valid_dataset()["cases"][0]])
        embedder = MockEmbedder()

        class ShiftingGenerator(MockGenerator):
            def get_digest(self) -> str:
                if self.call_count >= 1:
                    return "mutated_gen_digest_sha256"
                return self.digest

        responses = _make_standard_grounding_responses() * 2
        gen = ShiftingGenerator(responses)
        grounder = OllamaGrounder(generator=gen)

        res = run_comparison(index, ds, grounder=grounder, embedder=embedder)
        self.assertEqual(res["run_status"], "error")

    def test_cli_json_and_exit_1_on_error(self) -> None:
        # 1. Error de archivo: salida con exit code 1
        with patch("sys.stdout", new_callable=io.StringIO):
            exit_code = main(["--json", "--dataset", "evaluation/non_existent.json"])
            self.assertEqual(exit_code, 1)

        # 2. Ejecución con estado error: emite JSON válido y sale con código 1
        fake_report = {"schema_version": "retrieval-k-comparison-v1", "run_status": "error", "cases": []}
        ds_bytes = json.dumps(_make_valid_dataset(), ensure_ascii=False).encode("utf-8")
        with patch("experiments.compare_retrieval_k.load_index", return_value=_make_valid_index(5)), \
             patch("experiments.compare_retrieval_k.run_comparison", return_value=fake_report), \
             patch("pathlib.Path.read_bytes", return_value=ds_bytes), \
             patch("pathlib.Path.is_file", return_value=True), \
             patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            exit_code = main(["--json"])
            self.assertEqual(exit_code, 1)
            parsed = json.loads(mock_stdout.getvalue())
            self.assertEqual(parsed["run_status"], "error")

    def test_real_grounder_with_fake_transport(self) -> None:
        index = _make_valid_index(5)
        ds = _make_valid_dataset([_make_valid_dataset()["cases"][0]])
        embedder = MockEmbedder()
        responses = _make_standard_grounding_responses() * 2
        gen = MockGenerator(responses)
        grounder = OllamaGrounder(generator=gen)

        res = run_comparison(index, ds, grounder=grounder, embedder=embedder)
        self.assertEqual(res["run_status"], "completed")
        self.assertEqual(res["metadata"]["model"], DEFAULT_GENERATION_MODEL)
        self.assertEqual(res["summary"]["k4"]["answered_count"], 1)
        self.assertEqual(res["summary"]["k5"]["answered_count"], 1)


class TestComparisonBoundaryRegressions(unittest.TestCase):
    def test_retrieval_metric_survives_generation_error(self):
        from experiments.compare_retrieval_k import _compute_summary

        conditions = {"k4": {"status": "error", "annotated_passages_hit": {"annotated_hits": 1, "all_annotated_hit": True}},
                      "k5": {"status": "error", "annotated_passages_hit": {"annotated_hits": 0, "all_annotated_hit": False}}}
        summary = _compute_summary([{"conditions": conditions}])
        self.assertEqual(summary["k4"]["any_annotated_hit_count"], 1)
        self.assertEqual(summary["k4"]["retrievals_evaluated"], 1)
        self.assertEqual(summary["k4"]["valid_executions"], 0)
        self.assertEqual(summary["k5"]["any_annotated_hit_count"], 0)

    def test_client_model_and_empty_pin_fail_before_search(self):
        for changed in ("model", "digest"):
            gen = MockGenerator()
            grounder = OllamaGrounder(generator=gen)
            setattr(gen, changed, "other" if changed == "model" else None)
            embedder = MockEmbedder()
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                run_comparison(_make_valid_index(5), _make_valid_dataset(), grounder, embedder)
            self.assertEqual(len(embedder.calls), 0)


if __name__ == "__main__":
    unittest.main()
