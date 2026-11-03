"""Tests exhaustivos y deterministas offline para la evaluación de recuperación semántica."""

import copy
import io
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from rag_pensiones.chunking import ChunkRecord
from rag_pensiones.embeddings import (
    DIMENSION,
    DOCUMENT_PREFIX,
    MODEL_NAME,
    QUERY_PREFIX,
    SCHEMA_VERSION,
)
from rag_pensiones.evaluate import (
    DATASET_SCHEMA_VERSION,
    DEFAULT_DATASET_PATH,
    EVALUATION_SCHEMA_VERSION,
    collapse_spaces,
    compute_canonical_sha256,
    evaluate,
    load_dataset,
    main,
    validate_dataset,
)
from rag_pensiones.ingest import PROJECT_ROOT
from rag_pensiones.search import DEFAULT_INDEX_PATH, LEGACY_INDEX_PATH, load_index


class MockOllamaClient:
    """Cliente simulado offline para pruebas de evaluación sin contactar red."""

    def __init__(
        self,
        digest: str = "fake_test_digest_sha256",
        query_vector_map: dict[str, list[float]] | None = None,
        default_vector: list[float] | None = None,
    ) -> None:
        self.digest = digest
        self.query_vector_map = query_vector_map or {}
        if default_vector is not None:
            self.default_vector = default_vector
        else:
            self.default_vector = [1.0] + [0.0] * (DIMENSION - 1)
        self.calls: list[str] = []

    def get_digest(self) -> str:
        self.calls.append("get_digest")
        return self.digest

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.calls.append("embed_texts")
        vectors: list[list[float]] = []
        for text in texts:
            clean = text
            if clean.startswith(QUERY_PREFIX):
                clean = clean[len(QUERY_PREFIX):]
            if clean in self.query_vector_map:
                vectors.append(list(self.query_vector_map[clean]))
            else:
                vectors.append(list(self.default_vector))
        return vectors


def _make_chunk(
    idx: int,
    text: str,
    page_number: int = 1,
    doc_id: str = "doc_test_1",
    vec: list[float] | None = None,
) -> dict:
    if vec is None:
        vec = [0.0] * DIMENSION
        vec[idx % DIMENSION] = 1.0
    return {
        "chunk_id": f"chunk_id_{idx}",
        "doc_id": doc_id,
        "source_file": "test_doc.pdf",
        "published_at": "2025-03-31",
        "page_number": page_number,
        "start_char": 0,
        "end_char": len(text),
        "text": text,
        "strategy": "page-local-v1:max_chars=1000:overlap_chars=150",
        "embedding": list(vec),
    }


def _make_index(chunks: list[dict], digest: str = "fake_test_digest_sha256") -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "model": MODEL_NAME,
        "model_digest": digest,
        "dimension": DIMENSION,
        "document_prefix": DOCUMENT_PREFIX,
        "query_prefix": QUERY_PREFIX,
        "input_sha256": "4cac2ee74838d6c5caa37efa0f27336689cd1ff12b6cbf9aa1948e1a36314cdc",
        "chunks": chunks,
    }


def _make_case(
    case_id: str,
    question: str,
    answerable: bool = True,
    expected_answer: str = "Respuesta esperada.",
    evidence: list[dict] | None = None,
) -> dict:
    if evidence is None:
        if answerable:
            evidence = [
                {
                    "doc_id": "doc_test_1",
                    "page_number": 1,
                    "quotes": ["texto relevante"],
                }
            ]
        else:
            evidence = []
    return {
        "id": case_id,
        "question": question,
        "answerable": answerable,
        "expected_answer": expected_answer,
        "evidence": evidence,
    }


def _make_dataset(cases: list[dict], desc: str = "Dataset de prueba") -> dict:
    return {
        "schema_version": DATASET_SCHEMA_VERSION,
        "description": desc,
        "cases": cases,
    }


class TestEvaluateMetricsOffline(unittest.TestCase):
    """Pruebas deterministas de ranking, métricas [1, 2, null] y exclusión de abstención."""

    def test_known_ranking_1_2_null_metrics(self):
        """Verifica exactamente el cálculo conocido: ranks [1, 2, null] => Hit@1=1/3, Hit@k=2/3, MRR@k=0.5."""
        # Creamos 5 vectores base ortonormales
        v1 = [1.0] + [0.0] * (DIMENSION - 1)
        v2 = [0.0, 1.0] + [0.0] * (DIMENSION - 2)
        v3 = [0.0, 0.0, 1.0] + [0.0] * (DIMENSION - 3)
        v4 = [0.0, 0.0, 0.0, 1.0] + [0.0] * (DIMENSION - 4)
        v5 = [0.0, 0.0, 0.0, 0.0, 1.0] + [0.0] * (DIMENSION - 5)

        # Chunks en el índice:
        # chunk 1: text con gold de caso 1 (vector v1)
        # chunk 2: text con gold de caso 2 (vector v2)
        # chunk 3: text con gold de caso 3 (vector v3)
        # chunks 4 y 5: distractores (vectores v4, v5)
        chunks = [
            _make_chunk(1, "Texto con evidencia para el caso 1.", page_number=1, vec=v1),
            _make_chunk(2, "Texto con evidencia para el caso 2.", page_number=2, vec=v2),
            _make_chunk(3, "Texto con evidencia para el caso 3.", page_number=3, vec=v3),
            _make_chunk(4, "Distractor cuatro.", page_number=4, vec=v4),
            _make_chunk(5, "Distractor cinco.", page_number=5, vec=v5),
        ]
        index = _make_index(chunks)

        cases = [
            _make_case(
                "c1",
                "Pregunta caso 1",
                answerable=True,
                evidence=[{"doc_id": "doc_test_1", "page_number": 1, "quotes": ["evidencia para el caso 1"]}],
            ),
            _make_case(
                "c2",
                "Pregunta caso 2",
                answerable=True,
                evidence=[{"doc_id": "doc_test_1", "page_number": 2, "quotes": ["evidencia para el caso 2"]}],
            ),
            _make_case(
                "c3",
                "Pregunta caso 3",
                answerable=True,
                evidence=[{"doc_id": "doc_test_1", "page_number": 3, "quotes": ["evidencia para el caso 3"]}],
            ),
        ]
        dataset = _make_dataset(cases)

        # Para query 1: vector v1 -> chunk 1 en rank 1
        # Para query 2: vector (v4 + 0.5*v2) -> chunk 4 es rank 1, chunk 2 es rank 2
        v_q2 = [0.0, 0.5, 0.0, 1.0] + [0.0] * (DIMENSION - 4)
        # Para query 3: vector (v4 + 0.8*v5 + 0.6*v1) -> chunks 4, 5, 1 están en top 3; chunk 3 no entra en top-3 (rank null)
        v_q3 = [0.6, 0.0, 0.0, 1.0, 0.8] + [0.0] * (DIMENSION - 5)

        vector_map = {
            "Pregunta caso 1": v1,
            "Pregunta caso 2": v_q2,
            "Pregunta caso 3": v_q3,
        }
        client = MockOllamaClient(query_vector_map=vector_map)

        report = evaluate(index, dataset, top_k=3, client=client)

        metrics = report["metrics"]
        self.assertEqual(metrics["denominator"], 3)
        self.assertAlmostEqual(metrics["hit_at_1"], 1.0 / 3.0, places=6)
        self.assertAlmostEqual(metrics["hit_at_k"], 2.0 / 3.0, places=6)
        self.assertAlmostEqual(metrics["mrr_at_k"], 0.5, places=6)

        c_reports = report["cases"]
        self.assertEqual(c_reports[0]["first_relevant_rank"], 1)
        self.assertEqual(c_reports[0]["hit_at_1"], 1)
        self.assertEqual(c_reports[0]["hit_at_k"], 1)
        self.assertAlmostEqual(c_reports[0]["reciprocal_rank"], 1.0, places=6)

        self.assertEqual(c_reports[1]["first_relevant_rank"], 2)
        self.assertEqual(c_reports[1]["hit_at_1"], 0)
        self.assertEqual(c_reports[1]["hit_at_k"], 1)
        self.assertAlmostEqual(c_reports[1]["reciprocal_rank"], 0.5, places=6)

        self.assertIsNone(c_reports[2]["first_relevant_rank"])
        self.assertEqual(c_reports[2]["hit_at_1"], 0)
        self.assertEqual(c_reports[2]["hit_at_k"], 0)
        self.assertAlmostEqual(c_reports[2]["reciprocal_rank"], 0.0, places=6)

    def test_unanswerable_exclusion_from_metrics(self):
        """Verifica que las preguntas de abstención se buscan pero se excluyen de métricas."""
        v1 = [1.0] + [0.0] * (DIMENSION - 1)
        chunks = [_make_chunk(1, "Texto respondible", page_number=1, vec=v1)]
        index = _make_index(chunks)

        cases = [
            _make_case("c1", "Respondible", answerable=True, evidence=[{"doc_id": "doc_test_1", "page_number": 1, "quotes": ["respondible"]}]),
            _make_case("c2", "¿Cuál será mi pensión?", answerable=False),
            _make_case("c3", "¿Qué dice el informe 2026?", answerable=False),
        ]
        dataset = _make_dataset(cases)
        client = MockOllamaClient(default_vector=v1)

        report = evaluate(index, dataset, top_k=3, client=client)

        self.assertEqual(report["n_answerable"], 1)
        self.assertEqual(report["n_unanswerable"], 2)

        # Métricas calculadas SOLO sobre el caso respondible (denominador = 1)
        metrics = report["metrics"]
        self.assertEqual(metrics["denominator"], 1)
        self.assertAlmostEqual(metrics["hit_at_1"], 1.0, places=6)
        self.assertAlmostEqual(metrics["hit_at_k"], 1.0, places=6)
        self.assertAlmostEqual(metrics["mrr_at_k"], 1.0, places=6)

        # Verificar estado y razón en los casos de abstención
        c_reports = report["cases"]
        self.assertEqual(c_reports[1]["status"], "not_evaluated_no_generator")
        self.assertIn("abstención", c_reports[1]["reason"].lower())
        self.assertIsNone(c_reports[1]["first_relevant_rank"])
        self.assertIsNone(c_reports[1]["hit_at_1"])
        self.assertIsNone(c_reports[1]["hit_at_k"])
        self.assertIsNone(c_reports[1]["reciprocal_rank"])
        self.assertEqual(len(c_reports[1]["hits"]), 1)  # se buscó

    def test_zero_answerable_cases_returns_null_and_denominator_zero(self):
        """Verifica que si no hay preguntas respondibles las métricas son null y denominador 0."""
        v1 = [1.0] + [0.0] * (DIMENSION - 1)
        chunks = [_make_chunk(1, "Texto", page_number=1, vec=v1)]
        index = _make_index(chunks)

        cases = [
            _make_case("c1", "¿Abstención 1?", answerable=False),
            _make_case("c2", "¿Abstención 2?", answerable=False),
        ]
        dataset = _make_dataset(cases)
        client = MockOllamaClient(default_vector=v1)

        report = evaluate(index, dataset, top_k=3, client=client)

        self.assertEqual(report["n_answerable"], 0)
        self.assertEqual(report["n_unanswerable"], 2)
        metrics = report["metrics"]
        self.assertEqual(metrics["denominator"], 0)
        self.assertIsNone(metrics["hit_at_1"])
        self.assertIsNone(metrics["hit_at_k"])
        self.assertIsNone(metrics["mrr_at_k"])

    def test_gold_unavailable_in_chunks_counts_as_failure(self):
        """Verifica que si la cita no cabe en ningún chunk completo, gold_available es False y cuenta como fallo."""
        v1 = [1.0] + [0.0] * (DIMENSION - 1)
        # La página 1 existe en el índice, pero su texto no contiene la cita
        chunks = [_make_chunk(1, "Texto que no tiene la frase buscada", page_number=1, vec=v1)]
        index = _make_index(chunks)

        cases = [
            _make_case(
                "c1",
                "Pregunta",
                answerable=True,
                evidence=[{"doc_id": "doc_test_1", "page_number": 1, "quotes": ["frase totalmente inexistente en el chunk"]}],
            )
        ]
        dataset = _make_dataset(cases)
        client = MockOllamaClient(default_vector=v1)

        report = evaluate(index, dataset, top_k=3, client=client)

        case_rep = report["cases"][0]
        self.assertFalse(case_rep["gold_available"])
        self.assertIsNone(case_rep["first_relevant_rank"])
        self.assertEqual(case_rep["hit_at_1"], 0)
        self.assertEqual(case_rep["hit_at_k"], 0)
        self.assertAlmostEqual(case_rep["reciprocal_rank"], 0.0, places=6)

        metrics = report["metrics"]
        self.assertEqual(metrics["denominator"], 1)
        self.assertAlmostEqual(metrics["hit_at_1"], 0.0, places=6)
        self.assertAlmostEqual(metrics["hit_at_k"], 0.0, places=6)
        self.assertAlmostEqual(metrics["mrr_at_k"], 0.0, places=6)

    def test_all_quotes_and_whitespace_normalization(self):
        """Verifica que un chunk solo es relevante si contiene TODAS las citas, con normalización de espacios."""
        v1 = [1.0] + [0.0] * (DIMENSION - 1)
        v2 = [0.0, 1.0] + [0.0] * (DIMENSION - 2)

        chunk1_text = (
            "La AIReF estima que   el gasto total aumentará del 12,9 al 16,1% del PIB "
            "entre 2023 y 2050 y se reducirá \n posteriormente hasta el 14,7% en 2070."
        )
        # chunk 2 solo contiene la primera cita pero no la segunda
        chunk2_text = "La AIReF estima que el gasto aumentará del 12,9 al 16,1% del PIB entre 2023 y 2050."

        chunks = [
            _make_chunk(1, chunk1_text, page_number=95, vec=v1),
            _make_chunk(2, chunk2_text, page_number=95, vec=v2),
        ]
        index = _make_index(chunks)

        cases = [
            _make_case(
                "c1",
                "Gasto en pensiones",
                answerable=True,
                evidence=[
                    {
                        "doc_id": "doc_test_1",
                        "page_number": 95,
                        "quotes": [
                            "aumentará del 12,9 al 16,1% del PIB entre 2023 y 2050",
                            "posteriormente hasta el 14,7% en 2070.",
                        ],
                    }
                ],
            )
        ]
        dataset = _make_dataset(cases)

        # Si el buscador devuelve primero chunk 2 (v2), este no debe considerarse relevante
        # pues le falta la segunda cita. El relevante es chunk 1 (rank 2).
        client = MockOllamaClient(default_vector=v2)

        report = evaluate(index, dataset, top_k=2, client=client)
        case_rep = report["cases"][0]
        self.assertTrue(case_rep["gold_available"])
        self.assertEqual(case_rep["first_relevant_rank"], 2)
        self.assertEqual(case_rep["hit_at_1"], 0)
        self.assertEqual(case_rep["hit_at_k"], 1)

    def test_multiple_evidences_alternative_or(self):
        """Verifica que múltiples evidencias en una pregunta actúan como alternativas OR."""
        v1 = [1.0] + [0.0] * (DIMENSION - 1)
        v_ortho = [0.0, 1.0] + [0.0] * (DIMENSION - 2)
        chunks = [
            _make_chunk(1, "Texto en página 95 sin la cita A", page_number=95, vec=v_ortho),
            _make_chunk(2, "Texto alternativo B en página 97", page_number=97, vec=v1),
        ]
        index = _make_index(chunks)

        cases = [
            _make_case(
                "c1",
                "Pregunta con alternativas",
                answerable=True,
                evidence=[
                    {"doc_id": "doc_test_1", "page_number": 95, "quotes": ["Texto A no existente"]},
                    {"doc_id": "doc_test_1", "page_number": 97, "quotes": ["Texto alternativo B"]},
                ],
            )
        ]
        dataset = _make_dataset(cases)
        client = MockOllamaClient(default_vector=v1)

        report = evaluate(index, dataset, top_k=2, client=client)
        case_rep = report["cases"][0]
        self.assertTrue(case_rep["gold_available"])
        self.assertEqual(case_rep["first_relevant_rank"], 1)
        self.assertEqual(case_rep["hit_at_1"], 1)


class TestDatasetValidationAndSafetyOffline(unittest.TestCase):
    """Pruebas de validación estricta antes de la red, pureza y manejo de errores."""

    def test_nonfinite_extra_rejected_before_search(self):
        idx = _make_index([_make_chunk(1, "Texto")])
        ds = _make_dataset([_make_case("q1", "Pregunta")])
        for value in [float("nan"), float("inf"), -float("inf")]:
            ds["extra"] = {"nested": [value]}
            with patch("rag_pensiones.evaluate.search") as lookup:
                with self.assertRaises(ValueError):
                    evaluate(idx, ds)
                lookup.assert_not_called()

    def test_load_dataset_rejects_overflow_number(self):
        ds = _make_dataset([_make_case("q1", "Pregunta")])
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as tmp:
            path = Path(tmp) / "overflow.json"
            path.write_text(json.dumps(ds)[:-1] + ',"extra":1e309}', encoding="utf-8")
            with self.assertRaises(ValueError):
                load_dataset(path)

    def test_validation_happens_before_network(self):
        """Verifica que ante errores en el dataset o índice jamás se contacta al cliente de red."""
        class GuardClient:
            def get_digest(self):
                raise AssertionError("get_digest called despite validation error!")

            def embed_texts(self, texts):
                raise AssertionError("embed_texts called despite validation error!")

        idx = _make_index([_make_chunk(1, "Texto")])
        bad_dataset = {"schema_version": "wrong_version"}

        with self.assertRaises(ValueError):
            evaluate(idx, bad_dataset, client=GuardClient())

    def test_gold_doc_or_page_not_in_index_raises_value_error(self):
        """Verifica que si la evidencia cita un doc_id o page_number inexistente en el índice se rechaza."""
        idx = _make_index([_make_chunk(1, "Texto", page_number=1, doc_id="doc_A")])

        # doc_id inexistente
        bad_case_doc = _make_case("c1", "Q", evidence=[{"doc_id": "doc_INEXISTENTE", "page_number": 1, "quotes": ["Texto"]}])
        with self.assertRaises(ValueError) as ctx:
            evaluate(idx, _make_dataset([bad_case_doc]), client=MockOllamaClient())
        self.assertIn("not present in the index", str(ctx.exception))

        # page_number inexistente
        bad_case_page = _make_case("c2", "Q", evidence=[{"doc_id": "doc_A", "page_number": 999, "quotes": ["Texto"]}])
        with self.assertRaises(ValueError) as ctx:
            evaluate(idx, _make_dataset([bad_case_page]), client=MockOllamaClient())
        self.assertIn("not present in the index", str(ctx.exception))

    def test_duplicate_case_ids_rejected(self):
        """Verifica rechazo de IDs duplicados en el dataset."""
        c1 = _make_case("q1", "Pregunta 1")
        c2 = _make_case("q1", "Pregunta 2 con id duplicado")
        with self.assertRaises(ValueError):
            validate_dataset(_make_dataset([c1, c2]))

    def test_answerable_boolean_type_and_evidence_constraints(self):
        """Verifica comprobaciones de tipo en answerable y coherencia con la lista de evidencias."""
        # answerable no booleano
        with self.assertRaises(ValueError):
            validate_dataset(_make_dataset([_make_case("q1", "Q", answerable=1)]))  # type: ignore

        with self.assertRaises(ValueError):
            validate_dataset(_make_dataset([_make_case("q1", "Q", answerable="true")]))  # type: ignore

        # answerable=True con evidence=[]
        with self.assertRaises(ValueError):
            validate_dataset(_make_dataset([_make_case("q1", "Q", answerable=True, evidence=[])]))

        # answerable=False con evidence no vacía
        with self.assertRaises(ValueError):
            validate_dataset(_make_dataset([_make_case("q1", "Q", answerable=False, evidence=[{"doc_id": "d", "page_number": 1, "quotes": ["q"]}])]))

    def test_quotes_and_page_number_types(self):
        """Verifica rechazo de quotes vacías y page_number no enteros positivos."""
        # quotes vacías
        with self.assertRaises(ValueError):
            validate_dataset(_make_dataset([_make_case("q1", "Q", evidence=[{"doc_id": "d", "page_number": 1, "quotes": []}])]))

        # quote con solo espacios
        with self.assertRaises(ValueError):
            validate_dataset(_make_dataset([_make_case("q1", "Q", evidence=[{"doc_id": "d", "page_number": 1, "quotes": ["   "]}])]))

        # page_number booleano o <= 0
        with self.assertRaises(ValueError):
            validate_dataset(_make_dataset([_make_case("q1", "Q", evidence=[{"doc_id": "d", "page_number": True, "quotes": ["q"]}])]))
        with self.assertRaises(ValueError):
            validate_dataset(_make_dataset([_make_case("q1", "Q", evidence=[{"doc_id": "d", "page_number": 0, "quotes": ["q"]}])]))

    def test_top_k_invalid_rejected(self):
        """Verifica rechazo de top_k no enteros positivos o booleanos."""
        idx = _make_index([_make_chunk(1, "T")])
        ds = _make_dataset([_make_case("q1", "Q")])
        client = MockOllamaClient()

        for bad_k in [0, -1, True, False, "3", 2.5]:
            with self.assertRaises(ValueError):
                evaluate(idx, ds, top_k=bad_k, client=client)  # type: ignore

    def test_index_and_dataset_immutability(self):
        """Verifica que evaluate no muta en absoluto el índice ni el dataset."""
        v1 = [1.0] + [0.0] * (DIMENSION - 1)
        idx = _make_index([_make_chunk(1, "Texto", page_number=1, vec=v1)])
        ds = _make_dataset([_make_case("q1", "Pregunta")])

        idx_copy = copy.deepcopy(idx)
        ds_copy = copy.deepcopy(ds)

        evaluate(idx, ds, top_k=1, client=MockOllamaClient())

        self.assertEqual(idx, idx_copy)
        self.assertEqual(ds, ds_copy)

    def test_search_failure_aborts_without_partial_results(self):
        """Verifica que un fallo de red o del buscador aborta toda la evaluación."""
        class FailClient:
            def __init__(self):
                self.calls = 0

            def get_digest(self):
                return "fake_test_digest_sha256"

            def embed_texts(self, texts):
                self.calls += 1
                if self.calls == 2:
                    raise ConnectionError("Fallo de red en consulta 2")
                return [[1.0] + [0.0] * (DIMENSION - 1)]

        idx = _make_index([_make_chunk(1, "Texto")])
        ds = _make_dataset([_make_case("q1", "Pregunta 1"), _make_case("q2", "Pregunta 2")])

        with self.assertRaises(ConnectionError):
            evaluate(idx, ds, top_k=1, client=FailClient())


class TestCLIOffline(unittest.TestCase):
    """Pruebas de la CLI en modo texto, JSON y validación de errores."""

    def test_cli_text_output_spanish_and_disclaimer(self):
        """Verifica que la salida estándar en texto incluya métricas, tabla, abstención y advertencia."""
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as tmpdir:
            tmppath = Path(tmpdir)
            idx_file = tmppath / "test_idx.json"
            ds_file = tmppath / "test_ds.json"

            v1 = [1.0] + [0.0] * (DIMENSION - 1)
            idx_data = _make_index([_make_chunk(1, "Texto sobre pensiones", page_number=95, vec=v1)])
            ds_data = _make_dataset([
                _make_case("q1", "¿Cuánto aumenta?", answerable=True, evidence=[{"doc_id": "doc_test_1", "page_number": 95, "quotes": ["pensiones"]}]),
                _make_case("q6", "¿Mi pensión individual?", answerable=False),
            ])

            idx_file.write_text(json.dumps(idx_data, ensure_ascii=False), encoding="utf-8")
            ds_file.write_text(json.dumps(ds_data, ensure_ascii=False), encoding="utf-8")

            fake_client = MockOllamaClient(default_vector=v1)
            with patch("rag_pensiones.evaluate.OllamaEmbedder", return_value=fake_client):
                out_buf = io.StringIO()
                err_buf = io.StringIO()
                with patch("sys.stdout", out_buf), patch("sys.stderr", err_buf):
                    code = main(["--dataset", str(ds_file), "--index", str(idx_file), "--top-k", "1"])

                self.assertEqual(code, 0)
                self.assertEqual(err_buf.getvalue(), "")
                out = out_buf.getvalue()
                self.assertIn("EVALUACIÓN DE RECUPERACIÓN SEMÁNTICA", out)
                self.assertIn("q1", out)
                self.assertIn("Hit@1", out)
                self.assertIn("MRR@1", out)
                self.assertIn("PREGUNTAS DE ABSTENCIÓN", out)
                self.assertIn("not_evaluated_no_generator", out)
                self.assertIn("ADVERTENCIA METODOLÓGICA", out)
                self.assertIn("no constituye un test reservado", out)

    def test_cli_json_output(self):
        """Verifica salida JSON completa con --json."""
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as tmpdir:
            tmppath = Path(tmpdir)
            idx_file = tmppath / "test_idx.json"
            ds_file = tmppath / "test_ds.json"

            v1 = [1.0] + [0.0] * (DIMENSION - 1)
            idx_data = _make_index([_make_chunk(1, "Texto", page_number=1, vec=v1)])
            ds_data = _make_dataset([_make_case("q1", "Pregunta")])

            idx_file.write_text(json.dumps(idx_data, ensure_ascii=False), encoding="utf-8")
            ds_file.write_text(json.dumps(ds_data, ensure_ascii=False), encoding="utf-8")

            fake_client = MockOllamaClient(default_vector=v1)
            with patch("rag_pensiones.evaluate.OllamaEmbedder", return_value=fake_client):
                out_buf = io.StringIO()
                err_buf = io.StringIO()
                with patch("sys.stdout", out_buf), patch("sys.stderr", err_buf):
                    code = main(["--dataset", str(ds_file), "--index", str(idx_file), "--top-k", "2", "--json"])

                self.assertEqual(code, 0)
                self.assertEqual(err_buf.getvalue(), "")
                parsed = json.loads(out_buf.getvalue())
                self.assertEqual(parsed["schema_version"], EVALUATION_SCHEMA_VERSION)
                self.assertEqual(parsed["n_answerable"], 1)
                self.assertEqual(parsed["metrics"]["denominator"], 1)
                self.assertEqual(len(parsed["cases"]), 1)
                self.assertIn("dataset_sha256", parsed)

    def test_cli_errors_to_stderr_and_non_zero_exit_no_traceback(self):
        """Verifica que errores de entrada se imprimen a stderr sin traceback."""
        out_buf = io.StringIO()
        err_buf = io.StringIO()
        with patch("sys.stdout", out_buf), patch("sys.stderr", err_buf):
            code = main(["--top-k", "0"])
        self.assertNotEqual(code, 0)
        self.assertIn("Error:", err_buf.getvalue())
        self.assertNotIn("Traceback", err_buf.getvalue())


class TestRealDatasetAndRealIndexOffline(unittest.TestCase):
    """Pruebas sobre los archivos reales del proyecto: evaluation/pensiones_dev.json y embeddings index."""

    def test_load_real_pensiones_dev_json(self):
        """Verifica la carga del dataset real y la presencia de las 5 preguntas respondibles y 2 de abstención."""
        data = load_dataset(DEFAULT_DATASET_PATH)
        self.assertEqual(data["schema_version"], DATASET_SCHEMA_VERSION)
        cases = data["cases"]
        self.assertEqual(len(cases), 7)

        answerable_cases = [c for c in cases if c["answerable"]]
        unanswerable_cases = [c for c in cases if not c["answerable"]]

        self.assertEqual(len(answerable_cases), 5)
        self.assertEqual(len(unanswerable_cases), 2)

        # Verificar citas exactas de las preguntas respondibles
        q1 = answerable_cases[0]
        self.assertEqual(q1["evidence"][0]["page_number"], 95)
        self.assertIn("aumentará del 12,9 a l 16,1% del PIB entre 2023 y 2050 y se reducirá", q1["evidence"][0]["quotes"])
        self.assertIn("posteriormente hasta el 14,7% en 2070.", q1["evidence"][0]["quotes"])

        q2 = answerable_cases[1]
        self.assertEqual(q2["evidence"][0]["page_number"], 95)
        self.assertIn("pensiones del sistema de Seguridad Social, Clases Pasivas del Estado y pensiones no contributivas.", q2["evidence"][0]["quotes"])

        q3 = answerable_cases[2]
        self.assertEqual(q3["evidence"][0]["page_number"], 96)
        self.assertIn("A partir de 2028 se supone que el peso sobre PIB de las pensiones no contributivas se mantiene constante en el 0,2%.", q3["evidence"][0]["quotes"])

        q4 = answerable_cases[3]
        self.assertEqual(q4["evidence"][0]["page_number"], 97)
        self.assertIn("alcanzando un máximo de 17,3 millones de pensiones en 2055.", q4["evidence"][0]["quotes"])

        q5 = answerable_cases[4]
        self.assertEqual(q5["evidence"][0]["page_number"], 97)
        self.assertIn("Un pensionista puede ser beneficiario de más de una pensión.", q5["evidence"][0]["quotes"])
        self.assertIn("La ratio pensionistas / pensiones se mantiene estable alrededor del 0,9", q5["evidence"][0]["quotes"])

    def test_evaluate_real_dataset_against_real_embeddings_offline(self):
        """Verifica la compatibilidad del dataset real con el índice real y la disponibilidad de gold."""
        dataset = load_dataset(DEFAULT_DATASET_PATH)
        index = load_index(LEGACY_INDEX_PATH)

        fake_client = MockOllamaClient(digest=index["model_digest"])
        report = evaluate(index, dataset, top_k=3, client=fake_client)

        self.assertEqual(report["corpus_size"], 13)
        self.assertEqual(report["n_answerable"], 5)
        self.assertEqual(report["n_unanswerable"], 2)

        # En todas las 5 preguntas respondibles, el gold debe estar disponible en los 13 fragmentos indexados
        for c in report["cases"]:
            if c["answerable"]:
                self.assertTrue(c["gold_available"], f"Gold no disponible para {c['id']}")
            else:
                self.assertEqual(c["status"], "not_evaluated_no_generator")


if __name__ == "__main__":
    unittest.main()
