"""Tests exhaustivos y deterministas offline para la cobertura de consultas compuestas."""

import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from rag_pensiones.answer import (
    COVERAGE_METHOD,
    DEFAULT_GENERATION_MODEL,
    MAX_MESSAGES_BYTES,
    SCHEMA_VERSION,
    SCHEMA_VERSION_V2,
    answer,
    build_messages,
    format_text_answer,
    main,
    split_question,
)
from rag_pensiones.chunking import ChunkRecord
from rag_pensiones.embeddings import DIMENSION
from rag_pensiones.ingest import PROJECT_ROOT
from rag_pensiones.search import SearchHit


def _make_chunk_record(
    idx: int,
    text: str | None = None,
    doc_id: str = "doc_test_1",
    source_file: str = "test_doc.pdf",
    published_at: str = "2025-03-31",
    page_number: int = 95,
) -> ChunkRecord:
    if text is None:
        text = f"Texto de prueba para el fragmento {idx} con información oficial sobre pensiones."
    return ChunkRecord(
        chunk_id=f"chunk_id_{idx}",
        doc_id=doc_id,
        source_file=source_file,
        published_at=published_at,
        page_number=page_number,
        start_char=0,
        end_char=len(text),
        text=text,
        strategy="page-local-v1:max_chars=1000:overlap_chars=150",
    )


def _make_search_hit(idx: int, text: str | None = None, score: float = 0.95) -> SearchHit:
    return SearchHit(
        rank=idx,
        score=score,
        chunk=_make_chunk_record(idx=idx, text=text),
    )


class MockEmbedder:
    """Embedder simulado para pruebas de respuesta sin contactar con Ollama."""

    def __init__(self, digest: str = "test_embed_digest_sha256") -> None:
        self.digest = digest

    def get_digest(self) -> str:
        return self.digest

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[1.0] + [0.0] * (DIMENSION - 1)]


def _make_two_chunk_index() -> dict:
    """Crea un índice válido con dos fragmentos diferenciados para pruebas A/B."""
    chunks = [
        {
            "chunk_id": "chunk_id_1",
            "doc_id": "doc_test_1",
            "source_file": "test_doc.pdf",
            "published_at": "2025-03-31",
            "page_number": 95,
            "start_char": 0,
            "end_char": 108,
            "text": "aumentará del 12,9 a l 16,1% del PIB entre 2023 y 2050 y se reducirá posteriormente hasta el 14,7% en 2070.",
            "strategy": "page-local-v1:max_chars=1000:overlap_chars=150",
            "embedding": [1.0] + [0.0] * (DIMENSION - 1),
        },
        {
            "chunk_id": "chunk_id_2",
            "doc_id": "doc_test_1",
            "source_file": "test_doc.pdf",
            "published_at": "2025-03-31",
            "page_number": 96,
            "start_char": 0,
            "end_char": 120,
            "text": "La ratio de pensiones por pensionista se proyecta en 1,2 según el escenario base de la AIReF para el periodo analizado.",
            "strategy": "page-local-v1:max_chars=1000:overlap_chars=150",
            "embedding": [0.0, 1.0] + [0.0] * (DIMENSION - 2),
        },
    ]
    return {
        "schema_version": "embeddings-v1",
        "model": "embeddinggemma:300m",
        "model_digest": "test_embed_digest_sha256",
        "dimension": DIMENSION,
        "document_prefix": "title: none | text: ",
        "query_prefix": "task: search result | query: ",
        "input_sha256": "4cac2ee74838d6c5caa37efa0f27336689cd1ff12b6cbf9aa1948e1a36314cdc",
        "chunks": chunks,
    }


def _make_mock_response(
    status: str = "answered",
    claims: list[dict] | None = None,
    reason: str = "",
    model: str = DEFAULT_GENERATION_MODEL,
    stats: dict | None = None,
) -> dict:
    """Construye un payload crudo simulado de Ollama /api/chat conforme."""
    if claims is None:
        if status == "answered":
            claims = [
                {
                    "text": "El gasto aumentará hasta el 16,1% del PIB en 2050.",
                    "evidence": [
                        {
                            "source_id": "S1",
                            "quote": "aumentará del 12,9 a l 16,1% del PIB entre 2023 y 2050",
                        }
                    ],
                }
            ]
        else:
            claims = []

    content = json.dumps(
        {
            "status": status,
            "claims": claims,
            "reason": reason,
        },
        ensure_ascii=False,
    )
    stat_defaults = {
        "prompt_eval_count": 100,
        "eval_count": 40,
        "eval_duration": 400000000,
        "total_duration": 800000000,
    }
    if stats:
        stat_defaults.update(stats)
    return {
        "model": model,
        "done": True,
        "done_reason": "stop",
        "message": {
            "role": "assistant",
            "content": content,
        },
        **stat_defaults,
    }


class CompoundMockGenerator:
    """Generador simulado para pruebas compuestas con respuestas secuenciales e inspección."""

    def __init__(
        self,
        responses: list[dict | Exception] | None = None,
        model: str = DEFAULT_GENERATION_MODEL,
        digest: str = "test_gen_digest_sha256",
        digest_per_call: list[str] | None = None,
    ) -> None:
        self.model = model
        self.digest = digest
        self.digest_per_call = digest_per_call or []
        self.responses = responses or []
        self.calls: list[str] = []
        self.generate_calls: list[list[dict]] = []
        self._gen_count = 0
        self._digest_count = 0

    def get_digest(self) -> str:
        self.calls.append("get_digest")
        if self._digest_count < len(self.digest_per_call):
            d = self.digest_per_call[self._digest_count]
            self._digest_count += 1
            return d
        return self.digest

    def generate(self, messages: list[dict]) -> dict:
        self.calls.append("generate")
        self.generate_calls.append(messages)
        if self._gen_count < len(self.responses):
            resp = self.responses[self._gen_count]
            self._gen_count += 1
            if isinstance(resp, Exception):
                raise resp
            return resp
        raise RuntimeError("No more mocked responses available in CompoundMockGenerator")


class TestSplitQuestion(unittest.TestCase):
    def test_initial_conjunction_is_not_a_separator(self):
        for query in ("¿Y cuál es el gasto proyectado?", "Y qué ocurre con el gasto?"):
            with self.subTest(query=query):
                self.assertEqual(split_question(query), [query])

    def test_initial_conjunction_preserved_before_real_split(self):
        self.assertEqual(
            split_question("¿Y cuál es el gasto y cómo evoluciona?"),
            ["¿Y cuál es el gasto?", "¿cómo evoluciona?"],
        )

    """Pruebas exhaustivas de división de consultas compuestas mediante split_question."""

    def test_split_interrogatives_all_explicit(self):
        """Verifica la división por cada uno de los interrogativos explícitos requeridos."""
        interrogatives = [
            ("qué", "¿Qué gasto se prevé?"),
            ("cuál", "¿Cuál es la proyección?"),
            ("cuáles", "¿Cuáles son los factores?"),
            ("cuándo", "¿Cuándo alcanzará el máximo?"),
            ("cuánto", "¿Cuánto aumentará el gasto?"),
            ("cuánta", "¿Cuánta cuantía se destinará?"),
            ("cuántos", "¿Cuántos años abarcará?"),
            ("cuántas", "¿Cuántas pensiones habrá?"),
            ("cómo", "¿Cómo evolucionará la tasa?"),
            ("dónde", "¿Dónde se publican los datos?"),
            ("quién", "¿Quién elabora el informe?"),
            ("quiénes", "¿Quiénes son los beneficiarios?"),
            ("por qué", "¿Por qué se reduce en 2070?"),
        ]

        for word, subq in interrogatives:
            with self.subTest(interrogative=word):
                q = f"¿Cuál es el escenario base y {word} dice el informe?"
                parts = split_question(q)
                self.assertEqual(len(parts), 2)
                self.assertEqual(parts[0], "¿Cuál es el escenario base?")
                self.assertEqual(parts[1], f"¿{word} dice el informe?")

    def test_split_case_insensitivity_and_tildes(self):
        """Verifica que mayúsculas, minúsculas y tildes en los interrogativos se gestionan correctamente."""
        # Mayúsculas completas
        q1 = "¿CUÁL ES EL GASTO Y CUÁL ES LA RATIO?"
        p1 = split_question(q1)
        self.assertEqual(p1, ["¿CUÁL ES EL GASTO?", "¿CUÁL ES LA RATIO?"])

        # Primera letra mayúscula
        q2 = "¿Qué porcentaje representa y Cuánto costará?"
        p2 = split_question(q2)
        self.assertEqual(p2, ["¿Qué porcentaje representa?", "¿Cuánto costará?"])

        # Mayúsculas en DÓNDE y CÓMO
        q3 = "¿Quién autoriza el presupuesto Y DÓNDE se registra?"
        p3 = split_question(q3)
        self.assertEqual(p3, ["¿Quién autoriza el presupuesto?", "¿DÓNDE se registra?"])

        # Por qué con mayúsculas y espacios múltiples
        q4 = "¿Cuándo termina el periodo Y  POR QUÉ cambia la tendencia?"
        p4 = split_question(q4)
        self.assertEqual(p4, ["¿Cuándo termina el periodo?", "¿POR QUÉ cambia la tendencia?"])

    def test_split_question_mark_boundary_stripping_and_wrapping(self):
        """Verifica que signos ¿? se quitan de los límites de cada parte y se envuelven en ¿...?, preservando entidades."""
        # Signos en ambas partes
        q1 = "¿Cuál es la cuantía máxima? y ¿cuál es la mínima?"
        p1 = split_question(q1)
        self.assertEqual(p1, ["¿Cuál es la cuantía máxima?", "¿cuál es la mínima?"])

        # Sin signos originales
        q2 = "cuál es el régimen de la Seguridad Social y cuál es el de Clases Pasivas"
        p2 = split_question(q2)
        self.assertEqual(
            p2,
            [
                "¿cuál es el régimen de la Seguridad Social?",
                "¿cuál es el de Clases Pasivas?",
            ],
        )
        # Verificar preservación íntegra de entidades con mayúsculas y tildes
        self.assertIn("Seguridad Social", p2[0])
        self.assertIn("Clases Pasivas", p2[1])

        # Signos redundantes en bordes
        q3 = "??¿¿Cuál es la tasa?? y ??¿¿cuál es el total??"
        p3 = split_question(q3)
        self.assertEqual(p3, ["¿Cuál es la tasa?", "¿cuál es el total?"])

    def test_no_split_single_question_exact(self):
        """Verifica que consultas simples sin división devuelven exactamente [query.strip()]."""
        q1 = "¿Cuál es el gasto en pensiones en 2050?"
        self.assertEqual(split_question(q1), [q1])

        q2 = "   consulta sin signos ni conjunción   "
        self.assertEqual(split_question(q2), ["consulta sin signos ni conjunción"])

    def test_no_split_lists_and_years(self):
        """Verifica que conjunciones 'y' en listas o años no provocan división indebida."""
        # Años consecutivos con y
        q1 = "¿Qué gasto total proyecta el escenario base para 2050 y 2070?"
        self.assertEqual(split_question(q1), [q1])

        # Años pegados como y2070
        q2 = "previsión para 2050 y2070 en el informe"
        self.assertEqual(split_question(q2), [q2])

        # Entidades unidas por y
        q3 = "¿Cuál es la cobertura de la Seguridad Social y Clases Pasivas?"
        self.assertEqual(split_question(q3), [q3])

        # Combinación: año con y más interrogativo compuesto
        q4 = "¿Qué gasto se proyecta para 2050 y 2070 y cuál es la ratio de pensionistas?"
        p4 = split_question(q4)
        self.assertEqual(len(p4), 2)
        self.assertEqual(p4[0], "¿Qué gasto se proyecta para 2050 y 2070?")
        self.assertEqual(p4[1], "¿cuál es la ratio de pensionistas?")
        self.assertIn("2050 y 2070", p4[0])

    def test_other_domains_not_q5_patch(self):
        """Verifica que la regla opera de forma genérica sobre dominios ajenos a pensiones."""
        q1 = "¿Cuál es la capital de Francia y cuántos habitantes tiene París?"
        p1 = split_question(q1)
        self.assertEqual(
            p1,
            ["¿Cuál es la capital de Francia?", "¿cuántos habitantes tiene París?"],
        )

        q2 = "¿Quién descubrió la penicilina y en qué año se publicó el hallazgo?"
        p2 = split_question(q2)
        self.assertEqual(len(p2), 1)

        q3 = "¿Quién descubrió la penicilina y qué aplicaciones médicas tuvo?"
        p3 = split_question(q3)
        self.assertEqual(
            p3,
            ["¿Quién descubrió la penicilina?", "¿qué aplicaciones médicas tuvo?"],
        )

    def test_invalid_inputs_and_bool(self):
        """Verifica rechazo estricto de entradas vacías, tipos incorrectos y booleanos."""
        with self.assertRaises(ValueError):
            split_question("")
        with self.assertRaises(ValueError):
            split_question("   \n\t  ")
        with self.assertRaises(ValueError):
            split_question(None)  # type: ignore
        with self.assertRaises(ValueError):
            split_question(123)  # type: ignore
        with self.assertRaises(ValueError):
            split_question(True)  # type: ignore
        with self.assertRaises(ValueError):
            split_question(False)  # type: ignore

    def test_max_parts_limit(self):
        """Verifica soporte hasta 4 partes y error explícito si se superan las 4 partes."""
        # 3 partes
        q3 = "¿Cuál es A y cuál es B y cuál es C?"
        self.assertEqual(len(split_question(q3)), 3)

        # 4 partes (máximo admitido)
        q4 = "¿Cuál es A y cuál es B y cuál es C y cuál es D?"
        p4 = split_question(q4)
        self.assertEqual(len(p4), 4)

        # 5 partes (excede límite de 4)
        q5 = "¿Cuál es A y cuál es B y cuál es C y cuál es D y cuál es E?"
        with self.assertRaises(ValueError) as ctx:
            split_question(q5)
        self.assertIn("exceeds maximum allowed limit of 4 parts", str(ctx.exception))


class TestCompoundPipelineAndExecution(unittest.TestCase):
    """Pruebas del ciclo RAG compuesto: 1 búsqueda, N llamadas al generador, agregación y pureza."""

    def setUp(self):
        self.index = _make_two_chunk_index()

    def test_single_search_and_n_generator_calls_with_same_sources(self):
        """Verifica que se ejecuta UNA sola búsqueda vectorial y N llamadas a Ollama con las mismas fuentes."""
        embedder = MockEmbedder(digest="test_embed_digest_sha256")
        embed_spy = MagicMock(wraps=embedder.embed_texts)
        embedder.embed_texts = embed_spy

        resp1 = _make_mock_response()
        resp2 = _make_mock_response(
            claims=[
                {
                    "text": "La ratio es 1,2 según la AIReF.",
                    "evidence": [
                        {
                            "source_id": "S2",
                            "quote": "La ratio de pensiones por pensionista se proyecta en 1,2 según el escenario base de la AIReF para el periodo analizado.",
                        }
                    ],
                }
            ]
        )
        generator = CompoundMockGenerator(responses=[resp1, resp2])

        query = "¿Qué gasto se proyecta en 2050 y cuál es la ratio de pensiones?"
        report = answer(
            index=self.index,
            query=query,
            top_k=2,
            embedder=embedder,
            generator=generator,
        )

        # 1. Búsqueda semántica ejecutada UNA SOLA VEZ con la consulta original
        self.assertEqual(embed_spy.call_count, 1)
        self.assertIn(query, embed_spy.call_args[0][0][0])

        # 2. Generador contactado N veces (2 subpreguntas)
        self.assertEqual(len(generator.generate_calls), 2)

        # 3. Mismos IDs y fuentes en las llamadas al generador
        call1_user = json.loads(generator.generate_calls[0][1]["content"])
        call2_user = json.loads(generator.generate_calls[1][1]["content"])

        self.assertEqual(call1_user["pregunta"], "¿Qué gasto se proyecta en 2050?")
        self.assertEqual(call1_user["pregunta_original"], query)
        self.assertEqual(call2_user["pregunta"], "¿cuál es la ratio de pensiones?")
        self.assertEqual(call2_user["pregunta_original"], query)

        self.assertEqual(call1_user["fuentes"], call2_user["fuentes"])
        self.assertEqual(len(call1_user["fuentes"]), 2)
        self.assertEqual(call1_user["fuentes"][0]["source_id"], "S1")
        self.assertEqual(call1_user["fuentes"][1]["source_id"], "S2")

    def test_synthetic_distinct_ab_answers_without_omission(self):
        """Caso de regresión genérico: respuestas A y B se agregan sin omisión ni pérdida de datos."""
        embedder = MockEmbedder()
        resp_a = _make_mock_response(
            claims=[
                {
                    "text": "Gasto en 2050 alcanza el 16,1% del PIB.",
                    "evidence": [
                        {
                            "source_id": "S1",
                            "quote": "aumentará del 12,9 a l 16,1% del PIB entre 2023 y 2050",
                        }
                    ],
                }
            ]
        )
        resp_b = _make_mock_response(
            claims=[
                {
                    "text": "Ratio proyectada en 1,2 pensiones por pensionista.",
                    "evidence": [
                        {
                            "source_id": "S2",
                            "quote": "La ratio de pensiones por pensionista se proyecta en 1,2 según el escenario base de la AIReF para el periodo analizado.",
                        }
                    ],
                }
            ]
        )
        generator = CompoundMockGenerator(responses=[resp_a, resp_b])

        report = answer(
            index=self.index,
            query="¿Cuál es el gasto en 2050 y cuál es la ratio por pensionista?",
            top_k=2,
            embedder=embedder,
            generator=generator,
        )

        self.assertEqual(report["schema_version"], SCHEMA_VERSION_V2)
        self.assertEqual(report["status"], "answered")
        self.assertEqual(report["reason"], "")
        self.assertEqual(len(report["claims"]), 2)
        self.assertEqual(report["claims"][0]["text"], "Gasto en 2050 alcanza el 16,1% del PIB.")
        self.assertEqual(report["claims"][1]["text"], "Ratio proyectada en 1,2 pensiones por pensionista.")

        # Verificar coverage estructurado
        coverage = report["coverage"]
        self.assertEqual(coverage["method"], COVERAGE_METHOD)
        self.assertTrue(coverage["all_parts_reported_answered"])
        self.assertEqual(len(coverage["parts"]), 2)
        self.assertEqual(coverage["parts"][0]["question"], "¿Cuál es el gasto en 2050?")
        self.assertEqual(coverage["parts"][0]["status"], "answered")
        self.assertEqual(coverage["parts"][1]["question"], "¿cuál es la ratio por pensionista?")
        self.assertEqual(coverage["parts"][1]["status"], "answered")

    def test_answered_plus_insufficient_yields_global_insufficient_with_visible_partial(self):
        """Verifica que una parte insufficient convierte el estado global a insufficient sin falsear answered."""
        embedder = MockEmbedder()
        resp_a = _make_mock_response(
            status="answered",
            claims=[
                {
                    "text": "Gasto proyectado 16,1% del PIB.",
                    "evidence": [
                        {
                            "source_id": "S1",
                            "quote": "aumentará del 12,9 a l 16,1% del PIB entre 2023 y 2050",
                        }
                    ],
                }
            ],
        )
        resp_b = _make_mock_response(
            status="insufficient_evidence",
            claims=[],
            reason="El informe de la AIReF no contiene cálculos individuales ni estimaciones para 2045.",
        )
        generator = CompoundMockGenerator(responses=[resp_a, resp_b])

        report = answer(
            index=self.index,
            query="¿Cuál es el gasto en 2050 y cuánto cobraré en 2045?",
            top_k=2,
            embedder=embedder,
            generator=generator,
        )

        # Estado global DEBE ser insufficient_evidence
        self.assertEqual(report["schema_version"], SCHEMA_VERSION_V2)
        self.assertEqual(report["status"], "insufficient_evidence")
        self.assertEqual(report["claims"], [])
        self.assertIn("Evidencia insuficiente en 1 de 2 partes", report["reason"])
        self.assertIn("cálculos individuales", report["reason"])

        # Las respuestas parciales válidas DEBEN ser visibles en coverage.parts
        coverage = report["coverage"]
        self.assertFalse(coverage["all_parts_reported_answered"])
        self.assertEqual(len(coverage["parts"]), 2)

        # Parte 1 respondida se conserva intacta
        self.assertEqual(coverage["parts"][0]["status"], "answered")
        self.assertEqual(len(coverage["parts"][0]["claims"]), 1)
        self.assertEqual(coverage["parts"][0]["claims"][0]["text"], "Gasto proyectado 16,1% del PIB.")

        # Parte 2 insuficiente conserva su motivo
        self.assertEqual(coverage["parts"][1]["status"], "insufficient_evidence")
        self.assertEqual(coverage["parts"][1]["claims"], [])
        self.assertIn("cálculos individuales", coverage["parts"][1]["reason"])

    def test_technical_error_on_second_part_aborts_without_output(self):
        """Verifica que un error técnico en la segunda llamada aborta inmediatamente sin emitir reporte."""
        embedder = MockEmbedder()
        resp_a = _make_mock_response()
        resp_b_error = RuntimeError("Fallo de red en Ollama durante la generación")
        generator = CompoundMockGenerator(responses=[resp_a, resp_b_error])

        with self.assertRaises(RuntimeError) as ctx:
            answer(
                index=self.index,
                query="¿Cuál es el gasto en 2050 y cuál es la ratio?",
                top_k=2,
                embedder=embedder,
                generator=generator,
            )
        self.assertIn("Fallo de red en Ollama", str(ctx.exception))

    def test_generator_digest_change_between_parts_aborts(self):
        """Verifica aborto si el digest del modelo cambia entre llamadas de subpreguntas."""
        embedder = MockEmbedder()
        resp_a = _make_mock_response()
        resp_b = _make_mock_response()
        # Simula cambio de digest tras la primera llamada
        generator = CompoundMockGenerator(
            responses=[resp_a, resp_b],
            digest_per_call=[
                "digest_base",  # initial_digest
                "digest_base",  # part 1 before
                "digest_base",  # part 1 after
                "digest_MUTATED",  # part 2 before (cambió!)
            ],
        )

        with self.assertRaises(RuntimeError) as ctx:
            answer(
                index=self.index,
                query="¿Cuál es el gasto en 2050 y cuál es la ratio?",
                top_k=2,
                embedder=embedder,
                generator=generator,
            )
        self.assertIn("digest changed before generation", str(ctx.exception))

    def test_second_part_budget_overflow_fails_before_generation(self):
        """Verifica que si una parte excede el presupuesto UTF-8 de 6500 B, falla antes de llamar al generador."""
        embedder = MockEmbedder()
        huge_subquestion = "consulta con texto muy extenso. " * 300  # Hace que messages > 6500 B
        generator = CompoundMockGenerator()

        # Query compuesta cuya segunda parte es gigantesca
        compound_query = f"¿Cuál es el gasto en 2050? y ¿cuál es el detalle de {huge_subquestion}?"

        with self.assertRaises(ValueError) as ctx:
            answer(
                index=self.index,
                query=compound_query,
                top_k=2,
                embedder=embedder,
                generator=generator,
            )
        self.assertIn("exceeds conservative budget", str(ctx.exception))
        # El generador NUNCA debe haber sido contactado
        self.assertEqual(len(generator.generate_calls), 0)

    def test_stats_sum_hashes_order_and_parameters(self):
        """Verifica suma aritmética exacta de stats, hash canónico compuesto y parámetros."""
        embedder = MockEmbedder()
        resp_a = _make_mock_response(
            stats={
                "prompt_eval_count": 100,
                "eval_count": 40,
                "eval_duration": 400000000,
                "total_duration": 800000000,
                "load_duration": 10000000,
            }
        )
        resp_b = _make_mock_response(
            claims=[
                {
                    "text": "Ratio 1,2.",
                    "evidence": [
                        {
                            "source_id": "S2",
                            "quote": "La ratio de pensiones por pensionista se proyecta en 1,2 según el escenario base de la AIReF para el periodo analizado.",
                        }
                    ],
                }
            ],
            stats={
                "prompt_eval_count": 150,
                "eval_count": 50,
                "eval_duration": 500000000,
                "total_duration": 900000000,
                "load_duration": 20000000,
            },
        )
        generator = CompoundMockGenerator(responses=[resp_a, resp_b])

        report = answer(
            index=self.index,
            query="¿Cuál es el gasto en 2050 y cuál es la ratio?",
            top_k=2,
            embedder=embedder,
            generator=generator,
        )

        # Suma exacta de estadísticas
        self.assertEqual(report["stats"]["prompt_eval_count"], 250)
        self.assertEqual(report["stats"]["eval_count"], 90)
        self.assertEqual(report["stats"]["eval_duration"], 900000000)
        self.assertEqual(report["stats"]["total_duration"], 1700000000)
        self.assertEqual(report["stats"]["load_duration"], 30000000)

        # generation_calls registrado en parameters
        self.assertEqual(report["parameters"]["generation_calls"], 2)

        # Hash canónico global
        p1_sha = report["coverage"]["parts"][0]["prompt_sha256"]
        p2_sha = report["coverage"]["parts"][1]["prompt_sha256"]
        expected_global_hash = hashlib.sha256(
            json.dumps([p1_sha, p2_sha], separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        self.assertEqual(report["prompt_sha256"], expected_global_hash)

    def test_single_question_preserves_v1_identity(self):
        """Verifica que consultas simples sin división devuelven schema v1 y estructura idéntica a v1."""
        embedder = MockEmbedder()
        resp = _make_mock_response()
        generator = CompoundMockGenerator(responses=[resp])

        report = answer(
            index=self.index,
            query="¿Cuál es el gasto proyectado en 2050?",
            top_k=2,
            embedder=embedder,
            generator=generator,
        )

        self.assertEqual(report["schema_version"], SCHEMA_VERSION)
        self.assertNotIn("coverage", report)
        self.assertNotIn("generation_calls", report["parameters"])
        self.assertEqual(len(generator.generate_calls), 1)


class TestCoverageCLIAndOutput(unittest.TestCase):
    """Pruebas de la interfaz CLI en modo texto y JSON para consultas compuestas."""

    def test_cli_text_mode_compound_answered(self):
        """Verifica salida legible en texto para consulta compuesta respondida con desglose."""
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as tmpdir:
            idx_file = Path(tmpdir) / "idx.json"
            idx_file.write_text(json.dumps(_make_two_chunk_index(), ensure_ascii=False), encoding="utf-8")

            fake_embedder = MockEmbedder()
            resp1 = _make_mock_response()
            resp2 = _make_mock_response(
                claims=[
                    {
                        "text": "Ratio proyectada en 1,2.",
                        "evidence": [
                            {
                                "source_id": "S2",
                                "quote": "La ratio de pensiones por pensionista se proyecta en 1,2 según el escenario base de la AIReF para el periodo analizado.",
                            }
                        ],
                    }
                ]
            )
            fake_generator = CompoundMockGenerator(responses=[resp1, resp2])

            with patch("rag_pensiones.answer.OllamaEmbedder", return_value=fake_embedder), patch(
                "rag_pensiones.answer.OllamaGenerator", return_value=fake_generator
            ):
                out_buf = io.StringIO()
                err_buf = io.StringIO()
                with patch("sys.stdout", out_buf), patch("sys.stderr", err_buf):
                    code = main(
                        [
                            "--query",
                            "¿Cuál es el gasto en 2050 y cuál es la ratio?",
                            "--top-k",
                            "2",
                            "--index",
                            str(idx_file),
                        ]
                    )

                self.assertEqual(code, 0)
                out = out_buf.getvalue()
                self.assertIn("CONSULTA COMPUESTA", out)
                self.assertIn("Subpreguntas evaluadas: 2", out)
                self.assertIn("DESGLOSE POR SUBPREGUNTA:", out)
                self.assertIn("[1] Subpregunta: ¿Cuál es el gasto en 2050?", out)
                self.assertIn("[2] Subpregunta: ¿cuál es la ratio?", out)
                self.assertIn("aumentará del 12,9 a l 16,1%", out)
                self.assertIn("Ratio proyectada en 1,2.", out)

    def test_cli_text_mode_compound_partial_abstention(self):
        """Verifica que la salida CLI refleja la respuesta parcial y el motivo de abstención sin ocultarlo."""
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as tmpdir:
            idx_file = Path(tmpdir) / "idx.json"
            idx_file.write_text(json.dumps(_make_two_chunk_index(), ensure_ascii=False), encoding="utf-8")

            fake_embedder = MockEmbedder()
            resp1 = _make_mock_response()
            resp2 = _make_mock_response(
                status="insufficient_evidence",
                claims=[],
                reason="Sin datos sobre cálculos individuales en el informe AIReF.",
            )
            fake_generator = CompoundMockGenerator(responses=[resp1, resp2])

            with patch("rag_pensiones.answer.OllamaEmbedder", return_value=fake_embedder), patch(
                "rag_pensiones.answer.OllamaGenerator", return_value=fake_generator
            ):
                out_buf = io.StringIO()
                err_buf = io.StringIO()
                with patch("sys.stdout", out_buf), patch("sys.stderr", err_buf):
                    code = main(
                        [
                            "--query",
                            "¿Cuál es el gasto en 2050 y cuánto cobraré en 2040?",
                            "--top-k",
                            "2",
                            "--index",
                            str(idx_file),
                        ]
                    )

                self.assertEqual(code, 0)
                out = out_buf.getvalue()
                self.assertIn("Abstención / Respuesta parcial por evidencia insuficiente", out)
                self.assertIn("MOTIVO GLOBAL DE LA ABSTENCIÓN / RESPUESTA PARCIAL:", out)
                self.assertIn("Sin datos sobre cálculos individuales", out)
                self.assertIn("16,1% del PIB", out)

    def test_cli_json_mode_compound(self):
        """Verifica que --json emite schema v2 estructurado con coverage completo."""
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as tmpdir:
            idx_file = Path(tmpdir) / "idx.json"
            idx_file.write_text(json.dumps(_make_two_chunk_index(), ensure_ascii=False), encoding="utf-8")

            fake_embedder = MockEmbedder()
            resp1 = _make_mock_response()
            resp2 = _make_mock_response(
                claims=[
                    {
                        "text": "Ratio 1,2.",
                        "evidence": [
                            {
                                "source_id": "S2",
                                "quote": "La ratio de pensiones por pensionista se proyecta en 1,2 según el escenario base de la AIReF para el periodo analizado.",
                            }
                        ],
                    }
                ]
            )
            fake_generator = CompoundMockGenerator(responses=[resp1, resp2])

            with patch("rag_pensiones.answer.OllamaEmbedder", return_value=fake_embedder), patch(
                "rag_pensiones.answer.OllamaGenerator", return_value=fake_generator
            ):
                out_buf = io.StringIO()
                err_buf = io.StringIO()
                with patch("sys.stdout", out_buf), patch("sys.stderr", err_buf):
                    code = main(
                        [
                            "--query",
                            "¿Cuál es el gasto en 2050 y cuál es la ratio?",
                            "--top-k",
                            "2",
                            "--index",
                            str(idx_file),
                            "--json",
                        ]
                    )

                self.assertEqual(code, 0)
                parsed = json.loads(out_buf.getvalue())
                self.assertEqual(parsed["schema_version"], SCHEMA_VERSION_V2)
                self.assertEqual(parsed["status"], "answered")
                self.assertEqual(len(parsed["coverage"]["parts"]), 2)
                self.assertEqual(parsed["parameters"]["generation_calls"], 2)


if __name__ == "__main__":
    unittest.main()
