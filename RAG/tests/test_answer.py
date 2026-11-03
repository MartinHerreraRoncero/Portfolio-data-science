"""Tests exhaustivos y deterministas offline para la generación fundamentada de respuestas."""

import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch
import urllib.error
import urllib.request

from rag_pensiones.answer import (
    ALLOWED_MODELS,
    DEFAULT_GENERATION_MODEL,
    MAX_MESSAGES_BYTES,
    MAX_RESPONSE_BYTES,
    RESPONSE_JSON_SCHEMA,
    SCHEMA_VERSION,
    SYSTEM_PROMPT,
    OllamaGenerator,
    _NoRedirectHandler,
    _reject_constant,
    answer,
    build_messages,
    build_response_schema,
    collapse_spaces,
    format_text_answer,
    main,
    validate_answer,
)
from rag_pensiones.chunking import ChunkRecord
from rag_pensiones.embeddings import DIMENSION
from rag_pensiones.ingest import PROJECT_ROOT
from rag_pensiones.search import DEFAULT_INDEX_PATH, LEGACY_INDEX_PATH, SearchHit, load_index


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


def _make_valid_index(chunks: list[dict] | None = None) -> dict:
    if chunks is None:
        chunks = [
            {
                "chunk_id": "chunk_id_1",
                "doc_id": "doc_test_1",
                "source_file": "test_doc.pdf",
                "published_at": "2025-03-31",
                "page_number": 95,
                "start_char": 0,
                "end_char": 100,
                "text": "aumentará del 12,9 a l 16,1% del PIB entre 2023 y 2050 y se reducirá posteriormente hasta el 14,7% en 2070.",
                "strategy": "page-local-v1:max_chars=1000:overlap_chars=150",
                "embedding": [1.0] + [0.0] * (DIMENSION - 1),
            }
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


class MockEmbedder:
    """Embedder simulado para pruebas de respuesta sin contactar con Ollama."""

    def __init__(self, digest: str = "test_embed_digest_sha256") -> None:
        self.digest = digest

    def get_digest(self) -> str:
        return self.digest

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[1.0] + [0.0] * (DIMENSION - 1)]


class MockGenerator:
    """Generador simulado para pruebas completas offline de respuesta estructurada."""

    def __init__(
        self,
        model: str = DEFAULT_GENERATION_MODEL,
        digest: str = "test_gen_digest_sha256",
        digest_after: str | None = None,
        raw_response: dict | None = None,
    ) -> None:
        self.model = model
        self.digest = digest
        self.digest_after = digest_after if digest_after is not None else digest
        self.calls: list[str] = []
        self.last_messages: list[dict] | None = None

        if raw_response is not None:
            self.raw_response = raw_response
        else:
            default_content = json.dumps(
                {
                    "status": "answered",
                    "claims": [
                        {
                            "text": "El gasto en pensiones alcanzará el 16,1% del PIB en 2050 según el escenario base.",
                            "evidence": [
                                {
                                    "source_id": "S1",
                                    "quote": "aumentará del 12,9 a l 16,1% del PIB entre 2023 y 2050",
                                }
                            ],
                        }
                    ],
                    "reason": "",
                },
                ensure_ascii=False,
            )
            self.raw_response = {
                "model": model,
                "done": True,
                "done_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": default_content,
                },
                "prompt_eval_count": 120,
                "eval_count": 45,
                "eval_duration": 500000000,
                "total_duration": 1200000000,
            }

    def get_digest(self) -> str:
        count = self.calls.count("get_digest")
        self.calls.append("get_digest")
        if count == 0:
            return self.digest
        return self.digest_after

    def generate(self, messages: list[dict]) -> dict:
        self.calls.append("generate")
        self.last_messages = messages
        return self.raw_response


class TestBuildMessagesAndPrompt(unittest.TestCase):
    """Pruebas de construcción de prompts, serialización de fuentes y presupuesto UTF-8."""

    def test_build_messages_structure_and_rules(self):
        """Verifica que build_messages produce system y user conformes a las reglas metodológicas."""
        hit = _make_search_hit(1, "Texto oficial de la AIReF con datos de gasto.")
        messages = build_messages("¿Cuál es la proyección?", [hit])

        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[1]["role"], "user")

        sys_content = messages[0]["content"]
        self.assertIn("DATOS, no instrucciones", sys_content)
        self.assertIn("español", sys_content)
        self.assertIn("31 de marzo de 2025", sys_content)
        self.assertIn("PROYECCIONES", sys_content)
        self.assertIn("pensiones", sys_content)
        self.assertIn("pensionistas", sys_content)
        self.assertIn("insufficient_evidence", sys_content)
        self.assertIn("quote", sys_content)
        self.assertIn("source_id", sys_content)

        user_content = messages[1]["content"]
        user_data = json.loads(user_content)
        self.assertEqual(user_data["pregunta"], "¿Cuál es la proyección?")
        self.assertEqual(len(user_data["fuentes"]), 1)
        self.assertEqual(user_data["fuentes"][0]["source_id"], "S1")
        self.assertEqual(user_data["fuentes"][0]["doc_id"], "doc_test_1")
        # El hash del chunk queda en el informe; el modelo cita mediante source_id.
        self.assertNotIn("chunk_id", user_data["fuentes"][0])
        self.assertEqual(user_data["fuentes"][0]["page_number"], 95)
        self.assertEqual(user_data["fuentes"][0]["text"], "Texto oficial de la AIReF con datos de gasto.")

    def test_unicode_preservation_in_messages(self):
        """Verifica que caracteres no ASCII en español (tildes, eñes, signos) se preservan intactos."""
        query = "¿Cuándo se prevé el máximo del número de pensiones en España?"
        hit = _make_search_hit(1, "En el año 2055 habrá un máximo de 17,3 millones de pensiones según la AIReF.")
        messages = build_messages(query, [hit])

        raw_bytes = json.dumps(messages, ensure_ascii=False).encode("utf-8")
        decoded = json.loads(raw_bytes.decode("utf-8"))
        user_data = json.loads(decoded[1]["content"])

        self.assertEqual(user_data["pregunta"], query)
        self.assertIn("año 2055 habrá", user_data["fuentes"][0]["text"])

    def test_messages_budget_guard_exceeded_raises_value_error(self):
        """Verifica que un volumen excesivo de fuentes (>6500 bytes UTF-8) dispara la guardia sin generar."""
        huge_text = "Texto largo sobre pensiones que supera el presupuesto. " * 150
        hits = [_make_search_hit(i, huge_text) for i in range(1, 5)]

        with self.assertRaises(ValueError) as ctx:
            build_messages("consulta", hits)
        self.assertIn("exceeds conservative budget", str(ctx.exception))
        self.assertIn("6500", str(ctx.exception))

    def test_build_messages_query_and_hits_validation(self):
        """Verifica rechazo de queries vacías o tipos incorrectos en hits."""
        with self.assertRaises(ValueError):
            build_messages("", [_make_search_hit(1)])
        with self.assertRaises(ValueError):
            build_messages("   \n\t  ", [_make_search_hit(1)])
        with self.assertRaises(ValueError):
            build_messages(None, [_make_search_hit(1)])  # type: ignore
        with self.assertRaises(ValueError):
            build_messages("consulta", ["not_a_hit"])  # type: ignore

    def test_system_prompt_budget_limit_1200_bytes(self):
        """Guardia auxiliar; el presupuesto real de los siete contextos se comprueba aparte."""
        prompt_bytes = SYSTEM_PROMPT.encode("utf-8")
        self.assertLessEqual(
            len(prompt_bytes),
            1200,
            f"SYSTEM_PROMPT excede el límite de 1200 bytes UTF-8 ({len(prompt_bytes)} bytes)",
        )

    def test_build_messages_admits_all_baseline_top4_cases_within_6500_bytes(self):
        """Regresión: build_messages debe admitir las 7 consultas y fuentes reales de baseline_top4.json dentro de 6500 bytes."""
        baseline_path = PROJECT_ROOT / "evaluation" / "baseline_top4.json"
        self.assertTrue(baseline_path.exists(), f"baseline_top4.json no encontrado en {baseline_path}")

        data = json.loads(baseline_path.read_text(encoding="utf-8"))
        cases = data.get("cases", [])
        self.assertEqual(len(cases), 7, f"Se esperaban 7 casos en baseline_top4.json, encontrados {len(cases)}")

        for case in cases:
            case_id = case["id"]
            question = case["question"]
            hits_data = case["hits"]
            self.assertEqual(len(hits_data), 4, f"El caso {case_id} debe tener exactamente 4 hits")

            hits: list[SearchHit] = []
            for h in hits_data:
                c_data = h["chunk"]
                chunk = ChunkRecord(
                    chunk_id=c_data["chunk_id"],
                    doc_id=c_data["doc_id"],
                    source_file=c_data["source_file"],
                    published_at=c_data["published_at"],
                    page_number=c_data["page_number"],
                    start_char=c_data["start_char"],
                    end_char=c_data["end_char"],
                    text=c_data["text"],
                    strategy=c_data["strategy"],
                )
                hits.append(
                    SearchHit(
                        rank=h["rank"],
                        score=h["score"],
                        chunk=chunk,
                    )
                )

            # Construir mensajes sin red y con textos íntegros
            messages = build_messages(query=question, hits=hits)
            self.assertEqual(len(messages), 2)

            serialized = json.dumps(messages, ensure_ascii=False).encode("utf-8")
            self.assertLessEqual(
                len(serialized),
                MAX_MESSAGES_BYTES,
                f"Caso {case_id} ({question!r}) excede el presupuesto de {MAX_MESSAGES_BYTES} bytes: {len(serialized)} bytes",
            )


class TestBuildResponseSchema(unittest.TestCase):
    """Pruebas del esquema dinámico anyOf con candidatos de citas literales por fuente."""

    def test_build_response_schema_exact_anyof_two_sources(self):
        """Verifica esquema anyOf exacto sobre 2 fuentes y frases conocidas."""
        hit1 = _make_search_hit(
            1,
            "El gasto en pensiones aumentará del 12,9 al 16,1% del PIB. Posteriormente se reducirá al 14,7% en 2070.",
        )
        hit2 = _make_search_hit(
            2,
            "Un pensionista puede recibir más de una pensión. La ratio pensionistas por pensión es relevante.",
        )
        messages = build_messages("¿Proyección?", [hit1, hit2])

        schema = build_response_schema(messages)

        self.assertEqual(schema["type"], "object")
        evidence_items = schema["properties"]["claims"]["items"]["properties"]["evidence"]["items"]
        self.assertIn("anyOf", evidence_items)
        anyof = evidence_items["anyOf"]
        self.assertEqual(len(anyof), 2)

        # Fuente S1
        self.assertEqual(anyof[0]["properties"]["source_id"]["const"], "S1")
        self.assertEqual(
            anyof[0]["properties"]["quote"]["enum"],
            [
                "El gasto en pensiones aumentará del 12,9 al 16,1% del PIB.",
                "Posteriormente se reducirá al 14,7% en 2070.",
            ],
        )
        self.assertEqual(anyof[0]["required"], ["source_id", "quote"])
        self.assertFalse(anyof[0]["additionalProperties"])

        # Fuente S2
        self.assertEqual(anyof[1]["properties"]["source_id"]["const"], "S2")
        self.assertEqual(
            anyof[1]["properties"]["quote"]["enum"],
            [
                "Un pensionista puede recibir más de una pensión.",
                "La ratio pensionistas por pensión es relevante.",
            ],
        )
        self.assertEqual(anyof[1]["required"], ["source_id", "quote"])
        self.assertFalse(anyof[1]["additionalProperties"])

    def test_build_response_schema_immutability(self):
        """Verifica que build_response_schema no muta RESPONSE_JSON_SCHEMA ni messages."""
        schema_snapshot = copy.deepcopy(RESPONSE_JSON_SCHEMA)
        hit = _make_search_hit(1, "Texto suficiente sobre pensiones para extraer candidatos.")
        messages = build_messages("consulta", [hit])
        messages_snapshot = copy.deepcopy(messages)

        dyn_schema = build_response_schema(messages)

        self.assertEqual(RESPONSE_JSON_SCHEMA, schema_snapshot)
        self.assertEqual(messages, messages_snapshot)
        self.assertIsNot(dyn_schema, RESPONSE_JSON_SCHEMA)

    def test_build_response_schema_preserves_footnote_32(self):
        """Verifica que notas al pie como [32] se preservan intactas en los candidatos de cita."""
        text_with_note = "En el escenario base [32], el gasto proyectado de pensiones aumenta hasta 2050."
        hit = _make_search_hit(1, text_with_note)
        messages = build_messages("consulta", [hit])

        schema = build_response_schema(messages)
        anyof = schema["properties"]["claims"]["items"]["properties"]["evidence"]["items"]["anyOf"]
        quotes = anyof[0]["properties"]["quote"]["enum"]
        self.assertEqual(quotes, ["En el escenario base [32], el gasto proyectado de pensiones aumenta hasta 2050."])
        self.assertIn("[32]", quotes[0])

    def test_build_response_schema_source_without_candidates_omitted_and_empty_raises(self):
        """Verifica que fuentes sin candidatos se omiten del esquema y que si ninguna tiene candidatos falla."""
        # S1 sin candidatos válidos (< 12 chars), S2 con candidato válido
        hit_short = _make_search_hit(1, "Corto.")
        hit_valid = _make_search_hit(2, "Este segmento tiene más de doce caracteres válidos.")
        messages = build_messages("consulta", [hit_short, hit_valid])

        schema = build_response_schema(messages)
        anyof = schema["properties"]["claims"]["items"]["properties"]["evidence"]["items"]["anyOf"]
        self.assertEqual(len(anyof), 1)
        self.assertEqual(anyof[0]["properties"]["source_id"]["const"], "S2")

        # Verificar que S1 se mantiene completo en messages
        user_content = json.loads(messages[1]["content"])
        self.assertEqual(len(user_content["fuentes"]), 2)
        self.assertEqual(user_content["fuentes"][0]["source_id"], "S1")

        # Ninguna fuente tiene candidatos válidos -> error explícito antes de POST
        hit_short2 = _make_search_hit(2, "Diminuto.")
        messages_no_candidates = build_messages("consulta", [hit_short, hit_short2])
        with self.assertRaises(ValueError) as ctx:
            build_response_schema(messages_no_candidates)
        self.assertIn("No valid quote candidates found", str(ctx.exception))


class TestValidateAnswerPayload(unittest.TestCase):
    """Pruebas exhaustivas de validación del payload estructurado de Qwen."""

    def setUp(self):
        self.hit1 = _make_search_hit(
            1,
            "aumentará del 12,9 a l 16,1% del PIB entre 2023 y 2050 y se reducirá posteriormente hasta el 14,7% en 2070.",
        )
        self.hit2 = _make_search_hit(
            2,
            "Un pensionista puede ser beneficiario de más de una pensión. La ratio pensionistas / pensiones se mantiene estable.",
        )
        self.hits = [self.hit1, self.hit2]

    def test_valid_answered_payload(self):
        """Verifica payload answered conforme con citas verificables."""
        payload = {
            "status": "answered",
            "claims": [
                {
                    "text": "El gasto aumentará hasta el 16,1% en 2050.",
                    "evidence": [
                        {
                            "source_id": "S1",
                            "quote": "aumentará del 12,9 a l 16,1% del PIB entre 2023 y 2050",
                        }
                    ],
                }
            ],
            "reason": "",
        }
        res = validate_answer(payload, self.hits)
        self.assertEqual(res["status"], "answered")
        self.assertEqual(len(res["claims"]), 1)

    def test_quote_whitespace_collapsing(self):
        """Verifica que citas con saltos de línea o dobles espacios se validan tras colapsar espacios."""
        hit_with_newlines = _make_search_hit(
            1,
            "aumentará del 12,9\n  a l 16,1% del PIB\n  entre 2023 y 2050.",
        )
        payload = {
            "status": "answered",
            "claims": [
                {
                    "text": "Proyección del gasto en 2050.",
                    "evidence": [
                        {
                            "source_id": "S1",
                            "quote": "aumentará del 12,9 a l 16,1% del PIB entre 2023 y 2050.",
                        }
                    ],
                }
            ],
            "reason": "",
        }
        res = validate_answer(payload, [hit_with_newlines])
        self.assertEqual(res["status"], "answered")

    def test_valid_insufficient_evidence_payload(self):
        """Verifica payload de abstención con claims vacíos y reason explicativo."""
        payload = {
            "status": "insufficient_evidence",
            "claims": [],
            "reason": "El informe no contiene datos para calcular pensiones individuales.",
        }
        res = validate_answer(payload, self.hits)
        self.assertEqual(res["status"], "insufficient_evidence")
        self.assertEqual(res["claims"], [])
        self.assertTrue(len(res["reason"]) > 0)

    def test_reject_invalid_status_and_types(self):
        """Verifica rechazo de status inválidos, booleanos o números."""
        for bad_status in ("partial", "unknown", True, False, 123, None):
            payload = {"status": bad_status, "claims": [], "reason": "motivo"}
            with self.assertRaises(ValueError):
                validate_answer(payload, self.hits)

    def test_reject_answered_with_non_empty_reason(self):
        """Verifica que status 'answered' exige reason estrictamente ''."""
        payload = {
            "status": "answered",
            "claims": [
                {
                    "text": "Afirmación válida.",
                    "evidence": [
                        {
                            "source_id": "S1",
                            "quote": "aumentará del 12,9 a l 16,1% del PIB entre 2023 y 2050",
                        }
                    ],
                }
            ],
            "reason": "Razón indebida",
        }
        with self.assertRaises(ValueError):
            validate_answer(payload, self.hits)

    def test_reject_answered_with_empty_or_excessive_claims(self):
        """Verifica que status 'answered' exige entre 1 y 6 claims."""
        # 0 claims
        with self.assertRaises(ValueError):
            validate_answer({"status": "answered", "claims": [], "reason": ""}, self.hits)

        # 7 claims (excede máximo 6)
        claim_template = {
            "text": "Afirmación.",
            "evidence": [
                {
                    "source_id": "S1",
                    "quote": "aumentará del 12,9 a l 16,1% del PIB entre 2023 y 2050",
                }
            ],
        }
        with self.assertRaises(ValueError):
            validate_answer(
                {"status": "answered", "claims": [claim_template] * 7, "reason": ""},
                self.hits,
            )

    def test_reject_insufficient_evidence_with_non_empty_claims_or_empty_reason(self):
        """Verifica que insufficient_evidence exige claims=[] y reason no vacío."""
        # claims no vacío
        with self.assertRaises(ValueError):
            validate_answer(
                {
                    "status": "insufficient_evidence",
                    "claims": [{"text": "Inventada", "evidence": []}],
                    "reason": "motivo",
                },
                self.hits,
            )

        # reason vacío
        with self.assertRaises(ValueError):
            validate_answer(
                {"status": "insufficient_evidence", "claims": [], "reason": ""},
                self.hits,
            )
        with self.assertRaises(ValueError):
            validate_answer(
                {"status": "insufficient_evidence", "claims": [], "reason": "   \n\t "},
                self.hits,
            )

    def test_reject_extra_keys_at_all_levels(self):
        """Verifica rechazo de campos adicionales en nivel raíz, claim o evidence."""
        # Extra en raíz
        with self.assertRaises(ValueError):
            validate_answer(
                {
                    "status": "insufficient_evidence",
                    "claims": [],
                    "reason": "motivo",
                    "extra_key": 123,
                },
                self.hits,
            )

        # Extra en claim
        with self.assertRaises(ValueError):
            validate_answer(
                {
                    "status": "answered",
                    "claims": [
                        {
                            "text": "Afirmación.",
                            "evidence": [
                                {
                                    "source_id": "S1",
                                    "quote": "aumentará del 12,9 a l 16,1% del PIB entre 2023 y 2050",
                                }
                            ],
                            "score": 0.99,
                        }
                    ],
                    "reason": "",
                },
                self.hits,
            )

        # Extra en evidence
        with self.assertRaises(ValueError):
            validate_answer(
                {
                    "status": "answered",
                    "claims": [
                        {
                            "text": "Afirmación.",
                            "evidence": [
                                {
                                    "source_id": "S1",
                                    "quote": "aumentará del 12,9 a l 16,1% del PIB entre 2023 y 2050",
                                    "page": 95,
                                }
                            ],
                        }
                    ],
                    "reason": "",
                },
                self.hits,
            )

    def test_reject_quote_from_wrong_source_or_unknown_source_id(self):
        """Verifica que una cita válida de S2 atribuida a S1 o un source_id inexistente son rechazados."""
        # source_id inexistente S3
        with self.assertRaises(ValueError):
            validate_answer(
                {
                    "status": "answered",
                    "claims": [
                        {
                            "text": "Afirmación.",
                            "evidence": [
                                {
                                    "source_id": "S3",
                                    "quote": "aumentará del 12,9 a l 16,1% del PIB entre 2023 y 2050",
                                }
                            ],
                        }
                    ],
                    "reason": "",
                },
                self.hits,
            )

        # Cita existente en S2 atribuida a S1
        with self.assertRaises(ValueError):
            validate_answer(
                {
                    "status": "answered",
                    "claims": [
                        {
                            "text": "Afirmación sobre beneficiarios.",
                            "evidence": [
                                {
                                    "source_id": "S1",
                                    "quote": "Un pensionista puede ser beneficiario de más de una pensión.",
                                }
                            ],
                        }
                    ],
                    "reason": "",
                },
                self.hits,
            )

    def test_reject_fabricated_quote(self):
        """Verifica rechazo cuando la cita no existe en ninguna fuente."""
        with self.assertRaises(ValueError):
            validate_answer(
                {
                    "status": "answered",
                    "claims": [
                        {
                            "text": "Afirmación inventada.",
                            "evidence": [
                                {
                                    "source_id": "S1",
                                    "quote": "Esta frase inventada no existe en el texto de S1.",
                                }
                            ],
                        }
                    ],
                    "reason": "",
                },
                self.hits,
            )

    def test_reject_quote_length_bounds(self):
        """Verifica rechazo de citas menores a 12 caracteres o mayores a 700 caracteres."""
        # Menor a 12 caracteres
        with self.assertRaises(ValueError):
            validate_answer(
                {
                    "status": "answered",
                    "claims": [
                        {
                            "text": "Afirmación.",
                            "evidence": [{"source_id": "S1", "quote": "aumentará"}],
                        }
                    ],
                    "reason": "",
                },
                self.hits,
            )

        # Mayor a 700 caracteres
        huge_quote = "aumentará del 12,9 a l 16,1% " * 30
        with self.assertRaises(ValueError):
            validate_answer(
                {
                    "status": "answered",
                    "claims": [
                        {
                            "text": "Afirmación.",
                            "evidence": [{"source_id": "S1", "quote": huge_quote}],
                        }
                    ],
                    "reason": "",
                },
                self.hits,
            )

    def test_reject_claim_text_or_reason_over_1200_chars(self):
        """Verifica que claim text y reason no superen 1200 caracteres."""
        huge_claim = "Texto muy extenso. " * 70
        with self.assertRaises(ValueError):
            validate_answer(
                {
                    "status": "answered",
                    "claims": [
                        {
                            "text": huge_claim,
                            "evidence": [
                                {
                                    "source_id": "S1",
                                    "quote": "aumentará del 12,9 a l 16,1% del PIB entre 2023 y 2050",
                                }
                            ],
                        }
                    ],
                    "reason": "",
                },
                self.hits,
            )

        huge_reason = "Motivo de abstención muy largo. " * 50
        with self.assertRaises(ValueError):
            validate_answer(
                {
                    "status": "insufficient_evidence",
                    "claims": [],
                    "reason": huge_reason,
                },
                self.hits,
            )

    def test_reject_evidence_count_bounds_per_claim(self):
        """Verifica que cada claim tenga entre 1 y 3 evidencias."""
        # 0 evidencias
        with self.assertRaises(ValueError):
            validate_answer(
                {
                    "status": "answered",
                    "claims": [{"text": "Afirmación sin evidencia.", "evidence": []}],
                    "reason": "",
                },
                self.hits,
            )

        # 4 evidencias (excede 3)
        ev = {
            "source_id": "S1",
            "quote": "aumentará del 12,9 a l 16,1% del PIB entre 2023 y 2050",
        }
        with self.assertRaises(ValueError):
            validate_answer(
                {
                    "status": "answered",
                    "claims": [{"text": "Afirmación.", "evidence": [ev, ev, ev, ev]}],
                    "reason": "",
                },
                self.hits,
            )


class TestOllamaGeneratorTransport(unittest.TestCase):
    """Pruebas de transporte HTTP, cliente de Ollama, opciones y seguridad offline."""

    def test_k2_model_and_schema_are_sent_without_fallback(self):
        model = "hf.co/IFM/K2-Horizon-7B-GGUF:Q4_K_M"
        opener = MagicMock()
        opener.open.return_value.__enter__.return_value.read.return_value = b'{}'
        generator = OllamaGenerator(model=model, opener=opener)
        generator.chat([{"role": "user", "content": "consulta"}], RESPONSE_JSON_SCHEMA)
        payload = json.loads(opener.open.call_args[0][0].data)
        self.assertEqual(payload["model"], model)
        self.assertEqual(payload["format"], RESPONSE_JSON_SCHEMA)
        self.assertEqual(payload["options"]["num_ctx"], 8192)

    def test_generator_init_validations(self):
        """Verifica rechazo de modelos no autorizados y timeouts no válidos."""
        with self.assertRaises(ValueError):
            OllamaGenerator(model="llama3:8b")
        with self.assertRaises(ValueError):
            OllamaGenerator(timeout=0)
        with self.assertRaises(ValueError):
            OllamaGenerator(timeout=-10)
        with self.assertRaises(ValueError):
            OllamaGenerator(timeout=True)  # type: ignore

    def test_generate_payload_options_and_schema(self):
        """Verifica que generate() envía la URL, método, esquema y opciones deterministas requeridas."""
        fake_opener = MagicMock()
        resp_mock = MagicMock()
        resp_mock.read.return_value = json.dumps(
            {
                "model": DEFAULT_GENERATION_MODEL,
                "done": True,
                "done_reason": "stop",
                "message": {"role": "assistant", "content": '{"status":"insufficient_evidence","claims":[],"reason":"R"}'},
                "prompt_eval_count": 50,
                "eval_count": 20,
                "eval_duration": 1000,
                "total_duration": 2000,
            }
        ).encode("utf-8")
        fake_opener.open.return_value.__enter__.return_value = resp_mock

        gen = OllamaGenerator(opener=fake_opener)
        hit = _make_search_hit(1, "Texto oficial de la AIReF con datos de gasto.")
        messages = build_messages("¿Cuál es la proyección?", [hit])
        res = gen.generate(messages)

        self.assertEqual(fake_opener.open.call_count, 1)
        req = fake_opener.open.call_args[0][0]
        self.assertEqual(req.full_url, "http://127.0.0.1:11434/api/chat")
        self.assertEqual(req.get_method(), "POST")

        sent_payload = json.loads(req.data.decode("utf-8"))
        self.assertEqual(sent_payload["model"], DEFAULT_GENERATION_MODEL)
        self.assertFalse(sent_payload["stream"])
        self.assertFalse(sent_payload["think"])
        self.assertEqual(sent_payload["keep_alive"], "5m")
        expected_schema = build_response_schema(messages)
        self.assertEqual(sent_payload["format"], expected_schema)
        self.assertIn("anyOf", sent_payload["format"]["properties"]["claims"]["items"]["properties"]["evidence"]["items"])
        self.assertEqual(sent_payload["options"]["temperature"], 0)
        self.assertEqual(sent_payload["options"]["seed"], 42)
        self.assertEqual(sent_payload["options"]["num_ctx"], 8192)
        self.assertEqual(sent_payload["options"]["num_predict"], 1024)

    def test_response_size_limit_1mib(self):
        """Verifica que respuestas superiores a 1 MiB son rechazadas inmediatamente."""
        fake_opener = MagicMock()
        resp_mock = MagicMock()
        # Generar datos que superan MAX_RESPONSE_BYTES (1 MiB)
        resp_mock.read.return_value = b"X" * (MAX_RESPONSE_BYTES + 10)
        fake_opener.open.return_value.__enter__.return_value = resp_mock

        gen = OllamaGenerator(opener=fake_opener)
        messages = build_messages("consulta", [_make_search_hit(1)])
        with self.assertRaises(RuntimeError) as ctx:
            gen.generate(messages)
        self.assertIn("exceeded 1 MiB limit", str(ctx.exception))

    def test_redirects_blocked_by_handler(self):
        """Verifica que el manejador de redirecciones bloquea intentos de salto HTTP."""
        handler = _NoRedirectHandler()
        req = urllib.request.Request("http://127.0.0.1:11434/api/chat")
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            handler.redirect_request(req, None, 302, "Found", {}, "http://evil.com")
        self.assertIn("HTTP redirects are blocked", str(ctx.exception))

    def test_get_digest_success_and_errors(self):
        """Verifica lectura de digest exacto y detección de modelo ausente o digest vacío."""
        fake_opener = MagicMock()
        resp_mock = MagicMock()
        resp_mock.read.return_value = json.dumps(
            {
                "models": [
                    {"name": "qwen3.5:9b", "digest": "sha256:abc12345"},
                    {"name": "qwen3.5:4b-q4_K_M", "digest": "sha256:def67890"},
                ]
            }
        ).encode("utf-8")
        fake_opener.open.return_value.__enter__.return_value = resp_mock

        gen = OllamaGenerator(model="qwen3.5:9b", opener=fake_opener)
        digest = gen.get_digest()
        self.assertEqual(digest, "sha256:abc12345")

        # Modelo ausente
        resp_mock.read.return_value = json.dumps({"models": [{"name": "otro_modelo", "digest": "123"}]}).encode("utf-8")
        with self.assertRaises(ValueError):
            gen.get_digest()

        # Digest vacío
        resp_mock.read.return_value = json.dumps(
            {"models": [{"name": "qwen3.5:9b", "digest": "   "}]}
        ).encode("utf-8")
        with self.assertRaises(ValueError):
            gen.get_digest()

    def test_transport_network_errors_handling(self):
        """Verifica propagación clara de errores HTTP, fallos de conexión y timeouts."""
        fake_opener = MagicMock()
        gen = OllamaGenerator(opener=fake_opener)
        messages = build_messages("consulta", [_make_search_hit(1)])

        # HTTP error 500
        fake_opener.open.side_effect = urllib.error.HTTPError("http://127.0.0.1:11434", 500, "Internal Error", {}, None)
        with self.assertRaises(RuntimeError):
            gen.generate(messages)

        # URLError (conexión rechazada)
        fake_opener.open.side_effect = urllib.error.URLError("Connection refused")
        with self.assertRaises(ConnectionError):
            gen.generate(messages)

        # TimeoutError
        fake_opener.open.side_effect = TimeoutError("Timed out")
        with self.assertRaises(TimeoutError):
            gen.generate(messages)

    def test_audit_httperror_oversized_and_closed(self):
        """Verifica que HTTPError lee máximo 1 MiB + 1, cierra la respuesta y emite error compacto sin volcar cuerpo."""
        fake_opener = MagicMock()
        gen = OllamaGenerator(opener=fake_opener)
        messages = build_messages("consulta", [_make_search_hit(1)])

        mock_fp = MagicMock()
        mock_fp.read.return_value = b"E" * (MAX_RESPONSE_BYTES + 10)
        mock_fp.close = MagicMock()

        http_err = urllib.error.HTTPError(
            url="http://127.0.0.1:11434/api/chat",
            code=502,
            msg="Bad Gateway",
            hdrs={},
            fp=mock_fp,
        )
        fake_opener.open.side_effect = http_err

        with self.assertRaises(RuntimeError) as ctx:
            gen.generate(messages)

        err_msg = str(ctx.exception)
        self.assertIn("HTTP error 502", err_msg)
        self.assertIn("Response exceeded 1 MiB limit", err_msg)
        self.assertNotIn("EEEEEEEE", err_msg)
        mock_fp.read.assert_called_with(MAX_RESPONSE_BYTES + 1)
        mock_fp.close.assert_called_once()

    def test_ollama_generator_capture_chat_request_seam(self):
        """Verifica la costura opt-in de captura del payload real en OllamaGenerator."""
        fake_opener = MagicMock()
        resp_mock = MagicMock()
        resp_mock.read.return_value = json.dumps({"message": {"content": "{}"}}).encode("utf-8")
        fake_opener.open.return_value.__enter__.return_value = resp_mock

        # 1. Por defecto: capture_chat_request está ausente/desactivado y no existe last_chat_request
        gen_default = OllamaGenerator(opener=fake_opener)
        self.assertFalse(getattr(gen_default, "capture_chat_request", False))
        self.assertFalse(hasattr(gen_default, "last_chat_request"))

        messages = [{"role": "user", "content": "test"}]
        schema = {"type": "object"}
        gen_default.chat(messages, schema)
        self.assertFalse(hasattr(gen_default, "last_chat_request"))

        # 2. Opt-in: capture_chat_request = True captura deepcopy del payload antes de serializar
        gen_capture = OllamaGenerator(opener=fake_opener)
        gen_capture.capture_chat_request = True
        gen_capture.chat(messages, schema)

        self.assertTrue(hasattr(gen_capture, "last_chat_request"))
        captured = gen_capture.last_chat_request
        self.assertIsInstance(captured, dict)

        # Bytes exactos del request capturado coinciden con el cuerpo transmitido
        req_sent = fake_opener.open.call_args[0][0]
        actual_bytes = req_sent.data
        expected_bytes = json.dumps(captured, ensure_ascii=False).encode("utf-8")
        self.assertEqual(actual_bytes, expected_bytes)

        # 7 claves exactas en el body del payload (incluyendo keep_alive: '5m')
        self.assertEqual(
            set(captured.keys()),
            {"model", "messages", "stream", "think", "keep_alive", "format", "options"},
        )
        self.assertEqual(captured["keep_alive"], "5m")

        # 4 opciones declaradas exactas, sin alteración de muestreo
        self.assertEqual(
            captured["options"],
            {"temperature": 0, "seed": 42, "num_ctx": 8192, "num_predict": 1024},
        )


class TestAnswerFullPipeline(unittest.TestCase):
    """Pruebas del flujo completo de answer() con comprobación de metadatos e invariantes."""

    def test_full_pipeline_answered_flow(self):
        """Verifica el flujo answered completo y la estructura canónica del reporte."""
        index = _make_valid_index()
        embedder = MockEmbedder()
        generator = MockGenerator(model="qwen3.5:9b")

        report = answer(
            index=index,
            query="¿Cuál es la proyección de gasto en 2050?",
            top_k=1,
            model="qwen3.5:9b",
            embedder=embedder,
            generator=generator,
        )

        self.assertEqual(report["schema_version"], SCHEMA_VERSION)
        self.assertEqual(report["query"], "¿Cuál es la proyección de gasto en 2050?")
        self.assertEqual(report["status"], "answered")
        self.assertEqual(len(report["claims"]), 1)
        self.assertEqual(report["reason"], "")
        self.assertEqual(len(report["sources"]), 1)
        self.assertEqual(report["sources"][0]["source_id"], "S1")
        self.assertEqual(report["top_k_requested"], 1)
        self.assertEqual(report["corpus_size"], 1)
        self.assertEqual(report["retrieval_model"], "embeddinggemma:300m")
        self.assertEqual(report["generation_model"], "qwen3.5:9b")
        self.assertEqual(report["generation_model_digest"], "test_gen_digest_sha256")
        self.assertEqual(report["parameters"]["temperature"], 0)
        self.assertEqual(report["parameters"]["seed"], 42)
        self.assertEqual(report["parameters"]["num_ctx"], 8192)
        self.assertEqual(report["parameters"]["num_predict"], 1024)
        self.assertEqual(report["parameters"]["top_k"], 1)
        self.assertEqual(report["stats"]["prompt_eval_count"], 120)
        self.assertEqual(report["stats"]["eval_count"], 45)
        self.assertTrue(report["validation"]["citation_quotes_checked"])
        self.assertFalse(report["validation"]["semantic_faithfulness_evaluated"])

    def test_full_pipeline_insufficient_evidence_flow(self):
        """Verifica el flujo completo ante pregunta que exige abstención."""
        index = _make_valid_index()
        embedder = MockEmbedder()
        raw_abstention = {
            "model": "qwen3.5:9b",
            "done": True,
            "done_reason": "stop",
            "message": {
                "role": "assistant",
                "content": json.dumps(
                    {
                        "status": "insufficient_evidence",
                        "claims": [],
                        "reason": "El informe no contiene datos para calcular pensiones individuales.",
                    },
                    ensure_ascii=False,
                ),
            },
            "prompt_eval_count": 80,
            "eval_count": 25,
            "eval_duration": 300000000,
            "total_duration": 800000000,
        }
        generator = MockGenerator(model="qwen3.5:9b", raw_response=raw_abstention)

        report = answer(
            index=index,
            query="¿Cuál será mi pensión mensual en 2045?",
            top_k=1,
            model="qwen3.5:9b",
            embedder=embedder,
            generator=generator,
        )

        self.assertEqual(report["status"], "insufficient_evidence")
        self.assertEqual(report["claims"], [])
        self.assertIn("pensiones individuales", report["reason"])

    def test_index_and_hits_immutability(self):
        """Verifica que answer() no muta en absoluto el índice ni sus vectores."""
        index = _make_valid_index()
        index_snapshot = copy.deepcopy(index)
        embedder = MockEmbedder()
        generator = MockGenerator()

        answer(index=index, query="consulta", top_k=1, embedder=embedder, generator=generator)

        self.assertEqual(index, index_snapshot)

    def test_digest_changed_during_generation_aborts(self):
        """Verifica aborto si el digest del modelo cambia durante la generación."""
        index = _make_valid_index()
        generator = MockGenerator(digest="digest_A", digest_after="digest_MUTATED")

        with self.assertRaises(RuntimeError) as ctx:
            answer(index=index, query="consulta", top_k=1, embedder=MockEmbedder(), generator=generator)
        self.assertIn("digest changed during generation", str(ctx.exception))

    def test_raw_response_anomalies_aborted(self):
        """Verifica rechazo de respuestas incompletas, truncadas, con roles erróneos o tool_calls."""
        index = _make_valid_index()
        embedder = MockEmbedder()

        # done = False
        raw_bad_done = {"model": "qwen3.5:9b", "done": False, "done_reason": "stop"}
        with self.assertRaises(ValueError):
            answer(index=index, query="q", embedder=embedder, generator=MockGenerator(raw_response=raw_bad_done))

        # done_reason = "length" (truncado)
        raw_length = {
            "model": "qwen3.5:9b",
            "done": True,
            "done_reason": "length",
            "message": {"role": "assistant", "content": "parcial"},
        }
        with self.assertRaises(ValueError):
            answer(index=index, query="q", embedder=embedder, generator=MockGenerator(raw_response=raw_length))

        # model erróneo
        raw_bad_model = {
            "model": "wrong:model",
            "done": True,
            "done_reason": "stop",
            "message": {"role": "assistant", "content": "{}"},
        }
        with self.assertRaises(ValueError):
            answer(index=index, query="q", embedder=embedder, generator=MockGenerator(raw_response=raw_bad_model))

        # tool_calls presentes
        raw_tools = {
            "model": "qwen3.5:9b",
            "done": True,
            "done_reason": "stop",
            "message": {
                "role": "assistant",
                "content": "{}",
                "tool_calls": [{"name": "fake_tool"}],
            },
            "prompt_eval_count": 10,
            "eval_count": 10,
            "eval_duration": 10,
            "total_duration": 20,
        }
        with self.assertRaises(ValueError):
            answer(index=index, query="q", embedder=embedder, generator=MockGenerator(raw_response=raw_tools))

    def test_stats_validations_and_limits(self):
        """Verifica validación de estadísticas y límite superior de prompt_eval_count."""
        index = _make_valid_index()
        embedder = MockEmbedder()

        # prompt_eval_count supera límite (8192-1024-256 = 6912)
        raw_overflow = {
            "model": "qwen3.5:9b",
            "done": True,
            "done_reason": "stop",
            "message": {
                "role": "assistant",
                "content": '{"status":"insufficient_evidence","claims":[],"reason":"R"}',
            },
            "prompt_eval_count": 7000,
            "eval_count": 10,
            "eval_duration": 10,
            "total_duration": 20,
        }
        with self.assertRaises(ValueError) as ctx:
            answer(index=index, query="q", embedder=embedder, generator=MockGenerator(raw_response=raw_overflow))
        self.assertIn("exceeds maximum allowed budget", str(ctx.exception))

    def test_pre_network_validations_abort_before_contacting_clients(self):
        """Verifica que fallos en index, query, top_k o model abortan antes de contactar red."""
        index = _make_valid_index()

        class BombClient:
            def get_digest(self):
                raise AssertionError("Client contacted despite invalid input!")

        # Query inválida
        with self.assertRaises(ValueError):
            answer(index, "   ", embedder=BombClient(), generator=BombClient())  # type: ignore

        # top_k booleano o inválido
        with self.assertRaises(ValueError):
            answer(index, "consulta", top_k=True, embedder=BombClient(), generator=BombClient())  # type: ignore
        with self.assertRaises(ValueError):
            answer(index, "consulta", top_k=0, embedder=BombClient(), generator=BombClient())  # type: ignore

        # Modelo no permitido
        with self.assertRaises(ValueError):
            answer(index, "consulta", model="llama3:8b", embedder=BombClient(), generator=BombClient())  # type: ignore

        # Generator con modelo discordante
        mismatched_gen = MockGenerator(model="qwen3.5:4b-q4_K_M")
        with self.assertRaises(ValueError):
            answer(index, "consulta", model="qwen3.5:9b", generator=mismatched_gen)

    def test_model_4b_runs_without_fallback(self):
        """Verifica que el modelo 4B ejecuta con sus parámetros sin intentar fallback a 9B."""
        index = _make_valid_index()
        embedder = MockEmbedder()
        gen_4b = MockGenerator(model="qwen3.5:4b-q4_K_M")

        report = answer(
            index=index,
            query="¿Cuál es la proyección?",
            top_k=1,
            model="qwen3.5:4b-q4_K_M",
            embedder=embedder,
            generator=gen_4b,
        )
        self.assertEqual(report["generation_model"], "qwen3.5:4b-q4_K_M")

    def test_audit_role_and_content_anomalies(self):
        """Auditoría: rechazo de rol incorrecto, content malformado/truncado, NaN/Infinity y overflow 1e999."""
        index = _make_valid_index()
        embedder = MockEmbedder()

        valid_raw = {
            "model": "qwen3.5:9b",
            "done": True,
            "done_reason": "stop",
            "message": {
                "role": "assistant",
                "content": '{"status":"insufficient_evidence","claims":[],"reason":"Razón válida"}',
            },
            "prompt_eval_count": 50,
            "eval_count": 20,
            "eval_duration": 1000,
            "total_duration": 2000,
        }

        # 1. Rol incorrecto
        for bad_role in ("user", "system", "tool", "", None, 123):
            with self.subTest(bad_role=bad_role):
                raw = copy.deepcopy(valid_raw)
                raw["message"]["role"] = bad_role
                with self.assertRaises(ValueError):
                    answer(index=index, query="consulta", embedder=embedder, generator=MockGenerator(raw_response=raw))

        # 2. Content JSON malformado / truncado / vacío
        for bad_content in (
            '{"status": "answered", "claims":',
            '{"status": "answered"',
            "",
            "   \n\t  ",
            "{bad_json: true}",
            None,
        ):
            with self.subTest(bad_content=bad_content):
                raw = copy.deepcopy(valid_raw)
                raw["message"]["content"] = bad_content
                with self.assertRaises(ValueError):
                    answer(index=index, query="consulta", embedder=embedder, generator=MockGenerator(raw_response=raw))

        # 3. NaN, Infinity anidados y overflow 1e999
        for bad_json in (
            '{"status":"answered","claims":[],"reason":"","val":NaN}',
            '{"status":"answered","claims":[],"reason":"","val":Infinity}',
            '{"status":"answered","claims":[],"reason":"","val":-Infinity}',
            '{"status":"answered","claims":[],"reason":"","nested":{"x":NaN}}',
            '{"status":"answered","claims":[],"reason":"","val":1e999}',
            '{"status":"answered","claims":[{"text":"D","evidence":[{"source_id":"S1","quote":"q","num":1e999}]}],"reason":""}',
        ):
            with self.subTest(bad_json=bad_json):
                raw = copy.deepcopy(valid_raw)
                raw["message"]["content"] = bad_json
                with self.assertRaisesRegex(ValueError, "Malformed JSON in assistant content"):
                    answer(index=index, query="consulta", embedder=embedder, generator=MockGenerator(raw_response=raw))

    def test_audit_stats_validation_anomalies(self):
        """Auditoría: rechazo de estadísticas obligatorias booleanas, negativas, ausentes y opcionales inválidas."""
        index = _make_valid_index()
        embedder = MockEmbedder()

        valid_raw = {
            "model": "qwen3.5:9b",
            "done": True,
            "done_reason": "stop",
            "message": {
                "role": "assistant",
                "content": '{"status":"insufficient_evidence","claims":[],"reason":"R"}',
            },
            "prompt_eval_count": 50,
            "eval_count": 20,
            "eval_duration": 1000,
            "total_duration": 2000,
        }

        # Estadísticas obligatorias: booleanas, negativas, ausentes o no-enteras
        int_stats = ("prompt_eval_count", "eval_count", "eval_duration", "total_duration")
        for stat in int_stats:
            for bad_val in (True, False, -1, -50, 1.5, "10", 1e999, float("inf"), float("nan")):
                with self.subTest(stat=stat, bad_val=bad_val):
                    raw = copy.deepcopy(valid_raw)
                    raw[stat] = bad_val
                    with self.assertRaises(ValueError):
                        answer(index=index, query="q", embedder=embedder, generator=MockGenerator(raw_response=raw))

            # Ausente
            with self.subTest(stat=stat, missing=True):
                raw = copy.deepcopy(valid_raw)
                del raw[stat]
                with self.assertRaises(ValueError):
                    answer(index=index, query="q", embedder=embedder, generator=MockGenerator(raw_response=raw))

        # Estadísticas opcionales: booleanas, negativas o no-enteras
        for opt_stat in ("load_duration", "prompt_eval_duration"):
            for bad_val in (True, False, -1, 1.5, "200", 1e999):
                with self.subTest(opt_stat=opt_stat, bad_val=bad_val):
                    raw = copy.deepcopy(valid_raw)
                    raw[opt_stat] = bad_val
                    with self.assertRaises(ValueError):
                        answer(index=index, query="q", embedder=embedder, generator=MockGenerator(raw_response=raw))


class TestCLIAndOutput(unittest.TestCase):
    """Pruebas de la interfaz de línea de comandos en modo texto y JSON."""

    def test_cli_text_output_answered(self):
        """Verifica salida legible en texto para respuestas fundamentadas con citas."""
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as tmpdir:
            idx_file = Path(tmpdir) / "idx.json"
            idx_file.write_text(json.dumps(_make_valid_index(), ensure_ascii=False), encoding="utf-8")

            fake_embedder = MockEmbedder()
            fake_generator = MockGenerator()

            with patch("rag_pensiones.answer.OllamaEmbedder", return_value=fake_embedder), patch(
                "rag_pensiones.answer.OllamaGenerator", return_value=fake_generator
            ):
                out_buf = io.StringIO()
                err_buf = io.StringIO()
                with patch("sys.stdout", out_buf), patch("sys.stderr", err_buf):
                    code = main(
                        [
                            "--query",
                            "¿Cuál es el gasto?",
                            "--top-k",
                            "1",
                            "--index",
                            str(idx_file),
                        ]
                    )

                self.assertEqual(code, 0)
                self.assertEqual(err_buf.getvalue(), "")
                out = out_buf.getvalue()
                self.assertIn("RESPUESTA RAG — PENSIONES AIREF 2025", out)
                self.assertIn("Respondida con citas comprobadas", out)
                self.assertIn("[S1]", out)
                self.assertIn("aumentará del 12,9 a l 16,1%", out)
                self.assertIn("NOTA METODOLÓGICA:", out)

    def test_cli_text_output_insufficient(self):
        """Verifica salida en modo texto para abstenciones con motivo."""
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as tmpdir:
            idx_file = Path(tmpdir) / "idx.json"
            idx_file.write_text(json.dumps(_make_valid_index(), ensure_ascii=False), encoding="utf-8")

            fake_embedder = MockEmbedder()
            raw_abstention = {
                "model": "qwen3.5:9b",
                "done": True,
                "done_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": json.dumps(
                        {
                            "status": "insufficient_evidence",
                            "claims": [],
                            "reason": "Pregunta de cálculo individual fuera del alcance.",
                        },
                        ensure_ascii=False,
                    ),
                },
                "prompt_eval_count": 60,
                "eval_count": 20,
                "eval_duration": 100,
                "total_duration": 200,
            }
            fake_generator = MockGenerator(raw_response=raw_abstention)

            with patch("rag_pensiones.answer.OllamaEmbedder", return_value=fake_embedder), patch(
                "rag_pensiones.answer.OllamaGenerator", return_value=fake_generator
            ):
                out_buf = io.StringIO()
                err_buf = io.StringIO()
                with patch("sys.stdout", out_buf), patch("sys.stderr", err_buf):
                    code = main(["--query", "¿Cuánto cobraré?", "--index", str(idx_file)])

                self.assertEqual(code, 0)
                out = out_buf.getvalue()
                self.assertIn("Abstención por evidencia insuficiente", out)
                self.assertIn("MOTIVO DE LA ABSTENCIÓN:", out)
                self.assertIn("cálculo individual fuera del alcance", out)

    def test_cli_json_mode(self):
        """Verifica salida JSON parseable y conforme en stdout con flag --json."""
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as tmpdir:
            idx_file = Path(tmpdir) / "idx.json"
            idx_file.write_text(json.dumps(_make_valid_index(), ensure_ascii=False), encoding="utf-8")

            fake_embedder = MockEmbedder()
            fake_generator = MockGenerator()

            with patch("rag_pensiones.answer.OllamaEmbedder", return_value=fake_embedder), patch(
                "rag_pensiones.answer.OllamaGenerator", return_value=fake_generator
            ):
                out_buf = io.StringIO()
                err_buf = io.StringIO()
                with patch("sys.stdout", out_buf), patch("sys.stderr", err_buf):
                    code = main(["--query", "consulta", "--index", str(idx_file), "--json"])

                self.assertEqual(code, 0)
                self.assertEqual(err_buf.getvalue(), "")
                parsed = json.loads(out_buf.getvalue())
                self.assertEqual(parsed["schema_version"], SCHEMA_VERSION)
                self.assertEqual(parsed["status"], "answered")

    def test_cli_errors_to_stderr_no_traceback(self):
        """Verifica que errores en CLI escriben a stderr sin volcar trazas de traceback."""
        # Query vacía
        out_buf = io.StringIO()
        err_buf = io.StringIO()
        with patch("sys.stdout", out_buf), patch("sys.stderr", err_buf):
            code = main(["--query", "   "])
        self.assertNotEqual(code, 0)
        self.assertIn("Error:", err_buf.getvalue())
        self.assertNotIn("Traceback", err_buf.getvalue())

        # top-k inválido
        err_buf = io.StringIO()
        with patch("sys.stdout", out_buf), patch("sys.stderr", err_buf):
            code = main(["--query", "valida", "--top-k", "0"])
        self.assertNotEqual(code, 0)
        self.assertIn("Error:", err_buf.getvalue())

        # Índice fuera de proyecto
        err_buf = io.StringIO()
        with patch("sys.stdout", out_buf), patch("sys.stderr", err_buf):
            code = main(["--query", "valida", "--index", "C:/outside/index.json"])
        self.assertNotEqual(code, 0)
        self.assertIn("Error:", err_buf.getvalue())

    def test_real_index_offline_answer(self):
        """Verifica ejecución de answer() sobre el índice real data/processed/airef_2025_embeddings.json."""
        index_data = load_index(LEGACY_INDEX_PATH)
        embedder = MockEmbedder(digest=index_data["model_digest"])

        # Generador que responde con una cita existente en el chunk de la pág 95
        raw_ans = {
            "model": "qwen3.5:9b",
            "done": True,
            "done_reason": "stop",
            "message": {
                "role": "assistant",
                "content": json.dumps(
                    {
                        "status": "answered",
                        "claims": [
                            {
                                "text": "El gasto aumentará del 12,9 al 16,1% del PIB.",
                                "evidence": [
                                    {
                                        "source_id": "S1",
                                        "quote": "Evolución del gasto en pensiones",
                                    }
                                ],
                            }
                        ],
                        "reason": "",
                    },
                    ensure_ascii=False,
                ),
            },
            "prompt_eval_count": 150,
            "eval_count": 40,
            "eval_duration": 400000000,
            "total_duration": 900000000,
        }
        generator = MockGenerator(model="qwen3.5:9b", raw_response=raw_ans)

        report = answer(
            index=index_data,
            query="¿Cuál es la evolución del gasto?",
            top_k=4,
            model="qwen3.5:9b",
            embedder=embedder,
            generator=generator,
        )

        self.assertEqual(report["status"], "answered")
        self.assertEqual(len(report["sources"]), 4)
        self.assertEqual(report["corpus_size"], 13)


if __name__ == "__main__":
    unittest.main()
