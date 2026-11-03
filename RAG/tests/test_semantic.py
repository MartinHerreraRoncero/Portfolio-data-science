"""Tests deterministas offline para el flujo RAG semántico local con planificación y recuperación por necesidad."""

import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from rag_pensiones.answer import (
    ALLOWED_MODELS,
    DEFAULT_GENERATION_MODEL,
    answer,
    validate_chat_stats,
)
from rag_pensiones.chunking import ChunkRecord
from rag_pensiones.embeddings import DIMENSION
from rag_pensiones.ingest import PROJECT_ROOT
from rag_pensiones.planning import (
    PLAN_JSON_SCHEMA,
    RESOLVE_JSON_SCHEMA,
    OllamaPlanner,
    validate_resolved_question,
)
from rag_pensiones.search import DEFAULT_INDEX_PATH, SearchHit, load_index
from rag_pensiones.semantic import (
    SCHEMA_VERSION,
    answer_semantic,
    format_text_semantic,
    main,
)


def _make_unit_vector(active_idx: int) -> list[float]:
    vec = [0.0] * DIMENSION
    vec[active_idx] = 1.0
    return vec


def _make_semantic_test_index() -> dict:
    chunks = [
        {
            "chunk_id": "chunk_c1",
            "doc_id": "doc_airef_1",
            "source_file": "airef_test.pdf",
            "published_at": "2025-03-31",
            "page_number": 95,
            "start_char": 0,
            "end_char": 95,
            "text": "El gasto en pensiones alcanzará el 16,1% del PIB en 2050 según el escenario base de la AIReF.",
            "strategy": "page-local-v1:max_chars=1000:overlap_chars=150",
            "embedding": _make_unit_vector(0),
        },
        {
            "chunk_id": "chunk_c2",
            "doc_id": "doc_airef_1",
            "source_file": "airef_test.pdf",
            "published_at": "2025-03-31",
            "page_number": 96,
            "start_char": 0,
            "end_char": 98,
            "text": "El gasto en pensiones se reducirá hasta el 14,7% del PIB en 2070 según el informe oficial de la AIReF.",
            "strategy": "page-local-v1:max_chars=1000:overlap_chars=150",
            "embedding": _make_unit_vector(1),
        },
        {
            "chunk_id": "chunk_c3",
            "doc_id": "doc_airef_1",
            "source_file": "airef_test.pdf",
            "published_at": "2025-03-31",
            "page_number": 97,
            "start_char": 0,
            "end_char": 90,
            "text": "La ratio de pensiones por pensionista se sitúa en 1,18 en 2050 según las estimaciones de AIReF.",
            "strategy": "page-local-v1:max_chars=1000:overlap_chars=150",
            "embedding": _make_unit_vector(2),
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


class SemanticMockEmbedder:
    """Embedder determinista para pruebas de búsqueda semántica sin conexión."""

    def __init__(self, digest: str = "test_embed_digest_sha256") -> None:
        self.digest = digest
        self.recorded_queries: list[str] = []

    def get_digest(self) -> str:
        return self.digest

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        results = []
        for text in texts:
            self.recorded_queries.append(text)
            clean = text.lower()
            if "2050" in clean and "2070" not in clean and "ratio" not in clean:
                results.append(_make_unit_vector(0))
            elif "2070" in clean:
                results.append(_make_unit_vector(1))
            elif "ratio" in clean:
                results.append(_make_unit_vector(2))
            else:
                # Vector que activa tanto c1 como c2
                vec = [0.0] * DIMENSION
                vec[0] = 0.8
                vec[1] = 0.6
                results.append(vec)
        return results


class SemanticMockGenerator:
    """Generador determinista para chat (plan/resolución) y generate (respuestas con citas)."""

    def __init__(
        self,
        model: str = DEFAULT_GENERATION_MODEL,
        digest: str = "test_gen_digest_sha256",
        plan_content: dict | None = None,
        resolve_content: dict | None = None,
        answer_responses: list[dict] | None = None,
        mutate_digest_at_call: int | None = None,
        mutated_digest: str = "mutated_digest_sha256",
    ) -> None:
        self.model = model
        self.digest = digest
        self.calls: list[str] = []
        self.plan_content = plan_content
        self.resolve_content = resolve_content
        self.answer_responses = list(answer_responses) if answer_responses else []
        self.mutate_digest_at_call = mutate_digest_at_call
        self.mutated_digest = mutated_digest
        self.recorded_generate_queries: list[str] = []
        self.recorded_resolve_deps: list[list[dict]] = []

    def get_digest(self) -> str:
        count = len(self.calls)
        self.calls.append("get_digest")
        if self.mutate_digest_at_call is not None and count >= self.mutate_digest_at_call:
            return self.mutated_digest
        return self.digest

    def chat(self, messages: list[dict], response_schema: dict) -> dict:
        self.calls.append("chat")
        if response_schema == PLAN_JSON_SCHEMA:
            if self.plan_content is not None:
                content = json.dumps(self.plan_content, ensure_ascii=False)
            else:
                content = json.dumps(
                    {
                        "subquestions": [
                            {
                                "id": "p1",
                                "question": "¿Cuál es el gasto proyectado para 2050?",
                                "required_evidence": "Gasto proyectado 2050.",
                                "depends_on": [],
                            }
                        ]
                    },
                    ensure_ascii=False,
                )
        elif response_schema == RESOLVE_JSON_SCHEMA:
            # Registrar dependencias recibidas para verificación de pruebas
            for m in messages:
                if m.get("role") == "user":
                    payload = json.loads(m.get("content", "{}"))
                    self.recorded_resolve_deps.append(payload.get("resultados_padres", []))

            if self.resolve_content is not None:
                content = json.dumps(self.resolve_content, ensure_ascii=False)
            else:
                content = json.dumps(
                    {"question": "¿Cuál es la diferencia de gasto proyectado en 2050 frente a 2070?"},
                    ensure_ascii=False,
                )
        else:
            content = json.dumps({"status": "answered", "claims": [], "reason": ""})

        return {
            "model": self.model,
            "done": True,
            "done_reason": "stop",
            "message": {"role": "assistant", "content": content},
            "prompt_eval_count": 120,
            "eval_count": 45,
            "eval_duration": 300000000,
            "total_duration": 750000000,
        }

    def generate(self, messages: list[dict]) -> dict:
        self.calls.append("generate")
        for m in messages:
            if m.get("role") == "user":
                payload = json.loads(m.get("content", "{}"))
                self.recorded_generate_queries.append(payload.get("pregunta", ""))

        if self.answer_responses:
            raw = self.answer_responses.pop(0)
            return raw

        default_content = json.dumps(
            {
                "status": "answered",
                "claims": [
                    {
                        "text": "El gasto en pensiones alcanzará el 16,1% del PIB en 2050.",
                        "evidence": [
                            {
                                "source_id": "S1",
                                "quote": "alcanzará el 16,1% del PIB en 2050",
                            }
                        ],
                    }
                ],
                "reason": "",
            },
            ensure_ascii=False,
        )
        return {
            "model": self.model,
            "done": True,
            "done_reason": "stop",
            "message": {"role": "assistant", "content": default_content},
            "prompt_eval_count": 140,
            "eval_count": 50,
            "eval_duration": 400000000,
            "total_duration": 850000000,
        }


class TestSemanticPipeline(unittest.TestCase):
    """Pruebas del flujo RAG semántico local con planificación, DAG y recuperación por necesidad."""

    def test_plan_simple_single_node_flow(self):
        """Verifica ejecución de un plan de 1 nodo simple sin dependencias ni resoluciones."""
        index = _make_semantic_test_index()
        embedder = SemanticMockEmbedder()
        generator = SemanticMockGenerator()

        report = answer_semantic(
            index=index,
            query="¿Cuál es el gasto proyectado para 2050?",
            top_k=2,
            embedder=embedder,
            generator=generator,
        )

        self.assertEqual(report["schema_version"], SCHEMA_VERSION)
        self.assertEqual(report["status"], "answered")
        self.assertEqual(len(report["claims"]), 1)
        self.assertEqual(report["reason"], "")
        self.assertEqual(len(report["parts"]), 1)

        p1 = report["parts"][0]
        self.assertEqual(p1["id"], "p1")
        self.assertEqual(p1["depends_on"], [])
        self.assertIsNone(p1["resolved_question"])
        self.assertEqual(p1["execution_status"], "executed")
        self.assertEqual(p1["status"], "answered")
        self.assertNotIn("resolution", p1)

        # Llamadas exactas
        self.assertEqual(report["api_calls"]["plan"], 1)
        self.assertEqual(report["api_calls"]["resolutions"], 0)
        self.assertEqual(report["api_calls"]["searches"], 1)
        self.assertEqual(report["api_calls"]["answers"], 1)
        self.assertEqual(report["api_calls"]["total"], 3)

        # Validación metodológica
        self.assertTrue(report["validation"]["citation_quotes_checked"])
        self.assertFalse(report["validation"]["semantic_faithfulness_evaluated"])
        self.assertFalse(report["validation"]["plan_semantics_verified"])

    def test_plan_compound_independent_nodes_different_queries_and_evidence(self):
        """Verifica que búsquedas reciben distintas consultas y evidencias A/B sustentan cada parte."""
        index = _make_semantic_test_index()
        embedder = SemanticMockEmbedder()

        plan_content = {
            "subquestions": [
                {
                    "id": "p1",
                    "question": "¿Cuál es la proyección de gasto en 2050?",
                    "required_evidence": "Gasto 2050.",
                    "depends_on": [],
                },
                {
                    "id": "p2",
                    "question": "¿Cuál es la proyección de gasto en 2070?",
                    "required_evidence": "Gasto 2070.",
                    "depends_on": [],
                },
            ]
        }
        ans_p1 = {
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
                                "text": "En 2050 el gasto alcanzará el 16,1% del PIB.",
                                "evidence": [
                                    {"source_id": "S1", "quote": "alcanzará el 16,1% del PIB en 2050"}
                                ],
                            }
                        ],
                        "reason": "",
                    }
                ),
            },
            "prompt_eval_count": 100,
            "eval_count": 30,
            "eval_duration": 200,
            "total_duration": 500,
        }
        ans_p2 = {
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
                                "text": "En 2070 el gasto se reducirá al 14,7% del PIB.",
                                "evidence": [
                                    {"source_id": "S2", "quote": "se reducirá hasta el 14,7% del PIB en 2070"}
                                ],
                            }
                        ],
                        "reason": "",
                    }
                ),
            },
            "prompt_eval_count": 110,
            "eval_count": 35,
            "eval_duration": 220,
            "total_duration": 520,
        }

        generator = SemanticMockGenerator(
            plan_content=plan_content,
            answer_responses=[ans_p1, ans_p2],
        )

        report = answer_semantic(
            index=index,
            query="Compara las proyecciones de gasto entre 2050 y 2070.",
            top_k=1,
            embedder=embedder,
            generator=generator,
        )

        self.assertEqual(report["status"], "answered")
        self.assertEqual(len(report["parts"]), 2)
        self.assertEqual(len(report["claims"]), 2)

        # Comprobación de que las búsquedas recibieron consultas distintas
        self.assertEqual(len(embedder.recorded_queries), 2)
        self.assertEqual(
            embedder.recorded_queries[0],
            "task: search result | query: ¿Cuál es la proyección de gasto en 2050?",
        )
        self.assertEqual(
            embedder.recorded_queries[1],
            "task: search result | query: ¿Cuál es la proyección de gasto en 2070?",
        )

        # Afirmaciones A y B sustentadas por fuentes distintas
        self.assertEqual(report["parts"][0]["sources_ids"], ["S1"])
        self.assertEqual(report["parts"][1]["sources_ids"], ["S2"])

    def test_dag_dependent_resolution_uses_parent_claims(self):
        """Verifica DAG de 2 nodos donde p2 depende de p1 y se resuelve usando hallazgos del padre."""
        index = _make_semantic_test_index()
        embedder = SemanticMockEmbedder()

        plan_content = {
            "subquestions": [
                {
                    "id": "p1",
                    "question": "¿Cuál es la proyección de gasto en 2050?",
                    "required_evidence": "Gasto 2050.",
                    "depends_on": [],
                },
                {
                    "id": "p2",
                    "question": "¿Cuál es la diferencia de gasto frente a 2070?",
                    "required_evidence": "Gasto 2070.",
                    "depends_on": ["p1"],
                },
            ]
        }
        resolve_content = {
            "question": "¿Cuál es el gasto en 2070 para contrastar con el 16,1% de 2050?"
        }
        ans_p1 = {
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
                                "text": "El gasto en 2050 es 16,1% del PIB.",
                                "evidence": [{"source_id": "S1", "quote": "alcanzará el 16,1% del PIB en 2050"}],
                            }
                        ],
                        "reason": "",
                    }
                ),
            },
            "prompt_eval_count": 100,
            "eval_count": 30,
            "eval_duration": 200,
            "total_duration": 500,
        }
        ans_p2 = {
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
                                "text": "El gasto en 2070 es 14,7% del PIB.",
                                "evidence": [{"source_id": "S2", "quote": "se reducirá hasta el 14,7% del PIB en 2070"}],
                            }
                        ],
                        "reason": "",
                    }
                ),
            },
            "prompt_eval_count": 100,
            "eval_count": 30,
            "eval_duration": 200,
            "total_duration": 500,
        }

        generator = SemanticMockGenerator(
            plan_content=plan_content,
            resolve_content=resolve_content,
            answer_responses=[ans_p1, ans_p2],
        )

        report = answer_semantic(
            index=index,
            query="Compara el gasto de 2050 frente a 2070.",
            top_k=1,
            embedder=embedder,
            generator=generator,
        )

        self.assertEqual(report["status"], "answered")
        self.assertEqual(report["api_calls"]["resolutions"], 1)
        self.assertIn("resolution", report["parts"][1])
        self.assertEqual(
            report["parts"][1]["resolved_question"],
            "¿Cuál es el gasto en 2070 para contrastar con el 16,1% de 2050?",
        )

        # Búsqueda para p2 usó la consulta resuelta, no la original
        self.assertEqual(
            embedder.recorded_queries[1],
            "task: search result | query: ¿Cuál es el gasto en 2070 para contrastar con el 16,1% de 2050?",
        )

        # Verificación de que la resolución recibió las afirmaciones y citas del padre
        deps_sent = generator.recorded_resolve_deps[0]
        self.assertEqual(len(deps_sent), 1)
        self.assertIn("16,1%", deps_sent[0]["hallazgos"][0]["afirmacion"])

    def test_sources_deduplication_and_score_tracking(self):
        """Verifica que fuentes compartidas conservan ID global S1..SN y registran retrievals por consulta."""
        index = _make_semantic_test_index()

        # Embedder que para ambas consultas devuelve vector que recupera c1 y c2
        class OverlapEmbedder(SemanticMockEmbedder):
            def embed_texts(self, texts: list[str]) -> list[list[float]]:
                # Para p1 devuelve [0.9, 0.4, 0], para p2 devuelve [0.4, 0.9, 0]
                if "p1" in texts[0] or "2050" in texts[0]:
                    v = [0.0] * DIMENSION
                    v[0] = 0.9
                    v[1] = 0.4
                    return [v]
                v = [0.0] * DIMENSION
                v[0] = 0.4
                v[1] = 0.9
                return [v]

        embedder = OverlapEmbedder()

        plan_content = {
            "subquestions": [
                {
                    "id": "p1",
                    "question": "¿Gasto en 2050?",
                    "required_evidence": "Evidencia 2050.",
                    "depends_on": [],
                },
                {
                    "id": "p2",
                    "question": "¿Gasto en 2070?",
                    "required_evidence": "Evidencia 2070.",
                    "depends_on": [],
                },
            ]
        }
        ans_p1 = {
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
                                "text": "Dato 2050.",
                                "evidence": [{"source_id": "S1", "quote": "alcanzará el 16,1% del PIB en 2050"}],
                            }
                        ],
                        "reason": "",
                    }
                ),
            },
            "prompt_eval_count": 100,
            "eval_count": 20,
            "eval_duration": 100,
            "total_duration": 200,
        }
        ans_p2 = {
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
                                "text": "Dato 2070.",
                                "evidence": [{"source_id": "S2", "quote": "se reducirá hasta el 14,7% del PIB en 2070"}],
                            }
                        ],
                        "reason": "",
                    }
                ),
            },
            "prompt_eval_count": 100,
            "eval_count": 20,
            "eval_duration": 100,
            "total_duration": 200,
        }

        generator = SemanticMockGenerator(plan_content=plan_content, answer_responses=[ans_p1, ans_p2])

        report = answer_semantic(
            index=index,
            query="Compara 2050 y 2070",
            top_k=2,
            embedder=embedder,
            generator=generator,
        )

        # En top_k=2 para ambas, c1 y c2 son recuperados en ambas
        # Fuentes globales deduplicadas: exactamente 2 fuentes S1 y S2
        self.assertEqual(len(report["sources"]), 2)
        s1 = report["sources"][0]
        s2 = report["sources"][1]
        self.assertEqual(s1["source_id"], "S1")
        self.assertEqual(s2["source_id"], "S2")

        # S1 y S2 tienen 2 entradas en retrievals (para p1 y p2 con sus respectivos rangos y scores)
        self.assertEqual(len(s1["retrievals"]), 2)
        self.assertEqual(len(s2["retrievals"]), 2)
        self.assertEqual(s1["retrievals"][0]["subquestion_id"], "p1")
        self.assertEqual(s1["retrievals"][1]["subquestion_id"], "p2")

    def test_cross_node_citation_rejected(self):
        """Verifica que un nodo no puede citar una fuente que no fue recuperada en su propia búsqueda."""
        index = _make_semantic_test_index()
        embedder = SemanticMockEmbedder()

        plan_content = {
            "subquestions": [
                {
                    "id": "p1",
                    "question": "¿Gasto 2050?",
                    "required_evidence": "Evidencia 2050.",
                    "depends_on": [],
                },
                {
                    "id": "p2",
                    "question": "¿Gasto 2070?",
                    "required_evidence": "Evidencia 2070.",
                    "depends_on": [],
                },
            ]
        }
        ans_p1 = {
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
                                "text": "Dato 2050.",
                                "evidence": [{"source_id": "S1", "quote": "alcanzará el 16,1% del PIB en 2050"}],
                            }
                        ],
                        "reason": "",
                    }
                ),
            },
            "prompt_eval_count": 100,
            "eval_count": 20,
            "eval_duration": 100,
            "total_duration": 200,
        }
        # ans_p2 intenta citar S1, pero en su búsqueda top-k=1 solo recupera c2 (que es S2)
        ans_p2_illegal = {
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
                                "text": "Dato 2070.",
                                "evidence": [{"source_id": "S1", "quote": "alcanzará el 16,1% del PIB en 2050"}],
                            }
                        ],
                        "reason": "",
                    }
                ),
            },
            "prompt_eval_count": 100,
            "eval_count": 20,
            "eval_duration": 100,
            "total_duration": 200,
        }

        generator = SemanticMockGenerator(
            plan_content=plan_content,
            answer_responses=[ans_p1, ans_p2_illegal],
        )

        with self.assertRaises(ValueError) as ctx:
            answer_semantic(
                index=index,
                query="Compara 2050 y 2070",
                top_k=1,
                embedder=embedder,
                generator=generator,
            )
        self.assertIn("Unknown or invalid source_id", str(ctx.exception))

    def test_insufficient_parent_blocks_dependent_child_without_calls(self):
        """Verifica que un padre con status != answered bloquea al hijo sin buscar ni generar para él."""
        index = _make_semantic_test_index()
        embedder = SemanticMockEmbedder()

        plan_content = {
            "subquestions": [
                {
                    "id": "p1",
                    "question": "¿Gasto en 2050?",
                    "required_evidence": "Evidencia 2050.",
                    "depends_on": [],
                },
                {
                    "id": "p2",
                    "question": "¿Impacto de dicho gasto en 2050?",
                    "required_evidence": "Impacto 2050.",
                    "depends_on": ["p1"],
                },
            ]
        }
        ans_p1_insufficient = {
            "model": "qwen3.5:9b",
            "done": True,
            "done_reason": "stop",
            "message": {
                "role": "assistant",
                "content": json.dumps(
                    {
                        "status": "insufficient_evidence",
                        "claims": [],
                        "reason": "No hay evidencia suficiente sobre el gasto.",
                    }
                ),
            },
            "prompt_eval_count": 100,
            "eval_count": 20,
            "eval_duration": 100,
            "total_duration": 200,
        }

        generator = SemanticMockGenerator(
            plan_content=plan_content,
            answer_responses=[ans_p1_insufficient],
        )

        report = answer_semantic(
            index=index,
            query="Gasto en 2050 y su impacto",
            top_k=1,
            embedder=embedder,
            generator=generator,
        )

        self.assertEqual(report["status"], "insufficient_evidence")
        self.assertEqual(report["claims"], [])
        self.assertIn("Dependencia bloqueada", report["reason"])

        # p2 registrado como blocked_dependency
        p2 = report["parts"][1]
        self.assertEqual(p2["execution_status"], "blocked_dependency")
        self.assertEqual(p2["status"], "insufficient_evidence")
        self.assertIsNone(p2["resolved_question"])
        self.assertIsNone(p2["answer_prompt_sha256"])

        # Llamadas exactas: solo 1 búsqueda (para p1), 0 resoluciones, 1 generación (para p1)
        self.assertEqual(report["api_calls"]["plan"], 1)
        self.assertEqual(report["api_calls"]["resolutions"], 0)
        self.assertEqual(report["api_calls"]["searches"], 1)
        self.assertEqual(report["api_calls"]["answers"], 1)
        self.assertEqual(report["api_calls"]["total"], 3)

        # Hash canónico tiene null para el nodo bloqueado
        self.assertIn(None, report["prompt_hashes"])

    def test_pin_model_digest_mismatch_aborts_all(self):
        """Verifica que si el digest del modelo cambia durante el flujo se aborta inmediatamente."""
        index = _make_semantic_test_index()
        embedder = SemanticMockEmbedder()
        # Mutar el digest después de la llamada al plan
        generator = SemanticMockGenerator(mutate_digest_at_call=3)

        with self.assertRaises(RuntimeError) as ctx:
            answer_semantic(
                index=index,
                query="¿Cuál es el gasto en 2050?",
                embedder=embedder,
                generator=generator,
            )
        self.assertIn("digest changed", str(ctx.exception))

    def test_pre_network_validations(self):
        """Verifica rechazo inmediato antes de red por query vacía, top_k inválido o modelo incorrecto."""
        index = _make_semantic_test_index()
        embedder = SemanticMockEmbedder()
        generator = SemanticMockGenerator()

        # Query vacía
        with self.assertRaises(ValueError):
            answer_semantic(index=index, query="   ", embedder=embedder, generator=generator)

        # Query > 1000 chars
        with self.assertRaises(ValueError):
            answer_semantic(index=index, query="q" * 1001, embedder=embedder, generator=generator)

        # Query boolean
        with self.assertRaises(ValueError):
            answer_semantic(index=index, query=True, embedder=embedder, generator=generator)  # type: ignore

        # top_k booleano o <= 0
        with self.assertRaises(ValueError):
            answer_semantic(index=index, query="q", top_k=True, embedder=embedder, generator=generator)  # type: ignore
        with self.assertRaises(ValueError):
            answer_semantic(index=index, query="q", top_k=0, embedder=embedder, generator=generator)

        # Modelo no autorizado
        with self.assertRaises(ValueError):
            answer_semantic(index=index, query="q", model="unauthorized", embedder=embedder, generator=generator)

        # Modelo del generador no coincide
        with self.assertRaises(ValueError):
            answer_semantic(index=index, query="q", model="qwen3.5:4b-q4_K_M", embedder=embedder, generator=generator)

        # Sin llamadas al generador ni embedder
        self.assertEqual(len(generator.calls), 0)
        self.assertEqual(len(embedder.recorded_queries), 0)

    def test_invalid_plan_aborts_before_embedding(self):
        """Verifica que un plan inválido emitido por el modelo aborta antes de realizar búsquedas."""
        index = _make_semantic_test_index()
        embedder = SemanticMockEmbedder()

        # Plan con año no autorizado (2099)
        bad_plan = {
            "subquestions": [
                {
                    "id": "p1",
                    "question": "¿Cuál es el gasto en 2099?",
                    "required_evidence": "Gasto 2099.",
                    "depends_on": [],
                }
            ]
        }
        generator = SemanticMockGenerator(plan_content=bad_plan)

        with self.assertRaises(ValueError) as ctx:
            answer_semantic(
                index=index,
                query="¿Gasto en 2050?",
                embedder=embedder,
                generator=generator,
            )
        self.assertIn("unauthorized years", str(ctx.exception))
        # Ninguna búsqueda ejecutada
        self.assertEqual(len(embedder.recorded_queries), 0)

    def test_purity_no_vectors_in_report_nor_index_mutations(self):
        """Verifica pureza: no se exponen vectores crudos y el índice permanece inmutable."""
        index = _make_semantic_test_index()
        index_copy = copy.deepcopy(index)
        embedder = SemanticMockEmbedder()
        generator = SemanticMockGenerator()

        report = answer_semantic(
            index=index,
            query="¿Cuál es el gasto proyectado para 2050?",
            embedder=embedder,
            generator=generator,
        )

        # Índice idéntico sin mutación
        self.assertEqual(index, index_copy)

        # Fuentes sin campo embedding
        for src in report["sources"]:
            self.assertNotIn("embedding", src)
            self.assertNotIn("vector", src)


class TestSemanticCLIAndBaseline(unittest.TestCase):
    """Pruebas de la CLI de rag_pensiones.semantic y preservación del baseline answer.py."""

    def test_cli_text_output_answered(self):
        """Verifica salida en modo texto de la CLI con plan, desglose y fuentes."""
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as tmpdir:
            idx_file = Path(tmpdir) / "idx.json"
            idx_file.write_text(json.dumps(_make_semantic_test_index(), ensure_ascii=False), encoding="utf-8")

            fake_embedder = SemanticMockEmbedder()
            fake_generator = SemanticMockGenerator()

            with patch("rag_pensiones.semantic.OllamaEmbedder", return_value=fake_embedder), patch(
                "rag_pensiones.semantic.OllamaGenerator", return_value=fake_generator
            ):
                out_buf = io.StringIO()
                err_buf = io.StringIO()
                with patch("sys.stdout", out_buf), patch("sys.stderr", err_buf):
                    code = main(["--query", "¿Cuál es el gasto en 2050?", "--index", str(idx_file)])

                self.assertEqual(code, 0)
                self.assertEqual(err_buf.getvalue(), "")
                out = out_buf.getvalue()
                self.assertIn("RESPUESTA RAG SEMÁNTICA — PENSIONES AIREF 2025", out)
                self.assertIn("PLAN SEMÁNTICO:", out)
                self.assertIn("DESGLOSE POR NODO:", out)
                self.assertIn("FUENTES RECUPERADAS (DEDUPLICADAS):", out)
                self.assertIn("NOTA METODOLÓGICA:", out)

    def test_cli_json_output(self):
        """Verifica salida JSON parseable y conforme a schema_version rag-semantic-v1."""
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as tmpdir:
            idx_file = Path(tmpdir) / "idx.json"
            idx_file.write_text(json.dumps(_make_semantic_test_index(), ensure_ascii=False), encoding="utf-8")

            fake_embedder = SemanticMockEmbedder()
            fake_generator = SemanticMockGenerator()

            with patch("rag_pensiones.semantic.OllamaEmbedder", return_value=fake_embedder), patch(
                "rag_pensiones.semantic.OllamaGenerator", return_value=fake_generator
            ):
                out_buf = io.StringIO()
                err_buf = io.StringIO()
                with patch("sys.stdout", out_buf), patch("sys.stderr", err_buf):
                    code = main(["--query", "¿Cuál es el gasto en 2050?", "--index", str(idx_file), "--json"])

                self.assertEqual(code, 0)
                self.assertEqual(err_buf.getvalue(), "")
                parsed = json.loads(out_buf.getvalue())
                self.assertEqual(parsed["schema_version"], "rag-semantic-v1")
                self.assertEqual(parsed["status"], "answered")
                self.assertIn("plan", parsed)
                self.assertIn("parts", parsed)
                self.assertIn("sources", parsed)

    def test_cli_technical_error_stderr_no_traceback(self):
        """Verifica que errores técnicos escriben a stderr sin volcar traza de traceback."""
        err_buf = io.StringIO()
        out_buf = io.StringIO()
        with patch("sys.stdout", out_buf), patch("sys.stderr", err_buf):
            code = main(["--query", "   "])
        self.assertNotEqual(code, 0)
        self.assertIn("Error:", err_buf.getvalue())
        self.assertNotIn("Traceback", err_buf.getvalue())

    def test_baseline_answer_intact(self):
        """Verifica que el baseline answer.py sigue funcionando intacto sin afectación."""
        index = _make_semantic_test_index()
        embedder = SemanticMockEmbedder()
        generator = SemanticMockGenerator()

        report = answer(
            index=index,
            query="¿Cuál es la proyección de gasto en 2050?",
            top_k=1,
            embedder=embedder,  # type: ignore
            generator=generator,  # type: ignore
        )

        self.assertEqual(report["schema_version"], "rag-answer-v1")
        self.assertEqual(report["status"], "answered")


VALID_SHA256_HEX = "a" * 64


def _make_injected_plan_report(**overrides) -> dict:
    """Construye un reporte canónico válido de planner.plan() permitiendo mutaciones quirúrgicas."""
    base: dict = {
        "plan": {
            "subquestions": [
                {
                    "id": "p1",
                    "question": "¿Cuál es el gasto en 2050?",
                    "required_evidence": "Gasto 2050.",
                    "depends_on": [],
                }
            ]
        },
        "model": DEFAULT_GENERATION_MODEL,
        "model_digest": "test_gen_digest_sha256",
        "prompt_sha256": VALID_SHA256_HEX,
        "stats": {
            "prompt_eval_count": 100,
            "eval_count": 30,
            "eval_duration": 200,
            "total_duration": 500,
        },
        "parameters": {"temperature": 0, "seed": 42},
    }
    for k, v in overrides.items():
        if v is None and k in base:
            del base[k]
        else:
            base[k] = v
    return base


class InjectedPlanMockPlanner:
    """Mock determinista de planner inyectado para validar defensas en frontera sobre plan y resolución."""

    def __init__(
        self,
        plan_report: dict,
        resolve_report: dict | None = None,
        model: str = DEFAULT_GENERATION_MODEL,
    ) -> None:
        self.plan_report = plan_report
        self.resolve_report = resolve_report
        self.model = model
        self.generator = None

    def plan(self, query: str, scope_years: set[str] | None = None) -> dict:
        return copy.deepcopy(self.plan_report)

    def resolve(self, *args, **kwargs) -> dict:
        if self.resolve_report is not None:
            return copy.deepcopy(self.resolve_report)
        raise NotImplementedError("Resolve no invocado en pruebas exclusivas de plan")


def _make_injected_2node_plan_report(**overrides) -> dict:
    """Construye un reporte válido de 2 nodos p1->p2 permitiendo mutaciones."""
    return _make_injected_plan_report(
        plan={
            "subquestions": [
                {
                    "id": "p1",
                    "question": "¿Cuál es la proyección de gasto en 2050?",
                    "required_evidence": "Gasto 2050.",
                    "depends_on": [],
                },
                {
                    "id": "p2",
                    "question": "¿Cuál es el impacto del gasto en 2050?",
                    "required_evidence": "Impacto 2050.",
                    "depends_on": ["p1"],
                },
            ]
        },
        **overrides,
    )


def _make_injected_resolve_report(**overrides) -> dict:
    """Construye un reporte canónico válido de planner.resolve() permitiendo mutaciones quirúrgicas."""
    base: dict = {
        "question": "¿Cuál es el impacto del gasto en 2050 según los hallazgos?",
        "model": DEFAULT_GENERATION_MODEL,
        "model_digest": "test_gen_digest_sha256",
        "prompt_sha256": VALID_SHA256_HEX,
        "stats": {
            "prompt_eval_count": 110,
            "eval_count": 35,
            "eval_duration": 220,
            "total_duration": 520,
        },
        "parameters": {"temperature": 0, "seed": 42},
    }
    for k, v in overrides.items():
        if v is None and k in base:
            del base[k]
        else:
            base[k] = v
    return base


class TestInjectedPlannerPlanValidation(unittest.TestCase):
    """Regresiones en frontera para planner inyectado: validación estricta de plan antes de búsqueda/generación."""

    def setUp(self) -> None:
        self.index = _make_semantic_test_index()

    def test_injected_planner_plan_stats_anomalies(self) -> None:
        """Rechaza stats de plan anómalas (NaN, Inf, bool, negativo, faltante, >6912, extra) sin buscar ni generar."""
        bad_cases = [
            ("nan", {"prompt_eval_count": float("nan"), "eval_count": 30, "eval_duration": 200, "total_duration": 500}),
            ("inf", {"prompt_eval_count": float("inf"), "eval_count": 30, "eval_duration": 200, "total_duration": 500}),
            ("bool", {"prompt_eval_count": True, "eval_count": 30, "eval_duration": 200, "total_duration": 500}),
            ("negativo", {"prompt_eval_count": -1, "eval_count": 30, "eval_duration": 200, "total_duration": 500}),
            ("faltante", {"eval_count": 30, "eval_duration": 200, "total_duration": 500}),
            ("prompt_eval_count6913", {"prompt_eval_count": 6913, "eval_count": 30, "eval_duration": 200, "total_duration": 500}),
            ("extra_no_finito", {"prompt_eval_count": 100, "eval_count": 30, "eval_duration": 200, "total_duration": 500, "extra_inf": float("inf")}),
            ("extra_campo", {"prompt_eval_count": 100, "eval_count": 30, "eval_duration": 200, "total_duration": 500, "extra_stat": 42}),
        ]
        for name, bad_stats in bad_cases:
            with self.subTest(case=name):
                embedder = SemanticMockEmbedder()
                generator = SemanticMockGenerator()
                planner = InjectedPlanMockPlanner(_make_injected_plan_report(stats=bad_stats))
                with self.assertRaises(ValueError):
                    answer_semantic(self.index, "¿Gasto en 2050?", embedder=embedder, generator=generator, planner=planner)
                self.assertEqual(len(embedder.recorded_queries), 0)
                self.assertEqual(len(generator.recorded_generate_queries), 0)

    def test_injected_planner_plan_hash_anomalies(self) -> None:
        """Rechaza prompt_sha256 vacío, no hex o con longitud incorrecta sin contactar buscador ni generador."""
        bad_hashes = [
            ("vacio", ""),
            ("whitespace", "   "),
            ("no_hex", "g" * 64),
            ("longitud_63", "a" * 63),
            ("longitud_65", "a" * 65),
            ("no_str", 1234567890),
        ]
        for name, bad_hash in bad_hashes:
            with self.subTest(case=name):
                embedder = SemanticMockEmbedder()
                generator = SemanticMockGenerator()
                planner = InjectedPlanMockPlanner(_make_injected_plan_report(prompt_sha256=bad_hash))
                with self.assertRaises(ValueError):
                    answer_semantic(self.index, "¿Gasto en 2050?", embedder=embedder, generator=generator, planner=planner)
                self.assertEqual(len(embedder.recorded_queries), 0)
                self.assertEqual(len(generator.recorded_generate_queries), 0)

    def test_injected_planner_plan_parameters_anomalies(self) -> None:
        """Rechaza parameters no serializables en JSON estricto (NaN, Inf, no dict) antes de búsqueda."""
        bad_params_cases = [
            ("nan", {"temperature": float("nan")}),
            ("inf", {"temperature": float("inf")}),
            ("no_dict_int", 123),
            ("no_dict_bool", True),
            ("no_dict_str", "temp=0"),
            ("no_dict_none", None),
        ]
        for name, bad_params in bad_params_cases:
            with self.subTest(case=name):
                embedder = SemanticMockEmbedder()
                generator = SemanticMockGenerator()
                planner = InjectedPlanMockPlanner(_make_injected_plan_report(parameters=bad_params))
                with self.assertRaises(ValueError):
                    answer_semantic(self.index, "¿Gasto en 2050?", embedder=embedder, generator=generator, planner=planner)
                self.assertEqual(len(embedder.recorded_queries), 0)
                self.assertEqual(len(generator.recorded_generate_queries), 0)

    def test_injected_planner_plan_valid_control(self) -> None:
        """Control positivo: plan válido inyectado ejecuta y preserva contadores, stats, hash y parámetros."""
        embedder = SemanticMockEmbedder()
        generator = SemanticMockGenerator()
        valid_report = _make_injected_plan_report()
        planner = InjectedPlanMockPlanner(valid_report)

        res = answer_semantic(self.index, "¿Gasto en 2050?", embedder=embedder, generator=generator, planner=planner)

        self.assertEqual(res["status"], "answered")
        self.assertEqual(res["plan"]["prompt_sha256"], VALID_SHA256_HEX)
        self.assertEqual(res["plan"]["stats"], valid_report["stats"])
        self.assertEqual(res["plan"]["parameters"], valid_report["parameters"])
        self.assertEqual(len(embedder.recorded_queries), 1)
        self.assertEqual(len(generator.recorded_generate_queries), 1)


class TestInjectedPlannerResolveValidation(unittest.TestCase):
    """Regresiones en frontera para planner inyectado: validación estricta de resolve antes de búsqueda del hijo."""

    def setUp(self) -> None:
        self.index = _make_semantic_test_index()

    def test_injected_planner_resolve_stats_anomalies(self) -> None:
        """Rechaza stats de resolución anómalas (NaN, Inf, bool, negativo, faltante, >6912, extra) sin buscar el hijo."""
        bad_cases = [
            ("nan", {"prompt_eval_count": 100, "eval_count": float("nan"), "eval_duration": 200, "total_duration": 500}),
            ("inf", {"prompt_eval_count": 100, "eval_count": 30, "eval_duration": 200, "total_duration": float("inf")}),
            ("bool", {"prompt_eval_count": 100, "eval_count": 30, "eval_duration": True, "total_duration": 500}),
            ("negativo", {"prompt_eval_count": -5, "eval_count": 30, "eval_duration": 200, "total_duration": 500}),
            ("faltante", {"prompt_eval_count": 100, "eval_count": 30, "eval_duration": 200}),
            ("prompt_eval_count6913", {"prompt_eval_count": 6913, "eval_count": 30, "eval_duration": 200, "total_duration": 500}),
            ("extra_no_finito", {"prompt_eval_count": 100, "eval_count": 30, "eval_duration": 200, "total_duration": 500, "extra_nan": float("nan")}),
        ]
        for name, bad_stats in bad_cases:
            with self.subTest(case=name):
                embedder = SemanticMockEmbedder()
                generator = SemanticMockGenerator()
                planner = InjectedPlanMockPlanner(
                    plan_report=_make_injected_2node_plan_report(),
                    resolve_report=_make_injected_resolve_report(stats=bad_stats),
                )
                with self.assertRaises(ValueError):
                    answer_semantic(self.index, "¿Cuál es la proyección de gasto en 2050 y su impacto?", embedder=embedder, generator=generator, planner=planner)
                self.assertEqual(len(embedder.recorded_queries), 1)
                self.assertEqual(len(generator.recorded_generate_queries), 1)

    def test_injected_planner_resolve_hash_anomalies(self) -> None:
        """Rechaza prompt_sha256 de resolve vacío, no hex o con longitud incorrecta sin buscar el hijo."""
        bad_hashes = [
            ("vacio", ""),
            ("whitespace", "   "),
            ("no_hex", "z" * 64),
            ("longitud_63", "a" * 63),
            ("longitud_65", "a" * 65),
            ("no_str", False),
        ]
        for name, bad_hash in bad_hashes:
            with self.subTest(case=name):
                embedder = SemanticMockEmbedder()
                generator = SemanticMockGenerator()
                planner = InjectedPlanMockPlanner(
                    plan_report=_make_injected_2node_plan_report(),
                    resolve_report=_make_injected_resolve_report(prompt_sha256=bad_hash),
                )
                with self.assertRaises(ValueError):
                    answer_semantic(self.index, "¿Cuál es la proyección de gasto en 2050 y su impacto?", embedder=embedder, generator=generator, planner=planner)
                self.assertEqual(len(embedder.recorded_queries), 1)
                self.assertEqual(len(generator.recorded_generate_queries), 1)

    def test_injected_planner_resolve_parameters_anomalies(self) -> None:
        """Rechaza parameters de resolve no serializables en JSON estricto (NaN, Inf, no dict)."""
        bad_params_cases = [
            ("nan", {"temperature": float("nan")}),
            ("inf", {"temperature": float("inf")}),
            ("no_dict_int", 123),
            ("no_dict_bool", True),
            ("no_dict_none", None),
        ]
        for name, bad_params in bad_params_cases:
            with self.subTest(case=name):
                embedder = SemanticMockEmbedder()
                generator = SemanticMockGenerator()
                planner = InjectedPlanMockPlanner(
                    plan_report=_make_injected_2node_plan_report(),
                    resolve_report=_make_injected_resolve_report(parameters=bad_params),
                )
                with self.assertRaises(ValueError):
                    answer_semantic(self.index, "¿Cuál es la proyección de gasto en 2050 y su impacto?", embedder=embedder, generator=generator, planner=planner)
                self.assertEqual(len(embedder.recorded_queries), 1)
                self.assertEqual(len(generator.recorded_generate_queries), 1)

    def test_injected_planner_resolve_question_anomalies(self) -> None:
        """Rechaza question de resolve inválida (vacía, tipos, 601 chars, omite 2050, añade 2070 no autorizado)."""
        bad_questions = [
            ("vacio", ""),
            ("solo_espacios", "   \t\n  "),
            ("tipo_no_str_int", 123),
            ("tipo_no_str_none", None),
            ("601chars", "¿" + "x" * 599 + "?"),
            ("omitir_2050", "¿Cuál es el impacto del gasto según los hallazgos?"),
            ("anadir_2070_no_permitido", "¿Cuál es el impacto del gasto en 2050 comparado con 2070?"),
        ]
        for name, bad_q in bad_questions:
            with self.subTest(case=name):
                embedder = SemanticMockEmbedder()
                generator = SemanticMockGenerator()
                planner = InjectedPlanMockPlanner(
                    plan_report=_make_injected_2node_plan_report(),
                    resolve_report=_make_injected_resolve_report(question=bad_q),
                )
                with self.assertRaises(ValueError):
                    answer_semantic(self.index, "¿Cuál es la proyección de gasto en 2050 y su impacto?", embedder=embedder, generator=generator, planner=planner)
                self.assertEqual(len(embedder.recorded_queries), 1)
                self.assertEqual(len(generator.recorded_generate_queries), 1)

    def test_injected_planner_resolve_valid_control(self) -> None:
        """Control positivo: trim preservado y año legítimo derivado de claims del padre aceptado en la búsqueda del hijo."""
        embedder = SemanticMockEmbedder()
        plan_report = _make_injected_plan_report(
            plan={
                "subquestions": [
                    {"id": "p1", "question": "¿Cuál es el año clave?", "required_evidence": "Año clave.", "depends_on": []},
                    {"id": "p2", "question": "¿Qué gasto supone dicho año?", "required_evidence": "Gasto dicho año.", "depends_on": ["p1"]},
                ]
            }
        )
        resolve_report = _make_injected_resolve_report(
            question="   ¿Qué gasto supone el año 2050 según el informe?   "
        )
        ans_p1 = {
            "model": DEFAULT_GENERATION_MODEL,
            "done": True,
            "done_reason": "stop",
            "message": {
                "role": "assistant",
                "content": json.dumps({
                    "status": "answered",
                    "claims": [{"text": "El año clave es 2050.", "evidence": [{"source_id": "S1", "quote": "alcanzará el 16,1% del PIB en 2050"}]}],
                    "reason": "",
                }),
            },
            "prompt_eval_count": 100,
            "eval_count": 25,
            "eval_duration": 150,
            "total_duration": 400,
        }
        ans_p2 = {
            "model": DEFAULT_GENERATION_MODEL,
            "done": True,
            "done_reason": "stop",
            "message": {
                "role": "assistant",
                "content": json.dumps({
                    "status": "answered",
                    "claims": [{"text": "En 2050 alcanzará el 16,1% del PIB.", "evidence": [{"source_id": "S1", "quote": "alcanzará el 16,1% del PIB en 2050"}]}],
                    "reason": "",
                }),
            },
            "prompt_eval_count": 110,
            "eval_count": 30,
            "eval_duration": 180,
            "total_duration": 450,
        }
        generator = SemanticMockGenerator(answer_responses=[ans_p1, ans_p2])
        planner = InjectedPlanMockPlanner(plan_report=plan_report, resolve_report=resolve_report)

        report = answer_semantic(
            self.index,
            "¿Cuál es el año de referencia y qué gasto supone?",
            top_k=1,
            embedder=embedder,
            generator=generator,
            planner=planner,
        )

        self.assertEqual(report["status"], "answered")
        self.assertEqual(report["api_calls"]["resolutions"], 1)
        self.assertEqual(report["parts"][1]["resolved_question"], "¿Qué gasto supone el año 2050 según el informe?")
        self.assertEqual(len(embedder.recorded_queries), 2)
        self.assertEqual(
            embedder.recorded_queries[1],
            "task: search result | query: ¿Qué gasto supone el año 2050 según el informe?",
        )
        self.assertEqual(len(generator.recorded_generate_queries), 2)
        self.assertEqual(report["parts"][1]["sources_ids"], ["S1"])


if __name__ == "__main__":
    unittest.main()
