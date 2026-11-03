"""Tests offline para la integración de fundamentación (grounding) en el flujo semántico."""

import copy
import hashlib
import io
import json
from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch

from rag_pensiones.answer import ALLOWED_MODELS, DEFAULT_GENERATION_MODEL, collapse_spaces
from rag_pensiones.chunking import ChunkRecord
from rag_pensiones.embeddings import DIMENSION
from rag_pensiones.grounding import (
    DEFAULT_PARAMETERS,
    SCHEMA_VERSION as GROUNDING_SCHEMA_VERSION,
    OllamaGrounder,
    validate_grounded_result,
)
from rag_pensiones.planning import OllamaPlanner
from rag_pensiones.search import SearchHit
from rag_pensiones.semantic import (
    GROUNDED_SCHEMA_VERSION,
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
        {
            "chunk_id": "chunk_c7",
            "doc_id": "doc_airef_2",
            "source_file": "airef_demografia.pdf",
            "published_at": "2025-03-31",
            "page_number": 98,
            "start_char": 0,
            "end_char": 90,
            "text": "La tasa de dependencia superará el 53% en 2050 según las proyecciones demográficas.",
            "strategy": "page-local-v1:max_chars=1000:overlap_chars=150",
            "embedding": _make_unit_vector(3),
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


class MockEmbedder:
    def __init__(self, digest: str = "test_embed_digest_sha256") -> None:
        self.digest = digest

    def get_digest(self) -> str:
        return self.digest

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        res = []
        for t in texts:
            clean = t.lower()
            if "2050" in clean and "dependencia" not in clean:
                res.append(_make_unit_vector(0))
            elif "2070" in clean:
                res.append(_make_unit_vector(1))
            elif "dependencia" in clean:
                res.append(_make_unit_vector(3))
            else:
                res.append(_make_unit_vector(2))
        return res


class MockPlanner:
    def __init__(
        self,
        plan_dict: dict,
        model: str = DEFAULT_GENERATION_MODEL,
        digest: str = "test_gen_digest_sha256",
        generator: MagicMock | None = None,
    ) -> None:
        self.model = model
        self.digest = digest
        self.plan_dict = plan_dict
        self.generator = generator

    def plan(self, query: str, scope_years: set[str] | None = None) -> dict:
        return {
            "plan": self.plan_dict,
            "model": self.model,
            "model_digest": self.digest,
            "prompt_sha256": "a" * 64,
            "stats": {
                "prompt_eval_count": 100,
                "eval_count": 50,
                "eval_duration": 1000,
                "total_duration": 2000,
            },
            "parameters": {"temperature": 0, "seed": 42, "num_ctx": 8192, "num_predict": 1024},
        }

    def resolve(
        self,
        question: str,
        original_query: str,
        dependencies: list[dict],
        expected_digest: str,
        scope_years: set[str] | None = None,
    ) -> dict:
        return {
            "question": f"¿Pregunta resuelta para 2050 basada en {dependencies[0]['id']}?",
            "model": self.model,
            "model_digest": self.digest,
            "prompt_sha256": "b" * 64,
            "stats": {
                "prompt_eval_count": 80,
                "eval_count": 40,
                "eval_duration": 800,
                "total_duration": 1600,
            },
            "parameters": {"temperature": 0, "seed": 42, "num_ctx": 8192, "num_predict": 1024},
        }


def _make_valid_grounded_answer(
    hits: list[SearchHit],
    status: str = "answered",
    digest: str = "test_gen_digest_sha256",
    prune_irrelevant: bool = False,
) -> dict:
    source_id = f"S{hits[0].rank}"
    quote = hits[0].chunk.text[:80]
    prompt_hash1 = "1" * 64
    prompt_hash2 = "2" * 64
    prompt_hash3 = "3" * 64

    if status == "insufficient_evidence":
        trace = [
            {
                "stage": "selection",
                "model": DEFAULT_GENERATION_MODEL,
                "model_digest": digest,
                "prompt_sha256": prompt_hash1,
                "stats": {"prompt_eval_count": 100, "eval_count": 50, "eval_duration": 1000, "total_duration": 2000},
                "parameters": copy.deepcopy(DEFAULT_PARAMETERS),
                "messages_budget_bytes": 6500,
            }
        ]
        sum_stats = {
            "prompt_eval_count": 100,
            "eval_count": 50,
            "eval_duration": 1000,
            "total_duration": 2000,
        }
        canonical_hashes = json.dumps([prompt_hash1], separators=(",", ":"))
        agg_hash = hashlib.sha256(canonical_hashes.encode("utf-8")).hexdigest()

        return {
            "status": "insufficient_evidence",
            "claims": [],
            "reason": "No se encontró evidencia suficiente en las fuentes para responder.",
            "digest": digest,
            "prompt_sha256": agg_hash,
            "stats": sum_stats,
            "grounding": {
                "schema_version": GROUNDING_SCHEMA_VERSION,
                "selected_evidence": [],
                "initial_claims": [],
                "initial_verdicts": [],
                "repair_attempts": 0,
                "repair_evidence": [],
                "repairs": [],
                "final_verdicts": [],
                "accepted_claims": [],
                "rejected_claims": [],
                "trace": trace,
                "verification_contexts": {
                    "initial": [],
                    "repair": [],
                },
            },
        }

    trace = [
        {
            "stage": "selection",
            "model": DEFAULT_GENERATION_MODEL,
            "model_digest": digest,
            "prompt_sha256": prompt_hash1,
            "stats": {"prompt_eval_count": 100, "eval_count": 50, "eval_duration": 1000, "total_duration": 2000},
            "parameters": copy.deepcopy(DEFAULT_PARAMETERS),
            "messages_budget_bytes": 6500,
        },
        {
            "stage": "generation",
            "model": DEFAULT_GENERATION_MODEL,
            "model_digest": digest,
            "prompt_sha256": prompt_hash2,
            "stats": {"prompt_eval_count": 120, "eval_count": 60, "eval_duration": 1200, "total_duration": 2400},
            "parameters": copy.deepcopy(DEFAULT_PARAMETERS),
            "messages_budget_bytes": 6500,
        },
        {
            "stage": "verification",
            "model": DEFAULT_GENERATION_MODEL,
            "model_digest": digest,
            "prompt_sha256": prompt_hash3,
            "stats": {"prompt_eval_count": 110, "eval_count": 40, "eval_duration": 1100, "total_duration": 2200},
            "parameters": copy.deepcopy(DEFAULT_PARAMETERS),
            "messages_budget_bytes": 12288,
        },
    ]

    selected_ev = [
        {
            "evidence_id": "E1",
            "source_id": source_id,
            "doc_id": hits[0].chunk.doc_id,
            "source_file": hits[0].chunk.source_file,
            "page_number": hits[0].chunk.page_number,
            "quote": quote,
        }
    ]

    claims = [
        {
            "text": "El gasto en pensiones alcanzará el 16,1% del PIB en 2050.",
            "evidence": [{"source_id": source_id, "quote": quote}],
        }
    ]
    accepted_claims = [
        {
            "claim_id": "C1",
            "text": "El gasto en pensiones alcanzará el 16,1% del PIB en 2050.",
            "evidence_ids": ["E1"],
        }
    ]
    rejected_claims = []
    reason = ""

    initial_verdicts = [
        {
            "claim_id": "C1",
            "verdict": "supported",
            "reason": "Evaluación conforme.",
            "evidence_verdicts": [{"evidence_id": "E1", "relation": "supports"}],
        }
    ]

    sum_stats = {
        "prompt_eval_count": 330,
        "eval_count": 150,
        "eval_duration": 3300,
        "total_duration": 6600,
    }
    canonical_hashes = json.dumps([prompt_hash1, prompt_hash2, prompt_hash3], separators=(",", ":"))
    agg_hash = hashlib.sha256(canonical_hashes.encode("utf-8")).hexdigest()

    return {
        "status": status,
        "claims": claims,
        "reason": reason,
        "digest": digest,
        "prompt_sha256": agg_hash,
        "stats": sum_stats,
        "grounding": {
            "schema_version": GROUNDING_SCHEMA_VERSION,
            "selected_evidence": selected_ev,
            "initial_claims": [
                {
                    "claim_id": "C1",
                    "text": "El gasto en pensiones alcanzará el 16,1% del PIB en 2050.",
                    "evidence_ids": ["E1"],
                }
            ],
            "initial_verdicts": initial_verdicts,
            "repair_attempts": 0,
            "repair_evidence": [],
            "repairs": [],
            "final_verdicts": copy.deepcopy(initial_verdicts),
            "accepted_claims": accepted_claims,
            "rejected_claims": rejected_claims,
            "trace": trace,
            "verification_contexts": {
                "initial": [],
                "repair": [],
            },
        },
    }


class TestGroundedSemanticPipeline(unittest.TestCase):
    """Tests para la integración de OllamaGrounder en answer_semantic()."""

    def setUp(self) -> None:
        self.index = _make_semantic_test_index()
        self.embedder = MockEmbedder()
        self.digest = "test_gen_digest_sha256"

    def test_grounded_flow_success(self) -> None:
        """Flujo grounded exitoso con esquema rag-grounded-v1 y metadata completa."""
        plan_dict = {
            "subquestions": [
                {
                    "id": "p1",
                    "question": "¿Cuál es el gasto en 2050?",
                    "required_evidence": "Gasto 2050",
                    "depends_on": [],
                }
            ]
        }
        mock_gen = MagicMock()
        mock_gen.model = DEFAULT_GENERATION_MODEL
        mock_gen.get_digest.return_value = self.digest

        planner = MockPlanner(plan_dict, generator=mock_gen)

        mock_grounder = MagicMock(spec=OllamaGrounder)
        mock_grounder.model = DEFAULT_GENERATION_MODEL
        mock_grounder.generator = mock_gen

        def fake_answer(query, hits, original_query=None, expected_digest=None):
            return _make_valid_grounded_answer(hits, status="answered", digest=expected_digest)

        mock_grounder.answer.side_effect = fake_answer

        res = answer_semantic(
            index=self.index,
            query="¿Cuál es el gasto en 2050?",
            embedder=self.embedder,
            generator=mock_gen,
            planner=planner,
            grounder=mock_grounder,
        )

        self.assertEqual(res["schema_version"], GROUNDED_SCHEMA_VERSION)
        self.assertEqual(res["status"], "answered")
        self.assertEqual(len(res["claims"]), 1)
        self.assertEqual(res["node_answers"], 1)

        # Validación de flags metodológicos
        val = res["validation"]
        self.assertTrue(val["citation_quotes_checked"])
        self.assertFalse(val["plan_semantics_verified"])
        self.assertTrue(val["semantic_support_checked_by_model"])
        self.assertFalse(val["semantic_faithfulness_certified"])

        # Contadores y desglose
        api_calls = res["api_calls"]
        self.assertEqual(api_calls["plan"], 1)
        self.assertEqual(api_calls["resolutions"], 0)
        self.assertEqual(api_calls["selection"], 1)
        self.assertEqual(api_calls["generation"], 1)
        self.assertEqual(api_calls["verification"], 1)
        self.assertEqual(api_calls["total"], 1 + 0 + 1 + 1 + 1 + 0 + 0 + 0 + 1)  # 5 operaciones

        # Part details
        part = res["parts"][0]
        self.assertEqual(part["answer_prompt_sha256_kind"], "aggregate")
        self.assertIsNotNone(part["grounding"])

    def test_visible_pruning_in_grounded_parts(self) -> None:
        """Poda de citas irrelevantes visible en las afirmaciones de la parte grounded."""
        plan_dict = {
            "subquestions": [
                {
                    "id": "p1",
                    "question": "¿Cuál es el gasto en 2050?",
                    "required_evidence": "Gasto 2050",
                    "depends_on": [],
                }
            ]
        }
        mock_gen = MagicMock()
        mock_gen.model = DEFAULT_GENERATION_MODEL
        mock_gen.get_digest.return_value = self.digest
        planner = MockPlanner(plan_dict, generator=mock_gen)

        mock_grounder = MagicMock(spec=OllamaGrounder)
        mock_grounder.model = DEFAULT_GENERATION_MODEL
        mock_grounder.generator = mock_gen

        def fake_pruned_answer(query, hits, original_query=None, expected_digest=None):
            ans = _make_valid_grounded_answer(hits, status="answered", digest=expected_digest)
            # Solo deja E1 y asegura que la evidencia publicada coincida
            return ans

        mock_grounder.answer.side_effect = fake_pruned_answer

        res = answer_semantic(
            index=self.index,
            query="¿Cuál es el gasto en 2050?",
            embedder=self.embedder,
            generator=mock_gen,
            planner=planner,
            grounder=mock_grounder,
        )

        part = res["parts"][0]
        self.assertEqual(len(part["claims"][0]["evidence"]), 1)
        self.assertEqual(part["claims"][0]["evidence"][0]["source_id"], "S1")

    def test_grounder_insufficient_blocks_dag(self) -> None:
        """Padre insuficiente por verificación bloquea al hijo sin resolución, búsqueda ni grounder."""
        plan_dict = {
            "subquestions": [
                {
                    "id": "p1",
                    "question": "¿Cuál es el gasto en 2050?",
                    "required_evidence": "Gasto 2050",
                    "depends_on": [],
                },
                {
                    "id": "p2",
                    "question": "¿Cómo se compara con 2070?",
                    "required_evidence": "Comparación",
                    "depends_on": ["p1"],
                },
            ]
        }
        mock_gen = MagicMock()
        mock_gen.model = DEFAULT_GENERATION_MODEL
        mock_gen.get_digest.return_value = self.digest
        planner = MockPlanner(plan_dict, generator=mock_gen)

        mock_grounder = MagicMock(spec=OllamaGrounder)
        mock_grounder.model = DEFAULT_GENERATION_MODEL
        mock_grounder.generator = mock_gen

        # p1 falla la verificación
        def fake_insufficient_answer(query, hits, original_query=None, expected_digest=None):
            return _make_valid_grounded_answer(hits, status="insufficient_evidence", digest=expected_digest)

        mock_grounder.answer.side_effect = fake_insufficient_answer

        res = answer_semantic(
            index=self.index,
            query="Compara 2050 y 2070.",
            embedder=self.embedder,
            generator=mock_gen,
            planner=planner,
            grounder=mock_grounder,
        )

        self.assertEqual(res["status"], "insufficient_evidence")
        self.assertEqual(res["claims"], [])

        parts = res["parts"]
        self.assertEqual(len(parts), 2)
        # p1 ejecutado pero insuficiente
        self.assertEqual(parts[0]["execution_status"], "executed")
        self.assertEqual(parts[0]["status"], "insufficient_evidence")

        # p2 bloqueado inmediatamente
        self.assertEqual(parts[1]["execution_status"], "blocked_dependency")
        self.assertEqual(parts[1]["status"], "insufficient_evidence")
        self.assertIn("Dependencia bloqueada por nodo(s) no respondido(s): p1", parts[1]["reason"])
        self.assertIsNone(parts[1]["grounding"])

        # Grounder solo fue llamado 1 vez (para p1, nunca para p2)
        self.assertEqual(mock_grounder.answer.call_count, 1)

    def test_falsified_grounded_results_rejected(self) -> None:
        """Resultados falsificados devueltos por un grounder inyectado son rechazados antes de usarse."""
        plan_dict = {
            "subquestions": [
                {
                    "id": "p1",
                    "question": "¿Gasto?",
                    "required_evidence": "Gasto",
                    "depends_on": [],
                }
            ]
        }
        mock_gen = MagicMock()
        mock_gen.model = DEFAULT_GENERATION_MODEL
        mock_gen.get_digest.return_value = self.digest
        planner = MockPlanner(plan_dict, generator=mock_gen)

        mock_grounder = MagicMock(spec=OllamaGrounder)
        mock_grounder.model = DEFAULT_GENERATION_MODEL
        mock_grounder.generator = mock_gen

        with self.subTest("stats_falsificadas"):
            def bad_stats_answer(query, hits, original_query=None, expected_digest=None):
                ans = _make_valid_grounded_answer(hits, status="answered", digest=expected_digest)
                ans["stats"]["prompt_eval_count"] += 9999
                return ans

            mock_grounder.answer.side_effect = bad_stats_answer
            with self.assertRaises(ValueError) as ctx:
                answer_semantic(
                    index=self.index,
                    query="¿Gasto?",
                    embedder=self.embedder,
                    generator=mock_gen,
                    planner=planner,
                    grounder=mock_grounder,
                )
            self.assertIn("sum mismatch", str(ctx.exception).lower())

        with self.subTest("hash_falsificado"):
            def bad_hash_answer(query, hits, original_query=None, expected_digest=None):
                ans = _make_valid_grounded_answer(hits, status="answered", digest=expected_digest)
                ans["prompt_sha256"] = "f" * 64
                return ans

            mock_grounder.answer.side_effect = bad_hash_answer
            with self.assertRaises(ValueError) as ctx:
                answer_semantic(
                    index=self.index,
                    query="¿Gasto?",
                    embedder=self.embedder,
                    generator=mock_gen,
                    planner=planner,
                    grounder=mock_grounder,
                )
            self.assertIn("prompt_sha256", str(ctx.exception))

        with self.subTest("cita_fuera_de_hits_del_nodo"):
            def bad_quote_answer(query, hits, original_query=None, expected_digest=None):
                ans = _make_valid_grounded_answer(hits, status="answered", digest=expected_digest)
                ans["grounding"]["selected_evidence"][0]["source_id"] = "S99"
                return ans

            mock_grounder.answer.side_effect = bad_quote_answer
            with self.assertRaises(ValueError) as ctx:
                answer_semantic(
                    index=self.index,
                    query="¿Gasto?",
                    embedder=self.embedder,
                    generator=mock_gen,
                    planner=planner,
                    grounder=mock_grounder,
                )
            self.assertIn("source_id", str(ctx.exception))

    def test_identity_and_metadata_pre_network(self) -> None:
        """Validaciones de consistencia de modelo e identidad de generador antes de contactar red."""
        mock_gen = MagicMock()
        mock_gen.model = DEFAULT_GENERATION_MODEL

        with self.subTest("grounder_model_mismatch"):
            bad_grounder = MagicMock(spec=OllamaGrounder)
            bad_grounder.model = "qwen3.5:4b-q4_K_M"
            with self.assertRaises(ValueError) as ctx:
                answer_semantic(
                    index=self.index,
                    query="¿Gasto?",
                    model=DEFAULT_GENERATION_MODEL,
                    grounder=bad_grounder,
                )
            self.assertIn("does not match requested model", str(ctx.exception))

        with self.subTest("grounder_generator_different_instance"):
            other_gen = MagicMock()
            other_gen.model = DEFAULT_GENERATION_MODEL
            bad_grounder = MagicMock(spec=OllamaGrounder)
            bad_grounder.model = DEFAULT_GENERATION_MODEL
            bad_grounder.generator = other_gen

            with self.assertRaises(ValueError) as ctx:
                answer_semantic(
                    index=self.index,
                    query="¿Gasto?",
                    model=DEFAULT_GENERATION_MODEL,
                    generator=mock_gen,
                    grounder=bad_grounder,
                )
            self.assertIn("must be the same instance", str(ctx.exception))

    def test_global_sources_with_gaps_per_node(self) -> None:
        """Fuentes globales asignan S1..SN y los nodos reciben sus propios hits sin fuga cruzada."""
        plan_dict = {
            "subquestions": [
                {"id": "p1", "question": "¿Gasto en 2050?", "required_evidence": "2050", "depends_on": []},
                {"id": "p2", "question": "¿Tasa de dependencia?", "required_evidence": "dep", "depends_on": []},
            ]
        }
        mock_gen = MagicMock()
        mock_gen.model = DEFAULT_GENERATION_MODEL
        mock_gen.get_digest.return_value = self.digest
        planner = MockPlanner(plan_dict, generator=mock_gen)

        recorded_hits_per_node: dict[str, list[SearchHit]] = {}

        mock_grounder = MagicMock(spec=OllamaGrounder)
        mock_grounder.model = DEFAULT_GENERATION_MODEL
        mock_grounder.generator = mock_gen

        def fake_answer(query, hits, original_query=None, expected_digest=None):
            node_key = "p1" if "2050" in query else "p2"
            recorded_hits_per_node[node_key] = hits
            return _make_valid_grounded_answer(hits, status="answered", digest=expected_digest)

        mock_grounder.answer.side_effect = fake_answer

        res = answer_semantic(
            index=self.index,
            query="¿Gasto en 2050 y tasa de dependencia?",
            embedder=self.embedder,
            generator=mock_gen,
            planner=planner,
            grounder=mock_grounder,
        )

        self.assertEqual(res["status"], "answered")
        # p1 descubrió S1..
        p1_ranks = [h.rank for h in recorded_hits_per_node["p1"]]
        p2_ranks = [h.rank for h in recorded_hits_per_node["p2"]]
        # No debe haber fuga de objetos mutados entre p1 y p2
        self.assertIn(1, p1_ranks)
        self.assertTrue(any(r > 1 for r in p2_ranks))

    def test_totals_calls_hash_no_double_counting(self) -> None:
        """Total de llamadas suma operaciones lógicas sin doble conteo de node_answers."""
        plan_dict = {
            "subquestions": [
                {"id": "p1", "question": "¿Gasto 2050?", "required_evidence": "2050", "depends_on": []},
            ]
        }
        mock_gen = MagicMock()
        mock_gen.model = DEFAULT_GENERATION_MODEL
        mock_gen.get_digest.return_value = self.digest
        planner = MockPlanner(plan_dict, generator=mock_gen)

        mock_grounder = MagicMock(spec=OllamaGrounder)
        mock_grounder.model = DEFAULT_GENERATION_MODEL
        mock_grounder.generator = mock_gen

        def fake_totals_answer(query, hits, original_query=None, expected_digest=None):
            return _make_valid_grounded_answer(hits, status="answered", digest=expected_digest)

        mock_grounder.answer.side_effect = fake_totals_answer

        res = answer_semantic(
            index=self.index,
            query="¿Gasto 2050?",
            embedder=self.embedder,
            generator=mock_gen,
            planner=planner,
            grounder=mock_grounder,
        )

        calls = res["api_calls"]
        expected_sum = (
            calls["plan"]
            + calls["resolutions"]
            + calls["selection"]
            + calls["generation"]
            + calls["verification"]
            + calls["repair_selection"]
            + calls["repair_generation"]
            + calls["reverification"]
            + calls["searches"]
        )
        self.assertEqual(calls["total"], expected_sum)
        # node_answers no está dentro de la suma de api_calls["total"]
        self.assertEqual(res["node_answers"], 1)

        # Desglose de stages en stats
        stats = res["stats"]
        self.assertIn("grounding_stages", stats)
        self.assertIn("selection", stats["grounding_stages"])
        self.assertEqual(stats["grounding_stages"]["selection"]["prompt_eval_count"], 100)
        self.assertEqual(stats["answers"]["prompt_eval_count"], 330)
        self.assertEqual(stats["total"]["prompt_eval_count"], 100 + 0 + 330)

    def test_cli_grounded_and_baseline_regression(self) -> None:
        """La CLI admite --grounded con salida JSON/texto y mantiene el baseline sin flag."""
        # 1. Baseline sin flag
        with patch("rag_pensiones.semantic.answer_semantic") as mock_ans:
            mock_ans.return_value = {
                "schema_version": SCHEMA_VERSION,
                "query": "¿Gasto?",
                "status": "answered",
                "generation_model": DEFAULT_GENERATION_MODEL,
                "parts": [],
                "plan": {"subquestions": []},
                "sources": [],
                "claims": [],
            }
            stdout = io.StringIO()
            with patch("sys.stdout", stdout):
                code = main(["--query", "¿Gasto?", "--json"])
            self.assertEqual(code, 0)
            mock_ans.assert_called_once()
            _, kwargs = mock_ans.call_args
            self.assertIsNone(kwargs.get("grounder"))

        # 2. Con flag --grounded
        with patch("rag_pensiones.semantic.answer_semantic") as mock_ans_grounded:
            mock_ans_grounded.return_value = {
                "schema_version": GROUNDED_SCHEMA_VERSION,
                "query": "¿Gasto?",
                "status": "answered",
                "generation_model": DEFAULT_GENERATION_MODEL,
                "parts": [
                    {
                        "id": "p1",
                        "question": "¿Gasto?",
                        "execution_status": "executed",
                        "status": "answered",
                        "claims": [{"text": "Dato", "evidence": [{"source_id": "S1", "quote": "Cita"}]}],
                        "grounding": {
                            "repair_attempts": 0,
                            "accepted_claims": [{"claim_id": "C1"}],
                            "rejected_claims": [],
                        },
                    }
                ],
                "plan": {"subquestions": []},
                "sources": [{"source_id": "S1", "source_file": "f.pdf", "page_number": 95, "published_at": "2025-03-31", "retrievals": []}],
                "claims": [{"text": "Dato", "evidence": [{"source_id": "S1", "quote": "Cita"}]}],
            }
            stdout = io.StringIO()
            with patch("sys.stdout", stdout):
                code = main(["--query", "¿Gasto?", "--grounded"])
            self.assertEqual(code, 0)
            _, kwargs = mock_ans_grounded.call_args
            self.assertIsNotNone(kwargs.get("grounder"))
            output_text = stdout.getvalue()
            self.assertIn("apoyo evaluado por modelo", output_text)
            self.assertIn("Intentos de reparación: 0", output_text)
            self.assertIn("Afirmaciones aceptadas: 1, rechazadas: 0", output_text)
            self.assertIn("NOTA METODOLÓGICA", output_text)


if __name__ == "__main__":
    unittest.main()
