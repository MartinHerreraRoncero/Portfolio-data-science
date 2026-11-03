"""Tests deterministas offline para la planificación semántica local y resolución de dependencias."""

import copy
import json
import unittest
from unittest.mock import MagicMock

from rag_pensiones.answer import (
    ALLOWED_MODELS,
    DEFAULT_GENERATION_MODEL,
    MAX_MESSAGES_BYTES,
    OllamaGenerator,
    parse_chat_response,
)
from rag_pensiones.planning import (
    DEFAULT_SCOPE_YEARS,
    PLAN_JSON_SCHEMA,
    RESOLVE_JSON_SCHEMA,
    OllamaPlanner,
    build_plan_messages,
    build_resolve_messages,
    extract_years,
    validate_plan,
)


def _make_valid_plan(subquestions: list[dict] | None = None) -> dict:
    if subquestions is None:
        subquestions = [
            {
                "id": "p1",
                "question": "¿Cuál es el gasto en pensiones proyectado para 2050?",
                "required_evidence": "Proyección del gasto en pensiones para el año 2050.",
                "depends_on": [],
            }
        ]
    return {"subquestions": subquestions}


class PlanningMockGenerator:
    """Mock determinista de OllamaGenerator para pruebas offline de planificación y resolución."""

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
        self.last_schema: dict | None = None
        self.raw_response = raw_response

    def get_digest(self) -> str:
        count = self.calls.count("get_digest")
        self.calls.append("get_digest")
        if count == 0:
            return self.digest
        return self.digest_after

    def chat(self, messages: list[dict], response_schema: dict) -> dict:
        self.calls.append("chat")
        self.last_messages = messages
        self.last_schema = response_schema
        if self.raw_response is not None:
            return self.raw_response
        default_content = json.dumps(
            {
                "subquestions": [
                    {
                        "id": "p1",
                        "question": "¿Cuál es el gasto en pensiones proyectado para 2050?",
                        "required_evidence": "Proyección del gasto en pensiones para el año 2050.",
                        "depends_on": [],
                    }
                ]
            },
            ensure_ascii=False,
        )
        return {
            "model": self.model,
            "done": True,
            "done_reason": "stop",
            "message": {
                "role": "assistant",
                "content": default_content,
            },
            "prompt_eval_count": 140,
            "eval_count": 55,
            "eval_duration": 500000000,
            "total_duration": 1200000000,
        }


class TestValidatePlan(unittest.TestCase):
    """Pruebas exhaustivas de validación estricta de planes semánticos."""

    def test_validate_plan_simple_single_node(self):
        """Verifica que un plan simple de 1 nodo conforme es válido."""
        plan = _make_valid_plan()
        res = validate_plan(plan, query="¿Cuál es el gasto en 2050?")
        self.assertEqual(len(res["subquestions"]), 1)
        self.assertEqual(res["subquestions"][0]["id"], "p1")
        self.assertEqual(res["subquestions"][0]["depends_on"], [])

    def test_validate_plan_compound_valid_dependencies(self):
        """Verifica plan compuesto de 3 nodos con dependencias topológicamente válidas."""
        plan = {
            "subquestions": [
                {
                    "id": "p1",
                    "question": "¿Cuál es la proyección de gasto para 2050?",
                    "required_evidence": "Gasto proyectado 2050.",
                    "depends_on": [],
                },
                {
                    "id": "p2",
                    "question": "¿Cuál es la proyección de gasto para 2070?",
                    "required_evidence": "Gasto proyectado 2070.",
                    "depends_on": [],
                },
                {
                    "id": "p3",
                    "question": "¿Cuál es la diferencia entre el gasto proyectado de 2050 y 2070?",
                    "required_evidence": "Comparativa de gasto entre ambos años.",
                    "depends_on": ["p1", "p2"],
                },
            ]
        }
        res = validate_plan(plan, query="Compara el gasto en pensiones entre 2050 y 2070.")
        self.assertEqual(len(res["subquestions"]), 3)
        self.assertEqual(res["subquestions"][2]["depends_on"], ["p1", "p2"])

    def test_validate_plan_query_without_explicit_connectors(self):
        """Verifica admisión de consulta comparativa sin conectores 'y' explícitos."""
        plan = {
            "subquestions": [
                {
                    "id": "p1",
                    "question": "¿Cuál es el gasto en pensiones en 2050?",
                    "required_evidence": "Datos 2050.",
                    "depends_on": [],
                },
                {
                    "id": "p2",
                    "question": "¿Cuál es el gasto en pensiones en 2070?",
                    "required_evidence": "Datos 2070.",
                    "depends_on": [],
                },
            ]
        }
        res = validate_plan(plan, query="Diferencia de pensiones 2050 frente a 2070")
        self.assertEqual(len(res["subquestions"]), 2)

    def test_reject_invalid_ids_disordered_or_duplicates(self):
        """Verifica rechazo de IDs no secuenciales, desordenados o duplicados."""
        # Comienza en p2
        plan = {"subquestions": [{"id": "p2", "question": "q", "required_evidence": "e", "depends_on": []}]}
        with self.assertRaises(ValueError) as ctx:
            validate_plan(plan, query="consulta")
        self.assertIn("expected strictly ordered 'p1'", str(ctx.exception))

        # Salto de ID p1 -> p3
        plan_skip = {
            "subquestions": [
                {"id": "p1", "question": "q1", "required_evidence": "e1", "depends_on": []},
                {"id": "p3", "question": "q2", "required_evidence": "e2", "depends_on": ["p1"]},
            ]
        }
        with self.assertRaises(ValueError):
            validate_plan(plan_skip, query="consulta")

        # IDs duplicados
        plan_dup = {
            "subquestions": [
                {"id": "p1", "question": "q1", "required_evidence": "e1", "depends_on": []},
                {"id": "p1", "question": "q2", "required_evidence": "e2", "depends_on": []},
            ]
        }
        with self.assertRaises(ValueError):
            validate_plan(plan_dup, query="consulta")

    def test_reject_self_forward_duplicate_or_cycle_dependencies(self):
        """Verifica rechazo de auto-dependencias, dependencias adelantadas y duplicadas."""
        # Auto-dependencia p1 -> p1
        plan_self = {"subquestions": [{"id": "p1", "question": "q", "required_evidence": "e", "depends_on": ["p1"]}]}
        with self.assertRaises(ValueError) as ctx:
            validate_plan(plan_self, query="consulta")
        self.assertIn("forward dependency", str(ctx.exception))

        # Adelantada p1 -> p2
        plan_forward = {
            "subquestions": [
                {"id": "p1", "question": "q1", "required_evidence": "e1", "depends_on": ["p2"]},
                {"id": "p2", "question": "q2", "required_evidence": "e2", "depends_on": []},
            ]
        }
        with self.assertRaises(ValueError):
            validate_plan(plan_forward, query="consulta")

        # Dependencias duplicadas en un nodo
        plan_dup_dep = {
            "subquestions": [
                {"id": "p1", "question": "q1", "required_evidence": "e1", "depends_on": []},
                {"id": "p2", "question": "q2", "required_evidence": "e2", "depends_on": ["p1", "p1"]},
            ]
        }
        with self.assertRaises(ValueError) as ctx:
            validate_plan(plan_dup_dep, query="consulta")
        self.assertIn("duplicate IDs", str(ctx.exception))

    def test_reject_node_counts_zero_or_exceeding_four(self):
        """Verifica que el plan exige entre 1 y 4 nodos exactamente."""
        with self.assertRaises(ValueError):
            validate_plan({"subquestions": []}, query="consulta")

        huge_nodes = [
            {"id": f"p{i}", "question": f"q{i}", "required_evidence": f"e{i}", "depends_on": []}
            for i in range(1, 6)
        ]
        with self.assertRaises(ValueError) as ctx:
            validate_plan({"subquestions": huge_nodes}, query="consulta")
        self.assertIn("between 1 and 4", str(ctx.exception))

    def test_reject_extra_keys_and_types_and_bools(self):
        """Verifica rechazo de claves adicionales y tipos booleanos en raíz y nodos."""
        # Extra en raíz
        with self.assertRaises(ValueError):
            validate_plan({"subquestions": [], "extra": 123}, query="consulta")

        # Booleano en raíz
        with self.assertRaises(ValueError):
            validate_plan(True, query="consulta")  # type: ignore

        # Extra en nodo
        bad_node = {"id": "p1", "question": "q", "required_evidence": "e", "depends_on": [], "extra": "x"}
        with self.assertRaises(ValueError):
            validate_plan({"subquestions": [bad_node]}, query="consulta")

        # Booleano en question
        bad_q = {"id": "p1", "question": True, "required_evidence": "e", "depends_on": []}
        with self.assertRaises(ValueError):
            validate_plan({"subquestions": [bad_q]}, query="consulta")

    def test_reject_empty_strings_and_length_boundaries(self):
        """Verifica rechazo de cadenas vacías o longitudes que excedan los límites."""
        # Question vacía
        with self.assertRaises(ValueError):
            validate_plan(
                {"subquestions": [{"id": "p1", "question": "   ", "required_evidence": "e", "depends_on": []}]},
                query="consulta",
            )

        # Question > 600 chars
        long_q = "a" * 601
        with self.assertRaises(ValueError):
            validate_plan(
                {"subquestions": [{"id": "p1", "question": long_q, "required_evidence": "e", "depends_on": []}]},
                query="consulta",
            )

        # Evidence vacía
        with self.assertRaises(ValueError):
            validate_plan(
                {"subquestions": [{"id": "p1", "question": "q", "required_evidence": "", "depends_on": []}]},
                query="consulta",
            )

        # Evidence > 250 chars
        long_ev = "e" * 251
        with self.assertRaises(ValueError):
            validate_plan(
                {"subquestions": [{"id": "p1", "question": "q", "required_evidence": long_ev, "depends_on": []}]},
                query="consulta",
            )

        # Query > 1000 chars o vacía
        with self.assertRaises(ValueError):
            validate_plan(_make_valid_plan(), query="")
        with self.assertRaises(ValueError):
            validate_plan(_make_valid_plan(), query="q" * 1001)

    def test_years_preservation_and_restriction(self):
        """Verifica que los años del original deben conservarse en questions y no admitir años ajenos."""
        # Query con 2050 y 2070. Si falta 2070 en questions -> rechazo aunque esté en required_evidence
        plan_missing_year = {
            "subquestions": [
                {
                    "id": "p1",
                    "question": "¿Cuál es la previsión para 2050?",
                    "required_evidence": "Datos de 2050 y 2070.",
                    "depends_on": [],
                }
            ]
        }
        with self.assertRaises(ValueError) as ctx:
            validate_plan(plan_missing_year, query="Compara las previsiones de 2050 y 2070.")
        self.assertIn("Original query years missing from subquestions", str(ctx.exception))
        self.assertIn("2070", str(ctx.exception))

        # Query con 2050. Subpregunta introduce año no autorizado 2030 (no está en query ni scope 2025)
        plan_unauthorized_year = {
            "subquestions": [
                {
                    "id": "p1",
                    "question": "¿Cuál es la previsión para 2050 y 2030?",
                    "required_evidence": "Datos 2050.",
                    "depends_on": [],
                }
            ]
        }
        with self.assertRaises(ValueError) as ctx:
            validate_plan(plan_unauthorized_year, query="Previsión de 2050.")
        self.assertIn("unauthorized years", str(ctx.exception))
        self.assertIn("2030", str(ctx.exception))

        # Año de scope por defecto (2025) está autorizado aunque no esté en la query
        plan_with_scope = {
            "subquestions": [
                {
                    "id": "p1",
                    "question": "¿Qué afirma el informe de 2025 sobre el gasto de 2050?",
                    "required_evidence": "Informe de 2025.",
                    "depends_on": [],
                }
            ]
        }
        res = validate_plan(plan_with_scope, query="Gasto de 2050.")
        self.assertEqual(len(res["subquestions"]), 1)

        # Scope_years personalizado
        res_custom = validate_plan(
            {
                "subquestions": [
                    {
                        "id": "p1",
                        "question": "¿Previsión 2050 según publicación 2024?",
                        "required_evidence": "Evidencia 2024.",
                        "depends_on": [],
                    }
                ]
            },
            query="Previsión 2050.",
            scope_years={"2024"},
        )
        self.assertEqual(len(res_custom["subquestions"]), 1)


class TestOllamaPlannerPlan(unittest.TestCase):
    """Pruebas del método plan() de OllamaPlanner."""

    def test_plan_full_success_flow(self):
        """Verifica que plan() genera estructura conforme sin incluir respuestas, fuentes ni gold."""
        mock_gen = PlanningMockGenerator(model="qwen3.5:9b")
        planner = OllamaPlanner(generator=mock_gen, model="qwen3.5:9b")

        report = planner.plan("¿Cuál es el gasto en 2050?")

        self.assertIn("plan", report)
        self.assertIn("model", report)
        self.assertIn("model_digest", report)
        self.assertIn("prompt_sha256", report)
        self.assertIn("stats", report)
        self.assertIn("parameters", report)

        # Pureza: sin respuestas, fuentes ni gold
        self.assertNotIn("status", report)
        self.assertNotIn("claims", report)
        self.assertNotIn("sources", report)
        self.assertNotIn("gold", report)

        self.assertEqual(report["model"], "qwen3.5:9b")
        self.assertEqual(report["model_digest"], "test_gen_digest_sha256")
        self.assertEqual(len(report["plan"]["subquestions"]), 1)

    def test_pre_network_validation_aborts_without_contacting_generator(self):
        """Verifica que validaciones previas de query abortan antes de invocar al generador."""
        mock_gen = PlanningMockGenerator()
        planner = OllamaPlanner(generator=mock_gen)

        with self.assertRaises(ValueError):
            planner.plan("")
        with self.assertRaises(ValueError):
            planner.plan("   \t\n  ")
        with self.assertRaises(ValueError):
            planner.plan(True)  # type: ignore

        self.assertEqual(len(mock_gen.calls), 0)

    def test_model_4b_runs_without_fallback(self):
        """Verifica que el modelo 4B ejecuta con sus parámetros sin intentar fallback."""
        mock_gen = PlanningMockGenerator(model="qwen3.5:4b-q4_K_M")
        planner = OllamaPlanner(generator=mock_gen, model="qwen3.5:4b-q4_K_M")

        report = planner.plan("¿Cuál es la proyección para 2050?")
        self.assertEqual(report["model"], "qwen3.5:4b-q4_K_M")

    def test_unauthorized_model_rejected(self):
        """Verifica rechazo inmediato de modelos no autorizados."""
        with self.assertRaises(ValueError) as ctx:
            OllamaPlanner(model="unauthorized:model")
        self.assertIn("not allowed", str(ctx.exception))

    def test_digest_changed_during_plan_aborts(self):
        """Verifica aborto si el digest del modelo cambia durante la planificación."""
        mock_gen = PlanningMockGenerator(
            digest="digest_initial",
            digest_after="digest_mutated",
        )
        planner = OllamaPlanner(generator=mock_gen)

        with self.assertRaises(RuntimeError) as ctx:
            planner.plan("¿Cuál es el gasto en 2050?")
        self.assertIn("digest changed during generation", str(ctx.exception))

    def test_shared_transport_uses_plan_json_schema(self):
        """Verifica que plan() envía exactamente PLAN_JSON_SCHEMA en el transporte chat."""
        mock_gen = PlanningMockGenerator()
        planner = OllamaPlanner(generator=mock_gen)

        planner.plan("¿Cuál es el gasto en 2050?")
        self.assertIn("chat", mock_gen.calls)
        self.assertEqual(mock_gen.last_schema, PLAN_JSON_SCHEMA)

    def test_plan_messages_budget_guard(self):
        """Verifica que build_plan_messages no excede el presupuesto de 6500 bytes."""
        messages = build_plan_messages("Consulta de prueba")
        raw_bytes = json.dumps(messages, ensure_ascii=False).encode("utf-8")
        self.assertLessEqual(len(raw_bytes), MAX_MESSAGES_BYTES)


class TestOllamaPlannerResolve(unittest.TestCase):
    """Pruebas del método resolve() de OllamaPlanner."""

    def setUp(self):
        self.valid_dependencies = [
            {
                "question": "¿Cuál es la proyección de gasto para 2050?",
                "status": "answered",
                "claims": [
                    {
                        "text": "El gasto aumentará al 16,1% del PIB en 2050.",
                        "evidence": [{"source_id": "S1", "quote": "aumentará al 16,1% del PIB en 2050"}],
                    }
                ],
            }
        ]

    def test_resolve_success_flow(self):
        """Verifica reescritura conforme de pregunta dependiente a autónoma."""
        res_payload = {
            "model": "qwen3.5:9b",
            "done": True,
            "done_reason": "stop",
            "message": {
                "role": "assistant",
                "content": json.dumps(
                    {"question": "¿Cuál es el impacto macroeconómico del 16,1% de gasto en 2050?"}
                ),
            },
            "prompt_eval_count": 160,
            "eval_count": 40,
            "eval_duration": 400000000,
            "total_duration": 1100000000,
        }
        mock_gen = PlanningMockGenerator(raw_response=res_payload)
        planner = OllamaPlanner(generator=mock_gen)

        result = planner.resolve(
            question="¿Cuál es su impacto en 2050?",
            original_query="Compara gasto de 2050 y su impacto",
            dependencies=self.valid_dependencies,
            expected_digest="test_gen_digest_sha256",
        )

        self.assertIn("question", result)
        self.assertIn("model", result)
        self.assertIn("model_digest", result)
        self.assertIn("prompt_sha256", result)
        self.assertIn("stats", result)
        self.assertIn("parameters", result)
        self.assertEqual(result["question"], "¿Cuál es el impacto macroeconómico del 16,1% de gasto en 2050?")

    def test_resolve_expected_digest_pin(self):
        """Verifica que resolve aborta si el digest inicial o final difiere de expected_digest."""
        mock_gen = PlanningMockGenerator(digest="digest_real")
        planner = OllamaPlanner(generator=mock_gen)

        # Mismatch previo
        with self.assertRaises(RuntimeError) as ctx:
            planner.resolve(
                question="¿Pregunta 2050?",
                original_query="Consulta 2050",
                dependencies=self.valid_dependencies,
                expected_digest="digest_otro",
            )
        self.assertIn("digest mismatch before resolve", str(ctx.exception))

        # Mismatch posterior (cambio en vuelo)
        mock_gen_mut = PlanningMockGenerator(digest="digest_ok", digest_after="digest_mutado")
        planner_mut = OllamaPlanner(generator=mock_gen_mut)
        with self.assertRaises(RuntimeError) as ctx:
            planner_mut.resolve(
                question="¿Pregunta 2050?",
                original_query="Consulta 2050",
                dependencies=self.valid_dependencies,
                expected_digest="digest_ok",
            )
        self.assertIn("digest changed during resolve", str(ctx.exception))

    def test_reject_insufficient_parent_dependencies(self):
        """Verifica que padres con insuficiencia o sin claims son rechazados antes del generador."""
        mock_gen = PlanningMockGenerator()
        planner = OllamaPlanner(generator=mock_gen)

        insufficient_deps = [
            {
                "question": "¿Dato?",
                "status": "insufficient_evidence",
                "claims": [],
            }
        ]
        with self.assertRaises(ValueError) as ctx:
            planner.resolve(
                question="¿Pregunta?",
                original_query="Consulta",
                dependencies=insufficient_deps,
                expected_digest="test_gen_digest_sha256",
            )
        self.assertIn("only 'answered' parents can be resolved", str(ctx.exception))
        # Generador no fue llamado
        self.assertNotIn("chat", mock_gen.calls)

    def test_resolve_years_preservation_and_checks(self):
        """Verifica que resolve no pierde años del nodo ni introduce años no autorizados."""
        # 1. Pérdida de año del nodo
        res_payload_drop_year = {
            "model": "qwen3.5:9b",
            "done": True,
            "done_reason": "stop",
            "message": {
                "role": "assistant",
                "content": json.dumps({"question": "¿Cuál es el impacto macroeconómico?"}),
            },
            "prompt_eval_count": 100,
            "eval_count": 30,
            "eval_duration": 300000000,
            "total_duration": 900000000,
        }
        mock_gen = PlanningMockGenerator(raw_response=res_payload_drop_year)
        planner = OllamaPlanner(generator=mock_gen)
        with self.assertRaises(ValueError) as ctx:
            planner.resolve(
                question="¿Cuál es el impacto en 2050?",
                original_query="Gasto en 2050",
                dependencies=self.valid_dependencies,
                expected_digest="test_gen_digest_sha256",
            )
        self.assertIn("dropped years from dependent question", str(ctx.exception))

        # 2. Introducción de año no autorizado
        res_payload_new_year = {
            "model": "qwen3.5:9b",
            "done": True,
            "done_reason": "stop",
            "message": {
                "role": "assistant",
                "content": json.dumps({"question": "¿Cuál es el impacto en 2050 y 2099?"}),
            },
            "prompt_eval_count": 100,
            "eval_count": 30,
            "eval_duration": 300000000,
            "total_duration": 900000000,
        }
        mock_gen2 = PlanningMockGenerator(raw_response=res_payload_new_year)
        planner2 = OllamaPlanner(generator=mock_gen2)
        with self.assertRaises(ValueError) as ctx:
            planner2.resolve(
                question="¿Cuál es el impacto en 2050?",
                original_query="Gasto en 2050",
                dependencies=self.valid_dependencies,
                expected_digest="test_gen_digest_sha256",
            )
        self.assertIn("unauthorized years", str(ctx.exception))

    def test_resolve_messages_budget_overflow_raises_without_truncation(self):
        """Verifica que contexto excesivo de dependencias (>6500 bytes) falla sin truncar."""
        mock_gen = PlanningMockGenerator()
        planner = OllamaPlanner(generator=mock_gen)

        huge_claims = [
            {
                "text": "Afirmación muy extensa sobre previsiones presupuestarias. " * 30,
                "evidence": [{"source_id": "S1", "quote": "cita larga " * 20}],
            }
            for _ in range(10)
        ]
        huge_deps = [
            {
                "question": "Pregunta de contexto muy largo",
                "status": "answered",
                "claims": huge_claims,
            }
        ]

        with self.assertRaises(ValueError) as ctx:
            planner.resolve(
                question="¿Pregunta?",
                original_query="Consulta original",
                dependencies=huge_deps,
                expected_digest="test_gen_digest_sha256",
            )
        self.assertIn("exceeds budget of 6500 bytes", str(ctx.exception))
        self.assertNotIn("chat", mock_gen.calls)


class TestSharedTransportAndParsing(unittest.TestCase):
    """Pruebas de verificación de transporte /api/chat compartido y parse_chat_response."""

    def test_ollama_generator_chat_method_structure(self):
        """Verifica que OllamaGenerator.chat envía parámetros idénticos y el esquema provisto."""
        fake_opener = MagicMock()
        resp_mock = MagicMock()
        resp_mock.read.return_value = json.dumps({"test": "ok"}).encode("utf-8")
        fake_opener.open.return_value.__enter__.return_value = resp_mock

        gen = OllamaGenerator(opener=fake_opener)
        schema = {"type": "object", "properties": {"res": {"type": "string"}}}
        messages = [{"role": "user", "content": "hola"}]

        res = gen.chat(messages, schema)
        self.assertEqual(res, {"test": "ok"})

        req = fake_opener.open.call_args[0][0]
        self.assertEqual(req.full_url, "http://127.0.0.1:11434/api/chat")
        sent = json.loads(req.data.decode("utf-8"))
        self.assertEqual(sent["format"], schema)
        self.assertEqual(sent["options"]["temperature"], 0)
        self.assertEqual(sent["options"]["seed"], 42)
        self.assertEqual(sent["options"]["num_ctx"], 8192)
        self.assertEqual(sent["options"]["num_predict"], 1024)
        self.assertFalse(sent["think"])
        self.assertFalse(sent["stream"])

    def test_parse_chat_response_strict_anomalies(self):
        """Verifica detección de anomalías en parse_chat_response."""
        # Non-dict raw_response
        with self.assertRaises(ValueError):
            parse_chat_response("not_a_dict", "qwen3.5:9b")  # type: ignore

        # Wrong model
        with self.assertRaises(ValueError):
            parse_chat_response({"model": "wrong"}, "qwen3.5:9b")

        # Incomplete / done_reason != stop
        with self.assertRaises(ValueError):
            parse_chat_response({"model": "qwen3.5:9b", "done": False}, "qwen3.5:9b")

        # Bad stats
        raw_bad_stats = {
            "model": "qwen3.5:9b",
            "done": True,
            "done_reason": "stop",
            "message": {"role": "assistant", "content": '{"ok": true}'},
            "prompt_eval_count": -1,
            "eval_count": 10,
            "eval_duration": 10,
            "total_duration": 20,
        }
        with self.assertRaises(ValueError):
            parse_chat_response(raw_bad_stats, "qwen3.5:9b")

        # prompt_eval_count > 6912
        raw_huge_stat = copy.deepcopy(raw_bad_stats)
        raw_huge_stat["prompt_eval_count"] = 7000
        with self.assertRaises(ValueError):
            parse_chat_response(raw_huge_stat, "qwen3.5:9b")


if __name__ == "__main__":
    unittest.main()
