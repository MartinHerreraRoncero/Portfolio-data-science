"""Tests exhaustivos y deterministas offline para el módulo de fundamentación (grounding)."""

import copy
import hashlib
import json
import unittest
from itertools import product
from unittest.mock import MagicMock

from rag_pensiones.answer import (
    ALLOWED_MODELS,
    DEFAULT_GENERATION_MODEL,
    MAX_MESSAGES_BYTES,
    MAX_PROMPT_EVAL_COUNT,
    collapse_spaces,
)
from rag_pensiones.chunking import ChunkRecord
from rag_pensiones.grounding import (
    DEFAULT_PARAMETERS,
    MAX_EVIDENCE_BYTES,
    MAX_MESSAGES_BYTES_DEFAULT,
    MAX_MESSAGES_BYTES_VERIFICATION,
    SCHEMA_VERSION,
    SELECTION_JSON_SCHEMA,
    VERIFICATION_SYSTEM_PROMPT,
    OllamaGrounder,
    answer,
    build_generation_messages,
    build_repair_generation_messages,
    build_repair_selection_messages,
    build_selection_messages,
    build_selection_schema,
    build_source_contexts_for_claims,
    build_verification_messages,
    build_verification_schema,
    format_to_public_claims,
    prune_irrelevant_evidence,
    normalize_verification_payload,
    validate_generation_payload,
    validate_grounded_result,
    validate_repair_generation_payload,
    validate_selection_payload,
    validate_verification_payload,
    verify,
)
from rag_pensiones.search import SearchHit


def _make_chunk(idx: int, text: str) -> ChunkRecord:
    return ChunkRecord(
        chunk_id=f"chunk_id_{idx}",
        doc_id=f"doc_test_{idx}",
        source_file=f"informe_pensiones_{idx}.pdf",
        published_at="2025-03-31",
        page_number=idx * 10,
        start_char=0,
        end_char=len(text),
        text=text,
        strategy="page-local-v1:max_chars=1000:overlap_chars=150",
    )


def _make_hit(rank: int, text: str, score: float = 0.95) -> SearchHit:
    return SearchHit(
        rank=rank,
        score=score,
        chunk=_make_chunk(rank, text),
    )


class FakeGrounderGenerator:
    """Generador simulado offline para probar el flujo de grounding sin conexión externa."""

    def __init__(
        self,
        responses: list[dict] | None = None,
        model: str = DEFAULT_GENERATION_MODEL,
        digest: str = "test_gen_digest_sha256",
        digest_mutations: dict[int, str] | None = None,
    ) -> None:
        self.model = model
        self.digest = digest
        self.digest_mutations = digest_mutations or {}
        self.call_count = 0
        self.responses = list(responses or [])
        self.history: list[tuple[list[dict], dict]] = []

    def get_digest(self) -> str:
        if self.call_count in self.digest_mutations:
            return self.digest_mutations[self.call_count]
        return self.digest

    def chat(self, messages: list[dict], response_schema: dict) -> dict:
        self.history.append((messages, response_schema))
        if self.call_count >= len(self.responses):
            raise RuntimeError(f"Unexpected chat call #{self.call_count + 1}, responses exhausted")
        resp_data = self.responses[self.call_count]
        self.call_count += 1

        payload = resp_data.get("payload", resp_data)
        stats = resp_data.get("stats", {})
        model = resp_data.get("model", self.model)
        done = resp_data.get("done", True)
        done_reason = resp_data.get("done_reason", "stop")

        raw = {
            "model": model,
            "done": done,
            "done_reason": done_reason,
            "message": {
                "role": resp_data.get("role", "assistant"),
                "content": json.dumps(payload, ensure_ascii=False)
                if isinstance(payload, (dict, list))
                else str(payload),
            },
            "prompt_eval_count": stats.get("prompt_eval_count", 150),
            "eval_count": stats.get("eval_count", 80),
            "eval_duration": stats.get("eval_duration", 1200000),
            "total_duration": stats.get("total_duration", 2500000),
        }
        for opt in ("load_duration", "prompt_eval_duration"):
            if opt in stats:
                raw[opt] = stats[opt]
        return raw


class KeyedGrounderGenerator(FakeGrounderGenerator):
    """Simula el transporte con claves de Ollama manteniendo fixtures canónicas."""

    def chat(self, messages: list[dict], response_schema: dict) -> dict:
        raw = super().chat(messages, response_schema)
        payload = json.loads(raw["message"]["content"])
        if isinstance(payload.get("verdicts"), list):
            payload = {"verdicts": {
                v["claim_id"]: {
                    "verdict": v["verdict"], "reason": v["reason"],
                    "evidence_verdicts": {e["evidence_id"]: {"relation": e["relation"]}
                                          for e in v["evidence_verdicts"]},
                } for v in payload["verdicts"]
            }}
            raw["message"]["content"] = json.dumps(payload)
        return raw


class TestVerificationTransport(unittest.TestCase):
    def setUp(self) -> None:
        self.claims = [{"claim_id": "C1", "text": "Dato de prueba", "evidence_ids": ["E1"]}]
        self.payload = {"verdicts": {"C1": {
            "verdict": "supported", "reason": "Apoyo directo.",
            "evidence_verdicts": {"E1": {"relation": "supports"}},
        }}}

    def test_schema_locks_claim_and_evidence_keys(self) -> None:
        schema = build_verification_schema(self.claims)
        verdicts = schema["properties"]["verdicts"]
        self.assertEqual(verdicts["required"], ["C1"])
        self.assertFalse(verdicts["additionalProperties"])
        for branch in verdicts["properties"]["C1"]["anyOf"]:
            refs = branch["properties"]["evidence_verdicts"]
            self.assertEqual(set(refs["properties"]), {"E1"})
            self.assertEqual(refs["required"], ["E1"])
            self.assertFalse(refs["additionalProperties"])

    def test_schema_relations_cover_exactly_the_existing_verdict_invariants(self) -> None:
        relation_names = ("supports", "partial", "irrelevant", "contradicts")
        for count in range(1, 5):
            refs = [f"E{i + 1}" for i in range(count)]
            claims = [{"claim_id": "C1", "text": "Control lógico", "evidence_ids": refs}]
            branches = build_verification_schema(claims)["properties"]["verdicts"]["properties"]["C1"]["anyOf"]
            covered = set()
            for branch in branches:
                fields = branch["properties"]
                choices = [fields["evidence_verdicts"]["properties"][eid]["properties"]["relation"]["enum"]
                           for eid in refs]
                for relations in product(*choices):
                    expected = ("contradicted" if "contradicts" in relations else
                                "supported" if "supports" in relations or relations.count("partial") >= 2 else
                                "insufficient_evidence")
                    self.assertEqual(fields["verdict"]["const"], expected)
                    covered.add(relations)
            self.assertEqual(covered, set(product(relation_names, repeat=count)))

    def test_unknown_missing_and_extra_transport_fields_are_rejected(self) -> None:
        bad_payloads = []
        for path, value in (("unknown_claim", {}), ("extra_ref", {"relation": "supports"}),
                            ("missing_ref", None), ("extra_relation_field", "invalid")):
            bad = copy.deepcopy(self.payload)
            if path == "unknown_claim":
                bad["verdicts"]["C2"] = value
            elif path == "extra_ref":
                bad["verdicts"]["C1"]["evidence_verdicts"]["E999"] = value
            elif path == "missing_ref":
                bad["verdicts"]["C1"]["evidence_verdicts"].clear()
            else:
                bad["verdicts"]["C1"]["evidence_verdicts"]["E1"]["extra"] = value
            bad_payloads.append(bad)
        for bad in bad_payloads:
            with self.subTest(payload=bad), self.assertRaises(ValueError):
                normalize_verification_payload(bad, self.claims)

    def test_standalone_keyed_transport_keeps_public_api(self) -> None:
        generator = FakeGrounderGenerator([self.payload])
        evidence = [{"evidence_id": "E1", "source_id": "S1", "quote": "Dato de prueba respaldado."}]
        result = OllamaGrounder(generator=generator).verify(self.claims, evidence)
        self.assertEqual(result["verdicts"][0]["claim_id"], "C1")
        self.assertEqual(result["verdicts"][0]["evidence_verdicts"],
                         [{"evidence_id": "E1", "relation": "supports"}])

    def test_standalone_verification_allows_four_reference_evidences(self) -> None:
        from rag_pensiones.evaluate_grounding import evaluate_pairs

        claims = [{"claim_id": "C1", "text": "Hecho conjunto", "evidence_ids": ["E1", "E2", "E3", "E4"]}]
        evidence = [{"evidence_id": eid, "source_id": f"S{i}", "quote": "Pasaje sintético de referencia."}
                    for i, eid in enumerate(claims[0]["evidence_ids"], start=1)]
        payload = {"verdicts": {"C1": {
            "verdict": "supported", "reason": "Apoyo conjunto.",
            "evidence_verdicts": {eid: {"relation": "partial"} for eid in claims[0]["evidence_ids"]},
        }}}
        generator = FakeGrounderGenerator([payload])
        dataset = {"schema_version": "grounding-pairs-v1", "annotation": "Control de transporte, sin juicio semántico.",
                   "cases": [{"id": "four_refs", "split": "dev", "kind": "format", "claim": claims[0]["text"],
                              "evidence": [e["quote"] for e in evidence], "expected": "supported", "source_kind": "sintetico"}]}
        result = evaluate_pairs(dataset, grounder=OllamaGrounder(generator=generator))
        self.assertEqual(result["run_status"], "completed")
        self.assertEqual(len(result["cases"][0]["evidence_verdicts"]), 4)
        schema = generator.history[0][1]
        for branch in schema["properties"]["verdicts"]["properties"]["C1"]["anyOf"]:
            self.assertEqual(branch["properties"]["evidence_verdicts"]["required"], ["E1", "E2", "E3", "E4"])


class TestGroundingPipeline(unittest.TestCase):
    """Tests funcionales del pipeline de 3 etapas y vuelta única de reparación."""

    def setUp(self) -> None:
        self.text1 = (
            "El gasto en pensiones contributivas de jubilación alcanzará el 16,1% del PIB en el año 2050 "
            "según las proyecciones oficiales de la AIReF publicadas en 2025."
        )
        self.text2 = (
            "La tasa de dependencia superará el 53% en el horizonte proyectado para el año 2050, "
            "lo que incrementará la presión financiera sobre el sistema público."
        )
        self.text3 = "CAPÍTULO 3. PROYECCIONES DEMOGRÁFICAS Y GASTO PÚBLICO EN PENSIONES."
        self.text4 = (
            "En el año 2023, las pensiones del sistema de Seguridad Social representaron un 12,9% del PIB "
            "con un total superior a 10 millones de prestaciones."
        )

        self.hit1 = _make_hit(1, self.text1)
        self.hit2 = _make_hit(2, self.text2)
        self.hit3 = _make_hit(3, self.text3)
        self.hit4 = _make_hit(4, self.text4)
        self.hits = [self.hit1, self.hit2, self.hit3, self.hit4]

    def test_selection_schema_pairs_only_literal_passages_with_their_sources(self) -> None:
        schema = build_selection_schema(build_selection_messages("¿Cuál es el dato?", self.hits))
        alternatives = schema["anyOf"][0]["properties"]["evidence"]["items"]["anyOf"]
        sources = {f"S{hit.rank}": collapse_spaces(hit.chunk.text) for hit in self.hits}
        self.assertTrue(alternatives)
        for alternative in alternatives:
            properties = alternative["properties"]
            source_id = properties["source_id"]["const"]
            for quote in properties["quote"]["enum"]:
                self.assertIn(quote, sources[source_id])
                self.assertNotIn("...", quote)
        self.assertEqual(schema["anyOf"][0]["properties"]["reason"], {"const": ""})

    def test_full_valid_3stage_flow(self) -> None:
        """Flujo válido de 3 etapas: selection -> generation -> verification sin reparación."""
        quote1 = "El gasto en pensiones contributivas de jubilación alcanzará el 16,1% del PIB en el año 2050"
        quote2 = "La tasa de dependencia superará el 53% en el horizonte proyectado para el año 2050"

        responses = [
            # 1. selection
            {
                "payload": {
                    "status": "selected",
                    "evidence": [
                        {"source_id": "S1", "quote": quote1},
                        {"source_id": "S2", "quote": quote2},
                    ],
                    "reason": "",
                }
            },
            # 2. generation
            {
                "payload": {
                    "status": "answered",
                    "claims": [
                        {
                            "claim_id": "C1",
                            "text": "El gasto en pensiones alcanzará el 16,1% del PIB en 2050.",
                            "evidence_ids": ["E1"],
                        },
                        {
                            "claim_id": "C2",
                            "text": "La tasa de dependencia superará el 53% en 2050.",
                            "evidence_ids": ["E2"],
                        },
                    ],
                    "reason": "",
                }
            },
            # 3. verification
            {
                "payload": {
                    "verdicts": [
                        {
                            "claim_id": "C1",
                            "verdict": "supported",
                            "reason": "Cifra y horizonte coinciden con la fuente.",
                            "evidence_verdicts": [{"evidence_id": "E1", "relation": "supports"}],
                        },
                        {
                            "claim_id": "C2",
                            "verdict": "supported",
                            "reason": "Tasa y año 2050 respaldados exactamente.",
                            "evidence_verdicts": [{"evidence_id": "E2", "relation": "supports"}],
                        },
                    ]
                }
            },
        ]

        fake_gen = FakeGrounderGenerator(responses)
        grounder = OllamaGrounder(generator=fake_gen)
        res = grounder.answer(query="¿Cuál es la proyección de gasto y dependencia para 2050?", hits=self.hits,
                              original_query="¿Qué dice el informe y cuál es mi pensión individual?")
        for messages, _ in fake_gen.history:
            user_data = json.loads(next(m["content"] for m in messages if m["role"] == "user"))
            self.assertNotIn("pregunta_original", user_data)

        self.assertEqual(res["status"], "answered")
        self.assertEqual(res["reason"], "")
        self.assertEqual(len(res["claims"]), 2)
        self.assertEqual(res["grounding"]["repair_attempts"], 0)
        self.assertEqual(len(res["grounding"]["trace"]), 3)
        self.assertEqual(fake_gen.call_count, 3)

        # Validación exhaustiva en frontera
        val = validate_grounded_result(res, self.hits, fake_gen.digest)
        self.assertEqual(val["status"], "answered")
        self.assertEqual(len(val["claims"]), 2)
        self.assertIn("source_id", val["claims"][0]["evidence"][0])

    def test_pruning_header_quote(self) -> None:
        """Poda de encabezado irrelevante: claim apoyada retira cita irrelevante."""
        header_quote = "CAPÍTULO 3. PROYECCIONES DEMOGRÁFICAS Y GASTO PÚBLICO EN PENSIONES."
        substantive_quote = "El gasto en pensiones contributivas de jubilación alcanzará el 16,1% del PIB en el año 2050"

        responses = [
            # 1. selection
            {
                "payload": {
                    "status": "selected",
                    "evidence": [
                        {"source_id": "S3", "quote": header_quote},
                        {"source_id": "S1", "quote": substantive_quote},
                    ],
                    "reason": "",
                }
            },
            # 2. generation (asigna E1 y E2 a C1)
            {
                "payload": {
                    "status": "answered",
                    "claims": [
                        {
                            "claim_id": "C1",
                            "text": "El gasto en pensiones llegará al 16,1% del PIB en 2050.",
                            "evidence_ids": ["E1", "E2"],
                        }
                    ],
                    "reason": "",
                }
            },
            # 3. verification (E1 es irrelevant, E2 supports)
            {
                "payload": {
                    "verdicts": [
                        {
                            "claim_id": "C1",
                            "verdict": "supported",
                            "reason": "E2 aporta soporte completo; E1 es encabezado.",
                            "evidence_verdicts": [
                                {"evidence_id": "E1", "relation": "irrelevant"},
                                {"evidence_id": "E2", "relation": "supports"},
                            ],
                        }
                    ]
                }
            },
        ]

        fake_gen = FakeGrounderGenerator(responses)
        grounder = OllamaGrounder(generator=fake_gen)
        res = grounder.answer(query="¿Cuál es el gasto en 2050?", hits=self.hits)

        self.assertEqual(res["status"], "answered")
        # En claims públicas E1 debe haber sido podada; solo queda E2 (S1)
        published_evidence = res["claims"][0]["evidence"]
        self.assertEqual(len(published_evidence), 1)
        self.assertEqual(published_evidence[0]["source_id"], "S1")
        self.assertEqual(published_evidence[0]["quote"], substantive_quote)

        val = validate_grounded_result(res, self.hits, fake_gen.digest)
        self.assertEqual(val["status"], "answered")

    def test_multipension_quote_does_not_support_2055(self) -> None:
        """Cita multipensión de 2023 no apoya afirmación de 2055."""
        quote2023 = "En el año 2023, las pensiones del sistema de Seguridad Social representaron un 12,9% del PIB"

        responses = [
            # 1. selection
            {
                "payload": {
                    "status": "selected",
                    "evidence": [{"source_id": "S4", "quote": quote2023}],
                    "reason": "",
                }
            },
            # 2. generation
            {
                "payload": {
                    "status": "answered",
                    "claims": [
                        {
                            "claim_id": "C1",
                            "text": "El gasto en pensiones en 2055 será del 12,9% del PIB.",
                            "evidence_ids": ["E1"],
                        }
                    ],
                    "reason": "",
                }
            },
            # 3. verification: rechaza porque es 2023, no 2055
            {
                "payload": {
                    "verdicts": [
                        {
                            "claim_id": "C1",
                            "verdict": "insufficient_evidence",
                            "reason": "La cita refiere a 2023 observado, no a 2055 proyectado.",
                            "evidence_verdicts": [{"evidence_id": "E1", "relation": "irrelevant"}],
                        }
                    ]
                }
            },
            # 4. repair_selection (no encuentra cita para 2055 en los hits)
            {
                "payload": {
                    "status": "insufficient_evidence",
                    "evidence": [],
                    "reason": "No existen datos para 2055 en las fuentes disponibles.",
                }
            },
            # 5. repair_generation
            {
                "payload": {
                    "repairs": [
                        {
                            "claim_id": "C1",
                            "status": "insufficient_evidence",
                            "text": "",
                            "evidence_ids": [],
                            "reason": "Sin datos proyectados para 2055.",
                        }
                    ]
                }
            },
        ]

        fake_gen = FakeGrounderGenerator(responses)
        grounder = OllamaGrounder(generator=fake_gen)
        res = grounder.answer(query="¿Cuál es la previsión para 2055?", hits=self.hits)

        self.assertEqual(res["status"], "insufficient_evidence")
        self.assertEqual(res["claims"], [])
        self.assertIn("C1", res["reason"])
        self.assertEqual(res["grounding"]["repair_attempts"], 1)
        # Reverification no debe llamarse porque ninguna reparación fue answered
        self.assertEqual(len(res["grounding"]["trace"]), 5)
        self.assertEqual(fake_gen.call_count, 5)

        val = validate_grounded_result(res, self.hits, fake_gen.digest)
        self.assertEqual(val["status"], "insufficient_evidence")

    def test_complementarity_supported_vs_single_partial_rejected(self) -> None:
        """Aclaración lógica: dos partial se complementan; un único partial no puede ser supported."""
        claims = [
            {"claim_id": "C1", "text": "Afirmación complementaria", "evidence_ids": ["E1", "E2"]},
            {"claim_id": "C2", "text": "Afirmación con único soporte parcial", "evidence_ids": ["E1"]},
        ]
        evidence = [
            {"evidence_id": "E1", "source_id": "S1", "quote": "Cita parcial primera parte"},
            {"evidence_id": "E2", "source_id": "S2", "quote": "Cita parcial segunda parte"},
        ]

        # Subtest 1: Complementariedad válida (>= 2 partial sin contradicts -> supported)
        with self.subTest("complementariedad_valida"):
            payload_ok = {
                "verdicts": [
                    {
                        "claim_id": "C1",
                        "verdict": "supported",
                        "reason": "Dos evidencias parciales se complementan.",
                        "evidence_verdicts": [
                            {"evidence_id": "E1", "relation": "partial"},
                            {"evidence_id": "E2", "relation": "partial"},
                        ],
                    }
                ]
            }
            validated = validate_verification_payload(payload_ok, [claims[0]], evidence)
            self.assertEqual(validated["verdicts"][0]["verdict"], "supported")

        # Subtest 2: Único partial no puede sostener supported
        with self.subTest("unico_partial_rechazado_para_supported"):
            payload_inval = {
                "verdicts": [
                    {
                        "claim_id": "C2",
                        "verdict": "supported",
                        "reason": "Intento inválido de apoyar con único partial.",
                        "evidence_verdicts": [
                            {"evidence_id": "E1", "relation": "partial"},
                        ],
                    }
                ]
            }
            with self.assertRaises(ValueError) as ctx:
                validate_verification_payload(payload_inval, [claims[1]], evidence)
            self.assertIn("requires at least one 'supports' or at least two 'partial'", str(ctx.exception))

        # Subtest 3: Un partial y un irrelevant tampoco puede ser supported
        with self.subTest("partial_e_irrelevant_rechazado_para_supported"):
            payload_partial_irrel = {
                "verdicts": [
                    {
                        "claim_id": "C1",
                        "verdict": "supported",
                        "reason": "Un partial y un irrelevant.",
                        "evidence_verdicts": [
                            {"evidence_id": "E1", "relation": "partial"},
                            {"evidence_id": "E2", "relation": "irrelevant"},
                        ],
                    }
                ]
            }
            with self.assertRaises(ValueError):
                validate_verification_payload(payload_partial_irrel, [claims[0]], evidence)

    def test_repair_with_passage_not_selected_initially(self) -> None:
        """Reparación exitosa seleccionando un pasaje no seleccionado en la etapa inicial."""
        quote1 = "El gasto en pensiones contributivas de jubilación alcanzará el 16,1% del PIB en el año 2050"
        quote2 = "La tasa de dependencia superará el 53% en el horizonte proyectado para el año 2050"

        responses = [
            # 1. selection (solo selecciona S1)
            {
                "payload": {
                    "status": "selected",
                    "evidence": [{"source_id": "S1", "quote": quote1}],
                    "reason": "",
                }
            },
            # 2. generation (afirma sobre dependencia usando S1 por error)
            {
                "payload": {
                    "status": "answered",
                    "claims": [
                        {
                            "claim_id": "C1",
                            "text": "La tasa de dependencia superará el 53% en 2050.",
                            "evidence_ids": ["E1"],
                        }
                    ],
                    "reason": "",
                }
            },
            # 3. verification (falla porque E1 no tiene dependencia)
            {
                "payload": {
                    "verdicts": [
                        {
                            "claim_id": "C1",
                            "verdict": "insufficient_evidence",
                            "reason": "E1 habla de gasto de jubilación, no de tasa de dependencia.",
                            "evidence_verdicts": [{"evidence_id": "E1", "relation": "irrelevant"}],
                        }
                    ]
                }
            },
            # 4. repair_selection (ahora selecciona S2 que no estaba inicialmente)
            {
                "payload": {
                    "status": "selected",
                    "evidence": [{"source_id": "S2", "quote": quote2}],
                    "reason": "",
                }
            },
            # 5. repair_generation (corrige usando la nueva evidencia E2)
            {
                "payload": {
                    "repairs": [
                        {
                            "claim_id": "C1",
                            "status": "answered",
                            "text": "La tasa de dependencia superará el 53% en 2050.",
                            "evidence_ids": ["E2"],
                            "reason": "",
                        }
                    ]
                }
            },
            # 6. reverification (verifica C1 con E2 -> supported)
            {
                "payload": {
                    "verdicts": [
                        {
                            "claim_id": "C1",
                            "verdict": "supported",
                            "reason": "E2 respalda exactamente la tasa de dependencia del 53% en 2050.",
                            "evidence_verdicts": [{"evidence_id": "E2", "relation": "supports"}],
                        }
                    ]
                }
            },
        ]

        fake_gen = FakeGrounderGenerator(responses)
        grounder = OllamaGrounder(generator=fake_gen)
        res = grounder.answer(query="¿Cuál es la tasa de dependencia en 2050?", hits=self.hits)

        self.assertEqual(res["status"], "answered")
        self.assertEqual(res["grounding"]["repair_attempts"], 1)
        self.assertEqual(len(res["grounding"]["selected_evidence"]), 1)
        self.assertEqual(len(res["grounding"]["repair_evidence"]), 1)
        self.assertEqual(res["grounding"]["repair_evidence"][0]["evidence_id"], "E2")
        self.assertEqual(len(res["grounding"]["trace"]), 6)
        self.assertEqual(fake_gen.call_count, 6)

        val = validate_grounded_result(res, self.hits, fake_gen.digest)
        self.assertEqual(val["status"], "answered")
        self.assertEqual(val["claims"][0]["evidence"][0]["source_id"], "S2")

    def test_persistent_failure_forces_insufficient_and_preserves_accepted(self) -> None:
        """Fallo persistente tras reparación: estado insufficient pero conserva aceptadas."""
        quote1 = "El gasto en pensiones contributivas de jubilación alcanzará el 16,1% del PIB en el año 2050"
        quote4 = "En el año 2023, las pensiones del sistema de Seguridad Social representaron un 12,9% del PIB"

        responses = [
            # 1. selection
            {
                "payload": {
                    "status": "selected",
                    "evidence": [
                        {"source_id": "S1", "quote": quote1},
                        {"source_id": "S4", "quote": quote4},
                    ],
                    "reason": "",
                }
            },
            # 2. generation: C1 (apoyada) y C2 (sobre 2070 pero con cita de 2023)
            {
                "payload": {
                    "status": "answered",
                    "claims": [
                        {
                            "claim_id": "C1",
                            "text": "El gasto en jubilación llegará al 16,1% del PIB en 2050.",
                            "evidence_ids": ["E1"],
                        },
                        {
                            "claim_id": "C2",
                            "text": "El gasto en 2070 será del 12,9% del PIB.",
                            "evidence_ids": ["E2"],
                        },
                    ],
                    "reason": "",
                }
            },
            # 3. verification: C1 supported, C2 contradicted
            {
                "payload": {
                    "verdicts": [
                        {
                            "claim_id": "C1",
                            "verdict": "supported",
                            "reason": "Dato 2050 correcto.",
                            "evidence_verdicts": [{"evidence_id": "E1", "relation": "supports"}],
                        },
                        {
                            "claim_id": "C2",
                            "verdict": "contradicted",
                            "reason": "La cita contradice la fecha: refiere a 2023.",
                            "evidence_verdicts": [{"evidence_id": "E2", "relation": "contradicts"}],
                        },
                    ]
                }
            },
            # 4. repair_selection (intenta buscar 2070)
            {
                "payload": {
                    "status": "insufficient_evidence",
                    "evidence": [],
                    "reason": "No se encontraron datos de 2070.",
                }
            },
            # 5. repair_generation (declara insuficiente para C2)
            {
                "payload": {
                    "repairs": [
                        {
                            "claim_id": "C2",
                            "status": "insufficient_evidence",
                            "text": "",
                            "evidence_ids": [],
                            "reason": "Sin datos de 2070 en el corpus.",
                        }
                    ]
                }
            },
        ]

        fake_gen = FakeGrounderGenerator(responses)
        grounder = OllamaGrounder(generator=fake_gen)
        res = grounder.answer(query="¿Gasto en 2050 y 2070?", hits=self.hits)

        self.assertEqual(res["status"], "insufficient_evidence")
        self.assertEqual(res["claims"], [])
        self.assertIn("C2", res["reason"])
        self.assertEqual(res["grounding"]["repair_attempts"], 1)

        # C1 aceptada permanece en accepted_claims
        accepted = res["grounding"]["accepted_claims"]
        self.assertEqual(len(accepted), 1)
        self.assertEqual(accepted[0]["claim_id"], "C1")

        # C2 permanece en rejected_claims
        rejected = res["grounding"]["rejected_claims"]
        self.assertEqual(len(rejected), 1)
        self.assertEqual(rejected[0]["claim_id"], "C2")

        val = validate_grounded_result(res, self.hits, fake_gen.digest)
        self.assertEqual(val["status"], "insufficient_evidence")

    def test_early_abstentions(self) -> None:
        """Abstenciones tempranas en selection o generation no intentan reparar sin objetivos."""
        with self.subTest("abstencion_en_selection"):
            fake_gen = FakeGrounderGenerator(
                [
                    {
                        "payload": {
                            "status": "insufficient_evidence",
                            "evidence": [],
                            "reason": "Fuentes sin relación con la consulta.",
                        }
                    }
                ]
            )
            grounder = OllamaGrounder(generator=fake_gen)
            res = grounder.answer(query="¿Pregunta sin evidencia?", hits=self.hits)
            self.assertEqual(res["status"], "insufficient_evidence")
            self.assertEqual(len(res["grounding"]["trace"]), 1)
            self.assertEqual(fake_gen.call_count, 1)
            val = validate_grounded_result(res, self.hits, fake_gen.digest)
            self.assertEqual(val["status"], "insufficient_evidence")

        with self.subTest("abstencion_en_generation"):
            quote1 = "El gasto en pensiones contributivas de jubilación alcanzará el 16,1% del PIB en el año 2050"
            fake_gen = FakeGrounderGenerator(
                [
                    {
                        "payload": {
                            "status": "selected",
                            "evidence": [{"source_id": "S1", "quote": quote1}],
                            "reason": "",
                        }
                    },
                    {
                        "payload": {
                            "status": "insufficient_evidence",
                            "claims": [],
                            "reason": "Cita insuficiente para responder a la necesidad específica.",
                        }
                    },
                ]
            )
            grounder = OllamaGrounder(generator=fake_gen)
            res = grounder.answer(query="¿Pregunta específica?", hits=self.hits)
            self.assertEqual(res["status"], "insufficient_evidence")
            self.assertEqual(len(res["grounding"]["trace"]), 2)
            self.assertEqual(fake_gen.call_count, 2)
            val = validate_grounded_result(res, self.hits, fake_gen.digest)
            self.assertEqual(val["status"], "insufficient_evidence")

    def test_maximum_one_repair_turn_and_no_loop(self) -> None:
        """Garantiza que existe como máximo una sola vuelta de reparación (repair_attempts <= 1)."""
        quote1 = "El gasto en pensiones contributivas de jubilación alcanzará el 16,1% del PIB en el año 2050"
        responses = [
            # 1. selection
            {"payload": {"status": "selected", "evidence": [{"source_id": "S1", "quote": quote1}], "reason": ""}},
            # 2. generation
            {"payload": {"status": "answered", "claims": [{"claim_id": "C1", "text": "Texto", "evidence_ids": ["E1"]}], "reason": ""}},
            # 3. verification (falla)
            {"payload": {"verdicts": [{"claim_id": "C1", "verdict": "contradicted", "reason": "Fallo 1", "evidence_verdicts": [{"evidence_id": "E1", "relation": "contradicts"}]}]}},
            # 4. repair_selection
            {"payload": {"status": "selected", "evidence": [{"source_id": "S1", "quote": quote1}], "reason": ""}},
            # 5. repair_generation
            {"payload": {"repairs": [{"claim_id": "C1", "status": "answered", "text": "Texto 2", "evidence_ids": ["E2"], "reason": ""}]}},
            # 6. reverification (falla de nuevo)
            {"payload": {"verdicts": [{"claim_id": "C1", "verdict": "contradicted", "reason": "Fallo persistente", "evidence_verdicts": [{"evidence_id": "E2", "relation": "contradicts"}]}]}},
        ]
        fake_gen = FakeGrounderGenerator(responses)
        grounder = OllamaGrounder(generator=fake_gen)
        res = grounder.answer(query="Consulta", hits=self.hits)

        self.assertEqual(res["status"], "insufficient_evidence")
        self.assertEqual(res["grounding"]["repair_attempts"], 1)
        self.assertEqual(fake_gen.call_count, 6)
        validate_grounded_result(res, self.hits, fake_gen.digest)

    def test_zero_omitted_targets_extra_ids_cross_citations(self) -> None:
        """Rechaza objetivos omitidos, IDs no autorizados y citas cruzadas."""
        failed_claims = [
            {"claim_id": "C1", "text": "Texto C1", "reason": "Fallo"},
            {"claim_id": "C2", "text": "Texto C2", "reason": "Fallo"},
        ]
        repair_evidence = [{"evidence_id": "E3", "source_id": "S1", "quote": "Cita repair"}]

        with self.subTest("omite_objetivo_fallido"):
            # Solo devuelve C1, omitiendo C2
            payload_omitted = {
                "repairs": [
                    {"claim_id": "C1", "status": "answered", "text": "Reparado", "evidence_ids": ["E3"], "reason": ""}
                ]
            }
            with self.assertRaises(ValueError):
                validate_repair_generation_payload(payload_omitted, failed_claims, repair_evidence)

        with self.subTest("id_extra_o_no_correspondiente"):
            payload_extra = {
                "repairs": [
                    {"claim_id": "C1", "status": "answered", "text": "Reparado", "evidence_ids": ["E3"], "reason": ""},
                    {"claim_id": "C99", "status": "answered", "text": "Extra", "evidence_ids": ["E3"], "reason": ""},
                ]
            }
            with self.assertRaises(ValueError):
                validate_repair_generation_payload(payload_extra, failed_claims, repair_evidence)

        with self.subTest("cita_no_existente_en_catalogo"):
            payload_bad_ref = {
                "repairs": [
                    {"claim_id": "C1", "status": "answered", "text": "Reparado", "evidence_ids": ["E999"], "reason": ""},
                    {"claim_id": "C2", "status": "insufficient_evidence", "text": "", "evidence_ids": [], "reason": "No hay"},
                ]
            }
            with self.assertRaises(ValueError):
                validate_repair_generation_payload(payload_bad_ref, failed_claims, repair_evidence)


class TestGroundingAdversarialAndLimits(unittest.TestCase):
    """Tests adversos de digests, modelos, presupuestos, esquemas y consistencia lógica."""

    def setUp(self) -> None:
        self.text1 = (
            "El gasto en pensiones contributivas de jubilación alcanzará el 16,1% del PIB en el año 2050 "
            "según las proyecciones oficiales de la AIReF publicadas en 2025."
        )
        self.hit1 = _make_hit(1, self.text1)
        self.hits = [self.hit1]

    def test_digest_mutation_aborts_immediately(self) -> None:
        """Mutación de digest antes o durante la generación aborta con RuntimeError."""
        quote1 = "El gasto en pensiones contributivas de jubilación alcanzará el 16,1% del PIB en el año 2050"
        responses = [
            {"payload": {"status": "selected", "evidence": [{"source_id": "S1", "quote": quote1}], "reason": ""}},
        ]
        # Cambia digest en la llamada 1
        fake_gen = FakeGrounderGenerator(responses, digest_mutations={1: "corrupted_digest"})
        grounder = OllamaGrounder(generator=fake_gen)
        with self.assertRaises(RuntimeError) as ctx:
            grounder.answer(query="¿Gasto?", hits=self.hits, expected_digest="test_gen_digest_sha256")
        self.assertIn("digest", str(ctx.exception).lower())

    def test_messages_budget_guard_exceeded(self) -> None:
        """Carga que excede 6500 bytes UTF-8 aborta antes de contactar con el modelo."""
        huge_query = "¿Consulta? " + ("A" * 7000)
        with self.assertRaises(ValueError) as ctx:
            build_selection_messages(huge_query, self.hits)
        self.assertIn("exceeds budget", str(ctx.exception))

    def test_quote_budget_and_length_invariants(self) -> None:
        """Validación de límites de longitud de citas (12..700 chars) y presupuesto UTF-8 (<=2400)."""
        with self.subTest("cita_demasiado_corta"):
            payload_short = {
                "status": "selected",
                "evidence": [{"source_id": "S1", "quote": "Corta"}],
                "reason": "",
            }
            with self.assertRaises(ValueError):
                validate_selection_payload(payload_short, self.hits)

        with self.subTest("cita_inexistente_en_fuente"):
            payload_fake = {
                "status": "selected",
                "evidence": [{"source_id": "S1", "quote": "Esta frase de prueba no existe en la fuente en absoluto."}],
                "reason": "",
            }
            with self.assertRaises(ValueError):
                validate_selection_payload(payload_fake, self.hits)

        with self.subTest("presupuesto_total_citas_excedido"):
            # Cita de 650 caracteres repetida en 4 fuentes artificiales
            long_text = ("Texto largo oficial sobre pensiones públicas que cumple los requisitos. " * 15)[:650]
            hit_long1 = _make_hit(1, long_text)
            hit_long2 = _make_hit(2, long_text)
            hit_long3 = _make_hit(3, long_text)
            hit_long4 = _make_hit(4, long_text)
            hits_long = [hit_long1, hit_long2, hit_long3, hit_long4]

            # 4 * 650 = 2600 bytes > 2400 bytes límite
            payload_over_budget = {
                "status": "selected",
                "evidence": [
                    {"source_id": "S1", "quote": long_text},
                    {"source_id": "S2", "quote": long_text},
                    {"source_id": "S3", "quote": long_text},
                    {"source_id": "S4", "quote": long_text},
                ],
                "reason": "",
            }
            with self.assertRaises(ValueError) as ctx:
                validate_selection_payload(payload_over_budget, hits_long)
            self.assertIn("exceeds budget", str(ctx.exception))

    def test_stats_validation_prompt_eval_count_limit(self) -> None:
        """Estadística de prompt_eval_count superior a 6912 aborta la llamada."""
        quote1 = "El gasto en pensiones contributivas de jubilación alcanzará el 16,1% del PIB en el año 2050"
        responses = [
            {
                "payload": {"status": "selected", "evidence": [{"source_id": "S1", "quote": quote1}], "reason": ""},
                "stats": {"prompt_eval_count": 7000, "eval_count": 50, "eval_duration": 1000, "total_duration": 2000},
            }
        ]
        fake_gen = FakeGrounderGenerator(responses)
        grounder = OllamaGrounder(generator=fake_gen)
        with self.assertRaises(ValueError) as ctx:
            grounder.answer(query="¿Gasto?", hits=self.hits)
        self.assertIn("exceeds maximum allowed budget", str(ctx.exception))

    def test_validate_grounded_result_falsification_checks(self) -> None:
        """Rechazo de sumas de stats alteradas, hashes falsificados y parámetros inválidos."""
        quote1 = "El gasto en pensiones contributivas de jubilación alcanzará el 16,1% del PIB en el año 2050"
        responses = [
            {"payload": {"status": "selected", "evidence": [{"source_id": "S1", "quote": quote1}], "reason": ""}},
            {"payload": {"status": "answered", "claims": [{"claim_id": "C1", "text": "Gasto 16,1%", "evidence_ids": ["E1"]}], "reason": ""}},
            {"payload": {"verdicts": [{"claim_id": "C1", "verdict": "supported", "reason": "OK", "evidence_verdicts": [{"evidence_id": "E1", "relation": "supports"}]}]}},
        ]
        fake_gen = FakeGrounderGenerator(responses)
        grounder = OllamaGrounder(generator=fake_gen)
        res = grounder.answer(query="¿Gasto?", hits=self.hits)
        digest = fake_gen.digest

        with self.subTest("resultado_valido_pasa"):
            val = validate_grounded_result(res, self.hits, digest)
            self.assertEqual(val["status"], "answered")

        with self.subTest("suma_stats_falsificada"):
            bad_res = copy.deepcopy(res)
            bad_res["stats"]["prompt_eval_count"] += 999
            with self.assertRaises(ValueError) as ctx:
                validate_grounded_result(bad_res, self.hits, digest)
            self.assertIn("Stats prompt_eval_count sum mismatch", str(ctx.exception))

        with self.subTest("prompt_sha256_agregado_falsificado"):
            bad_res = copy.deepcopy(res)
            bad_res["prompt_sha256"] = "a" * 64
            with self.assertRaises(ValueError) as ctx:
                validate_grounded_result(bad_res, self.hits, digest)
            self.assertIn("prompt_sha256 aggregate mismatch", str(ctx.exception))

        with self.subTest("parametro_con_nan_rechazado"):
            bad_res = copy.deepcopy(res)
            bad_res["grounding"]["trace"][0]["parameters"]["temperature"] = float("nan")
            with self.assertRaises(ValueError):
                validate_grounded_result(bad_res, self.hits, digest)

    def test_standalone_verify_api(self) -> None:
        """La función pública verify() opera independientemente con validación estricta."""
        claims = [{"claim_id": "C1", "text": "Afirmación verificada", "evidence_ids": ["E1"]}]
        evidence = [{"evidence_id": "E1", "source_id": "S1", "quote": "Cita de soporte"}]

        responses = [
            {
                "payload": {
                    "verdicts": [
                        {
                            "claim_id": "C1",
                            "verdict": "supported",
                            "reason": "Respaldo directo verificado.",
                            "evidence_verdicts": [{"evidence_id": "E1", "relation": "supports"}],
                        }
                    ]
                }
            }
        ]
        fake_gen = FakeGrounderGenerator(responses)
        v_res = verify(claims, evidence, expected_digest=fake_gen.digest, generator=fake_gen)

        self.assertIn("verdicts", v_res)
        self.assertEqual(v_res["verdicts"][0]["verdict"], "supported")
        self.assertEqual(v_res["digest"], fake_gen.digest)
        self.assertEqual(len(v_res["prompt_sha256"]), 64)


class TestValidateGroundedResultMutationsAndFlows(unittest.TestCase):
    """Tests exhaustivos de mutación y validación de flujos para validate_grounded_result."""

    def setUp(self) -> None:
        self.quote1 = "El gasto en pensiones contributivas de jubilación alcanzará el 16,1% del PIB en el año 2050"
        self.quote2 = "El gasto en pensiones se situará en el 14,7% del PIB en 2070 según el escenario base"
        self.hit1 = _make_hit(1, self.quote1)
        self.hit2 = _make_hit(2, self.quote2)
        self.hits = [self.hit1, self.hit2]

    def _make_first_pass_answered_result(self) -> tuple[dict, FakeGrounderGenerator]:
        responses = [
            {"payload": {"status": "selected", "evidence": [{"source_id": "S1", "quote": self.quote1}], "reason": ""}},
            {"payload": {"status": "answered", "claims": [{"claim_id": "C1", "text": "Gasto 16,1% en 2050.", "evidence_ids": ["E1"]}], "reason": ""}},
            {"payload": {"verdicts": [{"claim_id": "C1", "verdict": "supported", "reason": "Soporte completo.", "evidence_verdicts": [{"evidence_id": "E1", "relation": "supports"}]}]}},
        ]
        fake_gen = FakeGrounderGenerator(responses)
        grounder = OllamaGrounder(generator=fake_gen)
        res = grounder.answer(query="¿Gasto en 2050?", hits=self.hits)
        return res, fake_gen

    def _make_repaired_answered_result(self) -> tuple[dict, FakeGrounderGenerator]:
        responses = [
            # 1. selection
            {"payload": {"status": "selected", "evidence": [{"source_id": "S1", "quote": self.quote1}], "reason": ""}},
            # 2. generation
            {"payload": {"status": "answered", "claims": [{"claim_id": "C1", "text": "Gasto 14,7% en 2050.", "evidence_ids": ["E1"]}], "reason": ""}},
            # 3. verification (falla inicial)
            {"payload": {"verdicts": [{"claim_id": "C1", "verdict": "contradicted", "reason": "La cita indica 16,1%, no 14,7%.", "evidence_verdicts": [{"evidence_id": "E1", "relation": "contradicts"}]}]}},
            # 4. repair_selection (busca en hits y añade S2)
            {"payload": {"status": "selected", "evidence": [{"source_id": "S2", "quote": self.quote2}], "reason": ""}},
            # 5. repair_generation (corrige afirmación con E2)
            {"payload": {"repairs": [{"claim_id": "C1", "status": "answered", "text": "Gasto 14,7% en 2070 según el escenario base.", "evidence_ids": ["E2"], "reason": ""}]}},
            # 6. reverification (verifica C1 con E2 -> supported)
            {"payload": {"verdicts": [{"claim_id": "C1", "verdict": "supported", "reason": "Corregido conforme a E2.", "evidence_verdicts": [{"evidence_id": "E2", "relation": "supports"}]}]}},
        ]
        fake_gen = KeyedGrounderGenerator(responses)
        grounder = OllamaGrounder(generator=fake_gen)
        res = grounder.answer(query="¿Gasto en 2070?", hits=self.hits)
        return res, fake_gen

    def test_valid_flow_first_pass_answered(self) -> None:
        """Flujo válido en primera pasada con 3 etapas valida exitosamente."""
        res, fake_gen = self._make_first_pass_answered_result()
        val = validate_grounded_result(res, self.hits, fake_gen.digest, expected_model=DEFAULT_GENERATION_MODEL)
        self.assertEqual(val["status"], "answered")
        self.assertEqual(len(val["claims"]), 1)
        self.assertEqual(val["grounding"]["repair_attempts"], 0)
        self.assertEqual(len(val["grounding"]["trace"]), 3)

    def test_valid_flow_after_repair_answered(self) -> None:
        """Flujo válido con reparación exitosa en 6 etapas valida correctamente."""
        res, fake_gen = self._make_repaired_answered_result()
        val = validate_grounded_result(res, self.hits, fake_gen.digest, expected_model=DEFAULT_GENERATION_MODEL)
        self.assertEqual(val["status"], "answered")
        self.assertEqual(val["grounding"]["repair_attempts"], 1)
        self.assertEqual(len(val["grounding"]["trace"]), 6)
        self.assertEqual(len(val["grounding"]["accepted_claims"]), 1)
        self.assertEqual(len(val["grounding"]["rejected_claims"]), 0)
        for index, ref in ((2, "E1"), (5, "E2")):
            schema = fake_gen.history[index][1]
            for branch in schema["properties"]["verdicts"]["properties"]["C1"]["anyOf"]:
                self.assertEqual(branch["properties"]["evidence_verdicts"]["required"], [ref])

    def test_initial_claims_immutability_during_repair(self) -> None:
        """initial_claims permanece inmutable y sin clave 'reason' tras una reparación."""
        res, _ = self._make_repaired_answered_result()
        initial_claims = res["grounding"]["initial_claims"]
        self.assertEqual(len(initial_claims), 1)
        # Cada claim original solo debe tener 'claim_id', 'text', 'evidence_ids'
        self.assertEqual(set(initial_claims[0].keys()), {"claim_id", "text", "evidence_ids"})
        self.assertNotIn("reason", initial_claims[0])

    def test_valid_flow_early_selection_abstention(self) -> None:
        """Abstención temprana en selection con 1 sola etapa valida sin historial inventado."""
        responses = [
            {"payload": {"status": "insufficient_evidence", "evidence": [], "reason": "No hay evidencia sobre el tema."}}
        ]
        fake_gen = FakeGrounderGenerator(responses)
        grounder = OllamaGrounder(generator=fake_gen)
        res = grounder.answer(query="¿Tema ausente?", hits=self.hits)

        val = validate_grounded_result(res, self.hits, fake_gen.digest, expected_model=DEFAULT_GENERATION_MODEL)
        self.assertEqual(val["status"], "insufficient_evidence")
        self.assertEqual(val["claims"], [])
        self.assertEqual(len(val["grounding"]["trace"]), 1)
        self.assertEqual(val["grounding"]["trace"][0]["stage"], "selection")
        self.assertEqual(val["grounding"]["selected_evidence"], [])
        self.assertEqual(val["grounding"]["initial_claims"], [])
        self.assertEqual(val["grounding"]["final_verdicts"], [])

    def test_valid_flow_generation_abstention(self) -> None:
        """Abstención en generation con 2 etapas valida sin veredictos ni reparaciones."""
        responses = [
            {"payload": {"status": "selected", "evidence": [{"source_id": "S1", "quote": self.quote1}], "reason": ""}},
            {"payload": {"status": "insufficient_evidence", "claims": [], "reason": "Las fuentes no permiten formular afirmaciones certeras."}},
        ]
        fake_gen = FakeGrounderGenerator(responses)
        grounder = OllamaGrounder(generator=fake_gen)
        res = grounder.answer(query="¿Pregunta ambigua?", hits=self.hits)

        val = validate_grounded_result(res, self.hits, fake_gen.digest, expected_model=DEFAULT_GENERATION_MODEL)
        self.assertEqual(val["status"], "insufficient_evidence")
        self.assertEqual(val["claims"], [])
        self.assertEqual(len(val["grounding"]["trace"]), 2)
        self.assertEqual(val["grounding"]["trace"][0]["stage"], "selection")
        self.assertEqual(val["grounding"]["trace"][1]["stage"], "generation")
        self.assertEqual(val["grounding"]["initial_claims"], [])

    def test_mutation_public_text_changed_fails(self) -> None:
        """Mutación de texto en claims públicas es detectada y rechazada."""
        res, fake_gen = self._make_first_pass_answered_result()
        bad_res = copy.deepcopy(res)
        bad_res["claims"][0]["text"] = "Texto público alterado falsamente sin justificación."
        with self.assertRaises(ValueError):
            validate_grounded_result(bad_res, self.hits, fake_gen.digest)

    def test_mutation_final_verdicts_empty_fails(self) -> None:
        """final_verdicts vacío en resultado con afirmaciones contestadas es rechazado."""
        res, fake_gen = self._make_first_pass_answered_result()
        bad_res = copy.deepcopy(res)
        bad_res["grounding"]["final_verdicts"] = []
        with self.assertRaises(ValueError):
            validate_grounded_result(bad_res, self.hits, fake_gen.digest)

    def test_mutation_initial_or_accepted_claims_empty_fails(self) -> None:
        """initial_claims o accepted_claims vacíos en respuesta answered son rechazados."""
        res, fake_gen = self._make_first_pass_answered_result()

        bad_res1 = copy.deepcopy(res)
        bad_res1["grounding"]["initial_claims"] = []
        with self.assertRaises(ValueError):
            validate_grounded_result(bad_res1, self.hits, fake_gen.digest)

        bad_res2 = copy.deepcopy(res)
        bad_res2["grounding"]["accepted_claims"] = []
        with self.assertRaises(ValueError):
            validate_grounded_result(bad_res2, self.hits, fake_gen.digest)

    def test_mutation_omit_verification_recalculating_hashes_and_stats_fails(self) -> None:
        """Omitir la etapa verification recalculando hashes y stats es detectado por stages."""
        res, fake_gen = self._make_first_pass_answered_result()
        bad_res = copy.deepcopy(res)
        # Cortar la traza dejando solo selection y generation
        bad_res["grounding"]["trace"] = bad_res["grounding"]["trace"][:2]
        t_hashes = [t["prompt_sha256"] for t in bad_res["grounding"]["trace"]]
        bad_res["prompt_sha256"] = hashlib.sha256(
            json.dumps(t_hashes, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        bad_res["stats"]["prompt_eval_count"] = sum(t["stats"]["prompt_eval_count"] for t in bad_res["grounding"]["trace"])
        bad_res["stats"]["eval_count"] = sum(t["stats"]["eval_count"] for t in bad_res["grounding"]["trace"])
        bad_res["stats"]["eval_duration"] = sum(t["stats"]["eval_duration"] for t in bad_res["grounding"]["trace"])
        bad_res["stats"]["total_duration"] = sum(t["stats"]["total_duration"] for t in bad_res["grounding"]["trace"])

        with self.assertRaises(ValueError) as ctx:
            validate_grounded_result(bad_res, self.hits, fake_gen.digest)
        self.assertIn("trace stages count mismatch", str(ctx.exception))

    def test_mutation_model_changed_to_other_allowed_fails(self) -> None:
        """Cambio de modelo a otro permitido es detectado contra expected_model o inconsistencia interna."""
        res, fake_gen = self._make_first_pass_answered_result()

        # 1. Con expected_model explícito
        bad_res1 = copy.deepcopy(res)
        bad_res1["grounding"]["trace"][0]["model"] = "qwen3.5:4b-q4_K_M"
        with self.assertRaises(ValueError):
            validate_grounded_result(bad_res1, self.hits, fake_gen.digest, expected_model="qwen3.5:9b")

        # 2. Con expected_model=None infiriendo de la traza: primer item es 4B pero segundo es 9B
        with self.assertRaises(ValueError):
            validate_grounded_result(bad_res1, self.hits, fake_gen.digest)

    def test_mutation_parameters_temperature_1_fails(self) -> None:
        """Parámetros de traza con temperature=1 no coinciden con DEFAULT_PARAMETERS y fallan."""
        res, fake_gen = self._make_first_pass_answered_result()
        bad_res = copy.deepcopy(res)
        bad_res["grounding"]["trace"][0]["parameters"]["temperature"] = 1
        with self.assertRaises(ValueError) as ctx:
            validate_grounded_result(bad_res, self.hits, fake_gen.digest)
        self.assertIn("parameters mismatch", str(ctx.exception))

    def test_mutation_load_duration_nan_or_sum_99999_fails(self) -> None:
        """load_duration NaN o con suma incoherente (99999) es rechazado."""
        res, fake_gen = self._make_first_pass_answered_result()

        # NaN
        bad_res_nan = copy.deepcopy(res)
        bad_res_nan["stats"]["load_duration"] = float("nan")
        with self.assertRaises(ValueError):
            validate_grounded_result(bad_res_nan, self.hits, fake_gen.digest)

        # Suma incoherente 99999
        bad_res_sum = copy.deepcopy(res)
        bad_res_sum["grounding"]["trace"][0]["stats"]["load_duration"] = 500
        bad_res_sum["stats"]["load_duration"] = 99999
        with self.assertRaises(ValueError) as ctx:
            validate_grounded_result(bad_res_sum, self.hits, fake_gen.digest)
        self.assertIn("Stats load_duration sum mismatch", str(ctx.exception))

    def test_mutation_metadata_extra_or_page_number_nan_fails(self) -> None:
        """Claves extra en metadatos o page_number=NaN son rechazados."""
        res, fake_gen = self._make_first_pass_answered_result()

        # Clave extra en selected_evidence
        bad_res_extra = copy.deepcopy(res)
        bad_res_extra["grounding"]["selected_evidence"][0]["extra_meta_field"] = "malformed"
        with self.assertRaises(ValueError):
            validate_grounded_result(bad_res_extra, self.hits, fake_gen.digest)

        # page_number con NaN
        bad_res_page_nan = copy.deepcopy(res)
        bad_res_page_nan["grounding"]["selected_evidence"][0]["page_number"] = float("nan")
        with self.assertRaises(ValueError):
            validate_grounded_result(bad_res_page_nan, self.hits, fake_gen.digest)

    def test_source_contexts_only_cited_sources_deduped_and_ordered(self) -> None:
        """source_contexts extrae solo fuentes citadas, deduplicadas y ordenadas por primera referencia."""
        hits_by_source = {f"S{h.rank}": h for h in self.hits}
        evidence = [
            {"evidence_id": "E1", "source_id": "S2", "quote": self.quote2},
            {"evidence_id": "E2", "source_id": "S1", "quote": self.quote1},
            {"evidence_id": "E3", "source_id": "S2", "quote": self.quote2},
        ]
        # C1 cita E1 (S2), C2 cita E2 (S1) y E3 (S2)
        claims = [
            {"claim_id": "C1", "text": "Claim 1", "evidence_ids": ["E1"]},
            {"claim_id": "C2", "text": "Claim 2", "evidence_ids": ["E2", "E3"]},
        ]
        contexts = build_source_contexts_for_claims(claims, evidence, hits_by_source)
        # S2 aparece primero, luego S1; S2 no se duplica
        self.assertEqual(len(contexts), 2)
        self.assertEqual(contexts[0]["source_id"], "S2")
        self.assertEqual(contexts[0]["text"], collapse_spaces(self.hit2.chunk.text))
        self.assertEqual(contexts[1]["source_id"], "S1")
        self.assertEqual(contexts[1]["text"], collapse_spaces(self.hit1.chunk.text))

    def test_rag_verification_and_repair_only_receive_assigned_quotes(self) -> None:
        """No introducir el resto de los chunks en ninguno de los juicios de apoyo."""
        res, fake_gen = self._make_repaired_answered_result()
        val = validate_grounded_result(res, self.hits, fake_gen.digest)
        vc = val["grounding"]["verification_contexts"]
        self.assertEqual(vc, {"initial": [], "repair": []})
        for stage_index in (2, 5):
            payload = json.loads(fake_gen.history[stage_index][0][1]["content"])
            self.assertEqual(set(payload), {"afirmaciones", "catalogo_evidencias"})

    def test_adulterated_verification_context_rejected(self) -> None:
        """Contexto adulterado o falsificado en verification_contexts es detectado y rechazado."""
        res, fake_gen = self._make_first_pass_answered_result()

        # Texto de contexto alterado
        bad_res1 = copy.deepcopy(res)
        bad_res1["grounding"]["verification_contexts"]["initial"] = [
            {"source_id": "S1", "text": "Texto adulterado arbitrario."}]
        with self.assertRaises(ValueError) as ctx:
            validate_grounded_result(bad_res1, self.hits, fake_gen.digest)
        self.assertIn("verification_contexts mismatch", str(ctx.exception))

        # Fuente extra no citada añadida al contexto
        bad_res2 = copy.deepcopy(res)
        bad_res2["grounding"]["verification_contexts"]["initial"].append({"source_id": "S2", "text": collapse_spaces(self.hit2.chunk.text)})
        with self.assertRaises(ValueError) as ctx:
            validate_grounded_result(bad_res2, self.hits, fake_gen.digest)
        self.assertIn("verification_contexts mismatch", str(ctx.exception))

    def test_quote_not_contained_in_source_context_rejected(self) -> None:
        """build_verification_messages rechaza si una cita no está contenida en su contexto."""
        claims = [{"claim_id": "C1", "text": "Claim", "evidence_ids": ["E1"]}]
        evidence = [{"evidence_id": "E1", "source_id": "S1", "quote": "Cita que no existe en el contexto"}]
        source_contexts = [{"source_id": "S1", "text": "Texto oficial diferente que no contiene la cita."}]
        with self.assertRaises(ValueError) as ctx:
            build_verification_messages(claims, evidence, source_contexts=source_contexts)
        self.assertIn("not found in source context", str(ctx.exception))

    def test_uncited_source_in_context_rejected(self) -> None:
        """build_verification_messages rechaza contextos de fuentes que no están citadas."""
        claims = [{"claim_id": "C1", "text": "Claim", "evidence_ids": ["E1"]}]
        evidence = [{"evidence_id": "E1", "source_id": "S1", "quote": self.quote1}]
        # S2 no está citada
        source_contexts = [
            {"source_id": "S1", "text": self.quote1},
            {"source_id": "S2", "text": self.quote2},
        ]
        with self.assertRaises(ValueError) as ctx:
            build_verification_messages(claims, evidence, source_contexts=source_contexts)
        self.assertIn("Uncited source_id", str(ctx.exception))

    def test_wrong_order_in_source_contexts_rejected(self) -> None:
        """build_verification_messages rechaza contextos cuyo orden no coincide con la primera cita."""
        claims = [
            {"claim_id": "C1", "text": "Claim 1", "evidence_ids": ["E1"]},
            {"claim_id": "C2", "text": "Claim 2", "evidence_ids": ["E2"]},
        ]
        evidence = [
            {"evidence_id": "E1", "source_id": "S1", "quote": self.quote1},
            {"evidence_id": "E2", "source_id": "S2", "quote": self.quote2},
        ]
        # Invertir el orden (S2 antes que S1)
        source_contexts = [
            {"source_id": "S2", "text": self.quote2},
            {"source_id": "S1", "text": self.quote1},
        ]
        with self.assertRaises(ValueError) as ctx:
            build_verification_messages(claims, evidence, source_contexts=source_contexts)
        self.assertIn("sources order mismatch", str(ctx.exception))

    def test_verifier_prompt_rules_on_context_and_contradictions(self) -> None:
        """VERIFICATION_SYSTEM_PROMPT estipula reglas de contexto como datos y contradicciones de entidad."""
        self.assertIn("DATOS", VERIFICATION_SYSTEM_PROMPT)
        self.assertIn("nunca rescata una cita irrelevante", VERIFICATION_SYSTEM_PROMPT)
        self.assertIn("contradicciones de entidad", VERIFICATION_SYSTEM_PROMPT)

    def test_selection_schema_contiguous_multisentence_windows(self) -> None:
        """build_selection_schema genera candidatos de 1 oración y hasta 4 ventanas multioracionales contiguas."""
        multi_text = (
            "El gasto en pensiones alcanzará el 16,1% en 2050. "
            "Asimismo, este indicador crecerá de forma sostenida hasta 2060. "
            "Por otra parte, la tasa de cobertura se mantendrá estable."
        )
        hit_multi = _make_hit(1, multi_text)
        sel_messages = build_selection_messages("¿Gasto?", [hit_multi])
        schema = build_selection_schema(sel_messages)
        quotes = schema["anyOf"][0]["properties"]["evidence"]["items"]["anyOf"][0]["properties"]["quote"]["enum"]

        # Debe contener oraciones individuales y ventana multioracional contigua
        self.assertTrue(any("16,1%" in q and "Asimismo" not in q for q in quotes))
        multi_window = "El gasto en pensiones alcanzará el 16,1% en 2050. Asimismo, este indicador crecerá de forma sostenida hasta 2060."
        self.assertIn(multi_window, quotes)
        # Cada quote es una subcadena de multi_text y <= 700 caracteres
        for q in quotes:
            self.assertIn(q, collapse_spaces(multi_text))
            self.assertLessEqual(len(q), 700)

    def test_budget_guard_verification_12288_vs_default_6500(self) -> None:
        """Presupuesto 12288 bytes UTF-8 para verification vs 6500 bytes para otras etapas."""
        # Mensaje de verificación con tamaño entre 6500 y 12288 bytes
        claims = [{"claim_id": "C1", "text": "Claim", "evidence_ids": ["E1"]}]
        evidence = [{"evidence_id": "E1", "source_id": "S1", "quote": self.quote1}]
        # Crear contexto de ~7500 bytes que contiene quote1
        big_context = self.quote1 + " " + ("Texto descriptivo adicional del informe. " * 180)
        self.assertGreater(len(big_context.encode("utf-8")), 6500)
        self.assertLess(len(big_context.encode("utf-8")), 11500)

        # Debe tener éxito para verificación (presupuesto 12288)
        msgs = build_verification_messages(claims, evidence, source_contexts=[{"source_id": "S1", "text": big_context}])
        self.assertGreater(len(json.dumps(msgs, ensure_ascii=False).encode("utf-8")), 6500)
        self.assertLessEqual(len(json.dumps(msgs, ensure_ascii=False).encode("utf-8")), MAX_MESSAGES_BYTES_VERIFICATION)

        # Si excede 12288 bytes, verification falla
        huge_context = self.quote1 + " " + ("Texto enorme para desbordar. " * 500)
        with self.assertRaises(ValueError) as ctx:
            build_verification_messages(claims, evidence, source_contexts=[{"source_id": "S1", "text": huge_context}])
        self.assertIn(f"exceeds budget of {MAX_MESSAGES_BYTES_VERIFICATION} bytes", str(ctx.exception))

    def test_trace_messages_budget_bytes_validation(self) -> None:
        """validate_grounded_result exige messages_budget_bytes exacto (6500 o 12288) por etapa en la traza."""
        res, fake_gen = self._make_first_pass_answered_result()

        # Comprobar valores por defecto válidos
        val = validate_grounded_result(res, self.hits, fake_gen.digest)
        self.assertEqual(val["grounding"]["trace"][0]["messages_budget_bytes"], 6500)
        self.assertEqual(val["grounding"]["trace"][1]["messages_budget_bytes"], 6500)
        self.assertEqual(val["grounding"]["trace"][2]["messages_budget_bytes"], 12288)

        # Alterar presupuesto en la traza
        bad_res = copy.deepcopy(res)
        bad_res["grounding"]["trace"][2]["messages_budget_bytes"] = 6500
        with self.assertRaises(ValueError) as ctx:
            validate_grounded_result(bad_res, self.hits, fake_gen.digest)
        self.assertIn("messages_budget_bytes mismatch", str(ctx.exception))

    def test_regression_duplicate_quote_deduplication_and_single_partial_supported_rejected(self) -> None:
        """Cita idéntica repetida en selección se deduplica a 1; dos partials idénticos no rescatan supported."""
        # 1. En selección: misma cita repetida con mismo source_id se deduplica estrictamente a 1 evidencia
        payload_dup = {
            "status": "selected",
            "evidence": [
                {"source_id": "S1", "quote": self.quote1},
                {"source_id": "S1", "quote": self.quote1},
            ],
            "reason": "",
        }
        validated_sel = validate_selection_payload(payload_dup, self.hits)
        self.assertEqual(len(validated_sel["evidence"]), 1)
        self.assertEqual(validated_sel["evidence"][0]["evidence_id"], "E1")

        # 2. En verificación: una única evidencia con relación 'partial' no puede sostener veredicto 'supported'
        claims = [{"claim_id": "C1", "text": "Afirmación dependiente", "evidence_ids": ["E1"]}]
        payload_ver = {
            "verdicts": [
                {
                    "claim_id": "C1",
                    "verdict": "supported",
                    "reason": "Afirmación con soporte parcial no autosuficiente",
                    "evidence_verdicts": [{"evidence_id": "E1", "relation": "partial"}],
                }
            ]
        }
        with self.assertRaises(ValueError) as ctx:
            validate_verification_payload(payload_ver, claims, validated_sel["evidence"])
        self.assertIn("requires at least one 'supports' or at least two 'partial'", str(ctx.exception))

    def test_regression_selection_rejects_five_evidence_items(self) -> None:
        """La etapa de selección rechaza estrictamente más de 4 evidencias (ej. 5 evidencias)."""
        evidence_5 = [
            {"source_id": "S1", "quote": self.quote1},
            {"source_id": "S2", "quote": self.quote2},
            {"source_id": "S1", "quote": self.quote1},
            {"source_id": "S2", "quote": self.quote2},
            {"source_id": "S1", "quote": self.quote1},
        ]
        payload_5 = {
            "status": "selected",
            "evidence": evidence_5,
            "reason": "",
        }
        with self.assertRaises(ValueError) as ctx:
            validate_selection_payload(payload_5, self.hits)
        self.assertIn("between 1 and 4 items", str(ctx.exception))

    def test_regression_selection_abstention_cannot_have_invented_history_in_validate_grounded_result(self) -> None:
        """Abstención sin evidencia en selección no puede acompañarse de initial_claims/verdicts inventados."""
        res, fake_gen = self._make_first_pass_answered_result()

        # Caso A: selected_evidence vacía con initial_claims inventados
        bad_res_claims = copy.deepcopy(res)
        bad_res_claims["grounding"]["selected_evidence"] = []
        with self.assertRaises(ValueError) as ctx:
            validate_grounded_result(bad_res_claims, self.hits, fake_gen.digest)
        self.assertIn("Invented history after early abstention", str(ctx.exception))

        # Caso B: initial_claims vacíos pero initial_verdicts inventados
        bad_res_verdicts = copy.deepcopy(res)
        bad_res_verdicts["grounding"]["initial_claims"] = []
        with self.assertRaises(ValueError) as ctx:
            validate_grounded_result(bad_res_verdicts, self.hits, fake_gen.digest)
        self.assertIn("Invented history after early abstention", str(ctx.exception))

        # Caso C: coherencia estructural de etapas esperadas cuando selected_evidence es vacía
        bad_res_stages = copy.deepcopy(res)
        bad_res_stages["grounding"]["selected_evidence"] = []
        bad_res_stages["grounding"]["initial_claims"] = []
        bad_res_stages["grounding"]["initial_verdicts"] = []
        bad_res_stages["grounding"]["verification_contexts"] = {"initial": [], "repair": []}
        bad_res_stages["grounding"]["final_verdicts"] = []
        bad_res_stages["grounding"]["accepted_claims"] = []
        bad_res_stages["grounding"]["rejected_claims"] = []
        bad_res_stages["status"] = "insufficient_evidence"
        bad_res_stages["claims"] = []
        bad_res_stages["reason"] = "Abstención"
        # El trace aún contiene 3 etapas pero la reconstrucción de abstención espera solo 1 etapa ('selection')
        with self.assertRaises(ValueError) as ctx:
            validate_grounded_result(bad_res_stages, self.hits, fake_gen.digest)
        self.assertIn("trace stages count mismatch", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
