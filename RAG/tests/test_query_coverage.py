"""Offline unit tests for query coverage audit, schema validation, and publication gate."""

import copy
import hashlib
import json
import math
from pathlib import Path
import tempfile
from typing import Any
import unittest
from unittest.mock import MagicMock, patch
import urllib.error
import urllib.request

from experiments.preserved_q3_verifier import (
    Q3VerifierClient,
    VERIFIER_Q3_DIGEST,
    VERIFIER_Q3_MODEL,
    VERIFIER_Q3_PARAMETERS,
)
from experiments.query_coverage import (
    ALLOWED_DECISIONS,
    ALLOWED_REASON_CODES,
    COVERAGE_STAGE,
    PINNED_9B_DIGEST,
    PINNED_9B_MODEL,
    PINNED_9B_PARAMETERS,
    PINNED_27B_DIGEST,
    PINNED_27B_MODEL,
    PINNED_27B_PARAMETERS,
    PRESERVED_COVERAGE_SCHEMA_VERSION,
    UNAVAILABLE_TEMPLATES,
    _revalidate_selected_evidence,
    answer_covered,
    audit_coverage,
    build_coverage_messages,
    build_coverage_publication,
    CorrectionRecordedGenerator,
    build_coverage_schema,
    claim_inventory,
    correct_coverage_once,
    format_covered_answer,
    serialize_inventory_for_audit,
    validate_coverage_payload,
)
from experiments.probe_coverage_correction import (
    EXPECTED_CASE_IDS,
    PINNED_MANIFEST_SHA256,
    run_probe,
    validate_input_files,
)
from rag_pensiones.answer import DEFAULT_GENERATION_MODEL
from rag_pensiones.chunking import ChunkRecord
from rag_pensiones.grounding import DEFAULT_PARAMETERS
from rag_pensiones.ingest import PROJECT_ROOT
from rag_pensiones.search import SearchHit


class MockTransport:
    """Mock transport providing valid Ollama chat responses for unit tests."""

    def __init__(self, chat_response: dict | None = None, chat_error: Exception | None = None) -> None:
        self.chat_response = chat_response
        self.chat_error = chat_error
        self.calls: list[dict] = []

    def get_version(self) -> dict:
        return {"version": "0.40.11"}

    def get_tags(self) -> dict:
        return {
            "models": [
                {
                    "name": VERIFIER_Q3_MODEL,
                    "model": VERIFIER_Q3_MODEL,
                    "digest": VERIFIER_Q3_DIGEST,
                    "details": {"quantization_level": "Q3_K_M"},
                }
            ]
        }

    def show(self, model: str) -> dict:
        return {
            "details": {"quantization_level": "Q3_K_M"},
            "parameters": "presence_penalty 1.5\ntop_k 20\ntop_p 0.95\nmin_p 0.0\n",
        }

    def get_ps(self) -> dict:
        return {"models": []}

    def unload(self, model: str) -> dict:
        return {"status": "success"}

    def chat(self, request: dict) -> dict:
        self.calls.append(copy.deepcopy(request))
        if self.chat_error is not None:
            raise self.chat_error
        return copy.deepcopy(self.chat_response)


def _make_valid_raw_response(content_payload: dict) -> dict:
    return {
        "model": VERIFIER_Q3_MODEL,
        "done": True,
        "done_reason": "stop",
        "prompt_eval_count": 350,
        "eval_count": 80,
        "prompt_eval_duration": 100000000,
        "eval_duration": 200000000,
        "total_duration": 350000000,
        "load_duration": 10000000,
        "message": {
            "role": "assistant",
            "content": json.dumps(content_payload, ensure_ascii=False),
        },
    }


class TestQ3VerifierStageInvariants(unittest.TestCase):
    """Verifies allowed coverage stage vs forbidden generation/selection stages."""

    def test_allowed_stages_in_q3_verifier(self):
        transport = MockTransport(
            chat_response=_make_valid_raw_response({"parts": []})
        )
        client = Q3VerifierClient(expected_digest=VERIFIER_Q3_DIGEST, transport=transport)

        # Allowed stages
        for allowed in ("verification", "reverification", "query_coverage"):
            try:
                # Will fail on empty messages / schema before chat, or pass validation
                client.chat([{"role": "user", "content": "hi"}], {"type": "object"}, stage=allowed)
            except ValueError as exc:
                self.assertNotIn("Q3VerifierClient.chat accepts only", str(exc))

        # Forbidden stages
        for forbidden in ("selection", "generation", "repair_selection", "repair_generation", "custom"):
            with self.assertRaises(ValueError) as ctx:
                client.chat([{"role": "user", "content": "hi"}], {"type": "object"}, stage=forbidden)
            self.assertIn("Q3VerifierClient.chat accepts only", str(ctx.exception))


class TestClaimInventory(unittest.TestCase):
    """Verifies extraction and strict 1:1 alignment between canonical claims and publication."""

    def setUp(self):
        self.sample_report = {
            "schema_version": "preserved-direct-query-v1",
            "query": "¿Con qué año histórico se compara?",
            "outcome": {
                "status": "answered",
                "full_result": {
                    "grounding": {
                        "accepted_claims": [
                            {"claim_id": "C1", "text": "Se compara con el año 2015.", "evidence_ids": ["E1"]}
                        ]
                    }
                },
                "publication": {
                    "status": "answered",
                    "claims": [
                        {
                            "text": "Se compara con el año 2015.",
                            "evidence": [{"source_id": "S1", "quote": "Los niveles se comparan con 2015."}],
                        }
                    ],
                },
            },
        }

    def test_valid_alignment(self):
        inv = claim_inventory(self.sample_report)
        self.assertEqual(len(inv), 1)
        self.assertEqual(inv[0]["claim_id"], "C1")
        self.assertEqual(inv[0]["text"], "Se compara con el año 2015.")
        self.assertEqual(inv[0]["evidence"][0]["source_id"], "S1")
        self.assertEqual(inv[0]["publication_claim"]["text"], "Se compara con el año 2015.")

    def test_text_mismatch_raises_error(self):
        mismatched = copy.deepcopy(self.sample_report)
        mismatched["outcome"]["publication"]["claims"][0]["text"] = "Texto alterado sin alinear."
        with self.assertRaises(ValueError) as ctx:
            claim_inventory(mismatched)
        self.assertIn("Text alignment mismatch", str(ctx.exception))

    def test_count_mismatch_raises_error(self):
        mismatched = copy.deepcopy(self.sample_report)
        mismatched["outcome"]["publication"]["claims"].append(
            {"text": "Claim extra", "evidence": []}
        )
        with self.assertRaises(ValueError) as ctx:
            claim_inventory(mismatched)
        self.assertIn("Alignment count mismatch", str(ctx.exception))

    def test_error_outcome_returns_empty(self):
        err_report = {"outcome": {"status": "error", "error": "Crash"}}
        self.assertEqual(claim_inventory(err_report), [])

    def test_insufficient_evidence_with_no_claims_returns_empty(self):
        report = {
            "outcome": {
                "status": "insufficient_evidence",
                "full_result": {"grounding": {"accepted_claims": []}},
                "publication": {"claims": []},
            }
        }
        self.assertEqual(claim_inventory(report), [])

    def test_serialization_strips_private_metadata(self):
        inv = claim_inventory(self.sample_report)
        inv[0]["private_metadata"] = "secret"
        clean = serialize_inventory_for_audit(inv)
        self.assertEqual(len(clean), 1)
        self.assertEqual(set(clean[0].keys()), {"claim_id", "text", "evidence"})
        self.assertEqual(clean[0]["evidence"][0], {"source_id": "S1", "quote": "Los niveles se comparan con 2015."})


class TestPayloadValidation(unittest.TestCase):
    """Verifies strict deterministic validation of coverage payload, span partitions, and IDs."""

    def setUp(self):
        self.query = "¿Con qué año histórico se comparan los niveles futuros de generosidad y qué pensión mensual exacta cobraré si me jubilo en 2031?"
        self.inventory = [
            {
                "claim_id": "C1",
                "text": "Se comparan con 2015.",
                "evidence": [{"source_id": "S1", "quote": "Comparación con 2015."}],
                "publication_claim": {"text": "Se comparan con 2015.", "evidence": [{"source_id": "S1", "quote": "Comparación con 2015."}]},
            }
        ]

    def test_valid_partition_two_parts(self):
        payload = {
            "parts": [
                {
                    "query_span": "¿Con qué año histórico se comparan los niveles futuros de generosidad",
                    "decision": "answered",
                    "claim_ids": ["C1"],
                    "reason_code": "covered",
                },
                {
                    "query_span": "y qué pensión mensual exacta cobraré si me jubilo en 2031?",
                    "decision": "not_answered",
                    "claim_ids": [],
                    "reason_code": "not_in_evaluated_evidence",
                },
            ]
        }
        res = validate_coverage_payload(payload, self.query, self.inventory)
        self.assertEqual(res["status"], "partial")
        self.assertEqual(len(res["parts"]), 2)
        self.assertEqual(res["parts"][0]["start_char"], 0)
        self.assertEqual(res["parts"][0]["end_char"], 69)
        self.assertEqual(res["parts"][1]["start_char"], 70)
        self.assertEqual(res["parts"][1]["end_char"], len(self.query))

    def test_valid_single_part_answered(self):
        query = "¿En qué año sitúa el informe la aceleración del crecimiento?"
        payload = {
            "parts": [
                {
                    "query_span": query,
                    "decision": "answered",
                    "claim_ids": ["C1"],
                    "reason_code": "covered",
                }
            ]
        }
        res = validate_coverage_payload(payload, query, self.inventory)
        self.assertEqual(res["status"], "answered")

    def test_valid_single_part_not_answered(self):
        query = "¿Cuántas mujeres cobrarán una pensión contributiva en 2050 según esta proyección?"
        payload = {
            "parts": [
                {
                    "query_span": query,
                    "decision": "not_answered",
                    "claim_ids": [],
                    "reason_code": "scope_mismatch",
                }
            ]
        }
        res = validate_coverage_payload(payload, query, self.inventory)
        self.assertEqual(res["status"], "insufficient_evidence")

    def test_gap_with_non_whitespace_fails(self):
        payload = {
            "parts": [
                {
                    "query_span": "¿Con qué año",
                    "decision": "answered",
                    "claim_ids": ["C1"],
                    "reason_code": "covered",
                },
                {
                    "query_span": "los niveles futuros de generosidad y qué pensión mensual exacta cobraré si me jubilo en 2031?",
                    "decision": "not_answered",
                    "claim_ids": [],
                    "reason_code": "not_in_evaluated_evidence",
                },
            ]
        }
        with self.assertRaises(ValueError) as ctx:
            validate_coverage_payload(payload, self.query, self.inventory)
        self.assertIn("Non-whitespace characters", str(ctx.exception))

    def test_overlapping_spans_fail(self):
        payload = {
            "parts": [
                {
                    "query_span": "¿Con qué año histórico se comparan",
                    "decision": "answered",
                    "claim_ids": ["C1"],
                    "reason_code": "covered",
                },
                {
                    "query_span": "se comparan los niveles futuros de generosidad y qué pensión mensual exacta cobraré si me jubilo en 2031?",
                    "decision": "not_answered",
                    "claim_ids": [],
                    "reason_code": "not_in_evaluated_evidence",
                },
            ]
        }
        with self.assertRaises(ValueError) as ctx:
            validate_coverage_payload(payload, self.query, self.inventory)
        self.assertIn("not found sequentially", str(ctx.exception))

    def test_changed_or_invented_span_fails(self):
        payload = {
            "parts": [
                {
                    "query_span": "¿En qué año histórico modificado?",
                    "decision": "answered",
                    "claim_ids": ["C1"],
                    "reason_code": "covered",
                }
            ]
        }
        with self.assertRaises(ValueError) as ctx:
            validate_coverage_payload(payload, self.query, self.inventory)
        self.assertIn("not found sequentially", str(ctx.exception))

    def test_omitted_clause_at_end_fails(self):
        payload = {
            "parts": [
                {
                    "query_span": "¿Con qué año histórico se comparan los niveles futuros de generosidad",
                    "decision": "answered",
                    "claim_ids": ["C1"],
                    "reason_code": "covered",
                }
            ]
        }
        with self.assertRaises(ValueError) as ctx:
            validate_coverage_payload(payload, self.query, self.inventory)
        self.assertIn("omitted at end of query", str(ctx.exception))

    def test_extra_keys_in_payload_fail(self):
        payload = {
            "parts": [
                {
                    "query_span": self.query,
                    "decision": "answered",
                    "claim_ids": ["C1"],
                    "reason_code": "covered",
                }
            ],
            "extra_key": "forbidden",
        }
        with self.assertRaises(ValueError) as ctx:
            validate_coverage_payload(payload, self.query, self.inventory)
        self.assertIn("keys mismatch", str(ctx.exception))

    def test_unknown_claim_id_fails(self):
        payload = {
            "parts": [
                {
                    "query_span": self.query,
                    "decision": "answered",
                    "claim_ids": ["C99"],
                    "reason_code": "covered",
                }
            ]
        }
        with self.assertRaises(ValueError) as ctx:
            validate_coverage_payload(payload, self.query, self.inventory)
        self.assertIn("unknown claim ID 'C99'", str(ctx.exception))

    def test_duplicate_claim_ids_fail(self):
        payload = {
            "parts": [
                {
                    "query_span": self.query,
                    "decision": "answered",
                    "claim_ids": ["C1", "C1"],
                    "reason_code": "covered",
                }
            ]
        }
        with self.assertRaises(ValueError) as ctx:
            validate_coverage_payload(payload, self.query, self.inventory)
        self.assertIn("duplicate claim IDs", str(ctx.exception))

    def test_answered_with_empty_claim_ids_fails(self):
        payload = {
            "parts": [
                {
                    "query_span": self.query,
                    "decision": "answered",
                    "claim_ids": [],
                    "reason_code": "covered",
                }
            ]
        }
        with self.assertRaises(ValueError) as ctx:
            validate_coverage_payload(payload, self.query, self.inventory)
        self.assertIn("answered decision requires non-empty claim_ids", str(ctx.exception))

    def test_answered_with_scope_mismatch_fails(self):
        payload = {
            "parts": [
                {
                    "query_span": self.query,
                    "decision": "answered",
                    "claim_ids": ["C1"],
                    "reason_code": "scope_mismatch",
                }
            ]
        }
        with self.assertRaises(ValueError) as ctx:
            validate_coverage_payload(payload, self.query, self.inventory)
        self.assertIn("answered decision requires reason_code 'covered'", str(ctx.exception))

    def test_not_answered_with_non_empty_ids_fails(self):
        payload = {
            "parts": [
                {
                    "query_span": self.query,
                    "decision": "not_answered",
                    "claim_ids": ["C1"],
                    "reason_code": "scope_mismatch",
                }
            ]
        }
        with self.assertRaises(ValueError) as ctx:
            validate_coverage_payload(payload, self.query, self.inventory)
        self.assertIn("not_answered decision requires empty claim_ids", str(ctx.exception))

    def test_not_answered_with_covered_code_fails(self):
        payload = {
            "parts": [
                {
                    "query_span": self.query,
                    "decision": "not_answered",
                    "claim_ids": [],
                    "reason_code": "covered",
                }
            ]
        }
        with self.assertRaises(ValueError) as ctx:
            validate_coverage_payload(payload, self.query, self.inventory)
        self.assertIn("not_answered decision requires non-covered reason_code", str(ctx.exception))


class TestBuildPublication(unittest.TestCase):
    """Verifies that publication includes only linked claims and emits fixed-template refusals."""

    def setUp(self):
        self.query = "¿Pregunta parte 1 y pregunta parte 2?"
        self.inventory = [
            {
                "claim_id": "C1",
                "text": "Dato para parte 1.",
                "evidence": [{"source_id": "S1", "quote": "Cita parte 1."}],
                "publication_claim": {"text": "Dato para parte 1.", "evidence": [{"source_id": "S1", "quote": "Cita parte 1."}]},
            },
            {
                "claim_id": "C2",
                "text": "Dato no vinculado.",
                "evidence": [{"source_id": "S2", "quote": "Cita no vinculada."}],
                "publication_claim": {"text": "Dato no vinculado.", "evidence": [{"source_id": "S2", "quote": "Cita no vinculada."}]},
            },
        ]

    def test_unlinked_claims_are_not_published(self):
        validated = {
            "status": "partial",
            "parts": [
                {
                    "query_span": "¿Pregunta parte 1",
                    "decision": "answered",
                    "claim_ids": ["C1"],
                    "reason_code": "covered",
                },
                {
                    "query_span": "y pregunta parte 2?",
                    "decision": "not_answered",
                    "claim_ids": [],
                    "reason_code": "not_in_evaluated_evidence",
                },
            ],
        }
        pub = build_coverage_publication(self.query, self.inventory, validated)
        self.assertEqual(pub["status"], "partial")
        self.assertEqual(len(pub["claims"]), 1)
        self.assertEqual(pub["claims"][0]["text"], "Dato para parte 1.")
        # C2 was in inventory but unlinked to any answered part -> omitted!
        self.assertEqual(len(pub["missing_parts"]), 1)
        self.assertEqual(pub["missing_parts"][0]["query_span"], "y pregunta parte 2?")
        self.assertIn("Las evidencias evaluadas no contienen", pub["missing_parts"][0]["message"])

    def test_all_unanswered_scoped_abstention(self):
        validated = {
            "status": "insufficient_evidence",
            "parts": [
                {
                    "query_span": self.query,
                    "decision": "not_answered",
                    "claim_ids": [],
                    "reason_code": "scope_mismatch",
                }
            ],
        }
        pub = build_coverage_publication(self.query, self.inventory, validated)
        self.assertEqual(pub["status"], "insufficient_evidence")
        self.assertEqual(pub["claims"], [])
        self.assertIn("Las evidencias evaluadas no corresponden al alcance", pub["reason"])


class TestAuditCoverageOffline(unittest.TestCase):
    def test_generator_released_before_coverage_after_repair(self):
        query = "¿Año comparable?"
        inventory = [{"claim_id": "C1", "text": "Se compara con 2015.", "evidence": []}]
        payload = {"parts": [{"query_span": query, "decision": "answered", "claim_ids": ["C1"], "reason_code": "covered"}]}
        verifier = MagicMock()
        verifier.model = VERIFIER_Q3_MODEL
        verifier.calls = []
        verifier.chat.return_value = _make_valid_raw_response(payload)
        result = audit_coverage(query, inventory, verifier=verifier)
        self.assertEqual(result["status"], "completed")
        self.assertEqual([c[0] for c in verifier.method_calls], ["release_loaded", "chat"])
        verifier.release_loaded.assert_called_once_with("qwen3.5:9b")

    """Verifies audit execution, call preservation, error handling, and empty inventory skip."""

    def test_empty_inventory_skips_audit(self):
        mock_verifier = MagicMock()
        res = audit_coverage("¿Pregunta?", [], verifier=mock_verifier)
        self.assertEqual(res["status"], "completed")
        self.assertTrue(res["skipped"])
        self.assertEqual(res["calls"], [])
        # Verifier chat must NOT be called
        mock_verifier.chat.assert_not_called()
        self.assertEqual(res["validated_payload"]["status"], "insufficient_evidence")

    def test_successful_audit_call(self):
        inventory = [
            {
                "claim_id": "C1",
                "text": "Se compara con 2015.",
                "evidence": [{"source_id": "S1", "quote": "Cita"}],
                "publication_claim": {"text": "Se compara con 2015.", "evidence": [{"source_id": "S1", "quote": "Cita"}]},
            }
        ]
        query = "¿Con qué año se compara?"
        content_payload = {
            "parts": [
                {
                    "query_span": "¿Con qué año se compara?",
                    "decision": "answered",
                    "claim_ids": ["C1"],
                    "reason_code": "covered",
                }
            ]
        }
        transport = MockTransport(chat_response=_make_valid_raw_response(content_payload))
        client = Q3VerifierClient(expected_digest=VERIFIER_Q3_DIGEST, transport=transport)

        res = audit_coverage(query, inventory, verifier=client)
        self.assertEqual(res["status"], "completed")
        self.assertFalse(res["skipped"])
        self.assertEqual(len(res["calls"]), 1)
        self.assertEqual(res["calls"][0]["stage"], COVERAGE_STAGE)
        self.assertEqual(res["validated_payload"]["status"], "answered")

    def test_error_preserved_without_fallback(self):
        inventory = [
            {
                "claim_id": "C1",
                "text": "Se compara con 2015.",
                "evidence": [{"source_id": "S1", "quote": "Cita"}],
                "publication_claim": {"text": "Se compara con 2015.", "evidence": [{"source_id": "S1", "quote": "Cita"}]},
            }
        ]
        # Return invalid payload with extra key
        bad_payload = {"parts": [], "extra": True}
        transport = MockTransport(chat_response=_make_valid_raw_response(bad_payload))
        client = Q3VerifierClient(expected_digest=VERIFIER_Q3_DIGEST, transport=transport)

        res = audit_coverage("¿Pregunta?", inventory, verifier=client)
        self.assertEqual(res["status"], "error")
        self.assertIn("Coverage payload keys mismatch", res["error"])
        self.assertEqual(len(res["calls"]), 1)


class TestFormattingAndVisibleRefusal(unittest.TestCase):
    """Verifies that format_covered_answer renders both supported claims and missing parts."""

    def test_format_partial_answer(self):
        report = {
            "status": "partial",
            "publication": {
                "status": "partial",
                "claims": [
                    {
                        "text": "El informe compara los niveles futuros con el año 2015.",
                        "evidence": [{"source_id": "S1", "quote": "Se toman como referencia los datos de 2015."}],
                    }
                ],
                "missing_parts": [
                    {
                        "query_span": "qué pensión mensual exacta cobraré en 2031",
                        "reason_code": "not_in_evaluated_evidence",
                        "message": "Las evidencias evaluadas no contienen información para responder a: «qué pensión mensual exacta cobraré en 2031».",
                    }
                ],
            },
            "factual_report": {
                "outcome": {
                    "expanded_hits": [
                        {"rank": 1, "chunk": {"page_number": 95, "text": "..."}}
                    ]
                }
            },
        }

        rendered = format_covered_answer(report)
        self.assertIn("El informe compara los niveles futuros con el año 2015.", rendered)
        self.assertIn("[S1, pág. 95] Se toman como referencia los datos de 2015.", rendered)
        self.assertIn("Partes no respondidas en las fuentes evaluadas:", rendered)
        self.assertIn("Las evidencias evaluadas no contienen información para responder a: «qué pensión mensual exacta cobraré en 2031».", rendered)


class TestLiteralPartitionEdgeCases(unittest.TestCase):
    """Verifies strict sequential partition covering punctuation, accents, newlines, and whitespace."""

    def setUp(self):
        self.inventory = [
            {
                "claim_id": "C1",
                "text": "Afirmación 1.",
                "evidence": [{"source_id": "S1", "quote": "Cita 1."}],
                "publication_claim": {"text": "Afirmación 1.", "evidence": [{"source_id": "S1", "quote": "Cita 1."}]},
            },
            {
                "claim_id": "C2",
                "text": "Afirmación 2.",
                "evidence": [{"source_id": "S2", "quote": "Cita 2."}],
                "publication_claim": {"text": "Afirmación 2.", "evidence": [{"source_id": "S2", "quote": "Cita 2."}]},
            },
        ]

    def test_accented_punctuation_and_newlines_partition(self):
        query = "¿Cuál es el cálculo según la AIReF?\n¿Y qué déficit prevé para el año 2050?"
        part1 = "¿Cuál es el cálculo según la AIReF?"
        part2 = "¿Y qué déficit prevé para el año 2050?"
        payload = {
            "parts": [
                {
                    "query_span": part1,
                    "decision": "answered",
                    "claim_ids": ["C1"],
                    "reason_code": "covered",
                },
                {
                    "query_span": part2,
                    "decision": "not_answered",
                    "claim_ids": [],
                    "reason_code": "not_in_evaluated_evidence",
                },
            ]
        }
        res = validate_coverage_payload(payload, query, self.inventory)
        self.assertEqual(res["status"], "partial")
        self.assertEqual(len(res["parts"]), 2)
        self.assertEqual(res["parts"][0]["start_char"], 0)
        self.assertEqual(res["parts"][0]["end_char"], len(part1))
        self.assertEqual(res["parts"][1]["start_char"], len(part1) + 1)
        self.assertEqual(res["parts"][1]["end_char"], len(query))

    def test_leading_trailing_and_inner_whitespace_allowed(self):
        query = "  \t  ¿Gasto previsto? \n\n  \t ¿Evolución de ingresos?   \n"
        part1 = "¿Gasto previsto?"
        part2 = "¿Evolución de ingresos?"
        payload = {
            "parts": [
                {
                    "query_span": part1,
                    "decision": "answered",
                    "claim_ids": ["C1"],
                    "reason_code": "covered",
                },
                {
                    "query_span": part2,
                    "decision": "answered",
                    "claim_ids": ["C2"],
                    "reason_code": "covered",
                },
            ]
        }
        res = validate_coverage_payload(payload, query, self.inventory)
        self.assertEqual(res["status"], "answered")
        self.assertEqual(res["parts"][0]["start_char"], query.find(part1))
        self.assertEqual(res["parts"][1]["start_char"], query.find(part2))

    def test_valid_three_parts_partition(self):
        query = "¿Pregunta uno, pregunta dos y pregunta tres?"
        parts_spans = ["¿Pregunta uno,", "pregunta dos", "y pregunta tres?"]
        payload = {
            "parts": [
                {"query_span": parts_spans[0], "decision": "answered", "claim_ids": ["C1"], "reason_code": "covered"},
                {"query_span": parts_spans[1], "decision": "not_answered", "claim_ids": [], "reason_code": "scope_mismatch"},
                {"query_span": parts_spans[2], "decision": "not_answered", "claim_ids": [], "reason_code": "missing_from_answer"},
            ]
        }
        res = validate_coverage_payload(payload, query, self.inventory)
        self.assertEqual(res["status"], "partial")
        self.assertEqual(len(res["parts"]), 3)

    def test_valid_four_parts_partition(self):
        query = "¿Dato uno? ¿Dato dos? ¿Dato tres? ¿Dato cuatro?"
        spans = ["¿Dato uno?", "¿Dato dos?", "¿Dato tres?", "¿Dato cuatro?"]
        payload = {
            "parts": [
                {"query_span": spans[0], "decision": "answered", "claim_ids": ["C1"], "reason_code": "covered"},
                {"query_span": spans[1], "decision": "answered", "claim_ids": ["C2"], "reason_code": "covered"},
                {"query_span": spans[2], "decision": "not_answered", "claim_ids": [], "reason_code": "not_in_evaluated_evidence"},
                {"query_span": spans[3], "decision": "not_answered", "claim_ids": [], "reason_code": "scope_mismatch"},
            ]
        }
        res = validate_coverage_payload(payload, query, self.inventory)
        self.assertEqual(res["status"], "partial")
        self.assertEqual(len(res["parts"]), 4)

    def test_more_than_four_parts_rejected(self):
        query = "p1 p2 p3 p4 p5"
        payload = {
            "parts": [
                {"query_span": f"p{i}", "decision": "not_answered", "claim_ids": [], "reason_code": "not_in_evaluated_evidence"}
                for i in range(1, 6)
            ]
        }
        with self.assertRaises(ValueError) as ctx:
            validate_coverage_payload(payload, query, self.inventory)
        self.assertIn("length must be between 1 and 4", str(ctx.exception))

    def test_nonwhitespace_omission_inner_gap_rejects(self):
        query = "¿Gasto en pensiones, y déficit estructural?"
        payload = {
            "parts": [
                {"query_span": "¿Gasto en pensiones", "decision": "answered", "claim_ids": ["C1"], "reason_code": "covered"},
                {"query_span": "déficit estructural?", "decision": "not_answered", "claim_ids": [], "reason_code": "not_in_evaluated_evidence"},
            ]
        }
        with self.assertRaises(ValueError) as ctx:
            validate_coverage_payload(payload, query, self.inventory)
        self.assertIn("Non-whitespace characters omitted in query between", str(ctx.exception))

    def test_nonwhitespace_omission_at_start_rejects(self):
        query = "Consulta: ¿Cuál es el gasto previsto?"
        payload = {
            "parts": [
                {"query_span": "¿Cuál es el gasto previsto?", "decision": "answered", "claim_ids": ["C1"], "reason_code": "covered"},
            ]
        }
        with self.assertRaises(ValueError) as ctx:
            validate_coverage_payload(payload, query, self.inventory)
        self.assertIn("Non-whitespace characters omitted in query between", str(ctx.exception))

    def test_nonwhitespace_omission_at_end_rejects(self):
        query = "¿Cuál es el gasto previsto en 2050 según la proyección?"
        payload = {
            "parts": [
                {"query_span": "¿Cuál es el gasto previsto en 2050", "decision": "answered", "claim_ids": ["C1"], "reason_code": "covered"},
            ]
        }
        with self.assertRaises(ValueError) as ctx:
            validate_coverage_payload(payload, query, self.inventory)
        self.assertIn("omitted at end of query", str(ctx.exception))

    def test_overlapping_fragments_rejects(self):
        query = "¿Año de aceleración y tasa de crecimiento?"
        payload = {
            "parts": [
                {"query_span": "¿Año de aceleración y tasa", "decision": "answered", "claim_ids": ["C1"], "reason_code": "covered"},
                {"query_span": "tasa de crecimiento?", "decision": "not_answered", "claim_ids": [], "reason_code": "not_in_evaluated_evidence"},
            ]
        }
        with self.assertRaises(ValueError) as ctx:
            validate_coverage_payload(payload, query, self.inventory)
        self.assertIn("not found sequentially in original query after index", str(ctx.exception))

    def test_reversed_order_fragments_rejects(self):
        query = "¿Año de aceleración y tasa de crecimiento?"
        payload = {
            "parts": [
                {"query_span": "tasa de crecimiento?", "decision": "not_answered", "claim_ids": [], "reason_code": "not_in_evaluated_evidence"},
                {"query_span": "¿Año de aceleración y", "decision": "answered", "claim_ids": ["C1"], "reason_code": "covered"},
            ]
        }
        with self.assertRaises(ValueError) as ctx:
            validate_coverage_payload(payload, query, self.inventory)
        self.assertIn("Non-whitespace characters omitted in query between 0 and", str(ctx.exception))

    def test_changed_accent_or_character_rejects(self):
        query = "¿Cuál es el cálculo según la AIReF?"
        payload = {
            "parts": [
                {"query_span": "¿Cuál es el calculo según la AIReF?", "decision": "answered", "claim_ids": ["C1"], "reason_code": "covered"},
            ]
        }
        with self.assertRaises(ValueError) as ctx:
            validate_coverage_payload(payload, query, self.inventory)
        self.assertIn("not found sequentially", str(ctx.exception))


class TestAnswerCoveredDirect(unittest.TestCase):
    @patch("experiments.query_coverage.answer_preserved")
    def test_oversize_audit_retains_factual_report(self, factual):
        factual.return_value = copy.deepcopy(self.valid_factual_report)
        verifier = Q3VerifierClient(transport=MockTransport())
        with patch("experiments.query_coverage.build_coverage_messages", side_effect=ValueError("budget exceeded")):
            report = answer_covered({}, self.query, {}, [], verifier=verifier)
        self.assertEqual(report["status"], "error")
        self.assertIn("budget exceeded", report["error"])
        self.assertEqual(report["factual_report"], self.valid_factual_report)
        self.assertEqual(report["publication"]["claims"], [])
        self.assertEqual(report["coverage_audit"]["calls"], [])
        self.assertEqual(verifier.calls, [])

    @patch("experiments.query_coverage.answer_preserved")
    def test_invalid_inventory_retains_factual_report(self, factual):
        broken = copy.deepcopy(self.valid_factual_report)
        broken["outcome"]["publication"]["claims"][0]["text"] = "misaligned"
        factual.return_value = broken
        verifier = Q3VerifierClient(transport=MockTransport())
        report = answer_covered({}, self.query, {}, [], verifier=verifier)
        self.assertEqual(report["status"], "error")
        self.assertEqual(report["factual_report"], broken)
        self.assertEqual(report["publication"]["claims"], [])
        self.assertEqual(verifier.calls, [])

    """Verifies direct answer_covered execution across all gate states with immutable input preservation."""

    def setUp(self):
        self.query = "¿Cuál es el gasto previsto en 2050 y cuántas mujeres se jubilarán?"
        self.index = {
            "model": "Qwen/Qwen3-Embedding4B",
            "model_digest": "a" * 64,
            "chunks": [],
        }
        self.units = {"catalog": {"U1": {"unit_id": "U1"}}}
        self.pages = [{"page_number": 1, "text": "Texto de página."}]

        self.valid_factual_report = {
            "schema_version": "preserved-direct-query-v1",
            "query": self.query,
            "outcome": {
                "status": "answered",
                "expanded_hits": [
                    {"rank": 1, "chunk": {"page_number": 95, "text": "Evidencia 1"}}
                ],
                "full_result": {
                    "grounding": {
                        "accepted_claims": [
                            {
                                "claim_id": "C1",
                                "text": "El gasto previsto en pensiones alcanzará el 15% del PIB en 2050.",
                                "evidence_ids": ["E1"],
                            }
                        ]
                    }
                },
                "publication": {
                    "status": "answered",
                    "claims": [
                        {
                            "text": "El gasto previsto en pensiones alcanzará el 15% del PIB en 2050.",
                            "evidence": [
                                {"source_id": "S1", "quote": "El gasto en pensiones alcanzará el 15% del PIB en 2050."}
                            ],
                        }
                    ],
                },
            },
        }

    @patch("experiments.query_coverage.answer_preserved")
    def test_answered_state(self, mock_answer_preserved):
        mock_answer_preserved.return_value = copy.deepcopy(self.valid_factual_report)
        audit_payload = {
            "parts": [
                {
                    "query_span": self.query,
                    "decision": "answered",
                    "claim_ids": ["C1"],
                    "reason_code": "covered",
                }
            ]
        }
        transport = MockTransport(chat_response=_make_valid_raw_response(audit_payload))
        verifier = Q3VerifierClient(expected_digest=VERIFIER_Q3_DIGEST, transport=transport)
        mock_embedder = MagicMock()
        mock_generator = MagicMock()

        index_before = copy.deepcopy(self.index)
        units_before = copy.deepcopy(self.units)
        pages_before = copy.deepcopy(self.pages)
        factual_before = copy.deepcopy(self.valid_factual_report)

        report = answer_covered(
            self.index,
            self.query,
            self.units,
            self.pages,
            embedder=mock_embedder,
            generator=mock_generator,
            verifier=verifier,
        )

        self.assertEqual(self.index, index_before)
        self.assertEqual(self.units, units_before)
        self.assertEqual(self.pages, pages_before)
        self.assertEqual(self.valid_factual_report, factual_before)
        mock_answer_preserved.assert_called_once_with(
            self.index,
            self.query,
            self.units,
            self.pages,
            embedder=mock_embedder,
            generator=mock_generator,
            verifier=verifier,
            partial_generation=True,
        )

        self.assertEqual(report["schema_version"], PRESERVED_COVERAGE_SCHEMA_VERSION)
        self.assertEqual(report["status"], "answered")
        self.assertEqual(report["query"], self.query)
        self.assertEqual(report["factual_report"], self.valid_factual_report)
        self.assertIsNone(report["coverage_audit"]["error"])
        self.assertFalse(report["coverage_audit"]["skipped"])

        self.assertEqual(report["metadata"]["verifier_model"], VERIFIER_Q3_MODEL)
        self.assertEqual(report["metadata"]["verifier_digest"], VERIFIER_Q3_DIGEST)
        self.assertEqual(report["metadata"]["factual_status"], "answered")
        self.assertEqual(report["metadata"]["coverage_status"], "answered")

        audit_calls = report["coverage_audit"]["calls"]
        self.assertEqual(len(audit_calls), 1)
        self.assertEqual(audit_calls[0]["stage"], COVERAGE_STAGE)
        self.assertEqual(audit_calls[0]["model"], VERIFIER_Q3_MODEL)

        pub_claims = report["publication"]["claims"]
        self.assertEqual(len(pub_claims), 1)
        self.assertEqual(pub_claims[0]["text"], self.valid_factual_report["outcome"]["publication"]["claims"][0]["text"])
        self.assertEqual(pub_claims[0]["evidence"], self.valid_factual_report["outcome"]["publication"]["claims"][0]["evidence"])
        self.assertEqual(report["publication"]["missing_parts"], [])

    @patch("experiments.query_coverage.answer_preserved")
    def test_mixed_partial_and_visible_missing(self, mock_answer_preserved):
        mock_answer_preserved.return_value = copy.deepcopy(self.valid_factual_report)
        part1 = "¿Cuál es el gasto previsto en 2050"
        part2 = "y cuántas mujeres se jubilarán?"
        audit_payload = {
            "parts": [
                {
                    "query_span": part1,
                    "decision": "answered",
                    "claim_ids": ["C1"],
                    "reason_code": "covered",
                },
                {
                    "query_span": part2,
                    "decision": "not_answered",
                    "claim_ids": [],
                    "reason_code": "not_in_evaluated_evidence",
                },
            ]
        }
        transport = MockTransport(chat_response=_make_valid_raw_response(audit_payload))
        verifier = Q3VerifierClient(expected_digest=VERIFIER_Q3_DIGEST, transport=transport)

        index_before = copy.deepcopy(self.index)
        units_before = copy.deepcopy(self.units)
        pages_before = copy.deepcopy(self.pages)

        report = answer_covered(
            self.index,
            self.query,
            self.units,
            self.pages,
            verifier=verifier,
        )

        self.assertEqual(self.index, index_before)
        self.assertEqual(self.units, units_before)
        self.assertEqual(self.pages, pages_before)

        self.assertEqual(report["status"], "partial")
        self.assertEqual(report["publication"]["status"], "partial")

        self.assertEqual(len(report["publication"]["claims"]), 1)
        self.assertEqual(
            report["publication"]["claims"][0]["text"],
            self.valid_factual_report["outcome"]["publication"]["claims"][0]["text"],
        )
        self.assertEqual(
            report["publication"]["claims"][0]["evidence"],
            self.valid_factual_report["outcome"]["publication"]["claims"][0]["evidence"],
        )

        missing = report["publication"]["missing_parts"]
        self.assertEqual(len(missing), 1)
        self.assertEqual(missing[0]["query_span"], part2)
        self.assertEqual(missing[0]["reason_code"], "not_in_evaluated_evidence")
        self.assertIn("«y cuántas mujeres se jubilarán?»", missing[0]["message"])

        rendered = format_covered_answer(report)
        self.assertIn(self.valid_factual_report["outcome"]["publication"]["claims"][0]["text"], rendered)
        self.assertIn("Partes no respondidas en las fuentes evaluadas:", rendered)
        self.assertIn("«y cuántas mujeres se jubilarán?»", rendered)

    @patch("experiments.query_coverage.answer_preserved")
    def test_empty_inventory_no_chat(self, mock_answer_preserved):
        empty_factual = copy.deepcopy(self.valid_factual_report)
        empty_factual["outcome"]["status"] = "insufficient_evidence"
        empty_factual["outcome"]["full_result"]["grounding"]["accepted_claims"] = []
        empty_factual["outcome"]["publication"]["claims"] = []
        empty_factual["outcome"]["publication"]["status"] = "insufficient_evidence"
        mock_answer_preserved.return_value = empty_factual

        transport = MockTransport(chat_response=_make_valid_raw_response({"parts": []}))
        verifier = Q3VerifierClient(expected_digest=VERIFIER_Q3_DIGEST, transport=transport)

        index_before = copy.deepcopy(self.index)
        units_before = copy.deepcopy(self.units)
        pages_before = copy.deepcopy(self.pages)

        report = answer_covered(
            self.index,
            self.query,
            self.units,
            self.pages,
            verifier=verifier,
        )

        self.assertEqual(self.index, index_before)
        self.assertEqual(self.units, units_before)
        self.assertEqual(self.pages, pages_before)

        self.assertEqual(len(transport.calls), 0)
        self.assertTrue(report["coverage_audit"]["skipped"])
        self.assertEqual(report["coverage_audit"]["calls"], [])

        self.assertEqual(report["status"], "insufficient_evidence")
        self.assertEqual(report["publication"]["claims"], [])
        self.assertEqual(len(report["publication"]["missing_parts"]), 1)
        self.assertEqual(report["publication"]["missing_parts"][0]["query_span"], self.query.strip())
        self.assertEqual(report["publication"]["missing_parts"][0]["reason_code"], "not_in_evaluated_evidence")

    @patch("experiments.query_coverage.answer_preserved")
    def test_factual_error_no_audit(self, mock_answer_preserved):
        err_report = {
            "schema_version": "preserved-direct-query-v1",
            "query": self.query,
            "outcome": {
                "status": "error",
                "error": "Retrieval vector failure",
            },
        }
        mock_answer_preserved.return_value = copy.deepcopy(err_report)

        transport = MockTransport(chat_response=_make_valid_raw_response({"parts": []}))
        verifier = Q3VerifierClient(expected_digest=VERIFIER_Q3_DIGEST, transport=transport)

        index_before = copy.deepcopy(self.index)
        units_before = copy.deepcopy(self.units)
        pages_before = copy.deepcopy(self.pages)

        report = answer_covered(
            self.index,
            self.query,
            self.units,
            self.pages,
            verifier=verifier,
        )

        self.assertEqual(self.index, index_before)
        self.assertEqual(self.units, units_before)
        self.assertEqual(self.pages, pages_before)

        self.assertEqual(len(transport.calls), 0)
        self.assertIsNone(report["coverage_audit"])
        self.assertEqual(report["status"], "error")
        self.assertIn("Factual grounding error: Retrieval vector failure", report["error"])
        self.assertEqual(report["publication"]["status"], "error")
        self.assertEqual(report["publication"]["claims"], [])
        self.assertEqual(report["metadata"]["factual_status"], "error")
        self.assertEqual(report["metadata"]["coverage_status"], "error")

    @patch("experiments.query_coverage.answer_preserved")
    def test_raw_coverage_failure_preserves_calls_and_no_unchecked_claims(self, mock_answer_preserved):
        mock_answer_preserved.return_value = copy.deepcopy(self.valid_factual_report)

        transport = MockTransport(chat_error=RuntimeError("Ollama verifier timeout 504"))
        verifier = Q3VerifierClient(expected_digest=VERIFIER_Q3_DIGEST, transport=transport)

        index_before = copy.deepcopy(self.index)
        units_before = copy.deepcopy(self.units)
        pages_before = copy.deepcopy(self.pages)

        report = answer_covered(
            self.index,
            self.query,
            self.units,
            self.pages,
            verifier=verifier,
        )

        self.assertEqual(self.index, index_before)
        self.assertEqual(self.units, units_before)
        self.assertEqual(self.pages, pages_before)

        self.assertEqual(len(transport.calls), 1)
        self.assertEqual(report["status"], "error")
        self.assertIn("Coverage audit error: Ollama verifier timeout 504", report["error"])

        audit = report["coverage_audit"]
        self.assertIsNotNone(audit)
        self.assertEqual(audit["status"], "error")
        self.assertIn("Ollama verifier timeout 504", audit["error"])
        self.assertEqual(len(audit["calls"]), 1)
        self.assertEqual(audit["calls"][0]["stage"], COVERAGE_STAGE)
        self.assertEqual(audit["calls"][0]["model"], VERIFIER_Q3_MODEL)

        self.assertEqual(report["publication"]["status"], "error")
        self.assertEqual(report["publication"]["claims"], [])
        self.assertEqual(report["publication"]["missing_parts"], [])
        self.assertEqual(report["metadata"]["factual_status"], "answered")
        self.assertEqual(report["metadata"]["coverage_status"], "error")


class _SpyCoverageVerifier:
    """Mock verifier capturing chat calls made by audit_coverage for transport/schema validation."""

    def __init__(self, response_payload: dict, model: str = VERIFIER_Q3_MODEL) -> None:
        self.model = model
        self.response_payload = response_payload
        self.calls: list[dict] = []

    def chat(self, messages: list[dict], schema: dict, stage: str = COVERAGE_STAGE) -> dict:
        call = {
            "stage": stage,
            "model": self.model,
            "messages": copy.deepcopy(messages),
            "schema": copy.deepcopy(schema),
        }
        self.calls.append(call)
        return _make_valid_raw_response(self.response_payload)


def _assert_coverage_system_prompt_rules(test_case: unittest.TestCase, system_text: str) -> None:
    """Independent assertions verifying that system prompt contains required route-contract rules.

    Does not import or match COVERAGE_SYSTEM_PROMPT constant verbatim, ensuring tests
    fail if prompt is reverted to legacy wording lacking quote-only / claim-text rules.
    """
    test_case.assertTrue(
        "texto de la afirmación" in system_text or "texto de las afirmaciones" in system_text,
        "System prompt must instruct that the claim text itself must answer the question",
    )
    test_case.assertTrue(
        "dato" in system_text and "relación" in system_text,
        "System prompt must require both requested datum and relation in claim text",
    )
    test_case.assertTrue(
        "Las citas respaldan la afirmación pero no responden por sí solas lo omitido" in system_text
        or "citas respaldan la afirmación" in system_text,
        "System prompt must clarify citations do not substitute omitted answer text",
    )
    test_case.assertTrue(
        "missing_from_answer" in system_text,
        "System prompt must specify missing_from_answer reason code",
    )
    test_case.assertTrue(
        any(p in system_text for p in [
            "solo aparece en las citas",
            "solo figura en las citas",
            "dato solicitado solo",
        ]) and any(p in system_text for p in ["omitió", "omite", "omitido"]),
        "System prompt must instruct that datum present only in quotes and omitted from claim is missing_from_answer",
    )
    test_case.assertTrue(
        "scope_mismatch" in system_text and "sustituy" in system_text,
        "System prompt must instruct that substituted population/year/unit is scope_mismatch",
    )
    test_case.assertTrue(
        "paráfrasis" in system_text,
        "System prompt must explicitly permit genuine paraphrases preserving exact meaning",
    )


class TestSyntheticQueryCoverageContractSemantics(unittest.TestCase):
    """Offline unit tests exercising coverage audit contract semantics with synthetic data.

    Note: Actual LLM semantic validation of prompt interpretation remains for parent's
    fresh live inference; offline unit tests verify deterministic validation, interface
    invariants, schema compliance, immutability, and payload processing without scripted fake outputs.
    """

    def setUp(self):
        self.synthetic_query = "¿En qué ejercicio anterior se situó la tasa de cobertura de suelo?"
        self.synthetic_quote_1994 = "La tasa de cobertura de suelo fue del 35% en 1994, cota histórica a la que retornaría."
        self.synthetic_quote_2045 = "La tasa de cobertura proyectada alcanzará el 48% en 2045."

    def test_quote_only_omitted_answer_marked_missing_from_answer(self):
        """When evidence quote contains requested datum (1994) but claim text omits it, coverage marks not_answered / missing_from_answer."""
        inventory = [
            {
                "claim_id": "C1",
                "text": "La tasa de cobertura de suelo se proyecta en el 48% hacia el final del periodo.",
                "evidence": [
                    {"source_id": "S1", "quote": self.synthetic_quote_1994},
                    {"source_id": "S1", "quote": self.synthetic_quote_2045},
                ],
                "publication_claim": {
                    "text": "La tasa de cobertura de suelo se proyecta en el 48% hacia el final del periodo.",
                    "evidence": [
                        {"source_id": "S1", "quote": self.synthetic_quote_1994},
                        {"source_id": "S1", "quote": self.synthetic_quote_2045},
                    ],
                },
            }
        ]
        scripted_payload = {
            "parts": [
                {
                    "query_span": self.synthetic_query,
                    "decision": "not_answered",
                    "claim_ids": [],
                    "reason_code": "missing_from_answer",
                }
            ]
        }
        spy = _SpyCoverageVerifier(scripted_payload)
        audit_res = audit_coverage(self.synthetic_query, inventory, verifier=spy)
        self.assertEqual(audit_res["status"], "completed")

        # 1. Assert effective chat messages and schema transmitted
        self.assertEqual(len(spy.calls), 1)
        sent_messages = spy.calls[0]["messages"]
        sent_schema = spy.calls[0]["schema"]
        self.assertEqual(sent_schema, build_coverage_schema(self.synthetic_query, inventory))

        # 2. Assert effective SYSTEM message contains required rules independently
        _assert_coverage_system_prompt_rules(self, sent_messages[0]["content"])

        # 3. Assert main claim text and quote evidence transmitted literally and independently
        user_payload = json.loads(sent_messages[1]["content"])
        self.assertEqual(user_payload["consulta_original"], self.synthetic_query)
        self.assertEqual(len(user_payload["afirmaciones_aceptadas"]), 1)
        accepted_item = user_payload["afirmaciones_aceptadas"][0]
        self.assertEqual(accepted_item["claim_id"], "C1")
        self.assertEqual(accepted_item["text"], inventory[0]["text"])
        self.assertEqual(
            [e["quote"] for e in accepted_item["evidence"]],
            [self.synthetic_quote_1994, self.synthetic_quote_2045],
        )

        # 4. Assert validated payload and publication views
        validated = audit_res["validated_payload"]
        self.assertEqual(validated["status"], "insufficient_evidence")
        self.assertEqual(validated["parts"][0]["decision"], "not_answered")
        self.assertEqual(validated["parts"][0]["reason_code"], "missing_from_answer")

        pub = build_coverage_publication(self.synthetic_query, inventory, validated)
        self.assertEqual(pub["status"], "insufficient_evidence")
        self.assertEqual(pub["claims"], [])
        self.assertEqual(len(pub["missing_parts"]), 1)
        self.assertEqual(pub["missing_parts"][0]["reason_code"], "missing_from_answer")
        self.assertIn("La respuesta fundamentada no contiene datos respaldados para:", pub["missing_parts"][0]["message"])

    def test_explicit_main_answer_marked_covered(self):
        """When claim text explicitly states requested datum (1994) with requested relation, coverage marks answered / covered."""
        inventory = [
            {
                "claim_id": "C1",
                "text": "La tasa de cobertura de suelo retornaría a los niveles alcanzados en 1994.",
                "evidence": [
                    {"source_id": "S1", "quote": self.synthetic_quote_1994},
                ],
                "publication_claim": {
                    "text": "La tasa de cobertura de suelo retornaría a los niveles alcanzados en 1994.",
                    "evidence": [
                        {"source_id": "S1", "quote": self.synthetic_quote_1994},
                    ],
                },
            }
        ]
        scripted_payload = {
            "parts": [
                {
                    "query_span": self.synthetic_query,
                    "decision": "answered",
                    "claim_ids": ["C1"],
                    "reason_code": "covered",
                }
            ]
        }
        spy = _SpyCoverageVerifier(scripted_payload)
        audit_res = audit_coverage(self.synthetic_query, inventory, verifier=spy)
        self.assertEqual(audit_res["status"], "completed")

        # 1. Assert effective chat messages and schema
        self.assertEqual(len(spy.calls), 1)
        sent_messages = spy.calls[0]["messages"]
        _assert_coverage_system_prompt_rules(self, sent_messages[0]["content"])

        # 2. Assert main claim text and quote evidence transmitted literally and independently
        user_payload = json.loads(sent_messages[1]["content"])
        self.assertEqual(user_payload["consulta_original"], self.synthetic_query)
        accepted_item = user_payload["afirmaciones_aceptadas"][0]
        self.assertEqual(accepted_item["text"], inventory[0]["text"])
        self.assertEqual(accepted_item["evidence"][0]["quote"], self.synthetic_quote_1994)

        # 3. Assert validated payload and publication views
        validated = audit_res["validated_payload"]
        self.assertEqual(validated["status"], "answered")
        self.assertEqual(validated["parts"][0]["decision"], "answered")
        self.assertEqual(validated["parts"][0]["reason_code"], "covered")

        pub = build_coverage_publication(self.synthetic_query, inventory, validated)
        self.assertEqual(pub["status"], "answered")
        self.assertEqual(len(pub["claims"]), 1)
        self.assertEqual(pub["claims"][0]["text"], inventory[0]["text"])
        self.assertEqual(pub["missing_parts"], [])

    def test_valid_paraphrase_accepted_as_covered(self):
        """Genuine paraphrases preserving the exact requested datum and relation are accepted as covered through message path."""
        query = "¿En qué año previo volvería el registro a su valor histórico?"
        inventory = [
            {
                "claim_id": "C1",
                "text": "El registro volvería a cotas equiparables a las de 1994.",
                "evidence": [{"source_id": "S1", "quote": self.synthetic_quote_1994}],
                "publication_claim": {
                    "text": "El registro volvería a cotas equiparables a las de 1994.",
                    "evidence": [{"source_id": "S1", "quote": self.synthetic_quote_1994}],
                },
            }
        ]
        scripted_payload = {
            "parts": [
                {
                    "query_span": query,
                    "decision": "answered",
                    "claim_ids": ["C1"],
                    "reason_code": "covered",
                }
            ]
        }
        spy = _SpyCoverageVerifier(scripted_payload)
        audit_res = audit_coverage(query, inventory, verifier=spy)
        self.assertEqual(audit_res["status"], "completed")

        # Assert messages and rules
        sent_messages = spy.calls[0]["messages"]
        _assert_coverage_system_prompt_rules(self, sent_messages[0]["content"])
        user_payload = json.loads(sent_messages[1]["content"])
        self.assertEqual(user_payload["consulta_original"], query)
        self.assertEqual(user_payload["afirmaciones_aceptadas"][0]["text"], inventory[0]["text"])

        validated = audit_res["validated_payload"]
        self.assertEqual(validated["status"], "answered")
        pub = build_coverage_publication(query, inventory, validated)
        self.assertEqual(pub["status"], "answered")
        self.assertEqual(len(pub["claims"]), 1)

    def test_wrong_relation_despite_same_tokens_rejected(self):
        """When tokens match (1994 and cobertura) but assert wrong relation, coverage rejects through audit path."""
        inventory = [
            {
                "claim_id": "C1",
                "text": "La tasa de cobertura de suelo comenzó a medirse en 1994 tras la reforma agraria.",
                "evidence": [{"source_id": "S1", "quote": "En 1994 comenzó la serie tras la reforma agraria."}],
                "publication_claim": {
                    "text": "La tasa de cobertura de suelo comenzó a medirse en 1994 tras la reforma agraria.",
                    "evidence": [{"source_id": "S1", "quote": "En 1994 comenzó la serie tras la reforma agraria."}],
                },
            }
        ]
        scripted_payload = {
            "parts": [
                {
                    "query_span": self.synthetic_query,
                    "decision": "not_answered",
                    "claim_ids": [],
                    "reason_code": "missing_from_answer",
                }
            ]
        }
        spy = _SpyCoverageVerifier(scripted_payload)
        audit_res = audit_coverage(self.synthetic_query, inventory, verifier=spy)
        self.assertEqual(audit_res["status"], "completed")

        sent_messages = spy.calls[0]["messages"]
        _assert_coverage_system_prompt_rules(self, sent_messages[0]["content"])
        user_payload = json.loads(sent_messages[1]["content"])
        self.assertEqual(user_payload["afirmaciones_aceptadas"][0]["text"], inventory[0]["text"])

        validated = audit_res["validated_payload"]
        self.assertEqual(validated["status"], "insufficient_evidence")
        pub = build_coverage_publication(self.synthetic_query, inventory, validated)
        self.assertEqual(pub["claims"], [])
        self.assertEqual(len(pub["missing_parts"]), 1)

    def test_substituted_population_year_or_unit_rejected_as_scope_mismatch(self):
        """Substituting subpopulation, year horizon, or unit is rejected with scope_mismatch through audit path."""
        query = "¿Cuál fue la tasa de cobertura forestal en 1994?"
        inventory = [
            {
                "claim_id": "C1",
                "text": "La cobertura vegetal agregada general fue del 62% en 2008.",
                "evidence": [{"source_id": "S1", "quote": "La cobertura vegetal general alcanzó el 62% en 2008."}],
                "publication_claim": {
                    "text": "La cobertura vegetal agregada general fue del 62% en 2008.",
                    "evidence": [{"source_id": "S1", "quote": "La cobertura vegetal general alcanzó el 62% en 2008."}],
                },
            }
        ]
        scripted_payload = {
            "parts": [
                {
                    "query_span": query,
                    "decision": "not_answered",
                    "claim_ids": [],
                    "reason_code": "scope_mismatch",
                }
            ]
        }
        spy = _SpyCoverageVerifier(scripted_payload)
        audit_res = audit_coverage(query, inventory, verifier=spy)
        self.assertEqual(audit_res["status"], "completed")

        sent_messages = spy.calls[0]["messages"]
        _assert_coverage_system_prompt_rules(self, sent_messages[0]["content"])
        user_payload = json.loads(sent_messages[1]["content"])
        self.assertEqual(user_payload["afirmaciones_aceptadas"][0]["text"], inventory[0]["text"])

        validated = audit_res["validated_payload"]
        self.assertEqual(validated["status"], "insufficient_evidence")
        self.assertEqual(validated["parts"][0]["reason_code"], "scope_mismatch")
        pub = build_coverage_publication(query, inventory, validated)
        self.assertIn("Las evidencias evaluadas no corresponden al alcance solicitado en:", pub["reason"])

    def test_negative_control_prior_prompt_without_quote_only_rule_fails_route_assertion(self):
        """Demonstrates that _assert_coverage_system_prompt_rules fails on prior prompt lacking quote-only rules."""
        prior_prompt = (
            "Auditor de cobertura semántica de consultas RAG sobre pensiones.\n"
            "Tu función es auditar la correspondencia y completitud entre la consulta original y las afirmaciones aceptadas.\n"
            "REGLAS ESTRUCTURALES:\n"
            "1. Divide la consulta original en 1 a 4 partes disjuntas ('query_span') que cubran exactamente todo el texto de la consulta.\n"
            "2. Para cada parte determina la decisión ('answered' o 'not_answered'):\n"
            "   - 'answered': la parte queda genuinamente respondida por las afirmaciones aceptadas asociadas. Requiere 'claim_ids' no vacío y reason_code='covered'.\n"
            "   - 'not_answered': no hay afirmaciones que respondan esta necesidad con exactitud semántica. Requiere 'claim_ids' vacío y reason_code entre: 'scope_mismatch', 'not_in_evaluated_evidence', 'missing_from_answer'.\n"
            "3. CORRESPONDENCIA SEMÁNTICA:\n"
            "   - Verifica correspondencia estricta de: población, magnitud, unidad, periodo, observado frente a proyectado, total frente a componente y relaciones causales.\n"
            "   - La mera coincidencia de palabras (ej. 'mujeres' y un año) es insuficiente si los datos corresponden a totales generales o periodos distintos.\n"
            "   - Se admiten paráfrasis válidas si conservan el significado exacto.\n"
            "   - Si los datos evaluados sustituyen una subpoblación u horizonte por otro (ej. totales en vez de mujeres, o 2055 en vez de 2050), marca 'not_answered' con 'scope_mismatch'.\n"
            "4. NO inventes hechos, afirmaciones nuevas ni texto libre de respuesta en la salida."
        )
        with self.assertRaises(AssertionError) as ctx:
            _assert_coverage_system_prompt_rules(self, prior_prompt)
        self.assertIn("claim text itself must answer", str(ctx.exception))

    def test_mixed_known_and_unknown_parts_yields_partial_no_invented_number(self):
        """Mixed query yields partial coverage, retains supported claim, and produces visible refusal without invented numbers."""
        compound_query = "¿En qué ejercicio anterior se situó la tasa de cobertura de suelo y cuál sería la cuota exacta de la finca 88 en 2045?"
        part1 = "¿En qué ejercicio anterior se situó la tasa de cobertura de suelo"
        part2 = " y cuál sería la cuota exacta de la finca 88 en 2045?"

        inventory = [
            {
                "claim_id": "C1",
                "text": "La tasa de cobertura de suelo retornaría a los niveles alcanzados en 1994.",
                "evidence": [{"source_id": "S1", "quote": self.synthetic_quote_1994}],
                "publication_claim": {
                    "text": "La tasa de cobertura de suelo retornaría a los niveles alcanzados en 1994.",
                    "evidence": [{"source_id": "S1", "quote": self.synthetic_quote_1994}],
                },
            }
        ]
        scripted_payload = {
            "parts": [
                {
                    "query_span": part1,
                    "decision": "answered",
                    "claim_ids": ["C1"],
                    "reason_code": "covered",
                },
                {
                    "query_span": part2,
                    "decision": "not_answered",
                    "claim_ids": [],
                    "reason_code": "not_in_evaluated_evidence",
                },
            ]
        }
        spy = _SpyCoverageVerifier(scripted_payload)
        audit_res = audit_coverage(compound_query, inventory, verifier=spy)
        self.assertEqual(audit_res["status"], "completed")

        sent_messages = spy.calls[0]["messages"]
        _assert_coverage_system_prompt_rules(self, sent_messages[0]["content"])
        user_payload = json.loads(sent_messages[1]["content"])
        self.assertEqual(user_payload["consulta_original"], compound_query)
        self.assertEqual(user_payload["afirmaciones_aceptadas"][0]["text"], inventory[0]["text"])

        validated = audit_res["validated_payload"]
        self.assertEqual(validated["status"], "partial")

        pub = build_coverage_publication(compound_query, inventory, validated)
        self.assertEqual(pub["status"], "partial")
        self.assertEqual(len(pub["claims"]), 1)
        self.assertEqual(pub["claims"][0]["text"], inventory[0]["text"])
        # No invented individual number for finca 88
        for c in pub["claims"]:
            self.assertNotIn("finca 88", c["text"])
            self.assertNotIn("cuota exacta", c["text"])

        report = {
            "status": "partial",
            "publication": pub,
            "factual_report": {
                "outcome": {
                    "expanded_hits": [{"rank": 1, "chunk": {"page_number": 12, "text": "t"}}]
                }
            },
        }
        rendered = format_covered_answer(report)
        self.assertIn("1994", rendered)
        self.assertIn("Partes no respondidas en las fuentes evaluadas:", rendered)
        self.assertIn("« y cuál sería la cuota exacta de la finca 88 en 2045?»", rendered)

    def test_immutability_of_messages_inventory_and_quotes(self):
        """Coverage pipeline functions do not mutate query, inventory, or source quotes."""
        inventory = [
            {
                "claim_id": "C1",
                "text": "Afirmación sintética de prueba.",
                "evidence": [{"source_id": "S1", "quote": self.synthetic_quote_1994}],
                "publication_claim": {
                    "text": "Afirmación sintética de prueba.",
                    "evidence": [{"source_id": "S1", "quote": self.synthetic_quote_1994}],
                },
            }
        ]
        query = self.synthetic_query
        inv_snapshot = copy.deepcopy(inventory)
        query_snapshot = copy.deepcopy(query)

        # 1. build_coverage_messages
        msgs = build_coverage_messages(query, inventory)
        self.assertEqual(inventory, inv_snapshot)
        self.assertEqual(query, query_snapshot)

        # 2. validate_coverage_payload
        payload = {
            "parts": [
                {
                    "query_span": query,
                    "decision": "answered",
                    "claim_ids": ["C1"],
                    "reason_code": "covered",
                }
            ]
        }
        payload_snapshot = copy.deepcopy(payload)
        validated = validate_coverage_payload(payload, query, inventory)
        self.assertEqual(inventory, inv_snapshot)
        self.assertEqual(query, query_snapshot)
        self.assertEqual(payload, payload_snapshot)

        # 3. build_coverage_publication
        pub = build_coverage_publication(query, inventory, validated)
        self.assertEqual(inventory, inv_snapshot)
        self.assertEqual(query, query_snapshot)

        # 4. audit_coverage with spy
        spy = _SpyCoverageVerifier(payload)
        audit_res = audit_coverage(query, inventory, verifier=spy)
        self.assertEqual(inventory, inv_snapshot)
        self.assertEqual(query, query_snapshot)

    def test_coverage_budget_guards_and_limits(self):
        """Coverage message building enforces byte budget < MAX_MESSAGES_BYTES_VERIFICATION."""
        inventory = [
            {
                "claim_id": "C1",
                "text": "Texto",
                "evidence": [{"source_id": "S1", "quote": "Cita"}],
                "publication_claim": {"text": "Texto", "evidence": [{"source_id": "S1", "quote": "Cita"}]},
            }
        ]
        # Normal query builds messages well under budget
        msgs = build_coverage_messages(self.synthetic_query, inventory)
        serialized = json.dumps(msgs, ensure_ascii=False).encode("utf-8")
        from rag_pensiones.grounding import MAX_MESSAGES_BYTES_VERIFICATION
        self.assertLess(len(serialized), MAX_MESSAGES_BYTES_VERIFICATION)

        # Oversized query triggers budget guard
        oversized_query = "x" * (MAX_MESSAGES_BYTES_VERIFICATION + 500)
        with self.assertRaisesRegex(ValueError, "Coverage messages payload exceeds budget"):
            build_coverage_messages(oversized_query, inventory)


class _MockCorrectionGenerator:
    def __init__(
        self,
        responses: list[dict] | None = None,
        digest: str = PINNED_9B_DIGEST,
        model: str = PINNED_9B_MODEL,
        parameters: dict[str, Any] | None = None,
        record_request: bool = True,
        request_options_override: dict[str, Any] | None = None,
        calls_to_append: int = 1,
        initial_calls: list[dict] | None = None,
        mutate_prefix_on_call: bool = False,
        clear_history_on_call: bool = False,
        in_place_schema_mutation: dict | None = None,
        in_place_messages_mutation: dict | None = None,
    ):
        self.model = model
        self.digest = digest
        self.parameters = copy.deepcopy(parameters) if parameters is not None else copy.deepcopy(PINNED_9B_PARAMETERS)
        self.record_request = record_request
        self.request_options_override = copy.deepcopy(request_options_override) if request_options_override is not None else None
        self.calls_to_append = calls_to_append
        self.calls: list[dict] = copy.deepcopy(initial_calls) if initial_calls is not None else []
        self.mutate_prefix_on_call = mutate_prefix_on_call
        self.clear_history_on_call = clear_history_on_call
        self.in_place_schema_mutation = copy.deepcopy(in_place_schema_mutation) if isinstance(in_place_schema_mutation, dict) else in_place_schema_mutation
        self.in_place_messages_mutation = copy.deepcopy(in_place_messages_mutation) if isinstance(in_place_messages_mutation, dict) else in_place_messages_mutation
        self.responses = list(responses or [])
        self.idx = 0

    def get_digest(self) -> str:
        return self.digest

    def chat(self, messages: list[dict], response_schema: dict) -> dict:
        if self.mutate_prefix_on_call and self.calls:
            if callable(self.mutate_prefix_on_call):
                self.mutate_prefix_on_call(self.calls[0])
            else:
                self.calls[0]["mutated"] = True
        if self.clear_history_on_call is not None and self.clear_history_on_call is not False:
            if isinstance(self.clear_history_on_call, int) and not isinstance(self.clear_history_on_call, bool):
                self.calls = self.calls[:self.clear_history_on_call]
            else:
                self.calls.clear()
        if self.in_place_schema_mutation is not None and isinstance(response_schema, dict):
            if callable(self.in_place_schema_mutation):
                self.in_place_schema_mutation(response_schema)
            elif isinstance(self.in_place_schema_mutation, dict):
                response_schema.update(self.in_place_schema_mutation)
        if self.in_place_messages_mutation is not None and isinstance(messages, list) and messages:
            if callable(self.in_place_messages_mutation):
                self.in_place_messages_mutation(messages)
            elif isinstance(self.in_place_messages_mutation, dict):
                messages[0].update(self.in_place_messages_mutation)

        call_entry: dict[str, Any] = {
            "messages": copy.deepcopy(messages),
            "schema": copy.deepcopy(response_schema),
        }
        if self.record_request:
            opts = copy.deepcopy(self.request_options_override) if self.request_options_override is not None else copy.deepcopy(self.parameters)
            exact_request = {
                "model": self.model,
                "messages": copy.deepcopy(messages),
                "stream": False,
                "think": False,
                "keep_alive": "5m",
                "format": copy.deepcopy(response_schema),
                "options": opts,
            }
            call_entry["request"] = exact_request
            call_entry["effective_options"] = opts

        for _ in range(self.calls_to_append):
            self.calls.append(copy.deepcopy(call_entry))

        if self.idx < len(self.responses):
            resp = self.responses[self.idx]
            self.idx += 1
            if isinstance(resp, Exception):
                raise resp
            if self.calls_to_append > 0 and self.calls:
                self.calls[-1]["raw_response"] = resp
            return resp
        raise RuntimeError("No more mocked generator responses")


class _MockCorrectionVerifier:
    def __init__(
        self,
        responses: list[dict] | None = None,
        digest: str = PINNED_27B_DIGEST,
        model: str = PINNED_27B_MODEL,
        parameters: dict[str, Any] | None = None,
        record_request: bool = True,
        stage_options_overrides: dict[str, dict[str, Any]] | None = None,
        stage_request_overrides: dict[str, dict[str, Any]] | None = None,
        calls_to_append: int = 1,
        stage_calls_to_append: dict[str, int] | None = None,
        initial_calls: list[dict] | None = None,
        mutate_prefix_on_stage: set[str] | list[str] | None = None,
        clear_history_on_stage: set[str] | list[str] | None = None,
        in_place_schema_mutation_by_stage: dict[str, dict] | None = None,
        in_place_messages_mutation_by_stage: dict[str, dict] | None = None,
    ):
        self.model = model
        self.expected_digest = digest
        self.parameters = copy.deepcopy(parameters) if parameters is not None else copy.deepcopy(PINNED_27B_PARAMETERS)
        self.record_request = record_request
        self.stage_options_overrides = copy.deepcopy(stage_options_overrides) if stage_options_overrides is not None else None
        self.stage_request_overrides = copy.deepcopy(stage_request_overrides) if stage_request_overrides is not None else None
        self.calls_to_append = calls_to_append
        self.stage_calls_to_append = copy.deepcopy(stage_calls_to_append) if stage_calls_to_append is not None else {}
        self.calls: list[dict] = copy.deepcopy(initial_calls) if initial_calls is not None else []
        self.mutate_prefix_on_stage = mutate_prefix_on_stage
        self.clear_history_on_stage = clear_history_on_stage
        self.in_place_schema_mutation_by_stage = copy.deepcopy(in_place_schema_mutation_by_stage) if isinstance(in_place_schema_mutation_by_stage, dict) else (in_place_schema_mutation_by_stage or {})
        self.in_place_messages_mutation_by_stage = copy.deepcopy(in_place_messages_mutation_by_stage) if isinstance(in_place_messages_mutation_by_stage, dict) else (in_place_messages_mutation_by_stage or {})
        self.lifecycle_records: list[dict] = []
        self.responses = list(responses or [])
        self.idx = 0

    def get_digest(self) -> str:
        return self.expected_digest

    def release_loaded(self, model: str) -> bool:
        self.lifecycle_records.append({"action": "unload", "model": model})
        return True

    def chat(self, messages: list[dict], response_schema: dict, *, stage: str) -> dict:
        if stage not in ("verification", "reverification", "query_coverage"):
            raise ValueError(f"Q3VerifierClient.chat accepts only 'verification', 'reverification', or 'query_coverage' stages, got {stage!r}")

        if self.mutate_prefix_on_stage and self.calls:
            if isinstance(self.mutate_prefix_on_stage, dict) and stage in self.mutate_prefix_on_stage:
                mut = self.mutate_prefix_on_stage[stage]
                if callable(mut):
                    mut(self.calls[0])
                else:
                    self.calls[0]["mutated"] = True
            elif isinstance(self.mutate_prefix_on_stage, (set, list, tuple)) and stage in self.mutate_prefix_on_stage:
                self.calls[0]["mutated"] = True
        if self.clear_history_on_stage:
            if isinstance(self.clear_history_on_stage, dict) and stage in self.clear_history_on_stage:
                target_len = self.clear_history_on_stage[stage]
                if isinstance(target_len, int) and not isinstance(target_len, bool):
                    self.calls = self.calls[:target_len]
                else:
                    self.calls.clear()
            elif isinstance(self.clear_history_on_stage, (set, list, tuple)) and stage in self.clear_history_on_stage:
                self.calls.clear()
        if stage in self.in_place_schema_mutation_by_stage and isinstance(response_schema, dict):
            mut = self.in_place_schema_mutation_by_stage[stage]
            if callable(mut):
                mut(response_schema)
            elif isinstance(mut, dict):
                response_schema.update(mut)
        if stage in self.in_place_messages_mutation_by_stage and isinstance(messages, list) and messages:
            mut_m = self.in_place_messages_mutation_by_stage[stage]
            if callable(mut_m):
                mut_m(messages)
            elif isinstance(mut_m, dict):
                messages[0].update(mut_m)

        call_entry: dict[str, Any] = {
            "stage": stage,
            "model": self.model,
            "messages": copy.deepcopy(messages),
            "schema": copy.deepcopy(response_schema),
        }
        if self.record_request:
            opts = (
                copy.deepcopy(self.stage_options_overrides[stage])
                if self.stage_options_overrides and stage in self.stage_options_overrides
                else copy.deepcopy(self.parameters)
            )
            exact_request = {
                "model": self.model,
                "messages": copy.deepcopy(messages),
                "stream": False,
                "think": False,
                "format": copy.deepcopy(response_schema),
                "options": opts,
            }
            if self.stage_request_overrides and stage in self.stage_request_overrides:
                exact_request.update(copy.deepcopy(self.stage_request_overrides[stage]))
            call_entry["request"] = exact_request
            call_entry["effective_options"] = exact_request.get("options")

        n_append = self.stage_calls_to_append.get(stage, self.calls_to_append)
        for _ in range(n_append):
            self.calls.append(copy.deepcopy(call_entry))

        if self.idx < len(self.responses):
            resp = self.responses[self.idx]
            self.idx += 1
            if isinstance(resp, Exception):
                raise resp
            if n_append > 0 and self.calls:
                self.calls[-1]["raw_response"] = resp
            return resp
        raise RuntimeError(f"No more mocked verifier responses for stage {stage}")


def _mock_gen_resp(payload: dict) -> dict:
    return {
        "model": PINNED_9B_MODEL,
        "done": True,
        "done_reason": "stop",
        "prompt_eval_count": 120,
        "eval_count": 30,
        "prompt_eval_duration": 10000000,
        "eval_duration": 20000000,
        "total_duration": 30000000,
        "load_duration": 1000000,
        "message": {
            "role": "assistant",
            "content": json.dumps(payload, ensure_ascii=False),
        },
    }


def _mock_verif_resp(verdicts: list[dict]) -> dict:
    return {
        "model": PINNED_27B_MODEL,
        "done": True,
        "done_reason": "stop",
        "prompt_eval_count": 220,
        "eval_count": 35,
        "prompt_eval_duration": 20000000,
        "eval_duration": 30000000,
        "total_duration": 50000000,
        "load_duration": 2000000,
        "message": {
            "role": "assistant",
            "content": json.dumps({"verdicts": verdicts}, ensure_ascii=False),
        },
    }


def _mock_audit_resp(parts: list[dict]) -> dict:
    return {
        "model": PINNED_27B_MODEL,
        "done": True,
        "done_reason": "stop",
        "prompt_eval_count": 250,
        "eval_count": 45,
        "prompt_eval_duration": 25000000,
        "eval_duration": 40000000,
        "total_duration": 65000000,
        "load_duration": 2000000,
        "message": {
            "role": "assistant",
            "content": json.dumps({"parts": parts}, ensure_ascii=False),
        },
    }


class TestCorrectCoverageOnce(unittest.TestCase):
    """Offline unit tests for correct_coverage_once standalone helper."""

    def setUp(self):
        self.page_text = (
            "La tasa de generosidad disminuye del 65,6% en 2023 al 56,7% en 2050 y 53,7% en 2070. "
            "Por lo tanto, volvería a niveles próximos a los alcanzados en 2015."
        )
        self.pages = [
            {
                "doc_id": "doc_1",
                "page_number": 1,
                "source_file": "doc1.pdf",
                "published_at": "2025-01-01",
                "text": self.page_text,
            }
        ]
        self.chunk = ChunkRecord(
            chunk_id="chk_1",
            doc_id="doc_1",
            source_file="doc1.pdf",
            published_at="2025-01-01",
            page_number=1,
            start_char=0,
            end_char=len(self.page_text),
            text=self.page_text,
            strategy="sentence",
        )
        self.expanded_hits = [SearchHit(rank=1, score=0.95, chunk=self.chunk)]
        self.selected_evidence = [
            {
                "evidence_id": "E1",
                "source_id": "S1",
                "doc_id": "doc_1",
                "page_number": 1,
                "source_file": "doc1.pdf",
                "quote": "La tasa de generosidad disminuye del 65,6% en 2023 al 56,7% en 2050 y 53,7% en 2070.",
            }
        ]
        self.initial_inventory = [
            {
                "claim_id": "C1",
                "text": "La tasa disminuye al 56,7% en 2050.",
                "evidence": [{"source_id": "S1", "quote": self.selected_evidence[0]["quote"]}],
                "publication_claim": {
                    "text": "La tasa disminuye al 56,7% en 2050.",
                    "evidence": [{"source_id": "S1", "quote": self.selected_evidence[0]["quote"]}],
                },
            }
        ]
        self.factual_report = {
            "metadata": {
                "model": PINNED_9B_MODEL,
                "digest": PINNED_9B_DIGEST,
                "generator_parameters": copy.deepcopy(PINNED_9B_PARAMETERS),
                "verifier_model": PINNED_27B_MODEL,
                "verifier_digest": PINNED_27B_DIGEST,
                "verifier_parameters": copy.deepcopy(PINNED_27B_PARAMETERS),
            },
            "outcome": {
                "status": "answered",
                "expanded_hits": self.expanded_hits,
                "publication": {
                    "claims": [
                        {
                            "text": "La tasa disminuye al 56,7% en 2050.",
                            "evidence": [{"source_id": "S1", "quote": self.selected_evidence[0]["quote"]}],
                        }
                    ]
                },
                "full_result": {
                    "status": "answered",
                    "grounding": {
                        "accepted_claims": [
                            {
                                "claim_id": "C1",
                                "text": "La tasa disminuye al 56,7% en 2050.",
                            }
                        ],
                        "selected_evidence": self.selected_evidence,
                        "repair_evidence": [],
                    },
                },
            }
        }
        self.query = "¿Cuál es la tasa en 2050 y en 2070?"
        self.initial_coverage_audit = {
            "status": "completed",
            "validated_payload": {
                "status": "partial",
                "parts": [
                    {
                        "query_span": "¿Cuál es la tasa en 2050",
                        "decision": "answered",
                        "claim_ids": ["C1"],
                        "reason_code": "covered",
                        "start_char": 0,
                        "end_char": 24,
                    },
                    {
                        "query_span": " y en 2070?",
                        "decision": "not_answered",
                        "claim_ids": [],
                        "reason_code": "missing_from_answer",
                        "start_char": 24,
                        "end_char": 35,
                    },
                ],
            },
        }

    def test_missing_datum_success(self):
        """Trigger (a): missing_from_answer generates candidate, verifies, reaudits, and applies merged inventory."""
        gen_resp = _mock_gen_resp({
            "status": "answered",
            "claims": [
                {
                    "claim_id": "C1",
                    "text": "En 2070 la tasa disminuye al 53,7%.",
                    "evidence_ids": ["E1"],
                }
            ],
            "reason": "",
        })
        verif_resp = _mock_verif_resp([
            {
                "claim_id": "C1",
                "verdict": "supported",
                "reason": "dato respaldado",
                "evidence_verdicts": [{"evidence_id": "E1", "relation": "supports"}],
            }
        ])
        audit_resp = _mock_audit_resp([
            {
                "query_span": "¿Cuál es la tasa en 2050",
                "decision": "answered",
                "claim_ids": ["C1"],
                "reason_code": "covered",
            },
            {
                "query_span": " y en 2070?",
                "decision": "answered",
                "claim_ids": ["C2"],
                "reason_code": "covered",
            },
        ])
        generator = _MockCorrectionGenerator([gen_resp])
        verifier = _MockCorrectionVerifier([verif_resp, audit_resp])

        res = correct_coverage_once(
            self.query,
            self.factual_report,
            self.initial_inventory,
            self.initial_coverage_audit,
            pages=self.pages,
            generator=generator,
            verifier=verifier,
        )

        self.assertTrue(res["attempted"], msg=f"res: {res.get('diagnostics')}")
        self.assertTrue(res["applied"], msg=f"res: {res.get('diagnostics')}")
        self.assertEqual(len(res["inventory"]), 2)
        self.assertEqual(res["inventory"][0]["claim_id"], "C1")
        self.assertEqual(res["inventory"][1]["claim_id"], "C2")
        self.assertEqual(res["inventory"][1]["origin"], "coverage_correction")
        self.assertEqual(res["inventory"][1]["text"], "En 2070 la tasa disminuye al 53,7%.")
        self.assertEqual(res["coverage_audit"]["validated_payload"]["status"], "answered")

        # Added call bound: exactly 1 gen, 1 verif, 1 reaudit
        self.assertEqual(len(generator.calls), 1)
        self.assertEqual(len(verifier.calls), 2)
        self.assertEqual(verifier.calls[0]["stage"], "verification")
        self.assertEqual(verifier.calls[1]["stage"], "query_coverage")

        # Feedback in prompt includes missing query span and dynamic prior claim texts in user message
        gen_user_content = generator.calls[0]["messages"][1]["content"]
        self.assertIn("« y en 2070?» (motivo: missing_from_answer)", gen_user_content)
        self.assertIn("La tasa disminuye al 56,7% en 2050.", gen_user_content)
        self.assertIn("DATOS DE DIAGNÓSTICO, NO SON EVIDENCIA NUEVA", gen_user_content)

        # Interval comparison
        self.assertTrue(res["diagnostics"]["interval_comparison"]["retains_all"])
        self.assertTrue(res["diagnostics"]["interval_comparison"]["gained_unserved"])

    def test_over_abstention_success(self):
        """Trigger (b): empty inventory with factual insufficiency and non-empty selected evidence recovers supported part."""
        query = "¿Cuál es la tasa de generosidad en 2050?"
        factual_report = copy.deepcopy(self.factual_report)
        factual_report["outcome"]["status"] = "insufficient_evidence"
        factual_report["outcome"]["full_result"]["status"] = "insufficient_evidence"
        factual_report["outcome"]["full_result"]["grounding"]["accepted_claims"] = []
        factual_report["outcome"]["publication"] = {"claims": []}

        initial_inventory = []
        initial_coverage_audit = {
            "status": "completed",
            "validated_payload": {
                "status": "insufficient_evidence",
                "parts": [
                    {
                        "query_span": query,
                        "decision": "not_answered",
                        "claim_ids": [],
                        "reason_code": "not_in_evaluated_evidence",
                        "start_char": 0,
                        "end_char": len(query),
                    }
                ],
            },
        }

        gen_resp = _mock_gen_resp({
            "status": "answered",
            "claims": [
                {
                    "claim_id": "C1",
                    "text": "La tasa disminuye al 56,7% en 2050.",
                    "evidence_ids": ["E1"],
                }
            ],
            "reason": "",
        })
        verif_resp = _mock_verif_resp([
            {
                "claim_id": "C1",
                "verdict": "supported",
                "reason": "dato respaldado",
                "evidence_verdicts": [{"evidence_id": "E1", "relation": "supports"}],
            }
        ])
        audit_resp = _mock_audit_resp([
            {
                "query_span": query,
                "decision": "answered",
                "claim_ids": ["C1"],
                "reason_code": "covered",
            }
        ])
        generator = _MockCorrectionGenerator([gen_resp])
        verifier = _MockCorrectionVerifier([verif_resp, audit_resp])

        res = correct_coverage_once(
            query,
            factual_report,
            initial_inventory,
            initial_coverage_audit,
            pages=self.pages,
            generator=generator,
            verifier=verifier,
        )

        self.assertTrue(res["attempted"], msg=f"res: {res.get('diagnostics')}")
        self.assertTrue(res["applied"], msg=f"res: {res.get('diagnostics')}")
        self.assertEqual(len(res["inventory"]), 1)
        self.assertEqual(res["inventory"][0]["claim_id"], "C1")
        self.assertEqual(res["inventory"][0]["origin"], "coverage_correction")

        # Feedback in prompt does not guarantee answer
        gen_sys = generator.calls[0]["messages"][0]["content"]
        self.assertIn("Retroalimentación de sobreabstención", gen_sys)
        self.assertIn("no garantiza que la consulta pueda responderse por completo", gen_sys)
        # Ensure user message in overabstention branch is identical to standard JSON payload without feedback
        gen_user = generator.calls[0]["messages"][1]["content"]
        self.assertNotIn("Retroalimentación", gen_user)

    def test_missing_from_answer_outgoing_request_includes_prior_claims_and_distinguishes_evidence(self):
        """State handoff: outgoing helper request in missing_from_answer dynamically passes prior claim texts as diagnostic data, preserving query and evidence."""
        custom_inventory = [
            {
                "claim_id": "C1",
                "text": "Afirmación previa verificada número uno sobre pensiones.",
                "evidence": [{"source_id": "S1", "quote": self.selected_evidence[0]["quote"]}],
                "publication_claim": {
                    "text": "Afirmación previa verificada número uno sobre pensiones.",
                    "evidence": [{"source_id": "S1", "quote": self.selected_evidence[0]["quote"]}],
                },
            },
            {
                "claim_id": "C2",
                "text": "Afirmación previa verificada número dos con horizonte 2050.",
                "evidence": [{"source_id": "S1", "quote": self.selected_evidence[0]["quote"]}],
                "publication_claim": {
                    "text": "Afirmación previa verificada número dos con horizonte 2050.",
                    "evidence": [{"source_id": "S1", "quote": self.selected_evidence[0]["quote"]}],
                },
            },
        ]
        factual_rep = copy.deepcopy(self.factual_report)
        factual_rep["outcome"]["publication"]["claims"] = [
            {"text": custom_inventory[0]["text"], "evidence": copy.deepcopy(custom_inventory[0]["evidence"])},
            {"text": custom_inventory[1]["text"], "evidence": copy.deepcopy(custom_inventory[1]["evidence"])},
        ]
        factual_rep["outcome"]["full_result"]["grounding"]["accepted_claims"] = [
            {"claim_id": "C1", "text": custom_inventory[0]["text"]},
            {"claim_id": "C2", "text": custom_inventory[1]["text"]},
        ]
        coverage_audit_custom = {
            "status": "completed",
            "validated_payload": {
                "status": "partial",
                "parts": [
                    {
                        "query_span": "¿Cuál es la tasa en 2050",
                        "decision": "answered",
                        "claim_ids": ["C1", "C2"],
                        "reason_code": "covered",
                        "start_char": 0,
                        "end_char": 24,
                    },
                    {
                        "query_span": " y en 2070?",
                        "decision": "not_answered",
                        "claim_ids": [],
                        "reason_code": "missing_from_answer",
                        "start_char": 24,
                        "end_char": 35,
                    },
                ],
            },
        }
        factual_rep_snapshot = copy.deepcopy(factual_rep)
        inventory_snapshot = copy.deepcopy(custom_inventory)

        gen_resp = _mock_gen_resp({
            "status": "answered",
            "claims": [{"claim_id": "C1", "text": "En 2070 la tasa disminuye al 53,7%.", "evidence_ids": ["E1"]}],
            "reason": "",
        })
        verif_resp = _mock_verif_resp([
            {"claim_id": "C1", "verdict": "supported", "reason": "ok", "evidence_verdicts": [{"evidence_id": "E1", "relation": "supports"}]}
        ])
        audit_resp = _mock_audit_resp([
            {"query_span": "¿Cuál es la tasa en 2050", "decision": "answered", "claim_ids": ["C1", "C2"], "reason_code": "covered"},
            {"query_span": " y en 2070?", "decision": "answered", "claim_ids": ["C3"], "reason_code": "covered"},
        ])
        generator = _MockCorrectionGenerator([gen_resp])
        verifier = _MockCorrectionVerifier([verif_resp, audit_resp])

        res = correct_coverage_once(
            self.query,
            factual_rep,
            custom_inventory,
            coverage_audit_custom,
            pages=self.pages,
            generator=generator,
            verifier=verifier,
        )

        self.assertTrue(res["attempted"])
        self.assertTrue(res["applied"])
        self.assertEqual(len(generator.calls), 1)

        req = generator.calls[0]["request"]
        messages = req["messages"]
        sys_msg = messages[0]["content"]
        user_msg = messages[1]["content"]

        # System message retains binding rules without prompt pollution
        self.assertIn("Conserva el paquete completo de referencias", sys_msg)
        self.assertNotIn("Afirmación previa verificada", sys_msg)

        # User message contains verbatim query, catalog evidence, and clearly delimited prior claims
        self.assertIn(self.query, user_msg)
        self.assertIn("catalogo_evidencias", user_msg)
        self.assertIn("E1", user_msg)
        self.assertIn("« y en 2070?» (motivo: missing_from_answer)", user_msg)
        self.assertIn("Afirmación previa verificada número uno sobre pensiones.", user_msg)
        self.assertIn("Afirmación previa verificada número dos con horizonte 2050.", user_msg)
        self.assertIn("DATOS DE DIAGNÓSTICO, NO SON EVIDENCIA NUEVA", user_msg)
        self.assertIn("repetirlos o parafrasearlos es insuficiente", user_msg)

        # Inputs remain immutable
        self.assertEqual(factual_rep, factual_rep_snapshot)
        self.assertEqual(custom_inventory, inventory_snapshot)

    def test_oversized_prior_claim_feedback_fails_closed_with_zero_calls(self):
        """Existing 6500 UTF-8 budget guard: oversized prior-text context fails closed before contacting model with zero new calls."""
        oversized_text = "A" * 6000
        oversized_inventory = [
            {
                "claim_id": "C1",
                "text": oversized_text,
                "evidence": [{"source_id": "S1", "quote": self.selected_evidence[0]["quote"]}],
                "publication_claim": {
                    "text": oversized_text,
                    "evidence": [{"source_id": "S1", "quote": self.selected_evidence[0]["quote"]}],
                },
            }
        ]
        factual_rep = copy.deepcopy(self.factual_report)
        factual_rep["outcome"]["publication"]["claims"] = [
            {"text": oversized_text, "evidence": [{"source_id": "S1", "quote": self.selected_evidence[0]["quote"]}]}
        ]
        factual_rep["outcome"]["full_result"]["grounding"]["accepted_claims"] = [
            {"claim_id": "C1", "text": oversized_text}
        ]
        generator = _MockCorrectionGenerator([])
        verifier = _MockCorrectionVerifier([])

        res = correct_coverage_once(
            self.query,
            factual_rep,
            oversized_inventory,
            self.initial_coverage_audit,
            pages=self.pages,
            generator=generator,
            verifier=verifier,
        )

        self.assertTrue(res["attempted"])
        self.assertFalse(res["applied"])
        self.assertEqual(len(generator.calls), 0)  # Zero new calls made!
        self.assertEqual(len(verifier.calls), 0)
        self.assertEqual(res["inventory"], oversized_inventory)  # Exact fallback
        self.assertEqual(res["coverage_audit"], self.initial_coverage_audit)
        self.assertIn("exceed budget of 6500 bytes", res["diagnostics"]["reason"])

    def test_correct_mixed_partial(self):
        """Mixed query: Part 1 answered, Part 2 missing_from_answer, Part 3 unavailable."""
        query = "¿Cuál es la tasa en 2050, en 2070 y en 2090?"
        initial_coverage_audit = {
            "status": "completed",
            "validated_payload": {
                "status": "partial",
                "parts": [
                    {
                        "query_span": "¿Cuál es la tasa en 2050",
                        "decision": "answered",
                        "claim_ids": ["C1"],
                        "reason_code": "covered",
                        "start_char": 0,
                        "end_char": 24,
                    },
                    {
                        "query_span": ", en 2070",
                        "decision": "not_answered",
                        "claim_ids": [],
                        "reason_code": "missing_from_answer",
                        "start_char": 24,
                        "end_char": 33,
                    },
                    {
                        "query_span": " y en 2090?",
                        "decision": "not_answered",
                        "claim_ids": [],
                        "reason_code": "not_in_evaluated_evidence",
                        "start_char": 33,
                        "end_char": 44,
                    },
                ],
            },
        }

        gen_resp = _mock_gen_resp({
            "status": "answered",
            "claims": [
                {
                    "claim_id": "C1",
                    "text": "En 2070 disminuye al 53,7%.",
                    "evidence_ids": ["E1"],
                }
            ],
            "reason": "",
        })
        verif_resp = _mock_verif_resp([
            {
                "claim_id": "C1",
                "verdict": "supported",
                "reason": "dato respaldado",
                "evidence_verdicts": [{"evidence_id": "E1", "relation": "supports"}],
            }
        ])
        audit_resp = _mock_audit_resp([
            {
                "query_span": "¿Cuál es la tasa en 2050",
                "decision": "answered",
                "claim_ids": ["C1"],
                "reason_code": "covered",
            },
            {
                "query_span": ", en 2070",
                "decision": "answered",
                "claim_ids": ["C2"],
                "reason_code": "covered",
            },
            {
                "query_span": " y en 2090?",
                "decision": "not_answered",
                "claim_ids": [],
                "reason_code": "not_in_evaluated_evidence",
            },
        ])
        generator = _MockCorrectionGenerator([gen_resp])
        verifier = _MockCorrectionVerifier([verif_resp, audit_resp])

        res = correct_coverage_once(
            query,
            self.factual_report,
            self.initial_inventory,
            initial_coverage_audit,
            pages=self.pages,
            generator=generator,
            verifier=verifier,
        )

        self.assertTrue(res["attempted"], msg=f"res: {res.get('diagnostics')}")
        self.assertTrue(res["applied"], msg=f"res: {res.get('diagnostics')}")
        self.assertEqual(len(res["inventory"]), 2)
        self.assertEqual(res["coverage_audit"]["validated_payload"]["status"], "partial")

    def test_genuine_unavailable_part_not_applied(self):
        """When not_answered parts are solely unavailable in evaluated evidence (refused outside), zero calls are made."""
        audit_refused_outside = {
            "status": "completed",
            "validated_payload": {
                "status": "partial",
                "parts": [
                    {
                        "query_span": "¿Cuál es la tasa en 2050",
                        "decision": "answered",
                        "claim_ids": ["C1"],
                        "reason_code": "covered",
                        "start_char": 0,
                        "end_char": 24,
                    },
                    {
                        "query_span": " y en 2070?",
                        "decision": "not_answered",
                        "claim_ids": [],
                        "reason_code": "not_in_evaluated_evidence",
                        "start_char": 24,
                        "end_char": 35,
                    },
                ],
            },
        }
        generator = _MockCorrectionGenerator([])
        verifier = _MockCorrectionVerifier([])

        res = correct_coverage_once(
            self.query,
            self.factual_report,
            self.initial_inventory,
            audit_refused_outside,
            pages=self.pages,
            generator=generator,
            verifier=verifier,
        )

        self.assertFalse(res["attempted"])
        self.assertFalse(res["applied"])
        self.assertEqual(len(generator.calls), 0)
        self.assertEqual(len(verifier.calls), 0)

    def test_complete_path_zero_added_calls(self):
        """When coverage audit is already complete (all answered), zero calls are made."""
        audit_all_answered = {
            "status": "completed",
            "validated_payload": {
                "status": "answered",
                "parts": [
                    {
                        "query_span": self.query,
                        "decision": "answered",
                        "claim_ids": ["C1"],
                        "reason_code": "covered",
                        "start_char": 0,
                        "end_char": len(self.query),
                    }
                ],
            },
        }
        generator = _MockCorrectionGenerator([])
        verifier = _MockCorrectionVerifier([])

        res = correct_coverage_once(
            self.query,
            self.factual_report,
            self.initial_inventory,
            audit_all_answered,
            pages=self.pages,
            generator=generator,
            verifier=verifier,
        )

        self.assertFalse(res["attempted"])
        self.assertFalse(res["applied"])
        self.assertEqual(len(generator.calls), 0)
        self.assertEqual(len(verifier.calls), 0)

    def test_empty_selected_no_calls(self):
        """When selected_evidence is empty, zero calls are made even if inventory is empty."""
        rep = copy.deepcopy(self.factual_report)
        rep["outcome"]["status"] = "insufficient_evidence"
        rep["outcome"]["full_result"]["grounding"]["accepted_claims"] = []
        rep["outcome"]["full_result"]["grounding"]["selected_evidence"] = []
        rep["outcome"]["publication"]["claims"] = []
        audit = {
            "status": "completed",
            "validated_payload": {
                "status": "insufficient_evidence",
                "parts": [
                    {
                        "query_span": self.query,
                        "decision": "not_answered",
                        "claim_ids": [],
                        "reason_code": "not_in_evaluated_evidence",
                        "start_char": 0,
                        "end_char": len(self.query),
                    }
                ],
            },
        }

        generator = _MockCorrectionGenerator([])
        verifier = _MockCorrectionVerifier([])

        res = correct_coverage_once(
            self.query,
            rep,
            [],
            audit,
            pages=self.pages,
            generator=generator,
            verifier=verifier,
        )

        self.assertFalse(res["attempted"])
        self.assertFalse(res["applied"])
        self.assertEqual(len(generator.calls), 0)
        self.assertEqual(len(verifier.calls), 0)

    def test_unsupported_candidate_not_applied(self):
        """When candidate claims are unsupported by verifier, reaudit is NOT called and applied is False."""
        gen_resp = _mock_gen_resp({
            "status": "answered",
            "claims": [
                {
                    "claim_id": "C1",
                    "text": "Afirmación no respaldada.",
                    "evidence_ids": ["E1"],
                }
            ],
            "reason": "",
        })
        verif_resp = _mock_verif_resp([
            {
                "claim_id": "C1",
                "verdict": "insufficient_evidence",
                "reason": "falta apoyo",
                "evidence_verdicts": [{"evidence_id": "E1", "relation": "irrelevant"}],
            }
        ])
        generator = _MockCorrectionGenerator([gen_resp])
        verifier = _MockCorrectionVerifier([verif_resp])

        res = correct_coverage_once(
            self.query,
            self.factual_report,
            self.initial_inventory,
            self.initial_coverage_audit,
            pages=self.pages,
            generator=generator,
            verifier=verifier,
        )

        self.assertTrue(res["attempted"])
        self.assertFalse(res["applied"])
        self.assertEqual(res["inventory"], self.initial_inventory)
        self.assertEqual(len(generator.calls), 1)
        self.assertEqual(len(verifier.calls), 1)  # Only verification; reaudit skipped!

    def test_contradicted_candidate_fails_closed(self):
        """When candidate claim is contradicted, fails closed and reaudit is skipped."""
        gen_resp = _mock_gen_resp({
            "status": "answered",
            "claims": [
                {
                    "claim_id": "C1",
                    "text": "La tasa aumenta en 2070.",
                    "evidence_ids": ["E1"],
                }
            ],
            "reason": "",
        })
        verif_resp = _mock_verif_resp([
            {
                "claim_id": "C1",
                "verdict": "contradicted",
                "reason": "contradice datos",
                "evidence_verdicts": [{"evidence_id": "E1", "relation": "contradicts"}],
            }
        ])
        generator = _MockCorrectionGenerator([gen_resp])
        verifier = _MockCorrectionVerifier([verif_resp])

        res = correct_coverage_once(
            self.query,
            self.factual_report,
            self.initial_inventory,
            self.initial_coverage_audit,
            pages=self.pages,
            generator=generator,
            verifier=verifier,
        )

        self.assertTrue(res["attempted"])
        self.assertFalse(res["applied"])
        self.assertEqual(res["inventory"], self.initial_inventory)
        self.assertEqual(len(verifier.calls), 1)
        self.assertIn("contradicted", res["diagnostics"]["reason"])

    def test_claim_id_collision_handling(self):
        """When inventory has C1 and C2, candidate claim gets assigned C3 without collision."""
        inventory = [
            {"claim_id": "C1", "text": "Texto 1", "evidence": [], "publication_claim": {"text": "Texto 1", "evidence": []}},
            {"claim_id": "C2", "text": "Texto 2", "evidence": [], "publication_claim": {"text": "Texto 2", "evidence": []}},
        ]
        rep = copy.deepcopy(self.factual_report)
        rep["outcome"]["full_result"]["grounding"]["accepted_claims"] = [
            {"claim_id": "C1", "text": "Texto 1"},
            {"claim_id": "C2", "text": "Texto 2"},
        ]
        rep["outcome"]["publication"]["claims"] = [
            {"text": "Texto 1", "evidence": []},
            {"text": "Texto 2", "evidence": []},
        ]
        audit = {
            "status": "completed",
            "validated_payload": {
                "status": "partial",
                "parts": [
                    {"query_span": "¿Cuál es la tasa en 2050", "decision": "answered", "claim_ids": ["C1"], "reason_code": "covered", "start_char": 0, "end_char": 24},
                    {"query_span": " y en 2070?", "decision": "not_answered", "claim_ids": [], "reason_code": "missing_from_answer", "start_char": 24, "end_char": 35},
                ],
            },
        }
        gen_resp = _mock_gen_resp({
            "status": "answered",
            "claims": [{"claim_id": "C1", "text": "Texto 3.", "evidence_ids": ["E1"]}],
            "reason": "",
        })
        verif_resp = _mock_verif_resp([
            {"claim_id": "C1", "verdict": "supported", "reason": "ok", "evidence_verdicts": [{"evidence_id": "E1", "relation": "supports"}]}
        ])
        audit_resp = _mock_audit_resp([
            {"query_span": "¿Cuál es la tasa en 2050", "decision": "answered", "claim_ids": ["C1"], "reason_code": "covered"},
            {"query_span": " y en 2070?", "decision": "answered", "claim_ids": ["C3"], "reason_code": "covered"},
        ])
        generator = _MockCorrectionGenerator([gen_resp])
        verifier = _MockCorrectionVerifier([verif_resp, audit_resp])

        res = correct_coverage_once(
            self.query,
            rep,
            inventory,
            audit,
            pages=self.pages,
            generator=generator,
            verifier=verifier,
        )

        self.assertTrue(res["applied"])
        self.assertEqual(len(res["inventory"]), 3)
        self.assertEqual(res["inventory"][2]["claim_id"], "C3")

    def test_two_stage_e_ids_isolation(self):
        """Repair evidence IDs (e.g. E2) must not be merged into candidate generation binding."""
        factual_rep = copy.deepcopy(self.factual_report)
        factual_rep["outcome"]["full_result"]["grounding"]["repair_evidence"] = [
            {"evidence_id": "E2", "source_id": "S1", "quote": "Cita de reparación."}
        ]

        gen_resp = _mock_gen_resp({
            "status": "answered",
            "claims": [{"claim_id": "C1", "text": "Texto.", "evidence_ids": ["E1"]}],
            "reason": "",
        })
        verif_resp = _mock_verif_resp([
            {"claim_id": "C1", "verdict": "supported", "reason": "ok", "evidence_verdicts": [{"evidence_id": "E1", "relation": "supports"}]}
        ])
        audit_resp = _mock_audit_resp([
            {"query_span": "¿Cuál es la tasa en 2050", "decision": "answered", "claim_ids": ["C1"], "reason_code": "covered"},
            {"query_span": " y en 2070?", "decision": "answered", "claim_ids": ["C2"], "reason_code": "covered"},
        ])
        generator = _MockCorrectionGenerator([gen_resp])
        verifier = _MockCorrectionVerifier([verif_resp, audit_resp])

        res = correct_coverage_once(
            self.query,
            factual_rep,
            self.initial_inventory,
            self.initial_coverage_audit,
            pages=self.pages,
            generator=generator,
            verifier=verifier,
        )

        self.assertTrue(res["applied"])
        # Generation request only binds E1
        gen_bound_refs = res["diagnostics"]["generation"]["bound_refs"]
        self.assertEqual(gen_bound_refs, ["E1"])

    def test_invalid_json_nonfinite_unknown_keys_oversize(self):
        """Generator output with invalid JSON, nonfinite numbers, unknown keys, or oversize fails closed."""
        # 1. Invalid JSON preserves raw_response, request, and error
        bad_json_resp = {
            "model": DEFAULT_GENERATION_MODEL,
            "done": True,
            "done_reason": "stop",
            "prompt_eval_count": 10,
            "eval_count": 10,
            "prompt_eval_duration": 10,
            "eval_duration": 10,
            "total_duration": 20,
            "message": {"role": "assistant", "content": "not json at all {"},
        }
        gen = _MockCorrectionGenerator([bad_json_resp])
        verif = _MockCorrectionVerifier([])
        res = correct_coverage_once(
            self.query, self.factual_report, self.initial_inventory, self.initial_coverage_audit,
            pages=self.pages, generator=gen, verifier=verif
        )
        self.assertTrue(res["attempted"])
        self.assertFalse(res["applied"])
        g_trace = res["diagnostics"]["generation"]
        self.assertIsNotNone(g_trace)
        self.assertEqual(g_trace["raw_response"], bad_json_resp)
        self.assertIsNotNone(g_trace["request"])
        self.assertIn("JSON", str(g_trace["error"]))
        self.assertEqual(res["inventory"], self.initial_inventory)

        # 2. Unknown keys in payload
        unknown_keys_resp = _mock_gen_resp({
            "status": "answered",
            "claims": [{"claim_id": "C1", "text": "T", "evidence_ids": ["E1"]}],
            "reason": "",
            "extra_key": "forbidden",
        })
        gen2 = _MockCorrectionGenerator([unknown_keys_resp])
        res2 = correct_coverage_once(
            self.query, self.factual_report, self.initial_inventory, self.initial_coverage_audit,
            pages=self.pages, generator=gen2, verifier=verif
        )
        self.assertTrue(res2["attempted"])
        self.assertFalse(res2["applied"])
        self.assertIsNotNone(res2["diagnostics"]["generation"]["raw_response"])

        # 3. Nonfinite stats (e.g. prompt_eval_count = nan)
        bad_stats_resp = {
            "model": DEFAULT_GENERATION_MODEL,
            "done": True,
            "done_reason": "stop",
            "prompt_eval_count": float("nan"),
            "eval_count": 10,
            "prompt_eval_duration": 10,
            "eval_duration": 10,
            "total_duration": 20,
            "message": {"role": "assistant", "content": json.dumps({"status": "insufficient_evidence", "claims": [], "reason": ""})},
        }
        gen3 = _MockCorrectionGenerator([bad_stats_resp])
        res3 = correct_coverage_once(
            self.query, self.factual_report, self.initial_inventory, self.initial_coverage_audit,
            pages=self.pages, generator=gen3, verifier=verif
        )
        self.assertTrue(res3["attempted"])
        self.assertFalse(res3["applied"])
        self.assertIsNotNone(res3["diagnostics"]["generation"])
        self.assertEqual(res3["diagnostics"]["generation"]["raw_response"]["prompt_eval_count"], "NaN")
        self.assertTrue(math.isnan(bad_stats_resp["prompt_eval_count"]))
        # Strict serialization without allow_nan must succeed
        json.dumps(res3, allow_nan=False)

    def test_foreign_nonliteral_offsets_digest_mismatch(self):
        """Selected evidence quote not matching canonical page or digest change fails closed."""
        # 1. Quote mismatch
        bad_rep = copy.deepcopy(self.factual_report)
        bad_rep["outcome"]["full_result"]["grounding"]["selected_evidence"][0]["quote"] = (
            "Esta cita no existe en la página bajo ningún concepto."
        )
        gen = _MockCorrectionGenerator([])
        verif = _MockCorrectionVerifier([])
        res = correct_coverage_once(
            self.query, bad_rep, self.initial_inventory, self.initial_coverage_audit,
            pages=self.pages, generator=gen, verifier=verif
        )
        self.assertTrue(res["attempted"])
        self.assertFalse(res["applied"])
        self.assertIn("quote not found in source S1 text", res["diagnostics"]["reason"])

        # 2. Generator digest mismatch
        class _DigestMismatchGenerator(_MockCorrectionGenerator):
            def __init__(self):
                super().__init__()

            def chat(self, messages, response_schema):
                self.calls.append({"messages": copy.deepcopy(messages), "schema": copy.deepcopy(response_schema)})
                self.digest = "b" * 64
                return _mock_gen_resp({
                    "status": "answered",
                    "claims": [{"claim_id": "C1", "text": "T", "evidence_ids": ["E1"]}],
                    "reason": "",
                })

        gen = _DigestMismatchGenerator()
        res = correct_coverage_once(
            self.query, self.factual_report, self.initial_inventory, self.initial_coverage_audit,
            pages=self.pages, generator=gen, verifier=verif
        )
        self.assertFalse(res["applied"])
        self.assertIn("digest changed", res["diagnostics"]["reason"])

    def test_no_stage_mislabel_count_loss(self):
        """Verifier stages must strictly be 'verification' and 'query_coverage'; call count captured."""
        gen_resp = _mock_gen_resp({
            "status": "answered",
            "claims": [{"claim_id": "C1", "text": "Texto.", "evidence_ids": ["E1"]}],
            "reason": "",
        })
        verif_resp = _mock_verif_resp([
            {"claim_id": "C1", "verdict": "supported", "reason": "ok", "evidence_verdicts": [{"evidence_id": "E1", "relation": "supports"}]}
        ])
        audit_resp = _mock_audit_resp([
            {"query_span": "¿Cuál es la tasa en 2050", "decision": "answered", "claim_ids": ["C1"], "reason_code": "covered"},
            {"query_span": " y en 2070?", "decision": "answered", "claim_ids": ["C2"], "reason_code": "covered"},
        ])
        generator = _MockCorrectionGenerator([gen_resp])
        verifier = _MockCorrectionVerifier([verif_resp, audit_resp])

        res = correct_coverage_once(
            self.query, self.factual_report, self.initial_inventory, self.initial_coverage_audit,
            pages=self.pages, generator=generator, verifier=verifier
        )

        self.assertTrue(res["applied"])
        self.assertEqual(len(verifier.calls), 2)
        self.assertEqual(verifier.calls[0]["stage"], "verification")
        self.assertEqual(verifier.calls[1]["stage"], "query_coverage")
        self.assertEqual(res["diagnostics"]["generation"]["origin"], "coverage_correction")
        self.assertEqual(res["diagnostics"]["verification"]["origin"], "coverage_correction")
        self.assertEqual(res["diagnostics"]["reaudit"]["origin"], "coverage_correction")

    def test_failed_reaudit_and_changed_segmentation_preservation(self):
        """Reaudit losing prior intervals is rejected; changed segmentation preserving intervals is accepted."""
        # Case A: Reaudit drops previously answered part
        gen_resp = _mock_gen_resp({
            "status": "answered",
            "claims": [{"claim_id": "C1", "text": "Texto.", "evidence_ids": ["E1"]}],
            "reason": "",
        })
        verif_resp = _mock_verif_resp([
            {"claim_id": "C1", "verdict": "supported", "reason": "ok", "evidence_verdicts": [{"evidence_id": "E1", "relation": "supports"}]}
        ])
        audit_drops_prior = _mock_audit_resp([
            {"query_span": "¿Cuál es la tasa en 2050", "decision": "not_answered", "claim_ids": [], "reason_code": "missing_from_answer"},
            {"query_span": " y en 2070?", "decision": "answered", "claim_ids": ["C2"], "reason_code": "covered"},
        ])
        gen = _MockCorrectionGenerator([gen_resp])
        verif = _MockCorrectionVerifier([verif_resp, audit_drops_prior])

        res = correct_coverage_once(
            self.query, self.factual_report, self.initial_inventory, self.initial_coverage_audit,
            pages=self.pages, generator=gen, verifier=verif
        )
        self.assertTrue(res["attempted"])
        self.assertFalse(res["applied"])
        self.assertEqual(res["inventory"], self.initial_inventory)

        # Case B: Reaudit changes segmentation (splits part 1 in two) but retains union and gains new
        gen = _MockCorrectionGenerator([gen_resp])
        audit_changed_segmentation = _mock_audit_resp([
            {"query_span": "¿Cuál es la ", "decision": "answered", "claim_ids": ["C1"], "reason_code": "covered"},
            {"query_span": "tasa en 2050", "decision": "answered", "claim_ids": ["C1"], "reason_code": "covered"},
            {"query_span": " y en 2070?", "decision": "answered", "claim_ids": ["C2"], "reason_code": "covered"},
        ])
        verif = _MockCorrectionVerifier([verif_resp, audit_changed_segmentation])

        res2 = correct_coverage_once(
            self.query, self.factual_report, self.initial_inventory, self.initial_coverage_audit,
            pages=self.pages, generator=gen, verifier=verif
        )
        self.assertTrue(res2["attempted"])
        self.assertTrue(res2["applied"])
        self.assertEqual(len(res2["inventory"]), 2)

    def test_immutable_input_snapshots(self):
        """Inputs query, factual_report, inventory, coverage_audit, pages remain byte/object identical."""
        q_snap = copy.deepcopy(self.query)
        f_snap = copy.deepcopy(self.factual_report)
        i_snap = copy.deepcopy(self.initial_inventory)
        a_snap = copy.deepcopy(self.initial_coverage_audit)
        p_snap = copy.deepcopy(self.pages)

        gen_resp = _mock_gen_resp({
            "status": "answered",
            "claims": [{"claim_id": "C1", "text": "Texto.", "evidence_ids": ["E1"]}],
            "reason": "",
        })
        verif_resp = _mock_verif_resp([
            {"claim_id": "C1", "verdict": "supported", "reason": "ok", "evidence_verdicts": [{"evidence_id": "E1", "relation": "supports"}]}
        ])
        audit_resp = _mock_audit_resp([
            {"query_span": "¿Cuál es la tasa en 2050", "decision": "answered", "claim_ids": ["C1"], "reason_code": "covered"},
            {"query_span": " y en 2070?", "decision": "answered", "claim_ids": ["C2"], "reason_code": "covered"},
        ])
        gen = _MockCorrectionGenerator([gen_resp])
        verif = _MockCorrectionVerifier([verif_resp, audit_resp])

        correct_coverage_once(
            self.query, self.factual_report, self.initial_inventory, self.initial_coverage_audit,
            pages=self.pages, generator=gen, verifier=verif
        )

        self.assertEqual(self.query, q_snap)
        self.assertEqual(self.factual_report, f_snap)
        self.assertEqual(self.initial_inventory, i_snap)
        self.assertEqual(self.initial_coverage_audit, a_snap)
        self.assertEqual(self.pages, p_snap)

    def test_max1_correction_no_recursive_retry(self):
        """One attempt total; no recursive second attempt even if reaudit still has missing parts."""
        gen_resp = _mock_gen_resp({
            "status": "answered",
            "claims": [{"claim_id": "C1", "text": "Texto.", "evidence_ids": ["E1"]}],
            "reason": "",
        })
        verif_resp = _mock_verif_resp([
            {"claim_id": "C1", "verdict": "supported", "reason": "ok", "evidence_verdicts": [{"evidence_id": "E1", "relation": "supports"}]}
        ])
        # Reaudit still has missing_from_answer
        audit_still_missing = _mock_audit_resp([
            {"query_span": "¿Cuál es la tasa en 2050", "decision": "answered", "claim_ids": ["C1"], "reason_code": "covered"},
            {"query_span": " y en 2070?", "decision": "not_answered", "claim_ids": [], "reason_code": "missing_from_answer"},
        ])
        gen = _MockCorrectionGenerator([gen_resp])
        verif = _MockCorrectionVerifier([verif_resp, audit_still_missing])

        res = correct_coverage_once(
            self.query, self.factual_report, self.initial_inventory, self.initial_coverage_audit,
            pages=self.pages, generator=gen, verifier=verif
        )

        # Still max 1 call to generator, max 2 to verifier. No recursive loop.
        self.assertEqual(len(gen.calls), 1)
        self.assertEqual(len(verif.calls), 2)
        self.assertFalse(res["applied"])

    def test_revalidation_rejects_float_offsets_and_invented_hit_context(self):
        """Item 1: float offsets in SearchHit or raw dict are rejected even if quote appears inside text."""
        # Case A: ChunkRecord with float offsets
        bad_chunk = ChunkRecord(
            chunk_id="chk_float",
            doc_id="doc_1",
            source_file="doc1.pdf",
            published_at="2025-01-01",
            page_number=1,
            start_char=0.0,  # float!
            end_char=float(len(self.page_text)),  # float!
            text=self.page_text,
            strategy="sentence",
        )
        bad_hits = [SearchHit(rank=1, score=0.9, chunk=bad_chunk)]
        bad_rep = copy.deepcopy(self.factual_report)
        bad_rep["outcome"]["expanded_hits"] = bad_hits

        is_valid, err, _, _, _ = _revalidate_selected_evidence(bad_rep, self.pages)
        self.assertFalse(is_valid)
        self.assertIn("Invalid chunk start_char", err)

        res = correct_coverage_once(
            self.query, bad_rep, self.initial_inventory, self.initial_coverage_audit,
            pages=self.pages, generator=_MockCorrectionGenerator([]), verifier=_MockCorrectionVerifier([])
        )
        self.assertTrue(res["attempted"])
        self.assertFalse(res["applied"])
        self.assertIn("revalidation failed closed", res["diagnostics"]["reason"])

    def test_revalidation_rejects_bool_string_missing_out_of_range_offsets(self):
        """Item 3: missing coordinates (real deletion), bool, string, negative, or beyond page length fail closed."""
        cases = [
            {"start_char": True, "end_char": 20},
            {"start_char": 0, "end_char": False},
            {"start_char": -5, "end_char": 20},
            {"start_char": 10, "end_char": 5},
            {"start_char": 0, "end_char": len(self.page_text) + 100},
            {"start_char": "0", "end_char": 20},
            {"start_char": 0, "end_char": "20"},
            {"delete_start": True},
            {"delete_end": True},
            {"delete_both": True},
            {"delete_doc_id": True},
            {"delete_source_file": True},
            {"delete_page_number": True},
        ]
        for c in cases:
            bad_cdata = {
                "chunk_id": "chk_bad",
                "doc_id": "doc_1",
                "source_file": "doc1.pdf",
                "page_number": 1,
                "start_char": 0,
                "end_char": 20,
                "text": self.page_text,
            }
            if "start_char" in c:
                bad_cdata["start_char"] = c["start_char"]
            if "end_char" in c:
                bad_cdata["end_char"] = c["end_char"]
            if c.get("delete_start"):
                del bad_cdata["start_char"]
            if c.get("delete_end"):
                del bad_cdata["end_char"]
            if c.get("delete_both"):
                del bad_cdata["start_char"]
                del bad_cdata["end_char"]
            if c.get("delete_doc_id"):
                del bad_cdata["doc_id"]
            if c.get("delete_source_file"):
                del bad_cdata["source_file"]
            if c.get("delete_page_number"):
                del bad_cdata["page_number"]

            bad_rep = copy.deepcopy(self.factual_report)
            bad_rep["outcome"]["expanded_hits"] = [{"rank": 1, "score": 0.9, "chunk": bad_cdata}]
            is_valid, err, _, _, _ = _revalidate_selected_evidence(bad_rep, self.pages)
            self.assertFalse(is_valid, msg=f"Case {c} should have failed validation")
            self.assertIsNotNone(err)

            # Assert correct_coverage_once makes zero calls
            gen = _MockCorrectionGenerator([])
            verif = _MockCorrectionVerifier([])
            res = correct_coverage_once(
                self.query, bad_rep, self.initial_inventory, self.initial_coverage_audit,
                pages=self.pages, generator=gen, verifier=verif
            )
            self.assertTrue(res["attempted"])
            self.assertFalse(res["applied"])
            self.assertEqual(len(gen.calls), 0)
            self.assertEqual(len(verif.calls), 0)
            self.assertIn("revalidation failed closed", res["diagnostics"]["reason"])

    def test_revalidation_rejects_duplicate_source_identities_or_page_keys(self):
        """Item 1: duplicate ranks/source IDs or duplicate canonical page keys fail closed."""
        # 1. Duplicate source identities (two hits with rank=1)
        bad_rep = copy.deepcopy(self.factual_report)
        bad_rep["outcome"]["expanded_hits"] = [
            SearchHit(rank=1, score=0.95, chunk=self.chunk),
            SearchHit(rank=1, score=0.85, chunk=self.chunk),
        ]
        is_valid, err, _, _, _ = _revalidate_selected_evidence(bad_rep, self.pages)
        self.assertFalse(is_valid)
        self.assertIn("Duplicate source identity or rank: S1", err)

        # 2. Duplicate canonical page keys in pages
        dup_pages = [self.pages[0], copy.deepcopy(self.pages[0])]
        is_valid2, err2, _, _, _ = _revalidate_selected_evidence(self.factual_report, dup_pages)
        self.assertFalse(is_valid2)
        self.assertIn("Duplicate canonical page key", err2)

    def test_revalidation_allows_valid_multichunk_samepage(self):
        """Item 1: distinct chunks on same page with unique ranks are accepted without collision."""
        chunk1 = ChunkRecord(
            chunk_id="chk_1",
            doc_id="doc_1",
            source_file="doc1.pdf",
            published_at="2025-01-01",
            page_number=1,
            start_char=0,
            end_char=89,
            text=self.page_text[:89],
            strategy="sentence",
        )
        chunk2 = ChunkRecord(
            chunk_id="chk_2",
            doc_id="doc_1",
            source_file="doc1.pdf",
            published_at="2025-01-01",
            page_number=1,
            start_char=90,
            end_char=len(self.page_text),
            text=self.page_text[90:],
            strategy="sentence",
        )
        multi_rep = copy.deepcopy(self.factual_report)
        multi_rep["outcome"]["expanded_hits"] = [
            SearchHit(rank=1, score=0.9, chunk=chunk1),
            SearchHit(rank=2, score=0.8, chunk=chunk2),
        ]
        multi_rep["outcome"]["full_result"]["grounding"]["selected_evidence"] = [
            {
                "evidence_id": "E1",
                "source_id": "S1",
                "doc_id": "doc_1",
                "page_number": 1,
                "source_file": "doc1.pdf",
                "quote": self.page_text[:50],
            },
            {
                "evidence_id": "E2",
                "source_id": "S2",
                "doc_id": "doc_1",
                "page_number": 1,
                "source_file": "doc1.pdf",
                "quote": self.page_text[90:130],
            },
        ]
        is_valid, err, sel, hits, mappings = _revalidate_selected_evidence(multi_rep, self.pages)
        self.assertTrue(is_valid, msg=err)
        self.assertIn("S1", hits)
        self.assertIn("S2", hits)
        self.assertEqual(len(mappings), 2)

    def test_revalidation_handles_invalid_score_conversion_gracefully(self):
        """Item 5: malformed score (string, nan, bool) returns applied=False with diagnostics, not unhandled crash."""
        bad_rep = copy.deepcopy(self.factual_report)
        bad_rep["outcome"]["expanded_hits"] = [
            {"rank": 1, "score": "not_a_number", "chunk": self.chunk}
        ]
        res = correct_coverage_once(
            self.query, bad_rep, self.initial_inventory, self.initial_coverage_audit,
            pages=self.pages, generator=_MockCorrectionGenerator([]), verifier=_MockCorrectionVerifier([])
        )
        self.assertTrue(res["attempted"])
        self.assertFalse(res["applied"])
        self.assertEqual(res["inventory"], self.initial_inventory)
        self.assertIn("Invalid hit score", res["diagnostics"]["reason"])

    def test_bind_model_options_rejects_generator_digest_mismatch_against_original(self):
        """Item 2: generator client digest different from original metadata/pinned digest fails closed."""
        gen = _MockCorrectionGenerator([], digest="c" * 64)
        verif = _MockCorrectionVerifier([])
        res = correct_coverage_once(
            self.query, self.factual_report, self.initial_inventory, self.initial_coverage_audit,
            pages=self.pages, generator=gen, verifier=verif
        )
        self.assertTrue(res["attempted"])
        self.assertFalse(res["applied"])
        self.assertIn("Generator client digest mismatch", res["diagnostics"]["reason"])

    def test_bind_model_options_rejects_altered_source_metadata_options(self):
        """Item 2: altered options or bool confusion in report metadata fails closed."""
        # 1. Missing metadata
        bad_rep = copy.deepcopy(self.factual_report)
        del bad_rep["metadata"]
        res1 = correct_coverage_once(
            self.query, bad_rep, self.initial_inventory, self.initial_coverage_audit,
            pages=self.pages, generator=_MockCorrectionGenerator([]), verifier=_MockCorrectionVerifier([])
        )
        self.assertFalse(res1["applied"])
        self.assertIn("missing valid metadata", res1["diagnostics"]["reason"])

        # 2. Bool confusion in generator parameters
        bad_rep2 = copy.deepcopy(self.factual_report)
        bad_rep2["metadata"]["generator_parameters"]["seed"] = True
        res2 = correct_coverage_once(
            self.query, bad_rep2, self.initial_inventory, self.initial_coverage_audit,
            pages=self.pages, generator=_MockCorrectionGenerator([]), verifier=_MockCorrectionVerifier([])
        )
        self.assertFalse(res2["applied"])
        self.assertIn("invalid boolean type", res2["diagnostics"]["reason"])

    def test_bind_model_options_fails_on_missing_request_log(self):
        """Item 2: client failing to provide actual outgoing request log fails closed."""
        gen = _MockCorrectionGenerator(
            responses=[_mock_gen_resp({
                "status": "answered",
                "claims": [{"claim_id": "C1", "text": "Texto.", "evidence_ids": ["E1"]}],
                "reason": "",
            })],
            record_request=False,  # Simulates legacy RecordedGenerator without request options
        )
        verif = _MockCorrectionVerifier([])
        res = correct_coverage_once(
            self.query, self.factual_report, self.initial_inventory, self.initial_coverage_audit,
            pages=self.pages, generator=gen, verifier=verif
        )
        self.assertTrue(res["attempted"])
        self.assertFalse(res["applied"])
        self.assertIn("did not provide validated actual request/options log", res["diagnostics"]["reason"])

    def test_bind_model_options_rejects_request_option_swap_after_call(self):
        """Item 2: outgoing request options differing from pinned policy fail closed after call."""
        swapped_params = copy.deepcopy(PINNED_9B_PARAMETERS)
        swapped_params["temperature"] = 0.8  # Swapped from 0
        gen = _MockCorrectionGenerator(
            responses=[_mock_gen_resp({
                "status": "answered",
                "claims": [{"claim_id": "C1", "text": "Texto.", "evidence_ids": ["E1"]}],
                "reason": "",
            })],
            parameters=swapped_params,
        )
        verif = _MockCorrectionVerifier([])
        res = correct_coverage_once(
            self.query, self.factual_report, self.initial_inventory, self.initial_coverage_audit,
            pages=self.pages, generator=gen, verifier=verif
        )
        self.assertTrue(res["attempted"])
        self.assertFalse(res["applied"])
        self.assertIn("Generator actual request options invalid", res["diagnostics"]["reason"])

    def test_bind_model_options_rejects_verifier_policy_swap(self):
        """Item 2: verifier client with altered parameters or model fails closed."""
        swapped_verif_params = copy.deepcopy(PINNED_27B_PARAMETERS)
        swapped_verif_params["temperature"] = 0.5
        verif = _MockCorrectionVerifier([], parameters=swapped_verif_params)
        res = correct_coverage_once(
            self.query, self.factual_report, self.initial_inventory, self.initial_coverage_audit,
            pages=self.pages, generator=_MockCorrectionGenerator([]), verifier=verif
        )
        self.assertTrue(res["attempted"])
        self.assertFalse(res["applied"])
        self.assertIn("Verifier client parameters invalid", res["diagnostics"]["reason"])

    def test_authentic_model_options_success(self):
        """Item 2: authentic models, digests, parameters retain model_binding in diagnostics."""
        gen_resp = _mock_gen_resp({
            "status": "answered",
            "claims": [{"claim_id": "C1", "text": "En 2070 la tasa disminuye al 53,7%.", "evidence_ids": ["E1"]}],
            "reason": "",
        })
        verif_resp = _mock_verif_resp([
            {"claim_id": "C1", "verdict": "supported", "reason": "ok", "evidence_verdicts": [{"evidence_id": "E1", "relation": "supports"}]}
        ])
        audit_resp = _mock_audit_resp([
            {"query_span": "¿Cuál es la tasa en 2050", "decision": "answered", "claim_ids": ["C1"], "reason_code": "covered"},
            {"query_span": " y en 2070?", "decision": "answered", "claim_ids": ["C2"], "reason_code": "covered"},
        ])
        gen = _MockCorrectionGenerator([gen_resp])
        verif = _MockCorrectionVerifier([verif_resp, audit_resp])

        res = correct_coverage_once(
            self.query, self.factual_report, self.initial_inventory, self.initial_coverage_audit,
            pages=self.pages, generator=gen, verifier=verif
        )
        self.assertTrue(res["applied"])
        mb = res["diagnostics"]["model_binding"]
        self.assertIsNotNone(mb)
        self.assertEqual(mb["generator"]["model_before"], PINNED_9B_MODEL)
        self.assertEqual(mb["generator"]["digest_before"], PINNED_9B_DIGEST)
        self.assertEqual(mb["generator"]["model_after"], PINNED_9B_MODEL)
        self.assertEqual(mb["generator"]["digest_after"], PINNED_9B_DIGEST)
        self.assertEqual(mb["generator"]["effective_options"]["seed"], 42)
        self.assertEqual(mb["verifier"]["model_before"], PINNED_27B_MODEL)
        self.assertEqual(mb["verifier"]["digest_before"], PINNED_27B_DIGEST)
        self.assertEqual(mb["verifier"]["model_after"], PINNED_27B_MODEL)
        self.assertEqual(mb["verifier"]["digest_after"], PINNED_27B_DIGEST)
        self.assertEqual(mb["verifier"]["effective_options"]["top_k"], 20)

    def test_authentic_fixture_positive_compatibility(self):
        """Senior Item 1: authentic live fixture feeds unmodified factual_report directly to reach applied=True."""
        fixture_path = Path(__file__).resolve().parent / "fixtures" / "corrector_authentic_n02_v3.json"
        raw_bytes = fixture_path.read_bytes()
        fixture_sha256 = hashlib.sha256(raw_bytes).hexdigest()
        self.assertEqual(
            fixture_sha256,
            "252adc3552d13bb24efc661e64bf70166db37f61efc9d0c76ce71ab65230d2a4",
            "Authentic fixture file must NEVER be modified",
        )

        fixture_data = json.loads(raw_bytes.decode("utf-8"))
        query = fixture_data["query"]
        factual_report = fixture_data["factual_report"]
        coverage_audit = fixture_data["coverage_audit"]
        pages = fixture_data["pages"]

        raw_bytes_before = copy.deepcopy(raw_bytes)
        factual_report_snapshot = copy.deepcopy(factual_report)

        # Compute inventory directly from authentic factual_report
        inventory = claim_inventory(factual_report)
        self.assertEqual(len(inventory), 1)
        self.assertEqual(inventory[0]["claim_id"], "C1")

        # Mock generator producing candidate answering the unserved span
        gen_resp = _mock_gen_resp({
            "status": "answered",
            "claims": [
                {
                    "claim_id": "C1",
                    "text": "El indicador agregado de generosidad volvería a niveles próximos a los alcanzados en 2015.",
                    "evidence_ids": ["E1", "E2"],
                }
            ],
            "reason": "",
        })

        # Mock verifier: Call 1 (verification) supports the candidate claim
        verif_resp = _mock_verif_resp([
            {
                "claim_id": "C1",
                "verdict": "supported",
                "reason": "dato respaldado en E2",
                "evidence_verdicts": [
                    {"evidence_id": "E1", "relation": "irrelevant"},
                    {"evidence_id": "E2", "relation": "supports"},
                ],
            }
        ])

        # Mock verifier: Call 2 (reaudit query_coverage) marks entire query answered
        audit_resp = _mock_audit_resp([
            {
                "query_span": query,
                "decision": "answered",
                "claim_ids": ["C2"],
                "reason_code": "covered",
            }
        ])

        gen = _MockCorrectionGenerator([gen_resp])
        verif = _MockCorrectionVerifier([verif_resp, audit_resp])

        res = correct_coverage_once(
            query,
            factual_report,
            inventory,
            coverage_audit,
            pages=pages,
            generator=gen,
            verifier=verif,
        )

        self.assertTrue(res["attempted"], msg=f"res: {res.get('diagnostics')}")
        self.assertTrue(res["applied"], msg=f"res: {res.get('diagnostics')}")
        self.assertEqual(len(res["inventory"]), 2)
        self.assertEqual(res["inventory"][0]["claim_id"], "C1")
        self.assertEqual(res["inventory"][1]["claim_id"], "C2")
        self.assertEqual(res["inventory"][1]["origin"], "coverage_correction")
        self.assertEqual(res["coverage_audit"]["validated_payload"]["status"], "answered")

        # Assert fixture inputs remain unchanged
        self.assertEqual(factual_report, factual_report_snapshot)
        self.assertEqual(fixture_path.read_bytes(), raw_bytes_before)

    def test_metadata_alias_only_rejected(self):
        """Senior Item 1: metadata with generator_model/generator_digest aliases rejected, authentic model/digest required."""
        alias_rep = copy.deepcopy(self.factual_report)
        alias_rep["metadata"]["generator_model"] = alias_rep["metadata"].pop("model")
        alias_rep["metadata"]["generator_digest"] = alias_rep["metadata"].pop("digest")

        res = correct_coverage_once(
            self.query,
            alias_rep,
            self.initial_inventory,
            self.initial_coverage_audit,
            pages=self.pages,
            generator=_MockCorrectionGenerator([]),
            verifier=_MockCorrectionVerifier([]),
        )
        self.assertTrue(res["attempted"])
        self.assertFalse(res["applied"])
        self.assertIsNone(res["diagnostics"]["model_binding"])
        self.assertIn("alias keys", res["diagnostics"]["reason"])

    def test_metadata_wrong_model_digest_options_rejected(self):
        """Senior Item 1 & 2: original metadata with wrong model, digest, extra options, or mutated types fails closed."""
        cases = [
            ("model", "wrong:model", "model mismatch"),
            ("digest", "d" * 64, "digest mismatch"),
            ("verifier_model", "wrong:verifier", "verifier_model mismatch"),
            ("verifier_digest", "e" * 64, "verifier_digest mismatch"),
        ]
        for field, bad_val, err_pattern in cases:
            bad_rep = copy.deepcopy(self.factual_report)
            bad_rep["metadata"][field] = bad_val
            res = correct_coverage_once(
                self.query,
                bad_rep,
                self.initial_inventory,
                self.initial_coverage_audit,
                pages=self.pages,
                generator=_MockCorrectionGenerator([]),
                verifier=_MockCorrectionVerifier([]),
            )
            self.assertTrue(res["attempted"])
            self.assertFalse(res["applied"])
            self.assertIn(err_pattern, res["diagnostics"]["reason"])

        # Extra option in generator_parameters (repeat_penalty=999)
        bad_rep_opts = copy.deepcopy(self.factual_report)
        bad_rep_opts["metadata"]["generator_parameters"]["repeat_penalty"] = 999
        res_opts = correct_coverage_once(
            self.query,
            bad_rep_opts,
            self.initial_inventory,
            self.initial_coverage_audit,
            pages=self.pages,
            generator=_MockCorrectionGenerator([]),
            verifier=_MockCorrectionVerifier([]),
        )
        self.assertTrue(res_opts["attempted"])
        self.assertFalse(res_opts["applied"])
        self.assertIn("unexpected option", res_opts["diagnostics"]["reason"])

        # Extra option in verifier_parameters
        bad_rep_vopts = copy.deepcopy(self.factual_report)
        bad_rep_vopts["metadata"]["verifier_parameters"]["extra_param"] = 123
        res_vopts = correct_coverage_once(
            self.query,
            bad_rep_vopts,
            self.initial_inventory,
            self.initial_coverage_audit,
            pages=self.pages,
            generator=_MockCorrectionGenerator([]),
            verifier=_MockCorrectionVerifier([]),
        )
        self.assertTrue(res_vopts["attempted"])
        self.assertFalse(res_vopts["applied"])
        self.assertIn("unexpected option", res_vopts["diagnostics"]["reason"])

        # Mutated type in generator_parameters: temperature as float instead of int
        bad_rep_type = copy.deepcopy(self.factual_report)
        bad_rep_type["metadata"]["generator_parameters"]["temperature"] = 0.0
        res_type = correct_coverage_once(
            self.query,
            bad_rep_type,
            self.initial_inventory,
            self.initial_coverage_audit,
            pages=self.pages,
            generator=_MockCorrectionGenerator([]),
            verifier=_MockCorrectionVerifier([]),
        )
        self.assertTrue(res_type["attempted"])
        self.assertFalse(res_type["applied"])
        self.assertIn("must be int", res_type["diagnostics"]["reason"])

        # Mutated type in verifier_parameters: min_p as int instead of float
        bad_rep_vtype = copy.deepcopy(self.factual_report)
        bad_rep_vtype["metadata"]["verifier_parameters"]["min_p"] = 0
        res_vtype = correct_coverage_once(
            self.query,
            bad_rep_vtype,
            self.initial_inventory,
            self.initial_coverage_audit,
            pages=self.pages,
            generator=_MockCorrectionGenerator([]),
            verifier=_MockCorrectionVerifier([]),
        )
        self.assertTrue(res_vtype["attempted"])
        self.assertFalse(res_vtype["applied"])
        self.assertIn("must be float", res_vtype["diagnostics"]["reason"])

    def test_unknown_extra_option_only_in_generation_request(self):
        """Senior Item 2: extra repeat_penalty=999 appearing ONLY in generation request fails closed."""
        bad_opts = copy.deepcopy(PINNED_9B_PARAMETERS)
        bad_opts["repeat_penalty"] = 999

        gen_resp = _mock_gen_resp({
            "status": "answered",
            "claims": [{"claim_id": "C1", "text": "En 2070 la tasa disminuye al 53,7%.", "evidence_ids": ["E1"]}],
            "reason": "",
        })
        gen = _MockCorrectionGenerator([gen_resp], request_options_override=bad_opts)
        verif = _MockCorrectionVerifier([])

        res = correct_coverage_once(
            self.query,
            self.factual_report,
            self.initial_inventory,
            self.initial_coverage_audit,
            pages=self.pages,
            generator=gen,
            verifier=verif,
        )
        self.assertTrue(res["attempted"])
        self.assertFalse(res["applied"])
        self.assertIn("Generator actual request options invalid", res["diagnostics"]["reason"])
        self.assertIn("unexpected option", res["diagnostics"]["reason"])

    def test_option_type_or_unknown_key_only_in_factual_verification_request(self):
        """Senior Item 2: option type mismatch or unknown key ONLY in Call 2 verification request fails closed."""
        gen_resp = _mock_gen_resp({
            "status": "answered",
            "claims": [{"claim_id": "C1", "text": "En 2070 la tasa disminuye al 53,7%.", "evidence_ids": ["E1"]}],
            "reason": "",
        })
        verif_resp = _mock_verif_resp([
            {"claim_id": "C1", "verdict": "supported", "reason": "ok", "evidence_verdicts": [{"evidence_id": "E1", "relation": "supports"}]}
        ])

        # Case A: unknown key in Call 2
        bad_verif_opts = copy.deepcopy(PINNED_27B_PARAMETERS)
        bad_verif_opts["repeat_penalty"] = 999

        gen = _MockCorrectionGenerator([gen_resp])
        verif = _MockCorrectionVerifier(
            [verif_resp],
            stage_options_overrides={"verification": bad_verif_opts},
        )

        res = correct_coverage_once(
            self.query,
            self.factual_report,
            self.initial_inventory,
            self.initial_coverage_audit,
            pages=self.pages,
            generator=gen,
            verifier=verif,
        )
        self.assertTrue(res["attempted"])
        self.assertFalse(res["applied"])
        self.assertIn("Verifier actual request options invalid", res["diagnostics"]["reason"])

        # Case B: mutated type in Call 2 (top_k float instead of int)
        bad_verif_opts_b = copy.deepcopy(PINNED_27B_PARAMETERS)
        bad_verif_opts_b["top_k"] = 20.0

        gen_b = _MockCorrectionGenerator([gen_resp])
        verif_b = _MockCorrectionVerifier(
            [verif_resp],
            stage_options_overrides={"verification": bad_verif_opts_b},
        )

        res_b = correct_coverage_once(
            self.query,
            self.factual_report,
            self.initial_inventory,
            self.initial_coverage_audit,
            pages=self.pages,
            generator=gen_b,
            verifier=verif_b,
        )
        self.assertTrue(res_b["attempted"])
        self.assertFalse(res_b["applied"])
        self.assertIn("Verifier actual request options invalid", res_b["diagnostics"]["reason"])

    def test_temperature_swap_only_in_third_query_coverage_request(self):
        """Senior Item 2: temperature=0.8 appearing ONLY in Call 3 reaudit request causes applied=False and fallback."""
        gen_resp = _mock_gen_resp({
            "status": "answered",
            "claims": [{"claim_id": "C1", "text": "En 2070 la tasa disminuye al 53,7%.", "evidence_ids": ["E1"]}],
            "reason": "",
        })
        verif_resp = _mock_verif_resp([
            {"claim_id": "C1", "verdict": "supported", "reason": "ok", "evidence_verdicts": [{"evidence_id": "E1", "relation": "supports"}]}
        ])
        audit_resp = _mock_audit_resp([
            {"query_span": "¿Cuál es la tasa en 2050", "decision": "answered", "claim_ids": ["C1"], "reason_code": "covered"},
            {"query_span": " y en 2070?", "decision": "answered", "claim_ids": ["C2"], "reason_code": "covered"},
        ])

        bad_reaudit_opts = copy.deepcopy(PINNED_27B_PARAMETERS)
        bad_reaudit_opts["temperature"] = 0.8  # Mutated in Call 3 ONLY

        gen = _MockCorrectionGenerator([gen_resp])
        verif = _MockCorrectionVerifier(
            [verif_resp, audit_resp],
            stage_options_overrides={"query_coverage": bad_reaudit_opts},
        )

        res = correct_coverage_once(
            self.query,
            self.factual_report,
            self.initial_inventory,
            self.initial_coverage_audit,
            pages=self.pages,
            generator=gen,
            verifier=verif,
        )
        self.assertTrue(res["attempted"])
        self.assertFalse(res["applied"])
        self.assertEqual(res["inventory"], self.initial_inventory)
        self.assertEqual(res["coverage_audit"], self.initial_coverage_audit)
        r_trace = res["diagnostics"]["reaudit"]
        self.assertIsNotNone(r_trace)
        self.assertEqual(r_trace["status"], "error")
        self.assertIn("Coverage reaudit actual request options invalid", str(r_trace["error"]))
        self.assertIn("must be int", str(r_trace["error"]))

    def test_declared_params_unchanged_but_wire_request_changed(self):
        """Senior Item 2: client.parameters match pinned policy but wire request options mutated fails closed."""
        gen_resp = _mock_gen_resp({
            "status": "answered",
            "claims": [{"claim_id": "C1", "text": "Texto.", "evidence_ids": ["E1"]}],
            "reason": "",
        })
        wire_mutated_opts = copy.deepcopy(PINNED_9B_PARAMETERS)
        wire_mutated_opts["num_predict"] = 2048  # Wire changed from 1024

        gen = _MockCorrectionGenerator(
            [gen_resp],
            parameters=copy.deepcopy(PINNED_9B_PARAMETERS),  # Declared parameters remain unchanged
            request_options_override=wire_mutated_opts,       # Wire changed
        )
        verif = _MockCorrectionVerifier([])

        res = correct_coverage_once(
            self.query,
            self.factual_report,
            self.initial_inventory,
            self.initial_coverage_audit,
            pages=self.pages,
            generator=gen,
            verifier=verif,
        )
        self.assertTrue(res["attempted"])
        self.assertFalse(res["applied"])
        self.assertIn("Generator actual request options invalid", res["diagnostics"]["reason"])

    def test_verification_failure_preserves_attempted_call_and_trace(self):
        """Item 3: verification response failure preserves attempted verifier call, raw_response, request, and error."""
        gen_resp = _mock_gen_resp({
            "status": "answered",
            "claims": [{"claim_id": "C1", "text": "Texto candidate.", "evidence_ids": ["E1"]}],
            "reason": "",
        })
        bad_verif_resp = {
            "model": PINNED_27B_MODEL,
            "done": True,
            "done_reason": "stop",
            "prompt_eval_count": 220,
            "eval_count": 35,
            "prompt_eval_duration": 20000000,
            "eval_duration": 30000000,
            "total_duration": 50000000,
            "load_duration": 2000000,
            "message": {"role": "assistant", "content": "not json at all {"},
        }
        gen = _MockCorrectionGenerator([gen_resp])
        verif = _MockCorrectionVerifier([bad_verif_resp])

        res = correct_coverage_once(
            self.query, self.factual_report, self.initial_inventory, self.initial_coverage_audit,
            pages=self.pages, generator=gen, verifier=verif
        )
        self.assertTrue(res["attempted"])
        self.assertFalse(res["applied"])
        self.assertEqual(len(gen.calls), 1)
        self.assertEqual(len(verif.calls), 1)
        self.assertIsNotNone(res["diagnostics"]["generation"])
        v_trace = res["diagnostics"]["verification"]
        self.assertIsNotNone(v_trace)
        self.assertEqual(v_trace["raw_response"], bad_verif_resp)
        self.assertIsNotNone(v_trace["request"])
        self.assertIn("JSON", str(v_trace["error"]))
        self.assertIsNone(res["diagnostics"]["reaudit"])
        self.assertEqual(res["inventory"], self.initial_inventory)

    def test_reaudit_transport_error_retains_error_and_call_delta(self):
        """Item 3: reaudit transport error after two verifier calls retains reaudit error and call delta."""
        gen_resp = _mock_gen_resp({
            "status": "answered",
            "claims": [{"claim_id": "C1", "text": "En 2070 la tasa disminuye al 53,7%.", "evidence_ids": ["E1"]}],
            "reason": "",
        })
        verif_resp = _mock_verif_resp([
            {"claim_id": "C1", "verdict": "supported", "reason": "ok", "evidence_verdicts": [{"evidence_id": "E1", "relation": "supports"}]}
        ])
        gen = _MockCorrectionGenerator([gen_resp])
        verif = _MockCorrectionVerifier([verif_resp, RuntimeError("Transport connection lost")])

        res = correct_coverage_once(
            self.query, self.factual_report, self.initial_inventory, self.initial_coverage_audit,
            pages=self.pages, generator=gen, verifier=verif
        )
        self.assertTrue(res["attempted"])
        self.assertFalse(res["applied"])
        self.assertEqual(len(gen.calls), 1)
        self.assertEqual(len(verif.calls), 2)
        r_trace = res["diagnostics"]["reaudit"]
        self.assertIsNotNone(r_trace)
        self.assertEqual(r_trace["status"], "error")
        self.assertIn("Transport connection lost", str(r_trace["error"]))
        self.assertEqual(len(r_trace["calls"]), 1)
        self.assertEqual(res["inventory"], self.initial_inventory)

    def test_initial_audit_negative_start_char_fails_closed_zero_calls(self):
        """Item 4: start_char=-1 in initial audit fails validation and makes zero client calls."""
        bad_audit = copy.deepcopy(self.initial_coverage_audit)
        bad_audit["validated_payload"]["parts"][0]["start_char"] = -1
        gen = _MockCorrectionGenerator([])
        verif = _MockCorrectionVerifier([])

        res = correct_coverage_once(
            self.query, self.factual_report, self.initial_inventory, bad_audit,
            pages=self.pages, generator=gen, verifier=verif
        )
        self.assertFalse(res["attempted"])
        self.assertFalse(res["applied"])
        self.assertEqual(len(gen.calls), 0)
        self.assertEqual(len(verif.calls), 0)
        self.assertEqual(res["inventory"], self.initial_inventory)
        self.assertIn("coordinates", res["diagnostics"]["reason"])

    def test_inventory_mismatch_against_canonical_makes_zero_calls(self):
        """Item 4: inventory mismatch with canonical claim_inventory(factual_report) makes zero calls."""
        mismatched_inv = copy.deepcopy(self.initial_inventory)
        mismatched_inv[0]["text"] = "Texto inventado no coincidente."
        gen = _MockCorrectionGenerator([])
        verif = _MockCorrectionVerifier([])

        res = correct_coverage_once(
            self.query, self.factual_report, mismatched_inv, self.initial_coverage_audit,
            pages=self.pages, generator=gen, verifier=verif
        )
        self.assertFalse(res["attempted"])
        self.assertFalse(res["applied"])
        self.assertEqual(len(gen.calls), 0)
        self.assertEqual(len(verif.calls), 0)
        self.assertIn("Initial inventory mismatch", res["diagnostics"]["reason"])

    def test_malformed_initial_audit_intervals_make_zero_calls(self):
        """Item 4: initial audit with float, bool, unknown keys, missing keys, gap, or overlap makes zero calls."""
        cases = [
            {"patch": lambda a: a["validated_payload"]["parts"][0].__setitem__("start_char", 0.0)},
            {"patch": lambda a: a["validated_payload"]["parts"][0].__setitem__("start_char", False)},
            {"patch": lambda a: a["validated_payload"]["parts"][0].__setitem__("forbidden_extra", 123)},
            {"patch": lambda a: a["validated_payload"]["parts"][0].__delitem__("reason_code")},
            {"patch": lambda a: a["validated_payload"]["parts"][0].__setitem__("start_char", 5)},
            {"patch": lambda a: a["validated_payload"]["parts"][1].__setitem__("start_char", 10)},
            {"patch": lambda a: a["validated_payload"].__setitem__("forbidden_top", "bad")},
        ]
        for c in cases:
            bad_audit = copy.deepcopy(self.initial_coverage_audit)
            c["patch"](bad_audit)
            gen = _MockCorrectionGenerator([])
            verif = _MockCorrectionVerifier([])
            res = correct_coverage_once(
                self.query, self.factual_report, self.initial_inventory, bad_audit,
                pages=self.pages, generator=gen, verifier=verif
            )
            self.assertFalse(res["attempted"])
            self.assertFalse(res["applied"])
            self.assertEqual(len(gen.calls), 0)
            self.assertEqual(len(verif.calls), 0)

    def test_genuine_rejected_correction_returns_byte_identical_fallback(self):
        """Item 4: when candidate is contradicted or rejected, returns byte-identical fallback."""
        gen_resp = _mock_gen_resp({
            "status": "answered",
            "claims": [{"claim_id": "C1", "text": "Texto contradicho.", "evidence_ids": ["E1"]}],
            "reason": "",
        })
        verif_resp = _mock_verif_resp([
            {"claim_id": "C1", "verdict": "contradicted", "reason": "no coincide", "evidence_verdicts": [{"evidence_id": "E1", "relation": "contradicts"}]}
        ])
        gen = _MockCorrectionGenerator([gen_resp])
        verif = _MockCorrectionVerifier([verif_resp])

        res = correct_coverage_once(
            self.query, self.factual_report, self.initial_inventory, self.initial_coverage_audit,
            pages=self.pages, generator=gen, verifier=verif
        )
        self.assertTrue(res["attempted"])
        self.assertFalse(res["applied"])
        self.assertEqual(res["inventory"], self.initial_inventory)
        self.assertEqual(res["coverage_audit"], self.initial_coverage_audit)
        self.assertEqual(
            json.dumps(res["inventory"], sort_keys=True),
            json.dumps(self.initial_inventory, sort_keys=True),
        )
        self.assertEqual(
            json.dumps(res["coverage_audit"], sort_keys=True),
            json.dumps(self.initial_coverage_audit, sort_keys=True),
        )

    def test_get_digest_initial_error_fails_closed_zero_calls(self):
        """Item 4: initial get_digest network/policy exception returns applied=False with 0 LLM calls."""
        class _FailingDigestGen(_MockCorrectionGenerator):
            def get_digest(self) -> str:
                raise ConnectionError("Ollama daemon offline")

        class _FailingDigestVerif(_MockCorrectionVerifier):
            def get_digest(self) -> str:
                raise ConnectionError("Verifier daemon offline")

        # 1. Generator get_digest failure
        gen1 = _FailingDigestGen([])
        verif1 = _MockCorrectionVerifier([])
        res1 = correct_coverage_once(
            self.query, self.factual_report, self.initial_inventory, self.initial_coverage_audit,
            pages=self.pages, generator=gen1, verifier=verif1
        )
        self.assertTrue(res1["attempted"])
        self.assertFalse(res1["applied"])
        self.assertEqual(len(gen1.calls), 0)
        self.assertEqual(len(verif1.calls), 0)
        self.assertIn("Generator client get_digest failed before call", res1["diagnostics"]["reason"])
        self.assertEqual(res1["inventory"], self.initial_inventory)
        self.assertEqual(res1["coverage_audit"], self.initial_coverage_audit)

        # 2. Verifier get_digest failure
        gen2 = _MockCorrectionGenerator([])
        verif2 = _FailingDigestVerif([])
        res2 = correct_coverage_once(
            self.query, self.factual_report, self.initial_inventory, self.initial_coverage_audit,
            pages=self.pages, generator=gen2, verifier=verif2
        )
        self.assertTrue(res2["attempted"])
        self.assertFalse(res2["applied"])
        self.assertEqual(len(gen2.calls), 0)
        self.assertEqual(len(verif2.calls), 0)
        self.assertIn("Verifier client get_digest failed before call", res2["diagnostics"]["reason"])
        self.assertEqual(res2["inventory"], self.initial_inventory)
        self.assertEqual(res2["coverage_audit"], self.initial_coverage_audit)

    def test_post_gen_digest_error_preserves_generation_trace(self):
        """Item 4: get_digest error after Call 1 preserves generation trace and returns applied=False."""
        class _FailingPostGenDigest(_MockCorrectionGenerator):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.digest_call_count = 0

            def get_digest(self) -> str:
                self.digest_call_count += 1
                if self.digest_call_count > 1:
                    raise ConnectionError("Network dropped after generation")
                return self.digest

        gen_resp = _mock_gen_resp({
            "status": "answered",
            "claims": [{"claim_id": "C1", "text": "Texto válido.", "evidence_ids": ["E1"]}],
            "reason": "",
        })
        gen = _FailingPostGenDigest([gen_resp])
        verif = _MockCorrectionVerifier([])
        res = correct_coverage_once(
            self.query, self.factual_report, self.initial_inventory, self.initial_coverage_audit,
            pages=self.pages, generator=gen, verifier=verif
        )
        self.assertTrue(res["attempted"])
        self.assertFalse(res["applied"])
        self.assertEqual(len(gen.calls), 1)
        self.assertEqual(len(verif.calls), 0)
        self.assertIsNotNone(res["diagnostics"]["generation"])
        self.assertIsNotNone(res["diagnostics"]["generation"]["raw_response"])
        self.assertIsNotNone(res["diagnostics"]["generation"]["request"])
        self.assertEqual(len(res["diagnostics"]["generation"]["client_calls"]), 1)
        self.assertIn("Generator get_digest failed after call", res["diagnostics"]["generation"]["error"])
        self.assertEqual(res["inventory"], self.initial_inventory)

    def test_verification_messages_budget_or_builder_error_preserves_generation_trace(self):
        """Item 4: verification context overflow/builder ValueError after 1 gen preserves generation trace and fails closed."""
        gen_resp = _mock_gen_resp({
            "status": "answered",
            "claims": [{"claim_id": "C1", "text": "Texto válido.", "evidence_ids": ["E1"]}],
            "reason": "",
        })
        gen = _MockCorrectionGenerator([gen_resp])
        verif = _MockCorrectionVerifier([])

        import unittest.mock as mock
        with mock.patch(
            "experiments.query_coverage.build_verification_messages",
            side_effect=ValueError("Verification messages exceed budget of 20000 bytes UTF-8 (25000 bytes)")
        ):
            res = correct_coverage_once(
                self.query, self.factual_report, self.initial_inventory, self.initial_coverage_audit,
                pages=self.pages, generator=gen, verifier=verif
            )

        self.assertTrue(res["attempted"])
        self.assertFalse(res["applied"])
        self.assertEqual(len(gen.calls), 1)
        self.assertEqual(len(verif.calls), 0)
        self.assertIsNotNone(res["diagnostics"]["generation"])
        self.assertIsNotNone(res["diagnostics"]["generation"]["raw_response"])
        self.assertIsNotNone(res["diagnostics"]["generation"]["request"])
        self.assertEqual(len(res["diagnostics"]["generation"]["client_calls"]), 1)
        self.assertIsNotNone(res["diagnostics"]["verification"])
        self.assertIn("Verification messages exceed budget", res["diagnostics"]["verification"]["error"])
        self.assertEqual(res["inventory"], self.initial_inventory)
        self.assertEqual(res["coverage_audit"], self.initial_coverage_audit)

    def test_strict_serialization_with_nan_inf_in_stats_preserves_client_logs(self):
        """Item 5: NaN, +Inf, -Inf in generation/verification/reaudit serialize strictly with json.dumps(allow_nan=False) without mutating client logs."""
        # 1. NaN in generation stats
        gen_resp1 = _mock_gen_resp({
            "status": "answered",
            "claims": [{"claim_id": "C1", "text": "Texto.", "evidence_ids": ["E1"]}],
            "reason": "",
        })
        gen_resp1["total_duration"] = float("nan")
        gen1 = _MockCorrectionGenerator([gen_resp1])
        verif1 = _MockCorrectionVerifier([])
        res1 = correct_coverage_once(
            self.query, self.factual_report, self.initial_inventory, self.initial_coverage_audit,
            pages=self.pages, generator=gen1, verifier=verif1
        )
        self.assertFalse(res1["applied"])
        # Strict serialization without allow_nan must succeed
        ser1 = json.dumps(res1, allow_nan=False)
        self.assertIn('"NaN"', ser1)
        # Client log must remain unmutated with original float NaN
        self.assertTrue(math.isnan(gen1.calls[0]["raw_response"]["total_duration"]))
        self.assertEqual(res1["inventory"], self.initial_inventory)
        self.assertEqual(res1["coverage_audit"], self.initial_coverage_audit)

        # 2. +Inf in verification stats
        gen_resp2 = _mock_gen_resp({
            "status": "answered",
            "claims": [{"claim_id": "C1", "text": "Texto.", "evidence_ids": ["E1"]}],
            "reason": "",
        })
        verif_resp2 = _mock_verif_resp([
            {"claim_id": "C1", "verdict": "supported", "reason": "ok", "evidence_verdicts": [{"evidence_id": "E1", "relation": "supports"}]}
        ])
        verif_resp2["total_duration"] = float("inf")
        gen2 = _MockCorrectionGenerator([gen_resp2])
        verif2 = _MockCorrectionVerifier([verif_resp2])
        res2 = correct_coverage_once(
            self.query, self.factual_report, self.initial_inventory, self.initial_coverage_audit,
            pages=self.pages, generator=gen2, verifier=verif2
        )
        self.assertFalse(res2["applied"])
        ser2 = json.dumps(res2, allow_nan=False)
        self.assertIn('"Infinity"', ser2)
        self.assertTrue(math.isinf(verif2.calls[0]["raw_response"]["total_duration"]))
        self.assertGreater(verif2.calls[0]["raw_response"]["total_duration"], 0)
        self.assertEqual(res2["inventory"], self.initial_inventory)

        # 3. -Inf in reaudit stats
        gen_resp3 = _mock_gen_resp({
            "status": "answered",
            "claims": [{"claim_id": "C1", "text": "Texto.", "evidence_ids": ["E1"]}],
            "reason": "",
        })
        verif_resp3 = _mock_verif_resp([
            {"claim_id": "C1", "verdict": "supported", "reason": "ok", "evidence_verdicts": [{"evidence_id": "E1", "relation": "supports"}]}
        ])
        audit_resp3 = _mock_audit_resp([
            {"query_span": "¿Cuál es la tasa en 2050", "decision": "answered", "claim_ids": ["C1"], "reason_code": "covered"},
            {"query_span": " y en 2070?", "decision": "answered", "claim_ids": ["C2"], "reason_code": "covered"},
        ])
        audit_resp3["total_duration"] = float("-inf")
        gen3 = _MockCorrectionGenerator([gen_resp3])
        verif3 = _MockCorrectionVerifier([verif_resp3, audit_resp3])
        res3 = correct_coverage_once(
            self.query, self.factual_report, self.initial_inventory, self.initial_coverage_audit,
            pages=self.pages, generator=gen3, verifier=verif3
        )
        self.assertFalse(res3["applied"])
        ser3 = json.dumps(res3, allow_nan=False)
        self.assertIn('"-Infinity"', ser3)
        self.assertTrue(math.isinf(verif3.calls[1]["raw_response"]["total_duration"]))
        self.assertLess(verif3.calls[1]["raw_response"]["total_duration"], 0)
        self.assertEqual(res3["inventory"], self.initial_inventory)

    def test_top_p_nan_in_verification_request_fails_closed_and_serializes(self):
        """P1: top_p=NaN in verifier request options fails closed, serializes strictly with allow_nan=False, and keeps client logs immutable."""
        gen_resp = _mock_gen_resp({
            "status": "answered",
            "claims": [{"claim_id": "C1", "text": "Texto válido.", "evidence_ids": ["E1"]}],
            "reason": "",
        })
        verif_resp = _mock_verif_resp([
            {"claim_id": "C1", "verdict": "supported", "reason": "ok", "evidence_verdicts": [{"evidence_id": "E1", "relation": "supports"}]}
        ])
        gen = _MockCorrectionGenerator([gen_resp])
        nan_opts = copy.deepcopy(PINNED_27B_PARAMETERS)
        nan_opts["top_p"] = float("nan")
        verif = _MockCorrectionVerifier(
            [verif_resp],
            stage_options_overrides={"verification": nan_opts},
        )

        res = correct_coverage_once(
            self.query, self.factual_report, self.initial_inventory, self.initial_coverage_audit,
            pages=self.pages, generator=gen, verifier=verif
        )
        self.assertTrue(res["attempted"])
        self.assertFalse(res["applied"])
        self.assertEqual(res["inventory"], self.initial_inventory)
        self.assertEqual(res["coverage_audit"], self.initial_coverage_audit)

        # Strict JSON serialization without allow_nan must succeed
        ser = json.dumps(res, allow_nan=False)
        self.assertIn('"NaN"', ser)

        # model_binding effective_options must not contain float NaN
        mb_verif = res["diagnostics"]["model_binding"]["verifier"]
        self.assertIsNotNone(mb_verif)
        self.assertEqual(mb_verif["effective_options"]["top_p"], "NaN")

        # Client raw log must remain immutable with original float NaN
        client_top_p = verif.calls[0]["request"]["options"]["top_p"]
        self.assertTrue(math.isnan(client_top_p))
        self.assertIsInstance(client_top_p, float)

    def test_top_p_tiny_deviation_refused(self):
        """P1: top_p=0.9500005 in verifier request options is refused under exact equality (no epsilon/tolerance)."""
        gen_resp = _mock_gen_resp({
            "status": "answered",
            "claims": [{"claim_id": "C1", "text": "Texto válido.", "evidence_ids": ["E1"]}],
            "reason": "",
        })
        verif_resp = _mock_verif_resp([
            {"claim_id": "C1", "verdict": "supported", "reason": "ok", "evidence_verdicts": [{"evidence_id": "E1", "relation": "supports"}]}
        ])
        gen = _MockCorrectionGenerator([gen_resp])
        deviant_opts = copy.deepcopy(PINNED_27B_PARAMETERS)
        deviant_opts["top_p"] = 0.9500005
        verif = _MockCorrectionVerifier(
            [verif_resp],
            stage_options_overrides={"verification": deviant_opts},
        )

        res = correct_coverage_once(
            self.query, self.factual_report, self.initial_inventory, self.initial_coverage_audit,
            pages=self.pages, generator=gen, verifier=verif
        )
        self.assertTrue(res["attempted"])
        self.assertFalse(res["applied"])
        self.assertIn("value mismatch", res["diagnostics"]["reason"])
        self.assertEqual(res["inventory"], self.initial_inventory)

    def test_verification_rejects_altered_messages_or_generic_schema(self):
        """P1: Verifier sending generic schema {'type': 'object'} or altered messages fails closed."""
        gen_resp = _mock_gen_resp({
            "status": "answered",
            "claims": [{"claim_id": "C1", "text": "Texto.", "evidence_ids": ["E1"]}],
            "reason": "",
        })
        verif_resp = _mock_verif_resp([
            {"claim_id": "C1", "verdict": "supported", "reason": "ok", "evidence_verdicts": [{"evidence_id": "E1", "relation": "supports"}]}
        ])

        # 1. Generic schema in verification
        gen1 = _MockCorrectionGenerator([gen_resp])
        verif1 = _MockCorrectionVerifier(
            [verif_resp],
            stage_request_overrides={"verification": {"format": {"type": "object"}}},
        )
        res1 = correct_coverage_once(
            self.query, self.factual_report, self.initial_inventory, self.initial_coverage_audit,
            pages=self.pages, generator=gen1, verifier=verif1
        )
        self.assertFalse(res1["applied"])
        self.assertIn("format schema does not match planned schema", res1["diagnostics"]["reason"])

        # 2. Altered messages in verification
        gen2 = _MockCorrectionGenerator([gen_resp])
        verif2 = _MockCorrectionVerifier(
            [verif_resp],
            stage_request_overrides={"verification": {"messages": [{"role": "user", "content": "foreign"}]}},
        )
        res2 = correct_coverage_once(
            self.query, self.factual_report, self.initial_inventory, self.initial_coverage_audit,
            pages=self.pages, generator=gen2, verifier=verif2
        )
        self.assertFalse(res2["applied"])
        self.assertIn("messages do not match planned messages", res2["diagnostics"]["reason"])

    def test_reaudit_rejects_altered_messages_or_generic_schema(self):
        """P1: Coverage reaudit sending generic schema {'type': 'object'} or altered messages fails closed."""
        gen_resp = _mock_gen_resp({
            "status": "answered",
            "claims": [{"claim_id": "C1", "text": "Texto.", "evidence_ids": ["E1"]}],
            "reason": "",
        })
        verif_resp = _mock_verif_resp([
            {"claim_id": "C1", "verdict": "supported", "reason": "ok", "evidence_verdicts": [{"evidence_id": "E1", "relation": "supports"}]}
        ])
        audit_resp = _mock_audit_resp([
            {"query_span": "¿Cuál es la tasa en 2050", "decision": "answered", "claim_ids": ["C1"], "reason_code": "covered"},
            {"query_span": " y en 2070?", "decision": "answered", "claim_ids": ["C2"], "reason_code": "covered"},
        ])

        # 1. Generic schema in reaudit
        gen1 = _MockCorrectionGenerator([gen_resp])
        verif1 = _MockCorrectionVerifier(
            [verif_resp, audit_resp],
            stage_request_overrides={"query_coverage": {"format": {"type": "object"}}},
        )
        res1 = correct_coverage_once(
            self.query, self.factual_report, self.initial_inventory, self.initial_coverage_audit,
            pages=self.pages, generator=gen1, verifier=verif1
        )
        self.assertFalse(res1["applied"])
        self.assertIn("format schema does not match planned schema", res1["diagnostics"]["reason"])

        # 2. Altered messages in reaudit
        gen2 = _MockCorrectionGenerator([gen_resp])
        verif2 = _MockCorrectionVerifier(
            [verif_resp, audit_resp],
            stage_request_overrides={"query_coverage": {"messages": [{"role": "user", "content": "tampered"}]}},
        )
        res2 = correct_coverage_once(
            self.query, self.factual_report, self.initial_inventory, self.initial_coverage_audit,
            pages=self.pages, generator=gen2, verifier=verif2
        )
        self.assertFalse(res2["applied"])
        self.assertIn("messages do not match planned messages", res2["diagnostics"]["reason"])

    def test_tampered_model_stream_think_and_extra_omitted_keys_rejected(self):
        """P1: Tampered model, stream, think or extra/missing keys in client request are rejected."""
        gen_resp = _mock_gen_resp({
            "status": "answered",
            "claims": [{"claim_id": "C1", "text": "Texto.", "evidence_ids": ["E1"]}],
            "reason": "",
        })
        verif_resp = _mock_verif_resp([
            {"claim_id": "C1", "verdict": "supported", "reason": "ok", "evidence_verdicts": [{"evidence_id": "E1", "relation": "supports"}]}
        ])

        # 1. Wrong model in verification
        gen1 = _MockCorrectionGenerator([gen_resp])
        verif1 = _MockCorrectionVerifier(
            [verif_resp],
            stage_request_overrides={"verification": {"model": "wrong-model:27b"}},
        )
        res1 = correct_coverage_once(
            self.query, self.factual_report, self.initial_inventory, self.initial_coverage_audit,
            pages=self.pages, generator=gen1, verifier=verif1
        )
        self.assertFalse(res1["applied"])
        self.assertIn("model mismatch", res1["diagnostics"]["reason"])

        # 2. stream=True in verification
        gen2 = _MockCorrectionGenerator([gen_resp])
        verif2 = _MockCorrectionVerifier(
            [verif_resp],
            stage_request_overrides={"verification": {"stream": True}},
        )
        res2 = correct_coverage_once(
            self.query, self.factual_report, self.initial_inventory, self.initial_coverage_audit,
            pages=self.pages, generator=gen2, verifier=verif2
        )
        self.assertFalse(res2["applied"])
        self.assertIn("'stream' mismatch", res2["diagnostics"]["reason"])

        # 3. Extra key in verification
        gen3 = _MockCorrectionGenerator([gen_resp])
        verif3 = _MockCorrectionVerifier(
            [verif_resp],
            stage_request_overrides={"verification": {"extra_field": "forbidden"}},
        )
        res3 = correct_coverage_once(
            self.query, self.factual_report, self.initial_inventory, self.initial_coverage_audit,
            pages=self.pages, generator=gen3, verifier=verif3
        )
        self.assertFalse(res3["applied"])
        self.assertIn("keys mismatch", res3["diagnostics"]["reason"])

    def test_schema_min_items_true_vs_1_rejected_in_verification(self):
        """Fourth-senior P2: changing minItems:1 to minItems:true in verification schema fails closed under canonical JSON."""
        gen_resp = _mock_gen_resp({
            "status": "answered",
            "claims": [{"claim_id": "C1", "text": "En 2070 la tasa disminuye al 53,7%.", "evidence_ids": ["E1"]}],
            "reason": "",
        })
        verif_resp = _mock_verif_resp([
            {"claim_id": "C1", "verdict": "supported", "reason": "ok", "evidence_verdicts": [{"evidence_id": "E1", "relation": "supports"}]}
        ])
        gen = _MockCorrectionGenerator([gen_resp])

        def _mutate_schema(s: dict):
            s["properties"]["verdicts"]["minItems"] = True

        verif = _MockCorrectionVerifier(
            [verif_resp],
            in_place_schema_mutation_by_stage={"verification": _mutate_schema},
        )
        res = correct_coverage_once(
            self.query, self.factual_report, self.initial_inventory, self.initial_coverage_audit,
            pages=self.pages, generator=gen, verifier=verif
        )
        self.assertTrue(res["attempted"])
        self.assertFalse(res["applied"])
        self.assertIn("format schema does not match planned schema", res["diagnostics"]["reason"])
        self.assertEqual(res["inventory"], self.initial_inventory)
        self.assertEqual(res["coverage_audit"], self.initial_coverage_audit)

    def test_schema_min_items_true_vs_1_rejected_in_generation_and_reaudit(self):
        """Fourth-senior P2: changing minItems:1 to minItems:true in generation or reaudit schema fails closed."""
        gen_resp = _mock_gen_resp({
            "status": "answered",
            "claims": [{"claim_id": "C1", "text": "En 2070 la tasa disminuye al 53,7%.", "evidence_ids": ["E1"]}],
            "reason": "",
        })
        verif_resp = _mock_verif_resp([
            {"claim_id": "C1", "verdict": "supported", "reason": "ok", "evidence_verdicts": [{"evidence_id": "E1", "relation": "supports"}]}
        ])
        audit_resp = _mock_audit_resp([
            {"query_span": "¿Cuál es la tasa en 2050", "decision": "answered", "claim_ids": ["C1"], "reason_code": "covered"},
            {"query_span": " y en 2070?", "decision": "answered", "claim_ids": ["C2"], "reason_code": "covered"},
        ])

        # Subcase A: Generation schema in-place minItems: True
        def _mutate_gen_schema(s: dict):
            if "anyOf" in s and isinstance(s["anyOf"], list):
                s["anyOf"][0]["properties"]["claims"]["minItems"] = True
            elif "properties" in s and "claims" in s["properties"]:
                s["properties"]["claims"]["minItems"] = True

        gen_a = _MockCorrectionGenerator([gen_resp], in_place_schema_mutation=_mutate_gen_schema)
        verif_a = _MockCorrectionVerifier([])
        res_a = correct_coverage_once(
            self.query, self.factual_report, self.initial_inventory, self.initial_coverage_audit,
            pages=self.pages, generator=gen_a, verifier=verif_a
        )
        self.assertTrue(res_a["attempted"])
        self.assertFalse(res_a["applied"])
        self.assertIn("format schema does not match planned schema", res_a["diagnostics"]["reason"])

        # Subcase B: Reaudit schema in-place minItems: True
        def _mutate_reaudit_schema(s: dict):
            s["properties"]["parts"]["minItems"] = True

        gen_b = _MockCorrectionGenerator([gen_resp])
        verif_b = _MockCorrectionVerifier(
            [verif_resp, audit_resp],
            in_place_schema_mutation_by_stage={"query_coverage": _mutate_reaudit_schema},
        )
        res_b = correct_coverage_once(
            self.query, self.factual_report, self.initial_inventory, self.initial_coverage_audit,
            pages=self.pages, generator=gen_b, verifier=verif_b
        )
        self.assertTrue(res_b["attempted"])
        self.assertFalse(res_b["applied"])
        r_trace = res_b["diagnostics"]["reaudit"]
        self.assertIsNotNone(r_trace)
        self.assertIn("format schema does not match planned schema", str(r_trace["error"]))

    def test_schema_float_1_0_vs_int_1_rejected(self):
        """Fourth-senior P2: minItems: 1.0 (float) vs 1 (int) fails closed under canonical JSON exact string equality."""
        gen_resp = _mock_gen_resp({
            "status": "answered",
            "claims": [{"claim_id": "C1", "text": "En 2070 la tasa disminuye al 53,7%.", "evidence_ids": ["E1"]}],
            "reason": "",
        })
        verif_resp = _mock_verif_resp([
            {"claim_id": "C1", "verdict": "supported", "reason": "ok", "evidence_verdicts": [{"evidence_id": "E1", "relation": "supports"}]}
        ])
        gen = _MockCorrectionGenerator([gen_resp])

        def _mutate_schema_float(s: dict):
            s["properties"]["verdicts"]["minItems"] = 1.0

        verif = _MockCorrectionVerifier(
            [verif_resp],
            in_place_schema_mutation_by_stage={"verification": _mutate_schema_float},
        )
        res = correct_coverage_once(
            self.query, self.factual_report, self.initial_inventory, self.initial_coverage_audit,
            pages=self.pages, generator=gen, verifier=verif
        )
        self.assertTrue(res["attempted"])
        self.assertFalse(res["applied"])
        self.assertIn("format schema does not match planned schema", res["diagnostics"]["reason"])

    def test_generator_preexisting_history_with_no_new_call_rejected(self):
        """Fourth-senior P1 reproduction: Preexisting valid history in generator but 0 calls during invocation fails closed with client_calls=[] and request=None."""
        gen_resp = _mock_gen_resp({
            "status": "answered",
            "claims": [{"claim_id": "C1", "text": "En 2070 la tasa disminuye al 53,7%.", "evidence_ids": ["E1"]}],
            "reason": "",
        })
        prior_call = {
            "model": PINNED_9B_MODEL,
            "messages": [{"role": "user", "content": "prior query"}],
            "stream": False,
            "think": False,
            "keep_alive": "5m",
            "format": {"type": "object"},
            "options": copy.deepcopy(PINNED_9B_PARAMETERS),
        }
        initial_history = [
            {"request": prior_call, "effective_options": copy.deepcopy(PINNED_9B_PARAMETERS), "raw_response": {"done": True}}
        ]
        gen = _MockCorrectionGenerator([gen_resp], initial_calls=initial_history, calls_to_append=0)
        verif = _MockCorrectionVerifier([])

        res = correct_coverage_once(
            self.query, self.factual_report, self.initial_inventory, self.initial_coverage_audit,
            pages=self.pages, generator=gen, verifier=verif
        )
        self.assertTrue(res["attempted"])
        self.assertFalse(res["applied"])
        gen_diag = res["diagnostics"]["generation"]
        self.assertIsNotNone(gen_diag)
        self.assertEqual(gen_diag["client_calls"], [])
        self.assertIsNone(gen_diag["request"])
        self.assertIn("did not append a new call log entry", res["diagnostics"]["reason"])
        self.assertEqual(res["inventory"], self.initial_inventory)
        self.assertEqual(res["coverage_audit"], self.initial_coverage_audit)

    def test_generator_two_new_records_rejected(self):
        """Fourth-senior P1: Generator appending two new call records during a single invocation fails closed."""
        gen_resp = _mock_gen_resp({
            "status": "answered",
            "claims": [{"claim_id": "C1", "text": "En 2070 la tasa disminuye al 53,7%.", "evidence_ids": ["E1"]}],
            "reason": "",
        })
        gen = _MockCorrectionGenerator([gen_resp], calls_to_append=2)
        verif = _MockCorrectionVerifier([])

        res = correct_coverage_once(
            self.query, self.factual_report, self.initial_inventory, self.initial_coverage_audit,
            pages=self.pages, generator=gen, verifier=verif
        )
        self.assertTrue(res["attempted"])
        self.assertFalse(res["applied"])
        self.assertIn("appended multiple (2) call log entries", res["diagnostics"]["reason"])
        self.assertEqual(res["inventory"], self.initial_inventory)

    def test_verifier_missing_or_two_records_rejected(self):
        """Fourth-senior P1: Verifier recording 0 or 2 calls during verification or reaudit fails closed."""
        gen_resp = _mock_gen_resp({
            "status": "answered",
            "claims": [{"claim_id": "C1", "text": "En 2070 la tasa disminuye al 53,7%.", "evidence_ids": ["E1"]}],
            "reason": "",
        })
        verif_resp = _mock_verif_resp([
            {"claim_id": "C1", "verdict": "supported", "reason": "ok", "evidence_verdicts": [{"evidence_id": "E1", "relation": "supports"}]}
        ])
        audit_resp = _mock_audit_resp([
            {"query_span": "¿Cuál es la tasa en 2050", "decision": "answered", "claim_ids": ["C1"], "reason_code": "covered"},
            {"query_span": " y en 2070?", "decision": "answered", "claim_ids": ["C2"], "reason_code": "covered"},
        ])

        # 1. Verification records 0 calls
        gen1 = _MockCorrectionGenerator([gen_resp])
        verif1 = _MockCorrectionVerifier([verif_resp], stage_calls_to_append={"verification": 0})
        res1 = correct_coverage_once(
            self.query, self.factual_report, self.initial_inventory, self.initial_coverage_audit,
            pages=self.pages, generator=gen1, verifier=verif1
        )
        self.assertFalse(res1["applied"])
        self.assertIn("did not append a new call log entry", res1["diagnostics"]["reason"])

        # 2. Verification records 2 calls
        gen2 = _MockCorrectionGenerator([gen_resp])
        verif2 = _MockCorrectionVerifier([verif_resp], stage_calls_to_append={"verification": 2})
        res2 = correct_coverage_once(
            self.query, self.factual_report, self.initial_inventory, self.initial_coverage_audit,
            pages=self.pages, generator=gen2, verifier=verif2
        )
        self.assertFalse(res2["applied"])
        self.assertIn("appended multiple (2) call log entries", res2["diagnostics"]["reason"])

        # 3. Reaudit records 0 calls
        gen3 = _MockCorrectionGenerator([gen_resp])
        verif3 = _MockCorrectionVerifier([verif_resp, audit_resp], stage_calls_to_append={"query_coverage": 0})
        res3 = correct_coverage_once(
            self.query, self.factual_report, self.initial_inventory, self.initial_coverage_audit,
            pages=self.pages, generator=gen3, verifier=verif3
        )
        self.assertFalse(res3["applied"])
        r_trace3 = res3["diagnostics"]["reaudit"]
        self.assertIsNotNone(r_trace3)
        self.assertIn("did not append a new call log entry", str(r_trace3["error"]))

        # 4. Reaudit records 2 calls
        gen4 = _MockCorrectionGenerator([gen_resp])
        verif4 = _MockCorrectionVerifier([verif_resp, audit_resp], stage_calls_to_append={"query_coverage": 2})
        res4 = correct_coverage_once(
            self.query, self.factual_report, self.initial_inventory, self.initial_coverage_audit,
            pages=self.pages, generator=gen4, verifier=verif4
        )
        self.assertFalse(res4["applied"])
        r_trace4 = res4["diagnostics"]["reaudit"]
        self.assertIsNotNone(r_trace4)
        self.assertIn("appended multiple (2) call log entries", str(r_trace4["error"]))

    def test_history_prefix_mutation_or_clear_rejected(self):
        """Fourth-senior P1: Mutating prior history prefix or clearing call log during call fails closed."""
        gen_resp = _mock_gen_resp({
            "status": "answered",
            "claims": [{"claim_id": "C1", "text": "En 2070 la tasa disminuye al 53,7%.", "evidence_ids": ["E1"]}],
            "reason": "",
        })
        verif_resp = _mock_verif_resp([
            {"claim_id": "C1", "verdict": "supported", "reason": "ok", "evidence_verdicts": [{"evidence_id": "E1", "relation": "supports"}]}
        ])
        prior_record = {"old_stage": "setup", "data": 123}

        # 1. Generator mutates prefix during call
        gen1 = _MockCorrectionGenerator([gen_resp], initial_calls=[copy.deepcopy(prior_record)], mutate_prefix_on_call=True)
        verif1 = _MockCorrectionVerifier([])
        res1 = correct_coverage_once(
            self.query, self.factual_report, self.initial_inventory, self.initial_coverage_audit,
            pages=self.pages, generator=gen1, verifier=verif1
        )
        self.assertFalse(res1["applied"])
        self.assertIn("history prefix was mutated or replaced", res1["diagnostics"]["reason"])

        # 2. Generator clears history during call
        gen2 = _MockCorrectionGenerator([gen_resp], initial_calls=[copy.deepcopy(prior_record)], clear_history_on_call=True, calls_to_append=0)
        verif2 = _MockCorrectionVerifier([])
        res2 = correct_coverage_once(
            self.query, self.factual_report, self.initial_inventory, self.initial_coverage_audit,
            pages=self.pages, generator=gen2, verifier=verif2
        )
        self.assertFalse(res2["applied"])
        self.assertIn("cleared or truncated", res2["diagnostics"]["reason"])

        # 3. Verifier mutates prefix during verification
        gen3 = _MockCorrectionGenerator([gen_resp])
        verif3 = _MockCorrectionVerifier([verif_resp], initial_calls=[copy.deepcopy(prior_record)], mutate_prefix_on_stage={"verification"})
        res3 = correct_coverage_once(
            self.query, self.factual_report, self.initial_inventory, self.initial_coverage_audit,
            pages=self.pages, generator=gen3, verifier=verif3
        )
        self.assertFalse(res3["applied"])
        self.assertIn("history prefix was mutated or replaced", res3["diagnostics"]["reason"])

        # 4. Verifier clears history during reaudit
        audit_resp = _mock_audit_resp([
            {"query_span": "¿Cuál es la tasa en 2050", "decision": "answered", "claim_ids": ["C1"], "reason_code": "covered"},
            {"query_span": " y en 2070?", "decision": "answered", "claim_ids": ["C2"], "reason_code": "covered"},
        ])
        gen4 = _MockCorrectionGenerator([gen_resp])
        verif4 = _MockCorrectionVerifier(
            [verif_resp, audit_resp],
            initial_calls=[copy.deepcopy(prior_record)],
            clear_history_on_stage={"query_coverage"},
            stage_calls_to_append={"query_coverage": 0},
        )
        res4 = correct_coverage_once(
            self.query, self.factual_report, self.initial_inventory, self.initial_coverage_audit,
            pages=self.pages, generator=gen4, verifier=verif4
        )
        self.assertFalse(res4["applied"])
        r_trace4 = res4["diagnostics"]["reaudit"]
        self.assertIsNotNone(r_trace4)
        self.assertIn("cleared or truncated", str(r_trace4["error"]))

    def test_positive_preexisting_history_with_exactly_one_new_per_stage_succeeds(self):
        """Fourth-senior P1 & P2: Preexisting call logs with exactly 1 fresh call per stage succeeds with applied=True."""
        gen_resp = _mock_gen_resp({
            "status": "answered",
            "claims": [{"claim_id": "C1", "text": "En 2070 la tasa disminuye al 53,7%.", "evidence_ids": ["E1"]}],
            "reason": "",
        })
        verif_resp = _mock_verif_resp([
            {"claim_id": "C1", "verdict": "supported", "reason": "ok", "evidence_verdicts": [{"evidence_id": "E1", "relation": "supports"}]}
        ])
        audit_resp = _mock_audit_resp([
            {"query_span": "¿Cuál es la tasa en 2050", "decision": "answered", "claim_ids": ["C1"], "reason_code": "covered"},
            {"query_span": " y en 2070?", "decision": "answered", "claim_ids": ["C2"], "reason_code": "covered"},
        ])
        gen_prior = [{"stage": "old_gen", "model": "old", "request": {"model": "old"}, "raw_response": {}}]
        verif_prior = [{"stage": "old_verif", "model": "old", "request": {"model": "old"}, "raw_response": {}}]

        gen = _MockCorrectionGenerator([gen_resp], initial_calls=gen_prior)
        verif = _MockCorrectionVerifier([verif_resp, audit_resp], initial_calls=verif_prior)

        res = correct_coverage_once(
            self.query, self.factual_report, self.initial_inventory, self.initial_coverage_audit,
            pages=self.pages, generator=gen, verifier=verif
        )
        self.assertTrue(res["attempted"])
        self.assertTrue(res["applied"])

        # Strict JSON serialization without allow_nan succeeds
        ser = json.dumps(res, allow_nan=False)
        self.assertIsInstance(ser, str)

        # Diagnostics contain ONLY the fresh calls from this invocation
        self.assertEqual(len(res["diagnostics"]["generation"]["client_calls"]), 1)
        self.assertEqual(res["diagnostics"]["generation"]["client_calls"][0]["schema"], res["diagnostics"]["generation"]["request"]["format"])
        self.assertEqual(len(res["diagnostics"]["verification"]["client_calls"]), 1)
        self.assertEqual(len(res["diagnostics"]["reaudit"]["calls"]), 1)

        # Full client call lists have prior history + fresh calls
        self.assertEqual(len(gen.calls), 2)
        self.assertEqual(len(verif.calls), 3)

    def test_oldhistory_with_nan_posinf_neginf_and_1_to_true_mutation_rejected(self):
        """Fifth-senior P2: Old history containing NaN, +Inf, or -Inf with 1->True mutation fails closed without permissive equality."""
        cases = [
            ("nan", float("nan")),
            ("pos_inf", float("inf")),
            ("neg_inf", float("-inf")),
        ]
        for name, bad_val in cases:
            with self.subTest(non_finite=name):
                gen_resp = _mock_gen_resp({
                    "status": "answered",
                    "claims": [{"claim_id": "C1", "text": "En 2070 la tasa disminuye al 53,7%.", "evidence_ids": ["E1"]}],
                    "reason": "",
                })
                # Old history item has an integer field 1 and a non-finite value
                old_record = {"flag": 1, "metric": bad_val}

                def _mutate_1_to_true(record: dict):
                    record["flag"] = True

                gen = _MockCorrectionGenerator(
                    [gen_resp],
                    initial_calls=[copy.deepcopy(old_record)],
                    mutate_prefix_on_call=_mutate_1_to_true,
                )
                verif = _MockCorrectionVerifier([])

                res = correct_coverage_once(
                    self.query, self.factual_report, self.initial_inventory, self.initial_coverage_audit,
                    pages=self.pages, generator=gen, verifier=verif
                )
                self.assertTrue(res["attempted"])
                self.assertFalse(res["applied"])
                self.assertEqual(res["inventory"], self.initial_inventory)
                self.assertEqual(res["coverage_audit"], self.initial_coverage_audit)

                # Strict JSON serialization without allow_nan succeeds
                ser = json.dumps(res, allow_nan=False)
                self.assertIsInstance(ser, str)

                # Client raw call log must remain immutable with original float non-finite
                client_metric = gen.calls[0]["metric"]
                if math.isnan(bad_val):
                    self.assertTrue(math.isnan(client_metric))
                elif bad_val > 0:
                    self.assertTrue(math.isinf(client_metric) and client_metric > 0)
                else:
                    self.assertTrue(math.isinf(client_metric) and client_metric < 0)
                self.assertIsInstance(client_metric, float)

    def test_unchanged_invalid_history_fails_closed_with_zero_calls(self):
        """Fifth-senior P2: Unchanged invalid history (NaN, +Inf, -Inf, non-list) fails closed with zero LLM calls."""
        cases = [
            ("nan", float("nan")),
            ("pos_inf", float("inf")),
            ("neg_inf", float("-inf")),
        ]
        for name, bad_val in cases:
            with self.subTest(generator_non_finite=name):
                gen = _MockCorrectionGenerator([], initial_calls=[{"old": 1, "stat": bad_val}])
                verif = _MockCorrectionVerifier([])
                res = correct_coverage_once(
                    self.query, self.factual_report, self.initial_inventory, self.initial_coverage_audit,
                    pages=self.pages, generator=gen, verifier=verif
                )
                self.assertTrue(res["attempted"])
                self.assertFalse(res["applied"])
                self.assertEqual(len(gen.calls), 1)  # Zero new calls made
                self.assertEqual(len(verif.calls), 0)
                self.assertIn("history invalid", res["diagnostics"]["reason"].lower())
                ser = json.dumps(res, allow_nan=False)
                self.assertIsInstance(ser, str)

        # Verifier pre-history invalid
        verif_bad = _MockCorrectionVerifier([], initial_calls=[{"old": 1, "stat": float("nan")}])
        res_v = correct_coverage_once(
            self.query, self.factual_report, self.initial_inventory, self.initial_coverage_audit,
            pages=self.pages, generator=_MockCorrectionGenerator([]), verifier=verif_bad
        )
        self.assertTrue(res_v["attempted"])
        self.assertFalse(res_v["applied"])
        self.assertEqual(len(verif_bad.calls), 1)  # Zero new calls made
        self.assertIn("history invalid", res_v["diagnostics"]["reason"].lower())

    def test_mutation_plus_one_appended_record_in_all_three_stages_retains_diagnostics(self):
        """Fifth-senior P2: Mutated prefix with 1 appended call in Call 1, 2, and 3 retains non-empty client_calls/request/raw+error."""
        gen_resp = _mock_gen_resp({
            "status": "answered",
            "claims": [{"claim_id": "C1", "text": "En 2070 la tasa disminuye al 53,7%.", "evidence_ids": ["E1"]}],
            "reason": "",
        })
        verif_resp = _mock_verif_resp([
            {"claim_id": "C1", "verdict": "supported", "reason": "ok", "evidence_verdicts": [{"evidence_id": "E1", "relation": "supports"}]}
        ])
        audit_resp = _mock_audit_resp([
            {"query_span": "¿Cuál es la tasa en 2050", "decision": "answered", "claim_ids": ["C1"], "reason_code": "covered"},
            {"query_span": " y en 2070?", "decision": "answered", "claim_ids": ["C2"], "reason_code": "covered"},
        ])

        # Stage 1: Generation prefix mutation + 1 call appended
        gen1 = _MockCorrectionGenerator([gen_resp], initial_calls=[{"prior": 1}], mutate_prefix_on_call=True)
        verif1 = _MockCorrectionVerifier([])
        res1 = correct_coverage_once(
            self.query, self.factual_report, self.initial_inventory, self.initial_coverage_audit,
            pages=self.pages, generator=gen1, verifier=verif1
        )
        self.assertFalse(res1["applied"])
        gen_diag1 = res1["diagnostics"]["generation"]
        self.assertEqual(len(gen_diag1["client_calls"]), 1)
        self.assertIsNotNone(gen_diag1["request"])
        self.assertIsNotNone(gen_diag1["raw_response"])
        self.assertIsNotNone(gen_diag1["error"])
        self.assertIn("history prefix was mutated or replaced", gen_diag1["error"])
        json.dumps(res1, allow_nan=False)

        # Stage 2: Verification prefix mutation + 1 call appended
        gen2 = _MockCorrectionGenerator([gen_resp])
        verif2 = _MockCorrectionVerifier(
            [verif_resp],
            initial_calls=[{"prior": 1}],
            mutate_prefix_on_stage={"verification": lambda c: c.__setitem__("mutated", True)},
        )
        res2 = correct_coverage_once(
            self.query, self.factual_report, self.initial_inventory, self.initial_coverage_audit,
            pages=self.pages, generator=gen2, verifier=verif2
        )
        self.assertFalse(res2["applied"])
        verif_diag2 = res2["diagnostics"]["verification"]
        self.assertEqual(len(verif_diag2["client_calls"]), 1)
        self.assertIsNotNone(verif_diag2["request"])
        self.assertIsNotNone(verif_diag2["raw_response"])
        self.assertIsNotNone(verif_diag2["error"])
        self.assertIn("history prefix was mutated or replaced", verif_diag2["error"])
        json.dumps(res2, allow_nan=False)

        # Stage 3: Reaudit prefix mutation + 1 call appended
        gen3 = _MockCorrectionGenerator([gen_resp])
        verif3 = _MockCorrectionVerifier(
            [verif_resp, audit_resp],
            initial_calls=[{"prior": 1}],
            mutate_prefix_on_stage={"query_coverage": lambda c: c.__setitem__("mutated", True)},
        )
        res3 = correct_coverage_once(
            self.query, self.factual_report, self.initial_inventory, self.initial_coverage_audit,
            pages=self.pages, generator=gen3, verifier=verif3
        )
        self.assertFalse(res3["applied"])
        r_diag3 = res3["diagnostics"]["reaudit"]
        self.assertEqual(len(r_diag3["calls"]), 1)
        self.assertEqual(len(r_diag3["client_calls"]), 1)
        self.assertIsNotNone(r_diag3["request"])
        self.assertIsNotNone(r_diag3["raw_response"])
        self.assertIsNotNone(r_diag3["error"])
        self.assertIn("history prefix was mutated or replaced", str(r_diag3["error"]))
        json.dumps(res3, allow_nan=False)

    def test_truncation_retains_observed_after_snapshot_not_fake_empty(self):
        """Senior sixth review P2: When history shrinks 2->1 without appending new call, observed_after_history is retained exactly, fresh call identification is indeterminate, client_calls/calls are empty, request is None, and initial inventory/audit fall back."""
        gen_resp = _mock_gen_resp({
            "status": "answered",
            "claims": [{"claim_id": "C1", "text": "En 2070 la tasa disminuye al 53,7%.", "evidence_ids": ["E1"]}],
            "reason": "",
        })
        verif_resp = _mock_verif_resp([
            {"claim_id": "C1", "verdict": "supported", "reason": "ok", "evidence_verdicts": [{"evidence_id": "E1", "relation": "supports"}]}
        ])
        audit_resp = _mock_audit_resp([
            {"query_span": "¿Cuál es la tasa en 2050", "decision": "answered", "claim_ids": ["C1"], "reason_code": "covered"},
            {"query_span": " y en 2070?", "decision": "answered", "claim_ids": ["C2"], "reason_code": "covered"},
        ])

        # 1. Generation stage truncation: 2 distinct prior records -> retain 1, append 0
        prior_gen_calls = [
            {"call_idx": 1, "stage": "old_gen_1", "request": {"model": "old-9b-1", "options": {"temperature": 0.1}}},
            {"call_idx": 2, "stage": "old_gen_2", "request": {"model": "old-9b-2", "options": {"temperature": 0.2}}},
        ]
        prior_gen_copy = copy.deepcopy(prior_gen_calls)
        gen1 = _MockCorrectionGenerator([gen_resp], initial_calls=prior_gen_calls, clear_history_on_call=1, calls_to_append=0)
        verif1 = _MockCorrectionVerifier([])

        res1 = correct_coverage_once(
            self.query, self.factual_report, self.initial_inventory, self.initial_coverage_audit,
            pages=self.pages, generator=gen1, verifier=verif1
        )
        self.assertTrue(res1["attempted"])
        self.assertFalse(res1["applied"])
        self.assertEqual(res1["inventory"], self.initial_inventory)
        self.assertEqual(res1["coverage_audit"], self.initial_coverage_audit)

        gen_diag1 = res1["diagnostics"]["generation"]
        self.assertEqual(gen_diag1["observed_after_history"], [prior_gen_copy[0]])
        self.assertEqual(gen_diag1["calls_before"], 2)
        self.assertEqual(gen_diag1["observed_history_count"], 1)
        self.assertEqual(gen_diag1["history_status"], "indeterminate_truncation")
        self.assertEqual(gen_diag1["client_calls"], [])  # Never attribute remaining old record!
        self.assertIsNone(gen_diag1["request"])  # No false attribution of old request!
        self.assertIsNotNone(gen_diag1["raw_response"])  # Observed actual response preserved!
        self.assertIsNotNone(gen_diag1["error"])
        self.assertIn("cleared or truncated", gen_diag1["error"])
        self.assertIn("indeterminate", gen_diag1["error"])
        ser1 = json.dumps(res1, allow_nan=False)
        self.assertIsInstance(ser1, str)

        # 2. Verification stage truncation: 2 distinct prior records -> retain 1, append 0
        prior_verif_calls = [
            {"call_idx": 10, "stage": "old_verif_1", "request": {"model": "old-27b-1", "options": {"temperature": 0.1}}},
            {"call_idx": 20, "stage": "old_verif_2", "request": {"model": "old-27b-2", "options": {"temperature": 0.2}}},
        ]
        prior_verif_copy = copy.deepcopy(prior_verif_calls)
        gen2 = _MockCorrectionGenerator([gen_resp])
        verif2 = _MockCorrectionVerifier(
            [verif_resp],
            initial_calls=prior_verif_calls,
            clear_history_on_stage={"verification": 1},
            stage_calls_to_append={"verification": 0},
        )

        res2 = correct_coverage_once(
            self.query, self.factual_report, self.initial_inventory, self.initial_coverage_audit,
            pages=self.pages, generator=gen2, verifier=verif2
        )
        self.assertTrue(res2["attempted"])
        self.assertFalse(res2["applied"])
        self.assertEqual(res2["inventory"], self.initial_inventory)
        self.assertEqual(res2["coverage_audit"], self.initial_coverage_audit)

        verif_diag2 = res2["diagnostics"]["verification"]
        self.assertEqual(verif_diag2["observed_after_history"], [prior_verif_copy[0]])
        self.assertEqual(verif_diag2["calls_before"], 2)
        self.assertEqual(verif_diag2["observed_history_count"], 1)
        self.assertEqual(verif_diag2["history_status"], "indeterminate_truncation")
        self.assertEqual(verif_diag2["client_calls"], [])  # Never attribute remaining old record!
        self.assertIsNone(verif_diag2["request"])  # No false attribution of old request!
        self.assertIsNotNone(verif_diag2["raw_response"])  # Observed actual response preserved!
        self.assertIsNotNone(verif_diag2["error"])
        self.assertIn("cleared or truncated", verif_diag2["error"])
        self.assertIn("indeterminate", verif_diag2["error"])
        ser2 = json.dumps(res2, allow_nan=False)
        self.assertIsInstance(ser2, str)

        # 3. Coverage reaudit stage truncation: 2 distinct prior records -> retain 1, append 0
        prior_reaudit_calls = [
            {"call_idx": 100, "stage": "old_reaudit_1", "request": {"model": "old-27b-1", "options": {"temperature": 0.1}}},
            {"call_idx": 200, "stage": "old_reaudit_2", "request": {"model": "old-27b-2", "options": {"temperature": 0.2}}},
        ]
        prior_reaudit_copy = copy.deepcopy(prior_reaudit_calls)
        gen3 = _MockCorrectionGenerator([gen_resp])
        # Note: verifier will execute verification (append 1 call -> 3 calls), then on query_coverage shrinks to 1 and appends 0
        verif3 = _MockCorrectionVerifier(
            [verif_resp, audit_resp],
            initial_calls=prior_reaudit_calls,
            clear_history_on_stage={"query_coverage": 1},
            stage_calls_to_append={"query_coverage": 0},
        )

        res3 = correct_coverage_once(
            self.query, self.factual_report, self.initial_inventory, self.initial_coverage_audit,
            pages=self.pages, generator=gen3, verifier=verif3
        )
        self.assertTrue(res3["attempted"])
        self.assertFalse(res3["applied"])
        self.assertEqual(res3["inventory"], self.initial_inventory)
        self.assertEqual(res3["coverage_audit"], self.initial_coverage_audit)

        r_diag3 = res3["diagnostics"]["reaudit"]
        self.assertEqual(r_diag3["observed_after_history"], [prior_reaudit_copy[0]])
        self.assertEqual(r_diag3["calls_before"], 3)  # 2 prior + 1 verification call
        self.assertEqual(r_diag3["observed_history_count"], 1)
        self.assertEqual(r_diag3["history_status"], "indeterminate_truncation")
        self.assertEqual(r_diag3["client_calls"], [])  # Never attribute remaining old record!
        self.assertEqual(r_diag3["calls"], [])  # Never attribute remaining old record!
        self.assertIsNone(r_diag3["request"])  # No false attribution of old request!
        self.assertIsNotNone(r_diag3["raw_response"])  # Observed actual response preserved!
        self.assertIsNotNone(r_diag3["error"])
        self.assertIn("cleared or truncated", str(r_diag3["error"]))
        self.assertIn("indeterminate", str(r_diag3["error"]))
        ser3 = json.dumps(res3, allow_nan=False)
        self.assertIsInstance(ser3, str)


class TestCorrectionRecordedGenerator(unittest.TestCase):
    """Pruebas unitarias para el adaptador CorrectionRecordedGenerator."""

    def test_default_properties_and_isolation(self):
        gen = CorrectionRecordedGenerator()
        self.assertTrue(gen.capture_chat_request)
        self.assertIsNone(gen.last_chat_request)

        # 4 opciones declaradas de DEFAULT_PARAMETERS
        p = gen.parameters
        self.assertEqual(
            p,
            {"temperature": 0, "seed": 42, "num_ctx": 8192, "num_predict": 1024},
        )
        # La copia devuelta está aislada de la política
        p["temperature"] = 0.9
        self.assertEqual(gen.parameters["temperature"], 0)
        self.assertEqual(DEFAULT_PARAMETERS["temperature"], 0)

    def test_chat_captures_actual_wire_request_and_effective_options(self):
        fake_opener = MagicMock()
        resp_mock = MagicMock()
        resp_mock.read.return_value = json.dumps({"model": "qwen3.5:9b", "message": {"content": "{}"}}).encode("utf-8")
        fake_opener.open.return_value.__enter__.return_value = resp_mock

        gen = CorrectionRecordedGenerator(opener=fake_opener)
        messages = [{"role": "user", "content": "consulta"}]
        schema = {"type": "object"}
        res = gen.chat(messages, schema)

        self.assertEqual(len(gen.calls), 1)
        call = gen.calls[0]
        self.assertIn("request", call)
        self.assertIn("effective_options", call)

        req = call["request"]
        self.assertEqual(
            set(req.keys()),
            {"model", "messages", "stream", "think", "keep_alive", "format", "options"},
        )
        self.assertEqual(req["keep_alive"], "5m")
        self.assertEqual(
            call["effective_options"],
            {"temperature": 0, "seed": 42, "num_ctx": 8192, "num_predict": 1024},
        )

        # Equivalencia byte a byte entre request capturado y cuerpo transmitido
        expected_bytes = json.dumps(req, ensure_ascii=False).encode("utf-8")
        actual_bytes = fake_opener.open.call_args[0][0].data
        self.assertEqual(actual_bytes, expected_bytes)

    def test_failure_behavior_and_no_stale_borrowing(self):
        fake_opener = MagicMock()
        fake_opener.open.side_effect = urllib.error.HTTPError(
            "http://127.0.0.1:11434/api/chat", 500, "Server Error", {}, None
        )
        gen = CorrectionRecordedGenerator(opener=fake_opener)

        # 1. Error durante HTTP captura request real en call log
        with self.assertRaises(RuntimeError):
            gen.chat([{"role": "user", "content": "fallo"}], {"type": "object"})

        self.assertEqual(len(gen.calls), 1)
        self.assertIn("error", gen.calls[0])
        self.assertIsNotNone(gen.calls[0]["request"])
        self.assertEqual(gen.calls[0]["request"]["options"]["temperature"], 0)

        # 2. Llamada posterior que falla en pre-validación resetea request y no toma prestado el anterior
        with self.assertRaises(ValueError):
            gen.chat("no_es_lista", {"type": "object"})

        self.assertIsNone(gen.last_chat_request)


class TestProbeCoverageCorrection(unittest.TestCase):
    """Pruebas unitarias offline para probe_coverage_correction."""

    def setUp(self):
        self.results_path = PROJECT_ROOT / "evaluation/quality_followup_2026-10-28/live_mixed_v3/results.json"
        self.protocol_path = PROJECT_ROOT / "evaluation/quality_followup_2026-10-28/live_mixed_v3/protocol.json"

    def test_validate_input_files_on_actual_real_inputs(self):
        """Verifica que validate_input_files valida los 11 casos reales contra el manifiesto sin llamadas neuronales."""
        (
            res_data,
            proto_data,
            pages_p,
            queries_p,
            pages_lines,
            results_cases,
            manifest_p,
            act_manifest_sha,
        ) = validate_input_files(self.results_path, self.protocol_path)

        self.assertEqual(len(results_cases), 11)
        self.assertEqual([c["id"] for c in results_cases], list(EXPECTED_CASE_IDS))
        self.assertGreater(len(pages_lines), 0)
        self.assertTrue(manifest_p.is_file())
        self.assertEqual(act_manifest_sha, PINNED_MANIFEST_SHA256)

    def test_probe_validation_rejections(self):
        (PROJECT_ROOT / "tmp").mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "tmp") as tmp:
            tmp_path = Path(tmp)

            # 1. Directorio de salida existente es rechazado
            existing_out = tmp_path / "already_exists"
            existing_out.mkdir()
            with self.assertRaises(FileExistsError):
                run_probe(self.results_path, self.protocol_path, existing_out)

            # 2. Archivo de resultados o protocolo inexistente es rechazado
            with self.assertRaises(FileNotFoundError):
                run_probe(tmp_path / "missing_results.json", self.protocol_path, tmp_path / "out1")
            with self.assertRaises(FileNotFoundError):
                run_probe(self.results_path, tmp_path / "missing_protocol.json", tmp_path / "out2")

            # 3. Manifiesto con hash alterado es rechazado inmediatamente
            with patch("experiments.probe_coverage_correction.MANIFEST_PIN", "0" * 64):
                with self.assertRaises(ValueError):
                    validate_input_files(self.results_path, self.protocol_path)

            # 4. Resultados alterados que no coinciden con el digest fijado en el manifiesto
            bad_results = tmp_path / "altered_results.json"
            res_content = json.loads(self.results_path.read_text(encoding="utf-8"))
            res_content["run_status"] = "altered_batch"
            bad_results.write_text(json.dumps(res_content), encoding="utf-8")
            with self.assertRaises(ValueError):
                validate_input_files(bad_results, self.protocol_path)

            # 5. Protocolo alterado que no coincide con el digest fijado en el manifiesto
            bad_proto = tmp_path / "altered_protocol.json"
            proto_content = json.loads(self.protocol_path.read_text(encoding="utf-8"))
            proto_content["server_version"] = "9.99.99"
            bad_proto.write_text(json.dumps(proto_content), encoding="utf-8")
            with self.assertRaises(ValueError):
                validate_input_files(self.results_path, bad_proto)

    def test_synthetic_fixture_with_patched_manifest_rejects_query_mismatch(self):
        """Verifica rechazo estricto si una consulta de caso no coincide con el roster original."""
        (PROJECT_ROOT / "tmp").mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "tmp") as tmp:
            tmp_path = Path(tmp)
            syn_results_path = tmp_path / "syn_results.json"
            syn_proto_path = tmp_path / "syn_protocol.json"
            syn_pages_path = tmp_path / "syn_pages.jsonl"
            syn_queries_path = tmp_path / "syn_queries.json"
            syn_manifest_path = tmp_path / "syn_manifest.json"

            # Copiar contenido real en archivos sintéticos
            syn_proto_path.write_text(self.protocol_path.read_text(encoding="utf-8"), encoding="utf-8")
            syn_pages_path.write_text((PROJECT_ROOT / "data/processed/airef_2025_pages.jsonl").read_text(encoding="utf-8"), encoding="utf-8")
            syn_queries_path.write_text((PROJECT_ROOT / "evaluation/scope_coverage_2026-10-26/full_inputs.json").read_text(encoding="utf-8"), encoding="utf-8")

            # Alterar la consulta de b01 en results
            res_dict = json.loads(self.results_path.read_text(encoding="utf-8"))
            res_dict["cases"][0]["query"] = "¿Consulta alterada no idéntica al roster?"
            syn_results_path.write_text(json.dumps(res_dict, ensure_ascii=False), encoding="utf-8")

            # Crear manifiesto sintético que mapee a estos archivos
            import hashlib
            f_sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
            syn_manifest = {
                "schema_version": "saved-correction-probe-inputs-v1",
                "experiment_kind": "correction_on_saved_inputs",
                "files_sha256": {
                    str(syn_results_path.relative_to(PROJECT_ROOT)).replace("\\", "/"): f_sha(syn_results_path),
                    str(syn_proto_path.relative_to(PROJECT_ROOT)).replace("\\", "/"): f_sha(syn_proto_path),
                    str(syn_pages_path.relative_to(PROJECT_ROOT)).replace("\\", "/"): f_sha(syn_pages_path),
                    str(syn_queries_path.relative_to(PROJECT_ROOT)).replace("\\", "/"): f_sha(syn_queries_path),
                },
                "roles": {
                    "results": str(syn_results_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                    "protocol": str(syn_proto_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                    "pages": str(syn_pages_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                    "queries": str(syn_queries_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                },
            }
            syn_manifest_path.write_text(json.dumps(syn_manifest), encoding="utf-8")
            syn_manifest_pin = f_sha(syn_manifest_path)

            with patch("experiments.probe_coverage_correction.MANIFEST_PATH", syn_manifest_path):
                with patch("experiments.probe_coverage_correction.MANIFEST_PIN", syn_manifest_pin):
                    with self.assertRaises(ValueError) as ctx:
                        validate_input_files(syn_results_path, syn_proto_path)
                    self.assertIn("query mismatch", str(ctx.exception).lower())

    def test_probe_offline_run_all_cases(self):
        class _MockProbeGenerator(CorrectionRecordedGenerator):
            def __init__(self):
                super().__init__()
                self.model = PINNED_9B_MODEL

            def get_digest(self):
                return PINNED_9B_DIGEST

            def chat(self, messages, response_schema):
                self.capture_chat_request = True
                call = {"messages": copy.deepcopy(messages), "schema": copy.deepcopy(response_schema)}
                self.calls.append(call)
                payload = {
                    "model": self.model,
                    "messages": messages,
                    "stream": False,
                    "think": False,
                    "keep_alive": "5m",
                    "format": response_schema,
                    "options": copy.deepcopy(PINNED_9B_PARAMETERS),
                }
                self.last_chat_request = copy.deepcopy(payload)
                call["request"] = copy.deepcopy(payload)
                call["effective_options"] = copy.deepcopy(payload["options"])
                raw = {
                    "model": self.model,
                    "done": True,
                    "done_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": json.dumps({"status": "insufficient_evidence", "claims": [], "reason": "No data"}),
                    },
                    "total_duration": 1000,
                    "load_duration": 10,
                    "prompt_eval_count": 10,
                    "prompt_eval_duration": 10,
                    "eval_count": 10,
                    "eval_duration": 10,
                }
                call["raw_response"] = raw
                return raw

        class _MockProbeVerifier:
            def __init__(self):
                self.model = PINNED_27B_MODEL
                self.expected_digest = PINNED_27B_DIGEST
                self.parameters = copy.deepcopy(PINNED_27B_PARAMETERS)
                self.calls = []

            def get_digest(self):
                return PINNED_27B_DIGEST

            def release_loaded(self, model):
                return False

            def chat(self, messages, schema, stage="query_coverage"):
                call = {
                    "stage": stage,
                    "messages": messages,
                    "request": {"model": self.model, "options": copy.deepcopy(self.parameters)},
                }
                self.calls.append(call)
                return {
                    "model": self.model,
                    "done": True,
                    "done_reason": "stop",
                    "message": {"role": "assistant", "content": json.dumps({"parts": []})},
                    "total_duration": 1000,
                    "load_duration": 10,
                    "prompt_eval_count": 10,
                    "prompt_eval_duration": 10,
                    "eval_count": 10,
                    "eval_duration": 10,
                }

        (PROJECT_ROOT / "tmp").mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "tmp") as tmp:
            out_dir = Path(tmp) / "probe_output"
            mock_gen = _MockProbeGenerator()
            mock_verif = _MockProbeVerifier()

            report = run_probe(
                self.results_path,
                self.protocol_path,
                out_dir,
                generator=mock_gen,
                verifier=mock_verif,
            )

            self.assertEqual(report["experiment_kind"], "correction_on_saved_inputs")
            self.assertEqual(report["run_status"], "completed")
            self.assertEqual(len(report["cases"]), 11)
            self.assertEqual([c["id"] for c in report["cases"]], list(EXPECTED_CASE_IDS))

            # Verificar que protocol.json y results.json existen en out_dir
            proto_file = out_dir / "protocol.json"
            res_file = out_dir / "results.json"
            self.assertTrue(proto_file.is_file())
            self.assertTrue(res_file.is_file())

            # Verificación de JSON estricto (allow_nan=False)
            saved_proto = json.loads(proto_file.read_text(encoding="utf-8"))
            saved_res = json.loads(res_file.read_text(encoding="utf-8"))
            self.assertEqual(saved_proto["experiment_kind"], "correction_on_saved_inputs")
            self.assertEqual(saved_res["schema_version"], "coverage-correction-probe-v1")
            self.assertTrue(saved_res["metadata"]["hash_integrity_verified"])
            self.assertEqual(saved_res["metadata"]["summary"]["total_cases"], 11)

            # Comprobar caso ya cubierto (b01): no ejecuta llamadas LLM
            b01_case = next(c for c in report["cases"] if c["id"] == "b01")
            self.assertFalse(b01_case["attempted"])
            self.assertFalse(b01_case["applied"])
            self.assertEqual(b01_case["call_counts"]["generator"], 0)
            self.assertEqual(b01_case["call_counts"]["verifier"], 0)



if __name__ == "__main__":
    unittest.main()
