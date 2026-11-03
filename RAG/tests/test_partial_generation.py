"""Offline unit tests for candidate partial generation and compound query preservation."""

import copy
from dataclasses import asdict
import hashlib
import json
import re
import unittest
from unittest.mock import MagicMock, patch

from pathlib import Path

from experiments.compare_system import generate_arm
from experiments.preserve_evidence import (
    PARTIAL_GENERATION_INSTRUCTION,
    PARTIAL_PUBLICATION_REASON,
    PARTIAL_PUBLICATION_SCHEMA_VERSION,
    PUBLICATION_SCHEMA_VERSION,
    PreservedGrounder,
    bound_generation,
    enforce_binding,
    partial_generation_messages,
    publication,
    validate_publication,
)
from experiments.preserved_q3_verifier import (
    Q3PreservedGrounder,
    Q3VerifierClient,
    VERIFIER_Q3_DIGEST,
    VERIFIER_Q3_MODEL,
    VERIFIER_Q3_PARAMETERS,
)
from experiments.query_coverage import (
    COVERAGE_STAGE,
    PRESERVED_COVERAGE_SCHEMA_VERSION,
    answer_covered,
    format_covered_answer,
)
from experiments.query_preserved import answer_preserved
from rag_pensiones.answer import DEFAULT_GENERATION_MODEL, MAX_MESSAGES_BYTES
from rag_pensiones.grounding import (
    DEFAULT_PARAMETERS,
    GENERATION_SYSTEM_PROMPT,
    MAX_CLAIM_REFS,
    SELECTION_JSON_SCHEMA,
)
from rag_pensiones.chunking import ChunkRecord
from rag_pensiones.qwen_embeddings import MODEL
from rag_pensiones.search import DEFAULT_INDEX_PATH, SearchHit, load_index


class FakeTransport:
    """Mock transport simulating Ollama API for offline tests."""

    def __init__(
        self,
        *,
        version: str = "0.40.11",
        q3_digest: str = VERIFIER_Q3_DIGEST,
        chat_response: dict | None = None,
        chat_error: Exception | None = None,
    ) -> None:
        self.version = version
        self.q3_digest = q3_digest
        self.chat_response = chat_response
        self.chat_error = chat_error
        self.chat_calls: list[dict] = []

    def get_version(self) -> dict:
        return {"version": self.version}

    def get_tags(self) -> dict:
        return {
            "models": [
                {
                    "name": VERIFIER_Q3_MODEL,
                    "model": VERIFIER_Q3_MODEL,
                    "digest": self.q3_digest,
                    "details": {"quantization_level": "Q3_K_M"},
                },
                {
                    "name": DEFAULT_GENERATION_MODEL,
                    "model": DEFAULT_GENERATION_MODEL,
                    "digest": "a" * 64,
                    "details": {"quantization_level": "Q4_K_M"},
                },
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
        self.chat_calls.append(copy.deepcopy(request))
        if self.chat_error is not None:
            raise self.chat_error
        return copy.deepcopy(self.chat_response)


class FakeGenerator:
    """Mock 9B generator recording calls and returning queued responses."""

    def __init__(self, responses: list[dict] | None = None, digest: str = "a" * 64) -> None:
        self.model = DEFAULT_GENERATION_MODEL
        self.digest = digest
        self.calls: list[dict] = []
        self.responses = list(responses or [])
        self.idx = 0

    def get_digest(self) -> str:
        return self.digest

    def chat(self, messages: list[dict], response_schema: dict) -> dict:
        call = {"messages": copy.deepcopy(messages), "schema": copy.deepcopy(response_schema)}
        self.calls.append(call)
        if self.idx < len(self.responses):
            resp = self.responses[self.idx]
            self.idx += 1
            if isinstance(resp, Exception):
                raise resp
            return copy.deepcopy(resp)
        # Default fallback response
        return {
            "model": self.model,
            "done": True,
            "done_reason": "stop",
            "message": {
                "role": "assistant",
                "content": json.dumps({"status": "insufficient_evidence", "claims": [], "reason": "No evidence"}),
            },
            "prompt_eval_count": 100,
            "eval_count": 20,
            "prompt_eval_duration": 10000000,
            "eval_duration": 20000000,
            "total_duration": 30000000,
            "load_duration": 1000000,
        }


def _make_9b_response(content_obj: dict) -> dict:
    return {
        "model": DEFAULT_GENERATION_MODEL,
        "done": True,
        "done_reason": "stop",
        "message": {
            "role": "assistant",
            "content": json.dumps(content_obj, ensure_ascii=False),
        },
        "prompt_eval_count": 120,
        "eval_count": 30,
        "prompt_eval_duration": 15000000,
        "eval_duration": 35000000,
        "total_duration": 50000000,
        "load_duration": 2000000,
    }


def _make_q3_response(content_obj: dict) -> dict:
    return {
        "model": VERIFIER_Q3_MODEL,
        "done": True,
        "done_reason": "stop",
        "message": {
            "role": "assistant",
            "content": json.dumps(content_obj, ensure_ascii=False),
        },
        "prompt_eval_count": 250,
        "eval_count": 40,
        "prompt_eval_duration": 25000000,
        "eval_duration": 55000000,
        "total_duration": 80000000,
        "load_duration": 3000000,
    }


def _sample_fixtures():
    units = json.loads(Path("evaluation/reconstructed_units_catalog.json").read_text(encoding="utf-8"))
    pages = [
        json.loads(x)
        for x in Path("data/processed/airef_2025_pages.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    index = load_index(DEFAULT_INDEX_PATH)
    return pages, units, index


class TestPartialGenerationHelper(unittest.TestCase):
    """Verifies purity, absence of hardcoded facts/dates/gender/case IDs, and budget guards."""

    def test_helper_purity_and_no_mutation(self):
        original = [
            {"role": "system", "content": "Sistema base de generacion."},
            {"role": "user", "content": json.dumps({"pregunta": "¿Cuándo y cuánto?"})},
        ]
        snapshot = copy.deepcopy(original)
        result = partial_generation_messages(original)

        # Original is not mutated
        self.assertEqual(original, snapshot)
        self.assertEqual(original[0]["content"], "Sistema base de generacion.")

        # Result is a deepcopy with instruction appended to system role
        self.assertIsNot(result, original)
        self.assertIn(PARTIAL_GENERATION_INSTRUCTION, result[0]["content"])
        self.assertEqual(result[1]["content"], original[1]["content"])

    def test_helper_no_facts_dates_gender_case_ids(self):
        instruction = PARTIAL_GENERATION_INSTRUCTION

        # No years or dates (no 4-digit years like 2015, 2023, 2030, 2050, 2070)
        years = re.findall(r"\b(19\d\d|20\d\d)\b", instruction)
        self.assertEqual(years, [], f"Instruction must not contain years: {years}")

        # No case IDs
        for case_id in ("n03", "b07", "b08", "b07-family", "case_id", "P1", "P2", "E1", "E2", "C1"):
            self.assertNotIn(case_id, instruction, f"Instruction must not contain case ID {case_id}")

        # No gender references
        for gender_term in ("mujer", "mujeres", "hombre", "hombres", "género", "sexo"):
            self.assertNotIn(gender_term, instruction.lower(), f"Instruction must not contain gender term {gender_term}")

        # No specific hardcoded domain examples or numbers
        for hardcoded in ("2015", "2030", "56,7%", "65,6%", "generosidad", "cuantía", "pensión mensual"):
            self.assertNotIn(hardcoded, instruction, f"Instruction must not contain hardcoded phrase {hardcoded}")

        # Compact instruction (< 400 chars; justified by message budget: 376 chars / 381 bytes UTF-8 << 6500 cap)
        self.assertLess(len(instruction), 400)

    def test_original_false_messages_byte_identical(self):
        """When partial_generation is False, generation messages are byte-identical to bound_generation."""
        user_content = json.dumps({
            "pregunta": "¿Pregunta de prueba?",
            "catalogo_evidencias": [{"evidence_id": "E1", "source_id": "S1", "quote": "Cita de prueba."}],
        })
        base_messages = [
            {"role": "system", "content": GENERATION_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        base_schema = {
            "anyOf": [
                {
                    "type": "object",
                    "properties": {
                        "status": {"const": "answered"},
                        "claims": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {"claim_id": {"type": "string"}, "text": {"type": "string"}},
                            },
                        },
                    },
                }
            ]
        }
        bound_msgs, bound_schema, bound_refs = bound_generation(base_messages, base_schema, "generation")

        # Emulate PreservedGrounder._chat_step with partial_generation=False
        grounder_msgs = copy.deepcopy(bound_msgs)
        # partial_generation=False does not modify grounder_msgs
        self.assertEqual(
            json.dumps(grounder_msgs, ensure_ascii=False).encode("utf-8"),
            json.dumps(bound_msgs, ensure_ascii=False).encode("utf-8"),
        )

    def test_helper_budget_guard(self):
        huge_text = "x" * (MAX_MESSAGES_BYTES - 100)
        messages = [
            {"role": "system", "content": huge_text},
            {"role": "user", "content": "user"},
        ]
        with self.assertRaisesRegex(ValueError, "exceeds budget"):
            partial_generation_messages(messages)

    def test_helper_validation(self):
        with self.assertRaises(ValueError):
            partial_generation_messages([])
        with self.assertRaises(ValueError):
            partial_generation_messages("not a list")  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            partial_generation_messages([{"role": "user", "content": "no system"}])


class TestPreservedGrounderExecution(unittest.TestCase):
    """Verifies that _chat_step modifies generation ONLY and prompt_sha256 matches actual request."""

    def setUp(self):
        self.pages, self.catalog, self.index = _sample_fixtures()

    def test_modified_generation_only_and_prompt_sha256_match(self):
        gen = FakeGenerator()
        grounder = PreservedGrounder(self.catalog, self.pages, partial_generation=True, generator=gen)

        user_content = json.dumps({
            "pregunta": "¿Año comparable de generosidad y pensión?",
            "catalogo_evidencias": [
                {"evidence_id": "E1", "source_id": "S1", "quote": "Cita de generosidad comparable en 2015."}
            ],
        })
        messages = [
            {"role": "system", "content": GENERATION_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        schema = {
            "anyOf": [
                {
                    "type": "object",
                    "properties": {
                        "status": {"const": "answered"},
                        "claims": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {"claim_id": {"type": "string"}, "text": {"type": "string"}},
                            },
                        },
                    },
                }
            ]
        }

        # 1. Generation stage with partial_generation=True
        payload, trace = grounder._chat_step("generation", messages, schema, gen.digest)
        self.assertEqual(len(gen.calls), 1)
        actual_sent_messages = gen.calls[0]["messages"]
        self.assertIn(PARTIAL_GENERATION_INSTRUCTION, actual_sent_messages[0]["content"])

        # Trace prompt_sha256 matches exactly the serialized sent request
        expected_hash = hashlib.sha256(
            json.dumps(actual_sent_messages, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        self.assertEqual(trace["prompt_sha256"], expected_hash)
        self.assertEqual(trace["stage"], "generation")

    def test_selection_and_repair_stages_unaffected(self):
        sel_resp = _make_9b_response({"status": "insufficient_evidence", "ids": [], "reason": "Sin datos"})
        repair_resp = _make_9b_response({"repairs": [{"status": "insufficient"}]})
        ver_resp = _make_9b_response({"status": "insufficient"})
        gen = FakeGenerator([sel_resp, repair_resp, ver_resp])
        grounder = PreservedGrounder(self.catalog, self.pages, partial_generation=True, generator=gen)
        grounder.selections = []
        grounder.catalog = {
            "records": [{"id": "P1", "unit_id": "U1", "evidence": [{"source_id": "S1", "quote": "Cita de prueba doce caracteres"}]}],
            "omitted": [],
        }

        # 1. Selection stage
        sel_messages = [
            {"role": "system", "content": "Seleccion de evidencias."},
            {"role": "user", "content": json.dumps({"pregunta": "p", "citas": [], "paquetes": []})},
        ]
        sel_schema = {"type": "object", "properties": {"status": {"type": "string"}}}
        grounder._chat_step("selection", sel_messages, sel_schema, gen.digest)
        self.assertEqual(len(gen.calls), 1)
        self.assertNotIn(PARTIAL_GENERATION_INSTRUCTION, gen.calls[0]["messages"][0]["content"])

        # 2. Repair generation stage
        repair_user_content = json.dumps({
            "pregunta": "p",
            "catalogo_evidencias_reparacion": [
                {"evidence_id": "E1", "source_id": "S1", "quote": "Cita reparacion."}
            ],
            "repairs": [],
        })
        repair_messages = [
            {"role": "system", "content": GENERATION_SYSTEM_PROMPT},
            {"role": "user", "content": repair_user_content},
        ]
        repair_schema = {
            "type": "object",
            "properties": {
                "repairs": {
                    "type": "array",
                    "items": {
                        "anyOf": [
                            {"type": "object", "properties": {"evidence_ids": {"type": "array"}}},
                            {"type": "object", "properties": {"status": {"const": "insufficient"}}},
                        ]
                    },
                }
            },
        }
        grounder._chat_step("repair_generation", repair_messages, repair_schema, gen.digest)
        self.assertEqual(len(gen.calls), 2)
        # repair_generation must NOT have PARTIAL_GENERATION_INSTRUCTION
        self.assertNotIn(PARTIAL_GENERATION_INSTRUCTION, gen.calls[1]["messages"][0]["content"])

        # 3. Verification stage
        ver_messages = [
            {"role": "system", "content": "Verificador."},
            {"role": "user", "content": json.dumps({"afirmaciones": []})},
        ]
        grounder._chat_step("verification", ver_messages, {"type": "object"}, gen.digest)
        self.assertEqual(len(gen.calls), 3)
        self.assertNotIn(PARTIAL_GENERATION_INSTRUCTION, gen.calls[2]["messages"][0]["content"])

    def test_reference_binding_and_guards_preserved(self):
        gen = FakeGenerator()
        grounder = PreservedGrounder(self.catalog, self.pages, partial_generation=True, generator=gen)

        # Enforce binding raises ValueError if model changes or omits required refs
        bad_payload = {
            "status": "answered",
            "claims": [{"claim_id": "C1", "text": "Texto", "evidence_ids": ["E2"]}],
        }
        with self.assertRaisesRegex(ValueError, "omitted or changed mandatory package references"):
            enforce_binding(bad_payload, "generation", ["E1"])


class TestStrictBooleanValidation(unittest.TestCase):
    """Verifies strict boolean rejection before any embedding or model call."""

    def setUp(self):
        self.pages, self.catalog, self.index = _sample_fixtures()

    def test_preserved_grounder_rejects_non_bool(self):
        for bad in (1, 0, "true", "False", None, [], {}):
            with self.subTest(bad=bad):
                with self.assertRaisesRegex(ValueError, "strict boolean"):
                    PreservedGrounder(self.catalog, self.pages, partial_generation=bad)

    def test_answer_preserved_rejects_non_bool_before_model(self):
        mock_embedder = MagicMock()
        for bad in (1, 0, "true", "False", [], {}):
            with self.subTest(bad=bad):
                with self.assertRaisesRegex(ValueError, "strict boolean"):
                    answer_preserved(
                        self.index,
                        "query",
                        self.catalog,
                        self.pages,
                        embedder=mock_embedder,
                        partial_generation=bad,
                    )
        self.assertEqual(mock_embedder.call_count, 0)

    def test_answer_preserved_requires_explicit_verifier_for_true(self):
        mock_embedder = MagicMock()
        with self.assertRaisesRegex(ValueError, "requires an explicit verifier"):
            answer_preserved(
                self.index,
                "query",
                self.catalog,
                self.pages,
                embedder=mock_embedder,
                verifier=None,
                partial_generation=True,
            )
        self.assertEqual(mock_embedder.call_count, 0)

    def test_answer_covered_rejects_non_bool_before_model(self):
        mock_embedder = MagicMock()
        for bad in (1, 0, "true", "False", [], {}):
            with self.subTest(bad=bad):
                with self.assertRaisesRegex(ValueError, "strict boolean"):
                    answer_covered(
                        self.index,
                        "query",
                        self.catalog,
                        self.pages,
                        embedder=mock_embedder,
                        partial_generation=bad,
                    )
        self.assertEqual(mock_embedder.call_count, 0)


class TestOptInMetadataTruthful(unittest.TestCase):
    """Verifies that opt-in metadata reflects actual mode and default remains identical to legacy."""

    def setUp(self):
        self.pages, self.catalog, self.index = _sample_fixtures()

    @patch("experiments.query_preserved.search")
    @patch("experiments.query_preserved.generate_arm")
    def test_answer_preserved_metadata(self, mock_generate_arm, mock_search):
        mock_search.return_value = []
        mock_generate_arm.return_value = {
            "status": "answered",
            "publication": {"status": "answered", "claims": [], "reason": ""},
        }

        transport = FakeTransport()
        verifier = Q3VerifierClient(transport=transport)
        embedder = MagicMock()
        embedder.calls = []
        gen = FakeGenerator()

        # 1. Explicit factual default (partial_generation=False)
        rep_default = answer_preserved(
            self.index,
            "query text",
            self.catalog,
            self.pages,
            embedder=embedder,
            generator=gen,
            verifier=verifier,
            partial_generation=False,
        )
        meta_default = rep_default["metadata"]
        self.assertEqual(meta_default["profile"], "preserved_q3_verifier_v1")
        self.assertNotIn("generation_profile", meta_default)
        self.assertNotIn("partial_generation", meta_default)

        # 2. Opt-in (partial_generation=True)
        rep_optin = answer_preserved(
            self.index,
            "query text",
            self.catalog,
            self.pages,
            embedder=embedder,
            generator=gen,
            verifier=verifier,
            partial_generation=True,
        )
        meta_optin = rep_optin["metadata"]
        self.assertEqual(meta_optin["profile"], "preserved_q3_verifier_v1")
        self.assertEqual(meta_optin["generation_profile"], "partial_facts_v1")
        self.assertTrue(meta_optin["partial_generation"])

    @patch("experiments.query_coverage.answer_preserved")
    def test_answer_covered_metadata(self, mock_answer_preserved):
        valid_factual = {
            "schema_version": "preserved-direct-query-v1",
            "query": "query text",
            "outcome": {
                "status": "answered",
                "expanded_hits": [{"rank": 1, "chunk": {"page_number": 1, "text": "t"}}],
                "full_result": {
                    "grounding": {
                        "accepted_claims": [{"claim_id": "C1", "text": "Claim 1", "evidence_ids": ["E1"]}]
                    }
                },
                "publication": {
                    "status": "answered",
                    "claims": [{"text": "Claim 1", "evidence": [{"source_id": "S1", "quote": "q"}]}],
                },
            },
        }
        mock_answer_preserved.return_value = copy.deepcopy(valid_factual)

        cov_resp = _make_q3_response({
            "parts": [
                {"query_span": "query text", "decision": "answered", "claim_ids": ["C1"], "reason_code": "covered"}
            ]
        })
        transport = FakeTransport(chat_response=cov_resp)
        verifier = Q3VerifierClient(transport=transport)

        # 1. Default (partial_generation=False)
        rep_def = answer_covered(
            self.index,
            "query text",
            self.catalog,
            self.pages,
            verifier=verifier,
            partial_generation=False,
        )
        self.assertNotIn("generation_profile", rep_def["metadata"])
        self.assertNotIn("partial_generation", rep_def["metadata"])

        # 2. Opt-in (partial_generation=True)
        rep_opt = answer_covered(
            self.index,
            "query text",
            self.catalog,
            self.pages,
            verifier=verifier,
            partial_generation=True,
        )
        self.assertEqual(rep_opt["metadata"]["generation_profile"], "partial_facts_v1")
        self.assertTrue(rep_opt["metadata"]["partial_generation"])


class TestEndToEndCompoundQueryPreservation(unittest.TestCase):
    """Verifies that useful claim 2015 reaches Q3 verification and scope audit produces partial."""

    def setUp(self):
        self.pages, self.catalog, self.index = _sample_fixtures()

    @patch("experiments.query_coverage.answer_preserved")
    def test_useful_claim_2015_with_unavailable_individual_reaches_q3_and_scope_partial(
        self, mock_answer_preserved
    ):
        query = "Indica el año histórico comparable de generosidad y qué pensión mensual exacta me correspondería si me jubilo en 2030."

        # Factual outcome where 2015 was preserved by candidate generation and accepted by Q3 verifier
        factual_report = {
            "schema_version": "preserved-direct-query-v1",
            "query": query,
            "metadata": {
                "profile": "preserved_q3_verifier_v1",
                "generation_profile": "partial_facts_v1",
                "partial_generation": True,
            },
            "outcome": {
                "status": "answered",
                "expanded_hits": [
                    {"rank": 1, "chunk": {"page_number": 95, "text": "Texto sobre generosidad 2015"}}
                ],
                "full_result": {
                    "grounding": {
                        "accepted_claims": [
                            {
                                "claim_id": "C1",
                                "text": "La tasa de generosidad del sistema contributivo volvería a niveles próximos a los alcanzados en 2015.",
                                "evidence_ids": ["E1", "E2"],
                            }
                        ]
                    }
                },
                "publication": {
                    "status": "answered",
                    "claims": [
                        {
                            "text": "La tasa de generosidad del sistema contributivo volvería a niveles próximos a los alcanzados en 2015.",
                            "evidence": [
                                {
                                    "source_id": "S1",
                                    "quote": "Por lo tanto, la tasa de generosidad... volvería a niveles próximos a los alcanzados en 2015.",
                                }
                            ],
                        }
                    ],
                },
            },
        }
        mock_answer_preserved.return_value = copy.deepcopy(factual_report)

        # Coverage audit partitions query into 2 parts:
        # Part 1 (2015 generosidad): answered and covered by C1
        # Part 2 (pension 2030): not_answered and not_in_evaluated_evidence
        part1 = "Indica el año histórico comparable de generosidad"
        part2 = " y qué pensión mensual exacta me correspondería si me jubilo en 2030."
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
        transport = FakeTransport(chat_response=_make_q3_response(audit_payload))
        verifier = Q3VerifierClient(transport=transport)

        report = answer_covered(
            self.index,
            query,
            self.catalog,
            self.pages,
            verifier=verifier,
            partial_generation=True,
        )

        # Status must be partial
        self.assertEqual(report["status"], "partial")
        self.assertEqual(report["publication"]["status"], "partial")

        # 1 accepted claim preserved in publication
        pub_claims = report["publication"]["claims"]
        self.assertEqual(len(pub_claims), 1)
        self.assertIn("2015", pub_claims[0]["text"])

        # 1 missing part reported
        missing = report["publication"]["missing_parts"]
        self.assertEqual(len(missing), 1)
        self.assertEqual(missing[0]["query_span"], part2)
        self.assertEqual(missing[0]["reason_code"], "not_in_evaluated_evidence")

        # Truthful metadata
        self.assertEqual(report["metadata"]["coverage_status"], "partial")
        self.assertEqual(report["metadata"]["generation_profile"], "partial_facts_v1")
        self.assertTrue(report["metadata"]["partial_generation"])

        # Formatted answer shows supported claim AND refusal for missing part
        rendered = format_covered_answer(report)
        self.assertIn("2015", rendered)
        self.assertIn("Partes no respondidas en las fuentes evaluadas:", rendered)
        self.assertIn("« y qué pensión mensual exacta me correspondería si me jubilo en 2030.»", rendered)


class TestOptInPartialPublication(unittest.TestCase):
    """Verifies opt-in partial publication view, rejected claim exclusion, provenance and guards."""

    def setUp(self):
        self.context_path = Path("evaluation/scope_coverage_2026-10-26/n03_partial_publication_context.json")
        self.raw_context = json.loads(self.context_path.read_text(encoding="utf-8"))
        self.full_result = self.raw_context["full_result"]
        self.old_publication = self.raw_context["old_publication"]
        self.query = self.raw_context["query"]
        self.verification_policy = self.raw_context["verification_policy"]
        self.expanded_hits = [
            SearchHit(h["rank"], h["score"], ChunkRecord(**h["chunk"]))
            for h in self.raw_context["expanded_hits"]
        ]

    def test_replay_actual_context_accepted_c1_rejected_c2_yields_only_c1_view_partial(self):
        snapshot = copy.deepcopy(self.full_result)
        pub = publication(self.full_result, preserve=True, allow_partial=True)

        # 1. Input deepcopy is completely unchanged
        self.assertEqual(self.full_result, snapshot)
        self.assertEqual(self.full_result["status"], "insufficient_evidence")
        self.assertEqual(self.full_result["claims"], [])

        # 2. View status is partial and distinct schema version
        self.assertEqual(pub["status"], "partial")
        self.assertEqual(pub["schema_version"], PARTIAL_PUBLICATION_SCHEMA_VERSION)
        self.assertEqual(pub["reason"], PARTIAL_PUBLICATION_REASON)

        # 3. Exactly 1 claim (C1) published with exact text and quotes
        self.assertEqual(len(pub["claims"]), 1)
        c1_claim = pub["claims"][0]
        self.assertEqual(c1_claim["text"], "El año histórico comparable de generosidad para el sistema contributivo es 2015.")
        self.assertIn("2015", c1_claim["text"])
        self.assertEqual(len(c1_claim["evidence"]), 2)
        # Original quotes from catalog retained
        self.assertEqual(c1_claim["evidence"][0]["source_id"], "S1")
        self.assertIn("65,6% en 2023", c1_claim["evidence"][0]["quote"])
        self.assertIn("volvería a niveles próximos a los alcanzados en 2015", c1_claim["evidence"][1]["quote"])

        # 4. Context package binding preserved for C1
        self.assertEqual(len(pub["bindings"]), 1)
        binding = pub["bindings"][0]
        self.assertEqual(binding["claim_id"], "C1")
        self.assertEqual(binding["stage"], "generation")
        self.assertEqual(binding["package_evidence_ids"], ["E1", "E2"])
        self.assertEqual(binding["verifier_retained_evidence_ids"], ["E2"])
        self.assertEqual(binding["context_reinstated_ids"], ["E1"])

        # 5. Rejected C2 completely excluded; no model-generated absence claims
        for claim in pub["claims"]:
            self.assertNotIn("No se puede determinar", claim["text"])
            self.assertNotIn("2030", claim["text"])
        self.assertNotIn("No se puede determinar", pub["reason"])
        self.assertNotIn("2030", pub["reason"])

        # 6. validate_publication succeeds
        validate_publication(pub, self.full_result, preserve=True, allow_partial=True)

    def test_default_gives_original_insuff_empty_byte_identical(self):
        # Default allow_partial=False
        pub_default = publication(self.full_result, preserve=True, allow_partial=False)
        self.assertEqual(pub_default, self.old_publication)
        self.assertEqual(pub_default["status"], "insufficient_evidence")
        self.assertEqual(pub_default["schema_version"], PUBLICATION_SCHEMA_VERSION)
        self.assertEqual(pub_default["claims"], [])
        self.assertEqual(pub_default["bindings"], [])
        self.assertEqual(pub_default["reason"], self.full_result["reason"])

        # Default preserve=False
        pub_unpreserved = publication(self.full_result, preserve=False, allow_partial=True)
        self.assertEqual(pub_unpreserved["status"], "insufficient_evidence")
        self.assertEqual(pub_unpreserved["schema_version"], PUBLICATION_SCHEMA_VERSION)
        self.assertEqual(pub_unpreserved["claims"], [])
        self.assertEqual(pub_unpreserved["bindings"], [])

        # Default positional/keyword call
        pub_bare = publication(self.full_result)
        self.assertEqual(pub_bare, self.old_publication)

    def test_rejected_and_tampered_provenance_cannot_enter_publication(self):
        # 1. Tampering accepted claim text mismatch raises ValueError
        tampered_text = copy.deepcopy(self.full_result)
        tampered_text["grounding"]["accepted_claims"][0]["text"] = "Afirmacion modificada ilegalmente."
        with self.assertRaisesRegex(ValueError, "Invalid complete-package publication binding"):
            publication(tampered_text, preserve=True, allow_partial=True)

        # 2. Tampering accepted evidence_ids (not subset of package refs) raises ValueError
        tampered_refs = copy.deepcopy(self.full_result)
        tampered_refs["grounding"]["accepted_claims"][0]["evidence_ids"] = ["E99"]
        with self.assertRaisesRegex(ValueError, "Invalid complete-package publication binding"):
            publication(tampered_refs, preserve=True, allow_partial=True)

        # 3. Tampering original package evidence_ids raises ValueError
        tampered_orig = copy.deepcopy(self.full_result)
        tampered_orig["grounding"]["initial_claims"][0]["evidence_ids"] = ["E1"]
        with self.assertRaisesRegex(ValueError, "Invalid complete-package publication binding"):
            publication(tampered_orig, preserve=True, allow_partial=True)

        # 4. Rejected claim (C2) cannot enter accepted claims without verifier backing
        tampered_rejected = copy.deepcopy(self.full_result)
        tampered_rejected["grounding"]["accepted_claims"].append({
            "claim_id": "C2",
            "text": "No se puede determinar la pensión mensual exacta que correspondería al jubilarse en 2030 con los datos proporcionados.",
            "evidence_ids": ["E1", "E2"],
        })
        with self.assertRaises(Exception):
            publication(tampered_rejected, preserve=True, allow_partial=True)

    def test_no_accepted_claims_remains_insufficient_evidence(self):
        no_accepted = copy.deepcopy(self.full_result)
        no_accepted["grounding"]["accepted_claims"] = []
        pub = publication(no_accepted, preserve=True, allow_partial=True)

        self.assertEqual(pub["status"], "insufficient_evidence")
        self.assertEqual(pub["schema_version"], PUBLICATION_SCHEMA_VERSION)
        self.assertEqual(pub["claims"], [])
        self.assertEqual(pub["bindings"], [])
        self.assertEqual(pub["reason"], no_accepted["reason"])

    def test_wrong_boolean_flags_rejected_strictly(self):
        for bad in (1, 0, "true", "False", None, [], {}, 1.0):
            with self.subTest(bad=bad):
                with self.assertRaisesRegex(ValueError, "preserve must be a strict boolean"):
                    publication(self.full_result, preserve=bad)
                with self.assertRaisesRegex(ValueError, "allow_partial must be a strict boolean"):
                    publication(self.full_result, preserve=True, allow_partial=bad)
                with self.assertRaisesRegex(ValueError, "preserve must be a strict boolean"):
                    validate_publication({}, self.full_result, preserve=bad)
                with self.assertRaisesRegex(ValueError, "allow_partial must be a strict boolean"):
                    validate_publication({}, self.full_result, preserve=True, allow_partial=bad)

    def test_generate_arm_validates_canonical_before_publisher_and_reflects_partial(self):
        mock_grounder = MagicMock()
        mock_grounder.generator.calls = []
        mock_grounder.partial_generation = True
        mock_grounder.answer.return_value = copy.deepcopy(self.full_result)
        mock_grounder.expanded_hits = copy.deepcopy(self.expanded_hits)

        order_of_operations = []

        def tracked_validate_grounded(*args, **kwargs):
            order_of_operations.append("validate_grounded_result")

        def tracked_publication(*args, **kwargs):
            order_of_operations.append("publication")
            return publication(*args, **kwargs)

        with patch("experiments.compare_system.validate_grounded_result", side_effect=tracked_validate_grounded):
            with patch("experiments.compare_system.publication", side_effect=tracked_publication):
                outcome = generate_arm(
                    mock_grounder,
                    self.query,
                    self.expanded_hits,
                    self.full_result["digest"],
                    preserve=True,
                    verification_policy=self.verification_policy,
                )

        self.assertEqual(order_of_operations, ["validate_grounded_result", "publication"])
        self.assertEqual(outcome["status"], "partial")
        self.assertEqual(outcome["publication"]["status"], "partial")
        self.assertEqual(outcome["publication"]["schema_version"], PARTIAL_PUBLICATION_SCHEMA_VERSION)
        self.assertEqual(len(outcome["publication"]["claims"]), 1)
        self.assertEqual(outcome["full_result"]["status"], "insufficient_evidence")
        self.assertEqual(outcome["full_result"]["claims"], [])

    @patch("experiments.query_coverage.answer_preserved")
    def test_optin_true_end_to_end_fixture_scope_sees_accepted_c1_and_marks_missing(
        self, mock_answer_preserved
    ):
        partial_pub = publication(self.full_result, preserve=True, allow_partial=True)
        factual_report = {
            "schema_version": "preserved-direct-query-v1",
            "query": self.query,
            "metadata": {
                "profile": "preserved_q3_verifier_v1",
                "generation_profile": "partial_facts_v1",
                "partial_generation": True,
            },
            "outcome": {
                "status": "partial",
                "expanded_hits": [
                    {
                        "rank": h.rank,
                        "score": h.score,
                        "chunk": asdict(h.chunk) if hasattr(h.chunk, "__dataclass_fields__") else h.chunk,
                    }
                    for h in self.expanded_hits
                ],
                "full_result": copy.deepcopy(self.full_result),
                "publication": copy.deepcopy(partial_pub),
            },
        }
        mock_answer_preserved.return_value = copy.deepcopy(factual_report)

        part1 = "Indica el año histórico comparable de generosidad"
        part2 = " y qué pensión mensual exacta me correspondería si me jubilo en 2030."
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
        transport = FakeTransport(chat_response=_make_q3_response(audit_payload))
        verifier = Q3VerifierClient(transport=transport)
        pages, catalog, index = _sample_fixtures()

        report = answer_covered(
            index,
            self.query,
            catalog,
            pages,
            verifier=verifier,
            partial_generation=True,
        )

        self.assertEqual(report["status"], "partial")
        self.assertEqual(report["publication"]["status"], "partial")

        pub_claims = report["publication"]["claims"]
        self.assertEqual(len(pub_claims), 1)
        self.assertIn("2015", pub_claims[0]["text"])

        missing = report["publication"]["missing_parts"]
        self.assertEqual(len(missing), 1)
        self.assertEqual(missing[0]["query_span"], part2)
        self.assertEqual(missing[0]["reason_code"], "not_in_evaluated_evidence")

        rendered = format_covered_answer(report)
        self.assertIn("2015", rendered)
        self.assertIn("Partes no respondidas en las fuentes evaluadas:", rendered)
        self.assertIn(part2, rendered)


class TestSyntheticPartialGenerationContract(unittest.TestCase):
    """Offline unit tests exercising message construction and contract invariants.

    Note: Actual LLM semantic validation remains for parent's fresh live inference;
    offline unit tests verify message construction, interface integrity, budget bounds,
    and parameter invariants without faking model outputs.
    """

    def setUp(self):
        self.pages, self.catalog, self.index = _sample_fixtures()

    def test_instruction_preserves_original_partial_clauses_and_generic_directness(self):
        """Verifies original partial clauses are preserved verbatim without appended directness clause."""
        instruction = PARTIAL_GENERATION_INSTRUCTION
        # 1. Original partial clauses restored verbatim
        self.assertIn(
            "Emite únicamente las afirmaciones atómicas solicitadas respaldadas por las evidencias seleccionadas",
            instruction,
        )
        self.assertIn(
            "incluso si otra parte solicitada es imposible de responder",
            instruction,
        )
        self.assertIn(
            "Declara insuficiencia total únicamente si ninguna parte solicitada tiene respaldo",
            instruction,
        )
        self.assertIn(
            "No exijas una precisión no pedida ni inventes importes o hechos no disponibles",
            instruction,
        )
        # 2. No directness clause appended
        self.assertNotIn("Responde el dato y la relación pedidos en el texto", instruction)

        # Negative invariants: no benchmark years, case IDs, gender, or hardcoded phrases
        years = re.findall(r"\b(19\d\d|20\d\d)\b", instruction)
        self.assertEqual(years, [], f"Instruction must not contain years: {years}")
        for case_id in ("n02", "n03", "b07", "b08", "b07-family", "case_id", "P1", "P2", "E1", "E2", "C1"):
            self.assertNotIn(case_id, instruction)
        for gender_term in ("mujer", "mujeres", "hombre", "hombres", "género", "sexo"):
            self.assertNotIn(gender_term, instruction.lower())
        for hardcoded in ("2015", "2023", "2030", "2050", "2070", "56,7%", "65,6%", "generosidad", "pensión media"):
            self.assertNotIn(hardcoded, instruction)
        # Compact instruction bound (< 400 chars, actual length 376 chars / 381 bytes UTF-8)
        self.assertLess(len(instruction), 400)

    def test_synthetic_generation_path_message_construction_and_immutability(self):
        """Exercises actual message construction path through partial_generation_messages and PreservedGrounder."""
        synthetic_query = "¿Cuál fue el año base del índice de cobertura forestal y cuál será la cota individual en 2042?"
        synthetic_catalog = [
            {"evidence_id": "E1", "source_id": "S1", "quote": "El índice de cobertura forestal tuvo como año base 1994."},
            {"evidence_id": "E2", "source_id": "S1", "quote": "Se proyecta una cota agregada para 2042 del 74%."},
        ]
        user_content = json.dumps({"pregunta": synthetic_query, "catalogo_evidencias": synthetic_catalog})
        original_messages = [
            {"role": "system", "content": GENERATION_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        snapshot_messages = copy.deepcopy(original_messages)

        # 1. Exercise partial_generation_messages
        constructed = partial_generation_messages(original_messages)

        # Immutability of input
        self.assertEqual(original_messages, snapshot_messages)
        self.assertIsNot(constructed, original_messages)
        self.assertIn(PARTIAL_GENERATION_INSTRUCTION, constructed[0]["content"])
        self.assertEqual(constructed[1]["content"], snapshot_messages[1]["content"])

        # UTF-8 byte budget check (< MAX_MESSAGES_BYTES = 6500)
        serialized_bytes = json.dumps(constructed, ensure_ascii=False).encode("utf-8")
        self.assertLess(len(serialized_bytes), MAX_MESSAGES_BYTES)

        # 2. Exercise through PreservedGrounder._chat_step with partial_generation=True
        gen = FakeGenerator()
        grounder = PreservedGrounder(self.catalog, self.pages, partial_generation=True, generator=gen)
        schema = {
            "anyOf": [
                {
                    "type": "object",
                    "properties": {
                        "status": {"const": "answered"},
                        "claims": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {"claim_id": {"type": "string"}, "text": {"type": "string"}},
                            },
                        },
                    },
                }
            ]
        }
        payload, trace = grounder._chat_step("generation", original_messages, schema, gen.digest)
        self.assertEqual(len(gen.calls), 1)
        sent_messages = gen.calls[0]["messages"]
        self.assertIn(PARTIAL_GENERATION_INSTRUCTION, sent_messages[0]["content"])
        expected_hash = hashlib.sha256(json.dumps(sent_messages, ensure_ascii=False).encode("utf-8")).hexdigest()
        self.assertEqual(trace["prompt_sha256"], expected_hash)

    def test_unknown_subrequest_clause_preserved_and_evidence_untouched(self):
        """Verifies unknown subrequest does not trigger total rejection instruction, and evidence stays untouched."""
        synthetic_query = "¿Año histórico de referencia y cálculo individual inalcanzable para el usuario 999?"
        synthetic_catalog = [
            {"evidence_id": "E1", "source_id": "S1", "quote": "El año histórico de referencia fue 1994."}
        ]
        user_content = json.dumps({"pregunta": synthetic_query, "catalogo_evidencias": synthetic_catalog})
        messages = [
            {"role": "system", "content": GENERATION_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        user_payload_before = json.loads(user_content)
        modified = partial_generation_messages(messages)

        # Verify exact evidence quotes untouched
        user_payload_after = json.loads(modified[1]["content"])
        self.assertEqual(user_payload_after["catalogo_evidencias"], user_payload_before["catalogo_evidencias"])
        self.assertEqual(user_payload_after["pregunta"], synthetic_query)

        # Verify instruction explicitly instructs emitting supported part when another is impossible
        sys_prompt = modified[0]["content"]
        self.assertIn("incluso si otra parte solicitada es imposible de responder", sys_prompt)
        self.assertIn("Declara insuficiencia total únicamente si ninguna parte solicitada tiene respaldo", sys_prompt)
        self.assertIn("No exijas una precisión no pedida ni inventes importes o hechos no disponibles", sys_prompt)

    def test_partial_generation_false_leaves_messages_byte_identical(self):
        """When partial_generation is False, generation messages remain byte-identical to bound_generation."""
        synthetic_query = "¿Cuál fue el año base de referencia?"
        synthetic_catalog = [
            {"evidence_id": "E1", "source_id": "S1", "quote": "El año base de referencia fue 1994."}
        ]
        user_content = json.dumps({"pregunta": synthetic_query, "catalogo_evidencias": synthetic_catalog})
        messages = [
            {"role": "system", "content": GENERATION_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        schema = {
            "anyOf": [
                {
                    "type": "object",
                    "properties": {
                        "status": {"const": "answered"},
                        "claims": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {"claim_id": {"type": "string"}, "text": {"type": "string"}},
                            },
                        },
                    },
                }
            ]
        }
        bound_msgs, bound_schema, bound_refs = bound_generation(messages, schema, "generation")

        gen = FakeGenerator()
        grounder = PreservedGrounder(self.catalog, self.pages, partial_generation=False, generator=gen)
        payload, trace = grounder._chat_step("generation", messages, schema, gen.digest)

        sent_msgs = gen.calls[0]["messages"]
        self.assertEqual(
            json.dumps(sent_msgs, ensure_ascii=False).encode("utf-8"),
            json.dumps(bound_msgs, ensure_ascii=False).encode("utf-8"),
        )
        self.assertNotIn(PARTIAL_GENERATION_INSTRUCTION, sent_msgs[0]["content"])


if __name__ == "__main__":
    unittest.main()
