"""Exhaustive offline tests for measured mixed vs single 27B text comparison.

Tests Single27PreservedGrounder routes, truthful 27B canonical validation without masquerade,
no 27B unloads during text role switches, sampling options equivalence, cold state lifecycles,
pre-network validation guards, atomic checkpoint error preservation, and stats exclusion.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import re
import tempfile
import time
from typing import Any
import unittest
import unittest.mock

from experiments.compare_mixed_single27 import (
    ALLOWED_EXPERIMENT_MODELS,
    CANONICAL_ROLE_PARAMETERS,
    DEFAULT_INDEX_REL,
    DEFAULT_MODEL_SHOW_REL,
    DEFAULT_OLLAMA_DEFAULTS_REL,
    DEFAULT_PAGES_REL,
    DEFAULT_QUERIES_REL,
    DEFAULT_UNITS_REL,
    EFFECTIVE_GENERATION_OPTIONS,
    EMBEDDING_4B_MODEL,
    GENERATOR_9B_MODEL,
    PINNED_9B_DIGEST,
    PINNED_27B_DIGEST,
    PINNED_EMBEDDING_DIGEST,
    PROTOCOL_SCHEMA_VERSION,
    REQUIRED_INPUT_KEYS,
    REQUIRED_QUERY_ROSTER,
    RUN_RESULTS_SCHEMA_VERSION,
    SERVER_VERSION,
    TEXT_27B_MODEL,
    ComparisonEmbedder,
    ComparisonGenerator,
    MixedSingle27Transport,
    RecordingTransport,
    Single27PreservedGrounder,
    build_expected_schedule,
    canonical_file_sha256,
    compute_role_stats,
    ensure_cold_state,
    execute_attempt,
    get_source_closure,
    prepare,
    run,
    validate_queries_payload,
)
from experiments.compare_system import generate_arm
from experiments.preserve_evidence import (
    PreservedGrounder,
    publication,
    validate_catalog,
    validate_publication,
)
from experiments.preserved_q3_verifier import (
    Q3PreservedGrounder,
    Q3VerifierClient,
)
from experiments.query_coverage import (
    COVERAGE_STAGE,
    audit_coverage,
    build_coverage_publication,
    claim_inventory,
    format_covered_answer,
)
from rag_pensiones.answer import (
    ALLOWED_MODELS,
    DEFAULT_GENERATION_MODEL,
    validate_answer,
)
from rag_pensiones.chunking import ChunkRecord
from rag_pensiones.grounding import (
    DEFAULT_PARAMETERS,
    MAX_MESSAGES_BYTES,
    VERIFIER_Q3_DIGEST,
    VERIFIER_Q3_MODEL,
    VERIFIER_Q3_PARAMETERS,
    validate_grounded_result,
)
from rag_pensiones.ingest import PROJECT_ROOT
from rag_pensiones.qwen_embeddings import (
    DIMENSION,
    QUERY_PREFIX,
    SCHEMA,
)
from rag_pensiones.search import SearchHit, validate_index


class FakeTransport:
    """Mock Ollama transport for unit testing without live Ollama service."""

    def __init__(
        self,
        *,
        version: str = SERVER_VERSION,
        resident_models: list[str] | None = None,
        chat_responses: list[dict[str, Any]] | None = None,
        chat_error: Exception | None = None,
    ) -> None:
        self.version = version
        self.resident_models = list(resident_models or [])
        self.unloaded_models: list[str] = []
        self.chat_responses = list(chat_responses or [])
        self.chat_error = chat_error
        self.chat_calls: list[dict[str, Any]] = []
        self.embed_calls: list[dict[str, Any]] = []
        self.unload_calls: list[str] = []

    def get_version(self) -> dict[str, Any]:
        return {"version": self.version}

    def get_tags(self) -> dict[str, Any]:
        return {
            "models": [
                {
                    "name": GENERATOR_9B_MODEL,
                    "model": GENERATOR_9B_MODEL,
                    "digest": PINNED_9B_DIGEST,
                },
                {
                    "name": TEXT_27B_MODEL,
                    "model": TEXT_27B_MODEL,
                    "digest": PINNED_27B_DIGEST,
                    "details": {"quantization_level": "Q3_K_M"},
                },
                {
                    "name": EMBEDDING_4B_MODEL,
                    "model": EMBEDDING_4B_MODEL,
                    "digest": PINNED_EMBEDDING_DIGEST,
                },
            ]
        }

    def get_ps(self) -> dict[str, Any]:
        return {
            "models": [
                {
                    "name": m,
                    "model": m,
                    "size": 14000000000,
                    "size_vram": 12000000000,
                    "context_length": 8192,
                }
                for m in self.resident_models
            ]
        }

    def show(self, model: str) -> dict[str, Any]:
        if model == GENERATOR_9B_MODEL:
            return {
                "parameters": "presence_penalty 1.5\ntop_k 20\ntop_p 0.95\n",
                "details": {"quantization_level": "Q4_K_M"},
            }
        elif model == TEXT_27B_MODEL:
            return {
                "parameters": "",
                "details": {"quantization_level": "Q3_K_M"},
            }
        elif model == EMBEDDING_4B_MODEL:
            return {"parameters": "", "details": {"quantization_level": "Q4_K_M"}}
        raise ValueError(f"Unknown model {model}")

    def unload(self, model: str) -> dict[str, Any]:
        self.unload_calls.append(model)
        self.unloaded_models.append(model)
        if model in self.resident_models:
            self.resident_models.remove(model)
        return {"status": "unloaded"}

    def embed(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.embed_calls.append(copy.deepcopy(payload))
        texts = payload.get("input", [])
        return {
            "model": EMBEDDING_4B_MODEL,
            "embeddings": [[0.01] * 2560 for _ in texts],
        }

    def chat(self, request: dict[str, Any]) -> dict[str, Any]:
        self.chat_calls.append(copy.deepcopy(request))
        req_model = request.get("model", TEXT_27B_MODEL)
        if req_model not in self.resident_models:
            self.resident_models.append(req_model)
        if self.chat_error is not None:
            raise self.chat_error
        if self.chat_responses:
            resp = self.chat_responses.pop(0)
            if isinstance(resp, Exception):
                raise resp
            return copy.deepcopy(resp)

        # Default fallback response based on requested format
        fmt = request.get("format", {})
        content_obj: dict[str, Any] = {}

        is_selection = ("properties" in fmt and "ids" in fmt["properties"]) or any(
            "properties" in b and "ids" in b["properties"] for b in fmt.get("anyOf", [])
        )
        is_generation = ("properties" in fmt and "claims" in fmt["properties"]) or any(
            "properties" in b and "claims" in b["properties"] for b in fmt.get("anyOf", [])
        )
        is_verification = ("properties" in fmt and "verdicts" in fmt["properties"]) or any(
            "properties" in b and "verdicts" in b["properties"] for b in fmt.get("anyOf", [])
        )
        is_coverage = ("properties" in fmt and "parts" in fmt["properties"]) or any(
            "properties" in b and "parts" in b["properties"] for b in fmt.get("anyOf", [])
        )
        is_repair_generation = ("properties" in fmt and "repairs" in fmt["properties"])

        user_msg = next((m for m in request.get("messages", []) if m.get("role") == "user"), None)
        user_data: dict[str, Any] = {}
        if user_msg and isinstance(user_msg.get("content"), str):
            try:
                user_data = json.loads(user_msg["content"])
            except Exception:
                user_data = {}

        if is_selection:
            content_obj = {"status": "selected", "ids": ["P1"], "reason": ""}
        elif is_generation or is_repair_generation:
            claim_refs = None
            for branch in fmt.get("anyOf", []):
                if "properties" in branch and "claims" in branch["properties"]:
                    items_prop = branch["properties"]["claims"].get("items", {}).get("properties", {})
                    eid_prop = items_prop.get("evidence_ids", {})
                    if "const" in eid_prop and isinstance(eid_prop["const"], list):
                        claim_refs = list(eid_prop["const"])
                    break
            if claim_refs is None and "properties" in fmt and "claims" in fmt["properties"]:
                items_prop = fmt["properties"]["claims"].get("items", {}).get("properties", {})
                eid_prop = items_prop.get("evidence_ids", {})
                if "const" in eid_prop and isinstance(eid_prop["const"], list):
                    claim_refs = list(eid_prop["const"])
            if claim_refs is None and "catalogo_evidencias" in user_data and isinstance(user_data["catalogo_evidencias"], list):
                claim_refs = [
                    e["evidence_id"] for e in user_data["catalogo_evidencias"]
                    if isinstance(e, dict) and "evidence_id" in e
                ]
            if not claim_refs:
                claim_refs = ["E1"]

            claim_text = "En el ejercicio 2015 el saldo presupuestario del sistema público registró equilibrio presupuestario."
            if is_repair_generation:
                content_obj = {
                    "repairs": [
                        {
                            "claim_id": "C1",
                            "status": "answered",
                            "text": claim_text,
                            "evidence_ids": claim_refs,
                            "reason": "",
                        }
                    ]
                }
            else:
                content_obj = {
                    "status": "answered",
                    "claims": [
                        {
                            "claim_id": "C1",
                            "text": claim_text,
                            "evidence_ids": claim_refs,
                        }
                    ],
                    "reason": "",
                }
        elif is_verification:
            verdicts = []
            afirmaciones = user_data.get("afirmaciones", [])
            for c in afirmaciones:
                cid = c.get("claim_id", "C1")
                eids = c.get("evidence_ids", ["E1"])
                verdicts.append({
                    "claim_id": cid,
                    "verdict": "supported",
                    "reason": "Respaldo directo comprobado.",
                    "evidence_verdicts": [{"evidence_id": eid, "relation": "supports"} for eid in eids],
                })
            if not verdicts:
                verdicts = [
                    {
                        "claim_id": "C1",
                        "verdict": "supported",
                        "reason": "Respaldo directo comprobado.",
                        "evidence_verdicts": [{"evidence_id": "E1", "relation": "supports"}],
                    }
                ]
            content_obj = {"verdicts": verdicts}
        elif is_coverage:
            query_text = (
                user_data.get("consulta")
                or user_data.get("consulta_original")
                or user_data.get("query")
            )
            if not query_text and user_msg and isinstance(user_msg.get("content"), str):
                m_match = re.search(
                    r'["\']?(?:consulta|query)(?:_original)?["\']?\s*[:=]\s*["\']([^"\']+)["\']',
                    user_msg["content"],
                    re.IGNORECASE,
                )
                if m_match:
                    query_text = m_match.group(1).strip()
            if not query_text:
                query_text = "Q1"
            accepted_cids = [
                row["claim_id"] for row in user_data.get("afirmaciones_aceptadas", [])
                if isinstance(row, dict) and "claim_id" in row
            ]
            content_obj = {
                "parts": [
                    {
                        "query_span": query_text,
                        "decision": "answered",
                        "claim_ids": accepted_cids or ["C1"],
                        "reason_code": "covered",
                    }
                ]
            }

        return {
            "model": req_model,
            "done": True,
            "done_reason": "stop",
            "message": {"role": "assistant", "content": json.dumps(content_obj)},
            "prompt_eval_count": 150,
            "eval_count": 45,
            "eval_duration": 450000000,
            "total_duration": 1000000000,
            "load_duration": 10000000,
            "prompt_eval_duration": 50000000,
        }


def make_raw_chat_response(model: str, content_obj: dict[str, Any], prompt_eval: int = 150, eval_count: int = 45) -> dict[str, Any]:
    return {
        "model": model,
        "done": True,
        "done_reason": "stop",
        "message": {"role": "assistant", "content": json.dumps(content_obj)},
        "prompt_eval_count": prompt_eval,
        "eval_count": eval_count,
        "eval_duration": 400000000,
        "total_duration": 900000000,
        "load_duration": 10000000,
        "prompt_eval_duration": 50000000,
    }


SYNTHETIC_DOC_ID = "doc_synthetic_2015"
SYNTHETIC_SOURCE_FILE = "synthetic_pensiones_2015.pdf"
SYNTHETIC_PUBLISHED_AT = "2015-01-15"
SYNTHETIC_PAGE_1_TEXT = "Portada y resumen ejecutivo del estudio sintético de 2015. Contiene parámetros de referencia."
SYNTHETIC_PAGE_2_TEXT = "En el ejercicio 2015 el saldo presupuestario del sistema público registró equilibrio presupuestario."


def load_test_fixtures():
    """Generates a self-contained synthetic catalog, pages, and valid search hit without reading real corpus."""
    pages = [
        {
            "doc_id": SYNTHETIC_DOC_ID,
            "page_number": 1,
            "source_file": SYNTHETIC_SOURCE_FILE,
            "published_at": SYNTHETIC_PUBLISHED_AT,
            "text": SYNTHETIC_PAGE_1_TEXT,
        },
        {
            "doc_id": SYNTHETIC_DOC_ID,
            "page_number": 2,
            "source_file": SYNTHETIC_SOURCE_FILE,
            "published_at": SYNTHETIC_PUBLISHED_AT,
            "text": SYNTHETIC_PAGE_2_TEXT,
        },
    ]
    p = pages[1]
    units = {
        "schema_version": "reconstructed-units-catalog-v1",
        "units": [
            {
                "id": "U1",
                "spans": [
                    {
                        "doc_id": p["doc_id"],
                        "source_file": p["source_file"],
                        "published_at": p["published_at"],
                        "page_number": p["page_number"],
                        "start_char": 0,
                        "end_char": len(p["text"]),
                        "quote": p["text"],
                    }
                ],
            }
        ],
    }
    chunk = ChunkRecord(
        chunk_id="chunk_syn_1",
        doc_id=p["doc_id"],
        source_file=p["source_file"],
        published_at=p["published_at"],
        page_number=p["page_number"],
        start_char=0,
        end_char=len(p["text"]),
        text=p["text"],
        strategy="page-local-v1:max_chars=1000:overlap_chars=150",
    )
    hit = SearchHit(1, 0.95, chunk)
    return units, pages, [hit]


def make_synthetic_index(pages: list[dict], dim: int = 2560) -> dict[str, Any]:
    """Builds a small in-memory synthetic index strictly matching synthetic fixtures."""
    p = pages[1]
    text = p["text"]
    norm_val = 1.0 / math.sqrt(dim)
    return {
        "schema_version": SCHEMA,
        "model": EMBEDDING_4B_MODEL,
        "model_digest": PINNED_EMBEDDING_DIGEST,
        "dimension": dim,
        "document_prefix": "",
        "query_prefix": QUERY_PREFIX,
        "base_index_hash": "0" * 64,
        "chunks": [
            {
                "chunk_id": "chunk_syn_1",
                "doc_id": p["doc_id"],
                "source_file": p["source_file"],
                "published_at": p["published_at"],
                "page_number": p["page_number"],
                "start_char": 0,
                "end_char": len(text),
                "text": text,
                "strategy": "page-local-v1:max_chars=1000:overlap_chars=150",
                "embedding": [norm_val] * dim,
            }
        ],
    }


def build_test_protocol(
    tmpdir: str | Path,
    *,
    queries_path: Path | None = None,
    drift_input_key: str | None = None,
    omit_source_file: str | None = None,
    tamper_schedule: str | None = None,
) -> Path:
    """Builds a valid frozen protocol fixture with hermetic synthetic inputs for offline testing."""
    p_tmp = Path(tmpdir)
    proto_file = p_tmp / "protocol.json"
    closure = get_source_closure()
    if omit_source_file:
        closure.pop(omit_source_file, None)

    units, pages, _ = load_test_fixtures()
    syn_index = make_synthetic_index(pages)

    p_idx = p_tmp / "syn_index.json"
    p_idx.write_text(json.dumps(syn_index, ensure_ascii=False, indent=2), encoding="utf-8")

    p_unt = p_tmp / "syn_units.json"
    p_unt.write_text(json.dumps(units, ensure_ascii=False, indent=2), encoding="utf-8")

    p_pgs = p_tmp / "syn_pages.jsonl"
    p_pgs.write_text(
        "\n".join(json.dumps(p, ensure_ascii=False) for p in pages) + "\n",
        encoding="utf-8",
    )

    p_def = p_tmp / "syn_defaults.json"
    p_def.write_text(
        json.dumps(
            {
                "server_version": SERVER_VERSION,
                "version": SERVER_VERSION,
                "models": {
                    GENERATOR_9B_MODEL: {"digest": PINNED_9B_DIGEST},
                    TEXT_27B_MODEL: {"digest": PINNED_27B_DIGEST},
                    EMBEDDING_4B_MODEL: {"digest": PINNED_EMBEDDING_DIGEST},
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    p_shw = p_tmp / "syn_model_show.json"
    p_shw.write_text(
        json.dumps(
            {
                GENERATOR_9B_MODEL: {
                    "parameters": "presence_penalty 1.5\ntop_k 20\ntop_p 0.95\n",
                    "details": {"quantization_level": "Q4_K_M"},
                },
                TEXT_27B_MODEL: {
                    "parameters": "",
                    "details": {"quantization_level": "Q3_K_M"},
                },
                EMBEDDING_4B_MODEL: {
                    "parameters": "",
                    "details": {"quantization_level": "Q4_K_M"},
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    if queries_path is not None:
        p_qry = Path(queries_path).resolve()
    else:
        p_qry = p_tmp / "syn_queries.json"
        p_qry.write_text(
            json.dumps(
                {
                    "cases": [
                        {"id": cid, "query": f"Pregunta sintética {cid}"}
                        for cid in REQUIRED_QUERY_ROSTER
                    ]
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    paths = {
        "index": p_idx.relative_to(PROJECT_ROOT).as_posix(),
        "queries": p_qry.relative_to(PROJECT_ROOT).as_posix(),
        "units": p_unt.relative_to(PROJECT_ROOT).as_posix(),
        "pages": p_pgs.relative_to(PROJECT_ROOT).as_posix(),
        "defaults": p_def.relative_to(PROJECT_ROOT).as_posix(),
        "model_show": p_shw.relative_to(PROJECT_ROOT).as_posix(),
    }
    input_hashes = {
        k: canonical_file_sha256((PROJECT_ROOT / rel).resolve())
        for k, rel in paths.items()
    }
    if drift_input_key and drift_input_key in input_hashes:
        input_hashes[drift_input_key] = "0" * 64

    # Build the valid expected schedule from queries file if possible
    try:
        p_qry_resolved = (PROJECT_ROOT / paths["queries"]).resolve()
        qry_data = json.loads(p_qry_resolved.read_text(encoding="utf-8"))
        cases = validate_queries_payload(qry_data)
        schedule = build_expected_schedule(cases)
    except Exception:
        schedule = [{"case_id": "b01"}] * 22

    if tamper_schedule == "allmixed":
        schedule = [dict(att, arm="mixed") for att in schedule]
    elif tamper_schedule == "reordered":
        schedule = list(reversed(schedule))
    elif tamper_schedule == "swapped_arm":
        if schedule:
            schedule[0] = dict(schedule[0], arm="all27" if schedule[0].get("arm") == "mixed" else "mixed")
    elif tamper_schedule == "tampered_query":
        if schedule:
            schedule[0] = dict(schedule[0], query="TAMPERED QUERY TEXT")

    proto = {
        "schema_version": PROTOCOL_SCHEMA_VERSION,
        "frozen_files": closure,
        "paths": paths,
        "input_hashes": input_hashes,
        "schedule": schedule,
    }
    proto_file.write_text(json.dumps(proto, ensure_ascii=False, indent=2), encoding="utf-8")
    return proto_file


class TestSingle27PreservedGrounder(unittest.TestCase):
    """Verifies Single27PreservedGrounder routes all text stages to 27B without masquerade and without 27B unload."""

    def test_single27_all_text_stages_model27_and_truthful_canonical_validation(self):
        units, pages, hits = load_test_fixtures()

        sel_resp = make_raw_chat_response(TEXT_27B_MODEL, {"status": "selected", "ids": ["P1"], "reason": ""})
        gen_resp = make_raw_chat_response(TEXT_27B_MODEL, {
            "status": "answered",
            "claims": [{"claim_id": "C1", "text": "Texto atómico 27B.", "evidence_ids": ["E1"]}],
            "reason": "",
        })
        ver_resp = make_raw_chat_response(TEXT_27B_MODEL, {
            "verdicts": [
                {
                    "claim_id": "C1",
                    "verdict": "supported",
                    "reason": "Respaldo 27B.",
                    "evidence_verdicts": [{"evidence_id": "E1", "relation": "supports"}],
                }
            ]
        })

        transport = FakeTransport(chat_responses=[sel_resp, gen_resp, ver_resp])
        verifier = Q3VerifierClient(transport=transport)
        verifier._preflight_done = True

        gen27 = ComparisonGenerator(TEXT_27B_MODEL, transport=transport)
        grounder = Single27PreservedGrounder(units, pages, verifier=verifier, generator=gen27)

        # 1. Answer and check trace models
        result = grounder.answer("¿Pregunta de prueba?", hits, expected_digest=PINNED_27B_DIGEST)
        trace = result["grounding"]["trace"]
        self.assertEqual(len(trace), 3)

        # Selection on 27B
        self.assertEqual(trace[0]["stage"], "selection")
        self.assertEqual(trace[0]["model"], TEXT_27B_MODEL)
        self.assertEqual(trace[0]["model_digest"], PINNED_27B_DIGEST)
        self.assertEqual(trace[0]["parameters"], DEFAULT_PARAMETERS)

        # Generation on 27B
        self.assertEqual(trace[1]["stage"], "generation")
        self.assertEqual(trace[1]["model"], TEXT_27B_MODEL)
        self.assertEqual(trace[1]["model_digest"], PINNED_27B_DIGEST)
        self.assertEqual(trace[1]["parameters"], DEFAULT_PARAMETERS)

        # Verification on 27B
        self.assertEqual(trace[2]["stage"], "verification")
        self.assertEqual(trace[2]["model"], TEXT_27B_MODEL)
        self.assertEqual(trace[2]["model_digest"], PINNED_27B_DIGEST)
        self.assertEqual(trace[2]["parameters"], VERIFIER_Q3_PARAMETERS)

        # 2. Canonical validation with expected_model=TEXT_27B_MODEL succeeds truthfully
        validated = validate_grounded_result(
            result,
            grounder.expanded_hits,
            expected_digest=PINNED_27B_DIGEST,
            expected_model=TEXT_27B_MODEL,
            verification_policy=grounder.verification_policy,
        )
        self.assertEqual(validated["status"], "answered")

        # 3. Canonical validation with expected_model=DEFAULT_GENERATION_MODEL FAILS (proves no 9B masquerade)
        with self.assertRaisesRegex(ValueError, "model mismatch"):
            validate_grounded_result(
                result,
                grounder.expanded_hits,
                expected_digest=PINNED_27B_DIGEST,
                expected_model=DEFAULT_GENERATION_MODEL,
                verification_policy=grounder.verification_policy,
            )

    def test_single27_never_unloads_27b_during_text_role_switches(self):
        units, pages, hits = load_test_fixtures()

        sel_resp = make_raw_chat_response(TEXT_27B_MODEL, {"status": "selected", "ids": ["P1"], "reason": ""})
        gen_resp = make_raw_chat_response(TEXT_27B_MODEL, {
            "status": "answered",
            "claims": [{"claim_id": "C1", "text": "Claim 1", "evidence_ids": ["E1"]}],
            "reason": "",
        })
        ver_resp = make_raw_chat_response(TEXT_27B_MODEL, {
            "verdicts": [
                {
                    "claim_id": "C1",
                    "verdict": "supported",
                    "reason": "OK",
                    "evidence_verdicts": [{"evidence_id": "E1", "relation": "supports"}],
                }
            ]
        })

        # Start with 27B resident in ps
        transport = FakeTransport(
            resident_models=[TEXT_27B_MODEL],
            chat_responses=[sel_resp, gen_resp, ver_resp],
        )
        verifier = Q3VerifierClient(transport=transport)
        verifier._preflight_done = True

        gen27 = ComparisonGenerator(TEXT_27B_MODEL, transport=transport)
        grounder = Single27PreservedGrounder(units, pages, verifier=verifier, generator=gen27)

        outcome = generate_arm(
            grounder,
            "Consulta",
            hits,
            PINNED_27B_DIGEST,
            preserve=True,
            verification_policy=grounder.verification_policy,
            expected_model=TEXT_27B_MODEL,
        )
        self.assertEqual(outcome["status"], "answered")

        # In all27, TEXT_27B_MODEL was NEVER unloaded during text stages
        self.assertNotIn(TEXT_27B_MODEL, transport.unloaded_models)
        self.assertEqual(transport.unload_calls, [])

        # Directly verify all stages: selection, generation, repair_selection, repair_generation, verification, reverification
        # In Single27PreservedGrounder, release_loaded is NEVER called on 27B
        mock_verifier = Q3VerifierClient(transport=transport)
        mock_verifier.release_loaded = unittest.mock.MagicMock(wraps=mock_verifier.release_loaded)
        s27_grounder = Single27PreservedGrounder(units, pages, verifier=mock_verifier, generator=gen27)

        # Call nonverification stages: PreservedGrounder._chat_step is called directly, release_loaded never called
        s27_grounder.catalog = {"records": []}
        # Calling non-verification stage
        with unittest.mock.patch.object(PreservedGrounder, "_chat_step", return_value=({}, {})):
            s27_grounder._chat_step("selection", [], {}, PINNED_27B_DIGEST)
            s27_grounder._chat_step("generation", [], {}, PINNED_27B_DIGEST)
            s27_grounder._chat_step("repair_selection", [], {}, PINNED_27B_DIGEST)
            s27_grounder._chat_step("repair_generation", [], {}, PINNED_27B_DIGEST)
        mock_verifier.release_loaded.assert_not_called()

        # Verification stage calls release_loaded with 9B (DEFAULT_GENERATION_MODEL), NEVER with 27B
        with unittest.mock.patch.object(mock_verifier, "chat", return_value=make_raw_chat_response(TEXT_27B_MODEL, {"verdicts": []})):
            s27_grounder._chat_step("verification", [{"role": "user", "content": "{}"}], {"type": "object"}, PINNED_27B_DIGEST)
            s27_grounder._chat_step("reverification", [{"role": "user", "content": "{}"}], {"type": "object"}, PINNED_27B_DIGEST)
        for call_args in mock_verifier.release_loaded.call_args_list:
            self.assertNotEqual(call_args[0][0], TEXT_27B_MODEL)
            self.assertEqual(call_args[0][0], DEFAULT_GENERATION_MODEL)

    def test_original_mixed_route_unchanged(self):
        units, pages, hits = load_test_fixtures()

        sel_resp = make_raw_chat_response(GENERATOR_9B_MODEL, {"status": "selected", "ids": ["P1"], "reason": ""})
        gen_resp = make_raw_chat_response(GENERATOR_9B_MODEL, {
            "status": "answered",
            "claims": [{"claim_id": "C1", "text": "Afirmación 9B.", "evidence_ids": ["E1"]}],
            "reason": "",
        })
        ver_resp = make_raw_chat_response(TEXT_27B_MODEL, {
            "verdicts": [
                {
                    "claim_id": "C1",
                    "verdict": "supported",
                    "reason": "Soporte verificado",
                    "evidence_verdicts": [{"evidence_id": "E1", "relation": "supports"}],
                }
            ]
        })

        transport = FakeTransport(
            resident_models=[GENERATOR_9B_MODEL],
            chat_responses=[sel_resp, gen_resp, ver_resp],
        )
        verifier = Q3VerifierClient(transport=transport)
        verifier._preflight_done = True

        gen9b = ComparisonGenerator(GENERATOR_9B_MODEL, transport=transport)
        mixed_grounder = Q3PreservedGrounder(units, pages, verifier=verifier, generator=gen9b, partial_generation=True)

        outcome = generate_arm(
            mixed_grounder,
            "Consulta",
            hits,
            PINNED_9B_DIGEST,
            preserve=True,
            verification_policy=mixed_grounder.verification_policy,
            expected_model=GENERATOR_9B_MODEL,
        )
        self.assertEqual(outcome["status"], "answered")

        # In mixed, 9B WAS unloaded before 27B verification
        self.assertIn(GENERATOR_9B_MODEL, transport.unloaded_models)
        trace = outcome["full_result"]["grounding"]["trace"]
        self.assertEqual(trace[0]["model"], GENERATOR_9B_MODEL)
        self.assertEqual(trace[1]["model"], GENERATOR_9B_MODEL)
        self.assertEqual(trace[2]["model"], TEXT_27B_MODEL)

    def test_accepted_only_partial_publication_and_invalid_citation_rejection(self):
        units, pages, hits = load_test_fixtures()

        # Result with 2 claims: C1 supported, C2 contradicted -> status is insufficient_evidence
        full_res = {
            "status": "insufficient_evidence",
            "claims": [],
            "reason": "Parte fallida",
            "grounding": {
                "accepted_claims": [{"claim_id": "C1", "text": "Texto C1", "evidence_ids": ["E1"]}],
                "initial_verdicts": [{"claim_id": "C1", "verdict": "supported"}, {"claim_id": "C2", "verdict": "contradicted"}],
                "selected_evidence": [{"evidence_id": "E1", "source_id": "S1", "quote": hits[0].chunk.text[:50]}],
                "repair_evidence": [],
                "initial_claims": [{"claim_id": "C1", "text": "Texto C1", "evidence_ids": ["E1"]}],
            },
        }

        # allow_partial=True yields status='partial'
        pub = publication(full_res, preserve=True, allow_partial=True)
        self.assertEqual(pub["status"], "partial")
        self.assertEqual(len(pub["claims"]), 1)
        self.assertEqual(pub["claims"][0]["text"], "Texto C1")
        validate_publication(pub, full_res, preserve=True, allow_partial=True)

        # Invalid citation quote rejection: if quote does not match source, validate_answer raises ValueError
        bad_payload = {
            "status": "answered",
            "claims": [{"text": "Claim", "evidence": [{"source_id": "S1", "quote": "Cita completamente inventada no existente"}]}],
            "reason": "",
        }
        with self.assertRaisesRegex(ValueError, "quote not found in source"):
            validate_answer(bad_payload, hits)


class TestSamplingOptionsAndEquivalence(unittest.TestCase):
    """Verifies both arms send identical EFFECTIVE_GENERATION_OPTIONS despite 27B having no modelfile parameters."""

    def test_effective_sampling_options_sent_equally_in_both_arms(self):
        transport = FakeTransport()

        gen9b = ComparisonGenerator(GENERATOR_9B_MODEL, transport=transport)
        gen27 = ComparisonGenerator(TEXT_27B_MODEL, transport=transport)

        # 9B call
        gen9b.chat([{"role": "user", "content": "hi"}], {"type": "object"})
        # 27B call
        gen27.chat([{"role": "user", "content": "hi"}], {"type": "object"})

        self.assertEqual(len(transport.chat_calls), 2)
        req9b = transport.chat_calls[0]
        req27 = transport.chat_calls[1]

        # Verify options are identical in both requests sent to Ollama
        self.assertEqual(req9b["options"], req27["options"])
        self.assertEqual(req9b["options"], EFFECTIVE_GENERATION_OPTIONS)
        self.assertEqual(req27["options"], EFFECTIVE_GENERATION_OPTIONS)

        # Verify specific critical options
        for key in ("presence_penalty", "top_k", "top_p", "min_p", "repeat_penalty", "temperature", "seed", "num_predict"):
            self.assertEqual(req9b["options"][key], EFFECTIVE_GENERATION_OPTIONS[key])
            self.assertEqual(req27["options"][key], EFFECTIVE_GENERATION_OPTIONS[key])

    def test_canonical_role_parameters_remain_default_parameters(self):
        self.assertEqual(CANONICAL_ROLE_PARAMETERS, DEFAULT_PARAMETERS)
        self.assertNotIn("presence_penalty", CANONICAL_ROLE_PARAMETERS)


class TestLifecycleAndColdState(unittest.TestCase):
    """Verifies cold state lifecycle, unrelated model abort, and embedding release."""

    def test_unrelated_model_loaded_aborts_without_unloading(self):
        # Resident model is 'bespoke-minicheck:7b' (not in ALLOWED_EXPERIMENT_MODELS)
        transport = FakeTransport(resident_models=["bespoke-minicheck:7b"])

        with self.assertRaisesRegex(RuntimeError, "Unrelated models currently resident"):
            ensure_cold_state(transport)

        # Must NOT have unloaded the unrelated model!
        self.assertEqual(transport.unload_calls, [])
        self.assertIn("bespoke-minicheck:7b", transport.resident_models)

    def test_selected_models_unloaded_and_confirmed_absent(self):
        transport = FakeTransport(resident_models=[GENERATOR_9B_MODEL, TEXT_27B_MODEL])

        lifecycle = ensure_cold_state(transport)
        self.assertEqual(len(lifecycle), 2)
        self.assertEqual(transport.resident_models, [])
        self.assertIn(GENERATOR_9B_MODEL, transport.unload_calls)
        self.assertIn(TEXT_27B_MODEL, transport.unload_calls)

    def test_transport_restrictions(self):
        # Non-loopback base_url rejected
        with self.assertRaisesRegex(ValueError, "Only strict loopback URL"):
            MixedSingle27Transport(base_url="http://192.168.1.10:11434")

        # Timeout < 240 rejected
        with self.assertRaisesRegex(ValueError, "timeout must be a number >= 240.0s"):
            MixedSingle27Transport(timeout=180.0)

        # Unauthorized model rejected in show/chat/unload
        t = MixedSingle27Transport()
        with self.assertRaisesRegex(ValueError, "not permitted"):
            t.show("unauthorized:model")
        with self.assertRaisesRegex(ValueError, "not permitted"):
            t.unload("unauthorized:model")
        with self.assertRaisesRegex(ValueError, "not permitted"):
            t.chat({"model": "unauthorized:model"})
        with self.assertRaisesRegex(ValueError, "not permitted"):
            t.embed({"model": "unauthorized:model"})


class TestPreNetworkValidations(unittest.TestCase):
    """Verifies that all pre-network validations fail BEFORE constructing clients or contacting network."""

    _build_test_protocol = staticmethod(build_test_protocol)

    def test_prepare_refuses_overwrite(self):
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "evaluation") as tmpdir:
            proto_file = Path(tmpdir) / "protocol.json"
            proto_file.write_text("{}", encoding="utf-8")

            with self.assertRaises(FileExistsError):
                prepare(proto_file)

    def test_run_refuses_overwrite_before_network(self):
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "evaluation") as tmpdir:
            proto_file = Path(tmpdir) / "protocol.json"
            out_file = Path(tmpdir) / "results.json"
            out_file.write_text("{}", encoding="utf-8")

            class BombTransport:
                def get_version(self): raise AssertionError("Network contacted!")

            with self.assertRaises(FileExistsError):
                run(proto_file, out_file, transport=BombTransport())

    def test_run_detects_frozen_file_drift_before_network(self):
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "evaluation") as tmpdir:
            proto_file = Path(tmpdir) / "protocol.json"
            out_file = Path(tmpdir) / "results.json"

            fake_proto = {
                "schema_version": PROTOCOL_SCHEMA_VERSION,
                "frozen_files": {
                    "rag_pensiones/answer.py": "0" * 64,  # Intentionally drifted hash
                },
                "paths": {
                    "index": "data/processed/airef_2025_qwen3_4b_embeddings.json",
                    "queries": "evaluation/scope_coverage_2026-10-26/full_inputs.json",
                    "units": "evaluation/reconstructed_units_catalog.json",
                    "pages": "data/processed/airef_2025_pages.jsonl",
                },
                "schedule": [{"case_id": "b01"}] * 22,
            }
            proto_file.write_text(json.dumps(fake_proto), encoding="utf-8")

            class BombTransport:
                def get_version(self): raise AssertionError("Network contacted!")

            with self.assertRaisesRegex(ValueError, "Frozen file modified on disk"):
                run(proto_file, out_file, transport=BombTransport())

    def test_run_detects_gold_labels_in_queries_before_network(self):
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "evaluation") as tmpdir:
            bad_queries_file = Path(tmpdir) / "bad_queries.json"
            bad_queries_file.write_text(json.dumps({
                "cases": [{"id": f"c{i}", "query": f"q{i}", "gold_answer": "leaked"} for i in range(11)]
            }), encoding="utf-8")

            # 1. Test prepare independently rejects compound gold_answer before network
            proto_file_prep = Path(tmpdir) / "prep_proto.json"
            with self.assertRaisesRegex(ValueError, "Gold labels detected in query input"):
                prepare(proto_file_prep, queries_path=bad_queries_file)
            self.assertFalse(proto_file_prep.exists())

            # 2. Test run independently rejects compound gold_answer before network
            proto_file_run = self._build_test_protocol(tmpdir, queries_path=bad_queries_file)
            out_file = Path(tmpdir) / "out.json"

            class BombTransport:
                def get_version(self): raise AssertionError("Network contacted!")

            with self.assertRaisesRegex(ValueError, "Gold labels detected in query input"):
                run(proto_file_run, out_file, transport=BombTransport())
            self.assertFalse(out_file.exists())

    def test_run_detects_input_drift_before_network(self):
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "evaluation") as tmpdir:
            proto_file = self._build_test_protocol(tmpdir, drift_input_key="defaults")
            out_file = Path(tmpdir) / "out_drift.json"

            class BombTransport:
                def get_version(self): raise AssertionError("Network contacted!")

            with self.assertRaisesRegex(ValueError, "Input file hash mismatch"):
                run(proto_file, out_file, transport=BombTransport())
            self.assertFalse(out_file.exists())

    def test_run_detects_omitted_source_pin_before_network(self):
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "evaluation") as tmpdir:
            proto_file = self._build_test_protocol(tmpdir, omit_source_file="rag_pensiones/chunking.py")
            out_file = Path(tmpdir) / "out_omit.json"

            class BombTransport:
                def get_version(self): raise AssertionError("Network contacted!")

            with self.assertRaisesRegex(ValueError, "missing required source closure pins"):
                run(proto_file, out_file, transport=BombTransport())
            self.assertFalse(out_file.exists())

    def test_pre_network_rejects_extra_query_keys_and_labels_before_network(self):
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "evaluation") as tmpdir:
            # Extra key in case
            bad_cases_file = Path(tmpdir) / "bad_cases.json"
            bad_cases_file.write_text(json.dumps({
                "cases": [{"id": f"c{i}", "query": f"q{i}", "extra_prop": "v"} for i in range(11)]
            }), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unexpected extra keys"):
                prepare(Path(tmpdir) / "p1.json", queries_path=bad_cases_file)

            # Extra document key without label
            bad_doc_file = Path(tmpdir) / "bad_doc.json"
            bad_doc_file.write_text(json.dumps({
                "cases": [{"id": f"c{i}", "query": f"q{i}"} for i in range(11)],
                "extra_key": 123,
            }), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Query document keys must be exactly"):
                prepare(Path(tmpdir) / "p2.json", queries_path=bad_doc_file)

            # Document key containing label
            label_doc_file = Path(tmpdir) / "label_doc.json"
            label_doc_file.write_text(json.dumps({
                "cases": [{"id": f"c{i}", "query": f"q{i}"} for i in range(11)],
                "gold_standard": "val",
            }), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Gold labels detected in query input"):
                prepare(Path(tmpdir) / "p3.json", queries_path=label_doc_file)

    def test_pre_network_rejects_wrong_roster_before_network(self):
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "evaluation") as tmpdir:
            bad_roster = list(REQUIRED_QUERY_ROSTER)
            bad_roster[-1] = "invalid_case_id"
            bad_cases_file = Path(tmpdir) / "wrong_roster.json"
            bad_cases_file.write_text(
                json.dumps({
                    "cases": [{"id": cid, "query": f"Query {cid}"} for cid in bad_roster]
                }),
                encoding="utf-8",
            )

            # 1. prepare rejects before network/writing
            proto_file_prep = Path(tmpdir) / "prep_proto.json"
            with self.assertRaisesRegex(ValueError, "Query cases must match exact ordered roster"):
                prepare(proto_file_prep, queries_path=bad_cases_file)
            self.assertFalse(proto_file_prep.exists())

            # 2. run rejects before network
            proto_file_run = self._build_test_protocol(tmpdir, queries_path=bad_cases_file)
            out_file = Path(tmpdir) / "out.json"

            class BombTransport:
                def get_version(self): raise AssertionError("Network contacted!")

            with self.assertRaisesRegex(ValueError, "Query cases must match exact ordered roster"):
                run(proto_file_run, out_file, transport=BombTransport())
            self.assertFalse(out_file.exists())

    def test_pre_network_rejects_duplicate_roster_before_network(self):
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "evaluation") as tmpdir:
            dup_roster = list(REQUIRED_QUERY_ROSTER)
            dup_roster[1] = dup_roster[0]  # duplicate b01 instead of b02
            dup_cases_file = Path(tmpdir) / "duplicate_roster.json"
            dup_cases_file.write_text(
                json.dumps({
                    "cases": [{"id": cid, "query": f"Query {cid}"} for cid in dup_roster]
                }),
                encoding="utf-8",
            )

            # 1. prepare rejects before network/writing
            proto_file_prep = Path(tmpdir) / "prep_proto.json"
            with self.assertRaisesRegex(ValueError, "Query cases must match exact ordered roster"):
                prepare(proto_file_prep, queries_path=dup_cases_file)
            self.assertFalse(proto_file_prep.exists())

            # 2. run rejects before network
            proto_file_run = self._build_test_protocol(tmpdir, queries_path=dup_cases_file)
            out_file = Path(tmpdir) / "out.json"

            class BombTransport:
                def get_version(self): raise AssertionError("Network contacted!")

            with self.assertRaisesRegex(ValueError, "Query cases must match exact ordered roster"):
                run(proto_file_run, out_file, transport=BombTransport())
            self.assertFalse(out_file.exists())

    def test_run_rejects_allmixed_schedule_before_network(self):
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "evaluation") as tmpdir:
            proto_file = self._build_test_protocol(tmpdir, tamper_schedule="allmixed")
            out_file = Path(tmpdir) / "out_allmixed.json"

            class BombTransport:
                def get_version(self): raise AssertionError("Network contacted!")

            with self.assertRaisesRegex(ValueError, "mismatch against expected schedule"):
                run(proto_file, out_file, transport=BombTransport())
            self.assertFalse(out_file.exists())

    def test_run_rejects_reordered_schedule_before_network(self):
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "evaluation") as tmpdir:
            proto_file = self._build_test_protocol(tmpdir, tamper_schedule="reordered")
            out_file = Path(tmpdir) / "out_reordered.json"

            class BombTransport:
                def get_version(self): raise AssertionError("Network contacted!")

            with self.assertRaisesRegex(ValueError, "mismatch against expected schedule"):
                run(proto_file, out_file, transport=BombTransport())
            self.assertFalse(out_file.exists())

    def test_run_rejects_tampered_query_schedule_before_network(self):
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "evaluation") as tmpdir:
            proto_file = self._build_test_protocol(tmpdir, tamper_schedule="tampered_query")
            out_file = Path(tmpdir) / "out_tampered.json"

            class BombTransport:
                def get_version(self): raise AssertionError("Network contacted!")

            with self.assertRaisesRegex(ValueError, "mismatch against expected schedule"):
                run(proto_file, out_file, transport=BombTransport())
            self.assertFalse(out_file.exists())

    def test_protocol_fixtures_fully_isolated_inside_tmpdir(self):
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "evaluation") as tmpdir:
            proto_file = self._build_test_protocol(tmpdir)
            self.assertTrue(proto_file.is_file())
            proto = json.loads(proto_file.read_text(encoding="utf-8"))
            paths = proto["paths"]

            tmp_root = Path(tmpdir).resolve()
            for key in ("index", "queries", "units", "pages", "defaults", "model_show"):
                resolved = (PROJECT_ROOT / paths[key]).resolve()
                self.assertTrue(resolved.is_file(), f"Fixture file {key} missing: {resolved}")
                self.assertTrue(
                    resolved.is_relative_to(tmp_root),
                    f"Fixture {key} ({resolved}) escaped tmpdir ({tmp_root})",
                )
                self.assertEqual(
                    proto["input_hashes"][key],
                    canonical_file_sha256(resolved),
                    f"Hash mismatch for fixture {key}",
                )

            # Synthetic index and catalog satisfy actual canonical validators
            syn_idx = json.loads((PROJECT_ROOT / paths["index"]).read_text(encoding="utf-8"))
            validate_index(syn_idx)

            syn_unt = json.loads((PROJECT_ROOT / paths["units"]).read_text(encoding="utf-8"))
            syn_pgs = [
                json.loads(line)
                for line in (PROJECT_ROOT / paths["pages"]).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            validate_catalog(syn_unt, syn_pgs)

            # Source closure pins real source code outside tmpdir without corpus hashing
            for src_rel, _ in proto["frozen_files"].items():
                src_path = (PROJECT_ROOT / src_rel).resolve()
                self.assertTrue(src_path.is_file())
                self.assertFalse(src_path.is_relative_to(tmp_root))
                self.assertTrue(src_rel.endswith(".py"))


class TestExecutionAndCheckpointing(unittest.TestCase):
    """Verifies attempt execution, checkpointing, error preservation, and stats role groups."""

    _build_test_protocol = staticmethod(build_test_protocol)

    def test_local_attempt_error_preserved_and_next_attempt_continues(self):
        units, pages, hits = load_test_fixtures()
        syn_index = make_synthetic_index(pages)

        attempt_spec_fail = {"attempt_index": 0, "case_id": "b01", "query": "Q1", "arm": "all27"}
        attempt_spec_ok = {"attempt_index": 1, "case_id": "b01", "query": "Q1", "arm": "mixed"}

        # Attempt 0: chat raises error
        transport_fail = FakeTransport(chat_error=RuntimeError("Transient timeout in chat"))
        res_fail = execute_attempt(
            attempt_spec_fail,
            index_data=syn_index,
            units_data=units,
            pages_data=pages,
            transport=transport_fail,
        )

        self.assertEqual(res_fail["status"], "error")
        self.assertIn("Transient timeout in chat", res_fail["error"])

        # Attempt 1: succeeds with valid responses
        transport_ok = FakeTransport()
        res_ok = execute_attempt(
            attempt_spec_ok,
            index_data=syn_index,
            units_data=units,
            pages_data=pages,
            transport=transport_ok,
        )
        self.assertEqual(res_ok["status"], "completed")

    def test_response_done_reason_length_guard(self):
        units, pages, hits = load_test_fixtures()
        syn_index = make_synthetic_index(pages)
        validate_index(syn_index)

        # Response with done_reason='length' (truncated generation)
        trunc_resp = {
            "model": GENERATOR_9B_MODEL,
            "done": True,
            "done_reason": "length",
            "message": {"role": "assistant", "content": '{"status": "selected", "ids": ["P1"]'},
            "prompt_eval_count": 100,
            "eval_count": 1024,
            "eval_duration": 1000000,
            "total_duration": 2000000,
            "load_duration": 10000,
            "prompt_eval_duration": 50000,
        }
        transport = FakeTransport(chat_responses=[trunc_resp])
        attempt_spec = {"attempt_index": 0, "case_id": "b01", "query": "Q1", "arm": "mixed"}

        res = execute_attempt(
            attempt_spec,
            index_data=syn_index,
            units_data=units,
            pages_data=pages,
            transport=transport,
        )

        self.assertEqual(res["status"], "error")
        self.assertIn("truncated due to length limit", res["error"])
        # No unchecked publication was created
        self.assertIsNone(res["coverage_publication"])
        self.assertIsNone(res["coverage_audit"])
        # No hidden retry: exactly 1 chat call was executed before abort
        self.assertEqual(len(transport.chat_calls), 1)

    def test_message_prompt_budget_overflow_guard(self):
        transport = FakeTransport()
        gen = ComparisonGenerator(GENERATOR_9B_MODEL, transport=transport)

        # Message exceeding MAX_MESSAGES_BYTES
        huge_content = "X" * (MAX_MESSAGES_BYTES + 100)
        messages = [{"role": "user", "content": huge_content}]
        schema = {"type": "object", "properties": {"status": {"type": "string"}}}

        with self.assertRaisesRegex(ValueError, "Prompt message budget overflow"):
            gen.chat(messages, schema)

        # Verified: no network request sent, no hidden retry
        self.assertEqual(len(transport.chat_calls), 0)
        self.assertEqual(len(gen.calls), 0)

    def test_wrong_model_or_digest_during_execution_guard(self):
        units, pages, hits = load_test_fixtures()
        syn_index = make_synthetic_index(pages)

        # 1. Chat response returned wrong model
        wrong_model_resp = make_raw_chat_response("unauthorized_model_name:latest", {"status": "selected", "ids": ["P1"]})
        transport = FakeTransport(chat_responses=[wrong_model_resp])
        attempt_spec = {"attempt_index": 0, "case_id": "b01", "query": "Q1", "arm": "mixed"}

        res = execute_attempt(
            attempt_spec,
            index_data=syn_index,
            units_data=units,
            pages_data=pages,
            transport=transport,
        )

        self.assertEqual(res["status"], "error")
        self.assertIn("Model mismatch in chat response", res["error"])
        self.assertIsNone(res["coverage_publication"])
        self.assertEqual(len(transport.chat_calls), 1)

        # 2. Canonical validator rejects wrong digest during execution
        transport_digest = FakeTransport(chat_responses=[
            make_raw_chat_response(TEXT_27B_MODEL, {"status": "selected", "ids": ["P1"], "reason": ""}),
            make_raw_chat_response(TEXT_27B_MODEL, {"status": "answered", "claims": [{"claim_id": "C1", "text": "Claim", "evidence_ids": ["E1"]}], "reason": ""}),
            make_raw_chat_response(TEXT_27B_MODEL, {"verdicts": [{"claim_id": "C1", "verdict": "supported", "reason": "OK", "evidence_verdicts": [{"evidence_id": "E1", "relation": "supports"}]}]}),
        ])
        verifier_digest = Q3VerifierClient(transport=transport_digest)
        verifier_digest._preflight_done = True
        grounder = Single27PreservedGrounder(
            units,
            pages,
            verifier=verifier_digest,
            generator=ComparisonGenerator(TEXT_27B_MODEL, transport=transport_digest),
        )
        valid_result = grounder.answer("¿Pregunta?", hits, expected_digest=PINNED_27B_DIGEST)
        bad_trace_result = copy.deepcopy(valid_result)
        bad_trace_result["grounding"]["trace"][0]["model_digest"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "model_digest"):
            validate_grounded_result(
                bad_trace_result,
                grounder.expanded_hits,
                expected_digest=PINNED_27B_DIGEST,
                expected_model=TEXT_27B_MODEL,
                verification_policy=grounder.verification_policy,
            )

    def test_interrupted_checkpoint_status_and_propagation(self):
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "evaluation") as tmpdir:
            proto_file = self._build_test_protocol(tmpdir)
            out_file = Path(tmpdir) / "interrupted_results.json"

            attempt_counter = 0

            def fake_execute(attempt_spec, **kwargs):
                nonlocal attempt_counter
                attempt_counter += 1
                if attempt_counter == 3:
                    raise KeyboardInterrupt("Simulated user interrupt during attempt 3")
                return {
                    "attempt_index": attempt_spec.get("attempt_index", 0),
                    "case_id": "b01",
                    "query": "Q",
                    "arm": "mixed",
                    "status": "completed",
                    "query_wall_seconds": 1.0,
                }

            with unittest.mock.patch("experiments.compare_mixed_single27.execute_attempt", side_effect=fake_execute):
                transport = FakeTransport()
                with self.assertRaises(KeyboardInterrupt):
                    run(proto_file, out_file, transport=transport)

            # Checkpoint file exists on disk
            self.assertTrue(out_file.is_file())
            saved = json.loads(out_file.read_text(encoding="utf-8"))

            # Honest incomplete status and unfinished timestamp
            self.assertEqual(saved["run_status"], "incomplete")
            self.assertIsNone(saved["finished_at"])
            # Completed attempts (1 and 2) preserved on disk
            self.assertEqual(len(saved["attempts"]), 2)

    def test_per_chat_ps_snapshots_and_context_in_both_arms(self):
        units, pages, hits = load_test_fixtures()
        syn_index = make_synthetic_index(pages)

        for arm in ("mixed", "all27"):
            transport = FakeTransport()
            attempt_spec = {"attempt_index": 0, "case_id": "b01", "query": "Q1", "arm": arm}

            res = execute_attempt(
                attempt_spec,
                index_data=syn_index,
                units_data=units,
                pages_data=pages,
                transport=transport,
            )

            self.assertEqual(res["status"], "completed", f"Failed in arm {arm}: {res.get('error')}")
            snapshots = res["ps_snapshots"]
            # Must contain start_cold, after_embedding_release, chat snapshots, after_factual_outcome, after_coverage_audit
            event_tags = [s["event"] for s in snapshots]
            self.assertIn("start_cold", event_tags)
            self.assertIn("after_embedding_release", event_tags)
            self.assertIn("after_factual_outcome", event_tags)
            self.assertIn("after_coverage_audit", event_tags)

            # Check that per-chat snapshots were captured
            chat_snapshots = [s for s in snapshots if "after_chat_" in s["event"]]
            self.assertGreaterEqual(len(chat_snapshots), 4, f"Expected at least 4 chat snapshots in arm {arm}")

            for s in chat_snapshots:
                self.assertIn("request_id", s)
                self.assertIn("model", s)
                self.assertIn("stage", s)
                self.assertIn("timestamp", s)
                self.assertIn("models", s)
                for m in s["models"]:
                    self.assertIn("name", m)
                    self.assertIn("size", m)
                    self.assertIn("size_vram", m)
                    self.assertEqual(m.get("context_length"), 8192)

            if arm == "all27":
                # In all27, 27B is NEVER unloaded during text stages
                self.assertNotIn(TEXT_27B_MODEL, transport.unloaded_models)

    def test_retrieval_hit_content_and_record_hashes(self):
        units, pages, hits = load_test_fixtures()
        syn_index = make_synthetic_index(pages)
        transport = FakeTransport()
        attempt_spec = {"attempt_index": 0, "case_id": "b01", "query": "Q1", "arm": "mixed"}

        res = execute_attempt(
            attempt_spec,
            index_data=syn_index,
            units_data=units,
            pages_data=pages,
            transport=transport,
        )

        self.assertEqual(res["status"], "completed")
        ret_hits = res["retrieval"]["hits"]
        self.assertEqual(len(ret_hits), 1)
        h = ret_hits[0]

        # Content hash matches raw chunk text SHA-256
        expected_content_hash = hashlib.sha256(h["text"].encode("utf-8")).hexdigest()
        self.assertEqual(h["content_sha256"], expected_content_hash)

        # Record hash is valid 64-char hex SHA-256
        self.assertEqual(len(h["record_sha256"]), 64)
        self.assertTrue(all(c in "0123456789abcdef" for c in h["record_sha256"]))

        # Preserves canonical retrieval metadata
        self.assertEqual(h["rank"], 1)
        self.assertEqual(h["chunk_id"], "chunk_syn_1")
        self.assertEqual(h["source_id"], "S1")
        self.assertEqual(h["page_number"], pages[1]["page_number"])

    def test_stats_exclusive_role_groups_no_double_counting(self):
        calls = [
            {"stage": "selection", "stats": {"prompt_eval_count": 100, "eval_count": 20, "eval_duration": 1000, "total_duration": 2000}},
            {"stage": "generation", "stats": {"prompt_eval_count": 200, "eval_count": 40, "eval_duration": 2000, "total_duration": 3000}},
            {"stage": "verification", "stats": {"prompt_eval_count": 300, "eval_count": 60, "eval_duration": 3000, "total_duration": 4000}},
            {"stage": COVERAGE_STAGE, "stats": {"prompt_eval_count": 150, "eval_count": 30, "eval_duration": 1500, "total_duration": 2500}},
        ]

        stats = compute_role_stats(calls)

        # Generation: selection + generation (100 + 200 = 300 prompt_eval)
        self.assertEqual(stats["generation"]["prompt_eval_count"], 300)
        self.assertEqual(stats["generation"]["eval_count"], 60)

        # Verification: 300 prompt_eval
        self.assertEqual(stats["verification"]["prompt_eval_count"], 300)
        self.assertEqual(stats["verification"]["eval_count"], 60)

        # Coverage: 150 prompt_eval
        self.assertEqual(stats["coverage"]["prompt_eval_count"], 150)
        self.assertEqual(stats["coverage"]["eval_count"], 30)

        # Total text: sum of all three (300 + 300 + 150 = 750)
        self.assertEqual(stats["total_text"]["prompt_eval_count"], 750)
        self.assertEqual(stats["total_text"]["eval_count"], 150)
        self.assertEqual(
            stats["total_text"]["prompt_eval_count"],
            stats["generation"]["prompt_eval_count"] + stats["verification"]["prompt_eval_count"] + stats["coverage"]["prompt_eval_count"]
        )


if __name__ == "__main__":
    unittest.main()
