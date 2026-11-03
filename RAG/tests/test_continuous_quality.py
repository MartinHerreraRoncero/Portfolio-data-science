"""Exhaustive offline synthetic tests for bounded continuous quality regression gate.

Strictly self-contained: no live network calls, no real dataset reads inside unit tests.
Tests all integrity bindings, canonical validators, four parent-proven mutations,
mandatory partial gating (b08/n03), n02 quote-only debt, baseline regressions,
source quote alignment against canonical pages, tamper resistance, and runner separation.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any
import unittest
from unittest.mock import MagicMock, patch

from experiments.continuous_quality import (
    COVERED_MANDATORY_SOURCE_PATHS,
    COVERED_RESULTS_SCHEMA_VERSION,
    DEFAULT_REFERENCE_REL,
    EXPECTED_CASE_COUNT,
    EXPECTED_CASE_IDS,
    HISTORICAL_CANONICAL_ROLE_PARAMETERS,
    HISTORICAL_EFFECTIVE_GENERATION_OPTIONS,
    HISTORICAL_MANDATORY_SOURCE_PATHS,
    HISTORICAL_PROTOCOL_SCHEMA_VERSION,
    HISTORICAL_RESULTS_SCHEMA_VERSION,
    PINNED_9B_DIGEST,
    PINNED_27B_DIGEST,
    PINNED_EMBEDDING_DIGEST,
    PROTOCOL_SCHEMA_VERSION,
    REFERENCE_SCHEMA_VERSION,
    REQUIRED_SERVER_VERSION,
    SCHEMA_VERSION,
    SHA256_HEX_PATTERN,
    check,
    evaluate_case_reference,
    file_sha256,
    legacy_main as gate_main,
    prepare,
    run as gate_run,
    strict_load_json,
    validate_reference_ledger,
)
from experiments.preserved_q3_verifier import VERIFIER_Q3_MODEL, VERIFIER_Q3_PARAMETERS
from experiments.preserve_evidence import publication, validate_publication
from experiments.query_coverage import (
    build_coverage_publication,
    claim_inventory,
    validate_coverage_payload,
)
from experiments.validate_reranker import canonical_hash
from rag_pensiones.chunking import ChunkRecord
from rag_pensiones.grounding import DEFAULT_PARAMETERS
from rag_pensiones.ingest import PROJECT_ROOT
from rag_pensiones.search import SearchHit


SYNTHETIC_PAGE_TEXT = (
    "Opinión sobre la sostenibilidad de las AA. PP. a largo plazo. "
    "El gasto en pensiones crecerá por encima del 4% anual hasta 2050 impulsado por el aumento del número de pensiones y de la pensión media. "
    "El crecimiento del número de pensiones se acelera hasta 2036, cuando supera el 2%. "
    "La tasa de generosidad del sistema contributivo de pensiones disminuye del 65,6% en 2023 al 56,7% en 2050 y 53,7% en 2070. "
    "Por lo tanto, la tasa de generosidad, que es un indicador de suficiencia calculado con datos agregados, volvería a niveles próximos a los alcanzados en 2015. "
    "Esta trayectoria es el resultado de la diferente evolución de las pensiones frente a los salarios: mientras que las primeras crecen con el IPC, "
    "los segundos lo hacen con la productividad y el IPC. "
    "Aunque el sistema está cerrado a nuevos entrantes desde 2011, al inicio del horizonte de proyección la ratio sobre PIB se mantiene e incluso aumenta "
    "debido a que sigue aumentando el número de pensiones y la pensión media."
)

QUOTE_B01 = "El crecimiento del número de pensiones se acelera hasta 2036, cuando supera el 2%."
QUOTE_B02 = "El gasto en pensiones crecerá por encima del 4% anual hasta 2050 impulsado por el aumento del número de pensiones y de la pensión media."
QUOTE_B03 = "mientras que las primeras crecen con el IPC, los segundos lo hacen con la productividad y el IPC."
QUOTE_2015 = "la tasa de generosidad, que es un indicador de suficiencia calculado con datos agregados, volvería a niveles próximos a los alcanzados en 2015."
QUOTE_B05 = "debido a que sigue aumentando el número de pensiones y la pensión media."


def _make_synthetic_environment(tmp_path: Path) -> dict[str, Path]:
    pages_path = tmp_path / "synthetic_pages.jsonl"
    page_record = {
        "doc_id": "doc1",
        "source_file": "report.pdf",
        "published_at": "2025-03-31",
        "page_number": 96,
        "text": SYNTHETIC_PAGE_TEXT,
    }
    pages_path.write_text(json.dumps(page_record, ensure_ascii=False) + "\n", encoding="utf-8")
    pages_lines = [page_record]

    ref_cases = {
        "b01": {
            "query": "¿En qué año sitúa el informe la aceleración del crecimiento del número de pensiones por encima del 2%?",
            "answerability": "answerable",
            "expected_status": "answered",
            "required_facts": ["2036", ">2%"],
            "criteria": "Año 2036 y aceleración/crecimiento del número de pensiones superior al 2%.",
        },
        "b02": {
            "query": "Antes de 2050, ¿qué ritmo anual supera el crecimiento del gasto en pensiones y qué dos variables lo impulsan?",
            "answerability": "answerable",
            "expected_status": "answered",
            "required_facts": [">4% anual hasta 2050", "aumento del número de pensiones", "pensión media"],
            "criteria": "Crecimiento del gasto supera el 4% anual hasta 2050 impulsado por el número de pensiones y la pensión media.",
        },
        "b03": {
            "query": "¿Por qué baja la tasa de generosidad si la pensión media sigue subiendo? Compara la evolución de las pensiones y la de los salarios en el modelo.",
            "answerability": "answerable",
            "expected_status": "answered",
            "required_facts": ["pensiones crecen con IPC", "salarios crecen con productividad e IPC"],
            "criteria": "Pensiones crecen con el IPC mientras que salarios crecen con productividad e IPC.",
        },
        "b04": {
            "query": "¿Con qué año histórico compara el informe los niveles futuros de la tasa de generosidad y se trata de una medida agregada o de una garantía individual?",
            "answerability": "answerable",
            "expected_status": "answered",
            "required_facts": ["2015", "indicador de suficiencia / medida agregada, no garantía individual"],
            "criteria": "Niveles próximos a 2015 e indicador de suficiencia calculado con datos agregados.",
        },
        "b05": {
            "query": "¿Por qué puede aumentar inicialmente el peso del gasto de Clases Pasivas sobre el PIB aunque el régimen esté cerrado a nuevos cotizantes?",
            "answerability": "answerable",
            "expected_status": "answered",
            "required_facts": ["aumento del número de pensiones", "aumento de la pensión media"],
            "criteria": "Aumento inicial en Clases Pasivas por el crecimiento concurrente del número de pensiones y pensión media.",
        },
        "b06": {
            "query": "Según estas páginas, ¿qué porcentaje del PIB de Cataluña supondrá el gasto en pensiones en 2050?",
            "answerability": "outside",
            "expected_status": "insufficient_evidence",
            "required_facts": [],
            "must_abstain_query_spans": ["PIB de Cataluña"],
            "criteria": "Abstención explícita y acotada al alcance evaluado.",
        },
        "b07": {
            "query": "¿Cuántas mujeres cobrarán una pensión contributiva en 2050 según esta proyección?",
            "answerability": "outside",
            "expected_status": "insufficient_evidence",
            "required_facts": [],
            "must_abstain_query_spans": ["mujeres"],
            "criteria": "Abstención explícita y acotada.",
        },
        "b08": {
            "query": "¿Con qué año histórico se comparan los niveles futuros de generosidad y qué pensión mensual exacta cobraré si me jubilo en 2031?",
            "answerability": "mixed",
            "expected_status": "partial",
            "required_facts": ["2015"],
            "must_abstain_query_spans": ["2031"],
            "criteria": "Consulta mixta obligatoria: afirmación principal con 2015 y rechazo individual 2031.",
        },
        "n01": {
            "query": "Según estas páginas, ¿cuántos hombres recibirán una pensión contributiva en 2032?",
            "answerability": "outside",
            "expected_status": "insufficient_evidence",
            "required_facts": [],
            "must_abstain_query_spans": ["hombres"],
            "criteria": "Abstención explícita por ausencia de hombres en 2032.",
        },
        "n02": {
            "query": "¿A qué año pasado se aproximaría de nuevo el indicador agregado de generosidad?",
            "answerability": "answerable",
            "expected_status": "answered",
            "required_facts": ["2015"],
            "main_answer_requirement": "preferred",
            "criteria": "Año 2015 respaldado; preferido en texto principal, aceptable en citas con advertencia.",
        },
        "n03": {
            "query": "Indica el año histórico comparable de generosidad y qué pensión mensual exacta me correspondería si me jubilo en 2030.",
            "answerability": "mixed",
            "expected_status": "partial",
            "required_facts": ["2015"],
            "must_abstain_query_spans": ["2030"],
            "criteria": "Consulta mixta obligatoria: 2015 en texto principal con citas y rechazo individual 2030.",
        },
    }
    ref_path = tmp_path / "synthetic_reference.json"
    ref_data = {
        "schema_version": REFERENCE_SCHEMA_VERSION,
        "created_at": "2026-10-27",
        "description": "Synthetic reference ledger for unit tests.",
        "roster": list(EXPECTED_CASE_IDS),
        "cases": ref_cases,
    }
    ref_path.write_text(json.dumps(ref_data, ensure_ascii=False), encoding="utf-8")

    queries_path = tmp_path / "synthetic_queries.json"
    queries_data = {
        "cases": [{"id": cid, "query": ref_cases[cid]["query"]} for cid in EXPECTED_CASE_IDS]
    }
    queries_path.write_text(json.dumps(queries_data, ensure_ascii=False), encoding="utf-8")

    index_path = tmp_path / "synthetic_index.json"
    index_data = {"model": "qwen3-embedding:4b", "model_digest": PINNED_EMBEDDING_DIGEST, "chunks": []}
    index_path.write_text(json.dumps(index_data, ensure_ascii=False), encoding="utf-8")

    units_path = tmp_path / "synthetic_units.json"
    units_data = {"units": []}
    units_path.write_text(json.dumps(units_data, ensure_ascii=False), encoding="utf-8")

    rel_pgs = str(pages_path)
    rel_qry = str(queries_path)
    rel_idx = str(index_path)
    rel_unt = str(units_path)

    covered_sources = {p: "a" * 64 for p in COVERED_MANDATORY_SOURCE_PATHS}
    covered_sources.update({
        rel_pgs: file_sha256(pages_path),
        rel_qry: file_sha256(queries_path),
        rel_idx: file_sha256(index_path),
        rel_unt: file_sha256(units_path),
    })

    proto_path = tmp_path / "protocol.json"
    proto_data = {
        "schema_version": PROTOCOL_SCHEMA_VERSION,
        "server_version": REQUIRED_SERVER_VERSION,
        "generator_digest": PINNED_9B_DIGEST,
        "verifier_digest": PINNED_27B_DIGEST,
        "embedding_digest": PINNED_EMBEDDING_DIGEST,
        "generator_parameters": copy.deepcopy(DEFAULT_PARAMETERS),
        "verifier_parameters": copy.deepcopy(VERIFIER_Q3_PARAMETERS),
        "index_path": rel_idx,
        "queries_path": rel_qry,
        "units_path": rel_unt,
        "pages_path": rel_pgs,
        "frozen_files": covered_sources,
        "input_hashes": {
            "pages": canonical_hash(pages_lines),
            "queries": canonical_hash(queries_data),
            "index": canonical_hash(index_data),
            "units": canonical_hash(units_data),
        },
    }
    proto_path.write_text(json.dumps(proto_data, indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        "pages": pages_path,
        "queries": queries_path,
        "index": index_path,
        "units": units_path,
        "reference": ref_path,
        "protocol": proto_path,
    }


def _build_synthetic_case(
    cid: str,
    query: str,
    *,
    claim_text: str | None = None,
    quote: str | None = None,
    source_id: str = "S1",
    parts: list[dict[str, Any]] | None = None,
    page_number: int = 96,
    doc_id: str = "doc1",
    source_file: str = "report.pdf",
    status_override: str | None = None,
    force_insufficient_evidence: bool = False,
    generator_digest: str = PINNED_9B_DIGEST,
    generator_model: str = "qwen3.5:9b",
) -> dict[str, Any]:
    hits_raw = [
        {
            "rank": 1,
            "score": 0.9,
            "source_id": source_id,
            "page_number": page_number,
            "chunk": {
                "chunk_id": "c1",
                "doc_id": doc_id,
                "source_file": source_file,
                "published_at": "2025-03-31",
                "page_number": page_number,
                "start_char": 0,
                "end_char": len(SYNTHETIC_PAGE_TEXT),
                "text": SYNTHETIC_PAGE_TEXT,
                "strategy": "semantic",
            },
        }
    ]

    is_abstain = force_insufficient_evidence or ((cid in ("b06", "b07", "n01")) and (claim_text is None))
    is_partial = (not is_abstain) and (cid in ("b08", "n03"))

    t_stat = {
        "prompt_eval_count": 100,
        "eval_count": 50,
        "eval_duration": 1000000,
        "total_duration": 2000000,
        "load_duration": 0,
        "prompt_eval_duration": 0,
    }

    if is_abstain:
        trace = [
            {
                "stage": "selection",
                "model": generator_model,
                "model_digest": generator_digest,
                "prompt_sha256": "1" * 64,
                "stats": copy.deepcopy(t_stat),
                "parameters": copy.deepcopy(DEFAULT_PARAMETERS),
                "messages_budget_bytes": 6500,
            }
        ]
        agg_hash = hashlib.sha256(json.dumps(["1" * 64], separators=(",", ":")).encode("utf-8")).hexdigest()
        full_result = {
            "status": "insufficient_evidence",
            "claims": [],
            "reason": f"Las evidencias evaluadas no contienen datos para responder a {cid}.",
            "digest": generator_digest,
            "prompt_sha256": agg_hash,
            "stats": copy.deepcopy(t_stat),
            "grounding": {
                "schema_version": "rag-grounding-v1",
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
                "verification_contexts": {"initial": [], "repair": []},
            },
        }
        case_parts = parts or [
            {"query_span": query, "decision": "not_answered", "claim_ids": [], "reason_code": "not_in_evaluated_evidence"}
        ]
    else:
        c_text = claim_text or "Afirmación sintética fundamentada."
        q_text = quote or QUOTE_B01
        trace = [
            {
                "stage": "selection",
                "model": generator_model,
                "model_digest": generator_digest,
                "prompt_sha256": "1" * 64,
                "stats": copy.deepcopy(t_stat),
                "parameters": copy.deepcopy(DEFAULT_PARAMETERS),
                "messages_budget_bytes": 6500,
            },
            {
                "stage": "generation",
                "model": generator_model,
                "model_digest": generator_digest,
                "prompt_sha256": "2" * 64,
                "stats": copy.deepcopy(t_stat),
                "parameters": copy.deepcopy(DEFAULT_PARAMETERS),
                "messages_budget_bytes": 6500,
            },
            {
                "stage": "verification",
                "model": generator_model,
                "model_digest": generator_digest,
                "prompt_sha256": "3" * 64,
                "stats": copy.deepcopy(t_stat),
                "parameters": copy.deepcopy(DEFAULT_PARAMETERS),
                "messages_budget_bytes": 12288,
            },
        ]
        agg_hash = hashlib.sha256(json.dumps(["1" * 64, "2" * 64, "3" * 64], separators=(",", ":")).encode("utf-8")).hexdigest()
        full_stats = {
            "prompt_eval_count": 300,
            "eval_count": 150,
            "eval_duration": 3000000,
            "total_duration": 6000000,
            "load_duration": 0,
            "prompt_eval_duration": 0,
        }
        full_result = {
            "status": "answered",
            "claims": [
                {"text": c_text, "evidence": [{"source_id": source_id, "quote": q_text}]}
            ],
            "reason": "",
            "digest": generator_digest,
            "prompt_sha256": agg_hash,
            "stats": full_stats,
            "grounding": {
                "schema_version": "rag-grounding-v1",
                "selected_evidence": [
                    {
                        "evidence_id": "E1",
                        "source_id": source_id,
                        "doc_id": doc_id,
                        "source_file": source_file,
                        "page_number": page_number,
                        "quote": q_text,
                    }
                ],
                "initial_claims": [
                    {"claim_id": "C1", "text": c_text, "evidence_ids": ["E1"]}
                ],
                "initial_verdicts": [
                    {
                        "claim_id": "C1",
                        "verdict": "supported",
                        "reason": "OK",
                        "evidence_verdicts": [{"evidence_id": "E1", "relation": "supports"}],
                    }
                ],
                "repair_attempts": 0,
                "repair_evidence": [],
                "repairs": [],
                "final_verdicts": [
                    {
                        "claim_id": "C1",
                        "verdict": "supported",
                        "reason": "OK",
                        "evidence_verdicts": [{"evidence_id": "E1", "relation": "supports"}],
                    }
                ],
                "accepted_claims": [
                    {"claim_id": "C1", "text": c_text, "evidence_ids": ["E1"]}
                ],
                "rejected_claims": [],
                "trace": trace,
                "verification_contexts": {"initial": [], "repair": []},
            },
        }

        if is_partial:
            if cid == "b08":
                span1 = "¿Con qué año histórico se comparan los niveles futuros de generosidad"
                span2 = "y qué pensión mensual exacta cobraré si me jubilo en 2031?"
            else:
                span1 = "Indica el año histórico comparable de generosidad"
                span2 = "y qué pensión mensual exacta me correspondería si me jubilo en 2030."
            case_parts = parts or [
                {"query_span": span1, "decision": "answered", "claim_ids": ["C1"], "reason_code": "covered"},
                {"query_span": span2, "decision": "not_answered", "claim_ids": [], "reason_code": "not_in_evaluated_evidence"},
            ]
        else:
            case_parts = parts or [
                {"query_span": query, "decision": "answered", "claim_ids": ["C1"], "reason_code": "covered"}
            ]

    factual_pub = publication(full_result, preserve=True, allow_partial=True)
    inv = claim_inventory({"outcome": {"full_result": full_result, "publication": factual_pub}})
    clean_parts = [
        {
            "query_span": p["query_span"],
            "decision": p["decision"],
            "claim_ids": p["claim_ids"],
            "reason_code": p["reason_code"],
        }
        for p in case_parts
    ]
    val_payload = validate_coverage_payload({"parts": clean_parts}, query, inv)
    cov_pub = build_coverage_publication(query, inv, val_payload)

    status_val = status_override or cov_pub["status"]

    answer = {
        "schema_version": "preserved-query-coverage-v1",
        "status": status_val,
        "query": query,
        "factual_report": {
            "schema_version": "preserved-direct-query-v1",
            "query": query,
            "outcome": {
                "status": full_result["status"],
                "full_result": full_result,
                "publication": factual_pub,
                "expanded_hits": hits_raw,
                "verification_policy": None,
            },
            "retrieval": {"hits": hits_raw},
            "metadata": {},
        },
        "coverage_audit": {
            "status": "completed",
            "validated_payload": val_payload,
        },
        "publication": cov_pub,
        "metadata": {
            "profile": "preserved_query_coverage_v1",
            "verifier_model": "hf.co/gguf/Qwen3.8-27B-GGUF:Q3_K_M",
            "verifier_digest": PINNED_27B_DIGEST,
        },
    }

    return {
        "id": cid,
        "query": query,
        "attempted": True,
        "status": "completed",
        "wall_seconds": 1.0,
        "answer": answer,
        "formatted_answer": "Formatted answer text",
    }


def _build_synthetic_passed_results(
    proto_path: Path,
    env: dict[str, Path],
    *,
    generator_digest: str = PINNED_9B_DIGEST,
    generator_model: str = "qwen3.5:9b",
) -> dict[str, Any]:
    proto_data = strict_load_json(proto_path.read_bytes())
    ref_data = strict_load_json(env["reference"].read_bytes())

    cases: list[dict[str, Any]] = []

    # b01
    q_b01 = ref_data["cases"]["b01"]["query"]
    cases.append(_build_synthetic_case(
        "b01", q_b01,
        claim_text="El informe sitúa la aceleración del crecimiento del número de pensiones por encima del 2% en el año 2036.",
        quote=QUOTE_B01,
        generator_digest=generator_digest,
        generator_model=generator_model,
    ))

    # b02
    q_b02 = ref_data["cases"]["b02"]["query"]
    cases.append(_build_synthetic_case(
        "b02", q_b02,
        claim_text="El gasto en pensiones crecerá por encima del 4% anual hasta 2050 impulsado por el aumento del número de pensiones y de la pensión media.",
        quote=QUOTE_B02,
        generator_digest=generator_digest,
        generator_model=generator_model,
    ))

    # b03
    q_b03 = ref_data["cases"]["b03"]["query"]
    cases.append(_build_synthetic_case(
        "b03", q_b03,
        claim_text="Las pensiones crecen con el IPC mientras que los salarios aumentan con la productividad y el IPC.",
        quote=QUOTE_B03,
        generator_digest=generator_digest,
        generator_model=generator_model,
    ))

    # b04
    q_b04 = ref_data["cases"]["b04"]["query"]
    cases.append(_build_synthetic_case(
        "b04", q_b04,
        claim_text="La tasa de generosidad volvería a niveles próximos a 2015 según el indicador de suficiencia calculado con datos agregados.",
        quote=QUOTE_2015,
        generator_digest=generator_digest,
        generator_model=generator_model,
    ))

    # b05
    q_b05 = ref_data["cases"]["b05"]["query"]
    cases.append(_build_synthetic_case(
        "b05", q_b05,
        claim_text="Aumenta debido a que sigue aumentando el número de pensiones y la pensión media.",
        quote=QUOTE_B05,
        generator_digest=generator_digest,
        generator_model=generator_model,
    ))

    # b06 (abstain)
    q_b06 = ref_data["cases"]["b06"]["query"]
    cases.append(_build_synthetic_case(
        "b06", q_b06,
        generator_digest=generator_digest,
        generator_model=generator_model,
    ))

    # b07 (abstain)
    q_b07 = ref_data["cases"]["b07"]["query"]
    cases.append(_build_synthetic_case(
        "b07", q_b07,
        generator_digest=generator_digest,
        generator_model=generator_model,
    ))

    # b08 (partial)
    q_b08 = ref_data["cases"]["b08"]["query"]
    cases.append(_build_synthetic_case(
        "b08", q_b08,
        claim_text="Se compara con los niveles del año 2015.",
        quote=QUOTE_2015,
        generator_digest=generator_digest,
        generator_model=generator_model,
    ))

    # n01 (abstain)
    q_n01 = ref_data["cases"]["n01"]["query"]
    cases.append(_build_synthetic_case(
        "n01", q_n01,
        generator_digest=generator_digest,
        generator_model=generator_model,
    ))

    # n02 (quote-only debt baseline: 2015 in quote, omitted from main claim text)
    q_n02 = ref_data["cases"]["n02"]["query"]
    cases.append(_build_synthetic_case(
        "n02", q_n02,
        claim_text="El indicador agregado de generosidad desciende a lo largo de las proyecciones.",
        quote=QUOTE_2015,
        generator_digest=generator_digest,
        generator_model=generator_model,
    ))

    # n03 (partial)
    q_n03 = ref_data["cases"]["n03"]["query"]
    cases.append(_build_synthetic_case(
        "n03", q_n03,
        claim_text="El año histórico comparable es 2015.",
        quote=QUOTE_2015,
        generator_digest=generator_digest,
        generator_model=generator_model,
    ))

    return {
        "schema_version": COVERED_RESULTS_SCHEMA_VERSION,
        "run_status": "completed",
        "protocol_path": str(proto_path),
        "metadata": {
            "protocol_hash": canonical_hash(proto_data),
        },
        "cases": cases,
    }


def _build_synthetic_historical_environment(
    tmp_path: Path,
    env: dict[str, Path],
    *,
    arm: str = "mixed",
) -> tuple[Path, Path]:
    rel_pgs = str(env["pages"])
    rel_qry = str(env["queries"])
    rel_idx = str(env["index"])
    rel_unt = str(env["units"])

    hist_sources = {p: "b" * 64 for p in HISTORICAL_MANDATORY_SOURCE_PATHS}
    hist_sources.update({
        rel_pgs: file_sha256(env["pages"]),
        rel_qry: file_sha256(env["queries"]),
        rel_idx: file_sha256(env["index"]),
        rel_unt: file_sha256(env["units"]),
    })

    proto_path = tmp_path / f"hist_protocol_{arm}.json"
    proto_data = {
        "schema_version": HISTORICAL_PROTOCOL_SCHEMA_VERSION,
        "server_version": REQUIRED_SERVER_VERSION,
        "models": {
            "generator_9b": {"name": "qwen3.5:9b", "digest": PINNED_9B_DIGEST},
            "text_27b": {"name": VERIFIER_Q3_MODEL, "digest": PINNED_27B_DIGEST},
            "embedding_4b": {"name": "qwen3-embedding:4b", "digest": PINNED_EMBEDDING_DIGEST},
        },
        "effective_generation_options": copy.deepcopy(HISTORICAL_EFFECTIVE_GENERATION_OPTIONS),
        "canonical_role_parameters": copy.deepcopy(HISTORICAL_CANONICAL_ROLE_PARAMETERS),
        "verifier_parameters": copy.deepcopy(VERIFIER_Q3_PARAMETERS),
        "paths": {
            "pages": rel_pgs,
            "queries": rel_qry,
            "index": rel_idx,
            "units": rel_unt,
        },
        "frozen_files": hist_sources,
        "input_hashes": {
            "pages": file_sha256(env["pages"]),
            "queries": file_sha256(env["queries"]),
            "index": file_sha256(env["index"]),
            "units": file_sha256(env["units"]),
        },
    }
    proto_path.write_text(json.dumps(proto_data, indent=2), encoding="utf-8")

    generator_digest = PINNED_9B_DIGEST if arm == "mixed" else PINNED_27B_DIGEST
    generator_model = "qwen3.5:9b" if arm == "mixed" else VERIFIER_Q3_MODEL
    passed_covered = _build_synthetic_passed_results(
        env["protocol"],
        env,
        generator_digest=generator_digest,
        generator_model=generator_model,
    )
    attempts: list[dict[str, Any]] = []
    for c in passed_covered["cases"]:
        cid = c["id"]
        q = c["query"]
        ans = c["answer"]
        f_rep = ans["factual_report"]
        cov_audit = ans["coverage_audit"]
        cov_pub = ans["publication"]
        f_out = f_rep["outcome"]

        att = {
            "case_id": cid,
            "arm": arm,
            "query": q,
            "status": "completed",
            "attempted": True,
            "query_wall_seconds": 1.0,
            "formatted_answer": c.get("formatted_answer", "Formatted answer text"),
            "factual_outcome": copy.deepcopy(f_out),
            "publication": copy.deepcopy(cov_pub),
            "coverage_audit": copy.deepcopy(cov_audit),
            "coverage_publication": copy.deepcopy(cov_pub),
        }
        attempts.append(att)

    res_path = tmp_path / f"hist_results_{arm}.json"
    res_data = {
        "schema_version": HISTORICAL_RESULTS_SCHEMA_VERSION,
        "run_status": "completed",
        "protocol_path": str(proto_path),
        "protocol_sha256": file_sha256(proto_path),
        "attempts": attempts,
    }
    res_path.write_text(json.dumps(res_data, indent=2), encoding="utf-8")
    return res_path, proto_path


class TestContinuousQualityGate(unittest.TestCase):
    """Test suite ensuring strict bounded regression gating and integrity bindings."""

    def setUp(self) -> None:
        self.temp_dir_obj = tempfile.TemporaryDirectory()
        self.tmp_dir = Path(self.temp_dir_obj.name)
        self.env = _make_synthetic_environment(self.tmp_dir)

    def tearDown(self) -> None:
        self.temp_dir_obj.cleanup()

    def test_quote_only_n02_passes_whole_with_warning(self) -> None:
        results = _build_synthetic_passed_results(self.env["protocol"], self.env)
        res_file = self.tmp_dir / "results_pass.json"
        res_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
        out_file = self.tmp_dir / "report_pass.json"

        rep = check(res_file, self.env["reference"], out_file)
        self.assertTrue(rep["overall_pass"])
        self.assertEqual(rep["status"], "pass")
        self.assertEqual(rep["summary"]["total_cases"], EXPECTED_CASE_COUNT)
        self.assertEqual(rep["summary"]["whole_correct_count"], EXPECTED_CASE_COUNT)
        self.assertEqual(rep["summary"]["main_direct_count"], EXPECTED_CASE_COUNT - 1)
        self.assertEqual(rep["summary"]["safe_count"], EXPECTED_CASE_COUNT)
        self.assertIn("n02_quote_only_debt", rep["summary"]["warnings"])

        n02_res = next(c for c in rep["case_results"] if c["case_id"] == "n02")
        self.assertTrue(n02_res["whole_correct"])
        self.assertFalse(n02_res["main_direct_correct"])
        self.assertFalse(n02_res["direct"])
        self.assertTrue(n02_res["safe"])

    def test_quote_only_b01_fails_whole(self) -> None:
        results = _build_synthetic_passed_results(self.env["protocol"], self.env)
        ref_q = strict_load_json(self.env["reference"].read_bytes())["cases"]["b01"]["query"]
        b01_case = _build_synthetic_case(
            "b01", ref_q,
            claim_text="El informe no precisa el año exacto de la aceleración del crecimiento de pensiones.",
            quote=QUOTE_B01,
        )
        results["cases"][0] = b01_case
        res_file = self.tmp_dir / "results_b01_quote_only.json"
        res_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
        out_file = self.tmp_dir / "report_b01_qo.json"

        rep = check(res_file, self.env["reference"], out_file)
        self.assertFalse(rep["overall_pass"])
        b01_res = next(c for c in rep["case_results"] if c["case_id"] == "b01")
        self.assertFalse(b01_res["whole_correct"])
        self.assertIn("b01_main_missing_2036_or_2pct", b01_res["failures"])

    def test_b02_missing_number_of_pensions_fails(self) -> None:
        results = _build_synthetic_passed_results(self.env["protocol"], self.env)
        ref_q = strict_load_json(self.env["reference"].read_bytes())["cases"]["b02"]["query"]
        b02_case = _build_synthetic_case(
            "b02", ref_q,
            claim_text="El gasto en pensiones crecerá por encima del 4% anual hasta 2050 impulsado por la pensión media.",
            quote=QUOTE_B02,
        )
        results["cases"][1] = b02_case
        res_file = self.tmp_dir / "results_b02_missing_num.json"
        res_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
        out_file = self.tmp_dir / "report_b02_mn.json"

        rep = check(res_file, self.env["reference"], out_file)
        self.assertFalse(rep["overall_pass"])
        b02_res = next(c for c in rep["case_results"] if c["case_id"] == "b02")
        self.assertFalse(b02_res["whole_correct"])
        self.assertIn("b02_main_missing_number_or_mean", b02_res["failures"])

    def test_b02_missing_2050_fails(self) -> None:
        results = _build_synthetic_passed_results(self.env["protocol"], self.env)
        ref_q = strict_load_json(self.env["reference"].read_bytes())["cases"]["b02"]["query"]
        b02_case = _build_synthetic_case(
            "b02", ref_q,
            claim_text="El gasto en pensiones crecerá a un ritmo anual por encima del 4% impulsado por el aumento del número de pensiones y de la pensión media.",
            quote=QUOTE_B02,
        )
        results["cases"][1] = b02_case
        res_file = self.tmp_dir / "results_b02_missing_2050.json"
        res_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
        out_file = self.tmp_dir / "report_b02_m50.json"

        rep = check(res_file, self.env["reference"], out_file)
        self.assertFalse(rep["overall_pass"])
        b02_res = next(c for c in rep["case_results"] if c["case_id"] == "b02")
        self.assertFalse(b02_res["whole_correct"])
        self.assertIn("b02_main_missing_2050", b02_res["failures"])

    def test_inverted_relation_b01_fails(self) -> None:
        results = _build_synthetic_passed_results(self.env["protocol"], self.env)
        ref_q = strict_load_json(self.env["reference"].read_bytes())["cases"]["b01"]["query"]
        b01_case = _build_synthetic_case(
            "b01", ref_q,
            claim_text="El crecimiento del número de pensiones se sitúa por debajo del 2% en el año 2036.",
            quote=QUOTE_B01,
        )
        results["cases"][0] = b01_case
        res_file = self.tmp_dir / "results_b01_inv.json"
        res_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
        out_file = self.tmp_dir / "report_b01_inv.json"

        rep = check(res_file, self.env["reference"], out_file)
        self.assertFalse(rep["overall_pass"])
        b01_res = next(c for c in rep["case_results"] if c["case_id"] == "b01")
        self.assertFalse(b01_res["whole_correct"])
        self.assertTrue(any("inverted_relation" in f or "missing_growth_above" in f for f in b01_res["failures"]))

    def test_inverted_relation_b02_fails(self) -> None:
        results = _build_synthetic_passed_results(self.env["protocol"], self.env)
        ref_q = strict_load_json(self.env["reference"].read_bytes())["cases"]["b02"]["query"]
        b02_case = _build_synthetic_case(
            "b02", ref_q,
            claim_text="El gasto en pensiones crecerá por debajo del 4% anual hasta 2050 impulsado por el aumento del número de pensiones y de la pensión media.",
            quote=QUOTE_B02,
        )
        results["cases"][1] = b02_case
        res_file = self.tmp_dir / "results_b02_inv.json"
        res_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
        out_file = self.tmp_dir / "report_b02_inv.json"

        rep = check(res_file, self.env["reference"], out_file)
        self.assertFalse(rep["overall_pass"])
        b02_res = next(c for c in rep["case_results"] if c["case_id"] == "b02")
        self.assertFalse(b02_res["whole_correct"])
        self.assertTrue(any("inverted_relation" in f or "missing_growth_above" in f for f in b02_res["failures"]))

    def test_swapped_attribution_b03_fails(self) -> None:
        results = _build_synthetic_passed_results(self.env["protocol"], self.env)
        ref_q = strict_load_json(self.env["reference"].read_bytes())["cases"]["b03"]["query"]
        b03_case = _build_synthetic_case(
            "b03", ref_q,
            claim_text="Las pensiones crecen con la productividad mientras que los salarios aumentan con el IPC.",
            quote=QUOTE_B03,
        )
        results["cases"][2] = b03_case
        res_file = self.tmp_dir / "results_b03_swapped.json"
        res_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
        out_file = self.tmp_dir / "report_b03_swapped.json"

        rep = check(res_file, self.env["reference"], out_file)
        self.assertFalse(rep["overall_pass"])
        b03_res = next(c for c in rep["case_results"] if c["case_id"] == "b03")
        self.assertFalse(b03_res["whole_correct"])
        self.assertIn("b03_main_swapped_or_missing_attribution", b03_res["failures"])

    def test_b04_valid_sufficiency_paraphrase_passes(self) -> None:
        results = _build_synthetic_passed_results(self.env["protocol"], self.env)
        ref_q = strict_load_json(self.env["reference"].read_bytes())["cases"]["b04"]["query"]
        b04_case = _build_synthetic_case(
            "b04", ref_q,
            claim_text="La tasa de generosidad volvería a niveles próximos a 2015 según el indicador de suficiencia, no siendo una garantía individual.",
            quote=QUOTE_2015,
        )
        results["cases"][3] = b04_case
        res_file = self.tmp_dir / "results_b04_para.json"
        res_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
        out_file = self.tmp_dir / "report_b04_para.json"

        rep = check(res_file, self.env["reference"], out_file)
        self.assertTrue(rep["overall_pass"])
        b04_res = next(c for c in rep["case_results"] if c["case_id"] == "b04")
        self.assertTrue(b04_res["whole_correct"])
        self.assertTrue(b04_res["main_direct_correct"])

    def test_n02_empty_claim_fails_whole(self) -> None:
        # 1. Evaluator rejects empty claim text directly
        diag = evaluate_case_reference(
            "n02",
            {"status": "answered", "claims": [{"text": "", "evidence": [{"source_id": "S1", "quote": QUOTE_2015}]}]},
            {"S1": {"source_id": "S1", "page_number": 96, "doc_id": "doc1", "source_file": "report.pdf"}},
            {96: {"doc_id": "doc1", "source_file": "report.pdf", "text": SYNTHETIC_PAGE_TEXT}},
        )
        self.assertFalse(diag["whole_correct"])
        self.assertIn("claim_0_empty_text", diag["failures"])

        # 2. Canonical check rejects structurally invalid payload
        results = _build_synthetic_passed_results(self.env["protocol"], self.env)
        n02 = next(c for c in results["cases"] if c["id"] == "n02")
        n02["answer"]["publication"]["claims"][0]["text"] = ""
        res_file = self.tmp_dir / "results_n02_empty.json"
        res_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
        out_file = self.tmp_dir / "report_n02_empty.json"

        with self.assertRaises(ValueError):
            check(res_file, self.env["reference"], out_file)

    def test_n02_invalid_cite_fails_whole(self) -> None:
        # 1. Evaluator rejects unknown source id directly
        diag = evaluate_case_reference(
            "n02",
            {"status": "answered", "claims": [{"text": "Texto", "evidence": [{"source_id": "UNKNOWN_SOURCE", "quote": QUOTE_2015}]}]},
            {"S1": {"source_id": "S1", "page_number": 96, "doc_id": "doc1", "source_file": "report.pdf"}},
            {96: {"doc_id": "doc1", "source_file": "report.pdf", "text": SYNTHETIC_PAGE_TEXT}},
        )
        self.assertFalse(diag["whole_correct"])
        self.assertFalse(diag["safe"])
        self.assertTrue(any("unknown_source_id" in f for f in diag["failures"]))

        # 2. Canonical check rejects structurally invalid cite payload
        results = _build_synthetic_passed_results(self.env["protocol"], self.env)
        n02 = next(c for c in results["cases"] if c["id"] == "n02")
        n02["answer"]["publication"]["claims"][0]["evidence"][0]["source_id"] = "UNKNOWN_SOURCE"
        res_file = self.tmp_dir / "results_n02_badcite.json"
        res_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
        out_file = self.tmp_dir / "report_n02_badcite.json"

        with self.assertRaises(ValueError):
            check(res_file, self.env["reference"], out_file)

    def test_n02_missing_2015_everywhere_fails(self) -> None:
        results = _build_synthetic_passed_results(self.env["protocol"], self.env)
        ref_q = strict_load_json(self.env["reference"].read_bytes())["cases"]["n02"]["query"]
        n02_case = _build_synthetic_case(
            "n02", ref_q,
            claim_text="El indicador agregado desciende en las proyecciones.",
            quote="Opinión sobre la sostenibilidad de las AA. PP. a largo plazo.",
        )
        idx = next(i for i, c in enumerate(results["cases"]) if c["id"] == "n02")
        results["cases"][idx] = n02_case
        res_file = self.tmp_dir / "results_n02_m15.json"
        res_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
        out_file = self.tmp_dir / "report_n02_m15.json"

        rep = check(res_file, self.env["reference"], out_file)
        self.assertFalse(rep["overall_pass"])
        n02_res = next(c for c in rep["case_results"] if c["case_id"] == "n02")
        self.assertFalse(n02_res["whole_correct"])
        self.assertIn("n02_missing_2015", n02_res["failures"])

    def test_missing_partial_unknown_part_refusal_fails(self) -> None:
        results = _build_synthetic_passed_results(self.env["protocol"], self.env)
        ref_q = strict_load_json(self.env["reference"].read_bytes())["cases"]["b08"]["query"]
        b08_case = _build_synthetic_case(
            "b08", ref_q,
            claim_text="Se compara con 2015.",
            quote=QUOTE_2015,
            parts=[{"query_span": ref_q, "decision": "answered", "claim_ids": ["C1"], "reason_code": "covered"}],
        )
        idx = next(i for i, c in enumerate(results["cases"]) if c["id"] == "b08")
        results["cases"][idx] = b08_case
        res_file = self.tmp_dir / "results_b08_no_refusal.json"
        res_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
        out_file = self.tmp_dir / "report_b08_nr.json"

        rep = check(res_file, self.env["reference"], out_file)
        self.assertFalse(rep["overall_pass"])
        b08_res = next(c for c in rep["case_results"] if c["case_id"] == "b08")
        self.assertFalse(b08_res["whole_correct"])
        self.assertTrue(any("status_mismatch" in f or "missing_individual_refusal" in f for f in b08_res["failures"]))

    def test_outside_wrong_status_fails_safe(self) -> None:
        # 1. Evaluator rejects non-insufficient_evidence status for outside case
        diag = evaluate_case_reference(
            "b06",
            {"status": "partial", "claims": [], "reason": "No hay datos sobre Cataluña en estas páginas."},
            {},
            {96: {"doc_id": "doc1", "source_file": "report.pdf", "text": SYNTHETIC_PAGE_TEXT}},
        )
        self.assertFalse(diag["safe"])
        self.assertFalse(diag["whole_correct"])
        self.assertTrue(any("status_mismatch" in f or "failed_to_abstain" in f for f in diag["failures"]))

        # 2. Canonical check rejects mismatched coverage publication status
        results = _build_synthetic_passed_results(self.env["protocol"], self.env)
        ref_q = strict_load_json(self.env["reference"].read_bytes())["cases"]["b06"]["query"]
        b06_case = _build_synthetic_case(
            "b06", ref_q,
            status_override="partial",
        )
        b06_case["answer"]["publication"]["status"] = "partial"
        idx = next(i for i, c in enumerate(results["cases"]) if c["id"] == "b06")
        results["cases"][idx] = b06_case
        res_file = self.tmp_dir / "results_b06_partial.json"
        res_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
        out_file = self.tmp_dir / "report_b06_partial.json"

        with self.assertRaises(ValueError):
            check(res_file, self.env["reference"], out_file)

    def test_parent_proven_mutation_altered_b01_query(self) -> None:
        results = _build_synthetic_passed_results(self.env["protocol"], self.env)
        results["cases"][0]["query"] = "¿Pregunta alterada no congruente con referencia?"
        res_file = self.tmp_dir / "results_b01_altered.json"
        res_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
        out_file = self.tmp_dir / "report_b01_altered.json"

        with self.assertRaises(ValueError) as ctx:
            check(res_file, self.env["reference"], out_file)
        self.assertIn("Altered query rejected for case b01", str(ctx.exception))

    def test_parent_proven_mutation_status_error_rejected(self) -> None:
        # 1. Error string at root
        results = _build_synthetic_passed_results(self.env["protocol"], self.env)
        results["error"] = "Fatal pipeline error occurred"
        res_file = self.tmp_dir / "results_root_error.json"
        res_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
        with self.assertRaises(ValueError) as ctx:
            check(res_file, self.env["reference"], self.tmp_dir / "out1.json")
        self.assertIn("Results root contains error string", str(ctx.exception))

        # 2. Status error at case level
        results2 = _build_synthetic_passed_results(self.env["protocol"], self.env)
        results2["cases"][0]["status"] = "error"
        results2["cases"][0]["error"] = "Case execution died"
        res_file2 = self.tmp_dir / "results_case_error.json"
        res_file2.write_text(json.dumps(results2, indent=2), encoding="utf-8")
        with self.assertRaises(ValueError) as ctx2:
            check(res_file2, self.env["reference"], self.tmp_dir / "out2.json")
        self.assertIn("Technical error in case 0", str(ctx2.exception))

        # 3. Status error in covered answer
        results3 = _build_synthetic_passed_results(self.env["protocol"], self.env)
        results3["cases"][0]["answer"]["status"] = "error"
        results3["cases"][0]["answer"]["error"] = "Grounding error"
        res_file3 = self.tmp_dir / "results_ans_error.json"
        res_file3.write_text(json.dumps(results3, indent=2), encoding="utf-8")
        with self.assertRaises(ValueError) as ctx3:
            check(res_file3, self.env["reference"], self.tmp_dir / "out3.json")
        self.assertIn("Covered answer error in case b01", str(ctx3.exception))

    def test_parent_proven_mutation_fake_source_cite_rejected(self) -> None:
        # 1. Direct evaluate_case_reference validator rejects unknown source_id and marks safe=False
        diag = evaluate_case_reference(
            "b01",
            {"status": "answered", "claims": [{"text": "Texto", "evidence": [{"source_id": "FAKE_SOURCE", "quote": QUOTE_B01}]}]},
            {},
            {96: {"text": SYNTHETIC_PAGE_TEXT}},
        )
        self.assertFalse(diag["safe"])
        self.assertFalse(diag["whole_correct"])
        self.assertTrue(any("unknown_source_id: 'FAKE_SOURCE'" in f for f in diag["failures"]))

        # 2. Results payload containing tampered FAKE_SOURCE publication cite is rejected
        results = _build_synthetic_passed_results(self.env["protocol"], self.env)
        results["cases"][0]["answer"]["publication"]["claims"][0]["evidence"][0]["source_id"] = "FAKE_SOURCE"
        res_file = self.tmp_dir / "results_fake_source.json"
        res_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
        out_file = self.tmp_dir / "report_fake_source.json"

        with self.assertRaises(ValueError):
            check(res_file, self.env["reference"], out_file)

    def test_parent_proven_mutation_protocol_sha_mismatch_rejected(self) -> None:
        results = _build_synthetic_passed_results(self.env["protocol"], self.env)
        results["metadata"]["protocol_hash"] = "0" * 64
        res_file = self.tmp_dir / "results_zero_sha.json"
        res_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
        out_file = self.tmp_dir / "report_zero_sha.json"

        with self.assertRaises(ValueError) as ctx:
            check(res_file, self.env["reference"], out_file)
        self.assertIn("Covered results metadata protocol_hash mismatch", str(ctx.exception))

    def test_quote_in_wrong_canonical_page_rejected(self) -> None:
        diag = evaluate_case_reference(
            "b01",
            {"status": "answered", "claims": [{"text": "Texto", "evidence": [{"source_id": "S1", "quote": QUOTE_B01}]}]},
            {"S1": {"source_id": "S1", "page_number": 97, "doc_id": "doc1", "source_file": "report.pdf"}},
            {96: {"text": SYNTHETIC_PAGE_TEXT}},
        )
        self.assertFalse(diag["whole_correct"])
        self.assertTrue(any("page_number_97_not_in_canonical_pages" in f for f in diag["failures"]))

        diag2 = evaluate_case_reference(
            "b01",
            {"status": "answered", "claims": [{"text": "Texto", "evidence": [{"source_id": "S1", "quote": QUOTE_B01}]}]},
            {"S1": {"source_id": "S1", "page_number": 96, "doc_id": "wrong_doc", "source_file": "report.pdf"}},
            {96: {"doc_id": "doc1", "source_file": "report.pdf", "text": SYNTHETIC_PAGE_TEXT}},
        )
        self.assertFalse(diag2["whole_correct"])
        self.assertTrue(any("doc_id_mismatch_for_S1" in f for f in diag2["failures"]))

    def test_altered_quote_not_in_canonical_page_fails(self) -> None:
        diag = evaluate_case_reference(
            "b01",
            {"status": "answered", "claims": [{"text": "Texto", "evidence": [{"source_id": "S1", "quote": "Esta frase inventada no existe."}]}]},
            {"S1": {"source_id": "S1", "page_number": 96, "doc_id": "doc1", "source_file": "report.pdf"}},
            {96: {"doc_id": "doc1", "source_file": "report.pdf", "text": SYNTHETIC_PAGE_TEXT}},
        )
        self.assertFalse(diag["whole_correct"])
        self.assertTrue(any("quote_not_in_canonical_page_96" in f for f in diag["failures"]))

        results = _build_synthetic_passed_results(self.env["protocol"], self.env)
        results["cases"][0]["answer"]["publication"]["claims"][0]["evidence"][0]["quote"] = "Esta frase inventada no existe."
        res_file = self.tmp_dir / "results_bad_quote.json"
        res_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
        out_file = self.tmp_dir / "report_bad_quote.json"

        with self.assertRaises(ValueError):
            check(res_file, self.env["reference"], out_file)

    def test_coverage_publication_tampering_rejected(self) -> None:
        results = _build_synthetic_passed_results(self.env["protocol"], self.env)
        b08 = next(c for c in results["cases"] if c["id"] == "b08")
        b08["answer"]["publication"]["status"] = "answered"
        b08["answer"]["publication"]["missing_parts"] = []

        res_file = self.tmp_dir / "results_tampered_pub.json"
        res_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
        out_file = self.tmp_dir / "report_tampered.json"

        with self.assertRaises(ValueError) as ctx:
            check(res_file, self.env["reference"], out_file)
        self.assertIn("Coverage publication mismatch in case b08", str(ctx.exception))

    def test_accepted_claim_text_mismatch_rejected(self) -> None:
        results = _build_synthetic_passed_results(self.env["protocol"], self.env)
        b01 = results["cases"][0]
        b01["answer"]["factual_report"]["outcome"]["publication"]["claims"][0]["text"] = "Texto desalineado sin justificar."

        res_file = self.tmp_dir / "results_mismatched_claim.json"
        res_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
        out_file = self.tmp_dir / "report_mismatched.json"

        with self.assertRaises(ValueError) as ctx:
            check(res_file, self.env["reference"], out_file)
        self.assertIn("Publication lost package provenance", str(ctx.exception))

    def test_altered_input_bytes_rejected(self) -> None:
        results = _build_synthetic_passed_results(self.env["protocol"], self.env)
        res_file = self.tmp_dir / "results_input_mod.json"
        res_file.write_text(json.dumps(results, indent=2), encoding="utf-8")

        self.env["pages"].write_text("{\"doc_id\": \"doc1\", \"tampered\": true}\n", encoding="utf-8")

        with self.assertRaises(ValueError) as ctx:
            check(res_file, self.env["reference"], self.tmp_dir / "out_mod.json")
        self.assertIn("Pages input canonical hash mismatch", str(ctx.exception))

    def test_incomplete_or_noncompleted_run_rejected(self) -> None:
        results = _build_synthetic_passed_results(self.env["protocol"], self.env)
        results["run_status"] = "running"
        res_file = self.tmp_dir / "results_running.json"
        res_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
        with self.assertRaises(ValueError) as ctx:
            check(res_file, self.env["reference"], self.tmp_dir / "out_run.json")
        self.assertIn("Results run_status is not 'completed'", str(ctx.exception))

        results2 = _build_synthetic_passed_results(self.env["protocol"], self.env)
        results2["cases"][0]["attempted"] = False
        res_file2 = self.tmp_dir / "results_not_attempted.json"
        res_file2.write_text(json.dumps(results2, indent=2), encoding="utf-8")
        with self.assertRaises(ValueError) as ctx2:
            check(res_file2, self.env["reference"], self.tmp_dir / "out_att.json")
        self.assertIn("was not attempted", str(ctx2.exception))

    def test_non_finite_overflow_and_duplicate_keys_rejected(self) -> None:
        nan_json = b'{"case_id": "b01", "value": NaN}'
        with self.assertRaises(ValueError) as ctx:
            strict_load_json(nan_json)
        self.assertIn("Non-finite", str(ctx.exception))

        overflow_json = b'{"case_id": "b01", "value": 1e999}'
        with self.assertRaises(ValueError) as ctx2:
            strict_load_json(overflow_json)
        self.assertIn("Non-finite or overflow", str(ctx2.exception))

        dup_json = b'{"id": "b01", "id": "b02"}'
        with self.assertRaises(ValueError) as ctx3:
            strict_load_json(dup_json)
        self.assertIn("Duplicate", str(ctx3.exception))

    def test_invalid_roster_and_arm_rejected(self) -> None:
        results = _build_synthetic_passed_results(self.env["protocol"], self.env)
        results["cases"].pop()
        res_file = self.tmp_dir / "results_short_roster.json"
        res_file.write_text(json.dumps(results, indent=2), encoding="utf-8")

        with self.assertRaises(ValueError) as ctx:
            check(res_file, self.env["reference"], self.tmp_dir / "out_short.json")
        self.assertIn("Expected 11 cases", str(ctx.exception))

        results2 = _build_synthetic_passed_results(self.env["protocol"], self.env)
        res_file2 = self.tmp_dir / "results_bad_arm.json"
        res_file2.write_text(json.dumps(results2, indent=2), encoding="utf-8")
        with self.assertRaises(ValueError) as ctx2:
            check(res_file2, self.env["reference"], self.tmp_dir / "out_arm.json", arm="all27")
        self.assertIn("Arm cannot be 'all27'", str(ctx2.exception))

    def test_altered_reference_ledger_rejected(self) -> None:
        ref_data = strict_load_json(self.env["reference"].read_bytes())
        ref_data["roster"].pop()
        tampered_ref = self.tmp_dir / "tampered_ref.json"
        tampered_ref.write_text(json.dumps(ref_data, indent=2), encoding="utf-8")

        with self.assertRaises(ValueError) as ctx:
            validate_reference_ledger(ref_data)
        self.assertIn("Reference roster mismatch", str(ctx.exception))

    def test_mandatory_partial_missing_2015_fails_gate(self) -> None:
        results = _build_synthetic_passed_results(self.env["protocol"], self.env)
        ref_q = strict_load_json(self.env["reference"].read_bytes())["cases"]["b08"]["query"]
        b08_case = _build_synthetic_case(
            "b08", ref_q,
            force_insufficient_evidence=True,
        )

        idx = next(i for i, c in enumerate(results["cases"]) if c["id"] == "b08")
        results["cases"][idx] = b08_case

        res_file = self.tmp_dir / "results_b08_fail.json"
        res_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
        out_file = self.tmp_dir / "report_b08_fail.json"

        rep = check(res_file, self.env["reference"], out_file)
        self.assertFalse(rep["overall_pass"])
        self.assertEqual(rep["status"], "quality_fail")
        b08_res = next(c for c in rep["case_results"] if c["case_id"] == "b08")
        self.assertFalse(b08_res["whole_correct"])
        self.assertTrue(any("omitted_2015" in f or "status_mismatch" in f for f in b08_res["failures"]))

    def test_mandatory_partial_quote_only_2015_fails_direct(self) -> None:
        results = _build_synthetic_passed_results(self.env["protocol"], self.env)
        ref_q = strict_load_json(self.env["reference"].read_bytes())["cases"]["n03"]["query"]
        span1 = "Indica el año histórico comparable de generosidad"
        span2 = "y qué pensión mensual exacta me correspondería si me jubilo en 2030."

        n03_case = _build_synthetic_case(
            "n03", ref_q,
            claim_text="El año histórico comparable no queda indicado explícitamente.",
            quote=QUOTE_2015,
            parts=[
                {"query_span": span1, "decision": "answered", "claim_ids": ["C1"], "reason_code": "covered"},
                {"query_span": span2, "decision": "not_answered", "claim_ids": [], "reason_code": "not_in_evaluated_evidence"},
            ],
        )
        idx = next(i for i, c in enumerate(results["cases"]) if c["id"] == "n03")
        results["cases"][idx] = n03_case

        res_file = self.tmp_dir / "results_n03_fail.json"
        res_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
        out_file = self.tmp_dir / "report_n03_fail.json"

        rep = check(res_file, self.env["reference"], out_file)
        self.assertFalse(rep["overall_pass"])
        n03_res = next(c for c in rep["case_results"] if c["case_id"] == "n03")
        self.assertFalse(n03_res["main_direct_correct"])
        self.assertIn("n03_omitted_2015_in_main_answer", n03_res["failures"])

    def test_mandatory_partial_invented_amount_fails_safe(self) -> None:
        results = _build_synthetic_passed_results(self.env["protocol"], self.env)
        ref_q = strict_load_json(self.env["reference"].read_bytes())["cases"]["b08"]["query"]
        span1 = "¿Con qué año histórico se comparan los niveles futuros de generosidad"
        span2 = "y qué pensión mensual exacta cobraré si me jubilo en 2031?"

        b08_case = _build_synthetic_case(
            "b08", ref_q,
            claim_text="Se compara con 2015 y cobraré una pensión mensual de 1200 euros en 2031.",
            quote=QUOTE_2015,
            parts=[
                {"query_span": span1, "decision": "answered", "claim_ids": ["C1"], "reason_code": "covered"},
                {"query_span": span2, "decision": "not_answered", "claim_ids": [], "reason_code": "not_in_evaluated_evidence"},
            ],
        )
        idx = next(i for i, c in enumerate(results["cases"]) if c["id"] == "b08")
        results["cases"][idx] = b08_case

        res_file = self.tmp_dir / "results_b08_amount.json"
        res_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
        out_file = self.tmp_dir / "report_b08_amount.json"

        rep = check(res_file, self.env["reference"], out_file)
        self.assertFalse(rep["overall_pass"])
        b08_res = next(c for c in rep["case_results"] if c["case_id"] == "b08")
        self.assertFalse(b08_res["safe"])
        self.assertIn("b08_invented_personal_amount", b08_res["failures"])

    def test_outside_query_failed_to_abstain_fails_safe(self) -> None:
        results = _build_synthetic_passed_results(self.env["protocol"], self.env)
        ref_q = strict_load_json(self.env["reference"].read_bytes())["cases"]["b06"]["query"]

        b06_case = _build_synthetic_case(
            "b06", ref_q,
            claim_text="En 2050 supondrá el 15% del PIB de Cataluña.",
            quote=QUOTE_B02,
            parts=[{"query_span": ref_q, "decision": "answered", "claim_ids": ["C1"], "reason_code": "covered"}],
        )
        idx = next(i for i, c in enumerate(results["cases"]) if c["id"] == "b06")
        results["cases"][idx] = b06_case

        res_file = self.tmp_dir / "results_b06_fail.json"
        res_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
        out_file = self.tmp_dir / "report_b06_fail.json"

        rep = check(res_file, self.env["reference"], out_file)
        self.assertFalse(rep["overall_pass"])
        b06_res = next(c for c in rep["case_results"] if c["case_id"] == "b06")
        self.assertFalse(b06_res["safe"])
        self.assertIn("b06_failed_to_abstain", b06_res["failures"])

    def test_answerable_query_missing_fact_fails_whole(self) -> None:
        results = _build_synthetic_passed_results(self.env["protocol"], self.env)
        ref_q = strict_load_json(self.env["reference"].read_bytes())["cases"]["b01"]["query"]

        b01_case = _build_synthetic_case(
            "b01", ref_q,
            claim_text="El crecimiento del número de pensiones se acelera en el año 2036.",
            quote=QUOTE_B01,
        )
        results["cases"][0] = b01_case

        res_file = self.tmp_dir / "results_b01_missing_fact.json"
        res_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
        out_file = self.tmp_dir / "report_b01_missing.json"

        rep = check(res_file, self.env["reference"], out_file)
        self.assertFalse(rep["overall_pass"])
        b01_res = next(c for c in rep["case_results"] if c["case_id"] == "b01")
        self.assertFalse(b01_res["whole_correct"])
        self.assertIn("b01_main_missing_2036_or_2pct", b01_res["failures"])

    def test_improved_baseline_n02_regression_fails(self) -> None:
        ref_sha = file_sha256(self.env["reference"])
        baseline_report = {
            "schema_version": SCHEMA_VERSION,
            "overall_pass": True,
            "reference_sha256": ref_sha,
            "case_results": [
                {
                    "case_id": cid,
                    "whole_correct": True,
                    "main_direct_correct": True,
                    "safe": True,
                }
                for cid in EXPECTED_CASE_IDS
            ],
        }
        base_file = self.tmp_dir / "baseline.json"
        base_file.write_text(json.dumps(baseline_report, indent=2), encoding="utf-8")

        results = _build_synthetic_passed_results(self.env["protocol"], self.env)
        res_file = self.tmp_dir / "results_regress.json"
        res_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
        out_file = self.tmp_dir / "report_regress.json"

        rep = check(res_file, self.env["reference"], out_file, baseline_path=base_file)
        self.assertFalse(rep["overall_pass"])
        self.assertTrue(len(rep["summary"]["regressions"]) > 0)
        self.assertIn("Regression on n02: main_direct_correct dropped from True to False", rep["summary"]["regressions"][0])

    def test_output_overwrite_prevention(self) -> None:
        results = _build_synthetic_passed_results(self.env["protocol"], self.env)
        res_file = self.tmp_dir / "results_overwrite.json"
        res_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
        out_file = self.tmp_dir / "report_exists.json"
        out_file.write_text("existing content", encoding="utf-8")

        with self.assertRaises(FileExistsError):
            check(res_file, self.env["reference"], out_file)

    def test_prepare_does_not_leak_gold_and_creates_protocol(self) -> None:
        prep_out = self.tmp_dir / "prep_dir"
        proto = prepare(
            prep_out,
            index_path=str(self.env["index"]),
            queries_path=str(self.env["queries"]),
            units_path=str(self.env["units"]),
            pages_path=str(self.env["pages"]),
        )
        self.assertEqual(proto["schema_version"], "covered-pipeline-protocol-v1")
        self.assertEqual(proto["server_version"], "0.40.11")
        self.assertTrue((prep_out / "protocol.json").is_file())

        for path_key in ("index_path", "queries_path", "units_path", "pages_path"):
            self.assertNotIn("gold", proto[path_key].lower())
        for f_path in proto["frozen_files"].keys():
            self.assertNotIn("gold", f_path.lower())

    def test_mock_runner_separation_and_lifecycle(self) -> None:
        run_out = self.tmp_dir / "run_out"
        mock_covered_run = MagicMock()

        def _side_effect(*args: Any, **kwargs: Any) -> None:
            dest = args[4]
            results = _build_synthetic_passed_results(run_out / "protocol.json", self.env)
            Path(dest).write_text(json.dumps(results, indent=2), encoding="utf-8")

        mock_covered_run.side_effect = _side_effect

        with patch("experiments.evaluate_covered_pipeline.run", mock_covered_run):
            with patch("rag_pensiones.search.load_index", return_value={"mock": "index"}):
                rep = gate_run(
                    run_out,
                    reference_path=self.env["reference"],
                    index_path=str(self.env["index"]),
                    queries_path=str(self.env["queries"]),
                    units_path=str(self.env["units"]),
                    pages_path=str(self.env["pages"]),
                    structured_retrieval=False,
                )
                self.assertTrue(rep["overall_pass"])
                self.assertTrue((run_out / "protocol.json").is_file())
                self.assertTrue((run_out / "results.json").is_file())
                self.assertTrue((run_out / "quality_report.json").is_file())
                mock_covered_run.assert_called_once()

    def test_current_default_rejects_legacy_mock_results_without_structured_manifest(self) -> None:
        run_out = self.tmp_dir / 'current_default_rejects_legacy'
        def legacy_result(*args: Any, **kwargs: Any) -> None:
            results = _build_synthetic_passed_results(run_out / 'protocol.json', self.env)
            Path(args[4]).write_text(json.dumps(results), encoding='utf-8')
        with patch('experiments.evaluate_covered_pipeline.run', side_effect=legacy_result):
            with patch('rag_pensiones.search.load_index', return_value={'mock': 'index'}):
                with self.assertRaisesRegex(ValueError, 'Structured results manifest missing or changed'):
                    gate_run(run_out, reference_path=self.env['reference'],
                        index_path=str(self.env['index']),queries_path=str(self.env['queries']),
                        units_path=str(self.env['units']),pages_path=str(self.env['pages']))

    def test_cli_exit_codes(self) -> None:
        results = _build_synthetic_passed_results(self.env["protocol"], self.env)
        res_file = self.tmp_dir / "cli_results.json"
        res_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
        out_file = self.tmp_dir / "cli_report.json"

        # 1. Passing check -> exit 0
        code0 = gate_main(["check", "--results", str(res_file), "--reference", str(self.env["reference"]), "--output", str(out_file)])
        self.assertEqual(code0, 0)

        # 2. Quality fail check (b08 blanket refusal) -> exit 1
        ref_q = strict_load_json(self.env["reference"].read_bytes())["cases"]["b08"]["query"]
        b08_fail = _build_synthetic_case(
            "b08", ref_q,
            force_insufficient_evidence=True,
        )

        results_fail = copy.deepcopy(results)
        idx = next(i for i, c in enumerate(results_fail["cases"]) if c["id"] == "b08")
        results_fail["cases"][idx] = b08_fail

        res_file_fail = self.tmp_dir / "cli_results_fail.json"
        res_file_fail.write_text(json.dumps(results_fail, indent=2), encoding="utf-8")
        out_file_fail = self.tmp_dir / "cli_report_fail.json"

        code1 = gate_main(["check", "--results", str(res_file_fail), "--reference", str(self.env["reference"]), "--output", str(out_file_fail)])
        self.assertEqual(code1, 1)

        # 3. Invalid args / existing output -> exit 2
        code2 = gate_main(["check", "--results", str(res_file), "--reference", str(self.env["reference"]), "--output", str(out_file)])
        self.assertEqual(code2, 2)

    def test_cli_run_with_baseline(self) -> None:
        run_out = self.tmp_dir / "cli_run_out"
        mock_covered_run = MagicMock()

        def _side_effect(*args: Any, **kwargs: Any) -> None:
            dest = args[4]
            results = _build_synthetic_passed_results(run_out / "protocol.json", self.env)
            Path(dest).write_text(json.dumps(results, indent=2), encoding="utf-8")

        mock_covered_run.side_effect = _side_effect

        ref_sha = file_sha256(self.env["reference"])
        base_data = {
            "schema_version": SCHEMA_VERSION,
            "overall_pass": True,
            "reference_sha256": ref_sha,
            "case_results": [
                {
                    "case_id": cid,
                    "whole_correct": True,
                    "main_direct_correct": False if cid == "n02" else True,
                    "safe": True,
                }
                for cid in EXPECTED_CASE_IDS
            ],
        }
        base_file = self.tmp_dir / "cli_baseline.json"
        base_file.write_text(json.dumps(base_data, indent=2), encoding="utf-8")

        with patch("experiments.evaluate_covered_pipeline.run", mock_covered_run):
            with patch("rag_pensiones.search.load_index", return_value={"mock": "index"}):
                with patch("experiments.continuous_quality.DEFAULT_REFERENCE_REL", str(self.env["reference"])):
                    with patch("experiments.continuous_quality.DEFAULT_INDEX_REL", str(self.env["index"])):
                        with patch("experiments.continuous_quality.DEFAULT_QUERIES_REL", str(self.env["queries"])):
                            with patch("experiments.continuous_quality.DEFAULT_UNITS_REL", str(self.env["units"])):
                                with patch("experiments.continuous_quality.DEFAULT_PAGES_REL", str(self.env["pages"])):
                                    code = gate_main([
                                        "run",
                                        "--legacy-semantic",
                                        "--output-dir", str(run_out),
                                        "--reference", str(self.env["reference"]),
                                        "--baseline", str(base_file),
                                    ])
                                    self.assertEqual(code, 0)
                                    self.assertTrue((run_out / "quality_report.json").is_file())

    def test_manifests_mandatory_and_complete_rejected(self) -> None:
        results = _build_synthetic_passed_results(self.env["protocol"], self.env)
        res_file = self.tmp_dir / "report_base_res.json"
        res_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
        out_base = self.tmp_dir / "report_base.json"
        rep = check(res_file, self.env["reference"], out_base)
        self.assertTrue(rep["overall_pass"])

        # 1. Covered protocol with missing input_hashes is permitted when frozen_files pins all inputs
        proto1 = strict_load_json(self.env["protocol"].read_bytes())
        del proto1["input_hashes"]
        p1 = self.tmp_dir / "p_no_ih.json"
        p1.write_text(json.dumps(proto1), encoding="utf-8")
        r1 = copy.deepcopy(results)
        r1["metadata"]["protocol_hash"] = canonical_hash(proto1)
        r1_file = self.tmp_dir / "r1_file.json"
        r1_file.write_text(json.dumps(r1), encoding="utf-8")
        rep1 = check(r1_file, self.env["reference"], self.tmp_dir / "out1.json", protocol_path=p1)
        self.assertTrue(rep1["overall_pass"])

        # 2. Historical protocol requires input_hashes
        hist_res, hist_proto = _build_synthetic_historical_environment(self.tmp_dir, self.env)
        h_proto_data = strict_load_json(hist_proto.read_bytes())
        del h_proto_data["input_hashes"]
        hp_no_ih = self.tmp_dir / "hp_no_ih.json"
        hp_no_ih.write_text(json.dumps(h_proto_data), encoding="utf-8")
        h_res_data = strict_load_json(hist_res.read_bytes())
        h_res_data["protocol_sha256"] = file_sha256(hp_no_ih)
        hr_no_ih = self.tmp_dir / "hr_no_ih.json"
        hr_no_ih.write_text(json.dumps(h_res_data), encoding="utf-8")
        with self.assertRaises(ValueError) as ctx_h:
            check(hr_no_ih, self.env["reference"], self.tmp_dir / "out_h_no_ih.json", arm="mixed", protocol_path=hp_no_ih)
        self.assertIn("Historical protocol missing mandatory input_hashes manifest", str(ctx_h.exception))

        # 3. Empty input_hashes in covered protocol rejected
        proto2 = strict_load_json(self.env["protocol"].read_bytes())
        proto2["input_hashes"] = {}
        p2 = self.tmp_dir / "p_empty_ih.json"
        p2.write_text(json.dumps(proto2), encoding="utf-8")
        r2 = copy.deepcopy(results)
        r2["metadata"]["protocol_hash"] = canonical_hash(proto2)
        r2_f = self.tmp_dir / "r2_empty_ih.json"
        r2_f.write_text(json.dumps(r2), encoding="utf-8")
        with self.assertRaises(ValueError) as ctx2:
            check(r2_f, self.env["reference"], self.tmp_dir / "out2.json", protocol_path=p2)
        self.assertIn("Covered protocol input_hashes manifest is empty or invalid", str(ctx2.exception))

        # 4. Partial input_hashes (missing units) rejected
        proto3 = strict_load_json(self.env["protocol"].read_bytes())
        del proto3["input_hashes"]["units"]
        p3 = self.tmp_dir / "p_part_ih.json"
        p3.write_text(json.dumps(proto3), encoding="utf-8")
        r3 = copy.deepcopy(results)
        r3["metadata"]["protocol_hash"] = canonical_hash(proto3)
        r3_f = self.tmp_dir / "r3_part_ih.json"
        r3_f.write_text(json.dumps(r3), encoding="utf-8")
        with self.assertRaises(ValueError) as ctx3:
            check(r3_f, self.env["reference"], self.tmp_dir / "out3.json", protocol_path=p3)
        self.assertIn("Covered protocol input_hashes missing mandatory key: 'units'", str(ctx3.exception))

        # 5. Missing frozen_files rejected
        proto4 = strict_load_json(self.env["protocol"].read_bytes())
        del proto4["frozen_files"]
        p4 = self.tmp_dir / "p_no_ff.json"
        p4.write_text(json.dumps(proto4), encoding="utf-8")
        r4 = copy.deepcopy(results)
        r4["metadata"]["protocol_hash"] = canonical_hash(proto4)
        r4_f = self.tmp_dir / "r4_no_ff.json"
        r4_f.write_text(json.dumps(r4), encoding="utf-8")
        with self.assertRaises(ValueError) as ctx4:
            check(r4_f, self.env["reference"], self.tmp_dir / "out4.json", protocol_path=p4)
        self.assertIn("Protocol frozen_files manifest is missing or empty", str(ctx4.exception))

        # 6. Partial frozen_files (missing index) rejected
        proto5 = strict_load_json(self.env["protocol"].read_bytes())
        del proto5["frozen_files"][str(self.env["index"])]
        p5 = self.tmp_dir / "p_part_ff.json"
        p5.write_text(json.dumps(proto5), encoding="utf-8")
        r5 = copy.deepcopy(results)
        r5["metadata"]["protocol_hash"] = canonical_hash(proto5)
        r5_f = self.tmp_dir / "r5_part_ff.json"
        r5_f.write_text(json.dumps(r5), encoding="utf-8")
        with self.assertRaises(ValueError) as ctx5:
            check(r5_f, self.env["reference"], self.tmp_dir / "out5.json", protocol_path=p5)
        self.assertIn("missing from protocol frozen_files manifest", str(ctx5.exception))

    def test_altered_index_and_units_bytes_rejected(self) -> None:
        results = _build_synthetic_passed_results(self.env["protocol"], self.env)
        res_file = self.tmp_dir / "res_base_idx.json"
        res_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
        out_base = self.tmp_dir / "report_base_idx.json"
        rep = check(res_file, self.env["reference"], out_base)
        self.assertTrue(rep["overall_pass"])

        # 1. Alter index on disk
        self.env["index"].write_text('{"tampered": true}', encoding="utf-8")
        with self.assertRaises(ValueError) as ctx1:
            check(res_file, self.env["reference"], self.tmp_dir / "out_bad_idx.json")
        self.assertIn("Index input canonical hash mismatch", str(ctx1.exception))

        # Restore index and alter units on disk
        self.env["index"].write_text(json.dumps({"model": "qwen3-embedding:4b", "model_digest": PINNED_EMBEDDING_DIGEST, "chunks": []}), encoding="utf-8")
        self.env["units"].write_text('{"tampered": true}', encoding="utf-8")
        with self.assertRaises(ValueError) as ctx2:
            check(res_file, self.env["reference"], self.tmp_dir / "out_bad_unt.json")
        self.assertIn("Units input canonical hash mismatch", str(ctx2.exception))

    def test_malformed_models_and_server_rejected(self) -> None:
        results = _build_synthetic_passed_results(self.env["protocol"], self.env)
        res_file = self.tmp_dir / "res_base_srv.json"
        res_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
        out_base = self.tmp_dir / "report_base_srv.json"
        rep = check(res_file, self.env["reference"], out_base)
        self.assertTrue(rep["overall_pass"])

        # 1. Bad server_version
        proto1 = strict_load_json(self.env["protocol"].read_bytes())
        proto1["server_version"] = "0.39.0"
        p1 = self.tmp_dir / "p_bad_srv.json"
        p1.write_text(json.dumps(proto1), encoding="utf-8")
        r1 = copy.deepcopy(results)
        r1["metadata"]["protocol_hash"] = canonical_hash(proto1)
        r1_f = self.tmp_dir / "r1_bad_srv.json"
        r1_f.write_text(json.dumps(r1), encoding="utf-8")
        with self.assertRaises(ValueError) as ctx1:
            check(r1_f, self.env["reference"], self.tmp_dir / "out_bad_srv.json", protocol_path=p1)
        self.assertIn("server_version mismatch", str(ctx1.exception))

        # 2. Bad generator_digest
        proto2 = strict_load_json(self.env["protocol"].read_bytes())
        proto2["generator_digest"] = "0" * 64
        p2 = self.tmp_dir / "p_bad_gen.json"
        p2.write_text(json.dumps(proto2), encoding="utf-8")
        r2 = copy.deepcopy(results)
        r2["metadata"]["protocol_hash"] = canonical_hash(proto2)
        r2_f = self.tmp_dir / "r2_bad_gen.json"
        r2_f.write_text(json.dumps(r2), encoding="utf-8")
        with self.assertRaises(ValueError) as ctx2:
            check(r2_f, self.env["reference"], self.tmp_dir / "out_bad_gen.json", protocol_path=p2)
        self.assertIn("generator_digest mismatch", str(ctx2.exception))

        # 3. Bad verifier_digest
        proto3 = strict_load_json(self.env["protocol"].read_bytes())
        proto3["verifier_digest"] = "0" * 64
        p3 = self.tmp_dir / "p_bad_ver.json"
        p3.write_text(json.dumps(proto3), encoding="utf-8")
        r3 = copy.deepcopy(results)
        r3["metadata"]["protocol_hash"] = canonical_hash(proto3)
        r3_f = self.tmp_dir / "r3_bad_ver.json"
        r3_f.write_text(json.dumps(r3), encoding="utf-8")
        with self.assertRaises(ValueError) as ctx3:
            check(r3_f, self.env["reference"], self.tmp_dir / "out_bad_ver.json", protocol_path=p3)
        self.assertIn("verifier_digest mismatch", str(ctx3.exception))

        # 4. Bad embedding_digest
        proto4 = strict_load_json(self.env["protocol"].read_bytes())
        proto4["embedding_digest"] = "0" * 64
        p4 = self.tmp_dir / "p_bad_emb.json"
        p4.write_text(json.dumps(proto4), encoding="utf-8")
        r4 = copy.deepcopy(results)
        r4["metadata"]["protocol_hash"] = canonical_hash(proto4)
        r4_f = self.tmp_dir / "r4_bad_emb.json"
        r4_f.write_text(json.dumps(r4), encoding="utf-8")
        with self.assertRaises(ValueError) as ctx4:
            check(r4_f, self.env["reference"], self.tmp_dir / "out_bad_emb.json", protocol_path=p4)
        self.assertIn("embedding_digest mismatch", str(ctx4.exception))

    def test_historical_adapter_coverage_audit_and_publication_required(self) -> None:
        res_path, proto_path = _build_synthetic_historical_environment(self.tmp_dir, self.env)
        out_base = self.tmp_dir / "hist_out_pass.json"
        rep = check(res_path, self.env["reference"], out_base, arm="mixed", protocol_path=proto_path)
        self.assertTrue(rep["overall_pass"])
        self.assertEqual(rep["status"], "pass")

        # 1. Missing coverage_audit
        res_data1 = strict_load_json(res_path.read_bytes())
        del res_data1["attempts"][0]["coverage_audit"]
        r1 = self.tmp_dir / "hist_no_audit.json"
        r1.write_text(json.dumps(res_data1), encoding="utf-8")
        with self.assertRaises(ValueError) as ctx1:
            check(r1, self.env["reference"], self.tmp_dir / "out_no_audit.json", arm="mixed", protocol_path=proto_path)
        self.assertIn("missing coverage_audit dictionary", str(ctx1.exception))

        # 2. Missing coverage_publication
        res_data2 = strict_load_json(res_path.read_bytes())
        del res_data2["attempts"][0]["coverage_publication"]
        r2 = self.tmp_dir / "hist_no_covpub.json"
        r2.write_text(json.dumps(res_data2), encoding="utf-8")
        with self.assertRaises(ValueError) as ctx2:
            check(r2, self.env["reference"], self.tmp_dir / "out_no_covpub.json", arm="mixed", protocol_path=proto_path)
        self.assertIn("missing coverage_publication dictionary", str(ctx2.exception))

        # 3. coverage_audit error and missing coverage_publication
        res_data3 = strict_load_json(res_path.read_bytes())
        res_data3["attempts"][0]["coverage_audit"]["status"] = "error"
        res_data3["attempts"][0]["coverage_audit"]["error"] = "Coverage service failure"
        del res_data3["attempts"][0]["coverage_publication"]
        r3 = self.tmp_dir / "hist_err_nopub.json"
        r3.write_text(json.dumps(res_data3), encoding="utf-8")
        with self.assertRaises(ValueError) as ctx3:
            check(r3, self.env["reference"], self.tmp_dir / "out_err_nopub.json", arm="mixed", protocol_path=proto_path)
        self.assertIn("Coverage audit error", str(ctx3.exception))

        # 4. Illegitimate skipped coverage when inventory is non-empty
        res_data4 = strict_load_json(res_path.read_bytes())
        res_data4["attempts"][0]["coverage_audit"]["skipped"] = True
        r4 = self.tmp_dir / "hist_bad_skip.json"
        r4.write_text(json.dumps(res_data4), encoding="utf-8")
        with self.assertRaises(ValueError) as ctx4:
            check(r4, self.env["reference"], self.tmp_dir / "out_bad_skip.json", arm="mixed", protocol_path=proto_path)
        self.assertIn("marked skipped but inventory is non-empty", str(ctx4.exception))

    def test_exact_domain_counterexamples_fail_quality(self) -> None:
        results = _build_synthetic_passed_results(self.env["protocol"], self.env)
        res_file = self.tmp_dir / "domain_base_pass_res.json"
        res_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
        out_base = self.tmp_dir / "domain_base_pass.json"
        rep = check(res_file, self.env["reference"], out_base)
        self.assertTrue(rep["overall_pass"])
        self.assertEqual(rep["status"], "pass")

        ref_data = strict_load_json(self.env["reference"].read_bytes())

        # 1. b03 counterexample: 'Las pensiones crecen con IPC. Los salarios crecen exclusivamente con productividad'
        ref_q_b03 = ref_data["cases"]["b03"]["query"]
        b03_case = _build_synthetic_case(
            "b03", ref_q_b03,
            claim_text="Las pensiones crecen con IPC. Los salarios crecen exclusivamente con productividad",
            quote=QUOTE_B03,
        )
        res_b03 = copy.deepcopy(results)
        idx_b03 = next(i for i, c in enumerate(res_b03["cases"]) if c["id"] == "b03")
        res_b03["cases"][idx_b03] = b03_case
        f_b03 = self.tmp_dir / "res_b03_ce.json"
        f_b03.write_text(json.dumps(res_b03), encoding="utf-8")
        rep_b03 = check(f_b03, self.env["reference"], self.tmp_dir / "out_b03_ce.json")
        self.assertFalse(rep_b03["overall_pass"])
        self.assertEqual(rep_b03["status"], "quality_fail")
        b03_res = next(c for c in rep_b03["case_results"] if c["case_id"] == "b03")
        self.assertFalse(b03_res["whole_correct"])
        self.assertIn("b03_main_swapped_or_missing_attribution", b03_res["failures"])

        # 2. b02 counterexample: 'Después de 2050 el gasto supera el 4% anual, impulsado por el aumento del número de pensiones y una pensión media constante'
        ref_q_b02 = ref_data["cases"]["b02"]["query"]
        b02_case = _build_synthetic_case(
            "b02", ref_q_b02,
            claim_text="Después de 2050 el gasto supera el 4% anual, impulsado por el aumento del número de pensiones y una pensión media constante",
            quote=QUOTE_B02,
        )
        res_b02 = copy.deepcopy(results)
        idx_b02 = next(i for i, c in enumerate(res_b02["cases"]) if c["id"] == "b02")
        res_b02["cases"][idx_b02] = b02_case
        f_b02 = self.tmp_dir / "res_b02_ce.json"
        f_b02.write_text(json.dumps(res_b02), encoding="utf-8")
        rep_b02 = check(f_b02, self.env["reference"], self.tmp_dir / "out_b02_ce.json")
        self.assertFalse(rep_b02["overall_pass"])
        self.assertEqual(rep_b02["status"], "quality_fail")
        b02_res = next(c for c in rep_b02["case_results"] if c["case_id"] == "b02")
        self.assertFalse(b02_res["whole_correct"])
        self.assertTrue(any(f in b02_res["failures"] for f in ("b02_main_missing_2050", "b02_main_missing_increase_mechanism")))

        # 3. b05 counterexample: 'El aumento del número de pensiones compensa la caída de la pensión media'
        ref_q_b05 = ref_data["cases"]["b05"]["query"]
        b05_case = _build_synthetic_case(
            "b05", ref_q_b05,
            claim_text="El aumento del número de pensiones compensa la caída de la pensión media",
            quote=QUOTE_B05,
        )
        res_b05 = copy.deepcopy(results)
        idx_b05 = next(i for i, c in enumerate(res_b05["cases"]) if c["id"] == "b05")
        res_b05["cases"][idx_b05] = b05_case
        f_b05 = self.tmp_dir / "res_b05_ce.json"
        f_b05.write_text(json.dumps(res_b05), encoding="utf-8")
        rep_b05 = check(f_b05, self.env["reference"], self.tmp_dir / "out_b05_ce.json")
        self.assertFalse(rep_b05["overall_pass"])
        self.assertEqual(rep_b05["status"], "quality_fail")
        b05_res = next(c for c in rep_b05["case_results"] if c["case_id"] == "b05")
        self.assertFalse(b05_res["whole_correct"])
        self.assertIn("b05_main_missing_increase_cause", b05_res["failures"])

    def test_recomputed_hash_covered_profile_corruptions_fail_gate(self) -> None:
        results = _build_synthetic_passed_results(self.env["protocol"], self.env)
        base_proto = strict_load_json(self.env["protocol"].read_bytes())

        corruptions = [
            ("missing_server_version", lambda p: p.pop("server_version", None), "server_version mismatch"),
            ("drifted_server_version", lambda p: p.update(server_version="0.39.0"), "server_version mismatch"),
            ("missing_generator_parameters", lambda p: p.pop("generator_parameters", None), "must be a dict"),
            ("empty_generator_parameters", lambda p: p.update(generator_parameters={}), "options keys mismatch"),
            ("drifted_gen_temperature", lambda p: p["generator_parameters"].update(temperature=99), "value mismatch"),
            ("boolean_gen_temperature", lambda p: p["generator_parameters"].update(temperature=False), "boolean value masquerading"),
            ("added_gen_option", lambda p: p["generator_parameters"].update(extra_option=1), "added/drifted"),
            ("drifted_gen_seed", lambda p: p["generator_parameters"].update(seed=123), "value mismatch"),
            ("missing_verifier_parameters", lambda p: p.pop("verifier_parameters", None), "must be a dict"),
            ("empty_verifier_parameters", lambda p: p.update(verifier_parameters={}), "options keys mismatch"),
            ("drifted_ver_presence", lambda p: p["verifier_parameters"].update(presence_penalty=0.0), "value mismatch"),
            ("boolean_ver_temperature", lambda p: p["verifier_parameters"].update(temperature=True), "boolean value masquerading"),
            ("missing_embedding_digest", lambda p: p.pop("embedding_digest", None), "invalid format"),
            ("boolean_embedding_digest", lambda p: p.update(embedding_digest=False), "invalid format"),
            ("nonhex_embedding_digest", lambda p: p.update(embedding_digest="z" * 64), "invalid format"),
            ("drifted_embedding_digest", lambda p: p.update(embedding_digest="0" * 64), "embedding_digest mismatch"),
        ]

        for label, mutate_fn, expected_msg in corruptions:
            with self.subTest(corruption=label):
                tampered_p = copy.deepcopy(base_proto)
                mutate_fn(tampered_p)
                p_file = self.tmp_dir / f"cov_p_{label}.json"
                p_file.write_text(json.dumps(tampered_p), encoding="utf-8")

                tampered_res = copy.deepcopy(results)
                tampered_res["metadata"]["protocol_hash"] = canonical_hash(tampered_p)
                r_file = self.tmp_dir / f"cov_r_{label}.json"
                r_file.write_text(json.dumps(tampered_res), encoding="utf-8")

                with self.assertRaises(ValueError) as ctx:
                    check(r_file, self.env["reference"], self.tmp_dir / f"cov_out_{label}.json", protocol_path=p_file)
                self.assertIn(expected_msg, str(ctx.exception))

        # Exercise check's actual strict JSON load for NaN/+Infinity/-Infinity using deliberately serialized invalid protocol bytes
        base_proto_raw = json.dumps(base_proto, ensure_ascii=False)
        for token in ("NaN", "Infinity", "-Infinity", "+Infinity"):
            with self.subTest(covered_nonfinite=token):
                inv_proto_bytes = base_proto_raw.replace('"temperature": 0', f'"temperature": {token}', 1).encode("utf-8")
                safe_tok = token.replace("+", "pos_").replace("-", "neg_")
                p_file = self.tmp_dir / f"cov_p_nonfinite_{safe_tok}.json"
                p_file.write_bytes(inv_proto_bytes)

                tampered_res = copy.deepcopy(results)
                # Covered metadata hash is not computable because non-finite JSON cannot be serialized/parsed into valid data;
                # check() must reject at strict JSON parse boundary before hash comparison.
                tampered_res["metadata"]["protocol_hash"] = "0" * 64
                r_file = self.tmp_dir / f"cov_r_nonfinite_{safe_tok}.json"
                r_file.write_text(json.dumps(tampered_res), encoding="utf-8")

                with self.assertRaises(ValueError) as ctx:
                    check(r_file, self.env["reference"], self.tmp_dir / f"cov_out_nonfinite_{safe_tok}.json", protocol_path=p_file)
                self.assertTrue(any(term in str(ctx.exception) for term in ("Non-finite", "JSONDecodeError", "Expecting value")))

    def test_recomputed_hash_historical_profile_and_controls(self) -> None:
        # 1. Valid mixed arm control passes
        res_m, proto_m = _build_synthetic_historical_environment(self.tmp_dir, self.env, arm="mixed")
        out_m = self.tmp_dir / "hist_ctrl_mixed.json"
        rep_m = check(res_m, self.env["reference"], out_m, arm="mixed", protocol_path=proto_m)
        self.assertTrue(rep_m["overall_pass"])

        # 2. Valid all27 arm control passes
        res_a, proto_a = _build_synthetic_historical_environment(self.tmp_dir, self.env, arm="all27")
        out_a = self.tmp_dir / "hist_ctrl_all27.json"
        rep_a = check(res_a, self.env["reference"], out_a, arm="all27", protocol_path=proto_a)
        self.assertTrue(rep_a["overall_pass"])

        base_proto = strict_load_json(proto_m.read_bytes())
        base_res = strict_load_json(res_m.read_bytes())

        corruptions = [
            ("missing_server_version", lambda p: p.pop("server_version", None), "server_version mismatch"),
            ("drifted_server_version", lambda p: p.update(server_version="0.39.0"), "server_version mismatch"),
            ("missing_effective_options", lambda p: p.pop("effective_generation_options", None), "must be a dict"),
            ("empty_effective_options", lambda p: p.update(effective_generation_options={}), "options keys mismatch"),
            ("drifted_eff_temp", lambda p: p["effective_generation_options"].update(temperature=1), "value mismatch"),
            ("boolean_eff_temp", lambda p: p["effective_generation_options"].update(temperature=True), "boolean value masquerading"),
            ("missing_canonical_params", lambda p: p.pop("canonical_role_parameters", None), "must be a dict"),
            ("drifted_canonical_seed", lambda p: p["canonical_role_parameters"].update(seed=999), "value mismatch"),
            ("missing_verifier_params", lambda p: p.pop("verifier_parameters", None), "must be a dict"),
            ("drifted_ver_predict", lambda p: p["verifier_parameters"].update(num_predict=1024), "value mismatch"),
            ("missing_models", lambda p: p.pop("models", None), "missing or invalid models"),
            ("empty_models", lambda p: p.update(models={}), "missing or invalid models"),
            ("missing_embedding_4b", lambda p: p["models"].pop("embedding_4b", None), "missing mandatory model: 'embedding_4b'"),
            ("empty_embedding_4b", lambda p: p["models"].update(embedding_4b={}), "setup must be a non-empty dict"),
            ("drifted_embedding_name", lambda p: p["models"]["embedding_4b"].update(name="wrong-embedding"), "name mismatch"),
            ("drifted_embedding_digest", lambda p: p["models"]["embedding_4b"].update(digest="0" * 64), "digest mismatch"),
            ("boolean_embedding_digest", lambda p: p["models"]["embedding_4b"].update(digest=True), "invalid digest format"),
            ("nonhex_embedding_digest", lambda p: p["models"]["embedding_4b"].update(digest="z" * 64), "invalid digest format"),
        ]

        for label, mutate_fn, expected_msg in corruptions:
            with self.subTest(historical_corruption=label):
                tampered_p = copy.deepcopy(base_proto)
                mutate_fn(tampered_p)
                p_file = self.tmp_dir / f"hist_p_{label}.json"
                p_file.write_text(json.dumps(tampered_p), encoding="utf-8")

                tampered_res = copy.deepcopy(base_res)
                tampered_res["protocol_sha256"] = file_sha256(p_file)
                r_file = self.tmp_dir / f"hist_r_{label}.json"
                r_file.write_text(json.dumps(tampered_res), encoding="utf-8")

                with self.assertRaises(ValueError) as ctx:
                    check(r_file, self.env["reference"], self.tmp_dir / f"hist_out_{label}.json", arm="mixed", protocol_path=p_file)
                self.assertIn(expected_msg, str(ctx.exception))

        # Exercise check's actual strict JSON load for NaN/+Infinity/-Infinity with rebound historical raw-byte hash
        proto_m_str = proto_m.read_text(encoding="utf-8")
        for token in ("NaN", "Infinity", "-Infinity", "+Infinity"):
            with self.subTest(historical_nonfinite=token):
                inv_proto_str = proto_m_str.replace('"temperature": 0', f'"temperature": {token}', 1)
                safe_tok = token.replace("+", "pos_").replace("-", "neg_")
                ph_file = self.tmp_dir / f"hist_p_nonfinite_{safe_tok}.json"
                ph_file.write_text(inv_proto_str, encoding="utf-8")

                tampered_rh = copy.deepcopy(base_res)
                # Rebind historical raw-bytes SHA-256 so hash matches and proves rejection occurs at JSON load boundary
                tampered_rh["protocol_sha256"] = file_sha256(ph_file)
                rh_file = self.tmp_dir / f"hist_r_nonfinite_{safe_tok}.json"
                rh_file.write_text(json.dumps(tampered_rh), encoding="utf-8")

                with self.assertRaises(ValueError) as ctx:
                    check(rh_file, self.env["reference"], self.tmp_dir / f"hist_out_nonfinite_{safe_tok}.json", arm="mixed", protocol_path=ph_file)
                self.assertTrue(any(term in str(ctx.exception) for term in ("Non-finite", "JSONDecodeError", "Expecting value")))

    def test_recomputed_hash_source_manifest_corruptions(self) -> None:
        results = _build_synthetic_passed_results(self.env["protocol"], self.env)
        base_proto = strict_load_json(self.env["protocol"].read_bytes())

        # 1. Removal of experiments/continuous_quality.py from covered manifest
        p_no_cq = copy.deepcopy(base_proto)
        del p_no_cq["frozen_files"]["experiments/continuous_quality.py"]
        pf_no_cq = self.tmp_dir / "p_no_cq.json"
        pf_no_cq.write_text(json.dumps(p_no_cq), encoding="utf-8")
        r_no_cq = copy.deepcopy(results)
        r_no_cq["metadata"]["protocol_hash"] = canonical_hash(p_no_cq)
        rf_no_cq = self.tmp_dir / "r_no_cq.json"
        rf_no_cq.write_text(json.dumps(r_no_cq), encoding="utf-8")
        with self.assertRaises(ValueError) as ctx1:
            check(rf_no_cq, self.env["reference"], self.tmp_dir / "out_no_cq.json", protocol_path=pf_no_cq)
        self.assertIn("missing mandatory source paths", str(ctx1.exception))

        # 2. Removal of rag_pensiones/chunking.py from covered manifest
        p_no_chk = copy.deepcopy(base_proto)
        del p_no_chk["frozen_files"]["rag_pensiones/chunking.py"]
        pf_no_chk = self.tmp_dir / "p_no_chk.json"
        pf_no_chk.write_text(json.dumps(p_no_chk), encoding="utf-8")
        r_no_chk = copy.deepcopy(results)
        r_no_chk["metadata"]["protocol_hash"] = canonical_hash(p_no_chk)
        rf_no_chk = self.tmp_dir / "r_no_chk.json"
        rf_no_chk.write_text(json.dumps(r_no_chk), encoding="utf-8")
        with self.assertRaises(ValueError) as ctx2:
            check(rf_no_chk, self.env["reference"], self.tmp_dir / "out_no_chk.json", protocol_path=pf_no_chk)
        self.assertIn("missing mandatory source paths", str(ctx2.exception))

        # 3. Input-only manifest (only 4 input files, 0 python sources)
        p_input_only = copy.deepcopy(base_proto)
        p_input_only["frozen_files"] = {
            k: v for k, v in base_proto["frozen_files"].items()
            if not k.endswith(".py")
        }
        pf_io = self.tmp_dir / "p_io.json"
        pf_io.write_text(json.dumps(p_input_only), encoding="utf-8")
        r_io = copy.deepcopy(results)
        r_io["metadata"]["protocol_hash"] = canonical_hash(p_input_only)
        rf_io = self.tmp_dir / "r_io.json"
        rf_io.write_text(json.dumps(r_io), encoding="utf-8")
        with self.assertRaises(ValueError) as ctx3:
            check(rf_io, self.env["reference"], self.tmp_dir / "out_io.json", protocol_path=pf_io)
        self.assertIn("missing mandatory source paths", str(ctx3.exception))

        # 4. Unknown-only source manifest (does not satisfy mandatory inventory)
        p_unk = copy.deepcopy(base_proto)
        p_unk["frozen_files"] = {"unknown/fake_module.py": "c" * 64}
        pf_unk = self.tmp_dir / "p_unk.json"
        pf_unk.write_text(json.dumps(p_unk), encoding="utf-8")
        r_unk = copy.deepcopy(results)
        r_unk["metadata"]["protocol_hash"] = canonical_hash(p_unk)
        rf_unk = self.tmp_dir / "r_unk.json"
        rf_unk.write_text(json.dumps(r_unk), encoding="utf-8")
        with self.assertRaises(ValueError) as ctx4:
            check(rf_unk, self.env["reference"], self.tmp_dir / "out_unk.json", protocol_path=pf_unk)
        self.assertIn("missing mandatory source paths", str(ctx4.exception))

        # 5. Non-hex SHA in frozen_files
        p_nonhex = copy.deepcopy(base_proto)
        p_nonhex["frozen_files"]["experiments/continuous_quality.py"] = "g" * 64
        pf_nonhex = self.tmp_dir / "p_nonhex.json"
        pf_nonhex.write_text(json.dumps(p_nonhex), encoding="utf-8")
        r_nonhex = copy.deepcopy(results)
        r_nonhex["metadata"]["protocol_hash"] = canonical_hash(p_nonhex)
        rf_nonhex = self.tmp_dir / "r_nonhex.json"
        rf_nonhex.write_text(json.dumps(r_nonhex), encoding="utf-8")
        with self.assertRaises(ValueError) as ctx5:
            check(rf_nonhex, self.env["reference"], self.tmp_dir / "out_nonhex.json", protocol_path=pf_nonhex)
        self.assertIn("Invalid entry in frozen_files manifest", str(ctx5.exception))

        # 6. Backslash path alias in frozen_files
        p_bs = copy.deepcopy(base_proto)
        del p_bs["frozen_files"]["rag_pensiones/chunking.py"]
        p_bs["frozen_files"]["rag_pensiones\\chunking.py"] = "a" * 64
        pf_bs = self.tmp_dir / "p_bs.json"
        pf_bs.write_text(json.dumps(p_bs), encoding="utf-8")
        r_bs = copy.deepcopy(results)
        r_bs["metadata"]["protocol_hash"] = canonical_hash(p_bs)
        rf_bs = self.tmp_dir / "r_bs.json"
        rf_bs.write_text(json.dumps(r_bs), encoding="utf-8")
        with self.assertRaises(ValueError) as ctx6:
            check(rf_bs, self.env["reference"], self.tmp_dir / "out_bs.json", protocol_path=pf_bs)
        self.assertTrue(any(term in str(ctx6.exception) for term in ("Path alias", "missing mandatory source paths")))

        # 7. ./ path alias in frozen_files
        p_dot = copy.deepcopy(base_proto)
        del p_dot["frozen_files"]["rag_pensiones/chunking.py"]
        p_dot["frozen_files"]["./rag_pensiones/chunking.py"] = "a" * 64
        pf_dot = self.tmp_dir / "p_dot.json"
        pf_dot.write_text(json.dumps(p_dot), encoding="utf-8")
        r_dot = copy.deepcopy(results)
        r_dot["metadata"]["protocol_hash"] = canonical_hash(p_dot)
        rf_dot = self.tmp_dir / "r_dot.json"
        rf_dot.write_text(json.dumps(r_dot), encoding="utf-8")
        with self.assertRaises(ValueError) as ctx7:
            check(rf_dot, self.env["reference"], self.tmp_dir / "out_dot.json", protocol_path=pf_dot)
        self.assertTrue(any(term in str(ctx7.exception) for term in ("Path alias", "missing mandatory source paths")))

        # 8. Historical source removal
        res_h, proto_h = _build_synthetic_historical_environment(self.tmp_dir, self.env, arm="mixed")
        ph_data = strict_load_json(proto_h.read_bytes())
        del ph_data["frozen_files"]["rag_pensiones/search.py"]
        pf_h = self.tmp_dir / "pf_h_nomod.json"
        pf_h.write_text(json.dumps(ph_data), encoding="utf-8")
        rh_data = strict_load_json(res_h.read_bytes())
        rh_data["protocol_sha256"] = file_sha256(pf_h)
        rf_h = self.tmp_dir / "rf_h_nomod.json"
        rf_h.write_text(json.dumps(rh_data), encoding="utf-8")
        with self.assertRaises(ValueError) as ctx8:
            check(rf_h, self.env["reference"], self.tmp_dir / "out_h_nomod.json", arm="mixed", protocol_path=pf_h)
        self.assertIn("missing mandatory source paths", str(ctx8.exception))

    def test_b02_negation_and_polarity_controls(self) -> None:
        results = _build_synthetic_passed_results(self.env["protocol"], self.env)
        ref_q = strict_load_json(self.env["reference"].read_bytes())["cases"]["b02"]["query"]

        # 1. Negative statement: 'Hasta 2050 el gasto supera el 4% anual, aunque no aumenta el número de pensiones y la pensión media.'
        case_neg = _build_synthetic_case(
            "b02", ref_q,
            claim_text="Hasta 2050 el gasto supera el 4% anual, aunque no aumenta el número de pensiones y la pensión media.",
            quote=QUOTE_B02,
        )
        r_neg = copy.deepcopy(results)
        r_neg["cases"][1] = case_neg
        rf_neg = self.tmp_dir / "b02_neg.json"
        rf_neg.write_text(json.dumps(r_neg), encoding="utf-8")
        rep_neg = check(rf_neg, self.env["reference"], self.tmp_dir / "out_b02_neg.json")
        self.assertFalse(rep_neg["overall_pass"])
        b02_diag_neg = next(c for c in rep_neg["case_results"] if c["case_id"] == "b02")
        self.assertFalse(b02_diag_neg["whole_correct"])
        self.assertIn("b02_main_missing_increase_mechanism", b02_diag_neg["failures"])

        # 2. Individual factor negation: 'aumenta el número de pensiones pero no de la pensión media'
        case_ifn = _build_synthetic_case(
            "b02", ref_q,
            claim_text="Hasta 2050 el gasto supera el 4% anual, impulsado por el aumento del número de pensiones pero no de la pensión media.",
            quote=QUOTE_B02,
        )
        r_ifn = copy.deepcopy(results)
        r_ifn["cases"][1] = case_ifn
        rf_ifn = self.tmp_dir / "b02_ifn.json"
        rf_ifn.write_text(json.dumps(r_ifn), encoding="utf-8")
        rep_ifn = check(rf_ifn, self.env["reference"], self.tmp_dir / "out_b02_ifn.json")
        self.assertFalse(rep_ifn["overall_pass"])
        b02_diag_ifn = next(c for c in rep_ifn["case_results"] if c["case_id"] == "b02")
        self.assertFalse(b02_diag_ifn["whole_correct"])
        self.assertIn("b02_main_missing_increase_mechanism", b02_diag_ifn["failures"])

        # 3. Adversarial supportive quote masking wrong mainclaim
        case_adv = _build_synthetic_case(
            "b02", ref_q,
            claim_text="Hasta 2050 el gasto supera el 4% anual, pero no aumenta el número de pensiones y la pensión media.",
            quote=QUOTE_B02,
        )
        r_adv = copy.deepcopy(results)
        r_adv["cases"][1] = case_adv
        rf_adv = self.tmp_dir / "b02_adv.json"
        rf_adv.write_text(json.dumps(r_adv), encoding="utf-8")
        rep_adv = check(rf_adv, self.env["reference"], self.tmp_dir / "out_b02_adv.json")
        self.assertFalse(rep_adv["overall_pass"])
        b02_diag_adv = next(c for c in rep_adv["case_results"] if c["case_id"] == "b02")
        self.assertFalse(b02_diag_adv["whole_correct"])
        self.assertIn("b02_main_missing_increase_mechanism", b02_diag_adv["failures"])

        # 4. Valid idiom: 'no solo ... sino tambien'
        case_sol = _build_synthetic_case(
            "b02", ref_q,
            claim_text="Hasta 2050 el gasto supera el 4% anual, impulsado no solo por el aumento del número de pensiones sino también de la pensión media.",
            quote=QUOTE_B02,
        )
        r_sol = copy.deepcopy(results)
        r_sol["cases"][1] = case_sol
        rf_sol = self.tmp_dir / "b02_sol.json"
        rf_sol.write_text(json.dumps(r_sol), encoding="utf-8")
        rep_sol = check(rf_sol, self.env["reference"], self.tmp_dir / "out_b02_sol.json")
        b02_diag_sol = next(c for c in rep_sol["case_results"] if c["case_id"] == "b02")
        self.assertTrue(b02_diag_sol["whole_correct"])

        # 5. Valid idiom: 'no obstante'
        case_obs = _build_synthetic_case(
            "b02", ref_q,
            claim_text="Hasta 2050 el gasto supera el 4% anual; no obstante aumentan el número de pensiones y la pensión media.",
            quote=QUOTE_B02,
        )
        r_obs = copy.deepcopy(results)
        r_obs["cases"][1] = case_obs
        rf_obs = self.tmp_dir / "b02_obs.json"
        rf_obs.write_text(json.dumps(r_obs), encoding="utf-8")
        rep_obs = check(rf_obs, self.env["reference"], self.tmp_dir / "out_b02_obs.json")
        b02_diag_obs = next(c for c in rep_obs["case_results"] if c["case_id"] == "b02")
        self.assertTrue(b02_diag_obs["whole_correct"])

        # 6. Valid idiom: 'no disminuyen; aumentan...'
        case_dsm = _build_synthetic_case(
            "b02", ref_q,
            claim_text="Hasta 2050 el gasto supera el 4% anual; no disminuyen, aumentan el número de pensiones y la pensión media.",
            quote=QUOTE_B02,
        )
        r_dsm = copy.deepcopy(results)
        r_dsm["cases"][1] = case_dsm
        rf_dsm = self.tmp_dir / "b02_dsm.json"
        rf_dsm.write_text(json.dumps(r_dsm), encoding="utf-8")
        rep_dsm = check(rf_dsm, self.env["reference"], self.tmp_dir / "out_b02_dsm.json")
        b02_diag_dsm = next(c for c in rep_dsm["case_results"] if c["case_id"] == "b02")
        self.assertTrue(b02_diag_dsm["whole_correct"])

        # 7. Valid: subject-first negated decrease followed by true increase
        case_snd = _build_synthetic_case(
            "b02", ref_q,
            claim_text="Hasta 2050 el gasto supera el 4% anual; la pensión media no disminuye; aumentan el número de pensiones y la pensión media.",
            quote=QUOTE_B02,
        )
        r_snd = copy.deepcopy(results)
        r_snd["cases"][1] = case_snd
        rf_snd = self.tmp_dir / "b02_snd.json"
        rf_snd.write_text(json.dumps(r_snd), encoding="utf-8")
        rep_snd = check(rf_snd, self.env["reference"], self.tmp_dir / "out_b02_snd.json")
        b02_diag_snd = next(c for c in rep_snd["case_results"] if c["case_id"] == "b02")
        self.assertTrue(b02_diag_snd["whole_correct"])

        # 8. Valid: subject-first compound negated decrease followed by true increase
        case_cnd = _build_synthetic_case(
            "b02", ref_q,
            claim_text="Hasta 2050 el gasto supera el 4% anual; el número de pensiones y la pensión media no disminuyen; aumentan.",
            quote=QUOTE_B02,
        )
        r_cnd = copy.deepcopy(results)
        r_cnd["cases"][1] = case_cnd
        rf_cnd = self.tmp_dir / "b02_cnd.json"
        rf_cnd.write_text(json.dumps(r_cnd), encoding="utf-8")
        rep_cnd = check(rf_cnd, self.env["reference"], self.tmp_dir / "out_b02_cnd.json")
        b02_diag_cnd = next(c for c in rep_cnd["case_results"] if c["case_id"] == "b02")
        self.assertTrue(b02_diag_cnd["whole_correct"])

        # 9. Invalid: factor-first negated increase
        case_fni = _build_synthetic_case(
            "b02", ref_q,
            claim_text="Hasta 2050 el gasto supera el 4% anual; el número de pensiones no aumenta pero la pensión media sí aumenta.",
            quote=QUOTE_B02,
        )
        r_fni = copy.deepcopy(results)
        r_fni["cases"][1] = case_fni
        rf_fni = self.tmp_dir / "b02_fni.json"
        rf_fni.write_text(json.dumps(r_fni), encoding="utf-8")
        rep_fni = check(rf_fni, self.env["reference"], self.tmp_dir / "out_b02_fni.json")
        self.assertFalse(rep_fni["overall_pass"])
        b02_diag_fni = next(c for c in rep_fni["case_results"] if c["case_id"] == "b02")
        self.assertFalse(b02_diag_fni["whole_correct"])
        self.assertIn("b02_main_missing_increase_mechanism", b02_diag_fni["failures"])

        # 10. Invalid: bare factor negation
        case_bfn = _build_synthetic_case(
            "b02", ref_q,
            claim_text="Hasta 2050 el gasto supera el 4% anual; aumenta la pensión media pero el número de pensiones no.",
            quote=QUOTE_B02,
        )
        r_bfn = copy.deepcopy(results)
        r_bfn["cases"][1] = case_bfn
        rf_bfn = self.tmp_dir / "b02_bfn.json"
        rf_bfn.write_text(json.dumps(r_bfn), encoding="utf-8")
        rep_bfn = check(rf_bfn, self.env["reference"], self.tmp_dir / "out_b02_bfn.json")
        self.assertFalse(rep_bfn["overall_pass"])
        b02_diag_bfn = next(c for c in rep_bfn["case_results"] if c["case_id"] == "b02")
        self.assertFalse(b02_diag_bfn["whole_correct"])
        self.assertIn("b02_main_missing_increase_mechanism", b02_diag_bfn["failures"])

        # 11. Invalid: no-decrease-alone without affirmative increase
        case_nda = _build_synthetic_case(
            "b02", ref_q,
            claim_text="Hasta 2050 el gasto supera el 4% anual; la pensión media no disminuye y el número de pensiones no cae.",
            quote=QUOTE_B02,
        )
        r_nda = copy.deepcopy(results)
        r_nda["cases"][1] = case_nda
        rf_nda = self.tmp_dir / "b02_nda.json"
        rf_nda.write_text(json.dumps(r_nda), encoding="utf-8")
        rep_nda = check(rf_nda, self.env["reference"], self.tmp_dir / "out_b02_nda.json")
        self.assertFalse(rep_nda["overall_pass"])
        b02_diag_nda = next(c for c in rep_nda["case_results"] if c["case_id"] == "b02")
        self.assertFalse(b02_diag_nda["whole_correct"])
        self.assertIn("b02_main_missing_increase_mechanism", b02_diag_nda["failures"])

        # 12. Invalid: cross-subject salary growth after negated decline
        case_cs_sal = _build_synthetic_case(
            "b02", ref_q,
            claim_text="Hasta 2050 el gasto supera el 4% anual; el número de pensiones y la pensión media no disminuyen; aumentan los salarios.",
            quote=QUOTE_B02,
        )
        r_cs_sal = copy.deepcopy(results)
        r_cs_sal["cases"][1] = case_cs_sal
        rf_cs_sal = self.tmp_dir / "b02_cs_sal.json"
        rf_cs_sal.write_text(json.dumps(r_cs_sal), encoding="utf-8")
        rep_cs_sal = check(rf_cs_sal, self.env["reference"], self.tmp_dir / "out_b02_cs_sal.json")
        self.assertFalse(rep_cs_sal["overall_pass"])
        b02_diag_cs_sal = next(c for c in rep_cs_sal["case_results"] if c["case_id"] == "b02")
        self.assertFalse(b02_diag_cs_sal["whole_correct"])
        self.assertIn("b02_main_missing_increase_mechanism", b02_diag_cs_sal["failures"])

        # 13. Invalid: cross-subject revenue growth after negated decline
        case_cs_rev = _build_synthetic_case(
            "b02", ref_q,
            claim_text="Hasta 2050 el gasto supera el 4% anual; el número de pensiones y la pensión media no disminuyen; aumentan los ingresos.",
            quote=QUOTE_B02,
        )
        r_cs_rev = copy.deepcopy(results)
        r_cs_rev["cases"][1] = case_cs_rev
        rf_cs_rev = self.tmp_dir / "b02_cs_rev.json"
        rf_cs_rev.write_text(json.dumps(r_cs_rev), encoding="utf-8")
        rep_cs_rev = check(rf_cs_rev, self.env["reference"], self.tmp_dir / "out_b02_cs_rev.json")
        self.assertFalse(rep_cs_rev["overall_pass"])
        b02_diag_cs_rev = next(c for c in rep_cs_rev["case_results"] if c["case_id"] == "b02")
        self.assertFalse(b02_diag_cs_rev["whole_correct"])
        self.assertIn("b02_main_missing_increase_mechanism", b02_diag_cs_rev["failures"])

        # 14. Valid: second clause expressly increases both pension factors after negated decline
        case_exp = _build_synthetic_case(
            "b02", ref_q,
            claim_text="Hasta 2050 el gasto supera el 4% anual; el número de pensiones y la pensión media no disminuyen; aumentan el número de pensiones y la pensión media.",
            quote=QUOTE_B02,
        )
        r_exp = copy.deepcopy(results)
        r_exp["cases"][1] = case_exp
        rf_exp = self.tmp_dir / "b02_exp.json"
        rf_exp.write_text(json.dumps(r_exp), encoding="utf-8")
        rep_exp = check(rf_exp, self.env["reference"], self.tmp_dir / "out_b02_exp.json")
        b02_diag_exp = next(c for c in rep_exp["case_results"] if c["case_id"] == "b02")
        self.assertTrue(b02_diag_exp["whole_correct"])

    def test_b05_negation_and_polarity_controls(self) -> None:
        results = _build_synthetic_passed_results(self.env["protocol"], self.env)
        ref_q = strict_load_json(self.env["reference"].read_bytes())["cases"]["b05"]["query"]

        # 1. Negative statement: 'No aumenta el número de pensiones y la pensión media.'
        case_neg = _build_synthetic_case(
            "b05", ref_q,
            claim_text="No aumenta el número de pensiones y la pensión media.",
            quote=QUOTE_B05,
        )
        r_neg = copy.deepcopy(results)
        r_neg["cases"][4] = case_neg
        rf_neg = self.tmp_dir / "b05_neg.json"
        rf_neg.write_text(json.dumps(r_neg), encoding="utf-8")
        rep_neg = check(rf_neg, self.env["reference"], self.tmp_dir / "out_b05_neg.json")
        self.assertFalse(rep_neg["overall_pass"])
        b05_diag_neg = next(c for c in rep_neg["case_results"] if c["case_id"] == "b05")
        self.assertFalse(b05_diag_neg["whole_correct"])
        self.assertIn("b05_main_missing_increase_cause", b05_diag_neg["failures"])

        # 2. Individual factor negation: 'Aumenta el número de pensiones pero la pensión media no aumenta.'
        case_ifn = _build_synthetic_case(
            "b05", ref_q,
            claim_text="Aumenta el número de pensiones pero la pensión media no aumenta.",
            quote=QUOTE_B05,
        )
        r_ifn = copy.deepcopy(results)
        r_ifn["cases"][4] = case_ifn
        rf_ifn = self.tmp_dir / "b05_ifn.json"
        rf_ifn.write_text(json.dumps(r_ifn), encoding="utf-8")
        rep_ifn = check(rf_ifn, self.env["reference"], self.tmp_dir / "out_b05_ifn.json")
        self.assertFalse(rep_ifn["overall_pass"])
        b05_diag_ifn = next(c for c in rep_ifn["case_results"] if c["case_id"] == "b05")
        self.assertFalse(b05_diag_ifn["whole_correct"])
        self.assertIn("b05_main_missing_increase_cause", b05_diag_ifn["failures"])

        # 3. Adversarial quote masking wrong mainclaim
        case_adv = _build_synthetic_case(
            "b05", ref_q,
            claim_text="No aumenta el número de pensiones y la pensión media.",
            quote=QUOTE_B05,
        )
        r_adv = copy.deepcopy(results)
        r_adv["cases"][4] = case_adv
        rf_adv = self.tmp_dir / "b05_adv.json"
        rf_adv.write_text(json.dumps(r_adv), encoding="utf-8")
        rep_adv = check(rf_adv, self.env["reference"], self.tmp_dir / "out_b05_adv.json")
        self.assertFalse(rep_adv["overall_pass"])
        b05_diag_adv = next(c for c in rep_adv["case_results"] if c["case_id"] == "b05")
        self.assertFalse(b05_diag_adv["whole_correct"])
        self.assertIn("b05_main_missing_increase_cause", b05_diag_adv["failures"])

        # 4. Ni/nunca/sin aumento variations
        for label, bad_claim in [
            ("ni", "Ni el número de pensiones ni la pensión media aumentan."),
            ("sin", "Sin aumento del número de pensiones y de la pensión media."),
            ("nunca", "Nunca aumenta el número de pensiones y la pensión media."),
        ]:
            with self.subTest(b05_negative_variant=label):
                case_v = _build_synthetic_case("b05", ref_q, claim_text=bad_claim, quote=QUOTE_B05)
                r_v = copy.deepcopy(results)
                r_v["cases"][4] = case_v
                rf_v = self.tmp_dir / f"b05_{label}.json"
                rf_v.write_text(json.dumps(r_v), encoding="utf-8")
                rep_v = check(rf_v, self.env["reference"], self.tmp_dir / f"out_b05_{label}.json")
                self.assertFalse(rep_v["overall_pass"])
                b05_diag_v = next(c for c in rep_v["case_results"] if c["case_id"] == "b05")
                self.assertFalse(b05_diag_v["whole_correct"])
                self.assertIn("b05_main_missing_increase_cause", b05_diag_v["failures"])

        # 5. Valid idiom: 'no solo ... sino tambien'
        case_sol = _build_synthetic_case(
            "b05", ref_q,
            claim_text="Aumenta debido a que no solo aumenta el número de pensiones sino también la pensión media.",
            quote=QUOTE_B05,
        )
        r_sol = copy.deepcopy(results)
        r_sol["cases"][4] = case_sol
        rf_sol = self.tmp_dir / "b05_sol.json"
        rf_sol.write_text(json.dumps(r_sol), encoding="utf-8")
        rep_sol = check(rf_sol, self.env["reference"], self.tmp_dir / "out_b05_sol.json")
        b05_diag_sol = next(c for c in rep_sol["case_results"] if c["case_id"] == "b05")
        self.assertTrue(b05_diag_sol["whole_correct"])

        # 6. Valid idiom: 'no obstante'
        case_obs = _build_synthetic_case(
            "b05", ref_q,
            claim_text="No obstante las restricciones, aumentan el número de pensiones y la pensión media.",
            quote=QUOTE_B05,
        )
        r_obs = copy.deepcopy(results)
        r_obs["cases"][4] = case_obs
        rf_obs = self.tmp_dir / "b05_obs.json"
        rf_obs.write_text(json.dumps(r_obs), encoding="utf-8")
        rep_obs = check(rf_obs, self.env["reference"], self.tmp_dir / "out_b05_obs.json")
        b05_diag_obs = next(c for c in rep_obs["case_results"] if c["case_id"] == "b05")
        self.assertTrue(b05_diag_obs["whole_correct"])

        # 7. Valid idiom: 'no disminuyen; aumentan...'
        case_dsm = _build_synthetic_case(
            "b05", ref_q,
            claim_text="No disminuyen; aumentan el número de pensiones y la pensión media.",
            quote=QUOTE_B05,
        )
        r_dsm = copy.deepcopy(results)
        r_dsm["cases"][4] = case_dsm
        rf_dsm = self.tmp_dir / "b05_dsm.json"
        rf_dsm.write_text(json.dumps(r_dsm), encoding="utf-8")
        rep_dsm = check(rf_dsm, self.env["reference"], self.tmp_dir / "out_b05_dsm.json")
        b05_diag_dsm = next(c for c in rep_dsm["case_results"] if c["case_id"] == "b05")
        self.assertTrue(b05_diag_dsm["whole_correct"])

        # 8. Valid: subject-first negated decrease followed by true increase
        case_snd = _build_synthetic_case(
            "b05", ref_q,
            claim_text="La pensión media no disminuye; aumentan el número de pensiones y la pensión media.",
            quote=QUOTE_B05,
        )
        r_snd = copy.deepcopy(results)
        r_snd["cases"][4] = case_snd
        rf_snd = self.tmp_dir / "b05_snd.json"
        rf_snd.write_text(json.dumps(r_snd), encoding="utf-8")
        rep_snd = check(rf_snd, self.env["reference"], self.tmp_dir / "out_b05_snd.json")
        b05_diag_snd = next(c for c in rep_snd["case_results"] if c["case_id"] == "b05")
        self.assertTrue(b05_diag_snd["whole_correct"])

        # 9. Valid: subject-first compound negated decrease followed by true increase
        case_cnd = _build_synthetic_case(
            "b05", ref_q,
            claim_text="El número de pensiones y la pensión media no disminuyen; aumentan.",
            quote=QUOTE_B05,
        )
        r_cnd = copy.deepcopy(results)
        r_cnd["cases"][4] = case_cnd
        rf_cnd = self.tmp_dir / "b05_cnd.json"
        rf_cnd.write_text(json.dumps(r_cnd), encoding="utf-8")
        rep_cnd = check(rf_cnd, self.env["reference"], self.tmp_dir / "out_b05_cnd.json")
        b05_diag_cnd = next(c for c in rep_cnd["case_results"] if c["case_id"] == "b05")
        self.assertTrue(b05_diag_cnd["whole_correct"])

        # 10. Invalid: factor-first negated increase
        case_fni = _build_synthetic_case(
            "b05", ref_q,
            claim_text="El número de pensiones no aumenta pero la pensión media sí aumenta.",
            quote=QUOTE_B05,
        )
        r_fni = copy.deepcopy(results)
        r_fni["cases"][4] = case_fni
        rf_fni = self.tmp_dir / "b05_fni.json"
        rf_fni.write_text(json.dumps(r_fni), encoding="utf-8")
        rep_fni = check(rf_fni, self.env["reference"], self.tmp_dir / "out_b05_fni.json")
        self.assertFalse(rep_fni["overall_pass"])
        b05_diag_fni = next(c for c in rep_fni["case_results"] if c["case_id"] == "b05")
        self.assertFalse(b05_diag_fni["whole_correct"])
        self.assertIn("b05_main_missing_increase_cause", b05_diag_fni["failures"])

        # 11. Invalid: bare factor negation
        case_bfn = _build_synthetic_case(
            "b05", ref_q,
            claim_text="Aumenta la pensión media pero el número de pensiones no.",
            quote=QUOTE_B05,
        )
        r_bfn = copy.deepcopy(results)
        r_bfn["cases"][4] = case_bfn
        rf_bfn = self.tmp_dir / "b05_bfn.json"
        rf_bfn.write_text(json.dumps(r_bfn), encoding="utf-8")
        rep_bfn = check(rf_bfn, self.env["reference"], self.tmp_dir / "out_b05_bfn.json")
        self.assertFalse(rep_bfn["overall_pass"])
        b05_diag_bfn = next(c for c in rep_bfn["case_results"] if c["case_id"] == "b05")
        self.assertFalse(b05_diag_bfn["whole_correct"])
        self.assertIn("b05_main_missing_increase_cause", b05_diag_bfn["failures"])

        # 12. Invalid: no-decrease-alone without affirmative increase
        case_nda = _build_synthetic_case(
            "b05", ref_q,
            claim_text="La pensión media no disminuye y el número de pensiones no cae.",
            quote=QUOTE_B05,
        )
        r_nda = copy.deepcopy(results)
        r_nda["cases"][4] = case_nda
        rf_nda = self.tmp_dir / "b05_nda.json"
        rf_nda.write_text(json.dumps(r_nda), encoding="utf-8")
        rep_nda = check(rf_nda, self.env["reference"], self.tmp_dir / "out_b05_nda.json")
        self.assertFalse(rep_nda["overall_pass"])
        b05_diag_nda = next(c for c in rep_nda["case_results"] if c["case_id"] == "b05")
        self.assertFalse(b05_diag_nda["whole_correct"])
        self.assertIn("b05_main_missing_increase_cause", b05_diag_nda["failures"])

        # 13. Invalid: cross-subject salary growth after negated decline
        case_cs_sal = _build_synthetic_case(
            "b05", ref_q,
            claim_text="El número de pensiones y la pensión media no disminuyen; aumentan los salarios.",
            quote=QUOTE_B05,
        )
        r_cs_sal = copy.deepcopy(results)
        r_cs_sal["cases"][4] = case_cs_sal
        rf_cs_sal = self.tmp_dir / "b05_cs_sal.json"
        rf_cs_sal.write_text(json.dumps(r_cs_sal), encoding="utf-8")
        rep_cs_sal = check(rf_cs_sal, self.env["reference"], self.tmp_dir / "out_b05_cs_sal.json")
        self.assertFalse(rep_cs_sal["overall_pass"])
        b05_diag_cs_sal = next(c for c in rep_cs_sal["case_results"] if c["case_id"] == "b05")
        self.assertFalse(b05_diag_cs_sal["whole_correct"])
        self.assertIn("b05_main_missing_increase_cause", b05_diag_cs_sal["failures"])

        # 14. Invalid: cross-subject revenue growth after negated decline
        case_cs_rev = _build_synthetic_case(
            "b05", ref_q,
            claim_text="El número de pensiones y la pensión media no disminuyen; aumentan los ingresos.",
            quote=QUOTE_B05,
        )
        r_cs_rev = copy.deepcopy(results)
        r_cs_rev["cases"][4] = case_cs_rev
        rf_cs_rev = self.tmp_dir / "b05_cs_rev.json"
        rf_cs_rev.write_text(json.dumps(r_cs_rev), encoding="utf-8")
        rep_cs_rev = check(rf_cs_rev, self.env["reference"], self.tmp_dir / "out_b05_cs_rev.json")
        self.assertFalse(rep_cs_rev["overall_pass"])
        b05_diag_cs_rev = next(c for c in rep_cs_rev["case_results"] if c["case_id"] == "b05")
        self.assertFalse(b05_diag_cs_rev["whole_correct"])
        self.assertIn("b05_main_missing_increase_cause", b05_diag_cs_rev["failures"])

        # 15. Valid: second clause expressly increases both pension factors after negated decline
        case_exp = _build_synthetic_case(
            "b05", ref_q,
            claim_text="El número de pensiones y la pensión media no disminuyen; aumentan el número de pensiones y la pensión media.",
            quote=QUOTE_B05,
        )
        r_exp = copy.deepcopy(results)
        r_exp["cases"][4] = case_exp
        rf_exp = self.tmp_dir / "b05_exp.json"
        rf_exp.write_text(json.dumps(r_exp), encoding="utf-8")
        rep_exp = check(rf_exp, self.env["reference"], self.tmp_dir / "out_b05_exp.json")
        b05_diag_exp = next(c for c in rep_exp["case_results"] if c["case_id"] == "b05")
        self.assertTrue(b05_diag_exp["whole_correct"])


if __name__ == "__main__":
    unittest.main()
