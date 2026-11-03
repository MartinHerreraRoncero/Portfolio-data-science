"""Exhaustive offline tests for table-grounded RAG pipeline (Phase 2).

Tests fake clients/transport ONLY:
- No-table identical delegation (normal queries delegate unchanged).
- Factory opt-in, default behavior, and all27 arm rejection.
- Actual expanded vs unexpanded tables (retrieval absence detection).
- t06 two-row-style package with multi-concept typed facts.
- 9B wrong concept, year, value, delta, unit, and extra prose cannot publish.
- Supported typed facts pass guard and are literally cited.
- Missing and unsupported explicit requests safely refuse without narrative fallback.
- Malformed JSON technical failures are truthfully retained.
- Repair cannot evade numeric guard.
- Budget failure safe handling.
- Interleaved requests state isolation (no state leakage).
- All raw model payloads retained and expected model roles preserved.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import unittest
from typing import Any

from experiments.compare_mixed_single27 import (
    DEFAULT_PARAMETERS,
    EFFECTIVE_GENERATION_OPTIONS,
    EMBEDDING_4B_MODEL,
    GENERATOR_9B_MODEL,
    PINNED_9B_DIGEST,
    PINNED_27B_DIGEST,
    PINNED_EMBEDDING_DIGEST,
    SERVER_VERSION,
    TEXT_27B_MODEL,
    execute_attempt,
)
from experiments.preserved_q3_verifier import (
    Q3VerifierClient,
    VERIFIER_Q3_DIGEST,
    VERIFIER_Q3_MODEL,
    VERIFIER_Q3_PARAMETERS,
)
from experiments.table_grounded import (
    TABLE_FACT_SCHEMA,
    TABLE_GENERATION_SCHEMA,
    TABLE_REPAIR_SCHEMA,
    TableGrounder,
    execute_table_attempt,
)
from rag_pensiones.document_layout import _format_table_as_text
from rag_pensiones.grounding import collapse_spaces
from rag_pensiones.numeric_guard import parse_table_request
from rag_pensiones.qwen_embeddings import DIMENSION, QUERY_PREFIX, SCHEMA
from rag_pensiones.table_evidence import (
    MAX_SUM_QUOTE_UTF8_BYTES,
    build_table_packets,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def make_raw_chat_response(
    model: str,
    content_obj: dict[str, Any] | str,
    prompt_eval: int = 150,
    eval_count: int = 45,
) -> dict[str, Any]:
    """Constructs a raw Ollama chat completion response object."""
    if isinstance(content_obj, str):
        content_str = content_obj
    else:
        content_str = json.dumps(content_obj, ensure_ascii=False)

    return {
        "model": model,
        "done": True,
        "done_reason": "stop",
        "message": {"role": "assistant", "content": content_str},
        "prompt_eval_count": prompt_eval,
        "eval_count": eval_count,
        "eval_duration": 400000000,
        "total_duration": 900000000,
        "load_duration": 10000000,
        "prompt_eval_duration": 50000000,
    }



def _extract_required_fact_keys(fmt: dict[str, Any]) -> set[str] | None:
    """Extracts required keys from the fact schema passed in request format."""
    if not isinstance(fmt, dict):
        return None
    for b in fmt.get("anyOf", [fmt]):
        if not isinstance(b, dict):
            continue
        props = b.get("properties", {})
        for k in ("facts", "claims"):
            if k in props and isinstance(props[k], dict):
                items = props[k].get("items")
                if isinstance(items, dict) and "required" in items:
                    return set(items["required"])
        if "repairs" in props and isinstance(props["repairs"], dict):
            rep_items = props["repairs"].get("items")
            if isinstance(rep_items, dict):
                for rb in rep_items.get("anyOf", [rep_items]):
                    if not isinstance(rb, dict):
                        continue
                    rb_props = rb.get("properties", {})
                    if "facts" in rb_props and isinstance(rb_props["facts"], dict):
                        f_items = rb_props["facts"].get("items")
                        if isinstance(f_items, dict) and "required" in f_items:
                            return set(f_items["required"])
                    if "fact" in rb_props and isinstance(rb_props["fact"], dict):
                        if "required" in rb_props["fact"]:
                            return set(rb_props["fact"]["required"])
    return None


class PipelineFakeTransport:
    """Mock Ollama transport for offline testing of table-grounded pipeline."""

    def __init__(
        self,
        *,
        version: str = SERVER_VERSION,
        resident_models: list[str] | None = None,
        chat_responses: list[Any] | None = None,
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
            if callable(resp):
                resp = resp(request)
            if isinstance(resp, Exception):
                raise resp
            return copy.deepcopy(resp)

        # Fallback automated responses
        fmt = request.get("format", {})
        user_msg = next((m for m in request.get("messages", []) if m.get("role") == "user"), None)
        user_data: dict[str, Any] = {}
        if user_msg and isinstance(user_msg.get("content"), str):
            try:
                user_data = json.loads(user_msg["content"])
            except Exception:
                user_data = {}

        is_selection = ("properties" in fmt and "ids" in fmt["properties"]) or any(
            "properties" in b and "ids" in b["properties"] for b in fmt.get("anyOf", [])
        )
        is_generation = ("properties" in fmt and ("claims" in fmt["properties"] or "facts" in fmt["properties"])) or any(
            "properties" in b and ("claims" in b["properties"] or "facts" in b["properties"])
            for b in fmt.get("anyOf", [])
        )
        is_repair = "properties" in fmt and "repairs" in fmt["properties"]
        is_verification = ("properties" in fmt and "verdicts" in fmt["properties"]) or any(
            "properties" in b and "verdicts" in b["properties"] for b in fmt.get("anyOf", [])
        )
        is_coverage = ("properties" in fmt and "parts" in fmt["properties"]) or any(
            "properties" in b and "parts" in b["properties"] for b in fmt.get("anyOf", [])
        )

        if is_selection:
            props = fmt.get("properties", {})
            status_prop = props.get("status", {})
            only_insufficient = (
                status_prop.get("const") == "insufficient_evidence"
                or status_prop.get("enum") == ["insufficient_evidence"]
                or props.get("ids", {}).get("maxItems") == 0
            )
            if only_insufficient:
                content_obj = {
                    "status": "insufficient_evidence",
                    "ids": [],
                    "reason": user_data.get("aviso") or "Evidencia insuficiente para responder a la consulta sobre la tabla.",
                }
            else:
                content_obj = {"status": "selected", "ids": ["P1"], "reason": ""}
        elif is_generation:
            props = fmt.get("properties", {})
            status_prop = props.get("status", {})
            only_insufficient = (
                status_prop.get("const") == "insufficient_evidence"
                or status_prop.get("enum") == ["insufficient_evidence"]
                or props.get("facts", {}).get("maxItems") == 0
                or props.get("claims", {}).get("maxItems") == 0
            )
            if only_insufficient:
                content_obj = {
                    "status": "insufficient_evidence",
                    "facts": [],
                    "claims": [],
                    "reason": user_data.get("aviso") or "Evidencia insuficiente por límite de contexto.",
                }
            else:
                t_ctx = user_data.get("tabla_contexto")
                if t_ctx and t_ctx.get("celdas_disponibles"):
                    req_keys = _extract_required_fact_keys(fmt)
                    facts = []
                    for cell in t_ctx["celdas_disponibles"]:
                        cand = {
                            "cell_id": cell["cell_id"],
                            "concept": cell["concept"],
                            "header_path": cell["header_path"],
                            "year": cell.get("year"),
                            "ordered_delta_years": cell.get("ordered_delta_years"),
                            "unit": cell["unit"],
                            "measure": cell["measure"],
                            "value": cell["value"],
                            "doc_id": t_ctx["doc_id"],
                            "source": t_ctx["source"],
                            "date": t_ctx["date"],
                            "page": t_ctx["page"],
                            "table_id": t_ctx["table_id"],
                            "extraction_id": t_ctx["extraction_id"],
                            "evidence_ids": list(cell.get("evidence_ids", [])),
                        }
                        fact = {k: v for k, v in cand.items() if k in req_keys} if req_keys else cand
                        facts.append(fact)
                    content_obj = {"status": "answered", "facts": facts, "reason": ""}
                else:
                    content_obj = {
                        "status": "answered",
                        "claims": [
                            {
                                "claim_id": "C1",
                                "text": "El gasto en pensiones alcanzará el 16,1% en 2050.",
                                "evidence_ids": ["E1"],
                            }
                        ],
                        "reason": "",
                    }
        elif is_repair:
            repairs = []
            for claim in user_data.get("afirmaciones_a_reparar", [{"claim_id": "C1"}]):
                cid = claim.get("claim_id", "C1")
                t_ctx = user_data.get("tabla_contexto", {})
                cells = t_ctx.get("celdas_disponibles", [])
                matching_cell = None
                claim_text = claim.get("text", "")
                for cell in cells:
                    if cell.get("concept") and cell["concept"].lower() in claim_text.lower():
                        matching_cell = cell
                        break
                if matching_cell is None:
                    try:
                        c_idx = int(cid[1:]) - 1
                        if 0 <= c_idx < len(cells):
                            matching_cell = cells[c_idx]
                    except Exception:
                        pass
                if matching_cell is None and cells:
                    matching_cell = cells[0]
                if matching_cell:
                    req_keys = _extract_required_fact_keys(fmt)
                    cand = {
                        "cell_id": matching_cell["cell_id"],
                        "concept": matching_cell["concept"],
                        "header_path": matching_cell["header_path"],
                        "year": matching_cell.get("year"),
                        "ordered_delta_years": matching_cell.get("ordered_delta_years"),
                        "unit": matching_cell["unit"],
                        "measure": matching_cell["measure"],
                        "value": matching_cell["value"],
                        "doc_id": t_ctx["doc_id"],
                        "source": t_ctx["source"],
                        "date": t_ctx["date"],
                        "page": t_ctx["page"],
                        "table_id": t_ctx["table_id"],
                        "extraction_id": t_ctx["extraction_id"],
                        "evidence_ids": list(matching_cell.get("evidence_ids", [])),
                    }
                    fact = {k: v for k, v in cand.items() if k in req_keys} if req_keys else cand
                    repairs.append({
                        "claim_id": cid,
                        "status": "answered",
                        "facts": [fact],
                        "reason": "",
                    })
                else:
                    repairs.append({
                        "claim_id": cid,
                        "status": "insufficient_evidence",
                        "reason": "Evidencia insuficiente para reparar afirmación.",
                    })
            content_obj = {"repairs": repairs}
        elif is_verification:
            verdicts = []
            afirmaciones = user_data.get("afirmaciones", [])
            for c in afirmaciones:
                cid = c.get("claim_id", "C1")
                eids = c.get("evidence_ids", ["E1"])
                verdicts.append({
                    "claim_id": cid,
                    "verdict": "supported",
                    "reason": "Respaldo comprobado.",
                    "evidence_verdicts": [{"evidence_id": eid, "relation": "supports"} for eid in eids],
                })
            if not verdicts:
                verdicts = [
                    {
                        "claim_id": "C1",
                        "verdict": "supported",
                        "reason": "Respaldo comprobado.",
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
            if not query_text:
                query_text = "Consulta"
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
        else:
            content_obj = {"status": "ok"}

        return make_raw_chat_response(req_model, content_obj)


def _build_synthetic_fixtures() -> tuple[dict, dict, list[dict], dict]:
    """Builds synthetic table bundle, index, pages, and units for offline tests."""
    doc_id = "doc_table_synth_abcdef1234567890abcdef1234567890"
    source_file = "informe_pensiones_sintetico_2026.pdf"
    published_at = "2026-01-15"
    extraction_id = "synth_ext_01"

    table_data = {
        "table_id": "table_synth_cuadro1",
        "method": "reviewed_profile",
        "native_ref": None,
        "page_numbers": [95],
        "bbox": {"l": 100.0, "t": 200.0, "r": 500.0, "b": 350.0, "coord_origin": "TOPLEFT"},
        "dimensions": {"rows": 3, "cols": 4},
        "caption_text": "CUADRO 1. EMPLEO Y SALARIOS (% PIB)",
        "footnotes": ["Fuente: Estimación sintética AIReF"],
        "cells": [
            # Headers
            {
                "row_start": 0, "row_end": 1, "col_start": 0, "col_end": 1,
                "text": "Concepto", "raw_text": "Concepto", "is_empty": False,
                "page_number": 95, "bbox": {"l": 100.0, "t": 200.0, "r": 200.0, "b": 230.0, "coord_origin": "TOPLEFT"},
                "column_header": True, "row_header": True, "header_path": [], "row_label": "",
                "cell_id": "c_r0c0",
            },
            {
                "row_start": 0, "row_end": 1, "col_start": 1, "col_end": 2,
                "text": "2025", "raw_text": "2025", "is_empty": False,
                "page_number": 95, "bbox": {"l": 200.0, "t": 200.0, "r": 300.0, "b": 230.0, "coord_origin": "TOPLEFT"},
                "column_header": True, "row_header": False, "header_path": [], "row_label": "",
                "cell_id": "c_r0c1",
            },
            {
                "row_start": 0, "row_end": 1, "col_start": 2, "col_end": 3,
                "text": "2030", "raw_text": "2030", "is_empty": False,
                "page_number": 95, "bbox": {"l": 300.0, "t": 200.0, "r": 400.0, "b": 230.0, "coord_origin": "TOPLEFT"},
                "column_header": True, "row_header": False, "header_path": [], "row_label": "",
                "cell_id": "c_r0c2",
            },
            {
                "row_start": 0, "row_end": 1, "col_start": 3, "col_end": 4,
                "text": "Δ 2030-2025", "raw_text": "Δ 2030-2025", "is_empty": False,
                "page_number": 95, "bbox": {"l": 400.0, "t": 200.0, "r": 500.0, "b": 230.0, "coord_origin": "TOPLEFT"},
                "column_header": True, "row_header": False, "header_path": [], "row_label": "",
                "cell_id": "c_r0c3",
            },
            # Row 1: Empleo
            {
                "row_start": 1, "row_end": 2, "col_start": 0, "col_end": 1,
                "text": "Empleo", "raw_text": "Empleo", "is_empty": False,
                "page_number": 95, "bbox": {"l": 100.0, "t": 230.0, "r": 200.0, "b": 280.0, "coord_origin": "TOPLEFT"},
                "column_header": False, "row_header": True, "header_path": [], "row_label": "Empleo",
                "cell_id": "c_r1c0",
            },
            {
                "row_start": 1, "row_end": 2, "col_start": 1, "col_end": 2,
                "text": "10,2", "raw_text": "10,2", "is_empty": False,
                "page_number": 95, "bbox": {"l": 200.0, "t": 230.0, "r": 300.0, "b": 280.0, "coord_origin": "TOPLEFT"},
                "column_header": False, "row_header": False, "header_path": ["2025"], "row_label": "Empleo",
                "cell_id": "c_r1c1",
            },
            {
                "row_start": 1, "row_end": 2, "col_start": 2, "col_end": 3,
                "text": "10,5", "raw_text": "10,5", "is_empty": False,
                "page_number": 95, "bbox": {"l": 300.0, "t": 230.0, "r": 400.0, "b": 280.0, "coord_origin": "TOPLEFT"},
                "column_header": False, "row_header": False, "header_path": ["2030"], "row_label": "Empleo",
                "cell_id": "c_r1c2",
            },
            {
                "row_start": 1, "row_end": 2, "col_start": 3, "col_end": 4,
                "text": "0,3", "raw_text": "0,3", "is_empty": False,
                "page_number": 95, "bbox": {"l": 400.0, "t": 230.0, "r": 500.0, "b": 280.0, "coord_origin": "TOPLEFT"},
                "column_header": False, "row_header": False, "header_path": ["Δ 2030-2025"], "row_label": "Empleo",
                "cell_id": "c_r1c3",
            },
            # Row 2: Salarios
            {
                "row_start": 2, "row_end": 3, "col_start": 0, "col_end": 1,
                "text": "Salarios", "raw_text": "Salarios", "is_empty": False,
                "page_number": 95, "bbox": {"l": 100.0, "t": 280.0, "r": 200.0, "b": 330.0, "coord_origin": "TOPLEFT"},
                "column_header": False, "row_header": True, "header_path": [], "row_label": "Salarios",
                "cell_id": "c_r2c0",
            },
            {
                "row_start": 2, "row_end": 3, "col_start": 1, "col_end": 2,
                "text": "8,2", "raw_text": "8,2", "is_empty": False,
                "page_number": 95, "bbox": {"l": 200.0, "t": 280.0, "r": 300.0, "b": 330.0, "coord_origin": "TOPLEFT"},
                "column_header": False, "row_header": False, "header_path": ["2025"], "row_label": "Salarios",
                "cell_id": "c_r2c1",
            },
            {
                "row_start": 2, "row_end": 3, "col_start": 2, "col_end": 3,
                "text": "7,8", "raw_text": "7,8", "is_empty": False,
                "page_number": 95, "bbox": {"l": 300.0, "t": 280.0, "r": 400.0, "b": 330.0, "coord_origin": "TOPLEFT"},
                "column_header": False, "row_header": False, "header_path": ["2030"], "row_label": "Salarios",
                "cell_id": "c_r2c2",
            },
            {
                "row_start": 2, "row_end": 3, "col_start": 3, "col_end": 4,
                "text": "−0,4", "raw_text": "−0,4", "is_empty": False,
                "page_number": 95, "bbox": {"l": 400.0, "t": 280.0, "r": 500.0, "b": 330.0, "coord_origin": "TOPLEFT"},
                "column_header": False, "row_header": False, "header_path": ["Δ 2030-2025"], "row_label": "Salarios",
                "cell_id": "c_r2c3",
            },
        ],
    }

    formatted_table = _format_table_as_text(table_data)
    preamble = "En el escenario macroeconómico y fiscal se evalúan las siguientes magnitudes detalladas.\n\n"
    epilogue = "\n\nEstimaciones sujetas a la evolución demográfica del periodo proyectado."
    page_text = preamble + formatted_table + epilogue

    bundle_page = {
        "page_number": 95,
        "width": 595.32,
        "height": 841.92,
        "blocks": [
            {
                "type": "paragraph",
                "text": preamble.strip(),
                "bbox": {"l": 100.0, "t": 100.0, "r": 500.0, "b": 180.0, "coord_origin": "TOPLEFT"},
            },
            {
                "type": "table",
                "text": formatted_table,
                "bbox": {"l": 100.0, "t": 200.0, "r": 500.0, "b": 350.0, "coord_origin": "TOPLEFT"},
                "table_id": "table_synth_cuadro1",
            },
            {
                "type": "paragraph",
                "text": epilogue.strip(),
                "bbox": {"l": 100.0, "t": 360.0, "r": 500.0, "b": 420.0, "coord_origin": "TOPLEFT"},
            },
        ],
    }

    table_bundle = {
        "schema_version": "document-layout-v1",
        "metadata": {
            "doc_id": doc_id,
            "source_file": source_file,
            "published_at": published_at,
            "extraction_id": extraction_id,
        },
        "pages": [bundle_page],
        "tables": [table_data],
    }

    canonical_page = {
        "chunk_id": "page_95",
        "doc_id": doc_id,
        "source_file": source_file,
        "published_at": published_at,
        "page_number": 95,
        "start_char": 0,
        "end_char": len(page_text),
        "text": page_text,
    }

    pages_data = [canonical_page]

    # Set quote using collapse_spaces to strictly maintain canonical provenance
    units_data = {
        "schema_version": "reconstructed-units-catalog-v1",
        "units": [
            {
                "id": "U1",
                "spans": [
                    {
                        "doc_id": doc_id,
                        "source_file": source_file,
                        "published_at": published_at,
                        "page_number": 95,
                        "start_char": 0,
                        "end_char": len(page_text),
                        "quote": collapse_spaces(page_text),
                    }
                ],
            }
        ],
    }

    norm_val = 1.0 / math.sqrt(DIMENSION)
    index_data = {
        "schema_version": SCHEMA,
        "model": EMBEDDING_4B_MODEL,
        "model_digest": PINNED_EMBEDDING_DIGEST,
        "dimension": DIMENSION,
        "document_prefix": "",
        "query_prefix": QUERY_PREFIX,
        "base_index_hash": "0" * 64,
        "chunks": [
            {
                "chunk_id": "chunk_page_95",
                "doc_id": doc_id,
                "source_file": source_file,
                "published_at": published_at,
                "page_number": 95,
                "start_char": 0,
                "end_char": len(page_text),
                "text": page_text,
                "strategy": "layout_block",
                "embedding": [norm_val] * DIMENSION,
            }
        ],
    }

    return table_bundle, index_data, pages_data, units_data


class TestTableGroundedPipeline(unittest.TestCase):
    """Exhaustive unit test suite for TableGrounder and execute_table_attempt."""

    def setUp(self) -> None:
        self.bundle, self.index, self.pages, self.units = _build_synthetic_fixtures()

    def test_no_table_identical_delegation(self) -> None:
        """Normal query with no explicit table reference delegates pipeline unchanged."""
        attempt_spec = {
            "attempt_index": 1,
            "case_id": "case_narrative_b01",
            "query": "¿Qué evolución demográfica proyecta el informe para el periodo analizado?",
            "arm": "mixed",
        }

        transport_default = PipelineFakeTransport()
        res_default = execute_attempt(
            attempt_spec,
            index_data=self.index,
            units_data=self.units,
            pages_data=self.pages,
            transport=transport_default,
            grounder_factory=None,
        )

        transport_table = PipelineFakeTransport()
        res_table = execute_table_attempt(
            attempt_spec,
            index_data=self.index,
            units_data=self.units,
            pages_data=self.pages,
            transport=transport_table,
            table_bundle=self.bundle,
            enabled=True,
        )

        self.assertEqual(res_default["status"], res_table["status"])
        self.assertEqual(res_default["formatted_answer"], res_table["formatted_answer"])
        self.assertEqual(len(transport_default.chat_calls), len(transport_table.chat_calls))

        audit = res_table["table_audit"]
        self.assertFalse(audit["is_table_query"])
        self.assertEqual(audit["gate_decision"], "bypassed")

    def test_factory_opt_in_default_and_all27_validation(self) -> None:
        """Factory opt-in works for mixed arm and rejects all27 before inference."""
        attempt_spec_all27 = {
            "attempt_index": 2,
            "case_id": "case_all27_reject",
            "query": "Según el cuadro 1, ¿cuál es el porcentaje para Empleo?",
            "arm": "all27",
        }
        transport = PipelineFakeTransport()

        with self.assertRaisesRegex(ValueError, "only supported for the 'mixed' arm"):
            execute_attempt(
                attempt_spec_all27,
                index_data=self.index,
                units_data=self.units,
                pages_data=self.pages,
                transport=transport,
                grounder_factory=lambda *a, **k: None,
            )
        self.assertEqual(transport.chat_calls, [])

        with self.assertRaisesRegex(ValueError, "only supported for the 'mixed' arm"):
            execute_table_attempt(
                attempt_spec_all27,
                index_data=self.index,
                units_data=self.units,
                pages_data=self.pages,
                transport=transport,
                table_bundle=self.bundle,
                enabled=True,
            )
        self.assertEqual(transport.chat_calls, [])

        res_all27_default = execute_attempt(
            attempt_spec_all27,
            index_data=self.index,
            units_data=self.units,
            pages_data=self.pages,
            transport=transport,
            grounder_factory=None,
        )
        self.assertEqual(res_all27_default["arm"], "all27")

    def test_actual_expanded_vs_unexpanded_tables(self) -> None:
        """Contained table expands and builds package; unexpanded table safely refuses with retrieval_absence."""
        query = "Según el cuadro 1, ¿cuál es el valor de Empleo en 2025?"
        attempt_spec = {
            "attempt_index": 3,
            "case_id": "case_unexpanded",
            "query": query,
            "arm": "mixed",
        }

        # Case: Table on page 95, but index and units only retrieve page 1
        page_1_text = "Resumen de portada sin tablas en la pagina 1 de referencia general."
        unexpanded_pages = [
            {
                "chunk_id": "page_1",
                "doc_id": self.pages[0]["doc_id"],
                "source_file": self.pages[0]["source_file"],
                "published_at": self.pages[0]["published_at"],
                "page_number": 1,
                "start_char": 0,
                "end_char": len(page_1_text),
                "text": page_1_text,
            },
            self.pages[0],
        ]
        unexpanded_units = {
            "schema_version": "reconstructed-units-catalog-v1",
            "units": [
                {
                    "id": "U1",
                    "spans": [
                        {
                            "doc_id": self.pages[0]["doc_id"],
                            "source_file": self.pages[0]["source_file"],
                            "published_at": self.pages[0]["published_at"],
                            "page_number": 1,
                            "start_char": 0,
                            "end_char": len(page_1_text),
                            "quote": collapse_spaces(page_1_text),
                        }
                    ],
                }
            ],
        }
        norm_val = 1.0 / math.sqrt(DIMENSION)
        unexpanded_index = {
            "schema_version": SCHEMA,
            "model": EMBEDDING_4B_MODEL,
            "model_digest": PINNED_EMBEDDING_DIGEST,
            "dimension": DIMENSION,
            "document_prefix": "",
            "query_prefix": QUERY_PREFIX,
            "base_index_hash": "0" * 64,
            "chunks": [
                {
                    "chunk_id": "chunk_page_1",
                    "doc_id": self.pages[0]["doc_id"],
                    "source_file": self.pages[0]["source_file"],
                    "published_at": self.pages[0]["published_at"],
                    "page_number": 1,
                    "start_char": 0,
                    "end_char": len(page_1_text),
                    "text": page_1_text,
                    "strategy": "layout_block",
                    "embedding": [norm_val] * DIMENSION,
                }
            ],
        }

        transport_unexpanded = PipelineFakeTransport()
        res_unexpanded = execute_table_attempt(
            attempt_spec,
            index_data=unexpanded_index,
            units_data=unexpanded_units,
            pages_data=unexpanded_pages,
            transport=transport_unexpanded,
            table_bundle=self.bundle,
            enabled=True,
        )

        self.assertEqual(res_unexpanded["status"], "completed")
        self.assertEqual(res_unexpanded["factual_outcome"]["status"], "insufficient_evidence")
        self.assertEqual(len(transport_unexpanded.chat_calls), 1)
        self.assertEqual(len(transport_unexpanded.embed_calls), 1)
        self.assertNotEqual(res_unexpanded["factual_outcome"]["full_result"]["prompt_sha256"], "0" * 64)
        self.assertGreater(res_unexpanded["factual_outcome"]["full_result"]["stats"]["prompt_eval_count"], 0)
        self.assertTrue(res_unexpanded["table_audit"]["retrieval_absence"])
        self.assertEqual(res_unexpanded["table_audit"]["gate_decision"], "refused")
        self.assertEqual(res_unexpanded["table_audit"]["gate_stage"], "retrieval_expansion")

    def test_supported_typed_facts_pass_and_are_literally_cited(self) -> None:
        """Valid typed facts pass numeric guard, render canonical claims and are literally cited."""
        query = "Según el cuadro 1, ¿qué porcentaje del PIB figura para Empleo en 2025?"
        attempt_spec = {
            "attempt_index": 4,
            "case_id": "case_supported_typed_facts",
            "query": query,
            "arm": "mixed",
        }

        gen_9b_resp = make_raw_chat_response(
            GENERATOR_9B_MODEL,
            {
                "status": "answered",
                "facts": [
                    {
                        "cell_id": "c_r1c1",
                        "concept": "Empleo",
                        "header_path": ["2025"],
                        "year": 2025,
                        "ordered_delta_years": None,
                        "unit": "% PIB",
                        "measure": "level",
                        "value": "10.2",
                        "doc_id": self.pages[0]["doc_id"],
                        "source": self.pages[0]["source_file"],
                        "date": "2026-01-15",
                        "page": 95,
                        "table_id": "table_synth_cuadro1",
                        "extraction_id": "synth_ext_01",
                        "evidence_ids": ["E1", "E2"],
                    }
                ],
                "reason": "",
            },
        )

        transport = PipelineFakeTransport(
            chat_responses=[
                make_raw_chat_response(GENERATOR_9B_MODEL, {"status": "selected", "ids": ["P1"], "reason": ""}),
                gen_9b_resp,
            ]
        )

        res = execute_table_attempt(
            attempt_spec,
            index_data=self.index,
            units_data=self.units,
            pages_data=self.pages,
            transport=transport,
            table_bundle=self.bundle,
            enabled=True,
        )

        self.assertEqual(res["status"], "completed", f"Error in supported: {res.get('error')}")
        self.assertEqual(res["factual_outcome"]["status"], "answered")

        audit = res["table_audit"]
        self.assertEqual(audit["gate_decision"], "accepted")
        self.assertFalse(audit["numeric_mismatch"])
        self.assertEqual(len(audit["rendering"]), 1)
        self.assertEqual(audit["rendering"][0]["value"], "10.2")

        pub_claims = res["factual_outcome"]["publication"]["claims"]
        self.assertEqual(len(pub_claims), 1)
        self.assertIn("10.2 % PIB", pub_claims[0]["text"])
        self.assertTrue(any(c["source_id"] == "S1" for c in pub_claims[0]["evidence"]))

    def test_t06_two_row_style_package(self) -> None:
        """Multi-row queries produce a coherent package with both row facts without truncation."""
        query = "Según el cuadro 1, ¿cuál es el porcentaje para Empleo en 2025 y para Salarios en 2030?"
        attempt_spec = {
            "attempt_index": 5,
            "case_id": "case_t06_multi_row",
            "query": query,
            "arm": "mixed",
        }

        transport = PipelineFakeTransport()

        res = execute_table_attempt(
            attempt_spec,
            index_data=self.index,
            units_data=self.units,
            pages_data=self.pages,
            transport=transport,
            table_bundle=self.bundle,
            enabled=True,
        )

        self.assertEqual(res["status"], "completed", f"Error in t06: {res.get('error')}")
        self.assertEqual(res["factual_outcome"]["status"], "answered")

        pub_claims = res["factual_outcome"]["publication"]["claims"]
        self.assertEqual(len(pub_claims), 2)
        self.assertIn("10.2 % PIB", pub_claims[0]["text"])
        self.assertIn("7.8 % PIB", pub_claims[1]["text"])
        self.assertLessEqual(len(pub_claims[0]["evidence"]), 3)
        self.assertLessEqual(len(pub_claims[1]["evidence"]), 3)

        p95_text = self.pages[0]["text"]
        for pc in pub_claims:
            for ev in pc["evidence"]:
                self.assertIn(ev["quote"], p95_text)

        # Verify sent prompt contains all fields needed by 9B without private test data
        gen_calls = [
            c for c in transport.chat_calls
            if c.get("model") == GENERATOR_9B_MODEL and any(
                "facts" in b.get("properties", {}) for b in c.get("format", {}).get("anyOf", [])
            )
        ]
        self.assertTrue(len(gen_calls) > 0)
        user_msg = next(m for m in gen_calls[0]["messages"] if m["role"] == "user")
        user_data = json.loads(user_msg["content"])
        self.assertIn("tabla_contexto", user_data)
        t_ctx = user_data["tabla_contexto"]
        for key in ["doc_id", "source", "date", "page", "table_id", "extraction_id", "celdas_disponibles", "evidence_mapping"]:
            self.assertIn(key, t_ctx)
        self.assertEqual(len(t_ctx["celdas_disponibles"]), 2)
        self.assertLessEqual(len(json.dumps(gen_calls[0]["messages"]).encode("utf-8")), 6500)

    def test_9b_wrong_concept_year_value_delta_unit_extra_prose_cannot_publish(self) -> None:
        """9B invalid facts are rejected by guard and cannot reach publication even if fake 27B verifier supports them."""
        query = "Según el cuadro 1, ¿qué porcentaje del PIB figura para Empleo en 2025?"
        attempt_spec = {
            "attempt_index": 6,
            "case_id": "case_9b_adversarial",
            "query": query,
            "arm": "mixed",
        }

        doc_id = self.pages[0]["doc_id"]
        source_file = self.pages[0]["source_file"]
        base_fact = {
            "cell_id": "c_r1c1",
            "concept": "Empleo",
            "header_path": ["2025"],
            "year": 2025,
            "ordered_delta_years": None,
            "unit": "% PIB",
            "measure": "level",
            "value": "10.2",
            "doc_id": doc_id,
            "source": source_file,
            "date": "2026-01-15",
            "page": 95,
            "table_id": "table_synth_cuadro1",
            "extraction_id": "synth_ext_01",
            "evidence_ids": ["E1", "E2"],
        }

        fact_wrong_concept = copy.deepcopy(base_fact)
        fact_wrong_concept["concept"] = "ConceptoInventado"

        fact_wrong_year = copy.deepcopy(base_fact)
        fact_wrong_year["year"] = 2099

        fact_wrong_value = copy.deepcopy(base_fact)
        fact_wrong_value["value"] = "99.9"

        fact_wrong_unit = copy.deepcopy(base_fact)
        fact_wrong_unit["unit"] = "euros"

        fact_extra_prose = copy.deepcopy(base_fact)
        fact_extra_prose["narrative_hallucination"] = "Texto libre adicional no autorizado"

        for bad_fact, label in [
            (fact_wrong_concept, "wrong_concept"),
            (fact_wrong_year, "wrong_year"),
            (fact_wrong_value, "wrong_value"),
            (fact_wrong_unit, "wrong_unit"),
            (fact_extra_prose, "extra_prose"),
        ]:
            transport = PipelineFakeTransport(
                chat_responses=[
                    make_raw_chat_response(GENERATOR_9B_MODEL, {"status": "selected", "ids": ["P1"], "reason": ""}),
                    make_raw_chat_response(GENERATOR_9B_MODEL, {"status": "answered", "facts": [bad_fact], "reason": ""}),
                ]
            )

            res = execute_table_attempt(
                attempt_spec,
                index_data=self.index,
                units_data=self.units,
                pages_data=self.pages,
                transport=transport,
                table_bundle=self.bundle,
                enabled=True,
            )

            self.assertEqual(res["status"], "completed", f"Failed status for {label}")
            self.assertIsNotNone(res.get("factual_outcome"), f"factual_outcome is None for {label}")
            self.assertEqual(
                res["factual_outcome"]["status"],
                "insufficient_evidence",
                f"Failed for adversarial fact: {label}",
            )
            audit = res["table_audit"]
            self.assertEqual(audit["gate_decision"], "rejected", f"Failed for {label}")
            self.assertTrue(audit["numeric_mismatch"], f"Failed for {label}")
            self.assertEqual(len(res["factual_outcome"]["publication"]["claims"]), 0)

    def test_missing_and_unsupported_explicit_requests_safely_refuse(self) -> None:
        """Unsupported explicit requests safely refuse without falling through to narrative text."""
        unsupported_queries = [
            ("Según el cuadro 99, ¿cuál es el gasto?", "unknown_table"),
            ("Según el cuadro 1, ¿cuánto me corresponde de pensión para mí?", "personal_restriction"),
            ("Según el cuadro 1, ¿cuál es el ratio entre Empleo y Salarios?", "unsupported_calculation"),
            ("Según el cuadro 1, ¿cuál es el gasto en Defensa en 2025?", "unknown_concept"),
        ]

        for query, label in unsupported_queries:
            attempt_spec = {
                "attempt_index": 7,
                "case_id": f"case_{label}",
                "query": query,
                "arm": "mixed",
            }
            transport = PipelineFakeTransport()

            res = execute_table_attempt(
                attempt_spec,
                index_data=self.index,
                units_data=self.units,
                pages_data=self.pages,
                transport=transport,
                table_bundle=self.bundle,
                enabled=True,
            )

            self.assertEqual(res["status"], "completed", f"Failed status for {label}")
            self.assertIsNotNone(res.get("factual_outcome"), f"factual_outcome is None for {label}")
            self.assertEqual(res["factual_outcome"]["status"], "insufficient_evidence", f"Failed for {label}")
            self.assertEqual(len(res["factual_outcome"]["publication"]["claims"]), 0)
            self.assertEqual(len(transport.chat_calls), 1)
            self.assertEqual(len(transport.embed_calls), 1)
            self.assertNotEqual(res["factual_outcome"]["full_result"]["prompt_sha256"], "0" * 64)
            self.assertGreater(res["factual_outcome"]["full_result"]["stats"]["prompt_eval_count"], 0)
            self.assertEqual(res["table_audit"]["gate_decision"], "refused")

    def test_malformed_json_technical_failure_truthfully_retained(self) -> None:
        """Malformed JSON from model is truthfully captured as technical error."""
        attempt_spec = {
            "attempt_index": 8,
            "case_id": "case_malformed_json",
            "query": "Según el cuadro 1, ¿qué porcentaje figura para Empleo en 2025?",
            "arm": "mixed",
        }

        malformed_resp = make_raw_chat_response(GENERATOR_9B_MODEL, "ESTO NO ES UN JSON VALIDO {")

        transport = PipelineFakeTransport(
            chat_responses=[
                make_raw_chat_response(GENERATOR_9B_MODEL, {"status": "selected", "ids": ["P1"], "reason": ""}),
                malformed_resp,
            ]
        )

        res = execute_table_attempt(
            attempt_spec,
            index_data=self.index,
            units_data=self.units,
            pages_data=self.pages,
            transport=transport,
            table_bundle=self.bundle,
            enabled=True,
        )

        self.assertEqual(res["status"], "error")
        self.assertIsNotNone(res["error"])
        self.assertIn("JSON", res["error"])

    def test_repair_cannot_evade_guard(self) -> None:
        """A repair attempt trying to introduce invalid facts is rejected by the guard."""
        query = "Según el cuadro 1, ¿qué porcentaje del PIB figura para Empleo en 2025?"
        attempt_spec = {
            "attempt_index": 9,
            "case_id": "case_repair_guard",
            "query": query,
            "arm": "mixed",
        }

        valid_fact = {
            "cell_id": "c_r1c1",
            "concept": "Empleo",
            "header_path": ["2025"],
            "year": 2025,
            "ordered_delta_years": None,
            "unit": "% PIB",
            "measure": "level",
            "value": "10.2",
            "doc_id": self.pages[0]["doc_id"],
            "source": self.pages[0]["source_file"],
            "date": "2026-01-15",
            "page": 95,
            "table_id": "table_synth_cuadro1",
            "extraction_id": "synth_ext_01",
            "evidence_ids": ["E1", "E2"],
        }

        ver_refuted = make_raw_chat_response(
            TEXT_27B_MODEL,
            {
                "verdicts": [
                    {
                        "claim_id": "C1",
                        "verdict": "contradicted",
                        "reason": "Rechazo inicial para provocar reparación.",
                        "evidence_verdicts": [
                            {"evidence_id": "E1", "relation": "contradicts"},
                            {"evidence_id": "E2", "relation": "contradicts"},
                        ],
                    }
                ]
            },
        )

        evasion_fact = copy.deepcopy(valid_fact)
        evasion_fact["value"] = "99.9"
        repair_resp = make_raw_chat_response(
            GENERATOR_9B_MODEL,
            {
                "repairs": [
                    {
                        "claim_id": "C1",
                        "status": "answered",
                        "facts": [evasion_fact],
                        "reason": "",
                    }
                ]
            },
        )

        transport = PipelineFakeTransport(
            chat_responses=[
                make_raw_chat_response(GENERATOR_9B_MODEL, {"status": "selected", "ids": ["P1"], "reason": ""}),
                make_raw_chat_response(GENERATOR_9B_MODEL, {"status": "answered", "facts": [valid_fact], "reason": ""}),
                ver_refuted,
                make_raw_chat_response(GENERATOR_9B_MODEL, {"status": "selected", "ids": ["P1"], "reason": ""}),
                repair_resp,
            ]
        )

        res = execute_table_attempt(
            attempt_spec,
            index_data=self.index,
            units_data=self.units,
            pages_data=self.pages,
            transport=transport,
            table_bundle=self.bundle,
            enabled=True,
        )

        self.assertEqual(res["status"], "completed", f"Error in repair: {res.get('error')}")
        self.assertIsNotNone(res.get("factual_outcome"))
        self.assertEqual(res["factual_outcome"]["status"], "insufficient_evidence")
        self.assertEqual(len(res["factual_outcome"]["publication"]["claims"]), 0)

    def test_budget_failure_safe(self) -> None:
        """Table package exceeding evidence references or UTF-8 bytes budget safely refuses."""
        query = "Según el cuadro 1, ¿cuál es el porcentaje para Empleo en 2025?"
        attempt_spec = {
            "attempt_index": 10,
            "case_id": "case_budget_citation",
            "query": query,
            "arm": "mixed",
        }

        import experiments.table_grounded as tg_mod
        orig_max = tg_mod.MAX_MESSAGES_BYTES
        try:
            tg_mod.MAX_MESSAGES_BYTES = 2000
            transport = PipelineFakeTransport()
            res = execute_table_attempt(
                attempt_spec,
                index_data=self.index,
                units_data=self.units,
                pages_data=self.pages,
                transport=transport,
                table_bundle=self.bundle,
                enabled=True,
            )
        finally:
            tg_mod.MAX_MESSAGES_BYTES = orig_max

        self.assertEqual(res["status"], "completed", f"Error in budget: {res.get('error')}")
        self.assertEqual(res["factual_outcome"]["status"], "insufficient_evidence")
        self.assertEqual(res["table_audit"]["gate_decision"], "refused")
        self.assertEqual(res["table_audit"]["gate_stage"], "context_budget")
        self.assertEqual(len(res["factual_outcome"]["publication"]["claims"]), 0)
        self.assertNotEqual(res["factual_outcome"]["full_result"]["prompt_sha256"], "0" * 64)
        self.assertGreater(res["factual_outcome"]["full_result"]["stats"]["prompt_eval_count"], 0)
        self.assertGreaterEqual(len(transport.chat_calls), 2)
        for call in transport.chat_calls:
            serialized_len = len(json.dumps(call["messages"], ensure_ascii=False).encode("utf-8"))
            self.assertLessEqual(serialized_len, 2000)

    def test_interleaved_requests_no_state_leakage(self) -> None:
        """State is fully reset between interleaved table and narrative queries."""
        attempt_table = {
            "attempt_index": 11,
            "case_id": "case_table_seq",
            "query": "Según el cuadro 1, ¿qué porcentaje figura para Empleo en 2025?",
            "arm": "mixed",
        }
        attempt_narrative = {
            "attempt_index": 12,
            "case_id": "case_narrative_seq",
            "query": "¿Qué evolución demográfica proyecta el informe?",
            "arm": "mixed",
        }

        transport = PipelineFakeTransport()

        res1 = execute_table_attempt(
            attempt_table,
            index_data=self.index,
            units_data=self.units,
            pages_data=self.pages,
            transport=transport,
            table_bundle=self.bundle,
            enabled=True,
        )
        self.assertTrue(res1["table_audit"]["is_table_query"])

        res2 = execute_table_attempt(
            attempt_narrative,
            index_data=self.index,
            units_data=self.units,
            pages_data=self.pages,
            transport=transport,
            table_bundle=self.bundle,
            enabled=True,
        )
        self.assertFalse(res2["table_audit"]["is_table_query"])
        self.assertEqual(res2["table_audit"]["gate_decision"], "bypassed")

        res3 = execute_table_attempt(
            attempt_table,
            index_data=self.index,
            units_data=self.units,
            pages_data=self.pages,
            transport=transport,
            table_bundle=self.bundle,
            enabled=True,
        )
        self.assertTrue(res3["table_audit"]["is_table_query"])

    def test_raw_model_payloads_retained_and_expected_roles_preserved(self) -> None:
        """All raw model payloads are retained and expected model roles are strictly preserved."""
        attempt_spec = {
            "attempt_index": 13,
            "case_id": "case_raw_retained",
            "query": "Según el cuadro 1, ¿qué porcentaje del PIB figura para Empleo en 2025?",
            "arm": "mixed",
        }

        gen_9b_resp = make_raw_chat_response(
            GENERATOR_9B_MODEL,
            {
                "status": "answered",
                "facts": [
                    {
                        "cell_id": "c_r1c1",
                        "concept": "Empleo",
                        "header_path": ["2025"],
                        "year": 2025,
                        "ordered_delta_years": None,
                        "unit": "% PIB",
                        "measure": "level",
                        "value": "10.2",
                        "doc_id": self.pages[0]["doc_id"],
                        "source": self.pages[0]["source_file"],
                        "date": "2026-01-15",
                        "page": 95,
                        "table_id": "table_synth_cuadro1",
                        "extraction_id": "synth_ext_01",
                        "evidence_ids": ["E1", "E2"],
                    }
                ],
                "reason": "",
            },
        )

        transport = PipelineFakeTransport(
            chat_responses=[
                make_raw_chat_response(GENERATOR_9B_MODEL, {"status": "selected", "ids": ["P1"], "reason": ""}),
                gen_9b_resp,
            ]
        )

        res = execute_table_attempt(
            attempt_spec,
            index_data=self.index,
            units_data=self.units,
            pages_data=self.pages,
            transport=transport,
            table_bundle=self.bundle,
            enabled=True,
        )

        self.assertEqual(res["text_model"], GENERATOR_9B_MODEL)
        self.assertEqual(res["embedding_model"], EMBEDDING_4B_MODEL)

        calls = res["calls"]
        self.assertGreaterEqual(len(calls), 3)

        models_used = [c.get("model") for c in calls if isinstance(c, dict)]
        self.assertIn(GENERATOR_9B_MODEL, models_used)
        self.assertIn(TEXT_27B_MODEL, models_used)

        for c in calls:
            self.assertIn("raw_response", c)
            self.assertIsNotNone(c["raw_response"])

        audit = res["table_audit"]
        self.assertEqual(len(audit["original_typed_facts"]), 1)
        self.assertEqual(audit["original_typed_facts"][0]["cell_id"], "c_r1c1")
        self.assertGreaterEqual(len(audit["role_traces"]), 2)

    def test_table_query_exceeding_capacity_safely_refuses(self) -> None:
        """> capacity (more than MAX_CLAIM_REFS = 3 citations) safely refuses with citation_budget."""
        query = "Según el cuadro 1, ¿cuál es el porcentaje para Empleo en 2025, Salarios en 2025 y Pensiones en 2025?"
        attempt_spec = {
            "attempt_index": 20,
            "case_id": "case_capacity_refusal",
            "query": query,
            "arm": "mixed",
        }

        cap_bundle = copy.deepcopy(self.bundle)
        tbl = cap_bundle["tables"][0]
        tbl["dimensions"]["rows"] = 4
        tbl["bbox"]["b"] = 380.0
        cap_bundle["pages"][0]["blocks"][1]["bbox"]["b"] = 380.0
        # Row 3: Pensiones
        tbl["cells"].extend([
            {
                "row_start": 3, "row_end": 4, "col_start": 0, "col_end": 1,
                "text": "Pensiones", "raw_text": "Pensiones", "is_empty": False,
                "page_number": 95, "bbox": {"l": 100.0, "t": 330.0, "r": 200.0, "b": 380.0, "coord_origin": "TOPLEFT"},
                "column_header": False, "row_header": True, "header_path": [], "row_label": "Pensiones",
                "cell_id": "c_r3c0",
            },
            {
                "row_start": 3, "row_end": 4, "col_start": 1, "col_end": 2,
                "text": "15,3", "raw_text": "15,3", "is_empty": False,
                "page_number": 95, "bbox": {"l": 200.0, "t": 330.0, "r": 300.0, "b": 380.0, "coord_origin": "TOPLEFT"},
                "column_header": False, "row_header": False, "header_path": ["2025"], "row_label": "Pensiones",
                "cell_id": "c_r3c1",
            },
        ])

        formatted_table = _format_table_as_text(tbl)
        preamble = "En el escenario macroeconómico y fiscal se evalúan las siguientes magnitudes detalladas.\n\n"
        epilogue = "\n\nEstimaciones sujetas a la evolución demográfica del periodo proyectado."
        page_text = preamble + formatted_table + epilogue

        cap_page = copy.deepcopy(self.pages[0])
        cap_page["text"] = page_text
        cap_page["end_char"] = len(page_text)

        cap_bundle["pages"][0]["blocks"][1]["text"] = formatted_table

        cap_units = {
            "schema_version": "reconstructed-units-catalog-v1",
            "units": [
                {
                    "id": "U1",
                    "spans": [
                        {
                            "doc_id": cap_page["doc_id"],
                            "source_file": cap_page["source_file"],
                            "published_at": cap_page["published_at"],
                            "page_number": 95,
                            "start_char": 0,
                            "end_char": len(page_text),
                            "quote": collapse_spaces(page_text),
                        }
                    ],
                }
            ],
        }

        norm_val = 1.0 / math.sqrt(DIMENSION)
        cap_index = {
            "schema_version": SCHEMA,
            "model": EMBEDDING_4B_MODEL,
            "model_digest": PINNED_EMBEDDING_DIGEST,
            "dimension": DIMENSION,
            "document_prefix": "",
            "query_prefix": QUERY_PREFIX,
            "base_index_hash": "0" * 64,
            "chunks": [
                {
                    "chunk_id": "chunk_page_95_cap",
                    "doc_id": cap_page["doc_id"],
                    "source_file": cap_page["source_file"],
                    "published_at": cap_page["published_at"],
                    "page_number": 95,
                    "start_char": 0,
                    "end_char": len(page_text),
                    "text": page_text,
                    "strategy": "layout_block",
                    "embedding": [norm_val] * DIMENSION,
                }
            ],
        }

        transport = PipelineFakeTransport()
        res = execute_table_attempt(
            attempt_spec,
            index_data=cap_index,
            units_data=cap_units,
            pages_data=[cap_page],
            transport=transport,
            table_bundle=cap_bundle,
            enabled=True,
        )

        self.assertEqual(res["status"], "completed", f"Error in capacity: {res.get('error')}")
        self.assertEqual(res["factual_outcome"]["status"], "insufficient_evidence")
        self.assertEqual(res["table_audit"]["gate_decision"], "refused")
        self.assertEqual(res["table_audit"]["gate_stage"], "citation_budget")
        self.assertEqual(len(res["factual_outcome"]["publication"]["claims"]), 0)

    def test_unselected_rowrefs_rejected_even_if_same_table(self) -> None:
        """Unselected/unexpanded row references are rejected by the guard even if from the same table."""
        query = "Según el cuadro 1, ¿qué porcentaje del PIB figura para Empleo en 2025?"
        attempt_spec = {
            "attempt_index": 21,
            "case_id": "case_unselected_row",
            "query": query,
            "arm": "mixed",
        }
        unselected_fact = {
            "cell_id": "c_r2c2",
            "concept": "Salarios",
            "header_path": ["2030"],
            "year": 2030,
            "ordered_delta_years": None,
            "unit": "% PIB",
            "measure": "level",
            "value": "7.8",
            "doc_id": self.pages[0]["doc_id"],
            "source": self.pages[0]["source_file"],
            "date": "2026-01-15",
            "page": 95,
            "table_id": "table_synth_cuadro1",
            "extraction_id": "synth_ext_01",
            "evidence_ids": [
                "table_synth_cuadro1_caption",
                "table_synth_cuadro1_header",
                "table_synth_cuadro1_r2_row",
            ],
        }
        transport = PipelineFakeTransport(
            chat_responses=[
                make_raw_chat_response(GENERATOR_9B_MODEL, {"status": "selected", "ids": ["P1"], "reason": ""}),
                make_raw_chat_response(GENERATOR_9B_MODEL, {"status": "answered", "facts": [unselected_fact], "reason": ""}),
            ]
        )
        res = execute_table_attempt(
            attempt_spec,
            index_data=self.index,
            units_data=self.units,
            pages_data=self.pages,
            transport=transport,
            table_bundle=self.bundle,
            enabled=True,
        )
        self.assertEqual(res["status"], "completed")
        self.assertEqual(res["factual_outcome"]["status"], "insufficient_evidence")
        self.assertEqual(res["table_audit"]["gate_decision"], "rejected")
        self.assertEqual(res["table_audit"]["gate_stage"], "numeric_guard")

    def test_actual_pilot_smoke_with_fake_clients(self) -> None:
        """Smoke test on actual pilot v4 structured catalog and layout pages with fake clients (both 8.0 and 3.6 values)."""
        pilot_bundle_path = REPO_ROOT / "evaluation" / "document_layout_2026-10-27" / "pilot_v4" / "structured.json"
        pilot_pages_path = REPO_ROOT / "evaluation" / "layout_answer_effect_2026-10-27" / "prepared_v1" / "layout_pages.jsonl"
        if not pilot_bundle_path.exists() or not pilot_pages_path.exists():
            self.skipTest("Pilot artifacts not present")

        bundle = json.loads(pilot_bundle_path.read_text(encoding="utf-8"))
        pages = [json.loads(line) for line in pilot_pages_path.read_text(encoding="utf-8").splitlines() if line.strip()]

        query = "Según el cuadro 9, ¿cuál es el porcentaje para Sanidad en 2050 y para Educación en 2070?"
        attempt_spec = {
            "attempt_index": 22,
            "case_id": "case_pilot_smoke",
            "query": query,
            "arm": "mixed",
        }

        p95 = next(p for p in pages if p["page_number"] == 95)
        p95_units = {
            "schema_version": "reconstructed-units-catalog-v1",
            "units": [
                {
                    "id": "U95",
                    "spans": [
                        {
                            "doc_id": p95["doc_id"],
                            "source_file": p95["source_file"],
                            "published_at": p95["published_at"],
                            "page_number": 95,
                            "start_char": 0,
                            "end_char": len(p95["text"]),
                            "quote": collapse_spaces(p95["text"]),
                        }
                    ],
                }
            ],
        }
        norm_val = 1.0 / math.sqrt(DIMENSION)
        p95_index = {
            "schema_version": SCHEMA,
            "model": EMBEDDING_4B_MODEL,
            "model_digest": PINNED_EMBEDDING_DIGEST,
            "dimension": DIMENSION,
            "document_prefix": "",
            "query_prefix": QUERY_PREFIX,
            "base_index_hash": "0" * 64,
            "chunks": [
                {
                    "chunk_id": "chunk_pilot_p95",
                    "doc_id": p95["doc_id"],
                    "source_file": p95["source_file"],
                    "published_at": p95["published_at"],
                    "page_number": 95,
                    "start_char": 0,
                    "end_char": len(p95["text"]),
                    "text": p95["text"],
                    "strategy": "layout_block",
                    "embedding": [norm_val] * DIMENSION,
                }
            ],
        }

        transport = PipelineFakeTransport()
        res = execute_table_attempt(
            attempt_spec,
            index_data=p95_index,
            units_data=p95_units,
            pages_data=pages,
            transport=transport,
            table_bundle=bundle,
            enabled=True,
        )

        self.assertEqual(res["status"], "completed", f"Error in pilot smoke: {res.get('error')}")
        self.assertEqual(res["factual_outcome"]["status"], "answered")
        pub_claims = res["factual_outcome"]["publication"]["claims"]
        self.assertEqual(len(pub_claims), 2)
        has_8_0 = any("8,0" in c["text"].replace(".", ",") or "8.0" in c["text"] for c in pub_claims)
        has_3_6 = any("3,6" in c["text"].replace(".", ",") or "3.6" in c["text"] for c in pub_claims)
        self.assertTrue(has_8_0, f"Expected 8.0/8,0 in pub_claims: {pub_claims}")
        self.assertTrue(has_3_6, f"Expected 3.6/3,6 in pub_claims: {pub_claims}")
        for pc in pub_claims:
            self.assertLessEqual(len(pc["evidence"]), 3)

    def test_post_partialgeneration_overflow_safely_refuses(self) -> None:
        """Post-partialgeneration overflow safely refuses with real stats and not zero SHA."""
        query = "Según el cuadro 1, ¿qué porcentaje del PIB figura para Empleo en 2025?"
        attempt_spec = {
            "attempt_index": 30,
            "case_id": "case_overflow_refusal",
            "query": query,
            "arm": "mixed",
        }
        transport = PipelineFakeTransport()

        import experiments.table_grounded as tg_mod
        orig_max = tg_mod.MAX_MESSAGES_BYTES
        try:
            tg_mod.MAX_MESSAGES_BYTES = 200  # inject tight context budget to trigger generation overflow
            res = execute_table_attempt(
                attempt_spec,
                index_data=self.index,
                units_data=self.units,
                pages_data=self.pages,
                transport=transport,
                table_bundle=self.bundle,
                enabled=True,
            )
        finally:
            tg_mod.MAX_MESSAGES_BYTES = orig_max

        self.assertEqual(res["status"], "completed", f"Error in overflow: {res.get('error')}")
        self.assertEqual(res["factual_outcome"]["status"], "insufficient_evidence")
        self.assertEqual(res["table_audit"]["gate_decision"], "refused")
        self.assertEqual(res["table_audit"]["gate_stage"], "context_budget")
        self.assertNotEqual(res["factual_outcome"]["full_result"]["prompt_sha256"], "0" * 64)
        self.assertGreater(res["factual_outcome"]["full_result"]["stats"]["prompt_eval_count"], 0)
        self.assertEqual(len(res["factual_outcome"]["publication"]["claims"]), 0)
        self.assertGreaterEqual(len(transport.chat_calls), 2)

    def test_one_claim_correct_repair_in_multiclaim_query(self) -> None:
        """One claim repaired correctly in a multi-claim query publishes both claims."""
        query = "Según el cuadro 1, ¿cuál es el porcentaje para Empleo en 2025 y para Salarios en 2030?"
        attempt_spec = {
            "attempt_index": 31,
            "case_id": "case_oneclaim_repair",
            "query": query,
            "arm": "mixed",
        }

        # Initial 9B generation returns 2 claims (Empleo 10.2, Salarios 7.8)
        gen_9b_resp = make_raw_chat_response(
            GENERATOR_9B_MODEL,
            {
                "status": "answered",
                "facts": [
                    {
                        "cell_id": "c_r1c1",
                        "concept": "Empleo",
                        "header_path": ["2025"],
                        "year": 2025,
                        "ordered_delta_years": None,
                        "unit": "% PIB",
                        "measure": "level",
                        "value": "10.2",
                        "doc_id": self.pages[0]["doc_id"],
                        "source": self.pages[0]["source_file"],
                        "date": "2026-01-15",
                        "page": 95,
                        "table_id": "table_synth_cuadro1",
                        "extraction_id": "synth_ext_01",
                        "evidence_ids": ["E1", "E2"],
                    },
                    {
                        "cell_id": "c_r2c2",
                        "concept": "Salarios",
                        "header_path": ["2030"],
                        "year": 2030,
                        "ordered_delta_years": None,
                        "unit": "% PIB",
                        "measure": "level",
                        "value": "7.8",
                        "doc_id": self.pages[0]["doc_id"],
                        "source": self.pages[0]["source_file"],
                        "date": "2026-01-15",
                        "page": 95,
                        "table_id": "table_synth_cuadro1",
                        "extraction_id": "synth_ext_01",
                        "evidence_ids": ["E1", "E3"],
                    },
                ],
                "reason": "",
            },
        )

        # Initial verification: C1 is supported, C2 is contradicted
        def ver_resp(req: dict[str, Any]) -> dict[str, Any]:
            user_msg = next((m for m in req.get("messages", []) if m.get("role") == "user"), None)
            user_data = json.loads(user_msg["content"]) if user_msg and isinstance(user_msg.get("content"), str) else {}
            verdicts = []
            for c in user_data.get("afirmaciones", []):
                cid = c.get("claim_id", "C1")
                eids = c.get("evidence_ids", [])
                if cid == "C2":
                    verdicts.append({
                        "claim_id": cid,
                        "verdict": "contradicted",
                        "reason": "Rechazo inicial C2 para provocar reparacion.",
                        "evidence_verdicts": [
                            {"evidence_id": eid, "relation": "contradicts" if i == len(eids) - 1 else "supports"}
                            for i, eid in enumerate(eids)
                        ],
                    })
                else:
                    verdicts.append({
                        "claim_id": cid,
                        "verdict": "supported",
                        "reason": f"Soporte confirmado para {cid}.",
                        "evidence_verdicts": [{"evidence_id": eid, "relation": "supports"} for eid in eids],
                    })
            return make_raw_chat_response(TEXT_27B_MODEL, {"verdicts": verdicts})

        # Repair generation: 9B returns repaired fact for C2
        repair_resp = make_raw_chat_response(
            GENERATOR_9B_MODEL,
            {
                "repairs": [
                    {
                        "claim_id": "C2",
                        "status": "answered",
                        "facts": [
                            {
                                "cell_id": "c_r2c2",
                                "concept": "Salarios",
                                "header_path": ["2030"],
                                "year": 2030,
                                "ordered_delta_years": None,
                                "unit": "% PIB",
                                "measure": "level",
                                "value": "7.8",
                                "doc_id": self.pages[0]["doc_id"],
                                "source": self.pages[0]["source_file"],
                                "date": "2026-01-15",
                                "page": 95,
                                "table_id": "table_synth_cuadro1",
                                "extraction_id": "synth_ext_01",
                                "evidence_ids": ["E1", "E3"],
                            }
                        ],
                        "reason": "",
                    }
                ]
            },
        )

        # Re-verification: C2 is supported
        def rever_resp(req: dict[str, Any]) -> dict[str, Any]:
            user_msg = next((m for m in req.get("messages", []) if m.get("role") == "user"), None)
            user_data = json.loads(user_msg["content"]) if user_msg and isinstance(user_msg.get("content"), str) else {}
            verdicts = []
            for c in user_data.get("afirmaciones", []):
                cid = c.get("claim_id", "C2")
                eids = c.get("evidence_ids", [])
                verdicts.append({
                    "claim_id": cid,
                    "verdict": "supported",
                    "reason": "C2 reparada y respaldada.",
                    "evidence_verdicts": [{"evidence_id": eid, "relation": "supports"} for eid in eids],
                })
            return make_raw_chat_response(TEXT_27B_MODEL, {"verdicts": verdicts})

        transport = PipelineFakeTransport(
            chat_responses=[
                make_raw_chat_response(GENERATOR_9B_MODEL, {"status": "selected", "ids": ["P1"], "reason": ""}),
                gen_9b_resp,
                ver_resp,
                make_raw_chat_response(GENERATOR_9B_MODEL, {"status": "selected", "ids": ["P1"], "reason": ""}),
                repair_resp,
                rever_resp,
            ]
        )

        res = execute_table_attempt(
            attempt_spec,
            index_data=self.index,
            units_data=self.units,
            pages_data=self.pages,
            transport=transport,
            table_bundle=self.bundle,
            enabled=True,
        )

        self.assertEqual(res["status"], "completed", f"Error in multiclaim repair: {res.get('error')}")
        self.assertEqual(res["factual_outcome"]["status"], "answered")
        pub_claims = res["factual_outcome"]["publication"]["claims"]
        self.assertEqual(len(pub_claims), 2)
        has_empleo = any("10.2" in c["text"] for c in pub_claims)
        has_salarios = any("7.8" in c["text"] for c in pub_claims)
        self.assertTrue(has_empleo)
        self.assertTrue(has_salarios)

    def test_repair_wrong_numeric_and_extra_prose_fails_safely(self) -> None:
        """Wrong numeric value or extra prose in repair fails safely without publication."""
        query = "Según el cuadro 1, ¿qué porcentaje del PIB figura para Empleo en 2025?"
        attempt_spec = {
            "attempt_index": 32,
            "case_id": "case_bad_repair",
            "query": query,
            "arm": "mixed",
        }

        ver_contradicted = make_raw_chat_response(
            TEXT_27B_MODEL,
            {
                "verdicts": [
                    {
                        "claim_id": "C1",
                        "verdict": "contradicted",
                        "reason": "Rechazo para obligar reparacion.",
                        "evidence_verdicts": [
                            {"evidence_id": "E1", "relation": "contradicts"},
                            {"evidence_id": "E2", "relation": "contradicts"},
                        ],
                    }
                ]
            },
        )

        # Case A: Wrong numeric in repair (e.g. 99.9 instead of 10.2)
        repair_wrong_val = make_raw_chat_response(
            GENERATOR_9B_MODEL,
            {
                "repairs": [
                    {
                        "claim_id": "C1",
                        "status": "answered",
                        "facts": [
                            {
                                "cell_id": "c_r1c1",
                                "concept": "Empleo",
                                "header_path": ["2025"],
                                "year": 2025,
                                "ordered_delta_years": None,
                                "unit": "% PIB",
                                "measure": "level",
                                "value": "99.9",
                                "doc_id": self.pages[0]["doc_id"],
                                "source": self.pages[0]["source_file"],
                                "date": "2026-01-15",
                                "page": 95,
                                "table_id": "table_synth_cuadro1",
                                "extraction_id": "synth_ext_01",
                                "evidence_ids": ["E1", "E2"],
                            }
                        ],
                        "reason": "",
                    }
                ]
            },
        )

        transport_a = PipelineFakeTransport(
            chat_responses=[
                make_raw_chat_response(GENERATOR_9B_MODEL, {"status": "selected", "ids": ["P1"], "reason": ""}),
                make_raw_chat_response(
                    GENERATOR_9B_MODEL,
                    {
                        "status": "answered",
                        "facts": [
                            {
                                "cell_id": "c_r1c1",
                                "concept": "Empleo",
                                "header_path": ["2025"],
                                "year": 2025,
                                "ordered_delta_years": None,
                                "unit": "% PIB",
                                "measure": "level",
                                "value": "10.2",
                                "doc_id": self.pages[0]["doc_id"],
                                "source": self.pages[0]["source_file"],
                                "date": "2026-01-15",
                                "page": 95,
                                "table_id": "table_synth_cuadro1",
                                "extraction_id": "synth_ext_01",
                                "evidence_ids": ["E1", "E2"],
                            }
                        ],
                        "reason": "",
                    },
                ),
                ver_contradicted,
                make_raw_chat_response(GENERATOR_9B_MODEL, {"status": "selected", "ids": ["P1"], "reason": ""}),
                repair_wrong_val,
            ]
        )

        res_a = execute_table_attempt(
            attempt_spec,
            index_data=self.index,
            units_data=self.units,
            pages_data=self.pages,
            transport=transport_a,
            table_bundle=self.bundle,
            enabled=True,
        )

        self.assertEqual(res_a["status"], "completed")
        self.assertEqual(res_a["factual_outcome"]["status"], "insufficient_evidence")
        self.assertEqual(len(res_a["factual_outcome"]["publication"]["claims"]), 0)

        # Case B: Extra prose / ungrounded repair
        repair_extra_prose = make_raw_chat_response(
            GENERATOR_9B_MODEL,
            {
                "repairs": [
                    {
                        "claim_id": "C1",
                        "status": "answered",
                        "prose_hallucination": "Texto libre no tipado",
                        "reason": "",
                    }
                ]
            },
        )

        transport_b = PipelineFakeTransport(
            chat_responses=[
                make_raw_chat_response(GENERATOR_9B_MODEL, {"status": "selected", "ids": ["P1"], "reason": ""}),
                make_raw_chat_response(
                    GENERATOR_9B_MODEL,
                    {
                        "status": "answered",
                        "facts": [
                            {
                                "cell_id": "c_r1c1",
                                "concept": "Empleo",
                                "header_path": ["2025"],
                                "year": 2025,
                                "ordered_delta_years": None,
                                "unit": "% PIB",
                                "measure": "level",
                                "value": "10.2",
                                "doc_id": self.pages[0]["doc_id"],
                                "source": self.pages[0]["source_file"],
                                "date": "2026-01-15",
                                "page": 95,
                                "table_id": "table_synth_cuadro1",
                                "extraction_id": "synth_ext_01",
                                "evidence_ids": ["E1", "E2"],
                            }
                        ],
                        "reason": "",
                    },
                ),
                ver_contradicted,
                make_raw_chat_response(GENERATOR_9B_MODEL, {"status": "selected", "ids": ["P1"], "reason": ""}),
                repair_extra_prose,
            ]
        )

        res_b = execute_table_attempt(
            attempt_spec,
            index_data=self.index,
            units_data=self.units,
            pages_data=self.pages,
            transport=transport_b,
            table_bundle=self.bundle,
            enabled=True,
        )

        self.assertEqual(res_b["status"], "completed")
        self.assertEqual(res_b["factual_outcome"]["status"], "insufficient_evidence")
        self.assertEqual(len(res_b["factual_outcome"]["publication"]["claims"]), 0)

    def test_visible_eid_duplicates_unknown_and_internal_ids_rejected(self) -> None:
        """Visible EID duplicates, unknown EIDs, and arbitrary internal IDs are strictly rejected."""
        query = "Según el cuadro 1, ¿qué porcentaje del PIB figura para Empleo en 2025?"
        attempt_spec = {
            "attempt_index": 33,
            "case_id": "case_eid_rejections",
            "query": query,
            "arm": "mixed",
        }

        base_fact = {
            "cell_id": "c_r1c1",
            "concept": "Empleo",
            "header_path": ["2025"],
            "year": 2025,
            "ordered_delta_years": None,
            "unit": "% PIB",
            "measure": "level",
            "value": "10.2",
            "doc_id": self.pages[0]["doc_id"],
            "source": self.pages[0]["source_file"],
            "date": "2026-01-15",
            "page": 95,
            "table_id": "table_synth_cuadro1",
            "extraction_id": "synth_ext_01",
        }

        # Subcase 1: Visible EID duplicates in model response
        fact_dup_eid = copy.deepcopy(base_fact)
        fact_dup_eid["evidence_ids"] = ["E1", "E1", "E2"]

        # Subcase 2: Unknown visible EID
        fact_unknown_eid = copy.deepcopy(base_fact)
        fact_unknown_eid["evidence_ids"] = ["E1", "E99"]

        # Subcase 3: Raw internal fragment ID directly emitted by model
        fact_internal_id = copy.deepcopy(base_fact)
        fact_internal_id["evidence_ids"] = ["table_synth_cuadro1_caption"]

        for bad_fact, label in [
            (fact_dup_eid, "duplicate_eids"),
            (fact_unknown_eid, "unknown_eid"),
            (fact_internal_id, "internal_fragment_id"),
        ]:
            transport = PipelineFakeTransport(
                chat_responses=[
                    make_raw_chat_response(GENERATOR_9B_MODEL, {"status": "selected", "ids": ["P1"], "reason": ""}),
                    make_raw_chat_response(GENERATOR_9B_MODEL, {"status": "answered", "facts": [bad_fact], "reason": ""}),
                ]
            )

            res = execute_table_attempt(
                attempt_spec,
                index_data=self.index,
                units_data=self.units,
                pages_data=self.pages,
                transport=transport,
                table_bundle=self.bundle,
                enabled=True,
            )

            self.assertEqual(res["status"], "completed", f"Failed for {label}")
            self.assertEqual(res["factual_outcome"]["status"], "insufficient_evidence", f"Failed for {label}")
            self.assertEqual(res["table_audit"]["gate_decision"], "rejected", f"Failed for {label}")
            self.assertTrue(res["table_audit"]["numeric_mismatch"], f"Failed for {label}")
            self.assertEqual(len(res["factual_outcome"]["publication"]["claims"]), 0, f"Failed for {label}")

    def test_strict_enabled_validation_and_audit_preservation(self) -> None:
        """Strict boolean validation for enabled; audit preserves raw model facts and mapped guard facts."""
        attempt_spec = {
            "attempt_index": 34,
            "case_id": "case_strict_enabled",
            "query": "Según el cuadro 1, ¿qué porcentaje del PIB figura para Empleo en 2025?",
            "arm": "mixed",
        }
        transport = PipelineFakeTransport()

        for invalid_enabled in ["false", "true", 0, 1, None, [], {}]:
            with self.assertRaises(ValueError, msg=f"Should raise for enabled={invalid_enabled!r}"):
                execute_table_attempt(
                    attempt_spec,
                    index_data=self.index,
                    units_data=self.units,
                    pages_data=self.pages,
                    transport=transport,
                    table_bundle=self.bundle,
                    enabled=invalid_enabled,  # type: ignore[arg-type]
                )

        # Positive execution: verify exactly one query embedding and audit preservation
        transport_pos = PipelineFakeTransport()
        res = execute_table_attempt(
            attempt_spec,
            index_data=self.index,
            units_data=self.units,
            pages_data=self.pages,
            transport=transport_pos,
            table_bundle=self.bundle,
            enabled=True,
        )

        self.assertEqual(res["status"], "completed")
        self.assertEqual(len(transport_pos.embed_calls), 1)

        audit = res["table_audit"]
        self.assertIn("original_typed_facts", audit)
        self.assertIn("mapped_guard_facts", audit)
        self.assertEqual(len(audit["original_typed_facts"]), 1)
        self.assertEqual(len(audit["mapped_guard_facts"]), 1)

        raw_eids = audit["original_typed_facts"][0]["evidence_ids"]
        mapped_fids = audit["mapped_guard_facts"][0]["evidence_ids"]
        self.assertEqual(raw_eids, ["E1", "E2"])
        self.assertIn("table_synth_cuadro1_caption", mapped_fids)
        self.assertIn("table_synth_cuadro1_r1_row", mapped_fids)

    def test_table_fact_schema_requires_temporal_fields(self) -> None:
        """TABLE_FACT_SCHEMA requires both year and ordered_delta_years across all variants."""
        self.assertIn("year", TABLE_FACT_SCHEMA["required"])
        self.assertIn("ordered_delta_years", TABLE_FACT_SCHEMA["required"])
        self.assertEqual(TABLE_FACT_SCHEMA["properties"]["year"]["type"], ["integer", "null"])
        self.assertEqual(TABLE_FACT_SCHEMA["properties"]["ordered_delta_years"]["type"], ["array", "null"])

        # Check generation schema branches
        gen_branches = TABLE_GENERATION_SCHEMA.get("anyOf", [])
        facts_branch = next(
            (b for b in gen_branches if "facts" in b.get("properties", {}) and b["properties"]["facts"].get("minItems", 0) >= 1),
            None,
        )
        self.assertIsNotNone(facts_branch)
        self.assertIn("year", facts_branch["properties"]["facts"]["items"]["required"])
        self.assertIn("ordered_delta_years", facts_branch["properties"]["facts"]["items"]["required"])

        claims_branch = next(
            (b for b in gen_branches if "claims" in b.get("properties", {}) and b["properties"]["claims"].get("minItems", 0) >= 1),
            None,
        )
        self.assertIsNotNone(claims_branch)
        self.assertIn("year", claims_branch["properties"]["claims"]["items"]["required"])
        self.assertIn("ordered_delta_years", claims_branch["properties"]["claims"]["items"]["required"])

        # Check repair schema branches
        rep_items = TABLE_REPAIR_SCHEMA["properties"]["repairs"]["items"]["anyOf"]
        rep_facts_branch = next((b for b in rep_items if "facts" in b.get("properties", {})), None)
        self.assertIsNotNone(rep_facts_branch)
        self.assertIn("year", rep_facts_branch["properties"]["facts"]["items"]["required"])
        self.assertIn("ordered_delta_years", rep_facts_branch["properties"]["facts"]["items"]["required"])

        rep_fact_branch = next((b for b in rep_items if "fact" in b.get("properties", {})), None)
        self.assertIsNotNone(rep_fact_branch)
        self.assertIn("year", rep_fact_branch["properties"]["fact"]["required"])
        self.assertIn("ordered_delta_years", rep_fact_branch["properties"]["fact"]["required"])

    def test_regression_fake_nn_emits_only_schema_required_fields_level_and_delta(self) -> None:
        """Meaningful regression: fake NN that emits ONLY schema-required fact fields from sent context.

        Emulates observed valid grammar optional omission. Must no longer drop needed temporal fields:
        - Level: year is required integer, ordered_delta_years is required null.
        - Delta: year is required null, ordered_delta_years is required list[int].
        Both successfully pass guard and publish canonical grounded claims.
        """
        # Case A: Level measure (Empleo 2025 -> 10.2 % PIB)
        attempt_spec_level = {
            "attempt_index": 40,
            "case_id": "case_regression_level",
            "query": "Según el cuadro 1, ¿qué porcentaje del PIB figura para Empleo en 2025?",
            "arm": "mixed",
        }
        transport_level = PipelineFakeTransport()
        res_level = execute_table_attempt(
            attempt_spec_level,
            index_data=self.index,
            units_data=self.units,
            pages_data=self.pages,
            transport=transport_level,
            table_bundle=self.bundle,
            enabled=True,
        )

        self.assertEqual(res_level["status"], "completed", f"Error in level: {res_level.get('error')}")
        self.assertEqual(res_level["factual_outcome"]["status"], "answered")
        audit_level = res_level["table_audit"]
        self.assertEqual(audit_level["gate_decision"], "accepted")
        self.assertFalse(audit_level["numeric_mismatch"])
        self.assertEqual(len(audit_level["original_typed_facts"]), 1)
        orig_fact = audit_level["original_typed_facts"][0]
        self.assertEqual(orig_fact["year"], 2025)
        self.assertIsNone(orig_fact["ordered_delta_years"])
        self.assertEqual(orig_fact["value"], "10.2")

        pub_claims_level = res_level["factual_outcome"]["publication"]["claims"]
        self.assertEqual(len(pub_claims_level), 1)
        self.assertIn("10.2 % PIB", pub_claims_level[0]["text"])

        # Case B: Published Delta measure (Empleo Δ 2030-2025 -> 0.3 puntos PIB)
        attempt_spec_delta = {
            "attempt_index": 41,
            "case_id": "case_regression_delta",
            "query": "Según el cuadro 1, ¿cuál es la variación para Empleo entre 2030 y 2025?",
            "arm": "mixed",
        }
        transport_delta = PipelineFakeTransport()
        res_delta = execute_table_attempt(
            attempt_spec_delta,
            index_data=self.index,
            units_data=self.units,
            pages_data=self.pages,
            transport=transport_delta,
            table_bundle=self.bundle,
            enabled=True,
        )

        self.assertEqual(res_delta["status"], "completed", f"Error in delta: {res_delta.get('error')}")
        self.assertEqual(res_delta["factual_outcome"]["status"], "answered")
        audit_delta = res_delta["table_audit"]
        self.assertEqual(audit_delta["gate_decision"], "accepted")
        self.assertFalse(audit_delta["numeric_mismatch"])
        self.assertEqual(len(audit_delta["original_typed_facts"]), 1)
        orig_fact_delta = audit_delta["original_typed_facts"][0]
        self.assertIsNone(orig_fact_delta["year"])
        self.assertEqual(orig_fact_delta["ordered_delta_years"], [2030, 2025])
        self.assertEqual(orig_fact_delta["value"], "0.3")

        pub_claims_delta = res_delta["factual_outcome"]["publication"]["claims"]
        self.assertEqual(len(pub_claims_delta), 1)
        self.assertIn("0.3 puntos PIB", pub_claims_delta[0]["text"])

        # Case C: Generation variant emitting 'claims' key instead of 'facts'
        def claims_variant_handler(req: dict[str, Any]) -> dict[str, Any]:
            fmt = req.get("format", {})
            user_msg = next((m for m in req.get("messages", []) if m.get("role") == "user"), None)
            user_data = json.loads(user_msg["content"]) if user_msg else {}
            t_ctx = user_data.get("tabla_contexto", {})
            req_keys = _extract_required_fact_keys(fmt)
            claims = []
            for cell in t_ctx.get("celdas_disponibles", []):
                cand = {
                    "cell_id": cell["cell_id"],
                    "concept": cell["concept"],
                    "header_path": cell["header_path"],
                    "year": cell.get("year"),
                    "ordered_delta_years": cell.get("ordered_delta_years"),
                    "unit": cell["unit"],
                    "measure": cell["measure"],
                    "value": cell["value"],
                    "doc_id": t_ctx["doc_id"],
                    "source": t_ctx["source"],
                    "date": t_ctx["date"],
                    "page": t_ctx["page"],
                    "table_id": t_ctx["table_id"],
                    "extraction_id": t_ctx["extraction_id"],
                    "evidence_ids": list(cell.get("evidence_ids", [])),
                }
                claim_fact = {k: v for k, v in cand.items() if k in req_keys} if req_keys else cand
                claims.append(claim_fact)
            return make_raw_chat_response(
                GENERATOR_9B_MODEL,
                {"status": "answered", "claims": claims, "reason": ""},
            )

        transport_claims = PipelineFakeTransport(
            chat_responses=[
                make_raw_chat_response(GENERATOR_9B_MODEL, {"status": "selected", "ids": ["P1"], "reason": ""}),
                claims_variant_handler,
            ]
        )
        res_claims = execute_table_attempt(
            attempt_spec_level,
            index_data=self.index,
            units_data=self.units,
            pages_data=self.pages,
            transport=transport_claims,
            table_bundle=self.bundle,
            enabled=True,
        )
        self.assertEqual(res_claims["status"], "completed")
        self.assertEqual(res_claims["factual_outcome"]["status"], "answered")
        self.assertEqual(res_claims["table_audit"]["gate_decision"], "accepted")

    def test_regression_fake_nn_repair_schema_coverage(self) -> None:
        """Meaningful regression: fake NN that emits ONLY schema-required fact fields during repair.

        Covers both 'facts' and 'fact' repair schema branches using actual sent context.
        """
        attempt_spec = {
            "attempt_index": 42,
            "case_id": "case_regression_repair_coverage",
            "query": "Según el cuadro 1, ¿qué porcentaje del PIB figura para Empleo en 2025?",
            "arm": "mixed",
        }

        def ver_contradicted(req: dict[str, Any]) -> dict[str, Any]:
            user_msg = next((m for m in req.get("messages", []) if m.get("role") == "user"), None)
            user_data = json.loads(user_msg["content"]) if user_msg and isinstance(user_msg.get("content"), str) else {}
            verdicts = []
            for c in user_data.get("afirmaciones", []):
                cid = c.get("claim_id", "C1")
                eids = c.get("evidence_ids", [])
                verdicts.append({
                    "claim_id": cid,
                    "verdict": "contradicted",
                    "reason": "Rechazo inicial para obligar a ejecucion de reparacion.",
                    "evidence_verdicts": [{"evidence_id": eid, "relation": "contradicts"} for eid in eids],
                })
            return make_raw_chat_response(TEXT_27B_MODEL, {"verdicts": verdicts})

        def ver_supported(req: dict[str, Any]) -> dict[str, Any]:
            user_msg = next((m for m in req.get("messages", []) if m.get("role") == "user"), None)
            user_data = json.loads(user_msg["content"]) if user_msg and isinstance(user_msg.get("content"), str) else {}
            verdicts = []
            for c in user_data.get("afirmaciones", []):
                cid = c.get("claim_id", "C1")
                eids = c.get("evidence_ids", [])
                verdicts.append({
                    "claim_id": cid,
                    "verdict": "supported",
                    "reason": "Reparacion verificada y respaldada.",
                    "evidence_verdicts": [{"evidence_id": eid, "relation": "supports"} for eid in eids],
                })
            return make_raw_chat_response(TEXT_27B_MODEL, {"verdicts": verdicts})

        valid_initial_fact = {
            "cell_id": "c_r1c1",
            "concept": "Empleo",
            "header_path": ["2025"],
            "year": 2025,
            "ordered_delta_years": None,
            "unit": "% PIB",
            "measure": "level",
            "value": "10.2",
            "doc_id": self.pages[0]["doc_id"],
            "source": self.pages[0]["source_file"],
            "date": "2026-01-15",
            "page": 95,
            "table_id": "table_synth_cuadro1",
            "extraction_id": "synth_ext_01",
            "evidence_ids": ["E1", "E2"],
        }

        # Branch 1: repair with 'facts' array containing only schema-required fields
        def repair_facts_handler(req: dict[str, Any]) -> dict[str, Any]:
            fmt = req.get("format", {})
            user_msg = next((m for m in req.get("messages", []) if m.get("role") == "user"), None)
            user_data = json.loads(user_msg["content"]) if user_msg else {}
            t_ctx = user_data.get("tabla_contexto", {})
            cells = t_ctx.get("celdas_disponibles", [])
            req_keys = _extract_required_fact_keys(fmt)
            c = cells[0]
            cand = {
                "cell_id": c["cell_id"],
                "concept": c["concept"],
                "header_path": c["header_path"],
                "year": c.get("year"),
                "ordered_delta_years": c.get("ordered_delta_years"),
                "unit": c["unit"],
                "measure": c["measure"],
                "value": c["value"],
                "doc_id": t_ctx["doc_id"],
                "source": t_ctx["source"],
                "date": t_ctx["date"],
                "page": t_ctx["page"],
                "table_id": t_ctx["table_id"],
                "extraction_id": t_ctx["extraction_id"],
                "evidence_ids": list(c.get("evidence_ids", [])),
            }
            fact = {k: v for k, v in cand.items() if k in req_keys} if req_keys else cand
            return make_raw_chat_response(
                GENERATOR_9B_MODEL,
                {"repairs": [{"claim_id": "C1", "status": "answered", "facts": [fact], "reason": ""}]},
            )

        transport_rep_facts = PipelineFakeTransport(
            chat_responses=[
                make_raw_chat_response(GENERATOR_9B_MODEL, {"status": "selected", "ids": ["P1"], "reason": ""}),
                make_raw_chat_response(GENERATOR_9B_MODEL, {"status": "answered", "facts": [valid_initial_fact], "reason": ""}),
                ver_contradicted,
                make_raw_chat_response(GENERATOR_9B_MODEL, {"status": "selected", "ids": ["P1"], "reason": ""}),
                repair_facts_handler,
                ver_supported,
            ]
        )
        res_rep_facts = execute_table_attempt(
            attempt_spec,
            index_data=self.index,
            units_data=self.units,
            pages_data=self.pages,
            transport=transport_rep_facts,
            table_bundle=self.bundle,
            enabled=True,
        )
        self.assertEqual(res_rep_facts["status"], "completed", f"Error in repair facts: {res_rep_facts.get('error')}")
        self.assertEqual(res_rep_facts["factual_outcome"]["status"], "answered")

        # Branch 2: repair with single 'fact' object containing only schema-required fields
        def repair_single_fact_handler(req: dict[str, Any]) -> dict[str, Any]:
            fmt = req.get("format", {})
            user_msg = next((m for m in req.get("messages", []) if m.get("role") == "user"), None)
            user_data = json.loads(user_msg["content"]) if user_msg else {}
            t_ctx = user_data.get("tabla_contexto", {})
            cells = t_ctx.get("celdas_disponibles", [])
            req_keys = _extract_required_fact_keys(fmt)
            c = cells[0]
            cand = {
                "cell_id": c["cell_id"],
                "concept": c["concept"],
                "header_path": c["header_path"],
                "year": c.get("year"),
                "ordered_delta_years": c.get("ordered_delta_years"),
                "unit": c["unit"],
                "measure": c["measure"],
                "value": c["value"],
                "doc_id": t_ctx["doc_id"],
                "source": t_ctx["source"],
                "date": t_ctx["date"],
                "page": t_ctx["page"],
                "table_id": t_ctx["table_id"],
                "extraction_id": t_ctx["extraction_id"],
                "evidence_ids": list(c.get("evidence_ids", [])),
            }
            fact = {k: v for k, v in cand.items() if k in req_keys} if req_keys else cand
            return make_raw_chat_response(
                GENERATOR_9B_MODEL,
                {"repairs": [{"claim_id": "C1", "status": "answered", "fact": fact, "reason": ""}]},
            )

        transport_rep_single = PipelineFakeTransport(
            chat_responses=[
                make_raw_chat_response(GENERATOR_9B_MODEL, {"status": "selected", "ids": ["P1"], "reason": ""}),
                make_raw_chat_response(GENERATOR_9B_MODEL, {"status": "answered", "facts": [valid_initial_fact], "reason": ""}),
                ver_contradicted,
                make_raw_chat_response(GENERATOR_9B_MODEL, {"status": "selected", "ids": ["P1"], "reason": ""}),
                repair_single_fact_handler,
                ver_supported,
            ]
        )
        res_rep_single = execute_table_attempt(
            attempt_spec,
            index_data=self.index,
            units_data=self.units,
            pages_data=self.pages,
            transport=transport_rep_single,
            table_bundle=self.bundle,
            enabled=True,
        )
        self.assertEqual(res_rep_single["status"], "completed", f"Error in repair single: {res_rep_single.get('error')}")
        self.assertEqual(res_rep_single["factual_outcome"]["status"], "answered")

    def test_regression_negative_temporal_fields_omitted_or_wrong_blocked(self) -> None:
        """Negative cases: missing year, wrong year, missing delta, or wrong delta remain strictly blocked."""
        attempt_spec_level = {
            "attempt_index": 43,
            "case_id": "case_negative_temporal_level",
            "query": "Según el cuadro 1, ¿qué porcentaje del PIB figura para Empleo en 2025?",
            "arm": "mixed",
        }
        attempt_spec_delta = {
            "attempt_index": 44,
            "case_id": "case_negative_temporal_delta",
            "query": "Según el cuadro 1, ¿cuál es la variación para Empleo entre 2030 y 2025?",
            "arm": "mixed",
        }

        doc_id = self.pages[0]["doc_id"]
        source_file = self.pages[0]["source_file"]

        valid_level_fact = {
            "cell_id": "c_r1c1",
            "concept": "Empleo",
            "header_path": ["2025"],
            "year": 2025,
            "ordered_delta_years": None,
            "unit": "% PIB",
            "measure": "level",
            "value": "10.2",
            "doc_id": doc_id,
            "source": source_file,
            "date": "2026-01-15",
            "page": 95,
            "table_id": "table_synth_cuadro1",
            "extraction_id": "synth_ext_01",
            "evidence_ids": ["E1", "E2"],
        }

        # Case 1: Missing year key entirely (the exact observed diagnostic failure from live_synthetic_smoke_v1.json)
        fact_missing_year = copy.deepcopy(valid_level_fact)
        del fact_missing_year["year"]

        # Case 2: Year is None for level measure
        fact_none_year = copy.deepcopy(valid_level_fact)
        fact_none_year["year"] = None

        # Case 3: Wrong year for level measure
        fact_wrong_year = copy.deepcopy(valid_level_fact)
        fact_wrong_year["year"] = 2030

        for bad_fact, label in [
            (fact_missing_year, "missing_year_key"),
            (fact_none_year, "none_year_for_level"),
            (fact_wrong_year, "wrong_year_value"),
        ]:
            transport = PipelineFakeTransport(
                chat_responses=[
                    make_raw_chat_response(GENERATOR_9B_MODEL, {"status": "selected", "ids": ["P1"], "reason": ""}),
                    make_raw_chat_response(GENERATOR_9B_MODEL, {"status": "answered", "facts": [bad_fact], "reason": ""}),
                ]
            )
            res = execute_table_attempt(
                attempt_spec_level,
                index_data=self.index,
                units_data=self.units,
                pages_data=self.pages,
                transport=transport,
                table_bundle=self.bundle,
                enabled=True,
            )
            self.assertEqual(res["status"], "completed", f"Status error in {label}")
            self.assertEqual(res["factual_outcome"]["status"], "insufficient_evidence", f"Failed for {label}")
            audit = res["table_audit"]
            self.assertEqual(audit["gate_decision"], "rejected", f"Failed decision for {label}")
            self.assertTrue(audit["numeric_mismatch"], f"Failed mismatch for {label}")
            self.assertEqual(len(res["factual_outcome"]["publication"]["claims"]), 0)
            self.assertTrue(
                any("year" in r.lower() or "año" in r.lower() for r in audit["gate_reasons"]),
                f"Expected year rejection reason in {audit['gate_reasons']}",
            )

        valid_delta_fact = {
            "cell_id": "c_r1c3",
            "concept": "Empleo",
            "header_path": ["Δ 2030-2025"],
            "year": None,
            "ordered_delta_years": [2030, 2025],
            "unit": "puntos PIB",
            "measure": "delta",
            "value": "0.3",
            "doc_id": doc_id,
            "source": source_file,
            "date": "2026-01-15",
            "page": 95,
            "table_id": "table_synth_cuadro1",
            "extraction_id": "synth_ext_01",
            "evidence_ids": ["E1", "E2"],
        }

        # Case 4: Missing ordered_delta_years key entirely for delta measure
        fact_missing_delta = copy.deepcopy(valid_delta_fact)
        del fact_missing_delta["ordered_delta_years"]

        # Case 5: ordered_delta_years is None for delta measure
        fact_none_delta = copy.deepcopy(valid_delta_fact)
        fact_none_delta["ordered_delta_years"] = None

        # Case 6: Wrong ordered_delta_years (e.g. inverted [2025, 2030] instead of [2030, 2025])
        fact_wrong_delta = copy.deepcopy(valid_delta_fact)
        fact_wrong_delta["ordered_delta_years"] = [2025, 2030]

        for bad_fact, label in [
            (fact_missing_delta, "missing_delta_key"),
            (fact_none_delta, "none_delta_for_delta"),
            (fact_wrong_delta, "wrong_delta_years"),
        ]:
            transport = PipelineFakeTransport(
                chat_responses=[
                    make_raw_chat_response(GENERATOR_9B_MODEL, {"status": "selected", "ids": ["P1"], "reason": ""}),
                    make_raw_chat_response(GENERATOR_9B_MODEL, {"status": "answered", "facts": [bad_fact], "reason": ""}),
                ]
            )
            res = execute_table_attempt(
                attempt_spec_delta,
                index_data=self.index,
                units_data=self.units,
                pages_data=self.pages,
                transport=transport,
                table_bundle=self.bundle,
                enabled=True,
            )
            self.assertEqual(res["status"], "completed", f"Status error in {label}")
            self.assertEqual(res["factual_outcome"]["status"], "insufficient_evidence", f"Failed for {label}")
            audit = res["table_audit"]
            self.assertEqual(audit["gate_decision"], "rejected", f"Failed decision for {label}")
            self.assertTrue(audit["numeric_mismatch"], f"Failed mismatch for {label}")
            self.assertEqual(len(res["factual_outcome"]["publication"]["claims"]), 0)


if __name__ == "__main__":
    unittest.main()


from unittest.mock import patch


class TestDynamicIdentifiers(unittest.TestCase):
    def test_packet_ownership_free_values_and_isolation(self):
        from experiments.table_grounded import build_dynamic_fact_schema, TABLE_FACT_SCHEMA
        bundle, index, pages, units = _build_synthetic_fixtures()
        packets = build_table_packets(bundle, pages)
        ids = [p["cells"][0]["cell_id"] for p in packets]
        schema = build_dynamic_fact_schema(packets, ids)
        branches = schema.get("anyOf", [schema])
        for branch in branches:
            self.assertEqual({"type": "string"}, branch["properties"]["value"])
            owner = next(p for p in packets if p["cells"][0]["cell_id"] in branch["properties"]["cell_id"]["enum"])
            self.assertEqual({"const": owner["doc_id"]}, branch["properties"]["doc_id"])
        schema.clear()
        self.assertEqual({"type": "string"}, TABLE_FACT_SCHEMA["properties"]["cell_id"])
        for candidates in ([], [ids[0], ids[0]], ["foreign"]):
            with self.assertRaises(ValueError):
                build_dynamic_fact_schema(packets, candidates)
        with self.assertRaises(ValueError):
            build_dynamic_fact_schema(packets + [copy.deepcopy(packets[0])], ids)

    def test_actual_generation_repair_and_numeric_guard(self):
        from experiments import table_grounded as module
        original_init = module.TableGrounder.__init__
        original_chat = module.OllamaGrounder._chat_step
        seen = []
        def init(instance, *args, **kwargs):
            kwargs["table_dynamic_schema"] = True
            original_init(instance, *args, **kwargs)
        def chat(instance, stage, messages, schema, expected_digest):
            if stage in ("generation", "repair_generation") and instance._is_table_query and not instance._is_blocked:
                def visit(node):
                    if isinstance(node, dict):
                        props = node.get("properties", {})
                        if "cell_id" in props:
                            self.assertIn("enum", props["cell_id"])
                            self.assertIn("const", props["extraction_id"])
                            self.assertEqual({"type": "string"}, props["value"])
                            seen.append(stage)
                        for value in node.values(): visit(value)
                    elif isinstance(node, list):
                        for value in node: visit(value)
                visit(schema)
            return original_chat(instance, stage, messages, schema, expected_digest)
        with patch.object(module.TableGrounder, "__init__", init), patch.object(module.OllamaGrounder, "_chat_step", chat):
            for method in ("test_regression_fake_nn_emits_only_schema_required_fields_level_and_delta",
                           "test_regression_fake_nn_repair_schema_coverage", "test_repair_cannot_evade_guard"):
                case = TestTableGroundedPipeline(method)
                case.setUp()
                getattr(case, method)()
        self.assertIn("generation", seen)
        self.assertIn("repair_generation", seen)

    def test_strict_flag_before_inference(self):
        from experiments.table_grounded import execute_table_attempt
        for value in (1, None, "true"):
            with self.assertRaises(ValueError):
                execute_table_attempt({}, index_data={}, units_data={}, pages_data=[], transport=None,
                                      table_bundle={}, enabled=True, table_dynamic_schema=value)


class TestExplicitTableExpansion(unittest.TestCase):
    def setUp(self):
        self.bundle,self.index,self.pages,self.units=_build_synthetic_fixtures()
        self.packets=build_table_packets(self.bundle,self.pages)

    def test_unique_parent_all_rows_and_ambiguous_sources(self):
        from experiments.table_grounded import resolve_table_parent_unit
        parent=resolve_table_parent_unit(self.packets,self.units["units"])
        self.assertIsNotNone(parent)
        duplicate=copy.deepcopy(parent);duplicate["id"]+="_duplicate"
        self.assertIsNone(resolve_table_parent_unit(self.packets,[parent,duplicate]))
        foreign=copy.deepcopy(self.packets[0]);foreign["doc_id"]="foreign"
        self.assertIsNone(resolve_table_parent_unit(self.packets+[foreign],self.units["units"]))
        missing=copy.deepcopy(self.packets);missing[0]["evidence"]=[f for f in missing[0]["evidence"] if f["kind"]!="header"]
        self.assertIsNone(resolve_table_parent_unit(missing,self.units["units"]))
        overlap=copy.deepcopy(parent);overlap["spans"].append(copy.deepcopy(overlap["spans"][0]))
        self.assertIsNone(resolve_table_parent_unit(self.packets,[overlap]))
        for bad in (True,-1,0.5):
            malformed=copy.deepcopy(parent);malformed["spans"][0]["start_char"]=bad
            self.assertIsNone(resolve_table_parent_unit(self.packets,[malformed]))

    def test_raw_binding_budget_and_no_dense_hits(self):
        from experiments.table_grounded import resolve_table_parent_unit,_expand_explicit_table
        parent=resolve_table_parent_unit(self.packets,self.units["units"])
        pages={(p["doc_id"],p["page_number"]):p for p in self.pages}
        query="Según el cuadro 1, ¿qué porcentaje del PIB figura para Empleo en 2025?"
        hits,catalog,trace=_expand_explicit_table(parent,self.packets,[],self.units,pages,query,closed_note_tails=False,max_packages=1)
        self.assertLessEqual(len(trace["selected_parents"]),4)
        self.assertEqual("explicit_table_reference",trace["selected_parents"][0]["origin"])
        self.assertEqual(0.0,hits[0].score)
        with patch("experiments.table_grounded.expand_seeds", return_value=([],{},
                   {"selected_parents":[],"omitted_parents":[{"unit_id":parent["id"],"reason":"four_parent_limit"}]})):
            _,_,promoted=_expand_explicit_table(parent,self.packets,[],self.units,pages,query,closed_note_tails=False,max_packages=1)
        self.assertNotIn(parent["id"],{p["unit_id"] for p in promoted["omitted_parents"]})

        self.assertEqual(pages[(hits[0].chunk.doc_id,hits[0].chunk.page_number)]["text"][hits[0].chunk.start_char:hits[0].chunk.end_char],hits[0].chunk.text)
        corrupt=copy.deepcopy(parent);corrupt["spans"][0]["quote"]="invented quote"
        with self.assertRaises(ValueError):
            _expand_explicit_table(corrupt,self.packets,[],dict(self.units,units=[corrupt]),pages,query,closed_note_tails=False,max_packages=1)
        corrupt=copy.deepcopy(parent);corrupt["spans"][0]["end_char"]=10**8
        with self.assertRaises(ValueError):
            _expand_explicit_table(corrupt,self.packets,[],self.units,pages,query,closed_note_tails=False,max_packages=1)
        corrupt_packets=copy.deepcopy(self.packets);corrupt_packets[0]["evidence"][0]["quote"]="unbound fragment"
        with self.assertRaises(ValueError):
            _expand_explicit_table(parent,corrupt_packets,[],self.units,pages,query,closed_note_tails=False,max_packages=1)
        with self.assertRaises(ValueError):
            _expand_explicit_table(parent,self.packets,[],self.units,pages,"x"*10000,closed_note_tails=False,max_packages=1)

    def test_real_pipeline_outgoing_context_and_rank_immutability(self):
        from experiments import table_grounded as module
        original_init=module.TableGrounder.__init__
        original_answer=module.TableGrounder.answer
        observed=[]
        def init(instance,*args,**kwargs):
            kwargs.update(table_reference_expansion=True,table_dynamic_schema=True)
            original_init(instance,*args,**kwargs)
        def answer(instance,query,hits,*args,**kwargs):
            before=copy.deepcopy(hits)
            result=original_answer(instance,query,hits,*args,**kwargs)
            self.assertEqual(before,hits)
            if instance._is_table_query and not instance._is_blocked:
                self.assertEqual("explicit_table_reference",instance.expansion["selected_parents"][0]["origin"])
                self.assertEqual(len({p["unit_id"] for p in instance.expansion["selected_parents"]}),len(instance.expansion["selected_parents"]))
                self.assertLessEqual(len(instance.expansion["selected_parents"]),4)
                observed.append(True)
            return result
        with patch.object(module.TableGrounder,"__init__",init),patch.object(module.TableGrounder,"answer",answer):
            for method in ("test_regression_fake_nn_emits_only_schema_required_fields_level_and_delta","test_no_table_identical_delegation","test_repair_cannot_evade_guard"):
                case=TestTableGroundedPipeline(method);case.setUp();getattr(case,method)()
        self.assertTrue(observed)

    def test_expansion_flag_strict_and_requires_enabled(self):
        for flag,enabled in ((1,True),(None,True),(True,False)):
            with self.assertRaises(ValueError):
                execute_table_attempt({},index_data={},units_data={},pages_data=[],transport=None,table_bundle={},enabled=enabled,table_reference_expansion=flag)
