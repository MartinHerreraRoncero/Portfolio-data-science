"""Table-grounded RAG pipeline with structured evidence extraction and numeric guard.

Integrates literal table packet extraction, strict numeric fact verification,
and canonical claims publication with the mixed 9B generation + 27B verification RAG pipeline.
"""

from __future__ import annotations

import copy
import hashlib
import json
import time
from typing import Any, Callable

from experiments.compare_mixed_single27 import (
    EMBEDDING_4B_MODEL,
    GENERATOR_9B_MODEL,
    PINNED_9B_DIGEST,
    TEXT_27B_MODEL,
    ComparisonEmbedder,
    ComparisonGenerator,
    compute_role_stats,
    execute_attempt,
)
from experiments.preserve_evidence import (
    PreservedGrounder,
    bound_generation,
    enforce_binding,
    partial_generation_messages,
    publication,
    validate_catalog,
)
from experiments.preserved_q3_verifier import (
    Q3PreservedGrounder,
    Q3VerifierClient,
    VERIFIER_Q3_DIGEST,
    VERIFIER_Q3_MODEL,
)
from experiments.query_coverage import (
    audit_coverage,
    build_coverage_publication,
    format_covered_answer,
)
from experiments.reconstructed_units import (
    ReconstructedGrounder,
    candidate_messages,
    expand_seeds,
    make_hits,
    bundle_catalog,
)
from rag_pensiones.grounding import (
    MAX_CLAIM_REFS,
    MAX_EVIDENCE_BYTES,
    MAX_MESSAGES_BYTES,
    MAX_SELECTION_EVIDENCE,
    MIN_QUOTE_CHARS,
    MAX_QUOTE_CHARS,
    OllamaGrounder,
    build_selection_messages,
    collapse_spaces,
    validate_grounded_result,
)
from rag_pensiones.numeric_guard import (
    RE_EXPLICIT_TABLE,
    guard_table_facts,
    parse_canonical_decimal,
    parse_table_request,
    render_table_facts,
)
from rag_pensiones.chunking import ChunkRecord
from rag_pensiones.search import SearchHit, search
from rag_pensiones.table_evidence import (
    MAX_EVIDENCE_REFS,
    MAX_SUM_QUOTE_UTF8_BYTES,
    build_table_packets,
)

__all__ = [
    "TableGrounder",
    "execute_table_attempt",
    "BLOCKED_SELECTION_SCHEMA",
    "BLOCKED_GENERATION_SCHEMA",
    "TABLE_FACT_SCHEMA",
    "TABLE_GENERATION_SCHEMA",
    "TABLE_REPAIR_SCHEMA",
]

BLOCKED_SELECTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "status": {"const": "insufficient_evidence"},
        "ids": {"type": "array", "maxItems": 0},
        "reason": {"type": "string", "minLength": 1, "maxLength": 1200},
    },
    "required": ["status", "ids", "reason"],
    "additionalProperties": False,
}

BLOCKED_GENERATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "status": {"const": "insufficient_evidence"},
        "facts": {"type": "array", "maxItems": 0},
        "claims": {"type": "array", "maxItems": 0},
        "reason": {"type": "string", "minLength": 1, "maxLength": 1200},
    },
    "required": ["status", "reason"],
    "additionalProperties": False,
}

TABLE_FACT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "cell_id": {"type": "string"},
        "concept": {"type": "string"},
        "header_path": {"type": "array", "items": {"type": "string"}},
        "year": {"type": ["integer", "null"]},
        "ordered_delta_years": {
            "type": ["array", "null"],
            "items": {"type": "integer"},
        },
        "unit": {"type": "string"},
        "measure": {"type": "string", "enum": ["level", "delta"]},
        "value": {"type": "string"},
        "doc_id": {"type": "string"},
        "source": {"type": "string"},
        "date": {"type": "string"},
        "page": {"type": "integer"},
        "table_id": {"type": "string"},
        "extraction_id": {"type": "string"},
        "evidence_ids": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "cell_id",
        "concept",
        "header_path",
        "year",
        "ordered_delta_years",
        "unit",
        "measure",
        "value",
        "doc_id",
        "source",
        "date",
        "page",
        "table_id",
        "extraction_id",
        "evidence_ids",
    ],
    "additionalProperties": False,
}

TABLE_GENERATION_SCHEMA: dict[str, Any] = {
    "anyOf": [
        {
            "type": "object",
            "properties": {
                "status": {"const": "answered"},
                "facts": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 4,
                    "items": TABLE_FACT_SCHEMA,
                },
                "reason": {"const": ""},
            },
            "required": ["status", "facts", "reason"],
            "additionalProperties": False,
        },
        {
            "type": "object",
            "properties": {
                "status": {"const": "answered"},
                "claims": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 4,
                    "items": TABLE_FACT_SCHEMA,
                },
                "reason": {"const": ""},
            },
            "required": ["status", "claims", "reason"],
            "additionalProperties": False,
        },
        {
            "type": "object",
            "properties": {
                "status": {"const": "insufficient_evidence"},
                "facts": {"type": "array", "maxItems": 0},
                "reason": {"type": "string", "minLength": 1, "maxLength": 1200},
            },
            "required": ["status", "facts", "reason"],
            "additionalProperties": False,
        },
        {
            "type": "object",
            "properties": {
                "status": {"const": "insufficient_evidence"},
                "claims": {"type": "array", "maxItems": 0},
                "reason": {"type": "string", "minLength": 1, "maxLength": 1200},
            },
            "required": ["status", "claims", "reason"],
            "additionalProperties": False,
        },
    ]
}

TABLE_REPAIR_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "repairs": {
            "type": "array",
            "items": {
                "anyOf": [
                    {
                        "type": "object",
                        "properties": {
                            "claim_id": {"type": "string"},
                            "status": {"const": "answered"},
                            "facts": {
                                "type": "array",
                                "minItems": 1,
                                "maxItems": 4,
                                "items": TABLE_FACT_SCHEMA,
                            },
                            "reason": {"const": ""},
                        },
                        "required": ["claim_id", "status", "facts", "reason"],
                        "additionalProperties": False,
                    },
                    {
                        "type": "object",
                        "properties": {
                            "claim_id": {"type": "string"},
                            "status": {"const": "answered"},
                            "fact": TABLE_FACT_SCHEMA,
                            "reason": {"const": ""},
                        },
                        "required": ["claim_id", "status", "fact", "reason"],
                        "additionalProperties": False,
                    },
                    {
                        "type": "object",
                        "properties": {
                            "claim_id": {"type": "string"},
                            "status": {"const": "insufficient_evidence"},
                            "text": {"const": ""},
                            "evidence_ids": {"type": "array", "maxItems": 0},
                            "reason": {"type": "string", "minLength": 1, "maxLength": 1200},
                        },
                        "required": ["claim_id", "status", "reason"],
                        "additionalProperties": False,
                    },
                ]
            },
        }
    },
    "required": ["repairs"],
    "additionalProperties": False,
}


def _find_containing_hit(
    frag: dict[str, Any], expanded_hits: list[SearchHit]
) -> SearchHit | None:
    """Finds an expanded search hit that strictly contains the given fragment span."""
    for hit in expanded_hits:
        chunk = hit.chunk
        if (
            chunk.doc_id == frag.get("doc_id")
            and chunk.page_number == frag.get("page")
            and chunk.start_char <= frag["start_char"]
            and frag["end_char"] <= chunk.end_char
        ):
            return hit
    return None


def _build_table_package_citations(
    table_id: str,
    needed_packets: list[dict[str, Any]],
    expanded_hits: list[SearchHit],
    pages: list[dict[str, Any]],
) -> tuple[
    list[dict[str, str]],
    dict[str, str],
    dict[str, list[str]],
    dict[str, str],
    list[str],
]:
    """Builds coherent package citations for table grounding, combining adjacent caption+header if possible.

    Returns:
        (package_citations, source_mapping, eid_to_fragments, fragment_to_eid, missing_fragments)
    """
    source_mapping: dict[str, str] = {}
    package_citations: list[dict[str, str]] = []
    eid_to_fragments: dict[str, list[str]] = {}
    fragment_to_eid: dict[str, str] = {}
    missing_fragments: list[str] = []

    page_records: list[dict[str, Any]] = (
        list(pages.values()) if isinstance(pages, dict) else list(pages)
    )

    first_pkt = needed_packets[0]
    c_frag = next((f for f in first_pkt.get("evidence", []) if f["kind"] == "caption"), None)
    h_frag = next((f for f in first_pkt.get("evidence", []) if f["kind"] == "header"), None)

    combined_info: dict[str, Any] | None = None
    if c_frag is not None and h_frag is not None:
        if c_frag.get("doc_id") == h_frag.get("doc_id") and c_frag.get("page") == h_frag.get("page"):
            p_obj = next(
                (
                    p
                    for p in page_records
                    if p.get("doc_id") == c_frag.get("doc_id")
                    and p.get("page_number", p.get("page")) == c_frag.get("page")
                ),
                None,
            )
            if p_obj is not None:
                p_text = p_obj.get("text", "")
                c_s, c_e = c_frag["start_char"], c_frag["end_char"]
                h_s, h_e = h_frag["start_char"], h_frag["end_char"]
                s_char = min(c_s, h_s)
                e_char = max(c_e, h_e)
                span_len = e_char - s_char
                if 12 <= span_len <= 700 and 0 <= s_char < e_char <= len(p_text):
                    dummy = {
                        "doc_id": c_frag["doc_id"],
                        "page": c_frag["page"],
                        "start_char": s_char,
                        "end_char": e_char,
                    }
                    hit = _find_containing_hit(dummy, expanded_hits)
                    if hit is not None:
                        combined_info = {
                            "evidence_id": f"{table_id}_caption_header",
                            "doc_id": c_frag["doc_id"],
                            "page": c_frag["page"],
                            "start_char": s_char,
                            "end_char": e_char,
                            "quote": p_text[s_char:e_char],
                            "hit": hit,
                            "combines": [c_frag["evidence_id"], h_frag["evidence_id"]],
                        }

    if combined_info is not None:
        hit = combined_info["hit"]
        sid = f"S{hit.rank}"
        comb_quote = combined_info["quote"]
        for fid in combined_info["combines"]:
            source_mapping[fid] = sid
        source_mapping[combined_info["evidence_id"]] = sid
        package_citations.append({"source_id": sid, "quote": comb_quote})
        cur_eid = f"E{len(package_citations)}"
        eid_to_fragments[cur_eid] = list(combined_info["combines"])
        for fid in combined_info["combines"]:
            fragment_to_eid[fid] = cur_eid
        fragment_to_eid[combined_info["evidence_id"]] = cur_eid
    else:
        for frag in [c_frag, h_frag]:
            if frag is not None:
                hit = _find_containing_hit(frag, expanded_hits)
                if hit is None:
                    missing_fragments.append(frag["evidence_id"])
                else:
                    sid = f"S{hit.rank}"
                    source_mapping[frag["evidence_id"]] = sid
                    package_citations.append({"source_id": sid, "quote": frag["quote"]})
                    cur_eid = f"E{len(package_citations)}"
                    eid_to_fragments[cur_eid] = [frag["evidence_id"]]
                    fragment_to_eid[frag["evidence_id"]] = cur_eid

    seen_row_frag_ids: set[str] = set()
    for pkt in needed_packets:
        for frag in pkt.get("evidence", []):
            if frag["kind"] == "row":
                fid = frag["evidence_id"]
                if fid not in seen_row_frag_ids:
                    seen_row_frag_ids.add(fid)
                    hit = _find_containing_hit(frag, expanded_hits)
                    if hit is None:
                        missing_fragments.append(fid)
                    else:
                        sid = f"S{hit.rank}"
                        source_mapping[fid] = sid
                        package_citations.append({"source_id": sid, "quote": frag["quote"]})
                        cur_eid = f"E{len(package_citations)}"
                        eid_to_fragments[cur_eid] = [fid]
                        fragment_to_eid[fid] = cur_eid

    return (
        package_citations,
        source_mapping,
        eid_to_fragments,
        fragment_to_eid,
        missing_fragments,
    )


def build_dynamic_fact_schema(pkts: list[dict[str, Any]], candidate_cids: list[str]) -> dict[str, Any]:
    """Constrain source identities and cell ownership, leaving values to extraction."""
    if (not isinstance(pkts, list) or not pkts or not isinstance(candidate_cids, list)
            or not candidate_cids or any(not isinstance(c, str) or not c for c in candidate_cids)
            or len(set(candidate_cids)) != len(candidate_cids)):
        raise ValueError("Nonempty unique candidate cells and packets required")
    owners = {}
    branches = []
    identity_keys = ("doc_id", "source", "date", "page", "table_id", "extraction_id")
    for pkt in pkts:
        if not isinstance(pkt, dict) or not isinstance(pkt.get("cells"), list):
            raise ValueError("Invalid table packet")
        for key in identity_keys:
            value = pkt.get(key)
            if (key == "page" and (type(value) is not int or value < 1)) or (
                    key != "page" and (not isinstance(value, str) or not value.strip())):
                raise ValueError(f"Invalid packet identity: {key}")
        ids = []
        for cell in pkt["cells"]:
            cid = cell.get("cell_id") if isinstance(cell, dict) else None
            if not isinstance(cid, str) or not cid or cid in owners:
                raise ValueError("Invalid or ambiguous cell ownership")
            owners[cid] = pkt
            if cid in candidate_cids:
                ids.append(cid)
        if ids:
            branch = copy.deepcopy(TABLE_FACT_SCHEMA)
            branch["properties"]["cell_id"] = {"type": "string", "enum": ids}
            for key in identity_keys:
                branch["properties"][key] = {"const": pkt[key]}
            branches.append(branch)
    if any(cid not in owners for cid in candidate_cids):
        raise ValueError("Foreign candidate cell")
    return branches[0] if len(branches) == 1 else {"anyOf": branches}


def _with_dynamic_facts(schema, fact_schema):
    if schema == TABLE_FACT_SCHEMA:
        return copy.deepcopy(fact_schema)
    if isinstance(schema, dict):
        return {k: _with_dynamic_facts(v, fact_schema) for k, v in schema.items()}
    if isinstance(schema, list):
        return [_with_dynamic_facts(v, fact_schema) for v in schema]
    return copy.deepcopy(schema)


def resolve_table_parent_unit(needed_pkts, extracted_units):
    """Find a unique source-bound parent containing every required table fragment."""
    if not isinstance(needed_pkts, list) or not needed_pkts or not isinstance(extracted_units, list):
        return None
    fragments = []
    identity = None
    for pkt in needed_pkts:
        if not isinstance(pkt, dict):
            return None
        current = tuple(pkt.get(k) for k in ("doc_id", "source", "date", "table_id", "extraction_id"))
        if any(not isinstance(v, str) or not v for v in current) or (identity and identity != current):
            return None
        identity = current
        evidence = pkt.get("evidence")
        if not isinstance(evidence, list) or not {"caption", "header", "row"} <= {f.get("kind") for f in evidence if isinstance(f, dict)}:
            return None
        for f in evidence:
            if (not isinstance(f, dict) or tuple(f.get(k) for k in ("doc_id", "source", "date")) != identity[:3]
                or type(f.get("page")) is not int or f["page"] < 1
                or type(f.get("start_char")) is not int or type(f.get("end_char")) is not int
                or not 0 <= f["start_char"] < f["end_char"]
                or not isinstance(f.get("quote"), str) or not f["quote"]):
                return None
            fragments.append(f)
    matches, ids = [], set()
    for unit in extracted_units:
        if not isinstance(unit, dict) or not isinstance(unit.get("id"), str) or not unit["id"] or unit["id"] in ids:
            return None
        ids.add(unit["id"])
        spans = unit.get("spans")
        if not isinstance(spans, list) or not spans:
            continue
        valid_spans, previous = True, {}
        for s in spans:
            if (not isinstance(s, dict)
                or (s.get("doc_id"), s.get("source_file"), s.get("published_at")) != identity[:3]
                or type(s.get("page_number")) is not int or s["page_number"] < 1
                or type(s.get("start_char")) is not int or type(s.get("end_char")) is not int
                or not 0 <= s["start_char"] < s["end_char"]):
                valid_spans = False
                break
            key = (s["doc_id"], s["page_number"])
            if s["start_char"] < previous.get(key, 0):
                valid_spans = False
                break
            previous[key] = s["end_char"]
        if not valid_spans:
            continue
        def contains(s, f):
            return (isinstance(s, dict) and (s.get("doc_id"), s.get("source_file"), s.get("published_at")) == identity[:3]
                    and type(s.get("page_number")) is int and s["page_number"] == f["page"]
                    and type(s.get("start_char")) is int and type(s.get("end_char")) is int
                    and 0 <= s["start_char"] <= f["start_char"] < f["end_char"] <= s["end_char"])
        if all(any(contains(s, f) for s in spans) for f in fragments):
            matches.append(unit)
    return copy.deepcopy(matches[0]) if len(matches) == 1 else None


def _expand_explicit_table(parent, packets, seeds, extracted, pages, query, *, closed_note_tails, max_packages):
    # Validate original raw geometry and metadata before any slicing or model call.
    for span in parent["spans"]:
        page = pages.get((span.get("doc_id"), span.get("page_number")))
        if (page is None or type(span.get("start_char")) is not int or type(span.get("end_char")) is not int
            or not 0 <= span["start_char"] < span["end_char"] <= len(page["text"])
            or span.get("source_file") != page.get("source_file")
            or span.get("published_at") != page.get("published_at")):
            raise ValueError("Invalid structural parent raw span")
    for pkt in packets:
        for frag in pkt["evidence"]:
            page = pages.get((frag["doc_id"], frag["page"]))
            if (page is None or frag["end_char"] > len(page["text"])
                or collapse_spaces(page["text"][frag["start_char"]:frag["end_char"]]) != collapse_spaces(frag["quote"])):
                raise ValueError("Table fragment lost raw page provenance")
    make_hits([parent], pages)  # checks normalized catalog quotes against raw text
    try:
        _, _, dense = expand_seeds(seeds, extracted, pages, query,
                                   closed_note_tails=closed_note_tails, max_packages=max_packages)
    except ValueError:
        dense = {"selected_parents": [], "omitted_parents": []}
    by_id = {u["id"]: u for u in extracted["units"]}
    chosen, provenance = [], []
    omitted = [copy.deepcopy(p) for p in dense["omitted_parents"] if p["unit_id"] != parent["id"]]
    order = [dict(unit_id=parent["id"], origin="explicit_table_reference", reason="contains_all_required_fragments",
                  structural_score=0.0, score_semantics="placeholder_not_dense_similarity")]
    order += [p for p in dense["selected_parents"] if p["unit_id"] != parent["id"]]
    for item in order:
        if len(chosen) == 4:
            omitted.append(dict(unit_id=item["unit_id"], reason="four_parent_limit"))
            continue
        trial = chosen + [by_id[item["unit_id"]]]
        hits, mapping = make_hits(trial, pages)
        catalog = bundle_catalog(trial, mapping, closed_note_tails=closed_note_tails)
        try:
            build_selection_messages(query, hits)
            messages = candidate_messages({"pregunta": query}, catalog, max_packages=max_packages)
            if len(json.dumps(messages, ensure_ascii=False).encode("utf-8")) > MAX_MESSAGES_BYTES:
                raise ValueError("Structural expansion exceeds message budget")
        except ValueError:
            if not chosen:
                raise ValueError("Required table parent cannot fit intact")
            omitted.append(dict(unit_id=item["unit_id"], reason="message_budget"))
            continue
        chosen = trial
        provenance.append(copy.deepcopy(item))
    hits, mapping = make_hits(chosen, pages)
    return hits, bundle_catalog(chosen, mapping, closed_note_tails=closed_note_tails), {
        "selected_parents": provenance, "omitted_parents": omitted, "parent_policy": "explicit_table_first"}


class TableGrounder(Q3PreservedGrounder):
    """Grounder for table queries with structured evidence extraction and numeric guard."""

    def __init__(
        self,
        extracted: dict[str, Any],
        pages: list[dict[str, Any]],
        *,
        table_bundle: dict[str, Any],
        verifier: Any,
        generator: Any = None,
        table_dynamic_schema: bool = False,
        table_reference_expansion: bool = False,
        **kwargs: Any,
    ) -> None:
        if type(table_dynamic_schema) is not bool:
            raise ValueError("table_dynamic_schema must be a strict boolean")
        self.table_dynamic_schema = table_dynamic_schema
        if type(table_reference_expansion) is not bool:
            raise ValueError("table_reference_expansion must be a strict boolean")
        self.table_reference_expansion = table_reference_expansion
        if not isinstance(table_bundle, dict):
            raise ValueError("table_bundle must be a dict")
        self.table_bundle = table_bundle
        self.pages_raw = pages
        if generator is None and hasattr(verifier, "transport") and verifier.transport is not None:
            generator = ComparisonGenerator(
                GENERATOR_9B_MODEL, transport=verifier.transport
            )
        super().__init__(extracted, pages, verifier=verifier, generator=generator, **kwargs)

        self.all_table_packets = build_table_packets(table_bundle, pages)

        self._reset_query_state()

    def _reset_query_state(self) -> None:
        """Resets dynamic query state to prevent inter-request leakage."""
        self._is_table_query = False
        self._is_blocked = False
        self._blocked_reason = ""
        self._current_query = ""
        self._table_request: dict[str, Any] | None = None
        self._eligible_packets: list[dict[str, Any]] = []
        self._source_mapping: dict[str, str] = {}
        self._eid_to_fragments: dict[str, list[str]] = {}
        self._fragment_to_eid: dict[str, str] = {}
        self._claim_to_cell_id: dict[str, str] = {}
        self._active_package_citations: list[dict[str, str]] = []
        self._selected_package_refs: list[str] = []

        self.table_audit: dict[str, Any] = {
            "enabled": True,
            "table_dynamic_schema": self.table_dynamic_schema,
            "table_reference_expansion": self.table_reference_expansion,
            "is_table_query": False,
            "request": None,
            "retrieval_absence": False,
            "numeric_mismatch": False,
            "eligible_packets": [],
            "source_mapping": {},
            "original_typed_facts": [],
            "mapped_guard_facts": [],
            "gate_decision": "bypassed",
            "gate_reasons": [],
            "gate_stage": "none",
            "rendering": [],
            "refusals": [],
            "role_traces": [],
        }

    def answer(
        self,
        query: str,
        hits: list[SearchHit],
        original_query: str | None = None,
        expected_digest: str | None = None,
    ) -> dict[str, Any]:
        """Answers a query, routing table requests through the numeric guard."""
        self._reset_query_state()
        self._current_query = query

        real_hits: list[SearchHit] = []
        for h in hits:
            if isinstance(h, SearchHit):
                real_hits.append(h)
            else:
                chunk = h.chunk
                c_kwargs = {
                    "chunk_id": getattr(chunk, "chunk_id", ""),
                    "doc_id": getattr(chunk, "doc_id", ""),
                    "source_file": getattr(chunk, "source_file", ""),
                    "published_at": getattr(chunk, "published_at", ""),
                    "page_number": getattr(chunk, "page_number", 1),
                    "start_char": getattr(chunk, "start_char", 0),
                    "end_char": getattr(chunk, "end_char", 0),
                    "text": getattr(chunk, "text", ""),
                    "strategy": getattr(chunk, "strategy", "layout_block"),
                }
                real_hits.append(SearchHit(rank=h.rank, score=h.score, chunk=ChunkRecord(**c_kwargs)))

        req = parse_table_request(query, self.all_table_packets)
        if req is None:
            # Normal query with no explicit table reference: delegate existing pipeline unchanged
            self._is_table_query = False
            self.table_audit["is_table_query"] = False
            self.table_audit["gate_decision"] = "bypassed"
            return super().answer(query, real_hits, original_query=original_query, expected_digest=expected_digest)

        self._is_table_query = True
        self._table_request = copy.deepcopy(req)
        self.table_audit["is_table_query"] = True
        self.table_audit["request"] = copy.deepcopy(req)

        # Early gate: Unsupported explicit table reference
        if req.get("status") == "unsupported":
            reason = req.get("reason", "Consulta explícita de tabla no soportada.")
            self._is_blocked = True
            self._blocked_reason = reason
            self.table_audit["gate_decision"] = "refused"
            self.table_audit["gate_stage"] = "early_request_filter"
            self.table_audit["gate_reasons"] = [reason]
            self.table_audit["refusals"] = [reason]
            self.table_audit["retrieval_absence"] = False
            self.table_audit["eligible_packets"] = []
            self.expanded_hits = copy.deepcopy(real_hits)
            self.catalog = {"records": [], "omitted": []}
            self.expansion = {"selected_parents": [], "omitted_parents": []}
            self.selections = []
            self.bindings = []
            return OllamaGrounder.answer(self, query, real_hits, original_query, expected_digest)

        table_id = req.get("table_id")
        matching_packets = [p for p in self.all_table_packets if p.get("table_id") == table_id]
        target_cells = req.get("target_cells", [])
        target_cell_ids = {tc["cell_id"] for tc in target_cells}
        needed_packets: list[dict[str, Any]] = []
        for p in matching_packets:
            p_cids = {c["cell_id"] for c in p.get("cells", [])}
            if p_cids & target_cell_ids:
                needed_packets.append(p)

        if self.table_reference_expansion:
            parent = resolve_table_parent_unit(needed_packets, self.extracted["units"])
            try:
                if parent is None:
                    raise ValueError("No unique parent bound to all requested table fragments")
                self.expanded_hits, self.catalog, self.expansion = _expand_explicit_table(
                    parent, needed_packets, real_hits, self.extracted, self.pages, query,
                    closed_note_tails=self.closed_note_tails, max_packages=self.max_packages)
            except (ValueError, KeyError, TypeError) as exc:
                self._is_blocked = True
                self._blocked_reason = f"No se pudo ampliar la tabla con procedencia y contexto íntegros: {exc}"
                self.table_audit.update(gate_decision="refused", gate_stage="structural_expansion",
                                        gate_reasons=[self._blocked_reason], refusals=[self._blocked_reason])
                self.expanded_hits = copy.deepcopy(real_hits)
                self.expansion = {"selected_parents": [], "omitted_parents": []}
                self.catalog = {"records": [], "omitted": []}
                self.selections, self.bindings = [], []
                return OllamaGrounder.answer(self, query, real_hits, original_query, expected_digest)
        else:
            # Retain original retrieval top 4 and expand_seeds parent ranking
            self.expanded_hits, self.catalog, self.expansion = expand_seeds(
                real_hits,
                self.extracted,
                self.pages,
                query,
                closed_note_tails=self.closed_note_tails,
                max_packages=self.max_packages,
            )


        if not matching_packets or not needed_packets:
            reason = (
                f"Tabla {table_id!r} no encontrada en los paquetes de evidencia disponibles."
                if not matching_packets
                else "No se encontraron paquetes para las celdas solicitadas."
            )
            self._is_blocked = True
            self._blocked_reason = reason
            self.table_audit["retrieval_absence"] = True
            self.table_audit["gate_decision"] = "refused"
            self.table_audit["gate_stage"] = "retrieval_expansion"
            self.table_audit["gate_reasons"] = [reason]
            self.table_audit["refusals"] = [reason]
            self.table_audit["eligible_packets"] = []
            self.catalog = {"records": [], "omitted": []}
            self.selections = []
            self.bindings = []
            return OllamaGrounder.answer(self, query, self.expanded_hits, original_query, expected_digest)

        # Build coherent package citations, combining adjacent caption+header into <=700 slice if possible
        (
            package_citations,
            source_mapping,
            eid_to_fragments,
            fragment_to_eid,
            missing_fragments,
        ) = _build_table_package_citations(
            table_id, needed_packets, self.expanded_hits, self.pages
        )

        if missing_fragments:
            reason = (
                f"Las citas de la tabla {table_id!r} no están contenidas en las fuentes expandidas "
                f"de la misma página ({missing_fragments})."
            )
            self._is_blocked = True
            self._blocked_reason = reason
            self.table_audit["retrieval_absence"] = True
            self.table_audit["gate_decision"] = "refused"
            self.table_audit["gate_stage"] = "retrieval_expansion"
            self.table_audit["gate_reasons"] = [reason]
            self.table_audit["refusals"] = [reason]
            self.table_audit["eligible_packets"] = [p.get("table_id") for p in needed_packets]
            self.catalog = {"records": [], "omitted": []}
            self.selections = []
            self.bindings = []
            return OllamaGrounder.answer(self, query, self.expanded_hits, original_query, expected_digest)

        # Validate package citation and UTF-8 byte budgets against MAX_CLAIM_REFS (3)
        if len(package_citations) > MAX_CLAIM_REFS:
            reason = f"El paquete coherente de tabla supera el límite de {MAX_CLAIM_REFS} citas ({len(package_citations)})."
            self._is_blocked = True
            self._blocked_reason = reason
            self.table_audit["gate_decision"] = "refused"
            self.table_audit["gate_stage"] = "citation_budget"
            self.table_audit["gate_reasons"] = [reason]
            self.table_audit["refusals"] = [reason]
            self.table_audit["retrieval_absence"] = False
            self.table_audit["eligible_packets"] = [p.get("table_id") for p in needed_packets]
            self.catalog = {"records": [], "omitted": []}
            self.selections = []
            self.bindings = []
            return OllamaGrounder.answer(self, query, self.expanded_hits, original_query, expected_digest)

        total_utf8 = sum(len(c["quote"].encode("utf-8")) for c in package_citations)
        if total_utf8 > MAX_SUM_QUOTE_UTF8_BYTES:
            reason = f"El paquete coherente de tabla supera el presupuesto de {MAX_SUM_QUOTE_UTF8_BYTES} bytes UTF-8 ({total_utf8})."
            self._is_blocked = True
            self._blocked_reason = reason
            self.table_audit["gate_decision"] = "refused"
            self.table_audit["gate_stage"] = "citation_budget"
            self.table_audit["gate_reasons"] = [reason]
            self.table_audit["refusals"] = [reason]
            self.table_audit["retrieval_absence"] = False
            self.table_audit["eligible_packets"] = [p.get("table_id") for p in needed_packets]
            self.catalog = {"records": [], "omitted": []}
            self.selections = []
            self.bindings = []
            return OllamaGrounder.answer(self, query, self.expanded_hits, original_query, expected_digest)

        for c in package_citations:
            if not (MIN_QUOTE_CHARS <= len(c["quote"]) <= MAX_QUOTE_CHARS):
                reason = f"Cita de tabla fuera de longitud permitida [12..700]: {len(c['quote'])} chars"
                self._is_blocked = True
                self._blocked_reason = reason
                self.table_audit["gate_decision"] = "refused"
                self.table_audit["gate_stage"] = "citation_budget"
                self.table_audit["gate_reasons"] = [reason]
                self.table_audit["refusals"] = [reason]
                self.table_audit["retrieval_absence"] = False
                self.table_audit["eligible_packets"] = [p.get("table_id") for p in needed_packets]
                self.catalog = {"records": [], "omitted": []}
                self.selections = []
                self.bindings = []
                return OllamaGrounder.answer(self, query, self.expanded_hits, original_query, expected_digest)

        self._is_blocked = False
        self.table_audit["retrieval_absence"] = False
        self.table_audit["eligible_packets"] = [p.get("table_id") for p in needed_packets]
        self.table_audit["source_mapping"] = copy.deepcopy(source_mapping)
        self._eligible_packets = copy.deepcopy(needed_packets)
        self._source_mapping = copy.deepcopy(source_mapping)
        self._eid_to_fragments = copy.deepcopy(eid_to_fragments)
        self._fragment_to_eid = copy.deepcopy(fragment_to_eid)
        self._active_package_citations = copy.deepcopy(package_citations)

        # Replace narrative table splitting at selection hook with coherent table package
        table_package = {
            "id": "P1",
            "unit_id": f"table_{table_id}",
            "sentence": 1,
            "evidence": package_citations,
        }
        self.catalog = {"records": [table_package], "omitted": []}

        self.selections = []
        return OllamaGrounder.answer(self, query, self.expanded_hits, original_query, expected_digest)

    def _chat_step(
        self,
        stage: str,
        messages: list[dict],
        response_schema: dict,
        expected_digest: str,
    ) -> tuple[dict, dict]:
        """Routes chat steps, executing numeric guard on table generation outputs."""
        if not self._is_table_query:
            return super()._chat_step(stage, messages, response_schema, expected_digest)

        if stage in ("verification", "reverification"):
            # Route verification through Q3VerifierClient with 27B
            payload, trace = super()._chat_step(stage, messages, response_schema, expected_digest)
            self.table_audit["role_traces"].append(copy.deepcopy(trace))
            return payload, trace

        if stage in ("selection", "repair_selection"):
            if self._is_blocked:
                # Role switch: unload 27B verifier if resident before running 9B generator
                self.verifier.release_loaded(VERIFIER_Q3_MODEL)

                blocked_messages = [
                    {
                        "role": "system",
                        "content": (
                            "Selecciona paquetes de evidencia relevantes o declara evidencia insuficiente "
                            "si no es posible responder la consulta solicitada."
                        ),
                    },
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "pregunta": self._current_query,
                                "aviso": self._blocked_reason,
                                "catalogo": [],
                            },
                            ensure_ascii=False,
                        ),
                    },
                ]
                serialized_len = len(json.dumps(blocked_messages, ensure_ascii=False).encode("utf-8"))
                if serialized_len > MAX_MESSAGES_BYTES:
                    raise ValueError(f"Blocked selection prompt exceeds budget of {MAX_MESSAGES_BYTES} bytes")

                raw_payload, trace = OllamaGrounder._chat_step(
                    self, stage, blocked_messages, BLOCKED_SELECTION_SCHEMA, expected_digest
                )
                self.table_audit["role_traces"].append(copy.deepcopy(trace))
                self.selections.append({"stage": stage, "payload": copy.deepcopy(raw_payload)})

                payload = {
                    "status": "insufficient_evidence",
                    "evidence": [],
                    "reason": raw_payload.get("reason") or self._blocked_reason,
                }
                return payload, trace

            # Execute selection with candidate messages and coherent package
            payload, trace = super()._chat_step(stage, messages, response_schema, expected_digest)
            self.table_audit["role_traces"].append(copy.deepcopy(trace))
            return payload, trace

        if stage in ("generation", "repair_generation"):
            # Apply generic schema for 9B table generation
            gen_schema = TABLE_GENERATION_SCHEMA if stage == "generation" else TABLE_REPAIR_SCHEMA

            # Extract user message to obtain refs for binding from catalog
            # Search backwards for the last matching user message with the required catalog key
            repair = stage == "repair_generation"
            req_key = "catalogo_evidencias_reparacion" if repair else "catalogo_evidencias"
            target_idx = None
            target_user_data = None
            for idx in range(len(messages) - 1, -1, -1):
                m = messages[idx]
                if m.get("role") == "user":
                    try:
                        data = json.loads(m.get("content", ""))
                        if isinstance(data, dict) and req_key in data:
                            target_idx = idx
                            target_user_data = data
                            break
                    except Exception:
                        continue
            if target_idx is None:
                raise ValueError(f"No user message containing {req_key!r} found in chat messages")

            catalog = target_user_data.get(req_key, [])
            refs = [e["evidence_id"] for e in catalog]
            if len(refs) > MAX_CLAIM_REFS or len(refs) != len(set(refs)):
                raise ValueError("Complete package cannot fit unchanged per-claim reference limit")

            table_messages = copy.deepcopy(messages)

            # Compact model-visible table grounding context
            first_pkt = self._eligible_packets[0]
            target_cids = {tc["cell_id"] for tc in self._table_request.get("target_cells", [])}
            available_cells = []
            for pkt in self._eligible_packets:
                for cell in pkt.get("cells", []):
                    if cell["cell_id"] in target_cids:
                        seen_eids: set[str] = set()
                        visible_eids = []
                        for fid in cell.get("evidence_ids", []):
                            eid = self._fragment_to_eid.get(fid, fid)
                            if eid and eid not in seen_eids:
                                seen_eids.add(eid)
                                visible_eids.append(eid)
                        available_cells.append({
                            "cell_id": cell["cell_id"],
                            "concept": cell["row_label"],
                            "header_path": cell["header_path"],
                            "year": cell["year"],
                            "ordered_delta_years": cell["ordered_delta_years"],
                            "unit": cell["unit"],
                            "measure": cell["measure"],
                            "value": cell["value_decimal"],
                            "evidence_ids": visible_eids,
                        })

            table_ctx = {
                "table_id": self._table_request.get("table_id"),
                "doc_id": first_pkt.get("doc_id"),
                "source": first_pkt.get("source"),
                "date": first_pkt.get("date"),
                "page": first_pkt.get("page"),
                "extraction_id": first_pkt.get("extraction_id"),
                "evidence_mapping": {
                    "visible_to_authoritative": self._eid_to_fragments,
                    "authoritative_to_visible": self._fragment_to_eid,
                },
                "celdas_disponibles": available_cells,
            }

            target_user_data_copy = copy.deepcopy(target_user_data)
            target_user_data_copy["tabla_contexto"] = table_ctx
            table_messages[target_idx]["content"] = json.dumps(target_user_data_copy, ensure_ascii=False)

            table_messages[0]["content"] += (
                "\nConserva el paquete completo de referencias en cada afirmación respondida. "
                "Expresa el sujeto y los periodos necesarios sin depender de «este sistema» o "
                "«esta estimación». Responde solo lo solicitado; no añadas causas no explícitas. "
                "El paquete puede contener contexto: su presencia no demuestra apoyo semántico. "
                "Si no basta para contestar, abstente.\n"
                f"Contexto de tabla {table_ctx['table_id']}: doc_id={table_ctx['doc_id']}, "
                f"page={table_ctx['page']}, extraction_id={table_ctx['extraction_id']}."
            )

            budget_overflow = False
            budget_overflow_reason = ""
            if stage == "generation" and self.partial_generation:
                try:
                    table_messages = partial_generation_messages(table_messages)
                except ValueError as exc:
                    if "exceeds budget" in str(exc):
                        budget_overflow = True
                        budget_overflow_reason = (
                            f"Contexto de tabla supera el presupuesto de {MAX_MESSAGES_BYTES} bytes UTF-8 "
                            f"tras partial_generation_messages: {exc}"
                        )
                    else:
                        raise

            # Final 6500-byte check AFTER all mutations including partial_generation_messages
            serialized_bytes = len(json.dumps(table_messages, ensure_ascii=False).encode("utf-8"))
            if serialized_bytes > MAX_MESSAGES_BYTES:
                budget_overflow = True
                budget_overflow_reason = (
                    f"Contexto de tabla supera el presupuesto de {MAX_MESSAGES_BYTES} bytes UTF-8 ({serialized_bytes})."
                )

            if budget_overflow:
                self.table_audit["gate_decision"] = "refused"
                self.table_audit["gate_stage"] = "context_budget"
                self.table_audit["gate_reasons"] = [budget_overflow_reason]
                self.table_audit["refusals"] = [budget_overflow_reason]

                self.verifier.release_loaded(VERIFIER_Q3_MODEL)

                blocked_messages = [
                    {
                        "role": "system",
                        "content": (
                            "Declara evidencia insuficiente cuando el contexto supere el presupuesto disponible. "
                            "No incluyas afirmaciones numéricas ni hechos inventados."
                        ),
                    },
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "pregunta": self._current_query,
                                "aviso": budget_overflow_reason,
                                "catalogo": [],
                            },
                            ensure_ascii=False,
                        ),
                    },
                ]

                raw_payload, trace = OllamaGrounder._chat_step(
                    self, stage, blocked_messages, BLOCKED_GENERATION_SCHEMA, expected_digest
                )
                self.table_audit["role_traces"].append(copy.deepcopy(trace))
                final_reason = raw_payload.get("reason") or budget_overflow_reason

                if stage == "generation":
                    self.table_audit["original_typed_facts"] = []
                    self.table_audit["mapped_guard_facts"] = []
                    return {
                        "status": "insufficient_evidence",
                        "claims": [],
                        "reason": final_reason,
                    }, trace
                else:
                    # repair_generation
                    failed_list = target_user_data.get("afirmaciones_a_reparar", [{"claim_id": "C1"}])
                    adapted_repairs = [
                        {
                            "claim_id": c.get("claim_id", f"C{i+1}"),
                            "status": "insufficient_evidence",
                            "text": "",
                            "evidence_ids": [],
                            "reason": final_reason,
                        }
                        for i, c in enumerate(failed_list)
                    ]
                    adapted_payload = {"repairs": adapted_repairs}
                    return adapted_payload, trace

            # Role switch: unload 27B verifier if resident before running 9B generator
            if self.table_dynamic_schema:
                fact_schema = build_dynamic_fact_schema(self._eligible_packets, sorted(target_cids))
                gen_schema = _with_dynamic_facts(gen_schema, fact_schema)
            self.verifier.release_loaded(VERIFIER_Q3_MODEL)

            # Raw chat step directly using OllamaGrounder without second prose binder
            payload, trace = OllamaGrounder._chat_step(
                self, stage, table_messages, gen_schema, expected_digest
            )
            self.table_audit["role_traces"].append(copy.deepcopy(trace))

            allowed_visible_eids = set(self._eid_to_fragments.keys())

            if stage == "generation":
                raw_facts = payload.get("facts")
                if raw_facts is None:
                    raw_facts = payload.get("claims")
                if not isinstance(raw_facts, list):
                    raw_facts = []

                self.table_audit["original_typed_facts"] = copy.deepcopy(raw_facts)

                if payload.get("status") == "insufficient_evidence":
                    reason = payload.get("reason", "Modelo declaró evidencia insuficiente.")
                    self.table_audit["gate_decision"] = "refused"
                    self.table_audit["gate_stage"] = "generation_model_abstention"
                    self.table_audit["gate_reasons"] = [reason]
                    self.table_audit["refusals"] = [reason]
                    self.table_audit["mapped_guard_facts"] = []
                    return {
                        "status": "insufficient_evidence",
                        "claims": [],
                        "reason": reason,
                    }, trace

                # Validate model-visible evidence_ids (reject duplicates, unknown, or internal IDs)
                invalid_refs_reason: str | None = None
                for f in raw_facts:
                    if not isinstance(f, dict):
                        invalid_refs_reason = "Hecho no es un diccionario válido"
                        break
                    eids = f.get("evidence_ids")
                    if not isinstance(eids, list) or not eids:
                        invalid_refs_reason = f"Hecho {f.get('cell_id')} carece de evidence_ids como lista no vacía"
                        break
                    if len(eids) != len(set(eids)):
                        invalid_refs_reason = f"Hecho {f.get('cell_id')} contiene identificadores duplicados: {eids}"
                        break
                    unknown = [eid for eid in eids if eid not in allowed_visible_eids]
                    if unknown:
                        invalid_refs_reason = (
                            f"Hecho {f.get('cell_id')} contiene referencias desconocidas o privadas: {unknown}. "
                            f"Referencias visibles permitidas: {sorted(allowed_visible_eids)}"
                        )
                        break

                if invalid_refs_reason is not None:
                    self.table_audit["numeric_mismatch"] = True
                    self.table_audit["gate_decision"] = "rejected"
                    self.table_audit["gate_stage"] = "numeric_guard"
                    self.table_audit["gate_reasons"] = [invalid_refs_reason]
                    self.table_audit["refusals"] = [invalid_refs_reason]
                    self.table_audit["mapped_guard_facts"] = []
                    return {
                        "status": "insufficient_evidence",
                        "claims": [],
                        "reason": invalid_refs_reason,
                    }, trace

                # Map visible evidence_ids (E1, E2, etc.) to deduplicated internal fragment IDs
                mapped_facts = copy.deepcopy(raw_facts)
                for f in mapped_facts:
                    mapped_fids: list[str] = []
                    seen_fids: set[str] = set()
                    for eid in f["evidence_ids"]:
                        for fid in self._eid_to_fragments[eid]:
                            if fid not in seen_fids:
                                seen_fids.add(fid)
                                mapped_fids.append(fid)
                    f["evidence_ids"] = mapped_fids

                self.table_audit["mapped_guard_facts"] = copy.deepcopy(mapped_facts)

                # Apply numeric guard before constructing canonical claim text
                guard_res = guard_table_facts(
                    mapped_facts, self._table_request, self._eligible_packets
                )

                if guard_res.get("decision") != "accepted":
                    reasons = guard_res.get("reasons", ["Fallo en guardia numérica de tabla."])
                    self.table_audit["numeric_mismatch"] = True
                    self.table_audit["gate_decision"] = "rejected"
                    self.table_audit["gate_stage"] = "numeric_guard"
                    self.table_audit["gate_reasons"] = copy.deepcopy(reasons)
                    self.table_audit["refusals"] = copy.deepcopy(reasons)

                    refusal_payload = {
                        "status": "insufficient_evidence",
                        "claims": [],
                        "reason": "; ".join(reasons),
                    }
                    return refusal_payload, trace

                self.table_audit["numeric_mismatch"] = False
                self.table_audit["gate_decision"] = "accepted"
                self.table_audit["gate_stage"] = "numeric_guard"
                self.table_audit["gate_reasons"] = []

                # Render canonical claims from verified typed facts
                rendered = render_table_facts(
                    guard_res.get("accepted_facts", []),
                    self._table_request,
                    self._eligible_packets,
                )
                self.table_audit["rendering"] = copy.deepcopy(rendered)

                # Store mapping from claim_id to target cell_id for repair binding
                self._claim_to_cell_id = {
                    f"C{idx + 1}": r.get("cell_id") for idx, r in enumerate(rendered)
                }

                # Format claims for downstream pipeline with binding to complete selected literal refs
                grounded_claims = []
                for idx, r_claim in enumerate(rendered):
                    grounded_claims.append({
                        "claim_id": f"C{idx + 1}",
                        "text": r_claim["text"],
                        "evidence_ids": copy.deepcopy(refs),
                    })

                adapted_payload = {
                    "status": "answered",
                    "claims": grounded_claims,
                    "reason": "",
                }
                self.bindings.append({
                    "stage": stage,
                    "required_refs": refs,
                    "payload": copy.deepcopy(adapted_payload),
                })
                enforce_binding(adapted_payload, stage, refs)
                return adapted_payload, trace

            else:
                # stage == "repair_generation"
                repairs = payload.get("repairs", [])
                adapted_repairs = []
                for rep in repairs:
                    rep_cid = rep.get("claim_id", "C1")
                    if rep.get("status") == "insufficient_evidence":
                        adapted_repairs.append({
                            "claim_id": rep_cid,
                            "status": "insufficient_evidence",
                            "text": "",
                            "evidence_ids": [],
                            "reason": rep.get("reason") or "Evidencia insuficiente para reparar afirmación.",
                        })
                        continue

                    # Extract facts from repair item
                    rep_facts = rep.get("facts")
                    if rep_facts is None and "fact" in rep:
                        rep_facts = [rep["fact"]]
                    if not isinstance(rep_facts, list) or not rep_facts:
                        adapted_repairs.append({
                            "claim_id": rep_cid,
                            "status": "insufficient_evidence",
                            "text": "",
                            "evidence_ids": [],
                            "reason": "Reparación carece de hechos numéricos tipados requeridos.",
                        })
                        continue

                    # Validate model-visible evidence_ids (reject duplicates, unknown, or internal IDs)
                    invalid_refs: str | None = None
                    for f in rep_facts:
                        if not isinstance(f, dict):
                            invalid_refs = "Hecho en reparación no es un diccionario válido"
                            break
                        eids = f.get("evidence_ids")
                        if not isinstance(eids, list) or not eids:
                            invalid_refs = f"Hecho {f.get('cell_id')} en reparación carece de evidence_ids válidos"
                            break
                        if len(eids) != len(set(eids)):
                            invalid_refs = f"Hecho {f.get('cell_id')} contiene identificadores duplicados en reparación: {eids}"
                            break
                        unknown = [eid for eid in eids if eid not in allowed_visible_eids]
                        if unknown:
                            invalid_refs = (
                                f"Hecho {f.get('cell_id')} contiene referencias desconocidas o privadas en reparación: {unknown}. "
                                f"Referencias visibles permitidas: {sorted(allowed_visible_eids)}"
                            )
                            break

                    if invalid_refs is not None:
                        adapted_repairs.append({
                            "claim_id": rep_cid,
                            "status": "insufficient_evidence",
                            "text": "",
                            "evidence_ids": [],
                            "reason": invalid_refs,
                        })
                        continue

                    # Map visible evidence_ids to deduplicated internal fragment IDs
                    mapped_rep_facts = copy.deepcopy(rep_facts)
                    for f in mapped_rep_facts:
                        mapped_fids = []
                        seen_fids = set()
                        for eid in f["evidence_ids"]:
                            for fid in self._eid_to_fragments[eid]:
                                if fid not in seen_fids:
                                    seen_fids.add(fid)
                                    mapped_fids.append(fid)
                        f["evidence_ids"] = mapped_fids

                    # Bind repair claim_id to original cell(s), per-claim authoritative subrequest
                    target_cell = getattr(self, "_claim_to_cell_id", {}).get(rep_cid)
                    sub_request = copy.deepcopy(self._table_request)
                    if target_cell:
                        matching_targets = [
                            tc for tc in self._table_request.get("target_cells", [])
                            if tc.get("cell_id") == target_cell
                        ]
                        if matching_targets:
                            sub_request["target_cells"] = matching_targets

                    rep_guard = guard_table_facts(
                        mapped_rep_facts, sub_request, self._eligible_packets
                    )
                    if rep_guard.get("decision") != "accepted":
                        reasons = rep_guard.get("reasons") or ["Reparación rechazada por guardia."]
                        adapted_repairs.append({
                            "claim_id": rep_cid,
                            "status": "insufficient_evidence",
                            "text": "",
                            "evidence_ids": [],
                            "reason": "; ".join(reasons) or "Reparación rechazada por guardia.",
                        })
                    else:
                        rep_rendered = render_table_facts(
                            rep_guard.get("accepted_facts", []),
                            sub_request,
                            self._eligible_packets,
                        )
                        if not rep_rendered:
                            adapted_repairs.append({
                                "claim_id": rep_cid,
                                "status": "insufficient_evidence",
                                "text": "",
                                "evidence_ids": [],
                                "reason": "No se pudieron renderizar afirmaciones verificadas a partir de la reparación.",
                            })
                        else:
                            matched = next((r for r in rep_rendered if r.get("cell_id") == target_cell), None)
                            if matched is not None:
                                text = matched["text"]
                            elif len(rep_rendered) == 1:
                                text = rep_rendered[0]["text"]
                            else:
                                text = ""

                            if not text:
                                adapted_repairs.append({
                                    "claim_id": rep_cid,
                                    "status": "insufficient_evidence",
                                    "text": "",
                                    "evidence_ids": [],
                                    "reason": "No se pudo vincular la afirmación reparada a la celda original correspondiente.",
                                })
                            else:
                                adapted_repairs.append({
                                    "claim_id": rep_cid,
                                    "status": "answered",
                                    "text": text,
                                    "evidence_ids": copy.deepcopy(refs),
                                    "reason": "",
                                })

                adapted_repair_payload = {"repairs": adapted_repairs}
                self.bindings.append({
                    "stage": stage,
                    "required_refs": refs,
                    "payload": copy.deepcopy(adapted_repair_payload),
                })
                enforce_binding(adapted_repair_payload, stage, refs)
                return adapted_repair_payload, trace

        return super()._chat_step(stage, messages, response_schema, expected_digest)


def execute_table_attempt(
    attempt_spec: dict[str, Any],
    *,
    index_data: dict[str, Any],
    units_data: dict[str, Any],
    pages_data: list[dict[str, Any]],
    transport: Any,
    table_bundle: dict[str, Any],
    enabled: bool,
    table_dynamic_schema: bool = False,
    table_reference_expansion: bool = False,
) -> dict[str, Any]:
    """Executes a single evaluation attempt with optional table grounding.

    Returns the legacy raw attempt structure plus a separate 'table_audit' dictionary.
    """
    if type(table_reference_expansion) is not bool:
        raise ValueError("table_reference_expansion must be a strict boolean")
    if table_reference_expansion and not enabled:
        raise ValueError("table_reference_expansion requires table grounding")
    if type(table_dynamic_schema) is not bool:
        raise ValueError("table_dynamic_schema must be a strict boolean")
    if table_dynamic_schema and not enabled:
        raise ValueError("table_dynamic_schema requires table grounding")
    if type(enabled) is not bool:
        raise ValueError(
            f"enabled must be a strict boolean, got {type(enabled).__name__} ({enabled!r})"
        )

    if not enabled:
        raw_attempt = execute_attempt(
            attempt_spec,
            index_data=index_data,
            units_data=units_data,
            pages_data=pages_data,
            transport=transport,
            grounder_factory=None,
        )
        raw_attempt["table_audit"] = {
            "enabled": False,
            "is_table_query": False,
            "gate_decision": "bypassed",
        }
        return raw_attempt

    # enabled is True: validate arm before inference
    arm = attempt_spec.get("arm")
    if arm != "mixed":
        raise ValueError(
            f"grounder_factory is only supported for the 'mixed' arm, got arm={arm!r}"
        )

    captured_grounders: list[TableGrounder] = []

    def factory(
        units: dict[str, Any],
        pages: list[dict[str, Any]],
        *,
        verifier: Any,
        generator: Any = None,
        partial_generation: bool = True,
    ) -> TableGrounder:
        tg = TableGrounder(
            units,
            pages,
            table_bundle=table_bundle,
            verifier=verifier,
            generator=generator,
            partial_generation=partial_generation,
            table_dynamic_schema=table_dynamic_schema,
            table_reference_expansion=table_reference_expansion,
        )
        captured_grounders.append(tg)
        return tg

    raw_attempt = execute_attempt(
        attempt_spec,
        index_data=index_data,
        units_data=units_data,
        pages_data=pages_data,
        transport=transport,
        grounder_factory=factory,
    )

    if captured_grounders:
        raw_attempt["table_audit"] = copy.deepcopy(captured_grounders[0].table_audit)
    else:
        raw_attempt["table_audit"] = {
            "enabled": True,
            "is_table_query": False,
            "gate_decision": "bypassed",
        }

    return raw_attempt
