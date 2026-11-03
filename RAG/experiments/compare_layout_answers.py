"""Phase 2 & Phase 3 layout-aware comparison vs legacy pypdf text.

Phase 2 (Preparation): rebuilds fresh indexes for pages 95..99 for both baseline (legacy)
and candidate (layout_text), reanchors the 13 primary semantic units, constructs
the literal T95 table unit for diagnostic queries, builds the counterbalanced
34-attempt schedule, and freezes the complete protocol and all generated assets.

Phase 3 (Execution & Review Export): executes the frozen 34-attempt matrix using
compare_mixed_single27.execute_attempt (with arm="mixed" ALWAYS and condition routing),
performs strict pre- and post-attempt integrity checks against source, input, and protocol drift,
retains condition failures in the 34 denominator, and exports a neutral A/B review bundle.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Callable

from experiments.compare_mixed_single27 import (
    MixedSingle27Transport,
    execute_attempt as default_execute_attempt,
    validate_queries_payload,
)
from experiments.preserve_evidence import collapse_spaces, validate_catalog
from experiments.preserved_q3_verifier import (
    VERIFIER_Q3_DIGEST,
    VERIFIER_Q3_MODEL,
    VERIFIER_Q3_PARAMETERS,
)
from experiments.reanchor_layout_catalog import reanchor_catalog
from rag_pensiones.answer import DEFAULT_GENERATION_MODEL
from rag_pensiones.chunking import ChunkRecord, chunk_pages, write_chunks_jsonl
from rag_pensiones.grounding import DEFAULT_PARAMETERS
from rag_pensiones.ingest import PROJECT_ROOT, validate_project_path
from rag_pensiones.qwen_embeddings import (
    DIMENSION,
    MODEL as QWEN_EMBEDDING_MODEL,
    QUERY_PREFIX,
    SCHEMA as QWEN_SCHEMA,
    QwenEmbedder,
    validate_index as validate_qwen_index,
    vector,
)
from rag_pensiones.search import load_index

SCHEMA_VERSION = "layout-answer-effect-v1"
RESULTS_SCHEMA_VERSION = "layout-answer-effect-results-v1"
REVIEW_SCHEMA_VERSION = "layout-answer-review-v1"
REVIEW_KEY_SCHEMA_VERSION = "layout-answer-review-key-v1"
SERVER_VERSION = "0.40.11"

DEFAULT_LEGACY_PAGES_PATH = Path("data/processed/airef_2025_pages.jsonl")
DEFAULT_LAYOUT_BUNDLE_PATH = Path("evaluation/document_layout_2026-10-27/pilot_v4/structured.json")
DEFAULT_CATALOG_PATH = Path("evaluation/reconstructed_units_catalog.json")
DEFAULT_PRIMARY_QUERIES_PATH = Path("evaluation/scope_coverage_2026-10-26/full_inputs.json")
DEFAULT_SECONDARY_QUERIES_PATH = Path("evaluation/layout_answer_effect_2026-10-27/secondary_queries.json")
DEFAULT_BASELINE_INDEX_PATH = Path("data/processed/airef_2025_qwen3_4b_embeddings.json")
DEFAULT_OLLAMA_DEFAULTS_REL = "evaluation/verifier_capacity_2026-10-26/ollama_defaults_v0.40.11.json"
DEFAULT_REFERENCE_FREEZE_PATH = Path("evaluation/layout_answer_effect_2026-10-27/reference_freeze.json")
DEFAULT_MANIFEST_PATH = Path("evaluation/document_layout_2026-10-27/pilot_v4/manifest.json")

REQUIRED_INPUT_PATHS = (
    DEFAULT_LEGACY_PAGES_PATH.as_posix(),
    DEFAULT_LAYOUT_BUNDLE_PATH.as_posix(),
    DEFAULT_CATALOG_PATH.as_posix(),
    DEFAULT_PRIMARY_QUERIES_PATH.as_posix(),
    DEFAULT_SECONDARY_QUERIES_PATH.as_posix(),
    DEFAULT_BASELINE_INDEX_PATH.as_posix(),
    DEFAULT_REFERENCE_FREEZE_PATH.as_posix(),
    DEFAULT_MANIFEST_PATH.as_posix(),
    DEFAULT_OLLAMA_DEFAULTS_REL,
)

CLOSED_ARM_ROLE_FILES = {
    "baseline": {
        "pages_file": "baseline_pages.jsonl",
        "index_file": "baseline_index.json",
        "primary_catalog_file": "baseline_primary_catalog.json",
        "table_catalog_file": "baseline_table_catalog.json",
    },
    "layout": {
        "pages_file": "layout_pages.jsonl",
        "index_file": "layout_index.json",
        "primary_catalog_file": "layout_primary_catalog.json",
        "table_catalog_file": "layout_table_catalog.json",
    },
}


def _json_type_equal(a: Any, b: Any) -> bool:
    """Type-aware equality for JSON data structures (distinguishing bool from int/float)."""
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        if set(a.keys()) != set(b.keys()):
            return False
        return all(_json_type_equal(a[k], b[k]) for k in a)
    if isinstance(a, list):
        if len(a) != len(b):
            return False
        return all(_json_type_equal(x, y) for x, y in zip(a, b))
    return a == b

PINNED_9B_DIGEST = "56671c2ab9385f9cfcb404638e32cd62d88e3501d44822208363c010179a3c90"
PINNED_27B_DIGEST = VERIFIER_Q3_DIGEST
PINNED_EMBEDDING_DIGEST = "df5bd2e3c74cd8d069d21dc038f1b359fcdc9458fce1c99bd43c9eb1518ff907"

ALLOWED_EXPERIMENT_MODELS = (
    DEFAULT_GENERATION_MODEL,
    VERIFIER_Q3_MODEL,
    QWEN_EMBEDDING_MODEL,
)


def _reject_constant(val: str) -> None:
    raise ValueError(f"Non-finite JSON constant rejected: {val!r}")


def _parse_float(val: str) -> float:
    v = float(val)
    if not math.isfinite(v):
        raise ValueError("JSON float overflow or non-finite float")
    return v


def checkpoint_atomic(data: dict[str, Any], destination: Path) -> None:
    """Atomically checkpoint JSON state to disk."""
    temp = destination.with_name(f"{destination.name}.tmp_{os.getpid()}")
    temp.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temp, destination)


def compute_source_closure() -> dict[str, str]:
    """Computes SHA-256 for all .py files under rag_pensiones, experiments, and tests."""
    closure: dict[str, str] = {}
    for pkg in ("rag_pensiones", "experiments", "tests"):
        pkg_dir = PROJECT_ROOT / pkg
        if pkg_dir.exists():
            for py_file in sorted(pkg_dir.rglob("*.py")):
                rel_path = py_file.relative_to(PROJECT_ROOT).as_posix()
                closure[rel_path] = hashlib.sha256(py_file.read_bytes()).hexdigest()
    defaults_file = PROJECT_ROOT / DEFAULT_OLLAMA_DEFAULTS_REL
    if defaults_file.exists():
        closure[DEFAULT_OLLAMA_DEFAULTS_REL] = hashlib.sha256(defaults_file.read_bytes()).hexdigest()
    return closure


def build_schedule(primary: dict[str, Any], secondary: dict[str, Any]) -> list[dict[str, Any]]:
    """Build deterministic 34-attempt schedule alternating condition per question.

    Primary cohort: 11 benchmark queries (b01..b08, n01..n03).
    Table cohort: 6 diagnostic queries (t01..t06).
    Conditions: 'legacy' (baseline) and 'layout' (candidate).
    Arm is ALWAYS 'mixed'.
    """
    primary_cases = validate_queries_payload(primary)

    if not isinstance(secondary, dict):
        raise ValueError(f"secondary must be a dict, got {type(secondary).__name__}")
    sec_cases = secondary.get("cases")
    if not isinstance(sec_cases, list) or len(sec_cases) != 6:
        raise ValueError(
            f"Secondary queries must contain exactly 6 cases, got {len(sec_cases) if isinstance(sec_cases, list) else type(sec_cases)}"
        )

    expected_sec_ids = ["t01", "t02", "t03", "t04", "t05", "t06"]
    for idx, c in enumerate(sec_cases):
        if not isinstance(c, dict):
            raise ValueError(f"Invalid case at index {idx} in secondary queries")
        keys = set(c.keys())
        if keys != {"id", "query"}:
            extra = keys - {"id", "query"}
            if any(k.lower() in ("gold", "answer", "label") for k in extra):
                raise ValueError(f"Forbidden gold label keys in secondary query {c.get('id')!r}: {sorted(extra)}")
            raise ValueError(f"Secondary case keys must be exactly {{'id', 'query'}}, got {sorted(keys)}")
        cid = c.get("id")
        if cid != expected_sec_ids[idx]:
            raise ValueError(f"Secondary case at index {idx} expected id {expected_sec_ids[idx]!r}, got {cid!r}")
        q = c.get("query")
        if not isinstance(q, str) or not q.strip():
            raise ValueError(f"Query for secondary case {cid} must be a non-empty string")

    all_questions: list[dict[str, str]] = []
    for c in primary_cases:
        all_questions.append({"cohort": "primary", "case_id": c["id"], "query": c["query"]})
    for c in sec_cases:
        all_questions.append({"cohort": "table", "case_id": c["id"], "query": c["query"]})

    schedule: list[dict[str, Any]] = []
    attempt_idx = 1
    seen_keys: set[tuple[str, str, str]] = set()

    for q_idx, q_item in enumerate(all_questions):
        if q_idx % 2 == 0:
            conditions = ["legacy", "layout"]
        else:
            conditions = ["layout", "legacy"]

        for cond in conditions:
            key = (q_item["cohort"], q_item["case_id"], cond)
            if key in seen_keys:
                raise ValueError(f"Duplicate attempt key detected: {key}")
            seen_keys.add(key)

            schedule.append({
                "attempt_index": attempt_idx,
                "question_index": q_idx,
                "cohort": q_item["cohort"],
                "case_id": q_item["case_id"],
                "query": q_item["query"],
                "condition": cond,
                "arm": "mixed",
            })
            attempt_idx += 1

    if len(schedule) != 34:
        raise ValueError(f"Expected exactly 34 attempts in schedule, got {len(schedule)}")
    return schedule


def filter_and_validate_pages(
    legacy_pages_data: list[dict[str, Any]],
    layout_structured_data: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Filter legacy and layout pages to exactly 95..99 and verify identity and content."""
    legacy_95_99 = [
        p for p in legacy_pages_data
        if type(p.get("page_number")) is int and 95 <= p["page_number"] <= 99
    ]
    if len(legacy_95_99) != 5:
        raise ValueError(f"Expected exactly 5 legacy pages (95..99), got {len(legacy_95_99)}")
    legacy_by_num = {p["page_number"]: p for p in legacy_95_99}
    if set(legacy_by_num.keys()) != {95, 96, 97, 98, 99}:
        raise ValueError(f"Missing pages in legacy: expected 95..99, got {sorted(legacy_by_num.keys())}")

    pilot_pages = layout_structured_data.get("pages", [])
    if not isinstance(pilot_pages, list):
        raise ValueError("layout structured data missing 'pages' list")
    layout_95_99 = [
        p for p in pilot_pages
        if type(p.get("page_number")) is int and 95 <= p["page_number"] <= 99
    ]
    if len(layout_95_99) != 5:
        raise ValueError(f"Expected exactly 5 layout pages (95..99), got {len(layout_95_99)}")
    layout_by_num = {p["page_number"]: p for p in layout_95_99}
    if set(layout_by_num.keys()) != {95, 96, 97, 98, 99}:
        raise ValueError(f"Missing pages in layout: expected 95..99, got {sorted(layout_by_num.keys())}")

    meta = layout_structured_data.get("metadata", {})
    expected_doc_id = meta.get("doc_id")
    expected_source_file = meta.get("source_file")
    expected_published_at = meta.get("published_at")

    baseline_pages: list[dict[str, Any]] = []
    layout_pages: list[dict[str, Any]] = []

    for pnum in (95, 96, 97, 98, 99):
        l_page = legacy_by_num[pnum]
        lay_page = layout_by_num[pnum]

        if (
            l_page["doc_id"] != expected_doc_id
            or l_page["source_file"] != expected_source_file
            or l_page["published_at"] != expected_published_at
        ):
            raise ValueError(f"Legacy page {pnum} metadata mismatch with structured metadata")

        lay_legacy_text = lay_page.get("legacy_text")
        if lay_legacy_text is None:
            raise ValueError(f"Layout page {pnum} missing 'legacy_text'")
        if lay_legacy_text != l_page["text"]:
            raise ValueError(f"Layout page {pnum} 'legacy_text' does not match old page text exactly")

        layout_text = lay_page.get("layout_text")
        if not isinstance(layout_text, str) or not layout_text.strip():
            raise ValueError(f"Layout page {pnum} missing or empty 'layout_text'")

        baseline_pages.append({
            "doc_id": l_page["doc_id"],
            "source_file": l_page["source_file"],
            "published_at": l_page["published_at"],
            "page_number": pnum,
            "text": l_page["text"],
        })

        layout_pages.append({
            "doc_id": l_page["doc_id"],
            "source_file": l_page["source_file"],
            "published_at": l_page["published_at"],
            "page_number": pnum,
            "text": layout_text,
        })

    return baseline_pages, layout_pages


def extract_t95_units(
    baseline_page_95: dict[str, Any],
    layout_page_95: dict[str, Any],
    layout_structured_page_95: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Extract literal T95 unit for baseline (pypdf) and layout (Docling table)."""
    base_text = baseline_page_95["text"]
    cap_anchor = "CUADRO 9."
    cap_matches: list[int] = []
    idx = 0
    while True:
        pos = base_text.find(cap_anchor, idx)
        if pos == -1:
            break
        cap_matches.append(pos)
        idx = pos + 1
    if len(cap_matches) != 1:
        raise ValueError(f"Expected unique '{cap_anchor}' caption in baseline page 95, found {len(cap_matches)}")
    cap_pos = cap_matches[0]

    src_anchor = "Fuente: IGAE y AIReF"
    src_matches: list[int] = []
    idx = cap_pos
    while True:
        pos = base_text.find(src_anchor, idx)
        if pos == -1:
            break
        src_matches.append(pos)
        idx = pos + 1
    if len(src_matches) != 1:
        raise ValueError(f"Expected unique '{src_anchor}' after caption in baseline page 95, found {len(src_matches)}")
    src_pos = src_matches[0]
    src_end = src_pos + len(src_anchor)

    base_slice = base_text[cap_pos:src_end]
    base_quote = collapse_spaces(base_slice)

    t95_baseline = {
        "id": "T95",
        "kind": "body",
        "complete_end": True,
        "spans": [
            {
                "doc_id": baseline_page_95["doc_id"],
                "source_file": baseline_page_95["source_file"],
                "published_at": baseline_page_95["published_at"],
                "page_number": 95,
                "start_char": cap_pos,
                "end_char": src_end,
                "quote": base_quote,
            }
        ],
    }

    blocks = layout_structured_page_95.get("blocks", [])
    table_blocks = [
        b for b in blocks
        if isinstance(b, dict) and b.get("type") == "table" and "CUADRO 9" in b.get("text", "")
    ]
    if len(table_blocks) != 1:
        raise ValueError(f"Expected unique table block matching 'CUADRO 9' in layout page 95 blocks, found {len(table_blocks)}")
    table_block = table_blocks[0]
    block_text = table_block.get("text", "")
    if not block_text:
        raise ValueError("Layout table block text is empty")

    layout_text = layout_page_95["text"]
    lay_matches: list[int] = []
    idx = 0
    while True:
        pos = layout_text.find(block_text, idx)
        if pos == -1:
            break
        lay_matches.append(pos)
        idx = pos + 1
    if len(lay_matches) != 1:
        raise ValueError(f"Expected unique table block text in layout_text, found {len(lay_matches)}")
    lay_start = lay_matches[0]
    lay_end = lay_start + len(block_text)

    lay_slice = layout_text[lay_start:lay_end]
    lay_quote = collapse_spaces(lay_slice)

    t95_layout = {
        "id": "T95",
        "kind": "body",
        "complete_end": True,
        "spans": [
            {
                "doc_id": layout_page_95["doc_id"],
                "source_file": layout_page_95["source_file"],
                "published_at": layout_page_95["published_at"],
                "page_number": 95,
                "start_char": lay_start,
                "end_char": lay_end,
                "quote": lay_quote,
            }
        ],
    }

    return t95_baseline, t95_layout


def build_index(pages: list[dict[str, Any]], embedder: Any) -> dict[str, Any]:
    """Build deterministic Qwen 4B index embedding raw chunk text."""
    if not isinstance(pages, list) or not pages:
        raise ValueError("pages must be a non-empty list of page dicts")

    page_records: list[ChunkRecord] = []
    for p in pages:
        if not isinstance(p, dict):
            raise ValueError(f"Each page must be a dict, got {type(p).__name__}")
        for k in ("doc_id", "source_file", "published_at", "page_number", "text"):
            if k not in p:
                raise ValueError(f"Page missing required field: {k}")
        from rag_pensiones.ingest import PageRecord
        page_records.append(
            PageRecord(
                doc_id=p["doc_id"],
                source_file=p["source_file"],
                published_at=p["published_at"],
                page_number=p["page_number"],
                text=p["text"],
            )
        )

    if embedder is None or not hasattr(embedder, "get_digest") or not hasattr(embedder, "embed_texts"):
        raise ValueError("Valid embedder with get_digest and embed_texts required")

    digest_before = embedder.get_digest()
    if not isinstance(digest_before, str) or not digest_before.strip():
        raise ValueError(f"Invalid embedder digest before embedding: {digest_before!r}")

    chunks = chunk_pages(page_records, max_chars=1000, overlap_chars=150)
    if not chunks:
        raise ValueError("chunk_pages generated no chunks")

    texts = [c.text for c in chunks]
    raw_vectors = embedder.embed_texts(texts)

    digest_after = embedder.get_digest()
    if digest_after != digest_before:
        raise RuntimeError(
            f"Embedder digest mismatch: started with {digest_before!r}, became {digest_after!r}"
        )

    if not isinstance(raw_vectors, list) or len(raw_vectors) != len(texts):
        raise ValueError(
            f"Embedding count mismatch: expected {len(texts)} vectors, got {len(raw_vectors) if isinstance(raw_vectors, list) else type(raw_vectors)}"
        )

    norm_vectors = [vector(v) for v in raw_vectors]

    sorted_pages = sorted(pages, key=lambda p: (p["doc_id"], p["page_number"]))
    recipe_dict = {
        "max_chars": 1000,
        "overlap_chars": 150,
        "strategy": "page-local-v1:max_chars=1000:overlap_chars=150",
        "pages": [
            {
                "doc_id": p["doc_id"],
                "page_number": p["page_number"],
                "text_sha256": hashlib.sha256(p["text"].encode("utf-8")).hexdigest(),
            }
            for p in sorted_pages
        ],
    }
    base_index_hash = hashlib.sha256(
        json.dumps(recipe_dict, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()

    chunk_dicts = []
    for c, v in zip(chunks, norm_vectors):
        chunk_dicts.append({
            "chunk_id": c.chunk_id,
            "doc_id": c.doc_id,
            "source_file": c.source_file,
            "published_at": c.published_at,
            "page_number": c.page_number,
            "start_char": c.start_char,
            "end_char": c.end_char,
            "text": c.text,
            "strategy": c.strategy,
            "embedding": v,
        })

    index_data = {
        "schema_version": QWEN_SCHEMA,
        "model": QWEN_EMBEDDING_MODEL,
        "dimension": DIMENSION,
        "model_digest": digest_before,
        "document_prefix": "",
        "query_prefix": QUERY_PREFIX,
        "base_index_hash": base_index_hash,
        "chunks": chunk_dicts,
    }

    validate_qwen_index(index_data)
    return index_data


def verify_baseline_chunks_against_existing(
    generated_baseline_chunks: list[dict[str, Any]],
    baseline_index_path: Path,
) -> None:
    """Verify that freshly generated baseline chunks match existing Qwen index descriptors."""
    existing_index = load_index(baseline_index_path)
    existing_all_chunks = existing_index.get("chunks", [])
    existing_95_99 = [
        c for c in existing_all_chunks
        if isinstance(c, dict) and 95 <= c.get("page_number", 0) <= 99
    ]
    if len(generated_baseline_chunks) != len(existing_95_99):
        raise ValueError(
            f"Baseline chunk count mismatch: generated {len(generated_baseline_chunks)}, "
            f"existing index has {len(existing_95_99)} for pages 95..99"
        )
    for idx, (gen, orig) in enumerate(zip(generated_baseline_chunks, existing_95_99)):
        for fld in (
            "chunk_id",
            "doc_id",
            "source_file",
            "published_at",
            "page_number",
            "start_char",
            "end_char",
            "text",
            "strategy",
        ):
            if gen.get(fld) != orig.get(fld):
                raise ValueError(
                    f"Baseline chunk mismatch at chunk index {idx}, field {fld!r}: "
                    f"generated {gen.get(fld)!r} != existing {orig.get(fld)!r}"
                )


def verify_environment(transport: Any) -> None:
    """Verify actual server 0.40.11 and all 3 existing model digests."""
    if transport is None:
        raise RuntimeError("Transport cannot be None during environment verification")
    v_data = transport.get_version()
    if not isinstance(v_data, dict) or v_data.get("version") != SERVER_VERSION:
        raise RuntimeError(
            f"Server version mismatch: expected {SERVER_VERSION!r}, got {v_data.get('version')!r}"
        )
    tags_data = transport.get_tags()
    models = tags_data.get("models", []) if isinstance(tags_data, dict) else []
    digests_by_name: dict[str, str] = {}
    for m in models:
        if isinstance(m, dict):
            name = m.get("name") or m.get("model")
            digest = m.get("digest")
            if isinstance(name, str) and isinstance(digest, str):
                digests_by_name[name] = digest.strip()

    expected_models = {
        DEFAULT_GENERATION_MODEL: PINNED_9B_DIGEST,
        VERIFIER_Q3_MODEL: PINNED_27B_DIGEST,
        QWEN_EMBEDDING_MODEL: PINNED_EMBEDDING_DIGEST,
    }
    for m_name, exp_digest in expected_models.items():
        if m_name not in digests_by_name:
            raise RuntimeError(f"Required model {m_name!r} not found in Ollama /api/tags")
        if digests_by_name[m_name] != exp_digest:
            raise RuntimeError(
                f"Model digest mismatch for {m_name!r}: expected {exp_digest!r}, got {digests_by_name[m_name]!r}"
            )


def validate_protocol(protocol: dict[str, Any], *, protocol_dir: Path | None = None) -> None:
    """Strictly validate protocol schema, parameters, 34-attempt schedule, and source integrity."""
    if not isinstance(protocol, dict):
        raise ValueError(f"Protocol must be a dict, got {type(protocol).__name__}")

    # 1. Schema version
    if protocol.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"Protocol schema mismatch: expected {SCHEMA_VERSION!r}, got {protocol.get('schema_version')!r}"
        )

    # 2. Server version
    if protocol.get("server_version") != SERVER_VERSION:
        raise ValueError(
            f"Server version mismatch: expected {SERVER_VERSION!r}, got {protocol.get('server_version')!r}"
        )

    # 3. Models and digests
    models = protocol.get("models")
    if not isinstance(models, dict):
        raise ValueError("Protocol missing 'models' dict")
    expected_models = {
        "generation_model": DEFAULT_GENERATION_MODEL,
        "generation_digest": PINNED_9B_DIGEST,
        "verifier_model": VERIFIER_Q3_MODEL,
        "verifier_digest": PINNED_27B_DIGEST,
        "embedding_model": QWEN_EMBEDDING_MODEL,
        "embedding_digest": PINNED_EMBEDDING_DIGEST,
    }
    if set(models.keys()) != set(expected_models.keys()):
        raise ValueError(f"Protocol models keys mismatch: {sorted(models.keys())} != {sorted(expected_models.keys())}")
    for k, expected_v in expected_models.items():
        if models[k] != expected_v:
            raise ValueError(f"Protocol model {k} mismatch: expected {expected_v!r}, got {models[k]!r}")

    # 4. Parameters (strict type-aware JSON equality, forbid unknown keys/options)
    params = protocol.get("parameters")
    if not isinstance(params, dict) or set(params.keys()) != {"generation_parameters", "verifier_parameters"}:
        raise ValueError("Protocol parameters keys mismatch: expected {'generation_parameters', 'verifier_parameters'}")

    gen_params = params["generation_parameters"]
    if not isinstance(gen_params, dict) or set(gen_params.keys()) != set(DEFAULT_PARAMETERS.keys()):
        raise ValueError("Generation parameters keys mismatch against DEFAULT_PARAMETERS")
    for k, exp_val in DEFAULT_PARAMETERS.items():
        val = gen_params[k]
        if type(val) is not type(exp_val):
            raise ValueError(f"Generation parameter {k!r} type mismatch: expected {type(exp_val).__name__}, got {type(val).__name__}")
        if val != exp_val:
            raise ValueError(f"Generation parameter {k!r} value mismatch: expected {exp_val!r}, got {val!r}")

    ver_params = params["verifier_parameters"]
    if not isinstance(ver_params, dict) or set(ver_params.keys()) != set(VERIFIER_Q3_PARAMETERS.keys()):
        raise ValueError("Verifier parameters keys mismatch against VERIFIER_Q3_PARAMETERS")
    for k, exp_val in VERIFIER_Q3_PARAMETERS.items():
        val = ver_params[k]
        if type(val) is not type(exp_val):
            raise ValueError(f"Verifier parameter {k!r} type mismatch: expected {type(exp_val).__name__}, got {type(val).__name__}")
        if val != exp_val:
            raise ValueError(f"Verifier parameter {k!r} value mismatch: expected {exp_val!r}, got {val!r}")

    # 5. Execution options
    exec_options = protocol.get("execution_options")
    expected_options = {
        "top_k": 4,
        "max_packages": 1,
        "closed_note_tails": False,
        "partial_generation": True,
    }
    if not isinstance(exec_options, dict) or set(exec_options.keys()) != set(expected_options.keys()):
        raise ValueError(f"Execution options keys mismatch: {sorted(exec_options.keys()) if isinstance(exec_options, dict) else type(exec_options)}")
    for k, exp_val in expected_options.items():
        val = exec_options[k]
        if type(val) is not type(exp_val):
            raise ValueError(f"Execution option {k!r} type mismatch: expected {type(exp_val).__name__}, got {type(val).__name__}")
        if val != exp_val:
            raise ValueError(f"Execution option {k!r} value mismatch: expected {exp_val!r}, got {val!r}")

    # 6. Schedule check against build_schedule
    schedule = protocol.get("schedule")
    if not isinstance(schedule, list) or len(schedule) != 34:
        raise ValueError(f"Protocol schedule must contain exactly 34 attempts, got {len(schedule) if isinstance(schedule, list) else type(schedule)}")

    p_prim = PROJECT_ROOT / DEFAULT_PRIMARY_QUERIES_PATH
    p_sec = PROJECT_ROOT / DEFAULT_SECONDARY_QUERIES_PATH
    primary_queries = json.loads(p_prim.read_text(encoding="utf-8"))
    secondary_queries = json.loads(p_sec.read_text(encoding="utf-8"))
    expected_schedule = build_schedule(primary_queries, secondary_queries)
    if schedule != expected_schedule:
        raise ValueError("Protocol schedule does not match freshly computed schedule from frozen queries")

    # 7. Source freeze closure check
    frozen_sources = protocol.get("source_freeze")
    if not isinstance(frozen_sources, dict) or not frozen_sources:
        raise ValueError("Protocol missing or empty 'source_freeze'")
    current_closure = compute_source_closure()
    if set(frozen_sources.keys()) != set(current_closure.keys()):
        missing = set(current_closure.keys()) - set(frozen_sources.keys())
        extra = set(frozen_sources.keys()) - set(current_closure.keys())
        raise ValueError(f"Source closure keys mismatch: missing {sorted(missing)}, extra {sorted(extra)}")
    for rel_path, exp_sha in frozen_sources.items():
        curr_sha = current_closure[rel_path]
        if curr_sha != exp_sha:
            raise ValueError(f"Source code drift detected for {rel_path}: expected {exp_sha}, got {curr_sha}")

    # 8. Manifest and reference freeze hashes
    p_manifest = PROJECT_ROOT / DEFAULT_MANIFEST_PATH
    if not p_manifest.is_file():
        raise FileNotFoundError(f"Source artifact manifest not found: {p_manifest}")
    curr_manifest_sha = hashlib.sha256(p_manifest.read_bytes()).hexdigest()
    if protocol.get("source_artifact_manifest_sha256") != curr_manifest_sha:
        raise ValueError(
            f"Source artifact manifest SHA mismatch: expected {protocol.get('source_artifact_manifest_sha256')}, got {curr_manifest_sha}"
        )

    p_ref_freeze = PROJECT_ROOT / DEFAULT_REFERENCE_FREEZE_PATH
    if not p_ref_freeze.is_file():
        raise FileNotFoundError(f"Reference freeze file not found: {p_ref_freeze}")
    curr_ref_freeze_sha = hashlib.sha256(p_ref_freeze.read_bytes()).hexdigest()
    if protocol.get("reference_freeze_sha256") != curr_ref_freeze_sha:
        raise ValueError(
            f"Reference freeze SHA mismatch: expected {protocol.get('reference_freeze_sha256')}, got {curr_ref_freeze_sha}"
        )

    # 9. Complete required input file hashes check
    input_hashes = protocol.get("input_file_hashes")
    if not isinstance(input_hashes, dict):
        raise ValueError("Protocol missing 'input_file_hashes' dict")

    expected_keys = set(Path(p).as_posix() for p in REQUIRED_INPUT_PATHS)
    raw_keys = list(input_hashes.keys())

    # Raw entry count must be exactly 9
    if len(raw_keys) != len(REQUIRED_INPUT_PATHS):
        missing = expected_keys - set(raw_keys)
        extra = set(raw_keys) - expected_keys
        raise ValueError(
            f"input_file_hashes keyset mismatch: missing {sorted(missing)}, extra {sorted(extra)} "
            f"(raw entry count must be exactly {len(REQUIRED_INPUT_PATHS)}, got {len(raw_keys)})"
        )

    # Detect alias duplicates before collapsing
    seen_canonical_keys: set[str] = set()
    for k in raw_keys:
        if not isinstance(k, str) or not k.strip():
            raise ValueError(f"Invalid input_file_hashes key: {k!r}")

        pk = Path(k)
        if pk.is_absolute():
            try:
                rel_k = pk.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
            except ValueError:
                rel_k = pk.as_posix()
        else:
            rel_k = (PROJECT_ROOT / pk).resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()

        if rel_k in seen_canonical_keys:
            raise ValueError(f"Alias duplicate detected in input_file_hashes: {k!r} resolves to already seen {rel_k!r}")
        seen_canonical_keys.add(rel_k)

    if seen_canonical_keys != expected_keys:
        missing = expected_keys - seen_canonical_keys
        extra = seen_canonical_keys - expected_keys
        raise ValueError(f"input_file_hashes keyset mismatch: missing {sorted(missing)}, extra {sorted(extra)}")

    for k in raw_keys:
        if k not in expected_keys:
            raise ValueError(f"input_file_hashes raw key is not canonical relative path: {k!r}")

    for file_path_str, exp_hash in input_hashes.items():
        fp = PROJECT_ROOT / file_path_str if not Path(file_path_str).is_absolute() else Path(file_path_str)
        if not fp.is_file():
            raise FileNotFoundError(f"Input file not found: {fp}")
        actual_hash = hashlib.sha256(fp.read_bytes()).hexdigest()
        if actual_hash != exp_hash:
            raise ValueError(f"Input file drift for {file_path_str}: expected {exp_hash}, got {actual_hash}")

    # 10. Validate catalog/index/pages assets and anti-arm-swap if protocol_dir is provided
    if protocol_dir is not None:
        pdir = Path(protocol_dir)
        gen_assets = protocol.get("generated_assets")
        if not isinstance(gen_assets, dict):
            raise ValueError("Protocol missing 'generated_assets' map")
        expected_asset_files = {
            "baseline_pages.jsonl",
            "layout_pages.jsonl",
            "baseline_index.json",
            "layout_index.json",
            "baseline_primary_catalog.json",
            "layout_primary_catalog.json",
            "baseline_table_catalog.json",
            "layout_table_catalog.json",
            "primary_queries.json",
            "secondary_queries.json",
            "reanchor_audit.json",
        }
        if set(gen_assets.keys()) != expected_asset_files:
            missing_assets = expected_asset_files - set(gen_assets.keys())
            extra_assets = set(gen_assets.keys()) - expected_asset_files
            raise ValueError(f"generated_assets keyset mismatch: missing {sorted(missing_assets)}, extra {sorted(extra_assets)}")

        for af, hashes in gen_assets.items():
            p_af = Path(af)
            if p_af.is_absolute() or len(p_af.parts) != 1 or ".." in p_af.parts:
                raise ValueError(f"Generated asset path {af!r} must be a simple workspace-contained filename")
            af_path = pdir / af
            if not af_path.is_file():
                raise FileNotFoundError(f"Generated asset file not found: {af_path}")
            af_bytes = af_path.read_bytes()
            curr_raw_sha = hashlib.sha256(af_bytes).hexdigest()
            if curr_raw_sha != hashes.get("raw_sha256"):
                raise ValueError(f"Asset raw_sha256 mismatch for {af}: expected {hashes.get('raw_sha256')}, got {curr_raw_sha}")
            if af.endswith(".jsonl"):
                lines_obj = [json.loads(line) for line in af_bytes.decode("utf-8").splitlines() if line.strip()]
                curr_canon_bytes = "\n".join(json.dumps(l, sort_keys=True, ensure_ascii=False) for l in lines_obj).encode("utf-8")
            else:
                obj = json.loads(af_bytes.decode("utf-8"))
                curr_canon_bytes = json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")
            curr_canon_sha = hashlib.sha256(curr_canon_bytes).hexdigest()
            if curr_canon_sha != hashes.get("canonical_sha256"):
                raise ValueError(f"Asset canonical_sha256 mismatch for {af}: expected {hashes.get('canonical_sha256')}, got {curr_canon_sha}")

        both_arms = protocol.get("both_arms")
        if not isinstance(both_arms, dict) or set(both_arms.keys()) != {"baseline", "layout"}:
            raise ValueError("both_arms must contain exactly 'baseline' and 'layout'")

        for arm, expected_roles in CLOSED_ARM_ROLE_FILES.items():
            arm_data = both_arms.get(arm)
            if not isinstance(arm_data, dict):
                raise ValueError(f"Arm {arm!r} must be a dict")
            for role, expected_fn in expected_roles.items():
                actual_fn = arm_data.get(role)
                if actual_fn != expected_fn:
                    raise ValueError(
                        f"Arm {arm!r} role {role!r} must be exact expected canonical filename {expected_fn!r}, "
                        f"got {actual_fn!r}; arm swap or corruption detected"
                    )

        legacy_raw = [
            json.loads(line) for line in (PROJECT_ROOT / DEFAULT_LEGACY_PAGES_PATH).read_text(encoding="utf-8").splitlines() if line.strip()
        ]
        layout_struct = json.loads((PROJECT_ROOT / DEFAULT_LAYOUT_BUNDLE_PATH).read_text(encoding="utf-8"))
        exp_b_pages, exp_l_pages = filter_and_validate_pages(legacy_raw, layout_struct)

        # 1. Baseline pages must match exp_b_pages
        b_pages_actual = [
            json.loads(line) for line in (pdir / CLOSED_ARM_ROLE_FILES["baseline"]["pages_file"]).read_text(encoding="utf-8").splitlines() if line.strip()
        ]
        exp_b_records = [
            {
                "chunk_id": f"page_{p['page_number']}",
                "doc_id": p["doc_id"],
                "source_file": p["source_file"],
                "published_at": p["published_at"],
                "page_number": p["page_number"],
                "start_char": 0,
                "end_char": len(p["text"]),
                "text": p["text"],
                "strategy": "raw_page",
            }
            for p in exp_b_pages
        ]
        if not _json_type_equal(b_pages_actual, exp_b_records):
            raise ValueError("Baseline pages do not match expected legacy extraction; arm swap or corruption detected")

        # 2. Layout pages must match exp_l_pages
        l_pages_actual = [
            json.loads(line) for line in (pdir / CLOSED_ARM_ROLE_FILES["layout"]["pages_file"]).read_text(encoding="utf-8").splitlines() if line.strip()
        ]
        exp_l_records = [
            {
                "chunk_id": f"page_{p['page_number']}",
                "doc_id": p["doc_id"],
                "source_file": p["source_file"],
                "published_at": p["published_at"],
                "page_number": p["page_number"],
                "start_char": 0,
                "end_char": len(p["text"]),
                "text": p["text"],
                "strategy": "raw_page",
            }
            for p in exp_l_pages
        ]
        if not _json_type_equal(l_pages_actual, exp_l_records):
            raise ValueError("Layout pages do not match expected layout extraction; arm swap or corruption detected")

        # 3. Deterministic source-derived unit list equality with original catalog (baseline) and reanchor_catalog (layout)
        orig_cat = json.loads((PROJECT_ROOT / DEFAULT_CATALOG_PATH).read_text(encoding="utf-8"))
        exp_prim_unit_count = len(orig_cat.get("units", []))
        exp_prim_span_count = sum(len(u.get("spans", [])) for u in orig_cat.get("units", []))
        if exp_prim_unit_count != 13 or exp_prim_span_count != 16:
            raise ValueError(f"Frozen original catalog does not contain real 13 units / 16 spans: got {exp_prim_unit_count} units, {exp_prim_span_count} spans")
        if any(u.get("id") == "T95" for u in orig_cat.get("units", [])):
            raise ValueError("Original catalog contains forbidden T95 unit")

        exp_l_pcat, _ = reanchor_catalog(orig_cat, exp_b_pages, exp_l_pages)
        if len(exp_l_pcat.get("units", [])) != 13 or sum(len(u.get("spans", [])) for u in exp_l_pcat.get("units", [])) != 16:
            raise ValueError("Reanchored layout primary catalog count mismatch against frozen original")
        if any(u.get("id") == "T95" for u in exp_l_pcat.get("units", [])):
            raise ValueError("Reanchored layout primary catalog contains forbidden T95 unit")

        p95_struct = next(p for p in layout_struct["pages"] if p["page_number"] == 95)
        exp_t95_base, exp_t95_lay = extract_t95_units(exp_b_pages[0], exp_l_pages[0], p95_struct)
        if exp_t95_base.get("id") != "T95" or exp_t95_lay.get("id") != "T95":
            raise ValueError("Extracted table unit missing T95 identity")

        exp_b_tunits = copy.deepcopy(orig_cat["units"]) + [exp_t95_base]
        exp_l_tunits = copy.deepcopy(exp_l_pcat["units"]) + [exp_t95_lay]

        # Baseline primary catalog file
        b_pcat_actual = json.loads((pdir / CLOSED_ARM_ROLE_FILES["baseline"]["primary_catalog_file"]).read_text(encoding="utf-8"))
        if not _json_type_equal(b_pcat_actual.get("units"), orig_cat.get("units")):
            raise ValueError("Baseline primary catalog units do not match frozen original catalog units; arm swap or corruption detected")
        if any(u.get("id") == "T95" for u in b_pcat_actual.get("units", [])):
            raise ValueError("Baseline primary catalog contains forbidden T95 unit")
        if len(b_pcat_actual.get("units", [])) != 13 or sum(len(u.get("spans", [])) for u in b_pcat_actual.get("units", [])) != 16:
            raise ValueError("Baseline primary catalog unit/span count mismatch")

        # Layout primary catalog file
        l_pcat_actual = json.loads((pdir / CLOSED_ARM_ROLE_FILES["layout"]["primary_catalog_file"]).read_text(encoding="utf-8"))
        if not _json_type_equal(l_pcat_actual.get("units"), exp_l_pcat.get("units")):
            raise ValueError("Layout primary catalog units do not match expected reanchored catalog units; arm swap or corruption detected")
        if any(u.get("id") == "T95" for u in l_pcat_actual.get("units", [])):
            raise ValueError("Layout primary catalog contains forbidden T95 unit")
        if len(l_pcat_actual.get("units", [])) != 13 or sum(len(u.get("spans", [])) for u in l_pcat_actual.get("units", [])) != 16:
            raise ValueError("Layout primary catalog unit/span count mismatch")

        # Baseline table catalog file
        b_tcat_actual = json.loads((pdir / CLOSED_ARM_ROLE_FILES["baseline"]["table_catalog_file"]).read_text(encoding="utf-8"))
        if not _json_type_equal(b_tcat_actual.get("units"), exp_b_tunits):
            raise ValueError("Baseline table catalog units do not match expected units (primary + T95); arm swap or corruption detected")
        if len(b_tcat_actual.get("units", [])) != 14 or sum(len(u.get("spans", [])) for u in b_tcat_actual.get("units", [])) != 17:
            raise ValueError("Baseline table catalog unit/span count mismatch")
        if not any(u.get("id") == "T95" for u in b_tcat_actual.get("units", [])):
            raise ValueError("Baseline table catalog missing T95 unit")

        # Layout table catalog file
        l_tcat_actual = json.loads((pdir / CLOSED_ARM_ROLE_FILES["layout"]["table_catalog_file"]).read_text(encoding="utf-8"))
        if not _json_type_equal(l_tcat_actual.get("units"), exp_l_tunits):
            raise ValueError("Layout table catalog units do not match expected units (reanchored primary + T95); arm swap or corruption detected")
        if len(l_tcat_actual.get("units", [])) != 14 or sum(len(u.get("spans", [])) for u in l_tcat_actual.get("units", [])) != 17:
            raise ValueError("Layout table catalog unit/span count mismatch")
        if not any(u.get("id") == "T95" for u in l_tcat_actual.get("units", [])):
            raise ValueError("Layout table catalog missing T95 unit")

        # 5. Index chunk descriptors check
        from rag_pensiones.ingest import PageRecord
        b_chunks_expected = chunk_pages([PageRecord(**p) for p in exp_b_pages], 1000, 150)
        l_chunks_expected = chunk_pages([PageRecord(**p) for p in exp_l_pages], 1000, 150)

        b_idx_actual = json.loads((pdir / CLOSED_ARM_ROLE_FILES["baseline"]["index_file"]).read_text(encoding="utf-8"))
        l_idx_actual = json.loads((pdir / CLOSED_ARM_ROLE_FILES["layout"]["index_file"]).read_text(encoding="utf-8"))

        if len(b_idx_actual["chunks"]) != len(b_chunks_expected):
            raise ValueError("Baseline index chunk count mismatch")
        for act_c, exp_c in zip(b_idx_actual["chunks"], b_chunks_expected):
            for k in ("chunk_id", "doc_id", "source_file", "published_at", "page_number", "start_char", "end_char", "text", "strategy"):
                if act_c.get(k) != getattr(exp_c, k):
                    raise ValueError(f"Baseline index chunk descriptor {k} mismatch against recipe")

        if len(l_idx_actual["chunks"]) != len(l_chunks_expected):
            raise ValueError("Layout index chunk count mismatch")
        for act_c, exp_c in zip(l_idx_actual["chunks"], l_chunks_expected):
            for k in ("chunk_id", "doc_id", "source_file", "published_at", "page_number", "start_char", "end_char", "text", "strategy"):
                if act_c.get(k) != getattr(exp_c, k):
                    raise ValueError(f"Layout index chunk descriptor {k} mismatch against recipe")

        # 6. Validate literal quotes in all catalogs
        validate_catalog(b_pcat_actual, exp_b_pages)
        validate_catalog(b_tcat_actual, exp_b_pages)
        validate_catalog(l_pcat_actual, exp_l_pages)
        validate_catalog(l_tcat_actual, exp_l_pages)


def prepare(
    output_dir: Path | str,
    *,
    legacy_pages_path: Path | str = DEFAULT_LEGACY_PAGES_PATH,
    layout_bundle_path: Path | str = DEFAULT_LAYOUT_BUNDLE_PATH,
    catalog_path: Path | str = DEFAULT_CATALOG_PATH,
    primary_queries_path: Path | str = DEFAULT_PRIMARY_QUERIES_PATH,
    secondary_queries_path: Path | str = DEFAULT_SECONDARY_QUERIES_PATH,
    baseline_index_path: Path | str = DEFAULT_BASELINE_INDEX_PATH,
    embedder: Any | None = None,
    transport: Any | None = None,
) -> dict[str, Any]:
    """Prepare frozen comparison protocol and assets into exclusive output directory."""
    out_p = Path(output_dir)
    if not out_p.is_absolute():
        out_p = (PROJECT_ROOT / out_p).resolve()
    else:
        out_p = out_p.resolve()

    validate_project_path(out_p)
    if out_p.exists() and any(out_p.iterdir()):
        raise FileExistsError(f"Target output directory already exists and is not empty: {out_p}")

    p_legacy = (PROJECT_ROOT / legacy_pages_path).resolve()
    p_layout = (PROJECT_ROOT / layout_bundle_path).resolve()
    p_catalog = (PROJECT_ROOT / catalog_path).resolve()
    p_primary = (PROJECT_ROOT / primary_queries_path).resolve()
    p_secondary = (PROJECT_ROOT / secondary_queries_path).resolve()
    p_index = (PROJECT_ROOT / baseline_index_path).resolve()
    p_ref_freeze = (PROJECT_ROOT / DEFAULT_REFERENCE_FREEZE_PATH).resolve()
    p_manifest = (PROJECT_ROOT / DEFAULT_MANIFEST_PATH).resolve()
    p_defaults = (PROJECT_ROOT / DEFAULT_OLLAMA_DEFAULTS_REL).resolve()

    required_inputs = [
        p_legacy,
        p_layout,
        p_catalog,
        p_primary,
        p_secondary,
        p_index,
        p_ref_freeze,
        p_manifest,
        p_defaults,
    ]
    for p in required_inputs:
        validate_project_path(p)
        if not p.is_file():
            raise FileNotFoundError(f"Required input file not found: {p}")

    hashes_before = {p.relative_to(PROJECT_ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in required_inputs}

    # Environment verification: ALWAYS verify live server and model pins before any embeddings
    if transport is None:
        transport = MixedSingle27Transport(timeout=240.0)
    verify_environment(transport)

    if embedder is None:
        from experiments.compare_mixed_single27 import ComparisonEmbedder
        embedder = ComparisonEmbedder(transport=transport)

    legacy_pages_raw = [
        json.loads(line, parse_constant=_reject_constant, parse_float=_parse_float)
        for line in p_legacy.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    layout_structured_data = json.loads(
        p_layout.read_text(encoding="utf-8"),
        parse_constant=_reject_constant,
        parse_float=_parse_float,
    )
    catalog_data = json.loads(
        p_catalog.read_text(encoding="utf-8"),
        parse_constant=_reject_constant,
        parse_float=_parse_float,
    )
    primary_queries_data = json.loads(
        p_primary.read_text(encoding="utf-8"),
        parse_constant=_reject_constant,
        parse_float=_parse_float,
    )
    secondary_queries_data = json.loads(
        p_secondary.read_text(encoding="utf-8"),
        parse_constant=_reject_constant,
        parse_float=_parse_float,
    )

    baseline_pages, layout_pages = filter_and_validate_pages(legacy_pages_raw, layout_structured_data)

    baseline_primary_catalog = copy.deepcopy(catalog_data)
    validate_catalog(baseline_primary_catalog, baseline_pages)

    layout_primary_catalog, reanchor_audit = reanchor_catalog(
        baseline_primary_catalog,
        baseline_pages,
        layout_pages,
    )
    validate_catalog(layout_primary_catalog, layout_pages)

    pilot_p95 = next(p for p in layout_structured_data["pages"] if p["page_number"] == 95)
    t95_baseline, t95_layout = extract_t95_units(baseline_pages[0], layout_pages[0], pilot_p95)

    baseline_table_catalog = copy.deepcopy(baseline_primary_catalog)
    baseline_table_catalog["units"].append(t95_baseline)
    validate_catalog(baseline_table_catalog, baseline_pages)

    layout_table_catalog = copy.deepcopy(layout_primary_catalog)
    layout_table_catalog["units"].append(t95_layout)
    validate_catalog(layout_table_catalog, layout_pages)

    schedule = build_schedule(primary_queries_data, secondary_queries_data)

    baseline_index = build_index(baseline_pages, embedder)
    layout_index = build_index(layout_pages, embedder)

    verify_baseline_chunks_against_existing(baseline_index["chunks"], p_index)

    hashes_after = {p.relative_to(PROJECT_ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in required_inputs}
    if hashes_before != hashes_after:
        raise RuntimeError("Input files were modified during prepare execution!")

    out_p.mkdir(parents=True, exist_ok=True)

    def _write_json(name: str, data: Any) -> None:
        (out_p / name).write_text(
            json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )

    from rag_pensiones.ingest import PageRecord
    write_chunks_jsonl([ChunkRecord(chunk_id=f"page_{p['page_number']}", doc_id=p["doc_id"], source_file=p["source_file"], published_at=p["published_at"], page_number=p["page_number"], start_char=0, end_char=len(p["text"]), text=p["text"], strategy="raw_page") for p in baseline_pages], out_p / "baseline_pages.jsonl")
    write_chunks_jsonl([ChunkRecord(chunk_id=f"page_{p['page_number']}", doc_id=p["doc_id"], source_file=p["source_file"], published_at=p["published_at"], page_number=p["page_number"], start_char=0, end_char=len(p["text"]), text=p["text"], strategy="raw_page") for p in layout_pages], out_p / "layout_pages.jsonl")

    _write_json("baseline_index.json", baseline_index)
    _write_json("layout_index.json", layout_index)
    _write_json("baseline_primary_catalog.json", baseline_primary_catalog)
    _write_json("layout_primary_catalog.json", layout_primary_catalog)
    _write_json("baseline_table_catalog.json", baseline_table_catalog)
    _write_json("layout_table_catalog.json", layout_table_catalog)
    _write_json("primary_queries.json", primary_queries_data)
    _write_json("secondary_queries.json", secondary_queries_data)
    _write_json("reanchor_audit.json", reanchor_audit)

    # Compute raw & canonical SHA for all generated asset files
    generated_assets: dict[str, dict[str, str]] = {}
    asset_files = [
        "baseline_pages.jsonl",
        "layout_pages.jsonl",
        "baseline_index.json",
        "layout_index.json",
        "baseline_primary_catalog.json",
        "layout_primary_catalog.json",
        "baseline_table_catalog.json",
        "layout_table_catalog.json",
        "primary_queries.json",
        "secondary_queries.json",
        "reanchor_audit.json",
    ]
    for af in asset_files:
        af_path = out_p / af
        af_bytes = af_path.read_bytes()
        raw_sha = hashlib.sha256(af_bytes).hexdigest()
        if af.endswith(".jsonl"):
            lines_obj = [json.loads(line) for line in af_bytes.decode("utf-8").splitlines() if line.strip()]
            canon_bytes = "\n".join(json.dumps(l, sort_keys=True, ensure_ascii=False) for l in lines_obj).encode("utf-8")
        else:
            obj = json.loads(af_bytes.decode("utf-8"))
            canon_bytes = json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")
        canon_sha = hashlib.sha256(canon_bytes).hexdigest()
        generated_assets[af] = {
            "raw_sha256": raw_sha,
            "canonical_sha256": canon_sha,
        }

    source_closure = compute_source_closure()
    manifest_sha = hashlib.sha256(p_manifest.read_bytes()).hexdigest()
    ref_freeze_sha = hashlib.sha256(p_ref_freeze.read_bytes()).hexdigest()

    protocol = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "server_version": SERVER_VERSION,
        "models": {
            "generation_model": DEFAULT_GENERATION_MODEL,
            "generation_digest": PINNED_9B_DIGEST,
            "verifier_model": VERIFIER_Q3_MODEL,
            "verifier_digest": PINNED_27B_DIGEST,
            "embedding_model": QWEN_EMBEDDING_MODEL,
            "embedding_digest": PINNED_EMBEDDING_DIGEST,
        },
        "parameters": {
            "generation_parameters": copy.deepcopy(DEFAULT_PARAMETERS),
            "verifier_parameters": copy.deepcopy(VERIFIER_Q3_PARAMETERS),
        },
        "execution_options": {
            "top_k": 4,
            "max_packages": 1,
            "closed_note_tails": False,
            "partial_generation": True,
        },
        "both_arms": {
            "baseline": {
                "pages_file": "baseline_pages.jsonl",
                "index_file": "baseline_index.json",
                "primary_catalog_file": "baseline_primary_catalog.json",
                "table_catalog_file": "baseline_table_catalog.json",
                "raw_sha256": hashlib.sha256("".join(p["text"] for p in baseline_pages).encode("utf-8")).hexdigest(),
                "canonical_sha256": generated_assets["baseline_pages.jsonl"]["canonical_sha256"],
                "pages_count": len(baseline_pages),
                "chunks_count": len(baseline_index["chunks"]),
                "primary_units_count": len(baseline_primary_catalog["units"]),
                "table_units_count": len(baseline_table_catalog["units"]),
            },
            "layout": {
                "pages_file": "layout_pages.jsonl",
                "index_file": "layout_index.json",
                "primary_catalog_file": "layout_primary_catalog.json",
                "table_catalog_file": "layout_table_catalog.json",
                "raw_sha256": hashlib.sha256("".join(p["text"] for p in layout_pages).encode("utf-8")).hexdigest(),
                "canonical_sha256": generated_assets["layout_pages.jsonl"]["canonical_sha256"],
                "pages_count": len(layout_pages),
                "chunks_count": len(layout_index["chunks"]),
                "primary_units_count": len(layout_primary_catalog["units"]),
                "table_units_count": len(layout_table_catalog["units"]),
            },
        },
        "generated_assets": generated_assets,
        "schedule": schedule,
        "counts": {
            "pages_count": len(baseline_pages),
            "primary_queries_count": len(primary_queries_data.get("cases", [])),
            "secondary_queries_count": len(secondary_queries_data.get("cases", [])),
            "total_queries_count": 17,
            "attempts_count": len(schedule),
            "primary_units_count": len(baseline_primary_catalog["units"]),
            "table_units_count": len(baseline_table_catalog["units"]),
            "baseline_chunks_count": len(baseline_index["chunks"]),
            "layout_chunks_count": len(layout_index["chunks"]),
        },
        "source_freeze": source_closure,
        "source_artifact_manifest_sha256": manifest_sha,
        "reference_freeze_sha256": ref_freeze_sha,
        "input_file_hashes": hashes_after,
        "reanchor_audit": {
            "original_unit_count": reanchor_audit["original_unit_count"],
            "new_unit_count": reanchor_audit["new_unit_count"],
            "original_span_count": reanchor_audit["original_span_count"],
            "new_span_count": reanchor_audit["new_span_count"],
            "normalization_rule": reanchor_audit["normalization_rule"],
        },
    }

    _write_json("protocol.json", protocol)
    return protocol


def run(
    protocol_path: Path | str,
    output_path: Path | str,
    *,
    transport: Any | None = None,
    execute_fn: Callable | None = None,
) -> dict[str, Any]:
    """Execute the 34-attempt evaluation matrix with strict pre- and post-attempt drift guards."""
    p_proto = Path(protocol_path)
    if not p_proto.is_absolute():
        p_proto = (PROJECT_ROOT / p_proto).resolve()
    else:
        p_proto = p_proto.resolve()

    p_out = Path(output_path)
    if not p_out.is_absolute():
        p_out = (PROJECT_ROOT / p_out).resolve()
    else:
        p_out = p_out.resolve()

    validate_project_path(p_proto)
    validate_project_path(p_out)

    if not p_proto.is_file():
        raise FileNotFoundError(f"Protocol file not found: {p_proto}")
    if p_out.exists():
        raise FileExistsError(f"Target run output destination already exists: {p_out}")

    original_protocol_bytes = p_proto.read_bytes()
    original_protocol_raw_sha = hashlib.sha256(original_protocol_bytes).hexdigest()
    protocol = json.loads(original_protocol_bytes.decode("utf-8"), parse_constant=_reject_constant, parse_float=_parse_float)

    # Helper to check protocol file on disk bytes
    def verify_protocol_on_disk(stage_label: str) -> None:
        if not p_proto.is_file():
            raise RuntimeError(f"Protocol file was deleted during {stage_label}: {p_proto}")
        current_bytes = p_proto.read_bytes()
        current_sha = hashlib.sha256(current_bytes).hexdigest()
        if current_sha != original_protocol_raw_sha:
            raise RuntimeError(
                f"Protocol file on disk was modified during {stage_label}: "
                f"expected {original_protocol_raw_sha}, got {current_sha}"
            )

    # Validate protocol completely before any attempt
    verify_protocol_on_disk("pre-run")
    validate_protocol(protocol, protocol_dir=p_proto.parent)

    if transport is None:
        transport = MixedSingle27Transport(timeout=240.0)

    verify_environment(transport)

    proto_dir = p_proto.parent
    both_arms = protocol["both_arms"]

    b_pages = [
        json.loads(line)
        for line in (proto_dir / both_arms["baseline"]["pages_file"]).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    b_index = json.loads((proto_dir / both_arms["baseline"]["index_file"]).read_text(encoding="utf-8"))
    b_pcat = json.loads((proto_dir / both_arms["baseline"]["primary_catalog_file"]).read_text(encoding="utf-8"))
    b_tcat = json.loads((proto_dir / both_arms["baseline"]["table_catalog_file"]).read_text(encoding="utf-8"))

    l_pages = [
        json.loads(line)
        for line in (proto_dir / both_arms["layout"]["pages_file"]).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    l_index = json.loads((proto_dir / both_arms["layout"]["index_file"]).read_text(encoding="utf-8"))
    l_pcat = json.loads((proto_dir / both_arms["layout"]["primary_catalog_file"]).read_text(encoding="utf-8"))
    l_tcat = json.loads((proto_dir / both_arms["layout"]["table_catalog_file"]).read_text(encoding="utf-8"))

    if execute_fn is None:
        execute_fn = default_execute_attempt

    schedule = protocol["schedule"]
    results: dict[str, Any] = {
        "schema_version": RESULTS_SCHEMA_VERSION,
        "protocol_path": str(p_proto),
        "protocol_raw_sha256": original_protocol_raw_sha,
        "run_status": "running",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "finished_at_utc": None,
        "expected_count": 34,
        "attempts": [],
        "summary": {
            "total_scheduled": 34,
            "completed_attempts": 0,
            "error_attempts": 0,
            "total_wall_seconds": 0.0,
        },
    }

    total_wall_seconds = 0.0
    completed_count = 0
    error_count = 0

    p_out.parent.mkdir(parents=True, exist_ok=True)

    for attempt in schedule:
        # Pre-attempt drift check on disk bytes and full protocol
        try:
            verify_protocol_on_disk(f"pre-attempt {attempt['attempt_index']}")
            validate_protocol(protocol, protocol_dir=p_proto.parent)
        except Exception as exc:
            results["run_status"] = "invalid"
            results["error"] = f"Pre-attempt integrity drift at attempt {attempt['attempt_index']}: {exc}"
            results["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
            checkpoint_atomic(results, p_out)
            raise RuntimeError(results["error"]) from exc

        cond = attempt["condition"]
        cohort = attempt["cohort"]

        if cond == "legacy":
            pages_data = b_pages
            index_data = b_index
            units_data = b_pcat if cohort == "primary" else b_tcat
            cat_file = both_arms["baseline"]["primary_catalog_file"] if cohort == "primary" else both_arms["baseline"]["table_catalog_file"]
        elif cond == "layout":
            pages_data = l_pages
            index_data = l_index
            units_data = l_pcat if cohort == "primary" else l_tcat
            cat_file = both_arms["layout"]["primary_catalog_file"] if cohort == "primary" else both_arms["layout"]["table_catalog_file"]
        else:
            raise ValueError(f"Unknown condition {cond!r}")

        input_hashes = {
            "pages_sha256": protocol["both_arms"]["baseline" if cond == "legacy" else "layout"]["raw_sha256"],
            "index_sha256": protocol["generated_assets"][both_arms["baseline" if cond == "legacy" else "layout"]["index_file"]]["canonical_sha256"],
            "catalog_sha256": protocol["generated_assets"][cat_file]["canonical_sha256"],
        }

        attempt_spec = {
            "attempt_index": attempt["attempt_index"],
            "case_id": attempt["case_id"],
            "query": attempt["query"],
            "arm": "mixed",
        }

        attempt_status = "completed"
        attempt_error = None
        raw_attempt: dict[str, Any] = {}

        try:
            raw_attempt = execute_fn(
                attempt_spec,
                index_data=index_data,
                units_data=units_data,
                pages_data=pages_data,
                transport=transport,
            )
            if not isinstance(raw_attempt, dict) or raw_attempt.get("status") == "error":
                attempt_status = "error"
                attempt_error = str(raw_attempt.get("error", "Factual grounding or coverage error")) if isinstance(raw_attempt, dict) else "Invalid attempt result"
                error_count += 1
            else:
                completed_count += 1
        except Exception as exc:
            attempt_status = "error"
            attempt_error = str(exc)
            error_count += 1
            raw_attempt = {
                "synthetic_exception_record": True,
                "status": "error",
                "error": str(exc),
                "arm": "mixed",
                "case_id": attempt["case_id"],
                "query": attempt["query"],
                "attempt_index": attempt["attempt_index"],
                "formatted_answer": "",
                "coverage_publication": None,
                "factual_outcome": None,
                "lifecycle_records": [],
                "stats_summary": {},
                "query_wall_seconds": 0.0,
            }

        query_wall = raw_attempt.get("query_wall_seconds", 0.0) if isinstance(raw_attempt, dict) else 0.0
        total_wall_seconds += query_wall

        raw_canonical_sha = hashlib.sha256(
            json.dumps(raw_attempt, sort_keys=True, ensure_ascii=False, allow_nan=False).encode("utf-8")
        ).hexdigest()

        attempt_record = {
            "attempt_index": attempt["attempt_index"],
            "question_index": attempt["question_index"],
            "cohort": cohort,
            "case_id": attempt["case_id"],
            "query": attempt["query"],
            "condition": cond,
            "arm": "mixed",
            "status": attempt_status,
            "error": attempt_error,
            "formatted_answer": raw_attempt.get("formatted_answer", "") if isinstance(raw_attempt, dict) else "",
            "coverage_publication": raw_attempt.get("coverage_publication") if isinstance(raw_attempt, dict) else None,
            "factual_outcome": raw_attempt.get("factual_outcome") if isinstance(raw_attempt, dict) else None,
            "lifecycle_records": raw_attempt.get("lifecycle_records", []) if isinstance(raw_attempt, dict) else [],
            "stats_summary": raw_attempt.get("stats_summary", {}) if isinstance(raw_attempt, dict) else {},
            "query_wall_seconds": query_wall,
            "input_hashes": input_hashes,
            "raw_attempt_canonical_sha256": raw_canonical_sha,
            "raw_attempt": raw_attempt,
        }
        results["attempts"].append(attempt_record)
        results["summary"]["completed_attempts"] = completed_count
        results["summary"]["error_attempts"] = error_count
        results["summary"]["total_wall_seconds"] = total_wall_seconds

        checkpoint_atomic(results, p_out)

        # Post-attempt drift check on disk bytes and full protocol
        try:
            verify_protocol_on_disk(f"post-attempt {attempt['attempt_index']}")
            validate_protocol(protocol, protocol_dir=p_proto.parent)
        except Exception as exc:
            results["run_status"] = "invalid"
            results["error"] = f"Post-attempt integrity drift at attempt {attempt['attempt_index']}: {exc}"
            results["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
            checkpoint_atomic(results, p_out)
            raise RuntimeError(results["error"]) from exc

    verify_protocol_on_disk("finalization")
    validate_protocol(protocol, protocol_dir=p_proto.parent)

    results["run_status"] = "completed"
    results["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    checkpoint_atomic(results, p_out)
    return results


def export_review(
    protocol_path: Path | str,
    results_path: Path | str,
    output_path: Path | str,
) -> dict[str, Any]:
    """Export neutral blinded A/B review bundle and sibling review_key.json."""
    p_proto = Path(protocol_path)
    if not p_proto.is_absolute():
        p_proto = (PROJECT_ROOT / p_proto).resolve()
    else:
        p_proto = p_proto.resolve()

    p_res = Path(results_path)
    if not p_res.is_absolute():
        p_res = (PROJECT_ROOT / p_res).resolve()
    else:
        p_res = p_res.resolve()

    p_out = Path(output_path)
    if not p_out.is_absolute():
        p_out = (PROJECT_ROOT / p_out).resolve()
    else:
        p_out = p_out.resolve()

    key_path = p_out.parent / "review_key.json"

    validate_project_path(p_proto)
    validate_project_path(p_res)
    validate_project_path(p_out)
    validate_project_path(key_path)

    if not p_proto.is_file():
        raise FileNotFoundError(f"Protocol file not found: {p_proto}")
    if not p_res.is_file():
        raise FileNotFoundError(f"Results file not found: {p_res}")

    protocol = json.loads(p_proto.read_text(encoding="utf-8"))
    results = json.loads(p_res.read_text(encoding="utf-8"))

    proto_sha = hashlib.sha256(p_proto.read_bytes()).hexdigest()
    if results.get("protocol_raw_sha256") != proto_sha:
        raise ValueError("Results protocol_raw_sha256 does not match protocol file hash")

    if results.get("run_status") != "completed":
        raise ValueError(f"Cannot export review for run with status {results.get('run_status')!r}; must be 'completed'")

    attempts = results.get("attempts", [])
    if len(attempts) != 34:
        raise ValueError(f"Expected exactly 34 attempts in results, got {len(attempts)}")

    schedule = protocol.get("schedule", [])
    if len(schedule) != 34:
        raise ValueError(f"Protocol schedule must contain 34 attempts, got {len(schedule)}")

    seen_tuple_keys: set[tuple[str, str, str]] = set()

    for idx in range(34):
        att = attempts[idx]
        sched = schedule[idx]

        for field in ("attempt_index", "question_index", "cohort", "case_id", "query", "condition", "arm"):
            if att.get(field) != sched.get(field):
                raise ValueError(
                    f"Attempt at index {idx} field {field!r} mismatch: "
                    f"results has {att.get(field)!r}, schedule has {sched.get(field)!r}"
                )

        tuple_key = (att["cohort"], att["case_id"], att["condition"])
        if tuple_key in seen_tuple_keys:
            raise ValueError(f"Duplicate attempt tuple key in results: {tuple_key}")
        seen_tuple_keys.add(tuple_key)

        cond = att["condition"]
        cohort = att["cohort"]
        arm_spec = protocol["both_arms"]["baseline" if cond == "legacy" else "layout"]
        cat_file = arm_spec["primary_catalog_file"] if cohort == "primary" else arm_spec["table_catalog_file"]
        expected_cat_sha = protocol["generated_assets"][cat_file]["canonical_sha256"]
        expected_pages_sha = arm_spec["raw_sha256"]
        idx_file = arm_spec["index_file"]
        expected_index_hash = protocol["generated_assets"][idx_file]["canonical_sha256"]

        att_hashes = att.get("input_hashes")
        if not isinstance(att_hashes, dict):
            raise ValueError(f"Attempt {idx+1} missing or invalid input_hashes dict")

        for req_h in ("pages_sha256", "catalog_sha256", "index_sha256"):
            val = att_hashes.get(req_h)
            if not isinstance(val, str) or not val.strip():
                raise ValueError(f"Attempt {idx+1} missing or empty input_hashes field {req_h!r}")

        if att_hashes["pages_sha256"] != expected_pages_sha:
            raise ValueError(f"Attempt {idx+1} input pages_sha256 mismatch against bound condition {cond!r}")
        if att_hashes["catalog_sha256"] != expected_cat_sha:
            raise ValueError(f"Attempt {idx+1} input catalog_sha256 mismatch against bound condition {cond!r}")
        if att_hashes["index_sha256"] != expected_index_hash:
            raise ValueError(f"Attempt {idx+1} input index_sha256 mismatch against bound condition {cond!r}")

        raw_att = att.get("raw_attempt")
        if not isinstance(raw_att, dict):
            raise ValueError(f"Attempt {idx+1} raw_attempt must be a dict")
        if raw_att.get("arm") != "mixed":
            raise ValueError(f"Attempt {idx+1} raw_attempt arm must be 'mixed', got {raw_att.get('arm')!r}")
        if raw_att.get("case_id") != att["case_id"]:
            raise ValueError(f"Attempt {idx+1} raw_attempt case_id mismatch: {raw_att.get('case_id')!r} != {att['case_id']!r}")
        if raw_att.get("query") != att["query"]:
            raise ValueError(f"Attempt {idx+1} raw_attempt query mismatch")
        if raw_att.get("attempt_index") != att["attempt_index"]:
            raise ValueError(f"Attempt {idx+1} raw_attempt attempt_index mismatch")

        raw_ans = raw_att.get("formatted_answer", "")
        if att.get("formatted_answer") != raw_ans:
            raise ValueError(f"Attempt {idx+1} formatted_answer altered from raw_attempt")

        raw_sha_field = att.get("raw_attempt_canonical_sha256")
        if not isinstance(raw_sha_field, str) or not raw_sha_field.strip():
            raise ValueError(f"Attempt {idx+1} missing or empty raw_attempt_canonical_sha256")

        raw_sha = hashlib.sha256(
            json.dumps(raw_att, sort_keys=True, ensure_ascii=False, allow_nan=False).encode("utf-8")
        ).hexdigest()
        if raw_sha_field != raw_sha:
            raise ValueError(f"Attempt {idx+1} raw_attempt_canonical_sha256 mismatch: expected {raw_sha}, got {raw_sha_field}")

    cases_by_qidx: dict[int, list[dict[str, Any]]] = {}
    for att in attempts:
        cases_by_qidx.setdefault(att["question_index"], []).append(att)

    if len(cases_by_qidx) != 17:
        raise ValueError(f"Expected 17 unique questions, got {len(cases_by_qidx)}")

    review_cases: list[dict[str, Any]] = []
    mapping_dict: dict[str, dict[str, str]] = {}

    for q_idx in range(17):
        pair = cases_by_qidx.get(q_idx, [])
        if len(pair) != 2:
            raise ValueError(f"Expected 2 attempts for question index {q_idx}, got {len(pair)}")
        att_first, att_second = pair[0], pair[1]

        cid = att_first["case_id"]
        if cid != att_second["case_id"]:
            raise ValueError(f"Case ID mismatch within question index {q_idx}: {cid} != {att_second['case_id']}")

        if att_first["condition"] == att_second["condition"]:
            raise ValueError(f"Question index {q_idx} has duplicate condition {att_first['condition']!r}")

        arm_A = {
            "formatted_answer": att_first.get("formatted_answer", ""),
            "coverage_publication": copy.deepcopy(att_first.get("coverage_publication")),
            "technical_error": att_first.get("error"),
        }
        arm_B = {
            "formatted_answer": att_second.get("formatted_answer", ""),
            "coverage_publication": copy.deepcopy(att_second.get("coverage_publication")),
            "technical_error": att_second.get("error"),
        }

        review_cases.append({
            "question_index": q_idx,
            "cohort": att_first["cohort"],
            "case_id": cid,
            "query": att_first["query"],
            "source_pages": [95, 96, 97, 98, 99],
            "arms": {
                "A": arm_A,
                "B": arm_B,
            },
        })

        mapping_dict[cid] = {
            "A": att_first["condition"],
            "B": att_second["condition"],
        }

    res_sha = hashlib.sha256(p_res.read_bytes()).hexdigest()

    review_bundle = {
        "schema_version": REVIEW_SCHEMA_VERSION,
        "protocol_raw_sha256": proto_sha,
        "results_raw_sha256": res_sha,
        "cases": review_cases,
    }

    # Check and reserve BOTH output destinations before creating or modifying either file
    if p_out.exists():
        raise FileExistsError(f"Export review destination already exists: {p_out}")
    if key_path.exists():
        raise FileExistsError(f"Review key destination already exists: {key_path}")

    p_out.parent.mkdir(parents=True, exist_ok=True)
    p_out.write_text(json.dumps(review_bundle, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    key_data = {
        "schema_version": REVIEW_KEY_SCHEMA_VERSION,
        "protocol_raw_sha256": proto_sha,
        "results_raw_sha256": res_sha,
        "mapping": mapping_dict,
    }
    key_path.write_text(json.dumps(key_data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    return review_bundle


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for Phase 2/3 layout answer comparison preparation, execution, and review."""
    parser = argparse.ArgumentParser(description="Layout answer effect preparation, execution, and review export.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # prepare
    prep_p = subparsers.add_parser("prepare", help="Prepare protocol and indexed assets.")
    prep_p.add_argument("--output-dir", required=True, type=Path, help="Directory to store frozen protocol and assets.")

    # run
    run_p = subparsers.add_parser("run", help="Run the 34-attempt evaluation matrix.")
    run_p.add_argument("--protocol", required=True, type=Path, help="Path to frozen protocol.json.")
    run_p.add_argument("--output", required=True, type=Path, help="Path for atomic results.json.")

    # export-review
    exp_p = subparsers.add_parser("export-review", help="Export neutral blinded A/B review bundle.")
    exp_p.add_argument("--protocol", required=True, type=Path, help="Path to protocol.json.")
    exp_p.add_argument("--results", required=True, type=Path, help="Path to results.json.")
    exp_p.add_argument("--output", required=True, type=Path, help="Path to output review bundle JSON.")

    args = parser.parse_args(argv)
    if args.command == "prepare":
        protocol = prepare(args.output_dir)
        print(f"Prepared layout answer effect protocol with {protocol['counts']['attempts_count']} attempts in {args.output_dir}")
        return 0
    elif args.command == "run":
        results = run(args.protocol, args.output)
        print(f"Executed run {results['run_status']} with {results['summary']['completed_attempts']} completed attempts to {args.output}")
        return 0
    elif args.command == "export-review":
        bundle = export_review(args.protocol, args.results, args.output)
        print(f"Exported review bundle with {len(bundle['cases'])} cases to {args.output}")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
