"""Bounded local continuous regression gate for RAG pension evaluation.

Enforces 11-case benchmark quality, strict source quote integrity against AIReF pages,
and mandatory partial / refusal compliance without extra judge calls or paid APIs.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import unicodedata
from typing import Any

from experiments.preserve_evidence import publication, validate_publication
from experiments.preserved_q3_verifier import (
    VERIFIER_Q3_DIGEST,
    VERIFIER_Q3_MODEL,
    VERIFIER_Q3_PARAMETERS,
)
from experiments.query_coverage import (
    build_coverage_publication,
    claim_inventory,
    replay_coverage_correction,
    format_covered_answer,
    validate_coverage_payload,
)
from experiments.validate_reranker import canonical_hash
from rag_pensiones.chunking import ChunkRecord
from rag_pensiones.grounding import DEFAULT_PARAMETERS, validate_grounded_result
from rag_pensiones.ingest import PROJECT_ROOT, validate_project_path
from rag_pensiones.search import SearchHit

SCHEMA_VERSION = "continuous-quality-report-v1"
REFERENCE_SCHEMA_VERSION = "continuous-quality-reference-v1"
PROTOCOL_SCHEMA_VERSION = "covered-pipeline-protocol-v1"
HISTORICAL_PROTOCOL_SCHEMA_VERSION = "mixed-vs-single27-protocol-v1"
HISTORICAL_RESULTS_SCHEMA_VERSION = "mixed-vs-single27-run-results-v1"
COVERED_RESULTS_SCHEMA_VERSION = "covered-pipeline-v1"

REQUIRED_SERVER_VERSION = "0.40.11"
EXPECTED_CASE_IDS = ("b01", "b02", "b03", "b04", "b05", "b06", "b07", "b08", "n01", "n02", "n03")
EXPECTED_CASE_COUNT = len(EXPECTED_CASE_IDS)

PINNED_9B_DIGEST = "56671c2ab9385f9cfcb404638e32cd62d88e3501d44822208363c010179a3c90"
PINNED_27B_DIGEST = VERIFIER_Q3_DIGEST
HISTORICAL_27B_DIGEST = "1844fd49a2fcf04662d7e81afe585e4846fec76cae77b0a15e3bd1efe45969ed"
SUPPORTED_27B_DIGESTS = (PINNED_27B_DIGEST, HISTORICAL_27B_DIGEST)
PINNED_EMBEDDING_DIGEST = "df5bd2e3c74cd8d069d21dc038f1b359fcdc9458fce1c99bd43c9eb1518ff907"

SHA256_HEX_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")

HISTORICAL_MANDATORY_SOURCE_PATHS: tuple[str, ...] = (
    "rag_pensiones/__init__.py",
    "rag_pensiones/answer.py",
    "rag_pensiones/chunking.py",
    "rag_pensiones/embeddings.py",
    "rag_pensiones/evaluate.py",
    "rag_pensiones/evaluate_grounding.py",
    "rag_pensiones/grounding.py",
    "rag_pensiones/ingest.py",
    "rag_pensiones/planning.py",
    "rag_pensiones/qwen_embeddings.py",
    "rag_pensiones/search.py",
    "rag_pensiones/semantic.py",
    "experiments/__init__.py",
    "experiments/answer_alignment.py",
    "experiments/bounded_context.py",
    "experiments/compare_alignment.py",
    "experiments/compare_bounded_context.py",
    "experiments/compare_context_fix.py",
    "experiments/compare_embeddings.py",
    "experiments/compare_four.py",
    "experiments/compare_four_verifier.py",
    "experiments/compare_hybrid.py",
    "experiments/compare_literal.py",
    "experiments/compare_mixed_single27.py",
    "experiments/compare_notes_only.py",
    "experiments/compare_planned_preserved.py",
    "experiments/compare_preserved.py",
    "experiments/compare_qwen_horizon.py",
    "experiments/compare_reconstructed.py",
    "experiments/compare_relational.py",
    "experiments/compare_requirements_query.py",
    "experiments/compare_retained_notes.py",
    "experiments/compare_retrieval_k.py",
    "experiments/compare_scoped_planning.py",
    "experiments/compare_system.py",
    "experiments/compare_verifier_capacity.py",
    "experiments/compare_verifier_quantization.py",
    "experiments/diagnose_fixed_evidence.py",
    "experiments/diagnose_total_component.py",
    "experiments/direct_alignment.py",
    "experiments/evaluate_coverage_gate.py",
    "experiments/evaluate_covered_pipeline.py",
    "experiments/evaluate_mixed_verifier.py",
    "experiments/evaluate_preserved_stability.py",
    "experiments/fixed_context.py",
    "experiments/four_strategies.py",
    "experiments/hybrid_search.py",
    "experiments/literal_alignment.py",
    "experiments/note_links.py",
    "experiments/planned_preserved.py",
    "experiments/preserve_evidence.py",
    "experiments/preserved_q3_verifier.py",
    "experiments/query_coverage.py",
    "experiments/query_planned_preserved.py",
    "experiments/query_preserved.py",
    "experiments/reconstructed_units.py",
    "experiments/relational_grounding.py",
    "experiments/requirements_query.py",
    "experiments/scoped_planning.py",
    "experiments/source_bound_trial.py",
    "experiments/validate_reranker.py",
)

COVERED_MANDATORY_SOURCE_PATHS: tuple[str, ...] = HISTORICAL_MANDATORY_SOURCE_PATHS + (
    "experiments/continuous_quality.py",
)
STRUCTURED_PROTOCOL_SCHEMA_VERSION = 'covered-pipeline-protocol-v3'
STRUCTURED_MANDATORY_SOURCE_PATHS = COVERED_MANDATORY_SOURCE_PATHS + (
    'rag_pensiones/structured_retrieval.py',
    'experiments/source_recovery_policy.py',
)

HISTORICAL_EFFECTIVE_GENERATION_OPTIONS: dict[str, Any] = {
    "temperature": 0,
    "seed": 42,
    "num_ctx": 8192,
    "num_predict": 1024,
    "presence_penalty": 1.5,
    "top_k": 20,
    "top_p": 0.95,
    "min_p": 0.0,
    "repeat_penalty": 1.0,
    "frequency_penalty": 0.0,
    "typical_p": 1.0,
    "repeat_last_n": 64,
}

HISTORICAL_CANONICAL_ROLE_PARAMETERS: dict[str, Any] = {
    "temperature": 0,
    "seed": 42,
    "num_ctx": 8192,
    "num_predict": 1024,
}

INC_TERMS_RE = (
    r"(?:aumentando|aumento|aumentos|aumenta|aumentan|aumentara|aumentaran|"
    r"crecimiento|crecimientos|crece|crecen|crecera|creceran|"
    r"subida|subidas|sube|suben|incremento|incrementos|incrementa|incrementan)"
)
DEC_TERMS_RE = (
    r"(?:caida|caidas|descenso|descensos|disminucion|disminuciones|reduccion|reducciones|"
    r"baja|bajas|menor|menores|cae|caen|disminuye|disminuyen|desciende|descienden|"
    r"compensa|compensando)"
)
CONST_TERMS_RE = (
    r"(?:constante|constantes|estable|estables|se\s+mantiene|permanece\s+constante)"
)


def _validate_options_dict(options: Any, expected: dict[str, Any], context: str) -> None:
    if not isinstance(options, dict) or isinstance(options, bool):
        raise ValueError(f"{context} must be a dict, got {type(options).__name__}")
    act_keys = set(options.keys())
    exp_keys = set(expected.keys())
    if act_keys != exp_keys:
        missing = sorted(exp_keys - act_keys)
        added = sorted(act_keys - exp_keys)
        details = []
        if missing:
            details.append(f"missing {missing}")
        if added:
            details.append(f"added/drifted {added}")
        raise ValueError(f"{context} options keys mismatch: {'; '.join(details)}")
    for k, exp_val in expected.items():
        act_val = options[k]
        if isinstance(act_val, bool):
            raise ValueError(f"{context} option {k!r} has boolean value masquerading as number: {act_val!r}")
        if not isinstance(act_val, (int, float)):
            raise ValueError(f"{context} option {k!r} must be numeric, got {type(act_val).__name__}")
        if math.isnan(act_val) or math.isinf(act_val):
            raise ValueError(f"{context} option {k!r} has non-finite value: {act_val!r}")
        if isinstance(exp_val, int) and not isinstance(exp_val, bool):
            if not isinstance(act_val, int) or isinstance(act_val, bool):
                raise ValueError(f"{context} option {k!r} expected int {exp_val!r}, got {type(act_val).__name__} {act_val!r}")
            if act_val != exp_val:
                raise ValueError(f"{context} option {k!r} value mismatch: expected {exp_val!r}, got {act_val!r}")
        elif isinstance(exp_val, float):
            if float(act_val) != float(exp_val):
                raise ValueError(f"{context} option {k!r} value mismatch: expected {exp_val!r}, got {act_val!r}")


def _has_active_negation(prefix: str) -> bool:
    """Check if the text immediately preceding a verb/term contains an active negation.
    Preserves affirmative idioms like 'no solo', 'no obstante'."""
    p = prefix.rstrip()
    if re.search(r"\bno\s+obstante\b[^.;]*?$", p) or re.search(r"\bno\s+solo\b[^.;]*?$", p):
        return False
    neg_match = re.search(
        r"\b(no|sin|nunca|jamas|tampoco|ni)(?:\s+(?:se|ha|hay|sigue|el|la|los|las|un|una|del?|por|con))*\s*$",
        p,
    )
    return bool(neg_match)


def _validate_both_pension_factors_increase(text: str) -> bool:
    has_num = bool(
        re.search(r"\bnumero\s+(?:de\s+)?pensiones\b|\bcantidad\s+de\s+pensiones\b", text)
        or ("numero" in text and "pensiones" in text)
    )
    has_mean = bool(re.search(r"\bpension\s+media\b|\bpensiones\s+medias\b", text))
    if not (has_num and has_mean):
        return False

    f_num_re = r"(?:(?:el\s+)?numero\s+(?:de\s+)?pensiones|cantidad\s+de\s+pensiones)"
    f_mean_re = r"(?:(?:la\s+)?pension\s+media|pensiones\s+medias)"
    f_terms = rf"(?:{f_num_re}|{f_mean_re})"
    inc_re = rf"\b{INC_TERMS_RE}\b"
    dec_const_re = rf"\b(?:{DEC_TERMS_RE}|{CONST_TERMS_RE})\b"

    # 1. Reject factor-specific direct negation (e.g. 'ni el número...', 'pero no de la pensión media', 'pensión media no')
    neg_factor_pattern = (
        r"\b(?:ni|sin)\s+(?:del?\s+|de\s+la\s+|el\s+|la\s+)?(?:numero\s+(?:de\s+)?pensiones|pension\s+media)\b"
        r"|\b(?:pero\s+)?no\s+(?:del?\s+|de\s+la\s+|el\s+|la\s+)?(?:numero\s+(?:de\s+)?pensiones|pension\s+media)\b"
        r"|\b(?:numero\s+(?:de\s+)?pensiones|pension\s+media)\b\s+no\s*(?:[.,;:!?]|$|\bpero\b|\by\b|\bsi\b)"
    )
    for m in re.finditer(neg_factor_pattern, text):
        prefix_start = m.start()
        if text[prefix_start:].startswith("no solo") or text[:prefix_start].rstrip().endswith("no solo"):
            continue
        return False

    # 2. Reject unnegated decrease or constant applied to either factor
    # Pattern 2a: decrease/constant precedes factor (e.g. "caída de la pensión media", "compensa la caída de la pensión media")
    pattern_dec_before = rf"(?P<dec>{dec_const_re})\s+(?:del?\s+|de\s+la\s+|la\s+|el\s+|una\s+)?(?P<f>{f_terms})"
    for m in re.finditer(pattern_dec_before, text):
        dec_start = m.start("dec")
        if not _has_active_negation(text[:dec_start]):
            return False

    # Pattern 2b: factor precedes decrease/constant within same clause (e.g. "la pensión media disminuye", "el número permanece constante")
    pattern_dec_after = rf"(?P<f>{f_terms})([^.;,]*?)(?P<dec>{dec_const_re})"
    for m in re.finditer(pattern_dec_after, text):
        dec_start = m.start("dec")
        if not _has_active_negation(text[:dec_start]):
            return False

    # 3. Reject negated increase applied to factors
    # Pattern 3a: negated increase before factor (e.g. "no aumenta el número de pensiones...", "sin aumento del número de pensiones...")
    pattern_neg_inc_before = (
        r"\b(no|sin|nunca|jamas|tampoco|ni)\s+(?!(?:solo|obstante)\b)"
        rf"(?:(?:se|ha|hay|sigue)\s+)?{inc_re}\s+[^.;]*?{f_terms}"
    )
    if re.search(pattern_neg_inc_before, text):
        return False

    # Pattern 3b: factor followed by negated increase (e.g. "la pensión media no aumenta")
    pattern_factor_neg_inc = (
        rf"{f_terms}[^.;,]*?\b(no|sin|nunca|jamas|tampoco|ni)\s+(?!(?:solo|obstante)\b)"
        rf"(?:(?:se|ha|hay|sigue)\s+)?{inc_re}\b"
    )
    if re.search(pattern_factor_neg_inc, text):
        return False

    # 4. Require affirmative increase for BOTH factors (either shared across compound or each individual)
    compound_factors = (
        r"(?:(?:el\s+)?numero\s+(?:de\s+)?pensiones\s+y\s+(?:de\s+)?(?:la\s+)?pension\s+media)"
        r"|(?:(?:la\s+)?pension\s+media\s+y\s+(?:del?\s+)?(?:el\s+)?numero\s+(?:de\s+)?pensiones)"
        r"|(?:tanto\s+(?:del?\s+)?(?:el\s+)?numero\s+(?:de\s+)?pensiones\s+como\s+(?:de\s+)?(?:la\s+)?pension\s+media)"
        r"|(?:no\s+solo\s+(?:del?\s+)?(?:el\s+)?numero\s+(?:de\s+)?pensiones\s+sino\s+tambien\s+(?:de\s+)?(?:la\s+)?pension\s+media)"
        r"|(?:no\s+solo\s+(?:por\s+)?(?:el\s+)?aumento\s+(?:del?\s+)?(?:el\s+)?numero\s+(?:de\s+)?pensiones\s+sino\s+tambien\s+(?:de\s+)?(?:la\s+)?pension\s+media)"
    )

    shared_increase = False
    # Case 4a: increase precedes compound factors (e.g. "aumento del número de pensiones y de la pensión media")
    for m in re.finditer(rf"(?P<inc>{inc_re})\s+[^.;]*?(?P<compound>{compound_factors})", text):
        if not _has_active_negation(text[:m.start("inc")]):
            shared_increase = True
            break

    # Case 4b: compound factors precede increase (e.g. "el número de pensiones y la pensión media aumentan")
    if not shared_increase:
        pattern_4b_direct = rf"(?P<compound>{compound_factors})[^.;,]*?(?P<inc>{inc_re})"
        for m in re.finditer(pattern_4b_direct, text):
            if not _has_active_negation(text[:m.start("inc")]):
                shared_increase = True
                break

    # Case 4b-2: clearly bare same-subject ellipsis after compound factors negated decline
    # (e.g. "el número de pensiones y la pensión media no disminuyen; aumentan.")
    if not shared_increase:
        pattern_4b_ellipsis = (
            rf"(?P<compound>{compound_factors})"
            rf"[^.;]*?\b(?:no\s+disminuyen|no\s+caen|no\s+se\s+reducen|no\s+descienden)\b\s*[;,]\s*"
            rf"(?:sino\s+que\s+|al\s+contrario,\s+|ambos\s+)?"
            rf"(?P<inc>{inc_re})"
            r"\s*(?:ambos|tambien|también)?\s*(?:[.,;:!?]|$)"
        )
        for m in re.finditer(pattern_4b_ellipsis, text):
            if not _has_active_negation(text[:m.start("inc")]):
                shared_increase = True
                break

    # Case 4c: "no solo aumenta X sino tambien Y"
    if not shared_increase:
        no_solo_inc = (
            rf"\bno\s+solo\s+{inc_re}\s+(?:del?\s+)?(?:el\s+)?numero\s+(?:de\s+)?pensiones\s+sino\s+tambien\s+(?:de\s+)?(?:la\s+)?pension\s+media"
            rf"|\bno\s+solo\s+{inc_re}\s+(?:de\s+la\s+)?pension\s+media\s+sino\s+tambien\s+(?:del?\s+)?(?:el\s+)?numero\s+(?:de\s+)?pensiones"
        )
        if re.search(no_solo_inc, text):
            shared_increase = True

    # Case 4d: individual affirmative increase for both factors independently
    num_increase = False
    for m in re.finditer(rf"(?P<inc>{inc_re})\s+[^.;,]*?(?P<f>{f_num_re})", text):
        if not _has_active_negation(text[:m.start("inc")]):
            num_increase = True
            break
    if not num_increase:
        pattern_num_direct = rf"(?P<f>{f_num_re})[^.;,]*?(?P<inc>{inc_re})"
        for m in re.finditer(pattern_num_direct, text):
            if not _has_active_negation(text[:m.start("inc")]):
                num_increase = True
                break
    if not num_increase:
        pattern_num_ellipsis = (
            rf"(?P<f>{f_num_re})"
            rf"[^.;]*?\b(?:no\s+disminuye|no\s+cae|no\s+se\s+reduce)\b\s*[;,]\s*"
            rf"(?:sino\s+que\s+|al\s+contrario,\s+)?"
            rf"(?P<inc>{inc_re})"
            r"\s*(?:tambien|también)?\s*(?:[.,;:!?]|$)"
        )
        for m in re.finditer(pattern_num_ellipsis, text):
            if not _has_active_negation(text[:m.start("inc")]):
                num_increase = True
                break

    mean_increase = False
    for m in re.finditer(rf"(?P<inc>{inc_re})\s+[^.;,]*?(?P<f>{f_mean_re})", text):
        if not _has_active_negation(text[:m.start("inc")]):
            mean_increase = True
            break
    if not mean_increase:
        pattern_mean_direct = rf"(?P<f>{f_mean_re})[^.;,]*?(?P<inc>{inc_re})"
        for m in re.finditer(pattern_mean_direct, text):
            if not _has_active_negation(text[:m.start("inc")]):
                mean_increase = True
                break
    if not mean_increase:
        pattern_mean_ellipsis = (
            rf"(?P<f>{f_mean_re})"
            rf"[^.;]*?\b(?:no\s+disminuye|no\s+cae|no\s+se\s+reduce)\b\s*[;,]\s*"
            rf"(?:sino\s+que\s+|al\s+contrario,\s+)?"
            rf"(?P<inc>{inc_re})"
            r"\s*(?:tambien|también)?\s*(?:[.,;:!?]|$)"
        )
        for m in re.finditer(pattern_mean_ellipsis, text):
            if not _has_active_negation(text[:m.start("inc")]):
                mean_increase = True
                break

    return shared_increase or (num_increase and mean_increase)

DEFAULT_INDEX_REL = "data/processed/airef_2025_qwen3_4b_embeddings.json"
DEFAULT_QUERIES_REL = "evaluation/scope_coverage_2026-10-26/full_inputs.json"
DEFAULT_UNITS_REL = "evaluation/reconstructed_units_catalog.json"
DEFAULT_PAGES_REL = "data/processed/airef_2025_pages.jsonl"
DEFAULT_OLLAMA_DEFAULTS_REL = "evaluation/verifier_capacity_2026-10-26/ollama_defaults_v0.40.11.json"
DEFAULT_DEFAULTS_REL = DEFAULT_OLLAMA_DEFAULTS_REL
DEFAULT_REFERENCE_REL = "evaluation/continuous_quality_2026-10-27/reference.json"


def _reject_constant(val: str) -> None:
    raise ValueError(f"Non-finite JSON numeric constant rejected: {val!r}")


def _parse_float(val: str) -> float:
    f = float(val)
    if math.isinf(f) or math.isnan(f):
        raise ValueError(f"Non-finite or overflow JSON float rejected: {val!r}")
    return f


def _strict_pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    res: dict[str, Any] = {}
    for k, v in pairs:
        if k in res:
            raise ValueError(f"Duplicate JSON key rejected: {k!r}")
        res[k] = v
    return res


def strict_load_json(data: str | bytes) -> Any:
    if isinstance(data, bytes):
        data = data.decode("utf-8")
    parsed = json.loads(
        data,
        parse_constant=_reject_constant,
        parse_float=_parse_float,
        object_pairs_hook=_strict_pairs_hook,
    )
    try:
        json.dumps(parsed, allow_nan=False)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Container contains non-compliant JSON values (NaN/Infinity): {exc}") from exc
    return parsed


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compute_source_closure() -> dict[str, str]:
    closure: dict[str, str] = {}
    for pkg in ("rag_pensiones", "experiments"):
        pkg_dir = PROJECT_ROOT / pkg
        for py_file in sorted(pkg_dir.rglob("*.py")):
            rel_path = py_file.relative_to(PROJECT_ROOT).as_posix()
            closure[rel_path] = file_sha256(py_file)
    closure[DEFAULT_OLLAMA_DEFAULTS_REL] = file_sha256(PROJECT_ROOT / DEFAULT_OLLAMA_DEFAULTS_REL)
    return closure


def normalize_text(text: str) -> str:
    nfkd = unicodedata.normalize("NFKD", text)
    ascii_text = nfkd.encode("ASCII", "ignore").decode("utf-8").lower()
    return re.sub(r"\s+", " ", ascii_text).strip()


def validate_reference_ledger(reference_data: dict[str, Any]) -> None:
    if not isinstance(reference_data, dict):
        raise ValueError("reference_data must be a dict")
    if reference_data.get("schema_version") != REFERENCE_SCHEMA_VERSION:
        raise ValueError(f"Invalid reference schema: {reference_data.get('schema_version')!r}")
    if reference_data.get("roster") != list(EXPECTED_CASE_IDS):
        raise ValueError(f"Reference roster mismatch: expected {list(EXPECTED_CASE_IDS)}, got {reference_data.get('roster')!r}")
    cases = reference_data.get("cases")
    if not isinstance(cases, dict) or set(cases.keys()) != set(EXPECTED_CASE_IDS):
        raise ValueError("Reference cases mismatch: must contain exactly 11 benchmark cases")
    for cid in EXPECTED_CASE_IDS:
        c_entry = cases[cid]
        if not isinstance(c_entry, dict):
            raise ValueError(f"Reference entry for {cid} must be a dict")
        q = c_entry.get("query")
        if not isinstance(q, str) or not q.strip():
            raise ValueError(f"Reference query for {cid} must be non-empty string")
        ans = c_entry.get("answerability")
        if ans not in ("answerable", "outside", "mixed"):
            raise ValueError(f"Invalid answerability for {cid}: {ans!r}")
        st = c_entry.get("expected_status")
        if st not in ("answered", "insufficient_evidence", "partial"):
            raise ValueError(f"Invalid expected_status for {cid}: {st!r}")
        rf = c_entry.get("required_facts")
        if not isinstance(rf, list):
            raise ValueError(f"required_facts for {cid} must be a list")
        if cid in ("b06", "b07", "n01"):
            if len(rf) != 0:
                raise ValueError(f"Outside case {cid} must have empty required_facts, got {rf!r}")
        else:
            if len(rf) == 0:
                raise ValueError(f"Case {cid} must have non-empty required_facts")
            for f in rf:
                if not isinstance(f, str) or not f.strip():
                    raise ValueError(f"required_facts in {cid} contains invalid item: {f!r}")
        crit = c_entry.get("criteria")
        if not isinstance(crit, str) or not crit.strip():
            raise ValueError(f"Reference entry for {cid} must have non-empty criteria")


def extract_texts_from_publication(pub: dict[str, Any] | None) -> tuple[str, str, str, str]:
    if not isinstance(pub, dict):
        return "", "", "", ""
    main_parts: list[str] = []
    quote_parts: list[str] = []
    refusal_parts: list[str] = []

    for c in pub.get("claims", []):
        if isinstance(c, dict) and isinstance(c.get("text"), str):
            main_parts.append(c["text"])
            for ev in c.get("evidence", []):
                if isinstance(ev, dict) and isinstance(ev.get("quote"), str):
                    quote_parts.append(ev["quote"])

    for m in pub.get("missing_parts", []):
        if isinstance(m, dict):
            if isinstance(m.get("message"), str):
                refusal_parts.append(m["message"])
            if isinstance(m.get("query_span"), str):
                refusal_parts.append(m["query_span"])
    if isinstance(pub.get("reason"), str) and pub["reason"].strip():
        refusal_parts.append(pub["reason"].strip())

    main_text = " ".join(main_parts)
    quote_text = " ".join(quote_parts)
    refusal_text = " ".join(refusal_parts)
    whole_text = " ".join(part for part in (main_text, quote_text, refusal_text) if part)
    return main_text, quote_text, refusal_text, whole_text


def extract_evidence_provenance(case_data: dict[str, Any], *, verified_structured=None, publication=None) -> dict[str, dict[str, Any]]:
    prov: dict[str, dict[str, Any]] = {}
    hits = (
        case_data.get("retrieval", {}).get("hits", [])
        or (case_data.get("answer", {}).get("factual_report", {}).get("retrieval", {}).get("hits", []) if isinstance(case_data.get("answer"), dict) else [])
        or case_data.get("factual_outcome", {}).get("expanded_hits", [])
    )
    for h in hits:
        if isinstance(h, dict):
            sid = h.get("source_id") or (f"S{h['rank']}" if "rank" in h else None)
            chunk = h.get("chunk") if isinstance(h.get("chunk"), dict) else h
            page_num = chunk.get("page_number") or h.get("page_number")
            doc_id = chunk.get("doc_id") or h.get("doc_id")
            source_file = chunk.get("source_file") or h.get("source_file")
            text = chunk.get("text") or h.get("text")
            if sid:
                prov[sid] = {
                    "source_id": sid,
                    "page_number": page_num,
                    "doc_id": doc_id,
                    "source_file": source_file,
                    "text": text,
                }

    full_result = (
        case_data.get("factual_outcome", {}).get("full_result")
        or (case_data.get("answer", {}).get("factual_report", {}).get("outcome", {}).get("full_result") if isinstance(case_data.get("answer"), dict) else None)
    )
    if isinstance(full_result, dict):
        grounding = full_result.get("grounding", {})
        if isinstance(grounding, dict):
            for ev_list in (grounding.get("selected_evidence", []), grounding.get("repair_evidence", [])):
                if isinstance(ev_list, list):
                    for item in ev_list:
                        if isinstance(item, dict) and "source_id" in item:
                            sid = item["source_id"]
                            existing = prov.get(sid, {})
                            prov[sid] = {
                                "source_id": sid,
                                "page_number": item.get("page_number", existing.get("page_number")),
                                "doc_id": item.get("doc_id", existing.get("doc_id")),
                                "source_file": item.get("source_file", existing.get("source_file")),
                                "quote": item.get("quote", existing.get("quote")),
                                "text": existing.get("text"),
                            }
    if verified_structured is not None and publication is not None:
        valid_ids = {h['observation_id']: h for h in verified_structured.get('hits', [])}
        for claim in publication.get('claims', []):
            for ev in claim.get('evidence', []):
                if ev.get('source_kind') == 'structured_cell':
                    hit = valid_ids.get(ev.get('observation_id'))
                    if hit is None or ev.get('source_id') != 'DB:' + hit['observation_id']:
                        raise ValueError('Structured evidence outside replayed source records')
                    prov[ev['source_id']] = dict(source_kind='structured_cell', structured_verified=True,
                        observation_id=hit['observation_id'],source_file=hit['source_path'],source_sha256=hit['source_sha256'],
                        source_sheet=hit['source_sheet'],source_cell=hit['source_cell'],unit=hit['unit'],value=hit['value'],quote=ev['quote'])
    return prov


def evaluate_case_reference(
    cid: str,
    publication_view: dict[str, Any] | None,
    provenance_map: dict[str, dict[str, Any]],
    pages_by_num: dict[int, dict[str, Any]],
    reference_case: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not isinstance(publication_view, dict):
        return {
            "whole_correct": False,
            "main_direct_correct": False,
            "safe": False,
            "direct": False,
            "warnings": [],
            "failures": ["missing_publication"],
        }

    failures: list[str] = []
    warnings: list[str] = []
    whole_correct = True
    main_direct_correct = True
    safe = True
    direct = True

    if cid not in EXPECTED_CASE_IDS:
        failures.append(f"unknown_case_id: {cid!r}")
        return {
            "whole_correct": False,
            "main_direct_correct": False,
            "safe": False,
            "direct": False,
            "warnings": [],
            "failures": failures,
        }

    claims = publication_view.get("claims", [])
    status = publication_view.get("status", "")
    main_text, quote_text, refusal_text, whole_text = extract_texts_from_publication(publication_view)

    norm_main = normalize_text(main_text)
    norm_whole = normalize_text(whole_text)
    norm_refusal = normalize_text(refusal_text)

    # Validate status according to reference contract
    expected_status = (
        reference_case.get("expected_status")
        if isinstance(reference_case, dict) and reference_case.get("expected_status")
        else ("insufficient_evidence" if cid in ("b06", "b07", "n01") else ("partial" if cid in ("b08", "n03") else "answered"))
    )

    if status != expected_status:
        failures.append(f"{cid}_status_mismatch: expected {expected_status}, got {status}")
        if cid in ("b06", "b07", "n01"):
            safe = False

    # 1. Structural & Source Provenance Integrity of Claims
    if expected_status in ("answered", "partial"):
        if not isinstance(claims, list) or len(claims) == 0:
            failures.append(f"{cid}_missing_claims_for_{expected_status}")

    if expected_status == "insufficient_evidence":
        if isinstance(claims, list) and len(claims) > 0:
            failures.append(f"{cid}_failed_to_abstain")
            safe = False

    if isinstance(claims, list):
        for c_idx, c in enumerate(claims):
            if not isinstance(c, dict):
                failures.append(f"claim_{c_idx}_not_dict")
                continue
            c_text = c.get("text")
            if not isinstance(c_text, str) or not c_text.strip():
                failures.append(f"claim_{c_idx}_empty_text")
            c_ev = c.get("evidence", [])
            if not isinstance(c_ev, list) or len(c_ev) == 0:
                failures.append(f"claim_{c_idx}_missing_evidence")
            else:
                for ev in c_ev:
                    if not isinstance(ev, dict):
                        failures.append("evidence_item_not_dict")
                        continue
                    sid = ev.get("source_id")
                    q = ev.get("quote")
                    if not isinstance(sid, str) or not sid.strip() or sid not in provenance_map:
                        failures.append(f"unknown_source_id: {sid!r}")
                        safe = False
                        continue
                    prov = provenance_map[sid]
                    if prov.get('source_kind') == 'structured_cell':
                        if (prov.get('structured_verified') is not True or ev.get('source_kind') != 'structured_cell'
                            or ev.get('observation_id') != prov.get('observation_id')
                            or ev.get('source_path') != prov.get('source_file')
                            or ev.get('source_sha256') != prov.get('source_sha256')
                            or ev.get('source_sheet') != prov.get('source_sheet')
                            or ev.get('source_cell') != prov.get('source_cell') or q != prov.get('quote')):
                            failures.append('structured_source_identity_mismatch');safe=False
                        continue
                    page_num = prov.get("page_number")
                    if page_num not in pages_by_num:
                        failures.append(f"page_number_{page_num}_not_in_canonical_pages")
                        continue
                    canonical_page = pages_by_num[page_num]
                    if prov.get("doc_id") and canonical_page.get("doc_id") and prov["doc_id"] != canonical_page["doc_id"]:
                        failures.append(f"doc_id_mismatch_for_{sid}")
                    if prov.get("source_file") and canonical_page.get("source_file") and prov["source_file"] != canonical_page["source_file"]:
                        failures.append(f"source_file_mismatch_for_{sid}")
                    if not isinstance(q, str) or len(q) < 12:
                        failures.append("quote_too_short_or_invalid")
                    elif normalize_text(q) not in normalize_text(canonical_page.get("text", "")):
                        failures.append(f"quote_not_in_canonical_page_{page_num}: {q[:30]!r}")

    # 2. Case-Specific Content & Reference Rubric
    # For b01-b05, b08, n03, all required facts MUST appear in MAIN accepted published claims.
    # Quote-only is never sufficient for these.
    if cid == "b01":
        has_2036 = bool(re.search(r"\b2036\b", norm_main))
        has_num_pensions = bool(re.search(r"\bnumero\s+(?:de\s+)?pensiones\b|\bcantidad\s+de\s+pensiones\b", norm_main) or ("numero" in norm_main and "pensiones" in norm_main))
        has_2pct = bool(re.search(r"(?<!\d)2\s?%(?!\d)|\b2\s*por\s*ciento\b|\bdos\s*por\s*ciento\b", norm_main))
        has_above = any(w in norm_main for w in ("encima", "supera", "superior", "mayor", "mas de", ">")) or "acelera" in norm_main
        is_inverted = bool(re.search(r"\b(?:por debajo|debajo|inferior|menor|menos|<)\s*(?:del?\s*)?(?:(?<!\d)2\s?%(?!\d)|\b2\s*por\s*ciento\b|\bdos\s*por\s*ciento\b)", norm_main))
        is_negated = bool(re.search(r"\b(?:no|sin|nunca)\s+(?:se\s+)?(?:acelera|supera|crece|alcanza)", norm_main))

        if not has_2036:
            failures.append("b01_main_missing_2036")
        if not has_num_pensions:
            failures.append("b01_main_missing_number_of_pensions")
        if not has_2pct:
            failures.append("b01_main_missing_2pct")
        if not (has_2036 and has_2pct):
            failures.append("b01_main_missing_2036_or_2pct")
        if is_inverted:
            failures.append("b01_main_inverted_relation")
        elif is_negated or not has_above:
            failures.append("b01_main_missing_growth_above_2pct")

    elif cid == "b02":
        has_2050 = bool(re.search(r"\b2050\b", norm_main))
        has_until_or_before_2050 = bool(re.search(r"\b(?:hasta|antes\s+de|previo\s+a)\s+(?:el\s+ano\s+)?2050\b|\b2050\b[^.]*\b(?:hasta|antes\s+de)\b", norm_main))
        has_after_2050 = bool(re.search(r"\b(?:despues\s+de|posterior\s+a|a\s+partir\s+de|tras)\s+(?:el\s+ano\s+)?2050\b", norm_main))
        has_4pct = bool(re.search(r"(?<!\d)4\s?%(?!\d)|\b4\s*por\s*ciento\b|\bcuatro\s*por\s*ciento\b", norm_main))
        has_annual = bool(re.search(r"\b(?:anual|al ano|al año|anualmente)\b", norm_main))
        has_above_4 = any(w in norm_main for w in ("encima", "supera", "superior", "mayor", "mas de", ">")) or bool(re.search(r"\b(?:crecera|crece|crecimiento|aumentara|aumento)\s+(?:por\s+)?encima\b", norm_main))
        is_inverted_4 = bool(re.search(r"\b(?:por debajo|debajo|inferior|menor|menos|<)\s*(?:del?\s*)?(?:(?<!\d)4\s?%(?!\d)|\b4\s*por\s*ciento\b|\bcuatro\s*por\s*ciento\b)", norm_main))
        is_negated_4 = bool(re.search(r"\b(?:no|sin|nunca)\s+(?:crecera|crece|supera|alcanza)", norm_main))
        has_num_pensions = bool(re.search(r"\bnumero\s+(?:de\s+)?pensiones\b|\bcantidad\s+de\s+pensiones\b", norm_main) or ("numero" in norm_main and "pensiones" in norm_main))
        has_mean_pension = bool(re.search(r"\bpension\s+media\b|\bpensiones\s+medias\b", norm_main))

        both_causes_increase = _validate_both_pension_factors_increase(norm_main)

        if not has_2050 or not has_until_or_before_2050 or has_after_2050:
            failures.append("b02_main_missing_2050")
        if not has_4pct:
            failures.append("b02_main_missing_4pct")
        if not has_annual:
            failures.append("b02_main_missing_annual")
        if not (has_2050 and has_4pct):
            failures.append("b02_main_missing_rate_or_variables")
        if is_inverted_4:
            failures.append("b02_main_inverted_relation")
        elif is_negated_4 or not has_above_4:
            failures.append("b02_main_missing_growth_above_4pct")
        if not (has_num_pensions and has_mean_pension):
            failures.append("b02_main_missing_number_or_mean")
        elif not both_causes_increase:
            failures.append("b02_main_missing_increase_mechanism")

    elif cid == "b03":
        has_pension_ipc = bool(re.search(r"(?:pension(?:es)?|primeras?)[^.]*\bipc\b|\bipc\b[^.]*(?:pension(?:es)?|primeras?)", norm_main))
        has_wages_prod = bool(re.search(r"(?:salarios?|sueldos?|segundos?)[^.]*\bproductividad\b|\bproductividad\b[^.]*(?:salarios?|sueldos?|segundos?)", norm_main))
        has_wages_both = bool(
            re.search(r"(?:salarios?|sueldos?|segundos?)[^.;]*?\b(?:productividad[^.;]*?\bipc|ipc[^.;]*?\bproductividad)\b", norm_main)
            or re.search(r"\b(?:productividad[^.;]*?\bipc|ipc[^.;]*?\bproductividad)[^.;]*?(?:salarios?|sueldos?|segundos?)\b", norm_main)
        )
        wages_exclusive_prod = bool(re.search(r"\b(?:exclusivamente|unicamente|solo)\s+(?:con\s+)?(?:la\s+)?productividad\b", norm_main))

        is_swapped = False
        if re.search(r"(?:pension(?:es)?|primeras?)[^.]*\bproductividad\b", norm_main) and not re.search(r"(?:salarios?|sueldos?|segundos?)[^.]*\bproductividad\b", norm_main):
            is_swapped = True
        if re.search(r"(?:salarios?|sueldos?|segundos?)[^.]*\bipc\b", norm_main) and not re.search(r"(?:salarios?|sueldos?|segundos?)[^.]*\bproductividad\b", norm_main):
            is_swapped = True
        if re.search(r"(?:pension(?:es)?|primeras?)\s+(?:crecen?|aumentan?)\s+con\s+(?:la\s+)?productividad", norm_main):
            is_swapped = True

        if is_swapped or wages_exclusive_prod or not (has_pension_ipc and has_wages_prod and has_wages_both):
            failures.append("b03_main_swapped_or_missing_attribution")

    elif cid == "b04":
        has_2015 = bool(re.search(r"\b2015\b", norm_main))
        has_agg_or_suff = any(t in norm_main for t in ("agregad", "datos agregados", "medida agregada", "nivel agregado", "indicador de suficiencia", "tasa de generosidad agregada", "suficiencia"))
        is_claimed_indiv = bool(re.search(r"\b(?:es|constituye|representa)\s+(?:una\s+)?garantia\s+individual\b", norm_main) and not re.search(r"\bno\s+(?:es|constituye|representa|supone)?\s*(?:una\s+)?garantia\s+individual\b", norm_main))

        if not has_2015:
            failures.append("b04_main_missing_2015")
        if not has_agg_or_suff:
            failures.append("b04_main_missing_aggregate_or_sufficiency")
        if not (has_2015 and has_agg_or_suff):
            failures.append("b04_main_missing_2015_or_aggregate")
        if is_claimed_indiv:
            failures.append("b04_main_claimed_individual_guarantee")

    elif cid == "b05":
        has_num_pensions = bool(re.search(r"\bnumero\s+(?:de\s+)?pensiones\b|\bcantidad\s+de\s+pensiones\b", norm_main) or ("numero" in norm_main and "pensiones" in norm_main))
        has_mean_pension = bool(re.search(r"\bpension\s+media\b|\bpensiones\s+medias\b", norm_main))

        both_causes_increase = _validate_both_pension_factors_increase(norm_main)

        if not has_num_pensions or not has_mean_pension:
            failures.append("b05_main_missing_number_or_mean")
        elif not both_causes_increase:
            failures.append("b05_main_missing_increase_cause")

    elif cid in ("b06", "b07", "n01"):
        if not norm_refusal:
            failures.append(f"{cid}_missing_refusal_reason")
        target_terms = ("cataluna", "cataluña", "regional", "territorial") if cid == "b06" else (("mujeres", "sexo", "genero") if cid == "b07" else ("hombres", "sexo", "genero"))
        has_target_in_refusal = any(t in norm_refusal for t in target_terms) or any(t in norm_refusal for t in ("alcance", "evidencias", "informacion", "datos"))
        missing_parts = publication_view.get("missing_parts", [])
        has_missing_coverage = any(any(t in normalize_text(m.get("query_span", "")) for t in target_terms) for m in missing_parts if isinstance(m, dict))
        if not (has_target_in_refusal or has_missing_coverage):
            failures.append(f"{cid}_refusal_unmatched_query_scope")

    elif cid in ("b08", "n03"):
        target_year = "2031" if cid == "b08" else "2030"
        has_2015_main = bool(re.search(r"\b2015\b", norm_main))
        if not has_2015_main:
            failures.append(f"{cid}_omitted_2015_in_main_answer")

        missing_parts = publication_view.get("missing_parts", [])
        missing_spans = [normalize_text(m.get("query_span", "")) for m in missing_parts if isinstance(m, dict)]
        has_span_coverage = any(target_year in s or any(w in s for w in ("mensual", "cobrare", "cuanto", "personal", "exacta")) for s in missing_spans)
        has_refusal_intent = any(term in norm_refusal for term in ("no se puede", "no es posible", "no dispone", "no contiene", "no proporciona", "insuficiente", "fuera de alcance", "no permite", "not_in_evaluated_evidence", "no contestada", "no respondida", "sin datos", "no cuenta"))
        has_unanswerable_target = (target_year in norm_refusal) or any(term in norm_refusal for term in ("individual", "personal", "mensual", "cuanto cobrare", "cobrare", "importe exacto", "pension exacta", "caso particular"))
        is_trivial_reason = norm_refusal.strip() in (target_year, "informacion", "no informacion", "ano", "año")

        if not (has_refusal_intent and has_unanswerable_target and (has_span_coverage or len(missing_parts) > 0)) or is_trivial_reason:
            failures.append(f"{cid}_missing_individual_refusal")

        if re.search(r"(?<!\w)(?:\d+[\d.,]*\s*(?:euros?|€|eur)\b|cobrare\s+\d+|mensual\s+de\s+\d+)", norm_main):
            safe = False
            failures.append(f"{cid}_invented_personal_amount")

    elif cid == "n02":
        has_2015_main = bool(re.search(r"\b2015\b", norm_main))
        has_2015_whole = bool(re.search(r"\b2015\b", norm_whole))
        if has_2015_main:
            main_direct_correct = True
            direct = True
        elif has_2015_whole:
            main_direct_correct = False
            direct = False
            warnings.append("n02_quote_only_debt")
        else:
            main_direct_correct = False
            direct = False
            failures.append("n02_missing_2015")

    if failures:
        whole_correct = False
        main_direct_correct = False

    return {
        "whole_correct": whole_correct and len(failures) == 0,
        "main_direct_correct": main_direct_correct and len(failures) == 0,
        "safe": safe,
        "direct": direct,
        "warnings": warnings,
        "failures": failures,
    }


def prepare(
    output_dir: Path | str,
    *,
    index_path: str | None = None,
    queries_path: str | None = None,
    units_path: str | None = None,
    pages_path: str | None = None,
    coverage_correction: bool = False,
    structured_retrieval: bool = False,
) -> dict[str, Any]:
    if type(coverage_correction) is not bool:
        raise ValueError("coverage_correction must be a strict boolean")
    if type(structured_retrieval) is not bool:
        raise ValueError('structured_retrieval must be a strict boolean')
    out_dir = Path(output_dir)
    if not out_dir.is_absolute():
        out_dir = (PROJECT_ROOT / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    target_proto = out_dir / "protocol.json"
    if target_proto.exists():
        raise FileExistsError(f"Protocol destination already exists: {target_proto}")

    closure = compute_source_closure()
    rel_idx = index_path or DEFAULT_INDEX_REL
    rel_qry = queries_path or DEFAULT_QUERIES_REL
    rel_unt = units_path or DEFAULT_UNITS_REL
    rel_pgs = pages_path or DEFAULT_PAGES_REL

    p_idx = (PROJECT_ROOT / rel_idx).resolve() if not Path(rel_idx).is_absolute() else Path(rel_idx)
    p_qry = (PROJECT_ROOT / rel_qry).resolve() if not Path(rel_qry).is_absolute() else Path(rel_qry)
    p_unt = (PROJECT_ROOT / rel_unt).resolve() if not Path(rel_unt).is_absolute() else Path(rel_unt)
    p_pgs = (PROJECT_ROOT / rel_pgs).resolve() if not Path(rel_pgs).is_absolute() else Path(rel_pgs)

    closure[rel_idx] = file_sha256(p_idx)
    closure[rel_qry] = file_sha256(p_qry)
    closure[rel_unt] = file_sha256(p_unt)
    closure[rel_pgs] = file_sha256(p_pgs)

    idx_data = strict_load_json(p_idx.read_bytes())
    qry_data = strict_load_json(p_qry.read_bytes())
    unt_data = strict_load_json(p_unt.read_bytes())
    pgs_lines = [strict_load_json(line) for line in p_pgs.read_text(encoding="utf-8").splitlines() if line.strip()]

    input_hashes = {
        "index": canonical_hash(idx_data),
        "queries": canonical_hash(qry_data),
        "units": canonical_hash(unt_data),
        "pages": canonical_hash(pgs_lines),
    }

    protocol: dict[str, Any] = {
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
        "frozen_files": closure,
        "input_hashes": input_hashes,
    }

    if coverage_correction:
        protocol["schema_version"] = "covered-pipeline-protocol-v2"
        protocol["coverage_correction"] = True
    if structured_retrieval:
        from rag_pensiones.structured_retrieval import source_manifest
        manifest = source_manifest()
        protocol['schema_version'] = STRUCTURED_PROTOCOL_SCHEMA_VERSION
        protocol['structured_retrieval'] = manifest
        protocol['frozen_files'].update(manifest['files'])

    target_proto.write_text(json.dumps(protocol, indent=2, ensure_ascii=False), encoding="utf-8")
    return protocol


def _reconstruct_search_hits(hits_raw: list[dict[str, Any]]) -> list[SearchHit]:
    hits: list[SearchHit] = []
    for h in hits_raw:
        if not isinstance(h, dict):
            continue
        rank = h.get("rank", 1)
        score = float(h.get("score", 0.0))
        chunk_data = h.get("chunk")
        if isinstance(chunk_data, dict):
            chunk = ChunkRecord(
                chunk_id=chunk_data.get("chunk_id", ""),
                doc_id=chunk_data.get("doc_id", ""),
                source_file=chunk_data.get("source_file", ""),
                published_at=chunk_data.get("published_at", ""),
                page_number=int(chunk_data.get("page_number", 1)),
                start_char=int(chunk_data.get("start_char", 0)),
                end_char=int(chunk_data.get("end_char", 0)),
                text=chunk_data.get("text", ""),
                strategy=chunk_data.get("strategy", "semantic"),
            )
        else:
            chunk = ChunkRecord(
                chunk_id=h.get("chunk_id", ""),
                doc_id=h.get("doc_id", "doc1"),
                source_file=h.get("source_file", "report.pdf"),
                published_at=h.get("published_at", "2025-03-31"),
                page_number=int(h.get("page_number", 1)),
                start_char=int(h.get("start_char", 0)),
                end_char=int(h.get("end_char", len(h.get("text", "")))),
                text=h.get("text", ""),
                strategy=h.get("strategy", "semantic"),
            )
        hits.append(SearchHit(rank=rank, score=score, chunk=chunk))
    return hits


def replay_structured_answer(answer, query, index, inventory, pdf_publication, *, case_status=None, formatted_answer=None):
    """Replay only the typed branch; the caller independently validates PDF canonical evidence."""
    from rag_pensiones.structured_retrieval import compose_publication, validate_structured_result
    factual = answer.get('factual_report', {})
    if (answer.get('query') != query or answer.get('metadata', {}).get('structured_profile') != 'certified_lookup_parallel_v1'
        or factual.get('query') != query or factual.get('metadata', {}).get('index_hash') != canonical_hash(index)):
        raise ValueError('Structured replay query/index/profile mismatch')
    structured = factual.get('retrieval', {}).get('structured')
    if not isinstance(structured, dict) or structured.get('status') == 'error':
        raise ValueError('Missing or failed structured retrieval')
    if structured.get('status') == 'not_applicable':
        validate_structured_result(query,index,structured)
        if 'joint_publication' in answer or 'structured_report' in answer:
            raise ValueError('Unexpected joint payload for inapplicable retrieval')
        final, verified = pdf_publication, None
    else:
        if answer.get('structured_report') != structured:
            raise ValueError('Structured transport changed')
        final = compose_publication(query,pdf_publication,structured,inventory,index)
        if canonical_hash(final) != canonical_hash(answer.get('joint_publication')):
            raise ValueError('Joint publication differs from source replay')
        if answer.get('metadata', {}).get('joint_coverage_status') != final['status']:
            raise ValueError('Joint coverage metadata mismatch')
        verified = structured
    if answer.get('status') != final['status'] or (case_status is not None and case_status != final['status']):
        raise ValueError('Joint/global status mismatch')
    if formatted_answer is not None and formatted_answer != format_covered_answer(answer):
        raise ValueError('Formatted answer differs from effective publication')
    return final, verified


def check(
    results_path: Path | str,
    reference_path: Path | str,
    output_path: Path | str,
    *,
    arm: str | None = None,
    baseline_path: Path | None = None,
    protocol_path: Path | None = None,
) -> dict[str, Any]:
    p_res = Path(results_path)
    if not p_res.is_absolute():
        p_res = (PROJECT_ROOT / p_res).resolve()
    p_ref = Path(reference_path)
    if not p_ref.is_absolute():
        p_ref = (PROJECT_ROOT / p_ref).resolve()
    p_out = Path(output_path)
    if not p_out.is_absolute():
        p_out = (PROJECT_ROOT / p_out).resolve()

    if p_out.exists():
        raise FileExistsError(f"Output path already exists: {p_out}")
    if not p_res.is_file():
        raise FileNotFoundError(f"Results file not found: {p_res}")
    if not p_ref.is_file():
        raise FileNotFoundError(f"Reference file not found: {p_ref}")

    results_data = strict_load_json(p_res.read_bytes())
    reference_data = strict_load_json(p_ref.read_bytes())

    validate_reference_ledger(reference_data)

    schema_ver = results_data.get("schema_version")
    if schema_ver not in (HISTORICAL_RESULTS_SCHEMA_VERSION, COVERED_RESULTS_SCHEMA_VERSION):
        raise ValueError(f"Unsupported results schema version: {schema_ver!r}")

    # Bind and validate protocol
    p_proto: Path | None = None
    if protocol_path is not None:
        p_proto = Path(protocol_path)
        if not p_proto.is_absolute():
            p_proto = (PROJECT_ROOT / p_proto).resolve()
    elif results_data.get("protocol_path"):
        p_proto = (PROJECT_ROOT / results_data["protocol_path"]).resolve()
    elif (p_res.parent / "protocol.json").is_file():
        p_proto = (p_res.parent / "protocol.json").resolve()

    if p_proto is None or not p_proto.is_file():
        raise ValueError("Could not resolve protocol freeze document for results")

    proto_bytes = p_proto.read_bytes()
    proto_sha256 = hashlib.sha256(proto_bytes).hexdigest()
    protocol_data = strict_load_json(proto_bytes)

    if schema_ver == HISTORICAL_RESULTS_SCHEMA_VERSION:
        if protocol_data.get("schema_version") != HISTORICAL_PROTOCOL_SCHEMA_VERSION:
            raise ValueError(f"Historical protocol schema mismatch: {protocol_data.get('schema_version')!r}")
        res_proto_sha = results_data.get("protocol_sha256")
        if not res_proto_sha or res_proto_sha != proto_sha256:
            raise ValueError(f"Historical protocol raw-bytes SHA-256 mismatch: results has {res_proto_sha!r}, expected {proto_sha256!r}")
        paths_dict = protocol_data.get("paths")
        if not isinstance(paths_dict, dict) or not {"index", "queries", "units", "pages"}.issubset(paths_dict.keys()):
            raise ValueError("Historical protocol missing required paths keys")
        pages_rel = paths_dict["pages"]
        queries_rel = paths_dict["queries"]
        index_rel = paths_dict["index"]
        units_rel = paths_dict["units"]

        # Validate recorded historical server, models, digests and options against supported profiles
        srv = protocol_data.get("server_version")
        if srv != REQUIRED_SERVER_VERSION:
            raise ValueError(f"Historical protocol server_version mismatch: expected {REQUIRED_SERVER_VERSION!r}, got {srv!r}")
        models = protocol_data.get("models")
        if not isinstance(models, dict) or isinstance(models, bool) or not models:
            raise ValueError("Historical protocol missing or invalid models dictionary")
        mandatory_models = ("generator_9b", "text_27b", "embedding_4b")
        for m_key in mandatory_models:
            if m_key not in models:
                raise ValueError(f"Historical protocol models dictionary missing mandatory model: {m_key!r}")
            m_info = models[m_key]
            if not isinstance(m_info, dict) or isinstance(m_info, bool) or not m_info:
                raise ValueError(f"Historical protocol model {m_key!r} setup must be a non-empty dict")
            m_name = m_info.get("name")
            if not isinstance(m_name, str) or isinstance(m_name, bool) or not m_name.strip():
                raise ValueError(f"Historical protocol model {m_key!r} missing or invalid name: {m_name!r}")
            if m_key == "generator_9b" and m_name != "qwen3.5:9b":
                raise ValueError(f"Historical protocol model generator_9b name mismatch: expected 'qwen3.5:9b', got {m_name!r}")
            elif m_key == "text_27b" and m_name != VERIFIER_Q3_MODEL:
                raise ValueError(f"Historical protocol model text_27b name mismatch: expected {VERIFIER_Q3_MODEL!r}, got {m_name!r}")
            elif m_key == "embedding_4b" and m_name != "qwen3-embedding:4b":
                raise ValueError(f"Historical protocol model embedding_4b name mismatch: expected 'qwen3-embedding:4b', got {m_name!r}")

            m_digest = m_info.get("digest")
            if not isinstance(m_digest, str) or isinstance(m_digest, bool) or len(m_digest) != 64 or not SHA256_HEX_PATTERN.fullmatch(m_digest):
                raise ValueError(f"Historical protocol model {m_key!r} invalid digest format: {m_digest!r}")
            if m_key == "generator_9b":
                if m_digest != PINNED_9B_DIGEST:
                    raise ValueError(f"Historical protocol model generator_9b digest mismatch: expected {PINNED_9B_DIGEST!r}, got {m_digest!r}")
            elif m_key == "text_27b":
                if m_digest not in SUPPORTED_27B_DIGESTS:
                    raise ValueError(f"Historical protocol model text_27b digest mismatch: expected one of {SUPPORTED_27B_DIGESTS!r}, got {m_digest!r}")
            elif m_key == "embedding_4b":
                if m_digest != PINNED_EMBEDDING_DIGEST:
                    raise ValueError(f"Historical protocol model embedding_4b digest mismatch: expected {PINNED_EMBEDDING_DIGEST!r}, got {m_digest!r}")

        _validate_options_dict(
            protocol_data.get("effective_generation_options"),
            HISTORICAL_EFFECTIVE_GENERATION_OPTIONS,
            "Historical effective_generation_options",
        )
        _validate_options_dict(
            protocol_data.get("canonical_role_parameters"),
            HISTORICAL_CANONICAL_ROLE_PARAMETERS,
            "Historical canonical_role_parameters",
        )
        _validate_options_dict(
            protocol_data.get("verifier_parameters"),
            VERIFIER_Q3_PARAMETERS,
            "Historical verifier_parameters",
        )
    else:
        structured_mode = protocol_data.get('schema_version') == STRUCTURED_PROTOCOL_SCHEMA_VERSION
        if protocol_data.get("schema_version") not in (PROTOCOL_SCHEMA_VERSION, "covered-pipeline-protocol-v2", STRUCTURED_PROTOCOL_SCHEMA_VERSION):
            raise ValueError(f"Covered protocol schema mismatch: {protocol_data.get('schema_version')!r}")
        correction_mode = protocol_data.get("schema_version") == "covered-pipeline-protocol-v2" or (structured_mode and protocol_data.get('coverage_correction') is True)
        if structured_mode:
            from rag_pensiones.structured_retrieval import validate_source_manifest
            allowed = {'schema_version','server_version','generator_digest','verifier_digest','embedding_digest','generator_parameters','verifier_parameters','index_path','queries_path','units_path','pages_path','frozen_files','input_hashes','structured_retrieval','coverage_correction'}
            if set(protocol_data)-allowed or ('coverage_correction'in protocol_data and protocol_data['coverage_correction']is not True):
                raise ValueError('Invalid structured v3 protocol shape')
            validate_source_manifest(protocol_data.get('structured_retrieval'))
            for path, expected in protocol_data['structured_retrieval']['files'].items():
                if protocol_data.get('frozen_files', {}).get(path) != expected:
                    raise ValueError('Structured asset pin disagrees with source manifest: ' + path)
            for path in ['rag_pensiones/structured_retrieval.py','experiments/source_recovery_policy.py']:
                if protocol_data.get('frozen_files', {}).get(path) != file_sha256(PROJECT_ROOT/path):
                    raise ValueError('Structured replay code differs from frozen protocol: ' + path)
            if results_data.get('metadata',{}).get('structured_retrieval')!=protocol_data['structured_retrieval']:
                raise ValueError('Structured results manifest missing or changed')
            if correction_mode and results_data.get('metadata',{}).get('coverage_correction')is not True:
                raise ValueError('Structured results missing actual correction flag')
        elif correction_mode:
            expected_keys = {"schema_version", "server_version", "generator_digest", "verifier_digest",
                             "embedding_digest", "generator_parameters", "verifier_parameters", "index_path",
                             "queries_path", "units_path", "pages_path", "frozen_files", "input_hashes", "coverage_correction"}
            if set(protocol_data) != expected_keys or protocol_data.get("coverage_correction") is not True:
                raise ValueError("Invalid v2 correction protocol shape/flag")
            if results_data.get("metadata", {}).get("coverage_correction") is not True:
                raise ValueError("Results missing actual coverage_correction flag")
        elif "coverage_correction" in protocol_data or "coverage_correction" in results_data.get("metadata", {}):
            raise ValueError("v1 forbids correction flag")
        calc_proto_hash = canonical_hash(protocol_data)
        meta_proto_hash = results_data.get("metadata", {}).get("protocol_hash")
        if not meta_proto_hash or meta_proto_hash != calc_proto_hash:
            raise ValueError(f"Covered results metadata protocol_hash mismatch: results has {meta_proto_hash!r}, expected {calc_proto_hash!r}")
        for path_k in ("index_path", "queries_path", "units_path", "pages_path"):
            if not protocol_data.get(path_k) or not isinstance(protocol_data[path_k], str):
                raise ValueError(f"Covered protocol missing mandatory path key: {path_k!r}")
            if "gold" in protocol_data[path_k].lower():
                raise ValueError(f"Gold paths are forbidden in runner protocol: {protocol_data[path_k]!r}")
        pages_rel = protocol_data["pages_path"]
        queries_rel = protocol_data["queries_path"]
        index_rel = protocol_data["index_path"]
        units_rel = protocol_data["units_path"]

        # Validate recorded covered server, models, digests and options against supported profiles
        srv = protocol_data.get("server_version")
        if srv != REQUIRED_SERVER_VERSION:
            raise ValueError(f"Covered protocol server_version mismatch: expected {REQUIRED_SERVER_VERSION!r}, got {srv!r}")
        for dig_k, exp_val in (
            ("generator_digest", PINNED_9B_DIGEST),
            ("verifier_digest", SUPPORTED_27B_DIGESTS),
            ("embedding_digest", PINNED_EMBEDDING_DIGEST),
        ):
            act_dig = protocol_data.get(dig_k)
            if not isinstance(act_dig, str) or isinstance(act_dig, bool) or len(act_dig) != 64 or not SHA256_HEX_PATTERN.fullmatch(act_dig):
                raise ValueError(f"Covered protocol {dig_k} invalid format: {act_dig!r}")
            if isinstance(exp_val, (tuple, list, set)):
                if act_dig not in exp_val:
                    raise ValueError(f"Covered protocol {dig_k} mismatch: expected one of {exp_val!r}, got {act_dig!r}")
            else:
                if act_dig != exp_val:
                    raise ValueError(f"Covered protocol {dig_k} mismatch: expected {exp_val!r}, got {act_dig!r}")
        _validate_options_dict(
            protocol_data.get("generator_parameters"),
            DEFAULT_PARAMETERS,
            "Covered generator_parameters",
        )
        _validate_options_dict(
            protocol_data.get("verifier_parameters"),
            VERIFIER_Q3_PARAMETERS,
            "Covered verifier_parameters",
        )

    # Schema-appropriate manifests requirement:
    # Historical requires both input_hashes + frozen_files.
    # Covered requires frozen_files (complete 4 input byte pins).
    frozen_files = protocol_data.get("frozen_files")
    if not isinstance(frozen_files, dict) or not frozen_files:
        raise ValueError("Protocol frozen_files manifest is missing or empty")

    input_hashes = protocol_data.get("input_hashes")
    if schema_ver == HISTORICAL_RESULTS_SCHEMA_VERSION:
        if not isinstance(input_hashes, dict) or not input_hashes:
            raise ValueError("Historical protocol missing mandatory input_hashes manifest")
        for k in ("pages", "queries", "index", "units"):
            if k not in input_hashes or not input_hashes[k]:
                raise ValueError(f"Historical protocol input_hashes missing mandatory key: {k!r}")
    else:
        if input_hashes is not None:
            if not isinstance(input_hashes, dict) or not input_hashes:
                raise ValueError("Covered protocol input_hashes manifest is empty or invalid")
            for k in ("pages", "queries", "index", "units"):
                if k not in input_hashes or not input_hashes[k]:
                    raise ValueError(f"Covered protocol input_hashes missing mandatory key: {k!r}")

    # Resolve all 4 input files
    p_pgs = (PROJECT_ROOT / pages_rel).resolve() if not Path(pages_rel).is_absolute() else Path(pages_rel)
    p_qry = (PROJECT_ROOT / queries_rel).resolve() if not Path(queries_rel).is_absolute() else Path(queries_rel)
    p_idx = (PROJECT_ROOT / index_rel).resolve() if not Path(index_rel).is_absolute() else Path(index_rel)
    p_unt = (PROJECT_ROOT / units_rel).resolve() if not Path(units_rel).is_absolute() else Path(units_rel)

    input_files: dict[str, tuple[str, Path]] = {
        "pages": (pages_rel, p_pgs),
        "queries": (queries_rel, p_qry),
        "index": (index_rel, p_idx),
        "units": (units_rel, p_unt),
    }

    for k, (rel, p_file) in input_files.items():
        if not p_file.is_file():
            raise FileNotFoundError(f"Required {k} file not found: {p_file}")

    # Load and parse immutable manifests
    pages_lines = [strict_load_json(line) for line in p_pgs.read_text(encoding="utf-8").splitlines() if line.strip()]
    queries_obj = strict_load_json(p_qry.read_bytes())
    index_obj = strict_load_json(p_idx.read_bytes())
    units_obj = strict_load_json(p_unt.read_bytes())

    parsed_inputs: dict[str, Any] = {
        "pages": pages_lines,
        "queries": queries_obj,
        "index": index_obj,
        "units": units_obj,
    }

    # Verify input_hashes when present (preserves exact canonical hash error messages)
    if input_hashes is not None:
        for k, parsed_obj in parsed_inputs.items():
            exp_h = input_hashes[k]
            act_canon = canonical_hash(parsed_obj)
            if schema_ver == HISTORICAL_RESULTS_SCHEMA_VERSION:
                act_sha = file_sha256(input_files[k][1])
                if exp_h != act_canon and exp_h != act_sha:
                    raise ValueError(f"Historical input hash mismatch for {k!r}: expected {exp_h!r}, got canonical {act_canon!r} and file SHA {act_sha!r}")
            else:
                if act_canon != exp_h:
                    raise ValueError(f"{k.capitalize()} input canonical hash mismatch: expected {exp_h!r}, got {act_canon!r}")

    # Build set of valid bound input identities from resolved input bindings
    bound_input_paths: set[str] = set()
    for rel_k, p_k in input_files.values():
        bound_input_paths.add(rel_k)
        bound_input_paths.add(str(p_k))
        bound_input_paths.add(p_k.as_posix())
        try:
            rel_proj = p_k.relative_to(PROJECT_ROOT)
            bound_input_paths.add(rel_proj.as_posix())
            bound_input_paths.add(str(rel_proj))
        except ValueError:
            pass

    # Verify frozen_files pins
    mandatory_sources = (
        HISTORICAL_MANDATORY_SOURCE_PATHS
        if schema_ver == HISTORICAL_RESULTS_SCHEMA_VERSION
        else COVERED_MANDATORY_SOURCE_PATHS
    )
    if schema_ver != HISTORICAL_RESULTS_SCHEMA_VERSION and structured_mode:
        mandatory_sources = STRUCTURED_MANDATORY_SOURCE_PATHS + tuple(protocol_data['structured_retrieval']['files'])

    for path_str, sha_val in frozen_files.items():
        if not isinstance(path_str, str) or isinstance(path_str, bool) or not path_str.strip():
            raise ValueError(f"Invalid path in frozen_files manifest: {path_str!r}")
        if not isinstance(sha_val, str) or isinstance(sha_val, bool) or len(sha_val) != 64 or not SHA256_HEX_PATTERN.fullmatch(sha_val):
            raise ValueError(f"Invalid entry in frozen_files manifest: {path_str!r} -> {sha_val!r}")

        # Directory traversal is strictly forbidden in all manifest paths
        norm_path = path_str.replace("\\", "/")
        if "/../" in norm_path or norm_path.startswith("../") or "/.." in norm_path or ".." in norm_path.split("/"):
            raise ValueError(f"Directory traversal in frozen_files manifest rejected: {path_str!r}")

        if path_str in bound_input_paths:
            # Recognized mandatory bound input file: valid absolute Windows or relative paths accepted.
            continue

        # Strict POSIX canonical repo-relative path required for all source modules.
        # Reject backslashes, leading /, leading ./, trailing /
        if "\\" in path_str or path_str.startswith(("/", "./")) or path_str.endswith("/"):
            raise ValueError(f"Path alias or non-canonical path in frozen_files manifest rejected: {path_str!r}")

    missing_sources = [p for p in mandatory_sources if p not in frozen_files]
    if missing_sources:
        raise ValueError(
            f"Protocol frozen_files manifest missing mandatory source paths "
            f"({len(missing_sources)} missing, e.g. {missing_sources[:3]!r})"
        )

    if schema_ver == COVERED_RESULTS_SCHEMA_VERSION:
        for k, (rel, p_file) in input_files.items():
            exp_sha = (
                frozen_files.get(rel)
                or frozen_files.get(str(p_file))
                or frozen_files.get(p_file.as_posix())
            )
            if not exp_sha:
                try:
                    rel_posix = p_file.relative_to(PROJECT_ROOT).as_posix()
                    exp_sha = frozen_files.get(rel_posix)
                except ValueError:
                    pass
            if not exp_sha:
                raise ValueError(f"Mandatory input {k!r} ({rel!r}) missing from protocol frozen_files manifest")
            act_sha = file_sha256(p_file)
            if act_sha != exp_sha:
                raise ValueError(f"{k.capitalize()} file SHA-256 mismatch against frozen_files: expected {exp_sha!r}, got {act_sha!r}")
    else:
        for k, (rel, p_file) in input_files.items():
            exp_sha = (
                frozen_files.get(rel)
                or frozen_files.get(str(p_file))
                or frozen_files.get(p_file.as_posix())
            )
            if exp_sha:
                act_sha = file_sha256(p_file)
                if act_sha != exp_sha:
                    raise ValueError(f"Historical {k} file SHA-256 mismatch against frozen_files: expected {exp_sha!r}, got {act_sha!r}")

    pages_by_num = {p["page_number"]: p for p in pages_lines if isinstance(p, dict) and "page_number" in p}
    frozen_queries = {c["id"]: c["query"] for c in queries_obj.get("cases", []) if isinstance(c, dict) and "id" in c and "query" in c}
    if schema_ver == HISTORICAL_RESULTS_SCHEMA_VERSION and protocol_data.get("schedule"):
        for item in protocol_data["schedule"]:
            if isinstance(item, dict) and "case_id" in item and "query" in item and item.get("arm") == arm:
                frozen_queries[item["case_id"]] = item["query"]

    # Technical error and completion checks at root
    if results_data.get("run_status") != "completed":
        raise ValueError(f"Results run_status is not 'completed', got: {results_data.get('run_status')!r}")
    if results_data.get("status") == "error":
        raise ValueError(f"Results status is 'error': {results_data.get('error')!r}")
    if results_data.get("error"):
        raise ValueError(f"Results root contains error string: {results_data.get('error')!r}")

    # Extract cases and validate bindings
    extracted_cases: list[dict[str, Any]] = []
    if schema_ver == HISTORICAL_RESULTS_SCHEMA_VERSION:
        if arm not in ("mixed", "all27"):
            raise ValueError(f"Arm must be 'mixed' or 'all27' for paired results, got: {arm!r}")
        raw_attempts = results_data.get("attempts", [])
        arm_attempts = [a for a in raw_attempts if isinstance(a, dict) and a.get("arm") == arm]
        if len(arm_attempts) != EXPECTED_CASE_COUNT:
            raise ValueError(f"Expected {EXPECTED_CASE_COUNT} attempts for arm {arm!r}, got {len(arm_attempts)}")
        for idx, a in enumerate(arm_attempts):
            cid = a.get("case_id")
            if cid != EXPECTED_CASE_IDS[idx]:
                raise ValueError(f"Roster mismatch at index {idx}: expected {EXPECTED_CASE_IDS[idx]}, got {cid}")
            if a.get("status") == "error":
                raise ValueError(f"Technical error in attempt {idx} ({cid}): {a.get('error')!r}")
            if a.get("error"):
                raise ValueError(f"Attempt {idx} ({cid}) contains error string: {a.get('error')!r}")
            if a.get("attempted") is False:
                raise ValueError(f"Attempt {idx} ({cid}) was not attempted")

            q = a.get("query", "")
            ref_q = reference_data["cases"][cid]["query"]
            if q != ref_q:
                raise ValueError(f"Altered query rejected for case {cid}: results has {q!r}, reference has {ref_q!r}")
            if cid in frozen_queries and q != frozen_queries[cid]:
                raise ValueError(f"Altered query rejected for case {cid}: results has {q!r}, frozen protocol queries has {frozen_queries[cid]!r}")

            full_result = a.get("factual_outcome", {}).get("full_result")
            factual_pub = a.get("factual_outcome", {}).get("publication") or a.get("publication")
            if not isinstance(full_result, dict) or not isinstance(factual_pub, dict):
                raise ValueError(f"Attempt {idx} ({cid}) missing full_result or publication")

            validate_publication(factual_pub, full_result, preserve=True, allow_partial=True)
            inv = claim_inventory({"outcome": {"full_result": full_result, "publication": factual_pub}})

            hits_raw = a.get("factual_outcome", {}).get("expanded_hits") or a.get("retrieval", {}).get("hits")
            if not hits_raw:
                raise ValueError(f"Attempt {idx} ({cid}) missing required hits for canonical grounding validation")
            hits_objs = _reconstruct_search_hits(hits_raw)
            exp_dig = protocol_data["models"]["generator_9b"]["digest"] if arm == "mixed" else protocol_data["models"]["text_27b"]["digest"]
            exp_model = protocol_data["models"]["generator_9b"]["name"] if arm == "mixed" else protocol_data["models"]["text_27b"]["name"]
            ver_pol = a.get("verification_policy") or a.get("factual_outcome", {}).get("verification_policy")
            validate_grounded_result(full_result, hits_objs, expected_digest=exp_dig, expected_model=exp_model, verification_policy=ver_pol)

            cov_audit = a.get("coverage_audit")
            if not isinstance(cov_audit, dict):
                raise ValueError(f"Attempt {idx} ({cid}) missing coverage_audit dictionary")
            if cov_audit.get("status") == "error":
                raise ValueError(f"Coverage audit error in attempt {idx} ({cid}): {cov_audit.get('error')!r}")
            if cov_audit.get("error"):
                raise ValueError(f"Coverage audit contains error in attempt {idx} ({cid}): {cov_audit.get('error')!r}")
            if cov_audit.get("status") != "completed":
                raise ValueError(f"Coverage audit not completed in attempt {idx} ({cid}): {cov_audit.get('status')!r}")

            cov_pub = a.get("coverage_publication")
            if not isinstance(cov_pub, dict):
                raise ValueError(f"Attempt {idx} ({cid}) missing coverage_publication dictionary")

            # Check skipped validity: legitimate only if empty inventory
            if cov_audit.get("skipped") is True and len(inv) > 0:
                raise ValueError(f"Attempt {idx} ({cid}) coverage audit marked skipped but inventory is non-empty")

            cov_payload = cov_audit.get("validated_payload")
            if not isinstance(cov_payload, dict):
                raise ValueError(f"Attempt {idx} ({cid}) missing validated_payload in coverage audit")
            raw_parts = cov_payload.get("parts")
            if not isinstance(raw_parts, list):
                raise ValueError(f"Attempt {idx} ({cid}) missing parts in coverage audit")
            clean_parts = [
                {
                    "query_span": p.get("query_span"),
                    "decision": p.get("decision"),
                    "claim_ids": p.get("claim_ids"),
                    "reason_code": p.get("reason_code"),
                }
                for p in raw_parts
                if isinstance(p, dict)
            ]
            val_p = validate_coverage_payload({"parts": clean_parts}, q, inv)
            replayed_pub = build_coverage_publication(q, inv, val_p)
            if cov_pub != replayed_pub:
                raise ValueError(f"Coverage publication mismatch in case {cid}: publication does not match replayed coverage payload")
            pub = cov_pub

            prov = extract_evidence_provenance(a)
            extracted_cases.append({
                "id": cid,
                "query": q,
                "status": a.get("status", ""),
                "publication": pub,
                "provenance": prov,
                "formatted_answer": a.get("formatted_answer", ""),
                "wall_seconds": a.get("query_wall_seconds"),
            })
    else:
        if arm is not None and arm != "mixed":
            raise ValueError(f"Arm cannot be {arm!r} for covered-pipeline-v1 results")
        raw_cases = results_data.get("cases", [])
        if len(raw_cases) != EXPECTED_CASE_COUNT:
            raise ValueError(f"Expected {EXPECTED_CASE_COUNT} cases, got {len(raw_cases)}")
        for idx, c in enumerate(raw_cases):
            if not isinstance(c, dict):
                raise ValueError(f"Case {idx} must be a dict")
            cid = c.get("id")
            if cid != EXPECTED_CASE_IDS[idx]:
                raise ValueError(f"Roster mismatch at index {idx}: expected {EXPECTED_CASE_IDS[idx]}, got {cid}")
            if c.get("status") == "error":
                raise ValueError(f"Technical error in case {idx} ({cid}): {c.get('error')!r}")
            if c.get("error"):
                raise ValueError(f"Case {idx} ({cid}) contains error string: {c.get('error')!r}")
            if c.get("attempted") is False:
                raise ValueError(f"Case {idx} ({cid}) was not attempted")

            q = c.get("query", "")
            ref_q = reference_data["cases"][cid]["query"]
            if q != ref_q:
                raise ValueError(f"Altered query rejected for case {cid}: results has {q!r}, reference has {ref_q!r}")
            if cid in frozen_queries and q != frozen_queries[cid]:
                raise ValueError(f"Altered query rejected for case {cid}: results has {q!r}, frozen protocol queries has {frozen_queries[cid]!r}")

            ans = c.get("answer")
            if not isinstance(ans, dict):
                raise ValueError(f"Case {cid} missing valid answer dictionary")
            if ans.get("status") == "error":
                raise ValueError(f"Covered answer error in case {cid}: {ans.get('error')!r}")
            if ans.get("error"):
                raise ValueError(f"Covered answer contains error string in case {cid}: {ans.get('error')!r}")

            factual_rep = ans.get("factual_report", {})
            if not isinstance(factual_rep, dict):
                raise ValueError(f"Case {cid} missing factual_report dict")
            if factual_rep.get("outcome", {}).get("status") == "error":
                raise ValueError(f"Factual grounding error in case {cid}: {factual_rep.get('outcome', {}).get('error')!r}")

            full_result = factual_rep.get("outcome", {}).get("full_result")
            factual_pub = factual_rep.get("outcome", {}).get("publication")
            if not isinstance(full_result, dict) or not isinstance(factual_pub, dict):
                raise ValueError(f"Case {cid} missing full_result or publication in factual report")

            validate_publication(factual_pub, full_result, preserve=True, allow_partial=True)
            inv = claim_inventory(factual_rep)

            hits_raw = factual_rep.get("outcome", {}).get("expanded_hits") or factual_rep.get("retrieval", {}).get("hits") or ans.get("retrieval", {}).get("hits")
            if not hits_raw:
                raise ValueError(f"Case {cid} missing required hits for canonical validation")
            hits_objs = _reconstruct_search_hits(hits_raw)
            exp_dig = protocol_data.get("generator_digest", PINNED_9B_DIGEST)
            ver_pol = factual_rep.get("outcome", {}).get("verification_policy")
            validate_grounded_result(full_result, hits_objs, expected_digest=exp_dig, expected_model="qwen3.5:9b", verification_policy=ver_pol)

            if correction_mode:
                if (ans.get("schema_version") != "preserved-query-coverage-v2"
                    or ans.get("metadata", {}).get("profile") != "preserved_query_coverage_v2"
                    or ans.get("metadata", {}).get("coverage_correction") is not True):
                    raise ValueError(f"Case {cid} correction version/profile mismatch")
                record = ans.get("coverage_correction")
                replayed = replay_coverage_correction(q, factual_rep, ans.get("initial_coverage_audit"), record, pages=pages_lines)
                if record.get("diagnostics", {}).get("error"):
                    raise ValueError(f"Case {cid} technical correction error")
                inv = replayed["inventory"]
                if ans.get("coverage_audit") != record.get("coverage_audit"):
                    raise ValueError(f"Case {cid} final correction audit mismatch")
                semantic_status = ans.get('publication', {}).get('status') if structured_mode else ans.get('status')
                if (semantic_status != record["coverage_audit"]["validated_payload"]["status"]
                    or ans["metadata"].get("coverage_status") != semantic_status
                    or (not structured_mode and c.get("status") != ans.get("status"))):
                    raise ValueError(f"Case {cid} final status mismatch")
                # Account for every real call on the shared clients, once per role.
                from experiments.evaluate_covered_pipeline import _extract_role_stats
                diagnostics = record["diagnostics"]
                outcome = factual_rep["outcome"]
                expected_calls = {
                    "calls": copy.deepcopy(outcome.get("calls", [])),
                    "verifier_calls": copy.deepcopy(outcome.get("verifier_calls", [])),
                    "coverage_calls": copy.deepcopy(ans["initial_coverage_audit"].get("calls", [])),
                }
                for trace_name, role in (("generation", "calls"), ("verification", "verifier_calls"),
                                          ("reaudit", "coverage_calls")):
                    trace = diagnostics.get(trace_name)
                    if isinstance(trace, dict):
                        expected_calls[role].extend(copy.deepcopy(trace.get("client_calls", [])))
                for role, calls in expected_calls.items():
                    if canonical_hash(c.get(role)) != canonical_hash(calls):
                        raise ValueError(f"Case {cid} {role} corrective call accounting mismatch")
                role_stats = {"generator": "calls", "factual_verifier": "verifier_calls", "coverage": "coverage_calls"}
                for role, key in role_stats.items():
                    if c.get("per_role_stats", {}).get(role) != _extract_role_stats(expected_calls[key]):
                        raise ValueError(f"Case {cid} {role} statistics mismatch")
            elif (ans.get("schema_version") != "preserved-query-coverage-v1"
                  or "coverage_correction" in ans or "initial_coverage_audit" in ans
                  or "coverage_correction" in ans.get("metadata", {})):
                raise ValueError(f"Case {cid} unexpected correction lineage/version")

            cov_audit = ans.get("coverage_audit", {})
            if not isinstance(cov_audit, dict):
                raise ValueError(f"Case {cid} missing coverage_audit dict")
            if cov_audit.get("status") == "error":
                raise ValueError(f"Coverage audit error in case {cid}: {cov_audit.get('error')!r}")

            cov_payload = cov_audit.get("validated_payload")
            if not isinstance(cov_payload, dict):
                raise ValueError(f"Case {cid} missing validated_payload in coverage audit")
            raw_parts = cov_payload.get("parts")
            if not isinstance(raw_parts, list):
                raise ValueError(f"Case {cid} missing parts in coverage audit")

            clean_parts = [
                {
                    "query_span": p.get("query_span"),
                    "decision": p.get("decision"),
                    "claim_ids": p.get("claim_ids"),
                    "reason_code": p.get("reason_code"),
                }
                for p in raw_parts
                if isinstance(p, dict)
            ]
            val_payload = validate_coverage_payload({"parts": clean_parts}, q, inv)
            replayed_pub = build_coverage_publication(q, inv, val_payload)
            final_pub = ans.get("publication")
            if final_pub != replayed_pub:
                raise ValueError(f"Coverage publication mismatch in case {cid}: publication does not match replayed coverage payload")

            verified_structured = None
            if structured_mode:
                final_pub, verified_structured = replay_structured_answer(
                    ans, q, index_obj, inv, final_pub, case_status=c.get('status'),
                    formatted_answer=c.get('formatted_answer'))
            elif ('joint_publication'in ans or 'structured_report'in ans or factual_rep.get('retrieval',{}).get('structured')):
                raise ValueError(f'Case {cid} structured data without a versioned source protocol')
            prov = extract_evidence_provenance(c,verified_structured=verified_structured,publication=final_pub)
            extracted_cases.append({
                "id": cid,
                "query": q,
                "status": c.get("status", ""),
                "publication": final_pub,
                "provenance": prov,
                "formatted_answer": c.get("formatted_answer", ""),
                "wall_seconds": c.get("wall_seconds"),
            })

    case_diagnostics: list[dict[str, Any]] = []
    warnings_list: list[str] = []

    for c in extracted_cases:
        cid = c["id"]
        pub = c["publication"]
        prov = c["provenance"]
        ref_case = reference_data["cases"].get(cid)
        diag = evaluate_case_reference(cid, pub, prov, pages_by_num, reference_case=ref_case)
        diag["case_id"] = cid
        diag["query"] = c["query"]
        diag["wall_seconds"] = c.get("wall_seconds")
        case_diagnostics.append(diag)
        for w in diag["warnings"]:
            if w not in warnings_list:
                warnings_list.append(w)

    whole_correct_count = sum(1 for d in case_diagnostics if d["whole_correct"])
    main_direct_count = sum(1 for d in case_diagnostics if d["main_direct_correct"])
    safe_count = sum(1 for d in case_diagnostics if d["safe"])

    # Optional baseline comparison
    regressions: list[str] = []
    if baseline_path is not None:
        p_base = Path(baseline_path)
        if not p_base.is_absolute():
            p_base = (PROJECT_ROOT / p_base).resolve()
        if not p_base.is_file():
            raise FileNotFoundError(f"Baseline file not found: {p_base}")
        base_data = strict_load_json(p_base.read_bytes())
        if not isinstance(base_data, dict):
            raise ValueError("Baseline data must be a JSON object")
        if base_data.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"Baseline schema mismatch: expected {SCHEMA_VERSION!r}, got {base_data.get('schema_version')!r}")
        if base_data.get("reference_sha256") != file_sha256(p_ref):
            raise ValueError(f"Baseline reference SHA mismatch: expected {file_sha256(p_ref)!r}, got {base_data.get('reference_sha256')!r}")
        base_cases_list = base_data.get("case_results")
        if not isinstance(base_cases_list, list) or len(base_cases_list) != EXPECTED_CASE_COUNT:
            raise ValueError(f"Baseline case_results must contain exactly {EXPECTED_CASE_COUNT} items")

        base_cases: dict[str, dict[str, Any]] = {}
        for idx, bc in enumerate(base_cases_list):
            if not isinstance(bc, dict):
                raise ValueError(f"Baseline case result {idx} must be a dict")
            bc_id = bc.get("case_id")
            if bc_id != EXPECTED_CASE_IDS[idx]:
                raise ValueError(f"Baseline case roster mismatch at index {idx}: expected {EXPECTED_CASE_IDS[idx]}, got {bc_id}")
            for bf in ("whole_correct", "main_direct_correct", "safe"):
                if type(bc.get(bf)) is not bool:
                    raise ValueError(f"Baseline case {bc_id} boolean field {bf!r} must be strict bool, got {type(bc.get(bf)).__name__}")
            base_cases[bc_id] = bc

        for d in case_diagnostics:
            cid = d["case_id"]
            bc = base_cases[cid]
            if bc["whole_correct"] and not d["whole_correct"]:
                regressions.append(f"Regression on {cid}: whole_correct dropped from True to False")
            if bc["safe"] and not d["safe"]:
                regressions.append(f"Regression on {cid}: safe dropped from True to False")
            if bc["main_direct_correct"] and not d["main_direct_correct"]:
                regressions.append(f"Regression on {cid}: main_direct_correct dropped from True to False")

    # Overall pass: all 11 whole correct, all safe, and no baseline regressions
    overall_pass = (whole_correct_count == EXPECTED_CASE_COUNT) and (safe_count == EXPECTED_CASE_COUNT) and (not regressions)

    measured_latencies = [c["wall_seconds"] for c in extracted_cases if type(c.get("wall_seconds")) in (int, float)]
    latency_summary = {
        "measured_cases": len(measured_latencies),
        "total_wall_seconds": round(sum(measured_latencies), 3) if measured_latencies else None,
        "mean_wall_seconds": round(sum(measured_latencies) / len(measured_latencies), 3) if measured_latencies else None,
        "min_wall_seconds": round(min(measured_latencies), 3) if measured_latencies else None,
        "max_wall_seconds": round(max(measured_latencies), 3) if measured_latencies else None,
        "note": "Descriptive measured latency only; no hard time threshold applied.",
    }

    evaluator_path = Path(__file__).resolve()
    evaluator_sha256 = file_sha256(evaluator_path)

    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "overall_pass": overall_pass,
        "status": "pass" if overall_pass else "quality_fail",
        "arm": arm,
        "evaluator_sha256": evaluator_sha256,
        "results_path": p_res.relative_to(PROJECT_ROOT).as_posix() if PROJECT_ROOT in p_res.parents else str(p_res),
        "results_sha256": file_sha256(p_res),
        "reference_path": p_ref.relative_to(PROJECT_ROOT).as_posix() if PROJECT_ROOT in p_ref.parents else str(p_ref),
        "reference_sha256": file_sha256(p_ref),
        "protocol_path": p_proto.relative_to(PROJECT_ROOT).as_posix() if PROJECT_ROOT in p_proto.parents else str(p_proto),
        "protocol_sha256": proto_sha256,
        "summary": {
            "total_cases": EXPECTED_CASE_COUNT,
            "whole_correct_count": whole_correct_count,
            "main_direct_count": main_direct_count,
            "safe_count": safe_count,
            "warnings": warnings_list,
            "regressions": regressions,
        },
        "latency_summary": latency_summary,
        "case_results": case_diagnostics,
    }

    p_out.parent.mkdir(parents=True, exist_ok=True)
    p_out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def run(
    output_dir: Path | str,
    *,
    reference_path: Path | str | None = None,
    baseline_path: Path | str | None = None,
    index_path: str | None = None,
    queries_path: str | None = None,
    units_path: str | None = None,
    pages_path: str | None = None,
    coverage_correction: bool = False,
    structured_retrieval: bool = True,
) -> dict[str, Any]:
    if type(coverage_correction) is not bool:
        raise ValueError("coverage_correction must be a strict boolean")
    if type(structured_retrieval) is not bool:
        raise ValueError('structured_retrieval must be a strict boolean')
    out_dir = Path(output_dir)
    if not out_dir.is_absolute():
        out_dir = (PROJECT_ROOT / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    proto_path = out_dir / "protocol.json"
    results_path = out_dir / "results.json"
    quality_path = out_dir / "quality_report.json"

    if proto_path.exists() or results_path.exists() or quality_path.exists():
        raise FileExistsError("Execution refused: one or more destination files already exist in output_dir")

    protocol = prepare(
        out_dir,
        index_path=index_path,
        queries_path=queries_path,
        units_path=units_path,
        pages_path=pages_path,
        structured_retrieval=structured_retrieval,
        **({"coverage_correction": True} if coverage_correction else {}),
    )

    from experiments.evaluate_covered_pipeline import run as covered_run
    from rag_pensiones.search import load_index

    p_idx = (PROJECT_ROOT / protocol["index_path"]).resolve()
    p_qry = (PROJECT_ROOT / protocol["queries_path"]).resolve()
    p_unt = (PROJECT_ROOT / protocol["units_path"]).resolve()
    p_pgs = (PROJECT_ROOT / protocol["pages_path"]).resolve()

    index_data = load_index(p_idx)
    queries_data = strict_load_json(p_qry.read_bytes())
    units_data = strict_load_json(p_unt.read_bytes())
    pages_data = [strict_load_json(line) for line in p_pgs.read_text(encoding="utf-8").splitlines() if line.strip()]

    covered_run(
        index_data,
        queries_data,
        units_data,
        pages_data,
        results_path,
        generator_digest=protocol["generator_digest"],
        verifier_digest=protocol["verifier_digest"],
        embedding_digest=protocol["embedding_digest"],
        transport=None,
        protocol=protocol,
    )

    ref_path = Path(reference_path) if reference_path else (PROJECT_ROOT / DEFAULT_REFERENCE_REL).resolve()
    base_path = Path(baseline_path) if baseline_path else None
    report = check(
        results_path,
        ref_path,
        quality_path,
        baseline_path=base_path,
        protocol_path=proto_path,
    )
    return report


def legacy_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prep_parser = subparsers.add_parser("prepare", help="Prepare a frozen covered-pipeline protocol")
    prep_parser.add_argument("--output-dir", type=Path, required=True, help="Directory to write protocol.json")

    chk_parser = subparsers.add_parser("check", help="Offline reference regression gate")
    chk_parser.add_argument("--results", type=Path, required=True, help="Path to results JSON")
    chk_parser.add_argument("--reference", type=Path, required=True, help="Path to reference.json")
    chk_parser.add_argument("--output", type=Path, required=True, help="Path to destination report JSON")
    chk_parser.add_argument("--arm", type=str, choices=["mixed", "all27"], default=None, help="Arm for paired comparisons")
    chk_parser.add_argument("--baseline", type=Path, default=None, help="Optional baseline report JSON")
    chk_parser.add_argument("--protocol", type=Path, default=None, help="Optional explicit protocol JSON")

    run_parser = subparsers.add_parser("run", help="Prepare, execute live inference, and score quality")
    run_parser.add_argument("--output-dir", type=Path, required=True, help="Directory for all generated artifacts")
    run_parser.add_argument("--reference", type=Path, default=None, help="Optional reference.json path")
    run_parser.add_argument("--baseline", type=Path, default=None, help="Optional baseline report JSON")

    prep_parser.add_argument("--coverage-correction", action="store_true")
    run_parser.add_argument("--coverage-correction", action=argparse.BooleanOptionalAction, default=None,
                            help="Single coverage correction (default for structured runs); opt out with --no-coverage-correction")
    run_parser.add_argument('--legacy-semantic', action='store_true', help='Explicit historical text-only evaluation; default executes the structured v3 pipeline')

    try:
        args = parser.parse_args(argv)
        if args.command == "prepare":
            prepare(args.output_dir, **({"coverage_correction": True} if args.coverage_correction else {}))
            return 0
        elif args.command == "check":
            rep = check(
                args.results,
                args.reference,
                args.output,
                arm=args.arm,
                baseline_path=args.baseline,
                protocol_path=args.protocol,
            )
            return 0 if rep["overall_pass"] else 1
        elif args.command == "run":
            correction = not args.legacy_semantic if args.coverage_correction is None else args.coverage_correction
            rep = run(
                args.output_dir,
                reference_path=args.reference,
                baseline_path=args.baseline,
                coverage_correction=correction,
                structured_retrieval=not args.legacy_semantic,
            )
            return 0 if rep["overall_pass"] else 1
        return 2
    except (ValueError, FileExistsError, FileNotFoundError) as exc:
        sys.stderr.write(f"Validation error: {exc}\n")
        return 2
    except Exception as exc:
        sys.stderr.write(f"Runtime error: {type(exc).__name__}: {exc}\n")
        return 2


def main(argv: list[str] | None = None) -> int:
    """Evaluate the active LangGraph 9B profile; legacy APIs remain archived."""
    from experiments.graph_quality import main as graph_main
    return graph_main(argv)


if __name__ == "__main__":
    sys.exit(main())
