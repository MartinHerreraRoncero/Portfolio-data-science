"""Query coverage audit and publication gate for the RAG pipeline.

Decides semantic correspondence and completeness between the original query
and accepted grounded claims, separating factual grounding certification from query coverage.
Constrained strictly to the evaluated evidence without new factual generation.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import time
from typing import Any

from experiments.preserve_evidence import (
    bound_generation,
    enforce_binding,
    partial_generation_messages,
)
from experiments.preserved_q3_verifier import (
    Q3VerifierClient,
    VERIFIER_Q3_DIGEST,
    VERIFIER_Q3_MODEL,
    VERIFIER_Q3_PARAMETERS,
)
from experiments.query_preserved import answer_preserved
from experiments.source_bound_trial import RecordedGenerator
from rag_pensiones.answer import (
    DEFAULT_GENERATION_MODEL,
    MAX_MESSAGES_BYTES,
    _parse_float,
    _reject_constant,
    parse_chat_response,
    validate_chat_stats,
)
from rag_pensiones.chunking import ChunkRecord
from rag_pensiones.grounding import (
    DEFAULT_PARAMETERS,
    GENERATION_JSON_SCHEMA,
    MAX_CLAIM_REFS,
    MAX_MESSAGES_BYTES_VERIFICATION,
    build_generation_messages,
    build_source_contexts_for_claims,
    build_verification_messages,
    build_verification_schema,
    collapse_spaces,
    normalize_verification_payload,
    validate_generation_payload,
    validate_verification_payload,
)
from rag_pensiones.search import SearchHit

COVERAGE_SCHEMA_VERSION = "query-coverage-v1"
PRESERVED_COVERAGE_SCHEMA_VERSION = "preserved-query-coverage-v1"
COVERAGE_STAGE = "query_coverage"

PINNED_9B_MODEL = DEFAULT_GENERATION_MODEL
PINNED_9B_DIGEST = "56671c2ab9385f9cfcb404638e32cd62d88e3501d44822208363c010179a3c90"
PINNED_9B_PARAMETERS: dict[str, Any] = copy.deepcopy(DEFAULT_PARAMETERS)

PINNED_27B_MODEL = VERIFIER_Q3_MODEL
PINNED_27B_DIGEST = VERIFIER_Q3_DIGEST
PINNED_27B_PARAMETERS: dict[str, Any] = copy.deepcopy(VERIFIER_Q3_PARAMETERS)

ALLOWED_SAMPLING_OPTIONS: frozenset[str] = frozenset({
    "temperature",
    "seed",
    "num_ctx",
    "num_predict",
    "presence_penalty",
    "top_k",
    "top_p",
    "min_p",
    "repeat_penalty",
    "frequency_penalty",
    "typical_p",
    "repeat_last_n",
})


class CorrectionRecordedGenerator(RecordedGenerator):
    """Subclasses RecordedGenerator to capture real wire requests and effective options for coverage correction."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if args or kwargs:
            super(RecordedGenerator, self).__init__(*args, **kwargs)
            self.calls = []
        else:
            super().__init__()
        self.capture_chat_request = True
        self.last_chat_request: dict[str, Any] | None = None

    @property
    def parameters(self) -> dict[str, Any]:
        """Exposes declared generation parameters as an independent copy of DEFAULT_PARAMETERS."""
        return copy.deepcopy(DEFAULT_PARAMETERS)

    def chat(self, messages: list[dict], response_schema: dict) -> dict[str, Any]:
        self.capture_chat_request = True
        self.last_chat_request = None
        calls_before = len(self.calls)
        try:
            return super().chat(messages, response_schema)
        finally:
            if len(self.calls) > calls_before:
                latest_call = self.calls[-1]
                if self.last_chat_request is not None:
                    latest_call["request"] = copy.deepcopy(self.last_chat_request)
                    latest_call["effective_options"] = copy.deepcopy(self.last_chat_request.get("options"))
                else:
                    latest_call["request"] = None
                    latest_call["effective_options"] = None


def _sanitize_diagnostic_data(data: Any) -> Any:
    """Recursively sanitize diagnostic representations for strict JSON compliance (allow_nan=False).

    Converts non-finite floats (NaN, +Inf, -Inf) to their explicit text token representations
    ('NaN', 'Infinity', '-Infinity') without mutating the underlying client raw call logs.
    """
    if isinstance(data, float):
        if math.isnan(data):
            return "NaN"
        if math.isinf(data):
            return "Infinity" if data > 0 else "-Infinity"
        return data
    elif isinstance(data, dict):
        return {k: _sanitize_diagnostic_data(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [_sanitize_diagnostic_data(v) for v in data]
    elif isinstance(data, tuple):
        return tuple(_sanitize_diagnostic_data(v) for v in data)
    elif isinstance(data, (set, frozenset)):
        return [_sanitize_diagnostic_data(v) for v in sorted(data)]
    return data


def _validate_role_options(
    options: Any,
    expected_required: dict[str, Any],
    role_name: str,
    *,
    exact_keys_only: bool = True,
) -> tuple[bool, str | None]:
    if not isinstance(options, dict) or isinstance(options, bool):
        return False, f"{role_name} options must be a dict, got {type(options).__name__}"
    if not isinstance(expected_required, dict) or isinstance(expected_required, bool):
        return False, f"{role_name} expected options must be a dict, got {type(expected_required).__name__}"

    # Require EXACT closed key set: no unexpected options, no missing options
    opt_keys = set(options.keys())
    exp_keys = set(expected_required.keys())

    extra_keys = opt_keys - exp_keys
    if extra_keys:
        return False, f"{role_name} contains unexpected option(s): {sorted(list(extra_keys))!r}"

    missing_keys = exp_keys - opt_keys
    if missing_keys:
        return False, f"{role_name} missing required option(s): {sorted(list(missing_keys))!r}"

    # Check exact typed canonical values and equality
    for k, exp_val in expected_required.items():
        act_val = options[k]
        if isinstance(act_val, bool) or isinstance(exp_val, bool):
            return False, f"{role_name} option {k!r} has invalid boolean type: {act_val!r}"

        if type(exp_val) is int:
            if type(act_val) is not int:
                return False, f"{role_name} option {k!r} must be int, got {type(act_val).__name__}"
            if act_val != exp_val:
                return False, f"{role_name} option {k!r} value mismatch: {act_val!r} != {exp_val!r}"
        elif type(exp_val) is float:
            if type(act_val) is not float:
                return False, f"{role_name} option {k!r} must be float, got {type(act_val).__name__}"
            if not math.isfinite(act_val):
                return False, f"{role_name} option {k!r} must be finite float, got {act_val!r}"
            if act_val != exp_val:
                return False, f"{role_name} option {k!r} value mismatch: {act_val!r} != {exp_val!r}"
        else:
            if type(act_val) is not type(exp_val):
                return False, f"{role_name} option {k!r} type mismatch: {type(act_val).__name__} != {type(exp_val).__name__}"
            if act_val != exp_val:
                return False, f"{role_name} option {k!r} value mismatch: {act_val!r} != {exp_val!r}"

    return True, None


def _is_history_prefix_intact(current_calls: list[Any], history_prefix: list[Any]) -> bool:
    if not isinstance(current_calls, list) or not isinstance(history_prefix, list):
        return False
    if len(current_calls) < len(history_prefix):
        return False
    current_prefix = current_calls[:len(history_prefix)]
    try:
        cur_json = json.dumps(current_prefix, sort_keys=True, allow_nan=False, ensure_ascii=False)
        hist_json = json.dumps(history_prefix, sort_keys=True, allow_nan=False, ensure_ascii=False)
    except (ValueError, TypeError, OverflowError):
        # Non-JSON or non-finite history is strictly invalid; NEVER fallback to permissive Python equality
        return False
    return cur_json == hist_json


def _validate_client_prehistory(client: Any, stage_name: str) -> tuple[bool, str | None, list[Any]]:
    """Validates that a client's prior calls history is strictly JSON-serializable without NaN or Inf before invocation."""
    calls = getattr(client, "calls", None)
    if calls is None:
        return True, None, []
    if not isinstance(calls, list):
        return False, f"{stage_name} client calls log is not a list ({type(calls).__name__})", []
    try:
        json.dumps(calls, sort_keys=True, allow_nan=False, ensure_ascii=False)
    except (ValueError, TypeError, OverflowError) as exc:
        return False, f"{stage_name} pre-existing calls history contains invalid non-JSON or non-finite values: {exc}", []
    return True, None, copy.deepcopy(calls)


def _capture_postcall_observations(
    client: Any,
    calls_before: int,
    history_before: list[Any],
    raw_response: Any,
    trace: dict[str, Any],
    stage_name: str,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """Captures observed postcall history/delta/rawresponse/error/request immediately after invocation."""
    if raw_response is not None:
        trace["raw_response"] = _sanitize_diagnostic_data(copy.deepcopy(raw_response))

    all_calls = getattr(client, "calls", None)
    observed_calls: list[dict[str, Any]] = []
    observed_req: dict[str, Any] | None = None

    if isinstance(all_calls, list):
        if len(all_calls) >= calls_before:
            observed_calls = copy.deepcopy(all_calls[calls_before:])
            if observed_calls and isinstance(observed_calls[0], dict) and isinstance(observed_calls[0].get("request"), dict):
                observed_req = copy.deepcopy(observed_calls[0].get("request"))
        else:
            trace["observed_after_history"] = _sanitize_diagnostic_data(copy.deepcopy(all_calls))
            trace["calls_before"] = calls_before
            trace["observed_history_count"] = len(all_calls)
            trace["history_status"] = "indeterminate_truncation"
            observed_calls = []
            observed_req = None

    sanitized_calls = _sanitize_diagnostic_data(copy.deepcopy(observed_calls))
    if "client_calls" in trace:
        trace["client_calls"] = sanitized_calls
    if "calls" in trace:
        trace["calls"] = sanitized_calls

    if observed_req is not None:
        trace["request"] = _sanitize_diagnostic_data(copy.deepcopy(observed_req))
    elif "request" in trace:
        trace["request"] = None

    return observed_calls, observed_req


def _extract_and_validate_single_fresh_call(
    client: Any,
    calls_before: int,
    history_prefix: list[dict[str, Any]],
    expected_stage: str | None,
    stage_name: str,
) -> tuple[bool, str | None, dict[str, Any] | None, list[dict[str, Any]]]:
    raw_calls = getattr(client, "calls", None)
    if not isinstance(raw_calls, list):
        return False, f"{stage_name} client missing valid calls list log", None, []

    if len(raw_calls) < calls_before:
        return (
            False,
            f"{stage_name} prior calls history was cleared or truncated ({len(raw_calls)} < {calls_before}); fresh call identification is indeterminate",
            None,
            [],
        )

    delta_calls = raw_calls[calls_before:]

    if not _is_history_prefix_intact(raw_calls, history_prefix):
        return False, f"{stage_name} prior calls history prefix was mutated or replaced", None, delta_calls

    if len(delta_calls) == 0:
        return False, f"{stage_name} call did not append a new call log entry", None, []
    if len(delta_calls) > 1:
        return False, f"{stage_name} call appended multiple ({len(delta_calls)}) call log entries; expected exactly 1", None, delta_calls

    fresh_call = delta_calls[0]
    if not isinstance(fresh_call, dict) or isinstance(fresh_call, bool):
        return False, f"{stage_name} fresh call entry is not a dict", None, delta_calls

    if expected_stage is not None:
        act_stage = fresh_call.get("stage")
        if act_stage != expected_stage:
            return False, f"{stage_name} fresh call stage mismatch: expected {expected_stage!r}, got {act_stage!r}", fresh_call, delta_calls

    req = fresh_call.get("request")
    if not isinstance(req, dict) or isinstance(req, bool) or "options" not in req:
        return False, f"{stage_name} call did not provide validated actual request/options log", fresh_call, delta_calls

    return True, None, fresh_call, delta_calls


def _validate_client_request(
    actual_request: Any,
    expected_request: dict[str, Any],
    stage_name: str,
) -> tuple[bool, str | None]:
    """Validates an actual outgoing client request against the frozen expected wire snapshot.

    Reused strictly across Call 1 (generation), Call 2 (verification), and Call 3 (reaudit).
    Enforces canonical JSON byte-for-byte serialization equality (sort_keys=True, allow_nan=False)
    across the entire request structure, rejecting type coercion (1 vs True, 1 vs 1.0) and non-finites.
    """
    if not isinstance(actual_request, dict) or isinstance(actual_request, bool):
        return False, f"{stage_name} actual request must be a dict, got {type(actual_request).__name__}"

    # 1. Closed key set validation
    act_keys = set(actual_request.keys())
    exp_keys = set(expected_request.keys())
    if act_keys != exp_keys:
        extra = act_keys - exp_keys
        missing = exp_keys - act_keys
        return False, f"{stage_name} actual request keys mismatch: missing {sorted(list(missing))!r}, extra {sorted(list(extra))!r}"

    # 2. Model identity check
    act_model = actual_request.get("model")
    exp_model = expected_request.get("model")
    if act_model != exp_model:
        return False, f"{stage_name} actual request model mismatch: expected {exp_model!r}, got {act_model!r}"

    # 3. Stream and think flags
    for flag in ("stream", "think"):
        if actual_request.get(flag) is not expected_request.get(flag):
            return False, f"{stage_name} actual request {flag!r} mismatch: expected {expected_request.get(flag)!r}, got {actual_request.get(flag)!r}"

    # 4. keep_alive check if in expected
    if "keep_alive" in expected_request:
        if actual_request.get("keep_alive") != expected_request.get("keep_alive"):
            return False, f"{stage_name} actual request keep_alive mismatch: expected {expected_request.get('keep_alive')!r}, got {actual_request.get('keep_alive')!r}"

    # 5. Role options validation (finite typed exact equality and canonical options JSON)
    act_options = actual_request.get("options")
    if not isinstance(act_options, dict) or isinstance(act_options, bool):
        return False, f"{stage_name} actual request options invalid: options must be a dict"
    ok_opts, err_opts = _validate_role_options(act_options, expected_request["options"], f"{stage_name} actual request options")
    if not ok_opts:
        return False, f"{stage_name} actual request options invalid: {err_opts}"

    try:
        act_opts_json = json.dumps(act_options, sort_keys=True, allow_nan=False, ensure_ascii=False)
        exp_opts_json = json.dumps(expected_request["options"], sort_keys=True, allow_nan=False, ensure_ascii=False)
    except (ValueError, TypeError) as exc:
        return False, f"{stage_name} actual request options invalid: {exc}"
    if act_opts_json != exp_opts_json:
        return False, f"{stage_name} actual request options invalid: canonical options mismatch"

    # 6. Messages canonical JSON validation (exact string equality, no Python 1==True or 1==1.0 coercion)
    act_msgs = actual_request.get("messages")
    exp_msgs = expected_request.get("messages")
    try:
        act_msgs_json = json.dumps(act_msgs, sort_keys=True, allow_nan=False, ensure_ascii=False)
        exp_msgs_json = json.dumps(exp_msgs, sort_keys=True, allow_nan=False, ensure_ascii=False)
    except (ValueError, TypeError) as exc:
        return False, f"{stage_name} actual request messages invalid: {exc}"
    if act_msgs_json != exp_msgs_json:
        return False, f"{stage_name} actual request messages do not match planned messages"

    # 7. Format schema canonical JSON validation (exact string equality, no Python 1==True or 1==1.0 coercion)
    act_fmt = actual_request.get("format")
    exp_fmt = expected_request.get("format")
    try:
        act_fmt_json = json.dumps(act_fmt, sort_keys=True, allow_nan=False, ensure_ascii=False)
        exp_fmt_json = json.dumps(exp_fmt, sort_keys=True, allow_nan=False, ensure_ascii=False)
    except (ValueError, TypeError) as exc:
        return False, f"{stage_name} actual request format schema invalid: {exc}"
    if act_fmt_json != exp_fmt_json:
        return False, f"{stage_name} actual request format schema does not match planned schema"

    # 8. ENTIRE request canonical JSON byte-for-byte serialization check
    try:
        act_full_json = json.dumps(actual_request, sort_keys=True, allow_nan=False, ensure_ascii=False)
        exp_full_json = json.dumps(expected_request, sort_keys=True, allow_nan=False, ensure_ascii=False)
    except (ValueError, TypeError) as exc:
        return False, f"{stage_name} actual request failed canonical JSON serialization: {exc}"

    if act_full_json != exp_full_json:
        return False, f"{stage_name} actual request canonical JSON mismatch with frozen planned wire snapshot"

    return True, None

ALLOWED_DECISIONS = ("answered", "not_answered")
ALLOWED_REASON_CODES = (
    "covered",
    "scope_mismatch",
    "not_in_evaluated_evidence",
    "missing_from_answer",
)
NON_COVERED_REASON_CODES = (
    "scope_mismatch",
    "not_in_evaluated_evidence",
    "missing_from_answer",
)

UNAVAILABLE_TEMPLATES: dict[str, str] = {
    "scope_mismatch": "Las evidencias evaluadas no corresponden al alcance solicitado en: «{span}».",
    "not_in_evaluated_evidence": "Las evidencias evaluadas no contienen información para responder a: «{span}».",
    "missing_from_answer": "La respuesta fundamentada no contiene datos respaldados para: «{span}».",
}

COVERAGE_SYSTEM_PROMPT = (
    "Auditor de cobertura semántica de consultas RAG sobre pensiones.\n"
    "Tu función es auditar la correspondencia y completitud entre la consulta original y las afirmaciones aceptadas.\n"
    "REGLAS ESTRUCTURALES:\n"
    "1. Divide la consulta original en 1 a 4 partes disjuntas ('query_span') que cubran exactamente todo el texto de la consulta.\n"
    "   - Cada 'query_span' debe ser un fragmento literal, no vacío y ordenado de la consulta original.\n"
    "   - Copia también todo preámbulo, contexto, conjunción y signo: el primer fragmento empieza al inicio de la consulta y el último termina al final. No omitas ninguna palabra.\n"
    "   - Ejemplo de división literal: 'Según el documento, ¿cuándo sucede A y por qué sucede B?' se divide en 'Según el documento, ¿cuándo sucede A' y ' y por qué sucede B?'. El preámbulo pertenece al primer fragmento; no es una necesidad aparte.\n"
    "   - Si la consulta plantea una sola necesidad indivisible, usa un único fragmento con la consulta entera.\n"
    "   - Si la consulta combina varias necesidades (ej. dato conocido y cálculo desconocido, o dos preguntas), separa cada necesidad en un fragmento distinto.\n"
    "   - Opciones o trasfondo (ej. 'medida agregada o garantía individual') piden elegir, no inventar necesidades separadas.\n"
    "2. Para cada parte determina la decisión ('answered' o 'not_answered'):\n"
    "   - 'answered': la parte queda genuinamente respondida en el texto de las afirmaciones aceptadas asociadas, expresando el dato y la relación solicitados. Las citas respaldan la afirmación pero no responden por sí solas lo omitido en el texto. Requiere 'claim_ids' no vacío y reason_code='covered'.\n"
    "   - 'not_answered': no hay afirmaciones cuyo texto responda esta necesidad con exactitud semántica. Si el dato solicitado solo aparece en las citas pero se omitió en el texto de la afirmación, marca 'missing_from_answer'. Requiere 'claim_ids' vacío y reason_code entre: 'scope_mismatch', 'not_in_evaluated_evidence', 'missing_from_answer'.\n"
    "3. CORRESPONDENCIA SEMÁNTICA:\n"
    "   - El texto de la afirmación debe responder explícitamente el dato solicitado con la relación pedida. Si el dato solicitado solo figura en las citas pero se omitió en el texto de la afirmación, la parte NO está respondida (marca 'not_answered' con reason_code='missing_from_answer').\n"
    "   - Verifica correspondencia estricta de: población, magnitud, unidad, periodo/año, observado frente a proyectado, total frente a componente y relaciones causales.\n"
    "   - La mera coincidencia de palabras o citas compartidas es insuficiente si no se expresa la relación solicitada o los datos corresponden a conceptos o periodos distintos.\n"
    "   - Se admiten paráfrasis válidas y respuestas parciales honestas si conservan el significado exacto y responden el dato y la relación pedidos.\n"
    "   - Si los datos evaluados sustituyen una subpoblación, magnitud, unidad u horizonte por otro (ej. totales en vez de mujeres, o 2055 en vez de 2050), marca 'not_answered' con 'scope_mismatch'.\n"
    "4. NO inventes hechos, afirmaciones nuevas ni texto libre de respuesta en la salida."
)


def claim_inventory(factual_report: dict) -> list[dict[str, Any]]:
    """Maps accepted canonical claim_id/text to their publication claims in verified 1:1 order/text.

    Asserts strict alignment between canonical grounding accepted_claims and publication claims.
    Returns inventory rows {claim_id, text, evidence, publication_claim} retaining trusted view/citations verbatim.
    """
    if not isinstance(factual_report, dict) or isinstance(factual_report, bool):
        raise ValueError(f"factual_report must be a dict, got {type(factual_report).__name__}")

    outcome = factual_report.get("outcome", factual_report)
    if not isinstance(outcome, dict) or isinstance(outcome, bool):
        raise ValueError("factual_report missing valid outcome object")

    if outcome.get("status") == "error":
        return []

    full_result = outcome.get("full_result")
    publication = outcome.get("publication")

    if not full_result or not publication:
        return []

    if not isinstance(full_result, dict) or not isinstance(publication, dict):
        raise ValueError("full_result and publication must be dicts")

    grounding = full_result.get("grounding", {})
    if not isinstance(grounding, dict):
        raise ValueError("grounding must be a dict")

    accepted_claims = grounding.get("accepted_claims", [])
    pub_claims = publication.get("claims", [])

    if not isinstance(accepted_claims, list) or not isinstance(pub_claims, list):
        raise ValueError("accepted_claims and publication claims must be lists")

    if not accepted_claims:
        if pub_claims:
            raise ValueError("Alignment mismatch: 0 accepted claims but non-empty publication claims")
        return []

    if len(accepted_claims) != len(pub_claims):
        raise ValueError(
            f"Alignment count mismatch: {len(accepted_claims)} accepted claims != {len(pub_claims)} publication claims"
        )

    inventory: list[dict[str, Any]] = []
    for idx, (accepted, pub) in enumerate(zip(accepted_claims, pub_claims)):
        if not isinstance(accepted, dict) or isinstance(accepted, bool):
            raise ValueError(f"Malformed accepted claim entry at index {idx}")
        if not isinstance(pub, dict) or isinstance(pub, bool):
            raise ValueError(f"Malformed publication claim entry at index {idx}")

        cid = accepted.get("claim_id")
        if not isinstance(cid, str) or not cid.strip():
            raise ValueError(f"Invalid claim_id in accepted claim at index {idx}: {cid!r}")

        acc_text = accepted.get("text")
        pub_text = pub.get("text")
        if not isinstance(acc_text, str) or acc_text != pub_text:
            raise ValueError(
                f"Text alignment mismatch at index {idx} ({cid}): accepted {acc_text!r} != publication {pub_text!r}"
            )

        ev = pub.get("evidence", [])
        if not isinstance(ev, list):
            raise ValueError(f"Invalid evidence list for publication claim {cid}")
        for item in ev:
            if not isinstance(item, dict) or isinstance(item, bool) or "source_id" not in item or "quote" not in item:
                raise ValueError(f"Invalid evidence item in publication claim {cid}: {item!r}")

        inventory.append({
            "claim_id": cid,
            "text": acc_text,
            "evidence": copy.deepcopy(ev),
            "publication_claim": copy.deepcopy(pub),
        })

    return inventory


def serialize_inventory_for_audit(inventory: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Strips private publication metadata and includes only claim_id, text, and literal evidence(source_id, quote)."""
    serialized = []
    for row in inventory:
        clean_ev = []
        for e in row.get("evidence", []):
            clean_ev.append({
                "source_id": str(e["source_id"]),
                "quote": str(e["quote"]),
            })
        serialized.append({
            "claim_id": str(row["claim_id"]),
            "text": str(row["text"]),
            "evidence": clean_ev,
        })
    return serialized


def build_coverage_messages(query: str, inventory: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Builds chat messages for coverage audit, validating byte budget before model contact."""
    if not isinstance(query, str) or isinstance(query, bool) or not query.strip():
        raise ValueError("query must be a non-empty string")
    if not isinstance(inventory, list):
        raise ValueError("inventory must be a list")

    clean_inv = serialize_inventory_for_audit(inventory)
    user_payload = {
        "consulta_original": query,
        "afirmaciones_aceptadas": clean_inv,
    }
    user_content = json.dumps(user_payload, ensure_ascii=False, indent=2)

    messages = [
        {"role": "system", "content": COVERAGE_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    serialized = json.dumps(messages, ensure_ascii=False).encode("utf-8")
    if len(serialized) > MAX_MESSAGES_BYTES_VERIFICATION:
        raise ValueError(
            f"Coverage messages payload exceeds budget of {MAX_MESSAGES_BYTES_VERIFICATION} bytes UTF-8 "
            f"({len(serialized)} bytes). Aborting without contacting model."
        )

    return messages


def build_coverage_schema(query: str, inventory: list[dict[str, Any]]) -> dict[str, Any]:
    """Builds strict JSON response schema for coverage audit."""
    valid_ids = [row["claim_id"] for row in inventory if isinstance(row, dict) and "claim_id" in row]
    part_properties: dict[str, Any] = {
        "query_span": {"type": "string", "minLength": 1},
        "decision": {"type": "string", "enum": list(ALLOWED_DECISIONS)},
        "claim_ids": {
            "type": "array",
            "items": {"type": "string", "enum": valid_ids} if valid_ids else {"type": "string"},
        },
        "reason_code": {
            "type": "string",
            "enum": list(ALLOWED_REASON_CODES),
        },
    }

    return {
        "type": "object",
        "properties": {
            "parts": {
                "type": "array",
                "minItems": 1,
                "maxItems": 4,
                "items": {
                    "type": "object",
                    "properties": part_properties,
                    "required": ["query_span", "decision", "claim_ids", "reason_code"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["parts"],
        "additionalProperties": False,
    }


def validate_coverage_payload(
    payload: dict[str, Any],
    query: str,
    inventory: list[dict[str, Any]],
) -> dict[str, Any]:
    """Strictly validates coverage payload against original query and accepted inventory.

    Enforces deterministic sequential find of query_span fragments covering every non-whitespace char.
    Validates claim ID constraints, reason codes, decision consistency, and derives overall status.
    """
    if not isinstance(payload, dict) or isinstance(payload, bool):
        raise ValueError("Coverage payload must be a dict")
    if set(payload.keys()) != {"parts"}:
        raise ValueError(f"Coverage payload keys mismatch: {sorted(payload.keys())}, expected ['parts']")

    parts = payload.get("parts")
    if not isinstance(parts, list) or isinstance(parts, bool):
        raise ValueError("Coverage parts must be a list")
    if not (1 <= len(parts) <= 4):
        raise ValueError(f"Coverage parts length must be between 1 and 4, got {len(parts)}")

    valid_ids = {row["claim_id"] for row in inventory if isinstance(row, dict) and "claim_id" in row}
    validated_parts: list[dict[str, Any]] = []
    current_pos = 0

    for idx, p in enumerate(parts):
        if not isinstance(p, dict) or isinstance(p, bool):
            raise ValueError(f"Part {idx} must be a dict")
        allowed_keys = {"query_span", "decision", "claim_ids", "reason_code"}
        if set(p.keys()) != allowed_keys:
            raise ValueError(f"Part {idx} keys mismatch: {sorted(p.keys())}, expected {sorted(allowed_keys)}")

        span = p.get("query_span")
        if not isinstance(span, str) or isinstance(span, bool) or not span.strip():
            raise ValueError(f"Part {idx} query_span must be non-empty string, got {span!r}")

        decision = p.get("decision")
        if decision not in ALLOWED_DECISIONS:
            raise ValueError(f"Part {idx} invalid decision {decision!r}, expected {ALLOWED_DECISIONS}")

        reason_code = p.get("reason_code")
        if reason_code not in ALLOWED_REASON_CODES:
            raise ValueError(f"Part {idx} invalid reason_code {reason_code!r}, expected {ALLOWED_REASON_CODES}")

        cids = p.get("claim_ids")
        if not isinstance(cids, list) or isinstance(cids, bool):
            raise ValueError(f"Part {idx} claim_ids must be a list")
        for cid in cids:
            if not isinstance(cid, str) or isinstance(cid, bool):
                raise ValueError(f"Part {idx} claim_ids contains non-string item: {cid!r}")
            if cid not in valid_ids:
                raise ValueError(f"Part {idx} contains unknown claim ID {cid!r} not in accepted inventory")
        if len(cids) != len(set(cids)):
            raise ValueError(f"Part {idx} contains duplicate claim IDs: {cids}")

        if decision == "answered":
            if len(cids) == 0:
                raise ValueError(f"Part {idx} answered decision requires non-empty claim_ids")
            if reason_code != "covered":
                raise ValueError(f"Part {idx} answered decision requires reason_code 'covered', got {reason_code!r}")
        elif decision == "not_answered":
            if len(cids) > 0:
                raise ValueError(f"Part {idx} not_answered decision requires empty claim_ids, got {cids}")
            if reason_code not in NON_COVERED_REASON_CODES:
                raise ValueError(
                    f"Part {idx} not_answered decision requires non-covered reason_code, got {reason_code!r}"
                )

        # Deterministic sequential find
        found_pos = query.find(span, current_pos)
        if found_pos == -1:
            raise ValueError(
                f"Part {idx} query_span {span!r} not found sequentially in original query after index {current_pos}"
            )
        gap = query[current_pos:found_pos]
        if gap.strip() != "":
            raise ValueError(
                f"Non-whitespace characters omitted in query between {current_pos} and {found_pos}: {gap!r}"
            )

        start_char = found_pos
        end_char = found_pos + len(span)
        current_pos = end_char

        part_entry = copy.deepcopy(p)
        part_entry["start_char"] = start_char
        part_entry["end_char"] = end_char
        validated_parts.append(part_entry)

    trailing = query[current_pos:]
    if trailing.strip() != "":
        raise ValueError(
            f"Non-whitespace characters omitted at end of query after index {current_pos}: {trailing!r}"
        )

    all_answered = all(p["decision"] == "answered" for p in validated_parts)
    none_answered = all(p["decision"] == "not_answered" for p in validated_parts)
    if all_answered:
        overall_status = "answered"
    elif none_answered:
        overall_status = "insufficient_evidence"
    else:
        overall_status = "partial"

    return {
        "status": overall_status,
        "parts": validated_parts,
    }


def audit_coverage(
    query: str,
    inventory: list[dict[str, Any]],
    *,
    verifier: Any,
    empty_reason_code: str = "not_in_evaluated_evidence",
) -> dict[str, Any]:
    """Audits coverage using Q3VerifierClient under distinct stage 'query_coverage'.

    Preserves raw request/response/errors/walltime/stats separately from factual grounding.
    If inventory is empty, returns deterministic whole-query not_answered publication without LLM call.
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be non-empty text")
    if not isinstance(inventory, list):
        raise ValueError("inventory must be a list")

    if empty_reason_code not in ("not_in_evaluated_evidence", "missing_from_answer"):
        raise ValueError("Invalid empty_reason_code")

    # If no accepted claims, deterministic whole-query not_answered publication (no new facts)
    if not inventory:
        whole_span = query.strip()
        deterministic_payload = {
            "parts": [
                {
                    "query_span": whole_span,
                    "decision": "not_answered",
                    "claim_ids": [],
                    "reason_code": empty_reason_code,
                }
            ]
        }
        validated = validate_coverage_payload(deterministic_payload, query, inventory)
        return {
            "status": "completed",
            "skipped": True,
            "model": getattr(verifier, "model", VERIFIER_Q3_MODEL) if verifier is not None else VERIFIER_Q3_MODEL,
            "calls": [],
            "wall_seconds": 0.0,
            "validated_payload": validated,
            "raw_response": None,
            "error": None,
        }

    if verifier is None or not hasattr(verifier, "chat"):
        raise ValueError("verifier client with chat method must be provided when inventory is non-empty")

    n_calls_before = len(getattr(verifier, "calls", []))
    started = time.perf_counter()
    raw_response = None
    try:
        messages = build_coverage_messages(query, inventory)
        schema = build_coverage_schema(query, inventory)
        if callable(getattr(verifier, "release_loaded", None)):
            verifier.release_loaded(DEFAULT_GENERATION_MODEL)
        raw_response = verifier.chat(messages, schema, stage=COVERAGE_STAGE)
        wall_seconds = time.perf_counter() - started
        parsed = parse_chat_response(raw_response, verifier.model)
        validated_stats = validate_chat_stats(parsed["stats"])
        validated_payload = validate_coverage_payload(parsed["payload"], query, inventory)

        all_calls = getattr(verifier, "calls", [])
        new_calls = all_calls[n_calls_before:]

        return {
            "status": "completed",
            "skipped": False,
            "model": verifier.model,
            "calls": copy.deepcopy(new_calls),
            "wall_seconds": wall_seconds,
            "stats": validated_stats,
            "validated_payload": validated_payload,
            "raw_response": copy.deepcopy(raw_response),
            "error": None,
        }
    except Exception as exc:
        wall_seconds = time.perf_counter() - started
        all_calls = getattr(verifier, "calls", [])
        new_calls = all_calls[n_calls_before:]
        return {
            "status": "error",
            "skipped": False,
            "model": getattr(verifier, "model", VERIFIER_Q3_MODEL),
            "calls": copy.deepcopy(new_calls),
            "wall_seconds": wall_seconds,
            "stats": None,
            "validated_payload": None,
            "raw_response": copy.deepcopy(raw_response) if raw_response is not None else None,
            "error": str(exc),
        }


def build_coverage_publication(
    query: str,
    inventory: list[dict[str, Any]],
    validated_payload: dict[str, Any],
) -> dict[str, Any]:
    """Builds publication view from validated coverage payload and trusted inventory.

    Subsets/deduplicates ONLY linked trusted publication_claim objects verbatim.
    Missing parts record literal query spans and fixed template messages limited to evaluated evidence.
    """
    if not isinstance(validated_payload, dict):
        raise ValueError("validated_payload must be a dict")
    status = validated_payload.get("status")
    parts = validated_payload.get("parts", [])

    inv_by_cid = {r["claim_id"]: r for r in inventory if isinstance(r, dict) and "claim_id" in r}

    linked_cids: list[str] = []
    for p in parts:
        if p.get("decision") == "answered":
            for cid in p.get("claim_ids", []):
                if cid not in linked_cids:
                    linked_cids.append(cid)

    pub_claims = [copy.deepcopy(inv_by_cid[cid]["publication_claim"]) for cid in linked_cids if cid in inv_by_cid]

    missing_parts = []
    for p in parts:
        if p.get("decision") == "not_answered":
            code = p.get("reason_code", "not_in_evaluated_evidence")
            span = p.get("query_span", "")
            tpl = UNAVAILABLE_TEMPLATES.get(
                code,
                "Las evidencias evaluadas no contienen información suficiente para responder: «{span}».",
            )
            msg = tpl.format(span=span)
            missing_parts.append({
                "query_span": span,
                "reason_code": code,
                "message": msg,
            })

    if status == "insufficient_evidence":
        if missing_parts:
            reason = " ".join(m["message"] for m in missing_parts)
        else:
            reason = "Las evidencias evaluadas no contienen datos para responder a la consulta."
    else:
        reason = ""

    return {
        "schema_version": "coverage-publication-v1",
        "status": status,
        "claims": pub_claims,
        "missing_parts": missing_parts,
        "reason": reason,
        "parts": copy.deepcopy(parts),
    }


def _answer_covered_initial(
    index: dict[str, Any],
    query: str,
    units: dict[str, Any],
    pages: list[dict[str, Any]],
    *,
    embedder: Any | None = None,
    generator: Any | None = None,
    verifier: Any | None = None,
    partial_generation: bool = True,
    _empty_reason_code: str | None = None,
    structured_retrieval: bool = True,
) -> dict[str, Any]:
    """Candidate explicit API executing answer_preserved followed by hot coverage audit.

    Retains the original factual report immutable and produces preserved-query-coverage-v1 schema.
    If factual generation errors or coverage audit fails, preserves raw calls without unchecked publication.
    """
    if type(partial_generation) is not bool:
        raise ValueError("partial_generation must be a strict boolean")

    if verifier is None:
        verifier = Q3VerifierClient(expected_digest=VERIFIER_Q3_DIGEST)

    preserved_kwargs: dict[str, Any] = {
        "embedder": embedder,
        "generator": generator,
        "verifier": verifier,
    }
    if partial_generation:
        preserved_kwargs["partial_generation"] = True
    if not structured_retrieval:
        preserved_kwargs['structured_retriever'] = False

    factual_report = answer_preserved(
        index,
        query,
        units,
        pages,
        **preserved_kwargs,
    )

    if factual_report.get("outcome", {}).get("status") == "error":
        err = factual_report["outcome"].get("error", "Factual grounding error")
        metadata = {
            "profile": "preserved_query_coverage_v1",
            "factual_status": "error",
            "coverage_status": "error",
        }
        if partial_generation:
            metadata["generation_profile"] = "partial_facts_v1"
            metadata["partial_generation"] = True
        return {
            "schema_version": PRESERVED_COVERAGE_SCHEMA_VERSION,
            "status": "error",
            "query": query,
            "error": f"Factual grounding error: {err}",
            "factual_report": copy.deepcopy(factual_report),
            "coverage_audit": None,
            "publication": {
                "schema_version": "coverage-publication-v1",
                "status": "error",
                "claims": [],
                "missing_parts": [],
                "reason": f"Factual grounding error: {err}",
                "parts": [],
            },
            "metadata": metadata,
        }

    try:
        inventory = claim_inventory(factual_report)
        coverage_audit = audit_coverage(query, inventory, verifier=verifier,
                                        **({"empty_reason_code": _empty_reason_code} if _empty_reason_code else {}))
    except Exception as exc:
        coverage_audit = {"status": "error", "error": str(exc), "raw_response": None}

    if coverage_audit.get("status") == "error":
        err = coverage_audit.get("error", "Coverage audit error")
        metadata = {
            "profile": "preserved_query_coverage_v1",
            "factual_status": factual_report.get("outcome", {}).get("status"),
            "coverage_status": "error",
        }
        if partial_generation:
            metadata["generation_profile"] = "partial_facts_v1"
            metadata["partial_generation"] = True
        return {
            "schema_version": PRESERVED_COVERAGE_SCHEMA_VERSION,
            "status": "error",
            "query": query,
            "error": f"Coverage audit error: {err}",
            "factual_report": copy.deepcopy(factual_report),
            "coverage_audit": copy.deepcopy(coverage_audit),
            "publication": {
                "schema_version": "coverage-publication-v1",
                "status": "error",
                "claims": [],
                "missing_parts": [],
                "reason": f"Coverage audit error: {err}",
                "parts": [],
            },
            "metadata": metadata,
        }

    publication = build_coverage_publication(query, inventory, coverage_audit["validated_payload"])

    metadata = {
        "profile": "preserved_query_coverage_v1",
        "factual_status": factual_report.get("outcome", {}).get("status"),
        "coverage_status": publication["status"],
        "verifier_model": getattr(verifier, "model", VERIFIER_Q3_MODEL),
        "verifier_digest": getattr(verifier, "expected_digest", VERIFIER_Q3_DIGEST),
    }
    if partial_generation:
        metadata["generation_profile"] = "partial_facts_v1"
        metadata["partial_generation"] = True

    return {
        "schema_version": PRESERVED_COVERAGE_SCHEMA_VERSION,
        "status": publication["status"],
        "query": query,
        "factual_report": copy.deepcopy(factual_report),
        "coverage_audit": copy.deepcopy(coverage_audit),
        "publication": publication,
        "metadata": metadata,
    }


def answer_covered(index, query, units, pages, *, embedder=None, generator=None,
                   verifier=None, partial_generation=True, coverage_correction=False, structured_retrieval=True):
    if type(structured_retrieval) is not bool:
        raise ValueError('structured_retrieval must be a strict boolean')
    report = _answer_covered_semantic(index, query, units, pages, embedder=embedder,
        generator=generator, verifier=verifier, partial_generation=partial_generation,
        coverage_correction=coverage_correction,
        **({'structured_retrieval': False} if not structured_retrieval else {}))
    structured = report.get('factual_report', {}).get('retrieval', {}).get('structured')
    if structured_retrieval:
        report['metadata']['structured_profile'] = 'certified_lookup_parallel_v1'
    if structured and structured.get('status') != 'not_applicable':
        from rag_pensiones.structured_retrieval import compose_publication
        report['structured_report'] = copy.deepcopy(structured)
        if report.get('status') == 'error':
            return report
        try:
            inventory = claim_inventory(report['factual_report'])
            if coverage_correction:
                inventory = replay_coverage_correction(query, report['factual_report'],
                    report['initial_coverage_audit'], report['coverage_correction'], pages=pages)['inventory']
            joint = compose_publication(query, report['publication'], structured,
                inventory, index)
            report['joint_publication'] = joint
            report['status'] = joint['status']
            report['metadata']['joint_coverage_status'] = joint['status']
        except Exception as exc:
            report['joint_publication'] = {'status': 'error', 'claims': [], 'missing_parts': [], 'reason': str(exc)}
            report['status'] = 'error'; report['error'] = 'Structured source validation error: ' + str(exc)
    return report


def _answer_covered_semantic(index, query, units, pages, *, embedder=None, generator=None,
                   verifier=None, partial_generation=True, coverage_correction=False, structured_retrieval=True):
    """Run the legacy flow, optionally correcting coverage once before publication."""
    if type(coverage_correction) is not bool:
        raise ValueError("coverage_correction must be a strict boolean")
    if not coverage_correction:
        return _answer_covered_initial(index, query, units, pages, embedder=embedder,
                                      generator=generator, verifier=verifier,
                                      partial_generation=partial_generation,
                                      **({'structured_retrieval': False} if not structured_retrieval else {}))
    if type(partial_generation) is not bool:
        raise ValueError("partial_generation must be a strict boolean")
    generator = generator if generator is not None else CorrectionRecordedGenerator()
    verifier = verifier if verifier is not None else Q3VerifierClient(expected_digest=VERIFIER_Q3_DIGEST)
    report = _answer_covered_initial(
        index, query, units, pages, embedder=embedder, generator=generator,
        verifier=verifier, partial_generation=partial_generation,
        _empty_reason_code="missing_from_answer",
        **({'structured_retrieval': False} if not structured_retrieval else {}))
    report["schema_version"] = "preserved-query-coverage-v2"
    report["metadata"].update(profile="preserved_query_coverage_v2", coverage_correction=True)
    report["initial_coverage_audit"] = None
    report["coverage_correction"] = None
    if report["status"] == "error":
        return report
    initial = copy.deepcopy(report["coverage_audit"])
    report["initial_coverage_audit"] = initial
    record = correct_coverage_once(
        query, report["factual_report"], claim_inventory(report["factual_report"]), initial,
        pages=pages, generator=generator, verifier=verifier)
    report["coverage_correction"] = record
    try:
        replayed = replay_coverage_correction(query, report["factual_report"], initial, record, pages=pages)
        if record["diagnostics"].get("error"):
            raise ValueError(record["diagnostics"]["error"])
        # Replay clocks are synthetic; retain the validated real execution's timings.
        report["coverage_audit"] = copy.deepcopy(record["coverage_audit"])
        report["publication"] = build_coverage_publication(
            query, replayed["inventory"], replayed["coverage_audit"]["validated_payload"])
        report["status"] = report["publication"]["status"]
        report["metadata"]["coverage_status"] = report["status"]
    except Exception as exc:
        report["status"] = "error"
        report["error"] = f"Coverage correction error: {exc}"
        report["metadata"]["coverage_status"] = "error"
        report["publication"] = {"schema_version": "coverage-publication-v1", "status": "error",
                                 "claims": [], "missing_parts": [], "parts": [], "reason": report["error"]}
    return report


def _replay_comparable(value):
    """Validate timings separately; exclude only nondeterministic wall clocks."""
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if key == "wall_seconds":
                if item is not None and (type(item) not in (int, float) or not math.isfinite(item) or item < 0):
                    raise ValueError("Invalid replay timing")
                result[key] = None if item is None else "<validated wall clock>"
            else:
                result[key] = _replay_comparable(item)
        return result
    if isinstance(value, list):
        return [_replay_comparable(item) for item in value]
    return value


def replay_coverage_correction(query, factual_report, initial_coverage_audit, record, *, pages):
    """Reexecute the existing guards with offline, request-asserting recorded transports.

    Proves local trace consistency, not authenticity of a remote model response.
    No accepted flag, candidate inventory or recorded publication is authoritative.
    """
    if not isinstance(record, dict) or set(record) != {"attempted", "applied", "inventory", "coverage_audit", "diagnostics"}:
        raise ValueError("Invalid correction record shape")
    json.dumps(record, allow_nan=False)
    diagnostics = record["diagnostics"]
    if not isinstance(diagnostics, dict):
        raise ValueError("Missing correction diagnostics")
    inventory = claim_inventory(factual_report)
    valid, error, initial_payload = _validate_and_recompute_coverage_audit(initial_coverage_audit, query, inventory)
    if not valid:
        raise ValueError(error)
    if inventory:
        if (initial_coverage_audit.get("model") != PINNED_27B_MODEL
                or initial_coverage_audit.get("error") is not None
                or initial_coverage_audit.get("skipped") is not False):
            raise ValueError("Invalid initial coverage identity/status")
        calls = initial_coverage_audit.get("calls")
        if not isinstance(calls, list) or len(calls) != 1 or calls[0].get("stage") != COVERAGE_STAGE:
            raise ValueError("Invalid initial audit call log")
        expected_request = dict(model=PINNED_27B_MODEL, messages=build_coverage_messages(query, inventory),
                                stream=False, think=False, format=build_coverage_schema(query, inventory),
                                options=copy.deepcopy(PINNED_27B_PARAMETERS))
        valid, error = _validate_client_request(calls[0].get("request"), expected_request, "Initial coverage")
        if not valid:
            raise ValueError(error)
        raw = initial_coverage_audit.get("raw_response")
        if raw != calls[0].get("raw_response") or calls[0].get("error"):
            raise ValueError("Initial audit raw response mismatch")
        parsed = parse_chat_response(raw, PINNED_27B_MODEL)
        if validate_chat_stats(parsed["stats"]) != initial_coverage_audit.get("stats"):
            raise ValueError("Initial coverage statistics mismatch")
        if validate_coverage_payload(parsed["payload"], query, inventory) != initial_payload:
            raise ValueError("Initial audit parsed payload mismatch")
    elif (initial_coverage_audit.get("calls") != [] or initial_coverage_audit.get("raw_response") is not None
          or initial_coverage_audit.get("skipped") is not True
          or initial_payload != audit_coverage(query, [], verifier=None,
              empty_reason_code=initial_payload["parts"][0]["reason_code"])["validated_payload"]):
        raise ValueError("Invalid deterministic empty audit")

    class OfflineClient:
        def __init__(self, generation):
            self.model = PINNED_9B_MODEL if generation else PINNED_27B_MODEL
            self.parameters = copy.deepcopy(PINNED_9B_PARAMETERS if generation else PINNED_27B_PARAMETERS)
            self.expected_digest = PINNED_9B_DIGEST if generation else PINNED_27B_DIGEST
            self.calls = []
            self.pending = [diagnostics.get(k) for k in (("generation",) if generation else ("verification", "reaudit"))]
            self.pending = [t for t in self.pending if isinstance(t, dict) and t.get("client_calls")]
            self.generation = generation

        def get_digest(self):
            return self.expected_digest

        def release_loaded(self, model):
            pass

        def chat(self, messages, response_schema, *, stage=None):
            if not self.pending:
                raise ValueError("Missing recorded corrective call")
            trace = self.pending.pop(0)
            entries = trace["client_calls"]
            if not isinstance(entries, list) or len(entries) != 1:
                raise ValueError("Corrective call count mismatch")
            request = dict(model=self.model, messages=copy.deepcopy(messages), stream=False,
                           think=False, format=copy.deepcopy(response_schema), options=copy.deepcopy(self.parameters))
            if self.generation:
                request["keep_alive"] = "5m"
            valid, error = _validate_client_request(entries[0].get("request"), request, "Offline replay")
            if not valid or (not self.generation and entries[0].get("stage") != stage):
                raise ValueError(error or "Replay stage mismatch")
            self.calls.append(copy.deepcopy(entries[0]))
            if entries[0].get("error"):
                raise RuntimeError(entries[0]["error"])
            raw = entries[0].get("raw_response")
            if raw != trace.get("raw_response"):
                raise ValueError("Trace/raw call response mismatch")
            return copy.deepcopy(raw)

    generator, verifier = OfflineClient(True), OfflineClient(False)
    recomputed = correct_coverage_once(query, factual_report, claim_inventory(factual_report),
                                      initial_coverage_audit, pages=pages, generator=generator, verifier=verifier)
    if generator.pending or verifier.pending:
        raise ValueError("Unused recorded correction calls")
    # Canonical JSON keeps booleans distinct from integer values in all security fields.
    expected = json.dumps(_replay_comparable(recomputed), sort_keys=True, allow_nan=False)
    actual = json.dumps(_replay_comparable(record), sort_keys=True, allow_nan=False)
    if actual != expected:
        raise ValueError("Correction record differs from independently replayed result")
    return recomputed


def format_covered_answer(report: dict[str, Any]) -> str:
    """Renders supported claims and citations alongside every missing part.

    Guarantees visible refusal for omitted parts with coherent JSON and no all-answered label after omission.
    """
    if not isinstance(report, dict):
        return "Error: report must be a dict"

    status = report.get("status")
    if status == "error":
        return "Error: " + str(report.get("error", "Unknown error in covered answer"))

    pub = report.get("joint_publication", report.get("publication"))
    if not isinstance(pub, dict):
        pub = report.get("outcome", {}).get("publication", {})
    if not isinstance(pub, dict):
        return "Error: missing publication view in report"

    pub_status = pub.get("status", status)
    if pub_status == "insufficient_evidence":
        return pub.get("reason", "Evidencias insuficientes para responder a la consulta.")

    sources: dict[str, Any] = {}
    factual = report.get("factual_report", {})
    if isinstance(factual, dict):
        outcome = factual.get("outcome", {})
        if isinstance(outcome, dict) and "expanded_hits" in outcome:
            sources = {f"S{h['rank']}": h['chunk'] for h in outcome['expanded_hits']}

    lines: list[str] = []
    claims = pub.get("claims", [])
    for claim in claims:
        lines.append(claim.get("text", ""))
        for ev in claim.get("evidence", []):
            sid = ev.get("source_id", "")
            quote = ev.get("quote", "")
            if sid in sources:
                src = sources[sid]
                lines.append(f"  [{sid}, pág. {src.get('page_number', '?')}] {quote}")
            else:
                lines.append(f"  [{sid}] {quote}")
        lines.append("")

    missing_parts = pub.get("missing_parts", [])
    if missing_parts:
        lines.append("Partes no respondidas en las fuentes evaluadas:")
        for m in missing_parts:
            lines.append(f"  - {m.get('message', m.get('query_span', ''))}")

    return "\n".join(lines).rstrip()


def _revalidate_selected_evidence(
    factual_report: dict,
    pages: list[dict],
) -> tuple[bool, str | None, list[dict] | None, dict[str, SearchHit] | None, list[dict] | None]:
    """Independently revalidates initial selected_evidence against expanded_hits and canonical pages."""
    try:
        if not isinstance(factual_report, dict) or isinstance(factual_report, bool):
            return False, "factual_report must be a dict", None, None, None
        if not isinstance(pages, list) or isinstance(pages, bool) or not pages:
            return False, "pages must be a non-empty list", None, None, None

        # Build canonical pages map with strict type checking and duplicate page key rejection
        pages_map: dict[tuple[str, int], dict] = {}
        for p in pages:
            if not isinstance(p, dict) or isinstance(p, bool):
                return False, f"Invalid page entry: {p!r}", None, None, None
            doc_id = p.get("doc_id")
            if not isinstance(doc_id, str) or not doc_id.strip():
                return False, f"Page missing valid doc_id: {p!r}", None, None, None
            p_num = p.get("page_number")
            if type(p_num) is not int or isinstance(p_num, bool) or p_num < 1:
                return False, f"Page has invalid page_number: {p_num!r}", None, None, None
            p_text = p.get("text")
            if not isinstance(p_text, str):
                return False, f"Page missing valid text: {p!r}", None, None, None
            p_source_file = p.get("source_file")
            if not isinstance(p_source_file, str) or not p_source_file.strip():
                return False, f"Page missing valid source_file: {p!r}", None, None, None

            page_key = (doc_id, p_num)
            if page_key in pages_map:
                return False, f"Duplicate canonical page key: {page_key}", None, None, None
            pages_map[page_key] = p

        outcome = factual_report.get("outcome", factual_report)
        if not isinstance(outcome, dict) or isinstance(outcome, bool):
            return False, "factual_report missing outcome dict", None, None, None
        full_result = outcome.get("full_result", outcome)
        if not isinstance(full_result, dict) or isinstance(full_result, bool):
            return False, "factual_report missing full_result dict", None, None, None

        grounding = full_result.get("grounding")
        if not isinstance(grounding, dict) or isinstance(grounding, bool):
            return False, "full_result missing grounding dict", None, None, None

        selected_evidence = grounding.get("selected_evidence")
        if not isinstance(selected_evidence, list) or isinstance(selected_evidence, bool) or not (1 <= len(selected_evidence) <= MAX_CLAIM_REFS):
            return (
                False,
                f"selected_evidence must be a list of 1 to {MAX_CLAIM_REFS} items, got {len(selected_evidence) if isinstance(selected_evidence, list) else type(selected_evidence).__name__}",
                None,
                None,
                None,
            )

        repair_evidence = grounding.get("repair_evidence", [])
        repair_eids = {e.get("evidence_id") for e in repair_evidence if isinstance(e, dict) and not isinstance(e, bool)}

        seen_eids = set()
        for ev in selected_evidence:
            if not isinstance(ev, dict) or isinstance(ev, bool):
                return False, "evidence item must be a dict", None, None, None
            eid = ev.get("evidence_id")
            if not isinstance(eid, str) or not eid.strip():
                return False, f"Invalid evidence_id {eid!r}", None, None, None
            if eid in repair_eids:
                return False, f"Selected evidence contains repair evidence ID {eid!r}", None, None, None
            if eid in seen_eids:
                return False, f"Duplicate evidence_id {eid!r} in selected_evidence", None, None, None
            seen_eids.add(eid)

        expanded_hits = outcome.get("expanded_hits")
        if expanded_hits is None:
            expanded_hits = full_result.get("expanded_hits")
        if not isinstance(expanded_hits, list) or isinstance(expanded_hits, bool) or not expanded_hits:
            return False, "expanded_hits missing or empty in factual_report", None, None, None

        hits_by_source: dict[str, SearchHit] = {}
        for h in expanded_hits:
            if isinstance(h, SearchHit):
                rank = h.rank
                score_val = h.score
                chunk = h.chunk
            elif isinstance(h, dict) and not isinstance(h, bool):
                rank = h.get("rank")
                score_val = h.get("score", 0.0)
                cdata = h.get("chunk")
                if isinstance(cdata, ChunkRecord):
                    chunk = cdata
                elif isinstance(cdata, dict) and not isinstance(cdata, bool):
                    for req_key in ("doc_id", "source_file", "page_number", "start_char", "end_char", "text"):
                        if req_key not in cdata:
                            return False, f"Missing chunk {req_key} in hit chunk: {cdata!r}", None, None, None

                    c_doc_id = cdata["doc_id"]
                    c_src = cdata["source_file"]
                    c_page = cdata["page_number"]
                    c_start = cdata["start_char"]
                    c_end = cdata["end_char"]
                    c_text = cdata["text"]

                    if not isinstance(c_doc_id, str) or not c_doc_id:
                        return False, f"Invalid chunk doc_id in hit: {c_doc_id!r}", None, None, None
                    if not isinstance(c_src, str) or not c_src:
                        return False, f"Invalid chunk source_file in hit: {c_src!r}", None, None, None
                    if type(c_page) is not int or isinstance(c_page, bool) or c_page < 1:
                        return False, f"Invalid chunk page_number in hit: {c_page!r}", None, None, None
                    if type(c_start) is not int or isinstance(c_start, bool) or c_start < 0:
                        return False, f"Invalid chunk start_char in hit: {c_start!r}", None, None, None
                    if type(c_end) is not int or isinstance(c_end, bool) or c_end <= c_start:
                        return False, f"Invalid chunk end_char in hit: {c_end!r}", None, None, None
                    if not isinstance(c_text, str):
                        return False, f"Invalid chunk text in hit: {c_text!r}", None, None, None

                    chunk_kwargs = {
                        "chunk_id": str(cdata.get("chunk_id", "chk_0")),
                        "doc_id": c_doc_id,
                        "source_file": c_src,
                        "published_at": str(cdata.get("published_at", "")),
                        "page_number": c_page,
                        "start_char": c_start,
                        "end_char": c_end,
                        "text": c_text,
                        "strategy": str(cdata.get("strategy", "sentence")),
                    }
                    chunk = ChunkRecord(**chunk_kwargs)
                else:
                    return False, f"Invalid chunk data in expanded hit: {cdata!r}", None, None, None
            else:
                return False, f"Invalid hit item in expanded_hits: {h!r}", None, None, None

            # Validate rank
            if type(rank) is not int or isinstance(rank, bool) or rank < 1:
                return False, f"Invalid hit rank: {rank!r}", None, None, None

            # Validate score (reject bool, non-numeric, nan, inf)
            if isinstance(score_val, bool) or not isinstance(score_val, (int, float)) or not math.isfinite(score_val):
                return False, f"Invalid hit score: {score_val!r}", None, None, None
            score = float(score_val)

            # Validate chunk coordinates strictly even for existing ChunkRecord instances
            if not isinstance(chunk, ChunkRecord):
                return False, f"Chunk is not a ChunkRecord: {chunk!r}", None, None, None
            if not isinstance(chunk.doc_id, str) or not chunk.doc_id:
                return False, f"Invalid chunk doc_id: {chunk.doc_id!r}", None, None, None
            if not isinstance(chunk.source_file, str) or not chunk.source_file:
                return False, f"Invalid chunk source_file: {chunk.source_file!r}", None, None, None
            if type(chunk.page_number) is not int or isinstance(chunk.page_number, bool) or chunk.page_number < 1:
                return False, f"Invalid chunk page_number: {chunk.page_number!r}", None, None, None
            if type(chunk.start_char) is not int or isinstance(chunk.start_char, bool) or chunk.start_char < 0:
                return False, f"Invalid chunk start_char: {chunk.start_char!r}", None, None, None
            if type(chunk.end_char) is not int or isinstance(chunk.end_char, bool) or chunk.end_char <= chunk.start_char:
                return False, f"Invalid chunk end_char: {chunk.end_char!r}", None, None, None
            if not isinstance(chunk.text, str):
                return False, f"Invalid chunk text: {chunk.text!r}", None, None, None

            sid = f"S{rank}"
            if sid in hits_by_source:
                return False, f"Duplicate source identity or rank: {sid}", None, None, None
            hits_by_source[sid] = SearchHit(rank=rank, score=score, chunk=chunk)

        source_context_mappings: list[dict] = []

        for ev in selected_evidence:
            sid = ev.get("source_id")
            if not isinstance(sid, str) or sid not in hits_by_source:
                return False, f"Evidence {ev.get('evidence_id')} references unknown source_id {sid!r}", None, None, None
            hit = hits_by_source[sid]

            for key in ("doc_id", "source_file", "page_number"):
                if key in ev and ev[key] != getattr(hit.chunk, key):
                    return (
                        False,
                        f"Evidence {ev.get('evidence_id')} {key} mismatch with hit {sid}: {ev[key]} != {getattr(hit.chunk, key)}",
                        None,
                        None,
                        None,
                    )

            quote = ev.get("quote")
            if not isinstance(quote, str) or not quote.strip():
                return False, f"Evidence {ev.get('evidence_id')} has invalid quote", None, None, None

            norm_quote = collapse_spaces(quote)
            norm_hit_text = collapse_spaces(hit.chunk.text)
            if norm_quote not in norm_hit_text:
                return False, f"Evidence {ev.get('evidence_id')} quote not found in source {sid} text", None, None, None

            page_key = (hit.chunk.doc_id, hit.chunk.page_number)
            page = pages_map.get(page_key)
            if page is None:
                return False, f"Page not found for doc_id {hit.chunk.doc_id}, page {hit.chunk.page_number}", None, None, None
            if page.get("source_file") != hit.chunk.source_file:
                return (
                    False,
                    f"Source file mismatch between hit and canonical page: {hit.chunk.source_file} != {page.get('source_file')}",
                    None,
                    None,
                    None,
                )

            page_text = page.get("text", "")
            if not isinstance(page_text, str):
                return False, f"Canonical page text must be string for {page_key}", None, None, None

            if norm_quote not in collapse_spaces(page_text):
                return False, f"Evidence {ev.get('evidence_id')} quote not found in canonical page text", None, None, None

            s_char = hit.chunk.start_char
            e_char = hit.chunk.end_char
            if type(s_char) is not int or isinstance(s_char, bool) or type(e_char) is not int or isinstance(e_char, bool):
                return False, f"Hit {sid} has non-integer coordinates: [{s_char}:{e_char}]", None, None, None
            if not (0 <= s_char < e_char <= len(page_text)):
                return False, f"Canonical chunk offsets [{s_char}:{e_char}] outside page text length {len(page_text)}", None, None, None
            chunk_slice = page_text[s_char:e_char]
            if collapse_spaces(hit.chunk.text) != collapse_spaces(chunk_slice):
                return False, f"Chunk text does not match canonical page slice at [{s_char}:{e_char}]", None, None, None

            source_context_mappings.append({
                "evidence_id": ev.get("evidence_id"),
                "source_id": sid,
                "doc_id": hit.chunk.doc_id,
                "page_number": hit.chunk.page_number,
                "source_file": hit.chunk.source_file,
                "quote": quote,
                "context_text": collapse_spaces(hit.chunk.text),
            })

        return True, None, copy.deepcopy(selected_evidence), hits_by_source, source_context_mappings
    except Exception as exc:
        return False, f"Malformed data conversion: {exc}", None, None, None


def _validate_initial_inventory(
    inventory: Any,
    factual_report: Any,
) -> tuple[bool, str | None]:
    """Validates that initial inventory aligns strictly with canonical claim_inventory(factual_report)."""
    if not isinstance(inventory, list) or isinstance(inventory, bool):
        return False, "inventory must be a list"
    if not isinstance(factual_report, dict) or isinstance(factual_report, bool):
        return False, "factual_report must be a dict"
    try:
        canonical = claim_inventory(factual_report)
    except Exception as exc:
        return False, f"canonical claim_inventory evaluation failed: {exc}"

    if len(inventory) != len(canonical):
        return (
            False,
            f"Inventory length mismatch with canonical claim_inventory: {len(inventory)} != {len(canonical)}",
        )
    for idx, (inv_item, can_item) in enumerate(zip(inventory, canonical)):
        if not isinstance(inv_item, dict) or isinstance(inv_item, bool):
            return False, f"Inventory item at index {idx} is not a dict"
        if inv_item != can_item:
            return (
                False,
                f"Inventory item at index {idx} does not match canonical claim_inventory item",
            )
    return True, None


def _answered_character_indices(parts_list: list[dict], query_len: int) -> set[int]:
    """Computes the set of query character indices covered by answered parts.

    Strictly validates that coordinates are non-bool integers within [0, query_len].
    Raises ValueError if any answered part has invalid coordinates.
    """
    if not isinstance(parts_list, list) or isinstance(parts_list, bool):
        raise ValueError("parts_list must be a list")
    indices: set[int] = set()
    for idx, p in enumerate(parts_list):
        if not isinstance(p, dict) or isinstance(p, bool):
            raise ValueError(f"Part {idx} is not a valid dict")
        if p.get("decision") == "answered":
            s = p.get("start_char")
            e = p.get("end_char")
            if type(s) is not int or isinstance(s, bool) or type(e) is not int or isinstance(e, bool):
                raise ValueError(f"Part {idx} answered coordinates are not int: [{s!r}:{e!r}]")
            if not (0 <= s < e <= query_len):
                raise ValueError(f"Part {idx} answered coordinates out of bounds: [{s}:{e}] for query_len {query_len}")
            indices.update(range(s, e))
    return indices


def _validate_and_recompute_coverage_audit(
    audit: Any,
    query: str,
    inventory: list[dict[str, Any]],
) -> tuple[bool, str | None, dict[str, Any] | None]:
    """Recomputes and validates coverage audit payload against original query and inventory.

    Requires exact typed agreement with supplied status/offsets/parts.
    Rejects non-integer, boolean, negative, out-of-range, overlap, or gap intervals and unknown/missing keys.
    Does not trust stored payloads.
    """
    if not isinstance(audit, dict) or isinstance(audit, bool):
        return False, "Coverage audit must be a dict", None
    if audit.get("status") != "completed":
        return False, f"Coverage audit status must be 'completed', got {audit.get('status')!r}", None

    val_payload = audit.get("validated_payload")
    if not isinstance(val_payload, dict) or isinstance(val_payload, bool):
        return False, "Coverage audit missing validated_payload dict", None

    if set(val_payload.keys()) != {"status", "parts"}:
        return False, f"validated_payload keys mismatch: {sorted(val_payload.keys())}, expected ['parts', 'status']", None

    supplied_status = val_payload.get("status")
    if supplied_status not in ("answered", "insufficient_evidence", "partial"):
        return False, f"Invalid status in validated_payload: {supplied_status!r}", None

    parts = val_payload.get("parts")
    if not isinstance(parts, list) or isinstance(parts, bool):
        return False, "Coverage audit parts must be a list", None
    if not (1 <= len(parts) <= 4):
        return False, f"Coverage audit parts count must be between 1 and 4, got {len(parts)}", None

    expected_keys = {"query_span", "decision", "claim_ids", "reason_code", "start_char", "end_char"}
    current_pos = 0

    for idx, p in enumerate(parts):
        if not isinstance(p, dict) or isinstance(p, bool):
            return False, f"Part {idx} is not a dict", None
        if set(p.keys()) != expected_keys:
            return False, f"Part {idx} keys mismatch: {sorted(p.keys())}, expected {sorted(expected_keys)}", None

        span = p.get("query_span")
        if not isinstance(span, str) or isinstance(span, bool) or not span.strip():
            return False, f"Part {idx} query_span must be non-empty string, got {span!r}", None

        decision = p.get("decision")
        if decision not in ALLOWED_DECISIONS:
            return False, f"Part {idx} invalid decision {decision!r}", None

        reason_code = p.get("reason_code")
        if reason_code not in ALLOWED_REASON_CODES:
            return False, f"Part {idx} invalid reason_code {reason_code!r}", None

        cids = p.get("claim_ids")
        if not isinstance(cids, list) or isinstance(cids, bool):
            return False, f"Part {idx} claim_ids must be a list", None

        s_char = p.get("start_char")
        e_char = p.get("end_char")
        if type(s_char) is not int or isinstance(s_char, bool) or type(e_char) is not int or isinstance(e_char, bool):
            return False, f"Part {idx} coordinates must be integers, got [{s_char!r}:{e_char!r}]", None
        if not (0 <= s_char < e_char <= len(query)):
            return False, f"Part {idx} coordinates [{s_char}:{e_char}] out of bounds for query length {len(query)}", None

        if query[s_char:e_char] != span:
            return False, f"Part {idx} span {span!r} does not match query slice [{s_char}:{e_char}]: {query[s_char:e_char]!r}", None

        if s_char < current_pos:
            return False, f"Part {idx} overlaps with prior part (start {s_char} < prior end {current_pos})", None
        gap = query[current_pos:s_char]
        if gap.strip() != "":
            return False, f"Part {idx} has non-whitespace gap from prior end {current_pos} to {s_char}: {gap!r}", None
        current_pos = e_char

    trailing = query[current_pos:]
    if trailing.strip() != "":
        return False, f"Trailing non-whitespace characters in query after {current_pos}: {trailing!r}", None

    # Recompute payload using validate_coverage_payload without trusting stored payload
    try:
        raw_parts = [
            {
                "query_span": p["query_span"],
                "decision": p["decision"],
                "claim_ids": list(p["claim_ids"]),
                "reason_code": p["reason_code"],
            }
            for p in parts
        ]
        recomputed = validate_coverage_payload({"parts": raw_parts}, query, inventory)
    except Exception as exc:
        return False, f"Recomputation of coverage payload failed: {exc}", None

    # Verify exact typed agreement with supplied status, offsets, and parts
    if recomputed["status"] != supplied_status:
        return (
            False,
            f"Supplied status {supplied_status!r} does not match recomputed status {recomputed['status']!r}",
            None,
        )
    if len(recomputed["parts"]) != len(parts):
        return (
            False,
            f"Supplied parts count {len(parts)} does not match recomputed count {len(recomputed['parts'])}",
            None,
        )
    for idx, (rec_p, sup_p) in enumerate(zip(recomputed["parts"], parts)):
        for k in expected_keys:
            if rec_p[k] != sup_p[k]:
                return (
                    False,
                    f"Part {idx} key {k!r} mismatch between supplied and recomputed: {sup_p[k]!r} != {rec_p[k]!r}",
                    None,
                )

    return True, None, recomputed


def correct_coverage_once(
    query: str,
    factual_report: dict,
    inventory: list[dict],
    coverage_audit: dict,
    *,
    pages: list[dict],
    generator: Any,
    verifier: Any,
) -> dict[str, Any]:
    """Single-attempt standalone correction helper for omitted data or over-abstention.

    Triggers ONLY when:
    (a) coverage audit is valid with missing_from_answer parts and non-empty selected evidence, OR
    (b) inventory is empty with factual generation insufficiency and non-empty selected evidence.

    Uses strictly initial selected_evidence revalidated against expanded_hits and canonical pages.
    Applies at most:
    - 1 generator (9B) call
    - 1 factual verification (27B) call (only if non-empty valid candidate claims)
    - 1 coverage reaudit (27B) call (only if supported candidate claims)

    Applies only if reaudit interval union retains all previously answered character intervals
    and gains at least one formerly unserved character interval. Otherwise exact original fallback.
    Inputs are treated as immutable.
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be non-empty text")
    if not isinstance(factual_report, dict):
        raise ValueError("factual_report must be a dict")
    if not isinstance(inventory, list):
        raise ValueError("inventory must be a list")
    if generator is None or not hasattr(generator, "chat"):
        raise ValueError("generator client with chat method must be explicitly provided")
    if verifier is None or not hasattr(verifier, "chat"):
        raise ValueError("verifier client with chat method must be explicitly provided")
    if not isinstance(pages, list):
        raise ValueError("pages must be a list of canonical page dicts")

    def _finalize_result(res_dict: dict[str, Any]) -> dict[str, Any]:
        return _sanitize_diagnostic_data(res_dict)

    # Fast safety check on technical errors
    outcome = factual_report.get("outcome", factual_report) if isinstance(factual_report, dict) else {}
    if not isinstance(outcome, dict) or outcome.get("status") == "error":
        return _finalize_result({
            "attempted": False,
            "applied": False,
            "inventory": copy.deepcopy(inventory),
            "coverage_audit": copy.deepcopy(coverage_audit),
            "diagnostics": {
                "attempted": False,
                "applied": False,
                "trigger": None,
                "reason": "Factual report contains technical error or invalid outcome",
                "error": outcome.get("error", "Factual technical error") if isinstance(outcome, dict) else "Invalid outcome",
            },
        })

    if coverage_audit is not None and isinstance(coverage_audit, dict) and coverage_audit.get("status") == "error":
        return _finalize_result({
            "attempted": False,
            "applied": False,
            "inventory": copy.deepcopy(inventory),
            "coverage_audit": copy.deepcopy(coverage_audit),
            "diagnostics": {
                "attempted": False,
                "applied": False,
                "trigger": None,
                "reason": "Coverage audit contains technical error",
                "error": coverage_audit.get("error", "Coverage audit error"),
            },
        })

    # Strict alignment check of initial inventory against canonical claim_inventory
    inv_ok, inv_err = _validate_initial_inventory(inventory, factual_report)
    if not inv_ok:
        return _finalize_result({
            "attempted": False,
            "applied": False,
            "inventory": copy.deepcopy(inventory),
            "coverage_audit": copy.deepcopy(coverage_audit),
            "diagnostics": {
                "attempted": False,
                "applied": False,
                "trigger": None,
                "reason": f"Initial inventory mismatch with canonical claim_inventory: {inv_err}",
                "error": inv_err,
            },
        })

    # Recompute and validate initial coverage audit payload without trusting stored payloads
    recomputed_initial_payload: dict[str, Any] | None = None
    if coverage_audit is not None:
        cov_ok, cov_err, recomputed_initial_payload = _validate_and_recompute_coverage_audit(
            coverage_audit, query, inventory
        )
        if not cov_ok:
            return _finalize_result({
                "attempted": False,
                "applied": False,
                "inventory": copy.deepcopy(inventory),
                "coverage_audit": copy.deepcopy(coverage_audit),
                "diagnostics": {
                    "attempted": False,
                    "applied": False,
                    "trigger": None,
                    "reason": f"Initial coverage audit validation failed: {cov_err}",
                    "error": cov_err,
                },
            })

    # Evaluate Trigger conditions
    full_result = outcome.get("full_result", outcome) if isinstance(outcome, dict) else {}
    grounding = full_result.get("grounding", {}) if isinstance(full_result, dict) else {}
    raw_selected_ev = grounding.get("selected_evidence") if isinstance(grounding, dict) else None
    has_selected_ev = isinstance(raw_selected_ev, list) and len(raw_selected_ev) > 0

    trigger_type: str | None = None
    missing_spans_for_feedback: list[tuple[str, str]] = []

    # Check (a) valid coverage with missing_from_answer
    if recomputed_initial_payload is not None and recomputed_initial_payload.get("status") != "error":
        parts = recomputed_initial_payload.get("parts", [])
        if isinstance(parts, list):
            missing_parts = [
                p for p in parts
                if isinstance(p, dict)
                and p.get("decision") == "not_answered"
                and p.get("reason_code") == "missing_from_answer"
            ]
            if missing_parts and has_selected_ev:
                trigger_type = "missing_from_answer"
                missing_spans_for_feedback = [
                    (p.get("query_span", ""), p.get("reason_code", "missing_from_answer"))
                    for p in missing_parts
                ]

    # Check (b) empty accepted inventory plus factual generation insufficiency and NONEMPTY selected evidence
    if len(inventory) == 0 and has_selected_ev:
        factual_status = outcome.get("status") or full_result.get("status")
        if factual_status == "insufficient_evidence":
            trigger_type = "over_abstention"
            missing_spans_for_feedback = []

    if trigger_type is None:
        return _finalize_result({
            "attempted": False,
            "applied": False,
            "inventory": copy.deepcopy(inventory),
            "coverage_audit": copy.deepcopy(coverage_audit),
            "diagnostics": {
                "attempted": False,
                "applied": False,
                "trigger": None,
                "reason": "No correction trigger condition met (already covered, refused outside evaluated evidence, or empty selected evidence)",
                "error": None,
            },
        })

    # At this point, correction is ATTEMPTED
    # 0. Role metadata validation from factual_report
    metadata = factual_report.get("metadata")
    if not isinstance(metadata, dict) or isinstance(metadata, bool):
        metadata = outcome.get("metadata")
    if not isinstance(metadata, dict) or isinstance(metadata, bool):
        return _finalize_result({
            "attempted": True,
            "applied": False,
            "inventory": copy.deepcopy(inventory),
            "coverage_audit": copy.deepcopy(coverage_audit),
            "diagnostics": {
                "attempted": True,
                "applied": False,
                "trigger": trigger_type,
                "reason": "Original factual_report missing valid metadata dictionary",
                "error": "Missing metadata",
                "model_binding": None,
            },
        })

    # Reject alias keys in metadata
    if "generator_model" in metadata or "generator_digest" in metadata:
        return _finalize_result({
            "attempted": True,
            "applied": False,
            "inventory": copy.deepcopy(inventory),
            "coverage_audit": copy.deepcopy(coverage_audit),
            "diagnostics": {
                "attempted": True,
                "applied": False,
                "trigger": trigger_type,
                "reason": "Original metadata uses alias keys (generator_model/generator_digest); authentic fields 'model' and 'digest' required",
                "error": "Alias keys rejected in metadata",
                "model_binding": None,
            },
        })

    # Validate generator metadata using authentic fields
    meta_gen_model = metadata.get("model")
    if meta_gen_model != PINNED_9B_MODEL:
        return _finalize_result({
            "attempted": True,
            "applied": False,
            "inventory": copy.deepcopy(inventory),
            "coverage_audit": copy.deepcopy(coverage_audit),
            "diagnostics": {
                "attempted": True,
                "applied": False,
                "trigger": trigger_type,
                "reason": f"Original metadata model mismatch: expected {PINNED_9B_MODEL!r}, got {meta_gen_model!r}",
                "error": "Generator model mismatch in metadata",
                "model_binding": None,
            },
        })

    meta_gen_digest = metadata.get("digest")
    if meta_gen_digest != PINNED_9B_DIGEST:
        return _finalize_result({
            "attempted": True,
            "applied": False,
            "inventory": copy.deepcopy(inventory),
            "coverage_audit": copy.deepcopy(coverage_audit),
            "diagnostics": {
                "attempted": True,
                "applied": False,
                "trigger": trigger_type,
                "reason": f"Original metadata digest mismatch: expected {PINNED_9B_DIGEST!r}, got {meta_gen_digest!r}",
                "error": "Generator digest mismatch in metadata",
                "model_binding": None,
            },
        })

    meta_gen_params = metadata.get("generator_parameters")
    ok, err = _validate_role_options(meta_gen_params, PINNED_9B_PARAMETERS, "generator metadata")
    if not ok:
        return _finalize_result({
            "attempted": True,
            "applied": False,
            "inventory": copy.deepcopy(inventory),
            "coverage_audit": copy.deepcopy(coverage_audit),
            "diagnostics": {
                "attempted": True,
                "applied": False,
                "trigger": trigger_type,
                "reason": f"Original metadata generator_parameters invalid: {err}",
                "error": err,
                "model_binding": None,
            },
        })

    # Validate verifier metadata
    meta_verif_model = metadata.get("verifier_model")
    if meta_verif_model != PINNED_27B_MODEL:
        return _finalize_result({
            "attempted": True,
            "applied": False,
            "inventory": copy.deepcopy(inventory),
            "coverage_audit": copy.deepcopy(coverage_audit),
            "diagnostics": {
                "attempted": True,
                "applied": False,
                "trigger": trigger_type,
                "reason": f"Original metadata verifier_model mismatch: expected {PINNED_27B_MODEL!r}, got {meta_verif_model!r}",
                "error": "Verifier model mismatch in metadata",
                "model_binding": None,
            },
        })

    meta_verif_digest = metadata.get("verifier_digest")
    if meta_verif_digest != PINNED_27B_DIGEST:
        return _finalize_result({
            "attempted": True,
            "applied": False,
            "inventory": copy.deepcopy(inventory),
            "coverage_audit": copy.deepcopy(coverage_audit),
            "diagnostics": {
                "attempted": True,
                "applied": False,
                "trigger": trigger_type,
                "reason": f"Original metadata verifier_digest mismatch: expected {PINNED_27B_DIGEST!r}, got {meta_verif_digest!r}",
                "error": "Verifier digest mismatch in metadata",
                "model_binding": None,
            },
        })

    meta_verif_params = metadata.get("verifier_parameters")
    ok, err = _validate_role_options(meta_verif_params, PINNED_27B_PARAMETERS, "verifier metadata")
    if not ok:
        return _finalize_result({
            "attempted": True,
            "applied": False,
            "inventory": copy.deepcopy(inventory),
            "coverage_audit": copy.deepcopy(coverage_audit),
            "diagnostics": {
                "attempted": True,
                "applied": False,
                "trigger": trigger_type,
                "reason": f"Original metadata verifier_parameters invalid: {err}",
                "error": err,
                "model_binding": None,
            },
        })

    # Validate generator client preflight
    client_gen_model = getattr(generator, "model", None)
    if client_gen_model != PINNED_9B_MODEL:
        return _finalize_result({
            "attempted": True,
            "applied": False,
            "inventory": copy.deepcopy(inventory),
            "coverage_audit": copy.deepcopy(coverage_audit),
            "diagnostics": {
                "attempted": True,
                "applied": False,
                "trigger": trigger_type,
                "reason": f"Generator client model mismatch: expected {PINNED_9B_MODEL!r}, got {client_gen_model!r}",
                "error": "Generator client model mismatch",
                "model_binding": None,
            },
        })

    if not callable(getattr(generator, "get_digest", None)):
        return _finalize_result({
            "attempted": True,
            "applied": False,
            "inventory": copy.deepcopy(inventory),
            "coverage_audit": copy.deepcopy(coverage_audit),
            "diagnostics": {
                "attempted": True,
                "applied": False,
                "trigger": trigger_type,
                "reason": "Generator client missing get_digest method",
                "error": "Generator client missing get_digest",
                "model_binding": None,
            },
        })

    try:
        gen_digest_before = generator.get_digest()
    except Exception as exc:
        return _finalize_result({
            "attempted": True,
            "applied": False,
            "inventory": copy.deepcopy(inventory),
            "coverage_audit": copy.deepcopy(coverage_audit),
            "diagnostics": {
                "attempted": True,
                "applied": False,
                "trigger": trigger_type,
                "reason": f"Generator client get_digest failed before call: {exc}",
                "error": str(exc),
                "model_binding": None,
            },
        })
    if gen_digest_before != PINNED_9B_DIGEST:
        return _finalize_result({
            "attempted": True,
            "applied": False,
            "inventory": copy.deepcopy(inventory),
            "coverage_audit": copy.deepcopy(coverage_audit),
            "diagnostics": {
                "attempted": True,
                "applied": False,
                "trigger": trigger_type,
                "reason": f"Generator client digest mismatch before call: expected {PINNED_9B_DIGEST!r}, got {gen_digest_before!r}",
                "error": "Generator digest mismatch",
                "model_binding": None,
            },
        })


    # Validate verifier client preflight
    client_verif_model = getattr(verifier, "model", None)
    if client_verif_model != PINNED_27B_MODEL:
        return _finalize_result({
            "attempted": True,
            "applied": False,
            "inventory": copy.deepcopy(inventory),
            "coverage_audit": copy.deepcopy(coverage_audit),
            "diagnostics": {
                "attempted": True,
                "applied": False,
                "trigger": trigger_type,
                "reason": f"Verifier client model mismatch: expected {PINNED_27B_MODEL!r}, got {client_verif_model!r}",
                "error": "Verifier client model mismatch",
                "model_binding": None,
            },
        })

    if not callable(getattr(verifier, "get_digest", None)):
        return _finalize_result({
            "attempted": True,
            "applied": False,
            "inventory": copy.deepcopy(inventory),
            "coverage_audit": copy.deepcopy(coverage_audit),
            "diagnostics": {
                "attempted": True,
                "applied": False,
                "trigger": trigger_type,
                "reason": "Verifier client missing get_digest method",
                "error": "Verifier client missing get_digest",
                "model_binding": None,
            },
        })

    try:
        verif_digest_before = verifier.get_digest()
    except Exception as exc:
        return _finalize_result({
            "attempted": True,
            "applied": False,
            "inventory": copy.deepcopy(inventory),
            "coverage_audit": copy.deepcopy(coverage_audit),
            "diagnostics": {
                "attempted": True,
                "applied": False,
                "trigger": trigger_type,
                "reason": f"Verifier client get_digest failed before call: {exc}",
                "error": str(exc),
                "model_binding": None,
            },
        })
    if verif_digest_before != PINNED_27B_DIGEST:
        return _finalize_result({
            "attempted": True,
            "applied": False,
            "inventory": copy.deepcopy(inventory),
            "coverage_audit": copy.deepcopy(coverage_audit),
            "diagnostics": {
                "attempted": True,
                "applied": False,
                "trigger": trigger_type,
                "reason": f"Verifier client digest mismatch before call: expected {PINNED_27B_DIGEST!r}, got {verif_digest_before!r}",
                "error": "Verifier digest mismatch",
                "model_binding": None,
            },
        })

    client_verif_params = getattr(verifier, "parameters", None)
    if client_verif_params is not None:
        ok, err = _validate_role_options(client_verif_params, PINNED_27B_PARAMETERS, "verifier client")
        if not ok:
            return _finalize_result({
                "attempted": True,
                "applied": False,
                "inventory": copy.deepcopy(inventory),
                "coverage_audit": copy.deepcopy(coverage_audit),
                "diagnostics": {
                    "attempted": True,
                    "applied": False,
                    "trigger": trigger_type,
                    "reason": f"Verifier client parameters invalid: {err}",
                    "error": err,
                    "model_binding": None,
                },
            })

    # Tracking variables for model binding diagnostics
    gen_digest_after: str | None = None
    verif_digest_after: str | None = None
    actual_gen_opts: dict[str, Any] | None = None
    actual_verif_opts: dict[str, Any] | None = None

    def _make_model_binding() -> dict[str, Any]:
        return {
            "generator": {
                "model_before": client_gen_model,
                "digest_before": gen_digest_before,
                "model_after": getattr(generator, "model", client_gen_model),
                "digest_after": gen_digest_after,
                "effective_options": _sanitize_diagnostic_data(copy.deepcopy(actual_gen_opts)) if actual_gen_opts is not None else None,
            },
            "verifier": {
                "model_before": client_verif_model,
                "digest_before": verif_digest_before,
                "model_after": getattr(verifier, "model", client_verif_model),
                "digest_after": verif_digest_after,
                "effective_options": _sanitize_diagnostic_data(copy.deepcopy(actual_verif_opts)) if actual_verif_opts is not None else None,
            },
        }

    # Validate pre-call history strictly before any LLM invocations
    ok_gen_pre, err_gen_pre, gen_history_before = _validate_client_prehistory(generator, "Generator")
    if not ok_gen_pre:
        return _finalize_result({
            "attempted": True,
            "applied": False,
            "inventory": copy.deepcopy(inventory),
            "coverage_audit": copy.deepcopy(coverage_audit),
            "diagnostics": {
                "attempted": True,
                "applied": False,
                "trigger": trigger_type,
                "reason": f"Pre-call generator history invalid: {err_gen_pre}",
                "error": err_gen_pre,
                "model_binding": _make_model_binding(),
            },
        })

    ok_verif_pre, err_verif_pre, verif_history_before = _validate_client_prehistory(verifier, "Verifier")
    if not ok_verif_pre:
        return _finalize_result({
            "attempted": True,
            "applied": False,
            "inventory": copy.deepcopy(inventory),
            "coverage_audit": copy.deepcopy(coverage_audit),
            "diagnostics": {
                "attempted": True,
                "applied": False,
                "trigger": trigger_type,
                "reason": f"Pre-call verifier history invalid: {err_verif_pre}",
                "error": err_verif_pre,
                "model_binding": _make_model_binding(),
            },
        })

    # 1. Independent revalidation of selected evidence
    is_valid, reval_err, selected_evidence, hits_by_source, source_context_mappings = _revalidate_selected_evidence(
        factual_report, pages
    )
    if not is_valid or selected_evidence is None or hits_by_source is None:
        return _finalize_result({
            "attempted": True,
            "applied": False,
            "inventory": copy.deepcopy(inventory),
            "coverage_audit": copy.deepcopy(coverage_audit),
            "diagnostics": {
                "attempted": True,
                "applied": False,
                "trigger": trigger_type,
                "reason": f"Initial selected evidence revalidation failed closed: {reval_err}",
                "error": reval_err,
                "model_binding": _make_model_binding(),
            },
        })

    # 2. Build generation messages with feedback
    try:
        base_messages = build_generation_messages(query, selected_evidence)
        partial_messages = partial_generation_messages(base_messages)
        bounded_messages, bounded_schema, refs = bound_generation(partial_messages, GENERATION_JSON_SCHEMA, "generation")

        if trigger_type == "missing_from_answer":
            spans_text = "\n".join(f"- «{span}» (motivo: {code})" for span, code in missing_spans_for_feedback)
            prior_claim_texts = [
                row["text"]
                for row in inventory
                if isinstance(row, dict) and isinstance(row.get("text"), str) and row["text"].strip()
            ]
            prior_claims_json = json.dumps(prior_claim_texts, ensure_ascii=False, indent=2)
            feedback_text = (
                "\n\nRetroalimentación de cobertura: las afirmaciones previas omitieron responder a las siguientes partes de la consulta:\n"
                f"{spans_text}\n\n"
                "Textos de afirmaciones previas ya verificadas (DATOS DE DIAGNÓSTICO, NO SON EVIDENCIA NUEVA):\n"
                f"```json\n{prior_claims_json}\n```\n\n"
                "Los textos previos anteriores NO suministran la información solicitada en las partes omitidas de la consulta; "
                "repetirlos o parafrasearlos es insuficiente. "
                "Genera afirmaciones atómicas principales que suministren directamente los datos o relaciones solicitados a partir "
                "exclusivamente del catálogo de evidencias seleccionado original, y no solo en las citas."
            )
            user_msg = next((m for m in bounded_messages if m.get("role") == "user"), None)
            if user_msg is None:
                raise ValueError("User message not found in bounded messages")
            user_msg["content"] += feedback_text
        else:
            feedback_text = (
                "\nRetroalimentación de sobreabstención: la generación anterior declaró insuficiencia total a pesar de disponer de evidencias seleccionadas. "
                "Emite únicamente las afirmaciones atómicas solicitadas respaldadas por las evidencias seleccionadas para aquellas partes que tengan soporte independiente. "
                "La presencia de evidencias seleccionadas no garantiza que la consulta pueda responderse por completo; si ninguna parte tiene respaldo, abstente."
            )
            bounded_messages[0]["content"] += feedback_text
        serialized_gen = json.dumps(bounded_messages, ensure_ascii=False).encode("utf-8")
        if len(serialized_gen) > MAX_MESSAGES_BYTES:
            raise ValueError(
                f"Corrective generation messages exceed budget of {MAX_MESSAGES_BYTES} bytes UTF-8 ({len(serialized_gen)} bytes)"
            )
    except Exception as exc:
        return _finalize_result({
            "attempted": True,
            "applied": False,
            "inventory": copy.deepcopy(inventory),
            "coverage_audit": copy.deepcopy(coverage_audit),
            "diagnostics": {
                "attempted": True,
                "applied": False,
                "trigger": trigger_type,
                "reason": f"Failed constructing generation messages: {exc}",
                "error": str(exc),
                "model_binding": _make_model_binding(),
            },
        })

    # 3. Call 1: 9B Generation
    gen_trace: dict[str, Any] = {
        "stage": "generation",
        "origin": "coverage_correction",
        "model": client_gen_model,
        "prompt_sha256": hashlib.sha256(serialized_gen).hexdigest(),
        "messages": copy.deepcopy(bounded_messages),
        "schema": copy.deepcopy(bounded_schema),
        "bound_refs": copy.deepcopy(refs),
        "raw_response": None,
        "stats": None,
        "wall_seconds": None,
        "validated_payload": None,
        "error": None,
        "request": None,
        "client_calls": [],
    }
    verif_trace: dict[str, Any] | None = None
    reaudit_trace: dict[str, Any] | None = None

    # Freeze immutable expected wire snapshot before external generation invocation
    expected_gen_request = {
        "model": PINNED_9B_MODEL,
        "messages": copy.deepcopy(bounded_messages),
        "stream": False,
        "think": False,
        "keep_alive": "5m",
        "format": copy.deepcopy(bounded_schema),
        "options": copy.deepcopy(PINNED_9B_PARAMETERS),
    }
    frozen_gen_expected = json.loads(json.dumps(expected_gen_request, ensure_ascii=False, allow_nan=False))

    gen_calls_before = len(gen_history_before)
    try:
        if callable(getattr(verifier, "release_loaded", None)):
            verifier.release_loaded(VERIFIER_Q3_MODEL)

        t0 = time.perf_counter()
        raw_gen_resp = None
        try:
            raw_gen_resp = generator.chat(copy.deepcopy(bounded_messages), copy.deepcopy(bounded_schema))
        finally:
            wall_gen = time.perf_counter() - t0
            gen_trace["wall_seconds"] = wall_gen
            _capture_postcall_observations(generator, gen_calls_before, gen_history_before, raw_gen_resp, gen_trace, "Generator")

        try:
            gen_digest_after = generator.get_digest() if callable(getattr(generator, "get_digest", None)) else None
        except Exception as exc:
            raise RuntimeError(f"Generator get_digest failed after call: {exc}")
        if gen_digest_after != PINNED_9B_DIGEST or gen_digest_after != gen_digest_before:
            raise RuntimeError(f"Generator digest changed: {gen_digest_before} -> {gen_digest_after}")

        # Validate generator actual request from calls log against frozen expected wire snapshot
        ok_call, err_call, fresh_gen_call, delta_calls = _extract_and_validate_single_fresh_call(
            generator, gen_calls_before, gen_history_before, None, "Generator"
        )
        if not ok_call:
            raise RuntimeError(err_call)

        actual_gen_opts = fresh_gen_call["request"].get("options")
        ok_req, err_req = _validate_client_request(fresh_gen_call["request"], frozen_gen_expected, "Generator")
        if not ok_req:
            raise RuntimeError(err_req)

        parsed_gen = parse_chat_response(raw_gen_resp, client_gen_model)
        stats_gen = validate_chat_stats(parsed_gen["stats"])
        gen_trace["stats"] = copy.deepcopy(stats_gen)
        val_gen = validate_generation_payload(parsed_gen["payload"], selected_evidence)
        enforce_binding(val_gen, "generation", refs)
        gen_trace["validated_payload"] = copy.deepcopy(val_gen)

        if val_gen.get("status") != "answered" or not val_gen.get("claims"):
            return _finalize_result({
                "attempted": True,
                "applied": False,
                "inventory": copy.deepcopy(inventory),
                "coverage_audit": copy.deepcopy(coverage_audit),
                "diagnostics": {
                    "attempted": True,
                    "applied": False,
                    "trigger": trigger_type,
                    "reason": f"Corrective generation did not produce answered claims (status={val_gen.get('status')})",
                    "error": None,
                    "original_selected_evidence": copy.deepcopy(selected_evidence),
                    "source_context_mappings": source_context_mappings,
                    "generation": gen_trace,
                    "verification": None,
                    "reaudit": None,
                    "model_binding": _make_model_binding(),
                },
            })

        candidate_claims = val_gen["claims"]
    except Exception as exc:
        gen_trace["error"] = str(exc)
        return _finalize_result({
            "attempted": True,
            "applied": False,
            "inventory": copy.deepcopy(inventory),
            "coverage_audit": copy.deepcopy(coverage_audit),
            "diagnostics": {
                "attempted": True,
                "applied": False,
                "trigger": trigger_type,
                "reason": f"Corrective generation failed: {exc}",
                "error": str(exc),
                "original_selected_evidence": copy.deepcopy(selected_evidence),
                "source_context_mappings": source_context_mappings,
                "generation": gen_trace,
                "verification": None,
                "reaudit": None,
                "model_binding": _make_model_binding(),
            },
        })

    # 4. Call 2: 27B Verification
    verif_calls_before = len(getattr(verifier, "calls", []))
    ok_v_pre, err_v_pre, verif_history_before = _validate_client_prehistory(verifier, "Verifier")
    verif_trace = {
        "stage": "verification",
        "origin": "coverage_correction",
        "model": client_verif_model,
        "prompt_sha256": None,
        "messages": None,
        "schema": None,
        "source_contexts": None,
        "raw_response": None,
        "stats": None,
        "wall_seconds": None,
        "validated_payload": None,
        "error": None,
        "request": None,
        "client_calls": [],
    }
    if not ok_v_pre:
        verif_trace["error"] = err_v_pre
        return _finalize_result({
            "attempted": True,
            "applied": False,
            "inventory": copy.deepcopy(inventory),
            "coverage_audit": copy.deepcopy(coverage_audit),
            "diagnostics": {
                "attempted": True,
                "applied": False,
                "trigger": trigger_type,
                "reason": f"Verifier pre-call history invalid: {err_v_pre}",
                "error": err_v_pre,
                "original_selected_evidence": copy.deepcopy(selected_evidence),
                "source_context_mappings": source_context_mappings,
                "generation": gen_trace,
                "verification": verif_trace,
                "reaudit": None,
                "model_binding": _make_model_binding(),
            },
        })

    try:
        source_contexts = build_source_contexts_for_claims(candidate_claims, selected_evidence, hits_by_source)
        verif_trace["source_contexts"] = copy.deepcopy(source_contexts)
        verif_msgs = build_verification_messages(candidate_claims, selected_evidence, source_contexts=source_contexts)
        verif_trace["messages"] = copy.deepcopy(verif_msgs)
        verif_schema = build_verification_schema(candidate_claims)
        verif_trace["schema"] = copy.deepcopy(verif_schema)
        ser_verif = json.dumps(verif_msgs, ensure_ascii=False).encode("utf-8")
        verif_trace["prompt_sha256"] = hashlib.sha256(ser_verif).hexdigest()
        if len(ser_verif) > MAX_MESSAGES_BYTES_VERIFICATION:
            raise ValueError(
                f"Verification messages exceed budget of {MAX_MESSAGES_BYTES_VERIFICATION} bytes UTF-8 ({len(ser_verif)} bytes)"
            )
        # Freeze immutable expected wire snapshot before external verification invocation
        expected_verif_request = {
            "model": PINNED_27B_MODEL,
            "messages": copy.deepcopy(verif_msgs),
            "stream": False,
            "think": False,
            "format": copy.deepcopy(verif_schema),
            "options": copy.deepcopy(PINNED_27B_PARAMETERS),
        }
        frozen_verif_expected = json.loads(json.dumps(expected_verif_request, ensure_ascii=False, allow_nan=False))
    except Exception as exc:
        verif_trace["error"] = str(exc)
        return _finalize_result({
            "attempted": True,
            "applied": False,
            "inventory": copy.deepcopy(inventory),
            "coverage_audit": copy.deepcopy(coverage_audit),
            "diagnostics": {
                "attempted": True,
                "applied": False,
                "trigger": trigger_type,
                "reason": f"Failed constructing verification messages: {exc}",
                "error": str(exc),
                "original_selected_evidence": copy.deepcopy(selected_evidence),
                "source_context_mappings": source_context_mappings,
                "generation": gen_trace,
                "verification": verif_trace,
                "reaudit": None,
                "model_binding": _make_model_binding(),
            },
        })

    try:
        if callable(getattr(verifier, "release_loaded", None)):
            verifier.release_loaded(PINNED_9B_MODEL)

        t1 = time.perf_counter()
        raw_verif_resp = None
        try:
            raw_verif_resp = verifier.chat(copy.deepcopy(verif_msgs), copy.deepcopy(verif_schema), stage="verification")
        finally:
            wall_verif = time.perf_counter() - t1
            verif_trace["wall_seconds"] = wall_verif
            _capture_postcall_observations(verifier, verif_calls_before, verif_history_before, raw_verif_resp, verif_trace, "Verifier")

        try:
            verif_digest_after = verifier.get_digest() if callable(getattr(verifier, "get_digest", None)) else None
        except Exception as exc:
            raise RuntimeError(f"Verifier get_digest failed after call: {exc}")
        if verif_digest_after != PINNED_27B_DIGEST or verif_digest_after != verif_digest_before:
            raise RuntimeError(f"Verifier digest changed: {verif_digest_before} -> {verif_digest_after}")

        # Validate verifier actual request from calls log against frozen expected wire snapshot
        ok_call, err_call, fresh_v_call, delta_calls = _extract_and_validate_single_fresh_call(
            verifier, verif_calls_before, verif_history_before, "verification", "Verifier"
        )
        if not ok_call:
            raise RuntimeError(err_call)

        actual_verif_opts = fresh_v_call["request"].get("options")
        ok_req, err_req = _validate_client_request(fresh_v_call["request"], frozen_verif_expected, "Verifier")
        if not ok_req:
            raise RuntimeError(err_req)

        parsed_verif = parse_chat_response(raw_verif_resp, client_verif_model)
        stats_verif = validate_chat_stats(parsed_verif["stats"])
        verif_trace["stats"] = copy.deepcopy(stats_verif)
        norm_verif = normalize_verification_payload(parsed_verif["payload"], candidate_claims)
        val_verif = validate_verification_payload(norm_verif, candidate_claims, selected_evidence)
        verif_trace["validated_payload"] = copy.deepcopy(val_verif)

        if any(v.get("verdict") == "contradicted" for v in val_verif.get("verdicts", [])):
            return _finalize_result({
                "attempted": True,
                "applied": False,
                "inventory": copy.deepcopy(inventory),
                "coverage_audit": copy.deepcopy(coverage_audit),
                "diagnostics": {
                    "attempted": True,
                    "applied": False,
                    "trigger": trigger_type,
                    "reason": "Corrective candidate claim contradicted evaluated evidence; fail closed on inconsistent claims",
                    "error": None,
                    "original_selected_evidence": copy.deepcopy(selected_evidence),
                    "source_context_mappings": source_context_mappings,
                    "generation": gen_trace,
                    "verification": verif_trace,
                    "reaudit": None,
                    "model_binding": _make_model_binding(),
                },
            })

        supported_v_cids = {v["claim_id"] for v in val_verif.get("verdicts", []) if v.get("verdict") == "supported"}
        supported_claims = [c for c in candidate_claims if c.get("claim_id") in supported_v_cids]

        if not supported_claims:
            return _finalize_result({
                "attempted": True,
                "applied": False,
                "inventory": copy.deepcopy(inventory),
                "coverage_audit": copy.deepcopy(coverage_audit),
                "diagnostics": {
                    "attempted": True,
                    "applied": False,
                    "trigger": trigger_type,
                    "reason": "None of the candidate claims were supported by verification",
                    "error": None,
                    "original_selected_evidence": copy.deepcopy(selected_evidence),
                    "source_context_mappings": source_context_mappings,
                    "generation": gen_trace,
                    "verification": verif_trace,
                    "reaudit": None,
                    "model_binding": _make_model_binding(),
                },
            })
    except Exception as exc:
        verif_trace["error"] = str(exc)
        return _finalize_result({
            "attempted": True,
            "applied": False,
            "inventory": copy.deepcopy(inventory),
            "coverage_audit": copy.deepcopy(coverage_audit),
            "diagnostics": {
                "attempted": True,
                "applied": False,
                "trigger": trigger_type,
                "reason": f"Candidate verification failed: {exc}",
                "error": str(exc),
                "original_selected_evidence": copy.deepcopy(selected_evidence),
                "source_context_mappings": source_context_mappings,
                "generation": gen_trace,
                "verification": verif_trace,
                "reaudit": None,
                "model_binding": _make_model_binding(),
            },
        })

    # 5. Build merged inventory with collision-free claim IDs
    existing_cids = {r["claim_id"] for r in inventory if isinstance(r, dict) and "claim_id" in r}
    existing_nums = []
    for cid in existing_cids:
        if cid.startswith("C") and cid[1:].isdigit():
            existing_nums.append(int(cid[1:]))
    next_cid_idx = (max(existing_nums) + 1) if existing_nums else 1

    sel_ev_by_id = {e["evidence_id"]: e for e in selected_evidence}
    candidate_inventory_rows: list[dict[str, Any]] = []

    for sc in supported_claims:
        assigned_cid = f"C{next_cid_idx}"
        next_cid_idx += 1
        existing_cids.add(assigned_cid)

        ev_packages = [
            {"source_id": sel_ev_by_id[eid]["source_id"], "quote": sel_ev_by_id[eid]["quote"]}
            for eid in sc.get("evidence_ids", [])
            if eid in sel_ev_by_id
        ]
        pub_claim = {
            "text": sc["text"],
            "evidence": copy.deepcopy(ev_packages),
        }
        candidate_inventory_rows.append({
            "claim_id": assigned_cid,
            "text": sc["text"],
            "evidence": copy.deepcopy(ev_packages),
            "publication_claim": pub_claim,
            "origin": "coverage_correction",
        })

    merged_inventory = copy.deepcopy(inventory) + candidate_inventory_rows

    # 6. Call 3: 27B Coverage Reaudit
    reaudit_calls_before = len(getattr(verifier, "calls", []))
    ok_v_reaudit_pre, err_v_reaudit_pre, reaudit_history_before = _validate_client_prehistory(verifier, "Coverage reaudit")
    reaudit_trace = {
        "stage": "query_coverage",
        "origin": "coverage_correction",
        "model": client_verif_model,
        "status": None,
        "validated_payload": None,
        "wall_seconds": None,
        "calls": [],
        "client_calls": [],
        "raw_response": None,
        "error": None,
        "request": None,
    }
    if not ok_v_reaudit_pre:
        reaudit_trace["status"] = "error"
        reaudit_trace["error"] = err_v_reaudit_pre
        return _finalize_result({
            "attempted": True,
            "applied": False,
            "inventory": copy.deepcopy(inventory),
            "coverage_audit": copy.deepcopy(coverage_audit),
            "diagnostics": {
                "attempted": True,
                "applied": False,
                "trigger": trigger_type,
                "reason": f"Coverage reaudit pre-call history invalid: {err_v_reaudit_pre}",
                "error": err_v_reaudit_pre,
                "original_selected_evidence": copy.deepcopy(selected_evidence),
                "source_context_mappings": source_context_mappings,
                "generation": gen_trace,
                "verification": verif_trace,
                "reaudit": reaudit_trace,
                "model_binding": _make_model_binding(),
            },
        })

    try:
        expected_reaudit_msgs = build_coverage_messages(query, merged_inventory)
        expected_reaudit_schema = build_coverage_schema(query, merged_inventory)
        expected_reaudit_request = {
            "model": PINNED_27B_MODEL,
            "messages": copy.deepcopy(expected_reaudit_msgs),
            "stream": False,
            "think": False,
            "format": copy.deepcopy(expected_reaudit_schema),
            "options": copy.deepcopy(PINNED_27B_PARAMETERS),
        }
        frozen_reaudit_expected = json.loads(json.dumps(expected_reaudit_request, ensure_ascii=False, allow_nan=False))
    except Exception as exc:
        reaudit_trace["error"] = str(exc)
        return _finalize_result({
            "attempted": True,
            "applied": False,
            "inventory": copy.deepcopy(inventory),
            "coverage_audit": copy.deepcopy(coverage_audit),
            "diagnostics": {
                "attempted": True,
                "applied": False,
                "trigger": trigger_type,
                "reason": f"Failed constructing reaudit messages: {exc}",
                "error": str(exc),
                "original_selected_evidence": copy.deepcopy(selected_evidence),
                "source_context_mappings": source_context_mappings,
                "generation": gen_trace,
                "verification": verif_trace,
                "reaudit": reaudit_trace,
                "model_binding": _make_model_binding(),
            },
        })

    t2 = time.perf_counter()
    new_audit = None
    try:
        try:
            new_audit = audit_coverage(query, copy.deepcopy(merged_inventory), verifier=verifier)
        finally:
            wall_reaudit = time.perf_counter() - t2
            reaudit_trace["wall_seconds"] = wall_reaudit
            raw_r_resp = new_audit.get("raw_response") if isinstance(new_audit, dict) else None
            _capture_postcall_observations(verifier, reaudit_calls_before, reaudit_history_before, raw_r_resp, reaudit_trace, "Coverage reaudit")

        if isinstance(new_audit, dict):
            reaudit_trace["status"] = new_audit.get("status")
            reaudit_trace["validated_payload"] = copy.deepcopy(new_audit.get("validated_payload"))
            if new_audit.get("wall_seconds") is not None:
                reaudit_trace["wall_seconds"] = new_audit.get("wall_seconds")
            if new_audit.get("error") is not None:
                reaudit_trace["error"] = new_audit.get("error")

        # Validate verifier actual request from calls log for Call 3 (reaudit)
        ok_call, err_call, fresh_reaudit_call, delta_calls = _extract_and_validate_single_fresh_call(
            verifier, reaudit_calls_before, reaudit_history_before, COVERAGE_STAGE, "Coverage reaudit"
        )
        if not ok_call:
            raise RuntimeError(err_call)

        actual_reaudit_opts = fresh_reaudit_call["request"].get("options")
        actual_verif_opts = actual_reaudit_opts
        ok_req, err_req = _validate_client_request(fresh_reaudit_call["request"], frozen_reaudit_expected, "Coverage reaudit")
        if not ok_req:
            raise RuntimeError(err_req)

        # Validate verifier digest after reaudit
        try:
            verif_digest_after = verifier.get_digest() if callable(getattr(verifier, "get_digest", None)) else None
        except Exception as exc:
            raise RuntimeError(f"Verifier get_digest failed after reaudit: {exc}")
        if verif_digest_after != PINNED_27B_DIGEST or verif_digest_after != verif_digest_before:
            raise RuntimeError(f"Verifier digest changed after reaudit: {verif_digest_before} -> {verif_digest_after}")

        if new_audit.get("status") != "completed":
            return _finalize_result({
                "attempted": True,
                "applied": False,
                "inventory": copy.deepcopy(inventory),
                "coverage_audit": copy.deepcopy(coverage_audit),
                "diagnostics": {
                    "attempted": True,
                    "applied": False,
                    "trigger": trigger_type,
                    "reason": f"Coverage reaudit failed with status {new_audit.get('status')}: {new_audit.get('error')}",
                    "error": new_audit.get("error"),
                    "original_selected_evidence": copy.deepcopy(selected_evidence),
                    "source_context_mappings": source_context_mappings,
                    "revalidation": {
                        "revalidated": True,
                        "evidence_count": len(selected_evidence),
                        "sources": sorted(list(hits_by_source.keys())),
                    },
                    "generation": gen_trace,
                    "verification": verif_trace,
                    "reaudit": reaudit_trace,
                    "model_binding": _make_model_binding(),
                },
            })

        # Recompute and validate final coverage audit payload without trusting stored payload
        recomp_ok, recomp_err, recomputed_new_payload = _validate_and_recompute_coverage_audit(
            new_audit, query, merged_inventory
        )
        if not recomp_ok or recomputed_new_payload is None:
            reaudit_trace["error"] = f"Reaudit payload validation failed: {recomp_err}"
            return _finalize_result({
                "attempted": True,
                "applied": False,
                "inventory": copy.deepcopy(inventory),
                "coverage_audit": copy.deepcopy(coverage_audit),
                "diagnostics": {
                    "attempted": True,
                    "applied": False,
                    "trigger": trigger_type,
                    "reason": f"Reaudit payload validation failed: {recomp_err}",
                    "error": recomp_err,
                    "original_selected_evidence": copy.deepcopy(selected_evidence),
                    "source_context_mappings": source_context_mappings,
                    "revalidation": {
                        "revalidated": True,
                        "evidence_count": len(selected_evidence),
                        "sources": sorted(list(hits_by_source.keys())),
                    },
                    "generation": gen_trace,
                    "verification": verif_trace,
                    "reaudit": reaudit_trace,
                    "model_binding": _make_model_binding(),
                },
            })

        orig_parts = (
            recomputed_initial_payload.get("parts", [])
            if recomputed_initial_payload is not None
            else []
        )
        new_parts = recomputed_new_payload.get("parts", [])

        orig_answered = _answered_character_indices(orig_parts, len(query))
        new_answered = _answered_character_indices(new_parts, len(query))

        retains_all = orig_answered.issubset(new_answered)
        gained_unserved = len(new_answered - orig_answered) > 0
        is_strict_superset = retains_all and gained_unserved

        interval_comp = {
            "orig_answered_indices": sorted(list(orig_answered)),
            "new_answered_indices": sorted(list(new_answered)),
            "retains_all": retains_all,
            "gained_unserved": gained_unserved,
            "gained_indices": sorted(list(new_answered - orig_answered)),
        }

        if is_strict_superset:
            return _finalize_result({
                "attempted": True,
                "applied": True,
                "inventory": merged_inventory,
                "coverage_audit": new_audit,
                "diagnostics": {
                    "attempted": True,
                    "applied": True,
                    "trigger": trigger_type,
                    "reason": "Coverage reaudit retained all prior answered intervals and gained new unserved interval",
                    "error": None,
                    "original_selected_evidence": copy.deepcopy(selected_evidence),
                    "source_context_mappings": source_context_mappings,
                    "revalidation": {
                        "revalidated": True,
                        "evidence_count": len(selected_evidence),
                        "sources": sorted(list(hits_by_source.keys())),
                    },
                    "generation": gen_trace,
                    "verification": verif_trace,
                    "reaudit": reaudit_trace,
                    "interval_comparison": interval_comp,
                    "model_binding": _make_model_binding(),
                },
            })
        else:
            return _finalize_result({
                "attempted": True,
                "applied": False,
                "inventory": copy.deepcopy(inventory),
                "coverage_audit": copy.deepcopy(coverage_audit),
                "diagnostics": {
                    "attempted": True,
                    "applied": False,
                    "trigger": trigger_type,
                    "reason": f"Coverage reaudit did not meet improvement condition (retains_all={retains_all}, gained_unserved={gained_unserved})",
                    "error": None,
                    "original_selected_evidence": copy.deepcopy(selected_evidence),
                    "source_context_mappings": source_context_mappings,
                    "revalidation": {
                        "revalidated": True,
                        "evidence_count": len(selected_evidence),
                        "sources": sorted(list(hits_by_source.keys())),
                    },
                    "generation": gen_trace,
                    "verification": verif_trace,
                    "reaudit": reaudit_trace,
                    "interval_comparison": interval_comp,
                    "model_binding": _make_model_binding(),
                },
            })
    except Exception as exc:
        reaudit_trace["status"] = "error"
        reaudit_trace["error"] = str(exc)
        return _finalize_result({
            "attempted": True,
            "applied": False,
            "inventory": copy.deepcopy(inventory),
            "coverage_audit": copy.deepcopy(coverage_audit),
            "diagnostics": {
                "attempted": True,
                "applied": False,
                "trigger": trigger_type,
                "reason": f"Coverage reaudit failed: {exc}",
                "error": str(exc),
                "original_selected_evidence": copy.deepcopy(selected_evidence),
                "source_context_mappings": source_context_mappings,
                "generation": gen_trace,
                "verification": verif_trace,
                "reaudit": reaudit_trace,
                "model_binding": _make_model_binding(),
            },
        })
