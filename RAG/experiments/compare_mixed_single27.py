"""Measured comparison of mixed (9B generation + 27B Q3 verification/coverage) vs all27 (27B for all text stages).

Embeddings remain Qwen3-Embedding:4B in both arms.
Cold alternating 22-attempt schedule over 11 queries from scope_coverage full_inputs.json.
Both arms receive identical effective generation sampling options.
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
import re
import sys
import time
from typing import Any
import urllib.error
import urllib.request

from experiments.compare_system import generate_arm
from experiments.preserve_evidence import PreservedGrounder, validate_catalog
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
    DEFAULT_BASE_URL,
    DEFAULT_GENERATION_MODEL,
    MAX_RESPONSE_BYTES,
    OllamaGenerator,
    _NoRedirectHandler,
    _parse_float,
    _reject_constant,
    parse_chat_response,
    validate_chat_stats,
)
from rag_pensiones.chunking import ChunkRecord
from rag_pensiones.grounding import (
    DEFAULT_PARAMETERS,
    MAX_MESSAGES_BYTES,
    MAX_MESSAGES_BYTES_VERIFICATION,
    VERIFIER_Q3_DIGEST,
    VERIFIER_Q3_MODEL,
    VERIFIER_Q3_PARAMETERS,
    validate_grounded_result,
)
from rag_pensiones.ingest import PROJECT_ROOT, validate_project_path
from rag_pensiones.qwen_embeddings import (
    DIMENSION,
    MODEL as QWEN_EMBEDDING_MODEL,
    QUERY_PREFIX,
    QwenEmbedder,
    _json,
    vector,
)
from rag_pensiones.search import SearchHit, load_index, search, validate_index

PROTOCOL_SCHEMA_VERSION = "mixed-vs-single27-protocol-v1"
RUN_RESULTS_SCHEMA_VERSION = "mixed-vs-single27-run-results-v1"
SERVER_VERSION = "0.40.11"

GENERATOR_9B_MODEL = DEFAULT_GENERATION_MODEL  # "qwen3.5:9b"
TEXT_27B_MODEL = VERIFIER_Q3_MODEL  # "hf.co/gguf/Qwen3.8-27B-GGUF:Q3_K_M"
EMBEDDING_4B_MODEL = QWEN_EMBEDDING_MODEL  # "qwen3-embedding:4b"

ALLOWED_EXPERIMENT_MODELS = (
    GENERATOR_9B_MODEL,
    TEXT_27B_MODEL,
    EMBEDDING_4B_MODEL,
)

PINNED_9B_DIGEST = "56671c2ab9385f9cfcb404638e32cd62d88e3501d44822208363c010179a3c90"
PINNED_27B_DIGEST = VERIFIER_Q3_DIGEST
PINNED_EMBEDDING_DIGEST = "df5bd2e3c74cd8d069d21dc038f1b359fcdc9458fce1c99bd43c9eb1518ff907"

DEFAULT_TIMEOUT = 240.0

# 9B inherits presence_penalty=1.5, top_k=20, top_p=0.95 from modelfile; 27B has no model parameters.
# Both arms explicitly send this complete effective configuration for generation/selection/repair roles:
EFFECTIVE_GENERATION_OPTIONS: dict[str, Any] = {
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

CANONICAL_ROLE_PARAMETERS: dict[str, Any] = copy.deepcopy(DEFAULT_PARAMETERS)

DEFAULT_INDEX_REL = "data/processed/airef_2025_qwen3_4b_embeddings.json"
DEFAULT_QUERIES_REL = "evaluation/scope_coverage_2026-10-26/full_inputs.json"
DEFAULT_UNITS_REL = "evaluation/reconstructed_units_catalog.json"
DEFAULT_PAGES_REL = "data/processed/airef_2025_pages.jsonl"
DEFAULT_OLLAMA_DEFAULTS_REL = (
    "evaluation/verifier_capacity_2026-10-26/ollama_defaults_v0.40.11.json"
)
DEFAULT_MODEL_SHOW_REL = "evaluation/mixed_single27_2026-10-27/model_show_before.json"

REQUIRED_INPUT_KEYS = frozenset(
    {"index", "queries", "units", "pages", "defaults", "model_show"}
)


def get_source_closure() -> dict[str, str]:
    """Computes SHA-256 for all .py files under rag_pensiones and experiments."""
    closure: dict[str, str] = {}
    for pkg in ("rag_pensiones", "experiments"):
        pkg_dir = PROJECT_ROOT / pkg
        for py_file in sorted(pkg_dir.glob("**/*.py")):
            rel_path = py_file.relative_to(PROJECT_ROOT).as_posix()
            closure[rel_path] = canonical_file_sha256(py_file)
    return closure


REQUIRED_QUERY_ROSTER: tuple[str, ...] = (
    "b01",
    "b02",
    "b03",
    "b04",
    "b05",
    "b06",
    "b07",
    "b08",
    "n01",
    "n02",
    "n03",
)


def validate_queries_payload(qry_data: Any) -> list[dict[str, Any]]:
    """Validates query document structure before any network execution.

    Requires exact query document keys {'cases'}, exactly 11 cases matching the
    exact ordered roster (b01..b08, n01..n03), and exact case keys {'id', 'query'}.
    Rejects any label, answer, extra keys, or wrong/duplicate/reordered cases locally.
    """
    if not isinstance(qry_data, dict) or set(qry_data.keys()) != {"cases"}:
        extra_keys = set(qry_data.keys()) - {"cases"} if isinstance(qry_data, dict) else set()
        if any(term in k.lower() for term in ("gold", "answer", "label") for k in extra_keys):
            raise ValueError(f"Gold labels detected in query input document keys: {sorted(extra_keys)}")
        raise ValueError(
            f"Query document keys must be exactly {{'cases'}}, got: {sorted(qry_data.keys()) if isinstance(qry_data, dict) else type(qry_data)}"
        )
    cases = qry_data["cases"]
    if not isinstance(cases, list) or len(cases) != len(REQUIRED_QUERY_ROSTER):
        raise ValueError(f"Expected exactly {len(REQUIRED_QUERY_ROSTER)} cases in queries file, got {len(cases) if isinstance(cases, list) else type(cases)}")
    for c in cases:
        if not isinstance(c, dict):
            raise ValueError(f"Invalid case in queries: {c!r}")
        keys = set(c.keys())
        if keys != {"id", "query"}:
            if "id" not in keys or "query" not in keys:
                raise ValueError(f"Case missing required keys {{'id', 'query'}}: {c!r}")
            extra = keys - {"id", "query"}
            if any(term in k.lower() for term in ("gold", "answer", "label") for k in extra):
                raise ValueError(f"Gold labels detected in query input case {c.get('id')!r}: unexpected keys {sorted(extra)}")
            raise ValueError(f"Case {c.get('id')!r} contains unexpected extra keys: {sorted(extra)}")
        if not isinstance(c["id"], str) or not c["id"].strip():
            raise ValueError(f"Invalid case id: {c.get('id')!r}")
        if not isinstance(c["query"], str) or not c["query"].strip():
            raise ValueError(f"Invalid query text in case {c.get('id')!r}")

    actual_ids = [c["id"] for c in cases]
    if actual_ids != list(REQUIRED_QUERY_ROSTER):
        raise ValueError(
            f"Query cases must match exact ordered roster {list(REQUIRED_QUERY_ROSTER)}, got {actual_ids}"
        )
    return cases


def build_expected_schedule(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Builds the deterministic 22-attempt schedule from validated query cases.

    Alternating order per query:
    - even query_index (0, 2, 4, 6, 8, 10): mixed then all27
    - odd query_index (1, 3, 5, 7, 9): all27 then mixed
    """
    schedule: list[dict[str, Any]] = []
    attempt_idx = 0
    for q_idx, case in enumerate(cases):
        cid = case["id"]
        q_text = case["query"]
        if q_idx % 2 == 0:
            arms_order = ["mixed", "all27"]
        else:
            arms_order = ["all27", "mixed"]

        for arm in arms_order:
            schedule.append({
                "attempt_index": attempt_idx,
                "query_index": q_idx,
                "case_id": cid,
                "query": q_text,
                "arm": arm,
            })
            attempt_idx += 1
    return schedule


def assert_finite(data: Any, path: str = "root") -> None:
    """Recursively validates that all floats and ints in data structure are finite."""
    if isinstance(data, bool):
        return
    if isinstance(data, float):
        if not math.isfinite(data):
            raise ValueError(f"Non-finite float at {path}: {data}")
    elif isinstance(data, (list, tuple)):
        for idx, item in enumerate(data):
            assert_finite(item, f"{path}[{idx}]")
    elif isinstance(data, dict):
        for k, v in data.items():
            if not isinstance(k, str):
                raise ValueError(f"Non-string dictionary key at {path}: {k!r}")
            assert_finite(v, f"{path}.{k}")


def canonical_file_sha256(path: Path) -> str:
    """Computes SHA-256 of raw file bytes on disk."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checkpoint_atomic(data: dict[str, Any], destination: Path) -> None:
    """Atomically writes JSON report to disk using .pending temporary file."""
    assert_finite(data)
    temp = destination.with_suffix(destination.suffix + ".pending")
    temp.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temp, destination)


class MixedSingle27Transport:
    """Strict loopback HTTP transport for Ollama 0.40.11.

    Restricted strictly to the 3 authorized models and necessary endpoints.
    Blocks redirects, ignores system proxies, and enforces 4 MiB max response cap.
    """

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT,
        opener: urllib.request.OpenerDirector | None = None,
    ) -> None:
        if base_url != DEFAULT_BASE_URL:
            raise ValueError(f"Only strict loopback URL {DEFAULT_BASE_URL!r} is allowed, got {base_url!r}")
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout < 240.0:
            raise ValueError(f"timeout must be a number >= 240.0s, got {timeout!r}")
        self.base_url = base_url.rstrip("/")
        self.timeout = float(timeout)
        self.allowed_models = ALLOWED_EXPERIMENT_MODELS
        if opener is not None:
            self._opener = opener
        else:
            self._opener = urllib.request.build_opener(
                urllib.request.ProxyHandler({}),
                _NoRedirectHandler(),
            )

    def _open_request(self, req: urllib.request.Request) -> bytes:
        max_bytes = 4 * 1024 * 1024  # 4 MiB cap
        try:
            with self._opener.open(req, timeout=self.timeout) as resp:
                raw = resp.read(max_bytes + 1)
                if len(raw) > max_bytes:
                    raise RuntimeError(f"Ollama response exceeded 4 MiB limit ({len(raw)} bytes)")
                return raw
        except urllib.error.HTTPError as exc:
            raw_err = b""
            try:
                try:
                    raw_err = exc.read(max_bytes + 1)
                finally:
                    exc.close()
            except Exception:
                raw_err = b""
            body = raw_err[:max_bytes].decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP error {exc.code} from Ollama: {exc.reason}. Response: {body}") from exc
        except urllib.error.URLError as exc:
            raise ConnectionError(f"Connection error contacting Ollama at {self.base_url}: {exc.reason}") from exc
        except TimeoutError as exc:
            raise TimeoutError(f"Timeout contacting Ollama after {self.timeout}s") from exc

    def get_version(self) -> dict[str, Any]:
        url = f"{self.base_url}/api/version"
        raw = self._open_request(urllib.request.Request(url, method="GET"))
        data = json.loads(raw.decode("utf-8"), parse_constant=_reject_constant, parse_float=_parse_float)
        assert_finite(data)
        if not isinstance(data, dict) or "version" not in data:
            raise ValueError("Malformed /api/version response")
        return data

    def get_tags(self) -> dict[str, Any]:
        url = f"{self.base_url}/api/tags"
        raw = self._open_request(urllib.request.Request(url, method="GET"))
        data = json.loads(raw.decode("utf-8"), parse_constant=_reject_constant, parse_float=_parse_float)
        assert_finite(data)
        if not isinstance(data, dict) or not isinstance(data.get("models"), list):
            raise ValueError("Malformed /api/tags response")
        return data

    def get_ps(self) -> dict[str, Any]:
        url = f"{self.base_url}/api/ps"
        raw = self._open_request(urllib.request.Request(url, method="GET"))
        data = json.loads(raw.decode("utf-8"), parse_constant=_reject_constant, parse_float=_parse_float)
        assert_finite(data)
        if not isinstance(data, dict) or not isinstance(data.get("models"), list):
            raise ValueError("Malformed /api/ps response")
        return data

    def show(self, model: str) -> dict[str, Any]:
        if model not in self.allowed_models:
            raise ValueError(f"Model {model!r} not permitted in /api/show")
        url = f"{self.base_url}/api/show"
        body = json.dumps({"model": model}, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
        raw = self._open_request(req)
        data = json.loads(raw.decode("utf-8"), parse_constant=_reject_constant, parse_float=_parse_float)
        assert_finite(data)
        if not isinstance(data, dict):
            raise ValueError("Malformed /api/show response")
        return data

    def unload(self, model: str) -> dict[str, Any]:
        if model not in self.allowed_models:
            raise ValueError(f"Model {model!r} not permitted for unload")
        url = f"{self.base_url}/api/generate"
        body = json.dumps({"model": model, "keep_alive": 0}, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
        raw = self._open_request(req)
        try:
            data = json.loads(raw.decode("utf-8"), parse_constant=_reject_constant, parse_float=_parse_float)
        except Exception:
            data = {"status": "unloaded"}
        return data

    def chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        assert_finite(payload)
        model = payload.get("model")
        if model not in self.allowed_models:
            raise ValueError(f"Model {model!r} not permitted in /api/chat")
        url = f"{self.base_url}/api/chat"
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
        raw = self._open_request(req)
        raw_text = raw.decode("utf-8", errors="replace")
        data = json.loads(raw_text, parse_constant=_reject_constant, parse_float=_parse_float)
        assert_finite(data)
        if not isinstance(data, dict):
            raise ValueError("Response from /api/chat must be a JSON object")
        return data

    def embed(self, payload: dict[str, Any]) -> dict[str, Any]:
        assert_finite(payload)
        model = payload.get("model")
        if model != EMBEDDING_4B_MODEL:
            raise ValueError(f"Model {model!r} not permitted in /api/embed; expected {EMBEDDING_4B_MODEL!r}")
        url = f"{self.base_url}/api/embed"
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
        raw = self._open_request(req)
        raw_text = raw.decode("utf-8", errors="replace")
        data = json.loads(raw_text, parse_constant=_reject_constant, parse_float=_parse_float)
        assert_finite(data)
        if not isinstance(data, dict):
            raise ValueError("Response from /api/embed must be a JSON object")
        return data


class ComparisonEmbedder(QwenEmbedder):
    """Embedder using strict injectable transport while maintaining QwenEmbedder semantics."""

    def __init__(self, *, transport: Any | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.transport = transport
        self.calls: list[dict[str, Any]] = []

    def get_digest(self) -> str:
        if self.transport is not None:
            tags = self.transport.get_tags()
            models = tags.get("models", []) if isinstance(tags, dict) else []
            for m in models:
                if isinstance(m, dict):
                    name = m.get("name") or m.get("model")
                    if name == QWEN_EMBEDDING_MODEL:
                        digest = m.get("digest")
                        if isinstance(digest, str) and digest.strip():
                            return digest.strip()
            raise ValueError(f"Model {QWEN_EMBEDDING_MODEL!r} not found in transport tags")
        return super().get_digest()

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not isinstance(texts, list) or any(not isinstance(x, str) or not x.strip() for x in texts):
            raise ValueError("Expected list of nonempty texts")
        if not texts:
            return []

        payload = {
            "model": QWEN_EMBEDDING_MODEL,
            "input": texts,
            "truncate": False,
            "dimensions": DIMENSION,
            "keep_alive": "5m",
        }
        call_record: dict[str, Any] = {
            "request": copy.deepcopy(payload),
            "raw_response": None,
            "error": None,
            "wall_seconds": 0.0,
        }
        self.calls.append(call_record)

        started = time.perf_counter()
        try:
            if self.transport is not None and hasattr(self.transport, "embed"):
                data = self.transport.embed(payload)
            else:
                req = urllib.request.Request(
                    self.base_url + "/api/embed",
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                )
                data = _json(self._open_request(req))

            call_record["raw_response"] = copy.deepcopy(data)
            if not isinstance(data, dict) or data.get("model") != QWEN_EMBEDDING_MODEL:
                raise ValueError("Embedding response model mismatch")
            values = data.get("embeddings")
            if not isinstance(values, list) or len(values) != len(texts):
                raise ValueError("Embedding count mismatch")
            return [vector(v) for v in values]
        except Exception as exc:
            call_record["error"] = str(exc)
            raise
        finally:
            call_record["wall_seconds"] = time.perf_counter() - started


class ComparisonGenerator(OllamaGenerator):
    """Generator for selection, generation, and repair text stages.

    Truthfully records and sends EFFECTIVE_GENERATION_OPTIONS to Ollama in both arms.
    """

    def __init__(
        self,
        model: str,
        *,
        transport: Any | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        **kwargs,
    ) -> None:
        super().__init__(model=model, timeout=timeout, **kwargs)
        self.transport = transport
        self.calls: list[dict[str, Any]] = []
        self.active_stage: str = "text_generation"

    def get_digest(self) -> str:
        if self.transport is not None:
            tags = self.transport.get_tags()
            models = tags.get("models", []) if isinstance(tags, dict) else []
            for m in models:
                if isinstance(m, dict):
                    name = m.get("name") or m.get("model")
                    if name == self.model:
                        digest = m.get("digest")
                        if isinstance(digest, str) and digest.strip():
                            return digest.strip()
            raise ValueError(f"Model {self.model!r} not found in transport tags")
        return super().get_digest()

    def chat(self, messages: list[dict], response_schema: dict) -> dict:
        if not isinstance(messages, list) or len(messages) == 0:
            raise ValueError("messages must be a non-empty list of message dicts")
        for m in messages:
            if not isinstance(m, dict) or "role" not in m or "content" not in m:
                raise ValueError(f"Invalid message format: {m!r}")
        if not isinstance(response_schema, dict) or not response_schema:
            raise ValueError("response_schema must be a non-empty dict")

        total_msg_bytes = sum(len(m.get("content", "").encode("utf-8")) for m in messages)
        if total_msg_bytes > MAX_MESSAGES_BYTES:
            raise ValueError(
                f"Prompt message budget overflow: {total_msg_bytes} bytes exceeds limit {MAX_MESSAGES_BYTES}"
            )

        exact_request = {
            "model": self.model,
            "messages": copy.deepcopy(messages),
            "stream": False,
            "think": False,
            "keep_alive": "5m",
            "format": copy.deepcopy(response_schema),
            "options": copy.deepcopy(EFFECTIVE_GENERATION_OPTIONS),
        }

        call_record: dict[str, Any] = {
            "stage": self.active_stage,
            "model": self.model,
            "request": copy.deepcopy(exact_request),
            "effective_options": copy.deepcopy(EFFECTIVE_GENERATION_OPTIONS),
            "raw_response": None,
            "error": None,
            "wall_time_seconds": 0.0,
        }
        self.calls.append(call_record)

        started = time.perf_counter()
        try:
            if self.transport is not None:
                raw_response = self.transport.chat(exact_request)
            else:
                url = f"{self.base_url}/api/chat"
                body_bytes = json.dumps(exact_request, ensure_ascii=False).encode("utf-8")
                req = urllib.request.Request(
                    url,
                    data=body_bytes,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                raw = self._open_request(req)
                raw_text = raw.decode("utf-8", errors="replace")
                raw_response = json.loads(raw_text, parse_constant=_reject_constant, parse_float=_parse_float)

            wall_time = time.perf_counter() - started
            call_record["wall_time_seconds"] = wall_time
            call_record["raw_response"] = copy.deepcopy(raw_response)

            if isinstance(raw_response, dict):
                resp_model = raw_response.get("model")
                if resp_model is not None and resp_model != self.model:
                    raise RuntimeError(
                        f"Model mismatch in chat response: expected {self.model!r}, got {resp_model!r}"
                    )
                done_reason = raw_response.get("done_reason")
                if done_reason == "length":
                    raise RuntimeError(
                        f"Model generation truncated due to length limit (done_reason={done_reason!r})"
                    )

            return raw_response
        except Exception as exc:
            wall_time = time.perf_counter() - started
            call_record["wall_time_seconds"] = wall_time
            call_record["error"] = str(exc)
            raise


class Single27PreservedGrounder(PreservedGrounder):
    """Grounder where all text stages (selection, generation, repair, verification) run on 27B Q3.

    Never unloads 27B during text role switches within a query attempt.
    """

    def __init__(
        self,
        extracted: dict,
        pages: list[dict],
        *,
        verifier: Q3VerifierClient,
        generator: Any | None = None,
        **kwargs,
    ) -> None:
        if verifier is None or not hasattr(verifier, "chat"):
            raise ValueError("verifier must be a valid Q3VerifierClient instance")
        kwargs.pop("model", None)
        kwargs.pop("partial_generation", None)
        super().__init__(
            extracted,
            pages,
            generator=generator,
            model=VERIFIER_Q3_MODEL,
            partial_generation=True,
            **kwargs,
        )
        self.verifier = verifier

    @property
    def verification_policy(self) -> dict[str, Any]:
        return copy.deepcopy(self.verifier.verification_policy)

    def _chat_step(
        self,
        stage: str,
        messages: list[dict],
        response_schema: dict,
        expected_digest: str,
    ) -> tuple[dict, dict]:
        if stage in ("verification", "reverification"):
            # Reuses Q3PreservedGrounder._chat_step: unloads 9B if resident (none in all27), runs Q3 verification
            return Q3PreservedGrounder._chat_step(self, stage, messages, response_schema, expected_digest)

        # Other stages (selection, generation, repair_selection, repair_generation):
        # Calls PreservedGrounder._chat_step DIRECTLY; NEVER unloads 27B for role switches!
        if hasattr(self.generator, "active_stage"):
            self.generator.active_stage = stage
        return PreservedGrounder._chat_step(self, stage, messages, response_schema, expected_digest)


def ensure_cold_state(transport: Any) -> list[dict[str, Any]]:
    """Enforces cold state before each attempt: unloads selected 3 models, aborts on unrelated models.

    Records lifecycle records with timestamps and durations.
    """
    ps_data = transport.get_ps()
    models = ps_data.get("models", []) if isinstance(ps_data, dict) else []
    resident_names: list[str] = []
    for m in models:
        if isinstance(m, dict):
            name = m.get("name") or m.get("model")
            if isinstance(name, str) and name:
                resident_names.append(name)

    unrelated = [n for n in resident_names if n not in ALLOWED_EXPERIMENT_MODELS]
    if unrelated:
        raise RuntimeError(
            f"Unrelated models currently resident in Ollama /api/ps: {unrelated}. "
            "Aborting before inference without unloading them."
        )

    lifecycle: list[dict[str, Any]] = []
    for name in resident_names:
        if name in ALLOWED_EXPERIMENT_MODELS:
            t0 = time.perf_counter()
            transport.unload(name)
            dur = time.perf_counter() - t0
            lifecycle.append({
                "action": "pre_attempt_unload",
                "model": name,
                "duration_seconds": dur,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

    # Confirm absent
    ps_after = transport.get_ps()
    after_models = ps_after.get("models", []) if isinstance(ps_after, dict) else []
    still_resident = [
        (m.get("name") or m.get("model"))
        for m in after_models
        if isinstance(m, dict) and (m.get("name") or m.get("model")) in ALLOWED_EXPERIMENT_MODELS
    ]
    if still_resident:
        raise RuntimeError(
            f"Selected models still resident after keep_alive=0 unload: {still_resident}"
        )

    return lifecycle


def capture_ps_snapshot(
    transport: Any,
    event_tag: str,
    *,
    request_id: str | None = None,
    model: str | None = None,
    stage: str | None = None,
) -> dict[str, Any]:
    """Captures /api/ps snapshot containing model sizes, vram and context_length if returned."""
    try:
        ps_data = transport.get_ps()
        if not isinstance(ps_data, dict):
            return {
                "event": event_tag,
                "request_id": request_id,
                "model": model,
                "stage": stage,
                "error": f"Invalid ps response type: {type(ps_data).__name__}",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        models_out = []
        for m in ps_data.get("models", []):
            if isinstance(m, dict):
                model_entry: dict[str, Any] = {
                    "name": m.get("name") or m.get("model"),
                    "size": m.get("size"),
                    "size_vram": m.get("size_vram"),
                    "expires_at": m.get("expires_at"),
                }
                if "context_length" in m:
                    model_entry["context_length"] = m["context_length"]
                elif isinstance(m.get("details"), dict) and "context_length" in m["details"]:
                    model_entry["context_length"] = m["details"]["context_length"]
                models_out.append(model_entry)

        snapshot: dict[str, Any] = {
            "event": event_tag,
            "models": models_out,
            "raw_ps": copy.deepcopy(ps_data),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if request_id is not None:
            snapshot["request_id"] = request_id
        if model is not None:
            snapshot["model"] = model
        if stage is not None:
            snapshot["stage"] = stage
        return snapshot
    except Exception as exc:
        err_snapshot: dict[str, Any] = {
            "event": event_tag,
            "error": str(exc),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if request_id is not None:
            err_snapshot["request_id"] = request_id
        if model is not None:
            err_snapshot["model"] = model
        if stage is not None:
            err_snapshot["stage"] = stage
        return err_snapshot


def detect_chat_stage(payload: dict[str, Any]) -> str:
    """Infers the functional text stage from chat payload structure."""
    fmt = payload.get("format", {}) if isinstance(payload, dict) else {}
    if not isinstance(fmt, dict):
        return "unknown"
    props = fmt.get("properties", {})
    any_of = fmt.get("anyOf", [])
    if "ids" in props or any("properties" in b and "ids" in b["properties"] for b in any_of):
        return "selection"
    if "repairs" in props:
        return "repair_generation"
    if "claims" in props or any("properties" in b and "claims" in b["properties"] for b in any_of):
        return "generation"
    if "verdicts" in props or any("properties" in b and "verdicts" in b["properties"] for b in any_of):
        return "verification"
    if "parts" in props or any("properties" in b and "parts" in b["properties"] for b in any_of):
        return COVERAGE_STAGE
    return "chat"


class RecordingTransport:
    """Wrapper that intercepts chat calls to capture /api/ps after EACH completed model chat.

    Compatible with injected FakeTransport or loopback MixedSingle27Transport.
    Captures actual /api/ps immediately after each completed chat, associating snapshot with request/model identifiers.
    """

    def __init__(self, inner: Any, snapshots: list[dict[str, Any]] | None = None) -> None:
        self._inner = inner
        self.snapshots = snapshots if snapshots is not None else []
        self._chat_count = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        self._chat_count += 1
        model = payload.get("model") if isinstance(payload, dict) else None
        stage = detect_chat_stage(payload)
        req_id = f"chat_{self._chat_count}_{stage}"

        try:
            resp = self._inner.chat(payload)
            snapshot = capture_ps_snapshot(
                self._inner,
                event_tag=f"after_{req_id}",
                request_id=req_id,
                model=model,
                stage=stage,
            )
            self.snapshots.append(snapshot)
            return resp
        except Exception as exc:
            snapshot = capture_ps_snapshot(
                self._inner,
                event_tag=f"after_{req_id}_error",
                request_id=req_id,
                model=model,
                stage=stage,
            )
            self.snapshots.append(snapshot)
            raise


def compute_role_stats(calls: list[dict[str, Any]]) -> dict[str, Any]:
    """Computes exclusive role stats without double counting."""
    categories: dict[str, list[dict[str, Any]]] = {
        "generation": [],
        "verification": [],
        "coverage": [],
    }
    for c in calls:
        st = c.get("stage")
        if st in ("selection", "generation", "repair_selection", "repair_generation", "text_generation"):
            categories["generation"].append(c)
        elif st in ("verification", "reverification"):
            categories["verification"].append(c)
        elif st == COVERAGE_STAGE:
            categories["coverage"].append(c)

    def sum_category(call_list: list[dict[str, Any]]) -> dict[str, Any]:
        stat_keys = (
            "prompt_eval_count",
            "eval_count",
            "eval_duration",
            "total_duration",
            "load_duration",
            "prompt_eval_duration",
        )
        out: dict[str, Any] = {}
        for k in stat_keys:
            vals = []
            for item in call_list:
                raw_resp = item.get("raw_response", {})
                if isinstance(raw_resp, dict) and k in raw_resp and raw_resp[k] is not None:
                    vals.append(raw_resp[k])
                elif "stats" in item and isinstance(item["stats"], dict) and k in item["stats"]:
                    vals.append(item["stats"][k])
            out[k] = sum(vals) if vals else None
        return out

    gen_sums = sum_category(categories["generation"])
    ver_sums = sum_category(categories["verification"])
    cov_sums = sum_category(categories["coverage"])
    all_sums = sum_category(categories["generation"] + categories["verification"] + categories["coverage"])

    return {
        "generation": gen_sums,
        "verification": ver_sums,
        "coverage": cov_sums,
        "total_text": all_sums,
    }


def execute_attempt(
    attempt_spec: dict[str, Any],
    *,
    index_data: dict[str, Any],
    units_data: dict[str, Any],
    pages_data: list[dict[str, Any]],
    transport: Any,
    grounder_factory: Any | None = None,
) -> dict[str, Any]:
    """Executes a single evaluation attempt within timed interval.

    Cold state unload happens strictly OUTSIDE the timed interval.
    Timed interval includes live QwenEmbedder retrieval and all text stages through formatted answer.
    """
    case_id = attempt_spec["case_id"]
    query = attempt_spec["query"]
    arm = attempt_spec["arm"]

    if grounder_factory is not None and arm != "mixed":
        raise ValueError(
            f"grounder_factory is only supported for the 'mixed' arm, got arm={arm!r}"
        )

    # 1. Cold state enforcement (OUTSIDE timed interval)
    pre_lifecycle = ensure_cold_state(transport)
    ps_snapshots: list[dict[str, Any]] = [capture_ps_snapshot(transport, "start_cold")]

    # Recording transport wrapper captures /api/ps after EACH completed model chat
    rec_transport = RecordingTransport(transport, snapshots=ps_snapshots)

    # 2. Clients setup
    embedder = ComparisonEmbedder(transport=rec_transport, timeout=DEFAULT_TIMEOUT)
    verifier = Q3VerifierClient(expected_digest=PINNED_27B_DIGEST, transport=rec_transport)

    if arm == "mixed":
        text_generator = ComparisonGenerator(GENERATOR_9B_MODEL, transport=rec_transport, timeout=DEFAULT_TIMEOUT)
        if grounder_factory is not None:
            grounder = grounder_factory(
                units_data,
                pages_data,
                verifier=verifier,
                generator=text_generator,
                partial_generation=True,
            )
        else:
            grounder = Q3PreservedGrounder(
                units_data,
                pages_data,
                verifier=verifier,
                generator=text_generator,
                partial_generation=True,
            )
        expected_gen_model = GENERATOR_9B_MODEL
    elif arm == "all27":
        text_generator = ComparisonGenerator(TEXT_27B_MODEL, transport=rec_transport, timeout=DEFAULT_TIMEOUT)
        grounder = Single27PreservedGrounder(
            units_data,
            pages_data,
            verifier=verifier,
            generator=text_generator,
        )
        expected_gen_model = TEXT_27B_MODEL
    else:
        raise ValueError(f"Unknown arm {arm!r}")

    all_lifecycle: list[dict[str, Any]] = list(pre_lifecycle)
    attempt_started = time.perf_counter()
    retrieval_wall_seconds: float = 0.0
    emb_unload_seconds: float = 0.0

    factual_outcome: dict[str, Any] | None = None
    coverage_audit: dict[str, Any] | None = None
    coverage_publication: dict[str, Any] | None = None
    formatted_answer: str = ""
    attempt_status = "completed"
    attempt_error: str | None = None
    hits: list[SearchHit] = []

    try:
        # Step A: Fresh live retrieval
        t_ret = time.perf_counter()
        hits = search(index_data, query, top_k=4, client=embedder)
        retrieval_wall_seconds = time.perf_counter() - t_ret

        # Step B: Release embedding model in BOTH arms for equal text memory budget
        t_emb_un = time.perf_counter()
        rec_transport.unload(EMBEDDING_4B_MODEL)
        emb_unload_seconds = time.perf_counter() - t_emb_un
        all_lifecycle.append({
            "action": "release_embedding_model",
            "model": EMBEDDING_4B_MODEL,
            "duration_seconds": emb_unload_seconds,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        ps_snapshots.append(capture_ps_snapshot(rec_transport, "after_embedding_release"))

        # Step C: Factual grounding (selection, generation, verification, repair)
        gen_digest = grounder.generator.get_digest()
        factual_outcome = generate_arm(
            grounder,
            query,
            hits,
            gen_digest,
            preserve=True,
            verification_policy=grounder.verification_policy,
            expected_model=expected_gen_model,
        )
        ps_snapshots.append(capture_ps_snapshot(rec_transport, "after_factual_outcome"))

        if factual_outcome.get("status") == "error":
            attempt_status = "error"
            attempt_error = str(factual_outcome.get("error", "Factual grounding failed"))
            formatted_answer = f"Error: {attempt_error}"
        else:
            # Step D: Coverage audit
            inventory = claim_inventory(factual_outcome)
            coverage_audit = audit_coverage(query, inventory, verifier=verifier)
            ps_snapshots.append(capture_ps_snapshot(rec_transport, "after_coverage_audit"))

            if coverage_audit.get("status") == "error":
                attempt_status = "error"
                attempt_error = str(coverage_audit.get("error", "Coverage audit failed"))
                formatted_answer = f"Error: {attempt_error}"
            else:
                coverage_publication = build_coverage_publication(
                    query,
                    inventory,
                    coverage_audit["validated_payload"],
                )
                report_for_formatting = {
                    "status": coverage_publication["status"],
                    "publication": coverage_publication,
                    "factual_report": {"outcome": factual_outcome},
                }
                formatted_answer = format_covered_answer(report_for_formatting)

    except Exception as exc:
        attempt_status = "error"
        attempt_error = str(exc)
        formatted_answer = f"Error: {exc}"

    query_wall_seconds = time.perf_counter() - attempt_started

    # Collect calls
    all_calls: list[dict[str, Any]] = []
    if hasattr(grounder.generator, "calls"):
        all_calls.extend(copy.deepcopy(grounder.generator.calls))
    if hasattr(verifier, "calls"):
        all_calls.extend(copy.deepcopy(verifier.calls))
    if hasattr(verifier, "lifecycle_records"):
        all_lifecycle.extend(copy.deepcopy(verifier.lifecycle_records))

    stats_summary = compute_role_stats(all_calls)

    retrieval_hits_records: list[dict[str, Any]] = []
    for h in hits:
        chunk_text = h.chunk.text if hasattr(h.chunk, "text") else ""
        content_sha256 = hashlib.sha256(chunk_text.encode("utf-8")).hexdigest()
        record_dict = {
            "chunk_id": h.chunk.chunk_id,
            "doc_id": h.chunk.doc_id,
            "source_file": h.chunk.source_file,
            "page_number": h.chunk.page_number,
            "start_char": h.chunk.start_char,
            "end_char": h.chunk.end_char,
            "text": chunk_text,
            "strategy": getattr(h.chunk, "strategy", None),
        }
        record_sha256 = hashlib.sha256(
            json.dumps(record_dict, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        retrieval_hits_records.append({
            "rank": h.rank,
            "score": h.score,
            "chunk_id": h.chunk.chunk_id,
            "source_id": f"S{h.rank}",
            "page_number": h.chunk.page_number,
            "content_sha256": content_sha256,
            "record_sha256": record_sha256,
            "text": chunk_text,
        })

    return {
        "attempt_index": attempt_spec["attempt_index"],
        "case_id": case_id,
        "query": query,
        "arm": arm,
        "text_model": expected_gen_model,
        "embedding_model": EMBEDDING_4B_MODEL,
        "status": attempt_status,
        "error": attempt_error,
        "query_wall_seconds": query_wall_seconds,
        "retrieval": {
            "retrieval_wall_seconds": retrieval_wall_seconds,
            "embedding_release_seconds": emb_unload_seconds,
            "hits": retrieval_hits_records,
            "embedder_calls": copy.deepcopy(embedder.calls),
        },
        "factual_outcome": copy.deepcopy(factual_outcome),
        "coverage_audit": copy.deepcopy(coverage_audit),
        "coverage_publication": copy.deepcopy(coverage_publication),
        "formatted_answer": formatted_answer,
        "lifecycle_records": all_lifecycle,
        "ps_snapshots": ps_snapshots,
        "calls": all_calls,
        "stats_summary": stats_summary,
    }


def prepare(
    protocol_path: Path | str,
    *,
    index_path: Path | str = DEFAULT_INDEX_REL,
    queries_path: Path | str = DEFAULT_QUERIES_REL,
    units_path: Path | str = DEFAULT_UNITS_REL,
    pages_path: Path | str = DEFAULT_PAGES_REL,
    defaults_path: Path | str = DEFAULT_OLLAMA_DEFAULTS_REL,
    model_show_path: Path | str = DEFAULT_MODEL_SHOW_REL,
) -> dict[str, Any]:
    """Prepares and freezes the experiment protocol without contacting any network.

    Refuses existing destination and validates all inputs strictly.
    """
    target = Path(protocol_path)
    if not target.is_absolute():
        target = (PROJECT_ROOT / target).resolve()
    else:
        target = target.resolve()

    validate_project_path(target)
    if target.exists():
        raise FileExistsError(f"Target protocol destination already exists: {target}")

    p_idx = (PROJECT_ROOT / index_path).resolve()
    p_qry = (PROJECT_ROOT / queries_path).resolve()
    p_unt = (PROJECT_ROOT / units_path).resolve()
    p_pgs = (PROJECT_ROOT / pages_path).resolve()
    p_def = (PROJECT_ROOT / defaults_path).resolve()
    p_shw = (PROJECT_ROOT / model_show_path).resolve()

    for p in (p_idx, p_qry, p_unt, p_pgs, p_def, p_shw):
        validate_project_path(p)
        if not p.is_file():
            raise FileNotFoundError(f"Required input file not found: {p}")

    # Read and validate decoded inputs (no gold)
    idx_data = json.loads(p_idx.read_text(encoding="utf-8"), parse_constant=_reject_constant)
    validate_index(idx_data)

    qry_data = json.loads(p_qry.read_text(encoding="utf-8"), parse_constant=_reject_constant)
    cases = validate_queries_payload(qry_data)

    unt_data = json.loads(p_unt.read_text(encoding="utf-8"), parse_constant=_reject_constant)
    pgs_lines = [
        json.loads(line, parse_constant=_reject_constant)
        for line in p_pgs.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    validate_catalog(unt_data, pgs_lines)

    shw_data = json.loads(p_shw.read_text(encoding="utf-8"), parse_constant=_reject_constant)
    if "qwen3.5:9b" not in shw_data or VERIFIER_Q3_MODEL not in shw_data:
        raise ValueError("model_show_before snapshot missing required models")

    # Build 22-attempt schedule using pure helper
    schedule = build_expected_schedule(cases)
    if len(schedule) != 22:
        raise ValueError(f"Schedule must have exactly 22 attempts, got {len(schedule)}")

    # Freeze all .py source files under rag_pensiones and experiments
    frozen_files: dict[str, str] = get_source_closure()

    input_hashes = {
        "index": canonical_file_sha256(p_idx),
        "queries": canonical_file_sha256(p_qry),
        "units": canonical_file_sha256(p_unt),
        "pages": canonical_file_sha256(p_pgs),
        "defaults": canonical_file_sha256(p_def),
        "model_show": canonical_file_sha256(p_shw),
    }

    protocol: dict[str, Any] = {
        "schema_version": PROTOCOL_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "server_version": SERVER_VERSION,
        "models": {
            "generator_9b": {
                "name": GENERATOR_9B_MODEL,
                "digest": PINNED_9B_DIGEST,
            },
            "text_27b": {
                "name": TEXT_27B_MODEL,
                "digest": PINNED_27B_DIGEST,
            },
            "embedding_4b": {
                "name": EMBEDDING_4B_MODEL,
                "digest": PINNED_EMBEDDING_DIGEST,
            },
        },
        "effective_generation_options": copy.deepcopy(EFFECTIVE_GENERATION_OPTIONS),
        "canonical_role_parameters": copy.deepcopy(CANONICAL_ROLE_PARAMETERS),
        "verifier_parameters": copy.deepcopy(VERIFIER_Q3_PARAMETERS),
        "schedule": schedule,
        "input_hashes": input_hashes,
        "frozen_files": frozen_files,
        "paths": {
            "index": p_idx.relative_to(PROJECT_ROOT).as_posix(),
            "queries": p_qry.relative_to(PROJECT_ROOT).as_posix(),
            "units": p_unt.relative_to(PROJECT_ROOT).as_posix(),
            "pages": p_pgs.relative_to(PROJECT_ROOT).as_posix(),
            "defaults": p_def.relative_to(PROJECT_ROOT).as_posix(),
            "model_show": p_shw.relative_to(PROJECT_ROOT).as_posix(),
        },
    }

    checkpoint_atomic(protocol, target)
    return protocol


def run(
    protocol_path: Path | str,
    output_path: Path | str,
    *,
    transport: Any | None = None,
) -> dict[str, Any]:
    """Runs the 22-attempt measured comparison following the frozen protocol.

    Mandatory pre-network validations verify protocol, frozen hashes, inputs, and output non-existence
    BEFORE constructing any live clients or contacting Ollama.
    """
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

    # MANDATORY PRE-NETWORK CHECK 1: Refuse overwrite
    if p_out.exists():
        raise FileExistsError(f"Output destination already exists: {p_out}")

    if not p_proto.is_file():
        raise FileNotFoundError(f"Protocol file not found: {p_proto}")

    # MANDATORY PRE-NETWORK CHECK 2: Read and validate protocol schema
    proto_bytes = p_proto.read_bytes()
    proto = json.loads(proto_bytes.decode("utf-8"), parse_constant=_reject_constant)
    if proto.get("schema_version") != PROTOCOL_SCHEMA_VERSION:
        raise ValueError(
            f"Invalid protocol schema: expected {PROTOCOL_SCHEMA_VERSION!r}, got {proto.get('schema_version')!r}"
        )

    # MANDATORY PRE-NETWORK CHECK 3: Verify disk bytes against all frozen files and complete source closure
    frozen_files = proto.get("frozen_files")
    if not isinstance(frozen_files, dict) or not frozen_files:
        raise ValueError("Protocol is missing frozen_files map")

    for rel_path, expected_hash in frozen_files.items():
        if not isinstance(rel_path, str) or not rel_path.strip():
            raise ValueError(f"Invalid path in frozen_files: {rel_path!r}")
        disk_path = (PROJECT_ROOT / rel_path).resolve()
        validate_project_path(disk_path)
        if not disk_path.is_file():
            raise FileNotFoundError(f"Frozen file missing on disk: {disk_path}")
        actual_hash = canonical_file_sha256(disk_path)
        if actual_hash != expected_hash:
            raise ValueError(
                f"Frozen file modified on disk: {rel_path} (expected {expected_hash}, got {actual_hash})"
            )

    complete_closure = get_source_closure()
    missing_sources = set(complete_closure.keys()) - set(frozen_files.keys())
    if missing_sources:
        raise ValueError(
            f"frozen_files is missing required source closure pins: {sorted(missing_sources)}"
        )
    extra_sources = set(frozen_files.keys()) - set(complete_closure.keys())
    if extra_sources:
        raise ValueError(
            f"frozen_files contains unexpected extra entries: {sorted(extra_sources)}"
        )

    # MANDATORY PRE-NETWORK CHECK 4: Verify exact required paths and input_hashes
    paths = proto.get("paths")
    if not isinstance(paths, dict):
        raise ValueError("Protocol is missing paths dictionary")
    if set(paths.keys()) != REQUIRED_INPUT_KEYS:
        missing_paths = sorted(REQUIRED_INPUT_KEYS - set(paths.keys()))
        extra_paths = sorted(set(paths.keys()) - REQUIRED_INPUT_KEYS)
        raise ValueError(
            f"Protocol paths must match exact required keys {sorted(REQUIRED_INPUT_KEYS)}; missing={missing_paths}, extra={extra_paths}"
        )

    input_hashes = proto.get("input_hashes")
    if not isinstance(input_hashes, dict):
        raise ValueError("Protocol is missing input_hashes dictionary")
    if set(input_hashes.keys()) != REQUIRED_INPUT_KEYS:
        missing_hashes = sorted(REQUIRED_INPUT_KEYS - set(input_hashes.keys()))
        extra_hashes = sorted(set(input_hashes.keys()) - REQUIRED_INPUT_KEYS)
        raise ValueError(
            f"Protocol input_hashes must match exact required keys {sorted(REQUIRED_INPUT_KEYS)}; missing={missing_hashes}, extra={extra_hashes}"
        )

    resolved_paths: dict[str, Path] = {}
    for key in sorted(REQUIRED_INPUT_KEYS):
        rel_str = paths[key]
        if not isinstance(rel_str, str) or not rel_str.strip():
            raise ValueError(f"Invalid path for input {key!r}: {rel_str!r}")
        p_res = (PROJECT_ROOT / rel_str).resolve()
        validate_project_path(p_res)
        if not p_res.is_file():
            raise FileNotFoundError(f"Required input file for {key!r} not found: {p_res}")

        expected_hash = input_hashes[key]
        if not isinstance(expected_hash, str) or len(expected_hash) != 64:
            raise ValueError(f"Invalid expected hash for input {key!r}: {expected_hash!r}")
        actual_hash = canonical_file_sha256(p_res)
        if actual_hash != expected_hash:
            raise ValueError(
                f"Input file hash mismatch for {key!r} ({rel_str}): expected {expected_hash}, got {actual_hash}"
            )
        resolved_paths[key] = p_res

    # MANDATORY PRE-NETWORK CHECK 5: Verify input contents and shape
    p_idx = resolved_paths["index"]
    p_qry = resolved_paths["queries"]
    p_unt = resolved_paths["units"]
    p_pgs = resolved_paths["pages"]
    p_def = resolved_paths["defaults"]
    p_shw = resolved_paths["model_show"]

    idx_data = json.loads(p_idx.read_text(encoding="utf-8"), parse_constant=_reject_constant)
    validate_index(idx_data)

    qry_data = json.loads(p_qry.read_text(encoding="utf-8"), parse_constant=_reject_constant)
    cases = validate_queries_payload(qry_data)

    unt_data = json.loads(p_unt.read_text(encoding="utf-8"), parse_constant=_reject_constant)
    pgs_lines = [
        json.loads(line, parse_constant=_reject_constant)
        for line in p_pgs.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    validate_catalog(unt_data, pgs_lines)

    shw_data = json.loads(p_shw.read_text(encoding="utf-8"), parse_constant=_reject_constant)
    if "qwen3.5:9b" not in shw_data or VERIFIER_Q3_MODEL not in shw_data:
        raise ValueError("model_show_before snapshot missing required models")

    expected_schedule = build_expected_schedule(cases)
    schedule = proto.get("schedule")
    if not isinstance(schedule, list) or len(schedule) != len(expected_schedule):
        raise ValueError(
            f"Schedule must have {len(expected_schedule)} attempts, got {len(schedule) if isinstance(schedule, list) else type(schedule)}"
        )

    if schedule != expected_schedule:
        for idx, (act, exp) in enumerate(zip(schedule, expected_schedule)):
            if act != exp:
                raise ValueError(
                    f"Schedule attempt {idx} mismatch against expected schedule: "
                    f"actual={act!r}, expected={exp!r}"
                )
        raise ValueError("Schedule does not match expected schedule generated from validated queries")

    # All pre-network validations passed! Initialize transport if not provided
    if transport is None:
        transport = MixedSingle27Transport(timeout=DEFAULT_TIMEOUT)

    # Validate server version and pinned tags/digests
    v_info = transport.get_version()
    if v_info.get("version") != SERVER_VERSION:
        raise RuntimeError(
            f"Ollama server version mismatch: expected {SERVER_VERSION!r}, got {v_info.get('version')!r}"
        )

    tags = transport.get_tags()
    tag_models = {
        (m.get("name") or m.get("model")): m.get("digest")
        for m in tags.get("models", [])
        if isinstance(m, dict)
    }

    if tag_models.get(GENERATOR_9B_MODEL) != PINNED_9B_DIGEST:
        raise RuntimeError(
            f"9B model digest mismatch: expected {PINNED_9B_DIGEST}, got {tag_models.get(GENERATOR_9B_MODEL)}"
        )
    if tag_models.get(TEXT_27B_MODEL) != PINNED_27B_DIGEST:
        raise RuntimeError(
            f"27B model digest mismatch: expected {PINNED_27B_DIGEST}, got {tag_models.get(TEXT_27B_MODEL)}"
        )
    if tag_models.get(EMBEDDING_4B_MODEL) != PINNED_EMBEDDING_DIGEST:
        raise RuntimeError(
            f"Embedding 4B model digest mismatch: expected {PINNED_EMBEDDING_DIGEST}, got {tag_models.get(EMBEDDING_4B_MODEL)}"
        )

    # Validate /api/show sampling metadata
    show_9b = transport.show(GENERATOR_9B_MODEL)
    show_27b = transport.show(TEXT_27B_MODEL)
    if not isinstance(show_9b, dict) or not isinstance(show_27b, dict):
        raise RuntimeError("/api/show must return valid dict for both models")

    # Initial checkpoint with run_status 'incomplete' (honest even if killed by uncatchable signal)
    results_report: dict[str, Any] = {
        "schema_version": RUN_RESULTS_SCHEMA_VERSION,
        "run_status": "incomplete",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "finished_at": None,
        "protocol_path": p_proto.relative_to(PROJECT_ROOT).as_posix(),
        "protocol_sha256": hashlib.sha256(proto_bytes).hexdigest(),
        "server_version": SERVER_VERSION,
        "digests": {
            GENERATOR_9B_MODEL: PINNED_9B_DIGEST,
            TEXT_27B_MODEL: PINNED_27B_DIGEST,
            EMBEDDING_4B_MODEL: PINNED_EMBEDDING_DIGEST,
        },
        "effective_generation_options": copy.deepcopy(EFFECTIVE_GENERATION_OPTIONS),
        "attempts": [],
    }
    checkpoint_atomic(results_report, p_out)

    # Execute schedule with per-attempt atomic checkpoints, preserving incomplete status on interruption
    try:
        for attempt_spec in schedule:
            att_res = execute_attempt(
                attempt_spec,
                index_data=idx_data,
                units_data=unt_data,
                pages_data=pgs_lines,
                transport=transport,
            )
            results_report["attempts"].append(att_res)
            checkpoint_atomic(results_report, p_out)
    except (KeyboardInterrupt, SystemExit, Exception):
        results_report["run_status"] = "incomplete"
        checkpoint_atomic(results_report, p_out)
        raise

    # Final completed status ONLY after all 22 attempts finish
    if len(results_report["attempts"]) == 22:
        has_error = any(a.get("status") == "error" for a in results_report["attempts"])
        results_report["run_status"] = "error" if has_error else "completed"
    else:
        results_report["run_status"] = "incomplete"
    results_report["finished_at"] = datetime.now(timezone.utc).isoformat()
    checkpoint_atomic(results_report, p_out)

    return results_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Measured comparison runner: Mixed (9B+27B) vs Single 27B."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prep_parser = subparsers.add_parser("prepare", help="Prepare and freeze protocol artifact.")
    prep_parser.add_argument("--protocol", required=True, help="Destination path for frozen protocol JSON.")

    run_parser = subparsers.add_parser("run", help="Run 22-attempt schedule following frozen protocol.")
    run_parser.add_argument("--protocol", required=True, help="Path to frozen protocol JSON.")
    run_parser.add_argument("--output", required=True, help="Destination path for run results JSON.")

    args = parser.parse_args(argv)

    if args.command == "prepare":
        res = prepare(args.protocol)
        print(f"Protocol prepared successfully: {len(res['schedule'])} attempts frozen.")
        return 0
    elif args.command == "run":
        res = run(args.protocol, args.output)
        print(f"Run completed with status {res['run_status']!r}: {len(res['attempts'])} attempts executed.")
        return 0 if res["run_status"] == "completed" else 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
