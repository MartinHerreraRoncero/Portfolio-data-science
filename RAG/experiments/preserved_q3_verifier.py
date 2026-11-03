"""Integration client and grounder for the selected Q3 verifier (Phase 1).

Maintains truthful stage identities, isolating claim-evidence verification
to the selected Qwen 27B Q3_K_M verifier while preserving legacy 9B generation.
"""

from __future__ import annotations

import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import time
from typing import Any

from experiments.compare_verifier_capacity import (
    DEFAULT_OLLAMA_DEFAULTS_PATH,
    OllamaResponseError,
    assert_finite,
    parse_show_parameters,
    resolve_effective_sampling,
    validate_sampling_match,
)
from experiments.compare_verifier_quantization import (
    REFERENCE_ARM_B_OPTIONS,
    REFERENCE_ARM_B_SHOW_PARAMETERS,
    QuantizationOllamaTransport,
)
from experiments.preserve_evidence import PreservedGrounder
from rag_pensiones.answer import (
    DEFAULT_GENERATION_MODEL,
    _parse_float,
    _reject_constant,
    parse_chat_response,
    validate_chat_stats,
)
from rag_pensiones.grounding import (
    DEFAULT_PARAMETERS,
    MAX_MESSAGES_BYTES_VERIFICATION,
    SHA256_HEX_PATTERN,
    VERIFIER_Q3_DIGEST,
    VERIFIER_Q3_MODEL,
    VERIFIER_Q3_PARAMETERS,
)
from rag_pensiones.ingest import PROJECT_ROOT

ALLOWED_RELEASE_MODELS = (
    DEFAULT_GENERATION_MODEL,
    VERIFIER_Q3_MODEL,
)


class Q3VerifierClient:
    """Verifier client dedicated strictly to claim-evidence verification using Qwen 27B Q3_K_M.

    Constructor validates configuration without network calls.
    Metadata preflight verifies Ollama server version 0.40.11, exact tag/digest/quantization,
    and sampling alignment against benchmark reference Arm B.
    """

    def __init__(
        self,
        *,
        expected_digest: str = VERIFIER_Q3_DIGEST,
        transport: Any | None = None,
    ) -> None:
        if (
            isinstance(expected_digest, bool)
            or not isinstance(expected_digest, str)
            or len(expected_digest) != 64
            or not SHA256_HEX_PATTERN.fullmatch(expected_digest)
        ):
            raise ValueError(
                f"expected_digest must be a 64-character hex SHA-256 string, got {expected_digest!r}"
            )
        self.model = VERIFIER_Q3_MODEL
        self.expected_digest = expected_digest
        self.parameters = copy.deepcopy(VERIFIER_Q3_PARAMETERS)
        self.calls: list[dict[str, Any]] = []
        self.lifecycle_records: list[dict[str, Any]] = []
        self._preflight_done = False

        if transport is None:
            transport = QuantizationOllamaTransport()
            transport.allowed_models = (DEFAULT_GENERATION_MODEL, VERIFIER_Q3_MODEL)
        self.transport = transport

    @property
    def verification_policy(self) -> dict[str, Any]:
        """Exposes the immutable trusted verification policy."""
        return {
            "model": self.model,
            "digest": self.expected_digest,
            "parameters": copy.deepcopy(self.parameters),
        }

    def preflight(self) -> dict[str, Any]:
        """Validates Ollama server version 0.40.11, model tag/digest/quantization, and sampling alignment."""
        # 1. Server version check
        v_info = self.transport.get_version()
        if not isinstance(v_info, dict) or v_info.get("version") != "0.40.11":
            raise RuntimeError(
                f"Ollama server version mismatch: reported {v_info.get('version')!r}, expected '0.40.11'"
            )

        # 2. Model tag, digest, and quantization level check
        tags_info = self.transport.get_tags()
        models = tags_info.get("models", []) if isinstance(tags_info, dict) else []
        model_found = False
        for m in models:
            if isinstance(m, dict):
                name = m.get("name") or m.get("model")
                if name == self.model:
                    model_found = True
                    act_digest = m.get("digest")
                    if act_digest != self.expected_digest:
                        raise RuntimeError(
                            f"Model {self.model!r} digest mismatch: expected {self.expected_digest!r}, got {act_digest!r}"
                        )
                    details = m.get("details", {}) if isinstance(m.get("details"), dict) else {}
                    quant = details.get("quantization_level")
                    if quant != "Q3_K_M":
                        raise RuntimeError(
                            f"Model {self.model!r} quantization mismatch: expected 'Q3_K_M', got {quant!r}"
                        )
                    break
        if not model_found:
            raise RuntimeError(f"Required model {self.model!r} not found in Ollama /api/tags")

        # 3. Model /api/show check
        show_info = self.transport.show(self.model)
        if not isinstance(show_info, dict):
            raise RuntimeError(f"/api/show for {self.model!r} must return a JSON object")
        show_details = show_info.get("details", {})
        if isinstance(show_details, dict) and "quantization_level" in show_details:
            if show_details["quantization_level"] != "Q3_K_M":
                raise RuntimeError(
                    f"Model {self.model!r} show quantization mismatch: expected 'Q3_K_M', got {show_details['quantization_level']!r}"
                )

        params_raw = show_info.get("parameters")
        show_params = parse_show_parameters(params_raw) if params_raw else {}

        # 4. Sampling alignment check against benchmark reference Arm B
        defaults_file = DEFAULT_OLLAMA_DEFAULTS_PATH
        if defaults_file.is_file():
            defaults_data = json.loads(
                defaults_file.read_bytes().decode("utf-8"),
                parse_constant=_reject_constant,
                parse_float=_parse_float,
            )
            server_defaults = defaults_data.get("defaults", {})
        else:
            server_defaults = {
                "temperature": 0.8,
                "top_k": 40,
                "top_p": 0.9,
                "min_p": 0.0,
                "typical_p": 1.0,
                "repeat_last_n": 64,
                "repeat_penalty": 1.0,
                "presence_penalty": 0.0,
                "frequency_penalty": 0.0,
                "draft_num_predict": 4,
            }

        ref_b_effective = resolve_effective_sampling(
            server_defaults,
            REFERENCE_ARM_B_SHOW_PARAMETERS,
            REFERENCE_ARM_B_OPTIONS,
        )
        eff_sampling = resolve_effective_sampling(
            server_defaults,
            show_params,
            self.parameters,
        )
        is_match, mismatches = validate_sampling_match(ref_b_effective, eff_sampling)
        if not is_match:
            raise RuntimeError(
                f"Effective sampling mismatch against reference Arm B: {mismatches}"
            )

        self._preflight_done = True
        return {
            "version": v_info.get("version"),
            "model": self.model,
            "digest": self.expected_digest,
            "quantization_level": "Q3_K_M",
            "effective_sampling": eff_sampling,
        }

    def get_digest(self) -> str:
        """Retrieves the active digest of the Q3 verifier model from transport."""
        if hasattr(self.transport, "get_digest") and callable(self.transport.get_digest):
            return self.transport.get_digest()
        tags_info = self.transport.get_tags()
        models = tags_info.get("models", []) if isinstance(tags_info, dict) else []
        for m in models:
            if isinstance(m, dict):
                name = m.get("name") or m.get("model")
                if name == self.model:
                    digest = m.get("digest")
                    if isinstance(digest, str) and digest.strip():
                        return digest.strip()
        raise RuntimeError(f"Model {self.model!r} not found in Ollama /api/tags")

    def release_loaded(self, model: str) -> bool:
        """Unloads the specified model using keep_alive=0 ONLY if resident in /api/ps.

        Accepts ONLY 'qwen3.5:9b' and 'hf.co/gguf/Qwen3.8-27B-GGUF:Q3_K_M'.
        Records lifecycle separately; never counts as inference.
        """
        if model not in ALLOWED_RELEASE_MODELS:
            raise ValueError(
                f"release_loaded accepts ONLY {ALLOWED_RELEASE_MODELS}, got {model!r}"
            )
        is_resident = False
        try:
            ps_info = self.transport.get_ps()
            models = ps_info.get("models", []) if isinstance(ps_info, dict) else []
            for m in models:
                if isinstance(m, dict):
                    name = m.get("name") or m.get("model")
                    if name == model:
                        is_resident = True
                        break
        except Exception as exc:
            self.lifecycle_records.append({"action": "inspect_residency", "model": model, "error": str(exc), "unloaded": False})
            raise

        unloaded = False
        if is_resident:
            try:
                self.transport.unload(model)
                unloaded = True
            except Exception as exc:
                self.lifecycle_records.append({"action": "unload", "model": model, "was_resident": True, "error": str(exc), "unloaded": False})
                raise

        record = {
            "action": "unload",
            "model": model,
            "was_resident": is_resident,
            "unloaded": unloaded,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self.lifecycle_records.append(record)
        return unloaded

    def chat(
        self,
        messages: list[dict],
        response_schema: dict,
        *,
        stage: str,
    ) -> dict[str, Any]:
        """Executes a single chat completion for verification, reverification, or query_coverage.

        Validates message budget, pins digests before and after, captures raw requests/responses
        and timing in .calls before any validation can fail, and validates strict output contracts.
        """
        if stage not in ("verification", "reverification", "query_coverage"):
            raise ValueError(
                f"Q3VerifierClient.chat accepts only 'verification', 'reverification', or 'query_coverage' stages, got {stage!r}"
            )

        if not isinstance(messages, list) or len(messages) == 0:
            raise ValueError("messages must be a non-empty list of message dicts")
        for m in messages:
            if not isinstance(m, dict) or "role" not in m or "content" not in m:
                raise ValueError(f"Invalid message item: {m!r}")

        if not isinstance(response_schema, dict) or isinstance(response_schema, bool) or not response_schema:
            raise ValueError("response_schema must be a non-empty dict")

        # Serialized message bytes preflight
        msg_bytes = json.dumps(messages, ensure_ascii=False).encode("utf-8")
        if len(msg_bytes) > MAX_MESSAGES_BYTES_VERIFICATION:
            raise ValueError(
                f"Messages payload in stage {stage!r} exceeds {MAX_MESSAGES_BYTES_VERIFICATION} bytes UTF-8 "
                f"({len(msg_bytes)} bytes). Aborting without contacting model."
            )

        # Run preflight if not done yet
        if not self._preflight_done:
            self.preflight()

        # Check digest before
        digest_before = self.get_digest()
        if digest_before != self.expected_digest:
            raise RuntimeError(
                f"Verifier model digest mismatch before {stage}: expected {self.expected_digest!r}, "
                f"got {digest_before!r}. Aborting."
            )

        exact_request = {
            "model": self.model,
            "messages": copy.deepcopy(messages),
            "stream": False,
            "think": False,
            "format": copy.deepcopy(response_schema),
            "options": copy.deepcopy(self.parameters),
        }

        call_record: dict[str, Any] = {
            "stage": stage,
            "model": self.model,
            "request": copy.deepcopy(exact_request),
            "attempted": True,
            "raw_response": None,
            "error": None,
            "wall_time_seconds": 0.0,
        }
        self.calls.append(call_record)

        t_start = time.perf_counter()
        raw_response = None
        try:
            raw_response = self.transport.chat(exact_request)
            wall_time = time.perf_counter() - t_start
            call_record["wall_time_seconds"] = wall_time
            call_record["raw_response"] = copy.deepcopy(raw_response)
        except Exception as exc:
            wall_time = time.perf_counter() - t_start
            call_record["wall_time_seconds"] = wall_time
            call_record["error"] = str(exc)
            if isinstance(exc, OllamaResponseError):
                call_record["raw_response"] = {"raw_text": exc.raw_text, "status_code": exc.status_code, "is_truncated": exc.is_truncated}
            raise

        # Check digest after
        digest_after = self.get_digest()
        if digest_after != self.expected_digest:
            err = (
                f"Verifier model digest mismatch after {stage}: expected {self.expected_digest!r}, "
                f"got {digest_after!r}. Aborting."
            )
            call_record["error"] = err
            raise RuntimeError(err)

        # Response validations
        try:
            assert_finite(raw_response)
            if not isinstance(raw_response, dict):
                raise ValueError("Response from /api/chat must be a dict")

            if raw_response.get("done") is not True:
                raise ValueError(f"Response 'done' flag is not True: {raw_response.get('done')!r}")

            done_reason = raw_response.get("done_reason")
            if done_reason != "stop":
                raise ValueError(f"Response done_reason is {done_reason!r}, expected 'stop'")

            resp_model = raw_response.get("model")
            if resp_model != self.model:
                raise ValueError(
                    f"Returned model {resp_model!r} does not match requested verifier model {self.model!r}"
                )

            prompt_eval_count = raw_response.get("prompt_eval_count")
            if (
                prompt_eval_count is None
                or isinstance(prompt_eval_count, bool)
                or not isinstance(prompt_eval_count, int)
                or prompt_eval_count < 0
            ):
                raise ValueError(
                    f"Missing or invalid prompt_eval_count: {prompt_eval_count!r}. Must be a non-negative integer."
                )

            eval_count = raw_response.get("eval_count")
            if (
                eval_count is None
                or isinstance(eval_count, bool)
                or not isinstance(eval_count, int)
                or eval_count < 0
            ):
                raise ValueError(
                    f"Missing or invalid eval_count: {eval_count!r}. Must be a non-negative integer."
                )

            num_predict = self.parameters["num_predict"]
            if prompt_eval_count + num_predict + 256 > 8192:
                raise ValueError(
                    f"Context budget exceeded: prompt_eval_count ({prompt_eval_count}) + "
                    f"num_predict ({num_predict}) + 256 = {prompt_eval_count + num_predict + 256} > 8192"
                )

            msg = raw_response.get("message")
            if not isinstance(msg, dict) or isinstance(msg, bool):
                raise ValueError("Response missing valid 'message' object")

            if msg.get("role") != "assistant":
                raise ValueError(f"Invalid message role {msg.get('role')!r}, expected 'assistant'")

            if msg.get("tool_calls"):
                raise ValueError("Unexpected tool_calls in verification response")

            # Check thinking
            thinking_str = None
            if "thinking" in msg and isinstance(msg["thinking"], str):
                thinking_str = msg["thinking"]
            elif "thinking" in raw_response and isinstance(raw_response["thinking"], str):
                thinking_str = raw_response["thinking"]

            if thinking_str is not None and len(thinking_str.strip()) > 0:
                raise ValueError(
                    f"Technical protocol violation: unexpected thinking content ({len(thinking_str)} chars) "
                    "returned when think=false"
                )

            content = msg.get("content")
            if not isinstance(content, str) or not content.strip():
                raise ValueError("Response message 'content' is empty or not a string")

            # Validate that content parses as valid finite JSON
            try:
                content_json = json.loads(content, parse_constant=_reject_constant, parse_float=_parse_float)
            except Exception as exc:
                raise ValueError(f"Malformed JSON in response content: {exc}") from exc
            assert_finite(content_json)

            # Validate stats
            raw_stats = {}
            for k in (
                "prompt_eval_count",
                "eval_count",
                "eval_duration",
                "total_duration",
                "load_duration",
                "prompt_eval_duration",
            ):
                if k in raw_response:
                    raw_stats[k] = raw_response[k]
            validate_chat_stats(raw_stats)

        except Exception as exc:
            call_record["error"] = str(exc)
            raise

        return raw_response


class Q3PreservedGrounder(PreservedGrounder):
    """Grounder that routes claim-evidence verification and reverification to Q3VerifierClient.

    Retains the legacy 9B base model for selection and generation.
    Swaps models at role switches to ensure only one LLM is resident in GPU memory.
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
        super().__init__(extracted, pages, generator=generator, model=DEFAULT_GENERATION_MODEL, **kwargs)
        self.verifier = verifier

    @property
    def verification_policy(self) -> dict[str, Any]:
        """Exposes the trusted verification policy from the verifier client."""
        return copy.deepcopy(self.verifier.verification_policy)

    def _chat_step(
        self,
        stage: str,
        messages: list[dict],
        response_schema: dict,
        expected_digest: str,
    ) -> tuple[dict, dict]:
        """Routes verification and reverification to Q3VerifierClient; other stages use super."""
        if stage in ("verification", "reverification"):
            # Role switch: unload 9B generator if resident before Q3 verification
            self.verifier.release_loaded(DEFAULT_GENERATION_MODEL)

            budget = MAX_MESSAGES_BYTES_VERIFICATION
            serialized = json.dumps(messages, ensure_ascii=False).encode("utf-8")
            if len(serialized) > budget:
                raise ValueError(
                    f"Messages payload in stage {stage!r} exceeds {budget} bytes UTF-8 "
                    f"({len(serialized)} bytes). Aborting without contacting verifier."
                )
            prompt_sha256 = hashlib.sha256(serialized).hexdigest()

            raw_response = self.verifier.chat(messages, response_schema, stage=stage)
            parsed = parse_chat_response(raw_response, self.verifier.model)
            validated_stats = validate_chat_stats(parsed["stats"])

            trace_item = {
                "stage": stage,
                "model": self.verifier.model,
                "model_digest": self.verifier.expected_digest,
                "prompt_sha256": prompt_sha256,
                "stats": validated_stats,
                "parameters": copy.deepcopy(self.verifier.parameters),
                "messages_budget_bytes": budget,
            }
            return parsed["payload"], trace_item

        # Non-verification stage (selection, generation, repair_selection, repair_generation):
        # Role switch: unload Q3 verifier if resident before running 9B generator
        self.verifier.release_loaded(VERIFIER_Q3_MODEL)

        return super()._chat_step(stage, messages, response_schema, expected_digest)
