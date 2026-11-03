"""Unit tests for Phase 3 table-grounding comparison runner and protocol.

Tests use synthetic fixtures and fake execution/transport ONLY (no network, Ollama, or PDF calls):
- Exact 31-attempt schedule generation and alternating first arm across paired questions.
- Protocol preparation, asset freezing, and strict validation.
- Rejection of tampered assets, models, parameters, source closure, or manifest/reference hashes.
- Full 31-condition execution with truthful passing of table_bundle and enabled flag.
- Calling execute_fn EXACTLY ONCE per scheduled condition with no TypeError retry.
- Strict rejection of fake canonical SHA and asset drift versus historical bindings.
- Preservation of technical failures within the full 31 denominator.
- Rejection of empty ({}) or invalid attempt result shapes without counting them as completed.
- Pre- and post-condition drift detection preserving executed prefix upon abort.
- Export of blinded neutral review bundle and exclusive review_key.json mapping.
- Rejection of output_path equal to review_key.json preventing overwrite collisions.
- Revalidation of current protocol and strict field binding in export_review.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Callable
import unittest

from experiments.compare_layout_answers import (
    PINNED_9B_DIGEST,
    PINNED_27B_DIGEST,
    PINNED_EMBEDDING_DIGEST,
    SERVER_VERSION,
)
from experiments.compare_mixed_single27 import DEFAULT_GENERATION_MODEL
from experiments.compare_table_grounding import (
    ASSET_FILES,
    DEFAULT_EXTRA_QUERIES_PATH,
    DEFAULT_HISTORICAL_PROTOCOL_PATH,
    DEFAULT_LAYOUT_INDEX_PATH,
    DEFAULT_LAYOUT_PAGES_PATH,
    DEFAULT_LAYOUT_PRIMARY_CATALOG_PATH,
    DEFAULT_LAYOUT_TABLE_CATALOG_PATH,
    DEFAULT_MANIFEST_PATH,
    DEFAULT_PRIMARY_QUERIES_PATH,
    DEFAULT_REFERENCE_FREEZE_PATH,
    DEFAULT_STRUCTURED_BUNDLE_PATH,
    DEFAULT_TABLE_QUERIES_PATH,
    HISTORICAL_PROTOCOL_SHA256,
    REQUIRED_INPUT_PATHS,
    SCHEMA_VERSION,
    build_schedule,
    compute_canonical_sha256,
    export_review,
    prepare,
    run,
    validate_protocol,
    _validate_attempt_result,
)
from experiments.preserve_evidence import publication
from experiments.preserved_q3_verifier import VERIFIER_Q3_MODEL
from experiments.table_grounded import execute_table_attempt
from rag_pensiones.ingest import PROJECT_ROOT
from rag_pensiones.qwen_embeddings import MODEL as QWEN_EMBEDDING_MODEL
from test_table_grounded_pipeline import (
    PipelineFakeTransport,
    _build_synthetic_fixtures,
)


class FakeTransport:
    """Mock Ollama transport for runner unit tests."""

    def __init__(self, version: str = SERVER_VERSION) -> None:
        self.version = version

    def get_version(self) -> dict[str, str]:
        return {"version": self.version}

    def get_tags(self) -> dict[str, Any]:
        return {
            "models": [
                {"name": DEFAULT_GENERATION_MODEL, "digest": PINNED_9B_DIGEST},
                {"name": VERIFIER_Q3_MODEL, "digest": PINNED_27B_DIGEST},
                {"name": QWEN_EMBEDDING_MODEL, "digest": PINNED_EMBEDDING_DIGEST},
            ]
        }


class FakeExecutionHandler:
    """Mock execution handler recording calls and simulating attempt results."""

    def __init__(
        self,
        *,
        fail_at_indices: set[int] | None = None,
        type_error_at_indices: set[int] | None = None,
        empty_at_indices: set[int] | None = None,
        on_attempt_cb: Callable[[int], None] | None = None,
    ) -> None:
        self.fail_at_indices = set(fail_at_indices or set())
        self.type_error_at_indices = set(type_error_at_indices or set())
        self.empty_at_indices = set(empty_at_indices or set())
        self.on_attempt_cb = on_attempt_cb
        self.recorded_calls: list[dict[str, Any]] = []

    def __call__(
        self,
        attempt_spec: dict[str, Any],
        *,
        index_data: dict[str, Any],
        units_data: dict[str, Any],
        pages_data: list[dict[str, Any]],
        transport: Any,
        table_bundle: dict[str, Any] | None = None,
        enabled: bool = False,
    ) -> dict[str, Any]:
        attempt_index = attempt_spec["attempt_index"]

        if self.on_attempt_cb is not None:
            self.on_attempt_cb(attempt_index)

        self.recorded_calls.append({
            "attempt_index": attempt_index,
            "case_id": attempt_spec["case_id"],
            "query": attempt_spec["query"],
            "enabled": enabled,
            "has_table_bundle": table_bundle is not None,
            "units_count": len(units_data.get("units", [])),
            "pages_count": len(pages_data),
        })

        if attempt_index in self.type_error_at_indices:
            raise TypeError(f"Simulated internal TypeError in execute_fn at attempt {attempt_index}")

        if attempt_index in self.fail_at_indices:
            raise RuntimeError(f"Simulated technical failure at attempt {attempt_index}")

        if attempt_index in self.empty_at_indices:
            return {}

        if not enabled:
            audit_is_table = False
            gate_decision = "bypassed"
        elif attempt_index > 11:
            audit_is_table = True
            gate_decision = "accepted"
        else:
            audit_is_table = False
            gate_decision = "bypassed"

        return {
            "arm": "mixed",
            "case_id": attempt_spec["case_id"],
            "query": attempt_spec["query"],
            "attempt_index": attempt_index,
            "status": "completed",
            "formatted_answer": f"Respuesta sintetizada para {attempt_spec['case_id']} (enabled={enabled})",
            "factual_outcome": {
                "status": "answered",
                "facts": [
                    {
                        "cell_id": "c_r1c1",
                        "concept": "Empleo",
                        "year": 2025,
                        "value": "10,1",
                        "unit": "% PIB",
                        "evidence_ids": ["ev_c1"],
                    }
                ],
            },
            "coverage_audit": {
                "status": "passed",
                "validated_payload": {"covered": True},
            },
            "coverage_publication": {
                "status": "covered",
                "claims": [{"claim_id": "c1", "text": "Dato comprobado", "status": "supported"}],
            },
            "lifecycle_records": [
                {"role": "generator", "model": DEFAULT_GENERATION_MODEL},
                {"role": "verifier", "model": VERIFIER_Q3_MODEL},
            ],
            "stats_summary": {"eval_count": 50},
            "query_wall_seconds": 0.05,
            "table_audit": {
                "enabled": enabled,
                "is_table_query": audit_is_table,
                "gate_decision": gate_decision,
            },
        }


class TestTableGroundingRunner(unittest.TestCase):
    """Test suite for Table Grounding comparison runner."""

    def setUp(self) -> None:
        self.p_prim = PROJECT_ROOT / DEFAULT_PRIMARY_QUERIES_PATH
        self.p_tbl = PROJECT_ROOT / DEFAULT_TABLE_QUERIES_PATH
        self.p_ext = PROJECT_ROOT / DEFAULT_EXTRA_QUERIES_PATH
        self.prim_data = json.loads(self.p_prim.read_text(encoding="utf-8"))
        self.tbl_data = json.loads(self.p_tbl.read_text(encoding="utf-8"))
        self.ext_data = json.loads(self.p_ext.read_text(encoding="utf-8"))

    # 1. Schedule construction & counterbalancing
    def test_build_schedule_exact_31_and_alternation(self) -> None:
        schedule = build_schedule(self.prim_data, self.tbl_data, self.ext_data)
        self.assertEqual(len(schedule), 31)

        # 11 primary questions: candidate only, question_index 0..10, attempt_index 1..11
        for idx in range(11):
            att = schedule[idx]
            self.assertEqual(att["attempt_index"], idx + 1)
            self.assertEqual(att["question_index"], idx)
            self.assertEqual(att["cohort"], "primary")
            self.assertEqual(att["condition"], "candidate")
            self.assertEqual(att["arm"], "mixed")
            self.assertIs(att["table_enabled"], True)

        # 10 paired questions: alternating first arm
        paired_expected_order = [
            ("control", "candidate"),  # Pair 0: t01
            ("candidate", "control"),  # Pair 1: t02
            ("control", "candidate"),  # Pair 2: t03
            ("candidate", "control"),  # Pair 3: t04
            ("control", "candidate"),  # Pair 4: t05
            ("candidate", "control"),  # Pair 5: t06
            ("control", "candidate"),  # Pair 6: u01
            ("candidate", "control"),  # Pair 7: u02
            ("control", "candidate"),  # Pair 8: u03
            ("candidate", "control"),  # Pair 9: u04
        ]

        for pair_idx, (exp_cond1, exp_cond2) in enumerate(paired_expected_order):
            att1 = schedule[11 + pair_idx * 2]
            att2 = schedule[11 + pair_idx * 2 + 1]

            q_idx = 11 + pair_idx
            self.assertEqual(att1["question_index"], q_idx)
            self.assertEqual(att2["question_index"], q_idx)
            self.assertEqual(att1["case_id"], att2["case_id"])

            self.assertEqual(att1["condition"], exp_cond1)
            self.assertIs(att1["table_enabled"], exp_cond1 == "candidate")

            self.assertEqual(att2["condition"], exp_cond2)
            self.assertIs(att2["table_enabled"], exp_cond2 == "candidate")

            exp_cohort = "table" if pair_idx < 6 else "extra"
            self.assertEqual(att1["cohort"], exp_cohort)
            self.assertEqual(att2["cohort"], exp_cohort)

    def test_build_schedule_rejects_gold_keys_and_tampered_cases(self) -> None:
        # Forbidden gold key in extra queries
        bad_ext = copy.deepcopy(self.ext_data)
        bad_ext["cases"][0]["gold_answer"] = "1.5"
        with self.assertRaises(ValueError) as ctx:
            build_schedule(self.prim_data, self.tbl_data, bad_ext)
        self.assertIn("forbidden gold label", str(ctx.exception).lower())

        # Wrong count in table queries
        bad_tbl = copy.deepcopy(self.tbl_data)
        bad_tbl["cases"].pop()
        with self.assertRaises(ValueError) as ctx:
            build_schedule(self.prim_data, bad_tbl, self.ext_data)
        self.assertIn("exactly 6 cases", str(ctx.exception))

    # 2. Preparation and validation happy path
    def test_prepare_and_validate_protocol_happy_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            proto_dir = Path(tmp_dir) / "protocol_workspace"
            transport = FakeTransport()

            protocol = prepare(proto_dir, transport=transport)
            self.assertEqual(protocol["schema_version"], SCHEMA_VERSION)
            self.assertEqual(len(protocol["schedule"]), 31)
            self.assertEqual(protocol["counts"]["total_queries_count"], 21)
            self.assertEqual(protocol["counts"]["attempts_count"], 31)

            # Check files on disk
            for af in ASSET_FILES:
                self.assertTrue((proto_dir / af).is_file(), f"Missing asset file: {af}")
            self.assertTrue((proto_dir / "protocol.json").is_file())

            # Validate protocol passes without exception
            validate_protocol(protocol, protocol_dir=proto_dir)

            # Both arms check
            both_arms = protocol["both_arms"]
            self.assertFalse(both_arms["control"]["table_enabled"])
            self.assertTrue(both_arms["candidate"]["table_enabled"])
            self.assertEqual(both_arms["control"]["pages_file"], both_arms["candidate"]["pages_file"])
            self.assertEqual(both_arms["control"]["index_file"], both_arms["candidate"]["index_file"])
            self.assertEqual(both_arms["control"]["primary_catalog_file"], both_arms["candidate"]["primary_catalog_file"])
            self.assertEqual(both_arms["control"]["table_catalog_file"], both_arms["candidate"]["table_catalog_file"])

    # 3. Protocol tampering rejection
    def test_validate_protocol_rejects_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            proto_dir = Path(tmp_dir) / "protocol_workspace"
            transport = FakeTransport()
            protocol = prepare(proto_dir, transport=transport)

            # 1. Bad server version
            bad_p = copy.deepcopy(protocol)
            bad_p["server_version"] = "0.40.12"
            with self.assertRaises(ValueError) as ctx:
                validate_protocol(bad_p, protocol_dir=proto_dir)
            self.assertIn("server version mismatch", str(ctx.exception).lower())

            # 2. Bad model digest
            bad_p = copy.deepcopy(protocol)
            bad_p["models"]["generation_digest"] = "0" * 64
            with self.assertRaises(ValueError) as ctx:
                validate_protocol(bad_p, protocol_dir=proto_dir)
            self.assertIn("protocol model", str(ctx.exception).lower())

            # 3. Bad parameter type (bool vs int)
            bad_p = copy.deepcopy(protocol)
            bad_p["execution_options"]["closed_note_tails"] = 0
            with self.assertRaises(ValueError) as ctx:
                validate_protocol(bad_p, protocol_dir=proto_dir)
            self.assertIn("type mismatch", str(ctx.exception).lower())

            # 4. Bad schedule count (e.g. 30 attempts)
            bad_p = copy.deepcopy(protocol)
            bad_p["schedule"].pop()
            with self.assertRaises(ValueError) as ctx:
                validate_protocol(bad_p, protocol_dir=proto_dir)
            self.assertIn("exactly 31 attempts", str(ctx.exception).lower())

            # 5. Catalog corruption: T95 in primary catalog
            prim_path = proto_dir / "layout_primary_catalog.json"
            orig_prim_bytes = prim_path.read_bytes()
            tampered_prim = json.loads(orig_prim_bytes.decode("utf-8"))
            tampered_prim["units"].append({
                "id": "T95",
                "kind": "body",
                "complete_end": True,
                "spans": [{"doc_id": "doc1", "source_file": "file.pdf", "published_at": "2025-01-01", "page_number": 95, "start_char": 0, "end_char": 10, "quote": "CUADRO 9."}]
            })
            prim_path.write_text(json.dumps(tampered_prim), encoding="utf-8")
            bad_p = copy.deepcopy(protocol)
            bad_p["generated_assets"]["layout_primary_catalog.json"]["raw_sha256"] = hashlib.sha256(prim_path.read_bytes()).hexdigest()
            bad_p["generated_assets"]["layout_primary_catalog.json"]["canonical_sha256"] = compute_canonical_sha256(prim_path.read_bytes())
            with self.assertRaises(ValueError) as ctx:
                validate_protocol(bad_p, protocol_dir=proto_dir)
            self.assertTrue(any(token in str(ctx.exception).lower() for token in ("historical", "canonical", "catalog", "forbidden t95", "mismatch")))
            prim_path.write_bytes(orig_prim_bytes)  # restore

            # Ensure no execution or export artifacts were created during validation
            self.assertFalse((proto_dir / "results.json").exists())
            self.assertFalse((proto_dir / "review_bundle.json").exists())
            self.assertFalse((proto_dir / "review_key.json").exists())

    # 4. Full 31-attempt execution
    def test_run_all_31_conditions_completed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            proto_dir = Path(tmp_dir) / "protocol_workspace"
            transport = FakeTransport()
            prepare(proto_dir, transport=transport)

            results_path = Path(tmp_dir) / "results.json"
            handler = FakeExecutionHandler()

            results = run(
                proto_dir / "protocol.json",
                results_path,
                transport=transport,
                execute_fn=handler,
            )

            self.assertEqual(results["run_status"], "completed")
            self.assertEqual(len(results["attempts"]), 31)
            self.assertEqual(results["summary"]["completed_attempts"], 31)
            self.assertEqual(results["summary"]["error_attempts"], 0)

            # Verify truthful call parameters recorded by handler: exactly 31 calls
            self.assertEqual(len(handler.recorded_calls), 31)
            for idx, call in enumerate(handler.recorded_calls):
                sched_item = results["attempts"][idx]
                self.assertEqual(call["attempt_index"], sched_item["attempt_index"])
                self.assertEqual(call["enabled"], sched_item["table_enabled"])
                self.assertTrue(call["has_table_bundle"])

            # Verify atomic checkpoints on disk
            self.assertTrue(results_path.is_file())
            disk_results = json.loads(results_path.read_text(encoding="utf-8"))
            self.assertEqual(disk_results["run_status"], "completed")
            self.assertEqual(len(disk_results["attempts"]), 31)

    # 5. Technical failures stay in 31 denominator
    def test_run_retains_technical_failures_in_denominator(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            proto_dir = Path(tmp_dir) / "protocol_workspace"
            transport = FakeTransport()
            prepare(proto_dir, transport=transport)

            results_path = Path(tmp_dir) / "results.json"
            # Fail at primary attempt 5 and diagnostic attempt 18
            handler = FakeExecutionHandler(fail_at_indices={5, 18})

            results = run(
                proto_dir / "protocol.json",
                results_path,
                transport=transport,
                execute_fn=handler,
            )

            self.assertEqual(results["run_status"], "completed")
            self.assertEqual(len(results["attempts"]), 31)
            self.assertEqual(results["summary"]["completed_attempts"], 29)
            self.assertEqual(results["summary"]["error_attempts"], 2)

            att5 = results["attempts"][4]
            self.assertEqual(att5["attempt_index"], 5)
            self.assertEqual(att5["status"], "error")
            self.assertTrue(att5["raw_attempt"]["synthetic_exception_record"])
            self.assertIn("Simulated technical failure at attempt 5", att5["error"])

            att18 = results["attempts"][17]
            self.assertEqual(att18["attempt_index"], 18)
            self.assertEqual(att18["status"], "error")
            self.assertTrue(att18["raw_attempt"]["synthetic_exception_record"])

    # 6. Pre- and post-condition drift detection
    def test_run_aborts_on_protocol_drift_preserving_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            proto_dir = Path(tmp_dir) / "protocol_workspace"
            transport = FakeTransport()
            prepare(proto_dir, transport=transport)

            proto_file = proto_dir / "protocol.json"

            def tamper_protocol_on_disk(attempt_idx: int) -> None:
                if attempt_idx == 2:
                    raw = json.loads(proto_file.read_text(encoding="utf-8"))
                    raw["tampered"] = True
                    proto_file.write_text(json.dumps(raw), encoding="utf-8")

            handler = FakeExecutionHandler(on_attempt_cb=tamper_protocol_on_disk)
            results_path = Path(tmp_dir) / "results.json"

            with self.assertRaises(RuntimeError) as ctx:
                run(proto_file, results_path, transport=transport, execute_fn=handler)
            self.assertIn("protocol file on disk was modified", str(ctx.exception).lower())

            # Preserves executed prefix in atomic checkpoint
            self.assertTrue(results_path.is_file())
            saved = json.loads(results_path.read_text(encoding="utf-8"))
            self.assertEqual(saved["run_status"], "invalid")
            self.assertEqual(len(saved["attempts"]), 2)
            self.assertIn("integrity drift", saved["error"].lower())

    # 7. Blinded review bundle export and review key
    def test_export_review_neutral_bundle_and_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            proto_dir = Path(tmp_dir) / "protocol_workspace"
            transport = FakeTransport()
            prepare(proto_dir, transport=transport)

            results_path = Path(tmp_dir) / "results.json"
            handler = FakeExecutionHandler(fail_at_indices={14})
            run(proto_dir / "protocol.json", results_path, transport=transport, execute_fn=handler)

            review_dir = Path(tmp_dir) / "review_output"
            review_bundle_path = review_dir / "review_bundle.json"

            bundle = export_review(
                proto_dir / "protocol.json",
                results_path,
                review_bundle_path,
            )

            self.assertEqual(bundle["schema_version"], "table-grounding-review-v1")
            self.assertEqual(len(bundle["cases"]), 21)

            # Check 11 primary questions: single neutral label
            for idx in range(11):
                case = bundle["cases"][idx]
                self.assertEqual(case["cohort"], "primary")
                self.assertEqual(list(case["arms"].keys()), ["single"])
                arm = case["arms"]["single"]
                self.assertIn("formatted_answer", arm)
                self.assertIn("coverage_publication", arm)
                self.assertNotIn("table_audit", arm)
                self.assertNotIn("candidate", arm)
                self.assertNotIn("control", arm)

            # Check 10 diagnostic questions: neutral A/B pair
            for idx in range(11, 21):
                case = bundle["cases"][idx]
                self.assertIn(case["cohort"], ("table", "extra"))
                self.assertEqual(set(case["arms"].keys()), {"A", "B"})
                arm_a = case["arms"]["A"]
                arm_b = case["arms"]["B"]
                self.assertNotIn("table_audit", arm_a)
                self.assertNotIn("table_audit", arm_b)
                self.assertNotIn("candidate", arm_a)
                self.assertNotIn("control", arm_a)

            # Check review_key.json
            key_file = review_dir / "review_key.json"
            self.assertTrue(key_file.is_file())
            key_data = json.loads(key_file.read_text(encoding="utf-8"))
            self.assertEqual(key_data["schema_version"], "table-grounding-review-key-v1")

            mapping = key_data["mapping"]
            self.assertEqual(len(mapping), 21)
            self.assertEqual(mapping["b01"], {"single": "candidate"})
            self.assertEqual(set(mapping["t01"].keys()), {"A", "B"})
            self.assertEqual(set(mapping["t01"].values()), {"control", "candidate"})

            # Refuse collision if review bundle or key already exists
            with self.assertRaises(FileExistsError):
                export_review(proto_dir / "protocol.json", results_path, review_bundle_path)

    # 8. Export review rejects tampered results
    def test_export_review_rejects_tampered_results(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            proto_dir = Path(tmp_dir) / "protocol_workspace"
            transport = FakeTransport()
            prepare(proto_dir, transport=transport)

            results_path = Path(tmp_dir) / "results.json"
            handler = FakeExecutionHandler()
            run(proto_dir / "protocol.json", results_path, transport=transport, execute_fn=handler)

            valid_res = json.loads(results_path.read_text(encoding="utf-8"))
            review_bundle_path = Path(tmp_dir) / "review.json"

            # Tamper 1: Altered formatted answer
            tampered_res = copy.deepcopy(valid_res)
            tampered_res["attempts"][0]["formatted_answer"] = "tampered text"
            t_path = Path(tmp_dir) / "t1_results.json"
            t_path.write_text(json.dumps(tampered_res), encoding="utf-8")
            with self.assertRaises(ValueError) as ctx:
                export_review(proto_dir / "protocol.json", t_path, review_bundle_path)
            self.assertIn("formatted_answer altered", str(ctx.exception))

            # Tamper 2: Invalid raw attempt canonical sha
            tampered_res = copy.deepcopy(valid_res)
            tampered_res["attempts"][0]["raw_attempt_canonical_sha256"] = "0" * 64
            t_path = Path(tmp_dir) / "t2_results.json"
            t_path.write_text(json.dumps(tampered_res), encoding="utf-8")
            with self.assertRaises(ValueError) as ctx:
                export_review(proto_dir / "protocol.json", t_path, review_bundle_path)
            self.assertIn("raw_attempt_canonical_sha256 mismatch", str(ctx.exception))

            # Tamper 3: Changed input hash
            tampered_res = copy.deepcopy(valid_res)
            tampered_res["attempts"][0]["input_hashes"]["pages_sha256"] = "0" * 64
            t_path = Path(tmp_dir) / "t3_results.json"
            t_path.write_text(json.dumps(tampered_res), encoding="utf-8")
            with self.assertRaises(ValueError) as ctx:
                export_review(proto_dir / "protocol.json", t_path, review_bundle_path)
            self.assertIn("input pages_sha256 mismatch", str(ctx.exception))

    # =========================================================================
    # Adversarial tests for Senior Auditor's 8 findings
    # =========================================================================

    # Finding 1: NO TypeError retry; execute_fn called exactly once per condition
    def test_run_calls_execute_fn_exactly_once_retaining_internal_typeerror(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            proto_dir = Path(tmp_dir) / "protocol_workspace"
            transport = FakeTransport()
            prepare(proto_dir, transport=transport)

            results_path = Path(tmp_dir) / "results.json"
            # Simulate internal TypeError on attempt 7
            handler = FakeExecutionHandler(type_error_at_indices={7})
            results = run(proto_dir / "protocol.json", results_path, transport=transport, execute_fn=handler)

            # Handler must have been called EXACTLY 31 times (no retry on attempt 7)
            self.assertEqual(len(handler.recorded_calls), 31)
            call7 = handler.recorded_calls[6]
            self.assertEqual(call7["attempt_index"], 7)
            self.assertTrue(call7["has_table_bundle"])
            self.assertTrue(call7["enabled"])

            # Attempt 7 must be retained as technical error
            att7 = results["attempts"][6]
            self.assertEqual(att7["status"], "error")
            self.assertIn("Simulated internal TypeError", att7["error"])
            self.assertEqual(results["summary"]["completed_attempts"], 30)
            self.assertEqual(results["summary"]["error_attempts"], 1)

    # Finding 2: Recompute and compare canonical SHA; reject fake string format
    def test_validate_protocol_rejects_fake_canonical_sha_and_tampered_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            proto_dir = Path(tmp_dir) / "protocol_workspace"
            transport = FakeTransport()
            protocol = prepare(proto_dir, transport=transport)

            # 1. Fake string in canonical_sha256 (not 64-char hex)
            bad_p = copy.deepcopy(protocol)
            bad_p["generated_assets"]["layout_pages.jsonl"]["canonical_sha256"] = "fake_sha_not_hex"
            with self.assertRaises(ValueError) as ctx:
                validate_protocol(bad_p, protocol_dir=proto_dir)
            self.assertIn("invalid canonical_sha256 format", str(ctx.exception).lower())

            # 2. Tampered content on disk with updated raw_sha but original canonical_sha
            pages_p = proto_dir / "layout_pages.jsonl"
            orig_bytes = pages_p.read_bytes()
            pages_lines = [json.loads(line) for line in orig_bytes.decode("utf-8").splitlines() if line.strip()]
            pages_lines[0]["text"] = "TEXTO_ALTERADO"
            new_bytes = "\n".join(json.dumps(l, sort_keys=True) for l in pages_lines).encode("utf-8") + b"\n"
            pages_p.write_bytes(new_bytes)

            bad_p2 = copy.deepcopy(protocol)
            bad_p2["generated_assets"]["layout_pages.jsonl"]["raw_sha256"] = hashlib.sha256(new_bytes).hexdigest()
            # canonical_sha256 left unchanged: must reject because recomputed canonical SHA does not match
            with self.assertRaises(ValueError) as ctx:
                validate_protocol(bad_p2, protocol_dir=proto_dir)
            self.assertIn("canonical sha mismatch", str(ctx.exception).lower())
            pages_p.write_bytes(orig_bytes)

    # Finding 3: Bind copied assets to frozen historical assets & structured bundle
    def test_validate_protocol_rejects_index_vector_permutation_and_bundle_substitution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            proto_dir = Path(tmp_dir) / "protocol_workspace"
            transport = FakeTransport()
            protocol = prepare(proto_dir, transport=transport)

            # Untouched base must validate cleanly
            validate_protocol(protocol, protocol_dir=proto_dir)

            # 1. Permute vectors in layout_index.json and update its self raw & canonical SHA
            idx_p = proto_dir / "layout_index.json"
            orig_idx_bytes = idx_p.read_bytes()
            idx_data = json.loads(orig_idx_bytes.decode("utf-8"))
            # Permute chunk order
            idx_data["chunks"] = idx_data["chunks"][::-1]
            tampered_bytes = json.dumps(idx_data, sort_keys=True).encode("utf-8")
            idx_p.write_bytes(tampered_bytes)

            bad_p = copy.deepcopy(protocol)
            bad_p["generated_assets"]["layout_index.json"]["raw_sha256"] = hashlib.sha256(tampered_bytes).hexdigest()
            bad_p["generated_assets"]["layout_index.json"]["canonical_sha256"] = compute_canonical_sha256(tampered_bytes)

            with self.assertRaises(ValueError) as ctx:
                validate_protocol(bad_p, protocol_dir=proto_dir)
            self.assertTrue(any(token in str(ctx.exception).lower() for token in ("historical", "canonical", "match", "mismatch", "drift")))
            idx_p.write_bytes(orig_idx_bytes)

            # 2. Substitute structured.json with empty/modified dict and update self hashes
            struct_p = proto_dir / "structured.json"
            orig_struct_bytes = struct_p.read_bytes()
            tampered_struct = {"tables": [], "pages": [], "metadata": {"doc_id": "fake"}}
            tampered_bytes = json.dumps(tampered_struct, sort_keys=True).encode("utf-8")
            struct_p.write_bytes(tampered_bytes)

            bad_p2 = copy.deepcopy(protocol)
            bad_p2["generated_assets"]["structured.json"]["raw_sha256"] = hashlib.sha256(tampered_bytes).hexdigest()
            bad_p2["generated_assets"]["structured.json"]["canonical_sha256"] = compute_canonical_sha256(tampered_bytes)

            with self.assertRaises(ValueError) as ctx:
                validate_protocol(bad_p2, protocol_dir=proto_dir)
            self.assertTrue(any(token in str(ctx.exception).lower() for token in ("authoritative", "bundle", "structured", "match", "mismatch", "drift")))
            struct_p.write_bytes(orig_struct_bytes)

            # Ensure no execution or export artifacts were created during validation
            self.assertFalse((proto_dir / "results.json").exists())
            self.assertFalse((proto_dir / "review_bundle.json").exists())
            self.assertFalse((proto_dir / "review_key.json").exists())

    # Finding 4: prepare verifies historical protocol and assets before copying + CRLF consistency
    def test_prepare_verifies_historical_bindings_and_crlf_raw_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            proto_dir = Path(tmp_dir) / "protocol_workspace"
            transport = FakeTransport()
            protocol = prepare(proto_dir, transport=transport)

            # Protocol reference_freeze_sha256 must match raw bytes SHA of reference_freeze.json
            p_ref = PROJECT_ROOT / DEFAULT_REFERENCE_FREEZE_PATH
            expected_ref_sha = hashlib.sha256(p_ref.read_bytes()).hexdigest()
            self.assertEqual(protocol["reference_freeze_sha256"], expected_ref_sha)

            # Historical protocol sha matches
            self.assertEqual(protocol["historical_protocol_sha256"], HISTORICAL_PROTOCOL_SHA256)

    # Finding 5: export_review revalidates current protocol and assets
    def test_export_review_revalidates_protocol_and_assets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            proto_dir = Path(tmp_dir) / "protocol_workspace"
            transport = FakeTransport()
            prepare(proto_dir, transport=transport)

            results_path = Path(tmp_dir) / "results.json"
            handler = FakeExecutionHandler()
            run(proto_dir / "protocol.json", results_path, transport=transport, execute_fn=handler)

            # Tamper protocol.json after run completes
            proto_file = proto_dir / "protocol.json"
            proto_data = json.loads(proto_file.read_text(encoding="utf-8"))
            proto_data["server_version"] = "9.99.99"
            proto_file.write_text(json.dumps(proto_data), encoding="utf-8")

            review_bundle_path = Path(tmp_dir) / "review.json"
            with self.assertRaises(ValueError) as ctx:
                export_review(proto_dir / "protocol.json", results_path, review_bundle_path)
            self.assertIn("server version mismatch", str(ctx.exception).lower())
            # Neither artifact written
            self.assertFalse(review_bundle_path.exists())
            self.assertFalse((Path(tmp_dir) / "review_key.json").exists())

    # Finding 6: export_review binds all outer fields strictly to raw_attempt
    def test_export_review_rejects_altered_coverage_error_or_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            proto_dir = Path(tmp_dir) / "protocol_workspace"
            transport = FakeTransport()
            prepare(proto_dir, transport=transport)

            results_path = Path(tmp_dir) / "results.json"
            handler = FakeExecutionHandler()
            run(proto_dir / "protocol.json", results_path, transport=transport, execute_fn=handler)

            valid_res = json.loads(results_path.read_text(encoding="utf-8"))

            # First, verify untouched base exports cleanly
            out_valid = Path(tmp_dir) / "out_valid.json"
            export_review(proto_dir / "protocol.json", results_path, out_valid)
            self.assertTrue(out_valid.is_file())
            out_valid.unlink()
            (Path(tmp_dir) / "review_key.json").unlink()

            # 1. Coverage-only tampering against valid raw-bound base
            tampered_cov = copy.deepcopy(valid_res)
            tampered_cov["attempts"][0]["coverage_publication"] = {"claims": [{"claim_id": "forged", "text": "falso"}]}
            t_cov_path = Path(tmp_dir) / "t_cov.json"
            t_cov_path.write_text(json.dumps(tampered_cov), encoding="utf-8")
            out_cov = Path(tmp_dir) / "out_cov.json"
            with self.assertRaises(ValueError):
                export_review(proto_dir / "protocol.json", t_cov_path, out_cov)
            self.assertFalse(out_cov.exists())
            self.assertFalse((Path(tmp_dir) / "review_key.json").exists())

            # 2. Error-only tampering against valid raw-bound base (completed status, raw error None)
            tampered_err = copy.deepcopy(valid_res)
            tampered_err["attempts"][0]["error"] = "injected error"
            t_err_path = Path(tmp_dir) / "t_err.json"
            t_err_path.write_text(json.dumps(tampered_err), encoding="utf-8")
            out_err = Path(tmp_dir) / "out_err.json"
            with self.assertRaises(ValueError):
                export_review(proto_dir / "protocol.json", t_err_path, out_err)
            self.assertFalse(out_err.exists())
            self.assertFalse((Path(tmp_dir) / "review_key.json").exists())

            # 3. Status-only tampering against valid raw-bound base (error remains None)
            tampered_stat = copy.deepcopy(valid_res)
            tampered_stat["attempts"][0]["status"] = "error"
            t_stat_path = Path(tmp_dir) / "t_stat.json"
            t_stat_path.write_text(json.dumps(tampered_stat), encoding="utf-8")
            out_stat = Path(tmp_dir) / "out_stat.json"
            with self.assertRaises(ValueError):
                export_review(proto_dir / "protocol.json", t_stat_path, out_stat)
            self.assertFalse(out_stat.exists())
            self.assertFalse((Path(tmp_dir) / "review_key.json").exists())

            # 4. Error tampering on an attempt that genuinely had a technical error
            results_err_path = Path(tmp_dir) / "results_with_err.json"
            handler_err = FakeExecutionHandler(fail_at_indices={1})
            run(proto_dir / "protocol.json", results_err_path, transport=transport, execute_fn=handler_err)
            valid_err_res = json.loads(results_err_path.read_text(encoding="utf-8"))

            tampered_err_rec = copy.deepcopy(valid_err_res)
            tampered_err_rec["attempts"][0]["error"] = "altered error message"
            t_err2_path = Path(tmp_dir) / "t_err2.json"
            t_err2_path.write_text(json.dumps(tampered_err_rec), encoding="utf-8")
            out_err2 = Path(tmp_dir) / "out_err2.json"
            with self.assertRaises(ValueError):
                export_review(proto_dir / "protocol.json", t_err2_path, out_err2)
            self.assertFalse(out_err2.exists())
            self.assertFalse((Path(tmp_dir) / "review_key.json").exists())

    # Finding 7: Collision of output_path equal to review_key.json
    def test_export_review_rejects_output_path_equal_to_review_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            proto_dir = Path(tmp_dir) / "protocol_workspace"
            transport = FakeTransport()
            prepare(proto_dir, transport=transport)

            results_path = Path(tmp_dir) / "results.json"
            handler = FakeExecutionHandler()
            run(proto_dir / "protocol.json", results_path, transport=transport, execute_fn=handler)

            review_dir = Path(tmp_dir) / "collision_dir"
            collision_path = review_dir / "review_key.json"

            # Must fail with ValueError BEFORE writing either file
            with self.assertRaises(ValueError) as ctx:
                export_review(proto_dir / "protocol.json", results_path, collision_path)
            self.assertIn("cannot be named review_key.json", str(ctx.exception))
            self.assertFalse(collision_path.exists())
            self.assertFalse((review_dir / "review_bundle.json").exists())

            # Case-variant on Windows (e.g. REVIEW_KEY.JSON)
            collision_path_upper = review_dir / "REVIEW_KEY.JSON"
            with self.assertRaises(ValueError) as ctx:
                export_review(proto_dir / "protocol.json", results_path, collision_path_upper)
            self.assertIn("cannot be named review_key.json", str(ctx.exception))
            self.assertFalse(collision_path_upper.exists())

    # Finding 8: execute_fn returning {} or invalid shape does NOT count as completed
    def test_run_rejects_empty_or_invalid_attempt_result_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            proto_dir = Path(tmp_dir) / "protocol_workspace"
            transport = FakeTransport()
            prepare(proto_dir, transport=transport)

            results_path = Path(tmp_dir) / "results.json"
            # Handler returns {} on attempt 3
            handler = FakeExecutionHandler(empty_at_indices={3})
            results = run(proto_dir / "protocol.json", results_path, transport=transport, execute_fn=handler)

            self.assertEqual(results["run_status"], "completed")
            self.assertEqual(results["summary"]["completed_attempts"], 30)
            self.assertEqual(results["summary"]["error_attempts"], 1)

            att3 = results["attempts"][2]
            self.assertEqual(att3["status"], "error")
            self.assertIn("Technical invalid attempt result", att3["error"])
            self.assertTrue(att3["raw_attempt"]["synthetic_exception_record"])

    # Regression P1 runner697: Binding query copies to authoritative originals
    def test_p1_runner697_binding_query_copies_to_authoritative_originals(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            transport = FakeTransport()

            # 1. Primary cohort: adversary modifies primary_queries.json copy + hashes + schedule while full_inputs.json unchanged
            proto_dir_prim = Path(tmp_dir) / "protocol_workspace_prim"
            protocol = prepare(proto_dir_prim, transport=transport)

            # Untouched base must validate cleanly
            validate_protocol(protocol, protocol_dir=proto_dir_prim)

            p_prim_copy = proto_dir_prim / "primary_queries.json"
            prim_obj = json.loads(p_prim_copy.read_text(encoding="utf-8"))
            prim_obj["cases"][0]["query"] = "Consulta primaria alterada por el adversario"
            new_prim_bytes = (json.dumps(prim_obj, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
            p_prim_copy.write_bytes(new_prim_bytes)

            tampered_proto = copy.deepcopy(protocol)
            tampered_proto["generated_assets"]["primary_queries.json"]["raw_sha256"] = hashlib.sha256(new_prim_bytes).hexdigest()
            tampered_proto["generated_assets"]["primary_queries.json"]["canonical_sha256"] = compute_canonical_sha256(new_prim_bytes)
            # Rebuild schedule with altered primary query
            tbl_obj = json.loads((proto_dir_prim / "table_queries.json").read_text(encoding="utf-8"))
            ext_obj = json.loads((proto_dir_prim / "extra_queries.json").read_text(encoding="utf-8"))
            tampered_proto["schedule"] = build_schedule(prim_obj, tbl_obj, ext_obj)

            with self.assertRaises(ValueError) as ctx_prim:
                validate_protocol(tampered_proto, protocol_dir=proto_dir_prim)
            self.assertTrue(
                any(s in str(ctx_prim.exception) for s in ("canonical_sha256 does not match authoritative original", "drifted from authoritative original", "does not match schedule built from authoritative")),
                f"Unexpected exception message: {ctx_prim.exception}",
            )

            # Protocol-only validation (without protocol_dir) must also reject
            with self.assertRaises(ValueError):
                validate_protocol(tampered_proto)

            # 2. Table cohort: adversary modifies table_queries.json copy + hashes + schedule while secondary_queries.json unchanged
            proto_dir_tbl = Path(tmp_dir) / "protocol_workspace_tbl"
            protocol_tbl = prepare(proto_dir_tbl, transport=transport)
            validate_protocol(protocol_tbl, protocol_dir=proto_dir_tbl)

            p_tbl_copy = proto_dir_tbl / "table_queries.json"
            tbl_obj = json.loads(p_tbl_copy.read_text(encoding="utf-8"))
            tbl_obj["cases"][0]["query"] = "Consulta tabular alterada por el adversario"
            new_tbl_bytes = (json.dumps(tbl_obj, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
            p_tbl_copy.write_bytes(new_tbl_bytes)

            tampered_tbl_proto = copy.deepcopy(protocol_tbl)
            tampered_tbl_proto["generated_assets"]["table_queries.json"]["raw_sha256"] = hashlib.sha256(new_tbl_bytes).hexdigest()
            tampered_tbl_proto["generated_assets"]["table_queries.json"]["canonical_sha256"] = compute_canonical_sha256(new_tbl_bytes)
            prim_orig = json.loads((proto_dir_tbl / "primary_queries.json").read_text(encoding="utf-8"))
            ext_orig = json.loads((proto_dir_tbl / "extra_queries.json").read_text(encoding="utf-8"))
            tampered_tbl_proto["schedule"] = build_schedule(prim_orig, tbl_obj, ext_orig)

            with self.assertRaises(ValueError):
                validate_protocol(tampered_tbl_proto, protocol_dir=proto_dir_tbl)
            with self.assertRaises(ValueError):
                validate_protocol(tampered_tbl_proto)

            # 3. Extra cohort: adversary modifies extra_queries.json copy + hashes + schedule while authoritative extra_queries.json unchanged
            proto_dir_ext = Path(tmp_dir) / "protocol_workspace_ext"
            protocol_ext = prepare(proto_dir_ext, transport=transport)
            validate_protocol(protocol_ext, protocol_dir=proto_dir_ext)

            p_ext_copy = proto_dir_ext / "extra_queries.json"
            ext_obj = json.loads(p_ext_copy.read_text(encoding="utf-8"))
            ext_obj["cases"][0]["query"] = "Consulta extra alterada por el adversario"
            new_ext_bytes = (json.dumps(ext_obj, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
            p_ext_copy.write_bytes(new_ext_bytes)

            tampered_ext_proto = copy.deepcopy(protocol_ext)
            tampered_ext_proto["generated_assets"]["extra_queries.json"]["raw_sha256"] = hashlib.sha256(new_ext_bytes).hexdigest()
            tampered_ext_proto["generated_assets"]["extra_queries.json"]["canonical_sha256"] = compute_canonical_sha256(new_ext_bytes)
            prim_orig_ext = json.loads((proto_dir_ext / "primary_queries.json").read_text(encoding="utf-8"))
            tbl_orig_ext = json.loads((proto_dir_ext / "table_queries.json").read_text(encoding="utf-8"))
            tampered_ext_proto["schedule"] = build_schedule(prim_orig_ext, tbl_orig_ext, ext_obj)

            with self.assertRaises(ValueError):
                validate_protocol(tampered_ext_proto, protocol_dir=proto_dir_ext)
            with self.assertRaises(ValueError):
                validate_protocol(tampered_ext_proto)

    # Regression P1 runner1338: export_review validates raw_attempt authoritatively
    def test_p1_runner1338_export_review_validates_raw_attempt_authoritatively(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            proto_dir = Path(tmp_dir) / "protocol_workspace"
            transport = FakeTransport()
            prepare(proto_dir, transport=transport)

            results_path = Path(tmp_dir) / "results.json"
            handler = FakeExecutionHandler()
            results = run(proto_dir / "protocol.json", results_path, transport=transport, execute_fn=handler)

            # Base export succeeds
            out_base = Path(tmp_dir) / "review_base.json"
            export_review(proto_dir / "protocol.json", results_path, out_base)
            self.assertTrue(out_base.is_file())
            (Path(tmp_dir) / "review_key.json").unlink()

            # 1. Adversary reduces completed raw to identity/status/answer with coverage None + recomputed raw SHA
            tampered_reduced = copy.deepcopy(results)
            att0 = tampered_reduced["attempts"][0]
            raw0 = att0["raw_attempt"]
            reduced_raw = {
                "arm": "mixed",
                "case_id": raw0["case_id"],
                "query": raw0["query"],
                "attempt_index": raw0["attempt_index"],
                "status": "completed",
                "formatted_answer": raw0["formatted_answer"],
                "coverage_publication": None,
            }
            att0["raw_attempt"] = reduced_raw
            att0["coverage_publication"] = None
            att0["raw_attempt_canonical_sha256"] = hashlib.sha256(
                json.dumps(reduced_raw, sort_keys=True, ensure_ascii=False, allow_nan=False).encode("utf-8")
            ).hexdigest()

            p_tampered_reduced = Path(tmp_dir) / "results_reduced.json"
            p_tampered_reduced.write_text(json.dumps(tampered_reduced), encoding="utf-8")
            out_reduced = Path(tmp_dir) / "out_reduced.json"

            with self.assertRaises(ValueError) as ctx_red:
                export_review(proto_dir / "protocol.json", p_tampered_reduced, out_reduced)
            self.assertIn("failed authoritative validation", str(ctx_red.exception))
            self.assertFalse(out_reduced.exists())
            self.assertFalse((Path(tmp_dir) / "review_key.json").exists())

            # 2. Missing table_audit in completed raw
            tampered_no_audit = copy.deepcopy(results)
            att0_no_audit = tampered_no_audit["attempts"][0]
            raw_no_audit = copy.deepcopy(att0_no_audit["raw_attempt"])
            del raw_no_audit["table_audit"]
            att0_no_audit["raw_attempt"] = raw_no_audit
            att0_no_audit["raw_attempt_canonical_sha256"] = hashlib.sha256(
                json.dumps(raw_no_audit, sort_keys=True, ensure_ascii=False, allow_nan=False).encode("utf-8")
            ).hexdigest()

            p_no_audit = Path(tmp_dir) / "results_no_audit.json"
            p_no_audit.write_text(json.dumps(tampered_no_audit), encoding="utf-8")
            out_no_audit = Path(tmp_dir) / "out_no_audit.json"

            with self.assertRaises(ValueError) as ctx_audit:
                export_review(proto_dir / "protocol.json", p_no_audit, out_no_audit)
            self.assertIn("failed authoritative validation", str(ctx_audit.exception))
            self.assertFalse(out_no_audit.exists())

            # 3. Missing factual_outcome in completed raw
            tampered_no_fo = copy.deepcopy(results)
            att0_no_fo = tampered_no_fo["attempts"][0]
            raw_no_fo = copy.deepcopy(att0_no_fo["raw_attempt"])
            del raw_no_fo["factual_outcome"]
            att0_no_fo["raw_attempt"] = raw_no_fo
            att0_no_fo["raw_attempt_canonical_sha256"] = hashlib.sha256(
                json.dumps(raw_no_fo, sort_keys=True, ensure_ascii=False, allow_nan=False).encode("utf-8")
            ).hexdigest()

            p_no_fo = Path(tmp_dir) / "results_no_fo.json"
            p_no_fo.write_text(json.dumps(tampered_no_fo), encoding="utf-8")
            out_no_fo = Path(tmp_dir) / "out_no_fo.json"

            with self.assertRaises(ValueError) as ctx_fo:
                export_review(proto_dir / "protocol.json", p_no_fo, out_no_fo)
            self.assertIn("failed authoritative validation", str(ctx_fo.exception))
            self.assertFalse(out_no_fo.exists())

            # 4. Missing coverage_audit in completed raw
            tampered_no_ca = copy.deepcopy(results)
            att0_no_ca = tampered_no_ca["attempts"][0]
            raw_no_ca = copy.deepcopy(att0_no_ca["raw_attempt"])
            del raw_no_ca["coverage_audit"]
            att0_no_ca["raw_attempt"] = raw_no_ca
            att0_no_ca["raw_attempt_canonical_sha256"] = hashlib.sha256(
                json.dumps(raw_no_ca, sort_keys=True, ensure_ascii=False, allow_nan=False).encode("utf-8")
            ).hexdigest()

            p_no_ca = Path(tmp_dir) / "results_no_ca.json"
            p_no_ca.write_text(json.dumps(tampered_no_ca), encoding="utf-8")
            out_no_ca = Path(tmp_dir) / "out_no_ca.json"

            with self.assertRaises(ValueError) as ctx_ca:
                export_review(proto_dir / "protocol.json", p_no_ca, out_no_ca)
            self.assertIn("failed authoritative validation", str(ctx_ca.exception))
            self.assertFalse(out_no_ca.exists())

            # 5. Positive control: numeric guard abstention is exportable
            abstain_results = copy.deepcopy(results)
            att0_abs = abstain_results["attempts"][0]
            att0_abs["raw_attempt"]["factual_outcome"] = {
                "status": "insufficient_evidence",
                "claims": [],
                "reason": "Abstención",
            }
            att0_abs["raw_attempt"]["coverage_publication"] = {
                "status": "insufficient_evidence",
                "parts": [{"query_span": att0_abs["query"], "decision": "not_answered", "claim_ids": [], "reason_code": "guard_refused"}],
            }
            att0_abs["coverage_publication"] = copy.deepcopy(att0_abs["raw_attempt"]["coverage_publication"])
            att0_abs["raw_attempt_canonical_sha256"] = hashlib.sha256(
                json.dumps(att0_abs["raw_attempt"], sort_keys=True, ensure_ascii=False, allow_nan=False).encode("utf-8")
            ).hexdigest()

            p_abs = Path(tmp_dir) / "results_abstain.json"
            p_abs.write_text(json.dumps(abstain_results), encoding="utf-8")
            out_abs = Path(tmp_dir) / "out_abstain.json"
            export_review(proto_dir / "protocol.json", p_abs, out_abs)
            self.assertTrue(out_abs.is_file())
            (Path(tmp_dir) / "review_key.json").unlink()

            # 6. Positive control: legitimate error records from technical exceptions before NN are exportable
            err_results_path = Path(tmp_dir) / "results_with_error.json"
            handler_with_err = FakeExecutionHandler(fail_at_indices={1})
            run(proto_dir / "protocol.json", err_results_path, transport=transport, execute_fn=handler_with_err)
            out_err = Path(tmp_dir) / "out_err_export.json"
            export_review(proto_dir / "protocol.json", err_results_path, out_err)
            self.assertTrue(out_err.is_file())


class TestRealExecuteTableAttemptInterfaceAndRunnerValidation(unittest.TestCase):
    """Regression test executing REAL execute_table_attempt with PipelineFakeTransport

    and validating results against _validate_attempt_result.
    Verifies:
    - CONTROL explicit table query produces table_audit: {enabled: False, is_table_query: False, gate_decision: 'bypassed'}
      and passes runner validation.
    - CANDIDATE explicit table query produces table_audit: {enabled: True, is_table_query: True, gate_decision: 'accepted'}
      and passes runner validation.
    - CANDIDATE narrative query delegates unchanged (is_table_query: False, gate_decision: 'bypassed')
      and passes runner validation.
    - Mock only NN (PipelineFakeTransport), NEVER audit flags or runner validator.
    - Execute called exactly once per condition (no second execute).
    - Status and fact-shape counterexamples are strictly rejected:
      * status 'answered'
      * status 'unknown'
      * non-dict factual_outcome
      * facts containing strings rather than dicts
      * missing coverage_audit
      * control attempt with is_table_query=True
    - Numeric guard abstention preserved as completed attempt.
    """

    def setUp(self) -> None:
        self.bundle, self.index, self.pages, self.units = _build_synthetic_fixtures()
        self.transport = PipelineFakeTransport()

    def test_control_explicit_table_query_accepted_with_bypass_false(self) -> None:
        attempt_spec = {
            "attempt_index": 12,
            "case_id": "t01",
            "query": "Cuadro 1 Empleo en 2025",
            "arm": "mixed",
        }
        scheduled = {
            "attempt_index": 12,
            "case_id": "t01",
            "query": "Cuadro 1 Empleo en 2025",
            "cohort": "table",
            "condition": "control",
            "arm": "mixed",
            "table_enabled": False,
        }
        res = execute_table_attempt(
            attempt_spec,
            index_data=self.index,
            units_data=self.units,
            pages_data=self.pages,
            transport=self.transport,
            table_bundle=self.bundle,
            enabled=False,
        )

        self.assertEqual(res["table_audit"]["enabled"], False)
        self.assertEqual(res["table_audit"]["is_table_query"], False)
        self.assertEqual(res["table_audit"]["gate_decision"], "bypassed")
        self.assertEqual(res["status"], "completed")

        is_valid, err = _validate_attempt_result(res, scheduled)
        self.assertTrue(is_valid, f"Control attempt failed validation unexpectedly: {err}")
        self.assertIsNone(err)

    def test_candidate_explicit_table_query_truthful_routing(self) -> None:
        attempt_spec = {
            "attempt_index": 13,
            "case_id": "t01",
            "query": "Cuadro 1 Empleo en 2025",
            "arm": "mixed",
        }
        scheduled = {
            "attempt_index": 13,
            "case_id": "t01",
            "query": "Cuadro 1 Empleo en 2025",
            "cohort": "table",
            "condition": "candidate",
            "arm": "mixed",
            "table_enabled": True,
        }
        res = execute_table_attempt(
            attempt_spec,
            index_data=self.index,
            units_data=self.units,
            pages_data=self.pages,
            transport=self.transport,
            table_bundle=self.bundle,
            enabled=True,
        )

        self.assertEqual(res["table_audit"]["enabled"], True)
        self.assertEqual(res["table_audit"]["is_table_query"], True)
        self.assertEqual(res["table_audit"]["gate_decision"], "accepted")
        self.assertEqual(res["status"], "completed")

        is_valid, err = _validate_attempt_result(res, scheduled)
        self.assertTrue(is_valid, f"Candidate table attempt failed validation: {err}")
        self.assertIsNone(err)

    def test_candidate_narrative_query_delegates_unchanged(self) -> None:
        attempt_spec = {
            "attempt_index": 1,
            "case_id": "b01",
            "query": "En el escenario macroeconómico y fiscal",
            "arm": "mixed",
        }
        scheduled = {
            "attempt_index": 1,
            "case_id": "b01",
            "query": "En el escenario macroeconómico y fiscal",
            "cohort": "primary",
            "condition": "candidate",
            "arm": "mixed",
            "table_enabled": True,
        }
        res = execute_table_attempt(
            attempt_spec,
            index_data=self.index,
            units_data=self.units,
            pages_data=self.pages,
            transport=self.transport,
            table_bundle=self.bundle,
            enabled=True,
        )

        self.assertEqual(res["table_audit"]["enabled"], True)
        self.assertEqual(res["table_audit"]["is_table_query"], False)
        self.assertEqual(res["table_audit"]["gate_decision"], "bypassed")
        self.assertEqual(res["status"], "completed")

        is_valid, err = _validate_attempt_result(res, scheduled)
        self.assertTrue(is_valid, f"Candidate narrative attempt failed validation: {err}")
        self.assertIsNone(err)

    def test_validator_rejects_counterexamples_and_tampered_shapes(self) -> None:
        attempt_spec = {
            "attempt_index": 13,
            "case_id": "t01",
            "query": "Cuadro 1 Empleo en 2025",
            "arm": "mixed",
        }
        scheduled = {
            "attempt_index": 13,
            "case_id": "t01",
            "query": "Cuadro 1 Empleo en 2025",
            "cohort": "table",
            "condition": "candidate",
            "arm": "mixed",
            "table_enabled": True,
        }
        base_res = execute_table_attempt(
            attempt_spec,
            index_data=self.index,
            units_data=self.units,
            pages_data=self.pages,
            transport=self.transport,
            table_bundle=self.bundle,
            enabled=True,
        )

        # Untouched base must validate before any mutations
        is_valid_base, err_base = _validate_attempt_result(base_res, scheduled)
        self.assertTrue(is_valid_base, f"Untouched base must validate before mutations: {err_base}")
        self.assertIsNone(err_base)

        # 1. Reject fake outer status 'answered'
        bad = copy.deepcopy(base_res)
        bad["status"] = "answered"
        is_valid, err = _validate_attempt_result(bad, scheduled)
        self.assertFalse(is_valid)
        self.assertIn("legacy raw status must be 'completed' or 'error'", str(err))

        # 2. Reject unknown status
        bad = copy.deepcopy(base_res)
        bad["status"] = "unknown"
        is_valid, err = _validate_attempt_result(bad, scheduled)
        self.assertFalse(is_valid)
        self.assertIn("legacy raw status must be 'completed' or 'error'", str(err))

        # 3. Reject non-dict factual_outcome
        bad = copy.deepcopy(base_res)
        bad["factual_outcome"] = "answered"
        is_valid, err = _validate_attempt_result(bad, scheduled)
        self.assertFalse(is_valid)
        self.assertIn("factual_outcome must be a dict", str(err))

        # 4. Reject invalid factual_outcome status 'supported' and 'unknown'
        bad = copy.deepcopy(base_res)
        bad["factual_outcome"]["status"] = "supported"
        is_valid, err = _validate_attempt_result(bad, scheduled)
        self.assertFalse(is_valid)
        self.assertIn("factual_outcome status must be 'answered', 'partial', or 'insufficient_evidence'", str(err))

        bad_unk = copy.deepcopy(base_res)
        bad_unk["factual_outcome"]["status"] = "unknown"
        is_valid_unk, err_unk = _validate_attempt_result(bad_unk, scheduled)
        self.assertFalse(is_valid_unk)
        self.assertIn("factual_outcome status must be 'answered', 'partial', or 'insufficient_evidence'", str(err_unk))

        # 5. Reject string facts in factual_outcome
        bad = copy.deepcopy(base_res)
        bad["factual_outcome"] = {"status": "answered", "facts": ["string_fact"]}
        is_valid, err = _validate_attempt_result(bad, scheduled)
        self.assertFalse(is_valid)
        self.assertIn("list of dicts", str(err))

        # 6. Reject missing coverage_audit
        bad = copy.deepcopy(base_res)
        del bad["coverage_audit"]
        is_valid, err = _validate_attempt_result(bad, scheduled)
        self.assertFalse(is_valid)
        self.assertIn("missing coverage_audit", str(err))

        # 7. Reject control attempt with is_table_query=True
        scheduled_ctrl = {
            "attempt_index": 12,
            "case_id": "t01",
            "query": "Cuadro 1 Empleo en 2025",
            "cohort": "table",
            "condition": "control",
            "arm": "mixed",
            "table_enabled": False,
        }
        ctrl_res = execute_table_attempt(
            {"attempt_index": 12, "case_id": "t01", "query": "Cuadro 1 Empleo en 2025", "arm": "mixed"},
            index_data=self.index,
            units_data=self.units,
            pages_data=self.pages,
            transport=self.transport,
            table_bundle=self.bundle,
            enabled=False,
        )
        is_valid_ctrl, err_ctrl = _validate_attempt_result(ctrl_res, scheduled_ctrl)
        self.assertTrue(is_valid_ctrl, f"Untouched control base must validate: {err_ctrl}")
        self.assertIsNone(err_ctrl)

        bad_ctrl = copy.deepcopy(ctrl_res)
        bad_ctrl["table_audit"]["is_table_query"] = True
        is_valid, err = _validate_attempt_result(bad_ctrl, scheduled_ctrl)
        self.assertFalse(is_valid)
        self.assertIn("is_table_query=False", str(err))

    def test_numeric_guard_abstention_preserved_as_completed(self) -> None:
        attempt_spec = {
            "attempt_index": 13,
            "case_id": "t01",
            "query": "Cuadro 1 Empleo en 2025",
            "arm": "mixed",
        }
        scheduled = {
            "attempt_index": 13,
            "case_id": "t01",
            "query": "Cuadro 1 Empleo en 2025",
            "cohort": "table",
            "condition": "candidate",
            "arm": "mixed",
            "table_enabled": True,
        }
        res = execute_table_attempt(
            attempt_spec,
            index_data=self.index,
            units_data=self.units,
            pages_data=self.pages,
            transport=self.transport,
            table_bundle=self.bundle,
            enabled=True,
        )

        res_abstain = copy.deepcopy(res)
        res_abstain["status"] = "completed"
        res_abstain["factual_outcome"] = {
            "status": "insufficient_evidence",
            "claims": [],
            "reason": "Guardia numérica denegó la publicación.",
        }
        res_abstain["coverage_publication"] = {
            "status": "insufficient_evidence",
            "parts": [
                {
                    "query_span": "Cuadro 1 Empleo en 2025",
                    "decision": "not_answered",
                    "claim_ids": [],
                    "reason_code": "not_in_evaluated_evidence",
                }
            ],
        }
        res_abstain["table_audit"]["gate_decision"] = "refused"

        is_valid, err = _validate_attempt_result(res_abstain, scheduled)
        self.assertTrue(is_valid, f"Numeric guard abstention should be valid completed attempt, got: {err}")
        self.assertIsNone(err)

    def test_single_execution_per_condition_no_second_execute(self) -> None:
        call_count = 0

        def wrapped_execute(*args: Any, **kwargs: Any) -> dict[str, Any]:
            nonlocal call_count
            call_count += 1
            return execute_table_attempt(*args, **kwargs)

        attempt_spec = {
            "attempt_index": 12,
            "case_id": "t01",
            "query": "Cuadro 1 Empleo en 2025",
            "arm": "mixed",
        }
        res = wrapped_execute(
            attempt_spec,
            index_data=self.index,
            units_data=self.units,
            pages_data=self.pages,
            transport=self.transport,
            table_bundle=self.bundle,
            enabled=False,
        )
        self.assertEqual(call_count, 1, "execute_table_attempt must be called exactly once")
        self.assertEqual(res["table_audit"]["enabled"], False)

    def test_completed_factual_partial_accepted_in_validation_and_export(self) -> None:
        """Completed factual outcome status 'partial' is accepted through validation and export.

        Verifies:
        - Realistic producer-publication semantics matching preserve_evidence.publication
          (status 'partial', schema 'context-package-partial-publication-v1', retained accepted claims,
          bindings) from underlying grounding 'insufficient_evidence'.
        - _validate_attempt_result accepts the completed partial attempt.
        - export_review authoritatively accepts and exports the partial attempt without alteration.
        - Supported, unknown, and malformed factual outcome statuses remain rejected.
        - All prior binding, schema, and outer status checks remain strictly enforced.
        """
        # Synthetic underlying grounding result with insufficient_evidence status,
        # containing one accepted claim and one rejected claim, with full package binding.
        selected_evidence = [
            {
                "evidence_id": "ev_1",
                "source_id": "s1",
                "quote": "El saldo presupuestario evoluciona favorablemente en el escenario base.",
                "page_number": 95,
                "source_file": "documento_sintetico.pdf",
                "published_at": "2025-01-01",
            },
            {
                "evidence_id": "ev_2",
                "source_id": "s1",
                "quote": "Las proyecciones macroeconómicas respaldan la senda de estabilidad fiscal.",
                "page_number": 95,
                "source_file": "documento_sintetico.pdf",
                "published_at": "2025-01-01",
            },
        ]
        accepted_claim_text = "El saldo presupuestario evoluciona favorablemente."
        rejected_claim_text = "El déficit alcanzará niveles críticos en 2030."

        underlying_result = {
            "status": "insufficient_evidence",
            "claims": [],
            "reason": "La evidencia seleccionada solo respalda parcialmente la consulta.",
            "grounding": {
                "selected_evidence": selected_evidence,
                "repair_evidence": [],
                "initial_claims": [
                    {
                        "claim_id": "c1",
                        "text": accepted_claim_text,
                        "evidence_ids": ["ev_1", "ev_2"],
                    },
                    {
                        "claim_id": "c2",
                        "text": rejected_claim_text,
                        "evidence_ids": ["ev_1", "ev_2"],
                    },
                ],
                "repairs": [],
                "initial_verdicts": [
                    {
                        "claim_id": "c1",
                        "verdict": "supported",
                        "reason": "Apoyo semántico verificado en ev_1.",
                        "evidence_verdicts": {
                            "ev_1": {"relation": "supports"},
                            "ev_2": {"relation": "irrelevant"},
                        },
                    },
                    {
                        "claim_id": "c2",
                        "verdict": "contradicted",
                        "reason": "No respaldado por las evidencias disponibles.",
                        "evidence_verdicts": {
                            "ev_1": {"relation": "irrelevant"},
                            "ev_2": {"relation": "contradicts"},
                        },
                    },
                ],
                "accepted_claims": [
                    {
                        "claim_id": "c1",
                        "text": accepted_claim_text,
                        "evidence_ids": ["ev_1"],
                    }
                ],
                "rejected_claims": [
                    {
                        "claim_id": "c2",
                        "text": rejected_claim_text,
                        "evidence_ids": ["ev_1", "ev_2"],
                        "reason": "Contradicho por la evidencia evaluada.",
                    }
                ],
            },
        }

        snapshot_underlying = copy.deepcopy(underlying_result)
        factual_outcome = publication(underlying_result, preserve=True, allow_partial=True)

        # Producer semantics assertions
        self.assertEqual(underlying_result, snapshot_underlying, "Producer publication must not modify underlying result")
        self.assertEqual(underlying_result["status"], "insufficient_evidence", "Underlying status must stay insufficient_evidence")
        self.assertEqual(factual_outcome["status"], "partial", "Publication status must be 'partial'")
        self.assertEqual(factual_outcome["schema_version"], "context-package-partial-publication-v1")
        self.assertEqual(len(factual_outcome["claims"]), 1, "Only accepted claim must survive in publication")
        self.assertEqual(factual_outcome["claims"][0]["text"], accepted_claim_text)
        self.assertNotIn(rejected_claim_text, [c.get("text") for c in factual_outcome["claims"]])

        # Bindings: complete evidence package retained while separating verifier retained vs context reinstated
        self.assertEqual(len(factual_outcome["bindings"]), 1)
        binding = factual_outcome["bindings"][0]
        self.assertEqual(binding["claim_id"], "c1")
        self.assertEqual(binding["stage"], "generation")
        self.assertEqual(binding["package_evidence_ids"], ["ev_1", "ev_2"])
        self.assertEqual(binding["verifier_retained_evidence_ids"], ["ev_1"])
        self.assertEqual(binding["context_reinstated_ids"], ["ev_2"])

        scheduled = {
            "attempt_index": 1,
            "case_id": "b01",
            "query": "En el escenario macroeconómico y fiscal",
            "cohort": "primary",
            "condition": "candidate",
            "arm": "mixed",
            "table_enabled": True,
        }
        partial_attempt = {
            "attempt_index": 1,
            "case_id": "b01",
            "query": "En el escenario macroeconómico y fiscal",
            "arm": "mixed",
            "status": "completed",
            "formatted_answer": "El saldo presupuestario evoluciona favorablemente en el escenario base.",
            "factual_outcome": factual_outcome,
            "coverage_audit": {
                "status": "passed",
                "validated_payload": {"covered": True},
            },
            "coverage_publication": {
                "status": "partial",
                "claims": [
                    {"claim_id": "c1", "text": accepted_claim_text, "status": "supported"}
                ],
            },
            "lifecycle_records": [
                {"role": "generator", "model": DEFAULT_GENERATION_MODEL},
                {"role": "verifier", "model": VERIFIER_Q3_MODEL},
            ],
            "stats_summary": {"eval_count": 45},
            "query_wall_seconds": 0.08,
            "table_audit": {
                "enabled": True,
                "is_table_query": False,
                "gate_decision": "bypassed",
            },
        }

        # 1. Direct runner validation of completed attempt with factual partial
        is_valid, err = _validate_attempt_result(partial_attempt, scheduled)
        self.assertTrue(is_valid, f"Completed attempt with factual partial must validate cleanly, got error: {err}")
        self.assertIsNone(err)

        # 2. End-to-end export_review validation with realistic partial attempt
        with tempfile.TemporaryDirectory() as tmp_dir:
            proto_dir = Path(tmp_dir) / "protocol_workspace"
            transport = FakeTransport()
            prepare(proto_dir, transport=transport)

            class PartialHandler(FakeExecutionHandler):
                def __call__(self, attempt_spec: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
                    res = super().__call__(attempt_spec, **kwargs)
                    if attempt_spec["attempt_index"] == 1:
                        res["factual_outcome"] = copy.deepcopy(partial_attempt["factual_outcome"])
                        res["coverage_publication"] = copy.deepcopy(partial_attempt["coverage_publication"])
                        res["formatted_answer"] = partial_attempt["formatted_answer"]
                    return res

            results_path = Path(tmp_dir) / "results_partial.json"
            handler = PartialHandler()
            results = run(proto_dir / "protocol.json", results_path, transport=transport, execute_fn=handler)

            self.assertEqual(results["run_status"], "completed")
            self.assertEqual(results["summary"]["completed_attempts"], 31)
            self.assertEqual(results["attempts"][0]["status"], "completed")
            self.assertEqual(results["attempts"][0]["factual_outcome"]["status"], "partial")

            review_bundle_path = Path(tmp_dir) / "review_bundle.json"
            bundle = export_review(proto_dir / "protocol.json", results_path, review_bundle_path)

            self.assertEqual(bundle["schema_version"], "table-grounding-review-v1")
            self.assertEqual(len(bundle["cases"]), 21)
            first_case_arm = bundle["cases"][0]["arms"]["single"]
            self.assertEqual(first_case_arm["formatted_answer"], partial_attempt["formatted_answer"])
            self.assertEqual(first_case_arm["coverage_publication"]["status"], "partial")
            self.assertIsNone(first_case_arm["technical_error"])

        # 3. Rejection of forbidden / malformed factual statuses
        # Forbidden: 'supported'
        bad_supp = copy.deepcopy(partial_attempt)
        bad_supp["factual_outcome"]["status"] = "supported"
        is_v_supp, err_supp = _validate_attempt_result(bad_supp, scheduled)
        self.assertFalse(is_v_supp)
        self.assertIn("factual_outcome status must be 'answered', 'partial', or 'insufficient_evidence'", str(err_supp))

        # Forbidden: 'unknown'
        bad_unk = copy.deepcopy(partial_attempt)
        bad_unk["factual_outcome"]["status"] = "unknown"
        is_v_unk, err_unk = _validate_attempt_result(bad_unk, scheduled)
        self.assertFalse(is_v_unk)
        self.assertIn("factual_outcome status must be 'answered', 'partial', or 'insufficient_evidence'", str(err_unk))

        # Malformed: factual_outcome not a dict
        bad_nondict = copy.deepcopy(partial_attempt)
        bad_nondict["factual_outcome"] = "partial"
        is_v_nd, err_nd = _validate_attempt_result(bad_nondict, scheduled)
        self.assertFalse(is_v_nd)
        self.assertIn("factual_outcome must be a dict", str(err_nd))

        # Malformed: claims contain non-dict elements
        bad_claims = copy.deepcopy(partial_attempt)
        bad_claims["factual_outcome"]["claims"] = ["texto_plano_no_dict"]
        is_v_cl, err_cl = _validate_attempt_result(bad_claims, scheduled)
        self.assertFalse(is_v_cl)
        self.assertIn("Claims in factual_outcome must be a list of dicts", str(err_cl))

        # Malformed: facts contain non-dict elements
        bad_facts = copy.deepcopy(partial_attempt)
        bad_facts["factual_outcome"]["facts"] = ["texto_plano_no_dict"]
        is_v_fa, err_fa = _validate_attempt_result(bad_facts, scheduled)
        self.assertFalse(is_v_fa)
        self.assertIn("Facts in factual_outcome must be a list of dicts", str(err_fa))

        # Outer status cannot be 'partial' - legacy raw status must strictly be 'completed' or 'error'
        bad_outer = copy.deepcopy(partial_attempt)
        bad_outer["status"] = "partial"
        is_v_out, err_out = _validate_attempt_result(bad_outer, scheduled)
        self.assertFalse(is_v_out)
        self.assertIn("legacy raw status must be 'completed' or 'error'", str(err_out))


if __name__ == "__main__":
    unittest.main()
