"""Unit tests for Phase 2 & Phase 3 layout answer effect preparation, execution, and review."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import tempfile
from typing import Any, Callable
import unittest
from unittest.mock import patch

from experiments.compare_layout_answers import (
    DEFAULT_BASELINE_INDEX_PATH,
    DEFAULT_CATALOG_PATH,
    DEFAULT_LAYOUT_BUNDLE_PATH,
    DEFAULT_LEGACY_PAGES_PATH,
    DEFAULT_MANIFEST_PATH,
    DEFAULT_PRIMARY_QUERIES_PATH,
    DEFAULT_REFERENCE_FREEZE_PATH,
    DEFAULT_SECONDARY_QUERIES_PATH,
    PINNED_9B_DIGEST,
    PINNED_27B_DIGEST,
    PINNED_EMBEDDING_DIGEST,
    SCHEMA_VERSION,
    SERVER_VERSION,
    build_index,
    build_schedule,
    compute_source_closure,
    export_review,
    extract_t95_units,
    filter_and_validate_pages,
    prepare,
    run,
    validate_protocol,
    verify_baseline_chunks_against_existing,
    verify_environment,
)
from experiments.preserve_evidence import collapse_spaces, validate_catalog
from experiments.reanchor_layout_catalog import reanchor_span
from rag_pensiones.grounding import DEFAULT_PARAMETERS
from rag_pensiones.ingest import PROJECT_ROOT
from rag_pensiones.qwen_embeddings import DIMENSION, MODEL as QWEN_MODEL, vector


class FakeEmbedder:
    """Mock embedder providing deterministic normalized 2560-D vectors without network."""

    def __init__(
        self,
        digest: str = PINNED_EMBEDDING_DIGEST,
        *,
        digest_after: str | None = None,
        dimension: int = DIMENSION,
        corrupt_vector: bool = False,
        count_mismatch: bool = False,
    ) -> None:
        self.digest = digest
        self.digest_after = digest_after if digest_after is not None else digest
        self.dimension = dimension
        self.corrupt_vector = corrupt_vector
        self.count_mismatch = count_mismatch
        self.call_count = 0
        self._get_digest_count = 0

    def get_digest(self) -> str:
        self._get_digest_count += 1
        return self.digest if self._get_digest_count <= 1 else self.digest_after

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.call_count += 1
        if self.count_mismatch:
            return [[1.0] + [0.0] * (self.dimension - 1)]

        vectors: list[list[float]] = []
        for idx, t in enumerate(texts):
            if self.corrupt_vector:
                vectors.append([float("nan")] * self.dimension)
                continue
            h = int(hashlib.sha256(t.encode("utf-8")).hexdigest()[:8], 16)
            pos = h % self.dimension
            vec = [0.0] * self.dimension
            vec[pos] = 1.0
            vectors.append(vec)
        return vectors


class FakeTransport:
    """Mock transport for Ollama server and model tags verification."""

    def __init__(
        self,
        server_version: str = SERVER_VERSION,
        generation_digest: str = PINNED_9B_DIGEST,
        verifier_digest: str = PINNED_27B_DIGEST,
        embedding_digest: str = PINNED_EMBEDDING_DIGEST,
        omit_model: str | None = None,
    ) -> None:
        self.server_version = server_version
        self.generation_digest = generation_digest
        self.verifier_digest = verifier_digest
        self.embedding_digest = embedding_digest
        self.omit_model = omit_model

    def get_version(self) -> dict[str, Any]:
        return {"version": self.server_version}

    def get_tags(self) -> dict[str, Any]:
        models = [
            {"name": "qwen3.5:9b", "digest": self.generation_digest},
            {"name": "hf.co/gguf/Qwen3.8-27B-GGUF:Q3_K_M", "digest": self.verifier_digest},
            {"name": "qwen3-embedding:4b", "digest": self.embedding_digest},
        ]
        if self.omit_model:
            models = [m for m in models if m["name"] != self.omit_model]
        return {"models": models}


class FakeExecutionHandler:
    """Mock execution handler simulating execute_attempt without network."""

    def __init__(
        self,
        fail_at_indices: set[int] | None = None,
        on_attempt_cb: Callable[[int], None] | None = None,
    ) -> None:
        self.calls: list[dict[str, Any]] = []
        self.fail_at_indices = fail_at_indices or set()
        self.on_attempt_cb = on_attempt_cb

    def __call__(
        self,
        attempt_spec: dict[str, Any],
        *,
        index_data: dict[str, Any],
        units_data: dict[str, Any],
        pages_data: list[dict[str, Any]],
        transport: Any,
    ) -> dict[str, Any]:
        idx = attempt_spec["attempt_index"]
        self.calls.append({
            "attempt_spec": copy.deepcopy(attempt_spec),
            "index_data": copy.deepcopy(index_data),
            "units_data": copy.deepcopy(units_data),
            "pages_data": copy.deepcopy(pages_data),
        })

        if self.on_attempt_cb:
            self.on_attempt_cb(idx)

        if idx in self.fail_at_indices:
            raise RuntimeError(f"Simulated technical failure at attempt {idx}")

        return {
            "attempt_index": idx,
            "case_id": attempt_spec["case_id"],
            "query": attempt_spec["query"],
            "arm": attempt_spec["arm"],
            "status": "completed",
            "error": None,
            "query_wall_seconds": 0.05,
            "formatted_answer": f"Simulated answer for {attempt_spec['case_id']}",
            "coverage_publication": {
                "status": "answered",
                "claims": [{"claim_id": "c1", "text": f"Claim text for {attempt_spec['case_id']}"}],
            },
            "factual_outcome": {"status": "answered"},
            "lifecycle_records": [],
            "stats_summary": {},
        }


class TestLayoutAnswerEffect(unittest.TestCase):
    """Test suite for Phase 2 & 3 layout answer effect preparation, execution, and review."""

    # 1. realreanchorUnicodecounterexamples
    def test_real_reanchor_unicode_counterexamples(self):
        with self.assertRaises(ValueError) as ctx:
            reanchor_span("a", "á")
        self.assertIn("not found", str(ctx.exception).lower())

        start, end = reanchor_span("a", "áa")
        self.assertEqual((start, end), (1, 2))
        self.assertEqual("áa"[start:end], "a")

        with self.assertRaises(ValueError):
            reanchor_span("\u0301", "á")

        with self.assertRaises(ValueError) as ctx:
            reanchor_span("e", "e\u0301")
        self.assertIn("not found", str(ctx.exception).lower())

        start, end = reanchor_span("é", "e\u0301")
        self.assertEqual((start, end), (0, 2))
        self.assertEqual("e\u0301"[start:end], "e\u0301")

        with self.assertRaises(ValueError) as ctx:
            reanchor_span("AA", "AAAA")
        self.assertIn("ambiguous", str(ctx.exception).lower())

    # 2. source-target identity/range/content mismatches
    def test_source_target_identity_range_content_mismatches(self):
        legacy_data = json.loads((PROJECT_ROOT / DEFAULT_LAYOUT_BUNDLE_PATH).read_text(encoding="utf-8"))
        pages = legacy_data["pages"]

        legacy_pages_mock = [
            {
                "doc_id": legacy_data["metadata"]["doc_id"],
                "source_file": legacy_data["metadata"]["source_file"],
                "published_at": legacy_data["metadata"]["published_at"],
                "page_number": p["page_number"],
                "text": p["legacy_text"],
            }
            for p in pages
            if 95 <= p["page_number"] <= 99
        ]

        corrupt_structured = copy.deepcopy(legacy_data)
        corrupt_structured["pages"][0]["legacy_text"] = "Corrupted legacy text mismatch"
        with self.assertRaises(ValueError) as ctx:
            filter_and_validate_pages(legacy_pages_mock, corrupt_structured)
        self.assertIn("does not match old page text exactly", str(ctx.exception))

        corrupt_doc_id = copy.deepcopy(legacy_pages_mock)
        corrupt_doc_id[0]["doc_id"] = "wrong_doc_id"
        with self.assertRaises(ValueError) as ctx:
            filter_and_validate_pages(corrupt_doc_id, legacy_data)
        self.assertIn("metadata mismatch", str(ctx.exception))

    # 3. exactbaselinechunkrecipe/descriptor mismatch
    def test_exact_baseline_chunk_recipe_descriptor_mismatch(self):
        index_data = json.loads((PROJECT_ROOT / DEFAULT_BASELINE_INDEX_PATH).read_text(encoding="utf-8"))
        chunks_95_99 = [
            c for c in index_data["chunks"]
            if 95 <= c.get("page_number", 0) <= 99
        ]
        self.assertTrue(len(chunks_95_99) > 0)

        verify_baseline_chunks_against_existing(chunks_95_99, PROJECT_ROOT / DEFAULT_BASELINE_INDEX_PATH)

        corrupted_chunks = copy.deepcopy(chunks_95_99)
        corrupted_chunks[0]["strategy"] = "corrupted-strategy-v2"
        with self.assertRaises(ValueError) as ctx:
            verify_baseline_chunks_against_existing(corrupted_chunks, PROJECT_ROOT / DEFAULT_BASELINE_INDEX_PATH)
        self.assertIn("mismatch at chunk index 0", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx:
            verify_baseline_chunks_against_existing(chunks_95_99[:-1], PROJECT_ROOT / DEFAULT_BASELINE_INDEX_PATH)
        self.assertIn("count mismatch", str(ctx.exception))

    # 4. 5page filtering excludes100/101
    def test_5page_filtering_excludes_100_101(self):
        structured_data = json.loads((PROJECT_ROOT / DEFAULT_LAYOUT_BUNDLE_PATH).read_text(encoding="utf-8"))
        all_pilot_pnums = [p["page_number"] for p in structured_data["pages"]]
        self.assertIn(100, all_pilot_pnums)
        self.assertIn(101, all_pilot_pnums)

        legacy_pages_all = [
            {
                "doc_id": structured_data["metadata"]["doc_id"],
                "source_file": structured_data["metadata"]["source_file"],
                "published_at": structured_data["metadata"]["published_at"],
                "page_number": p["page_number"],
                "text": p["legacy_text"],
            }
            for p in structured_data["pages"]
        ]

        b_pages, l_pages = filter_and_validate_pages(legacy_pages_all, structured_data)
        self.assertEqual(len(b_pages), 5)
        self.assertEqual(len(l_pages), 5)
        self.assertEqual([p["page_number"] for p in b_pages], [95, 96, 97, 98, 99])
        self.assertEqual([p["page_number"] for p in l_pages], [95, 96, 97, 98, 99])
        self.assertNotIn(100, [p["page_number"] for p in b_pages])
        self.assertNotIn(101, [p["page_number"] for p in b_pages])

    # 5. primaryunits unchanged/noT95, bothsecondaryunits literal
    def test_primary_units_unchanged_no_t95_and_both_secondary_units_literal(self):
        structured_data = json.loads((PROJECT_ROOT / DEFAULT_LAYOUT_BUNDLE_PATH).read_text(encoding="utf-8"))
        catalog_data = json.loads((PROJECT_ROOT / DEFAULT_CATALOG_PATH).read_text(encoding="utf-8"))

        legacy_pages = [
            {
                "doc_id": structured_data["metadata"]["doc_id"],
                "source_file": structured_data["metadata"]["source_file"],
                "published_at": structured_data["metadata"]["published_at"],
                "page_number": p["page_number"],
                "text": p["legacy_text"],
            }
            for p in structured_data["pages"]
            if 95 <= p["page_number"] <= 99
        ]
        b_pages, l_pages = filter_and_validate_pages(legacy_pages, structured_data)

        self.assertEqual(len(catalog_data["units"]), 13)
        self.assertFalse(any(u["id"] == "T95" for u in catalog_data["units"]))

        p95_struct = next(p for p in structured_data["pages"] if p["page_number"] == 95)
        t95_base, t95_lay = extract_t95_units(b_pages[0], l_pages[0], p95_struct)

        self.assertEqual(t95_base["id"], "T95")
        self.assertTrue(t95_base["complete_end"])
        base_slice = b_pages[0]["text"][t95_base["spans"][0]["start_char"]:t95_base["spans"][0]["end_char"]]
        self.assertTrue(base_slice.startswith("CUADRO 9."))
        self.assertTrue(base_slice.endswith("Fuente: IGAE y AIReF"))
        self.assertEqual(collapse_spaces(base_slice), t95_base["spans"][0]["quote"])

        self.assertEqual(t95_lay["id"], "T95")
        self.assertTrue(t95_lay["complete_end"])
        lay_slice = l_pages[0]["text"][t95_lay["spans"][0]["start_char"]:t95_lay["spans"][0]["end_char"]]
        self.assertTrue("CUADRO 9" in lay_slice)
        self.assertEqual(collapse_spaces(lay_slice), t95_lay["spans"][0]["quote"])

        sec_base_catalog = copy.deepcopy(catalog_data)
        sec_base_catalog["units"].append(t95_base)
        validate_catalog(sec_base_catalog, b_pages)

    # 6. inputmutation/nooverwrite
    def test_input_mutation_and_no_overwrite(self):
        catalog_orig = json.loads((PROJECT_ROOT / DEFAULT_CATALOG_PATH).read_text(encoding="utf-8"))
        queries_orig = json.loads((PROJECT_ROOT / DEFAULT_PRIMARY_QUERIES_PATH).read_text(encoding="utf-8"))
        sec_queries_orig = json.loads((PROJECT_ROOT / DEFAULT_SECONDARY_QUERIES_PATH).read_text(encoding="utf-8"))

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_p = Path(tmp_dir) / "test_prep_output"
            embedder = FakeEmbedder()
            transport = FakeTransport()

            protocol = prepare(out_p, embedder=embedder, transport=transport)
            self.assertEqual(protocol["schema_version"], SCHEMA_VERSION)
            self.assertTrue((out_p / "protocol.json").exists())

            catalog_after = json.loads((PROJECT_ROOT / DEFAULT_CATALOG_PATH).read_text(encoding="utf-8"))
            queries_after = json.loads((PROJECT_ROOT / DEFAULT_PRIMARY_QUERIES_PATH).read_text(encoding="utf-8"))
            sec_queries_after = json.loads((PROJECT_ROOT / DEFAULT_SECONDARY_QUERIES_PATH).read_text(encoding="utf-8"))
            self.assertEqual(catalog_orig, catalog_after)
            self.assertEqual(queries_orig, queries_after)
            self.assertEqual(sec_queries_orig, sec_queries_after)

            with self.assertRaises(FileExistsError):
                prepare(out_p, embedder=embedder, transport=transport)

    # 7. wrongdigest/count/nonfinite vectors
    def test_wrong_digest_count_nonfinite_vectors(self):
        structured_data = json.loads((PROJECT_ROOT / DEFAULT_LAYOUT_BUNDLE_PATH).read_text(encoding="utf-8"))
        legacy_pages = [
            {
                "doc_id": structured_data["metadata"]["doc_id"],
                "source_file": structured_data["metadata"]["source_file"],
                "published_at": structured_data["metadata"]["published_at"],
                "page_number": p["page_number"],
                "text": p["legacy_text"],
            }
            for p in structured_data["pages"]
            if 95 <= p["page_number"] <= 99
        ]
        b_pages, _ = filter_and_validate_pages(legacy_pages, structured_data)

        bad_digest_embedder = FakeEmbedder(digest="valid_hex_64" * 4, digest_after="other_hex_64" * 4)
        with self.assertRaises(RuntimeError) as ctx:
            build_index(b_pages, bad_digest_embedder)
        self.assertIn("digest mismatch", str(ctx.exception).lower())

        count_mismatch_embedder = FakeEmbedder(count_mismatch=True)
        with self.assertRaises(ValueError) as ctx:
            build_index(b_pages, count_mismatch_embedder)
        self.assertIn("count mismatch", str(ctx.exception).lower())

        corrupt_embedder = FakeEmbedder(corrupt_vector=True)
        with self.assertRaises(ValueError) as ctx:
            build_index(b_pages, corrupt_embedder)
        self.assertIn("nonfinite", str(ctx.exception).lower())

        wrong_dim_embedder = FakeEmbedder(dimension=128)
        with self.assertRaises(ValueError) as ctx:
            build_index(b_pages, wrong_dim_embedder)
        self.assertIn("expected2560", str(ctx.exception).lower())

    # 8. 17questions34counterbalancedconditions no duplicates
    def test_17_questions_34_counterbalanced_conditions_no_duplicates(self):
        primary_data = json.loads((PROJECT_ROOT / DEFAULT_PRIMARY_QUERIES_PATH).read_text(encoding="utf-8"))
        secondary_data = json.loads((PROJECT_ROOT / DEFAULT_SECONDARY_QUERIES_PATH).read_text(encoding="utf-8"))

        schedule = build_schedule(primary_data, secondary_data)
        self.assertEqual(len(schedule), 34)

        attempt_indices = [item["attempt_index"] for item in schedule]
        self.assertEqual(attempt_indices, list(range(1, 35)))
        self.assertTrue(all(item["arm"] == "mixed" for item in schedule))

        for q_idx in range(17):
            q_attempts = [item for item in schedule if item["question_index"] == q_idx]
            self.assertEqual(len(q_attempts), 2)
            if q_idx % 2 == 0:
                self.assertEqual([a["condition"] for a in q_attempts], ["legacy", "layout"])
            else:
                self.assertEqual([a["condition"] for a in q_attempts], ["layout", "legacy"])

        seen_keys = set()
        for item in schedule:
            key = (item["cohort"], item["case_id"], item["condition"])
            self.assertNotIn(key, seen_keys)
            seen_keys.add(key)
        self.assertEqual(len(seen_keys), 34)

        primary_attempts = [item for item in schedule if item["cohort"] == "primary"]
        table_attempts = [item for item in schedule if item["cohort"] == "table"]
        self.assertEqual(len(primary_attempts), 22)
        self.assertEqual(len(table_attempts), 12)

    # 9. fakebadserver/model/options rejected before embeddings
    def test_fake_bad_server_model_options_rejected_before_embeddings(self):
        bad_server_transport = FakeTransport(server_version="0.40.10")
        with self.assertRaises(RuntimeError) as ctx:
            verify_environment(bad_server_transport)
        self.assertIn("server version mismatch", str(ctx.exception).lower())

        missing_gen_transport = FakeTransport(omit_model="qwen3.5:9b")
        with self.assertRaises(RuntimeError) as ctx:
            verify_environment(missing_gen_transport)
        self.assertIn("required model 'qwen3.5:9b' not found", str(ctx.exception).lower())

        bad_ver_transport = FakeTransport(verifier_digest="wrong_digest_hex_00000000000000000000000000000000000000000000000000")
        with self.assertRaises(RuntimeError) as ctx:
            verify_environment(bad_ver_transport)
        self.assertIn("digest mismatch", str(ctx.exception).lower())

        embedder = FakeEmbedder()
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_p = Path(tmp_dir) / "test_abort_before_embed"
            with self.assertRaises(RuntimeError):
                prepare(out_p, embedder=embedder, transport=bad_server_transport)
            self.assertEqual(embedder.call_count, 0)

    # 10. frozeninput/sourcemanifest correct
    def test_frozen_input_and_source_manifest_correct(self):
        ref_freeze = json.loads((PROJECT_ROOT / DEFAULT_REFERENCE_FREEZE_PATH).read_text(encoding="utf-8"))
        for rel_path, expected_sha in ref_freeze.get("files", {}).items():
            full_path = PROJECT_ROOT / rel_path
            self.assertTrue(full_path.exists(), f"File {rel_path} in reference_freeze not found")
            actual_sha = hashlib.sha256(full_path.read_bytes()).hexdigest()
            self.assertEqual(
                actual_sha,
                expected_sha,
                f"Hash mismatch for {rel_path}: expected {expected_sha}, got {actual_sha}",
            )

        manifest = json.loads((PROJECT_ROOT / DEFAULT_MANIFEST_PATH).read_text(encoding="utf-8"))
        structured_file = PROJECT_ROOT / DEFAULT_LAYOUT_BUNDLE_PATH
        actual_struct_sha = hashlib.sha256(structured_file.read_bytes()).hexdigest()
        self.assertEqual(manifest["files"]["structured.json"]["sha256"], actual_struct_sha)

        closure = compute_source_closure()
        self.assertTrue(len(closure) >= 20)
        self.assertIn("experiments/compare_layout_answers.py", closure)
        self.assertIn("experiments/reanchor_layout_catalog.py", closure)
        self.assertIn("tests/test_layout_catalog.py", closure)
        self.assertIn("tests/test_layout_answer_effect.py", closure)

    # 11. exact34counterbalancedconditions and both catalogs routed right, actual mixed arm
    def test_run_exact_34_counterbalanced_and_catalogs_routed_right(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            proto_dir = Path(tmp_dir) / "proto_workspace"
            embedder = FakeEmbedder()
            transport = FakeTransport()

            protocol = prepare(proto_dir, embedder=embedder, transport=transport)
            handler = FakeExecutionHandler()

            results_path = Path(tmp_dir) / "results.json"
            results = run(proto_dir / "protocol.json", results_path, transport=transport, execute_fn=handler)

            self.assertEqual(len(handler.calls), 34)
            self.assertEqual(results["run_status"], "completed")
            self.assertEqual(results["summary"]["completed_attempts"], 34)
            self.assertEqual(results["summary"]["error_attempts"], 0)

            for call_rec in handler.calls:
                spec = call_rec["attempt_spec"]
                self.assertEqual(spec["arm"], "mixed")

                idx = spec["attempt_index"]
                sched_item = protocol["schedule"][idx - 1]
                self.assertEqual(sched_item["cohort"], call_rec["attempt_spec"]["case_id"][0:1] == "t" and "table" or "primary")

                if sched_item["cohort"] == "primary":
                    self.assertEqual(len(call_rec["units_data"]["units"]), 13)
                    self.assertFalse(any(u["id"] == "T95" for u in call_rec["units_data"]["units"]))
                    if sched_item["condition"] == "legacy":
                        self.assertIn("Opinión sobre la sostenibilidad", call_rec["pages_data"][0]["text"])
                    else:
                        self.assertIn("GRÁFICO 44. EMPLEOS NO FINANCIEROS", call_rec["pages_data"][0]["text"])
                else:
                    self.assertEqual(len(call_rec["units_data"]["units"]), 14)
                    self.assertTrue(any(u["id"] == "T95" for u in call_rec["units_data"]["units"]))
                    t95_unit = next(u for u in call_rec["units_data"]["units"] if u["id"] == "T95")
                    if sched_item["condition"] == "legacy":
                        self.assertTrue(t95_unit["spans"][0]["quote"].startswith("CUADRO 9."))
                    else:
                        self.assertIn("CUADRO 9", t95_unit["spans"][0]["quote"])

    # 12. errors retained and others run
    def test_run_errors_retained_in_denominator_and_others_continue(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            proto_dir = Path(tmp_dir) / "proto_workspace"
            embedder = FakeEmbedder()
            transport = FakeTransport()
            prepare(proto_dir, embedder=embedder, transport=transport)

            handler = FakeExecutionHandler(fail_at_indices={4, 10})
            results_path = Path(tmp_dir) / "results.json"
            results = run(proto_dir / "protocol.json", results_path, transport=transport, execute_fn=handler)

            self.assertEqual(len(results["attempts"]), 34)
            self.assertEqual(results["summary"]["completed_attempts"], 32)
            self.assertEqual(results["summary"]["error_attempts"], 2)
            self.assertEqual(results["run_status"], "completed")

            att4 = results["attempts"][3]
            self.assertEqual(att4["status"], "error")
            self.assertIn("Simulated technical failure at attempt 4", att4["error"])

            att10 = results["attempts"][9]
            self.assertEqual(att10["status"], "error")
            self.assertIn("Simulated technical failure at attempt 10", att10["error"])

            self.assertEqual(results["attempts"][0]["status"], "completed")
            self.assertEqual(results["attempts"][4]["status"], "completed")

    # 13. missing/duplicate/outputcollision rejects
    def test_run_output_collision_and_missing_rejects(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            proto_dir = Path(tmp_dir) / "proto_workspace"
            embedder = FakeEmbedder()
            transport = FakeTransport()
            prepare(proto_dir, embedder=embedder, transport=transport)

            results_path = Path(tmp_dir) / "results.json"

            with self.assertRaises(FileNotFoundError):
                run(Path(tmp_dir) / "missing_protocol.json", results_path, transport=transport)

            results_path.write_text("pre-existing content", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                run(proto_dir / "protocol.json", results_path, transport=transport)

    # 14. badserver/model/options/typefalse rejects pre-network/attempt
    def test_validate_protocol_bad_server_model_options_typefalse(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            proto_dir = Path(tmp_dir) / "proto_workspace"
            embedder = FakeEmbedder()
            transport = FakeTransport()
            protocol = prepare(proto_dir, embedder=embedder, transport=transport)

            validate_protocol(protocol, protocol_dir=proto_dir)

            bad_server_p = copy.deepcopy(protocol)
            bad_server_p["server_version"] = "0.40.10"
            with self.assertRaises(ValueError) as ctx:
                validate_protocol(bad_server_p, protocol_dir=proto_dir)
            self.assertIn("server version mismatch", str(ctx.exception).lower())

            bad_temp_p = copy.deepcopy(protocol)
            bad_temp_p["parameters"]["generation_parameters"]["temperature"] = False
            with self.assertRaises(ValueError) as ctx:
                validate_protocol(bad_temp_p, protocol_dir=proto_dir)
            self.assertIn("type mismatch", str(ctx.exception).lower())

            bad_extra_p = copy.deepcopy(protocol)
            bad_extra_p["parameters"]["generation_parameters"]["extra_param"] = 123
            with self.assertRaises(ValueError) as ctx:
                validate_protocol(bad_extra_p, protocol_dir=proto_dir)
            self.assertIn("keys mismatch", str(ctx.exception).lower())

            bad_topk_p = copy.deepcopy(protocol)
            bad_topk_p["execution_options"]["top_k"] = True
            with self.assertRaises(ValueError) as ctx:
                validate_protocol(bad_topk_p, protocol_dir=proto_dir)
            self.assertIn("type mismatch", str(ctx.exception).lower())

            bad_ver_p = copy.deepcopy(protocol)
            bad_ver_p["models"]["verifier_digest"] = "wrong_digest_000000000000000000000000000000000000000000000000000000"
            with self.assertRaises(ValueError) as ctx:
                validate_protocol(bad_ver_p, protocol_dir=proto_dir)
            self.assertIn("mismatch", str(ctx.exception).lower())

    # 15. manifestmissing/partial/source/inputdrift beforecalls invalidates
    def test_validate_protocol_source_input_manifest_drift(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            proto_dir = Path(tmp_dir) / "proto_workspace"
            embedder = FakeEmbedder()
            transport = FakeTransport()
            protocol = prepare(proto_dir, embedder=embedder, transport=transport)

            drift_proto = copy.deepcopy(protocol)
            drift_proto["source_freeze"]["experiments/compare_layout_answers.py"] = "0" * 64
            with self.assertRaises(ValueError) as ctx:
                validate_protocol(drift_proto, protocol_dir=proto_dir)
            self.assertIn("source code drift", str(ctx.exception).lower())

            drift_man = copy.deepcopy(protocol)
            drift_man["source_artifact_manifest_sha256"] = "0" * 64
            with self.assertRaises(ValueError) as ctx:
                validate_protocol(drift_man, protocol_dir=proto_dir)
            self.assertIn("manifest sha mismatch", str(ctx.exception).lower())

            drift_inp = copy.deepcopy(protocol)
            first_key = next(iter(drift_inp["input_file_hashes"]))
            drift_inp["input_file_hashes"][first_key] = "0" * 64
            with self.assertRaises(ValueError) as ctx:
                validate_protocol(drift_inp, protocol_dir=proto_dir)
            self.assertIn("input file drift", str(ctx.exception).lower())

    # 16. postattempt drift marksinvalid and stops
    def test_run_postattempt_drift_marks_invalid_and_stops(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            proto_dir = Path(tmp_dir) / "proto_workspace"
            embedder = FakeEmbedder()
            transport = FakeTransport()
            prepare(proto_dir, embedder=embedder, transport=transport)

            def corrupt_on_attempt(idx: int) -> None:
                if idx == 3:
                    pcat_file = proto_dir / "baseline_primary_catalog.json"
                    pcat_file.write_text("{\"corrupted\": true}", encoding="utf-8")

            handler = FakeExecutionHandler(on_attempt_cb=corrupt_on_attempt)
            results_path = Path(tmp_dir) / "results.json"

            with self.assertRaises(RuntimeError) as ctx:
                run(proto_dir / "protocol.json", results_path, transport=transport, execute_fn=handler)
            self.assertIn("post-attempt integrity drift", str(ctx.exception).lower())

            self.assertTrue(results_path.exists())
            saved_results = json.loads(results_path.read_text(encoding="utf-8"))
            self.assertEqual(saved_results["run_status"], "invalid")
            self.assertEqual(len(saved_results["attempts"]), 3)

    # 17. checkpoints complete
    def test_atomic_checkpoints_complete(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            proto_dir = Path(tmp_dir) / "proto_workspace"
            embedder = FakeEmbedder()
            transport = FakeTransport()
            prepare(proto_dir, embedder=embedder, transport=transport)

            results_path = Path(tmp_dir) / "results.json"
            handler = FakeExecutionHandler()
            run(proto_dir / "protocol.json", results_path, transport=transport, execute_fn=handler)

            self.assertTrue(results_path.exists())
            data = json.loads(results_path.read_text(encoding="utf-8"))
            self.assertEqual(data["run_status"], "completed")
            self.assertEqual(len(data["attempts"]), 34)

    # 18. exports onlyfinalbound34 andneutralvalidmapping with no rawtraces
    def test_export_review_only_completed_and_neutral_mapping(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            proto_dir = Path(tmp_dir) / "proto_workspace"
            embedder = FakeEmbedder()
            transport = FakeTransport()
            prepare(proto_dir, embedder=embedder, transport=transport)

            results_path = Path(tmp_dir) / "results.json"
            handler = FakeExecutionHandler()
            run(proto_dir / "protocol.json", results_path, transport=transport, execute_fn=handler)

            review_out = Path(tmp_dir) / "review_bundle.json"
            bundle = export_review(proto_dir / "protocol.json", results_path, review_out)

            self.assertEqual(len(bundle["cases"]), 17)
            raw_json = json.dumps(bundle)

            self.assertNotIn('"legacy"', raw_json)
            self.assertNotIn('"layout"', raw_json)
            self.assertNotIn("messages", raw_json)
            self.assertNotIn("embedding", raw_json)

            key_file = review_out.parent / "review_key.json"
            self.assertTrue(key_file.exists())
            key_data = json.loads(key_file.read_text(encoding="utf-8"))
            self.assertEqual(len(key_data["mapping"]), 17)

            for cid, arm_map in key_data["mapping"].items():
                self.assertEqual(set(arm_map.keys()), {"A", "B"})
                self.assertEqual(set(arm_map.values()), {"legacy", "layout"})

            corrupt_results_path = Path(tmp_dir) / "corrupt_results.json"
            res_dict = json.loads(results_path.read_text(encoding="utf-8"))
            res_dict["run_status"] = "invalid"
            corrupt_results_path.write_text(json.dumps(res_dict), encoding="utf-8")
            with self.assertRaises(ValueError) as ctx:
                export_review(proto_dir / "protocol.json", corrupt_results_path, Path(tmp_dir) / "other_review.json")
            self.assertIn("must be 'completed'", str(ctx.exception))

    # --- P1 & P2 Regression Tests for the 6 Senior Findings ---

    # Finding 1: prepare CLI transport=None must ALWAYS verify environment before any embeddings
    @patch("experiments.compare_layout_answers.MixedSingle27Transport")
    def test_prepare_default_cli_none_transport_verifies_environment(self, mock_transport_cls):
        mock_transport_cls.return_value = FakeTransport(server_version="0.40.10")
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_p = Path(tmp_dir) / "test_prep_bad_env"
            embedder = FakeEmbedder()
            with self.assertRaises(RuntimeError) as ctx:
                prepare(out_p, embedder=embedder, transport=None)
            self.assertIn("server version mismatch", str(ctx.exception).lower())
            self.assertEqual(embedder.call_count, 0)
            self.assertFalse((out_p / "protocol.json").exists())

    # Finding 2a: swapping baseline and layout assets is detected and rejected
    def test_validate_protocol_arm_swap_rejected(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            proto_dir = Path(tmp_dir) / "proto_workspace"
            embedder = FakeEmbedder()
            transport = FakeTransport()
            protocol = prepare(proto_dir, embedder=embedder, transport=transport)

            # Swap pages in both_arms
            swapped_p = copy.deepcopy(protocol)
            swapped_p["both_arms"]["baseline"]["pages_file"] = "layout_pages.jsonl"
            swapped_p["both_arms"]["layout"]["pages_file"] = "baseline_pages.jsonl"
            with self.assertRaises(ValueError) as ctx:
                validate_protocol(swapped_p, protocol_dir=proto_dir)
            self.assertIn("arm swap or corruption detected", str(ctx.exception).lower())

            # Swap table catalogs
            swapped_cat = copy.deepcopy(protocol)
            swapped_cat["both_arms"]["baseline"]["table_catalog_file"] = "layout_table_catalog.json"
            swapped_cat["both_arms"]["layout"]["table_catalog_file"] = "baseline_table_catalog.json"
            with self.assertRaises(ValueError) as ctx:
                validate_protocol(swapped_cat, protocol_dir=proto_dir)
            self.assertIn("arm swap or corruption detected", str(ctx.exception).lower())

    # Finding 2b: modifying vector in index file is detected by raw/canonical asset hash
    def test_validate_protocol_vector_file_hash_edited_rejected(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            proto_dir = Path(tmp_dir) / "proto_workspace"
            embedder = FakeEmbedder()
            transport = FakeTransport()
            protocol = prepare(proto_dir, embedder=embedder, transport=transport)

            # Corrupt vector file
            idx_file = proto_dir / "baseline_index.json"
            data = json.loads(idx_file.read_text(encoding="utf-8"))
            data["chunks"][0]["embedding"][0] += 0.001
            idx_file.write_text(json.dumps(data, indent=2), encoding="utf-8")

            with self.assertRaises(ValueError) as ctx:
                validate_protocol(protocol, protocol_dir=proto_dir)
            self.assertIn("raw_sha256 mismatch for baseline_index.json", str(ctx.exception).lower())

    # Finding 3: input_file_hashes must be exact complete keyset; missing/extra rejected
    def test_validate_protocol_input_hashes_subset_rejected(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            proto_dir = Path(tmp_dir) / "proto_workspace"
            embedder = FakeEmbedder()
            transport = FakeTransport()
            protocol = prepare(proto_dir, embedder=embedder, transport=transport)

            # Remove catalog path
            proto_no_cat = copy.deepcopy(protocol)
            del proto_no_cat["input_file_hashes"]["evaluation/reconstructed_units_catalog.json"]
            with self.assertRaises(ValueError) as ctx:
                validate_protocol(proto_no_cat, protocol_dir=proto_dir)
            self.assertIn("keyset mismatch: missing", str(ctx.exception).lower())
            self.assertIn("reconstructed_units_catalog.json", str(ctx.exception))

            # Remove legacy pages path
            proto_no_pgs = copy.deepcopy(protocol)
            del proto_no_pgs["input_file_hashes"]["data/processed/airef_2025_pages.jsonl"]
            with self.assertRaises(ValueError) as ctx:
                validate_protocol(proto_no_pgs, protocol_dir=proto_dir)
            self.assertIn("keyset mismatch: missing", str(ctx.exception).lower())
            self.assertIn("airef_2025_pages.jsonl", str(ctx.exception))

            # Remove query path
            proto_no_qry = copy.deepcopy(protocol)
            del proto_no_qry["input_file_hashes"]["evaluation/scope_coverage_2026-10-26/full_inputs.json"]
            with self.assertRaises(ValueError) as ctx:
                validate_protocol(proto_no_qry, protocol_dir=proto_dir)
            self.assertIn("keyset mismatch: missing", str(ctx.exception).lower())
            self.assertIn("full_inputs.json", str(ctx.exception))

            # Add extra unknown path
            proto_extra = copy.deepcopy(protocol)
            proto_extra["input_file_hashes"]["unknown/extra_input.json"] = "0" * 64
            with self.assertRaises(ValueError) as ctx:
                validate_protocol(proto_extra, protocol_dir=proto_dir)
            self.assertIn("keyset mismatch", str(ctx.exception).lower())
            self.assertIn("extra", str(ctx.exception).lower())

    # Finding 4: modifying protocol file on disk during execution causes immediate invalidation
    def test_run_protocol_disk_mutation_aborts_immediately(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            proto_dir = Path(tmp_dir) / "proto_workspace"
            embedder = FakeEmbedder()
            transport = FakeTransport()
            prepare(proto_dir, embedder=embedder, transport=transport)

            proto_file = proto_dir / "protocol.json"

            def modify_proto_on_disk(idx: int) -> None:
                if idx == 1:
                    raw = json.loads(proto_file.read_text(encoding="utf-8"))
                    raw["tampered"] = True
                    proto_file.write_text(json.dumps(raw), encoding="utf-8")

            handler = FakeExecutionHandler(on_attempt_cb=modify_proto_on_disk)
            results_path = Path(tmp_dir) / "results.json"

            with self.assertRaises(RuntimeError) as ctx:
                run(proto_file, results_path, transport=transport, execute_fn=handler)
            self.assertIn("protocol file on disk was modified", str(ctx.exception).lower())

            self.assertTrue(results_path.exists())
            saved = json.loads(results_path.read_text(encoding="utf-8"))
            self.assertEqual(saved["run_status"], "invalid")
            self.assertEqual(len(saved["attempts"]), 1)
            self.assertNotEqual(len(saved["attempts"]), 34)

    # Finding 5: export_review compares rows to schedule and input bindings
    def test_export_review_rejects_schedule_and_binding_tampering(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            proto_dir = Path(tmp_dir) / "proto_workspace"
            embedder = FakeEmbedder()
            transport = FakeTransport()
            prepare(proto_dir, embedder=embedder, transport=transport)

            results_path = Path(tmp_dir) / "results.json"
            handler = FakeExecutionHandler()
            run(proto_dir / "protocol.json", results_path, transport=transport, execute_fn=handler)

            valid_res = json.loads(results_path.read_text(encoding="utf-8"))
            review_out = Path(tmp_dir) / "review.json"

            # 1. Duplicate arm row
            dup_res = copy.deepcopy(valid_res)
            dup_res["attempts"][1] = copy.deepcopy(dup_res["attempts"][0])
            dup_res_path = Path(tmp_dir) / "dup_results.json"
            dup_res_path.write_text(json.dumps(dup_res), encoding="utf-8")
            with self.assertRaises(ValueError) as ctx:
                export_review(proto_dir / "protocol.json", dup_res_path, review_out)
            self.assertTrue(any(term in str(ctx.exception).lower() for term in ("duplicate", "mismatch")))

            # 2. Altered formatted answer
            alt_res = copy.deepcopy(valid_res)
            alt_res["attempts"][0]["formatted_answer"] = "tampered formatted answer"
            alt_res_path = Path(tmp_dir) / "alt_results.json"
            alt_res_path.write_text(json.dumps(alt_res), encoding="utf-8")
            with self.assertRaises(ValueError) as ctx:
                export_review(proto_dir / "protocol.json", alt_res_path, review_out)
            self.assertIn("formatted_answer altered from raw_attempt", str(ctx.exception))

            # 3. Changed row query
            qry_res = copy.deepcopy(valid_res)
            qry_res["attempts"][0]["query"] = "tampered query"
            qry_res_path = Path(tmp_dir) / "qry_results.json"
            qry_res_path.write_text(json.dumps(qry_res), encoding="utf-8")
            with self.assertRaises(ValueError) as ctx:
                export_review(proto_dir / "protocol.json", qry_res_path, review_out)
            self.assertIn("query", str(ctx.exception).lower())

            # 4. Changed row input hash
            inp_res = copy.deepcopy(valid_res)
            inp_res["attempts"][0]["input_hashes"]["pages_sha256"] = "0" * 64
            inp_res_path = Path(tmp_dir) / "inp_results.json"
            inp_res_path.write_text(json.dumps(inp_res), encoding="utf-8")
            with self.assertRaises(ValueError) as ctx:
                export_review(proto_dir / "protocol.json", inp_res_path, review_out)
            self.assertIn("input pages_sha256 mismatch", str(ctx.exception))

    # Finding 6: review_key.json must not be overwritten; reserve both files before writing either
    def test_export_review_refuses_existing_review_key(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            proto_dir = Path(tmp_dir) / "proto_workspace"
            embedder = FakeEmbedder()
            transport = FakeTransport()
            prepare(proto_dir, embedder=embedder, transport=transport)

            results_path = Path(tmp_dir) / "results.json"
            handler = FakeExecutionHandler()
            run(proto_dir / "protocol.json", results_path, transport=transport, execute_fn=handler)

            review_dir = Path(tmp_dir) / "review_folder"
            review_dir.mkdir(parents=True, exist_ok=True)
            key_path = review_dir / "review_key.json"
            key_content = b"{\"existing_key\": \"keep_untouched\"}\n"
            key_path.write_bytes(key_content)

            new_bundle_path = review_dir / "new_bundle.json"
            with self.assertRaises(FileExistsError) as ctx:
                export_review(proto_dir / "protocol.json", results_path, new_bundle_path)
            self.assertIn("review key destination already exists", str(ctx.exception).lower())

            # Key was preserved byte-identical and new bundle was NOT created
            self.assertEqual(key_path.read_bytes(), key_content)
            self.assertFalse(new_bundle_path.exists())

    # --- v2 Senior Fixes Regression Tests ---

    # Finding 1 (v2): primary catalog layout alias, content T95 in primary, changed unit same count, secondary missing T95
    def test_validate_protocol_catalog_t95_aliasing_or_content_rejected(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            proto_dir = Path(tmp_dir) / "proto_workspace"
            embedder = FakeEmbedder()
            transport = FakeTransport()
            protocol = prepare(proto_dir, embedder=embedder, transport=transport)

            # 1a. Alias primary_catalog_file to layout_table_catalog.json in protocol
            alias_p = copy.deepcopy(protocol)
            alias_p["both_arms"]["layout"]["primary_catalog_file"] = "layout_table_catalog.json"
            with self.assertRaises(ValueError) as ctx:
                validate_protocol(alias_p, protocol_dir=proto_dir)
            self.assertTrue(any(term in str(ctx.exception).lower() for term in ("canonical filename", "arm swap or corruption detected", "forbidden t95", "mismatch")))

            # 1b. Content T95 in primary catalog on disk
            prim_cat_path = proto_dir / "layout_primary_catalog.json"
            orig_prim_bytes = prim_cat_path.read_bytes()
            cat_data = json.loads(orig_prim_bytes.decode("utf-8"))
            cat_data["units"].append({
                "id": "T95",
                "kind": "body",
                "complete_end": True,
                "spans": [{"doc_id": "airef_2025_opinion_sostenibilidad", "source_file": "data/raw/airef_2025_opinion_sostenibilidad.pdf", "published_at": "2025-01-01", "page_number": 95, "start_char": 0, "end_char": 10, "quote": "CUADRO 9."}]
            })
            prim_cat_path.write_text(json.dumps(cat_data), encoding="utf-8")
            p_bytes = prim_cat_path.read_bytes()
            alias_hash_p = copy.deepcopy(protocol)
            alias_hash_p["generated_assets"]["layout_primary_catalog.json"]["raw_sha256"] = hashlib.sha256(p_bytes).hexdigest()
            alias_hash_p["generated_assets"]["layout_primary_catalog.json"]["canonical_sha256"] = hashlib.sha256(json.dumps(cat_data, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
            with self.assertRaises(ValueError) as ctx:
                validate_protocol(alias_hash_p, protocol_dir=proto_dir)
            self.assertTrue(any(term in str(ctx.exception).lower() for term in ("forbidden t95", "do not match", "count mismatch", "arm swap or corruption detected")))

            # 1c. Changed unit with same count (13 units) in primary catalog
            prim_cat_path.write_bytes(orig_prim_bytes)  # restore
            cat_data = json.loads(orig_prim_bytes.decode("utf-8"))
            cat_data["units"][0]["spans"][0]["quote"] = "tampered quote"
            prim_cat_path.write_text(json.dumps(cat_data), encoding="utf-8")
            p_bytes = prim_cat_path.read_bytes()
            tampered_p = copy.deepcopy(protocol)
            tampered_p["generated_assets"]["layout_primary_catalog.json"]["raw_sha256"] = hashlib.sha256(p_bytes).hexdigest()
            tampered_p["generated_assets"]["layout_primary_catalog.json"]["canonical_sha256"] = hashlib.sha256(json.dumps(cat_data, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
            with self.assertRaises(ValueError) as ctx:
                validate_protocol(tampered_p, protocol_dir=proto_dir)
            self.assertTrue(any(term in str(ctx.exception).lower() for term in ("do not match", "arm swap or corruption detected")))

            # 1d. Secondary missing T95
            prim_cat_path.write_bytes(orig_prim_bytes)  # restore
            tbl_cat_path = proto_dir / "layout_table_catalog.json"
            orig_tbl_bytes = tbl_cat_path.read_bytes()
            tbl_data = json.loads(orig_tbl_bytes.decode("utf-8"))
            tbl_data["units"] = [u for u in tbl_data["units"] if u.get("id") != "T95"]  # remove T95
            tbl_cat_path.write_text(json.dumps(tbl_data), encoding="utf-8")
            t_bytes = tbl_cat_path.read_bytes()
            no_t95_p = copy.deepcopy(protocol)
            no_t95_p["generated_assets"]["layout_table_catalog.json"]["raw_sha256"] = hashlib.sha256(t_bytes).hexdigest()
            no_t95_p["generated_assets"]["layout_table_catalog.json"]["canonical_sha256"] = hashlib.sha256(json.dumps(tbl_data, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
            with self.assertRaises(ValueError) as ctx:
                validate_protocol(no_t95_p, protocol_dir=proto_dir)
            self.assertTrue(any(term in str(ctx.exception).lower() for term in ("missing t95", "do not match", "unit/span count mismatch")))
            tbl_cat_path.write_bytes(orig_tbl_bytes)

    # Finding 2 (v2): closed role-path map, alternate valid index file rejected, swap primary/table, coherent swapped arm fields
    def test_validate_protocol_closed_role_paths_and_alternate_index_rejected(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            proto_dir = Path(tmp_dir) / "proto_workspace"
            embedder = FakeEmbedder()
            transport = FakeTransport()
            protocol = prepare(proto_dir, embedder=embedder, transport=transport)

            # 2a. Alternate valid index file with new normalized vectors pointed to by index_file
            alt_idx_path = proto_dir / "alternate_valid_index.json"
            orig_idx_data = json.loads((proto_dir / "baseline_index.json").read_text(encoding="utf-8"))
            alt_idx_path.write_text(json.dumps(orig_idx_data), encoding="utf-8")

            alt_p = copy.deepcopy(protocol)
            alt_p["both_arms"]["baseline"]["index_file"] = "alternate_valid_index.json"
            with self.assertRaises(ValueError) as ctx:
                validate_protocol(alt_p, protocol_dir=proto_dir)
            self.assertTrue(any(term in str(ctx.exception).lower() for term in ("exact expected canonical filename", "arm swap or corruption detected")))

            # 2b. Swap primary and table catalog paths
            swap_cat_p = copy.deepcopy(protocol)
            swap_cat_p["both_arms"]["baseline"]["primary_catalog_file"] = "baseline_table_catalog.json"
            swap_cat_p["both_arms"]["baseline"]["table_catalog_file"] = "baseline_primary_catalog.json"
            with self.assertRaises(ValueError) as ctx:
                validate_protocol(swap_cat_p, protocol_dir=proto_dir)
            self.assertTrue(any(term in str(ctx.exception).lower() for term in ("exact expected canonical filename", "arm swap or corruption detected")))

            # 2c. Coherent swapped arm fields
            swap_arm_p = copy.deepcopy(protocol)
            swap_arm_p["both_arms"]["baseline"] = copy.deepcopy(protocol["both_arms"]["layout"])
            swap_arm_p["both_arms"]["layout"] = copy.deepcopy(protocol["both_arms"]["baseline"])
            with self.assertRaises(ValueError) as ctx:
                validate_protocol(swap_arm_p, protocol_dir=proto_dir)
            self.assertTrue(any(term in str(ctx.exception).lower() for term in ("exact expected canonical filename", "arm swap or corruption detected")))

    # Finding 3 (v2): one/multiple execute exceptions recorded with identity and exported successfully in full 34 denominator
    def test_run_and_export_review_with_technical_failures(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            proto_dir = Path(tmp_dir) / "proto_workspace"
            embedder = FakeEmbedder()
            transport = FakeTransport()
            prepare(proto_dir, embedder=embedder, transport=transport)

            handler = FakeExecutionHandler(fail_at_indices={4, 10})
            results_path = Path(tmp_dir) / "results.json"
            results = run(proto_dir / "protocol.json", results_path, transport=transport, execute_fn=handler)

            self.assertEqual(results["run_status"], "completed")
            self.assertEqual(len(results["attempts"]), 34)
            self.assertEqual(results["summary"]["completed_attempts"], 32)
            self.assertEqual(results["summary"]["error_attempts"], 2)

            # Check synthetic_exception_record on failed rows
            failed_att4 = results["attempts"][3]
            self.assertEqual(failed_att4["status"], "error")
            self.assertEqual(failed_att4["raw_attempt"]["synthetic_exception_record"], True)
            self.assertEqual(failed_att4["raw_attempt"]["arm"], "mixed")
            self.assertEqual(failed_att4["raw_attempt"]["case_id"], failed_att4["case_id"])
            self.assertEqual(failed_att4["raw_attempt"]["query"], failed_att4["query"])
            self.assertEqual(failed_att4["raw_attempt"]["attempt_index"], 4)
            self.assertEqual(failed_att4["raw_attempt"]["formatted_answer"], "")
            self.assertEqual(failed_att4["formatted_answer"], "")
            self.assertTrue(len(failed_att4["raw_attempt_canonical_sha256"]) == 64)

            # Export review with technical errors present
            review_out = Path(tmp_dir) / "review_bundle.json"
            bundle = export_review(proto_dir / "protocol.json", results_path, review_out)

            self.assertEqual(len(bundle["cases"]), 17)
            # Find the case corresponding to attempt 4
            case_q1 = next(c for c in bundle["cases"] if c["question_index"] == failed_att4["question_index"])
            # One arm has technical error, one arm completed
            arm_err = case_q1["arms"]["A"] if case_q1["arms"]["A"]["technical_error"] else case_q1["arms"]["B"]
            arm_ok = case_q1["arms"]["B"] if case_q1["arms"]["A"]["technical_error"] else case_q1["arms"]["A"]
            self.assertIn("Simulated technical failure at attempt 4", arm_err["technical_error"])
            self.assertEqual(arm_err["formatted_answer"], "")
            self.assertTrue(len(arm_ok["formatted_answer"]) > 0)

    # Finding 4 (v2): deleted raw SHA, changed internal+external answer, mismatched index hash, copied other arm raw attempt rejected
    def test_export_review_tampering_raw_sha_index_hash_other_arm(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            proto_dir = Path(tmp_dir) / "proto_workspace"
            embedder = FakeEmbedder()
            transport = FakeTransport()
            prepare(proto_dir, embedder=embedder, transport=transport)

            results_path = Path(tmp_dir) / "results.json"
            handler = FakeExecutionHandler()
            run(proto_dir / "protocol.json", results_path, transport=transport, execute_fn=handler)

            valid_res = json.loads(results_path.read_text(encoding="utf-8"))
            review_out = Path(tmp_dir) / "review.json"

            # 4a. Deleted raw_attempt_canonical_sha256
            no_sha_res = copy.deepcopy(valid_res)
            del no_sha_res["attempts"][0]["raw_attempt_canonical_sha256"]
            p_no_sha = Path(tmp_dir) / "no_sha_results.json"
            p_no_sha.write_text(json.dumps(no_sha_res), encoding="utf-8")
            with self.assertRaises(ValueError) as ctx:
                export_review(proto_dir / "protocol.json", p_no_sha, review_out)
            self.assertIn("raw_attempt_canonical_sha256", str(ctx.exception))

            # 4b. Changed internal + external answer
            both_ans_res = copy.deepcopy(valid_res)
            both_ans_res["attempts"][0]["formatted_answer"] = "tampered both"
            both_ans_res["attempts"][0]["raw_attempt"]["formatted_answer"] = "tampered both"
            p_both = Path(tmp_dir) / "both_ans_results.json"
            p_both.write_text(json.dumps(both_ans_res), encoding="utf-8")
            with self.assertRaises(ValueError) as ctx:
                export_review(proto_dir / "protocol.json", p_both, review_out)
            self.assertIn("raw_attempt_canonical_sha256 mismatch", str(ctx.exception))

            # 4c. Mismatched index hash in input_hashes
            bad_idx_res = copy.deepcopy(valid_res)
            bad_idx_res["attempts"][0]["input_hashes"]["index_sha256"] = "0" * 64
            p_bad_idx = Path(tmp_dir) / "bad_idx_results.json"
            p_bad_idx.write_text(json.dumps(bad_idx_res), encoding="utf-8")
            with self.assertRaises(ValueError) as ctx:
                export_review(proto_dir / "protocol.json", p_bad_idx, review_out)
            self.assertIn("input index_sha256 mismatch", str(ctx.exception))

            # 4d. Copied other arm raw attempt with relabeled wrapper
            copy_arm_res = copy.deepcopy(valid_res)
            copy_arm_res["attempts"][0]["raw_attempt"] = copy.deepcopy(valid_res["attempts"][1]["raw_attempt"])
            copy_arm_res["attempts"][0]["raw_attempt_canonical_sha256"] = valid_res["attempts"][1]["raw_attempt_canonical_sha256"]
            copy_arm_res["attempts"][0]["formatted_answer"] = valid_res["attempts"][1]["formatted_answer"]
            p_copy_arm = Path(tmp_dir) / "copy_arm_results.json"
            p_copy_arm.write_text(json.dumps(copy_arm_res), encoding="utf-8")
            with self.assertRaises(ValueError) as ctx:
                export_review(proto_dir / "protocol.json", p_copy_arm, review_out)
            self.assertTrue(any(term in str(ctx.exception).lower() for term in ("attempt_index", "condition", "mismatch")))

    # Finding 5 (v2): extra absolute alias of existing relative path and two spellings same file rejected
    def test_validate_protocol_input_hashes_alias_and_spellings_rejected(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            proto_dir = Path(tmp_dir) / "proto_workspace"
            embedder = FakeEmbedder()
            transport = FakeTransport()
            protocol = prepare(proto_dir, embedder=embedder, transport=transport)

            # 5a. Extra absolute alias of existing relative path (10 entries)
            proto_abs = copy.deepcopy(protocol)
            abs_key = (PROJECT_ROOT / DEFAULT_LEGACY_PAGES_PATH).resolve().as_posix()
            proto_abs["input_file_hashes"][abs_key] = protocol["input_file_hashes"][DEFAULT_LEGACY_PAGES_PATH.as_posix()]
            with self.assertRaises(ValueError) as ctx:
                validate_protocol(proto_abs, protocol_dir=proto_dir)
            self.assertTrue(any(term in str(ctx.exception).lower() for term in ("raw entry count", "keyset mismatch", "extra", "alias duplicate")))

            # 5b. Two spellings of the same file (e.g. data/processed/./airef_2025_pages.jsonl)
            proto_spell = copy.deepcopy(protocol)
            del proto_spell["input_file_hashes"]["evaluation/reconstructed_units_catalog.json"]
            proto_spell["input_file_hashes"]["data/processed/./airef_2025_pages.jsonl"] = protocol["input_file_hashes"][DEFAULT_LEGACY_PAGES_PATH.as_posix()]
            with self.assertRaises(ValueError) as ctx:
                validate_protocol(proto_spell, protocol_dir=proto_dir)
            self.assertTrue(any(term in str(ctx.exception).lower() for term in ("alias duplicate", "not canonical relative path", "keyset mismatch")))


if __name__ == "__main__":
    unittest.main()
