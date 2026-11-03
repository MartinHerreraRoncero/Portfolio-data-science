"""Integration checks using actual immutable source pages and recorded model responses."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments import query_coverage as qc
from experiments import continuous_quality as cq
from experiments import evaluate_covered_pipeline as runner
from experiments.validate_reranker import canonical_hash

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "evaluation/quality_followup_2026-10-28"

def load(path):
    return json.loads(path.read_text(encoding="utf-8"))

class SavedClient:
    def __init__(self, generation, calls):
        self.model = qc.PINNED_9B_MODEL if generation else qc.PINNED_27B_MODEL
        self.expected_digest = qc.PINNED_9B_DIGEST if generation else qc.PINNED_27B_DIGEST
        self.parameters = copy.deepcopy(qc.PINNED_9B_PARAMETERS if generation else qc.PINNED_27B_PARAMETERS)
        self.pending = copy.deepcopy(calls)
        self.calls = []
    def get_digest(self):
        return self.expected_digest
    def release_loaded(self, model):
        pass
    def chat(self, messages, response_schema, *, stage=None):
        call = self.pending.pop(0)
        assert messages == call["request"]["messages"]
        assert response_schema == call["request"]["format"]
        if stage is not None:
            assert stage == call["stage"]
        self.calls.append(copy.deepcopy(call))
        return copy.deepcopy(call["raw_response"])

class TestCorrectionIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old = load(BASE / "live_mixed_v3/results.json")
        cls.saved = load(BASE / "saved_input_probe_v2/results.json")
        cls.pages = [json.loads(line) for line in (ROOT / "data/processed/airef_2025_pages.jsonl").read_text(encoding="utf-8").splitlines()]

    def record(self, case):
        return dict(attempted=case["attempted"], applied=case["applied"],
                    inventory=copy.deepcopy(case["corrected_inventory"]),
                    coverage_audit=copy.deepcopy(case["corrected_coverage_audit"]),
                    diagnostics=copy.deepcopy(case["diagnostics"]))

    def test_all_eleven_actual_saved_traces_replay_without_network(self):
        for old, saved in zip(self.old["cases"], self.saved["cases"]):
            with self.subTest(case=saved["id"]):
                record = self.record(saved)
                result = qc.replay_coverage_correction(saved["query"], old["answer"]["factual_report"],
                    saved["original_coverage_audit"], record, pages=self.pages)
                self.assertEqual(record["applied"], result["applied"])
                self.assertEqual(record["inventory"], result["inventory"])

    def test_altered_lineage_is_rejected(self):
        old, saved = self.old["cases"][9], self.saved["cases"][9]
        mutations = [
            lambda r: r.update(applied=False),
            lambda r: r.update(applied=1),
            lambda r: r["inventory"][-1].update(text="inventado"),
            lambda r: r["diagnostics"]["generation"]["client_calls"][0]["request"]["options"].update(seed=43),
            lambda r: r["diagnostics"]["generation"]["client_calls"][0]["request"].update(model="foreign"),
            lambda r: r["diagnostics"]["verification"]["raw_response"].update(model="foreign"),
            lambda r: r["diagnostics"]["reaudit"]["validated_payload"].update(status="insufficient_evidence"),
            lambda r: r["diagnostics"]["generation"].update(wall_seconds=float("inf")),
            lambda r: r.update(unknown=True),
            lambda r: r["diagnostics"].update(wall_seconds=1),
        ]
        for mutate in mutations:
            record = self.record(saved)
            mutate(record)
            with self.subTest(mutation=mutate), self.assertRaises((ValueError, TypeError)):
                qc.replay_coverage_correction(saved["query"], old["answer"]["factual_report"],
                    saved["original_coverage_audit"], record, pages=self.pages)
        initial = copy.deepcopy(saved["original_coverage_audit"])
        initial["validated_payload"]["parts"][0]["reason_code"] = "not_in_evaluated_evidence"
        with self.assertRaises(ValueError):
            qc.replay_coverage_correction(saved["query"], old["answer"]["factual_report"], initial,
                                         self.record(saved), pages=self.pages)

    def test_api_real_correction_preserves_canonical_and_refusals(self):
        for idx in (5, 8, 9, 10):
            old, saved = self.old["cases"][idx], self.saved["cases"][idx]
            factual = copy.deepcopy(old["answer"]["factual_report"])
            before = copy.deepcopy(factual)
            diag = saved["diagnostics"]
            gen = SavedClient(True, diag["generation"]["client_calls"])
            ver_calls = list(saved["original_coverage_audit"]["calls"])
            for key in ("verification", "reaudit"):
                if isinstance(diag.get(key), dict):
                    ver_calls += diag[key]["client_calls"]
            ver = SavedClient(False, ver_calls)
            with patch.object(qc, "answer_preserved", return_value=factual):
                answer = qc.answer_covered({}, saved["query"], {}, self.pages,
                                           generator=gen, verifier=ver, coverage_correction=True)
            with self.subTest(case=saved["id"]):
                self.assertNotEqual("error", answer["status"], answer.get("error"))
                self.assertEqual("preserved-query-coverage-v2", answer["schema_version"])
                self.assertEqual(saved["applied"], answer["coverage_correction"]["applied"])
                self.assertEqual(before, factual)
                self.assertFalse(gen.pending or ver.pending)
                if idx == 10:
                    self.assertEqual("over_abstention", answer["coverage_correction"]["diagnostics"]["trigger"])
                    self.assertEqual("missing_from_answer", answer["initial_coverage_audit"]["validated_payload"]["parts"][0]["reason_code"])
                    self.assertIn("2015", qc.format_covered_answer(answer))
                    self.assertIn("pág. 99", qc.format_covered_answer(answer))

    def test_api_legacy_identity_flags_and_technical_error(self):
        with patch.object(qc, "_answer_covered_initial", return_value={"legacy": True}) as initial:
            self.assertEqual({"legacy": True}, qc.answer_covered({}, "q", {}, [], structured_retrieval=False))
            self.assertIs(initial.call_args.kwargs["structured_retrieval"], False)
            self.assertNotIn("_empty_reason_code", initial.call_args.kwargs)
        for value in (1, 0, None, "true"):
            with self.assertRaises(ValueError):
                qc.answer_covered({}, "q", {}, [], coverage_correction=value)
        failure = {"outcome": {"status": "error", "error": "transport failed"}}
        with patch.object(qc, "answer_preserved", return_value=failure), patch.object(qc, "correct_coverage_once") as correct:
            answer = qc.answer_covered({}, "q", {}, [], generator=object(), verifier=object(), coverage_correction=True)
        self.assertEqual("error", answer["status"])
        self.assertIsNone(answer["coverage_correction"])
        self.assertIsNone(answer["initial_coverage_audit"])
        correct.assert_not_called()

    def test_prepare_runner_protocol_and_complete_offline_check(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as directory:
            directory = Path(directory)
            protocol = cq.prepare(directory, coverage_correction=True)
            runner.validate_protocol(protocol)
            for bad in (False, 1, None):
                tampered = copy.deepcopy(protocol)
                tampered["coverage_correction"] = bad
                with self.assertRaises(ValueError):
                    runner.validate_protocol(tampered)
            tampered = copy.deepcopy(protocol)
            tampered["schema_version"] = "covered-pipeline-protocol-v1"
            with self.assertRaises(ValueError):
                runner.validate_protocol(tampered)
            results = copy.deepcopy(self.old)
            results["metadata"].update(protocol_hash=canonical_hash(protocol),
                protocol_schema_version=protocol["schema_version"], coverage_correction=True)
            for case, saved in zip(results["cases"], self.saved["cases"]):
                ans = case["answer"]
                record = self.record(saved)
                ans.update(schema_version="preserved-query-coverage-v2",
                    initial_coverage_audit=saved["original_coverage_audit"], coverage_correction=record,
                    coverage_audit=record["coverage_audit"],
                    publication=qc.build_coverage_publication(case["query"], record["inventory"], record["coverage_audit"]["validated_payload"]))
                ans["status"] = ans["publication"]["status"]
                ans["metadata"].update(profile="preserved_query_coverage_v2", coverage_correction=True,
                                       coverage_status=ans["status"])
                outcome = ans["factual_report"]["outcome"]
                for key in ("calls", "verifier_calls"):
                    case[key] = copy.deepcopy(outcome.get(key, []))
                case["coverage_calls"] = copy.deepcopy(saved["original_coverage_audit"]["calls"])
                for trace_name, role in (("generation", "calls"), ("verification", "verifier_calls"), ("reaudit", "coverage_calls")):
                    trace = record["diagnostics"].get(trace_name)
                    if isinstance(trace, dict):
                        case[role].extend(copy.deepcopy(trace.get("client_calls", [])))
                case["per_role_stats"] = {role: runner._extract_role_stats(case[key]) for role, key in
                    (("generator", "calls"), ("factual_verifier", "verifier_calls"), ("coverage", "coverage_calls"))}
                case["status"] = ans["status"]
                case["formatted_answer"] = qc.format_covered_answer(ans)
            result_path = directory / "results.json"
            result_path.write_text(json.dumps(results), encoding="utf-8")
            report = cq.check(result_path, ROOT / cq.DEFAULT_REFERENCE_REL, directory / "quality.json", protocol_path=directory / "protocol.json")
            self.assertTrue(report["overall_pass"], report)
            results["cases"][9]["answer"]["coverage_correction"]["inventory"][-1]["text"] = "forged"
            result_path.write_text(json.dumps(results), encoding="utf-8")
            with self.assertRaises(ValueError):
                cq.check(result_path, ROOT / cq.DEFAULT_REFERENCE_REL, directory / "tampered.json", protocol_path=directory / "protocol.json")

if __name__ == "__main__":
    unittest.main()
