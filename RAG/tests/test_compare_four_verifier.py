import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from experiments.compare_four_verifier import run_controls


class VerifierCheckpointTests(unittest.TestCase):
    def check_run(self, interrupted):
        with tempfile.TemporaryDirectory(dir=Path.cwd() / "tmp") as directory:
            root = Path(directory); (root / "evaluation").mkdir()
            for filename in ["four_verifier_dev.json", "four_verifier_stress.json"]:
                (root / "evaluation" / filename).write_text("{}", encoding="utf-8")
            baseline = {"dataset_sha256": hashlib.sha256(b"{}").hexdigest(), "model_digest": "d", "run_status": "completed", "metrics": {}}
            (root / "evaluation/four_verifier_baseline.json").write_text(json.dumps(baseline), encoding="utf-8")
            grounder = Mock();grounder.generator.get_digest.return_value = "d"
            results = [{"run_status": "completed", "metrics": {}}] * 3
            if interrupted: results[1] = RuntimeError("Interrupted stress controls")
            with patch("experiments.compare_four_verifier.PROJECT_ROOT", root), patch("experiments.compare_four_verifier.OllamaGrounder", return_value=grounder), patch("experiments.compare_four_verifier.SpecialistVerifierGrounder", return_value=grounder), patch("experiments.compare_four_verifier.evaluate_pairs", side_effect=results), patch("sys.stdout", io.StringIO()):
                if interrupted:
                    with self.assertRaises(RuntimeError): run_controls()
                else:
                    self.assertEqual(run_controls()["run_status"], "completed")
            return json.loads((root / "evaluation/four_verifier_results.json").read_text(encoding="utf-8"))

    def test_partial_checkpoint_never_claims_completion(self):
        result = self.check_run(True)
        self.assertEqual(result["run_status"], "running")
        self.assertNotIn("baseline", result["datasets"].get("stress", {}))

    def test_completion_only_after_all_four_subexperiments(self):
        result = self.check_run(False)
        self.assertEqual(result["run_status"], "completed")
        self.assertTrue(all(set(methods) == {"baseline", "specialist"} for methods in result["datasets"].values()))
