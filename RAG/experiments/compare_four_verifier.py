"""Controles de desarrollo separados para el verificador base y especializado."""
import hashlib
import json
from pathlib import Path
import sys

from experiments.four_strategies import SpecialistVerifierGrounder
from rag_pensiones.evaluate_grounding import evaluate_pairs
from rag_pensiones.grounding import OllamaGrounder
from rag_pensiones.ingest import PROJECT_ROOT


def run_controls():
    base, specialist = OllamaGrounder(), SpecialistVerifierGrounder()
    digest = base.generator.get_digest()
    if specialist.generator.get_digest() != digest: raise RuntimeError("Different verifier digests")
    report = {"schema_version": "four-verifier-controls-v1", "run_status": "running", "datasets": {}}
    failed = False
    for name, filename in [("development", "four_verifier_dev.json"), ("stress", "four_verifier_stress.json")]:
        path = PROJECT_ROOT / "evaluation" / filename
        data = json.loads(path.read_text(encoding="utf-8"))
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        report["datasets"][name] = {}
        for method, grounder in [("baseline", base), ("specialist", specialist)]:
            if grounder.generator.get_digest() != digest: raise RuntimeError("Digest changed")
            if name == "development" and method == "baseline":
                result = json.loads((PROJECT_ROOT / "evaluation/four_verifier_baseline.json").read_text(encoding="utf-8"))
                if result["dataset_sha256"] != sha or result["model_digest"] != digest or result["run_status"] != "completed":
                    raise ValueError("Baseline capture provenance mismatch")
            else:
                result = evaluate_pairs(data, split="dev", grounder=grounder, dataset_sha256=sha)
            report["datasets"][name][method] = result
            if result["run_status"] != "completed": failed = True
            (PROJECT_ROOT / "evaluation/four_verifier_results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            print(name, method, result.get("metrics"), flush=True)
    report["run_status"] = "error" if failed else "completed"
    (PROJECT_ROOT / "evaluation/four_verifier_results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    try:
        result = run_controls()
        sys.exit(0 if result["run_status"] == "completed" else 1)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
