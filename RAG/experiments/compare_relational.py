"""Desarrollo pareado: recuperación fija, antecedentes y auditoría aislada por claim."""
import argparse
import copy
import json
from pathlib import Path
import sys
import time

from experiments.relational_grounding import AntecedentGrounder, AuditFailure, audit_claims
from experiments.validate_reranker import canonical_hash, checkpoint, output_path, validate_queries
from rag_pensiones.answer import DEFAULT_GENERATION_MODEL, _reject_constant
from rag_pensiones.embeddings import OllamaEmbedder
from rag_pensiones.grounding import OllamaGrounder, validate_grounded_result
from rag_pensiones.ingest import validate_project_path
from rag_pensiones.search import DEFAULT_INDEX_PATH, load_index, search, validate_index


def pin(grounder, digest):
    if grounder.model != DEFAULT_GENERATION_MODEL or grounder.generator.model != DEFAULT_GENERATION_MODEL:
        raise ValueError("Qwen9B required")
    if grounder.generator.get_digest() != digest:
        raise RuntimeError("Model digest changed")


def run_experiment(index, queries, output=None, baseline=None, antecedent=None, auditor=None, embedder=None):
    validate_queries(queries)
    if len(validate_index(index)) < 4:
        raise ValueError("Need at least four chunks")
    destination = output_path(output)
    baseline = baseline if baseline is not None else OllamaGrounder()
    antecedent = antecedent if antecedent is not None else AntecedentGrounder()
    auditor = auditor if auditor is not None else OllamaGrounder()
    digest = baseline.generator.get_digest()
    if not isinstance(digest, str) or not digest.strip():
        raise ValueError("Empty digest")
    for grounder in (baseline, antecedent, auditor):
        pin(grounder, digest)
    embedder = embedder if embedder is not None else OllamaEmbedder()
    report = {"schema_version": "relational-development-results-v1", "run_status": "running",
        "metadata": {"model": DEFAULT_GENERATION_MODEL, "digest": digest, "top_k": 4,
            "queries_canonical_sha256": canonical_hash(queries), "index_canonical_sha256": canonical_hash(index),
            "gold_access": "none", "design": "40 generation conditions plus20 derived audit conditions for20 queries; same antecedent result audited, not regenerated",
            "audit_scope": "one isolated claim and its assigned quotes per call, at most4calls",
            "usage_limit": "Failed-call usage may be unknown; derived conditions reuse generation and must not be counted as independent runs."}, "cases": []}
    failed = False
    for number, case in enumerate(queries["cases"]):
        methods = ["baseline", "antecedent"] if number % 2 == 0 else ["antecedent", "baseline"]
        row = dict(case, condition_order=methods, conditions={})
        report["cases"].append(row)
        try:
            hits = search(index, case["query"], top_k=4, client=embedder)
            row["retrieved"] = [{"rank": h.rank, "chunk_id": h.chunk.chunk_id,
                "cosine_score": h.score, "page_number": h.chunk.page_number} for h in hits]
        except Exception as exc:
            failed = True
            for method in methods + ["antecedent_plus_audit"]:
                row["conditions"][method] = {"status": "error", "error": f"Retrieval: {exc}"}
                checkpoint(report, destination)
            continue
        for method in methods:
            started = time.perf_counter()
            try:
                g = baseline if method == "baseline" else antecedent
                pin(g, digest)
                result = g.answer(query=case["query"], hits=copy.deepcopy(hits), expected_digest=digest)
                result = validate_grounded_result(result, hits, expected_digest=digest, expected_model=DEFAULT_GENERATION_MODEL)
                pin(g, digest)
                outcome = {"status": result["status"], "full_result": copy.deepcopy(result)}
            except Exception as exc:
                failed = True
                outcome = {"status": "error", "error": str(exc)}
            outcome["wall_seconds"] = time.perf_counter() - started
            row["conditions"][method] = outcome
            checkpoint(report, destination)
            print(case["id"], method, outcome["status"], file=sys.stderr, flush=True)
        started = time.perf_counter()
        try:
            parent = row["conditions"]["antecedent"]
            if parent["status"] == "error":
                raise RuntimeError("Antecedent generation failed; audit not run")
            pin(auditor, digest)
            audit = audit_claims(parent["full_result"], auditor, digest)
            pin(auditor, digest)
            derived = {"status": audit["effective_status"], "audit": audit,
                       "generation_reused_from": "antecedent"}
        except Exception as exc:
            failed = True
            derived = {"status": "error", "error": str(exc), "generation_reused_from": "antecedent"}
            if isinstance(exc, AuditFailure):
                derived["audit_failure_calls"] = copy.deepcopy(exc.calls)
        derived["audit_wall_seconds"] = time.perf_counter() - started
        row["conditions"]["antecedent_plus_audit"] = derived
        checkpoint(report, destination)
        print(case["id"], "antecedent_plus_audit", derived["status"], file=sys.stderr, flush=True)
    report["run_status"] = "error" if failed else "completed"
    checkpoint(report, destination)
    return report


def run_controls(controls, output=None, grounder=None):
    if not isinstance(controls, dict) or set(controls) != {"schema_version", "cases"} or controls["schema_version"] != "relational-controls-v1":
        raise ValueError("Invalid controls schema")
    cases = controls["cases"]
    if not isinstance(cases, list) or not 1 <= len(cases) <= 20:
        raise ValueError("Invalid controls count")
    seen = set()
    from experiments.relational_grounding import audit_input
    for c in cases:
        if not isinstance(c, dict) or set(c) != {"id", "claim", "evidence"} or not isinstance(c["id"], str) or not c["id"].strip() or c["id"] in seen:
            raise ValueError("Invalid control keys/ID")
        seen.add(c["id"])
        audit_input({"status": "answered", "claims": [{"text": c["claim"], "evidence": [{"quote": q} for q in c["evidence"]]}]})
    destination = output_path(output)
    grounder = grounder if grounder is not None else OllamaGrounder()
    digest = grounder.generator.get_digest()
    if not isinstance(digest, str) or not digest.strip():
        raise ValueError("Empty digest")
    pin(grounder, digest)
    report = {"schema_version": "relational-controls-results-v1", "run_status": "running",
              "controls_canonical_sha256": canonical_hash(controls), "digest": digest, "cases": []}
    failed = False
    for c in cases:
        row = {"id": c["id"], "conditions": {}}
        report["cases"].append(row)
        # Counterbalance two methods without allowing references into either verifier.
        order = ["baseline", "dimensional"] if len(report["cases"]) % 2 else ["dimensional", "baseline"]
        row["condition_order"] = order
        for method in order:
            try:
                pin(grounder, digest)
                if method == "baseline":
                    ev = [{"evidence_id": f"E{i}", "source_id": f"S{i}", "quote": q} for i, q in enumerate(c["evidence"], 1)]
                    claims = [{"claim_id": "C1", "text": c["claim"], "evidence_ids": [e["evidence_id"] for e in ev]}]
                    result = grounder.verify(claims, ev, expected_digest=digest)
                    verdict = result["verdicts"][0]["verdict"]
                else:
                    public = {"status": "answered", "claims": [{"text": c["claim"], "evidence": [{"quote": q} for q in c["evidence"]]}]}
                    result = audit_claims(public, grounder, digest)
                    verdict = result["verdicts"][0]["verdict"]
                pin(grounder, digest)
                row["conditions"][method] = {"status": "completed", "verdict": verdict, "result": result}
            except Exception as exc:
                failed = True
                row["conditions"][method] = {"status": "error", "error": str(exc)}
                if isinstance(exc, AuditFailure):
                    row["conditions"][method]["audit_failure_calls"] = copy.deepcopy(exc.calls)
            checkpoint(report, destination)
            print(c["id"], method, row["conditions"][method]["status"], file=sys.stderr, flush=True)
    report["run_status"] = "error" if failed else "completed"
    checkpoint(report, destination)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--queries", type=Path)
    group.add_argument("--controls", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX_PATH)
    try:
        args = parser.parse_args(argv)
        path = args.queries if args.queries else args.controls
        data = json.loads(validate_project_path(path).read_text(encoding="utf-8-sig"), parse_constant=_reject_constant)
        output_path(args.output)
        if args.controls:
            result = run_controls(data, output=args.output)
        else:
            validate_queries(data)
            index = load_index(validate_project_path(args.index))
            result = run_experiment(index, data, output=args.output)
        print(json.dumps({"run_status": result["run_status"], "cases": len(result["cases"])}, ensure_ascii=False))
        return 0 if result["run_status"] == "completed" else 1
    except SystemExit as exc:
        return exc.code
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
