"""Paired development comparison of original chunks and PDF parent reconstruction."""
import argparse
import copy
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

from experiments.reconstructed_units import ReconstructedGrounder
from experiments.source_bound_trial import RecordedGenerator, checkpoint
from experiments.validate_reranker import canonical_hash, output_path, validate_queries
from rag_pensiones.answer import DEFAULT_GENERATION_MODEL
from rag_pensiones.embeddings import OllamaEmbedder
from rag_pensiones.grounding import OllamaGrounder, validate_grounded_result
from rag_pensiones.ingest import validate_project_path
from rag_pensiones.search import DEFAULT_INDEX_PATH, load_index, search


def run(index, queries, units, pages, output):
    validate_queries(queries)
    destination = output_path(output)
    models = {"baseline": OllamaGrounder(generator=RecordedGenerator()),
              "reconstructed": ReconstructedGrounder(units, pages, generator=RecordedGenerator())}
    digest = models["baseline"].generator.get_digest()
    report = {"schema_version": "reconstructed-parents-trial-v1", "run_status": "running",
              "metadata": {"model": DEFAULT_GENERATION_MODEL, "digest": digest, "gold_access": "none",
                           "queries_hash": canonical_hash(queries), "index_hash": canonical_hash(index),
                           "units_hash": canonical_hash(units), "pages_hash": canonical_hash(pages),
                           "design": "Top4 seed search shared; candidate expands to PDF parents then selects one opener+sentence bundle; no planner"}, "cases": []}
    embedder = OllamaEmbedder()
    checkpoint(report, destination)
    for number, case in enumerate(queries["cases"]):
        order = list(models) if number % 2 == 0 else list(reversed(models))
        row = dict(case, conditions={}, condition_order=order)
        report["cases"].append(row)
        try:
            seeds = search(index, case["query"], top_k=4, client=embedder)
            row["seeds"] = [{"rank": s.rank, "score": s.score, "chunk_id": s.chunk.chunk_id} for s in seeds]
        except Exception as exc:
            row["conditions"] = {m: {"status": "error", "error": str(exc)} for m in models}
            checkpoint(report, destination)
            continue
        for method in order:
            g = models[method]
            g.generator.calls.clear()
            if method == "reconstructed":
                g.expansion, g.catalog, g.expanded_hits, g.selections = {}, {}, [], []
            started = time.perf_counter()
            try:
                result = g.answer(case["query"], copy.deepcopy(seeds), expected_digest=digest)
                evaluated_hits = seeds if method == "baseline" else g.expanded_hits
                validate_grounded_result(result, evaluated_hits, expected_digest=digest, expected_model=DEFAULT_GENERATION_MODEL)
                outcome = {"status": result["status"], "full_result": result}
            except Exception as exc:
                outcome = {"status": "error", "error": str(exc)}
            outcome.update(wall_seconds=time.perf_counter() - started, calls=copy.deepcopy(g.generator.calls))
            if method == "reconstructed":
                outcome.update(expansion=copy.deepcopy(g.expansion), catalog=copy.deepcopy(g.catalog),
                               expanded_hits=[{"rank": h.rank, "score": h.score, "chunk": asdict(h.chunk)} for h in g.expanded_hits],
                               selections=copy.deepcopy(g.selections))
            row["conditions"][method] = outcome
            checkpoint(report, destination)
            print(case["id"], method, outcome["status"], flush=True)
    report["run_status"] = "error" if any(o["status"] == "error" for r in report["cases"] for o in r["conditions"].values()) else "completed"
    checkpoint(report, destination)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=Path, default=Path("evaluation/reranker_validation_queries.json"))
    parser.add_argument("--units", type=Path, default=Path("evaluation/reconstructed_units_catalog.json"))
    parser.add_argument("--pages", type=Path, default=Path("data/processed/airef_2025_pages.jsonl"))
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX_PATH)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    read = lambda p: json.loads(validate_project_path(p).read_text(encoding="utf-8"))
    pages = [json.loads(line) for line in validate_project_path(args.pages).read_text(encoding="utf-8").splitlines()]
    result = run(load_index(validate_project_path(args.index)), read(args.queries), read(args.units), pages, args.output)
    return 1 if result["run_status"] == "error" else 0


if __name__ == "__main__":
    sys.exit(main())
