"""Paired grounded model comparison with shared active Qwen4B retrieval; no gold access."""
import argparse
import copy
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time
import urllib.request

from experiments.source_bound_trial import RecordedGenerator, checkpoint
from experiments.validate_reranker import canonical_hash, output_path, validate_queries
from rag_pensiones.answer import OllamaGenerator
from rag_pensiones.grounding import OllamaGrounder, validate_grounded_result
from rag_pensiones.ingest import validate_project_path
from rag_pensiones.qwen_embeddings import QwenEmbedder, MODEL as EMBEDDING_MODEL, _json
from rag_pensiones.search import DEFAULT_INDEX_PATH, load_index, search, validate_index

MODELS = {"qwen9b": "qwen3.5:9b", "horizon7b": "hf.co/IFM/K2-Horizon-7B-GGUF:Q4_K_M"}


class RecordedModel(RecordedGenerator):
    def __init__(self, model):
        OllamaGenerator.__init__(self, model=model)
        self.calls = []


class LocalControl:
    """Release model allocations between arms, without deleting model weights."""
    def __init__(self):
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def request(self, path, payload=None):
        data = None if payload is None else json.dumps(payload).encode()
        req = urllib.request.Request("http://127.0.0.1:11434" + path, data=data,
                                     headers={"Content-Type": "application/json"})
        with self.opener.open(req, timeout=60) as response:
            return _json(response.read())

    def snapshot(self):
        data = self.request("/api/ps")
        if not isinstance(data, dict) or not isinstance(data.get("models"), list):
            raise ValueError("Invalid Ollama memory snapshot")
        return data

    def unload(self, model):
        if model not in MODELS.values():
            raise ValueError("Only trial generators may be unloaded")
        result = self.request("/api/generate", {"model": model, "keep_alive": 0})
        if not isinstance(result, dict) or result.get("done") is not True or result.get("model") != model:
            raise RuntimeError("Ollama did not confirm generator release")
        if model in {m.get("name") for m in self.snapshot()["models"]}:
            raise RuntimeError("Generator still loaded after release")
        return result

    def ensure_no_other(self, model):
        snapshot = self.snapshot()
        others = set(MODELS.values()) - {model}
        if any(m.get("name") in others for m in snapshot["models"]):
            raise RuntimeError("Other trial generator is still resident")
        return snapshot


def run(index, queries, output, *, reverse=False, embedder=None, models=None, control=None):
    validate_queries(queries)
    records = validate_index(index)
    if index["model"] != EMBEDDING_MODEL:
        raise ValueError("This trial requires the active Qwen4B embedding index")
    destination = output_path(output)
    embedder = embedder if embedder is not None else QwenEmbedder(timeout=180)
    models = models if models is not None else {key: OllamaGrounder(generator=RecordedModel(name), model=name) for key, name in MODELS.items()}
    if set(models) != set(MODELS) or any(g.model != MODELS[key] for key, g in models.items()):
        raise ValueError("Trial generator identities must match exactly")
    control = control if control is not None else LocalControl()
    digests = {key: g.generator.get_digest() for key, g in models.items()}
    report = {"schema_version": "qwen-horizon-comparison-v1", "run_status": "running",
              "metadata": {"models": MODELS, "digests": digests, "embedding_model": EMBEDDING_MODEL,
                           "embedding_digest": index["model_digest"], "index_hash": canonical_hash(index),
                           "queries_hash": canonical_hash(queries), "gold_access": "none", "reverse": reverse,
                           "design": "Shared top4; each model selects/generates/verifies/repairs; no planner; generator released between arms"},
              "startup_releases": [], "cases": []}
    checkpoint(report, destination)
    try:
        for name in MODELS.values():
            report["startup_releases"].append(control.unload(name))
    except Exception as exc:
        report.update(run_status="error", startup_error=str(exc))
        checkpoint(report, destination)
        return report
    positions = {c["id"]: n for n, c in enumerate(queries["cases"])}
    cases = list(reversed(queries["cases"])) if reverse else queries["cases"]
    for case in cases:
        order = list(MODELS) if (positions[case["id"]] + int(reverse)) % 2 == 0 else list(reversed(MODELS))
        row = dict(case, condition_order=order, conditions={})
        report["cases"].append(row)
        embedder.calls.clear()
        started = time.perf_counter()
        try:
            ranked = search(index, case["query"], top_k=len(records), client=embedder)
            hits = ranked[:4]
            row["retrieval"] = {"status": "completed", "wall_seconds": time.perf_counter()-started,
                                "ranking": [{"rank": h.rank, "score": h.score, "chunk_id": h.chunk.chunk_id} for h in ranked],
                                "hits": [{"rank": h.rank, "score": h.score, "chunk": asdict(h.chunk)} for h in hits],
                                "embedding_calls": copy.deepcopy(embedder.calls)}
        except Exception as exc:
            row["retrieval"] = {"status": "error", "error": str(exc), "embedding_calls": copy.deepcopy(embedder.calls)}
            for key in order:
                row["conditions"][key] = {"status": "error", "model": MODELS[key], "error": "Shared retrieval: " + str(exc), "calls": []}
                checkpoint(report, destination)
            continue
        checkpoint(report, destination)
        for key in order:
            g = models[key]
            g.generator.calls.clear()
            started = time.perf_counter()
            outcome = {"model": MODELS[key]}
            try:
                control.ensure_no_other(MODELS[key])
                result = g.answer(case["query"], copy.deepcopy(hits), expected_digest=digests[key])
                validate_grounded_result(result, hits, expected_digest=digests[key], expected_model=MODELS[key])
                outcome.update(status=result["status"], full_result=result,
                               resident_snapshot=control.ensure_no_other(MODELS[key]))
            except Exception as exc:
                outcome.update(status="error", error=str(exc))
            finally:
                outcome.update(wall_seconds=time.perf_counter()-started, calls=copy.deepcopy(g.generator.calls))
                try:
                    outcome["release"] = control.unload(MODELS[key])
                except Exception as exc:
                    outcome.update(status="error", cleanup_error=str(exc))
            row["conditions"][key] = outcome
            checkpoint(report, destination)
            print(case["id"], key, outcome["status"], flush=True)
    report["run_status"] = "error" if any(o["status"] == "error" for c in report["cases"] for o in c["conditions"].values()) else "completed"
    checkpoint(report, destination)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX_PATH)
    parser.add_argument("--queries", type=Path, default=Path("evaluation/reranker_validation_queries.json"))
    parser.add_argument("--reverse", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    queries = _json(validate_project_path(args.queries).read_text(encoding="utf-8"))
    result = run(load_index(validate_project_path(args.index)), queries, args.output, reverse=args.reverse)
    return 1 if result["run_status"] == "error" else 0


if __name__ == "__main__":
    sys.exit(main())
