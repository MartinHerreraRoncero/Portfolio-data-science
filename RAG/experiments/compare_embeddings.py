"""Isolated paired embedding trial. Gold and semantic reviews are never read here."""
import argparse
import copy
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys
import time
import urllib.request

from experiments.source_bound_trial import RecordedGenerator, checkpoint
from experiments.validate_reranker import canonical_hash, output_path, validate_queries
from rag_pensiones.answer import DEFAULT_GENERATION_MODEL, _reject_constant
from rag_pensiones.chunking import ChunkRecord
from rag_pensiones.embeddings import OllamaEmbedder, validate_chunk_record
from rag_pensiones.grounding import OllamaGrounder, validate_grounded_result
from rag_pensiones.ingest import validate_project_path
from rag_pensiones.search import DEFAULT_INDEX_PATH, SearchHit, load_index, search, validate_index

MODEL = "qwen3-embedding:4b"
DIMENSION = 2560
QUERY_PREFIX = "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: "
SCHEMA = "embedding-comparison-index-v1"


def _json(raw):
    def finite_float(text):
        value = float(text)
        if not math.isfinite(value):
            raise ValueError("JSON float overflow")
        return value
    return json.loads(raw, parse_constant=_reject_constant, parse_float=finite_float)


def vector(value):
    """Validate a full-size vector; return its L2 normalization without mutating it."""
    if not isinstance(value, list) or len(value) != DIMENSION:
        raise ValueError("Expected2560embedding components")
    try:
        if any(isinstance(x, bool) or not isinstance(x, (float, int)) or not math.isfinite(x) for x in value):
            raise ValueError("Nonfinite or nonnumeric embedding")
        norm = math.hypot(*value)
    except OverflowError as exc:
        raise ValueError("Embedding numeric overflow") from exc
    if not math.isfinite(norm) or norm <= 0:
        raise ValueError("Embedding norm must be positive and finite")
    return [float(x) / norm for x in value]


class QwenEmbedder(OllamaEmbedder):
    """Local-only transport inherited from baseline; model and dimensions stay explicit."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.calls = []

    def get_digest(self):
        data = _json(self._open_request(urllib.request.Request(self.base_url + "/api/tags")))
        if not isinstance(data, dict) or not isinstance(data.get("models"), list):
            raise ValueError("Malformed model tags")
        for item in data["models"]:
            if isinstance(item, dict) and MODEL in (item.get("name"), item.get("model")):
                digest = item.get("digest")
                if isinstance(digest, str) and digest.strip():
                    return digest.strip()
                raise ValueError("Empty Qwen embedding digest")
        raise ValueError("Qwen embedding4B is not installed")

    def embed_texts(self, texts):
        if not isinstance(texts, list) or any(not isinstance(x, str) or not x.strip() for x in texts):
            raise ValueError("Expected list of nonempty texts")
        if not texts:
            return []
        payload = {"model": MODEL, "input": texts, "truncate": False,
                   "dimensions": DIMENSION, "keep_alive": "5m"}
        call = {"request": copy.deepcopy(payload)}
        self.calls.append(call)
        started = time.perf_counter()
        try:
            req = urllib.request.Request(self.base_url + "/api/embed", data=json.dumps(payload).encode(),
                                         headers={"Content-Type": "application/json"})
            data = _json(self._open_request(req))
            call["response"] = copy.deepcopy(data)
            if not isinstance(data, dict) or data.get("model") != MODEL:
                raise ValueError("Embedding response model mismatch")
            values = data.get("embeddings")
            if not isinstance(values, list) or len(values) != len(texts):
                raise ValueError("Embedding count mismatch")
            return [vector(v) for v in values]
        except Exception as exc:
            call["error"] = str(exc)
            raise
        finally:
            call["wall_seconds"] = time.perf_counter() - started


class RecordedBaseline(OllamaEmbedder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.calls = []

    def embed_texts(self, texts):
        call = {"input": copy.deepcopy(texts)}
        self.calls.append(call)
        started = time.perf_counter()
        try:
            values = super().embed_texts(texts)
            call["normalized_embeddings"] = copy.deepcopy(values)
            return values
        except Exception as exc:
            call["error"] = str(exc)
            raise
        finally:
            call["wall_seconds"] = time.perf_counter() - started


def validate_candidate(index, baseline=None):
    fields = {"schema_version", "model", "dimension", "model_digest", "document_prefix",
              "query_prefix", "base_index_hash", "chunks"}
    if not isinstance(index, dict) or set(index) != fields:
        raise ValueError("Candidate index fields invalid")
    if index["schema_version"] != SCHEMA or index["model"] != MODEL:
        raise ValueError("Candidate model/schema invalid")
    if type(index["dimension"]) is not int or index["dimension"] != DIMENSION:
        raise ValueError("Candidate dimension invalid")
    if index["document_prefix"] != "" or index["query_prefix"] != QUERY_PREFIX:
        raise ValueError("Candidate input prefixes invalid")
    if any(not isinstance(index[k], str) or not index[k].strip() for k in ["model_digest", "base_index_hash"]):
        raise ValueError("Candidate provenance missing")
    if not isinstance(index["chunks"], list) or not index["chunks"]:
        raise ValueError("Candidate chunks missing")
    records, ids = [], set()
    for item in index["chunks"]:
        if not isinstance(item, dict) or set(item) != set(ChunkRecord.__dataclass_fields__) | {"embedding"}:
            raise ValueError("Candidate chunk fields invalid")
        chunk = ChunkRecord(**{k: v for k, v in item.items() if k != "embedding"})
        validate_chunk_record(chunk)
        if chunk.chunk_id in ids:
            raise ValueError("Duplicate candidate chunk")
        ids.add(chunk.chunk_id)
        v = vector(item["embedding"])
        if not math.isclose(math.hypot(*item["embedding"]), 1.0, abs_tol=1e-5):
            raise ValueError("Candidate vector is not normalized")
        records.append((chunk, v))
    if baseline is not None:
        original = validate_index(baseline)
        if index["base_index_hash"] != canonical_hash(baseline):
            raise ValueError("Baseline index provenance mismatch")
        if canonical_hash([asdict(c) for c, _ in original]) != canonical_hash([asdict(c) for c, _ in records]):
            raise ValueError("Candidate changed original chunks or order")
    return records


def build_candidate_index(index, client):
    records = validate_index(index)
    digest = client.get_digest()
    values = client.embed_texts([chunk.text for chunk, _ in records])
    if client.get_digest() != digest:
        raise RuntimeError("Qwen embedding digest changed while indexing")
    if not isinstance(values, list) or len(values) != len(records):
        raise ValueError("Candidate embedding count invalid")
    candidate = {"schema_version": SCHEMA, "model": MODEL, "dimension": DIMENSION,
                 "model_digest": digest, "document_prefix": "", "query_prefix": QUERY_PREFIX,
                 "base_index_hash": canonical_hash(index),
                 "chunks": [dict(asdict(c), embedding=vector(v)) for (c, _), v in zip(records, values)]}
    validate_candidate(candidate, index)
    return candidate


def rank_candidate(index, query, client, k=4):
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Query must be nonempty text")
    if type(k) is not int or k <= 0:
        raise ValueError("k must be a positive integer")
    records = validate_candidate(index)
    digest = index["model_digest"]
    if client.get_digest() != digest:
        raise RuntimeError("Qwen embedding digest mismatch before query")
    values = client.embed_texts([QUERY_PREFIX + query.strip()])
    if client.get_digest() != digest:
        raise RuntimeError("Qwen embedding digest changed during query")
    if not isinstance(values, list) or len(values) != 1:
        raise ValueError("Expected one query embedding")
    q = vector(values[0])
    scored = [(i, max(-1.0, min(1.0, math.fsum(a*b for a, b in zip(q, v)))), c)
              for i, (c, v) in enumerate(records)]
    scored.sort(key=lambda row: (-row[1], row[0]))
    return [SearchHit(rank=rank, score=s, chunk=c) for rank, (_, s, c) in enumerate(scored[:k], 1)]


def run(baseline_index, candidate_index, queries, output, *, embedders=None, models=None):
    validate_queries(queries)
    originals = validate_index(baseline_index)
    validate_candidate(candidate_index, baseline_index)
    destination = output_path(output)
    embedders = embedders if embedders is not None else {"baseline": RecordedBaseline(), "qwen4b": QwenEmbedder(timeout=180)}
    models = models if models is not None else {m: OllamaGrounder(generator=RecordedGenerator()) for m in embedders}
    if set(embedders) != {"baseline", "qwen4b"} or set(models) != set(embedders):
        raise ValueError("Need exactly baseline and qwen4b")
    digest = models["baseline"].generator.get_digest()
    if models["qwen4b"].generator.get_digest() != digest:
        raise RuntimeError("Generator digests differ")
    report = {"schema_version": "embedding-comparison-results-v1", "run_status": "running",
              "metadata": {"model": DEFAULT_GENERATION_MODEL, "digest": digest, "gold_access": "none",
                           "queries_hash": canonical_hash(queries), "base_index_hash": canonical_hash(baseline_index),
                           "candidate_index_hash": canonical_hash(candidate_index), "top_k": 4}, "cases": []}
    checkpoint(report, destination)
    for number, case in enumerate(queries["cases"]):
        order = ["baseline", "qwen4b"] if number % 2 == 0 else ["qwen4b", "baseline"]
        row = dict(case, conditions={}, condition_order=order)
        report["cases"].append(row)
        for method in order:
            g, embedder = models[method], embedders[method]
            g.generator.calls.clear()
            embedder.calls.clear()
            started = time.perf_counter()
            retrieval = {"status": "running"}
            try:
                ranked = search(baseline_index, case["query"], top_k=len(originals), client=embedder) if method == "baseline" else rank_candidate(candidate_index, case["query"], embedder, k=len(originals))
                hits = ranked[:4]
                retrieval.update(status="completed", ranking=[{"rank": h.rank, "score": h.score, "chunk_id": h.chunk.chunk_id} for h in ranked],
                                 hits=[{"rank": h.rank, "score": h.score, "chunk": asdict(h.chunk)} for h in hits],
                                 wall_seconds=time.perf_counter()-started)
                result = g.answer(case["query"], copy.deepcopy(hits), expected_digest=digest)
                validate_grounded_result(result, hits, expected_digest=digest, expected_model=DEFAULT_GENERATION_MODEL)
                outcome = {"status": result["status"], "full_result": result}
            except Exception as exc:
                if retrieval["status"] == "running":
                    retrieval.update(status="error", error=str(exc))
                outcome = {"status": "error", "error": str(exc)}
            outcome.update(wall_seconds=time.perf_counter()-started, retrieval=retrieval,
                           embedding_calls=copy.deepcopy(embedder.calls), calls=copy.deepcopy(g.generator.calls))
            row["conditions"][method] = outcome
            checkpoint(report, destination)
            print(case["id"], method, outcome["status"], flush=True)
    report["run_status"] = "error" if any(o["status"] == "error" for c in report["cases"] for o in c["conditions"].values()) else "completed"
    checkpoint(report, destination)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["build", "run"])
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX_PATH)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--queries", type=Path, default=Path("evaluation/reranker_validation_queries.json"))
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    index = load_index(validate_project_path(args.index))
    if args.stage == "build":
        destination, trace_path = output_path(args.output), output_path(args.trace)
        if trace_path is None or destination == trace_path:
            raise ValueError("Build needs a separate fresh trace path")
        client = QwenEmbedder(timeout=180)
        trace = {"status": "running", "base_index_hash": canonical_hash(index)}
        try:
            result = build_candidate_index(index, client)
            checkpoint(result, destination)
            trace.update(status="completed", candidate_index_hash=canonical_hash(result))
        except Exception as exc:
            trace.update(status="error", error=str(exc))
            raise
        finally:
            trace["calls"] = client.calls
            checkpoint(trace, trace_path)
        print("Candidate index built:13original chunks expected; actual", len(result["chunks"]))
        return 0
    if args.candidate is None:
        raise ValueError("Run needs --candidate")
    read = lambda p: _json(validate_project_path(p).read_text(encoding="utf-8"))
    result = run(index, read(args.candidate), read(args.queries), args.output)
    return 1 if result["run_status"] == "error" else 0


if __name__ == "__main__":
    sys.exit(main())
