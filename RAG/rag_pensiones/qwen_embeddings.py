"""Active Qwen4B embeddings; model-specific validation without experiment dependencies.

The transport/normalization follows the previously audited isolated comparison.
Legacy Gemma support remains confined to explicit historical indices.
"""
import copy
import json
import math
import time
import urllib.request

from rag_pensiones.chunking import ChunkRecord
from rag_pensiones.embeddings import OllamaEmbedder, validate_chunk_record


def _reject_constant(value):
    raise ValueError("Nonfinite JSON constant: " + value)


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


def validate_index(index):
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
    records, ids, identities = [], set(), {}
    for item in index["chunks"]:
        if not isinstance(item, dict) or set(item) != set(ChunkRecord.__dataclass_fields__) | {"embedding"}:
            raise ValueError("Candidate chunk fields invalid")
        chunk = ChunkRecord(**{k: v for k, v in item.items() if k != "embedding"})
        validate_chunk_record(chunk)
        identity = (chunk.source_file, chunk.published_at)
        if chunk.doc_id in identities and identities[chunk.doc_id] != identity:
            raise ValueError("Inconsistent document identity")
        identities[chunk.doc_id] = identity
        if chunk.chunk_id in ids:
            raise ValueError("Duplicate candidate chunk")
        ids.add(chunk.chunk_id)
        v = vector(item["embedding"])
        if not math.isclose(math.hypot(*item["embedding"]), 1.0, abs_tol=1e-5):
            raise ValueError("Candidate vector is not normalized")
        records.append((chunk, v))
    return records
