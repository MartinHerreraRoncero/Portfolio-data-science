"""Embedding swap invariants and failure isolation; no real network or models."""
import copy
import json
import math
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from experiments import compare_embeddings as trial
from rag_pensiones import embeddings as baseline


def axis(i=0):
    values = [0.0] * trial.DIMENSION
    values[i] = 1.0
    return values


def base_index():
    return {"schema_version": baseline.SCHEMA_VERSION, "model": baseline.MODEL_NAME,
            "dimension": baseline.DIMENSION, "model_digest": "base-digest",
            "document_prefix": baseline.DOCUMENT_PREFIX, "query_prefix": baseline.QUERY_PREFIX,
            "input_sha256": "source-hash", "chunks": [
                {"chunk_id": f"c{i}", "doc_id": "public", "source_file": "public.pdf",
                 "published_at": "2025-03-31", "page_number": i+1, "start_char": 0,
                 "end_char": 30, "text": f"Texto publico {i}", "strategy": "test",
                 "embedding": [1.0] + [0.0]*(baseline.DIMENSION-1)} for i in range(2)]}


class FakeEmbedder:
    def __init__(self, values=None, digests=None):
        self.values = values
        self.digests = list(digests or ["qwen-digest"])
        self.calls = []
        self.inputs = []

    def get_digest(self):
        return self.digests.pop(0) if len(self.digests) > 1 else self.digests[0]

    def embed_texts(self, texts):
        self.inputs.append(texts)
        return copy.deepcopy(self.values if self.values is not None else [axis() for _ in texts])


class EmbeddingComparisonTests(unittest.TestCase):
    def test_payload_dimension_model_and_no_truncation(self):
        client = trial.QwenEmbedder()
        raw = json.dumps({"model": trial.MODEL, "embeddings": [axis()]}).encode()
        with patch.object(client, "_open_request", return_value=raw) as transport:
            self.assertEqual(client.embed_texts(["documento publico"]), [axis()])
        request = transport.call_args.args[0]
        payload = json.loads(request.data)
        self.assertEqual(payload["model"], trial.MODEL)
        self.assertEqual(payload["dimensions"], 2560)
        self.assertEqual(payload["input"], ["documento publico"])
        self.assertIs(payload["truncate"], False)
        self.assertEqual(request.full_url, "http://127.0.0.1:11434/api/embed")

    def test_invalid_input_is_rejected_before_network(self):
        client = trial.QwenEmbedder()
        with patch.object(client, "_open_request") as transport:
            for bad in ["text", [None], [""], ["  "]]:
                with self.subTest(bad=bad), self.assertRaises(ValueError):
                    client.embed_texts(bad)
            self.assertEqual(client.embed_texts([]), [])
        transport.assert_not_called()

    def test_bad_api_results_and_overflow_still_leave_serializable_diagnostics(self):
        bad_payloads = [b'{"model":"other","embeddings":[]}',
                        json.dumps({"model": trial.MODEL, "embeddings": []}).encode(),
                        json.dumps({"model": trial.MODEL, "embeddings": [[0.0]*2560]}).encode(),
                        b'{"model":"qwen3-embedding:4b","embeddings":[[NaN]]}',
                        b'{"model":"qwen3-embedding:4b","embeddings":[[1e999]]}', b'[]']
        for raw in bad_payloads:
            client = trial.QwenEmbedder()
            with self.subTest(raw=raw[:60]), patch.object(client, "_open_request", return_value=raw):
                with self.assertRaises(ValueError):
                    client.embed_texts(["texto"])
                json.dumps(client.calls, allow_nan=False)
                self.assertIn("error", client.calls[0])

    def test_vectors_reject_nonfinite_boolean_zero_wrong_dimension_and_overflow(self):
        for value in [True, float("nan"), float("inf"), "1", 10**400]:
            bad = axis(); bad[0] = value
            with self.subTest(value=str(value)[:20]), self.assertRaises(ValueError):
                trial.vector(bad)
        for bad in [[], [1.0]*768, [0.0]*2560, (1.0,)*2560]:
            with self.subTest(length=len(bad)), self.assertRaises(ValueError):
                trial.vector(bad)
        self.assertAlmostEqual(math.hypot(*trial.vector([3.0,4.0]+[0.0]*2558)), 1.0)

    def test_transport_errors_are_retained(self):
        client = trial.QwenEmbedder()
        with patch.object(client, "_open_request", side_effect=ConnectionError("offline")):
            with self.assertRaises(ConnectionError):
                client.embed_texts(["documento"])
        self.assertEqual(client.calls[0]["error"], "offline")

    def test_local_only_endpoint_and_exact_digest_tags(self):
        with self.assertRaises(ValueError):
            trial.QwenEmbedder(base_url="https://external.invalid")
        client = trial.QwenEmbedder()
        for tags in [{"models": []}, {"models": [{"name": trial.MODEL, "digest": ""}]}, []]:
            with self.subTest(tags=tags), patch.object(client, "_open_request", return_value=json.dumps(tags).encode()):
                with self.assertRaises(ValueError):
                    client.get_digest()
        with patch.object(client, "_open_request", return_value=json.dumps({"models": [{"model": trial.MODEL, "digest": " pinned "}]}).encode()):
            self.assertEqual(client.get_digest(), "pinned")

    def test_build_keeps_source_identity_and_uses_plain_documents(self):
        base = base_index(); original = copy.deepcopy(base); client = FakeEmbedder()
        candidate = trial.build_candidate_index(base, client)
        self.assertEqual(base, original)
        self.assertEqual(client.inputs, [[c["text"] for c in base["chunks"]]])
        for b, c in zip(base["chunks"], candidate["chunks"]):
            self.assertEqual({k:v for k,v in b.items() if k!='embedding'}, {k:v for k,v in c.items() if k!='embedding'})
        trial.validate_candidate(candidate, base)

    def test_changed_chunks_duplicate_metadata_and_wrong_prefix_are_rejected(self):
        base = base_index(); valid = trial.build_candidate_index(base, FakeEmbedder())
        for mode in ["text", "order", "duplicate", "prefix", "model", "base_hash", "dimension", "unnormalized"]:
            candidate = copy.deepcopy(valid)
            if mode == "text": candidate["chunks"][0]["text"] = "alterado"
            elif mode == "order": candidate["chunks"].reverse()
            elif mode == "duplicate": candidate["chunks"][1] = copy.deepcopy(candidate["chunks"][0])
            elif mode == "prefix": candidate["query_prefix"] = baseline.QUERY_PREFIX
            elif mode == "model": candidate["model"] = baseline.MODEL_NAME
            elif mode == "base_hash": candidate["base_index_hash"] = "other"
            elif mode == "dimension": candidate["dimension"] = 768
            else: candidate["chunks"][0]["embedding"][0] = 2.0
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                trial.validate_candidate(candidate, base)

    def test_digest_changes_abort_indexing_and_query(self):
        base = base_index(); candidate = trial.build_candidate_index(base, FakeEmbedder())
        with self.assertRaisesRegex(RuntimeError, "changed"):
            trial.build_candidate_index(base, FakeEmbedder(digests=["a", "b"]))
        for digests in [["wrong"], ["qwen-digest", "changed"]]:
            with self.subTest(digests=digests), self.assertRaises(RuntimeError):
                trial.rank_candidate(candidate, "pregunta", FakeEmbedder(digests=digests))

    def test_cosine_ranking_instruction_and_stable_ties(self):
        candidate = trial.build_candidate_index(base_index(), FakeEmbedder(values=[axis(0), axis(1)]))
        client = FakeEmbedder(values=[axis(1)])
        hits = trial.rank_candidate(candidate, "  pregunta publica  ", client, k=2)
        self.assertEqual([h.chunk.chunk_id for h in hits], ["c1", "c0"])
        self.assertEqual([h.score for h in hits], [1.0, 0.0])
        self.assertEqual(client.inputs, [[trial.QUERY_PREFIX + "pregunta publica"]])
        tied = trial.build_candidate_index(base_index(), FakeEmbedder())
        self.assertEqual([h.chunk.chunk_id for h in trial.rank_candidate(tied, "q", FakeEmbedder(), 2)], ["c0", "c1"])
        for k in [True, 0, -1, 1.5]:
            with self.subTest(k=k), self.assertRaises(ValueError):
                trial.rank_candidate(tied, "q", client, k)

    def test_condition_failure_is_not_abstention_and_does_not_skip_other_arm(self):
        base = base_index(); candidate = trial.build_candidate_index(base, FakeEmbedder())
        queries = {"schema_version":"reranker-validation-queries-v1", "cases":[{"id":f"q{i}","query":"publica"} for i in range(2)]}
        generator = SimpleNamespace(calls=[], get_digest=lambda:"generator")
        model = SimpleNamespace(generator=generator, answer=lambda *args,**kwargs:{"status":"insufficient_evidence","claims":[]})
        embedders = {"baseline":FakeEmbedder(), "qwen4b":FakeEmbedder()}
        with patch.object(trial, "search", side_effect=ConnectionError("retrieval failed")), patch.object(trial, "rank_candidate", return_value=[]), patch.object(trial, "validate_grounded_result"):
            result = trial.run(base,candidate,queries,None,embedders=embedders,models={"baseline":model,"qwen4b":model})
        self.assertEqual(result["run_status"], "error")
        self.assertEqual([c["condition_order"] for c in result["cases"]], [["baseline","qwen4b"],["qwen4b","baseline"]])
        for case in result["cases"]:
            self.assertEqual(case["conditions"]["baseline"]["status"], "error")
            self.assertEqual(case["conditions"]["baseline"]["retrieval"]["status"], "error")
            self.assertEqual(case["conditions"]["qwen4b"]["status"], "insufficient_evidence")


if __name__ == "__main__":
    unittest.main()
