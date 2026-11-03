import copy
import math
import unittest
from unittest.mock import patch

from experiments.hybrid_search import bm25_scores, fuse_rankings, hybrid_search, tokenize
from rag_pensiones.search import SearchHit
from test_compare_retrieval_k import _make_chunk, _make_valid_index


class HybridSearchTests(unittest.TestCase):
    def test_tokenization_preserves_negation_year_decimal_and_accents(self):
        self.assertEqual(tokenize("¿Qué PENSIÓN no contributiva en 2050: 16,1 y 16.1 %?"),
                         ["pension", "no", "contributiva", "2050", "16.1", "16.1"])

    def test_bm25_manual_idf_tf_length_and_unique_query_terms(self):
        # N=3, df(pension)=2, lengths 2,1,1 -> average 4/3.
        scores = bm25_scores(["pension pension", "pension", "credito"], "pension pension")
        idf = math.log1p(1.5 / 2.5)
        self.assertAlmostEqual(scores[0], idf * 2 * 2.2 / (2 + 1.2 * (.25 + .75 * 2 / (4 / 3))))
        self.assertAlmostEqual(scores[1], idf * 2.2 / (1 + 1.2 * (.25 + .75 / (4 / 3))))
        self.assertEqual(scores[2], 0)
        self.assertEqual(scores, bm25_scores(["pension pension", "pension", "credito"], "pension"))

    def test_empty_and_out_of_vocabulary(self):
        self.assertEqual(bm25_scores([], "pension"), [])
        for texts, q in [(["", "de la"], "pension"), (["pension"], "credito"), (["pension"], "de la")]:
            self.assertEqual(bm25_scores(texts, q), [0.] * len(texts))

    def test_numeric_and_type_boundaries(self):
        for kwargs in [{"k1": v} for v in [True, 0, -1, float("nan"), float("inf")]] + [{"b": v} for v in [True, -1, 2, float("nan"), float("inf")]]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                bm25_scores(["pension pension pension"], "pension", **kwargs)
        for kwargs in ({"k1": 1e308},):
            with self.assertRaises(ValueError):
                bm25_scores(["pension pension pension pension pension pension pension pension"], "pension", **kwargs)
        for constant in [True, 0, -1, 1.2, float("nan")]:
            with self.assertRaises(ValueError): fuse_rankings([], "pension", constant)
        for k in [True, 0, -1, 1.5]:
            with self.assertRaises(ValueError): hybrid_search({}, "pension", k)

    def test_rrf_manual_and_deterministic_tie_break(self):
        dense = [SearchHit(1, .9, _make_chunk(1, "credito")), SearchHit(2, .8, _make_chunk(2, "pension"))]
        before = copy.deepcopy(dense)
        fused = fuse_rankings(dense, "pension")
        self.assertEqual(fused[0].chunk.chunk_id, "chunk_id_2")
        self.assertAlmostEqual(fused[0].score, 1 / 62 + 1 / 61)
        self.assertAlmostEqual(fused[1].score, 1 / 61)
        self.assertEqual(dense, before)
        # Identical lexical scores tie by chunk ID, not incoming order.
        dense = [SearchHit(1, .9, _make_chunk(2, "pension")), SearchHit(2, .8, _make_chunk(1, "pension"))]
        self.assertEqual([h.chunk.chunk_id for h in fuse_rankings(dense, "pension")], ["chunk_id_2", "chunk_id_1"])

    def test_no_matches_fallback_is_exact_and_empty_is_safe(self):
        dense = [SearchHit(1, -.1, _make_chunk(1, "pension"))]
        self.assertEqual(fuse_rankings(dense, "credito"), dense)
        self.assertEqual(fuse_rankings([], "pension"), [])

    def test_bad_ranks_duplicate_ids_and_nonfinite_scores_rejected(self):
        h = SearchHit(1, .5, _make_chunk(1))
        for dense in [[h, SearchHit(2, .4, h.chunk)], [SearchHit(2, .5, h.chunk)], [SearchHit(1, float("nan"), h.chunk)]]:
            with self.assertRaises(ValueError): fuse_rankings(dense, "pension")

    def test_wrapper_uses_all_candidates_once(self):
        idx = _make_valid_index(6)
        dense = [SearchHit(i, 1 / i, _make_chunk(i)) for i in range(1, 7)]
        with patch("experiments.hybrid_search.search", return_value=dense) as search:
            self.assertEqual(len(hybrid_search(idx, "pension", 4)), 4)
            self.assertEqual(search.call_args.kwargs["top_k"], 6)
            self.assertEqual(search.call_count, 1)
