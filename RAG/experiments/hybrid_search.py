"""BM25 + búsqueda semántica, fusionados por RRF. Solo biblioteca estándar."""

from collections import Counter
import math
import re
import unicodedata

from rag_pensiones.search import SearchHit, search, validate_index

BM25_K1 = 1.2
BM25_B = 0.75
RRF_CONSTANT = 60
STOPWORDS = frozenset("a al algo ante como con cual cuando de del desde donde el ella en entre es esta este estos ha hasta la las lo los mas me mi o para pero por que quien se sin sobre su sus un una unos unas y".split())
TOKEN_PATTERN = re.compile(r"\d+(?:[.,]\d+)?|[^\W\d_]+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    if not isinstance(text, str):
        raise ValueError("text must be a string")
    norm = unicodedata.normalize("NFKD", text.casefold())
    norm = "".join(c for c in norm if not unicodedata.combining(c))
    return [t.replace(",", ".") for t in TOKEN_PATTERN.findall(norm) if t not in STOPWORDS]


def _number(value, name: str, lower: float, upper: float | None = None, strict: bool = False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number, not bool")
    if (value <= lower if strict else value < lower) or (upper is not None and value > upper):
        raise ValueError(f"{name} outside valid range")


def bm25_scores(texts: list[str], query: str, k1: float = BM25_K1, b: float = BM25_B) -> list[float]:
    _number(k1, "k1", 0, strict=True)
    _number(b, "b", 0, 1)
    if not isinstance(texts, list):
        raise ValueError("texts must be a list")
    documents = [Counter(tokenize(t)) for t in texts]
    terms = sorted(set(tokenize(query)))
    n = len(documents)
    lengths = [sum(d.values()) for d in documents]
    avgdl = sum(lengths) / n if n else 0
    if not terms or not avgdl:
        return [0.0] * n
    df = Counter(term for doc in documents for term in doc)
    result = []
    for doc, length in zip(documents, lengths):
        score = math.fsum(
            math.log1p((n - df[t] + 0.5) / (df[t] + 0.5))
            * doc[t] * (k1 + 1) / (doc[t] + k1 * (1 - b + b * length / avgdl))
            for t in terms if doc[t]
        )
        if not math.isfinite(score):
            raise ValueError("Non-finite BM25 score")
        result.append(score)
    return result


def lexical_ranking(dense: list[SearchHit], query: str) -> list[tuple[SearchHit, float]]:
    scores = bm25_scores([h.chunk.text for h in dense], query)
    return sorted(((h, s) for h, s in zip(dense, scores) if s > 0),
                  key=lambda hs: (-hs[1], hs[0].chunk.chunk_id))


def fuse_rankings(dense: list[SearchHit], query: str, rrf_constant: int = RRF_CONSTANT) -> list[SearchHit]:
    if isinstance(rrf_constant, bool) or not isinstance(rrf_constant, int) or rrf_constant <= 0:
        raise ValueError("rrf_constant must be a positive integer, not bool")
    if not isinstance(dense, list) or any(not isinstance(h, SearchHit) for h in dense):
        raise ValueError("dense must be a list of SearchHit")
    if [h.rank for h in dense] != list(range(1, len(dense) + 1)):
        raise ValueError("dense ranks must be consecutive, starting at one")
    ids = [h.chunk.chunk_id for h in dense]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate chunk IDs")
    if any(isinstance(h.score, bool) or not isinstance(h.score, (int, float)) or not math.isfinite(h.score) for h in dense):
        raise ValueError("Dense scores must be finite numbers")
    lexical = lexical_ranking(dense, query)
    if not lexical:
        return list(dense)
    lexical_ranks = {h.chunk.chunk_id: r for r, (h, _) in enumerate(lexical, 1)}
    fused = []
    for h in dense:
        rank = lexical_ranks.get(h.chunk.chunk_id)
        score = 1 / (rrf_constant + h.rank) + (1 / (rrf_constant + rank) if rank else 0)
        fused.append((score, h))
    fused.sort(key=lambda sh: (-sh[0], sh[1].rank, sh[1].chunk.chunk_id))
    return [SearchHit(rank=r, score=s, chunk=h.chunk) for r, (s, h) in enumerate(fused, 1)]


def hybrid_search(index: dict, query: str, top_k: int = 4, client=None) -> list[SearchHit]:
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer, not bool")
    chunks = validate_index(index)
    dense = search(index, query, top_k=len(chunks), client=client)
    return fuse_rankings(dense, query)[:top_k]
