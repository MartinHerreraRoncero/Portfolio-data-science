"""Búsqueda semántica exhaustiva mediante similitud coseno sobre fragmentos indexados."""

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import sys

from rag_pensiones.chunking import ChunkRecord
from rag_pensiones.embeddings import (
    DIMENSION,
    DOCUMENT_PREFIX,
    MODEL_NAME,
    OllamaEmbedder,
    QUERY_PREFIX,
    SCHEMA_VERSION,
    validate_chunk_record,
)
from rag_pensiones.ingest import PROJECT_ROOT, validate_project_path
from rag_pensiones.qwen_embeddings import QwenEmbedder, MODEL as QWEN_MODEL, validate_index as validate_qwen_index

LEGACY_INDEX_PATH = Path("data/processed/airef_2025_embeddings.json")
DEFAULT_INDEX_PATH = Path("data/processed/airef_2025_qwen3_4b_embeddings.json")


@dataclass(frozen=True)
class SearchHit:
    rank: int
    score: float
    chunk: ChunkRecord


def _reject_constant(val: str) -> None:
    raise ValueError(f"Invalid constant {val!r} found in JSON index")


def validate_index(index: dict) -> list[tuple[ChunkRecord, list[float]]]:
    """Valida exhaustivamente el esquema, metadatos y vectores del índice de embeddings."""
    if isinstance(index, dict) and index.get("model") == QWEN_MODEL:
        return validate_qwen_index(index)
    if not isinstance(index, dict):
        raise ValueError(f"Index must be a JSON object (dict), got {type(index).__name__}")

    if index.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"Invalid schema_version: expected {SCHEMA_VERSION!r}, got {index.get('schema_version')!r}"
        )

    if index.get("model") != MODEL_NAME:
        raise ValueError(f"Invalid model: expected {MODEL_NAME!r}, got {index.get('model')!r}")

    dim = index.get("dimension")
    if isinstance(dim, bool) or not isinstance(dim, int) or dim != DIMENSION:
        raise ValueError(f"Invalid dimension: expected integer {DIMENSION}, got {dim!r}")

    if index.get("document_prefix") != DOCUMENT_PREFIX:
        raise ValueError(
            f"Invalid document_prefix: expected {DOCUMENT_PREFIX!r}, got {index.get('document_prefix')!r}"
        )

    if index.get("query_prefix") != QUERY_PREFIX:
        raise ValueError(
            f"Invalid query_prefix: expected {QUERY_PREFIX!r}, got {index.get('query_prefix')!r}"
        )

    digest = index.get("model_digest")
    if not isinstance(digest, str) or not digest.strip():
        raise ValueError(f"model_digest must be a non-empty string, got {digest!r}")

    input_sha256 = index.get("input_sha256")
    if not isinstance(input_sha256, str) or not input_sha256.strip():
        raise ValueError(f"input_sha256 must be a non-empty string, got {input_sha256!r}")

    chunks = index.get("chunks")
    if not isinstance(chunks, list) or len(chunks) == 0:
        raise ValueError("chunks must be a non-empty list")

    seen_ids: set[str] = set()
    doc_registry: dict[str, tuple[str, str]] = {}
    validated_chunks: list[tuple[ChunkRecord, list[float]]] = []

    required_keys = {
        "chunk_id",
        "doc_id",
        "source_file",
        "published_at",
        "page_number",
        "start_char",
        "end_char",
        "text",
        "strategy",
        "embedding",
    }

    for idx, c in enumerate(chunks):
        if not isinstance(c, dict):
            raise ValueError(f"Chunk at index {idx} must be a dict, got {type(c).__name__}")

        missing = required_keys - set(c.keys())
        if missing:
            raise ValueError(f"Chunk at index {idx} missing required fields: {sorted(missing)}")

        chunk_record = ChunkRecord(
            chunk_id=c["chunk_id"],
            doc_id=c["doc_id"],
            source_file=c["source_file"],
            published_at=c["published_at"],
            page_number=c["page_number"],
            start_char=c["start_char"],
            end_char=c["end_char"],
            text=c["text"],
            strategy=c["strategy"],
        )
        validate_chunk_record(chunk_record, idx)

        if chunk_record.chunk_id in seen_ids:
            raise ValueError(f"Duplicate chunk_id detected at index {idx}: {chunk_record.chunk_id}")
        seen_ids.add(chunk_record.chunk_id)

        doc_meta = (chunk_record.source_file, chunk_record.published_at)
        if chunk_record.doc_id in doc_registry:
            if doc_registry[chunk_record.doc_id] != doc_meta:
                raise ValueError(
                    f"Inconsistent metadata for doc_id {chunk_record.doc_id!r} at chunk index {idx}: "
                    f"expected {doc_registry[chunk_record.doc_id]}, got {doc_meta}"
                )
        else:
            doc_registry[chunk_record.doc_id] = doc_meta

        vec = c["embedding"]
        if not isinstance(vec, list):
            raise ValueError(f"embedding at chunk index {idx} must be a list, got {type(vec).__name__}")
        if len(vec) != DIMENSION:
            raise ValueError(
                f"embedding at chunk index {idx} has length {len(vec)}, expected {DIMENSION}"
            )

        for elem_idx, val in enumerate(vec):
            if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(val):
                raise ValueError(
                    f"Non-finite, non-numeric, or boolean value at chunk {idx}, element {elem_idx}: {val!r}"
                )

        norm = math.hypot(*vec)
        if norm <= 0.0 or not math.isfinite(norm):
            raise ValueError(f"Vector at chunk index {idx} has invalid norm: {norm}")

        validated_chunks.append((chunk_record, vec))

    return validated_chunks


def load_index(path: Path) -> dict:
    """Carga y valida un archivo de índice JSON respetando los límites del proyecto."""
    resolved_path = validate_project_path(Path(path))
    if not resolved_path.is_file():
        raise FileNotFoundError(f"Index file not found: {resolved_path}")

    raw_bytes = resolved_path.read_bytes()
    if not raw_bytes.strip():
        raise ValueError("Index file is empty")

    try:
        text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"Index file is not valid UTF-8: {exc}") from exc

    try:
        data = json.loads(text, parse_constant=_reject_constant)
    except Exception as exc:
        raise ValueError(f"Malformed JSON in index file: {exc}") from exc

    validate_index(data)
    return data


def search(
    index: dict,
    query: str,
    top_k: int = 3,
    client: OllamaEmbedder | None = None,
) -> list[SearchHit]:
    """Ejecuta una búsqueda semántica determinista mediante similitud coseno exhaustiva."""
    if not isinstance(query, str):
        raise ValueError(f"query must be a string, got {type(query).__name__}")
    cleaned_query = query.strip()
    if not cleaned_query:
        raise ValueError("query cannot be empty or contain only whitespace")

    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError(f"top_k must be a positive integer (not bool), got {top_k!r}")

    # Validar completamente el índice ANTES de contactar con Ollama
    validated_chunks = validate_index(index)

    if client is None:
        client = QwenEmbedder() if index["model"] == QWEN_MODEL else OllamaEmbedder()

    expected_digest = index["model_digest"].strip()
    digest_before = client.get_digest()
    if digest_before != expected_digest:
        raise RuntimeError(
            f"Model digest mismatch before embedding: index specifies {expected_digest!r}, "
            f"but Ollama returned {digest_before!r}. Aborting search."
        )

    prefixed_query = f"{index['query_prefix']}{cleaned_query}"
    query_vectors = client.embed_texts([prefixed_query])

    digest_after = client.get_digest()
    if digest_after != expected_digest or digest_after != digest_before:
        raise RuntimeError(
            f"Model digest changed during embedding: started with {digest_before!r}, "
            f"became {digest_after!r}. Aborting search."
        )

    if not isinstance(query_vectors, list) or len(query_vectors) != 1:
        raise ValueError(
            f"Expected client to return a list of 1 embedding, got "
            f"{len(query_vectors) if isinstance(query_vectors, list) else type(query_vectors).__name__}"
        )

    q_vec = query_vectors[0]
    dimension = index["dimension"]
    if not isinstance(q_vec, list) or len(q_vec) != dimension:
        raise ValueError(
            f"Query embedding must have dimension {dimension}, got "
            f"{len(q_vec) if isinstance(q_vec, list) else type(q_vec).__name__}"
        )

    for elem_idx, val in enumerate(q_vec):
        if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(val):
            raise ValueError(
                f"Non-finite, non-numeric, or boolean value in query vector at index {elem_idx}: {val!r}"
            )

    norm_q = math.hypot(*q_vec)
    if norm_q <= 0.0 or not math.isfinite(norm_q):
        raise ValueError(f"Query vector has invalid norm: {norm_q}")

    q_norm = [float(x) / norm_q for x in q_vec]

    scored_candidates: list[tuple[int, float, ChunkRecord]] = []
    for orig_idx, (chunk_rec, doc_vec) in enumerate(validated_chunks):
        norm_d = math.hypot(*doc_vec)
        if norm_d <= 0.0 or not math.isfinite(norm_d):
            raise ValueError(f"Document vector at index {orig_idx} has invalid norm: {norm_d}")
        d_norm = [float(x) / norm_d for x in doc_vec]
        dot = math.fsum(a * b for a, b in zip(q_norm, d_norm))
        if not math.isfinite(dot):
            raise ValueError(f"Non-finite similarity score for chunk index {orig_idx}: {dot}")
        score = max(-1.0, min(1.0, dot))
        scored_candidates.append((orig_idx, score, chunk_rec))

    # Orden descendente por score; desempate estable por posición original (menor orig_idx primero)
    scored_candidates.sort(key=lambda item: (-item[1], item[0]))
    selected = scored_candidates[:top_k]

    hits = [
        SearchHit(rank=rank, score=score, chunk=chunk_rec)
        for rank, (_, score, chunk_rec) in enumerate(selected, start=1)
    ]
    return hits


def main(argv: list[str] | None = None) -> int:
    """Punto de entrada de la CLI de búsqueda semántica."""
    parser = argparse.ArgumentParser(
        description="Búsqueda semántica exhaustiva mediante similitud coseno sobre fragmentos indexados."
    )
    parser.add_argument("--query", required=True, type=str, help="Texto de la consulta.")
    parser.add_argument("--top-k", dest="top_k", type=int, default=3, help="Número de resultados principales.")
    parser.add_argument(
        "--index",
        type=Path,
        default=DEFAULT_INDEX_PATH,
        help="Ruta al archivo JSON con el índice de embeddings.",
    )
    parser.add_argument("--json", action="store_true", help="Emitir resultados en formato JSON UTF-8.")

    try:
        args = parser.parse_args(argv)

        if not isinstance(args.query, str) or not args.query.strip():
            sys.stderr.write("Error: query cannot be empty or contain only whitespace\n")
            return 1

        if isinstance(args.top_k, bool) or not isinstance(args.top_k, int) or args.top_k <= 0:
            sys.stderr.write(f"Error: top-k must be a positive integer, got {args.top_k!r}\n")
            return 1

        index_path = validate_project_path(args.index)
        index_data = load_index(index_path)

        hits = search(
            index=index_data,
            query=args.query,
            top_k=args.top_k,
            client=None,
        )

        if args.json:
            out_obj = {
                "query": args.query.strip(),
                "model": index_data["model"],
                "model_digest": index_data["model_digest"],
                "corpus_size": len(index_data["chunks"]),
                "top_k_requested": args.top_k,
                "hits": [asdict(h) for h in hits],
            }
            sys.stdout.write(json.dumps(out_obj, ensure_ascii=False, indent=2) + "\n")
        else:
            lines = [
                f"Consulta: {args.query.strip()}",
                f"Modelo: {index_data['model']}",
                f"Resultados recuperados: {len(hits)} (de {len(index_data['chunks'])} fragmentos indexados, solicitados top-k: {args.top_k})",
                "Nota: La similitud coseno mide proximidad semántica en el espacio vectorial; no representa una probabilidad ni garantiza que el fragmento contenga la respuesta o sea factual.",
                "",
            ]
            for hit in hits:
                lines.append(f"[Resultado {hit.rank}]")
                lines.append(f"  Puntuación (similitud coseno): {hit.score:.6f}")
                lines.append(f"  Documento: {hit.chunk.source_file}")
                lines.append(f"  Fecha de publicación: {hit.chunk.published_at}")
                lines.append(f"  Página: {hit.chunk.page_number}")
                lines.append(f"  ID de fragmento: {hit.chunk.chunk_id}")
                lines.append("  Texto del fragmento:")
                lines.append(hit.chunk.text)
                lines.append("")
            sys.stdout.write("\n".join(lines).rstrip() + "\n")

        return 0
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 1
    except Exception as exc:
        sys.stderr.write(f"Error: {exc}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
