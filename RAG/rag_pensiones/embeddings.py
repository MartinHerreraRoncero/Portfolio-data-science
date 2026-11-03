"""Generación de representaciones vectoriales locales (embeddings) con Ollama y EmbeddingGemma."""

import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import urllib.error
import urllib.request
import uuid

from rag_pensiones.chunking import ChunkRecord
from rag_pensiones.ingest import (
    PROJECT_ROOT,
    validate_project_path,
    validate_published_at,
)

MODEL_NAME = "embeddinggemma:300m"
DIMENSION = 768
DEFAULT_BASE_URL = "http://127.0.0.1:11434"
DEFAULT_TIMEOUT = 60.0
DOCUMENT_PREFIX = "title: none | text: "
QUERY_PREFIX = "task: search result | query: "
SCHEMA_VERSION = "embeddings-v1"


class OllamaEmbedder:
    """Cliente HTTP sin dependencias externas para generar embeddings locales mediante Ollama."""

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT,
        opener: urllib.request.OpenerDirector | None = None,
    ) -> None:
        if base_url != DEFAULT_BASE_URL:
            raise ValueError("Only http://127.0.0.1:11434 is permitted for local embeddings.")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        if opener is not None:
            self._opener = opener
        else:
            # Desactivar proxies del sistema para asegurar comunicación local estricta
            self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def _open_request(self, req: urllib.request.Request) -> bytes:
        try:
            with self._opener.open(req, timeout=self.timeout) as resp:
                return resp.read()
        except urllib.error.HTTPError as exc:
            body = ""
            try:
                body = exc.read().decode("utf-8", errors="replace")
            except Exception:
                pass
            raise RuntimeError(f"HTTP error {exc.code} from Ollama: {exc.reason}. Response: {body}") from exc
        except urllib.error.URLError as exc:
            raise ConnectionError(f"Connection error contacting Ollama at {self.base_url}: {exc.reason}") from exc
        except TimeoutError as exc:
            raise TimeoutError(f"Timeout contacting Ollama after {self.timeout}s") from exc

    def get_digest(self) -> str:
        """Consulta /api/tags para localizar el modelo exacto y su digest no vacío."""
        url = f"{self.base_url}/api/tags"
        req = urllib.request.Request(url, method="GET")
        raw = self._open_request(req)
        try:
            data = json.loads(raw.decode("utf-8"))
        except Exception as exc:
            raise ValueError(f"Malformed JSON from /api/tags: {exc}") from exc

        if not isinstance(data, dict) or "models" not in data or not isinstance(data["models"], list):
            raise ValueError("Invalid response format from /api/tags: 'models' list expected")

        for item in data["models"]:
            if not isinstance(item, dict):
                continue
            name = item.get("name")
            model = item.get("model")
            if name == MODEL_NAME or model == MODEL_NAME:
                digest = item.get("digest")
                if isinstance(digest, str) and digest.strip():
                    return digest.strip()
                raise ValueError(f"Model {MODEL_NAME!r} found in /api/tags but has an empty or invalid digest")

        raise ValueError(f"Model {MODEL_NAME!r} not found in Ollama /api/tags")

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Envía textos a /api/embed, valida tipos/dimensiones y normaliza L2."""
        if not isinstance(texts, list):
            raise ValueError(f"texts must be a list, got {type(texts).__name__}")
        if len(texts) == 0:
            return []
        for idx, text in enumerate(texts):
            if not isinstance(text, str):
                raise ValueError(f"All texts must be str; item at index {idx} is {type(text).__name__}")

        url = f"{self.base_url}/api/embed"
        payload = {
            "model": MODEL_NAME,
            "input": texts,
            "truncate": False,
            "dimensions": DIMENSION,
            "keep_alive": "5m",
        }
        body_bytes = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=body_bytes,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        raw = self._open_request(req)
        try:
            data = json.loads(raw.decode("utf-8"))
        except Exception as exc:
            raise ValueError(f"Malformed JSON from /api/embed: {exc}") from exc

        if not isinstance(data, dict):
            raise ValueError("Response from /api/embed must be a JSON object")

        resp_model = data.get("model")
        if resp_model != MODEL_NAME:
            raise ValueError(f"Response model {resp_model!r} does not match expected {MODEL_NAME!r}")

        embeddings = data.get("embeddings")
        if not isinstance(embeddings, list):
            raise ValueError("Response from /api/embed missing 'embeddings' list")

        if len(embeddings) != len(texts):
            raise ValueError(f"Expected {len(texts)} embeddings, received {len(embeddings)}")

        normalized_vectors: list[list[float]] = []
        for vec_idx, vec in enumerate(embeddings):
            if not isinstance(vec, list):
                raise ValueError(f"Embedding at index {vec_idx} is not a list")
            if len(vec) != DIMENSION:
                raise ValueError(
                    f"Embedding at index {vec_idx} has dimension {len(vec)}, expected {DIMENSION}"
                )
            for elem_idx, val in enumerate(vec):
                if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(val):
                    raise ValueError(
                        f"Non-finite, non-numeric, or boolean value at vector {vec_idx}, element {elem_idx}: {val!r}"
                    )
            norm = math.hypot(*vec)
            if norm <= 0.0 or not math.isfinite(norm):
                raise ValueError(f"Vector at index {vec_idx} has invalid norm: {norm}")
            normalized_vectors.append([float(x) / norm for x in vec])

        return normalized_vectors


def validate_chunk_record(chunk: ChunkRecord, index: int | None = None) -> None:
    """Valida tipos, rangos coherentes y metadatos de un ChunkRecord."""
    loc = f" at index {index}" if index is not None else ""
    if not hasattr(chunk, "chunk_id"):
        raise ValueError(f"Invalid chunk object{loc}: {chunk!r}")

    for str_field in ("chunk_id", "doc_id", "source_file", "strategy"):
        val = getattr(chunk, str_field, None)
        if not isinstance(val, str) or not val.strip():
            raise ValueError(f"Field {str_field!r}{loc} must be a non-empty string, got {val!r}")

    validate_published_at(getattr(chunk, "published_at", None))

    page_number = getattr(chunk, "page_number", None)
    if isinstance(page_number, bool) or not isinstance(page_number, int) or page_number < 1:
        raise ValueError(f"page_number{loc} must be a positive integer (not bool), got {page_number!r}")

    start_char = getattr(chunk, "start_char", None)
    end_char = getattr(chunk, "end_char", None)
    if isinstance(start_char, bool) or not isinstance(start_char, int) or start_char < 0:
        raise ValueError(f"start_char{loc} must be an integer >= 0 (not bool), got {start_char!r}")
    if isinstance(end_char, bool) or not isinstance(end_char, int) or end_char <= start_char:
        raise ValueError(
            f"end_char{loc} must be an integer > start_char ({start_char}), got {end_char!r}"
        )

    text = getattr(chunk, "text", None)
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"text{loc} must be a non-empty string")


def build_embeddings(
    chunks: list[ChunkRecord],
    client: OllamaEmbedder,
    input_sha256: str | None = None,
) -> dict:
    """Construye el conjunto de embeddings validando digest, prefijos e invariantes."""
    if not chunks:
        raise ValueError("chunks list cannot be empty")

    seen_ids: set[str] = set()
    prefixed_texts: list[str] = []
    for idx, chunk in enumerate(chunks):
        validate_chunk_record(chunk, idx)
        if chunk.chunk_id in seen_ids:
            raise ValueError(f"Duplicate chunk_id detected at index {idx}: {chunk.chunk_id}")
        seen_ids.add(chunk.chunk_id)
        prefixed_texts.append(f"{DOCUMENT_PREFIX}{chunk.text}")

    digest_before = client.get_digest()
    vectors = client.embed_texts(prefixed_texts)
    digest_after = client.get_digest()

    if digest_before != digest_after:
        raise RuntimeError(
            f"Model digest changed during embedding ({digest_before} -> {digest_after}). "
            "Aborting without publishing index."
        )

    chunk_records_out: list[dict] = []
    for chunk, vec in zip(chunks, vectors):
        c_dict = asdict(chunk)
        c_dict["embedding"] = vec
        chunk_records_out.append(c_dict)

    result: dict = {
        "schema_version": SCHEMA_VERSION,
        "model": MODEL_NAME,
        "model_digest": digest_before,
        "dimension": DIMENSION,
        "document_prefix": DOCUMENT_PREFIX,
        "query_prefix": QUERY_PREFIX,
        "chunks": chunk_records_out,
    }
    if input_sha256 is not None:
        result["input_sha256"] = input_sha256

    return result


def read_and_validate_chunks_jsonl(input_path: Path) -> tuple[list[ChunkRecord], str]:
    """Lee y valida todas las líneas de un JSONL de fragmentos y calcula su SHA-256."""
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    raw_bytes = input_path.read_bytes()
    if len(raw_bytes) == 0:
        raise ValueError("Input JSONL file is empty")
    input_sha256 = hashlib.sha256(raw_bytes).hexdigest()

    lines = raw_bytes.decode("utf-8").splitlines()
    chunks: list[ChunkRecord] = []
    seen_ids: set[str] = set()

    for line_num, line in enumerate(lines, start=1):
        line_str = line.strip()
        if not line_str:
            continue
        try:
            data = json.loads(line_str)
        except Exception as exc:
            raise ValueError(f"Malformed JSON on line {line_num}: {exc}") from exc

        if not isinstance(data, dict):
            raise ValueError(f"Line {line_num} is not a JSON object")

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
        }
        missing = required_keys - set(data.keys())
        if missing:
            raise ValueError(f"Missing required fields on line {line_num}: {sorted(missing)}")

        chunk = ChunkRecord(
            chunk_id=data["chunk_id"],
            doc_id=data["doc_id"],
            source_file=data["source_file"],
            published_at=data["published_at"],
            page_number=data["page_number"],
            start_char=data["start_char"],
            end_char=data["end_char"],
            text=data["text"],
            strategy=data["strategy"],
        )
        validate_chunk_record(chunk, line_num)
        if chunk.chunk_id in seen_ids:
            raise ValueError(f"Duplicate chunk_id on line {line_num}: {chunk.chunk_id}")
        seen_ids.add(chunk.chunk_id)
        chunks.append(chunk)

    if not chunks:
        raise ValueError("Input JSONL contains no valid chunk records")

    return chunks, input_sha256


def write_embeddings_json(data: dict, output_path: Path) -> None:
    """Escribe el índice de embeddings en formato JSON de manera atómica."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_name(f"{output_path.name}.tmp_{os.getpid()}_{uuid.uuid4().hex}")
    try:
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, allow_nan=False, indent=2)
        temp_path.replace(output_path)
    except Exception:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass
        raise


def main(argv: list[str] | None = None) -> int:
    """Punto de entrada de la CLI de generación de embeddings."""
    parser = argparse.ArgumentParser(description="Generate embeddings for text chunks using Ollama.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)

    input_path = validate_project_path(args.input)
    output_path = validate_project_path(args.output)
    if input_path == output_path:
        raise ValueError("Output path cannot be the same as input path.")

    # Validar todas las líneas y campos antes de contactar con Ollama
    chunks, input_sha256 = read_and_validate_chunks_jsonl(input_path)

    client = OllamaEmbedder()
    result = build_embeddings(chunks, client, input_sha256=input_sha256)
    write_embeddings_json(result, output_path)
    print(f"Generated embeddings for {len(chunks)} chunks to {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
