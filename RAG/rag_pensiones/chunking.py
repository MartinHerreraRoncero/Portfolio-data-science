"""Fragmentación determinista de páginas a fragmentos (chunks) con metadatos y cortes preferentes."""

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import uuid

from rag_pensiones.ingest import (
    PageRecord,
    validate_project_path,
    validate_published_at,
)

PATTERN_DOUBLE_NEWLINE = re.compile(r"\n[ \t\r]*\n")
PATTERN_SENTENCE_END = re.compile(r"[.!?]['\"»\)]?\s+")
PATTERN_SINGLE_NEWLINE = re.compile(r"\r?\n")
PATTERN_SPACE = re.compile(r"[ \t]+")


@dataclass(frozen=True)
class ChunkRecord:
    chunk_id: str
    doc_id: str
    source_file: str
    published_at: str
    page_number: int
    start_char: int
    end_char: int
    text: str
    strategy: str


def _find_preferred_cut(sub: str, rel_search_start: int, rel_search_end: int) -> int:
    """Encuentra la posición de corte según la jerarquía de separadores preferentes."""
    # 1. Doble salto de línea
    candidates = [
        m.end()
        for m in PATTERN_DOUBLE_NEWLINE.finditer(sub)
        if rel_search_start <= m.end() <= rel_search_end
    ]
    if candidates:
        return max(candidates)

    # 2. Fin de oración aproximado
    candidates = [
        m.end()
        for m in PATTERN_SENTENCE_END.finditer(sub)
        if rel_search_start <= m.end() <= rel_search_end
    ]
    if candidates:
        return max(candidates)

    # 3. Salto de línea simple
    candidates = [
        m.end()
        for m in PATTERN_SINGLE_NEWLINE.finditer(sub)
        if rel_search_start <= m.end() <= rel_search_end
    ]
    if candidates:
        return max(candidates)

    # 4. Espacio en blanco
    candidates = [
        m.end()
        for m in PATTERN_SPACE.finditer(sub)
        if rel_search_start <= m.end() <= rel_search_end
    ]
    if candidates:
        return max(candidates)

    # 5. Corte duro al límite de la ventana
    return rel_search_end


def _find_next_start(
    text: str,
    current_start: int,
    end: int,
    overlap_chars: int,
) -> int:
    """Calcula el siguiente start_char con solapamiento ajustado a límite de palabra."""
    if overlap_chars == 0:
        return end

    cand = end - overlap_chars
    # Garantizar avance estricto
    cand = max(current_start + 1, cand)

    total_len = len(text)
    # Comprobar si cand ya es un límite de inicio de palabra
    if cand == 0 or (text[cand - 1].isspace() and not text[cand].isspace()):
        next_start = cand
    else:
        # Ajustar hacia delante a límite de palabra sin superar end
        adj = cand
        while adj < end and text[adj].isspace():
            adj += 1
        if adj == cand:
            while adj < end and not text[adj].isspace():
                adj += 1
            while adj < end and text[adj].isspace():
                adj += 1

        if (
            adj <= end
            and adj < total_len
            and text[adj - 1].isspace()
            and not text[adj].isspace()
        ):
            next_start = adj
        else:
            next_start = cand

    # Salvaguardas para garantizar progreso (avance) y cobertura (sin huecos)
    return max(current_start + 1, min(next_start, end))


def chunk_pages(
    pages: list[PageRecord],
    max_chars: int = 1000,
    overlap_chars: int = 150,
) -> list[ChunkRecord]:
    """Divide las páginas en fragmentos preservando metadatos, offsets exactos y cobertura."""
    if isinstance(max_chars, bool) or not isinstance(max_chars, int) or max_chars <= 0:
        raise ValueError(f"max_chars must be a positive integer (not bool): {max_chars!r}")
    if (
        isinstance(overlap_chars, bool)
        or not isinstance(overlap_chars, int)
        or overlap_chars < 0
        or overlap_chars >= max_chars
    ):
        raise ValueError(
            f"overlap_chars must be an integer satisfying 0 <= overlap_chars < max_chars: {overlap_chars!r}"
        )

    if not hasattr(pages, "__iter__"):
        raise ValueError("pages must be an iterable of PageRecord")

    seen_pages: set[tuple[str, int]] = set()
    doc_registry: dict[str, tuple[str, str]] = {}
    strategy = f"page-local-v1:max_chars={max_chars}:overlap_chars={overlap_chars}"
    all_chunks: list[ChunkRecord] = []

    for page in pages:
        if (
            not hasattr(page, "doc_id")
            or not hasattr(page, "source_file")
            or not hasattr(page, "published_at")
            or not hasattr(page, "page_number")
            or not hasattr(page, "text")
        ):
            raise ValueError(f"Invalid page object: {page!r}")

        if not isinstance(page.doc_id, str) or not page.doc_id.strip():
            raise ValueError(f"doc_id must be a non-empty string: {page.doc_id!r}")
        if not isinstance(page.source_file, str) or not page.source_file.strip():
            raise ValueError(f"source_file must be a non-empty string: {page.source_file!r}")
        validate_published_at(page.published_at)

        if (
            isinstance(page.page_number, bool)
            or not isinstance(page.page_number, int)
            or page.page_number <= 0
        ):
            raise ValueError(f"page_number must be a positive integer (not bool): {page.page_number!r}")

        if not isinstance(page.text, str):
            raise ValueError(f"page text must be a string: {type(page.text).__name__}")

        page_key = (page.doc_id, page.page_number)
        if page_key in seen_pages:
            raise ValueError(f"Duplicate (doc_id, page_number) detected: {page_key}")
        seen_pages.add(page_key)

        expected_meta = (page.source_file, page.published_at)
        if page.doc_id in doc_registry:
            if doc_registry[page.doc_id] != expected_meta:
                raise ValueError(
                    f"Inconsistent metadata for doc_id {page.doc_id!r}: "
                    f"expected {doc_registry[page.doc_id]}, got {expected_meta}"
                )
        else:
            doc_registry[page.doc_id] = expected_meta

        # Ignorar páginas vacías o que solo contengan espacios en blanco
        if not page.text.strip():
            continue

        text = page.text
        total_chars = len(text)
        current_start = 0

        while current_start < total_chars:
            remaining = total_chars - current_start
            if remaining <= max_chars:
                end = total_chars
            else:
                hard_end = current_start + max_chars
                search_margin = int(max_chars * 0.4)
                search_start = max(current_start + 1, hard_end - search_margin)

                sub = text[current_start:hard_end]
                rel_search_start = search_start - current_start
                rel_search_end = hard_end - current_start

                cut_rel = _find_preferred_cut(sub, rel_search_start, rel_search_end)
                end = current_start + cut_rel

            chunk_text = text[current_start:end]
            raw_id = json.dumps([page.doc_id, page.source_file, page.published_at, page.page_number, current_start, end, strategy, chunk_text], ensure_ascii=False)
            chunk_id = hashlib.sha256(raw_id.encode("utf-8")).hexdigest()

            all_chunks.append(
                ChunkRecord(
                    chunk_id=chunk_id,
                    doc_id=page.doc_id,
                    source_file=page.source_file,
                    published_at=page.published_at,
                    page_number=page.page_number,
                    start_char=current_start,
                    end_char=end,
                    text=chunk_text,
                    strategy=strategy,
                )
            )

            if end >= total_chars:
                break

            current_start = _find_next_start(
                text=text,
                current_start=current_start,
                end=end,
                overlap_chars=overlap_chars,
            )

    return all_chunks


def write_chunks_jsonl(records: list[ChunkRecord], output_path: Path) -> None:
    """Escribe los registros de fragmentos a un archivo JSONL de forma atómica."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_name(f"{output_path.name}.tmp_{os.getpid()}_{uuid.uuid4().hex}")
    try:
        with open(temp_path, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")
        temp_path.replace(output_path)
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise


def read_pages_jsonl(input_path: Path) -> list[PageRecord]:
    """Lee y valida un archivo JSONL con registros de página."""
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    pages: list[PageRecord] = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line_str = line.strip()
            if not line_str:
                continue
            try:
                data = json.loads(line_str)
            except Exception as exc:
                raise ValueError(f"Malformed JSON on line {line_num}: {exc}") from exc
            if not isinstance(data, dict):
                raise ValueError(f"Line {line_num} is not a JSON object")
            required_keys = {"doc_id", "source_file", "published_at", "page_number", "text"}
            if not required_keys.issubset(data.keys()):
                missing = required_keys - set(data.keys())
                raise ValueError(f"Missing required fields on line {line_num}: {missing}")
            if isinstance(data["page_number"], bool) or not isinstance(data["page_number"], int):
                raise ValueError(f"Invalid page_number on line {line_num}: {data['page_number']!r}")
            pages.append(
                PageRecord(
                    doc_id=data["doc_id"],
                    source_file=data["source_file"],
                    published_at=data["published_at"],
                    page_number=data["page_number"],
                    text=data["text"],
                )
            )
    return pages


def main(argv: list[str] | None = None) -> int:
    """Punto de entrada de la CLI de fragmentación."""
    parser = argparse.ArgumentParser(description="Chunk extracted PDF pages into text fragments.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--max-chars", type=int, default=1000)
    parser.add_argument("--overlap-chars", type=int, default=150)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)

    input_path = validate_project_path(args.input)
    output_path = validate_project_path(args.output)
    if input_path == output_path:
        raise ValueError("Output path cannot be the same as input path.")

    # Validar entrada completa antes de tocar salida
    pages = read_pages_jsonl(input_path)
    chunks = chunk_pages(pages, max_chars=args.max_chars, overlap_chars=args.overlap_chars)
    write_chunks_jsonl(chunks, output_path)
    print(f"Generated {len(chunks)} chunks to {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
