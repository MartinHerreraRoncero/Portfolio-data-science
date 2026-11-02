"""Extracción reproducible de páginas PDF a formato JSONL sin embeddings ni LLM."""

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
import io
import json
import os
from pathlib import Path
import re
import sys
import uuid
from pypdf import PdfReader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")


@dataclass(frozen=True)
class PageRecord:
    doc_id: str
    source_file: str
    published_at: str
    page_number: int
    text: str


def validate_published_at(published_at: str) -> str:
    if not isinstance(published_at, str) or not DATE_PATTERN.match(published_at):
        raise ValueError(f"Invalid date format: {published_at!r}. Expected YYYY-MM-DD.")
    try:
        datetime.strptime(published_at, "%Y-%m-%d")
    except ValueError as exc:
        raise ValueError(f"Invalid calendar date: {published_at!r}") from exc
    return published_at


def validate_page_range(start_page: int, end_page: int | None, total_pages: int) -> tuple[int, int]:
    if total_pages < 1:
        raise ValueError("PDF has no pages.")
    if isinstance(start_page, bool) or not isinstance(start_page, int):
        raise ValueError(f"start_page must be int (not bool): {type(start_page).__name__}")
    if end_page is not None and (isinstance(end_page, bool) or not isinstance(end_page, int)):
        raise ValueError(f"end_page must be int (not bool): {type(end_page).__name__}")
    end = total_pages if end_page is None else end_page
    if not (1 <= start_page <= end <= total_pages):
        raise ValueError(
            f"Invalid page range: start={start_page}, end={end}, total={total_pages}. "
            "Must satisfy 1 <= start <= end <= total."
        )
    return start_page, end


def extract_pages(
    pdf_path: Path,
    published_at: str,
    start_page: int = 1,
    end_page: int | None = None,
) -> list[PageRecord]:
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    data = pdf_path.read_bytes()
    if len(data) == 0:
        raise ValueError("PDF file is empty.")
    valid_date = validate_published_at(published_at)
    try:
        reader = PdfReader(io.BytesIO(data))
    except Exception as exc:
        raise ValueError(f"Cannot parse PDF: {exc}") from exc
    total = len(reader.pages)
    start, end = validate_page_range(start_page, end_page, total)
    doc_id = hashlib.sha256(data).hexdigest()
    records: list[PageRecord] = []
    for p in range(start, end + 1):
        extracted = reader.pages[p - 1].extract_text()
        records.append(
            PageRecord(
                doc_id=doc_id,
                source_file=pdf_path.name,
                published_at=valid_date,
                page_number=p,
                text=extracted if extracted is not None else "",
            )
        )
    return records


def write_jsonl(records: list[PageRecord], output_path: Path) -> None:
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


def validate_project_path(path: Path) -> Path:
    candidate = path if path.is_absolute() else (PROJECT_ROOT / path)
    resolved = candidate.resolve()
    try:
        resolved.relative_to(PROJECT_ROOT)
    except ValueError:
        raise ValueError(f"Path '{path}' is outside project root '{PROJECT_ROOT}'")
    return resolved


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Extract pages from PDF to JSONL.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--published-at", required=True, type=str)
    parser.add_argument("--start-page", type=int, default=1)
    parser.add_argument("--end-page", type=int, default=None)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)

    input_path = validate_project_path(args.input)
    output_path = validate_project_path(args.output)
    if input_path == output_path:
        raise ValueError("Output path cannot be the same as input path.")
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    records = extract_pages(
        pdf_path=input_path,
        published_at=args.published_at,
        start_page=args.start_page,
        end_page=args.end_page,
    )
    write_jsonl(records, output_path)
    print(f"Extracted {len(records)} pages to {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
