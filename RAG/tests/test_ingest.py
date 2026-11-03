import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import uuid
from pypdf import PdfWriter
from rag_pensiones.ingest import (
    PROJECT_ROOT,
    PageRecord,
    extract_pages,
    main,
    validate_page_range,
    validate_published_at,
    write_jsonl,
)


class TestIngest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pdf_path = (
            PROJECT_ROOT / "Airef" / "2025_Opinion_Sostenibilidad_AAPP" / "Opinion sobre la Sostenibilidad de las AAPP a largo plazo Marzo 2025.pdf"
        )

    def test_invalid_dates(self):
        for invalid in ["2025-13-01", "2025-02-30", "2025/03/31", "2025-3-1", "invalid", True]:
            with self.assertRaises(ValueError):
                validate_published_at(invalid)

    def test_invalid_page_ranges_and_bool(self):
        for start, end, total in [(0, 5, 10), (5, 4, 10), (1, 15, 10), (True, 5, 10), (1, False, 10)]:
            with self.assertRaises(ValueError):
                validate_page_range(start, end, total)

    def test_empty_pdf_rejected(self):
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            empty_pdf = Path(f.name)
        try:
            with self.assertRaises(ValueError):
                extract_pages(empty_pdf, "2025-03-31")
        finally:
            empty_pdf.unlink(missing_ok=True)

    def test_zero_page_pdf_rejected(self):
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            zero_pdf = Path(f.name)
        try:
            writer = PdfWriter()
            with open(zero_pdf, "wb") as out:
                writer.write(out)
            with self.assertRaises(ValueError):
                extract_pages(zero_pdf, "2025-03-31")
        finally:
            zero_pdf.unlink(missing_ok=True)

    def test_blank_page_preservation(self):
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            blank_pdf = Path(f.name)
        try:
            writer = PdfWriter()
            writer.add_blank_page(width=100, height=100)
            with open(blank_pdf, "wb") as out:
                writer.write(out)
            records = extract_pages(blank_pdf, "2025-03-31")
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].page_number, 1)
            self.assertEqual(records[0].text, "")
        finally:
            blank_pdf.unlink(missing_ok=True)

    def test_metadata_determinism_and_unicode(self):
        rec1 = extract_pages(self.pdf_path, "2025-03-31", 95, 95)
        rec2 = extract_pages(self.pdf_path, "2025-03-31", 95, 95)
        self.assertEqual(rec1, rec2)
        self.assertEqual(len(rec1[0].doc_id), 64)
        self.assertEqual(rec1[0].source_file, self.pdf_path.name)
        self.assertEqual(rec1[0].published_at, "2025-03-31")
        self.assertEqual(rec1[0].page_number, 95)

    def test_write_jsonl_success_unicode_and_deterministic(self):
        records = [
            PageRecord("doc_id_1", "source.pdf", "2025-03-31", 1, "Pensión de jubilación: cálculo y variación."),
            PageRecord("doc_id_1", "source.pdf", "2025-03-31", 2, "Año 2025: revalorización según IPC y cuantía."),
        ]
        with tempfile.TemporaryDirectory() as tmp_dir:
            out1 = Path(tmp_dir) / "run1.jsonl"
            out2 = Path(tmp_dir) / "run2.jsonl"
            write_jsonl(records, out1)
            write_jsonl(records, out2)

            content1 = out1.read_text(encoding="utf-8")
            content2 = out2.read_text(encoding="utf-8")
            self.assertEqual(content1, content2)

            self.assertIn("Pensión de jubilación", content1)
            self.assertIn("Año 2025", content1)
            self.assertNotIn(r"\u", content1)

            lines = content1.strip().split("\n")
            self.assertEqual(len(lines), 2)
            for line, rec in zip(lines, records):
                parsed = json.loads(line)
                self.assertEqual(
                    parsed,
                    {
                        "doc_id": rec.doc_id,
                        "source_file": rec.source_file,
                        "published_at": rec.published_at,
                        "page_number": rec.page_number,
                        "text": rec.text,
                    },
                )

    def test_atomic_write_and_previous_output_intact(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_file = Path(tmp_dir) / "output.jsonl"
            out_file.write_text("PREVIOUS_CONTENT", encoding="utf-8")

            def faulty_records():
                yield PageRecord("id", "src", "2025-03-31", 1, "ok")
                raise RuntimeError("Extraction failed midway")

            with self.assertRaises(RuntimeError):
                write_jsonl(faulty_records(), out_file)

            self.assertEqual(out_file.read_text(encoding="utf-8"), "PREVIOUS_CONTENT")
            leftovers = [p for p in Path(tmp_dir).iterdir() if p != out_file]
            self.assertEqual(leftovers, [])

    def test_cli_rejects_same_input_output(self):
        with self.assertRaises(ValueError):
            main(
                [
                    "--input",
                    str(self.pdf_path),
                    "--published-at",
                    "2025-03-31",
                    "--output",
                    str(self.pdf_path),
                ]
            )

    def test_cli_rejects_paths_outside_project_root(self):
        marker = uuid.uuid4().hex
        outside_input = PROJECT_ROOT.parent / f"rag-test-input-{marker}.pdf"
        outside_output = PROJECT_ROOT.parent / f"rag-test-output-{marker}.jsonl"
        valid_output = PROJECT_ROOT / "tmp" / f"rag-test-valid-{marker}.jsonl"

        with self.assertRaises(ValueError):
            main(
                [
                    "--input",
                    str(outside_input),
                    "--published-at",
                    "2025-03-31",
                    "--output",
                    str(valid_output),
                ]
            )

        try:
            with self.assertRaises(ValueError):
                main(
                    [
                        "--input",
                        str(self.pdf_path),
                        "--published-at",
                        "2025-03-31",
                        "--output",
                        str(outside_output),
                    ]
                )
            self.assertFalse(outside_output.exists())
        finally:
            pass  # Rejection tests must never delete files outside the project.

    def test_cli_successful_extraction_pages_95_99(self):
        real_sha256 = hashlib.sha256(self.pdf_path.read_bytes()).hexdigest()
        tmp_base = PROJECT_ROOT / "tmp"
        created_tmp = not tmp_base.exists()
        if created_tmp:
            tmp_base.mkdir(parents=True, exist_ok=True)
        try:
            with tempfile.TemporaryDirectory(dir=tmp_base) as tmp_dir:
                out_file = Path(tmp_dir) / "cli_extracted.jsonl"
                exit_code = main(
                    [
                        "--input",
                        str(self.pdf_path),
                        "--published-at",
                        "2025-03-31",
                        "--start-page",
                        "95",
                        "--end-page",
                        "99",
                        "--output",
                        str(out_file),
                    ]
                )
                self.assertEqual(exit_code, 0)
                self.assertTrue(out_file.exists())
                lines = out_file.read_text(encoding="utf-8").strip().split("\n")
                self.assertEqual(len(lines), 5)
                records = [json.loads(line) for line in lines]
                self.assertEqual([r["page_number"] for r in records], [95, 96, 97, 98, 99])
                for r in records:
                    self.assertEqual(r["doc_id"], real_sha256)
                    self.assertEqual(r["published_at"], "2025-03-31")
                    self.assertEqual(r["source_file"], self.pdf_path.name)
        finally:
            if created_tmp and tmp_base.exists() and not any(tmp_base.iterdir()):
                tmp_base.rmdir()

    def test_real_pdf_evidence_p95_p97(self):
        records = extract_pages(self.pdf_path, "2025-03-31", 95, 99)
        self.assertEqual(len(records), 5)
        self.assertEqual([r.page_number for r in records], [95, 96, 97, 98, 99])
        self.assertIn("16,1", records[0].text)
        self.assertIn("pensionista", records[2].text.lower())
        self.assertIn("32", records[2].text)
        self.assertIn("Un pensionista puede ser beneficiario de más de una pensión", records[2].text)
        self.assertIn("La ratio pensionistas", records[2].text)


if __name__ == "__main__":
    unittest.main()
