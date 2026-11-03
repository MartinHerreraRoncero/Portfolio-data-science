import json
from pathlib import Path
import tempfile
import unittest
import uuid
from unittest.mock import patch

from rag_pensiones.chunking import (
    ChunkRecord,
    chunk_pages,
    main,
    read_pages_jsonl,
    write_chunks_jsonl,
)
from rag_pensiones.ingest import PROJECT_ROOT, PageRecord


class TestChunking(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pages_jsonl_path = PROJECT_ROOT / "data" / "processed" / "airef_2025_pages.jsonl"
        cls.valid_doc_id = "2af1f49f2d1554c42e0eae82f6855683224d31b89b974216c6d40f1a48680ea1"
        cls.valid_source = "Opinion sobre la Sostenibilidad de las AAPP a largo plazo Marzo 2025.pdf"
        cls.valid_published = "2025-03-31"

    def _make_page(self, page_number: int = 1, text: str = "Texto de prueba", doc_id: str | None = None) -> PageRecord:
        return PageRecord(
            doc_id=doc_id or self.valid_doc_id,
            source_file=self.valid_source,
            published_at=self.valid_published,
            page_number=page_number,
            text=text,
        )

    def test_invalid_sizes_and_bool(self):
        page = self._make_page(1, "Texto válido para pruebas de tamaño.")
        for invalid_max in [0, -1, -500, True, False, "1000", None]:
            with self.assertRaises(ValueError):
                chunk_pages([page], max_chars=invalid_max, overlap_chars=10)

        for invalid_overlap in [-1, -50, True, False, "150", 1000, 1001, None]:
            with self.assertRaises(ValueError):
                chunk_pages([page], max_chars=1000, overlap_chars=invalid_overlap)

    def test_empty_and_whitespace_only_pages(self):
        empty_page = self._make_page(1, "")
        spaces_page = self._make_page(2, "   \n\t  \n  ")
        normal_page = self._make_page(3, "Contenido relevante de la página tres.")

        chunks_empty = chunk_pages([empty_page])
        self.assertEqual(chunks_empty, [])

        chunks_spaces = chunk_pages([spaces_page])
        self.assertEqual(chunks_spaces, [])

        chunks_mixed = chunk_pages([empty_page, normal_page, spaces_page])
        self.assertEqual(len(chunks_mixed), 1)
        self.assertEqual(chunks_mixed[0].page_number, 3)
        self.assertEqual(chunks_mixed[0].text, normal_page.text)

    def test_unicode_preservation_and_exact_offsets(self):
        unicode_text = (
            "Pensión de jubilación: cálculo de € 1.200,50 & revalorización (IPC) — "
            "Δ 2050 = 6,2% 🚀. Cobertura estricta de caracteres UTF-8."
        )
        page = self._make_page(1, unicode_text)
        chunks = chunk_pages([page], max_chars=60, overlap_chars=15)
        self.assertGreater(len(chunks), 1)

        for chunk in chunks:
            self.assertEqual(chunk.text, unicode_text[chunk.start_char : chunk.end_char])
            self.assertGreater(len(chunk.text), 0)
            self.assertLessEqual(len(chunk.text), 60)

        # Verificar serialización JSON literal UTF-8 sin secuencias de escape \u
        tmp_base = PROJECT_ROOT / "tmp"
        created_tmp = not tmp_base.exists()
        if created_tmp:
            tmp_base.mkdir(parents=True, exist_ok=True)
        try:
            with tempfile.TemporaryDirectory(dir=tmp_base) as tmp_dir:
                jsonl_file = Path(tmp_dir) / "unicode.jsonl"
                write_chunks_jsonl(chunks, jsonl_file)
                raw_content = jsonl_file.read_text(encoding="utf-8")
                self.assertIn("Pensión de jubilación", raw_content)
                self.assertIn("€", raw_content)
                self.assertIn("Δ", raw_content)
                self.assertIn("🚀", raw_content)
                self.assertNotIn(r"\u", raw_content)
        finally:
            if created_tmp and tmp_base.exists() and not any(tmp_base.iterdir()):
                tmp_base.rmdir()

    def test_very_long_text_no_separators(self):
        long_text = "x" * 3500
        page = self._make_page(1, long_text)
        chunks = chunk_pages([page], max_chars=500, overlap_chars=100)
        self.assertGreater(len(chunks), 1)

        covered = set()
        for i, chunk in enumerate(chunks):
            self.assertEqual(chunk.text, long_text[chunk.start_char : chunk.end_char])
            self.assertLessEqual(len(chunk.text), 500)
            covered.update(range(chunk.start_char, chunk.end_char))
            if i > 0:
                self.assertGreater(chunk.start_char, chunks[i - 1].start_char)
                self.assertLessEqual(chunk.start_char, chunks[i - 1].end_char)

        self.assertEqual(covered, set(range(len(long_text))))
        self.assertEqual(chunks[0].start_char, 0)
        self.assertEqual(chunks[-1].end_char, len(long_text))

    def test_overlap_zero_and_near_maximum(self):
        text = "Palabra1 palabra2 palabra3 palabra4 palabra5 palabra6 palabra7 palabra8."
        page = self._make_page(1, text)

        # Overlap = 0: siguiente inicio es exactamente end previo
        chunks_zero = chunk_pages([page], max_chars=30, overlap_chars=0)
        for i in range(1, len(chunks_zero)):
            self.assertEqual(chunks_zero[i].start_char, chunks_zero[i - 1].end_char)

        # Overlap cercano al máximo: 29 sobre 30
        chunks_near_max = chunk_pages([page], max_chars=30, overlap_chars=29)
        self.assertGreater(len(chunks_near_max), 1)
        for i in range(1, len(chunks_near_max)):
            self.assertGreater(chunks_near_max[i].start_char, chunks_near_max[i - 1].start_char)
            self.assertLessEqual(chunks_near_max[i].start_char, chunks_near_max[i - 1].end_char)

    def test_determinism_and_ids(self):
        page1 = self._make_page(1, "Texto constante para comprobación de determinismo.")
        res1 = chunk_pages([page1], max_chars=30, overlap_chars=5)
        res2 = chunk_pages([page1], max_chars=30, overlap_chars=5)
        self.assertEqual(res1, res2)
        for c1, c2 in zip(res1, res2):
            self.assertEqual(c1.chunk_id, c2.chunk_id)
            self.assertEqual(len(c1.chunk_id), 64)

        # Cambio de texto modifica chunk_id
        page_mod_text = self._make_page(1, "Texto alterado para comprobación de determinismo.")
        res_mod_text = chunk_pages([page_mod_text], max_chars=30, overlap_chars=5)
        self.assertNotEqual(res1[0].chunk_id, res_mod_text[0].chunk_id)

        # Cambio de configuración modifica strategy y chunk_id
        res_mod_cfg = chunk_pages([page1], max_chars=30, overlap_chars=10)
        self.assertNotEqual(res1[0].strategy, res_mod_cfg[0].strategy)
        self.assertNotEqual(res1[0].chunk_id, res_mod_cfg[0].chunk_id)

        # Cambio de metadata (doc_id o página) modifica chunk_id
        page_mod_doc = self._make_page(1, page1.text, doc_id="f" * 64)
        res_mod_doc = chunk_pages([page_mod_doc], max_chars=30, overlap_chars=5)
        self.assertNotEqual(res1[0].chunk_id, res_mod_doc[0].chunk_id)

        page_mod_num = self._make_page(2, page1.text)
        res_mod_num = chunk_pages([page_mod_num], max_chars=30, overlap_chars=5)
        self.assertNotEqual(res1[0].chunk_id, res_mod_num[0].chunk_id)
        ambiguous = [PageRecord("a:b", "c.pdf", "2025-03-31", 1, "texto"), PageRecord("a", "b:c.pdf", "2025-03-31", 1, "texto")]
        hashes = [chunk_pages([p])[0].chunk_id for p in ambiguous]
        self.assertNotEqual(*hashes)

    def test_page_and_doc_separation_and_inconsistent_metadata(self):
        p1 = self._make_page(1, "Página uno del documento.")
        p2 = self._make_page(2, "Página dos del documento.")
        chunks = chunk_pages([p1, p2], max_chars=50, overlap_chars=10)
        for c in chunks:
            if "uno" in c.text:
                self.assertEqual(c.page_number, 1)
                self.assertNotIn("dos", c.text)
            if "dos" in c.text:
                self.assertEqual(c.page_number, 2)
                self.assertNotIn("uno", c.text)

        # Duplicado (doc_id, page_number)
        dup1 = self._make_page(1, "Página 1 primera")
        dup2 = self._make_page(1, "Página 1 duplicada")
        with self.assertRaises(ValueError):
            chunk_pages([dup1, dup2])

        # Metadatos inconsistentes para mismo doc_id
        inconsistent_source = PageRecord(self.valid_doc_id, "other.pdf", self.valid_published, 2, "texto")
        with self.assertRaises(ValueError):
            chunk_pages([p1, inconsistent_source])

        inconsistent_date = PageRecord(self.valid_doc_id, self.valid_source, "2024-01-01", 2, "texto")
        with self.assertRaises(ValueError):
            chunk_pages([p1, inconsistent_date])

        # Páginas con metadatos inválidos
        invalid_num_zero = PageRecord(self.valid_doc_id, self.valid_source, self.valid_published, 0, "texto")
        with self.assertRaises(ValueError):
            chunk_pages([invalid_num_zero])

        invalid_num_bool = PageRecord(self.valid_doc_id, self.valid_source, self.valid_published, True, "texto")
        with self.assertRaises(ValueError):
            chunk_pages([invalid_num_bool])

        invalid_date = PageRecord(self.valid_doc_id, self.valid_source, "fecha-invalida", 1, "texto")
        with self.assertRaises(ValueError):
            chunk_pages([invalid_date])

    def test_independent_invariants_coverage_and_progress(self):
        samples = [
            "Corto.",
            "Una frase un poco más larga que cabe justo en una ventana única.",
            "Párrafo 1.\n\nPárrafo 2 con más detalle.\n\nPárrafo 3 final.",
            "Línea 1\nLínea 2\nLínea 3\nLínea 4\nLínea 5\nLínea 6\nLínea 7\nLínea 8",
            "Palabras " * 200,
        ]
        for idx, text in enumerate(samples, start=1):
            page = self._make_page(idx, text)
            for max_chars, overlap_chars in [(50, 10), (100, 20), (200, 50), (40, 0)]:
                chunks = chunk_pages([page], max_chars=max_chars, overlap_chars=overlap_chars)
                self.assertGreater(len(chunks), 0)
                self.assertEqual(chunks[0].start_char, 0)
                self.assertEqual(chunks[-1].end_char, len(text))

                covered_chars = set()
                for i, c in enumerate(chunks):
                    self.assertGreater(c.end_char, c.start_char)
                    self.assertLessEqual(c.end_char - c.start_char, max_chars)
                    self.assertEqual(c.text, text[c.start_char : c.end_char])
                    covered_chars.update(range(c.start_char, c.end_char))
                    if i > 0:
                        prev = chunks[i - 1]
                        self.assertGreater(c.start_char, prev.start_char)
                        self.assertLessEqual(c.start_char, prev.end_char)

                self.assertEqual(covered_chars, set(range(len(text))))

    def test_preferred_cuts_hierarchy(self):
        # 1. Doble salto gana sobre oración, salto y espacio
        # text: 60 chars. max_chars=50. Last 40% es [30, 50].
        # Ponemos doble salto en el último 40%, además de punto y espacio
        base1 = ("a" * 32) + "\n\n" + ("b" * 5) + ". " + ("c" * 20)
        p1 = self._make_page(1, base1)
        chunks1 = chunk_pages([p1], max_chars=50, overlap_chars=0)
        self.assertEqual(chunks1[0].end_char, 34)  # 32 + len('\n\n')

        # 2. Fin de oración gana sobre salto simple y espacio
        base2 = ("a" * 32) + ". " + ("b" * 5) + "\n" + ("c" * 20)
        p2 = self._make_page(1, base2)
        chunks2 = chunk_pages([p2], max_chars=50, overlap_chars=0)
        self.assertEqual(chunks2[0].end_char, 34)  # 32 + len('. ')

        # 3. Salto de línea simple gana sobre espacio
        base3 = ("a" * 32) + "\n" + ("b" * 5) + " " + ("c" * 20)
        p3 = self._make_page(1, base3)
        chunks3 = chunk_pages([p3], max_chars=50, overlap_chars=0)
        self.assertEqual(chunks3[0].end_char, 33)  # 32 + len('\n')

        # 4. Espacio cuando no hay cortes superiores
        base4 = ("a" * 32) + " " + ("b" * 20)
        p4 = self._make_page(1, base4)
        chunks4 = chunk_pages([p4], max_chars=50, overlap_chars=0)
        self.assertEqual(chunks4[0].end_char, 33)  # 32 + len(' ')

        # 5. Corte duro cuando no hay ningún separador en el último 40%
        base5 = ("a" * 20) + " " + ("b" * 40)  # el espacio está antes del último 40% (índice 20 < 30)
        p5 = self._make_page(1, base5)
        chunks5 = chunk_pages([p5], max_chars=50, overlap_chars=0)
        self.assertEqual(chunks5[0].end_char, 50)

    def test_cli_pages_95_99_real_execution(self):
        tmp_base = PROJECT_ROOT / "tmp"
        created_tmp = not tmp_base.exists()
        if created_tmp:
            tmp_base.mkdir(parents=True, exist_ok=True)
        try:
            with tempfile.TemporaryDirectory(dir=tmp_base) as tmp_dir:
                out_file = Path(tmp_dir) / "cli_chunks_test.jsonl"
                exit_code = main(
                    [
                        "--input",
                        str(self.pages_jsonl_path),
                        "--max-chars",
                        "1000",
                        "--overlap-chars",
                        "150",
                        "--output",
                        str(out_file),
                    ]
                )
                self.assertEqual(exit_code, 0)
                self.assertTrue(out_file.exists())
                lines = out_file.read_text(encoding="utf-8").strip().split("\n")
                self.assertGreater(len(lines), 5)
                chunks_data = [json.loads(line) for line in lines]
                pages_found = {c["page_number"] for c in chunks_data}
                self.assertEqual(pages_found, {95, 96, 97, 98, 99})
                for c in chunks_data:
                    self.assertLessEqual(len(c["text"]), 1000)
                    self.assertIn("page-local-v1", c["strategy"])
        finally:
            if created_tmp and tmp_base.exists() and not any(tmp_base.iterdir()):
                tmp_base.rmdir()

    def test_cli_atomic_write_and_error_preservation(self):
        tmp_base = PROJECT_ROOT / "tmp"
        created_tmp = not tmp_base.exists()
        if created_tmp:
            tmp_base.mkdir(parents=True, exist_ok=True)
        try:
            with tempfile.TemporaryDirectory(dir=tmp_base) as tmp_dir:
                out_file = Path(tmp_dir) / "existing_output.jsonl"
                out_file.write_text("CONTENIDO_PREVIO", encoding="utf-8")

                # 1. Entrada con JSONL malformado
                bad_jsonl = Path(tmp_dir) / "malformed.jsonl"
                bad_jsonl.write_text('{"doc_id": "test"}\n{linea invalida}\n', encoding="utf-8")

                with self.assertRaises(ValueError):
                    main(
                        [
                            "--input",
                            str(bad_jsonl),
                            "--output",
                            str(out_file),
                        ]
                    )
                self.assertEqual(out_file.read_text(encoding="utf-8"), "CONTENIDO_PREVIO")

                # 2. Error durante la escritura atómica
                def failing_writer(*args, **kwargs):
                    raise IOError("Disk error during write")

                with patch("rag_pensiones.chunking.open", side_effect=IOError("Simulated write error")):
                    with self.assertRaises(IOError):
                        write_chunks_jsonl(
                            [
                                ChunkRecord(
                                    chunk_id="id",
                                    doc_id="doc",
                                    source_file="src",
                                    published_at="2025-03-31",
                                    page_number=1,
                                    start_char=0,
                                    end_char=5,
                                    text="texto",
                                    strategy="strat",
                                )
                            ],
                            out_file,
                        )
                self.assertEqual(out_file.read_text(encoding="utf-8"), "CONTENIDO_PREVIO")
                leftovers = [p for p in Path(tmp_dir).iterdir() if p.name.startswith("existing_output.jsonl.tmp_")]
                self.assertEqual(leftovers, [])

                # 3. Rechazo input == output
                with self.assertRaises(ValueError):
                    main(["--input", str(out_file), "--output", str(out_file)])

                # 4. Rechazo rutas fuera del proyecto
                outside_input = PROJECT_ROOT.parent / f"outside_in_{uuid.uuid4().hex}.jsonl"
                with self.assertRaises(ValueError):
                    main(["--input", str(outside_input), "--output", str(out_file)])

                outside_output = PROJECT_ROOT.parent / f"outside_out_{uuid.uuid4().hex}.jsonl"
                with self.assertRaises(ValueError):
                    main(["--input", str(self.pages_jsonl_path), "--output", str(outside_output)])
                self.assertFalse(outside_output.exists())
        finally:
            if created_tmp and tmp_base.exists() and not any(tmp_base.iterdir()):
                tmp_base.rmdir()

    def test_corpus_evidence_preservation(self):
        pages = read_pages_jsonl(self.pages_jsonl_path)
        chunks = chunk_pages(pages, max_chars=1000, overlap_chars=150)

        chunks_p95 = [c for c in chunks if c.page_number == 95]
        chunks_p97 = [c for c in chunks if c.page_number == 97]

        # Comprobar presencia de la cifra 16,1 en la página 95
        has_16_1 = any("16,1" in c.text for c in chunks_p95)
        self.assertTrue(has_16_1, "La cifra 16,1 debe estar preservada en al menos un chunk de p95")

        # Comprobar presencia de la frase de la nota 32 en la página 97
        note_phrase = "Un pensionista puede ser beneficiario de más de una pensión"
        has_note_phrase = any(note_phrase in c.text for c in chunks_p97)
        self.assertTrue(has_note_phrase, "La frase de la nota debe estar preservada en al menos un chunk de p97")

        # Invariante metodológico: no asumir que la nota al pie y el párrafo del cuerpo queden juntos
        body_phrase = "senda de crecimiento descendente una vez jubilado"
        chunks_with_both = [c for c in chunks_p97 if note_phrase in c.text and body_phrase in c.text]
        # Validamos que el sistema no dependa de que coexistan en el mismo chunk
        self.assertEqual(len(chunks_with_both), 0)


if __name__ == "__main__":
    unittest.main()
