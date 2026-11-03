"""Unit tests for layout catalog reanchoring with exact raw offsets."""

import copy
import hashlib
import unittest

from experiments.preserve_evidence import collapse_spaces, validate_catalog
from experiments.reanchor_layout_catalog import (
    TYPOGRAPHIC_DASHES,
    canonical_map,
    reanchor_catalog,
    reanchor_span,
)


class TestLayoutCatalog(unittest.TestCase):
    """Test suite for canonical mapping, span reanchoring, and catalog reanchoring."""

    def setUp(self):
        self.doc_id = "test_doc_sha256_abcdef1234567890"
        self.source_file = "test_document.pdf"
        self.published_at = "2025-03-31"

    def _make_page(self, page_number: int, text: str) -> dict:
        return {
            "doc_id": self.doc_id,
            "source_file": self.source_file,
            "published_at": self.published_at,
            "page_number": page_number,
            "text": text,
        }

    # 1. whitespace+dash+decomposedaccent rawoffsets exact
    def test_canonical_map_whitespace_dash_decomposed_accents(self):
        # Precomposed vs decomposed
        precomposed = "pensión"
        decomposed = "pensio\u0301n"
        norm_pre, offset_pre = canonical_map(precomposed)
        norm_dec, offset_dec = canonical_map(decomposed)
        self.assertEqual(norm_pre, norm_dec)
        self.assertEqual(len(norm_pre), len(offset_pre))
        self.assertEqual(len(norm_dec), len(offset_dec))

        # Typography dashes mapping to '-'
        for dash in TYPOGRAPHIC_DASHES:
            sample = f"Real Decreto {dash} Ley"
            norm, offsets = canonical_map(sample)
            self.assertEqual(norm, "RealDecreto-Ley")
            dash_idx = norm.index('-')
            start, end = offsets[dash_idx]
            self.assertEqual(sample[start:end], dash)

        # Raw offsets exact
        page_text = "Prefacio. En el Real Decreto \u2013 Ley 2/2023, la pensio\u0301n subirá un 12,9%. Fin."
        quote = "Real Decreto - Ley 2/2023, la pensión subirá un 12,9%."
        start, end = reanchor_span(quote, page_text)
        expected_slice = "Real Decreto \u2013 Ley 2/2023, la pensio\u0301n subirá un 12,9%."
        self.assertEqual(page_text[start:end], expected_slice)
        self.assertEqual(collapse_spaces(page_text[start:end]), expected_slice)

    # 2. minus/%/decimals not altered
    def test_minus_percent_decimals_not_altered(self):
        text = "El déficit bajó -1,4% y luego a \u22120,3% con tasa de 16,1%."
        norm, offsets = canonical_map(text)
        self.assertIn("-1,4%", norm)
        self.assertIn("16,1%", norm)
        # Verify characters are preserved as literal chars, not numeric casts or deleted
        self.assertTrue(any(text[s:e] == "%" for s, e in offsets))
        self.assertTrue(any(text[s:e] == "," for s, e in offsets))
        self.assertTrue(any(text[s:e] == "-" for s, e in offsets))
        self.assertTrue(any(text[s:e] == "\u2212" for s, e in offsets))

        # Test span reanchoring with percentages and decimals
        page_text = "Texto previo: gasto del 12,9 al 16,1% del PIB. Más texto."
        quote = "gasto del 12,9 al 16,1% del PIB."
        start, end = reanchor_span(quote, page_text)
        self.assertEqual(page_text[start:end], quote)

    # 3. canonical duplicates/overlaps reject
    def test_canonical_duplicates_and_overlaps_reject(self):
        # Exact duplicate
        page_text = "El gasto aumentará. Otro texto. El gasto aumentará."
        quote = "El gasto aumentará."
        with self.assertRaises(ValueError) as ctx:
            reanchor_span(quote, page_text)
        self.assertIn("Ambiguous", str(ctx.exception))

        # Overlapping occurrences (e.g., "ABABA" has "ABA" at pos 0 and pos 2)
        overlapping_page = "ABABA inicio de párrafo."
        overlapping_quote = "ABA"
        with self.assertRaises(ValueError) as ctx:
            reanchor_span(overlapping_quote, overlapping_page)
        self.assertIn("Ambiguous", str(ctx.exception))

        # Overlapping repeated characters
        page_aaa = "AAAA texto de prueba"
        quote_aa = "AA"
        with self.assertRaises(ValueError) as ctx:
            reanchor_span(quote_aa, page_aaa)
        self.assertIn("Ambiguous", str(ctx.exception))

    # 4. missing changednumber rejects
    def test_missing_changed_number_rejects(self):
        source_text = "El gasto crecerá un 16,1% del PIB hasta 2050."
        target_text = "El gasto crecerá un 16,2% del PIB hasta 2050."
        quote = "El gasto crecerá un 16,1% del PIB hasta 2050."

        with self.assertRaises(ValueError) as ctx:
            reanchor_span(quote, target_text, unit_id="U1", page_number=1)
        err = str(ctx.exception)
        self.assertIn("Normalized quote not found", err)
        self.assertIn("U1", err)
        self.assertIn("1", err)

        # In reanchor_catalog
        source_pages = [self._make_page(1, source_text)]
        target_pages = [self._make_page(1, target_text)]
        catalog = {
            "units": [
                {
                    "id": "U1",
                    "kind": "body",
                    "complete_end": True,
                    "spans": [
                        {
                            "doc_id": self.doc_id,
                            "source_file": self.source_file,
                            "published_at": self.published_at,
                            "page_number": 1,
                            "start_char": 0,
                            "end_char": len(quote),
                            "quote": quote,
                        }
                    ],
                }
            ]
        }
        with self.assertRaises(ValueError) as ctx:
            reanchor_catalog(catalog, source_pages, target_pages)
        err_cat = str(ctx.exception)
        self.assertIn("U1", err_cat)
        self.assertIn("1", err_cat)

    # 5. invalid sourcequote/offset/page identity/duplicates reject
    def test_invalid_source_quote_offset_page_identity_duplicates_reject(self):
        page_1 = self._make_page(1, "Texto original en página uno.")
        page_2 = self._make_page(2, "Texto original en página dos.")

        # Invalid source quote (doesn't match source page text)
        bad_quote_catalog = {
            "units": [
                {
                    "id": "U1",
                    "kind": "body",
                    "complete_end": True,
                    "spans": [
                        {
                            "doc_id": self.doc_id,
                            "source_file": self.source_file,
                            "published_at": self.published_at,
                            "page_number": 1,
                            "start_char": 0,
                            "end_char": 14,
                            "quote": "Texto diferente",  # Doesn't match page text
                        }
                    ],
                }
            ]
        }
        with self.assertRaises(ValueError):
            reanchor_catalog(bad_quote_catalog, [page_1], [page_1])

        # Invalid source offsets (start >= end)
        bad_offsets_catalog = {
            "units": [
                {
                    "id": "U1",
                    "kind": "body",
                    "complete_end": True,
                    "spans": [
                        {
                            "doc_id": self.doc_id,
                            "source_file": self.source_file,
                            "published_at": self.published_at,
                            "page_number": 1,
                            "start_char": 10,
                            "end_char": 5,
                            "quote": "Texto",
                        }
                    ],
                }
            ]
        }
        with self.assertRaises(ValueError):
            reanchor_catalog(bad_offsets_catalog, [page_1], [page_1])

        # Bool offsets reject
        bool_offsets_catalog = {
            "units": [
                {
                    "id": "U1",
                    "kind": "body",
                    "complete_end": True,
                    "spans": [
                        {
                            "doc_id": self.doc_id,
                            "source_file": self.source_file,
                            "published_at": self.published_at,
                            "page_number": 1,
                            "start_char": True,
                            "end_char": 5,
                            "quote": "Texto",
                        }
                    ],
                }
            ]
        }
        with self.assertRaises(ValueError):
            reanchor_catalog(bool_offsets_catalog, [page_1], [page_1])

        # Page identity mismatch (missing page in target)
        valid_catalog = {
            "units": [
                {
                    "id": "U1",
                    "kind": "body",
                    "complete_end": True,
                    "spans": [
                        {
                            "doc_id": self.doc_id,
                            "source_file": self.source_file,
                            "published_at": self.published_at,
                            "page_number": 1,
                            "start_char": 0,
                            "end_char": 14,
                            "quote": "Texto original",
                        }
                    ],
                }
            ]
        }
        with self.assertRaises(ValueError) as ctx:
            reanchor_catalog(valid_catalog, [page_1, page_2], [page_1])
        self.assertIn("identical unique page identity", str(ctx.exception).lower())

        # Duplicate page in source_pages
        duplicate_source_pages = [page_1, copy.deepcopy(page_1)]
        with self.assertRaises(ValueError) as ctx:
            reanchor_catalog(valid_catalog, duplicate_source_pages, [page_1])
        self.assertIn("duplicate", str(ctx.exception).lower())

        # Empty / non-string / bool inputs reject
        with self.assertRaises(ValueError):
            canonical_map("")
        with self.assertRaises(ValueError):
            canonical_map("   ")
        with self.assertRaises(ValueError):
            canonical_map(123)
        with self.assertRaises(ValueError):
            canonical_map(True)
        with self.assertRaises(ValueError):
            reanchor_span("", "some text")
        with self.assertRaises(ValueError):
            reanchor_span("quote", "")

    # 6. crosspagebody+note retain semanticIDs/spancount and exacttargetquotes
    def test_crosspage_body_and_note_reanchor(self):
        p1_src_text = "Párrafo inicial que continúa en la siguiente página "
        p2_src_text = "con la conclusión del párrafo. Nota al pie sobre pensiones."

        p1_tgt_text = "Cabecera.\nPárrafo inicial que continúa en la siguiente página\nPie."
        p2_tgt_text = "Cabecera.\ncon la conclusión del párrafo.\nNota al pie sobre pensiones.\nFin."

        p1_src = self._make_page(1, p1_src_text)
        p2_src = self._make_page(2, p2_src_text)
        p1_tgt = self._make_page(1, p1_tgt_text)
        p2_tgt = self._make_page(2, p2_tgt_text)

        source_catalog = {
            "profile": "Test profile",
            "scope": "Test scope",
            "units": [
                {
                    "id": "U1",
                    "kind": "body",
                    "complete_end": True,
                    "spans": [
                        {
                            "doc_id": self.doc_id,
                            "source_file": self.source_file,
                            "published_at": self.published_at,
                            "page_number": 1,
                            "start_char": 0,
                            "end_char": len(p1_src_text) - 1,
                            "quote": collapse_spaces(p1_src_text[:-1]),
                        },
                        {
                            "doc_id": self.doc_id,
                            "source_file": self.source_file,
                            "published_at": self.published_at,
                            "page_number": 2,
                            "start_char": 0,
                            "end_char": 30,
                            "quote": "con la conclusión del párrafo.",
                        },
                    ],
                },
                {
                    "id": "U2",
                    "kind": "note",
                    "complete_end": False,
                    "spans": [
                        {
                            "doc_id": self.doc_id,
                            "source_file": self.source_file,
                            "published_at": self.published_at,
                            "page_number": 2,
                            "start_char": 31,
                            "end_char": len(p2_src_text),
                            "quote": "Nota al pie sobre pensiones.",
                        }
                    ],
                },
            ],
        }

        new_catalog, audit = reanchor_catalog(source_catalog, [p1_src, p2_src], [p1_tgt, p2_tgt])

        # Validate structure preservation
        self.assertEqual(len(new_catalog["units"]), 2)
        self.assertEqual(new_catalog["units"][0]["id"], "U1")
        self.assertEqual(new_catalog["units"][0]["kind"], "body")
        self.assertTrue(new_catalog["units"][0]["complete_end"])
        self.assertEqual(len(new_catalog["units"][0]["spans"]), 2)

        self.assertEqual(new_catalog["units"][1]["id"], "U2")
        self.assertEqual(new_catalog["units"][1]["kind"], "note")
        self.assertFalse(new_catalog["units"][1]["complete_end"])
        self.assertEqual(len(new_catalog["units"][1]["spans"]), 1)

        # Target quotes match collapse_spaces(target_text[start:end])
        for unit in new_catalog["units"]:
            for span in unit["spans"]:
                tgt_page = p1_tgt if span["page_number"] == 1 else p2_tgt
                raw_slice = tgt_page["text"][span["start_char"]:span["end_char"]]
                self.assertEqual(collapse_spaces(raw_slice), span["quote"])

        # Audit report checks
        self.assertEqual(audit["original_unit_count"], 2)
        self.assertEqual(audit["new_unit_count"], 2)
        self.assertEqual(audit["original_span_count"], 3)
        self.assertEqual(audit["new_span_count"], 3)
        self.assertIn("normalization_rule", audit)
        self.assertIn("source_canonical_raw_sha256", audit)
        self.assertIn("target_canonical_raw_sha256", audit)

        # Catalog scope/profile says reanchored
        self.assertIn("reanchored", new_catalog["profile"].lower())
        self.assertIn("reanchored", new_catalog["scope"].lower())
        self.assertIn("reanchored", new_catalog["audit"].lower())

    # 7. original objects unchanged
    def test_original_objects_unchanged(self):
        page_1 = self._make_page(1, "Texto original que no debe ser modificado.")
        page_tgt = self._make_page(1, "Nuevo texto con: Texto original que no debe ser modificado. Extra.")
        catalog = {
            "units": [
                {
                    "id": "U1",
                    "kind": "body",
                    "complete_end": True,
                    "spans": [
                        {
                            "doc_id": self.doc_id,
                            "source_file": self.source_file,
                            "published_at": self.published_at,
                            "page_number": 1,
                            "start_char": 0,
                            "end_char": 42,
                            "quote": "Texto original que no debe ser modificado.",
                        }
                    ],
                }
            ]
        }
        cat_copy = copy.deepcopy(catalog)
        src_copy = copy.deepcopy([page_1])
        tgt_copy = copy.deepcopy([page_tgt])

        new_catalog, _ = reanchor_catalog(catalog, [page_1], [page_tgt])

        self.assertEqual(catalog, cat_copy)
        self.assertEqual([page_1], src_copy)
        self.assertEqual([page_tgt], tgt_copy)

    # 8. identical source-target sourceunit offsets valid
    def test_identical_source_target_offsets_valid(self):
        page_text = "El gasto total en pensiones aumentará del 12,9 al 16,1% del PIB."
        page = self._make_page(1, page_text)
        catalog = {
            "units": [
                {
                    "id": "U1",
                    "kind": "body",
                    "complete_end": True,
                    "spans": [
                        {
                            "doc_id": self.doc_id,
                            "source_file": self.source_file,
                            "published_at": self.published_at,
                            "page_number": 1,
                            "start_char": 0,
                            "end_char": len(page_text),
                            "quote": page_text,
                        }
                    ],
                }
            ]
        }
        new_catalog, audit = reanchor_catalog(catalog, [page], [page])
        self.assertEqual(new_catalog["units"][0]["spans"][0]["start_char"], 0)
        self.assertEqual(new_catalog["units"][0]["spans"][0]["end_char"], len(page_text))
        self.assertEqual(new_catalog["units"][0]["spans"][0]["quote"], page_text)
        self.assertEqual(audit["source_canonical_raw_sha256"], audit["target_canonical_raw_sha256"])

    # 9. Unicode grapheme boundaries, combining marks, and accent alterations rejected
    def test_reanchor_unicode_grapheme_boundaries_and_accents(self):
        # "a" in "á" REJECT
        with self.assertRaises(ValueError) as ctx:
            reanchor_span("a", "á")
        self.assertIn("not found", str(ctx.exception).lower())

        # "a" in "áa" -> (1, 2)
        start, end = reanchor_span("a", "áa")
        self.assertEqual((start, end), (1, 2))
        self.assertEqual("áa"[start:end], "a")

        # standalone combining mark in "á" REJECT
        with self.assertRaises(ValueError):
            reanchor_span("\u0301", "á")

        # "e" in "e\u0301" REJECT
        with self.assertRaises(ValueError) as ctx:
            reanchor_span("e", "e\u0301")
        self.assertIn("not found", str(ctx.exception).lower())

        # "é" in "e\u0301" -> (0, 2)
        start, end = reanchor_span("é", "e\u0301")
        self.assertEqual((start, end), (0, 2))
        self.assertEqual("e\u0301"[start:end], "e\u0301")


if __name__ == "__main__":
    unittest.main()
