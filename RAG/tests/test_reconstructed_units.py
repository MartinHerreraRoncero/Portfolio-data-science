"""Literal provenance, paragraph anchoring and cross-page reconstruction."""
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from experiments.reconstructed_units import (
    normalized_positions, extract_units, make_hits, bundle_catalog, expand_seeds,
)


def page(number, text):
    return dict(doc_id="d", source_file="public.pdf", published_at="2025-01-01",
                page_number=number, text=text)


def span(p):
    return dict(p, start_char=0, end_char=len(p["text"]), quote=" ".join(p["text"].split())) | {"text": None}


class ReconstructedTests(unittest.TestCase):
    def test_different_documents_are_rejected_before_reading_pdf(self):
        a, b = page(1, "Primera página."), page(2, "Segunda página.")
        b["doc_id"] = "otro"
        with self.assertRaisesRegex(ValueError, "one document"):
            extract_units("not-read.pdf", [a, b])

    def test_normalization_keeps_reversible_offsets(self):
        raw = "  Sujeto\n\t y  dato 2,4. "
        text, starts, ends = normalized_positions(raw)
        i = text.index("dato")
        self.assertEqual(raw[starts[i]:ends[-1]], "dato 2,4.")

    def test_bundle_repeats_opener_as_separate_literal_evidence(self):
        p = page(1, "El régimen A es un sistema público. Aunque el sistema está cerrado desde 2017, mantiene gasto.")
        u = dict(id="U1", spans=[span(p)])
        hits, mapping = make_hits([u], {("d", 1): p})
        catalog = bundle_catalog([u], mapping)
        evidence = catalog["records"][1]["evidence"]
        self.assertEqual(len(evidence), 2)
        self.assertEqual(evidence[0]["quote"], "El régimen A es un sistema público.")
        self.assertIn("cerrado desde 2017", evidence[1]["quote"])
        self.assertTrue(all(e["quote"] in hits[0].chunk.text for e in evidence))

    def test_cross_page_sentence_is_never_a_synthetic_quote(self):
        p1 = page(1, "La proporción de mayores de 66 años")
        p2 = page(2, "sobre la población de 16 a 66 años alcanza el 49%.")
        u = dict(id="U1", spans=[span(p1), span(p2)])
        hits, mapping = make_hits([u], {("d", 1): p1, ("d", 2): p2})
        evidence = bundle_catalog([u], mapping)["records"][0]["evidence"]
        self.assertEqual(len(evidence), 2)
        self.assertEqual({h.chunk.page_number for h in hits}, {1, 2})
        for e, h in zip(evidence, hits):
            self.assertEqual(e["quote"], h.chunk.text)

    def test_oversized_sentence_is_omitted_not_shortened(self):
        p = page(1, "Texto " + "extenso " * 100 + ".")
        u = dict(id="U1", spans=[span(p)])
        _, mapping = make_hits([u], {("d", 1): p})
        c = bundle_catalog([u], mapping)
        self.assertEqual(c["records"], [])
        self.assertEqual(c["omitted"][0]["reason"], "quote_length")

    def test_incomplete_tail_is_not_a_sentence_candidate(self):
        p = page(1, "La primera frase aporta el sujeto. Una segunda frase queda cortada")
        u = dict(id="U1", spans=[span(p)])
        _, mapping = make_hits([u], {("d", 1): p})
        c = bundle_catalog([u], mapping)
        self.assertEqual(len(c["records"]), 1)
        self.assertNotIn("segunda", c["records"][0]["evidence"][0]["quote"])

    def test_tampered_span_is_rejected(self):
        p = page(1, "Un dato del documento original.")
        s = span(p); s["quote"] = "Una cifra inventada."
        with self.assertRaisesRegex(ValueError, "provenance"):
            make_hits([dict(id="U1", spans=[s])], {("d", 1): p})

    def test_no_seed_link_cannot_expand_to_unrelated_page(self):
        p = page(1, "Un párrafo con procedencia independiente.")
        seed = SimpleNamespace(rank=1, chunk=SimpleNamespace(doc_id="other", page_number=1, start_char=0, end_char=10))
        with self.assertRaisesRegex(ValueError, "No parent units"):
            expand_seeds([seed], {"units": [dict(id="U1", spans=[span(p)])]}, {("d", 1): p}, "Consulta")

    def test_layout_continuation_skips_header_and_keeps_notes_separate(self):
        p1 = page(1, "Cabecera\nLa primera parte del texto continúa")
        p2 = page(2, "Cabecera\nen la página siguiente.\nUna nota independiente.")
        f = {"/BaseFont": "CenturyGothic"}
        bold = {"/BaseFont": "CenturyGothic-Bold"}
        class FakePage:
            def __init__(self, p, fragments): self.p, self.fragments = p, fragments
            def extract_text(self, visitor_text=None):
                if visitor_text:
                    for t, font, size in self.fragments: visitor_text(t, None, None, font, size)
                return self.p["text"]
        reader = SimpleNamespace(pages=[
            FakePage(p1, [("Cabecera", f, 9), ("La primera parte del texto continúa", bold, 11.04)]),
            FakePage(p2, [("Cabecera", f, 9), ("en la página siguiente.", f, 11.04), ("Una nota independiente.", f, 9.96)])])
        with patch("experiments.reconstructed_units.PdfReader", return_value=reader):
            units = extract_units("public.pdf", [p1, p2])["units"]
        self.assertEqual(len(units), 2)
        self.assertEqual(len(units[0]["spans"]), 2)
        self.assertEqual(units[1]["kind"], "note")
        self.assertNotIn("Cabecera", units[0]["spans"][0]["quote"])

    def test_paragraph_can_continue_after_a_complete_sentence_at_page_end(self):
        p1, p2 = page(1, "El sujeto de este párrafo queda identificado."), page(2, "Esta continuación mantiene el mismo sujeto.")
        font = {"/BaseFont": "CenturyGothic"}
        class FakePage:
            def __init__(self, p, bold): self.p, self.bold = p, bold
            def extract_text(self, visitor_text=None):
                if visitor_text: visitor_text(self.p["text"], None, None, {"/BaseFont": "CenturyGothic-Bold"} if self.bold else font, 11.04)
                return self.p["text"]
        with patch("experiments.reconstructed_units.PdfReader", return_value=SimpleNamespace(pages=[FakePage(p1, True), FakePage(p2, False)])):
            units = extract_units("public.pdf", [p1, p2])["units"]
        self.assertEqual(len(units), 1)
        self.assertEqual(len(units[0]["spans"]), 2)

    def test_interleaved_note_is_never_absorbed_into_body_quote(self):
        text = "La apertura identifica el sujeto. Una nota ajena se inserta aquí. El cuerpo continúa después."
        p = page(1, text)
        fragments = [("La apertura identifica el sujeto.", "CenturyGothic-Bold", 11.04),
                     ("Una nota ajena se inserta aquí.", "CenturyGothic", 9.96),
                     ("El cuerpo continúa después.", "CenturyGothic", 11.04)]
        class FakePage:
            def extract_text(self, visitor_text=None):
                if visitor_text:
                    for t, font, size in fragments: visitor_text(t, None, None, {"/BaseFont": font}, size)
                return text
        with patch("experiments.reconstructed_units.PdfReader", return_value=SimpleNamespace(pages=[FakePage()])):
            units = extract_units("public.pdf", [p])["units"]
        self.assertEqual(len(units[0]["spans"]), 2)
        self.assertTrue(all("nota ajena" not in s["quote"] for s in units[0]["spans"]))


if __name__ == "__main__":
    unittest.main()
