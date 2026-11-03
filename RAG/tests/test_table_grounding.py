"""Tests unitarios rigurosos y adversariales para evidencia tabular y guardia numérica.

Incluye:
- Pruebas sintéticas con esquema completo de layout, procedencia y celdas físicas.
- Pruebas adversariales contra citas falsas, omitidas, de solo cabecera o de otras filas.
- Pruebas contra omisión de procedencia, orden de cabeceras, claves de texto libre y tipos booleanos.
- Pruebas de anti-falsificación (forged request expected_value).
- Pruebas de delimitación de tokens de tabla (tabla 1 vs tabla 10).
- Pruebas de múltiples años y descarte de asignaciones implícitas globales.
- Smoke test sobre los artefactos públicos reales del piloto v4 (sin PDF ni respuestas gold)
  para empaquetado y diez consultas (seis previas y cuatro adicionales).
"""

from __future__ import annotations

import copy
from decimal import Decimal
import json
from pathlib import Path
import unicodedata
import unittest

from rag_pensiones.document_layout import _format_table_as_text
from rag_pensiones.numeric_guard import (
    _build_target_cell,
    guard_table_facts,
    parse_canonical_decimal,
    parse_table_request,
    render_table_facts,
)
from rag_pensiones.table_evidence import (
    MAX_SUM_QUOTE_UTF8_BYTES,
    _infer_column_meta,
    build_table_packets,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _make_synthetic_bundle_and_pages() -> tuple[dict, list[dict]]:
    """Genera un bundle y páginas sintéticas con el mismo esquema de layout pero valores no-gold."""
    doc_id = "synthetic_doc_abcdef1234567890abcdef1234567890"
    source_file = "synthetic_report_2026.pdf"
    published_at = "2026-01-15"
    extraction_id = "synth_ext_01"

    table = {
        "table_id": "table_synth_cuadro1",
        "method": "reviewed_profile",
        "native_ref": None,
        "page_numbers": [95],
        "bbox": {"l": 100.0, "t": 200.0, "r": 500.0, "b": 350.0, "coord_origin": "TOPLEFT"},
        "dimensions": {"rows": 3, "cols": 4},
        "caption_text": "CUADRO 1. EMPLEO Y SALARIOS (% PIB)",
        "footnotes": ["Fuente: Estimación sintética"],
        "cells": [
            # Fila 0: Encabezados de columna
            {
                "row_start": 0,
                "row_end": 1,
                "col_start": 0,
                "col_end": 1,
                "text": "Concepto",
                "raw_text": "Concepto",
                "is_empty": False,
                "page_number": 95,
                "bbox": {"l": 100.0, "t": 200.0, "r": 200.0, "b": 230.0, "coord_origin": "TOPLEFT"},
                "column_header": True,
                "row_header": True,
                "header_path": [],
                "row_label": "",
                "cell_id": "c_r0c0",
            },
            {
                "row_start": 0,
                "row_end": 1,
                "col_start": 1,
                "col_end": 2,
                "text": "2025",
                "raw_text": "2025",
                "is_empty": False,
                "page_number": 95,
                "bbox": {"l": 200.0, "t": 200.0, "r": 300.0, "b": 230.0, "coord_origin": "TOPLEFT"},
                "column_header": True,
                "row_header": False,
                "header_path": [],
                "row_label": "",
                "cell_id": "c_r0c1",
            },
            {
                "row_start": 0,
                "row_end": 1,
                "col_start": 2,
                "col_end": 3,
                "text": "2030",
                "raw_text": "2030",
                "is_empty": False,
                "page_number": 95,
                "bbox": {"l": 300.0, "t": 200.0, "r": 400.0, "b": 230.0, "coord_origin": "TOPLEFT"},
                "column_header": True,
                "row_header": False,
                "header_path": [],
                "row_label": "",
                "cell_id": "c_r0c2",
            },
            {
                "row_start": 0,
                "row_end": 1,
                "col_start": 3,
                "col_end": 4,
                "text": "Δ 2030-2025",
                "raw_text": "Δ 2030-2025",
                "is_empty": False,
                "page_number": 95,
                "bbox": {"l": 400.0, "t": 200.0, "r": 500.0, "b": 230.0, "coord_origin": "TOPLEFT"},
                "column_header": True,
                "row_header": False,
                "header_path": [],
                "row_label": "",
                "cell_id": "c_r0c3",
            },
            # Fila 1: Empleo (10,1 / 10,5 / 0,3 publicado != 0,4 resta)
            {
                "row_start": 1,
                "row_end": 2,
                "col_start": 0,
                "col_end": 1,
                "text": "Empleo",
                "raw_text": "Empleo",
                "is_empty": False,
                "page_number": 95,
                "bbox": {"l": 100.0, "t": 230.0, "r": 200.0, "b": 280.0, "coord_origin": "TOPLEFT"},
                "column_header": False,
                "row_header": True,
                "header_path": [],
                "row_label": "Empleo",
                "cell_id": "c_r1c0",
            },
            {
                "row_start": 1,
                "row_end": 2,
                "col_start": 1,
                "col_end": 2,
                "text": "10,1",
                "raw_text": "10,1",
                "is_empty": False,
                "page_number": 95,
                "bbox": {"l": 200.0, "t": 230.0, "r": 300.0, "b": 280.0, "coord_origin": "TOPLEFT"},
                "column_header": False,
                "row_header": False,
                "header_path": ["2025"],
                "row_label": "Empleo",
                "cell_id": "c_r1c1",
            },
            {
                "row_start": 1,
                "row_end": 2,
                "col_start": 2,
                "col_end": 3,
                "text": "10,5",
                "raw_text": "10,5",
                "is_empty": False,
                "page_number": 95,
                "bbox": {"l": 300.0, "t": 230.0, "r": 400.0, "b": 280.0, "coord_origin": "TOPLEFT"},
                "column_header": False,
                "row_header": False,
                "header_path": ["2030"],
                "row_label": "Empleo",
                "cell_id": "c_r1c2",
            },
            {
                "row_start": 1,
                "row_end": 2,
                "col_start": 3,
                "col_end": 4,
                "text": "0,3",
                "raw_text": "0,3",
                "is_empty": False,
                "page_number": 95,
                "bbox": {"l": 400.0, "t": 230.0, "r": 500.0, "b": 280.0, "coord_origin": "TOPLEFT"},
                "column_header": False,
                "row_header": False,
                "header_path": ["Δ 2030-2025"],
                "row_label": "Empleo",
                "cell_id": "c_r1c3",
            },
            # Fila 2: Salarios (8,2 / 7,8 / −0,4 con Unicode minus)
            {
                "row_start": 2,
                "row_end": 3,
                "col_start": 0,
                "col_end": 1,
                "text": "Salarios",
                "raw_text": "Salarios",
                "is_empty": False,
                "page_number": 95,
                "bbox": {"l": 100.0, "t": 280.0, "r": 200.0, "b": 330.0, "coord_origin": "TOPLEFT"},
                "column_header": False,
                "row_header": True,
                "header_path": [],
                "row_label": "Salarios",
                "cell_id": "c_r2c0",
            },
            {
                "row_start": 2,
                "row_end": 3,
                "col_start": 1,
                "col_end": 2,
                "text": "8,2",
                "raw_text": "8,2",
                "is_empty": False,
                "page_number": 95,
                "bbox": {"l": 200.0, "t": 280.0, "r": 300.0, "b": 330.0, "coord_origin": "TOPLEFT"},
                "column_header": False,
                "row_header": False,
                "header_path": ["2025"],
                "row_label": "Salarios",
                "cell_id": "c_r2c1",
            },
            {
                "row_start": 2,
                "row_end": 3,
                "col_start": 2,
                "col_end": 3,
                "text": "7,8",
                "raw_text": "7,8",
                "is_empty": False,
                "page_number": 95,
                "bbox": {"l": 300.0, "t": 280.0, "r": 400.0, "b": 330.0, "coord_origin": "TOPLEFT"},
                "column_header": False,
                "row_header": False,
                "header_path": ["2030"],
                "row_label": "Salarios",
                "cell_id": "c_r2c2",
            },
            {
                "row_start": 2,
                "row_end": 3,
                "col_start": 3,
                "col_end": 4,
                "text": "−0,4",
                "raw_text": "−0,4",
                "is_empty": False,
                "page_number": 95,
                "bbox": {"l": 400.0, "t": 280.0, "r": 500.0, "b": 330.0, "coord_origin": "TOPLEFT"},
                "column_header": False,
                "row_header": False,
                "header_path": ["Δ 2030-2025"],
                "row_label": "Salarios",
                "cell_id": "c_r2c3",
            },
        ],
    }

    formatted_table = _format_table_as_text(table)
    page_text = f"Preámbulo del documento sintético sobre empleo y salarios.\n\n{formatted_table}\n\nTexto posterior con notas explicativas."

    bundle_page = {
        "page_number": 95,
        "width": 595.32,
        "height": 841.92,
        "layout_text": page_text,
        "blocks": [
            {
                "type": "text",
                "text": "Preámbulo del documento sintético sobre empleo y salarios.",
                "bbox": {"l": 100.0, "t": 100.0, "r": 500.0, "b": 150.0, "coord_origin": "TOPLEFT"},
            },
            {
                "type": "table",
                "text": formatted_table,
                "bbox": {"l": 100.0, "t": 200.0, "r": 500.0, "b": 350.0, "coord_origin": "TOPLEFT"},
            },
        ],
    }

    bundle = {
        "schema_version": "document-layout-v1",
        "metadata": {
            "doc_id": doc_id,
            "extraction_id": extraction_id,
            "source_file": source_file,
            "published_at": published_at,
        },
        "pages": [bundle_page],
        "tables": [table],
    }

    canonical_page = {
        "chunk_id": "page_95",
        "doc_id": doc_id,
        "source_file": source_file,
        "published_at": published_at,
        "page_number": 95,
        "start_char": 0,
        "end_char": len(page_text),
        "text": page_text,
    }

    return bundle, [canonical_page]


def _make_synthetic_bundle_with_nested_labels() -> tuple[dict, list[dict]]:
    """Genera un bundle sintético con etiquetas anidadas 'Modelizados' y 'No modelizados'."""
    bundle, pages = _make_synthetic_bundle_and_pages()
    t = bundle["tables"][0]
    t["caption_text"] = "CUADRO 1. MODELIZADOS Y NO MODELIZADOS (% PIB)"
    t["cells"][4]["text"] = "Modelizados"
    t["cells"][4]["raw_text"] = "Modelizados"
    t["cells"][4]["row_label"] = "Modelizados"
    for c in t["cells"][5:8]:
        c["row_label"] = "Modelizados"
    t["cells"][8]["text"] = "No modelizados"
    t["cells"][8]["raw_text"] = "No modelizados"
    t["cells"][8]["row_label"] = "No modelizados"
    for c in t["cells"][9:12]:
        c["row_label"] = "No modelizados"
    ft = _format_table_as_text(t)
    page_text = f"Preámbulo del documento sintético.\n\n{ft}\n\nTexto posterior."
    bundle["pages"][0]["layout_text"] = page_text
    bundle["pages"][0]["blocks"][0]["text"] = "Preámbulo del documento sintético."
    bundle["pages"][0]["blocks"][1]["text"] = ft
    pages[0]["text"] = page_text
    pages[0]["end_char"] = len(page_text)
    return bundle, pages


def _build_valid_fact(target_cell: dict, evidence_ids: list[str]) -> dict:
    """Helper para construir un hecho tipado válido con procedencia completa para una celda objetivo."""
    return {
        "cell_id": target_cell["cell_id"],
        "concept": target_cell["concept"],
        "header_path": target_cell["header_path"],
        "year": target_cell.get("year"),
        "ordered_delta_years": target_cell.get("ordered_delta_years"),
        "unit": target_cell["unit"],
        "measure": target_cell["measure"],
        "value": target_cell["expected_value"],
        "doc_id": target_cell["doc_id"],
        "source": target_cell["source"],
        "date": "2026-01-15",
        "page": target_cell["page"],
        "table_id": target_cell["table_id"],
        "extraction_id": target_cell["extraction_id"],
        "evidence_ids": copy.deepcopy(evidence_ids),
    }


def _load_pilot_packets() -> list[dict]:
    """Helper para cargar paquetes reales del piloto v4."""
    pilot_bundle_path = REPO_ROOT / "evaluation/document_layout_2026-10-27/pilot_v4/structured.json"
    pilot_pages_path = REPO_ROOT / "evaluation/layout_answer_effect_2026-10-27/prepared_v1/layout_pages.jsonl"
    bundle = json.loads(pilot_bundle_path.read_text(encoding="utf-8"))
    pages = [
        json.loads(line)
        for line in pilot_pages_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return build_table_packets(bundle, pages)


class TestTableGrounding(unittest.TestCase):
    """Batería exhaustiva de tests para tabla de evidencias y guardia numérica."""

    def test_packets_happy_path_level_and_delta(self):
        bundle, pages = _make_synthetic_bundle_and_pages()
        packets = build_table_packets(bundle, pages)

        self.assertEqual(len(packets), 2)
        p_empleo = next(p for p in packets if p["row_label"] == "Empleo")
        self.assertEqual(p_empleo["doc_id"], bundle["metadata"]["doc_id"])
        self.assertEqual(p_empleo["source"], bundle["metadata"]["source_file"])
        self.assertEqual(p_empleo["page"], 95)
        self.assertEqual(p_empleo["table_id"], "table_synth_cuadro1")

        cells = p_empleo["cells"]
        self.assertEqual(len(cells), 3)

        c2025 = next(c for c in cells if c["year"] == 2025)
        self.assertEqual(c2025["measure"], "level")
        self.assertEqual(c2025["unit"], "% PIB")
        self.assertEqual(c2025["value_decimal"], "10.1")
        self.assertEqual(c2025["header_path"], ["2025"])
        self.assertEqual(
            c2025["evidence_ids"],
            ["table_synth_cuadro1_caption", "table_synth_cuadro1_header", "table_synth_cuadro1_r1_row"],
        )

        cdelta = next(c for c in cells if c["measure"] == "delta")
        self.assertEqual(cdelta["ordered_delta_years"], [2030, 2025])
        self.assertEqual(cdelta["unit"], "puntos PIB")
        self.assertEqual(cdelta["value_decimal"], "0.3")

        p_sal = next(p for p in packets if p["row_label"] == "Salarios")
        c_sal_delta = next(c for c in p_sal["cells"] if c["measure"] == "delta")
        self.assertEqual(c_sal_delta["value_decimal"], "-0.4")

        for p in packets:
            self.assertLessEqual(len(p["evidence"]), 4)
            for ev in p["evidence"]:
                self.assertGreaterEqual(len(ev["quote"]), 12)
                self.assertLessEqual(len(ev["quote"]), 700)
                page_text = pages[0]["text"]
                self.assertEqual(page_text[ev["start_char"]:ev["end_char"]], ev["quote"])

            sum_quote_bytes = sum(len(ev["quote"].encode("utf-8")) for ev in p["evidence"])
            self.assertLessEqual(sum_quote_bytes, MAX_SUM_QUOTE_UTF8_BYTES)

    def test_pure_input_immutability(self):
        bundle, pages = _make_synthetic_bundle_and_pages()
        b_copy = copy.deepcopy(bundle)
        p_copy = copy.deepcopy(pages)

        packets = build_table_packets(bundle, pages)
        self.assertEqual(bundle, b_copy)
        self.assertEqual(pages, p_copy)

        query = "Según el cuadro 1, ¿qué porcentaje del PIB figura para Empleo en 2025?"
        req = parse_table_request(query, packets)
        tc = req["target_cells"][0]
        ev_ids = packets[0]["cells"][0]["evidence_ids"]
        fact = _build_valid_fact(tc, ev_ids)
        facts = [fact]
        facts_copy = copy.deepcopy(facts)
        guard_table_facts(facts, req, packets)
        self.assertEqual(facts, facts_copy)

    def test_missing_pages_outside_subset_excluded(self):
        bundle, pages = _make_synthetic_bundle_and_pages()
        t101 = copy.deepcopy(bundle["tables"][0])
        t101["table_id"] = "table_p101_extra"
        t101["page_numbers"] = [101]
        bundle["tables"].append(t101)

        packets = build_table_packets(bundle, pages)
        self.assertTrue(all(p["page"] == 95 for p in packets))
        self.assertFalse(any(p["table_id"] == "table_p101_extra" for p in packets))

    def test_reconstruction_and_literal_anchoring_validation(self):
        bundle, pages = _make_synthetic_bundle_and_pages()
        bundle["pages"][0]["blocks"][1]["text"] = "TEXTO_CORRUPTO_QUE_NO_COINCIDE"
        with self.assertRaises(ValueError):
            build_table_packets(bundle, pages)

        bundle, pages = _make_synthetic_bundle_and_pages()
        pages[0]["text"] = "Página sin la tabla contenida literalmente"
        with self.assertRaises(ValueError):
            build_table_packets(bundle, pages)

    def test_bbox_validation(self):
        bundle, pages = _make_synthetic_bundle_and_pages()
        bundle["tables"][0]["bbox"]["l"] = float("nan")
        with self.assertRaises(ValueError):
            build_table_packets(bundle, pages)

        bundle, pages = _make_synthetic_bundle_and_pages()
        bundle["tables"][0]["bbox"]["r"] = float("inf")
        with self.assertRaises(ValueError):
            build_table_packets(bundle, pages)

        bundle, pages = _make_synthetic_bundle_and_pages()
        bundle["tables"][0]["bbox"]["l"] = 450.0
        bundle["tables"][0]["bbox"]["r"] = 200.0
        with self.assertRaises(ValueError):
            build_table_packets(bundle, pages)

    def test_spanish_comma_unicode_minus_and_accents(self):
        self.assertEqual(parse_canonical_decimal("15,3"), "15.3")
        self.assertEqual(parse_canonical_decimal("0,3%"), "0.3")
        self.assertEqual(parse_canonical_decimal("−1,4"), "-1.4")
        self.assertEqual(parse_canonical_decimal("–0,5"), "-0.5")
        self.assertEqual(parse_canonical_decimal("—2,1"), "-2.1")
        self.assertEqual(parse_canonical_decimal("+0,3"), "0.3")
        self.assertEqual(parse_canonical_decimal("1.234,56"), "1234.56")
        self.assertEqual(parse_canonical_decimal("0,0"), "0.0")

    def test_reject_malformed_nonfinite_numbers(self):
        for bad in (float("nan"), float("inf"), float("-inf"), True, False, None, "nan", "inf", "null", "abc"):
            with self.assertRaises(ValueError):
                parse_canonical_decimal(bad)

    def test_table_packets_rejects_altered_row_header_metadata(self):
        bundle, pages = _make_synthetic_bundle_and_pages()
        # Alterar row_label en una celda de datos para contradecir la cabecera física "Empleo"
        bundle["tables"][0]["cells"][5]["row_label"] = "EtiquetaFalsa"
        with self.assertRaises(ValueError):
            build_table_packets(bundle, pages)

        # Alterar header_path en una celda de datos para contradecir la cabecera física ["2025"]
        bundle, pages = _make_synthetic_bundle_and_pages()
        bundle["tables"][0]["cells"][5]["header_path"] = ["CabeceraFalsa"]
        with self.assertRaises(ValueError):
            build_table_packets(bundle, pages)

    def test_parse_request_no_table_returns_none(self):
        bundle, pages = _make_synthetic_bundle_and_pages()
        packets = build_table_packets(bundle, pages)
        self.assertIsNone(parse_table_request("¿Cuál es la tasa de inflación en 2025?", packets))
        self.assertIsNone(parse_table_request("Explica el crecimiento del empleo", packets))

    def test_parse_request_table1_does_not_match_table10(self):
        bundle, pages = _make_synthetic_bundle_and_pages()
        bundle["tables"][0]["table_id"] = "table_synth_cuadro10"
        bundle["tables"][0]["caption_text"] = "CUADRO 10. EMPLEO Y SALARIOS (% PIB)"
        # Actualizar página
        bundle, pages = _make_synthetic_bundle_and_pages()
        bundle["tables"][0]["table_id"] = "table_synth_cuadro10"
        bundle["tables"][0]["caption_text"] = "CUADRO 10. EMPLEO Y SALARIOS (% PIB)"
        ft = _format_table_as_text(bundle["tables"][0])
        bundle["pages"][0]["layout_text"] = ft
        bundle["pages"][0]["blocks"][1]["text"] = ft
        pages[0]["text"] = ft
        packets = build_table_packets(bundle, pages)

        # Petición que busca cuadro 1 debe rechazar y no coincidir con cuadro 10 por substring
        req = parse_table_request("Según el cuadro 1, ¿qué porcentaje figura para Empleo en 2025?", packets)
        self.assertEqual(req["status"], "unsupported")
        self.assertIn("no encontrado", req["reason"])

    def test_parse_request_supported_level(self):
        bundle, pages = _make_synthetic_bundle_and_pages()
        packets = build_table_packets(bundle, pages)
        query = "Según el cuadro 1, ¿qué porcentaje del PIB figura para Empleo en 2025?"
        req = parse_table_request(query, packets)

        self.assertIsNotNone(req)
        self.assertEqual(req["status"], "supported")
        self.assertEqual(len(req["target_cells"]), 1)
        tc = req["target_cells"][0]
        self.assertEqual(tc["concept"], "Empleo")
        self.assertEqual(tc["year"], 2025)
        self.assertEqual(tc["measure"], "level")
        self.assertEqual(tc["unit"], "% PIB")
        self.assertEqual(tc["expected_value"], "10.1")

    def test_parse_request_supported_delta(self):
        bundle, pages = _make_synthetic_bundle_and_pages()
        packets = build_table_packets(bundle, pages)
        query = "En el cuadro 1, ¿qué variación publicada para Empleo corresponde a 2030 menos 2025, en puntos del PIB?"
        req = parse_table_request(query, packets)

        self.assertIsNotNone(req)
        self.assertEqual(req["status"], "supported")
        self.assertEqual(len(req["target_cells"]), 1)
        tc = req["target_cells"][0]
        self.assertEqual(tc["concept"], "Empleo")
        self.assertEqual(tc["ordered_delta_years"], [2030, 2025])
        self.assertEqual(tc["measure"], "delta")
        self.assertEqual(tc["unit"], "puntos PIB")
        self.assertEqual(tc["expected_value"], "0.3")

    def test_parse_request_published_delta_differs_from_subtraction(self):
        bundle, pages = _make_synthetic_bundle_and_pages()
        packets = build_table_packets(bundle, pages)
        query = "En el cuadro 1, ¿qué variación publicada para Empleo figura entre 2030 y 2025 en puntos del PIB?"
        req = parse_table_request(query, packets)

        self.assertEqual(req["status"], "supported")
        self.assertEqual(req["target_cells"][0]["expected_value"], "0.3")
        self.assertNotEqual(req["target_cells"][0]["expected_value"], "0.4")

    def test_parse_request_multi_year_support_and_no_pruning(self):
        bundle, pages = _make_synthetic_bundle_and_pages()
        packets = build_table_packets(bundle, pages)
        # Un concepto con dos años explícitos: debe soportar ambos sin podar el segundo
        query = "Según el cuadro 1, ¿cuáles son los porcentajes para Empleo en 2025 y 2030?"
        req = parse_table_request(query, packets)

        self.assertEqual(req["status"], "supported")
        self.assertEqual(len(req["target_cells"]), 2)
        y_list = [c["year"] for c in req["target_cells"]]
        self.assertEqual(y_list, [2025, 2030])

    def test_parse_request_multirow_conjunction(self):
        bundle, pages = _make_synthetic_bundle_and_pages()
        packets = build_table_packets(bundle, pages)
        query = "Según el cuadro 1, ¿qué porcentajes del PIB figuran para Empleo en 2025 y Salarios en 2030, respectivamente?"
        req = parse_table_request(query, packets)

        self.assertEqual(req["status"], "supported")
        self.assertEqual(len(req["target_cells"]), 2)
        c_emp = next(c for c in req["target_cells"] if c["concept"] == "Empleo")
        self.assertEqual(c_emp["year"], 2025)
        self.assertEqual(c_emp["expected_value"], "10.1")

        c_sal = next(c for c in req["target_cells"] if c["concept"] == "Salarios")
        self.assertEqual(c_sal["year"], 2030)
        self.assertEqual(c_sal["expected_value"], "7.8")

        # Multi-concepto donde un concepto carece de año en su cláusula debe ser rechazado
        bad_conj = "Según el cuadro 1, ¿qué porcentajes figuran para Empleo y Salarios en 2030?"
        req_bad = parse_table_request(bad_conj, packets)
        self.assertEqual(req_bad["status"], "unsupported")

    def test_parse_request_rejects_unsupported_qualifiers(self):
        bundle, pages = _make_synthetic_bundle_and_pages()
        packets = build_table_packets(bundle, pages)

        # Geográficos (Galicia, Cataluña, etc.)
        req_gal = parse_table_request("Según el cuadro 1, ¿cuál es el porcentaje de Empleo en Galicia en 2025?", packets)
        self.assertEqual(req_gal["status"], "unsupported")

        # Demográficos (mujeres)
        req_muj = parse_table_request("Según el cuadro 1, ¿cuál es el Empleo para mujeres en 2025?", packets)
        self.assertEqual(req_muj["status"], "unsupported")

        # Ratios / operaciones
        req_rat = parse_table_request("Según el cuadro 1, ¿cuál es la ratio de Salarios a Empleo en 2025?", packets)
        self.assertEqual(req_rat["status"], "unsupported")

    def test_guard_happy_path(self):
        bundle, pages = _make_synthetic_bundle_and_pages()
        packets = build_table_packets(bundle, pages)
        req = parse_table_request("Según el cuadro 1, ¿qué porcentaje del PIB figura para Empleo en 2025?", packets)
        tc = req["target_cells"][0]
        ev_ids = packets[0]["cells"][0]["evidence_ids"]

        facts = [_build_valid_fact(tc, ev_ids)]
        res = guard_table_facts(facts, req, packets)
        self.assertEqual(res["decision"], "accepted")
        self.assertEqual(res["reasons"], [])
        self.assertEqual(len(res["accepted_facts"]), 1)

    def test_guard_rejects_omitted_evidence_ids(self):
        bundle, pages = _make_synthetic_bundle_and_pages()
        packets = build_table_packets(bundle, pages)
        req = parse_table_request("Según el cuadro 1, ¿qué porcentaje del PIB figura para Empleo en 2025?", packets)
        tc = req["target_cells"][0]
        ev_ids = packets[0]["cells"][0]["evidence_ids"]

        fact = _build_valid_fact(tc, ev_ids)
        fact["evidence_ids"] = []
        res = guard_table_facts([fact], req, packets)
        self.assertEqual(res["decision"], "rejected")

    def test_guard_rejects_fake_and_unrelated_evidence_ids(self):
        bundle, pages = _make_synthetic_bundle_and_pages()
        packets = build_table_packets(bundle, pages)
        req = parse_table_request("Según el cuadro 1, ¿qué porcentaje del PIB figura para Empleo en 2025?", packets)
        tc = req["target_cells"][0]
        ev_ids = packets[0]["cells"][0]["evidence_ids"]

        # Fake ID
        fact_fake = _build_valid_fact(tc, ev_ids + ["table_synth_fake_ev_999"])
        res_fake = guard_table_facts([fact_fake], req, packets)
        self.assertEqual(res_fake["decision"], "rejected")

        # Unrelated ID de otra fila (fila 2 Salarios en vez de fila 1 Empleo)
        sal_row_id = "table_synth_cuadro1_r2_row"
        fact_unrelated = _build_valid_fact(tc, [ev_ids[0], ev_ids[1], sal_row_id])
        res_unr = guard_table_facts([fact_unrelated], req, packets)
        self.assertEqual(res_unr["decision"], "rejected")

    def test_guard_rejects_header_only_evidence_ids(self):
        bundle, pages = _make_synthetic_bundle_and_pages()
        packets = build_table_packets(bundle, pages)
        req = parse_table_request("Según el cuadro 1, ¿qué porcentaje del PIB figura para Empleo en 2025?", packets)
        tc = req["target_cells"][0]

        # Solo cabecera y título, falta la fila
        fact_header_only = _build_valid_fact(tc, ["table_synth_cuadro1_caption", "table_synth_cuadro1_header"])
        res = guard_table_facts([fact_header_only], req, packets)
        self.assertEqual(res["decision"], "rejected")
        self.assertTrue(any("omite la cita obligatoria de fila" in r for r in res["reasons"]))

    def test_guard_rejects_missing_provenance_fields(self):
        bundle, pages = _make_synthetic_bundle_and_pages()
        packets = build_table_packets(bundle, pages)
        req = parse_table_request("Según el cuadro 1, ¿qué porcentaje del PIB figura para Empleo en 2025?", packets)
        tc = req["target_cells"][0]
        ev_ids = packets[0]["cells"][0]["evidence_ids"]

        for field in ("doc_id", "source", "date", "page", "table_id", "extraction_id"):
            fact = _build_valid_fact(tc, ev_ids)
            del fact[field]
            res = guard_table_facts([fact], req, packets)
            self.assertEqual(res["decision"], "rejected", f"Debería rechazar al omitir {field}")
            self.assertTrue(any(field in r for r in res["reasons"]))

    def test_guard_rejects_wrong_header_order(self):
        bundle, pages = _make_synthetic_bundle_and_pages()
        packets = build_table_packets(bundle, pages)
        req = parse_table_request("Según el cuadro 1, ¿qué porcentaje del PIB figura para Empleo en 2025?", packets)
        tc = req["target_cells"][0]
        ev_ids = packets[0]["cells"][0]["evidence_ids"]

        fact = _build_valid_fact(tc, ev_ids)
        fact["header_path"] = ["2030", "2025"]  # Orden erróneo o inventado
        res = guard_table_facts([fact], req, packets)
        self.assertEqual(res["decision"], "rejected")
        self.assertTrue(any("header_path" in r for r in res["reasons"]))

    def test_guard_rejects_forged_request_expected_value(self):
        bundle, pages = _make_synthetic_bundle_and_pages()
        packets = build_table_packets(bundle, pages)
        req = parse_table_request("Según el cuadro 1, ¿qué porcentaje del PIB figura para Empleo en 2025?", packets)
        ev_ids = packets[0]["cells"][0]["evidence_ids"]

        # Falsificar expected_value en la petición
        req["target_cells"][0]["expected_value"] = "99.9"
        # Hecho que intenta concordar con la petición falsificada
        forged_fact = _build_valid_fact(req["target_cells"][0], ev_ids)
        forged_fact["value"] = "99.9"

        res = guard_table_facts([forged_fact], req, packets)
        self.assertEqual(res["decision"], "rejected")
        self.assertTrue(any("falsificada" in r.lower() or "no coincide" in r.lower() for r in res["reasons"]))

    def test_guard_rejects_extra_free_prose_keys(self):
        bundle, pages = _make_synthetic_bundle_and_pages()
        packets = build_table_packets(bundle, pages)
        req = parse_table_request("Según el cuadro 1, ¿qué porcentaje del PIB figura para Empleo en 2025?", packets)
        tc = req["target_cells"][0]
        ev_ids = packets[0]["cells"][0]["evidence_ids"]

        fact = _build_valid_fact(tc, ev_ids)
        fact["free_prose_explanation"] = "El empleo creció debido a políticas macroeconómicas."
        res = guard_table_facts([fact], req, packets)
        self.assertEqual(res["decision"], "rejected")
        self.assertTrue(any("desconocidas" in r for r in res["reasons"]))

    def test_guard_rejects_bool_as_int(self):
        bundle, pages = _make_synthetic_bundle_and_pages()
        packets = build_table_packets(bundle, pages)
        req = parse_table_request("Según el cuadro 1, ¿qué porcentaje del PIB figura para Empleo en 2025?", packets)
        tc = req["target_cells"][0]
        ev_ids = packets[0]["cells"][0]["evidence_ids"]

        fact_bool_page = _build_valid_fact(tc, ev_ids)
        fact_bool_page["page"] = True
        res_page = guard_table_facts([fact_bool_page], req, packets)
        self.assertEqual(res_page["decision"], "rejected")

        fact_bool_year = _build_valid_fact(tc, ev_ids)
        fact_bool_year["year"] = True
        res_year = guard_table_facts([fact_bool_year], req, packets)
        self.assertEqual(res_year["decision"], "rejected")

    def test_guard_rejects_computed_delta_instead_of_published(self):
        bundle, pages = _make_synthetic_bundle_and_pages()
        packets = build_table_packets(bundle, pages)
        req = parse_table_request("En el cuadro 1, ¿qué variación figura para Empleo entre 2030 y 2025?", packets)
        tc = req["target_cells"][0]
        ev_ids = packets[0]["cells"][2]["evidence_ids"]

        fact = _build_valid_fact(tc, ev_ids)
        fact["value"] = "0.4"  # Resta recalculada 10,5 - 10,1 = 0,4 en vez de 0,3
        res = guard_table_facts([fact], req, packets)
        self.assertEqual(res["decision"], "rejected")
        self.assertTrue(any("Valor numérico" in r for r in res["reasons"]))

    def test_render_table_facts_happy_and_rejected(self):
        bundle, pages = _make_synthetic_bundle_and_pages()
        packets = build_table_packets(bundle, pages)
        req = parse_table_request("Según el cuadro 1, ¿qué porcentaje del PIB figura para Empleo en 2025?", packets)
        tc = req["target_cells"][0]
        ev_ids = packets[0]["cells"][0]["evidence_ids"]

        valid_facts = [_build_valid_fact(tc, ev_ids)]
        claims = render_table_facts(valid_facts, req, packets)
        self.assertEqual(len(claims), 1)
        self.assertEqual(claims[0]["status"], "answered")
        self.assertIn("10.1 % PIB", claims[0]["text"])
        self.assertEqual(claims[0]["concept"], "Empleo")

        bad_facts = copy.deepcopy(valid_facts)
        bad_facts[0]["value"] = "99.9"
        bad_claims = render_table_facts(bad_facts, req, packets)
        self.assertEqual(bad_claims, [])

    def test_actual_pilot_v4_smoke_and_ten_queries(self):
        """Smoke test sobre los artefactos públicos reales del piloto v4 (sin PDF ni gold)."""
        pilot_bundle_path = REPO_ROOT / "evaluation/document_layout_2026-10-27/pilot_v4/structured.json"
        pilot_pages_path = REPO_ROOT / "evaluation/layout_answer_effect_2026-10-27/prepared_v1/layout_pages.jsonl"
        sec_queries_path = REPO_ROOT / "evaluation/layout_answer_effect_2026-10-27/secondary_queries.json"
        extra_queries_path = REPO_ROOT / "evaluation/table_grounding_2026-10-27/extra_queries.json"

        self.assertTrue(pilot_bundle_path.exists())
        self.assertTrue(pilot_pages_path.exists())
        self.assertTrue(sec_queries_path.exists())
        self.assertTrue(extra_queries_path.exists())

        bundle = json.loads(pilot_bundle_path.read_text(encoding="utf-8"))
        pages = [
            json.loads(line)
            for line in pilot_pages_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

        # 1. Empaquetado exitoso sobre datos reales
        packets = build_table_packets(bundle, pages)
        # En pages (95-99) solo está el Cuadro 9 de la página 95 (el Cuadro 10 está en p101 y se excluye)
        self.assertEqual(len(packets), 9)
        self.assertTrue(all(p["page"] == 95 for p in packets))
        self.assertTrue(all(p["table_id"] == "table_37a2d0c0cb8bdb90_p95_cuadro9" for p in packets))

        # Verificar integridad de citas y presupuesto de cada paquete real
        for p in packets:
            self.assertLessEqual(len(p["evidence"]), 4)
            sum_bytes = sum(len(ev["quote"].encode("utf-8")) for ev in p["evidence"])
            self.assertLessEqual(sum_bytes, MAX_SUM_QUOTE_UTF8_BYTES)
            for ev in p["evidence"]:
                self.assertGreaterEqual(len(ev["quote"]), 12)
                self.assertLessEqual(len(ev["quote"]), 700)
                # Anclaje literal exacto en la página 95
                p95_text = next(pg["text"] for pg in pages if pg["page_number"] == 95)
                self.assertEqual(p95_text[ev["start_char"]:ev["end_char"]], ev["quote"])

        # 2. Evaluación de las diez consultas públicas (sin acceder a respuestas de referencia)
        sec_cases = json.loads(sec_queries_path.read_text(encoding="utf-8"))["cases"]
        extra_cases = json.loads(extra_queries_path.read_text(encoding="utf-8"))["cases"]
        all_cases = sec_cases + extra_cases
        self.assertEqual(len(all_cases), 10)

        results_by_id = {}
        for c in all_cases:
            cid = c["id"]
            req = parse_table_request(c["query"], packets)
            results_by_id[cid] = req

        # t01: Pensiones 2041 -> supported
        self.assertEqual(results_by_id["t01"]["status"], "supported")
        self.assertEqual(results_by_id["t01"]["target_cells"][0]["concept"], "Pensiones")
        self.assertEqual(results_by_id["t01"]["target_cells"][0]["year"], 2041)
        self.assertEqual(results_by_id["t01"]["target_cells"][0]["unit"], "% PIB")

        # t02: Desempleo 2050 -> supported
        self.assertEqual(results_by_id["t02"]["status"], "supported")
        self.assertEqual(results_by_id["t02"]["target_cells"][0]["concept"], "Desempleo")
        self.assertEqual(results_by_id["t02"]["target_cells"][0]["year"], 2050)

        # t03: variación Pensiones 2070 menos 2050 -> supported
        self.assertEqual(results_by_id["t03"]["status"], "supported")
        self.assertEqual(results_by_id["t03"]["target_cells"][0]["ordered_delta_years"], [2070, 2050])
        self.assertEqual(results_by_id["t03"]["target_cells"][0]["unit"], "puntos PIB")

        # t04: variación Pensiones 2029 menos 2023 -> supported
        self.assertEqual(results_by_id["t04"]["status"], "supported")
        self.assertEqual(results_by_id["t04"]["target_cells"][0]["ordered_delta_years"], [2029, 2023])

        # t05: Intereses 2070 -> supported
        self.assertEqual(results_by_id["t05"]["status"], "supported")
        self.assertEqual(results_by_id["t05"]["target_cells"][0]["year"], 2070)

        # t06: Sanidad 2050 y Educación 2070 -> supported, 2 celdas
        self.assertEqual(results_by_id["t06"]["status"], "supported")
        self.assertEqual(len(results_by_id["t06"]["target_cells"]), 2)

        # u01: Intereses 2041 -> supported
        self.assertEqual(results_by_id["u01"]["status"], "supported")
        self.assertEqual(results_by_id["u01"]["target_cells"][0]["year"], 2041)

        # u02: Educación variación 2029 menos 2023 -> supported
        self.assertEqual(results_by_id["u02"]["status"], "supported")
        self.assertEqual(results_by_id["u02"]["target_cells"][0]["ordered_delta_years"], [2029, 2023])

        # u03: Desempleo en Cataluña en 2050 -> unsupported por restricción geográfica
        self.assertEqual(results_by_id["u03"]["status"], "unsupported")

        # u04: Intereses en 2032 -> unsupported por año no disponible
        self.assertEqual(results_by_id["u04"]["status"], "unsupported")

    def test_nested_row_labels_no_modelizados_vs_modelizados(self):
        bundle, pages = _make_synthetic_bundle_with_nested_labels()
        packets = build_table_packets(bundle, pages)

        # Solicitar 'No modelizados' debe seleccionar 'No modelizados' y NO 'Modelizados'
        q_no = "Según el cuadro 1, ¿qué porcentaje del PIB figura para No modelizados en 2025?"
        req_no = parse_table_request(q_no, packets)
        self.assertEqual(req_no["status"], "supported")
        self.assertEqual(len(req_no["target_cells"]), 1)
        self.assertEqual(req_no["target_cells"][0]["concept"], "No modelizados")
        self.assertEqual(req_no["target_cells"][0]["year"], 2025)

        # Solicitar 'Modelizados' debe seleccionar 'Modelizados' y NO 'No modelizados'
        q_mod = "Según el cuadro 1, ¿qué porcentaje del PIB figura para Modelizados en 2025?"
        req_mod = parse_table_request(q_mod, packets)
        self.assertEqual(req_mod["status"], "supported")
        self.assertEqual(len(req_mod["target_cells"]), 1)
        self.assertEqual(req_mod["target_cells"][0]["concept"], "Modelizados")

        # Solicitar 'No modelizados' con instrucción de negación de usuario debe ser rechazada
        q_neg = "Según el cuadro 1, ¿qué porcentaje del PIB figura para No modelizados en 2025? No uses la fila de No modelizados."
        req_neg = parse_table_request(q_neg, packets)
        self.assertEqual(req_neg["status"], "unsupported")

    def test_adversarial_unknown_location_and_population_qualifiers(self):
        bundle, pages = _make_synthetic_bundle_and_pages()
        packets = build_table_packets(bundle, pages)

        # Geografía inventada / desconocida
        q_geo1 = "Según el cuadro 1, ¿qué porcentaje del PIB figura para Empleo en 2025 en Andrómeda?"
        req_geo1 = parse_table_request(q_geo1, packets)
        self.assertEqual(req_geo1["status"], "unsupported")

        q_geo2 = "Según el cuadro 1, ¿qué porcentaje del PIB figura para Empleo en 2025 en Narnia?"
        req_geo2 = parse_table_request(q_geo2, packets)
        self.assertEqual(req_geo2["status"], "unsupported")

        # Población o demografía desconocida
        q_pop = "Según el cuadro 1, ¿qué porcentaje del PIB figura para Empleo en 2025 entre los jubilados?"
        req_pop = parse_table_request(q_pop, packets)
        self.assertEqual(req_pop["status"], "unsupported")

    def test_adversarial_extra_question_and_clauses(self):
        bundle, pages = _make_synthetic_bundle_and_pages()
        packets = build_table_packets(bundle, pages)

        q_extra = "Según el cuadro 1, ¿qué porcentaje del PIB figura para Empleo en 2025 y además cuál es el déficit público?"
        req_extra = parse_table_request(q_extra, packets)
        self.assertEqual(req_extra["status"], "unsupported")

    def test_adversarial_user_negation_instructions(self):
        bundle, pages = _make_synthetic_bundle_and_pages()
        packets = build_table_packets(bundle, pages)

        q_neg1 = "Según el cuadro 1, ¿qué porcentaje del PIB figura para Empleo en 2025? No uses la fila de Empleo."
        self.assertEqual(parse_table_request(q_neg1, packets)["status"], "unsupported")

        q_neg2 = "Según el cuadro 1, ¿qué porcentaje del PIB figura para Empleo en 2025, excluyendo las prestaciones contributivas?"
        self.assertEqual(parse_table_request(q_neg2, packets)["status"], "unsupported")

        q_neg3 = "Según el cuadro 1, ¿qué porcentaje del PIB figura para Empleo en 2025 sin incluir partidas extraordinarias?"
        self.assertEqual(parse_table_request(q_neg3, packets)["status"], "unsupported")

    def test_adversarial_multiple_deltas_all_or_unsupported(self):
        pilot_bundle_path = REPO_ROOT / "evaluation/document_layout_2026-10-27/pilot_v4/structured.json"
        pilot_pages_path = REPO_ROOT / "evaluation/layout_answer_effect_2026-10-27/prepared_v1/layout_pages.jsonl"
        bundle = json.loads(pilot_bundle_path.read_text(encoding="utf-8"))
        pages = [json.loads(line) for line in pilot_pages_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        packets = build_table_packets(bundle, pages)

        # Dos variaciones válidas publicadas para Pensiones: ambas deben ser retornadas (2 celdas objetivo)
        q_multi = "En el cuadro 9, ¿qué variaciones publicadas para Pensiones corresponden a 2070 menos 2050 y 2029 menos 2023, en puntos del PIB?"
        req_multi = parse_table_request(q_multi, packets)
        self.assertEqual(req_multi["status"], "supported")
        self.assertEqual(len(req_multi["target_cells"]), 2)
        d_years = [c["ordered_delta_years"] for c in req_multi["target_cells"]]
        self.assertEqual(d_years, [[2070, 2050], [2029, 2023]])

        # Si una de las dos variaciones no existe en la tabla, debe rechazar por completo (all or unsupported)
        q_bad_multi = "En el cuadro 9, ¿qué variaciones publicadas para Pensiones corresponden a 2070 menos 2050 y 2035 menos 2023, en puntos del PIB?"
        req_bad = parse_table_request(q_bad_multi, packets)
        self.assertEqual(req_bad["status"], "unsupported")

    def test_measure_unit_mismatch_rejected(self):
        bundle, pages = _make_synthetic_bundle_and_pages()
        packets = build_table_packets(bundle, pages)

        # Medida de nivel (porcentaje) solicitada con años de variación
        q_mismatch1 = "Según el cuadro 1, ¿qué porcentaje del PIB figura para Empleo en 2030 menos 2025?"
        self.assertEqual(parse_table_request(q_mismatch1, packets)["status"], "unsupported")

        # Medida de variación solicitada con año de nivel único
        q_mismatch2 = "En el cuadro 1, ¿qué variación publicada para Empleo figura en 2025?"
        self.assertEqual(parse_table_request(q_mismatch2, packets)["status"], "unsupported")

    def test_adversarial_operators_and_extra_digits(self):
        bundle, pages = _make_synthetic_bundle_and_pages()
        packets = build_table_packets(bundle, pages)

        # Operador de ratio/división '/'
        q_slash = "Según el cuadro 1, ¿cuál es Salarios / Empleo en 2025?"
        self.assertEqual(parse_table_request(q_slash, packets)["status"], "unsupported")

        # Dígitos adicionales no reconocidos como año
        q_digits = "Según el cuadro 1, ¿qué porcentaje figura para Empleo en 2025 con valor 2100?"
        self.assertEqual(parse_table_request(q_digits, packets)["status"], "unsupported")

    def test_deduplicate_repeated_years_or_targets(self):
        bundle, pages = _make_synthetic_bundle_and_pages()
        packets = build_table_packets(bundle, pages)

        # Repetición del mismo año 2025 dos veces debe deduplicar a 1 celda objetivo coherente
        q_dup = "Según el cuadro 1, ¿cuáles son los porcentajes para Empleo en 2025 y 2025?"
        req_dup = parse_table_request(q_dup, packets)
        self.assertEqual(req_dup["status"], "supported")
        self.assertEqual(len(req_dup["target_cells"]), 1)
        self.assertEqual(req_dup["target_cells"][0]["year"], 2025)

    def test_guard_safely_rejects_malformed_and_nonfinite_forged_expected_value(self):
        bundle, pages = _make_synthetic_bundle_and_pages()
        packets = build_table_packets(bundle, pages)
        req = parse_table_request("Según el cuadro 1, ¿qué porcentaje del PIB figura para Empleo en 2025?", packets)
        ev_ids = packets[0]["cells"][0]["evidence_ids"]
        fact = _build_valid_fact(req["target_cells"][0], ev_ids)

        for bad_ev in ("bad_decimal_str", "NaN", float("nan"), float("inf"), float("-inf"), 10.1):
            req_forged = copy.deepcopy(req)
            req_forged["target_cells"][0]["expected_value"] = bad_ev
            res = guard_table_facts([fact], req_forged, packets)
            self.assertEqual(res["decision"], "rejected", f"Debería rechazar de forma segura con {bad_ev!r}")
            self.assertTrue(any("falsificada" in r.lower() or "inválido" in r.lower() or "malformado" in r.lower() for r in res["reasons"]))

    def test_guard_financial_facts_decimal_canonicalization_and_json_dumps(self):
        bundle, pages = _make_synthetic_bundle_and_pages()
        packets = build_table_packets(bundle, pages)
        req = parse_table_request("Según el cuadro 1, ¿qué porcentaje del PIB figura para Empleo en 2025?", packets)
        tc = req["target_cells"][0]
        ev_ids = packets[0]["cells"][0]["evidence_ids"]

        # Entrada con Decimal exacto
        fact = _build_valid_fact(tc, ev_ids)
        fact["value"] = Decimal("10.1")
        facts = [fact]

        res = guard_table_facts(facts, req, packets)
        self.assertEqual(res["decision"], "accepted")
        self.assertEqual(len(res["accepted_facts"]), 1)

        # Comprobar inmutabilidad de la entrada: el hecho original conserva Decimal("10.1")
        self.assertIsInstance(facts[0]["value"], Decimal)

        # Comprobar canonicalización de la salida a string Decimal serializable
        accepted_val = res["accepted_facts"][0]["value"]
        self.assertIsInstance(accepted_val, str)
        self.assertEqual(accepted_val, "10.1")

        # Comprobar serialización directa a JSON sin excepciones
        dumped = json.dumps(res["accepted_facts"])
        self.assertIn('"value": "10.1"', dumped)

    def test_guard_rejects_binary_floats_in_facts(self):
        bundle, pages = _make_synthetic_bundle_and_pages()
        packets = build_table_packets(bundle, pages)
        req = parse_table_request("Según el cuadro 1, ¿qué porcentaje del PIB figura para Empleo en 2025?", packets)
        tc = req["target_cells"][0]
        ev_ids = packets[0]["cells"][0]["evidence_ids"]

        fact = _build_valid_fact(tc, ev_ids)
        fact["value"] = 10.1  # Float binario prohibido
        res = guard_table_facts([fact], req, packets)
        self.assertEqual(res["decision"], "rejected")
        self.assertTrue(any("float" in r.lower() for r in res["reasons"]))

    def test_adversarial_why_question_rejected(self):
        bundle, pages = _make_synthetic_bundle_and_pages()
        packets = build_table_packets(bundle, pages)

        # Preguntas causales o explicativas con 'por qué' deben ser rechazadas
        q_why1 = "Según el cuadro 1, ¿por qué el gasto en Empleo en 2025 es 10,1?"
        req_why1 = parse_table_request(q_why1, packets)
        self.assertEqual(req_why1["status"], "unsupported")

        q_why2 = "Según el cuadro 1, por qué la cifra de Salarios cambia en 2030?"
        req_why2 = parse_table_request(q_why2, packets)
        self.assertEqual(req_why2["status"], "unsupported")

    def test_adversarial_count_vs_percent_rejected(self):
        bundle, pages = _make_synthetic_bundle_and_pages()
        packets = build_table_packets(bundle, pages)

        # Preguntas de conteo ('cuántas' / 'cuántos') en lugar de lecturas numéricas / porcentajes
        q_count1 = "Según el cuadro 1, ¿cuántas pensiones o empleos hay en 2025?"
        req_count1 = parse_table_request(q_count1, packets)
        self.assertEqual(req_count1["status"], "unsupported")

        q_count2 = "Según el cuadro 1, ¿cuántos salarios se pagaron en 2030?"
        req_count2 = parse_table_request(q_count2, packets)
        self.assertEqual(req_count2["status"], "unsupported")

    def test_unicode_nfd_and_accent_variants(self):
        bundle, pages = _make_synthetic_bundle_and_pages()
        packets = build_table_packets(bundle, pages)

        # Variante NFD (acentos descompuestos en Unicode)
        q_nfd = unicodedata.normalize("NFD", "Según el cuadro 1, ¿qué porcentaje del PIB figura para Empleo en 2025?")
        req_nfd = parse_table_request(q_nfd, packets)
        self.assertEqual(req_nfd["status"], "supported")
        self.assertEqual(len(req_nfd["target_cells"]), 1)
        self.assertEqual(req_nfd["target_cells"][0]["concept"], "Empleo")

        # Variante sin signo de apertura '¿'
        q_no_open = "Según el cuadro 1, qué porcentaje del PIB figura para Empleo en 2025?"
        req_no_open = parse_table_request(q_no_open, packets)
        self.assertEqual(req_no_open["status"], "supported")

        # Variante sin tilde en interrogativo ('que' en vez de 'qué')
        q_no_acc = "Según el cuadro 1, ¿que porcentaje del PIB figura para Empleo en 2025?"
        req_no_acc = parse_table_request(q_no_acc, packets)
        self.assertEqual(req_no_acc["status"], "supported")

        # Variante interrogativa 'cual es'
        q_cual = "Según el cuadro 1, ¿cual es el porcentaje para Empleo en 2025?"
        req_cual = parse_table_request(q_cual, packets)
        self.assertEqual(req_cual["status"], "supported")

    def test_adversarial_multiconcept_extra_unknown_year_rejected(self):
        bundle, pages = _make_synthetic_bundle_and_pages()
        packets = build_table_packets(bundle, pages)

        # Caso sintético con año desconocido 2099 tras última fila
        q_synth = "Según el cuadro 1, ¿qué porcentajes del PIB figuran para Empleo en 2025 y Salarios en 2030 y 2099?"
        req_synth = parse_table_request(q_synth, packets)
        self.assertEqual(req_synth["status"], "unsupported")

        # Caso real del piloto Cuadro 9
        pilot_packets = _load_pilot_packets()
        q_pilot = "Según el cuadro 9, ¿qué porcentajes del PIB figuran para Sanidad en 2050 y Educación en 2070 y 2099?"
        req_pilot = parse_table_request(q_pilot, pilot_packets)
        self.assertEqual(req_pilot["status"], "unsupported")

    def test_adversarial_multiconcept_extra_known_year_orphan_rejected(self):
        bundle, pages = _make_synthetic_bundle_and_pages()
        packets = build_table_packets(bundle, pages)

        # Caso sintético con año conocido 2025 huérfano tras última fila
        q_synth = "Según el cuadro 1, ¿qué porcentajes del PIB figuran para Empleo en 2025 y Salarios en 2030 y 2025?"
        req_synth = parse_table_request(q_synth, packets)
        self.assertEqual(req_synth["status"], "unsupported")

        # Caso real del piloto Cuadro 9 con año conocido 2023 huérfano tras última fila
        pilot_packets = _load_pilot_packets()
        q_pilot = "Según el cuadro 9, ¿qué porcentajes del PIB figuran para Sanidad en 2050 y Educación en 2070 y 2023?"
        req_pilot = parse_table_request(q_pilot, pilot_packets)
        self.assertEqual(req_pilot["status"], "unsupported")

    def test_adversarial_multiconcept_extra_sameyear_occurrence_rejected(self):
        bundle, pages = _make_synthetic_bundle_and_pages()
        packets = build_table_packets(bundle, pages)

        # Repetición adicional del mismo año en primera fila
        q1 = "Según el cuadro 1, ¿qué porcentajes del PIB figuran para Empleo en 2025 y 2025 y Salarios en 2030?"
        req1 = parse_table_request(q1, packets)
        self.assertEqual(req1["status"], "unsupported")

        # Repetición adicional del mismo año en segunda fila
        q2 = "Según el cuadro 1, ¿qué porcentajes del PIB figuran para Empleo en 2025 y Salarios en 2030 y 2030?"
        req2 = parse_table_request(q2, packets)
        self.assertEqual(req2["status"], "unsupported")

    def test_adversarial_mixed_level_and_delta_rejected(self):
        bundle, pages = _make_synthetic_bundle_and_pages()
        packets = build_table_packets(bundle, pages)

        # Fila única combinando nivel y variación temporal
        q_single = "Según el cuadro 1, ¿qué datos figuran para Empleo en 2025 y variación 2030 menos 2025 en puntos del PIB?"
        req_single = parse_table_request(q_single, packets)
        self.assertEqual(req_single["status"], "unsupported")

        # Sonda real del piloto Cuadro 9 combinando nivel 2023 y delta 2029-2023
        pilot_packets = _load_pilot_packets()
        q_pilot = "Según el cuadro 9, ¿qué datos figuran para Pensiones en 2023 y variación 2029 menos 2023 en puntos del PIB?"
        req_pilot = parse_table_request(q_pilot, pilot_packets)
        self.assertEqual(req_pilot["status"], "unsupported")

        # Multi-fila combinando nivel para una fila y delta para otra
        q_multi = "Según el cuadro 1, ¿qué datos figuran para Empleo en 2025 y variación de Salarios 2030 menos 2025 en puntos del PIB?"
        req_multi = parse_table_request(q_multi, packets)
        self.assertEqual(req_multi["status"], "unsupported")

        # Multi-concepto con variación temporal
        q_delta_multi = "Según el cuadro 1, ¿qué variaciones publicadas para Empleo y Salarios corresponden a 2030 menos 2025 en puntos del PIB?"
        req_delta_multi = parse_table_request(q_delta_multi, packets)
        self.assertEqual(req_delta_multi["status"], "unsupported")

    def test_positive_multiconcept_two_rows_same_year_supported(self):
        bundle, pages = _make_synthetic_bundle_and_pages()
        packets = build_table_packets(bundle, pages)

        # Dos filas distintas con el mismo año 2025: debe resolver 2 celdas objetivo sin podas ni deduplicación errónea
        q_same = "Según el cuadro 1, ¿qué porcentajes del PIB figuran para Empleo en 2025 y Salarios en 2025?"
        req_same = parse_table_request(q_same, packets)
        self.assertEqual(req_same["status"], "supported")
        self.assertEqual(len(req_same["target_cells"]), 2)
        c_emp = next(c for c in req_same["target_cells"] if c["concept"] == "Empleo")
        self.assertEqual(c_emp["year"], 2025)
        self.assertEqual(c_emp["expected_value"], "10.1")
        c_sal = next(c for c in req_same["target_cells"] if c["concept"] == "Salarios")
        self.assertEqual(c_sal["year"], 2025)
        self.assertEqual(c_sal["expected_value"], "8.2")

        # Caso en datos reales del piloto Cuadro 9 (Sanidad y Educación en 2050)
        pilot_packets = _load_pilot_packets()
        q_pilot = "Según el cuadro 9, ¿qué porcentajes del PIB figuran para Sanidad en 2050 y Educación en 2050?"
        req_pilot = parse_table_request(q_pilot, pilot_packets)
        self.assertEqual(req_pilot["status"], "supported")
        self.assertEqual(len(req_pilot["target_cells"]), 2)
        c_san = next(c for c in req_pilot["target_cells"] if c["concept"] == "Sanidad")
        self.assertEqual(c_san["year"], 2050)
        self.assertEqual(c_san["expected_value"], "8.0")
        c_edu = next(c for c in req_pilot["target_cells"] if c["concept"] == "Educación")
        self.assertEqual(c_edu["year"], 2050)
        self.assertEqual(c_edu["expected_value"], "3.5")

    def test_positive_multiconcept_two_rows_different_years_supported(self):
        bundle, pages = _make_synthetic_bundle_and_pages()
        packets = build_table_packets(bundle, pages)

        q_diff = "Según el cuadro 1, ¿qué porcentajes del PIB figuran para Empleo en 2025 y Salarios en 2030?"
        req_diff = parse_table_request(q_diff, packets)
        self.assertEqual(req_diff["status"], "supported")
        self.assertEqual(len(req_diff["target_cells"]), 2)
        c_emp = next(c for c in req_diff["target_cells"] if c["concept"] == "Empleo")
        self.assertEqual(c_emp["year"], 2025)
        self.assertEqual(c_emp["expected_value"], "10.1")
        c_sal = next(c for c in req_diff["target_cells"] if c["concept"] == "Salarios")
        self.assertEqual(c_sal["year"], 2030)
        self.assertEqual(c_sal["expected_value"], "7.8")


class TestSeniorCoreAuditRegressions(unittest.TestCase):
    """Pruebas de regresión independientes y adversariales para los 4 hallazgos de auditoría senior."""

    def test_p1_unit_detection_percent_alone_does_not_establish_pib(self):
        # 1. '%' solo en caption NO establece denominador PIB
        y_lvl, d_lvl, u_lvl, m_lvl = _infer_column_meta(["2025"], "CUADRO 1. EMPLEO Y SALARIOS (%)")
        self.assertEqual(y_lvl, 2025)
        self.assertIsNone(d_lvl)
        self.assertNotEqual(u_lvl, "% PIB")
        self.assertEqual(u_lvl, "%")
        self.assertEqual(m_lvl, "level")

        y_d, d_d, u_d, m_d = _infer_column_meta(["Δ 2030-2025"], "CUADRO 1. EMPLEO Y SALARIOS (%)")
        self.assertIsNone(y_d)
        self.assertEqual(d_d, [2030, 2025])
        self.assertNotEqual(u_d, "puntos PIB")
        self.assertEqual(u_d, "puntos")
        self.assertEqual(m_d, "delta")

        # 2. Denominador no-PIB explícito (% población)
        y_pob, _, u_pob, _ = _infer_column_meta(["2025"], "CUADRO 1. POBLACIÓN ACTIVA (% población)")
        self.assertNotEqual(u_pob, "% PIB")
        self.assertEqual(u_pob, "% población")

        _, d_pob, ud_pob, _ = _infer_column_meta(["Δ 2030-2025"], "CUADRO 1. POBLACIÓN ACTIVA (% población)")
        self.assertNotEqual(ud_pob, "puntos PIB")
        self.assertEqual(ud_pob, "puntos población")

        # 3. Control positivo con PIB explícito grounded (incluyendo espacios '( % PIB )')
        y_pos, _, u_pos, _ = _infer_column_meta(["2025"], "CUADRO 1. GASTO ( % PIB )")
        self.assertEqual(u_pos, "% PIB")

        _, d_pos, ud_pos, _ = _infer_column_meta(["Δ 2030-2025"], "CUADRO 1. GASTO ( % PIB )")
        self.assertEqual(ud_pos, "puntos PIB")

        # Sondas directas confirmadas por auditoría senior (P1 unit scope probes):
        # a) 'CUADRO 1. PIB (millones de euros)' -> 'millones de euros', NUNCA '% PIB'
        y_eur, _, u_eur, _ = _infer_column_meta(["2025"], "CUADRO 1. PIB (millones de euros)")
        self.assertEqual(y_eur, 2025)
        self.assertEqual(u_eur, "millones de euros")
        self.assertNotEqual(u_eur, "% PIB")

        # b) 'CUADRO 1. Pensiones y PIB (% población)' -> '% población', NUNCA '% PIB'
        y_pen_pob, _, u_pen_pob, _ = _infer_column_meta(["2025"], "CUADRO 1. Pensiones y PIB (% población)")
        self.assertEqual(y_pen_pob, 2025)
        self.assertEqual(u_pen_pob, "% población")
        self.assertNotEqual(u_pen_pob, "% PIB")

        # c) 'CUADRO 1. Pensiones (% PIB)' -> '% PIB'
        y_pen_pib, _, u_pen_pib, _ = _infer_column_meta(["2025"], "CUADRO 1. Pensiones (% PIB)")
        self.assertEqual(y_pen_pib, 2025)
        self.assertEqual(u_pen_pib, "% PIB")

        # d) 'CUADRO 1. Pensiones (% del PIB)' -> '% PIB'
        y_del_pib, _, u_del_pib, _ = _infer_column_meta(["2025"], "CUADRO 1. Pensiones (% del PIB)")
        self.assertEqual(y_del_pib, 2025)
        self.assertEqual(u_del_pib, "% PIB")

        # e) Delta con '% del PIB' -> 'puntos PIB'
        _, d_del_pib, ud_del_pib, _ = _infer_column_meta(["Δ 2030-2025"], "CUADRO 1. Pensiones (% del PIB)")
        self.assertEqual(ud_del_pib, "puntos PIB")

        # Sondas bare_percent_unit_probe.json (sin paréntesis y con 'en'):
        # bare '% del PIB': 'CUADRO 1. GASTO % del PIB' -> '% PIB'
        y_bare_del, _, u_bare_del, _ = _infer_column_meta(["2025"], "CUADRO 1. GASTO % del PIB")
        self.assertEqual(y_bare_del, 2025)
        self.assertEqual(u_bare_del, "% PIB")

        # bare '% PIB': 'CUADRO 1. GASTO % PIB' -> '% PIB'
        y_bare_pib, _, u_bare_pib, _ = _infer_column_meta(["2025"], "CUADRO 1. GASTO % PIB")
        self.assertEqual(y_bare_pib, 2025)
        self.assertEqual(u_bare_pib, "% PIB")

        # 'en % del PIB': 'CUADRO 1. GASTO en % del PIB' -> '% PIB'
        y_en_del, _, u_en_del, _ = _infer_column_meta(["2025"], "CUADRO 1. GASTO en % del PIB")
        self.assertEqual(y_en_del, 2025)
        self.assertEqual(u_en_del, "% PIB")

        # Delta inheritance con bare '% del PIB' y bare '% PIB'
        _, d_b_del, ud_b_del, _ = _infer_column_meta(["Δ 2030-2025"], "CUADRO 1. GASTO % del PIB")
        self.assertEqual(ud_b_del, "puntos PIB")

        _, d_b_pib, ud_b_pib, _ = _infer_column_meta(["Δ 2030-2025"], "CUADRO 1. GASTO % PIB")
        self.assertEqual(ud_b_pib, "puntos PIB")

        # f) Conflicto en el mismo título entre denominadores (falla cerrado con ValueError)
        with self.assertRaises(ValueError):
            _infer_column_meta(["2025"], "CUADRO 1. Pensiones (% PIB) y Salarios (% población)")

        # g) Conflicto en el mismo título entre moneda y porcentaje (falla cerrado con ValueError)
        with self.assertRaises(ValueError):
            _infer_column_meta(["2025"], "CUADRO 1. PIB (millones de euros) y Pensiones (% PIB)")

        # h) Conflicto entre título y cabecera (moneda vs porcentaje)
        with self.assertRaises(ValueError):
            _infer_column_meta(["2025 (% PIB)"], "CUADRO 1. PIB (millones de euros)")

        with self.assertRaises(ValueError):
            _infer_column_meta(["2025 (millones de euros)"], "CUADRO 1. GASTO (% PIB)")

        # 4. Denominadores conflictivos entre caption y header deben rechazar con ValueError
        with self.assertRaises(ValueError):
            _infer_column_meta(["2025 (% población)"], "CUADRO 1. GASTO (% PIB)")

        with self.assertRaises(ValueError):
            _infer_column_meta(["2025 (% PIB)"], "CUADRO 1. POBLACIÓN ACTIVA (% población)")

        # 5. Integración con guard: petición pidiendo PIB sobre tabla con '%' es unsupported
        bundle, pages = _make_synthetic_bundle_and_pages()
        t = bundle["tables"][0]
        t["caption_text"] = "CUADRO 1. EMPLEO Y SALARIOS (%)"
        ft = _format_table_as_text(t)
        p_text = f"Preámbulo.\n\n{ft}\n\nNotas."
        bundle["pages"][0]["layout_text"] = p_text
        bundle["pages"][0]["blocks"][1]["text"] = ft
        pages[0]["text"] = p_text
        pages[0]["end_char"] = len(p_text)

        pkts = build_table_packets(bundle, pages)
        self.assertEqual(pkts[0]["cells"][0]["unit"], "%")

        # Consulta pidiendo PIB sobre tabla de '%' debe ser rechazada
        req_pib = parse_table_request("Según el cuadro 1, ¿cuál es el porcentaje del PIB para Empleo en 2025?", pkts)
        self.assertEqual(req_pib["status"], "unsupported")

        # Fact reclamando falsamente '% PIB' es rechazado por la guardia
        req_valid = parse_table_request("Según el cuadro 1, ¿cuál es el porcentaje para Empleo en 2025?", pkts)
        self.assertEqual(req_valid["status"], "supported")
        tc = req_valid["target_cells"][0]
        self.assertEqual(tc["unit"], "%")
        fact_forged_unit = _build_valid_fact(tc, pkts[0]["cells"][0]["evidence_ids"])
        fact_forged_unit["unit"] = "% PIB"
        res_guard = guard_table_facts([fact_forged_unit], req_valid, pkts)
        self.assertEqual(res_guard["decision"], "rejected")

    def test_p1_regression_gdp_millions_euros_full_pipeline(self):
        """Regresión completa build_table_packets -> parse -> guard para PIB en millones de euros."""
        bundle, pages = _make_synthetic_bundle_and_pages()
        t = bundle["tables"][0]
        t["caption_text"] = "CUADRO 1. PIB (millones de euros)"
        ft = _format_table_as_text(t)
        p_text = f"Preámbulo.\n\n{ft}\n\nNotas."
        bundle["pages"][0]["layout_text"] = p_text
        bundle["pages"][0]["blocks"][1]["text"] = ft
        pages[0]["text"] = p_text
        pages[0]["end_char"] = len(p_text)

        pkts = build_table_packets(bundle, pages)
        self.assertEqual(pkts[0]["cells"][0]["unit"], "millones de euros")
        self.assertNotEqual(pkts[0]["cells"][0]["unit"], "% PIB")
        self.assertNotEqual(pkts[0]["cells"][0]["unit"], "%")

        # 1. Consulta solicitando % PIB sobre tabla de millones de euros debe ser unsupported
        req_pib = parse_table_request("Según el cuadro 1, ¿cuál es el porcentaje del PIB para Empleo en 2025?", pkts)
        self.assertEqual(req_pib["status"], "unsupported")
        self.assertIn("millones de euros", req_pib["reason"])

        # 2. Consultas solicitando porcentaje (palabra 'porcentaje', 'porcentajes', bare '%' entre espacios, '%' con puntuación y fullwidth '％') sin PIB sobre tabla monetaria DEBEN ser unsupported
        tc_syn = _build_target_cell(pkts[0], pkts[0]["cells"][0])
        fact_synthetic = _build_valid_fact(tc_syn, pkts[0]["cells"][0]["evidence_ids"])
        for q_pct_variant in [
            "Según el cuadro 1, ¿cuál es el porcentaje para Empleo en 2025?",
            "Según el cuadro 1, ¿qué porcentajes figuran para Empleo en 2025?",
            "Según el cuadro 1, ¿qué % figura para Empleo en 2025?",
            "Según el cuadro 1, ¿qué % para Empleo en 2025?",
            "Según el cuadro 1, ¿qué ％ figura para Empleo en 2025?",
        ]:
            req_pct = parse_table_request(q_pct_variant, pkts)
            self.assertEqual(req_pct["status"], "unsupported", f"Fallo al rechazar consulta porcentual en tabla monetaria: {q_pct_variant}")
            self.assertIn("monetarias", req_pct["reason"].lower())
            res_pct = guard_table_facts([fact_synthetic], req_pct, pkts)
            self.assertEqual(res_pct["decision"], "rejected", f"Guardia no rechazó petición no soportada para: {q_pct_variant}")

        # 3. Consulta pidiendo 'puntos' sobre tabla monetaria DEBE ser unsupported
        req_pts = parse_table_request("En el cuadro 1, ¿cuál es la variación en puntos para Empleo entre 2030 y 2025?", pkts)
        self.assertEqual(req_pts["status"], "unsupported")
        self.assertIn("monetarias", req_pts["reason"].lower())

        # 4. Control positivo: consulta solicitando el dato / valor monetario es supported y aceptada
        req_val = parse_table_request("Según el cuadro 1, ¿cuál es el dato para Empleo en 2025?", pkts)
        self.assertEqual(req_val["status"], "supported")
        tc = req_val["target_cells"][0]
        self.assertEqual(tc["unit"], "millones de euros")
        fact_valid = _build_valid_fact(tc, pkts[0]["cells"][0]["evidence_ids"])
        res_guard_valid = guard_table_facts([fact_valid], req_val, pkts)
        self.assertEqual(res_guard_valid["decision"], "accepted")

        # 5. Control positivo con importe explícito en millones de euros
        req_curr = parse_table_request("Según el cuadro 1, ¿cuál es el importe en millones de euros para Empleo en 2025?", pkts)
        self.assertEqual(req_curr["status"], "supported")
        tc_curr = req_curr["target_cells"][0]
        self.assertEqual(tc_curr["unit"], "millones de euros")
        fact_curr = _build_valid_fact(tc_curr, pkts[0]["cells"][0]["evidence_ids"])
        res_curr = guard_table_facts([fact_curr], req_curr, pkts)
        self.assertEqual(res_curr["decision"], "accepted")

        # 6. Hecho reclamando falsamente '% PIB' o unidad equivocada es rechazado por la guardia
        fact_wrong_unit = _build_valid_fact(tc, pkts[0]["cells"][0]["evidence_ids"])
        fact_wrong_unit["unit"] = "% PIB"
        res_guard = guard_table_facts([fact_wrong_unit], req_val, pkts)
        self.assertEqual(res_guard["decision"], "rejected")
        self.assertTrue(any("unidad" in r.lower() for r in res_guard["reasons"]))

    def test_p1_regression_currency_symbol_and_family_matrix_full_pipeline(self):
        """Regresión completa build_table_packets -> parse -> guard para matriz de moneda (€, euros, millones de €) y familia de unidades."""
        # Base válida independiente: tabla sintética con % PIB
        bundle, pages = _make_synthetic_bundle_and_pages()
        pkts_pct = build_table_packets(bundle, pages)
        self.assertEqual(pkts_pct[0]["cells"][0]["unit"], "% PIB")
        tc_pct_empleo = _build_target_cell(pkts_pct[0], pkts_pct[0]["cells"][0])
        fact_pct_valid = _build_valid_fact(tc_pct_empleo, pkts_pct[0]["cells"][0]["evidence_ids"])

        # 1. Matriz de contraejemplos de moneda contra tabla de % PIB (deben ser unsupported y guard rechazado):
        # a) Bare € spaced
        # b) Puntuación / adyacencia gramatical (€)
        # c) Palabra 'euros' y 'euro'
        # d) Compuesto 'millones de €' y 'miles de €'
        currency_queries_on_pct = [
            "Según el cuadro 1, ¿qué € figura para Empleo en 2025?",
            "Según el cuadro 1, ¿cuál es el importe (€) para Empleo en 2025?",
            "Según el cuadro 1, ¿cuál es el importe en euros para Empleo en 2025?",
            "Según el cuadro 1, ¿qué euro figura para Empleo en 2025?",
            "Según el cuadro 1, ¿cuál es el importe en millones de € para Empleo en 2025?",
            "Según el cuadro 1, ¿cuál es el importe en miles de € para Empleo en 2025?",
        ]
        for q_curr in currency_queries_on_pct:
            req = parse_table_request(q_curr, pkts_pct)
            self.assertEqual(req["status"], "unsupported", f"Fallo al rechazar consulta monetaria en tabla % PIB: {q_curr}")
            self.assertIn("importe monetario", req["reason"].lower())
            self.assertIn("% pib", req["reason"].lower())
            res = guard_table_facts([fact_pct_valid], req, pkts_pct)
            self.assertEqual(res["decision"], "rejected", f"Guardia no rechazó petición monetaria unsupported: {q_curr}")

        # 2. Contraejemplo con datos reales del piloto: Cuadro 9 (% PIB) con bare € (probe raíz)
        pilot_pkts = _load_pilot_packets()
        q_pilot_bare_euro = "Según el cuadro 9, ¿qué € figura para Pensiones en 2023?"
        req_pilot = parse_table_request(q_pilot_bare_euro, pilot_pkts)
        self.assertEqual(req_pilot["status"], "unsupported", "Fallo al rechazar bare € en Cuadro 9 del piloto")
        self.assertIn("importe monetario", req_pilot["reason"].lower())
        self.assertIn("% pib", req_pilot["reason"].lower())
        pkt_pensiones = next(p for p in pilot_pkts if p.get("table_id") == "table_37a2d0c0cb8bdb90_p95_cuadro9" and p.get("row_label") == "Pensiones")
        cell_2023 = next(c for c in pkt_pensiones["cells"] if c.get("year") == 2023)
        tc_pilot = _build_target_cell(pkt_pensiones, cell_2023)
        fact_pilot = _build_valid_fact(tc_pilot, cell_2023["evidence_ids"])
        res_pilot = guard_table_facts([fact_pilot], req_pilot, pilot_pkts)
        self.assertEqual(res_pilot["decision"], "rejected")

        # 3. Controles positivos en tabla monetaria (millones de euros):
        bundle_m, pages_m = _make_synthetic_bundle_and_pages()
        t_m = bundle_m["tables"][0]
        t_m["caption_text"] = "CUADRO 1. PIB (millones de euros)"
        ft_m = _format_table_as_text(t_m)
        p_text_m = f"Preámbulo.\n\n{ft_m}\n\nNotas."
        bundle_m["pages"][0]["layout_text"] = p_text_m
        bundle_m["pages"][0]["blocks"][1]["text"] = ft_m
        pages_m[0]["text"] = p_text_m
        pages_m[0]["end_char"] = len(p_text_m)
        pkts_m = build_table_packets(bundle_m, pages_m)
        self.assertEqual(pkts_m[0]["cells"][0]["unit"], "millones de euros")

        currency_positive_queries = [
            ("Según el cuadro 1, ¿qué € figura para Empleo en 2025?", "bare €"),
            ("Según el cuadro 1, ¿cuál es el importe (€) para Empleo en 2025?", "parenthesized (€)"),
            ("Según el cuadro 1, ¿cuál es el importe en euros para Empleo en 2025?", "word euros"),
            ("Según el cuadro 1, ¿cuál es el importe en millones de € para Empleo en 2025?", "compound millones de €"),
            ("Según el cuadro 1, ¿cuál es el dato para Empleo en 2025?", "generic monetary value"),
        ]
        for q_pos, desc in currency_positive_queries:
            req_m = parse_table_request(q_pos, pkts_m)
            self.assertEqual(req_m["status"], "supported", f"Fallo en control positivo monetario ({desc}): {q_pos}")
            tc_m = req_m["target_cells"][0]
            self.assertEqual(tc_m["unit"], "millones de euros")
            fact_m = _build_valid_fact(tc_m, pkts_m[0]["cells"][0]["evidence_ids"])
            res_m = guard_table_facts([fact_m], req_m, pkts_m)
            self.assertEqual(res_m["decision"], "accepted", f"Guardia no aceptó hecho monetario válido ({desc}): {q_pos}")

        # 4. Controles positivos en tabla porcentual (% PIB):
        percent_positive_queries = [
            ("Según el cuadro 1, ¿cuál es el porcentaje del PIB para Empleo en 2025?", "% del PIB"),
            ("Según el cuadro 1, ¿qué % figura para Empleo en 2025?", "bare %"),
            ("Según el cuadro 1, ¿cuál es el porcentaje para Empleo en 2025?", "word porcentaje"),
        ]
        for q_pct_pos, desc in percent_positive_queries:
            req_pct_p = parse_table_request(q_pct_pos, pkts_pct)
            self.assertEqual(req_pct_p["status"], "supported", f"Fallo en control positivo porcentual ({desc}): {q_pct_pos}")
            tc_p = req_pct_p["target_cells"][0]
            self.assertEqual(tc_p["unit"], "% PIB")
            fact_p = _build_valid_fact(tc_p, pkts_pct[0]["cells"][0]["evidence_ids"])
            res_p = guard_table_facts([fact_p], req_pct_p, pkts_pct)
            self.assertEqual(res_p["decision"], "accepted", f"Guardia no aceptó hecho porcentual válido ({desc}): {q_pct_pos}")

        # 5. Control negativo simétrico: consulta porcentual sobre tabla monetaria debe ser unsupported
        req_sym_pct = parse_table_request("Según el cuadro 1, ¿qué % figura para Empleo en 2025?", pkts_m)
        self.assertEqual(req_sym_pct["status"], "unsupported")
        self.assertIn("monetarias", req_sym_pct["reason"].lower())
        res_sym = guard_table_facts([fact_pct_valid], req_sym_pct, pkts_m)
        self.assertEqual(res_sym["decision"], "rejected")

        # 6. Consultas con familias de unidades contradictorias (deben ser unsupported):
        q_conflict_pct_curr = "Según el cuadro 1, ¿qué porcentaje y € figuran para Empleo en 2025?"
        req_conflict1 = parse_table_request(q_conflict_pct_curr, pkts_pct)
        self.assertEqual(req_conflict1["status"], "unsupported")
        self.assertIn("incompatibles", req_conflict1["reason"].lower())

        q_conflict_pts_curr = "En el cuadro 1, ¿cuál es la variación en puntos y € para Empleo entre 2030 y 2025?"
        req_conflict2 = parse_table_request(q_conflict_pts_curr, pkts_pct)
        self.assertEqual(req_conflict2["status"], "unsupported")
        self.assertIn("incompatibles", req_conflict2["reason"].lower())

        # 7. Modificadores / calificadores fuera de la gramática (rechazo estricto por residuales / full scope):
        unknown_qualifier_queries = [
            "Según el cuadro 1, ¿qué $ figura para Empleo en 2025?",
            "Según el cuadro 1, ¿cuál es el importe en dólares para Empleo en 2025?",
            "Según el cuadro 1, ¿cuál es el importe nominal en euros para Empleo en 2025?",
            "Según el cuadro 1, ¿cuál es el importe real para Empleo en 2025?",
        ]
        for q_unk in unknown_qualifier_queries:
            req_unk = parse_table_request(q_unk, pkts_m)
            self.assertEqual(req_unk["status"], "unsupported", f"Fallo al rechazar calificador desconocido: {q_unk}")
            self.assertIn("no soportada", req_unk["reason"].lower())

        # 8. Control positivo de variación en puntos del PIB sobre celda delta
        q_pts_pos = "En el cuadro 1, ¿qué variación publicada para Empleo figura entre 2030 y 2025 en puntos del PIB?"
        req_pts_pos = parse_table_request(q_pts_pos, pkts_pct)
        self.assertEqual(req_pts_pos["status"], "supported")
        tc_d = req_pts_pos["target_cells"][0]
        self.assertEqual(tc_d["unit"], "puntos PIB")
        fact_d = _build_valid_fact(tc_d, pkts_pct[0]["cells"][2]["evidence_ids"])
        res_d = guard_table_facts([fact_d], req_pts_pos, pkts_pct)
        self.assertEqual(res_d["decision"], "accepted")

    def test_p1_regression_pib_mention_population_percent_full_pipeline(self):
        """Regresión completa build_table_packets -> parse -> guard para mención de PIB con % población."""
        bundle, pages = _make_synthetic_bundle_and_pages()
        t = bundle["tables"][0]
        t["caption_text"] = "CUADRO 1. Pensiones y PIB (% población)"
        ft = _format_table_as_text(t)
        p_text = f"Preámbulo.\n\n{ft}\n\nNotas."
        bundle["pages"][0]["layout_text"] = p_text
        bundle["pages"][0]["blocks"][1]["text"] = ft
        pages[0]["text"] = p_text
        pages[0]["end_char"] = len(p_text)

        pkts = build_table_packets(bundle, pages)
        self.assertEqual(pkts[0]["cells"][0]["unit"], "% población")
        self.assertNotEqual(pkts[0]["cells"][0]["unit"], "% PIB")

        # 1. Consulta solicitando PIB sobre tabla de población debe ser unsupported
        req_pib = parse_table_request("Según el cuadro 1, ¿cuál es el porcentaje del PIB para Empleo en 2025?", pkts)
        self.assertEqual(req_pib["status"], "unsupported")
        self.assertIn("% población", req_pib["reason"])

        # 2. Controles positivos: consultas con palabra 'porcentaje', bare '%' y '％' sobre tabla de % población son supported
        for q_pct_pop in [
            "Según el cuadro 1, ¿cuál es el porcentaje para Empleo en 2025?",
            "Según el cuadro 1, ¿qué % figura para Empleo en 2025?",
            "Según el cuadro 1, ¿qué ％ figura para Empleo en 2025?",
        ]:
            req_pop = parse_table_request(q_pct_pop, pkts)
            self.assertEqual(req_pop["status"], "supported", f"Fallo al soportar consulta porcentual legítima: {q_pct_pop}")
            tc_pop = req_pop["target_cells"][0]
            self.assertEqual(tc_pop["unit"], "% población")
            fact_valid_pop = _build_valid_fact(tc_pop, pkts[0]["cells"][0]["evidence_ids"])
            res_pop = guard_table_facts([fact_valid_pop], req_pop, pkts)
            self.assertEqual(res_pop["decision"], "accepted", f"Guardia no aceptó hecho legítimo para: {q_pct_pop}")

        # 3. Hecho reclamando falsamente '% PIB' es rechazado por la guardia
        req_val = parse_table_request("Según el cuadro 1, ¿cuál es el porcentaje para Empleo en 2025?", pkts)
        tc = req_val["target_cells"][0]
        fact_wrong_unit = _build_valid_fact(tc, pkts[0]["cells"][0]["evidence_ids"])
        fact_wrong_unit["unit"] = "% PIB"
        res_guard = guard_table_facts([fact_wrong_unit], req_val, pkts)
        self.assertEqual(res_guard["decision"], "rejected")
        self.assertTrue(any("unidad" in r.lower() for r in res_guard["reasons"]))

    def test_p1_regression_samecaption_conflicting_denominators_fail_closed(self):
        """Regresión completa para títulos con cláusulas de unidad conflictivas (debe fallar cerrado)."""
        # Conflicto entre denominadores en el mismo título (% PIB vs % población)
        bundle, pages = _make_synthetic_bundle_and_pages()
        t = bundle["tables"][0]
        t["caption_text"] = "CUADRO 1. Pensiones (% PIB) y Salarios (% población)"
        ft = _format_table_as_text(t)
        p_text = f"Preámbulo.\n\n{ft}\n\nNotas."
        bundle["pages"][0]["layout_text"] = p_text
        bundle["pages"][0]["blocks"][1]["text"] = ft
        pages[0]["text"] = p_text
        pages[0]["end_char"] = len(p_text)

        with self.assertRaises(ValueError) as ctx:
            build_table_packets(bundle, pages)
        self.assertIn("múltiples denominadores incompatibles", str(ctx.exception))

        # Conflicto moneda vs porcentaje en el mismo título
        bundle2, pages2 = _make_synthetic_bundle_and_pages()
        t2 = bundle2["tables"][0]
        t2["caption_text"] = "CUADRO 1. Pensiones (% PIB) y PIB (millones de euros)"
        ft2 = _format_table_as_text(t2)
        p_text2 = f"Preámbulo.\n\n{ft2}\n\nNotas."
        bundle2["pages"][0]["layout_text"] = p_text2
        bundle2["pages"][0]["blocks"][1]["text"] = ft2
        pages2[0]["text"] = p_text2
        pages2[0]["end_char"] = len(p_text2)

        with self.assertRaises(ValueError) as ctx2:
            build_table_packets(bundle2, pages2)
        self.assertIn("conflictivas", str(ctx2.exception))

    def test_p1_regression_positive_explicit_gdp_and_del_pib_full_pipeline(self):
        """Regresión completa para control positivo '% del PIB', '% PIB', 'en % del PIB' y parentizados."""
        test_captions = [
            "CUADRO 1. GASTO EN PENSIONES (% del PIB)",
            "CUADRO 1. GASTO EN PENSIONES (% PIB)",
            "CUADRO 1. GASTO EN PENSIONES % del PIB",
            "CUADRO 1. GASTO EN PENSIONES % PIB",
            "CUADRO 1. GASTO EN PENSIONES en % del PIB",
        ]
        for cap in test_captions:
            bundle, pages = _make_synthetic_bundle_and_pages()
            t = bundle["tables"][0]
            t["caption_text"] = cap
            ft = _format_table_as_text(t)
            p_text = f"Preámbulo.\n\n{ft}\n\nNotas."
            bundle["pages"][0]["layout_text"] = p_text
            bundle["pages"][0]["blocks"][1]["text"] = ft
            pages[0]["text"] = p_text
            pages[0]["end_char"] = len(p_text)

            pkts = build_table_packets(bundle, pages)
            self.assertEqual(pkts[0]["cells"][0]["unit"], "% PIB", f"Fallo de unidad nivel en {cap}")
            self.assertEqual(pkts[0]["cells"][2]["unit"], "puntos PIB", f"Fallo de unidad delta en {cap}")

            # 1. Consulta de nivel soportada y aceptada por guard
            req = parse_table_request("Según el cuadro 1, ¿qué porcentaje del PIB figura para Empleo en 2025?", pkts)
            self.assertEqual(req["status"], "supported", f"Fallo al soportar consulta nivel en {cap}")
            tc = req["target_cells"][0]
            self.assertEqual(tc["unit"], "% PIB")
            fact = _build_valid_fact(tc, pkts[0]["cells"][0]["evidence_ids"])
            res_guard = guard_table_facts([fact], req, pkts)
            self.assertEqual(res_guard["decision"], "accepted", f"Fallo al aceptar hecho nivel en {cap}")

            # 2. Consulta de delta soportada y aceptada por guard (herencia de unidad de variación)
            req_d = parse_table_request("En el cuadro 1, ¿qué variación publicada para Empleo figura entre 2030 y 2025 en puntos del PIB?", pkts)
            self.assertEqual(req_d["status"], "supported", f"Fallo al soportar consulta delta en {cap}")
            tc_d = req_d["target_cells"][0]
            self.assertEqual(tc_d["unit"], "puntos PIB")
            fact_d = _build_valid_fact(tc_d, pkts[0]["cells"][2]["evidence_ids"])
            res_guard_d = guard_table_facts([fact_d], req_d, pkts)
            self.assertEqual(res_guard_d["decision"], "accepted", f"Fallo al aceptar hecho delta en {cap}")

    def test_p1_regression_unknown_units_and_negative_controls_full_pipeline(self):
        """Regresión completa para unidades desconocidas y controles negativos (no inventar GDP)."""
        # 1. Unidad desconocida explícita (toneladas)
        bundle, pages = _make_synthetic_bundle_and_pages()
        t = bundle["tables"][0]
        t["caption_text"] = "CUADRO 1. PRODUCCIÓN (toneladas)"
        ft = _format_table_as_text(t)
        p_text = f"Preámbulo.\n\n{ft}\n\nNotas."
        bundle["pages"][0]["layout_text"] = p_text
        bundle["pages"][0]["blocks"][1]["text"] = ft
        pages[0]["text"] = p_text
        pages[0]["end_char"] = len(p_text)

        pkts = build_table_packets(bundle, pages)
        self.assertNotEqual(pkts[0]["cells"][0]["unit"], "% PIB")
        self.assertNotEqual(pkts[0]["cells"][0]["unit"], "%")

        # Consulta pidiendo PIB sobre tabla de toneladas debe ser rechazada
        req_pib = parse_table_request("Según el cuadro 1, ¿cuál es el porcentaje del PIB para Empleo en 2025?", pkts)
        self.assertEqual(req_pib["status"], "unsupported")

        # 2. Mención de PIB con porcentaje de población no parentizado
        bundle2, pages2 = _make_synthetic_bundle_and_pages()
        t2 = bundle2["tables"][0]
        t2["caption_text"] = "CUADRO 1. Pensiones y PIB % población"
        ft2 = _format_table_as_text(t2)
        p_text2 = f"Preámbulo.\n\n{ft2}\n\nNotas."
        bundle2["pages"][0]["layout_text"] = p_text2
        bundle2["pages"][0]["blocks"][1]["text"] = ft2
        pages2[0]["text"] = p_text2
        pages2[0]["end_char"] = len(p_text2)

        pkts2 = build_table_packets(bundle2, pages2)
        self.assertEqual(pkts2[0]["cells"][0]["unit"], "% población")
        self.assertNotEqual(pkts2[0]["cells"][0]["unit"], "% PIB")

        req_pib2 = parse_table_request("Según el cuadro 1, ¿cuál es el porcentaje del PIB para Empleo en 2025?", pkts2)
        self.assertEqual(req_pib2["status"], "unsupported")

    def test_p1_fragments_long_caption_and_header_fail_closed(self):
        bundle, pages = _make_synthetic_bundle_and_pages()
        t = bundle["tables"][0]

        # 1. Caption largo (> 700 caracteres) debe fallar cerrado (ValueError), no omitirse silenciosamente
        long_cap = "CUADRO 1. " + ("DESCRIPCION_MUY_LARGA_DE_GASTO_EN_PENSIONES_" * 25) + " (% PIB)"
        self.assertGreater(len(long_cap), 700)
        t["caption_text"] = long_cap
        ft = _format_table_as_text(t)
        p_text = f"Preámbulo.\n\n{ft}\n\nNotas."
        bundle["pages"][0]["layout_text"] = p_text
        bundle["pages"][0]["blocks"][1]["text"] = ft
        pages[0]["text"] = p_text
        pages[0]["end_char"] = len(p_text)

        with self.assertRaises(ValueError) as ctx_cap:
            build_table_packets(bundle, pages)
        self.assertIn("caption supera el límite de 700 caracteres", str(ctx_cap.exception))

        # Control positivo correspondiente con header corto y coherente
        bundle_ctrl, pages_ctrl = _make_synthetic_bundle_and_pages()
        t_ctrl = bundle_ctrl["tables"][0]
        short_hdr = "2025_CORTO"
        for c in t_ctrl["cells"]:
            if c.get("col_start") == 1 and c.get("col_end") == 2:
                if c.get("column_header"):
                    c["text"] = short_hdr
                    c["raw_text"] = short_hdr
                else:
                    c["header_path"] = [short_hdr]
        ft_ctrl = _format_table_as_text(t_ctrl)
        p_text_ctrl = f"Preámbulo.\n\n{ft_ctrl}\n\nNotas."
        bundle_ctrl["pages"][0]["layout_text"] = p_text_ctrl
        bundle_ctrl["pages"][0]["blocks"][1]["text"] = ft_ctrl
        pages_ctrl[0]["text"] = p_text_ctrl
        pages_ctrl[0]["end_char"] = len(p_text_ctrl)
        pkts_ctrl = build_table_packets(bundle_ctrl, pages_ctrl)
        self.assertGreater(len(pkts_ctrl), 0)

        # 2. Header largo (> 700 caracteres) debe fallar cerrado (ValueError), no omitirse silenciosamente
        bundle2, pages2 = _make_synthetic_bundle_and_pages()
        t2 = bundle2["tables"][0]
        long_hdr = "ENCABEZADO_DE_COLUMNA_EXTREMADAMENTE_LARGO_" * 20
        self.assertGreater(len(long_hdr), 700)
        for c in t2["cells"]:
            if c.get("col_start") == 1 and c.get("col_end") == 2:
                if c.get("column_header"):
                    c["text"] = long_hdr
                    c["raw_text"] = long_hdr
                else:
                    c["header_path"] = [long_hdr]
        ft2 = _format_table_as_text(t2)
        p_text2 = f"Preámbulo.\n\n{ft2}\n\nNotas."
        bundle2["pages"][0]["layout_text"] = p_text2
        bundle2["pages"][0]["blocks"][1]["text"] = ft2
        pages2[0]["text"] = p_text2
        pages2[0]["end_char"] = len(p_text2)

        with self.assertRaises(ValueError) as ctx_hdr:
            build_table_packets(bundle2, pages2)
        self.assertIn("header supera el límite de 700 caracteres", str(ctx_hdr.exception))

        # 3. Guard rechaza hechos si falta el fragmento de cabecera o de título obligatorio
        bundle3, pages3 = _make_synthetic_bundle_and_pages()
        pkts3 = build_table_packets(bundle3, pages3)
        req3 = parse_table_request("Según el cuadro 1, ¿qué porcentaje del PIB figura para Empleo en 2025?", pkts3)
        self.assertEqual(req3["status"], "supported")
        tc3 = req3["target_cells"][0]
        cell_ev_ids = pkts3[0]["cells"][0]["evidence_ids"]

        # Fact sin fragmento de cabecera
        no_hdr_ids = [eid for eid in cell_ev_ids if not eid.endswith("_header")]
        fact_no_hdr = _build_valid_fact(tc3, no_hdr_ids)
        res_no_hdr = guard_table_facts([fact_no_hdr], req3, pkts3)
        self.assertEqual(res_no_hdr["decision"], "rejected")

        # Fact sin fragmento de título
        no_cap_ids = [eid for eid in cell_ev_ids if not eid.endswith("_caption")]
        fact_no_cap = _build_valid_fact(tc3, no_cap_ids)
        res_no_cap = guard_table_facts([fact_no_cap], req3, pkts3)
        self.assertEqual(res_no_cap["decision"], "rejected")

    def test_p1_numeric_guard_duplicate_normalized_row_labels_unsupported(self):
        bundle, pages = _make_synthetic_bundle_and_pages()
        pkts = build_table_packets(bundle, pages)

        # 1. Dos filas 'Empleo' con cifras diferentes (10.1 vs 15.3) no deben resolverse arbitrariamente a la última
        pkt_emp1 = copy.deepcopy(pkts[0])
        pkt_emp2 = copy.deepcopy(pkts[0])
        pkt_emp2["cells"][0]["cell_id"] = "c_r1c1_dup"
        pkt_emp2["cells"][0]["value_decimal"] = "15.3"
        pkt_sal = copy.deepcopy(pkts[1])  # 'Salarios' es único

        ambiguous_packets = [pkt_emp1, pkt_emp2, pkt_sal]

        q_emp = "Según el cuadro 1, ¿cuál es el porcentaje del PIB para Empleo en 2025?"
        req_emp = parse_table_request(q_emp, ambiguous_packets)
        self.assertEqual(req_emp["status"], "unsupported")
        self.assertIn("Ambigüedad en etiqueta de fila", req_emp["reason"])

        # 2. Dos filas 'Empleo' con la misma cifra (10.1 y 10.1) tampoco pueden resolver identidad unívoca
        pkt_emp2_same = copy.deepcopy(pkts[0])
        ambiguous_same = [pkt_emp1, pkt_emp2_same, pkt_sal]
        req_same = parse_table_request(q_emp, ambiguous_same)
        self.assertEqual(req_same["status"], "unsupported")

        # 3. Duplicados normalizados por mayúsculas/minúsculas o acentos NFD
        pkt_emp_lower = copy.deepcopy(pkts[0])
        pkt_emp_lower["row_label"] = "empleo"
        pkt_emp_lower["cells"][0]["row_label"] = "empleo"
        ambiguous_case = [pkt_emp1, pkt_emp_lower, pkt_sal]
        req_case = parse_table_request(q_emp, ambiguous_case)
        self.assertEqual(req_case["status"], "unsupported")

        # 4. Control positivo: la otra fila única ('Salarios') en la misma tabla sigue siendo utilizable
        q_sal = "Según el cuadro 1, ¿cuál es el porcentaje del PIB para Salarios en 2025?"
        req_sal = parse_table_request(q_sal, ambiguous_packets)
        self.assertEqual(req_sal["status"], "supported")
        self.assertEqual(len(req_sal["target_cells"]), 1)
        self.assertEqual(req_sal["target_cells"][0]["concept"], "Salarios")
        self.assertEqual(req_sal["target_cells"][0]["expected_value"], "8.2")

        # 5. Control positivo: 'No modelizados' vs 'Modelizados' coincidencia por etiqueta más larga
        b_nested, p_nested = _make_synthetic_bundle_with_nested_labels()
        pkts_nested = build_table_packets(b_nested, p_nested)
        q_no_mod = "Según el cuadro 1, ¿cuál es el porcentaje del PIB para No modelizados en 2025?"
        req_no_mod = parse_table_request(q_no_mod, pkts_nested)
        self.assertEqual(req_no_mod["status"], "supported")
        self.assertEqual(req_no_mod["target_cells"][0]["concept"], "No modelizados")

    def test_p1_numeric_guard_duplicate_year_columns_unsupported_per_row(self):
        """Regresión P1 guard546: dos columnas físicas para el mismo año y concepto deben ser unsupported."""
        bundle, pages = _make_synthetic_bundle_and_pages()
        pkts = build_table_packets(bundle, pages)

        # 1. Fixture coherente: fila Empleo con dos celdas físicas para 2025 (distintas IDs, coordenadas, header_path correcto)
        pkt_emp = copy.deepcopy(pkts[0])
        cell_2025_a = copy.deepcopy(pkt_emp["cells"][0])
        cell_2025_a["cell_id"] = "c_r1c1_a"
        cell_2025_a["bbox"] = {"l": 200.0, "t": 230.0, "r": 250.0, "b": 280.0, "coord_origin": "TOPLEFT"}
        cell_2025_a["header_path"] = ["2025"]
        cell_2025_a["value_decimal"] = "10.5"

        cell_2025_b_diff = copy.deepcopy(pkt_emp["cells"][0])
        cell_2025_b_diff["cell_id"] = "c_r1c1_b"
        cell_2025_b_diff["bbox"] = {"l": 250.0, "t": 230.0, "r": 300.0, "b": 280.0, "coord_origin": "TOPLEFT"}
        cell_2025_b_diff["header_path"] = ["2025"]
        cell_2025_b_diff["value_decimal"] = "15.3"

        # Celda 2030 única en la misma fila Empleo
        cell_2030 = copy.deepcopy(pkt_emp["cells"][1])

        # Test A: valores diferentes ("10.5" vs "15.3") -> whole unsupported, nunca tomar primera/última
        pkt_emp_diff = copy.deepcopy(pkt_emp)
        pkt_emp_diff["cells"] = [cell_2025_a, cell_2025_b_diff, cell_2030]
        test_packets_diff = [pkt_emp_diff, copy.deepcopy(pkts[1])]

        q_2025 = "Según el cuadro 1, ¿qué porcentaje del PIB figura para Empleo en 2025?"
        req_diff = parse_table_request(q_2025, test_packets_diff)
        self.assertEqual(req_diff["status"], "unsupported")
        self.assertIn("Ambigüedad", req_diff["reason"])

        # Intentar verificar hecho sobre una de las celdas ambiguas es rechazado por guard
        tc_diff = _build_target_cell(pkt_emp_diff, cell_2025_a)
        fact_diff = _build_valid_fact(tc_diff, cell_2025_a["evidence_ids"])
        res_guard_diff = guard_table_facts([fact_diff], {"status": "supported", "target_cells": [tc_diff]}, test_packets_diff)
        self.assertEqual(res_guard_diff["decision"], "rejected")
        self.assertTrue(any("ambigua" in r.lower() for r in res_guard_diff["reasons"]))

        # Test B: valores iguales ("10.5" y "10.5") -> igualmente unsupported
        cell_2025_b_same = copy.deepcopy(cell_2025_b_diff)
        cell_2025_b_same["value_decimal"] = "10.5"
        pkt_emp_same = copy.deepcopy(pkt_emp)
        pkt_emp_same["cells"] = [cell_2025_a, cell_2025_b_same, cell_2030]
        test_packets_same = [pkt_emp_same, copy.deepcopy(pkts[1])]

        req_same = parse_table_request(q_2025, test_packets_same)
        self.assertEqual(req_same["status"], "unsupported")
        self.assertIn("Ambigüedad", req_same["reason"])

        # Test C: otro año único (2030) en la misma fila con celda 2025 duplicada sigue siendo válido
        q_2030 = "Según el cuadro 1, ¿qué porcentaje del PIB figura para Empleo en 2030?"
        req_2030 = parse_table_request(q_2030, test_packets_diff)
        self.assertEqual(req_2030["status"], "supported")
        self.assertEqual(len(req_2030["target_cells"]), 1)
        self.assertEqual(req_2030["target_cells"][0]["year"], 2030)
        self.assertEqual(req_2030["target_cells"][0]["expected_value"], "10.5")

        # Test D: dos filas DIFERENTES que comparten el año 2025 (cada una única en su fila) es positivo
        # Salarios tiene su propio 2025 único (8.2)
        q_sal_2025 = "Según el cuadro 1, ¿qué porcentaje del PIB figura para Salarios en 2025?"
        req_sal = parse_table_request(q_sal_2025, test_packets_diff)
        self.assertEqual(req_sal["status"], "supported")
        self.assertEqual(req_sal["target_cells"][0]["expected_value"], "8.2")

        # Consulta multiconcepto unívoca: Empleo en 2030 (único) y Salarios en 2025 (único)
        q_multi = "Según el cuadro 1, ¿qué porcentajes del PIB figuran para Empleo en 2030 y Salarios en 2025, respectivamente?"
        req_multi = parse_table_request(q_multi, test_packets_diff)
        self.assertEqual(req_multi["status"], "supported")
        self.assertEqual(len(req_multi["target_cells"]), 2)

    def test_p1_numeric_guard_duplicate_published_delta_columns_unsupported(self):
        """Regresión P1 guard546: dos columnas físicas de variación publicada para el mismo concepto deben ser unsupported."""
        bundle, pages = _make_synthetic_bundle_and_pages()
        pkts = build_table_packets(bundle, pages)

        # Fixture: fila Empleo con dos celdas físicas de variación publicada para [2030, 2025]
        pkt_emp = copy.deepcopy(pkts[0])
        delta_a = copy.deepcopy(pkt_emp["cells"][2])
        delta_a["cell_id"] = "c_r1c3_a"
        delta_a["bbox"] = {"l": 400.0, "t": 230.0, "r": 450.0, "b": 280.0, "coord_origin": "TOPLEFT"}
        delta_a["header_path"] = ["Δ 2030-2025"]
        delta_a["value_decimal"] = "0.3"

        delta_b_diff = copy.deepcopy(pkt_emp["cells"][2])
        delta_b_diff["cell_id"] = "c_r1c3_b"
        delta_b_diff["bbox"] = {"l": 450.0, "t": 230.0, "r": 500.0, "b": 280.0, "coord_origin": "TOPLEFT"}
        delta_b_diff["header_path"] = ["Δ 2030-2025"]
        delta_b_diff["value_decimal"] = "0.9"

        # Test A: diferentes valores ("0.3" vs "0.9") -> unsupported
        pkt_diff = copy.deepcopy(pkt_emp)
        pkt_diff["cells"] = [pkt_emp["cells"][0], pkt_emp["cells"][1], delta_a, delta_b_diff]
        test_pkts_diff = [pkt_diff, copy.deepcopy(pkts[1])]

        q_delta = "En el cuadro 1, ¿qué variación publicada para Empleo corresponde a 2030 menos 2025, en puntos del PIB?"
        req_diff = parse_table_request(q_delta, test_pkts_diff)
        self.assertEqual(req_diff["status"], "unsupported")
        self.assertIn("Ambigüedad", req_diff["reason"])

        # Guard rechaza hecho sobre celda de variación ambigua
        tc_delta = _build_target_cell(pkt_diff, delta_a)
        fact_diff = _build_valid_fact(tc_delta, delta_a["evidence_ids"])
        res_guard_diff = guard_table_facts([fact_diff], {"status": "supported", "target_cells": [tc_delta]}, test_pkts_diff)
        self.assertEqual(res_guard_diff["decision"], "rejected")
        self.assertTrue(any("ambigua" in r.lower() for r in res_guard_diff["reasons"]))

        # Test B: mismos valores ("0.3" y "0.3") -> igualmente unsupported
        delta_b_same = copy.deepcopy(delta_b_diff)
        delta_b_same["value_decimal"] = "0.3"
        pkt_same = copy.deepcopy(pkt_emp)
        pkt_same["cells"] = [pkt_emp["cells"][0], pkt_emp["cells"][1], delta_a, delta_b_same]
        test_pkts_same = [pkt_same, copy.deepcopy(pkts[1])]

        req_same = parse_table_request(q_delta, test_pkts_same)
        self.assertEqual(req_same["status"], "unsupported")
        self.assertIn("Ambigüedad", req_same["reason"])

        # Test C: nivel 2025 único sigue siendo soportado
        q_level = "Según el cuadro 1, ¿qué porcentaje del PIB figura para Empleo en 2025?"
        req_lvl = parse_table_request(q_level, test_pkts_diff)
        self.assertEqual(req_lvl["status"], "supported")
        self.assertEqual(req_lvl["target_cells"][0]["expected_value"], "10.1")

    def test_p2_cell_bbox_containment_within_table(self):
        # 1. Regresión fixture: celda c_r1c1 {l:0, t:0, r:10, b:10} vs tabla {100, 200, 500, 350} debe rechazar
        bundle, pages = _make_synthetic_bundle_and_pages()
        t = bundle["tables"][0]
        t["cells"][5]["bbox"] = {"l": 0.0, "t": 0.0, "r": 10.0, "b": 10.0, "coord_origin": "TOPLEFT"}

        with self.assertRaises(ValueError) as ctx1:
            build_table_packets(bundle, pages)
        self.assertIn("no está estrictamente contenido dentro de su tabla", str(ctx1.exception))

        # 2. Celda que cruza parcialmente el límite izquierdo (l: 95.0 < 100.0) debe rechazar
        bundle2, pages2 = _make_synthetic_bundle_and_pages()
        t2 = bundle2["tables"][0]
        t2["cells"][5]["bbox"] = {"l": 95.0, "t": 230.0, "r": 200.0, "b": 280.0, "coord_origin": "TOPLEFT"}

        with self.assertRaises(ValueError) as ctx2:
            build_table_packets(bundle2, pages2)
        self.assertIn("no está estrictamente contenido dentro de su tabla", str(ctx2.exception))

        # 3. Celda que cruza parcialmente el límite derecho (r: 505.0 > 500.0) debe rechazar
        bundle3, pages3 = _make_synthetic_bundle_and_pages()
        t3 = bundle3["tables"][0]
        t3["cells"][7]["bbox"] = {"l": 400.0, "t": 230.0, "r": 505.0, "b": 280.0, "coord_origin": "TOPLEFT"}

        with self.assertRaises(ValueError) as ctx3:
            build_table_packets(bundle3, pages3)
        self.assertIn("no está estrictamente contenido dentro de su tabla", str(ctx3.exception))

        # 4. Origen de coordenadas incompatible (BOTTOMLEFT vs TOPLEFT) debe rechazar
        bundle4, pages4 = _make_synthetic_bundle_and_pages()
        t4 = bundle4["tables"][0]
        t4["cells"][5]["bbox"] = {"l": 200.0, "t": 230.0, "r": 300.0, "b": 280.0, "coord_origin": "BOTTOMLEFT"}

        with self.assertRaises(ValueError) as ctx4:
            build_table_packets(bundle4, pages4)
        self.assertIn("origen de coordenadas no soportado o incompatible", str(ctx4.exception))

        # 5. Control positivo: celda exactamente en los límites de la tabla (100, 200, 500, 350) soportada
        bundle5, pages5 = _make_synthetic_bundle_and_pages()
        t5 = bundle5["tables"][0]
        t5["cells"][5]["bbox"] = {"l": 100.0, "t": 200.0, "r": 500.0, "b": 350.0, "coord_origin": "TOPLEFT"}
        pkts5 = build_table_packets(bundle5, pages5)
        self.assertEqual(len(pkts5), 2)

        # 6. Control positivo: precheck piloto real (155 celdas sin violaciones de contención)
        precheck_file = REPO_ROOT / "evaluation" / "table_grounding_2026-10-27" / "bbox_pilot_precheck.json"
        if precheck_file.exists():
            data = json.loads(precheck_file.read_text(encoding="utf-8"))
            self.assertEqual(data.get("cells_checked"), 155)
            self.assertEqual(data.get("violations"), [])
        pilot_pkts = _load_pilot_packets()
        self.assertGreater(len(pilot_pkts), 0)

    def test_numeric_guard_header_path_newline_normalization_and_strict_invariants(self):
        """Header path newline encodings (CRLF, LF, bare CR) are accepted as equivalent.

        Verifies:
        - CRLF (\r\n), LF (\n), and bare CR (\r) are accepted on synthetic delta years.
        - Symmetric equivalence: CRLF source matches LF and bare CR facts.
        - Multi-segment headers with mixed newline encodings are accepted.
        - Wrong years, reversed delta years, altered signs (+, −), segment count mismatch,
          segment order alteration, and non-newline whitespace (extra spaces, leading/trailing,
          tabs) are strictly rejected.
        - Original input facts and source literal strings remain completely unchanged after validation.
        - Uses synthetic fixtures completely unrelated to the benchmark.
        """
        bundle, pages = _make_synthetic_bundle_and_pages()
        pkts = build_table_packets(bundle, pages)

        # Build synthetic delta cell with multi-line header_path
        pkt_emp = copy.deepcopy(pkts[0])
        cdelta = copy.deepcopy(pkt_emp["cells"][2])
        source_header_lf = "Δ 2030-2025\n(puntos PIB)"
        source_header_crlf = "Δ 2030-2025\r\n(puntos PIB)"
        source_header_cr = "Δ 2030-2025\r(puntos PIB)"

        cdelta["header_path"] = [source_header_lf]
        cdelta["ordered_delta_years"] = [2030, 2025]
        cdelta["measure"] = "delta"
        cdelta["unit"] = "puntos PIB"
        cdelta["value_decimal"] = "0.3"

        pkt_emp["cells"][2] = cdelta
        tc = _build_target_cell(pkt_emp, cdelta)
        synthetic_packets = [pkt_emp]
        synthetic_request = {
            "status": "supported",
            "table_id": pkt_emp["table_id"],
            "table_caption": pkt_emp["table_caption"],
            "target_cells": [tc],
        }

        # 1. Acceptance of CRLF, LF, and bare CR equivalents
        # Case A: Fact with CRLF against LF source
        fact_crlf = _build_valid_fact(tc, cdelta["evidence_ids"])
        fact_crlf["header_path"] = [source_header_crlf]
        res_crlf = guard_table_facts([fact_crlf], synthetic_request, synthetic_packets)
        self.assertEqual(res_crlf["decision"], "accepted")
        self.assertEqual(len(res_crlf["accepted_facts"]), 1)
        # Verify literal immutability
        self.assertEqual(fact_crlf["header_path"], [source_header_crlf])
        self.assertEqual(cdelta["header_path"], [source_header_lf])

        # Case B: Fact with LF against LF source
        fact_lf = _build_valid_fact(tc, cdelta["evidence_ids"])
        fact_lf["header_path"] = [source_header_lf]
        res_lf = guard_table_facts([fact_lf], synthetic_request, synthetic_packets)
        self.assertEqual(res_lf["decision"], "accepted")
        self.assertEqual(fact_lf["header_path"], [source_header_lf])

        # Case C: Fact with bare CR against LF source
        fact_cr = _build_valid_fact(tc, cdelta["evidence_ids"])
        fact_cr["header_path"] = [source_header_cr]
        res_cr = guard_table_facts([fact_cr], synthetic_request, synthetic_packets)
        self.assertEqual(res_cr["decision"], "accepted")
        self.assertEqual(fact_cr["header_path"], [source_header_cr])

        # Case D: Symmetric check - source has CRLF, facts have LF and bare CR
        cdelta_crlf = copy.deepcopy(cdelta)
        cdelta_crlf["header_path"] = [source_header_crlf]
        pkt_crlf = copy.deepcopy(pkt_emp)
        pkt_crlf["cells"][2] = cdelta_crlf
        tc_crlf = _build_target_cell(pkt_crlf, cdelta_crlf)
        req_crlf = {"status": "supported", "table_id": pkt_crlf["table_id"], "table_caption": pkt_crlf["table_caption"], "target_cells": [tc_crlf]}
        pkts_crlf = [pkt_crlf]

        res_sym_lf = guard_table_facts([fact_lf], req_crlf, pkts_crlf)
        self.assertEqual(res_sym_lf["decision"], "accepted")
        res_sym_cr = guard_table_facts([fact_cr], req_crlf, pkts_crlf)
        self.assertEqual(res_sym_cr["decision"], "accepted")

        # Case E: Multi-segment header with mixed newlines
        cdelta_multi = copy.deepcopy(cdelta)
        cdelta_multi["header_path"] = ["Columna Superior\nNivel 1", "Δ 2030-2025\n(puntos PIB)"]
        pkt_multi = copy.deepcopy(pkt_emp)
        pkt_multi["cells"][2] = cdelta_multi
        tc_multi = _build_target_cell(pkt_multi, cdelta_multi)
        req_multi = {"status": "supported", "table_id": pkt_multi["table_id"], "table_caption": pkt_multi["table_caption"], "target_cells": [tc_multi]}
        pkts_multi = [pkt_multi]

        fact_multi = _build_valid_fact(tc_multi, cdelta_multi["evidence_ids"])
        fact_multi["header_path"] = ["Columna Superior\r\nNivel 1", "Δ 2030-2025\r(puntos PIB)"]
        res_multi = guard_table_facts([fact_multi], req_multi, pkts_multi)
        self.assertEqual(res_multi["decision"], "accepted")

        # 2. Strict rejection of semantically or structurally altered headers
        # Wrong years (2040 instead of 2030)
        fact_wrong_years = _build_valid_fact(tc, cdelta["evidence_ids"])
        fact_wrong_years["header_path"] = ["Δ 2040-2025\n(puntos PIB)"]
        res = guard_table_facts([fact_wrong_years], synthetic_request, synthetic_packets)
        self.assertEqual(res["decision"], "rejected")
        self.assertTrue(any("header_path" in r for r in res["reasons"]))

        # Reversed years (2025-2030 instead of 2030-2025)
        fact_rev_years = _build_valid_fact(tc, cdelta["evidence_ids"])
        fact_rev_years["header_path"] = ["Δ 2025-2030\n(puntos PIB)"]
        res = guard_table_facts([fact_rev_years], synthetic_request, synthetic_packets)
        self.assertEqual(res["decision"], "rejected")
        self.assertTrue(any("header_path" in r for r in res["reasons"]))

        # Altered sign (+ instead of Δ)
        fact_sign_plus = _build_valid_fact(tc, cdelta["evidence_ids"])
        fact_sign_plus["header_path"] = ["+ 2030-2025\n(puntos PIB)"]
        res = guard_table_facts([fact_sign_plus], synthetic_request, synthetic_packets)
        self.assertEqual(res["decision"], "rejected")
        self.assertTrue(any("header_path" in r for r in res["reasons"]))

        # Altered sign (− instead of Δ)
        fact_sign_minus = _build_valid_fact(tc, cdelta["evidence_ids"])
        fact_sign_minus["header_path"] = ["− 2030-2025\n(puntos PIB)"]
        res = guard_table_facts([fact_sign_minus], synthetic_request, synthetic_packets)
        self.assertEqual(res["decision"], "rejected")
        self.assertTrue(any("header_path" in r for r in res["reasons"]))

        # Segment count mismatch (split into two segments instead of one)
        fact_seg_count = _build_valid_fact(tc, cdelta["evidence_ids"])
        fact_seg_count["header_path"] = ["Δ 2030-2025", "(puntos PIB)"]
        res = guard_table_facts([fact_seg_count], synthetic_request, synthetic_packets)
        self.assertEqual(res["decision"], "rejected")
        self.assertTrue(any("header_path" in r for r in res["reasons"]))

        # Segment order mismatch
        fact_rev_order = _build_valid_fact(tc_multi, cdelta_multi["evidence_ids"])
        fact_rev_order["header_path"] = ["Δ 2030-2025\n(puntos PIB)", "Columna Superior\nNivel 1"]
        res = guard_table_facts([fact_rev_order], req_multi, pkts_multi)
        self.assertEqual(res["decision"], "rejected")
        self.assertTrue(any("header_path" in r for r in res["reasons"]))

        # Whitespace other than newline: internal double space
        fact_ws_double = _build_valid_fact(tc, cdelta["evidence_ids"])
        fact_ws_double["header_path"] = ["Δ  2030-2025\n(puntos PIB)"]
        res = guard_table_facts([fact_ws_double], synthetic_request, synthetic_packets)
        self.assertEqual(res["decision"], "rejected")
        self.assertTrue(any("header_path" in r for r in res["reasons"]))

        # Whitespace other than newline: leading space
        fact_ws_lead = _build_valid_fact(tc, cdelta["evidence_ids"])
        fact_ws_lead["header_path"] = [" Δ 2030-2025\n(puntos PIB)"]
        res = guard_table_facts([fact_ws_lead], synthetic_request, synthetic_packets)
        self.assertEqual(res["decision"], "rejected")
        self.assertTrue(any("header_path" in r for r in res["reasons"]))

        # Whitespace other than newline: trailing space
        fact_ws_trail = _build_valid_fact(tc, cdelta["evidence_ids"])
        fact_ws_trail["header_path"] = ["Δ 2030-2025\n(puntos PIB) "]
        res = guard_table_facts([fact_ws_trail], synthetic_request, synthetic_packets)
        self.assertEqual(res["decision"], "rejected")
        self.assertTrue(any("header_path" in r for r in res["reasons"]))

        # Whitespace other than newline: trailing space before newline
        fact_ws_pre_nl = _build_valid_fact(tc, cdelta["evidence_ids"])
        fact_ws_pre_nl["header_path"] = ["Δ 2030-2025 \n(puntos PIB)"]
        res = guard_table_facts([fact_ws_pre_nl], synthetic_request, synthetic_packets)
        self.assertEqual(res["decision"], "rejected")
        self.assertTrue(any("header_path" in r for r in res["reasons"]))

        # Whitespace other than newline: tab
        fact_ws_tab = _build_valid_fact(tc, cdelta["evidence_ids"])
        fact_ws_tab["header_path"] = ["Δ\t2030-2025\n(puntos PIB)"]
        res = guard_table_facts([fact_ws_tab], synthetic_request, synthetic_packets)
        self.assertEqual(res["decision"], "rejected")
        self.assertTrue(any("header_path" in r for r in res["reasons"]))

        # Malformed header_path: not a list
        fact_not_list = _build_valid_fact(tc, cdelta["evidence_ids"])
        fact_not_list["header_path"] = "Δ 2030-2025\n(puntos PIB)"
        res = guard_table_facts([fact_not_list], synthetic_request, synthetic_packets)
        self.assertEqual(res["decision"], "rejected")
        self.assertTrue(any("header_path" in r for r in res["reasons"]))

        # Malformed header_path: list containing non-string
        fact_non_str = _build_valid_fact(tc, cdelta["evidence_ids"])
        fact_non_str["header_path"] = [12345]
        res = guard_table_facts([fact_non_str], synthetic_request, synthetic_packets)
        self.assertEqual(res["decision"], "rejected")
        self.assertTrue(any("header_path" in r for r in res["reasons"]))

        # Immutability verification of facts and source literals
        self.assertEqual(fact_crlf["header_path"], [source_header_crlf])
        self.assertEqual(fact_cr["header_path"], [source_header_cr])
        self.assertEqual(fact_lf["header_path"], [source_header_lf])
        self.assertEqual(cdelta["header_path"], [source_header_lf])


if __name__ == "__main__":
    unittest.main()
