"""Tests sintéticos para extracción de layout de documentos y geometría de tablas."""

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from rag_pensiones.document_layout import (
    _bbox_inside,
    _format_table_as_text,
    check_bbox_relationship,
    extract_document,
    generate_extraction_id,
    normalize_bbox_to_topleft,
    to_page_records,
    write_bundle,
)
from rag_pensiones.ingest import (
    PageRecord,
    validate_page_range,
    validate_published_at,
)
from rag_pensiones.table_geometry import (
    extract_region_cells,
    extract_reviewed_tables,
    validate_regions,
)


class _FakeInputFormat:
    PDF = "pdf"


class _FakeAcceleratorOptions:
    def __init__(self, num_threads: int = 4):
        self.num_threads = num_threads


class _FakeTableStructureOptions:
    def __init__(self, mode: object = None, do_cell_matching: bool = True):
        self.mode = mode
        self.do_cell_matching = do_cell_matching


class _FakeTableFormerMode:
    ACCURATE = "accurate"


class _FakePdfPipelineOptions:
    def __init__(self):
        self.do_ocr = False
        self.do_code_enrichment = False
        self.do_formula_enrichment = False
        self.do_picture_classification = False
        self.do_picture_description = False
        self.enable_remote_services = False
        self.accelerator_options = None
        self.table_structure_options = None
        self.artifacts_path = None


class _FakePdfFormatOption:
    def __init__(self, pipeline_options: object = None):
        self.pipeline_options = pipeline_options


class _FakeDocumentConverter:
    def __init__(self, format_options: object = None):
        self.format_options = format_options

    def convert(self, pdf_path: object, page_range: object = None):
        raise NotImplementedError


def _get_fake_docling_modules():
    import types
    mod_docling = types.ModuleType("docling")
    mod_conv = types.ModuleType("docling.document_converter")
    mod_conv.DocumentConverter = _FakeDocumentConverter
    mod_conv.PdfFormatOption = _FakePdfFormatOption

    mod_datamodel = types.ModuleType("docling.datamodel")
    mod_base_models = types.ModuleType("docling.datamodel.base_models")
    mod_base_models.InputFormat = _FakeInputFormat

    mod_pipe_opts = types.ModuleType("docling.datamodel.pipeline_options")
    mod_pipe_opts.PdfPipelineOptions = _FakePdfPipelineOptions
    mod_pipe_opts.TableStructureOptions = _FakeTableStructureOptions
    mod_pipe_opts.TableFormerMode = _FakeTableFormerMode
    mod_pipe_opts.AcceleratorOptions = _FakeAcceleratorOptions

    mod_docling.document_converter = mod_conv
    mod_docling.datamodel = mod_datamodel
    mod_datamodel.base_models = mod_base_models
    mod_datamodel.pipeline_options = mod_pipe_opts

    return {
        "docling": mod_docling,
        "docling.document_converter": mod_conv,
        "docling.datamodel": mod_datamodel,
        "docling.datamodel.base_models": mod_base_models,
        "docling.datamodel.pipeline_options": mod_pipe_opts,
    }


class TestDocumentLayoutSynthetic(unittest.TestCase):
    """Suite de pruebas sintéticas sin descargas ni oráculos externos."""

    def test_legacy_page_record_schema_and_physical_range(self) -> None:
        """Verifica compatibilidad estricta con PageRecord y preservación de rango físico."""
        doc_id = "a" * 64
        bundle = {
            "schema_version": "document-layout-v1",
            "metadata": {
                "doc_id": doc_id,
                "source_file": "presupuesto.pdf",
                "published_at": "2025-03-01",
                "page_range": [95, 97],
            },
            "pages": [
                {
                    "page_number": 95,
                    "width": 595.32,
                    "height": 841.92,
                    "legacy_text": "Texto pypdf 95",
                    "layout_text": "Layout estructurado 95",
                    "blocks": [],
                    "warnings": [],
                },
                {
                    "page_number": 96,
                    "width": 595.32,
                    "height": 841.92,
                    "legacy_text": "Texto pypdf 96",
                    "layout_text": "Layout estructurado 96",
                    "blocks": [],
                    "warnings": [],
                },
                {
                    "page_number": 97,
                    "width": 595.32,
                    "height": 841.92,
                    "legacy_text": "Texto pypdf 97",
                    "layout_text": "Layout estructurado 97",
                    "blocks": [],
                    "warnings": [],
                },
            ],
            "tables": [],
            "native_docling": {},
        }
        
        # 1. Vista legacy
        records_legacy = to_page_records(bundle, view="legacy")
        self.assertEqual(len(records_legacy), 3)
        for idx, p in enumerate([95, 96, 97]):
            rec = records_legacy[idx]
            self.assertIsInstance(rec, PageRecord)
            self.assertEqual(rec.doc_id, doc_id)
            self.assertEqual(rec.source_file, "presupuesto.pdf")
            self.assertEqual(rec.published_at, "2025-03-01")
            self.assertEqual(rec.page_number, p)
            self.assertEqual(rec.text, f"Texto pypdf {p}")
            
        # 2. Vista layout
        records_layout = to_page_records(bundle, view="layout")
        self.assertEqual(len(records_layout), 3)
        self.assertEqual(records_layout[0].text, "Layout estructurado 95")
        
        # 3. Vista inválida
        with self.assertRaises(ValueError):
            to_page_records(bundle, view="desconocida")

    def test_geometry_callback_crop_bounds_and_merged_headers_and_scalars(self) -> None:
        """Verifica crop bounds, merge de spans, y preservación de vacíos vs 0, comas y signos."""
        region = {
            "profile_id": "test_tab",
            "page_number": 95,
            "page_size": [595.32, 841.92],
            "row_edges": [100.0, 120.0, 140.0, 160.0],  # 3 filas
            "column_edges": [50.0, 100.0, 150.0, 200.0],  # 3 columnas
            "header_rows": 1,
            "row_header_columns": [0],
            "spans": [
                {"row_start": 0, "row_end": 1, "col_start": 1, "col_end": 3}  # Cabecera col 1..2 unificada
            ],
        }
        
        # Simulación de crop_text determinista
        data_map = {
            (50.0, 100.0, 100.0, 120.0): "Concepto",
            (100.0, 100.0, 200.0, 120.0): "Periodo 2024-2025",
            (50.0, 120.0, 100.0, 140.0): "Gasto Pensiones",
            (100.0, 120.0, 150.0, 140.0): "-12,5%",
            (150.0, 120.0, 200.0, 140.0): "0",
            (50.0, 140.0, 100.0, 160.0): "Variación Neta",
            (100.0, 140.0, 150.0, 160.0): "   ",  # Blanco
            (150.0, 140.0, 200.0, 160.0): "+3,14 M€",
        }
        
        def crop_callback(l: float, t: float, r: float, b: float) -> str:
            return data_map.get((l, t, r, b), "")
            
        cells = extract_region_cells(region, crop_callback)
        # Total celdas: 3x3 = 9 menos 1 celda secundaria en span = 8
        self.assertEqual(len(cells), 8)
        
        # Celda span raíz
        span_cell = [c for c in cells if c["row_start"] == 0 and c["col_start"] == 1][0]
        self.assertEqual(span_cell["row_end"], 1)
        self.assertEqual(span_cell["col_end"], 3)
        self.assertEqual(span_cell["text"], "Periodo 2024-2025")
        self.assertTrue(span_cell["column_header"])
        self.assertEqual(span_cell["bbox"]["coord_origin"], "TOPLEFT")
        
        # Verificación de valor negativo con coma y porcentaje
        neg_cell = [c for c in cells if c["row_start"] == 1 and c["col_start"] == 1][0]
        self.assertEqual(neg_cell["raw_text"], "-12,5%")
        self.assertEqual(neg_cell["text"], "-12,5%")
        self.assertFalse(neg_cell["is_empty"])
        self.assertEqual(neg_cell["header_path"], ["Periodo 2024-2025"])
        self.assertEqual(neg_cell["row_label"], "Gasto Pensiones")
        
        # Verificación de "0" vs celda vacía (0 no es vacío)
        zero_cell = [c for c in cells if c["row_start"] == 1 and c["col_start"] == 2][0]
        self.assertEqual(zero_cell["text"], "0")
        self.assertFalse(zero_cell["is_empty"])
        
        empty_cell = [c for c in cells if c["row_start"] == 2 and c["col_start"] == 1][0]
        self.assertEqual(empty_cell["raw_text"], "   ")
        self.assertEqual(empty_cell["text"], "")
        self.assertTrue(empty_cell["is_empty"])
        
        # Verificación de signo más y unidades
        plus_cell = [c for c in cells if c["row_start"] == 2 and c["col_start"] == 2][0]
        self.assertEqual(plus_cell["text"], "+3,14 M€")

    def test_invalid_profile_rejections(self) -> None:
        """Verifica rechazo estricto de SHA incorrecto, tamaños dispares, bordes no finitos o solapados."""
        valid_doc_id = "b" * 64
        page_sizes = {95: (595.32, 841.92)}
        base_table = {
            "profile_id": "cuadro",
            "page_number": 95,
            "page_size": [595.32, 841.92],
            "row_edges": [10.0, 20.0, 30.0],
            "column_edges": [10.0, 20.0, 30.0],
            "header_rows": 1,
            "spans": [],
        }
        
        # 1. SHA inválido
        with self.assertRaises(ValueError):
            validate_regions({"doc_id": "incorrecto", "tables": [base_table]}, valid_doc_id, page_sizes)
            
        # 2. Tamaño de página no coincide
        with self.assertRaises(ValueError):
            validate_regions(
                {"doc_id": valid_doc_id, "tables": [{**base_table, "page_size": [600.0, 900.0]}]},
                valid_doc_id,
                page_sizes,
            )
            
        # 3. Borde no finito (NaN o Inf)
        with self.assertRaises(ValueError):
            validate_regions(
                {"doc_id": valid_doc_id, "tables": [{**base_table, "row_edges": [10.0, float("nan"), 30.0]}]},
                valid_doc_id,
                page_sizes,
            )
            
        # 4. Bordes no crecientes
        with self.assertRaises(ValueError):
            validate_regions(
                {"doc_id": valid_doc_id, "tables": [{**base_table, "column_edges": [10.0, 30.0, 20.0]}]},
                valid_doc_id,
                page_sizes,
            )
            
        # 5. Spans solapados
        overlapping_spans = [
            {"row_start": 0, "row_end": 2, "col_start": 0, "col_end": 2},
            {"row_start": 1, "row_end": 2, "col_start": 1, "col_end": 2},
        ]
        with self.assertRaises(ValueError):
            validate_regions(
                {"doc_id": valid_doc_id, "tables": [{**base_table, "spans": overlapping_spans}]},
                valid_doc_id,
                page_sizes,
            )
            
        # 6. Spans fuera de límites
        out_of_bounds_spans = [
            {"row_start": 0, "row_end": 10, "col_start": 0, "col_end": 1}
        ]
        with self.assertRaises(ValueError):
            validate_regions(
                {"doc_id": valid_doc_id, "tables": [{**base_table, "spans": out_of_bounds_spans}]},
                valid_doc_id,
                page_sizes,
            )
            
        # 7. Rechazo de booleanos en column_edges y row_edges
        with self.assertRaises(ValueError):
            validate_regions(
                {"doc_id": valid_doc_id, "tables": [{**base_table, "column_edges": [10.0, False, 30.0]}]},
                valid_doc_id,
                page_sizes,
            )
        with self.assertRaises(ValueError):
            validate_regions(
                {"doc_id": valid_doc_id, "tables": [{**base_table, "column_edges": [10.0, True, 30.0]}]},
                valid_doc_id,
                page_sizes,
            )
        with self.assertRaises(ValueError):
            validate_regions(
                {"doc_id": valid_doc_id, "tables": [{**base_table, "row_edges": [10.0, False, 30.0]}]},
                valid_doc_id,
                page_sizes,
            )
        with self.assertRaises(ValueError):
            validate_regions(
                {"doc_id": valid_doc_id, "tables": [{**base_table, "row_edges": [10.0, True, 30.0]}]},
                valid_doc_id,
                page_sizes,
            )
            
        # 8. Rechazo de missing, None, vacíos, no-strings y duplicados en profile_id
        no_id_table = {k: v for k, v in base_table.items() if k != "profile_id"}
        with self.assertRaises(ValueError):
            validate_regions({"doc_id": valid_doc_id, "tables": [no_id_table]}, valid_doc_id, page_sizes)
        with self.assertRaises(ValueError):
            validate_regions({"doc_id": valid_doc_id, "tables": [{**base_table, "profile_id": None}]}, valid_doc_id, page_sizes)
        with self.assertRaises(ValueError):
            validate_regions({"doc_id": valid_doc_id, "tables": [{**base_table, "profile_id": ""}]}, valid_doc_id, page_sizes)
        with self.assertRaises(ValueError):
            validate_regions({"doc_id": valid_doc_id, "tables": [{**base_table, "profile_id": "   "}]}, valid_doc_id, page_sizes)
        with self.assertRaises(ValueError):
            validate_regions({"doc_id": valid_doc_id, "tables": [{**base_table, "profile_id": 123}]}, valid_doc_id, page_sizes)
        with self.assertRaises(ValueError):
            validate_regions(
                {
                    "doc_id": valid_doc_id,
                    "tables": [
                        {**base_table, "profile_id": "cuadro_duplicado"},
                        {**base_table, "profile_id": "cuadro_duplicado"},
                    ],
                },
                valid_doc_id,
                page_sizes,
            )
            
        # 8b. Validación estricta de header_rows y row_header_columns (P2)
        with self.assertRaises(ValueError):
            validate_regions({"doc_id": valid_doc_id, "tables": [{**base_table, "header_rows": True}]}, valid_doc_id, page_sizes)
        with self.assertRaises(ValueError):
            validate_regions({"doc_id": valid_doc_id, "tables": [{**base_table, "header_rows": -1}]}, valid_doc_id, page_sizes)
        with self.assertRaises(ValueError):
            validate_regions({"doc_id": valid_doc_id, "tables": [{**base_table, "header_rows": 3}]}, valid_doc_id, page_sizes)
        with self.assertRaises(ValueError):
            validate_regions({"doc_id": valid_doc_id, "tables": [{**base_table, "row_header_columns": [True]}]}, valid_doc_id, page_sizes)
        with self.assertRaises(ValueError):
            validate_regions({"doc_id": valid_doc_id, "tables": [{**base_table, "row_header_columns": [-1]}]}, valid_doc_id, page_sizes)
        with self.assertRaises(ValueError):
            validate_regions({"doc_id": valid_doc_id, "tables": [{**base_table, "row_header_columns": [2]}]}, valid_doc_id, page_sizes)
        with self.assertRaises(ValueError):
            validate_regions({"doc_id": valid_doc_id, "tables": [{**base_table, "row_header_columns": [0, 0]}]}, valid_doc_id, page_sizes)
        with self.assertRaises(ValueError):
            validate_regions({"doc_id": valid_doc_id, "tables": [{**base_table, "row_header_columns": "0"}]}, valid_doc_id, page_sizes)
            
        # 9. Validación de fecha ISO y calendario
        with self.assertRaises(ValueError):
            validate_published_at("2025/03/01")
        with self.assertRaises(ValueError):
            validate_published_at("2025-02-31")
            
        # 10. Rango de páginas inválido
        with self.assertRaises(ValueError):
            validate_page_range(start_page=10, end_page=5, total_pages=20)
        with self.assertRaises(ValueError):
            validate_page_range(start_page=0, end_page=5, total_pages=20)

    def test_fake_docling_payload_and_provenance(self) -> None:
        """Verifica serialización textual y detección de contención de cajas."""
        table = {
            "table_id": "test_tab",
            "dimensions": {"rows": 2, "cols": 2},
            "caption_text": "Cuadro Test",
            "footnotes": ["Nota al pie"],
            "cells": [
                {"row_start": 0, "row_end": 1, "col_start": 0, "col_end": 1, "text": "H1"},
                {"row_start": 0, "row_end": 1, "col_start": 1, "col_end": 2, "text": "H2"},
                {"row_start": 1, "row_end": 2, "col_start": 0, "col_end": 1, "text": "V1"},
                {"row_start": 1, "row_end": 2, "col_start": 1, "col_end": 2, "text": "V2"},
            ],
        }
        text_table = _format_table_as_text(table)
        self.assertIn("### Cuadro Test", text_table)
        self.assertIn("| H1 | H2 |", text_table)
        self.assertIn("| V1 | V2 |", text_table)
        self.assertIn("* Nota al pie", text_table)
        
        # Verificación de contención geométrica
        inner = {"l": 105.0, "t": 105.0, "r": 195.0, "b": 195.0}
        outer = {"l": 100.0, "t": 100.0, "r": 200.0, "b": 200.0}
        disjoint = {"l": 300.0, "t": 300.0, "r": 400.0, "b": 400.0}
        self.assertTrue(_bbox_inside(inner, outer))
        self.assertFalse(_bbox_inside(disjoint, outer))

    def test_mixed_origin_coordinate_normalization_and_containment(self) -> None:
        """Verifica la normalización de BOTTOMLEFT (Docling) a TOPLEFT con page_h real."""
        page_h = 800.0
        
        # En BOTTOMLEFT de Docling: origen abajo-izquierda, t > b
        # Un objeto físico de y=100 a y=300 desde el borde superior tiene en BOTTOMLEFT:
        # bottom = 800 - 300 = 500, top = 800 - 100 = 700
        native_bottomleft = {"l": 50.0, "r": 200.0, "b": 500.0, "t": 700.0, "coord_origin": "BOTTOMLEFT"}
        normalized = normalize_bbox_to_topleft(native_bottomleft, page_h)
        self.assertIsNotNone(normalized)
        self.assertEqual(normalized["t"], 100.0)
        self.assertEqual(normalized["b"], 300.0)
        self.assertEqual(normalized["l"], 50.0)
        self.assertEqual(normalized["r"], 200.0)
        self.assertEqual(normalized["coord_origin"], "TOPLEFT")
        
        # Objeto sin bbox debe ser None
        self.assertIsNone(normalize_bbox_to_topleft(None, page_h))

    def test_mixed_origin_fully_contained_suppression_and_partial_overlap(self) -> None:
        """Verifica detección de contención total, solapamiento parcial y separación entre orígenes mixtos."""
        page_h = 1000.0
        # Tabla revisada en TOPLEFT: [50, 100, 250, 400]
        reviewed_topleft = {"l": 50.0, "t": 100.0, "r": 250.0, "b": 400.0, "coord_origin": "TOPLEFT"}
        
        # Caso 1: Nativa totalmente contenida (físicamente [60, 150, 240, 350])
        # En BOTTOMLEFT: bottom = 1000 - 350 = 650, top = 1000 - 150 = 850
        native_contained = {"l": 60.0, "r": 240.0, "b": 650.0, "t": 850.0, "coord_origin": "BOTTOMLEFT"}
        self.assertEqual(check_bbox_relationship(native_contained, reviewed_topleft, page_h), "contained")
        
        # Caso 2: Nativa con solapamiento parcial (físicamente [60, 250, 350, 500])
        # En BOTTOMLEFT: bottom = 1000 - 500 = 500, top = 1000 - 250 = 750
        native_partial = {"l": 60.0, "r": 350.0, "b": 500.0, "t": 750.0, "coord_origin": "BOTTOMLEFT"}
        self.assertEqual(check_bbox_relationship(native_partial, reviewed_topleft, page_h), "partial")
        
        # Caso 3: Nativa totalmente separada (físicamente [60, 800, 200, 900])
        # En BOTTOMLEFT: bottom = 1000 - 900 = 100, top = 1000 - 800 = 200
        native_disjoint = {"l": 60.0, "r": 200.0, "b": 100.0, "t": 200.0, "coord_origin": "BOTTOMLEFT"}
        self.assertEqual(check_bbox_relationship(native_disjoint, reviewed_topleft, page_h), "disjoint")

    def test_top_left_reading_order_with_mixed_origins(self) -> None:
        """Verifica que el orden de lectura en páginas de una columna use TOPLEFT normalizado con page_h."""
        page_h = 1000.0
        
        # Bloque A: Título superior (nativo BOTTOMLEFT, t=950, b=900 -> TOPLEFT t=50)
        block_a = {
            "type": "text",
            "text": "Título Superior",
            "bbox": {"l": 50.0, "r": 500.0, "b": 900.0, "t": 950.0, "coord_origin": "BOTTOMLEFT"},
        }
        # Bloque B: Tabla revisada (TOPLEFT, t=200, b=400)
        block_b = {
            "type": "table",
            "text": "Tabla Central",
            "bbox": {"l": 50.0, "r": 500.0, "t": 200.0, "b": 400.0, "coord_origin": "TOPLEFT"},
        }
        # Bloque C: Texto inferior (nativo BOTTOMLEFT, t=500, b=400 -> TOPLEFT t=500)
        block_c = {
            "type": "text",
            "text": "Texto Inferior",
            "bbox": {"l": 50.0, "r": 500.0, "b": 400.0, "t": 500.0, "coord_origin": "BOTTOMLEFT"},
        }
        # Bloque D: Sin bbox
        block_d = {
            "type": "text",
            "text": "Texto Sin Bbox",
            "bbox": None,
        }
        
        blocks = [block_c, block_d, block_a, block_b]
        
        def _sort_key(b: dict) -> tuple[float, float]:
            nb = normalize_bbox_to_topleft(b.get("bbox"), page_h)
            if nb is None:
                return (float("inf"), float("inf"))
            return (nb["t"], nb["l"])
            
        blocks.sort(key=_sort_key)
        
        ordered_texts = [b["text"] for b in blocks]
        self.assertEqual(ordered_texts, ["Título Superior", "Tabla Central", "Texto Inferior", "Texto Sin Bbox"])

    def test_absent_optional_dependencies_lazy_failure(self) -> None:
        """Verifica fallos limpios y accionables cuando dependencias opcionales faltan."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir) / "test.pdf"
            tmp_path.write_bytes(b"%PDF-1.4\n...")
            
            # Falla pypdfium2 en extract_reviewed_tables
            with patch.dict(sys.modules, {"pypdfium2": None}):
                with self.assertRaises(ImportError) as cm:
                    extract_reviewed_tables(tmp_path, {}, 1, 1)
                self.assertIn("pypdfium2 is required", str(cm.exception))
                
            # Falla docling en extract_document
            fake_rec = [PageRecord(doc_id="a"*64, source_file="test.pdf", published_at="2025-03-01", page_number=1, text="legacy")]
            with patch("rag_pensiones.document_layout.extract_pages", return_value=fake_rec):
                with patch.dict(sys.modules, {"docling": None, "docling.document_converter": None}):
                    with self.assertRaises(ImportError) as cm:
                        extract_document(tmp_path, "2025-03-01", 1, 1, artifacts_path=Path("tmp"))
                    self.assertIn("docling is required", str(cm.exception))
                
            # Falla duckdb en write_bundle
            bundle = {
                "schema_version": "document-layout-v1",
                "metadata": {"doc_id": "test", "page_range": [1, 1]},
                "pages": [],
                "tables": [],
            }
            with patch.dict(sys.modules, {"duckdb": None}):
                with tempfile.TemporaryDirectory() as out_tmp:
                    with self.assertRaises(ImportError) as cm:
                        write_bundle(bundle, Path(out_tmp) / "out", with_duckdb=True)
                    self.assertIn("duckdb is required", str(cm.exception))

    def test_write_bundle_no_overwrite_and_duckdb_roundtrip(self) -> None:
        """Verifica que write_bundle rechace sobreescrituras y DuckDB conserve tipos VARCHAR y valores exactos."""
        import duckdb
        
        doc_id = "c" * 64
        extraction_id = "ext_roundtrip_123"
        bundle = {
            "schema_version": "document-layout-v1",
            "metadata": {
                "doc_id": doc_id,
                "extraction_id": extraction_id,
                "source_file": "reporte.pdf",
                "published_at": "2025-03-01",
                "page_range": [95, 95],
                "source_sha256": doc_id,
                "profile_sha256": "def456",
            },
            "pages": [
                {
                    "page_number": 95,
                    "width": 595.32,
                    "height": 841.92,
                    "legacy_text": "Texto legacy 95",
                    "layout_text": "Texto layout 95",
                    "blocks": [],
                    "warnings": [],
                }
            ],
            "tables": [
                {
                    "table_id": f"table_{extraction_id}_p95_cuadro",
                    "method": "reviewed_profile",
                    "native_ref": None,
                    "page_numbers": [95],
                    "bbox": {"l": 10.0, "t": 20.0, "r": 30.0, "b": 40.0, "coord_origin": "TOPLEFT"},
                    "dimensions": {"rows": 1, "cols": 3},
                    "caption_text": "Cuadro 1",
                    "footnotes": ["Nota"],
                    "warnings": [],
                    "cells": [
                        {
                            "cell_id": f"table_{extraction_id}_p95_cuadro_r0c0_0",
                            "row_start": 0,
                            "row_end": 1,
                            "col_start": 0,
                            "col_end": 1,
                            "raw_text": "-15,4%",
                            "text": "-15,4%",
                            "is_empty": False,
                            "page_number": 95,
                            "bbox": {"l": 10.0, "t": 20.0, "r": 20.0, "b": 40.0, "coord_origin": "TOPLEFT"},
                            "column_header": False,
                            "row_header": False,
                            "header_path": ["Cabecera"],
                            "row_label": "Fila 1",
                        },
                        {
                            "cell_id": f"table_{extraction_id}_p95_cuadro_r0c1_1",
                            "row_start": 0,
                            "row_end": 1,
                            "col_start": 1,
                            "col_end": 2,
                            "raw_text": "0",
                            "text": "0",
                            "is_empty": False,
                            "page_number": 95,
                            "bbox": None,
                            "column_header": False,
                            "row_header": False,
                            "header_path": ["Cabecera"],
                            "row_label": "Fila 1",
                        },
                        {
                            "cell_id": f"table_{extraction_id}_p95_cuadro_r0c2_2",
                            "row_start": 0,
                            "row_end": 1,
                            "col_start": 2,
                            "col_end": 3,
                            "raw_text": "",
                            "text": "",
                            "is_empty": True,
                            "page_number": 95,
                            "bbox": None,
                            "column_header": False,
                            "row_header": False,
                            "header_path": ["Cabecera"],
                            "row_label": "Fila 1",
                        },
                    ],
                }
            ],
            "native_docling": {"document": "test"},
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            out_path = Path(temp_dir) / "output_bundle"
            
            # 1. Primera escritura
            res = write_bundle(bundle, out_path, with_duckdb=True)
            self.assertTrue(out_path.exists())
            self.assertTrue((out_path / "structured.json").exists())
            self.assertTrue((out_path / "native_docling.json").exists())
            self.assertTrue((out_path / "pages.jsonl").exists())
            self.assertTrue((out_path / "tables.duckdb").exists())
            self.assertTrue((out_path / "manifest.json").exists())
            
            # 2. Protección contra sobreescritura
            with self.assertRaises(FileExistsError):
                write_bundle(bundle, out_path, with_duckdb=True)
                
            # 3. Roundtrip y verificación SQL en DuckDB
            conn = duckdb.connect(str(out_path / "tables.duckdb"))
            try:
                # Comprobar documentos
                doc_row = conn.execute("SELECT doc_id, source_file, published_at FROM documents").fetchone()
                self.assertEqual(doc_row[0], doc_id)
                self.assertEqual(doc_row[1], "reporte.pdf")
                self.assertEqual(doc_row[2], "2025-03-01")
                
                # Comprobar tablas
                tab_row = conn.execute("SELECT table_id, method, num_rows, num_cols FROM tables").fetchone()
                self.assertEqual(tab_row[0], f"table_{extraction_id}_p95_cuadro")
                self.assertEqual(tab_row[1], "reviewed_profile")
                self.assertEqual(tab_row[2], 1)
                self.assertEqual(tab_row[3], 3)
                
                # Comprobar celdas y preservación exacta de VARCHAR
                cell_rows = conn.execute(
                    "SELECT col_start, raw_text, text, is_empty FROM cells ORDER BY col_start"
                ).fetchall()
                self.assertEqual(len(cell_rows), 3)
                
                # Celda 0: -15,4%
                self.assertEqual(cell_rows[0][1], "-15,4%")
                self.assertEqual(cell_rows[0][2], "-15,4%")
                self.assertFalse(cell_rows[0][3])
                
                # Celda 1: 0 (no vacío)
                self.assertEqual(cell_rows[1][1], "0")
                self.assertEqual(cell_rows[1][2], "0")
                self.assertFalse(cell_rows[1][3])
                
                # Celda 2: vacío
                self.assertEqual(cell_rows[2][1], "")
                self.assertEqual(cell_rows[2][2], "")
                self.assertTrue(cell_rows[2][3])
            finally:
                conn.close()

    def test_deterministic_extraction_id(self) -> None:
        """Verifica generación determinista de extraction_id y diferenciación ante cambios."""
        id1 = generate_extraction_id("docA", [1, 5], "sha_prof1", {"mode": "acc"}, {"docling": "2.143.0"})
        id2 = generate_extraction_id("docA", [1, 5], "sha_prof1", {"mode": "acc"}, {"docling": "2.143.0"})
        self.assertEqual(id1, id2)
        
        # Distinto document SHA
        id_doc = generate_extraction_id("docB", [1, 5], "sha_prof1", {"mode": "acc"}, {"docling": "2.143.0"})
        self.assertNotEqual(id1, id_doc)
        
        # Distinto page_range
        id_range = generate_extraction_id("docA", [1, 6], "sha_prof1", {"mode": "acc"}, {"docling": "2.143.0"})
        self.assertNotEqual(id1, id_range)
        
        # Distinto profile SHA
        id_prof = generate_extraction_id("docA", [1, 5], "sha_prof2", {"mode": "acc"}, {"docling": "2.143.0"})
        self.assertNotEqual(id1, id_prof)
        
        # Distinto settings / version
        id_sett = generate_extraction_id("docA", [1, 5], "sha_prof1", {"mode": "fast"}, {"docling": "2.143.0"})
        self.assertNotEqual(id1, id_sett)
        
        self.assertIsInstance(id1, str)
        self.assertGreater(len(id1), 0)

    def test_multipage_native_table_different_heights_and_partial_override(self) -> None:
        """Verifica la evaluación multipágina con alturas independientes y no supresión parcial."""
        from types import SimpleNamespace
        
        p1 = SimpleNamespace(size=SimpleNamespace(width=600.0, height=800.0))
        p2 = SimpleNamespace(size=SimpleNamespace(width=600.0, height=1200.0))
        docling_pages = {1: p1, 2: p2}
        
        # Región de anulación solo en página 1
        override_boxes_by_page = {
            1: [{"l": 50.0, "t": 100.0, "r": 250.0, "b": 400.0, "coord_origin": "TOPLEFT"}],
            2: [],
        }
        
        prov1 = SimpleNamespace(
            page_no=1,
            bbox=SimpleNamespace(l=60.0, r=240.0, b=450.0, t=650.0, coord_origin=SimpleNamespace(value="BOTTOMLEFT")),
        )
        prov2 = SimpleNamespace(
            page_no=2,
            bbox=SimpleNamespace(l=60.0, r=240.0, b=900.0, t=1100.0, coord_origin=SimpleNamespace(value="BOTTOMLEFT")),
        )
        
        def evaluate_multipage_coverage(prov_list, pages_map, overrides):
            total_prov = len(prov_list)
            covered = 0
            partial = 0
            unknown = 0
            warnings = []
            
            if total_prov == 0:
                unknown += 1
                warnings.append("Unknown table provenance: native table kept")
            else:
                for prov_item in prov_list:
                    prov_p = getattr(prov_item, "page_no", None)
                    if prov_p is None:
                        unknown += 1
                        warnings.append("Unknown provenance page_no: native table kept")
                        continue
                    doc_p = pages_map.get(prov_p)
                    if doc_p is None or not hasattr(doc_p, "size") or doc_p.size is None:
                        unknown += 1
                        warnings.append(f"Missing actual page height for page {prov_p}: native table kept")
                        continue
                    actual_h = float(doc_p.size.height)
                    pb = getattr(prov_item, "bbox", None)
                    if pb is None:
                        unknown += 1
                        warnings.append(f"Provenance bbox on page {prov_p} is null: native table kept")
                        continue
                    prov_box = {
                        "l": float(pb.l),
                        "t": float(pb.t),
                        "r": float(pb.r),
                        "b": float(pb.b),
                        "coord_origin": getattr(pb.coord_origin, "value", str(pb.coord_origin)),
                    }
                    prov_status = "disjoint"
                    for obox in overrides.get(prov_p, []):
                        rel = check_bbox_relationship(prov_box, obox, actual_h)
                        if rel == "contained":
                            prov_status = "contained"
                            break
                        elif rel == "partial":
                            prov_status = "partial"
                    if prov_status == "contained":
                        covered += 1
                    elif prov_status == "partial":
                        partial += 1
            should_suppress = (total_prov > 0 and covered == total_prov and unknown == 0)
            if not should_suppress and (covered > 0 or partial > 0):
                warnings.append("Partial coverage on multipage table: native table kept to prevent loss of unoverridden pages")
            elif unknown > 0:
                warnings.append("Incomplete provenance metadata: native table kept")
            return should_suppress, warnings

        # Caso 1: Cobertura parcial (P1 cubierta, P2 descubierta)
        suppress1, warn1 = evaluate_multipage_coverage([prov1, prov2], docling_pages, override_boxes_by_page)
        self.assertFalse(suppress1)
        self.assertTrue(any("Partial coverage on multipage table" in w for w in warn1))
        
        # Caso 2: Cobertura total (P1 y P2 ambas cubiertas usando sus respectivas alturas)
        override_both = {
            1: [{"l": 50.0, "t": 100.0, "r": 250.0, "b": 400.0, "coord_origin": "TOPLEFT"}],
            2: [{"l": 50.0, "t": 50.0, "r": 250.0, "b": 350.0, "coord_origin": "TOPLEFT"}],
        }
        suppress2, warn2 = evaluate_multipage_coverage([prov1, prov2], docling_pages, override_both)
        self.assertTrue(suppress2)
        
        # Caso 3: Falta altura en P2 (no inventar default)
        suppress3, warn3 = evaluate_multipage_coverage([prov1, prov2], {1: p1}, override_both)
        self.assertFalse(suppress3)
        self.assertTrue(any("Missing actual page height for page 2" in w for w in warn3))

    def test_effective_native_tables_in_layout_text_and_multipage_handling(self) -> None:
        """Verifica la inclusión de tablas nativas efectivas en layout_text, preservación de escalares y manejo multipágina."""
        from types import SimpleNamespace
        
        def _make_docling_cell(r: int, c: int, text: str):
            return SimpleNamespace(
                start_row_offset_idx=r,
                end_row_offset_idx=r + 1,
                start_col_offset_idx=c,
                end_col_offset_idx=c + 1,
                text=text,
                bbox=SimpleNamespace(l=50.0, r=200.0, b=500.0, t=550.0, coord_origin=SimpleNamespace(value="BOTTOMLEFT")),
                column_header=(r == 0),
                row_header=False,
            )

        fake_docling_mods = _get_fake_docling_modules()
        with patch.dict(sys.modules, fake_docling_mods):
            with tempfile.TemporaryDirectory() as tmp_dir:
                pdf_path = Path(tmp_dir) / "sample.pdf"
                pdf_path.write_bytes(b"%PDF-1.4\n...")
                
                # 1. Caso sin perfil (no-profile fake-doc): tabla nativa única debe aparecer en layout_text
                p1_size = SimpleNamespace(width=600.0, height=800.0, size=SimpleNamespace(width=600.0, height=800.0))
                native_table1 = SimpleNamespace(
                    prov=[SimpleNamespace(page_no=1, bbox=SimpleNamespace(l=50.0, r=250.0, b=300.0, t=500.0, coord_origin=SimpleNamespace(value="BOTTOMLEFT")))],
                    data=SimpleNamespace(
                        num_rows=2,
                        num_cols=2,
                        table_cells=[
                            _make_docling_cell(0, 0, "ColA"),
                            _make_docling_cell(0, 1, "ColB"),
                            _make_docling_cell(1, 0, "Fila1"),
                            _make_docling_cell(1, 1, "SCALAR_NO_PROFILE_999"),
                        ],
                    ),
                    self_ref="#/tables/0",
                )
                fake_doc1 = SimpleNamespace(
                    export_to_dict=lambda: {"mock": "doc1"},
                    pages={1: p1_size},
                    tables=[native_table1],
                    iterate_items=lambda page_no: [
                        (SimpleNamespace(text="Párrafo Introductorio", prov=[SimpleNamespace(bbox=SimpleNamespace(l=50.0, r=300.0, b=700.0, t=750.0, coord_origin=SimpleNamespace(value="BOTTOMLEFT")))]), 0),
                        (native_table1, 0),
                    ],
                )
                fake_records1 = [PageRecord(doc_id="d1"*32, source_file=pdf_path.name, published_at="2025-03-01", page_number=1, text="legacy p1")]
                
                with patch("rag_pensiones.document_layout.extract_pages", return_value=fake_records1):
                    with patch("docling.document_converter.DocumentConverter") as MockConverter:
                        MockConverter.return_value.convert.return_value = SimpleNamespace(document=fake_doc1)
                        bundle1 = extract_document(pdf_path, "2025-03-01", 1, 1, artifacts_path=Path(tmp_dir))
                        
                        self.assertEqual(len(bundle1["pages"]), 1)
                        page1_layout = bundle1["pages"][0]["layout_text"]
                        self.assertIn("SCALAR_NO_PROFILE_999", page1_layout)
                        self.assertIn("Párrafo Introductorio", page1_layout)
                        self.assertEqual(len(bundle1["tables"]), 1)
                        self.assertEqual(bundle1["tables"][0]["method"], "docling_native")
                        self.assertEqual(bundle1["native_docling"], {"mock": "doc1"})
                        
                        # Verificar orden físico (párrafo arriba, tabla abajo según TOPLEFT)
                        intro_idx = page1_layout.index("Párrafo Introductorio")
                        scalar_idx = page1_layout.index("SCALAR_NO_PROFILE_999")
                        self.assertLess(intro_idx, scalar_idx)

                # 2. Caso con anulación parcial (partial-profile fake-doc):
                # Página 1 sobreescrita por tabla revisada; Página 2 conserva tabla nativa no sobreescrita
                p2_size = SimpleNamespace(width=600.0, height=800.0, size=SimpleNamespace(width=600.0, height=800.0))
                native_table2_p1 = SimpleNamespace(
                    prov=[SimpleNamespace(page_no=1, bbox=SimpleNamespace(l=50.0, r=250.0, b=300.0, t=500.0, coord_origin=SimpleNamespace(value="BOTTOMLEFT")))],
                    data=SimpleNamespace(
                        num_rows=1,
                        num_cols=1,
                        table_cells=[_make_docling_cell(0, 0, "OVERRIDDEN_NATIVE")],
                    ),
                    self_ref="#/tables/0",
                )
                native_table2_p2 = SimpleNamespace(
                    prov=[SimpleNamespace(page_no=2, bbox=SimpleNamespace(l=50.0, r=250.0, b=300.0, t=500.0, coord_origin=SimpleNamespace(value="BOTTOMLEFT")))],
                    data=SimpleNamespace(
                        num_rows=2,
                        num_cols=2,
                        table_cells=[
                            _make_docling_cell(0, 0, "H1"),
                            _make_docling_cell(0, 1, "H2"),
                            _make_docling_cell(1, 0, "Data1"),
                            _make_docling_cell(1, 1, "SCALAR_SURVIVING_PAGE2"),
                        ],
                    ),
                    self_ref="#/tables/1",
                )
                fake_doc2 = SimpleNamespace(
                    export_to_dict=lambda: {"mock": "doc2"},
                    pages={1: p1_size, 2: p2_size},
                    tables=[native_table2_p1, native_table2_p2],
                    iterate_items=lambda page_no: [
                        (native_table2_p1 if page_no == 1 else native_table2_p2, 0)
                    ],
                )
                fake_records2 = [
                    PageRecord(doc_id="d2"*32, source_file=pdf_path.name, published_at="2025-03-01", page_number=1, text="legacy p1"),
                    PageRecord(doc_id="d2"*32, source_file=pdf_path.name, published_at="2025-03-01", page_number=2, text="legacy p2"),
                ]
                
                doc2_id = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
                profile_partial = {
                    "doc_id": doc2_id,
                    "tables": [
                        {
                            "profile_id": "cuadro_p1",
                            "page_number": 1,
                            "page_size": [600.0, 800.0],
                            "row_edges": [280.0, 400.0, 520.0],
                            "column_edges": [40.0, 150.0, 260.0],
                            "header_rows": 1,
                            "spans": [],
                        }
                    ],
                }
                prof_path = Path(tmp_dir) / "profile.json"
                prof_path.write_text(json.dumps(profile_partial), encoding="utf-8")
                
                fake_reviewed = [
                    {
                        "table_id": "table_rev_p1_cuadro_p1",
                        "method": "reviewed_profile",
                        "native_ref": None,
                        "page_numbers": [1],
                        "bbox": {"l": 40.0, "t": 280.0, "r": 260.0, "b": 520.0, "coord_origin": "TOPLEFT"},
                        "dimensions": {"rows": 2, "cols": 2},
                        "cells": [
                            {"row_start": 0, "row_end": 1, "col_start": 0, "col_end": 1, "text": "RevH1"},
                            {"row_start": 0, "row_end": 1, "col_start": 1, "col_end": 2, "text": "RevH2"},
                            {"row_start": 1, "row_end": 2, "col_start": 0, "col_end": 1, "text": "RevD1"},
                            {"row_start": 1, "row_end": 2, "col_start": 1, "col_end": 2, "text": "SCALAR_REVIEWED_PAGE1"},
                        ],
                        "caption_text": "Tabla Revisada P1",
                        "footnotes": [],
                        "warnings": [],
                    }
                ]
                
                with patch("rag_pensiones.document_layout.extract_pages", return_value=fake_records2):
                    with patch("rag_pensiones.document_layout.extract_reviewed_tables", return_value=fake_reviewed):
                        with patch("docling.document_converter.DocumentConverter") as MockConverter:
                            MockConverter.return_value.convert.return_value = SimpleNamespace(document=fake_doc2)
                            bundle2 = extract_document(pdf_path, "2025-03-01", 1, 2, artifacts_path=Path(tmp_dir), regions_path=prof_path)
                            
                            # Página 1 tiene la tabla revisada y no la nativa sobreescrita
                            p1_layout = bundle2["pages"][0]["layout_text"]
                            self.assertIn("SCALAR_REVIEWED_PAGE1", p1_layout)
                            self.assertNotIn("OVERRIDDEN_NATIVE", p1_layout)
                            
                            # Página 2 tiene la tabla nativa preservada
                            p2_layout = bundle2["pages"][1]["layout_text"]
                            self.assertIn("SCALAR_SURVIVING_PAGE2", p2_layout)
                            
                            # Ambas tablas están en bundle["tables"]
                            table_methods = [t["method"] for t in bundle2["tables"]]
                            self.assertIn("reviewed_profile", table_methods)
                            self.assertIn("docling_native", table_methods)
                            self.assertEqual(bundle2["native_docling"], {"mock": "doc2"})

                # 3. Caso multipágina con procedencia ambigua:
                multipage_table = SimpleNamespace(
                    prov=[
                        SimpleNamespace(page_no=1, bbox=SimpleNamespace(l=50.0, r=250.0, b=300.0, t=500.0, coord_origin=SimpleNamespace(value="BOTTOMLEFT"))),
                        SimpleNamespace(page_no=2, bbox=SimpleNamespace(l=50.0, r=250.0, b=300.0, t=500.0, coord_origin=SimpleNamespace(value="BOTTOMLEFT"))),
                    ],
                    data=SimpleNamespace(
                        num_rows=2,
                        num_cols=2,
                        table_cells=[
                            _make_docling_cell(0, 0, "MultiH1"),
                            _make_docling_cell(0, 1, "MultiH2"),
                            _make_docling_cell(1, 0, "MultiVal1"),
                            _make_docling_cell(1, 1, "SCALAR_MULTIPAGE_777"),
                        ],
                    ),
                    self_ref="#/tables/multi",
                )
                fake_doc3 = SimpleNamespace(
                    export_to_dict=lambda: {"mock": "doc3"},
                    pages={1: p1_size, 2: p2_size},
                    tables=[multipage_table],
                    iterate_items=lambda page_no: [(multipage_table, 0)],
                )
                
                with patch("rag_pensiones.document_layout.extract_pages", return_value=fake_records2):
                    with patch("docling.document_converter.DocumentConverter") as MockConverter:
                        MockConverter.return_value.convert.return_value = SimpleNamespace(document=fake_doc3)
                        bundle3 = extract_document(pdf_path, "2025-03-01", 1, 2, artifacts_path=Path(tmp_dir), regions_path=None)
                        
                        p1_text = bundle3["pages"][0]["layout_text"]
                        p2_text = bundle3["pages"][1]["layout_text"]
                        
                        self.assertIn("[Tabla multipágina: páginas 1-2 (procedencia de celdas ambigua)]", p1_text)
                        self.assertIn("SCALAR_MULTIPAGE_777", p1_text)
                        
                        # Las celdas se preservan UNA SOLA VEZ (no duplicadas en página 2)
                        self.assertNotIn("SCALAR_MULTIPAGE_777", p2_text)
                        self.assertIn("Tabla multipágina", p2_text)
                        self.assertIn("ver página 1", p2_text)
                        
                        self.assertTrue(any("procedencia de celdas ambigua" in w for w in bundle3["pages"][0]["warnings"]))
                        self.assertTrue(any("renderizada en página 1" in w for w in bundle3["pages"][1]["warnings"]))
                        self.assertEqual(bundle3["native_docling"], {"mock": "doc3"})

    def test_format_table_as_text_colspan_and_rowspan_data_value_once(self) -> None:
        """Verifica que el formateador renderice el texto de celdas combinadas una sola vez en el ancla sin duplicar valores."""
        # 1. Colspan: celda abarca cols 0..2
        tbl_colspan = {
            "dimensions": {"rows": 1, "cols": 3},
            "cells": [
                {"row_start": 0, "row_end": 1, "col_start": 0, "col_end": 2, "text": "MergedCol100"},
                {"row_start": 0, "row_end": 1, "col_start": 2, "col_end": 3, "text": "Col3"},
            ],
        }
        res_col = _format_table_as_text(tbl_colspan)
        self.assertIn("| MergedCol100 |  | Col3 |", res_col)
        self.assertEqual(res_col.count("MergedCol100"), 1)
        
        # 2. Rowspan: celda abarca rows 0..2 en col 0
        tbl_rowspan = {
            "dimensions": {"rows": 2, "cols": 2},
            "cells": [
                {"row_start": 0, "row_end": 2, "col_start": 0, "col_end": 1, "text": "RowSpanHeader"},
                {"row_start": 0, "row_end": 1, "col_start": 1, "col_end": 2, "text": "R0C1"},
                {"row_start": 1, "row_end": 2, "col_start": 1, "col_end": 2, "text": "R1C1"},
            ],
        }
        res_row = _format_table_as_text(tbl_rowspan)
        self.assertEqual(res_row.count("RowSpanHeader"), 1)
        lines = res_row.splitlines()
        # Primera fila de datos: RowSpanHeader en col 0
        self.assertTrue(lines[0].startswith("| RowSpanHeader | R0C1 |"))
        # Segunda fila de datos (tras separador markdown): marcador vacío en col 0
        self.assertEqual(lines[2], "|  | R1C1 |")
        
        # 3. Celdas separadas con el mismo valor literal: deben preservarse ambas
        tbl_separate = {
            "dimensions": {"rows": 1, "cols": 2},
            "cells": [
                {"row_start": 0, "row_end": 1, "col_start": 0, "col_end": 1, "text": "SameVal"},
                {"row_start": 0, "row_end": 1, "col_start": 1, "col_end": 2, "text": "SameVal"},
            ],
        }
        res_sep = _format_table_as_text(tbl_separate)
        self.assertIn("| SameVal | SameVal |", res_sep)
        self.assertEqual(res_sep.count("SameVal"), 2)

    def test_profile_page_existence_in_actual_pdf(self) -> None:
        """Verifica rechazo estricto de páginas en perfiles que no existen en el PDF real."""
        valid_doc_id = "e" * 64
        table_p2 = {
            "profile_id": "cuadro_p2",
            "page_number": 2,
            "page_size": [600.0, 800.0],
            "row_edges": [10.0, 20.0, 30.0],
            "column_edges": [10.0, 20.0, 30.0],
            "header_rows": 1,
            "spans": [],
        }
        prof = {"doc_id": valid_doc_id, "tables": [table_p2]}
        
        # 1. El PDF solo tiene página 1: debe rechazarse porque la página 2 no existe en el PDF
        with self.assertRaises(ValueError) as cm1:
            validate_regions(prof, valid_doc_id, {1: (600.0, 800.0)})
        self.assertIn("profile page 2 does not exist in document pages", str(cm1.exception))
        
        # 2. El PDF tiene páginas 1 y 2: debe aceptarse sin error
        validate_regions(prof, valid_doc_id, {1: (600.0, 800.0), 2: (600.0, 800.0)})
        
        # 3. Rechazo de tipos booleanos o no enteros en page_number
        with self.assertRaises(ValueError):
            validate_regions(
                {"doc_id": valid_doc_id, "tables": [{**table_p2, "page_number": True}]},
                valid_doc_id,
                {1: (600.0, 800.0), 2: (600.0, 800.0)},
            )
            
        # 4. Rechazo en single_column_pages inexistentes en el PDF
        prof_single = {
            "doc_id": valid_doc_id,
            "tables": [],
            "single_column_pages": [2],
        }
        with self.assertRaises(ValueError) as cm2:
            validate_regions(prof_single, valid_doc_id, {1: (600.0, 800.0)})
        self.assertIn("single_column_page 2 does not exist", str(cm2.exception))
        
        # 5. Distinción: página 2 válida fuera del rango solicitado (ej. solicitando solo página 1 de un PDF de 2 páginas)
        with tempfile.TemporaryDirectory() as tmp_dir:
            pdf_path = Path(tmp_dir) / "test2p.pdf"
            pdf_path.write_bytes(b"%PDF-1.4\n...")
            
            # Mockeando pdfium para simular documento de 2 páginas
            from types import SimpleNamespace
            fake_page1 = SimpleNamespace(get_size=lambda: (600.0, 800.0), close=lambda: None)
            fake_page2 = SimpleNamespace(get_size=lambda: (600.0, 800.0), close=lambda: None)
            class FakePdfiumDoc:
                def __len__(self):
                    return 2
                def get_page(self, idx):
                    return fake_page1 if idx == 0 else fake_page2
                def close(self):
                    pass
            fake_pdfium_doc = FakePdfiumDoc()
            fake_pdfium_mod = SimpleNamespace(PdfDocument=lambda p: fake_pdfium_doc)
            
            prof_doc_id = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
            prof_2p = {"doc_id": prof_doc_id, "tables": [{**table_p2, "page_number": 2}]}
            
            with patch.dict(sys.modules, {"pypdfium2": fake_pdfium_mod}):
                # Solicitando página 1..1: la tabla de p2 está fuera del rango pero es válida; no debe fallar
                res = extract_reviewed_tables(pdf_path, prof_2p, start_page=1, end_page=1)
                self.assertEqual(len(res), 0)

    def test_two_column_reading_order_not_interleaved_without_single_column_profile(self) -> None:
        """Verifica que páginas de dos columnas retengan el orden nativo de Docling sin intercalar columnas geométricamente."""
        from types import SimpleNamespace
        
        fake_docling_mods = _get_fake_docling_modules()
        with patch.dict(sys.modules, fake_docling_mods):
            with tempfile.TemporaryDirectory() as tmp_dir:
                pdf_path = Path(tmp_dir) / "twocol.pdf"
                pdf_path.write_bytes(b"%PDF-1.4\n...")
                
                p1_size = SimpleNamespace(width=600.0, height=800.0, size=SimpleNamespace(width=600.0, height=800.0))
                # 4 bloques en orden nativo de Docling: columna izquierda de arriba a abajo, luego columna derecha
                b_lt = SimpleNamespace(text="Left Col Top", prov=[SimpleNamespace(bbox=SimpleNamespace(l=50.0, r=250.0, b=600.0, t=700.0, coord_origin=SimpleNamespace(value="BOTTOMLEFT")))])
                b_lb = SimpleNamespace(text="Left Col Bottom", prov=[SimpleNamespace(bbox=SimpleNamespace(l=50.0, r=250.0, b=400.0, t=500.0, coord_origin=SimpleNamespace(value="BOTTOMLEFT")))])
                b_rt = SimpleNamespace(text="Right Col Top", prov=[SimpleNamespace(bbox=SimpleNamespace(l=350.0, r=550.0, b=600.0, t=700.0, coord_origin=SimpleNamespace(value="BOTTOMLEFT")))])
                b_rb = SimpleNamespace(text="Right Col Bottom", prov=[SimpleNamespace(bbox=SimpleNamespace(l=350.0, r=550.0, b=400.0, t=500.0, coord_origin=SimpleNamespace(value="BOTTOMLEFT")))])
                
                fake_doc = SimpleNamespace(
                    export_to_dict=lambda: {"mock": "twocol"},
                    pages={1: p1_size},
                    tables=[],
                    iterate_items=lambda page_no: [(b_lt, 0), (b_lb, 0), (b_rt, 0), (b_rb, 0)],
                )
                fake_rec = [PageRecord(doc_id="f"*64, source_file=pdf_path.name, published_at="2025-03-01", page_number=1, text="legacy")]
                
                with patch("rag_pensiones.document_layout.extract_pages", return_value=fake_rec):
                    with patch("docling.document_converter.DocumentConverter") as MockConv:
                        MockConv.return_value.convert.return_value = SimpleNamespace(document=fake_doc)
                        bundle = extract_document(pdf_path, "2025-03-01", 1, 1, artifacts_path=Path(tmp_dir), regions_path=None)
                        
                        layout_text = bundle["pages"][0]["layout_text"]
                        # Debe mantener el orden nativo de dos columnas sin intercalarse por TOPLEFT
                        expected = "Left Col Top\n\nLeft Col Bottom\n\nRight Col Top\n\nRight Col Bottom"
                        self.assertEqual(layout_text, expected)

    def test_native_table_inline_position_without_profile(self) -> None:
        """Verifica que tablas nativas se intercalen en su posición original de iterate_items en lugar de apilarse al final."""
        from types import SimpleNamespace
        
        def _make_cell(r: int, c: int, text: str):
            return SimpleNamespace(
                start_row_offset_idx=r,
                end_row_offset_idx=r + 1,
                start_col_offset_idx=c,
                end_col_offset_idx=c + 1,
                text=text,
                bbox=SimpleNamespace(l=50.0, r=200.0, b=500.0, t=550.0, coord_origin=SimpleNamespace(value="BOTTOMLEFT")),
                column_header=(r == 0),
                row_header=False,
            )
            
        fake_docling_mods = _get_fake_docling_modules()
        with patch.dict(sys.modules, fake_docling_mods):
            with tempfile.TemporaryDirectory() as tmp_dir:
                pdf_path = Path(tmp_dir) / "inline_tab.pdf"
                pdf_path.write_bytes(b"%PDF-1.4\n...")
                
                p1_size = SimpleNamespace(width=600.0, height=800.0, size=SimpleNamespace(width=600.0, height=800.0))
                p_before = SimpleNamespace(text="Parrafo Inicial", prov=[SimpleNamespace(bbox=SimpleNamespace(l=50.0, r=300.0, b=700.0, t=750.0, coord_origin=SimpleNamespace(value="BOTTOMLEFT")))])
                p_after = SimpleNamespace(text="Parrafo Final", prov=[SimpleNamespace(bbox=SimpleNamespace(l=50.0, r=300.0, b=200.0, t=250.0, coord_origin=SimpleNamespace(value="BOTTOMLEFT")))])
                
                nat_tbl = SimpleNamespace(
                    prov=[SimpleNamespace(page_no=1, bbox=SimpleNamespace(l=50.0, r=250.0, b=400.0, t=500.0, coord_origin=SimpleNamespace(value="BOTTOMLEFT")))],
                    data=SimpleNamespace(num_rows=1, num_cols=1, table_cells=[_make_cell(0, 0, "VALOR_TABLA_INTERMEDIA")]),
                    self_ref="#/tables/0",
                )
                
                fake_doc = SimpleNamespace(
                    export_to_dict=lambda: {"mock": "inline"},
                    pages={1: p1_size},
                    tables=[nat_tbl],
                    iterate_items=lambda page_no: [(p_before, 0), (nat_tbl, 0), (p_after, 0)],
                )
                fake_rec = [PageRecord(doc_id="g"*64, source_file=pdf_path.name, published_at="2025-03-01", page_number=1, text="legacy")]
                
                with patch("rag_pensiones.document_layout.extract_pages", return_value=fake_rec):
                    with patch("docling.document_converter.DocumentConverter") as MockConv:
                        MockConv.return_value.convert.return_value = SimpleNamespace(document=fake_doc)
                        bundle = extract_document(pdf_path, "2025-03-01", 1, 1, artifacts_path=Path(tmp_dir), regions_path=None)
                        
                        txt = bundle["pages"][0]["layout_text"]
                        idx_before = txt.index("Parrafo Inicial")
                        idx_table = txt.index("VALOR_TABLA_INTERMEDIA")
                        idx_after = txt.index("Parrafo Final")
                        
                        self.assertLess(idx_before, idx_table)
                        self.assertLess(idx_table, idx_after)

    def test_page_warnings_empty_figure_and_mixed(self) -> None:
        """Verifica advertencias conservadoras para páginas vacías, solo figuras y mixtas texto+figura."""
        from types import SimpleNamespace
        
        fake_docling_mods = _get_fake_docling_modules()
        with patch.dict(sys.modules, fake_docling_mods):
            with tempfile.TemporaryDirectory() as tmp_dir:
                pdf_path = Path(tmp_dir) / "warnings.pdf"
                pdf_path.write_bytes(b"%PDF-1.4\n...")
                
                p_size = SimpleNamespace(width=600.0, height=800.0, size=SimpleNamespace(width=600.0, height=800.0))
                
                # Caso 1: Página física vacía (sin texto legacy ni blocks)
                fake_doc_empty = SimpleNamespace(
                    export_to_dict=lambda: {},
                    pages={1: p_size},
                    tables=[],
                    pictures=[],
                    iterate_items=lambda page_no: [],
                )
                fake_rec_empty = [PageRecord(doc_id="1"*64, source_file=pdf_path.name, published_at="2025-03-01", page_number=1, text="")]
                
                with patch("rag_pensiones.document_layout.extract_pages", return_value=fake_rec_empty):
                    with patch("docling.document_converter.DocumentConverter") as MockConv:
                        MockConv.return_value.convert.return_value = SimpleNamespace(document=fake_doc_empty)
                        b_empty = extract_document(pdf_path, "2025-03-01", 1, 1, artifacts_path=Path(tmp_dir))
                        
                        p1 = b_empty["pages"][0]
                        self.assertEqual(p1["layout_text"], "")
                        self.assertTrue(any("No extractable text found" in w for w in p1["warnings"]))
                        self.assertFalse(any("Non-text figure" in w for w in p1["warnings"]))
                
                # Caso 2: Página solo con figuras (sin texto)
                fake_pic = SimpleNamespace(prov=[SimpleNamespace(page_no=1)])
                fake_doc_fig = SimpleNamespace(
                    export_to_dict=lambda: {},
                    pages={1: p_size},
                    tables=[],
                    pictures=[fake_pic],
                    iterate_items=lambda page_no: [],
                )
                with patch("rag_pensiones.document_layout.extract_pages", return_value=fake_rec_empty):
                    with patch("docling.document_converter.DocumentConverter") as MockConv:
                        MockConv.return_value.convert.return_value = SimpleNamespace(document=fake_doc_fig)
                        b_fig = extract_document(pdf_path, "2025-03-01", 1, 1, artifacts_path=Path(tmp_dir))
                        
                        p1 = b_fig["pages"][0]
                        self.assertEqual(p1["layout_text"], "")
                        self.assertTrue(any("No extractable text found" in w for w in p1["warnings"]))
                        self.assertTrue(any("Non-text figure/image content" in w for w in p1["warnings"]))
                
                # Caso 3: Página mixta (texto de párrafo + figura)
                p_item = SimpleNamespace(text="Texto Normal", prov=[SimpleNamespace(bbox=SimpleNamespace(l=50.0, r=300.0, b=700.0, t=750.0, coord_origin=SimpleNamespace(value="BOTTOMLEFT")))])
                fake_doc_mixed = SimpleNamespace(
                    export_to_dict=lambda: {},
                    pages={1: p_size},
                    tables=[],
                    pictures=[fake_pic],
                    iterate_items=lambda page_no: [(p_item, 0)],
                )
                fake_rec_text = [PageRecord(doc_id="1"*64, source_file=pdf_path.name, published_at="2025-03-01", page_number=1, text="Texto legacy")]
                with patch("rag_pensiones.document_layout.extract_pages", return_value=fake_rec_text):
                    with patch("docling.document_converter.DocumentConverter") as MockConv:
                        MockConv.return_value.convert.return_value = SimpleNamespace(document=fake_doc_mixed)
                        b_mixed = extract_document(pdf_path, "2025-03-01", 1, 1, artifacts_path=Path(tmp_dir))
                        
                        p1 = b_mixed["pages"][0]
                        self.assertIn("Texto Normal", p1["layout_text"])
                        self.assertTrue(any("Non-text figure/image content" in w for w in p1["warnings"]))
                        self.assertFalse(any("No extractable text found" in w for w in p1["warnings"]))


if __name__ == "__main__":
    unittest.main()
