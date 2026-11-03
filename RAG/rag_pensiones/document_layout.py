"""Extracción de layout de documentos con Docling local y tablas revisadas."""

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
from typing import Any

from rag_pensiones.ingest import (
    PageRecord,
    extract_pages,
    validate_page_range,
    validate_published_at,
)
from rag_pensiones.table_geometry import extract_reviewed_tables


def _format_table_as_text(table: dict[str, Any]) -> str:
    """Formatea una tabla estructurada como texto Markdown legible para layout_text."""
    lines: list[str] = []
    if table.get("caption_text"):
        lines.append(f"### {table['caption_text']}")
    
    rows_count = table["dimensions"]["rows"]
    cols_count = table["dimensions"]["cols"]
    grid: list[list[str]] = [["" for _ in range(cols_count)] for _ in range(rows_count)]
    
    for cell in table.get("cells", []):
        r_s, r_e = cell["row_start"], cell["row_end"]
        c_s, c_e = cell["col_start"], cell["col_end"]
        val = str(cell.get("text", ""))
        # Renderizar cada celda combinada exactamente una vez en su ancla (r_s, c_s)
        if 0 <= r_s < rows_count and 0 <= c_s < cols_count:
            grid[r_s][c_s] = val
        # Las restantes posiciones cubiertas quedan como marcadores vacíos "" (sin replicar valores)
        for r in range(r_s, min(r_e, rows_count)):
            for c in range(c_s, min(c_e, cols_count)):
                if (r, c) != (r_s, c_s):
                    grid[r][c] = ""
    
    for r_idx, row in enumerate(grid):
        lines.append("| " + " | ".join(row) + " |")
        if r_idx == 0:
            lines.append("| " + " | ".join(["---"] * cols_count) + " |")
            
    for fn in table.get("footnotes", []):
        lines.append(f"* {fn}")
        
    return "\n".join(lines)


def normalize_bbox_to_topleft(bbox: dict[str, Any] | None, page_h: float) -> dict[str, float] | None:
    """Normaliza cualquier BoundingBox (BOTTOMLEFT o TOPLEFT) a TOPLEFT usando la altura real de página."""
    if bbox is None:
        return None
    origin = str(bbox.get("coord_origin", "TOPLEFT")).upper()
    l = float(bbox["l"])
    r = float(bbox["r"])
    t = float(bbox["t"])
    b = float(bbox["b"])
    norm_l = min(l, r)
    norm_r = max(l, r)
    if "BOTTOM" in origin:
        norm_t = page_h - max(t, b)
        norm_b = page_h - min(t, b)
    else:
        norm_t = min(t, b)
        norm_b = max(t, b)
    return {"l": norm_l, "t": norm_t, "r": norm_r, "b": norm_b, "coord_origin": "TOPLEFT"}


def check_bbox_relationship(inner: dict[str, Any] | None, outer: dict[str, Any] | None, page_h: float) -> str:
    """Determina si inner está 'contained', 'partial' o 'disjoint' respecto a outer en TOPLEFT."""
    if inner is None or outer is None:
        return "disjoint"
    in_norm = normalize_bbox_to_topleft(inner, page_h)
    out_norm = normalize_bbox_to_topleft(outer, page_h)
    if in_norm is None or out_norm is None:
        return "disjoint"
    
    tol = 3.0
    is_inside = (
        in_norm["l"] >= out_norm["l"] - tol
        and in_norm["r"] <= out_norm["r"] + tol
        and in_norm["t"] >= out_norm["t"] - tol
        and in_norm["b"] <= out_norm["b"] + tol
    )
    if is_inside:
        return "contained"
    
    inter_l = max(in_norm["l"], out_norm["l"])
    inter_r = min(in_norm["r"], out_norm["r"])
    inter_t = max(in_norm["t"], out_norm["t"])
    inter_b = min(in_norm["b"], out_norm["b"])
    
    if inter_r > inter_l and inter_b > inter_t:
        return "partial"
    return "disjoint"


def _bbox_inside(inner: dict[str, Any] | None, outer: dict[str, Any] | None) -> bool:
    """Compatibilidad básica de contención directa."""
    if not inner or not outer:
        return False
    tol = 5.0
    return (
        inner["l"] >= outer["l"] - tol
        and inner["r"] <= outer["r"] + tol
        and inner["t"] >= outer["t"] - tol
        and inner["b"] <= outer["b"] + tol
    )


def generate_extraction_id(
    doc_id: str,
    page_range: list[int] | tuple[int, int],
    profile_sha: str | None = None,
    settings: dict[str, Any] | None = None,
    package_versions: dict[str, str] | None = None,
) -> str:
    """Genera un identificador de extracción determinista a partir de docSHA, rango, perfil y entorno."""
    payload = {
        "doc_id": doc_id,
        "page_range": [int(page_range[0]), int(page_range[1])],
        "profile_sha": profile_sha or "none",
        "settings": settings or {},
        "package_versions": package_versions or {},
    }
    canon = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()[:16]


def extract_document(
    pdf_path: Path,
    published_at: str,
    start_page: int = 1,
    end_page: int | None = None,
    *,
    artifacts_path: Path,
    regions_path: Path | None = None,
) -> dict[str, Any]:
    """Extrae páginas y tablas de un PDF usando Docling local y perfiles revisados."""
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    data = pdf_path.read_bytes()
    if len(data) == 0:
        raise ValueError("PDF file is empty.")
    
    valid_date = validate_published_at(published_at)
    doc_id = hashlib.sha256(data).hexdigest()
    
    # 1. Extracción de referencia pypdf (baseline y legacy_text)
    legacy_records = extract_pages(
        pdf_path=pdf_path,
        published_at=valid_date,
        start_page=start_page,
        end_page=end_page,
    )
    if not legacy_records:
        raise ValueError("No pages extracted from PDF.")
    
    start = legacy_records[0].page_number
    end = legacy_records[-1].page_number
    legacy_by_page = {rec.page_number: rec.text for rec in legacy_records}
    
    # 2. Entorno y configuración determinista
    pkg_versions: dict[str, str] = {}
    for pkg in ("docling", "docling-core", "docling-ibm-models", "pypdfium2", "duckdb", "pypdf"):
        try:
            pkg_versions[pkg] = importlib.metadata.version(pkg)
        except Exception:
            pass
            
    extraction_settings = {
        "table_structure_mode": "accurate",
        "cell_matching": True,
        "ocr": False,
        "cpu_threads": 4,
    }
    
    profile: dict[str, Any] | None = None
    profile_sha256: str | None = None
    reviewed_tables: list[dict[str, Any]] = []
    
    if regions_path is not None:
        rpath = Path(regions_path)
        if not rpath.exists():
            raise FileNotFoundError(f"Regions file not found: {rpath}")
        rbytes = rpath.read_bytes()
        profile_sha256 = hashlib.sha256(rbytes).hexdigest()
        profile = json.loads(rbytes.decode("utf-8"))
        
    extraction_id = generate_extraction_id(
        doc_id=doc_id,
        page_range=[start, end],
        profile_sha=profile_sha256,
        settings=extraction_settings,
        package_versions=pkg_versions,
    )
    
    if profile is not None:
        reviewed_tables = extract_reviewed_tables(
            pdf_path=pdf_path,
            profile=profile,
            start_page=start,
            end_page=end,
            extraction_id=extraction_id,
        )
    
    # 3. Conversión Docling
    os.environ["HF_HUB_OFFLINE"] = "1"
    try:
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import (
            PdfPipelineOptions,
            TableStructureOptions,
            TableFormerMode,
            AcceleratorOptions,
        )
    except ImportError as exc:
        raise ImportError(
            "docling is required for document extraction. Install requirements-layout.txt."
        ) from exc
    
    pipe_opts = PdfPipelineOptions()
    pipe_opts.do_ocr = False
    pipe_opts.do_code_enrichment = False
    pipe_opts.do_formula_enrichment = False
    pipe_opts.do_picture_classification = False
    pipe_opts.do_picture_description = False
    pipe_opts.enable_remote_services = False
    pipe_opts.accelerator_options = AcceleratorOptions(num_threads=4)
    pipe_opts.table_structure_options = TableStructureOptions(
        mode=TableFormerMode.ACCURATE,
        do_cell_matching=True,
    )
    if artifacts_path is not None:
        pipe_opts.artifacts_path = Path(artifacts_path)
        
    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipe_opts)}
    )
    conv_result = converter.convert(pdf_path, page_range=(start, end))
    docling_doc = conv_result.document
    native_export = docling_doc.export_to_dict()
    
    # 4. Procesar tablas
    effective_tables: list[dict[str, Any]] = []
    effective_tables.extend(reviewed_tables)
    
    # Mapas de regiones sobreescritas por página
    override_boxes_by_page: dict[int, list[dict[str, Any]]] = {}
    for rt in reviewed_tables:
        for p in rt["page_numbers"]:
            override_boxes_by_page.setdefault(p, []).append(rt["bbox"])
    
    effective_native_by_item_id: dict[int, dict[str, Any]] = {}
    effective_native_by_ref: dict[str, dict[str, Any]] = {}
    
    # Convertir tablas nativas de Docling si no están sobreescritas
    for idx, nat_table in enumerate(getattr(docling_doc, "tables", [])):
        prov_list = getattr(nat_table, "prov", []) or []
        t_pages: list[int] = []
        t_bbox: dict[str, Any] | None = None
        t_warnings: list[str] = []
        
        if prov_list:
            for p_item in prov_list:
                if hasattr(p_item, "page_no") and p_item.page_no is not None:
                    t_pages.append(p_item.page_no)
            p0 = prov_list[0]
            if hasattr(p0, "bbox") and p0.bbox:
                t_bbox = {
                    "l": float(p0.bbox.l),
                    "t": float(p0.bbox.t),
                    "r": float(p0.bbox.r),
                    "b": float(p0.bbox.b),
                    "coord_origin": getattr(p0.bbox.coord_origin, "value", str(p0.bbox.coord_origin)),
                }
            else:
                t_warnings.append("Native table provenance bbox is null")
        else:
            t_warnings.append("Unknown table provenance: native table kept")
        
        # Evaluar cada provenance en su propia página y con su altura real (sin inventar default)
        total_prov = len(prov_list)
        covered_prov_count = 0
        partial_prov_count = 0
        unknown_prov_count = 0
        
        if total_prov == 0:
            unknown_prov_count += 1
        else:
            for prov_item in prov_list:
                prov_p = getattr(prov_item, "page_no", None)
                if prov_p is None:
                    unknown_prov_count += 1
                    t_warnings.append("Unknown provenance page_no: native table kept")
                    continue
                
                doc_p_item = docling_doc.pages.get(prov_p) if hasattr(docling_doc, "pages") else None
                actual_page_h = None
                if doc_p_item is not None:
                    if hasattr(doc_p_item, "size") and doc_p_item.size is not None and hasattr(doc_p_item.size, "height"):
                        actual_page_h = float(doc_p_item.size.height)
                    elif hasattr(doc_p_item, "height") and doc_p_item.height is not None:
                        actual_page_h = float(doc_p_item.height)
                
                if actual_page_h is None:
                    unknown_prov_count += 1
                    t_warnings.append(f"Missing actual page height for page {prov_p}: native table kept")
                    continue
                pb = getattr(prov_item, "bbox", None)
                if pb is None:
                    unknown_prov_count += 1
                    t_warnings.append(f"Provenance bbox on page {prov_p} is null: native table kept")
                    continue
                
                prov_box = {
                    "l": float(pb.l),
                    "t": float(pb.t),
                    "r": float(pb.r),
                    "b": float(pb.b),
                    "coord_origin": getattr(pb.coord_origin, "value", str(pb.coord_origin)),
                }
                
                prov_status = "disjoint"
                for obox in override_boxes_by_page.get(prov_p, []):
                    rel = check_bbox_relationship(prov_box, obox, actual_page_h)
                    if rel == "contained":
                        prov_status = "contained"
                        break
                    elif rel == "partial":
                        prov_status = "partial"
                
                if prov_status == "contained":
                    covered_prov_count += 1
                elif prov_status == "partial":
                    partial_prov_count += 1
        
        # Supresión estricta: ÚNICAMENTE si TODAS las áreas de provenance están totalmente cubiertas
        if total_prov > 0 and covered_prov_count == total_prov and unknown_prov_count == 0:
            continue
        elif covered_prov_count > 0 or partial_prov_count > 0:
            t_warnings.append(
                "Partial coverage on multipage table: native table kept to prevent loss of unoverridden pages"
            )
        elif unknown_prov_count > 0:
            t_warnings.append("Incomplete provenance metadata: native table kept")
        
        # Convertir celdas nativas con nombrespaced IDs
        table_id = f"table_{extraction_id}_native_{idx}"
        table_cells: list[dict[str, Any]] = []
        for c_idx, c in enumerate(nat_table.data.table_cells):
            c_bbox = None
            if hasattr(c, "bbox") and c.bbox is not None:
                c_bbox = {
                    "l": float(c.bbox.l),
                    "t": float(c.bbox.t),
                    "r": float(c.bbox.r),
                    "b": float(c.bbox.b),
                    "coord_origin": getattr(c.bbox.coord_origin, "value", str(c.bbox.coord_origin)),
                }
            else:
                t_warnings.append("Native cell bbox is null")
            
            c_page = None
            unique_pages = list(set(t_pages))
            if len(unique_pages) == 1:
                c_page = unique_pages[0]
            elif len(unique_pages) > 1:
                t_warnings.append("Ambiguous multipage cell page_number is null")
            else:
                t_warnings.append("Cell page_number is null due to missing table page provenance")
            
            raw_c_text = str(c.text or "")
            c_text = raw_c_text.strip()
            cell_id = f"{table_id}_r{c.start_row_offset_idx}c{c.start_col_offset_idx}_{c_idx}"
            table_cells.append({
                "cell_id": cell_id,
                "row_start": c.start_row_offset_idx,
                "row_end": c.end_row_offset_idx,
                "col_start": c.start_col_offset_idx,
                "col_end": c.end_col_offset_idx,
                "raw_text": raw_c_text,
                "text": c_text,
                "is_empty": (len(c_text) == 0),
                "page_number": c_page,
                "bbox": c_bbox,
                "column_header": bool(c.column_header),
                "row_header": bool(c.row_header),
                "header_path": [],
                "row_label": "",
            })
            
        caption_text = ""
        if hasattr(nat_table, "caption_text"):
            caption_text = str(nat_table.caption_text(docling_doc) or "")
            
        nat_tbl_dict = {
            "table_id": table_id,
            "method": "docling_native",
            "native_ref": getattr(nat_table, "self_ref", f"#/tables/{idx}"),
            "page_numbers": t_pages,
            "bbox": t_bbox,
            "dimensions": {
                "rows": nat_table.data.num_rows,
                "cols": nat_table.data.num_cols,
            },
            "cells": table_cells,
            "caption_text": caption_text,
            "footnotes": [],
            "warnings": list(set(t_warnings)),
        }
        effective_tables.append(nat_tbl_dict)
        effective_native_by_item_id[id(nat_table)] = nat_tbl_dict
        if hasattr(nat_table, "self_ref") and nat_table.self_ref:
            effective_native_by_ref[nat_table.self_ref] = nat_tbl_dict
        effective_native_by_ref[f"#/tables/{idx}"] = nat_tbl_dict
    
    # 5. Construir páginas y layout_text
    single_col_pages = set(profile.get("single_column_pages", [])) if profile else set()
    pages_bundle: list[dict[str, Any]] = []
    
    for p in range(start, end + 1):
        p_item = docling_doc.pages.get(p) if hasattr(docling_doc, "pages") else None
        p_w = 595.32
        p_h = 841.92
        if p_item is not None:
            if hasattr(p_item, "size") and p_item.size is not None:
                p_w = float(getattr(p_item.size, "width", p_w))
                p_h = float(getattr(p_item.size, "height", p_h))
            elif hasattr(p_item, "width") and hasattr(p_item, "height"):
                p_w = float(p_item.width)
                p_h = float(p_item.height)
        
        legacy_txt = legacy_by_page.get(p, "")
        p_warnings: list[str] = []
        
        # Detección de figuras/imágenes no textuales en esta página
        has_figures = False
        for pic in getattr(docling_doc, "pictures", []):
            for prov in getattr(pic, "prov", []) or []:
                if getattr(prov, "page_no", None) == p:
                    has_figures = True
                    break
            if has_figures:
                break
        
        # Bloques de contenido en la página
        blocks: list[dict[str, Any]] = []
        emitted_table_ids: set[str] = set()
        
        for item, _ in docling_doc.iterate_items(page_no=p):
            # Detectar si el item es una figura / imagen
            item_label = getattr(item, "label", None)
            label_str = getattr(item_label, "value", str(item_label)).lower() if item_label else ""
            if "picture" in label_str or "figure" in label_str or "image" in label_str or hasattr(item, "image"):
                has_figures = True
            
            # Si el item es una tabla nativa:
            if hasattr(item, "data") and hasattr(item.data, "table_cells"):
                nat_tbl = effective_native_by_item_id.get(id(item))
                if nat_tbl is None and hasattr(item, "self_ref"):
                    nat_tbl = effective_native_by_ref.get(item.self_ref)
                
                if nat_tbl is not None:
                    tbl_id = nat_tbl["table_id"]
                    t_pages = nat_tbl.get("page_numbers", [])
                    u_pages = sorted(list(set(t_pages)))
                    
                    if not u_pages or len(u_pages) == 1:
                        blocks.append({
                            "type": "table",
                            "text": _format_table_as_text(nat_tbl),
                            "bbox": nat_tbl.get("bbox"),
                        })
                        emitted_table_ids.add(tbl_id)
                    else:
                        p_min = u_pages[0]
                        p_max = u_pages[-1]
                        if p == p_min:
                            label = f"### [Tabla multipágina: páginas {p_min}-{p_max} (procedencia de celdas ambigua)]"
                            t_text = _format_table_as_text(nat_tbl)
                            full_text = f"{label}\n{t_text}" if t_text else label
                            blocks.append({
                                "type": "table",
                                "text": full_text,
                                "bbox": nat_tbl.get("bbox"),
                            })
                            p_warnings.append(
                                f"Tabla multipágina {tbl_id} abarca páginas {p_min}-{p_max} con procedencia de celdas ambigua; preservada una vez en página {p_min}"
                            )
                            emitted_table_ids.add(tbl_id)
                        elif p in u_pages:
                            blocks.append({
                                "type": "table",
                                "text": f"### [Tabla multipágina: páginas {p_min}-{p_max} (procedencia de celdas ambigua) - ver página {p_min}]",
                                "bbox": None,
                            })
                            p_warnings.append(
                                f"Tabla multipágina {tbl_id} (páginas {p_min}-{p_max}) con procedencia de celdas ambigua renderizada en página {p_min}"
                            )
                            emitted_table_ids.add(tbl_id)
                continue
            
            item_text = getattr(item, "text", "")
            if not item_text:
                continue
            
            item_bbox = None
            if hasattr(item, "prov") and item.prov:
                ib = item.prov[0].bbox
                item_bbox = {
                    "l": float(ib.l),
                    "t": float(ib.t),
                    "r": float(ib.r),
                    "b": float(ib.b),
                    "coord_origin": getattr(ib.coord_origin, "value", str(ib.coord_origin)),
                }
            
            # Si el bloque cae dentro de una región de tabla sobreescrita en esta página, suprimirlo
            inside_overridden = False
            for obox in override_boxes_by_page.get(p, []):
                if check_bbox_relationship(item_bbox, obox, p_h) == "contained":
                    inside_overridden = True
                    break
            if inside_overridden:
                continue
            
            if item_bbox is None:
                p_warnings.append("Text block bbox is null")
            
            blocks.append({
                "type": "text",
                "text": str(item_text),
                "bbox": item_bbox,
            })
            
        # Añadir tablas efectivas no emitidas durante iterate_items (ej. tablas revisadas de perfil)
        for tbl in effective_tables:
            tbl_id = tbl.get("table_id")
            if tbl_id in emitted_table_ids:
                continue
            tbl_pages = tbl.get("page_numbers", [])
            u_pages = sorted(list(set(tbl_pages)))
            if not u_pages:
                if p == start:
                    blocks.append({
                        "type": "table",
                        "text": _format_table_as_text(tbl),
                        "bbox": tbl.get("bbox"),
                    })
                    emitted_table_ids.add(tbl_id)
                    p_warnings.append(f"Table {tbl_id} has unknown page provenance; placed on page {start}")
            elif len(u_pages) == 1:
                if u_pages[0] == p:
                    blocks.append({
                        "type": "table",
                        "text": _format_table_as_text(tbl),
                        "bbox": tbl.get("bbox"),
                    })
                    emitted_table_ids.add(tbl_id)
            else:
                p_min = u_pages[0]
                p_max = u_pages[-1]
                if p == p_min:
                    label = f"### [Tabla multipágina: páginas {p_min}-{p_max} (procedencia de celdas ambigua)]"
                    t_text = _format_table_as_text(tbl)
                    full_text = f"{label}\n{t_text}" if t_text else label
                    blocks.append({
                        "type": "table",
                        "text": full_text,
                        "bbox": tbl.get("bbox"),
                    })
                    emitted_table_ids.add(tbl_id)
                    p_warnings.append(
                        f"Tabla multipágina {tbl_id} abarca páginas {p_min}-{p_max} con procedencia de celdas ambigua; preservada una vez en página {p_min}"
                    )
                elif p in u_pages:
                    blocks.append({
                        "type": "table",
                        "text": f"### [Tabla multipágina: páginas {p_min}-{p_max} (procedencia de celdas ambigua) - ver página {p_min}]",
                        "bbox": None,
                    })
                    emitted_table_ids.add(tbl_id)
                    p_warnings.append(
                        f"Tabla multipágina {tbl_id} (páginas {p_min}-{p_max}) con procedencia de celdas ambigua renderizada en página {p_min}"
                    )
        
        # Ordenación física TOPLEFT ÚNICAMENTE en páginas explícitamente declaradas como single_column_pages
        if p in single_col_pages:
            def _block_sort_key(b: dict[str, Any]) -> tuple[float, float]:
                nb = normalize_bbox_to_topleft(b.get("bbox"), p_h)
                if nb is None:
                    return (float("inf"), float("inf"))
                return (nb["t"], nb["l"])
            blocks.sort(key=_block_sort_key)
            
        layout_txt = "\n\n".join(b["text"] for b in blocks if b.get("text")).strip()
        
        has_text_content = bool(legacy_txt.strip() or layout_txt)
        if not has_text_content:
            p_warnings.append("No extractable text found; content flagged for OCR or visual review (unknown content)")
        elif not legacy_txt.strip() and layout_txt:
            p_warnings.append("Missing text layer; content flagged for OCR/review")
            
        if has_figures:
            p_warnings.append("Non-text figure/image content detected; flagged for visual review")
            
        pages_bundle.append({
            "page_number": p,
            "width": p_w,
            "height": p_h,
            "legacy_text": legacy_txt,
            "layout_text": layout_txt,
            "blocks": blocks,
            "warnings": p_warnings,
        })
    
    pkg_versions: dict[str, str] = {}
    for pkg in ("docling", "docling-core", "docling-ibm-models", "pypdfium2", "duckdb", "pypdf"):
        try:
            pkg_versions[pkg] = importlib.metadata.version(pkg)
        except Exception:
            pass
            
    return {
        "schema_version": "document-layout-v1",
        "metadata": {
            "doc_id": doc_id,
            "extraction_id": extraction_id,
            "source_file": pdf_path.name,
            "published_at": valid_date,
            "parser": "docling",
            "package_versions": pkg_versions,
            "settings": extraction_settings,
            "page_range": [start, end],
            "source_sha256": doc_id,
            "profile_sha256": profile_sha256,
        },
        "pages": pages_bundle,
        "tables": effective_tables,
        "native_docling": native_export,
    }


def to_page_records(bundle: dict[str, Any], view: str = "legacy") -> list[PageRecord]:
    """Genera registros PageRecord compatibles con el pipeline a partir del bundle."""
    if view not in ("legacy", "layout"):
        raise ValueError(f"Unknown view: {view!r}. Expected 'legacy' or 'layout'.")
    
    meta = bundle.get("metadata", {})
    doc_id = meta.get("doc_id", "")
    source_file = meta.get("source_file", "")
    published_at = meta.get("published_at", "")
    
    records: list[PageRecord] = []
    for page in bundle.get("pages", []):
        p_num = page["page_number"]
        txt = page.get("legacy_text", "") if view == "legacy" else page.get("layout_text", "")
        records.append(
            PageRecord(
                doc_id=doc_id,
                source_file=source_file,
                published_at=published_at,
                page_number=p_num,
                text=txt,
            )
        )
    return records


def write_bundle(
    bundle: dict[str, Any],
    output_dir: Path,
    *,
    with_duckdb: bool = True,
    text_view: str = "legacy",
) -> dict[str, Any]:
    """Escribe los artefactos de la extracción en una carpeta nueva y exclusiva."""
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory already exists and is not empty: {output_dir}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    meta = bundle.get("metadata", {})
    
    # 1. structured.json (sin el payload masivo de native_docling para ligereza)
    clean_bundle = {
        "schema_version": bundle.get("schema_version", "document-layout-v1"),
        "metadata": meta,
        "pages": bundle.get("pages", []),
        "tables": bundle.get("tables", []),
    }
    struct_path = output_dir / "structured.json"
    struct_path.write_text(json.dumps(clean_bundle, ensure_ascii=False, indent=2), encoding="utf-8")
    
    # 2. native_docling.json
    nat_path = output_dir / "native_docling.json"
    nat_path.write_text(
        json.dumps(bundle.get("native_docling", {}), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    
    # 3. pages.jsonl (utiliza legacy_text por defecto para máxima compatibilidad)
    pages_path = output_dir / "pages.jsonl"
    records = to_page_records(bundle, view=text_view)
    with open(pages_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")
            
    # 4. tables.duckdb
    duck_path = output_dir / "tables.duckdb"
    if with_duckdb:
        try:
            import duckdb
        except ImportError as exc:
            raise ImportError(
                "duckdb is required to write tables.duckdb. Install duckdb or use with_duckdb=False."
            ) from exc
            
        conn = duckdb.connect(str(duck_path))
        try:
            conn.execute(
                """
                CREATE TABLE documents (
                    doc_id VARCHAR,
                    extraction_id VARCHAR PRIMARY KEY,
                    source_file VARCHAR,
                    published_at VARCHAR,
                    page_range_start INTEGER,
                    page_range_end INTEGER,
                    profile_sha256 VARCHAR,
                    source_sha256 VARCHAR,
                    created_at VARCHAR
                );
                CREATE TABLE tables (
                    table_id VARCHAR PRIMARY KEY,
                    doc_id VARCHAR,
                    extraction_id VARCHAR,
                    page_number INTEGER,
                    method VARCHAR,
                    num_rows INTEGER,
                    num_cols INTEGER,
                    caption_text VARCHAR,
                    bbox_json VARCHAR
                );
                CREATE TABLE cells (
                    cell_id VARCHAR PRIMARY KEY,
                    table_id VARCHAR,
                    doc_id VARCHAR,
                    extraction_id VARCHAR,
                    page_number INTEGER,
                    row_start INTEGER,
                    row_end INTEGER,
                    col_start INTEGER,
                    col_end INTEGER,
                    raw_text VARCHAR,
                    text VARCHAR,
                    is_empty BOOLEAN,
                    column_header BOOLEAN,
                    row_header BOOLEAN,
                    header_path VARCHAR,
                    row_label VARCHAR,
                    bbox_json VARCHAR
                );
                """
            )
            p_range = meta.get("page_range", [1, 1])
            ext_id = meta.get("extraction_id", "default_ext")
            conn.execute(
                "INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    meta.get("doc_id"),
                    ext_id,
                    meta.get("source_file"),
                    meta.get("published_at"),
                    p_range[0],
                    p_range[1],
                    meta.get("profile_sha256"),
                    meta.get("source_sha256"),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            for t in bundle.get("tables", []):
                t_page = t["page_numbers"][0] if t.get("page_numbers") else None
                conn.execute(
                    "INSERT INTO tables VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        t["table_id"],
                        meta.get("doc_id"),
                        ext_id,
                        t_page,
                        t["method"],
                        t["dimensions"]["rows"],
                        t["dimensions"]["cols"],
                        t.get("caption_text", ""),
                        json.dumps(t.get("bbox")),
                    ),
                )
                for c_idx, c in enumerate(t.get("cells", [])):
                    c_id = c.get("cell_id", f"{t['table_id']}_r{c['row_start']}c{c['col_start']}_{c_idx}")
                    conn.execute(
                        "INSERT INTO cells VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            c_id,
                            t["table_id"],
                            meta.get("doc_id"),
                            ext_id,
                            c.get("page_number"),
                            c["row_start"],
                            c["row_end"],
                            c["col_start"],
                            c["col_end"],
                            str(c.get("raw_text", "")),
                            str(c.get("text", "")),
                            bool(c.get("is_empty", False)),
                            bool(c.get("column_header", False)),
                            bool(c.get("row_header", False)),
                            json.dumps(c.get("header_path", [])),
                            str(c.get("row_label", "")),
                            json.dumps(c.get("bbox")),
                        ),
                    )
        finally:
            conn.close()
            
    # 5. manifest.json
    written_files = [struct_path, nat_path, pages_path]
    if with_duckdb and duck_path.exists():
        written_files.append(duck_path)
        
    manifest_files: dict[str, dict[str, Any]] = {}
    for p in written_files:
        p_bytes = p.read_bytes()
        manifest_files[p.name] = {
            "sha256": hashlib.sha256(p_bytes).hexdigest(),
            "bytes": len(p_bytes),
        }
        
    manifest = {
        "status": "completed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "files": manifest_files,
        "metadata": meta,
        "options": {
            "with_duckdb": with_duckdb,
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    
    return {
        "output_dir": str(output_dir),
        "manifest": manifest,
        "files": {name: str(output_dir / name) for name in manifest_files},
    }
