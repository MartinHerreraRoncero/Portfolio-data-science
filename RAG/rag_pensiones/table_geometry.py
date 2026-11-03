"""Validación geométrica y extracción de tablas revisadas por coordenadas PDF."""

from collections.abc import Callable
import hashlib
import math
from pathlib import Path
from typing import Any


def validate_regions(profile: dict[str, Any], doc_id: str, page_sizes: dict[int, tuple[float, float]]) -> None:
    """Valida exhaustivamente un perfil geométrico de tablas contra un documento."""
    if not isinstance(profile, dict):
        raise ValueError("Profile must be a dictionary.")
    
    prof_doc_id = profile.get("doc_id")
    if not prof_doc_id or prof_doc_id != doc_id:
        raise ValueError(f"Profile doc_id {prof_doc_id!r} does not match document doc_id {doc_id!r}")
    
    tables = profile.get("tables")
    if not isinstance(tables, list):
        raise ValueError("Profile 'tables' must be a list.")
    
    seen_profile_ids: set[str] = set()
    for idx, table in enumerate(tables):
        if not isinstance(table, dict):
            raise ValueError(f"Table at index {idx} must be a dictionary.")
        
        if "profile_id" not in table or table.get("profile_id") is None:
            raise ValueError(f"Table {idx}: missing required non-empty string 'profile_id'")
        prof_id = table["profile_id"]
        if not isinstance(prof_id, str) or not prof_id.strip():
            raise ValueError(f"Table {idx}: profile_id must be a non-empty string, got {prof_id!r}")
        if prof_id in seen_profile_ids:
            raise ValueError(f"Duplicate profile table ID rejected: {prof_id!r}")
        seen_profile_ids.add(prof_id)
        
        page_num = table.get("page_number")
        if isinstance(page_num, bool) or not isinstance(page_num, int) or page_num < 1:
            raise ValueError(f"Table {idx}: page_number must be strict int >= 1 (not bool), got {page_num!r}")
        
        t_page_size = table.get("page_size")
        if not isinstance(t_page_size, (list, tuple)) or len(t_page_size) != 2:
            raise ValueError(f"Table {idx}: page_size must be a 2-element sequence, got {t_page_size!r}")
        
        page_w, page_h = float(t_page_size[0]), float(t_page_size[1])
        if not (math.isfinite(page_w) and math.isfinite(page_h) and page_w > 0 and page_h > 0):
            raise ValueError(f"Table {idx}: page_size must contain positive finite numbers: {t_page_size}")
        
        if page_sizes:
            if page_num not in page_sizes:
                raise ValueError(
                    f"Table {idx}: profile page {page_num} does not exist in document pages (available: {sorted(page_sizes.keys())})"
                )
            actual_w, actual_h = page_sizes[page_num]
            if abs(page_w - actual_w) > 1.5 or abs(page_h - actual_h) > 1.5:
                raise ValueError(
                    f"Table {idx}: page size mismatch for page {page_num}: profile {t_page_size} vs document {(actual_w, actual_h)}"
                )
        
        row_edges = table.get("row_edges")
        if not isinstance(row_edges, list) or len(row_edges) < 2:
            raise ValueError(f"Table {idx}: row_edges must contain at least 2 edges, got {row_edges!r}")
        
        for r_i, y in enumerate(row_edges):
            if isinstance(y, bool) or not isinstance(y, (int, float)) or not math.isfinite(y):
                raise ValueError(f"Table {idx}: row edge at index {r_i} is not a finite number: {y!r}")
            if y < 0 or y > page_h:
                raise ValueError(f"Table {idx}: row edge {y} is outside page bounds [0, {page_h}]")
            if r_i > 0 and y <= row_edges[r_i - 1]:
                raise ValueError(f"Table {idx}: row_edges must be strictly increasing: {row_edges[r_i - 1]} >= {y}")
        
        col_edges = table.get("column_edges")
        if not isinstance(col_edges, list) or len(col_edges) < 2:
            raise ValueError(f"Table {idx}: column_edges must contain at least 2 edges, got {col_edges!r}")
        
        for c_j, x in enumerate(col_edges):
            if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):
                raise ValueError(f"Table {idx}: column edge at index {c_j} is not a finite number: {x!r}")
            if x < 0 or x > page_w:
                raise ValueError(f"Table {idx}: column edge {x} is outside page bounds [0, {page_w}]")
            if c_j > 0 and x <= col_edges[c_j - 1]:
                raise ValueError(f"Table {idx}: column_edges must be strictly increasing: {col_edges[c_j - 1]} >= {x}")
        
        num_rows = len(row_edges) - 1
        num_cols = len(col_edges) - 1
        
        # Validar header_rows
        if "header_rows" in table:
            header_rows = table["header_rows"]
            if isinstance(header_rows, bool) or not isinstance(header_rows, int):
                raise ValueError(f"Table {idx}: header_rows must be a strict integer (not bool), got {header_rows!r}")
            if not (0 <= header_rows <= num_rows):
                raise ValueError(f"Table {idx}: header_rows {header_rows} out of range [0, {num_rows}]")
        
        # Validar row_header_columns
        if "row_header_columns" in table:
            row_header_columns = table["row_header_columns"]
            if not isinstance(row_header_columns, list):
                raise ValueError(f"Table {idx}: row_header_columns must be a list, got {row_header_columns!r}")
            seen_rh_cols: set[int] = set()
            for rh_idx in row_header_columns:
                if isinstance(rh_idx, bool) or not isinstance(rh_idx, int):
                    raise ValueError(f"Table {idx}: row_header_columns items must be strict int (not bool), got {rh_idx!r}")
                if not (0 <= rh_idx < num_cols):
                    raise ValueError(f"Table {idx}: row_header_column {rh_idx} out of range [0, {num_cols})")
                if rh_idx in seen_rh_cols:
                    raise ValueError(f"Table {idx}: duplicate row_header_column index: {rh_idx}")
                seen_rh_cols.add(rh_idx)
        
        spans = table.get("spans", [])
        if not isinstance(spans, list):
            raise ValueError(f"Table {idx}: spans must be a list.")
        
        covered_cells: set[tuple[int, int]] = set()
        for s_idx, span in enumerate(spans):
            if not isinstance(span, dict):
                raise ValueError(f"Table {idx}: span {s_idx} must be a dict.")
            r_s, r_e = span.get("row_start"), span.get("row_end")
            c_s, c_e = span.get("col_start"), span.get("col_end")
            for name, val in [("row_start", r_s), ("row_end", r_e), ("col_start", c_s), ("col_end", c_e)]:
                if isinstance(val, bool) or not isinstance(val, int):
                    raise ValueError(f"Table {idx}, span {s_idx}: {name} must be an integer, got {val!r}")
            if not (0 <= r_s < r_e <= num_rows):
                raise ValueError(f"Table {idx}, span {s_idx}: rows [{r_s}, {r_e}) out of bounds [0, {num_rows}]")
            if not (0 <= c_s < c_e <= num_cols):
                raise ValueError(f"Table {idx}, span {s_idx}: cols [{c_s}, {c_e}) out of bounds [0, {num_cols}]")
            
            for r in range(r_s, r_e):
                for c in range(c_s, c_e):
                    if (r, c) in covered_cells:
                        raise ValueError(f"Table {idx}: overlapping spans detected covering cell ({r}, {c})")
                    covered_cells.add((r, c))
        
        caption_bbox = table.get("caption_bbox")
        if caption_bbox is not None:
            if not isinstance(caption_bbox, (list, tuple)) or len(caption_bbox) != 4:
                raise ValueError(f"Table {idx}: caption_bbox must be 4 coordinates, got {caption_bbox!r}")
            x0, y0, x1, y1 = caption_bbox
            if not all(isinstance(v, (int, float)) and math.isfinite(v) and not isinstance(v, bool) for v in (x0, y0, x1, y1)):
                raise ValueError(f"Table {idx}: non-finite coordinates in caption_bbox: {caption_bbox}")
            if not (0 <= x0 < x1 <= page_w and 0 <= y0 < y1 <= page_h):
                raise ValueError(f"Table {idx}: caption_bbox out of page bounds: {caption_bbox}")
        
        footnote_bboxes = table.get("footnote_bboxes", [])
        if not isinstance(footnote_bboxes, list):
            raise ValueError(f"Table {idx}: footnote_bboxes must be a list.")
        for f_idx, fbox in enumerate(footnote_bboxes):
            if not isinstance(fbox, (list, tuple)) or len(fbox) != 4:
                raise ValueError(f"Table {idx}: footnote_bbox {f_idx} must be 4 coordinates, got {fbox!r}")
            x0, y0, x1, y1 = fbox
            if not all(isinstance(v, (int, float)) and math.isfinite(v) and not isinstance(v, bool) for v in (x0, y0, x1, y1)):
                raise ValueError(f"Table {idx}: non-finite coordinates in footnote_bbox {f_idx}: {fbox}")
            if not (0 <= x0 < x1 <= page_w and 0 <= y0 < y1 <= page_h):
                raise ValueError(f"Table {idx}: footnote_bbox {f_idx} out of page bounds: {fbox}")
    
    if "single_column_pages" in profile:
        scp = profile["single_column_pages"]
        if not isinstance(scp, list):
            raise ValueError("Profile 'single_column_pages' must be a list.")
        for p_idx, sc_p in enumerate(scp):
            if isinstance(sc_p, bool) or not isinstance(sc_p, int) or sc_p < 1:
                raise ValueError(f"Profile single_column_pages[{p_idx}] must be strict int >= 1 (not bool), got {sc_p!r}")
            if page_sizes and sc_p not in page_sizes:
                raise ValueError(
                    f"Profile single_column_page {sc_p} does not exist in document pages (available: {sorted(page_sizes.keys())})"
                )


def extract_region_cells(region: dict[str, Any], crop_text: Callable[[float, float, float, float], str]) -> list[dict[str, Any]]:
    """Extrae celdas estructuradas a partir de una cuadrícula geométrica y callback crop_text."""
    row_edges = region["row_edges"]
    col_edges = region["column_edges"]
    num_rows = len(row_edges) - 1
    num_cols = len(col_edges) - 1
    header_rows = region.get("header_rows", 0)
    row_header_cols = region.get("row_header_columns", [])
    spans = region.get("spans", [])
    page_num = region.get("page_number")
    
    spanned_cells: dict[tuple[int, int], dict[str, int]] = {}
    for span in spans:
        r_s, r_e = span["row_start"], span["row_end"]
        c_s, c_e = span["col_start"], span["col_end"]
        for r in range(r_s, r_e):
            for c in range(c_s, c_e):
                spanned_cells[(r, c)] = span
    
    grid: list[list[dict[str, Any] | None]] = [[None for _ in range(num_cols)] for _ in range(num_rows)]
    cells: list[dict[str, Any]] = []
    
    for r in range(num_rows):
        for c in range(num_cols):
            if (r, c) in spanned_cells:
                span = spanned_cells[(r, c)]
                r_s, r_e = span["row_start"], span["row_end"]
                c_s, c_e = span["col_start"], span["col_end"]
                if (r, c) != (r_s, c_s):
                    continue
            else:
                r_s, r_e, c_s, c_e = r, r + 1, c, c + 1
            
            x0 = float(col_edges[c_s])
            x1 = float(col_edges[c_e])
            y0 = float(row_edges[r_s])
            y1 = float(row_edges[r_e])
            
            raw_text = crop_text(x0, y0, x1, y1)
            if raw_text is None:
                raw_text = ""
            text = raw_text.strip()
            is_empty = (len(text) == 0)
            
            is_col_header = (r_s < header_rows)
            is_row_header = any(col_idx in row_header_cols for col_idx in range(c_s, c_e))
            
            cell: dict[str, Any] = {
                "row_start": r_s,
                "row_end": r_e,
                "col_start": c_s,
                "col_end": c_e,
                "raw_text": raw_text,
                "text": text,
                "is_empty": is_empty,
                "page_number": page_num,
                "bbox": {
                    "l": x0,
                    "t": y0,
                    "r": x1,
                    "b": y1,
                    "coord_origin": "TOPLEFT",
                },
                "column_header": is_col_header,
                "row_header": is_row_header,
                "header_path": [],
                "row_label": "",
            }
            cells.append(cell)
            for r_sub in range(r_s, r_e):
                for c_sub in range(c_s, c_e):
                    grid[r_sub][c_sub] = cell
    
    # Derivación determinista de header_path y row_label únicamente a partir de celdas existentes
    for cell in cells:
        c_idx = cell["col_start"]
        r_idx = cell["row_start"]
        
        # Header path: concatenación de textos de cabecera de columna superiores
        if not cell["column_header"] and header_rows > 0:
            path: list[str] = []
            for hr in range(header_rows):
                h_cell = grid[hr][c_idx]
                if h_cell and h_cell["text"] and (not path or path[-1] != h_cell["text"]):
                    path.append(h_cell["text"])
            cell["header_path"] = path
        
        # Row label: texto de cabecera de fila a la izquierda en la misma fila
        if not cell["row_header"] and row_header_cols:
            labels: list[str] = []
            for rh_col in row_header_cols:
                rh_cell = grid[r_idx][rh_col]
                if rh_cell and rh_cell["text"] and rh_cell != cell:
                    labels.append(rh_cell["text"])
            cell["row_label"] = " ".join(labels)
            
    return cells


def extract_reviewed_tables(
    pdf_path: Path,
    profile: dict[str, Any],
    start_page: int,
    end_page: int,
    extraction_id: str | None = None,
) -> list[dict[str, Any]]:
    """Extrae tablas revisadas de un PDF usando PDFium y coordenadas TOPLEFT."""
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    try:
        import pypdfium2 as pdfium
    except ImportError as exc:
        raise ImportError(
            "pypdfium2 is required for reviewed table extraction. Install requirements-layout.txt."
        ) from exc
    
    doc = pdfium.PdfDocument(pdf_path)
    try:
        total_pages = len(doc)
        page_sizes: dict[int, tuple[float, float]] = {}
        for p in range(1, total_pages + 1):
            page = doc.get_page(p - 1)
            w, h = page.get_size()
            page_sizes[p] = (float(w), float(h))
            page.close()
        
        doc_id = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
        validate_regions(profile, doc_id, page_sizes)
        
        if extraction_id is None:
            import json
            prof_bytes = json.dumps(profile, sort_keys=True).encode("utf-8")
            prof_sha = hashlib.sha256(prof_bytes).hexdigest()
            payload = {
                "doc_id": doc_id,
                "page_range": [start_page, end_page],
                "profile_sha": prof_sha,
            }
            canon = json.dumps(payload, sort_keys=True, separators=(",", ":"))
            extraction_id = hashlib.sha256(canon.encode("utf-8")).hexdigest()[:16]
        
        results: list[dict[str, Any]] = []
        for table_prof in profile.get("tables", []):
            p_num = table_prof["page_number"]
            if not (start_page <= p_num <= end_page):
                continue
            
            page = doc.get_page(p_num - 1)
            try:
                page_w, page_h = page.get_size()
                textpage = page.get_textpage()
                try:
                    def crop_text(l: float, t: float, r: float, b: float) -> str:
                        pdf_left = min(float(l), float(r))
                        pdf_right = max(float(l), float(r))
                        pdf_bottom = page_h - max(float(t), float(b))
                        pdf_top = page_h - min(float(t), float(b))
                        return textpage.get_text_bounded(
                            left=pdf_left,
                            bottom=pdf_bottom,
                            right=pdf_right,
                            top=pdf_top,
                        )
                    
                    caption_text = ""
                    if table_prof.get("caption_bbox"):
                        caption_text = crop_text(*table_prof["caption_bbox"]).strip()
                    
                    footnotes: list[str] = []
                    for fbox in table_prof.get("footnote_bboxes", []):
                        fn_text = crop_text(*fbox).strip()
                        if fn_text:
                            footnotes.append(fn_text)
                    
                    cells = extract_region_cells(table_prof, crop_text)
                    row_edges = table_prof["row_edges"]
                    col_edges = table_prof["column_edges"]
                    profile_id = table_prof["profile_id"]
                    table_id = f"table_{extraction_id}_p{p_num}_{profile_id}"
                    for c_idx, c in enumerate(cells):
                        c["cell_id"] = f"{table_id}_r{c['row_start']}c{c['col_start']}_{c_idx}"
                    
                    table_dict: dict[str, Any] = {
                        "table_id": table_id,
                        "method": "reviewed_profile",
                        "native_ref": None,
                        "page_numbers": [p_num],
                        "bbox": {
                            "l": float(col_edges[0]),
                            "t": float(row_edges[0]),
                            "r": float(col_edges[-1]),
                            "b": float(row_edges[-1]),
                            "coord_origin": "TOPLEFT",
                        },
                        "dimensions": {
                            "rows": len(row_edges) - 1,
                            "cols": len(col_edges) - 1,
                        },
                        "cells": cells,
                        "caption_text": caption_text,
                        "footnotes": footnotes,
                        "warnings": [],
                    }
                    results.append(table_dict)
                finally:
                    textpage.close()
            finally:
                page.close()
        return results
    finally:
        doc.close()
