"""Extracción y validación de paquetes de evidencia tabular para RAG.

Define la extracción estructurada de filas de tablas con anclaje literal,
coordenadas geométricas, celdas tipadas con Decimal canonical, IDs de evidencia
estables y fragmentos de evidencia acotados en presupuesto UTF-8.

Schema Specification:
--------------------
TablePacket:
    doc_id: str (SHA256 hex del documento)
    source: str (nombre del archivo fuente)
    date: str (fecha de publicación YYYY-MM-DD)
    page: int (número de página 1-based)
    table_id: str (identificador único de la tabla)
    extraction_id: str (identificador de la extracción)
    table_caption: str (título/pie de la tabla)
    row_label: str (etiqueta de encabezado de fila de la cuadrícula)
    cells: list[TableStructuredCell]
    evidence: list[TableEvidenceFragment] (1 a 4 fragmentos literales)

TableStructuredCell:
    cell_id: str (identificador único de celda)
    row_label: str
    header_path: list[str] (orden exacto derivado de la cuadrícula de cabeceras)
    year: int | None (int estricto para nivel, None para variación)
    ordered_delta_years: list[int] | None (ej. [2070, 2050] para variación, None para nivel)
    unit: str ("% PIB" | "puntos PIB")
    measure: str ("level" | "delta")
    value_text: str (texto literal de la celda)
    value_decimal: str (representación canónica Decimal, sin floats)
    bbox: dict[str, float] (l, t, r, b en TOPLEFT)
    row_index: int
    col_index: int
    doc_id: str
    source: str
    date: str
    page: int
    table_id: str
    extraction_id: str
    evidence_ids: list[str] (IDs estables de fragmentos requeridos: fila + cabecera + título)

TableEvidenceFragment:
    evidence_id: str (ID estable y determinista del fragmento)
    doc_id: str
    source: str
    date: str
    page: int
    start_char: int (offset literal en texto de página)
    end_char: int (offset literal en texto de página)
    quote: str (texto literal del fragmento, 12..700 caracteres)
    kind: str ("caption" | "header" | "row")
    bbox: dict[str, float]
"""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
import math
import re
from typing import Any

from rag_pensiones.document_layout import _format_table_as_text

MAX_FRAGMENT_CHARS = 700
MIN_FRAGMENT_CHARS = 12
MAX_EVIDENCE_REFS = 4
MAX_SUM_QUOTE_UTF8_BYTES = 2400

RE_DELTA_HEADER = re.compile(
    r"[Δ\u0394\u2206]\s*(\d{4})\s*[-–—\r\n]+\s*(\d{4})|"
    r"variaci[oó]n\s*(\d{4})\s*[-–—\r\n]+\s*(\d{4})",
    re.IGNORECASE,
)
RE_YEAR_HEADER = re.compile(r"\b(19\d\d|20\d\d)\b")

# Cláusulas explícitas de denominador en porcentajes o puntos (requiere cláusula de unidad explícita, nunca sustantivo aislado)
RE_EXPLICIT_DENOM = re.compile(
    r"(?:"
    r"\(\s*(?:en\s+)?%\s*(?:de\s+(?:la\s+|los\s+|las\s+)?|del?\s+)?([a-záéíóúA-ZÁÉÍÓÚ]+)\s*\)|"
    r"\(\s*(?:en\s+)?(?:puntos|porcentajes?)\s*(?:de\s+(?:la\s+|los\s+|las\s+)?|del?\s+)?([a-záéíóúA-ZÁÉÍÓÚ]+)\s*\)|"
    r"(?<!\w)(?:en\s+)?%\s*(?:de\s+(?:la\s+|los\s+|las\s+)?|del?\s+)?([a-záéíóúA-ZÁÉÍÓÚ]+)\b|"
    r"(?<!\w)(?:en\s+)?(?:puntos|porcentajes?)\s*(?:de\s+(?:la\s+|los\s+|las\s+)?|del?\s+)?([a-záéíóúA-ZÁÉÍÓÚ]+)\b"
    r")",
    re.IGNORECASE,
)

# Cláusulas explícitas monetarias/absolutas (ej. millones de euros, euros)
RE_EXPLICIT_CURRENCY = re.compile(
    r"(?:"
    r"\(\s*(millones\s+de\s+euros|miles\s+de\s+euros|millones\s+de\s+€|miles\s+de\s+€|euros|€)\s*\)|"
    r"(?<!\w)(millones\s+de\s+euros|miles\s+de\s+euros|millones\s+de\s+€|miles\s+de\s+€)\b"
    r")",
    re.IGNORECASE,
)

# Porcentaje genérico sin denominador explícito (ej. (%) o %)
RE_BARE_PERCENT = re.compile(r"(?:\(\s*%\s*\)|(?<!\w)(?:en\s+)?%(?!\w))", re.IGNORECASE)


def parse_canonical_decimal(raw: Any) -> str:
    """Parsea texto de celda con coma española o signo menos Unicode a cadena canónica Decimal.

    Rechaza tipos booleanos, nulos, float binarios, NaN, Infinito y cadenas no numéricas malformadas.
    Acepta str, int o Decimal exacto, y preserva la escala decimal original evitando -0.
    """
    if isinstance(raw, bool) or raw is None:
        raise ValueError(f"Valor no numérico rechazado (bool/None): {raw!r}")
    if isinstance(raw, float):
        raise ValueError(f"Float binario rechazado: {raw!r}")
    if isinstance(raw, Decimal):
        if not raw.is_finite():
            raise ValueError(f"Decimal no finito rechazado: {raw!r}")
        dec = raw
        if dec == 0:
            dec = dec.copy_abs()
        return format(dec, "f")
    elif isinstance(raw, int):
        s = str(raw)
    elif isinstance(raw, str):
        s = raw.strip()
    else:
        raise ValueError(f"Tipo no soportado para parseo decimal: {type(raw).__name__}")

    # Limpiar símbolo de porcentaje terminal y espacios
    s = s.rstrip("%").strip()

    # Normalizar signos menos Unicode a guion ASCII
    s = s.replace("\u2212", "-").replace("\u2013", "-").replace("\u2014", "-")

    # Rechazar literales especiales no finitos
    lower = s.lower()
    if lower in ("nan", "infinity", "-infinity", "+infinity", "inf", "-inf", "+inf", "null", "none"):
        raise ValueError(f"Valor no finito o nulo rechazado: {raw!r}")

    # Manejar separadores decimales españoles
    if "." in s and "," in s:
        s = s.replace(".", "").replace(",", ".")
    elif "," in s:
        s = s.replace(",", ".")

    try:
        dec = Decimal(s)
    except InvalidOperation as err:
        raise ValueError(f"Cadena decimal malformada {raw!r}: {err}") from err

    if not dec.is_finite():
        raise ValueError(f"El valor Decimal {raw!r} no es finito")

    # Normalizar -0 a 0 manteniendo la escala decimal canónica
    if dec == 0:
        dec = dec.copy_abs()
    return format(dec, "f")


def _validate_bbox(bbox: dict[str, Any] | None, page_w: float, page_h: float, label: str) -> dict[str, float]:
    """Valida exhaustivamente un bounding box garantizando coordenadas finitas y contenidas."""
    if not isinstance(bbox, dict):
        raise ValueError(f"{label}: bbox debe ser un diccionario, obtenido {type(bbox).__name__}")
    for key in ("l", "t", "r", "b"):
        if key not in bbox:
            raise ValueError(f"{label}: bbox falta la clave requerida {key!r}")
        val = bbox[key]
        if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(val):
            raise ValueError(f"{label}: coordenada {key} no es un número finito: {val!r}")

    l = float(bbox["l"])
    t = float(bbox["t"])
    r = float(bbox["r"])
    b = float(bbox["b"])

    if l < -5.0 or t < -5.0 or r > page_w + 5.0 or b > page_h + 5.0:
        raise ValueError(f"{label}: bbox fuera de los límites de página [0, {page_w}] x [0, {page_h}]: {bbox}")
    if l > r or t > b:
        raise ValueError(f"{label}: bbox invertido (l > r o t > b): {bbox}")

    return {
        "l": round(l, 3),
        "t": round(t, 3),
        "r": round(r, 3),
        "b": round(b, 3),
        "coord_origin": str(bbox.get("coord_origin", "TOPLEFT")).upper(),
    }


def _validate_cell_bbox_within_table(
    bbox: dict[str, Any] | None,
    t_bbox: dict[str, float],
    page_w: float,
    page_h: float,
    label: str,
) -> dict[str, float]:
    """Valida exhaustivamente un bounding box de celda garantizando contención estricta dentro de su tabla."""
    c_bbox = _validate_bbox(bbox, page_w, page_h, label)

    c_origin = c_bbox.get("coord_origin", "TOPLEFT").upper()
    t_origin = str(t_bbox.get("coord_origin", "TOPLEFT")).upper()

    if c_origin != "TOPLEFT" or t_origin != "TOPLEFT" or c_origin != t_origin:
        raise ValueError(
            f"{label}: origen de coordenadas no soportado o incompatible (celda: {c_origin!r}, tabla: {t_origin!r})"
        )

    t_l = t_bbox["l"]
    t_t = t_bbox["t"]
    t_r = t_bbox["r"]
    t_b = t_bbox["b"]

    c_l = c_bbox["l"]
    c_t = c_bbox["t"]
    c_r = c_bbox["r"]
    c_b = c_bbox["b"]

    if c_l < t_l or c_t < t_t or c_r > t_r or c_b > t_b:
        raise ValueError(
            f"{label}: bbox de celda [{c_l}, {c_t}, {c_r}, {c_b}] no está estrictamente contenido dentro de su tabla [{t_l}, {t_t}, {t_r}, {t_b}]"
        )

    return c_bbox


def _parse_unit_clauses(text: str) -> tuple[str | None, str | None, bool]:
    """Extrae denominadores, monedas y porcentaje genérico de un texto.

    Falla cerrado (ValueError) si detecta cláusulas de unidad mutuamente incompatibles
    dentro del mismo texto (ej. múltiples denominadores diferentes, o mezcla de moneda y porcentaje).
    """
    if not text:
        return None, None, False

    # 1. Denominadores explícitos (% del PIB, % población, puntos PIB, etc.)
    denoms: set[str] = set()
    for m in RE_EXPLICIT_DENOM.finditer(text):
        word = next((g for g in m.groups() if g is not None), "").strip()
        if not word:
            continue
        if word.lower() in ("pib", "gdp"):
            denoms.add("PIB")
        else:
            denoms.add(word.lower())

    # 2. Cláusulas monetarias explícitas (millones de euros, etc.)
    currencies: set[str] = set()
    for m in RE_EXPLICIT_CURRENCY.finditer(text):
        curr_raw = next((g for g in m.groups() if g is not None), "").strip().lower()
        if not curr_raw:
            continue
        if "millones" in curr_raw:
            currencies.add("millones de euros")
        elif "miles" in curr_raw:
            currencies.add("miles de euros")
        elif "euro" in curr_raw or "€" in curr_raw:
            currencies.add("euros")
        else:
            currencies.add(curr_raw)

    # Detección de ambigüedad dentro del mismo texto (falla cerrado)
    if len(denoms) > 1:
        raise ValueError(
            f"Ambigüedad: múltiples denominadores incompatibles en el mismo texto: {sorted(denoms)}"
        )
    if len(currencies) > 1:
        raise ValueError(
            f"Ambigüedad: múltiples unidades monetarias incompatibles en el mismo texto: {sorted(currencies)}"
        )
    if denoms and currencies:
        raise ValueError(
            f"Ambigüedad: cláusulas de unidad conflictivas (porcentaje con denominador {sorted(denoms)} "
            f"vs moneda {sorted(currencies)}) en el mismo texto"
        )

    # 3. Porcentaje genérico sin denominador (ej. (%) o %)
    # Solo relevante si no hay denominador explícito ni moneda
    has_bare_percent = False
    if not denoms and not currencies:
        has_bare_percent = bool(RE_BARE_PERCENT.search(text))

    denom = next(iter(denoms)) if denoms else None
    currency = next(iter(currencies)) if currencies else None

    return denom, currency, has_bare_percent


def _extract_denominator(text: str) -> str | None:
    """Extrae el denominador explícito de un texto (ej. 'PIB', 'población')."""
    denom, _, _ = _parse_unit_clauses(text)
    return denom


def _infer_column_meta(header_path: list[str], table_caption: str) -> tuple[int | None, list[int] | None, str, str]:
    """Infiere año o años ordenados de variación, unidad y tipo de medida a partir de la cabecera."""
    full_header = " ".join(header_path)

    cap_denom, cap_curr, cap_pct = _parse_unit_clauses(table_caption)
    hdr_denom, hdr_curr, hdr_pct = _parse_unit_clauses(full_header)

    # Validar coherencia entre título y cabecera (rechazar conflictos entre título y cabecera)
    if cap_denom and hdr_denom and cap_denom.lower() != hdr_denom.lower():
        raise ValueError(
            f"Conflicto de denominadores entre título ({cap_denom!r}) y cabecera ({hdr_denom!r})"
        )
    if cap_curr and hdr_curr and cap_curr.lower() != hdr_curr.lower():
        raise ValueError(
            f"Conflicto de unidades monetarias entre título ({cap_curr!r}) y cabecera ({hdr_curr!r})"
        )
    if cap_curr and hdr_denom:
        raise ValueError(
            f"Conflicto de unidad entre título ({cap_curr!r}) y cabecera ({hdr_denom!r})"
        )
    if cap_denom and hdr_curr:
        raise ValueError(
            f"Conflicto de unidad entre título ({cap_denom!r}) y cabecera ({hdr_curr!r})"
        )
    if cap_curr and hdr_pct:
        raise ValueError(
            f"Conflicto de unidad entre título ({cap_curr!r}) y cabecera ('%')"
        )
    if cap_pct and hdr_curr:
        raise ValueError(
            f"Conflicto de unidad entre título ('%') y cabecera ({hdr_curr!r})"
        )

    effective_denom = hdr_denom or cap_denom
    effective_curr = hdr_curr or cap_curr
    has_percent = hdr_pct or cap_pct or ("%" in table_caption or "%" in full_header)

    # Detectar variación/delta publicada
    m_delta = RE_DELTA_HEADER.search(full_header)
    if m_delta:
        y1_str = m_delta.group(1) or m_delta.group(3)
        y2_str = m_delta.group(2) or m_delta.group(4)
        y1, y2 = int(y1_str), int(y2_str)
        # Unidad de variación: si el nivel es % PIB, la variación son puntos PIB
        if effective_denom == "PIB":
            unit = "puntos PIB"
        elif effective_denom:
            unit = f"puntos {effective_denom}"
        elif effective_curr:
            unit = effective_curr
        elif has_percent:
            unit = "puntos"
        else:
            unit = "puntos"
        return None, [y1, y2], unit, "delta"

    # Detectar año de nivel
    m_year = RE_YEAR_HEADER.search(full_header)
    if m_year:
        year = int(m_year.group(1))
        if effective_denom == "PIB":
            unit = "% PIB"
        elif effective_denom:
            unit = f"% {effective_denom}"
        elif effective_curr:
            unit = effective_curr
        elif has_percent:
            unit = "%"
        else:
            unit = "nivel"
        return year, None, unit, "level"

    if effective_denom == "PIB":
        unit = "% PIB"
    elif effective_denom:
        unit = f"% {effective_denom}"
    elif effective_curr:
        unit = effective_curr
    elif has_percent:
        unit = "%"
    else:
        unit = "unidad"
    return None, None, unit, "level"


def build_table_packets(bundle: dict[str, Any], pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Construye paquetes literales y estructurados por fila a partir del layout y páginas.

    Valida la identidad del documento, el anclaje literal unívoco en el texto de página,
    la correspondencia geométrica y de cuadrícula, la unicidad de celdas y genera celdas
    tipadas con Decimal canónico e IDs estables de evidencia sin mutar las entradas.
    """
    if not isinstance(bundle, dict):
        raise ValueError("bundle debe ser un diccionario")
    if not isinstance(pages, list) or not pages:
        raise ValueError("pages debe ser una lista no vacía")

    meta = bundle.get("metadata")
    if not isinstance(meta, dict):
        raise ValueError("bundle falta metadata")

    doc_id = meta.get("doc_id")
    source_file = meta.get("source_file")
    published_at = meta.get("published_at")
    extraction_id = meta.get("extraction_id", "")

    if not doc_id or not source_file or not published_at:
        raise ValueError("Metadatos de documento incompletos en bundle")

    # Validar páginas solicitadas y su identidad
    pages_by_number: dict[int, dict[str, Any]] = {}
    for p in pages:
        if not isinstance(p, dict):
            raise ValueError("Cada página debe ser un diccionario")
        p_num = p.get("page_number")
        if isinstance(p_num, bool) or not isinstance(p_num, int) or p_num < 1:
            raise ValueError(f"page_number inválido en página: {p_num!r}")
        if p_num in pages_by_number:
            raise ValueError(f"Página duplicada en pages: {p_num}")
        if p.get("doc_id") != doc_id or p.get("source_file") != source_file or p.get("published_at") != published_at:
            raise ValueError(f"Discrepancia de identidad documental en página {p_num}")
        pages_by_number[p_num] = p

    bundle_pages_by_num: dict[int, dict[str, Any]] = {}
    for bp in bundle.get("pages", []):
        bp_num = bp.get("page_number")
        if isinstance(bp_num, int) and bp_num in pages_by_number:
            bundle_pages_by_num[bp_num] = bp

    tables = bundle.get("tables", [])
    if not isinstance(tables, list):
        raise ValueError("bundle 'tables' debe ser una lista")

    packets: list[dict[str, Any]] = []
    seen_table_ids: set[str] = set()

    for table in tables:
        if not isinstance(table, dict):
            raise ValueError("Cada elemento de tables debe ser un diccionario")

        t_id = table.get("table_id")
        if not t_id or not isinstance(t_id, str):
            raise ValueError(f"table_id inválido o ausente: {t_id!r}")
        if t_id in seen_table_ids:
            raise ValueError(f"table_id duplicado rechazado: {t_id!r}")
        seen_table_ids.add(t_id)

        t_pages = table.get("page_numbers", [])
        if not isinstance(t_pages, list) or not t_pages:
            raise ValueError(f"Tabla {t_id}: page_numbers debe ser una lista no vacía")

        # Exclusión estricta de páginas fuera del subconjunto solicitado (ej. páginas 100/101 excluidas)
        if any(pn not in pages_by_number for pn in t_pages):
            continue

        page_num = t_pages[0]
        page_rec = pages_by_number[page_num]
        b_page = bundle_pages_by_num.get(page_num, {})
        page_w = float(b_page.get("width", 595.32))
        page_h = float(b_page.get("height", 841.92))

        # Validar bbox de tabla
        t_bbox = _validate_bbox(table.get("bbox"), page_w, page_h, f"Tabla {t_id}")

        # Reconstrucción de tabla idéntica al layout_text
        reconstructed_text = _format_table_as_text(table)

        # Buscar el bloque de layout correspondiente en bundle['pages']
        found_block = False
        for blk in b_page.get("blocks", []):
            if blk.get("type") == "table" and blk.get("text") == reconstructed_text:
                found_block = True
                break
        if not found_block:
            raise ValueError(f"Tabla {t_id}: el texto reconstruido difiere del bloque del layout")

        # Anclaje literal unívoco en el texto de la página
        page_text = page_rec.get("text") or page_rec.get("layout_text", "")
        if not page_text:
            raise ValueError(f"Página {page_num} carece de texto literal")

        t_pos = page_text.find(reconstructed_text)
        if t_pos < 0:
            raise ValueError(f"Tabla {t_id}: texto reconstruido no ancla literalmente en página {page_num}")
        if page_text.find(reconstructed_text, t_pos + 1) >= 0:
            raise ValueError(f"Tabla {t_id}: anclaje no unívoco en el texto de página {page_num}")

        # Dimensiones y validación de cuadrícula
        dims = table.get("dimensions", {})
        num_rows = dims.get("rows")
        num_cols = dims.get("cols")
        if not isinstance(num_rows, int) or not isinstance(num_cols, int) or num_rows < 1 or num_cols < 1:
            raise ValueError(f"Tabla {t_id}: dimensiones de cuadrícula inválidas {dims}")

        cells = table.get("cells", [])
        if not isinstance(cells, list) or not cells:
            raise ValueError(f"Tabla {t_id}: celdas ausentes o vacías")

        # Construir matriz física de la cuadrícula a partir de las celdas
        grid_matrix: list[list[str]] = [["" for _ in range(num_cols)] for _ in range(num_rows)]
        cells_by_anchor: dict[tuple[int, int], dict[str, Any]] = {}
        seen_cell_ids: set[str] = set()

        for cell in cells:
            if not isinstance(cell, dict):
                raise ValueError("Celda inválida en tabla")
            c_id = cell.get("cell_id")
            if not c_id or not isinstance(c_id, str):
                raise ValueError(f"Celda falta cell_id: {cell}")
            if c_id in seen_cell_ids:
                raise ValueError(f"cell_id duplicado {c_id!r} en tabla {t_id}")
            seen_cell_ids.add(c_id)

            r_s, r_e = cell.get("row_start"), cell.get("row_end")
            c_s, c_e = cell.get("col_start"), cell.get("col_end")
            if not (
                isinstance(r_s, int)
                and isinstance(r_e, int)
                and isinstance(c_s, int)
                and isinstance(c_e, int)
                and 0 <= r_s < r_e <= num_rows
                and 0 <= c_s < c_e <= num_cols
            ):
                raise ValueError(f"Coordenadas de celda inválidas en {c_id}: {r_s}:{r_e}, {c_s}:{c_e}")

            coord = (r_s, c_s)
            if coord in cells_by_anchor:
                raise ValueError(f"Posición de ancla duplicada en celda {c_id}: {coord}")
            cells_by_anchor[coord] = cell

            # Validar bbox de celda contenido en página/tabla
            _validate_cell_bbox_within_table(cell.get("bbox"), t_bbox, page_w, page_h, f"Celda {c_id}")

            # Poblar la matriz de cuadrícula física
            val = str(cell.get("text", "")).strip()
            grid_matrix[r_s][c_s] = val

        # Validar y derivar encabezados físicos de columna y fila
        # Columna 0 representa los encabezados de fila
        physical_row_headers: dict[int, str] = {r: grid_matrix[r][0] for r in range(num_rows)}
        # Fila 0 representa los encabezados de columna principales
        physical_col_headers: dict[int, list[str]] = {
            c: [grid_matrix[0][c]] if grid_matrix[0][c] else [] for c in range(num_cols)
        }

        # Validar confianza de cuadrícula: metadatos de celda no pueden contradecir la física de la cuadrícula
        for cell in cells:
            r_s, c_s = cell["row_start"], cell["col_start"]
            if r_s > 0 and c_s > 0:
                # Celda de datos
                declared_row_label = str(cell.get("row_label", "")).strip()
                phys_row_label = physical_row_headers[r_s]
                if declared_row_label and phys_row_label and declared_row_label != phys_row_label:
                    raise ValueError(
                        f"Metadatos de row_label {declared_row_label!r} contradicen encabezado físico {phys_row_label!r} en {cell['cell_id']}"
                    )

                declared_hpath = cell.get("header_path")
                if declared_hpath is not None:
                    if not isinstance(declared_hpath, list):
                        raise ValueError(f"header_path debe ser una lista en {cell['cell_id']}")
                    # Si se declara header_path, debe coincidir con los encabezados físicos de la columna en orden
                    phys_hpath = physical_col_headers[c_s]
                    clean_decl = [str(h).strip() for h in declared_hpath if str(h).strip()]
                    clean_phys = [str(h).strip() for h in phys_hpath if str(h).strip()]
                    if clean_decl != clean_phys:
                        raise ValueError(
                            f"header_path {clean_decl} contradice encabezados físicos {clean_phys} en celda {cell['cell_id']}"
                        )

        # Construir líneas exactas de tabla formateada para anclaje literal
        caption_text = table.get("caption_text", "").strip()
        caption_line = f"### {caption_text}" if caption_text else None
        header_line = "| " + " | ".join(grid_matrix[0]) + " |"
        row_lines: dict[int, str] = {
            r: "| " + " | ".join(grid_matrix[r]) + " |" for r in range(1, num_rows)
        }

        # IDs de evidencia estables para caption y header
        caption_ev_id = f"{t_id}_caption"
        header_ev_id = f"{t_id}_header"

        # Construir fragmentos compartidos (caption y header)
        caption_fragment: dict[str, Any] | None = None
        if caption_line:
            c_pos = page_text.find(caption_line, t_pos)
            if c_pos < 0:
                raise ValueError(f"Tabla {t_id}: caption {caption_line!r} no ancla literalmente en página {page_num}")
            c_end = c_pos + len(caption_line)
            c_quote = page_text[c_pos:c_end]
            if len(c_quote) > MAX_FRAGMENT_CHARS:
                raise ValueError(
                    f"Tabla {t_id}: caption supera el límite de {MAX_FRAGMENT_CHARS} caracteres ({len(c_quote)} caracteres)"
                )
            if len(c_quote) < MIN_FRAGMENT_CHARS:
                c_end = min(len(page_text), c_pos + MIN_FRAGMENT_CHARS)
                c_quote = page_text[c_pos:c_end]
                if len(c_quote) < MIN_FRAGMENT_CHARS:
                    raise ValueError(f"Tabla {t_id}: caption demasiado corto ({len(c_quote)} caracteres)")
            caption_fragment = {
                "evidence_id": caption_ev_id,
                "doc_id": doc_id,
                "source": source_file,
                "date": published_at,
                "page": page_num,
                "start_char": c_pos,
                "end_char": c_end,
                "quote": c_quote,
                "kind": "caption",
                "bbox": t_bbox,
            }

        header_fragment: dict[str, Any] | None = None
        h_pos = page_text.find(header_line, t_pos)
        if h_pos < 0:
            raise ValueError(f"Tabla {t_id}: header no ancla literalmente en página {page_num}")
        h_end = h_pos + len(header_line)
        h_quote = page_text[h_pos:h_end]
        if len(h_quote) > MAX_FRAGMENT_CHARS:
            raise ValueError(
                f"Tabla {t_id}: header supera el límite de {MAX_FRAGMENT_CHARS} caracteres ({len(h_quote)} caracteres)"
            )
        if len(h_quote) < MIN_FRAGMENT_CHARS:
            h_end = min(len(page_text), h_pos + MIN_FRAGMENT_CHARS)
            h_quote = page_text[h_pos:h_end]
            if len(h_quote) < MIN_FRAGMENT_CHARS:
                raise ValueError(f"Tabla {t_id}: header demasiado corto ({len(h_quote)} caracteres)")
        header_fragment = {
            "evidence_id": header_ev_id,
            "doc_id": doc_id,
            "source": source_file,
            "date": published_at,
            "page": page_num,
            "start_char": h_pos,
            "end_char": h_end,
            "quote": h_quote,
            "kind": "header",
            "bbox": t_bbox,
        }

        # Generar un paquete por cada fila de datos (filas 1 a num_rows-1)
        for r_idx in range(1, num_rows):
            row_label = physical_row_headers[r_idx]
            row_ev_id = f"{t_id}_r{r_idx}_row"

            # 1. Fragmento literal de la fila actual
            r_line = row_lines[r_idx]
            r_pos = page_text.find(r_line, t_pos)
            if r_pos < 0:
                raise ValueError(f"Fila {r_idx} ({row_label}) no ancla literalmente en texto de página")
            r_end = r_pos + len(r_line)
            r_quote = page_text[r_pos:r_end]
            if len(r_quote) > MAX_FRAGMENT_CHARS:
                raise ValueError(
                    f"Fila {r_idx} ({row_label}): fragmento de fila supera el límite de {MAX_FRAGMENT_CHARS} caracteres ({len(r_quote)} caracteres)"
                )
            if len(r_quote) < MIN_FRAGMENT_CHARS:
                r_end = min(len(page_text), r_pos + MIN_FRAGMENT_CHARS)
                r_quote = page_text[r_pos:r_end]
                if len(r_quote) < MIN_FRAGMENT_CHARS:
                    raise ValueError(f"Fila {r_idx} ({row_label}): fragmento demasiado corto ({len(r_quote)} caracteres)")

            row_cells_list = [
                cells_by_anchor[(r_idx, c)]
                for c in range(1, num_cols)
                if (r_idx, c) in cells_by_anchor
            ]
            row_cell_bboxes = [
                _validate_cell_bbox_within_table(c.get("bbox"), t_bbox, page_w, page_h, f"Celda {c['cell_id']}")
                for c in row_cells_list
                if "bbox" in c
            ]
            if row_cell_bboxes:
                r_bbox = {
                    "l": min(b["l"] for b in row_cell_bboxes),
                    "t": min(b["t"] for b in row_cell_bboxes),
                    "r": max(b["r"] for b in row_cell_bboxes),
                    "b": max(b["b"] for b in row_cell_bboxes),
                    "coord_origin": "TOPLEFT",
                }
            else:
                r_bbox = t_bbox

            row_fragment = {
                "evidence_id": row_ev_id,
                "doc_id": doc_id,
                "source": source_file,
                "date": published_at,
                "page": page_num,
                "start_char": r_pos,
                "end_char": r_end,
                "quote": r_quote,
                "kind": "row",
                "bbox": r_bbox,
            }

            # Construir lista completa de evidencias requeridas para esta fila (caption + header + row)
            evidence_fragments: list[dict[str, Any]] = []
            if caption_fragment:
                evidence_fragments.append(caption_fragment)
            evidence_fragments.append(header_fragment)
            evidence_fragments.append(row_fragment)

            # Validar presupuesto de suma de bytes UTF-8 de citas literales (<= 2400 bytes)
            total_quote_bytes = sum(len(ev["quote"].encode("utf-8")) for ev in evidence_fragments)
            if total_quote_bytes > MAX_SUM_QUOTE_UTF8_BYTES:
                raise ValueError(
                    f"Suma de citas UTF-8 en fila {row_label} supera el límite de {MAX_SUM_QUOTE_UTF8_BYTES} bytes ({total_quote_bytes} bytes)"
                )
            if len(evidence_fragments) > MAX_EVIDENCE_REFS:
                raise ValueError(f"Fila {row_label} supera el límite de {MAX_EVIDENCE_REFS} referencias de evidencia")

            required_evidence_ids = [ev["evidence_id"] for ev in evidence_fragments]

            # Construir celdas numéricas estructuradas de esta fila
            structured_cells: list[dict[str, Any]] = []
            for c_idx in range(1, num_cols):
                if (r_idx, c_idx) not in cells_by_anchor:
                    continue
                cell = cells_by_anchor[(r_idx, c_idx)]
                raw_text = str(cell.get("text", "")).strip()
                if not raw_text or raw_text in ("-", "—", "...", "") or cell.get("is_empty"):
                    continue

                try:
                    canon_dec = parse_canonical_decimal(raw_text)
                except ValueError:
                    # Celda no numérica
                    continue

                h_path = physical_col_headers[c_idx]
                year, deltas, unit, measure = _infer_column_meta(h_path, caption_text)

                structured_cells.append(
                    {
                        "cell_id": cell["cell_id"],
                        "row_label": row_label,
                        "header_path": h_path,
                        "year": year,
                        "ordered_delta_years": deltas,
                        "unit": unit,
                        "measure": measure,
                        "value_text": raw_text,
                        "value_decimal": canon_dec,
                        "bbox": _validate_cell_bbox_within_table(cell.get("bbox"), t_bbox, page_w, page_h, f"Celda {cell['cell_id']}"),
                        "row_index": r_idx,
                        "col_index": c_idx,
                        "doc_id": doc_id,
                        "source": source_file,
                        "date": published_at,
                        "page": page_num,
                        "table_id": t_id,
                        "extraction_id": extraction_id,
                        "evidence_ids": required_evidence_ids,
                    }
                )

            if not structured_cells:
                continue

            packet = {
                "doc_id": doc_id,
                "source": source_file,
                "date": published_at,
                "page": page_num,
                "table_id": t_id,
                "extraction_id": extraction_id,
                "table_caption": caption_text,
                "row_label": row_label,
                "cells": structured_cells,
                "evidence": evidence_fragments,
            }
            packets.append(packet)

    return packets


# Acceso dinámico de atributos para evitar importaciones circulares en inicialización de módulos
def __getattr__(name: str) -> Any:
    if name in ("guard_table_facts", "parse_table_request", "render_table_facts"):
        from rag_pensiones import numeric_guard
        return getattr(numeric_guard, name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
