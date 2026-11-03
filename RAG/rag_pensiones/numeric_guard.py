"""Guardia numérica y parseo de peticiones de tablas para RAG.

Implementa el parseo determinista de consultas a celdas de tabla exactas,
la guardia estricta de hechos numéricos contra celdas autoritativas validadas
de la cuadrícula física y la generación canónica de afirmaciones fundamentadas.

Schema Specification:
--------------------
Fact Schema (for verification adapter):
    cell_id: str             # ID único de la celda de la tabla (ej. "table_..._r1c1")
    concept: str             # Etiqueta de concepto / fila (ej. "Empleo")
    header_path: list[str]   # Ruta de encabezados en orden exacto (ej. ["2025"])
    year: int | None         # Año exacto (int estricto, no bool) para nivel, None para delta
    ordered_delta_years: list[int] | None  # [año_fin, año_inicio] para delta, None para nivel
    unit: str                # Unidad exacta ("% PIB" para nivel, "puntos PIB" para delta)
    measure: str             # "level" | "delta"
    value: str | Decimal     # Valor numérico como string Decimal canónico (sin floats)
    doc_id: str              # SHA256 hex del documento fuente
    source: str              # Nombre del archivo fuente
    date: str                # Fecha de publicación YYYY-MM-DD
    page: int                # Número de página (int >= 1, no bool)
    table_id: str            # ID de la tabla
    extraction_id: str       # ID de extracción
    evidence_ids: list[str]  # Lista completa de IDs de evidencia del paquete [caption, header, row]

TableRequest (Supported):
    status: "supported"
    table_id: str
    table_caption: str
    target_cells: list[TargetCellDescriptor]

TargetCellDescriptor:
    cell_id: str
    concept: str
    row_label: str
    header_path: list[str]
    year: int | None
    ordered_delta_years: list[int] | None
    unit: str
    measure: str
    expected_value: str (representación canónica Decimal, solo para validación interna)
    doc_id: str
    source: str
    page: int
    table_id: str
    extraction_id: str

TableRequest (Unsupported):
    status: "unsupported"
    reason: str

GuardResult:
    decision: "accepted" | "rejected"
    reasons: list[str]
    accepted_facts: list[dict]

RenderedClaim:
    claim_id: str
    text: str
    concept: str
    year: int | None
    ordered_delta_years: list[int] | None
    unit: str
    measure: str
    value: str (Decimal canonical string)
    cell_id: str
    evidence_ids: list[str]
    status: "answered"
"""

from __future__ import annotations

import copy
from decimal import Decimal, InvalidOperation
import re
from typing import Any
import unicodedata

from rag_pensiones.table_evidence import build_table_packets, parse_canonical_decimal

__all__ = [
    "build_table_packets",
    "guard_table_facts",
    "parse_canonical_decimal",
    "parse_table_request",
    "render_table_facts",
]

# Detección de referencia explícita con token de tabla exacto (no substrings ambiguos como tabla 1 vs 10)
RE_EXPLICIT_TABLE = re.compile(
    r"(?<!\w)(?:seg[uú]n\s+(?:el|la)|en\s+(?:el|la)|el|la)?\s*(?:cuadro|tabla)\s*(\d+|[a-zA-Z0-9_\-]+)(?!\w)",
    re.IGNORECASE,
)

# Preguntas causales o explicativas no soportadas ('por qué')
RE_CAUSAL_QUESTION = re.compile(r"(?<!\w)por\s+qu[eé](?!\w)", re.IGNORECASE)

# Instrucciones de exclusión o negación del usuario (fuera de etiquetas de fila confiables)
RE_USER_NEGATION = re.compile(
    r"(?<!\w)(?:no\s+uses?|no\s+utilices?|no\s+incluy\w*|no\s+tomes?|no\s+apliques?|no\s+quiero|sin\b(?!\s*prtr)|excepto|excluy\w*|salvo|no)(?!\w)",
    re.IGNORECASE,
)

# Parejas de años para variación (ordenadas de forma explícita)
RE_DELTA_PAIR = re.compile(
    r"(?<!\w)(\d{4})\s*(?:menos|[-–—])\s*(\d{4})(?!\w)|(?<!\w)(?:entre|de)\s+(\d{4})\s+(?:y|a|menos|[-–—])\s+(\d{4})(?!\w)",
    re.IGNORECASE,
)

RE_SINGLE_YEAR = re.compile(r"(?<!\w)(19\d\d|20\d\d)(?!\w)")

# Directivas permitidas sobre columnas publicadas
RE_DIRECTIVE = re.compile(
    r"(?<!\w)(?:usa|usando|utiliza|utilizando)\s+(?:la\s+)?(?:columna\s+(?:de\s+)?)?(?:variaci[oó]n\s+publicada|variaciones\s+publicadas|variaci[oó]n|variaciones)(?!\w)|"
    r"(?<!\w)(?:usa|usando|utiliza|utilizando)\s+(?:la\s+)?(?:columna|fila)(?!\w)",
    re.IGNORECASE,
)

# Unidades y medidas permitidas (delta vs level)
RE_MEASURES_UNITS = re.compile(
    r"(?<!\w)(?:"
    r"variaci[oó]n\s+publicada|variaciones\s+publicadas|variaci[oó]n|variaciones|diferencias?|delta|[Δ\u0394\u2206]|"
    r"puntos\s+del?\s+pib|puntos\s+pib|puntos|"
    r"porcentajes?\s+del?\s+pib|porcentajes?\s+de\s+pib|porcentajes?\s+pib|porcentajes?|"
    r"[%％]\s*del?\s*pib|[%％]|"
    r"millones\s+de\s+euros?|miles\s+de\s+euros?|millones\s+de\s+€|miles\s+de\s+€|euros?|€|"
    r"pib|gdp"
    r")(?!\w)",
    re.IGNORECASE,
)

# Sintaxis gramatical permitida de formulación de preguntas tabulares (tokens atómicos sin engullir medidas/etiquetas)
RE_STRUCTURAL_TOKENS = re.compile(
    r"(?<!\w)(?:"
    # Interrogativos específicos de lectura numérica (sin incluir wh-words explicativas o conteos)
    r"¿?\s*qu[eé]|"
    r"¿?\s*cu[aá]les?\s+son|"
    r"¿?\s*cu[aá]l\s+es|"
    r"¿?\s*cu[aá]les?|"
    r"¿?\s*cu[aá]l|"
    r"¿?\s*cu[aá]nto\s+es|"
    r"¿?\s*cu[aá]nto|"
    # Predicados y verbos de cuadrícula
    r"figuran?\s+para|"
    r"figuran?|"
    r"corresponden?\s+a|"
    r"corresponden?|"
    r"aparecen?\s+para|"
    r"aparecen?|"
    r"muestran?|"
    r"hay\s+para|"
    r"hay|"
    r"se\s+publican?|"
    r"publicad[ao]s?|"
    r"respecto\s+a|"
    # Conectores, preposiciones y artículos
    r"respectivamente|"
    r"para|"
    r"en|"
    r"de|"
    r"del|"
    r"a|"
    r"al|"
    r"y|"
    r"o|"
    r"los|"
    r"las|"
    r"el|"
    r"la|"
    # Sustantivos estructurales
    r"dato[s]?|"
    r"cifra[s]?|"
    r"valor[es]?|"
    r"nivel[es]?|"
    r"tasa[s]?|"
    r"importe[s]?|"
    r"amb[ao]s|"
    r"columnas?|"
    r"filas?|"
    r"celdas?|"
    r"cuadro[s]?|"
    r"tabla[s]?"
    r")(?!\w)",
    re.IGNORECASE,
)

ALLOWED_FACT_KEYS = {
    "cell_id",
    "concept",
    "row_label",
    "header_path",
    "year",
    "ordered_delta_years",
    "unit",
    "measure",
    "value",
    "value_decimal",
    "doc_id",
    "source",
    "date",
    "page",
    "table_id",
    "extraction_id",
    "evidence_ids",
}


class _SpanTracker:
    def __init__(self, text: str):
        self.text = text
        self.length = len(text)
        self.consumed = [False] * self.length

    def mark(self, start: int, end: int) -> None:
        for i in range(max(0, start), min(self.length, end)):
            self.consumed[i] = True

    def has_overlap(self, start: int, end: int) -> bool:
        return any(self.consumed[max(0, start) : min(self.length, end)])

    def residual_unconsumed(self) -> list[tuple[int, int, str]]:
        res: list[tuple[int, int, str]] = []
        in_un = False
        s = 0
        for i, c in enumerate(self.consumed):
            if not c and not in_un:
                in_un = True
                s = i
            elif c and in_un:
                in_un = False
                res.append((s, i, self.text[s:i]))
        if in_un:
            res.append((s, self.length, self.text[s:]))
        return res


def parse_table_request(query: str, packets: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Parsea una consulta hacia celdas exactas de tabla o retorna None si no hay mención explícita.

    Retorna None únicamente si no hay referencia explícita a cuadro/tabla.
    Si la referencia es explícita pero no soportada, ambigua o contiene restricciones residuales
    no cubiertas por la gramática de la cuadrícula, retorna status='unsupported' con su razón
    explicada, nunca fallback narrativo.
    """
    if not isinstance(query, str):
        return None
    if not isinstance(packets, list):
        raise ValueError("packets debe ser una lista de paquetes de tabla")

    # Normalización canónica Unicode NFC para preservar límites exactos de caracteres
    query = unicodedata.normalize("NFC", query)

    # 1. Detección de referencia explícita a cuadro o tabla
    m_table = RE_EXPLICIT_TABLE.search(query)
    if not m_table:
        return None

    table_token = m_table.group(1).lower()

    # 2. Búsqueda dinámica con coincidencia exacta de token (ej. 'tabla 1' no puede casar con 'tabla 10')
    matched_packets: list[dict[str, Any]] = []
    seen_table_ids: set[str] = set()

    for p in packets:
        t_id = str(p.get("table_id", "")).lower()
        t_cap = str(p.get("table_caption", "")).lower()
        pattern_cap = r"\b(?:cuadro|tabla)\s*" + re.escape(table_token) + r"\b"
        pattern_id = r"(?:cuadro|tabla|_)" + re.escape(table_token) + r"(?:_|$|\b)"
        if re.search(pattern_cap, t_cap) or re.search(pattern_id, t_id):
            matched_packets.append(p)
            seen_table_ids.add(p.get("table_id"))

    if not matched_packets:
        return {
            "status": "unsupported",
            "reason": f"Tabla o cuadro {table_token!r} no encontrado en los paquetes de evidencia disponibles.",
        }

    if len(seen_table_ids) > 1:
        return {
            "status": "unsupported",
            "reason": f"Referencia ambigua: {table_token!r} coincide con múltiples tablas distintas: {sorted(seen_table_ids)}",
        }

    # 3. Extracción de conceptos (filas) disponibles en la tabla
    def _norm_label(text: str) -> str:
        nfkd = unicodedata.normalize("NFKD", text.strip())
        no_accents = "".join(c for c in nfkd if not unicodedata.combining(c))
        return " ".join(no_accents.lower().split())

    packets_by_norm_label: dict[str, list[dict[str, Any]]] = {}
    row_packets_by_label: dict[str, dict[str, Any]] = {}
    for p in matched_packets:
        rl = p.get("row_label", "").strip()
        if rl:
            n_rl = _norm_label(rl)
            packets_by_norm_label.setdefault(n_rl, []).append(p)
            row_packets_by_label[rl] = p

    if not packets_by_norm_label:
        return {
            "status": "unsupported",
            "reason": "La tabla identificada no contiene filas de datos estructurados.",
        }

    tracker = _SpanTracker(query)

    # Paso A: Consumir la mención explícita al cuadro/tabla
    tracker.mark(m_table.start(), m_table.end())

    # Paso B: Coincidencia voraz de etiquetas de fila (longest non-overlapping literal matches)
    # Ordenar por longitud descendente para que 'No modelizados' consuma su span antes que 'Modelizados'
    sorted_labels = sorted(row_packets_by_label.keys(), key=lambda x: len(x), reverse=True)
    matched_label_occurrences: list[tuple[int, int, str]] = []

    for lbl in sorted_labels:
        lbl_clean = lbl.strip()
        if not lbl_clean:
            continue
        pattern = r"(?<!\w)" + re.escape(lbl_clean) + r"(?!\w)"
        for m in re.finditer(pattern, query, flags=re.IGNORECASE):
            s, e = m.span()
            if not tracker.has_overlap(s, e):
                tracker.mark(s, e)
                matched_label_occurrences.append((s, e, lbl))

    matched_label_occurrences.sort(key=lambda x: x[0])
    matched_labels: list[str] = []
    for _, _, lbl in matched_label_occurrences:
        if lbl not in matched_labels:
            matched_labels.append(lbl)

    if not matched_labels:
        return {
            "status": "unsupported",
            "reason": "No se reconoció ninguna fila o concepto válido de la tabla en la consulta.",
        }

    # Verificar que cada concepto coincidente resuelva a una fila autoritativa única (sin duplicados normalizados)
    for lbl in matched_labels:
        n_lbl = _norm_label(lbl)
        pkts = packets_by_norm_label.get(n_lbl, [])
        if len(pkts) > 1:
            return {
                "status": "unsupported",
                "reason": f"Ambigüedad en etiqueta de fila: {lbl!r} coincide con múltiples filas en la tabla.",
            }

    # Paso C: Detección estricta de preguntas causales y negaciones/exclusiones de usuario
    if RE_CAUSAL_QUESTION.search(query):
        return {
            "status": "unsupported",
            "reason": "La consulta contiene preguntas causales o explicativas no soportadas ('por qué').",
        }

    for m in RE_USER_NEGATION.finditer(query):
        s, e = m.span()
        if not tracker.has_overlap(s, e):
            return {
                "status": "unsupported",
                "reason": f"La consulta contiene restricciones de exclusión o negación no soportadas ({m.group(0)!r}).",
            }

    # Paso D: Consumir pares de variación temporal (múltiples deltas soportadas)
    detected_deltas: list[tuple[int, int, list[int]]] = []
    for m in RE_DELTA_PAIR.finditer(query):
        s, e = m.span()
        if not tracker.has_overlap(s, e):
            y1_s = m.group(1) or m.group(3)
            y2_s = m.group(2) or m.group(4)
            tracker.mark(s, e)
            detected_deltas.append((s, e, [int(y1_s), int(y2_s)]))

    # Paso E: Consumir años individuales (niveles)
    detected_single_years: list[tuple[int, int, int]] = []
    for m in RE_SINGLE_YEAR.finditer(query):
        s, e = m.span()
        if not tracker.has_overlap(s, e):
            tracker.mark(s, e)
            detected_single_years.append((s, e, int(m.group(1))))

    # Paso F: Consumir directivas estructurales sobre columnas publicadas
    for m in RE_DIRECTIVE.finditer(query):
        s, e = m.span()
        if not tracker.has_overlap(s, e):
            tracker.mark(s, e)

    # Paso G: Consumir unidades y términos de medida
    has_level_measure_token = False
    has_delta_measure_token = False
    for m in RE_MEASURES_UNITS.finditer(query):
        s, e = m.span()
        tok = m.group(0).lower()
        if any(k in tok for k in ("variaci", "diferencia", "delta", "puntos", "Δ", "∆", "\u0394", "\u2206")):
            has_delta_measure_token = True
        if any(k in tok for k in ("porcentaje", "%", "％")):
            has_level_measure_token = True
        if not tracker.has_overlap(s, e):
            tracker.mark(s, e)

    # Paso H: Consumir sintaxis gramatical permitida (tokens atómicos sin engullir medidas/etiquetas)
    for m in RE_STRUCTURAL_TOKENS.finditer(query):
        s, e = m.span()
        if not tracker.has_overlap(s, e):
            tracker.mark(s, e)

    # Paso I: Auditoría exhaustiva de tokens residuales (palabras, dígitos u operadores no consumidos)
    residuals = tracker.residual_unconsumed()
    for s, e, text in residuals:
        cleaned = re.sub(r"[\s\?¿\.,;:\(\)\"\'«»]", "", text)
        if cleaned:
            return {
                "status": "unsupported",
                "reason": f"La consulta contiene restricciones, términos o cláusulas no soportadas ({text.strip()!r}).",
            }

    # Paso J: Comprobación de coherencia entre medida solicitada y dimensión temporal
    # Política conservadora estricta: rechazar combinaciones mixtas nivel+delta y multiconcepto+delta
    if detected_deltas and detected_single_years:
        return {
            "status": "unsupported",
            "reason": "Consultas combinadas de nivel y variación temporal no están soportadas.",
        }

    if has_level_measure_token and has_delta_measure_token:
        return {
            "status": "unsupported",
            "reason": "Consultas con términos o medidas contradictorias de nivel y variación no están soportadas.",
        }

    if detected_deltas:
        if has_level_measure_token and not has_delta_measure_token:
            return {
                "status": "unsupported",
                "reason": "Conflicto entre medida solicitada (porcentaje / nivel) y especificación temporal de variación.",
            }
    elif has_delta_measure_token:
        return {
            "status": "unsupported",
            "reason": "Consulta de variación carece de especificación de años de variación válidos.",
        }

    if len(matched_labels) > 1 and detected_deltas:
        return {
            "status": "unsupported",
            "reason": "Consultas multiconcepto con variación temporal no están soportadas.",
        }

    # Paso K: Construcción y resolución biunívoca de celdas objetivo
    target_cells: list[dict[str, Any]] = []

    # Caso 1: Multi-concepto (conjunción de niveles acotada por cláusula)
    if len(matched_labels) > 1:
        # Requerir correspondencia exacta entre recuento de ocurrencias temporales y conceptos
        if len(detected_single_years) != len(matched_labels):
            return {
                "status": "unsupported",
                "reason": (
                    f"Discrepancia temporal en multiconcepto: se detectaron {len(detected_single_years)} "
                    f"especificaciones temporales para {len(matched_labels)} conceptos solicitados."
                ),
            }

        if len(matched_label_occurrences) != len(matched_labels):
            return {
                "status": "unsupported",
                "reason": "La consulta repite o duplica conceptos de forma ambigua.",
            }

        clauses = re.split(r"\s+y\s+|,\s*", query, flags=re.IGNORECASE)

        # 1. Ninguna cláusula sin concepto reconocido puede contener años (átomos temporales huérfanos)
        for cl in clauses:
            cl_years = list(RE_SINGLE_YEAR.finditer(cl))
            cl_labels = [
                lbl
                for lbl in matched_labels
                if re.search(r"(?<!\w)" + re.escape(lbl.strip()) + r"(?!\w)", cl, flags=re.IGNORECASE)
            ]
            if cl_years and not cl_labels:
                return {
                    "status": "unsupported",
                    "reason": f"Cláusula temporal huérfana no asignada a ningún concepto ({cl.strip()!r}).",
                }
            if len(cl_labels) > 1 and cl_years:
                return {
                    "status": "unsupported",
                    "reason": f"Múltiples conceptos en una misma cláusula temporal ({cl.strip()!r}).",
                }

        # 2. Cada concepto debe tener exactamente una cláusula con exactamente un año
        for lbl in matched_labels:
            pkt = packets_by_norm_label[_norm_label(lbl)][0]
            matching_clauses = [
                cl
                for cl in clauses
                if re.search(r"(?<!\w)" + re.escape(lbl.strip()) + r"(?!\w)", cl, flags=re.IGNORECASE)
            ]
            if len(matching_clauses) != 1:
                return {
                    "status": "unsupported",
                    "reason": f"El concepto {lbl!r} no tiene una delimitación de cláusula unívoca.",
                }
            c_clause = matching_clauses[0]
            c_years = [int(y.group(1)) for y in RE_SINGLE_YEAR.finditer(c_clause)]
            if len(c_years) != 1:
                return {
                    "status": "unsupported",
                    "reason": f"El concepto {lbl!r} debe tener exactamente un año explícito en su cláusula.",
                }
            target_year = c_years[0]
            matching_cells = [
                c for c in pkt["cells"] if c.get("measure") == "level" and c.get("year") == target_year
            ]
            if not matching_cells:
                return {
                    "status": "unsupported",
                    "reason": f"No se encontró la celda para el concepto {lbl!r} en el año {target_year}.",
                }
            if len(matching_cells) > 1:
                return {
                    "status": "unsupported",
                    "reason": f"Ambigüedad: múltiples celdas físicas ({len(matching_cells)}) para el concepto {lbl!r} en el año {target_year}.",
                }
            target_cells.append(_build_target_cell(pkt, matching_cells[0]))

    # Caso 2: Concepto único con variaciones temporales (soporta 1 o múltiples deltas: todas o unsupported)
    elif detected_deltas:
        lbl = matched_labels[0]
        pkt = packets_by_norm_label[_norm_label(lbl)][0]
        for _, _, ordered_years in detected_deltas:
            matching_cells = [
                c for c in pkt["cells"] if c.get("measure") == "delta" and c.get("ordered_delta_years") == ordered_years
            ]
            if not matching_cells:
                return {
                    "status": "unsupported",
                    "reason": f"Variación publicada para {ordered_years} no existe en la tabla para {lbl!r}.",
                }
            if len(matching_cells) > 1:
                return {
                    "status": "unsupported",
                    "reason": f"Ambigüedad: múltiples celdas físicas ({len(matching_cells)}) de variación publicada para {ordered_years} en el concepto {lbl!r}.",
                }
            target_cells.append(_build_target_cell(pkt, matching_cells[0]))

    # Caso 3: Concepto único con niveles temporales (soporta 1 o múltiples años)
    else:
        lbl = matched_labels[0]
        pkt = packets_by_norm_label[_norm_label(lbl)][0]
        if not detected_single_years:
            return {
                "status": "unsupported",
                "reason": f"No se identificó ningún año válido para el concepto {lbl!r}.",
            }
        for _, _, target_year in detected_single_years:
            matching_cells = [
                c for c in pkt["cells"] if c.get("measure") == "level" and c.get("year") == target_year
            ]
            if not matching_cells:
                return {
                    "status": "unsupported",
                    "reason": f"El año {target_year} no está disponible para {lbl!r} en la tabla.",
                }
            if len(matching_cells) > 1:
                return {
                    "status": "unsupported",
                    "reason": f"Ambigüedad: múltiples celdas físicas ({len(matching_cells)}) para el concepto {lbl!r} en el año {target_year}.",
                }
            target_cells.append(_build_target_cell(pkt, matching_cells[0]))

    # Deduplicación coherente de celdas objetivo idénticas (evita listas con celdas repetidas)
    seen_cids: set[str] = set()
    deduped_targets: list[dict[str, Any]] = []
    for tc in target_cells:
        if tc["cell_id"] not in seen_cids:
            seen_cids.add(tc["cell_id"])
            deduped_targets.append(tc)
    target_cells = deduped_targets

    if not target_cells:
        return {
            "status": "unsupported",
            "reason": "No se encontraron celdas objetivo válidas para la consulta.",
        }

    # Validación semántica estricta de familia de unidad solicitada vs publicada
    has_query_percent = bool(
        re.search(r"\bporcentajes?\b", query, re.IGNORECASE)
        or "%" in query
        or "％" in query
    )
    has_query_points = bool(re.search(r"\bpuntos\b", query, re.IGNORECASE))
    has_query_currency = bool(
        "€" in query
        or re.search(r"\b(?:millones\s+de\s+euros?|miles\s+de\s+euros?|euros?)\b", query, re.IGNORECASE)
    )
    has_query_pib = bool(re.search(r"\b(?:pib|gdp)\b", query, re.IGNORECASE))

    # Rechazar consultas con familias de unidad conflictivas
    if has_query_percent and has_query_currency:
        return {
            "status": "unsupported",
            "reason": "La consulta combina unidades incompatibles (porcentaje y moneda).",
        }
    if has_query_points and has_query_currency:
        return {
            "status": "unsupported",
            "reason": "La consulta combina unidades incompatibles (puntos y moneda).",
        }

    # 1. Denominador PIB explícito requerido
    if has_query_pib:
        for tc in target_cells:
            if "pib" not in str(tc.get("unit", "")).lower():
                return {
                    "status": "unsupported",
                    "reason": f"La consulta solicita datos en PIB pero la tabla publica unidades en {tc.get('unit')!r}.",
                }

    # 2. Familia porcentaje: '%'/porcentaje no puede responder con moneda ni variación en puntos
    if has_query_percent:
        for tc in target_cells:
            u = str(tc.get("unit", "")).lower()
            if any(k in u for k in ("euro", "€", "millones", "miles")):
                return {
                    "status": "unsupported",
                    "reason": f"La consulta solicita porcentaje pero la tabla publica unidades monetarias ({tc.get('unit')!r}).",
                }
            if "puntos" in u or tc.get("measure") == "delta":
                return {
                    "status": "unsupported",
                    "reason": f"La consulta solicita porcentaje pero la celda corresponde a variación en {tc.get('unit')!r}.",
                }
            if "%" not in u and "porcentaje" not in u:
                return {
                    "status": "unsupported",
                    "reason": f"La consulta solicita porcentaje pero la tabla publica unidades en {tc.get('unit')!r}.",
                }

    # 3. Familia puntos: 'puntos' no puede responder con moneda ni nivel porcentual
    if has_query_points:
        for tc in target_cells:
            u = str(tc.get("unit", "")).lower()
            if any(k in u for k in ("euro", "€", "millones", "miles")):
                return {
                    "status": "unsupported",
                    "reason": f"La consulta solicita puntos pero la tabla publica unidades monetarias ({tc.get('unit')!r}).",
                }
            if tc.get("measure") != "delta" or "puntos" not in u:
                return {
                    "status": "unsupported",
                    "reason": f"La consulta solicita puntos pero la tabla publica nivel en {tc.get('unit')!r}.",
                }

    # 4. Familia moneda: solicitud explícita de moneda requiere unidad monetaria en la tabla
    if has_query_currency:
        for tc in target_cells:
            u = str(tc.get("unit", "")).lower()
            if not any(k in u for k in ("euro", "€", "millones", "miles")):
                return {
                    "status": "unsupported",
                    "reason": f"La consulta solicita importe monetario pero la tabla publica unidades en {tc.get('unit')!r}.",
                }

    first_pkt = matched_packets[0]
    return {
        "status": "supported",
        "table_id": first_pkt["table_id"],
        "table_caption": first_pkt["table_caption"],
        "target_cells": target_cells,
    }


def _build_target_cell(packet: dict[str, Any], cell: dict[str, Any]) -> dict[str, Any]:
    """Construye el descriptor canónico de celda solicitada."""
    return {
        "cell_id": cell["cell_id"],
        "concept": cell["row_label"],
        "row_label": cell["row_label"],
        "header_path": cell["header_path"],
        "year": cell.get("year"),
        "ordered_delta_years": cell.get("ordered_delta_years"),
        "unit": cell["unit"],
        "measure": cell["measure"],
        "expected_value": cell["value_decimal"],
        "doc_id": packet["doc_id"],
        "source": packet["source"],
        "page": packet["page"],
        "table_id": packet["table_id"],
        "extraction_id": packet["extraction_id"],
    }


def _normalize_header_path_newlines(header_path: list[str]) -> list[str]:
    """Normaliza únicamente la codificación de saltos de línea (CRLF, LF, bare CR) a LF en cada segmento."""
    return [segment.replace("\r\n", "\n").replace("\r", "\n") for segment in header_path]


def guard_table_facts(
    facts: list[dict[str, Any]],
    request: dict[str, Any] | None,
    packets: list[dict[str, Any]],
) -> dict[str, Any]:
    """Valida exhaustivamente hechos tabulares contra celdas exactas y paquetes autoritativos.

    Acepta únicamente hechos tipados que coincidan exactamente con todas las celdas
    solicitadas re-resueltas contra los paquetes físicos originales:
    - concept / row_label
    - header_path en orden exacto derivado de la cuadrícula
    - year / ordered_delta_years con comprobación estricta de tipos (no bool-as-int)
    - unit exacta (% PIB vs puntos PIB)
    - measure ("level" | "delta")
    - Decimal signed value (sin floats)
    - doc_id, source, date, page, table_id, extraction_id
    - evidence_ids completas para la fila (título + cabecera + fila)
    Rechaza hechos faltantes, sobrantes, duplicados, claves libres/extra desconocidas,
    peticiones falsificadas y citas no pertenecientes al paquete.
    """
    if not isinstance(request, dict) or request.get("status") != "supported":
        reason = request.get("reason", "Petición no soportada o inválida") if isinstance(request, dict) else "Sin petición"
        return {"decision": "rejected", "reasons": [reason], "accepted_facts": []}

    target_cells = request.get("target_cells", [])
    if not target_cells:
        return {"decision": "rejected", "reasons": ["Petición soportada no contiene celdas solicitadas"], "accepted_facts": []}

    # Deduplicación / coherencia de target_cells en la petición recibida
    target_cids = [tc.get("cell_id") for tc in target_cells if isinstance(tc, dict)]
    if len(target_cids) != len(set(target_cids)):
        return {
            "decision": "rejected",
            "reasons": ["Petición inválida: target_cells contiene celdas duplicadas"],
            "accepted_facts": [],
        }

    if not isinstance(facts, list):
        return {"decision": "rejected", "reasons": ["facts debe ser una lista de diccionarios"], "accepted_facts": []}

    reasons: list[str] = []

    # Construir índice autoritativo a partir de los paquetes físicos
    authoritative_cell_index: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for p in packets:
        for c in p.get("cells", []):
            authoritative_cell_index[c["cell_id"]] = (c, p)

    # Validar que los objetivos de la petición existan en los paquetes y no hayan sido falsificados
    for tc in target_cells:
        cid = tc.get("cell_id")
        if not cid or cid not in authoritative_cell_index:
            return {
                "decision": "rejected",
                "reasons": [f"Objetivo de petición {cid!r} no existe en los paquetes de evidencia autoritativos"],
                "accepted_facts": [],
            }
        source_cell, source_packet = authoritative_cell_index[cid]
        # Anti-forgery: si la petición fue alterada externamente para esperar un valor distinto al real de la tabla, rechazar
        if tc.get("expected_value") is not None:
            raw_ev = tc["expected_value"]
            if isinstance(raw_ev, (bool, float)) or raw_ev is None:
                return {
                    "decision": "rejected",
                    "reasons": [f"Petición falsificada detectada para {cid}: expected_value tipo inválido ({type(raw_ev).__name__})"],
                    "accepted_facts": [],
                }
            try:
                ev_str = parse_canonical_decimal(raw_ev)
                if Decimal(ev_str) != Decimal(source_cell["value_decimal"]):
                    return {
                        "decision": "rejected",
                        "reasons": [f"Petición falsificada detectada para {cid}: expected_value difiere del dato físico original"],
                        "accepted_facts": [],
                    }
            except (ValueError, InvalidOperation) as err:
                return {
                    "decision": "rejected",
                    "reasons": [f"Petición falsificada detectada para {cid}: expected_value malformado o no finito ({raw_ev!r}): {err}"],
                    "accepted_facts": [],
                }

        # Validar unicidad física de la celda objetivo dentro de su paquete autoritativo
        if source_cell.get("measure") == "level":
            dup_cells = [
                c for c in source_packet.get("cells", [])
                if c.get("measure") == "level" and c.get("year") == source_cell.get("year")
            ]
            if len(dup_cells) > 1:
                return {
                    "decision": "rejected",
                    "reasons": [
                        f"Celda solicitada {cid} es ambigua en su tabla: múltiples celdas físicas ({len(dup_cells)}) para el año {source_cell.get('year')}"
                    ],
                    "accepted_facts": [],
                }
        elif source_cell.get("measure") == "delta":
            dup_cells = [
                c for c in source_packet.get("cells", [])
                if c.get("measure") == "delta" and c.get("ordered_delta_years") == source_cell.get("ordered_delta_years")
            ]
            if len(dup_cells) > 1:
                return {
                    "decision": "rejected",
                    "reasons": [
                        f"Celda solicitada {cid} es ambigua en su tabla: múltiples celdas físicas ({len(dup_cells)}) para la variación {source_cell.get('ordered_delta_years')}"
                    ],
                    "accepted_facts": [],
                }

    # Validar duplicados en facts
    seen_cell_ids: set[str] = set()
    for f in facts:
        if not isinstance(f, dict):
            return {"decision": "rejected", "reasons": ["Cada hecho en facts debe ser un diccionario"], "accepted_facts": []}
        cid = f.get("cell_id")
        if not cid:
            reasons.append("Hecho carece de cell_id")
            continue
        if cid in seen_cell_ids:
            reasons.append(f"Hecho duplicado para cell_id {cid!r}")
        seen_cell_ids.add(cid)

    if reasons:
        return {"decision": "rejected", "reasons": reasons, "accepted_facts": []}

    # Verificar correspondencia biunívoca en recuento
    if len(facts) < len(target_cells):
        return {
            "decision": "rejected",
            "reasons": [f"Faltan hechos solicitados: esperados {len(target_cells)}, recibidos {len(facts)}"],
            "accepted_facts": [],
        }
    if len(facts) > len(target_cells):
        return {
            "decision": "rejected",
            "reasons": [f"Hechos adicionales no solicitados: esperados {len(target_cells)}, recibidos {len(facts)}"],
            "accepted_facts": [],
        }

    target_by_id = {tc["cell_id"]: tc for tc in target_cells}

    REQUIRED_FACT_FIELDS = [
        "cell_id",
        "concept",
        "header_path",
        "unit",
        "measure",
        "value",
        "doc_id",
        "source",
        "date",
        "page",
        "table_id",
        "extraction_id",
        "evidence_ids",
    ]

    for fact in facts:
        cid = fact.get("cell_id")
        if cid not in target_by_id:
            reasons.append(f"cell_id {cid!r} no coincide con ninguna celda solicitada")
            continue

        source_cell, source_packet = authoritative_cell_index[cid]

        # 1. Rechazo de claves libres desconocidas o no tipadas
        unknown_keys = set(fact.keys()) - ALLOWED_FACT_KEYS
        if unknown_keys:
            reasons.append(f"Claves desconocidas o prosa no tipada en hecho {cid}: {sorted(unknown_keys)}")
            continue

        # 2. Validación estricta de presencia de todos los campos requeridos (sin verdad implícita)
        missing_fields = [fld for fld in REQUIRED_FACT_FIELDS if fld not in fact or fact[fld] is None]
        if missing_fields:
            reasons.append(f"Campos requeridos ausentes o nulos en hecho {cid}: {missing_fields}")
            continue

        # 3. Comprobación estricta de tipos conscientes (no bool-as-int)
        if isinstance(fact.get("page"), bool) or not isinstance(fact.get("page"), int):
            reasons.append(f"page en {cid} debe ser un entero estricto, no booleano")
            continue

        f_measure = fact.get("measure")
        if f_measure not in ("level", "delta") or f_measure != source_cell["measure"]:
            reasons.append(f"measure erróneo en {cid}: esperado {source_cell['measure']!r}, obtenido {f_measure!r}")
            continue

        if f_measure == "level":
            f_year = fact.get("year")
            if isinstance(f_year, bool) or not isinstance(f_year, int):
                reasons.append(f"year en {cid} debe ser un entero estricto para nivel, no booleano")
                continue
            if f_year != source_cell["year"]:
                reasons.append(f"Año erróneo en {cid}: esperado {source_cell['year']}, obtenido {f_year}")
            if fact.get("ordered_delta_years") is not None:
                reasons.append(f"ordered_delta_years debe ser None para nivel en {cid}")
        else:
            f_deltas = fact.get("ordered_delta_years")
            if not isinstance(f_deltas, list) or len(f_deltas) != 2 or any(isinstance(y, bool) or not isinstance(y, int) for y in f_deltas):
                reasons.append(f"ordered_delta_years en {cid} debe ser una lista de 2 enteros estrictos")
                continue
            if f_deltas != source_cell["ordered_delta_years"]:
                reasons.append(f"Años de variación ordenados erróneos en {cid}: esperados {source_cell['ordered_delta_years']}, obtenidos {f_deltas}")
            if fact.get("year") is not None:
                reasons.append(f"year debe ser None para variación en {cid}")

        # 4. header_path en orden exacto derivado de la cuadrícula
        f_hpath = fact.get("header_path")
        s_hpath = source_cell.get("header_path")
        if (
            not isinstance(f_hpath, list)
            or any(not isinstance(s, str) for s in f_hpath)
            or not isinstance(s_hpath, list)
            or any(not isinstance(s, str) for s in s_hpath)
            or _normalize_header_path_newlines(f_hpath) != _normalize_header_path_newlines(s_hpath)
        ):
            reasons.append(
                f"header_path en {cid} no coincide en orden exacto: esperado {source_cell['header_path']}, obtenido {f_hpath}"
            )

        # 5. Concepto / row_label
        f_concept = str(fact.get("concept") or fact.get("row_label", "")).strip().lower()
        s_concept = str(source_cell["row_label"]).strip().lower()
        if f_concept != s_concept:
            reasons.append(f"Concepto erróneo en {cid}: esperado {source_cell['row_label']!r}, obtenido {fact.get('concept')!r}")

        # 6. Unidad (% PIB vs puntos PIB)
        f_unit = str(fact.get("unit", "")).strip().lower().replace("del ", "")
        s_unit = str(source_cell["unit"]).strip().lower().replace("del ", "")
        if f_unit != s_unit:
            reasons.append(f"Unidad errónea en {cid}: esperada {source_cell['unit']!r}, obtenida {fact.get('unit')!r}")

        # 7. Valor numérico Decimal canónico (nunca float ni recomputado)
        f_val = fact.get("value")
        if isinstance(f_val, (bool, float)) or f_val is None:
            reasons.append(f"Valor numérico en {cid} no puede ser bool, null ni float; use string o Decimal canónico")
            continue
        try:
            f_canon = parse_canonical_decimal(f_val)
            f_dec = Decimal(f_canon)
            s_dec = Decimal(source_cell["value_decimal"])
            if f_dec != s_dec:
                reasons.append(
                    f"Valor numérico en {cid} no coincide con dato publicado: esperado {source_cell['value_decimal']}, obtenido {f_val}"
                )
        except (ValueError, TypeError, InvalidOperation) as err:
            reasons.append(f"Valor numérico inválido en {cid} ({f_val!r}): {err}")

        # 8. Metadatos de procedencia completos
        if fact.get("doc_id") != source_packet["doc_id"]:
            reasons.append(f"doc_id discrepante en {cid}")
        if fact.get("source") != source_packet["source"]:
            reasons.append(f"source discrepante en {cid}")
        if fact.get("date") != source_packet["date"]:
            reasons.append(f"date discrepante en {cid}")
        if fact.get("page") != source_packet["page"]:
            reasons.append(f"page discrepante en {cid}")
        if fact.get("table_id") != source_packet["table_id"]:
            reasons.append(f"table_id discrepante en {cid}")
        if fact.get("extraction_id") != source_packet["extraction_id"]:
            reasons.append(f"extraction_id discrepante en {cid}")

        # 9. Validación exhaustiva de evidence_ids
        f_ev_ids = fact.get("evidence_ids")
        if not isinstance(f_ev_ids, list) or not f_ev_ids:
            reasons.append(f"evidence_ids debe ser una lista no vacía en {cid}")
            continue

        if len(f_ev_ids) != len(set(f_ev_ids)):
            reasons.append(f"evidence_ids contiene identificadores duplicados en {cid}")
            continue

        required_ev_ids = set(source_cell["evidence_ids"])
        given_ev_ids = set(f_ev_ids)

        # Verificar que incluya la cita obligatoria de fila
        expected_row_id = f"{source_packet['table_id']}_r{source_cell['row_index']}_row"
        if expected_row_id not in given_ev_ids:
            reasons.append(f"evidence_ids en {cid} omite la cita obligatoria de fila {expected_row_id!r}")

        # Verificar que incluya la cita obligatoria de cabecera
        expected_header_id = f"{source_packet['table_id']}_header"
        if expected_header_id not in given_ev_ids:
            reasons.append(f"evidence_ids en {cid} omite la cita obligatoria de cabecera {expected_header_id!r}")

        # Verificar que incluya la cita obligatoria de título si la tabla tiene caption
        if source_packet.get("table_caption"):
            expected_caption_id = f"{source_packet['table_id']}_caption"
            if expected_caption_id not in given_ev_ids:
                reasons.append(f"evidence_ids en {cid} omite la cita obligatoria de título {expected_caption_id!r}")

        # Verificar que no contenga referencias falsas o ajenas
        if given_ev_ids != required_ev_ids:
            diff = given_ev_ids ^ required_ev_ids
            reasons.append(f"evidence_ids en {cid} no coincide con las citas requeridas del paquete: diferencias {diff}")

    if reasons:
        return {"decision": "rejected", "reasons": reasons, "accepted_facts": []}

    accepted_facts: list[dict[str, Any]] = []
    for f in facts:
        f_copy = copy.deepcopy(f)
        if "value" in f_copy and f_copy["value"] is not None:
            try:
                f_copy["value"] = parse_canonical_decimal(f_copy["value"])
            except Exception:
                pass
        if "value_decimal" in f_copy and f_copy["value_decimal"] is not None:
            try:
                f_copy["value_decimal"] = parse_canonical_decimal(f_copy["value_decimal"])
            except Exception:
                pass
        accepted_facts.append(f_copy)

    return {"decision": "accepted", "reasons": [], "accepted_facts": accepted_facts}


def render_table_facts(
    facts: list[dict[str, Any]],
    request: dict[str, Any] | None,
    packets: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Genera afirmaciones canónicas verificadas a partir de hechos validados por la guardia.

    Primero ejecuta guard_table_facts; ante cualquier rechazo produce una lista vacía,
    sin inventar explicaciones ni sustituir respuestas fallidas con valores esperados.
    """
    guard_res = guard_table_facts(facts, request, packets)
    if guard_res.get("decision") != "accepted":
        return []

    accepted_facts = guard_res.get("accepted_facts", [])
    claims: list[dict[str, Any]] = []

    for idx, fact in enumerate(accepted_facts):
        cid = f"claim_table_{idx + 1}"
        concept = fact.get("concept") or fact.get("row_label", "")
        unit = fact.get("unit", "")
        val = parse_canonical_decimal(fact.get("value") or fact.get("value_decimal"))

        if fact.get("measure") == "level":
            year = fact.get("year")
            claim_text = f"Para {concept} en {year}, la cifra publicada es de {val} {unit}."
        else:
            y1, y2 = fact.get("ordered_delta_years", [None, None])
            claim_text = f"Para {concept}, la variación publicada ({y1} menos {y2}) es de {val} {unit}."

        ev_ids = fact.get("evidence_ids") or []
        claims.append(
            {
                "claim_id": cid,
                "text": claim_text,
                "concept": concept,
                "year": fact.get("year"),
                "ordered_delta_years": fact.get("ordered_delta_years"),
                "unit": unit,
                "measure": fact.get("measure"),
                "value": val,
                "cell_id": fact.get("cell_id"),
                "evidence_ids": ev_ids,
                "status": "answered",
            }
        )

    return claims
