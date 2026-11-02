"""Bounded, source-pinned Excel to DuckDB pilot. No RAG or model calls."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP, localcontext
import hashlib
import json
from pathlib import Path
import re
import unicodedata
import xml.etree.ElementTree as ET
from zipfile import ZipFile

import duckdb

ROOT = Path(__file__).resolve().parents[1]
NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
      "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
      "c": "http://schemas.openxmlformats.org/drawingml/2006/chart"}
BOOKS = {
    "2025": ("Airef/2025_Opinion_Sostenibilidad_AAPP/"
             "Graficos-y-cuadros Segunda Opinión sobre la sostenibilidad de las Administraciones públicas a largo plazo.xlsx",
             "a74921530a00b649860530503abe6a532d540e778a150015dfe5864710ddccfa",
             "Opinion sobre la Sostenibilidad de las AAPP a largo plazo Marzo 2025.pdf"),
    "2026": ("Airef/2026_Estudio_Regla_Gasto_Pensiones/"
             "AIReF.-Tablas-y-graficos.-Estudio-de-evaluacion-de-la-regla-de-gasto-en-pensiones.xlsx",
             "ee7b6e18fe3f5c9197fce6d086b8cb51c366f4e069b539c5f38fec62a55597c5",
             "Estudio_Regla_Gasto_Pensiones-2026.pdf"),
}
PDF_HASHES = {
    "2025": "2af1f49f2d1554c42e0eae82f6855683224d31b89b974216c6d40f1a48680ea1",
    "2026": "a41ffe179cb55e31fed1dba5ac28c83b521eb0f74fe0655d0ed3141c42727787",
}
METRICS = ["Empleos no Financieros", "Modelizados", "Pensiones", "Intereses",
           "Sanidad", "Cuidados larga duración", "Educación", "Desempleo", "No modelizados"]
BLOCKS = {
    "cuadro9_2025": dict(book="2025", sheet="Cuadro 9 ", title_cell="C4",
                         title="EMPLEOS NO FINANCIEROS SIN PRTR (%PIB)", index="Cuadros", page=95),
    "cuadro12_2026": dict(book="2026", sheet="Cuadro 12", title_cell="B3",
                          title="CUADRO 12. EMPLEOS NO FINANCIEROS SIN PRTR (% PIB)", index="Cuadros", page=65),
    "grafico20_2026": dict(book="2026", sheet="Gráfico 20", title_cell="B3",
                           title="GRÁFICO 20. EMPLEOS NO FINANCIEROS SIN PRTR (% PIB)", index="Gráficos", page=65),
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Nonempty text required")
    return " ".join(unicodedata.normalize("NFC", text).casefold().split())


@dataclass(frozen=True)
class Cell:
    address: str
    kind: str
    value: str | None
    formula: str | None
    style: str

    def number(self) -> Decimal:
        if self.kind != "n" or self.value is None:
            raise ValueError(f"Numeric cached value required at {self.address}")
        value = Decimal(self.value)
        if not value.is_finite() or abs(value) >= Decimal(10) ** 14:
            raise ValueError(f"Invalid numeric value at {self.address}")
        with localcontext() as ctx:
            ctx.prec = 50
            if value != value.quantize(Decimal("1e-24")):
                raise ValueError(f"Unsupported precision at {self.address}")
        return value


class ExcelXML:
    """Read source XML without recalculating, saving or dropping chart extensions."""

    def __init__(self, path: Path):
        self.path = path
        with ZipFile(path) as z:
            self.parts = {name: z.read(name) for name in z.namelist() if name.endswith(".xml")}
            rel = ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))
        self.strings = []
        if "xl/sharedStrings.xml" in self.parts:
            self.strings = ["".join(n.itertext()) for n in
                            ET.fromstring(self.parts["xl/sharedStrings.xml"]).findall("m:si", NS)]
        links = {n.attrib["Id"]: n.attrib["Target"] for n in rel}
        self.sheets = {}
        for sheet in ET.fromstring(self.parts["xl/workbook.xml"]).findall("m:sheets/m:sheet", NS):
            target = links[sheet.attrib[f"{{{NS['r']}}}id"]]
            self.sheets[sheet.attrib["name"]] = target.lstrip("/") if target.startswith("/") else "xl/" + target

    def read(self, sheet: str) -> dict[str, Cell]:
        result = {}
        shared = {}
        for node in ET.fromstring(self.parts[self.sheets[sheet]]).findall("m:sheetData/m:row/m:c", NS):
            value, formula = node.find("m:v", NS), node.find("m:f", NS)
            kind = node.get("t", "n")
            text = value.text if value is not None else None
            if kind == "s" and text is not None:
                text = self.strings[int(text)]
            elif kind == "inlineStr":
                text = "".join(t.text or "" for t in node.findall("m:is//m:t", NS))
            if value is None and formula is None and text is None:
                continue
            expression = formula.text if formula is not None else None
            if formula is not None and formula.get("t") == "shared":
                key = formula.attrib["si"]
                if expression:
                    shared[key] = (node.attrib["r"], expression)
                else:
                    anchor, expression = shared[key]
                    expression = translate_shared(expression, anchor, node.attrib["r"])
            if formula is not None and not expression:
                raise ValueError("Empty formula outside the pilot")
            cell = Cell(node.attrib["r"], kind, text, expression, node.get("s", "0"))
            result[cell.address] = cell
        return result

    def chart_bindings(self) -> list[tuple[str, str, str, str]]:
        result = []
        for name, data in self.parts.items():
            if not name.startswith("xl/charts/chart"):
                continue
            for series in ET.fromstring(data).findall(".//c:ser", NS):
                refs = [n.text for n in series.findall(".//c:f", NS)]
                if refs and any("'Gráfico 20'!" in (r or "") for r in refs):
                    if len(refs) != 3:
                        raise ValueError("Unexpected graph series binding")
                    result.append((name, *refs))
        expected = {(f"'Gráfico 20'!$B${r}", "'Gráfico 20'!$C$5:$AL$5",
                     f"'Gráfico 20'!$C${r}:$AL${r}") for r in (6, 7, 8)}
        if len(result) != 3 or {r[1:] for r in result} != expected:
            raise ValueError("Chart 20 source ranges changed")
        return result


def assert_text(cells: dict[str, Cell], address: str, expected: str) -> None:
    if normalized(cells[address].value or " ") != normalized(expected):
        raise ValueError(f"Unexpected label at {address}: expected {expected}")


def close(actual: Decimal, expected: Decimal, where: str) -> None:
    # Excel cached binary arithmetic can differ in the last stored digits.
    if abs(actual - expected) > Decimal("1e-10"):
        raise ValueError(f"Formula/cache mismatch: {where}")


def column(n: int) -> str:
    result = ""
    while n:
        n, r = divmod(n - 1, 26)
        result = chr(65 + r) + result
    return result


def translate_shared(expression: str, anchor: str, target: str) -> str:
    """Only the simple relative formulas actually admitted by this pilot."""
    if not re.fullmatch(r"\+?[A-Z]+\d+-[A-Z]+\d+|[A-Z]+\d+\+1", expression):
        raise ValueError("Unsupported shared formula")

    def coordinates(address):
        letters, row = re.fullmatch(r"([A-Z]+)(\d+)", address).groups()
        col = 0
        for letter in letters:
            col = col * 26 + ord(letter) - 64
        return col, int(row)

    a, t = coordinates(anchor), coordinates(target)

    def replace(match):
        col, row = coordinates(match.group())
        col, row = col + t[0] - a[0], row + t[1] - a[1]
        if col < 1 or row < 1:
            raise ValueError("Invalid translated reference")
        return f"{column(col)}{row}"

    return re.sub(r"[A-Z]+\d+", replace, expression)


def extract_block(block: str, cells: dict[str, Cell]) -> list[dict]:
    spec = BLOCKS[block]
    assert_text(cells, spec["title_cell"], spec["title"])
    output = []

    def add(metric, series, kind, year, start, end, address):
        cell = cells[address]
        value = cell.number()
        output.append(dict(block=block, metric=metric, series=series, kind=kind,
                           year=year, start_year=start, end_year=end,
                           unit="percent_gdp" if kind == "level" else "percentage_points_gdp",
                           value=value, source_cell=address, formula=cell.formula))

    if block == "grafico20_2026":
        years = {}
        for n in range(3, 39):
            col, expected = column(n), 2015 + n - 3
            cell = cells[f"{col}5"]
            if cell.number() != expected or (n > 3 and cell.formula != f"{column(n-1)}5+1"):
                raise ValueError("Graph year header changed")
            years[col] = expected
        for row, label, first, last in [(6, "Histórico", 2015, 2025),
                                         (7, "AIReF 2026", 2025, 2050),
                                         (8, "Opinión 2025", 2023, 2050)]:
            assert_text(cells, f"B{row}", label)
            for col, year in years.items():
                address = f"{col}{row}"
                if first <= year <= last:
                    if cells[address].formula is not None:
                        raise ValueError("Unexpected formula in graph values")
                    add(METRICS[0], label, "level", year, None, None, address)
                elif address in cells and cells[address].value is not None:
                    raise ValueError("Graph series support changed")
    else:
        old = block == "cuadro9_2025"
        header, first_row = (6, 7) if old else (5, 6)
        levels = list(zip("DEFGH", [2023, 2029, 2041, 2050, 2070])) if old else list(zip("DEF", [2025, 2030, 2050]))
        deltas = [("J", "D", "E"), ("K", "E", "G"), ("L", "G", "H")] if old else [("H", "D", "E"), ("I", "E", "F")]
        years = dict(levels)
        for col, year in levels:
            if cells[f"{col}{header}"].number() != year:
                raise ValueError("Table year header changed")
        for col, a, b in deltas:
            expected = f"Δ {years[b]}-{years[a]}" if old else f"{years[a]}-{years[b]}"
            assert_text(cells, f"{col}{header}", expected)
        for offset, metric in enumerate(METRICS):
            row = first_row + offset
            assert_text(cells, f"B{row}", metric)
            for col, year in levels:
                address = f"{col}{row}"
                formula = cells[address].formula
                if old and metric == "Modelizados":
                    if formula != f"SUM({col}9:{col}14)":
                        raise ValueError("Unexpected subtotal formula")
                    close(cells[address].number(), sum((cells[f"{col}{r}"].number() for r in range(9, 15)), Decimal(0)), address)
                elif formula is not None:
                    raise ValueError("Unexpected level formula")
                add(metric, "Opinión 2025" if old else "AIReF 2026", "level", year, None, None, address)
            for col, a, b in deltas:
                address = f"{col}{row}"
                if cells[address].formula != ("+" if old else "") + f"{b}{row}-{a}{row}":
                    raise ValueError("Unexpected delta formula")
                close(cells[address].number(), cells[f"{b}{row}"].number() - cells[f"{a}{row}"].number(), address)
                add(metric, "Opinión 2025" if old else "AIReF 2026", "published_change", None, years[a], years[b], address)
    return output


DDL = """
CREATE TABLE sources(book VARCHAR PRIMARY KEY, xlsx_path VARCHAR, xlsx_sha256 VARCHAR,
                     pdf_path VARCHAR, pdf_sha256 VARCHAR);
CREATE TABLE blocks(block VARCHAR PRIMARY KEY, book VARCHAR REFERENCES sources(book),
                    sheet VARCHAR, title VARCHAR, index_page INTEGER, page_basis VARCHAR);
CREATE TABLE raw_cells(block VARCHAR REFERENCES blocks(block), address VARCHAR, kind VARCHAR,
                       raw_value VARCHAR, formula VARCHAR, style_id VARCHAR, PRIMARY KEY(block,address));
CREATE TABLE source_parts(book VARCHAR REFERENCES sources(book), part VARCHAR, xml VARCHAR,
                          PRIMARY KEY(book,part));
CREATE TABLE chart_bindings(part VARCHAR, label_ref VARCHAR, years_ref VARCHAR, values_ref VARCHAR);
CREATE TABLE observations(block VARCHAR REFERENCES blocks(block), metric VARCHAR, series VARCHAR,
    kind VARCHAR CHECK(kind IN ('level','published_change')), year INTEGER, start_year INTEGER, end_year INTEGER,
    unit VARCHAR, value DECIMAL(38,24) NOT NULL, source_cell VARCHAR, formula VARCHAR,
    FOREIGN KEY(block, source_cell) REFERENCES raw_cells(block,address),
    PRIMARY KEY(block,source_cell),
    CHECK((kind='level' AND year IS NOT NULL AND start_year IS NULL AND end_year IS NULL AND unit='percent_gdp') OR
          (kind='published_change' AND year IS NULL AND start_year IS NOT NULL AND end_year IS NOT NULL
           AND start_year < end_year AND unit='percentage_points_gdp')));
"""


def build(output_dir: Path) -> dict:
    output_dir = output_dir.resolve()
    output_dir.relative_to(ROOT)
    if output_dir.exists():
        raise FileExistsError("Output directory already exists; no overwrite")
    books, sources = {}, []
    for key, (name, expected, pdf_name) in BOOKS.items():
        path = ROOT / name
        if sha(path) != expected:
            raise ValueError(f"Source workbook version changed: {key}")
        pdf = path.parent / pdf_name
        if sha(pdf) != PDF_HASHES[key]:
            raise ValueError(f"Source PDF version changed: {key}")
        sources.append((key, name, expected, pdf.relative_to(ROOT).as_posix(), PDF_HASHES[key]))
        books[key] = ExcelXML(path)
    extracted, raw = {}, {}
    for block, spec in BLOCKS.items():
        book = books[spec["book"]]
        raw[block] = book.read(spec["sheet"])
        extracted[block] = extract_block(block, raw[block])
        index = book.read(spec["index"])
        matches = [c for c in index.values() if c.value == spec["sheet"]]
        if len(matches) != 1:
            raise ValueError("Missing/ambiguous index entry")
        row = re.search(r"\d+$", matches[0].address).group()
        if index[f"E{row}"].number() != spec["page"]:
            raise ValueError("Index page changed")
    bindings = books["2026"].chart_bindings()
    expected_counts = {"cuadro9_2025": 72, "cuadro12_2026": 45, "grafico20_2026": 65}
    if {k: len(v) for k, v in extracted.items()} != expected_counts:
        raise ValueError("Unexpected observation counts")
    output_dir.mkdir(parents=True)
    db = output_dir / "pilot.duckdb"
    with duckdb.connect(str(db)) as con:
        con.execute("BEGIN TRANSACTION")
        con.execute(DDL)
        con.executemany("INSERT INTO sources VALUES (?,?,?,?,?)", sources)
        for block, spec in BLOCKS.items():
            con.execute("INSERT INTO blocks VALUES (?,?,?,?,?,?)", [block, spec["book"], spec["sheet"], spec["title"], spec["page"], "excel_index_not_physical_page_verified"])
            con.executemany("INSERT INTO raw_cells VALUES (?,?,?,?,?,?)", [(block, c.address, c.kind, c.value, c.formula, c.style) for c in raw[block].values()])
            con.executemany("INSERT INTO observations VALUES (?,?,?,?,?,?,?,?,?,?,?)", [tuple(r.values()) for r in extracted[block]])
        for key, book in books.items():
            names = {"xl/styles.xml", "xl/workbook.xml", "xl/sharedStrings.xml"}
            names.update(book.sheets[s["sheet"]] for s in BLOCKS.values() if s["book"] == key)
            names.update(book.sheets[s["index"]] for s in BLOCKS.values() if s["book"] == key)
            if key == "2026":
                names.update(b[0] for b in bindings)
            con.executemany("INSERT INTO source_parts VALUES (?,?,?)", [(key, n, book.parts[n].decode("utf-8")) for n in sorted(names)])
        con.executemany("INSERT INTO chart_bindings VALUES (?,?,?,?)", bindings)
        con.execute("COMMIT")
    manifest = dict(schema_version="excel-pilot-v1", status="built_pending_independent_validation",
                    counts=expected_counts, total=182, sources=[dict(zip(["book", "xlsx_path", "xlsx_sha256", "pdf_path", "pdf_sha256"], s)) for s in sources],
                    builder_sha256=sha(Path(__file__)), duckdb_version=duckdb.__version__, db_sha256=sha(db),
                    formula_tolerance="1e-10", rag_integrated=False)
    for source in sources:
        if sha(ROOT / source[1]) != source[2] or sha(ROOT / source[3]) != source[4]:
            raise ValueError("Source changed during build")
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def lookup(db: Path, *, block: str, metric: str, series: str, year: int | None = None,
           start_year: int | None = None, end_year: int | None = None,
           kind: str = "level") -> dict:
    """Exact structured lookup; no nearest-series fallback, interpolation or free SQL."""
    if block not in BLOCKS or kind not in ("level", "published_change"):
        raise ValueError("Unknown block or kind")
    for number in (year, start_year, end_year):
        if number is not None and type(number) is not int:
            raise ValueError("Years must be integers")
    if kind == "level":
        if year is None or start_year is not None or end_year is not None:
            raise ValueError("Level lookup requires one year")
    elif year is not None or start_year is None or end_year is None or start_year >= end_year:
        raise ValueError("Published change requires an ordered interval")
    requested_metric, requested_series = normalized(metric), normalized(series)
    aliases = {normalized(m): m for m in METRICS}
    aliases[normalized("Cuidados de larga duración")] = "Cuidados larga duración"
    canonical_metric = aliases.get(requested_metric)
    if canonical_metric is None:
        raise LookupError("Metric not available")
    series_labels = {normalized(s): s for s in ["Opinión 2025", "AIReF 2026", "Histórico"]}
    canonical_series = series_labels.get(requested_series)
    if canonical_series is None:
        raise LookupError("Series not available")
    with duckdb.connect(str(db), read_only=True) as con:
        cursor = con.execute("""SELECT o.*, b.sheet, b.title, b.index_page, b.page_basis,
            s.xlsx_path, s.xlsx_sha256, s.pdf_path, s.pdf_sha256, r.raw_value
            FROM observations o JOIN blocks b USING(block) JOIN sources s USING(book)
            JOIN raw_cells r ON r.block=o.block AND r.address=o.source_cell
            WHERE o.block=? AND o.metric=? AND o.series=? AND o.kind=?
            AND o.year IS NOT DISTINCT FROM ? AND o.start_year IS NOT DISTINCT FROM ?
            AND o.end_year IS NOT DISTINCT FROM ?""",
            [block, canonical_metric, canonical_series, kind, year, start_year, end_year])
        names = [c[0] for c in cursor.description]
        rows = cursor.fetchall()
    if len(rows) != 1:
        raise LookupError("No unique observation for the requested series and period")
    result = dict(zip(names, rows[0]))
    result["value"] = str(result["value"])
    result["display_value"] = str(Decimal(result["value"]).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))
    return result


def difference(db: Path, *, block: str, metric: str, series: str, start_year: int, end_year: int) -> dict:
    if type(start_year) is not int or type(end_year) is not int or start_year >= end_year:
        raise ValueError("Difference requires increasing years")
    first = lookup(db, block=block, metric=metric, series=series, year=start_year)
    last = lookup(db, block=block, metric=metric, series=series, year=end_year)
    # Decimal context > 38 prevents implicit rounding of source precision.
    with localcontext() as ctx:
        ctx.prec = 50
        value = Decimal(last["value"]) - Decimal(first["value"])
    return dict(kind="computed_change", unit="percentage_points_gdp", value=str(value),
                display_value=str(value.quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)),
                operation="end_minus_start_unrounded", inputs=[first, last])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    builder = commands.add_parser("build")
    builder.add_argument("--output-dir", type=Path, required=True)
    query = commands.add_parser("query")
    query.add_argument("--db", type=Path, required=True)
    query.add_argument("--block", choices=BLOCKS, required=True)
    query.add_argument("--metric", required=True)
    query.add_argument("--series", required=True)
    query.add_argument("--kind", choices=["level", "published_change", "computed_change"], default="level")
    query.add_argument("--year", type=int)
    query.add_argument("--start-year", type=int)
    query.add_argument("--end-year", type=int)
    args = parser.parse_args(argv)
    try:
        if args.command == "build":
            result = build(args.output_dir)
        else:
            kwargs = dict(block=args.block, metric=args.metric, series=args.series)
            if args.kind == "computed_change":
                if args.year is not None:
                    raise ValueError("Computed change does not accept year")
                result = difference(args.db, **kwargs, start_year=args.start_year, end_year=args.end_year)
            else:
                result = lookup(args.db, **kwargs, kind=args.kind, year=args.year,
                                start_year=args.start_year, end_year=args.end_year)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (ValueError, LookupError, OSError, duckdb.Error) as exc:
        print(json.dumps({"status": "error", "reason": str(exc)}, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
