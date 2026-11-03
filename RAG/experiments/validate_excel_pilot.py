"""Independent cell-by-cell pilot validation using openpyxl, never the XML importer."""
from __future__ import annotations

import argparse
from decimal import Decimal
import json
from pathlib import Path
import warnings
import xml.etree.ElementTree as ET
from zipfile import ZipFile

import duckdb
import openpyxl

from experiments.excel_pilot import ROOT, BOOKS, PDF_HASHES, lookup, difference, sha


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate(db: Path) -> dict:
    expected = {}
    labels = ["Empleos no Financieros", "Modelizados", "Pensiones", "Intereses", "Sanidad",
              "Cuidados larga duración", "Educación", "Desempleo", "No modelizados"]
    with warnings.catch_warnings():
        # Reading only: unsupported presentation extensions are never saved.
        warnings.simplefilter("ignore", UserWarning)
        books = {key: (openpyxl.load_workbook(ROOT / info[0], data_only=True),
                       openpyxl.load_workbook(ROOT / info[0], data_only=False)) for key, info in BOOKS.items()}
    for block, key, sheet_name, header, first, level_cols, change_cols in [
        ("cuadro9_2025", "2025", "Cuadro 9 ", 6, 7, [4, 5, 6, 7, 8], [(10, 2023, 2029), (11, 2029, 2050), (12, 2050, 2070)]),
        ("cuadro12_2026", "2026", "Cuadro 12", 5, 6, [4, 5, 6], [(8, 2025, 2030), (9, 2030, 2050)]),
    ]:
        values, formulas = books[key][0][sheet_name], books[key][1][sheet_name]
        for offset, label in enumerate(labels):
            row = first + offset
            require(values.cell(row, 2).value.strip() == label, "Independent row label check failed")
            for col in level_cols:
                cell = values.cell(row, col)
                expected[(block, cell.coordinate)] = (label, "Opinión 2025" if key == "2025" else "AIReF 2026",
                    "level", int(values.cell(header, col).value), None, None, "percent_gdp", cell.value,
                    formulas[cell.coordinate].value if formulas[cell.coordinate].data_type == "f" else None)
            for col, start, end in change_cols:
                cell = values.cell(row, col)
                expected[(block, cell.coordinate)] = (label, "Opinión 2025" if key == "2025" else "AIReF 2026",
                    "published_change", None, start, end, "percentage_points_gdp", cell.value,
                    formulas[cell.coordinate].value[1:])
    graph = books["2026"][0]["Gráfico 20"]
    for row, label, first, last in [(6, "Histórico", 2015, 2025), (7, "AIReF 2026", 2025, 2050), (8, "Opinión 2025", 2023, 2050)]:
        require(graph.cell(row, 2).value == label, "Graph label changed")
        for col in range(3, 39):
            year = graph.cell(5, col).value
            require(year == 2015 + col - 3, "Independent graph year check failed")
            cell = graph.cell(row, col)
            if first <= year <= last:
                expected[("grafico20_2026", cell.coordinate)] = (labels[0], label, "level", int(year), None, None,
                                                                  "percent_gdp", cell.value, None)
            else:
                require(cell.value is None, "Unexpected graph observation outside support")
    require(len(expected) == 182, "Independent expected count mismatch")
    with duckdb.connect(str(db), read_only=True) as con:
        page_basis = "excel_index_not_physical_page_verified"
        expected_blocks = {
            ("cuadro9_2025", "2025", "Cuadro 9 ", "EMPLEOS NO FINANCIEROS SIN PRTR (%PIB)", 95, page_basis),
            ("cuadro12_2026", "2026", "Cuadro 12", "CUADRO 12. EMPLEOS NO FINANCIEROS SIN PRTR (% PIB)", 65, page_basis),
            ("grafico20_2026", "2026", "Gráfico 20", "GRÁFICO 20. EMPLEOS NO FINANCIEROS SIN PRTR (% PIB)", 65, page_basis),
        }
        require(set(con.execute("SELECT * FROM blocks").fetchall()) == expected_blocks, "Block metadata mismatch")
        for key, index_sheet, row, page, sheet in [
            ("2025", "Cuadros", 17, 95, "Cuadro 9 "), ("2026", "Cuadros", 19, 65, "Cuadro 12"),
            ("2026", "Gráficos", 36, 65, "Gráfico 20"),
        ]:
            index = books[key][0][index_sheet]
            require(index[f"E{row}"].value == page and index[f"F{row}"].value == sheet, "Original index mismatch")
        rows = con.execute("SELECT block,source_cell,metric,series,kind,year,start_year,end_year,unit,value,formula FROM observations").fetchall()
        require(len(rows) == len(expected), "Observation count mismatch")
        require({r[:2] for r in rows} == set(expected), "Missing/extra source cells")
        for row in rows:
            key, actual = row[:2], row[2:]
            reference = expected[key]
            require(actual[:7] == reference[:7], f"Dimension mismatch: {key}")
            require(abs(actual[7] - Decimal(str(reference[7]))) <= Decimal("1e-11"), f"Independent numeric mismatch: {key}")
            formula = reference[8]
            if formula and formula.startswith("="):
                formula = formula[1:]
            require(actual[8] == formula, f"Formula mismatch: {key}")
        expected_raw_keys = set()
        for block, key, sheet in [("cuadro9_2025", "2025", "Cuadro 9 "),
                                   ("cuadro12_2026", "2026", "Cuadro 12"), ("grafico20_2026", "2026", "Gráfico 20")]:
            expected_raw_keys.update((block, cell.coordinate) for row in books[key][1][sheet]
                                     for cell in row if cell.value is not None)
        require(set(con.execute("SELECT block,address FROM raw_cells").fetchall()) == expected_raw_keys, "Raw cell inventory mismatch")
        raw_checks = 0
        for key, sheet, address, text, formula in con.execute("SELECT b.book,b.sheet,r.address,r.raw_value,r.formula FROM raw_cells r JOIN blocks b USING(block)").fetchall():
            cell = books[key][0][sheet][address]
            if cell.data_type == "n" and cell.value is not None:
                require(abs(Decimal(text) - Decimal(str(cell.value))) <= Decimal("1e-11"), "Raw numeric mismatch")
            else:
                require(text == (str(cell.value) if cell.value is not None else None), "Raw text mismatch")
            expected_formula = books[key][1][sheet][address]
            require(formula == (expected_formula.value[1:] if expected_formula.data_type == "f" else None), "Raw formula mismatch")
            raw_checks += 1
        exact_values = con.execute("SELECT o.value,r.raw_value FROM observations o JOIN raw_cells r ON o.block=r.block AND o.source_cell=r.address").fetchall()
        require(all(value == Decimal(raw) for value, raw in exact_values), "Typed value differs from raw XML number")
        source_numbers = {}
        main_ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
        for block, key, part in [("cuadro9_2025", "2025", "xl/worksheets/sheet12.xml"),
                                  ("cuadro12_2026", "2026", "xl/worksheets/sheet12.xml"),
                                  ("grafico20_2026", "2026", "xl/worksheets/sheet42.xml")]:
            with ZipFile(ROOT / BOOKS[key][0]) as z:
                nodes = ET.fromstring(z.read(part)).findall("m:sheetData/m:row/m:c", main_ns)
                for node in nodes:
                    value = node.find("m:v", main_ns)
                    if node.get("t", "n") == "n" and value is not None and value.text is not None:
                        source_numbers[(block, node.attrib["r"])] = Decimal(value.text)
        for block, address, value in con.execute("SELECT block,source_cell,value FROM observations").fetchall():
            require(source_numbers.get((block, address)) == value, f"Exact original XML numeric mismatch: {block}/{address}")
        source_rows = con.execute("SELECT * FROM sources ORDER BY book").fetchall()
        require({r[0] for r in source_rows} == set(BOOKS), "Sources changed")
        for key, xlsx, digest, pdf, pdf_digest in source_rows:
            require(xlsx == BOOKS[key][0] and digest == BOOKS[key][1] and sha(ROOT / xlsx) == digest, "Workbook hash mismatch")
            expected_pdf = str(Path(BOOKS[key][0]).parent / BOOKS[key][2]).replace("\\", "/")
            require(pdf == expected_pdf and pdf_digest == PDF_HASHES[key] and sha(ROOT / pdf) == pdf_digest, "PDF provenance mismatch")
        parts = con.execute("SELECT * FROM source_parts").fetchall()
        expected_parts = {(key, part) for key in BOOKS for part in ["xl/sharedStrings.xml", "xl/styles.xml", "xl/workbook.xml"]}
        expected_parts.update(("2025", f"xl/worksheets/sheet{n}.xml") for n in [2, 12])
        expected_parts.update(("2026", f"xl/worksheets/sheet{n}.xml") for n in [1, 2, 12, 42])
        expected_parts.add(("2026", "xl/charts/chart28.xml"))
        require({(key, part) for key, part, _ in parts} == expected_parts, "Source XML inventory mismatch")
        for key, part, xml in parts:
            with ZipFile(ROOT / BOOKS[key][0]) as z:
                require(xml == z.read(part).decode("utf-8"), "Preserved source XML mismatch")
        bindings = con.execute("SELECT * FROM chart_bindings").fetchall()
        with ZipFile(ROOT / BOOKS["2026"][0]) as z:
            namespace = {"c": "http://schemas.openxmlformats.org/drawingml/2006/chart"}
            nodes = ET.fromstring(z.read("xl/charts/chart28.xml")).findall(".//c:ser", namespace)
            original_bindings = [("xl/charts/chart28.xml", *(n.text for n in series.findall(".//c:f", namespace))) for series in nodes]
        require(len(bindings) == 3 and set(bindings) == set(original_bindings), "Chart binding mismatch")
    examples = []
    for series, expected_display, cell in [("Opinión 2025", "45.0", "R8"), ("AIReF 2026", "45.7", "R7")]:
        result = lookup(db, block="grafico20_2026", metric=labels[0], series=series, year=2030)
        require(result["display_value"] == expected_display and result["source_cell"] == cell, "Known series counterexample failed")
        examples.append(result)
    for block, series, metric, year, display in [
        ("cuadro9_2025", "Opinión 2025", "Pensiones", 2041, "15.3"),
        ("cuadro9_2025", "Opinión 2025", "Intereses", 2041, "3.7"),
        ("cuadro9_2025", "Opinión 2025", "Cuidados de larga duración", 2050, "1.4"),
        ("cuadro12_2026", "AIReF 2026", "Intereses", 2050, "4.4"),
    ]:
        result = lookup(db, block=block, metric=metric, series=series, year=year)
        require(result["display_value"] == display, "Known table counterexample failed")
        examples.append(result)
    computed = difference(db, block="cuadro9_2025", metric="Intereses", series="Opinión 2025", start_year=2023, end_year=2050)
    require(computed["display_value"] == "2.3", "Deterministic difference failed")
    negative_cases = [
        dict(block="grafico20_2026", metric=labels[0], series="Opinión 2025", year=2022),
        dict(block="grafico20_2026", metric=labels[0], series="Histórico", year=2030),
        dict(block="grafico20_2026", metric=labels[0], series="Opinión 2024", year=2030),
        dict(block="cuadro9_2025", metric="Intereses", series="Opinión 2025", year=2030),
        dict(block="cuadro12_2026", metric="Intereses", series="Opinión 2025", year=2030),
        dict(block="cuadro12_2026", metric="Intereses en Cataluña", series="AIReF 2026", year=2030),
    ]
    for kwargs in negative_cases:
        try:
            lookup(db, **kwargs)
        except LookupError:
            pass
        else:
            raise ValueError(f"Unsafe fallback: {kwargs}")
    for book in books.values():
        for workbook in book:
            workbook.close()
    return dict(status="pass", observations_checked=182, raw_cells_checked=raw_checks,
                exact_xml_numeric_matches=len(exact_values), independent_reader="openpyxl " + openpyxl.__version__,
                independent_numeric_tolerance="1e-11", source_hashes_unchanged=True,
                safe_missing_queries=len(negative_cases), example_queries=examples, computed_example=computed,
                db_sha256=sha(db), builder_sha256=sha(ROOT / "experiments/excel_pilot.py"),
                validator_sha256=sha(Path(__file__)), rag_integrated=False,
                limits=["Three explicit source blocks only; not a general Excel importer",
                        "Structured queries only; natural-language routing not implemented",
                        "Cached formulas checked but not recalculated by Excel",
                        "PDF page labels from workbook indexes; physical PDF locations not validated"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("Output exists; no overwrite")
    try:
        report = validate(args.db)
    except Exception as exc:
        report = dict(status="fail", error=str(exc), rag_integrated=False)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in ("example_queries", "computed_example")}, ensure_ascii=False, indent=2))
    return int(report["status"] != "pass")


if __name__ == "__main__":
    raise SystemExit(main())
