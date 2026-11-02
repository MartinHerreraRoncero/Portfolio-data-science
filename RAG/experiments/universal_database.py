"""Complete universal workbook archive and deliberately restricted semantic view."""
from __future__ import annotations

import argparse
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET
from zipfile import ZipFile

import duckdb
import openpyxl

from experiments.excel_pilot import ROOT, BOOKS, lookup as pilot_lookup, normalized, sha
from experiments.validate_excel_pilot import validate as validate_pilot

UNIVERSAL = {
    "2025": (str(Path(BOOKS["2025"][0]).parent / "AIReF_2025_universal_completo.xlsx"),
             "0eb73379454fe4d46020f33403b62c0e66ed1d2e0345c0f1ed25dd33f2225a91"),
    "2026": (str(Path(BOOKS["2026"][0]).parent / "AIReF_2026_universal_completo.xlsx"),
             "efef1e8c1bdd670b505ef965d436b03cbaeef6609213e75022c28b16ef1d0800"),
}
HEADERS = ("id documento bloque titulo metrica_literal serie_id serie_literal periodo_categoria_literal unidad valor "
           "decimal_XML_texto estado_dato hoja_valor celda_valor parte_grafico_imagen tipo grupo_detalle "
           "formula_texto contexto_unidad_titulo titulo_eje_literal revision").split()
PILOT = ROOT / "evaluation/excel_pilot_2026-10-31/run_v2/pilot.duckdb"
PILOT_SHA = "d581a53964128aaea3c029dccd633a103a4f7f0a8f35e0e0efad1f4b4c47bc8a"
NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def dumps(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def reconcile(row, approved):
    """Validate the explicit semantic mapping; never promote by numeric coincidence."""
    require(row["hoja_valor"] == approved["sheet"] and row["bloque"] == approved["sheet"], "Block/source sheet mismatch")
    require(row["celda_valor"] == approved["source_cell"], "Source cell mismatch")
    require(row["decimal_XML_texto"] is not None and Decimal(row["decimal_XML_texto"]) == approved["value"], "Exact value mismatch")
    series = row["serie_literal"]
    if approved["block"] == "cuadro9_2025" and series == "Opinion 2025":
        series = "Opinión 2025"  # reviewed, explicit spelling alias
    require(series == approved["series"], "Series mismatch")
    if approved["block"] == "grafico20_2026":
        require(row["tipo"] == "posicion_serie" and row["estado_dato"] == "dato", "Graph value type mismatch")
        require(row["contexto_unidad_titulo"] == "% PIB", "Graph unit context mismatch")
        require(normalized(row["metrica_literal"]) == normalized("Empleos no financieros sin PRTR (% PIB)"), "Graph metric mismatch")
        require(row["parte_grafico_imagen"] == "xl/charts/chart28.xml", "Graph identity mismatch")
    else:
        require(normalized(row["metrica_literal"]) == normalized(approved["metric"]), "Metric mismatch")
        require(row["estado_dato"] == "dato_numerico", "Table numeric state mismatch")
        require(row["tipo"] == ("nivel" if approved["kind"] == "level" else "variacion_publicada"), "Table kind mismatch")
        require(row["unidad"] == ("% PIB" if approved["kind"] == "level" else "puntos PIB"), "Table unit mismatch")
        require(row["formula_texto"] == ("=" + approved["formula"] if approved["formula"] else None), "Formula provenance mismatch")
    period = str(approved["year"]) if approved["kind"] == "level" else f"{approved['end_year']}-{approved['start_year']}"
    # Universal tables keep the label as printed; 2026 interval headings run start–end.
    if approved["block"] == "cuadro12_2026" and approved["kind"] != "level":
        period = f"{approved['start_year']}-{approved['end_year']}"
    elif approved["block"] == "cuadro9_2025" and approved["kind"] != "level":
        period = "Δ " + period
    require(row["periodo_categoria_literal"] == period, "Period mismatch")


def build(output: Path):
    output = output.resolve()
    output.relative_to(ROOT)
    require(not output.exists(), "Output exists; no overwrite")
    require(sha(PILOT) == PILOT_SHA, "Pinned pilot changed")
    validate_pilot(PILOT)
    for path, digest in UNIVERSAL.values():
        require(sha(ROOT / path) == digest, "Universal workbook version changed")
    output.mkdir(parents=True)
    db = output / "universal.duckdb"
    inventory = []
    # Bulk load avoids a separate transaction for every archived row.
    with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as temp:
        archive = Path(temp) / "rows.jsonl"
        with archive.open("w", encoding="utf-8") as stream:
            for book, (name, digest) in UNIVERSAL.items():
                workbook = openpyxl.load_workbook(ROOT / name, read_only=True, data_only=False)
                for sheet in workbook:
                    count = 0
                    for number, row in enumerate(sheet.iter_rows(values_only=True), 1):
                        stream.write(dumps(dict(book=book, sheet=sheet.title, row_number=number, cells_json=dumps(row))) + "\n")
                        count += 1
                    inventory.append((book, sheet.title, count, sheet.max_column))
                workbook.close()
        with duckdb.connect(str(db)) as con:
            con.execute("BEGIN")
            con.execute("CREATE TABLE workbook_files(book VARCHAR PRIMARY KEY,path VARCHAR,sha256 VARCHAR,content BLOB)")
            for book, (name, digest) in UNIVERSAL.items():
                con.execute("INSERT INTO workbook_files VALUES (?,?,?,?)", [book, name, digest, (ROOT / name).read_bytes()])
            con.execute("CREATE TABLE sheet_inventory(book VARCHAR,sheet VARCHAR,rows INTEGER,cols INTEGER,PRIMARY KEY(book,sheet))")
            con.executemany("INSERT INTO sheet_inventory VALUES (?,?,?,?)", inventory)
            con.execute("""CREATE TABLE workbook_rows AS SELECT * FROM read_json(?, columns={
                book:'VARCHAR',sheet:'VARCHAR',row_number:'INTEGER',cells_json:'VARCHAR'},format='newline_delimited')""", [str(archive)])
            con.execute("ALTER TABLE workbook_rows ADD PRIMARY KEY(book,sheet,row_number)")
            columns = ",".join(f"json_extract_string(cells_json,'$[{i}]') AS \"{h}\"" for i, h in enumerate(HEADERS))
            con.execute(f"CREATE VIEW staging_records AS SELECT book,row_number,{columns} FROM workbook_rows WHERE sheet='Datos_universales' AND row_number>1")
            for book in UNIVERSAL:
                header = con.execute("SELECT cells_json FROM workbook_rows WHERE book=? AND sheet='Datos_universales' AND row_number=1", [book]).fetchone()
                require(json.loads(header[0]) == HEADERS, "Universal header mismatch")
            require(not con.execute("SELECT 1 FROM staging_records GROUP BY book,id HAVING count(*)>1 LIMIT 1").fetchone(), "Duplicate source record id")
            # Preserve canonical pilot provenance without changing its validated API.
            with duckdb.connect(str(PILOT), read_only=True) as pilot:
                for table in ["sources", "blocks", "raw_cells", "observations"]:
                    sql = pilot.execute("SELECT sql FROM duckdb_tables() WHERE table_name=?", [table]).fetchone()[0]
                    # Cross-table foreign keys are recreated as local pilot tables in dependency order.
                    con.execute(sql)
                    rows = pilot.execute(f"SELECT * FROM {table}").fetchall()
                    con.executemany(f"INSERT INTO {table} VALUES ({','.join('?' for _ in rows[0])})", rows)
            con.execute("CREATE TABLE admissions(block VARCHAR,source_cell VARCHAR,book VARCHAR,row_number INTEGER,record_id VARCHAR,source_row_sha256 VARCHAR,PRIMARY KEY(block,source_cell),UNIQUE(book,row_number))")
            cursor = con.execute("SELECT o.*,b.book,b.sheet FROM observations o JOIN blocks b USING(block)")
            names = [c[0] for c in cursor.description]
            for values in cursor.fetchall():
                approved = dict(zip(names, values))
                match = con.execute("SELECT * FROM staging_records WHERE book=? AND hoja_valor=? AND celda_valor=? AND bloque=?",
                                    [approved["book"], approved["sheet"], approved["source_cell"], approved["sheet"]])
                records = match.fetchall()
                require(len(records) == 1, "Missing/ambiguous universal pilot correspondence")
                row = dict(zip([c[0] for c in match.description], records[0]))
                reconcile(row, approved)
                payload = con.execute("SELECT cells_json FROM workbook_rows WHERE book=? AND sheet='Datos_universales' AND row_number=?", [row["book"], row["row_number"]]).fetchone()[0]
                con.execute("INSERT INTO admissions VALUES (?,?,?,?,?,?)", [approved["block"], approved["source_cell"], row["book"], row["row_number"], row["id"], hashlib.sha256(payload.encode()).hexdigest()])
            con.execute("""CREATE VIEW validated_observations AS SELECT o.*,a.book,a.row_number,a.record_id,
                'validated_pilot_v1' AS semantic_status FROM observations o JOIN admissions a USING(block,source_cell)""")
            con.execute("""CREATE VIEW review_queue AS SELECT s.* FROM staging_records s
                WHERE NOT EXISTS(SELECT 1 FROM admissions a WHERE a.book=s.book AND a.row_number=s.row_number)""")
            con.execute("COMMIT")
    for name, digest in UNIVERSAL.values():
        require(sha(ROOT / name) == digest, "Source changed during import")
    manifest = dict(status="built_pending_validation", book_rows={book: sum(n for b, _, n, _ in inventory if b == book) for book in UNIVERSAL},
                    source_sha256={b: d for b, (_, d) in UNIVERSAL.items()}, pilot_sha256=PILOT_SHA,
                    db_sha256=sha(db), builder_sha256=sha(Path(__file__)), rag_integrated=False)
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def xml_rows(content):
    """Independent OOXML reader for checking every archived row, not openpyxl."""
    from io import BytesIO
    with ZipFile(BytesIO(content)) as z:
        strings = []
        if "xl/sharedStrings.xml" in z.namelist():
            strings = ["".join(t.text or "" for t in item.findall(".//m:t", NS)) for item in ET.fromstring(z.read("xl/sharedStrings.xml")).findall("m:si", NS)]
        rel = {n.get("Id"): n.get("Target") for n in ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))}
        for sheet in ET.fromstring(z.read("xl/workbook.xml")).findall("m:sheets/m:sheet", NS):
            target = rel[sheet.attrib["{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"]]
            path = target.lstrip("/") if target.startswith("/") else "xl/" + target
            root = ET.fromstring(z.read(path))
            dimension = root.find("m:dimension", NS).get("ref").split(":")[-1]
            width = 0
            for letter in ''.join(c for c in dimension if c.isalpha()):
                width = width * 26 + ord(letter) - 64
            height = int(''.join(c for c in dimension if c.isdigit()))
            sparse = {}
            shared_formulas = {}
            for row in root.findall("m:sheetData/m:row", NS):
                values = {}
                for cell in row.findall("m:c", NS):
                    col = 0
                    for letter in ''.join(c for c in cell.get("r") if c.isalpha()):
                        col = col * 26 + ord(letter) - 64
                    value, formula = cell.find("m:v", NS), cell.find("m:f", NS)
                    text = value.text if value is not None else None
                    kind = cell.get("t", "n")
                    if formula is not None:
                        if formula.get("t") == "shared":
                            key = formula.get("si")
                            if formula.text:
                                shared_formulas[key] = (cell.get("r"), "=" + formula.text)
                            anchor, expression = shared_formulas[key]
                            from openpyxl.formula.translate import Translator
                            item = Translator(expression, origin=anchor).translate_formula(cell.get("r"))
                        else:
                            item = "=" + (formula.text or "")
                    elif kind == "s":
                        item = strings[int(text)] if text is not None else None
                    elif kind == "inlineStr":
                        nodes = cell.findall("m:is//m:t", NS)
                        item = ''.join(t.text or '' for t in nodes) if nodes else None
                    elif text is None:
                        item = None
                    elif kind == "b":
                        item = bool(int(text))
                    elif kind == "n":
                        item = float(text) if any(c in text for c in '.eE') else int(text)
                    else:
                        item = text
                    values[col] = item
                sparse[int(row.get("r"))] = values
            yield sheet.get("name"), sparse, (height, width)


def validate(db: Path):
    require(sha(PILOT) == PILOT_SHA, "Pilot changed")
    counts, checked = {}, 0
    with duckdb.connect(str(db), read_only=True) as con:
        staging_columns = ",".join(f"json_extract_string(cells_json,'$[{i}]') AS \"{h}\"" for i, h in enumerate(HEADERS))
        expected_staging = f"SELECT book,row_number,{staging_columns} FROM workbook_rows WHERE sheet='Datos_universales' AND row_number>1"
        expected_admitted = "SELECT o.*,a.book,a.row_number,a.record_id,'validated_pilot_v1' AS semantic_status FROM observations o JOIN admissions a USING(block,source_cell)"
        expected_queue = "SELECT s.* FROM staging_records s WHERE NOT EXISTS(SELECT 1 FROM admissions a WHERE a.book=s.book AND a.row_number=s.row_number)"
        for view, expected in [("staging_records", expected_staging), ("validated_observations", expected_admitted), ("review_queue", expected_queue)]:
            columns = [c[0] for c in con.execute(f"SELECT * FROM {view} LIMIT 0").description]
            expected_columns = [c[0] for c in con.execute(expected + " LIMIT 0").description]
            require(columns == expected_columns, f"View columns mismatch: {view}")
            require(not con.execute(f"(SELECT * FROM {view} EXCEPT ALL ({expected})) UNION ALL (({expected}) EXCEPT ALL SELECT * FROM {view}) LIMIT 1").fetchone(), f"View content mismatch: {view}")
        sources = con.execute("SELECT * FROM workbook_files ORDER BY book").fetchall()
        require({s[0] for s in sources} == set(UNIVERSAL), "Source inventory mismatch")
        expected_inventory = set()
        for book, path, digest, content in sources:
            require(path == UNIVERSAL[book][0] and digest == UNIVERSAL[book][1], "Source identity mismatch")
            require(hashlib.sha256(content).hexdigest() == digest == sha(ROOT / path), "Source bytes/hash mismatch")
            for sheet, sparse, original_dimensions in xml_rows(content):
                rows = con.execute("SELECT row_number,cells_json FROM workbook_rows WHERE book=? AND sheet=? ORDER BY row_number", [book, sheet]).fetchall()
                inventory = con.execute("SELECT rows,cols FROM sheet_inventory WHERE book=? AND sheet=?", [book, sheet]).fetchone()
                require(inventory == original_dimensions, "Original sheet dimension mismatch")
                require(inventory is not None and len(rows) == inventory[0], "Sheet row count mismatch")
                require([r[0] for r in rows] == list(range(1, inventory[0]+1)), "Row inventory mismatch")
                require(max(sparse, default=0) <= inventory[0], "Missing original rows")
                for n, payload in rows:
                    expected = [sparse.get(n, {}).get(c) for c in range(1, inventory[1]+1)]
                    require(max(sparse.get(n, {}), default=0) <= inventory[1], "Missing original columns")
                    require(json.loads(payload) == expected, f"Archived row mismatch: {book}/{sheet}/{n}")
                    checked += 1
                expected_inventory.add((book, sheet))
        require(set(con.execute("SELECT book,sheet FROM sheet_inventory").fetchall()) == expected_inventory, "Sheet inventory mismatch")
        require(set(con.execute("SELECT DISTINCT book,sheet FROM workbook_rows").fetchall()) == expected_inventory, "Unexpected archived sheet")
        with duckdb.connect(str(PILOT), read_only=True) as pilot:
            for table in ["sources", "blocks", "raw_cells", "observations"]:
                require(set(con.execute(f"SELECT * FROM {table}").fetchall()) == set(pilot.execute(f"SELECT * FROM {table}").fetchall()), f"Pilot {table} mismatch")
        approved = con.execute("SELECT o.*,b.book,b.sheet FROM observations o JOIN blocks b USING(block)")
        names = [c[0] for c in approved.description]
        for values in approved.fetchall():
            item = dict(zip(names, values))
            admission = con.execute("SELECT book,row_number,record_id,source_row_sha256 FROM admissions WHERE block=? AND source_cell=?", [item["block"], item["source_cell"]]).fetchall()
            require(len(admission) == 1 and admission[0][0] == item["book"], "Admission provenance mismatch")
            book, number, record_id, digest = admission[0]
            result = con.execute("SELECT * FROM staging_records WHERE book=? AND row_number=?", [book, number])
            row = dict(zip([c[0] for c in result.description], result.fetchone()))
            require(row["id"] == record_id, "Admission record id mismatch")
            reconcile(row, item)
            payload = con.execute("SELECT cells_json FROM workbook_rows WHERE book=? AND sheet='Datos_universales' AND row_number=?", [book, number]).fetchone()[0]
            require(hashlib.sha256(payload.encode()).hexdigest() == digest, "Admission row hash mismatch")
        require(con.execute("SELECT count(*) FROM admissions").fetchone()[0] == 182, "Admission count mismatch")
        for book in UNIVERSAL:
            total = con.execute("SELECT count(*) FROM staging_records WHERE book=?", [book]).fetchone()[0]
            accepted = con.execute("SELECT count(*) FROM validated_observations WHERE book=?", [book]).fetchone()[0]
            pending = con.execute("SELECT count(*) FROM review_queue WHERE book=?", [book]).fetchone()[0]
            require(total == accepted + pending, "Review coverage mismatch")
            counts[book] = dict(imported=total, admitted=accepted, pending=pending)
    examples = [query(db, block="grafico20_2026", metric="Empleos no Financieros", series=s, year=2030) for s in ["Opinión 2025", "AIReF 2026"]]
    require([e["display_value"] for e in examples] == ["45.0", "45.7"], "Series counterexample failed")
    return dict(status="pass", counts=counts, archived_rows_checked=checked, admitted=182, examples=examples,
                db_sha256=sha(db), code_sha256=sha(Path(__file__)), rag_integrated=False,
                limits=["Only 182 pilot observations semantically admitted", "No natural-language routing", "PDF physical citations pending"])


def query(db: Path, **kwargs):
    # The original API only sees the 182 reconciled observations, never staging.
    result = pilot_lookup(db, **kwargs)
    with duckdb.connect(str(db), read_only=True) as con:
        found = con.execute("SELECT book,row_number,record_id FROM admissions WHERE block=? AND source_cell=?", [result["block"], result["source_cell"]]).fetchall()
    require(len(found) == 1, "Observation is not semantically admitted")
    result["universal_book"], result["universal_row"], result["universal_record_id"] = found[0]
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("build")
    p.add_argument("--output-dir", type=Path, required=True)
    p = sub.add_parser("validate")
    p.add_argument("--db", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "build":
        result = build(args.output_dir)
    else:
        require(not args.output.exists(), "Validation output exists")
        try:
            result = validate(args.db)
        except Exception as exc:
            result = dict(status="fail", error=str(exc))
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return int(result["status"] == "fail")


if __name__ == "__main__":
    raise SystemExit(main())
