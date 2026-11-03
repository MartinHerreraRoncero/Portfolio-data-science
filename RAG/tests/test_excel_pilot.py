"""Regression for source attribution, precision and failure-closed pilot extraction."""
from dataclasses import replace
from decimal import Decimal
from pathlib import Path
import importlib.util
import shutil
import tempfile
import unittest
from unittest.mock import patch

if not all(importlib.util.find_spec(name) for name in ("duckdb", "openpyxl")):
    raise unittest.SkipTest("Optional pilot dependencies: requirements-excel-pilot.txt")

import duckdb

from experiments.excel_pilot import (ROOT, BOOKS, Cell, ExcelXML, build, difference,
                                    extract_block, lookup, translate_shared)
from experiments.validate_excel_pilot import validate


class TestExcelPilot(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        (ROOT / "tmp").mkdir(exist_ok=True)
        cls.temp = tempfile.TemporaryDirectory(dir=ROOT / "tmp")
        cls.output = Path(cls.temp.name) / "build"
        cls.manifest = build(cls.output)
        cls.db = cls.output / "pilot.duckdb"
        cls.cells = ExcelXML(ROOT / BOOKS["2025"][0]).read("Cuadro 9 ")

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def query(self, **kwargs):
        args = dict(block="grafico20_2026", metric="Empleos no Financieros", series="Opinión 2025", year=2030)
        args.update(kwargs)
        return lookup(self.db, **args)

    def test_complete_independent_validation(self):
        report = validate(self.db)
        self.assertEqual(report["observations_checked"], 182)
        self.assertEqual(report["exact_xml_numeric_matches"], 182)
        self.assertEqual(report["status"], "pass")

    def test_series_identity_counterexample(self):
        previous, current = self.query(), self.query(series="AIReF 2026")
        self.assertEqual((previous["display_value"], previous["source_cell"]), ("45.0", "R8"))
        self.assertEqual((current["display_value"], current["source_cell"]), ("45.7", "R7"))
        self.assertNotEqual(previous["value"], current["value"])

    def test_exact_source_precision_and_provenance(self):
        result = self.query()
        self.assertEqual(Decimal(result["value"]), Decimal(result["raw_value"]))
        self.assertEqual(result["sheet"], "Gráfico 20")
        self.assertEqual(len(result["xlsx_sha256"]), 64)
        self.assertEqual(result["page_basis"], "excel_index_not_physical_page_verified")

    def test_absent_series_year_scope_do_not_fallback(self):
        for args in [dict(series="Histórico"), dict(series="Opinión 2024"), dict(year=2022),
                     dict(metric="Empleos no Financieros Cataluña"), dict(series="' OR 1=1 --")]:
            with self.subTest(args=args), self.assertRaises(LookupError):
                self.query(**args)

    def test_missing_series_and_wrong_year_type(self):
        for args in [dict(series=""), dict(year=True), dict(year=2030.0), dict(year="2030")]:
            with self.subTest(args=args), self.assertRaises(ValueError):
                self.query(**args)

    def test_preserves_series_overlap(self):
        historical = self.query(series="Histórico", year=2025)
        current = self.query(series="AIReF 2026", year=2025)
        self.assertEqual(historical["value"], current["value"])
        self.assertNotEqual(historical["series"], current["series"])

    def test_concept_alias(self):
        result = self.query(block="cuadro9_2025", metric="Cuidados de larga duración", year=2050)
        self.assertEqual(result["display_value"], "1.4")

    def test_levels_not_interpolated(self):
        with self.assertRaises(LookupError):
            self.query(block="cuadro9_2025", metric="Intereses", year=2030)

    def test_change_kinds_and_units(self):
        published = lookup(self.db, block="cuadro9_2025", metric="Pensiones", series="Opinión 2025",
                           kind="published_change", start_year=2050, end_year=2070)
        self.assertEqual(published["unit"], "percentage_points_gdp")
        self.assertEqual(published["display_value"], "-1.4")
        self.assertEqual(published["formula"], "+H9-G9")
        computed = difference(self.db, block="cuadro9_2025", metric="Intereses", series="Opinión 2025",
                              start_year=2023, end_year=2050)
        self.assertEqual(computed["display_value"], "2.3")
        self.assertEqual(computed["kind"], "computed_change")
        self.assertEqual(len(computed["inputs"]), 2)

    def test_invalid_intervals(self):
        for a, b in [(2050, 2023), (2023, 2023), (True, 2050), (2023, None)]:
            with self.subTest(a=a, b=b), self.assertRaises(ValueError):
                difference(self.db, block="cuadro9_2025", metric="Intereses", series="Opinión 2025", start_year=a, end_year=b)

    def test_no_overwrite(self):
        with self.assertRaises(FileExistsError):
            build(self.output)

    def test_reject_changed_source_hash(self):
        with patch("experiments.excel_pilot.sha", return_value="0" * 64), self.assertRaisesRegex(ValueError, "version changed"):
            build(Path(self.temp.name) / "bad")

    def test_changed_headers_labels_and_cached_formula(self):
        for address, changes in [("D6", dict(value="2024")), ("B9", dict(value="Intereses")),
                                 ("J9", dict(value="999")), ("J9", dict(formula="+D9-E9"))]:
            cells = dict(self.cells)
            cells[address] = replace(cells[address], **changes)
            with self.subTest(address=address, changes=changes), self.assertRaises(ValueError):
                extract_block("cuadro9_2025", cells)

    def test_numeric_missing_error_and_zero(self):
        for kind, value in [("e", "#N/A"), ("n", None), ("n", "NaN"), ("n", "Infinity"), ("n", "1e-25")]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                Cell("A1", kind, value, None, "0").number()
        self.assertEqual(Cell("A1", "n", "0", None, "0").number(), Decimal(0))

    def test_shared_formula_translation(self):
        self.assertEqual(translate_shared("+E7-D7", "J7", "J15"), "+E15-D15")
        self.assertEqual(translate_shared("Z5+1", "AA5", "AB5"), "AA5+1")
        with self.assertRaises(ValueError):
            translate_shared('INDIRECT("A1")', "B1", "B2")

    def test_validation_detects_tampered_series(self):
        copied = Path(self.temp.name) / "tampered.duckdb"
        shutil.copyfile(self.db, copied)
        with duckdb.connect(str(copied)) as con:
            con.execute("UPDATE observations SET series='AIReF 2026' WHERE block='grafico20_2026' AND source_cell='R8'")
        with self.assertRaisesRegex(ValueError, "Dimension mismatch"):
            validate(copied)

    def test_validation_detects_tampered_provenance(self):
        cases = [
            ("UPDATE blocks SET sheet='Wrong sheet' WHERE block='grafico20_2026'", "Block metadata"),
            ("UPDATE blocks SET index_page=999 WHERE block='grafico20_2026'", "Block metadata"),
            ("DELETE FROM source_parts WHERE part='xl/styles.xml'", "XML inventory"),
            ("UPDATE chart_bindings SET values_ref=years_ref", "Chart binding"),
            ("UPDATE sources SET pdf_sha256='wrong' WHERE book='2026'", "PDF provenance"),
        ]
        for i, (sql, error) in enumerate(cases):
            with self.subTest(sql=sql):
                copied = Path(self.temp.name) / f"provenance{i}.duckdb"
                shutil.copyfile(self.db, copied)
                with duckdb.connect(str(copied)) as con:
                    con.execute(sql)
                with self.assertRaisesRegex(ValueError, error):
                    validate(copied)

    def test_builder_rejects_replaced_pdf(self):
        from experiments.excel_pilot import sha

        def changed_pdf(path):
            return "0" * 64 if path.suffix == ".pdf" else sha(path)

        with patch("experiments.excel_pilot.sha", side_effect=changed_pdf), self.assertRaisesRegex(ValueError, "PDF version changed"):
            build(Path(self.temp.name) / "changed_pdf")

    def test_sub_tolerance_coordinated_numeric_tampering(self):
        copied = Path(self.temp.name) / "tiny_tampering.duckdb"
        shutil.copyfile(self.db, copied)
        with duckdb.connect(str(copied)) as con:
            con.execute("UPDATE observations SET value=value+0.000000000001 WHERE block='grafico20_2026' AND source_cell='R8'")
            con.execute("UPDATE raw_cells SET raw_value='45.012911959783521' WHERE block='grafico20_2026' AND address='R8'")
        with self.assertRaisesRegex(ValueError, "Exact original XML numeric mismatch"):
            validate(copied)


if __name__ == "__main__":
    unittest.main()
