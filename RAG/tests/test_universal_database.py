"""Source reconciliation and admission boundary regression for the universal DB."""
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest

if not all(importlib.util.find_spec(n) for n in ("duckdb", "openpyxl")):
    raise unittest.SkipTest("Optional Excel dependencies required")

import duckdb
from experiments.universal_database import ROOT, HEADERS, build, query, reconcile, validate

DB = ROOT / "evaluation/universal_database_2026-10-31/run_v3/universal.duckdb"


class TestUniversalDatabase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not DB.exists():
            raise unittest.SkipTest("Build the universal database integration fixture first")
        cls.temp = tempfile.TemporaryDirectory(dir=ROOT / "tmp")
        with duckdb.connect(str(DB), read_only=True) as con:
            cursor = con.execute("SELECT o.*,b.book,b.sheet FROM observations o JOIN blocks b USING(block) WHERE block='grafico20_2026' AND source_cell='R8'")
            cls.approved = dict(zip([c[0] for c in cursor.description], cursor.fetchone()))
            cursor = con.execute("SELECT * FROM staging_records WHERE book='2026' AND bloque='Gráfico 20' AND celda_valor='R8'")
            cls.record = dict(zip([c[0] for c in cursor.description], cursor.fetchone()))

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_exact_admission_mapping(self):
        reconcile(self.record, self.approved)

    def test_reject_semantic_mutations(self):
        for field, value in [("serie_literal", "AIReF 2026"), ("periodo_categoria_literal", "2031"),
                             ("decimal_XML_texto", "45.012911959783521"), ("contexto_unidad_titulo", "%"),
                             ("hoja_valor", "Gráfico 21"), ("celda_valor", "R7"),
                             ("tipo", "metadato"), ("parte_grafico_imagen", "xl/charts/chart29.xml")]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                reconcile(dict(self.record, **{field: value}), self.approved)

    def test_distinct_series_and_unadmitted_scope(self):
        for series, expected in [("Opinión 2025", "45.0"), ("AIReF 2026", "45.7")]:
            result = query(DB, block="grafico20_2026", metric="Empleos no Financieros", series=series, year=2030)
            self.assertEqual(result["display_value"], expected)
            self.assertEqual(result["universal_book"], "2026")
        with self.assertRaises(ValueError):
            query(DB, block="grafico21_2026", metric="Intereses", series="AIReF 2026", year=2030)
        with self.assertRaises(LookupError):
            query(DB, block="grafico20_2026", metric="Empleos no Financieros", series="Histórico", year=2030)

    def test_partition_counts(self):
        with duckdb.connect(str(DB), read_only=True) as con:
            self.assertEqual(con.execute("SELECT count(*) FROM staging_records").fetchone()[0], 64649)
            self.assertEqual(con.execute("SELECT count(*) FROM validated_observations").fetchone()[0], 182)
            self.assertEqual(con.execute("SELECT count(*) FROM review_queue").fetchone()[0], 64467)

    def test_no_overwrite(self):
        with self.assertRaisesRegex(ValueError, "Output exists"):
            build(DB.parent)

    def test_source_snapshot_tampering(self):
        path = Path(self.temp.name) / "wrong_source.duckdb"
        shutil.copyfile(DB, path)
        with duckdb.connect(str(path)) as con:
            con.execute("UPDATE workbook_files SET sha256='wrong' WHERE book='2025'")
        with self.assertRaisesRegex(ValueError, "Source identity mismatch"):
            validate(path)

    def test_tampered_view_fails_even_with_same_count(self):
        statements = [
            "CREATE OR REPLACE VIEW validated_observations AS SELECT o.*,a.book,a.row_number,a.record_id,'false_admission' AS semantic_status FROM observations o JOIN admissions a USING(block,source_cell)",
            "CREATE OR REPLACE VIEW review_queue AS SELECT s.* REPLACE('false_revision' AS revision) FROM staging_records s WHERE NOT EXISTS(SELECT 1 FROM admissions a WHERE a.book=s.book AND a.row_number=s.row_number)",
        ]
        for i, sql in enumerate(statements):
            with self.subTest(view=i):
                path = Path(self.temp.name) / f"tampered{i}.duckdb"
                shutil.copyfile(DB, path)
                with duckdb.connect(str(path)) as con:
                    con.execute(sql)
                with self.assertRaisesRegex(ValueError, "View content mismatch"):
                    validate(path)

    def test_reject_added_empty_columns(self):
        path = Path(self.temp.name) / "padded.duckdb"
        shutil.copyfile(DB, path)
        with duckdb.connect(str(path)) as con:
            con.execute("UPDATE sheet_inventory SET cols=cols+1 WHERE book='2025' AND sheet='Control'")
        with self.assertRaisesRegex(ValueError, "Original sheet dimension mismatch"):
            validate(path)


if __name__ == "__main__":
    unittest.main()
