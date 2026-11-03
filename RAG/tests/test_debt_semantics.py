"""Debt/balance semantic identity and source reconciliation integration tests."""
import importlib.util
from decimal import Decimal
from pathlib import Path
import re
import shutil
import tempfile
import unittest
if not all(importlib.util.find_spec(n) for n in ['duckdb','openpyxl']):
    raise unittest.SkipTest('Optional Excel dependencies required')
import duckdb
import openpyxl
from experiments.debt_semantics import ROOT,build,query,validate
from experiments.excel_pilot import BOOKS

DB=ROOT/'evaluation/debt_semantics_2026-10-31/run_v1/reviewed.duckdb'


class TestDebtSemantics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not DB.exists():raise unittest.SkipTest('Build debt integration fixture first')
        cls.temp=tempfile.TemporaryDirectory(dir=ROOT/'tmp')
    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def test_all_336_against_independent_original_reader(self):
        workbook=openpyxl.load_workbook(ROOT/BOOKS['2026'][0],read_only=True,data_only=True)
        with duckdb.connect(str(DB),read_only=True) as con:
            rows=con.execute('SELECT block,series,year,unit,value,source_cell,source_series_id FROM debt_observations').fetchall()
        self.assertEqual(len(rows),336)
        sheets={s:{c.coordinate:c.value for row in workbook[s].iter_rows() for c in row if c.value is not None} for s in {r[0] for r in rows}}
        for block,series,year,unit,value,cell,series_id in rows:
            cells=sheets[block];col,row=re.fullmatch(r'([A-Z]+)(\d+)',cell).groups()
            header=5 if block=='Gráfico 2' else 6
            self.assertEqual(series,cells[f'{col}{header}'])
            self.assertEqual(year,cells[f'B{row}'])
            self.assertEqual(unit,'percent_gdp')
            self.assertLessEqual(abs(value-Decimal(str(cells[cell]))),Decimal('1e-11'))
            self.assertTrue(series_id.endswith('_s1') or series_id.endswith('_s2'))
        workbook.close()

    def q(self,**kwargs):
        args=dict(book='2026',block='Gráfico 3',metric='Saldo público',series='Saldo total',year=1995)
        args.update(kwargs);return query(DB,**args)

    def test_total_vs_primary_and_sign(self):
        total=self.q();primary=self.q(series='Saldo primario')
        self.assertEqual(total['display_value'],'-6.8')
        self.assertEqual(primary['display_value'],'-1.9')
        self.assertEqual(total['source_cell'],'D7')
        self.assertEqual(primary['source_cell'],'C7')

    def test_complete_series_not_duplicated_by_partial_trace(self):
        result=self.q()
        self.assertEqual(result['source_series_id'],'chart3_s2')
        with duckdb.connect(str(DB),read_only=True) as con:
            self.assertEqual(con.execute("SELECT count(*) FROM admission_review_queue WHERE book='2026' AND bloque IN ('Gráfico 2','Gráfico 3','Gráfico 29','Gráfico 32')").fetchone()[0],186)

    def test_unknown_scope_year_and_vintage_rejected(self):
        for args in [dict(series='AIReF 2026'),dict(series='Histórico'),dict(metric='Saldo Cataluña'),dict(year=2051)]:
            with self.subTest(args=args),self.assertRaises(LookupError):self.q(**args)
        with self.assertRaises(ValueError):self.q(year=True)

    def test_previous560_and_queue_unchanged_contract(self):
        with duckdb.connect(str(DB),read_only=True) as con:
            self.assertEqual(con.execute('SELECT count(*) FROM family_observations').fetchone()[0],560)
            self.assertEqual(con.execute('SELECT count(*) FROM admitted_observations').fetchone()[0],896)
            self.assertEqual(con.execute('SELECT count(*) FROM admission_review_queue').fetchone()[0],63753)

    def test_no_overwrite(self):
        with self.assertRaisesRegex(ValueError,'Output exists'):build(DB.parent)

    def test_duplication_rejected(self):
        path=Path(self.temp.name)/'duplicate.duckdb';shutil.copyfile(DB,path)
        with duckdb.connect(str(path)) as con:
            con.execute('CREATE OR REPLACE VIEW admitted_observations AS SELECT * FROM family_observations UNION ALL SELECT * FROM debt_observations UNION ALL SELECT * FROM debt_observations')
        with self.assertRaisesRegex(ValueError,'multiplicity'):validate(path)

    def test_tiny_value_change_rejected(self):
        path=Path(self.temp.name)/'tiny.duckdb';shutil.copyfile(DB,path)
        with duckdb.connect(str(path)) as con:
            con.execute("UPDATE debt_observations SET value=value+0.000000000001 WHERE block='Gráfico 3' AND source_cell='D7'")
        with self.assertRaisesRegex(ValueError,'Debt semantic'):validate(path)


if __name__=='__main__':unittest.main()
