"""Exact conversion and axis association regressions, plus independent source reads."""
import copy
from decimal import Decimal,localcontext
import importlib.util
from pathlib import Path
import shutil
import tempfile
import unittest
import xml.etree.ElementTree as ET
if not all(importlib.util.find_spec(n) for n in ['duckdb','openpyxl']):
    raise unittest.SkipTest('Optional Excel dependencies required')
import duckdb
import openpyxl
from experiments.scaled_semantics import ROOT,ScaleReviewer,format_effect,check_new,query,build,C
from experiments.excel_pilot import BOOKS

DB=ROOT/'evaluation/scaled_semantics_2026-10-31/run_v3/scaled.duckdb'


class TestScaledSemantics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not DB.exists():raise unittest.SkipTest('Build scale integration fixture first')
        cls.temp=tempfile.TemporaryDirectory(dir=ROOT/'tmp')
    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def test_formats_percent_literal_and_no_double_scaling(self):
        self.assertEqual(format_effect('0.0%'),(Decimal(1),Decimal(100)))
        self.assertEqual(format_effect('0.0%;-0.0%'),(Decimal(1),Decimal(100)))
        self.assertEqual(format_effect('0.0"%"'),(Decimal(1),Decimal(1)))
        self.assertEqual(format_effect(r'0.0\%'),(Decimal(1),Decimal(1)))
        self.assertEqual(format_effect('#,##0'),(Decimal(1),Decimal(1)))
        self.assertEqual(format_effect('0.0,,'),(Decimal(1000000),Decimal(1)))
        for code in ['0%;0','[>=100]0;0%','0%%']:
            with self.subTest(code=code),self.assertRaises(ValueError):format_effect(code)

    def test_all_raw_values_and_conversions_independent_reader(self):
        with duckdb.connect(str(DB),read_only=True) as con:
            rows=con.execute('SELECT book,source_sheet,source_cell,raw_value,value,divisor,multiplier,display_unit FROM scale_observations').fetchall()
        self.assertEqual(len(rows),3574)
        books={b:openpyxl.load_workbook(ROOT/path,read_only=True,data_only=True) for b,(path,_,_) in BOOKS.items()}
        sheets={}
        for book,sheet,*_ in rows:
            if (book,sheet) not in sheets:
                sheets[book,sheet]={c.coordinate:c.value for r in books[book][sheet].iter_rows() for c in r if c.value is not None}
        for book,sheet,cell,raw,value,divisor,multiplier,display in rows:
            self.assertLessEqual(abs(raw-Decimal(str(sheets[book,sheet][cell]))),Decimal('1e-10'))
            with localcontext() as ctx:
                ctx.prec=60;self.assertEqual(value,raw*multiplier/divisor)
            if display=='thousands':self.assertEqual(divisor,Decimal(1000))
        for book in books.values():book.close()

    def test_real_gdp_divisor_and_raw_preserved(self):
        result=query(DB,book='2026',block='Gráfico 8A',series_id='chart8_s1',category='2022')
        self.assertEqual(Decimal(result['raw_value']),Decimal(1375863))
        self.assertEqual(Decimal(result['value']),Decimal('1375.863'))
        self.assertEqual(result['unit'],'billion_euros')
        self.assertEqual(Decimal(result['divisor']),Decimal(1000))

    def test_percentage_applied_once_with_cell_and_axis_format(self):
        with duckdb.connect(str(DB),read_only=True) as con:
            rows=con.execute('SELECT raw_value,value,multiplier,cell_format,axis_format FROM scale_observations WHERE multiplier=100').fetchall()
        self.assertEqual(len(rows),170)
        for raw,value,multiplier,cell,axis in rows:
            self.assertEqual(value,raw*100)
            self.assertIn('%',axis)

    def test_correct_axis_among_unrelated_secondary_axis(self):
        with duckdb.connect(str(DB),read_only=True) as con:
            c=con.execute("SELECT * FROM staging_records WHERE book='2026' AND bloque='Gráfico 8A' LIMIT 1")
            row=dict(zip([n[0] for n in c.description],c.fetchone()))
        reviewer=ScaleReviewer()
        try:
            part=row['parte_grafico_imagen'];root=ET.fromstring(reviewer.sources['2026'].xml.parts[part])
            axis=copy.deepcopy(root.find('.//c:valAx',C));axis.find('c:axId',C).set('val','999999')
            axis.find('c:dispUnits/c:builtInUnit',C).set('val','millions')
            root.find('.//c:plotArea',C).append(axis)
            reviewer.sources['2026'].xml.parts[part]=ET.tostring(root)
            binding=reviewer.binding(row)
            self.assertEqual(binding['divisor'],Decimal(1000))
            self.assertNotEqual(binding['axis_id'],'999999')
        finally:reviewer.close()

    def test_scope_conflict_and_unknown_percent_stay_pending(self):
        with duckdb.connect(str(DB),read_only=True) as con:
            self.assertEqual(con.execute("SELECT count(*) FROM scale_observations WHERE block='Gráfico 56' AND series_literal='Intereses (% PIB)'").fetchone()[0],0)
            self.assertEqual(con.execute("SELECT count(*) FROM scale_review WHERE status='series_scope_conflict'").fetchone()[0],19)
        with self.assertRaises(LookupError):query(DB,book='2025',block='Gráfico 56',series_id='not_a_series',category='2050')

    def test_prior_count_and_partition(self):
        with duckdb.connect(str(DB),read_only=True) as con:
            self.assertEqual(con.execute('SELECT count(*) FROM usable_keys').fetchone()[0],9992)
            self.assertEqual(con.execute('SELECT count(*) FROM scaled_usable_keys').fetchone()[0],13566)
            self.assertEqual(con.execute('SELECT count(*) FROM scaled_numeric_queue').fetchone()[0],33647)

    def test_no_overwrite(self):
        with self.assertRaisesRegex(ValueError,'Output exists'):build(DB.parent)

    def test_modified_conversion_and_duplicate_view_rejected(self):
        for name,sql,error in [
            ('conversion',"UPDATE scale_observations SET value=raw_value WHERE block='Gráfico 8A'",'Converted observation'),
            ('view','CREATE OR REPLACE VIEW scaled_usable_keys AS SELECT * FROM usable_keys UNION ALL SELECT book,row_number FROM scale_observations UNION ALL SELECT book,row_number FROM scale_observations','multiplicity')]:
            path=Path(self.temp.name)/(name+'.duckdb');shutil.copyfile(DB,path)
            with duckdb.connect(str(path)) as con:
                con.execute(sql)
                with self.assertRaisesRegex(ValueError,error):check_new(con)


if __name__=='__main__':unittest.main()
