"""Whole-corpus classification, exact source identity and unit/scale regressions."""
import importlib.util
from decimal import Decimal
import json
from pathlib import Path
import shutil
import tempfile
import unittest
if not all(importlib.util.find_spec(n) for n in ['duckdb','openpyxl']):
    raise unittest.SkipTest('Optional Excel dependencies required')
import duckdb
import openpyxl
from experiments.bulk_semantics import ROOT,BOOKS,Reviewer,ref_cells,numeric_candidate,check_new,query,build

DB=ROOT/'evaluation/bulk_semantics_2026-10-31/run_v4/bulk.duckdb'


class TestBulkSemantics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not DB.exists():raise unittest.SkipTest('Build batch integration fixture first')
        cls.temp=tempfile.TemporaryDirectory(dir=ROOT/'tmp')
    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def row(self,book,block):
        with duckdb.connect(str(DB),read_only=True) as con:
            c=con.execute('SELECT * FROM staging_records WHERE book=? AND bloque=? AND estado_dato=? LIMIT 1',[book,block,'dato'])
            return dict(zip([n[0] for n in c.description],c.fetchone()))

    def test_all_admitted_values_independent_original_reader(self):
        with duckdb.connect(str(DB),read_only=True) as con:
            rows=con.execute('SELECT book,block,source_sheet,source_cell,value,unit_cell,unit_text FROM bulk_observations').fetchall()
        self.assertGreater(len(rows),1000)
        books={b:openpyxl.load_workbook(ROOT/path,read_only=True,data_only=True) for b,(path,_,_) in BOOKS.items()}
        sheets={}
        for book,block,sheet,*_ in rows:
            for target in [block,sheet]:
                if (book,target) not in sheets:
                    sheets[book,target]={c.coordinate:c.value for r in books[book][target].iter_rows() for c in r if c.value is not None}
        for book,block,sheet,cell,value,unit_cell,unit_text in rows:
            self.assertLessEqual(abs(value-Decimal(str(sheets[book,sheet][cell]))),Decimal('1e-10'))
            self.assertEqual(sheets[book,block][unit_cell],unit_text)
        for book in books.values():book.close()

    def test_every_record_has_one_classification(self):
        with duckdb.connect(str(DB),read_only=True) as con:
            self.assertEqual(con.execute('SELECT count(*) FROM record_review').fetchone()[0],64649)
            self.assertEqual(con.execute('SELECT count(*) FROM record_review WHERE status=?',['usable_previous']).fetchone()[0],896)
            self.assertEqual(con.execute('SELECT count(*) FROM record_review WHERE status IN (?,?) AND numeric_candidate',['absent','source_error']).fetchone()[0],0)
            self.assertEqual(con.execute('SELECT count(*) FROM usable_keys').fetchone()[0]+con.execute('SELECT count(*) FROM bulk_review_queue').fetchone()[0],64649)

    def test_horizon_and_symbol_are_not_economic_numbers(self):
        for kind,state in [('metadato_proyeccion','dato_numerico'),('simbolo_sin_interpretar','texto_publicado')]:
            self.assertFalse(numeric_candidate(dict(tipo=kind,estado_dato=state,decimal_XML_texto='2060')))

    def test_ranges_reject_external_and_rectangular(self):
        self.assertEqual(ref_cells("'Gráfico 2'!$B$2:$B$4"),('Gráfico 2',['B2','B3','B4']))
        for ref in ["'[other.xlsx]S'!A1:A2",'S!A1:B2','S!A2:A1','SUM(A1:A2)']:
            with self.subTest(ref=ref),self.assertRaises(ValueError):ref_cells(ref)

    def test_scaled_axes_and_conflicting_metric_blocked(self):
        reviewer=Reviewer()
        try:
            for book,sheet in [('2026','Gráfico 8A'),('2026','Anexo I. Gr 13A')]:
                status,result=reviewer.review(self.row(book,sheet))
                self.assertIsNone(result)
                self.assertEqual(status,'binding_or_axis_review')
            status,result=reviewer.review(self.row('2025','Gráfico RE 4.1'))
            self.assertIsNone(result)
            self.assertIn(status,['metric_evidence_review','binding_or_axis_review'])
        finally:reviewer.close()

    def test_named_series_not_overwritten_by_last_category(self):
        reviewer=Reviewer()
        try:
            binding=reviewer.binding(self.row('2025','Gráfico 5A'))
            self.assertEqual(binding['label'],'Hombres')
            self.assertEqual(binding['mapping']['1991'],'C5')
        finally:reviewer.close()

    def test_query_and_no_fallback(self):
        with duckdb.connect(str(DB),read_only=True) as con:
            book,block,sid,category=con.execute('SELECT book,block,series_id,category_literal FROM bulk_observations LIMIT 1').fetchone()
        result=query(DB,book=book,block=block,series_id=sid,category=category)
        self.assertEqual(result['series_id'],sid)
        with self.assertRaises(LookupError):query(DB,book=book,block=block,series_id='not_a_series',category=category)
        with self.assertRaises(ValueError):query(DB,book=book,block=block,series_id=sid,category=1991)

    def test_no_overwrite(self):
        with self.assertRaisesRegex(ValueError,'Output exists'):build(DB.parent)

    def test_classification_and_duplicate_view_tampering(self):
        for name,statement,error in [
            ('class',"UPDATE record_review SET numeric_candidate=true WHERE status='absent'",'classification'),
            ('view','CREATE OR REPLACE VIEW usable_keys AS SELECT book,row_number FROM admitted_observations UNION ALL SELECT book,row_number FROM bulk_observations UNION ALL SELECT book,row_number FROM bulk_observations','multiplicity')]:
            path=Path(self.temp.name)/(name+'.duckdb');shutil.copyfile(DB,path)
            with duckdb.connect(str(path)) as con:
                con.execute(statement)
                with self.assertRaisesRegex(ValueError,error):check_new(con)


if __name__=='__main__':unittest.main()
