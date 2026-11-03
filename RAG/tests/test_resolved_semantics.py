"""Large-batch literal evidence, partial caches and relative-unit regressions."""
import importlib.util
import json
from decimal import Decimal,localcontext
from pathlib import Path
import shutil
import tempfile
import unittest
import xml.etree.ElementTree as ET
if not all(importlib.util.find_spec(n) for n in ['duckdb','openpyxl']):
    raise unittest.SkipTest('Optional Excel dependencies required')
import duckdb
import openpyxl
from experiments.resolved_semantics import ROOT,Resolver,check_new,query,build,C
from experiments.bulk_semantics import ref_cells
from experiments.excel_pilot import BOOKS

DB=ROOT/'evaluation/resolved_semantics_2026-10-31/run_v3/resolved.duckdb'


class TestResolvedSemantics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not DB.exists():raise unittest.SkipTest('Build resolved integration fixture first')
        cls.temp=tempfile.TemporaryDirectory(dir=ROOT/'tmp')
    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def test_all_new_raw_category_and_conversion_independent_reader(self):
        with duckdb.connect(str(DB),read_only=True) as con:
            rows=con.execute('SELECT book,source_sheet,source_cell,raw_value,value,divisor,multiplier,category_ref,point_index,category_literal FROM resolved_observations').fetchall()
        self.assertEqual(len(rows),8366)
        books={book:openpyxl.load_workbook(ROOT/name,read_only=True,data_only=True) for book,(name,_,_) in BOOKS.items()}
        sheets={}
        for book,sheet,_,_,_,_,_,ref,*_ in rows:
            cat_sheet,_=ref_cells(ref)
            for name in [sheet,cat_sheet]:
                if (book,name) not in sheets:
                    sheets[book,name]={c.coordinate:c.value for r in books[book][name].iter_rows() for c in r if c.value is not None}
        for book,sheet,cell,raw,value,divisor,multiplier,ref,index,category in rows:
            self.assertLessEqual(abs(raw-Decimal(str(sheets[book,sheet][cell]))),Decimal('1e-10'))
            cat_sheet,addresses=ref_cells(ref)
            original_category=sheets[book,cat_sheet][addresses[index]]
            self.assertEqual(str(original_category),category)
            with localcontext() as ctx:
                ctx.prec=60;self.assertEqual(value,raw*multiplier/divisor)
        for book in books.values():book.close()

    def row(self,book,block,sid=None):
        with duckdb.connect(str(DB),read_only=True) as con:
            c=con.execute('SELECT * FROM staging_records WHERE book=? AND bloque=? AND estado_dato=? AND (? IS NULL OR serie_id=?) LIMIT 1',[book,block,'dato',sid,sid])
            return dict(zip([n[0] for n in c.description],c.fetchone()))

    def test_fertility_level_vs_percentage_change(self):
        with duckdb.connect(str(DB),read_only=True) as con:
            levels=con.execute("SELECT DISTINCT unit FROM resolved_observations WHERE book='2025' AND block='Gráfico 1A'").fetchall()
            changes=con.execute("SELECT DISTINCT unit FROM resolved_observations WHERE book='2025' AND block='Gráfico 1B'").fetchall()
        self.assertEqual(levels,[('children_per_woman',)])
        # The source sheet says change but does not explicitly establish %;
        # the index alone does not authorize an absolute or percentage unit.
        self.assertEqual(changes,[])

    def test_partial_prefix_and_index_query(self):
        r=query(DB,book='2026',block='Gráfico 22',series_id='chart30_s2',point_index=0)
        self.assertEqual(r['category_literal'],'1995')
        self.assertEqual(r['source_cell'],'C5')
        self.assertEqual(r['binding_mode'],'cache_verified_prefix')
        with self.assertRaises(LookupError):query(DB,book='2026',block='Gráfico 22',series_id='chart30_s2',point_index=31)
        with self.assertRaises(ValueError):query(DB,book='2026',block='Gráfico 22',series_id='chart30_s2',point_index=True)

    def test_bad_partial_cache_indices_and_numbers_rejected(self):
        row=self.row('2026','Gráfico 22','chart30_s2')
        for mutation in ['duplicate_index','wrong_value']:
            reviewer=Resolver()
            try:
                part=row['parte_grafico_imagen'];root=ET.fromstring(reviewer.sources['2026'].xml.parts[part])
                series=root.findall('.//c:ser',C)[1];points=series.findall('c:val/c:numRef/c:numCache/c:pt',C)
                if mutation=='duplicate_index':points[0].set('idx',points[1].get('idx'))
                else:points[0].find('c:v',C).text='999'
                reviewer.sources['2026'].xml.parts[part]=ET.tostring(root)
                with self.assertRaises(ValueError):reviewer.binding(row)
            finally:reviewer.close()

    def test_compound_literal_title_and_conflict(self):
        reviewer=Resolver()
        try:
            row=dict(titulo='ESPERANZA DE VIDA (AÑOS). AL NACER',metrica_literal='ESPERANZA DE VIDA (AÑOS). AL NACER')
            self.assertIsNotNone(reviewer.title_proof(row,[('B2','ESPERANZA DE VIDA (AÑOS)'),('B3','AL NACER')]))
            row=dict(titulo='EVOLUCIÓN DEL SALDO PRIMARIO (%PIB)',metrica_literal='EVOLUCIÓN DEL SALDO PRIMARIO (%PIB)')
            self.assertIsNone(reviewer.title_proof(row,[('C2','EVOLUCIÓN DEL GASTO PRIMARIO (%PIB)')]))
        finally:reviewer.close()

    def test_mixed_units_and_other_series_legend_not_unit_evidence(self):
        reviewer=Resolver()
        try:
            row=self.row('2025','Gráfico 78')
            self.assertIsNone(reviewer.unit_spec(row,reviewer.evidence(row),dict(multiplier=Decimal(1))))
            row=self.row('2025','Gráfico 56')
            for key in ['contexto_unidad_titulo']:
                row[key]=None
            row['serie_literal']='Tipo de interés implícito'
            # A different series named Intereses(%PIB) cannot establish GDP unit.
            spec=reviewer.unit_spec(row,reviewer.evidence(row),dict(multiplier=Decimal(1)))
            self.assertIsNone(spec)
        finally:reviewer.close()

    def test_previous_keys_and_partition(self):
        with duckdb.connect(str(DB),read_only=True) as con:
            self.assertEqual(con.execute('SELECT count(*) FROM scaled_usable_keys').fetchone()[0],13566)
            self.assertEqual(con.execute('SELECT count(*) FROM resolved_usable_keys').fetchone()[0],21932)
            self.assertEqual(con.execute('SELECT count(*) FROM resolved_numeric_queue').fetchone()[0],25281)

    def test_no_overwrite(self):
        with self.assertRaisesRegex(ValueError,'Output exists'):build(DB.parent)

    def test_tampered_unit_and_duplicate_view_rejected(self):
        for name,sql,error in [
            ('unit',"UPDATE resolved_observations SET unit='percent_change_as_labelled' WHERE block='Gráfico 1A'",'Resolved content'),
            ('view','CREATE OR REPLACE VIEW resolved_usable_keys AS SELECT * FROM scaled_usable_keys UNION ALL SELECT book,row_number FROM resolved_observations UNION ALL SELECT book,row_number FROM resolved_observations','multiplicity')]:
            path=Path(self.temp.name)/(name+'.duckdb');shutil.copyfile(DB,path)
            with duckdb.connect(str(path)) as con:
                con.execute(sql)
                with self.assertRaisesRegex(ValueError,error):check_new(con)


if __name__=='__main__':unittest.main()
