"""Independent raw/category/name reads and adversarial indexed chart controls."""
import importlib.util
from decimal import Decimal,localcontext
from pathlib import Path
import shutil
import tempfile
import unittest
import xml.etree.ElementTree as ET
from zipfile import ZipFile
if not all(importlib.util.find_spec(n) for n in ['duckdb','openpyxl']):
    raise unittest.SkipTest('Optional Excel dependencies required')
import duckdb
import openpyxl
from experiments.indexed_semantics import ROOT,IndexedResolver,check_new,query,build,C
from experiments.bulk_semantics import ref_cells
from experiments.excel_pilot import BOOKS

DB=ROOT/'evaluation/indexed_semantics_2026-10-31/run_v4/indexed.duckdb'


class TestIndexedSemantics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not DB.exists():raise unittest.SkipTest('Build indexed integration fixture first')
        cls.temp=tempfile.TemporaryDirectory(dir=ROOT/'tmp')
    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def test_all_raw_categories_names_and_conversions_independent_reader(self):
        with duckdb.connect(str(DB),read_only=True) as con:
            c=con.execute('SELECT * FROM indexed_observations');names=[n[0] for n in c.description]
            rows=[dict(zip(names,row)) for row in c.fetchall()]
        self.assertEqual(len(rows),4341)
        books={b:openpyxl.load_workbook(ROOT/name,read_only=True,data_only=True) for b,(name,_,_) in BOOKS.items()}
        files={b:ZipFile(ROOT/name) for b,(name,_,_) in BOOKS.items()};sheets={};parts={}
        for r in rows:
            source_names=[r['source_sheet']]
            if r['category_basis']=='source_cells':source_names.append(ref_cells(r['category_ref'])[0])
            if r['name_ref']:source_names.append(ref_cells(r['name_ref'])[0])
            for sheet in source_names:
                key=r['book'],sheet
                if key not in sheets:sheets[key]={c.coordinate:c.value for row in books[r['book']][sheet].iter_rows() for c in row if c.value is not None}
        for r in rows:
            raw=Decimal(str(sheets[r['book'],r['source_sheet']][r['source_cell']]))
            self.assertLessEqual(abs(raw-r['raw_value']),Decimal('1e-10'))
            with localcontext() as ctx:
                ctx.prec=60;self.assertEqual(r['value'],r['raw_value']*r['multiplier']/r['divisor'])
            if r['category_basis']=='source_cells':
                sheet,cells=ref_cells(r['category_ref']);category=str(sheets[r['book'],sheet][cells[r['point_index']]])
            else:
                key=r['book'],r['chart_part']
                if key not in parts:parts[key]=ET.fromstring(files[r['book']].read(r['chart_part']))
                node=parts[key].findall('.//c:ser',C)[int(r['series_id'].split('_s')[1])-1]
                cat=node.find('c:cat/c:strLit',C)
                if cat is None:cat=node.find('c:cat/c:numLit',C)
                category=next(p.find('c:v',C).text for p in cat.findall('c:pt',C) if int(p.get('idx'))==r['point_index'])
            self.assertEqual(category,r['category_literal'])
            if r['name_ref']:
                sheet,cells=ref_cells(r['name_ref']);name=sheets[r['book'],sheet][cells[0]]
                if isinstance(name,(int,float)) and name==int(name):name=str(int(name))
                self.assertEqual(str(name),r['series_literal'])
        for b in books.values():b.close()
        for f in files.values():f.close()

    def row(self,book,block,sid=None):
        with duckdb.connect(str(DB),read_only=True) as con:
            c=con.execute('SELECT * FROM staging_records WHERE book=? AND bloque=? AND estado_dato=? AND (? IS NULL OR serie_id=?) LIMIT 1',[book,block,'dato',sid,sid])
            return dict(zip([n[0] for n in c.description],c.fetchone()))

    def test_distribution_scope_and_numeric_name(self):
        with duckdb.connect(str(DB),read_only=True) as con:
            result=con.execute("SELECT DISTINCT series_literal,name_basis,unit,semantic_scope FROM indexed_observations WHERE block='Gráfico 34'").fetchall()
            self.assertEqual(len(result),2)
            for label,basis,unit,scope in result:
                self.assertIn(label,['2018','2023']);self.assertEqual(basis,'source_numeric_scalar')
                self.assertEqual(unit,'percent_plot_display');self.assertEqual(scope,'literal_plot_display_only')
            self.assertEqual(con.execute("SELECT count(*) FROM indexed_observations WHERE semantic_scope='literal_plot_display_only'").fetchone()[0],2284)

    def test_date_serial_prefix_not_invented_tail(self):
        r=query(DB,book='2025',block='Gráfico 67',series_id='chart116_s1',point_index=554)
        self.assertEqual(r['series_literal'],'Italia')
        self.assertIn(r['binding_mode'],['cache_verified_shared_indices'])
        with self.assertRaises(LookupError):query(DB,book='2025',block='Gráfico 67',series_id='chart116_s1',point_index=555)

    def test_literal_categories_exist_and_empty_declared_axis_rejected(self):
        with duckdb.connect(str(DB),read_only=True) as con:
            self.assertGreater(con.execute("SELECT count(*) FROM indexed_observations WHERE category_basis='literal_chart_points'").fetchone()[0],400)
        row=self.row('2025','Gráfico 69','chart119_s1');review=IndexedResolver()
        try:
            with self.assertRaises(ValueError):review.binding(row)
        finally:review.close()

    def test_corrupted_literal_indices_rejected(self):
        with duckdb.connect(str(DB),read_only=True) as con:
            book,block,sid=con.execute("SELECT book,block,series_id FROM indexed_observations WHERE category_basis='literal_chart_points' LIMIT 1").fetchone()
        row=self.row(book,block,sid);review=IndexedResolver()
        try:
            part=row['parte_grafico_imagen'];root=ET.fromstring(review.sources[book].xml.parts[part]);node=root.findall('.//c:ser',C)[int(sid.split('_s')[1])-1]
            cat=node.find('c:cat/c:strLit',C)
            if cat is None:cat=node.find('c:cat/c:numLit',C)
            pts=cat.findall('c:pt',C);pts[0].set('idx',pts[1].get('idx'))
            review.sources[book].xml.parts[part]=ET.tostring(root)
            with self.assertRaises(ValueError):review.binding(row)
        finally:review.close()

    def test_reviewed_typo_only_not_fuzzy(self):
        review=IndexedResolver()
        try:
            row=self.row('2026','Anexo I. Gr 5')
            self.assertIsNotNone(review.title_proof(row,review.evidence(row)))
            false=dict(row,titulo='Evolución del saldo público',metrica_literal='Evolución del saldo público')
            self.assertIsNone(review.title_proof(false,review.evidence(row)))
        finally:review.close()

    def test_blank_point_and_no_fallback(self):
        row=self.row('2025','Gráfico 34');review=IndexedResolver()
        try:
            binding=review.binding(row);index=int(row['id'].split('_p')[1]);binding['categories'][index]=' '
            modified=dict(row,periodo_categoria_literal=' ')
            status,result=review.review(modified);self.assertEqual(status,'unproved_category_point');self.assertIsNone(result)
        finally:review.close()
        with self.assertRaises(ValueError):query(DB,book='2025',block='Gráfico 34',series_id='unknown',point_index=True)
        with self.assertRaises(LookupError):query(DB,book='2025',block='Gráfico 34',series_id='unknown',point_index=0)

    def test_partition_and_no_overwrite(self):
        with duckdb.connect(str(DB),read_only=True) as con:
            self.assertEqual(con.execute('SELECT count(*) FROM resolved_usable_keys').fetchone()[0],21932)
            self.assertEqual(con.execute('SELECT count(*) FROM indexed_usable_keys').fetchone()[0],26273)
            self.assertEqual(con.execute('SELECT count(*) FROM indexed_numeric_queue').fetchone()[0],20940)
        with self.assertRaisesRegex(ValueError,'Output exists'):build(DB.parent)

    def test_unit_tampering_and_duplicate_view_rejected(self):
        for name,sql,error in [
            ('unit',"UPDATE indexed_observations SET unit='percent_population' WHERE block='Gráfico 34'",'Indexed content'),
            ('view','CREATE OR REPLACE VIEW indexed_usable_keys AS SELECT * FROM resolved_usable_keys UNION ALL SELECT book,row_number FROM indexed_observations UNION ALL SELECT book,row_number FROM indexed_observations','multiplicity')]:
            path=Path(self.temp.name)/(name+'.duckdb');shutil.copyfile(DB,path)
            with duckdb.connect(str(path)) as con:
                con.execute(sql)
                with self.assertRaisesRegex(ValueError,error):check_new(con)


if __name__=='__main__':unittest.main()
