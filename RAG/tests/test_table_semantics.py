"""Independent reads and adversarial checks for heterogeneous published tables."""
import importlib.util
from decimal import Decimal,localcontext
from pathlib import Path
import json
import posixpath
import shutil
import tempfile
import unittest
import xml.etree.ElementTree as ET
from zipfile import ZipFile
if not all(importlib.util.find_spec(n) for n in ['duckdb','openpyxl','pypdf']):raise unittest.SkipTest('Optional source-review dependencies required')
import duckdb
import openpyxl
from experiments.table_semantics import ROOT,TableResolver,check_new,query_table,query_chart,PREVIOUS,PREVIOUS_SHA
from experiments.excel_pilot import BOOKS,sha
from experiments.bulk_semantics import C,ref_cells

DB=ROOT/'evaluation/table_semantics_2026-10-31/run_v3/tables.duckdb'
M={'m':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
R='http://schemas.openxmlformats.org/officeDocument/2006/relationships'


class TestTableSemantics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not DB.exists():raise unittest.SkipTest('Build table integration fixture first')
        cls.temp=tempfile.TemporaryDirectory(dir=ROOT/'tmp')
    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def row(self,book,block,cell):
        with duckdb.connect(str(DB),read_only=True) as con:
            c=con.execute('SELECT * FROM grid_numeric_queue WHERE book=? AND bloque=? AND celda_valor=? ORDER BY row_number DESC LIMIT 1',[book,block,cell]);values=c.fetchone();self.assertIsNotNone(values)
            return dict(zip([n[0] for n in c.description],values))

    def test_all_tables_against_independent_xml_and_headers(self):
        with duckdb.connect(str(DB),read_only=True) as con:
            c=con.execute('SELECT * FROM table_observations');ns=[n[0] for n in c.description];rows=[dict(zip(ns,r)) for r in c.fetchall()]
        self.assertEqual(len(rows),1003)
        books={b:openpyxl.load_workbook(ROOT/name,read_only=True,data_only=True) for b,(name,_,_) in BOOKS.items()}
        formulas={b:openpyxl.load_workbook(ROOT/name,read_only=True,data_only=False) for b,(name,_,_) in BOOKS.items()}
        files={b:ZipFile(ROOT/name) for b,(name,_,_) in BOOKS.items()};values={};cells={};fvalues={};paths={};merges={}
        norm=lambda s:''.join(c for c in str(s).lower() if c.isalnum())
        text=lambda v:str(int(v)) if isinstance(v,(int,float)) and v==int(v) else str(v).strip()
        try:
            for b,z in files.items():
                rels={n.get('Id'):posixpath.normpath(posixpath.join('xl',n.get('Target'))) for n in ET.fromstring(z.read('xl/_rels/workbook.xml.rels'))}
                for n in ET.fromstring(z.read('xl/workbook.xml')).findall('m:sheets/m:sheet',M):paths[b,n.get('name')]=rels[n.get('{'+R+'}id')]
            for r in rows:
                k=r['book'],r['source_sheet']
                if k not in cells:
                    root=ET.fromstring(files[k[0]].read(paths[k]));cells[k]={c.get('r'):c for c in root.findall('.//m:sheetData/m:row/m:c',M)}
                    merges[k]={n.get('ref') for n in root.findall('m:mergeCells/m:mergeCell',M)}
                    values[k]={c.coordinate:c.value for rr in books[k[0]][k[1]].iter_rows() for c in rr if c.value is not None}
                    fvalues[k]={c.coordinate:c.value for rr in formulas[k[0]][k[1]].iter_rows() for c in rr if c.value is not None}
                node=cells[k][r['source_cell']];self.assertIn(node.get('t'),[None,'n']);raw=Decimal(node.find('m:v',M).text)
                self.assertEqual(raw,r['raw_value'])
                with localcontext() as ctx:ctx.prec=60;self.assertEqual(raw*r['multiplier']/r['divisor'],r['value'])
                self.assertEqual(text(values[k][r['series_cell']]),r['series_literal'])
                self.assertEqual(text(values[k][r['category_cell']]),r['category_literal'])
                self.assertIn(norm(r['metric']),norm(values[k][r['metric_cell']]))
                self.assertEqual(text(values[k][r['title_cell']]),r['title_literal'])
                f=fvalues[k][r['source_cell']];f=f if isinstance(f,str) and f.startswith('=') else None
                self.assertEqual(f,r['formula'])
                e=json.loads(r['evidence'])
                if 'metric_merge' in e:self.assertIn(e['metric_merge'],merges[k])
        finally:
            for b in [*books.values(),*formulas.values()]:b.close()
            for f in files.values():f.close()

    def test_all_raw_categories_names_and_conversions_independently(self):
        with duckdb.connect(str(DB),read_only=True) as con:
            c=con.execute('SELECT * FROM context_observations');names=[n[0] for n in c.description]
            rows=[dict(zip(names,row)) for row in c.fetchall()]
        self.assertEqual(len(rows),592)
        books={b:openpyxl.load_workbook(ROOT/name,read_only=True,data_only=True) for b,(name,_,_) in BOOKS.items()}
        files={b:ZipFile(ROOT/name) for b,(name,_,_) in BOOKS.items()};sheets={};xml_sheets={};charts={};paths={}
        try:
            for b,z in files.items():
                rels={n.get('Id'):posixpath.normpath(posixpath.join('xl',n.get('Target'))) for n in ET.fromstring(z.read('xl/_rels/workbook.xml.rels'))}
                for n in ET.fromstring(z.read('xl/workbook.xml')).findall('m:sheets/m:sheet',M):paths[b,n.get('name')]=rels[n.get('{'+R+'}id')]
            for r in rows:
                source_names=[r['source_sheet']]
                if r['category_basis'] in ['source_cells','reviewed_source_grid']:source_names.append(ref_cells(r['category_ref'])[0])
                if r['name_ref']:source_names.append(ref_cells(r['name_ref'])[0])
                for sheet in source_names:
                    k=r['book'],sheet
                    if k not in sheets:
                        sheets[k]={c.coordinate:c.value for row in books[k[0]][sheet].iter_rows() for c in row if c.value is not None}
                        xml_sheets[k]={c.get('r'):c for c in ET.fromstring(files[k[0]].read(paths[k])).findall('.//m:sheetData/m:row/m:c',M)}
            for r in rows:
                cell=xml_sheets[r['book'],r['source_sheet']][r['source_cell']]
                self.assertIn(cell.get('t'),[None,'n']);raw=Decimal(cell.find('m:v',M).text)
                self.assertEqual(raw,r['raw_value'])
                with localcontext() as ctx:
                    ctx.prec=60;self.assertEqual(r['value'],raw*r['multiplier']/r['divisor'])
                ck=r['book'],r['chart_part']
                if ck not in charts:charts[ck]=ET.fromstring(files[ck[0]].read(ck[1]))
                node=charts[ck].findall('.//c:ser',C)[int(r['series_id'].split('_s')[1])-1]
                self.assertEqual(node.find('c:val/c:numRef/c:f',C).text,r['value_ref'])
                if r['category_basis'] in ['source_cells','reviewed_source_grid']:
                    sheet,cells=ref_cells(r['category_ref']);cat=str(sheets[r['book'],sheet][cells[r['point_index']]])
                else:
                    literal=node.find('c:cat/c:strLit',C)
                    if literal is None:literal=node.find('c:cat/c:numLit',C)
                    cat=next(p.find('c:v',C).text for p in literal.findall('c:pt',C) if int(p.get('idx'))==r['point_index'])
                self.assertEqual(cat,r['category_literal'])
                if r['name_ref']:
                    self.assertEqual(node.find('c:tx/c:strRef/c:f',C).text,r['name_ref'])
                    sheet,cells=ref_cells(r['name_ref']);parts=[sheets[r['book'],sheet][a] for a in cells]
                    text=lambda v:str(int(v)) if isinstance(v,(int,float)) and v==int(v) else str(v)
                    self.assertEqual(' | '.join(map(text,parts)),r['series_literal'])
                else:self.assertEqual(node.find('c:tx/c:v',C).text,r['series_literal'])
        finally:
            for b in books.values():b.close()
            for f in files.values():f.close()

    def test_levels_differences_and_model_horizons_separate(self):
        base=query_table(DB,book='2025',block='Cuadro 12',source_cell='D7')
        delta=query_table(DB,book='2025',block='Cuadro 12',source_cell='E8')
        self.assertEqual((base['role'],base['unit']),('published_base_level','percent_rate_as_labelled'))
        self.assertEqual((delta['role'],delta['unit']),('scenario_difference_from_base','percentage_points_rate_as_labelled'))
        r=query_table(DB,book='2026',block='Cuadro 5',source_cell='C9')
        self.assertEqual((r['metric'],r['series_literal'],r['category_literal'],r['unit']),('Saldo Público (% PIB)','MTA','t+5','percentage_points_gdp'))
        r=query_table(DB,book='2026',block='Cuadro 5',source_cell='G7')
        self.assertEqual((r['series_literal'],r['category_literal'],r['unit']),('MEGAIReF eficiencia','t+10','percent_deviation_from_no_measures'))

    def test_explicit_merges_and_institution_identity(self):
        row=self.row('2026','Cuadro 11','D6');r=TableResolver()
        try:
            self.assertEqual(r.table(row)[0],'usable_table')
            r.table_merges['2026','Cuadro 11']=[]
            with self.assertRaisesRegex(ValueError,'merged B range'):r.table(row)
        finally:r.close()
        r=TableResolver()
        try:
            with self.assertRaisesRegex(ValueError,'semantic identity'):r.table(dict(row,serie_literal='AIReF 2025'))
        finally:r.close()
        out=query_table(DB,book='2026',block='Cuadro 11',source_cell='D6')
        self.assertEqual(out['series_literal'],'INE 2024');self.assertEqual(out['category_literal'],'2025')

    def test_published_base_auxiliary_conflict_excluded(self):
        with duckdb.connect(str(DB),read_only=True) as con:
            self.assertEqual(con.execute("SELECT count(*) FROM table_review WHERE status='published_base_auxiliary_conflict'").fetchone()[0],22)
            self.assertEqual(con.execute("SELECT count(*) FROM table_review WHERE status='auxiliary_table_cells'").fetchone()[0],196)
            self.assertEqual(con.execute("SELECT count(*) FROM table_observations WHERE block='Cuadro 14'").fetchone()[0],0)
            book,block,data=con.execute('SELECT * FROM table_conflicts').fetchone()
        self.assertEqual((book,block),('2025','Cuadro 14'));details=json.loads(data);self.assertEqual(len(details),4)
        self.assertEqual(Decimal(details[0]['published_value']),Decimal('3.3'))
        self.assertNotEqual(Decimal(details[0]['published_value']),Decimal(details[0]['auxiliary_value']))
        with self.assertRaises(LookupError):query_table(DB,book='2025',block='Cuadro 14',source_cell='D7')

    def test_wrong_scenario_formula_and_cache_rejected(self):
        row=self.row('2025','Cuadro 12','E8');r=TableResolver()
        try:
            r.sources['2025'].sheet(row['bloque'])[1]['E8']='=J9-J7'
            with self.assertRaisesRegex(ValueError,'Wrong scenario/base'):r.table(dict(row,formula_texto='=J9-J7'))
        finally:r.close()
        r=TableResolver()
        try:
            r.sources['2025'].sheet(row['bloque'])[0]['E8'].find('m:v',M).text='0'
            with self.assertRaisesRegex(ValueError,'formula cache mismatch'):r.table(dict(row,decimal_XML_texto='0'))
        finally:r.close()

    def test_reference_formula_recovery_is_bounded(self):
        r=query_table(DB,book='2025',block='Gráfico 37',source_cell='K8')
        self.assertEqual(r['scope'],'reference_line_not_independent_forecast');self.assertEqual(r['formula'],'=K5')
        e=json.loads(r['evidence']);self.assertIsNone(e['universal_formula']);self.assertEqual(e['formula_resolution'],'original_formula_absent_from_universal')
        row=self.row('2025','Gráfico 37','K8');review=TableResolver()
        try:
            review.sources['2025'].sheet('Gráfico 37')[1]['K8']='=K5+1'
            with self.assertRaisesRegex(ValueError,'Unsupported reference-line'):review.table(row)
        finally:review.close()
        with duckdb.connect(str(DB),read_only=True) as con:
            self.assertEqual(con.execute("SELECT count(*) FROM (SELECT book,source_sheet,source_cell,count(*) n FROM table_observations GROUP BY ALL HAVING n<>1)").fetchone()[0],0)

    def test_percentage_format_applied_once_and_population_totals(self):
        with duckdb.connect(str(DB),read_only=True) as con:
            self.assertEqual(con.execute("SELECT DISTINCT multiplier FROM table_observations WHERE book='2026' AND block='Anexo I. Cuadro 1'").fetchall(),[(Decimal(100),)])
            self.assertEqual(con.execute("SELECT DISTINCT multiplier FROM table_observations WHERE book='2025' AND block='Cuadro 4'").fetchall(),[(Decimal(1),)])
            source_residuals={'2002':Decimal('0.000002436927978500000000'),'2008':Decimal('-0.000006569016341900000000')}
            for book,year,total in con.execute("SELECT book,category_literal,sum(value) FROM table_observations WHERE unit='percent_age_population' GROUP BY book,block,category_literal").fetchall():
                if book=='2026' and year in source_residuals:self.assertEqual(total-100,source_residuals[year])
                else:self.assertLess(abs(total-100),Decimal('1e-10'))

    def test_partition_old_conflicts_and_exact_chart_api(self):
        with duckdb.connect(str(DB),read_only=True) as con:
            self.assertEqual(con.execute('SELECT count(*) FROM grid_usable_keys').fetchone()[0],34416)
            self.assertEqual(con.execute('SELECT count(*) FROM table_usable_keys').fetchone()[0],36011)
            self.assertEqual(con.execute('SELECT count(*) FROM table_numeric_queue').fetchone()[0],11202)
            self.assertEqual(con.execute('SELECT count(*) FROM grid_conflicts JOIN table_usable_keys USING(book,row_number)').fetchone()[0],0)
            book,block,sid,idx=con.execute('SELECT book,block,series_id,point_index FROM context_observations LIMIT 1').fetchone()
        self.assertEqual(sha(PREVIOUS),PREVIOUS_SHA)
        self.assertEqual(query_chart(DB,book=book,block=block,series_id=sid,point_index=idx)['series_id'],sid)
        with self.assertRaises(ValueError):query_chart(DB,book=book,block=block,series_id=sid,point_index=True)
        with self.assertRaises(LookupError):query_table(DB,book='2026',block='Cuadro 11',source_cell='Z999')

    def test_tampering_and_duplicate_view_rejected(self):
        path=Path(self.temp.name)/'tampered.duckdb';shutil.copyfile(DB,path)
        with duckdb.connect(str(path)) as con:
            con.execute('BEGIN');con.execute("UPDATE table_observations SET unit='percent_rate_as_labelled' WHERE book='2025' AND block='Cuadro 12' AND source_cell='E8'")
            with self.assertRaisesRegex(ValueError,'Table evidence mismatch'):check_new(con)
            con.execute('ROLLBACK');con.execute('BEGIN');con.execute("UPDATE table_conflicts SET evidence='[]'")
            with self.assertRaisesRegex(ValueError,'Table evidence mismatch'):check_new(con)
            con.execute('ROLLBACK');con.execute('DROP VIEW table_usable_keys')
            con.execute('CREATE VIEW table_usable_keys AS SELECT * FROM grid_usable_keys UNION ALL SELECT book,row_number FROM table_observations UNION ALL SELECT book,row_number FROM context_observations UNION ALL SELECT book,row_number FROM table_observations')
            with self.assertRaisesRegex(ValueError,'View multiplicity'):check_new(con)


if __name__=='__main__':unittest.main()
