"""Independent quantity reads, pair geometry checks and semantic adversaries."""
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
from experiments.quantity_semantics import ROOT,QuantityResolver,check_new,query,PREVIOUS,PREVIOUS_SHA
from experiments.excel_pilot import BOOKS,sha
from experiments.bulk_semantics import C,ref_cells

DB=ROOT/'evaluation/quantity_semantics_2026-10-31/run_v2/quantities.duckdb'
M={'m':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
R='http://schemas.openxmlformats.org/officeDocument/2006/relationships'


class TestQuantitySemantics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not DB.exists():raise unittest.SkipTest('Build quantity fixture first')
        cls.temp=tempfile.TemporaryDirectory(dir=ROOT/'tmp')
    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def row(self,book,block,label=None):
        with duckdb.connect(str(DB),read_only=True) as con:
            c=con.execute('SELECT * FROM reconciled_numeric_queue WHERE book=? AND bloque=? AND (? IS NULL OR serie_literal=?) ORDER BY row_number LIMIT 1',[book,block,label,label]);v=c.fetchone();self.assertIsNotNone(v)
            return dict(zip([n[0] for n in c.description],v))

    def test_all_raw_categories_names_and_conversions_independently(self):
        with duckdb.connect(str(DB),read_only=True) as con:
            c=con.execute('SELECT * FROM quantity_observations');names=[n[0] for n in c.description]
            rows=[dict(zip(names,row)) for row in c.fetchall()]
        self.assertEqual(len(rows),1025)
        books={b:openpyxl.load_workbook(ROOT/name,read_only=True,data_only=True) for b,(name,_,_) in BOOKS.items()}
        files={b:ZipFile(ROOT/name) for b,(name,_,_) in BOOKS.items()};sheets={};xml_sheets={};charts={};paths={}
        try:
            for b,z in files.items():
                rels={n.get('Id'):posixpath.normpath(posixpath.join('xl',n.get('Target'))) for n in ET.fromstring(z.read('xl/_rels/workbook.xml.rels'))}
                for n in ET.fromstring(z.read('xl/workbook.xml')).findall('m:sheets/m:sheet',M):paths[b,n.get('name')]=rels[n.get('{'+R+'}id')]
            for r in rows:
                source_names=[r['source_sheet']]
                if r['category_basis']=='source_cells':source_names.append(ref_cells(r['category_ref'])[0])
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
                if r['category_basis']=='source_cells':
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

    def test_all_scatter_pairs_independent_xml_and_not_admitted(self):
        with duckdb.connect(str(DB),read_only=True) as con:
            c=con.execute('SELECT * FROM reviewed_scatter_pairs');ns=[n[0] for n in c.description];rows=[dict(zip(ns,r)) for r in c.fetchall()]
            self.assertEqual(len(rows),1007);self.assertEqual(con.execute('SELECT count(*) FROM reviewed_scatter_pairs JOIN quantity_usable_keys USING(book,row_number)').fetchone()[0],0)
        z=ZipFile(ROOT/BOOKS['2025'][0])
        try:
            rels={n.get('Id'):posixpath.normpath(posixpath.join('xl',n.get('Target'))) for n in ET.fromstring(z.read('xl/_rels/workbook.xml.rels'))}
            part=next(rels[n.get('{'+R+'}id')] for n in ET.fromstring(z.read('xl/workbook.xml')).findall('m:sheets/m:sheet',M) if n.get('name')=='Gráfico 15')
            cells={n.get('r'):n for n in ET.fromstring(z.read(part)).findall('.//m:sheetData/m:row/m:c',M)};chart=ET.fromstring(z.read('xl/charts/chart36.xml'))
            for r in rows:
                self.assertEqual(Decimal(cells[r['x_cell']].find('m:v',M).text),r['raw_x']);self.assertEqual(Decimal(cells[r['y_cell']].find('m:v',M).text),r['raw_y'])
                self.assertEqual(str(int(Decimal(cells[r['year_cell']].find('m:v',M).text))),r['year_label'])
                self.assertEqual(r['x_cell'][1:],r['y_cell'][1:]);self.assertEqual(r['year_cell'][1:],r['y_cell'][1:])
                node=chart.findall('.//c:ser',C)[int(r['series_id'].split('_s')[1])-1]
                for coordinate,key in [('x','raw_x'),('y','raw_y')]:
                    cache=node.find(f'c:{coordinate}Val/c:numRef/c:numCache',C)
                    v=next(n.find('c:v',C).text for n in cache.findall('c:pt',C) if int(n.get('idx'))==r['point_index'])
                    self.assertEqual(Decimal(v),r[key])
                self.assertEqual(r['unit_status'],'not_explicitly_stated')
                e=json.loads(r['evidence']);self.assertEqual(e['scope'],'geometry_verified_units_unresolved')
                self.assertEqual(e['country_identity'],'Japan_from_series_label' if r['series_literal']=='Japón' else 'not_available_per_point')
        finally:z.close()

    def test_conditional_fertility_difference_and_cohort_time(self):
        with duckdb.connect(str(DB),read_only=True) as con:
            self.assertEqual(con.execute("SELECT unit,count(*) FROM quantity_observations WHERE block='Anexo I. Gr 3B' GROUP BY ALL").fetchall(),[('children_per_motherhood_entrant_as_labelled',45)])
            self.assertEqual(con.execute("SELECT unit,count(*) FROM quantity_observations WHERE block='Anexo I. Gr 7B' GROUP BY ALL").fetchall(),[('children_per_woman_difference_as_labelled',50)])
            self.assertEqual(con.execute("SELECT category_semantics,count(*) FROM quantity_observations WHERE block='Anexo I. Gr 4A' GROUP BY ALL").fetchall(),[('year_generation_turns_30',40)])
            book,block,sid,idx,category=con.execute("SELECT book,block,series_id,point_index,category_literal FROM quantity_observations WHERE block='Anexo I. Gr 4A' ORDER BY point_index LIMIT 1").fetchone()
        r=query(DB,book=book,block=block,series_id=sid,point_index=idx)
        self.assertEqual(category,'1965');self.assertEqual(r['category_literal'],'1965');self.assertEqual(r['category_semantics'],'year_generation_turns_30')

    def test_growth_horizons_and_conflicts_kept_separate(self):
        with duckdb.connect(str(DB),read_only=True) as con:
            self.assertEqual(con.execute("SELECT count(*) FROM quantity_observations WHERE block='Gráfico 51'").fetchone()[0],110)
            self.assertEqual(con.execute("SELECT count(*) FROM quantity_observations WHERE block='Gráfico 54'").fetchone()[0],0)
            self.assertEqual(con.execute("SELECT count(*) FROM quantity_review WHERE status='growth_decomposition_conflict'").fetchone()[0],165)
            groups=con.execute('SELECT block,source_group,evidence FROM growth_decomposition_reviews ORDER BY block,source_group').fetchall()
        self.assertEqual(len(groups),4)
        for block,group,text in groups:
            p=json.loads(text)
            if block=='Gráfico 51':
                self.assertEqual(p['status'],'pass');self.assertEqual(len(p['valid_indices']),9 if group==0 else 46)
                self.assertLessEqual(Decimal(p['max_absolute_residual']),Decimal('1e-12'))
            else:self.assertEqual(p['status'],'conflict');self.assertTrue(p['conflicts'])

    def test_decomposition_partial_triplet_and_reference_mismatch_rejected(self):
        row=self.row('2025','Gráfico 51','Crecimiento por demografía');r=QuantityResolver()
        try:
            r.sources['2025'].sheet('Gráfico 51')[0].pop('D7')
            with self.assertRaisesRegex(ValueError,'Partial decomposition'):r.review_quantity(row)
        finally:r.close()
        r=QuantityResolver()
        try:
            b=dict(r.binding(row),value_ref="'Gráfico 51'!D12:BF12")
            with self.assertRaisesRegex(ValueError,'one reviewed horizon'):r.decomposition(row,b)
        finally:r.close()

    def test_pdf_unit_and_header_proofs_cannot_be_missing(self):
        row=self.row('2025','Gráfico RE 1.7','INE 2024');r=QuantityResolver()
        try:
            r.pdf_pages['2025',46]='Tasa de dependencia sin denominador'
            self.assertIsNone(r.review_quantity(row)[1])
        finally:r.close()
        r=QuantityResolver()
        try:
            r.sources['2025'].sheet('Gráfico RE 1.7')[1]['L5']='OTRA MAGNITUD'
            self.assertIsNone(r.review_quantity(row)[1])
        finally:r.close()

    def test_scatter_corrupt_cache_and_axis_not_accepted(self):
        row=self.row('2025','Gráfico 15');r=QuantityResolver()
        try:
            part='xl/charts/chart36.xml';root=ET.fromstring(r.sources['2025'].xml.parts[part]);node=root.findall('.//c:ser',C)[0]
            node.find('c:xVal/c:numRef/c:numCache/c:pt/c:v',C).text='99'
            r.sources['2025'].xml.parts[part]=ET.tostring(root)
            with self.assertRaisesRegex(ValueError,'cache/cell conflict'):r.scatter_pair(row)
        finally:r.close()
        r=QuantityResolver()
        try:
            part='xl/charts/chart36.xml';root=ET.fromstring(r.sources['2025'].xml.parts[part]);axis=root.findall('.//c:valAx',C)[0];axis.find('c:axPos',C).set('val','l')
            r.sources['2025'].xml.parts[part]=ET.tostring(root)
            with self.assertRaises(ValueError):r.scatter_pair(row)
        finally:r.close()
        r=QuantityResolver()
        try:
            r.scatter_pair(row)
            with self.assertRaisesRegex(ValueError,'Scatter source identity'):r.scatter_pair(dict(row,parte_grafico_imagen='xl/charts/chart99.xml'))
        finally:r.close()

    def test_old_exclusions_scatter_and_partition(self):
        with duckdb.connect(str(DB),read_only=True) as con:
            self.assertEqual(con.execute('SELECT count(*) FROM reconciled_usable_keys').fetchone()[0],36905)
            self.assertEqual(con.execute('SELECT count(*) FROM quantity_usable_keys').fetchone()[0],37930)
            self.assertEqual(con.execute('SELECT count(*) FROM quantity_numeric_queue').fetchone()[0],9283)
            self.assertEqual(con.execute("SELECT count(*) FROM quantity_numeric_queue WHERE quantity_reason='scatter_units_unresolved'").fetchone()[0],1007)
            self.assertEqual(con.execute('SELECT count(*) FROM grid_conflicts JOIN quantity_usable_keys USING(book,row_number)').fetchone()[0],0)
            self.assertEqual(con.execute("SELECT count(*) FROM quantity_observations WHERE series_literal IN ('Minimo','Bandas','Rango mundial','Rango Emisores')").fetchone()[0],0)
        self.assertEqual(sha(PREVIOUS),PREVIOUS_SHA)

    def test_precise_query_rejects_scatter_unknown_and_boolean(self):
        with duckdb.connect(str(DB),read_only=True) as con:book,block,sid,idx=con.execute('SELECT book,block,series_id,point_index FROM quantity_observations LIMIT 1').fetchone()
        self.assertEqual(query(DB,book=book,block=block,series_id=sid,point_index=idx)['series_id'],sid)
        with self.assertRaises(ValueError):query(DB,book=book,block=block,series_id=sid,point_index=True)
        with self.assertRaises(LookupError):query(DB,book='2025',block='Gráfico 15',series_id='chart36_s1',point_index=0)
        with self.assertRaises(LookupError):query(DB,book=book,block=block,series_id='invalid',point_index=idx)

    def test_mutations_and_duplicate_view_rejected(self):
        path=Path(self.temp.name)/'tampered.duckdb';shutil.copyfile(DB,path)
        with duckdb.connect(str(path)) as con:
            con.execute('BEGIN');con.execute("UPDATE quantity_observations SET category_semantics='birth_year'")
            with self.assertRaisesRegex(ValueError,'Quantity evidence mismatch'):check_new(con)
            con.execute('ROLLBACK');con.execute('BEGIN');con.execute('UPDATE reviewed_scatter_pairs SET raw_x=99')
            with self.assertRaisesRegex(ValueError,'Quantity evidence mismatch'):check_new(con)
            con.execute('ROLLBACK');con.execute('DROP VIEW quantity_usable_keys')
            con.execute('CREATE VIEW quantity_usable_keys AS SELECT * FROM reconciled_usable_keys UNION ALL SELECT book,row_number FROM quantity_observations UNION ALL SELECT book,row_number FROM quantity_observations')
            with self.assertRaisesRegex(ValueError,'View multiplicity'):check_new(con)


if __name__=='__main__':unittest.main()
