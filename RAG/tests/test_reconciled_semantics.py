"""Independent whole-vector, header, category and raw-value reconciliation checks."""
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
if not all(importlib.util.find_spec(n) for n in ['duckdb','openpyxl','pypdf']):raise unittest.SkipTest('Optional Excel dependencies required')
import duckdb
import openpyxl
from openpyxl.utils.cell import range_boundaries,coordinate_to_tuple,get_column_letter
from experiments.reconciled_semantics import ROOT,Reconciler,check_new,query,PREVIOUS,PREVIOUS_SHA
from experiments.excel_pilot import BOOKS,sha
from experiments.bulk_semantics import C,ref_cells

DB=ROOT/'evaluation/reconciled_semantics_2026-10-31/run_v3/reconciled.duckdb'
M={'m':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
R='http://schemas.openxmlformats.org/officeDocument/2006/relationships'


class TestReconciledSemantics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not DB.exists():raise unittest.SkipTest('Build reconciliation fixture first')
        cls.temp=tempfile.TemporaryDirectory(dir=ROOT/'tmp')
    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def row(self,block,sid=None):
        with duckdb.connect(str(DB),read_only=True) as con:
            c=con.execute('SELECT * FROM table_numeric_queue WHERE book=? AND bloque=? AND (? IS NULL OR serie_id=?) ORDER BY row_number LIMIT 1',['2026',block,sid,sid]);v=c.fetchone();self.assertIsNotNone(v)
            return dict(zip([n[0] for n in c.description],v))

    def test_all_values_categories_and_names_independent_original_reader(self):
        with duckdb.connect(str(DB),read_only=True) as con:
            c=con.execute('SELECT * FROM reconciled_observations');names=[n[0] for n in c.description];rows=[dict(zip(names,r)) for r in c.fetchall()]
        self.assertEqual(len(rows),894)
        path=ROOT/BOOKS['2026'][0];book=openpyxl.load_workbook(path,read_only=True,data_only=True);z=ZipFile(path)
        values={};raw={};merges={};charts={}
        text=lambda v:str(int(v)) if isinstance(v,(int,float)) and v==int(v) else str(v)
        norm=lambda s:''.join(c for c in s.lower() if c.isalnum())
        try:
            rels={n.get('Id'):posixpath.normpath(posixpath.join('xl',n.get('Target'))) for n in ET.fromstring(z.read('xl/_rels/workbook.xml.rels'))}
            paths={n.get('name'):rels[n.get('{'+R+'}id')] for n in ET.fromstring(z.read('xl/workbook.xml')).findall('m:sheets/m:sheet',M)}
            for r in rows:
                sheet=r['source_sheet']
                if sheet not in raw:
                    root=ET.fromstring(z.read(paths[sheet]));raw[sheet]={n.get('r'):n for n in root.findall('.//m:sheetData/m:row/m:c',M)}
                    values[sheet]={c.coordinate:c.value for rr in book[sheet].iter_rows() for c in rr if c.value is not None}
                    merges[sheet]=[n.get('ref') for n in root.findall('m:mergeCells/m:mergeCell',M)]
                node=raw[sheet][r['source_cell']];self.assertIn(node.get('t'),[None,'n']);v=Decimal(node.find('m:v',M).text);self.assertEqual(v,r['raw_value'])
                with localcontext() as ctx:ctx.prec=60;self.assertEqual(v*r['multiplier']/r['divisor'],r['value'])
                _,cc=ref_cells(r['category_ref']);self.assertEqual(text(values[sheet][cc[r['point_index']]]),r['category_literal'])
                _,nc=ref_cells(r['name_ref']);parts=[]
                for cell in nc:
                    value=values[sheet].get(cell)
                    if value is None:
                        row,col=coordinate_to_tuple(cell);matches=[]
                        for ref in merges[sheet]:
                            c1,r1,c2,r2=range_boundaries(ref)
                            if r1<=row<=r2 and c1<=col<=c2:matches.append(ref)
                        self.assertEqual(len(matches),1);ref=matches[0];self.assertEqual(range_boundaries(ref)[1],range_boundaries(ref)[3]);value=values[sheet][ref.split(':')[0]]
                    parts.append(text(value))
                if r['chart_part'] not in charts:charts[r['chart_part']]=ET.fromstring(z.read(r['chart_part']))
                ser=charts[r['chart_part']].findall('.//c:ser',C)[int(r['series_id'].split('_s')[1])-1]
                literal=ser.find('c:val/c:numLit',C)
                if literal is not None:
                    point=next(p for p in literal.findall('c:pt',C) if int(p.get('idx'))==r['point_index'])
                    self.assertEqual(Decimal(point.find('c:v',C).text),v)
                    self.assertEqual(ser.find('c:tx/c:v',C).text,r['series_literal'])
                    self.assertEqual(norm(' | '.join(parts)),norm(r['series_literal']))
                    self.assertIsNone(r['universal_value_sheet']);self.assertEqual(r['universal_value_cell'],f"cache[{r['point_index']}]")
                else:self.assertEqual(' | '.join(parts),r['series_literal'])
        finally:book.close();z.close()

    def test_far_point_corruption_blocks_the_entire_literal_vector(self):
        row=self.row('Gráfico 10','chart11_s1');r=Reconciler()
        try:
            part=row['parte_grafico_imagen'];root=ET.fromstring(r.sources['2026'].xml.parts[part]);ser=root.findall('.//c:ser',C)[0]
            point=ser.findall('c:val/c:numLit/c:pt',C)[-1];point.find('c:v',C).text='99'
            r.sources['2026'].xml.parts[part]=ET.tostring(root)
            self.assertEqual(r.review_new(row)[0],'reconciliation_evidence_review')
            self.assertNotIn(('2026',part,row['serie_id']),r.bindings)
        finally:r.close()

    def test_missing_category_and_duplicate_index_rejected(self):
        row=self.row('Gráfico 10','chart11_s1')
        for mutation in ['remove_categories','duplicate_index']:
            r=Reconciler()
            try:
                part=row['parte_grafico_imagen'];root=ET.fromstring(r.sources['2026'].xml.parts[part]);ser=root.findall('.//c:ser',C)[0]
                if mutation=='remove_categories':ser.remove(ser.find('c:cat',C))
                else:ser.findall('c:val/c:numLit/c:pt',C)[-1].set('idx','0')
                r.sources['2026'].xml.parts[part]=ET.tostring(root)
                self.assertEqual(r.review_new(row)[0],'reconciliation_evidence_review')
            finally:r.close()

    def test_duration_categories_recovered_only_from_reviewed_grid(self):
        row=self.row('Gráfico 11A');r=Reconciler()
        try:
            status,record=r.review_new(row);self.assertEqual(status,'usable_reconciled');self.assertEqual(record[7],'1')
            self.assertIsNone(row['periodo_categoria_literal']);self.assertEqual(record[31],'reviewed_grid_missing_chart_categories')
            self.assertIsNone(r.review_new(dict(row,periodo_categoria_literal='2099'))[1])
        finally:r.close()
        r=Reconciler()
        try:
            r.sources['2026'].sheet('Gráfico 11A')[1]['B7']=None
            self.assertEqual(r.review_new(row)[0],'reconciliation_evidence_review')
        finally:r.close()

    def test_merged_names_preserve_original_and_require_original_merge(self):
        row=self.row('Gráfico 14B','chart22_s2');r=Reconciler()
        try:
            b=r.binding(row);self.assertEqual(b['label'],'Total | AIReF 2026');self.assertEqual(row['serie_literal'],'AIReF 2026')
        finally:r.close()
        r=Reconciler()
        try:
            source=r.sources['2026'];part=source.xml.sheets['Gráfico 14B'];root=ET.fromstring(source.xml.parts[part]);merges=root.find('m:mergeCells',M)
            for n in list(merges):
                if n.get('ref')=='C22:F22':merges.remove(n)
            source.xml.parts[part]=ET.tostring(root)
            self.assertEqual(r.review_new(row)[0],'reconciliation_evidence_review')
        finally:r.close()

    def test_markers_and_wrong_header_are_excluded_with_vector_evidence(self):
        with duckdb.connect(str(DB),read_only=True) as con:
            self.assertEqual(con.execute("SELECT count(*) FROM reconciled_review WHERE status='forecast_marker_not_measure'").fetchone()[0],142)
            self.assertEqual(con.execute("SELECT count(*) FROM reconciled_review WHERE status='literal_header_conflict'").fetchone()[0],71)
            self.assertEqual(con.execute("SELECT count(*) FROM reconciled_observations WHERE block='Gráfico 17'").fetchone()[0],0)
            e=con.execute("SELECT evidence FROM literal_vector_reviews WHERE part='xl/charts/chart25.xml' AND series_id='chart25_s2'").fetchone()[0]
        e=json.loads(e);self.assertEqual(e['literal_name'],'Previsión');self.assertEqual(e['source_name'],'Variable');self.assertEqual(e['points'],71)

    def test_scopes_and_percentage_conversion(self):
        with duckdb.connect(str(DB),read_only=True) as con:
            self.assertEqual(con.execute("SELECT count(*) FROM reconciled_observations WHERE semantic_scope='literal_plot_display_only'").fetchone()[0],544)
            self.assertEqual(con.execute("SELECT count(*) FROM reconciled_observations WHERE block='Gráfico 10' AND multiplier=100 AND unit='percent_plot_display'").fetchone()[0],460)
            self.assertEqual(con.execute("SELECT count(*) FROM reconciled_observations WHERE block='Gráfico 16' AND multiplier=1 AND unit='percent_population_as_labelled'").fetchone()[0],71)
            self.assertEqual(con.execute("SELECT count(*) FROM reconciled_observations WHERE block IN ('Gráfico 11A','Gráfico 11B') AND multiplier=100").fetchone()[0],96)

    def test_identity_bounds_exact_query_and_partition(self):
        row=self.row('Gráfico 10','chart11_s1');r=Reconciler()
        try:
            self.assertEqual(r.review_new(dict(row,id='invalid'))[0],'point_identity_review')
            self.assertEqual(r.review_new(dict(row,id='chart11_s1_p999'))[0],'point_outside_reconciled_vector')
        finally:r.close()
        with duckdb.connect(str(DB),read_only=True) as con:
            self.assertEqual(con.execute('SELECT count(*) FROM table_usable_keys').fetchone()[0],36011)
            self.assertEqual(con.execute('SELECT count(*) FROM reconciled_usable_keys').fetchone()[0],36905)
            self.assertEqual(con.execute('SELECT count(*) FROM reconciled_numeric_queue').fetchone()[0],10308)
            self.assertEqual(con.execute('SELECT count(*) FROM grid_conflicts JOIN reconciled_usable_keys USING(book,row_number)').fetchone()[0],0)
        self.assertEqual(sha(PREVIOUS),PREVIOUS_SHA)
        self.assertEqual(query(DB,book='2026',block='Gráfico 10',series_id='chart11_s1',point_index=0)['series_literal'],'2018')
        with self.assertRaises(ValueError):query(DB,book='2026',block='Gráfico 10',series_id='chart11_s1',point_index=True)
        with self.assertRaises(LookupError):query(DB,book='2026',block='Gráfico 17',series_id='chart25_s2',point_index=0)

    def test_data_evidence_and_view_tampering_rejected(self):
        path=Path(self.temp.name)/'tampered.duckdb';shutil.copyfile(DB,path)
        with duckdb.connect(str(path)) as con:
            con.execute('BEGIN');con.execute("UPDATE reconciled_observations SET universal_series='wrong'")
            with self.assertRaisesRegex(ValueError,'Reconciled evidence mismatch'):check_new(con)
            con.execute('ROLLBACK');con.execute('BEGIN');con.execute("UPDATE literal_vector_reviews SET evidence='{}'")
            with self.assertRaisesRegex(ValueError,'Reconciled evidence mismatch'):check_new(con)
            con.execute('ROLLBACK');con.execute('DROP VIEW reconciled_usable_keys')
            con.execute('CREATE VIEW reconciled_usable_keys AS SELECT * FROM table_usable_keys UNION ALL SELECT book,row_number FROM reconciled_observations UNION ALL SELECT book,row_number FROM reconciled_observations')
            with self.assertRaisesRegex(ValueError,'View multiplicity'):check_new(con)


if __name__=='__main__':unittest.main()
