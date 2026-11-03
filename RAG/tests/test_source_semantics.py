"""Independent original XML reads and adversarial semantic evidence checks."""
import importlib.util
from decimal import Decimal,localcontext
from pathlib import Path
import posixpath
import shutil
import tempfile
import unittest
import xml.etree.ElementTree as ET
from zipfile import ZipFile
if not all(importlib.util.find_spec(n) for n in ['duckdb','openpyxl','pypdf']):
    raise unittest.SkipTest('Optional source-review dependencies required')
import duckdb
import openpyxl
from experiments.source_semantics import ROOT,SourceResolver,check_new,query
from experiments.excel_pilot import BOOKS,PDF_HASHES,sha
from experiments.bulk_semantics import C,ref_cells

DB=ROOT/'evaluation/source_semantics_2026-10-31/run_v2/source.duckdb'
M={'m':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
R='http://schemas.openxmlformats.org/officeDocument/2006/relationships'


class TestSourceSemantics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not DB.exists():raise unittest.SkipTest('Build source integration fixture first')
        cls.temp=tempfile.TemporaryDirectory(dir=ROOT/'tmp')
    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def row(self,book,block,sid=None):
        with duckdb.connect(str(DB),read_only=True) as con:
            c=con.execute('SELECT * FROM staging_records WHERE book=? AND bloque=? AND estado_dato=? AND (? IS NULL OR serie_id=?) LIMIT 1',[book,block,'dato',sid,sid])
            values=c.fetchone();self.assertIsNotNone(values)
            return dict(zip([n[0] for n in c.description],values))

    def test_all_raw_categories_names_and_conversions_independently(self):
        with duckdb.connect(str(DB),read_only=True) as con:
            c=con.execute('SELECT * FROM source_observations');names=[n[0] for n in c.description]
            rows=[dict(zip(names,row)) for row in c.fetchall()]
        self.assertEqual(len(rows),5432)
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

    def test_unit_families_kept_separate(self):
        with duckdb.connect(str(DB),read_only=True) as con:
            self.assertEqual(con.execute("SELECT unit,count(*) FROM source_observations WHERE block='Gráfico 68' GROUP BY unit").fetchall(),[('basis_points_risk_spread_as_published',1842)])
            self.assertEqual(con.execute("SELECT count(*) FROM source_observations WHERE block='Gráfico 66' AND unit='percent_policy_rate_as_published'").fetchone()[0],906)
            self.assertEqual(con.execute("SELECT count(*) FROM source_observations WHERE block='Gráfico 65' AND unit='percent_annual_inflation_as_published'").fetchone()[0],220)
            self.assertEqual(con.execute("SELECT count(*) FROM source_observations WHERE block IN ('Anexo I. Gr 10A','Anexo I. Gr 10B') AND unit='years'").fetchone()[0],1324)
            self.assertEqual(con.execute("SELECT count(*) FROM source_observations WHERE block IN ('Gráfico 65','Gráfico 66','Gráfico 68') AND (multiplier<>1 OR divisor<>1)").fetchone()[0],0)

    def test_complete_hierarchy_and_missing_level_rejected(self):
        row=self.row('2026','Gráfico 14B','chart22_s1');r=SourceResolver()
        try:
            b=r.binding(row);self.assertEqual(b['label'],'Total | Histórico');self.assertEqual(b['name_basis'],'complete_source_header_range')
        finally:r.close()
        row=self.row('2026','Gráfico 14B','chart22_s2');r=SourceResolver()
        try:
            with self.assertRaisesRegex(ValueError,'Incomplete hierarchical'):r.binding(row)
        finally:r.close()
        with duckdb.connect(str(DB),read_only=True) as con:self.assertEqual(con.execute("SELECT count(*) FROM source_observations WHERE name_basis='complete_source_header_range'").fetchone()[0],48)

    def test_helpers_and_signed_population_pyramids_stay_pending(self):
        with duckdb.connect(str(DB),read_only=True) as con:
            for reason,n in [('drawing_helper_not_economic_series',49),('signed_pyramid_scope_review',240)]:
                self.assertEqual(con.execute('SELECT count(*) FROM source_review WHERE status=?',[reason]).fetchone()[0],n)
            self.assertEqual(con.execute("SELECT count(*) FROM indexed_numeric_queue s JOIN source_review r USING(book,row_number) WHERE r.status='drawing_helper_not_economic_series' AND s.bloque IN ('Gráfico 25C','Gráfico 25D')").fetchone()[0],20)
            self.assertEqual(con.execute("SELECT count(*) FROM source_observations WHERE lower(series_literal)='auxiliar' OR block IN ('Gráfico 12A','Gráfico 12B','Gráfico 12C')").fetchone()[0],0)

    def test_pdf_evidence_is_required_and_archived(self):
        r=SourceResolver();row=self.row('2025','Gráfico 68')
        try:
            self.assertEqual(r.review(row)[0],'usable_source')
            r.pdf_pages['2025',130]='prima de riesgo sin unidad'
            self.assertEqual(r.review(row)[0],'supplemental_evidence_conflict')
        finally:r.close()
        with duckdb.connect(str(DB),read_only=True) as con:
            self.assertEqual(con.execute('SELECT book,page FROM source_pdf_evidence ORDER BY book,page').fetchall(),[('2025',129),('2025',130),('2026',88)])
            for book,page,digest,text in con.execute('SELECT * FROM source_pdf_evidence').fetchall():
                self.assertEqual(digest,PDF_HASHES[book]);self.assertGreater(len(text),1000)

    def test_exact_title_headers_and_series_identity(self):
        row=self.row('2026','Anexo I. Gr 10A');r=SourceResolver()
        try:
            self.assertEqual(r.review(row)[0],'usable_source')
            bad=dict(row,titulo=row['titulo'].replace('Hombres','Mujeres'))
            self.assertIsNone(r.review(bad)[1])
            r.sources['2026'].sheet(row['bloque'])[1]['B3']='ANEXO I. GRÁFICO 10A. MUJERES'
            r.extra_cache.clear();self.assertIsNone(r.review(row)[1])
        finally:r.close()
        row=self.row('2025','Gráfico 68');row['serie_literal']='Alemania';r=SourceResolver()
        try:self.assertEqual(r.review(row)[0],'series_identity_review')
        finally:r.close()

    def test_no_fallback_boolean_or_wrong_series(self):
        with duckdb.connect(str(DB),read_only=True) as con:
            args=con.execute('SELECT book,block,series_id,point_index FROM source_observations LIMIT 1').fetchone()
        book,block,sid,point=args
        self.assertEqual(query(DB,book=book,block=block,series_id=sid,point_index=point)['series_id'],sid)
        with self.assertRaises(ValueError):query(DB,book=book,block=block,series_id=sid,point_index=True)
        with self.assertRaises(LookupError):query(DB,book=book,block=block,series_id='nonexistent',point_index=point)

    def test_partition_and_frozen_predecessor(self):
        with duckdb.connect(str(DB),read_only=True) as con:
            self.assertEqual(con.execute('SELECT count(*) FROM indexed_usable_keys').fetchone()[0],26273)
            self.assertEqual(con.execute('SELECT count(*) FROM source_usable_keys').fetchone()[0],31705)
            self.assertEqual(con.execute('SELECT count(*) FROM source_numeric_queue').fetchone()[0],15508)
        from experiments.source_semantics import PREVIOUS,PREVIOUS_SHA
        self.assertEqual(sha(PREVIOUS),PREVIOUS_SHA)

    def test_changed_evidence_and_duplicate_view_rejected(self):
        path=Path(self.temp.name)/'tampered.duckdb';shutil.copyfile(DB,path)
        with duckdb.connect(str(path)) as con:
            con.execute("UPDATE source_pdf_evidence SET text='missing units' WHERE book='2025' AND page=130")
            with self.assertRaisesRegex(ValueError,'Source evidence/table mismatch'):check_new(con)
            con.execute('DROP VIEW source_usable_keys')
            con.execute('CREATE VIEW source_usable_keys AS SELECT * FROM indexed_usable_keys UNION ALL SELECT book,row_number FROM source_observations UNION ALL SELECT book,row_number FROM source_observations')
            # Restore the unrelated evidence mutation before testing multiplicity.
            with duckdb.connect(str(DB),read_only=True) as original:
                text=original.execute("SELECT text FROM source_pdf_evidence WHERE book='2025' AND page=130").fetchone()[0]
            con.execute("UPDATE source_pdf_evidence SET text=? WHERE book='2025' AND page=130",[text])
            with self.assertRaisesRegex(ValueError,'View multiplicity'):check_new(con)


if __name__=='__main__':unittest.main()
