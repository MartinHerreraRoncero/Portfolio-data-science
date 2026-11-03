"""Independent raw-cell checks and adversarial worksheet category recovery."""
import importlib.util
from decimal import Decimal,localcontext
from pathlib import Path
import posixpath
import json
import shutil
import tempfile
import unittest
import xml.etree.ElementTree as ET
from zipfile import ZipFile
if not all(importlib.util.find_spec(n) for n in ['duckdb','openpyxl','pypdf']):
    raise unittest.SkipTest('Optional source-review dependencies required')
import duckdb
import openpyxl
from experiments.grid_semantics import ROOT,GridResolver,check_new,query,GRIDS,PREVIOUS,PREVIOUS_SHA
from experiments.excel_pilot import BOOKS,sha
from experiments.bulk_semantics import C,ref_cells

DB=ROOT/'evaluation/grid_semantics_2026-10-31/run_v1/grid.duckdb'
M={'m':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
R='http://schemas.openxmlformats.org/officeDocument/2006/relationships'


class TestGridSemantics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not DB.exists():raise unittest.SkipTest('Build grid integration fixture first')
        cls.temp=tempfile.TemporaryDirectory(dir=ROOT/'tmp')
    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def row(self,book,block,sid=None):
        with duckdb.connect(str(DB),read_only=True) as con:
            c=con.execute('SELECT * FROM source_numeric_queue WHERE book=? AND bloque=? AND (? IS NULL OR serie_id=?) LIMIT 1',[book,block,sid,sid])
            values=c.fetchone();self.assertIsNotNone(values)
            return dict(zip([n[0] for n in c.description],values))

    def test_all_raw_categories_names_and_conversions_independently(self):
        with duckdb.connect(str(DB),read_only=True) as con:
            c=con.execute('SELECT * FROM grid_observations');names=[n[0] for n in c.description]
            rows=[dict(zip(names,row)) for row in c.fetchall()]
        self.assertEqual(len(rows),2711)
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

    def test_recovered_categories_are_physical_source_rows(self):
        with duckdb.connect(str(DB),read_only=True) as con:
            c=con.execute("SELECT * FROM grid_observations WHERE category_basis='reviewed_source_grid'");names=[n[0] for n in c.description]
            rows=[dict(zip(names,row)) for row in c.fetchall()]
        self.assertEqual(len(rows),1891)
        for r in rows:
            self.assertIsNone(r['universal_category'])
            sheet,cc=ref_cells(r['category_ref']);_,vc=ref_cells(r['value_ref'])
            self.assertEqual(sheet,r['source_sheet']);self.assertEqual(cc[r['point_index']][1:],vc[r['point_index']][1:])
            self.assertTrue(cc[r['point_index']].startswith('B'))
            self.assertEqual(json.loads(r['grid_evidence'])['basis'],'reviewed_source_grid')

    def test_missing_category_or_wrong_title_not_invented(self):
        row=self.row('2025','Gráfico 115');r=GridResolver()
        try:
            b=r.binding(row);self.assertEqual(b['categories'][0],'2024')
            r.sources['2025'].sheet(row['bloque'])[1]['B8']=None
            r.bindings.clear();self.assertIsNone(r.review(row)[1])
        finally:r.close()
        r=GridResolver()
        try:
            profile=GRIDS[row['book'],row['bloque']]
            r.sources['2025'].sheet(row['bloque'])[1][profile['title_address']]='OTRO GRÁFICO'
            with self.assertRaisesRegex(ValueError,'Grid title/source changed'):r.binding(row)
        finally:r.close()

    def test_wrong_header_reference_and_category_conflict_rejected(self):
        row=self.row('2025','Gráfico 115');r=GridResolver()
        try:
            part=row['parte_grafico_imagen'];root=ET.fromstring(r.sources['2025'].xml.parts[part]);node=root.findall('.//c:ser',C)[int(row['serie_id'].split('_s')[1])-1]
            node.find('c:tx/c:strRef/c:f',C).text="'Gráfico 115'!$E$5"
            r.sources['2025'].xml.parts[part]=ET.tostring(root)
            with self.assertRaisesRegex(ValueError,'header is not aligned'):r.binding(row)
        finally:r.close()
        r=GridResolver()
        try:
            self.assertEqual(r.review(dict(row,periodo_categoria_literal='2099'))[0],'grid_category_conflict')
        finally:r.close()

    def test_literal_conflicts_proved_against_original_xml(self):
        files={b:ZipFile(ROOT/name) for b,(name,_,_) in BOOKS.items()};r=GridResolver();seen=set()
        try:
            with duckdb.connect(str(DB),read_only=True) as con:
                rows=con.execute('SELECT c.book,c.row_number,c.evidence,s.parte_grafico_imagen,s.serie_id,s.bloque FROM grid_conflicts c JOIN staging_records s USING(book,row_number)').fetchall()
                self.assertEqual(len(rows),1230)
                self.assertEqual(con.execute('SELECT count(*) FROM grid_conflicts JOIN grid_usable_keys USING(book,row_number)').fetchone()[0],0)
            for book,_,evidence,part,sid,block in rows:
                key=book,part,sid
                if key in seen:continue
                seen.add(key);e=json.loads(evidence)
                node=ET.fromstring(files[book].read(part)).findall('.//c:ser',C)[int(sid.split('_s')[1])-1]
                self.assertEqual(node.find('c:tx/c:v',C).text,e['series_name'])
                point=next(p for p in node.findall('c:val/c:numLit/c:pt',C) if int(p.get('idx'))==e['chart_point'])
                self.assertEqual(point.find('c:v',C).text,e['literal_value'])
                self.assertEqual(r.sources[book].text(block,e['name_cell']),e['series_name'])
                self.assertEqual(r.sources[book].number(block,e['source_cell']),Decimal(e['source_value']))
                self.assertNotEqual(Decimal(e['literal_value']),Decimal(e['source_value']))
            self.assertEqual(len(seen),30)
        finally:
            r.close()
            for f in files.values():f.close()

    def test_literal_values_are_not_fallback_candidates(self):
        r=GridResolver()
        try:
            row=self.row('2026','Gráfico 10')
            self.assertEqual(r.review(row)[0],'literal_values_not_reconciled')
            with self.assertRaises(LookupError):query(DB,book='2026',block='Gráfico 10',series_id=row['serie_id'],point_index=0)
        finally:r.close()

    def test_unit_scales_and_old_exclusions(self):
        with duckdb.connect(str(DB),read_only=True) as con:
            self.assertEqual(con.execute("SELECT count(*) FROM grid_observations WHERE block IN ('Gráfico 105','Gráfico 106') AND unit='percent_interest_rate_as_published'").fetchone()[0],984)
            self.assertEqual(con.execute("SELECT count(*) FROM grid_observations WHERE block IN ('Gráfico 105','Gráfico 106') AND (multiplier<>1 OR divisor<>1)").fetchone()[0],0)
            self.assertEqual(con.execute("SELECT count(*) FROM grid_observations WHERE lower(series_literal)='auxiliar' OR block IN ('Gráfico 12A','Gráfico 12B','Gráfico 12C')").fetchone()[0],0)
            self.assertEqual(con.execute("SELECT count(*) FROM grid_observations WHERE unit='percentage_points_gdp'").fetchone()[0],793)

    def test_partition_and_precise_query(self):
        with duckdb.connect(str(DB),read_only=True) as con:
            self.assertEqual(con.execute('SELECT count(*) FROM source_usable_keys').fetchone()[0],31705)
            self.assertEqual(con.execute('SELECT count(*) FROM grid_usable_keys').fetchone()[0],34416)
            self.assertEqual(con.execute('SELECT count(*) FROM grid_numeric_queue').fetchone()[0],12797)
            book,block,sid,point=con.execute('SELECT book,block,series_id,point_index FROM grid_observations LIMIT 1').fetchone()
        self.assertEqual(sha(PREVIOUS),PREVIOUS_SHA)
        self.assertEqual(query(DB,book=book,block=block,series_id=sid,point_index=point)['series_id'],sid)
        with self.assertRaises(ValueError):query(DB,book=book,block=block,series_id=sid,point_index=True)

    def test_changed_record_conflict_and_view_multiplicity_rejected(self):
        path=Path(self.temp.name)/'tampered.duckdb';shutil.copyfile(DB,path)
        with duckdb.connect(str(path)) as con:
            con.execute('BEGIN');con.execute("UPDATE grid_observations SET category_literal='2099' WHERE row_number=(SELECT min(row_number) FROM grid_observations)")
            with self.assertRaisesRegex(ValueError,'Grid evidence/table mismatch'):check_new(con)
            con.execute('ROLLBACK');con.execute('BEGIN');con.execute("UPDATE grid_conflicts SET evidence='{}'")
            with self.assertRaisesRegex(ValueError,'Grid evidence/table mismatch'):check_new(con)
            con.execute('ROLLBACK');con.execute('DROP VIEW grid_usable_keys')
            con.execute('CREATE VIEW grid_usable_keys AS SELECT * FROM source_usable_keys UNION ALL SELECT book,row_number FROM grid_observations UNION ALL SELECT book,row_number FROM grid_observations')
            with self.assertRaisesRegex(ValueError,'View multiplicity'):check_new(con)


if __name__=='__main__':unittest.main()
