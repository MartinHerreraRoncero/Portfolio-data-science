import unittest,tempfile,shutil,json,posixpath,xml.etree.ElementTree as ET
from pathlib import Path
from decimal import Decimal
from zipfile import ZipFile
import duckdb,openpyxl
from experiments.header_name_semantics import ROOT,HeaderResolver,check_new,query,SPECS
from experiments.excel_pilot import BOOKS
from experiments.bulk_semantics import C,ref_cells
DB=ROOT/'evaluation/header_name_semantics_2026-10-31/run_v3/header_names.duckdb'
M={'m':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
R='http://schemas.openxmlformats.org/officeDocument/2006/relationships'
class TestHeaderNames(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  if not DB.exists():raise unittest.SkipTest('Build header fixture first')
 def row(self):
  with duckdb.connect(str(DB),read_only=True) as c:
   q=c.execute("SELECT * FROM formula_numeric_queue WHERE book='2025' AND bloque='Gráfico 76' ORDER BY row_number LIMIT 1");return dict(zip([n[0] for n in q.description],q.fetchone()))
 def test_independent_xml_values_and_headers(self):
  with duckdb.connect(str(DB),read_only=True) as c:
   q=c.execute('SELECT * FROM header_observations');ns=[n[0] for n in q.description];rows=[dict(zip(ns,r)) for r in q.fetchall()]
  self.assertEqual(len(rows),239)
  for b in ['2025','2026']:
   book=openpyxl.load_workbook(ROOT/BOOKS[b][0],read_only=True,data_only=True);z=ZipFile(ROOT/BOOKS[b][0])
   try:
    rels={n.get('Id'):posixpath.normpath(posixpath.join('xl',n.get('Target'))) for n in ET.fromstring(z.read('xl/_rels/workbook.xml.rels'))}
    paths={n.get('name'):rels[n.get('{'+R+'}id')] for n in ET.fromstring(z.read('xl/workbook.xml')).findall('m:sheets/m:sheet',M)};raw={};values={}
    for r in rows:
     if r['book']!=b:continue
     sh=r['source_sheet']
     if sh not in raw:
      raw[sh]={n.get('r'):n for n in ET.fromstring(z.read(paths[sh])).findall('.//m:sheetData/m:row/m:c',M)}
      values[sh]={c.coordinate:c.value for rr in book[sh].iter_rows() for c in rr if c.value is not None}
     self.assertEqual(Decimal(raw[sh][r['source_cell']].find('m:v',M).text),r['raw_value']);self.assertEqual(r['raw_value'],r['value'])
     _,nc=ref_cells(r['name_ref']);self.assertEqual(str(values[sh][nc[0]]),r['series_literal'])
     _,cc=ref_cells(r['category_ref']);self.assertEqual(str(values[sh][cc[r['point_index']]]),r['category_literal'])
     self.assertIsNone(r['universal_series']);self.assertNotIn(r['series_literal'],['Histórico','Proyectado'])
     self.assertEqual(r['unit'],'years' if sh=='Gráfico 70' else 'percentage_points_gdp')
   finally:book.close();z.close()
 def test_header_and_caption_mutation_rejected(self):
  row=self.row()
  for cell in ['C4','B2']:
   r=HeaderResolver()
   try:
    r.sources['2025'].sheet('Gráfico 76')[1][cell]='OTRA MAGNITUD'
    self.assertIsNone(r.review_header(row)[1])
   finally:r.close()
 def test_reference_and_existing_name_rejected(self):
  row=self.row();r=HeaderResolver()
  try:
   self.assertIsNone(r.review_header(dict(row,serie_literal='Wrong'))[1])
   part=row['parte_grafico_imagen'];root=ET.fromstring(r.sources['2025'].xml.parts[part]);root.find('.//c:ser/c:val/c:numRef/c:f',C).text="'Gráfico 76'!$D$5:$D$80"
   r.sources['2025'].xml.parts[part]=ET.tostring(root)
   self.assertIsNone(r.review_header(row)[1])
  finally:r.close()
 def test_scale_mutation_rejected(self):
  row=self.row();r=HeaderResolver()
  try:
   part=row['parte_grafico_imagen'];root=ET.fromstring(r.sources['2025'].xml.parts[part]);root.find('.//c:valAx/c:numFmt',C).set('formatCode','0%');r.sources['2025'].xml.parts[part]=ET.tostring(root)
   self.assertIsNone(r.review_header(row)[1])
  finally:r.close()
 def test_partition_query_and_no_fallback(self):
  with duckdb.connect(str(DB),read_only=True) as c:
   self.assertEqual(c.execute('SELECT count(*) FROM header_usable_keys').fetchone()[0],38250);self.assertEqual(c.execute('SELECT count(*) FROM header_numeric_queue').fetchone()[0],8963)
  r=query(DB,book='2025',block='Gráfico 76',series_id='chart129_s1',point_index=1);self.assertEqual(r['series_literal'],'Diferencial interés-crecimiento')
  with self.assertRaises(ValueError):query(DB,book='2025',block='Gráfico 76',series_id='chart129_s1',point_index=True)
  with self.assertRaises(LookupError):query(DB,book='2025',block='Gráfico 76',series_id='wrong',point_index=1)
 def test_tampering_and_multiplicity_rejected(self):
  with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as d:
   p=Path(d)/'copy.duckdb';shutil.copyfile(DB,p)
   with duckdb.connect(str(p)) as c:
    c.execute('BEGIN');c.execute("UPDATE header_observations SET unit='wrong'")
    with self.assertRaises(ValueError):check_new(c)
    c.execute('ROLLBACK');c.execute('DROP VIEW header_usable_keys');c.execute('CREATE VIEW header_usable_keys AS SELECT * FROM formula_usable_keys UNION ALL SELECT book,row_number FROM header_observations UNION ALL SELECT book,row_number FROM header_observations')
    with self.assertRaises(ValueError):check_new(c)
if __name__=='__main__':unittest.main()
