import unittest,tempfile,shutil
from pathlib import Path
from decimal import Decimal
import xml.etree.ElementTree as ET,posixpath
from zipfile import ZipFile
import duckdb,openpyxl
from experiments.formula_name_semantics import ROOT,FormulaResolver,check_new,query
from experiments.excel_pilot import BOOKS
DB=ROOT/'evaluation/formula_name_semantics_2026-10-31/run_v2/formula_names.duckdb'
class TestFormulaNames(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  if not DB.exists():raise unittest.SkipTest('Build formula fixture first')
 def row(self):
  with duckdb.connect(str(DB),read_only=True) as c:
   q=c.execute("SELECT * FROM quantity_numeric_queue WHERE book='2025' AND bloque='Gráfico 27' ORDER BY row_number LIMIT 1");return dict(zip([n[0] for n in q.description],q.fetchone()))
 def test_independent_values_names_and_categories(self):
  b=openpyxl.load_workbook(ROOT/BOOKS['2025'][0],read_only=True,data_only=True)
  try:
   vals={c.coordinate:c.value for rr in b['Gráfico 27'].iter_rows() for c in rr if c.value is not None}
   with duckdb.connect(str(DB),read_only=True) as c:
    q=c.execute('SELECT * FROM formula_observations');ns=[n[0] for n in q.description];rows=[dict(zip(ns,r)) for r in q.fetchall()]
   self.assertEqual(len(rows),81)
   with ZipFile(ROOT/BOOKS['2025'][0]) as z:
    m={'m':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    rels={n.get('Id'):posixpath.normpath(posixpath.join('xl',n.get('Target'))) for n in ET.fromstring(z.read('xl/_rels/workbook.xml.rels'))}
    part=next(rels[n.get('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id')] for n in ET.fromstring(z.read('xl/workbook.xml')).findall('m:sheets/m:sheet',m) if n.get('name')=='Gráfico 27')
    raw={n.get('r'):n for n in ET.fromstring(z.read(part)).findall('.//m:sheetData/m:row/m:c',m)}
   from experiments.bulk_semantics import ref_cells
   for r in rows:
    self.assertEqual(Decimal(raw[r['source_cell']].find('m:v',m).text),r['raw_value'])
    _,cc=ref_cells(r['category_ref']);self.assertEqual(str(vals[cc[r['point_index']]]),r['category_literal'])
    self.assertEqual(vals[{'chart51_s1':'C4','chart51_s2':'D4','chart51_s3':'E4'}[r['series_id']]],r['series_literal'])
    self.assertEqual(r['raw_value'],r['value']);self.assertEqual(r['unit'],'percent_growth_as_published')
  finally:b.close()
 def test_external_or_general_formula_rejected(self):
  r=FormulaResolver()
  try:
   for f in ['=F4&" other"','=SUM(F4)','=OTHER!F4&" observ"']:
    with self.assertRaises(ValueError):r.formula_name(r.sources['2025'],'Gráfico 27','C4',f)
  finally:r.close()
 def test_changed_string_cache_rejected(self):
  r=FormulaResolver();row=self.row()
  try:
   from experiments.semantic_families import M
   r.sources['2025'].sheet('Gráfico 27')[0]['C4'].find('m:v',M).text='PIB real observ'
   self.assertIsNone(r.review_name(row)[1])
  finally:r.close()
 def test_caption_unit_change_rejected(self):
  r=FormulaResolver();row=self.row()
  try:
   r.sources['2025'].sheet('Gráfico 27')[1]['L5']='OTRA MAGNITUD SIN UNIDAD'
   self.assertIsNone(r.review_name(row)[1])
  finally:r.close()
 def test_wrong_series_identity_rejected(self):
  r=FormulaResolver();row=self.row();row['serie_literal']='PIB real observ'
  try:self.assertIsNone(r.review_name(row)[1])
  finally:r.close()
 def test_partition_and_query(self):
  with duckdb.connect(str(DB),read_only=True) as c:
   self.assertEqual(c.execute('SELECT count(*) FROM formula_usable_keys').fetchone()[0],38011)
   self.assertEqual(c.execute('SELECT count(*) FROM formula_numeric_queue').fetchone()[0],9202)
  self.assertEqual(query(DB,book='2025',block='Gráfico 27',series_id='chart51_s1',point_index=0)['series_literal'],'PIB nominal observ')
  with self.assertRaises(ValueError):query(DB,book='2025',block='Gráfico 27',series_id='chart51_s1',point_index=True)
  with self.assertRaises(LookupError):query(DB,book='2025',block='Gráfico 27',series_id='wrong',point_index=0)
 def test_mutation_and_multiplicity(self):
  with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as d:
   p=Path(d)/'copy.duckdb';shutil.copyfile(DB,p)
   with duckdb.connect(str(p)) as c:
    c.execute('BEGIN');c.execute("UPDATE formula_observations SET series_literal='wrong'")
    with self.assertRaises(ValueError):check_new(c)
    c.execute('ROLLBACK');c.execute('DROP VIEW formula_usable_keys')
    c.execute('CREATE VIEW formula_usable_keys AS SELECT * FROM quantity_usable_keys UNION ALL SELECT book,row_number FROM formula_observations UNION ALL SELECT book,row_number FROM formula_observations')
    with self.assertRaises(ValueError):check_new(c)
if __name__=='__main__':unittest.main()
