"""Independent XML reads and adversarial unit/axis/category checks."""
import unittest,tempfile,shutil,json,posixpath,xml.etree.ElementTree as ET
from pathlib import Path
from decimal import Decimal,localcontext
from zipfile import ZipFile
import duckdb,openpyxl
from experiments.labour_unit_semantics import ROOT,LabourResolver,check_new,query,PREVIOUS,PREVIOUS_SHA
from experiments.excel_pilot import BOOKS,sha
from experiments.bulk_semantics import C,ref_cells
M={'m':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
R='http://schemas.openxmlformats.org/officeDocument/2006/relationships'
DB=ROOT/'evaluation/labour_unit_semantics_2026-10-31/run_v1/labour_units.duckdb'

class TestLabourUnits(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  if not DB.exists():raise unittest.SkipTest('Build labour fixture first')
 def row(self,book,block,label):
  with duckdb.connect(str(DB),read_only=True) as c:
   q=c.execute('SELECT * FROM header_numeric_queue WHERE book=? AND bloque=? AND serie_literal=? ORDER BY row_number LIMIT 1',[book,block,label]);v=q.fetchone();self.assertIsNotNone(v);return dict(zip([n[0] for n in q.description],v))
 def test_independent_xml_values_names_categories_and_conversion(self):
  with duckdb.connect(str(DB),read_only=True) as c:
   q=c.execute('SELECT * FROM labour_observations');ns=[n[0] for n in q.description];rows=[dict(zip(ns,v)) for v in q.fetchall()]
  self.assertEqual(len(rows),136)
  for book in ['2025','2026']:
   path=ROOT/BOOKS[book][0];z=ZipFile(path);b=openpyxl.load_workbook(path,read_only=True,data_only=True)
   try:
    rels={n.get('Id'):posixpath.normpath(posixpath.join('xl',n.get('Target'))) for n in ET.fromstring(z.read('xl/_rels/workbook.xml.rels'))}
    paths={n.get('name'):rels[n.get('{'+R+'}id')] for n in ET.fromstring(z.read('xl/workbook.xml')).findall('m:sheets/m:sheet',M)};raw={};values={};charts={}
    for r in rows:
     if r['book']!=book:continue
     sh=r['source_sheet']
     if sh not in raw:
      raw[sh]={n.get('r'):n for n in ET.fromstring(z.read(paths[sh])).findall('.//m:sheetData/m:row/m:c',M)}
      values[sh]={c.coordinate:c.value for rr in b[sh].iter_rows() for c in rr if c.value is not None}
     self.assertEqual(Decimal(raw[sh][r['source_cell']].find('m:v',M).text),r['raw_value'])
     with localcontext() as ctx:ctx.prec=60;self.assertEqual(r['raw_value']*r['multiplier']/r['divisor'],r['value'])
     _,nc=ref_cells(r['name_ref']);self.assertEqual(str(values[sh][nc[0]]),r['series_literal'])
     _,cc=ref_cells(r['category_ref']);self.assertEqual(str(values[sh][cc[r['point_index']]]),r['category_literal'])
     if r['chart_part'] not in charts:charts[r['chart_part']]=ET.fromstring(z.read(r['chart_part']))
     node=charts[r['chart_part']].findall('.//c:ser',C)[int(r['series_id'].split('_s')[1])-1]
     self.assertEqual(node.find('c:val/c:numRef/c:f',C).text,r['value_ref'])
     if r['category_basis']=='reviewed_source_grid':
      self.assertIsNone(node.find('c:cat',C));self.assertIsNone(r['universal_category'])
      self.assertEqual(cc[r['point_index']][1:],r['source_cell'][1:]);self.assertTrue(cc[r['point_index']].startswith('B'))
      self.assertEqual(json.loads(r['grid_evidence'])['origin'],'reviewed_original_grid')
   finally:b.close();z.close()
 def test_mixed_smi_units_associated_with_correct_axis(self):
  with duckdb.connect(str(DB),read_only=True) as c:
   result=c.execute("SELECT series_literal,unit,multiplier,count(*) FROM labour_observations WHERE block='Gráfico 31' GROUP BY ALL ORDER BY unit").fetchall()
  self.assertEqual(result,[('SMI mensual (14 pagas)','euros_monthly_14_payments',Decimal(1),8),('Subida acumulada desde 2018','percent_cumulative_smi_change_from_2018',Decimal(100),8)])
  for label in ['SMI mensual (14 pagas)','Subida acumulada desde 2018']:
   row=self.row('2025','Gráfico 31',label);r=LabourResolver()
   try:
    part=row['parte_grafico_imagen'];root=ET.fromstring(r.sources['2025'].xml.parts[part]);node=root.findall('.//c:ser',C)[int(row['serie_id'].split('_s')[1])-1]
    group=next(p for p in root.iter() if node in list(p));ids={n.get('val') for n in group.findall('c:axId',C)}
    axis=next(a for a in root.findall('.//c:valAx',C) if a.find('c:axId',C).get('val') in ids)
    axis.find('c:numFmt',C).set('formatCode','0%' if label.startswith('SMI') else '0')
    r.sources['2025'].xml.parts[part]=ET.tostring(root)
    self.assertIsNone(r.review_labour(row)[1])
   finally:r.close()
 def test_grid_recovery_preserves_original_and_rejects_contradiction(self):
  row=self.row('2026','Gráfico 31.A.','Déficit primario y stock-flujo');self.assertIsNone(row['periodo_categoria_literal']);r=LabourResolver()
  try:
   status,result=r.review_labour(row);self.assertEqual(status,'usable_labour_unit');self.assertEqual(result[7],'2025');self.assertIsNone(result[36])
   self.assertIsNone(r.review_labour(dict(row,periodo_categoria_literal='2099'))[1])
  finally:r.close()
  r=LabourResolver()
  try:
   r.sources['2026'].sheet('Gráfico 31.A.')[1]['B11']=None
   self.assertIsNone(r.review_labour(row)[1])
  finally:r.close()
  with duckdb.connect(str(DB),read_only=True) as c:self.assertEqual(c.execute("SELECT count(*) FROM labour_observations WHERE category_basis='reviewed_source_grid'").fetchone()[0],56)
 def test_title_pdf_and_header_reference_changes_rejected(self):
  row=self.row('2026','Gráfico 31.A.','Déficit primario y stock-flujo')
  for mutation in ['title','pdf','name_ref']:
   r=LabourResolver()
   try:
    if mutation=='title':r.sources['2026'].sheet('Gráfico 31.A.')[1]['B3']='OTRA MAGNITUD'
    elif mutation=='pdf':r.pdf_pages['2026',71]='Contribuciones sin unidad acreditada'
    else:
     part=row['parte_grafico_imagen'];root=ET.fromstring(r.sources['2026'].xml.parts[part]);node=root.findall('.//c:ser',C)[int(row['serie_id'].split('_s')[1])-1]
     node.find('c:tx/c:strRef/c:f',C).text="'Gráfico 31.A.'!$H$6";r.sources['2026'].xml.parts[part]=ET.tostring(root)
    self.assertIsNone(r.review_labour(row)[1])
   finally:r.close()
 def test_participation_scope_periods_and_nominal_real_distinction(self):
  with duckdb.connect(str(DB),read_only=True) as c:
   self.assertEqual(c.execute("SELECT count(*) FROM labour_observations WHERE block='Gráfico 24' AND semantic_scope='literal_plot_display_only' AND category_semantics='age_group_as_labelled'").fetchone()[0],10)
   self.assertEqual(c.execute("SELECT count(*) FROM labour_observations WHERE block='Gráfico 31.B.' AND category_semantics='period_as_labelled'").fetchone()[0],6)
   self.assertEqual(c.execute("SELECT unit,count(*) FROM labour_observations WHERE block='Gráfico 32' GROUP BY unit ORDER BY unit").fetchall(),[('percent_nominal_smi_growth',7),('percent_real_smi_growth_cpi_deflated',7)])
   self.assertEqual(c.execute("SELECT count(*) FROM labour_observations WHERE unit='percentage_points_gdp' AND multiplier=1").fetchone()[0],84)
 def test_partition_exact_query_and_old_exclusions(self):
  with duckdb.connect(str(DB),read_only=True) as c:
   self.assertEqual(c.execute('SELECT count(*) FROM header_usable_keys').fetchone()[0],38250)
   self.assertEqual(c.execute('SELECT count(*) FROM labour_usable_keys').fetchone()[0],38386)
   self.assertEqual(c.execute('SELECT count(*) FROM labour_numeric_queue').fetchone()[0],8827)
   self.assertEqual(c.execute('SELECT count(*) FROM reviewed_scatter_pairs JOIN labour_usable_keys USING(book,row_number)').fetchone()[0],0)
   self.assertEqual(c.execute('SELECT count(*) FROM grid_conflicts JOIN labour_usable_keys USING(book,row_number)').fetchone()[0],0)
  self.assertEqual(sha(PREVIOUS),PREVIOUS_SHA)
  r=query(DB,book='2025',block='Gráfico 31',series_id='chart57_s1',point_index=0);self.assertEqual(r['unit'],'euros_monthly_14_payments')
  with self.assertRaises(ValueError):query(DB,book='2025',block='Gráfico 31',series_id='chart57_s1',point_index=True)
  with self.assertRaises(LookupError):query(DB,book='2025',block='Gráfico 31',series_id='unknown',point_index=0)
 def test_tampering_and_view_multiplicity(self):
  with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as d:
   p=Path(d)/'copy.duckdb';shutil.copyfile(DB,p)
   with duckdb.connect(str(p)) as c:
    c.execute('BEGIN');c.execute("UPDATE labour_observations SET unit='percentage_points_gdp' WHERE block='Gráfico 31'")
    with self.assertRaisesRegex(ValueError,'Labour evidence mismatch'):check_new(c)
    c.execute('ROLLBACK');c.execute('BEGIN');c.execute("UPDATE labour_pdf_evidence SET text='missing unit'")
    with self.assertRaisesRegex(ValueError,'Labour evidence mismatch'):check_new(c)
    c.execute('ROLLBACK');c.execute('DROP VIEW labour_usable_keys')
    c.execute('CREATE VIEW labour_usable_keys AS SELECT * FROM header_usable_keys UNION ALL SELECT book,row_number FROM labour_observations UNION ALL SELECT book,row_number FROM labour_observations')
    with self.assertRaisesRegex(ValueError,'View multiplicity'):check_new(c)

if __name__=='__main__':unittest.main()
