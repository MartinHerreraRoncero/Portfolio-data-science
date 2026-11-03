import unittest,json,tempfile,shutil,posixpath,xml.etree.ElementTree as ET
from pathlib import Path
from decimal import Decimal
from zipfile import ZipFile
from unittest.mock import patch
import duckdb
from experiments import ready_recovery_semantics as m
from experiments.excel_pilot import BOOKS
DB=m.ROOT/'evaluation/ready_recovery_2026-10-31/run_v6/ready_recovery.duckdb'
M={'m':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
R='http://schemas.openxmlformats.org/officeDocument/2006/relationships'
C={'c':'http://schemas.openxmlformats.org/drawingml/2006/chart'}
class XML:
 def __init__(self,path):
  self.z=ZipFile(path);rels={n.get('Id'):posixpath.normpath(posixpath.join('xl',n.get('Target'))) for n in ET.fromstring(self.z.read('xl/_rels/workbook.xml.rels'))}
  self.paths={n.get('name'):rels[n.get('{'+R+'}id')] for n in ET.fromstring(self.z.read('xl/workbook.xml')).findall('m:sheets/m:sheet',M)};self.cache={}
  self.strings=[''.join(n.itertext()) for n in ET.fromstring(self.z.read('xl/sharedStrings.xml')).findall('m:si',M)] if 'xl/sharedStrings.xml' in self.z.namelist() else []
 def node(self,s,a):
  if s not in self.cache:self.cache[s]={n.get('r'):n for n in ET.fromstring(self.z.read(self.paths[s])).findall('.//m:sheetData/m:row/m:c',M)}
  return self.cache[s][a]
 def value(self,s,a):
  n=self.node(s,a);v=n.find('m:v',M)
  if n.get('t')=='s':return self.strings[int(v.text)]
  if n.get('t')=='inlineStr':return ''.join(n.find('m:is',M).itertext())
  return Decimal(v.text)
class TestReadyRecovery(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  if not DB.exists():raise unittest.SkipTest('Build final ready fixture first')
  with duckdb.connect(str(DB),read_only=True) as c:
   q=c.execute('SELECT * FROM ready_observations');cls.rows=[dict(zip([n[0] for n in q.description],v)) for v in q.fetchall()]
 def candidate(self,number=None,book='2025',block=None):
  with duckdb.connect(str(DB),read_only=True) as c:
   q=c.execute('SELECT * FROM staging_records WHERE book=? AND '+('row_number=?' if number is not None else 'bloque=? AND (book,row_number) IN (SELECT book,row_number FROM ready_observations)')+' ORDER BY row_number LIMIT 1',[book,number if number is not None else block]);return dict(zip([n[0] for n in q.description],q.fetchone()))
 def test_independent_xml_all117_values_categories_names_and_formulas(self):
  self.assertEqual(len(self.rows),117);readers={b:XML(m.ROOT/BOOKS[b][0]) for b in ['2025','2026']}
  try:
   for r in self.rows:
    x=readers[r['book']];s=r['source_sheet'];raw=x.value(s,r['source_cell']);self.assertEqual(raw,r['raw_value']);self.assertEqual(raw*r['multiplier']/r['divisor'],r['value'])
    if r['series_id']=='g37_variaciones':
     a,b,ca,cb,category,index=m.DIFFERENCES[r['source_cell']];self.assertEqual(x.node(s,r['source_cell']).find('m:f',M).text,a+'-'+b);self.assertEqual(r['unit'],'percentage_points_gdp');self.assertEqual(r['multiplier'],1);self.assertLessEqual(abs(raw-(x.value(s,a)-x.value(s,b))),Decimal('1e-14'));self.assertEqual(r['category_literal'],f'{int(x.value(s,ca))} menos {int(x.value(s,cb))}');continue
    part=ET.fromstring(x.z.read(r['chart_part']));node=part.findall('.//c:ser',C)[int(r['series_id'].split('_s')[1])-1]
    self.assertEqual(node.find('c:val/c:numRef/c:f',C).text,r['value_ref']);self.assertEqual(node.find('c:tx/c:strRef/c:f',C).text,r['name_ref'])
    sheet,cells=m.ref_cells(r['name_ref']);self.assertEqual(x.value(sheet,cells[0]),r['series_literal'])
    sheet,cells=m.ref_cells(r['category_ref']);v=x.value(sheet,cells[r['point_index']]);category=str(int(v)) if isinstance(v,Decimal) and v==int(v) else str(v);self.assertEqual(category,r['category_literal'])
    if r['universal_category'] is None:self.assertEqual(r['category_basis'],'reviewed_source_grid')
    else:self.assertEqual(category.replace('\n',' / '),r['universal_category'])
    if s=='Gráfico 12' and r['book']=='2026':
     target=r['source_cell'][:-2]+'17';formula=x.node(s,r['source_cell']).find('m:f',M)
     if formula.text is None:
      masters=[(a,n.find('m:f',M)) for a,n in x.cache[s].items() if n.find('m:f',M) is not None and n.find('m:f',M).text is not None and n.find('m:f',M).get('si')==formula.get('si')]
      self.assertEqual(formula.get('t'),'shared');self.assertEqual(len(masters),1);anchor,master=masters[0];self.assertEqual(master.get('t'),'shared');self.assertIsNotNone(master.get('si'));self.assertEqual(master.text,'-'+anchor[:-2]+'17');self.assertEqual(anchor,'F18');self.assertEqual(master.get('ref'),'F18:BC18')
     else:self.assertEqual(formula.text,'-'+target)
     self.assertEqual(raw,-x.value(s,target));self.assertEqual(r['unit'],'percent_gdp')
  finally:
   for x in readers.values():x.z.close()
 def test_independent_quarantine_source_contradiction(self):
  x=XML(m.ROOT/BOOKS['2025'][0]);sheet='Gráfico RE 3.4'
  try:
   self.assertEqual(x.value(sheet,'D16'),'Demorada');self.assertEqual(x.value(sheet,'D7'),'JUBILACIÓN ANTICIPADA');self.assertIn('ANTICIPADAS',x.value(sheet,'C2'))
   node=ET.fromstring(x.z.read('xl/charts/chart91.xml')).findall('.//c:ser',C)[0]
   self.assertEqual(node.find('c:tx/c:strRef/c:f',C).text,"'Gráfico RE 3.4'!$D$16");self.assertEqual(node.find('c:val/c:numRef/c:f',C).text,"'Gráfico RE 3.4'!$F$7:$R$7")
   with duckdb.connect(str(DB),read_only=True) as c:
    rows=c.execute('SELECT row_number,evidence FROM ready_quarantine ORDER BY row_number').fetchall()
   self.assertEqual([r[0] for r in rows],list(range(18313,18326)))
   for _,proof in rows:
    e=json.loads(proof);self.assertEqual(x.value(sheet,e['source_cell']),Decimal(e['raw_value']))
  finally:x.z.close()
 def test_partition_and_full_new_layer(self):
  with duckdb.connect(str(DB),read_only=True) as c:
   self.assertEqual(len(m.check_new(c)),117);self.assertEqual(c.execute('SELECT count(*) FROM ready_usable_keys').fetchone()[0],38492);self.assertEqual(c.execute('SELECT count(*) FROM ready_numeric_queue').fetchone()[0],8721)
   self.assertEqual(c.execute('SELECT book,row_number FROM labour_usable_keys EXCEPT ALL SELECT book,row_number FROM ready_usable_keys ORDER BY row_number').fetchall(),[('2025',i) for i in range(18313,18324)])
   self.assertEqual(c.execute('SELECT count(*) FROM ready_quarantine').fetchone()[0],13)
   self.assertFalse(c.execute('SELECT book,row_number FROM ready_numeric_queue INTERSECT SELECT book,row_number FROM ready_usable_keys').fetchall())
   self.assertEqual(c.execute('SELECT count(*) FROM (SELECT book,row_number FROM ready_numeric_queue UNION SELECT book,row_number FROM ready_usable_keys)').fetchone()[0],47213)
   blocked=c.execute("SELECT row_number,status FROM ready_review WHERE status<>'usable_ready' ORDER BY row_number").fetchall();self.assertEqual([r[0] for r in blocked],[18324,18325]);self.assertTrue(all('literal_series_source_conflict' in r[1] for r in blocked))
 def test_physical_row_conflict_never_promoted(self):
  r=m.ReadyResolver()
  try:
   for i in range(18313,18326):
    row=self.candidate(i);self.assertEqual(row['serie_literal'],'Demorada')
    if i>=18324:self.assertIsNone(r.review_ready(row)[1])
    with self.assertRaises(LookupError):m.query(DB,book='2025',block='Gráfico RE 3.4',series_id='chart91_s1',point_index=i-18313)
   self.assertEqual(r.sources['2025'].text('Gráfico RE 3.4','D7'),'JUBILACIÓN ANTICIPADA')
  finally:r.close()
 def test_grid_axes_refs_names_and_caption_fail_closed(self):
  row=self.candidate(book='2026',block='Gráfico 12')
  for kind in ['axis','ref','name','category','caption','formula']:
   r=m.ReadyResolver()
   try:
    self.assertIsNotNone(r.review_ready(row)[1],'Valid baseline before mutation');src=r.sources['2026'];part=row['parte_grafico_imagen'];root=ET.fromstring(src.xml.parts[part]);n=root.findall('.//c:ser',C)[3]
    if kind=='axis':root.find('.//c:valAx/c:numFmt',C).set('formatCode','0%')
    elif kind=='ref':n.find('c:val/c:numRef/c:f',C).text="'Gráfico 12'!$E$17:$AI$17"
    elif kind=='name':src.sheet('Gráfico 12')[1]['D18']='Otra serie'
    elif kind=='category':src.sheet('Gráfico 12')[1]['E5']=None
    elif kind=='caption':src.sheet('Gráfico 12')[1]['B3']='OTRA MAGNITUD'
    else:src.sheet('Gráfico 12')[1]['E18']='=E17'
    src.xml.parts[part]=ET.tostring(root);self.assertIsNone(r.review_ready(row)[1],kind)
   finally:r.close()
 def test_category_and_universal_identity_mutations(self):
  for field,value in [('periodo_categoria_literal','Media / otro'),('serie_literal','Otra serie'),('decimal_XML_texto','999'),('hoja_valor','Gráfico 16'),('unidad','%')]:
   r=m.ReadyResolver()
   try:
    row=self.candidate(12191);self.assertIsNotNone(r.review_ready(row)[1]);self.assertIsNone(r.review_ready(dict(row,**{field:value}))[1],field)
   finally:r.close()
 def test_difference_formula_period_unit_and_cache_mutations(self):
  for kind in ['formula','period','fractional_period','unit','cache','id']:
   row=self.candidate(51057);r=m.ReadyResolver()
   try:
    self.assertIsNotNone(r.review_ready(row)[1]);src=r.sources['2025']
    if kind=='formula':src.sheet('Gráfico 37')[1]['Q3']='=Q6+K6'
    elif kind=='period':src.sheet('Gráfico 37')[0]['Q4'].find('m:v',M).text='2030'
    elif kind=='fractional_period':src.sheet('Gráfico 37')[0]['Q4'].find('m:v',M).text='2029.5'
    elif kind=='unit':row['unidad']='% PIB'
    elif kind=='id':row['id']='g37_AL3'
    else:src.sheet('Gráfico 37')[0]['Q3'].find('m:v',M).text='1'
    self.assertIsNone(r.review_ready(row)[1],kind)
   finally:r.close()
 def test_query_exact_identity_and_no_fallback(self):
  r=self.rows[0];self.assertEqual(m.query(DB,book=r['book'],block=r['block'],series_id=r['series_id'],point_index=r['point_index'])['row_number'],r['row_number'])
  with self.assertRaises(LookupError):m.query(DB,book=r['book'],block='otro',series_id=r['series_id'],point_index=r['point_index'])
  with self.assertRaises(ValueError):m.query(DB,book=r['book'],block=r['block'],series_id=r['series_id'],point_index=True)
 def test_ready_hashes_and_no_unselected_admission(self):
  with patch.object(m,'REGISTRY_SHA','0'*64):
   with self.assertRaises(ValueError):m.selected()
  self.assertEqual(len(m.selected()),119);self.assertEqual({(r['book'],r['row_number']) for r in self.rows},m.selected()-{('2025',18324),('2025',18325)})
 def test_tampered_values_and_views_detected(self):
  with tempfile.TemporaryDirectory(dir=m.ROOT/'tmp') as directory:
   db=Path(directory)/'tampered.duckdb';shutil.copyfile(DB,db)
   with duckdb.connect(str(db)) as c:c.execute('UPDATE ready_observations SET value=value+1 WHERE row_number=51057')
   with duckdb.connect(str(db),read_only=True) as c:
    with self.assertRaises(ValueError):m.check_new(c)
   shutil.copyfile(DB,db)
   with duckdb.connect(str(db)) as c:c.execute('CREATE OR REPLACE VIEW ready_usable_keys AS SELECT * FROM labour_usable_keys')
   with duckdb.connect(str(db),read_only=True) as c:
    with self.assertRaises(ValueError):m.check_new(c)
if __name__=='__main__':unittest.main()
