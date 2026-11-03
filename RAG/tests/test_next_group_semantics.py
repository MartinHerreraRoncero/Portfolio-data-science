import unittest,json,tempfile,shutil,xml.etree.ElementTree as ET
from pathlib import Path
from decimal import Decimal
from unittest.mock import patch
import duckdb
from test_ready_recovery_semantics import XML,M
from experiments import next_group_semantics as m
from experiments.excel_pilot import BOOKS
DB=m.ROOT/'evaluation/next_group_semantics_2026-10-31/run_v2/next_groups.duckdb'
C={'c':'http://schemas.openxmlformats.org/drawingml/2006/chart'}
class TestNextGroups(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  if not DB.exists():raise RuntimeError('Build next group fixture first')
  with duckdb.connect(str(DB),read_only=True)as c:
   q=c.execute('SELECT * FROM next_observations ORDER BY book,row_number');ns=[n[0]for n in q.description];cls.rows=[dict(zip(ns,v))for v in q.fetchall()]
 def candidate(self,book,block,sid):
  with duckdb.connect(str(DB),read_only=True)as c:
   q=c.execute('SELECT * FROM blocked_numeric_queue WHERE book=? AND bloque=? AND serie_id=? ORDER BY row_number LIMIT 1',[book,block,sid]);v=q.fetchone();self.assertIsNotNone(v);return dict(zip([n[0]for n in q.description],v))
 def test_independent_xml_all619_values_categories_series_units_and_scale(self):
  self.assertEqual(len(self.rows),619);readers={b:XML(m.ROOT/BOOKS[b][0])for b in['2025','2026']}
  try:
   for r in self.rows:
    x=readers[r['book']];sheet=r['source_sheet'];raw=x.value(sheet,r['source_cell']);self.assertEqual(raw,r['raw_value']);divisor=1000 if sheet=='Anexo I. Gr 15A'else 1;self.assertEqual(r['divisor'],divisor);self.assertEqual(r['multiplier'],1);self.assertEqual(raw/Decimal(divisor),r['value'])
    root=ET.fromstring(x.z.read(r['chart_part']));node=root.findall('.//c:ser',C)[int(r['series_id'].split('_s')[1])-1];self.assertEqual(node.find('c:val/c:numRef/c:f',C).text,r['value_ref']);values={int(n.get('idx')):Decimal(n.find('c:v',C).text)for n in node.findall('c:val/c:numRef/c:numCache/c:pt',C)};self.assertEqual(values[r['point_index']],raw)
    sh,cc=m.ref_cells(r['category_ref']);v=x.value(sh,cc[r['point_index']]);cat=str(int(v))if isinstance(v,Decimal)and v==int(v)else str(v);self.assertEqual(cat,r['category_literal'])
    if r['universal_category']is None:self.assertEqual(r['category_basis'],'reviewed_source_grid')
    else:self.assertEqual(cat.replace('\n',' / '),r['universal_category'])
    ns,nc=m.ref_cells(r['name_ref']);self.assertEqual(x.value(ns,nc[0]),r['series_literal'])
    if sheet=='Gráfico 48':self.assertIsNone(r['universal_series']);self.assertEqual(r['series_literal'],'Histórico');self.assertEqual(r['source_cell'][-1],'6');self.assertIsNone(node.find('c:tx',C));self.assertEqual(r['unit'],'percent_pension_average_over_average_wage')
    else:self.assertEqual(r['series_literal'],r['universal_series'])
    proof=json.loads(r['quantity_evidence']);self.assertEqual(proof['source_category'],r['category_literal']);self.assertIsNotNone(proof['universal_metric'])
    if sheet=='Gráfico 56':self.assertEqual(r['metric_literal'],x.value(sheet,'E4'));self.assertEqual(r['unit'],'percent_gdp');self.assertNotEqual(r['metric_literal'],proof['universal_metric']);self.assertIn('TIPO DE INTERÉS',proof['universal_metric'])
    if sheet=='Gráfico 78':self.assertEqual(r['unit'],'percent_gdp');self.assertIn('gross_treasury',r['quantity_role'])
    if sheet=='Anexo I. Gr 12B':self.assertEqual(r['quantity_role'],'remaining_life_expectancy_at_age65');self.assertEqual(r['unit'],'years')
    if sheet=='Gráfico 13B':self.assertEqual(r['quantity_role'],'resident_population_age15_74');self.assertEqual(r['unit'],'thousand_persons')
  finally:
   for x in readers.values():x.z.close()
 def test_full_new_recalculation_preserved_partition_and_reasons(self):
  with duckdb.connect(str(DB),read_only=True)as c:
   self.assertEqual(len(m.check_new(c)),619);self.assertEqual(c.execute('SELECT count(*)FROM next_usable_keys').fetchone()[0],40498);self.assertEqual(c.execute('SELECT count(*)FROM next_numeric_queue').fetchone()[0],6715);self.assertFalse(c.execute('SELECT * FROM blocked_usable_keys EXCEPT ALL SELECT * FROM next_usable_keys').fetchall());self.assertFalse(c.execute('SELECT book,row_number FROM next_numeric_queue INTERSECT SELECT * FROM next_usable_keys').fetchall());self.assertFalse(c.execute('SELECT book,row_number FROM ready_quarantine INTERSECT SELECT * FROM next_usable_keys').fetchall());self.assertFalse(c.execute('SELECT n.book,n.row_number FROM next_numeric_queue n JOIN blocked_numeric_queue p USING(book,row_number) WHERE n.next_reason IS DISTINCT FROM p.blocked_reason').fetchall())
 def test_no_helper_forecast_marker_or_unreviewed_series_admitted(self):
  self.assertNotIn(('2025','Gráfico 17','chart38_s5'),m.PROFILES)
  with duckdb.connect(str(DB),read_only=True)as c:
   self.assertFalse(c.execute("SELECT t.book,t.row_number FROM next_observations t JOIN blocked_numeric_queue s USING(book,row_number) WHERE s.blocked_reason IN('drawing_helper_not_economic_series','forecast_marker_not_measure') OR s.serie_id='chart38_s5'").fetchall());self.assertEqual(c.execute('SELECT count(*)FROM ready_quarantine').fetchone()[0],13);self.assertEqual(c.execute('SELECT reason,count(*)FROM next_deferrals GROUP BY ALL ORDER BY reason').fetchall(),[('monetary_MM_euro_unit_contract_pending',76),('population_identity_contract_pending',55)]);self.assertFalse(c.execute('SELECT book,row_number FROM next_deferrals INTERSECT SELECT book,row_number FROM next_observations').fetchall())
 def test_independent_labour_decomposition_twelve_rows_and_roles(self):
  x=XML(m.ROOT/BOOKS['2025'][0])
  try:
   for rr in range(5,18):
    if rr==15:continue
    total=x.value('Gráfico 17','C'+str(rr));components=[x.value('Gráfico 17',col+str(rr))for col in'DEFG'];self.assertLessEqual(abs(sum(components,Decimal(0))-total),Decimal('1e-14'))
   labour=[r for r in self.rows if r['block']=='Gráfico 17'];self.assertEqual(len(labour),60);self.assertEqual(sum(r['quantity_role']=='potential_real_output_growth_labour_total'for r in labour),12);self.assertTrue(all(r['unit']=='percentage_points_potential_real_output_growth'for r in labour));self.assertFalse(any(r['source_cell'][1:]=='15'for r in labour));self.assertEqual(sum(r['universal_category']!=r['category_literal']for r in labour),10)
  finally:x.z.close()
 def test_grid_mutations_on_positive_baseline_fail_closed(self):
  row=self.candidate('2026','Gráfico 13A','chart19_s5')
  for kind in['ref','name','category','cache','axis','peer']:
   r=m.NextResolver()
   try:
    self.assertIsNotNone(r.review_next(row)[1]);src=r.sources['2026'];part=row['parte_grafico_imagen'];root=ET.fromstring(src.xml.parts[part]);node=root.findall('.//c:ser',C)[4]
    if kind=='ref':node.find('c:val/c:numRef/c:f',C).text="'Gráfico 13A'!$H$22:$H$82"
    elif kind=='name':src.sheet(row['bloque'])[1]['G21']='Otra entidad'
    elif kind=='category':src.sheet(row['bloque'])[1]['B22']=None
    elif kind=='cache':node.find('c:val/c:numRef/c:numCache/c:pt/c:v',C).text='999'
    elif kind=='axis':root.find('.//c:valAx/c:numFmt',C).set('formatCode','0%')
    else:
     for n in root.findall('.//c:ser',C):
      cat=n.find('c:cat',C)
      if cat is not None:n.remove(cat)
    src.xml.parts[part]=ET.tostring(root);r.bindings.clear();self.assertIsNone(r.review_next(row)[1],kind)
   finally:r.close()
 def test_migration_divisor_and_signed_value_preserved(self):
  row=self.candidate('2026','Anexo I. Gr 15A','chart73_s5');r=m.NextResolver()
  try:
   status,values=r.review_next(row);self.assertIsNotNone(values);d=dict(zip(m.FIELDS,values));self.assertEqual(d['value'],d['raw_value']/1000);self.assertEqual(d['quantity_role'],'annual_net_external_migration_flow');src=r.sources['2026'];part=row['parte_grafico_imagen'];root=ET.fromstring(src.xml.parts[part]);root.find('.//c:valAx/c:dispUnits/c:builtInUnit',C).set('val','millions');src.xml.parts[part]=ET.tostring(root);r.bindings.clear();self.assertIsNone(r.review_next(row)[1])
  finally:r.close()
 def test_historical_recovered_name_is_value_row_aligned(self):
  row=self.candidate('2025','Gráfico 48','chart85_s3');r=m.NextResolver()
  try:
   self.assertIsNotNone(r.review_next(row)[1]);r.sources['2025'].sheet('Gráfico 48')[1]['C6']='Otra serie';r.bindings.clear();self.assertIsNone(r.review_next(row)[1])
   self.assertFalse(any(v['series_id']=='chart85_s1'for v in self.rows))
  finally:r.close()
 def test_mixed_metric_and_labour_checksum_mutations_fail_closed(self):
  cases=[('Gráfico 56','chart100_s3','header'),('Gráfico 17','chart38_s1','sum'),('Gráfico 17','chart38_s1','category'),('Gráfico 78','chart132_s3','axis')]
  for block,sid,kind in cases:
   row=self.candidate('2025',block,sid);r=m.NextResolver()
   try:
    self.assertIsNotNone(r.review_next(row)[1]);src=r.sources['2025']
    if kind=='header':src.sheet(block)[1]['E4']='Tipo de interés'
    elif kind=='sum':src.sheet(block)[0]['C5'].find('m:v',M).text='999'
    elif kind=='category':row=dict(row,periodo_categoria_literal='Periodo inventado')
    else:
     part=row['parte_grafico_imagen'];root=ET.fromstring(src.xml.parts[part]);axis=next(a for a in root.findall('.//c:valAx',C)if a.find('c:axId',C).get('val')=='1648572527');axis.find('c:numFmt',C).set('formatCode','0%');src.xml.parts[part]=ET.tostring(root)
    r.bindings.clear();self.assertIsNone(r.review_next(row)[1],kind)
   finally:r.close()
 def test_queries_exact_identity_without_fallback(self):
  seen=set()
  for row in self.rows:
   key=row['book'],row['block'],row['series_id']
   if key in seen:continue
   seen.add(key);self.assertEqual(m.query(DB,book=key[0],block=key[1],series_id=key[2],point_index=row['point_index'])['row_number'],row['row_number'])
  with self.assertRaises(LookupError):m.query(DB,book='2026',block='Anexo I. Gr 7A',series_id='chart58_s5',point_index=40)
  with self.assertRaises(ValueError):m.query(DB,book='2025',block='Gráfico 48',series_id='chart85_s3',point_index=True)
 def test_data_view_and_source_hash_tampering_detected(self):
  with patch.object(m,'PREVIOUS_SHA','0'*64):
   with duckdb.connect(str(DB),read_only=True)as c:
    with self.assertRaises(ValueError):m.collect(c)
  with tempfile.TemporaryDirectory(dir=m.ROOT/'tmp')as directory:
   db=Path(directory)/'bad.duckdb';shutil.copyfile(DB,db)
   with duckdb.connect(str(db))as c:c.execute('UPDATE next_observations SET value=value+1 WHERE row_number=(SELECT min(row_number)FROM next_observations)')
   with duckdb.connect(str(db),read_only=True)as c:
    with self.assertRaises(ValueError):m.check_new(c)
   shutil.copyfile(DB,db)
   with duckdb.connect(str(db))as c:c.execute('CREATE OR REPLACE VIEW next_usable_keys AS SELECT * FROM blocked_usable_keys')
   with duckdb.connect(str(db),read_only=True)as c:
    with self.assertRaises(ValueError):m.check_new(c)
if __name__=='__main__':unittest.main()
