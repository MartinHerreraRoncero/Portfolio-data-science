import unittest,json,tempfile,shutil,xml.etree.ElementTree as ET
from pathlib import Path
from decimal import Decimal
from collections import Counter
from unittest.mock import patch
import duckdb
from test_ready_recovery_semantics import XML,M
from experiments import extended_family_semantics as m
from experiments.excel_pilot import BOOKS
DB=m.ROOT/'evaluation/extended_family_semantics_2026-10-31/run_v2/extended_families.duckdb'
C={'c':'http://schemas.openxmlformats.org/drawingml/2006/chart'}
BLOCKS=[('2025','Gráfico 12A'),('2025','Gráfico 12B'),('2025','Gráfico 12C'),('2026','Anexo I. Gr 16A'),('2026','Anexo I. Gr 16B'),('2026','Anexo I. Gr 16C')]
BAD_CELLS={'AF11','AF12','AF13','AG14','AG15','AG16'}
class TestExtendedFamilies(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  if not DB.exists():raise RuntimeError('Build extended family fixture first')
  with duckdb.connect(str(DB),read_only=True)as c:
   q=c.execute('SELECT * FROM extended_family_observations ORDER BY book,row_number');ns=[n[0]for n in q.description];cls.rows=[dict(zip(ns,v))for v in q.fetchall()]
 def candidate(self,book,sid,point=None):
  with duckdb.connect(str(DB),read_only=True)as c:
   q=c.execute('SELECT * FROM final_family_numeric_queue WHERE book=?AND serie_id=?'+('AND id=?'if point is not None else'')+' ORDER BY row_number LIMIT 1',[book,sid]+([sid+'_p'+str(point)]if point is not None else[]));v=q.fetchone();self.assertIsNotNone(v);return dict(zip([n[0]for n in q.description],v))
 def test_independent_exact639_selection_and_raw_xml_cache(self):
  with duckdb.connect(str(m.PREVIOUS),read_only=True)as c:
   keys=set()
   for book,block in BLOCKS:keys.update(c.execute('SELECT book,row_number FROM final_family_numeric_queue WHERE book=?AND bloque=?',[book,block]).fetchall())
   keys.update(c.execute("SELECT book,row_number FROM final_family_numeric_queue WHERE book='2025'AND bloque='Gráfico 54'AND celda_valor NOT IN('AF11','AF12','AF13','AG14','AG15','AG16')").fetchall())
  self.assertEqual(len(keys),639);self.assertEqual({(r['book'],r['row_number'])for r in self.rows},keys);xs={b:XML(m.ROOT/BOOKS[b][0])for b in['2025','2026']}
  try:
   for r in self.rows:
    x=xs[r['book']];s=r['source_sheet'];raw=x.value(s,r['source_cell']);self.assertEqual(raw,r['raw_value']);node=ET.fromstring(x.z.read(r['chart_part'])).findall('.//c:ser',C)[int(r['series_id'].split('_s')[1])-1];self.assertEqual(node.find('c:val/c:numRef/c:f',C).text,r['value_ref']);vp={int(n.get('idx')):Decimal(n.find('c:v',C).text)for n in node.findall('c:val/c:numRef/c:numCache/c:pt',C)};self.assertEqual(vp[r['point_index']],raw);cs,cc=m.ref_cells(r['category_ref']);v=x.value(cs,cc[r['point_index']]);category=str(int(v))if isinstance(v,Decimal)and v==int(v)else str(v);self.assertEqual(category,r['category_literal']);self.assertEqual(category,r['universal_category']);self.assertEqual(r['multiplier'],1)
  finally:
   for x in xs.values():x.z.close()
 def test480_pyramid_signed_scale_sex_year_merge_and_positive_magnitude(self):
  xs={b:XML(m.ROOT/BOOKS[b][0])for b in['2025','2026']};rows=[r for r in self.rows if r['unit']=='thousand_persons_signed_plot'];self.assertEqual(len(rows),480)
  try:
   for r in rows:
    x=xs[r['book']];s=r['source_sheet'];col=r['source_cell'][0];header=4 if r['book']=='2025'else 20;sex='Hombres'if col in'CD'else'Mujeres';direction=-1 if sex=='Hombres'else 1;anchor=('C'if sex=='Hombres'else'E')+str(header);year=int(x.value(s,col+str(header+1)));p=json.loads(r['quantity_evidence']);self.assertEqual(x.value(s,anchor),sex);self.assertEqual(p['sex'],sex);self.assertEqual(p['reference_year'],year);self.assertEqual(p['sex_direction'],direction);self.assertGreaterEqual(r['raw_value']*direction,0);self.assertEqual(r['divisor'],1000);self.assertEqual(r['value'],r['raw_value']/1000);self.assertEqual(Decimal(p['plot_value_thousands']),r['value']);self.assertEqual(Decimal(p['population_thousands']),r['value']*direction);self.assertGreaterEqual(Decimal(p['population_thousands']),0);self.assertFalse(p['value_is_population_count']);self.assertEqual(p['age_group'],r['category_literal']);self.assertEqual(p['population_scope_literal'],x.value(s,'J5'if r['book']=='2025'else'B3'));self.assertIn('birth country',p['scope_interpretation'])
    root=ET.fromstring(x.z.read(r['chart_part']));node=root.findall('.//c:ser',C)[int(r['series_id'].split('_s')[1])-1];name=node.find('c:tx/c:strRef/c:strCache',C);cached={int(n.get('idx')):n.find('c:v',C).text for n in name.findall('c:pt',C)};self.assertEqual(cached,{0:sex,1:str(year)});self.assertEqual(r['series_literal'],sex+' | '+str(year))
   c=[r for r in rows if r['book']=='2026'and r['block']=='Anexo I. Gr 16C'];self.assertEqual({json.loads(r['quantity_evidence'])['sex']for r in c if r['series_id']=='chart77_s2'},{'Mujeres'});self.assertEqual({json.loads(r['quantity_evidence'])['reference_year']for r in c if r['series_id']=='chart77_s2'},{2025});self.assertEqual({json.loads(r['quantity_evidence'])['sex']for r in c if r['series_id']=='chart77_s3'},{'Hombres'});self.assertEqual({json.loads(r['quantity_evidence'])['reference_year']for r in c if r['series_id']=='chart77_s3'},{2050})
  finally:
   for x in xs.values():x.z.close()
 def test_independent160_scope_partitions_and159_valid_growth_triplets(self):
  xs={b:XML(m.ROOT/BOOKS[b][0])for b in['2025','2026']}
  try:
   for book in['2025','2026']:
    x=xs[book];sheets=['Gráfico 12A','Gráfico 12B','Gráfico 12C']if book=='2025'else['Anexo I. Gr 16A','Anexo I. Gr 16B','Anexo I. Gr 16C'];start=6 if book=='2025'else 22
    for col in'CDEF':
     for rr in range(start,start+20):self.assertLessEqual(abs(x.value(sheets[0],col+str(rr))+x.value(sheets[1],col+str(rr))-x.value(sheets[2],col+str(rr))),Decimal('1e-8'))
   growth=[r for r in self.rows if r['block']=='Gráfico 54'];self.assertEqual(len(growth),159);self.assertEqual(sum(r['series_literal']=='Total'for r in growth),53)
   for r in growth:
    p=json.loads(r['quantity_evidence'])['current_triplet'];values=[xs['2025'].value('Gráfico 54',a)for a in p['cells']];self.assertLessEqual(abs(values[0]+values[1]-values[2]),Decimal('1e-12'));self.assertEqual(p['year'],r['category_literal']);self.assertEqual(r['value'],r['raw_value']);self.assertEqual(r['unit'],'percent_published_growth'if r['series_literal']=='Total'else'percentage_points_published_growth_contribution')
  finally:
   for x in xs.values():x.z.close()
 def test_only_six_bad_source_points_stay_out_with_exact_diagnostics(self):
  with duckdb.connect(str(DB),read_only=True)as c:
   q=c.execute('SELECT * FROM extended_family_diagnostics');ns=[n[0]for n in q.description];diag=[dict(zip(ns,v))for v in q.fetchall()];self.assertEqual(len(diag),6);self.assertFalse(c.execute('SELECT book,row_number FROM extended_family_diagnostics INTERSECT SELECT * FROM extended_family_usable_keys').fetchall());self.assertEqual(c.execute('SELECT count(*)FROM extended_family_diagnostics d JOIN extended_family_numeric_queue q USING(book,row_number)').fetchone()[0],6)
  proof=[json.loads(d['evidence'])for d in diag];self.assertEqual({p['source_cell']for p in proof},BAD_CELLS);self.assertEqual(Counter(p['year']for p in proof),Counter({'2023':3,'2024':3}));self.assertTrue(all(abs(Decimal(p['residual']))>Decimal('1e-12')for p in proof))
  for sid,point in[('chart98_s1',7),('chart98_s2',7),('chart98_s5',7),('chart98_s3',8),('chart98_s4',8),('chart98_s6',8)]:
   row=self.candidate('2025',sid,point);r=m.ExtendedResolver()
   try:status,result,diagnostic=r.review_extended(row);self.assertEqual(status,'growth_decomposition_conflict');self.assertIsNone(result);self.assertIsNotNone(diagnostic)
   finally:r.close()
 def test_query_returns_explicit_positive_population_not_negative_population(self):
  for row in self.rows:
   q=m.query(DB,book=row['book'],block=row['block'],series_id=row['series_id'],point_index=row['point_index']);self.assertEqual(q['row_number'],row['row_number'])
   if row['unit']=='thousand_persons_signed_plot':self.assertFalse(q['value_is_population_count']);self.assertEqual(Decimal(q['plot_value_thousands']),row['value']);self.assertGreaterEqual(Decimal(q['population_thousands']),0);self.assertIn('sex_direction',q['population_transformation']);self.assertEqual(q['age_group'],row['category_literal'])
  with self.assertRaises(ValueError):m.query(DB,book='2025',block='Gráfico 54',series_id='chart98_s1',point_index=True)
  with self.assertRaises(LookupError):m.query(DB,book='2025',block='Gráfico 54',series_id='chart98_s1',point_index=7)
 def test_partition639_and_every_inherited_table_diagnostic_preserved(self):
  with duckdb.connect(str(m.PREVIOUS),read_only=True)as a,duckdb.connect(str(DB),read_only=True)as b:
   self.assertEqual(len(m.check_new(b)),639)
   for name in['final_family_usable_keys','family_observations','family_sources','remaining_internal_observations','pending_family_diagnostics','pending_family_source_incidents','final_family_diagnostics','next_deferrals','ready_quarantine']:
    aq=a.execute('SELECT * FROM '+name);schema=aq.description;data=aq.fetchall();bq=b.execute('SELECT * FROM '+name);self.assertEqual(schema,bq.description,name);self.assertEqual(Counter(data),Counter(bq.fetchall()),name)
   self.assertEqual(b.execute('SELECT count(*)FROM extended_family_usable_keys').fetchone()[0],42211);self.assertEqual(b.execute('SELECT count(*)FROM extended_family_numeric_queue').fetchone()[0],5002);self.assertFalse(b.execute('SELECT * FROM remaining_internal_keys INTERSECT SELECT * FROM extended_family_usable_keys').fetchall());self.assertFalse(b.execute('SELECT book,row_number FROM ready_quarantine INTERSECT SELECT * FROM extended_family_usable_keys').fetchall());self.assertFalse(b.execute('SELECT book,row_number FROM extended_family_numeric_queue INTERSECT SELECT * FROM extended_family_usable_keys').fetchall());self.assertFalse(b.execute('SELECT n.book,n.row_number FROM extended_family_numeric_queue n JOIN final_family_numeric_queue p USING(book,row_number)WHERE n.extended_family_reason IS DISTINCT FROM p.final_family_reason').fetchall());self.assertFalse(b.execute("SELECT t.book,t.row_number FROM extended_family_observations t JOIN final_family_numeric_queue p USING(book,row_number)WHERE p.final_family_reason IN('drawing_helper_not_economic_series','forecast_marker_not_measure','internal_only_scope')").fetchall())
 def test_positive_pyramid_sex_year_merge_scale_scope_and_cache_mutations(self):
  row=self.candidate('2026','chart75_s2')
  for kind in['sex','year','merge','scale','format','scope','cache','reference']:
   r=m.ExtendedResolver()
   try:
    self.assertIsNotNone(r.review_extended(row)[1]);s=r.sources['2026'];part=row['parte_grafico_imagen'];root=ET.fromstring(s.xml.parts[part]);node=root.findall('.//c:ser',C)[1]
    if kind=='sex':s.sheet(row['bloque'])[1]['C20']='Mujeres'
    elif kind=='year':s.sheet(row['bloque'])[1]['D21']=2025
    elif kind=='merge':
     path=s.xml.sheets[row['bloque']];sheet=ET.fromstring(s.xml.parts[path]);parent=sheet.find('m:mergeCells',M)
     for n in list(parent):
      if n.get('ref')=='C20:D20':parent.remove(n)
     s.xml.parts[path]=ET.tostring(sheet)
    elif kind=='scale':root.find('.//c:valAx/c:dispUnits/c:builtInUnit',C).set('val','millions')
    elif kind=='format':root.find('.//c:valAx/c:numFmt',C).set('formatCode','#,##0')
    elif kind=='scope':s.sheet(row['bloque'])[1]['B3']='POBLACIÓN NACIDA EN ESPAÑA'
    elif kind=='cache':node.find('c:val/c:numRef/c:numCache/c:pt/c:v',C).text='999'
    else:node.find('c:val/c:numRef/c:f',C).text="'Anexo I. Gr 16A'!$F$22:$F$41"
    s.xml.parts[part]=ET.tostring(root);r.bindings.clear();self.assertIsNone(r.review_extended(row)[1],kind)
   finally:r.close()
 def test_coherent_wrong_male_sign_cannot_be_hidden_with_abs(self):
  row=self.candidate('2026','chart75_s2');r=m.ExtendedResolver()
  try:
   self.assertIsNotNone(r.review_extended(row)[1]);s=r.sources['2026'];cell=row['celda_valor'];positive=str(-Decimal(row['decimal_XML_texto']));s.sheet(row['bloque'])[0][cell].find('m:v',M).text=positive;root=ET.fromstring(s.xml.parts[row['parte_grafico_imagen']]);root.findall('.//c:ser',C)[1].find('c:val/c:numRef/c:numCache/c:pt/c:v',C).text=positive;s.xml.parts[row['parte_grafico_imagen']]=ET.tostring(root);r.bindings.clear();row=dict(row,decimal_XML_texto=positive);status,result,_=r.review_extended(row);self.assertIsNone(result);self.assertIn('sex direction',status)
  finally:r.close()
 def test_growth_point_baseline_and_local_triplet_corruption(self):
  row=self.candidate('2025','chart98_s3',9);r=m.ExtendedResolver()
  try:
   self.assertIsNotNone(r.review_extended(row)[1]);s=r.sources['2025'];s.sheet('Gráfico 54')[0]['AH16'].find('m:v',M).text='999';status,result,proof=r.review_extended(row);self.assertEqual(status,'growth_decomposition_conflict');self.assertIsNone(result);self.assertEqual(proof['year'],'2025')
  finally:r.close()
 def test_source_hash_data_and_view_tampering_fail_closed(self):
  with patch.object(m,'PREVIOUS_SHA','0'*64):
   with duckdb.connect(str(DB),read_only=True)as c:
    with self.assertRaises(ValueError):m.collect(c)
  with tempfile.TemporaryDirectory(dir=m.ROOT/'tmp')as directory:
   db=Path(directory)/'bad.duckdb';shutil.copyfile(DB,db)
   with duckdb.connect(str(db))as c:c.execute("UPDATE extended_family_observations SET value=-value WHERE book='2026'AND series_id='chart75_s2'")
   with duckdb.connect(str(db),read_only=True)as c:
    with self.assertRaises(ValueError):m.check_new(c)
   shutil.copyfile(DB,db)
   with duckdb.connect(str(db))as c:c.execute('CREATE OR REPLACE VIEW extended_family_usable_keys AS SELECT * FROM final_family_usable_keys UNION ALL SELECT * FROM remaining_internal_keys')
   with duckdb.connect(str(db),read_only=True)as c:
    with self.assertRaises(ValueError):m.check_new(c)
if __name__=='__main__':unittest.main()
