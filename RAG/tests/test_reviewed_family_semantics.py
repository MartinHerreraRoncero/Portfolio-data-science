import unittest,json,tempfile,shutil,xml.etree.ElementTree as ET
from pathlib import Path
from decimal import Decimal
from collections import Counter
from unittest.mock import patch
import duckdb
from test_ready_recovery_semantics import XML,M
from experiments import reviewed_family_semantics as m
from experiments.excel_pilot import BOOKS
DB=m.ROOT/'evaluation/reviewed_family_semantics_2026-10-31/run_v2/reviewed_families.duckdb'
C={'c':'http://schemas.openxmlformats.org/drawingml/2006/chart'}
TABLES={'2025':['Cuadro 8 ','Cuadro 10 ','Cuadro 11','Cuadro 21'],'2026':['Cuadro 1 y 3','Cuadro 2 y 8','Cuadro 9','Cuadro 10']}
SIDS=['chart76_s3','chart76_s4','chart94_s2','chart96_s2','chart97_s2','chart103_s2','chart104_s3','chart106_s3','chart107_s5','chart107_s6']
class TestReviewedFamilies(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  if not DB.exists():raise RuntimeError('Build reviewed family fixture first')
  with duckdb.connect(str(DB),read_only=True)as c:
   q=c.execute('SELECT * FROM reviewed_family_observations ORDER BY book,row_number');ns=[x[0]for x in q.description];cls.rows=[dict(zip(ns,v))for v in q.fetchall()]
 def candidate(self,book,block,cell=None,sid=None):
  with duckdb.connect(str(m.PREVIOUS),read_only=True)as c:
   sql='SELECT * FROM extended_family_numeric_queue WHERE book=?AND bloque=?';args=[book,block]
   if cell:sql+='AND celda_valor=?';args.append(cell)
   if sid:sql+='AND serie_id=?';args.append(sid)
   q=c.execute(sql+' ORDER BY row_number LIMIT 1',args);v=q.fetchone();self.assertIsNotNone(v);return dict(zip([x[0]for x in q.description],v))
 def positive(self,r,row):
  status,out=r.review_row(row);self.assertEqual(status,'usable_reviewed_family');self.assertIsNotNone(out);return out
 def test_independent_exact_selection251_and_raw_xml(self):
  with duckdb.connect(str(m.PREVIOUS),read_only=True)as c:
   keys=set()
   for book,blocks in TABLES.items():
    for b in blocks:keys.update(c.execute('SELECT book,row_number FROM extended_family_numeric_queue WHERE book=?AND bloque=?',[book,b]).fetchall())
   self.assertEqual(len(keys),161)
   for sid in SIDS:keys.update(c.execute("SELECT book,row_number FROM extended_family_numeric_queue WHERE book='2025'AND serie_id=?",[sid]).fetchall())
  self.assertEqual(len(keys),251);self.assertEqual(keys,{(r['book'],r['row_number'])for r in self.rows});xs={b:XML(m.ROOT/BOOKS[b][0])for b in['2025','2026']}
  try:
   for r in self.rows:
    x=xs[r['book']];raw=x.value(r['source_sheet'],r['source_cell']);self.assertEqual(raw,r['raw_value']);self.assertEqual(r['value'],raw*r['multiplier']/r['divisor']);self.assertEqual(r['universal_value_sheet'],r['source_sheet']);self.assertEqual(r['universal_value_cell'],r['source_cell'])
    if r['series_id']is not None:
     node=ET.fromstring(x.z.read(r['chart_part'])).findall('.//c:ser',C)[int(r['series_id'].split('_s')[1])-1];self.assertIsNone(node.find('c:tx',C));self.assertEqual(node.find('c:val/c:numRef/c:f',C).text,r['value_ref']);vp={int(n.get('idx')):Decimal(n.find('c:v',C).text)for n in node.findall('c:val/c:numRef/c:numCache/c:pt',C)};self.assertEqual(vp[r['point_index']],raw);sh,cc=m.ref_cells(r['category_ref']);v=x.value(sh,cc[r['point_index']]);category=str(int(v))if isinstance(v,Decimal)and v==int(v)else str(v);self.assertEqual(category,r['category_literal']);self.assertIn(category,[str(y)for y in range(2015,2024)]);self.assertIsNone(r['universal_series'])
    else:self.assertIsNone(r['point_index']);self.assertIsNone(r['chart_part']);self.assertTrue(json.loads(r['quantity_evidence'])['public_table_not_internal_model'])
  finally:
   for x in xs.values():x.z.close()
 def test_mixed_table_units_roles_and_scale_once(self):
  tables=[r for r in self.rows if r['series_id']is None];self.assertEqual(len(tables),161)
  for r in tables:
   sh=r['block'];cell=r['source_cell'];col=''.join(x for x in cell if x.isalpha());rr=int(''.join(x for x in cell if x.isdigit()));mult=Decimal(100)if sh=='Cuadro 11'or(sh=='Cuadro 10 'and col in'IJK')else Decimal(1);self.assertEqual(r['multiplier'],mult);self.assertEqual(r['divisor'],1)
   if sh=='Cuadro 8 ':self.assertEqual(r['unit'],'dimensionless_elasticity_nominal_gdp'if col=='D'else'percent_total_resources_2023');self.assertEqual(r['category_literal'],'2030-2070'if col=='D'else'2023')
   if sh=='Cuadro 10 ':self.assertEqual(r['unit'],'percentage_points_gdp'if col in'EFG'else'percentage_points_pension_average_over_average_wage');self.assertIn(r['category_literal'],['2023','2050','2070']);self.assertEqual(r['quantity_role'],'published_total_reform_impact'if rr==6 else'published_measure_impact')
   if sh=='Cuadro 11':self.assertEqual(r['unit'],'percent_internal_rate_of_return'if rr in[6,17]else'percentage_points_internal_rate_of_return');self.assertEqual(r['quantity_role'],'published_before_after_level'if rr in[6,17]else'published_measure_difference')
   if sh=='Cuadro 21':self.assertIn('16_66',r['unit'])if rr in[7,8]else self.assertIn('thousand_persons',r['unit']);self.assertEqual(r['category_literal'],'2024-2050'if rr==6 else'2050');self.assertIn('not_recomputed',r['quantity_role'])if rr==8 else None
   if sh=='Cuadro 2 y 8':self.assertEqual(r['category_literal'],'2022-2050');self.assertEqual(r['quantity_role'],'published_rule_reported_value'if col in'CFI'else'published_rule_limit');self.assertIn('period_context_pdf',json.loads(r['quantity_evidence']))
 def test_historical_identity_context_no_vintage_inference(self):
  histories=[r for r in self.rows if r['series_id']is not None];self.assertEqual(len(histories),90)
  for r in histories:
   p=json.loads(r['quantity_evidence']);self.assertTrue(p['historical_vintage_not_inferred']);self.assertIsNone(p['universal_series']);self.assertEqual(r['series_literal'].strip(),'Histórico');self.assertEqual(r['unit'],'percent_gdp');self.assertEqual(r['raw_value'],r['value']);self.assertEqual(r['quantity_role'],'historical_level_as_source_labelled')
  for block in['Gráfico 44','Gráfico RE 4.3']:
   rr=[r for r in histories if r['block']==block];self.assertEqual(len(rr),18);self.assertEqual(len({json.dumps(json.loads(r['quantity_evidence'])['physical_context'])for r in rr}),2)
 def test_exact_queries_and_null_identity(self):
  for r in self.rows:
   out=m.query_table(DB,book=r['book'],block=r['block'],row_number=r['row_number'])if r['series_id']is None else m.query(DB,book=r['book'],block=r['block'],series_id=r['series_id'],point_index=r['point_index']);self.assertEqual(out['record_id'],r['record_id']);self.assertEqual(Decimal(out['value']),r['value'])
  r=self.rows[0]
  with self.assertRaises(LookupError):m.query_table(DB,book=r['book'],block='wrong',row_number=r['row_number'])
  with self.assertRaises(ValueError):m.query_table(DB,book='2025',block='Cuadro 11',row_number=True)
  with self.assertRaises(ValueError):m.query(DB,book='2025',block='Gráfico 50',series_id='chart94_s2',point_index=True)
  with self.assertRaises(LookupError):m.query(DB,book='2025',block='Gráfico 50',series_id='chart94_s2',point_index=500)
 def test_table_physical_period_format_and_metric_fail_closed(self):
  cases=[('2025','Cuadro 10 ','F6','F5',2060),('2026','Cuadro 10','C7','C6',2026),('2026','Cuadro 9','E6','E5',2035),('2025','Cuadro 11','E6','D4','Different metric')]
  for book,block,cell,anchor,value in cases:
   r=m.ReviewedResolver()
   try:
    row=self.candidate(book,block,cell);self.positive(r,row);s=r.sources[book];original=s.text
    with patch.object(s,'text',side_effect=lambda sh,a,old=original: value if(sh,a)==(block,anchor)else old(sh,a)):self.assertIsNone(r.review_row(row)[1])
   finally:r.close()
  r=m.ReviewedResolver()
  try:
   row=self.candidate('2025','Cuadro 11','D6');self.positive(r,row);cell=r.sources['2025'].workbook['Cuadro 11']['D6']
   with patch.object(type(cell),'number_format',new=property(lambda _: '0.0')):self.assertIsNone(r.review_row(row)[1])
  finally:r.close()
 def test_table_original_metadata_and_raw_fail_closed(self):
  r=m.ReviewedResolver()
  try:
   row=self.candidate('2025','Cuadro 21','E8');self.positive(r,row)
   for field,value in[('serie_id','invented'),('periodo_categoria_literal','2000'),('decimal_XML_texto','0'),('celda_valor','E7'),('metrica_literal','wrong')]:
    changed=dict(row);changed[field]=value;self.assertIsNone(r.review_row(changed)[1])
   s=r.sources['2025'];old=s.number
   with patch.object(s,'number',side_effect=lambda sh,a: Decimal(999)if(sh,a)==('Cuadro 21','E8')else old(sh,a)):self.assertIsNone(r.review_row(row)[1])
  finally:r.close()
 def test_chart_header_context_and_category_fail_closed(self):
  r=m.ReviewedResolver()
  try:
   row=self.candidate('2025','Gráfico 44',sid='chart76_s3');self.positive(r,row);s=r.sources['2025'];old=s.text
   for anchor in['B13','B14','C3']:
    with patch.object(s,'text',side_effect=lambda sh,a: 'WRONG'if(sh,a)==('Gráfico 44',anchor)else old(sh,a)):self.assertIsNone(r.review_row(row)[1])
   changed=dict(row,periodo_categoria_literal='2024');self.assertIsNone(r.review_row(changed)[1])
  finally:r.close()
 def test_chart_refs_axis_and_cache_fail_closed(self):
  row=self.candidate('2025','Gráfico 50',sid='chart94_s2')
  for mode in['reference','axis','display','cache','name']:
   r=m.ReviewedResolver()
   try:
    self.positive(r,row);s=r.sources['2025'];part=row['parte_grafico_imagen'];root=ET.fromstring(s.xml.parts[part]);node=root.findall('.//c:ser',C)[1]
    if mode=='reference':node.find('c:val/c:numRef/c:f',C).text="'Gráfico 50'!$X$9:$AF$9"
    elif mode=='axis':root.find('.//c:valAx/c:numFmt',C).set('formatCode','0%')
    elif mode=='display':ET.SubElement(root.find('.//c:valAx',C),'{'+C['c']+'}dispUnits')
    elif mode=='cache':node.find('c:val/c:numRef/c:numCache/c:pt/c:v',C).text='999'
    else:ET.SubElement(node,'{'+C['c']+'}tx')
    s.xml.parts[part]=ET.tostring(root);self.assertIsNone(r.review_row(row)[1])
   finally:r.close()
 def test_all_previous_tables_views_keys_and_reasons_preserved(self):
  with duckdb.connect(str(DB),read_only=True)as c:
   m.check_new(c);c.execute("ATTACH '"+str(m.PREVIOUS).replace("'","''")+"' AS baseline (READ_ONLY)")
   names=[x[0]for x in c.execute("SELECT table_name FROM information_schema.tables WHERE table_catalog='baseline'AND table_schema='main'AND table_type='BASE TABLE'").fetchall()]
   self.assertGreater(len(names),60)
   for name in names:
    self.assertEqual(c.execute('DESCRIBE main."'+name+'"').fetchall(),c.execute('DESCRIBE baseline."'+name+'"').fetchall(),name)
    for a,b in[('main','baseline'),('baseline','main')]:self.assertFalse(c.execute('SELECT * FROM '+a+'."'+name+'" EXCEPT ALL SELECT * FROM '+b+'."'+name+'"').fetchall(),name)
   self.assertEqual(c.execute('SELECT count(*)FROM reviewed_family_usable_keys').fetchone()[0],42462);self.assertEqual(c.execute('SELECT count(*)FROM reviewed_family_numeric_queue').fetchone()[0],4751);self.assertEqual(c.execute('SELECT count(*)FROM remaining_internal_keys').fetchone()[0],168);self.assertEqual(c.execute('SELECT count(*)FROM ready_quarantine').fetchone()[0],13);self.assertEqual(c.execute('SELECT count(*)FROM next_deferrals').fetchone()[0],131)
   self.assertFalse(c.execute('SELECT * FROM remaining_internal_keys INTERSECT SELECT * FROM reviewed_family_usable_keys').fetchall());self.assertFalse(c.execute('SELECT book,row_number FROM ready_quarantine INTERSECT SELECT * FROM reviewed_family_usable_keys').fetchall());self.assertFalse(c.execute('SELECT book,row_number,reviewed_family_reason FROM reviewed_family_numeric_queue EXCEPT SELECT book,row_number,extended_family_reason FROM extended_family_numeric_queue').fetchall())
 def test_tamper_and_hashes_rejected(self):
  with patch.object(m,'PREVIOUS_SHA','wrong'):
   with duckdb.connect(str(DB),read_only=True)as c:
    with self.assertRaises(ValueError):m.check_new(c)
  for mode in['data','view']:
   with tempfile.TemporaryDirectory()as td:
    dest=Path(td)/'copy.duckdb';shutil.copyfile(DB,dest)
    with duckdb.connect(str(dest))as c:
     if mode=='data':c.execute('UPDATE reviewed_family_observations SET value=value+1 WHERE row_number=(SELECT min(row_number)FROM reviewed_family_observations)')
     else:c.execute('CREATE OR REPLACE VIEW reviewed_family_usable_keys AS SELECT * FROM extended_family_usable_keys')
     with self.assertRaises(ValueError):m.check_new(c)
if __name__=='__main__':unittest.main()

