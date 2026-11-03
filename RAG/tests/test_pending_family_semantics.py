import unittest,json,tempfile,shutil,xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from decimal import Decimal,ROUND_HALF_UP
from unittest.mock import patch
import duckdb
from test_ready_recovery_semantics import XML,M
from experiments import pending_family_semantics as m
from experiments.excel_pilot import BOOKS
DB=m.ROOT/'evaluation/pending_family_semantics_2026-10-31/run_v4/pending_families.duckdb'
C={'c':'http://schemas.openxmlformats.org/drawingml/2006/chart'}
TARGETS=[('2026','chart21_s4'),('2025','chart146_s3'),('2025','chart124_s2'),('2026','chart41_s3'),('2026','chart2_s2'),('2026','chart3_s3'),('2026','chart35_s1'),('2026','chart37_s2'),('2025','chart52_s1'),('2025','chart52_s2'),('2025','chart50_s1'),('2025','chart50_s2'),('2025','chart50_s4'),('2025','chart105_s1'),('2025','chart105_s2'),('2025','chart105_s3')]
class TestPendingFamilies(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  if not DB.exists():raise RuntimeError('Build pending family fixture first')
  with duckdb.connect(str(DB),read_only=True)as c:
   q=c.execute('SELECT * FROM pending_family_observations ORDER BY book,row_number');names=[n[0]for n in q.description];cls.rows=[dict(zip(names,v))for v in q.fetchall()]
 def candidate(self,book,sid):
  with duckdb.connect(str(DB),read_only=True)as c:
   q=c.execute('SELECT * FROM next_numeric_queue WHERE book=? AND serie_id=? ORDER BY row_number LIMIT 1',[book,sid]);value=q.fetchone();self.assertIsNotNone(value);return dict(zip([x[0]for x in q.description],value))
 def test_independent_exact_selection_and_xml423_values_names_categories(self):
  with duckdb.connect(str(m.PREVIOUS),read_only=True)as c:
   expected=set()
   for book,sid in TARGETS:expected.update(c.execute('SELECT book,row_number FROM next_numeric_queue WHERE book=? AND serie_id=?',[book,sid]).fetchall())
  self.assertEqual(len(expected),423);self.assertEqual({(r['book'],r['row_number'])for r in self.rows},expected);readers={b:XML(m.ROOT/BOOKS[b][0])for b in['2025','2026']}
  try:
   for r in self.rows:
    x=readers[r['book']];s=r['source_sheet'];raw=x.value(s,r['source_cell']);self.assertEqual(raw,r['raw_value']);self.assertEqual(raw,r['value']);self.assertEqual(r['multiplier'],1);self.assertEqual(r['divisor'],1)
    node=ET.fromstring(x.z.read(r['chart_part'])).findall('.//c:ser',C)[int(r['series_id'].split('_s')[1])-1];self.assertEqual(node.find('c:val/c:numRef/c:f',C).text,r['value_ref']);vp={int(n.get('idx')):Decimal(n.find('c:v',C).text)for n in node.findall('c:val/c:numRef/c:numCache/c:pt',C)};self.assertEqual(vp[r['point_index']],raw)
    ns,nc=m.ref_cells(r['name_ref']);self.assertEqual(x.value(ns,nc[0]),r['series_literal']);cs,cc=m.ref_cells(r['category_ref']);v=x.value(cs,cc[r['point_index']]);category=str(int(v))if isinstance(v,Decimal)and v==int(v)else str(v);self.assertEqual(category,r['category_literal'])
    if r['universal_category']is None:self.assertEqual(r['category_basis'],'reviewed_source_grid')
    else:self.assertEqual(category.replace('\n',' / '),r['universal_category'])
    if r['series_id']in['chart21_s4','chart146_s3','chart124_s2','chart41_s3','chart2_s2','chart3_s3','chart35_s1','chart37_s2']:
     self.assertIsNone(r['universal_series']);self.assertEqual(r['series_literal'],{'chart21_s4':'< 15 años','chart146_s3':'Ajuste acumulado','chart124_s2':'Deuda Pública','chart41_s3':'Saldo primario','chart2_s2':'Deuda Pública','chart3_s3':'Saldo primario','chart35_s1':'Deuda sobre PIB','chart37_s2':'Deuda Pública'}[r['series_id']])
    else:self.assertEqual(r['series_literal'],r['universal_series'])
  finally:
   for x in readers.values():x.z.close()
 def test_new_layer_partition_reasons_and_excluded_roles(self):
  with duckdb.connect(str(DB),read_only=True)as c:
   self.assertEqual(len(m.check_new(c)),423);self.assertEqual(c.execute('SELECT count(*)FROM family_usable_keys').fetchone()[0],40921);self.assertEqual(c.execute('SELECT count(*)FROM family_numeric_queue').fetchone()[0],6292);self.assertFalse(c.execute('SELECT * FROM next_usable_keys EXCEPT ALL SELECT * FROM family_usable_keys').fetchall());self.assertFalse(c.execute('SELECT book,row_number FROM family_numeric_queue INTERSECT SELECT * FROM family_usable_keys').fetchall());self.assertFalse(c.execute('SELECT book,row_number FROM ready_quarantine INTERSECT SELECT * FROM family_usable_keys').fetchall());self.assertFalse(c.execute('SELECT book,row_number FROM next_deferrals INTERSECT SELECT book,row_number FROM pending_family_observations').fetchall())
   self.assertFalse(c.execute("SELECT t.book,t.row_number FROM pending_family_observations t JOIN next_numeric_queue s USING(book,row_number)WHERE s.next_reason IN('drawing_helper_not_economic_series','forecast_marker_not_measure')OR s.serie_literal='Auxiliar'").fetchall());self.assertFalse(c.execute("SELECT n.book,n.row_number FROM family_numeric_queue n JOIN next_numeric_queue p USING(book,row_number)WHERE n.family_reason IS DISTINCT FROM p.next_reason AND n.family_reason<>'literal_series_source_conflict'").fetchall())
 def test_legacy_family_tables_schema_content_and_prior_diagnostics_unchanged(self):
  with duckdb.connect(str(m.PREVIOUS),read_only=True)as a,duckdb.connect(str(DB),read_only=True)as b:
   for name in['family_observations','family_sources','next_deferrals','ready_quarantine','next_usable_keys']:
    aq=a.execute('SELECT * FROM '+name);schema=aq.description;values=aq.fetchall();bq=b.execute('SELECT * FROM '+name);self.assertEqual(schema,bq.description,name);self.assertEqual(Counter(values),Counter(bq.fetchall()),name)
   self.assertEqual(b.execute('SELECT count(*)FROM next_deferrals').fetchone()[0],131);self.assertEqual(b.execute('SELECT count(*)FROM ready_quarantine').fetchone()[0],13)
 def test_cross_sheet287_independent_source_and_no_admission(self):
  x=XML(m.ROOT/BOOKS['2025'][0])
  try:
   with duckdb.connect(str(DB),read_only=True)as c:
    rows=c.execute('SELECT book,row_number,status,evidence FROM pending_family_diagnostics ORDER BY row_number').fetchall();self.assertEqual(len(rows),287);self.assertFalse(c.execute('SELECT book,row_number FROM pending_family_diagnostics INTERSECT SELECT * FROM family_usable_keys').fetchall());self.assertEqual(c.execute("SELECT count(*)FROM family_numeric_queue WHERE family_reason='literal_series_source_conflict'AND(book,row_number)IN(SELECT book,row_number FROM pending_family_diagnostics)").fetchone()[0],287)
   for _,_,status,text in rows:
    p=json.loads(text);self.assertEqual(status,'literal_series_source_conflict');self.assertEqual(x.value('Gráfico 100',p['source_cell']),Decimal(p['raw_value']));node=ET.fromstring(x.z.read(p['chart_part'])).findall('.//c:ser',C)[3];self.assertEqual(node.find('c:val/c:numRef/c:f',C).text,"'Gráfico 100'!$F$6:$F$46");self.assertIn('ANOMALÍA',p['source_metric']);self.assertNotEqual(p['source_metric'],p['universal_metric'])
  finally:x.z.close()
 def test_cross_document130_incidents_keep104_excel_and_defer26(self):
  with duckdb.connect(str(DB),read_only=True)as c:
   rows=c.execute('SELECT book,row_number,previously_admitted,evidence FROM pending_family_source_incidents').fetchall();self.assertEqual(len(rows),130);self.assertEqual(sum(r[2]for r in rows),104);self.assertEqual(c.execute('SELECT count(*)FROM pending_family_source_incidents i JOIN family_usable_keys k USING(book,row_number)').fetchone()[0],104);self.assertEqual(c.execute('SELECT count(*)FROM pending_family_observations WHERE book=\'2026\'AND block=\'Gráfico 36\'').fetchone()[0],0)
  x=XML(m.ROOT/BOOKS['2026'][0])
  try:
   for _,_,prior,text in rows:
    p=json.loads(text);self.assertEqual(p['scope'],'Excel snapshot verified; PDF numeric correspondence unresolved');self.assertEqual(x.value('Gráfico 36',p['source_cell']),Decimal(p['raw_value']));self.assertEqual(p['original_rule'],'explicit_unit_named_complete_axis_v1'if prior else None);self.assertNotEqual(Decimal(p['numeric_conflict_2050']['excel_total']).quantize(Decimal('.1')),Decimal(p['numeric_conflict_2050']['pdf_total_approx']))
  finally:x.z.close()
 def test_independent_productivity_total_components_blank_and_growth_units(self):
  x=XML(m.ROOT/BOOKS['2025'][0])
  try:
   for rr in range(5,17):
    if rr==14:continue
    self.assertLessEqual(abs(x.value('Gráfico 26','D'+str(rr))+x.value('Gráfico 26','E'+str(rr))-x.value('Gráfico 26','C'+str(rr))),Decimal('1e-14'))
   rows=[r for r in self.rows if r['block']=='Gráfico 26'];self.assertEqual(len(rows),33);self.assertEqual(sum(r['quantity_role']=='potential_real_output_growth_productivity_total'for r in rows),11);self.assertEqual(sum(r['category_literal']!=r['universal_category']for r in rows),0);self.assertFalse(any(r['source_cell'][1:]=='14'for r in rows));self.assertTrue(all(r['unit']=='percentage_points_potential_real_output_growth'for r in rows))
  finally:x.z.close()
 def test_re4_override_numerical_anchors_years_and_both_captions_fail_closed(self):
  row=self.candidate('2025','chart105_s1');r=m.FamilyResolver()
  try:
   status,result=r.review_family(row);self.assertIsNotNone(result);d=dict(zip(m.FIELDS,result));self.assertEqual(d['metric_literal'],'EVOLUCIÓN DEL SALDO PRIMARIO (% PIB)');self.assertEqual(d['quantity_role'],'primary_balance_not_expenditure');p=json.loads(d['quantity_evidence']);self.assertIn('SALDO',p['universal_metric']);self.assertIn('GASTO',p['bounded_caption_override']['source_caption']);source=r.sources['2025'];sheet='Gráfico RE 4.1'
   for cell in['BG9','BG6','CA9','CA6']:
    node=source.sheet(sheet)[0][cell].find('m:v',M);old=node.text;node.text='999'
    try:
     with self.assertRaises(ValueError,msg=cell):r.primary_balance_override()
    finally:node.text=old
   for cell in['BG5','CA5','AL5','AF5']:
    texts=source.sheet(sheet)[1];old=texts[cell];texts[cell]='2099'
    try:
     with self.assertRaises(ValueError,msg=cell):r.primary_balance_override()
    finally:texts[cell]=old
   old=source.sheet(sheet)[1]['C2'];source.sheet(sheet)[1]['C2']='EVOLUCIÓN DEL SALDO PRIMARIO (% PIB)';self.assertIsNone(r.review_family(row)[1]);source.sheet(sheet)[1]['C2']=old;r.pdf_pages['2025',123]='EVOLUCIÓN DEL GASTO PRIMARIO';self.assertIsNone(r.review_family(row)[1])
  finally:r.close()
 def test_dependency_denominator_and_entry_flow_scope(self):
  for r in self.rows:
   p=json.loads(r['quantity_evidence'])
   if r['block']=='Gráfico 14A':self.assertEqual(r['unit'],'percent_under15_population_over_population_age15_74');self.assertEqual(p['denominator'],'population aged 15-74');self.assertEqual(p['ratio_definition_multiplier'],100);self.assertEqual(r['value'],r['raw_value'])
   if r['block']=='Gráfico 28A':self.assertEqual(r['quantity_role'],'monthly_entry_flow_rate_not_unemployment_exit_probability');self.assertEqual(p['denominator'],'total employed in previous month t-1');self.assertEqual(p['frequency'],'monthly')
   if r['block']in['Gráfico 3','Gráfico 32']and r['book']=='2026':self.assertEqual(r['metric_literal'],'Saldo primario');self.assertEqual(r['quantity_role'],'primary_public_balance_not_total_balance')
 def test_positive_baseline_headers_refs_axes_cache_and_checksum_mutations(self):
  for kind in['header','ref','axis','cache','category']:
   row=self.candidate('2026','chart21_s4');r=m.FamilyResolver()
   try:
    self.assertIsNotNone(r.review_family(row)[1]);source=r.sources['2026'];part=row['parte_grafico_imagen'];root=ET.fromstring(source.xml.parts[part]);node=root.findall('.//c:ser',C)[3]
    if kind=='header':source.sheet(row['bloque'])[1]['D20']='> 74 años'
    elif kind=='ref':node.find('c:val/c:numRef/c:f',C).text="'Gráfico 14A'!$E$21:$E$69"
    elif kind=='axis':root.find('.//c:valAx/c:numFmt',C).set('formatCode','0%')
    elif kind=='cache':node.find('c:val/c:numRef/c:numCache/c:pt/c:v',C).text='999'
    else:row=dict(row,periodo_categoria_literal='2099')
    source.xml.parts[part]=ET.tostring(root);r.bindings.clear();self.assertIsNone(r.review_family(row)[1],kind)
   finally:r.close()
  row=self.candidate('2025','chart50_s1');r=m.FamilyResolver()
  try:
   self.assertIsNotNone(r.review_family(row)[1]);r.sources['2025'].sheet('Gráfico 26')[0]['C5'].find('m:v',M).text='999';self.assertIsNone(r.review_family(row)[1])
  finally:r.close()
 def test_cumulative_adjustment_categories_and_no_empty_extension(self):
  r=self.candidate('2025','chart146_s3');value=m.query(DB,book='2025',block='Gráfico 91',series_id='chart146_s3',point_index=0);self.assertEqual(value['category_literal'],'2025');self.assertEqual(value['quantity_role'],'cumulative_structural_adjustment_as_labelled_not_annual_adjustment');self.assertEqual(Decimal(value['raw_value']),0)
  with self.assertRaises(LookupError):m.query(DB,book='2025',block='Gráfico 91',series_id='chart146_s3',point_index=26)
  with self.assertRaises(ValueError):m.query(DB,book='2025',block='Gráfico 91',series_id='chart146_s3',point_index=True)
  for row in self.rows:
   self.assertEqual(m.query(DB,book=row['book'],block=row['block'],series_id=row['series_id'],point_index=row['point_index'])['row_number'],row['row_number'])
 def test_source_redirect_and_data_view_hash_tampering_rejected(self):
  resolver=m.FamilyResolver()
  try:
   with duckdb.connect(str(DB),read_only=True)as c:
    q=c.execute('SELECT * FROM next_numeric_queue');names=[n[0]for n in q.description];data=[dict(zip(names,v))for v in q.fetchall()];self.assertEqual(len(m.diagnostics(c,resolver,data)),287);part='xl/charts/chart160.xml';root=ET.fromstring(resolver.sources['2025'].xml.parts[part]);root.findall('.//c:ser',C)[3].find('c:val/c:numRef/c:f',C).text="'Gráfico 101'!$F$5:$F$45";resolver.sources['2025'].xml.parts[part]=ET.tostring(root)
    with self.assertRaises(ValueError):m.diagnostics(c,resolver,data)
  finally:resolver.close()
  with patch.object(m,'PREVIOUS_SHA','0'*64):
   with duckdb.connect(str(DB),read_only=True)as c:
    with self.assertRaises(ValueError):m.collect(c)
  with tempfile.TemporaryDirectory(dir=m.ROOT/'tmp')as directory:
   db=Path(directory)/'bad.duckdb';shutil.copyfile(DB,db)
   with duckdb.connect(str(db))as c:c.execute('UPDATE pending_family_observations SET value=value+1 WHERE row_number=(SELECT min(row_number)FROM pending_family_observations)')
   with duckdb.connect(str(db),read_only=True)as c:
    with self.assertRaises(ValueError):m.check_new(c)
   shutil.copyfile(DB,db)
   with duckdb.connect(str(db))as c:c.execute('CREATE OR REPLACE VIEW family_usable_keys AS SELECT * FROM next_usable_keys')
   with duckdb.connect(str(db),read_only=True)as c:
    with self.assertRaises(ValueError):m.check_new(c)
if __name__=='__main__':unittest.main()
