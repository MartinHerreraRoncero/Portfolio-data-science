import unittest,json,tempfile,shutil,xml.etree.ElementTree as ET
from pathlib import Path
from decimal import Decimal,localcontext
from collections import Counter
from unittest.mock import patch
import duckdb
from test_ready_recovery_semantics import XML,M
from experiments import final_family_semantics as m
from experiments.excel_pilot import BOOKS
DB=m.ROOT/'evaluation/final_family_semantics_2026-10-31/run_v5/final_families.duckdb'
C={'c':'http://schemas.openxmlformats.org/drawingml/2006/chart'}
TARGETS=[('2025','chart2_s1'),('2025','chart2_s2'),('2025','chart65_s1'),('2025','chart65_s2'),*[('2025','chart78_s'+str(i))for i in range(1,5)],('2025','chart70_s3'),('2025','chart68_s1'),('2025','chart68_s4'),('2025','chart73_s2'),*[('2025','chart90_s'+str(i))for i in range(1,5)],*[('2026','chart53_s'+str(i))for i in range(1,5)],('2026','chart54_s2')]
class TestFinalFamilies(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  if not DB.exists():raise RuntimeError('Build final family fixture first')
  with duckdb.connect(str(DB),read_only=True)as c:
   q=c.execute('SELECT * FROM final_family_observations ORDER BY book,row_number');ns=[n[0]for n in q.description];cls.rows=[dict(zip(ns,v))for v in q.fetchall()]
 def candidate(self,book,sid):
  with duckdb.connect(str(DB),read_only=True)as c:
   q=c.execute('SELECT * FROM remaining_numeric_queue WHERE book=?AND serie_id=?ORDER BY row_number LIMIT 1',[book,sid]);value=q.fetchone();self.assertIsNotNone(value);return dict(zip([n[0]for n in q.description],value))
 def test_exact_independent_selection_and330_xml_values_cache_scaling(self):
  with duckdb.connect(str(m.PREVIOUS),read_only=True)as c:
   expected=set()
   for book,sid in TARGETS:expected.update(c.execute('SELECT book,row_number FROM remaining_numeric_queue WHERE book=?AND serie_id=?',[book,sid]).fetchall())
  self.assertEqual(len(expected),330);self.assertEqual({(r['book'],r['row_number'])for r in self.rows},expected);xs={b:XML(m.ROOT/BOOKS[b][0])for b in['2025','2026']}
  try:
   for r in self.rows:
    x=xs[r['book']];s=r['source_sheet'];raw=x.value(s,r['source_cell']);self.assertEqual(raw,r['raw_value']);factor=100 if s in['Gráfico 46','Gráfico RE 3.3']else 1;self.assertEqual(raw*factor,r['value']);self.assertEqual(r['multiplier'],factor);self.assertEqual(r['divisor'],1);node=ET.fromstring(x.z.read(r['chart_part'])).findall('.//c:ser',C)[int(r['series_id'].split('_s')[1])-1];self.assertEqual(node.find('c:val/c:numRef/c:f',C).text,r['value_ref']);vp={int(n.get('idx')):Decimal(n.find('c:v',C).text)for n in node.findall('c:val/c:numRef/c:numCache/c:pt',C)};self.assertEqual(vp[r['point_index']],raw);cs,cc=m.ref_cells(r['category_ref']);v=x.value(cs,cc[r['point_index']]);category=str(int(v))if isinstance(v,Decimal)and v==int(v)else str(v);self.assertEqual(category,r['category_literal'])
    if r['universal_category']is None:self.assertEqual(r['category_basis'],'reviewed_source_grid')
    else:self.assertEqual(category,r['universal_category'])
    if r['name_ref'].startswith('source_headers:'):headers=json.loads(r['name_ref'].split(':',1)[1]);self.assertEqual(' | '.join(x.value(s,a)for a,_ in headers),r['series_literal'])
    else:
     ns,nc=m.ref_cells(r['name_ref']);v=x.value(ns,nc[0]);self.assertEqual(str(int(v))if isinstance(v,Decimal)and v==int(v)else str(v),r['series_literal'])
  finally:
   for x in xs.values():x.z.close()
 def test_cohort178_bound_headers_merges_standard_area_and98_pairs(self):
  x=XML(m.ROOT/BOOKS['2026'][0]);pairs=0
  try:
   for sheet,ranges in[('Anexo I. Gr 4A',[(range(22,62),'D','E'),(range(22,62),'F','G')]),('Anexo I. Gr 4B',[(range(33,51),'D','E')])]:
    for rows,upper,lower in ranges:
     for rr in rows:self.assertLessEqual(x.value(sheet,lower+str(rr)),x.value(sheet,upper+str(rr)));pairs+=1
   self.assertEqual(pairs,98);self.assertEqual(x.node('Anexo I. Gr 4A','B22').find('m:f',M).text,'1935+30');self.assertEqual(x.node('Anexo I. Gr 4B','B22').find('m:f',M).text,'1946+30')
   rows=[r for r in self.rows if r['book']=='2026'];self.assertEqual(len(rows),178)
   for r in rows:
    p=json.loads(r['quantity_evidence']);year=int(r['category_literal']);self.assertEqual(p['time_coordinate']['plot_year'],year);self.assertEqual(p['time_coordinate']['cohort_birth_year'],year-30);self.assertIsNone(p['time_coordinate']['period_year']);self.assertEqual(r['unit'],'children_per_woman');self.assertEqual(p['area_grouping'],'standard');self.assertTrue(p['not_band_width']);self.assertTrue(p['no_confidence_interval_claim']);self.assertEqual(p['original_plot_label'],r['universal_series']);q=m.query(DB,book='2026',block=r['block'],series_id=r['series_id'],point_index=r['point_index']);self.assertEqual(q['bound'],p['bound']);self.assertEqual(q['resolved_bound_label'],r['series_literal']);self.assertEqual(q['original_plot_label'],r['universal_series'])
    root=ET.fromstring(x.z.read(r['chart_part']));node=root.findall('.//c:ser',C)[int(r['series_id'].split('_s')[1])-1];parent=next(a for a in root.iter()if node in list(a));self.assertEqual(parent.find('c:grouping',C).get('val'),'standard')
   self.assertEqual(min(json.loads(r['quantity_evidence'])['time_coordinate']['cohort_birth_year']for r in rows),1935);self.assertEqual(max(json.loads(r['quantity_evidence'])['time_coordinate']['cohort_birth_year']for r in rows),1974)
  finally:x.z.close()
 def test42_percent_changes_from_explicit_source_levels_and_years(self):
  x=XML(m.ROOT/BOOKS['2025'][0]);rows=[r for r in self.rows if r['block']=='Gráfico 1B'];self.assertEqual(len(rows),42)
  try:
   for r in rows:
    cell=r['source_cell'];col=cell[0];rr=int(cell[1:]);current=x.value('Gráfico 1A',cell);previous=x.value('Gráfico 1A',col+str(rr-1));self.assertEqual(x.value('Gráfico 1A','B'+str(rr)),x.value('Gráfico 1B','B'+str(rr)));self.assertEqual(x.value('Gráfico 1A','B'+str(rr))-x.value('Gráfico 1A','B'+str(rr-1)),1)
    with localcontext()as ctx:ctx.prec=60;computed=100*(current/previous-1)
    self.assertLessEqual(abs(r['raw_value']-computed),Decimal('1e-12'));self.assertEqual(r['unit'],'percent_yoy_fertility_change_from_verified_levels');self.assertIn('derived percent change',json.loads(r['quantity_evidence'])['unit_basis'])
  finally:x.z.close()
 def test_vat_expense_ai_retirement_and_growth_contracts(self):
  x=XML(m.ROOT/BOOKS['2025'][0])
  try:
   weights=[r for r in self.rows if r['block']=='Gráfico 39'];self.assertEqual(len(weights),13);self.assertLessEqual(abs(sum((r['raw_value']for r in weights),Decimal(0))-100),Decimal('1e-12'))
   for col in'CDEFG':self.assertLessEqual(abs(sum((x.value('Gráfico 46',col+str(rr))for rr in range(5,8)),Decimal(0))-x.value('Gráfico 46',col+'8')),Decimal('1e-14'))
   for r in self.rows:
    p=json.loads(r['quantity_evidence'])
    if r['block']=='Gráfico RE 2.3':self.assertEqual(p['population'],'companies with at least 10 employees');self.assertEqual(p['source_year'],2024);self.assertEqual(p['ue8'],['AT','DE','BE','DK','FI','FR','NL','SE']);self.assertTrue(p['aggregation_weighting_not_inferred'])
    if r['block']=='Gráfico 46':self.assertEqual(r['unit'],'percent_period_average_annual_ss_pension_expenditure_growth'if r['series_literal']=='Gasto'else'percentage_points_period_average_annual_ss_pension_expenditure_growth');self.assertEqual(p['period_header'],'Promedio');self.assertEqual(p['annual_growth_context_pdf']['page'],96)
    if r['block']in['Gráfico 38','Gráfico 41']:
     self.assertEqual(r['metric_literal'],r['series_literal']);self.assertEqual(r['unit'],'percent_mean_vat_rate_as_labelled'if r['series_literal']=='Tipo medio'else'euros_mean_expenditure_as_labelled');self.assertTrue(p['no_per_capita_assumption']);self.assertTrue(p['no_2025_price_base_assumption'])
    if r['block']=='Gráfico RE 3.3':self.assertEqual(r['unit'],'percent_total_retirement_entries_as_published');self.assertEqual(p['bounded_caption_conflict']['published_denominator'],'total retirement entries');self.assertTrue(p['bounded_caption_conflict']['not_conditional_on_delayed_retirements']);self.assertTrue(p['bounded_caption_conflict']['not_assumed_exhaustive']);self.assertEqual(p['bounded_caption_conflict']['source_B3'],'GRÁFICO RE_3.2')
  finally:x.z.close()
 def test_broken_age_references_retained_and_resolved_against_explicit_peer(self):
  for sid,original,ref in[('chart68_s4',"'GR 39 Y 40'!#REF!","'Gráfico 38'!$D$5:$D$17"),('chart73_s2',"'GR 42 Y 43'!#REF!","'Gráfico 41'!$B$6:$B$18")]:
   rows=[r for r in self.rows if r['series_id']==sid];self.assertEqual(len(rows),13)
   for r in rows:p=json.loads(r['quantity_evidence']);self.assertEqual(p['original_broken_category_ref'],original);self.assertEqual(p['resolved_category_ref'],ref);self.assertIsNone(r['universal_category']);self.assertEqual(r['category_basis'],'reviewed_source_grid')
 def test_diagnostics295_have_no_origin_or_causal_claim_and_no_admissions(self):
  with duckdb.connect(str(DB),read_only=True)as c:
   rows=c.execute('SELECT book,row_number,status,evidence FROM final_family_diagnostics').fetchall();self.assertEqual(len(rows),295);self.assertFalse(c.execute('SELECT book,row_number FROM final_family_diagnostics INTERSECT SELECT * FROM final_family_usable_keys').fetchall());self.assertEqual(Counter(r[2]for r in rows),Counter({'unresolved_fiscal_literal_vector_without_category_alignment':205,'numeric_vector_reconciled_but_metric_identity_unresolved':90}))
  for book,_,status,text in rows:
   p=json.loads(text)
   if book=='2025':self.assertTrue(p['category_alignment_unproven']);self.assertFalse(p['literal_has_category']);self.assertIn('No reassignment',p['not_claimed_origin']);self.assertTrue(p['own_source_grid'])
   else:self.assertTrue(p['metric_not_accredited']);self.assertTrue(p['no_causal_interpretation']);self.assertEqual(p['candidate_variable_header'],'excess')
 def test_full_recalculation_partition_internal_exclusion_and_legacy_fingerprints(self):
  with duckdb.connect(str(m.PREVIOUS),read_only=True)as a,duckdb.connect(str(DB),read_only=True)as b:
   self.assertEqual(len(m.check_new(b)),330)
   for name in['remaining_usable_keys','remaining_internal_observations','family_observations','family_sources','pending_family_diagnostics','pending_family_source_incidents','next_deferrals','ready_quarantine']:
    aq=a.execute('SELECT * FROM '+name);schema=aq.description;data=aq.fetchall();bq=b.execute('SELECT * FROM '+name);self.assertEqual(schema,bq.description,name);self.assertEqual(Counter(data),Counter(bq.fetchall()),name)
   self.assertEqual(b.execute('SELECT count(*)FROM final_family_usable_keys').fetchone()[0],41572);self.assertEqual(b.execute('SELECT count(*)FROM final_family_numeric_queue').fetchone()[0],5641);self.assertFalse(b.execute('SELECT * FROM remaining_internal_keys INTERSECT SELECT * FROM final_family_usable_keys').fetchall());self.assertFalse(b.execute('SELECT book,row_number FROM ready_quarantine INTERSECT SELECT * FROM final_family_usable_keys').fetchall());self.assertFalse(b.execute('SELECT n.book,n.row_number FROM final_family_numeric_queue n JOIN remaining_numeric_queue p USING(book,row_number)WHERE n.final_family_reason IS DISTINCT FROM p.remaining_reason').fetchall());self.assertFalse(b.execute("SELECT t.book,t.row_number FROM final_family_observations t JOIN remaining_numeric_queue p USING(book,row_number)WHERE p.remaining_reason IN('drawing_helper_not_economic_series','forecast_marker_not_measure','internal_only_scope')").fetchall())
 def test_boundary_headers_grouping_merge_pair_and_cohort_formula_fail_closed(self):
  row=self.candidate('2026','chart53_s4')
  for kind in['header','merge','stacking','pair','formula']:
   r=m.FinalResolver()
   try:
    self.assertIsNotNone(r.review_final(row)[1]);s=r.sources['2026'];part=row['parte_grafico_imagen'];root=ET.fromstring(s.xml.parts[part]);node=root.findall('.//c:ser',C)[3]
    if kind=='header':s.sheet(row['bloque'])[1]['E21']='Máximo'
    elif kind=='merge':
     path=s.xml.sheets[row['bloque']];sheet=ET.fromstring(s.xml.parts[path]);merges=sheet.find('m:mergeCells',M)
     for n in list(merges):
      if n.get('ref')=='D20:E20':merges.remove(n)
     s.xml.parts[path]=ET.tostring(sheet)
    elif kind=='stacking':next(p for p in root.iter()if node in list(p)).find('c:grouping',C).set('val','stacked')
    elif kind=='pair':s.sheet(row['bloque'])[0]['D22'].find('m:v',M).text='0'
    else:s.sheet(row['bloque'])[1]['B22']='=1935+31'
    s.xml.parts[part]=ET.tostring(root);r.bindings.clear();self.assertIsNone(r.review_final(row)[1],kind)
   finally:r.close()
 def test_rate_equation_and_broken_reference_recovery_fail_closed(self):
  row=self.candidate('2025','chart2_s1');r=m.FinalResolver()
  try:
   self.assertIsNotNone(r.review_final(row)[1]);r.sources['2025'].sheet('Gráfico 1A')[0]['C5'].find('m:v',M).text='0'
   with self.assertRaises(ValueError):r.percent_change_proof()
  finally:r.close()
  row=self.candidate('2025','chart68_s4')
  for kind in['ageheader','broken','peer','axis']:
   r=m.FinalResolver()
   try:
    self.assertIsNotNone(r.review_final(row)[1]);s=r.sources['2025'];part=row['parte_grafico_imagen'];root=ET.fromstring(s.xml.parts[part]);node=root.findall('.//c:ser',C)[3]
    if kind=='ageheader':s.sheet('Gráfico 38')[1]['D4']='Otra categoría'
    elif kind=='broken':node.find('c:cat/c:multiLvlStrRef/c:f',C).text="'Otro'!#REF!"
    elif kind=='peer':root.findall('.//c:ser',C)[0].find('c:cat/c:strRef/c:f',C).text="'Gráfico 38'!$D$6:$D$18"
    else:
     axis=next(a for a in root.findall('.//c:valAx',C)if a.find('c:axId',C).get('val')==m.PROFILES['2025','Gráfico 38','chart68_s4']['axis_id']);axis.find('c:numFmt',C).set('formatCode','0%')
    s.xml.parts[part]=ET.tostring(root);r.bindings.clear();self.assertIsNone(r.review_final(row)[1],kind)
   finally:r.close()
 def test_exact_queries_no_fallback_data_and_view_tampering(self):
  for row in self.rows:self.assertEqual(m.query(DB,book=row['book'],block=row['block'],series_id=row['series_id'],point_index=row['point_index'])['row_number'],row['row_number'])
  with self.assertRaises(ValueError):m.query(DB,book='2026',block='Anexo I. Gr 4A',series_id='chart53_s1',point_index=True)
  with self.assertRaises(LookupError):m.query(DB,book='2025',block='Gráfico 30A',series_id='chart55_s1',point_index=0)
  with patch.object(m,'PREVIOUS_SHA','0'*64):
   with duckdb.connect(str(DB),read_only=True)as c:
    with self.assertRaises(ValueError):m.collect(c)
  with tempfile.TemporaryDirectory(dir=m.ROOT/'tmp')as directory:
   db=Path(directory)/'bad.duckdb';shutil.copyfile(DB,db)
   with duckdb.connect(str(db))as c:c.execute('UPDATE final_family_observations SET value=value+1 WHERE row_number=(SELECT min(row_number)FROM final_family_observations)')
   with duckdb.connect(str(db),read_only=True)as c:
    with self.assertRaises(ValueError):m.check_new(c)
   shutil.copyfile(DB,db)
   with duckdb.connect(str(db))as c:c.execute('CREATE OR REPLACE VIEW final_family_usable_keys AS SELECT * FROM remaining_usable_keys UNION ALL SELECT * FROM remaining_internal_keys')
   with duckdb.connect(str(db),read_only=True)as c:
    with self.assertRaises(ValueError):m.check_new(c)
if __name__=='__main__':unittest.main()
