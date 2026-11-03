import unittest,json,tempfile,shutil,xml.etree.ElementTree as ET
from pathlib import Path
from decimal import Decimal,ROUND_HALF_UP
from unittest.mock import patch
import duckdb
from test_ready_recovery_semantics import XML,M
from experiments import closure_family_semantics as m
from experiments.excel_pilot import BOOKS
DB=m.ROOT/'evaluation/closure_family_semantics_2026-10-31/run_v3/closure_families.duckdb'
C={'c':'http://schemas.openxmlformats.org/drawingml/2006/chart'}
S25={'chart18_s1','chart18_s2','chart63_s1','chart89_s1','chart92_s1','chart150_s1','chart125_s1','chart125_s2','chart125_s3'}
S26={'chart44_s4','chart17_s3','chart25_s2'}
class TestClosureFamilies(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  if not DB.exists():raise RuntimeError('Build closure fixture first')
  with duckdb.connect(str(DB),read_only=True)as c:
   q=c.execute('SELECT * FROM closure_family_observations ORDER BY book,row_number');ns=[x[0]for x in q.description];cls.rows=[dict(zip(ns,v))for v in q.fetchall()]
 def candidate(self,book,sid=None,block=None,cell=None):
  with duckdb.connect(str(m.PREVIOUS),read_only=True)as c:
   sql='SELECT * FROM reviewed_family_numeric_queue WHERE book=?';args=[book]
   for name,v in[('serie_id',sid),('bloque',block),('celda_valor',cell)]:
    if v is not None:sql+='AND '+name+'=?';args.append(v)
   q=c.execute(sql+' ORDER BY row_number LIMIT 1',args);v=q.fetchone();self.assertIsNotNone(v);return dict(zip([x[0]for x in q.description],v))
 def positive(self,r,row):
  status,out=r.review_closure(row);self.assertEqual(status,'usable_closure_family');self.assertIsNotNone(out);return out
 def test_independent_exact145_raw_xml_and123_cache_values(self):
  with duckdb.connect(str(m.PREVIOUS),read_only=True)as c:
   keys=set()
   for book,sids in[('2025',S25),('2026',S26)]:
    for sid in sids:keys.update(c.execute('SELECT book,row_number FROM reviewed_family_numeric_queue WHERE book=?AND serie_id=?',[book,sid]).fetchall())
   keys.update(c.execute("SELECT book,row_number FROM reviewed_family_numeric_queue WHERE book='2025'AND bloque='Gráfico 64'AND celda_valor IN('C5','D5','E5')").fetchall());keys.update(c.execute("SELECT book,row_number FROM reviewed_family_numeric_queue WHERE book='2025'AND bloque='Cuadro 14'AND(celda_valor IN('D7','E7','F7','G7')OR substr(celda_valor,1,1)IN('E','F','G')AND cast(substr(celda_valor,2)AS INTEGER)BETWEEN8 AND13)".replace('BETWEEN8','BETWEEN 8').replace('AND13','AND 13')).fetchall())
  self.assertEqual(len(keys),145);self.assertEqual(keys,{(r['book'],r['row_number'])for r in self.rows});xs={b:XML(m.ROOT/BOOKS[b][0])for b in['2025','2026']};chart_count=0
  try:
   for r in self.rows:
    x=xs[r['book']];raw=x.value(r['source_sheet'],r['source_cell']);self.assertEqual(raw,r['raw_value']);self.assertEqual(r['value'],raw*r['multiplier']/r['divisor']);self.assertEqual(json.loads(r['quantity_evidence'])['unit'],r['unit'])
    if r['series_id']is not None:
     chart_count+=1;node=ET.fromstring(x.z.read(r['chart_part'])).findall('.//c:ser',C)[int(r['series_id'].split('_s')[1])-1];vp=node.find('c:val/c:numRef/c:numCache',C)
     if vp is None:vp=node.find('c:val/c:numLit',C)
     cache={int(n.get('idx')):Decimal(n.find('c:v',C).text)for n in vp.findall('c:pt',C)};self.assertEqual(cache[r['point_index']],raw)
   self.assertEqual(chart_count,123)
  finally:
   for x in xs.values():x.z.close()
 def test_decisions_exhaustive4849_disjoint_and_frozen(self):
  inv=m.load_inventory();self.assertEqual(len(inv),4849);self.assertEqual(len({(a['book'],a['row_number'])for a in inv}),4849);self.assertEqual(sum(a['original_queue']for a in inv),4751)
  with duckdb.connect(str(DB),read_only=True)as c:
   self.assertEqual(dict(c.execute('SELECT disposition,count(*)FROM closure_family_decisions GROUP BY ALL').fetchall()),dict(usable=145,excluded=421,internal_validated=168,blocked=4115));self.assertEqual(c.execute('SELECT count(*)FROM closure_family_decisions WHERE NOT coverage_complete').fetchone()[0],0);self.assertEqual(c.execute('SELECT count(*)FROM closure_family_numeric_queue').fetchone()[0],4704);self.assertEqual(c.execute('SELECT count(*)FROM closure_family_public_observations').fetchone()[0],42509);self.assertFalse(c.execute('SELECT book,row_number FROM closure_family_public_observations GROUP BY ALL HAVING count(*)<>1').fetchall());self.assertFalse(c.execute('SELECT * FROM closure_family_usable_keys INTERSECT SELECT book,row_number FROM closure_family_numeric_queue').fetchall())
  with patch.object(m,'INVENTORY_SHA','wrong'):
   with self.assertRaises(ValueError):m.load_inventory()
 def test100_quarantine98_withdrawals_physical_years_chart86_preserved(self):
  x=XML(m.ROOT/BOOKS['2025'][0])
  try:
   with duckdb.connect(str(DB),read_only=True)as c:
    q=c.execute('SELECT * FROM closure_quarantine');ns=[v[0]for v in q.description];rr=[dict(zip(ns,v))for v in q.fetchall()];self.assertEqual(len(rr),100);self.assertEqual(sum(r['was_admitted']for r in rr),98)
    for r in rr:
     col=''.join(v for v in r['source_cell']if v.isalpha());self.assertEqual(str(int(x.value('Gráfico 49',col+'20'))),r['physical_column_year']);self.assertEqual(x.value('Gráfico 49',r['source_cell']),Decimal(r['raw_xml']));self.assertTrue(r['chart_year']is None or int(r['chart_year'])+1==int(r['physical_column_year']))
    self.assertEqual(c.execute("SELECT count(*)FROM indexed_observations WHERE book='2025'AND series_id IN('chart86_s1','chart86_s2')").fetchone()[0],100);self.assertEqual(c.execute("SELECT count(*)FROM indexed_observations i JOIN closure_family_usable_keys k USING(book,row_number)WHERE i.book='2025'AND i.series_id IN('chart86_s1','chart86_s2')").fetchone()[0],100);self.assertFalse(c.execute('SELECT book,row_number FROM closure_quarantine INTERSECT SELECT * FROM closure_family_usable_keys').fetchall())
  finally:x.z.close()
 def test_nawru71_exact_C_vector_marker_D_and_no_forecast_inference(self):
  x=XML(m.ROOT/BOOKS['2026'][0]);rows=[r for r in self.rows if r['block']=='Gráfico 17'];self.assertEqual(len(rows),71)
  try:
   self.assertEqual({r['category_literal']for r in rows},{str(y)for y in range(1980,2051)})
   for r in rows:
    i=r['point_index'];self.assertEqual(r['source_cell'],'C'+str(7+i));self.assertEqual(r['value'],x.value('Gráfico 17',r['source_cell']));self.assertEqual(r['universal_series'],'Previsión');p=json.loads(r['quantity_evidence']);self.assertEqual(p['bounded_name_override']['source_variable_header'],'Variable');self.assertTrue(p['bounded_name_override']['forecast_classification_not_inferred']);self.assertIn('NAWRU',r['metric_literal']);self.assertEqual(r['unit'],'percent_nawru')
  finally:x.z.close()
 def test_local_units_migration_tfp_retirement_structural_and_table22(self):
  for r in self.rows:
   sh=r['block'];proof=json.loads(r['quantity_evidence'])
   if sh=='Cuadro 14':
    rr=int(r['source_cell'][1:]);self.assertEqual(r['unit'],'percent_interest_rate_new_issues'if rr==7 else'percentage_points_interest_rate_new_issues');self.assertNotIn('gdp',r['unit']);self.assertEqual(Decimal(proof['pdf_display_value']),r['raw_value'].quantize(Decimal('.1'),rounding=ROUND_HALF_UP));self.assertTrue(proof['auxiliary_levels_not_certified'])
   if sh=='Gráfico 64':self.assertEqual(r['multiplier'],100);self.assertEqual(r['category_literal'],'2008-2014');self.assertLessEqual(abs(Decimal(proof['period_triplet']['residual'])),Decimal('1e-12'))
   if sh in['Gráfico RE 3.2','Gráfico RE 3.5']:self.assertEqual(r['unit'],'percent_total_retirement_entries');self.assertEqual(r['multiplier'],100)
   if sh=='Gráfico 9B':self.assertEqual(r['unit'],'thousand_persons_net_external_migration');self.assertIn('derived',r['semantic_scope'])
   if sh=='Gráfico RE 2.1':self.assertIn('additional_annual_us_tfp',r['unit']);self.assertIn('relative_no_AI',r['semantic_scope'])
   if sh=='Gráfico 95':self.assertEqual(r['unit'],'percentage_points_gdp');self.assertIn('2025-2040',r['semantic_scope'])
 def test_failed_periods_remain_blocked_and_raw_unchanged(self):
  with duckdb.connect(str(DB),read_only=True)as c:
   d=c.execute("SELECT status,evidence FROM closure_family_diagnostics WHERE status IN('debt_period_decomposition_conflict','debt_2050_decomposition_conflict')").fetchall();self.assertEqual(len(d),10)
   for status,s in d:self.assertGreater(abs(Decimal(json.loads(s)['residual'])),Decimal('1e-12'))
   self.assertEqual(c.execute("SELECT count(*)FROM closure_family_numeric_queue WHERE book='2026'AND bloque='Gráfico 30'").fetchone()[0],7);self.assertEqual(c.execute("SELECT count(*)FROM closure_family_numeric_queue WHERE book='2025'AND bloque='Gráfico 64'").fetchone()[0],3);self.assertEqual(c.execute("SELECT count(*)FROM closure_family_diagnostics WHERE status='published_auxiliary_interest_rate_base_conflict'").fetchone()[0],50)
 def test_exclusions421_positive_roles_and_orphan24_blocked(self):
  with duckdb.connect(str(DB),read_only=True)as c:
   m.verify_exclusions(c,m.load_inventory())
   inv=m.load_inventory();aliases=[a for a in inv if'duplicate_pair'in a];self.assertEqual(len(aliases),156);self.assertEqual(len([a for a in inv if a['disposition']=='excluded'and a['book']=='2025'and a['series_id']in['chart55_s2','chart56_s2']]),62);orphans=[a for a in inv if a['book']=='2025'and a['block']in['Gráfico 38','Gráfico 39','Gráfico 46','Gráfico 69']and a['source_sheet']is None and a['original_raw_decimal']=='1'];self.assertEqual(len(orphans),24);self.assertEqual({a['disposition']for a in orphans},{'blocked'});self.assertFalse(c.execute("SELECT d.book,d.row_number FROM closure_family_decisions d JOIN closure_family_usable_keys k USING(book,row_number)WHERE d.disposition IN('blocked','excluded','internal_validated')").fetchall())
 def test_Base3_overlay_and_media3_not_panel_and_pyramid_typed_fields(self):
  for i in range(3):
   out=m.query_safe(DB,book='2026',row_number=1831+i);self.assertEqual(out['series_literal'],'Base');p=out['metadata_overlay'];self.assertEqual(p['first_registration_year'],2024-i);self.assertEqual(p['currency_price_year'],2024);self.assertTrue(p['not_longitudinal_panel']);self.assertTrue(p['not_confidence_interval']);self.assertIn('lower_edge',out['safe_quantity_role']);self.assertEqual(Decimal(out['value']),Decimal(p['original_raw']))
  mean=[r for r in self.rows if r['block']=='Gráfico RE_2.4.'];self.assertEqual(len(mean),3)
  for r in mean:p=json.loads(r['quantity_evidence']);self.assertEqual(p['first_registration_year'],2024-r['point_index']);self.assertTrue(p['not_longitudinal_panel']);self.assertEqual(r['unit'],'thousand_current_2024_euros')
  with duckdb.connect(str(DB),read_only=True)as c:rr=c.execute("SELECT book,row_number FROM extended_family_observations WHERE unit='thousand_persons_signed_plot'ORDER BY value LIMIT 1").fetchone()
  out=m.query_safe(DB,book=rr[0],row_number=rr[1]);self.assertTrue(out['safe_current_view']);self.assertFalse(out['value_is_population_count']);self.assertLess(Decimal(out['plot_value_thousands']),0);self.assertGreaterEqual(Decimal(out['population_thousands']),0)
 def test_safe_exact_api_rejects_all_quarantines_internal_and_unknown(self):
  with duckdb.connect(str(DB),read_only=True)as c:
   self.assertFalse(c.execute('SELECT book,row_number FROM closure_family_public_observations INTERSECT SELECT book,row_number FROM closure_quarantine').fetchall());self.assertFalse(c.execute('SELECT book,row_number FROM closure_family_public_observations INTERSECT SELECT book,row_number FROM ready_quarantine').fetchall());self.assertFalse(c.execute('SELECT book,row_number FROM closure_family_public_observations INTERSECT SELECT * FROM remaining_internal_keys').fetchall());cases=c.execute('SELECT book,row_number FROM closure_quarantine ORDER BY row_number LIMIT 1').fetchall()+c.execute('SELECT book,row_number FROM ready_quarantine LIMIT 1').fetchall()+c.execute('SELECT * FROM remaining_internal_keys LIMIT 1').fetchall()
  for b,n in cases:
   with self.assertRaises(LookupError):m.query_safe(DB,book=b,row_number=n)
  with self.assertRaises(ValueError):m.query_safe(DB,book='2025',row_number=True)
  with self.assertRaises(LookupError):m.query_safe(DB,book='2025',row_number=999999)
  for r in self.rows[::10]:
   out=m.query_table(DB,book=r['book'],block=r['block'],row_number=r['row_number'])if r['series_id']is None else m.query(DB,book=r['book'],block=r['block'],series_id=r['series_id'],point_index=r['point_index']);self.assertEqual(Decimal(out['value']),r['value']);self.assertEqual(out['unit'],r['unit'])
 def test_nawru_and_metric_reference_unit_mutations_fail_closed(self):
  for mode in['C_D_swap','name','category','unit','C_raw']:
   r=m.ClosureResolver()
   try:
    row=self.candidate('2026',sid='chart25_s2');self.positive(r,row);s=r.sources['2026'];part='xl/charts/chart25.xml';root=ET.fromstring(s.xml.parts[part]);node=root.findall('.//c:ser',C)[1]
    if mode=='C_raw':
     old=s.number
     with patch.object(s,'number',side_effect=lambda sh,a:Decimal(999)if(sh,a)==('Gráfico 17','C7')else old(sh,a)):self.assertIsNone(r.review_closure(row)[1])
     continue
    if mode=='C_D_swap':node.find('c:val/c:numLit',C).clear();replacement=root.findall('.//c:ser',C)[0].find('c:val/c:numLit',C);node.find('c:val',C).remove(node.find('c:val/c:numLit',C));node.find('c:val',C).append(ET.fromstring(ET.tostring(replacement)))
    elif mode=='name':node.find('c:tx/c:v',C).text='Different'
    elif mode=='category':node.find('c:cat/c:numLit/c:pt/c:v',C).text='9999'
    else:root.find('.//c:valAx/c:numFmt',C).set('formatCode','0%')
    s.xml.parts[part]=ET.tostring(root);self.assertIsNone(r.review_closure(row)[1])
   finally:r.close()
 def test_local_table_headers_and_original_payload_fail_closed(self):
  r=m.ClosureResolver()
  try:
   row=self.candidate('2025',block='Cuadro 14',cell='E8');self.positive(r,row);s=r.sources['2025'];old=s.text
   for anchor in['D5','E6','C8','B8']:
    with patch.object(s,'text',side_effect=lambda sh,a: 'WRONG'if(sh,a)==('Cuadro 14',anchor)else old(sh,a)):self.assertIsNone(r.review_closure(row)[1])
   for field,value in[('decimal_XML_texto','0'),('serie_literal','invented'),('metrica_literal','gastoPIB'),('celda_valor','I8')]:
    a=dict(row);a[field]=value;self.assertIsNone(r.review_closure(a)[1])
  finally:r.close()
 def test_all_previous_tables_unchanged_and_public_unit_value_equals_origin(self):
  with duckdb.connect(str(DB),read_only=True)as c:
   m.check_new(c);c.execute("ATTACH '"+str(m.PREVIOUS).replace("'","''")+"' AS baseline (READ_ONLY)");names=[a[0]for a in c.execute("SELECT table_name FROM information_schema.tables WHERE table_catalog='baseline'AND table_schema='main'AND table_type='BASE TABLE'").fetchall()];self.assertGreaterEqual(len(names),70)
   for name in names:
    self.assertEqual(c.execute('DESCRIBE main."'+name+'"').fetchall(),c.execute('DESCRIBE baseline."'+name+'"').fetchall(),name)
    for a,b in[('main','baseline'),('baseline','main')]:self.assertFalse(c.execute('SELECT * FROM '+a+'."'+name+'" EXCEPT ALL SELECT * FROM '+b+'."'+name+'"').fetchall(),name)
   for name in m.BASE_OBSERVATION_TABLES+['closure_family_observations']:
    original='SELECT t.book,t.row_number,t.unit,t.value FROM '+name+' t JOIN closure_family_usable_keys k USING(book,row_number)';view="SELECT book,row_number,unit,value FROM closure_family_public_observations WHERE source_layer='"+name+"'"
    for a,b in[(original,view),(view,original)]:self.assertFalse(c.execute('('+a+')EXCEPT ALL('+b+')').fetchall(),name)
   self.assertEqual(c.execute('SELECT count(*)FROM remaining_internal_keys').fetchone()[0],168);self.assertEqual(c.execute('SELECT count(*)FROM ready_quarantine').fetchone()[0],13);self.assertEqual(c.execute('SELECT count(*)FROM next_deferrals').fetchone()[0],131);self.assertEqual(c.execute('SELECT count(*)FROM extended_family_diagnostics').fetchone()[0],6)
 def test_tamper_source_inventory_observation_decisions_and_safe_view(self):
  with patch.object(m,'PREVIOUS_SHA','wrong'):
   with duckdb.connect(str(DB),read_only=True)as c:
    with self.assertRaises(ValueError):m.check_new(c)
  for mode in['data','decision','safeview']:
   with tempfile.TemporaryDirectory()as td:
    db=Path(td)/'copy.duckdb';shutil.copyfile(DB,db)
    with duckdb.connect(str(db))as c:
     if mode=='data':c.execute('UPDATE closure_family_observations SET value=value+1 WHERE row_number=(SELECT min(row_number)FROM closure_family_observations)')
     elif mode=='decision':c.execute("UPDATE closure_family_decisions SET disposition='usable'WHERE row_number=(SELECT min(row_number)FROM closure_family_decisions WHERE disposition='blocked')")
     else:c.execute('CREATE OR REPLACE VIEW closure_family_public_observations AS SELECT * FROM indexed_observations')
     with self.assertRaises(ValueError):m.check_new(c)
if __name__=='__main__':unittest.main()
