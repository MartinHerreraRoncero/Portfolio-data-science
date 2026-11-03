import unittest,json,tempfile,shutil,xml.etree.ElementTree as ET
from pathlib import Path
from decimal import Decimal
from unittest.mock import patch
import duckdb
from test_ready_recovery_semantics import XML
from experiments import source_recovery as m
DB=m.ROOT/'evaluation/source_recovery_2026-11-01/run_v5/source_recovery.duckdb'
class TestSourceRecovery(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  if not DB.exists():raise RuntimeError('Build the source recovery fixture first')
  with duckdb.connect(str(DB),read_only=True)as c:
   q=c.execute('SELECT * FROM source_recovery_observations ORDER BY book,row_number');ns=[x[0]for x in q.description];cls.rows=[dict(zip(ns,r))for r in q.fetchall()]
 def candidate(self,sid):
  with duckdb.connect(str(m.PREVIOUS),read_only=True)as c:
   q=c.execute('SELECT * FROM closure_family_numeric_queue WHERE serie_id=?ORDER BY row_number LIMIT 1',[sid]);v=q.fetchone();self.assertIsNotNone(v);return dict(zip([x[0]for x in q.description],v))
 def positive(self,r,row,family):
  status,value=r.recover(row,family);self.assertEqual(status,'usable_recovered_source');self.assertIsNotNone(value)
 def test_independent_selection1170_without_reintroducing_wrong_net(self):
  with duckdb.connect(str(m.PREVIOUS),read_only=True)as c:
   expected=set(c.execute("SELECT book,row_number FROM closure_family_numeric_queue WHERE book='2025'AND(bloque IN('Gráfico 15','Gráfico 23')OR serie_id IN('chart55_s1','chart56_s1')OR(bloque='Gráfico 78'AND celda_valor<>'Q7'))").fetchall())
  self.assertEqual(len(expected),1170);self.assertEqual({(a['book'],a['row_number'])for a in self.rows},expected);self.assertEqual(len(self.rows),1170)
 def test_all_recovered_raw_xml_cache_and_original_metadata(self):
  x=XML(m.ROOT/m.read(m.PLAN)['families']['gr15_recovery_proposal.json']['source_file'])
  try:
   with duckdb.connect(str(DB),read_only=True)as c:
    q=c.execute('SELECT * FROM staging_records');ns=[a[0]for a in q.description];stage={(a['book'],a['row_number']):a for a in[dict(zip(ns,r))for r in q.fetchall()]}
   for row in self.rows:
    raw=x.value(row['source_sheet'],row['source_cell']);self.assertEqual(raw,row['raw_value']);self.assertEqual(row['value'],raw/(1000 if row['block']=='Gráfico 78'else 1));old=stage[row['book'],row['row_number']];self.assertEqual(row['universal_category'],old['periodo_categoria_literal']);self.assertEqual(row['universal_series'],old['serie_literal']);self.assertEqual(row['universal_value_cell'],old['celda_valor']);self.assertEqual(Decimal(old['decimal_XML_texto']),raw)
    root=ET.fromstring(x.z.read(row['chart_part']));node=root.findall('.//c:ser',m.C)[int(row['series_id'].split('_s')[1])-1];tag='c:yVal'if row['block']in['Gráfico 15','Gráfico 30A','Gráfico 30B']else'c:val';cache={int(p.get('idx')):p.find('c:v',m.C).text for p in node.findall(tag+'/c:numRef/c:numCache/c:pt',m.C)};self.assertEqual(Decimal(cache[row['point_index']]),raw)
  finally:x.z.close()
 def test_paired1069_units_coordinates_years_periods_and_no_country_inference(self):
  x=XML(m.ROOT/m.read(m.PLAN)['families']['gr15_recovery_proposal.json']['source_file'])
  try:
   with duckdb.connect(str(DB),read_only=True)as c:q=c.execute('SELECT * FROM source_recovery_scatter_pairs');ns=[a[0]for a in q.description];rows=[dict(zip(ns,r))for r in q.fetchall()]
   self.assertEqual(len(rows),1069);self.assertEqual(sum(r['family']=='gr15'for r in rows),1007)
   for r in rows:
    sheet='Gráfico 15'if r['family']=='gr15'else'Gráfico 30A'if r['series_id']=='chart55_s1'else'Gráfico 30B';self.assertEqual(x.value(sheet,r['x_source_cell']),r['x_value']);self.assertEqual(x.value(sheet,r['y_source_cell']),r['y_value']);p=json.loads(r['evidence'])
    if r['family']=='gr15':
     self.assertEqual(r['x_unit'],r['y_unit']);self.assertEqual(r['x_unit'],'percent_annual_change');self.assertEqual(r['x_period'],str(int(x.value(sheet,'B'+str(r['point_index']+6)))));self.assertEqual(r['geography_literal'],'JPN'if r['series_id']=='chart36_s2'else None);self.assertTrue(p['not_group_aggregate']);self.assertEqual(p['source_scope'],'underlying_excel_scatter_points')
    else:
     self.assertEqual((r['x_period'],r['y_period']),('2018-2019','2022-2023'));self.assertEqual(r['geography_literal'],x.value(sheet,'C'+str(r['point_index']+5)));self.assertEqual(p['transition_horizon'],'consecutive_quarters');self.assertIsNone(p['seasonal_adjustment']);self.assertTrue(p['quarter_count_per_average_not_inferred'])
  finally:x.z.close()
 def test_plot_bounds998_and9_underlying_outliers_are_explicit(self):
  with duckdb.connect(str(DB),read_only=True)as c:rows=c.execute("SELECT book,row_number,x_value,y_value,within_plot_bounds FROM source_recovery_scatter_pairs WHERE family='gr15'").fetchall()
  self.assertEqual(sum(r[4]for r in rows),998);self.assertEqual(sum(not r[4]for r in rows),9)
  for b,n,x,y,inside in rows:self.assertEqual(inside,-10<=x<=10 and -4<=y<=4)
  b,n,*_=next(r for r in rows if not r[4]);self.assertFalse(m.query_scatter(DB,book=b,row_number=n)['within_plot_bounds'])
  with self.assertRaises(LookupError):m.query_scatter(DB,book=b,row_number=n,require_plot_bounds=True)
 def test_current_population_reference_flags_do_not_assign_countries(self):
  raw=json.loads((m.AUDITOR/'worldbank_SP.POP.1564.TO.json').read_text(encoding='utf-8'));data=raw[1];levels={(a['countryiso3code'],int(a['date'])):a['value']for a in data if a['value']is not None};countries={a[0]for a in levels};matched=0;missing=0
  with duckdb.connect(str(DB),read_only=True)as c:rows=c.execute("SELECT y_value,y_period,evidence FROM source_recovery_scatter_pairs WHERE family='gr15'").fetchall()
  for y,year,ev in rows:
   year=int(year);candidates=[co for co in countries if(co,year)in levels and(co,year-1)in levels and abs(100*(levels[co,year]/levels[co,year-1]-1)-float(y))<1e-10];p=json.loads(ev);self.assertEqual(p['current_y_formula_matches_any_reference_country_same_year'],bool(candidates));self.assertEqual(p['reference_country_match_count'],len(candidates));matched+=bool(candidates);missing+=not bool(candidates);self.assertNotIn('country_candidate',p);self.assertTrue(p['reference_comparison_not_joint_xy_country_validation'])
  self.assertEqual((matched,missing),(931,76))
 def test_treasury74_originals_and_one_external_replacement(self):
  with duckdb.connect(str(DB),read_only=True)as c:
   self.assertEqual(c.execute("SELECT count(*)FROM source_recovery_observations WHERE block='Gráfico 78'").fetchone()[0],74);wrong=c.execute("SELECT book,row_number,replacement_observation_id FROM source_recovery_replacements WHERE replacement_kind='authoritative_external_replacement'").fetchall();self.assertEqual({r[:2]for r in wrong},{('2025',37525),('2025',37582)})
   self.assertEqual(c.execute("SELECT decimal_XML_texto FROM staging_records WHERE book='2025'AND row_number=37525").fetchone()[0],'75318')
  for b,n,oid in wrong:
   with self.assertRaises(LookupError):m.query_safe(DB,book=b,row_number=n)
   result=m.query_observation(DB,observation_id=oid);self.assertEqual(Decimal(result['value']),Decimal('75.138'));self.assertEqual(Decimal(result['raw_value']),Decimal(75138));self.assertEqual(result['source_variant'],'authoritative_external_correction_not_airef_original');self.assertEqual(result['pdf_page'],3)
 def test_fiscal3444_repaired_and957_verified_publication_points(self):
  with duckdb.connect(str(DB),read_only=True)as c:
   self.assertEqual(c.execute('SELECT count(*)FROM source_recovery_retired_objects').fetchone()[0],3444);removed=set(c.execute('SELECT * FROM closure_family_usable_keys EXCEPT SELECT * FROM source_recovery_usable_keys').fetchall());expected=set(c.execute("SELECT book,row_number FROM closure_family_public_observations WHERE book='2025'AND source_sheet='Gráfico 100'AND block<>'Gráfico 100'").fetchall());self.assertEqual(removed,expected);self.assertEqual(len(removed),1722)
   self.assertEqual(c.execute('SELECT count(*)FROM source_recovery_fiscal_points').fetchone()[0],957);self.assertEqual(c.execute('SELECT count(DISTINCT figure)FROM source_recovery_fiscal_points').fetchone()[0],12)
   self.assertFalse(c.execute("SELECT * FROM source_recovery_public_observations WHERE source_sheet='Gráfico 100'AND block<>'Gráfico 100'").fetchall());sample=next(iter(removed))
  with self.assertRaises(LookupError):m.query_safe(DB,book=sample[0],row_number=sample[1])
  result=m.query_figure(DB,book='2025',figure=112,series='Base',year=2050);self.assertEqual(Decimal(result['value']).quantize(Decimal('.1')),Decimal('128.9'))
  with self.assertRaises(LookupError):m.query_figure(DB,book='2025',figure=112,series='Base',year=2010)
  with self.assertRaises(LookupError):m.query_figure(DB,book='2025',figure=111,series='Emisiones netas cero (Net0)',year=2050)

 def test_activity27_is_percent_and_keeps_original_publication_snapshot(self):
  rows=[a for a in self.rows if a['block']=='Gráfico 23'];self.assertEqual(len(rows),27)
  for a in rows:
   self.assertEqual(a['unit'],'percent_labour_force_of_population_aged60_74');p=json.loads(a['quantity_evidence']);self.assertTrue(p['no_value_replacement_from_later_technical_document']);self.assertTrue(p['no_inferred_specific_country_year_average']);self.assertTrue(p['no_inferred_pca_score']);self.assertEqual(p['source_country_literal'],a['category_literal']);self.assertEqual(a['value'],a['raw_value'])
 def test_positive_baselines_and_scatter_axis_reference_cache_rejections(self):
  row=self.candidate('chart36_s1');r=m.RecoveryResolver()
  try:
   s=r.sources['2025'];part=row['parte_grafico_imagen'];original=s.xml.parts[part]
   for kind in['reference','axis','cache']:
    self.positive(r,row,'gr15');root=ET.fromstring(original);node=root.findall('.//c:ser',m.C)[0];index=int(row['id'].split('_p')[1])
    if kind=='reference':node.find('c:xVal/c:numRef/c:f',m.C).text="'Gráfico 15'!$E$6:$E$1012"
    elif kind=='axis':root.find('.//c:valAx/c:scaling/c:max',m.C).set('val','11')
    else:
     pt=next(a for a in node.findall('c:yVal/c:numRef/c:numCache/c:pt',m.C)if int(a.get('idx'))==index);pt.find('c:v',m.C).text=str(Decimal(pt.find('c:v',m.C).text)+1)
    s.xml.parts[part]=ET.tostring(root);self.assertIsNone(r.recover(row,'gr15')[1],kind);s.xml.parts[part]=original
   self.positive(r,row,'gr15')
  finally:r.close()
 def test_transition_null_name_and_period_and_treasury_scale_fail_closed(self):
  r=m.RecoveryResolver()
  try:
   row=self.candidate('chart55_s1');self.positive(r,row,'gr30');self.assertIsNone(row['serie_literal']);s=r.sources['2025'];old=s.text
   with patch.object(s,'text',side_effect=lambda sh,cell:'Promedio 2017-2019'if sh=='Gráfico 30A'and cell=='D4'else old(sh,cell)):self.assertIsNone(r.recover(row,'gr30')[1])
   self.positive(r,row,'gr30');row=self.candidate('chart133_s1');self.positive(r,row,'gr78');part=row['parte_grafico_imagen'];original=s.xml.parts[part];root=ET.fromstring(original)
   axis=next(a for a in root.findall('.//c:valAx',m.C)if a.find('c:axId',m.C).get('val')=='83499887');axis.find('c:dispUnits/c:builtInUnit',m.C).set('val','millions');s.xml.parts[part]=ET.tostring(root);self.assertIsNone(r.recover(row,'gr78')[1]);s.xml.parts[part]=original;self.positive(r,row,'gr78')
  finally:r.close()
 def test_frozen_source_and_publication_manifests_reject_changed_evidence(self):
  with duckdb.connect(str(DB),read_only=True)as c:
   self.assertEqual(len(m.file_proofs(c)),6)
   for name in['PREVIOUS_SHA','PLAN_SHA','SOURCE_SHA','REFERENCE_SHA','FISCAL_SHA','FISCAL_BAD_SHA','TEMPERATURE_LINKS_SHA']:
    with patch.object(m,name,'0'*64):
     with self.assertRaises(ValueError,msg=name):m.file_proofs(c)
 def test_full_new_layer_partition_and_all_previous_tables_unchanged(self):
  with duckdb.connect(str(DB),read_only=True)as c:
   self.assertEqual(len(m.check_new(c)['rows']),1170);self.assertEqual(c.execute('SELECT count(*)FROM source_recovery_usable_keys').fetchone()[0],41957);self.assertEqual(c.execute('SELECT count(*)FROM source_recovery_public_observations').fetchone()[0],41958);self.assertEqual(c.execute('SELECT count(*)FROM source_recovery_numeric_queue').fetchone()[0],5256)
   self.assertEqual(dict(c.execute('SELECT state,count(*)FROM source_recovery_resolution GROUP BY state').fetchall()),dict(admitted_validated=1170,blocked=1221,excluded=421,internal_validated=168,repaired_by_verified_figure_registry=3444,replaced_by_official_fact=2));c.execute("ATTACH '"+m.PREVIOUS.as_posix()+"'AS previous(READ_ONLY)")
   names=[a[0]for a in c.execute("SELECT table_name FROM information_schema.tables WHERE table_catalog='previous'AND table_type='BASE TABLE'").fetchall()];self.assertGreater(len(names),80)
   for table in names:self.assertFalse(c.execute(f'SELECT * FROM main.{table} EXCEPT ALL SELECT * FROM previous.main.{table}').fetchall(),table);self.assertFalse(c.execute(f'SELECT * FROM previous.main.{table} EXCEPT ALL SELECT * FROM main.{table}').fetchall(),table)
 def test_current_safe_queries_reject_legacy_quarantine_internal_and_fallback(self):
  with duckdb.connect(str(DB),read_only=True)as c:
   keys=c.execute('SELECT book,row_number FROM source_recovery_quarantine WHERE was_admitted ORDER BY row_number LIMIT 2').fetchall()+c.execute('SELECT book,row_number FROM closure_quarantine ORDER BY row_number LIMIT 2').fetchall()+c.execute('SELECT book,row_number FROM remaining_internal_observations LIMIT 1').fetchall()+c.execute('SELECT book,row_number FROM ready_quarantine LIMIT 1').fetchall()
  for b,n in keys:
   with self.assertRaises(LookupError):m.query_safe(DB,book=b,row_number=n)
  for a in self.rows[::100]:self.assertEqual(m.query_safe(DB,book=a['book'],row_number=a['row_number'])['observation_id'],m.aid(a['book'],a['row_number']))
  with self.assertRaises(ValueError):m.query_safe(DB,book='2025',row_number=True)
  with self.assertRaises(LookupError):m.query_observation(DB,observation_id='official:unknown')
 def test_tampered_new_values_and_unsafe_public_view_are_detected(self):
  with tempfile.TemporaryDirectory(dir=m.ROOT/'tmp')as directory:
   bad=Path(directory)/'bad.duckdb';shutil.copyfile(DB,bad)
   with duckdb.connect(str(bad))as c:c.execute('UPDATE source_recovery_observations SET value=value+1 WHERE row_number=(SELECT min(row_number)FROM source_recovery_observations)')
   with duckdb.connect(str(bad),read_only=True)as c:
    with self.assertRaises(ValueError):m.check_new(c)
   shutil.copyfile(DB,bad)
   with duckdb.connect(str(bad))as c:c.execute("CREATE OR REPLACE VIEW source_recovery_public_observations AS SELECT *,'airef:'||book||':'||row_number observation_id FROM closure_family_public_observations")
   with duckdb.connect(str(bad),read_only=True)as c:
    with self.assertRaises(ValueError):m.check_new(c)
if __name__=='__main__':unittest.main()
