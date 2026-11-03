import copy,json,unittest,xml.etree.ElementTree as ET
from unittest.mock import patch
from decimal import Decimal as D
import duckdb
from experiments import source_recovery_phase2 as m
DB=m.ROOT/'evaluation/source_recovery_phase2_2026-11-01/run_v6/source_recovery_phase2.duckdb'

class Recovery2Tests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.c=duckdb.connect(str(m.BASE),read_only=True);cls.r=m.Resolver();q=cls.c.execute('SELECT * FROM staging_records');cols=[a[0]for a in q.description];cls.stage={(r['book'],r['row_number']):r for r in[dict(zip(cols,v))for v in q.fetchall()]}
 @classmethod
 def tearDownClass(cls):cls.r.close();cls.c.close()
 def source_row(self,family):
  a=next(p for p in m.asset('selection.json')if p['family']==family);return copy.deepcopy(self.stage[a['book'],a['row_number']])
 def test_frozen_primary_files_and_originals(self):m.guards(self.c)
 def test_validator_normalizes_cli_string_path_before_inherited_chain(self):
  manifest=dict(code_sha256='seal',db_sha256='seal',asset_manifest_sha256=m.ASSET_SHA)
  with patch.object(m,'read',return_value=manifest),patch.object(m,'sha',return_value='seal'),patch.object(m,'validate_previous',return_value={'status':'pass'})as previous,patch.object(m.duckdb,'connect'),patch.object(m,'check_new',return_value={}):
   self.assertEqual(m.validate(str(DB))['status'],'pass');self.assertIsInstance(previous.call_args.args[0],m.Path)
 def test_independent_primary_totals_and_unique_quarter_vector(self):
  proof=m.primary_checks(self.r);self.assertEqual(proof['fertility_baseline_same_ieee754_values'],26);self.assertEqual(proof['ine_total_observed_anchor'],'1.10')
 def test_changed_official_fertility_total_rejected(self):
  original=m.asset
  def changed(name):
   a=original(name)
   if name=='proj_25naasfr_ES_2025_2050.json':a['value']['0']=999
   return a
  with patch.object(m,'asset',side_effect=changed):
   with self.assertRaisesRegex(ValueError,'fertility comparator'):m.primary_checks(self.r)
 def test_source_asset_hash_change_rejected(self):
  original=m.sha
  with patch.object(m,'sha',side_effect=lambda p:'changed'if p==m.ASSETS/'manifest.json'else original(p)):
   with self.assertRaisesRegex(ValueError,'manifest changed'):m.guards(self.c)
 def test_selection_is_exactly_blocked775_distinct(self):
  selected=m.asset('selection.json');keys={(p['book'],p['row_number'])for p in selected};blocked=set(self.c.execute("SELECT book,row_number FROM source_recovery_resolution WHERE state='blocked'").fetchall());self.assertEqual(len(keys),775);self.assertTrue(keys<=blocked)
  for p in selected:self.assertEqual(m.digest(self.stage[p['book'],p['row_number']]),p['payload_sha256'])
 def test_all_ranges_source_cache_geometry_unit_and_centre(self):
  selected=[p for p in m.asset('selection.json')if p['family']=='range'];self.assertEqual(len(selected),596)
  for p in selected:
   result=dict(zip(m.FIELDS,self.r.range(self.stage[p['book'],p['row_number']],self.c)));self.assertIn(result['quantity_role'],['depicted_range_lower_edge_not_point_forecast','depicted_range_upper_edge_not_point_forecast']);proof=json.loads(result['quantity_evidence']);self.assertIsNone(proof['confidence_level']);self.assertTrue(proof['not_point_forecast'])
  self.assertEqual(len(self.r.range_pairs),549)
 def test_changed_standard_grouping_rejected(self):
  row=self.source_row('range');s,sh,root,node,i=self.r.ser(row);part=row['parte_grafico_imagen'];old=s.xml.parts[part];changed=ET.fromstring(old);changed.find('.//c:areaChart/c:grouping',m.C).set('val','stacked');s.xml.parts[part]=ET.tostring(changed)
  try:
   with self.assertRaisesRegex(ValueError,'standard area'):self.r.range(row,self.c)
  finally:s.xml.parts[part]=old
 def test_changed_range_fill_rejected(self):
  row=self.source_row('range');s,sh,root,node,i=self.r.ser(row);part=row['parte_grafico_imagen'];old=s.xml.parts[part];changed=ET.fromstring(old);changed.find('.//c:areaChart/c:ser/c:spPr',m.C).set('changed','true');s.xml.parts[part]=ET.tostring(changed)
  try:
   with self.assertRaisesRegex(ValueError,'style'):self.r.range(row,self.c)
  finally:s.xml.parts[part]=old
 def test_changed_raw_cell_rejected(self):
  row=self.source_row('range');row['decimal_XML_texto']='42'
  with self.assertRaisesRegex(ValueError,'universal/raw'):self.r.range(row,self.c)
 def test_fertility55_scope_and_vintage_roles(self):
  roles={}
  for p in m.asset('selection.json'):
   if p['family']!='fertility':continue
   result=dict(zip(m.FIELDS,self.r.fertility_point(self.stage[p['book'],p['row_number']])));proof=json.loads(result['quantity_evidence']);self.assertEqual(proof['population_scope'],'resident_population_all_maternal_nationalities');roles[result['source_cell']]=result['quantity_role']
  self.assertEqual(len(roles),55);self.assertEqual(roles['H55'],'airef2025_model_projection_not_observed');self.assertIn('observed',roles['G55'])
 def test_wrong_fertility_header_rejected(self):
  row=self.source_row('fertility');source=self.r.sources[row['book']];values=source.sheet(row['hoja_valor'])[1];old=values['B19'];values['B19']='otra población'
  try:
   with self.assertRaisesRegex(ValueError,'header/calendar'):self.r.fertility_point(row)
  finally:values['B19']=old
 def test_all_quarters_and_annual_markers_keep_raw_original_category(self):
  quarters=set();markers=0
  for p in m.asset('selection.json'):
   if p['family']!='quarterly_debt':continue
   row=self.stage[p['book'],p['row_number']];result=dict(zip(m.FIELDS,self.r.quarterly(row)));quarters.add(result['category_literal']);markers+=row['celda_valor'].startswith('C');self.assertEqual(result['universal_category'],row['periodo_categoria_literal']);self.assertEqual(result['value'],D(row['decimal_XML_texto'])*100)
  self.assertEqual(len(quarters),49);self.assertEqual(markers,26);self.assertEqual(min(quarters),'2012-Q4');self.assertEqual(max(quarters),'2024-Q4')
 def test_changed_reference_quarter_rejected(self):
  row=self.source_row('quarterly_debt');original=m.asset
  def changed(name):
   value=original(name)
   if name=='gr61_quarter_reference_map.json':
    for p in value:p['reference_quarter']='1900-Q1'
   return value
  with patch.object(m,'asset',side_effect=changed):
   with self.assertRaisesRegex(ValueError,'dated reference'):self.r.quarterly(row)
 def test_physical_tir_new_ids_and_years_and_scenarios(self):
  facts,retired=m.physical_facts(self.c,self.r,self.stage);self.assertEqual(len(facts),100);self.assertEqual(len(retired),100)
  records=[dict(zip(m.FIELDS+['observation_id'],p))for p in facts];self.assertEqual(len({p['observation_id']for p in records}),100);first=next(p for p in records if p['source_cell']=='D22');self.assertEqual(first['category_literal'],'2022');self.assertEqual(first['universal_category'],'2021');self.assertIsNone(first['row_number']);self.assertTrue(first['observation_id'].startswith('physical:'));self.assertEqual(first['value'],first['raw_value']*100)
  overlay=self.r.overlays['2025',18176];self.assertEqual(overlay['scenario_label'],'EV según proyección AIReF');self.assertNotIn('constant',overlay['scenario'])
 def test_orphans_have161_positive_points_without_assigning_one(self):
  retired,coverage=m.orphan_registry(self.c,self.r,self.stage);self.assertEqual(len(retired),24);self.assertEqual(len(coverage),161);self.assertTrue(all(p[3].startswith('published:')for p in retired));self.assertTrue(all(json.loads(p[4])['not_pointwise_replacement']for p in retired))
 def test_current_public_layer_and_unresolved_partition(self):
  with duckdb.connect(str(DB),read_only=True)as c:
   self.assertEqual(c.execute('SELECT count(*)FROM recovery2_public_observations').fetchone()[0],42833);self.assertEqual(c.execute("SELECT count(*)FROM recovery2_resolution WHERE state='blocked'").fetchone()[0],322);self.assertFalse(c.execute('SELECT book,row_number FROM recovery2_retired INTERSECT SELECT * FROM recovery2_usable_keys').fetchall());self.assertFalse(c.execute("SELECT * FROM recovery2_public_observations WHERE source_sheet='Gráfico 100'AND block<>'Gráfico 100'").fetchall());self.assertEqual(c.execute('SELECT count(*)FROM recovery2_numeric_queue').fetchone()[0],4481)
 def test_sql_upper_and_population_overlay_merge(self):
  with duckdb.connect(str(DB),read_only=True)as c:
   role,proof=c.execute("SELECT quantity_role,quantity_evidence FROM recovery2_public_observations WHERE book='2026'AND row_number=7607").fetchone();self.assertEqual(role,'depicted_range_upper_edge_not_point_forecast');self.assertEqual(json.loads(proof)['population_scope'],'resident_population_all_maternal_nationalities');self.assertIsNone(json.loads(proof).get('confidence_level'));self.assertFalse(json.loads(proof)['confidence_interpretation_allowed'])
 def test_query_current_observations_and_no_legacy_fallback(self):
  p=m.query_safe(DB,book='2026',row_number=7607);self.assertEqual(p['population_scope'],'resident_population_all_maternal_nationalities');self.assertEqual(p['quantity_role'],'depicted_range_upper_edge_not_point_forecast');self.assertIn('source_xlsx_sha256',p)
  physical=m.query_observation(DB,observation_id='physical:airef:2025:Gr49:D22:lifeconstant2024');self.assertEqual(physical['category_literal'],'2022');self.assertEqual(physical['universal_category'],'2021');self.assertIsNone(physical['row_number'])
  for book,n in[('2025',18077),('2025',16970),('2025',37525)]:
   with self.assertRaises(LookupError):m.query_safe(DB,book=book,row_number=n)
 def test_query_range_is_typed_not_point_or_ci(self):
  with duckdb.connect(str(DB),read_only=True)as c:key=c.execute('SELECT range_id FROM recovery2_ranges ORDER BY range_id LIMIT 1').fetchone()[0]
  p=m.query_range(DB,range_id=key);self.assertIsNone(p['confidence_level']);self.assertIsNone(p['method']);self.assertFalse(p['confidence_interpretation_allowed']);self.assertTrue(p['not_point_forecast']);self.assertLessEqual(D(p['lower_value']),D(p['central_value']));self.assertLessEqual(D(p['central_value']),D(p['upper_value']))
  with self.assertRaises(LookupError):m.query_range(DB,range_id='absent')

if __name__=='__main__':unittest.main()
