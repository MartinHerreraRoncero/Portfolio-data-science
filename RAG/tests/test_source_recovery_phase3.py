import unittest,json,copy,tempfile,shutil,re
from unittest.mock import patch
from decimal import Decimal as D
import duckdb
from experiments import source_recovery_phase3 as m
DB=m.ROOT/'evaluation/source_recovery_phase3_2026-11-01/run_v1/source_recovery_phase3.duckdb'

class Recovery3Tests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.c=duckdb.connect(str(m.BASE),read_only=True);cls.r=m.Resolver();q=cls.c.execute('SELECT * FROM staging_records');names=[x[0]for x in q.description];cls.stage={(p['book'],p['row_number']):p for p in[dict(zip(names,x))for x in q.fetchall()]}
 @classmethod
 def tearDownClass(cls):cls.r.close();cls.c.close()
 def test_all322_pending_keys_and_146_source_policy_representations(self):
  selected=m.asset('selection.json');keys={(p['book'],p['row_number'])for p in selected};self.assertEqual(len(keys),322);self.assertEqual(keys,set(self.c.execute("SELECT book,row_number FROM recovery2_resolution WHERE state='blocked'").fetchall()))
  for p in selected:self.assertEqual(m.digest(self.stage[p['book'],p['row_number']]),p['payload_sha256'])
  with duckdb.connect(str(DB),read_only=True)as c:self.assertEqual(c.execute('SELECT count(*)FROM recovery3_source_variants').fetchone()[0],146);self.assertEqual(c.execute('SELECT count(*)FROM recovery3_policy_withdrawals').fetchone()[0],104)
 def test_primary_evidence_and_predecessor_certificate(self):m.guards(self.c)
 def test_changed_primary_manifest_rejected(self):
  original=m.sha
  with patch.object(m,'sha',side_effect=lambda p:'changed'if p==m.ASSETS/'manifest.json'else original(p)):
   with self.assertRaisesRegex(ValueError,'manifest changed'):m.guards(self.c)
 def test_all122_admitted_source_cache_and_series_units(self):
  counts={}
  for a in m.asset('selection.json'):
   f=a['family']
   if f not in['excess','excise','smi']:continue
   row=self.stage[a['book'],a['row_number']];result=dict(zip(m.FIELDS,self.r.admitted(row,f,self.r.proposals[a['proposal']])));self.assertEqual(result['raw_value'],D(row['decimal_XML_texto']));self.assertEqual(result['universal_series'],row['serie_literal']);self.assertEqual(result['universal_category'],row['periodo_categoria_literal']);counts[f]=counts.get(f,0)+1
   if f=='excess':
    proof=json.loads(result['quantity_evidence']);self.assertIsNone(proof['formula']);self.assertIsNone(proof['residual_orientation']);self.assertFalse(proof['independent_causal_identification_allowed']);self.assertIsNone(result['universal_value_sheet']);self.assertEqual(result['source_sheet'],'Gráfico RE_2.3.')
   elif f=='excise':self.assertIn('excise',result['unit']);self.assertTrue(json.loads(result['quantity_evidence'])['wrong_original_IVA_scope_preserved'])
   else:self.assertEqual(result['value'],result['raw_value']*100);self.assertNotEqual(result['unit'],'percent_affected_workers')
  self.assertEqual(counts,dict(excess=90,excise=26,smi=6))
 def test_unnamed_excise_weight_remains_null(self):
  p=next(a for a in m.asset('selection.json')if a['family']=='excise'and self.stage[a['book'],a['row_number']]['serie_literal']is None);row=self.stage[p['book'],p['row_number']];result=dict(zip(m.FIELDS,self.r.admitted(row,'excise',self.r.proposals[p['proposal']])));self.assertIsNone(result['series_literal']);self.assertIsNone(result['universal_series']);self.assertEqual(result['unit'],'percent_total_excise_revenue_by_age')
 def test_changed_monthly_excess_header_rejected(self):
  a=next(p for p in m.asset('selection.json')if p['family']=='excess');row=self.stage[a['book'],a['row_number']];s=self.r.sources['2026'];values=s.sheet('Gráfico RE_2.3.')[1];old=values['C6'];values['C6']='level'
  try:
   with self.assertRaisesRegex(ValueError,'header changed'):self.r.admitted(row,'excess',self.r.proposals[a['proposal']])
  finally:values['C6']=old
 def test_alias48_and_wrong_registry62_have_verified_replacements(self):
  aliases,retired,coverage=m.routing(self.c,self.r,self.stage);self.assertEqual(len(aliases),48);self.assertEqual(len(retired),110);self.assertEqual(len(coverage),240)
  for a in aliases:self.assertTrue(json.loads(a[3])['not_value_only_deduplication'])
  for a in retired[48:]:self.assertTrue(json.loads(a[4])['not_pointwise_mapping'])
 def test_retirement13_new_id_periods_and_pdf6_precision(self):
  facts,retired=m.new_facts(self.c,self.r,self.stage);records=[dict(zip(m.FIELDS+['observation_id'],p))for p in facts];self.assertEqual(len(retired),13);self.assertEqual(len(records),19);self.assertEqual(len({p['observation_id']for p in records}),19)
  early=[p for p in records if p['book']=='2025'];self.assertEqual({p['category_literal']for p in early if '-01'in p['category_literal']},{'2024-01','2025-01'});self.assertTrue(all(p['series_literal']=='Jubilación anticipada'and p['universal_series']=='Demorada'and p['row_number']is None for p in early));self.assertIn('2024',{p['category_literal']for p in early})
  pdf=[p for p in records if p['book']=='2026'];self.assertEqual(len(pdf),6);self.assertNotIn('Mejor punto de partida 2025',{p['series_literal']for p in pdf});self.assertTrue(all(json.loads(p['quantity_evidence'])['precision']=='0.1'for p in pdf));self.assertEqual({p['value']for p in pdf},{D('-4.8'),D('.9'),D('-4.5'),D('1.8'),D('-5.5')})
 def test_internal_dependencies28_and_qualitative_nodes7(self):
  internal,nodes=m.private_and_diagram(self.r,self.stage);self.assertEqual(len(internal),28);self.assertEqual(len(nodes),7);self.assertTrue(all(json.loads(p[-1])['actual_rate_kind']is None for p in internal));self.assertTrue(all(p[5]==40 and not json.loads(p[-1])['quant_risk_interpretation_allowed']for p in nodes))
 def test_default_counts_partition_and_all_restricted_scopes_absent(self):
  with duckdb.connect(str(DB),read_only=True)as c:
   self.assertEqual(c.execute('SELECT count(*)FROM recovery3_public_observations').fetchone()[0],42870);self.assertEqual(c.execute('SELECT count(*)FROM recovery3_usable_keys').fetchone()[0],42750);self.assertEqual(c.execute('SELECT count(*)FROM recovery3_outside_default').fetchone()[0],4463)
   for name in['recovery3_source_variants','recovery3_private_inputs','recovery3_diagram_nodes','recovery3_retired']:
    self.assertFalse(c.execute('SELECT * FROM recovery3_usable_keys INTERSECT SELECT book,row_number FROM '+name).fetchall())
   self.assertFalse(c.execute('SELECT observation_id FROM recovery3_public_observations GROUP BY observation_id HAVING count(*)<>1').fetchall());self.assertEqual(c.execute('SELECT count(*)FROM recovery3_source_variants WHERE NOT calculation_safe').fetchone()[0],16)
 def test_source_selection_and_incoherence_policy_fail_closed(self):
  with self.assertRaises(ValueError):m.query_as_reported(DB,book='2025',row_number=20840,source_preference=None,allow_unreconciled=True)
  with self.assertRaises(LookupError):m.query_as_reported(DB,book='2025',row_number=20840,source_preference='attached_workbook',allow_unreconciled=False)
  p=m.query_as_reported(DB,book='2025',row_number=20840,source_preference='attached_workbook',allow_unreconciled=True);self.assertFalse(p['calculation_safe']);self.assertTrue(p['source_variant_proof']['numeric_reconciliation_failed']);self.assertEqual(D(p['value']),D(p['raw_value'])*100);self.assertIn('unresolved',p['warning']);self.assertIn('source_xlsx_sha256',p)
  p=m.query_as_reported(DB,book='2026',row_number=5811,source_preference='attached_workbook',allow_unreconciled=False);self.assertEqual(p['category_literal'],'2050');self.assertTrue(p['calculation_safe']);self.assertEqual(p['calculation_scope'],'within_selected_workbook_only');self.assertEqual(D(p['value']),D('-4.4034706351058936'));self.assertFalse(p['source_variant_proof']['pdf_comparison_confirmed'])
  for book,n in[('2026',5811),('2026',5726),('2025',20840),('2025',18761),('2026',5078),('2025',18313),('2025',40835),('2025',707)]:
   with self.assertRaises(LookupError):m.query_safe(DB,book=book,row_number=n)
 def test_specialized_private_and_diagram_queries(self):
  p=m.query_internal_input(DB,book='2025',row_number=707);self.assertTrue(p['internal_only']);self.assertIsNone(p['actual_rate_kind']);self.assertFalse(p['unqualified_economic_forecast_allowed'])
  d=m.query_diagram_node(DB,book='2025',row_number=40835);self.assertFalse(d['quant_risk_interpretation_allowed']);self.assertEqual(D(d['bubble_size']),40);self.assertFalse(d['bubble_cardinality_interpretation_allowed'])
 def test_monthly_last102_not_150_and_excise_context_overlay(self):
  p=m.query_safe(DB,book='2026',row_number=1830);self.assertEqual(p['category_literal'],'2007-06');self.assertEqual(D(p['value']),D('102.54706197048399'));self.assertIsNone(json.loads(p['quantity_evidence'])['formula'])
  p=m.query_safe(DB,book='2025',row_number=17335);self.assertIn('excise',p['metric_literal']);self.assertTrue(p['wrong_original_IVA_scope_preserved']);self.assertNotIn('IVA',p['metric_literal'])
 def test_positive_control_then_numeric_and_same_count_public_view_mutation_rejected(self):
  with tempfile.TemporaryDirectory(dir=m.ROOT/'tmp',prefix='recovery3-mutation-')as directory:
   altered=m.Path(directory)/'mutation.duckdb';shutil.copyfile(DB,altered)
   with duckdb.connect(str(altered))as c:
    self.assertEqual(m.check_new(c)['counts']['recovery3_public_observations'],42870)
    ident=c.execute('SELECT observation_id FROM recovery3_new_facts ORDER BY observation_id LIMIT 1').fetchone()[0];c.execute('BEGIN');c.execute('UPDATE recovery3_new_facts SET value=value+.1 WHERE observation_id=?',[ident])
    with self.assertRaisesRegex(ValueError,'reconstruction differs: recovery3_new_facts'):m.check_new(c)
    c.execute('ROLLBACK');remove=c.execute('SELECT observation_id FROM recovery3_public_observations WHERE row_number IS NOT NULL ORDER BY observation_id LIMIT 1').fetchone()[0];self.assertRegex(remove,r'^airef:202[56]:[0-9]+$');sql=dict(m.views())['recovery3_public_observations'];leak="SELECT "+','.join(m.FIELDS)+",'fault_injection' source_layer,'airef:'||book||':'||row_number observation_id FROM recovery3_source_variants WHERE book='2025'AND row_number=18761";c.execute("CREATE OR REPLACE VIEW recovery3_public_observations AS SELECT * FROM("+sql+")WHERE observation_id<>'"+remove+"' UNION ALL BY NAME "+leak);self.assertEqual(c.execute('SELECT count(*)FROM recovery3_public_observations').fetchone()[0],42870)
    with self.assertRaisesRegex(ValueError,'Phase3 view changed: recovery3_public_observations'):m.check_new(c)

if __name__=='__main__':unittest.main()
