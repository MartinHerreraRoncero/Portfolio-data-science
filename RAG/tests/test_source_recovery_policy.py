import unittest,json,tempfile,shutil
from unittest.mock import patch
from decimal import Decimal as D
import duckdb
from experiments import source_recovery_policy as m
DB=m.ROOT/'evaluation/source_recovery_policy_2026-11-01/run_v3/source_recovery_policy.duckdb'

class SourcePolicyTests(unittest.TestCase):
 def test_exact16_user_decisions_and130_still_restricted(self):
  with duckdb.connect(str(DB),read_only=True)as c:
   expected={('2025',n)for n in[18761,18816,18872,18927,18981,18990,20840,20842,20844]}|{('2026',n)for n in range(5077,5084)};self.assertEqual(set(c.execute('SELECT book,row_number FROM policy_source_priority').fetchall()),expected);self.assertEqual(c.execute('SELECT count(*)FROM policy_restricted_variants').fetchone()[0],130);self.assertEqual(c.execute('SELECT count(*)FROM policy_usable_keys').fetchone()[0],42766);self.assertEqual(c.execute('SELECT count(*)FROM policy_public_observations').fetchone()[0],42880);self.assertEqual(c.execute("SELECT count(*)FROM policy_resolution WHERE state='admitted_user_authorized_source_priority'").fetchone()[0],16)
 def test_manifest_preserves_literal_user_authorization(self):
  p=m.read(m.PLAN);self.assertIn('haz que prevalezca el total publicado',p['user_instruction_quote']);self.assertIn('manten lo del excel tal cual viene',p['user_instruction_quote']);self.assertFalse(p['global_calculation_prohibition']);self.assertTrue(p['do_not_replace_published_values_to_force_sum'])
 def test_all16_values_units_series_periods_and_original_nulls_unchanged(self):
  with duckdb.connect(str(DB),read_only=True)as c:
   for book,n in c.execute('SELECT book,row_number FROM policy_source_priority').fetchall():
    p=m.query_safe(DB,book=book,row_number=n);q=c.execute('SELECT '+','.join(m.FIELDS)+' FROM recovery3_source_variants WHERE book=?AND row_number=?',[book,n]);original=dict(zip(m.FIELDS,q.fetchone()))
    for field in['value','raw_value','unit','metric_literal','series_literal','category_literal','source_sheet','source_cell','universal_category','universal_series','universal_value_sheet','universal_value_cell']:
     self.assertEqual(p[field],str(original[field])if isinstance(original[field],D)else original[field],field)
    self.assertEqual(p['admission_basis'],'user_authorized_source_priority');self.assertFalse(p['calculation_safe']);self.assertTrue(p['numeric_reconciliation_failed']);self.assertFalse(p['global_calculation_prohibition']);self.assertFalse(p['reconciliation_required_for_read']);self.assertFalse(json.loads(p['quantity_evidence'])['source_selector_required']);self.assertIn('desajuste',p['warning']);self.assertNotIn('do not calculate',p['quantity_evidence']);self.assertEqual(p['quantity_role'],'user_authorized_reported_source_quantity_unreconciled')
 def test_published_total_is_never_replaced_by_sum(self):
  for year,expected,components in[(2023,'6.1569957184953044',[18761,18816]),(2024,'5.508152087617769',[18872,18927])]:
   p=m.query_total(DB,book='2025',figure=54,year=year);self.assertEqual(D(p['value']),D(expected));self.assertEqual(p['series_literal'],'Total');self.assertNotEqual(D(p['value']),sum(D(m.query_safe(DB,book='2025',row_number=n)['value'])for n in components));self.assertEqual(p['source_priority'],'published_total_preferred_never_replace_by_sum')
  for book,figure,year in[('2025',64,2024),('2026',54,2024),('2025',54,'2023'),('2025',54,2025)]:
   with self.assertRaises(ValueError):m.query_total(DB,book=book,figure=figure,year=year)
 def test_gr64_reported_total_and_components_not_synthesized(self):
  values=[D(m.query_safe(DB,book='2025',row_number=n)['value'])for n in[20840,20842,20844]];self.assertEqual(values,[D('-20.757654378409629'),D('21.282368061555432'),D('-2.6670561451890083')]);self.assertNotEqual(values[0]+values[1],values[2])
 def test_workbook_gr30_preferred_without_allow_unreconciled_gate(self):
  p=m.query_safe(DB,book='2026',row_number=5078);self.assertEqual(D(p['value']),D('-5.2257679432943718'));self.assertEqual(p['source_priority'],'attached_workbook_preferred_over_published_pdf_labels');self.assertFalse(p['calculation_safe']);self.assertFalse(p['global_calculation_prohibition'])
 def test_pdf6_require_explicit_source_and_keep_one_decimal_precision(self):
  with duckdb.connect(str(DB),read_only=True)as c:
   ids=c.execute('SELECT observation_id,preferred_observation_id FROM policy_pdf_default_withdrawals').fetchall();self.assertEqual(len(ids),6)
  for ident,preferred in ids:
   with self.assertRaises(LookupError):m.query_observation(DB,observation_id=ident)
   with self.assertRaises(LookupError):m.query_observation(DB,observation_id=ident,source_preference='attached_workbook')
   p=m.query_observation(DB,observation_id=ident,source_preference='published_pdf');self.assertFalse(p['default_preferred']);self.assertEqual(p['preferred_observation_id'],preferred);self.assertEqual(json.loads(p['quantity_evidence'])['precision'],'0.1')
  with self.assertRaises(LookupError):m.query_observation(DB,observation_id='airef:2026:5078',source_preference='published_pdf')
 def test_no130_private_geometry_or_fiscal_quarantine_leak(self):
  with duckdb.connect(str(DB),read_only=True)as c:
   for table in['policy_restricted_variants','recovery3_private_inputs','recovery3_diagram_nodes','source_recovery_quarantine']:
    self.assertFalse(c.execute('SELECT * FROM policy_usable_keys INTERSECT SELECT book,row_number FROM '+table).fetchall())
   self.assertFalse(c.execute('SELECT observation_id FROM policy_public_observations INTERSECT SELECT observation_id FROM policy_pdf_default_withdrawals').fetchall())
  for book,n in[('2026',5811),('2026',5726),('2025',707),('2025',40835),('2025',37525)]:
   with self.assertRaises(LookupError):m.query_safe(DB,book=book,row_number=n)
 def test_changed_policy_or_base_hash_fails_closed(self):
  original=m.sha
  with duckdb.connect(str(DB),read_only=True)as c:
   for target in[m.PLAN,m.BASE]:
    with patch.object(m,'sha',side_effect=lambda p:'bad'if p==target else original(p)):
     with self.assertRaises(ValueError):m.guards(c)
 def test_positive_control_then_raw_and_view_mutations_rejected(self):
  with tempfile.TemporaryDirectory(dir=m.ROOT/'tmp',prefix='source-policy-mutation-')as directory:
   db=m.Path(directory)/'test.duckdb';shutil.copyfile(DB,db)
   with duckdb.connect(str(db))as c:
    self.assertEqual(m.check_new(c)['inherited_tables_preserved'],118);c.execute('BEGIN');c.execute("UPDATE policy_source_priority SET value=value+.1 WHERE book='2025'AND row_number=18981")
    with self.assertRaisesRegex(ValueError,'values or evidence changed'):m.check_new(c)
    c.execute('ROLLBACK');c.execute('UPDATE source_recovery_external_facts SET value=value+.1')
    with self.assertRaisesRegex(ValueError,'Inherited source table changed'):m.check_new(c)
    c.execute('UPDATE source_recovery_external_facts SET value=value-.1');expected=dict(m.views())['policy_public_observations'];c.execute("CREATE OR REPLACE VIEW policy_public_observations AS SELECT * FROM("+expected+")WHERE observation_id<>'airef:2026:5078'UNION ALL BY NAME SELECT f.*,'fault_injection' source_layer FROM recovery3_new_facts f WHERE observation_id='published-pdf:airef:2026:Gr30:2050:interest-spending:precision0.1'");self.assertEqual(c.execute('SELECT count(*)FROM policy_public_observations').fetchone()[0],42880)
    with self.assertRaisesRegex(ValueError,'Source policy view changed'):m.check_new(c)

if __name__=='__main__':unittest.main()
