"""Declarative classification, frozen baseline and zero-admission invariants."""
import unittest,tempfile,json,copy,shutil,csv,subprocess,sys
from pathlib import Path
import duckdb
from experiments.curation_registry import ROOT,DEFAULT_RULES,load_rules,baseline,classify,plan_rows,build,validate,sha,dump,group_key,group_kind
class TestCurationRegistry(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.cfg=load_rules(DEFAULT_RULES);cls.snapshot=baseline(cls.cfg);cls.rows,cls.groups=classify(cls.cfg,cls.snapshot)
 def test_partition_counts_and_states(self):
  self.assertEqual(len(self.rows),8827);self.assertEqual(len(self.snapshot['admitted']),38386)
  self.assertEqual({s:sum(r[3]==s for r in self.rows) for s in ['recuperable','bloqueado','excluido']},{'recuperable':306,'bloqueado':8176,'excluido':345})
  self.assertEqual(len({(r[0],r[1]) for r in self.rows}),8827);self.assertFalse({(r[0],r[1]) for r in self.rows}&self.snapshot['admitted'])
  self.assertEqual(len({r[2] for r in self.rows}),332);self.assertEqual(len({(r['book'],r['bloque']) for r in self.snapshot['pending']}),128)
  self.assertEqual(sum(g[6] for g in self.groups),47213);self.assertEqual(sum(g[7] for g in self.groups),38386)
 def test_default_and_technical_exclusion_fail_closed(self):
  cfg=copy.deepcopy(self.cfg);cfg['rules']=[];rows,_=classify(cfg,self.snapshot)
  self.assertTrue(all(r[3]=='bloqueado' and r[8]=='provisional' for r in rows))
  cfg['rules']=[dict(id='unsafe_technical',when={},status='excluido',action='test',priority=1,confidence='confirmed',evidence={'kind':'technical_plot_role'},phase='excluded',precedence=1)]
  rows,_=classify(cfg,self.snapshot);self.assertEqual(sum(r[3]=='excluido' for r in rows),189)
 def test_duplicate_proof_requires_semantics_and_canonical_key(self):
  dup=[r for r in self.rows if r[5]=='resource_duplicate_alias'];self.assertEqual(len(dup),156)
  self.assertTrue(all(r[3]=='excluido' and json.loads(r[10])['canonical_role'] in ['published_series_level','plot_reference_level'] for r in dup))
  snapshot=copy.deepcopy(self.snapshot);target=next(r for r in snapshot['pending'] if (r['book'],r['row_number'])==(dup[0][0],dup[0][1]));target['unidad']='points_of_other_metric'
  rows,_=classify(self.cfg,snapshot);actual=next(r for r in rows if r[:2]==dup[0][:2]);self.assertEqual(actual[3],'bloqueado')
  cfg=copy.deepcopy(self.cfg);cfg['duplicate_pairs'][0]['canonical_row_number']=-1
  rows,_=classify(cfg,self.snapshot);self.assertEqual(sum(r[3]=='excluido' for r in rows),344)
 def test_rule_operators_and_ambiguity_rejected(self):
  for mutation in ['sql','badfield','badstate','badphase','unsafeexcluded']:
   cfg=copy.deepcopy(self.cfg);r=cfg['rules'][0]
   if mutation=='sql':r['when']={'book':{'sql':'DROP TABLE x'}}
   elif mutation=='badfield':r['when']={'arbitrary_expression':{'eq':'x'}}
   elif mutation=='badstate':r['status']='admitted'
   elif mutation=='badphase':r['phase']='ready'
   else:r['status']='excluido';r['evidence']={'kind':'unreviewed_default'}
   with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as d:
    path=Path(d)/'rules.json';path.write_text(json.dumps(cfg),encoding='utf-8')
    with self.assertRaises(ValueError):load_rules(path)
  cfg=copy.deepcopy(self.cfg);rule=copy.deepcopy(cfg['rules'][0]);rule['id']='ambiguous';cfg['rules'].append(rule)
  with self.assertRaisesRegex(ValueError,'Ambiguous'):classify(cfg,self.snapshot)
 def test_hash_certificate_and_extractor_are_required(self):
  cfg=copy.deepcopy(self.cfg);cfg['baseline']['db_sha256']='0'*64
  with self.assertRaisesRegex(ValueError,'baseline hash'):baseline(cfg)
  cfg=copy.deepcopy(self.cfg);cfg['baseline']['database']='../outside.duckdb'
  with self.assertRaises(ValueError):baseline(cfg)
 def test_certificate_counts_and_plan_registry_required(self):
  with tempfile.TemporaryDirectory(dir=ROOT/'evaluation') as d:
   cert=json.loads((ROOT/self.cfg['baseline']['validation']).read_text(encoding='utf-8'));cert['admitted']-=1
   p=Path(d)/'certificate.json';p.write_text(json.dumps(cert),encoding='utf-8');cfg=copy.deepcopy(self.cfg);cfg['baseline']['validation']=str(p.relative_to(ROOT))
   with self.assertRaisesRegex(ValueError,'Baseline counts changed'):baseline(cfg)
   missing=Path(d)/'missing.duckdb'
   result=subprocess.run([sys.executable,'-m','experiments.curation_registry','plan','--registry',str(missing)],cwd=ROOT,capture_output=True,text=True)
   self.assertNotEqual(result.returncode,0);self.assertNotIn('"status": "plan"',result.stdout)
 def test_group_identity_and_plan_are_not_fact_counts(self):
  row=self.snapshot['pending'][0];self.assertNotEqual(group_key(row),group_key(dict(row,parte_grafico_imagen='different')))
  self.assertNotEqual(group_key(dict(row,serie_id=None)),group_key(dict(row,serie_id='')))
  kinds={g[5] for g in self.groups};self.assertTrue('chart_series' in kinds and 'tabular_source_group' in kinds)
  plan=plan_rows(self.rows,self.groups);self.assertEqual(sum(r[4] for r in plan),306)
  self.assertEqual(sum(r[4] for r in plan if r[3]=='ready'),119);self.assertEqual(sum(r[4] for r in plan if r[3]=='semantic_contract'),187)
 def test_complete_ledger_replay_and_tampering(self):
  with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as d:
   out=Path(d)/'new';report=build(DEFAULT_RULES,out);self.assertEqual(report['new_admissions'],0)
   result=validate(out/'registry.duckdb',DEFAULT_RULES);self.assertEqual(result['status'],'pass');self.assertFalse(result['numeric_base_revalidation_executed'])
   with duckdb.connect(str(out/'registry.duckdb')) as c:c.execute("UPDATE ledger SET status='recuperable' WHERE rule_id='default'")
   manifest=json.loads((out/'manifest.json').read_text());manifest['registry_sha256']=sha(out/'registry.duckdb');(out/'manifest.json').write_text(json.dumps(manifest),encoding='utf-8')
   with self.assertRaisesRegex(ValueError,'Ledger classifications differ'):validate(out/'registry.duckdb',DEFAULT_RULES)
 def test_plan_and_config_hash_tampering_detected(self):
  with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as d:
   out=Path(d)/'new';build(DEFAULT_RULES,out)
   with (out/'batch_plan.csv').open('a',encoding='utf-8') as f:f.write('bad,line\n')
   with self.assertRaisesRegex(ValueError,'Export differs'):validate(out/'registry.duckdb',DEFAULT_RULES)
   cfg=copy.deepcopy(self.cfg);cfg['default']['priority']=9;p=Path(d)/'rules.json';p.write_text(json.dumps(cfg),encoding='utf-8')
   with self.assertRaisesRegex(ValueError,'configuration/code'):validate(out/'registry.duckdb',p)
 def test_coverage_exports_tampering_detected(self):
  for export in ['group_coverage.csv','block_coverage.csv']:
   with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as d:
    out=Path(d)/'new';build(DEFAULT_RULES,out)
    with (out/export).open('a',encoding='utf-8') as f:f.write('bad,row\n')
    with self.assertRaisesRegex(ValueError,'Export differs'):validate(out/'registry.duckdb',DEFAULT_RULES)

if __name__=='__main__':unittest.main()
