import unittest,json,copy,tempfile
from pathlib import Path
from unittest.mock import patch
import duckdb
from experiments import closure_curation_registry as m
from experiments import curation_registry as core
RULES=m.ROOT/'evaluation/closure_family_semantics_2026-10-31/run_v3/curation_rules.json'
class TestClosureRegistry(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.cfg=core.load_rules(RULES);cls.snapshot=core.baseline(cls.cfg)
 def test_certified76_upgrade_only_and_strict345_preserved(self):
  before,oldgroups=core.classify(self.cfg,self.snapshot);after,groups=m.classify(self.cfg,self.snapshot);self.assertEqual(oldgroups,groups);old={(r[0],r[1]):r for r in before};new={(r[0],r[1]):r for r in after};changed={key for key in old if old[key]!=new[key]};self.assertEqual(changed,m.TECHNICAL_KEYS);self.assertEqual(len(changed),76)
  for key in changed:
   self.assertEqual(old[key][3],'bloqueado');self.assertEqual(new[key][3],'excluido');self.assertEqual(old[key][4],new[key][4]);self.assertEqual(old[key][11],new[key][11]);self.assertEqual(json.loads(new[key][10])['closure_decision']['disposition'],'excluded')
  self.assertEqual(sum(r[3]=='excluido'for r in after),421);self.assertEqual(sum(r[3]=='bloqueado'for r in after),4283);self.assertEqual(sum(r[3]=='excluido'for r in before),345);self.assertEqual(sum(json.loads(r[10]).get('kind')=='verified_semantic_duplicate'for r in after),156)
 def test_same_schema_build_validate_exports_and_no_new_admission(self):
  with tempfile.TemporaryDirectory()as td:
   output=Path(td)/'registry';b=m.build(RULES,output);v=m.validate(output/'registry.duckdb',RULES);self.assertEqual(b['states'],dict(bloqueado=4283,excluido=421));self.assertEqual(v['status'],'pass');self.assertEqual(v['new_admissions'],0);self.assertFalse(v['numeric_base_revalidation_executed']);self.assertEqual(v['admitted_unchanged'],42509);self.assertEqual(v['classified'],4704)
   with duckdb.connect(str(output/'registry.duckdb'),read_only=True)as c:self.assertEqual(c.execute('SELECT count(*)FROM ledger').fetchone()[0],4704);self.assertEqual(len(c.execute('DESCRIBE ledger').fetchall()),12)
   (output/'group_coverage.csv').write_text('tampered',encoding='utf8')
   with self.assertRaises(ValueError):m.validate(output/'registry.duckdb',RULES)
 def test_frozen_certificate_or_original_source_changed_rejected(self):
  m.classify(self.cfg,self.snapshot);read=Path.read_text
  def badcert(path,*args,**kwargs):
   text=read(path,*args,**kwargs)
   if path==m.CERT:
    x=json.loads(text);x['coverage_complete']=False;return json.dumps(x)
   return text
  with patch.object(Path,'read_text',badcert):
   with self.assertRaises(ValueError):m.classify(self.cfg,self.snapshot)
  sha=core.sha
  with duckdb.connect(str(m.DB),read_only=True)as c:source=core.local(c.execute('SELECT xlsx_path FROM sources LIMIT 1').fetchone()[0])
  with patch.object(core,'sha',side_effect=lambda p:'wrong'if Path(p)==source else sha(p)):
   with self.assertRaises(ValueError):m.classify(self.cfg,self.snapshot)
 def test_pending_payload_and_explicit_rule_proof_required(self):
  m.classify(self.cfg,self.snapshot);snap=copy.deepcopy(self.snapshot);key=next(iter(m.TECHNICAL_KEYS));row=next(r for r in snap['pending']if(r['book'],r['row_number'])==key);row['decimal_XML_texto']='999'
  with self.assertRaises(ValueError):m.classify(self.cfg,snap)
  cfg=copy.deepcopy(self.cfg);rule=next(r for r in cfg['rules']if r['id']=='closure_final_20');rule['evidence']['source_files_frozen']=False
  with self.assertRaises(ValueError):m.classify(cfg,self.snapshot)
 def test_orphan24_and_keys_outside_catalog_never_promoted(self):
  after,_=m.classify(self.cfg,self.snapshot);bykey={(r[0],r[1]):r for r in after};inv=json.loads(m.INVENTORY.read_text(encoding='utf8'));orphans=[a for a in inv['records']if a['book']=='2025'and a['block']in['Gráfico 38','Gráfico 39','Gráfico 46','Gráfico 69']and a['source_sheet']is None and a['original_raw_decimal']=='1'];self.assertEqual(len(orphans),24)
  for a in orphans:self.assertEqual(bykey[a['book'],a['row_number']][3],'bloqueado')
  cfg=copy.deepcopy(self.cfg);a=orphans[0];template=copy.deepcopy(next(r for r in cfg['rules']if r['id']=='closure_final_20'));template.update(id='hostile_orphan_override',when=dict(book=dict(eq=a['book']),row_number=dict(eq=a['row_number'])),precedence=99999);cfg['rules'].append(template)
  with self.assertRaises(ValueError):m.classify(cfg,self.snapshot)
if __name__=='__main__':unittest.main()
