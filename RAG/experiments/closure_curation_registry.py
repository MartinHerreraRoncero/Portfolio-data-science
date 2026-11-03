"""Certified closure evidence adapter for the separate curation ledger, never numeric admission."""
import json,types
from pathlib import Path
import duckdb
from experiments import curation_registry as core
ROOT=core.ROOT
DB=ROOT/'evaluation/closure_family_semantics_2026-10-31/run_v3/closure_families.duckdb'
CERT=DB.with_name('validation.json')
EXTRACTOR=ROOT/'experiments/closure_family_semantics.py'
INVENTORY=ROOT/'evaluation/closure_family_semantics_2026-10-31/research_bindings/closure_inventory4849.json'
DB_SHA='a7f4b526affb63a7b89275623937ebfdde52985920d39ee6e89ac4c658385627'
EXTRACTOR_SHA='f107c9fe22c5e4d702664764a9a2382a55af0eab790f95ed223d6b5368c62323'
INVENTORY_SHA='f0f8a0319be301ba911fcc349b0bc22ae00b0694ad5e2598049d3f4da502a57e'
CORE_SHA='b5505531433e1406c495da092872eb8bd5873398ab298e7539f5250a4f627d05'
TECHNICAL_KEYS=frozenset([('2025', 15845), ('2025', 15846), ('2025', 15847), ('2025', 15848), ('2025', 15849), ('2025', 15850), ('2025', 15851), ('2025', 15852), ('2025', 15853), ('2025', 15854), ('2025', 15855), ('2025', 15856), ('2025', 15857), ('2025', 15858), ('2025', 15859), ('2025', 15860), ('2025', 15861), ('2025', 15862), ('2025', 15863), ('2025', 15864), ('2025', 15865), ('2025', 15866), ('2025', 15867), ('2025', 15868), ('2025', 15869), ('2025', 15870), ('2025', 15871), ('2025', 15872), ('2025', 15873), ('2025', 15874), ('2025', 15875), ('2025', 15907), ('2025', 15908), ('2025', 15909), ('2025', 15910), ('2025', 15911), ('2025', 15912), ('2025', 15913), ('2025', 15914), ('2025', 15915), ('2025', 15916), ('2025', 15917), ('2025', 15918), ('2025', 15919), ('2025', 15920), ('2025', 15921), ('2025', 15922), ('2025', 15923), ('2025', 15924), ('2025', 15925), ('2025', 15926), ('2025', 15927), ('2025', 15928), ('2025', 15929), ('2025', 15930), ('2025', 15931), ('2025', 15932), ('2025', 15933), ('2025', 15934), ('2025', 15935), ('2025', 15936), ('2025', 15937), ('2026', 1834), ('2026', 1835), ('2026', 1836), ('2026', 3437), ('2026', 3440), ('2026', 3442), ('2026', 3443), ('2026', 3446), ('2026', 3447), ('2026', 3449), ('2026', 3450), ('2026', 3451), ('2026', 3453), ('2026', 3454)])
def certified_proof(cfg):
 core.require(core.sha(Path(core.__file__))==CORE_SHA,'Frozen core registry code changed')
 core.require(core.local(cfg['baseline']['database'])==DB and core.local(cfg['baseline']['validation'])==CERT and core.local(cfg['baseline']['extractor'])==EXTRACTOR,'Adapter only accepts the certified closure baseline')
 core.require(core.sha(DB)==DB_SHA==cfg['baseline']['db_sha256'],'Certified closure database changed');core.require(core.sha(EXTRACTOR)==EXTRACTOR_SHA,'Certified closure extractor changed');core.require(core.sha(INVENTORY)==INVENTORY_SHA,'Certified closure inventory changed');cert=json.loads(CERT.read_text(encoding='utf8'));core.require(cert['status']==cert['previous_validation']=='pass'and cert['db_sha256']==DB_SHA and cert['code_sha256']==EXTRACTOR_SHA and cert['inventory_sha256']==INVENTORY_SHA and cert['coverage_complete']and cert['review_domain']==4849 and cert['admitted']==42509 and cert['numeric_pending']==4704,'Closure certificate or coverage invalid')
 with duckdb.connect(str(DB),read_only=True)as c:
  for path,expected in c.execute('SELECT xlsx_path,xlsx_sha256 FROM sources UNION ALL SELECT pdf_path,pdf_sha256 FROM sources').fetchall():core.require(core.sha(core.local(path))==expected,'Certified original source changed')
  q=c.execute('SELECT book,row_number,disposition,coverage_complete,decision_json FROM closure_family_decisions');decisions={(b,n):(d,cov,json.loads(j))for b,n,d,cov,j in q.fetchall()};q=c.execute('SELECT * FROM closure_family_numeric_queue');ns=[a[0]for a in q.description];pending={}
  for v in q.fetchall():
   a=dict(zip(ns,v));a['classification_reason']=a[cfg['baseline']['reason_field']];pending[a['book'],a['row_number']]=a
 core.require(len(decisions)==4849 and all(d[1]for d in decisions.values()),'Closure decision coverage invalid')
 for key in TECHNICAL_KEYS:core.require(key in pending and key in decisions and decisions[key][0]=='excluded'and decisions[key][2]['disposition']=='excluded','Certified technical exclusion missing')
 return decisions,pending

def classify(cfg,snapshot):
 ledger,groups=core.classify(cfg,snapshot);decisions,pending=certified_proof(cfg);out=[];replaced=set()
 for old,row in zip(ledger,snapshot['pending']):
  key=row['book'],row['row_number'];rules=[r for r in cfg['rules']if core.matches(r,row)]
  if rules:
   precedence=max(r['precedence']for r in rules);rules=[r for r in rules if r['precedence']==precedence];core.require(len(rules)==1,'Ambiguous closure rule');rule=rules[0]
  else:rule=dict(cfg['default'],id='default')
  requested=rule['status']=='excluido'and rule['evidence'].get('kind')=='technical_plot_role'
  if requested and row['classification_reason']not in['forecast_marker_not_measure','drawing_helper_not_economic_series']:
   core.require(key in TECHNICAL_KEYS,'Technical exclusion key outside certified catalog')
  if key not in TECHNICAL_KEYS:out.append(old);continue
  ev=rule['evidence'];core.require(requested and ev.get('decision_relation')=='closure_family_decisions'and ev.get('review_complete')is True and ev.get('source_files_frozen')is True,'Explicit certified technical rule required')
  core.require(core.digest(row)==core.digest(pending[key]),'Certified pending payload changed');decision=decisions[key][2];core.require(decision['coverage_complete']and decision['disposition']=='excluded'and decision['book']==key[0]and decision['row_number']==key[1],'Certified per-key technical decision invalid');core.require(old[3]=='bloqueado'and old[6]=='verify_technical_plot_role','Only legacy technical-proof downgrade may be adapted')
  evidence=dict(ev,base_database_sha256=DB_SHA,rule_id=rule['id'],classification_not_admission=True,proof_adapter='closure_curation_registry',closure_decision=decision,closure_code_sha256=EXTRACTOR_SHA,closure_inventory_sha256=INVENTORY_SHA,source_payload_sha256=core.digest(row));out.append((old[0],old[1],old[2],'excluido',old[4],rule['id'],rule['action'],rule['priority'],rule['confidence'],'excluded',core.dump(evidence),old[11]));replaced.add(key)
 core.require(replaced==TECHNICAL_KEYS,'Certified technical exclusion coverage incomplete');return out,groups

def _run(function,*args):
 # A private globals copy reuses the frozen core implementation without patching it.
 namespace=dict(function.__globals__);namespace.update(classify=classify,__file__=__file__);copy=types.FunctionType(function.__code__,namespace,function.__name__,function.__defaults__,function.__closure__);return copy(*args)
def build(rules,output):return _run(core.build,Path(rules),Path(output))
def validate(registry,rules):return _run(core.validate,Path(registry),Path(rules))
