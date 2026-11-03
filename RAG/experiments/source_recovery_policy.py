"""User-authorized source priority; reported figures stay unreconciled and unchanged."""
import argparse,json,shutil
from pathlib import Path
from decimal import Decimal as D
import duckdb
from experiments import source_recovery_phase3 as prior
from experiments.source_recovery import ROOT,FIELDS,require,read,dumps,digest,aid,sha
BASE=ROOT/'evaluation/source_recovery_phase3_2026-11-01/run_v1/source_recovery_phase3.duckdb'
BASE_SHA='f9999745f6c131c1a18f8ea60a2bbc00a3932413b43810754cc7c4a9f9ce73fa'
BASE_CODE_SHA='312e006c4d30af16842faf478f486fb2f64109ea3d9198de6b64ffdfc3742135'
PLAN=ROOT/'evaluation/source_recovery_policy_2026-11-01/assets/policy.json'
PLAN_SHA='306e5c48c56b822dfa3f92e7358d1750813858f0a8a0e55c2d856374f39aba38'
EXACT_KEYS={('2025',n)for n in[18761,18816,18872,18927,18981,18990,20840,20842,20844]}|{('2026',n)for n in range(5077,5084)}

def guards(c):
 require(sha(BASE)==BASE_SHA and sha(ROOT/'experiments/source_recovery_phase3.py')==BASE_CODE_SHA,'Certified base snapshot/code changed');require(sha(PLAN)==PLAN_SHA,'Authorized user policy changed');p=read(PLAN);require(sha(BASE.parent/'validation.json')==p['base_certificate_sha256'],'Base numeric certificate changed');cert=read(BASE.parent/'validation.json');require(cert['status']=='pass'and cert['db_sha256']==BASE_SHA and cert['code_sha256']==BASE_CODE_SHA,'Base not certified');require(sha(prior.ASSETS/'manifest.json')==p['base_proof_manifest_sha256'],'Base source evidence manifest changed');prior.guards(c);return p

def collect(c):
 p=guards(c);require({(a['book'],a['row_number'])for a in p['selected']}==EXACT_KEYS,'User policy exact16 domain changed');q=c.execute('SELECT * FROM recovery3_source_variants WHERE NOT calculation_safe');names=[n[0]for n in q.description];records={(a['book'],a['row_number']):a for a in[dict(zip(names,x))for x in q.fetchall()]};require(set(records)==EXACT_KEYS,'Unreconciled source selection changed');selected=[]
 for a in p['selected']:
  row=records[a['book'],a['row_number']];require(digest(row)==a['source_payload_sha256']and row['variant_family']==a['family'],'Priority source record changed');old=json.loads(row['variant_proof']);evidence={k:v for k,v in old.items()if k not in['warning','source_preference_required','source_selector_required']};evidence.update(policy_id=p['policy_id'],admission_basis='user_authorized_source_priority',source_priority=p['source_priority'][a['family']],numeric_reconciliation_failed=True,calculation_safe=False,global_calculation_prohibition=False,reconciliation_is_not_precondition_for_source_read=True,source_selector_required=False,prior_source_selector_required=old.get('source_selector_required'),do_not_replace_published_values_to_force_sum=True,prior_restriction_is_not_current_policy=True,warning='Cifras reportadas con desajuste aritmético conservado por prioridad de fuente autorizada; no sustituir valores para forzar la suma.');row.update(quantity_role='user_authorized_reported_source_quantity_unreconciled',quantity_evidence=dumps(evidence));selected.append(tuple(row[n]for n in FIELDS)+(aid(a['book'],a['row_number']),a['family'],False,True,dumps(evidence)))
 withdrawals=[]
 for a in p['pdf_withdrawals']:
  q=c.execute('SELECT * FROM recovery3_new_facts WHERE observation_id=?',[a['observation_id']]);names=[n[0]for n in q.description];v=q.fetchone();require(v is not None and digest(dict(zip(names,v)))==a['source_payload_sha256'],'Published PDF reference changed');require(a['preferred_observation_id']in{r[len(FIELDS)]for r in selected},'PDF preferred workbook reference is not admitted');withdrawals.append((a['observation_id'],a['preferred_observation_id'],'published_pdf_reference_retained_explicit_source_only'))
 return selected,withdrawals

def views():
 return[
 ('policy_usable_keys','SELECT * FROM recovery3_usable_keys UNION ALL SELECT book,row_number FROM policy_source_priority'),
 ('policy_public_observations',"SELECT p.* FROM recovery3_public_observations p WHERE NOT EXISTS(SELECT 1FROM policy_pdf_default_withdrawals w WHERE w.observation_id=p.observation_id) UNION ALL BY NAME SELECT o.*,'user_authorized_source_priority' source_layer FROM policy_source_priority o"),
 ('policy_restricted_variants','SELECT v.* FROM recovery3_source_variants v WHERE NOT EXISTS(SELECT 1FROM policy_source_priority p WHERE p.book=v.book AND p.row_number=v.row_number)'),
 ('policy_resolution',"SELECT r.* REPLACE(CASE WHEN p.book IS NOT NULL THEN 'admitted_user_authorized_source_priority'ELSE r.state END AS state)FROM recovery3_resolution r LEFT JOIN policy_source_priority p USING(book,row_number)")]

def preserve(c):
 name='__policy_certified_base';c.execute("ATTACH '"+str(BASE).replace("'","''")+"' AS "+name+' (READ_ONLY)')
 try:
  tables=c.execute("SELECT table_name FROM information_schema.tables WHERE table_catalog=?AND table_schema='main'AND table_type='BASE TABLE'",[name]).fetchall()
  for(n,)in tables:
   safe='"'+n.replace('"','""')+'"';require(not c.execute('SELECT * FROM main.'+safe+' EXCEPT ALL SELECT * FROM '+name+'.main.'+safe).fetchall()and not c.execute('SELECT * FROM '+name+'.main.'+safe+' EXCEPT ALL SELECT * FROM main.'+safe).fetchall(),'Inherited source table changed: '+n)
  original=c.execute("SELECT view_name,sql FROM duckdb_views()WHERE database_name=?AND NOT internal",[name]).fetchall();current=dict(c.execute("SELECT view_name,sql FROM duckdb_views()WHERE database_name=current_database()AND NOT internal").fetchall());require(all(current.get(n)==sql for n,sql in original),'Inherited source view changed');return len(tables)
 finally:c.execute('DETACH '+name)

def summary(c):
 return dict(default_original_keys=c.execute('SELECT count(*)FROM policy_usable_keys').fetchone()[0],default_observations=c.execute('SELECT count(*)FROM policy_public_observations').fetchone()[0],user_authorized_source_keys=c.execute('SELECT count(*)FROM policy_source_priority').fetchone()[0],remaining_restricted_variants=c.execute('SELECT count(*)FROM policy_restricted_variants').fetchone()[0],pdf_references_outside_default=c.execute('SELECT count(*)FROM policy_pdf_default_withdrawals').fetchone()[0],numeric_reconciliation_claimed=False,reconciliation_required_for_read=False,extraction_changed=False,rag_integrated=False,base_sha256=BASE_SHA,base_code_sha256=BASE_CODE_SHA,policy_sha256=PLAN_SHA,code_sha256=sha(Path(__file__)))

def build(output):
 output=Path(output).resolve();require(output.is_relative_to(ROOT)and not output.exists(),'New project output directory required');require(sha(BASE)==BASE_SHA,'Base changed');output.mkdir(parents=True);db=output/'source_recovery_policy.duckdb';shutil.copyfile(BASE,db)
 with duckdb.connect(str(db))as c:
  rows,withdrawals=collect(c);c.execute('CREATE TABLE policy_source_priority AS SELECT o.*,NULL::VARCHAR observation_id,NULL::VARCHAR policy_family,NULL::BOOLEAN calculation_safe,NULL::BOOLEAN numeric_reconciliation_failed,NULL::VARCHAR policy_evidence FROM source_recovery_observations o WHERE FALSE');c.execute('ALTER TABLE policy_source_priority ADD PRIMARY KEY(book,row_number)');c.executemany('INSERT INTO policy_source_priority VALUES('+','.join('?'for _ in rows[0])+')',rows);c.execute('CREATE TABLE policy_pdf_default_withdrawals(observation_id VARCHAR PRIMARY KEY,preferred_observation_id VARCHAR,reason VARCHAR)');c.executemany('INSERT INTO policy_pdf_default_withdrawals VALUES(?,?,?)',withdrawals)
  for n,sql in views():c.execute('CREATE VIEW '+n+' AS '+sql)
  report=summary(c);report.update(status='built_pending_validation')
 report['db_sha256']=sha(db);(output/'manifest.json').write_text(dumps(report),encoding='utf8');return report

def check_new(c):
 rows,withdrawals=collect(c);require(set(c.execute('SELECT * FROM policy_source_priority').fetchall())==set(rows),'User-priority values or evidence changed');require(set(c.execute('SELECT * FROM policy_pdf_default_withdrawals').fetchall())==set(withdrawals),'PDF source preference changed')
 for n,sql in views():require(not c.execute('SELECT * FROM '+n+' EXCEPT ALL ('+sql+')').fetchall()and not c.execute('('+sql+')EXCEPT ALL SELECT * FROM '+n).fetchall(),'Source policy view changed: '+n)
 report=summary(c);require([report[k]for k in['default_original_keys','default_observations','user_authorized_source_keys','remaining_restricted_variants','pdf_references_outside_default']]==[42766,42880,16,130,6],'Source priority public counts changed');require(not c.execute('SELECT * FROM policy_usable_keys INTERSECT SELECT book,row_number FROM policy_restricted_variants').fetchall(),'Restricted source leaked into default');require(not c.execute('SELECT * FROM policy_usable_keys INTERSECT SELECT book,row_number FROM source_recovery_quarantine UNION ALL SELECT * FROM policy_usable_keys INTERSECT SELECT book,row_number FROM recovery3_private_inputs UNION ALL SELECT * FROM policy_usable_keys INTERSECT SELECT book,row_number FROM recovery3_diagram_nodes').fetchall(),'Quarantined/private/diagram source leaked into default');report['inherited_tables_preserved']=preserve(c);return report

def validate(db):
 db=Path(db);manifest=read(db.parent/'manifest.json');require(manifest['code_sha256']==sha(Path(__file__))and manifest['db_sha256']==sha(db),'Policy build data/code seal changed')
 with duckdb.connect(str(db),read_only=True)as c:report=check_new(c)
 report.update(status='pass',db_sha256=sha(db),validation_scope='source_selection_policy_plus_certified_base_preservation',full_numeric_chain_rerun=False);return report

def query_observation(db,*,observation_id,source_preference=None):
 require(source_preference in[None,'attached_workbook','published_pdf'],'Unknown source preference')
 with duckdb.connect(str(db),read_only=True)as c:
  if source_preference=='published_pdf':
   allowed=c.execute('SELECT preferred_observation_id FROM policy_pdf_default_withdrawals WHERE observation_id=?',[observation_id]).fetchone()
   if not allowed:raise LookupError('Exact retained published PDF observation required')
   d=prior.query_observation(db,observation_id=observation_id);d.update(default_preferred=False,source_preference='published_pdf',preferred_observation_id=allowed[0],warning='Referencia PDF conservada; el usuario ha dado prioridad al Excel para esta figura.');return d
  d=prior.rowdict(c.execute('SELECT * FROM policy_public_observations WHERE observation_id=?',[observation_id]));d=prior.enrich(c,d);selected=c.execute('SELECT policy_evidence FROM policy_source_priority WHERE observation_id=?',[observation_id]).fetchone()
  if selected:
   proof=json.loads(selected[0]);d.update(admission_basis='user_authorized_source_priority',source_preference='attached_workbook',numeric_reconciliation_failed=True,calculation_safe=False,global_calculation_prohibition=False,do_not_replace_published_values_to_force_sum=True,source_priority=proof['source_priority'],warning=proof['warning'],reconciliation_required_for_read=False)
  else:
   old=prior.query_observation(db,observation_id=observation_id);old.update(d);d=old
  d['safe_policy_default_view']=True;return d

def query_safe(db,*,book,row_number):
 require(book in['2025','2026']and type(row_number)is int and row_number>0,'Exact source key required');return query_observation(db,observation_id=aid(book,row_number))

def query_total(db,*,book,figure,year):
 require(book=='2025'and type(figure)is int and figure==54 and type(year)is int and year in[2023,2024],'Exact2025figure54published-total year required')
 with duckdb.connect(str(db),read_only=True)as c:
  ids=c.execute("SELECT observation_id FROM policy_source_priority WHERE book=?AND policy_family='variant54'AND series_literal='Total'AND category_literal=?",[book,str(year)]).fetchall()
  if len(ids)!=1:raise LookupError('No unique published Total row; no sum fallback')
 return query_observation(db,observation_id=ids[0][0])

def main():
 p=argparse.ArgumentParser();p.add_argument('command',choices=['build','validate']);p.add_argument('--output-dir');p.add_argument('--database');a=p.parse_args();report=build(a.output_dir)if a.command=='build'else validate(Path(a.database))
 if a.command=='validate':(Path(a.database).parent/'validation.json').write_text(dumps(report),encoding='utf8')
 print(dumps(report))
if __name__=='__main__':main()
