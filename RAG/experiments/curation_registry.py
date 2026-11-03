"""One separate declarative curation ledger; classifications never admit observations."""
import argparse,csv,hashlib,json,re
from pathlib import Path
from decimal import Decimal
import duckdb
ROOT=Path(__file__).resolve().parents[1]
DEFAULT_RULES=ROOT/'experiments/curation_rules.json'
STATES={'recuperable','bloqueado','excluido'}
CONFIDENCE={'confirmed','provisional'}
PHASES={'ready','semantic_contract','blocked','excluded'}
MATCH_FIELDS={'book','row_number','bloque','serie_id','serie_literal','parte_grafico_imagen','classification_reason','tipo','celda_valor','hoja_valor','unidad','metrica_literal','periodo_categoria_literal'}
RULE_FIELDS={'id','when','status','action','priority','confidence','evidence','phase','precedence'}
def require(ok,message):
 if not ok:raise ValueError(message)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(v):return json.dumps(v,ensure_ascii=False,sort_keys=True,separators=(',',':'),default=str)
def digest(v):return hashlib.sha256(dump(v).encode()).hexdigest()
def local(path):
 p=(ROOT/path).resolve();require(p.is_relative_to(ROOT),'Path outside project');return p

def load_rules(path):
 cfg=json.loads(Path(path).read_text(encoding='utf-8'));require(set(cfg)=={'version','baseline','default','rules','duplicate_pairs'},'Unexpected configuration fields');require(cfg['version']==1,'Unsupported config version')
 ids=set()
 for r in [dict(cfg['default'],id='default',when={},precedence=-1),*cfg['rules']]:
  require(set(r)<=RULE_FIELDS and {'id','when','status','action','priority','confidence','evidence','phase','precedence'}<=set(r),'Invalid rule fields')
  require(r['id'] not in ids,'Duplicate rule id');ids.add(r['id'])
  require(r['status'] in STATES and r['confidence'] in CONFIDENCE and r['phase'] in PHASES,'Invalid rule enum')
  require(r['phase'] in {'recuperable':{'ready','semantic_contract'},'bloqueado':{'blocked'},'excluido':{'excluded'}}[r['status']],'State and work phase conflict')
  require(isinstance(r['evidence'],dict),'Evidence must be structured data')
  if r['status']=='excluido':require(r['evidence'].get('kind') in ['technical_plot_role','verified_semantic_duplicate'],'Excluded state requires verified evidence kind')
  require(type(r['priority']) is int and type(r['precedence']) is int and isinstance(r['action'],str) and isinstance(r['evidence'],dict),'Invalid rule data')
  for field,predicate in r['when'].items():
   require(field in MATCH_FIELDS and isinstance(predicate,dict) and len(predicate)==1,'Invalid match field/operator')
   op,value=next(iter(predicate.items()));require(op in {'eq','in','is_null'},'Unsupported operator')
   if op=='in':require(isinstance(value,list) and value,'Empty or invalid membership')
   if op=='is_null':require(type(value) is bool,'Invalid null predicate')
   for v in value if op=='in' else [value]:require(v is None or type(v) in [str,int,bool],'Only scalar rule values allowed')
 require(cfg['default']['status']=='bloqueado' and cfg['default']['confidence']=='provisional','Default must remain conservative')
 pairs=cfg['duplicate_pairs'];require(len({(p['book'],p['row_number']) for p in pairs})==len(pairs),'Duplicate canonical mappings')
 for p in pairs:require(set(p)=={'book','row_number','canonical_book','canonical_row_number'},'Invalid duplicate mapping')
 return cfg

def matches(r,row):
 for field,predicate in r['when'].items():
  op,value=next(iter(predicate.items()));actual=row[field]
  if op=='eq' and actual!=value:return False
  if op=='in' and actual not in value:return False
  if op=='is_null' and (actual is None)!=value:return False
 return True

def baseline(cfg):
 b=cfg['baseline'];views=b['relations']
 for field,prefix,suffix in [('database','evaluation','.duckdb'),('validation','evaluation','.json'),('extractor','experiments','.py')]:
  candidate=Path(b[field]);require(candidate.parts[0]==prefix and candidate.suffix==suffix,'Unapproved baseline artifact path')
 for name in views.values():require(isinstance(name,str) and re.fullmatch('[a-z][a-z0-9_]*',name),'Unsafe source relation name')
 path=local(b['database']);cert=json.loads(local(b['validation']).read_text(encoding='utf-8'))
 require(cert['status']=='pass' and cert.get('previous_validation')=='pass','Baseline numeric validation not PASS')
 require(sha(path)==b['db_sha256']==cert['db_sha256'],'Frozen baseline hash changed')
 require(sha(local(b['extractor']))==cert['code_sha256'],'Frozen extractor hash changed')
 with duckdb.connect(str(path),read_only=True) as c:
  q=c.execute('SELECT * FROM '+views['pending']+' ORDER BY book,row_number');ns=[n[0] for n in q.description];pending=[dict(zip(ns,v)) for v in q.fetchall()]
  require(b['reason_field'] in ns,'Configured reason field absent')
  for row in pending:row['classification_reason']=row[b['reason_field']]
  admitted=set(c.execute('SELECT book,row_number FROM '+views['admitted']).fetchall())
  require(len(admitted)==b['expected_admitted']==cert['admitted'] and len(pending)==b['expected_pending']==cert['numeric_pending'],'Baseline counts changed')
  keys={(r['book'],r['row_number']) for r in pending};require(len(keys)==len(pending) and not keys&admitted,'Baseline key collision')
  q=c.execute('SELECT * FROM '+views['canonical_tables']);ns=[n[0] for n in q.description];canonical={(r['book'],r['row_number']):r for r in [dict(zip(ns,v)) for v in q.fetchall()]}
  q=c.execute(f"SELECT s.book,s.row_number,s.bloque,s.serie_id,s.parte_grafico_imagen FROM {views['staging']} s JOIN (SELECT * FROM {views['admitted']} UNION ALL SELECT book,row_number FROM {views['pending']}) k USING(book,row_number)");ns=[n[0] for n in q.description];inventory=[dict(zip(ns,v)) for v in q.fetchall()]
 require(len(inventory)==len(pending)+len(admitted),'Numeric inventory partition changed')
 return dict(pending=pending,admitted=admitted,canonical=canonical,inventory=inventory,certificate=cert)

def duplicate_evidence(row,pair,snapshot):
 if pair is None:return None
 key=pair['canonical_book'],pair['canonical_row_number'];t=snapshot['canonical'].get(key)
 if t is None or key not in snapshot['admitted']:return None
 if (row['book'],row['bloque'],row['tipo'])!=('2025','Gráfico 37','posicion_serie'):return None
 if (row['book'],row['bloque'],row['hoja_valor'],row['celda_valor'])!=(t['book'],t['block'],t['source_sheet'],t['source_cell']):return None
 if row['metrica_literal']!='RECURSOS NO FINANCIEROS SIN PRTR (%PIB)' or t['metric']!='GRÁFICO 37. RECURSOS NO FINANCIEROS SIN PRTR(%PIB)':return None
 if row['unidad']!='% PIB' or t['unit']!='percent_gdp' or row['serie_literal']!=t['series_literal'] or row['periodo_categoria_literal']!=t['category_literal']:return None
 try:
  if Decimal(row['decimal_XML_texto'])!=t['raw_value']:return None
 except Exception:return None
 if (t['scope'],t['role']) not in [('source_quantity_as_labelled','published_series_level'),('reference_line_not_independent_forecast','plot_reference_level')]:return None
 formula=row['formula_texto'] or None
 if formula!=t['formula']:
  e=json.loads(t['evidence'])
  if not (formula is None and t['role']=='plot_reference_level' and e.get('formula_resolution')=='original_formula_absent_from_universal' and e.get('universal_formula') is None):return None
  inputs=e.get('formula_inputs',{})
  if len(inputs)!=1 or Decimal(next(iter(inputs.values())))!=t['raw_value']:return None
 return dict(kind='verified_semantic_duplicate',canonical_book=key[0],canonical_row_number=key[1],canonical_table='table_observations',canonical_scope=t['scope'],canonical_role=t['role'],matched_fields=['source_location','raw_decimal','unit_alias','metric_alias','series','category','formula_or_certified_recovery'],admission_unchanged=True)

def group_key(row):return digest([row['book'],row['bloque'],row['serie_id'],row['parte_grafico_imagen']])
def group_kind(row):
 if row['parte_grafico_imagen'] and re.fullmatch(r'chart[0-9]+_s[1-9][0-9]*',row['serie_id'] or ''):return 'chart_series'
 return 'tabular_source_group' if row['parte_grafico_imagen'] is None else 'other_graphic_group'

def classify(cfg,snapshot):
 pairs={(p['book'],p['row_number']):p for p in cfg['duplicate_pairs']};ledger=[]
 for row in snapshot['pending']:
  found=[r for r in cfg['rules'] if matches(r,row)]
  if found:
   best=max(r['precedence'] for r in found);found=[r for r in found if r['precedence']==best];require(len(found)==1,'Ambiguous highest-precedence rules');r=found[0]
  else:r=dict(cfg['default'],id='default')
  status=r['status'];e=dict(r['evidence']);action=r['action'];priority=r['priority'];confidence=r['confidence'];phase=r['phase']
  if status=='excluido' and e.get('kind')=='technical_plot_role' and row['classification_reason'] not in ['forecast_marker_not_measure','drawing_helper_not_economic_series']:
   status='bloqueado';confidence='provisional';action='verify_technical_plot_role';priority=0;phase='blocked';e=dict(kind='technical_role_not_proved',admission_pending=True)
  if status=='excluido' and e.get('kind')=='verified_semantic_duplicate':
   proof=duplicate_evidence(row,pairs.get((row['book'],row['row_number'])),snapshot)
   if proof:e=proof
   else:status='bloqueado';confidence='provisional';action='verify_semantic_duplicate';priority=0;phase='blocked';e=dict(kind='duplicate_not_proved',admission_pending=True)
  e.update(base_database_sha256=cfg['baseline']['db_sha256'],rule_id=r['id'],classification_not_admission=True)
  ledger.append((row['book'],row['row_number'],group_key(row),status,row['classification_reason'],r['id'],action,priority,confidence,phase,dump(e),digest(row)))
 groups={}
 for row in snapshot['inventory']:
  g=group_key(row)
  if g not in groups:groups[g]=[g,row['book'],row['bloque'],row['serie_id'],row['parte_grafico_imagen'],group_kind(row),0,0]
  groups[g][6]+=1;groups[g][7]+=int((row['book'],row['row_number']) in snapshot['admitted'])
 return ledger,[tuple(groups[g]) for g in sorted(groups)]

def plan_rows(ledger,groups):
 metadata={r[0]:r for r in groups};families={}
 for row in ledger:
  if row[3]!='recuperable':continue
  g=metadata[row[2]];key=g[1],g[2],row[6],row[9]
  item=families.setdefault(key,[g[1],g[2],row[6],row[9],0,set(),row[7]])
  item[4]+=1;item[5].add(row[2]);item[6]=max(item[6],row[7])
 return sorted([(b,block,action,phase,n,len(gs),priority) for b,block,action,phase,n,gs,priority in families.values()],key=lambda r:(-r[6],r[:4]))

def write_csv(path,columns,rows):
 with Path(path).open('w',encoding='utf-8',newline='') as f:
  writer=csv.writer(f);writer.writerow(columns);writer.writerows(rows)

def build(rules,output):
 require(not output.exists(),'Output directory exists');cfg=load_rules(rules);snap=baseline(cfg);ledger,groups=classify(cfg,snap);plan=plan_rows(ledger,groups)
 output.mkdir(parents=True);db=output/'registry.duckdb'
 with duckdb.connect(str(db)) as c:
  c.execute('CREATE TABLE ledger(book VARCHAR,row_number INTEGER,group_id VARCHAR,status VARCHAR,reason VARCHAR,rule_id VARCHAR,action VARCHAR,priority INTEGER,confidence VARCHAR,phase VARCHAR,evidence VARCHAR,source_payload_sha256 VARCHAR,PRIMARY KEY(book,row_number))');c.executemany('INSERT INTO ledger VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',ledger)
  c.execute('CREATE TABLE source_groups(group_id VARCHAR PRIMARY KEY,book VARCHAR,block VARCHAR,series_id VARCHAR,part VARCHAR,group_kind VARCHAR,numeric_rows INTEGER,admitted_rows INTEGER)');c.executemany('INSERT INTO source_groups VALUES (?,?,?,?,?,?,?,?)',groups)
  c.execute("CREATE VIEW group_coverage AS SELECT g.*,numeric_rows-admitted_rows pending_rows,CASE WHEN admitted_rows=0 THEN 'none_admitted' WHEN admitted_rows=numeric_rows THEN 'all_numeric_admitted' ELSE 'partially_admitted' END coverage FROM source_groups g")
  counts=c.execute('SELECT status,count(*) FROM ledger GROUP BY status ORDER BY status').fetchall()
 report=dict(status='built_pending_ledger_validation',admitted_unchanged=len(snap['admitted']),pending_classified=len(ledger),numeric_candidates=len(snap['admitted'])+len(ledger),states=dict(counts),pending_groups=len({r[2] for r in ledger}),pending_blocks=len({(r['book'],r['bloque']) for r in snap['pending']}),numeric_base_revalidation_executed=False,reused_numeric_certificate=cfg['baseline']['validation'],base_db_sha256=cfg['baseline']['db_sha256'],rules_sha256=sha(rules),code_sha256=sha(Path(__file__)),registry_sha256=sha(db),rag_integrated=False,new_admissions=0)
 write_csv(output/'batch_plan.csv',['book','block','action','phase','candidate_rows','source_groups','priority'],plan)
 with duckdb.connect(str(db),read_only=True) as c:
  q=c.execute('SELECT * FROM group_coverage ORDER BY book,block,series_id,part');write_csv(output/'group_coverage.csv',[n[0] for n in q.description],q.fetchall())
  q=c.execute('SELECT book,block,sum(numeric_rows) numeric_rows,sum(admitted_rows) admitted_rows,sum(pending_rows) pending_rows,count(*) source_groups FROM group_coverage GROUP BY book,block ORDER BY book,block');write_csv(output/'block_coverage.csv',[n[0] for n in q.description],q.fetchall())
 (output/'manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');return report

def validate(registry,rules):
 cfg=load_rules(rules);snap=baseline(cfg);expected,groups=classify(cfg,snap)
 manifest=json.loads(registry.with_name('manifest.json').read_text(encoding='utf-8'))
 require(manifest['rules_sha256']==sha(rules) and manifest['code_sha256']==sha(Path(__file__)),'Registry configuration/code changed')
 require(manifest['registry_sha256']==sha(registry),'Registry hash changed')
 with duckdb.connect(str(registry),read_only=True) as c:
  actual=c.execute('SELECT * FROM ledger').fetchall();require(len(actual)==len(expected) and set(actual)==set(expected),'Ledger classifications differ')
  actual_groups=c.execute('SELECT * FROM source_groups').fetchall();require(len(actual_groups)==len(groups) and set(actual_groups)==set(groups),'Source groups differ')
  require(not c.execute("SELECT * FROM ledger WHERE status NOT IN ('recuperable','bloqueado','excluido') OR evidence IS NULL").fetchall(),'Invalid ledger state/evidence')
  q="SELECT g.*,numeric_rows-admitted_rows pending_rows,CASE WHEN admitted_rows=0 THEN 'none_admitted' WHEN admitted_rows=numeric_rows THEN 'all_numeric_admitted' ELSE 'partially_admitted' END coverage FROM source_groups g"
  require(not c.execute('SELECT * FROM group_coverage EXCEPT ALL '+q).fetchall() and not c.execute(q+' EXCEPT ALL SELECT * FROM group_coverage').fetchall(),'Group coverage view differs')
  states=dict(c.execute('SELECT status,count(*) FROM ledger GROUP BY status').fetchall())
 def verify_csv(name,columns,rows):
  with (registry.parent/name).open(encoding='utf-8',newline='') as f:actual=list(csv.reader(f))
  wanted=[columns]+[['' if v is None else str(v) for v in row] for row in rows]
  require(actual==wanted,'Export differs: '+name)
 verify_csv('batch_plan.csv',['book','block','action','phase','candidate_rows','source_groups','priority'],plan_rows(expected,groups))
 with duckdb.connect(str(registry),read_only=True) as c:
  q=c.execute('SELECT * FROM group_coverage ORDER BY book,block,series_id,part');verify_csv('group_coverage.csv',[n[0] for n in q.description],q.fetchall())
  q=c.execute('SELECT book,block,sum(numeric_rows) numeric_rows,sum(admitted_rows) admitted_rows,sum(pending_rows) pending_rows,count(*) source_groups FROM group_coverage GROUP BY book,block ORDER BY book,block');verify_csv('block_coverage.csv',[n[0] for n in q.description],q.fetchall())
 return dict(status='pass',validation_scope='complete_curation_ledger_and_frozen_numeric_certificate',classified=len(expected),admitted_unchanged=len(snap['admitted']),states=states,new_admissions=0,numeric_base_revalidation_executed=False,base_db_sha256=cfg['baseline']['db_sha256'],rules_sha256=sha(rules),code_sha256=sha(Path(__file__)),registry_sha256=sha(registry),rag_integrated=False)

def main():
 p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='command',required=True)
 b=s.add_parser('build');b.add_argument('--rules',type=Path,default=DEFAULT_RULES);b.add_argument('--output-dir',type=Path,required=True)
 b=s.add_parser('validate');b.add_argument('--rules',type=Path,default=DEFAULT_RULES);b.add_argument('--registry',type=Path,required=True);b.add_argument('--output',type=Path,required=True)
 b=s.add_parser('plan');b.add_argument('--rules',type=Path,default=DEFAULT_RULES);b.add_argument('--registry',type=Path,required=True);b.add_argument('--phase',choices=['ready','semantic_contract','all'],default='ready');b.add_argument('--book');b.add_argument('--block')
 b=s.add_parser('revalidate-base');b.add_argument('--rules',type=Path,default=DEFAULT_RULES);b.add_argument('--output',type=Path,required=True)
 a=p.parse_args()
 if a.command=='build':report=build(a.rules,a.output_dir)
 elif a.command=='plan':
  validation=validate(a.registry,a.rules)
  with duckdb.connect(str(a.registry),read_only=True) as c:l=c.execute('SELECT * FROM ledger').fetchall();g=c.execute('SELECT * FROM source_groups').fetchall()
  tasks=[r for r in plan_rows(l,g) if (a.phase=='all' or r[3]==a.phase) and (a.book is None or r[0]==a.book) and (a.block is None or r[1]==a.block)];report=dict(status='plan',phase=a.phase,families=tasks,ledger_validation=validation['status'],new_admissions=0,rag_integrated=False)
 elif a.command=='revalidate-base':
  require(not a.output.exists(),'Report exists');cfg=load_rules(a.rules);baseline(cfg)
  require(cfg['baseline']['extractor']=='experiments/labour_unit_semantics.py','Full numeric validator needs approved central adapter')
  from experiments.labour_unit_semantics import validate as numeric_validate
  report=numeric_validate(local(cfg['baseline']['database']));report['validation_scope']='fresh_complete_numeric_base_revalidation';a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
 else:
  require(not a.output.exists(),'Report exists')
  try:report=validate(a.registry,a.rules)
  except Exception as e:report=dict(status='fail',error=str(e),rag_integrated=False)
  a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
 print(json.dumps(report,ensure_ascii=False,indent=2));return int(report['status']=='fail')
if __name__=='__main__':raise SystemExit(main())
