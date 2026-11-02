"""Recover only the 119 classified ready candidates; no RAG integration."""
import argparse,json,re,shutil,xml.etree.ElementTree as ET
from pathlib import Path
from decimal import Decimal
import duckdb
from openpyxl.utils.cell import get_column_letter
from experiments.labour_unit_semantics import FIELDS,validate as validate_previous
from experiments.quantity_semantics import QuantityResolver,QUANTITIES
from experiments.source_semantics import SourceResolver
from experiments.bulk_semantics import C,ref_cells
from experiments.indexed_semantics import points
from experiments.excel_pilot import ROOT,sha,PDF_HASHES
from experiments.universal_database import require
from experiments import curation_registry as registry
PREVIOUS=ROOT/'evaluation/labour_unit_semantics_2026-10-31/run_v1/labour_units.duckdb'
PREVIOUS_SHA='cda1a2161603569ee0cf6d7cd00c0baebb2c47318d6b3b8ed4f7eb9df9b8da77'
REGISTRY=ROOT/'evaluation/curation_registry_2026-10-31/run_v5/registry.duckdb'
REGISTRY_SHA='6660468d598d2b2456a31229197150c42f497a2c57efcdb987a207a927ba11d9'
RULES_SHA='67affbaab2d454bc113442234a8f24a152f6841f8d58d573d0f00b6fdf982e5c'
PROFILES={
 ('2025','Gráfico 89'):('chart144_s5','C4',"'Gráfico 89'!$C$5:$C$53",'Envejecimiento','percentage_points_gdp','0'),
 ('2026','Gráfico 37'):('chart46_s5','C4',"'Gráfico 37'!$C$5:$C$30",'Envejecimiento','percentage_points_gdp','0'),
 ('2026','Gráfico 12'):('chart18_s4','D18',"'Gráfico 12'!$E$18:$AI$18",'Transferencias implícitas','percent_gdp','#,##0.0')}
CATEGORY_KEYS={12191,12192,12204,12205,12230,12231,14257,14258,18288,18324,18325}
DIFFERENCES={'Q3':('Q6','K6','Q4','K4','2029 menos 2023',0),'AL3':('AL6','Q6','AL4','Q4','2050 menos 2029',1)}
def dumps(x):return json.dumps(x,ensure_ascii=False,sort_keys=True,default=str)
def selected():
 require(sha(PREVIOUS)==PREVIOUS_SHA and sha(REGISTRY)==REGISTRY_SHA and sha(registry.DEFAULT_RULES)==RULES_SHA,'Frozen ready inputs changed')
 registry.validate(REGISTRY,registry.DEFAULT_RULES)
 with duckdb.connect(str(REGISTRY),read_only=True) as c:rows=c.execute("SELECT book,row_number FROM ledger WHERE status='recuperable' AND phase='ready' ORDER BY book,row_number").fetchall()
 require(len(rows)==len(set(rows))==119,'Ready selection changed');return set(rows)
class ReadyResolver(QuantityResolver):
 def binding(self,row):
  p=PROFILES.get((row['book'],row['bloque']))
  if p is None:return super().binding(row)
  sid,header,vr,label,unit,fmt=p;source=self.sources[row['book']];sheet=row['bloque']
  require(row['serie_id']==sid and row['serie_literal']==label,'Ready series changed')
  part='xl/charts/'+sid.split('_')[0]+'.xml';require(row['parte_grafico_imagen']==part,'Ready part changed')
  root=ET.fromstring(source.xml.parts[part]);node=root.findall('.//c:ser',C)[int(sid.split('_s')[1])-1]
  require(node.find('c:cat',C) is None and node.find('c:bubbleSize',C) is None,'Ready category origin changed')
  name=node.find('c:tx/c:strRef/c:f',C);value=node.find('c:val/c:numRef/c:f',C)
  require(name is not None and name.text==f"'{sheet}'!${re.match('[A-Z]+',header)[0]}${re.search('[0-9]+',header)[0]}",'Ready name reference changed')
  require(value is not None and value.text==vr,'Ready value reference changed')
  require(source.text(sheet,header)==label,'Ready physical name changed')
  cache=node.find('c:tx/c:strRef/c:strCache',C);require(cache is not None and points(cache)=={0:label},'Ready name cache changed')
  _,cells=ref_cells(vr)
  cats=['B'+re.search('[0-9]+',a)[0] for a in cells] if header=='C4' else [get_column_letter(i)+'5' for i in range(5,36)]
  categories=[self.category(source,sheet,a) for a in cats]
  expected_years={('2025','Gráfico 89'):(2022,2070),('2026','Gráfico 37'):(2025,2050),('2026','Gráfico 12'):(2020,2050)}[row['book'],sheet]
  require(categories==[str(y) for y in range(expected_years[0],expected_years[1]+1)],'Ready physical category vector changed')
  parent=next(x for x in root.iter() if node in list(x));ids={x.get('val') for x in parent.findall('c:axId',C)}
  axes=[x for x in root.findall('.//c:valAx',C) if x.find('c:axId',C).get('val') in ids]
  require(len(axes)==1,'Ready associated axis ambiguous');axis=axes[0];nf=axis.find('c:numFmt',C)
  require(nf is not None and nf.get('formatCode')==fmt and axis.find('c:dispUnits',C) is None,'Ready axis units changed')
  vp=points(node.find('c:val/c:numRef/c:numCache',C))
  require(set(vp)==set(range(len(cells))),'Ready value cache vector changed')
  for i,a in enumerate(cells):require(Decimal(vp[i])==source.number(sheet,a),'Ready value cache mismatch')
  return dict(label=label,name_ref=name.text,category_ref=f"'{sheet}'!{cats[0]}:{cats[-1]}",value_ref=vr,value_sheet=sheet,cells=cells,categories=categories,valid_indices=set(range(len(cells))),axis_id=axis.find('c:axId',C).get('val'),axis_format=fmt,display_unit='none',divisor=Decimal(1),multiplier=Decimal(1),mode='bounded_ready_same_source_grid',category_basis='reviewed_source_grid',name_basis='source_scalar')
 def unit_spec(self,row,texts,binding):
  p=PROFILES.get((row['book'],row['bloque']))
  if p:
   proof=self.prove(row);require(binding['multiplier']==binding['divisor']==1,'Ready scale changed')
   self.supplements[row['book'],row['bloque']]=proof
   return 'unidad del contexto publicado y eje asociado',p[4],'reviewed_ready_quantity_profile',('quantity_source_profile',dumps(proof))
  return super().unit_spec(row,texts,binding)
 def review_ready(self,row):
  try:
   if row['book']=='2025' and row['row_number'] in {18324,18325}:return 'literal_series_source_conflict:chart_D16_Demorada_but_values_F7_R7_anticipada',None
   if row['tipo']=='variacion_publicada':return self.difference(row)
   b=self.binding(row);m=re.fullmatch(re.escape(row['serie_id'])+r'_p([0-9]+)',row['id'] or '');require(m is not None,'Ready point identity')
   i=int(m[1]);require(i<len(b['categories']) and i in b['valid_indices'],'Ready point outside binding')
   cat=b['categories'][i];original=row['periodo_categoria_literal'];evidence={}
   if (row['book'],row['bloque']) in PROFILES:
    require(original is None,'Ready omitted category changed');action='physical_same_source_grid'
   else:
    require(row['book']=='2025' and row['row_number'] in CATEGORY_KEYS,'Unreviewed category key')
    require(isinstance(cat,str) and '\n' in cat and cat.replace('\n',' / ')==original,'Category normalization not lossless')
    action='lossless_newline_representation';evidence=dict(original_source_category=cat,universal_category=original,normalization="newline -> ' / '")
   shadow=dict(row,periodo_categoria_literal=cat);status,result=SourceResolver.review(self,shadow)
   if result is None:return status,None
   source=self.sources[row['book']];physical_formula=source.text(row['hoja_valor'],row['celda_valor']);physical_formula=physical_formula if isinstance(physical_formula,str) and physical_formula.startswith('=') else None
   if (row['book'],row['bloque'])==('2026','Gráfico 12'):
    input_cell=re.match('[A-Z]+',row['celda_valor'])[0]+'17'
    require(physical_formula=='=-'+input_cell and source.number(row['hoja_valor'],row['celda_valor'])==-source.number(row['hoja_valor'],input_cell),'Implicit funding sign/formula changed')
    evidence.update(formula_inputs={input_cell:str(source.number(row['hoja_valor'],input_cell))})
   evidence.update(source_formula=physical_formula,universal_formula=row['formula_texto'])
   evidence.update(action=action,category_ref=b['category_ref'],name_ref=b['name_ref'],value_ref=b['value_ref'],source_part=row['parte_grafico_imagen'])
   role='total_ageing_debt_impact_vs_no_ageing_not_additional_component' if row['serie_literal']=='Envejecimiento' else ('implicit_pension_funding_as_labelled' if row['serie_literal']=='Transferencias implícitas' else 'source_quantity_as_labelled')
   return 'usable_ready',result+(original,dumps(evidence),row['serie_literal'],row['hoja_valor'],row['celda_valor'],'{}',role,'source_category_as_labelled',dumps(evidence))
  except (ValueError,ArithmeticError,KeyError,IndexError,AttributeError,TypeError,StopIteration) as e:return 'ready_evidence_review:'+str(e),None
 def difference(self,row):
  require(row['book']=='2025' and row['bloque']=='Gráfico 37' and row['serie_id']=='g37_variaciones' and row['parte_grafico_imagen'] is None,'Difference identity changed')
  cell=row['celda_valor'];require(cell in DIFFERENCES and row['hoja_valor']=='Gráfico 37','Difference source changed')
  a,b,ca,cb,category,index=DIFFERENCES[cell];s=self.sources['2025'];sheet='Gráfico 37';formula='='+a+'-'+b
  require(row['id']=='g37_'+cell,'Difference record identity changed')
  require(row['unidad']=='puntos PIB' and row['serie_literal']=='Variacion recursos' and row['periodo_categoria_literal']==category,'Difference semantic identity changed')
  title='GRÁFICO 37. RECURSOS NO FINANCIEROS SIN PRTR(%PIB)'
  require(row['titulo']==row['metrica_literal']==s.text(sheet,'B2')==title,'Difference metric changed')
  require(row['formula_texto']==s.text(sheet,cell)==formula,'Difference formula changed')
  raw=s.number(sheet,cell);require(raw==Decimal(row['decimal_XML_texto']),'Difference raw value changed')
  first=s.number(sheet,ca);second=s.number(sheet,cb);expected_periods={'Q3':(Decimal(2029),Decimal(2023)),'AL3':(Decimal(2050),Decimal(2029))}[cell];require((first,second)==expected_periods,'Difference physical period changed')
  x=s.number(sheet,a);y=s.number(sheet,b);residual=raw-(x-y)
  require(abs(residual)<=Decimal('1e-14'),'Difference cached arithmetic inconsistent')
  proof=dict(formula=formula,inputs={a:str(x),b:str(y)},period_cells=[ca,cb],period_values=[str(first),str(second)],raw_cache=str(raw),exact_decimal_difference=str(x-y),cache_rounding_residual=str(residual),cache_tolerance='1e-14',role='published_period_difference_not_level',universal_category=category)
  d={n:None for n in FIELDS};d.update(book='2025',row_number=row['row_number'],record_id=row['id'],block=sheet,metric_literal=title,series_id=row['serie_id'],series_literal=row['serie_literal'],category_literal=category,unit='percentage_points_gdp',value=raw,source_sheet=sheet,source_cell=cell,unit_cell='B2',unit_text=title,value_ref=f"'{sheet}'!{cell}",category_ref=f"'{sheet}'!{ca},{cb}",rule_id='reviewed_ready_period_difference',raw_value=raw,cell_format=s.workbook[sheet][cell].number_format,divisor=Decimal(1),multiplier=Decimal(1),scope_literal='published_period_difference_not_level',point_index=index,binding_mode='explicit_difference_formula',unit_basis='difference_of_percent_gdp_levels',metric_evidence=dumps([['B2',title]]),category_basis='two_explicit_period_cells',name_basis='universal_published_difference_label',semantic_scope='published_period_difference_not_level',title_resolution='exact_source_title',supplemental_evidence=dumps(proof),universal_category=category,grid_evidence='{}',universal_series=row['serie_literal'],universal_value_sheet=row['hoja_valor'],universal_value_cell=cell,reconciliation_evidence=dumps(proof),quantity_role='published_period_difference_not_level',category_semantics='end_period_minus_start_period_as_labelled',quantity_evidence=dumps(proof))
  return 'usable_ready',tuple(d[n] for n in FIELDS)
def quarantine(c,resolver):
 s=resolver.sources['2025'];sheet='Gráfico RE 3.4';part='xl/charts/chart91.xml';root=ET.fromstring(s.xml.parts[part]);node=root.findall('.//c:ser',C)[0]
 require(node.find('c:tx/c:strRef/c:f',C).text=="'Gráfico RE 3.4'!$D$16" and node.find('c:val/c:numRef/c:f',C).text=="'Gráfico RE 3.4'!$F$7:$R$7" and node.find('c:cat/c:strRef/c:f',C).text=="'Gráfico RE 3.4'!$F$6:$R$6",'Quarantine source references changed')
 require(s.text(sheet,'D16')=='Demorada' and s.text(sheet,'D7')=='JUBILACIÓN ANTICIPADA' and s.text(sheet,'C2')=='JUBILACIONES ANTICIPADAS. DATOS AGREGADOS (%ALTAS TOTALES)','Quarantine physical conflict changed')
 require(points(node.find('c:tx/c:strRef/c:strCache',C))=={0:'Demorada'},'Quarantine name cache changed')
 vp=points(node.find('c:val/c:numRef/c:numCache',C));require(set(vp)==set(range(13)),'Quarantine value cache incomplete')
 cursor=c.execute("SELECT * FROM staging_records WHERE book='2025' AND row_number BETWEEN 18313 AND 18325 ORDER BY row_number");names=[x[0] for x in cursor.description];source_rows=[dict(zip(names,v)) for v in cursor.fetchall()];require(len(source_rows)==13,'Quarantine keys changed')
 admitted=set(c.execute('SELECT * FROM labour_usable_keys').fetchall());result=[]
 for i,row in enumerate(source_rows):
  cell=get_column_letter(6+i)+'7';require(row['row_number']==18313+i and row['serie_id']=='chart91_s1' and row['id']=='chart91_s1_p'+str(i) and row['serie_literal']=='Demorada' and row['hoja_valor']==sheet and row['celda_valor']==cell and row['parte_grafico_imagen']==part,'Quarantine staged identity changed')
  raw=s.number(sheet,cell);require(raw==Decimal(row['decimal_XML_texto'])==Decimal(vp[i]),'Quarantine raw provenance changed');prior=('2025',row['row_number']) in admitted;require(prior==(i<11),'Quarantine prior membership changed')
  evidence=dict(chart_part=part,name_ref="'Gráfico RE 3.4'!$D$16",name='Demorada',value_ref="'Gráfico RE 3.4'!$F$7:$R$7",category_ref="'Gráfico RE 3.4'!$F$6:$R$6",value_row_label_cell='D7',value_row_label='JUBILACIÓN ANTICIPADA',caption_cell='C2',caption=s.text(sheet,'C2'),source_cell=cell,raw_value=str(raw),other_delayed_source_cell=get_column_letter(6+i)+'16',other_delayed_value=str(s.number(sheet,get_column_letter(6+i)+'16')),action='quarantine_without_relabelling')
  result.append(('2025',row['row_number'],prior,'literal_series_source_conflict',dumps(evidence)))
 return result
def collect(c):
 keys=selected();q=c.execute('SELECT * FROM labour_numeric_queue ORDER BY book,row_number');names=[x[0] for x in q.description];data=[dict(zip(names,x)) for x in q.fetchall()]
 require(len(data)==8827 and c.execute('SELECT count(*) FROM labour_usable_keys').fetchone()[0]==38386,'Previous partition changed')
 require(keys<={(r['book'],r['row_number']) for r in data},'Ready key absent');resolver=ReadyResolver();rows=[];reviews=[]
 try:
  for row in data:
   if (row['book'],row['row_number']) not in keys:continue
   status,result=resolver.review_ready(row);reviews.append((row['book'],row['row_number'],status))
   if result is not None:rows.append(result)
  pages=[(b,p,PDF_HASHES[b],t) for (b,p),t in sorted(resolver.pdf_pages.items())];quarantined=quarantine(c,resolver)
 finally:resolver.close()
 return rows,reviews,pages,quarantined
def expressions(c):
 columns=c.execute('SELECT * FROM labour_numeric_queue LIMIT 0').description;staging={x[0] for x in c.execute('SELECT * FROM staging_records LIMIT 0').description}
 projection=','.join('s."'+n+'"' if n in staging else 'NULL::'+str(t)+' AS "'+n+'"' for n,t,*_ in columns)
 safe="SELECT p.* FROM labour_usable_keys p WHERE NOT EXISTS(SELECT 1 FROM ready_quarantine q WHERE q.book=p.book AND q.row_number=p.row_number) UNION ALL SELECT book,row_number FROM ready_observations"
 pending="SELECT s.*,COALESCE(r.status,s.labour_reason) ready_reason FROM labour_numeric_queue s LEFT JOIN ready_review r USING(book,row_number) WHERE COALESCE(r.status,'')<>'usable_ready'"
 restored="SELECT "+projection+",q.reason AS ready_reason FROM staging_records s JOIN ready_quarantine q USING(book,row_number) WHERE q.previously_admitted"
 return [('ready_usable_keys',safe),('ready_numeric_queue',pending+' UNION ALL '+restored)]

def build(output):
 output=Path(output).resolve();require(output.is_relative_to(ROOT) and not output.exists(),'Output must be new project directory');require(sha(PREVIOUS)==PREVIOUS_SHA,'Previous database changed')
 output.mkdir(parents=True);db=output/'ready_recovery.duckdb';shutil.copyfile(PREVIOUS,db)
 with duckdb.connect(str(db)) as c:
  rows,reviews,pages,quarantined=collect(c);c.execute('CREATE TABLE ready_observations AS SELECT * FROM labour_observations WHERE false');c.execute('ALTER TABLE ready_observations ADD PRIMARY KEY(book,row_number)')
  if rows:c.executemany('INSERT INTO ready_observations VALUES ('+','.join('?' for _ in FIELDS)+')',rows)
  c.execute('CREATE TABLE ready_review(book VARCHAR,row_number INTEGER,status VARCHAR,PRIMARY KEY(book,row_number))');c.executemany('INSERT INTO ready_review VALUES (?,?,?)',reviews)
  c.execute('CREATE TABLE ready_pdf_evidence(book VARCHAR,page INTEGER,pdf_sha256 VARCHAR,text VARCHAR,PRIMARY KEY(book,page))')
  if pages:c.executemany('INSERT INTO ready_pdf_evidence VALUES (?,?,?,?)',pages)
  c.execute('CREATE TABLE ready_quarantine(book VARCHAR,row_number INTEGER,previously_admitted BOOLEAN,reason VARCHAR,evidence VARCHAR,PRIMARY KEY(book,row_number))');c.executemany('INSERT INTO ready_quarantine VALUES (?,?,?,?,?)',quarantined)
  for name,sql in expressions(c):c.execute('CREATE VIEW '+name+' AS '+sql)
  sources=c.execute('SELECT * FROM sources ORDER BY book').fetchall()
 report=dict(status='built_pending_validation',added=len(rows),admitted=38375+len(rows),numeric_pending=8838-len(rows),previous_admitted_stored=38386,previous_admitted_quarantined=11,selected_ready=119,previous_sha256=PREVIOUS_SHA,registry_sha256=REGISTRY_SHA,rules_sha256=RULES_SHA,code_sha256=sha(Path(__file__)),db_sha256=sha(db),sources=sources,rag_integrated=False)
 (output/'manifest.json').write_text(dumps(report),encoding='utf-8');return report
def check_new(c):
 rows,reviews,pages,quarantined=collect(c)
 for table,expected in [('ready_observations',rows),('ready_review',reviews),('ready_pdf_evidence',pages),('ready_quarantine',quarantined)]:
  actual=c.execute('SELECT * FROM '+table).fetchall();require(len(actual)==len(expected) and set(actual)==set(expected),'Ready evidence mismatch: '+table)
 for name,sql in expressions(c):
  require(c.execute('SELECT * FROM '+name+' LIMIT 0').description==c.execute(sql+' LIMIT 0').description,'Ready view columns changed')
  for a,b in [(name,'('+sql+')'),('('+sql+')',name)]:require(not c.execute('SELECT * FROM '+a+' EXCEPT ALL SELECT * FROM '+b).fetchall(),'Ready view multiplicity changed')
 require(c.execute('SELECT count(*) FROM (SELECT book,row_number FROM ready_usable_keys GROUP BY ALL HAVING count(*)<>1)').fetchone()[0]==0,'Ready admission collision')
 require(c.execute('SELECT count(*) FROM ready_usable_keys').fetchone()[0]==38375+len(rows) and c.execute('SELECT count(*) FROM ready_numeric_queue').fetchone()[0]==8838-len(rows),'Safe partition counts changed')
 require(not c.execute('SELECT book,row_number FROM ready_numeric_queue INTERSECT SELECT book,row_number FROM ready_usable_keys').fetchall(),'Safe queue overlaps admissions')
 require(c.execute('SELECT count(*) FROM (SELECT book,row_number FROM ready_numeric_queue UNION SELECT book,row_number FROM ready_usable_keys)').fetchone()[0]==47213,'Safe candidate partition changed')
 return rows
def validate(db):
 previous=validate_previous(db)
 with duckdb.connect(str(db),read_only=True) as c:
  rows=check_new(c);stats=c.execute('SELECT book,block,unit,quantity_role,count(*) FROM ready_observations GROUP BY ALL ORDER BY book,block').fetchall();blocked=c.execute("SELECT * FROM ready_review WHERE status<>'usable_ready' ORDER BY book,row_number").fetchall()
 return dict(status='pass',previous_validation=previous['status'],added=len(rows),admitted=38375+len(rows),numeric_pending=8838-len(rows),previous_admitted_stored=38386,previous_admitted_quarantined=11,selected_ready=119,blocked_ready=blocked,stats=stats,db_sha256=sha(Path(db)),code_sha256=sha(Path(__file__)),previous_sha256=PREVIOUS_SHA,registry_sha256=REGISTRY_SHA,rules_sha256=RULES_SHA,rag_integrated=False)
def query(db,*,book,block,series_id,point_index):
 require(book in ['2025','2026'] and all(isinstance(x,str) and x.strip() for x in [block,series_id]) and type(point_index) is int and point_index>=0,'Exact ready identity required')
 with duckdb.connect(str(db),read_only=True) as c:
  q=c.execute('SELECT t.*,s.xlsx_path,s.xlsx_sha256,s.pdf_path,s.pdf_sha256 FROM ready_observations t JOIN ready_usable_keys k USING(book,row_number) JOIN sources s USING(book) WHERE book=? AND block=? AND series_id=? AND point_index=?',[book,block,series_id,point_index]);names=[n[0] for n in q.description];rows=q.fetchall()
 if len(rows)!=1:raise LookupError('No unique ready point; no fallback')
 return {n:str(v) if isinstance(v,Decimal) else v for n,v in zip(names,rows[0])}
def main():
 p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='command',required=True);b=s.add_parser('build');b.add_argument('--output-dir',type=Path,required=True);b=s.add_parser('validate');b.add_argument('--db',type=Path,required=True);b.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.command=='build':report=build(a.output_dir)
 else:
  require(not a.output.exists(),'Report exists')
  try:report=validate(a.db)
  except Exception as e:report=dict(status='fail',error=str(e))
  a.output.write_text(dumps(report),encoding='utf-8')
 print(dumps(report));return int(report['status']=='fail')
if __name__=='__main__':raise SystemExit(main())
