"""Bounded recovery of reviewed pending source groups, without RAG integration."""
import argparse,json,re,shutil,xml.etree.ElementTree as ET
from pathlib import Path
from decimal import Decimal
import duckdb
from pypdf import PdfReader
from experiments.resolved_semantics import canon
from experiments.ready_recovery_semantics import FIELDS,validate as validate_previous
from experiments.source_semantics import SourceResolver
from experiments.bulk_semantics import C,ref_cells
from experiments.indexed_semantics import points
from experiments.semantic_families import M
from experiments.scaled_semantics import format_effect
from experiments.excel_pilot import ROOT,sha,PDF_HASHES
from experiments.universal_database import require
PREVIOUS=ROOT/'evaluation/ready_recovery_2026-10-31/run_v6/ready_recovery.duckdb'
PREVIOUS_SHA='3d3931271f26c9c12d1c3e5f12e5f12ca263adbecde625fd915cfef23122fa8f'
GR22='PROYECCIÓN DE LA TASA DE ACTIVIDAD (% POBLACIÓN 16-74 AÑOS)'
SPECS={
 ('2026','Anexo I. Gr 2A'):dict(title='Evolución del número de hijos por mujer. España en el contexto mundial y de principales países emisores',headers=[('B2','ANEXO I. GRÁFICO 2. ÍNDICE COYUNTURAL DE FECUNDIDAD EN EL ENTORNO INTERNACIONAL'),('B3','ANEXO I. GRÁFICO 2A. EVOLUCIÓN DEL NÚMERO DE HIJOS POR MUJER. ESPAÑA EN EL CONTEXTO MUNDIAL Y DE PAÍSES EMISORES'),('B19','Número de hijos por mujer')],unit='children_per_woman',factor=1,role='minimum_across_country_group_period_fertility_icf',category_semantics='observation_calendar_year_as_labelled',pdf=(80,['ÍNDICE COYUNTURAL DE FECUNDIDAD','ANEXO I. GRÁFICO 2'])),
 ('2026','Anexo I. Gr 2B'):dict(title='Evolución del número de hijos por mujer. España en el contexto europeo.',headers=[('B2','ANEXO I. GRÁFICO 2. ENTORNO INTERNACIONAL'),('B3','ANEXO I. GRÁFICO 2B. EVOLUCIÓN DEL NÚMERO DE HIJOS POR MUJER. ESPAÑA EN EL CONTEXTO EUROPEO'),('B19','Número de hijos por mujer')],unit='children_per_woman',factor=1,role='minimum_across_country_group_period_fertility_icf',category_semantics='observation_calendar_year_as_labelled',pdf=(80,['ÍNDICE COYUNTURAL DE FECUNDIDAD','ANEXO I. GRÁFICO 2'])),
 ('2025','Gráfico 48'):dict(title='TASA DE GENEROSIDAD',headers=[('B2','GRÁFICO 48. '),('C2','TASA DE GENEROSIDAD'),('C8','AIReF 2025')],unit='percent_pension_average_over_average_wage',factor=1,role='pension_generosity_contributory_ss_and_clases_pasivas_excluding_noncontributory',category_semantics='observation_calendar_year_as_labelled',pdf=(99,['TASA DE GENEROSIDAD','pensión media','salario medio','contributivas','Clases Pasivas'])),
 ('2025','Gráfico 28B'):dict(title='TASA DE ENTRADAS AL EMPLEO Y TASA DE SALIDAS AL NO EMPLEO. COMPARATIVA 2018-2019 Y 2022-2023. TASA DE SALIDA DEL EMPLEO AL NO EMPLEO. SEGÚN DURACIÓN DEL EMPLEO',headers=[('G4','TASA DE ENTRADAS AL EMPLEO Y TASA DE SALIDAS AL NO EMPLEO. COMPARATIVA 2018-2019 Y 2022-2023'),('G5','TASA DE SALIDA DEL EMPLEO AL NO EMPLEO. SEGÚN DURACIÓN DEL EMPLEO')],unit='percent_monthly_employment_exit_over_previous_month_employed',factor=1,role='monthly_employment_exit_rate_as_labelled',category_semantics='duration_of_employment_months_as_labelled',pdf=(65,['TASA DE SALIDA DEL EMPLEO AL NO','tasas calculadas a nivel mensual','total de trabajadores empleados en el mes t-1'])),
 ('2025','Gráfico 22'):dict(title=GR22,headers=[('F4',GR22)],unit='percent_population_age_16_74_activity_rate',factor=100,role='activity_rate_age16_74_as_labelled',category_semantics='observation_calendar_year_as_labelled',pdf=None)}
NAMES={
 ('2026','Anexo I. Gr 2A','chart49_s3'):dict(headers=[('F20','Países emisores'),('G21','Mínimo')],merge='F20:G20',val="'Anexo I. Gr 2A'!$G$22:$G$95",cat="'Anexo I. Gr 2A'!$B$22:$B$95",fmt='0'),
 ('2026','Anexo I. Gr 2A','chart49_s4'):dict(headers=[('D20','Mundial'),('E21','Mínimo')],merge='D20:E20',val="'Anexo I. Gr 2A'!$E$22:$E$95",cat="'Anexo I. Gr 2A'!$B$22:$B$95",fmt='0'),
 ('2026','Anexo I. Gr 2B','chart50_s2'):dict(headers=[('E20','Mínimo UE')],merge=None,val="'Anexo I. Gr 2B'!$E$21:$E$95",cat="'Anexo I. Gr 2B'!$B$21:$B$95",fmt='#,##0.0'),
 ('2025','Gráfico 22','chart43_s1'):dict(headers=[('F4',GR22)],merge=None,val="'Gráfico 22'!$C$5:$C$64",cat="'Gráfico 22'!$B$5:$B$64",fmt='0.0%')}
CLIMATE_PDF=ROOT/'evaluation/blocked_group_semantics_2026-10-31/evidence/climate_DT5_2025.pdf'
CLIMATE_SHA='de85002efac3d0bf1d8698b123506a02376917be1bbd2e006f131f71f6e20f26'
CLIMATE_URL='https://www.airef.es/wp-content/uploads/2025/04/Documentos-tecnicos-segunda-opinion-sostenibilidad-a-largo-plazo-AAPP/AIReF.-DT-Implicaciones-economicas-del-cambio-climatico.pdf'
for block,title,page,caption,unit in [('Gráfico 100','ANOMALÍA CLIMÁTICA (AUMENTO DE TEMPERATURAS)',150,'ANOMALÍA CLIMATICA (AUMENTO DE TEMPERATURAS)','anomaly_degrees_celsius_global_vs_preindustrial'),('Gráfico 124','INVERSIÓN PÚBLICA SOBRE PIB',161,'INVERSIÓN PÚBLICA SOBRE PIB','percent_gdp')]:
 SPECS['2025',block]=dict(title=title,headers=[('B3',block+'. '+title)],unit=unit,factor=1,role='climate_scenario_quantity_as_labelled',category_semantics='source_calendar_year_without_historical_forecast_inference',pdf=(page,[caption]),external=True)
def dumps(x):return json.dumps(x,ensure_ascii=False,sort_keys=True,default=str)
class BlockedResolver(SourceResolver):
 def __init__(self):
  super().__init__();self.external_pages={}
 def external(self,block):
  require(sha(CLIMATE_PDF)==CLIMATE_SHA,'Climate technical PDF changed');pages={4:['Opinión','Oxford'],14:['grados centígrados','temperatura media anual mundial','preindustrial']} if block=='Gráfico 100' else {4:['Opinión','Oxford'],33:['INVERSIÓN PÚBLICA (% PIB)']}
  missing=set(pages)-self.external_pages.keys()
  if missing:
   reader=PdfReader(CLIMATE_PDF)
   try:
    for number in missing:self.external_pages[number]=' '.join(reader.pages[number-1].extract_text().split())
   finally:reader.close()
  for number,fragments in pages.items():require(all(canon(t) in canon(self.external_pages[number]) for t in fragments),'Climate technical PDF evidence missing')
  return dict(path=str(CLIMATE_PDF.relative_to(ROOT)),url=CLIMATE_URL,sha256=CLIMATE_SHA,pages=pages)
 def profile(self,row):
  p=SPECS[row['book'],row['bloque']];source=self.sources[row['book']]
  require(row['titulo']==row['metrica_literal']==p['title'],'Blocked universal metric changed')
  require(all(source.text(row['bloque'],a)==t for a,t in p['headers']),'Blocked source caption changed');return p
 def binding(self,row):
  p=self.profile(row);key=row['book'],row['bloque'],row['serie_id'];n=NAMES.get(key)
  if n is None:
   b=super().binding(row)
   if row['bloque'] in ['Gráfico 100','Gráfico 124']:
    sid=re.fullmatch(r'chart(156|157|199|200)_s([1-7])',row['serie_id'] or '');require(sid is not None,'Unreviewed climate series');chart=int(sid[1]);index=int(sid[2]);require(index!=4,'Climate delayed transition binding not reviewed');require(chart in ({156,157} if row['bloque']=='Gráfico 100' else {199,200}),'Climate chart identity changed');col={1:'H',2:'I',3:'G',5:'E',6:'D',7:'C'}[index];sheet=row['bloque']
    require(b['name_ref']==f"'{sheet}'!${col}$5" and b['value_ref']==f"'{sheet}'!${col}$6:${col}$46" and b['category_ref']==f"'{sheet}'!$B$6:$B$46" and b['axis_format']==('#,##0.0' if chart==157 else '#,##0') and b['axis_id']=='1659750944','Climate source binding changed');require(b['categories']==list(map(str,range(2010,2051))) and b['multiplier']==b['divisor']==1 and b['display_unit']=='none','Climate category/scale changed');return b
   if row['bloque']=='Gráfico 48':
    require(row['serie_id']=='chart85_s2' and b['name_ref']=="'Gráfico 48'!$C$8" and b['value_ref']=="'Gráfico 48'!$D$8:$BG$8" and b['category_ref']=="'Gráfico 48'!$D$5:$BG$5" and b['axis_format']=='#,##0','Generosity row/axis conflict')
   else:
    col={'chart53_s1':'C','chart53_s2':'D'}[row['serie_id']];require(b['name_ref']==f"'Gráfico 28B'!${col}$4" and b['value_ref']==f"'Gráfico 28B'!${col}$5:${col}$40" and b['category_ref']=="'Gráfico 28B'!$B$5:$B$40" and b['axis_format']=='#,##0','Employment exit binding changed')
   require(b['divisor']==1 and b['multiplier']==p['factor'] and b['display_unit']=='none','Blocked axis scaling changed');return b
  label=' | '.join(t for a,t in n['headers']);require(row['serie_literal'] in [None,label],'Originally unnamed series changed');source=self.sources[row['book']];sheet=row['bloque'];part='xl/charts/'+row['serie_id'].split('_')[0]+'.xml';require(row['parte_grafico_imagen']==part,'Blocked chart part changed')
  root=ET.fromstring(source.xml.parts[part]);node=root.findall('.//c:ser',C)[int(row['serie_id'].split('_s')[1])-1]
  if sheet=='Gráfico 22':
   require(len(root.findall('.//c:ser',C))==1 and node.find('c:tx/c:strRef/c:f',C).text=="'Gráfico 22'!$C$4" and source.text(sheet,'C4') is None,'Sole activity series changed');cache=node.find('c:tx/c:strRef/c:strCache',C);require(cache is not None and not points(cache),'Sole activity series name cache changed')
  else:
   literal=node.find('c:tx/c:v',C);require(node.find('c:tx/c:strRef',C) is None and literal is not None and literal.text is None,'Minimum series name origin changed')
  require(all(source.text(sheet,a)==t for a,t in n['headers']),'Recovered physical name changed')
  if n['merge']:
   merges=[x.get('ref') for x in ET.fromstring(source.xml.parts[source.xml.sheets[sheet]]).findall('m:mergeCells/m:mergeCell',M)];require(n['merge'] in merges,'Recovered hierarchy merge missing')
  label=' | '.join(t for a,t in n['headers']);vr=node.find('c:val/c:numRef/c:f',C);cr=node.find('c:cat/c:numRef/c:f',C)
  require(vr is not None and cr is not None and vr.text==n['val'] and cr.text==n['cat'],'Recovered vector references changed');vs,vc=ref_cells(vr.text);cs,cc=ref_cells(cr.text)
  require(vs==cs==sheet and len(vc)==len(cc),'Recovered source vector shape changed');categories=[self.category(source,cs,a) for a in cc];require(all(isinstance(v,str) and v.strip() for v in categories),'Recovered category missing')
  vp=points(node.find('c:val/c:numRef/c:numCache',C));cp=points(node.find('c:cat/c:numRef/c:numCache',C));require(set(vp)==set(cp)==set(range(len(vc))),'Recovered cache vector incomplete')
  for i,(v,c) in enumerate(zip(vc,cc)):require(Decimal(vp[i])==source.number(sheet,v) and Decimal(cp[i])==source.number(sheet,c),'Recovered source/cache mismatch')
  parent=next(x for x in root.iter() if node in list(x));ids={x.get('val') for x in parent.findall('c:axId',C)};axes=[x for x in root.findall('.//c:valAx',C) if x.find('c:axId',C).get('val') in ids];require(len(axes)==1,'Recovered associated axis ambiguous');axis=axes[0];fmt=axis.find('c:numFmt',C)
  require(fmt is not None and fmt.get('formatCode')==n['fmt'] and axis.find('c:dispUnits',C) is None,'Recovered axis changed');div,mult=format_effect(n['fmt']);require(div==1 and mult==p['factor'],'Recovered axis scale changed')
  return dict(label=label,name_ref='source_headers:'+dumps(n['headers']),category_ref=cr.text,value_ref=vr.text,value_sheet=sheet,cells=vc,categories=categories,valid_indices=set(range(len(vc))),axis_id=axis.find('c:axId',C).get('val'),axis_format=n['fmt'],display_unit='none',divisor=div,multiplier=mult,mode='bounded_physical_header_recovery',category_basis='source_cells',name_basis='sole_series_metric_caption' if sheet=='Gráfico 22' else 'explicit_minimum_header_with_country_group')
 def title_proof(self,row,texts):return self.profile(row)['headers']
 def unit_spec(self,row,texts,binding):
  p=self.profile(row);require(binding['multiplier']==p['factor'] and binding['divisor']==1,'Blocked unit scale changed');evidence=dict(headers=p['headers'],unit=p['unit'],role=p['role'],category_semantics=p['category_semantics'],axis_id=binding['axis_id'],axis_format=binding['axis_format'])
  if p['pdf']:evidence['pdf']=self.pdf(row['book'],*p['pdf'])
  if p.get('external'):evidence['technical_pdf']=self.external(row['bloque'])
  self.supplements[row['book'],row['bloque']]=evidence
  return 'unidad explícita fuente y denominador PDF',p['unit'],'bounded_source_group_contract',('source_group_contract',dumps(evidence))
 def review_blocked(self,row):
  try:
   require((row['book'],row['bloque']) in SPECS,'Unreviewed blocked group');p=self.profile(row)
   if row['bloque']=='Gráfico 48' and row['serie_id']!='chart85_s2':return 'literal_series_source_conflict' if row['serie_id']=='chart85_s1' else 'unnamed_historical_series_contract_pending',None
   if (row['book'],row['bloque'],row['serie_id']) in NAMES:require(row['serie_literal'] is None,'Universal unnamed series no longer null')
   b=self.binding(row);shadow=dict(row,serie_literal=b['label']);status,result=SourceResolver.review(self,shadow)
   if result is None:return status,None
   evidence=dict(self.supplements[row['book'],row['bloque']],name_profile=NAMES.get((row['book'],row['bloque'],row['serie_id'])),universal_series=row['serie_literal'],resolved_series=b['label'],universal_category=row['periodo_categoria_literal'])
   return 'usable_blocked',result+(row['periodo_categoria_literal'],dumps(evidence),row['serie_literal'],row['hoja_valor'],row['celda_valor'],'{}',p['role'],p['category_semantics'],dumps(evidence))
  except (ValueError,ArithmeticError,KeyError,IndexError,AttributeError,TypeError,StopIteration) as e:return 'blocked_evidence_review:'+str(e),None

def collect(c):
 require(sha(PREVIOUS)==PREVIOUS_SHA,'Previous database changed');require(c.execute('SELECT count(*) FROM ready_usable_keys').fetchone()[0]==38492,'Previous admissions changed')
 q=c.execute('SELECT * FROM ready_numeric_queue ORDER BY book,row_number');names=[n[0] for n in q.description];data=[dict(zip(names,v)) for v in q.fetchall()];require(len(data)==8721,'Previous queue changed');r=BlockedResolver();rows=[];reviews=[]
 try:
  for row in data:
   if (row['book'],row['bloque']) not in SPECS:continue
   status,result=r.review_blocked(row);reviews.append((row['book'],row['row_number'],status))
   if result is not None:rows.append(result)
  pages=[(b,p,PDF_HASHES[b],text) for (b,p),text in sorted(r.pdf_pages.items())]
  external=[(str(CLIMATE_PDF.relative_to(ROOT)),CLIMATE_URL,CLIMATE_SHA,page,text) for page,text in sorted(r.external_pages.items())]
 finally:r.close()
 conflicts=[(b,n,dumps(dict(name_ref="'Gráfico 48'!$C$6",name='Histórico',value_ref="'Gráfico 48'!$D$8:$BG$8",value_row_header_cell='C8',value_row_label='AIReF 2025',action='retain_pending_without_relabelling'))) for b,n,status in reviews if status=='literal_series_source_conflict']
 return rows,reviews,pages,external,conflicts

def expressions():return [('blocked_usable_keys','SELECT * FROM ready_usable_keys UNION ALL SELECT book,row_number FROM blocked_observations'),('blocked_numeric_queue',"SELECT s.*,CASE WHEN r.status='literal_series_source_conflict' THEN r.status ELSE s.ready_reason END blocked_reason FROM ready_numeric_queue s LEFT JOIN blocked_review r USING(book,row_number) WHERE COALESCE(r.status,'')<>'usable_blocked'")]
def build(output):
 output=Path(output).resolve();require(output.is_relative_to(ROOT) and not output.exists(),'Output must be new project directory');require(sha(PREVIOUS)==PREVIOUS_SHA,'Previous database changed');output.mkdir(parents=True);db=output/'blocked_groups.duckdb';shutil.copyfile(PREVIOUS,db)
 with duckdb.connect(str(db)) as c:
  rows,reviews,pages,external,conflicts=collect(c);c.execute('CREATE TABLE blocked_observations AS SELECT * FROM ready_observations WHERE false');c.execute('ALTER TABLE blocked_observations ADD PRIMARY KEY(book,row_number)')
  if rows:c.executemany('INSERT INTO blocked_observations VALUES ('+','.join('?' for _ in FIELDS)+')',rows)
  c.execute('CREATE TABLE blocked_review(book VARCHAR,row_number INTEGER,status VARCHAR,PRIMARY KEY(book,row_number))');c.executemany('INSERT INTO blocked_review VALUES (?,?,?)',reviews)
  c.execute('CREATE TABLE blocked_pdf_evidence(book VARCHAR,page INTEGER,pdf_sha256 VARCHAR,text VARCHAR,PRIMARY KEY(book,page))')
  if pages:c.executemany('INSERT INTO blocked_pdf_evidence VALUES (?,?,?,?)',pages)
  c.execute('CREATE TABLE blocked_external_evidence(path VARCHAR,url VARCHAR,sha256 VARCHAR,page INTEGER,text VARCHAR,PRIMARY KEY(path,page))')
  if external:c.executemany('INSERT INTO blocked_external_evidence VALUES (?,?,?,?,?)',external)
  c.execute('CREATE TABLE blocked_binding_evidence(book VARCHAR,row_number INTEGER,evidence VARCHAR,PRIMARY KEY(book,row_number))')
  if conflicts:c.executemany('INSERT INTO blocked_binding_evidence VALUES (?,?,?)',conflicts)
  for name,sql in expressions():c.execute('CREATE VIEW '+name+' AS '+sql)
  sources=c.execute('SELECT * FROM sources ORDER BY book').fetchall()
 report=dict(status='built_pending_validation',added=len(rows),admitted=38492+len(rows),numeric_pending=8721-len(rows),previous_sha256=PREVIOUS_SHA,code_sha256=sha(Path(__file__)),db_sha256=sha(db),sources=sources,external_source=dict(path=str(CLIMATE_PDF.relative_to(ROOT)),url=CLIMATE_URL,sha256=CLIMATE_SHA),rag_integrated=False)
 (output/'manifest.json').write_text(dumps(report),encoding='utf-8');return report

def check_new(c):
 rows,reviews,pages,external,conflicts=collect(c)
 for table,expected in [('blocked_observations',rows),('blocked_review',reviews),('blocked_pdf_evidence',pages),('blocked_external_evidence',external),('blocked_binding_evidence',conflicts)]:
  actual=c.execute('SELECT * FROM '+table).fetchall();require(len(actual)==len(expected) and set(actual)==set(expected),'Blocked evidence mismatch: '+table)
 for name,sql in expressions():
  require(c.execute('SELECT * FROM '+name+' LIMIT 0').description==c.execute(sql+' LIMIT 0').description,'Blocked view columns changed')
  for a,b in [(name,'('+sql+')'),('('+sql+')',name)]:require(not c.execute('SELECT * FROM '+a+' EXCEPT ALL SELECT * FROM '+b).fetchall(),'Blocked view multiplicity changed')
 require(c.execute('SELECT count(*) FROM blocked_usable_keys').fetchone()[0]==38492+len(rows) and c.execute('SELECT count(*) FROM blocked_numeric_queue').fetchone()[0]==8721-len(rows),'Blocked counts changed')
 require(not c.execute('SELECT book,row_number FROM blocked_usable_keys INTERSECT SELECT book,row_number FROM blocked_numeric_queue').fetchall(),'Blocked partition overlap');require(c.execute('SELECT count(*) FROM (SELECT book,row_number FROM blocked_usable_keys UNION SELECT book,row_number FROM blocked_numeric_queue)').fetchone()[0]==47213,'Blocked candidate partition changed')
 require(c.execute('SELECT count(*) FROM (SELECT book,row_number FROM blocked_usable_keys GROUP BY ALL HAVING count(*)<>1)').fetchone()[0]==0,'Blocked admission collision');require(not c.execute('SELECT book,row_number FROM ready_quarantine INTERSECT SELECT book,row_number FROM blocked_usable_keys').fetchall(),'Quarantine admitted')
 return rows

def validate(db):
 previous=validate_previous(db)
 with duckdb.connect(str(db),read_only=True) as c:
  rows=check_new(c);stats=c.execute('SELECT book,block,unit,quantity_role,count(*) FROM blocked_observations GROUP BY ALL ORDER BY book,block').fetchall();reviews=c.execute("SELECT status,count(*) FROM blocked_review WHERE status<>'usable_blocked' GROUP BY ALL").fetchall()
 return dict(status='pass',previous_validation=previous['status'],added=len(rows),admitted=38492+len(rows),numeric_pending=8721-len(rows),stats=stats,blocked_review=reviews,previous_sha256=PREVIOUS_SHA,db_sha256=sha(Path(db)),code_sha256=sha(Path(__file__)),external_source_sha256=CLIMATE_SHA,rag_integrated=False)

def query(db,*,book,block,series_id,point_index):
 require(book in ['2025','2026'] and all(isinstance(x,str) and x.strip() for x in [block,series_id]) and type(point_index) is int and point_index>=0,'Exact blocked identity required')
 with duckdb.connect(str(db),read_only=True) as c:
  q=c.execute('SELECT t.*,s.xlsx_path,s.xlsx_sha256,s.pdf_path,s.pdf_sha256 FROM blocked_observations t JOIN blocked_usable_keys k USING(book,row_number) JOIN sources s USING(book) WHERE book=? AND block=? AND series_id=? AND point_index=?',[book,block,series_id,point_index]);names=[n[0] for n in q.description];rows=q.fetchall()
 if len(rows)!=1:raise LookupError('No unique blocked point; no fallback')
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
