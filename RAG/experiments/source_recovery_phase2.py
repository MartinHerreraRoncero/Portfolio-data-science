"""Versioned typed range/calendar recovery and independently bound physical pension facts."""
import argparse,json,re,shutil,xml.etree.ElementTree as ET
from decimal import Decimal as D
from pathlib import Path
import duckdb
from pypdf import PdfReader
from experiments.source_recovery import ROOT,RecoveryResolver,FIELDS,C,require,read,dumps,digest,aid,sha,validate as validate_previous
from experiments.source_recovery import query_observation as query_previous
from experiments.excel_pilot import ExcelXML
from experiments.bulk_semantics import ref_cells
from experiments.indexed_semantics import points
BASE=ROOT/'evaluation/source_recovery_2026-11-01/run_v5/source_recovery.duckdb'
BASE_SHA='52ba5a93f3ca0948ca4b43703c7a428358234f2a28386a8e8c8c32c9d4594ae5'
ASSETS=ROOT/'evaluation/source_recovery_phase2_2026-11-01/assets'
ASSET_SHA='5bae9793855728be1f23c05a03df1eda1c1b015b1b71d71ca98e8ae8c577589d'

def asset(name):return read(ASSETS/name)
def guards(c):
 require(sha(BASE)==BASE_SHA,'Certified phase1 changed');require(sha(ASSETS/'manifest.json')==ASSET_SHA,'Phase2 asset manifest changed')
 require(sha(ROOT/'experiments/source_recovery.py')=='b887f1d637d8241b8114d7521418e34c589dd7b5408b77332e65a66c1ad0374d','Certified phase1 production code changed')
 for p in asset('manifest.json'):require(sha(ASSETS/p['file'])==p['sha256'],'Phase2 evidence changed: '+p['file'])
 for b,x,h,p,ph in c.execute('SELECT * FROM sources').fetchall():require(sha(ROOT/x)==h and sha(ROOT/p)==ph,'Original source changed: '+b)

def xml_signature(s):
 n=ET.fromstring(s)if isinstance(s,str)else s
 return(n.tag,tuple(sorted(n.attrib.items())),(n.text or'').strip(),tuple(xml_signature(c)for c in n))

def statistic(data,**coordinates):
 require(set(coordinates)==set(data['id']),'Incomplete official dimension selector');index=0
 for dim,size in zip(data['id'],data['size']):index=index*size+data['dimension'][dim]['category']['index'][coordinates[dim]]
 return D(str(data['value'][str(index)]))

def primary_checks(r):
 population=(ASSETS/'eurostat_proj25_population_metadata.html').read_text(encoding='utf8');require('Statistical population is total population on 1 January'in population,'Primary population total scope absent')
 api=asset('proj_25naasfr_ES_2025_2050.json');require(api['id']==['freq','age','projection','unit','geo','time'],'Official fertility dimension order changed')
 for p in r.fertility['rows']:
  if p['source_role']=='eurostat_baseline_projection':
   official=statistic(api,freq='A',age='TOTAL',projection='BSL',unit='NR',geo='ES',time=str(p['year']));require(float(official)==float(D(p['raw_exact'])),'Official total baseline fertility comparator differs')
 ine=asset('ine_fertility_1441_last3.json');total=[x for x in ine if x['COD']=='IDB72160'and x['Nombre']=='Fecundidad. Total Nacional. Ambas nacionalidades.'];require(len(total)==1 and total[0]['T3_Unidad']=='Hijos por mujer','INE total-nationalities scope changed');anchor=next(x for x in total[0]['Data']if x['Anyo']==2024);require(D(str(anchor['Valor']))==D('1.10')and D(r.fertility['rows'][0]['raw_exact']).quantize(D('.01'))==D('1.10'),'Observed2024 total fertility anchor changed')
 debt=asset('eurostat_gov10q_ggdebt_ES.json');require(debt['id']==['freq','na_item','sector','unit','geo','time'],'Official debt dimensions changed');dates=debt['dimension']['time']['category']['index'];ordered=sorted(dates,key=dates.get);require(len(ordered)==52 and ordered[0]=='2012-Q1'and ordered[-1]=='2024-Q4','Explicit official reference calendar changed');reference=[statistic(debt,freq='Q',na_item='GD',sector='S13',unit='PC_GDP',geo='ES',time=t)for t in ordered]
 mapping=asset('gr61_quarter_reference_map.json');source=[];s=r.sources['2025'];require(len(mapping)==49,'Quarter map domain changed')
 for j,p in enumerate(mapping):
  raw=s.number('Gráfico 61',p['source_cell']);require(raw is not None and p['reference_quarter']==ordered[j+3],'Quarter map not aligned with explicit official dates');source.append(raw*100)
  if p['annual_reference_cell']:
   row=re.search('[0-9]+',p['source_cell'])[0];require(str(s.text('Gráfico 61','B'+row))==p['reference_quarter'][:4]and p['reference_quarter'].endswith('Q4')and s.number('Gráfico 61',p['annual_reference_cell'])==raw,'Quarter physical annual anchor differs')
 scores=[sum(abs(value-reference[j+lag])for j,value in enumerate(source))/49 for lag in range(4)];require(min(range(4),key=lambda k:scores[k])==3 and scores[3]<D('.06')and min(scores[:3])>D('1.8'),'Official quarterly vector alignment is not uniquely supported');require(sum(a.quantize(D('.1'))==b.quantize(D('.1'))for a,b in zip(source,reference[3:]))==41,'Quarterly reference agreement changed');return dict(fertility_baseline_same_ieee754_values=26,ine_total_observed_anchor='1.10',quarterly_unique_alignment='2012Q4–2024Q4',quarterly_mean_absolute_reference_difference=str(scores[3]),reference_differences_do_not_replace_source_values=True)

class Resolver(RecoveryResolver):
 def __init__(self):
  super().__init__();self.ranges=asset('range_geometry_recovery_proposal.json');self.fertility=asset('anexo7A_recovery_proposal.json');self.debt=asset('gr61_recovery_proposal.json');self.range_pairs={};self.overlays={}
  self.range_lookup={(p['book'],p['row_number']):(f,p)for f in self.ranges['profiles']for p in f['points']}
  self.fertility_lookup={(p['book'],p['row_number']):p for p in self.fertility['rows']};self.debt_lookup={(p['book'],p['row_number']):p for p in self.debt['points']}
  self.formats={}
 def result(self,row,raw,value,unit,metric,series,category,role,proof,**kw):
  key=row['book'],row['hoja_valor']
  if key not in self.formats:self.formats[key]={c.coordinate:c.number_format for rr in self.sources[key[0]].workbook[key[1]].iter_rows()for c in rr if c.value is not None}
  d=dict.fromkeys(FIELDS);d.update(book=row['book'],row_number=row['row_number'],record_id=row['id'],block=row['bloque'],metric_literal=metric,series_id=row['serie_id'],series_literal=series,category_literal=category,unit=unit,value=value,source_sheet=row['hoja_valor'],source_cell=row['celda_valor'],chart_part=row['parte_grafico_imagen'],rule_id='recovery2_typed_primary_bound_v1',raw_value=raw,cell_format=self.formats[key].get(row['celda_valor']),display_unit='none',divisor=D(1),multiplier=D(1),scope_literal=row['metrica_literal'],unit_basis='pinned_primary_reference_and_physical_source',metric_evidence=dumps(proof),semantic_scope='source_snapshot_typed_quantity',title_resolution='bounded_physical_reference',supplemental_evidence=dumps(proof),universal_category=row['periodo_categoria_literal'],grid_evidence=dumps(proof),universal_series=row['serie_literal'],universal_value_sheet=row['hoja_valor'],universal_value_cell=row['celda_valor'],reconciliation_evidence='{}',quantity_role=role,category_semantics='explicit_source_period',quantity_evidence=dumps(proof));d.update(kw);return tuple(d[n]for n in FIELDS)
 def overlay(self,key,**values):self.overlays.setdefault(key,{}).update(values)
 def check_raw(self,row,p,cell,raw):
  require(row['celda_valor']==cell and raw==D(p.get('raw_exact',p.get('original_raw_exact'))),'Phase2 physical raw identity changed')
 def range(self,row,c):
  f,p=self.range_lookup[row['book'],row['row_number']];s,sh,root,node,i=self.ser(row);vr,cell,raw=self.scalar(row);self.check_raw(row,p,p['original_source_cell'],raw)
  require(row['bloque']==f['block']and row['parte_grafico_imagen']==f['chart_part'],'Range figure identity changed')
  for addr,literal in f['central_context']:
   actual=s.text(sh,addr);require(actual==literal or isinstance(literal,(int,float))and actual is not None and D(str(actual))==D(str(literal)),'Range physical context changed')
  area=next((n for n in root.findall('.//c:areaChart',C)if node in list(n)),None);require(area is not None and area.find('c:grouping',C).get('val')=='standard','Range is not standard area geometry')
  for role,ref,fill in[('upper',f['upper_ref'],f['upper_fill_xml']),('lower',f['lower_ref'],f['lower_fill_xml'])]:
   edge=next((n for n in area.findall('c:ser',C)if n.find('c:val/c:numRef/c:f',C)is not None and n.find('c:val/c:numRef/c:f',C).text==ref),None)
   require(edge is not None and xml_signature(edge.find('c:spPr',C))==xml_signature(fill),'Range endpoint source/style changed')
   ss,cells=ref_cells(ref);cache=points(edge.find('c:val/c:numRef/c:numCache',C));physical={j:s.number(ss,a)for j,a in enumerate(cells)if s.text(ss,a)is not None};require(set(cache)==set(physical)and all(D(cache[j])==v for j,v in physical.items()),'Complete range endpoint cache/physical vector differs')
   require(s.number(sh,p[role+'_cell'])==D(p[role+'_raw']),'Range physical endpoint changed')
  lower,central,upper=[D(p[x+'_raw'])*D(f['multiplier'])/D(f['divisor'])for x in['lower','central','upper']]
  require(s.number(sh,p['central_cell'])==D(p['central_raw'])and lower<=central<=upper,'Range centre containment changed')
  ck=p['existing_central_key'];central_pub=c.execute('SELECT value,unit,category_literal,source_cell FROM source_recovery_public_observations WHERE book=?AND row_number=?',[ck['book'],ck['row_number']]).fetchone()
  require(central_pub and central_pub[0]==central and central_pub[1]==f['unit']and central_pub[2]==str(p['year'])and central_pub[3]==p['central_cell'],'Range independently admitted centre differs')
  _,year=self.categorical(row);require(year==str(p['year']),'Range explicit source year differs')
  role='range_lower_edge'if cell==p['lower_cell']else'range_upper_edge';require(cell==p['lower_cell']or cell==p['upper_cell'],'Unknown range endpoint')
  proof=dict(profile=f['block'],source_area_grouping='standard',graphical_role=role,lower_cell=p['lower_cell'],upper_cell=p['upper_cell'],central_cell=p['central_cell'],central_observation_id=aid(ck['book'],ck['row_number']),method=None,confidence_level=None,distribution=None,not_point_forecast=True,confidence_interpretation_allowed=False,original_series_literal=row['serie_literal'])
  pairid=f"range:{row['book']}:{f['chart_part']}:{p['lower_cell']}:{p['upper_cell']}"
  pairproof={k:v for k,v in proof.items()if k not in['graphical_role','original_series_literal']};pairproof['quantity_role']='depicted_range_geometry_not_point_forecast'
  pair=(pairid,row['book'],f['block'],str(p['year']),f['unit'],lower,central,upper,p['lower_cell'],p['central_cell'],p['upper_cell'],aid(ck['book'],ck['row_number']),dumps(pairproof))
  require(pairid not in self.range_pairs or self.range_pairs[pairid]==pair,'Conflicting duplicate range');self.range_pairs[pairid]=pair
  ups=c.execute('SELECT book,row_number FROM source_recovery_public_observations WHERE book=?AND block=?AND chart_part=?AND source_cell=?AND category_literal=?AND value=?AND unit=?',[row['book'],f['block'],f['chart_part'],p['upper_cell'],str(p['year']),upper,f['unit']]).fetchall()
  require(len(ups)<=1,'Ambiguous admitted upper endpoint')
  for key in ups:self.overlay(key,quantity_role='depicted_range_upper_edge_not_point_forecast',range_id=pairid,confidence_level=None,method=None,confidence_interpretation_allowed=False)
  if sh=='Anexo I. Gr 7A':proof.update(population_scope='resident_population_all_maternal_nationalities',wrong_original_population_header_preserved=True)
  return self.result(row,raw,lower if role=='range_lower_edge'else upper,f['unit'],p['central_metric_literal'],role,year,'depicted_'+role+'_not_point_forecast',proof,value_ref=vr,multiplier=D(f['multiplier']),divisor=D(f['divisor']),binding_mode='verified_standard_area_range_geometry')
 def fertility_point(self,row):
  p=self.fertility_lookup[row['book'],row['row_number']];s,sh,root,node,i=self.ser(row);vr,cell,raw=self.scalar(row);self.check_raw(row,p,p['source_cell'],raw);nr,label=self.named(row);yearcell='B'+re.search('[0-9]+',cell)[0];year=str(s.text(sh,yearcell));require(node.find('c:cat',C)is None and row['periodo_categoria_literal']is None,'Fertility null inherited category changed')
  refs=root.findall('.//c:ser/c:cat/c:numRef',C);require(refs,'Shared fertility chart axis missing')
  for ref in refs:
   ss,cells=ref_cells(ref.find('c:f',C).text);require(ss==sh and cells[i]==yearcell and D(points(ref.find('c:numCache',C))[i])==D(year),'Fertility shared axis/physical calendar differs')
  require(row['serie_id']==p['series_id']and year==str(p['year'])and s.text(sh,'B19')==p['bad_original_header_literal'],'Fertility original header/calendar changed')
  if 'external_source_cell'in p:require(self.sources['2025'].number(p['external_source_sheet'],p['external_source_cell'])==raw,'Fertility prior vintage exact comparator changed')
  proof=dict(p,population_override_evidence='bounded2026AnexoIGr7A_TotalPopulation_primary_reference',not_current_actual_when_projection=True,no_original_header_rewrite=True)
  return self.result(row,raw,raw,p['unit'],'Total resident fertility, all maternal nationalities',label,year,p['source_role'],proof,value_ref=vr,name_ref=nr,binding_mode='independently_validated_total_population_comparator',category_semantics='physical_year_and_shared_chart_axis_original_category_null')
 def quarterly(self,row):
  p=self.debt_lookup[row['book'],row['row_number']];s,sh,root,node,i=self.ser(row);vr,cell,raw=self.scalar(row);self.check_raw(row,p,p['source_cell'],raw)
  require(row['serie_id']==p['series_id']and row['serie_literal']==p['original_series_literal']and row['periodo_categoria_literal']==p['original_category_literal'],'Quarterly original metadata changed')
  mapping=asset('gr61_quarter_reference_map.json');refcell='D'+re.search('[0-9]+',cell)[0];m=next(v for v in mapping if v['source_cell']==refcell);require(m['reference_quarter']==p['resolved_quarter'],'Quarterly independent dated reference differs')
  require('TRIMESTRAL'in s.text(sh,'B2'),'Quarterly source frequency changed')
  if cell.startswith('C'):require(s.number(sh,refcell)==raw and m['annual_reference_cell']==cell,'Annual marker is not matching Q4 source')
  proof=dict(p,no_current_reference_value_replacement=True,no_cause_of_reference_difference_assumed=True,not_annual_average=True,calendar_proof=self.debt['calendar_basis'])
  return self.result(row,raw,raw*100,'percent_gdp','General government debt at quarter end',row['serie_literal'],p['resolved_quarter'],p['role'],proof,value_ref=vr,multiplier=D(100),binding_mode='uniquely_anchored_primary_quarter_calendar',category_semantics='quarter_end_not_annual_average')

def physical_facts(c,r,stage):
 p=asset('gr49_physical_recovery_proposal.json');dt=ExcelXML(ASSETS/'official_tir_graphs.xlsx');sheets={n:dt.read(n)for n in['Gráfico 16','Gráfico 18 y 19']};facts=[];retired=[]
 pdf=PdfReader(ASSETS/'official_tir_technical.pdf');t30=pdf.pages[29].extract_text();t39=pdf.pages[38].extract_text();require('2024'in t30 and 'constante'in t30 and '%'in t39 and 'IRP'in t39,'TIR primary constant scenario/unit absent')
 for a in p['points']:
  cell=a['source_cell'];col=re.match('[A-Z]+',cell)[0];raw=r.sources['2025'].number('Gráfico 49',cell);ds=sheets[a['dt_sheet']];require(raw==D(a['source_raw'])==ds[a['dt_cell']].number(),'Physical/independent TIR numeric mismatch');require(D(ds[col+'4'].value)==a['year']and D(str(r.sources['2025'].text('Gráfico 49',col+'20')))==a['year'],'Independent physical TIR year mismatch')
  old=stage['2025',a['old_corrupt_object_point_key'][0]];require(old['celda_valor']==cell and old['serie_id'].startswith('chart87_')and D(old['decimal_XML_texto'])==raw,'Corrupt object physical-cell lineage mismatch')
  ident=f'physical:airef:2025:Gr49:{cell}:lifeconstant2024';proof=dict(source_cell=cell,source_year=a['year'],dt_sheet=a['dt_sheet'],dt_cell=a['dt_cell'],dt_sha256=p['dt_sha256'],dt_pdf_sha256=p['dt_pdf_sha256'],scenario='constant_life_expectancy_at_2024_value',observation_kind=p['observation_kind'],original_chart_category=old['periodo_categoria_literal'],physical_cell_replacement_not_old_index_year_reinterpretation=True,raw_original_preserved=True,source_variant='physical_cells_verified_against_official_technical_document',population_scope=p['population_scope'])
  d=dict.fromkeys(FIELDS);d.update(book='2025',record_id=ident,block='Gráfico 49',metric_literal='Pension internal rate of return',series_literal=a['dt_label'],category_literal=str(a['year']),unit='percent_internal_rate_of_return',value=raw*100,source_sheet='Gráfico 49',source_cell=cell,rule_id='recovery2_independent_physical_tir_v1',raw_value=raw,multiplier=D(100),divisor=D(1),quantity_role='modelled_pension_return_scenario_not_observed',quantity_evidence=dumps(proof),universal_category=old['periodo_categoria_literal'],universal_series=old['serie_literal'],universal_value_sheet=old['hoja_valor'],universal_value_cell=cell,category_semantics='explicit_physical_and_independent_DT_year',binding_mode='physical_cell_new_identity_not_rehabilitated_chart_object',supplemental_evidence=dumps(proof));facts.append(tuple(d[n]for n in FIELDS)+(ident,));retired.append(('2025',old['row_number'],'replaced_corrupt_chart_calendar_by_new_physical_fact',ident,dumps(proof)))
 for a in asset('gr49_existing100_scenario_overlay.json')['points']:
  ds=sheets[a['dt_sheet']];require(r.sources['2025'].number('Gráfico 49',a['source_cell'])==D(a['raw'])==ds[a['dt_cell']].number(),'Existing TIR scenario independent match changed');col=re.match('[A-Z]+',a['source_cell'])[0];require(D(ds[col+'4'].value)==a['year'],'Existing TIR independent year changed');q=c.execute('SELECT source_cell,value,category_literal FROM source_recovery_public_observations WHERE book=?AND row_number=?',[a['book'],a['row_number']]).fetchone();require(q and q[0]==a['source_cell']and q[1]==D(a['raw'])*100 and q[2]==str(a['year']),'Existing TIR public identity changed');r.overlay((a['book'],a['row_number']),scenario='life_expectancy_evolving_according_to_AIR_eF_demographic_projection',scenario_label='EV según proyección AIReF',quantity_role='modelled_pension_return_scenario_not_observed',not_observed=True,independent_dt_cell=a['dt_cell'],independent_dt_sheet=a['dt_sheet'])
 require(len(facts)==100 and len(retired)==100,'Physical TIR domain changed');return facts,retired

def orphan_registry(c,r,stage):
 retired=[];coverage=[]
 for f in asset('orphan24_publication_evidence.json')['profiles']:
  r.pdf('2025',f['pdf_page'],['GRÁFICO',str(f['figure'])])
  for a in f['canonical_published_points']:
   q=c.execute('SELECT series_id,series_literal,category_literal,unit,value,source_cell FROM source_recovery_public_observations WHERE book=\'2025\'AND row_number=?',[a['row_number']]).fetchone();require(q==tuple(a[k]if k!='value'else D(a[k])for k in['series_id','series_literal','category_literal','unit','value','source_cell']),'Orphan positive published coverage differs');coverage.append(('2025',f['figure'],a['row_number'],f['pdf_page'],dumps(a)))
  for a in f['corrupt_objects']:
   row=stage['2025',a['row_number']];require(all(row[k]==v for k,v in a['original'].items()if k in row),'Orphan original source metadata changed');s,sh,root,node,i=r.ser(row);require(xml_signature(node)==xml_signature(a['series_xml']),'Orphan XML changed');require(D(row['decimal_XML_texto'])==1,'Orphan cache sentinel differs');retired.append(('2025',a['row_number'],'retired_unpublished_damaged_object_with_positive_figure_coverage',f'published:2025:Gr{f["figure"]}',dumps(dict(figure=f['figure'],pdf_page=f['pdf_page'],canonical_points=len(f['canonical_published_points']),not_economic_value_one=True,not_pointwise_replacement=True))))
 require(len(retired)==24 and len(coverage)==161,'Orphan repair coverage domain changed');return retired,coverage

def collect(c):
 guards(c);q=c.execute('SELECT * FROM staging_records');cols=[a[0]for a in q.description];stage={(p['book'],p['row_number']):p for p in[dict(zip(cols,v))for v in q.fetchall()]};r=Resolver();rows=[];reviews=[]
 try:
  primary_checks(r)
  for a in asset('selection.json'):
   row=stage[a['book'],a['row_number']];require(digest(row)==a['payload_sha256'],'Authorized phase2 source payload changed')
   try:result=r.range(row,c)if a['family']=='range'else r.fertility_point(row)if a['family']=='fertility'else r.quarterly(row);rows.append(result);status='validated'
   except(ValueError,KeyError,TypeError,IndexError,StopIteration,AttributeError,ArithmeticError)as e:status='review:'+str(e)
   reviews.append((a['book'],a['row_number'],a['family'],status))
  for p in r.fertility['existing_public_metadata_overlays']:
   q=c.execute('SELECT source_cell,value FROM source_recovery_public_observations WHERE book=?AND row_number=?',[p['book'],p['row_number']]).fetchone();require(q==(p['source_cell'],D(p['canonical_value_preserved'])),'Existing fertility population overlay raw identity changed');r.overlay((p['book'],p['row_number']),population_scope=p['correct_population_scope'],wrong_original_population_header_preserved=True)
  facts,tirretired=physical_facts(c,r,stage);orphans,coverage=orphan_registry(c,r,stage)
  return dict(rows=rows,reviews=reviews,ranges=list(r.range_pairs.values()),overlays=[(k[0],k[1],dumps(v))for k,v in sorted(r.overlays.items())],facts=facts,retired=tirretired+orphans,coverage=coverage)
 finally:r.close()

def views():
 return[
 ('recovery2_usable_keys','SELECT * FROM source_recovery_usable_keys UNION ALL SELECT book,row_number FROM recovery2_observations'),
 ('recovery2_public_observations',"SELECT p.* REPLACE(COALESCE(json_extract_string(m.evidence,'$.quantity_role'),p.quantity_role)AS quantity_role,CASE WHEN m.evidence IS NULL THEN p.quantity_evidence ELSE CAST(json_merge_patch(COALESCE(p.quantity_evidence,'{}'),m.evidence)AS VARCHAR)END AS quantity_evidence),m.evidence recovery2_metadata_overlay FROM source_recovery_public_observations p LEFT JOIN recovery2_metadata_overlays m USING(book,row_number) UNION ALL BY NAME SELECT o.*,'recovery2_observations' source_layer,'airef:'||book||':'||row_number observation_id FROM recovery2_observations o UNION ALL BY NAME SELECT f.*,'recovery2_new_physical_facts' source_layer FROM recovery2_physical_facts f"),
 ('recovery2_resolution',"SELECT p.book,p.row_number,CASE WHEN o.book IS NOT NULL THEN 'admitted_validated'WHEN r.book IS NOT NULL THEN r.reason ELSE p.state END state,p.original_queue FROM source_recovery_resolution p LEFT JOIN recovery2_observations o USING(book,row_number)LEFT JOIN recovery2_retired r USING(book,row_number)"),
 ('recovery2_numeric_queue',"SELECT s.*,r.state recovery2_state FROM staging_records s JOIN source_recovery_numeric_queue p USING(book,row_number)LEFT JOIN recovery2_resolution r USING(book,row_number)WHERE NOT EXISTS(SELECT 1 FROM recovery2_usable_keys k WHERE k.book=s.book AND k.row_number=s.row_number)")]

def build(output):
 output=Path(output).resolve();require(output.is_relative_to(ROOT)and not output.exists(),'New output folder required');require(sha(BASE)==BASE_SHA,'Certified phase1 changed');output.mkdir(parents=True);db=output/'source_recovery_phase2.duckdb';shutil.copyfile(BASE,db)
 with duckdb.connect(str(db))as c:
  data=collect(c);c.execute('CREATE TABLE recovery2_observations AS SELECT * FROM source_recovery_observations WHERE FALSE');c.execute('ALTER TABLE recovery2_observations ADD PRIMARY KEY(book,row_number)');c.execute('CREATE TABLE recovery2_physical_facts AS SELECT o.*,NULL::VARCHAR observation_id FROM source_recovery_observations o WHERE FALSE');c.execute('ALTER TABLE recovery2_physical_facts ADD PRIMARY KEY(observation_id)')
  schemas=[('recovery2_review','book VARCHAR,row_number INTEGER,family VARCHAR,status VARCHAR','reviews','book,row_number'),('recovery2_ranges','range_id VARCHAR,book VARCHAR,block VARCHAR,year VARCHAR,unit VARCHAR,lower_value DECIMAL(38,24),central_value DECIMAL(38,24),upper_value DECIMAL(38,24),lower_cell VARCHAR,central_cell VARCHAR,upper_cell VARCHAR,central_observation_id VARCHAR,evidence VARCHAR','ranges','range_id'),('recovery2_metadata_overlays','book VARCHAR,row_number INTEGER,evidence VARCHAR','overlays','book,row_number'),('recovery2_retired','book VARCHAR,row_number INTEGER,reason VARCHAR,replacement_reference VARCHAR,evidence VARCHAR','retired','book,row_number'),('recovery2_publication_coverage','book VARCHAR,figure INTEGER,row_number INTEGER,pdf_page INTEGER,evidence VARCHAR','coverage','book,figure,row_number')]
  for name,schema,key,pk in schemas:c.execute('CREATE TABLE '+name+'('+schema+',PRIMARY KEY('+pk+'))')
  for name,key in[('recovery2_observations','rows'),('recovery2_physical_facts','facts')]+[(n,k)for n,s,k,p in schemas]:
   if data[key]:c.executemany('INSERT INTO '+name+' VALUES('+','.join('?'for _ in data[key][0])+')',data[key])
  for n,sql in views():c.execute('CREATE VIEW '+n+' AS '+sql)
  report=summary(c);report.update(status='built_pending_validation',review=c.execute("SELECT status,count(*)FROM recovery2_review WHERE status<>'validated'GROUP BY status").fetchall(),db_sha256=None)
 report['db_sha256']=sha(db);(output/'manifest.json').write_text(dumps(report),encoding='utf8');return report

def summary(c):
 return dict(added_original=c.execute('SELECT count(*)FROM recovery2_observations').fetchone()[0],new_physical_facts=c.execute('SELECT count(*)FROM recovery2_physical_facts').fetchone()[0],original_usable=c.execute('SELECT count(*)FROM recovery2_usable_keys').fetchone()[0],public_observations=c.execute('SELECT count(*)FROM recovery2_public_observations').fetchone()[0],raw_outside=c.execute('SELECT count(*)FROM recovery2_numeric_queue').fetchone()[0],range_pairs=c.execute('SELECT count(*)FROM recovery2_ranges').fetchone()[0],metadata_overlays=c.execute('SELECT count(*)FROM recovery2_metadata_overlays').fetchone()[0],resolution_counts=dict(c.execute('SELECT state,count(*)FROM recovery2_resolution GROUP BY state').fetchall()),previous_sha256=BASE_SHA,asset_manifest_sha256=ASSET_SHA,code_sha256=sha(Path(__file__)),full_user_goal_complete=False,rag_integrated=False)

def check_new(c):
 data=collect(c)
 for name,key in[('recovery2_observations','rows'),('recovery2_physical_facts','facts'),('recovery2_review','reviews'),('recovery2_ranges','ranges'),('recovery2_metadata_overlays','overlays'),('recovery2_retired','retired'),('recovery2_publication_coverage','coverage')]:
  actual=c.execute('SELECT * FROM '+name).fetchall();require(len(actual)==len(data[key])and set(actual)==set(data[key]),'Recomputed phase2 evidence differs: '+name)
 require(len(data['rows'])==775 and len(data['facts'])==100 and len(data['ranges'])==549 and len(data['retired'])==124,'Authorized phase2 domain not fully recovered')
 require(c.execute('SELECT count(*)FROM recovery2_usable_keys').fetchone()[0]==42732 and c.execute('SELECT count(*)FROM recovery2_public_observations').fetchone()[0]==42833,'Phase2 public counts differ')
 require(c.execute("SELECT count(*)FROM recovery2_resolution WHERE state='blocked'").fetchone()[0]==322,'Active unresolved domain differs')
 require(not c.execute('SELECT book,row_number FROM recovery2_retired INTERSECT SELECT * FROM recovery2_usable_keys').fetchall(),'Retired original points publicly re-admitted');require(not c.execute('SELECT observation_id FROM recovery2_public_observations GROUP BY observation_id HAVING count(*)<>1').fetchall(),'Phase2 duplicate public identities')
 for n,sql in views():require(not c.execute('SELECT * FROM '+n+' EXCEPT ALL ('+sql+')').fetchall()and not c.execute('('+sql+')EXCEPT ALL SELECT * FROM '+n).fetchall(),'Phase2 view changed: '+n)
 return summary(c)

def validate(db):
 db=Path(db)
 manifest=read(Path(db).parent/'manifest.json');require(manifest['code_sha256']==sha(Path(__file__))and manifest['db_sha256']==sha(Path(db))and manifest['asset_manifest_sha256']==ASSET_SHA,'Phase2 build code/data seal changed')
 previous=validate_previous(db)
 with duckdb.connect(str(db),read_only=True)as c:report=check_new(c)
 report.update(status='pass',previous_validation=previous['status'],db_sha256=sha(Path(db)));return report

def query_observation(db,*,observation_id):
 require(isinstance(observation_id,str)and observation_id.strip(),'Exact observation identity required')
 with duckdb.connect(str(db),read_only=True)as c:
  q=c.execute('SELECT * FROM recovery2_public_observations WHERE observation_id=?',[observation_id]);names=[x[0]for x in q.description];rows=q.fetchall()
  if len(rows)!=1:raise LookupError('No unique safe phase2 observation; no legacy fallback')
  result={n:str(v)if isinstance(v,D)else v for n,v in zip(names,rows[0])}
  previous=c.execute('SELECT count(*)FROM source_recovery_public_observations WHERE observation_id=?',[observation_id]).fetchone()[0]
  if result['book']is not None:
   source=c.execute('SELECT xlsx_path,xlsx_sha256,pdf_path,pdf_sha256 FROM sources WHERE book=?',[result['book']]).fetchone();result.update(source_xlsx_path=source[0],source_xlsx_sha256=source[1],source_report_pdf_path=source[2],source_report_pdf_sha256=source[3])
  if result['row_number']is not None:
   overlay=c.execute('SELECT evidence FROM recovery2_metadata_overlays WHERE book=?AND row_number=?',[result['book'],result['row_number']]).fetchone()
   if overlay:result['recovery2_metadata_overlay']=json.loads(overlay[0]);result.update(result['recovery2_metadata_overlay'])
 if previous:
  inherited=query_previous(db,observation_id=observation_id);inherited.update(result);result=inherited
 result['safe_recovery2_view']=True;return result

def query_safe(db,*,book,row_number):
 require(book in['2025','2026']and type(row_number)is int and row_number>0,'Exact source key required');return query_observation(db,observation_id=aid(book,row_number))

def query_range(db,*,range_id):
 with duckdb.connect(str(db),read_only=True)as c:
  q=c.execute('SELECT * FROM recovery2_ranges WHERE range_id=?',[range_id]);rows=q.fetchall();names=[x[0]for x in q.description]
  if len(rows)!=1:raise LookupError('Exact range identity required')
  d={n:str(v)if isinstance(v,D)else v for n,v in zip(names,rows[0])};d.update(confidence_level=None,method=None,not_point_forecast=True,confidence_interpretation_allowed=False);return d

def main():
 p=argparse.ArgumentParser();p.add_argument('command',choices=['build','validate']);p.add_argument('--output-dir');p.add_argument('--database');a=p.parse_args();result=build(a.output_dir)if a.command=='build'else validate(a.database)
 if a.command=='validate':(Path(a.database).parent/'validation.json').write_text(dumps(result),encoding='utf8')
 print(dumps(result))
if __name__=='__main__':main()

