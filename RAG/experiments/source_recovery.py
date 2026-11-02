"""Repair provenance, recover typed source snapshots and replace a documented Treasury typo. No RAG integration."""
import argparse,hashlib,json,re,shutil,xml.etree.ElementTree as ET
from decimal import Decimal
from pathlib import Path
import duckdb
from pypdf import PdfReader
from experiments.closure_family_semantics import FIELDS,validate as validate_previous
from experiments.reviewed_family_semantics import ReviewedResolver
from experiments.bulk_semantics import C,ref_cells
from experiments.indexed_semantics import points
from experiments.excel_pilot import ROOT,sha
from experiments.universal_database import require
PREVIOUS=ROOT/'evaluation/closure_family_semantics_2026-10-31/run_v3/closure_families.duckdb'
PREVIOUS_SHA='a7f4b526affb63a7b89275623937ebfdde52985920d39ee6e89ac4c658385627'
RESEARCH=ROOT/'evaluation/source_recovery_2026-11-01/research_writer'
AUDITOR=RESEARCH.parent/'research_auditor'
PRIMARY=RESEARCH/'primary_sources'
PLAN=RESEARCH/'recovery_phase1_plan.json'
PLAN_SHA='6543cc5eddf1e01b69f5b1da2045076961d92e952af1c74b9fd2f169e871a840'
SOURCES=RESEARCH/'recovery_primary_sources.json'
SOURCE_SHA='49517010a4c8cf7beb8e04c2ce5a12b99ec843551702987e950ce22d6234ce9b'
FISCAL=RESEARCH/'published_fiscal_figure_manifest.json'
FISCAL_SHA='7b1b1b19e87f0395537b3329cf86ddf487354de69e419e75f570898f244b6491'
FISCAL_BAD=RESEARCH/'nonpublication_fiscal_curve_objects3444.json'
FISCAL_BAD_SHA='262bd75fd0ba2f68d37f1a4d304e853563185ee375778386ff68da4e688d31b6'
TEMPERATURE_LINKS=RESEARCH/'canonical_temperature_links1722.json'
TEMPERATURE_LINKS_SHA='e6d239c575eff61144f1d2a4fae5903cc6ee1fde1c19a61fa2ee8b4fd3d934a5'
REFERENCE=RESEARCH/'recovery_reference_population_matches.json'
REFERENCE_SHA='abe9b99f332fce7c6d575fd02895e0f534dd4a2859ae35c01bb4f9e9e0733b73'
OFFICIAL_ID='official:tesoro:strategy2022:net-issuance:2021'
def dumps(v):return json.dumps(v,ensure_ascii=False,sort_keys=True,default=str)
def digest(v):return hashlib.sha256(dumps(v).encode()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def aid(book,row_number):return f'airef:{book}:{row_number}'
def text(v):return None if v is None else str(v)
def file_proofs(c):
 require(sha(PREVIOUS)==PREVIOUS_SHA,'Previous snapshot changed');require(sha(PLAN)==PLAN_SHA,'Authorized recovery plan changed');require(sha(SOURCES)==SOURCE_SHA,'Primary source manifest changed');require(sha(REFERENCE)==REFERENCE_SHA,'Frozen World Bank comparison changed');require(all(sha(p)==h for p,h in[(FISCAL,FISCAL_SHA),(FISCAL_BAD,FISCAL_BAD_SHA),(TEMPERATURE_LINKS,TEMPERATURE_LINKS_SHA)]),'Frozen figure/provenance manifest changed')
 for book,xlsx,xh,pdf,ph in c.execute('SELECT * FROM sources').fetchall():require(sha(ROOT/xlsx)==xh and sha(ROOT/pdf)==ph,'Original source changed: '+book)
 files=read(SOURCES)
 for f in files:require(sha(PRIMARY/f['file'])==f['sha256'],'Primary external source changed: '+f['file'])
 return files
class RecoveryResolver(ReviewedResolver):
 def __init__(self):
  super().__init__();self.plan=read(PLAN);self.families=self.plan['families'];self.paired=[];self.references={(a['row'],a['japan']):a for a in read(REFERENCE)};self.chart_cache={}
 def ser(self,row):
  sheet=row['hoja_valor'];s=self.sources[row['book']];part=row['parte_grafico_imagen'];m=re.fullmatch(r'chart([0-9]+)_s([0-9]+)',row['serie_id']or'');require(m and part=='xl/charts/chart'+m[1]+'.xml','Recovery series/chart identity changed');chart_key=(row['book'],part,s.xml.parts[part]);
  if chart_key not in self.chart_cache:self.chart_cache[chart_key]=ET.fromstring(chart_key[2])
  root=self.chart_cache[chart_key];node=root.findall('.//c:ser',C)[int(m[2])-1];index=re.fullmatch(re.escape(row['serie_id'])+r'_p([0-9]+)',row['id']or'');require(index,'Recovery source point identity changed');return s,sheet,root,node,int(index[1])
 def bound(self,row,container):
  s,sheet,root,node,i=self.ser(row);ref=node.find(container+'/c:numRef/c:f',C);require(ref is not None,'Expected numeric source reference missing');ss,cells=ref_cells(ref.text);require(ss==sheet and i<len(cells),'Foreign or incomplete source vector');all_cache=points(node.find(container+'/c:numRef/c:numCache',C));cache={j:v for j,v in all_cache.items()if v!='#N/A'};require(all(s.text(ss,cells[j])=='#N/A'for j,v in all_cache.items()if v=='#N/A'),'Cache NA/source error differs');physical={j:s.number(ss,a)for j,a in enumerate(cells)if s.text(ss,a)not in[None,'#N/A']};physical={j:v for j,v in physical.items()if v is not None};require(set(cache)==set(physical)and all(Decimal(cache[j])==v for j,v in physical.items()),'Complete XML/cache vector differs');require(i in physical,'Point outside numeric source domain');return ref.text,cells[i],physical[i]
 def scalar(self,row):
  ref,cell,raw=self.bound(row,'c:val');require(cell==row['celda_valor']and raw==Decimal(row['decimal_XML_texto']),'Recovery universal/raw source changed');return ref,cell,raw
 def named(self,row):
  s,sheet,root,node,i=self.ser(row);f=node.find('c:tx/c:strRef/c:f',C);require(f is not None,'Expected physical series name missing');ss,cells=ref_cells(f.text);require(ss==sheet and len(cells)==1,'Foreign series label');label=text(s.text(sheet,cells[0]));require(label==row['serie_literal'],'Original series label differs');cache=points(node.find('c:tx/c:strRef/c:strCache',C));require(cache.get(0)==label,'Name cache/header differs');return f.text,label
 def categorical(self,row):
  s,sheet,root,node,i=self.ser(row);nodecat=node.find('c:cat/c:strRef',C)
  if nodecat is None:nodecat=node.find('c:cat/c:numRef',C)
  require(nodecat is not None,'Category source reference missing');ref=nodecat.find('c:f',C).text;ss,cells=ref_cells(ref);require(ss==sheet and i<len(cells),'Category vector identity changed');value=self.category(s,ss,cells[i]);cache=nodecat.find('c:strCache',C)
  if cache is None:cache=nodecat.find('c:numCache',C)
  cp=points(cache);require(cp.get(i)==value and value==row['periodo_categoria_literal'],'Source category/cache/universal differs');return ref,value
 def pair(self,row,family):
  s,sheet,root,node,i=self.ser(row);xr,xc,x=self.bound(row,'c:xVal');yr,yc,y=self.bound(row,'c:yVal');require(yc==row['celda_valor']and y==Decimal(row['decimal_XML_texto']),'Scatter original Y changed');nr,label=self.named(row)if family=='gr15'else('source_header:E4',text(s.text(sheet,'E4')));require(row['serie_literal']is None and node.find('c:tx',C)is None,'Unnamed transition series identity changed')if family=='gr30'else None;require(row['periodo_categoria_literal']is not None and float(Decimal(row['periodo_categoria_literal']))==float(x),'Original scatter X category changed');bounds={}
  for a in root.findall('.//c:valAx',C):
   pos=a.find('c:axPos',C).get('val');sc=a.find('c:scaling',C);limits={k:Decimal(sc.find('c:'+k,C).get('val'))if sc.find('c:'+k,C)is not None else None for k in['min','max']};bounds[pos]=limits
  if family=='gr15':
   p=next(p for p in self.families['gr15_recovery_proposal.json']['groups']if p['series_id']==row['serie_id']);require(sheet=='Gráfico 15'and nr=="'Gráfico 15'!$"+p['label_cell'][0]+'$4'and label==p['label'],'Scatter source group changed');require(xr=="'Gráfico 15'!$"+p['x_range'][0]+'$6:$'+p['x_range'][0]+'$1012'and yr=="'Gráfico 15'!$"+p['y_range'][0]+'$6:$'+p['y_range'][0]+'$1012','Scatter coordinate references changed');year=int(s.text(sheet,'B'+str(i+6)));require(1971<=year<=2023,'Scatter source year changed');require(bounds['b']=={'min':Decimal(-10),'max':Decimal(10)}and bounds['l']=={'min':Decimal(-4),'max':Decimal(4)},'Scatter plot bounds changed');inside=-10<=x<=10 and -4<=y<=4;unit='percent_annual_change';country=p['country'];category=str(year);metric='Annual change in population aged15–64';xperiod=yperiod=str(year);proof=dict(pdf=self.pdf('2025',50,['PIB en volumen','15-64','Banco Mundial']),x_metric='annual_real_gdp_growth',y_metric='annual_population_growth_ages15_64',country=country,no_inferred_country=country is None,not_group_aggregate=True,not_independent_country_value_validation=True,source_scope='underlying_excel_scatter_points',within_plot_bounds=inside,axis_limits={'x':[-10,10],'y':[-4,4]},bounds_are_point_centres_not_exact_marker_visibility=True,external_current_reference_values_not_replacements=True)
  else:
   p=next(p for p in self.families['gr30_recovery_proposal.json']['profiles']if p['series_id']==row['serie_id']);require(sheet==p['block']and xr==p['x_ref']and yr==p['y_ref'],'Transition source references changed');require(s.text(sheet,'D4')==p['source_x_label']and s.text(sheet,'E4')==p['source_y_label'],'Transition period headers changed');country=text(s.text(sheet,'C'+str(i+5)));require(country==p['points'][i]['country_literal'],'Transition explicit geography changed');category=country;unit=p['canonical_unit'];metric=p['transition'];inside=None;xperiod=p['published_x_period'];yperiod=p['published_y_period'];proof=dict(pdf=self.pdf('2025',68,['DESEMPLEO','EMPLEO','Eurostat']),x_metric=p['transition'],y_metric=p['transition'],country=country,denominator=p['denominator'],transition_horizon='consecutive_quarters',seasonal_adjustment=None,quarter_count_per_average_not_inferred=True,mean_weighting_not_recomputed=True,current_api_values_not_replacements=True,source_scope='published_period_average_comparison')
  if family=='gr15':
   ref=self.references[(i+6,country=='JPN')];require(ref['year']==year and float(ref['x'])==float(x)and float(ref['y'])==float(y),'Frozen external reference point changed');proof.update(current_y_formula_matches_any_reference_country_same_year=bool(ref['pop_country_matches']),reference_country_match_count=len(ref['pop_country_matches']),reference_comparison_not_joint_xy_country_validation=True,no_claim_cause_of_reference_difference=True)
  proof.update(universal_x_category_equivalence='same IEEE754 double when original category float serialization differs',x_source_cell=xc,y_source_cell=yc,x_raw=str(x),y_raw=str(y),x_unit=unit,y_unit=unit,x_period=xperiod,y_period=yperiod,universal_category=row['periodo_categoria_literal']);self.paired.append((row['book'],row['row_number'],aid(row['book'],row['row_number']),family,row['serie_id'],i,x,y,unit,unit,xperiod,yperiod,label,country,xc,yc,inside,dumps(proof)))
  return self.result(row,y,y,unit,metric,label,category,'typed_source_scatter_pair_not_country_or_group_aggregate'if family=='gr15'else'published_consecutive_quarter_transition_period_average',proof,value_ref=yr,name_ref=nr,point_index=i,multiplier=Decimal(1),divisor=Decimal(1),binding_mode='source_snapshot_paired_coordinates',category_basis='explicit_source_year'if family=='gr15'else'explicit_geography_code',category_semantics='annual_source_year_without_inferred_country'if family=='gr15'else'explicit_country_and_two_published_average_windows')

 def other(self,row,family):
  s,sheet,root,node,i=self.ser(row);vr,cell,raw=self.scalar(row);nr,label=self.named(row);cr,category=self.categorical(row)
  if family=='gr78':
   p=self.families['gr78_recovery_proposal.json'];require(sheet=='Gráfico 78'and vr in["'Gráfico 78'!$C$6:$U$6","'Gráfico 78'!$C$7:$U$7"]and cr=="'Gráfico 78'!$C$4:$U$4",'Treasury source row/category changed');require(cell!='Q7','Known incorrect2021net must use external replacement');require(s.text(sheet,'B2')=='Gráfico 78. NECESIDADES DE FINANCIACIÓN HISTÓRICAS DEL TESORO (MM€ Y % PIB)','Treasury metric caption changed');parent=next(a for a in root.iter()if node in list(a));ids={a.get('val')for a in parent.findall('c:axId',C)};axes=[a for a in root.findall('.//c:valAx',C)if a.find('c:axId',C).get('val')in ids];axis=next(a for a in axes if a.find('c:axId',C).get('val')=='83499887');require(axis.find('c:dispUnits/c:builtInUnit',C).get('val')=='thousands','Treasury display divisor changed');require(s.number(sheet,'E5')==Decimal(224285),'Verified2009unit anchor changed');unit='billion_euros';value=raw/Decimal(1000);div=Decimal(1000);metric='Treasury redemptions'if cell.endswith('6')else'Treasury net issuance';role='reported_treasury_amount_with_verified_million_to_billion_scale';proof=dict(pdf=self.pdf('2025',134,['TESORO']),raw_grid_unit='million_euros',display_divisor='1000',official_scale_anchor='Moncloa2009gross224285million',incorrect_original_2021_net_not_reused=True,source_projection_marker=category=='25(p)',observation_status='source_explicit_projection'if category=='25(p)'else'source_not_marked_projection',no_implicit_actual_status=True)
  else:
   p=self.families['gr23_recovery_proposal.json'];sp=next(a for a in p['series']if a['series_id']==row['serie_id']);require(sheet=='Gráfico 23'and[nr,cr,vr]==sp['all_ooxml_refs'],'Activity cluster references changed');require(label==sp['original_series_literal'],'Activity source series changed');point=next(a for a in sp['rows']if a['row_number']==row['row_number']);require(category==point['source_country_literal']and s.text(sheet,point['source_country_cell'])==category,'Activity explicit geography changed');t=PdfReader(PRIMARY/p['unit_primary_source_file']).pages[p['unit_primary_pdf_page']-1].extract_text();require('Tasa de Actividad (%)'in t and '60'in t and '74'in t,'Activity independent unit definition absent');unit=p['canonical_unit'];value=raw;div=Decimal(1);metric='Labour-force participation rate of population aged60–74';role='reported_activity_comparison_point_not_pca_or_assumed_single_year';proof=dict(pdf=self.pdf('2025',56,['60','74']),unit_primary_file=p['unit_primary_source_file'],unit_primary_pdf_page=22,unit_primary_sha256=p['unit_primary_source_sha256'],source_country_literal=category,period_context_literal='2011–2023 and projection2024–2070',point_period_literal=label if label in['España (2011-2023)','Proyección España (24-70)']else None,no_inferred_specific_country_year_average=True,no_inferred_pca_score=True,no_value_replacement_from_later_technical_document=True)
  proof.update(unit=unit,multiplier='1',divisor=str(div),original_series=row['serie_literal'],original_category=row['periodo_categoria_literal'])
  return self.result(row,raw,value,unit,metric,label,category,role,proof,value_ref=vr,category_ref=cr,name_ref=nr,point_index=i,multiplier=Decimal(1),divisor=div,binding_mode='verified_physical_source_snapshot',category_basis='explicit_source_category',category_semantics='source_category_as_labelled')
 def recover(self,row,family):
  try:return 'usable_recovered_source',self.pair(row,family)if family in['gr15','gr30']else self.other(row,family)
  except(ValueError,KeyError,StopIteration,ArithmeticError,AttributeError,TypeError,IndexError)as e:return 'source_recovery_evidence_review:'+str(e),None

def fiscal_records(c,r):
 bad=read(FISCAL_BAD);require(len(bad)==3444 and sum(a['was_admitted']for a in bad)==1722,'Authorized fiscal artifact domain changed');require({(a['book'],a['row_number'])for a in bad if a['was_admitted']}==set(c.execute("SELECT book,row_number FROM closure_family_public_observations WHERE book='2025'AND source_sheet='Gráfico 100'AND block<>'Gráfico 100'").fetchall()),'Approved fiscal withdrawal keys differ');q=c.execute("SELECT * FROM staging_records WHERE book='2025'");names=[n[0]for n in q.description];stage={a['row_number']:a for a in[dict(zip(names,v))for v in q.fetchall()]};rows=[];links=[];figures=[]
 canonical={}
 for b,n,cell,year,series,unit,v in c.execute("SELECT book,row_number,source_cell,category_literal,series_literal,unit,value FROM closure_family_public_observations WHERE book='2025'AND block='Gráfico 100'ORDER BY row_number").fetchall():canonical.setdefault((cell,year,series,v),[]).append((b,n,unit))
 for a in bad:
  st=stage[a['row_number']];require(all(st[k]==a[k]for k in['book','row_number','bloque','serie_id','serie_literal','periodo_categoria_literal','parte_grafico_imagen','hoja_valor','celda_valor','decimal_XML_texto']),'Fiscal original payload changed');source_caption=r.sources['2025'].text('Gráfico 100','B3');require(source_caption=='Gráfico 100. ANOMALÍA CLIMÁTICA (AUMENTO DE TEMPERATURAS)','Temperature source metric changed');fig=int(a['bloque'].split()[-1]);proof=dict(expected_fiscal_caption=r.sources['2025'].text(a['bloque'],'B2'),actual_reference_source_caption=source_caption if st['hoja_valor']=='Gráfico 100'else None,original_raw=st['decimal_XML_texto'],original_year=st['periodo_categoria_literal'],original_part=st['parte_grafico_imagen'],replacement_is_verified_figure_manifest=True,no_pointwise_reassignment=True,no_origin_inference_for_literals=st['hoja_valor']is None)
  temp=None
  if st['hoja_valor']=='Gráfico 100':
   source_year=r.category(r.sources['2025'],'Gráfico 100','B'+re.search('[0-9]+',st['celda_valor'])[0]);require(st['periodo_categoria_literal']is None or st['periodo_categoria_literal']==source_year,'Temperature original/source year differs');matches=canonical[(st['celda_valor'],source_year,st['serie_literal'],Decimal(st['decimal_XML_texto']))];require(len(matches)==2 and len({m[2]for m in matches})==1,'Explicit temperature canonical identity changed');temp=aid(matches[0][0],matches[0][1]);proof.update(canonical_temperature_key=temp,other_identical_temperature_key=aid(matches[1][0],matches[1][1]),correct_source_unit=matches[0][2])
  rows.append(('2025',a['row_number'],a['was_admitted'],'nonpublication_fiscal_plot_object',dumps(proof)));links.append(('2025',a['row_number'],'verified_publication_figure',str(fig),None,dumps(proof)))
 for p in read(FISCAL):
  s=r.sources['2025'];sheet=p['source_sheet'];require(s.text(sheet,'B2')==p['caption'],'Published fiscal caption changed');part=p['published_chart_part'];root=ET.fromstring(s.xml.parts[part]);expected={a['row_number']:a for a in p['records']};require(len(expected)==p['point_count'],'Fiscal manifest cardinality differs');r.pdf('2025',p['pdf_page'],['GRÁFICO '+str(p['figure'])])
  for a in p['records']:
   q=c.execute("SELECT series_id,series_literal,category_literal,unit,value,source_cell FROM closure_family_public_observations WHERE book='2025'AND row_number=?",[a['row_number']]);v=q.fetchone();require(v is not None and v[0]==a['series_id']and v[1]==a['series_literal']and v[2]==a['category_literal']and v[3]==a['unit']and v[4]==Decimal(a['value'])and v[5]==a['source_cell'],'Published canonical point changed');i=int(re.search(r'_p([0-9]+)$',stage[a['row_number']]['id'])[1]);node=root.findall('.//c:ser',C)[int(a['series_id'].split('_s')[1])-1];vr=node.find('c:val/c:numRef/c:f',C).text;cat=node.find('c:cat/c:numRef/c:f',C);cat=node.find('c:cat/c:strRef/c:f',C)if cat is None else cat;cr=cat.text if cat is not None else None;vs,vc=ref_cells(vr);cs,cc=ref_cells(cr)if cr is not None else(sheet,['B'+str(j+5)for j in range(29)]);require(cat is not None or(int(points(node.find('c:cat/c:numLit',C))[i])==int(s.number(sheet,cc[i]))),'Published fiscal literal category/grid mismatch');require(vs==cs==sheet and vc[i]==a['source_cell']and str(int(s.number(cs,cc[i])))==a['category_literal']and s.number(vs,vc[i])==v[4],'Published fiscal physical data mismatch');require(Decimal(points(node.find('c:val/c:numRef/c:numCache',C))[i])==v[4],'Published fiscal cache mismatch');figures.append(('2025',p['figure'],a['series_literal'],int(a['category_literal']),aid('2025',a['row_number']),a['row_number'],p['pdf_page'],part,dumps(dict(figure_caption=p['caption'],source_scope='verified_published_curve',pdf_display_anchors=p['pdf_numeric_anchors']))))
  for a in p['pdf_numeric_anchors']:require(s.number(sheet,a['source_cell']).quantize(Decimal('.1'))==Decimal(a['pdf_display']),'Published fiscal numeric anchor differs')
 require(len(figures)==957,'Published fiscal figure coverage changed');return rows,links,figures

def external_fact():
 p=PRIMARY/'tesoro_2022_strategy.pdf';t=PdfReader(p).pages[2].extract_text();require('En millones de euros'in t and re.search(r'EMISIONES NETAS\s+109[.]922\s+75[.]138',t),'Official Treasury2021 table identity changed');return(OFFICIAL_ID,'Tesoro','net_issuance',2021,'billion_euros',Decimal('75.138'),'million_euros',Decimal(75138),'75.138',str(p.relative_to(ROOT)),sha(p),3,'EMISIONES NETAS /2021',dumps(dict(source_year=2022,observation_year=2021,source_kind='official_treasury_report_table',reported_not_airef_excel_amount=True,raw_printed_spanish_thousands='75.138',raw_numeric_million_euros='75138',raw_airef_Q7_not_rewritten='75318',source_scope='authoritative_external_correction',source_url=next(f['url']for f in read(SOURCES)if f['file']=='tesoro_2022_strategy.pdf'))))

def collect(c):
 files=file_proofs(c);plan=read(PLAN);require(len(plan['selected'])==1170,'Authorized selection cardinality changed');q=c.execute('SELECT * FROM closure_family_numeric_queue');names=[a[0]for a in q.description];pending={(a['book'],a['row_number']):a for a in[dict(zip(names,v))for v in q.fetchall()]};require(len(pending)==4704,'Previous pending domain changed');r=RecoveryResolver();rows=[];reviews=[]
 try:
  for p in plan['selected']:
   row=pending[p['book'],p['row_number']];require(digest(row)==p['payload_sha256'],'Recovery original metadata changed');status,result=r.recover(row,p['family']);reviews.append((row['book'],row['row_number'],status))
   if result is not None:rows.append(result)
  retired,replacements,figures=fiscal_records(c,r);pairs=r.paired;external=external_fact();badnet=[]
  for n in[37525,37582]:
   row=pending['2025',n];require(row['bloque']=='Gráfico 78'and row['celda_valor']=='Q7'and Decimal(row['decimal_XML_texto'])==Decimal(75318),'Treasury incorrect source point domain changed');ev=dumps(dict(original_excel_Q7='75318',official_numeric_million_euros='75138',external_observation_id=OFFICIAL_ID,source_pdf_page=3,no_raw_rewrite=True));badnet.append(('2025',n,False,'official_source_replaces_incorrect_reported_net_issuance',ev));replacements.append(('2025',n,'authoritative_external_replacement',OFFICIAL_ID,OFFICIAL_ID,ev))
  quarantine=[a for a in retired if a[2]]+badnet;require(len(quarantine)==1724,'Quarantine domain changed');available={(a[0],a[1])for a in rows};retired_keys={(a[0],a[1])for a in retired};netkeys={('2025',37525),('2025',37582)};q=c.execute('SELECT book,row_number,disposition FROM closure_family_decisions');previous_decisions={(b,n):d for b,n,d in q.fetchall()};resolution=[]
  for key,row in sorted(pending.items()):
   state='admitted_validated'if key in available else'repaired_by_verified_figure_registry'if key in retired_keys else'replaced_by_official_fact'if key in netkeys else previous_decisions.get(key,'blocked');state='blocked'if state=='usable'else state;resolution.append((key[0],key[1],state,True))
  resolution.extend((a[0],a[1],'repaired_by_verified_figure_registry',False)for a in retired if a[2]);require(len(resolution)==6426,'Recovery resolution domain changed');return dict(rows=rows,reviews=reviews,pairs=pairs,retired=retired,replacements=replacements,figures=figures,external=[external],quarantine=quarantine,resolution=resolution,files=files)
 finally:r.close()

def sql_views():
 usable="SELECT book,row_number FROM closure_family_usable_keys EXCEPT SELECT book,row_number FROM source_recovery_quarantine WHERE was_admitted UNION ALL SELECT book,row_number FROM source_recovery_observations"
 queue="SELECT s.*,COALESCE(r.state,'blocked') source_recovery_state FROM staging_records s JOIN(SELECT * FROM closure_family_usable_keys UNION SELECT book,row_number FROM closure_family_numeric_queue)u USING(book,row_number)LEFT JOIN source_recovery_resolution r USING(book,row_number)WHERE NOT EXISTS(SELECT 1FROM source_recovery_usable_keys k WHERE k.book=s.book AND k.row_number=s.row_number)"
 old="SELECT p.*,'airef:'||p.book||':'||p.row_number observation_id FROM closure_family_public_observations p JOIN source_recovery_usable_keys k USING(book,row_number)"
 new="SELECT o.*,'source_recovery_observations' source_layer,'airef:'||o.book||':'||o.row_number observation_id FROM source_recovery_observations o"
 external="SELECT NULL::VARCHAR book,NULL::INTEGER row_number,observation_id,'external_official_facts' source_layer,'Gráfico 78' block,metric metric_literal,'Emisión neta' series_literal,CAST(year AS VARCHAR)category_literal,unit,value,raw_numeric raw_value,raw_grid_unit original_amount_unit,source_file source_pdf_path,source_sha256 source_pdf_sha256,pdf_page,source_locator,proof quantity_evidence FROM source_recovery_external_facts"
 return[('source_recovery_usable_keys',usable),('source_recovery_numeric_queue',queue),('source_recovery_public_observations',old+' UNION ALL BY NAME '+new+' UNION ALL BY NAME '+external)]

def build(output):
 output=Path(output).resolve();require(output.is_relative_to(ROOT)and not output.exists(),'Output must be a new project directory');require(sha(PREVIOUS)==PREVIOUS_SHA,'Previous database changed');output.mkdir(parents=True);db=output/'source_recovery.duckdb';shutil.copyfile(PREVIOUS,db)
 with duckdb.connect(str(db))as c:
  require(not c.execute("SELECT table_name FROM information_schema.tables WHERE table_name LIKE 'source_recovery_%'").fetchall(),'Recovery namespace collision');data=collect(c);c.execute('CREATE TABLE source_recovery_observations AS SELECT * FROM reviewed_family_observations WHERE FALSE');c.execute('ALTER TABLE source_recovery_observations ADD PRIMARY KEY(book,row_number)')
  if data['rows']:c.executemany('INSERT INTO source_recovery_observations VALUES('+','.join('?'for _ in FIELDS)+')',data['rows'])
  schemas=[('source_recovery_review','book VARCHAR,row_number INTEGER,status VARCHAR',data['reviews'],'book,row_number'),('source_recovery_scatter_pairs','book VARCHAR,row_number INTEGER,observation_id VARCHAR,family VARCHAR,series_id VARCHAR,point_index INTEGER,x_value DECIMAL(38,24),y_value DECIMAL(38,24),x_unit VARCHAR,y_unit VARCHAR,x_period VARCHAR,y_period VARCHAR,source_group VARCHAR,geography_literal VARCHAR,x_source_cell VARCHAR,y_source_cell VARCHAR,within_plot_bounds BOOLEAN,evidence VARCHAR',data['pairs'],'book,row_number'),('source_recovery_retired_objects','book VARCHAR,row_number INTEGER,was_admitted BOOLEAN,reason VARCHAR,evidence VARCHAR',data['retired'],'book,row_number'),('source_recovery_quarantine','book VARCHAR,row_number INTEGER,was_admitted BOOLEAN,reason VARCHAR,evidence VARCHAR',data['quarantine'],'book,row_number'),('source_recovery_replacements','book VARCHAR,row_number INTEGER,replacement_kind VARCHAR,target_reference VARCHAR,replacement_observation_id VARCHAR,evidence VARCHAR',data['replacements'],'book,row_number'),('source_recovery_fiscal_points','book VARCHAR,figure INTEGER,series VARCHAR,year INTEGER,observation_id VARCHAR,row_number INTEGER,pdf_page INTEGER,chart_part VARCHAR,evidence VARCHAR',data['figures'],'book,figure,series,year'),('source_recovery_resolution','book VARCHAR,row_number INTEGER,state VARCHAR,original_queue BOOLEAN',data['resolution'],'book,row_number'),('source_recovery_external_facts','observation_id VARCHAR,provider VARCHAR,metric VARCHAR,year INTEGER,unit VARCHAR,value DECIMAL(38,24),raw_grid_unit VARCHAR,raw_numeric DECIMAL(38,24),raw_printed VARCHAR,source_file VARCHAR,source_sha256 VARCHAR,pdf_page INTEGER,source_locator VARCHAR,proof VARCHAR',data['external'],'observation_id'),('source_recovery_primary_files','file VARCHAR,url VARCHAR,sha256 VARCHAR',[(f['file'],f.get('url'),f['sha256'])for f in data['files']],'file')]
  for name,schema,values,key in schemas:
   c.execute('CREATE TABLE '+name+'('+schema+',PRIMARY KEY('+key+'))')
   if values:c.executemany('INSERT INTO '+name+' VALUES('+','.join('?'for _ in values[0])+')',values)
  for name,sql in sql_views():c.execute('CREATE VIEW '+name+' AS '+sql)
  counts=dict(c.execute('SELECT state,count(*)FROM source_recovery_resolution GROUP BY state').fetchall());added=len(data['rows']);review=c.execute("SELECT status,count(*)FROM source_recovery_review WHERE status<>'usable_recovered_source'GROUP BY status").fetchall()
 report=dict(status='built_pending_validation',added=added,withdrawn=1722,external_facts=1,original_usable=42509-1722+added,public_observations=42510-1722+added,original_numeric_pending=4704+1722-added,repaired_fiscal_objects=3444,published_fiscal_points=957,resolution_counts=counts,review=review,previous_sha256=PREVIOUS_SHA,plan_sha256=PLAN_SHA,primary_manifest_sha256=SOURCE_SHA,code_sha256=sha(Path(__file__)),db_sha256=sha(db),full_user_goal_complete=False,rag_integrated=False);(output/'manifest.json').write_text(dumps(report),encoding='utf8');return report

def check_new(c):
 data=collect(c)
 for table,key in[('source_recovery_observations','rows'),('source_recovery_review','reviews'),('source_recovery_scatter_pairs','pairs'),('source_recovery_retired_objects','retired'),('source_recovery_quarantine','quarantine'),('source_recovery_replacements','replacements'),('source_recovery_fiscal_points','figures'),('source_recovery_external_facts','external'),('source_recovery_resolution','resolution')]:
  actual=c.execute('SELECT * FROM '+table).fetchall();expected=data[key];require(len(actual)==len(expected)and set(actual)==set(expected),'Recovery table differs from source proof: '+table)
 require(len(data['rows'])==1170,'Not all authorized physical source observations recovered');require(len(data['pairs'])==1069,'Paired source count changed');require(c.execute('SELECT count(*)FROM source_recovery_usable_keys').fetchone()[0]==41957,'Recovered original usable count changed');require(c.execute('SELECT count(*)FROM source_recovery_public_observations').fetchone()[0]==41958,'Public observation count changed');require(c.execute('SELECT count(*)FROM source_recovery_numeric_queue').fetchone()[0]==5256,'Original raw outside count changed');require(not c.execute('SELECT * FROM source_recovery_usable_keys INTERSECT SELECT book,row_number FROM source_recovery_numeric_queue').fetchall(),'Recovery raw public/outside overlap');require(c.execute('SELECT count(*)FROM(SELECT * FROM source_recovery_usable_keys UNION SELECT book,row_number FROM source_recovery_numeric_queue)').fetchone()[0]==47213,'Original raw universe changed');require(not c.execute('SELECT observation_id FROM source_recovery_public_observations GROUP BY observation_id HAVING count(*)<>1').fetchall(),'Public observation identity collision');require(not c.execute('SELECT book,row_number FROM source_recovery_quarantine INTERSECT SELECT * FROM source_recovery_usable_keys').fetchall(),'Quarantined source publicly admitted');require(not c.execute("SELECT * FROM source_recovery_public_observations WHERE source_sheet='Gráfico 100'AND block<>'Gráfico 100'").fetchall(),'Temperature values still attributed to fiscal metrics');require(c.execute("SELECT count(*)FROM source_recovery_scatter_pairs WHERE family='gr15'AND within_plot_bounds").fetchone()[0]==998 and c.execute("SELECT count(*)FROM source_recovery_scatter_pairs WHERE family='gr15'AND NOT within_plot_bounds").fetchone()[0]==9,'Scatter viewport scope changed')
 for name,sql in sql_views():
  require(not c.execute('SELECT * FROM '+name+' EXCEPT ALL ('+sql+')').fetchall()and not c.execute('('+sql+') EXCEPT ALL SELECT * FROM '+name).fetchall(),'Recovery public view changed: '+name)
 return data

def validate(db):
 previous=validate_previous(db)
 with duckdb.connect(str(db),read_only=True)as c:data=check_new(c);counts=dict(c.execute('SELECT state,count(*)FROM source_recovery_resolution GROUP BY state').fetchall())
 return dict(status='pass',previous_validation=previous['status'],added=1170,withdrawn=1722,external_facts=1,original_usable=41957,public_observations=41958,original_numeric_pending=5256,repaired_fiscal_objects=3444,published_fiscal_points=957,resolution_counts=counts,previous_sha256=PREVIOUS_SHA,plan_sha256=PLAN_SHA,primary_manifest_sha256=SOURCE_SHA,code_sha256=sha(Path(__file__)),db_sha256=sha(Path(db)),full_user_goal_complete=False,rag_integrated=False)

def query_observation(db,*,observation_id):
 require(isinstance(observation_id,str)and observation_id.strip(),'Exact observation identity required')
 with duckdb.connect(str(db),read_only=True)as c:
  q=c.execute('SELECT * FROM source_recovery_public_observations WHERE observation_id=?',[observation_id]);names=[a[0]for a in q.description];rows=q.fetchall()
  if len(rows)!=1:raise LookupError('No unique recovered public observation; no legacy fallback')
  result={n:str(v)if isinstance(v,Decimal)else v for n,v in zip(names,rows[0])}
  if result.get('book')is not None:
   src=c.execute('SELECT xlsx_path,xlsx_sha256,pdf_path,pdf_sha256 FROM sources WHERE book=?',[result['book']]).fetchone();result.update(source_xlsx_path=src[0],source_xlsx_sha256=src[1],source_report_pdf_path=src[2],source_report_pdf_sha256=src[3]);overlay=c.execute('SELECT evidence FROM closure_metadata_overlays WHERE book=?AND row_number=?',[result['book'],result['row_number']]).fetchone()
   if overlay:result['metadata_overlay']=json.loads(overlay[0]);result['safe_quantity_role']=result['metadata_overlay']['quantity_role']
  else:result['source_variant']='authoritative_external_correction_not_airef_original'
 if result['unit']=='thousand_persons_signed_plot':
  proof=json.loads(result['quantity_evidence']);result.update(population_thousands=proof['population_thousands'],plot_value_thousands=proof['plot_value_thousands'],sex=proof['sex'],reference_year=proof['reference_year'],age_group=proof['age_group'],population_scope_literal=proof['population_scope_literal'],value_is_population_count=False)
 result['safe_recovery_view']=True;return result

def query_safe(db,*,book,row_number):
 require(book in['2025','2026']and type(row_number)is int and row_number>0,'Exact original source key required');return query_observation(db,observation_id=aid(book,row_number))

def query_figure(db,*,book,figure,series,year):
 require(book in['2025','2026']and type(figure)is int and figure>0 and isinstance(series,str)and series.strip()and type(year)is int,'Exact publication figure identity required')
 with duckdb.connect(str(db),read_only=True)as c:rows=c.execute('SELECT observation_id,evidence FROM source_recovery_fiscal_points WHERE book=?AND figure=?AND series=?AND year=?',[book,figure,series,year]).fetchall()
 if len(rows)!=1:raise LookupError('No unique published fiscal series/year; no cross-panel or raw-object fallback')
 result=query_observation(db,observation_id=rows[0][0]);result.update(publication_figure=figure,publication_manifest=json.loads(rows[0][1]));return result

def query_scatter(db,*,book,row_number,require_plot_bounds=False):
 require(book in['2025','2026']and type(row_number)is int and row_number>0 and type(require_plot_bounds)is bool,'Exact source scatter key required')
 with duckdb.connect(str(db),read_only=True)as c:q=c.execute('SELECT * FROM source_recovery_scatter_pairs WHERE book=?AND row_number=?',[book,row_number]);names=[a[0]for a in q.description];rows=q.fetchall()
 if len(rows)!=1:raise LookupError('No unique source scatter point')
 result={n:str(v)if isinstance(v,Decimal)else v for n,v in zip(names,rows[0])};require(query_safe(db,book=book,row_number=row_number)['observation_id']==result['observation_id'],'Scatter point not currently safe')
 if require_plot_bounds and result['within_plot_bounds']is not True:raise LookupError('Point centre outside verified plot bounds or bounds not certified')
 result['source_context']=json.loads(result['evidence']);return result

def explain_source_record(db,*,book,row_number):
 require(book in['2025','2026']and type(row_number)is int and row_number>0,'Exact original source key required')
 with duckdb.connect(str(db),read_only=True)as c:
  state=c.execute('SELECT state FROM source_recovery_resolution WHERE book=?AND row_number=?',[book,row_number]).fetchone();replacements=c.execute('SELECT replacement_kind,target_reference,replacement_observation_id,evidence FROM source_recovery_replacements WHERE book=?AND row_number=?',[book,row_number]).fetchall()
 if state is None:raise LookupError('Source record not in recovery resolution domain')
 return dict(book=book,row_number=row_number,state=state[0],replacements=[dict(kind=k,target=t,observation_id=o,evidence=json.loads(e))for k,t,o,e in replacements],no_automatic_point_reassignment=True)

def main():
 p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='command',required=True);b=s.add_parser('build');b.add_argument('--output-dir',type=Path,required=True);b=s.add_parser('validate');b.add_argument('--db',type=Path,required=True);b.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.command=='build':report=build(a.output_dir)
 else:
  require(not a.output.exists(),'Validation report already exists')
  try:report=validate(a.db)
  except Exception as e:report=dict(status='fail',error=str(e))
  a.output.write_text(dumps(report),encoding='utf8')
 print(dumps(report));return int(report['status']=='fail')
if __name__=='__main__':raise SystemExit(main())
