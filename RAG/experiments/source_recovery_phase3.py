"""Typed source admissions, routing repairs and explicitly restricted conflicting variants."""
import argparse,json,re,shutil,datetime,xml.etree.ElementTree as ET
from decimal import Decimal as D
from pathlib import Path
import duckdb
from pypdf import PdfReader
from experiments import source_recovery_phase2 as phase2
from experiments.source_recovery import ROOT,FIELDS,C,require,read,dumps,digest,aid,sha
from experiments.excel_pilot import ExcelXML
from experiments.indexed_semantics import points
BASE=ROOT/'evaluation/source_recovery_phase2_2026-11-01/run_v6/source_recovery_phase2.duckdb'
BASE_SHA='77de869cb10865ec9083dff697f2418030a811e48ee0d5af0c6d5e7e47521fff'
BASE_CODE_SHA='ea63584cce122a1e9504c9bc351b1a77e78ce29390f76e2524e8ed4cce920491'
ASSETS=ROOT/'evaluation/source_recovery_phase3_2026-11-01/assets'
ASSET_SHA='b3d1896cdc6da1596c0cf7fd5dd46a94d5bf7b64def590bf410a614878013e4c'

def asset(name):return read(ASSETS/name)
def guards(c):
 require(sha(BASE)==BASE_SHA and sha(ROOT/'experiments/source_recovery_phase2.py')==BASE_CODE_SHA,'Frozen phase2 snapshot/code changed');require(sha(ASSETS/'manifest.json')==ASSET_SHA,'Phase3 evidence manifest changed')
 cert=read(BASE.parent/'validation.json');require(cert['status']=='pass'and cert['db_sha256']==BASE_SHA and cert['code_sha256']==BASE_CODE_SHA,'Phase2 numeric chain not certified')
 for p in asset('manifest.json'):require(sha(ASSETS/p['file'])==p['sha256'],'Frozen phase3 primary evidence changed: '+p['file'])
 for b,x,h,p,ph in c.execute('SELECT * FROM sources').fetchall():require(sha(ROOT/x)==h and sha(ROOT/p)==ph,'Original source changed: '+b)

class Resolver(phase2.Resolver):
 def __init__(self):
  super().__init__();self.proposals={p['proposal']:asset(p['proposal'])for p in asset('selection.json')};self.external={};self.external_sheets={}
 def independent(self,filename,sheet,cell):
  if filename not in self.external:self.external[filename]=ExcelXML(ASSETS/filename)
  key=filename,sheet
  if key not in self.external_sheets:self.external_sheets[key]=self.external[filename].read(sheet)
  return self.external_sheets[key][cell]
 def record(self,row,raw,value,unit,metric,series,category,role,proof,**kw):return self.result(row,raw,value,unit,metric,series,category,role,proof,rule_id='recovery3_primary_typed_source_v1',**kw)
 def admitted(self,row,family,p):
  original=row;s,sh,root,node,i=self.ser(row)
  if family=='excess':
   point=next(a for a in p['points']if a['source_key']==row['row_number']);sh='Gráfico RE_2.3.';cell=point['value_cell'];raw=s.number(sh,cell);vr=None;literal=node.find('c:val/c:numLit',C);require(literal is not None and D(points(literal)[i])==raw==D(row['decimal_XML_texto'])and row['hoja_valor']is None,'Literal excess cache/physical bridge changed');row=dict(row,hoja_valor=sh,celda_valor=cell)
  else:vr,cell,raw=self.scalar(row)
  if family=='excess':
   point=next(a for a in p['points']if a['source_key']==row['row_number']);require(cell==point['value_cell']and raw==D(point['literal_value'])and s.number(sh,point['date_cell'])==D(point['excel_date_serial']),'Monthly excess physical coordinate changed');require(s.text(sh,'C6')=='excess'and row['serie_literal']is None,'Monthly excess null series/header changed');dt=s.text(sh,point['date_cell']);date=datetime.datetime(1899,12,30)+datetime.timedelta(days=int(point['excel_date_serial']));require(str(dt)==point['date']and str(date)==point['date'],'Monthly explicit physical date/serial differs');cat=node.find('c:cat/c:numLit',C);require(cat is not None and D(points(cat)[i])==D(point['excel_date_serial']),'Monthly literal chart calendar differs');unit='thousand_affiliates';value=raw;metric=p['metric'];label=None;category=date.strftime('%Y-%m');role='reported_model_estimate_excess_formula_and_orientation_unknown';proof=dict(source_unit_scale=p['source_unit_scale'],population_scope=p['population'],formula=None,residual_orientation=None,no_observed_or_counterfactual_levels_invented=True,independent_causal_identification_allowed=False,source_header='excess',source_literal_category=row['periodo_categoria_literal'],reported150thousand_is_approximate_context_not_each_month_or_replacement=True,source_pdf_sha256=p['source_pdf_sha256'],source_pdf_page=7)
  elif family=='excise':
   point=next(a for a in p['observations']if a['source_key']==row['row_number']);require(cell==point['source_cell']and raw==D(point['raw'])==self.independent('airef_dt_ingresos_2025.xlsx',point['dt_sheet'],point['dt_value_cell']).number(),'Independent excise tax raw match changed')
   if point['figure']==41:_,label=self.named(row)
   else:require(row['serie_literal']is None and node.find('c:tx',C)is None,'Unnamed excise weight series changed');label=None
   caption=self.independent('airef_dt_ingresos_2025.xlsx',point['dt_sheet'],p['dt_caption_cells'][point['dt_sheet']]).value;require('IIEE'in caption.upper().replace('.',''),'Independent excise tax scope caption absent');_,category=self.categorical(row);require(category==point['age_group']and self.independent('airef_dt_ingresos_2025.xlsx',point['dt_sheet'],point['dt_age_cell']).value==category,'Independent excise age group differs');unit='percent_average_excise_tax_rate'if point['figure']==41 else'percent_total_excise_revenue_by_age';value=raw;metric=point['metric'];role='reported_excise_tax_rate'if point['figure']==41 else'reported_excise_revenue_composition_weight';proof=dict(point,wrong_original_IVA_scope_preserved=True,independent_source_file='airef_dt_ingresos_2025.xlsx',independent_source_sha256=p['source_sha256'],pdf_sha256=p['pdf_sha256'],bounded_tax_scope_override=True)
  else:
   point=next(a for a in p['observations']if a['row_number']==row['row_number']);require(cell==point['source_cell']and raw==D(point['raw'])==self.independent('smi_1.xlsx',point['dt_sheet'],point['dt_cell']).number(),'Independent SMI component differs');_,label=self.named(row);_,category=self.categorical(row);require(label==point['original_label']and category==point['year'],'SMI source series/year changed');unit='percent_policy_intensity_indicator'if row['serie_id'].endswith('_s3')else'percentage_points_policy_intensity_indicator';value=raw*100;metric='Published SMI policy intensity indicator decomposition';role=point['semantic_role'];proof=dict(point,population_context=p['population_context'],quantity_scope=p['quantity_scope'],global_method_formula_preserved_not_redefined=True,not_raw_affected_workers_fraction=True,not_individual_salary_adjustment_factor=True,independent_causal_identification_allowed=False,source_method_conflict_preserved=True,primary_pdf_file='smi_0.pdf',primary_pdf_sha256=next(a['sha256']for a in p['source_files']if a['file']=='smi_0.pdf'))
  return self.record(row,raw,value,unit,metric,label,category,role,proof,value_ref=vr,multiplier=D(100)if family=='smi'else D(1),binding_mode='independently_accredited_typed_source_snapshot',universal_value_sheet=original['hoja_valor'],universal_value_cell=original['celda_valor'])

 def variant(self,row,family,p):
  s,sh,root,node,i=self.ser(row);vr,cell,raw=self.scalar(row);array=p['points']if family=='variant36'else p['observations'];point=next(a for a in array if a['row_number']==row['row_number']);require(cell==point.get('source_cell',point.get('original_source_cell'))and raw==D(point['raw']),'Reported source variant raw changed')
  label=row['serie_literal'];mult=D(point.get('multiplier',1));value=raw*mult;unit=point['unit'];proof=dict(point,raw_original_preserved=True,missing_cause=None,source_variant_family=family,source_preference_required=True)
  if family=='variant36':
   category=self.category(s,sh,point['physical_year_cell']);require(category==str(point['resolved_year'])and row['periodo_categoria_literal']is None,'Ageing source explicit physical year differs');rr=re.search('[0-9]+',cell)[0];components=[s.number(sh,col+rr)for col in'DEFG'];require(all(x is not None for x in components)and abs(sum(components)-raw)<=D('1e-12'),'Ageing workbook components no longer reconcile');metric=point['metric'];label='Ageing spending contribution';proof.update(calculation_safe=True,calculation_within_workbook=True,numeric_reconciliation_failed=False,pdf_comparison_confirmed=False,warning='Official workbook differs from report approximation; no claim they are the same numeric version')
  else:
   category='2050 vs Opinion2025'if family=='variant30'else point.get('period_literal',point.get('year'));metric='Reported debt ratio change contribution or total'if family in['variant30','variant64']else'Reported education spending growth component or total';proof.update(calculation_safe=False,numeric_reconciliation_failed=True,warning='Reported figures have an unresolved arithmetic inconsistency; do not calculate or describe as a reconciled decomposition')
   if family=='variant30':require(s.text(sh,'B'+re.search('[0-9]+',cell)[0])==point['series_literal'],'Debt workbook variant physical label changed')
   if family=='variant54':require(raw==self.independent('dt_social_1.xlsx',point['dt_source_sheet'],point['dt_source_cell']).number(),'Independent education source variant differs')
  record=self.record(row,raw,value,unit,metric,label,category,'explicit_source_reported_variant_not_default_validated_quantity',proof,value_ref=vr,multiplier=mult,binding_mode='explicit_restricted_source_variant')
  return record+(family,bool(proof['calculation_safe']),dumps(proof))

def public_row(c,book,n):
 q=c.execute('SELECT '+','.join(FIELDS)+' FROM recovery2_public_observations WHERE book=?AND row_number=?',[book,n]);v=q.fetchone();require(v is not None,'Expected existing public source missing');return dict(zip(FIELDS,v))

def routing(c,r,stage):
 aliases=[];retired=[];coverage=[];p=asset('gr48_alias_recovery_proposal.json')
 for a in p['mapping']:
  row=stage['2025',a['original_key']];s,sh,root,node,i=r.ser(row);vr,cell,raw=r.scalar(row);nr,label=r.named(row);cr,year=r.categorical(row);require(row['id']==a['original_point']and cell==a['source_cell']and raw==D(a['raw'])and label==p['wrong_name']and year==a['source_year'],'Wrong historic alias source identity changed');canonical=public_row(c,'2025',a['canonical_key']);require(canonical['source_cell']==cell and canonical['category_literal']==year and canonical['value']==raw==D(a['value'])and canonical['unit']==a['unit']and canonical['series_literal']==a['canonical_label']and canonical['series_id']==a['canonical_series'],'Canonical model projection alias differs');require(vr==p['common_value_ref']and cr==p['common_year_ref']and nr==p['old_name_ref'],'Historic alias source reference changed');proof=dict(a,not_value_only_deduplication=True,wrong_historical_label_not_publicly_rehabilitated=True);aliases.append(('2025',a['original_key'],aid('2025',a['canonical_key']),dumps(proof)));retired.append(('2025',a['original_key'],'repaired_wrong_historical_alias_to_exact_existing_model_projection',aid('2025',a['canonical_key']),dumps(proof)))
 p=asset('gr40_wrong_object_registry_proposal.json');r.pdf('2025',p['pdf_original_page'],['GRÁFICO','40'])
 for a in p['canonical_published_points']:
  q=public_row(c,'2025',a['row_number']);raw=r.sources['2025'].number('Gráfico 40',a['source_cell']);require(q['source_cell']==a['source_cell']and q['category_literal']==a['category_literal']and q['value']==D(a['value'])==raw==r.independent('airef_dt_ingresos_2025.xlsx',p['source_dt_sheet'],a['source_cell']).number(),'Correct fiscal figure40/independent DT coverage differs');coverage.append(('2025','Gráfico 40',a['row_number'],p['pdf_original_page'],dumps(a)))
 for a in p['wrong_literal_chart_objects']:
  row=stage['2025',a['row_number']];require(all(row[k]==v for k,v in a.items()),'Bad figure40 original literal changed');s,sh,root,node,i=r.ser(row);lit=node.find('c:val/c:numLit',C);require(lit is not None and D(points(lit)[i])==D(row['decimal_XML_texto']),'Bad literal figure40 cache changed');retired.append(('2025',a['row_number'],'repaired_incompatible_unpublished_object_by_verified_figure_registry','published:2025:Gr40',dumps(dict(canonical_points=240,not_pointwise_mapping=True,physical_origin_unknown=True))))
 require(len(aliases)==48 and len(retired)==110 and len(coverage)==240,'Routing repair domain differs');return aliases,retired,coverage

def new_facts(c,r,stage):
 facts=[];retired=[];p=asset('re34_13_recovery_proposal.json');r.pdf('2025',110,['JUBILACIÓN','ANTICIPADA']);s=r.sources['2025'];sh='Gráfico RE 3.4';require(s.text(sh,'D7')=='JUBILACIÓN ANTICIPADA','Retirement physical metric header differs')
 for a in p['observations']:
  old=stage['2025',a['old_object_key']];vr,cell,raw=r.scalar(old);require(cell==a['physical_source_cell']and raw==D(a['raw'])and old['serie_literal']==a['original_label']and old['periodo_categoria_literal']==a['original_period'],'Early retirement original raw object changed');col=re.match('[A-Z]+',cell)[0];header=s.text(sh,col+'6');period=str(int(s.number(sh,col+'6')))if a['period_role']=='annual'else re.search(r'20[0-9]{2}',str(header))[0]+'-01';require((period==a['original_period']if a['period_role']=='annual'else re.sub(r'[\s/]+','',str(header))==re.sub(r'[\s/]+','',a['original_period']))and(raw*100).quantize(D('.1'))==D(a['pdf_printed_percent']),'Early retirement physical period/printed value differs');ident=f'physical:airef:2025:RE3.4:{cell}:early-retirement';proof=dict(a,source_variant='explicit_physical_header_and_published_pdf',raw_original_preserved=True,not_original_chart_series_rehabilitation=True,period_comparability=a['period_role'],metric_population='Seguridad Social aggregate new retirement awards',pdf_page=110)
  row=dict(old,row_number=None,id=ident);result=r.record(row,raw,raw*100,'percent_new_retirement_awards','Early retirement new awards as share of all new retirement awards','Jubilación anticipada',period,'published_early_retirement_award_share',proof,value_ref=vr,multiplier=D(100),binding_mode='new_physical_identity_after_wrong_chart_name_retirement');facts.append(result+(ident,));retired.append(('2025',old['row_number'],'replaced_wrong_retirement_series_by_new_physical_fact',ident,dumps(proof)))
 p=asset('gr30_2026_pdf6_recovery_proposal.json');report=PdfReader(ASSETS/Path(p['pdf_file']).name);page=report.pages[69].extract_text();require('GRÁFICO 30'in page and '2050'in page,'Published PDF debt decomposition context absent');slugs={'Gasto intereses':'interest-spending','Crecimiento Real':'real-growth','Inflación':'inflation','Déficit primario':'primary-deficit','Aj. stock flujo':'stock-flow-adjustment','Diferencia de la ratio a en 2050':'net-ratio-difference'}
 for a in p['observations']:
  ident='published-pdf:airef:2026:Gr30:2050:'+slugs[a['series_literal']]+':precision0.1';printed=D(a['value_printed']);require(a['value_printed'].replace('.',',')in page or a['value_printed']in page,'Published PDF printed value missing');proof=dict(a,source_pdf_file=str((ASSETS/Path(p['pdf_file']).name).relative_to(ROOT)),source_pdf_sha256=p['pdf_sha256'],source_pdf_page=70,source_render_file=str((ASSETS/Path(p['render_file']).name).relative_to(ROOT)),source_render_sha256=sha(ASSETS/Path(p['render_file']).name),precision='0.1',not_exact_unrounded_number=True,not_replacement_of_excel_raw=True,no_unlabelled_seventh_value_inferred=True,not_required_to_sum_without_printed_rounding=True)
  d=dict.fromkeys(FIELDS);d.update(book='2026',record_id=ident,block='Gráfico 30',metric_literal='Difference from Opinion2025 in debt/GDP ratio in2050',series_literal=a['series_literal'],category_literal='2050 vs Opinion2025',unit='percentage_points_GDP_debt_ratio',value=printed,raw_value=printed,rule_id='recovery3_published_pdf_display_fact_v1',quantity_role='published_pdf_numeric_label_at_one_decimal_precision',quantity_evidence=dumps(proof),supplemental_evidence=dumps(proof),binding_mode='visually_verified_pdf_label_mapping_not_excel_raw',category_semantics='projection_horizon2050_comparison_with2025_vintage');facts.append(tuple(d[n]for n in FIELDS)+(ident,))
 require(len(facts)==19 and len(retired)==13,'New source fact domain differs');return facts,retired

def private_and_diagram(r,stage):
 internal=[];nodes=[];p=asset('cuadro14_aux28_internal_proposal.json');s=r.sources['2025'];sh='Cuadro 14'
 for a in p['exact_published_formula_dependencies']:
  require(s.text(sh,a['published_cell'])==a['formula'],'Published formula dependency changed');require(abs(s.number(sh,a['published_cell'])-(s.number(sh,a['aux_scenario_input'])-s.number(sh,a['aux_baseline_input'])))<D('1e-12'),'Published/auxiliary dependency does not reconcile')
 for a in p['observations']:
  row=stage[a['book'],a['row_number']];raw=s.number(sh,a['source_cell']);require(row['hoja_valor']==sh and row['celda_valor']==a['source_cell']and raw==D(row['decimal_XML_texto'])==D(a['raw']),'Private auxiliary raw identity changed');proof=dict(a,actual_rate_kind=None,comparison_conflict_cause=None,not_published_baseline=True,not_unqualified_economic_forecast=True,no_stock_or_new_issuance_rate_claim=True);internal.append((a['book'],a['row_number'],a['source_cell'],raw,'percent',a['scenario_literal'],a['period_literal'],dumps(proof)))
 p=asset('gr98_diagram7_recovery_proposal.json')
 for a in p['points']:
  row=stage[a['book'],a['row_number']];s,sh,root,node,i=r.ser(row);require(i==a['point_index'],'Diagram original point index differs');coords=[]
  for container,key,cellkey in[('c:xVal','x','x_cell'),('c:yVal','y','y_cell'),('c:bubbleSize','bubble_size','bubble_cell')]:
   ref,cell,raw=r.bound(row,container);require(cell==a[cellkey]and raw==D(a[key]),'Diagram source coordinate/cache differs');coords.append(raw)
  require(s.text(sh,a['scenario_cell'])==a['scenario_literal']and coords[2]==40 and D(row['decimal_XML_texto'])==coords[1],'Diagram scenario/bubble geometry differs');proof=dict(a,quant_risk_interpretation_allowed=False,source_kind='qualitative_scenario_geometry',not_economic_numeric_observation=True,not_probability_or_percent=True,no_cardinality_from_bubble_size=True,no_year_inferred=True);nodes.append((a['book'],a['row_number'],a['scenario_literal'],*coords,dumps(proof)))
 require(len(internal)==28 and len(nodes)==7,'Private/qualitative domain differs');return internal,nodes

def existing_context(c,r):
 variants=[];withdrawals=[];overlays=[];p=asset('gr36_2026_calendar26_proposal.json')
 for a in p['existing_component_context_overlays']:
  row=public_row(c,a['book'],a['row_number']);require(row['source_cell']==a['source_cell']and row['series_id']==a['series_id']and row['category_literal']==a['category_literal']and row['value']==D(a['value']),'Existing ageing component source identity changed');proof=dict(source_variant_family='variant36',source_preference_required=True,calculation_safe=True,calculation_within_workbook=True,numeric_reconciliation_failed=False,pdf_comparison_confirmed=False,source_variant='official_workbook_snapshot_with_pdf_discrepancy',warning='Workbook component version conflicts with published report approximation; select attached_workbook explicitly',policy_withdrawal_not_data_error=True,old_quantity_evidence=row['quantity_evidence']);row.update(quantity_role='explicit_source_reported_variant_not_default_validated_quantity',quantity_evidence=dumps(proof));variants.append(tuple(row[n]for n in FIELDS)+('variant36',True,dumps(proof)));withdrawals.append((a['book'],a['row_number'],'existing_component_requires_explicit_workbook_source_selection',dumps(proof)))
 p=asset('iiee_26_recovery_proposal.json')
 for book,n,series,label,age,cell,value,unit in p['existing_admitted_figure_context_overlay']:
  row=public_row(c,book,n);require(row['source_cell']==cell and row['value']==D(value)and row['category_literal']==age and row['unit']==unit and row['series_id']==series,'Existing excise spending context overlay changed');require(r.independent('airef_dt_ingresos_2025.xlsx','Gráfico 12',cell).number()==D(value),'Existing excise expenditure independent DT match differs');overlays.append((book,n,dumps(dict(metric_literal='Average expenditure associated with excise and other product taxes by age',tax_scope='excise_taxes_and_other_product_taxes_as_defined_D214_excluding_VAT',wrong_original_IVA_scope_preserved=True,original_metric_literal=row['metric_literal'],numeric_value_not_changed=True))))
 require(len(variants)==104 and len(overlays)==13,'Existing context policy domain changed');return variants,withdrawals,overlays

def collect(c):
 guards(c);q=c.execute('SELECT * FROM staging_records');cols=[a[0]for a in q.description];stage={(p['book'],p['row_number']):p for p in[dict(zip(cols,x))for x in q.fetchall()]};selected=asset('selection.json');require(len(selected)==322 and {(p['book'],p['row_number'])for p in selected}==set(c.execute("SELECT book,row_number FROM recovery2_resolution WHERE state='blocked'").fetchall()),'Authorized exact322 pending domain changed');r=Resolver();rows=[];variants=[];reviews=[];states=[]
 try:
  for a in selected:
   row=stage[a['book'],a['row_number']];require(digest(row)==a['payload_sha256'],'Phase3 original universal payload changed');family=a['family'];p=r.proposals[a['proposal']];status='validated'
   try:
    if family in['excess','excise','smi']:rows.append(r.admitted(row,family,p));state='admitted_validated_typed_source'
    elif family.startswith('variant'):
     record=r.variant(row,family,p);variants.append(record);state='restricted_reported_workbook_variant_with_pdf_conflict'if record[-2]else'restricted_numerically_unreconciled_source_variant'
    else:state={'alias':'repaired_wrong_historical_alias','registry':'repaired_verified_figure_registry','early_physical':'replaced_by_new_physical_retirement_fact','internal':'validated_internal_auxiliary_input','diagram':'validated_qualitative_diagram_geometry'}[family]
   except(ValueError,KeyError,IndexError,AttributeError,TypeError,StopIteration,ArithmeticError)as e:status='review:'+str(e);state='blocked'
   reviews.append((a['book'],a['row_number'],family,status));states.append((a['book'],a['row_number'],state))
  aliases,retired,coverage=routing(c,r,stage);facts,earlyretired=new_facts(c,r,stage);internal,diagram=private_and_diagram(r,stage);oldvariants,withdrawals,overlays=existing_context(c,r)
  return dict(rows=rows,variants=variants+oldvariants,reviews=reviews,states=states,aliases=aliases,retired=retired+earlyretired,coverage=coverage,facts=facts,internal=internal,diagram=diagram,withdrawals=withdrawals,overlays=overlays)
 finally:r.close()

def views():
 return[
 ('recovery3_usable_keys','SELECT * FROM recovery2_usable_keys EXCEPT SELECT book,row_number FROM recovery3_policy_withdrawals UNION ALL SELECT book,row_number FROM recovery3_observations'),
 ('recovery3_public_observations',"SELECT p.* REPLACE(COALESCE(json_extract_string(m.evidence,'$.metric_literal'),p.metric_literal)AS metric_literal,CASE WHEN m.evidence IS NULL THEN p.quantity_evidence ELSE CAST(json_merge_patch(COALESCE(p.quantity_evidence,'{}'),m.evidence)AS VARCHAR)END AS quantity_evidence),m.evidence recovery3_metadata_overlay FROM recovery2_public_observations p LEFT JOIN recovery3_metadata_overlays m USING(book,row_number)WHERE p.row_number IS NULL OR NOT EXISTS(SELECT 1FROM recovery3_policy_withdrawals w WHERE w.book=p.book AND w.row_number=p.row_number) UNION ALL BY NAME SELECT o.*,'recovery3_observations' source_layer,'airef:'||book||':'||row_number observation_id FROM recovery3_observations o UNION ALL BY NAME SELECT f.*,'recovery3_new_physical_or_pdf_fact' source_layer FROM recovery3_new_facts f"),
 ('recovery3_resolution',"SELECT p.book,p.row_number,COALESCE(r.state,p.state)state,p.original_queue FROM recovery2_resolution p LEFT JOIN recovery3_states r USING(book,row_number)"),
 ('recovery3_outside_default',"SELECT s.*,COALESCE(r.state,w.reason)recovery3_state FROM staging_records s JOIN(SELECT book,row_number FROM recovery2_numeric_queue UNION SELECT book,row_number FROM recovery3_policy_withdrawals)old USING(book,row_number)LEFT JOIN recovery3_resolution r USING(book,row_number)LEFT JOIN recovery3_policy_withdrawals w USING(book,row_number)WHERE NOT EXISTS(SELECT 1FROM recovery3_usable_keys k WHERE k.book=s.book AND k.row_number=s.row_number)")]

def summary(c):
 counts={name:c.execute('SELECT count(*)FROM '+name).fetchone()[0]for name in['recovery3_observations','recovery3_source_variants','recovery3_new_facts','recovery3_aliases','recovery3_retired','recovery3_private_inputs','recovery3_diagram_nodes','recovery3_policy_withdrawals','recovery3_usable_keys','recovery3_public_observations','recovery3_outside_default']}
 return dict(counts=counts,resolution_counts=dict(c.execute('SELECT state,count(*)FROM recovery3_resolution GROUP BY state').fetchall()),unresolved_arithmetic_source_keys=c.execute('SELECT count(*)FROM recovery3_source_variants WHERE NOT calculation_safe').fetchone()[0],restricted_source_policy_pending=True,previous_sha256=BASE_SHA,previous_code_sha256=BASE_CODE_SHA,asset_manifest_sha256=ASSET_SHA,code_sha256=sha(Path(__file__)),full_user_goal_complete=False,rag_integrated=False)

def build(output):
 output=Path(output).resolve();require(output.is_relative_to(ROOT)and not output.exists(),'New project output directory required');require(sha(BASE)==BASE_SHA,'Frozen previous database changed');output.mkdir(parents=True);db=output/'source_recovery_phase3.duckdb';shutil.copyfile(BASE,db)
 with duckdb.connect(str(db))as c:
  data=collect(c)
  for name in['recovery3_observations']:c.execute('CREATE TABLE '+name+' AS SELECT * FROM source_recovery_observations WHERE FALSE');c.execute('ALTER TABLE '+name+' ADD PRIMARY KEY(book,row_number)')
  c.execute('CREATE TABLE recovery3_new_facts AS SELECT o.*,NULL::VARCHAR observation_id FROM source_recovery_observations o WHERE FALSE');c.execute('ALTER TABLE recovery3_new_facts ADD PRIMARY KEY(observation_id)');c.execute('CREATE TABLE recovery3_source_variants AS SELECT o.*,NULL::VARCHAR variant_family,NULL::BOOLEAN calculation_safe,NULL::VARCHAR variant_proof FROM source_recovery_observations o WHERE FALSE');c.execute('ALTER TABLE recovery3_source_variants ADD PRIMARY KEY(book,row_number)')
  schemas=[('recovery3_review','book VARCHAR,row_number INTEGER,family VARCHAR,status VARCHAR','reviews','book,row_number'),('recovery3_states','book VARCHAR,row_number INTEGER,state VARCHAR','states','book,row_number'),('recovery3_aliases','book VARCHAR,row_number INTEGER,canonical_observation_id VARCHAR,evidence VARCHAR','aliases','book,row_number'),('recovery3_retired','book VARCHAR,row_number INTEGER,reason VARCHAR,replacement_reference VARCHAR,evidence VARCHAR','retired','book,row_number'),('recovery3_publication_coverage','book VARCHAR,figure VARCHAR,row_number INTEGER,pdf_page INTEGER,evidence VARCHAR','coverage','book,figure,row_number'),('recovery3_private_inputs','book VARCHAR,row_number INTEGER,source_cell VARCHAR,value DECIMAL(38,24),unit VARCHAR,scenario VARCHAR,period VARCHAR,evidence VARCHAR','internal','book,row_number'),('recovery3_diagram_nodes','book VARCHAR,row_number INTEGER,scenario VARCHAR,x DECIMAL(38,24),y DECIMAL(38,24),bubble_size DECIMAL(38,24),evidence VARCHAR','diagram','book,row_number'),('recovery3_policy_withdrawals','book VARCHAR,row_number INTEGER,reason VARCHAR,evidence VARCHAR','withdrawals','book,row_number'),('recovery3_metadata_overlays','book VARCHAR,row_number INTEGER,evidence VARCHAR','overlays','book,row_number')]
  for name,schema,key,pk in schemas:c.execute('CREATE TABLE '+name+'('+schema+',PRIMARY KEY('+pk+'))')
  for name,key in[('recovery3_observations','rows'),('recovery3_source_variants','variants'),('recovery3_new_facts','facts')]+[(n,k)for n,s,k,p in schemas]:
   if data[key]:c.executemany('INSERT INTO '+name+' VALUES('+','.join('?'for _ in data[key][0])+')',data[key])
  c.execute('CREATE TABLE recovery3_primary_files(file VARCHAR PRIMARY KEY,path VARCHAR,url VARCHAR,sha256 VARCHAR)');c.executemany('INSERT INTO recovery3_primary_files VALUES(?,?,?,?)',[(p['file'],str((ASSETS/p['file']).relative_to(ROOT)),p.get('url'),p['sha256'])for p in asset('manifest.json')])
  for n,sql in views():c.execute('CREATE VIEW '+n+' AS '+sql)
  report=summary(c);report.update(status='built_pending_validation',review=c.execute("SELECT status,count(*)FROM recovery3_review WHERE status<>'validated'GROUP BY status").fetchall())
 report['db_sha256']=sha(db);(output/'manifest.json').write_text(dumps(report),encoding='utf8');return report

def check_new(c):
 data=collect(c)
 files={(p['file'],str((ASSETS/p['file']).relative_to(ROOT)),p.get('url'),p['sha256'])for p in asset('manifest.json')};require(set(c.execute('SELECT * FROM recovery3_primary_files').fetchall())==files,'Phase3 primary provenance catalogue changed')
 for name,key in[('recovery3_observations','rows'),('recovery3_source_variants','variants'),('recovery3_new_facts','facts'),('recovery3_review','reviews'),('recovery3_states','states'),('recovery3_aliases','aliases'),('recovery3_retired','retired'),('recovery3_publication_coverage','coverage'),('recovery3_private_inputs','internal'),('recovery3_diagram_nodes','diagram'),('recovery3_policy_withdrawals','withdrawals'),('recovery3_metadata_overlays','overlays')]:
  actual=c.execute('SELECT * FROM '+name).fetchall();require(len(actual)==len(data[key])and set(actual)==set(data[key]),'Phase3 independent source reconstruction differs: '+name)
 require(len(data['rows'])==122 and len(data['variants'])==146 and len(data['facts'])==19 and len(data['retired'])==123,'Phase3 approved source domain not fully reconstructed');require(c.execute('SELECT count(*)FROM recovery3_usable_keys').fetchone()[0]==42750 and c.execute('SELECT count(*)FROM recovery3_public_observations').fetchone()[0]==42870,'Phase3 default public counts differ');require(c.execute('SELECT count(*)FROM recovery3_outside_default').fetchone()[0]==4463,'Raw original default/outside partition differs');require(not c.execute('SELECT * FROM recovery3_usable_keys INTERSECT SELECT book,row_number FROM recovery3_source_variants').fetchall(),'Restricted variant leaked into default');require(not c.execute('SELECT * FROM recovery3_usable_keys INTERSECT SELECT book,row_number FROM recovery3_retired').fetchall(),'Wrong original object rehabilitated');require(not c.execute('SELECT * FROM recovery3_usable_keys INTERSECT SELECT book,row_number FROM recovery3_private_inputs UNION ALL SELECT * FROM recovery3_usable_keys INTERSECT SELECT book,row_number FROM recovery3_diagram_nodes').fetchall(),'Private/geometry data leaked into default');require(c.execute('SELECT count(*)FROM recovery3_source_variants WHERE NOT calculation_safe').fetchone()[0]==16,'Unreconciled source domain differs')
 for n,sql in views():require(not c.execute('SELECT * FROM '+n+' EXCEPT ALL ('+sql+')').fetchall()and not c.execute('('+sql+')EXCEPT ALL SELECT * FROM '+n).fetchall(),'Phase3 view changed: '+n)
 return summary(c)

def validate(db):
 db=Path(db);manifest=read(db.parent/'manifest.json');require(manifest['code_sha256']==sha(Path(__file__))and manifest['db_sha256']==sha(db),'Phase3 build data/code seal changed')
 previous=phase2.validate_previous(db)
 with duckdb.connect(str(db),read_only=True)as c:phase2.check_new(c);report=check_new(c)
 report.update(status='pass',previous_validation=previous['status'],phase2_layer_recomputed=True,db_sha256=sha(db));return report

def rowdict(q):
 names=[n[0]for n in q.description];rows=q.fetchall()
 if len(rows)!=1:raise LookupError('No unique explicitly selected source observation; no legacy fallback')
 return{n:str(v)if isinstance(v,D)else v for n,v in zip(names,rows[0])}

def enrich(c,d):
 if d.get('book'):
  s=c.execute('SELECT xlsx_path,xlsx_sha256,pdf_path,pdf_sha256 FROM sources WHERE book=?',[d['book']]).fetchone();d.update(source_xlsx_path=s[0],source_xlsx_sha256=s[1],source_report_pdf_path=s[2],source_report_pdf_sha256=s[3])
 return d

def query_observation(db,*,observation_id):
 require(isinstance(observation_id,str)and observation_id.strip(),'Exact observation identity required')
 with duckdb.connect(str(db),read_only=True)as c:
  result=enrich(c,rowdict(c.execute('SELECT * FROM recovery3_public_observations WHERE observation_id=?',[observation_id])));inherited=c.execute('SELECT count(*)FROM recovery2_public_observations WHERE observation_id=?',[observation_id]).fetchone()[0]
  overlay=c.execute('SELECT evidence FROM recovery3_metadata_overlays WHERE book=?AND row_number=?',[result.get('book'),result.get('row_number')]).fetchone()
  if overlay:result['recovery3_metadata_overlay']=json.loads(overlay[0]);result.update(result['recovery3_metadata_overlay'])
 if inherited:
  old=phase2.query_observation(db,observation_id=observation_id);old.update(result);result=old
 result['safe_recovery3_default_view']=True;return result

def query_safe(db,*,book,row_number):
 require(book in['2025','2026']and type(row_number)is int and row_number>0,'Exact source key required');return query_observation(db,observation_id=aid(book,row_number))

def query_as_reported(db,*,book,row_number,source_preference,allow_unreconciled):
 require(source_preference=='attached_workbook'and type(allow_unreconciled)is bool,'Explicit attached_workbook source preference and reconciliation policy required')
 with duckdb.connect(str(db),read_only=True)as c:
  d=enrich(c,rowdict(c.execute('SELECT * FROM recovery3_source_variants WHERE book=?AND row_number=?',[book,row_number])));proof=json.loads(d['variant_proof'])
  if not d['calculation_safe']and not allow_unreconciled:raise LookupError('Numerically unreconciled source requires explicit allow_unreconciled=True')
  d.update(source_record_id=aid(book,row_number),source_preference=source_preference,source_variant_proof=proof,warning=proof['warning'],default_economic_use_allowed=False,no_claim_pdf_and_workbook_are_same_numeric_version=True,calculation_scope='within_selected_workbook_only'if d['calculation_safe']else'prohibited_due_to_unresolved_arithmetic',coherent_calculation_within_workbook_only=bool(d['calculation_safe']));return d

def query_internal_input(db,*,book,row_number):
 with duckdb.connect(str(db),read_only=True)as c:
  d=enrich(c,rowdict(c.execute('SELECT * FROM recovery3_private_inputs WHERE book=?AND row_number=?',[book,row_number])));d.update(actual_rate_kind=None,comparison_conflict_cause=None,unqualified_economic_forecast_allowed=False,internal_only=True);return d

def query_diagram_node(db,*,book,row_number):
 with duckdb.connect(str(db),read_only=True)as c:
  d=enrich(c,rowdict(c.execute('SELECT * FROM recovery3_diagram_nodes WHERE book=?AND row_number=?',[book,row_number])));d.update(quant_risk_interpretation_allowed=False,coordinate_unit='source_diagram_coordinate_only',bubble_cardinality_interpretation_allowed=False,economic_forecast_allowed=False);return d

def explain_source_record(db,*,book,row_number):
 with duckdb.connect(str(db),read_only=True)as c:
  d=rowdict(c.execute('SELECT * FROM staging_records WHERE book=?AND row_number=?',[book,row_number]));d['current_default_usable']=bool(c.execute('SELECT count(*)FROM recovery3_usable_keys WHERE book=?AND row_number=?',[book,row_number]).fetchone()[0]);state=c.execute('SELECT state FROM recovery3_resolution WHERE book=?AND row_number=?',[book,row_number]).fetchone();d['recovery3_state']=state[0]if state else None;retired=c.execute('SELECT reason,replacement_reference,evidence FROM recovery3_retired WHERE book=?AND row_number=?',[book,row_number]).fetchone();d['replacement_reference']=retired[1]if retired else None;d['restricted_source_variant']=bool(c.execute('SELECT count(*)FROM recovery3_source_variants WHERE book=?AND row_number=?',[book,row_number]).fetchone()[0]);return d

def export_resolution(db):
 db=Path(db);certificate=read(db.parent/'validation.json');require(certificate['status']=='pass'and certificate['db_sha256']==sha(db),'Export requires certified current snapshot')
 with duckdb.connect(str(db),read_only=True)as c:
  q=c.execute("""SELECT r.book,r.row_number,r.state,r.state reason,CASE WHEN p.observation_id IS NOT NULL THEN 'current_default'WHEN v.book IS NOT NULL THEN 'query_as_reported'WHEN i.book IS NOT NULL THEN 'query_internal_input'WHEN d.book IS NOT NULL THEN 'query_diagram_node'WHEN t.book IS NOT NULL THEN 'replacement_reference'ELSE 'inherited_nondefault_disposition'END route,CASE WHEN v.book IS NOT NULL THEN 'attached_workbook'END source_selector,v.calculation_safe,COALESCE(v.variant_proof,t.evidence,i.evidence,d.evidence,p.quantity_evidence,old.evidence)supporting_evidence,t.replacement_reference,s.bloque original_block,s.serie_literal original_series,s.periodo_categoria_literal original_category,s.hoja_valor original_source_sheet,s.celda_valor original_source_cell,s.decimal_XML_texto original_numeric_text,COALESCE(p.metric_literal,v.metric_literal,target.metric_literal)normalized_metric,COALESCE(p.series_literal,v.series_literal,target.series_literal)normalized_series,COALESCE(p.category_literal,v.category_literal,target.category_literal)normalized_period,COALESCE(p.unit,v.unit,i.unit,target.unit)normalized_unit,COALESCE(p.value,v.value,i.value,target.value)normalized_value,x.xlsx_path,x.xlsx_sha256,x.pdf_path,x.pdf_sha256 FROM recovery3_resolution r JOIN staging_records s USING(book,row_number)LEFT JOIN recovery3_public_observations p USING(book,row_number)LEFT JOIN recovery3_source_variants v USING(book,row_number)LEFT JOIN recovery3_retired t USING(book,row_number)LEFT JOIN recovery3_private_inputs i USING(book,row_number)LEFT JOIN recovery3_diagram_nodes d USING(book,row_number)LEFT JOIN source_recovery_retired_objects old USING(book,row_number)LEFT JOIN recovery3_public_observations target ON target.observation_id=t.replacement_reference LEFT JOIN sources x ON x.book=r.book ORDER BY r.book,r.row_number""");names=[n[0]for n in q.description];records=[dict(zip(names,v))for v in q.fetchall()]
  q=c.execute('SELECT w.*,v.variant_family,v.calculation_safe,v.variant_proof FROM recovery3_policy_withdrawals w JOIN recovery3_source_variants v USING(book,row_number)ORDER BY book,row_number');names=[n[0]for n in q.description];withdrawals=[dict(zip(names,v))for v in q.fetchall()]
 require(len(records)==6426 and len(withdrawals)==104,'Resolution export domain changed')
 for name,value in[('resolution_export.json',records),('policy_withdrawals104_export.json',withdrawals)]:
  p=db.parent/name;payload=dumps(value);require(not p.exists()or p.read_text(encoding='utf8')==payload,'Existing resolution export differs');p.write_text(payload,encoding='utf8')
 return dict(resolution_records=len(records),policy_withdrawal_records=len(withdrawals))

def main():
 p=argparse.ArgumentParser();p.add_argument('command',choices=['build','validate']);p.add_argument('--output-dir');p.add_argument('--database');a=p.parse_args();report=build(a.output_dir)if a.command=='build'else validate(Path(a.database))
 if a.command=='validate':(Path(a.database).parent/'validation.json').write_text(dumps(report),encoding='utf8')
 print(dumps(report))
if __name__=='__main__':main()
