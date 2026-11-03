import unittest,json,tempfile,shutil,xml.etree.ElementTree as ET
from pathlib import Path
from decimal import Decimal
from datetime import datetime,timedelta
from collections import Counter
from unittest.mock import patch
import duckdb
from test_ready_recovery_semantics import XML,M
from experiments import remaining_group_semantics as m
from experiments.excel_pilot import BOOKS
DB=m.ROOT/'evaluation/remaining_group_semantics_2026-10-31/run_v5/remaining_groups.duckdb'
C={'c':'http://schemas.openxmlformats.org/drawingml/2006/chart'}
TARGETS=[('2025','chart116_s1'),('2025','chart117_s1'),('2025','chart54_s1'),('2025','chart54_s2'),('2025','chart54_s3'),*[( '2026','chart61_s'+str(i))for i in range(1,5)],*[( '2026','chart5_s'+str(i))for i in range(1,6)],('2026','chart7_s1'),('2026','chart7_s2')]
def optional(x,s,a):
 try:n=x.node(s,a)
 except KeyError:return None
 v=n.find('m:v',M)
 return x.value(s,a)if v is not None and v.text is not None or n.find('m:is',M)is not None else None
class TestRemainingGroups(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  if not DB.exists():raise RuntimeError('Build remaining group fixture first')
  with duckdb.connect(str(DB),read_only=True)as c:
   q=c.execute('SELECT * FROM remaining_observations ORDER BY book,row_number');names=[n[0]for n in q.description];cls.rows=[dict(zip(names,v))for v in q.fetchall()];q=c.execute('SELECT * FROM remaining_internal_observations ORDER BY book,row_number');names=[n[0]for n in q.description];cls.internal=[dict(zip(names,v))for v in q.fetchall()]
 def candidate(self,book,sid):
  with duckdb.connect(str(DB),read_only=True)as c:
   q=c.execute('SELECT * FROM family_numeric_queue WHERE book=?AND serie_id=?ORDER BY row_number LIMIT 1',[book,sid]);value=q.fetchone();self.assertIsNotNone(value);return dict(zip([n[0]for n in q.description],value))
 def input_data(self,c):
  q=c.execute('SELECT * FROM family_numeric_queue');names=[n[0]for n in q.description];return[dict(zip(names,v))for v in q.fetchall()]
 def test_independent_selection321_and_raw_xml_cache_values(self):
  with duckdb.connect(str(m.PREVIOUS),read_only=True)as c:
   expected=set()
   for book,sid in TARGETS:expected.update(c.execute('SELECT book,row_number FROM family_numeric_queue WHERE book=?AND serie_id=?',[book,sid]).fetchall())
  self.assertEqual(len(expected),321);self.assertEqual({(r['book'],r['row_number'])for r in self.rows},expected);readers={b:XML(m.ROOT/BOOKS[b][0])for b in['2025','2026']}
  try:
   for r in self.rows:
    x=readers[r['book']];s=r['source_sheet'];raw=x.value(s,r['source_cell']);self.assertEqual(raw,r['raw_value']);factor=100 if s=='Gráfico 5'else 1;self.assertEqual(r['multiplier'],factor);self.assertEqual(r['divisor'],1);self.assertEqual(raw*factor,r['value']);node=ET.fromstring(x.z.read(r['chart_part'])).findall('.//c:ser',C)[int(r['series_id'].split('_s')[1])-1];self.assertEqual(node.find('c:val/c:numRef/c:f',C).text,r['value_ref']);vp={int(n.get('idx')):n.find('c:v',C).text for n in node.findall('c:val/c:numRef/c:numCache/c:pt',C)};self.assertEqual(Decimal(vp[r['point_index']]),raw);ns,nc=m.ref_cells(r['name_ref']);self.assertEqual(x.value(ns,nc[0]),r['series_literal'])
    if r['series_id']=='chart7_s2':self.assertIsNone(r['universal_series']);self.assertEqual(r['series_literal'],'Histórico')
    else:self.assertEqual(r['series_literal'],r['universal_series'])
  finally:
   for x in readers.values():x.z.close()
 def test_dates_are_explicit_same_row_serial_style_epoch_not_calendar_fill(self):
  x=XML(m.ROOT/BOOKS['2025'][0])
  try:
   wb=ET.fromstring(x.z.read('xl/workbook.xml'));pr=wb.find('m:workbookPr',M);self.assertNotIn(pr.get('date1904')if pr is not None else None,['1','true']);styles=ET.fromstring(x.z.read('xl/styles.xml'));cellstyles=list(styles.find('m:cellXfs',M));custom={int(n.get('numFmtId')):n.get('formatCode')for n in styles.findall('m:numFmts/m:numFmt',M)}
   rows=[r for r in self.rows if r['block']=='Gráfico 67'];self.assertEqual(len(rows),118)
   for r in rows:
    i=r['point_index'];self.assertTrue(555<=i<=613);cell='B'+str(7+i);serial=x.value('Gráfico 67',cell);self.assertEqual(serial,int(serial));actual=datetime(1899,12,30)+timedelta(days=int(serial));self.assertEqual(r['category_literal'],str(actual));self.assertIsNone(r['universal_category']);p=json.loads(r['quantity_evidence']);self.assertEqual(p['excel_serial'],str(serial));self.assertEqual(p['physical_date'],actual.isoformat());self.assertEqual(p['original_category_ref'],"'Gráfico 67'!$B$7:$B$561");self.assertEqual(r['unit'],'percent_ten_year_sovereign_bond_yield');style=cellstyles[int(x.node('Gráfico 67',cell).get('s','0'))];fmt=int(style.get('numFmtId'));self.assertTrue(fmt in range(14,23)or fmt in custom);self.assertIn('date',p['rule'])
   self.assertEqual(min(r['category_literal']for r in rows),'2022-10-28 00:00:00');self.assertEqual(max(r['category_literal']for r in rows),'2023-01-18 00:00:00')
  finally:x.z.close()
 def test_logit_source_hierarchy_leaf_literals_groups_and_labelled_sum(self):
  x=XML(m.ROOT/BOOKS['2025'][0]);groups={3:'Formación',4:'Formación',5:'Edad (años)',6:'Edad (años)',7:'Edad (años)',8:'Edad (años)',9:'Duración desempleo',10:'Duración desempleo'}
  try:
   rows=[r for r in self.rows if r['block']=='Gráfico 29'];self.assertEqual(len(rows),42)
   for r in rows:
    i=r['point_index'];leaf=x.value('Gráfico 29','C'+str(5+i));outer=optional(x,'Gráfico 29','B'+str(5+i));literal=leaf if r['series_id']=='chart54_s3'else(outer+' | 'if outer is not None else'')+leaf;self.assertEqual(r['category_literal'],literal);self.assertEqual(r['universal_category'],literal.replace('\n',' / '));q=m.query(DB,book='2025',block='Gráfico 29',series_id=r['series_id'],point_index=i);self.assertEqual(q['group'],groups.get(i));self.assertEqual(q['contrast'],leaf);p=json.loads(r['quantity_evidence']);self.assertEqual(p['raw_outer_literal'],outer);self.assertEqual(r['unit'],'literal_model_coefficient_as_labelled')
   for rr in range(5,19):self.assertLessEqual(abs(x.value('Gráfico 29','F'+str(rr))-x.value('Gráfico 29','D'+str(rr))-x.value('Gráfico 29','E'+str(rr))),Decimal('1e-14'))
   self.assertEqual(sum(r['quantity_role']=='logit_total_as_labelled_not_additional_component'for r in rows),14)
  finally:x.z.close()
 def test_fertility_period_vs_cohort_coordinates_and_na_domains(self):
  x=XML(m.ROOT/BOOKS['2026'][0]);domains={1:range(16),2:range(16,51),3:range(35),4:range(34,51)}
  try:
   self.assertEqual(x.node('Anexo I. Gr 8B','B21').find('m:f',M).text,'1960+30');rows=[r for r in self.rows if r['block']=='Anexo I. Gr 8B'];self.assertEqual(len(rows),103)
   for r in rows:
    i=r['point_index'];sid=int(r['series_id'].split('_s')[1]);year=int(x.value('Anexo I. Gr 8B','B'+str(21+i)));self.assertIn(i,domains[sid]);self.assertEqual(r['category_literal'],str(year));self.assertEqual(r['universal_category'],str(year));p=json.loads(r['quantity_evidence'])['time_coordinate'];self.assertEqual(p['plot_year'],year)
    if sid in[1,2]:self.assertEqual(p['cohort_birth_year'],year-30);self.assertIsNone(p['period_year']);self.assertIn('CCF',r['metric_literal'])
    else:self.assertEqual(p['period_year'],year);self.assertIsNone(p['cohort_birth_year']);self.assertIn('ICF',r['metric_literal'])
   overlap=[r for r in rows if r['category_literal']=='2024'and r['series_id']in['chart61_s3','chart61_s4']];self.assertEqual(len(overlap),2);self.assertEqual({r['series_literal']for r in overlap},{'ICF observado','ICF proyectado'})
   for cell in['E56','F21']:self.assertEqual(x.node('Anexo I. Gr 8B',cell).get('t'),'e');self.assertEqual(x.node('Anexo I. Gr 8B',cell).find('m:v',M).text,'#N/A')
  finally:x.z.close()
 def test_growth_components_total_percent_and_generosity_2025_bridge(self):
  x=XML(m.ROOT/BOOKS['2026'][0])
  try:
   rows=[r for r in self.rows if r['block']=='Gráfico 5'];self.assertEqual(len(rows),21);self.assertEqual(sum(r['series_literal']=='Medidas'for r in rows),1)
   for r in rows:self.assertEqual(r['unit'],'percent_annual_pension_expenditure_growth'if r['series_literal']=='TOTAL'else'percentage_points_annual_pension_expenditure_growth')
   for col in'CDEFG':
    formula=x.node('Gráfico 5',col+'11').find('m:f',M)
    if formula.text is None:
     master=x.node('Gráfico 5','E11').find('m:f',M);self.assertIn(col,'FG');self.assertEqual(formula.attrib,{'t':'shared','si':'0'});self.assertEqual(master.attrib,{'t':'shared','ref':'E11:K11','si':'0'});self.assertEqual(master.text,'SUM(E7:E10)');resolved='SUM('+col+'7:'+col+'10)'
    else:resolved=formula.text
    self.assertEqual(resolved,'SUM('+col+'7:'+col+'10)');values=[optional(x,'Gráfico 5',col+str(rr))for rr in range(7,11)];self.assertLessEqual(abs(sum((v for v in values if v is not None),Decimal(0))-x.value('Gráfico 5',col+'11')),Decimal('1e-14'))
   bridge=[r for r in self.rows if r['block']=='Gráfico 7'and r['category_literal']=='2025'];self.assertEqual(len(bridge),2);self.assertEqual({r['series_literal']for r in bridge},{'Histórico','AIReF 2026 '});self.assertEqual(x.value('Gráfico 7','M6'),x.value('Gráfico 7','M8'));self.assertTrue(all(r['unit']=='percent_pension_average_over_average_wage'for r in bridge))
  finally:x.z.close()
 def test_internal168_xml_headers_scenarios_periods_and132_formulas(self):
  x=XML(m.ROOT/BOOKS['2025'][0]);formulas=0;roles=Counter()
  try:
   self.assertEqual(len(self.internal),168)
   for r in self.internal:
    s=r['source_sheet'];raw=x.value(s,r['source_cell']);self.assertEqual(raw,r['raw_value']);self.assertEqual(raw,r['value']);self.assertEqual(x.value(s,r['metric_cell']),r['metric']);self.assertEqual(x.value(s,r['scenario_cell']),r['scenario']);self.assertEqual(x.value(s,r['period_cell']),r['period_literal']);self.assertIn('internal_model_input',r['scope']);roles[r['value_role']]+=1;self.assertEqual(r['universal_metric'],r['metric']);self.assertEqual(r['universal_series'],r['scenario']);self.assertEqual(r['universal_category'],r['period_literal']);base=x.value(s,r['base_input_cell'])
    if r['published_formula']:
     formulas+=1;self.assertEqual(x.node(s,r['published_cell']).find('m:f',M).text,r['published_formula'][1:]);pub=x.value(s,r['published_cell']);self.assertEqual(pub,r['published_raw_value']);expected=raw if r['value_role']=='base_period_average'else raw-base;self.assertLessEqual(abs(pub-expected),Decimal('1e-14'));self.assertIsNotNone(r['published_row_number'])
    else:self.assertEqual(raw,base);self.assertEqual(x.value(s,r['published_cell']),'..');self.assertIsNone(r['published_raw_value']);self.assertIsNone(r['published_row_number'])
    self.assertEqual(r['unit'],'percent_annual_change_as_labelled'if s in['Cuadro 12','Cuadro 13','Cuadro 15']else'percent_gdp')
   self.assertEqual(formulas,132);self.assertEqual(sorted(roles.values()),[24,36,108])
  finally:x.z.close()
 def test_internal_query_is_exact_and_never_publicly_admitted(self):
  for r in self.internal:
   q=m.query_internal(DB,book=r['book'],block=r['block'],scenario=r['scenario'],period=r['period_literal']);self.assertEqual(q['row_number'],r['row_number']);self.assertEqual(Decimal(q['raw_value']),r['raw_value'])
  r=self.internal[0]
  with self.assertRaises(LookupError):m.query_internal(DB,book=r['book'],block=r['block'],scenario=r['scenario']+' ',period=r['period_literal'])
  with self.assertRaises(LookupError):m.query_internal(DB,book=r['book'],block=r['block'],scenario=r['scenario'],period='2025-2030')
  with self.assertRaises(ValueError):m.query_internal(DB,book='2025',block='Cuadro 14',scenario='Base',period=r['period_literal'])
  with self.assertRaises(LookupError):m.query(DB,book='2025',block=r['block'],series_id='Base',point_index=0)
  with duckdb.connect(str(DB),read_only=True)as c:self.assertFalse(c.execute('SELECT * FROM remaining_internal_keys INTERSECT SELECT * FROM remaining_usable_keys').fetchall());self.assertEqual(c.execute("SELECT count(*)FROM remaining_numeric_queue WHERE remaining_reason='internal_only_scope'").fetchone()[0],168)
 def test_full_layer_partition_and_legacy_fingerprints(self):
  with duckdb.connect(str(m.PREVIOUS),read_only=True)as a,duckdb.connect(str(DB),read_only=True)as b:
   self.assertEqual(len(m.check_new(b)),321)
   for name in['family_observations','family_sources','family_usable_keys','pending_family_diagnostics','pending_family_source_incidents','next_deferrals','ready_quarantine']:
    aq=a.execute('SELECT * FROM '+name);schema=aq.description;values=aq.fetchall();bq=b.execute('SELECT * FROM '+name);self.assertEqual(schema,bq.description,name);self.assertEqual(Counter(values),Counter(bq.fetchall()),name)
   self.assertEqual(b.execute('SELECT count(*)FROM remaining_usable_keys').fetchone()[0],41242);self.assertEqual(b.execute('SELECT count(*)FROM remaining_numeric_queue').fetchone()[0],5971);self.assertFalse(b.execute('SELECT book,row_number FROM remaining_numeric_queue INTERSECT SELECT * FROM remaining_usable_keys').fetchall());self.assertFalse(b.execute('SELECT book,row_number FROM ready_quarantine INTERSECT SELECT * FROM remaining_usable_keys').fetchall());self.assertFalse(b.execute("SELECT r.book,r.row_number FROM remaining_observations r JOIN family_numeric_queue p USING(book,row_number)WHERE p.family_reason IN('drawing_helper_not_economic_series','forecast_marker_not_measure')").fetchall());self.assertFalse(b.execute("SELECT n.book,n.row_number FROM remaining_numeric_queue n JOIN family_numeric_queue p USING(book,row_number)WHERE n.remaining_reason IS DISTINCT FROM p.family_reason AND n.remaining_reason<>'internal_only_scope'").fetchall())
 def test_dates_reference_epoch_style_serial_and_cache_fail_closed(self):
  row=self.candidate('2025','chart116_s1')
  for kind in['reference','style','serial','epoch','cache']:
   r=m.RemainingResolver()
   try:
    self.assertIsNotNone(r.review_remaining(row)[1]);s=r.sources['2025'];part=row['parte_grafico_imagen'];root=ET.fromstring(s.xml.parts[part]);node=root.findall('.//c:ser',C)[0]
    if kind=='reference':node.find('c:cat/c:numRef/c:f',C).text="'Gráfico 67'!$B$7:$B$562"
    elif kind=='style':s.sheet('Gráfico 67')[0]['B562'].set('s','0')
    elif kind=='serial':s.sheet('Gráfico 67')[0]['B562'].find('m:v',M).text='45000'
    elif kind=='epoch':s.workbook.epoch=datetime(1904,1,1)
    else:next(n for n in node.findall('c:val/c:numRef/c:numCache/c:pt',C)if n.get('idx')=='555').find('c:v',C).text='999'
    s.xml.parts[part]=ET.tostring(root);r.bindings.clear();self.assertIsNone(r.review_remaining(row)[1],kind)
   finally:r.close()
 def test_logit_static_hierarchy_and_total_leaf_checksum_fail_closed(self):
  row=self.candidate('2025','chart54_s1')
  for kind in['outercache','physicalblank','checksum','totalleaf']:
   r=m.RemainingResolver()
   try:
    self.assertIsNotNone(r.review_remaining(row)[1]);s=r.sources['2025'];part='xl/charts/chart54.xml';root=ET.fromstring(s.xml.parts[part])
    if kind=='outercache':root.findall('.//c:ser',C)[0].findall('c:cat/c:multiLvlStrRef/c:multiLvlStrCache/c:lvl',C)[1].findall('c:pt',C)[3].find('c:v',C).text='Otrogrupo'
    elif kind=='physicalblank':s.sheet('Gráfico 29')[1]['B9']='Formación'
    elif kind=='checksum':
     s.sheet('Gráfico 29')[0]['F6'].find('m:v',M).text='999'
     with self.assertRaises(ValueError):r.logit_sum_proof()
    else:row2=self.candidate('2025','chart54_s3');self.assertIsNotNone(r.review_remaining(row2)[1]);root.findall('.//c:ser',C)[2].find('c:cat/c:strRef/c:strCache/c:pt/c:v',C).text='Otrocontraste'
    s.xml.parts[part]=ET.tostring(root);r.bindings.clear();self.assertIsNone(r.review_remaining(row2 if kind=='totalleaf'else row)[1],kind)
   finally:r.close()
 def test_fertility_axis_formula_domain_and_growth_total_fail_closed(self):
  row=self.candidate('2026','chart61_s2');r=m.RemainingResolver()
  try:
   self.assertIsNotNone(r.review_remaining(row)[1]);r.sources['2026'].sheet('Anexo I. Gr 8B')[1]['B21']='=1960+31';self.assertIsNone(r.review_remaining(row)[1])
  finally:r.close()
  row=self.candidate('2026','chart61_s4');r=m.RemainingResolver()
  try:
   self.assertIsNotNone(r.review_remaining(row)[1]);source=r.sources['2026'];cell=source.sheet('Anexo I. Gr 8B')[0]['F21'];self.assertEqual(cell.get('t'),'e');cell.set('t','n');cell.find('m:v',M).text='0';part='xl/charts/chart61.xml';root=ET.fromstring(source.xml.parts[part]);node=root.findall('.//c:ser',C)[3];point=next(n for n in node.findall('c:val/c:numRef/c:numCache/c:pt',C)if n.get('idx')=='0');self.assertEqual(point.find('c:v',C).text,'#N/A');point.find('c:v',C).text='0';source.xml.parts[part]=ET.tostring(root);r.bindings.clear();status,result=r.review_remaining(row);self.assertIsNone(result);self.assertIn('Fertility observed/projected domain changed',status)
  finally:r.close()
  row=self.candidate('2026','chart5_s1');r=m.RemainingResolver()
  try:
   self.assertIsNotNone(r.review_remaining(row)[1]);r.sources['2026'].sheet('Gráfico 5')[0]['C11'].find('m:v',M).text='999'
   with self.assertRaises(ValueError):r.pension_growth_proof()
  finally:r.close()
 def test_internal_formula_scenario_period_unit_and_raw_fail_closed(self):
  for kind in['formula','scenario','period','unit','raw']:
   r=m.RemainingResolver()
   try:
    with duckdb.connect(str(DB),read_only=True)as c:
     data=self.input_data(c);self.assertEqual(len(m.internal_rows(c,r,data)),168);s=r.sources['2025'];sheet='Cuadro 12'
     if kind=='formula':s.sheet(sheet)[1]['E8']='=J8+J7'
     elif kind=='scenario':s.sheet(sheet)[1]['C8']='Inventado'
     elif kind=='period':s.sheet(sheet)[1]['J6']='Promedio 2024-2030'
     elif kind=='unit':s.sheet(sheet)[1]['I5']='PIB real (%PIB)'
     else:s.sheet(sheet)[0]['J8'].find('m:v',M).text='999'
     with self.assertRaises(ValueError,msg=kind):m.internal_rows(c,r,data)
   finally:r.close()
 def test_public_exact_query_no_fallback_and_tampering_detected(self):
  for row in self.rows:self.assertEqual(m.query(DB,book=row['book'],block=row['block'],series_id=row['series_id'],point_index=row['point_index'])['row_number'],row['row_number'])
  with self.assertRaises(ValueError):m.query(DB,book='2025',block='Gráfico 67',series_id='chart116_s1',point_index=True)
  with self.assertRaises(LookupError):m.query(DB,book='2026',block='Gráfico 7',series_id='chart7_s2',point_index=35)
  with patch.object(m,'PREVIOUS_SHA','0'*64):
   with duckdb.connect(str(DB),read_only=True)as c:
    with self.assertRaises(ValueError):m.collect(c)
  with tempfile.TemporaryDirectory(dir=m.ROOT/'tmp')as directory:
   db=Path(directory)/'bad.duckdb';shutil.copyfile(DB,db)
   with duckdb.connect(str(db))as c:c.execute('UPDATE remaining_internal_observations SET value=value+1 WHERE row_number=(SELECT min(row_number)FROM remaining_internal_observations)')
   with duckdb.connect(str(db),read_only=True)as c:
    with self.assertRaises(ValueError):m.check_new(c)
   shutil.copyfile(DB,db)
   with duckdb.connect(str(db))as c:c.execute('CREATE OR REPLACE VIEW remaining_usable_keys AS SELECT * FROM family_usable_keys UNION ALL SELECT * FROM remaining_internal_keys')
   with duckdb.connect(str(db),read_only=True)as c:
    with self.assertRaises(ValueError):m.check_new(c)
if __name__=='__main__':unittest.main()
