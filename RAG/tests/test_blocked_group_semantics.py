import unittest,json,tempfile,shutil,xml.etree.ElementTree as ET
from pathlib import Path
from decimal import Decimal
from unittest.mock import patch
import duckdb
from pypdf import PdfReader
from test_ready_recovery_semantics import XML,M
from experiments import blocked_group_semantics as m
from experiments.excel_pilot import BOOKS
DB=m.ROOT/'evaluation/blocked_group_semantics_2026-10-31/run_v4/blocked_groups.duckdb'
C={'c':'http://schemas.openxmlformats.org/drawingml/2006/chart'}
class TestBlockedGroups(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  if not DB.exists():raise RuntimeError('Build blocked group fixture before running tests')
  with duckdb.connect(str(DB),read_only=True) as c:
   q=c.execute('SELECT * FROM blocked_observations ORDER BY book,row_number');ns=[x[0] for x in q.description];cls.rows=[dict(zip(ns,v)) for v in q.fetchall()]
 def candidate(self,book,block,sid=None):
  with duckdb.connect(str(DB),read_only=True) as c:
   q=c.execute('SELECT * FROM ready_numeric_queue WHERE book=? AND bloque=? '+('AND serie_id=? ' if sid else 'AND (book,row_number) IN (SELECT book,row_number FROM blocked_observations) ')+'ORDER BY row_number LIMIT 1',[book,block]+([sid] if sid else []));v=q.fetchone();self.assertIsNotNone(v);return dict(zip([x[0] for x in q.description],v))
 def test_independent_xml_all1387_exact_values_categories_series_and_scales(self):
  self.assertEqual(len(self.rows),1387);xs={b:XML(m.ROOT/BOOKS[b][0]) for b in ['2025','2026']}
  try:
   for r in self.rows:
    x=xs[r['book']];sheet=r['source_sheet'];raw=x.value(sheet,r['source_cell']);self.assertEqual(raw,r['raw_value']);factor=100 if sheet=='Gráfico 22' else 1;self.assertEqual(raw*factor,r['value']);self.assertEqual(r['multiplier'],factor);self.assertEqual(r['divisor'],1)
    root=ET.fromstring(x.z.read(r['chart_part']));node=root.findall('.//c:ser',C)[int(r['series_id'].split('_s')[1])-1];self.assertEqual(node.find('c:val/c:numRef/c:f',C).text,r['value_ref']);sheet2,cells=m.ref_cells(r['category_ref']);v=x.value(sheet2,cells[r['point_index']]);category=str(int(v)) if isinstance(v,Decimal) and v==int(v) else str(v);self.assertEqual(category,r['category_literal']);self.assertEqual(r['category_literal'],r['universal_category'])
    if r['universal_series'] is None:
     self.assertTrue(r['name_ref'].startswith('source_headers:'));headers=json.loads(r['name_ref'].split(':',1)[1]);self.assertEqual(' | '.join(x.value(sheet,a) for a,_ in headers),r['series_literal'])
     if sheet.startswith('Anexo'):
      self.assertEqual(r['series_literal'],{'chart49_s3':'Países emisores | Mínimo','chart49_s4':'Mundial | Mínimo','chart50_s2':'Mínimo UE'}[r['series_id']]);self.assertIn('Mínimo',r['series_literal']);self.assertEqual(r['unit'],'children_per_woman');self.assertEqual(r['quantity_role'],'minimum_across_country_group_period_fertility_icf');self.assertNotIn('cohort',r['category_semantics'])
     else:self.assertEqual(r['unit'],'percent_population_age_16_74_activity_rate');self.assertIsNone(x.node(sheet,'C4').find('m:v',M)) if 'C4' in x.cache[sheet] else None
    else:
     ns,nc=m.ref_cells(r['name_ref']);self.assertEqual(x.value(ns,nc[0]),r['series_literal']);self.assertEqual(r['series_literal'],r['universal_series'])
    if sheet=='Gráfico 100':self.assertEqual(r['unit'],'anomaly_degrees_celsius_global_vs_preindustrial')
    if sheet=='Gráfico 124':self.assertEqual(r['unit'],'percent_gdp')
  finally:
   for x in xs.values():x.z.close()
 def test_preserved_partition_quarantine_and_full_new_recalculation(self):
  with duckdb.connect(str(DB),read_only=True) as c:
   self.assertEqual(len(m.check_new(c)),1387);self.assertEqual(c.execute('SELECT count(*) FROM blocked_usable_keys').fetchone()[0],39879);self.assertEqual(c.execute('SELECT count(*) FROM blocked_numeric_queue').fetchone()[0],7334)
   self.assertFalse(c.execute('SELECT * FROM ready_usable_keys EXCEPT ALL SELECT * FROM blocked_usable_keys').fetchall());self.assertFalse(c.execute('SELECT book,row_number FROM ready_quarantine INTERSECT SELECT * FROM blocked_usable_keys').fetchall());self.assertFalse(c.execute('SELECT book,row_number FROM blocked_numeric_queue INTERSECT SELECT * FROM blocked_usable_keys').fetchall())
   self.assertEqual(c.execute('SELECT count(*) FROM ready_quarantine').fetchone()[0],13)
   self.assertFalse(c.execute("SELECT n.book,n.row_number FROM blocked_numeric_queue n JOIN ready_numeric_queue p USING(book,row_number) WHERE n.blocked_reason IS DISTINCT FROM p.ready_reason AND n.blocked_reason<>'literal_series_source_conflict'").fetchall())
 def test_independent_minimum_merges_and_physical_identity(self):
  x=XML(m.ROOT/BOOKS['2026'][0])
  try:
   for sheet,merges in [('Anexo I. Gr 2A',['D20:E20','F20:G20']),('Anexo I. Gr 2B',[])]:
    actual=[n.get('ref') for n in ET.fromstring(x.z.read(x.paths[sheet])).findall('m:mergeCells/m:mergeCell',M)];self.assertTrue(set(merges)<=set(actual));self.assertEqual(x.value(sheet,'B19'),'Número de hijos por mujer')
   self.assertEqual(x.value('Anexo I. Gr 2A','E21'),'Mínimo');self.assertEqual(x.value('Anexo I. Gr 2A','G21'),'Mínimo');self.assertEqual(x.value('Anexo I. Gr 2B','E20'),'Mínimo UE')
  finally:x.z.close()
 def test_technical_pdf_proof_is_pinned_and_independent(self):
  self.assertEqual(m.sha(m.CLIMATE_PDF),'de85002efac3d0bf1d8698b123506a02376917be1bbd2e006f131f71f6e20f26');r=PdfReader(m.CLIMATE_PDF)
  try:
   self.assertEqual(len(r.pages),40);t=' '.join(r.pages[13].extract_text().split());self.assertIn('grados centígrados',t);self.assertIn('temperatura media anual mundial',t);self.assertIn('preindustrial',t);self.assertIn('INVERSIÓN PÚBLICA (% PIB)',r.pages[32].extract_text())
  finally:r.close()
  with duckdb.connect(str(DB),read_only=True) as c:self.assertEqual(c.execute('SELECT page FROM blocked_external_evidence ORDER BY page').fetchall(),[(4,),(14,),(33,)])
 def test_minimum_headers_merges_and_universal_null_fail_closed(self):
  row=self.candidate('2026','Anexo I. Gr 2A','chart49_s3')
  for kind in ['header','merge','name','category']:
   r=m.BlockedResolver()
   try:
    self.assertIsNotNone(r.review_blocked(row)[1]);src=r.sources['2026'];sheet=row['bloque']
    if kind=='header':src.sheet(sheet)[1]['G21']='Máximo'
    elif kind=='name':row2=dict(row,serie_literal='Inventado')
    elif kind=='category':src.sheet(sheet)[0]['B22'].find('m:v',M).text='1900'
    else:
     path=src.xml.sheets[sheet];root=ET.fromstring(src.xml.parts[path]);parent=root.find('m:mergeCells',M)
     for n in list(parent):
      if n.get('ref')=='F20:G20':parent.remove(n)
     src.xml.parts[path]=ET.tostring(root)
    r.bindings.clear();self.assertIsNone(r.review_blocked(row2 if kind=='name' else row)[1],kind)
   finally:r.close()
 def test_climate_refs_axes_values_caption_and_external_source_fail_closed(self):
  row=self.candidate('2025','Gráfico 100','chart156_s1')
  for kind in ['ref','axis','value','caption','external']:
   r=m.BlockedResolver()
   try:
    self.assertIsNotNone(r.review_blocked(row)[1]);src=r.sources['2025'];part=row['parte_grafico_imagen'];root=ET.fromstring(src.xml.parts[part]);node=root.findall('.//c:ser',C)[0]
    if kind=='ref':node.find('c:val/c:numRef/c:f',C).text="'Gráfico 100'!$I$6:$I$46"
    elif kind=='axis':root.find('.//c:valAx/c:numFmt',C).set('formatCode','0%')
    elif kind=='value':src.sheet(row['bloque'])[0][row['celda_valor']].find('m:v',M).text='999'
    elif kind=='caption':src.sheet(row['bloque'])[1]['B3']='Temperatura absoluta'
    src.xml.parts[part]=ET.tostring(root);r.bindings.clear()
    if kind=='external':
     with patch.object(m,'CLIMATE_SHA','0'*64):self.assertIsNone(r.review_blocked(row)[1])
    else:self.assertIsNone(r.review_blocked(row)[1],kind)
   finally:r.close()
 def test_conflicting_generosity_and_delayed_scenarios_stay_pending(self):
  r=m.BlockedResolver()
  try:
   self.assertIsNotNone(r.review_blocked(self.candidate('2025','Gráfico 48','chart85_s2'))[1]);self.assertEqual(r.review_blocked(self.candidate('2025','Gráfico 48','chart85_s1'))[0],'literal_series_source_conflict');self.assertIsNone(r.review_blocked(self.candidate('2025','Gráfico 48','chart85_s3'))[1])
   for sid in ['chart156_s4','chart157_s4']:self.assertIsNone(r.review_blocked(self.candidate('2025','Gráfico 100',sid))[1])
   with duckdb.connect(str(DB),read_only=True) as c:self.assertEqual(c.execute('SELECT count(*) FROM blocked_binding_evidence').fetchone()[0],48)
  finally:r.close()
 def test_gr22_caption_fraction_scaling_and_metadata(self):
  row=self.candidate('2025','Gráfico 22','chart43_s1');r=m.BlockedResolver()
  try:
   status,v=r.review_blocked(row);self.assertIsNotNone(v);d=dict(zip(m.FIELDS,v));self.assertIsNone(d['universal_series']);self.assertEqual(d['value'],d['raw_value']*100);self.assertEqual(d['series_literal'],m.GR22);self.assertNotIn('forecast',d['semantic_scope']);r.sources['2025'].sheet('Gráfico 22')[1]['C4']='Otra serie';r.bindings.clear();self.assertIsNone(r.review_blocked(row)[1])
  finally:r.close()
 def test_query_exact_group_and_point_no_fallback(self):
  seen=set()
  for row in self.rows:
   key=row['book'],row['block'],row['series_id']
   if key in seen:continue
   seen.add(key);self.assertEqual(m.query(DB,book=key[0],block=key[1],series_id=key[2],point_index=row['point_index'])['row_number'],row['row_number'])
  with self.assertRaises(LookupError):m.query(DB,book='2025',block='Gráfico 48',series_id='chart85_s1',point_index=8)
  with self.assertRaises(ValueError):m.query(DB,book='2025',block='Gráfico 22',series_id='chart43_s1',point_index=True)
 def test_tampered_data_views_and_base_hash_fail_closed(self):
  with patch.object(m,'PREVIOUS_SHA','0'*64):
   with duckdb.connect(str(DB),read_only=True) as c:
    with self.assertRaises(ValueError):m.collect(c)
  with tempfile.TemporaryDirectory(dir=m.ROOT/'tmp') as d:
   db=Path(d)/'bad.duckdb';shutil.copyfile(DB,db)
   with duckdb.connect(str(db)) as c:c.execute('UPDATE blocked_observations SET value=value+1 WHERE row_number=(SELECT min(row_number) FROM blocked_observations)')
   with duckdb.connect(str(db),read_only=True) as c:
    with self.assertRaises(ValueError):m.check_new(c)
   shutil.copyfile(DB,db)
   with duckdb.connect(str(db)) as c:c.execute('CREATE OR REPLACE VIEW blocked_usable_keys AS SELECT * FROM ready_usable_keys')
   with duckdb.connect(str(db),read_only=True) as c:
    with self.assertRaises(ValueError):m.check_new(c)
if __name__=='__main__':unittest.main()
