import copy,json,threading,time,unittest
from types import SimpleNamespace
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch
from rag_pensiones import structured_retrieval as m

AUDIT=m.ROOT/'evaluation/structured_rag_2026-11-01/audit'
class FixedRetriever:
 def __init__(self,result):self.result=result
 def context_book(self,index):return None
 def retrieve(self,query,**kwargs):return copy.deepcopy(self.result)

class StructuredRetrievalTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.service=m.CertifiedStructuredRetriever();cls.rows=cls.service.catalog();cls.cases=json.loads((AUDIT/'query_cases.json').read_text(encoding='utf8'))['cases'];cls.results={p['id']:cls.service.retrieve(p['query'])for p in cls.cases}
 def test198_independent_primary_records_in_one_catalog(self):
  fixtures=json.loads((AUDIT/'primary_expected_records.json').read_text(encoding='utf8'))['facts'];self.assertEqual(len(fixtures),198);catalog={r['observation_id']:r for r in self.rows}
  for p in fixtures:
   r=catalog[p['observation_id']];self.assertEqual(Decimal(str(r['value'])),Decimal(p['value_exact']));self.assertEqual(r['unit'],p['unit']);self.assertEqual(r['source_sheet'],p['source_sheet']);self.assertEqual(r['source_cell'],p['source_cell']);self.assertEqual(r['book'],p['book']);self.assertEqual(r['series_name'],p.get('series',p.get('series_literal')));self.assertEqual(r['metric_name'],p.get('metric',p.get('metric_literal')))
 def test18_positive_queries_use_exact_records_and_policy_warnings(self):
  for p in self.cases:
   if not p['id'].startswith('p'):continue
   with self.subTest(case=p['id']):
    r=self.results[p['id']];self.assertEqual(r['status'],'matched',r);self.assertEqual([h['observation_id']for h in r['hits']],p['expected_ids']);self.assertEqual(Decimal(r['hits'][0]['value']),Decimal(p['expected_value']));self.assertEqual(r['hits'][0]['unit'],p['expected_unit']);self.assertEqual(r['hits'][0]['source_cell'],p['expected_source_cell']);self.assertNotIn('pdf_page',r['hits'][0])
    if p['db_expectation']=='resolved_as_reported_with_warning':self.assertTrue(r['hits'][0]['warning']);self.assertTrue(r['hits'][0]['numeric_reconciliation_failed']);self.assertFalse(r['hits'][0]['calculation_safe']);self.assertFalse(r['hits'][0]['global_calculation_prohibition'])
 def test9_independent_scope_and_source_boundaries(self):
  for p in self.cases:
   if p['id'].startswith('n'):
    r=self.results[p['id']];self.assertIn(r['status'],['ambiguous','unmatched']);self.assertEqual(r['hits'],[],r)
 def test_search_and_database_run_concurrently_with_barrier(self):
  barrier=threading.Barrier(2);threads=[]
  def branch(value):threads.append(threading.get_ident());barrier.wait(timeout=3);return value
  self.assertEqual(m.retrieve_parallel(lambda:branch('semantic'),lambda:branch('structured')),('semantic','structured'));self.assertEqual(len(set(threads)),2)
 def test_index_context_requires_pdf_sha_and_complete_metadata(self):
  sources=self.service._sources;book='2025';p=sources[book];item=dict(source_file=Path(p['pdf']).name,published_at='2025-03-31',doc_id=p['pdf_sha']);self.assertEqual(self.service.context_book(dict(chunks=[item])),book)
  for field,value in[('doc_id','0'*64),('published_at','2026-03-31'),('source_file','other.pdf')]:
   bad=dict(item);bad[field]=value;self.assertIsNone(self.service.context_book(dict(chunks=[bad])))
 def test_query_global_report_has_priority_over_index_context_per_clause(self):
  from rag_pensiones.search import load_index,DEFAULT_INDEX_PATH
  context=self.service.context_book(load_index(DEFAULT_INDEX_PATH));self.assertEqual(context,'2025')
  positive='En el informe2026, que gasto en pensiones se preve en2050 en el Cuadro12; que empleos no financieros da la serie AIReF2026 en2030 en el Gráfico20';r=self.service.retrieve(positive,context_book=context);self.assertEqual(r['status'],'matched',r);self.assertEqual({h['observation_id']for h in r['hits']},{'airef:2026:388','airef:2026:4179'})
  query='En el informe2026, que gasto en pensiones se preve en2050 en el Cuadro12; que gasto en pensiones se preve en2050 en el Cuadro9';r=self.service.retrieve(query,context_book=context);self.assertEqual([h['observation_id']for h in r['hits']],['airef:2026:388']);self.assertTrue(all(p.get('book')=='2026'for p in r['intents']));self.assertEqual(r['status'],'partial')
 def test_unknown_explicit_report_never_falls_back_to_index_book(self):
  control=self.service.retrieve('En el informe2025, que gasto en pensiones se preve en2050 en el Cuadro9',context_book='2025');self.assertEqual(control['status'],'matched',control);self.assertEqual(control['hits'][0]['observation_id'],'airef:2025:454')
  for year in[2027,2024,2030]:
   q=f'En el informe{year}, que gasto en pensiones se preve en2050 en el Cuadro9';r=self.service.retrieve(q,context_book='2025');self.assertEqual(r['hits'],[],r);self.assertEqual(r['status'],'unmatched');self.assertEqual(r['intents'][0]['book'],str(year))
 def test_contradictory_metric_not_authorized_by_monometrical_figure(self):
  control=self.service.retrieve('En el Gráfico20 del informe2026, serie AIReF2026, ¿qué empleos no financieros en2030?');self.assertEqual(control['status'],'matched',control);self.assertEqual(control['hits'][0]['observation_id'],'airef:2026:4179')
  for q in['En el Gráfico20 del informe2026, serie AIReF2026, ¿qué gasto sanitario en2030?','En el Gráfico15 del informe2025, Japón, ¿qué crecimiento del PIB en1971?']:
   r=self.service.retrieve(q);self.assertNotEqual(r['status'],'matched');self.assertEqual(r['hits'],[],r);self.assertEqual(r['intents'][0].get('book'),'2026'if'20 del'in q else'2025')
 def test_title_words_do_not_select_series_or_invented_series(self):
  control=self.service.retrieve('En el Gráfico54 del informe2025, ¿qué Total publicado de crecimiento del gasto educativo en2023?');self.assertEqual(control['status'],'matched',control);self.assertEqual(control['hits'][0]['observation_id'],'airef:2025:18981')
  row=next(r for r in self.rows if r['book']=='2025'and r['object_key']==('grafico','54')and r['period']=='2016')
  for suffix in['',' de la serie inventada']:
   r=self.service.retrieve('Cual es el valor de '+row['metric_name']+' del Grafico54 del informe2025 en2016'+suffix);self.assertNotEqual(r['status'],'matched',r);self.assertEqual(r['hits'],[],r);self.assertEqual(r['intents'][0].get('book'),'2025')
 def test_temporal_operation_is_not_silently_changed_to_point_year(self):
  for query in['Antes de2050, ¿qué ritmo anual supera el gasto en pensiones?','En el Cuadro9 de2025, ¿qué gasto en pensiones hay antes de2050?','En el Gráfico20 de2026, ¿cuál es el máximo de empleos no financieros hasta2050?']:
   r=self.service.retrieve(query);self.assertEqual(r['hits'],[]);self.assertNotEqual(r['status'],'matched')
 def test_requested_dimension_must_match_before_any_numeric_publication(self):
  for query in['En el Cuadro9 del informe2025, ¿qué crecimiento anual del gasto en pensiones en2050?','En el Cuadro9 del informe2025, ¿cuántas pensiones se prevén en2050?','En el Cuadro12 de2026, ¿qué gasto en pensiones corresponde a Brasil en2050?','En el Cuadro9 de2025, ¿qué pensión media en euros se prevé en2050?']:
   r=self.service.retrieve(query);self.assertEqual(r['hits'],[],r);self.assertNotEqual(r['status'],'matched');self.assertEqual(self.make_joint(query,r)['claims'],[])
 def make_joint(self,query,structured,pdf=None,inventory=None):
  r=copy.deepcopy(structured);r['query']=query
  return m.compose_publication(query,pdf or dict(status='insufficient_evidence',parts=[],claims=[],missing_parts=[]),r,inventory or[],{},retriever=FixedRetriever(r))
 def test_numeric_and_cause_without_semantic_evidence_is_partial(self):
  queries=['En el Cuadro12 del informe2026, ¿qué gasto en pensiones se prevé en2050? Explica por qué ocurre.','En el Cuadro12 del informe2026, ¿qué gasto en pensiones se prevé en2050 y qué implica?','En el Cuadro12 del informe2026, ¿qué gasto en pensiones se prevé en2050 y a qué se debe?']
  for query in queries:
   with self.subTest(query=query):
    result=self.service.retrieve(query);pub=self.make_joint(query,result);self.assertEqual(pub['status'],'partial',pub);self.assertTrue(pub['missing_parts']);self.assertTrue(pub['claims'])
 def test_legacy_number_words_never_compete_with_database_lookup(self):
  p=next(c for c in self.cases if c['id']=='p06');q=p['query'];structured=self.results['p06'];claim=dict(text='El gasto previsto es de dieciséis coma uno puntos del PIB.',evidence=[dict(source_id='S1',quote='Texto legado')]);inventory=[dict(claim_id='C1',text=claim['text'],publication_claim=claim)];pdf=dict(status='answered',parts=[dict(start_char=0,end_char=len(q),decision='answered',claim_ids=['C1'])],claims=[claim],missing_parts=[])
  pub=self.make_joint(q,structured,pdf,inventory);self.assertEqual(pub['status'],'answered');self.assertEqual(len(pub['claims']),1);self.assertTrue(pub['claims'][0]['evidence'][0]['source_id'].startswith('DB:'));self.assertNotIn('dieciséis',str(pub))
 def test_ambiguous_series_cannot_be_answered_by_legacy_numbers(self):
  p=next(c for c in self.cases if c['id']=='n01');q=p['query'];claim=dict(text='Serán cuarenta y seis.',evidence=[dict(source_id='S1',quote='Wrong series')]);inv=[dict(claim_id='C1',text=claim['text'],publication_claim=claim)];pdf=dict(status='answered',parts=[dict(start_char=0,end_char=len(q),decision='answered',claim_ids=['C1'])],claims=[claim],missing_parts=[]);pub=self.make_joint(q,self.results['n01'],pdf,inv);self.assertEqual(pub['status'],'insufficient_evidence');self.assertEqual(pub['claims'],[]);self.assertTrue(pub['missing_parts'])
 def test_both_total_years_and_partial_supported_horizons(self):
  q='En el Gráfico54 de2025, ¿cuáles son los totales publicados para2023 y2024?';r=self.service.retrieve(q);self.assertEqual({h['period']for h in r['hits']},{'2023','2024'});self.assertEqual(self.make_joint(q,r)['status'],'answered')
  q='En el Cuadro12 de2026, ¿qué gasto en pensiones se prevé en2050 y2070?';r=self.service.retrieve(q);pub=self.make_joint(q,r);self.assertEqual([h['period']for h in r['hits']],['2050']);self.assertEqual(pub['status'],'partial');self.assertTrue(pub['missing_parts'])
 def test_database_failure_is_visible_and_never_fabricates(self):
  bad=m.CertifiedStructuredRetriever(m.ROOT/'tmp/nonexistent-structured.duckdb');q='En el Cuadro12 de2026, ¿qué gasto en pensiones se prevé en2050?';r=m.retrieve_for_index(q,{},bad);self.assertEqual(r['status'],'error');self.assertEqual(r['hits'],[]);pub=self.make_joint(q,r);self.assertEqual(pub['status'],'insufficient_evidence');self.assertIn('base estructurada',pub['reason'])
 def test_mixed_text_part_can_be_independently_covered(self):
  q='En el Cuadro12 de2026, ¿qué gasto en pensiones se prevé en2050? Explica la causa.';r=self.service.retrieve(q);second=next(p for p in r['intents']if p['status']=='not_applicable');claim=dict(text='La evolución demográfica explica el cambio.',evidence=[dict(source_id='S1',quote='Fuente textual')]);inv=[dict(claim_id='C1',text=claim['text'],publication_claim=claim)];pdf=dict(status='partial',parts=[dict(start_char=second['start'],end_char=second['end'],decision='answered',claim_ids=['C1'])],claims=[claim],missing_parts=[]);pub=self.make_joint(q,r,pdf,inv);self.assertEqual(pub['status'],'answered',pub);self.assertEqual(len(pub['claims']),2)
 def test_unmatched_numeric_part_is_explicit_in_mixed_publication_and_format(self):
  from experiments.query_coverage import format_covered_answer
  for request in ['qué pensión mensual exacta cobraría si me retiro en2037?', 'qué gasto sanitario corresponde a una persona en2039 en el Cuadro12 de2026?']:
   with self.subTest(request=request):
    q='Describe el ámbito del régimen; '+request
    result=self.service.retrieve(q,context_book='2025')
    numeric=next(p for p in result['intents']if p['status']!='not_applicable')
    self.assertEqual(numeric['status'],'unmatched',result)
    textual=next(p for p in result['intents']if p['status']=='not_applicable')
    claim=dict(text='El ámbito del régimen depende de los requisitos publicados.',evidence=[dict(source_id='S1',quote='Requisitos publicados del régimen.')])
    inventory=[dict(claim_id='C1',text=claim['text'],publication_claim=claim)]
    pdf=dict(status='partial',claims=[claim],parts=[dict(start_char=textual['start'],end_char=textual['end'],decision='answered',claim_ids=['C1'])],missing_parts=[dict(query_span=numeric['query_span'],reason_code='not_in_evaluated_evidence',message='No consta una prestación individual respaldada.')])
    publication=self.make_joint(q,result,pdf,inventory)
    self.assertEqual(publication['status'],'partial')
    self.assertEqual(publication['claims'],[claim])
    self.assertEqual(publication['pdf_missing_parts'],pdf['missing_parts'])
    self.assertIn(numeric['query_span'],publication['missing_parts'][0]['message'])
    self.assertIn(numeric['clarification'],publication['missing_parts'][0]['message'])
    report=dict(status='partial',joint_publication=publication,factual_report=dict(outcome=dict(expanded_hits=[dict(rank=1,chunk=dict(page_number=2))])))
    formatted=format_covered_answer(report)
    self.assertIn(claim['text'],formatted)
    self.assertIn(numeric['query_span'],formatted)
    self.assertIn('Las evidencias evaluadas no contienen información suficiente',formatted)
 def test_real_preserved_entry_wires_concurrent_branches(self):
  from experiments import query_preserved as entry
  from test_query_preserved import PreservedQueryTests
  from test_answer_alignment import Generator
  index,units,pages,hits=PreservedQueryTests().inputs();barrier=threading.Barrier(2)
  def semantic(*args,**kwargs):barrier.wait(timeout=3);return hits
  def database(*args,**kwargs):barrier.wait(timeout=3);return dict(status='not_applicable',hits=[],intents=[])
  outcome=dict(status='insufficient_evidence',publication=dict(status='insufficient_evidence',claims=[],reason='No dato'))
  with patch.object(entry,'search',side_effect=semantic),patch.object(entry,'retrieve_for_index',side_effect=database),patch.object(entry,'generate_arm',return_value=outcome):report=entry.answer_preserved(index,'Pregunta',units,pages,embedder=SimpleNamespace(calls=[]),generator=Generator())
  self.assertEqual(report['retrieval']['structured']['status'],'not_applicable');self.assertEqual(len(report['retrieval']['hits']),4);self.assertEqual(report['outcome'],outcome)
 def test_real_covered_entry_preserves_pdf_canonical_and_renders_cell_citation(self):
  from experiments import query_coverage as entry
  from rag_pensiones.search import load_index,DEFAULT_INDEX_PATH
  case=next(c for c in self.cases if c['id']=='p06');q=case['query'];pdf=dict(status='insufficient_evidence',claims=[],parts=[],missing_parts=[],reason='No PDF fact');factual=dict(retrieval=dict(structured=self.results['p06']),outcome=dict(status='insufficient_evidence',full_result=dict(grounding=dict(accepted_claims=[])),publication=copy.deepcopy(pdf)));base=dict(status='insufficient_evidence',query=q,metadata={},factual_report=factual,publication=copy.deepcopy(pdf));before=copy.deepcopy(factual['outcome'])
  with patch.object(entry,'_answer_covered_semantic',return_value=base):report=entry.answer_covered(load_index(DEFAULT_INDEX_PATH),q,{},[])
  self.assertEqual(report['factual_report']['outcome'],before);self.assertEqual(report['joint_publication']['status'],'answered',report);text=entry.format_covered_answer(report);self.assertIn('Cuadro 12',text);self.assertIn('DB:airef:2026:388',text);self.assertNotIn('pág.',text)
 def test_real_covered_entry_uses_replayed_corrected_inventory(self):
  from test_structured_quality_protocol import mixed_corrected_answer
  from rag_pensiones.search import load_index,DEFAULT_INDEX_PATH
  query,answer,pages=mixed_corrected_answer(load_index(DEFAULT_INDEX_PATH))
  self.assertTrue(answer['coverage_correction']['applied'],answer)
  self.assertEqual(answer['joint_publication']['status'],'answered',answer)
  self.assertEqual(len(answer['joint_publication']['claims']),2)
  self.assertIn('C2',[cid for part in answer['joint_publication']['parts']for cid in part.get('claim_ids',[])])
  self.assertEqual(len(answer['factual_report']['outcome']['publication']['claims']),1)
 def test_failed_correction_replay_returns_error_without_joint_claims(self):
  from test_structured_quality_protocol import mixed_corrected_answer
  from rag_pensiones.search import load_index,DEFAULT_INDEX_PATH
  from experiments.query_coverage import format_covered_answer
  query,answer,pages=mixed_corrected_answer(load_index(DEFAULT_INDEX_PATH),fail_replay=True)
  self.assertEqual(answer['status'],'error')
  self.assertEqual(answer['joint_publication']['claims'],[])
  self.assertIn('Invalid correction lineage',answer['error'])
  self.assertTrue(format_covered_answer(answer).startswith('Error:'))

if __name__=='__main__':unittest.main()
