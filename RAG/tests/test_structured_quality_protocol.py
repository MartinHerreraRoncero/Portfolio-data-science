import copy,json,tempfile,unittest
from unittest.mock import patch
from pathlib import Path
from experiments import continuous_quality as quality
from experiments import evaluate_covered_pipeline as runner
from experiments import query_coverage as covered
from rag_pensiones import structured_retrieval as retrieval
from rag_pensiones.search import DEFAULT_INDEX_PATH,load_index

def mixed_corrected_answer(index, *, fail_replay=False):
 """Real correction/replay on canonical quotes with controlled, recorded model replies."""
 from test_query_coverage import _MockCorrectionGenerator, _MockCorrectionVerifier, _mock_audit_resp
 root=retrieval.ROOT;directory=root/'evaluation/quality_followup_2026-10-28'
 old=json.loads((directory/'live_mixed_v3/results.json').read_text(encoding='utf8'))['cases'][9]
 saved=json.loads((directory/'saved_input_probe_v2/results.json').read_text(encoding='utf8'))['cases'][9]
 pages=[json.loads(line)for line in(root/'data/processed/airef_2025_pages.jsonl').read_text(encoding='utf8').splitlines()]
 numeric='Según el Cuadro12 de2026, ¿cuál es el gasto en pensiones en2050?'
 query=numeric+' Explica según el informe de2025: '+saved['query'];textual=query[len(numeric):]
 factual=copy.deepcopy(old['answer']['factual_report']);factual['query']=query
 factual['metadata']['index_hash']=quality.canonical_hash(index)
 factual.setdefault('retrieval',{})['structured']=retrieval.retrieve_for_index(query,index)
 before=copy.deepcopy(factual)
 first=dict(query_span=numeric,decision='not_answered',claim_ids=[],reason_code='not_in_evaluated_evidence')
 initial=[first,dict(query_span=textual,decision='not_answered',claim_ids=[],reason_code='missing_from_answer')]
 corrected=[first,dict(query_span=textual,decision='answered',claim_ids=['C2'],reason_code='covered')]
 diagnostics=saved['diagnostics']
 generator=_MockCorrectionGenerator([copy.deepcopy(diagnostics['generation']['raw_response'])])
 verifier=_MockCorrectionVerifier([_mock_audit_resp(initial),copy.deepcopy(diagnostics['verification']['raw_response']),_mock_audit_resp(corrected)])
 actual_replay=covered.replay_coverage_correction;replay_calls=[]
 def fail_joint_replay(*args,**kwargs):
  replay_calls.append(True)
  if len(replay_calls)>1:raise ValueError('Invalid correction lineage')
  return actual_replay(*args,**kwargs)
 with patch.object(covered,'answer_preserved',return_value=factual):
  if fail_replay:
   with patch.object(covered,'replay_coverage_correction',side_effect=fail_joint_replay):
    answer=covered.answer_covered(index,query,{},pages,generator=generator,verifier=verifier,coverage_correction=True)
  else:answer=covered.answer_covered(index,query,{},pages,generator=generator,verifier=verifier,coverage_correction=True)
 assert factual==before
 assert generator.idx==1 and verifier.idx==3
 if fail_replay:assert len(replay_calls)==2
 return query,answer,pages

class StructuredProtocolTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.index=load_index(DEFAULT_INDEX_PATH);cls.query='Según el Cuadro12 de2026, ¿cuál es el gasto en pensiones en2050?';cls.structured=retrieval.retrieve_for_index(cls.query,cls.index);assert cls.structured['status']=='matched',cls.structured
  cls.pdf=dict(schema_version='coverage-publication-v1',status='insufficient_evidence',claims=[],parts=[],missing_parts=[],reason='Sin dato PDF');cls.joint=retrieval.compose_publication(cls.query,cls.pdf,cls.structured,[],cls.index)
  cls.answer=dict(query=cls.query,status=cls.joint['status'],metadata=dict(structured_profile='certified_lookup_parallel_v1',joint_coverage_status=cls.joint['status']),factual_report=dict(query=cls.query,metadata=dict(index_hash=quality.canonical_hash(cls.index)),retrieval=dict(structured=cls.structured)),publication=cls.pdf,structured_report=cls.structured,joint_publication=cls.joint)
 def make_protocol(self,structured=True,correction=False):
  with tempfile.TemporaryDirectory(dir=retrieval.ROOT/'tmp',prefix='structured-protocol-')as directory:return quality.prepare(Path(directory),structured_retrieval=structured,coverage_correction=correction)
 def replay(self,answer=None,**kwargs):return quality.replay_structured_answer(answer or self.answer,self.query,self.index,[],self.pdf,**kwargs)
 def test_current_protocol_pins_database_policy_certificate_and_code(self):
  p=self.make_protocol();self.assertEqual(p['schema_version'],'covered-pipeline-protocol-v3');runner.validate_protocol(p);self.assertEqual(p['structured_retrieval']['database_sha256'],retrieval.DATABASE_SHA)
  for name,digest in p['structured_retrieval']['files'].items():self.assertEqual(p['frozen_files'][name],digest)
  for name in['rag_pensiones/structured_retrieval.py','experiments/source_recovery_policy.py']:self.assertIn(name,p['frozen_files'])
 def test_legacy_prepare_stays_explicitly_legacy_and_correction_v3_supported(self):
  old=self.make_protocol(False);self.assertEqual(old['schema_version'],'covered-pipeline-protocol-v1');self.assertNotIn('structured_retrieval',old);runner.validate_protocol(old)
  p=self.make_protocol(True,True);self.assertEqual(p['schema_version'],'covered-pipeline-protocol-v3');self.assertIs(p['coverage_correction'],True);runner.validate_protocol(p)
 def test_manifest_and_byte_pin_mutations_rejected(self):
  original=self.make_protocol()
  for kind in['manifest','asset','code']:
   p=copy.deepcopy(original)
   if kind=='manifest':p['structured_retrieval']['database_sha256']='0'*64
   elif kind=='asset':p['frozen_files'][p['structured_retrieval']['database_path']]='0'*64
   else:p['frozen_files']['rag_pensiones/structured_retrieval.py']='0'*64
   with self.assertRaises(ValueError):runner.validate_protocol(p)
 def test_positive_replay_keeps_pdf_separate_and_verifies_effective_formatter(self):
  before=copy.deepcopy(self.pdf);pub,verified=self.replay(case_status='answered',formatted_answer=covered.format_covered_answer(self.answer));self.assertEqual(pub,self.joint);self.assertEqual(verified,self.structured);self.assertEqual(self.pdf,before)
 def test_hit_identity_value_unit_cell_and_period_mutations_rejected(self):
  for key,value in[('value','999'),('unit','euros'),('source_cell','Z99'),('period','2070'),('series','Serie inventada')]:
   answer=copy.deepcopy(self.answer);answer['factual_report']['retrieval']['structured']['hits'][0][key]=value;answer['structured_report']=copy.deepcopy(answer['factual_report']['retrieval']['structured'])
   with self.assertRaises(ValueError):self.replay(answer)
 def test_joint_status_formatter_query_and_index_mutations_rejected(self):
  for kind in['joint','status','metadata','query','top_query','index','formatted']:
   answer=copy.deepcopy(self.answer)
   if kind=='joint':answer['joint_publication']['claims'][0]['text']='Cifra alterada'
   elif kind=='status':answer['status']='insufficient_evidence'
   elif kind=='metadata':answer['metadata']['joint_coverage_status']='partial'
   elif kind=='query':answer['factual_report']['query']='Otra pregunta'
   elif kind=='top_query':answer['query']='Otra pregunta'
   elif kind=='index':answer['factual_report']['metadata']['index_hash']='0'*64
   with self.assertRaises(ValueError):self.replay(answer,formatted_answer='Texto manipulado'if kind=='formatted'else None)
 def test_typed_provenance_is_not_a_fabricated_pdf_page(self):
  pub,verified=self.replay();prov=quality.extract_evidence_provenance({},verified_structured=verified,publication=pub);self.assertEqual(len(prov),1);record=next(iter(prov.values()));self.assertEqual(record['source_kind'],'structured_cell');self.assertIs(record['structured_verified'],True);self.assertNotIn('page_number',record);self.assertIn('Cuadro 12',record['source_sheet'])
 def test_applied_correction_and_structured_publication_replay_the_same_inventory(self):
  query,answer,pages=mixed_corrected_answer(self.index)
  self.assertTrue(answer['coverage_correction']['applied'],answer)
  replayed=covered.replay_coverage_correction(query,answer['factual_report'],answer['initial_coverage_audit'],answer['coverage_correction'],pages=pages)
  self.assertEqual([r['claim_id']for r in replayed['inventory']],['C1','C2'])
  publication,verified=quality.replay_structured_answer(answer,query,self.index,replayed['inventory'],answer['publication'],case_status='answered',formatted_answer=covered.format_covered_answer(answer))
  self.assertEqual(publication,answer['joint_publication'])
  self.assertEqual(len(publication['claims']),2)
  self.assertEqual(verified['hits'][0]['observation_id'],'airef:2026:388')
  tampered=copy.deepcopy(answer['coverage_correction']);tampered['inventory'][-1]['text']='Inventado'
  with self.assertRaises(ValueError):covered.replay_coverage_correction(query,answer['factual_report'],answer['initial_coverage_audit'],tampered,pages=pages)

if __name__=='__main__':unittest.main()
