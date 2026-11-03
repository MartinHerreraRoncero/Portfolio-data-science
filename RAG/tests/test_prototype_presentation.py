import copy
import hashlib
from decimal import Decimal
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import duckdb
from experiments import prototype_presentation as p,source_recovery_policy as policy


def fact(value='1.235'):
    return dict(need_id='n',observation_id='o',book='2037',metric='Medida publicada',series='Serie literal',
        period='2050',value=value,unit='euros',derived=False,source_path='publicado.xlsx',source_sha256='sha-private',
        source_sheet='Hoja',source_cell='B2')


def report():
    from rag_pensiones.graph_validation import numeric_claim
    f=fact();return dict(numeric_facts=[f],publication=dict(status='answered',claims=[numeric_claim(f)],missing_parts=[],reason=''))


class FakeClient:
    def __init__(self,text=None):
        self.text=text or '{{METRIC_A}} ({{SERIES_A}}) se sitúa en {{VALUE_UNIT_A}} en {{PERIOD_A}}.';self.calls=[]
    def chat(self,messages,schema,stage):
        self.calls.append(dict(stage=stage,request=messages));return dict(text=self.text)


class PrototypePresentationTests(unittest.TestCase):
    def test_certified_source_api_has_its_original_hash(self):
        self.assertEqual(hashlib.sha256(Path(policy.__file__).read_bytes()).hexdigest(),
                         '5f56e524578b784668bcaf6278c4ac8faa8a1cefe5aabc12065224a184196ab5')

    def test_sql_rounding_and_actual_column_display_preserve_exact_value(self):
        for value,expected in [('1.235','1,24'),('-1.235','-1,24'),('0','0,00')]:
            self.assertEqual(p.rounded_derived(value),expected)
        with tempfile.TemporaryDirectory()as folder:
            db=Path(folder)/'synthetic.duckdb'
            with duckdb.connect(str(db))as connection:
                connection.execute('CREATE TABLE policy_public_observations(observation_id VARCHAR,value DECIMAL(38,12))')
                connection.executemany('INSERT INTO policy_public_observations VALUES (?,?)',[('o','1.235'),('neg','-1.235'),('zero','0')])
                connection.execute('CREATE TABLE policy_source_priority(observation_id VARCHAR,policy_evidence VARCHAR)')
            with patch.object(policy.prior,'enrich',side_effect=lambda c,d:d),patch.object(policy.prior,'query_observation',return_value={}):
                row=policy.query_observation(db,observation_id='o')
            self.assertEqual(Decimal(row['value']),Decimal('1.235'))
            self.assertNotIn('display_value',row)
            for oid,value,expected in [('o','1.235','1,24'),('neg','-1.235','-1,24'),('zero','0','0,00')]:
                self.assertEqual(p.rounded_observation(db,oid,value),expected)
            with self.assertRaises(ValueError):p.rounded_observation(db,'o','999')
            with duckdb.connect(str(db),read_only=True)as connection:
                self.assertEqual(connection.execute('SELECT value FROM policy_public_observations').fetchone()[0],Decimal('1.235'))

    def test_round_display_sources_separate_and_exact_report_immutable(self):
        original=report();before=copy.deepcopy(original)
        with patch.object(p.policy,'query_observation',return_value=fact()),patch.object(p,'rounded_observation',return_value='1,24'):
            shown=p.present_report(original,database='unused',client=FakeClient())
        self.assertIn('1,24 euros',shown['answer']);self.assertIn('Serie literal',shown['answer'])
        self.assertEqual(shown['sources'],'publicado.xlsx — hoja Hoja — celda B2')
        self.assertNotIn('sha-private',shown['answer']+shown['sources']);self.assertNotIn('DB:o',shown['sources'])
        self.assertEqual(shown['narration'],'protected_markers');self.assertEqual(original,before)
        self.assertIn('se sitúa en 1,24 euros en 2050',shown['answer'])
        self.assertNotIn('Según',shown['answer']);self.assertNotIn('publicado.xlsx',shown['answer'])

    def test_narrator_preserves_fields_and_rejects_extra_digits_or_missing_markers(self):
        atoms={'{{A}}':'Medida 1,24 euros, Serie original.','{{B}}':'Otra medida 0,00 %.'}
        for invalid in ['{{A}} y {{B}} 99',
                        '{{A}} {{SERIES_NEW}} {{B}}','{{A}}','{{A}} {{A}} {{B}}']:
            with self.subTest(invalid=invalid),self.assertRaises(ValueError):p.fill_narration(invalid,atoms)
        self.assertEqual(p.fill_narration('{{A}} Asimismo, {{B}}',atoms),atoms['{{A}}']+' Asimismo, '+atoms['{{B}}'])
        self.assertEqual(p.fill_narration('Una medida: {{A}} y otra: {{B}}.',atoms),'Una medida: '+atoms['{{A}}']+' y otra: '+atoms['{{B}}']+'.')
        self.assertEqual(p.fill_narration('{{A}}',{'{{A}}':'Literal {{B}} preservado.'}),'Literal {{B}} preservado.')

    def test_bad_narrator_falls_back_without_altering_validated_facts(self):
        r=report()
        with patch.object(p.policy,'query_observation',return_value=fact()),patch.object(p,'rounded_observation',return_value='1,24'):
            display=p.present_report(r,database='unused',client=FakeClient('{{METRIC_A}} {{SERIES_A}} {{VALUE_UNIT_A}} {{PERIOD_A}} 999'))
        self.assertEqual(display['narration'],'deterministic_fallback');self.assertIn('1,24 euros',display['answer'])
        self.assertNotIn('999',display['answer']);self.assertIn('number',display['narration_error'])

    def test_real_client_narrator_is_only_9b_and_keeps_model_resident(self):
        from rag_pensiones.graph_types import CHAT_MODEL,CHAT_DIGEST
        client=p.narrator_client()
        class Offline:
            def request(self,method,path,payload=None):
                if path=='/api/tags':return dict(models=[dict(name=CHAT_MODEL,digest=CHAT_DIGEST)])
                return dict(model=CHAT_MODEL,done=True,done_reason='stop',prompt_eval_count=32,eval_count=16,
                            message=dict(content=json.dumps(dict(text='{{METRIC_A}} ({{SERIES_A}}) se sitúa en {{VALUE_UNIT_A}} en {{PERIOD_A}}.'))))
        client.transport=Offline()
        with patch.object(p.policy,'query_observation',return_value=fact()),patch.object(p,'rounded_observation',return_value='1,24'):
            shown=p.present_report(report(),database='unused',client=client)
        request=shown['calls'][0]['request'];self.assertEqual(request['model'],CHAT_MODEL)
        self.assertEqual(request['keep_alive'],-1);self.assertFalse(request['think'])
        self.assertNotIn('num_gpu',request['options'])
        from rag_pensiones.graph_types import stage_chat_settings
        self.assertEqual(request['options']['num_predict'],stage_chat_settings('presentation',request['messages'])['options']['num_predict'])
        self.assertLessEqual(len(json.dumps(request,ensure_ascii=False,separators=(',',':')).encode()),6400)

    def test_failed_qualitative_validation_is_not_reported_as_missing_number(self):
        r=dict(interpretation=dict(needs=[dict(kind='text',periods=[])]),retrieval=dict(documents=[dict(source_file='Manual.pdf',page_number=2)]),
            publication=dict(status='insufficient_evidence',claims=[],missing_parts=[dict(reason_code='textual_semantic_rejected')]))
        answer=p.present_report(r,database='unused',client=FakeClient())['answer']
        self.assertIn('respuesta generada no superó',answer)
        self.assertNotIn('cifra',answer);self.assertNotIn('periodo',answer);self.assertNotIn('magnitud',answer)
        r['publication']['missing_parts']=[dict(reason_code='not_in_evaluated_evidence')]
        answer=p.present_report(r,database='unused',client=FakeClient())['answer']
        self.assertIn('evidencias evaluadas',answer);self.assertNotIn('respuesta generada no superó',answer)

    def test_changed_source_fails_and_errors_clarifications_never_call_narrator(self):
        with patch.object(p.policy,'query_observation',return_value=dict(fact('2.235'),display_value='2.24')):
            with self.assertRaises(ValueError):p.present_report(report(),database='unused',client=FakeClient())
        for status in ['error','clarification']:
            r=dict(publication=dict(status=status,reason='Mensaje original',claims=[]));client=FakeClient()
            self.assertEqual(p.present_report(r,database='unused',client=client)['answer'],'Mensaje original')
            self.assertEqual(client.calls,[])


if __name__=='__main__':unittest.main()
