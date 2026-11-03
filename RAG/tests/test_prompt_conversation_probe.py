import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from experiments import prompt_conversation_probe as probe


class PromptProbeTests(unittest.TestCase):
    def test_override_is_scoped_and_restored_even_on_error(self):
        before=dict(probe.pipeline.PROMPTS)
        with self.assertRaises(RuntimeError):
            with probe.prompts_for('candidate'):
                self.assertEqual(probe.pipeline.PROMPTS['interpret'],probe.CANDIDATE['interpret'])
                self.assertEqual(probe.pipeline.PROMPTS['generate'],before['generate'])
                raise RuntimeError('deliberate')
        self.assertEqual(probe.pipeline.PROMPTS,before)

    def test_real_goal_schema_is_used_and_expected_answers_are_not_sent(self):
        received=[]
        def fake(services,stage,payload,schema):
            received.append(payload)
            self.assertEqual(stage,'interpret')
            self.assertIn('answer_requests',schema['properties'])
            return dict(answer_requests=[dict(question_fragment='Concepto solicitado',requested_measure='Concepto',answer_kind='reference',report_publication_year=None)])
        case=dict(id='one',query='Explica la primera',context={},expected=dict(marker='NOT_FOR_MODEL'))
        with patch.object(probe.pipeline,'_chat',side_effect=fake):
            result=probe.interpret_case(case,'candidate',lambda:SimpleNamespace(calls=[]))
        self.assertIsNone(result['error']);self.assertEqual(result['interpretation']['needs'][0]['kind'],'text')
        self.assertNotIn('NOT_FOR_MODEL',json.dumps(received));self.assertEqual(set(received[0]),{'question','context'})

    def test_error_is_saved_and_prompts_restored(self):
        before=dict(probe.pipeline.PROMPTS)
        with patch.object(probe.pipeline,'_chat',side_effect=RuntimeError('test failure')):
            result=probe.interpret_case(dict(id='one',query='Pregunta',context={}), 'candidate',lambda:SimpleNamespace(calls=[]))
        self.assertIn('test failure',result['error']);self.assertEqual(probe.pipeline.PROMPTS,before)

    def test_outputs_cannot_overwrite_and_invalid_case_ids_are_rejected(self):
        with tempfile.TemporaryDirectory(dir=probe.ROOT/'tmp')as folder:
            root=Path(folder);path=root/'cases.json'
            path.write_text(json.dumps(dict(cases=[dict(id='safe',query='Pregunta',context={},expected={})])),encoding='utf-8')
            probe.prepare(path,root/'new')
            with self.assertRaises(FileExistsError):probe.prepare(path,root/'new')
            path.write_text(json.dumps(dict(cases=[dict(id='../escape',query='Pregunta',context={})])),encoding='utf-8')
            with self.assertRaises(ValueError):probe.read_cases(path)

    def test_candidate_prompts_do_not_embed_the_pension_answer(self):
        candidate=' '.join(probe.CANDIDATE.values()).casefold()
        for word in ['cotizaciones','pensiones','999999']:
            self.assertNotIn(word,candidate)

    def test_full_query_keeps_failed_calls_and_does_not_carry_them_to_next_arm(self):
        service=SimpleNamespace(client=SimpleNamespace(calls=[]));attempts=[]
        request=dict(messages=[dict(role='user',content='synthetic')],options=dict(num_predict=1,num_ctx=8192))
        def fail(*args,**kwargs):
            attempts.append(True)
            if len(attempts)==1:service.client.calls.append(dict(stage='interpret',endpoint='/api/chat',request=request))
            raise RuntimeError('synthetic failure')
        with tempfile.TemporaryDirectory(dir=probe.ROOT/'tmp')as folder:
            root=Path(folder);cases=root/'cases.json'
            cases.write_text(json.dumps(dict(cases=[dict(id='one',query='Pregunta',context={})])),encoding='utf-8')
            with patch.object(probe.pipeline,'load_default_services',return_value=service),patch.object(probe.pipeline,'answer_graph',side_effect=fail):
                probe.full_query(cases,root/'new','one')
            baseline=json.loads((root/'new/baseline.json').read_text(encoding='utf-8'))
            candidate=json.loads((root/'new/candidate.json').read_text(encoding='utf-8'))
            self.assertEqual(json.loads((root/'new/protocol.json').read_text(encoding='utf-8'))['mode'],'full_pipeline')
            self.assertEqual(len(baseline['calls']),1);self.assertEqual(len(baseline['request_sizes']),1)
            self.assertEqual(candidate['calls'],[]);self.assertIn('synthetic failure',candidate['error'])


if __name__=='__main__':unittest.main()
