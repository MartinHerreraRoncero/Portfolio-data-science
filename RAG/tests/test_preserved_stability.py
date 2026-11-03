"""Offline evaluation harness checks; mocks do not certify answer quality."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from experiments import evaluate_preserved_stability as trial


def inputs():
    queries=dict(schema_version='reranker-validation-queries-v1',cases=[dict(id='a',query='Pregunta A'),dict(id='b',query='Pregunta B')])
    schedule=[dict(repetition=r,id=cid) for r,ids in [(1,['a','b']),(2,['b','a']),(3,['a','b'])] for cid in ids]
    return dict(model='qwen3-embedding:4b'),queries,schedule


def answer(status='answered'):
    return dict(metadata=dict(digest='a'*64,profile='preserved_v1',planner_used=False,closed_note_tails=False,max_packages=1),
                outcome=dict(status=status,error='synthetic internal failure'))


class PreservedStabilityTests(unittest.TestCase):
    def test_schedule_requires_exact_three_repetitions_no_bool_duplicate_or_omission(self):
        _,queries,schedule=inputs();trial.validate_schedule(queries,schedule)
        bad=[None,schedule[:-1],schedule+[schedule[0]],copy.deepcopy(schedule),copy.deepcopy(schedule),copy.deepcopy(schedule)]
        bad[3][0]['repetition']=True;bad[4][0]['id']='unknown';bad[5][1]=copy.deepcopy(bad[5][0])
        for value in bad:
            with self.subTest(value_type=type(value)),self.assertRaises(ValueError):trial.validate_schedule(queries,value)

    def test_new_clients_and_literal_queries_follow_schedule_without_extra_options(self):
        index,queries,schedule=inputs();gens=[];embs=[];seen=[]
        def gen():
            obj=MagicMock(calls=[]);obj.get_digest.return_value='a'*64;gens.append(obj);return obj
        def emb(**kwargs):
            obj=SimpleNamespace(calls=[]);embs.append(obj);return obj
        def observed(idx,query,units,pages,**kwargs):
            self.assertIs(idx,index);self.assertEqual(set(kwargs),{'generator','embedder'})
            seen.append((query,id(kwargs['generator']),id(kwargs['embedder'])));return answer()
        with patch.object(trial,'validate_index'),patch.object(trial,'validate_catalog'),\
             patch.object(trial,'RecordedGenerator',side_effect=gen),patch.object(trial,'QwenEmbedder',side_effect=emb),\
             patch.object(trial,'answer_preserved',side_effect=observed),contextlib.redirect_stdout(io.StringIO()):
            result=trial.run(index,queries,{},[],None,expected_digest='a'*64,schedule=schedule)
        self.assertEqual([q for q,_,_ in seen],['Pregunta A','Pregunta B','Pregunta B','Pregunta A','Pregunta A','Pregunta B'])
        self.assertEqual(len({id(g) for g in gens}),6);self.assertEqual(len({id(e) for e in embs}),6)
        self.assertEqual(result['run_status'],'completed');self.assertEqual(len(result['cases']),6)

    def test_internal_error_and_exception_are_kept_and_schedule_continues_without_retry(self):
        index,queries,schedule=inputs();gen=MagicMock(calls=[]);gen.get_digest.return_value='a'*64
        returns=[answer(),answer('error'),ValueError('synthetic transport failure'),answer(),answer(),answer()]
        with patch.object(trial,'validate_index'),patch.object(trial,'validate_catalog'),\
             patch.object(trial,'RecordedGenerator',return_value=gen),patch.object(trial,'QwenEmbedder',return_value=SimpleNamespace(calls=[])),\
             patch.object(trial,'answer_preserved',side_effect=returns) as call,contextlib.redirect_stdout(io.StringIO()):
            result=trial.run(index,queries,{},[],None,expected_digest='a'*64,schedule=schedule)
        self.assertEqual(call.call_count,6);self.assertEqual(len(result['cases']),6)
        self.assertEqual(result['run_status'],'error')
        self.assertIn('synthetic internal',result['cases'][1]['error'])
        self.assertIn('synthetic transport',result['cases'][2]['error'])
        self.assertIn('answer',result['cases'][1]);self.assertEqual(result['cases'][-1]['status'],'answered')

    def test_actual_digest_profile_and_invalid_stats_are_not_silently_accepted(self):
        index,queries,schedule=inputs()
        for field,value in [('digest','wrong'),('planner_used',True),('closed_note_tails',True),('profile','other'),('max_packages',2)]:
            payload=answer();payload['metadata'][field]=value
            gen=MagicMock(calls=[{'raw_response':None}]);gen.get_digest.return_value='a'*64
            with self.subTest(field=field),patch.object(trial,'validate_index'),patch.object(trial,'validate_catalog'),\
                 patch.object(trial,'RecordedGenerator',return_value=gen),patch.object(trial,'QwenEmbedder',return_value=SimpleNamespace(calls=[])),\
                 patch.object(trial,'answer_preserved',return_value=payload),contextlib.redirect_stdout(io.StringIO()):
                result=trial.run(index,queries,{},[],None,expected_digest='a'*64,schedule=schedule)
            self.assertEqual(result['run_status'],'error')
            for out in result['cases']:
                self.assertNotIn('answer',out);self.assertTrue(all(v is None for v in out['new_llm_stats'].values()))
                self.assertEqual(out['invalid_answer'],payload)

    def test_changed_frozen_file_fails_before_clients(self):
        with tempfile.TemporaryDirectory(dir='tmp') as folder:
            folder=Path(folder);data=folder/'data';data.write_text('changed',encoding='utf-8')
            protocol=folder/'p.json';protocol.write_text(json.dumps({'frozen_files':{str(data):'0'*64}}),encoding='utf-8')
            with patch.object(trial,'RecordedGenerator') as gen,self.assertRaisesRegex(ValueError,'Frozen file changed'):
                trial.main(['--protocol',str(protocol),'--output',str(folder/'r.json')])
            gen.assert_not_called()

    def test_input_hash_mismatch_fails_before_clients(self):
        index,queries,schedule=inputs()
        with tempfile.TemporaryDirectory(dir='tmp') as folder:
            folder=Path(folder)
            files={'queries':queries,'units':{},'pages':[]}
            for name,data in files.items():(folder/(name+'.json')).write_text(json.dumps(data),encoding='utf-8')
            (folder/'pages.jsonl').write_text('',encoding='utf-8')
            protocol=dict(frozen_files={},schedule=schedule,index_hash='wrong')
            (folder/'p.json').write_text(json.dumps(protocol),encoding='utf-8')
            with patch.object(trial,'load_index',return_value=index),patch.object(trial,'RecordedGenerator') as gen,\
                 self.assertRaisesRegex(ValueError,'Input differs'):
                trial.main(['--protocol',str(folder/'p.json'),'--queries',str(folder/'queries.json'),
                    '--units',str(folder/'units.json'),'--pages',str(folder/'pages.jsonl'),'--output',str(folder/'r.json')])
            gen.assert_not_called()

    def test_existing_output_refused_before_clients_without_overwrite(self):
        index,queries,schedule=inputs()
        with tempfile.TemporaryDirectory(dir='tmp') as folder:
            path=Path(folder)/'r.json';path.write_text('original',encoding='utf-8')
            with patch.object(trial,'validate_index'),patch.object(trial,'validate_catalog'),\
                 patch.object(trial,'RecordedGenerator') as gen,self.assertRaises(ValueError):
                trial.run(index,queries,{},[],path,expected_digest='a'*64,schedule=schedule)
            gen.assert_not_called();self.assertEqual(path.read_text(encoding='utf-8'),'original')

    def test_external_digest_change_after_answer_prevents_publication(self):
        index,queries,schedule=inputs()
        def fresh():
            gen=MagicMock(calls=[])
            gen.get_digest.side_effect=['a'*64,'changed_after_answer']
            return gen
        with patch.object(trial,'validate_index'),patch.object(trial,'validate_catalog'),\
             patch.object(trial,'RecordedGenerator',side_effect=fresh),\
             patch.object(trial,'QwenEmbedder',return_value=SimpleNamespace(calls=[])),\
             patch.object(trial,'answer_preserved',side_effect=lambda *a,**k:answer()),contextlib.redirect_stdout(io.StringIO()):
            result=trial.run(index,queries,{},[],None,expected_digest='a'*64,schedule=schedule)
        self.assertEqual(result['run_status'],'error')
        self.assertTrue(all(c['status']=='error' and 'answer' not in c for c in result['cases']))
        self.assertTrue(all(c['invalid_answer']['metadata']['digest']=='a'*64 for c in result['cases']))

    def test_client_construction_failure_is_recorded_without_previous_call_leak(self):
        index,queries,schedule=inputs();first=MagicMock(calls=[]);first.get_digest.return_value='a'*64
        def fresh():
            gen=MagicMock(calls=[]);gen.get_digest.return_value='a'*64;return gen
        generators=[first,RuntimeError('generator init failed')]+[fresh() for _ in range(4)]
        embedders=[SimpleNamespace(calls=[]),RuntimeError('embedder init failed')]+[SimpleNamespace(calls=[]) for _ in range(3)]
        with patch.object(trial,'validate_index'),patch.object(trial,'validate_catalog'),\
             patch.object(trial,'RecordedGenerator',side_effect=generators),patch.object(trial,'QwenEmbedder',side_effect=embedders),\
             patch.object(trial,'answer_preserved',side_effect=lambda *a,**k:answer()),contextlib.redirect_stdout(io.StringIO()):
            result=trial.run(index,queries,{},[],None,expected_digest='a'*64,schedule=schedule)
        self.assertEqual(len(result['cases']),6);self.assertEqual(result['run_status'],'error')
        self.assertIn('generator init failed',result['cases'][1]['error'])
        self.assertIn('embedder init failed',result['cases'][2]['error'])
        self.assertTrue(all(result['cases'][i]['calls']==[] for i in [1,2]))
        self.assertEqual(result['cases'][-1]['status'],'answered')

    def test_checkpoint_precedes_next_run_and_physical_stats_are_summed(self):
        index,queries,schedule=inputs();seen=[]
        def fresh():
            gen=MagicMock(calls=[]);gen.get_digest.return_value='a'*64;return gen
        with tempfile.TemporaryDirectory(dir='tmp') as folder:
            path=Path(folder)/'result.json'
            def observed(*args,**kwargs):
                checkpoint=json.loads(path.read_text(encoding='utf-8'))
                self.assertEqual(len(checkpoint['cases']),len(seen))
                seen.append(args[1])
                kwargs['generator'].calls.extend([
                    {'raw_response':dict(prompt_eval_count=10,eval_count=3,eval_duration=4,total_duration=5)},
                    {'raw_response':dict(prompt_eval_count=20,eval_count=4,eval_duration=6,total_duration=7)}])
                return answer()
            with patch.object(trial,'validate_index'),patch.object(trial,'validate_catalog'),\
                 patch.object(trial,'RecordedGenerator',side_effect=fresh),\
                 patch.object(trial,'QwenEmbedder',side_effect=lambda **k:SimpleNamespace(calls=[])),\
                 patch.object(trial,'answer_preserved',side_effect=observed),contextlib.redirect_stdout(io.StringIO()):
                result=trial.run(index,queries,{},[],path,expected_digest='a'*64,schedule=schedule)
            self.assertEqual(json.loads(path.read_text(encoding='utf-8')),result)
        for row in result['cases']:
            self.assertEqual(row['new_llm_stats'],dict(prompt_eval_count=30,eval_count=7,eval_duration=10,total_duration=12))


if __name__=='__main__':unittest.main()
