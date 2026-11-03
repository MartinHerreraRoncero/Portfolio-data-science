"""Structural note priority, finite budgets and default compatibility."""
import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from experiments import compare_retained_notes as trial
from experiments.compare_retained_notes import FrozenPlanPlanner
from experiments.planned_preserved import PlannedNotesGrounder
from experiments.reconstructed_units import expand_seeds
from rag_pensiones.chunking import ChunkRecord
from rag_pensiones.search import SearchHit
from test_answer_alignment import Generator


def fixture(notes=1, certified=True, kind='note', oversized=False):
    texts=[(f'B{i}', 'body', 'El texto del cuerpo aporta una explicación documental suficientemente extensa sobre la proyección del sistema.') for i in range(5)]
    texts += [(f'N{i}',kind,'Una nota completa sin punto final' if not oversized else 'Nota extensa '+('texto '*2000).strip()) for i in range(notes)]
    raw='\n'.join(t for _,_,t in texts)
    page=dict(doc_id='d',source_file='public.pdf',published_at='2025-01-01',page_number=1,text=raw)
    units=[];start=0
    for uid,k,text in texts:
        span={key:page[key] for key in ['doc_id','source_file','published_at','page_number']}
        span.update(start_char=start,end_char=start+len(text),quote=text)
        unit=dict(id=uid,kind=k,spans=[span])
        if k=='note' and certified:unit['source_boundary']='closed'
        units.append(unit);start+=len(text)+1
    hit=SearchHit(1,.8,ChunkRecord('seed','d','public.pdf','2025-01-01',1,0,len(raw),raw,'synthetic'))
    return [hit],dict(units=units),{('d',1):page}


class RetainNotesTests(unittest.TestCase):
    def test_cli_integrity_guard_checks_relative_paths_before_inference(self):
        with patch.object(trial,'_json',return_value=dict(frozen_files={'AGENTS.md':'0'*64})), self.assertRaisesRegex(ValueError,'Frozen file changed'):
            trial.main(['--protocol','evaluation/retain_notes_protocol.json','--output','evaluation/unused_test.json'])

    def expand(self,data,*,prefer=True,query='Consulta'):
        return expand_seeds(*data,query,closed_note_tails=True,prefer_retrieved_notes=prefer)

    def test_retrieved_closed_note_survives_without_increasing_four_parent_limit(self):
        data=fixture();before=copy.deepcopy(data)
        _,old,old_exp=self.expand(data,prefer=False)
        hits,new,new_exp=self.expand(data)
        self.assertNotIn('N0',[r['unit_id'] for r in old['records']])
        self.assertIn('N0',[r['unit_id'] for r in new['records']])
        self.assertEqual(len(new_exp['selected_parents']),4)
        self.assertEqual(new_exp['selected_parents'][0]['unit_id'],'N0')
        self.assertEqual(data,before)

    def test_unreviewed_body_or_unretrieved_note_does_not_receive_priority(self):
        for data in [fixture(certified=False),fixture(kind='body')]:
            _,_,exp=self.expand(data)
            self.assertNotIn('N0',[x['unit_id'] for x in exp['selected_parents']])
        seeds,units,pages=fixture()
        seeds[0]=SearchHit(1,.8,ChunkRecord('seed','d','public.pdf','2025-01-01',1,0,100,pages[('d',1)]['text'][:100],'synthetic'))
        _,_,exp=self.expand((seeds,units,pages))
        self.assertNotIn('N0',[x['unit_id'] for x in exp['selected_parents']])

    def test_many_notes_respect_cap_ties_and_query_independence(self):
        data=fixture(notes=6)
        _,_,first=self.expand(data)
        _,_,second=self.expand(data,query='Otra pregunta con un valor diferente 42')
        selected=[x['unit_id'] for x in first['selected_parents']]
        self.assertEqual(selected,['N0','N1','N2','N3'])
        self.assertEqual(first,second)
        self.assertTrue(all(x['reason']=='four_parent_limit' for x in first['omitted_parents']))

    def test_oversized_note_is_omitted_explicitly_and_body_can_still_fit(self):
        _,_,exp=self.expand(fixture(oversized=True))
        self.assertNotIn('N0',[x['unit_id'] for x in exp['selected_parents']])
        omitted=next(x for x in exp['omitted_parents'] if x['unit_id']=='N0')
        self.assertEqual(omitted['reason'],'message_budget')
        self.assertEqual(len(exp['selected_parents']),4)

    def test_real_s1_notes_survive_and_default_reproduces_all_previous_nodes(self):
        source=json.loads(Path('evaluation/planned_preserved_results.json').read_text(encoding='utf-8'))
        units=json.loads(Path('evaluation/context_fix_units.json').read_text(encoding='utf-8'))
        pages=[json.loads(x) for x in Path('data/processed/airef_2025_pages.jsonl').read_text(encoding='utf-8').splitlines()]
        for case in source['cases']:
            for node in case['conditions']['planned']['answer']['nodes']:
                seeds=[SearchHit(h['rank'],h['score'],ChunkRecord(**h['chunk'])) for h in node['seed_hits']]
                old=PlannedNotesGrounder(units,pages);old.prepare_evidence(node['query'],seeds)
                self.assertEqual(old.pending['expansion'],node['expansion'])
                if case['id']=='s1':
                    new=PlannedNotesGrounder(units,pages,prefer_retrieved_notes=True)
                    new.prepare_evidence(node['query'],seeds)
                    self.assertIn('U6',[r['unit_id'] for r in new.pending['catalog']['records']])
                    self.assertIn('0,9',str(new.pending['catalog']))
                    self.assertLessEqual(len(new.pending['expansion']['selected_parents']),4)

    def test_saved_plan_is_reused_without_chat_and_bound_to_query_and_digest(self):
        source=json.loads(Path('evaluation/planned_preserved_results.json').read_text(encoding='utf-8'))
        case=source['cases'][0];gen=Generator()
        gen.get_digest=lambda:case['conditions']['planned']['answer']['semantic']['plan']['model_digest']
        planner=FrozenPlanPlanner(case,gen);plan=planner.plan(case['query'])
        self.assertEqual(plan['plan']['subquestions'],planner.saved['subquestions'])
        plan['plan']['subquestions'].clear()
        self.assertTrue(planner.saved['subquestions']);self.assertEqual(gen.count,0)
        with self.assertRaises(ValueError):planner.plan('Otra pregunta')
        gen.get_digest=lambda:'a'*64
        with self.assertRaises(ValueError):planner.plan(case['query'])


if __name__=='__main__':unittest.main()
