"""Integration/provenance tests; fake support verdicts test wiring, not model quality."""
import copy
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from experiments import planned_preserved as trial
from rag_pensiones.search import load_index, DEFAULT_INDEX_PATH, validate_index, SearchHit
from test_answer_alignment import Generator
from test_grounded_semantic import MockPlanner


class WiringGenerator(Generator):
    def chat(self, messages, response_schema):
        data = json.loads(messages[1]['content'])
        if 'paquetes' in data:
            if 'sin dato' in data['pregunta']:
                self.payload = dict(status='insufficient_evidence', ids=[], reason='No hay dato solicitado.')
            else:
                package = next((p for p in data['paquetes'] if len(p['quotes']) >= 2), data['paquetes'][0])
                self.payload = dict(status='selected', ids=[package['id']], reason='')
        elif 'afirmaciones' not in data:
            props = response_schema['anyOf'][0]['properties']['claims']['items']['properties']
            refs = props['evidence_ids']['const']
            quote = data['catalogo_evidencias'][-1]['quote']
            self.payload = dict(status='answered', reason='', claims=[dict(claim_id='C1',
                text=quote[:250], evidence_ids=refs)])
        else:
            refs = data['afirmaciones'][0]['evidence_ids']
            self.payload = dict(verdicts={'C1':dict(verdict='supported', reason='Synthetic wiring verdict.',
                evidence_verdicts={ref:dict(relation='supports' if i==len(refs)-1 else 'irrelevant')
                                   for i, ref in enumerate(refs)})})
        return super().chat(messages, response_schema)


class PlannedPreservedTests(unittest.TestCase):
    def setUp(self):
        self.index = load_index(DEFAULT_INDEX_PATH)
        self.units = json.loads(Path('evaluation/context_fix_units.json').read_text(encoding='utf-8'))
        self.pages = [json.loads(x) for x in Path('data/processed/airef_2025_pages.jsonl').read_text(encoding='utf-8').splitlines()]
        chunks = [c for c,_ in validate_index(self.index)]
        self.seeds = [[SearchHit(1,.8,chunks[3])], [SearchHit(1,.7,chunks[-1])]]

    def run_plan(self, questions, *, dependent=False, generator=None):
        gen = generator if generator is not None else WiringGenerator()
        plan = dict(subquestions=[dict(id=f'p{i+1}', question=q, required_evidence='Texto de la fuente.',
            depends_on=['p1'] if dependent and i else []) for i,q in enumerate(questions)])
        planner = MockPlanner(plan, generator=gen, digest='a'*64)
        seeds = [self.seeds[i%2] for i in range(len(questions))]
        with patch('rag_pensiones.semantic.search', side_effect=seeds) as search:
            result = trial.answer_planned_preserved(self.index, 'Consulta de prueba.', self.units, self.pages,
                generator=gen, planner=planner, embedder=SimpleNamespace(calls=[]))
        return result, gen, search

    def test_multiple_nodes_expand_before_validation_deduplicate_and_keep_raw_retrieval(self):
        before = copy.deepcopy(self.units)
        result, gen, search = self.run_plan(['Pregunta uno','Pregunta dos','Pregunta tres'])
        self.assertEqual(search.call_count,3)
        self.assertEqual(result['publication']['status'],'answered')
        nodes=result['nodes']; sources={s['source_id']:s for s in result['semantic']['sources']}
        self.assertEqual(nodes[0]['expanded_hits'],nodes[2]['expanded_hits'])
        self.assertTrue(set(h['rank'] for h in nodes[0]['expanded_hits']).isdisjoint(h['rank'] for h in nodes[1]['expanded_hits']))
        self.assertEqual(len(sources),len(result['semantic']['sources']))
        for part,node in zip(result['semantic']['parts'],nodes):
            self.assertEqual(len(part['retrievals']),1)
            seed_source=sources[part['retrievals'][0]['source_id']]
            self.assertEqual(seed_source['chunk_id'],node['seed_hits'][0]['chunk']['chunk_id'])
            self.assertNotIn(seed_source['chunk_id'],[h['chunk']['chunk_id'] for h in node['expanded_hits']])
            self.assertEqual(len(part['claims'][0]['evidence']),1)
            self.assertGreater(len(node['publication']['claims'][0]['evidence']),1)
            for claim in node['publication']['claims']:
                for ev in claim['evidence']:self.assertIn(ev['source_id'],part['sources_ids'])
        self.assertEqual(self.units,before)
        self.assertTrue(any(s.get('expansions') for s in sources.values()))
        self.assertEqual(sum(len(n['calls']) for n in nodes),len(gen.calls))

    def test_partial_failure_keeps_parts_but_does_not_publish_complete_global_answer(self):
        result,_,_=self.run_plan(['Pregunta uno','Pregunta sin dato'])
        self.assertEqual(result['publication']['status'],'insufficient_evidence')
        self.assertEqual(result['publication']['claims'],[])
        self.assertTrue(result['publication']['parts'][0]['claims'])

    def test_abstention_blocks_dependent_node_without_retrieval_or_resolution(self):
        result,_,search=self.run_plan(['Pregunta sin dato','Pregunta dependiente'],dependent=True)
        self.assertEqual(search.call_count,1)
        self.assertEqual(result['semantic']['api_calls']['resolutions'],0)
        self.assertEqual(result['publication']['parts'][1]['execution_status'],'blocked_dependency')
        self.assertEqual(len(result['nodes']),1)

    def test_answered_parent_resolves_child_and_keeps_global_sources(self):
        seen=[]
        def resolve(planner,**kwargs):
            seen.append(kwargs['dependencies'])
            return dict(question=kwargs['question'],model=planner.model,model_digest=planner.digest,
                prompt_sha256='b'*64,stats=dict(prompt_eval_count=20,eval_count=10,eval_duration=100,total_duration=200),
                parameters=dict(temperature=0,seed=42,num_ctx=8192,num_predict=1024))
        with patch.object(MockPlanner,'resolve',resolve):
            result,_,search=self.run_plan(['Pregunta uno','Pregunta dependiente'],dependent=True)
        self.assertEqual(search.call_count,2)
        self.assertEqual(result['semantic']['api_calls']['resolutions'],1)
        self.assertEqual(seen[0][0]['claims'],result['semantic']['parts'][0]['claims'])
        self.assertEqual(result['publication']['status'],'answered')

    def test_tampered_catalog_or_seed_fails_before_grounding(self):
        units=copy.deepcopy(self.units);units['units'][0]['spans'][0]['quote']='Texto falsificado.'
        gen=WiringGenerator()
        with self.assertRaises(ValueError):trial.PlannedNotesGrounder(units,self.pages,generator=gen)
        g=trial.PlannedNotesGrounder(self.units,self.pages,generator=gen)
        seed=self.seeds[0][0];bad=replace(seed.chunk,text='Texto falsificado.')
        with self.assertRaisesRegex(ValueError,'provenance'):g.prepare_evidence('Pregunta',[SearchHit(1,.8,bad)])
        self.assertEqual(gen.count,0)

    def test_swapped_or_duplicate_registered_sources_rejected_before_generation(self):
        for mode in ['foreign','duplicate']:
            gen=WiringGenerator();g=trial.PlannedNotesGrounder(self.units,self.pages,generator=gen)
            hits=g.prepare_evidence('Pregunta',self.seeds[0])
            bad=copy.deepcopy(hits)
            if mode=='foreign':bad[0]=SearchHit(1,.8,self.seeds[1][0].chunk)
            else:bad=[SearchHit(1,h.score,h.chunk) for h in hits]
            with self.assertRaisesRegex(ValueError,'registration'):g.answer('Pregunta',bad,expected_digest='a'*64)
            self.assertEqual(gen.count,0)

    def test_global_registry_rejects_chunk_id_reused_for_another_passage(self):
        original=trial.PlannedNotesGrounder.prepare_evidence; seen=[]
        def prepare(g,query,hits):
            result=original(g,query,hits)
            if seen:
                h=result[0];result[0]=SearchHit(h.rank,h.score,replace(h.chunk,chunk_id=seen[0]))
            else:seen.append(result[0].chunk.chunk_id)
            return result
        with patch.object(trial.PlannedNotesGrounder,'prepare_evidence',prepare),self.assertRaisesRegex(ValueError,'conflicts'):
            self.run_plan(['Pregunta uno','Pregunta dos'])

    def test_digest_change_aborts_without_published_result(self):
        with self.assertRaisesRegex(RuntimeError,'digest'):
            self.run_plan(['Pregunta uno'],generator=WiringGenerator(change=True))

    def test_direct_control_uses_notes_and_same_one_package_configuration(self):
        gen=WiringGenerator()
        with patch.object(trial,'search',return_value=self.seeds[0]),patch.object(trial,'generate_arm',return_value=dict(status='insufficient_evidence')) as generate:
            result=trial.answer_direct_notes(self.index,'Pregunta',self.units,self.pages,generator=gen,embedder=SimpleNamespace(calls=[]))
        grounder=generate.call_args.args[0]
        self.assertTrue(grounder.closed_note_tails);self.assertEqual(grounder.max_packages,1)
        self.assertFalse(result['metadata']['planner_used'])


if __name__=='__main__':unittest.main()
