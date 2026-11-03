"""Selected entry point uses the original profile and publishes its complete context."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from experiments import query_preserved as query
from rag_pensiones.search import DEFAULT_INDEX_PATH, LEGACY_INDEX_PATH, load_index, SearchHit, validate_index
from test_answer_alignment import Generator


class PreservedQueryTests(unittest.TestCase):
    def inputs(self):
        index = load_index(DEFAULT_INDEX_PATH)
        units = json.loads(Path('evaluation/reconstructed_units_catalog.json').read_text(encoding='utf-8'))
        pages = [json.loads(x) for x in Path('data/processed/airef_2025_pages.jsonl').read_text(encoding='utf-8').splitlines()]
        hits = [SearchHit(i+1, .7, c) for i,(c,_) in enumerate(validate_index(index)[:4])]
        return index, units, pages, hits

    def test_entry_uses_legacy_grounder_and_shared_top4_without_planner_or_fixed_flags(self):
        index, units, pages, hits = self.inputs(); gen = Generator()
        outcome = dict(status='insufficient_evidence', publication=dict(status='insufficient_evidence', claims=[], reason='No hay dato.'))
        with patch.object(query, 'search', return_value=hits) as search, patch.object(query, 'generate_arm', return_value=outcome) as generate:
            report = query.answer_preserved(index, 'Pregunta', units, pages, embedder=SimpleNamespace(calls=[]), generator=gen)
        g = generate.call_args.args[0]
        self.assertIs(type(g), query.PreservedGrounder)
        self.assertFalse(g.closed_note_tails); self.assertEqual(g.max_packages, 1)
        self.assertEqual(search.call_args.kwargs['top_k'], 4)
        self.assertTrue(generate.call_args.kwargs['preserve'])
        self.assertFalse(report['metadata']['planner_used'])
        self.assertEqual(report['outcome'], outcome)

    def test_invalid_query_index_or_source_provenance_stops_before_model_calls(self):
        index, units, pages, _ = self.inputs()
        for mode in ['query', 'index', 'catalog']:
            i=copy.deepcopy(index); u=copy.deepcopy(units); text='Pregunta'
            if mode=='query': text=' '
            elif mode=='index': i=load_index(LEGACY_INDEX_PATH)
            else: u['units'][0]['spans'][0]['quote']='Cita inventada.'
            gen=Generator()
            with patch.object(query, 'search') as search, self.assertRaises(ValueError):
                query.answer_preserved(i,text,u,pages,generator=gen)
            self.assertEqual(gen.count,0); search.assert_not_called()

    def test_human_output_includes_restored_antecedent_quotes_and_physical_pages(self):
        old=json.loads(Path('evaluation/preserved_results.json').read_text(encoding='utf-8'))
        outcome=next(c for c in old['cases'] if c['id']=='n04')['conditions']['preserved']
        before=copy.deepcopy(outcome)
        text=query.format_answer(dict(outcome=outcome))
        self.assertIn('2011',text); self.assertIn('pág. 96',text)
        for ev in outcome['publication']['claims'][0]['evidence']: self.assertIn(ev['quote'],text)
        self.assertIn('Clases Pasivas del Estado refleja',text)
        self.assertEqual(outcome,before)

    def test_abstention_and_technical_error_are_not_rewritten_as_answers(self):
        reason='El informe no ofrece desglose para Andalucía.'
        report=dict(outcome=dict(status='insufficient_evidence',publication=dict(status='insufficient_evidence',claims=[],reason=reason)))
        self.assertEqual(query.format_answer(report),reason)
        self.assertEqual(query.format_answer(dict(outcome=dict(status='error',error='Digest changed'))),'Error: Digest changed')


if __name__ == '__main__': unittest.main()
