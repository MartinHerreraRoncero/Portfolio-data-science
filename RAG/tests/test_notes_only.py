"""The ablation changes note closure only and retains single-package enforcement."""
import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from experiments import compare_notes_only as trial
from experiments.reconstructed_units import expand_seeds
from rag_pensiones.search import DEFAULT_INDEX_PATH, load_index, SearchHit
from rag_pensiones.chunking import ChunkRecord
from test_answer_alignment import Generator
from test_fixed_context import records, selected
from test_preserved_evidence import fixture


class NotesOnlyTests(unittest.TestCase):
    def test_tail_is_literal_and_control_catalog_and_input_stay_unchanged(self):
        units = json.loads(Path('evaluation/context_fix_units.json').read_text(encoding='utf-8'))
        pages = [json.loads(x) for x in Path('data/processed/airef_2025_pages.jsonl').read_text(encoding='utf-8').splitlines()]
        before = copy.deepcopy(units)
        models = trial.build_models(units, pages)
        frozen = json.loads(Path('evaluation/context_fix_inputs.json').read_text(encoding='utf-8'))
        case = next(c for c in frozen['cases'] if c['id'] == 'v03')
        hits = [SearchHit(h['rank'], h['score'], ChunkRecord(**h['chunk'])) for h in case['hits']]
        catalogs = {}
        for key, g in models.items():
            _, catalog, _ = expand_seeds(hits, g.extracted, g.pages, case['query'], closed_note_tails=g.closed_note_tails)
            catalogs[key] = catalog
            self.assertEqual(g.max_packages, 1)
        self.assertNotIn('0,9', str(catalogs['legacy']))
        self.assertIn('0,9', str(catalogs['notes_only']))
        self.assertEqual(units, before)
        self.assertFalse(any('source_boundary' in u for u in models['legacy'].extracted['units']))

    def test_single_package_schema_and_validator_reject_two_for_both_selection_stages(self):
        units, pages, _ = fixture()
        for stage in ['selection', 'repair_selection']:
            gen = Generator(selected(['P1', 'P2']))
            grounder = trial.NotesOnlyGrounder(units, pages, generator=gen)
            grounder.catalog = dict(records=records(), omitted=[]); grounder.selections = []
            with self.assertRaises(ValueError):
                grounder._chat_step(stage, [dict(role='user', content='{}')], {}, 'a'*64)
            schema = gen.calls[0]['schema']
            self.assertEqual(schema['anyOf'][0]['properties']['ids']['maxItems'], 1)

    def test_non_note_certificate_rejected(self):
        units, pages, _ = fixture()
        units['units'][0].update(kind='body', source_boundary='closed')
        with self.assertRaisesRegex(ValueError, 'restricted'):
            trial.NotesOnlyGrounder(units, pages, generator=Generator())

    def test_fresh_paired_arms_share_hits_alternate_and_pin_digest(self):
        units = json.loads(Path('evaluation/context_fix_units.json').read_text(encoding='utf-8'))
        pages = [json.loads(x) for x in Path('data/processed/airef_2025_pages.jsonl').read_text(encoding='utf-8').splitlines()]
        inputs = json.loads(Path('evaluation/context_fix_inputs.json').read_text(encoding='utf-8'))
        inputs['cases'] = inputs['cases'][:2]; inputs['generator_digest'] = 'a'*64
        index = load_index(DEFAULT_INDEX_PATH); models = trial.build_models(units, pages)
        for g in models.values(): g.generator = Generator()
        with patch.object(trial, 'generate_arm', return_value=dict(status='insufficient_evidence')) as generate:
            result = trial.run(inputs, index, units, pages, None, models=models)
        self.assertEqual([c['condition_order'] for c in result['cases']], [list(trial.ARMS), list(reversed(trial.ARMS))])
        self.assertEqual(generate.call_args_list[0].args[2], generate.call_args_list[1].args[2])
        self.assertTrue(all(c.kwargs['preserve'] for c in generate.call_args_list))
        inputs['generator_digest'] = 'b'*64
        with patch.object(trial, 'generate_arm') as generate, self.assertRaisesRegex(ValueError, 'digests'):
            trial.run(inputs, index, units, pages, None, models=models)
        generate.assert_not_called()


if __name__ == '__main__': unittest.main()
