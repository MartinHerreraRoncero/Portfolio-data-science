"""Source-tail certification, literal unions and immutable budgets for the correction."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from experiments import compare_context_fix as trial
from experiments.fixed_context import FixedPreservedGrounder, restore_packages
from experiments.reconstructed_units import bundle_catalog, make_hits
from experiments.validate_reranker import canonical_hash
from rag_pensiones.search import DEFAULT_INDEX_PATH, load_index
from test_answer_alignment import Generator
import test_reconstructed_units as reconstructed_tests
import test_preserved_evidence as preserved_tests
import test_compare_preserved as comparison_tests


def citation(text='Una cifra del documento.', sid='S1'):
    return dict(source_id=sid, quote=text)


def records():
    a, b, c = citation(), citation('Un segundo dato literal.'), citation('Un tercer dato literal.')
    return [dict(id='P1', unit_id='U1', evidence=[a, b]), dict(id='P2', unit_id='U2', evidence=[a, c])]


def selected(ids):
    return dict(status='selected', ids=ids, reason='')


class ContextFixTests(unittest.TestCase):
    def unit(self, text, kind='note', closed=True):
        p = reconstructed_tests.page(1, text)
        u = dict(id='U1', kind=kind, spans=[reconstructed_tests.span(p)])
        if closed: u['source_boundary'] = 'closed'
        _, mapping = make_hits([u], {('d', 1): p})
        return u, mapping

    def test_certified_note_tail_survives_verbatim_and_default_legacy_does_not_change(self):
        text = 'Una persona puede cobrar varias pensiones. La ratio se mantiene alrededor del 0,9'
        u, mapping = self.unit(text)
        self.assertEqual(len(bundle_catalog([u], mapping)['records']), 1)
        fixed = bundle_catalog([u], mapping, closed_note_tails=True)
        self.assertEqual(len(fixed['records']), 2)
        self.assertEqual(fixed['records'][1]['evidence'][1]['quote'], 'La ratio se mantiene alrededor del 0,9')
        self.assertTrue(all(e['quote'] in text for r in fixed['records'] for e in r['evidence']))

    def test_unreviewed_note_and_unfinished_body_remain_excluded(self):
        for kind, closed in [('note', False), ('body', False), ('body', True)]:
            u, mapping = self.unit('La primera frase está completa. Una segunda frase queda en la medida', kind, closed)
            self.assertEqual(len(bundle_catalog([u], mapping, closed_note_tails=True)['records']), 1)

    def test_complete_note_without_any_terminal_punctuation_is_selectable(self):
        text = 'Referencia documental completa sin un punto final'
        u, mapping = self.unit(text)
        self.assertEqual(bundle_catalog([u], mapping)['records'], [])
        fixed = bundle_catalog([u], mapping, closed_note_tails=True)
        self.assertEqual(fixed['records'][0]['evidence'], [citation(text)])

    def test_closed_cross_page_note_keeps_two_exact_quotes(self):
        p1 = reconstructed_tests.page(1, 'La nota identifica un indicador')
        p2 = reconstructed_tests.page(2, 'y termina en el valor 0,9')
        u = dict(id='U1', kind='note', source_boundary='closed', spans=[reconstructed_tests.span(p1), reconstructed_tests.span(p2)])
        _, mapping = make_hits([u], {('d', 1): p1, ('d', 2): p2})
        ev = bundle_catalog([u], mapping, closed_note_tails=True)['records'][0]['evidence']
        self.assertEqual([e['quote'] for e in ev], [p1['text'], p2['text']])

    def test_body_cannot_receive_note_boundary_certificate(self):
        units, pages, _ = preserved_tests.fixture()
        units['units'][0].update(kind='body', source_boundary='closed')
        with self.assertRaisesRegex(ValueError, 'restricted'):
            FixedPreservedGrounder(units, pages, generator=Generator())

    def test_complementary_union_is_stable_deduplicated_and_does_not_mutate_records(self):
        data = records(); before = copy.deepcopy(data)
        union = restore_packages(selected(['P1', 'P2']), data)
        self.assertEqual(union['evidence'], [data[0]['evidence'][0], data[0]['evidence'][1], data[1]['evidence'][1]])
        self.assertEqual(data, before)

    def test_duplicate_unknown_or_excessive_package_ids_are_errors(self):
        for ids in [['P1', 'P1'], ['missing'], [], ['P1', 'P2', 'P3', 'P4'], [{}]]:
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                restore_packages(selected(ids), records())

    def test_union_reference_limit_and_utf8_byte_boundary_are_enforced(self):
        data = [dict(id='P1', evidence=[citation('é'*400, 'S1'), citation('é'*400, 'S2')]), dict(id='P2', evidence=[citation('é'*400, 'S3')])]
        self.assertEqual(len(restore_packages(selected(['P1', 'P2']), data)['evidence']), 3)
        data[1]['evidence'][0]['quote'] += 'é'
        with self.assertRaisesRegex(ValueError, 'limits'): restore_packages(selected(['P1', 'P2']), data)
        data = records(); data[1]['evidence'].append(citation('Un cuarto dato literal.', 'S2'))
        with self.assertRaisesRegex(ValueError, 'limits'): restore_packages(selected(['P1', 'P2']), data)

    def test_abstention_is_distinct_from_invalid_state(self):
        p = dict(status='insufficient_evidence', ids=[], reason='No caben las citas necesarias.')
        self.assertEqual(restore_packages(p, records())['reason'], p['reason'])
        for bad in [p | dict(ids=['P1']), p | dict(reason=' '), selected(['P1']) | dict(expected=True)]:
            with self.assertRaises(ValueError): restore_packages(bad, records())

    def test_selection_schema_and_validator_allow_joint_evidence_with_single_call(self):
        units, pages, _ = preserved_tests.fixture(); gen = Generator(selected(['P1', 'P2']))
        g = FixedPreservedGrounder(units, pages, generator=gen)
        g.catalog = dict(records=records(), omitted=[]); g.selections = []
        payload, _ = g._chat_step('selection', [dict(role='user', content=json.dumps(dict(pregunta='Dos datos')))], {}, 'a'*64)
        self.assertEqual(len(payload['evidence']), 3)
        self.assertEqual(len(gen.calls), 1)
        self.assertEqual(gen.calls[0]['schema']['anyOf'][0]['properties']['ids']['maxItems'], 3)
        self.assertEqual(g.selections[0]['payload'], selected(['P1', 'P2']))

    def test_invalid_union_retains_selection_raw_and_digest_drift_aborts(self):
        units, pages, _ = preserved_tests.fixture(); gen = Generator(selected(['P1', 'P1']))
        g = FixedPreservedGrounder(units, pages, generator=gen); g.catalog = dict(records=records(), omitted=[]); g.selections=[]
        with self.assertRaises(ValueError): g._chat_step('repair_selection', [dict(role='user', content='{}')], {}, 'a'*64)
        self.assertEqual(g.selections[0]['payload']['ids'], ['P1', 'P1'])
        self.assertIn('raw_response', gen.calls[0])
        gen.change=True
        with self.assertRaisesRegex(RuntimeError, 'digest'):
            g._chat_step('selection', [dict(role='user', content='{}')], {}, 'a'*64)

    def inputs(self):
        index, hits, _, _ = comparison_tests.RunnerTests().setup_trial()
        cases=[dict(id=f'q{i}', query=f'Pregunta {i}', split='known', hits=[dict(rank=h.rank,score=h.score,chunk=vars(h.chunk)) for h in hits]) for i in range(6)]
        data=dict(schema_version='context-fix-inputs-v1', index_hash=canonical_hash(index), generator_digest='a'*64, sources={}, cases=cases)
        return index, hits, data

    def test_frozen_hits_shared_and_fresh_arms_isolated_even_when_one_fails(self):
        index, hits, data = self.inputs()
        models={k:comparison_tests.Grounder(fail=k=='fixed') for k in trial.ARMS}
        with patch.object(trial, 'validate_catalog'), patch.object(trial, 'validate_source_hits'), patch('experiments.compare_system.validate_grounded_result'):
            r=trial.run(data,index,{'units':[]},[],None,models=models)
        self.assertEqual(len({tuple(c['condition_order']) for c in r['cases']}),6)
        self.assertTrue(all(h==hits for g in models.values() for h in g.seen))
        self.assertTrue(all(c['conditions']['baseline']['status']=='insufficient_evidence' and c['conditions']['fixed']['status']=='error' for c in r['cases']))

    def test_tampered_hits_labels_and_pinned_digest_fail_before_generation(self):
        index, _, base = self.inputs()
        for mode in ['text','labels','index']:
            data=copy.deepcopy(base)
            if mode=='text':data['cases'][0]['hits'][0]['chunk']['text']='Tampered'
            elif mode=='labels':data['cases'][0]['expected']='accept'
            else:data['index_hash']='changed'
            with self.assertRaises(ValueError):trial.validate_inputs(data,index)
        models={k:comparison_tests.Grounder() for k in trial.ARMS}
        base['generator_digest']='b'*64
        with patch.object(trial, 'validate_catalog'),patch.object(trial, 'validate_source_hits'),self.assertRaisesRegex(ValueError,'digests'):
            trial.run(base,index,{'units':[]},[],None,models=models)
        self.assertTrue(all(not g.seen for g in models.values()))

    def test_canonical_page_text_identity_and_offsets_are_required(self):
        data=json.loads(Path('evaluation/context_fix_inputs.json').read_text(encoding='utf-8'))
        pages=[json.loads(x) for x in Path('data/processed/airef_2025_pages.jsonl').read_text(encoding='utf-8').splitlines()]
        trial.validate_source_hits(data,pages)
        data['cases'][0]['hits'][0]['chunk']['text']='Texto fuera de la página'
        with self.assertRaisesRegex(ValueError,'provenance'):trial.validate_source_hits(data,pages)


if __name__=='__main__': unittest.main()
