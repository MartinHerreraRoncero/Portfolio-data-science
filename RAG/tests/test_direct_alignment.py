"""Isolation, fixed policy and error accounting for the direct experimental judge."""
import copy
import json
import unittest

from experiments.answer_alignment import REFUSAL
from experiments.direct_alignment import DirectChecker, apply_direct, run_controls, validate_assessment
from test_answer_alignment import Generator, view


def assessment(outcome='accept'):
    return dict(outcome=outcome, reason='La respuesta corresponde a lo solicitado.')


class DirectTests(unittest.TestCase):
    def test_full_texts_are_preserved_without_citations_metadata_or_labels(self):
        original = view()
        original['claims'].append(dict(text='Además, son dos colectivos distintos.', evidence=copy.deepcopy(original['claims'][0]['evidence'])))
        original['claims'][0]['evidence'][0]['quote'] += ' MARCADOR_SOLO_CITA'
        original.update(bindings=['MARCADOR_BINDING'], expected='MARCADOR_GOLD')
        query = '¿Qué cobertura nacional habrá en 2036? 📘'
        gen = Generator(assessment())
        audit = DirectChecker(gen).check(query, original, expected_digest='a'*64)
        self.assertEqual(len(gen.calls), 1)
        self.assertEqual(json.loads(gen.calls[0]['messages'][1]['content']),
                         dict(pregunta=query, respuesta=[c['text'] for c in original['claims']]))
        serialized = json.dumps(gen.calls[0]['messages'])
        for marker in ('MARCADOR_SOLO_CITA', 'MARCADOR_BINDING', 'MARCADOR_GOLD'):
            self.assertNotIn(marker, serialized)
        self.assertEqual(audit['trace']['messages_budget_bytes'], 6500)
        self.assertEqual(set(gen.calls[0]['schema']['properties']), {'outcome', 'reason'})

    def test_all_outcomes_fixed_refusal_and_deep_copies(self):
        for outcome in ('accept', 'reject', 'uncertain'):
            original = view(); original['bindings'] = ['binding']
            before = copy.deepcopy(original)
            audit = DirectChecker(Generator(assessment(outcome))).check('Pregunta', original, expected_digest='a'*64)
            result = apply_direct(original, audit)
            self.assertEqual(original, before)
            self.assertIsNot(result, original)
            if outcome == 'accept':
                self.assertEqual(result, original)
            else:
                self.assertEqual((result['status'], result['claims'], result['bindings'], result['reason']),
                                 ('insufficient_evidence', [], [], REFUSAL))

    def test_prior_abstentions_skip_inference_and_cannot_be_revived(self):
        original = dict(status='insufficient_evidence', claims=[], reason='No hay evidencia regional.')
        gen = Generator(fail=True)
        audit = DirectChecker(gen).check('Pregunta', original, expected_digest='a'*64)
        self.assertEqual(gen.count, 0)
        self.assertEqual(apply_direct(original, audit), original)
        with self.assertRaises(ValueError):
            apply_direct(original, dict(decision='accept', assessment=assessment()))

    def test_invalid_assessments_are_errors_and_cannot_override_decision(self):
        for bad in ({}, dict(outcome='yes', reason='x'), dict(outcome='accept', reason=' '),
                    dict(outcome='accept', reason='x'*301), dict(outcome='accept', reason='x', extra=True)):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                validate_assessment(bad)
        with self.assertRaisesRegex(ValueError, 'disagrees'):
            apply_direct(view(), dict(decision='accept', assessment=assessment('reject')))

    def test_digest_drift_and_oversized_messages_abort_without_semantic_retries(self):
        gen = Generator(assessment(), change=True)
        with self.assertRaisesRegex(RuntimeError, 'digest changed'):
            DirectChecker(gen).check('Pregunta', view(), expected_digest='a'*64)
        self.assertEqual(gen.count, 1)
        gen = Generator(assessment())
        with self.assertRaisesRegex(ValueError, 'budget'):
            DirectChecker(gen).check('Pregunta '*1000, view(), expected_digest='a'*64)
        self.assertEqual(gen.count, 0)

    def test_error_keeps_input_and_raw_response_without_claiming_safe_abstention(self):
        data = dict(schema_version='answer-alignment-controls-v1', notice='Synthetic',
                    cases=[dict(id='x', query='Pregunta', answer=view())])
        gen = Generator(dict(outcome='invalid', reason='Malformed'))
        report = run_controls(data, None, checker=DirectChecker(gen))
        row = report['cases'][0]
        self.assertEqual(report['run_status'], 'error')
        self.assertEqual(row['input_view'], view())
        self.assertIn('raw_response', row['calls'][0])
        self.assertNotIn('output_view', row)
        self.assertEqual(gen.count, 1)

    def test_labels_and_duplicate_ids_are_rejected_before_inference(self):
        base = dict(schema_version='answer-alignment-controls-v1', notice='Synthetic',
                    cases=[dict(id='x', query='Pregunta', answer=view())])
        for change in ('label', 'duplicate'):
            data = copy.deepcopy(base)
            if change == 'label': data['cases'][0]['expected'] = 'accept'
            else: data['cases'].append(copy.deepcopy(data['cases'][0]))
            gen = Generator(assessment())
            with self.assertRaises(ValueError):
                run_controls(data, None, checker=DirectChecker(gen))
            self.assertEqual(gen.count, 0)


if __name__ == '__main__':
    unittest.main()
