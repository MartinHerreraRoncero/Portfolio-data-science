"""Output length regressions; all evidence is synthetic and no model is loaded."""
import copy
import unittest

from experiments import prototype_presentation as presentation
from rag_pensiones import graph_pipeline as pipeline
from rag_pensiones.graph_planning import checker_schema, decision_schema, decode_checker, review_schema
from rag_pensiones.graph_validation import InvalidPlan, check_shape


def assert_no_character_cap(test, value):
    if isinstance(value, dict):
        test.assertNotIn('maxLength', value)
        for child in value.values():
            assert_no_character_cap(test, child)
    elif isinstance(value, list):
        for child in value:
            assert_no_character_cap(test, child)


class ResponseCharacterLimitsTests(unittest.TestCase):
    def test_all_active_output_schemas_are_uncapped(self):
        schemas = [pipeline.INTERPRET_SCHEMA, pipeline.SELECTION_SCHEMA, pipeline.PLAN_SCHEMA,
                   pipeline.CHECK_SCHEMA, pipeline.CLAIM_SCHEMA, pipeline.GENERATE_SCHEMA,
                   pipeline.REVIEW_SCHEMA, decision_schema([], ['metric'], allow_explanation=True),
                   checker_schema({'n1': 'unavailable'}),
                   review_schema([{'claim_id': 'c1'}], [{'need_id': 'n1'}])]
        for schema in schemas:
            assert_no_character_cap(self, schema)

    def test_long_clarifications_and_reasons_preserve_accents_and_endings(self):
        text = 'Explicación íntegra de la medida y sus condiciones. ' * 50 + '¿Qué ámbito necesitas?'
        schema = decision_schema([], ['metric'], allow_explanation=True)
        for result in [dict(action='clarify', field='metric', question=text),
                       dict(action='explain', reason=text), dict(action='unavailable', reason=text)]:
            self.assertIs(check_shape(result, schema), result)
            self.assertTrue((result.get('question') or result['reason']).endswith('¿Qué ámbito necesitas?'))
        checked = decode_checker({'n1': dict(decision='falta_capacidad_acreditada', reason=text)}, {'n1': 'unavailable'})
        self.assertEqual(checked['n1']['reason'], text)

    def test_claims_and_quotes_can_exceed_previous_limits(self):
        quote = 'La evidencia publicada conserva sus condiciones y ámbito. ' * 30
        claim = dict(need_id='n1', text=quote * 2, evidence=[dict(source_id='s1', quote=quote)])
        self.assertIs(check_shape(claim, pipeline.CLAIM_SCHEMA), claim)
        self.assertEqual(claim['evidence'][0]['quote'], quote)

    def test_no_implicit_hundred_thousand_character_cap(self):
        text = 'á' * 100001 + ' final íntegro'
        self.assertEqual(check_shape(text, {'type': 'string'}), text)
        # Explicit bounds supplied by other callers still retain normal schema semantics.
        with self.assertRaises(InvalidPlan):
            check_shape(text, {'type': 'string', 'maxLength': 8})

    def test_removing_caps_keeps_identity_type_and_minimum_checks(self):
        schema = decision_schema([], ['metric'])
        for result in [dict(action='clarify', field='unknown', question='¿Ámbito?'),
                       dict(action='clarify', field='metric', question=''),
                       dict(action='unavailable', reason=12)]:
            with self.subTest(result=result), self.assertRaises(InvalidPlan):
                check_shape(result, schema)
        with self.assertRaises(InvalidPlan):
            check_shape(dict(need_id='n1', text='Texto', evidence=[dict(source_id='s', quote='corta')]), pipeline.CLAIM_SCHEMA)

    def test_long_clarification_reaches_presentation_without_edits(self):
        text = 'Aclaración con información íntegra y acentos. ' * 3000 + 'Fin de la pregunta.'
        shown = presentation.present_report(dict(status='clarification', publication=dict(
            status='clarification', claims=[], reason=text)), database='unused')
        self.assertEqual(shown['answer'], text)
        self.assertEqual(shown['narration'], 'passthrough')

    def test_narration_longer_than_1200_characters_is_preserved(self):
        prefix = 'Explicación detallada con acentos y contexto. ' * 50
        published = 'Una conclusión íntegra y respaldada.'
        report = dict(numeric_facts=[], publication=dict(status='answered', missing_parts=[],
                      claims=[dict(need_id='n1', text=published, evidence=[])], reason=''))
        before = copy.deepcopy(report)
        test = self

        class Client:
            def __init__(self):
                self.calls = []

            def chat(self, messages, schema, stage):
                assert_no_character_cap(test, schema)
                result = dict(text=prefix + '{{TEXT_A}}')
                check_shape(result, schema)
                self.calls.append(dict(stage=stage))
                return result

        shown = presentation.present_report(report, database='unused', client=Client())
        self.assertEqual(shown['answer'], prefix + published)
        self.assertEqual(shown['narration'], 'protected_markers')
        self.assertEqual(report, before)


if __name__ == '__main__':
    unittest.main()
