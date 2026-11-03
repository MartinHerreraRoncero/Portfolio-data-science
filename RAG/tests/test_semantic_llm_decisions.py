"""Offline contracts: models choose meaning; code preserves authority and IDs."""
import copy
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import test_graph_current as current
from test_graph_grounding import need, source
from rag_pensiones import graph_pipeline as p
from rag_pensiones.graph_goals import explanation_need
from rag_pensiones.graph_planning import material_axes, decision_schema, grounding_inputs, apply_grounding
from rag_pensiones.graph_validation import check_shape, InvalidPlan


class SemanticDecisionTests(unittest.TestCase):
    def test_clarification_vocabulary_is_not_a_diagnosis_of_ambiguity(self):
        exact=need(series='Total',unit='euros',periods=['2024'])
        self.assertEqual(material_axes(exact,[]),material_axes(exact,[source(series='Total'),source('b',series='Component')]))
        schema=decision_schema([dict(capability_id='real')],material_axes(exact,[]),edition_choices=['2025'])
        check_shape(dict(action='select',capability_id='real'),schema)
        check_shape(dict(action='clarify',field='metric',question='Which accounting perimeter?'),schema)
        check_shape(dict(action='clarify',field='edition',question='Which publication should I compare?'),schema)
        with self.assertRaises(InvalidPlan):check_shape(dict(action='select',capability_id='invented'),schema)

    def test_explanation_is_an_explicit_model_choice_not_a_fallback(self):
        n=need(metric='Warehouse loss',periods=['2024']);before=copy.deepcopy(n)
        schema=decision_schema([],[],allow_explanation=True)
        check_shape(dict(action='explain',reason='Sources distinguish gross and net losses'),schema)
        with self.assertRaises(InvalidPlan):check_shape(dict(action='explain',reason='Related'),decision_schema([],[]))
        auxiliary=explanation_need(n)
        self.assertEqual(auxiliary['periods'],['2024']);self.assertEqual(auxiliary['series'],n['series']);self.assertEqual(n,before)
        for action in [dict(action='unavailable',reason='No relevant records'),dict(action='clarify',field='metric',question='Which loss?')]:
            services=SimpleNamespace(client=SimpleNamespace(calls=[]))
            nodes=current.CurrentQuestionTests().nodes(services)
            state=dict(query='Warehouse loss in 2024',effective_context=dict(ambiguity_policy='explain_supported_alternatives'),interpretation=dict(needs=[n]),candidates=[])
            with patch.object(p,'_plan_one_need',return_value=(action,[],[],{})):
                result=nodes['plan'](state)
            self.assertEqual(result['interpretation']['needs'],[n])
            self.assertEqual(result['planning_decisions']['n1'],action)

    def test_null_grounding_cannot_erase_an_explicit_unmatched_clarification(self):
        n=need(metric='Warehouse throughput',series='standard operating case',periods=['2024'])
        inputs=grounding_inputs(dict(needs=[n]),[source(series='Reference',periods=['2024'])],'Throughput in 2024')
        out,records=apply_grounding(dict(needs=[n]),inputs,dict(n1=dict(scenario=None,target_years=['2024'])))
        self.assertEqual(out['needs'][0]['series'],n['series'])
        self.assertEqual(records['n1']['decision']['scenario']['action'],'unresolved')
        n['series']=None
        inputs=grounding_inputs(dict(needs=[n]),[source(periods=['2024'])],'Throughput in 2024')
        out,records=apply_grounding(dict(needs=[n]),inputs,dict(n1=dict(scenario=None,target_years=['2024'])))
        self.assertIsNone(out['needs'][0]['series'])
        self.assertEqual(records['n1']['decision']['scenario']['action'],'no_restriction')
