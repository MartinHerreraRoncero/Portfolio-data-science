import unittest
from rag_pensiones.graph_goals import explanation_need,text_stage_question
from rag_pensiones.graph_validation import validate_text_claims_isolated

class AnswerRelevanceTests(unittest.TestCase):
    def test_empty_auxiliary_focus_does_not_suppress_original_text_goal(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from test_graph_current import CurrentQuestionTests
        from rag_pensiones import graph_pipeline as pipeline
        original=dict(need_id='n1',kind='data',metric='Saldo X',query_span='Calcula X',periods=[],series=None,unit='unspecified')
        auxiliary=explanation_need(original)
        native=dict(need_id='native',kind='text',metric='Explica Y',book=None)
        quote='El proceso Y describe entradas y salidas documentadas.'
        source=dict(source_id='s',text=quote,book='2025',doc_id='doc',page_number=1,source_file='Fuente.pdf',source_path='Fuente.pdf',source_sha256='abc')
        client=SimpleNamespace(calls=[],chat=lambda *args:self.fail('Unexpected auxiliary source selection'))
        nodes=CurrentQuestionTests().nodes(SimpleNamespace(client=client))
        state=dict(query='Calcula X y explica Y',effective_context={},interpretation=dict(needs=[original,native,auxiliary]),documents=[source],text_source_focus=[])
        def generate(services,stage,payload,schema):
            self.assertIn('evidence_options',payload);self.assertNotIn('passages',payload)
            self.assertNotIn('definition_variant',schema['properties']['claims']['items']['properties'])
            return dict(claims=[dict(need_id='native',text=quote,evidence=[dict(evidence_id=payload['evidence_options'][0]['evidence_id'])])])
        with patch.object(pipeline,'_chat',side_effect=generate):result=nodes['generate'](state)
        self.assertNotIn('errors',result)
        self.assertEqual([c['need_id']for c in result['text_claims']],['native'])
        state.update(result,repair_text_need_id=auxiliary['need_id'])
        with patch.object(pipeline,'_chat',side_effect=AssertionError('Empty auxiliary selection must not generate')):
            retried=nodes['generate'](state)
        self.assertEqual([c['need_id']for c in retried['text_claims']],['native'])

    def test_pending_action_comes_from_requested_goals_not_auxiliary_explanation(self):
        from experiments.prototype_chat import report_answer_kind
        report=dict(interpretation=dict(needs=[dict(need_id='n1',kind='data'),dict(need_id='n1_explanation',kind='text')]))
        self.assertEqual(report_answer_kind(report),'quantity')
        report['interpretation']['needs']=[dict(need_id='n1',kind='text')]
        self.assertEqual(report_answer_kind(report),'text')
        report['interpretation']['needs'].append(dict(need_id='n2',kind='data'))
        self.assertEqual(report_answer_kind(report),'unspecified')

    def test_pending_action_is_bounded_metadata_and_does_not_mutate_preferences(self):
        from experiments.prototype_chat import combined_context
        base=dict(history=[dict(role='user',content='Preferencia')])
        context=combined_context(base,[],pending_answer_kind='quantity')
        self.assertEqual(context['pending_answer_kind'],'quantity')
        self.assertNotIn('pending_answer_kind',base)
        with self.assertRaises(ValueError):combined_context(base,[],pending_answer_kind='invented')

    def test_pending_action_does_not_leak_between_conversations(self):
        from experiments.prototype_chat import ConversationStore
        store=ConversationStore();first=store.create();second=store.create()
        store.session(first)['answer_kind']='quantity'
        self.assertNotIn('answer_kind',store.session(second))
        store.clear(first)
        self.assertNotIn(first,store.sessions)

    def test_empty_semantic_selection_does_not_restore_all_sources(self):
        from rag_pensiones.graph_evidence import evidence_options
        documents=[dict(source_id='s',text='Un pasaje que no responde a la petición.')]
        self.assertEqual(evidence_options(documents,[]),[])
        self.assertTrue(evidence_options(documents,None))

    def test_literal_choices_remain_substrings_and_do_not_adopt_model_quotes(self):
        from rag_pensiones.graph_evidence import evidence_options,attach_evidence
        documents=[dict(source_id='source',text=('Entrada y salida son componentes distintos. '*30))]
        options=evidence_options(documents,['source'],byte_budget=500)
        self.assertTrue(options)
        self.assertLessEqual(sum(len(o['quote'].encode())for o in options),500)
        self.assertTrue(all(o['quote']in documents[0]['text']for o in options))
        claims=[dict(text='Interpretación condicional',evidence=[dict(evidence_id=options[0]['evidence_id'])])]
        attached=attach_evidence(claims,options)
        self.assertEqual(attached[0]['evidence'][0]['quote'],options[0]['quote'])
        self.assertIn('evidence_id',claims[0]['evidence'][0])

    def test_ambiguous_original_request_survives_auxiliary_goal(self):
        original=dict(need_id='n1',kind='data',metric='Saldo disponible del almacén',query_span='Qué saldo queda',periods=[],series=None,unit='unspecified')
        auxiliary=explanation_need(original,['Entradas','Salidas'])
        self.assertIn(original['metric'],auxiliary['metric'])
        self.assertIn(original['query_span'],auxiliary['query_span'])
        self.assertEqual(text_stage_question('Qué saldo queda',[auxiliary]),'Qué saldo queda')
        self.assertEqual(original['kind'],'data')

    def test_unknown_identifier_does_not_abort_other_supported_claims(self):
        quote='La diferencia entre entradas y salidas mide el saldo.'
        source=dict(source_id='s1',text=quote,book='2025',doc_id='doc',page_number=1,source_file='Fuente.pdf',source_path='Fuente.pdf',source_sha256='abc')
        needs=[dict(need_id='active',kind='text',book=None)]
        claims=[dict(need_id='invented',text='Otra respuesta',evidence=[]),dict(need_id='active',text=quote,evidence=[dict(source_id='s1',quote=quote)])]
        accepted,rejected=validate_text_claims_isolated(claims,needs,[source])
        self.assertEqual([c['need_id']for c in accepted],['active'])
        self.assertEqual(rejected[0]['code'],'unknown_need_id')
