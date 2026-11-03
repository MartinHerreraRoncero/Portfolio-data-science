import unittest
from unittest.mock import patch
from types import SimpleNamespace
import copy
import json
from rag_pensiones.graph_conversation import resolve

class ConversationTests(unittest.TestCase):
    def test_repair_validation_and_action_calls_receive_only_their_own_instructions(self):
        from rag_pensiones.graph_conversation import task_prompt
        repair=task_prompt(dict(task='repair_resolution'))
        validation=task_prompt(dict(task='validate_resolution'))
        action=task_prompt(dict(task='classify_action'))
        dependency=task_prompt(dict(task='classify_dependency'))
        self.assertIn('CUALITATIVA',repair);self.assertNotIn('preserve_previous_action',repair)
        self.assertIn('proposed_request',validation);self.assertNotIn('reference_options',validation)
        self.assertIn('preserve_previous_action',action);self.assertNotIn('proposed_request',action)
        self.assertIn('dependency',dependency);self.assertNotIn('proposed_request',dependency)

    def test_numeric_option_label_uses_original_requests_and_selected_meaning(self):
        context=dict(history=[dict(role='user',content='Calcula el saldo'),dict(role='assistant',content='(1) entradas menos salidas; (2) entradas menos devoluciones')])
        calls=[]
        def chat(payload,schema):
            calls.append(payload)
            if payload.get('task')=='classify_dependency':return dict(dependency='dependent')
            if payload.get('task')=='classify_action':return dict(kind='quantity',preserve_previous_action=True)
            return dict(reference_id='reference_1',status='resolved',question='Calcula entradas menos salidas'if 'repair_request'in payload else'Calcula la opción 1: entradas menos salidas',kind='quantity',clarification='')
        result=resolve('la primera',context,chat)
        self.assertEqual(len(calls),3)
        self.assertEqual(calls[0].get('task'),'classify_dependency')
        self.assertEqual(calls[2]['history_user_only'],['Calcula el saldo'])
        self.assertNotIn('1',result['question'])
        self.assertIn('Calcula el saldo',result['question'])
        self.assertIn('entradas menos salidas',result['question'])

    def test_first_question_has_no_extra_call_and_preserves_context(self):
        context=dict(history=[dict(role='user',content='Preferencia general')])
        result=resolve('Pregunta inicial',context,lambda *args:self.fail('Unexpected model call'))
        self.assertEqual(result,dict(question='Pregunta inicial',context=context,resolution=None))

    def test_resolution_compacts_assistant_but_preserves_selected_meaning(self):
        context=dict(publication_policy='latest_supported',history=[dict(role='user',content='Quiero calcular un saldo'),dict(role='assistant',content='Uno: entradas menos salidas')])
        def chat(payload,*args):
            if payload.get('task')=='classify_dependency':return dict(dependency='dependent')
            if payload.get('task')=='validate_resolution':return dict(valid=True,clarification='')
            return dict(reference_id='',status='resolved',question='Calcula entradas menos salidas',kind='quantity',clarification='')
        result=resolve('me refiero a la versión uno',context,chat)
        self.assertEqual(result['question'],'Calcula entradas menos salidas')
        self.assertEqual(result['context']['publication_policy'],'latest_supported')
        self.assertNotIn('assistant',str(result['context']))
        self.assertEqual(result['context']['history'][-1]['content'],'me refiero a la versión uno')

    def test_ambiguous_reference_is_not_guessed(self):
        context=dict(history=[dict(role='assistant',content='Varias opciones')])
        def chat(payload,*args):
            if payload.get('task')=='classify_dependency':return dict(dependency='dependent')
            return dict(reference_id='',status='ambiguous',question='',kind='unspecified',clarification='¿Cuál opción?')
        result=resolve('esa',context,chat)
        self.assertEqual(result['resolution']['status'],'ambiguous')

    def test_assistant_numeric_facts_are_not_adopted(self):
        context=dict(history=[dict(role='assistant',content='El resultado es 999999 en 2026')])
        def chat(payload,*args):
            if payload.get('task')=='classify_dependency':return dict(dependency='dependent')
            return dict(reference_id='',status='resolved',question='Saldo de 999999 en 2026',kind='quantity',clarification='')
        result=resolve('dame el saldo',context,chat)
        self.assertEqual(result['resolution']['status'],'ambiguous')
        self.assertNotIn('999999',str(result['context']))

    def test_qualitative_antecedent_repair_keeps_untrusted_number_out_of_retrieval(self):
        context=dict(history=[dict(role='user',content='Explica el indicador'),dict(role='assistant',content='El índice de durabilidad alcanza el 33%.')])
        calls=[]
        def chat(payload,schema):
            calls.append(payload)
            if payload.get('task')=='classify_dependency':return dict(dependency='dependent')
            if payload.get('task')=='validate_resolution':return dict(valid=True,clarification='')
            question='Explica el índice de durabilidad'if payload.get('task')=='repair_resolution'else'Explica el índice de durabilidad del 33%'
            return dict(reference_id='',status='resolved',question=question,kind='text',clarification='')
        result=resolve('explica ese índice',context,chat)
        self.assertEqual(result['question'],'Explica el índice de durabilidad')
        self.assertEqual([c.get('task')for c in calls],['classify_dependency',None,'repair_resolution','validate_resolution'])
        self.assertEqual(calls[2]['allowed_user_numbers'],[])
        self.assertNotIn('33',str(result['context']))

    def test_independent_topic_is_preserved(self):
        context=dict(history=[dict(role='assistant',content='Opciones anteriores')])
        calls=[]
        def chat(payload,*args):
            calls.append(payload)
            if payload.get('task')=='classify_dependency':return dict(dependency='independent')
            return dict(reference_id='',status='resolved',question='Explica el viento',kind='text',clarification='')
        result=resolve('Explica el viento',context,chat)
        self.assertEqual(result['question'],'Explica el viento')
        self.assertEqual(result['resolution']['kind'],'unspecified')
        self.assertEqual(len(calls),1)
        self.assertEqual(calls[0].get('task'),'classify_dependency')

    def test_replay_uses_original_execution_context_not_compacted_context(self):
        from rag_pensiones.graph_validation import replay_graph_report
        original=dict(history=[dict(role='assistant',content='Opciones anteriores')])
        report=dict(query='La primera',context=original,execution_context=original,effective_context=dict(history=[]),manifest={},stage_calls=[],session_id='test')
        received=[]
        def replay(query,**kwargs):received.append(kwargs['context']);return copy.deepcopy(report)
        with patch('rag_pensiones.graph_validation.runtime_manifest',return_value={}),patch('rag_pensiones.graph_pipeline.answer_graph',side_effect=replay):
            replay_graph_report(report,services=SimpleNamespace())
        self.assertEqual(received,[original])

    def test_rigid_presentation_patches_are_removed(self):
        from experiments import prototype_presentation as presentation
        self.assertFalse(hasattr(presentation,'reviewed_funding_context'))
        self.assertFalse(hasattr(presentation,'related_source_excerpts'))
        result=presentation.present_report(dict(publication=dict(reference_id='',status='insufficient_evidence',claims=[]),retrieval=dict(documents=[])),database=None)
        self.assertEqual(result['related_passages'],[])

    def test_ambiguous_resolution_uses_real_checkpoint_interrupt(self):
        from rag_pensiones.graph_pipeline import build_graph
        from langgraph.checkpoint.memory import MemorySaver
        def mock_chat(messages,schema,stage):
            payload=json.loads(messages[-1]['content'])
            if payload.get('task')=='classify_dependency':return dict(dependency='dependent')
            return dict(reference_id='',status='ambiguous',question='',kind='unspecified',clarification='¿Qué opción eliges?')
        client=SimpleNamespace(calls=[],chat=mock_chat)
        graph=build_graph(SimpleNamespace(client=client),checkpointer=MemorySaver())
        context=dict(history=[dict(role='assistant',content='Varias listas posibles')])
        config=dict(configurable=dict(thread_id='ambiguous_test'))
        state=graph.invoke(dict(query='esa',context=context,effective_context=context,errors=[]),config)
        self.assertTrue(state['__interrupt__'])
        self.assertEqual(graph.get_state(config).next,('clarify',))

    def test_numbered_options_are_generic_and_unspecified_kind_is_not_explanation(self):
        from rag_pensiones.graph_conversation import options
        context=dict(history=[dict(role='user',content='Calcula el saldo'),dict(role='assistant',content='(1) entradas menos salidas; (2) existencias finales. Más contexto.')])
        self.assertEqual([o['text']for o in options(context)],['entradas menos salidas','existencias finales.'])
        def chat(payload,*args):
            if payload.get('task')=='classify_dependency':return dict(dependency='dependent')
            if payload.get('task')=='classify_action':return dict(kind='unspecified',preserve_previous_action=True)
            return dict(reference_id='reference_1',status='resolved',question='Saldo solicitado',kind='unspecified',clarification='')
        result=resolve('la primera',context,chat)
        self.assertIn('entradas menos salidas',result['question']);self.assertNotIn('Explicar',result['question'])

    def test_selected_option_does_not_authorize_assistant_numeric_facts(self):
        context=dict(history=[dict(role='assistant',content='(1) Saldo en 2099: 999999 euros; (2) Otra medida.')])
        def chat(payload,*args):
            if payload.get('task')=='classify_dependency':return dict(dependency='dependent')
            if payload.get('task')=='classify_action':return dict(kind='quantity',preserve_previous_action=True)
            return dict(reference_id='reference_1',status='resolved',question='Saldo solicitado',kind='quantity',clarification='')
        with self.assertRaisesRegex(ValueError,'numeric facts'):
            resolve('la primera',context,chat)

    def test_assistant_definition_does_not_change_pending_calculation(self):
        context=dict(pending_answer_kind='quantity',history=[dict(role='user',content='Calcula saldo'),dict(role='assistant',content='1. Interpretación: entradas menos salidas')])
        def chat(payload,schema):
            if payload.get('task')=='classify_dependency':return dict(dependency='dependent')
            if payload.get('task')=='classify_action':return dict(kind='quantity',preserve_previous_action=True)
            return dict(reference_id='reference_1',status='resolved',question='Explica esa interpretación',kind='text',clarification='')
        result=resolve('la primera',context,chat)
        self.assertEqual(result['resolution']['kind'],'quantity')
        self.assertIn('Calcular el valor',result['question'])
        self.assertNotIn('Explica esa interpretación',result['question'])

    def test_new_explicit_explanation_can_change_pending_action(self):
        context=dict(pending_answer_kind='quantity',history=[dict(role='user',content='Calcula saldo'),dict(role='assistant',content='1. Entradas menos salidas')])
        def chat(payload,schema):
            if payload.get('task')=='classify_dependency':return dict(dependency='dependent')
            if payload.get('task')=='classify_action':return dict(kind='text',preserve_previous_action=False)
            return dict(reference_id='reference_1',status='resolved',question='Calcula saldo',kind='quantity',clarification='')
        result=resolve('explícame la primera',context,chat)
        self.assertEqual(result['resolution']['kind'],'text')
        self.assertIn('Explicar el significado',result['question'])

    def test_unresolved_selection_cannot_silently_repeat_the_old_question(self):
        context=dict(history=[dict(role='assistant',content='Una explicación sin lista')])
        def chat(payload,schema):
            if payload.get('task')=='classify_dependency':return dict(dependency='dependent')
            if payload.get('task')=='validate_resolution':return dict(valid=False,clarification='¿Qué opción?')
            return dict(reference_id='',status='resolved',question='Pregunta anterior',kind='unspecified',clarification='')
        result=resolve('versión uno',context,chat)
        self.assertEqual(result['resolution']['status'],'ambiguous')
        self.assertIn('model_resolution',result['resolution'])

    def test_unchanged_opaque_reference_is_verified_and_cannot_silently_retrieve(self):
        context=dict(history=[dict(role='user',content='¿Qué concluye el ensayo?'),dict(role='assistant',content='El indicador térmico no mide la durabilidad.')])
        calls=[]
        def chat(payload,schema):
            calls.append(payload)
            if payload.get('task')=='classify_dependency':return dict(dependency='dependent')
            if payload.get('task')=='validate_resolution':return dict(valid=False,clarification='¿Te refieres al indicador térmico?')
            return dict(reference_id='',status='resolved',question='¿Por qué no basta ese indicador?',kind='text',clarification='')
        result=resolve('¿Por qué no basta ese indicador?',context,chat)
        self.assertEqual(len(calls),3)
        self.assertEqual([c.get('task')for c in calls],['classify_dependency',None,'validate_resolution'])
        self.assertEqual(calls[-1]['proposed_request'],'¿Por qué no basta ese indicador?')
        self.assertEqual(result['resolution']['status'],'ambiguous')

    def test_valid_mixed_request_does_not_require_a_numbered_list(self):
        context=dict(history=[dict(role='user',content='Calcula X'),dict(role='assistant',content='X es un indicador')])
        def chat(payload,schema):
            if payload.get('task')=='classify_dependency':return dict(dependency='dependent')
            if payload.get('task'):return dict(valid=True,clarification='')
            return dict(reference_id='',status='resolved',question='Calcula X y explica su causa',kind='unspecified',clarification='')
        result=resolve('dame ese valor y explica su causa',context,chat)
        self.assertEqual(result['resolution']['status'],'resolved')
        self.assertEqual(result['question'],'Calcula X y explica su causa')

    def test_synthetic_same_topic_new_metric_is_independent_and_bypasses_rewrite(self):
        context=dict(
            history=[
                dict(role='user',content='¿Qué dicen las fuentes sobre la eficiencia del almacén?'),
                dict(role='assistant',content='El almacén registró 3.175 entradas en 2024.')
            ]
        )
        calls=[]
        def chat(payload,schema):
            calls.append(payload)
            if payload.get('task')=='classify_dependency':
                return dict(dependency='independent')
            self.fail('Bypassed calls should not be reached')
        result=resolve('¿Cuál es el coste actual del almacén?',context,chat)
        self.assertEqual(result['question'],'¿Cuál es el coste actual del almacén?')
        self.assertEqual(result['resolution']['kind'],'unspecified')
        self.assertEqual(result['resolution']['status'],'resolved')
        self.assertEqual(len(calls),1)
        self.assertEqual(calls[0].get('task'),'classify_dependency')
        self.assertEqual(result['context']['history'],[dict(role='user',content='¿Cuál es el coste actual del almacén?')])
        self.assertNotIn('3.175',str(result['context']))

    def test_synthetic_unrelated_new_topic_is_independent(self):
        context=dict(
            history=[
                dict(role='user',content='Explica la capacidad de la batería'),
                dict(role='assistant',content='Existen varios indicadores eléctricos.')
            ]
        )
        calls=[]
        def chat(payload,schema):
            calls.append(payload)
            if payload.get('task')=='classify_dependency':
                return dict(dependency='independent')
            self.fail('Should not be called')
        result=resolve('¿Qué distancia hay a la Luna?',context,chat)
        self.assertEqual(result['question'],'¿Qué distancia hay a la Luna?')
        self.assertEqual(result['resolution']['kind'],'unspecified')
        self.assertEqual(len(calls),1)

    def test_synthetic_no_old_filter_leakage_and_history_immutability(self):
        orig_history=[
            dict(role='user',content='Calcula el saldo del almacén en 2020'),
            dict(role='assistant',content='Quedaban 1000 unidades.')
        ]
        context=dict(
            history=copy.deepcopy(orig_history),
            filters=dict(year=2020,regime='general'),
            pending_answer_kind='quantity'
        )
        stored_history_snapshot=copy.deepcopy(context['history'])
        def chat(payload,schema):
            if payload.get('task')=='classify_dependency':
                return dict(dependency='independent')
            return dict()
        result=resolve('Explica la resistencia del material',context,chat)
        self.assertEqual(result['question'],'Explica la resistencia del material')
        self.assertNotIn('filters',result['context'])
        self.assertNotIn('pending_answer_kind',result['context'])
        self.assertNotIn('1000',str(result['context']))
        self.assertEqual(result['context']['history'],[dict(role='user',content='Explica la resistencia del material')])
        self.assertEqual(context['history'],stored_history_snapshot)
        self.assertIn('filters',context)

    def test_synthetic_preservation_of_prefs_and_current_clarification(self):
        context=dict(
            publication_policy='latest_supported',
            ambiguity_policy='strict',
            report_year=2024,
            clarification_answer='Sección A',
            clarification_field='population',
            filters=dict(old_filter=True),
            pending_answer_kind='quantity',
            history=[dict(role='assistant',content='Hecho previo 42')]
        )
        def chat(payload,schema):
            if payload.get('task')=='classify_dependency':
                return dict(dependency='independent')
            return dict()
        result=resolve('¿Cuántas unidades se almacenan?',context,chat)
        self.assertEqual(result['context']['publication_policy'],'latest_supported')
        self.assertEqual(result['context']['ambiguity_policy'],'strict')
        self.assertEqual(result['context']['report_year'],2024)
        self.assertEqual(result['context']['clarification_answer'],'Sección A')
        self.assertEqual(result['context']['clarification_field'],'population')
        self.assertNotIn('filters',result['context'])
        self.assertNotIn('pending_answer_kind',result['context'])
        self.assertNotIn('42',str(result['context']))
        self.assertEqual(result['context']['history'],[dict(role='user',content='¿Cuántas unidades se almacenan?')])

    def test_synthetic_dependent_selection_preserves_resolution_and_guards(self):
        context=dict(
            pending_answer_kind='quantity',
            history=[
                dict(role='user',content='Calcula el saldo del almacén'),
                dict(role='assistant',content='(1) entradas menos salidas; (2) entradas menos devoluciones.')
            ]
        )
        calls=[]
        def chat(payload,schema):
            calls.append(payload)
            if payload.get('task')=='classify_dependency':
                return dict(dependency='dependent')
            if payload.get('task')=='classify_action':
                return dict(kind='quantity',preserve_previous_action=True)
            return dict(reference_id='reference_2',status='resolved',question='Calcula entradas menos devoluciones',kind='quantity',clarification='')
        result=resolve('la segunda',context,chat)
        self.assertEqual(len(calls),3)
        self.assertEqual(calls[0].get('task'),'classify_dependency')
        self.assertIn('entradas menos devoluciones',result['question'])
        self.assertEqual(result['resolution']['kind'],'quantity')
        self.assertEqual(result['context']['selected_option'],'entradas menos devoluciones.')

    def test_assistant_cannot_inject_preferences_or_active_clarification(self):
        from rag_pensiones.graph_conversation import independent_context
        context=dict(role='assistant',content='Un texto',report_year=2099,clarification_answer='9999',clarification_field='period')
        result=independent_context(context,'Una pregunta nueva')
        self.assertEqual(result,dict(history=[dict(role='user',content='Una pregunta nueva')]))

    def test_synthetic_unresolved_reference_triggers_genuine_ambiguity(self):
        context=dict(
            history=[
                dict(role='assistant',content='Hay varios coeficientes reductores posibles.')
            ]
        )
        calls=[]
        def chat(payload,schema):
            calls.append(payload)
            if payload.get('task')=='classify_dependency':
                return dict(dependency='dependent')
            if payload.get('task')=='validate_resolution':
                return dict(valid=False,clarification='¿A qué coeficiente te refieres?')
            return dict(reference_id='',status='resolved',question='Aplica ese coeficiente',kind='unspecified',clarification='')
        result=resolve('aplica ese coeficiente',context,chat)
        self.assertEqual(result['resolution']['status'],'ambiguous')
        self.assertEqual(result['resolution']['clarification'],'¿A qué coeficiente te refieres?')
        self.assertEqual(len(calls),3)

if __name__=='__main__':unittest.main()
