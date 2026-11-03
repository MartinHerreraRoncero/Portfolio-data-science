"""Synthetic controls for literal selection independently of source entailment."""
import copy
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from rag_pensiones.graph_evidence import evidence_options,attach_evidence
from rag_pensiones.graph_validation import align_literal_quote


class LiteralEvidenceTests(unittest.TestCase):
    def test_body_number_equal_to_page_number_is_not_a_header(self):
        text='Se evaluaron 4\nmateriales. La muestra fue limitada.'
        options=evidence_options([dict(source_id='s',text=text,start_char=0,page_number=4)],None)
        self.assertIn('Se evaluaron 4\nmateriales.',[o['quote']for o in options])

    def test_header_and_cut_opening_clause_are_not_offered_as_complete_evidence(self):
        text='Informe del ensayo\n3 de abril de 2035 4\nsi se supera la temperatura, se revisará la pieza. El ensayo no demuestra una mejora de durabilidad.'
        docs=[dict(source_id='s',text=text,start_char=0,page_number=4)]
        options=evidence_options(docs,None,query='durabilidad mejora')
        self.assertEqual([o['quote']for o in options],['El ensayo no demuestra una mejora de durabilidad.'])
        self.assertTrue(all(o['quote']in text for o in options))
        self.assertEqual(docs[0]['text'],text)
        docs[0]=dict(source_id='s',text='continúa una frase cortada. El ensayo no demuestra una mejora de durabilidad.',start_char=1600)
        self.assertEqual([o['quote']for o in evidence_options(docs,None)],['El ensayo no demuestra una mejora de durabilidad.'])

    def test_native_semantic_focus_is_separate_from_auxiliary_empty_focus_and_cached(self):
        from rag_pensiones import graph_pipeline as p
        from test_graph_current import CurrentQuestionTests
        docs=[dict(source_id='s'+str(i),book='2035',text='El ensayo evalúa la resistencia del material.',
            doc_id='d',page_number=i+1,source_file='Manual.pdf',source_path='Manual.pdf',source_sha256='hash')for i in range(4)]
        service=SimpleNamespace(client=SimpleNamespace(calls=[]))
        state=dict(query='¿Qué concluye el ensayo?',interpretation=dict(needs=[dict(need_id='n',kind='text',metric='resistencia',book=None)]),
            documents=docs,text_source_focus=[],effective_context={})
        stages=[]
        def respond(services,stage,payload,schema):
            stages.append(stage)
            if stage=='document_focus':return dict(source_ids=['s2'])
            self.assertEqual({o['source_id']for o in payload['evidence_options']},{'s2'})
            return dict(claims=[dict(need_id='n',text='El ensayo estudia la resistencia.',evidence=[dict(evidence_id=payload['evidence_options'][0]['evidence_id'])])])
        nodes=CurrentQuestionTests().nodes(service)
        with patch.object(p,'_chat',side_effect=respond):
            state.update(nodes['generate'](state));state.update(nodes['generate'](state))
        self.assertEqual(stages,['document_focus','generate','generate'])
        self.assertEqual(state['text_source_focus'],[])
        self.assertEqual(state['text_claims'][0]['evidence'][0]['source_id'],'s2')

    def test_canonical_footnotes_and_ocr_survive_selection(self):
        text='La eficiencia de la batería 17 depende d e\nla temperatura ambiente.'
        docs=[dict(source_id='battery',text=text)]
        options=evidence_options(docs,None,query='eficiencia temperatura')
        claims=[dict(text='El documento relaciona eficiencia y temperatura.',evidence=[dict(evidence_id=options[0]['evidence_id'])])]
        before=copy.deepcopy(claims)
        attached=attach_evidence(claims,options)
        self.assertEqual(attached[0]['evidence'][0]['quote'],text)
        self.assertEqual(align_literal_quote(text,text),dict(quote=text,start_char=0,end_char=len(text)))
        self.assertEqual(claims,before)

    def test_long_first_source_cannot_fill_all_choices_with_repetitions(self):
        docs=[dict(source_id='a',text='El agua afecta a la estabilidad del material. '*50),
              dict(source_id='b',text='La temperatura afecta a la estabilidad del material.')]
        options=evidence_options(docs,None,byte_budget=200,query='estabilidad material')
        self.assertEqual({o['source_id']for o in options},{'a','b'})
        self.assertLessEqual(sum(len(o['quote'].encode('utf-8'))for o in options),200)
        self.assertEqual(len(options),2)

    def test_relevant_complete_sentence_is_selected_ahead_of_headers(self):
        docs=[dict(source_id='a',text='Manual técnico. El almacenamiento depende de la temperatura. Final corta')]
        options=evidence_options(docs,None,query='almacenamiento temperatura')
        self.assertEqual(options[0]['quote'],'El almacenamiento depende de la temperatura.')
        self.assertFalse(any('Final corta'in o['quote']for o in options))

    def test_empty_and_explicit_focus_are_not_expanded(self):
        docs=[dict(source_id='a',text='El agua afecta a la estabilidad del material.')]
        self.assertEqual(evidence_options([],None),[])
        self.assertEqual(evidence_options(docs,[]),[])
        self.assertEqual(evidence_options(docs,['missing']),[])
        self.assertEqual(evidence_options(docs,None,byte_budget=0),[])

    def test_unknown_identifier_fails_closed_without_mutation(self):
        options=evidence_options([dict(source_id='a',text='La temperatura afecta al material.')],None)
        claims=[dict(need_id='bad',text='Texto',evidence=[dict(evidence_id='not-offered')]),
                dict(need_id='good',text='La temperatura afecta al material.',evidence=[dict(evidence_id=options[0]['evidence_id'])])]
        before=copy.deepcopy(claims)
        rejected=[];accepted=attach_evidence(claims,options,rejections=rejected)
        self.assertEqual([c['need_id']for c in accepted],['good'])
        self.assertEqual(rejected[0]['code'],'unknown_evidence_id')
        self.assertEqual(claims,before)


if __name__=='__main__':unittest.main()
