"""Role isolation and literal offsets; separate semantic tests use local model controls."""
import copy
import json
import unittest

from experiments.literal_alignment import LiteralChecker, DIMENSIONS, validate_extraction, validate_pair
from experiments.answer_alignment import apply_alignment
from experiments.compare_literal import run_extract, run_compare
from test_answer_alignment import Generator


def spans(**kwargs):
    p={d:[] for d in DIMENSIONS}
    for dim,items in kwargs.items():p[dim]=[dict(text_id=t,quote=q) for t,q in items]
    return p


def fixture():
    query='¿Qué gasto total se proyecta en 2038?'
    view=dict(status='answered',reason='',claims=[dict(text='Clases Pasivas representa el 0,7% del PIB en 2038.',
        evidence=[dict(source_id='S1',quote='La fuente indica 15,4% del PIB total en 2038. MARCADOR_SOLO_CITA')])])
    q=spans(entity=[('Q','gasto total')],period=[('Q','2038')],measure=[('Q','gasto total')],nature=[('Q','proyecta')])
    a=spans(entity=[('A1','Clases Pasivas')],period=[('A1','2038')],measure=[('A1','0,7% del PIB')])
    return query,view,q,a


class SequenceGenerator(Generator):
    def __init__(self,payloads,**kwargs):super().__init__(**kwargs);self.payloads=copy.deepcopy(payloads)
    def chat(self,messages,response_schema):
        self.payload=self.payloads.pop(0)
        return super().chat(messages,response_schema)


def comparisons(q,mismatch=None):
    return {d:dict(outcome=('mismatch' if d==mismatch else 'match') if q[d] else 'not_required',reason='Control de correspondencia.') for d in DIMENSIONS}


class LiteralTests(unittest.TestCase):
    def test_question_answer_and_comparator_cannot_see_citations_or_each_other_raw_text(self):
        query,view,q,a=fixture();gen=SequenceGenerator([q,a,comparisons(q,'entity')])
        checker=LiteralChecker(gen);audit=checker.check(query,view,expected_digest='a'*64)
        self.assertEqual(audit['decision'],'reject')
        users=[json.loads(c['messages'][1]['content']) for c in gen.calls]
        self.assertEqual(users[0]['texts'],{'Q':query})
        self.assertEqual(users[1]['texts'],{'A1':view['claims'][0]['text']})
        self.assertEqual(set(users[2]),{'pregunta','respuesta'})
        serialized=json.dumps([c['messages'] for c in gen.calls])
        self.assertNotIn('MARCADOR_SOLO_CITA',serialized);self.assertNotIn('15,4%',serialized)
        self.assertEqual(apply_alignment(view,audit)['status'],'insufficient_evidence')

    def test_non_required_dimensions_are_schema_locked_not_relabelled_after_model(self):
        query,view,q,a=fixture();bad=comparisons(q);bad['scope']['outcome']='uncertain'
        gen=SequenceGenerator([q,a,bad]);checker=LiteralChecker(gen)
        with self.assertRaisesRegex(ValueError,'requirement'):
            checker.check(query,view,expected_digest='a'*64)
        props=gen.calls[2]['schema']['properties']
        self.assertEqual(props['scope']['properties']['outcome'],{'const':'not_required'})
        self.assertNotIn('not_required',props['period']['properties']['outcome']['enum'])

    def test_exact_offsets_use_python_unicode_characters(self):
        text='📘 La pensión media es de 900 euros.'
        out=validate_extraction(spans(measure=[('A1','900 euros')]),{'A1':text})
        s=out['measure'][0];self.assertEqual(text[s['start']:s['end']],'900 euros')
        self.assertEqual(s['start'],text.index('900 euros'))

    def test_fabricated_quote_and_wrong_role_source_and_duplicate_are_rejected(self):
        for p in [spans(measure=[('Q','dato inventado')]),spans(measure=[('A1','dato')]),spans(measure=[('Q','dato'),('Q','dato')])]:
            with self.assertRaises(ValueError):validate_extraction(p,{'Q':'Un dato literal.'})

    def test_failed_question_extraction_still_retains_answer_extraction_and_raw(self):
        query,view,q,a=fixture();q['measure']=[dict(text_id='Q',quote='dato inventado')]
        gen=SequenceGenerator([q,a]);checker=LiteralChecker(gen)
        data=dict(schema_version='answer-alignment-controls-v1',notice='Synthetic',cases=[dict(id='x',query=query,answer=view)])
        r=run_extract(data,None,checker=checker);row=r['cases'][0]
        self.assertEqual(row['status'],'error');self.assertEqual(len(row['calls']),2)
        self.assertIn('raw_response',row['calls'][0]);self.assertIn('validated',row['diagnostics'][1])

    def test_tampered_offsets_or_source_text_are_rejected_before_comparison(self):
        query,view,q,a=fixture();checker=LiteralChecker(SequenceGenerator([q,a]))
        pair=checker.extract_pair(query,view,expected_digest='a'*64)
        altered=copy.deepcopy(pair);altered['answer']['measure'][0]['start']+=1
        with self.assertRaisesRegex(ValueError,'offsets'):validate_pair(altered)
        with self.assertRaisesRegex(ValueError,'sources differ'):validate_pair(pair,query+' Otro requisito.',view)

    def test_no_extraction_means_no_vacuous_acceptance(self):
        query,view,q,a=fixture();checker=LiteralChecker(SequenceGenerator([spans(),a]))
        with self.assertRaisesRegex(ValueError,'empty extraction'):
            checker.extract_pair(query,view,expected_digest='a'*64)

    def test_abstention_uses_no_extraction_and_keeps_reason(self):
        checker=LiteralChecker(SequenceGenerator([]))
        view=dict(status='insufficient_evidence',claims=[],reason='No hay información suficiente.')
        audit=checker.check('Pregunta',view,expected_digest='a'*64)
        self.assertEqual(audit['decision'],'abstained');self.assertEqual(checker.generator.count,0)
        self.assertEqual(apply_alignment(view,audit),view)

    def test_budget_and_digest_errors_are_not_semantic_rejections(self):
        query,view,q,a=fixture();checker=LiteralChecker(SequenceGenerator([q,a],change=True))
        with self.assertRaisesRegex(ValueError,'digest'):
            checker.extract_pair(query,view,expected_digest='a'*64)
        gen=SequenceGenerator([a]);checker=LiteralChecker(gen)
        with self.assertRaisesRegex(ValueError,'budget'):
            checker.extract_pair(query*1000,view,expected_digest='a'*64)
        self.assertEqual(gen.count,1) # The independent answer extraction is attempted despite question overflow.

    def test_saved_comparison_keeps_source_immutable_and_original_answer_available(self):
        query,view,q,a=fixture();data=dict(schema_version='answer-alignment-controls-v1',notice='Synthetic',cases=[dict(id='x',query=query,answer=view)])
        extraction=run_extract(data,None,checker=LiteralChecker(SequenceGenerator([q,a])))
        before=copy.deepcopy(extraction)
        comparison=run_compare(extraction,None,checker=LiteralChecker(SequenceGenerator([comparisons(q,'measure')])))
        self.assertEqual(comparison['cases'][0]['audit']['decision'],'reject')
        self.assertEqual(comparison['cases'][0]['input_view'],view);self.assertEqual(extraction,before)

    def test_comparison_failure_retains_returned_payload_and_trace(self):
        query,view,q,a=fixture();data=dict(schema_version='answer-alignment-controls-v1',notice='Synthetic',cases=[dict(id='x',query=query,answer=view)])
        extraction=run_extract(data,None,checker=LiteralChecker(SequenceGenerator([q,a])))
        comparison=run_compare(extraction,None,checker=LiteralChecker(SequenceGenerator([{'bad':'payload'}])))
        row=comparison['cases'][0];self.assertEqual(row['status'],'error')
        self.assertIn('trace',row['diagnostics'][0]);self.assertIn('raw_response',row['calls'][0])


if __name__=='__main__':unittest.main()
