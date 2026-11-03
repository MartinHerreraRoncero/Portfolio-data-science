"""Whitespace localization restores original characters; it does not judge truth."""
import copy
import json
from pathlib import Path
import unittest

from rag_pensiones.graph_validation import align_literal_quote,validate_text_claims,InvalidPlan


class QuoteAlignmentTests(unittest.TestCase):
    def test_spaces_newlines_and_nbsp_restore_original_span_and_offsets(self):
        passage='Antes. El gasto\n  público\u00a0creció en 2024. Después.'
        quote='El gasto público creció en 2024.'
        aligned=align_literal_quote(quote,passage)
        self.assertEqual(aligned['quote'],'El gasto\n  público\u00a0creció en 2024.')
        self.assertEqual(passage[aligned['start_char']:aligned['end_char']],aligned['quote'])
        self.assertEqual(align_literal_quote(quote,passage+' '+passage),aligned)

    def test_digits_punctuation_paraphrase_ellipsis_and_word_merging_are_rejected(self):
        for quote,passage in [('12','1 2'),('1 2','12'),('La tasa fue 12.3 en 2024.','La tasa fue 12,3 en 2024.'),
            ('El dato es de 2025.','El dato es de 2024.'),('La pensión media aumenta.','La prestación habitual aumenta.'),
            ('El gasto ... aumentó.','El gasto público aumentó.')]:
            with self.subTest(quote=quote),self.assertRaises(InvalidPlan):align_literal_quote(quote,passage)

    def test_source_and_report_guards_stay_active_after_whitespace_alignment(self):
        source=dict(source_id='one',text='El gasto\n público creció en 2024.',book='2026',doc_id='sha',
            page_number=5,source_file='source.pdf',source_path='source.pdf',source_sha256='sha')
        raw=[dict(need_id='n1',text='Un texto que aún necesita revisión semántica.',
                  evidence=[dict(source_id='one',quote='El gasto público creció en 2024.')])]
        result=validate_text_claims(copy.deepcopy(raw),[dict(need_id='n1',kind='text',book='2026')],[source])
        self.assertEqual(result[0]['evidence'][0]['quote'],source['text'])
        self.assertNotIn('supported',result[0])
        with self.assertRaisesRegex(InvalidPlan,'different report'):
            validate_text_claims(copy.deepcopy(raw),[dict(need_id='n1',kind='text',book='2025')],[source])
        bad=copy.deepcopy(raw);bad[0]['evidence'][0]['source_id']='another'
        with self.assertRaises(InvalidPlan):validate_text_claims(bad,[dict(need_id='n1',kind='text')],[source])

    def test_all_eight_recorded_model_quotes_restore_exact_original_source_spans(self):
        root=Path(__file__).resolve().parents[1]
        qa=json.loads((root/'evaluation/langgraph_rag_2026-11-01/design_audit/quote_whitespace_alignment_diagnosis_v1.json').read_text(encoding='utf-8'))
        results=json.loads((root/qa['results_path']).read_text(encoding='utf-8'))
        cases={c['id']:c for c in results['cases']}
        for row in qa['quotes']:
            source=next(s for s in cases[row['case_id']]['answer']['retrieval']['documents']if s['source_id']==row['source_id'])
            with self.subTest(case=row['case_id'],source=row['source_id']):
                actual=align_literal_quote(row['raw_generated_quote'],source['text'])
                self.assertEqual(actual['quote'],row['restored_original_quote'])
                self.assertEqual(actual['start_char'],row['chunk_span_start'])
                self.assertEqual(actual['end_char'],row['chunk_span_end_exclusive'])


if __name__=='__main__':unittest.main()
