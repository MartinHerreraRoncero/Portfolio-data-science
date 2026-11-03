"""Full grounded flow: mandatory context survives pruning without forging verdicts."""
import copy
import json
import unittest

from experiments.preserve_evidence import (
    PreservedGrounder, bound_generation, enforce_binding, publication,
    validate_publication, validate_catalog,
)
from rag_pensiones.grounding import (
    GENERATION_JSON_SCHEMA, REPAIR_GENERATION_JSON_SCHEMA,
    build_generation_messages, build_repair_generation_messages, validate_grounded_result,
)
from rag_pensiones.chunking import ChunkRecord
from rag_pensiones.search import SearchHit


def fixture():
    text = 'El régimen A es un sistema público. Aunque el sistema está cerrado desde 2017, mantiene gasto.'
    p = dict(doc_id='d', source_file='public.pdf', published_at='2025-01-01', page_number=1, text=text)
    span = {k: v for k, v in p.items() if k != 'text'} | dict(start_char=0, end_char=len(text), quote=text)
    units = {'units': [dict(id='U1', spans=[span])]}
    hit = SearchHit(1, .9, ChunkRecord('c', 'd', 'public.pdf', '2025-01-01', 1, 0, len(text), text, 'synthetic'))
    return units, [p], [hit]


class Generator:
    model = 'qwen3.5:9b'
    def __init__(self, payloads): self.payloads = copy.deepcopy(payloads); self.calls = []
    def get_digest(self): return 'a'*64
    def chat(self, messages, response_schema):
        self.calls.append(dict(messages=copy.deepcopy(messages), schema=copy.deepcopy(response_schema)))
        payload = self.payloads.pop(0)
        return dict(model=self.model, done=True, done_reason='stop',
                    message=dict(role='assistant', content=json.dumps(payload)),
                    prompt_eval_count=150, eval_count=80, eval_duration=1000, total_duration=2000)


def selected(): return {'status': 'selected', 'ids': ['P2'], 'reason': ''}
def generated(text='El régimen A está cerrado a nuevos entrantes desde 2017.', refs=None):
    return dict(status='answered', claims=[dict(claim_id='C1', text=text, evidence_ids=refs or ['E1','E2'])], reason='')
def verdict(label='supported', refs=('E1','E2')):
    return {'verdicts': {'C1': dict(verdict=label, reason='Control sintético.', evidence_verdicts={
        refs[0]: {'relation':'irrelevant'}, refs[1]: {'relation':'supports' if label=='supported' else 'contradicts'}})}}


class PreservationTests(unittest.TestCase):
    def run_flow(self, payloads):
        units, pages, hits = fixture()
        gen = Generator(payloads)
        g = PreservedGrounder(units, pages, generator=gen)
        result = g.answer('¿Desde cuándo está cerrado el régimen A?', hits, expected_digest='a'*64)
        validate_grounded_result(result, g.expanded_hits, expected_digest='a'*64, expected_model=gen.model)
        return g, gen, result

    def test_context_survives_verifier_pruning_but_verdict_and_raw_result_are_unchanged(self):
        g, gen, result = self.run_flow([selected(), generated(), verdict()])
        before = copy.deepcopy(result)
        self.assertEqual(len(result['claims'][0]['evidence']), 1)
        out = publication(result, preserve=True)
        self.assertEqual(len(out['claims'][0]['evidence']), 2)
        self.assertEqual(out['claims'][0]['evidence'][0]['quote'], 'El régimen A es un sistema público.')
        self.assertEqual(out['bindings'][0]['context_reinstated_ids'], ['E1'])
        self.assertEqual(result['grounding']['initial_verdicts'][0]['evidence_verdicts'][0]['relation'], 'irrelevant')
        self.assertEqual(result, before)
        refs = gen.calls[1]['schema']['anyOf'][0]['properties']['claims']['items']['properties']['evidence_ids']
        self.assertEqual(refs['const'], ['E1','E2'])
        validate_publication(out, result, preserve=True)

    def test_model_that_omits_opener_is_rejected_even_if_transport_ignores_schema(self):
        with self.assertRaisesRegex(ValueError, 'mandatory package'):
            self.run_flow([selected(), generated(refs=['E2'])])

    def test_repair_uses_new_package_ids_and_does_not_restore_original_package(self):
        repair = {'repairs': [dict(claim_id='C1', status='answered', text='El régimen A se cerró en 2017.', evidence_ids=['E3','E4'], reason='')]}
        g, gen, result = self.run_flow([selected(), generated('El régimen A se cerró en 2018.'), verdict('contradicted'), selected(), repair, verdict(refs=('E3','E4'))])
        out = publication(result, preserve=True)
        self.assertEqual(result['grounding']['repair_attempts'], 1)
        self.assertEqual(out['bindings'][0]['package_evidence_ids'], ['E3','E4'])
        self.assertEqual(out['bindings'][0]['verifier_retained_evidence_ids'], ['E4'])
        self.assertEqual(out['claims'][0]['text'], repair['repairs'][0]['text'])
        self.assertEqual(len(out['claims'][0]['evidence']), 2)

    def test_absence_and_its_reason_are_not_rewritten_or_given_claims(self):
        g, gen, result = self.run_flow([dict(status='insufficient_evidence', ids=[], reason='No hay ámbito regional.')])
        out = publication(result, preserve=True)
        self.assertEqual((out['status'], out['claims'], out['reason']), ('insufficient_evidence', [], 'No hay ámbito regional.'))
        self.assertEqual(len(gen.calls), 1)

    def test_binding_is_not_a_semantic_validator(self):
        wrong = 'El régimen A sigue abierto a nuevos entrantes.'
        g, gen, result = self.run_flow([selected(), generated(wrong), verdict()])
        # An erroneous verifier can still accept the claim. Do not silently claim otherwise.
        self.assertEqual(publication(result, preserve=True)['claims'][0]['text'], wrong)

    def test_publication_tampering_is_rejected_and_baseline_keeps_original_citations(self):
        g, gen, result = self.run_flow([selected(), generated(), verdict()])
        self.assertEqual(publication(result)['claims'], result['claims'])
        out = publication(result, preserve=True)
        out['claims'][0]['evidence'].pop(0)
        with self.assertRaisesRegex(ValueError, 'provenance'):
            validate_publication(out, result, preserve=True)

    def test_cross_page_context_remains_two_literal_citations(self):
        units, pages, hits = fixture()
        a = dict(pages[0], text='La ratio divide la población mayor de 67 años')
        b = dict(a, page_number=2, text='sobre la población de 18 a 67 años y alcanza el 47%.')
        spans = [{k:v for k,v in p.items() if k!='text'} | dict(start_char=0,end_char=len(p['text']),quote=p['text']) for p in (a,b)]
        unit = {'units':[dict(id='U1',spans=spans)]}
        hit = SearchHit(1,.9,ChunkRecord('c','d','public.pdf','2025-01-01',1,0,len(a['text']),a['text'],'synthetic'))
        gen = Generator([dict(status='selected',ids=['P1'],reason=''),generated('La ratio usa edades de 18 a 67 años.'),verdict()])
        g = PreservedGrounder(unit,[a,b],generator=gen)
        result = g.answer('¿Qué edades usa el denominador?', [hit], expected_digest='a'*64)
        validate_grounded_result(result,g.expanded_hits,expected_digest='a'*64,expected_model=gen.model)
        evidence = publication(result,preserve=True)['claims'][0]['evidence']
        self.assertEqual([e['quote'] for e in evidence],[a['text'],b['text']])
        self.assertNotEqual(evidence[0]['source_id'],evidence[1]['source_id'])

    def test_identity_offsets_and_literal_tampering_are_rejected_before_model(self):
        for field, value in [('source_file','other.pdf'),('published_at','2026-01-01'),('start_char',True),('end_char',99999),('quote','Texto inventado.')]:
            with self.subTest(field=field):
                units,pages,hits = fixture(); units['units'][0]['spans'][0][field]=value
                with self.assertRaises(ValueError): validate_catalog(units,pages)

    def test_empty_repair_cannot_be_answered_and_ref_overflow_is_not_truncated(self):
        messages = build_repair_generation_messages('Pregunta',[],[dict(claim_id='C1',text='Dato erróneo',reason='Falta evidencia')])
        _,schema,refs = bound_generation(messages,REPAIR_GENERATION_JSON_SCHEMA,'repair_generation')
        self.assertEqual(len(schema['properties']['repairs']['items']['anyOf']),1)
        with self.assertRaisesRegex(ValueError,'mandatory package'):
            enforce_binding({'repairs':[dict(status='answered',evidence_ids=[])]},'repair_generation',refs)
        ev = [dict(evidence_id=f'E{i}',source_id='S1',quote='Texto sintético del documento.') for i in range(4)]
        with self.assertRaisesRegex(ValueError,'reference limit'):
            bound_generation(build_generation_messages('Pregunta',ev),GENERATION_JSON_SCHEMA,'generation')


if __name__ == '__main__': unittest.main()
