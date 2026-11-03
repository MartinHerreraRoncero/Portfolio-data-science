"""Source superscript links preserve the relevant note without global note priority."""
import copy
import json
from pathlib import Path
import unittest

from experiments.note_links import extract_note_links
from experiments.planned_preserved import PlannedNotesGrounder
from rag_pensiones.chunking import ChunkRecord
from rag_pensiones.search import SearchHit
from test_answer_alignment import Generator


class LinkedNotesTests(unittest.TestCase):
    def setUp(self):
        self.units=json.loads(Path('evaluation/linked_note_units.json').read_text(encoding='utf-8'))
        self.pdf_path=Path('Airef/2025_Opinion_Sostenibilidad_AAPP/Opinion sobre la Sostenibilidad de las AAPP a largo plazo Marzo 2025.pdf').resolve()
        self.units['note_links_profile']['pdf_path']=str(self.pdf_path)
        self.pages=[json.loads(x) for x in Path('data/processed/airef_2025_pages.jsonl').read_text(encoding='utf-8').splitlines()]
        self.source=json.loads(Path('evaluation/planned_preserved_results.json').read_text(encoding='utf-8'))

    def prepare(self,node,units=None,prefer=True):
        gen=Generator();g=PlannedNotesGrounder(units if units is not None else self.units,self.pages,
            generator=gen,prefer_retrieved_notes=prefer)
        seeds=[SearchHit(h['rank'],h['score'],ChunkRecord(**h['chunk'])) for h in node['seed_hits']]
        g.prepare_evidence(node['query'],seeds)
        return g

    def test_links_reproduce_pdf_fonts_and_preserve_all_original_unit_text(self):
        original=json.loads(Path('evaluation/context_fix_units.json').read_text(encoding='utf-8'))
        rebuilt=extract_note_links(self.pdf_path,original,self.pages)
        self.assertEqual(rebuilt,self.units)
        self.assertEqual(rebuilt['units'],original['units'])
        self.assertEqual([(x['body_id'],x['note_id'],x['reference']['number']) for x in rebuilt['note_links']],
            [('U5','U6','32'),('U8','U9','33'),('U10','U12','34'),('U11','U13','35')])

    def test_ratio_note_follows_its_parent_and_noncontributive_paragraph_survives(self):
        for case in self.source['cases']:
            if case['id'] not in ['s1','v02']:continue
            for node in case['conditions']['planned']['answer']['nodes']:
                g=self.prepare(node)
                parents=[x['unit_id'] for x in g.pending['expansion']['selected_parents']]
                self.assertLessEqual(len(parents),4)
                if case['id']=='s1':
                    self.assertEqual(parents[parents.index('U5')+1],'U6')
                    self.assertIn('0,9',str(g.pending['catalog']))
                elif node['node_id']=='p1':
                    self.assertIn('U3',parents)
                    self.assertIn('media d',str(g.pending['catalog']))

    def test_tampered_marker_offsets_numbers_and_roles_fail_before_model(self):
        node=next(c for c in self.source['cases'] if c['id']=='s1')['conditions']['planned']['answer']['nodes'][0]
        for mutation in ['offset','number','role','duplicate']:
            units=copy.deepcopy(self.units);link=units['note_links'][0]
            if mutation=='offset':link['reference']['start_char']+=1
            elif mutation=='number':link['definition']['number']='9999'
            elif mutation=='role':link['body_id']=link['note_id']
            else:units['note_links'].append(copy.deepcopy(link))
            with self.assertRaises(ValueError):self.prepare(node,units)

    def test_links_do_not_change_default_or_expand_unretrieved_notes(self):
        note=next(u for u in self.units['units'] if u['id']=='U6')['spans'][0]
        for case in self.source['cases']:
            for node in case['conditions']['planned']['answer']['nodes']:
                original=self.prepare(node,prefer=False)
                self.assertEqual(original.pending['expansion'],node['expansion'])
                retrieved=any(h['chunk']['doc_id']==note['doc_id'] and h['chunk']['page_number']==note['page_number']
                    and max(h['chunk']['start_char'],note['start_char'])<min(h['chunk']['end_char'],note['end_char']) for h in node['seed_hits'])
                if not retrieved:
                    fixed=self.prepare(node)
                    self.assertNotIn('U6',[p['unit_id'] for p in fixed.pending['expansion']['selected_parents']])

    def test_year_2035_cannot_be_forged_as_note_35(self):
        units=copy.deepcopy(self.units)
        link=next(l for l in units['note_links'] if l['reference']['number']=='35')
        body=next(u for u in units['units'] if u['id']=='U2');span=body['spans'][0]
        page=next(p for p in self.pages if p['doc_id']==span['doc_id'] and p['page_number']==span['page_number'])
        start=page['text'].index('2035',span['start_char'],span['end_char'])+2
        link['body_id']='U2';link['reference'].update(page_number=span['page_number'],start_char=start,end_char=start+2)
        node=next(c for c in self.source['cases'] if c['id']=='s1')['conditions']['planned']['answer']['nodes'][0]
        with self.assertRaisesRegex(ValueError,'superscript'):self.prepare(node,units)


if __name__=='__main__':unittest.main()
