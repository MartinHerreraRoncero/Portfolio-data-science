"""Source-only footnote links from the report's superscript font profile."""
import copy
import hashlib
from functools import lru_cache
from pathlib import Path

from pypdf import PdfReader
from experiments.reconstructed_units import normalized_positions
from rag_pensiones.grounding import collapse_spaces
from rag_pensiones.ingest import validate_project_path


def source_markers(pdf_path, pages):
    reader = PdfReader(pdf_path)
    markers = []
    for page in pages:
        pdf_page = reader.pages[page['page_number']-1]
        if pdf_page.extract_text() != page['text']:
            raise ValueError('PDF differs from canonical page')
        fragments = []
        def visitor(text, cm, tm, font, size):
            name = str(font.get('/BaseFont','')) if font else ''
            role = 'reference' if abs(size-6.96)<.15 else 'definition' if abs(size-6.48)<.15 else None
            marker = 'CenturyGothic' in name and role is not None and text.strip().isdigit()
            aligned = 'CenturyGothic' in name and (abs(size-11.04)<.2 or abs(size-9.96)<.15)
            if text.strip() and (marker or aligned):fragments.append((collapse_spaces(text),marker,role,size))
        pdf_page.extract_text(visitor_text=visitor)
        normalized, starts, ends = normalized_positions(page['text'])
        cursor = 0
        for text, is_marker, role, size in fragments:
            position = normalized.find(text,cursor)
            if position < 0:raise ValueError('Cannot align PDF marker profile')
            start,end = starts[position],ends[position+len(text)-1]
            cursor = position+len(text)
            if is_marker:markers.append(dict(doc_id=page['doc_id'],page_number=page['page_number'],
                start_char=start,end_char=end,number=text,role=role,font_size=size))
    return markers


def marker_key(marker):
    return tuple(marker[k] for k in ['doc_id','page_number','start_char','end_char','number','role','font_size'])


@lru_cache(maxsize=4)
def verified_marker_inventory(pdf_path, digest, canonical_pages):
    path=Path(pdf_path)
    if hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
        raise ValueError('PDF changed during marker validation')
    pages=[dict(doc_id=d,page_number=n,text=t) for d,n,t in canonical_pages]
    return frozenset(marker_key(m) for m in source_markers(path,pages))


def extract_note_links(pdf_path, units, pages):
    pdf_path=validate_project_path(Path(pdf_path))
    markers=source_markers(pdf_path,pages)
    links = []
    for definition in markers:
        if definition['role']!='definition':continue
        page = next(p for p in pages if (p['doc_id'],p['page_number'])==(definition['doc_id'],definition['page_number']))
        notes = [u for u in units['units'] if u.get('kind')=='note' and u.get('source_boundary')=='closed'
                 and (u['spans'][0]['doc_id'],u['spans'][0]['page_number'])==(definition['doc_id'],definition['page_number'])
                 and definition['end_char']<=u['spans'][0]['start_char']
                 and not page['text'][definition['end_char']:u['spans'][0]['start_char']].strip()]
        if len(notes)!=1:continue
        for reference in markers:
            if reference['role']!='reference' or reference['doc_id']!=definition['doc_id'] or reference['number']!=definition['number']:continue
            for body in units['units']:
                if body.get('kind')=='body' and any(s['doc_id']==reference['doc_id'] and s['page_number']==reference['page_number']
                    and s['start_char']<=reference['start_char']<reference['end_char']<=s['end_char'] for s in body['spans']):
                    links.append(dict(body_id=body['id'],note_id=notes[0]['id'],reference=reference,definition=definition))
    output=copy.deepcopy(units)
    output['note_links']=links
    output['note_links_profile']=dict(pdf_path=str(pdf_path),pdf_sha256=hashlib.sha256(pdf_path.read_bytes()).hexdigest(),
        profile='CenturyGothic references6.96pt and definitions6.48pt; linked by same document+number; canonical offsets verified')
    validate_note_links(output,{(p['doc_id'],p['page_number']):p for p in pages})
    return output


def validate_note_links(extracted, pages_by_key):
    profile=extracted['note_links_profile'];pdf_path=validate_project_path(Path(profile['pdf_path']))
    digest=hashlib.sha256(pdf_path.read_bytes()).hexdigest()
    if digest!=profile['pdf_sha256']:raise ValueError('Source PDF digest mismatch')
    pages=tuple(sorted((p['doc_id'],p['page_number'],p['text']) for p in pages_by_key.values()))
    inventory=verified_marker_inventory(str(pdf_path),digest,pages)
    units={u['id']:u for u in extracted['units']}
    mapping={};seen=set()
    for link in extracted['note_links']:
        body,note=units[link['body_id']],units[link['note_id']]
        ref,definition=link['reference'],link['definition']
        if ref['role']!='reference' or definition['role']!='definition' or any(marker_key(m) not in inventory for m in [ref,definition]):
            raise ValueError('Link is not a verified PDF superscript marker')
        if (body.get('kind')!='body' or note.get('kind')!='note' or note.get('source_boundary')!='closed'
                or ref['doc_id']!=definition['doc_id'] or ref['number']!=definition['number']):
            raise ValueError('Invalid body/footnote link')
        for marker in [ref,definition]:
            page=pages_by_key[(marker['doc_id'],marker['page_number'])]
            if (not 0<=marker['start_char']<marker['end_char']<=len(page['text'])
                    or page['text'][marker['start_char']:marker['end_char']].strip()!=marker['number']):
                raise ValueError('Footnote marker lost canonical provenance')
        if not any(s['doc_id']==ref['doc_id'] and s['page_number']==ref['page_number']
                and s['start_char']<=ref['start_char']<ref['end_char']<=s['end_char'] for s in body['spans']):
            raise ValueError('Reference marker is outside body')
        first=note['spans'][0];page=pages_by_key[(definition['doc_id'],definition['page_number'])]
        if ((first['doc_id'],first['page_number'])!=(definition['doc_id'],definition['page_number'])
                or definition['end_char']>first['start_char'] or page['text'][definition['end_char']:first['start_char']].strip()):
            raise ValueError('Definition marker is not adjacent to note')
        key=(ref['doc_id'],ref['page_number'],ref['start_char'],ref['end_char'])
        if key in seen:raise ValueError('Ambiguous duplicate footnote reference')
        seen.add(key);mapping.setdefault(body['id'],[]).append(note['id'])
    return mapping
