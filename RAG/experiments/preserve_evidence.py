"""Experimental citation-package binding, separate from semantic certification.

The canonical grounder and its verifier verdicts remain untouched. Publication
retains context quotes even when the verifier prunes them as irrelevant; that
does not relabel those quotes as semantic support. Only accepted claims publish.
"""
import copy
import json

from experiments.reconstructed_units import ReconstructedGrounder, make_hits
from rag_pensiones.answer import MAX_MESSAGES_BYTES
from rag_pensiones.grounding import (
    MAX_CLAIM_REFS, format_to_public_claims, collapse_spaces,
)


def validate_catalog(extracted, pages):
    """Check literal spans and document identity before any model is contacted."""
    mapping = {(p['doc_id'], p['page_number']): p for p in pages}
    if len(mapping) != len(pages):
        raise ValueError('Duplicate canonical page')
    units = extracted.get('units')
    if not isinstance(units, list) or not units:
        raise ValueError('Expected nonempty units')
    seen = set()
    for unit in units:
        uid = unit.get('id')
        if not isinstance(uid, str) or not uid or uid in seen:
            raise ValueError('Invalid or duplicate unit ID')
        seen.add(uid)
        if not isinstance(unit.get('spans'), list) or not unit['spans']:
            raise ValueError('Empty unit spans')
        for span in unit['spans']:
            page = mapping.get((span.get('doc_id'), span.get('page_number')))
            if page is None or any(span.get(k) != page[k] for k in ('source_file', 'published_at')):
                raise ValueError('Unit document identity mismatch')
            start, end = span.get('start_char'), span.get('end_char')
            if (type(start) is not int or type(end) is not int
                    or not 0 <= start < end <= len(page['text'])):
                raise ValueError('Invalid canonical offsets')
            if collapse_spaces(page['text'][start:end]) != span.get('quote'):
                raise ValueError('Unit quote lost canonical provenance')
    make_hits(units, mapping)


def bound_generation(messages, response_schema, stage):
    """Constrain every answered claim to the complete selected literal package."""
    modified = copy.deepcopy(messages)
    schema = copy.deepcopy(response_schema)
    data = json.loads(next(m['content'] for m in modified if m['role'] == 'user'))
    repair = stage == 'repair_generation'
    catalog = data['catalogo_evidencias_reparacion' if repair else 'catalogo_evidencias']
    refs = [e['evidence_id'] for e in catalog]
    if len(refs) > MAX_CLAIM_REFS or len(refs) != len(set(refs)):
        raise ValueError('Complete package cannot fit unchanged per-claim reference limit')
    if repair:
        branches = schema['properties']['repairs']['items']['anyOf']
        if not refs:
            branches[:] = branches[1:]
        else:
            branches[0]['properties']['evidence_ids'] = {'type': 'array', 'const': refs}
    elif not refs:
        schema['anyOf'] = schema['anyOf'][1:]
    else:
        schema['anyOf'][0]['properties']['claims']['items']['properties']['evidence_ids'] = {
            'type': 'array', 'const': refs}
    modified[0]['content'] += (
        '\nConserva el paquete completo de referencias en cada afirmación respondida. '
        'Expresa el sujeto y los periodos necesarios sin depender de «este sistema» o '
        '«esta estimación». Responde solo lo solicitado; no añadas causas no explícitas. '
        'El paquete puede contener contexto: su presencia no demuestra apoyo semántico. '
        'Si no basta para contestar, abstente.')
    return modified, schema, refs


def enforce_binding(payload, stage, refs):
    claims = payload.get('repairs', []) if stage == 'repair_generation' else payload.get('claims', [])
    for claim in claims:
        answered = claim.get('status') == 'answered' if stage == 'repair_generation' else payload.get('status') == 'answered'
        if answered and (not refs or claim.get('evidence_ids') != refs):
            raise ValueError('Model omitted or changed mandatory package references')


PUBLICATION_SCHEMA_VERSION = 'context-package-publication-v1'
PARTIAL_PUBLICATION_SCHEMA_VERSION = 'context-package-partial-publication-v1'
PARTIAL_PUBLICATION_REASON = (
    'Publicación parcial restringida a las afirmaciones aceptadas por el verificador.'
)


def publication(result, *, preserve=False, allow_partial: bool = False):
    """Publish validated accepted text; keep context and verifier-retained refs distinct.

    Callers first validate the canonical grounded result against the actual hits.
    No claim text, reason, semantic verdict or rejected claim is changed here.
    """
    if type(preserve) is not bool:
        raise ValueError('preserve must be a strict boolean')
    if type(allow_partial) is not bool:
        raise ValueError('allow_partial must be a strict boolean')
    output = {'schema_version': PUBLICATION_SCHEMA_VERSION,
              'status': result['status'], 'claims': copy.deepcopy(result['claims']),
              'reason': result['reason'], 'bindings': []}
    if not preserve:
        return output

    is_answered = result['status'] == 'answered'
    is_partial = (
        allow_partial
        and result['status'] == 'insufficient_evidence'
        and bool(result.get('grounding', {}).get('accepted_claims'))
    )

    if not is_answered and not is_partial:
        return output

    g = result['grounding']
    initial_verdicts = {v['claim_id']: v for v in g['initial_verdicts']}
    catalogs = g['selected_evidence'] + g['repair_evidence']
    complete_claims = []
    bindings = []
    for accepted in g['accepted_claims']:
        cid = accepted['claim_id']
        initial = initial_verdicts[cid]['verdict'] == 'supported'
        evidence = g['selected_evidence'] if initial else g['repair_evidence']
        originals = g['initial_claims'] if initial else g['repairs']
        original = next(c for c in originals if c['claim_id'] == cid)
        refs = [e['evidence_id'] for e in evidence]
        if (not refs or len(refs) > MAX_CLAIM_REFS or original['evidence_ids'] != refs
                or accepted['text'] != original['text']
                or not set(accepted['evidence_ids']) <= set(refs)):
            raise ValueError('Invalid complete-package publication binding')
        complete_claims.append(dict(claim_id=cid, text=accepted['text'], evidence_ids=refs))
        bindings.append({'claim_id': cid,
            'stage': 'generation' if initial else 'repair_generation',
            'package_evidence_ids': refs,
            'verifier_retained_evidence_ids': copy.deepcopy(accepted['evidence_ids']),
            'context_reinstated_ids': [r for r in refs if r not in accepted['evidence_ids']]})

    published_claims = format_to_public_claims(complete_claims, catalogs)

    if is_partial:
        return {
            'schema_version': PARTIAL_PUBLICATION_SCHEMA_VERSION,
            'status': 'partial',
            'claims': published_claims,
            'reason': PARTIAL_PUBLICATION_REASON,
            'bindings': bindings,
        }

    output['claims'] = published_claims
    output['bindings'] = bindings
    return output


def validate_publication(output, result, *, preserve=False, allow_partial: bool = False):
    if type(preserve) is not bool:
        raise ValueError('preserve must be a strict boolean')
    if type(allow_partial) is not bool:
        raise ValueError('allow_partial must be a strict boolean')
    expected = publication(result, preserve=preserve, allow_partial=allow_partial)
    if output != expected:
        raise ValueError('Publication lost package provenance or changed grounded content')


PARTIAL_GENERATION_INSTRUCTION = (
    '\nEmite únicamente las afirmaciones atómicas solicitadas respaldadas por las evidencias seleccionadas, '
    'incluso si otra parte solicitada es imposible de responder. '
    'Declara insuficiencia total únicamente si ninguna parte solicitada tiene respaldo. '
    'No exijas una precisión no pedida ni inventes importes o hechos no disponibles.'
)


def partial_generation_messages(messages: list[dict]) -> list[dict]:
    """Append compact universal instruction to system prompt for partial generation."""
    if not isinstance(messages, list) or not messages:
        raise ValueError('messages must be a nonempty list of dicts')
    for m in messages:
        if not isinstance(m, dict) or 'role' not in m or 'content' not in m:
            raise ValueError(f'Invalid message item: {m!r}')
    modified = copy.deepcopy(messages)
    sys_msg = next((m for m in modified if m.get('role') == 'system'), None)
    if sys_msg is None:
        raise ValueError('System message not found in messages')
    sys_msg['content'] += PARTIAL_GENERATION_INSTRUCTION
    serialized = json.dumps(modified, ensure_ascii=False).encode('utf-8')
    if len(serialized) > MAX_MESSAGES_BYTES:
        raise ValueError(
            f'Messages payload exceeds budget of {MAX_MESSAGES_BYTES} bytes UTF-8 '
            f'({len(serialized)} bytes).'
        )
    return modified


class PreservedGrounder(ReconstructedGrounder):
    """Bind generation/repair refs, then publish full context as a separate view."""
    def __init__(self, extracted, pages, partial_generation: bool = False, **kwargs):
        if type(partial_generation) is not bool:
            raise ValueError('partial_generation must be a strict boolean')
        validate_catalog(extracted, pages)
        super().__init__(extracted, pages, **kwargs)
        self.bindings = []
        self.partial_generation = partial_generation

    def answer(self, *args, **kwargs):
        self.bindings = []
        return super().answer(*args, **kwargs)

    def _chat_step(self, stage, messages, response_schema, expected_digest):
        if stage in ('selection', 'repair_selection'):
            oversized = [r for r in self.catalog['records'] if len(r['evidence']) > MAX_CLAIM_REFS]
            self.catalog['records'] = [r for r in self.catalog['records'] if r not in oversized]
            self.catalog['omitted'].extend({'unit_id': r['unit_id'], 'reason': 'per_claim_reference_limit'} for r in oversized)
        if stage not in ('generation', 'repair_generation'):
            return super()._chat_step(stage, messages, response_schema, expected_digest)
        modified, schema, refs = bound_generation(messages, response_schema, stage)
        if stage == 'generation' and self.partial_generation:
            modified = partial_generation_messages(modified)
        payload, trace = super()._chat_step(stage, modified, schema, expected_digest)
        self.bindings.append({'stage': stage, 'required_refs': refs, 'payload': copy.deepcopy(payload)})
        enforce_binding(payload, stage, refs)
        return payload, trace
