"""Source-certified note tails and bounded unions of complementary evidence packages."""
import copy
import json

from experiments.preserve_evidence import PreservedGrounder
from experiments.reconstructed_units import candidate_messages
from experiments.source_bound_trial import selection_schema
from rag_pensiones.grounding import MAX_CLAIM_REFS, OllamaGrounder


def restore_packages(payload, records):
    if not isinstance(payload, dict) or set(payload) != {'status', 'ids', 'reason'}:
        raise ValueError('Invalid package selection fields')
    ids, reason = payload['ids'], payload['reason']
    if not isinstance(ids, list) or not isinstance(reason, str):
        raise ValueError('Invalid package selection types')
    if payload['status'] == 'insufficient_evidence':
        if ids or not reason.strip():
            raise ValueError('Incoherent package abstention')
        return dict(status='insufficient_evidence', evidence=[], reason=reason)
    mapping = {r['id']: r for r in records}
    if (payload['status'] != 'selected' or reason != '' or not 1 <= len(ids) <= 3
            or any(not isinstance(i, str) or i not in mapping for i in ids)
            or len(ids) != len(set(ids))):
        raise ValueError('Invalid complementary package IDs')
    union = []
    for pid in ids:
        for item in mapping[pid]['evidence']:
            if item not in union:
                union.append(copy.deepcopy(item))
    if (not union or len(union) > MAX_CLAIM_REFS
            or any(not isinstance(e, dict) or set(e) != {'source_id', 'quote'} or not isinstance(e['source_id'], str)
                   or not e['source_id'].strip() or not isinstance(e['quote'], str)
                   or not 12 <= len(e['quote']) <= 700 for e in union)
            or sum(len(e['quote'].encode('utf-8')) for e in union) > 2400):
        raise ValueError('Complementary union exceeds original citation limits')
    return dict(status='selected', evidence=union, reason='')


class FixedPreservedGrounder(PreservedGrounder):
    def __init__(self, extracted, pages, **kwargs):
        super().__init__(extracted, pages, **kwargs)
        for u in extracted['units']:
            if 'source_boundary' in u and (u['kind'] != 'note' or u['source_boundary'] != 'closed'):
                raise ValueError('Closed boundary certificates are restricted to reviewed notes')
        self.closed_note_tails, self.max_packages = True, 3

    def _chat_step(self, stage, messages, response_schema, expected_digest):
        if stage not in ('selection', 'repair_selection'):
            return super()._chat_step(stage, messages, response_schema, expected_digest)
        oversized = [r for r in self.catalog['records'] if len(r['evidence']) > MAX_CLAIM_REFS]
        self.catalog['records'] = [r for r in self.catalog['records'] if r not in oversized]
        self.catalog['omitted'].extend(dict(unit_id=r['unit_id'], reason='per_claim_reference_limit') for r in oversized)
        records = self.catalog['records']
        if not records:
            raise ValueError('No evidence packages within original citation limits')
        data = json.loads(next(m['content'] for m in messages if m['role'] == 'user'))
        schema = selection_schema(records)
        ids_schema = schema['anyOf'][0]['properties']['ids']
        ids_schema.update(maxItems=3, uniqueItems=True)
        payload, trace = OllamaGrounder._chat_step(
            self, stage, candidate_messages(data, self.catalog, max_packages=3), schema, expected_digest)
        self.selections.append(dict(stage=stage, payload=copy.deepcopy(payload)))
        return restore_packages(payload, records), trace
