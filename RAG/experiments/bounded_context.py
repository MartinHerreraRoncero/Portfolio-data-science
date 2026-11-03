"""Constrain the decoder to package combinations that satisfy the original citation limits."""
import copy
from itertools import combinations
import json

from experiments.fixed_context import FixedPreservedGrounder, restore_packages
from experiments.reconstructed_units import candidate_messages
from experiments.source_bound_trial import selection_schema
from rag_pensiones.grounding import MAX_CLAIM_REFS, OllamaGrounder


def feasible_package_ids(records):
    choices, seen = [], set()
    for size in range(1, min(3, len(records)) + 1):
        for group in combinations(records, size):
            ids = [r['id'] for r in group]
            try:
                restored = restore_packages(dict(status='selected', ids=ids, reason=''), records)
            except ValueError:
                continue
            fingerprint = frozenset((e['source_id'], e['quote']) for e in restored['evidence'])
            if fingerprint not in seen:
                seen.add(fingerprint)
                choices.append(ids)
    return choices


class BoundedPreservedGrounder(FixedPreservedGrounder):
    def _chat_step(self, stage, messages, response_schema, expected_digest):
        if stage not in ('selection', 'repair_selection'):
            return super()._chat_step(stage, messages, response_schema, expected_digest)
        oversized = [r for r in self.catalog['records'] if len(r['evidence']) > MAX_CLAIM_REFS]
        self.catalog['records'] = [r for r in self.catalog['records'] if r not in oversized]
        self.catalog['omitted'].extend(dict(unit_id=r['unit_id'], reason='per_claim_reference_limit') for r in oversized)
        records = self.catalog['records']
        choices = feasible_package_ids(records)
        if not choices:
            raise ValueError('No feasible evidence combination within original limits')
        data = json.loads(next(m['content'] for m in messages if m['role'] == 'user'))
        schema = selection_schema(records)
        schema['anyOf'][0]['properties']['ids'].update(maxItems=3, uniqueItems=True, enum=choices)
        payload, trace = OllamaGrounder._chat_step(
            self, stage, candidate_messages(data, self.catalog, max_packages=3), schema, expected_digest)
        self.selections.append(dict(stage=stage, payload=copy.deepcopy(payload), permitted_combinations=len(choices)))
        if payload.get('status') == 'selected' and payload.get('ids') not in choices:
            raise ValueError('Selection not in the feasible schema combinations')
        return restore_packages(payload, records), trace
