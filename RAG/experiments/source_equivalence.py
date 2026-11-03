"""Generic, domain-agnostic equivalence engine for evaluation sources."""
from decimal import Decimal
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Set, Union

REQUIRED_CONTRACT_FIELDS = (
    'measure',
    'scope',
    'publication',
    'scenario',
    'unit',
    'period',
    'quantity_type',
)

REQUIRED_MEMBER_FIELDS = (
    'card_id',
    'observation_id',
    'source_path',
    'source_sha256',
    'source_sheet',
    'source_cell',
    'period',
    'unit',
    'book',
    'value',
)


def validate_registry(catalog: Dict[str, Any], registry: Dict[str, Any]) -> Dict[str, Any]:
    """Validate equivalence registry integrity, schema, freshness, and catalog identity."""
    if not isinstance(catalog, dict):
        raise TypeError("catalog must be a dict")
    if not isinstance(registry, dict):
        raise TypeError("registry must be a dict")
    if registry.get('schema_version')!='reviewed-source-equivalence-v1' or registry.get('review_status')!='manual_source_review':
        raise ValueError('Equivalence registry is not a reviewed source annotation')

    # Stale registry validation against catalog_hash
    catalog_hash = catalog.get('catalog_hash')
    if not catalog_hash or not isinstance(catalog_hash, str):
        raise ValueError("catalog must contain a non-empty 'catalog_hash'")
    registry_catalog_hash = registry.get('catalog_hash')
    if not registry_catalog_hash or not isinstance(registry_catalog_hash, str) or not registry_catalog_hash.strip():
        raise ValueError("Registry must contain a non-empty 'catalog_hash'")
    if registry_catalog_hash != catalog_hash:
        raise ValueError(
            f"Stale equivalence registry: registry catalog_hash '{registry_catalog_hash}' "
            f"does not match catalog catalog_hash '{catalog_hash}'"
        )

    groups = registry.get('groups')
    if not isinstance(groups, list):
        raise ValueError("Registry must contain a 'groups' list")

    cards_by_id = {c['card_id']: c for c in catalog.get('cards', [])}
    observations_by_id = catalog.get('observations', {})

    seen_observations: Dict[str, int] = {}
    seen_card_periods: Dict[tuple, int] = {}

    for g_idx, group in enumerate(groups):
        if not isinstance(group, dict):
            raise ValueError(f"Group at index {g_idx} must be a dict")

        reason = group.get('reason')
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError(f"Group at index {g_idx} must contain a non-empty 'reason' string")

        contract = group.get('contract')
        if not isinstance(contract, dict) or not contract:
            raise ValueError(f"Group at index {g_idx} must contain a non-empty 'contract' dict")

        # Validate contract fields
        for field in REQUIRED_CONTRACT_FIELDS:
            val = contract.get(field)
            if val is None or (isinstance(val, str) and not val.strip()):
                raise ValueError(
                    f"Group at index {g_idx} contract field '{field}' must be non-empty"
                )

        members = group.get('members')
        if not isinstance(members, list) or len(members) == 0:
            raise ValueError(f"Group at index {g_idx} must contain a non-empty 'members' list")

        contract_unit = str(contract['unit'])
        contract_period = str(contract['period'])
        contract_publication = str(contract['publication'])

        group_obs_ids: Set[str] = set()
        group_card_ids: Set[str] = set()

        for m_idx, member in enumerate(members):
            if not isinstance(member, dict):
                raise ValueError(f"Member {m_idx} in group {g_idx} must be a dict")

            for field in REQUIRED_MEMBER_FIELDS:
                if field not in member or member[field] is None:
                    raise ValueError(
                        f"Member {m_idx} in group {g_idx} is missing required field '{field}'"
                    )
                if isinstance(member[field], str) and not member[field].strip():
                    raise ValueError(
                        f"Member {m_idx} in group {g_idx} field '{field}' cannot be empty"
                    )

            card_id = str(member['card_id'])
            obs_id = str(member['observation_id'])

            # Reject duplicate IDs inside the same group
            if obs_id in group_obs_ids:
                raise ValueError(f"Duplicate observation_id '{obs_id}' in group {g_idx}")
            group_obs_ids.add(obs_id)

            if card_id in group_card_ids:
                raise ValueError(f"Duplicate card_id '{card_id}' in group {g_idx}")
            group_card_ids.add(card_id)

            # Reject conflicting IDs across groups
            if obs_id in seen_observations:
                raise ValueError(
                    f"Conflicting observation_id '{obs_id}' found in multiple groups "
                    f"({seen_observations[obs_id]} and {g_idx})"
                )
            seen_observations[obs_id] = g_idx

            card_period_key = (card_id, str(member['period']))
            if card_period_key in seen_card_periods:
                raise ValueError(
                    f"Conflicting card_id '{card_id}' for period '{member['period']}' "
                    f"found in multiple groups ({seen_card_periods[card_period_key]} and {g_idx})"
                )
            seen_card_periods[card_period_key] = g_idx

            # Verify observation against catalog
            if obs_id not in observations_by_id:
                raise ValueError(
                    f"Observation '{obs_id}' from member {m_idx} (group {g_idx}) not found in catalog"
                )
            obs = observations_by_id[obs_id]

            # Verify card against catalog
            if card_id not in cards_by_id:
                raise ValueError(
                    f"Card '{card_id}' from member {m_idx} (group {g_idx}) not found in catalog"
                )
            card = cards_by_id[card_id]

            # Verify observation belongs to card
            if obs_id not in card.get('observation_ids', []):
                raise ValueError(
                    f"Observation '{obs_id}' is not in card '{card_id}' observation_ids"
                )

            # Exact provenance checks against catalog observation
            if str(member['source_path']) != str(obs.get('source_path')):
                raise ValueError(
                    f"Provenance mismatch source_path for obs '{obs_id}': "
                    f"'{member['source_path']}' != '{obs.get('source_path')}'"
                )
            if str(member['source_sha256']) != str(obs.get('source_sha256')):
                raise ValueError(
                    f"Provenance mismatch source_sha256 for obs '{obs_id}': "
                    f"'{member['source_sha256']}' != '{obs.get('source_sha256')}'"
                )
            if str(member['source_sheet']) != str(obs.get('source_sheet')):
                raise ValueError(
                    f"Provenance mismatch source_sheet for obs '{obs_id}': "
                    f"'{member['source_sheet']}' != '{obs.get('source_sheet')}'"
                )
            if str(member['source_cell']) != str(obs.get('source_cell')):
                raise ValueError(
                    f"Provenance mismatch source_cell for obs '{obs_id}': "
                    f"'{member['source_cell']}' != '{obs.get('source_cell')}'"
                )
            if str(member['period']) != str(obs.get('period')):
                raise ValueError(
                    f"Period mismatch for obs '{obs_id}': "
                    f"'{member['period']}' != '{obs.get('period')}'"
                )

            # Exact numerical value check
            try:
                member_val = Decimal(str(member['value']))
                obs_val = Decimal(str(obs.get('value')))
                if not member_val.is_finite() or not obs_val.is_finite():
                    raise ValueError('Nonfinite source observation')
                if member_val != obs_val:
                    raise ValueError(
                        f"Value mismatch for obs '{obs_id}': {member_val} != {obs_val}"
                    )
            except Exception as e:
                raise ValueError(f"Value validation error for obs '{obs_id}': {e}")

            # Verify card unit and book match member
            if str(member['unit']) != str(card.get('unit')):
                raise ValueError(
                    f"Unit mismatch for card '{card_id}': '{member['unit']}' != '{card.get('unit')}'"
                )
            if str(member['book']) != str(card.get('book')):
                raise ValueError(
                    f"Book mismatch for card '{card_id}': '{member['book']}' != '{card.get('book')}'"
                )

            # Coincidence of unit, book, period with group contract
            if str(member['unit']) != contract_unit:
                raise ValueError(
                    f"Member unit '{member['unit']}' does not match group contract unit '{contract_unit}' "
                    f"in group {g_idx}"
                )
            if str(member['period']) != contract_period:
                raise ValueError(
                    f"Member period '{member['period']}' does not match group contract period '{contract_period}' "
                    f"in group {g_idx}"
                )
            if str(member['book']) != contract_publication:
                raise ValueError(
                    f"Member book '{member['book']}' does not match group contract publication '{contract_publication}' "
                    f"in group {g_idx}"
                )

    return registry


def accepted_sources(
    catalog: Dict[str, Any],
    seeds: Iterable[str],
    period: Any,
    registry: Optional[Union[Dict[str, Any], str, Path]] = None,
) -> Set[str]:
    """Expand seed card_ids strictly according to verified equivalence groups."""
    if not isinstance(catalog, dict):
        raise TypeError("catalog must be a dict")

    cards_by_id = {c['card_id']: c for c in catalog.get('cards', [])}
    seed_set = {str(s) for s in seeds}
    for s in seed_set:
        if s not in cards_by_id:
            raise KeyError(f"Seed card '{s}' not found in catalog")

    if not seed_set:
        return set()

    if registry is None:
        return set(seed_set)

    if isinstance(registry, (str, Path)):
        reg_path = Path(registry)
        if not reg_path.is_file():
            raise FileNotFoundError(f"Equivalence registry file not found: {reg_path}")
        parsed_registry = json.loads(reg_path.read_text(encoding='utf-8'))
    elif isinstance(registry, dict):
        parsed_registry = registry
    else:
        raise TypeError(f"Invalid registry type: {type(registry)}")

    validate_registry(catalog, parsed_registry)

    accepted = set(seed_set)
    target_period = str(period)

    for group in parsed_registry.get('groups', []):
        if str(group['contract']['period']) != target_period:
            continue
        group_cards = {str(m['card_id']) for m in group['members']}
        if seed_set & group_cards:
            accepted.update(group_cards)

    return accepted
