"""Mechanical semantic cards over certified public facts, plus hybrid discovery.

Question interpretation is deliberately absent: BM25 and vector similarity only
produce candidates. A planner must choose from their actual identifiers.
"""
from collections import Counter, defaultdict
from dataclasses import asdict
from decimal import Decimal
import copy
import hashlib
import json
import math
from pathlib import Path
import re
import unicodedata

from rag_pensiones.graph_types import (CatalogCard, EMBED_DIGEST, EMBED_DIMENSION,
                                       EMBED_MODEL)
from rag_pensiones.qwen_embeddings import QUERY_PREFIX, vector
from rag_pensiones.graph_models import MAX_EMBED_INPUT_BYTES
from rag_pensiones.graph_units import unit_family,unit_matches,published_variation
from rag_pensiones.structured_retrieval import CertifiedStructuredRetriever, DEFAULT_DATABASE
from experiments import source_recovery_policy as source_policy

SCHEMA = 'graph-semantic-catalog-v1'


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def tokenize(text):
    plain = ''.join(c for c in unicodedata.normalize('NFKD', str(text))
                    if not unicodedata.combining(c)).lower()
    return re.findall(r'[a-z0-9]+', plain)


def _content(catalog):
    return {k: catalog[k] for k in ['schema_version', 'database_sha256', 'sources',
                                    'cards', 'observations']}


def build_catalog(database: Path = DEFAULT_DATABASE) -> dict:
    service = CertifiedStructuredRetriever(database)
    rows = service.catalog()  # Certified public view; never the regex query resolver.
    groups = defaultdict(list)
    observations = {}
    sources = copy.deepcopy(service._sources)
    for row in rows:
        oid = row['observation_id']
        value = Decimal(str(row['value']))
        if not value.is_finite():
            raise ValueError('Nonfinite source value')
        role = str(row.get('quantity_role') or row.get('kind') or '')
        scope = str(row.get('scope_literal') or row.get('scope') or '')
        identity = (row['book'], str(row['block']), row['metric_name'],
                    row['series_name'], row['unit'], row['period_kind'], role, scope,
                    str(row.get('original_title') or ''), row.get('source_pdf_path'),
                    row.get('source_pdf_sha256'))
        groups[identity].append(row)
        source = sources.get(row['book'], {})
        source_path = source.get('xlsx') or row.get('source_pdf_path')
        source_sha = source.get('xlsx_sha') or row.get('source_pdf_sha256')
        if not source_path or not source_sha:
            raise ValueError('Observation lacks certified provenance')
        record = dict(observation_id=oid, book=row['book'], block=str(row['block']),
                      metric=row['metric_name'], series=row['series_name'],
                      period=row['period'], period_kind=row['period_kind'],
                      unit=row['unit'], value=str(row['value']),
                      source_sheet=row.get('source_sheet'), source_cell=row.get('source_cell'),
                      quantity_role=role, scope=scope,
                      calculation_safe=row.get('calculation_safe'),
                      numeric_reconciliation_failed=row.get('numeric_reconciliation_failed'),
                      warning=row.get('warning'), source_priority=row.get('source_priority'),
                      source_path=source_path, source_sha256=source_sha,
                      source_pdf_page=row.get('source_pdf_page'), source_pdf_quote=row.get('source_pdf_quote'))
        if oid in observations and observations[oid] != record:
            raise ValueError('Conflicting observation identity')
        if row.get('numeric_reconciliation_failed') is True or not source.get('xlsx'):
            # The public view stores numbers; the certified API enriches user policy.
            enriched = source_policy.query_observation(service.database, observation_id=oid)
            if Decimal(str(enriched['value'])) != value:
                raise ValueError('Policy enrichment changed the catalog value')
            for field in ['warning', 'source_priority', 'global_calculation_prohibition',
                          'numeric_reconciliation_failed', 'calculation_safe']:
                record[field] = enriched.get(field)
            if not source.get('xlsx'):
                record['source_pdf_page']=enriched.get('pdf_page')
                record['source_locator']=enriched.get('source_locator')
                record['source_variant']=enriched.get('source_variant')
                record['original_amount_unit']=enriched.get('original_amount_unit')
                record['raw_value']=str(enriched.get('raw_value'))
        else:
            record['global_calculation_prohibition'] = row.get('global_calculation_prohibition')
        observations[oid] = record
    cards = []
    for key, members in sorted(groups.items(), key=lambda item: repr(item[0])):
        book, block, metric, series, unit, pkind, role, scope, title, pdf_path, pdf_sha = key
        source = sources.get(book, {})
        path, sha = source.get('xlsx') or pdf_path, source.get('xlsx_sha') or pdf_sha
        if not path or not sha:
            raise ValueError('Card lacks certified provenance')
        periods = sorted({m['period'] for m in members})
        ids = sorted({m['observation_id'] for m in members})
        warnings = _period_warnings(ids, observations)
        description = '\n'.join(f'{label}: {value}' for label, value in [
            ('Informe', book), ('Objeto', block), ('Título', title), ('Magnitud', metric),
            ('Serie', series), ('Unidad', unit), ('Ámbito', scope), ('Rol', role),
            ('Tipo de periodo', pkind), ('Periodos disponibles', ', '.join(periods))]
            if value is not None and value != '')
        if warnings:
            description += '\nAvisos por observación (no por serie completa): ' + ', '.join(
                sorted({w['period'] for w in warnings}))
        card = CatalogCard('card:' + canonical_hash(key)[:24], book, block, metric,
                           series, unit, pkind, role, scope, title, periods, ids,
                           path, sha, description, warnings)
        cards.append(asdict(card))
    catalog = dict(schema_version=SCHEMA, database_sha256=service_database_hash(service),
                   sources=sources, cards=cards, observations=observations)
    catalog['catalog_hash'] = canonical_hash(_content(catalog))
    validate_catalog(catalog)
    return catalog


def service_database_hash(service):
    service.verify_snapshot()
    return hashlib.sha256(service.database.read_bytes()).hexdigest()


def _period_warnings(ids, observations):
    if any(oid not in observations for oid in ids):
        raise ValueError('Unknown observation in card warnings')
    fields = ['period', 'warning', 'source_priority', 'calculation_safe',
              'numeric_reconciliation_failed', 'global_calculation_prohibition']
    return [dict(observation_id=oid, **{field: observations[oid].get(field) for field in fields})
            for oid in sorted(ids) if observations[oid].get('warning')
            or observations[oid].get('calculation_safe') is False]


def validate_catalog(catalog):
    if not isinstance(catalog, dict) or set(catalog) - {'schema_version', 'database_sha256', 'sources', 'cards', 'observations', 'catalog_hash', 'embedding_index'}:
        raise ValueError('Unknown catalog fields')
    if catalog.get('schema_version') != SCHEMA or catalog.get('catalog_hash') != canonical_hash(_content(catalog)):
        raise ValueError('Catalog content/hash mismatch')
    ids = [c['card_id'] for c in catalog['cards']]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate card identifier')
    for card in catalog['cards']:
        if card.get('period_warnings') != _period_warnings(card['observation_ids'], catalog['observations']):
            raise ValueError('Card period warnings differ from observation policy')
        for oid in card['observation_ids']:
            if oid not in catalog['observations']:
                raise ValueError('Card references unknown observation')
            row = catalog['observations'][oid]
            for k in ['book', 'block', 'metric', 'series', 'unit', 'period_kind', 'scope', 'quantity_role']:
                if row[k] != card[k]:
                    raise ValueError('Card semantic identity mismatch: ' + k)
            if row['period'] not in card['periods'] or not Decimal(row['value']).is_finite():
                raise ValueError('Invalid card period/value')
    return catalog


def verify_catalog_source(catalog, database: Path = DEFAULT_DATABASE):
    """Authenticate a loaded card set against certified public rows, not its self hash."""
    validate_catalog(catalog)
    expected = build_catalog(database)
    if catalog['database_sha256'] != expected['database_sha256'] or catalog['catalog_hash'] != expected['catalog_hash']:
        raise ValueError('Catalog does not match certified source rows')
    return catalog


def embedding_description(card):
    """Bound vector input; preserve the complete source card for planning/audit."""
    description=card['description']
    if len(description.encode())<=MAX_EMBED_INPUT_BYTES:return description
    periods=card['periods']
    summary=(f"Periodos disponibles: {len(periods)} etiquetas; primera: {periods[0] if periods else ''}; "
             f"última: {periods[-1] if periods else ''}. Lista exacta completa en metadata de la ficha.")
    result='\n'.join(summary if line.startswith('Periodos disponibles: ')else line
                     for line in description.splitlines())
    if len(result.encode())>MAX_EMBED_INPUT_BYTES:
        raise ValueError('Embedding identity exceeds byte budget; cannot truncate semantic fields')
    return result


def attach_embeddings(catalog, embeddings):
    validate_catalog(catalog)
    if len(embeddings) != len(catalog['cards']):
        raise ValueError('Card/vector count mismatch')
    result = copy.deepcopy(catalog)
    payload = dict(catalog_hash=catalog['catalog_hash'], model=EMBED_MODEL,
                   model_digest=EMBED_DIGEST, dimension=EMBED_DIMENSION,
                   embedding_text_hash=canonical_hash([embedding_description(c)for c in catalog['cards']]),
                   document_prefix='', query_prefix=QUERY_PREFIX,
                   vectors={c['card_id']: vector(v) for c, v in zip(catalog['cards'], embeddings)})
    payload['index_hash'] = canonical_hash(payload)
    result['embedding_index'] = payload
    return result


def _vectors(catalog):
    index = catalog.get('embedding_index')
    if index is None:
        return None
    payload = {k: v for k, v in index.items() if k != 'index_hash'}
    expected = dict(catalog_hash=catalog['catalog_hash'], model=EMBED_MODEL,
                    model_digest=EMBED_DIGEST, dimension=EMBED_DIMENSION,
                    embedding_text_hash=canonical_hash([embedding_description(c)for c in catalog['cards']]),
                    document_prefix='', query_prefix=QUERY_PREFIX)
    if any(index.get(k) != v for k, v in expected.items()) or index.get('index_hash') != canonical_hash(payload):
        raise ValueError('Stale or incompatible catalog embedding index')
    if set(index['vectors']) != {c['card_id'] for c in catalog['cards']}:
        raise ValueError('Catalog embedding identifiers differ')
    return {k: vector(v) for k, v in index['vectors'].items()}


def retrieve_catalog(catalog: dict, query: str, query_vector: list[float] | None = None,
                     top_k: int = 10, *, allowed_card_ids: set[str] | None = None) -> list[dict]:
    validate_catalog(catalog)
    if not isinstance(query, str) or not query.strip() or type(top_k) is not int or top_k < 1:
        raise ValueError('Nonempty query and positive integer top_k required')
    vectors = _vectors(catalog)
    if (vectors is None) != (query_vector is None):
        raise ValueError('Both document and query vectors required for hybrid retrieval')
    qvector = vector(query_vector) if query_vector is not None else None
    if allowed_card_ids is not None and (not isinstance(allowed_card_ids, set)
            or not allowed_card_ids <= {c['card_id'] for c in catalog['cards']}):
        raise ValueError('Unknown or invalid allowed card identifiers')
    pool = [c for c in catalog['cards'] if allowed_card_ids is None or c['card_id'] in allowed_card_ids]
    docs = [tokenize(c['description']) for c in pool]
    n = len(docs)
    if not n:
        return []
    avg = sum(map(len, docs)) / n or 1
    counts = [Counter(d) for d in docs]
    df = Counter(t for d in docs for t in set(d))
    scores = []
    for card, doc, count in zip(pool, docs, counts):
        bm25 = 0.0
        for term in sorted(set(tokenize(query))):
            freq = count[term]
            bm25 += math.log(1 + (n - df[term] + .5) / (df[term] + .5)) * freq * 2.5 / (freq + 1.5 * (.25 + .75 * len(doc) / avg))
        cosine = sum(a * b for a, b in zip(qvector, vectors[card['card_id']])) if qvector is not None else None
        scores.append(dict(card=card, bm25=bm25, cosine=cosine))
    lexical = sorted((s for s in scores if s['bm25'] > 0), key=lambda s: (-s['bm25'], s['card']['card_id']))
    ranks = {s['card']['card_id']: 1 / (60 + i) for i, s in enumerate(lexical, 1)}
    if vectors is not None:
        semantic = sorted(scores, key=lambda s: (-s['cosine'], s['card']['card_id']))
        for i, score in enumerate(semantic, 1):
            oid = score['card']['card_id']
            ranks[oid] = ranks.get(oid, 0) + 1 / (60 + i)
    selected = [s for s in scores if s['card']['card_id'] in ranks]
    selected.sort(key=lambda s: (-ranks[s['card']['card_id']], s['card']['card_id']))
    return [dict(**copy.deepcopy(s['card']), retrieval=dict(mode='hybrid_rrf' if vectors is not None else 'bm25_only',
                 bm25=s['bm25'], cosine=s['cosine'], rrf=ranks[s['card']['card_id']], rank=i))
            for i, s in enumerate(selected[:top_k], 1)]


def candidate_matches_need(card: dict, need: dict, *, metadata_discovery: bool = False) -> bool:
    """Constrain discovery using interpreted metadata; this does not authorize a plan."""
    from rag_pensiones.graph_validation import metadata_period_endpoints
    if need.get('kind') != 'data':
        return False
    if need.get('book') is not None and card['book'] != need['book']:
        return False
    if need.get('source_block') is not None:
        from rag_pensiones.graph_source_policy import canonical_block
        if canonical_block(card['block'])!=canonical_block(need['source_block']):return False
    unit = card['unit'].lower()
    requested = need.get('unit', 'unspecified')
    if not unit_matches(requested,card['unit'],card.get('quantity_role',''),allow_level_operands=True):return False
    available=set(card['periods']);requested_periods=set(need.get('periods',[]))
    endpoints={p for label in requested_periods for p in metadata_period_endpoints(label)}
    variation=published_variation(card['unit'],card.get('quantity_role',''))
    if (need.get('quantity_request')=='change'and not variation and card['period_kind']=='year'
        and len(endpoints)!=2 and(not metadata_discovery or len(endpoints)==1)):
        return False  # A point level cannot answer a one-year published change.
    if card['period_kind']=='interval'and len(endpoints)==1:
        return False  # An interval statistic needs both requested endpoints.
    if (need.get('quantity_request')=='change'and card['period_kind']=='year'
        and published_variation(card['unit'],card.get('quantity_role',''))
        and len({p for label in requested_periods for p in metadata_period_endpoints(label)})==2):
        return False  # Two independent annual deltas are not a full-span change.
    if metadata_discovery:
        # Interpreted years are tentative during discovery. Keep dimensions and
        # temporal roles strict; only source-label availability is deferred.
        variation=published_variation(card['unit'],card.get('quantity_role',''))
        if card['period_kind']=='interval'and not variation:
            if need.get('quantity_request')=='change':return False
            return need.get('quantity_request')=='mean'or any(len(metadata_period_endpoints(p))==2 for p in requested_periods)
        return True
    relative_change=(requested in {'percent','unspecified'}and need.get('quantity_request')=='change'
                     and published_variation(card['unit'],card.get('quantity_role','')))
    if requested not in {'percentage_points_gdp','percentage_points'}and not relative_change:
        # A published mean over an interval is not the level at its last year.
        # A requested average interval must itself be an available source label.
        return requested_periods<=available
    if requested in {'percentage_points_gdp','percentage_points'}and unit_family(unit)=={'percentage_points_gdp':'percent_gdp','percentage_points':'percent'}[requested]:
        # Potential subtraction operands are point levels, never interval means.
        if card['period_kind']!='year':return False
        endpoints={p for period in requested_periods for p in metadata_period_endpoints(period)}
        return endpoints<=available
    if requested_periods<=available:return True
    if card['period_kind']!='interval':return False
    endpoints={p for period in requested_periods for p in metadata_period_endpoints(period)}
    # Both endpoints must refer to one actually published interval. Two adjacent
    # intervals cannot impersonate a full-span published change.
    return len(endpoints)==2 and any(metadata_period_endpoints(p)==endpoints for p in available)


def retrieve_catalog_for_needs(catalog: dict, query: str, query_vector: list[float] | None,
                               needs: list[dict], top_k_per_need: int = 10, *, metadata_discovery: bool = False) -> list[dict]:
    """Filter before ranking and fairly expose independent needs and unspecified editions."""
    if type(top_k_per_need) is not int or top_k_per_need < 1:
        raise ValueError('Positive per-need retrieval budget required')
    lists = []
    for need in needs:
        if need.get('kind') != 'data':
            continue
        eligible = [c for c in catalog['cards'] if candidate_matches_need(c, need,metadata_discovery=metadata_discovery)
                    and derivation_operands_eligible(c,need,catalog,metadata_discovery=metadata_discovery)]
        groups = defaultdict(set)
        for card in eligible:
            groups[card['book'] if need.get('book') is None else 'requested'].add(card['card_id'])
        # Interpreted metric/qualifiers keep follow-up context without mixing
        # distinct concepts from a multi-need question in the lexical ranking.
        search_query = ' '.join(str(t) for t in [need.get('metric'), need.get('series'),
                                *need.get('scope_terms', [])] if t and str(t).strip())
        search_query = search_query or need.get('query_span') or query
        ranked = [retrieve_catalog(catalog, search_query, query_vector,
                                  top_k=top_k_per_need, allowed_card_ids=groups[book])
                  for book in sorted(groups, key=lambda b: str(b))]
        rows = [part[rank] for rank in range(top_k_per_need) for part in ranked if rank < len(part)]
        # An unspecified edition keeps at least one result from every represented book.
        limit = max(top_k_per_need, len(ranked))
        lists.append((need['need_id'], rows[:limit]))
    merged = {}; ordered = []
    for rank in range(max((len(rows) for _, rows in lists), default=0)):
        for need_id, rows in lists:
            if rank >= len(rows):
                continue
            card = rows[rank]; cid = card['card_id']
            if cid not in merged:
                merged[cid] = card
                card['matched_need_ids'] = []
                card['need_retrieval'] = []
                ordered.append(card)
            merged[cid]['matched_need_ids'].append(need_id)
            merged[cid]['need_retrieval'].append(dict(need_id=need_id, rank=rank + 1,
                                                      scores=copy.deepcopy(card['retrieval'])))
    return ordered


def derivation_operands_eligible(card,need,catalog,*,metadata_discovery=False):
    """Exclude impossible level operands before they consume a retrieval slot."""
    from rag_pensiones.graph_validation import metadata_period_endpoints
    pair=(need.get('unit'),unit_family(card['unit'],card.get('quantity_role','')))
    level_change=(need.get('quantity_request')=='change'and card['period_kind']=='year'
                  and not published_variation(card['unit'],card.get('quantity_role','')))
    if not level_change and pair not in {('percentage_points_gdp','percent_gdp'),('percentage_points','percent')}:return True
    if need.get('quantity_request')!='change'or card['period_kind']!='year':return False
    endpoints={p for label in need.get('periods',[])for p in metadata_period_endpoints(label)}
    keys=['book','metric','series','unit','scope','quantity_role']
    if metadata_discovery and(len(endpoints)==0 or len(endpoints)>2):
        # Existence of a possible safe pair only admits a source to grounding.
        # It neither chooses endpoints nor changes the interpreted need.
        safe_periods=set()
        for oid in card['observation_ids']:
            row=catalog['observations'][oid]
            if row['period']not in card['periods']or(endpoints and row['period']not in endpoints):continue
            if (row.get('calculation_safe')is not True or row.get('global_calculation_prohibition')is True
                or not Decimal(row['value']).is_finite()
                or any(row.get(key)!=card.get(key)for key in keys)):continue
            safe_periods.add(row['period'])
        return len(safe_periods)>=2
    if len(endpoints)!=2:return False
    rows=[catalog['observations'][oid]for oid in card['observation_ids']
          if catalog['observations'][oid]['period']in endpoints]
    if len(rows)!=2 or any(r.get('calculation_safe')is not True or r.get('global_calculation_prohibition')is True for r in rows):return False
    return (all(rows[0].get(k)==rows[1].get(k)for k in keys)
            and all(Decimal(r['value']).is_finite()for r in rows))
