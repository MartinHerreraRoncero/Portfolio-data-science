"""User-authorized, primary-source-reviewed precedence between published records.

The registry is data: it never alters the certified database, source labels or
values. Only its exact reviewed pairs can activate priority and disclosure.
"""
import copy
from decimal import Decimal
import hashlib
from pathlib import Path

from pypdf import PdfReader
from rag_pensiones.graph_catalog import canonical_hash,candidate_matches_need,tokenize
from rag_pensiones.qwen_embeddings import _json
from rag_pensiones.structured_retrieval import ROOT
from experiments import source_recovery_policy as api

POLICY_PATH=ROOT/'data/processed/graph_source_policy_v3.json'
POLICY_SHA256='b40981290a63a0a36b27a7f2fea0690e598f7725ef6307ee442ca7af1d91975e'


def canonical_block(label):
    return ''.join(tokenize(label))


def verify_policy_asset(policy=None):
    raw=POLICY_PATH.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=POLICY_SHA256:
        raise ValueError('Reviewed source policy bytes changed')
    loaded=_json(raw)
    if policy is not None and canonical_hash(loaded)!=canonical_hash(policy):
        raise ValueError('In-memory source policy changed')
    return loaded


def _source_path(relative):
    path=(ROOT/relative).resolve();path.relative_to(ROOT)
    return path


def load_source_policy(catalog,database):
    policy=verify_policy_asset()
    if policy['schema_version']!='curated-summary-source-policy-v1':raise ValueError('Unknown source policy')
    if policy['catalog_hash']!=catalog['catalog_hash']:raise ValueError('Policy catalog changed')
    db=policy['certified_database']
    if (_source_path(db['path'])!=Path(database).resolve()
        or hashlib.sha256(Path(database).read_bytes()).hexdigest()!=db['sha256']):
        raise ValueError('Policy certified database changed')
    for source in policy['sources']:
        if hashlib.sha256(_source_path(source['source_path']).read_bytes()).hexdigest()!=source['source_sha256']:
            raise ValueError('Policy primary source changed')
    cards={c['card_id']:c for c in catalog['cards']}
    fields=['book','block','metric','series','unit','period_kind','scope','quantity_role',
            'period','source_sheet','source_cell','source_path','source_sha256']
    for group in policy['groups']:
        for period in group['periods']:
            records=[period['preferred_record'],*period['comparison_records']]
            for record in records:
                if record is None:continue
                oid=record['observation_id'];row=catalog['observations'].get(oid)
                card=cards.get(record['card_id'])
                if row is None or card is None or oid not in card['observation_ids']:
                    raise ValueError('Policy observation/card binding changed')
                if any(row.get(k)!=record.get(k)for k in fields):
                    raise ValueError('Policy record identity changed')
                if Decimal(row['value'])!=Decimal(record['value_exact']) or Decimal(record['raw_ooxml'])!=Decimal(row['value']):
                    raise ValueError('Policy original value changed')
                actual=api.query_observation(database,observation_id=oid)
                if (Decimal(str(actual['value']))!=Decimal(row['value'])
                    or any(actual.get(k)!=row.get(k)for k in ['book','unit','source_sheet','source_cell'])
                    or (actual.get('metric_literal')or actual.get('metric'))!=row['metric']
                    or (actual.get('series_literal')or actual.get('series'))!=row['series']):
                    raise ValueError('Policy certified API differs')
    readers={}
    for evidence in policy['primary_review']['evidence']:
        path=_source_path(evidence['source_path'])
        if hashlib.sha256(path.read_bytes()).hexdigest()!=evidence['source_sha256'] or evidence['doc_id']!=evidence['source_sha256']:
            raise ValueError('Policy narrative source changed')
        if path not in readers:readers[path]=PdfReader(path)
        page=evidence['page_number']
        if type(page)is not int or not 1<=page<=len(readers[path].pages):raise ValueError('Policy page unavailable')
        text=' '.join((readers[path].pages[page-1].extract_text()or'').split())
        if ' '.join(evidence['quote'].split())not in text:raise ValueError('Policy quote differs from primary PDF')
    return policy


def _entries_for_need(policy,need,catalog,*,candidate_card_ids=None):
    """Bind reviewed precedence to eligible source members, never metric wording."""
    if not candidate_card_ids:return []
    bound=set(candidate_card_ids)
    if (need.get('kind')!='data' or need.get('source_block') is not None
        or need.get('population')=='individual' or need.get('unit')not in {'unspecified','percent_gdp'}):return []
    cards={c['card_id']:c for c in catalog['cards']};result=[]
    for group in policy['groups']:
        preferred=cards[group['preferred_card_id']]
        if not candidate_matches_need(preferred,need):continue
        related={group['preferred_card_id'],*group['comparison_card_ids']}
        if not bound & related:continue
        identity=' '.join(str(preferred.get(k)or'')for k in ['metric','series','scope','title']).casefold()
        if any(term.casefold()not in identity for term in need.get('scope_terms',[])):continue
        if need.get('series') is not None and canonical_block(need['series'])!=canonical_block(preferred['series']):continue
        for entry in group['periods']:
            if (entry['period']not in need.get('periods',[]) or not entry.get('apply_summary_priority')
                or entry.get('relation')!='different' or not entry.get('preferred_record')):continue
            if not entry['comparison_records']:continue
            if all(Decimal(r['value_exact'])==Decimal(entry['preferred_record']['value_exact'])for r in entry['comparison_records']):continue
            result.append((group,entry))
    return result


def expand_policy_candidates(catalog,candidates,needs,policy):
    result=copy.deepcopy(candidates);byid={c['card_id']:c for c in result}
    allcards={c['card_id']:c for c in catalog['cards']}
    for need in needs:
        bound={c['card_id']for c in candidates if need['need_id']in c.get('matched_need_ids',[])
               and candidate_matches_need(c,need)}
        for group,entry in _entries_for_need(policy,need,catalog,candidate_card_ids=bound):
            related={group['preferred_card_id'],*group['comparison_card_ids']}
            # A reviewed relation is followed only after a member was retrieved
            # for this need; it never selects a policy by keywords or expected value.
            if not any(cid in byid and need['need_id']in byid[cid].get('matched_need_ids',[])for cid in related):continue
            for cid in sorted(related):
                card=allcards[cid]
                if not candidate_matches_need(card,need):continue
                if cid not in byid:
                    byid[cid]=copy.deepcopy(card);byid[cid].update(matched_need_ids=[],need_retrieval=[],
                        retrieval=dict(mode='curated_relation',relation=group['group_id']))
                    result.append(byid[cid])
                item=byid[cid]
                if need['need_id']not in item['matched_need_ids']:item['matched_need_ids'].append(need['need_id'])
                annotation=dict(policy_id=policy['policy_id'],group_id=group['group_id'],need_id=need['need_id'],
                    period=entry['period'],role='preferred_summary'if cid==group['preferred_card_id']else'comparison',
                    preferred_card_id=group['preferred_card_id'],comparison_card_ids=group['comparison_card_ids'],
                    primary_evidence=[dict(doc_id=doc,pages=sorted({e['page_number']for e in policy['primary_review']['evidence']if e['doc_id']==doc}))
                        for doc in sorted({s['source_sha256']for s in policy['sources']if s['book']==group['book']and s['kind']=='pdf'})])
                if annotation not in item.setdefault('source_policy',[]):item['source_policy'].append(annotation)
    return result


def enforce_source_policy(plan,interpretation,candidates,policy,catalog):
    from rag_pensiones.graph_validation import InvalidPlan
    if plan['status']!='ready':return
    byid={c['card_id']:c for c in candidates};needs={n['need_id']:n for n in interpretation['needs']}
    for selection in plan['selections']:
        need=needs[selection['need_id']]
        if selection['operation']!='lookup':continue
        applicable={a['group_id']:a for c in candidates for a in c.get('source_policy',[])
                    if a['need_id']==need['need_id']and a['period']in selection['periods']}
        for group,entry in _entries_for_need(policy,need,catalog,candidate_card_ids=selection['card_ids']):
            if group['group_id']not in applicable or entry['period']not in selection['periods']:continue
            if not selection['card_ids']:continue
            if not set(selection['card_ids'])&{group['preferred_card_id'],*group['comparison_card_ids']}:continue
            if (group['preferred_card_id']not in selection['card_ids']
                or set(selection['card_ids'])&set(group['comparison_card_ids'])):
                raise InvalidPlan('Reviewed source policy requires the summary as main; chart values are supporting disclosures')


def add_policy_support(facts,interpretation,candidates,services):
    from rag_pensiones.graph_validation import execute_plan
    result=[];needs={n['need_id']:n for n in interpretation['needs']}
    for original in facts:
        fact=copy.deepcopy(original);result.append(fact)
        if fact.get('derived'):continue
        need=needs[fact['need_id']]
        for group,entry in _entries_for_need(services.source_policy,need,services.catalog,candidate_card_ids=[fact['card_id']]):
            if fact['observation_id']!=entry['preferred_observation_id']:continue
            supports=[]
            for record in entry['comparison_records']:
                selection=dict(need_id=need['need_id'],card_ids=[record['card_id']],periods=[fact['period']],operation='lookup')
                plan=dict(status='ready',selections=[selection])
                support=execute_plan(plan,interpretation,[next(c for c in services.catalog['cards']if c['card_id']==record['card_id'])],services)[0]
                support.update(supporting_only=True,source_policy_group=group['group_id'])
                supports.append(support);result.append(support)
            fact['source_policy_comparisons']=supports
            fact['source_policy_id']=services.source_policy['policy_id']
            fact['source_policy_note']='Se prioriza el cuadro resumen. El gráfico publica '+', '.join(
                f"{s['value']} [{s['unit']}] ({s['source_sheet']}!{s['source_cell']}, {s['series']}, {s['period']})"for s in supports)+'. Discrepancia no explicada ni conciliada; se conservan ambos valores publicados.'
    if len(result)>80:raise ValueError('Policy support exceeds bounded result budget')
    return result
