"""Fixed, source-reviewed meanings; labels and hierarchy never authorize arithmetic."""
import copy
import hashlib
import re
from rag_pensiones.graph_catalog import canonical_hash,tokenize,candidate_matches_need,validate_catalog
from rag_pensiones.qwen_embeddings import _json
from rag_pensiones.structured_retrieval import ROOT
from rag_pensiones.graph_units import unit_matches

CONTEXT_PATH=ROOT/'data/processed/graph_semantic_context_v3.json'
CONTEXT_SHA256='ed435e1536d5bb24c26ef6c00cc2b31869e9de33b17d3b0f4a20507fb197d5c6'
FORECAST_DEFAULT_POLICY='verified_report_base_v1'


def _fixed(registry=None):
    raw=CONTEXT_PATH.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=CONTEXT_SHA256:raise ValueError('Reviewed semantic context bytes changed')
    fixed=_json(raw)
    if registry is not None and canonical_hash(registry)!=canonical_hash(fixed):raise ValueError('In-memory semantic context changed')
    return fixed


def _entries(registry):
    if registry is None:return {}
    _fixed(registry)
    return {cid:entry for entry in registry['entries']for cid in entry['source_card_ids']}


def verify_semantic_context(registry,catalog):
    _fixed(registry);validate_catalog(catalog)
    if registry['schema_version']!='source-semantic-context-registry-v3'or registry['catalog_hash']!=catalog['catalog_hash']:
        raise ValueError('Semantic context catalog identity changed')
    for relative,expected in registry['source_hashes'].items():
        path=(ROOT/relative).resolve();path.relative_to(ROOT)
        if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:raise ValueError('Semantic context primary source changed')
    cards={c['card_id']:c for c in catalog['cards']};seen=set()
    for entry in registry['entries']:
        if entry['status']!='verified'or not entry['primary_evidence_ids']:raise ValueError('Unverified semantic context entry')
        if any(eid not in registry['primary_evidence']for eid in entry['primary_evidence_ids']):raise ValueError('Missing semantic evidence')
        for cid in entry['source_card_ids']:
            card=cards[cid]
            if cid in seen or any(card.get(key)!=value for key,value in entry['raw'].items()):raise ValueError('Semantic context raw card binding changed')
            seen.add(cid);xlsx=entry['xlsx_evidence']
            if card['source_path']!=xlsx['source_path']or card['source_sha256']!=xlsx['source_sha256']or card['block']!=xlsx['sheet']:
                raise ValueError('Semantic context source binding changed')
            cells={catalog['observations'][oid]['source_cell']for oid in card['observation_ids']}
            if not set(xlsx['existing_observation_cells'])<=cells:raise ValueError('Semantic context observation binding changed')
            for parent_id in entry['proposed'].get('parent_card_ids',[]):
                parent=cards[parent_id]
                if parent_id==cid or any(parent.get(key)!=card.get(key)for key in ['book','unit','quantity_role']):
                    raise ValueError('Semantic parent edition/unit/role changed')
    if len(seen)!=71:raise ValueError('Semantic context source extent changed')
    return registry


def load_semantic_context(catalog):
    return verify_semantic_context(_fixed(),catalog)


def compact_semantic_context(card,registry):
    entry=_entries(registry).get(card['card_id'])
    if entry is None:return None
    if any(card.get(key)!=value for key,value in entry['raw'].items()):raise ValueError('Candidate differs from reviewed semantic source')
    context=entry['proposed'];result=dict(source_scope=context['aggregation_role'],semantic_label=context['semantic_label'],
                                        series_role=context['series_role'])
    if context.get('scope_qualifier'):result['scope_qualifier']=context['scope_qualifier']
    if context.get('context_forecast_vintage')and not re.search(r'\d{4}',str(card.get('series',''))):
        result['forecast_vintage']=context['context_forecast_vintage']
    if context.get('forecast_role'):result['forecast_role']=context['forecast_role']
    return result


def prefer_current_forecast(need,rows,registry):
    """Default only within a reviewed family that has a usable report base.

    This runs after grounding. Unknown roles and explicitly requested series or
    objects remain eligible; a base lacking the requested horizon has no veto.
    """
    if registry is None or need.get('book')is None or need.get('series')is not None or need.get('source_block')is not None:return rows
    entries=_entries(registry)
    def family(card):
        entry=entries.get(card['card_id'])
        if entry is None:return None
        if any(card.get(key)!=value for key,value in entry['raw'].items()):raise ValueError('Candidate differs from reviewed forecast source')
        context=entry['proposed']
        return (context['semantic_label'],card.get('book'),card.get('unit'),card.get('quantity_role'),card.get('period_kind'))
    bases=[card for card in rows if card.get('book')==need['book']and card['card_id']in entries
           and entries[card['card_id']]['proposed'].get('forecast_role')=='report_base'
           and candidate_matches_need(card,need)]
    result=[]
    for card in rows:
        entry=entries.get(card['card_id']);role=entry['proposed'].get('forecast_role')if entry else None
        alternatives={'prior_report','external_updated','demographic_alternative'}
        comparable=candidate_matches_need(card,need)and any(family(card)==family(base)and
            (bool(need.get('periods'))or set(card['periods'])==set(base['periods']))for base in bases)
        if role not in alternatives or not comparable:result.append(card)
    return result


def expand_context_candidates(catalog,candidates,needs,registry, *,metadata_discovery=False):
    entries=_entries(registry);result=copy.deepcopy(candidates);byid={c['card_id']:c for c in result}
    cards={c['card_id']:c for c in catalog['cards']}
    for need in needs:
        if need.get('kind')!='data'or need.get('source_block')is not None:continue
        seeds=[c for c in list(result)if need['need_id']in c.get('matched_need_ids',[])and c['card_id']in entries]
        for seed in seeds:
            context=entries[seed['card_id']]['proposed'];vintage=context.get('context_forecast_vintage')
            requested=set(re.findall(r'\d{4}',str(need.get('series')or'')))
            if not vintage or requested and requested!={vintage}:continue
            for cid,entry in entries.items():
                target=cards[cid];other=entry['proposed']
                if (other['semantic_label']!=context['semantic_label']or other.get('context_forecast_vintage')!=vintage
                    or any(seed.get(key)!=target.get(key)for key in ['book','unit','quantity_role','period_kind'])
                    or not unit_matches(need.get('unit','unspecified'),target['unit'],target.get('quantity_role',''),
                                        allow_level_operands=need.get('quantity_request')=='change')
                    or not candidate_matches_need(target,need,metadata_discovery=metadata_discovery)):continue
                if cid not in byid:
                    byid[cid]=copy.deepcopy(target);byid[cid].update(matched_need_ids=[],need_retrieval=[],
                        retrieval=dict(mode='reviewed_semantic_context',source_card_id=seed['card_id']))
                    result.append(byid[cid])
                if need['need_id']not in byid[cid]['matched_need_ids']:byid[cid]['matched_need_ids'].append(need['need_id'])
    for card in result:
        context=compact_semantic_context(card,registry)
        if context is not None:card['semantic_context']=context
    return result


def _user_history(context):
    if isinstance(context,list):return [text for item in context for text in _user_history(item)]
    if not isinstance(context,dict):return []
    if context.get('role')=='assistant':return []
    if context.get('role')=='user':return [context['content']]if isinstance(context.get('content'),str)else[]
    result=[context['clarification_answer']]if isinstance(context.get('clarification_answer'),str)else[]
    for key in ['messages','history','previous_context','context']:
        if key in context:result.extend(_user_history(context[key]))
    return result


def subtotal_scope_allowed(card,registry,query=None,context=None):
    entry=_entries(registry).get(card['card_id'])
    if entry is None or entry['proposed']['aggregation_role']!='subtotal':return True
    selector=entry['raw']['series']if entry['proposed']['series_role']=='model_group'else entry['raw']['metric']
    normal=lambda text:[word[:-1]if len(word)>4 and word.endswith('s')else word for word in tokenize(text)]
    wanted=normal(selector)
    for text in ([query]if isinstance(query,str)else[])+_user_history(context):
        words=normal(text)
        for index in range(len(words)-len(wanted)+1):
            if words[index:index+len(wanted)]!=wanted:continue
            before=index-1
            while before>=0 and words[before]in {'el','la','los','las','un','una'}:before-=1
            if wanted[0]not in {'no','sin'}and before>=0 and words[before]in {'no','sin'}:continue
            return True
    return False


def enforce_context_scope(plan,candidates,registry,query=None,context=None):
    if registry is None:return
    from rag_pensiones.graph_validation import InvalidPlan
    cards={c['card_id']:c for c in candidates}
    for selection in plan['selections']:
        for cid in selection['card_ids']:
            if not subtotal_scope_allowed(cards[cid],registry,query,context):
                raise InvalidPlan('Source subtotal requires its explicit subgroup selector in original user question/history')


def capability_options(capabilities,candidates,registry,*,query=None,context=None):
    cards={card['card_id']:card for card in candidates};options=[]
    for capability in capabilities:
        selection=capability['selection'];sources=[]
        for cid in selection['card_ids']:
            card=cards[cid]
            semantic=compact_semantic_context(card,registry)
            row=source_measure_view(card,semantic,query=query,context=context)
            sources.append({key:value for key,value in row.items()if value is not None and value!=''})
        options.append(dict(capability_id=capability['capability_id'],operation=selection['operation'],
                            periods=copy.deepcopy(selection['periods']),source_metadata=sources))
    return options


def source_measure_view(card,semantic_context,*,query=None,context=None):
    """One measure identity, its literal dimensions, then published provenance."""
    row=dict(measure=semantic_context['semantic_label']if semantic_context is not None else card['metric'])
    for key in ['card_id','series','book','unit','scope','block','quantity_role']:
        row[key]=copy.deepcopy(card.get(key))
    row['series_role']=semantic_context['series_role']if semantic_context is not None else None
    row['semantic_context']=copy.deepcopy(semantic_context)
    alias=' '.join(tokenize(card['metric']))
    if (query is None or semantic_context is None or(alias and any(
        ' '+alias+' 'in ' '+' '.join(tokenize(text))+' 'for text in [query,*_user_history(context)]))):
        row['published_measure_name']=card['metric']
    if semantic_context is None:row['container_title']=card.get('title')
    return row
