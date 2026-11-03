"""Independent structural/source checks and deterministic replay of the9B graph."""
import copy
from decimal import Decimal
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import re

from rag_pensiones.graph_catalog import canonical_hash, validate_catalog,candidate_matches_need
from rag_pensiones.graph_documents import validate_document_corpus
from rag_pensiones.graph_types import CHAT_MODEL, CHAT_DIGEST, CHAT_OPTIONS, EMBED_MODEL, EMBED_DIGEST,PROFILE,stage_chat_settings,model_lifecycle_policy
from rag_pensiones.graph_units import unit_matches,published_variation
from rag_pensiones.structured_retrieval import ROOT
from experiments import source_recovery_policy as policy


class InvalidPlan(ValueError):
    pass


class TextContentRejection(InvalidPlan):
    """A generated claim fails evidence/content checks for an existing text need."""
    def __init__(self,message,*,code='text_content_rejected'):
        super().__init__(message);self.code=code


def contains_financial_quantity(text):
    """Recognize digit amounts with explicit percent/point/currency units only.

    Bare numbers, years, durations and spelled-out amounts are not classified.
    This checks generated claim text, never intent or source quotations.
    """
    number=r'(?<!\w)[+-]?\d+(?:[.,]\d+)*'
    scale=r'(?:(?:mil|miles|mill[oó]n|millones)\s+(?:de\s+)?)?'
    unit=r'(?:[%％]|por\s+ciento|porcentajes?|puntos?\s+(?:porcentuales?|b[aá]sicos?|(?:del?\s+)?PIB)|p\.?\s*p\.?|'+scale+r'(?:euros?|d[oó]lares?|libras?|EUR|USD|GBP|[€$£]))(?!\w)'
    return bool(re.search(number+r'\s*'+unit,text,re.IGNORECASE)
                or re.search(r'(?:[€$£]|\b(?:EUR|USD|GBP))\s*[+-]?\d',text,re.IGNORECASE))


def check_shape(value, schema):
    if 'oneOf'in schema:
        accepted=0
        for branch in schema['oneOf']:
            try:check_shape(value,branch)
            except InvalidPlan:continue
            accepted+=1
        if accepted!=1:raise InvalidPlan('JSON must match exactly one allowed decision branch')
    types = {'object':dict, 'array':list, 'string':str, 'boolean':bool, 'integer':int, 'null':type(None)}
    declared = schema.get('type')
    if declared is not None:
        allowed = declared if isinstance(declared,list) else [declared]
        if not any(type(value) is types[t] for t in allowed):
            raise InvalidPlan('JSON field type differs from schema')
    if 'enum' in schema and value not in schema['enum']:
        raise InvalidPlan('Unknown enumerated JSON value')
    if isinstance(value, dict):
        if not set(schema.get('required',[])) <= set(value):
            raise InvalidPlan('Missing required JSON fields')
        props = schema.get('properties',{})
        if schema.get('additionalProperties') is False and set(value)-set(props):
            raise InvalidPlan('Unknown JSON fields')
        for key,item in value.items():
            if key in props:
                check_shape(item,props[key])
    elif isinstance(value,list):
        if len(value)<schema.get('minItems',0) or len(value)>schema.get('maxItems',100000):
            raise InvalidPlan('Invalid JSON array length')
        for item in value:
            check_shape(item,schema.get('items',{}))
    elif isinstance(value,str):
        if (len(value)<schema.get('minLength',0)
                or ('maxLength'in schema and len(value)>schema['maxLength'])):
            raise InvalidPlan('Invalid JSON string length')
    return value


def runtime_manifest(services):
    from rag_pensiones.graph_pipeline import PROMPTS
    from rag_pensiones.graph_source_policy import POLICY_PATH,POLICY_SHA256
    services.authenticate()
    validate_catalog(services.catalog)
    validate_document_corpus(services.documents)
    files = list((ROOT/'rag_pensiones').glob('*.py'))
    files += list((ROOT/'experiments').glob('source_recovery*.py'))
    files += list((ROOT/'experiments').glob('*semantics.py'))
    files += [ROOT/'experiments'/name for name in ['excel_pilot.py','universal_database.py','validate_excel_pilot.py']]
    files += [ROOT/'experiments/query_graph.py',ROOT/'experiments/build_graph_assets.py',
              ROOT/'requirements.txt',ROOT/'requirements-langgraph.txt',
              services.database, services.database.parent/'validation.json', policy.PLAN,POLICY_PATH]
    for source in services.catalog['sources'].values():
        files += [ROOT/source['xlsx'],ROOT/source['pdf']]
    for source in services.documents['sources']:
        files.append(ROOT/source['source_path'])
    for source_path in {r['source_path']for r in services.catalog['observations'].values()}:
        files.append(ROOT/source_path)
    hashes={p.resolve().relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(set(files)) if p.is_file()}
    result=dict(schema_version='langgraph-rag-manifest-v1',profile=PROFILE,
                chat_model=CHAT_MODEL,chat_digest=CHAT_DIGEST,chat_options=copy.deepcopy(CHAT_OPTIONS),
                chat_stage_settings=dict(default=stage_chat_settings('default'),plan_check_batch=stage_chat_settings('plan_check_batch')),
                embedding_model=EMBED_MODEL,embedding_digest=EMBED_DIGEST,
                model_lifecycle_policy=model_lifecycle_policy(),
                catalog_hash=services.catalog['catalog_hash'], database_sha256=services.catalog['database_sha256'],
                document_corpus_hash=services.documents['corpus_hash'],scope=services.scope,
                reference_text_only=services.scope=='legacy_2025_reference',
                exposed_card_ids=[] if services.scope=='legacy_2025_reference' else [c['card_id'] for c in services.catalog['cards']],
                catalog_embedding_hash=(services.catalog.get('embedding_index')or{}).get('index_hash'),
                document_embedding_hash=(services.documents.get('embedding_index')or{}).get('index_hash'),
                prompt_hash=canonical_hash(PROMPTS),files=hashes,python=platform.python_version(),
                dependencies={p:importlib.metadata.version(p) for p in ['langgraph','langgraph-checkpoint-sqlite','duckdb','pypdf']},
                max_repairs=1,max_chat_calls=10)
    result['source_validation']='compiled_source_contract_v1'
    result['source_policy']=dict(policy_id=services.source_policy['policy_id'],path=POLICY_PATH.relative_to(ROOT).as_posix(),sha256=POLICY_SHA256)
    from rag_pensiones.graph_context import CONTEXT_PATH,CONTEXT_SHA256,FORECAST_DEFAULT_POLICY
    if getattr(services,'semantic_context',None)is not None:
        result['files'][CONTEXT_PATH.relative_to(ROOT).as_posix()]=CONTEXT_SHA256
        result['semantic_context']=dict(registry_id=services.semantic_context['registry_id'],path=CONTEXT_PATH.relative_to(ROOT).as_posix(),sha256=CONTEXT_SHA256)
        result['source_forecast_default']=FORECAST_DEFAULT_POLICY
    else:result['semantic_context']=None
    result['retrieval_diversification']=dict(enabled=True,pool_size=50,final_per_need=10,
        policy='one_per_source_object_then_score_fill')
    if getattr(services,'reranker',None)is not None:
        from rag_pensiones.graph_reranker import MODEL,REVISION,MODEL_DIR
        result['reranker']=dict(model=MODEL,revision=REVISION,device='cpu',pool_size=50,final_per_need=10,
            query_uses_interpretation=True,diversity_policy='one_per_source_object_then_score_fill',
            verified_files=json.loads((MODEL_DIR/'verified_files.json').read_text(encoding='utf-8')))
    result['manifest_hash']=canonical_hash(result)
    return result


def verify_manifest(manifest, *, services):
    if canonical_hash(manifest)!=canonical_hash(runtime_manifest(services)):
        raise ValueError('Graph runtime/assets manifest changed')
    return manifest


def metadata_period_endpoints(period):
    """Interpret an existing period label, never the user's natural language."""
    parts=period.split('-')
    return set(parts)if len(parts)==2 and all(len(p)==4 and p.isdecimal()for p in parts)else{period}


def validate_plan(plan, interpretation, candidates):
    cards={c['card_id']:c for c in candidates}
    needs={n['need_id']:n for n in interpretation['needs']}
    if len(needs)!=len(interpretation['needs']):
        raise InvalidPlan('Duplicate need identifier')
    if plan['status']=='clarification':
        if not plan.get('clarification') or plan.get('clarification_field') not in {'edition','scenario','unit','population','metric','period'}:
            raise InvalidPlan('Clarification must identify a material ambiguity')
        return plan
    selected=plan['selections']
    if {s['need_id'] for s in selected}!=set(needs) or len(selected)!=len(needs):
        raise InvalidPlan('Plan must preserve every interpreted need')
    for selection in selected:
        need=needs[selection['need_id']]
        if need['kind']=='text':
            if selection['card_ids']:
                raise InvalidPlan('Text need cannot fabricate a numeric source')
            continue
        for card_id in selection['card_ids']:
            if card_id not in cards:
                raise InvalidPlan('Selected source outside retrieved candidates')
            card=cards[card_id]
            if need.get('source_block') is not None:
                from rag_pensiones.graph_source_policy import canonical_block
                if canonical_block(card['block'])!=canonical_block(need['source_block']):
                    raise InvalidPlan('Wrong explicitly requested source object')
            if need.get('book') and card['book']!=need['book']:
                raise InvalidPlan('Wrong report edition')
            if need.get('series') and need['series'].casefold()!=str(card['series']).casefold():
                raise InvalidPlan('Wrong requested series')
            if not set(selection['periods'])<=set(card['periods']):
                raise InvalidPlan('Selected period unavailable in source')
            if (need.get('quantity_request')=='change'and card['period_kind']=='year'
                and published_variation(card['unit'],card.get('quantity_role',''))
                and len({p for label in need['periods']for p in metadata_period_endpoints(label)})==2):
                raise InvalidPlan('Two annual published variations are not one interval change')
            if not candidate_matches_need(dict(card,periods=selection['periods']),need):
                raise InvalidPlan('Selected source does not publish the temporal shape of the requested horizon')
            if selection['operation']=='difference'and card['period_kind']=='interval':
                raise InvalidPlan('An interval average/change is not a point-level subtraction operand')
            requested=set(need['periods']);chosen=set(selection['periods'])
            if selection['operation']=='published_change':
                chosen|={p for period in selection['periods']for p in metadata_period_endpoints(period)}
            elif selection['operation']=='difference':
                requested={p for period in need['periods']for p in metadata_period_endpoints(period)}
            if requested and not requested<=chosen:
                raise InvalidPlan('Plan silently substitutes a requested horizon')
            unit=card['unit'].lower()
            requested=need['unit']
            if not unit_matches(requested,card['unit'],card.get('quantity_role',''),
                                allow_level_operands=selection['operation']=='difference'):
                raise InvalidPlan('Requested unit family differs from the published source')
            identity=' '.join(str(card.get(k)or'') for k in ['metric','series','scope','title']).casefold()
            for term in need['scope_terms']:
                if term.casefold() not in identity:
                    raise InvalidPlan('Requested population/territory not evidenced by source')
            if need['population']=='individual' and not any(t in unit for t in ['per_person','individual']):
                raise InvalidPlan('An aggregate cannot answer a personal pension')
            if any(t in unit for t in ['signed_plot']) or any(t in card['quantity_role'] for t in ['scatter','paired_coordinates']):
                raise InvalidPlan('Graph coordinates are not a scalar aggregate')
        if selection['card_ids'] and not selection['periods']:
            raise InvalidPlan('A data lookup requires explicit periods')
        if selection['operation']=='published_change' and any(cards[c]['period_kind']!='interval' for c in selection['card_ids']):
            raise InvalidPlan('Published change requires an interval source')
    return plan


def execute_plan(plan, interpretation, candidates, services, *,query=None,context=None):
    from rag_pensiones.graph_context import enforce_context_scope
    registry=getattr(services,'semantic_context',None)
    if getattr(services,'scope',None)=='production_full'and registry is None:raise ValueError('Production semantic context registry is required before SQL')
    if registry is not None:
        authenticated={c['card_id']:c for c in services.catalog['cards']};offered={c['card_id']:c for c in candidates}
        for selection in plan['selections']:
            for cid in selection['card_ids']:
                if cid not in authenticated or any(offered[cid].get(key)!=value for key,value in authenticated[cid].items()):
                    raise ValueError('Execution candidate differs from authenticated catalog card')
    enforce_context_scope(plan,candidates,registry,query,context)
    cards={c['card_id']:c for c in candidates}
    facts=[]
    for selection in plan['selections']:
        local=[]
        for cid in selection['card_ids']:
            card=cards[cid]
            for period in selection['periods']:
                rows=[services.catalog['observations'][oid] for oid in card['observation_ids']
                      if services.catalog['observations'][oid]['period']==period]
                if not rows or len({r['value'] for r in rows})!=1:
                    raise InvalidPlan('No unique source value for selected period')
                row=sorted(rows,key=lambda r:r['observation_id'])[0]
                actual=policy.query_observation(services.database,observation_id=row['observation_id'])
                for key in ['book','unit','source_sheet','source_cell']:
                    if actual.get(key)!=row.get(key):
                        raise ValueError('Source API differs from catalog: '+key)
                if (actual.get('metric_literal')or actual.get('metric'))!=row['metric'] or (actual.get('series_literal')or actual.get('series'))!=row['series'] or Decimal(str(actual['value']))!=Decimal(row['value']):
                    raise ValueError('Source value/series/magnitude changed')
                source=Path(card['source_path'])
                path=(ROOT/source).resolve()
                path.relative_to(ROOT)
                if hashlib.sha256(path.read_bytes()).hexdigest()!=card['source_sha256']:
                    raise ValueError('Cited original source changed')
                fact=dict(row,need_id=selection['need_id'],card_id=cid,source_path=card['source_path'],
                          source_sha256=card['source_sha256'],derived=False)
                for key in ['warning','source_priority','calculation_safe','numeric_reconciliation_failed','global_calculation_prohibition']:
                    fact[key]=actual.get(key)
                if not Decimal(fact['value']).is_finite():
                    raise ValueError('Nonfinite numeric result')
                local.append(fact)
        if selection['operation']=='difference' and local:
            if len(local)!=2 or any(f['calculation_safe'] is not True or f['numeric_reconciliation_failed'] is True for f in local):
                raise InvalidPlan('Difference lacks two calculation-safe operands')
            keys=['book','metric','series','unit','scope','quantity_role']
            if any(local[0][k]!=local[1][k] for k in keys):
                raise InvalidPlan('Operand comparability not accredited')
            value=Decimal(local[1]['value'])-Decimal(local[0]['value'])
            original_unit=local[0]['unit']
            result_unit=('percentage_points_'+original_unit[len('percent_'):] if original_unit.startswith('percent_')
                         else'percentage_points'if original_unit=='percent'
                         else original_unit[:-len('percent')]+'percentage_points'if original_unit.endswith('_percent')
                         else original_unit)
            derived=dict(local[0],observation_id='derived:'+canonical_hash([f['observation_id'] for f in local])[:24],
                         value=str(value),unit=result_unit,operand_unit=original_unit,
                         period=local[0]['period']+' → '+local[1]['period'],derived=True,
                         operands=local,formula='operand_2 - operand_1')
            local=[derived]
        facts.extend(local)
    if len(facts)>40:
        raise InvalidPlan('Lookup exceeds bounded result budget')
    return facts


def numeric_claim(fact):
    if fact.get('supporting_only'):raise InvalidPlan('Supporting source cannot become an independent main claim')
    label=f"{fact['book'] or 'Fuente oficial'}; {fact['metric']}; {fact['series']}; {fact['period']}: {fact['value']} [{fact['unit']}]."
    if fact['derived']:
        label='Diferencia calculada ('+fact['formula']+'): '+label
    if fact.get('warning'):
        label+=' Aviso: '+fact['warning']
    if fact.get('source_policy_note'):label+=' Aviso de fuentes: '+fact['source_policy_note']
    operands=fact.get('operands',[fact])+fact.get('source_policy_comparisons',[])
    evidence=[dict(source_id='DB:'+f['observation_id'],source_kind='structured_cell',
                   observation_id=f['observation_id'],source_path=f['source_path'],source_sha256=f['source_sha256'],
                   source_file=Path(f['source_path']).name,source_sheet=f['source_sheet'],source_cell=f['source_cell'],
                   unit=f['unit'],value=f['value'],quote=f"{f['metric']}; {f['series']}; {f['period']}: {f['value']} [{f['unit']}]")
              for f in operands]
    for item,operand in zip(evidence,operands):
        if operand.get('supporting_only'):item['supporting_only']=True
        if operand.get('source_sheet')is None and operand.get('source_pdf_page')is not None:
            item.pop('quote',None)  # A table locator is not a fabricated verbatim quotation.
            item.update(source_kind='structured_official_pdf',doc_id=operand['source_sha256'],
                        page_number=operand['source_pdf_page'],source_locator=operand.get('source_locator'),
                        source_variant=operand.get('source_variant'),raw_value=operand.get('raw_value'),
                        original_amount_unit=operand.get('original_amount_unit'))
    return dict(text=label,evidence=evidence,need_id=fact['need_id'])


def align_literal_quote(quote,canonical_passage):
    """Align layout whitespace only; restore an exact source substring and offsets."""
    if not isinstance(quote,str)or not isinstance(canonical_passage,str):raise InvalidPlan('Invalid quote text')
    def normalize(text):
        characters=[];offsets=[];punctuation='.,;:!?%()[]{}«»'
        for match in re.finditer(r'\s+|\S',text):
            token=match.group(0)
            if token.isspace():
                before=text[:match.start()].rstrip();after=text[match.end():].lstrip()
                left=before[-1:]or'';right=after[:1]or''
                decimal=((left in {'.',','}and before[:-1].rstrip()[-1:].isdigit()and right.isdigit())or
                         (right in {'.',','}and left.isdigit()and after[1:].lstrip()[:1].isdigit()))
                if not decimal and((left and left in punctuation)or(right and right in punctuation)):continue
                token=' '
            characters.append(token);offsets.append((match.start(),match.end()))
        return ''.join(characters),offsets
    normalized,_=normalize(quote.strip())
    if not normalized:raise TextContentRejection('Empty quote')
    passage,offsets=normalize(canonical_passage);index=passage.find(normalized)
    while index>=0:
        end_index=index+len(normalized);left=passage[index-1:index]if index else'';right=passage[end_index:end_index+1]
        partial_word=((normalized[0].isalnum()and(left.isalnum()or left=='_'))or
                      (normalized[-1].isalnum()and(right.isalnum()or right=='_')))
        partial_number=(normalized[0].isdigit()and left in {'.',',','+','-'}and
                        (left in {'+','-'}or passage[max(0,index-2):index-1].isdigit()))
        partial_number|=normalized[-1].isdigit()and right in {'.',','}and passage[end_index+1:end_index+2].isdigit()
        if not partial_word and not partial_number:break
        index=passage.find(normalized,index+1)
    if index<0:raise TextContentRejection('Quote outside retrieved canonical passage')
    start=offsets[index][0];end=offsets[index+len(normalized)-1][1]
    return dict(quote=canonical_passage[start:end],start_char=start,end_char=end)


def validate_text_claims(claims, needs, documents, *, hybrid=False):
    source_map={d['source_id']:d for d in documents}
    need_map={n['need_id']:n for n in needs if n['kind']=='text'}
    for claim in claims:
        if claim['need_id'] not in need_map:raise TextContentRejection('Text claim outside a textual need',code='unknown_need_id')
        if hybrid and contains_financial_quantity(claim['text']):
            raise TextContentRejection('Hybrid text must explain causes without financial figures or percentages',code='hybrid_financial_quantity')
        if not claim['text'].strip() or not claim['evidence']:
            raise TextContentRejection('Text claim lacks content or evidence')
        need=need_map[claim['need_id']]
        for evidence in claim['evidence']:
            source=source_map.get(evidence['source_id'])
            if source is None or len(evidence['quote'])<12:
                raise TextContentRejection('Quote outside retrieved canonical passage')
            aligned=align_literal_quote(evidence['quote'],source['text'])
            if len(aligned['quote'])<12:raise TextContentRejection('Quote too short after whitespace alignment')
            evidence.update(quote=aligned['quote'],quote_start_char=aligned['start_char'],quote_end_char=aligned['end_char'])
            if need.get('book') and source['book']!=need['book']:
                raise TextContentRejection('Textual explanation belongs to a different report')
            evidence.update({k:source[k] for k in ['doc_id','page_number','source_file','source_path','source_sha256']})
            evidence['source_kind']='pdf_text'
    return claims


def validate_text_claims_isolated(claims,needs,documents,*,hybrid=False):
    accepted=[];rejections=[]
    for index,claim in enumerate(claims):
        try:accepted.extend(validate_text_claims([copy.deepcopy(claim)],needs,documents,hybrid=hybrid))
        except TextContentRejection as exc:
            rejections.append(dict(need_id=claim['need_id'],claim_index=index,reason=str(exc),code=exc.code,stage='generate'))
    return accepted,rejections


def replay_graph_report(report, *, services=None):
    from rag_pensiones.graph_pipeline import answer_graph, GraphServices, load_default_services
    from rag_pensiones.graph_models import GraphStageClient
    if services is None:
        services=load_default_services(scope=report['manifest']['scope'])
    if runtime_manifest(services)!=report['manifest']:
        raise ValueError('Graph manifest changed before replay')
    calls=copy.deepcopy(report['stage_calls'])
    class ReplayTransport:
        def request(self,method,path,payload=None):
            if path=='/api/tags':
                return {'models':[{'name':CHAT_MODEL,'digest':CHAT_DIGEST},{'name':EMBED_MODEL,'digest':EMBED_DIGEST}]}
            if not calls:
                raise ValueError('Replay requested an unrecorded inference')
            call=calls.pop(0)
            if call['endpoint']!=path or canonical_hash(call['request'])!=canonical_hash(payload) or call['model']!=payload['model']:
                raise ValueError('Replay inference differs from recorded request')
            expected_digest=CHAT_DIGEST if payload['model']==CHAT_MODEL else EMBED_DIGEST
            if call.get('digest_before')!=expected_digest or call.get('digest_after')!=expected_digest or not call.get('attempted') or not isinstance(call.get('response'),dict):
                raise ValueError('Invalid model trace in replay')
            return copy.deepcopy(call['response'])
    # Source services were authenticated on construction; preserve their verified
    # asset identities while replacing only the inference transport.
    replay_services=copy.copy(services)
    replay_services.client=GraphStageClient(transport=ReplayTransport())
    replayed=answer_graph(report['query'],_execution_query=report.get('execution_query'),context=report.get('execution_context',report.get('effective_context',report['context'])),services=replay_services,
                          thread_id=report['session_id'])
    if calls:
        raise ValueError('Replay left unused model calls')
    if len(replayed['stage_calls'])!=len(report['stage_calls']):raise ValueError('Replay stage count changed')
    for original,reproduced in zip(report['stage_calls'],replayed['stage_calls']):
        for field in ['stage','attempted','error','digest_before','digest_after','parsed_response']:
            if canonical_hash(original.get(field))!=canonical_hash(reproduced.get(field)):
                raise ValueError('Replay stage validation differs: '+field)
    for key in [k for k in ['execution_query','execution_context','standalone_query','conversation_resolution']if k in report]+['resolved_query','status','publication','retrieval','plan','numeric_facts','manifest',
                'raw_interpretation','raw_intent_audit','interpretation','planning_decisions',
                'offered_capabilities','ambiguity_summaries','checker_verdicts','need_outcomes','text_rejections','text_coverage',
                'grounding_raw','grounding_records','checked_numeric_results','raw_result_checks']:
        if canonical_hash(replayed.get(key))!=canonical_hash(report.get(key)):
            raise ValueError('Graph replay differs: '+key)
    return replayed
