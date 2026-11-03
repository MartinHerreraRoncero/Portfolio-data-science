"""Question-bound intent contracts and source-bound planning capabilities."""
import copy
import json
import re
from decimal import Decimal
from rag_pensiones.graph_catalog import canonical_hash,candidate_matches_need
from rag_pensiones.graph_units import unit_family,published_variation,UNIT_FAMILIES
from rag_pensiones.graph_types import PROFILE

QUANTITY_REQUESTS=('level','change','contribution','mean','unspecified')
UNIT_WIRE_LABELS=dict(unspecified='no especificada',percent_gdp='porcentaje del PIB',
    percentage_points_gdp='puntos del PIB',percent='porcentaje',percentage_points='puntos porcentuales',
    euros='euros',persons='personas',pensions_count='número de pensiones')
WIRE_UNIT_CODES={label:code for code,label in UNIT_WIRE_LABELS.items()}


def literal_query_spans(query):
    if not isinstance(query,str)or not query.strip():raise ValueError('Nonempty question required')
    result=[query]
    # Only complete literal questions, never isolated report/date prefixes.
    for match in re.finditer(r'¿[^?]*\?',query):
        part=match.group(0)
        if part not in result:result.append(part)
    return result[:8]


COUNT_SCHEMA=dict(type='object',properties=dict(requested_result_count=dict(type='integer',enum=[1,2,3,4])),
                  required=['requested_result_count'],additionalProperties=False)


def literal_year_choices(query,context=None):
    text=query+'\n'+json.dumps(context,ensure_ascii=False,allow_nan=False)
    # Value provenance only. The model, not this scan, assigns publication/data roles.
    return sorted(set(re.findall(r'(?<!\d)\d{4}(?!\d)',text)))


def named_intent_schema(query,context,count):
    if type(count)is not int or not 1<=count<=4:raise ValueError('Invalid requested result count')
    years=literal_year_choices(query,context)
    def nullable(description=None):
        result=dict(type=['string','null'])
        if description:result['description']=description
        return result
    properties=dict(
        report_publication_year=dict(type=['string','null'],enum=[*years,None],description='PUBLICACIÓN, no dato/escenario. null si no se pide.'),
        target_data_periods=dict(type='array',items=dict(type='string',enum=years)if years else dict(type='string'),maxItems=12 if years else 0,description='AÑOS del DATO: level=uno; change/contribution/mean=dos extremos. No publicación/versión.'),
        requested_unit=dict(type='string',enum=list(UNIT_WIRE_LABELS.values()),description='Unidad expresamente solicitada. No inferirla por contribución o crecimiento. Sin unidad: no especificada. Peso sobre PIB: porcentaje del PIB; puntos del PIB: puntos del PIB.'),
        requested_measure=dict(type='string',description='Nombre de la medida y calificadores conceptuales, o relación narrativa pedida. Años, publicación, unidad y escenario van en sus campos separados.'),
        answer_kind=dict(type='string',enum=['data','text'],description='data=cantidades/comparaciones/contribuciones, aun ambiguas. text=explicación narrativa.'),
        requested_quantity=dict(type='string',enum=list(QUANTITY_REQUESTS),description='level=nivel; change=variación; contribution=aportación; mean=promedio.'),
        population_type=dict(type='string',enum=['aggregate','individual','subgroup','unspecified']),
        explicit_population_group=nullable('Subpoblación expresa; si no, null.'),
        explicit_geographic_area=nullable('Territorio expreso, no organismo/informe/año; si no, null.'),
        explicit_forecast_scenario=nullable('Copia la referencia literal a escenario/previsión de pregunta o historia USER, sin canonicalizar leyendas. null si no se pide o sólo se nombra publicación.'),
        explicit_table_or_chart_name=nullable(),
        question_fragment=dict(type='string',enum=literal_query_spans(query)))
    item=dict(type='object',properties=properties,required=list(properties),additionalProperties=False)
    return dict(type='object',properties=dict(answer_requests=dict(type='array',items=item,minItems=count,maxItems=count)),
                required=['answer_requests'],additionalProperties=False)


def decode_named_intent(raw,query,context=None,*,count,schema=None):
    from rag_pensiones.graph_validation import check_shape
    check_shape(raw,schema if schema is not None else named_intent_schema(query,context,count))
    def clean(value):
        if isinstance(value,str)and value.strip().casefold()in {'','null','none'}:return None
        return value.strip()if isinstance(value,str)else value
    from rag_pensiones.graph_context import _user_history
    from rag_pensiones.graph_catalog import tokenize
    original_texts=[query,*_user_history(context)];hint_audit=[];needs=[];retained={};mapping={}
    for index,item in enumerate(raw['answer_requests'],1):
        original_id='n'+str(index)
        identity=json.dumps(item,sort_keys=True,ensure_ascii=False,allow_nan=False)
        if identity in retained:
            mapping[original_id]=retained[identity];continue
        retained[identity]=original_id;mapping[original_id]=original_id
        qualifiers=[clean(item[k])for k in ['explicit_population_group','explicit_geographic_area']]
        periods=copy.deepcopy(item['target_data_periods'])
        if item['requested_quantity']in {'mean','contribution'}and len(periods)==2 and len(set(periods))==2:
            periods=['-'.join(sorted(periods))]
        series=clean(item['explicit_forecast_scenario'])
        normalized=' '.join(tokenize(series))if series is not None else ''
        if series is not None and(not normalized or not any(
            ' '+normalized+' 'in ' '+' '.join(tokenize(text))+' 'for text in original_texts)):
            hint_audit.append(dict(need_id='n'+str(index),discarded_hint=series,reason='not_literal_in_original_user_text'))
            series=None
        needs.append(dict(need_id='n'+str(index),query_span=item['question_fragment'],kind=item['answer_kind'],
            metric=item['requested_measure'],population=item['population_type'],
            scope_terms=list(dict.fromkeys(q for q in qualifiers if q is not None)),
            book=clean(item['report_publication_year']),series=series,
            source_block=clean(item['explicit_table_or_chart_name']),unit=WIRE_UNIT_CODES[item['requested_unit']],
            periods=periods,quantity_request=item['requested_quantity']))
    topics=[' | '.join(str(value)for value in [n['metric'],n['book'],n.get('series'),n.get('source_block'),n.get('unit'),n.get('population'),*n['periods'],*n['scope_terms']]
                      if value is not None and str(value)and value!='unspecified')for n in needs]
    result=dict(resolved_query='\n'.join([query,*topics]),needs=needs)
    result['request_coalescing']=dict(raw_count=len(raw['answer_requests']),retained_count=len(needs),original_to_retained=mapping)
    if hint_audit:result['discarded_scenario_hints']=hint_audit
    return result


def interpretation_schema(query,base_schema):
    schema=copy.deepcopy(base_schema)
    schema['properties']['needs']['items']['properties']['query_span']={
        'type':'string','enum':literal_query_spans(query)}
    return schema


def normalize_intent(raw,query):
    result=copy.deepcopy(raw);spans=literal_query_spans(query)
    for need in result['needs']:
        for field in ['book','series','source_block']:
            value=need.get(field)
            if isinstance(value,str):
                value=value.strip()
                if value.casefold()in {'','null','none'}:value=None
            need[field]=value
        if need['query_span']not in spans:raise ValueError('Intent span is outside question-bound literal choices')
        need.setdefault('quantity_request','unspecified')
        if need['quantity_request']not in QUANTITY_REQUESTS:raise ValueError('Unknown quantity request')
    return result


def compact_intent_draft(intent,query):
    spans=literal_query_spans(query);result=copy.deepcopy(intent)
    for need in result['needs']:
        need['span_index']=spans.index(need.pop('query_span'))
    return result


def prepare_need_candidates(need,candidates,services):
    from rag_pensiones.graph_source_policy import _entries_for_need
    from rag_pensiones.graph_context import prefer_current_forecast
    rows=[copy.deepcopy(c)for c in candidates if candidate_matches_need(c,need)
          and('matched_need_ids'not in c or need['need_id']in c['matched_need_ids'])]
    rows.sort(key=lambda c:min((r['rank']for r in c.get('need_retrieval',[])
                                if r['need_id']==need['need_id']),default=10**9))
    forced=set();comparisons=set()
    if need.get('quantity_request','unspecified')in {'level','unspecified'}:
        for group,entry in _entries_for_need(services.source_policy,need,services.catalog,
                candidate_card_ids={c['card_id']for c in rows}):
            forced.add(group['preferred_card_id']);comparisons.update(group['comparison_card_ids'])
    if forced:
        # Only replace members of the authorized comparison family. Other
        # measures in this publication remain candidates for semantic choice.
        rows=[c for c in rows if c['card_id']not in comparisons or c['card_id']in forced]
        for cid in sorted(forced-{c['card_id']for c in rows}):
            source=next(c for c in services.catalog['cards']if c['card_id']==cid)
            row=copy.deepcopy(source);row.update(matched_need_ids=[need['need_id']],need_retrieval=[],
                retrieval=dict(mode='curated_source_registry'))
            rows.append(row)
    return prefer_current_forecast(need,rows,getattr(services,'semantic_context',None))


def compile_capabilities(need,offered,services):
    from rag_pensiones.graph_validation import validate_plan,InvalidPlan,metadata_period_endpoints
    capabilities=[];request=need.get('quantity_request','unspecified')
    for card in prepare_need_candidates(need,offered,services):
        requested=list(dict.fromkeys(need.get('periods',[])));periods=[];operation='lookup'
        if not requested:
            if len(card['periods'])!=1:continue
            periods=list(card['periods'])
        elif set(requested)<=set(card['periods']):periods=requested
        else:
            endpoints={p for label in requested for p in metadata_period_endpoints(label)}
            if card['period_kind']=='interval':
                periods=[label for label in card['periods']if metadata_period_endpoints(label)==endpoints]
            elif card['period_kind']=='year'and len(endpoints)==2 and endpoints<=set(card['periods']):
                periods=sorted(endpoints)
        if not periods:continue
        family=unit_family(card['unit'],card.get('quantity_role',''))
        variation=published_variation(card['unit'],card.get('quantity_role',''))
        if request=='change'and variation and card['period_kind']=='year'and len(periods)==2:continue
        if request=='level'and variation:continue
        level_operand=((need['unit'],family)in {('percentage_points_gdp','percent_gdp'),('percentage_points','percent')})
        if level_operand or (request=='change'and not variation):
            if request!='change'or card['period_kind']!='year'or len(periods)!=2:continue
            if family in {'percent','percent_gdp'}and not level_operand:continue
            operands=[services.catalog['observations'][oid]for oid in card['observation_ids']
                      if services.catalog['observations'][oid]['period']in periods]
            if len(operands)!=2 or any(r.get('calculation_safe')is not True for r in operands):continue
            identity=['book','metric','series','unit','scope','quantity_role']
            if any(operands[0].get(k)!=operands[1].get(k)for k in identity):continue
            if any(not Decimal(r['value']).is_finite()for r in operands):continue
            operation='difference'
        elif card['period_kind']=='interval'and variation and request!='mean':operation='published_change'
        elif card['period_kind']=='interval'and not set(requested)<=set(card['periods']):continue
        selection=dict(need_id=need['need_id'],card_ids=[card['card_id']],periods=periods,
                       operation=operation,reason='Compiled from certified source metadata')
        try:validate_plan(dict(status='ready',selections=[selection]),dict(needs=[need]),[card])
        except InvalidPlan:continue
        identity=dict(profile=PROFILE,need=need,selection=selection,catalog_hash=services.catalog['catalog_hash'])
        capabilities.append(dict(capability_id='cap:'+canonical_hash(identity)[:24],selection=selection,
                                 source_card_id=card['card_id']))
    return capabilities


def material_axes(need,candidates):
    # Vocabulary only: the model decides whether any clarification is necessary.
    return ['edition','unit','period','population','metric','scenario']


def decision_schema(capabilities,axes,*,edition_choices=None,allow_explanation=False):
    def enum(values):return dict(type='string',enum=list(values))
    def obj(properties):return dict(type='object',properties=properties,required=list(properties),additionalProperties=False)
    branches=[]
    if capabilities:branches.append(obj(dict(action=enum(['select']),capability_id=enum([c['capability_id']for c in capabilities]))))
    if axes:
        fields=list(axes)
        question=dict(type='string',minLength=1)
        branches.append(obj(dict(action=enum(['clarify']),field=enum(fields),question=question)))
    if allow_explanation:
        branches.append(obj(dict(action=enum(['explain']),reason=dict(type='string',minLength=1))))
    branches.append(obj(dict(action=enum(['unavailable']),reason=dict(type='string',minLength=1))))
    return dict(type='object',oneOf=branches)


def canonical_decision(need,decision,capabilities):
    action=decision['action']
    if action=='select':
        chosen=next((c for c in capabilities if c['capability_id']==decision['capability_id']),None)
        if chosen is None:raise ValueError('Selected capability was not actually offered')
        return copy.deepcopy(chosen['selection'])
    return dict(need_id=need['need_id'],card_ids=[],periods=[],operation='lookup',
                reason=decision.get('reason')or'Pending material clarification')


CHECKER_APPROVALS=dict(select='consulta_respaldada',clarify='aclaración_necesaria',
                       unavailable='falta_capacidad_acreditada')


def checker_schema(actions):
    properties={}
    for nid,action in actions.items():
        if action not in CHECKER_APPROVALS:raise ValueError('Unsupported checker action')
        properties[nid]=dict(type='object',properties=dict(
            decision=dict(type='string',enum=[CHECKER_APPROVALS[action],'rechazar']),
            reason=dict(type='string',minLength=1)),
            required=['decision','reason'],additionalProperties=False)
    return dict(type='object',properties=properties,required=list(actions),additionalProperties=False)


def decode_checker(raw,actions):
    from rag_pensiones.graph_validation import check_shape
    check_shape(raw,checker_schema(actions))
    return {nid:dict(valid=row['decision']==CHECKER_APPROVALS[actions[nid]],
        issue='none'if row['decision']!='rechazar'else'meaning',reason=row['reason'])for nid,row in raw.items()}


def review_schema(claims,needs):
    def obj(properties):return dict(type='object',properties=properties,required=list(properties),additionalProperties=False)
    claim_ids=[c['claim_id']for c in claims];need_ids=[n['need_id']for n in needs]
    if len(claim_ids)!=len(set(claim_ids))or len(need_ids)!=len(set(need_ids)):raise ValueError('Duplicate review identifier')
    judgments={cid:obj(dict(supported=dict(type='boolean'),reason=dict(type='string')))for cid in claim_ids}
    coverage={}
    for nid in need_ids:
        coverage[nid]=obj(dict(requested_result_present=dict(type='boolean'),requested_scope_supported=dict(type='boolean'),
            claim_ids=dict(type='array',items=dict(type='string'),maxItems=max(1,len(claim_ids)))))
    return obj(dict(judgments=obj(judgments),coverage=obj(coverage)))


def decode_review(raw,claims,needs):
    from rag_pensiones.graph_validation import check_shape
    check_shape(raw,review_schema(claims,needs))
    judgments={cid:dict(claim_id=cid,**copy.deepcopy(row))for cid,row in raw['judgments'].items()}
    owners={claim['claim_id']:claim['need_id']for claim in claims};coverage=[]
    for need in needs:
        nid=need['need_id'];row=copy.deepcopy(raw['coverage'][nid]);witnesses=row['claim_ids'];issues=[]
        positive=row['requested_result_present']and row['requested_scope_supported']
        valid_witnesses=bool(witnesses)and len(set(witnesses))==len(witnesses)and all(
            owners.get(cid)==nid and judgments[cid]['supported']for cid in witnesses)
        answered=positive and valid_witnesses
        if (witnesses or positive)and not valid_witnesses:issues.append('invalid_or_missing_owned_approved_witnesses')
        if answered:row['reason']='covered'
        elif not any(owner==nid for owner in owners.values())or row['requested_scope_supported']:row['reason']='missing_from_answer'
        elif row['requested_result_present']:row['reason']='scope_mismatch'
        else:row['reason']='not_in_evaluated_evidence'
        coverage.append(dict(need_id=nid,**row,answered=answered,validation_issues=issues))
    return dict(judgments=[dict(claim_id=c['claim_id'],**copy.deepcopy(raw['judgments'][c['claim_id']]))for c in claims],
                coverage=coverage)


def grounding_inputs(interpretation,candidates,query,context=None):
    """Offer only metadata identities and question/history-literal source years."""
    from rag_pensiones.graph_validation import metadata_period_endpoints
    from rag_pensiones.graph_context import source_measure_view
    literals=set(literal_year_choices(query,context));inputs={}
    for need in interpretation['needs']:
        if need['kind']!='data':continue
        targets={p for label in need['periods']for p in metadata_period_endpoints(label)}
        rows=[c for c in candidates if candidate_matches_need(c,need,metadata_discovery=True)
              and('matched_need_ids'not in c or need['need_id']in c['matched_need_ids'])]
        if not rows:continue
        sources=[]
        for card in rows:
            years=sorted(literals&targets&{p for label in card['periods']for p in metadata_period_endpoints(label)})
            source=source_measure_view(card,card.get('semantic_context'),query=query,context=context)
            semantic=source.pop('semantic_context')
            remainder=copy.deepcopy(semantic)if semantic is not None else{}
            if remainder.get('semantic_label')==source['measure']:remainder.pop('semantic_label',None)
            if remainder.get('series_role')==source['series_role']:remainder.pop('series_role',None)
            for key in ['source_scope','scope_qualifier','forecast_role','forecast_vintage']:
                source[key]=remainder.pop(key,None)
            source['semantic_context']=remainder or None
            source.setdefault('published_measure_name',None)  # Uniform codec; no unused alias text.
            source['period_kind']=card['period_kind']
            # Uniform fields allow the lossless candidate-table codec.
            source['container_title']=card.get('title')if card.get('semantic_context')is None else None
            source['years']=years;sources.append(source)
        labels=sorted({c['series']for c in rows if isinstance(c.get('series'),str)})
        exact=next((label for label in labels if need.get('series')is not None
                    and label.casefold()==need['series'].casefold()),None)
        inputs[need['need_id']]=dict(before=dict(series=need.get('series'),periods=copy.deepcopy(need['periods'])),
            exact_series=exact,series_labels=labels,years=sorted({year for s in sources for year in s['years']}),sources=sources)
    return inputs


def unresolved_scenario_token(offered):
    token='[sin vínculo acreditado]'
    while token in offered['series_labels']:token+='?'
    return token


def grounding_schema(inputs):
    def obj(properties):return dict(type='object',properties=properties,required=list(properties),additionalProperties=False)
    properties={}
    for nid,offered in inputs.items():
        exact=offered['exact_series']
        scenario=(dict(type='null')if offered['before']['series']is None else
                  dict(type='string',enum=[exact])if exact is not None else
                  dict(type=['string','null'],enum=[*offered['series_labels'],None,unresolved_scenario_token(offered)]))
        years=offered['years']
        target=(dict(type='null')if offered['before']['periods']and not years else
                dict(type=['array','null'],items=dict(type='string',enum=years)if years else dict(type='string'),
                     minItems=1 if offered['before']['periods']else 0,maxItems=len(years)))
        properties[nid]=obj(dict(scenario=scenario,
            target_years=target))
    return obj(properties)


def apply_grounding(interpretation,inputs,raw):
    from rag_pensiones.graph_validation import check_shape,InvalidPlan,metadata_period_endpoints
    check_shape(raw,grounding_schema(inputs));result=copy.deepcopy(interpretation);records={}
    for need in result['needs']:
        nid=need['need_id']
        if nid not in inputs:continue
        choice=raw[nid];label=choice['scenario'];years=choice['target_years']
        action=('no_restriction'if label is None else'unresolved'if label==unresolved_scenario_token(inputs[nid])else'bind')
        scenario=dict(action=action,label=label if action=='bind'else None)
        if years is not None and len(set(years))!=len(years):raise InvalidPlan('Duplicate grounded horizon')
        if scenario['action']=='bind':need['series']=scenario['label']
        elif scenario['action']=='no_restriction'and inputs[nid]['before']['series'] is not None:
            scenario=dict(action='unresolved',label=None)
        # unresolved deliberately preserves the original hint, never a fabricated ID.
        raw_years={p for label in need['periods']for p in metadata_period_endpoints(label)}
        missing=(raw_years-set(inputs[nid]['years']))-{need.get('book')}
        horizon_status=('unresolved'if missing or years is None or(not years and need['periods'])else
                        'not_requested'if not years else'bound')
        if horizon_status!='unresolved':
            need['periods']=(['-'.join(sorted(years))]if need.get('quantity_request')in {'mean','contribution'}
                             and len(years)==2 else copy.deepcopy(years))
        records[nid]=dict(before=copy.deepcopy(inputs[nid]['before']),
            after=dict(series=need['series'],periods=copy.deepcopy(need['periods'])),
            decision=dict(scenario=scenario,target_years=copy.deepcopy(years)),horizon_status=horizon_status,
            offered=copy.deepcopy(inputs[nid]))
    return result,records
