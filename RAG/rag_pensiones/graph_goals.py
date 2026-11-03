"""Interpreted result goals; original question/context remain authoritative."""
import copy
import json
from rag_pensiones.graph_planning import literal_year_choices,named_intent_schema,decode_named_intent
from rag_pensiones.graph_context import _user_history
from rag_pensiones.graph_validation import check_shape,InvalidPlan


def data_period_context(context):
    """An answer to an edition clarification is not provenance for a data year."""
    if isinstance(context,list):return [data_period_context(item)for item in context]
    if not isinstance(context,dict):return context
    if context.get('role')=='assistant':return {}
    result=copy.deepcopy(context)
    field=context.get('clarification_field')
    if field=='edition':result.pop('clarification_answer',None)
    result.pop('clarification_question',None)
    for key in ['messages','history','previous_context','context']:
        if key in result:result[key]=data_period_context(result[key])
    return result


def publication_policy(context):
    """Explicit application preference, never inferred from assistant prose."""
    if not isinstance(context,dict)or context.get('role')=='assistant':return None
    if context.get('publication_policy')=='latest_supported':return 'latest_supported'
    for key in ['previous_context','context']:
        policy=publication_policy(context.get(key))
        if policy:return policy
    return None


def explain_alternatives(context):
    if not isinstance(context,dict)or context.get('role')=='assistant':return False
    return context.get('ambiguity_policy')=='explain_supported_alternatives'or any(
        explain_alternatives(context.get(key))for key in ['previous_context','context'])


def explanation_need(need,source_labels=()):
    result=copy.deepcopy(need)
    result.update(need_id=need['need_id']+'_explanation',kind='text',
        quantity_request='unspecified',
        metric='Explica la evidencia pertinente y sus límites para: '+need['metric']+'. Conserva ámbito y periodo pedidos. Distingue conceptos y alternativas sólo si las fuentes los respaldan. La cifra solicitada sigue pendiente; no la sustituyas por otra magnitud.',
        query_span=need['query_span'])
    return result


def text_stage_question(query,needs):
    return query


def source_publication_years(services):
    books={s['book']for s in services.documents.get('sources',[])if s.get('book')is not None}
    if services.scope!='legacy_2025_reference':books.update(services.catalog.get('sources',{}))
    return sorted(books)


def nominal_publication_years(query,context,publication_years):
    user_years=set(literal_year_choices(query,{'history':_user_history(context)}))
    return sorted(set(publication_years)&user_years)


def goal_schema(query,context=None,*,publication_years=()):
    fields=dict(
        question_fragment=dict(type='string',minLength=1,description='Descripción interpretada de ESTE resultado en orden de aparición; puede parafrasear, no es cita del usuario. Conserva sus condiciones compartidas aplicables, no las exclusivas de otra petición.'),
        requested_measure=dict(type='string',minLength=1,description='Sólo resultado o relación de este fragmento, no los de otras cláusulas.'),
        answer_kind=dict(type='string',enum=['quantity','reference','explanation'],description='quantity=valores/cálculos/comparaciones numéricas de magnitudes; reference=referencia histórica, definición o clasificación del documento; explanation=causa o relación cualitativa, sin pedir valores numéricos.'),
        report_publication_year=dict(type=['string','null'],enum=[*nominal_publication_years(query,context,publication_years),None],description='Edición nominal expresamente solicitada, sólo entre publicaciones reales y años presentes en texto USER; nunca año del dato. null sin edición nominal, también ante preferencia relativa cuya cobertura aún se desconoce.'))
    item=dict(type='object',properties=fields,required=list(fields),additionalProperties=False)
    properties=dict(answer_requests=dict(type='array',items=item,minItems=1,maxItems=4))
    return dict(type='object',properties=properties,required=list(properties),additionalProperties=False)


def decode_goals(raw,query,context=None,*,publication_years=()):
    check_shape(raw,goal_schema(query,context,publication_years=publication_years))
    goals=[];identities={};mapping={}
    for index,item in enumerate(raw['answer_requests'],1):
        if not item['question_fragment'].strip():raise InvalidPlan('Nonempty goal description required')
        nid='n'+str(index);identity=json.dumps(item,sort_keys=True,ensure_ascii=False,allow_nan=False)
        if identity in identities:mapping[nid]=identities[identity];continue
        identities[identity]=nid;mapping[nid]=nid
        goals.append(dict(need_id=nid,**copy.deepcopy(item)))
    return goals,dict(derived_result_count=len(raw['answer_requests']),raw_count=len(raw['answer_requests']),
                      retained_count=len(goals),original_to_retained=mapping)


def quantitative_goal_schema(query,context,goal,*,publication_years=()):
    if goal['report_publication_year']not in [*nominal_publication_years(query,context,publication_years),None]:
        raise InvalidPlan('Goal publication lacks original USER/source provenance')
    schema=named_intent_schema(query,context,1)
    fields=schema['properties']['answer_requests']['items']['properties']
    fields['question_fragment']=dict(type='string',enum=[goal['question_fragment']])
    fields['answer_kind']=dict(type='string',enum=['data'])
    fields['report_publication_year']=dict(type=['string','null'],enum=[goal['report_publication_year']])
    years=literal_year_choices(query,{'history':_user_history(data_period_context(context))})
    fields['target_data_periods']['items']=dict(type='string',enum=years)if years else dict(type='string')
    fields['target_data_periods']['maxItems']=12 if years else 0
    fields['population_type']['description']='Ámbito del resultado cuantitativo FOCUS. individual sólo si pide calcular para una persona; clasificar agregado frente a garantía personal no es un filtro personal.'
    return schema


def extract_goal_needs(raw,query,context,scope,chat,*,publication_years=()):
    """chat is called only for production quantity goals; raw inputs stay intact."""
    goals,audit=decode_goals(raw,query,context,publication_years=publication_years);needs=[];replies={};hint_notes=[]
    for goal in goals:
        nid=goal['need_id']
        if goal['answer_kind']=='quantity'and scope!='legacy_2025_reference':
            schema=quantitative_goal_schema(query,context,goal,publication_years=publication_years)
            reply=chat(dict(question=query,context=context,requested_result_count=1,
                            focus={k:goal[k]for k in ['question_fragment','requested_measure']}),schema)
            replies[nid]=copy.deepcopy(reply)
            decoded=decode_named_intent(reply,query,context,count=1,schema=schema)
            need=decoded['needs'][0];need['need_id']=nid
            for note in decoded.get('discarded_scenario_hints',[]):
                hint_notes.append(dict(note,need_id=nid))
        else:
            need=dict(need_id=nid,query_span=goal['question_fragment'],kind='text',metric=goal['requested_measure'],
                      population='unspecified',scope_terms=[],book=goal['report_publication_year'],series=None,
                      source_block=None,unit='unspecified',periods=[],quantity_request='unspecified')
        needs.append(need)
    topics=[' | '.join(str(v)for v in [n['metric'],n['book'],n.get('series'),n.get('source_block'),n.get('unit'),n.get('population'),*n['periods'],*n['scope_terms']]if v is not None and str(v)and v!='unspecified')for n in needs]
    result=dict(needs=needs,resolved_query='\n'.join([query,*topics]),request_coalescing=audit)
    if isinstance(context,dict)and isinstance(context.get('clarification_answer'),str):
        result['resolved_query']+='\nAclaración de USER: '+context['clarification_answer']
    if hint_notes:result['discarded_scenario_hints']=hint_notes
    return result,dict(by_goal=replies,source_scope=scope)

