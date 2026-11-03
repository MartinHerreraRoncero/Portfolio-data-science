"""Display-only rounding and a constrained narrator over already published claims."""
import copy
from decimal import Decimal
import json
from pathlib import Path
import re
import duckdb
from experiments import source_recovery_policy as policy

UNITS={'percent_gdp':'% del PIB','percentage_points_gdp':'puntos del PIB','percent':'%',
       'percentage_points':'puntos porcentuales','euros':'euros','persons':'personas','pensions_count':'pensiones'}


def rounded_derived(value):
    with duckdb.connect(':memory:')as connection:
        result=connection.execute('SELECT round(CAST(? AS DECIMAL(38,12)),2)',[str(value)]).fetchone()[0]
    return format(result,'.2f').replace('.',',')


def rounded_observation(database,observation_id,expected_value):
    with duckdb.connect(str(database),read_only=True)as connection:
        rows=connection.execute('SELECT value, ROUND(value,2) AS display_value FROM policy_public_observations WHERE observation_id=?',[observation_id]).fetchall()
    if len(rows)!=1 or Decimal(str(rows[0][0]))!=Decimal(str(expected_value)):
        raise ValueError('Display SQL differs from validated source value')
    return format(Decimal(str(rows[0][1])),'.2f').replace('.',',')


def displayed_fact(fact,database):
    if fact.get('derived'):return rounded_derived(fact['value'])
    actual=policy.query_observation(database,observation_id=fact['observation_id'])
    if Decimal(str(actual['value']))!=Decimal(str(fact['value'])):
        raise ValueError('Display source value differs from validated value')
    for key in ['book','unit','source_sheet','source_cell']:
        if key in fact and actual.get(key)!=fact[key]:raise ValueError('Display source differs: '+key)
    return rounded_observation(database,fact['observation_id'],fact['value'])


def sources_paragraph(claims):
    sources=[]
    for claim in claims:
        for evidence in claim.get('evidence',[]):
            name=evidence.get('source_file')or Path(evidence.get('source_path','Fuente publicada')).name
            parts=[name]
            if evidence.get('page_number')is not None:parts.append('pág. '+str(evidence['page_number']))
            if evidence.get('source_sheet'):parts.append('hoja '+evidence['source_sheet'])
            if evidence.get('source_cell'):parts.append('celda '+evidence['source_cell'])
            source=' — '.join(parts)
            if source not in sources:sources.append(source)
    return '; '.join(sources)


def fill_narration(template,atoms):
    if not isinstance(template,str):raise ValueError('Narrator must return text')
    remainder=template
    for marker in atoms:
        if template.count(marker)!=1:raise ValueError('Narrator changed or omitted a protected fact')
        remainder=remainder.replace(marker,'')
    if any(char.isdigit()for char in remainder):raise ValueError('Narrator invented a number')
    if '{{'in remainder or '}}'in remainder:raise ValueError('Narrator invented a marker')
    pattern='|'.join(re.escape(marker)for marker in atoms)
    return re.sub(pattern,lambda match:atoms[match.group(0)],template).strip()


def narrator_client():
    from rag_pensiones.graph_models import GraphStageClient
    return GraphStageClient()


def present_report(report,*,database,client=None):
    publication=report['publication'];status=publication['status']
    if status in {'error','clarification'}or not publication.get('claims'):
        if status=='insufficient_evidence':
            retrieved=report.get('retrieval',{}).get('documents',[])
            sources=list(dict.fromkeys(str(p.get('source_file','Documento'))+' — pág. '+str(p.get('page_number','?'))for p in retrieved))
            reasons={part.get('reason_code')for part in publication.get('missing_parts',[])}
            if 'textual_semantic_rejected'in reasons:
                answer='He recuperado documentos, pero la respuesta generada no superó la comprobación de sus citas o de lo que estas respaldan.'if retrieved else'La respuesta generada no superó la comprobación de evidencia.'
            else:
                answer='No he podido respaldar una respuesta a tu pregunta con las evidencias evaluadas.'
            answer+=' Esto no demuestra que la información no exista en las fuentes.'
            if sources:answer+=' Los documentos recuperados están disponibles para revisión; su recuperación por sí sola no valida una respuesta.'
            return dict(answer=answer,sources='Documentos recuperados: '+'; '.join(sources)if sources else'',calls=[],narration='passthrough',related_passages=[])
        return dict(answer=publication.get('reason',report.get('formatted_answer','')),sources='',calls=[],narration='passthrough')
    from rag_pensiones.graph_validation import numeric_claim
    from rag_pensiones.graph_units import unit_family
    claims=publication['claims'];facts=[f for f in report.get('numeric_facts',[])if not f.get('supporting_only')]
    numeric={numeric_claim(f)['text']:f for f in facts};atoms={};descriptions=[];fallback=[]
    for index,claim in enumerate(claims):
        text=claim['text'];fact=numeric.get(text)
        if fact is not None:
            value=displayed_fact(fact,database);unit=UNITS.get(unit_family(fact['unit']),UNITS.get(fact['unit'],fact['unit']))
            series=' ('+str(fact['series'])+')'if fact.get('series')else''
            qualifier='La diferencia calculada de 'if fact.get('derived')else''
            text=f"{qualifier}{fact['metric']}{series} en {fact['period']}: {value} {unit}."
            letter=chr(65+index)
            fields={'metric':fact['metric'],'period':fact['period'],'value_unit':value+' '+unit}
            if fact.get('series'):fields['series']=str(fact['series'])
            if fact.get('derived'):fields['operation']='diferencia calculada'
            description={}
            for field,content in fields.items():
                marker='{{'+field.upper()+'_'+letter+'}}';atoms[marker]=content;description[field]=marker
            descriptions.append(description)
            for key in ['warning','source_policy_note']:
                if fact.get(key):text+=' '+str(fact[key])
        else:
            marker='{{TEXT_'+chr(65+index)+'}}';atoms[marker]=text;descriptions.append(dict(published_text=marker))
        fallback.append(text)
    answer=' '.join(fallback);client=client or narrator_client();error=None
    schema=dict(type='object',properties=dict(text=dict(type='string')),required=['text'],additionalProperties=False)
    messages=[dict(role='system',content='Redacta una frase breve y natural en español con los hechos YA publicados. Usa cada marcador exactamente una vez dentro de tu frase. Los marcadores fijan magnitud, serie, periodo y valor con unidad: no los cambies ni escribas otras cifras o unidades. No añadas fuentes, causas, comparaciones ni hechos. Las fuentes se mostrarán aparte. Responde JSON con text.'),
              dict(role='user',content=json.dumps(dict(published_facts=descriptions,protected_fields=atoms),ensure_ascii=False,separators=(',',':')))]
    start=len(client.calls)
    structured=bool(claims and all(c['need_id'].endswith('_explanation')and c.get('definition_variant')for c in claims))
    if not structured:
        try:answer=fill_narration(client.chat(messages,schema,'presentation')['text'],atoms)
        except Exception as exc:error=type(exc).__name__+': '+str(exc)
    # A malformed narrator reply changes no fact: use the deterministic sentences.
    if claims and all(c['need_id'].endswith('_explanation')and c.get('definition_variant')for c in claims):
        answer='\n'.join(str(i)+'. Interpretación analítica propuesta: '+c['text']for i,c in enumerate(claims,1))
    missing=' '.join(item.get('message','')for item in publication.get('missing_parts',[]))
    if missing:answer+=' '+missing
    if error is None:
        notices=' '.join(str(f.get(key))for f in facts for key in ['warning','source_policy_note']if f.get(key))
        if notices:answer+=' '+notices
    return dict(answer=answer,sources=sources_paragraph(claims),related_passages=[],calls=copy.deepcopy(client.calls[start:]),
                narration='structured_interpretations'if structured else'deterministic_fallback'if error else'protected_markers',narration_error=error)
