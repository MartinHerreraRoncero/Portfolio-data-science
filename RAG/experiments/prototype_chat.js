'use strict';

const GRAPH_NODES = [
  ['interpret','Entender la pregunta',260,40],['intent_audit','Confirmar lo que pides',260,90],
  ['embed','Convertir pregunta a vectores',260,140],['catalog','Buscar y diversificar datos',155,200],['documents','Buscar en documentos',365,200],
  ['ground','Relacionar datos y pregunta',260,260],['plan','Preparar las consultas',260,310],
  ['validate','Validar plan y consultar datos',260,360],['clarify','Pedirte una aclaración',440,310],
  ['execute','Consultar base de datos',260,420],['generate','Redactar la respuesta',260,470],
  ['review','Contrastar con las fuentes',260,520],['prepare_repair','Preparar una corrección',440,470],['publish','Entregar la respuesta',260,580]
];
const FLOW_GROUPS = GRAPH_NODES.map(([id,title,x,y])=>({id,title,detail:'',nodes:[id],x,y}));
const OUTER_STAGES = [
  ['context','Recuperar conversación y preferencias','before'],
  ['load','Cargar y comprobar fuentes e índices','before'],
  ['report','Guardar informe del grafo','after'],
  ['presentation','Preparar texto y citas para el chat','after'],
  ['persist','Guardar memoria y trazas','after']
].map(([id,title,location])=>({id,title,location,detail:'',nodes:[id]}));
const GRAPH_EDGES = [
  ['__start__','interpret',''],['interpret','intent_audit','continuar'],['interpret','clarify','referencia ambigua'],
  ['intent_audit','embed',''],['embed','catalog','paralelo'],['embed','documents','paralelo'],
  ['catalog','ground','unión de ambas ramas'],['documents','ground','unión de ambas ramas'],
  ['ground','plan',''],['plan','validate',''],['validate','execute','plan válido'],
  ['validate','clarify','requiere aclaración'],['validate','plan','inválido · reparaciones < 2'],
  ['validate','publish','inválido · límite de reparaciones'],['clarify','interpret','reanudar con aclaración'],
  ['execute','generate',''],['generate','review',''],['review','publish','respuesta revisada'],
  ['review','prepare_repair','texto pendiente · reparaciones < 1'],['prepare_repair','generate','reintentar'],
  ['publish','__end__',''],
  ...['interpret','intent_audit','embed','ground','plan','validate','clarify','execute','generate','review'].map(node=>[node,'publish','error'])
];
const NODE_LABELS = {context:'Preparando el contexto',report:'Guardando el informe',persist:'Guardando memoria y trazas',load:'Preparando los recursos locales',interpret:'Interpretando la pregunta',intent_audit:'Revisando la intención',embed:'Calculando embeddings',catalog:'Consultando el catálogo',documents:'Recuperando documentos',ground:'Vinculando evidencias',plan:'Planificando la consulta',validate:'Validando el plan',clarify:'Esperando tu aclaración',execute:'Ejecutando consultas',generate:'Redactando la respuesta',review:'Contrastando la respuesta',prepare_repair:'Preparando una revisión',publish:'Publicando el resultado',presentation:'Preparando la presentación'};
const STATUS_LABELS = {answered:'Respuesta completada',partial:'Respuesta parcial',insufficient_evidence:'Evidencia insuficiente',clarification:'Aclaración necesaria',error:'Consulta interrumpida'};

function createFlow(){return {nodes:{},cursor:0,terminal:null,observed:false,complete:false,lost:false};}
function applyEvent(flow,event){
  if(!Object.hasOwn(NODE_LABELS,event.node) || !['start','end','error','waiting'].includes(event.state))return;
  if(event.sequence && event.sequence<=flow.cursor)return;
  if(event.sequence)flow.cursor=event.sequence;
  const current=flow.nodes[event.node] || {count:0,state:'pending'};
  flow.nodes[event.node]={count:current.count+(event.state==='start'?1:0),state:({start:'active',end:'done',error:'error',waiting:'waiting'})[event.state]};
  flow.observed=true;
}
function groupState(flow,group){
  const values=group.nodes.map(node=>flow.nodes[node]).filter(Boolean);
  if(values.some(v=>v.state==='error'))return 'error';
  if(values.some(v=>v.state==='active'))return flow.terminal?(flow.terminal==='error'?'error':'unknown'):'active';
  if(values.some(v=>v.state==='waiting'))return 'waiting';
  if(values.length)return 'done';
  return flow.terminal?(flow.complete?'skipped':'unknown'):'pending';
}
function ingestProgress(flow,progress){
  for(const event of progress.events||[])applyEvent(flow,event);
  flow.lost=flow.lost||Boolean(progress.truncated);
  flow.complete=(flow.complete||Boolean(progress.done))&&!flow.lost;
}
function tracePath(value){return typeof value==='string' && /^\/report\/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/.test(value)?value:null;}
function collectCitations(report){
  const result=[],seen=new Map();
  for(const claim of report?.publication?.claims || []){
    for(const evidence of claim.evidence || []){
      const key=JSON.stringify([evidence.source_id,evidence.source_file,evidence.page_number,evidence.source_sheet,evidence.source_cell,evidence.quote]);
      if(seen.has(key)){seen.get(key).claims.push(claim.text);continue;}
      const citation={...evidence,claims:[claim.text],number:result.length+1};result.push(citation);seen.set(key,citation);
    }
  }
  return result;
}
function citationLabel(citation){
  return [citation.source_file || 'Fuente documental',citation.page_number!=null?'pág. '+citation.page_number:'',citation.source_sheet?'hoja '+citation.source_sheet:'',citation.source_cell?'celda '+citation.source_cell:''].filter(Boolean).join(' · ');
}
if(typeof module!=='undefined')module.exports={FLOW_GROUPS,GRAPH_NODES,GRAPH_EDGES,OUTER_STAGES,createFlow,applyEvent,ingestProgress,groupState,tracePath,collectCitations,citationLabel};

if(typeof document!=='undefined'){
const byId=id=>document.getElementById(id);
let session=null,conversation=null,busy=false,flow=createFlow(),pollController=null;
const groupElements=new Map();
const element=(tag,className,text)=>{const node=document.createElement(tag);if(className)node.className=className;if(text!=null)node.textContent=text;return node;};
function diagram(){
  const root=byId('flow-diagram');root.replaceChildren();groupElements.clear();root.className='flow-diagram actual-graph';
  const ns='http://www.w3.org/2000/svg',svg=document.createElementNS(ns,'svg');
  svg.setAttribute('viewBox','0 0 520 630');svg.setAttribute('class','graph-wires');svg.setAttribute('aria-hidden','true');
  const defs=document.createElementNS(ns,'defs'),marker=document.createElementNS(ns,'marker');
  for(const [key,value] of Object.entries({id:'graph-arrow',viewBox:'0 0 10 10',refX:'9',refY:'5',markerWidth:'5',markerHeight:'5',orient:'auto-start-reverse'}))marker.setAttribute(key,value);
  const tip=document.createElementNS(ns,'path');tip.setAttribute('d','M 0 0 L 10 5 L 0 10 z');tip.setAttribute('fill','#8b9eaa');marker.append(tip);defs.append(marker);svg.append(defs);
  const positions=Object.fromEntries(GRAPH_NODES.map(([id,,x,y])=>[id,{x,y}]));
  positions.__start__={x:260,y:0};positions.__end__={x:260,y:630};
  for(const [from,to,condition] of GRAPH_EDGES){
    const a=positions[from],b=positions[to],wire=document.createElementNS(ns,'path');
    let d;
    if(condition==='error'&&from==='clarify')d='M 506 310 H 515 V 605 H 35 V 580 H 194';
    else if(condition==='error'||condition.includes('límite'))d=`M ${a.x-66} ${a.y} H 35 V ${b.y} H ${b.x-66}`;
    else if(to==='clarify')d=`M ${a.x+66} ${a.y} H 440 V ${b.y-18}`;
    else if(from==='clarify')d=`M 440 292 V 12 H 260 V 22`;
    else if(from==='validate'&&to==='plan')d='M 194 360 H 105 V 310 H 194';
    else if(to==='prepare_repair')d='M 326 520 H 440 V 488';
    else if(from==='prepare_repair')d='M 374 470 H 326';
    else d=`M ${a.x} ${a.y+18} V ${(a.y+b.y)/2} H ${b.x} V ${b.y-18}`;
    wire.setAttribute('d',d);wire.setAttribute('class',condition==='error'?'graph-error-route':'graph-route');wire.setAttribute('marker-end','url(#graph-arrow)');
    const title=document.createElementNS(ns,'title');title.textContent=from+' → '+to+(condition?' · '+condition:'');wire.append(title);svg.append(wire);
  }
  root.append(svg);
  for(const group of FLOW_GROUPS){
    const node=element('div','flow-node pending');node.dataset.group=group.id;node.setAttribute('title',group.id==='catalog'?'Catálogo: hasta 50 candidatos por necesidad; seleccionar hasta 10 representantes diversos y mostrar sus filas elegibles al modelo. Reranker desactivado.':'Nodo LangGraph: '+group.id);
    node.setAttribute('style',`left:${(group.x-66)/520*100}%;top:${(group.y-18)/630*100}%;width:${132/520*100}%;`);
    const copy=element('span','node-copy');copy.append(element('span','node-title',group.title),element('span','node-detail',group.detail));
    node.append(copy,element('span','node-indicator'));groupElements.set(group.id,node);root.append(node);
  }
  for(const [label,x,y] of [['Inicio',260,0],['Fin',260,625],['error → salida',48,560],['reintentar',100,337],['aclarar ↶',442,240],['reparar ↶',440,548]]){
    const text=element('span','graph-annotation',label);text.setAttribute('style',`left:${x/520*100}%;top:${y/630*100}%;`);root.append(text);
  }
  const routes=element('details','graph-routes'),summary=element('summary','','Ver condiciones y conexiones');routes.append(summary);
  for(const [from,to,condition] of GRAPH_EDGES)routes.append(element('p','',from+' → '+to+(condition?' · '+condition:'')));
  byId('graph-connections')?.remove();routes.id='graph-connections';root.parentNode?.append(routes);
}
function renderFlow(){
  let observed=0;
  for(const group of FLOW_GROUPS){
    const node=groupElements.get(group.id),state=groupState(flow,group);node.className='flow-node '+state;
    const active=group.nodes.find(name=>flow.nodes[name]?.state==='active');
    const count=Math.max(0,...group.nodes.map(name=>flow.nodes[name]?.count||0));
    const label=state==='skipped'?'No ejecutado':state==='unknown'?'Sin observación disponible':state==='error'?'Etapa interrumpida':state==='waiting'?'Esperando tu aclaración':state==='active'?'En curso':state==='done'?'Completado'+(count>1?' · '+count+' pasadas':''):group.detail;
    node.querySelector('.node-detail').textContent=label;node.setAttribute('aria-label',group.title+': '+label);
    if(FLOW_GROUPS.includes(group)&&group.nodes.some(name=>flow.nodes[name]))observed++;
  }
  byId('flow-counter').textContent=observed+' de '+FLOW_GROUPS.length+' nodos observados';
}
function status(text,error=false){byId('status').textContent=text;byId('status').className='query-status'+(error?' is-error':'');}
function state(value){busy=value;for(const id of ['ask','new','close'])byId(id).disabled=value;byId('query').disabled=value;byId('composer').setAttribute('aria-busy',String(value));}
function scrollChat(){byId('chat').scrollTop=byId('chat').scrollHeight;}
function showDetails(data){
  byId('detail-empty').hidden=true;
  const count=collectCitations(data.report).length,calls=(data.report?.stage_calls||[]).length;
  byId('detail-label').textContent=STATUS_LABELS[data.status]||data.status||'Error';
  const meta=element('div','detail-meta');meta.append(element('span','',STATUS_LABELS[data.status]||data.status||'Error'),element('span','',count+' citas'),element('span','',calls+' llamadas registradas'));
  byId('detail-meta').replaceChildren(meta);const path=tracePath(data.trace_url);
  byId('detail-trace').hidden=!path;if(path)byId('detail-trace').href=path;
  byId('detail-json').hidden=false;byId('detail-json').textContent=JSON.stringify(data.report||{error:data.error},null,2);
}
function message(role,text,data){
  byId('welcome')?.remove();byId('chat').querySelector('.empty-chat')?.remove();
  const node=element('article','message '+role),label=element('div','message-label');
  if(role!=='user')label.append(element('span','message-avatar','R'));
  label.append(element('span','',role==='user'?'Tú':'Asistente documental'));
  if(data?.status)label.append(element('span','answer-badge',STATUS_LABELS[data.status]||data.status));
  node.append(label,element('div','message-body',text));
  if(data){
    const citations=collectCitations(data.report);
    if(citations.length){
      const list=element('div','citation-list');list.append(element('p','citation-heading','EVIDENCIAS DE LA RESPUESTA'));
      for(const citation of citations){
        const detail=element('details','citation'),summary=element('summary');
        summary.append(element('span','citation-number','['+citation.number+']'),element('span','citation-label',citationLabel(citation)));
        const content=element('div','citation-content');
        if(citation.quote){content.append(element('p','',citation.source_kind==='pdf_text'?'Extracto citado':'Registro de evidencia'),element('blockquote','',citation.quote));}
        for(const claim of citation.claims)content.append(element('p','','Afirmación asociada: '+claim));
        if(!citation.quote)content.append(element('p','','Consulta el localizador de la fuente en la traza completa.'));
        detail.append(summary,content);list.append(detail);
      }
      node.append(list);
    }else if(data.sources)node.append(element('p','sources-fallback',data.sources));
    const actions=element('div','message-actions'),path=tracePath(data.trace_url);
    if(path){const link=element('a','','Traza y fuentes ↗');link.href=path;link.target='_blank';link.rel='noopener';actions.append(link);}
    const detailButton=element('button','text-button','Ver detalle');detailButton.type='button';detailButton.onclick=()=>{showDetails(data);byId('query-details').open=true;};actions.append(detailButton);node.append(actions);
  }
  byId('chat').append(node);scrollChat();return node;
}
function loading(){
  const node=message('assistant',''),label=element('span','loading-label');label.append(element('i'),element('span','','Conectando tu pregunta con las fuentes…'));node.querySelector('.message-body').append(label);return node;
}
async function pollProgress(requestId,signal){
  while(!signal.aborted){
    try{
      const response=await fetch('/progress/'+requestId+'?after='+flow.cursor,{signal});
      if(response.ok){const data=await response.json();ingestProgress(flow,data);renderFlow();
        const active=Object.keys(flow.nodes).filter(node=>flow.nodes[node].state==='active');
        if(active.length)status(active.map(node=>NODE_LABELS[node]).join(' · ')+'…');
        if(data.truncated)status('Parte del historial de ejecución ya no está disponible.');
        if(data.done)return;
      }else if(response.status!==404){byId('live-badge').textContent='Sin telemetría';return;}
    }catch(error){if(signal.aborted)return;byId('live-badge').textContent='Reconectando…';}
    await new Promise(resolve=>{const timer=setTimeout(done,450);function done(){clearTimeout(timer);signal.removeEventListener('abort',done);resolve();}signal.addEventListener('abort',done,{once:true});});
  }
}
async function send(){
  if(busy)return;const text=byId('query').value.trim();if(!text)return;
  const pending=session,body={query:pending?pending.query:text,request_id:crypto.randomUUID()};
  if(conversation)body.conversation_id=conversation;if(pending){body.thread_id=pending.thread_id;body.resume=text;}
  message('user',text);byId('query').value='';state(true);flow=createFlow();renderFlow();
  byId('live-badge').className='live-badge running';byId('live-badge').textContent='En directo';
  status('Preparando la consulta…');const waiting=loading();pollController=new AbortController();const polling=pollProgress(body.request_id,pollController.signal);
  try{
    const response=await fetch('/query',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}),data=await response.json();
    if(!response.ok)throw new Error(data.error||'Error HTTP '+response.status);
    // Capture fast stages that completed between polling intervals.
    const finalRead=new AbortController(),timer=setTimeout(()=>finalRead.abort(),1500);
    try{const result=await fetch('/progress/'+body.request_id+'?after='+flow.cursor,{signal:finalRead.signal});if(result.ok)ingestProgress(flow,await result.json());}catch(error){/* A telemetry failure cannot discard the response. */}finally{clearTimeout(timer);}
    pollController.abort();await polling;
    waiting.remove();conversation=data.conversation_id;session=data.status==='clarification'?{query:body.query,thread_id:data.session_id}:null;
    message(data.status==='error'?'error':'assistant',data.formatted_answer||data.error||'No se obtuvo respuesta.',data);showDetails(data);
    flow.terminal=data.status;renderFlow();byId('clarify').hidden=!session;
    byId('query').placeholder=session?'Escribe la aclaración para continuar…':'Escribe tu pregunta sobre las fuentes…';
    byId('live-badge').textContent=data.status==='clarification'?'Tu turno':data.status==='error'?'Interrumpido':'Finalizado';
    status(STATUS_LABELS[data.status]||data.status,data.status==='error');
  }catch(error){
    pollController.abort();await polling;waiting.remove();flow.terminal='error';renderFlow();
    message('error','No se pudo completar la consulta. '+error.message);byId('query').value=text;
    byId('live-badge').textContent='Interrumpido';status('La consulta no se ha completado. Puedes volver a enviarla.',true);
    showDetails({status:'error',error:error.message});
  }finally{pollController.abort();byId('live-badge').className='live-badge';state(false);byId('query').focus();}
}
async function reset(close){
  if(busy)return;state(true);status(close?'Liberando los modelos locales…':'Preparando una conversación nueva…');
  try{
    const response=await fetch(close?'/end-session':'/new-conversation',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({conversation_id:conversation})}),data=await response.json();
    if(!response.ok)throw new Error(data.error||'No se pudo cerrar la sesión');
    conversation=data.conversation_id||null;session=null;byId('chat').replaceChildren();byId('clarify').hidden=true;byId('query').value='';
    byId('query').placeholder='Escribe tu pregunta sobre las fuentes…';flow=createFlow();renderFlow();
    byId('live-badge').textContent='En espera';byId('query-details').open=false;byId('detail-empty').hidden=false;byId('detail-meta').replaceChildren();byId('detail-trace').hidden=true;byId('detail-json').hidden=true;byId('detail-label').textContent='Sin consulta';
    byId('chat').append(element('p','empty-chat',close?'Sesión cerrada. Puedes iniciar otra conversación cuando quieras.':'Conversación nueva. Escribe tu primera pregunta.'));
    status(close?'Sesión cerrada. Modelos liberados.':'Nueva conversación. Memoria vacía.');
  }catch(error){status('Error: '+error.message,true);}finally{state(false);byId('query').focus();}
}
byId('composer').onsubmit=event=>{event.preventDefault();return send();};
byId('query').onkeydown=event=>{if(event.key==='Enter'&&!event.shiftKey&&!event.isComposing){event.preventDefault();send();}};
byId('new').onclick=()=>reset(false);byId('close').onclick=()=>reset(true);
for(const button of document.querySelectorAll('[data-suggestion]'))button.onclick=()=>{if(!busy){byId('query').value=button.dataset.suggestion;byId('query').focus();}};
let expandedGraph=false;byId('expand-graph').onclick=()=>{expandedGraph=!expandedGraph;byId('flow-card').className='flow-card'+(expandedGraph?' expanded-graph':'');byId('expand-graph').textContent=expandedGraph?'Reducir grafo ↙':'Ampliar grafo ↗';byId('expand-graph').setAttribute('aria-pressed',String(expandedGraph));};
diagram();renderFlow();
}
