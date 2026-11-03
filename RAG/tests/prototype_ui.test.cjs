const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');
const path=require('node:path');
const {webcrypto}=require('node:crypto');
const ui=require('../experiments/prototype_chat.js');
const group=id=>ui.FLOW_GROUPS.find(group=>group.id===id);

test('Parallel stages, retries, duplicates and errors describe actual events',()=>{
  const flow=ui.createFlow();let sequence=0;
  const event=(node,state)=>ui.applyEvent(flow,{node,state,sequence:++sequence});
  event('catalog','start');event('documents','start');event('catalog','end');
  assert.equal(ui.groupState(flow,group('catalog')),'done');
  assert.equal(ui.groupState(flow,group('documents')),'active');
  ui.applyEvent(flow,{node:'catalog',state:'start',sequence:1});
  assert.equal(flow.nodes.catalog.state,'done');
  event('documents','error');assert.equal(ui.groupState(flow,group('documents')),'error');
  event('generate','start');event('generate','end');event('generate','start');
  assert.equal(flow.nodes.generate.count,2);assert.equal(ui.groupState(flow,group('generate')),'active');
  flow.terminal='error';flow.complete=true;assert.equal(ui.groupState(flow,group('generate')),'error');
  assert.equal(ui.groupState(flow,group('publish')),'skipped');
  ui.applyEvent(flow,{node:'__proto__',state:'end',sequence:100});assert.equal(flow.cursor,sequence);
});

test('Missing telemetry is not presented as a completed or skipped execution',()=>{
  const flow=ui.createFlow();flow.terminal='answered';
  assert.equal(ui.groupState(flow,group('publish')),'unknown');
  ui.applyEvent(flow,{node:'generate',state:'start',sequence:1});
  assert.equal(ui.groupState(flow,group('generate')),'unknown');
});

test('A lost range stays unknown after subsequent complete telemetry reads',()=>{
  const flow=ui.createFlow();
  ui.ingestProgress(flow,{truncated:true,done:false,events:[{node:'publish',state:'start',sequence:510}]});
  ui.ingestProgress(flow,{truncated:false,done:true,events:[{node:'publish',state:'end',sequence:511}]});
  flow.terminal='answered';assert.equal(flow.complete,false);assert.equal(ui.groupState(flow,group('embed')),'unknown');
});

test('Citations deduplicate sources, preserve quotations and associate every claim',()=>{
  const evidence={source_id:'one',source_file:'Informe.pdf',page_number:53,quote:'Texto literal',source_kind:'pdf_text'};
  const citations=ui.collectCitations({publication:{claims:[{text:'Primera',evidence:[evidence]},{text:'Segunda',evidence:[evidence,{source_id:'two',source_file:'Tabla.xlsx',source_sheet:'Gasto',source_cell:'B2'}]}]}});
  assert.equal(citations.length,2);assert.deepEqual(citations[0].claims,['Primera','Segunda']);
  assert.equal(citations[0].quote,'Texto literal');assert.equal(citations[1].number,2);
  assert.match(ui.citationLabel(citations[1]),/hoja Gasto.*celda B2/);
});

test('Only local, validated report URLs become links',()=>{
  assert.equal(ui.tracePath('/report/12345678-1234-1234-1234-123456789abc'),'/report/12345678-1234-1234-1234-123456789abc');
  for(const input of ['javascript:alert(1)','https://foreign.example','//foreign/report','/report/../../secret',null])assert.equal(ui.tracePath(input),null);
});

// A small DOM adapter exercises the actual controller without a browser or models.
class Element{
  constructor(tag){this.tagName=tag;this.children=[];this.dataset={};this.attributes={};this.value='';this.className='';this.hidden=false;this._text='';}
  set textContent(value){this._text=String(value);this.children=[];}
  get textContent(){return this._text+this.children.map(child=>child.textContent).join('');}
  append(...nodes){for(const node of nodes){node.parent=this;this.children.push(node);}}
  replaceChildren(...nodes){this.children=[];this._text='';this.append(...nodes);}
  remove(){if(this.parent)this.parent.children=this.parent.children.filter(node=>node!==this);}
  setAttribute(name,value){this.attributes[name]=value;}
  focus(){this.focused=true;}
  querySelector(selector){return this.querySelectorAll(selector)[0]||null;}
  querySelectorAll(selector){const cls=selector.slice(1);return this.children.flatMap(child=>[...(child.className.split(' ').includes(cls)?[child]:[]),...child.querySelectorAll(selector)]);}
}
function harness({hangFinalProgress=false}={}){
  const elements=new Map(),html=fs.readFileSync(path.join(__dirname,'../experiments/prototype_chat_ui.html'),'utf8');
  for(const match of html.matchAll(/id="([^"]+)"/g)){const node=new Element('div');node.id=match[1];elements.set(node.id,node);}
  const find=(node,id)=>node.id===id?node:node.children.map(child=>find(child,id)).find(Boolean);
  const document={createElement:tag=>new Element(tag),createElementNS:(ns,tag)=>new Element(tag),getElementById:id=>elements.get(id)||[...elements.values()].map(node=>find(node,id)).find(Boolean),querySelectorAll:()=>[]};
  const calls=[];let resolveQuery,progressReads=0;
  const response=data=>({ok:true,status:200,json:async()=>data});
  const fetch=async(url,options={})=>{
    if(url.startsWith('/progress/')){
      if(hangFinalProgress&&++progressReads>1)return new Promise((resolve,reject)=>options.signal.addEventListener('abort',()=>reject(new Error('telemetry timeout')),{once:true}));
      return response({events:[{node:'interpret',state:'start',sequence:1},{node:'interpret',state:'end',sequence:2}],done:true});
    }
    calls.push({url,body:JSON.parse(options.body)});
    if(url==='/query')return new Promise(resolve=>{resolveQuery=resolve;});
    return response({conversation_id:'new-conversation'});
  };
  const context=vm.createContext({document,fetch,crypto:webcrypto,AbortController,setTimeout:(fn,ms)=>setTimeout(fn,Math.min(ms,20)),clearTimeout,console});
  vm.runInContext(fs.readFileSync(path.join(__dirname,'../experiments/prototype_chat.js'),'utf8'),context);
  return {elements,calls,context,reply:data=>resolveQuery(response(data)),send:()=>elements.get('composer').onsubmit({preventDefault(){}}),reset:()=>elements.get('new').onclick()};
}

test('A message appears immediately; citations and details render safely after the response',async()=>{
  const h=harness(),input=h.elements.get('query');input.value='<img src=x onerror=alert(1)>';
  const pending=h.send();
  assert.equal(h.elements.get('chat').children[0].className,'message user');
  assert.match(h.elements.get('chat').children[0].textContent,/<img/);
  assert.equal(h.elements.get('ask').disabled,true);
  h.reply({status:'answered',conversation_id:'conversation',formatted_answer:'Respuesta',trace_url:'javascript:alert(1)',report:{publication:{claims:[{text:'Afirmación',evidence:[{source_id:'one',source_file:'Fuente.pdf',page_number:4,quote:'<script>unsafe</script>',source_kind:'pdf_text'}]}]}}});
  await pending;
  const chat=h.elements.get('chat');assert.equal(chat.children.length,2);
  assert.equal(chat.querySelectorAll('.citation').length,1);
  assert.match(chat.querySelectorAll('.citation')[0].textContent,/<script>unsafe/);
  assert.equal(h.elements.get('detail-trace').hidden,true);assert.equal(h.elements.get('ask').disabled,false);
  assert.match(h.elements.get('detail-json').textContent,/Afirmación/);
});

test('Clarification uses the same composer and resumes the original question',async()=>{
  const h=harness();h.elements.get('query').value='Pregunta original';
  let pending=h.send();h.reply({status:'clarification',conversation_id:'conversation',session_id:'thread',formatted_answer:'¿Qué año?'});await pending;
  assert.equal(h.elements.get('clarify').hidden,false);
  h.elements.get('query').value='2050';pending=h.send();
  assert.equal(h.calls[1].body.query,'Pregunta original');assert.equal(h.calls[1].body.resume,'2050');assert.equal(h.calls[1].body.thread_id,'thread');
  h.reply({status:'answered',conversation_id:'conversation',formatted_answer:'Respuesta'});await pending;
  assert.equal(h.elements.get('clarify').hidden,true);
  assert.deepEqual(h.elements.get('chat').children.map(node=>node.className),['message user','message assistant','message user','message assistant']);
  await h.reset();assert.equal(h.elements.get('chat').children.length,1);assert.equal(h.elements.get('detail-json').hidden,true);
});

test('A returned error clears a failed clarification and allows a new question',async()=>{
  const h=harness();h.elements.get('query').value='Pregunta';
  let pending=h.send();h.reply({status:'clarification',conversation_id:'conversation',session_id:'thread',formatted_answer:'Aclara'});await pending;
  h.elements.get('query').value='Aclaración';pending=h.send();h.reply({status:'error',conversation_id:'conversation',formatted_answer:'Interrumpido'});await pending;
  assert.equal(h.elements.get('query').disabled,false);assert.equal(h.elements.get('clarify').hidden,true);
  h.elements.get('query').value='Pregunta nueva';pending=h.send();assert.equal(h.calls[2].body.resume,undefined);
  h.reply({status:'insufficient_evidence',conversation_id:'conversation',formatted_answer:'Sin evidencia'});await pending;
  assert.equal(h.elements.get('ask').disabled,false);
});

test('A hanging final telemetry read never prevents displaying a received answer',async()=>{
  const h=harness({hangFinalProgress:true});h.elements.get('query').value='Pregunta';
  const pending=h.send();h.reply({status:'answered',conversation_id:'conversation',formatted_answer:'Respuesta recibida'});await pending;
  assert.equal(h.elements.get('ask').disabled,false);assert.match(h.elements.get('chat').textContent,/Respuesta recibida/);
});


test('Outer runtime stages retain their own telemetry outside the 14 graph nodes',()=>{
  const flow=ui.createFlow();
  for(const stage of ui.OUTER_STAGES){
    ui.applyEvent(flow,{node:stage.id,state:'start'});assert.equal(ui.groupState(flow,stage),'active');
    ui.applyEvent(flow,{node:stage.id,state:'end'});assert.equal(ui.groupState(flow,stage),'done');
    assert.equal(ui.GRAPH_NODES.some(node=>node[0]===stage.id),false);
  }
  assert.equal(ui.GRAPH_NODES.length,14);
});


test('Compact graph draws exactly the 14 real nodes without extended explanations',()=>{
  const h=harness(),diagram=h.elements.get('flow-diagram');
  const boxes=diagram.children.filter(node=>node.dataset.group);
  assert.equal(boxes.length,14);
  assert.deepEqual(boxes.map(node=>node.dataset.group).sort(),ui.GRAPH_NODES.map(node=>node[0]).sort());
  assert.equal(diagram.querySelectorAll('.node-substep').length,0);
  assert.equal(diagram.querySelectorAll('.node-operations').length,0);
  h.elements.get('expand-graph').onclick();assert.equal(h.elements.get('flow-card').className,'flow-card expanded-graph');
  h.elements.get('expand-graph').onclick();assert.equal(h.elements.get('flow-card').className,'flow-card');
});
