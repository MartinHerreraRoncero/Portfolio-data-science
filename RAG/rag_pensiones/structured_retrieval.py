"""Concurrent, source-bound numeric retrieval; no model-generated SQL or arithmetic."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass,asdict
from decimal import Decimal
from pathlib import Path
import copy,hashlib,json,re,threading,unicodedata
from types import MappingProxyType
import duckdb
from experiments import source_recovery_policy as policy

ROOT=Path(__file__).resolve().parents[1]
DEFAULT_DATABASE=ROOT/'evaluation/source_recovery_policy_2026-11-01/run_v3/source_recovery_policy.duckdb'
DATABASE_SHA='92fc885172be97c3069218f1b107f49510d2428293c28fdd1b900df20ffffdd2'
POLICY_CODE_SHA='5f56e524578b784668bcaf6278c4ac8faa8a1cefe5aabc12065224a184196ab5'
OBJECT=re.compile(r'\b(cuadro|gr[aá]fico|gr\.)\s*(RE\s*)?(\d+(?:[._]\d+)?\s*[AB]?)\b',re.I)
STOP=set('el la los las un una unos unas de del en a al por para con que qué cuanto cuantos cuantas cual cuales se y o es son da tiene tienen valor cifra gasto publicado publicada representa segun serie del informe estudio opinion documento total'.split())
STOP.update('segun dato datos dentro preve preven previsto prevista proyecta proyectado aparece muestra indica publica adjunto excel año anos aumento publicado variacion contribucion crecimiento'.split())
STOP.update('cifras valores cifras representan previsto previstos previstas cuantia cantidad dato datos numero año años sin diferencia cuanto cuanta cuantas cuantos'.split())
STOP.update('porcentaje porcentajes entre publicados publicadas'.split())
REGIONS=['andalucia','aragon','asturias','baleares','canarias','cantabria','castilla la mancha','castilla y leon','cataluna','extremadura','galicia','madrid','murcia','navarra','pais vasco','la rioja','valencia','ceuta','melilla']

def norm(s):
 s=''.join(c for c in unicodedata.normalize('NFKD',str(s or''))if not unicodedata.combining(c)).lower();s=re.sub(r'(?<=[a-z])(?=\d)|(?<=\d)(?=[a-z])',' ',s)
 return re.sub(r'\s+',' ',re.sub(r'[^a-z0-9]+',' ',s)).strip()
ALIASES={'sanitario':'sanidad','sanitaria':'sanidad','educativo':'educacion','educativa':'educacion'}
def tokens(s):
 words=[ALIASES.get(w,w)for w in norm(s).split()if w not in STOP and not w.isdigit()]
 return{w[:-2]if len(w)>5 and w.endswith('es')else w[:-1]if len(w)>4 and w.endswith('s')else w for w in words}
def number(v):return format(Decimal(str(v)),'f').rstrip('0').rstrip('.')if '.'in format(Decimal(str(v)),'f')else str(v)
def object_key(s):
 m=OBJECT.search(str(s))
 return('cuadro'if m and m[1].lower()=='cuadro'else'grafico',('re'if m and m[2]else'')+re.sub(r'\s+','',m[3]).lower().replace('_','.'))if m else None
def report_references(text):
 books=[];spans=[];phrases=[]
 for m in re.finditer(r'\b(?:informe|estudio|documento)(?:\s+(?:de|del|la|las|el|airef))*\s*((?:19|20|21)\d{2})\b',text,re.I):books.append(m[1]);spans.append(m.span(1));phrases.append(m[0])
 for obj in OBJECT.finditer(text):
  m=re.match(r'\s*(?:/|de\s*)(?:la\s+|el\s+|del\s+)?(?:opini[oó]n\s*(?:de\s*)?|informe\s*(?:de\s*)?|estudio\s*(?:de\s*)?)?((?:19|20|21)\d{2})\b',text[obj.end():],re.I)
  if m:books.append(m[1]);spans.append((obj.end()+m.start(1),obj.end()+m.end(1)))
 return set(books),spans,phrases
def numeric_intent(text):
 if re.search(r'^\s*(?:y\s+)?(?:¿?por\s+qu[eé]|explica|¿?c[oó]mo|justifica|interpreta)',text,re.I):return False
 if re.search(r'¿\s*(?:por\s+qu[eé]|c[oó]mo|a\s+qu[eé]\s+se\s+debe)',text,re.I):return False
 explicit_scope=bool(OBJECT.search(text)or re.search(r'\b(?:informe|estudio|documento)\s*(?:de\s*)?(?:19|20|21)\d{2}\b',text,re.I))
 question=re.search(r'\bque\s+([a-z]+)',norm(text))
 if question and not explicit_scope and question[1]not in{'valor','valores','cifra','cifras','porcentaje','porcentajes','gasto','gastos','variacion','crecimiento','aumento','contribucion','importe','ratio','diferencia','total','numero','saldo','pension','pensiones','salario','intereses','empleos','cuantia','dato','datos'}:return False
 if not OBJECT.search(text)and re.search(r'\b(antes\s+de|despues\s+de|desde|hasta|maximo|minimo|umbral|supera)\b',norm(text)):return False
 return bool(explicit_scope or re.search(r'cu[aá]nt|porcentaje|variaci[oó]n|aumento|(?<!\d)(?:19|20|21)\d{2}(?!\d)',text,re.I))
def split_intents(query):
 boundaries=[m.start()for m in re.finditer(r';|\by\s+(?=(?:(?:a|de|para)\s+)?(?:qu[eé]|cu[aá]l|cu[aá]nt|por\s+qu[eé]|c[oó]mo)|explica|indica|(?:el\s+)?(?:cuadro|gr[aá]fico)|(?:su|sus|la|las)\s+(?:causa|raz[oó]n|motivo))',query,re.I)]
 boundaries += [m.end()for m in re.finditer(r'[?!](?=\s*\S)|\.(?=\s+[A-Za-zÁÉÍÓÚáéíóú¿])',query)]
 boundaries += [m.start()for m in re.finditer(r'\b(?:explica|justifica|interpreta|por\s+qu[eé])\b',query,re.I)if re.search(r'cu[aá]nt|valor|cifra|porcentaje|gasto|variaci[oó]n|importe|saldo|contribuci[oó]n',query[:m.start()],re.I)]
 cuts=sorted(set([0]+boundaries+[len(query)]))
 return[(a,b,query[a:b])for a,b in zip(cuts,cuts[1:])if query[a:b].strip(' ;?¿,.')]

@dataclass(frozen=True)
class StructuredHit:
 observation_id:str
 book:str|None
 block:str
 metric:str|None
 series:str|None
 period:str
 period_kind:str
 unit:str
 raw_value:str
 value:str
 source_sheet:str|None
 source_cell:str|None
 source_path:str
 source_sha256:str
 quantity_role:str|None
 warning:str|None
 aliases:tuple[str,...]=()
 numeric_reconciliation_failed:bool|None=None
 calculation_safe:bool|None=None
 global_calculation_prohibition:bool|None=None
 source_priority:str|None=None

class CertifiedStructuredRetriever:
 def __init__(self,database=DEFAULT_DATABASE):
  self.database=Path(database);self._signature=None;self._rows=None;self._sources={};self._source_signatures=set();self._lock=threading.RLock()
 def verify_snapshot(self):
  if not self.database.is_file():raise ValueError('Structured database unavailable')
  certificate=self.database.parent/'validation.json';signature=tuple((p.stat().st_size,p.stat().st_mtime_ns)for p in[self.database,certificate,ROOT/'experiments/source_recovery_policy.py'])
  with self._lock:
   if self._signature==signature:return
   if hashlib.sha256(self.database.read_bytes()).hexdigest()!=DATABASE_SHA:raise ValueError('Structured database certified hash mismatch')
   cert=json.loads(certificate.read_text(encoding='utf8'))
   if cert.get('status')!='pass'or cert.get('db_sha256')!=DATABASE_SHA or cert.get('code_sha256')!=POLICY_CODE_SHA or hashlib.sha256((ROOT/'experiments/source_recovery_policy.py').read_bytes()).hexdigest()!=POLICY_CODE_SHA:raise ValueError('Structured source policy certificate mismatch')
   self._signature=signature;self._rows=None
 def catalog(self):
  self.verify_snapshot()
  with self._lock:
   if self._rows is not None:return self._rows
   with duckdb.connect(str(self.database),read_only=True)as c:
    self._sources={b:dict(xlsx=x,xlsx_sha=h,pdf=p,pdf_sha=ph)for b,x,h,p,ph in c.execute('SELECT * FROM sources').fetchall()}
    q=c.execute('SELECT p.*,s.decimal_XML_texto original_numeric_text,s.titulo original_title FROM policy_public_observations p LEFT JOIN staging_records s USING(book,row_number)');names=[n[0]for n in q.description];raw=[dict(zip(names,x))for x in q.fetchall()]
   rows=[]
   for d in raw:
    for left,right in [('metric_literal','metric'),('series_literal','series')]:
     if d.get(left)is not None and d.get(right)is not None and norm(d[left])!=norm(d[right]):raise ValueError('Conflicting structured semantic aliases: '+left+'/'+right)
    metric=d.get('metric_literal')or d.get('metric');series=d.get('series_literal')or d.get('series');period=d.get('category_literal');kind=d.get('kind')or'category'
    if d.get('year')is not None:period=str(d['year']);kind='year'
    elif d.get('start_year')is not None and d.get('end_year')is not None:period=f"{d['start_year']}-{d['end_year']}";kind='interval'
    elif period and re.fullmatch(r'\d{4}',period):kind='year'
    elif period and re.fullmatch(r'\d{4}\s*[-–]\s*\d{4}',period):kind='interval'
    elif period and 'vs Opinion'in period:kind='comparison_horizon'
    if not period or d.get('value')is None:continue
    row=dict(d,metric_name=metric,series_name=series,period=str(period),period_kind=kind,object_key=object_key(d.get('block')),scope_text=' '.join(str(d.get(k)or'')for k in['metric_literal','metric','series_literal','series','scope_literal','scope','original_title','quantity_evidence']))
    rows.append(row)
   self._rows=tuple(MappingProxyType(r)for r in rows);return self._rows
 def context_book(self,index):
  self.catalog();items=index.get('chunks',index.get('items',index.get('records',index.get('embeddings',[]))))
  chunks=[x.get('chunk',x)for x in items]if isinstance(items,list)else[]
  if not chunks:return None
  metas={(str(x.get('doc_id','')),Path(str(x.get('source_file',''))).name,str(x.get('published_at','')))for x in chunks}
  if len(metas)!=1:return None
  docid,name,date=next(iter(metas));matches=[b for b,p in self._sources.items()if docid==p['pdf_sha']and norm(Path(p['pdf']).name)==norm(name)and date.startswith(b)]
  return matches[0]if len(matches)==1 else None
 def _periods(self,text,book_spans,series_labels):
  clean=text
  for a,b in book_spans:clean=clean[:a]+' '*(b-a)+clean[b:]
  for label in series_labels:
   words=re.findall(r'[A-Za-zÀ-ÿ]+|\d+',label or'')
   if words:clean=re.sub(r'\s*'.join(map(re.escape,words)),' ',clean,flags=re.I)
  intervals=list(re.finditer(r'(?<!\d)((?:19|20|21)\d{2})\s*(?:[-–]|a\s*|y\s*)((?:19|20|21)\d{2})(?!\d)',clean))
  if intervals and(any('-'in m[0]or'–'in m[0]for m in intervals)or re.search(r'\b(entre|desde|variacion|aumento|cambio|periodo|diferencial)\b',norm(clean))):return[(m[1]+'-'+m[2],'interval')for m in intervals]
  return[(v,'year')for v in dict.fromkeys(re.findall(r'(?<!\d)(?:19|20|21)\d{2}(?!\d)',clean))]
 def _resolve(self,text,start,end,context_book,context_basis='verified_index_document_context'):
  rows=self.catalog();obj=OBJECT.search(text);key=object_key(text);books,book_spans,book_phrases=report_references(text);ntext=norm(text);numeric=numeric_intent(text)
  base=dict(start=start,end=end,query_span=text,status='not_applicable',observation_ids=[],clarification=None)
  if not numeric:return base,[]
  if re.search(r'\b(antes\s+de|despues\s+de|desde|hasta|maximo|minimo|umbral|supera)\b',norm(text)):return dict(base,status='unmatched',clarification='La operación temporal o de umbral no equivale a una cifra puntual; hace falta una consulta específica.'),[]
  if len(books)>1:return dict(base,status='ambiguous',clarification='Especifica qué informe corresponde a cada dato.'),[]
  book=next(iter(books))if books else context_book
  base.update(book=book,book_basis='explicit'if books else context_basis if book else'unresolved')
  candidates=[r for r in rows if (not key or r['object_key']==key)and(not book or r['book']==book)]
  if not candidates:return dict(base,status='unmatched',clarification='No hay una cifra admitida para ese informe y objeto.'),[]
  if not book:return dict(base,status='ambiguous',clarification='Especifica el informe:2025 o2026.'),[]
  if re.search(r'personalmente|cobrar[eé]|mi\s+pensi[oó]n|por\s+persona|importe\s+individual',text,re.I):candidates=[r for r in candidates if any(t in r['unit']for t in['per_person','individual'])]
  requested=[s for s in REGIONS+['mujeres','hombres','clases pasivas','no contributivas','contributivas']if re.search(r'\b'+re.escape(s)+r'\b',ntext)]
  for scope in requested:candidates=[r for r in candidates if scope in norm(r['scope_text'])]
  for qualifier in['sin prtr','con prtr']:
   if qualifier in ntext:candidates=[r for r in candidates if qualifier in norm(r['scope_text'])]
  if not candidates:return dict(base,status='unmatched',clarification='La población, territorio o concepto solicitado no coincide con esas cifras.'),[]
  requested_concepts=tokens(text)&{'sanidad','educacion','pension','deuda','pib','interes'}
  if requested_concepts:
   candidates=[r for r in candidates if requested_concepts<=tokens(' '.join(str(r.get(k)or'')for k in['metric_name','series_name','original_title','scope_literal']))]
  if not candidates:return dict(base,status='unmatched',clarification='El concepto solicitado contradice la métrica acreditada de esa figura.'),[]
  if 'pdf'in ntext and key==('grafico','30'):return dict(base,status='unmatched',clarification='El PDF es una fuente alternativa explícita; la consulta principal de este gráfico da prioridad al Excel.'),[]
  labels={r['metric_name']for r in candidates if r['metric_name']};matches=[v for v in labels if tokens(v)and tokens(v)<=tokens(text)];matches=[v for v in matches if not any(tokens(v)<tokens(other)for other in matches)]
  if matches:candidates=[r for r in candidates if r['metric_name']in matches]
  elif len(labels)>1 and not(key and key[0]=='grafico'and len({norm(r.get('original_title'))for r in candidates})==1):return dict(base,status='ambiguous',clarification='Especifica el concepto exacto del cuadro o gráfico.'),[]
  series_text=norm(text)
  for label in set(matches)|{r.get('original_title')for r in candidates}|set(book_phrases)|({obj[0]}if obj else set()):
   if label:series_text=re.sub(r'\b'+re.escape(norm(label))+r'\b',' ',series_text)
  labels={r['series_name']for r in candidates if r['series_name']};qtokens=tokens(series_text)|({'total'}if re.search(r'\btotal\b',series_text)else set())
  smatches=[v for v in labels if re.search(r'\b'+re.escape(norm(v))+r'\b',series_text)or(tokens(v)|({'total'}if norm(v)=='total'else set()))and(tokens(v)|({'total'}if norm(v)=='total'else set()))<=qtokens and (not re.search(r'\d',v)or all(y in series_text for y in re.findall(r'\d{4}',v)))]
  smatches=[v for v in smatches if not any(tokens(v)<tokens(other)for other in smatches)]
  if re.search(r'\bserie\b',text,re.I)and not smatches:return dict(base,status='unmatched',clarification='La serie solicitada no está identificada por una etiqueta acreditada.'),[]
  if smatches:candidates=[r for r in candidates if r['series_name']in smatches]
  if not matches:
   content=series_text
   for label in smatches:content=re.sub(r'\b'+re.escape(norm(label))+r'\b',' ',content)
   requested_words=tokens(content);known=set().union(*(tokens(' '.join(str(r.get(k)or'')for k in['metric_name','original_title','scope_literal']))for r in candidates))
   known|=set().union(*(tokens(label)for label in smatches))if smatches else set()
   if any(norm(label)=='total'for label in smatches):known.add('total')
   if requested_words-known:return dict(base,status='unmatched',clarification='El concepto solicitado no está positivamente identificado en esta figura.'),[]
  periods=self._periods(text,book_spans,smatches)
  if not periods:return dict(base,status='ambiguous',clarification='Especifica el periodo exacto de la cifra.'),[]
  operation_text=norm(text)
  for label in smatches+matches:operation_text=re.sub(r'\b'+re.escape(norm(label))+r'\b',' ',operation_text)
  wants_growth=bool(re.search(r'\b(crecimiento|ritmo|tasa)\b',operation_text))
  wants_count=bool(re.search(r'\b(?:numero|cantidad)\s+(?:de\s+)?(?:pensiones|personas|pensionistas|afiliados|beneficiarios)|\bcuant[ao]s\s+(?:pensiones|personas|pensionistas|afiliados|beneficiarios)',norm(text)))
  hits=[];failures=[]
  for period,pkind in periods:
   selected=[r for r in candidates if (r['period']==period and(r['period_kind']==pkind or pkind=='year'and r['period_kind']=='year'))or r['period_kind']=='comparison_horizon'and pkind=='year'and r['period'].startswith(period+' ')]
   if re.search(r'puntos\s+porcentuales',text,re.I):selected=[r for r in selected if 'percentage_points'in r['unit'].lower()]
   elif re.search(r'%\s*(?:del\s+)?pib|porcentaje\s+(?:del\s+)?pib',text,re.I):selected=[r for r in selected if r['unit'].lower()=='percent_gdp']
   if re.search(r'\beuros\b',text,re.I):selected=[r for r in selected if 'euro'in r['unit']]
   if wants_growth:selected=[r for r in selected if 'annual'in r['unit']or('change'in r['unit']and'percent_gdp'not in r['unit']and'percentage_points'not in r['unit']and r['period_kind']=='year')]
   if wants_count:selected=[r for r in selected if not r['unit'].startswith('percent')and any(t in r['unit']for t in['person','pension','affiliat','count'])]
   groups={}
   for r in selected:groups.setdefault((r['metric_name'],r['series_name'],r['period'],r['unit'],r['source_sheet'],r['source_cell'],str(r['value'])),[]).append(r)
   if not groups:failures.append((period,'unmatched','No hay una cifra admitida con ese periodo, concepto y unidad exactos.'));continue
   if any('scatter'in str(r.get('quantity_role'))or r.get('binding_mode')=='source_snapshot_paired_coordinates'or'signed_plot'in r['unit']for r in selected):failures.append((period,'unmatched','Son coordenadas o pares de un gráfico; indica punto y coordenada, no una cifra agregada.'));continue
   if len(groups)>1:failures.append((period,'ambiguous','Hay varias series, unidades o puntos: especifica cuál necesitas.'));continue
   aliases=next(iter(groups.values()));r=sorted(aliases,key=lambda x:x['observation_id'])[0];d=policy.query_observation(self.database,observation_id=r['observation_id']);source=self._sources.get(r['book'],{});path=source.get('xlsx')or d.get('source_pdf_path');source_sha=source.get('xlsx_sha')or d.get('source_pdf_sha256')
   if any(d.get(k)!=r.get(k)for k in['book','source_sheet','source_cell','unit'])or Decimal(str(d['value']))!=Decimal(str(r['value']))or(d.get('metric_literal')or d.get('metric'))!=r['metric_name']or(d.get('series_literal')or d.get('series'))!=r['series_name']:raise ValueError('Structured catalog hit differs from source-policy API')
   if not path or not source_sha:raise ValueError('Structured hit missing source provenance')
   source_file=(ROOT/path).resolve();source_signature=(str(source_file),source_file.stat().st_size,source_file.stat().st_mtime_ns,source_sha)
   with self._lock:
    if source_signature not in self._source_signatures:
     if hashlib.sha256(source_file.read_bytes()).hexdigest()!=source_sha:raise ValueError('Structured cited workbook/source hash changed')
     self._source_signatures.add(source_signature)
   hit=StructuredHit(r['observation_id'],r['book'],str(r['block']).strip(),r['metric_name'],r['series_name'],r['period'],r['period_kind'],r['unit'],str(r.get('raw_value')if r.get('raw_value')is not None else r.get('original_numeric_text')or r['value']),str(r['value']),r.get('source_sheet'),r.get('source_cell'),path,source_sha,r.get('quantity_role')or r.get('kind'),d.get('warning'),tuple(a['observation_id']for a in aliases),d.get('numeric_reconciliation_failed'),d.get('calculation_safe'),d.get('global_calculation_prohibition'),d.get('source_priority'));hits.append(asdict(hit))
  bound_words=set().union(*(tokens(' '.join(str(h.get(k)or'')for k in['metric','series','block']))for h in hits))|requested_concepts|{'pib','porcentual','punto','euro','millon','mil','prtr','cuadro','grafico','gr','airef'}
  if any(norm(h.get('series'))=='total'for h in hits):bound_words.add('total')
  bound_numbers={n for h in hits for k in['book','block','period','series']for n in re.findall(r'\d+',str(h.get(k)or''))};bound_spans=[];residue=[]
  for word in re.finditer(r'\w+',text):
   allowed=not(tokens(word[0])-bound_words)and all(n in bound_numbers for n in re.findall(r'\d+',word[0]))
   (bound_spans if allowed else residue).append((start+word.start(),start+word.end()))
  missing_numbers={n for p,_,_ in failures for n in re.findall(r'\d+',p)}
  if any(tokens(text[a-start:b-start])or not re.findall(r'\d+',text[a-start:b-start])or not set(re.findall(r'\d+',text[a-start:b-start]))<=missing_numbers for a,b in residue):return dict(base,status='unmatched',clarification='Un calificador del dato solicitado no coincide con la métrica, unidad o población acreditada.',covered_spans=[]),[]
  state='partial'if hits and(failures or residue)else'ambiguous'if any(p[1]=='ambiguous'for p in failures)else'unmatched'if failures else'matched'
  clarification=' '.join(p[2]for p in failures)or('La base acredita la cifra; la petición adicional necesita evidencia textual.'if residue else None)
  return dict(base,status=state,observation_ids=[h['observation_id']for h in hits],missing_periods=[p[0]for p in failures],covered_spans=bound_spans if hits else[],uncovered_spans=residue,clarification=clarification),hits
 def retrieve(self,query,*,context_book=None):
  result=dict(schema_version='structured-retrieval-v1',query=query,status='not_applicable',hits=[],intents=[],database_sha256=DATABASE_SHA)
  try:
   if not isinstance(query,str)or not query.strip():raise ValueError('Nonempty query required')
   books,_,_=report_references(query)
   if books:context_book=next(iter(books))if len(books)==1 else None
   context_basis='query_global_report_reference'if len(books)==1 else'verified_index_document_context'
   for start,end,text in split_intents(query):
    intent,hits=self._resolve(text,start,end,context_book,context_basis);result['intents'].append(intent);result['hits'].extend(hits)
   states={p['status']for p in result['intents']};result['status']='partial'if result['hits']and states&{'partial','ambiguous','unmatched'}else'ambiguous'if'ambiguous'in states else'unmatched'if'unmatched'in states else'partial'if'partial'in states else'matched'if'matched'in states else'not_applicable'
  except Exception as e:result.update(status='error',hits=[],intents=[dict(start=a,end=b,query_span=t,status='error'if numeric_intent(t)else'not_applicable',covered_spans=[])for a,b,t in split_intents(query)],error=f'{type(e).__name__}: {e}')
  return result

_DEFAULT=CertifiedStructuredRetriever()
def default_retriever():return _DEFAULT
def source_manifest():
 service=default_retriever();service.verify_snapshot()
 if hashlib.sha256(policy.PLAN.read_bytes()).hexdigest()!=policy.PLAN_SHA:raise ValueError('Structured source-priority authorization changed')
 paths=[DEFAULT_DATABASE,DEFAULT_DATABASE.parent/'validation.json',policy.PLAN,policy.BASE.parent/'validation.json',ROOT/'requirements.txt']
 with duckdb.connect(str(DEFAULT_DATABASE),read_only=True)as c:
  for _,xlsx,xsha,pdf,psha in c.execute('SELECT * FROM sources').fetchall():
   for p,expected in[(ROOT/xlsx,xsha),(ROOT/pdf,psha)]:
    if hashlib.sha256(p.read_bytes()).hexdigest()!=expected:raise ValueError('Structured original source changed')
    paths.append(p)
 return dict(schema_version='structured-source-manifest-v1',database_path=DEFAULT_DATABASE.relative_to(ROOT).as_posix(),database_sha256=DATABASE_SHA,files={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()for p in paths})
def validate_source_manifest(manifest):
 if manifest!=source_manifest():raise ValueError('Structured source manifest differs from certified runtime')
def retrieve_parallel(semantic_callable,structured_callable):
 with ThreadPoolExecutor(max_workers=2,thread_name_prefix='rag-retrieval')as pool:
  semantic=pool.submit(semantic_callable);structured=pool.submit(structured_callable)
  return semantic.result(),structured.result()

def retrieve_for_index(query,index,retriever=None):
 service=retriever or default_retriever()
 try:return service.retrieve(query,context_book=service.context_book(index))
 except Exception as e:return dict(schema_version='structured-retrieval-v1',query=query,status='error',hits=[],intents=[dict(start=a,end=b,query_span=t,status='error'if numeric_intent(t)else'not_applicable',covered_spans=[])for a,b,t in split_intents(query)],error=f'{type(e).__name__}: {e}',database_sha256=DATABASE_SHA)

def validate_structured_result(query,index,structured,*,retriever=None):
 service=retriever or default_retriever();fresh=retrieve_for_index(query,index,service)
 if json.dumps(fresh,sort_keys=True,default=str)!=json.dumps(structured,sort_keys=True,default=str):raise ValueError('Structured result no longer matches source-bound retrieval')
 return fresh

def compose_publication(query,pdf_publication,structured,inventory,index,*,retriever=None):
 """Keep PDF canonical claims separate; combine only independently covered query spans."""
 if not structured or structured.get('status')=='not_applicable':return None
 validate_structured_result(query,index,structured,retriever=retriever)
 numeric=[p for p in structured.get('intents',[])if p['status']!='not_applicable'];covered=set();claims=[];parts=[];kept_ids=set();byid={r['claim_id']:r for r in inventory}
 for part in pdf_publication.get('parts',[]):
  if part.get('decision')!='answered':continue
  a,b=part['start_char'],part['end_char'];overlaps=[p for p in numeric if a<p['end']and b>p['start']];chosen=[]
  # A PDF claim spanning an authoritative numeric need cannot be reassigned to its residual prose.
  if overlaps:continue
  for cid in part.get('claim_ids',[]):
   row=byid.get(cid)
   if row is None:continue
   chosen.append(cid)
   if cid not in kept_ids:claims.append(copy.deepcopy(row['publication_claim']));kept_ids.add(cid)
  if chosen:
   allowed=set(range(a,b))
   for p in numeric:allowed-=set(range(p['start'],p['end']))
   covered|=allowed;parts.append(dict(source_kind='pdf_text',start=a,end=b,claim_ids=chosen,decision='answered'))
 seen=set()
 for p in numeric:
  if p['status']in['matched','partial']:
   for a,b in p.get('covered_spans',[]):covered|=set(range(a,b))
   parts.append(dict(source_kind='structured_database',start=p['start'],end=p['end'],observation_ids=p['observation_ids'],decision=p['status']))
 for h in structured.get('hits',[]):
  if h['observation_id']in seen:continue
  seen.add(h['observation_id']);sid='DB:'+h['observation_id'];label='; '.join(str(v)for v in[h['book'],h['block'],h['metric'],h['series'],h['period']]if v);text=f"{label}: {number(h['value'])} [{h['unit']}]."
  if h.get('warning'):text+=' '+h['warning']
  quote=f"{h['source_path']} — {h['source_sheet']}!{h['source_cell']}; raw={h['raw_value']}; normalizado={h['value']} {h['unit']}; SHA256={h['source_sha256']}"
  flags={k:h.get(k)for k in['numeric_reconciliation_failed','calculation_safe','global_calculation_prohibition','source_priority']}
  claims.append(dict(text=text,source_policy=flags,evidence=[dict(source_id=sid,quote=quote,source_kind='structured_cell',observation_id=h['observation_id'],source_path=h['source_path'],source_sheet=h['source_sheet'],source_cell=h['source_cell'],source_sha256=h['source_sha256'],source_policy=flags)]))
 required={i for i,c in enumerate(query)if c.isalnum()};missing=required-covered
 # A failed numeric branch cannot authorize numbers merely because a PDF claim looked plausible.
 unavailable=[]
 for a,b,text in split_intents(query):
  if any(i in missing for i in range(a,b)):
   diagnostics=[p.get('clarification')for p in numeric if p['start']<b and p['end']>a and p.get('clarification')];message=' '.join(diagnostics)or('No se pudo consultar la base estructurada: '+structured.get('error','')if structured.get('status')=='error'else'Las evidencias no cubren completamente esta parte de la pregunta.')
   message=f'Las evidencias evaluadas no contienen información suficiente para responder a: «{text}». '+message
   unavailable.append(dict(query_span=text,reason_code='structured_source_unresolved'if diagnostics or structured.get('status')=='error'else'not_in_evaluated_evidence',message=message))
 status='answered'if not missing else'partial'if claims else'insufficient_evidence'
 return dict(schema_version='joint-source-publication-v1',status=status,claims=claims,missing_parts=unavailable,parts=parts,reason=' '.join(p['message']for p in unavailable),pdf_missing_parts=copy.deepcopy(pdf_publication.get('missing_parts',[])),structured_status=structured['status'])
