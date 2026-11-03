"""Page-bound documentary corpus and hybrid retrieval, separate from numeric data."""
from collections import Counter
import copy
import hashlib
import math
from pathlib import Path
import duckdb
from pypdf import PdfReader

from rag_pensiones.graph_catalog import canonical_hash, tokenize
from rag_pensiones.graph_types import EMBED_MODEL, EMBED_DIGEST, EMBED_DIMENSION
from rag_pensiones.qwen_embeddings import QUERY_PREFIX, vector
from rag_pensiones.structured_retrieval import ROOT, DEFAULT_DATABASE, CertifiedStructuredRetriever


def _corpus_content(corpus):
    return {k: corpus[k] for k in ['schema_version', 'scope', 'sources', 'pages', 'chunks']}


def source_registry(database=DEFAULT_DATABASE):
    CertifiedStructuredRetriever(database).verify_snapshot()
    with duckdb.connect(str(database), read_only=True) as connection:
        rows = connection.execute('SELECT * FROM sources').fetchall()
    return [dict(book=book, source_path=pdf, source_sha256=sha)
            for book, _, _, pdf, sha in rows]


def _chunk(page, start, end, source):
    text = page['text'][start:end]
    return dict(source_id='PDF:' + canonical_hash([page['doc_id'], page['page_number'], start, end, text])[:24],
                doc_id=page['doc_id'], book=source['book'], source_path=source['source_path'],
                source_sha256=source['source_sha256'], source_file=Path(source['source_path']).name,
                page_number=page['page_number'], start_char=start, end_char=end, text=text)


def build_document_corpus(database: Path = DEFAULT_DATABASE, *, chunk_chars=1800, overlap=200):
    if type(chunk_chars) is not int or type(overlap) is not int or not 0 <= overlap < chunk_chars <= 4000:
        raise ValueError('Invalid document chunk dimensions')
    sources = source_registry(database)
    pages, chunks = [], []
    for source in sources:
        path = ROOT / source['source_path']
        if hashlib.sha256(path.read_bytes()).hexdigest() != source['source_sha256']:
            raise ValueError('Original documentary source changed')
        reader = PdfReader(path)
        for number, pdf_page in enumerate(reader.pages, 1):
            text = pdf_page.extract_text() or ''
            page = dict(doc_id=source['source_sha256'], source_file=path.name,
                        book=source['book'], page_number=number, text=text)
            pages.append(page)
            for start in range(0, len(text), chunk_chars - overlap):
                end = min(len(text), start + chunk_chars)
                if text[start:end].strip():
                    chunks.append(_chunk(page, start, end, source))
                if end == len(text):
                    break
    corpus = dict(schema_version='graph-document-corpus-v1', scope='production_full',
                  sources=sources, pages=pages, chunks=chunks)
    corpus['corpus_hash'] = canonical_hash(_corpus_content(corpus))
    validate_document_corpus(corpus)
    return corpus


def legacy_document_corpus(index, pages, *, database=DEFAULT_DATABASE):
    """Convert the exact accepted13chunks; never label this as a full-report corpus."""
    from rag_pensiones.search import validate_index
    from rag_pensiones.search import DEFAULT_INDEX_PATH
    from rag_pensiones.qwen_embeddings import _json
    accepted=_json((ROOT/DEFAULT_INDEX_PATH).read_bytes())
    if canonical_hash(index)!=canonical_hash(accepted):
        raise ValueError('Legacy reference must use the exact accepted index')
    records = validate_index(index)
    registry = {s['source_sha256']: s for s in source_registry(database)}
    page_map = {(p['doc_id'], p['page_number']): p for p in pages}
    sources, converted, used_pages = {}, [], {}
    vectors = []
    for chunk, embedding in records:
        if chunk.doc_id not in registry or registry[chunk.doc_id]['book'] != '2025':
            raise ValueError('Legacy reference requires certified2025source')
        page = page_map[(chunk.doc_id, chunk.page_number)]
        if page['text'][chunk.start_char:chunk.end_char] != chunk.text:
            raise ValueError('Legacy indexed text differs from canonical page')
        source = registry[chunk.doc_id]
        sources[source['source_sha256']] = source
        used_pages[(chunk.doc_id, chunk.page_number)] = dict(page, book='2025')
        converted.append(_chunk(page, chunk.start_char, chunk.end_char, source))
        vectors.append(embedding)
    corpus = dict(schema_version='graph-document-corpus-v1', scope='legacy_2025_reference',
                  sources=list(sources.values()), pages=list(used_pages.values()), chunks=converted)
    corpus['corpus_hash'] = canonical_hash(_corpus_content(corpus))
    verify_document_sources(corpus,database=database)
    return attach_document_embeddings(corpus, vectors)


def validate_document_corpus(corpus):
    if corpus.get('schema_version') != 'graph-document-corpus-v1' or corpus.get('scope') not in {'production_full', 'legacy_2025_reference'}:
        raise ValueError('Unknown documentary corpus')
    if corpus.get('corpus_hash') != canonical_hash(_corpus_content(corpus)):
        raise ValueError('Documentary corpus hash mismatch')
    pages = {(p['doc_id'], p['page_number']): p for p in corpus['pages']}
    if len(pages) != len(corpus['pages']):
        raise ValueError('Duplicate documentary page identity')
    sources = {s['source_sha256']: s for s in corpus['sources']}
    seen = set()
    for chunk in corpus['chunks']:
        page = pages.get((chunk['doc_id'], chunk['page_number']))
        source = sources.get(chunk['doc_id'])
        if page is None or source is None or source['source_sha256'] != chunk['source_sha256']:
            raise ValueError('Unknown documentary provenance')
        if page['text'][chunk['start_char']:chunk['end_char']] != chunk['text']:
            raise ValueError('Chunk text differs from canonical page')
        expected = _chunk(page, chunk['start_char'], chunk['end_char'], source)
        if expected != chunk or chunk['source_id'] in seen:
            raise ValueError('Document chunk identity mismatch')
        seen.add(chunk['source_id'])
    return corpus


def verify_document_sources(corpus, *, database=DEFAULT_DATABASE):
    """Authenticate text against PDF bytes; a rehashed self-consistent forgery fails."""
    validate_document_corpus(corpus)
    registry={s['source_sha256']:s for s in source_registry(database)}
    if corpus['scope']=='production_full' and {s['source_sha256'] for s in corpus['sources']}!=set(registry):
        raise ValueError('Full corpus must expose both certified reports')
    for source in corpus['sources']:
        if source!=registry.get(source['source_sha256']):
            raise ValueError('Document outside the certified source registry')
        path=ROOT/source['source_path']
        if hashlib.sha256(path.read_bytes()).hexdigest()!=source['source_sha256']:
            raise ValueError('Document PDF bytes changed')
        reader=PdfReader(path)
        pages=[p for p in corpus['pages'] if p['doc_id']==source['source_sha256']]
        if corpus['scope']=='production_full' and {p['page_number'] for p in pages}!=set(range(1,len(reader.pages)+1)):
            raise ValueError('Full corpus page coverage incomplete')
        for page in pages:
            n=page['page_number']
            if type(n)is not int or not 1<=n<=len(reader.pages) or (reader.pages[n-1].extract_text()or'')!=page['text']:
                raise ValueError('Canonical page text differs from original PDF')
            if page.get('book')!=source['book'] or page['source_file']!=path.name:
                raise ValueError('Canonical page provenance differs')
            if corpus['scope']=='production_full':
                covered=set()
                for chunk in corpus['chunks']:
                    if chunk['doc_id']==page['doc_id'] and chunk['page_number']==n:
                        covered.update(range(chunk['start_char'],chunk['end_char']))
                if any(i not in covered and not c.isspace() for i,c in enumerate(page['text'])):
                    raise ValueError('Full corpus silently omits page text')
    return corpus


def attach_document_embeddings(corpus, embeddings):
    validate_document_corpus(corpus)
    if len(embeddings) != len(corpus['chunks']):
        raise ValueError('Document embedding count mismatch')
    result = copy.deepcopy(corpus)
    index = dict(corpus_hash=corpus['corpus_hash'], model=EMBED_MODEL, model_digest=EMBED_DIGEST,
                 dimension=EMBED_DIMENSION, query_prefix=QUERY_PREFIX, document_prefix='',
                 vectors={c['source_id']: vector(v) for c, v in zip(corpus['chunks'], embeddings)})
    index['index_hash'] = canonical_hash(index)
    result['embedding_index'] = index
    return result


def retrieve_documents(corpus, query, query_vector=None, top_k=6, *, books=None):
    validate_document_corpus(corpus)
    if not isinstance(query, str) or not query.strip() or type(top_k) is not int or top_k < 1:
        raise ValueError('Invalid documentary query')
    index = corpus.get('embedding_index')
    if (index is None) != (query_vector is None):
        raise ValueError('Both query and document vectors required')
    vectors = None
    if index is not None:
        expected = dict(corpus_hash=corpus['corpus_hash'], model=EMBED_MODEL, model_digest=EMBED_DIGEST,
                        dimension=EMBED_DIMENSION, query_prefix=QUERY_PREFIX, document_prefix='')
        if any(index.get(k) != v for k, v in expected.items()) or index['index_hash'] != canonical_hash({k:v for k,v in index.items() if k!='index_hash'}):
            raise ValueError('Stale documentary embedding index')
        if set(index['vectors']) != {c['source_id'] for c in corpus['chunks']}:
            raise ValueError('Document embedding IDs differ')
        vectors = {k:vector(v) for k,v in index['vectors'].items()}
        query_vector = vector(query_vector)
    docs = [tokenize(c['text']) for c in corpus['chunks']]
    n = len(docs)
    if not n:
        return []
    average = sum(map(len, docs))/n or 1
    df = Counter(t for doc in docs for t in set(doc))
    scores = []
    for chunk, doc in zip(corpus['chunks'], docs):
        if books is not None and chunk['book']not in books:continue
        tf=Counter(doc)
        score=sum(math.log(1+(n-df[t]+.5)/(df[t]+.5))*tf[t]*2.5/(tf[t]+1.5*(.25+.75*len(doc)/average)) for t in sorted(set(tokenize(query))))
        cosine=sum(a*b for a,b in zip(query_vector,vectors[chunk['source_id']])) if vectors is not None else None
        scores.append((chunk,score,cosine))
    lexical=sorted((s for s in scores if s[1]>0),key=lambda s:(-s[1],s[0]['source_id']))
    ranks={s[0]['source_id']:1/(60+i) for i,s in enumerate(lexical,1)}
    if vectors is not None:
        for i,s in enumerate(sorted(scores,key=lambda s:(-s[2],s[0]['source_id'])),1):
            sid=s[0]['source_id'];ranks[sid]=ranks.get(sid,0)+1/(60+i)
    ranked=sorted((s for s in scores if s[0]['source_id'] in ranks),key=lambda s:(-ranks[s[0]['source_id']],s[0]['source_id']))
    return [dict(**copy.deepcopy(s[0]),retrieval=dict(mode='hybrid_rrf' if vectors is not None else 'bm25_only',bm25=s[1],cosine=s[2],rrf=ranks[s[0]['source_id']],rank=i)) for i,s in enumerate(ranked[:top_k],1)]
