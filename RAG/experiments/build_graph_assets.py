"""Build immutable, resumable semantic indices using existing local models only."""
import os
os.environ['LANGSMITH_TRACING']='false'
os.environ['LANGCHAIN_TRACING_V2']='false'
import argparse
import copy
import json
from pathlib import Path

from rag_pensiones.graph_catalog import build_catalog,attach_embeddings,canonical_hash,embedding_description
from rag_pensiones.graph_documents import build_document_corpus,attach_document_embeddings
from rag_pensiones.graph_models import GraphStageClient,MAX_EMBED_INPUT_BYTES
from rag_pensiones.graph_types import EMBED_MODEL,EMBED_DIGEST,EMBED_DIMENSION
from rag_pensiones.structured_retrieval import ROOT


def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,allow_nan=False,separators=(',',':')),encoding='utf-8')
    os.replace(temporary,path)


def _preflight_inputs(texts):
    if any(not isinstance(t,str)or not t.strip()or len(t.encode())>MAX_EMBED_INPUT_BYTES for t in texts):
        raise ValueError('Embedding build input exceeds byte budget or is empty')


def _embedded(texts,client,cache_path,*,batch_size=8,resume=False):
    _preflight_inputs(texts)
    from rag_pensiones.qwen_embeddings import _json,vector
    identity=dict(model=EMBED_MODEL,model_digest=EMBED_DIGEST,dimension=EMBED_DIMENSION,
                  input_hash=canonical_hash(texts),schema_version='graph-embedding-build-cache-v1')
    cache=dict(identity,rows={},calls=[])
    if cache_path.exists():
        if not resume:raise FileExistsError('Embedding cache exists; explicit resume required')
        cache=_json(cache_path.read_bytes())
        if any(cache.get(k)!=v for k,v in identity.items()):raise ValueError('Embedding resume inputs/model changed')
        for key,row in cache['rows'].items():
            index=int(key)
            if index<0 or index>=len(texts)or row['text_hash']!=canonical_hash(texts[index]):
                raise ValueError('Embedding cache row changed')
            call=cache['calls'][row['call_index']]
            pos=row['position']
            if (call['request']['model']!=EMBED_MODEL or call['response']['model']!=EMBED_MODEL
                or call['digest_before']!=EMBED_DIGEST or call['digest_after']!=EMBED_DIGEST
                or call.get('error')or call['request']['input'][pos]!=texts[index]
                or canonical_hash(vector(call['response']['embeddings'][pos]))!=canonical_hash(row['vector'])):
                raise ValueError('Embedding cache lacks matching actual model trace')
    client.get_embedding_digest()
    remaining=[i for i in range(len(texts))if str(i)not in cache['rows']]
    for offset in range(0,len(remaining),batch_size):
        indices=remaining[offset:offset+batch_size]
        outputs=client.embed([texts[i]for i in indices])
        call_index=len(cache['calls']);cache['calls'].append(copy.deepcopy(client.calls[-1]))
        for pos,(index,output)in enumerate(zip(indices,outputs)):
            cache['rows'][str(index)]=dict(text_hash=canonical_hash(texts[index]),vector=output,call_index=call_index,position=pos)
        atomic_json(cache_path,cache)
    return [cache['rows'][str(i)]['vector']for i in range(len(texts))]


def build_assets(output_dir,*,embed=False,batch_size=8,resume=False,client=None):
    directory=Path(output_dir).resolve();directory.relative_to(ROOT)
    if type(batch_size)is not int or not 1<=batch_size<=16:raise ValueError('Embedding batch must be1..16')
    directory.mkdir(parents=True,exist_ok=True)
    if (directory/'manifest.json').exists():raise FileExistsError('Completed graph assets are immutable; use a new directory')
    metadata=directory/'metadata.json'
    if metadata.exists():
        if not resume:raise FileExistsError('Build metadata exists; explicit resume required')
        from rag_pensiones.qwen_embeddings import _json
        prior=_json(metadata.read_bytes())
        catalog=build_catalog();documents=build_document_corpus()
        if prior!={'catalog_hash':catalog['catalog_hash'],'corpus_hash':documents['corpus_hash']}:
            raise ValueError('Build sources changed before resume')
    else:
        catalog=build_catalog();documents=build_document_corpus()
        atomic_json(metadata,dict(catalog_hash=catalog['catalog_hash'],corpus_hash=documents['corpus_hash']))
    if embed:
        catalog_texts=[embedding_description(c)for c in catalog['cards']]
        document_texts=[c['text']for c in documents['chunks']]
        _preflight_inputs(catalog_texts);_preflight_inputs(document_texts)
        client=client or GraphStageClient()
        catalog=attach_embeddings(catalog,_embedded(catalog_texts,client,
            directory/'catalog_embedding_cache.json',batch_size=batch_size,resume=resume))
        documents=attach_document_embeddings(documents,_embedded(document_texts,client,
            directory/'document_embedding_cache.json',batch_size=batch_size,resume=resume))
    atomic_json(directory/'catalog.json',catalog);atomic_json(directory/'documents.json',documents)
    manifest=dict(schema_version='graph-assets-build-v1',catalog_hash=catalog['catalog_hash'],
        corpus_hash=documents['corpus_hash'],cards=len(catalog['cards']),observations=len(catalog['observations']),
        pages=len(documents['pages']),chunks=len(documents['chunks']),hybrid=embed,
        catalog_embedding_hash=(catalog.get('embedding_index')or{}).get('index_hash'),
        document_embedding_hash=(documents.get('embedding_index')or{}).get('index_hash'))
    atomic_json(directory/'manifest.json',manifest)
    return manifest


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--embed',action='store_true',help='Run existing local Qwen4B embeddings (resource-intensive)')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--batch-size',type=int,default=8)
    args=parser.parse_args(argv)
    print(json.dumps(build_assets(args.output_dir,embed=args.embed,batch_size=args.batch_size,resume=args.resume),ensure_ascii=False,indent=2))
    return 0


if __name__=='__main__':raise SystemExit(main())
