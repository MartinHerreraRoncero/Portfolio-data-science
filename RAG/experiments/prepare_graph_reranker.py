"""Explicit public download; never executed automatically by a user query."""
import hashlib
import json
from rag_pensiones.graph_reranker import MODEL,REVISION,MODEL_DIR


def main():
    from huggingface_hub import snapshot_download
    snapshot_download(MODEL,revision=REVISION,token=False,local_dir=MODEL_DIR,
        allow_patterns=['*.json','*.safetensors','*.txt','*.model','*.jinja'])
    files={}
    for path in sorted(MODEL_DIR.iterdir()):
        if not path.is_file()or path.name=='verified_files.json':continue
        digest=hashlib.sha256()
        with path.open('rb')as stream:
            for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
        files[path.name]=digest.hexdigest()
    (MODEL_DIR/'verified_files.json').write_text(json.dumps(dict(model=MODEL,revision=REVISION,files=files),indent=2),encoding='utf-8')
    print('Prepared pinned reranker:',MODEL,REVISION)


if __name__=='__main__':main()
