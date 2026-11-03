"""Cross-process retrieval must preserve exact scores, not merely top IDs."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from rag_pensiones import graph_catalog as c,graph_documents as d
from rag_pensiones.structured_retrieval import ROOT


class GraphDeterminismTests(unittest.TestCase):
    def test_catalog_and_document_scores_identical_across_hashseeds(self):
        texts=['gasto pensiones porcentaje producto interior bruto futuro escenario agregado',
               'pensiones gasto gasto futuro futuro escenario base población',
               'producto interior bruto indicador agregado revisión población porcentaje']
        cards=[];observations={};sources=[];pages=[];chunks=[]
        for i,text in enumerate(texts):
            identity=dict(book='2026',block=str(i),metric='Gasto',series='Base',unit='percent_gdp',
                          period_kind='year',scope='aggregate',quantity_role='level')
            oid=str(i);observations[oid]=dict(identity,period='2050',value='1.2')
            cards.append(dict(identity,card_id=oid,periods=['2050'],observation_ids=[oid],period_warnings=[],description=text))
            source=dict(book='2026',source_path='synthetic.pdf',source_sha256='sha'+oid)
            page=dict(doc_id=source['source_sha256'],page_number=1,text=text)
            sources.append(source);pages.append(page);chunks.append(d._chunk(page,0,len(text),source))
        catalog=dict(schema_version=c.SCHEMA,database_sha256='synthetic',sources={},cards=cards,observations=observations)
        catalog['catalog_hash']=c.canonical_hash(c._content(catalog))
        docs=dict(schema_version='graph-document-corpus-v1',scope='production_full',sources=sources,pages=pages,chunks=chunks)
        docs['corpus_hash']=c.canonical_hash(d._corpus_content(docs))
        script="import json,sys;from rag_pensiones.graph_catalog import retrieve_catalog;from rag_pensiones.graph_documents import retrieve_documents;x=json.load(open(sys.argv[1],encoding='utf-8'));q='gasto pensiones porcentaje producto interior bruto futuro escenario población agregado';print(json.dumps([retrieve_catalog(x['catalog'],q),retrieve_documents(x['docs'],q)],sort_keys=True,allow_nan=False))"
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp')as folder:
            path=Path(folder)/'inputs.json';path.write_text(json.dumps(dict(catalog=catalog,docs=docs)),encoding='utf-8')
            outputs=[]
            for seed in ['1','2']:
                env=dict(os.environ,PYTHONHASHSEED=seed)
                result=subprocess.run([sys.executable,'-c',script,str(path)],cwd=ROOT,env=env,capture_output=True,text=True,timeout=45,check=True)
                outputs.append(result.stdout)
        self.assertEqual(outputs[0],outputs[1])


if __name__=='__main__':unittest.main()
