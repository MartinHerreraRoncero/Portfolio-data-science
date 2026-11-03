import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.build_graph_assets import _embedded,build_assets
from rag_pensiones.graph_models import GraphStageClient
from rag_pensiones.structured_retrieval import ROOT
from test_graph_models import FakeTransport


class GraphAssetTests(unittest.TestCase):
    def test_preflight_checks_late_oversized_document_before_any_model_request(self):
        catalog={'catalog_hash':'test','cards':[{'description':'Magnitud: Gasto','periods':['2050']}]}
        documents={'corpus_hash':'test','chunks':[{'text':'válido'},{'text':'x'*6401}]}
        client=GraphStageClient(transport=FakeTransport())
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp')as directory, \
             patch('experiments.build_graph_assets.build_catalog',return_value=catalog), \
             patch('experiments.build_graph_assets.build_document_corpus',return_value=documents):
            with self.assertRaisesRegex(ValueError,'byte budget'):build_assets(Path(directory),embed=True,client=client)
        self.assertEqual(client.transport.requests,[])

    def test_preflight_rejects_large_identity_before_any_model_request(self):
        catalog={'catalog_hash':'test','cards':[{'description':'Magnitud: '+'x'*6401,'periods':['2050']}]}
        documents={'corpus_hash':'test','chunks':[{'text':'válido'}]}
        client=GraphStageClient(transport=FakeTransport())
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp')as directory, \
             patch('experiments.build_graph_assets.build_catalog',return_value=catalog), \
             patch('experiments.build_graph_assets.build_document_corpus',return_value=documents):
            with self.assertRaisesRegex(ValueError,'identity'):build_assets(Path(directory),embed=True,client=client)
        self.assertEqual(client.transport.requests,[])

    def test_embedding_cache_resumes_without_new_inference_and_rejects_forgery(self):
        client=GraphStageClient(transport=FakeTransport())
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp')as directory:
            path=Path(directory)/'cache.json'
            original=_embedded(['first','second','third'],client,path,batch_size=2)
            inference_count=len(client.calls)
            repeated=_embedded(['first','second','third'],client,path,batch_size=2,resume=True)
            self.assertEqual(original,repeated)
            self.assertEqual(len(client.calls),inference_count)
            with self.assertRaisesRegex(FileExistsError,'explicit resume'):
                _embedded(['first','second','third'],client,path)
            with self.assertRaisesRegex(ValueError,'changed'):
                _embedded(['another'],client,path,resume=True)
            cache=json.loads(path.read_text(encoding='utf-8'))
            cache['rows']['0']['vector'][1]=1.
            path.write_text(json.dumps(cache),encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'matching actual'):
                _embedded(['first','second','third'],client,path,resume=True)


if __name__=='__main__':unittest.main()
