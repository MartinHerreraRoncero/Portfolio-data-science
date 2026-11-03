"""Active embedding dispatch and legacy reproducibility without network access."""
import copy
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from rag_pensiones import qwen_embeddings as qwen
from rag_pensiones.answer import answer
from rag_pensiones.search import DEFAULT_INDEX_PATH, LEGACY_INDEX_PATH, load_index, search, validate_index
from rag_pensiones.semantic import answer_semantic
from rag_pensiones.evaluate import evaluate, load_dataset, DEFAULT_DATASET_PATH


class ActiveQwenTests(unittest.TestCase):
    def setUp(self):
        self.index = load_index(DEFAULT_INDEX_PATH)
        self.client = SimpleNamespace(get_digest=lambda:self.index["model_digest"],
                                      embed_texts=lambda texts:[self.index["chunks"][0]["embedding"]])

    def test_default_qwen_and_explicit_legacy_both_validate(self):
        self.assertEqual(self.index["model"], qwen.MODEL)
        self.assertEqual(self.index["dimension"], 2560)
        old = load_index(LEGACY_INDEX_PATH)
        self.assertEqual(old["model"], "embeddinggemma:300m")
        self.assertEqual([c.chunk_id for c,_ in validate_index(self.index)], [c.chunk_id for c,_ in validate_index(old)])

    def test_default_search_creates_only_qwen_client_and_uses_its_prefix(self):
        inputs=[]
        def embed(texts):
            inputs.extend(texts)
            return [self.index["chunks"][0]["embedding"]]
        self.client.embed_texts=embed
        with patch('rag_pensiones.search.QwenEmbedder',return_value=self.client) as active, patch('rag_pensiones.search.OllamaEmbedder') as legacy:
            hits=search(self.index,' pregunta ',top_k=4)
        active.assert_called_once(); legacy.assert_not_called()
        self.assertEqual(inputs,[qwen.QUERY_PREFIX+'pregunta'])
        self.assertEqual(len(hits),4)
        self.assertEqual(hits[0].chunk.chunk_id,self.index['chunks'][0]['chunk_id'])

    def test_wrong_qwen_dimensions_and_document_identity_are_rejected(self):
        for mode in ['dimension','prefix','identity','boolean','zero']:
            d=copy.deepcopy(self.index)
            if mode=='dimension':d['dimension']=768
            elif mode=='prefix':d['query_prefix']='task: search result | query: '
            elif mode=='identity':d['chunks'][1]['source_file']='different.pdf'
            elif mode=='boolean':d['chunks'][0]['embedding'][0]=True
            else:d['chunks'][0]['embedding']=[0.0]*2560
            with self.subTest(mode=mode),self.assertRaises(ValueError):validate_index(d)

    def test_default_retrieval_evaluator_uses_active_qwen_client(self):
        dataset=load_dataset(DEFAULT_DATASET_PATH)
        with patch('rag_pensiones.evaluate.QwenEmbedder',return_value=self.client),patch('rag_pensiones.evaluate.OllamaEmbedder') as legacy:
            report=evaluate(self.index,dataset,top_k=4)
        legacy.assert_not_called()
        self.assertEqual(report['corpus_size'],13)

    def test_answer_and_semantic_create_active_embedder_without_gemma_fallback(self):
        query='¿Qué dice el informe?'
        stats={'prompt_eval_count':10,'eval_count':5,'eval_duration':100,'total_duration':200}
        generator=SimpleNamespace(model='qwen3.5:9b',get_digest=lambda:'g',generate=lambda messages:{
            'model':'qwen3.5:9b','done':True,'done_reason':'stop',
            'message':{'role':'assistant','content':json.dumps({'status':'insufficient_evidence','claims':[],'reason':'Sin respuesta.'})},**stats})
        planner=SimpleNamespace(model='qwen3.5:9b',plan=lambda *args,**kwargs:{
            'plan':{'subquestions':[{'id':'p1','question':query,'required_evidence':'Texto del informe.','depends_on':[]}]},
            'model':'qwen3.5:9b','model_digest':'g','prompt_sha256':'a'*64,'stats':stats,'parameters':{'temperature':0,'seed':42}})
        with patch('rag_pensiones.answer.QwenEmbedder',return_value=self.client),patch('rag_pensiones.answer.OllamaEmbedder') as legacy:
            result=answer(self.index,query,generator=generator)
            self.assertEqual(result['retrieval_model'],qwen.MODEL)
            legacy.assert_not_called()
        with patch('rag_pensiones.semantic.QwenEmbedder',return_value=self.client),patch('rag_pensiones.semantic.OllamaEmbedder') as legacy:
            result=answer_semantic(self.index,query,generator=generator,planner=planner)
            self.assertEqual(result['retrieval_model'],qwen.MODEL)
            legacy.assert_not_called()


if __name__=='__main__':unittest.main()
