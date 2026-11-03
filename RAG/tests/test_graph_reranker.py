import math
import unittest
from unittest.mock import patch
from rag_pensiones.graph_reranker import order_candidates,LocalReranker
from rag_pensiones.graph_catalog import retrieve_catalog_for_needs


class RerankerTests(unittest.TestCase):
    def test_scores_preserve_identity_previous_scores_and_stable_ties(self):
        cards=[dict(card_id=str(i),retrieval=dict(rank=i+1,bm25=1))for i in range(12)]
        ranked=order_candidates(cards,[0.1]*11+[0.9],10)
        self.assertEqual(ranked[0]['card_id'],'11');self.assertEqual(ranked[1]['card_id'],'0')
        self.assertEqual(ranked[0]['retrieval']['rank'],12);self.assertNotIn('reranker',cards[0]['retrieval'])

    def test_invalid_scores_rejected(self):
        for scores in [[],[float('nan')],[float('inf')],[True]]:
            with self.assertRaises(ValueError):order_candidates([dict(card_id='a')],scores,1)

    def test_worker_failure_is_explicit(self):
        from types import SimpleNamespace
        with patch('rag_pensiones.graph_reranker.Path.is_file',return_value=True),patch('rag_pensiones.graph_reranker.subprocess.run',return_value=SimpleNamespace(returncode=1,stderr='model missing')):
            with self.assertRaisesRegex(RuntimeError,'model missing'):LocalReranker().rank('Question',[dict(card_id='a')])

    def test_missing_environment_is_explicit_without_starting_worker(self):
        with patch('rag_pensiones.graph_reranker.Path.is_file',return_value=False),patch('rag_pensiones.graph_reranker.subprocess.run')as run:
            with self.assertRaisesRegex(RuntimeError,'environment unavailable'):LocalReranker().rank('Question',[dict(card_id='a')])
            run.assert_not_called()

    def test_pool_is_widened_before_final_ten(self):
        cards=[dict(card_id=str(i),book='x',retrieval=dict(rank=i+1))for i in range(30)]
        class Fake:
            def rank(self,query,rows,limit,*,need=None):
                self.query=query;self.count=len(rows);self.need=need
                return order_candidates(rows,[1 if c['card_id']=='25'else 0 for c in rows],limit)
        fake=Fake()
        def retrieval(catalog,query,vector,top_k,allowed_card_ids):return cards[:top_k]
        with patch('rag_pensiones.graph_catalog.candidate_matches_need',return_value=True),patch('rag_pensiones.graph_catalog.derivation_operands_eligible',return_value=True),patch('rag_pensiones.graph_catalog.retrieve_catalog',side_effect=retrieval):
            args=({'cards':cards},'Complete query with scenario',None,[dict(need_id='n1',kind='data',book=None)])
            old=retrieve_catalog_for_needs(*args)
            new=retrieve_catalog_for_needs(*args,reranker=fake)
        self.assertEqual(len(new),10);self.assertEqual(fake.count,30);self.assertEqual(fake.query,args[1])
        self.assertEqual(fake.need,args[3][0])
        self.assertNotIn('25',[c['card_id']for c in old]);self.assertEqual(new[0]['card_id'],'25')

    def test_interpretation_is_transported_without_mutation(self):
        import copy
        from rag_pensiones.graph_reranker import reranker_request
        need=dict(metric='Reservoir volume',quantity_request='level',series='Reference',periods=['2040'],unit='m3')
        before=copy.deepcopy(need);request=reranker_request('Volume in 2040?',need)
        self.assertIn('"quantity_request": "level"',request);self.assertIn('Reference',request)
        self.assertEqual(need,before);self.assertEqual(reranker_request('Original'), 'Original')

    def test_diversity_preserves_scores_and_fills_with_multiple_rows(self):
        cards=[dict(card_id='a',book='x',source_path='one',block='table'),
               dict(card_id='b',book='x',source_path='one',block='table'),
               dict(card_id='c',book='x',source_path='one',block='chart')]
        result=order_candidates(cards,[0.9,0.8,0.1],2)
        self.assertEqual([r['card_id']for r in result],['a','c'])
        self.assertEqual(result[1]['retrieval']['reranker']['score'],0.1)
        self.assertEqual(result[1]['retrieval']['reranker']['pure_rank'],3)
        full=order_candidates(cards,[0.9,0.8,0.1],3)
        self.assertEqual([r['card_id']for r in full],['a','c','b'])
        self.assertEqual([r['card_id']for r in order_candidates(cards,[0.9,0.8,0.1],2,diversity=False)],['a','b'])
