import copy
from functools import lru_cache
import json
import unittest

from rag_pensiones import graph_documents as m
from rag_pensiones.graph_catalog import canonical_hash
from rag_pensiones.qwen_embeddings import _json
from rag_pensiones.search import DEFAULT_INDEX_PATH,load_index


@lru_cache(maxsize=1)
def reference_documents():
    index=load_index(m.ROOT/DEFAULT_INDEX_PATH)
    pages=[_json(s)for s in (m.ROOT/'data/processed/airef_2025_pages.jsonl').read_text(encoding='utf-8').splitlines()]
    return m.legacy_document_corpus(index,pages)


class GraphDocumentTests(unittest.TestCase):
    def test_reference_is_exact13_chunks_and_correct_pdf_identity(self):
        corpus=reference_documents()
        self.assertEqual(corpus['scope'],'legacy_2025_reference')
        self.assertEqual(len(corpus['chunks']),13)
        self.assertEqual({p['book']for p in corpus['pages']},{'2025'})
        m.verify_document_sources(corpus)

    def test_rehashed_self_consistent_fake_page_still_fails_pdf_authentication(self):
        corpus=copy.deepcopy(reference_documents())
        page=corpus['pages'][0]
        page['text']='X'+page['text'][1:]
        sources={s['source_sha256']:s for s in corpus['sources']}
        for i,chunk in enumerate(corpus['chunks']):
            if (chunk['doc_id'],chunk['page_number'])==(page['doc_id'],page['page_number']):
                corpus['chunks'][i]=m._chunk(page,chunk['start_char'],chunk['end_char'],sources[chunk['doc_id']])
        corpus['corpus_hash']=canonical_hash(m._corpus_content(corpus))
        m.validate_document_corpus(corpus)
        with self.assertRaisesRegex(ValueError,'original PDF'):
            m.verify_document_sources(corpus)

    def test_production_coverage_requires_both_reports_all_pages(self):
        corpus=copy.deepcopy(reference_documents())
        corpus['scope']='production_full'
        corpus['corpus_hash']=canonical_hash(m._corpus_content(corpus))
        with self.assertRaisesRegex(ValueError,'both certified'):
            m.verify_document_sources(corpus)

    def test_legacy_rejects_modified_embeddings_even_with_same_pdf_text(self):
        index=load_index(m.ROOT/DEFAULT_INDEX_PATH)
        index['chunks'][0]['embedding'][0]+=0.1
        pages=[_json(s)for s in (m.ROOT/'data/processed/airef_2025_pages.jsonl').read_text(encoding='utf-8').splitlines()]
        with self.assertRaisesRegex(ValueError,'exact accepted index'):
            m.legacy_document_corpus(index,pages)


if __name__=='__main__':unittest.main()
