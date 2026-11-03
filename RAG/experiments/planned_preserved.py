"""Existing semantic planner with source-validated, single-package note preservation."""
import copy
from dataclasses import asdict

from experiments.compare_notes_only import NotesOnlyGrounder
from experiments.compare_system import generate_arm
from experiments.preserve_evidence import publication, validate_catalog, validate_publication
from experiments.reconstructed_units import expand_seeds
from experiments.source_bound_trial import RecordedGenerator
from experiments.validate_reranker import canonical_hash
from rag_pensiones.answer import DEFAULT_GENERATION_MODEL
from rag_pensiones.grounding import OllamaGrounder, validate_grounded_result
from rag_pensiones.qwen_embeddings import MODEL, QwenEmbedder
from rag_pensiones.search import search, validate_index
from rag_pensiones.semantic import answer_semantic


class PlannedNotesGrounder(NotesOnlyGrounder):
    def __init__(self, units, pages, *, prefer_retrieved_notes=False, **kwargs):
        if type(prefer_retrieved_notes) is not bool:
            raise ValueError('prefer_retrieved_notes must be a boolean')
        super().__init__(copy.deepcopy(units), copy.deepcopy(pages), **kwargs)
        self.prefer_retrieved_notes = prefer_retrieved_notes
        self.node_records = []
        self.pending = None

    def prepare_evidence(self, query, hits):
        if self.pending is not None:
            raise ValueError('Previous prepared evidence was not consumed')
        pages = list(self.pages.values())
        validate_catalog(self.extracted, pages)
        for hit in hits:
            chunk = hit.chunk
            page = self.pages.get((chunk.doc_id, chunk.page_number))
            if (page is None or chunk.source_file != page['source_file']
                    or chunk.published_at != page['published_at']
                    or not 0 <= chunk.start_char < chunk.end_char <= len(page['text'])
                    or page['text'][chunk.start_char:chunk.end_char] != chunk.text):
                raise ValueError('Retrieved seed lost canonical page provenance')
        expanded, catalog, expansion = expand_seeds(hits, self.extracted, self.pages, query,
            closed_note_tails=True, max_packages=1, prefer_retrieved_notes=self.prefer_retrieved_notes)
        self.pending = dict(query=query, hits=expanded, catalog=catalog, expansion=expansion)
        self.node_records.append(dict(query=query, status='prepared',
            seed_hits=[dict(rank=h.rank, score=h.score, chunk=asdict(h.chunk)) for h in hits]))
        return copy.deepcopy(expanded)

    def answer(self, query, hits, original_query=None, expected_digest=None):
        prepared = self.pending
        self.pending = None
        if prepared is None or prepared['query'] != query:
            raise ValueError('Grounding requires evidence prepared for this query')
        if (len(hits) != len(prepared['hits']) or
                any(a.chunk != b.chunk or a.score != b.score for a, b in zip(hits, prepared['hits']))
                or any(type(h.rank) is not int or h.rank <= 0 for h in hits)
                or len({h.rank for h in hits}) != len(hits)):
            raise ValueError('Prepared sources changed during global registration')
        remap = {f'S{a.rank}': f'S{b.rank}' for a, b in zip(prepared['hits'], hits)}
        self.catalog = copy.deepcopy(prepared['catalog'])
        for record in self.catalog['records']:
            for evidence in record['evidence']:
                evidence['source_id'] = remap[evidence['source_id']]
        self.expanded_hits = copy.deepcopy(hits)
        self.expansion = copy.deepcopy(prepared['expansion'])
        self.selections, self.bindings = [], []
        record = self.node_records[-1]
        first_call = len(getattr(self.generator, 'calls', []))
        try:
            # Expansion already happened before semantic source registration.
            result = OllamaGrounder.answer(self, query, hits, original_query, expected_digest)
            record['full_result'] = copy.deepcopy(result)
            validate_grounded_result(result, hits, expected_digest, expected_model=self.model)
            view = publication(result, preserve=True)
            validate_publication(view, result, preserve=True)
            record.update(status=view['status'], publication=view)
            return result
        finally:
            record.update(expanded_hits=[dict(rank=h.rank, score=h.score, chunk=asdict(h.chunk)) for h in hits],
                catalog=copy.deepcopy(self.catalog), expansion=copy.deepcopy(self.expansion),
                selections=copy.deepcopy(self.selections), bindings=copy.deepcopy(self.bindings),
                calls=copy.deepcopy(getattr(self.generator, 'calls', [])[first_call:]))


def validate_profile(index, units, pages):
    validate_index(index)
    if index['model'] != MODEL:
        raise ValueError('Preserved planner requires Qwen4B index')
    validate_catalog(units, pages)


def metadata(index, units, pages, digest, planned):
    return dict(model=DEFAULT_GENERATION_MODEL, digest=digest, embedding_model=MODEL,
        embedding_digest=index['model_digest'], index_hash=canonical_hash(index),
        units_hash=canonical_hash(units), pages_hash=canonical_hash(pages),
        closed_note_tails=True, max_packages=1, top_k=4, planner_used=planned,
        original_query_coverage_verified=False)


def answer_direct_notes(index, query, units, pages, *, embedder=None, generator=None):
    validate_profile(index, units, pages)
    if not isinstance(query, str) or not query.strip() or len(query.strip()) > 1000:
        raise ValueError('Invalid query')
    generator = generator if generator is not None else RecordedGenerator()
    embedder = embedder if embedder is not None else QwenEmbedder(timeout=180)
    grounder = NotesOnlyGrounder(copy.deepcopy(units), copy.deepcopy(pages), generator=generator)
    digest = generator.get_digest()
    hits = search(index, query.strip(), top_k=4, client=embedder)
    outcome = generate_arm(grounder, query.strip(), hits, digest, preserve=True)
    return dict(schema_version='direct-notes-query-v1', query=query.strip(),
        metadata=metadata(index, units, pages, digest, False), outcome=outcome,
        retrieval=dict(hits=[dict(rank=h.rank, score=h.score, chunk=asdict(h.chunk)) for h in hits],
                       embedding_calls=copy.deepcopy(getattr(embedder, 'calls', []))))


def answer_planned_preserved(index, query, units, pages, *, embedder=None, generator=None, planner=None,
                             prefer_retrieved_notes=False, query_builder=None):
    validate_profile(index, units, pages)
    generator = generator if generator is not None else RecordedGenerator()
    embedder = embedder if embedder is not None else QwenEmbedder(timeout=180)
    grounder = PlannedNotesGrounder(units, pages, generator=generator,
                                  prefer_retrieved_notes=prefer_retrieved_notes)
    canonical = answer_semantic(index, query, top_k=4, model=DEFAULT_GENERATION_MODEL,
                               embedder=embedder, generator=generator, planner=planner, grounder=grounder,
                               query_builder=query_builder)
    executed = [p for p in canonical['parts'] if p['execution_status'] == 'executed']
    if len(executed) != len(grounder.node_records):
        raise ValueError('Prepared node count differs from executed semantic nodes')
    views = {}
    sources = {s['source_id']: s for s in canonical['sources']}
    for part, record in zip(executed, grounder.node_records):
        full = record['full_result']
        if part['claims'] != full['claims'] or part['grounding'] != full['grounding']:
            raise ValueError('Semantic aggregation changed canonical grounding')
        if record['query'] != part.get('execution_query', part['resolved_question'] or part['question']):
            raise ValueError('Semantic node/query binding mismatch')
        validate_publication(record['publication'], full, preserve=True)
        for hit in record['expanded_hits']:
            source = sources[f"S{hit['rank']}"]
            if any(source[k] != hit['chunk'][k] for k in ('chunk_id','doc_id','source_file','published_at','page_number','text')):
                raise ValueError('Published source registry differs from expanded node evidence')
        record['node_id'] = part['id']
        views[part['id']] = record['publication']
    parts = [dict(id=p['id'], status=p['status'], execution_status=p['execution_status'],
                  claims=copy.deepcopy(views[p['id']]['claims']) if p['id'] in views else [], reason=p['reason'])
             for p in canonical['parts']]
    view = dict(status=canonical['status'], reason=canonical['reason'], parts=parts,
                claims=[c for p in parts for c in p['claims']] if canonical['status'] == 'answered' else [])
    meta = metadata(index, units, pages, canonical['generation_model_digest'], True)
    meta['prefer_retrieved_notes'] = prefer_retrieved_notes
    if query_builder is not None:
        meta['execution_query_builder'] = getattr(query_builder, '__name__', type(query_builder).__name__)
    return dict(schema_version='planned-preserved-query-v1', query=canonical['query'], metadata=meta,
        semantic=canonical, publication=view, nodes=grounder.node_records,
        embedding_calls=copy.deepcopy(getattr(embedder, 'calls', [])))
