"""Paired source-recall checks using frozen real traces and saved embeddings."""
import hashlib
import json
from pathlib import Path
import time
from rag_pensiones.graph_catalog import retrieve_catalog_for_needs
from rag_pensiones.graph_reranker import LocalReranker,order_candidates,source_object,MODEL,REVISION,reranker_request

ROOT=Path(__file__).resolve().parents[1]
PENSION_2070=['card:a6541ae9b1cb4c6d3f019a1f','card:76135aac6b91cc760c2882f6',
              'card:69738fcc385ff8ac06ff6205','card:973dd69334b67fd53eff4635']
CASES=[
    ('2070_previous_failure','5c5362d9-5219-4270-9bf7-bbc25346d63b',PENSION_2070),
    ('2070_previous_success','ae239f21-b606-45dc-9066-ff28db2377e3',PENSION_2070),
    ('2050_pension_level','2560e284-40a7-415c-a96b-5f03a0aef9cb',['card:08bd77071ae8e9b2aea694c9']),
    ('2050_public_deficit','a599bf8f-4dba-4002-9836-47abd550f078',['card:d861d0959a10fe0af77598a6']),
    ('2050_income_tax','fe40c5a4-befb-49c9-b62b-83c6e32d7e97',['card:8125bce933adb0b0b1a90c74'])]


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def build_protocol(catalog, registry=None):
    from experiments.source_equivalence import accepted_sources
    if registry is None:
        registry=ROOT/'experiments/fixtures/source_equivalence_2026-11-03.json'
    cases=[];cards={c['card_id']:c for c in catalog['cards']}
    for name,trace,expected in CASES:
        path=ROOT/f'outputs/prototipo/{trace}.json';report=json.loads(path.read_text(encoding='utf-8'))
        if report['manifest']['catalog_embedding_hash']!=catalog['embedding_index']['index_hash']:raise ValueError('Replay index mismatch')
        need=report['interpretation']['needs'][0];period=need['periods'][0]
        accepted=accepted_sources(catalog, expected, period, registry=registry)
        witnesses=[]
        for cid in sorted(accepted):
            card=cards[cid]
            row=next(catalog['observations'][oid]for oid in card['observation_ids']if catalog['observations'][oid]['period']==period)
            witnesses.append(dict(card_id=cid,book=card['book'],metric=card['metric'],series=card['series'],unit=card['unit'],
                observation={k:row.get(k)for k in ['period','value','source_path','source_sha256','source_sheet','source_cell']}))
        cases.append(dict(id=name,trace=trace,trace_sha256=digest(path),query=report['resolved_query'],need=need,
            expected_card_ids=sorted(accepted),witnesses=witnesses))
    registry_data=json.loads(Path(registry).read_text(encoding='utf-8'))if isinstance(registry,(str,Path))else registry
    registry_hash=hashlib.sha256(json.dumps(registry_data,sort_keys=True,ensure_ascii=False).encode('utf-8')).hexdigest()
    return dict(model=MODEL,revision=REVISION,catalog_hash=catalog['catalog_hash'],index_hash=catalog['embedding_index']['index_hash'],
        equivalence_registry_sha256=registry_hash,
        cases=cases,metric='source witness hit@10 and reciprocal rank; not answer accuracy',
        caveat='Case2070 was already used for development. Other four are regression controls, not an independent generalization benchmark.')


def summarize(rows,expected):
    hits=[i+1 for i,c in enumerate(rows)if c['card_id']in expected]
    return dict(hit=bool(hits),first_rank=min(hits)if hits else None,mrr=1/min(hits)if hits else 0,
        distinct_objects=len({source_object(c)for c in rows}),ids=[c['card_id']for c in rows])


def run(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    catalog=json.loads((ROOT/'data/processed/langgraph_9b_assets/catalog.json').read_text(encoding='utf-8'))
    protocol=build_protocol(catalog)
    protocol_path=output/'protocol.json';protocol_path.write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding='utf-8')
    reranker=LocalReranker();results=[]
    for case in protocol['cases']:
        report=json.loads((ROOT/f"outputs/prototipo/{case['trace']}.json").read_text(encoding='utf-8'))
        vector=next(c['response']['embeddings'][0]for c in report['stage_calls']if c.get('endpoint')=='/api/embed')
        args=(catalog,case['query'],vector,[case['need']]);start=time.monotonic()
        baseline=retrieve_catalog_for_needs(*args,metadata_discovery=True,diversity=False)
        baseline_seconds=time.monotonic()-start
        class Capture:
            def rank(self,query,rows,limit,*,need=None):self.rows=rows;return rows[:limit]
        capture=Capture();retrieve_catalog_for_needs(*args,metadata_discovery=True,reranker=capture)
        pool=capture.rows;expected=set(case['expected_card_ids'])
        diversity_only=order_candidates(pool,[-i for i in range(len(pool))],10)
        start=time.monotonic();all_ranked=reranker.rank(case['query'],pool,len(pool),need=case['need'])
        rerank_seconds=time.monotonic()-start
        scores={c['card_id']:c['retrieval']['reranker']['score']for c in all_ranked}
        raw_scores=[scores[c['card_id']]for c in pool]
        reranked=order_candidates(pool,raw_scores,10)
        score_only=order_candidates(pool,raw_scores,10,diversity=False)
        result=dict(id=case['id'],baseline=summarize(baseline,expected),diversity_only=summarize(diversity_only,expected),
            reranker_only=summarize(score_only,expected),reranker_diversity=summarize(reranked,expected),
            pool_contains_target=bool(expected&{c['card_id']for c in pool}),baseline_seconds=baseline_seconds,
            reranker_seconds=rerank_seconds,reranker_request=reranker_request(case['query'],case['need']),
            pool=[dict(card_id=c['card_id'],book=c['book'],block=c['block'],metric=c['metric'],series=c['series'],unit=c['unit'],score=scores[c['card_id']])for c in pool])
        results.append(result);(output/'results_partial.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps({k:result[k]for k in ['id','baseline','diversity_only','reranker_only','reranker_diversity','reranker_seconds']}),flush=True)
    improvements=[r['id']for r in results if not r['baseline']['hit']and r['reranker_diversity']['hit']]
    regressions=[r['id']for r in results if r['baseline']['hit']and not r['reranker_diversity']['hit']]
    summary=dict(protocol_sha256=digest(protocol_path),cases=results,improvements=improvements,regressions=regressions,
        source_recall={arm:sum(r[arm]['hit']for r in results)for arm in ['baseline','diversity_only','reranker_only','reranker_diversity']},
        meets_source_recall_gate=bool(improvements)and not regressions,production_enabled=False,
        limitation='Five source-recall replays only; no chatbot calls, new embeddings, general quality certification, or baseline promotion.')
    (output/'comparison.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    return summary


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--output-dir',required=True);args=parser.parse_args()
    run(args.output_dir)
