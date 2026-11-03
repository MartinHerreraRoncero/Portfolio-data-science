"""Adjudicate saved identifiers only; no retrieval, embeddings or model calls."""
import hashlib
import json
from pathlib import Path
from experiments.source_equivalence import accepted_sources

ROOT=Path(__file__).resolve().parents[1]
ARMS=['baseline','diversity_only','reranker_only','reranker_diversity']


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def review(folder,output):
    folder=Path(folder);output=Path(output)
    if output.exists():raise FileExistsError(output)
    original_path=folder/'comparison.json';protocol_path=folder/'protocol.json'
    registry_path=ROOT/'experiments/fixtures/source_equivalence_2026-11-03.json'
    catalog=json.loads((ROOT/'data/processed/langgraph_9b_assets/catalog.json').read_text(encoding='utf-8'))
    registry=json.loads(registry_path.read_text(encoding='utf-8'))
    protocol=json.loads(protocol_path.read_text(encoding='utf-8'));original=json.loads(original_path.read_text(encoding='utf-8'))
    if original['protocol_sha256']!=sha(protocol_path):raise ValueError('Historical protocol changed')
    if protocol['catalog_hash']!=catalog['catalog_hash']:raise ValueError('Historical catalog changed')
    specs={c['id']:c for c in protocol['cases']};results=[]
    for old in original['cases']:
        spec=specs[old['id']];period=spec['need']['periods'][0]
        expected=accepted_sources(catalog,spec['expected_card_ids'],period,registry)
        result=dict(id=old['id'],original_expected=spec['expected_card_ids'],reviewed_expected=sorted(expected),
            added_sources=sorted(expected-set(spec['expected_card_ids'])),original_contract_review={},query_alignment='not_independently_certified')
        for arm in ARMS:
            ids=old[arm]['ids'];positions=[i+1 for i,cid in enumerate(ids)if cid in expected]
            first=min(positions)if positions else None
            result['original_contract_review'][arm]=dict(hit=bool(positions),first_rank=first,
                mrr=1/first if first else 0,old_hit=old[arm]['hit'],old_first_rank=old[arm]['first_rank'],
                unchanged_ids=ids)
        trace=json.loads((ROOT/f"outputs/prototipo/{spec['trace']}.json").read_text(encoding='utf-8'))
        result['original_user_context']=trace.get('effective_context')
        result.update(registry.get('case_review_notes',{}).get(old['id'],{}))
        results.append(result)
    counts={arm:sum(c['original_contract_review'][arm]['hit']for c in results)for arm in ARMS}
    clean=[c for c in results if c['query_alignment']!='publication_target_conflict']
    data=dict(review_kind='retrospective_source_equivalence_adjudication_not_new_execution',
        input_hashes={str(original_path):sha(original_path),str(protocol_path):sha(protocol_path),str(registry_path):sha(registry_path)},
        original_contract_source_hits=counts,total_original_contracts=5,
        excluding_flagged_contract=dict(total=len(clean),source_hits={a:sum(c['original_contract_review'][a]['hit']for c in clean)for a in ARMS}),
        cases=results,reranker_enabled=False,model_calls=0,new_embeddings=0,new_retrieval_calls=0,
        limitation='Retrospective annotations informed by observed outputs. No new benchmark, no baseline promotion, no answer-accuracy certification.')
    output.write_bytes((json.dumps(data,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))
    # Detect accidental mutation of the historical artifacts.
    if any(sha(Path(p))!=h for p,h in data['input_hashes'].items()):raise ValueError('Review inputs changed')
    print(json.dumps(dict(original_contract_source_hits=counts,excluding_flagged_contract=data['excluding_flagged_contract'])))
    return data


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--saved-dir',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();review(args.saved_dir,args.output)
