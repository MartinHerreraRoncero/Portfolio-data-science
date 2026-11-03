"""Five fixed ranking checks; no chatbot calls or new embedding generation."""
import json
from pathlib import Path
import time
from rag_pensiones.graph_reranker import LocalReranker,MODEL,REVISION
from rag_pensiones.graph_catalog import retrieve_catalog_for_needs

ROOT=Path(__file__).resolve().parents[1]


def card(cid,metric,series,unit,years):
    return dict(card_id=cid,title=metric,metric=metric,series=series,unit=unit,
        periods=years,book='synthetic',period_kind='year',scope='whole system',quantity_role='')


def run(output):
    reranker=LocalReranker();results=[]
    catalog=json.loads((ROOT/'data/processed/langgraph_9b_assets/catalog.json').read_text(encoding='utf-8'))
    report=json.loads((ROOT/'outputs/prototipo/5c5362d9-5219-4270-9bf7-bbc25346d63b.json').read_text(encoding='utf-8'))
    if report['manifest']['catalog_embedding_hash']!=catalog['embedding_index']['index_hash']:raise ValueError('Replay index mismatch')
    vector=next(c['response']['embeddings'][0]for c in report['stage_calls']if c.get('endpoint')=='/api/embed')
    need=report['interpretation']['needs'][0];q=report['resolved_query'];start=time.monotonic()
    before=retrieve_catalog_for_needs(catalog,q,vector,[need],metadata_discovery=True)
    after=retrieve_catalog_for_needs(catalog,q,vector,[need],metadata_discovery=True,reranker=reranker)
    expected={'card:a6541ae9b1cb4c6d3f019a1f','card:76135aac6b91cc760c2882f6','card:69738fcc385ff8ac06ff6205','card:973dd69334b67fd53eff4635'}
    results.append(dict(id='real_2070_level',passed=bool(expected&{c['card_id']for c in after}),
        baseline_target_present=bool(expected&{c['card_id']for c in before}),seconds=time.monotonic()-start,
        ranked=[dict(id=c['card_id'],series=c['series'],metric=c['metric'],reranker=c['retrieval']['reranker'])for c in after]))
    cases=[
        ('explicit_scenario','Consumo eléctrico previsto para 2040 en el escenario base.',
         [card('alternative','Consumo eléctrico','demanda alta','GWh',['2040']),card('base','Consumo eléctrico','escenario base','GWh',['2040'])],'base'),
        ('year_coverage','Producción industrial prevista para 2070.',
         [card('short','Producción industrial','base','toneladas',['2025','2050']),card('long','Producción industrial','base','toneladas',['2025','2070'])],'long'),
        ('aggregate_scope','Gasto total de la red de hospitales en 2024.',
         [card('component','Gasto en medicamentos de la red de hospitales','medicamentos','euros',['2024']),card('total','Gasto total de la red de hospitales','total','euros',['2024'])],'total'),
        ('unrelated_pool','¿Cuál es el déficit del transporte urbano en 2024?',
         [card('weather','Temperatura media','verano','grados',['2024']),card('education','Número de alumnos','primaria','personas',['2024'])],None)]
    for name,q,candidates,expected_id in cases:
        start=time.monotonic();ranked=reranker.rank(q,candidates,len(candidates))
        # No threshold authorizes an answer. The unrelated case checks low scores only.
        passed=ranked[0]['card_id']==expected_id if expected_id else max(c['retrieval']['reranker']['score']for c in ranked)<0.1
        results.append(dict(id=name,passed=passed,expected=expected_id,seconds=time.monotonic()-start,
            ranked=[dict(id=c['card_id'],score=c['retrieval']['reranker']['score'])for c in ranked]))
    data=dict(model=MODEL,revision=REVISION,cases=results,passed=sum(r['passed']for r in results),total=5,
        limitation='Ranking checks only. Not conversational quality, calibrated probability, or factual verification.')
    path=Path(output);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(passed=data['passed'],total=5,cases=[dict(id=r['id'],passed=r['passed'],seconds=round(r['seconds'],2))for r in results])))
    return data


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);args=parser.parse_args()
    raise SystemExit(0 if run(args.output)['passed']==5 else 1)
