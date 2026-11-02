"""Named complete annual debt/balance series; partial drawing series stay pending."""
import argparse
import json
from pathlib import Path
from decimal import Decimal, ROUND_HALF_UP
import shutil
import xml.etree.ElementTree as ET

import duckdb
from experiments.excel_pilot import ROOT, sha
from experiments.semantic_families import Original, staged, record, FIELDS, C, query as old_query, validate as validate_previous
from experiments.universal_database import require

PREVIOUS=ROOT/'evaluation/semantic_families_2026-10-31/run_v2/families.duckdb'
PREVIOUS_SHA='a16646facb0cfdc17bd536acb02d4a1d71e3490c34c14f68274882647cb6f2aa'
PROFILES={
 2:(2,6,'Deuda pública','Deuda pública (% PIB)','GRÁFICO 2. DEUDA  PÚBLICA (%PIB)', [('C','Deuda Pública')],87),
 3:(3,7,'Saldo público','Saldo público (% PIB)','GRÁFICO 3. SALDO PÚBLICO (% PIB)', [('C','Saldo primario'),('D','Saldo total')],174),
 29:(37,7,'Deuda pública','Deuda (% PIB)','GRÁFICO 29. DEUDA  PÚBLICA (%PIB)', [('C','Deuda Pública')],87),
 32:(41,7,'Saldo público','Saldo público (% PIB)','GRÁFICO 32. SALDO PÚBLICO (%PIB)', [('C','Saldo primario'),('D','Saldo total')],174),
}


def collect(con):
    source=Original('2026')
    records=[]
    try:
        for number,(chart,first,metric,title,original_title,series,positions) in PROFILES.items():
            sheet=f'Gráfico {number}'
            require(source.text(sheet,'B3')==original_title,'Debt title/unit mismatch')
            part=f'xl/charts/chart{chart}.xml'
            nodes=ET.fromstring(source.xml.parts[part]).findall('.//c:ser',C)
            rows=staged(con,'2026',sheet)
            require(len(rows)==positions,'Debt staging position count changed')
            for index,(col,label) in enumerate(series,1):
                node=nodes[index-1]
                label_cell=f'{col}{first-1}'
                require(source.text(sheet,label_cell)==label,'Debt original label mismatch')
                require(node.find('c:tx/c:strRef/c:f',C).text==f"'{sheet}'!${col}${first-1}",'Debt name binding mismatch')
                require(node.find('c:cat/c:numRef/c:f',C).text==f"'{sheet}'!$B${first}:$B${first+55}",'Debt year binding mismatch')
                require(node.find('c:val/c:numRef/c:f',C).text==f"'{sheet}'!${col}${first}:${col}${first+55}",'Debt value binding mismatch')
                selected=[r for r in rows if r['serie_id']==f'chart{chart}_s{index}']
                require(len(selected)==56,'Complete named series count changed')
                by_year={r['periodo_categoria_literal']:r for r in selected}
                require(set(by_year)=={str(y) for y in range(1995,2051)},'Debt years missing/duplicated')
                for year in range(1995,2051):
                    original_row=first+year-1995
                    cell=f'{col}{original_row}'
                    row=by_year[str(year)]
                    require(source.number(sheet,f'B{original_row}')==year,'Original debt year mismatch')
                    require(row['serie_literal']==label and row['titulo']==title and row['metrica_literal']==title,'Debt semantic label mismatch')
                    require(row['hoja_valor']==sheet and row['celda_valor']==cell and row['parte_grafico_imagen']==part,'Debt source binding mismatch')
                    require(row['estado_dato']=='dato' and row['tipo']=='posicion_serie' and row['unidad']=='ver_contexto_literal' and row['contexto_unidad_titulo']=='% PIB','Debt state/unit mismatch')
                    require(row['formula_texto'] is None and not str(source.text(sheet,cell)).startswith('='),'Unexpected debt formula')
                    item=record(row,metric,label,'level',year=year,family='named_annual_debt_balance')
                    require(item[11]==source.number(sheet,cell),'Exact original debt number mismatch')
                    records.append(item)
    finally: source.workbook.close()
    require(len(records)==336,'Debt family count changed')
    return records


def build(output):
    output=Path(output).resolve(); output.relative_to(ROOT)
    require(not output.exists(),'Output exists; no overwrite')
    require(sha(PREVIOUS)==PREVIOUS_SHA,'Previous database changed')
    validate_previous(PREVIOUS)
    output.mkdir(parents=True)
    db=output/'reviewed.duckdb'; shutil.copyfile(PREVIOUS,db)
    with duckdb.connect(str(db)) as con:
        rows=collect(con)
        con.execute('BEGIN')
        con.execute('CREATE TABLE debt_observations AS SELECT * FROM family_observations WHERE false')
        con.execute('ALTER TABLE debt_observations ADD PRIMARY KEY(book,row_number)')
        con.executemany('INSERT INTO debt_observations VALUES ('+','.join('?' for _ in FIELDS)+')',rows)
        con.execute('CREATE VIEW admitted_observations AS SELECT * FROM family_observations UNION ALL SELECT * FROM debt_observations')
        con.execute('''CREATE VIEW admission_review_queue AS SELECT s.* FROM staging_records s WHERE NOT EXISTS
            (SELECT 1 FROM admitted_observations a WHERE a.book=s.book AND a.row_number=s.row_number)''')
        con.execute('COMMIT')
    report=dict(status='built_pending_validation',admitted=896,added=336,pending=63753,previous_sha256=PREVIOUS_SHA,
                code_sha256=sha(Path(__file__)),db_sha256=sha(db),rag_integrated=False)
    (output/'manifest.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report


def validate(db):
    previous=validate_previous(db)
    with duckdb.connect(str(db),read_only=True) as con:
        expected=collect(con)
        actual=con.execute('SELECT * FROM debt_observations').fetchall()
        require(len(actual)==336 and set(actual)==set(expected),'Debt semantic content mismatch')
        legacy=con.execute('SELECT * FROM family_observations').fetchall()
        require(len({(r[0],r[1]) for r in actual+legacy})==896,'Admission source collision')
        require([c[0] for c in con.execute('SELECT * FROM admitted_observations LIMIT 0').description]==FIELDS,'Admission view columns mismatch')
        combined=con.execute('SELECT * FROM admitted_observations').fetchall()
        require(len(combined)==896 and set(combined)==set(expected+legacy),'Admission view content/multiplicity mismatch')
        queue='SELECT s.* FROM staging_records s WHERE NOT EXISTS(SELECT 1 FROM admitted_observations a WHERE a.book=s.book AND a.row_number=s.row_number)'
        require([c[0] for c in con.execute('SELECT * FROM admission_review_queue LIMIT 0').description]==[c[0] for c in con.execute(queue+' LIMIT 0').description],'Pending view columns mismatch')
        require(not con.execute('(SELECT * FROM admission_review_queue EXCEPT ALL ('+queue+')) UNION ALL (('+queue+') EXCEPT ALL SELECT * FROM admission_review_queue) LIMIT 1').fetchone(),'Pending content mismatch')
        require(con.execute('SELECT count(*) FROM admission_review_queue').fetchone()[0]==63753,'Pending count mismatch')
        counts=con.execute('SELECT block,series,count(*) FROM debt_observations GROUP BY ALL ORDER BY block,series').fetchall()
    examples=[query(db,book='2026',block='Gráfico 3',metric='Saldo público',series=s,year=2050) for s in ['Saldo primario','Saldo total']]
    return dict(status='pass',admitted=896,added=336,pending=63753,counts=counts,examples=examples,
                prior_validation=previous['status'],db_sha256=sha(Path(db)),code_sha256=sha(Path(__file__)),rag_integrated=False,
                limits=['186 partial or unnamed drawing positions in these blocks remain pending',
                        'No historical/projected split or vintage inferred from appearance', 'PDF physical citations and RAG routing pending'])


def query(db,*,book,block,metric,series,year):
    require(book in ['2025','2026'] and all(isinstance(t,str) and t.strip() for t in [block,metric,series]) and type(year) is int,'Explicit book/block/metric/series/integer year required')
    with duckdb.connect(str(db),read_only=True) as con:
        cursor=con.execute('''SELECT a.*,s.xlsx_path,s.xlsx_sha256,s.pdf_path,s.pdf_sha256 FROM admitted_observations a
          JOIN sources s ON s.book=a.book WHERE a.book=? AND block=? AND metric=? AND series=? AND year=? AND kind='level' ''',[book,block,metric,series,year])
        names=[c[0] for c in cursor.description]; rows=cursor.fetchall()
    if len(rows)!=1: raise LookupError('No unique admitted level; no fallback')
    result=dict(zip(names,rows[0])); result['value']=str(result['value'])
    result['display_value']=str(Decimal(result['value']).quantize(Decimal('.1'),rounding=ROUND_HALF_UP))
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__); sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('build');p.add_argument('--output-dir',type=Path,required=True)
    p=sub.add_parser('validate');p.add_argument('--db',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    if args.command=='build': report=build(args.output_dir)
    else:
        require(not args.output.exists(),'Output exists')
        try: report=validate(args.db)
        except Exception as exc: report=dict(status='fail',error=str(exc))
        args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2));return int(report['status']=='fail')


if __name__=='__main__':raise SystemExit(main())
