"""Whole-corpus triage and exact, unit-explicit chart admission in one batch."""
from __future__ import annotations
import argparse
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re
import shutil
import xml.etree.ElementTree as ET

import duckdb
from experiments.excel_pilot import ROOT, BOOKS, sha
from experiments.semantic_families import Original, decimal, C
from experiments.debt_semantics import validate as validate_previous
from experiments.universal_database import require

PREVIOUS=ROOT/'evaluation/debt_semantics_2026-10-31/run_v1/reviewed.duckdb'
PREVIOUS_SHA='f2be13a44851cb948427eacd4f7fb2e1d19a0534c251bcf85a47823aaac6a1df'
# Literal units, no scaling, no guessed denominator. Generic % is intentionally absent.
UNITS={
 'años':'years', 'miles de personas':'thousand_persons',
 'grupos quinquenales de edad, miles de personas':'thousand_persons_age_groups',
 'miles de residentes':'thousand_residents', 'millones':'million_source_units',
 'miles de millones de euros':'billion_euros', 'miles de euros':'thousand_euros',
 'número de hijos por mujer':'children_per_woman', '%pib':'percent_gdp',
 '% pib':'percent_gdp', 'puntos pib':'percentage_points_gdp',
 'puntos de pib':'percentage_points_gdp',
 'diferencias interanuales en puntos del pib':'interannual_percentage_points_gdp',
 'millones de us$':'million_usd', 'millones de toneladas de co2e':'million_tonnes_co2e',
}
COLS='book row_number record_id block metric_literal series_id series_literal category_literal unit value source_sheet source_cell unit_cell unit_text value_ref category_ref name_ref chart_part rule_id'.split()
META_TYPES={'metadato_proyeccion','ranking_publicado','simbolo_sin_interpretar','supuesto_literal'}


def norm(value):return ' '.join(str(value or '').casefold().split())


def ref_cells(ref):
    """Local 1D Excel ranges only; reject external books, rectangles and expressions."""
    if not isinstance(ref,str) or '!' not in ref:raise ValueError('Missing range')
    sheet,span=ref.rsplit('!',1)
    if sheet.startswith("'") and sheet.endswith("'"):sheet=sheet[1:-1].replace("''", "'")
    if '[' in sheet or ']' in sheet:raise ValueError('External workbook')
    parts=span.replace('$','').split(':')
    if len(parts)==1:parts*=2
    if len(parts)!=2:raise ValueError('Multiple ranges')
    coords=[]
    for part in parts:
        match=re.fullmatch(r'([A-Z]+)([1-9][0-9]*)',part)
        if match is None:raise ValueError('Non-A1 range')
        col=0
        for char in match[1]:col=26*col+ord(char)-64
        coords.append((col,int(match[2])))
    (a,b),(c,d)=coords
    if (a!=c and b!=d) or c<a or d<b or max(c-a,d-b)>20000:raise ValueError('Not a bounded 1D range')
    from experiments.excel_pilot import column
    return sheet,[f'{column(x)}{y}' for x,y in
                  ([(x,b) for x in range(a,c+1)] if b==d else [(a,y) for y in range(b,d+1)])]


def numeric_candidate(row):
    if row['tipo'] in META_TYPES:return False
    if row['estado_dato'] not in ['dato','dato_numerico','dato_transcrito']:return False
    try:return Decimal(row['decimal_XML_texto']).is_finite()
    except (TypeError,ValueError,ArithmeticError):return False


class Reviewer:
    def __init__(self):
        self.sources={book:Original(book) for book in BOOKS}
        self.bindings={};self.unit_proofs={};self.formats={}

    def close(self):
        for source in self.sources.values():source.workbook.close()

    def binding(self,row):
        key=(row['book'],row['parte_grafico_imagen'],row['serie_id'])
        if key in self.bindings:return self.bindings[key]
        source=self.sources[row['book']]
        part=row['parte_grafico_imagen']
        match=re.fullmatch(r'chart([0-9]+)_s([0-9]+)',row['serie_id'] or '')
        if not match or part!=f'xl/charts/chart{match[1]}.xml':raise ValueError('Series identity')
        chart_root=ET.fromstring(source.xml.parts[part])
        # Raw values must not be relabeled with a scaled display unit.
        for axis in chart_root.findall('.//c:valAx',C):
            if axis.find('c:dispUnits',C) is not None:raise ValueError('Scaled display axis')
            fmt=axis.find('c:numFmt',C)
            if fmt is not None and ('%' in fmt.get('formatCode','') or re.search(r'[0#?],\s*,|[0#?],\s*(?:$|[;\"])',fmt.get('formatCode',''))):
                raise ValueError('Scaled axis format')
        nodes=chart_root.findall('.//c:ser',C)
        node=nodes[int(match[2])-1]
        # Bubble and multi-level axes need a different semantic contract.
        if node.find('c:bubbleSize',C) is not None:raise ValueError('Bubble series')
        name_node=node.find('c:tx/c:strRef/c:f',C)
        literal=node.find('c:tx/c:v',C)
        name_ref=name_node.text if name_node is not None else None
        if name_ref:
            sheet,cells=ref_cells(name_ref)
            if len(cells)!=1:raise ValueError('Non-scalar name')
            label=source.text(sheet,cells[0])
        elif literal is not None:label=literal.text
        else:raise ValueError('Unnamed series')
        if not isinstance(label,str) or not label.strip():raise ValueError('Empty name')
        category=node.find('c:cat/c:numRef/c:f',C)
        if category is None:category=node.find('c:cat/c:strRef/c:f',C)
        value=node.find('c:val/c:numRef/c:f',C)
        if category is None or value is None:raise ValueError('Missing simple axis refs')
        cs,cc=ref_cells(category.text);vs,vc=ref_cells(value.text)
        if len(cc)!=len(vc):raise ValueError('Unequal axes')
        categories=[]
        for address in cc:
            category_label=source.text(cs,address)
            if isinstance(category_label,str) and category_label.startswith('='):
                cached=source.number(cs,address)
                category_label=str(int(cached)) if cached is not None and cached==int(cached) else (str(cached) if cached is not None else None)
            categories.append(category_label)
        keys=[str(v) if v is not None else None for v in categories]
        # Repeated/blank categories are not uniquely queryable under this contract.
        if any(v is None for v in keys) or len(set(keys))!=len(keys):raise ValueError('Ambiguous categories')
        result=dict(label=label,name_ref=name_ref,category_ref=category.text,value_ref=value.text,
                    category_sheet=cs,value_sheet=vs,mapping=dict(zip(keys,vc)))
        self.bindings[key]=result
        return result

    def review(self,row):
        context=norm(row['contexto_unidad_titulo'])
        unit=UNITS.get(context)
        if unit is None:return 'unit_review',None
        if not isinstance(row['serie_literal'],str) or not row['serie_literal'].strip():return 'series_name_review',None
        try: binding=self.binding(row)
        except (KeyError,ValueError,IndexError,AttributeError):return 'binding_or_axis_review',None
        if row['serie_literal']!=binding['label']:return 'series_identity_review',None
        category=row['periodo_categoria_literal']
        cell=binding['mapping'].get(category)
        if cell is None:return 'category_review',None
        if row['hoja_valor']!=binding['value_sheet'] or row['celda_valor']!=cell:return 'source_binding_review',None
        source=self.sources[row['book']]
        try:
            value=decimal(row['decimal_XML_texto'])
            if value!=source.number(binding['value_sheet'],cell):return 'source_value_review',None
        except (ValueError,ArithmeticError):return 'source_value_review',None
        if row['tipo']!='posicion_serie' or row['unidad']!='ver_contexto_literal':return 'record_type_review',None
        # Explicit unit evidence must occur in original title/axis/label text, not just normalized metadata.
        proofkey=(row['book'],row['bloque'],context)
        if proofkey not in self.unit_proofs:
            needle=''.join(context.split())
            texts=source.sheet(row['bloque'])[1]
            proof=next(((a,t) for a,t in texts.items() if isinstance(t,str) and not t.startswith('=')
                        and needle in ''.join(norm(t).split())),None)
            self.unit_proofs[proofkey]=proof
        proof=self.unit_proofs[proofkey]
        if proof is None:return 'unit_evidence_review',None
        # Full literal metric must be independently present on the source sheet.
        # This rejects conflicting index titles (e.g. SALDO vs GASTO primario).
        def canonical(text):return ''.join(c for c in norm(text) if c.isalnum())
        metric=canonical(row['metrica_literal'])
        if not metric or row['metrica_literal']!=row['titulo'] or not any(
            metric in canonical(text) for text in source.sheet(row['bloque'])[1].values()
            if isinstance(text,str) and not text.startswith('=')):
            return 'metric_evidence_review',None
        # 'millions' without a measured quantity in the title stays unresolved.
        if unit=='million_source_units' and not any(token in norm(row['titulo']) for token in ['pensiones','personas','población']):
            return 'unit_review',None
        # Do not interpret proportions formatted as Excel percentages as unscaled literal percentages.
        fmtkey=(row['book'],binding['value_sheet'])
        if fmtkey not in self.formats:
            self.formats[fmtkey]={c.coordinate:c.number_format for r in source.workbook[binding['value_sheet']].iter_rows()
                                  for c in r if c.value is not None}
        fmt=self.formats[fmtkey].get(cell,'General')
        if '%' in fmt:return 'scale_review',None
        result=(row['book'],row['row_number'],row['id'],row['bloque'],row['metrica_literal'],row['serie_id'],
                row['serie_literal'],category,unit,value,binding['value_sheet'],cell,proof[0],proof[1],
                binding['value_ref'],binding['category_ref'],binding['name_ref'],row['parte_grafico_imagen'],'explicit_unit_named_complete_axis_v1')
        return 'usable_bulk',result


def classify(con):
    previous=set(con.execute('SELECT book,row_number FROM admitted_observations').fetchall())
    require(len(previous)==896,'Predecessor admissions changed')
    cursor=con.execute('SELECT * FROM staging_records ORDER BY book,row_number')
    names=[c[0] for c in cursor.description];rows=[dict(zip(names,r)) for r in cursor.fetchall()]
    reviewer=Reviewer();decisions=[];admitted=[]
    try:
        for row in rows:
            candidate=numeric_candidate(row)
            if (row['book'],row['row_number']) in previous:status='usable_previous'
            elif row['estado_dato']=='ausente':status='absent'
            elif row['estado_dato']=='error':status='source_error'
            elif not candidate:status='not_numeric_observation'
            elif row['tipo']!='posicion_serie':status='table_or_manual_review'
            else:
                status,result=reviewer.review(row)
                if result is not None:admitted.append(result)
            decisions.append((row['book'],row['row_number'],row['id'],candidate,status))
    finally:reviewer.close()
    require(len(decisions)==64649,'Corpus inventory changed')
    return decisions,admitted


DDL='''CREATE TABLE bulk_observations(book VARCHAR,row_number INTEGER,record_id VARCHAR,block VARCHAR,metric_literal VARCHAR,
 series_id VARCHAR,series_literal VARCHAR,category_literal VARCHAR,unit VARCHAR,value DECIMAL(38,24),source_sheet VARCHAR,
 source_cell VARCHAR,unit_cell VARCHAR,unit_text VARCHAR,value_ref VARCHAR,category_ref VARCHAR,name_ref VARCHAR,chart_part VARCHAR,
 rule_id VARCHAR,PRIMARY KEY(book,row_number));
 CREATE TABLE record_review(book VARCHAR,row_number INTEGER,record_id VARCHAR,numeric_candidate BOOLEAN,status VARCHAR,
 PRIMARY KEY(book,row_number));
 CREATE VIEW usable_keys AS SELECT book,row_number FROM admitted_observations UNION ALL SELECT book,row_number FROM bulk_observations;
 CREATE VIEW usable_records AS SELECT s.* FROM staging_records s JOIN usable_keys u USING(book,row_number);
 CREATE VIEW bulk_review_queue AS SELECT s.*,r.status AS review_reason,r.numeric_candidate FROM staging_records s
 JOIN record_review r USING(book,row_number) WHERE r.status NOT IN ('usable_previous','usable_bulk');'''


def build(output):
    output=Path(output).resolve();output.relative_to(ROOT)
    require(not output.exists(),'Output exists; no overwrite')
    require(sha(PREVIOUS)==PREVIOUS_SHA,'Predecessor changed')
    output.mkdir(parents=True);db=output/'bulk.duckdb';shutil.copyfile(PREVIOUS,db)
    with duckdb.connect(str(db)) as con:
        decisions,rows=classify(con)
        con.execute('BEGIN');con.execute(DDL)
        con.executemany('INSERT INTO record_review VALUES (?,?,?,?,?)',decisions)
        if rows:con.executemany('INSERT INTO bulk_observations VALUES ('+','.join('?' for _ in COLS)+')',rows)
        con.execute('COMMIT')
    report=dict(status='built_pending_validation',added=len(rows),admitted=896+len(rows),previous_sha256=PREVIOUS_SHA,
                code_sha256=sha(Path(__file__)),db_sha256=sha(db),rag_integrated=False)
    (output/'manifest.json').write_text(json.dumps(report,indent=2),encoding='utf-8');return report


def check_new(con):
    decisions,expected=classify(con)
    actual=con.execute('SELECT * FROM bulk_observations').fetchall()
    require(len(actual)==len(expected) and set(actual)==set(expected),'Bulk admitted content mismatch')
    actual_review=con.execute('SELECT * FROM record_review').fetchall()
    require(len(actual_review)==64649 and set(actual_review)==set(decisions),'Whole-corpus classification mismatch')
    keys={(b,n) for b,n,*_ in expected}|set(con.execute('SELECT book,row_number FROM admitted_observations').fetchall())
    require(len(keys)==896+len(expected),'Admission collision')
    for view,expression in [
        ('usable_keys','SELECT book,row_number FROM admitted_observations UNION ALL SELECT book,row_number FROM bulk_observations'),
        ('usable_records','SELECT s.* FROM staging_records s JOIN usable_keys u USING(book,row_number)'),
        ('bulk_review_queue',"SELECT s.*,r.status AS review_reason,r.numeric_candidate FROM staging_records s JOIN record_review r USING(book,row_number) WHERE r.status NOT IN ('usable_previous','usable_bulk')")]:
        require([c[0] for c in con.execute(f'SELECT * FROM {view} LIMIT 0').description]==[c[0] for c in con.execute(expression+' LIMIT 0').description],f'View columns mismatch: {view}')
        require(not con.execute(f'(SELECT * FROM {view} EXCEPT ALL ({expression})) UNION ALL (({expression}) EXCEPT ALL SELECT * FROM {view}) LIMIT 1').fetchone(),f'View content/multiplicity mismatch: {view}')
    return expected


def validate(db):
    prior=validate_previous(db)  # One complete inherited check at batch closure.
    with duckdb.connect(str(db),read_only=True) as con:
        rows=check_new(con)
        statuses=con.execute('SELECT status,numeric_candidate,count(*) FROM record_review GROUP BY ALL ORDER BY count(*) DESC').fetchall()
        candidates=con.execute('SELECT count(*) FROM record_review WHERE numeric_candidate').fetchone()[0]
        units=con.execute('SELECT unit,count(*),count(distinct book||block) FROM bulk_observations GROUP BY unit ORDER BY count(*) DESC').fetchall()
        sources=con.execute('SELECT count(*) FROM (SELECT DISTINCT book,source_sheet,source_cell FROM bulk_observations)').fetchone()[0]
    return dict(status='pass',added=len(rows),admitted=896+len(rows),numeric_candidates=candidates,
                numeric_candidates_pending=candidates-896-len(rows),whole_corpus=64649,statuses=statuses,units=units,
                unique_new_source_cells=sources,previous_validation=prior['status'],db_sha256=sha(Path(db)),code_sha256=sha(Path(__file__)),
                rag_integrated=False,limits=['Literal category preserved; no automatic calendar-year interpretation',
                'Explicit units, no scaling and no generic percentage denominator inference',
                'Repeated source cells remain distinct series records; query requires series_id',
                'Numeric candidates include repeated values, not distinct economic facts'])


def query(db,*,book,block,series_id,category):
    require(book in BOOKS and all(isinstance(t,str) and t.strip() for t in [block,series_id,category]),'Exact identity required')
    with duckdb.connect(str(db),read_only=True) as con:
        cursor=con.execute('''SELECT b.*,s.xlsx_path,s.xlsx_sha256,s.pdf_path,s.pdf_sha256 FROM bulk_observations b
             JOIN sources s ON s.book=b.book WHERE b.book=? AND block=? AND series_id=? AND category_literal=?''',[book,block,series_id,category])
        names=[c[0] for c in cursor.description];rows=cursor.fetchall()
    if len(rows)!=1:raise LookupError('No unique reviewed chart value; no fallback')
    result=dict(zip(names,rows[0]));result['value']=str(result['value']);return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='command',required=True)
    b=s.add_parser('build');b.add_argument('--output-dir',type=Path,required=True)
    b=s.add_parser('validate');b.add_argument('--db',type=Path,required=True);b.add_argument('--output',type=Path,required=True)
    a=p.parse_args(argv)
    if a.command=='build':report=build(a.output_dir)
    else:
        require(not a.output.exists(),'Report exists')
        try:report=validate(a.db)
        except Exception as exc:report=dict(status='fail',error=str(exc))
        a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2));return int(report['status']=='fail')


if __name__=='__main__':raise SystemExit(main())
