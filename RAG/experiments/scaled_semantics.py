"""Explicit display-axis conversions and scoped percentages, preserving raw values."""
from __future__ import annotations
import argparse
from decimal import Decimal, localcontext
import json
from pathlib import Path
import re
import shutil
import xml.etree.ElementTree as ET

import duckdb
from experiments.bulk_semantics import Reviewer, UNITS, COLS, C, norm, ref_cells, numeric_candidate, validate as validate_previous
from experiments.excel_pilot import ROOT,sha
from experiments.semantic_families import decimal
from experiments.universal_database import require

PREVIOUS=ROOT/'evaluation/bulk_semantics_2026-10-31/run_v4/bulk.duckdb'
PREVIOUS_SHA='c7b3d54605cda3d3b8524688d50d879f79e8922e3f6e1fa16ea341db903c0aac'
DIVISORS={name:Decimal(10)**power for name,power in [('hundreds',2),('thousands',3),('tenThousands',4),
 ('hundredThousands',5),('millions',6),('tenMillions',7),('hundredMillions',8),('billions',9),('trillions',12)]}
# Exact block/title profiles: a generic % is never given an inferred denominator.
PERCENT_PROFILES={
 ('2025','Gráfico 67'):('RENTABILIDAD DE LA DEUDA A 10 AÑOS (%)','percent_interest_rate_as_published'),
 ('2025','Gráfico 120'):('TIPO DE INTERÉS IMPLÍCITO (%)','percent_interest_rate_as_published'),
 ('2025','Gráfico 119'):('TIPO INTERÉS NUEVAS EMISIONES (%)','percent_interest_rate_as_published'),
 ('2025','Gráfico 82'):('TIPOS DE INTERÉS DE LAS NUEVAS EMISIONES (%), RANGO DE +/- 50 PUNTOS BASICOS','percent_interest_rate_as_published'),
 ('2025','Gráfico 69'):('COSTE MEDIO DE LA DEUDA DEL ESTADO (%)','percent_interest_rate_as_published'),
 ('2025','Gráfico 56'):('TIPO DE INTERÉS IMPLÍCITO Y NUEVAS EMISIONES (%)','percent_interest_rate_as_published'),
 ('2025','Gráfico 14'):('CRECIMIENTO DEL PIB POTENCIAL (%)','percent_growth_as_published'),
 ('2025','Gráfico 16'):('CRECIMIENTO DEL PIB PER CÁPITA (%)','percent_growth_as_published'),
 ('2025','Gráfico 27'):('ESCENARIO CENTRAL DEL PIB. TASAS DE CRECIMIENTO DEL PIB REAL, NOMINAL Y DEL DEFLACTOR (%)','percent_growth_as_published'),
}
POPULATION_CONTEXTS={norm(t) for t in ['% DE LA POBLACIÓN DE 16 A 66 AÑOS','%POBLACIÓN 16-66 AÑOS',
 '% POBLACIÓN 16-66 AÑOS','% POBLACIÓN 16-74 AÑOS','% población 15-74 años']}
EXTRA='raw_value axis_id display_unit axis_format cell_format divisor multiplier scope_literal'.split()
FIELDS=COLS+EXTRA


def format_effect(code):
    """Uniform numeric format sections only; quoted/escaped % is a literal."""
    require(isinstance(code,str),'Missing explicit format')
    require(not re.search(r'\[[<>=]',code),'Conditional number format')
    cleaned=re.sub(r'"[^"]*"','',code)
    cleaned=re.sub(r'\\.|_.|\*.','',cleaned)
    effects=[]
    for section in cleaned.split(';')[:3]:
        if not re.search(r'[0#?]',section):continue
        require(section.count('%')<=1,'Multiple percent signs')
        commas=re.findall(r'[0#?](,+)(?=[^0#?.,]|$)',section)
        divisor=Decimal(1000)**(len(commas[-1]) if commas else 0)
        effects.append((divisor,Decimal(100) if '%' in section else Decimal(1)))
    if not effects:return Decimal(1),Decimal(1)
    require(len(set(effects))==1,'Mixed scaling across format sections')
    return effects[0]


class ScaleReviewer(Reviewer):
    def binding(self,row):
        key=(row['book'],row['parte_grafico_imagen'],row['serie_id'])
        if key in self.bindings:return self.bindings[key]
        source=self.sources[row['book']];part=row['parte_grafico_imagen']
        match=re.fullmatch(r'chart([0-9]+)_s([1-9][0-9]*)',row['serie_id'] or '')
        require(match is not None and part==f'xl/charts/chart{match[1]}.xml','Series identity')
        root=ET.fromstring(source.xml.parts[part]);node=root.findall('.//c:ser',C)[int(match[2])-1]
        require(node.find('c:bubbleSize',C) is None,'Bubble series')
        group=next(parent for parent in root.iter() if node in list(parent))
        ids={n.get('val') for n in group.findall('c:axId',C)}
        axes=[axis for axis in root.findall('.//c:valAx',C) if axis.find('c:axId',C).get('val') in ids]
        require(len(axes)==1,'Value axis not uniquely associated to series')
        axis=axes[0];axis_id=axis.find('c:axId',C).get('val')
        fmt=axis.find('c:numFmt',C);require(fmt is not None,'No explicit axis format')
        axis_format=fmt.get('formatCode');format_divisor,multiplier=format_effect(axis_format)
        display=axis.find('c:dispUnits',C);display_name='none';axis_divisor=Decimal(1)
        if display is not None:
            builtin=display.find('c:builtInUnit',C);custom=display.find('c:custUnit',C)
            require((builtin is None)!=(custom is None),'Ambiguous display units')
            if builtin is not None:
                display_name=builtin.get('val','thousands');require(display_name in DIVISORS,'Unknown display unit')
                axis_divisor=DIVISORS[display_name]
            else:
                display_name='custom';axis_divisor=decimal(custom.get('val'))
                require(axis_divisor>0,'Nonpositive custom divisor')
        require(not (axis_divisor!=1 and multiplier!=1),'Mixed percentage/display units')
        require(not (axis_divisor!=1 and format_divisor!=1),'Two simultaneous divisors')
        divisor=axis_divisor*format_divisor
        name_node=node.find('c:tx/c:strRef/c:f',C);literal=node.find('c:tx/c:v',C)
        name_ref=name_node.text if name_node is not None else None
        if name_ref:
            ns,nc=ref_cells(name_ref);require(len(nc)==1,'Non-scalar name');label=source.text(ns,nc[0])
        else:label=literal.text if literal is not None else None
        require(isinstance(label,str) and label.strip(),'Unnamed series')
        cat=node.find('c:cat/c:numRef/c:f',C)
        if cat is None:cat=node.find('c:cat/c:strRef/c:f',C)
        val=node.find('c:val/c:numRef/c:f',C);require(cat is not None and val is not None,'Missing simple axes')
        cs,cc=ref_cells(cat.text);vs,vc=ref_cells(val.text);require(len(cc)==len(vc),'Unequal axes')
        keys=[]
        for address in cc:
            label_value=source.text(cs,address)
            if isinstance(label_value,str) and label_value.startswith('='):
                cached=source.number(cs,address)
                label_value=str(int(cached)) if cached is not None and cached==int(cached) else (str(cached) if cached is not None else None)
            keys.append(str(label_value) if label_value is not None else None)
        require(None not in keys and len(keys)==len(set(keys)),'Ambiguous categories')
        result=dict(label=label,name_ref=name_ref,category_ref=cat.text,value_ref=val.text,value_sheet=vs,
                    mapping=dict(zip(keys,vc)),axis_id=axis_id,axis_format=axis_format,display_unit=display_name,
                    divisor=divisor,multiplier=multiplier)
        self.bindings[key]=result;return result

    def review(self,row):
        context=norm(row['contexto_unidad_titulo']);unit=UNITS.get(context)
        if context in POPULATION_CONTEXTS:unit='percent_population_as_labelled'
        elif context=='%':
            profile=PERCENT_PROFILES.get((row['book'],row['bloque']))
            if profile and row['titulo']==profile[0]:unit=profile[1]
        if unit is None:return 'scope_or_unit_review',None
        if unit=='percent_interest_rate_as_published' and 'pib' in norm(row['serie_literal']):
            return 'series_scope_conflict',None
        try:binding=self.binding(row)
        except (ValueError,KeyError,IndexError,AttributeError,StopIteration):return 'scale_axis_or_binding_review',None
        if row['serie_literal']!=binding['label']:return 'series_identity_review',None
        cell=binding['mapping'].get(row['periodo_categoria_literal'])
        if cell is None:return 'category_review',None
        if row['hoja_valor']!=binding['value_sheet'] or row['celda_valor']!=cell:return 'source_binding_review',None
        source=self.sources[row['book']]
        try:
            raw=decimal(row['decimal_XML_texto'])
            if raw!=source.number(binding['value_sheet'],cell):return 'source_value_review',None
        except (ValueError,ArithmeticError):return 'source_value_review',None
        if row['tipo']!='posicion_serie' or row['unidad']!='ver_contexto_literal':return 'record_type_review',None
        canon=lambda s:''.join(c for c in norm(s) if c.isalnum())
        texts=source.sheet(row['bloque'])[1]
        if row['metrica_literal']!=row['titulo'] or not canon(row['titulo']):return 'metric_evidence_review',None
        title_proof=next(((a,t) for a,t in texts.items() if isinstance(t,str) and not t.startswith('=') and canon(row['titulo']) in canon(t)),None)
        if title_proof is None:return 'metric_evidence_review',None
        proof=next(((a,t) for a,t in texts.items() if isinstance(t,str) and not t.startswith('=') and ''.join(context.split()) in ''.join(norm(t).split())),None)
        if proof is None:return 'unit_evidence_review',None
        if unit=='million_source_units' and not any(t in norm(row['titulo']) for t in ['pensiones','personas','población']):return 'scope_or_unit_review',None
        fmtkey=(row['book'],binding['value_sheet'])
        if fmtkey not in self.formats:
            self.formats[fmtkey]={c.coordinate:c.number_format for r in source.workbook[binding['value_sheet']].iter_rows() for c in r if c.value is not None}
        cell_format=self.formats[fmtkey].get(cell,'General')
        try:cell_divisor,cell_multiplier=format_effect(cell_format)
        except ValueError:return 'cell_format_review',None
        # Cell display scaling does not necessarily define chart-axis scaling: reject conflict.
        if cell_divisor!=1:return 'cell_format_review',None
        if cell_multiplier!=1 and binding['multiplier']==1:return 'cell_axis_format_conflict',None
        if binding['multiplier']!=1 and not unit.startswith('percent_'):return 'percentage_unit_conflict',None
        try:
            with localcontext() as ctx:
                ctx.prec=60;value=decimal(str(raw*binding['multiplier']/binding['divisor']))
        except (ValueError,ArithmeticError):return 'conversion_precision_review',None
        result=(row['book'],row['row_number'],row['id'],row['bloque'],row['metrica_literal'],row['serie_id'],
                row['serie_literal'],row['periodo_categoria_literal'],unit,value,binding['value_sheet'],cell,proof[0],proof[1],
                binding['value_ref'],binding['category_ref'],binding['name_ref'],row['parte_grafico_imagen'],'explicit_axis_scale_or_scoped_percent_v1',
                raw,binding['axis_id'],binding['display_unit'],binding['axis_format'],cell_format,binding['divisor'],binding['multiplier'],row['contexto_unidad_titulo'])
        return 'usable_scaled',result


def collect(con):
    previous=set(con.execute('SELECT book,row_number FROM usable_keys').fetchall());require(len(previous)==9992,'Previous admissions changed')
    c=con.execute('SELECT s.* FROM staging_records s JOIN record_review r USING(book,row_number) WHERE r.numeric_candidate ORDER BY s.book,s.row_number')
    names=[n[0] for n in c.description];rows=c.fetchall();reviewer=ScaleReviewer();records=[];decisions=[]
    try:
        for raw in rows:
            row=dict(zip(names,raw))
            if (row['book'],row['row_number']) in previous:continue
            if row['tipo']!='posicion_serie':status='table_or_manual_review';record=None
            else:status,record=reviewer.review(row)
            decisions.append((row['book'],row['row_number'],status))
            if record is not None:records.append(record)
    finally:reviewer.close()
    require(len(decisions)==37221,'Numeric pending inventory changed')
    return records,decisions


DDL='''CREATE TABLE scale_observations(book VARCHAR,row_number INTEGER,record_id VARCHAR,block VARCHAR,metric_literal VARCHAR,
 series_id VARCHAR,series_literal VARCHAR,category_literal VARCHAR,unit VARCHAR,value DECIMAL(38,24),source_sheet VARCHAR,
 source_cell VARCHAR,unit_cell VARCHAR,unit_text VARCHAR,value_ref VARCHAR,category_ref VARCHAR,name_ref VARCHAR,chart_part VARCHAR,
 rule_id VARCHAR,raw_value DECIMAL(38,24),axis_id VARCHAR,display_unit VARCHAR,axis_format VARCHAR,cell_format VARCHAR,
 divisor DECIMAL(38,24),multiplier DECIMAL(38,24),scope_literal VARCHAR,PRIMARY KEY(book,row_number));
 CREATE TABLE scale_review(book VARCHAR,row_number INTEGER,status VARCHAR,PRIMARY KEY(book,row_number));
 CREATE VIEW scaled_usable_keys AS SELECT * FROM usable_keys UNION ALL SELECT book,row_number FROM scale_observations;
 CREATE VIEW scaled_numeric_queue AS SELECT s.*,r.status AS scale_reason FROM staging_records s JOIN scale_review r
 USING(book,row_number) WHERE r.status<>'usable_scaled';'''


def build(output):
    output=Path(output).resolve();output.relative_to(ROOT);require(not output.exists(),'Output exists')
    require(sha(PREVIOUS)==PREVIOUS_SHA,'Predecessor changed')
    output.mkdir(parents=True);db=output/'scaled.duckdb';shutil.copyfile(PREVIOUS,db)
    with duckdb.connect(str(db)) as con:
        rows,decisions=collect(con);con.execute('BEGIN');con.execute(DDL)
        if rows:con.executemany('INSERT INTO scale_observations VALUES ('+','.join('?' for _ in FIELDS)+')',rows)
        con.executemany('INSERT INTO scale_review VALUES (?,?,?)',decisions);con.execute('COMMIT')
    result=dict(status='built_pending_validation',added=len(rows),admitted=9992+len(rows),numeric_pending=37221-len(rows),
                previous_sha256=PREVIOUS_SHA,code_sha256=sha(Path(__file__)),db_sha256=sha(db),rag_integrated=False)
    (output/'manifest.json').write_text(json.dumps(result,indent=2),encoding='utf-8');return result


def check_new(con):
    expected,decisions=collect(con)
    actual=con.execute('SELECT * FROM scale_observations').fetchall()
    require(len(actual)==len(expected) and set(actual)==set(expected),'Converted observation mismatch')
    actual_review=con.execute('SELECT * FROM scale_review').fetchall()
    require(len(actual_review)==len(decisions) and set(actual_review)==set(decisions),'Scale classification mismatch')
    for view,expr in [('scaled_usable_keys','SELECT * FROM usable_keys UNION ALL SELECT book,row_number FROM scale_observations'),
                     ('scaled_numeric_queue',"SELECT s.*,r.status AS scale_reason FROM staging_records s JOIN scale_review r USING(book,row_number) WHERE r.status<>'usable_scaled'")]:
        require([n[0] for n in con.execute(f'SELECT * FROM {view} LIMIT 0').description]==[n[0] for n in con.execute(expr+' LIMIT 0').description],'Scale view schema mismatch')
        require(not con.execute(f'(SELECT * FROM {view} EXCEPT ALL ({expr})) UNION ALL (({expr}) EXCEPT ALL SELECT * FROM {view}) LIMIT 1').fetchone(),'Scale view content/multiplicity mismatch')
    require(con.execute('SELECT count(*) FROM (SELECT book,row_number FROM scaled_usable_keys GROUP BY ALL HAVING count(*)>1)').fetchone()[0]==0,'Admission collision')
    return expected


def validate(db):
    previous=validate_previous(db)
    with duckdb.connect(str(db),read_only=True) as con:
        rows=check_new(con)
        stats=con.execute('SELECT unit,divisor,multiplier,count(*) FROM scale_observations GROUP BY ALL ORDER BY count(*) DESC').fetchall()
        statuses=con.execute('SELECT status,count(*) FROM scale_review GROUP BY status ORDER BY count(*) DESC').fetchall()
        blocks=con.execute('SELECT count(*) FROM (SELECT DISTINCT book,block FROM scale_observations)').fetchone()[0]
    return dict(status='pass',added=len(rows),admitted=9992+len(rows),numeric_pending=37221-len(rows),blocks=blocks,
                conversions=[[u,str(d),str(m),n] for u,d,m,n in stats],review=statuses,
                previous_validation=previous['status'],db_sha256=sha(Path(db)),code_sha256=sha(Path(__file__)),rag_integrated=False)


def query(db,*,book,block,series_id,category):
    require(book in ['2025','2026'] and all(isinstance(t,str) and t.strip() for t in [block,series_id,category]),'Exact identity required')
    with duckdb.connect(str(db),read_only=True) as con:
        c=con.execute('''SELECT t.*,s.xlsx_path,s.xlsx_sha256,s.pdf_path,s.pdf_sha256 FROM scale_observations t
           JOIN sources s ON s.book=t.book WHERE t.book=? AND block=? AND series_id=? AND category_literal=?''',[book,block,series_id,category])
        names=[n[0] for n in c.description];rows=c.fetchall()
    if len(rows)!=1:raise LookupError('No unique converted/scoped observation; no fallback')
    return {n:str(v) if isinstance(v,Decimal) else v for n,v in zip(names,rows[0])}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    b=sub.add_parser('build');b.add_argument('--output-dir',type=Path,required=True)
    b=sub.add_parser('validate');b.add_argument('--db',type=Path,required=True);b.add_argument('--output',type=Path,required=True)
    a=p.parse_args(argv)
    if a.command=='build':report=build(a.output_dir)
    else:
        require(not a.output.exists(),'Report exists')
        try:report=validate(a.db)
        except Exception as exc:report=dict(status='fail',error=str(exc))
        a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2));return int(report['status']=='fail')


if __name__=='__main__':raise SystemExit(main())
