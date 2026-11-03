"""Literal chart categories, numeric scalar names and exact shared-index evidence."""
from __future__ import annotations
import argparse
from decimal import Decimal
import json
import math
from pathlib import Path
import re
import shutil
import xml.etree.ElementTree as ET

import duckdb
from experiments.resolved_semantics import Resolver,FIELDS as PRIOR_FIELDS,canon,validate as validate_previous
from experiments.scaled_semantics import DIVISORS,format_effect
from experiments.bulk_semantics import C,norm,ref_cells
from experiments.semantic_families import decimal
from experiments.excel_pilot import ROOT,sha
from experiments.universal_database import require

PREVIOUS=ROOT/'evaluation/resolved_semantics_2026-10-31/run_v3/resolved.duckdb'
PREVIOUS_SHA='b61219b198149cae247789e1624e789a2e74b91317f04b9017aa25197c883a1a'
FIELDS=PRIOR_FIELDS+'category_basis name_basis semantic_scope title_resolution'.split()
# Explicit reviewed typo aliases only, never fuzzy title matching.
TITLE_ALIASES={
 ('2026','Gráfico 14B'):('Comparativa de tasas de dependencia','GRÁFICO 14B. COMPRATIVA DE TASAS DE DEPENDENCIA'),
 ('2026','Anexo I. Gr 5'):('Evolución de las proyecciones del número de hijos por mujer de la AIReF','ANEXO I. GRÁFICO 5. EVOLUCIÓN ED LAS PROYECCIONES DEL NÚMERO DE HIJOS POR MUJER DE LA AIREF'),
 ('2026','Anexo I. Gr 9B'):('Esperanza de vida a los 65 años. Según sexo','ANEXO I. GRÁFICO 9B. ESPERANZA DE VIDA AL LOS 65 AÑOS. SEGÚN SEXO'),
}
PLOT_PERCENT={
 ('2025','Gráfico 34'):'COMPARACIÓN DISTRIBUCIÓN SALARIAL 2018-2023 (12 PAGAS)',
 ('2025','Gráfico 35'):'COMPARACIÓN DISTRIBUCIÓN SALARIAL 2020-2023 (12 PAGAS)',
 ('2026','Gráfico 10'):'Distribución de las bases de cotización mensuales, 2018, 2023 y 2024',
 ('2026','Anexo I. Gr 14'):'Perfil etario de los flujos de inmigración',
 ('2026','Anexo I. Gr 3A'):'Proporción de mujeres que tienen hijos',
 ('2026','Anexo I. Gr 8A'):'Proporción de mujeres que tienen hijos. Según nacionalidad',
}


def points(node):
    values={}
    for p in node.findall('c:pt',C):
        index=int(p.get('idx'));require(index>=0 and index not in values,'Duplicate/invalid literal/cache index')
        value=p.find('c:v',C);require(value is not None and value.text is not None,'Empty point value')
        values[index]=value.text
    return values


class IndexedResolver(Resolver):
    def __init__(self):
        super().__init__();self.title_modes={};self.failed_bindings=set()

    def binding(self,row):
        key=(row['book'],row['parte_grafico_imagen'],row['serie_id'])
        if key in self.bindings:return self.bindings[key]
        source=self.sources[row['book']];part=row['parte_grafico_imagen']
        m=re.fullmatch(r'chart([0-9]+)_s([1-9][0-9]*)',row['serie_id'] or '')
        require(m is not None and part==f'xl/charts/chart{m[1]}.xml','Series identity')
        root=ET.fromstring(source.xml.parts[part]);node=root.findall('.//c:ser',C)[int(m[2])-1]
        require(node.find('c:bubbleSize',C) is None,'Bubble series unsupported')
        parent=next(p for p in root.iter() if node in list(p));ids={x.get('val') for x in parent.findall('c:axId',C)}
        axes=[a for a in root.findall('.//c:valAx',C) if a.find('c:axId',C).get('val') in ids]
        require(len(axes)==1,'Ambiguous value axis');axis=axes[0]
        fmt=axis.find('c:numFmt',C);require(fmt is not None,'Missing axis format')
        axis_format=fmt.get('formatCode');fd,multiplier=format_effect(axis_format)
        display=axis.find('c:dispUnits',C);ad=Decimal(1);display_name='none'
        if display is not None:
            bi=display.find('c:builtInUnit',C);cu=display.find('c:custUnit',C)
            require((bi is None)!=(cu is None),'Ambiguous display units')
            if bi is not None:
                display_name=bi.get('val','thousands');require(display_name in DIVISORS,'Unknown display unit');ad=DIVISORS[display_name]
            else:display_name='custom';ad=decimal(cu.get('val'));require(ad>0,'Invalid divisor')
        require(not(ad!=1 and (fd!=1 or multiplier!=1)),'Mixed axis scaling')
        name=node.find('c:tx/c:strRef/c:f',C);literal=node.find('c:tx/c:v',C)
        name_ref=name.text if name is not None else None
        if name_ref:
            ns,nc=ref_cells(name_ref);require(len(nc)==1,'Nonscalar name')
            label=source.text(ns,nc[0]);name_basis='source_text_scalar'
            if isinstance(label,str) and label.startswith('='):
                label=source.number(ns,nc[0]);name_basis='source_numeric_formula_cache'
            if isinstance(label,(int,float,Decimal)) and not isinstance(label,bool):
                require(Decimal(str(label)).is_finite(),'Nonfinite series name')
                label=str(int(label)) if label==int(label) else str(label);name_basis='source_numeric_scalar'
        else:label=literal.text if literal is not None else None;name_basis='literal_chart_name'
        require(isinstance(label,str) and label.strip(),'Unnamed series')
        val=node.find('c:val/c:numRef',C);require(val is not None,'No referenced source values')
        vr=val.find('c:f',C);require(vr is not None,'Missing value ref');vs,vc=ref_cells(vr.text)
        cat=node.find('c:cat/c:numRef',C)
        if cat is None:cat=node.find('c:cat/c:strRef',C)
        category_basis='source_cells';cr_text=None;cs=None;cc=None;cp=None
        if cat is not None:
            cr=cat.find('c:f',C);require(cr is not None,'Missing category ref');cr_text=cr.text;cs,cc=ref_cells(cr_text)
            categories=[self.category(source,cs,a) for a in cc]
        else:
            cat=node.find('c:cat/c:strLit',C)
            if cat is None:cat=node.find('c:cat/c:numLit',C)
            require(cat is not None,'No explicit category source')
            cp=points(cat);count=cat.find('c:ptCount',C)
            require(count is not None and 0<int(count.get('val'))<=20001,'Missing/invalid literal point count')
            size=int(count.get('val'));require(set(cp)==set(range(size)),'Literal categories not explicit/contiguous')
            categories=[cp[i] for i in range(size)];category_basis='literal_chart_points'
            cr_text=f'xml:{part}/cat/{cat.tag.split("}")[-1]}'
        common=min(len(vc),len(categories));require(common>0,'Empty shared axis')
        partial=len(vc)!=len(categories)
        valid=set(range(common))
        if partial:
            if cp is None:
                cache=cat.find('c:numCache',C)
                if cache is None:cache=cat.find('c:strCache',C)
                require(cache is not None,'Missing category cache');cp=points(cache)
            cache=val.find('c:numCache',C);require(cache is not None,'Missing value cache');vp=points(cache)
            valid=set()
            for i in range(common):
                if categories[i] is None:continue
                try:original_value=source.number(vs,vc[i])
                except ValueError:continue
                if original_value is None:continue
                require(i in cp and i in vp,'Unproved shared cache index')
                require(decimal(vp[i])==original_value,'Shared cached value mismatch')
                if category_basis=='source_cells':
                    try:original_numeric_category=source.number(cs,cc[i])
                    except ValueError:original_numeric_category=None
                    # Excel dates are numeric serials even when the reader exposes datetime.
                    # Verify their raw serial; retain the original formatted category separately.
                    if original_numeric_category is not None:
                        require(decimal(cp[i])==original_numeric_category,'Shared numeric category mismatch')
                    else:require(cp[i]==categories[i],'Shared text category mismatch')
                else:require(cp[i]==categories[i],'Literal category mismatch')
                valid.add(i)
            require(valid,'No verified shared indices')
        result=dict(label=label,name_ref=name_ref,category_ref=cr_text,value_ref=vr.text,value_sheet=vs,
                    cells=vc[:common],categories=categories[:common],valid_indices=valid,axis_id=axis.find('c:axId',C).get('val'),
                    axis_format=axis_format,display_unit=display_name,divisor=ad*fd,multiplier=multiplier,
                    mode='cache_verified_shared_indices' if partial else 'complete_indexed_axis',
                    category_basis=category_basis,name_basis=name_basis)
        self.bindings[key]=result;return result

    def title_proof(self,row,texts):
        proof=super().title_proof(row,texts)
        if proof:return proof
        alias=TITLE_ALIASES.get((row['book'],row['bloque']))
        if alias is None or row['titulo']!=alias[0] or row['metrica_literal']!=alias[0]:return None
        found=next(((a,t) for a,t in texts if canon(t)==canon(alias[1])),None)
        if found:
            self.title_modes[(row['book'],row['bloque'])]='explicit_reviewed_typo_alias'
            return [found]
        return None

    def unit_spec(self,row,texts,binding):
        # TIR is the measured return; life-expectancy scenarios do not change its unit to years.
        if re.search(r'\btir\b',norm(row['titulo'])) and binding['multiplier']==100:
            proof=next(((a,t) for a,t in texts if re.search(r'\btir\b',norm(t))),None)
            if proof:return 'TIR + eje porcentual','percent_internal_return_as_published','return_title_and_percent_axis',proof
        spec=super().unit_spec(row,texts,binding)
        if spec is not None:return spec
        if PLOT_PERCENT.get((row['book'],row['bloque']))!=row['titulo']:return None
        if binding['multiplier']!=100 or binding['divisor']!=1:return None
        # The numeric display is explicit; its economic denominator is deliberately not asserted.
        return 'porcentaje del eje, denominador no inferido','percent_plot_display','associated_axis_percent_only',(
                'axis_format:'+binding['axis_id'],binding['axis_format'])

    def review(self,row):
        key=(row['book'],row['parte_grafico_imagen'],row['serie_id'])
        if key in self.failed_bindings:return 'binding_unresolved',None
        try:b=self.binding(row)
        except (ValueError,ArithmeticError,KeyError,IndexError,AttributeError,StopIteration):
            self.failed_bindings.add(key);return 'binding_unresolved',None
        m=re.fullmatch(re.escape(row['serie_id'])+r'_p([0-9]+)',row['id'] or '')
        if m is None:return 'point_identity_review',None
        index=int(m[1])
        if index>=len(b['cells']) or index not in b['valid_indices'] or b['categories'][index] is None or not str(b['categories'][index]).strip():
            return 'unproved_category_point',None
        status,result=super().review(row)
        if result is None:return status,None
        scope='literal_plot_display_only' if result[8]=='percent_plot_display' else 'source_quantity_as_labelled'
        return 'usable_indexed',result+(b['category_basis'],b['name_basis'],scope,
               self.title_modes.get((row['book'],row['bloque']),'literal_title_evidence'))


def collect(con):
    require(con.execute('SELECT count(*) FROM resolved_usable_keys').fetchone()[0]==21932,'Previous count changed')
    c=con.execute('SELECT * FROM resolved_numeric_queue ORDER BY book,row_number');names=[n[0] for n in c.description]
    data=c.fetchall();require(len(data)==25281,'Pending inventory changed')
    resolver=IndexedResolver();rows=[];decisions=[]
    try:
        for values in data:
            row=dict(zip(names,values))
            if row['tipo']!='posicion_serie':status,result='table_or_manual_review',None
            else:status,result=resolver.review(row)
            decisions.append((row['book'],row['row_number'],status))
            if result is not None:rows.append(result)
    finally:resolver.close()
    return rows,decisions


def build(output):
    output=Path(output).resolve();output.relative_to(ROOT);require(not output.exists(),'Output exists')
    require(sha(PREVIOUS)==PREVIOUS_SHA,'Predecessor changed')
    output.mkdir(parents=True);db=output/'indexed.duckdb';shutil.copyfile(PREVIOUS,db)
    with duckdb.connect(str(db)) as con:
        rows,decisions=collect(con);con.execute('BEGIN')
        con.execute('CREATE TABLE indexed_observations AS SELECT *,NULL::VARCHAR AS category_basis,NULL::VARCHAR AS name_basis,NULL::VARCHAR AS semantic_scope,NULL::VARCHAR AS title_resolution FROM resolved_observations WHERE false')
        con.execute('ALTER TABLE indexed_observations ADD PRIMARY KEY(book,row_number)')
        if rows:con.executemany('INSERT INTO indexed_observations VALUES ('+','.join('?' for _ in FIELDS)+')',rows)
        con.execute('CREATE TABLE indexed_review(book VARCHAR,row_number INTEGER,status VARCHAR,PRIMARY KEY(book,row_number))')
        con.executemany('INSERT INTO indexed_review VALUES (?,?,?)',decisions)
        con.execute('CREATE VIEW indexed_usable_keys AS SELECT * FROM resolved_usable_keys UNION ALL SELECT book,row_number FROM indexed_observations')
        con.execute("CREATE VIEW indexed_numeric_queue AS SELECT s.*,r.status AS indexed_reason FROM resolved_numeric_queue s JOIN indexed_review r USING(book,row_number) WHERE r.status<>'usable_indexed'")
        con.execute('COMMIT')
    report=dict(status='built_pending_validation',added=len(rows),admitted=21932+len(rows),numeric_pending=25281-len(rows),
                previous_sha256=PREVIOUS_SHA,code_sha256=sha(Path(__file__)),db_sha256=sha(db),rag_integrated=False)
    (output/'manifest.json').write_text(json.dumps(report,indent=2),encoding='utf-8');return report


def check_new(con):
    expected,decisions=collect(con);actual=con.execute('SELECT * FROM indexed_observations').fetchall()
    require(len(actual)==len(expected) and set(actual)==set(expected),'Indexed content mismatch')
    actual=con.execute('SELECT * FROM indexed_review').fetchall()
    require(len(actual)==25281 and set(actual)==set(decisions),'Indexed classification mismatch')
    for view,expr in [('indexed_usable_keys','SELECT * FROM resolved_usable_keys UNION ALL SELECT book,row_number FROM indexed_observations'),
      ('indexed_numeric_queue',"SELECT s.*,r.status AS indexed_reason FROM resolved_numeric_queue s JOIN indexed_review r USING(book,row_number) WHERE r.status<>'usable_indexed'")]:
        require([n[0] for n in con.execute(f'SELECT * FROM {view} LIMIT 0').description]==[n[0] for n in con.execute(expr+' LIMIT 0').description],'Indexed view schema mismatch')
        require(not con.execute(f'(SELECT * FROM {view} EXCEPT ALL ({expr})) UNION ALL (({expr}) EXCEPT ALL SELECT * FROM {view}) LIMIT 1').fetchone(),'Indexed view content/multiplicity mismatch')
    require(con.execute('SELECT count(*) FROM (SELECT book,row_number FROM indexed_usable_keys GROUP BY ALL HAVING count(*)>1)').fetchone()[0]==0,'Admission collision')
    return expected


def validate(db):
    previous=validate_previous(db)
    with duckdb.connect(str(db),read_only=True) as con:
        rows=check_new(con)
        stats=con.execute('SELECT unit,category_basis,name_basis,semantic_scope,count(*) FROM indexed_observations GROUP BY ALL ORDER BY count(*) DESC').fetchall()
        blocks=con.execute('SELECT count(*) FROM (SELECT DISTINCT book,block FROM indexed_observations)').fetchone()[0]
        reasons=con.execute('SELECT status,count(*) FROM indexed_review GROUP BY status ORDER BY count(*) DESC').fetchall()
    return dict(status='pass',added=len(rows),admitted=21932+len(rows),numeric_pending=25281-len(rows),blocks=blocks,
                stats=stats,review=reasons,previous_validation=previous['status'],code_sha256=sha(Path(__file__)),db_sha256=sha(Path(db)),rag_integrated=False)


def query(db,*,book,block,series_id,point_index):
    require(book in ['2025','2026'] and all(isinstance(t,str) and t.strip() for t in [block,series_id]) and type(point_index) is int and point_index>=0,'Exact identity and integer point_index required')
    with duckdb.connect(str(db),read_only=True) as con:
        c=con.execute('''SELECT t.*,s.xlsx_path,s.xlsx_sha256,s.pdf_path,s.pdf_sha256 FROM indexed_observations t
           JOIN sources s ON s.book=t.book WHERE t.book=? AND block=? AND series_id=? AND point_index=?''',[book,block,series_id,point_index])
        names=[n[0] for n in c.description];rows=c.fetchall()
    if len(rows)!=1:raise LookupError('No unique indexed point; no fallback')
    return {n:str(v) if isinstance(v,Decimal) else v for n,v in zip(names,rows[0])}


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
