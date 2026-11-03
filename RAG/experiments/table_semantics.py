"""Typed published tables and bounded interest-rate context, isolated from the RAG."""
from __future__ import annotations
import argparse
from decimal import Decimal,localcontext
import json
from pathlib import Path
import re
import shutil
import xml.etree.ElementTree as ET
import duckdb
from experiments.grid_semantics import GridResolver,FIELDS as CHART_FIELDS,validate as validate_previous
from experiments.resolved_semantics import canon
from experiments.scaled_semantics import format_effect
from experiments.excel_pilot import ROOT,sha,PDF_HASHES
from experiments.semantic_families import decimal,M
from experiments.universal_database import require

PREVIOUS=ROOT/'evaluation/grid_semantics_2026-10-31/run_v1/grid.duckdb'
PREVIOUS_SHA='195e9af295d9b86f8792680b0413217284ab728def1c437db402e9b6c20b805c'
PROFILES=Path(__file__).with_name('table_profiles.json')
PROFILES_SHA='d26d0e309e1261bff346a7971496d8b1ed5e2eb8e9ceccbcaaaa7aedae391eac'
TABLES={(p['book'],p['block']):p for p in json.loads(PROFILES.read_text(encoding='utf-8'))}
TABLE_FIELDS='book row_number source_id block metric series_literal category_literal unit raw_value value multiplier divisor source_sheet source_cell metric_cell series_cell category_cell title_cell title_literal formula role scope evidence cell_format'.split()
CHART_KEYS={('2025','Gráfico 57'),('2025','Gráfico 75'),('2026','Gráfico 21')}


def label(value):
    if isinstance(value,(int,float,Decimal)) and not isinstance(value,bool):return str(int(value)) if value==int(value) else str(value)
    return str(value).strip() if value is not None else None


class TableResolver(GridResolver):
    def __init__(self):
        super().__init__();self.table_merges={};self.table_conflicts={}

    def unit_spec(self,row,texts,binding):
        if (row['book'],row['bloque']) in CHART_KEYS and row['serie_literal'] in ['Tipo de interés implícito','Tipo interés nuevas emisiones']:
            require(binding['divisor']==1 and binding['multiplier']==1,'Projection rate scale conflict')
            if row['book']=='2025':
                proof=self.pdf('2025',118,['el coste medio de las nuevas emisiones aumentará','3,8% en 2070','del mínimo del 1,9% hasta aproximarse al 3,9%'])
                caption=self.pdf('2025',119 if row['bloque']=='Gráfico 57' else 133,[row['bloque'].upper()+'. PROYECCIÓN TIPO MEDIO DE EMISIÓN Y TIPO IMPLÍCITO'])
            else:
                proof=self.pdf('2026',66,['tipo de interés implícito de la deuda aumentaría','2,7 % hasta el 3,8 %','GRÁFICO 21. PROYECCIÓN TIPO MEDIO DE EMISIÓN Y TIPO IMPLÍCITO']);caption=proof
            evidence=dict(unit_context=proof,caption=caption);self.supplements[row['book'],row['bloque']]=evidence
            return 'tipo de interés porcentual del contexto de la proyección','percent_interest_rate_as_published','reviewed_projection_pdf_context',('pdf_projection_context',json.dumps(evidence,ensure_ascii=False))
        return super().unit_spec(row,texts,binding)

    def table(self,row):
        p=TABLES[row['book'],row['bloque']];source=self.sources[row['book']];sheet=row['bloque'];ws=source.workbook[sheet]
        require(source.text(sheet,p['title_cell'])==p['title'],'Table source title changed')
        if 'universal_title' in p:require(row['titulo']==p['universal_title'],'Universal table title changed')
        else:require(canon(row['titulo']) in canon(p['title']),'Resource title conflict')
        cell=row['celda_valor'];match=re.fullmatch('([A-Z]+)([0-9]+)',cell or '');require(match is not None and row['hoja_valor']==sheet,'Table source identity')
        col,rr=match[1],int(match[2]);family=p['family'];evidence={};role='published_level';metric_cell=None;series_cell=None;category_cell=None;scope='source_quantity_as_labelled'
        text=lambda a:label(source.text(sheet,a))
        if family=='climate':
            if col in 'IJKL':return 'auxiliary_table_cells',None
            require(col in 'DEFG' and 7<=rr<=13,'Climate table outside published range')
            conflicts=[]
            for published,auxiliary in zip('DEFG','IJKL'):
                left=source.number(sheet,published+'7');right=source.number(sheet,auxiliary+'7')
                require(left is not None and right is not None,'Missing climate base')
                if abs(left-right)>Decimal('1e-12'):
                    conflicts.append(dict(published_cell=published+'7',published_value=str(left),auxiliary_cell=auxiliary+'7',auxiliary_value=str(right)))
            if conflicts:
                self.table_conflicts[row['book'],sheet]=json.dumps(conflicts,ensure_ascii=False,sort_keys=True)
                return 'published_base_auxiliary_conflict',None
            metric_cell='D5';series_cell=f'C{rr}';category_cell=col+'6'
            metric=text(metric_cell);series=text(series_cell);category=text(category_cell)
            require('%' in metric,'Missing climate unit')
            gdp='PIB)' in metric.replace(' ','')
            role='published_base_level' if rr==7 else 'scenario_difference_from_base'
            unit=('percent_gdp' if gdp else 'percent_rate_as_labelled') if rr==7 else ('percentage_points_gdp' if gdp else 'percentage_points_rate_as_labelled')
            require(text('B8')=='Diferencias sobre el escenario base','Missing differences label')
            evidence['difference_label']='B8';evidence['source_metric']=metric
        elif family=='fiscal_scenarios':
            require(rr in range(6,10) and col in 'DEFHIJLMN','Fiscal table outside range')
            metric_cell=f'B{rr}';category_cell=col+'5';series_cell=next(start+'4' for group,start in [('DEF','D'),('HIJ','H'),('LMN','L')] if col in group)
            metric=text(metric_cell);series=text(series_cell);category=text(category_cell);unit='percent_gdp'
            require(text('B5')=='(en % PIB)','Fiscal unit evidence missing');evidence['unit_cell']='B5'
        elif family=='demography':
            require(col in ('DEFGHI' if row['book']=='2025' else 'DEFGHIJ'),'Demography column outside range')
            merge_key=row['book'],sheet
            if merge_key not in self.table_merges:
                root=ET.fromstring(source.xml.parts[source.xml.sheets[sheet]])
                self.table_merges[merge_key]=[m.get('ref') for m in root.findall('m:mergeCells/m:mergeCell',M)]
            ranges=[]
            for ref in self.table_merges[merge_key]:
                bounds=re.fullmatch(r'B([0-9]+):B([0-9]+)',ref)
                if bounds and int(bounds[1])<=rr<=int(bounds[2]):ranges.append((ref,int(bounds[1])))
            require(len(ranges)==1,'Metric must have one explicit merged B range')
            metric_cell='B'+str(ranges[0][1]);series_cell='C'+str(rr);category_cell=col+('4' if row['book']=='2025' else '5')
            metric=text(metric_cell);series=text(series_cell);category=text(category_cell);evidence['metric_merge']=ranges[0][0]
            if 'Miles' in metric:unit='thousand_persons'
            elif 'hijos por mujer' in metric:unit='children_per_woman'
            elif 'Años' in metric:unit='years'
            elif 'Tasa de dependencia' in metric and '%' in metric:unit='percent_dependency_as_labelled'
            else:return 'table_unit_unresolved',None
            role='published_period_average' if category and 'Promedio' in category else 'published_level'
        elif family=='labour':
            require(rr in range(7,12) and col in ('CDEFGH' if row['book']=='2025' and sheet=='Cuadro 6' else 'CDEFGHIJK'),'Labour table outside range')
            width=2 if row['book']=='2025' and sheet=='Cuadro 6' else 3
            ci=ord(col)-ord('C');start=chr(ord('C')+(ci//width)*width)
            metric_cell=f'B{rr}';series_cell=col+'6';category_cell=start+'5'
            metric=text(metric_cell);series=text(series_cell);category=text(category_cell)
            require('% Y PUNTOS PORCENTUALES' in p['title'],'Mixed unit caption changed')
            unit='percentage_points_gdp' if '% PIB' in metric else 'percent_deviation_from_no_measures'
            role='deviation_from_no_measures';scope='counterfactual_deviation_as_published'
        elif family=='pension_scenarios':
            require(col in 'DEF' and 5<=rr<=18,'Pension sensitivity table outside range')
            metric_cell=p['title_cell'];series_cell=f'B{rr}';category_cell=col+'4'
            metric='Gasto en pensiones' if sheet=='Cuadro 24' else 'Deuda';series=text(series_cell);category=text(category_cell)
            require('DIFERENCIAS RESPECTO AL ESCENARIO BASE (%PIB)' in p['title'],'Pension difference unit changed')
            unit='percentage_points_gdp';role='scenario_difference_from_base';evidence['scenario_description_cell']='C'+str(rr);evidence['scenario_description']=text('C'+str(rr))
        elif family=='age_structure':
            require(rr in range(6,9) and col in ('DEFG' if row['book']=='2025' else 'DEFGHI'),'Age table outside range')
            metric_cell=f'C{rr}';series_cell='D10' if row['book']=='2025' else 'C10';category_cell=col+'5'
            metric=text(metric_cell);series=text(series_cell);category=text(category_cell);unit='percent_age_population';role='population_share';evidence['series_kind']='source_citation'
        elif family=='resource_grid':
            require(row['parte_grafico_imagen'] is None and row['tipo']=='posicion_serie' and 5<=rr<=9,'Not a bounded resource-grid position')
            metric_cell=p['title_cell'];series_cell=f'B{rr}';category_cell=col+'4'
            metric=row['metrica_literal'];series=text(series_cell);category=label(source.number(sheet,category_cell));unit='percent_gdp'
            require(canon(metric) in canon(p['title']),'Resource metric mismatch')
            role='plot_reference_level' if series and series.startswith('Nivel ') else 'published_series_level'
            if role=='plot_reference_level':scope='reference_line_not_independent_forecast'
        else:return 'table_family_unresolved',None
        require(all(isinstance(v,str) and v.strip() for v in [metric,series,category]),'Table label missing')
        require(canon(metric)==canon(row['metrica_literal']) and canon(series)==canon(row['serie_literal']) and canon(category)==canon(row['periodo_categoria_literal']),'Table semantic identity mismatch')
        raw=source.number(sheet,cell);require(raw is not None and raw==decimal(row['decimal_XML_texto']),'Table raw value mismatch')
        fmt=ws[cell].number_format;divisor,multiplier=format_effect(fmt)
        require(divisor==1 and (multiplier==1 or unit.startswith('percent')),'Table format/unit conflict')
        formula=source.text(sheet,cell);formula=formula if isinstance(formula,str) and formula.startswith('=') else None
        if formula!=(row['formula_texto'] or None):
            require(family=='resource_grid' and role=='plot_reference_level' and not row['formula_texto'] and formula,'Formula identity changed')
            ref=re.fullmatch(r'=([A-Z]+[0-9]+)',formula);require(ref is not None,'Unsupported reference-line formula')
            target=source.number(sheet,ref[1]);require(target is not None and abs(raw-target)<=Decimal('1e-12'),'Reference-line cache mismatch')
            evidence.update(universal_formula=None,formula_resolution='original_formula_absent_from_universal',formula_inputs={ref[1]:str(target)})
        if family=='climate' and formula:
            m=re.fullmatch(r'=([I-L][0-9]+)(?:-([I-L][0-9]+))?',formula);require(m is not None,'Unsupported climate formula')
            expected=source.number(sheet,m[1]);refs={m[1]:str(expected)}
            if m[2]:
                require(rr>7 and m[1]==chr(ord(col)+5)+str(rr) and m[2]==chr(ord(col)+5)+'7','Wrong scenario/base formula references')
                refs[m[2]]=str(source.number(sheet,m[2]));expected-=source.number(sheet,m[2])
            else:require(rr==7 and m[1]==chr(ord(col)+5)+'7','Wrong base formula reference')
            require(abs(expected-raw)<=Decimal('1e-12'),'Climate formula cache mismatch')
            evidence['formula_inputs']=refs;evidence['formula_residual']=str(raw-expected)
        with localcontext() as ctx:ctx.prec=60;value=decimal(str(raw*multiplier/divisor))
        evidence.update(family=family,unit=unit,metric_cell=metric_cell,series_cell=series_cell,category_cell=category_cell,title_cell=p['title_cell'])
        result=(row['book'],row['row_number'],row['id'],sheet,metric,series,category,unit,raw,value,multiplier,divisor,sheet,cell,metric_cell,series_cell,category_cell,p['title_cell'],p['title'],formula,role,scope,json.dumps(evidence,ensure_ascii=False,sort_keys=True),fmt)
        return 'usable_table',result


def collect(con):
    require(sha(PROFILES)==PROFILES_SHA,'Table profiles changed')
    require(con.execute('SELECT count(*) FROM grid_usable_keys').fetchone()[0]==34416,'Previous admissions changed')
    c=con.execute('SELECT * FROM grid_numeric_queue ORDER BY book,row_number');names=[n[0] for n in c.description];data=c.fetchall()
    require(len(data)==12797,'Pending inventory changed');resolver=TableResolver();tables=[];charts=[];decisions=[];seen=set()
    try:
        for values in data:
            row=dict(zip(names,values));key=row['book'],row['bloque'];status=row['grid_reason'];record=None
            if key in TABLES:
                try:status,record=resolver.table(row)
                except (ValueError,ArithmeticError,KeyError,AttributeError,TypeError):status,record='table_evidence_review',None
                if record is not None:
                    physical=record[0],record[12],record[13]
                    if physical in seen:status,record='duplicate_table_source_cell',None
                    else:seen.add(physical);tables.append(record)
            elif key in CHART_KEYS and row['grid_reason']=='unit_still_unresolved':
                status,record=resolver.review(row)
                if record is not None:charts.append(record)
            decisions.append((row['book'],row['row_number'],status))
        pages=[(b,p,PDF_HASHES[b],t) for (b,p),t in sorted(resolver.pdf_pages.items())]
        conflicts=[(b,s,e) for (b,s),e in sorted(resolver.table_conflicts.items())]
    finally:resolver.close()
    return tables,charts,decisions,pages,conflicts


def build(output):
    require(not output.exists(),'Output exists');require(sha(PREVIOUS)==PREVIOUS_SHA,'Previous database changed')
    output.mkdir(parents=True);db=output/'tables.duckdb';shutil.copyfile(PREVIOUS,db)
    with duckdb.connect(str(db)) as con:
        tables,charts,decisions,pages,conflicts=collect(con)
        types=['DECIMAL(38,24)' if n in ['raw_value','value','multiplier','divisor'] else ('INTEGER' if n=='row_number' else 'VARCHAR') for n in TABLE_FIELDS]
        con.execute('CREATE TABLE table_observations('+','.join(n+' '+t for n,t in zip(TABLE_FIELDS,types))+',PRIMARY KEY(book,row_number),UNIQUE(book,source_sheet,source_cell))')
        if tables:con.executemany('INSERT INTO table_observations VALUES ('+','.join('?' for _ in TABLE_FIELDS)+')',tables)
        con.execute('CREATE TABLE context_observations AS SELECT * FROM grid_observations WHERE false')
        if charts:con.executemany('INSERT INTO context_observations VALUES ('+','.join('?' for _ in CHART_FIELDS)+')',charts)
        con.execute('CREATE TABLE table_review(book VARCHAR,row_number INTEGER,status VARCHAR,PRIMARY KEY(book,row_number))');con.executemany('INSERT INTO table_review VALUES (?,?,?)',decisions)
        con.execute('CREATE TABLE table_pdf_evidence(book VARCHAR,page INTEGER,pdf_sha256 VARCHAR,text VARCHAR,PRIMARY KEY(book,page))')
        if pages:con.executemany('INSERT INTO table_pdf_evidence VALUES (?,?,?,?)',pages)
        con.execute('CREATE TABLE table_conflicts(book VARCHAR,block VARCHAR,evidence VARCHAR,PRIMARY KEY(book,block))')
        if conflicts:con.executemany('INSERT INTO table_conflicts VALUES (?,?,?)',conflicts)
        con.execute('CREATE VIEW table_usable_keys AS SELECT * FROM grid_usable_keys UNION ALL SELECT book,row_number FROM table_observations UNION ALL SELECT book,row_number FROM context_observations')
        con.execute("CREATE VIEW table_numeric_queue AS SELECT s.*,r.status AS table_reason FROM grid_numeric_queue s JOIN table_review r USING(book,row_number) WHERE r.status NOT IN ('usable_table','usable_grid')")
    n=len(tables)+len(charts);report=dict(status='built_pending_validation',added=n,tables=len(tables),charts=len(charts),admitted=34416+n,numeric_pending=12797-n,previous_sha256=PREVIOUS_SHA,code_sha256=sha(Path(__file__)),profiles_sha256=sha(PROFILES),db_sha256=sha(db),rag_integrated=False)
    (output/'manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');return report


def check_new(con):
    tables,charts,decisions,pages,conflicts=collect(con)
    for table,expected in [('table_observations',tables),('context_observations',charts),('table_review',decisions),('table_pdf_evidence',pages),('table_conflicts',conflicts)]:
        actual=con.execute('SELECT * FROM '+table).fetchall();require(len(actual)==len(expected) and set(actual)==set(expected),'Table evidence mismatch: '+table)
    for view,sql in [('table_usable_keys','SELECT * FROM grid_usable_keys UNION ALL SELECT book,row_number FROM table_observations UNION ALL SELECT book,row_number FROM context_observations'),('table_numeric_queue',"SELECT s.*,r.status AS table_reason FROM grid_numeric_queue s JOIN table_review r USING(book,row_number) WHERE r.status NOT IN ('usable_table','usable_grid')")]:
        require(con.execute('SELECT * FROM '+view+' LIMIT 0').description==con.execute(sql+' LIMIT 0').description,'View columns changed')
        for a,b in [(view,'('+sql+')'),('('+sql+')',view)]:require(not con.execute(f'SELECT * FROM {a} EXCEPT ALL SELECT * FROM {b}').fetchall(),'View multiplicity changed')
    require(con.execute('SELECT count(*) FROM (SELECT book,row_number,count(*) n FROM table_usable_keys GROUP BY ALL HAVING n<>1)').fetchone()[0]==0,'Admission overlap')
    return tables,charts


def validate(db):
    previous=validate_previous(db)
    with duckdb.connect(str(db),read_only=True) as con:
        tables,charts=check_new(con);blocks=con.execute('SELECT count(*) FROM (SELECT book,block FROM table_observations UNION SELECT book,block FROM context_observations)').fetchone()[0]
        stats=con.execute('SELECT unit,role,count(*) FROM table_observations GROUP BY ALL ORDER BY count(*) DESC').fetchall();reasons=con.execute('SELECT status,count(*) FROM table_review GROUP BY ALL ORDER BY count(*) DESC').fetchall()
    n=len(tables)+len(charts);return dict(status='pass',added=n,tables=len(tables),charts=len(charts),admitted=34416+n,numeric_pending=12797-n,blocks=blocks,stats=stats,review=reasons,previous_validation=previous['status'],code_sha256=sha(Path(__file__)),profiles_sha256=sha(PROFILES),db_sha256=sha(Path(db)),rag_integrated=False)


def query_table(db,*,book,block,source_cell):
    require(book in ['2025','2026'] and all(isinstance(t,str) and t.strip() for t in [block,source_cell]),'Exact table identity required')
    with duckdb.connect(str(db),read_only=True) as con:
        c=con.execute('SELECT t.*,s.xlsx_path,s.xlsx_sha256,s.pdf_path,s.pdf_sha256 FROM table_observations t JOIN sources s USING(book) WHERE t.book=? AND block=? AND source_cell=?',[book,block,source_cell]);names=[n[0] for n in c.description];rows=c.fetchall()
    if len(rows)!=1:raise LookupError('No unique table cell; no fallback')
    return {n:str(v) if isinstance(v,Decimal) else v for n,v in zip(names,rows[0])}


def query_chart(db,*,book,block,series_id,point_index):
    require(book in ['2025','2026'] and all(isinstance(t,str) and t.strip() for t in [block,series_id]) and type(point_index) is int and point_index>=0,'Exact chart identity required')
    with duckdb.connect(str(db),read_only=True) as con:
        c=con.execute('SELECT t.*,s.xlsx_path,s.xlsx_sha256,s.pdf_path,s.pdf_sha256 FROM context_observations t JOIN sources s USING(book) WHERE t.book=? AND block=? AND series_id=? AND point_index=?',[book,block,series_id,point_index]);names=[n[0] for n in c.description];rows=c.fetchall()
    if len(rows)!=1:raise LookupError('No unique context point; no fallback')
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
