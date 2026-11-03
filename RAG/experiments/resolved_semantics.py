"""Batch resolution of explicit source units, compound titles and verified partial axes."""
from __future__ import annotations
import argparse
from decimal import Decimal,localcontext
import json
from pathlib import Path
import re
import shutil
import xml.etree.ElementTree as ET

import duckdb
from experiments.scaled_semantics import ScaleReviewer,FIELDS as SCALE_FIELDS,DIVISORS,POPULATION_CONTEXTS,PERCENT_PROFILES,format_effect,validate as validate_previous
from experiments.bulk_semantics import UNITS,norm,ref_cells,C
from experiments.excel_pilot import ROOT,sha
from experiments.semantic_families import decimal
from experiments.universal_database import require

PREVIOUS=ROOT/'evaluation/scaled_semantics_2026-10-31/run_v3/scaled.duckdb'
PREVIOUS_SHA='c23264cfa4bcc74d16055a74472b182feffd89d19fe1ddc654fa2a77175d0d24'
FIELDS=SCALE_FIELDS+'point_index binding_mode unit_basis metric_evidence'.split()


def canon(text):return ''.join(c for c in norm(text) if c.isalnum())


class Resolver(ScaleReviewer):
    def __init__(self):
        super().__init__();self.extra_cache={};self.metric_cache={};self.unit_cache={}

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
        axes=[a for a in root.findall('.//c:valAx',C) if a.find('c:axId',C).get('val') in ids]
        require(len(axes)==1,'Ambiguous value axis')
        axis=axes[0];fmt=axis.find('c:numFmt',C);require(fmt is not None,'Missing axis format')
        axis_format=fmt.get('formatCode');fd,multiplier=format_effect(axis_format)
        display=axis.find('c:dispUnits',C);ad=Decimal(1);display_name='none'
        if display is not None:
            bi=display.find('c:builtInUnit',C);cu=display.find('c:custUnit',C)
            require((bi is None)!=(cu is None),'Ambiguous display unit')
            if bi is not None:
                display_name=bi.get('val','thousands');require(display_name in DIVISORS,'Unknown display unit');ad=DIVISORS[display_name]
            else:display_name='custom';ad=decimal(cu.get('val'));require(ad>0,'Invalid divisor')
        require(not(ad!=1 and (fd!=1 or multiplier!=1)),'Mixed axis scaling')
        name=node.find('c:tx/c:strRef/c:f',C);literal=node.find('c:tx/c:v',C)
        name_ref=name.text if name is not None else None
        if name_ref:
            ns,nc=ref_cells(name_ref);require(len(nc)==1,'Nonscalar label');label=source.text(ns,nc[0])
        else:label=literal.text if literal is not None else None
        require(isinstance(label,str) and label.strip(),'Unnamed series')
        cat=node.find('c:cat/c:numRef',C)
        if cat is None:cat=node.find('c:cat/c:strRef',C)
        val=node.find('c:val/c:numRef',C)
        require(cat is not None and val is not None,'No simple source axes')
        cr=cat.find('c:f',C);vr=val.find('c:f',C);require(cr is not None and vr is not None,'No source refs')
        cs,cc=ref_cells(cr.text);vs,vc=ref_cells(vr.text)
        require(len(vc)<=len(cc),'Values longer than categories')
        partial=len(vc)<len(cc)
        if partial:
            # Source refs alone do not prove which subset a partial drawing uses.
            # Require a contiguous cache prefix and exact original cell equality.
            cp={int(p.get('idx')):p.find('c:v',C).text for p in cat.findall('.//c:pt',C)}
            vp={int(p.get('idx')):p.find('c:v',C).text for p in val.findall('.//c:pt',C)}
            require(len(cp)==len(cat.findall('.//c:pt',C)) and len(vp)==len(val.findall('.//c:pt',C)),'Duplicate cache indices')
            require(set(vp)==set(range(len(vc))) and set(vp)<=set(cp),'Unproved partial prefix')
            for i,address in enumerate(vc):
                require(decimal(vp[i])==source.number(vs,address),'Partial cache/value mismatch')
                actual=self.category(source,cs,cc[i])
                if isinstance(source.text(cs,cc[i]),(int,float)) or str(source.text(cs,cc[i])).startswith('='):
                    require(decimal(cp[i])==source.number(cs,cc[i]),'Partial category cache mismatch')
                else:require(cp[i]==actual,'Partial text category mismatch')
        categories=[self.category(source,cs,address) for address in cc[:len(vc)]]
        require(None not in categories,'Missing category')
        result=dict(label=label,name_ref=name_ref,category_ref=cr.text,value_ref=vr.text,value_sheet=vs,
                    cells=vc,categories=categories,axis_id=axis.find('c:axId',C).get('val'),axis_format=axis_format,
                    display_unit=display_name,divisor=ad*fd,multiplier=multiplier,
                    mode='cache_verified_prefix' if partial else 'complete_indexed_axis')
        self.bindings[key]=result;return result

    @staticmethod
    def category(source,sheet,address):
        value=source.text(sheet,address)
        if isinstance(value,str) and value.startswith('='):
            number=source.number(sheet,address)
            return str(int(number)) if number is not None and number==int(number) else (str(number) if number is not None else None)
        return str(value) if value is not None else None

    def evidence(self,row):
        key=(row['book'],row['bloque'],row['parte_grafico_imagen'])
        if key not in self.extra_cache:
            source=self.sources[row['book']]
            texts=[(a,t) for a,t in source.sheet(row['bloque'])[1].items() if isinstance(t,str) and not t.startswith('=')]
            root=ET.fromstring(source.xml.parts[row['parte_grafico_imagen']])
            for index,title in enumerate(root.findall('.//c:title',C)):
                text=''.join(n.text or '' for n in title.iter() if n.tag.endswith('}t'))
                if text:texts.append((f'chart_title_{index}',text))
            self.extra_cache[key]=texts
        return self.extra_cache[key]

    def title_proof(self,row,texts):
        if row['metrica_literal']!=row['titulo'] or not canon(row['titulo']):return None
        whole=next(((a,t) for a,t in texts if canon(row['titulo']) in canon(t)),None)
        if whole:return [whole]
        # Compound headings must each have literal evidence; no fuzzy vocabulary matching.
        parts=[p.strip() for p in re.split(r'\.\s+',row['titulo']) if canon(p)]
        if len(parts)<2:return None
        proofs=[]
        for part in parts:
            found=next(((a,t) for a,t in texts if canon(part) in canon(t)),None)
            if found is None:return None
            proofs.append(found)
        return proofs

    def unit_spec(self,row,texts,binding):
        key=(row['book'],row['bloque'],row['parte_grafico_imagen'])
        if key not in self.unit_cache:
            # A different series' legend is not evidence of the current series' unit.
            root=ET.fromstring(self.sources[row['book']].xml.parts[row['parte_grafico_imagen']])
            excluded=set()
            for series in root.findall('.//c:ser',C):
                for ref in series.findall('c:tx/c:strRef/c:f',C)+series.findall('c:cat/c:numRef/c:f',C)+series.findall('c:cat/c:strRef/c:f',C):
                    try:
                        sheet,cells=ref_cells(ref.text)
                        if sheet==row['bloque']:excluded.update(cells)
                    except ValueError:pass
            self.unit_cache[key]=[(a,t) for a,t in texts if a not in excluded]
        texts=self.unit_cache[key]
        context=norm(row['contexto_unidad_titulo']);unit=UNITS.get(context);basis='universal_explicit_context'
        mixed=norm(row['titulo'])+' '+context
        if ('% pib' in mixed or '%pib' in mixed) and any(s in mixed for s in ['mm€','millones de euros','miles de euros']):return None
        # The subject of a percentage change is not the unit of that change.
        if 'tasa de variación' in norm(row['titulo']) and '%' in row['titulo']:
            proof=next(((a,t) for a,t in texts if 'tasa de variación' in norm(t) and '%' in t),None)
            if proof:return 'tasa de variación (%)','percent_change_as_labelled','explicit_relative_quantity',proof
            return None
        if context in POPULATION_CONTEXTS:unit='percent_population_as_labelled'
        elif context in ['var. anual %','var anual %']:unit='percent_annual_change_as_labelled'
        elif context in ['p. p.','p.p.','pp']:unit='percentage_points_as_labelled'
        elif context in ['%altas totales','%altas']:unit='percent_pension_entries_as_labelled'
        elif context=='% vab':unit='percent_gross_value_added_as_labelled'
        elif context=='%':
            profile=PERCENT_PROFILES.get((row['book'],row['bloque']))
            if profile and row['titulo']==profile[0]:unit=profile[1]
        choices=[]
        if unit:choices.append((context,unit,basis))
        # Source units absent in the universal are resolved from explicit source wording.
        for phrase,canonical in [('número de hijos por mujer','children_per_woman'),
          ('miles de personas','thousand_persons'),('miles de residentes','thousand_residents'),
          ('millones de personas','million_persons'),('puntos de pib','percentage_points_gdp'),
          ('puntos pib','percentage_points_gdp'),('%pib','percent_gdp'),('% pib','percent_gdp'),
          ('% de la población entre 15 y 74 años','percent_population_as_labelled'),
          ('% de la población activa','percent_population_as_labelled')]:
            choices.append((phrase,canonical,'literal_source_unit'))
        if any(t in norm(row['titulo']) for t in ['esperanza de vida','edad media','edad efectiva']):
            choices.append(('años','years','literal_source_age_unit'))
        if re.search(r'\btir\b',norm(row['titulo'])) and binding['multiplier']==100:
            choices.append(('tir','percent_internal_return_as_labelled','exact_tir_title_and_percent_axis'))
        for phrase,canonical,basis in choices:
            proof=next(((a,t) for a,t in texts if ''.join(phrase.split()) in ''.join(norm(t).split())),None)
            if proof:return phrase,canonical,basis,proof
        return None

    def review(self,row):
        try:binding=self.binding(row)
        except (ValueError,ArithmeticError,KeyError,IndexError,StopIteration,AttributeError):return 'binding_still_unresolved',None
        if row['serie_literal']!=binding['label']:return 'series_identity_review',None
        point=re.fullmatch(re.escape(row['serie_id'])+r'_p([0-9]+)',row['id'] or '')
        if point is None:return 'point_identity_review',None
        index=int(point[1])
        if index>=len(binding['cells']):return 'point_outside_source',None
        cell=binding['cells'][index]
        if row['periodo_categoria_literal']!=binding['categories'][index]:return 'category_identity_review',None
        if row['hoja_valor']!=binding['value_sheet'] or row['celda_valor']!=cell:return 'source_identity_review',None
        source=self.sources[row['book']]
        try:
            raw=decimal(row['decimal_XML_texto'])
            if raw!=source.number(binding['value_sheet'],cell):return 'source_value_review',None
        except (ValueError,ArithmeticError):return 'source_value_review',None
        if row['tipo']!='posicion_serie' or row['unidad']!='ver_contexto_literal':return 'record_type_review',None
        texts=self.evidence(row);metric=self.title_proof(row,texts)
        if metric is None:return 'metric_still_unresolved',None
        spec=self.unit_spec(row,texts,binding)
        if spec is None:return 'unit_still_unresolved',None
        scope,unit,basis,proof=spec
        if unit=='percent_interest_rate_as_published' and 'pib' in norm(row['serie_literal']):return 'series_unit_conflict',None
        if unit=='million_source_units' and not any(t in norm(row['titulo']) for t in ['pensiones','personas','población']):return 'unit_still_unresolved',None
        if binding['multiplier']!=1 and not unit.startswith('percent'):return 'axis_unit_conflict',None
        key=(row['book'],binding['value_sheet'])
        if key not in self.formats:
            self.formats[key]={c.coordinate:c.number_format for r in source.workbook[binding['value_sheet']].iter_rows() for c in r if c.value is not None}
        fmt=self.formats[key].get(cell,'General')
        try:cell_divisor,cell_multiplier=format_effect(fmt)
        except ValueError:return 'format_unresolved',None
        if cell_divisor!=1 or (cell_multiplier!=1 and binding['multiplier']==1):return 'format_conflict',None
        try:
            with localcontext() as ctx:
                ctx.prec=60;value=decimal(str(raw*binding['multiplier']/binding['divisor']))
        except (ValueError,ArithmeticError):return 'precision_review',None
        output=(row['book'],row['row_number'],row['id'],row['bloque'],row['metrica_literal'],row['serie_id'],
          row['serie_literal'],row['periodo_categoria_literal'],unit,value,binding['value_sheet'],cell,proof[0],proof[1],
          binding['value_ref'],binding['category_ref'],binding['name_ref'],row['parte_grafico_imagen'],'source_unit_and_indexed_axis_v1',
          raw,binding['axis_id'],binding['display_unit'],binding['axis_format'],fmt,binding['divisor'],binding['multiplier'],scope,
          index,binding['mode'],basis,json.dumps(metric,ensure_ascii=False))
        return 'usable_resolved',output


def collect(con):
    previous=set(con.execute('SELECT book,row_number FROM scaled_usable_keys').fetchall());require(len(previous)==13566,'Previous admissions changed')
    cursor=con.execute('SELECT * FROM scaled_numeric_queue ORDER BY book,row_number');names=[n[0] for n in cursor.description]
    rows=cursor.fetchall();require(len(rows)==33647,'Pending inventory changed')
    resolver=Resolver();records=[];decisions=[]
    try:
        for values in rows:
            row=dict(zip(names,values))
            if row['tipo']!='posicion_serie':status,result='table_or_manual_review',None
            else:status,result=resolver.review(row)
            decisions.append((row['book'],row['row_number'],status))
            if result is not None:records.append(result)
    finally:resolver.close()
    return records,decisions


def build(output):
    output=Path(output).resolve();output.relative_to(ROOT);require(not output.exists(),'Output exists')
    require(sha(PREVIOUS)==PREVIOUS_SHA,'Predecessor changed')
    output.mkdir(parents=True);db=output/'resolved.duckdb';shutil.copyfile(PREVIOUS,db)
    with duckdb.connect(str(db)) as con:
        rows,decisions=collect(con);con.execute('BEGIN')
        con.execute('CREATE TABLE resolved_observations AS SELECT *,NULL::INTEGER AS point_index,NULL::VARCHAR AS binding_mode,NULL::VARCHAR AS unit_basis,NULL::VARCHAR AS metric_evidence FROM scale_observations WHERE false')
        con.execute('ALTER TABLE resolved_observations ADD PRIMARY KEY(book,row_number)')
        if rows:con.executemany('INSERT INTO resolved_observations VALUES ('+','.join('?' for _ in FIELDS)+')',rows)
        con.execute('CREATE TABLE resolved_review(book VARCHAR,row_number INTEGER,status VARCHAR,PRIMARY KEY(book,row_number))')
        con.executemany('INSERT INTO resolved_review VALUES (?,?,?)',decisions)
        con.execute('CREATE VIEW resolved_usable_keys AS SELECT * FROM scaled_usable_keys UNION ALL SELECT book,row_number FROM resolved_observations')
        con.execute("CREATE VIEW resolved_numeric_queue AS SELECT s.*,r.status AS resolved_reason FROM scaled_numeric_queue s JOIN resolved_review r USING(book,row_number) WHERE r.status<>'usable_resolved'")
        con.execute('COMMIT')
    report=dict(status='built_pending_validation',added=len(rows),admitted=13566+len(rows),numeric_pending=33647-len(rows),
                previous_sha256=PREVIOUS_SHA,code_sha256=sha(Path(__file__)),db_sha256=sha(db),rag_integrated=False)
    (output/'manifest.json').write_text(json.dumps(report,indent=2),encoding='utf-8');return report


def check_new(con):
    expected,decisions=collect(con);actual=con.execute('SELECT * FROM resolved_observations').fetchall()
    require(len(actual)==len(expected) and set(actual)==set(expected),'Resolved content mismatch')
    reviews=con.execute('SELECT * FROM resolved_review').fetchall()
    require(len(reviews)==33647 and set(reviews)==set(decisions),'Resolved classification mismatch')
    for view,expression in [('resolved_usable_keys','SELECT * FROM scaled_usable_keys UNION ALL SELECT book,row_number FROM resolved_observations'),
      ('resolved_numeric_queue',"SELECT s.*,r.status AS resolved_reason FROM scaled_numeric_queue s JOIN resolved_review r USING(book,row_number) WHERE r.status<>'usable_resolved'")]:
        require([n[0] for n in con.execute(f'SELECT * FROM {view} LIMIT 0').description]==[n[0] for n in con.execute(expression+' LIMIT 0').description],'Resolved view schema mismatch')
        require(not con.execute(f'(SELECT * FROM {view} EXCEPT ALL ({expression})) UNION ALL (({expression}) EXCEPT ALL SELECT * FROM {view}) LIMIT 1').fetchone(),'Resolved view multiplicity/content mismatch')
    require(con.execute('SELECT count(*) FROM (SELECT book,row_number FROM resolved_usable_keys GROUP BY ALL HAVING count(*)>1)').fetchone()[0]==0,'Admission collision')
    return expected


def validate(db):
    prior=validate_previous(db)
    with duckdb.connect(str(db),read_only=True) as con:
        rows=check_new(con)
        counts=con.execute('SELECT unit,binding_mode,count(*) FROM resolved_observations GROUP BY ALL ORDER BY count(*) DESC').fetchall()
        statuses=con.execute('SELECT status,count(*) FROM resolved_review GROUP BY status ORDER BY count(*) DESC').fetchall()
        blocks=con.execute('SELECT count(*) FROM (SELECT DISTINCT book,block FROM resolved_observations)').fetchone()[0]
    return dict(status='pass',added=len(rows),admitted=13566+len(rows),numeric_pending=33647-len(rows),blocks=blocks,counts=counts,
                review=statuses,previous_validation=prior['status'],code_sha256=sha(Path(__file__)),db_sha256=sha(Path(db)),rag_integrated=False)


def query(db,*,book,block,series_id,point_index):
    require(book in ['2025','2026'] and all(isinstance(t,str) and t.strip() for t in [block,series_id]) and type(point_index) is int and point_index>=0,'Exact identity and integer point_index required')
    with duckdb.connect(str(db),read_only=True) as con:
        c=con.execute('''SELECT r.*,s.xlsx_path,s.xlsx_sha256,s.pdf_path,s.pdf_sha256 FROM resolved_observations r
          JOIN sources s ON s.book=r.book WHERE r.book=? AND block=? AND series_id=? AND point_index=?''',[book,block,series_id,point_index])
        names=[n[0] for n in c.description];rows=c.fetchall()
    if len(rows)!=1:raise LookupError('No unique resolved point; no fallback')
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
