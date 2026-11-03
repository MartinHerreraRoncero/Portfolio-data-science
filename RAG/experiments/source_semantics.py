"""Pinned PDF context, reviewed title variants and complete hierarchical names."""
from __future__ import annotations
import argparse
from decimal import Decimal
import json
from pathlib import Path
import re
import shutil
import xml.etree.ElementTree as ET
import duckdb
from pypdf import PdfReader
from experiments.indexed_semantics import IndexedResolver,FIELDS as OLD_FIELDS,points,validate as validate_previous
from experiments.resolved_semantics import canon
from experiments.scaled_semantics import DIVISORS,format_effect
from experiments.bulk_semantics import C,norm,ref_cells
from experiments.semantic_families import decimal
from experiments.excel_pilot import ROOT,BOOKS,PDF_HASHES,sha
from experiments.universal_database import require

PREVIOUS=ROOT/'evaluation/indexed_semantics_2026-10-31/run_v4/indexed.duckdb'
PREVIOUS_SHA='940753d873554a81bf6485047b8d7c9fa8779cd5dd5a2a921ac8e30ec1fcf208'
PROFILES=Path(__file__).with_name('source_title_profiles.json')
PROFILES_SHA='99a696d913a8f4d923705aab6782e7704ca98add0d41436fe8269b7dfc874c77'
FIELDS=OLD_FIELDS+['supplemental_evidence']
TITLE_PROFILES={(p['book'],p['block']):p for p in json.loads(PROFILES.read_text(encoding='utf-8'))}


class SourceResolver(IndexedResolver):
    def __init__(self):
        require(sha(PROFILES)==PROFILES_SHA,'Title profiles changed')
        super().__init__();self.pdf_readers={};self.pdf_pages={};self.supplements={}

    def pdf(self,book,page,fragments):
        key=(book,page)
        if key not in self.pdf_pages:
            path=(ROOT/BOOKS[book][0]).parent/BOOKS[book][2]
            if book not in self.pdf_readers:
                require(sha(path)==PDF_HASHES[book],'PDF source changed')
                self.pdf_readers[book]=PdfReader(path)
            self.pdf_pages[key]=' '.join(self.pdf_readers[book].pages[page-1].extract_text().split())
        text=self.pdf_pages[key]
        require(all(canon(f) in canon(text) for f in fragments),'PDF evidence missing')
        return dict(book=book,page=page,pdf_sha256=PDF_HASHES[book],fragments=fragments)

    def close(self):
        super().close()
        for reader in self.pdf_readers.values():reader.close()

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
            ns,nc=ref_cells(name_ref);require(0<len(nc)<=4,'Name range outside reviewed hierarchy bound')
            parts=[]
            for cell in nc:
                label=source.text(ns,cell)
                require(label is not None and str(label).strip(),'Incomplete hierarchical name')
                if isinstance(label,str) and label.startswith('='):label=source.number(ns,cell)
                if isinstance(label,(int,float,Decimal)) and not isinstance(label,bool):
                    require(Decimal(str(label)).is_finite(),'Nonfinite series name')
                    label=str(int(label)) if label==int(label) else str(label)
                require(isinstance(label,str) and label.strip(),'Invalid name component');parts.append(label)
            label=' | '.join(parts)
            name_basis='complete_source_header_range' if len(nc)>1 else 'source_scalar'
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
        key=(row['book'],row['bloque'])
        if row['titulo']!=row['metrica_literal']:return None
        profile=TITLE_PROFILES.get(key)
        if profile and row['titulo']==profile['universal_title']:
            evidence=profile['source_headers']
            if all(any(a==address and canon(t)==canon(expected) for a,t in texts) for address,expected in evidence):
                self.title_modes[key]='explicit_reviewed_source_variant'
                return evidence
        if key[0]=='2026' and key[1] in ['Anexo I. Gr 10A','Anexo I. Gr 10B']:
            suffix='Hombres' if key[1].endswith('A') else 'Mujeres'
            expected=f'Esperanza de vida al nacer. Según sexo en España y el entorno inmediato (años). {suffix}'
            if row['titulo']!=expected:return None
            headers=[('B2','ANEXO I. GRÁFICO 10. EVOLUCIÓN DE LA ESPERANZA DE VIDA AL NACER. SEGÚN SEXO EN ENTORNO INTERNACIONAL'),('B3',f'ANEXO I. GRÁFICO 10{key[1][-1]}. {suffix.upper()}')]
            if not all(any(a==address and canon(t)==canon(expected) for a,t in texts) for address,expected in headers):return None
            proof=self.pdf('2026',88,['ANEXO I. GRÁFICO 10. ESPERANZA DE VIDA AL NACER. SEGÚN SEXO EN ESPAÑA y EL ENTORNO INMEDIATO. (AÑOS)',f'ANEXO I. GRÁFICO 10.{key[1][-1]}. {suffix.upper()}'])
            self.supplements[key]=proof;self.title_modes[key]='source_headers_and_pdf_caption'
            return headers+[(f'pdf:2026:88',json.dumps(proof,ensure_ascii=False))]
        return None

    def unit_spec(self,row,texts,binding):
        key=(row['book'],row['bloque']);unit=None;proof=None
        if key==('2025','Gráfico 68') and row['titulo']=='DIFERENCIAL CON ALEMANIA (PRIMA DE RIESGO)' and binding['label'] in ['España','Francia','Italia']:
            require(binding['divisor']==1 and binding['multiplier']==1,'Risk spread scaling conflict')
            proof=self.pdf('2025',130,['GRÁFICO 68. DIFERENCIAL CON ALEMANIA (PRIMA DE RIESGO)','La prima de riesgo española','70 puntos básicos'])
            unit='basis_points_risk_spread_as_published'
        elif key==('2025','Gráfico 66') and row['titulo']=='TIPOS ESPERADOS PARA DICIEMBRE DE 2025 Y 2026' and binding['label'] in ['Tipo FED dic25','Tipo BCE dic25','Tipo FED dic26','Tipo BCE dic26']:
            require(binding['divisor']==1 and binding['multiplier']==1,'Policy rate scaling conflict')
            proof=self.pdf('2025',129,['GRÁFICO 66. TIPOS ESPERADOS PARA DICIEMBRE DE 2025 Y 2026','El BCE ha recortado la tasa de depósito','4,00% hasta el 2,50%'])
            unit='percent_policy_rate_as_published'
        elif key==('2025','Gráfico 65') and row['titulo']=='INFLACIÓN ARMONIZADA Y TIPO DE FACILIDAD DE DEPÓSITO' and binding['label'] in ['Inlfación Eurozona','Subyacente','Facilidad de depósito del BCE']:
            require(binding['divisor']==1 and binding['multiplier']==1,'Inflation/rate scaling conflict')
            proof=self.pdf('2025',129,['GRÁFICO 65. INFLACIÓN ARMONIZADA Y TIPO DE FACILIDAD DE DEPÓSITO','variación anual','4,00% hasta el 2,50%'])
            unit='percent_policy_rate_as_published' if binding['label']=='Facilidad de depósito del BCE' else 'percent_annual_inflation_as_published'
        elif key[0]=='2026' and key[1] in ['Anexo I. Gr 10A','Anexo I. Gr 10B']:
            # title_proof already requires both source headers and this exact PDF caption.
            if self.title_modes.get(key)=='source_headers_and_pdf_caption':
                proof=self.pdf('2026',88,['ESPERANZA DE VIDA AL NACER','(AÑOS)']);unit='years'
        if unit:
            self.supplements[key]=proof
            return 'unidad acreditada por contexto PDF del gráfico',unit,'reviewed_pdf_chart_context',(f"pdf:{proof['book']}:{proof['page']}",json.dumps(proof,ensure_ascii=False))
        if key==('2025','Gráfico 55') and row['titulo']=='RENTABILIDAD DE LA DEUDA PÚBLICA ESPAÑOLA (1, 10 Y 30 AÑOS)' and binding['multiplier']==100:
            found=next(((a,t) for a,t in texts if canon(row['titulo']) in canon(t)),None)
            if found:return 'rentabilidad + eje porcentual','percent_interest_rate_as_published','return_title_and_percent_axis',found
        for phrase,unit in [('ptos pib','percentage_points_gdp'),('% de la población 15-74','percent_population_as_labelled'),('horas medias por ocupado','hours_per_employed_as_labelled'),('coeficientes estimados modelo logit','logit_coefficient_as_labelled')]:
            found=next(((a,t) for a,t in texts if phrase in norm(t)),None)
            if found and phrase in norm(row['titulo']):return phrase,unit,'explicit_source_quantity',found
        return super().unit_spec(row,texts,binding)

    def review(self,row):
        if norm(row['serie_literal']) in ['auxiliar','start & end']:
            return 'drawing_helper_not_economic_series',None
        if row['book']=='2025' and row['bloque'] in ['Gráfico 12A','Gráfico 12B','Gráfico 12C']:
            # Signed pyramid coordinates need a separate display/count contract.
            return 'signed_pyramid_scope_review',None
        try:status,result=super().review(row)
        except (ValueError,ArithmeticError):return 'supplemental_evidence_conflict',None
        if result is None:return status,None
        extra=self.supplements.get((row['book'],row['bloque']),{})
        return 'usable_source',result+(json.dumps(extra,ensure_ascii=False,sort_keys=True),)


def collect(con):
    require(con.execute('SELECT count(*) FROM indexed_usable_keys').fetchone()[0]==26273,'Previous admissions changed')
    c=con.execute('SELECT * FROM indexed_numeric_queue ORDER BY book,row_number');names=[n[0] for n in c.description];data=c.fetchall()
    require(len(data)==20940,'Pending inventory changed');resolver=SourceResolver();records=[];decisions=[]
    try:
        for values in data:
            row=dict(zip(names,values))
            status,result=resolver.review(row) if row['tipo']=='posicion_serie' else ('table_or_manual_review',None)
            decisions.append((row['book'],row['row_number'],status))
            if result is not None:records.append(result)
        pages=[(b,p,PDF_HASHES[b],t) for (b,p),t in sorted(resolver.pdf_pages.items())]
    finally:resolver.close()
    return records,decisions,pages


def build(output):
    require(not output.exists(),'Output exists');require(sha(PREVIOUS)==PREVIOUS_SHA,'Previous database changed')
    output.mkdir(parents=True);db=output/'source.duckdb';shutil.copyfile(PREVIOUS,db)
    with duckdb.connect(str(db)) as con:
        rows,decisions,pages=collect(con)
        con.execute('CREATE TABLE source_observations AS SELECT *,NULL::VARCHAR AS supplemental_evidence FROM indexed_observations WHERE false')
        if rows:con.executemany('INSERT INTO source_observations VALUES ('+','.join('?' for _ in FIELDS)+')',rows)
        con.execute('CREATE TABLE source_review(book VARCHAR,row_number INTEGER,status VARCHAR,PRIMARY KEY(book,row_number))')
        con.executemany('INSERT INTO source_review VALUES (?,?,?)',decisions)
        con.execute('CREATE TABLE source_pdf_evidence(book VARCHAR,page INTEGER,pdf_sha256 VARCHAR,text VARCHAR,PRIMARY KEY(book,page))')
        if pages:con.executemany('INSERT INTO source_pdf_evidence VALUES (?,?,?,?)',pages)
        con.execute('CREATE VIEW source_usable_keys AS SELECT * FROM indexed_usable_keys UNION ALL SELECT book,row_number FROM source_observations')
        con.execute("CREATE VIEW source_numeric_queue AS SELECT s.*,r.status AS source_reason FROM indexed_numeric_queue s JOIN source_review r USING(book,row_number) WHERE r.status<>'usable_source'")
    report=dict(status='built_pending_validation',added=len(rows),admitted=26273+len(rows),numeric_pending=20940-len(rows),previous_sha256=PREVIOUS_SHA,code_sha256=sha(Path(__file__)),profiles_sha256=sha(PROFILES),db_sha256=sha(db),rag_integrated=False)
    (output/'manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');return report


def check_new(con):
    rows,decisions,pages=collect(con)
    for table,expected in [('source_observations',rows),('source_review',decisions),('source_pdf_evidence',pages)]:
        actual=con.execute('SELECT * FROM '+table).fetchall()
        require(len(actual)==len(expected) and set(actual)==set(expected),'Source evidence/table mismatch: '+table)
    for view,sql in [('source_usable_keys','SELECT * FROM indexed_usable_keys UNION ALL SELECT book,row_number FROM source_observations'),('source_numeric_queue',"SELECT s.*,r.status AS source_reason FROM indexed_numeric_queue s JOIN source_review r USING(book,row_number) WHERE r.status<>'usable_source'")]:
        require(con.execute('SELECT * FROM '+view+' LIMIT 0').description==con.execute(sql+' LIMIT 0').description,'View columns changed')
        for a,b in [(view,'('+sql+')'),('('+sql+')',view)]:require(not con.execute(f'SELECT * FROM {a} EXCEPT ALL SELECT * FROM {b}').fetchall(),'View multiplicity changed')
    require(con.execute('SELECT count(*) FROM (SELECT book,row_number,count(*) n FROM source_usable_keys GROUP BY ALL HAVING n<>1)').fetchone()[0]==0,'Admission overlap')
    return rows


def validate(db):
    previous=validate_previous(db)
    with duckdb.connect(str(db),read_only=True) as con:
        rows=check_new(con);blocks=con.execute('SELECT count(*) FROM (SELECT DISTINCT book,block FROM source_observations)').fetchone()[0]
        stats=con.execute('SELECT unit,name_basis,semantic_scope,count(*) FROM source_observations GROUP BY ALL ORDER BY count(*) DESC').fetchall()
        reasons=con.execute('SELECT status,count(*) FROM source_review GROUP BY ALL ORDER BY count(*) DESC').fetchall()
    return dict(status='pass',added=len(rows),admitted=26273+len(rows),numeric_pending=20940-len(rows),blocks=blocks,stats=stats,review=reasons,previous_validation=previous['status'],code_sha256=sha(Path(__file__)),profiles_sha256=sha(PROFILES),db_sha256=sha(Path(db)),rag_integrated=False)


def query(db,*,book,block,series_id,point_index):
    require(book in ['2025','2026'] and all(isinstance(t,str) and t.strip() for t in [block,series_id]) and type(point_index) is int and point_index>=0,'Exact identity and integer point_index required')
    with duckdb.connect(str(db),read_only=True) as con:
        c=con.execute('SELECT t.*,s.xlsx_path,s.xlsx_sha256,s.pdf_path,s.pdf_sha256 FROM source_observations t JOIN sources s ON s.book=t.book WHERE t.book=? AND block=? AND series_id=? AND point_index=?',[book,block,series_id,point_index]);names=[n[0] for n in c.description];rows=c.fetchall()
    if len(rows)!=1:raise LookupError('No unique source point; no fallback')
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
