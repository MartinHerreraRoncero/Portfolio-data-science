"""Reviewed worksheet grids for omitted chart categories; literal-cache conflicts stay excluded."""
from __future__ import annotations
import argparse
from decimal import Decimal
import json
from pathlib import Path
import re
import shutil
import xml.etree.ElementTree as ET
import duckdb
from experiments.source_semantics import SourceResolver,FIELDS as OLD_FIELDS,validate as validate_previous
from experiments.resolved_semantics import canon
from experiments.indexed_semantics import points
from experiments.scaled_semantics import DIVISORS,format_effect
from experiments.bulk_semantics import C,norm,ref_cells
from experiments.semantic_families import decimal
from experiments.excel_pilot import ROOT,sha,PDF_HASHES
from experiments.universal_database import require

PREVIOUS=ROOT/'evaluation/source_semantics_2026-10-31/run_v2/source.duckdb'
PREVIOUS_SHA='e6ad72a79cc2427b01a97b6bdad382ef2652894a5f3a0b037168ede21c9edf34'
PROFILES=Path(__file__).with_name('grid_profiles.json')
PROFILES_SHA='f2b972a2e4ad6fb3ed2a6d3337799d0b98ca8cb679d61a20a2c9c503e93aeec2'
GRIDS={(p['book'],p['block']):p for p in json.loads(PROFILES.read_text(encoding='utf-8'))}
FIELDS=OLD_FIELDS+['universal_category','grid_evidence']


class GridResolver(SourceResolver):
    def __init__(self):
        require(sha(PROFILES)==PROFILES_SHA,'Grid profiles changed')
        super().__init__();self.literal_reviews={};self.conflicts=[]

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
        elif node.find('c:cat',C) is None and (row['book'],row['bloque']) in GRIDS:
            grid=GRIDS[row['book'],row['bloque']]
            require(vs==row['bloque'] and source.text(vs,grid['title_address'])==grid['title'],'Grid title/source changed')
            require(name_ref is not None,'Grid requires explicit header name reference')
            ns,nc=ref_cells(name_ref);value_cols={re.match('[A-Z]+',a)[0] for a in vc}
            require(len(value_cols)==1 and ns==vs and len(nc)==1,'Grid requires vertical values and scalar header')
            column=next(iter(value_cols));require(nc[0]==f"{column}{grid['header_row']}",'Grid name header is not aligned')
            rr=[int(re.search(r'[0-9]+',a)[0]) for a in vc]
            require(rr==list(range(rr[0],rr[0]+len(rr))) and rr[0]>grid['header_row'],'Grid values not contiguous below header')
            cc=[grid['category_column']+str(i) for i in rr];cs=vs
            categories=[self.category(source,cs,a) for a in cc]
            cr_text=f"'{cs}'!${grid['category_column']}${rr[0]}:${grid['category_column']}${rr[-1]}"
            category_basis='reviewed_source_grid'
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

    def unit_spec(self,row,texts,binding):
        key=row['book'],row['bloque']
        if key[0]=='2025' and key[1] in ['Gráfico 105','Gráfico 106']:
            title='TIPOS DE INTERÉS DE INTERVENCIÓN DEL BCE' if key[1].endswith('105') else 'TIPOS DE INTERÉS, BONO A 10 AÑOS'
            if row['titulo']==title:
                require(binding['multiplier']==1 and binding['divisor']==1,'Climate interest scale conflict')
                proof=self.pdf('2025',154,['GRÁFICO 105. TIPOS DE INTERÉS DE INTERVENCIÓN DEL BCE','GRÁFICO 106. TIPOS DE INTERÉS, BONO A 10 AÑOS','TIPOS DE INTERÉS DE LAS NUEVAS EMISIONES (%)'])
                self.supplements[key]=proof
                return 'tipos porcentuales publicados','percent_interest_rate_as_published','reviewed_pdf_interest_context',('pdf:2025:154',json.dumps(proof,ensure_ascii=False))
        if key[0]=='2025' and key[1] in ['Gráfico 119','Gráfico 120'] and '%' in row['titulo']:
            title='TIPO INTERÉS NUEVAS EMISIONES (%)' if key[1].endswith('119') else 'TIPO DE INTERÉS IMPLÍCITO (%)'
            found=next(((a,t) for a,t in texts if canon(title) in canon(t)),None)
            if row['titulo']==title and found:return 'tipo de interés (%)','percent_interest_rate_as_published','literal_source_rate_unit',found
        return super().unit_spec(row,texts,binding)

    def literal_review(self,row):
        key=row['book'],row['parte_grafico_imagen'],row['serie_id']
        if key in self.literal_reviews:return self.literal_reviews[key]
        root=ET.fromstring(self.sources[row['book']].xml.parts[row['parte_grafico_imagen']])
        node=root.findall('.//c:ser',C)[int(row['serie_id'].split('_s')[1])-1]
        literal=node.find('c:val/c:numLit',C)
        if literal is None:return None
        result=('literal_values_not_reconciled',{})
        # These five objects contain old literal vectors alongside current worksheet tables.
        if row['book']=='2025' and row['bloque'] in ['Gráfico 111','Gráfico 112','Gráfico 113','Gráfico 121','Gráfico 122']:
            label=node.find('c:tx/c:v',C);cat=node.find('c:cat/c:numLit',C)
            if label is not None and cat is not None:
                source=self.sources[row['book']];values=source.sheet(row['bloque'])[1]
                headers=[a for a,t in values.items() if t==label.text and int(re.search(r'\d+',a)[0])<=6]
                if len(headers)==1:
                    col=re.match('[A-Z]+',headers[0])[0];vp=points(literal);cp=points(cat)
                    categories={}
                    for address in values:
                        if re.fullmatch(r'B[0-9]+',address):
                            text=self.category(source,row['bloque'],address)
                            if text:categories.setdefault(text,[]).append(address)
                    for index in sorted(set(cp)&set(vp)):
                        addresses=categories.get(cp[index],[])
                        if len(addresses)!=1:continue
                        cell=col+re.search(r'\d+',addresses[0])[0]
                        try:actual=source.number(row['bloque'],cell);cached=decimal(vp[index])
                        except (ValueError,ArithmeticError):continue
                        if actual is not None and actual!=cached:
                            result=('literal_series_source_conflict',dict(series_name=label.text,name_cell=headers[0],chart_point=index,category=cp[index],category_cell=addresses[0],source_cell=cell,source_value=str(actual),literal_value=vp[index]))
                            break
        self.literal_reviews[key]=result;return result

    def review(self,row):
        if row['tipo']!='posicion_serie':return 'table_or_manual_review',None
        if not re.fullmatch(r'chart[0-9]+_s[1-9][0-9]*',row['serie_id'] or ''):return 'binding_unresolved',None
        try:literal=self.literal_review(row)
        except (ValueError,ArithmeticError,KeyError,IndexError,AttributeError):literal=('literal_evidence_review',{})
        if literal is not None:
            status,evidence=literal
            if evidence:self.conflicts.append((row['book'],row['row_number'],status,json.dumps(evidence,ensure_ascii=False,sort_keys=True)))
            return status,None
        original=row['periodo_categoria_literal'];shadow=row;proof={}
        try:b=self.binding(row)
        except (ValueError,ArithmeticError,KeyError,IndexError,AttributeError,StopIteration):return 'binding_unresolved',None
        if b['category_basis']=='reviewed_source_grid':
            match=re.fullmatch(re.escape(row['serie_id'])+r'_p([0-9]+)',row['id'] or '')
            if match is None:return 'point_identity_review',None
            index=int(match[1])
            if index>=len(b['categories']) or not b['categories'][index]:return 'unproved_grid_category',None
            category=b['categories'][index]
            if original is not None and original!=category:return 'grid_category_conflict',None
            shadow=dict(row,periodo_categoria_literal=category)
            proof=dict(basis='reviewed_source_grid',category_ref=b['category_ref'],name_ref=b['name_ref'],title=GRIDS[row['book'],row['bloque']])
        status,result=super().review(shadow)
        if result is None:return status,None
        return 'usable_grid',result+(original,json.dumps(proof,ensure_ascii=False,sort_keys=True))


def collect(con):
    require(con.execute('SELECT count(*) FROM source_usable_keys').fetchone()[0]==31705,'Previous admissions changed')
    c=con.execute('SELECT * FROM source_numeric_queue ORDER BY book,row_number');names=[n[0] for n in c.description];data=c.fetchall()
    require(len(data)==15508,'Pending inventory changed');resolver=GridResolver();records=[];decisions=[]
    try:
        for values in data:
            row=dict(zip(names,values));status,result=resolver.review(row)
            decisions.append((row['book'],row['row_number'],status))
            if result is not None:records.append(result)
        pages=[(b,p,PDF_HASHES[b],t) for (b,p),t in sorted(resolver.pdf_pages.items())];conflicts=resolver.conflicts
    finally:resolver.close()
    return records,decisions,pages,conflicts


def build(output):
    require(not output.exists(),'Output exists');require(sha(PREVIOUS)==PREVIOUS_SHA,'Previous database changed')
    output.mkdir(parents=True);db=output/'grid.duckdb';shutil.copyfile(PREVIOUS,db)
    with duckdb.connect(str(db)) as con:
        rows,decisions,pages,conflicts=collect(con)
        con.execute('CREATE TABLE grid_observations AS SELECT *,NULL::VARCHAR AS universal_category,NULL::VARCHAR AS grid_evidence FROM source_observations WHERE false')
        if rows:con.executemany('INSERT INTO grid_observations VALUES ('+','.join('?' for _ in FIELDS)+')',rows)
        con.execute('CREATE TABLE grid_review(book VARCHAR,row_number INTEGER,status VARCHAR,PRIMARY KEY(book,row_number))')
        con.executemany('INSERT INTO grid_review VALUES (?,?,?)',decisions)
        con.execute('CREATE TABLE grid_pdf_evidence(book VARCHAR,page INTEGER,pdf_sha256 VARCHAR,text VARCHAR,PRIMARY KEY(book,page))')
        if pages:con.executemany('INSERT INTO grid_pdf_evidence VALUES (?,?,?,?)',pages)
        con.execute('CREATE TABLE grid_conflicts(book VARCHAR,row_number INTEGER,status VARCHAR,evidence VARCHAR,PRIMARY KEY(book,row_number))')
        if conflicts:con.executemany('INSERT INTO grid_conflicts VALUES (?,?,?,?)',conflicts)
        con.execute('CREATE VIEW grid_usable_keys AS SELECT * FROM source_usable_keys UNION ALL SELECT book,row_number FROM grid_observations')
        con.execute("CREATE VIEW grid_numeric_queue AS SELECT s.*,r.status AS grid_reason FROM source_numeric_queue s JOIN grid_review r USING(book,row_number) WHERE r.status<>'usable_grid'")
    report=dict(status='built_pending_validation',added=len(rows),admitted=31705+len(rows),numeric_pending=15508-len(rows),conflicting_rows=len(conflicts),previous_sha256=PREVIOUS_SHA,code_sha256=sha(Path(__file__)),profiles_sha256=sha(PROFILES),db_sha256=sha(db),rag_integrated=False)
    (output/'manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');return report


def check_new(con):
    rows,decisions,pages,conflicts=collect(con)
    for table,expected in [('grid_observations',rows),('grid_review',decisions),('grid_pdf_evidence',pages),('grid_conflicts',conflicts)]:
        actual=con.execute('SELECT * FROM '+table).fetchall()
        require(len(actual)==len(expected) and set(actual)==set(expected),'Grid evidence/table mismatch: '+table)
    for view,sql in [('grid_usable_keys','SELECT * FROM source_usable_keys UNION ALL SELECT book,row_number FROM grid_observations'),('grid_numeric_queue',"SELECT s.*,r.status AS grid_reason FROM source_numeric_queue s JOIN grid_review r USING(book,row_number) WHERE r.status<>'usable_grid'")]:
        require(con.execute('SELECT * FROM '+view+' LIMIT 0').description==con.execute(sql+' LIMIT 0').description,'View columns changed')
        for a,b in [(view,'('+sql+')'),('('+sql+')',view)]:require(not con.execute(f'SELECT * FROM {a} EXCEPT ALL SELECT * FROM {b}').fetchall(),'View multiplicity changed')
    require(con.execute('SELECT count(*) FROM (SELECT book,row_number,count(*) n FROM grid_usable_keys GROUP BY ALL HAVING n<>1)').fetchone()[0]==0,'Admission overlap')
    return rows


def validate(db):
    previous=validate_previous(db)
    with duckdb.connect(str(db),read_only=True) as con:
        rows=check_new(con);blocks=con.execute('SELECT count(*) FROM (SELECT DISTINCT book,block FROM grid_observations)').fetchone()[0]
        stats=con.execute('SELECT unit,category_basis,semantic_scope,count(*) FROM grid_observations GROUP BY ALL ORDER BY count(*) DESC').fetchall()
        reasons=con.execute('SELECT status,count(*) FROM grid_review GROUP BY ALL ORDER BY count(*) DESC').fetchall()
    return dict(status='pass',added=len(rows),admitted=31705+len(rows),numeric_pending=15508-len(rows),blocks=blocks,stats=stats,review=reasons,previous_validation=previous['status'],code_sha256=sha(Path(__file__)),profiles_sha256=sha(PROFILES),db_sha256=sha(Path(db)),rag_integrated=False)


def query(db,*,book,block,series_id,point_index):
    require(book in ['2025','2026'] and all(isinstance(t,str) and t.strip() for t in [block,series_id]) and type(point_index) is int and point_index>=0,'Exact identity and integer point_index required')
    with duckdb.connect(str(db),read_only=True) as con:
        c=con.execute('SELECT t.*,s.xlsx_path,s.xlsx_sha256,s.pdf_path,s.pdf_sha256 FROM grid_observations t JOIN sources s ON s.book=t.book WHERE t.book=? AND block=? AND series_id=? AND point_index=?',[book,block,series_id,point_index]);names=[n[0] for n in c.description];rows=c.fetchall()
    if len(rows)!=1:raise LookupError('No unique grid point; no fallback')
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
