"""Whole literal vectors reconciled to source grids; names recovered only from explicit merges."""
from __future__ import annotations
import argparse
from decimal import Decimal
import json
from pathlib import Path
import re
import shutil
import xml.etree.ElementTree as ET
import duckdb
from openpyxl.utils.cell import range_boundaries,coordinate_to_tuple
from experiments.table_semantics import TableResolver,validate as validate_previous,label
from experiments.source_semantics import SourceResolver
from experiments.grid_semantics import FIELDS as OLD_FIELDS
from experiments.indexed_semantics import points
from experiments.resolved_semantics import canon
from experiments.scaled_semantics import DIVISORS,format_effect
from experiments.bulk_semantics import C,norm,ref_cells
from experiments.semantic_families import decimal,M
from experiments.excel_pilot import ROOT,sha
from experiments.universal_database import require

PREVIOUS=ROOT/'evaluation/table_semantics_2026-10-31/run_v3/tables.duckdb'
PREVIOUS_SHA='da01966b0f23fc01b640e725971ff0d5519cccb80a0fd13c57c793074a940178'
PROFILES=Path(__file__).with_name('reconciled_profiles.json')
PROFILES_SHA='efa9e70e1ca937748015dd20ee2386c71f737ff5a3c708921b9c742916a49245'
LITERALS={(p['book'],p['part'],p['series_id']):p for p in json.loads(PROFILES.read_text(encoding='utf-8'))}
MERGED={('2026','Gráfico 14B'),('2026','Anexo I. Gr 8A')}
FIELDS=OLD_FIELDS+'universal_series universal_value_sheet universal_value_cell reconciliation_evidence'.split()


def source_range(sheet,cells):
    require(cells,'Empty grid range')
    return f"'{sheet}'!{cells[0]}:{cells[-1]}"


class Reconciler(TableResolver):
    def __init__(self):
        require(sha(PROFILES)==PROFILES_SHA,'Reconciliation profiles changed')
        super().__init__();self.merge_evidence={};self.literal_evidence={};self.literal_failures={}

    def name_component(self,book,sheet,cell):
        source=self.sources[book];value=source.text(sheet,cell)
        if value is not None:return label(value),None
        require((book,sheet) in MERGED,'No reviewed merged-name profile')
        root=ET.fromstring(source.xml.parts[source.xml.sheets[sheet]]);row,col=coordinate_to_tuple(cell);matches=[]
        for node in root.findall('m:mergeCells/m:mergeCell',M):
            ref=node.get('ref');c1,r1,c2,r2=range_boundaries(ref)
            if r1<=row<=r2 and c1<=col<=c2:
                require(r1==r2==row,'Only horizontal header merges reviewed')
                top=ref.split(':')[0];matches.append((ref,top))
        require(len(matches)==1,'Name component lacks unique explicit merge')
        ref,top=matches[0];text=label(source.text(sheet,top));require(isinstance(text,str) and text.strip(),'Empty merged header')
        return text,dict(cell=cell,merge=ref,anchor=top,value=text)

    def literal_binding(self,row,p):
        key=row['book'],row['parte_grafico_imagen'],row['serie_id']
        if key in self.literal_failures:raise ValueError(self.literal_failures[key])
        source=self.sources[row['book']];sheet=row['bloque']
        require(source.text(sheet,p['title_cell'])==p['title'],'Literal source title changed')
        root=ET.fromstring(source.xml.parts[p['part']]);node=root.findall('.//c:ser',C)[int(p['series_id'].split('_s')[1])-1]
        val=node.find('c:val/c:numLit',C);require(val is not None,'Literal value origin changed')
        cat=node.find('c:cat/c:numLit',C)
        if cat is None:cat=node.find('c:cat/c:strLit',C)
        category_basis='literal_categories_verified_against_grid'
        grid_only=node.find('c:cat',C) is None and (row['book'],row['bloque'])==('2026','Gráfico 11A')
        if cat is None and not grid_only:
            ref=node.find('c:cat/c:numRef',C)
            if ref is None:ref=node.find('c:cat/c:strRef',C)
            require(ref is not None and ref.find('c:f',C) is not None,'Missing referenced category source')
            cs,cc=ref_cells(ref.find('c:f',C).text)
            require(cs==sheet and cc==p['category_cells'],'Category reference differs from reviewed grid')
            cat=ref.find('c:numCache',C)
            if cat is None:cat=ref.find('c:strCache',C)
            category_basis='source_cells_verified_literal_values'
        require(cat is not None or grid_only,'Literal category origin changed')
        vp=points(val);n=len(p['value_cells']);expected=set(range(n))
        if grid_only:
            cp={i:self.category(source,sheet,a) for i,a in enumerate(p['category_cells'])}
            require(all(isinstance(v,str) and v.strip() for v in cp.values()),'Grid category missing')
            category_basis='reviewed_grid_missing_chart_categories'
        else:cp=points(cat)
        require(set(vp)==set(cp)==expected,'Literal vector indices incomplete/duplicate')
        for parent in ([val] if grid_only else [val,cat]):
            count=parent.find('c:ptCount',C);require(count is not None and int(count.get('val'))==n,'Literal point count changed')
        require(len(p['category_cells'])==n,'Grid category count changed')
        literal=node.find('c:tx/c:v',C);name=literal.text if literal is not None else None
        require(name==p['literal_name'],'Literal series name changed')
        source_name=label(source.text(sheet,p['name_cell']));require(source_name==p['source_name'],'Source header changed')
        categories=[]
        for i,(value_cell,category_cell) in enumerate(zip(p['value_cells'],p['category_cells'])):
            require(decimal(vp[i])==source.number(sheet,value_cell),'Literal/source vector conflict')
            actual=self.category(source,sheet,category_cell);require(actual is not None and actual.strip(),'Source category absent')
            try:num=source.number(sheet,category_cell)
            except ValueError:num=None
            if num is not None:require(decimal(cp[i])==num,'Numeric literal category conflict')
            else:require(cp[i]==actual,'Text literal category conflict')
            categories.append(cp[i])
        evidence=dict(origin='whole_numLit_vector_reconciled_to_original_grid',part=p['part'],series_id=p['series_id'],points=n,name_cell=p['name_cell'],source_name=source_name,literal_name=name,value_cells=p['value_cells'],category_cells=p['category_cells'],outcome=p['outcome'])
        self.literal_evidence[key]=evidence
        if p['outcome']=='forecast_marker':
            require(name is None and source_name=='Previsión' and set(decimal(v) for v in vp.values())<={Decimal(0),Decimal(1)},'Forecast marker evidence changed')
            raise ValueError('forecast_marker_not_measure')
        if p['outcome']=='header_conflict':
            require(name is not None and canon(name)!=canon(source_name),'Header conflict no longer established')
            raise ValueError('literal_header_conflict')
        require(name and canon(name)==canon(source_name),'Literal/header name conflict')
        group=next(parent for parent in root.iter() if node in list(parent));ids={v.get('val') for v in group.findall('c:axId',C)}
        axes=[a for a in root.findall('.//c:valAx',C) if a.find('c:axId',C).get('val') in ids]
        require(len(axes)==1,'Literal value axis ambiguous');axis=axes[0];fmt=axis.find('c:numFmt',C)
        require(fmt is not None and axis.find('c:dispUnits',C) is None,'Literal axis format/display unsupported')
        divisor,multiplier=format_effect(fmt.get('formatCode'));require(divisor==1,'Literal display divisor unsupported')
        b=dict(label=name,name_ref=source_range(sheet,[p['name_cell']]),category_ref=source_range(sheet,p['category_cells']),value_ref=source_range(sheet,p['value_cells']),value_sheet=sheet,cells=p['value_cells'],categories=categories,valid_indices=expected,axis_id=axis.find('c:axId',C).get('val'),axis_format=fmt.get('formatCode'),display_unit='none',divisor=divisor,multiplier=multiplier,mode='whole_literal_vector_reconciled',category_basis=category_basis,name_basis='literal_name_verified_against_grid')
        self.bindings[key]=b;return b

    def binding(self,row):
        key=(row['book'],row['parte_grafico_imagen'],row['serie_id'])
        if key in self.bindings:return self.bindings[key]
        if key in LITERALS:return self.literal_binding(row,LITERALS[key])
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
                if label is None:
                    label,proof=self.name_component(row['book'],ns,cell)
                    if proof:self.merge_evidence.setdefault(key,[]).append(proof)
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
        key=row['book'],row['parte_grafico_imagen'],row['serie_id'];p=LITERALS.get(key)
        if p and p['outcome']=='admit' and row['titulo']==row['metrica_literal']==p['universal_title']:
            require(self.sources[row['book']].text(row['bloque'],p['title_cell'])==p['title'],'Literal title evidence changed')
            return [(p['title_cell'],p['title'])]
        if (row['book'],row['bloque'])==('2026','Anexo I. Gr 8A') and row['titulo']==row['metrica_literal']=='Proporción de mujeres que tienen hijos. Según nacionalidad':
            source=self.sources['2026'];sheet=row['bloque']
            expected=[('B3','ANEXO I. GRÁFICO 8A. PROPORCIÓN DE MUJERES QUE TIENEN HIJOS'),('C20','Española'),('E20','Extranjera')]
            if all(source.text(sheet,a)==t for a,t in expected):return expected
        return super().title_proof(row,texts)

    def unit_spec(self,row,texts,binding):
        p=LITERALS.get((row['book'],row['parte_grafico_imagen'],row['serie_id']))
        if p and p['outcome']=='admit':
            unit=p['unit'];require(unit and self.sources[row['book']].text(row['bloque'],p['title_cell'])==p['title'],'Literal unit profile missing')
            if row['bloque'] in ['Gráfico 9','Gráfico 10','Gráfico 11A','Gráfico 11B']:require(binding['multiplier']==100,'Expected percentage axis')
            else:require(binding['multiplier']==1,'Unexpected literal scaling')
            return 'unidad del perfil fuente revisado',unit,'reconciled_literal_profile',(p['title_cell'],p['title']+'; eje '+binding['axis_format'])
        return super().unit_spec(row,texts,binding)

    def review_new(self,row):
        key=row['book'],row['parte_grafico_imagen'],row['serie_id'];merged=(row['book'],row['bloque']) in MERGED
        if key not in LITERALS and not merged:return row['table_reason'],None
        try:b=self.binding(row)
        except (ValueError,ArithmeticError,KeyError,IndexError,AttributeError,StopIteration) as exc:
            reason=str(exc)
            if key in LITERALS:self.literal_failures[key]=reason
            return reason if reason in ['forecast_marker_not_measure','literal_header_conflict'] else 'reconciliation_evidence_review',None
        shadow=dict(row,serie_literal=b['label']);proof=self.literal_evidence.get(key,{})
        if key in LITERALS:
            point=re.fullmatch(re.escape(row['serie_id'])+r'_p([0-9]+)',row['id'] or '')
            if point is None:return 'point_identity_review',None
            index=int(point[1])
            if index>=len(b['cells']):return 'point_outside_reconciled_vector',None
            require(row['hoja_valor'] is None and row['celda_valor']==f'cache[{index}]','Literal staging provenance changed')
            shadow.update(hoja_valor=b['value_sheet'],celda_valor=b['cells'][index])
            if b['category_basis']=='reviewed_grid_missing_chart_categories' and row['periodo_categoria_literal'] is None:
                shadow['periodo_categoria_literal']=b['categories'][index]
        status,result=SourceResolver.review(self,shadow)
        if result is None:return status,None
        if merged:proof=dict(name_basis='explicit_original_merges',components=self.merge_evidence.get(key,[]))
        return 'usable_reconciled',result+(row['periodo_categoria_literal'],json.dumps(dict(origin=b['mode']),sort_keys=True),row['serie_literal'],row['hoja_valor'],row['celda_valor'],json.dumps(proof,ensure_ascii=False,sort_keys=True))


def collect(con):
    require(con.execute('SELECT count(*) FROM table_usable_keys').fetchone()[0]==36011,'Previous admissions changed')
    c=con.execute('SELECT * FROM table_numeric_queue ORDER BY book,row_number');names=[n[0] for n in c.description];data=c.fetchall();require(len(data)==11202,'Pending inventory changed')
    resolver=Reconciler();rows=[];decisions=[]
    try:
        for values in data:
            row=dict(zip(names,values));status,result=resolver.review_new(row);decisions.append((row['book'],row['row_number'],status))
            if result is not None:rows.append(result)
        reviews=[(b,part,sid,json.dumps(e,ensure_ascii=False,sort_keys=True)) for (b,part,sid),e in sorted(resolver.literal_evidence.items())]
    finally:resolver.close()
    return rows,decisions,reviews


def build(output):
    require(not output.exists(),'Output exists');require(sha(PREVIOUS)==PREVIOUS_SHA,'Previous database changed')
    output.mkdir(parents=True);db=output/'reconciled.duckdb';shutil.copyfile(PREVIOUS,db)
    with duckdb.connect(str(db)) as con:
        rows,decisions,reviews=collect(con)
        con.execute('CREATE TABLE reconciled_observations AS SELECT *,NULL::VARCHAR AS universal_series,NULL::VARCHAR AS universal_value_sheet,NULL::VARCHAR AS universal_value_cell,NULL::VARCHAR AS reconciliation_evidence FROM grid_observations WHERE false')
        if rows:con.executemany('INSERT INTO reconciled_observations VALUES ('+','.join('?' for _ in FIELDS)+')',rows)
        con.execute('CREATE TABLE reconciled_review(book VARCHAR,row_number INTEGER,status VARCHAR,PRIMARY KEY(book,row_number))');con.executemany('INSERT INTO reconciled_review VALUES (?,?,?)',decisions)
        con.execute('CREATE TABLE literal_vector_reviews(book VARCHAR,part VARCHAR,series_id VARCHAR,evidence VARCHAR,PRIMARY KEY(book,part,series_id))')
        if reviews:con.executemany('INSERT INTO literal_vector_reviews VALUES (?,?,?,?)',reviews)
        con.execute('CREATE VIEW reconciled_usable_keys AS SELECT * FROM table_usable_keys UNION ALL SELECT book,row_number FROM reconciled_observations')
        con.execute("CREATE VIEW reconciled_numeric_queue AS SELECT s.*,r.status AS reconciled_reason FROM table_numeric_queue s JOIN reconciled_review r USING(book,row_number) WHERE r.status<>'usable_reconciled'")
    report=dict(status='built_pending_validation',added=len(rows),admitted=36011+len(rows),numeric_pending=11202-len(rows),previous_sha256=PREVIOUS_SHA,code_sha256=sha(Path(__file__)),profiles_sha256=sha(PROFILES),db_sha256=sha(db),rag_integrated=False)
    (output/'manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');return report


def check_new(con):
    rows,decisions,reviews=collect(con)
    for table,expected in [('reconciled_observations',rows),('reconciled_review',decisions),('literal_vector_reviews',reviews)]:
        actual=con.execute('SELECT * FROM '+table).fetchall();require(len(actual)==len(expected) and set(actual)==set(expected),'Reconciled evidence mismatch: '+table)
    for view,sql in [('reconciled_usable_keys','SELECT * FROM table_usable_keys UNION ALL SELECT book,row_number FROM reconciled_observations'),('reconciled_numeric_queue',"SELECT s.*,r.status AS reconciled_reason FROM table_numeric_queue s JOIN reconciled_review r USING(book,row_number) WHERE r.status<>'usable_reconciled'")]:
        require(con.execute('SELECT * FROM '+view+' LIMIT 0').description==con.execute(sql+' LIMIT 0').description,'View columns changed')
        for a,b in [(view,'('+sql+')'),('('+sql+')',view)]:require(not con.execute(f'SELECT * FROM {a} EXCEPT ALL SELECT * FROM {b}').fetchall(),'View multiplicity changed')
    require(con.execute('SELECT count(*) FROM (SELECT book,row_number,count(*) n FROM reconciled_usable_keys GROUP BY ALL HAVING n<>1)').fetchone()[0]==0,'Admission overlap')
    return rows


def validate(db):
    previous=validate_previous(db)
    with duckdb.connect(str(db),read_only=True) as con:
        rows=check_new(con);blocks=con.execute('SELECT count(*) FROM (SELECT DISTINCT book,block FROM reconciled_observations)').fetchone()[0]
        stats=con.execute('SELECT unit,category_basis,name_basis,semantic_scope,count(*) FROM reconciled_observations GROUP BY ALL ORDER BY count(*) DESC').fetchall();reasons=con.execute('SELECT status,count(*) FROM reconciled_review GROUP BY ALL ORDER BY count(*) DESC').fetchall()
    return dict(status='pass',added=len(rows),admitted=36011+len(rows),numeric_pending=11202-len(rows),blocks=blocks,stats=stats,review=reasons,previous_validation=previous['status'],code_sha256=sha(Path(__file__)),profiles_sha256=sha(PROFILES),db_sha256=sha(Path(db)),rag_integrated=False)


def query(db,*,book,block,series_id,point_index):
    require(book in ['2025','2026'] and all(isinstance(t,str) and t.strip() for t in [block,series_id]) and type(point_index) is int and point_index>=0,'Exact reconciled identity required')
    with duckdb.connect(str(db),read_only=True) as con:
        c=con.execute('SELECT t.*,s.xlsx_path,s.xlsx_sha256,s.pdf_path,s.pdf_sha256 FROM reconciled_observations t JOIN sources s USING(book) WHERE t.book=? AND block=? AND series_id=? AND point_index=?',[book,block,series_id,point_index]);names=[n[0] for n in c.description];rows=c.fetchall()
    if len(rows)!=1:raise LookupError('No unique reconciled point; no fallback')
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
