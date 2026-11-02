"""Recover omitted chart names from bounded physical measure headers; no RAG integration."""
import argparse,json,re,shutil,xml.etree.ElementTree as ET
from pathlib import Path
from decimal import Decimal
import duckdb
from experiments.formula_name_semantics import FormulaResolver,FIELDS,validate as validate_previous
from experiments.source_semantics import SourceResolver
from experiments.indexed_semantics import points
from experiments.scaled_semantics import DIVISORS,format_effect
from experiments.bulk_semantics import C,ref_cells,norm
from experiments.semantic_families import decimal,M
from experiments.excel_pilot import ROOT,sha
from experiments.universal_database import require
PREVIOUS=ROOT/'evaluation/formula_name_semantics_2026-10-31/run_v2/formula_names.duckdb'
PREVIOUS_SHA='5026cb5216ca1ad81e60c8e62890597405df460c751447ad9558e96f7ec0f47d'
SPECS={
 ('2025','Gráfico 76'):dict(sheet='Gráfico 76',header='C4',name='Diferencial interés-crecimiento',title_cell='B2',title='Gráfico 76. DIFERENCIAL INTERÉS-CRECIMIENTO, (PUNTOS PIB)',unit='percentage_points_gdp',parts={'chart129_s1':("'Gráfico 76'!$B$5:$B$80","'Gráfico 76'!$C$5:$C$80"),'chart129_s2':("'Gráfico 76'!$B$5:$B$80","'Gráfico 76'!$C$5:$C$34")}),
 ('2026','Gráfico 33'):dict(sheet='Gráfico 33',header='C5',name='Diferencial interés-crecimiento',title_cell='B3',title='GRÁFICO 33. DIFERENCIAL INTERÉS-CRECIMIENTO, (PUNTOS PIB)',unit='percentage_points_gdp',parts={'chart42_s1':("'Gráfico 33'!$B$6:$B$80","'Gráfico 33'!$C$6:$C$60"),'chart42_s2':("'Gráfico 33'!$B$6:$B$80","'Gráfico 33'!$C$6:$C$35")}),
 ('2025','Gráfico 70'):dict(sheet='Gráfico 70',header='C5',name='Vida media de la deuda del Estado en circulación (años)',title_cell='B2',title='Gráfico 70. VIDA MEDIA DE LA DEUDA DEL ESTADO (AÑOS)',unit='years',parts={'chart122_s1':("'Gráfico 70'!$B$7:$B$31","'Gráfico 70'!$C$7:$C$31"),'chart123_s1':("'Gráfico 70'!$B$7:$B$31","'Gráfico 70'!$C$7:$C$31")})}

class HeaderResolver(FormulaResolver):
    def profile(self,row):
        p=SPECS.get((row['book'],row['bloque']));require(p is not None,'Unreviewed header scope')
        require(row['serie_id'] in p['parts'],'Unreviewed physical series')
        source=self.sources[row['book']]
        require(source.text(p['sheet'],p['title_cell'])==p['title'],'Source measure title changed')
        require(source.text(p['sheet'],p['header'])==p['name'],'Source measure header changed')
        return p
    def binding(self,row):
        key=(row['book'],row['parte_grafico_imagen'],row['serie_id'])
        if key in self.bindings:return self.bindings[key]
        source=self.sources[row['book']];part=row['parte_grafico_imagen']
        m=re.fullmatch(r'chart([0-9]+)_s([1-9][0-9]*)',row['serie_id'] or '')
        require(m is not None and part==f'xl/charts/chart{m[1]}.xml','Series identity')
        root=ET.fromstring(source.xml.parts[part]);node=root.findall('.//c:ser',C)[int(m[2])-1]
        p=self.profile(row)
        actual_cat=node.find('c:cat/c:numRef/c:f',C)
        if actual_cat is None:actual_cat=node.find('c:cat/c:strRef/c:f',C)
        actual_val=node.find('c:val/c:numRef/c:f',C)
        require(actual_cat is not None and actual_val is not None and (actual_cat.text,actual_val.text)==p['parts'][row['serie_id']],'Header series references changed')
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
                require(not (isinstance(label,str) and label.startswith('=')),'Formula names outside header contract')
                if isinstance(label,(int,float,Decimal)) and not isinstance(label,bool):
                    require(Decimal(str(label)).is_finite(),'Nonfinite series name')
                    label=str(int(label)) if label==int(label) else str(label)
                require(isinstance(label,str) and label.strip(),'Invalid name component');parts.append(label)
            label=' | '.join(parts)
            name_basis='complete_source_header_range' if len(nc)>1 else 'source_scalar'
        else:
            p=self.profile(row)
            require(node.find('c:tx',C) is None,'Header recovery requires absent chart name')
            label=source.text(p['sheet'],p['header']);require(label==p['name'],'Source measure header changed')
            name_ref=f"'{p['sheet']}'!{p['header']}";name_basis='reviewed_source_measure_header'
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
        if (row['book'],row['bloque'])==('2026','Gráfico 33'):
            require(row['titulo']==row['metrica_literal']=='Diferencial interés-crecimiento (p.p.)','Universal differential title changed')
            p=self.profile(row);return [(p['title_cell'],p['title'])]
        return super().title_proof(row,texts)

    def unit_spec(self,row,texts,binding):
        p=self.profile(row);require(binding['multiplier']==binding['divisor']==1,'Header unit scale conflict')
        return p['name'],p['unit'],'explicit_source_measure_header',(p['title_cell'],p['title'])

    def review_header(self,row):
        if row['serie_literal'] is not None:return 'unexpected_existing_series_name',None
        try:
            p=self.profile(row);b=self.binding(row)
        except (ValueError,KeyError,IndexError,ArithmeticError,AttributeError):return 'header_evidence_review',None
        shadow=dict(row,serie_literal=b['label']);status,result=SourceResolver.review(self,shadow)
        if result is None:return status,None
        evidence=json.dumps(dict(header=p['header'],name=p['name'],title_cell=p['title_cell'],value_ref=b['value_ref'],category_ref=b['category_ref'],name_origin='reviewed_source_measure_header',universal_series=None,scope='source_quantity_as_labelled'),ensure_ascii=False,sort_keys=True)
        return 'usable_header_name',result+(row['periodo_categoria_literal'],'{}',None,row['hoja_valor'],row['celda_valor'],'{}','source_measure_header','source_category_as_labelled',evidence)

def collect(con):
    require(con.execute('SELECT count(*) FROM formula_usable_keys').fetchone()[0]==38011,'Previous count changed')
    c=con.execute('SELECT * FROM formula_numeric_queue ORDER BY book,row_number');ns=[n[0] for n in c.description];data=c.fetchall();require(len(data)==9202,'Queue changed')
    r=HeaderResolver();rows=[];reviews=[]
    try:
        for v in data:
            row=dict(zip(ns,v));status=row['quantity_reason'];result=None
            if (row['book'],row['bloque']) in SPECS:status,result=r.review_header(row)
            reviews.append((row['book'],row['row_number'],status))
            if result is not None:rows.append(result)
    finally:r.close()
    return rows,reviews

def build(output):
    require(not output.exists(),'Output exists');require(sha(PREVIOUS)==PREVIOUS_SHA,'Previous hash changed')
    output.mkdir(parents=True);db=output/'header_names.duckdb';shutil.copyfile(PREVIOUS,db)
    with duckdb.connect(str(db)) as c:
        rows,reviews=collect(c)
        c.execute('CREATE TABLE header_observations AS SELECT * FROM formula_observations WHERE false')
        c.executemany('INSERT INTO header_observations VALUES ('+','.join('?' for _ in FIELDS)+')',rows)
        c.execute('CREATE TABLE header_review(book VARCHAR,row_number INTEGER,status VARCHAR,PRIMARY KEY(book,row_number))');c.executemany('INSERT INTO header_review VALUES (?,?,?)',reviews)
        c.execute('CREATE VIEW header_usable_keys AS SELECT * FROM formula_usable_keys UNION ALL SELECT book,row_number FROM header_observations')
        c.execute("CREATE VIEW header_numeric_queue AS SELECT s.*,r.status AS header_reason FROM formula_numeric_queue s JOIN header_review r USING(book,row_number) WHERE r.status<>'usable_header_name'")
    report=dict(status='built_pending_validation',added=len(rows),admitted=38011+len(rows),numeric_pending=9202-len(rows),previous_sha256=PREVIOUS_SHA,code_sha256=sha(Path(__file__)),db_sha256=sha(db),rag_integrated=False)
    (output/'manifest.json').write_text(json.dumps(report,indent=2),encoding='utf-8');return report

def check_new(c):
    rows,reviews=collect(c)
    for table,expected in [('header_observations',rows),('header_review',reviews)]:
        actual=c.execute('SELECT * FROM '+table).fetchall();require(len(actual)==len(expected) and set(actual)==set(expected),'Header evidence mismatch: '+table)
    for view,sql in [('header_usable_keys','SELECT * FROM formula_usable_keys UNION ALL SELECT book,row_number FROM header_observations'),('header_numeric_queue',"SELECT s.*,r.status AS header_reason FROM formula_numeric_queue s JOIN header_review r USING(book,row_number) WHERE r.status<>'usable_header_name'")]:
        require(c.execute('SELECT * FROM '+view+' LIMIT 0').description==c.execute(sql+' LIMIT 0').description,'View columns changed')
        for a,b in [(view,'('+sql+')'),('('+sql+')',view)]:require(not c.execute(f'SELECT * FROM {a} EXCEPT ALL SELECT * FROM {b}').fetchall(),'View multiplicity changed')
    require(c.execute('SELECT count(*) FROM (SELECT book,row_number,count(*) n FROM header_usable_keys GROUP BY ALL HAVING n<>1)').fetchone()[0]==0,'Admission overlap')
    return rows

def validate(db):
    previous=validate_previous(db)
    with duckdb.connect(str(db),read_only=True) as c:rows=check_new(c)
    return dict(status='pass',added=len(rows),admitted=38011+len(rows),numeric_pending=9202-len(rows),previous_validation=previous['status'],code_sha256=sha(Path(__file__)),db_sha256=sha(Path(db)),rag_integrated=False)

def query(db,*,book,block,series_id,point_index):
    require((book,block) in SPECS and isinstance(series_id,str) and series_id.strip() and type(point_index) is int and point_index>=0,'Exact header-name identity required')
    with duckdb.connect(str(db),read_only=True) as c:
        q=c.execute('SELECT t.*,s.xlsx_path,s.xlsx_sha256,s.pdf_path,s.pdf_sha256 FROM header_observations t JOIN sources s USING(book) WHERE t.book=? AND block=? AND series_id=? AND point_index=?',[book,block,series_id,point_index]);ns=[n[0] for n in q.description];rows=q.fetchall()
    if len(rows)!=1:raise LookupError('No unique header-name point; no fallback')
    return {n:str(v) if isinstance(v,Decimal) else v for n,v in zip(ns,rows[0])}

def main():
    p=argparse.ArgumentParser();s=p.add_subparsers(dest='command',required=True)
    b=s.add_parser('build');b.add_argument('--output-dir',type=Path,required=True)
    b=s.add_parser('validate');b.add_argument('--db',type=Path,required=True);b.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.command=='build':report=build(a.output_dir)
    else:
        require(not a.output.exists(),'Report exists')
        try:report=validate(a.db)
        except Exception as e:report=dict(status='fail',error=str(e))
        a.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2));return int(report['status']=='fail')
if __name__=='__main__':raise SystemExit(main())
