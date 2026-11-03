"""Restricted source-pinned textual names; no RAG integration."""
import argparse,json,re,shutil,xml.etree.ElementTree as ET
from pathlib import Path
from decimal import Decimal
import duckdb
from experiments.quantity_semantics import QuantityResolver,FIELDS,PREVIOUS as _unused,validate as validate_previous
from experiments.source_semantics import SourceResolver
from experiments.indexed_semantics import points
from experiments.scaled_semantics import DIVISORS,format_effect
from experiments.bulk_semantics import C,ref_cells,norm
from experiments.semantic_families import decimal,M
from experiments.excel_pilot import ROOT,sha
from experiments.universal_database import require
PREVIOUS=ROOT/'evaluation/quantity_semantics_2026-10-31/run_v2/quantities.duckdb'
PREVIOUS_SHA='797b64b9782aa55a0d5b09424c3e93c3d4c6c7fe20934cf7953f2651f28ca82e'
TITLE='ESCENARIO CENTRAL DEL PIB. TASAS DE CRECIMIENTO DEL PIB REAL, NOMINAL Y DEL DEFLACTOR (%)'

class FormulaResolver(QuantityResolver):
    def formula_name(self,source,sheet,cell,formula):
        require(sheet=='Gráfico 27' and cell in ['C4','D4','E4'],'Unreviewed formula-name cell')
        expected={'C4':'F4','D4':'G4','E4':'H4'}[cell]
        require(formula=='='+expected+'&" observ"','Unsupported name formula')
        base=source.text(sheet,expected);require(isinstance(base,str) and not base.startswith('='),'Name input not literal text')
        node=source.sheet(sheet)[0][cell];v=node.find('m:v',M)
        require(node.get('t')=='str' and v is not None and v.text==base+' observ','Text name cache mismatch')
        return v.text
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
                if isinstance(label,str) and label.startswith('='):label=self.formula_name(source,ns,cell,label)
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

    def unit_spec(self,row,texts,binding):
        require(row['book']=='2025' and row['bloque']=='Gráfico 27' and row['titulo']==row['metrica_literal']==TITLE,'Formula-name scope changed')
        require(self.sources['2025'].text('Gráfico 27','L5')==TITLE,'Source growth caption changed')
        require(binding['multiplier']==binding['divisor']==1,'Unexpected growth scale')
        return 'crecimiento (%)','percent_growth_as_published','explicit_growth_caption',('L5',TITLE)

    def review_name(self,row):
        require(row['book']=='2025' and row['bloque']=='Gráfico 27','Unreviewed scope')
        status,result=SourceResolver.review(self,row)
        if result is None:return status,None
        name_ref=self.binding(row)['name_ref'];sheet,cells=ref_cells(name_ref)
        evidence=json.dumps(dict(name_ref=name_ref,formula=self.sources['2025'].text(sheet,cells[0]),rule='exact_local_A1_text_concatenation'),sort_keys=True)
        return 'usable_formula_name',result+(row['periodo_categoria_literal'],'{}',row['serie_literal'],row['hoja_valor'],row['celda_valor'],'{}','published_growth','source_category_as_labelled',evidence)

def collect(con):
    require(con.execute('SELECT count(*) FROM quantity_usable_keys').fetchone()[0]==37930,'Previous count changed')
    c=con.execute('SELECT * FROM quantity_numeric_queue ORDER BY book,row_number');ns=[n[0] for n in c.description];data=c.fetchall();require(len(data)==9283,'Queue changed')
    r=FormulaResolver();rows=[];reviews=[]
    try:
        for v in data:
            row=dict(zip(ns,v));status=row['quantity_reason'];result=None
            if (row['book'],row['bloque'])==('2025','Gráfico 27'):status,result=r.review_name(row)
            reviews.append((row['book'],row['row_number'],status))
            if result is not None:rows.append(result)
    finally:r.close()
    return rows,reviews

def build(output):
    require(not output.exists(),'Output exists');require(sha(PREVIOUS)==PREVIOUS_SHA,'Previous hash changed')
    output.mkdir(parents=True);db=output/'formula_names.duckdb';shutil.copyfile(PREVIOUS,db)
    with duckdb.connect(str(db)) as c:
        rows,reviews=collect(c)
        c.execute('CREATE TABLE formula_observations AS SELECT * FROM quantity_observations WHERE false')
        c.executemany('INSERT INTO formula_observations VALUES ('+','.join('?' for _ in FIELDS)+')',rows)
        c.execute('CREATE TABLE formula_review(book VARCHAR,row_number INTEGER,status VARCHAR,PRIMARY KEY(book,row_number))');c.executemany('INSERT INTO formula_review VALUES (?,?,?)',reviews)
        c.execute('CREATE VIEW formula_usable_keys AS SELECT * FROM quantity_usable_keys UNION ALL SELECT book,row_number FROM formula_observations')
        c.execute("CREATE VIEW formula_numeric_queue AS SELECT s.*,r.status AS formula_reason FROM quantity_numeric_queue s JOIN formula_review r USING(book,row_number) WHERE r.status<>'usable_formula_name'")
    report=dict(status='built_pending_validation',added=len(rows),admitted=37930+len(rows),numeric_pending=9283-len(rows),previous_sha256=PREVIOUS_SHA,code_sha256=sha(Path(__file__)),db_sha256=sha(db),rag_integrated=False)
    (output/'manifest.json').write_text(json.dumps(report,indent=2),encoding='utf-8');return report

def check_new(c):
    rows,reviews=collect(c)
    for table,expected in [('formula_observations',rows),('formula_review',reviews)]:
        actual=c.execute('SELECT * FROM '+table).fetchall();require(len(actual)==len(expected) and set(actual)==set(expected),'Formula evidence mismatch: '+table)
    for view,sql in [('formula_usable_keys','SELECT * FROM quantity_usable_keys UNION ALL SELECT book,row_number FROM formula_observations'),('formula_numeric_queue',"SELECT s.*,r.status AS formula_reason FROM quantity_numeric_queue s JOIN formula_review r USING(book,row_number) WHERE r.status<>'usable_formula_name'")]:
        require(c.execute('SELECT * FROM '+view+' LIMIT 0').description==c.execute(sql+' LIMIT 0').description,'View columns changed')
        for a,b in [(view,'('+sql+')'),('('+sql+')',view)]:require(not c.execute(f'SELECT * FROM {a} EXCEPT ALL SELECT * FROM {b}').fetchall(),'View multiplicity changed')
    require(c.execute('SELECT count(*) FROM (SELECT book,row_number,count(*) n FROM formula_usable_keys GROUP BY ALL HAVING n<>1)').fetchone()[0]==0,'Admission overlap')
    return rows

def validate(db):
    previous=validate_previous(db)
    with duckdb.connect(str(db),read_only=True) as c:rows=check_new(c)
    return dict(status='pass',added=len(rows),admitted=37930+len(rows),numeric_pending=9283-len(rows),previous_validation=previous['status'],code_sha256=sha(Path(__file__)),db_sha256=sha(Path(db)),rag_integrated=False)

def query(db,*,book,block,series_id,point_index):
    require(book=='2025' and block=='Gráfico 27' and isinstance(series_id,str) and series_id.strip() and type(point_index) is int and point_index>=0,'Exact formula-name identity required')
    with duckdb.connect(str(db),read_only=True) as c:
        q=c.execute('SELECT t.*,s.xlsx_path,s.xlsx_sha256,s.pdf_path,s.pdf_sha256 FROM formula_observations t JOIN sources s USING(book) WHERE t.book=? AND block=? AND series_id=? AND point_index=?',[book,block,series_id,point_index]);ns=[n[0] for n in q.description];rows=q.fetchall()
    if len(rows)!=1:raise LookupError('No unique formula-name point; no fallback')
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
