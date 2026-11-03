"""Evidence-bounded quantities and verified scatter pairs with unresolved units."""
from __future__ import annotations
import argparse
from decimal import Decimal
import json
from pathlib import Path
import re
import shutil
import xml.etree.ElementTree as ET
import duckdb
from experiments.reconciled_semantics import Reconciler,FIELDS as OLD_FIELDS,validate as validate_previous
from experiments.source_semantics import SourceResolver
from experiments.resolved_semantics import canon
from experiments.scaled_semantics import format_effect
from experiments.bulk_semantics import C,ref_cells
from experiments.indexed_semantics import points
from experiments.semantic_families import decimal
from experiments.excel_pilot import ROOT,sha,PDF_HASHES
from experiments.universal_database import require

PREVIOUS=ROOT/'evaluation/reconciled_semantics_2026-10-31/run_v3/reconciled.duckdb'
PREVIOUS_SHA='67259746e56544940e22cf2c242492a3342eebcec70b2e49df9b9dc4e780d410'
PROFILES=Path(__file__).with_name('quantity_profiles.json')
PROFILES_SHA='cc8fa3dd3c0349bed249872c2358b9b6114b1fc467beeede95c47605fbbe3f0b'
QUANTITIES={(p['book'],p['block']):p for p in json.loads(PROFILES.read_text(encoding='utf-8'))}
FIELDS=OLD_FIELDS+'quantity_role category_semantics quantity_evidence'.split()
PAIR_FIELDS='book row_number source_id block series_id series_literal point_index raw_x raw_y x_cell y_cell year_label year_cell x_ref y_ref x_axis_title y_axis_title universal_x_label unit_status evidence'.split()
SCATTER_TITLE='RELACIÓN ENTRE PIB Y POBLACIÓN EN EDAD DE TRABAJAR EN LAS ECONOMÍAS DESARROLLADAS (1971 2023'


class QuantityResolver(Reconciler):
    def __init__(self):
        require(sha(PROFILES)==PROFILES_SHA,'Quantity profiles changed')
        super().__init__();self.quantity_proofs={};self.scatter_bindings={};self.decompositions={}

    def prove(self,row):
        key=row['book'],row['bloque'];p=QUANTITIES[key];source=self.sources[row['book']]
        require(row['titulo']==row['metrica_literal']==p['universal_title'],'Quantity title identity changed')
        require(all(source.text(row['bloque'],a)==t for a,t in p['headers']),'Quantity source headers changed')
        if key not in self.quantity_proofs:
            proof=dict(headers=p['headers'],pdfs=[])
            for item in p['pdfs']:proof['pdfs'].append(self.pdf(row['book'],item['page'],item['fragments']))
            self.quantity_proofs[key]=proof
        return self.quantity_proofs[key]

    def decomposition(self,row,binding):
        p=QUANTITIES[row['book'],row['bloque']]
        if not p['additivity']:return None
        component={'Crecimiento por demografía':0,'Crecimiento por otros factores':1,'Total':2}[row['serie_literal']]
        actual_sheet,actual_cells=ref_cells(binding['value_ref']);matches=[]
        for index,group in enumerate(p['additivity']):
            sheet,cells=ref_cells(group[component])
            if actual_sheet==sheet and set(actual_cells)<=set(cells):matches.append((index,group,cells))
        require(len(matches)==1,'Decomposition references do not identify one reviewed horizon')
        index,group,cells=matches[0];key=row['book'],row['bloque'],index
        if key not in self.decompositions:
            refs=[ref_cells(v) for v in group];require(all(s==row['bloque'] for s,_ in refs),'Foreign growth decomposition')
            require(len({len(c) for _,c in refs})==1,'Growth component lengths differ')
            valid=[];empty=[];residuals=[];conflicts=[];source=self.sources[row['book']]
            for i,triple in enumerate(zip(*(c for _,c in refs))):
                values=[source.number(row['bloque'],cell) for cell in triple]
                if all(v is None for v in values):empty.append(i);continue
                require(all(v is not None for v in values),'Partial decomposition triplet')
                residual=values[0]+values[1]-values[2];residuals.append(abs(residual));valid.append(i)
                if abs(residual)>Decimal('1e-12'):conflicts.append(dict(index=i,cells=list(triple),values=list(map(str,values)),residual=str(residual)))
            require(valid,'No complete decomposition points')
            self.decompositions[key]=dict(refs=group,valid_indices=valid,empty_indices=empty,max_absolute_residual=str(max(residuals)),conflicts=conflicts,status='conflict' if conflicts else 'pass')
        proof=self.decompositions[key]
        point=re.fullmatch(re.escape(row['serie_id'])+r'_p([0-9]+)',row['id'] or '');require(point is not None,'Decomposition point identity')
        i=int(point[1]);require(i<len(binding['cells']) and binding['cells'][i] in cells,'Decomposition point outside reviewed grid')
        require(cells.index(binding['cells'][i]) in proof['valid_indices'],'Decomposition point lacks complete triplet')
        return proof

    def title_proof(self,row,texts):
        if (row['book'],row['bloque']) in QUANTITIES:
            self.prove(row);self.title_modes[row['book'],row['bloque']]='reviewed_quantity_profile_headers'
            return QUANTITIES[row['book'],row['bloque']]['headers']
        return super().title_proof(row,texts)

    def unit_spec(self,row,texts,binding):
        key=row['book'],row['bloque']
        if key in QUANTITIES:
            p=QUANTITIES[key];unit=p['units'].get(row['serie_literal']);require(unit is not None,'Unreviewed quantity role')
            proof=dict(self.prove(row));decomposition=self.decomposition(row,binding)
            if decomposition is not None:
                require(decomposition['status']=='pass','Growth decomposition conflict');proof['additivity']=decomposition
            self.supplements[key]=proof
            require(binding['multiplier']==1 and binding['divisor']==1,'Unexpected source quantity scale')
            return 'unidad del perfil de magnitud acreditado',unit,'reviewed_quantity_source_profile',('quantity_source_profile',json.dumps(proof,ensure_ascii=False,sort_keys=True))
        return super().unit_spec(row,texts,binding)

    def review_quantity(self,row):
        key=row['book'],row['bloque'];p=QUANTITIES[key]
        if row['serie_literal'] not in p['units']:return 'quantity_role_not_admitted',None
        if p['additivity']:
            proof=self.decomposition(row,self.binding(row))
            if proof['status']!='pass':return 'growth_decomposition_conflict',None
        status,result=SourceResolver.review(self,row)
        if result is None:return status,None
        role=p['roles'].get(row['serie_literal'],'source_quantity_as_labelled')
        proof=dict(self.supplements.get(key,self.prove(row)),unit=p['units'][row['serie_literal']],role=role,category_semantics=p['category_semantics'])
        return 'usable_quantity',result+(row['periodo_categoria_literal'],'{}',row['serie_literal'],row['hoja_valor'],row['celda_valor'],'{}',role,p['category_semantics'],json.dumps(proof,ensure_ascii=False,sort_keys=True))

    def scatter_binding(self,row):
        key=row['serie_id']
        source=self.sources['2025'];sheet='Gráfico 15';part='xl/charts/chart36.xml'
        require(row['book']=='2025' and row['bloque']==sheet and row['parte_grafico_imagen']==part,'Scatter source identity')
        require(source.text(sheet,'I6')==SCATTER_TITLE and row['titulo']==row['metrica_literal']==SCATTER_TITLE,'Scatter title evidence')
        if key in self.scatter_bindings:return self.scatter_bindings[key]
        match=re.fullmatch('chart36_s([12])',key or '');require(match is not None,'Unreviewed scatter series')
        root=ET.fromstring(source.xml.parts[part]);node=root.findall('.//c:ser',C)[int(match[1])-1]
        group=next(p for p in root.iter() if node in list(p));require(group.tag.endswith('}scatterChart'),'Not a scatter group')
        ids={n.get('val') for n in group.findall('c:axId',C)};axes={}
        for axis in root.findall('.//c:valAx',C):
            if axis.find('c:axId',C).get('val') not in ids:continue
            position=axis.find('c:axPos',C).get('val');require(position not in axes,'Duplicate scatter axis position')
            fmt=axis.find('c:numFmt',C);require(fmt is not None and format_effect(fmt.get('formatCode'))==(Decimal(1),Decimal(1)) and axis.find('c:dispUnits',C) is None,'Scatter scaling unresolved')
            title=axis.find('c:title',C);axes[position]=dict(id=axis.find('c:axId',C).get('val'),title=''.join(n.text or '' for n in title.iter() if n.tag.endswith('}t')) if title is not None else None,format=fmt.get('formatCode'))
        require(set(axes)=={'b','l'},'Scatter X/Y axis orientation unproved')
        data={}
        for coordinate in ['x','y']:
            ref=node.find(f'c:{coordinate}Val/c:numRef',C);require(ref is not None and ref.find('c:f',C) is not None,'Scatter reference missing')
            formula=ref.find('c:f',C).text;ss,cells=ref_cells(formula);require(ss==sheet,'Foreign scatter cells')
            cache=ref.find('c:numCache',C);require(cache is not None,'Scatter cache missing');pv=points(cache)
            count=cache.find('c:ptCount',C);require(count is not None and int(count.get('val'))==len(cells) and set(pv)==set(range(len(cells))),'Scatter indices incomplete')
            data[coordinate]=dict(ref=formula,cells=cells,cache=pv)
        require(len(data['x']['cells'])==len(data['y']['cells']),'Scatter pair lengths differ')
        name=node.find('c:tx/c:strRef/c:f',C);require(name is not None,'Scatter name reference missing')
        ss,nc=ref_cells(name.text);require(ss==sheet and len(nc)==1,'Scatter name not scalar')
        literal=source.text(sheet,nc[0]);require(literal==row['serie_literal'],'Scatter series label changed')
        xcol=re.match('[A-Z]+',data['x']['cells'][0])[0];ycol=re.match('[A-Z]+',data['y']['cells'][0])[0]
        require(source.text(sheet,xcol+'5')==axes['b']['title']=='Crecimiento del PIB','Scatter X title changed')
        require(source.text(sheet,ycol+'5')==axes['l']['title']=='Población en edad de trabajar','Scatter Y title changed')
        b=dict(data=data,axes=axes,label=literal,name_ref=name.text);self.scatter_bindings[key]=b;return b

    def scatter_pair(self,row):
        b=self.scatter_binding(row);source=self.sources['2025'];sheet='Gráfico 15'
        require(row['serie_literal']==b['label'],'Scatter label mismatch')
        match=re.fullmatch(re.escape(row['serie_id'])+'_p([0-9]+)',row['id'] or '');require(match is not None,'Scatter point ID invalid');i=int(match[1])
        require(i<len(b['data']['x']['cells']),'Scatter point outside source')
        xc=b['data']['x']['cells'][i];yc=b['data']['y']['cells'][i]
        require(re.search('[0-9]+',xc)[0]==re.search('[0-9]+',yc)[0],'Scatter source rows not paired')
        xv=source.number(sheet,xc);yv=source.number(sheet,yc);require(xv is not None and yv is not None,'Scatter coordinate missing')
        require(decimal(b['data']['x']['cache'][i])==xv and decimal(b['data']['y']['cache'][i])==yv,'Scatter cache/cell conflict')
        require(row['hoja_valor']==sheet and row['celda_valor']==yc and decimal(row['decimal_XML_texto'])==yv,'Scatter staging Y mismatch')
        require(decimal(row['periodo_categoria_literal']) in [xv,Decimal(str(float(xv)))],'Scatter universal X encoding conflict')
        year_cell='B'+re.search('[0-9]+',yc)[0];require(source.text(sheet,'B5')=='year','Scatter year header missing')
        year=source.number(sheet,year_cell);require(year is not None and year==int(year),'Scatter year label invalid')
        evidence=json.dumps(dict(axes=b['axes'],name_ref=b['name_ref'],x_cache_index=i,y_cache_index=i,year_header='B5',scope='geometry_verified_units_unresolved',country_identity='Japan_from_series_label' if b['label']=='Japón' else 'not_available_per_point'),ensure_ascii=False,sort_keys=True)
        return ('2025',row['row_number'],row['id'],sheet,row['serie_id'],b['label'],i,xv,yv,xc,yc,str(int(year)),year_cell,b['data']['x']['ref'],b['data']['y']['ref'],b['axes']['b']['title'],b['axes']['l']['title'],row['periodo_categoria_literal'],'not_explicitly_stated',evidence)


def collect(con):
    require(con.execute('SELECT count(*) FROM reconciled_usable_keys').fetchone()[0]==36905,'Previous admissions changed')
    c=con.execute('SELECT * FROM reconciled_numeric_queue ORDER BY book,row_number');names=[n[0] for n in c.description];data=c.fetchall();require(len(data)==10308,'Pending inventory changed')
    resolver=QuantityResolver();rows=[];pairs=[];decisions=[]
    try:
        for values in data:
            row=dict(zip(names,values));key=row['book'],row['bloque'];status=row['reconciled_reason'];result=None
            if key in QUANTITIES:
                try:status,result=resolver.review_quantity(row)
                except (ValueError,ArithmeticError,KeyError,AttributeError,IndexError,TypeError):status='quantity_evidence_review'
                if result is not None:rows.append(result)
            elif key==('2025','Gráfico 15'):
                try:pairs.append(resolver.scatter_pair(row));status='scatter_units_unresolved'
                except (ValueError,ArithmeticError,KeyError,AttributeError,IndexError,TypeError,StopIteration):status='scatter_pair_evidence_review'
            decisions.append((row['book'],row['row_number'],status))
        pages=[(b,p,PDF_HASHES[b],t) for (b,p),t in sorted(resolver.pdf_pages.items())]
        decomposition_reviews=[(b,block,group,json.dumps(e,ensure_ascii=False,sort_keys=True)) for (b,block,group),e in sorted(resolver.decompositions.items())]
    finally:resolver.close()
    return rows,pairs,decisions,pages,decomposition_reviews


def build(output):
    require(not output.exists(),'Output exists');require(sha(PREVIOUS)==PREVIOUS_SHA,'Previous database changed')
    output.mkdir(parents=True);db=output/'quantities.duckdb';shutil.copyfile(PREVIOUS,db)
    with duckdb.connect(str(db)) as con:
        rows,pairs,decisions,pages,decomposition_reviews=collect(con)
        con.execute('CREATE TABLE quantity_observations AS SELECT *,NULL::VARCHAR AS quantity_role,NULL::VARCHAR AS category_semantics,NULL::VARCHAR AS quantity_evidence FROM reconciled_observations WHERE false')
        if rows:con.executemany('INSERT INTO quantity_observations VALUES ('+','.join('?' for _ in FIELDS)+')',rows)
        types=['DECIMAL(38,24)' if n in ['raw_x','raw_y'] else ('INTEGER' if n in ['row_number','point_index'] else 'VARCHAR') for n in PAIR_FIELDS]
        con.execute('CREATE TABLE reviewed_scatter_pairs('+','.join(n+' '+t for n,t in zip(PAIR_FIELDS,types))+',PRIMARY KEY(book,row_number))')
        if pairs:con.executemany('INSERT INTO reviewed_scatter_pairs VALUES ('+','.join('?' for _ in PAIR_FIELDS)+')',pairs)
        con.execute('CREATE TABLE quantity_review(book VARCHAR,row_number INTEGER,status VARCHAR,PRIMARY KEY(book,row_number))');con.executemany('INSERT INTO quantity_review VALUES (?,?,?)',decisions)
        con.execute('CREATE TABLE quantity_pdf_evidence(book VARCHAR,page INTEGER,pdf_sha256 VARCHAR,text VARCHAR,PRIMARY KEY(book,page))')
        if pages:con.executemany('INSERT INTO quantity_pdf_evidence VALUES (?,?,?,?)',pages)
        con.execute('CREATE TABLE growth_decomposition_reviews(book VARCHAR,block VARCHAR,source_group INTEGER,evidence VARCHAR,PRIMARY KEY(book,block,source_group))')
        if decomposition_reviews:con.executemany('INSERT INTO growth_decomposition_reviews VALUES (?,?,?,?)',decomposition_reviews)
        con.execute('CREATE VIEW quantity_usable_keys AS SELECT * FROM reconciled_usable_keys UNION ALL SELECT book,row_number FROM quantity_observations')
        con.execute("CREATE VIEW quantity_numeric_queue AS SELECT s.*,r.status AS quantity_reason FROM reconciled_numeric_queue s JOIN quantity_review r USING(book,row_number) WHERE r.status<>'usable_quantity'")
    report=dict(status='built_pending_validation',added=len(rows),admitted=36905+len(rows),numeric_pending=10308-len(rows),reviewed_scatter_pairs=len(pairs),previous_sha256=PREVIOUS_SHA,code_sha256=sha(Path(__file__)),profiles_sha256=sha(PROFILES),db_sha256=sha(db),rag_integrated=False)
    (output/'manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');return report


def check_new(con):
    rows,pairs,decisions,pages,decomposition_reviews=collect(con)
    for table,expected in [('quantity_observations',rows),('reviewed_scatter_pairs',pairs),('quantity_review',decisions),('quantity_pdf_evidence',pages),('growth_decomposition_reviews',decomposition_reviews)]:
        actual=con.execute('SELECT * FROM '+table).fetchall();require(len(actual)==len(expected) and set(actual)==set(expected),'Quantity evidence mismatch: '+table)
    for view,sql in [('quantity_usable_keys','SELECT * FROM reconciled_usable_keys UNION ALL SELECT book,row_number FROM quantity_observations'),('quantity_numeric_queue',"SELECT s.*,r.status AS quantity_reason FROM reconciled_numeric_queue s JOIN quantity_review r USING(book,row_number) WHERE r.status<>'usable_quantity'")]:
        require(con.execute('SELECT * FROM '+view+' LIMIT 0').description==con.execute(sql+' LIMIT 0').description,'View columns changed')
        for a,b in [(view,'('+sql+')'),('('+sql+')',view)]:require(not con.execute(f'SELECT * FROM {a} EXCEPT ALL SELECT * FROM {b}').fetchall(),'View multiplicity changed')
    require(con.execute('SELECT count(*) FROM (SELECT book,row_number,count(*) n FROM quantity_usable_keys GROUP BY ALL HAVING n<>1)').fetchone()[0]==0,'Admission overlap')
    require(con.execute('SELECT count(*) FROM reviewed_scatter_pairs JOIN quantity_usable_keys USING(book,row_number)').fetchone()[0]==0,'Unresolved scatter was admitted')
    return rows,pairs


def validate(db):
    previous=validate_previous(db)
    with duckdb.connect(str(db),read_only=True) as con:
        rows,pairs=check_new(con);blocks=con.execute('SELECT count(*) FROM (SELECT DISTINCT book,block FROM quantity_observations)').fetchone()[0]
        stats=con.execute('SELECT unit,quantity_role,category_semantics,count(*) FROM quantity_observations GROUP BY ALL ORDER BY count(*) DESC').fetchall();reasons=con.execute('SELECT status,count(*) FROM quantity_review GROUP BY ALL ORDER BY count(*) DESC').fetchall()
    return dict(status='pass',added=len(rows),admitted=36905+len(rows),numeric_pending=10308-len(rows),reviewed_scatter_pairs=len(pairs),blocks=blocks,stats=stats,review=reasons,previous_validation=previous['status'],code_sha256=sha(Path(__file__)),profiles_sha256=sha(PROFILES),db_sha256=sha(Path(db)),rag_integrated=False)


def query(db,*,book,block,series_id,point_index):
    require(book in ['2025','2026'] and all(isinstance(t,str) and t.strip() for t in [block,series_id]) and type(point_index) is int and point_index>=0,'Exact quantity identity required')
    with duckdb.connect(str(db),read_only=True) as con:
        c=con.execute('SELECT t.*,s.xlsx_path,s.xlsx_sha256,s.pdf_path,s.pdf_sha256 FROM quantity_observations t JOIN sources s USING(book) WHERE t.book=? AND block=? AND series_id=? AND point_index=?',[book,block,series_id,point_index]);names=[n[0] for n in c.description];rows=c.fetchall()
    if len(rows)!=1:raise LookupError('No unique quantity point; no fallback')
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
