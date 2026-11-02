"""Bounded labour-unit and debt-contribution profiles; no RAG integration."""
import argparse,json,re,shutil
from pathlib import Path
from decimal import Decimal
import duckdb
from experiments.header_name_semantics import FIELDS,validate as validate_previous
from experiments.quantity_semantics import QuantityResolver
from experiments.source_semantics import SourceResolver
from experiments.grid_semantics import GridResolver,GRIDS
from experiments.excel_pilot import ROOT,sha,PDF_HASHES
from experiments.universal_database import require

PREVIOUS=ROOT/'evaluation/header_name_semantics_2026-10-31/run_v3/header_names.duckdb'
PREVIOUS_SHA='15a9cc3a2e9a2ddd4c20f7ca34a7343751fa01bc00558f74cd330820c18d9c02'
PREVIOUS_COUNT=38250
PREVIOUS_PENDING=8963
SPECS={('2025', 'Gráfico 31'): {'caption_cell': 'H4', 'caption': 'EVOLUCIÓN SMI 2018-2025 (%SUBIDA ACUMULADA Y EUROS)', 'universal_title': 'EVOLUCIÓN SMI 2018-2025 (%SUBIDA ACUMULADA Y EUROS)', 'units': {'SMI mensual (14 pagas)': ('euros_monthly_14_payments', 1, 'published_smi_monthly_level'), 'Subida acumulada desde 2018': ('percent_cumulative_smi_change_from_2018', 100, 'cumulative_smi_change')}, 'category_semantics': 'calendar_year_as_labelled'}, ('2025', 'Gráfico 32'): {'caption_cell': 'H4', 'caption': 'CRECIMIENTO DEL SALARIO MÍNIMO, NOMINAL Y REAL (DEFLACTADO CON IPC)', 'universal_title': 'CRECIMIENTO DEL SALARIO MÍNIMO, NOMINAL Y REAL (DEFLACTADO CON IPC)', 'units': {'Tasa de crecimiento nominal SMI ': ('percent_nominal_smi_growth', 100, 'nominal_smi_growth'), 'Tasa de crecimiento real SMI ': ('percent_real_smi_growth_cpi_deflated', 100, 'real_smi_growth')}, 'category_semantics': 'calendar_year_as_labelled'}, ('2025', 'Gráfico 33'): {'caption_cell': 'H5', 'caption': 'INCIDENCIA SMI (% DE TRABAJADORES AFECTADOS)', 'universal_title': 'INCIDENCIA SMI (% DE TRABAJADORES AFECTADOS)', 'units': {'Hasta el 100% del SMI': ('percent_workers_affected_as_labelled', 100, 'smi_incidence_as_labelled'), 'Hasta el 125% del SMI': ('percent_workers_affected_as_labelled', 100, 'smi_incidence_as_labelled')}, 'category_semantics': 'calendar_year_as_labelled'}, ('2025', 'Gráfico 24'): {'caption_cell': 'G5', 'caption': 'COMPARACIÓN TASAS DE PARTICIPACIÓN POR GRUPOS DE EDAD. INFORME DE ENVEJECIMIENTO - AIREF ', 'universal_title': 'COMPARACIÓN TASAS DE PARTICIPACIÓN POR GRUPOS DE EDAD. INFORME DE ENVEJECIMIENTO - AIREF ', 'units': {'AIReF': ('percent_plot_display', 100, 'participation_comparison_as_labelled'), 'Informe envejecimiento': ('percent_plot_display', 100, 'participation_comparison_as_labelled')}, 'category_semantics': 'age_group_as_labelled'}, ('2026', 'Gráfico 31.A.'): {'caption_cell': 'B3', 'caption': 'GRÁFICO 31. A. CONTRIBUCIONES A LA VARIACIÓN DE LA DEUDA (P.P), POR AÑOS ', 'universal_title': 'Contribuciones a la variación de la deuda (p. p.). Por años', 'units': {'Var. Deuda': ('percentage_points_gdp', 1, 'published_debt_change'), 'Diferencial interés-crecimiento': ('percentage_points_gdp', 1, 'debt_change_contribution'), 'Déficit primario y stock-flujo': ('percentage_points_gdp', 1, 'debt_change_contribution')}, 'category_semantics': 'calendar_year_as_labelled'}, ('2026', 'Gráfico 31.B.'): {'caption_cell': 'B3', 'caption': 'GRÁFICO 31. B. CONTRIBUCIONES A LA VARIACIÓN DE LA DEUDA (P.P), POR PERIODOS', 'universal_title': 'Contribuciones a la variación de la deuda (p. p.). Por periodos', 'units': {'Var. Deuda': ('percentage_points_gdp', 1, 'published_debt_change'), 'Diferencial interés-crecimiento': ('percentage_points_gdp', 1, 'debt_change_contribution'), 'Déficit primario y stock-flujo': ('percentage_points_gdp', 1, 'debt_change_contribution')}, 'category_semantics': 'period_as_labelled'}}

class LabourResolver(QuantityResolver):
    def scope(self,row):
        key=row['book'],row['bloque'];p=SPECS.get(key)
        require(p is not None,'Unreviewed labour scope')
        require(row['titulo']==row['metrica_literal']==p['universal_title'],'Labour metric identity changed')
        require(self.sources[row['book']].text(row['bloque'],p['caption_cell'])==p['caption'],'Labour source caption changed')
        require(row['serie_literal'] in p['units'],'Unreviewed labour series')
        return p

    def binding(self,row):
        self.scope(row)
        if row['book']=='2026':return GridResolver.binding(self,row)
        return SourceResolver.binding(self,row)

    def title_proof(self,row,texts):
        p=self.scope(row)
        return [(p['caption_cell'],p['caption'])]

    def unit_spec(self,row,texts,binding):
        p=self.scope(row);unit,multiplier,role=p['units'][row['serie_literal']]
        require(binding['divisor']==1 and binding['multiplier']==Decimal(multiplier),'Labour associated-axis scale conflict')
        evidence=dict(caption_cell=p['caption_cell'],caption=p['caption'],unit=unit,role=role,axis_id=binding['axis_id'],axis_format=binding['axis_format'],multiplier=multiplier)
        if row['book']=='2026':
            evidence['pdf']=self.pdf('2026',71,['ratio de deuda pública','123 % del PIB','CONTRIBUCIONES A LA VARIACIÓN DE LA DEUDA (P. P.), POR AÑOS Y PERIODOS'])
        elif row['bloque'] in ['Gráfico 31','Gráfico 32']:
            evidence['pdf']=self.pdf('2025',71,['el SMI ha pasado de 735 a 1.184 euros','subida acumulada del 61%','GRÁFICO 31. EVOLUCIÓN SMI 2018-2025','GRÁFICO 32. CRECIMIENTO DEL SALARIO MÍNIMO'])
        self.supplements[row['book'],row['bloque']]=evidence
        return 'unidad y eje del perfil laboral',unit,'reviewed_labour_associated_axis',(p['caption_cell'],json.dumps(evidence,ensure_ascii=False,sort_keys=True))

    def review_labour(self,row):
        try:
            p=self.scope(row);b=self.binding(row)
            point=re.fullmatch(re.escape(row['serie_id'])+r'_p([0-9]+)',row['id'] or '')
            if point is None:return 'point_identity_review',None
            index=int(point[1])
            if index>=len(b['categories']) or index not in b['valid_indices']:return 'labour_point_outside_source',None
            shadow=row;grid={}
            if b['category_basis']=='reviewed_source_grid':
                category=b['categories'][index]
                require(isinstance(category,str) and category.strip(),'Missing labour grid category')
                require(row['periodo_categoria_literal'] is None or row['periodo_categoria_literal']==category,'Labour grid category contradiction')
                shadow=dict(row,periodo_categoria_literal=category)
                grid=dict(origin='reviewed_original_grid',profile=GRIDS[row['book'],row['bloque']],category_ref=b['category_ref'],name_ref=b['name_ref'])
            status,result=SourceResolver.review(self,shadow)
            if result is None:return status,None
            unit,multiplier,role=p['units'][row['serie_literal']]
            evidence=dict(self.supplements[row['book'],row['bloque']],category_origin=b['category_basis'],grid=grid)
            return 'usable_labour_unit',result+(row['periodo_categoria_literal'],json.dumps(grid,ensure_ascii=False,sort_keys=True),row['serie_literal'],row['hoja_valor'],row['celda_valor'],'{}',role,p['category_semantics'],json.dumps(evidence,ensure_ascii=False,sort_keys=True))
        except (ValueError,ArithmeticError,KeyError,IndexError,AttributeError,TypeError,StopIteration):
            return 'labour_unit_evidence_review',None

def collect(c):
    require(c.execute('SELECT count(*) FROM header_usable_keys').fetchone()[0]==PREVIOUS_COUNT,'Previous admissions changed')
    q=c.execute('SELECT * FROM header_numeric_queue ORDER BY book,row_number');ns=[n[0] for n in q.description];data=q.fetchall();require(len(data)==PREVIOUS_PENDING,'Previous queue changed')
    r=LabourResolver();rows=[];reviews=[]
    try:
        for values in data:
            row=dict(zip(ns,values));status=row['header_reason'];result=None
            if (row['book'],row['bloque']) in SPECS:status,result=r.review_labour(row)
            reviews.append((row['book'],row['row_number'],status))
            if result is not None:rows.append(result)
        pages=[(b,p,PDF_HASHES[b],t) for (b,p),t in sorted(r.pdf_pages.items())]
    finally:r.close()
    return rows,reviews,pages

def build(output):
    require(not output.exists(),'Output exists');require(sha(PREVIOUS)==PREVIOUS_SHA,'Previous database changed')
    output.mkdir(parents=True);db=output/'labour_units.duckdb';shutil.copyfile(PREVIOUS,db)
    with duckdb.connect(str(db)) as c:
        rows,reviews,pages=collect(c)
        c.execute('CREATE TABLE labour_observations AS SELECT * FROM header_observations WHERE false')
        if rows:c.executemany('INSERT INTO labour_observations VALUES ('+','.join('?' for _ in FIELDS)+')',rows)
        c.execute('CREATE TABLE labour_review(book VARCHAR,row_number INTEGER,status VARCHAR,PRIMARY KEY(book,row_number))');c.executemany('INSERT INTO labour_review VALUES (?,?,?)',reviews)
        c.execute('CREATE TABLE labour_pdf_evidence(book VARCHAR,page INTEGER,pdf_sha256 VARCHAR,text VARCHAR,PRIMARY KEY(book,page))')
        if pages:c.executemany('INSERT INTO labour_pdf_evidence VALUES (?,?,?,?)',pages)
        c.execute('CREATE VIEW labour_usable_keys AS SELECT * FROM header_usable_keys UNION ALL SELECT book,row_number FROM labour_observations')
        c.execute("CREATE VIEW labour_numeric_queue AS SELECT s.*,r.status AS labour_reason FROM header_numeric_queue s JOIN labour_review r USING(book,row_number) WHERE r.status<>'usable_labour_unit'")
    report=dict(status='built_pending_validation',added=len(rows),admitted=PREVIOUS_COUNT+len(rows),numeric_pending=PREVIOUS_PENDING-len(rows),previous_sha256=PREVIOUS_SHA,code_sha256=sha(Path(__file__)),db_sha256=sha(db),rag_integrated=False)
    (output/'manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');return report

def check_new(c):
    rows,reviews,pages=collect(c)
    for table,expected in [('labour_observations',rows),('labour_review',reviews),('labour_pdf_evidence',pages)]:
        actual=c.execute('SELECT * FROM '+table).fetchall();require(len(actual)==len(expected) and set(actual)==set(expected),'Labour evidence mismatch: '+table)
    for view,sql in [('labour_usable_keys','SELECT * FROM header_usable_keys UNION ALL SELECT book,row_number FROM labour_observations'),('labour_numeric_queue',"SELECT s.*,r.status AS labour_reason FROM header_numeric_queue s JOIN labour_review r USING(book,row_number) WHERE r.status<>'usable_labour_unit'")]:
        require(c.execute('SELECT * FROM '+view+' LIMIT 0').description==c.execute(sql+' LIMIT 0').description,'View columns changed')
        for a,b in [(view,'('+sql+')'),('('+sql+')',view)]:require(not c.execute(f'SELECT * FROM {a} EXCEPT ALL SELECT * FROM {b}').fetchall(),'View multiplicity changed')
    require(c.execute('SELECT count(*) FROM (SELECT book,row_number,count(*) n FROM labour_usable_keys GROUP BY ALL HAVING n<>1)').fetchone()[0]==0,'Admission overlap')
    return rows

def validate(db):
    previous=validate_previous(db)
    with duckdb.connect(str(db),read_only=True) as c:
        rows=check_new(c);stats=c.execute('SELECT book,block,unit,semantic_scope,count(*) FROM labour_observations GROUP BY ALL ORDER BY book,block,unit').fetchall()
    return dict(status='pass',added=len(rows),admitted=PREVIOUS_COUNT+len(rows),numeric_pending=PREVIOUS_PENDING-len(rows),stats=stats,previous_validation=previous['status'],code_sha256=sha(Path(__file__)),db_sha256=sha(Path(db)),rag_integrated=False)

def query(db,*,book,block,series_id,point_index):
    require((book,block) in SPECS and isinstance(series_id,str) and series_id.strip() and type(point_index) is int and point_index>=0,'Exact labour identity required')
    with duckdb.connect(str(db),read_only=True) as c:
        q=c.execute('SELECT t.*,s.xlsx_path,s.xlsx_sha256,s.pdf_path,s.pdf_sha256 FROM labour_observations t JOIN sources s USING(book) WHERE t.book=? AND block=? AND series_id=? AND point_index=?',[book,block,series_id,point_index]);ns=[n[0] for n in q.description];rows=q.fetchall()
    if len(rows)!=1:raise LookupError('No unique labour point; no fallback')
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
        a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2));return int(report['status']=='fail')
if __name__=='__main__':raise SystemExit(main())
