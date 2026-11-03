"""Reviewed fiscal table and annual chart families; no RAG integration."""
from __future__ import annotations

import argparse
from decimal import Decimal, ROUND_HALF_UP, localcontext
import hashlib
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import duckdb
import openpyxl

from experiments.excel_pilot import ROOT, BOOKS, PDF_HASHES, ExcelXML, column, sha
from experiments.universal_database import validate as validate_base, require

BASE = ROOT / "evaluation/universal_database_2026-10-31/run_v3/universal.duckdb"
BASE_SHA = "1ea42682bac9efe2374eeee72ec8391e0aac40c29818cbef03192609a2c8e329"
M = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
C = {"c": "http://schemas.openxmlformats.org/drawingml/2006/chart"}
# Explicit review: names, references, direction and support are not inferred from values.
GRAPHS = {
    19: (27, "Recursos", "Evolución de los recursos hasta 2050 (% PIB)",
         "GRÁFICO 19. EVOLUCIÓN DE LOS RECURSOS HASTA 2050 (% PIB)", [(6,"Histórico",2015),(8,"AIReF 2026",2025),(7,"Opinión 2025",2022)]),
    23: (31, "Sanidad", "Evolución del gasto en sanidad (% PIB)",
         "GRÁFICO 23. EVOLUCIÓN DEL GASTO EN SANIDAD (% PIB)", [(6,"Histórico",2015),(7,"AIReF 2026",2025),(8,"Opinión 2025",2025)]),
    24: (32, "Cuidados de larga duración", "Evolución del gasto en cuidados de larga duración (% PIB)",
         "GRÁFICO 24. EVOLUCIÓN DEL GASTO EN CUIDADOS DE LARGA DURACIÓN (% PIB)", [(6,"Histórico",2015),(7,"AIReF",2025),(8,"Opinión 2025",2025)]),
    25: (33, "Educación", "Evolución del gasto en educación (% PIB)",
         "GRÁFICO 25. EVOLUCIÓN DEL GASTO EN EDUCACIÓN (% PIB)", [(6,"Histórico",2015),(7,"AIReF",2025),(8,"Opinión 2025",2023)]),
    26: (34, "Desempleo", "Evolución del gasto en prestaciones por desempleo (% PIB)",
         "GRÁFICO 26. EVOLUCIÓN DEL GASTO EN PRESTACIONES POR DESEMPLEO (% PIB)", [(6,"Histórico",2015),(7,"AIReF",2025),(8,"Opinión 2025",2023)]),
}
TABLE_LABELS = ["Recursos no Financieros", "Modelizados", "IVA e impuestos s/ productos", "IRPF",
                "Imp. sobre Sociedades", "Cotizaciones sociales", "No modelizados"]
FIELDS = "book row_number record_id block metric series kind year start_year end_year unit value source_sheet source_cell source_formula source_object source_series_id family".split()


def decimal(text):
    require(text is not None, "Missing decimal")
    value = Decimal(text)
    require(value.is_finite(), "Non-finite decimal")
    with localcontext() as ctx:
        ctx.prec = 60
        require(abs(value) < Decimal(10)**14 and value == value.quantize(Decimal('1e-24')), "Numeric precision outside schema")
    return value


class Original:
    def __init__(self, book):
        path, expected, _ = BOOKS[book]
        require(sha(ROOT/path) == expected, "Original workbook changed")
        self.xml = ExcelXML(ROOT/path)
        self.workbook = openpyxl.load_workbook(ROOT/path, read_only=True, data_only=False)
        self.cached = {}

    def sheet(self, name):
        if name not in self.cached:
            nodes = {n.get('r'): n for n in ET.fromstring(self.xml.parts[self.xml.sheets[name]]).findall('m:sheetData/m:row/m:c', M)}
            formulas = {c.coordinate: c.value for row in self.workbook[name].iter_rows() for c in row if c.value is not None}
            self.cached[name] = (nodes, formulas)
        return self.cached[name]

    def number(self, sheet, address):
        node = self.sheet(sheet)[0].get(address)
        if node is None or node.find('m:v', M) is None:
            return None
        require(node.get('t','n') == 'n', f"Non-numeric original {sheet}/{address}")
        return decimal(node.find('m:v', M).text)

    def text(self, sheet, address):
        return self.sheet(sheet)[1].get(address)


def staged(con, book, sheet):
    cursor = con.execute("SELECT * FROM staging_records WHERE book=? AND bloque=? ORDER BY row_number", [book, sheet])
    names = [c[0] for c in cursor.description]
    return [dict(zip(names, row)) for row in cursor.fetchall()]


def record(row, metric, series, kind, year=None, start=None, end=None, family=None):
    return (row['book'], row['row_number'], row['id'], row['bloque'], metric, series, kind, year, start, end,
            'percent_gdp' if kind=='level' else 'percentage_points_gdp', decimal(row['decimal_XML_texto']),
            row['hoja_valor'], row['celda_valor'], row['formula_texto'], row['parte_grafico_imagen'], row['serie_id'], family)


def graph_records(con, original, number):
    chart, metric, title, original_title, profiles = GRAPHS[number]
    sheet = f'Gráfico {number}'
    require(original.text(sheet,'B3') == original_title, "Graph title/unit changed")
    part = f'xl/charts/chart{chart}.xml'
    series_nodes = ET.fromstring(original.xml.parts[part]).findall('.//c:ser', C)
    require(len(series_nodes)==3, "Chart series count changed")
    rows = staged(con,'2026',sheet)
    require(len(rows)==108, "Graph position inventory changed")
    indexed = {(r['serie_id'],r['periodo_categoria_literal']):r for r in rows}
    require(len(indexed)==108, "Duplicate graph position")
    output = []
    for index, (source_row,label,first) in enumerate(profiles,1):
        node = series_nodes[index-1]
        require(original.text(sheet,f'B{source_row}')==label, "Original series label changed")
        name_ref = node.find('c:tx/c:strRef/c:f', C)
        name_literal = node.find('c:tx/c:v', C)
        if name_ref is not None:
            require(name_ref.text==f"'{sheet}'!$B${source_row}", "Series name binding mismatch")
        else:
            require(name_literal is not None and name_literal.text==label, "Literal series name mismatch")
        cat = node.find('c:cat/c:numRef/c:f',C)
        val = node.find('c:val/c:numRef/c:f',C)
        require(cat is not None and cat.text==f"'{sheet}'!$C$5:$AL$5", "Year axis binding mismatch")
        require(val is not None and val.text==f"'{sheet}'!$C${source_row}:$AL${source_row}", "Value axis binding mismatch")
        for year in range(2015,2051):
            col = column(year-2015+3)
            require(original.number(sheet,f'{col}5')==year, "Original year axis mismatch")
            row = indexed[(f'chart{chart}_s{index}',str(year))]
            require(row['hoja_valor']==sheet and row['celda_valor']==f'{col}{source_row}', "Value source identity mismatch")
            require(row['serie_literal']==label and row['titulo']==title and row['metrica_literal']==title, "Graph semantic identity mismatch")
            require(row['parte_grafico_imagen']==part and row['tipo']=='posicion_serie', "Graph type/object mismatch")
            require(row['contexto_unidad_titulo']=='% PIB' and row['unidad']=='ver_contexto_literal', "Graph unit mismatch")
            value = original.number(sheet,row['celda_valor'])
            supported = first <= year <= (2025 if label=='Histórico' else 2050)
            require((value is not None)==supported, "Series support changed")
            if not supported:
                require(row['decimal_XML_texto'] is None and row['estado_dato']=='ausente', "Absent position contains value")
                continue
            require(row['estado_dato']=='dato' and decimal(row['decimal_XML_texto'])==value, "Graph exact numeric mismatch")
            require(row['formula_texto'] is None and not str(original.text(sheet,row['celda_valor'])).startswith('='), "Unexpected graph formula")
            output.append(record(row,metric,label,'level',year=year,family='annual_fiscal_gdp'))
    return output


def table_records(con, original):
    sheet='Cuadro 7 '
    require(original.text(sheet,'C2')=='RECURSOS NO FINANCIEROS SIN PRTR EN % SOBRE EL PIB', "Table title/unit changed")
    rows=staged(con,'2025',sheet)
    require(len(rows)==56, "Table inventory changed")
    indexed={r['celda_valor']:r for r in rows}
    require(len(indexed)==56, "Duplicate table source cell")
    output=[]
    years=dict(zip('EFGHI',[2023,2029,2041,2050,2070]))
    changes={'K':('E','F'),'L':('F','H'),'M':('H','I')}
    for col,year in years.items():
        require(original.number(sheet,f'{col}5')==year, "Table year changed")
    for col,(a,b) in changes.items():
        require(''.join(original.text(sheet,f'{col}5').split())==f'Δ{years[b]}-{years[a]}', "Change header direction mismatch")
    for source_row,metric in enumerate(TABLE_LABELS,6):
        require(original.text(sheet,f'C{source_row}')==metric, "Table row label changed")
        for col in [*years,*changes]:
            address=f'{col}{source_row}'
            row=indexed[address]
            require(row['hoja_valor']==sheet and row['metrica_literal']==metric and row['serie_literal']=='Opinion 2025', "Table semantic identity mismatch")
            require(row['estado_dato']=='dato_numerico', "Table state mismatch")
            value=original.number(sheet,address)
            require(decimal(row['decimal_XML_texto'])==value, "Table exact value mismatch")
            formula=original.text(sheet,address)
            formula=formula if isinstance(formula,str) and formula.startswith('=') else None
            require(row['formula_texto']==formula, "Table formula provenance mismatch")
            if col in years:
                require(row['periodo_categoria_literal']==str(years[col]) and row['unidad']=='% PIB' and row['tipo']=='nivel', "Table level dimensions mismatch")
                if source_row==7:
                    require(formula==f'=SUM({col}8:{col}11)', "Subtotal formula changed")
                    expected=sum((original.number(sheet,f'{col}{r}') for r in range(8,12)),Decimal(0))
                elif source_row==12:
                    require(formula==f'={col}6-{col}7', "Remainder formula changed")
                    expected=original.number(sheet,f'{col}6')-original.number(sheet,f'{col}7')
                else:
                    require(formula is None, "Unexpected level formula")
                    expected=value
                output.append(record(row,metric,'Opinion 2025','level',year=years[col],family='fiscal_levels_and_changes'))
            else:
                a,b=changes[col]
                expected=original.number(sheet,f'{b}{source_row}')-original.number(sheet,f'{a}{source_row}')
                require(formula==f'={b}{source_row}-{a}{source_row}', "Change formula direction mismatch")
                require(''.join(row['periodo_categoria_literal'].split())==f'Δ{years[b]}-{years[a]}', "Change period mismatch")
                require(row['unidad']=='puntos PIB' and row['tipo']=='variacion_publicada', "Change unit/kind mismatch")
                output.append(record(row,metric,'Opinion 2025','published_change',start=years[a],end=years[b],family='fiscal_levels_and_changes'))
            require(abs(value-expected)<=Decimal('1e-10'), "Formula/cache mismatch")
    return output


def reviewed_records(con):
    old=con.execute("""SELECT a.book,a.row_number,a.record_id,b.sheet,o.metric,o.series,o.kind,o.year,o.start_year,o.end_year,
         o.unit,o.value,b.sheet,o.source_cell,CASE WHEN o.formula IS NULL THEN NULL ELSE '='||o.formula END,
         s.parte_grafico_imagen,s.serie_id,'pilot' FROM observations o JOIN blocks b USING(block)
         JOIN admissions a USING(block,source_cell) JOIN staging_records s ON s.book=a.book AND s.row_number=a.row_number""").fetchall()
    require(len(old)==182, "Initial admissions changed")
    originals={book:Original(book) for book in BOOKS}
    try:
        added=table_records(con,originals['2025'])
        for number in GRAPHS:
            added.extend(graph_records(con,originals['2026'],number))
    finally:
        for original in originals.values(): original.workbook.close()
    require(len(added)==378, "Reviewed family counts changed")
    return old+added


DDL="""CREATE TABLE family_observations(book VARCHAR,row_number INTEGER,record_id VARCHAR,block VARCHAR,metric VARCHAR,
 series VARCHAR,kind VARCHAR,year INTEGER,start_year INTEGER,end_year INTEGER,unit VARCHAR,value DECIMAL(38,24),
 source_sheet VARCHAR,source_cell VARCHAR,source_formula VARCHAR,source_object VARCHAR,source_series_id VARCHAR,family VARCHAR,
 PRIMARY KEY(book,row_number),UNIQUE(book,record_id));
 CREATE VIEW family_validated_observations AS SELECT * FROM family_observations;
 CREATE VIEW family_review_queue AS SELECT s.* FROM staging_records s WHERE NOT EXISTS
 (SELECT 1 FROM family_observations f WHERE f.book=s.book AND f.row_number=s.row_number);
"""


def build(output):
    output=Path(output).resolve(); output.relative_to(ROOT)
    require(not output.exists(), "Output exists; no overwrite")
    require(sha(BASE)==BASE_SHA, "Base database changed")
    validate_base(BASE)
    output.mkdir(parents=True)
    db=output/'families.duckdb'
    shutil.copyfile(BASE,db)
    with duckdb.connect(str(db)) as con:
        records=reviewed_records(con)
        con.execute('BEGIN')
        con.execute(DDL)
        con.executemany('INSERT INTO family_observations VALUES ('+','.join('?' for _ in FIELDS)+')',records)
        con.execute('CREATE TABLE family_sources(book VARCHAR PRIMARY KEY,xlsx_sha256 VARCHAR,xlsx_bytes BLOB,pdf_sha256 VARCHAR)')
        for book,(name,digest,pdfname) in BOOKS.items():
            require(sha(ROOT/name)==digest and sha((ROOT/name).parent/pdfname)==PDF_HASHES[book], "Original source changed")
            con.execute('INSERT INTO family_sources VALUES (?,?,?,?)',[book,digest,(ROOT/name).read_bytes(),PDF_HASHES[book]])
        con.execute('COMMIT')
    manifest=dict(status='built_pending_validation',base_sha256=BASE_SHA,code_sha256=sha(Path(__file__)),
                  db_sha256=sha(db),admitted=560,added=378,rag_integrated=False)
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    return manifest


def validate(db):
    base=validate_base(db)
    with duckdb.connect(str(db),read_only=True) as con:
        expected=reviewed_records(con)
        require(set(con.execute('SELECT * FROM family_observations').fetchall())==set(expected), "Family semantic content mismatch")
        require(con.execute('SELECT count(*) FROM family_observations').fetchone()[0]==len(expected)==560, "Family count mismatch")
        require([c[0] for c in con.execute('SELECT * FROM family_validated_observations LIMIT 0').description]==FIELDS, "Family view columns mismatch")
        require(set(con.execute('SELECT * FROM family_validated_observations').fetchall())==set(expected), "Family view content mismatch")
        require(con.execute('SELECT count(*) FROM family_validated_observations').fetchone()[0]==len(expected), "Family view multiplicity mismatch")
        # Pending records must be precisely the complement, including duplicates.
        queue="SELECT s.* FROM staging_records s WHERE NOT EXISTS(SELECT 1 FROM family_observations f WHERE f.book=s.book AND f.row_number=s.row_number)"
        require([c[0] for c in con.execute('SELECT * FROM family_review_queue LIMIT 0').description]==[c[0] for c in con.execute(queue+' LIMIT 0').description], "Queue columns mismatch")
        require(not con.execute('(SELECT * FROM family_review_queue EXCEPT ALL ('+queue+')) UNION ALL (('+queue+') EXCEPT ALL SELECT * FROM family_review_queue) LIMIT 1').fetchone(), "Family pending coverage mismatch")
        sources=con.execute('SELECT * FROM family_sources').fetchall()
        require({r[0] for r in sources}==set(BOOKS), "Family source inventory mismatch")
        for book,digest,content,pdfdigest in sources:
            require(digest==BOOKS[book][1] and hashlib.sha256(content).hexdigest()==digest and pdfdigest==PDF_HASHES[book], "Family source archive mismatch")
        counts=con.execute('SELECT book,block,family,count(*) FROM family_observations GROUP BY ALL ORDER BY book,block').fetchall()
        pending=con.execute('SELECT count(*) FROM family_review_queue').fetchone()[0]
        require(pending==64089, "Pending count mismatch")
    return dict(status='pass',admitted=560,added=378,pending=pending,counts=counts,
                base_archived_rows_checked=base['archived_rows_checked'],db_sha256=sha(Path(db)),code_sha256=sha(Path(__file__)),
                rag_integrated=False,notes=['Literal AIReF labels retained; no vintage inferred',
                 'Graph 22 deferred: repeated names and unequal axis lengths', 'PDF physical citations still pending'])


def query(db,*,book,block,metric,series,year=None,start_year=None,end_year=None,kind='level'):
    require(isinstance(book,str) and book in BOOKS and all(isinstance(t,str) and t.strip() for t in [block,metric,series]), "Explicit identity required")
    require(kind in ['level','published_change'], "Unsupported observation kind")
    if kind=='level':
        require(type(year) is int and start_year is None and end_year is None, "Level requires integer year only")
    else:
        require(year is None and type(start_year) is int and type(end_year) is int and start_year<end_year, "Change requires ordered integer years")
    with duckdb.connect(str(db),read_only=True) as con:
        cursor=con.execute('''SELECT f.*,s.xlsx_path,s.xlsx_sha256,s.pdf_path,s.pdf_sha256,w.path AS universal_path,w.sha256 AS universal_sha256
          FROM family_validated_observations f JOIN sources s ON s.book=f.book JOIN workbook_files w ON w.book=f.book
          WHERE f.book=? AND block=? AND metric=? AND series=? AND kind=? AND year IS NOT DISTINCT FROM ?
          AND start_year IS NOT DISTINCT FROM ? AND end_year IS NOT DISTINCT FROM ?''',[book,block,metric,series,kind,year,start_year,end_year])
        names=[c[0] for c in cursor.description]; rows=cursor.fetchall()
    if len(rows)!=1: raise LookupError('No unique admitted observation; no fallback')
    result=dict(zip(names,rows[0]))
    result['value']=str(result['value'])
    result['display_value']=str(Decimal(result['value']).quantize(Decimal('.1'),rounding=ROUND_HALF_UP))
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    p=commands.add_parser('build'); p.add_argument('--output-dir',type=Path,required=True)
    p=commands.add_parser('validate'); p.add_argument('--db',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    if args.command=='build': report=build(args.output_dir)
    else:
        require(not args.output.exists(),'Validation output exists')
        try: report=validate(args.db)
        except Exception as exc: report=dict(status='fail',error=str(exc))
        args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))
    return int(report['status']=='fail')


if __name__=='__main__': raise SystemExit(main())
