"""Independent original-cell checks and adversarial family admission tests."""
import importlib.util
from decimal import Decimal
from pathlib import Path
import re
import shutil
import tempfile
import unittest
from unittest.mock import patch

if not all(importlib.util.find_spec(n) for n in ('duckdb','openpyxl')):
    raise unittest.SkipTest('Optional Excel dependencies required')

import duckdb
import openpyxl
from experiments.semantic_families import ROOT, BOOKS, Original, graph_records, table_records, query, validate, build

DB=ROOT/'evaluation/semantic_families_2026-10-31/run_v2/families.duckdb'


class TestSemanticFamilies(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not DB.exists(): raise unittest.SkipTest('Build family integration fixture first')
        cls.temp=tempfile.TemporaryDirectory(dir=ROOT/'tmp')

    @classmethod
    def tearDownClass(cls): cls.temp.cleanup()

    def test_all_new_observations_against_independent_original_reader(self):
        with duckdb.connect(str(DB),read_only=True) as con:
            rows=con.execute("SELECT book,block,metric,series,kind,year,start_year,end_year,unit,value,source_cell FROM family_observations WHERE family<>'pilot'").fetchall()
        self.assertEqual(len(rows),378)
        workbooks={book:openpyxl.load_workbook(ROOT/name,read_only=True,data_only=True) for book,(name,_,_) in BOOKS.items()}
        sheets={}
        for book,block,*_ in rows:
            if (book,block) not in sheets:
                sheets[book,block]={c.coordinate:c.value for r in workbooks[book][block].iter_rows() for c in r if c.value is not None}
        for book,block,metric,series,kind,year,start,end,unit,value,address in rows:
            source=sheets[book,block]
            self.assertLessEqual(abs(value-Decimal(str(source[address]))),Decimal('1e-11'))
            col,num=re.fullmatch(r'([A-Z]+)(\d+)',address).groups()
            if book=='2026':
                self.assertEqual(series,source[f'B{num}'])
                self.assertEqual(year,source[f'{col}5'])
                self.assertEqual((kind,unit),('level','percent_gdp'))
            else:
                self.assertEqual(metric,source[f'C{num}'])
                self.assertEqual(series,'Opinion 2025')
                if kind=='level': self.assertEqual(year,source[f'{col}5'])
                else:
                    label=''.join(source[f'{col}5'].split())
                    self.assertEqual(label,f'Δ{end}-{start}')
                    self.assertEqual(unit,'percentage_points_gdp')
        for workbook in workbooks.values(): workbook.close()

    def q(self,**kwargs):
        args=dict(book='2026',block='Gráfico 19',metric='Recursos',series='AIReF 2026',year=2030)
        args.update(kwargs)
        return query(DB,**args)

    def test_graph19_series_order_not_position(self):
        current=self.q(); previous=self.q(series='Opinión 2025')
        self.assertEqual(current['source_cell'],'R8')
        self.assertEqual(previous['source_cell'],'R7')
        self.assertEqual(current['display_value'],'43.3')
        self.assertEqual(previous['display_value'],'42.2')

    def test_literal_airef_not_assigned_vintage(self):
        result=self.q(block='Gráfico 24',metric='Cuidados de larga duración',series='AIReF',year=2050)
        self.assertEqual(result['display_value'],'1.6')
        with self.assertRaises(LookupError):
            self.q(block='Gráfico 24',metric='Cuidados de larga duración',series='AIReF 2026',year=2050)

    def test_absent_ambiguous_and_foreign_scope_rejected(self):
        for args in [dict(series='Histórico'),dict(year=2021,series='Opinión 2025'),dict(metric='Recursos Cataluña'),
                     dict(block='Gráfico 22',metric='Intereses',series='Intereses')]:
            with self.subTest(args=args),self.assertRaises(LookupError): self.q(**args)
        with self.assertRaises(ValueError): self.q(year=True)

    def test_table_level_and_published_change(self):
        result=query(DB,book='2025',block='Cuadro 7 ',metric='IRPF',series='Opinion 2025',year=2050)
        self.assertEqual(result['display_value'],'11.2')
        changed=query(DB,book='2025',block='Cuadro 7 ',metric='IRPF',series='Opinion 2025',kind='published_change',start_year=2029,end_year=2050)
        self.assertEqual(changed['source_formula'],'=H9-F9')
        self.assertEqual(changed['unit'],'percentage_points_gdp')

    def test_partition_and_legacy_preservation(self):
        with duckdb.connect(str(DB),read_only=True) as con:
            self.assertEqual(con.execute('SELECT count(*) FROM family_review_queue').fetchone()[0],64089)
            self.assertEqual(con.execute('SELECT count(*) FROM family_observations').fetchone()[0],560)
            self.assertEqual(con.execute('SELECT count(*) FROM validated_observations').fetchone()[0],182)

    def test_no_overwrite(self):
        with self.assertRaisesRegex(ValueError,'Output exists'): build(DB.parent)

    def test_source_semantic_mutations_rejected(self):
        original=Original('2026')
        original_text=original.text
        def false_label(sheet,address):
            return 'Opinión 2025' if (sheet,address)==('Gráfico 19','B8') else original_text(sheet,address)
        try:
            with duckdb.connect(str(DB),read_only=True) as con:
                with patch.object(original,'text',side_effect=false_label),self.assertRaisesRegex(ValueError,'series label'):
                    graph_records(con,original,19)
        finally: original.workbook.close()

    def test_duplicate_view_rejected(self):
        path=Path(self.temp.name)/'duplicated.duckdb'; shutil.copyfile(DB,path)
        with duckdb.connect(str(path)) as con:
            con.execute('CREATE OR REPLACE VIEW family_validated_observations AS SELECT * FROM family_observations UNION ALL SELECT * FROM family_observations')
        with self.assertRaisesRegex(ValueError,'multiplicity'): validate(path)

    def test_small_numeric_tampering_rejected(self):
        path=Path(self.temp.name)/'tiny_change.duckdb'; shutil.copyfile(DB,path)
        with duckdb.connect(str(path)) as con:
            con.execute("UPDATE family_observations SET value=value+0.000000000001 WHERE book='2026' AND block='Gráfico 19' AND source_cell='R8'")
        with self.assertRaisesRegex(ValueError,'semantic content'): validate(path)


if __name__=='__main__': unittest.main()
