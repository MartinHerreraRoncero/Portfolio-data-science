"""Exhaustive source review and safe public view; no RAG integration."""
import argparse,json,re,shutil,hashlib,xml.etree.ElementTree as ET
from pathlib import Path
from decimal import Decimal,ROUND_HALF_UP
import duckdb
from experiments.reviewed_family_semantics import ReviewedResolver,FIELDS,validate as validate_previous
from experiments.source_semantics import SourceResolver
from experiments.bulk_semantics import C,ref_cells
from experiments.indexed_semantics import points
from experiments.semantic_families import M
from experiments.excel_pilot import ROOT,sha,PDF_HASHES
from experiments.universal_database import require
PREVIOUS=ROOT/'evaluation/reviewed_family_semantics_2026-10-31/run_v2/reviewed_families.duckdb'
PREVIOUS_SHA='58fe6c68037b75dc6de0b34b321e9db56299bafea972399323e6c476c95ce0f4'
INVENTORY=ROOT/'evaluation/closure_family_semantics_2026-10-31/research_bindings/closure_inventory4849.json'
INVENTORY_SHA='f0f8a0319be301ba911fcc349b0bc22ae00b0694ad5e2598049d3f4da502a57e'
PROFILES_LIST=[{'book': '2025',
  'block': 'Cuadro 14',
  'sid': None,
  'headers': [('B2', 'CUADRO 14'),
              ('C2', 'TIPOS DE INTERÉS DE LAS NUEVAS EMISIONES (%), DIFERENCIAS S/ EL ESCENARIO BASE'),
              ('D5', 'Tipos de interés de las nuevas emisiones (%)'),
              ('I5', 'Tipos de interés de las nuevas emisiones (%)'),
              ('C6', 'Escenario'),
              ('D6', 'Promedio 2000-2024'),
              ('E6', 'Promedio 2025-2030'),
              ('F6', 'Promedio 2031-2040'),
              ('G6', 'Promedio 2041-2050'),
              ('I6', 'Promedio 2000-2024'),
              ('J6', 'Promedio 2025-2030'),
              ('C7', 'Base'),
              ('B8', 'Diferencias sobre el escenario base'),
              ('C8', 'Emisiones netas cero (Net0)'),
              ('D8', '..'),
              ('C9', 'Net0 + transformación económica'),
              ('D9', '..'),
              ('C10', 'Transición retardada'),
              ('D10', '..'),
              ('C11', 'Baja demanda'),
              ('D11', '..'),
              ('C12', 'Catástrofe climática'),
              ('D12', '..'),
              ('C13', 'Dificultades climáticas'),
              ('D13', '..')],
  'keys': [(713, 'F8', 'Cuadro 14!F8'),
           (728, 'E10', 'Cuadro 14!E10'),
           (738, 'G11', 'Cuadro 14!G11'),
           (745, 'F12', 'Cuadro 14!F12'),
           (704, 'E7', 'Cuadro 14!E7'),
           (706, 'G7', 'Cuadro 14!G7'),
           (720, 'E9', 'Cuadro 14!E9'),
           (714, 'G8', 'Cuadro 14!G8'),
           (721, 'F9', 'Cuadro 14!F9'),
           (744, 'E12', 'Cuadro 14!E12'),
           (746, 'G12', 'Cuadro 14!G12'),
           (730, 'G10', 'Cuadro 14!G10'),
           (753, 'F13', 'Cuadro 14!F13'),
           (752, 'E13', 'Cuadro 14!E13'),
           (712, 'E8', 'Cuadro 14!E8'),
           (737, 'F11', 'Cuadro 14!F11'),
           (705, 'F7', 'Cuadro 14!F7'),
           (722, 'G9', 'Cuadro 14!G9'),
           (703, 'D7', 'Cuadro 14!D7'),
           (729, 'F10', 'Cuadro 14!F10'),
           (754, 'G13', 'Cuadro 14!G13'),
           (736, 'E11', 'Cuadro 14!E11')],
  'mode': 'published_table',
  'unit': 'percent_gdp',
  'multiplier': '1',
  'pdf': [154, ['TIPOS DE INTERÉS', 'DIFERENCIAS']],
  'scope': 'published_literal_not_reconciled_to_auxiliary_levels'},
 {'book': '2025',
  'block': 'Gráfico 9B',
  'sid': 'chart18_s1',
  'headers': [('G4', 'FLUJOS MIGRATORIOS EXTERIORES OBSERVADOS (MILES DE PERSONAS)'), ('G5', 'FLUJO NETO DE MIGRACIÓN EXTERIOR')],
  'keys': [(6247, 'C12', 'chart18_s1_p7'),
           (6243, 'C8', 'chart18_s1_p3'),
           (6254, 'C19', 'chart18_s1_p14'),
           (6244, 'C9', 'chart18_s1_p4'),
           (6251, 'C16', 'chart18_s1_p11'),
           (6257, 'C22', 'chart18_s1_p17'),
           (6249, 'C14', 'chart18_s1_p9'),
           (6256, 'C21', 'chart18_s1_p16'),
           (6241, 'C6', 'chart18_s1_p1'),
           (6248, 'C13', 'chart18_s1_p8'),
           (6252, 'C17', 'chart18_s1_p12'),
           (6255, 'C20', 'chart18_s1_p15'),
           (6246, 'C11', 'chart18_s1_p6'),
           (6250, 'C15', 'chart18_s1_p10'),
           (6253, 'C18', 'chart18_s1_p13'),
           (6259, 'C24', 'chart18_s1_p19'),
           (6242, 'C7', 'chart18_s1_p2'),
           (6245, 'C10', 'chart18_s1_p5'),
           (6258, 'C23', 'chart18_s1_p18')],
  'mode': 'normal',
  'unit': 'thousand_persons_net_external_migration',
  'multiplier': '1',
  'pdf': [34, ['FLUJO NETO DE MIGRACIÓN EXTERIOR']],
  'label_cell': 'G5',
  'role': 'published_net_migration_flow_not_gross',
  'scope': 'before_2021_derived_from_population_changes_not_directly_observed',
  'part': 'xl/charts/chart18.xml',
  'refs': ["'Gráfico 9B'!$C$4", "'Gráfico 9B'!$B$5:$B$26", "'Gráfico 9B'!$C$5:$C$26"],
  'node_xml': '<ns0:ser xmlns:ns0="http://schemas.openxmlformats.org/drawingml/2006/chart" xmlns:ns1="http://schemas.openxmlformats.org/drawingml/2006/main" '
              'xmlns:ns2="http://schemas.microsoft.com/office/drawing/2014/chart"><ns0:idx val="0" /><ns0:order val="0" /><ns0:tx><ns0:strRef><ns0:f>\'Gráfico '
              '9B\'!$C$4</ns0:f><ns0:strCache><ns0:ptCount val="1" /></ns0:strCache></ns0:strRef></ns0:tx><ns0:spPr><ns1:ln w="28575" '
              'cap="rnd"><ns1:solidFill><ns1:schemeClr val="accent6" /></ns1:solidFill><ns1:prstDash val="sysDot" /><ns1:round /></ns1:ln><ns1:effectLst '
              '/></ns0:spPr><ns0:marker><ns0:symbol val="none" /></ns0:marker><ns0:cat><ns0:numRef><ns0:f>\'Gráfico '
              '9B\'!$B$5:$B$26</ns0:f><ns0:numCache><ns0:formatCode>General</ns0:formatCode><ns0:ptCount val="22" /><ns0:pt '
              'idx="0"><ns0:v>2002</ns0:v></ns0:pt><ns0:pt idx="1"><ns0:v>2003</ns0:v></ns0:pt><ns0:pt idx="2"><ns0:v>2004</ns0:v></ns0:pt><ns0:pt '
              'idx="3"><ns0:v>2005</ns0:v></ns0:pt><ns0:pt idx="4"><ns0:v>2006</ns0:v></ns0:pt><ns0:pt idx="5"><ns0:v>2007</ns0:v></ns0:pt><ns0:pt '
              'idx="6"><ns0:v>2008</ns0:v></ns0:pt><ns0:pt idx="7"><ns0:v>2009</ns0:v></ns0:pt><ns0:pt idx="8"><ns0:v>2010</ns0:v></ns0:pt><ns0:pt '
              'idx="9"><ns0:v>2011</ns0:v></ns0:pt><ns0:pt idx="10"><ns0:v>2012</ns0:v></ns0:pt><ns0:pt idx="11"><ns0:v>2013</ns0:v></ns0:pt><ns0:pt '
              'idx="12"><ns0:v>2014</ns0:v></ns0:pt><ns0:pt idx="13"><ns0:v>2015</ns0:v></ns0:pt><ns0:pt idx="14"><ns0:v>2016</ns0:v></ns0:pt><ns0:pt '
              'idx="15"><ns0:v>2017</ns0:v></ns0:pt><ns0:pt idx="16"><ns0:v>2018</ns0:v></ns0:pt><ns0:pt idx="17"><ns0:v>2019</ns0:v></ns0:pt><ns0:pt '
              'idx="18"><ns0:v>2020</ns0:v></ns0:pt><ns0:pt idx="19"><ns0:v>2021</ns0:v></ns0:pt><ns0:pt idx="20"><ns0:v>2022</ns0:v></ns0:pt><ns0:pt '
              'idx="21"><ns0:v>2023</ns0:v></ns0:pt></ns0:numCache></ns0:numRef></ns0:cat><ns0:val><ns0:numRef><ns0:f>\'Gráfico '
              '9B\'!$C$5:$C$26</ns0:f><ns0:numCache><ns0:formatCode>0.0</ns0:formatCode><ns0:ptCount val="22" /><ns0:pt '
              'idx="1"><ns0:v>662.37499999999488</ns0:v></ns0:pt><ns0:pt idx="2"><ns0:v>664.81600000000117</ns0:v></ns0:pt><ns0:pt '
              'idx="3"><ns0:v>635.67999999999813</ns0:v></ns0:pt><ns0:pt idx="4"><ns0:v>663.36200000000235</ns0:v></ns0:pt><ns0:pt '
              'idx="5"><ns0:v>777.59400000000232</ns0:v></ns0:pt><ns0:pt idx="6"><ns0:v>434.85599999999874</ns0:v></ns0:pt><ns0:pt '
              'idx="7"><ns0:v>138.56599999999852</ns0:v></ns0:pt><ns0:pt idx="8"><ns0:v>77.349000000003727</ns0:v></ns0:pt><ns0:pt '
              'idx="9"><ns0:v>68.398999999997415</ns0:v></ns0:pt><ns0:pt idx="10"><ns0:v>-155.96399999999886</ns0:v></ns0:pt><ns0:pt '
              'idx="11"><ns0:v>-252.2020000000027</ns0:v></ns0:pt><ns0:pt idx="12"><ns0:v>-101.78699999999721</ns0:v></ns0:pt><ns0:pt '
              'idx="13"><ns0:v>-4.5600000000033845</ns0:v></ns0:pt><ns0:pt idx="14"><ns0:v>78.536999999998216</ns0:v></ns0:pt><ns0:pt '
              'idx="15"><ns0:v>179.01900000000333</ns0:v></ns0:pt><ns0:pt idx="16"><ns0:v>328.82500000000118</ns0:v></ns0:pt><ns0:pt '
              'idx="17"><ns0:v>457.18500000000199</ns0:v></ns0:pt><ns0:pt idx="18"><ns0:v>235.20899999999958</ns0:v></ns0:pt><ns0:pt '
              'idx="19"><ns0:v>191.09399999999999</ns0:v></ns0:pt></ns0:numCache></ns0:numRef></ns0:val><ns0:smooth val="0" /><ns0:extLst><ns0:ext '
              'uri="{C3380CC4-5D6E-409C-BE32-E72D297353CC}"><ns2:uniqueId val="{00000000-45B6-4529-9309-106A5E38B992}" /></ns0:ext></ns0:extLst></ns0:ser>',
  'root_axes': ['<ns0:valAx xmlns:ns0="http://schemas.openxmlformats.org/drawingml/2006/chart" '
                'xmlns:ns1="http://schemas.openxmlformats.org/drawingml/2006/main"><ns0:axId val="282032559" /><ns0:scaling><ns0:orientation val="minMax" '
                '/><ns0:min val="-300" /></ns0:scaling><ns0:delete val="0" /><ns0:axPos val="l" /><ns0:numFmt formatCode="#,##0" sourceLinked="0" '
                '/><ns0:majorTickMark val="none" /><ns0:minorTickMark val="none" /><ns0:tickLblPos val="nextTo" /><ns0:spPr><ns1:noFill '
                '/><ns1:ln><ns1:solidFill><ns1:sysClr val="windowText" lastClr="000000" /></ns1:solidFill></ns1:ln><ns1:effectLst '
                '/></ns0:spPr><ns0:txPr><ns1:bodyPr rot="-60000000" spcFirstLastPara="1" vertOverflow="ellipsis" vert="horz" wrap="square" anchor="ctr" '
                'anchorCtr="1" /><ns1:lstStyle /><ns1:p><ns1:pPr><ns1:defRPr sz="900" b="0" i="0" u="none" strike="noStrike" kern="1200" '
                'baseline="0"><ns1:solidFill><ns1:sysClr val="windowText" lastClr="000000" /></ns1:solidFill><ns1:latin typeface="Gill Sans MT" '
                'panose="020B0502020104020203" pitchFamily="34" charset="0" /><ns1:ea typeface="+mn-ea" /><ns1:cs typeface="+mn-cs" '
                '/></ns1:defRPr></ns1:pPr><ns1:endParaRPr lang="es-ES" /></ns1:p></ns0:txPr><ns0:crossAx val="281938127" /><ns0:crosses val="autoZero" '
                '/><ns0:crossBetween val="between" /><ns0:majorUnit val="150" /></ns0:valAx>'],
  'value_ref': "'Gráfico 9B'!$C$5:$C$26",
  'category_ref': "'Gráfico 9B'!$B$5:$B$26",
  'name_ref': "'Gráfico 9B'!$C$4",
  'physical_value_cells': ['C5',
                           'C6',
                           'C7',
                           'C8',
                           'C9',
                           'C10',
                           'C11',
                           'C12',
                           'C13',
                           'C14',
                           'C15',
                           'C16',
                           'C17',
                           'C18',
                           'C19',
                           'C20',
                           'C21',
                           'C22',
                           'C23',
                           'C24',
                           'C25',
                           'C26'],
  'physical_category_cells': ['B5',
                              'B6',
                              'B7',
                              'B8',
                              'B9',
                              'B10',
                              'B11',
                              'B12',
                              'B13',
                              'B14',
                              'B15',
                              'B16',
                              'B17',
                              'B18',
                              'B19',
                              'B20',
                              'B21',
                              'B22',
                              'B23',
                              'B24',
                              'B25',
                              'B26']},
 {'book': '2025',
  'block': 'Gráfico 9B',
  'sid': 'chart18_s2',
  'headers': [('G4', 'FLUJOS MIGRATORIOS EXTERIORES OBSERVADOS (MILES DE PERSONAS)'), ('G5', 'FLUJO NETO DE MIGRACIÓN EXTERIOR')],
  'keys': [(6283, 'D26', 'chart18_s2_p21'), (6281, 'D24', 'chart18_s2_p19'), (6282, 'D25', 'chart18_s2_p20')],
  'mode': 'normal',
  'unit': 'thousand_persons_net_external_migration',
  'multiplier': '1',
  'pdf': [34, ['FLUJO NETO DE MIGRACIÓN EXTERIOR']],
  'label_cell': 'G5',
  'role': 'published_net_migration_flow_not_gross',
  'scope': 'before_2021_derived_from_population_changes_not_directly_observed',
  'part': 'xl/charts/chart18.xml',
  'refs': ["'Gráfico 9B'!$D$4", "'Gráfico 9B'!$B$5:$B$26", "'Gráfico 9B'!$D$5:$D$26"],
  'node_xml': '<ns0:ser xmlns:ns0="http://schemas.openxmlformats.org/drawingml/2006/chart" xmlns:ns1="http://schemas.openxmlformats.org/drawingml/2006/main" '
              'xmlns:ns2="http://schemas.microsoft.com/office/drawing/2014/chart"><ns0:idx val="3" /><ns0:order val="1" /><ns0:tx><ns0:strRef><ns0:f>\'Gráfico '
              '9B\'!$D$4</ns0:f><ns0:strCache><ns0:ptCount val="1" /></ns0:strCache></ns0:strRef></ns0:tx><ns0:spPr><ns1:ln w="28575" '
              'cap="rnd"><ns1:solidFill><ns1:schemeClr val="accent6" /></ns1:solidFill><ns1:prstDash val="solid" /><ns1:round /></ns1:ln><ns1:effectLst '
              '/></ns0:spPr><ns0:marker><ns0:symbol val="none" /></ns0:marker><ns0:cat><ns0:numRef><ns0:f>\'Gráfico '
              '9B\'!$B$5:$B$26</ns0:f><ns0:numCache><ns0:formatCode>General</ns0:formatCode><ns0:ptCount val="22" /><ns0:pt '
              'idx="0"><ns0:v>2002</ns0:v></ns0:pt><ns0:pt idx="1"><ns0:v>2003</ns0:v></ns0:pt><ns0:pt idx="2"><ns0:v>2004</ns0:v></ns0:pt><ns0:pt '
              'idx="3"><ns0:v>2005</ns0:v></ns0:pt><ns0:pt idx="4"><ns0:v>2006</ns0:v></ns0:pt><ns0:pt idx="5"><ns0:v>2007</ns0:v></ns0:pt><ns0:pt '
              'idx="6"><ns0:v>2008</ns0:v></ns0:pt><ns0:pt idx="7"><ns0:v>2009</ns0:v></ns0:pt><ns0:pt idx="8"><ns0:v>2010</ns0:v></ns0:pt><ns0:pt '
              'idx="9"><ns0:v>2011</ns0:v></ns0:pt><ns0:pt idx="10"><ns0:v>2012</ns0:v></ns0:pt><ns0:pt idx="11"><ns0:v>2013</ns0:v></ns0:pt><ns0:pt '
              'idx="12"><ns0:v>2014</ns0:v></ns0:pt><ns0:pt idx="13"><ns0:v>2015</ns0:v></ns0:pt><ns0:pt idx="14"><ns0:v>2016</ns0:v></ns0:pt><ns0:pt '
              'idx="15"><ns0:v>2017</ns0:v></ns0:pt><ns0:pt idx="16"><ns0:v>2018</ns0:v></ns0:pt><ns0:pt idx="17"><ns0:v>2019</ns0:v></ns0:pt><ns0:pt '
              'idx="18"><ns0:v>2020</ns0:v></ns0:pt><ns0:pt idx="19"><ns0:v>2021</ns0:v></ns0:pt><ns0:pt idx="20"><ns0:v>2022</ns0:v></ns0:pt><ns0:pt '
              'idx="21"><ns0:v>2023</ns0:v></ns0:pt></ns0:numCache></ns0:numRef></ns0:cat><ns0:val><ns0:numRef><ns0:f>\'Gráfico '
              '9B\'!$D$5:$D$26</ns0:f><ns0:numCache><ns0:formatCode>0.0</ns0:formatCode><ns0:ptCount val="22" /><ns0:pt '
              'idx="19"><ns0:v>191.09399999999999</ns0:v></ns0:pt><ns0:pt idx="20"><ns0:v>727.005</ns0:v></ns0:pt><ns0:pt '
              'idx="21"><ns0:v>642.29600000000005</ns0:v></ns0:pt></ns0:numCache></ns0:numRef></ns0:val><ns0:smooth val="0" /><ns0:extLst><ns0:ext '
              'uri="{C3380CC4-5D6E-409C-BE32-E72D297353CC}"><ns2:uniqueId val="{00000001-45B6-4529-9309-106A5E38B992}" /></ns0:ext></ns0:extLst></ns0:ser>',
  'root_axes': ['<ns0:valAx xmlns:ns0="http://schemas.openxmlformats.org/drawingml/2006/chart" '
                'xmlns:ns1="http://schemas.openxmlformats.org/drawingml/2006/main"><ns0:axId val="282032559" /><ns0:scaling><ns0:orientation val="minMax" '
                '/><ns0:min val="-300" /></ns0:scaling><ns0:delete val="0" /><ns0:axPos val="l" /><ns0:numFmt formatCode="#,##0" sourceLinked="0" '
                '/><ns0:majorTickMark val="none" /><ns0:minorTickMark val="none" /><ns0:tickLblPos val="nextTo" /><ns0:spPr><ns1:noFill '
                '/><ns1:ln><ns1:solidFill><ns1:sysClr val="windowText" lastClr="000000" /></ns1:solidFill></ns1:ln><ns1:effectLst '
                '/></ns0:spPr><ns0:txPr><ns1:bodyPr rot="-60000000" spcFirstLastPara="1" vertOverflow="ellipsis" vert="horz" wrap="square" anchor="ctr" '
                'anchorCtr="1" /><ns1:lstStyle /><ns1:p><ns1:pPr><ns1:defRPr sz="900" b="0" i="0" u="none" strike="noStrike" kern="1200" '
                'baseline="0"><ns1:solidFill><ns1:sysClr val="windowText" lastClr="000000" /></ns1:solidFill><ns1:latin typeface="Gill Sans MT" '
                'panose="020B0502020104020203" pitchFamily="34" charset="0" /><ns1:ea typeface="+mn-ea" /><ns1:cs typeface="+mn-cs" '
                '/></ns1:defRPr></ns1:pPr><ns1:endParaRPr lang="es-ES" /></ns1:p></ns0:txPr><ns0:crossAx val="281938127" /><ns0:crosses val="autoZero" '
                '/><ns0:crossBetween val="between" /><ns0:majorUnit val="150" /></ns0:valAx>'],
  'value_ref': "'Gráfico 9B'!$D$5:$D$26",
  'category_ref': "'Gráfico 9B'!$B$5:$B$26",
  'name_ref': "'Gráfico 9B'!$D$4",
  'physical_value_cells': ['D5',
                           'D6',
                           'D7',
                           'D8',
                           'D9',
                           'D10',
                           'D11',
                           'D12',
                           'D13',
                           'D14',
                           'D15',
                           'D16',
                           'D17',
                           'D18',
                           'D19',
                           'D20',
                           'D21',
                           'D22',
                           'D23',
                           'D24',
                           'D25',
                           'D26'],
  'physical_category_cells': ['B5',
                              'B6',
                              'B7',
                              'B8',
                              'B9',
                              'B10',
                              'B11',
                              'B12',
                              'B13',
                              'B14',
                              'B15',
                              'B16',
                              'B17',
                              'B18',
                              'B19',
                              'B20',
                              'B21',
                              'B22',
                              'B23',
                              'B24',
                              'B25',
                              'B26']},
 {'book': '2025',
  'block': 'Gráfico RE 2.1',
  'sid': 'chart63_s1',
  'headers': [('C4', 'Aghion y Bunel (2025)'),
              ('D4', 'Acemoglu (2024)'),
              ('E4', ' Filippucci et al (2024)'),
              ('B5', 'Median'),
              ('I5', 'ESTIMACIONES DE IMPACTO DE LA IA EN LA PTF (PP)'),
              ('B6', 'Up Chart'),
              ('B7', 'Low Chart')],
  'keys': [(16680, 'E5', 'chart63_s1_p2'), (16678, 'C5', 'chart63_s1_p0'), (16679, 'D5', 'chart63_s1_p1')],
  'mode': 'normal',
  'unit': 'percentage_points_additional_annual_us_tfp_growth_next_ten_years',
  'multiplier': '1',
  'pdf': [80, ['PTF', 'PP']],
  'label_cell': 'B5',
  'role': 'published_study_central_estimate',
  'scope': 'additional_annual_US_TFP_growth_next_ten_years_relative_no_AI',
  'part': 'xl/charts/chart63.xml',
  'refs': ["'Gráfico RE 2.1'!$C$6:$E$6", "'Gráfico RE 2.1'!$C$7:$E$7", "'Gráfico RE 2.1'!$C$4:$E$4", "'Gráfico RE 2.1'!$C$5:$E$5"],
  'node_xml': '<ns0:ser xmlns:ns0="http://schemas.openxmlformats.org/drawingml/2006/chart" xmlns:ns1="http://schemas.openxmlformats.org/drawingml/2006/main" '
              'xmlns:ns2="http://schemas.microsoft.com/office/drawing/2014/chart" xmlns:ns3="http://schemas.microsoft.com/office/drawing/2012/chart"><ns0:idx '
              'val="0" /><ns0:order val="0" /><ns0:spPr><ns1:solidFill><ns1:schemeClr val="accent6"><ns1:lumMod val="60000" /><ns1:lumOff val="40000" '
              '/></ns1:schemeClr></ns1:solidFill><ns1:ln><ns1:noFill /></ns1:ln><ns1:effectLst /></ns0:spPr><ns0:invertIfNegative val="0" '
              '/><ns0:dLbls><ns0:dLbl><ns0:idx val="0" /><ns0:layout><ns0:manualLayout><ns0:x val="4.6418128654970733E-2" /><ns0:y val="0" '
              '/></ns0:manualLayout></ns0:layout><ns0:showLegendKey val="0" /><ns0:showVal val="1" /><ns0:showCatName val="0" /><ns0:showSerName val="0" '
              '/><ns0:showPercent val="0" /><ns0:showBubbleSize val="0" /><ns0:extLst><ns0:ext uri="{CE6537A1-D6FC-4f65-9D91-7224C49458BB}" /><ns0:ext '
              'uri="{C3380CC4-5D6E-409C-BE32-E72D297353CC}"><ns2:uniqueId val="{00000000-B027-4F48-B518-4DF87DAB4289}" '
              '/></ns0:ext></ns0:extLst></ns0:dLbl><ns0:dLbl><ns0:idx val="2" /><ns0:layout><ns0:manualLayout><ns0:x val="5.879629629629618E-2" /><ns0:y '
              'val="2.0158730158730157E-2" /></ns0:manualLayout></ns0:layout><ns0:showLegendKey val="0" /><ns0:showVal val="1" /><ns0:showCatName val="0" '
              '/><ns0:showSerName val="0" /><ns0:showPercent val="0" /><ns0:showBubbleSize val="0" /><ns0:extLst><ns0:ext '
              'uri="{CE6537A1-D6FC-4f65-9D91-7224C49458BB}" /><ns0:ext uri="{C3380CC4-5D6E-409C-BE32-E72D297353CC}"><ns2:uniqueId '
              'val="{00000001-B027-4F48-B518-4DF87DAB4289}" /></ns0:ext></ns0:extLst></ns0:dLbl><ns0:spPr><ns1:noFill /><ns1:ln><ns1:noFill '
              '/></ns1:ln><ns1:effectLst /></ns0:spPr><ns0:txPr><ns1:bodyPr rot="0" spcFirstLastPara="1" vertOverflow="ellipsis" vert="horz" wrap="square" '
              'lIns="38100" tIns="19050" rIns="38100" bIns="19050" anchor="ctr" anchorCtr="1"><ns1:spAutoFit /></ns1:bodyPr><ns1:lstStyle '
              '/><ns1:p><ns1:pPr><ns1:defRPr sz="900" b="0" i="0" u="none" strike="noStrike" kern="1200" baseline="0"><ns1:solidFill><ns1:sysClr '
              'val="windowText" lastClr="000000" /></ns1:solidFill><ns1:latin typeface="Gill Sans MT" panose="020B0502020104020203" pitchFamily="34" '
              'charset="0" /><ns1:ea typeface="+mn-ea" /><ns1:cs typeface="+mn-cs" /></ns1:defRPr></ns1:pPr><ns1:endParaRPr lang="es-ES" '
              '/></ns1:p></ns0:txPr><ns0:showLegendKey val="0" /><ns0:showVal val="1" /><ns0:showCatName val="0" /><ns0:showSerName val="0" /><ns0:showPercent '
              'val="0" /><ns0:showBubbleSize val="0" /><ns0:showLeaderLines val="0" /><ns0:extLst><ns0:ext '
              'uri="{CE6537A1-D6FC-4f65-9D91-7224C49458BB}"><ns3:showLeaderLines val="1" /><ns3:leaderLines><ns0:spPr><ns1:ln w="9525" cap="flat" cmpd="sng" '
              'algn="ctr"><ns1:solidFill><ns1:schemeClr val="tx1"><ns1:lumMod val="35000" /><ns1:lumOff val="65000" '
              '/></ns1:schemeClr></ns1:solidFill><ns1:round /></ns1:ln><ns1:effectLst '
              '/></ns0:spPr></ns3:leaderLines></ns0:ext></ns0:extLst></ns0:dLbls><ns0:errBars><ns0:errBarType val="both" /><ns0:errValType val="cust" '
              '/><ns0:noEndCap val="0" /><ns0:plus><ns0:numRef><ns0:f>\'Gráfico RE '
              '2.1\'!$C$6:$E$6</ns0:f><ns0:numCache><ns0:formatCode>General</ns0:formatCode><ns0:ptCount val="3" /><ns0:pt '
              'idx="0"><ns0:v>0.55999999999999994</ns0:v></ns0:pt><ns0:pt '
              'idx="2"><ns0:v>8.9999999999999969E-2</ns0:v></ns0:pt></ns0:numCache></ns0:numRef></ns0:plus><ns0:minus><ns0:numRef><ns0:f>\'Gráfico RE '
              '2.1\'!$C$7:$E$7</ns0:f><ns0:numCache><ns0:formatCode>General</ns0:formatCode><ns0:ptCount val="3" /><ns0:pt '
              'idx="0"><ns0:v>0.6100000000000001</ns0:v></ns0:pt><ns0:pt '
              'idx="2"><ns0:v>0.29000000000000004</ns0:v></ns0:pt></ns0:numCache></ns0:numRef></ns0:minus><ns0:spPr><ns1:noFill /><ns1:ln w="28575" cap="flat" '
              'cmpd="sng" algn="ctr"><ns1:solidFill><ns1:schemeClr val="bg1"><ns1:lumMod val="50000" /></ns1:schemeClr></ns1:solidFill><ns1:round '
              "/></ns1:ln><ns1:effectLst /></ns0:spPr></ns0:errBars><ns0:cat><ns0:strRef><ns0:f>'Gráfico RE 2.1'!$C$4:$E$4</ns0:f><ns0:strCache><ns0:ptCount "
              'val="3" /><ns0:pt idx="0"><ns0:v>Aghion y Bunel (2025)</ns0:v></ns0:pt><ns0:pt idx="1"><ns0:v>Acemoglu (2024)</ns0:v></ns0:pt><ns0:pt '
              'idx="2"><ns0:v> Filippucci et al (2024)</ns0:v></ns0:pt></ns0:strCache></ns0:strRef></ns0:cat><ns0:val><ns0:numRef><ns0:f>\'Gráfico RE '
              '2.1\'!$C$5:$E$5</ns0:f><ns0:numCache><ns0:formatCode>0.00</ns0:formatCode><ns0:ptCount val="3" /><ns0:pt '
              'idx="0"><ns0:v>0.68</ns0:v></ns0:pt><ns0:pt idx="1"><ns0:v>7.0000000000000007E-2</ns0:v></ns0:pt><ns0:pt '
              'idx="2"><ns0:v>0.53</ns0:v></ns0:pt></ns0:numCache></ns0:numRef></ns0:val><ns0:extLst><ns0:ext '
              'uri="{C3380CC4-5D6E-409C-BE32-E72D297353CC}"><ns2:uniqueId val="{00000002-B027-4F48-B518-4DF87DAB4289}" /></ns0:ext></ns0:extLst></ns0:ser>',
  'root_axes': ['<ns0:valAx xmlns:ns0="http://schemas.openxmlformats.org/drawingml/2006/chart" '
                'xmlns:ns1="http://schemas.openxmlformats.org/drawingml/2006/main"><ns0:axId val="1807476896" /><ns0:scaling><ns0:orientation val="minMax" '
                '/></ns0:scaling><ns0:delete val="0" /><ns0:axPos val="l" /><ns0:numFmt formatCode="0.0" sourceLinked="0" /><ns0:majorTickMark val="none" '
                '/><ns0:minorTickMark val="none" /><ns0:tickLblPos val="nextTo" /><ns0:spPr><ns1:noFill /><ns1:ln><ns1:solidFill><ns1:sysClr val="windowText" '
                'lastClr="000000" /></ns1:solidFill></ns1:ln><ns1:effectLst /></ns0:spPr><ns0:txPr><ns1:bodyPr rot="-60000000" spcFirstLastPara="1" '
                'vertOverflow="ellipsis" vert="horz" wrap="square" anchor="ctr" anchorCtr="1" /><ns1:lstStyle /><ns1:p><ns1:pPr><ns1:defRPr sz="900" b="0" '
                'i="0" u="none" strike="noStrike" kern="1200" baseline="0"><ns1:solidFill><ns1:sysClr val="windowText" lastClr="000000" '
                '/></ns1:solidFill><ns1:latin typeface="Gill Sans MT" panose="020B0502020104020203" pitchFamily="34" charset="0" /><ns1:ea typeface="+mn-ea" '
                '/><ns1:cs typeface="+mn-cs" /></ns1:defRPr></ns1:pPr><ns1:endParaRPr lang="es-ES" /></ns1:p></ns0:txPr><ns0:crossAx val="1807487712" '
                '/><ns0:crosses val="autoZero" /><ns0:crossBetween val="between" /></ns0:valAx>'],
  'value_ref': "'Gráfico RE 2.1'!$C$5:$E$5",
  'category_ref': "'Gráfico RE 2.1'!$C$4:$E$4",
  'name_ref': None,
  'physical_value_cells': ['C5', 'D5', 'E5'],
  'physical_category_cells': ['C4', 'D4', 'E4']},
 {'book': '2025',
  'block': 'Gráfico RE 3.2',
  'sid': 'chart89_s1',
  'headers': [('B2', 'GRÁFICO RE_3.2'), ('C2', 'JUBILACIONES DEMORADAS. MCVL (%ALTAS TOTALES)'), ('B6', '%personas que demoran')],
  'keys': [(18289, 'C6', 'chart89_s1_p0'), (18292, 'F6', 'chart89_s1_p3'), (18291, 'E6', 'chart89_s1_p2'), (18290, 'D6', 'chart89_s1_p1')],
  'mode': 'normal',
  'unit': 'percent_total_retirement_entries',
  'multiplier': '100',
  'pdf': [107, ['DEMORADAS', 'ALTAS TOTALES']],
  'label_cell': 'B6',
  'role': 'published_delayed_retirement_share_total_entries',
  'part': 'xl/charts/chart89.xml',
  'refs': ["'Gráfico RE 3.2'!$C$5:$F$5", "'Gráfico RE 3.2'!$C$6:$F$6"],
  'node_xml': '<ns0:ser xmlns:ns0="http://schemas.openxmlformats.org/drawingml/2006/chart" xmlns:ns1="http://schemas.openxmlformats.org/drawingml/2006/main" '
              'xmlns:ns2="http://schemas.microsoft.com/office/drawing/2012/chart" xmlns:ns3="http://schemas.microsoft.com/office/drawing/2014/chart"><ns0:idx '
              'val="0" /><ns0:order val="0" /><ns0:spPr><ns1:solidFill><ns1:schemeClr val="accent1" /></ns1:solidFill><ns1:ln><ns1:noFill '
              '/></ns1:ln><ns1:effectLst /></ns0:spPr><ns0:invertIfNegative val="0" /><ns0:dLbls><ns0:numFmt formatCode="0.0%" sourceLinked="0" '
              '/><ns0:spPr><ns1:noFill /><ns1:ln><ns1:noFill /></ns1:ln><ns1:effectLst /></ns0:spPr><ns0:txPr><ns1:bodyPr rot="0" spcFirstLastPara="1" '
              'vertOverflow="ellipsis" vert="horz" wrap="square" lIns="38100" tIns="19050" rIns="38100" bIns="19050" anchor="ctr" anchorCtr="1"><ns1:spAutoFit '
              '/></ns1:bodyPr><ns1:lstStyle /><ns1:p><ns1:pPr><ns1:defRPr sz="900" b="0" i="0" u="none" strike="noStrike" kern="1200" '
              'baseline="0"><ns1:solidFill><ns1:schemeClr val="tx1"><ns1:lumMod val="75000" /><ns1:lumOff val="25000" '
              '/></ns1:schemeClr></ns1:solidFill><ns1:latin typeface="+mn-lt" /><ns1:ea typeface="+mn-ea" /><ns1:cs typeface="+mn-cs" '
              '/></ns1:defRPr></ns1:pPr><ns1:endParaRPr lang="es-ES" /></ns1:p></ns0:txPr><ns0:showLegendKey val="0" /><ns0:showVal val="1" /><ns0:showCatName '
              'val="0" /><ns0:showSerName val="0" /><ns0:showPercent val="0" /><ns0:showBubbleSize val="0" /><ns0:showLeaderLines val="0" '
              '/><ns0:extLst><ns0:ext uri="{CE6537A1-D6FC-4f65-9D91-7224C49458BB}"><ns2:showLeaderLines val="1" /><ns2:leaderLines><ns0:spPr><ns1:ln w="9525" '
              'cap="flat" cmpd="sng" algn="ctr"><ns1:solidFill><ns1:schemeClr val="tx1"><ns1:lumMod val="35000" /><ns1:lumOff val="65000" '
              '/></ns1:schemeClr></ns1:solidFill><ns1:round /></ns1:ln><ns1:effectLst '
              "/></ns0:spPr></ns2:leaderLines></ns0:ext></ns0:extLst></ns0:dLbls><ns0:cat><ns0:numRef><ns0:f>'Gráfico RE "
              '3.2\'!$C$5:$F$5</ns0:f><ns0:numCache><ns0:formatCode>General</ns0:formatCode><ns0:ptCount val="4" /><ns0:pt '
              'idx="0"><ns0:v>2019</ns0:v></ns0:pt><ns0:pt idx="1"><ns0:v>2021</ns0:v></ns0:pt><ns0:pt idx="2"><ns0:v>2022</ns0:v></ns0:pt><ns0:pt '
              'idx="3"><ns0:v>2023</ns0:v></ns0:pt></ns0:numCache></ns0:numRef></ns0:cat><ns0:val><ns0:numRef><ns0:f>\'Gráfico RE '
              '3.2\'!$C$6:$F$6</ns0:f><ns0:numCache><ns0:formatCode>0.00%</ns0:formatCode><ns0:ptCount val="4" /><ns0:pt '
              'idx="0"><ns0:v>5.0159018627896408E-2</ns0:v></ns0:pt><ns0:pt idx="1"><ns0:v>4.8272078990674712E-2</ns0:v></ns0:pt><ns0:pt '
              'idx="2"><ns0:v>5.4567656261924752E-2</ns0:v></ns0:pt><ns0:pt '
              'idx="3"><ns0:v>8.0520335913057794E-2</ns0:v></ns0:pt></ns0:numCache></ns0:numRef></ns0:val><ns0:extLst><ns0:ext '
              'uri="{C3380CC4-5D6E-409C-BE32-E72D297353CC}"><ns3:uniqueId val="{00000000-5327-4811-BF9D-2588EF2D7F57}" /></ns0:ext></ns0:extLst></ns0:ser>',
  'root_axes': ['<ns0:valAx xmlns:ns0="http://schemas.openxmlformats.org/drawingml/2006/chart" '
                'xmlns:ns1="http://schemas.openxmlformats.org/drawingml/2006/main"><ns0:axId val="717459455" /><ns0:scaling><ns0:orientation val="minMax" '
                '/></ns0:scaling><ns0:delete val="0" /><ns0:axPos val="l" /><ns0:majorGridlines><ns0:spPr><ns1:ln w="9525" cap="flat" cmpd="sng" '
                'algn="ctr"><ns1:noFill /><ns1:round /></ns1:ln><ns1:effectLst /></ns0:spPr></ns0:majorGridlines><ns0:numFmt formatCode="0%" sourceLinked="0" '
                '/><ns0:majorTickMark val="none" /><ns0:minorTickMark val="none" /><ns0:tickLblPos val="nextTo" /><ns0:spPr><ns1:noFill /><ns1:ln><ns1:noFill '
                '/></ns1:ln><ns1:effectLst /></ns0:spPr><ns0:txPr><ns1:bodyPr rot="-60000000" spcFirstLastPara="1" vertOverflow="ellipsis" vert="horz" '
                'wrap="square" anchor="ctr" anchorCtr="1" /><ns1:lstStyle /><ns1:p><ns1:pPr><ns1:defRPr sz="900" b="0" i="0" u="none" strike="noStrike" '
                'kern="1200" baseline="0"><ns1:solidFill><ns1:schemeClr val="tx1"><ns1:lumMod val="65000" /><ns1:lumOff val="35000" '
                '/></ns1:schemeClr></ns1:solidFill><ns1:latin typeface="+mn-lt" /><ns1:ea typeface="+mn-ea" /><ns1:cs typeface="+mn-cs" '
                '/></ns1:defRPr></ns1:pPr><ns1:endParaRPr lang="es-ES" /></ns1:p></ns0:txPr><ns0:crossAx val="717452799" /><ns0:crosses val="autoZero" '
                '/><ns0:crossBetween val="between" /></ns0:valAx>'],
  'value_ref': "'Gráfico RE 3.2'!$C$6:$F$6",
  'category_ref': "'Gráfico RE 3.2'!$C$5:$F$5",
  'name_ref': None,
  'physical_value_cells': ['C6', 'D6', 'E6', 'F6'],
  'physical_category_cells': ['C5', 'D5', 'E5', 'F5'],
  'scope': 'published_quantity_as_source_labelled'},
 {'book': '2025',
  'block': 'Gráfico 72',
  'sid': 'chart125_s1',
  'headers': [('B2', 'Gráfico 72. DIFERENCIA CON OPINIÓN DE 2023'),
              ('B4', 'Diferencia de la ratio de deuda en 2070'),
              ('B5', 'Mejor punto de partida 2024'),
              ('B6', 'Mayor acumulación de deuda hasta 2070')],
  'keys': [(36605, 'C5', 'chart125_s1_p0')],
  'mode': 'normal',
  'unit': 'percentage_points_gdp',
  'multiplier': '1',
  'pdf': [131, ['5 puntos', '2070']],
  'role': 'published_debt_ratio_difference_component',
  'scope': 'difference_vs_Opinion2023_at_2070',
  'horizon': '2070',
  'part': 'xl/charts/chart125.xml',
  'refs': ["'Gráfico 72'!$B$5", "'Gráfico 72'!$C$5"],
  'node_xml': '<ns0:ser xmlns:ns0="http://schemas.openxmlformats.org/drawingml/2006/chart" xmlns:ns1="http://schemas.openxmlformats.org/drawingml/2006/main" '
              'xmlns:ns2="http://schemas.microsoft.com/office/drawing/2014/chart" xmlns:ns3="http://schemas.microsoft.com/office/drawing/2012/chart"><ns0:idx '
              'val="1" /><ns0:order val="1" /><ns0:tx><ns0:strRef><ns0:f>\'Gráfico 72\'!$B$5</ns0:f><ns0:strCache><ns0:ptCount val="1" /><ns0:pt '
              'idx="0"><ns0:v>Mejor punto de partida 2024</ns0:v></ns0:pt></ns0:strCache></ns0:strRef></ns0:tx><ns0:spPr><ns1:solidFill><ns1:schemeClr '
              'val="bg1"><ns1:lumMod val="85000" /></ns1:schemeClr></ns1:solidFill><ns1:ln><ns1:noFill /></ns1:ln><ns1:effectLst '
              '/></ns0:spPr><ns0:invertIfNegative val="0" /><ns0:dLbls><ns0:dLbl><ns0:idx val="0" /><ns0:layout><ns0:manualLayout><ns0:x '
              'val="-3.5096950489533372E-3" /><ns0:y val="7.037550585061339E-2" /></ns0:manualLayout></ns0:layout><ns0:dLblPos val="ctr" /><ns0:showLegendKey '
              'val="0" /><ns0:showVal val="1" /><ns0:showCatName val="0" /><ns0:showSerName val="0" /><ns0:showPercent val="0" /><ns0:showBubbleSize val="0" '
              '/><ns0:extLst><ns0:ext uri="{CE6537A1-D6FC-4f65-9D91-7224C49458BB}" /><ns0:ext uri="{C3380CC4-5D6E-409C-BE32-E72D297353CC}"><ns2:uniqueId '
              'val="{00000000-6C44-4976-A5F1-BDB1C5C23A10}" /></ns0:ext></ns0:extLst></ns0:dLbl><ns0:spPr><ns1:noFill /><ns1:ln><ns1:noFill '
              '/></ns1:ln><ns1:effectLst /></ns0:spPr><ns0:txPr><ns1:bodyPr rot="0" spcFirstLastPara="1" vertOverflow="ellipsis" vert="horz" wrap="square" '
              'lIns="38100" tIns="19050" rIns="38100" bIns="19050" anchor="ctr" anchorCtr="1"><ns1:spAutoFit /></ns1:bodyPr><ns1:lstStyle '
              '/><ns1:p><ns1:pPr><ns1:defRPr sz="1000" b="0" i="0" u="none" strike="noStrike" kern="1200" baseline="0"><ns1:solidFill><ns1:schemeClr '
              'val="tx1"><ns1:lumMod val="50000" /><ns1:lumOff val="50000" /></ns1:schemeClr></ns1:solidFill><ns1:latin typeface="Gill Sans MT" '
              'panose="020B0502020104020203" pitchFamily="34" charset="0" /><ns1:ea typeface="+mn-ea" /><ns1:cs typeface="+mn-cs" '
              '/></ns1:defRPr></ns1:pPr><ns1:endParaRPr lang="es-ES" /></ns1:p></ns0:txPr><ns0:dLblPos val="inEnd" /><ns0:showLegendKey val="0" /><ns0:showVal '
              'val="1" /><ns0:showCatName val="0" /><ns0:showSerName val="0" /><ns0:showPercent val="0" /><ns0:showBubbleSize val="0" /><ns0:showLeaderLines '
              'val="0" /><ns0:extLst><ns0:ext uri="{CE6537A1-D6FC-4f65-9D91-7224C49458BB}"><ns3:showLeaderLines val="1" /><ns3:leaderLines><ns0:spPr><ns1:ln '
              'w="9525" cap="flat" cmpd="sng" algn="ctr"><ns1:solidFill><ns1:schemeClr val="tx1"><ns1:lumMod val="35000" /><ns1:lumOff val="65000" '
              '/></ns1:schemeClr></ns1:solidFill><ns1:round /></ns1:ln><ns1:effectLst '
              "/></ns0:spPr></ns3:leaderLines></ns0:ext></ns0:extLst></ns0:dLbls><ns0:val><ns0:numRef><ns0:f>'Gráfico "
              '72\'!$C$5</ns0:f><ns0:numCache><ns0:formatCode>#,##0</ns0:formatCode><ns0:ptCount val="1" /><ns0:pt '
              'idx="0"><ns0:v>-7.428249360531737</ns0:v></ns0:pt></ns0:numCache></ns0:numRef></ns0:val><ns0:extLst><ns0:ext '
              'uri="{C3380CC4-5D6E-409C-BE32-E72D297353CC}"><ns2:uniqueId val="{00000001-6C44-4976-A5F1-BDB1C5C23A10}" /></ns0:ext></ns0:extLst></ns0:ser>',
  'root_axes': ['<ns0:valAx xmlns:ns0="http://schemas.openxmlformats.org/drawingml/2006/chart" '
                'xmlns:ns1="http://schemas.openxmlformats.org/drawingml/2006/main"><ns0:axId val="180534160" /><ns0:scaling><ns0:orientation val="minMax" '
                '/></ns0:scaling><ns0:delete val="0" /><ns0:axPos val="l" /><ns0:numFmt formatCode="0" sourceLinked="0" /><ns0:majorTickMark val="none" '
                '/><ns0:minorTickMark val="none" /><ns0:tickLblPos val="nextTo" /><ns0:spPr><ns1:noFill /><ns1:ln><ns1:noFill /></ns1:ln><ns1:effectLst '
                '/></ns0:spPr><ns0:txPr><ns1:bodyPr rot="-60000000" spcFirstLastPara="1" vertOverflow="ellipsis" vert="horz" wrap="square" anchor="ctr" '
                'anchorCtr="1" /><ns1:lstStyle /><ns1:p><ns1:pPr><ns1:defRPr sz="1000" b="0" i="0" u="none" strike="noStrike" kern="1200" '
                'baseline="0"><ns1:solidFill><ns1:schemeClr val="tx1"><ns1:lumMod val="65000" /><ns1:lumOff val="35000" '
                '/></ns1:schemeClr></ns1:solidFill><ns1:latin typeface="Gill Sans MT" panose="020B0502020104020203" pitchFamily="34" charset="0" /><ns1:ea '
                'typeface="+mn-ea" /><ns1:cs typeface="+mn-cs" /></ns1:defRPr></ns1:pPr><ns1:endParaRPr lang="es-ES" /></ns1:p></ns0:txPr><ns0:crossAx '
                'val="180525840" /><ns0:crosses val="autoZero" /><ns0:crossBetween val="between" /></ns0:valAx>'],
  'value_ref': "'Gráfico 72'!$C$5",
  'category_ref': None,
  'name_ref': "'Gráfico 72'!$B$5",
  'physical_value_cells': ['C5'],
  'physical_category_cells': []},
 {'book': '2026',
  'block': 'Gráfico 17',
  'sid': 'chart25_s2',
  'headers': [('B3', 'GRÁFICO 17. NAWRU (%)'), ('B6', 'YEAR'), ('C6', 'Variable'), ('D6', 'Previsión')],
  'keys': [(3681, 'cache[12]', 'chart25_s2_p12'),
           (3690, 'cache[21]', 'chart25_s2_p21'),
           (3736, 'cache[67]', 'chart25_s2_p67'),
           (3680, 'cache[11]', 'chart25_s2_p11'),
           (3697, 'cache[28]', 'chart25_s2_p28'),
           (3709, 'cache[40]', 'chart25_s2_p40'),
           (3714, 'cache[45]', 'chart25_s2_p45'),
           (3738, 'cache[69]', 'chart25_s2_p69'),
           (3678, 'cache[9]', 'chart25_s2_p9'),
           (3700, 'cache[31]', 'chart25_s2_p31'),
           (3701, 'cache[32]', 'chart25_s2_p32'),
           (3713, 'cache[44]', 'chart25_s2_p44'),
           (3734, 'cache[65]', 'chart25_s2_p65'),
           (3691, 'cache[22]', 'chart25_s2_p22'),
           (3705, 'cache[36]', 'chart25_s2_p36'),
           (3722, 'cache[53]', 'chart25_s2_p53'),
           (3735, 'cache[66]', 'chart25_s2_p66'),
           (3671, 'cache[2]', 'chart25_s2_p2'),
           (3672, 'cache[3]', 'chart25_s2_p3'),
           (3675, 'cache[6]', 'chart25_s2_p6'),
           (3682, 'cache[13]', 'chart25_s2_p13'),
           (3694, 'cache[25]', 'chart25_s2_p25'),
           (3720, 'cache[51]', 'chart25_s2_p51'),
           (3721, 'cache[52]', 'chart25_s2_p52'),
           (3726, 'cache[57]', 'chart25_s2_p57'),
           (3732, 'cache[63]', 'chart25_s2_p63'),
           (3703, 'cache[34]', 'chart25_s2_p34'),
           (3715, 'cache[46]', 'chart25_s2_p46'),
           (3729, 'cache[60]', 'chart25_s2_p60'),
           (3731, 'cache[62]', 'chart25_s2_p62'),
           (3669, 'cache[0]', 'chart25_s2_p0'),
           (3674, 'cache[5]', 'chart25_s2_p5'),
           (3692, 'cache[23]', 'chart25_s2_p23'),
           (3698, 'cache[29]', 'chart25_s2_p29'),
           (3711, 'cache[42]', 'chart25_s2_p42'),
           (3683, 'cache[14]', 'chart25_s2_p14'),
           (3704, 'cache[35]', 'chart25_s2_p35'),
           (3706, 'cache[37]', 'chart25_s2_p37'),
           (3710, 'cache[41]', 'chart25_s2_p41'),
           (3686, 'cache[17]', 'chart25_s2_p17'),
           (3688, 'cache[19]', 'chart25_s2_p19'),
           (3695, 'cache[26]', 'chart25_s2_p26'),
           (3699, 'cache[30]', 'chart25_s2_p30'),
           (3708, 'cache[39]', 'chart25_s2_p39'),
           (3717, 'cache[48]', 'chart25_s2_p48'),
           (3725, 'cache[56]', 'chart25_s2_p56'),
           (3730, 'cache[61]', 'chart25_s2_p61'),
           (3733, 'cache[64]', 'chart25_s2_p64'),
           (3739, 'cache[70]', 'chart25_s2_p70'),
           (3685, 'cache[16]', 'chart25_s2_p16'),
           (3693, 'cache[24]', 'chart25_s2_p24'),
           (3702, 'cache[33]', 'chart25_s2_p33'),
           (3719, 'cache[50]', 'chart25_s2_p50'),
           (3723, 'cache[54]', 'chart25_s2_p54'),
           (3728, 'cache[59]', 'chart25_s2_p59'),
           (3670, 'cache[1]', 'chart25_s2_p1'),
           (3677, 'cache[8]', 'chart25_s2_p8'),
           (3679, 'cache[10]', 'chart25_s2_p10'),
           (3684, 'cache[15]', 'chart25_s2_p15'),
           (3689, 'cache[20]', 'chart25_s2_p20'),
           (3707, 'cache[38]', 'chart25_s2_p38'),
           (3712, 'cache[43]', 'chart25_s2_p43'),
           (3718, 'cache[49]', 'chart25_s2_p49'),
           (3696, 'cache[27]', 'chart25_s2_p27'),
           (3716, 'cache[47]', 'chart25_s2_p47'),
           (3724, 'cache[55]', 'chart25_s2_p55'),
           (3727, 'cache[58]', 'chart25_s2_p58'),
           (3737, 'cache[68]', 'chart25_s2_p68'),
           (3673, 'cache[4]', 'chart25_s2_p4'),
           (3676, 'cache[7]', 'chart25_s2_p7'),
           (3687, 'cache[18]', 'chart25_s2_p18')],
  'mode': 'nawru_literal',
  'unit': 'percent_nawru',
  'multiplier': '1',
  'pdf': [60, ['GRÁFICO 17. NAWRU (%)']],
  'label_cell': 'B3',
  'role': 'published_nawru_rate_not_forecast_marker',
  'scope': 'NAWRU_by_caption_not_literal_Prevision_forecast_classification',
  'part': 'xl/charts/chart25.xml',
  'refs': [],
  'node_xml': '<ns0:ser xmlns:ns0="http://schemas.openxmlformats.org/drawingml/2006/chart" xmlns:ns1="http://schemas.openxmlformats.org/drawingml/2006/main" '
              'xmlns:ns2="http://schemas.microsoft.com/office/drawing/2014/chart"><ns0:idx val="1" /><ns0:order val="0" '
              '/><ns0:tx><ns0:v>Previsión</ns0:v></ns0:tx><ns0:spPr><ns1:ln w="28575" cap="rnd"><ns1:solidFill><ns1:srgbClr val="8C2633" '
              '/></ns1:solidFill><ns1:prstDash val="solid" /><ns1:round /></ns1:ln><ns1:effectLst /></ns0:spPr><ns0:marker><ns0:symbol val="none" '
              '/></ns0:marker><ns0:cat><ns0:numLit><ns0:formatCode>General</ns0:formatCode><ns0:ptCount val="71" /><ns0:pt '
              'idx="0"><ns0:v>1980</ns0:v></ns0:pt><ns0:pt idx="1"><ns0:v>1981</ns0:v></ns0:pt><ns0:pt idx="2"><ns0:v>1982</ns0:v></ns0:pt><ns0:pt '
              'idx="3"><ns0:v>1983</ns0:v></ns0:pt><ns0:pt idx="4"><ns0:v>1984</ns0:v></ns0:pt><ns0:pt idx="5"><ns0:v>1985</ns0:v></ns0:pt><ns0:pt '
              'idx="6"><ns0:v>1986</ns0:v></ns0:pt><ns0:pt idx="7"><ns0:v>1987</ns0:v></ns0:pt><ns0:pt idx="8"><ns0:v>1988</ns0:v></ns0:pt><ns0:pt '
              'idx="9"><ns0:v>1989</ns0:v></ns0:pt><ns0:pt idx="10"><ns0:v>1990</ns0:v></ns0:pt><ns0:pt idx="11"><ns0:v>1991</ns0:v></ns0:pt><ns0:pt '
              'idx="12"><ns0:v>1992</ns0:v></ns0:pt><ns0:pt idx="13"><ns0:v>1993</ns0:v></ns0:pt><ns0:pt idx="14"><ns0:v>1994</ns0:v></ns0:pt><ns0:pt '
              'idx="15"><ns0:v>1995</ns0:v></ns0:pt><ns0:pt idx="16"><ns0:v>1996</ns0:v></ns0:pt><ns0:pt idx="17"><ns0:v>1997</ns0:v></ns0:pt><ns0:pt '
              'idx="18"><ns0:v>1998</ns0:v></ns0:pt><ns0:pt idx="19"><ns0:v>1999</ns0:v></ns0:pt><ns0:pt idx="20"><ns0:v>2000</ns0:v></ns0:pt><ns0:pt '
              'idx="21"><ns0:v>2001</ns0:v></ns0:pt><ns0:pt idx="22"><ns0:v>2002</ns0:v></ns0:pt><ns0:pt idx="23"><ns0:v>2003</ns0:v></ns0:pt><ns0:pt '
              'idx="24"><ns0:v>2004</ns0:v></ns0:pt><ns0:pt idx="25"><ns0:v>2005</ns0:v></ns0:pt><ns0:pt idx="26"><ns0:v>2006</ns0:v></ns0:pt><ns0:pt '
              'idx="27"><ns0:v>2007</ns0:v></ns0:pt><ns0:pt idx="28"><ns0:v>2008</ns0:v></ns0:pt><ns0:pt idx="29"><ns0:v>2009</ns0:v></ns0:pt><ns0:pt '
              'idx="30"><ns0:v>2010</ns0:v></ns0:pt><ns0:pt idx="31"><ns0:v>2011</ns0:v></ns0:pt><ns0:pt idx="32"><ns0:v>2012</ns0:v></ns0:pt><ns0:pt '
              'idx="33"><ns0:v>2013</ns0:v></ns0:pt><ns0:pt idx="34"><ns0:v>2014</ns0:v></ns0:pt><ns0:pt idx="35"><ns0:v>2015</ns0:v></ns0:pt><ns0:pt '
              'idx="36"><ns0:v>2016</ns0:v></ns0:pt><ns0:pt idx="37"><ns0:v>2017</ns0:v></ns0:pt><ns0:pt idx="38"><ns0:v>2018</ns0:v></ns0:pt><ns0:pt '
              'idx="39"><ns0:v>2019</ns0:v></ns0:pt><ns0:pt idx="40"><ns0:v>2020</ns0:v></ns0:pt><ns0:pt idx="41"><ns0:v>2021</ns0:v></ns0:pt><ns0:pt '
              'idx="42"><ns0:v>2022</ns0:v></ns0:pt><ns0:pt idx="43"><ns0:v>2023</ns0:v></ns0:pt><ns0:pt idx="44"><ns0:v>2024</ns0:v></ns0:pt><ns0:pt '
              'idx="45"><ns0:v>2025</ns0:v></ns0:pt><ns0:pt idx="46"><ns0:v>2026</ns0:v></ns0:pt><ns0:pt idx="47"><ns0:v>2027</ns0:v></ns0:pt><ns0:pt '
              'idx="48"><ns0:v>2028</ns0:v></ns0:pt><ns0:pt idx="49"><ns0:v>2029</ns0:v></ns0:pt><ns0:pt idx="50"><ns0:v>2030</ns0:v></ns0:pt><ns0:pt '
              'idx="51"><ns0:v>2031</ns0:v></ns0:pt><ns0:pt idx="52"><ns0:v>2032</ns0:v></ns0:pt><ns0:pt idx="53"><ns0:v>2033</ns0:v></ns0:pt><ns0:pt '
              'idx="54"><ns0:v>2034</ns0:v></ns0:pt><ns0:pt idx="55"><ns0:v>2035</ns0:v></ns0:pt><ns0:pt idx="56"><ns0:v>2036</ns0:v></ns0:pt><ns0:pt '
              'idx="57"><ns0:v>2037</ns0:v></ns0:pt><ns0:pt idx="58"><ns0:v>2038</ns0:v></ns0:pt><ns0:pt idx="59"><ns0:v>2039</ns0:v></ns0:pt><ns0:pt '
              'idx="60"><ns0:v>2040</ns0:v></ns0:pt><ns0:pt idx="61"><ns0:v>2041</ns0:v></ns0:pt><ns0:pt idx="62"><ns0:v>2042</ns0:v></ns0:pt><ns0:pt '
              'idx="63"><ns0:v>2043</ns0:v></ns0:pt><ns0:pt idx="64"><ns0:v>2044</ns0:v></ns0:pt><ns0:pt idx="65"><ns0:v>2045</ns0:v></ns0:pt><ns0:pt '
              'idx="66"><ns0:v>2046</ns0:v></ns0:pt><ns0:pt idx="67"><ns0:v>2047</ns0:v></ns0:pt><ns0:pt idx="68"><ns0:v>2048</ns0:v></ns0:pt><ns0:pt '
              'idx="69"><ns0:v>2049</ns0:v></ns0:pt><ns0:pt '
              'idx="70"><ns0:v>2050</ns0:v></ns0:pt></ns0:numLit></ns0:cat><ns0:val><ns0:numLit><ns0:formatCode>General</ns0:formatCode><ns0:ptCount val="71" '
              '/><ns0:pt idx="0"><ns0:v>10.42395</ns0:v></ns0:pt><ns0:pt idx="1"><ns0:v>11.381133999999999</ns0:v></ns0:pt><ns0:pt '
              'idx="2"><ns0:v>12.367829</ns0:v></ns0:pt><ns0:pt idx="3"><ns0:v>12.580954999999999</ns0:v></ns0:pt><ns0:pt '
              'idx="4"><ns0:v>13.639915</ns0:v></ns0:pt><ns0:pt idx="5"><ns0:v>14.445715</ns0:v></ns0:pt><ns0:pt '
              'idx="6"><ns0:v>14.888036</ns0:v></ns0:pt><ns0:pt idx="7"><ns0:v>15.971059</ns0:v></ns0:pt><ns0:pt '
              'idx="8"><ns0:v>16.354711000000002</ns0:v></ns0:pt><ns0:pt idx="9"><ns0:v>16.538768000000001</ns0:v></ns0:pt><ns0:pt '
              'idx="10"><ns0:v>16.870926999999998</ns0:v></ns0:pt><ns0:pt idx="11"><ns0:v>17.086500999999998</ns0:v></ns0:pt><ns0:pt '
              'idx="12"><ns0:v>17.174147999999999</ns0:v></ns0:pt><ns0:pt idx="13"><ns0:v>17.926382</ns0:v></ns0:pt><ns0:pt '
              'idx="14"><ns0:v>17.793616</ns0:v></ns0:pt><ns0:pt idx="15"><ns0:v>17.325467</ns0:v></ns0:pt><ns0:pt '
              'idx="16"><ns0:v>17.027183999999998</ns0:v></ns0:pt><ns0:pt idx="17"><ns0:v>16.552948000000001</ns0:v></ns0:pt><ns0:pt '
              'idx="18"><ns0:v>16.011092999999999</ns0:v></ns0:pt><ns0:pt idx="19"><ns0:v>15.246767</ns0:v></ns0:pt><ns0:pt '
              'idx="20"><ns0:v>14.80429</ns0:v></ns0:pt><ns0:pt idx="21"><ns0:v>13.900472000000001</ns0:v></ns0:pt><ns0:pt '
              'idx="22"><ns0:v>13.905925</ns0:v></ns0:pt><ns0:pt idx="23"><ns0:v>13.680688</ns0:v></ns0:pt><ns0:pt '
              'idx="24"><ns0:v>13.588664</ns0:v></ns0:pt><ns0:pt idx="25"><ns0:v>13.38017</ns0:v></ns0:pt><ns0:pt '
              'idx="26"><ns0:v>13.511958999999999</ns0:v></ns0:pt><ns0:pt idx="27"><ns0:v>13.601281999999999</ns0:v></ns0:pt><ns0:pt '
              'idx="28"><ns0:v>14.108631000000001</ns0:v></ns0:pt><ns0:pt idx="29"><ns0:v>15.288513</ns0:v></ns0:pt><ns0:pt '
              'idx="30"><ns0:v>15.493067</ns0:v></ns0:pt><ns0:pt idx="31"><ns0:v>15.734017</ns0:v></ns0:pt><ns0:pt '
              'idx="32"><ns0:v>16.392267</ns0:v></ns0:pt><ns0:pt idx="33"><ns0:v>16.720652000000001</ns0:v></ns0:pt><ns0:pt '
              'idx="34"><ns0:v>16.608533000000001</ns0:v></ns0:pt><ns0:pt idx="35"><ns0:v>16.416786999999999</ns0:v></ns0:pt><ns0:pt '
              'idx="36"><ns0:v>16.142043999999999</ns0:v></ns0:pt><ns0:pt idx="37"><ns0:v>15.75977</ns0:v></ns0:pt><ns0:pt '
              'idx="38"><ns0:v>15.335279999999999</ns0:v></ns0:pt><ns0:pt idx="39"><ns0:v>14.841343999999999</ns0:v></ns0:pt><ns0:pt '
              'idx="40"><ns0:v>14.782971</ns0:v></ns0:pt><ns0:pt idx="41"><ns0:v>14.168621</ns0:v></ns0:pt><ns0:pt '
              'idx="42"><ns0:v>13.379911999999999</ns0:v></ns0:pt><ns0:pt idx="43"><ns0:v>12.901655</ns0:v></ns0:pt><ns0:pt '
              'idx="44"><ns0:v>12.38772</ns0:v></ns0:pt><ns0:pt idx="45"><ns0:v>11.888729</ns0:v></ns0:pt><ns0:pt '
              'idx="46"><ns0:v>11.502101</ns0:v></ns0:pt><ns0:pt idx="47"><ns0:v>11.180298000000001</ns0:v></ns0:pt><ns0:pt '
              'idx="48"><ns0:v>10.893064000000001</ns0:v></ns0:pt><ns0:pt idx="49"><ns0:v>10.684941</ns0:v></ns0:pt><ns0:pt '
              'idx="50"><ns0:v>10.543478</ns0:v></ns0:pt><ns0:pt idx="51"><ns0:v>10.438043220000001</ns0:v></ns0:pt><ns0:pt '
              'idx="52"><ns0:v>10.3336627878</ns0:v></ns0:pt><ns0:pt idx="53"><ns0:v>10.230326159922001</ns0:v></ns0:pt><ns0:pt '
              'idx="54"><ns0:v>10.128022898322779</ns0:v></ns0:pt><ns0:pt idx="55"><ns0:v>10.02674266933955</ns0:v></ns0:pt><ns0:pt '
              'idx="56"><ns0:v>9.9264752426461573</ns0:v></ns0:pt><ns0:pt idx="57"><ns0:v>9.8272104902196951</ns0:v></ns0:pt><ns0:pt '
              'idx="58"><ns0:v>9.7289383853174982</ns0:v></ns0:pt><ns0:pt idx="59"><ns0:v>9.6316490014643232</ns0:v></ns0:pt><ns0:pt '
              'idx="60"><ns0:v>9.5353325114496794</ns0:v></ns0:pt><ns0:pt idx="61"><ns0:v>9.439979186335183</ns0:v></ns0:pt><ns0:pt '
              'idx="62"><ns0:v>9.3455793944718302</ns0:v></ns0:pt><ns0:pt idx="63"><ns0:v>9.2521236005271117</ns0:v></ns0:pt><ns0:pt '
              'idx="64"><ns0:v>9.1596023645218398</ns0:v></ns0:pt><ns0:pt idx="65"><ns0:v>9.0680063408766216</ns0:v></ns0:pt><ns0:pt '
              'idx="66"><ns0:v>8.9773262774678546</ns0:v></ns0:pt><ns0:pt idx="67"><ns0:v>8.8875530146931752</ns0:v></ns0:pt><ns0:pt '
              'idx="68"><ns0:v>8.7986774845462428</ns0:v></ns0:pt><ns0:pt idx="69"><ns0:v>8.7106907097007795</ns0:v></ns0:pt><ns0:pt '
              'idx="70"><ns0:v>8.6235838026037719</ns0:v></ns0:pt></ns0:numLit></ns0:val><ns0:smooth val="0" /><ns0:extLst><ns0:ext '
              'uri="{C3380CC4-5D6E-409C-BE32-E72D297353CC}"><ns2:uniqueId val="{00000001-1112-4DD8-9B73-A18C2C6F2DF9}" /></ns0:ext></ns0:extLst></ns0:ser>',
  'root_axes': ['<ns0:valAx xmlns:ns0="http://schemas.openxmlformats.org/drawingml/2006/chart" '
                'xmlns:ns1="http://schemas.openxmlformats.org/drawingml/2006/main"><ns0:axId val="278125167" /><ns0:scaling><ns0:orientation val="minMax" '
                '/></ns0:scaling><ns0:delete val="0" /><ns0:axPos val="l" /><ns0:numFmt formatCode="0" sourceLinked="0" /><ns0:majorTickMark val="none" '
                '/><ns0:minorTickMark val="none" /><ns0:tickLblPos val="nextTo" /><ns0:spPr><ns1:noFill /><ns1:ln><ns1:solidFill><ns1:schemeClr val="tx1" '
                '/></ns1:solidFill></ns1:ln><ns1:effectLst /></ns0:spPr><ns0:txPr><ns1:bodyPr rot="-60000000" spcFirstLastPara="1" vertOverflow="ellipsis" '
                'vert="horz" wrap="square" anchor="ctr" anchorCtr="1" /><ns1:lstStyle /><ns1:p><ns1:pPr><ns1:defRPr sz="900" b="0" i="0" u="none" '
                'strike="noStrike" kern="1200" baseline="0"><ns1:solidFill><ns1:sysClr val="windowText" lastClr="000000" /></ns1:solidFill><ns1:latin '
                'typeface="Gill Sans MT" panose="020B0502020104020203" pitchFamily="34" charset="0" /><ns1:ea typeface="+mn-ea" /><ns1:cs typeface="+mn-cs" '
                '/></ns1:defRPr></ns1:pPr><ns1:endParaRPr lang="es-ES" /></ns1:p></ns0:txPr><ns0:crossAx val="278124687" /><ns0:crosses val="autoZero" '
                '/><ns0:crossBetween val="between" /></ns0:valAx>',
                '<ns0:valAx xmlns:ns0="http://schemas.openxmlformats.org/drawingml/2006/chart" '
                'xmlns:ns1="http://schemas.openxmlformats.org/drawingml/2006/main"><ns0:axId val="278128527" /><ns0:scaling><ns0:orientation val="minMax" '
                '/><ns0:max val="0.9" /></ns0:scaling><ns0:delete val="0" /><ns0:axPos val="r" /><ns0:numFmt formatCode="General" sourceLinked="1" '
                '/><ns0:majorTickMark val="out" /><ns0:minorTickMark val="none" /><ns0:tickLblPos val="none" /><ns0:spPr><ns1:noFill /><ns1:ln><ns1:noFill '
                '/></ns1:ln><ns1:effectLst /></ns0:spPr><ns0:txPr><ns1:bodyPr rot="-60000000" spcFirstLastPara="1" vertOverflow="ellipsis" vert="horz" '
                'wrap="square" anchor="ctr" anchorCtr="1" /><ns1:lstStyle /><ns1:p><ns1:pPr><ns1:defRPr sz="900" b="0" i="0" u="none" strike="noStrike" '
                'kern="1200" baseline="0"><ns1:solidFill><ns1:schemeClr val="tx1"><ns1:lumMod val="65000" /><ns1:lumOff val="35000" '
                '/></ns1:schemeClr></ns1:solidFill><ns1:latin typeface="+mn-lt" /><ns1:ea typeface="+mn-ea" /><ns1:cs typeface="+mn-cs" '
                '/></ns1:defRPr></ns1:pPr><ns1:endParaRPr lang="es-ES" /></ns1:p></ns0:txPr><ns0:crossAx val="278102607" /><ns0:crosses val="max" '
                '/><ns0:crossBetween val="between" /></ns0:valAx>'],
  'value_ref': None,
  'category_ref': None,
  'name_ref': None,
  'physical_value_cells': ['C7',
                           'C8',
                           'C9',
                           'C10',
                           'C11',
                           'C12',
                           'C13',
                           'C14',
                           'C15',
                           'C16',
                           'C17',
                           'C18',
                           'C19',
                           'C20',
                           'C21',
                           'C22',
                           'C23',
                           'C24',
                           'C25',
                           'C26',
                           'C27',
                           'C28',
                           'C29',
                           'C30',
                           'C31',
                           'C32',
                           'C33',
                           'C34',
                           'C35',
                           'C36',
                           'C37',
                           'C38',
                           'C39',
                           'C40',
                           'C41',
                           'C42',
                           'C43',
                           'C44',
                           'C45',
                           'C46',
                           'C47',
                           'C48',
                           'C49',
                           'C50',
                           'C51',
                           'C52',
                           'C53',
                           'C54',
                           'C55',
                           'C56',
                           'C57',
                           'C58',
                           'C59',
                           'C60',
                           'C61',
                           'C62',
                           'C63',
                           'C64',
                           'C65',
                           'C66',
                           'C67',
                           'C68',
                           'C69',
                           'C70',
                           'C71',
                           'C72',
                           'C73',
                           'C74',
                           'C75',
                           'C76',
                           'C77'],
  'physical_category_cells': ['B7',
                              'B8',
                              'B9',
                              'B10',
                              'B11',
                              'B12',
                              'B13',
                              'B14',
                              'B15',
                              'B16',
                              'B17',
                              'B18',
                              'B19',
                              'B20',
                              'B21',
                              'B22',
                              'B23',
                              'B24',
                              'B25',
                              'B26',
                              'B27',
                              'B28',
                              'B29',
                              'B30',
                              'B31',
                              'B32',
                              'B33',
                              'B34',
                              'B35',
                              'B36',
                              'B37',
                              'B38',
                              'B39',
                              'B40',
                              'B41',
                              'B42',
                              'B43',
                              'B44',
                              'B45',
                              'B46',
                              'B47',
                              'B48',
                              'B49',
                              'B50',
                              'B51',
                              'B52',
                              'B53',
                              'B54',
                              'B55',
                              'B56',
                              'B57',
                              'B58',
                              'B59',
                              'B60',
                              'B61',
                              'B62',
                              'B63',
                              'B64',
                              'B65',
                              'B66',
                              'B67',
                              'B68',
                              'B69',
                              'B70',
                              'B71',
                              'B72',
                              'B73',
                              'B74',
                              'B75',
                              'B76',
                              'B77']},
 {'book': '2026',
  'block': 'Gráfico 35',
  'sid': 'chart44_s4',
  'headers': [('B3', 'GRÁFICO 35. INTERVALO DE DEUDA (%PIB), ESCENARIOS DE EVOLUCIÓN DEL PIB'),
              ('C5', 'Inercial'),
              ('D5', '+0.5'),
              ('E5', '-0.5'),
              ('F5', '+0.5'),
              ('G5', '-0.5'),
              ('H5', '+0.5'),
              ('I5', '-0.5'),
              ('J5', '+0.5')],
  'keys': [(5679, 'C10', 'chart44_s4_p3'),
           (5677, 'C8', 'chart44_s4_p1'),
           (5678, 'C9', 'chart44_s4_p2'),
           (5680, 'C11', 'chart44_s4_p4'),
           (5681, 'C12', 'chart44_s4_p5'),
           (5676, 'C7', 'chart44_s4_p0')],
  'mode': 'normal',
  'unit': 'percent_gdp',
  'multiplier': '1',
  'pdf': [72, ['INTERVALO DE DEUDA']],
  'label_cell': 'C5',
  'role': 'published_inertial_curve_point_as_labelled',
  'scope': 'no_observed_forecast_cutoff_inferred',
  'part': 'xl/charts/chart44.xml',
  'refs': ["'Gráfico 35'!$B$7:$B$37", "'Gráfico 35'!$C$7:$C$12"],
  'node_xml': '<ns0:ser xmlns:ns0="http://schemas.openxmlformats.org/drawingml/2006/chart" xmlns:ns1="http://schemas.openxmlformats.org/drawingml/2006/main" '
              'xmlns:ns2="http://schemas.microsoft.com/office/drawing/2014/chart"><ns0:idx val="3" /><ns0:order val="3" /><ns0:spPr><ns1:ln w="25400" '
              'cap="rnd"><ns1:solidFill><ns1:srgbClr val="83082A" /></ns1:solidFill><ns1:round /></ns1:ln><ns1:effectLst /></ns0:spPr><ns0:marker><ns0:symbol '
              'val="none" /></ns0:marker><ns0:cat><ns0:numRef><ns0:f>\'Gráfico '
              '35\'!$B$7:$B$37</ns0:f><ns0:numCache><ns0:formatCode>0</ns0:formatCode><ns0:ptCount val="31" /><ns0:pt '
              'idx="0"><ns0:v>2020</ns0:v></ns0:pt><ns0:pt idx="1"><ns0:v>2021</ns0:v></ns0:pt><ns0:pt idx="2"><ns0:v>2022</ns0:v></ns0:pt><ns0:pt '
              'idx="3"><ns0:v>2023</ns0:v></ns0:pt><ns0:pt idx="4"><ns0:v>2024</ns0:v></ns0:pt><ns0:pt idx="5"><ns0:v>2025</ns0:v></ns0:pt><ns0:pt '
              'idx="6"><ns0:v>2026</ns0:v></ns0:pt><ns0:pt idx="7"><ns0:v>2027</ns0:v></ns0:pt><ns0:pt idx="8"><ns0:v>2028</ns0:v></ns0:pt><ns0:pt '
              'idx="9"><ns0:v>2029</ns0:v></ns0:pt><ns0:pt idx="10"><ns0:v>2030</ns0:v></ns0:pt><ns0:pt idx="11"><ns0:v>2031</ns0:v></ns0:pt><ns0:pt '
              'idx="12"><ns0:v>2032</ns0:v></ns0:pt><ns0:pt idx="13"><ns0:v>2033</ns0:v></ns0:pt><ns0:pt idx="14"><ns0:v>2034</ns0:v></ns0:pt><ns0:pt '
              'idx="15"><ns0:v>2035</ns0:v></ns0:pt><ns0:pt idx="16"><ns0:v>2036</ns0:v></ns0:pt><ns0:pt idx="17"><ns0:v>2037</ns0:v></ns0:pt><ns0:pt '
              'idx="18"><ns0:v>2038</ns0:v></ns0:pt><ns0:pt idx="19"><ns0:v>2039</ns0:v></ns0:pt><ns0:pt idx="20"><ns0:v>2040</ns0:v></ns0:pt><ns0:pt '
              'idx="21"><ns0:v>2041</ns0:v></ns0:pt><ns0:pt idx="22"><ns0:v>2042</ns0:v></ns0:pt><ns0:pt idx="23"><ns0:v>2043</ns0:v></ns0:pt><ns0:pt '
              'idx="24"><ns0:v>2044</ns0:v></ns0:pt><ns0:pt idx="25"><ns0:v>2045</ns0:v></ns0:pt><ns0:pt idx="26"><ns0:v>2046</ns0:v></ns0:pt><ns0:pt '
              'idx="27"><ns0:v>2047</ns0:v></ns0:pt><ns0:pt idx="28"><ns0:v>2048</ns0:v></ns0:pt><ns0:pt idx="29"><ns0:v>2049</ns0:v></ns0:pt><ns0:pt '
              'idx="30"><ns0:v>2050</ns0:v></ns0:pt></ns0:numCache></ns0:numRef></ns0:cat><ns0:val><ns0:numRef><ns0:f>\'Gráfico '
              '35\'!$C$7:$C$12</ns0:f><ns0:numCache><ns0:formatCode>#,##0.0</ns0:formatCode><ns0:ptCount val="6" /><ns0:pt '
              'idx="0"><ns0:v>119.27910652896618</ns0:v></ns0:pt><ns0:pt idx="1"><ns0:v>115.69677629800385</ns0:v></ns0:pt><ns0:pt '
              'idx="2"><ns0:v>109.32084662499099</ns0:v></ns0:pt><ns0:pt idx="3"><ns0:v>105.18211784123102</ns0:v></ns0:pt><ns0:pt '
              'idx="4"><ns0:v>101.64604002935403</ns0:v></ns0:pt><ns0:pt '
              'idx="5"><ns0:v>100.65628823010613</ns0:v></ns0:pt></ns0:numCache></ns0:numRef></ns0:val><ns0:smooth val="0" /><ns0:extLst><ns0:ext '
              'uri="{C3380CC4-5D6E-409C-BE32-E72D297353CC}"><ns2:uniqueId val="{00000044-B728-4FE5-AC66-DF7F4CF60DD4}" /></ns0:ext></ns0:extLst></ns0:ser>',
  'root_axes': ['<ns0:valAx xmlns:ns0="http://schemas.openxmlformats.org/drawingml/2006/chart" '
                'xmlns:ns1="http://schemas.openxmlformats.org/drawingml/2006/main"><ns0:axId val="1966996464" /><ns0:scaling><ns0:orientation val="minMax" '
                '/><ns0:min val="80" /></ns0:scaling><ns0:delete val="0" /><ns0:axPos val="l" /><ns0:numFmt formatCode="0" sourceLinked="0" '
                '/><ns0:majorTickMark val="none" /><ns0:minorTickMark val="none" /><ns0:tickLblPos val="low" /><ns0:spPr><ns1:noFill /><ns1:ln><ns1:noFill '
                '/></ns1:ln><ns1:effectLst /></ns0:spPr><ns0:txPr><ns1:bodyPr rot="-60000000" spcFirstLastPara="1" vertOverflow="ellipsis" vert="horz" '
                'wrap="square" anchor="ctr" anchorCtr="1" /><ns1:lstStyle /><ns1:p><ns1:pPr><ns1:defRPr sz="1000" b="0" i="0" u="none" strike="noStrike" '
                'kern="1200" baseline="0"><ns1:solidFill><ns1:schemeClr val="tx1"><ns1:lumMod val="65000" /><ns1:lumOff val="35000" '
                '/></ns1:schemeClr></ns1:solidFill><ns1:latin typeface="Gill Sans MT" panose="020B0502020104020203" pitchFamily="34" charset="0" /><ns1:ea '
                'typeface="+mn-ea" /><ns1:cs typeface="Times New Roman" panose="02020603050405020304" pitchFamily="18" charset="0" '
                '/></ns1:defRPr></ns1:pPr><ns1:endParaRPr lang="es-ES" /></ns1:p></ns0:txPr><ns0:crossAx val="1966995216" /><ns0:crosses val="autoZero" '
                '/><ns0:crossBetween val="between" /><ns0:majorUnit val="20" /></ns0:valAx>'],
  'value_ref': "'Gráfico 35'!$C$7:$C$12",
  'category_ref': "'Gráfico 35'!$B$7:$B$37",
  'name_ref': None,
  'physical_value_cells': ['C7', 'C8', 'C9', 'C10', 'C11', 'C12'],
  'physical_category_cells': ['B7',
                              'B8',
                              'B9',
                              'B10',
                              'B11',
                              'B12',
                              'B13',
                              'B14',
                              'B15',
                              'B16',
                              'B17',
                              'B18',
                              'B19',
                              'B20',
                              'B21',
                              'B22',
                              'B23',
                              'B24',
                              'B25',
                              'B26',
                              'B27',
                              'B28',
                              'B29',
                              'B30',
                              'B31',
                              'B32',
                              'B33',
                              'B34',
                              'B35',
                              'B36',
                              'B37']},
 {'book': '2025',
  'block': 'Gráfico RE 3.5',
  'sid': 'chart92_s1',
  'headers': [('B2', 'GRÁFICO RE_3.5'),
              ('C2', 'JUBILACIONES ANTICIPADAS VOLUNTARIAS (%ALTAS TOTALES)'),
              ('B5', '% Personas que anticipan'),
              ('B8', 'Fuente: MCVL')],
  'keys': [(18327, 'D5', 'chart92_s1_p1'), (18328, 'E5', 'chart92_s1_p2'), (18329, 'F5', 'chart92_s1_p3'), (18326, 'C5', 'chart92_s1_p0')],
  'mode': 'normal',
  'unit': 'percent_total_retirement_entries',
  'multiplier': '100',
  'pdf': [111, ['ANTICIPADAS', 'ALTAS TOTALES']],
  'label_cell': 'B5',
  'role': 'published_voluntary_early_retirement_share_total_entries',
  'part': 'xl/charts/chart92.xml',
  'refs': ["'Gráfico RE 3.5'!$C$4:$F$4", "'Gráfico RE 3.5'!$C$5:$F$5"],
  'node_xml': '<ns0:ser xmlns:ns0="http://schemas.openxmlformats.org/drawingml/2006/chart" xmlns:ns1="http://schemas.openxmlformats.org/drawingml/2006/main" '
              'xmlns:ns2="http://schemas.microsoft.com/office/drawing/2012/chart" xmlns:ns3="http://schemas.microsoft.com/office/drawing/2014/chart"><ns0:idx '
              'val="0" /><ns0:order val="0" /><ns0:spPr><ns1:solidFill><ns1:schemeClr val="accent1" /></ns1:solidFill><ns1:ln><ns1:noFill '
              '/></ns1:ln><ns1:effectLst /></ns0:spPr><ns0:invertIfNegative val="0" /><ns0:dLbls><ns0:numFmt formatCode="0.0%" sourceLinked="0" '
              '/><ns0:spPr><ns1:noFill /><ns1:ln><ns1:noFill /></ns1:ln><ns1:effectLst /></ns0:spPr><ns0:txPr><ns1:bodyPr rot="0" spcFirstLastPara="1" '
              'vertOverflow="ellipsis" vert="horz" wrap="square" lIns="38100" tIns="19050" rIns="38100" bIns="19050" anchor="ctr" anchorCtr="1"><ns1:spAutoFit '
              '/></ns1:bodyPr><ns1:lstStyle /><ns1:p><ns1:pPr><ns1:defRPr sz="900" b="0" i="0" u="none" strike="noStrike" kern="1200" '
              'baseline="0"><ns1:solidFill><ns1:schemeClr val="tx1"><ns1:lumMod val="75000" /><ns1:lumOff val="25000" '
              '/></ns1:schemeClr></ns1:solidFill><ns1:latin typeface="+mn-lt" /><ns1:ea typeface="+mn-ea" /><ns1:cs typeface="+mn-cs" '
              '/></ns1:defRPr></ns1:pPr><ns1:endParaRPr lang="es-ES" /></ns1:p></ns0:txPr><ns0:showLegendKey val="0" /><ns0:showVal val="1" /><ns0:showCatName '
              'val="0" /><ns0:showSerName val="0" /><ns0:showPercent val="0" /><ns0:showBubbleSize val="0" /><ns0:showLeaderLines val="0" '
              '/><ns0:extLst><ns0:ext uri="{CE6537A1-D6FC-4f65-9D91-7224C49458BB}"><ns2:showLeaderLines val="1" /><ns2:leaderLines><ns0:spPr><ns1:ln w="9525" '
              'cap="flat" cmpd="sng" algn="ctr"><ns1:solidFill><ns1:schemeClr val="tx1"><ns1:lumMod val="35000" /><ns1:lumOff val="65000" '
              '/></ns1:schemeClr></ns1:solidFill><ns1:round /></ns1:ln><ns1:effectLst '
              "/></ns0:spPr></ns2:leaderLines></ns0:ext></ns0:extLst></ns0:dLbls><ns0:cat><ns0:numRef><ns0:f>'Gráfico RE "
              '3.5\'!$C$4:$F$4</ns0:f><ns0:numCache><ns0:formatCode>General</ns0:formatCode><ns0:ptCount val="4" /><ns0:pt '
              'idx="0"><ns0:v>2019</ns0:v></ns0:pt><ns0:pt idx="1"><ns0:v>2021</ns0:v></ns0:pt><ns0:pt idx="2"><ns0:v>2022</ns0:v></ns0:pt><ns0:pt '
              'idx="3"><ns0:v>2023</ns0:v></ns0:pt></ns0:numCache></ns0:numRef></ns0:cat><ns0:val><ns0:numRef><ns0:f>\'Gráfico RE '
              '3.5\'!$C$5:$F$5</ns0:f><ns0:numCache><ns0:formatCode>0.00%</ns0:formatCode><ns0:ptCount val="4" /><ns0:pt '
              'idx="0"><ns0:v>0.16728759654702408</ns0:v></ns0:pt><ns0:pt idx="1"><ns0:v>0.21119034558420186</ns0:v></ns0:pt><ns0:pt '
              'idx="2"><ns0:v>0.23025261390521254</ns0:v></ns0:pt><ns0:pt '
              'idx="3"><ns0:v>0.17841264613864646</ns0:v></ns0:pt></ns0:numCache></ns0:numRef></ns0:val><ns0:extLst><ns0:ext '
              'uri="{C3380CC4-5D6E-409C-BE32-E72D297353CC}"><ns3:uniqueId val="{00000000-808E-4122-A72F-A727FC2BF7E5}" /></ns0:ext></ns0:extLst></ns0:ser>',
  'root_axes': ['<ns0:valAx xmlns:ns0="http://schemas.openxmlformats.org/drawingml/2006/chart" '
                'xmlns:ns1="http://schemas.openxmlformats.org/drawingml/2006/main"><ns0:axId val="717459455" /><ns0:scaling><ns0:orientation val="minMax" '
                '/></ns0:scaling><ns0:delete val="0" /><ns0:axPos val="l" /><ns0:majorGridlines><ns0:spPr><ns1:ln w="9525" cap="flat" cmpd="sng" '
                'algn="ctr"><ns1:noFill /><ns1:round /></ns1:ln><ns1:effectLst /></ns0:spPr></ns0:majorGridlines><ns0:numFmt formatCode="0%" sourceLinked="0" '
                '/><ns0:majorTickMark val="none" /><ns0:minorTickMark val="none" /><ns0:tickLblPos val="nextTo" /><ns0:spPr><ns1:noFill /><ns1:ln><ns1:noFill '
                '/></ns1:ln><ns1:effectLst /></ns0:spPr><ns0:txPr><ns1:bodyPr rot="-60000000" spcFirstLastPara="1" vertOverflow="ellipsis" vert="horz" '
                'wrap="square" anchor="ctr" anchorCtr="1" /><ns1:lstStyle /><ns1:p><ns1:pPr><ns1:defRPr sz="900" b="0" i="0" u="none" strike="noStrike" '
                'kern="1200" baseline="0"><ns1:solidFill><ns1:schemeClr val="tx1"><ns1:lumMod val="65000" /><ns1:lumOff val="35000" '
                '/></ns1:schemeClr></ns1:solidFill><ns1:latin typeface="+mn-lt" /><ns1:ea typeface="+mn-ea" /><ns1:cs typeface="+mn-cs" '
                '/></ns1:defRPr></ns1:pPr><ns1:endParaRPr lang="es-ES" /></ns1:p></ns0:txPr><ns0:crossAx val="717452799" /><ns0:crosses val="autoZero" '
                '/><ns0:crossBetween val="between" /></ns0:valAx>'],
  'value_ref': "'Gráfico RE 3.5'!$C$5:$F$5",
  'category_ref': "'Gráfico RE 3.5'!$C$4:$F$4",
  'name_ref': None,
  'physical_value_cells': ['C5', 'D5', 'E5', 'F5'],
  'physical_category_cells': ['C4', 'D4', 'E4', 'F4'],
  'scope': 'published_quantity_as_source_labelled'},
 {'book': '2026',
  'block': 'Gráfico RE_2.4.',
  'sid': 'chart17_s3',
  'headers': [('B3', 'GRÁFICO RE_2.4. BASE MEDIA ANUAL DE COTIZACIÓN DE INMIGRANTES (MILES DE EUROS)'),
              ('C6', 'Año 1'),
              ('D6', 'Año 2'),
              ('E6', 'Año 3'),
              ('B7', 'Base'),
              ('B8', 'Rango perfiles demográficos'),
              ('B9', 'Media ponderada')],
  'keys': [(1837, 'cache[0]', 'chart17_s3_p0'), (1838, 'cache[1]', 'chart17_s3_p1'), (1839, 'cache[2]', 'chart17_s3_p2')],
  'mode': 'normal',
  'unit': 'thousand_current_2024_euros',
  'multiplier': '1',
  'pdf': [47, ['BASE MEDIA ANUAL', '2024']],
  'label_cell': 'B9',
  'role': 'published_demographic_weighted_mean_annual_contribution_base',
  'scope': 'cross_section_first_registration_cohorts_not_longitudinal_panel',
  'part': 'xl/charts/chart17.xml',
  'refs': [],
  'node_xml': '<ns0:ser xmlns:ns0="http://schemas.openxmlformats.org/drawingml/2006/chart" xmlns:ns1="http://schemas.openxmlformats.org/drawingml/2006/main" '
              'xmlns:ns2="http://schemas.microsoft.com/office/drawing/2012/chart" xmlns:ns3="http://schemas.microsoft.com/office/drawing/2014/chart"><ns0:idx '
              'val="0" /><ns0:order val="2" /><ns0:tx><ns0:v>Media ponderada</ns0:v></ns0:tx><ns0:spPr><ns1:ln w="28575" cap="rnd"><ns1:solidFill><ns1:srgbClr '
              'val="E7E6E6"><ns1:lumMod val="50000" /></ns1:srgbClr></ns1:solidFill><ns1:round /></ns1:ln><ns1:effectLst /></ns0:spPr><ns0:marker><ns0:symbol '
              'val="none" /></ns0:marker><ns0:dLbls><ns0:numFmt formatCode="#,##0.0" sourceLinked="0" /><ns0:spPr><ns1:noFill /><ns1:ln><ns1:noFill '
              '/></ns1:ln><ns1:effectLst /></ns0:spPr><ns0:txPr><ns1:bodyPr rot="0" spcFirstLastPara="1" vertOverflow="ellipsis" vert="horz" wrap="square" '
              'lIns="38100" tIns="19050" rIns="38100" bIns="19050" anchor="ctr" anchorCtr="1"><ns1:spAutoFit /></ns1:bodyPr><ns1:lstStyle '
              '/><ns1:p><ns1:pPr><ns1:defRPr sz="1200" b="0" i="0" u="none" strike="noStrike" kern="1200" baseline="0"><ns1:solidFill><ns1:sysClr '
              'val="windowText" lastClr="000000" /></ns1:solidFill><ns1:latin typeface="Gill Sans MT" panose="020B0502020104020203" pitchFamily="34" '
              'charset="0" /><ns1:ea typeface="+mn-ea" /><ns1:cs typeface="+mn-cs" /></ns1:defRPr></ns1:pPr><ns1:endParaRPr lang="es-ES" '
              '/></ns1:p></ns0:txPr><ns0:dLblPos val="t" /><ns0:showLegendKey val="0" /><ns0:showVal val="1" /><ns0:showCatName val="0" /><ns0:showSerName '
              'val="0" /><ns0:showPercent val="0" /><ns0:showBubbleSize val="0" /><ns0:showLeaderLines val="0" /><ns0:extLst><ns0:ext '
              'uri="{CE6537A1-D6FC-4f65-9D91-7224C49458BB}"><ns2:showLeaderLines val="1" /><ns2:leaderLines><ns0:spPr><ns1:ln w="9525" cap="flat" cmpd="sng" '
              'algn="ctr"><ns1:solidFill><ns1:schemeClr val="tx1"><ns1:lumMod val="35000" /><ns1:lumOff val="65000" '
              '/></ns1:schemeClr></ns1:solidFill><ns1:round /></ns1:ln><ns1:effectLst '
              '/></ns0:spPr></ns2:leaderLines></ns0:ext></ns0:extLst></ns0:dLbls><ns0:cat><ns0:strLit><ns0:ptCount val="3" /><ns0:pt idx="0"><ns0:v>Año '
              '1</ns0:v></ns0:pt><ns0:pt idx="1"><ns0:v>Año 2</ns0:v></ns0:pt><ns0:pt idx="2"><ns0:v>Año '
              '3</ns0:v></ns0:pt></ns0:strLit></ns0:cat><ns0:val><ns0:numLit><ns0:formatCode>General</ns0:formatCode><ns0:ptCount val="3" /><ns0:pt '
              'idx="0"><ns0:v>8.5982939346039036</ns0:v></ns0:pt><ns0:pt idx="1"><ns0:v>14.022461435385697</ns0:v></ns0:pt><ns0:pt '
              'idx="2"><ns0:v>13.857210393179182</ns0:v></ns0:pt></ns0:numLit></ns0:val><ns0:smooth val="0" /><ns0:extLst><ns0:ext '
              'uri="{C3380CC4-5D6E-409C-BE32-E72D297353CC}"><ns3:uniqueId val="{00000002-B225-4A63-B236-62DF43D35CF2}" /></ns0:ext></ns0:extLst></ns0:ser>',
  'root_axes': ['<ns0:valAx xmlns:ns0="http://schemas.openxmlformats.org/drawingml/2006/chart" '
                'xmlns:ns1="http://schemas.openxmlformats.org/drawingml/2006/main"><ns0:axId val="498052207" /><ns0:scaling><ns0:orientation val="minMax" '
                '/></ns0:scaling><ns0:delete val="0" /><ns0:axPos val="l" /><ns0:title><ns0:tx><ns0:rich><ns1:bodyPr rot="-5400000" spcFirstLastPara="1" '
                'vertOverflow="ellipsis" vert="horz" wrap="square" anchor="ctr" anchorCtr="1" /><ns1:lstStyle /><ns1:p><ns1:pPr><ns1:defRPr sz="1000" b="0" '
                'i="0" u="none" strike="noStrike" kern="1200" baseline="0"><ns1:solidFill><ns1:sysClr val="windowText" lastClr="000000" '
                '/></ns1:solidFill><ns1:latin typeface="Gill Sans MT" panose="020B0502020104020203" pitchFamily="34" charset="0" /><ns1:ea typeface="+mn-ea" '
                '/><ns1:cs typeface="+mn-cs" /></ns1:defRPr></ns1:pPr><ns1:r><ns1:rPr lang="es-ES" /><ns1:t>Miles de '
                'Euros</ns1:t></ns1:r></ns1:p></ns0:rich></ns0:tx><ns0:overlay val="0" /><ns0:spPr><ns1:noFill /><ns1:ln><ns1:noFill /></ns1:ln><ns1:effectLst '
                '/></ns0:spPr><ns0:txPr><ns1:bodyPr rot="-5400000" spcFirstLastPara="1" vertOverflow="ellipsis" vert="horz" wrap="square" anchor="ctr" '
                'anchorCtr="1" /><ns1:lstStyle /><ns1:p><ns1:pPr><ns1:defRPr sz="1000" b="0" i="0" u="none" strike="noStrike" kern="1200" '
                'baseline="0"><ns1:solidFill><ns1:sysClr val="windowText" lastClr="000000" /></ns1:solidFill><ns1:latin typeface="Gill Sans MT" '
                'panose="020B0502020104020203" pitchFamily="34" charset="0" /><ns1:ea typeface="+mn-ea" /><ns1:cs typeface="+mn-cs" '
                '/></ns1:defRPr></ns1:pPr><ns1:endParaRPr lang="es-ES" /></ns1:p></ns0:txPr></ns0:title><ns0:numFmt formatCode="#,##0" sourceLinked="0" '
                '/><ns0:majorTickMark val="none" /><ns0:minorTickMark val="none" /><ns0:tickLblPos val="nextTo" /><ns0:spPr><ns1:noFill '
                '/><ns1:ln><ns1:solidFill><ns1:sysClr val="windowText" lastClr="000000" /></ns1:solidFill></ns1:ln><ns1:effectLst '
                '/></ns0:spPr><ns0:txPr><ns1:bodyPr rot="-60000000" spcFirstLastPara="1" vertOverflow="ellipsis" vert="horz" wrap="square" anchor="ctr" '
                'anchorCtr="1" /><ns1:lstStyle /><ns1:p><ns1:pPr><ns1:defRPr sz="1000" b="0" i="0" u="none" strike="noStrike" kern="1200" '
                'baseline="0"><ns1:solidFill><ns1:sysClr val="windowText" lastClr="000000" /></ns1:solidFill><ns1:latin typeface="Gill Sans MT" '
                'panose="020B0502020104020203" pitchFamily="34" charset="0" /><ns1:ea typeface="+mn-ea" /><ns1:cs typeface="+mn-cs" '
                '/></ns1:defRPr></ns1:pPr><ns1:endParaRPr lang="en-US" /></ns1:p></ns0:txPr><ns0:crossAx val="486499615" /><ns0:crosses val="autoZero" '
                '/><ns0:crossBetween val="between" /></ns0:valAx>'],
  'value_ref': None,
  'category_ref': None,
  'name_ref': None,
  'physical_value_cells': ['C9', 'D9', 'E9'],
  'physical_category_cells': ['C6', 'D6', 'E6']},
 {'book': '2025',
  'block': 'Gráfico 95',
  'sid': 'chart150_s1',
  'headers': [('B2', 'Gráfico 95. AJUSTE ESTRUCTURAL ACUMULADO 2025-2040 (PUNTOS PIB) BAJO DISTINTOS SUPUESTOS DE INCREMENTO DE GASTO EN DEFENSA'),
              ('C4', 'Ajuste sin ∆ GD'),
              ('D4', 'Aj. con ∆ 0,5 ptos.'),
              ('E4', 'Aj. con ∆ 1 pto.'),
              ('F4', 'Aj. con ∆ 1,5 ptos.')],
  'keys': [(40402, 'F5', 'chart150_s1_p3'), (40400, 'D5', 'chart150_s1_p1'), (40399, 'C5', 'chart150_s1_p0'), (40401, 'E5', 'chart150_s1_p2')],
  'mode': 'normal',
  'unit': 'percentage_points_gdp',
  'multiplier': '1',
  'pdf': [143, ['AJUSTE ESTRUCTURAL', 'PUNTOS PIB']],
  'label_cell': 'B2',
  'role': 'published_cumulative_structural_adjustment_defence_scenario',
  'scope': '2025-2040_as_labelled',
  'part': 'xl/charts/chart150.xml',
  'refs': ["'Gráfico 95'!$C$4:$F$4", "'Gráfico 95'!$C$5:$F$5"],
  'node_xml': '<ns0:ser xmlns:ns0="http://schemas.openxmlformats.org/drawingml/2006/chart" xmlns:ns1="http://schemas.openxmlformats.org/drawingml/2006/main" '
              'xmlns:ns2="http://schemas.microsoft.com/office/drawing/2014/chart" xmlns:ns3="http://schemas.microsoft.com/office/drawing/2012/chart"><ns0:idx '
              'val="0" /><ns0:order val="0" /><ns0:spPr><ns1:solidFill><ns1:srgbClr val="83082A" /></ns1:solidFill><ns1:ln w="25400"><ns1:noFill '
              '/></ns1:ln><ns1:effectLst /></ns0:spPr><ns0:invertIfNegative val="0" /><ns0:dPt><ns0:idx val="1" /><ns0:invertIfNegative val="0" '
              '/><ns0:bubble3D val="0" /><ns0:spPr><ns1:solidFill><ns1:sysClr val="window" lastClr="FFFFFF"><ns1:lumMod val="75000" '
              '/></ns1:sysClr></ns1:solidFill><ns1:ln w="25400"><ns1:noFill /></ns1:ln><ns1:effectLst /></ns0:spPr><ns0:extLst><ns0:ext '
              'uri="{C3380CC4-5D6E-409C-BE32-E72D297353CC}"><ns2:uniqueId val="{00000001-8A16-4DAF-B703-D7D40942F479}" '
              '/></ns0:ext></ns0:extLst></ns0:dPt><ns0:dPt><ns0:idx val="2" /><ns0:invertIfNegative val="0" /><ns0:bubble3D val="0" '
              '/><ns0:spPr><ns1:solidFill><ns1:sysClr val="window" lastClr="FFFFFF"><ns1:lumMod val="50000" /></ns1:sysClr></ns1:solidFill><ns1:ln '
              'w="25400"><ns1:noFill /></ns1:ln><ns1:effectLst /></ns0:spPr><ns0:extLst><ns0:ext uri="{C3380CC4-5D6E-409C-BE32-E72D297353CC}"><ns2:uniqueId '
              'val="{00000003-8A16-4DAF-B703-D7D40942F479}" /></ns0:ext></ns0:extLst></ns0:dPt><ns0:dPt><ns0:idx val="3" /><ns0:invertIfNegative val="0" '
              '/><ns0:bubble3D val="0" /><ns0:spPr><ns1:solidFill><ns1:sysClr val="windowText" lastClr="000000"><ns1:lumMod val="95000" /><ns1:lumOff '
              'val="5000" /></ns1:sysClr></ns1:solidFill><ns1:ln w="25400"><ns1:noFill /></ns1:ln><ns1:effectLst /></ns0:spPr><ns0:extLst><ns0:ext '
              'uri="{C3380CC4-5D6E-409C-BE32-E72D297353CC}"><ns2:uniqueId val="{00000005-8A16-4DAF-B703-D7D40942F479}" '
              '/></ns0:ext></ns0:extLst></ns0:dPt><ns0:dLbls><ns0:spPr><ns1:noFill /><ns1:ln><ns1:noFill /></ns1:ln><ns1:effectLst '
              '/></ns0:spPr><ns0:txPr><ns1:bodyPr rot="0" spcFirstLastPara="1" vertOverflow="ellipsis" vert="horz" wrap="square" lIns="38100" tIns="19050" '
              'rIns="38100" bIns="19050" anchor="ctr" anchorCtr="1"><ns1:spAutoFit /></ns1:bodyPr><ns1:lstStyle /><ns1:p><ns1:pPr><ns1:defRPr sz="1050" b="0" '
              'i="0" u="none" strike="noStrike" kern="1200" baseline="0"><ns1:solidFill><ns1:schemeClr val="bg1" /></ns1:solidFill><ns1:latin typeface="Gill '
              'Sans MT" panose="020B0502020104020203" pitchFamily="34" charset="0" /><ns1:ea typeface="+mn-ea" /><ns1:cs typeface="+mn-cs" '
              '/></ns1:defRPr></ns1:pPr><ns1:endParaRPr lang="es-ES" /></ns1:p></ns0:txPr><ns0:dLblPos val="inEnd" /><ns0:showLegendKey val="0" /><ns0:showVal '
              'val="1" /><ns0:showCatName val="0" /><ns0:showSerName val="0" /><ns0:showPercent val="0" /><ns0:showBubbleSize val="0" /><ns0:showLeaderLines '
              'val="0" /><ns0:extLst><ns0:ext uri="{CE6537A1-D6FC-4f65-9D91-7224C49458BB}"><ns3:showLeaderLines val="1" /><ns3:leaderLines><ns0:spPr><ns1:ln '
              'w="9525" cap="flat" cmpd="sng" algn="ctr"><ns1:solidFill><ns1:schemeClr val="tx1"><ns1:lumMod val="35000" /><ns1:lumOff val="65000" '
              '/></ns1:schemeClr></ns1:solidFill><ns1:round /></ns1:ln><ns1:effectLst '
              "/></ns0:spPr></ns3:leaderLines></ns0:ext></ns0:extLst></ns0:dLbls><ns0:cat><ns0:strRef><ns0:f>'Gráfico "
              '95\'!$C$4:$F$4</ns0:f><ns0:strCache><ns0:ptCount val="4" /><ns0:pt idx="0"><ns0:v>Ajuste sin ∆ GD</ns0:v></ns0:pt><ns0:pt idx="1"><ns0:v>Aj. '
              'con ∆ 0,5 ptos.</ns0:v></ns0:pt><ns0:pt idx="2"><ns0:v>Aj. con ∆ 1 pto.</ns0:v></ns0:pt><ns0:pt idx="3"><ns0:v>Aj. con ∆ 1,5 '
              "ptos.</ns0:v></ns0:pt></ns0:strCache></ns0:strRef></ns0:cat><ns0:val><ns0:numRef><ns0:f>'Gráfico "
              '95\'!$C$5:$F$5</ns0:f><ns0:numCache><ns0:formatCode>#,##0.0</ns0:formatCode><ns0:ptCount val="4" /><ns0:pt '
              'idx="0"><ns0:v>3.16</ns0:v></ns0:pt><ns0:pt idx="1"><ns0:v>3.680000000000001</ns0:v></ns0:pt><ns0:pt '
              'idx="2"><ns0:v>4.2399999999999984</ns0:v></ns0:pt><ns0:pt '
              'idx="3"><ns0:v>4.7599999999999989</ns0:v></ns0:pt></ns0:numCache></ns0:numRef></ns0:val><ns0:extLst><ns0:ext '
              'uri="{C3380CC4-5D6E-409C-BE32-E72D297353CC}"><ns2:uniqueId val="{00000006-8A16-4DAF-B703-D7D40942F479}" /></ns0:ext></ns0:extLst></ns0:ser>',
  'root_axes': ['<ns0:valAx xmlns:ns0="http://schemas.openxmlformats.org/drawingml/2006/chart" '
                'xmlns:ns1="http://schemas.openxmlformats.org/drawingml/2006/main"><ns0:axId val="759906015" /><ns0:scaling><ns0:orientation val="minMax" '
                '/></ns0:scaling><ns0:delete val="0" /><ns0:axPos val="l" /><ns0:numFmt formatCode="#,##0.0" sourceLinked="0" /><ns0:majorTickMark val="none" '
                '/><ns0:minorTickMark val="none" /><ns0:tickLblPos val="nextTo" /><ns0:spPr><ns1:noFill /><ns1:ln><ns1:noFill /></ns1:ln><ns1:effectLst '
                '/></ns0:spPr><ns0:txPr><ns1:bodyPr rot="-60000000" spcFirstLastPara="1" vertOverflow="ellipsis" vert="horz" wrap="square" anchor="ctr" '
                'anchorCtr="1" /><ns1:lstStyle /><ns1:p><ns1:pPr><ns1:defRPr sz="1000" b="0" i="0" u="none" strike="noStrike" kern="1200" '
                'baseline="0"><ns1:solidFill><ns1:schemeClr val="tx1"><ns1:lumMod val="65000" /><ns1:lumOff val="35000" '
                '/></ns1:schemeClr></ns1:solidFill><ns1:latin typeface="Gill Sans MT" panose="020B0502020104020203" pitchFamily="34" charset="0" /><ns1:ea '
                'typeface="+mn-ea" /><ns1:cs typeface="+mn-cs" /></ns1:defRPr></ns1:pPr><ns1:endParaRPr lang="es-ES" /></ns1:p></ns0:txPr><ns0:crossAx '
                'val="759904351" /><ns0:crosses val="autoZero" /><ns0:crossBetween val="between" /></ns0:valAx>'],
  'value_ref': "'Gráfico 95'!$C$5:$F$5",
  'category_ref': "'Gráfico 95'!$C$4:$F$4",
  'name_ref': None,
  'physical_value_cells': ['C5', 'D5', 'E5', 'F5'],
  'physical_category_cells': ['C4', 'D4', 'E4', 'F4']},
 {'book': '2026',
  'block': 'Gráfico 30',
  'sid': 'chart38_s3',
  'headers': [('B3', 'GRÁFICO 30. DIFERENCIA CON OPINIÓN DE 2025 EN LA RATIO DE DEUDA EN 2050 (P.P)'),
              ('B7', 'Diferencia de la ratio a en 2050'),
              ('B8', 'Mejor punto de partida 2025'),
              ('B9', 'Gasto intereses'),
              ('B10', 'Crecimiento Real'),
              ('B11', 'Inflación'),
              ('B12', 'Déficit primario'),
              ('B13', 'Aj. stock flujo')],
  'keys': [(5079, 'C10', 'chart38_s3_p0')],
  'mode': 'normal',
  'unit': 'percentage_points_gdp',
  'multiplier': '1',
  'pdf': [70, ['DIFERENCIA', '2050']],
  'role': 'published_debt_ratio_difference_component',
  'scope': 'difference_vs_Opinion2025_at_2050',
  'horizon': '2050',
  'part': 'xl/charts/chart38.xml',
  'refs': ["'Gráfico 30'!$B$10", "'Gráfico 30'!$C$10"],
  'node_xml': '<ns0:ser xmlns:ns0="http://schemas.openxmlformats.org/drawingml/2006/chart" xmlns:ns1="http://schemas.openxmlformats.org/drawingml/2006/main" '
              'xmlns:ns2="http://schemas.microsoft.com/office/drawing/2012/chart" xmlns:ns3="http://schemas.microsoft.com/office/drawing/2014/chart"><ns0:idx '
              'val="3" /><ns0:order val="3" /><ns0:tx><ns0:strRef><ns0:f>\'Gráfico 30\'!$B$10</ns0:f><ns0:strCache><ns0:ptCount val="1" /><ns0:pt '
              'idx="0"><ns0:v>Crecimiento Real</ns0:v></ns0:pt></ns0:strCache></ns0:strRef></ns0:tx><ns0:spPr><ns1:solidFill><ns1:schemeClr '
              'val="bg1"><ns1:lumMod val="65000" /></ns1:schemeClr></ns1:solidFill><ns1:ln w="25400"><ns1:noFill /></ns1:ln><ns1:effectLst '
              '/></ns0:spPr><ns0:invertIfNegative val="0" /><ns0:dLbls><ns0:spPr><ns1:noFill /><ns1:ln><ns1:noFill /></ns1:ln><ns1:effectLst '
              '/></ns0:spPr><ns0:txPr><ns1:bodyPr rot="0" spcFirstLastPara="1" vertOverflow="ellipsis" vert="horz" wrap="square" lIns="38100" tIns="19050" '
              'rIns="38100" bIns="19050" anchor="ctr" anchorCtr="1"><ns1:spAutoFit /></ns1:bodyPr><ns1:lstStyle /><ns1:p><ns1:pPr><ns1:defRPr sz="1050" b="0" '
              'i="0" u="none" strike="noStrike" kern="1200" baseline="0"><ns1:solidFill><ns1:schemeClr val="tx1"><ns1:lumMod val="75000" /><ns1:lumOff '
              'val="25000" /></ns1:schemeClr></ns1:solidFill><ns1:latin typeface="Gill Sans MT" panose="020B0502020104020203" pitchFamily="34" charset="0" '
              '/><ns1:ea typeface="+mn-ea" /><ns1:cs typeface="+mn-cs" /></ns1:defRPr></ns1:pPr><ns1:endParaRPr lang="es-ES" '
              '/></ns1:p></ns0:txPr><ns0:showLegendKey val="0" /><ns0:showVal val="1" /><ns0:showCatName val="0" /><ns0:showSerName val="0" /><ns0:showPercent '
              'val="0" /><ns0:showBubbleSize val="0" /><ns0:showLeaderLines val="0" /><ns0:extLst><ns0:ext '
              'uri="{CE6537A1-D6FC-4f65-9D91-7224C49458BB}"><ns2:showLeaderLines val="1" /><ns2:leaderLines><ns0:spPr><ns1:ln w="9525" cap="flat" cmpd="sng" '
              'algn="ctr"><ns1:solidFill><ns1:schemeClr val="tx1"><ns1:lumMod val="35000" /><ns1:lumOff val="65000" '
              '/></ns1:schemeClr></ns1:solidFill><ns1:round /></ns1:ln><ns1:effectLst '
              "/></ns0:spPr></ns2:leaderLines></ns0:ext></ns0:extLst></ns0:dLbls><ns0:val><ns0:numRef><ns0:f>'Gráfico "
              '30\'!$C$10</ns0:f><ns0:numCache><ns0:formatCode>#,##0.0</ns0:formatCode><ns0:ptCount val="1" /><ns0:pt '
              'idx="0"><ns0:v>1.0117904247063159</ns0:v></ns0:pt></ns0:numCache></ns0:numRef></ns0:val><ns0:extLst><ns0:ext '
              'uri="{C3380CC4-5D6E-409C-BE32-E72D297353CC}"><ns3:uniqueId val="{00000002-722B-482F-96C6-FF4AFB22CA97}" /></ns0:ext></ns0:extLst></ns0:ser>',
  'root_axes': ['<ns0:valAx xmlns:ns0="http://schemas.openxmlformats.org/drawingml/2006/chart" '
                'xmlns:ns1="http://schemas.openxmlformats.org/drawingml/2006/main"><ns0:axId val="180534160" /><ns0:scaling><ns0:orientation val="minMax" '
                '/></ns0:scaling><ns0:delete val="0" /><ns0:axPos val="l" /><ns0:numFmt formatCode="0" sourceLinked="0" /><ns0:majorTickMark val="none" '
                '/><ns0:minorTickMark val="none" /><ns0:tickLblPos val="nextTo" /><ns0:spPr><ns1:noFill /><ns1:ln><ns1:noFill /></ns1:ln><ns1:effectLst '
                '/></ns0:spPr><ns0:txPr><ns1:bodyPr rot="-60000000" spcFirstLastPara="1" vertOverflow="ellipsis" vert="horz" wrap="square" anchor="ctr" '
                'anchorCtr="1" /><ns1:lstStyle /><ns1:p><ns1:pPr><ns1:defRPr sz="1000" b="0" i="0" u="none" strike="noStrike" kern="1200" '
                'baseline="0"><ns1:solidFill><ns1:schemeClr val="tx1"><ns1:lumMod val="65000" /><ns1:lumOff val="35000" '
                '/></ns1:schemeClr></ns1:solidFill><ns1:latin typeface="Gill Sans MT" panose="020B0502020104020203" pitchFamily="34" charset="0" /><ns1:ea '
                'typeface="+mn-ea" /><ns1:cs typeface="+mn-cs" /></ns1:defRPr></ns1:pPr><ns1:endParaRPr lang="es-ES" /></ns1:p></ns0:txPr><ns0:crossAx '
                'val="180525840" /><ns0:crosses val="autoZero" /><ns0:crossBetween val="between" /></ns0:valAx>'],
  'value_ref': "'Gráfico 30'!$C$10",
  'category_ref': None,
  'name_ref': "'Gráfico 30'!$B$10",
  'physical_value_cells': ['C10'],
  'physical_category_cells': []},
 {'book': '2025',
  'block': 'Gráfico 64',
  'sid': 'chart113_s3',
  'headers': [('B2', 'Gráfico 64. CONTRIBUCIONES A LA VARIACIÓN DE LA DEUDA DE ESPAÑA'),
              ('C4', 'Variación Deuda'),
              ('D4', 'Diferencial interés-crecimiento'),
              ('E4', 'Saldo primario + ajuste stock flujo'),
              ('B5', '2008-2014'),
              ('B6', '2015-2024')],
  'keys': [(20843, 'C5', 'chart113_s3_p0')],
  'mode': 'debt_decomposition',
  'unit': 'percentage_points_gdp',
  'multiplier': '100',
  'pdf': [128, ['65 puntos']],
  'part': 'xl/charts/chart113.xml',
  'refs': ["'Gráfico 64'!$C$4", "'Gráfico 64'!$B$5:$B$6", "'Gráfico 64'!$C$5:$C$6"],
  'node_xml': '<ns0:ser xmlns:ns0="http://schemas.openxmlformats.org/drawingml/2006/chart" xmlns:ns1="http://schemas.openxmlformats.org/drawingml/2006/main" '
              'xmlns:ns2="http://schemas.microsoft.com/office/drawing/2012/chart" xmlns:ns3="http://schemas.microsoft.com/office/drawing/2014/chart"><ns0:idx '
              'val="0" /><ns0:order val="0" /><ns0:tx><ns0:strRef><ns0:f>\'Gráfico 64\'!$C$4</ns0:f><ns0:strCache><ns0:ptCount val="1" /><ns0:pt '
              'idx="0"><ns0:v>Variación Deuda</ns0:v></ns0:pt></ns0:strCache></ns0:strRef></ns0:tx><ns0:spPr><ns1:ln w="25400" cap="rnd"><ns1:noFill '
              '/><ns1:round /></ns1:ln><ns1:effectLst /></ns0:spPr><ns0:marker><ns0:symbol val="circle" /><ns0:size val="10" '
              '/><ns0:spPr><ns1:solidFill><ns1:schemeClr val="accent6"><ns1:lumMod val="20000" /><ns1:lumOff val="80000" '
              '/></ns1:schemeClr></ns1:solidFill><ns1:ln w="12700"><ns1:solidFill><ns1:schemeClr val="tx1"><ns1:lumMod val="50000" /><ns1:lumOff val="50000" '
              '/></ns1:schemeClr></ns1:solidFill></ns1:ln><ns1:effectLst /></ns0:spPr></ns0:marker><ns0:dLbls><ns0:dLbl><ns0:idx val="0" '
              '/><ns0:layout><ns0:manualLayout><ns0:x val="-0.17152631650031741" /><ns0:y val="-5.7205720572057209E-2" '
              '/></ns0:manualLayout></ns0:layout><ns0:spPr><ns1:noFill /><ns1:ln><ns1:noFill /></ns1:ln><ns1:effectLst /></ns0:spPr><ns0:txPr><ns1:bodyPr '
              'rot="0" spcFirstLastPara="1" vertOverflow="ellipsis" vert="horz" wrap="square" lIns="38100" tIns="19050" rIns="38100" bIns="19050" anchor="ctr" '
              'anchorCtr="1"><ns1:noAutofit /></ns1:bodyPr><ns1:lstStyle /><ns1:p><ns1:pPr><ns1:defRPr sz="900" b="1" i="0" u="none" strike="noStrike" '
              'kern="1200" baseline="0"><ns1:solidFill><ns1:schemeClr val="tx1"><ns1:lumMod val="75000" /><ns1:lumOff val="25000" '
              '/></ns1:schemeClr></ns1:solidFill><ns1:latin typeface="Gill Sans MT" panose="020B0502020104020203" pitchFamily="34" charset="0" /><ns1:ea '
              'typeface="+mn-ea" /><ns1:cs typeface="+mn-cs" /></ns1:defRPr></ns1:pPr><ns1:endParaRPr lang="es-ES" /></ns1:p></ns0:txPr><ns0:dLblPos val="r" '
              '/><ns0:showLegendKey val="0" /><ns0:showVal val="1" /><ns0:showCatName val="0" /><ns0:showSerName val="0" /><ns0:showPercent val="0" '
              '/><ns0:showBubbleSize val="0" /><ns0:extLst><ns0:ext uri="{CE6537A1-D6FC-4f65-9D91-7224C49458BB}"><ns2:layout><ns0:manualLayout><ns0:w '
              'val="0.23293785190012714" /><ns0:h val="8.3008356545961004E-2" /></ns0:manualLayout></ns2:layout></ns0:ext><ns0:ext '
              'uri="{C3380CC4-5D6E-409C-BE32-E72D297353CC}"><ns3:uniqueId val="{00000003-CBBE-4822-B6E8-8E9F5D18B80C}" '
              '/></ns0:ext></ns0:extLst></ns0:dLbl><ns0:spPr><ns1:noFill /><ns1:ln><ns1:noFill /></ns1:ln><ns1:effectLst /></ns0:spPr><ns0:txPr><ns1:bodyPr '
              'rot="0" spcFirstLastPara="1" vertOverflow="ellipsis" vert="horz" wrap="square" lIns="38100" tIns="19050" rIns="38100" bIns="19050" anchor="ctr" '
              'anchorCtr="1"><ns1:spAutoFit /></ns1:bodyPr><ns1:lstStyle /><ns1:p><ns1:pPr><ns1:defRPr sz="900" b="1" i="0" u="none" strike="noStrike" '
              'kern="1200" baseline="0"><ns1:solidFill><ns1:schemeClr val="tx1"><ns1:lumMod val="75000" /><ns1:lumOff val="25000" '
              '/></ns1:schemeClr></ns1:solidFill><ns1:latin typeface="Gill Sans MT" panose="020B0502020104020203" pitchFamily="34" charset="0" /><ns1:ea '
              'typeface="+mn-ea" /><ns1:cs typeface="+mn-cs" /></ns1:defRPr></ns1:pPr><ns1:endParaRPr lang="es-ES" /></ns1:p></ns0:txPr><ns0:dLblPos val="r" '
              '/><ns0:showLegendKey val="0" /><ns0:showVal val="1" /><ns0:showCatName val="0" /><ns0:showSerName val="0" /><ns0:showPercent val="0" '
              '/><ns0:showBubbleSize val="0" /><ns0:showLeaderLines val="0" /><ns0:extLst><ns0:ext '
              'uri="{CE6537A1-D6FC-4f65-9D91-7224C49458BB}"><ns2:showLeaderLines val="1" /><ns2:leaderLines><ns0:spPr><ns1:ln w="9525" cap="flat" cmpd="sng" '
              'algn="ctr"><ns1:solidFill><ns1:schemeClr val="tx1"><ns1:lumMod val="35000" /><ns1:lumOff val="65000" '
              '/></ns1:schemeClr></ns1:solidFill><ns1:round /></ns1:ln><ns1:effectLst '
              "/></ns0:spPr></ns2:leaderLines></ns0:ext></ns0:extLst></ns0:dLbls><ns0:cat><ns0:strRef><ns0:f>'Gráfico "
              '64\'!$B$5:$B$6</ns0:f><ns0:strCache><ns0:ptCount val="2" /><ns0:pt idx="0"><ns0:v>2008-2014</ns0:v></ns0:pt><ns0:pt '
              'idx="1"><ns0:v>2015-2024</ns0:v></ns0:pt></ns0:strCache></ns0:strRef></ns0:cat><ns0:val><ns0:numRef><ns0:f>\'Gráfico '
              '64\'!$C$5:$C$6</ns0:f><ns0:numCache><ns0:formatCode>0.0%</ns0:formatCode><ns0:ptCount val="2" /><ns0:pt '
              'idx="0"><ns0:v>0.6542932091652941</ns0:v></ns0:pt><ns0:pt '
              'idx="1"><ns0:v>-2.6670561451890083E-2</ns0:v></ns0:pt></ns0:numCache></ns0:numRef></ns0:val><ns0:smooth val="0" /><ns0:extLst><ns0:ext '
              'uri="{C3380CC4-5D6E-409C-BE32-E72D297353CC}"><ns3:uniqueId val="{00000004-CBBE-4822-B6E8-8E9F5D18B80C}" /></ns0:ext></ns0:extLst></ns0:ser>',
  'root_axes': ['<ns0:valAx xmlns:ns0="http://schemas.openxmlformats.org/drawingml/2006/chart" '
                'xmlns:ns1="http://schemas.openxmlformats.org/drawingml/2006/main"><ns0:axId val="820971632" /><ns0:scaling><ns0:orientation val="minMax" '
                '/><ns0:max val="0.70000000000000007" /></ns0:scaling><ns0:delete val="0" /><ns0:axPos val="l" /><ns0:numFmt formatCode="0%" sourceLinked="0" '
                '/><ns0:majorTickMark val="none" /><ns0:minorTickMark val="none" /><ns0:tickLblPos val="nextTo" /><ns0:spPr><ns1:noFill /><ns1:ln><ns1:noFill '
                '/></ns1:ln><ns1:effectLst /></ns0:spPr><ns0:txPr><ns1:bodyPr rot="-60000000" spcFirstLastPara="1" vertOverflow="ellipsis" vert="horz" '
                'wrap="square" anchor="ctr" anchorCtr="1" /><ns1:lstStyle /><ns1:p><ns1:pPr><ns1:defRPr sz="800" b="0" i="0" u="none" strike="noStrike" '
                'kern="1200" baseline="0"><ns1:solidFill><ns1:schemeClr val="tx1"><ns1:lumMod val="65000" /><ns1:lumOff val="35000" '
                '/></ns1:schemeClr></ns1:solidFill><ns1:latin typeface="Gill Sans MT" panose="020B0502020104020203" pitchFamily="34" charset="0" /><ns1:ea '
                'typeface="+mn-ea" /><ns1:cs typeface="+mn-cs" /></ns1:defRPr></ns1:pPr><ns1:endParaRPr lang="es-ES" /></ns1:p></ns0:txPr><ns0:crossAx '
                'val="820970976" /><ns0:crosses val="autoZero" /><ns0:crossBetween val="between" /><ns0:majorUnit val="0.1" /></ns0:valAx>'],
  'value_ref': "'Gráfico 64'!$C$5:$C$6",
  'category_ref': "'Gráfico 64'!$B$5:$B$6",
  'name_ref': "'Gráfico 64'!$C$4",
  'physical_value_cells': ['C5', 'C6'],
  'physical_category_cells': ['B5', 'B6'],
  'scope': 'published_quantity_as_source_labelled'},
 {'book': '2026',
  'block': 'Gráfico 30',
  'sid': 'chart38_s6',
  'headers': [('B3', 'GRÁFICO 30. DIFERENCIA CON OPINIÓN DE 2025 EN LA RATIO DE DEUDA EN 2050 (P.P)'),
              ('B7', 'Diferencia de la ratio a en 2050'),
              ('B8', 'Mejor punto de partida 2025'),
              ('B9', 'Gasto intereses'),
              ('B10', 'Crecimiento Real'),
              ('B11', 'Inflación'),
              ('B12', 'Déficit primario'),
              ('B13', 'Aj. stock flujo')],
  'keys': [(5082, 'C13', 'chart38_s6_p0')],
  'mode': 'normal',
  'unit': 'percentage_points_gdp',
  'multiplier': '1',
  'pdf': [70, ['DIFERENCIA', '2050']],
  'role': 'published_debt_ratio_difference_component',
  'scope': 'difference_vs_Opinion2025_at_2050',
  'horizon': '2050',
  'part': 'xl/charts/chart38.xml',
  'refs': ["'Gráfico 30'!$B$13", "'Gráfico 30'!$C$13"],
  'node_xml': '<ns0:ser xmlns:ns0="http://schemas.openxmlformats.org/drawingml/2006/chart" xmlns:ns1="http://schemas.openxmlformats.org/drawingml/2006/main" '
              'xmlns:ns2="http://schemas.microsoft.com/office/drawing/2012/chart" xmlns:ns3="http://schemas.microsoft.com/office/drawing/2014/chart"><ns0:idx '
              'val="6" /><ns0:order val="6" /><ns0:tx><ns0:strRef><ns0:f>\'Gráfico 30\'!$B$13</ns0:f><ns0:strCache><ns0:ptCount val="1" /><ns0:pt '
              'idx="0"><ns0:v>Aj. stock flujo</ns0:v></ns0:pt></ns0:strCache></ns0:strRef></ns0:tx><ns0:spPr><ns1:pattFill '
              'prst="pct20"><ns1:fgClr><ns1:schemeClr val="accent1" /></ns1:fgClr><ns1:bgClr><ns1:schemeClr val="bg1" /></ns1:bgClr></ns1:pattFill><ns1:ln '
              'w="25400"><ns1:noFill /></ns1:ln><ns1:effectLst /></ns0:spPr><ns0:invertIfNegative val="0" /><ns0:dLbls><ns0:spPr><ns1:noFill '
              '/><ns1:ln><ns1:noFill /></ns1:ln><ns1:effectLst /></ns0:spPr><ns0:txPr><ns1:bodyPr rot="0" spcFirstLastPara="1" vertOverflow="ellipsis" '
              'vert="horz" wrap="square" lIns="38100" tIns="19050" rIns="38100" bIns="19050" anchor="ctr" anchorCtr="1"><ns1:spAutoFit '
              '/></ns1:bodyPr><ns1:lstStyle /><ns1:p><ns1:pPr><ns1:defRPr sz="1050" b="0" i="0" u="none" strike="noStrike" kern="1200" '
              'baseline="0"><ns1:solidFill><ns1:schemeClr val="tx1"><ns1:lumMod val="75000" /><ns1:lumOff val="25000" '
              '/></ns1:schemeClr></ns1:solidFill><ns1:latin typeface="Gill Sans MT" panose="020B0502020104020203" pitchFamily="34" charset="0" /><ns1:ea '
              'typeface="+mn-ea" /><ns1:cs typeface="+mn-cs" /></ns1:defRPr></ns1:pPr><ns1:endParaRPr lang="es-ES" /></ns1:p></ns0:txPr><ns0:showLegendKey '
              'val="0" /><ns0:showVal val="1" /><ns0:showCatName val="0" /><ns0:showSerName val="0" /><ns0:showPercent val="0" /><ns0:showBubbleSize val="0" '
              '/><ns0:showLeaderLines val="0" /><ns0:extLst><ns0:ext uri="{CE6537A1-D6FC-4f65-9D91-7224C49458BB}"><ns2:showLeaderLines val="1" '
              '/><ns2:leaderLines><ns0:spPr><ns1:ln w="9525" cap="flat" cmpd="sng" algn="ctr"><ns1:solidFill><ns1:schemeClr val="tx1"><ns1:lumMod val="35000" '
              '/><ns1:lumOff val="65000" /></ns1:schemeClr></ns1:solidFill><ns1:round /></ns1:ln><ns1:effectLst '
              "/></ns0:spPr></ns2:leaderLines></ns0:ext></ns0:extLst></ns0:dLbls><ns0:val><ns0:numRef><ns0:f>'Gráfico "
              '30\'!$C$13</ns0:f><ns0:numCache><ns0:formatCode>#,##0.0</ns0:formatCode><ns0:ptCount val="1" /><ns0:pt '
              'idx="0"><ns0:v>1.8150755277700541</ns0:v></ns0:pt></ns0:numCache></ns0:numRef></ns0:val><ns0:extLst><ns0:ext '
              'uri="{C3380CC4-5D6E-409C-BE32-E72D297353CC}"><ns3:uniqueId val="{00000006-722B-482F-96C6-FF4AFB22CA97}" /></ns0:ext></ns0:extLst></ns0:ser>',
  'root_axes': ['<ns0:valAx xmlns:ns0="http://schemas.openxmlformats.org/drawingml/2006/chart" '
                'xmlns:ns1="http://schemas.openxmlformats.org/drawingml/2006/main"><ns0:axId val="180534160" /><ns0:scaling><ns0:orientation val="minMax" '
                '/></ns0:scaling><ns0:delete val="0" /><ns0:axPos val="l" /><ns0:numFmt formatCode="0" sourceLinked="0" /><ns0:majorTickMark val="none" '
                '/><ns0:minorTickMark val="none" /><ns0:tickLblPos val="nextTo" /><ns0:spPr><ns1:noFill /><ns1:ln><ns1:noFill /></ns1:ln><ns1:effectLst '
                '/></ns0:spPr><ns0:txPr><ns1:bodyPr rot="-60000000" spcFirstLastPara="1" vertOverflow="ellipsis" vert="horz" wrap="square" anchor="ctr" '
                'anchorCtr="1" /><ns1:lstStyle /><ns1:p><ns1:pPr><ns1:defRPr sz="1000" b="0" i="0" u="none" strike="noStrike" kern="1200" '
                'baseline="0"><ns1:solidFill><ns1:schemeClr val="tx1"><ns1:lumMod val="65000" /><ns1:lumOff val="35000" '
                '/></ns1:schemeClr></ns1:solidFill><ns1:latin typeface="Gill Sans MT" panose="020B0502020104020203" pitchFamily="34" charset="0" /><ns1:ea '
                'typeface="+mn-ea" /><ns1:cs typeface="+mn-cs" /></ns1:defRPr></ns1:pPr><ns1:endParaRPr lang="es-ES" /></ns1:p></ns0:txPr><ns0:crossAx '
                'val="180525840" /><ns0:crosses val="autoZero" /><ns0:crossBetween val="between" /></ns0:valAx>'],
  'value_ref': "'Gráfico 30'!$C$13",
  'category_ref': None,
  'name_ref': "'Gráfico 30'!$B$13",
  'physical_value_cells': ['C13'],
  'physical_category_cells': []},
 {'book': '2025',
  'block': 'Gráfico 64',
  'sid': 'chart113_s2',
  'headers': [('B2', 'Gráfico 64. CONTRIBUCIONES A LA VARIACIÓN DE LA DEUDA DE ESPAÑA'),
              ('C4', 'Variación Deuda'),
              ('D4', 'Diferencial interés-crecimiento'),
              ('E4', 'Saldo primario + ajuste stock flujo'),
              ('B5', '2008-2014'),
              ('B6', '2015-2024')],
  'keys': [(20841, 'E5', 'chart113_s2_p0')],
  'mode': 'debt_decomposition',
  'unit': 'percentage_points_gdp',
  'multiplier': '100',
  'pdf': [128, ['65 puntos']],
  'part': 'xl/charts/chart113.xml',
  'refs': ["'Gráfico 64'!$E$4", "'Gráfico 64'!$B$5:$B$6", "'Gráfico 64'!$E$5:$E$6"],
  'node_xml': '<ns0:ser xmlns:ns0="http://schemas.openxmlformats.org/drawingml/2006/chart" xmlns:ns1="http://schemas.openxmlformats.org/drawingml/2006/main" '
              'xmlns:ns2="http://schemas.microsoft.com/office/drawing/2012/chart" xmlns:ns3="http://schemas.microsoft.com/office/drawing/2014/chart"><ns0:idx '
              'val="1" /><ns0:order val="2" /><ns0:tx><ns0:strRef><ns0:f>\'Gráfico 64\'!$E$4</ns0:f><ns0:strCache><ns0:ptCount val="1" /><ns0:pt '
              'idx="0"><ns0:v>Saldo primario + ajuste stock flujo</ns0:v></ns0:pt></ns0:strCache></ns0:strRef></ns0:tx><ns0:spPr><ns1:solidFill><ns1:srgbClr '
              'val="83082A" /></ns1:solidFill><ns1:ln><ns1:noFill /></ns1:ln><ns1:effectLst /></ns0:spPr><ns0:invertIfNegative val="0" '
              '/><ns0:dLbls><ns0:spPr><ns1:noFill /><ns1:ln><ns1:noFill /></ns1:ln><ns1:effectLst /></ns0:spPr><ns0:txPr><ns1:bodyPr rot="0" '
              'spcFirstLastPara="1" vertOverflow="ellipsis" vert="horz" wrap="square" lIns="38100" tIns="19050" rIns="38100" bIns="19050" anchor="ctr" '
              'anchorCtr="1"><ns1:spAutoFit /></ns1:bodyPr><ns1:lstStyle /><ns1:p><ns1:pPr><ns1:defRPr sz="800" b="0" i="0" u="none" strike="noStrike" '
              'kern="1200" baseline="0"><ns1:solidFill><ns1:schemeClr val="bg1" /></ns1:solidFill><ns1:latin typeface="Gill Sans MT" '
              'panose="020B0502020104020203" pitchFamily="34" charset="0" /><ns1:ea typeface="+mn-ea" /><ns1:cs typeface="+mn-cs" '
              '/></ns1:defRPr></ns1:pPr><ns1:endParaRPr lang="es-ES" /></ns1:p></ns0:txPr><ns0:dLblPos val="ctr" /><ns0:showLegendKey val="0" /><ns0:showVal '
              'val="1" /><ns0:showCatName val="0" /><ns0:showSerName val="0" /><ns0:showPercent val="0" /><ns0:showBubbleSize val="0" /><ns0:showLeaderLines '
              'val="0" /><ns0:extLst><ns0:ext uri="{CE6537A1-D6FC-4f65-9D91-7224C49458BB}"><ns2:showLeaderLines val="1" /><ns2:leaderLines><ns0:spPr><ns1:ln '
              'w="9525" cap="flat" cmpd="sng" algn="ctr"><ns1:solidFill><ns1:schemeClr val="tx1"><ns1:lumMod val="35000" /><ns1:lumOff val="65000" '
              '/></ns1:schemeClr></ns1:solidFill><ns1:round /></ns1:ln><ns1:effectLst '
              "/></ns0:spPr></ns2:leaderLines></ns0:ext></ns0:extLst></ns0:dLbls><ns0:cat><ns0:strRef><ns0:f>'Gráfico "
              '64\'!$B$5:$B$6</ns0:f><ns0:strCache><ns0:ptCount val="2" /><ns0:pt idx="0"><ns0:v>2008-2014</ns0:v></ns0:pt><ns0:pt '
              'idx="1"><ns0:v>2015-2024</ns0:v></ns0:pt></ns0:strCache></ns0:strRef></ns0:cat><ns0:val><ns0:numRef><ns0:f>\'Gráfico '
              '64\'!$E$5:$E$6</ns0:f><ns0:numCache><ns0:formatCode>0.0%</ns0:formatCode><ns0:ptCount val="2" /><ns0:pt '
              'idx="0"><ns0:v>0.46259636171023027</ns0:v></ns0:pt><ns0:pt '
              'idx="1"><ns0:v>0.21282368061555432</ns0:v></ns0:pt></ns0:numCache></ns0:numRef></ns0:val><ns0:extLst><ns0:ext '
              'uri="{C3380CC4-5D6E-409C-BE32-E72D297353CC}"><ns3:uniqueId val="{00000002-CBBE-4822-B6E8-8E9F5D18B80C}" /></ns0:ext></ns0:extLst></ns0:ser>',
  'root_axes': ['<ns0:valAx xmlns:ns0="http://schemas.openxmlformats.org/drawingml/2006/chart" '
                'xmlns:ns1="http://schemas.openxmlformats.org/drawingml/2006/main"><ns0:axId val="820971632" /><ns0:scaling><ns0:orientation val="minMax" '
                '/><ns0:max val="0.70000000000000007" /></ns0:scaling><ns0:delete val="0" /><ns0:axPos val="l" /><ns0:numFmt formatCode="0%" sourceLinked="0" '
                '/><ns0:majorTickMark val="none" /><ns0:minorTickMark val="none" /><ns0:tickLblPos val="nextTo" /><ns0:spPr><ns1:noFill /><ns1:ln><ns1:noFill '
                '/></ns1:ln><ns1:effectLst /></ns0:spPr><ns0:txPr><ns1:bodyPr rot="-60000000" spcFirstLastPara="1" vertOverflow="ellipsis" vert="horz" '
                'wrap="square" anchor="ctr" anchorCtr="1" /><ns1:lstStyle /><ns1:p><ns1:pPr><ns1:defRPr sz="800" b="0" i="0" u="none" strike="noStrike" '
                'kern="1200" baseline="0"><ns1:solidFill><ns1:schemeClr val="tx1"><ns1:lumMod val="65000" /><ns1:lumOff val="35000" '
                '/></ns1:schemeClr></ns1:solidFill><ns1:latin typeface="Gill Sans MT" panose="020B0502020104020203" pitchFamily="34" charset="0" /><ns1:ea '
                'typeface="+mn-ea" /><ns1:cs typeface="+mn-cs" /></ns1:defRPr></ns1:pPr><ns1:endParaRPr lang="es-ES" /></ns1:p></ns0:txPr><ns0:crossAx '
                'val="820970976" /><ns0:crosses val="autoZero" /><ns0:crossBetween val="between" /><ns0:majorUnit val="0.1" /></ns0:valAx>'],
  'value_ref': "'Gráfico 64'!$E$5:$E$6",
  'category_ref': "'Gráfico 64'!$B$5:$B$6",
  'name_ref': "'Gráfico 64'!$E$4",
  'physical_value_cells': ['E5', 'E6'],
  'physical_category_cells': ['B5', 'B6'],
  'scope': 'published_quantity_as_source_labelled'},
 {'book': '2025',
  'block': 'Gráfico 64',
  'sid': 'chart113_s1',
  'headers': [('B2', 'Gráfico 64. CONTRIBUCIONES A LA VARIACIÓN DE LA DEUDA DE ESPAÑA'),
              ('C4', 'Variación Deuda'),
              ('D4', 'Diferencial interés-crecimiento'),
              ('E4', 'Saldo primario + ajuste stock flujo'),
              ('B5', '2008-2014'),
              ('B6', '2015-2024')],
  'keys': [(20839, 'D5', 'chart113_s1_p0')],
  'mode': 'debt_decomposition',
  'unit': 'percentage_points_gdp',
  'multiplier': '100',
  'pdf': [128, ['65 puntos']],
  'part': 'xl/charts/chart113.xml',
  'refs': ["'Gráfico 64'!$D$4", "'Gráfico 64'!$B$5:$B$6", "'Gráfico 64'!$D$5:$D$6"],
  'node_xml': '<ns0:ser xmlns:ns0="http://schemas.openxmlformats.org/drawingml/2006/chart" xmlns:ns1="http://schemas.openxmlformats.org/drawingml/2006/main" '
              'xmlns:ns2="http://schemas.microsoft.com/office/drawing/2014/chart" xmlns:ns3="http://schemas.microsoft.com/office/drawing/2012/chart"><ns0:idx '
              'val="2" /><ns0:order val="1" /><ns0:tx><ns0:strRef><ns0:f>\'Gráfico 64\'!$D$4</ns0:f><ns0:strCache><ns0:ptCount val="1" /><ns0:pt '
              'idx="0"><ns0:v>Diferencial interés-crecimiento</ns0:v></ns0:pt></ns0:strCache></ns0:strRef></ns0:tx><ns0:spPr><ns1:solidFill><ns1:srgbClr '
              'val="FFFFFF"><ns1:lumMod val="85000" /></ns1:srgbClr></ns1:solidFill><ns1:ln><ns1:noFill /></ns1:ln><ns1:effectLst '
              '/></ns0:spPr><ns0:invertIfNegative val="0" /><ns0:dLbls><ns0:dLbl><ns0:idx val="1" /><ns0:dLblPos val="inEnd" /><ns0:showLegendKey val="0" '
              '/><ns0:showVal val="1" /><ns0:showCatName val="0" /><ns0:showSerName val="0" /><ns0:showPercent val="0" /><ns0:showBubbleSize val="0" '
              '/><ns0:extLst><ns0:ext uri="{CE6537A1-D6FC-4f65-9D91-7224C49458BB}" /><ns0:ext uri="{C3380CC4-5D6E-409C-BE32-E72D297353CC}"><ns2:uniqueId '
              'val="{00000000-CBBE-4822-B6E8-8E9F5D18B80C}" /></ns0:ext></ns0:extLst></ns0:dLbl><ns0:spPr><ns1:noFill /><ns1:ln><ns1:noFill '
              '/></ns1:ln><ns1:effectLst /></ns0:spPr><ns0:txPr><ns1:bodyPr rot="0" spcFirstLastPara="1" vertOverflow="ellipsis" vert="horz" wrap="square" '
              'lIns="38100" tIns="19050" rIns="38100" bIns="19050" anchor="ctr" anchorCtr="1"><ns1:spAutoFit /></ns1:bodyPr><ns1:lstStyle '
              '/><ns1:p><ns1:pPr><ns1:defRPr sz="800" b="0" i="0" u="none" strike="noStrike" kern="1200" baseline="0"><ns1:solidFill><ns1:schemeClr '
              'val="tx1"><ns1:lumMod val="75000" /><ns1:lumOff val="25000" /></ns1:schemeClr></ns1:solidFill><ns1:latin typeface="Gill Sans MT" '
              'panose="020B0502020104020203" pitchFamily="34" charset="0" /><ns1:ea typeface="+mn-ea" /><ns1:cs typeface="+mn-cs" '
              '/></ns1:defRPr></ns1:pPr><ns1:endParaRPr lang="es-ES" /></ns1:p></ns0:txPr><ns0:dLblPos val="inBase" /><ns0:showLegendKey val="0" '
              '/><ns0:showVal val="1" /><ns0:showCatName val="0" /><ns0:showSerName val="0" /><ns0:showPercent val="0" /><ns0:showBubbleSize val="0" '
              '/><ns0:showLeaderLines val="0" /><ns0:extLst><ns0:ext uri="{CE6537A1-D6FC-4f65-9D91-7224C49458BB}"><ns3:showLeaderLines val="1" '
              '/><ns3:leaderLines><ns0:spPr><ns1:ln w="9525" cap="flat" cmpd="sng" algn="ctr"><ns1:solidFill><ns1:schemeClr val="tx1"><ns1:lumMod val="35000" '
              '/><ns1:lumOff val="65000" /></ns1:schemeClr></ns1:solidFill><ns1:round /></ns1:ln><ns1:effectLst '
              "/></ns0:spPr></ns3:leaderLines></ns0:ext></ns0:extLst></ns0:dLbls><ns0:cat><ns0:strRef><ns0:f>'Gráfico "
              '64\'!$B$5:$B$6</ns0:f><ns0:strCache><ns0:ptCount val="2" /><ns0:pt idx="0"><ns0:v>2008-2014</ns0:v></ns0:pt><ns0:pt '
              'idx="1"><ns0:v>2015-2024</ns0:v></ns0:pt></ns0:strCache></ns0:strRef></ns0:cat><ns0:val><ns0:numRef><ns0:f>\'Gráfico '
              '64\'!$D$5:$D$6</ns0:f><ns0:numCache><ns0:formatCode>0.0%</ns0:formatCode><ns0:ptCount val="2" /><ns0:pt '
              'idx="0"><ns0:v>0.19169684745506385</ns0:v></ns0:pt><ns0:pt '
              'idx="1"><ns0:v>-0.20757654378409629</ns0:v></ns0:pt></ns0:numCache></ns0:numRef></ns0:val><ns0:extLst><ns0:ext '
              'uri="{C3380CC4-5D6E-409C-BE32-E72D297353CC}"><ns2:uniqueId val="{00000001-CBBE-4822-B6E8-8E9F5D18B80C}" /></ns0:ext></ns0:extLst></ns0:ser>',
  'root_axes': ['<ns0:valAx xmlns:ns0="http://schemas.openxmlformats.org/drawingml/2006/chart" '
                'xmlns:ns1="http://schemas.openxmlformats.org/drawingml/2006/main"><ns0:axId val="820971632" /><ns0:scaling><ns0:orientation val="minMax" '
                '/><ns0:max val="0.70000000000000007" /></ns0:scaling><ns0:delete val="0" /><ns0:axPos val="l" /><ns0:numFmt formatCode="0%" sourceLinked="0" '
                '/><ns0:majorTickMark val="none" /><ns0:minorTickMark val="none" /><ns0:tickLblPos val="nextTo" /><ns0:spPr><ns1:noFill /><ns1:ln><ns1:noFill '
                '/></ns1:ln><ns1:effectLst /></ns0:spPr><ns0:txPr><ns1:bodyPr rot="-60000000" spcFirstLastPara="1" vertOverflow="ellipsis" vert="horz" '
                'wrap="square" anchor="ctr" anchorCtr="1" /><ns1:lstStyle /><ns1:p><ns1:pPr><ns1:defRPr sz="800" b="0" i="0" u="none" strike="noStrike" '
                'kern="1200" baseline="0"><ns1:solidFill><ns1:schemeClr val="tx1"><ns1:lumMod val="65000" /><ns1:lumOff val="35000" '
                '/></ns1:schemeClr></ns1:solidFill><ns1:latin typeface="Gill Sans MT" panose="020B0502020104020203" pitchFamily="34" charset="0" /><ns1:ea '
                'typeface="+mn-ea" /><ns1:cs typeface="+mn-cs" /></ns1:defRPr></ns1:pPr><ns1:endParaRPr lang="es-ES" /></ns1:p></ns0:txPr><ns0:crossAx '
                'val="820970976" /><ns0:crosses val="autoZero" /><ns0:crossBetween val="between" /><ns0:majorUnit val="0.1" /></ns0:valAx>'],
  'value_ref': "'Gráfico 64'!$D$5:$D$6",
  'category_ref': "'Gráfico 64'!$B$5:$B$6",
  'name_ref': "'Gráfico 64'!$D$4",
  'physical_value_cells': ['D5', 'D6'],
  'physical_category_cells': ['B5', 'B6'],
  'scope': 'published_quantity_as_source_labelled'},
 {'book': '2026',
  'block': 'Gráfico 30',
  'sid': 'chart38_s4',
  'headers': [('B3', 'GRÁFICO 30. DIFERENCIA CON OPINIÓN DE 2025 EN LA RATIO DE DEUDA EN 2050 (P.P)'),
              ('B7', 'Diferencia de la ratio a en 2050'),
              ('B8', 'Mejor punto de partida 2025'),
              ('B9', 'Gasto intereses'),
              ('B10', 'Crecimiento Real'),
              ('B11', 'Inflación'),
              ('B12', 'Déficit primario'),
              ('B13', 'Aj. stock flujo')],
  'keys': [(5080, 'C11', 'chart38_s4_p0')],
  'mode': 'normal',
  'unit': 'percentage_points_gdp',
  'multiplier': '1',
  'pdf': [70, ['DIFERENCIA', '2050']],
  'role': 'published_debt_ratio_difference_component',
  'scope': 'difference_vs_Opinion2025_at_2050',
  'horizon': '2050',
  'part': 'xl/charts/chart38.xml',
  'refs': ["'Gráfico 30'!$B$11", "'Gráfico 30'!$C$11"],
  'node_xml': '<ns0:ser xmlns:ns0="http://schemas.openxmlformats.org/drawingml/2006/chart" xmlns:ns1="http://schemas.openxmlformats.org/drawingml/2006/main" '
              'xmlns:ns2="http://schemas.microsoft.com/office/drawing/2012/chart" xmlns:ns3="http://schemas.microsoft.com/office/drawing/2014/chart"><ns0:idx '
              'val="4" /><ns0:order val="4" /><ns0:tx><ns0:strRef><ns0:f>\'Gráfico 30\'!$B$11</ns0:f><ns0:strCache><ns0:ptCount val="1" /><ns0:pt '
              'idx="0"><ns0:v>Inflación</ns0:v></ns0:pt></ns0:strCache></ns0:strRef></ns0:tx><ns0:spPr><ns1:solidFill><ns1:schemeClr val="bg1"><ns1:lumMod '
              'val="85000" /></ns1:schemeClr></ns1:solidFill><ns1:ln w="25400"><ns1:noFill /></ns1:ln><ns1:effectLst /></ns0:spPr><ns0:invertIfNegative '
              'val="0" /><ns0:dLbls><ns0:spPr><ns1:noFill /><ns1:ln><ns1:noFill /></ns1:ln><ns1:effectLst /></ns0:spPr><ns0:txPr><ns1:bodyPr rot="0" '
              'spcFirstLastPara="1" vertOverflow="ellipsis" vert="horz" wrap="square" lIns="38100" tIns="19050" rIns="38100" bIns="19050" anchor="ctr" '
              'anchorCtr="1"><ns1:spAutoFit /></ns1:bodyPr><ns1:lstStyle /><ns1:p><ns1:pPr><ns1:defRPr sz="1050" b="0" i="0" u="none" strike="noStrike" '
              'kern="1200" baseline="0"><ns1:solidFill><ns1:schemeClr val="bg1"><ns1:lumMod val="65000" /></ns1:schemeClr></ns1:solidFill><ns1:latin '
              'typeface="Gill Sans MT" panose="020B0502020104020203" pitchFamily="34" charset="0" /><ns1:ea typeface="+mn-ea" /><ns1:cs typeface="+mn-cs" '
              '/></ns1:defRPr></ns1:pPr><ns1:endParaRPr lang="es-ES" /></ns1:p></ns0:txPr><ns0:showLegendKey val="0" /><ns0:showVal val="1" /><ns0:showCatName '
              'val="0" /><ns0:showSerName val="0" /><ns0:showPercent val="0" /><ns0:showBubbleSize val="0" /><ns0:showLeaderLines val="0" '
              '/><ns0:extLst><ns0:ext uri="{CE6537A1-D6FC-4f65-9D91-7224C49458BB}"><ns2:showLeaderLines val="1" /><ns2:leaderLines><ns0:spPr><ns1:ln w="9525" '
              'cap="flat" cmpd="sng" algn="ctr"><ns1:solidFill><ns1:schemeClr val="tx1"><ns1:lumMod val="35000" /><ns1:lumOff val="65000" '
              '/></ns1:schemeClr></ns1:solidFill><ns1:round /></ns1:ln><ns1:effectLst '
              "/></ns0:spPr></ns2:leaderLines></ns0:ext></ns0:extLst></ns0:dLbls><ns0:val><ns0:numRef><ns0:f>'Gráfico "
              '30\'!$C$11</ns0:f><ns0:numCache><ns0:formatCode>#,##0.0</ns0:formatCode><ns0:ptCount val="1" /><ns0:pt '
              'idx="0"><ns0:v>1.1818039153335249</ns0:v></ns0:pt></ns0:numCache></ns0:numRef></ns0:val><ns0:extLst><ns0:ext '
              'uri="{C3380CC4-5D6E-409C-BE32-E72D297353CC}"><ns3:uniqueId val="{00000003-722B-482F-96C6-FF4AFB22CA97}" /></ns0:ext></ns0:extLst></ns0:ser>',
  'root_axes': ['<ns0:valAx xmlns:ns0="http://schemas.openxmlformats.org/drawingml/2006/chart" '
                'xmlns:ns1="http://schemas.openxmlformats.org/drawingml/2006/main"><ns0:axId val="180534160" /><ns0:scaling><ns0:orientation val="minMax" '
                '/></ns0:scaling><ns0:delete val="0" /><ns0:axPos val="l" /><ns0:numFmt formatCode="0" sourceLinked="0" /><ns0:majorTickMark val="none" '
                '/><ns0:minorTickMark val="none" /><ns0:tickLblPos val="nextTo" /><ns0:spPr><ns1:noFill /><ns1:ln><ns1:noFill /></ns1:ln><ns1:effectLst '
                '/></ns0:spPr><ns0:txPr><ns1:bodyPr rot="-60000000" spcFirstLastPara="1" vertOverflow="ellipsis" vert="horz" wrap="square" anchor="ctr" '
                'anchorCtr="1" /><ns1:lstStyle /><ns1:p><ns1:pPr><ns1:defRPr sz="1000" b="0" i="0" u="none" strike="noStrike" kern="1200" '
                'baseline="0"><ns1:solidFill><ns1:schemeClr val="tx1"><ns1:lumMod val="65000" /><ns1:lumOff val="35000" '
                '/></ns1:schemeClr></ns1:solidFill><ns1:latin typeface="Gill Sans MT" panose="020B0502020104020203" pitchFamily="34" charset="0" /><ns1:ea '
                'typeface="+mn-ea" /><ns1:cs typeface="+mn-cs" /></ns1:defRPr></ns1:pPr><ns1:endParaRPr lang="es-ES" /></ns1:p></ns0:txPr><ns0:crossAx '
                'val="180525840" /><ns0:crosses val="autoZero" /><ns0:crossBetween val="between" /></ns0:valAx>'],
  'value_ref': "'Gráfico 30'!$C$11",
  'category_ref': None,
  'name_ref': "'Gráfico 30'!$B$11",
  'physical_value_cells': ['C11'],
  'physical_category_cells': []},
 {'book': '2026',
  'block': 'Gráfico 30',
  'sid': 'chart38_s7',
  'headers': [('B3', 'GRÁFICO 30. DIFERENCIA CON OPINIÓN DE 2025 EN LA RATIO DE DEUDA EN 2050 (P.P)'),
              ('B7', 'Diferencia de la ratio a en 2050'),
              ('B8', 'Mejor punto de partida 2025'),
              ('B9', 'Gasto intereses'),
              ('B10', 'Crecimiento Real'),
              ('B11', 'Inflación'),
              ('B12', 'Déficit primario'),
              ('B13', 'Aj. stock flujo')],
  'keys': [(5083, 'C7', 'chart38_s7_p0')],
  'mode': 'normal',
  'unit': 'percentage_points_gdp',
  'multiplier': '1',
  'pdf': [70, ['DIFERENCIA', '2050']],
  'role': 'published_debt_ratio_difference_component',
  'scope': 'difference_vs_Opinion2025_at_2050',
  'horizon': '2050',
  'part': 'xl/charts/chart38.xml',
  'refs': ["'Gráfico 30'!$B$7", "'Gráfico 30'!$C$7"],
  'node_xml': '<ns0:ser xmlns:ns0="http://schemas.openxmlformats.org/drawingml/2006/chart" xmlns:ns1="http://schemas.openxmlformats.org/drawingml/2006/main" '
              'xmlns:ns2="http://schemas.microsoft.com/office/drawing/2014/chart" xmlns:ns3="http://schemas.microsoft.com/office/drawing/2012/chart"><ns0:idx '
              'val="0" /><ns0:order val="0" /><ns0:tx><ns0:strRef><ns0:f>\'Gráfico 30\'!$B$7</ns0:f><ns0:strCache><ns0:ptCount val="1" /><ns0:pt '
              'idx="0"><ns0:v>Diferencia de la ratio a en 2050</ns0:v></ns0:pt></ns0:strCache></ns0:strRef></ns0:tx><ns0:spPr><ns1:ln w="25400" '
              'cap="rnd"><ns1:noFill /><ns1:round /></ns1:ln><ns1:effectLst /></ns0:spPr><ns0:marker><ns0:symbol val="diamond" /><ns0:size val="8" '
              '/><ns0:spPr><ns1:solidFill><ns1:schemeClr val="bg1" /></ns1:solidFill><ns1:ln w="9525"><ns1:solidFill><ns1:schemeClr val="tx1" '
              '/></ns1:solidFill></ns1:ln><ns1:effectLst /></ns0:spPr></ns0:marker><ns0:dLbls><ns0:dLbl><ns0:idx val="0" '
              '/><ns0:layout><ns0:manualLayout><ns0:x val="-9.3654515407796249E-2" /><ns0:y val="5.3531876354651543E-2" '
              '/></ns0:manualLayout></ns0:layout><ns0:dLblPos val="r" /><ns0:showLegendKey val="0" /><ns0:showVal val="1" /><ns0:showCatName val="0" '
              '/><ns0:showSerName val="0" /><ns0:showPercent val="0" /><ns0:showBubbleSize val="0" /><ns0:extLst><ns0:ext '
              'uri="{CE6537A1-D6FC-4f65-9D91-7224C49458BB}" /><ns0:ext uri="{C3380CC4-5D6E-409C-BE32-E72D297353CC}"><ns2:uniqueId '
              'val="{00000007-722B-482F-96C6-FF4AFB22CA97}" /></ns0:ext></ns0:extLst></ns0:dLbl><ns0:spPr><ns1:noFill /><ns1:ln><ns1:noFill '
              '/></ns1:ln><ns1:effectLst /></ns0:spPr><ns0:txPr><ns1:bodyPr rot="0" spcFirstLastPara="1" vertOverflow="ellipsis" vert="horz" wrap="square" '
              'lIns="38100" tIns="19050" rIns="38100" bIns="19050" anchor="ctr" anchorCtr="1"><ns1:spAutoFit /></ns1:bodyPr><ns1:lstStyle '
              '/><ns1:p><ns1:pPr><ns1:defRPr sz="1050" b="1" i="0" u="none" strike="noStrike" kern="1200" baseline="0"><ns1:solidFill><ns1:schemeClr val="bg1" '
              '/></ns1:solidFill><ns1:latin typeface="Gill Sans MT" panose="020B0502020104020203" pitchFamily="34" charset="0" /><ns1:ea typeface="+mn-ea" '
              '/><ns1:cs typeface="+mn-cs" /></ns1:defRPr></ns1:pPr><ns1:endParaRPr lang="es-ES" /></ns1:p></ns0:txPr><ns0:dLblPos val="b" '
              '/><ns0:showLegendKey val="0" /><ns0:showVal val="1" /><ns0:showCatName val="0" /><ns0:showSerName val="0" /><ns0:showPercent val="0" '
              '/><ns0:showBubbleSize val="0" /><ns0:showLeaderLines val="0" /><ns0:extLst><ns0:ext '
              'uri="{CE6537A1-D6FC-4f65-9D91-7224C49458BB}"><ns3:showLeaderLines val="1" /><ns3:leaderLines><ns0:spPr><ns1:ln w="9525" cap="flat" cmpd="sng" '
              'algn="ctr"><ns1:solidFill><ns1:schemeClr val="tx1"><ns1:lumMod val="35000" /><ns1:lumOff val="65000" '
              '/></ns1:schemeClr></ns1:solidFill><ns1:round /></ns1:ln><ns1:effectLst '
              "/></ns0:spPr></ns3:leaderLines></ns0:ext></ns0:extLst></ns0:dLbls><ns0:val><ns0:numRef><ns0:f>'Gráfico "
              '30\'!$C$7</ns0:f><ns0:numCache><ns0:formatCode>#,##0.0</ns0:formatCode><ns0:ptCount val="1" /><ns0:pt '
              'idx="0"><ns0:v>-6.4416255274099701</ns0:v></ns0:pt></ns0:numCache></ns0:numRef></ns0:val><ns0:smooth val="0" /><ns0:extLst><ns0:ext '
              'uri="{C3380CC4-5D6E-409C-BE32-E72D297353CC}"><ns2:uniqueId val="{00000008-722B-482F-96C6-FF4AFB22CA97}" /></ns0:ext></ns0:extLst></ns0:ser>',
  'root_axes': ['<ns0:valAx xmlns:ns0="http://schemas.openxmlformats.org/drawingml/2006/chart" '
                'xmlns:ns1="http://schemas.openxmlformats.org/drawingml/2006/main"><ns0:axId val="180534160" /><ns0:scaling><ns0:orientation val="minMax" '
                '/></ns0:scaling><ns0:delete val="0" /><ns0:axPos val="l" /><ns0:numFmt formatCode="0" sourceLinked="0" /><ns0:majorTickMark val="none" '
                '/><ns0:minorTickMark val="none" /><ns0:tickLblPos val="nextTo" /><ns0:spPr><ns1:noFill /><ns1:ln><ns1:noFill /></ns1:ln><ns1:effectLst '
                '/></ns0:spPr><ns0:txPr><ns1:bodyPr rot="-60000000" spcFirstLastPara="1" vertOverflow="ellipsis" vert="horz" wrap="square" anchor="ctr" '
                'anchorCtr="1" /><ns1:lstStyle /><ns1:p><ns1:pPr><ns1:defRPr sz="1000" b="0" i="0" u="none" strike="noStrike" kern="1200" '
                'baseline="0"><ns1:solidFill><ns1:schemeClr val="tx1"><ns1:lumMod val="65000" /><ns1:lumOff val="35000" '
                '/></ns1:schemeClr></ns1:solidFill><ns1:latin typeface="Gill Sans MT" panose="020B0502020104020203" pitchFamily="34" charset="0" /><ns1:ea '
                'typeface="+mn-ea" /><ns1:cs typeface="+mn-cs" /></ns1:defRPr></ns1:pPr><ns1:endParaRPr lang="es-ES" /></ns1:p></ns0:txPr><ns0:crossAx '
                'val="180525840" /><ns0:crosses val="autoZero" /><ns0:crossBetween val="between" /></ns0:valAx>'],
  'value_ref': "'Gráfico 30'!$C$7",
  'category_ref': None,
  'name_ref': "'Gráfico 30'!$B$7",
  'physical_value_cells': ['C7'],
  'physical_category_cells': []},
 {'book': '2026',
  'block': 'Gráfico 30',
  'sid': 'chart38_s2',
  'headers': [('B3', 'GRÁFICO 30. DIFERENCIA CON OPINIÓN DE 2025 EN LA RATIO DE DEUDA EN 2050 (P.P)'),
              ('B7', 'Diferencia de la ratio a en 2050'),
              ('B8', 'Mejor punto de partida 2025'),
              ('B9', 'Gasto intereses'),
              ('B10', 'Crecimiento Real'),
              ('B11', 'Inflación'),
              ('B12', 'Déficit primario'),
              ('B13', 'Aj. stock flujo')],
  'keys': [(5078, 'C9', 'chart38_s2_p0')],
  'mode': 'normal',
  'unit': 'percentage_points_gdp',
  'multiplier': '1',
  'pdf': [70, ['DIFERENCIA', '2050']],
  'role': 'published_debt_ratio_difference_component',
  'scope': 'difference_vs_Opinion2025_at_2050',
  'horizon': '2050',
  'part': 'xl/charts/chart38.xml',
  'refs': ["'Gráfico 30'!$B$9", "'Gráfico 30'!$C$9"],
  'node_xml': '<ns0:ser xmlns:ns0="http://schemas.openxmlformats.org/drawingml/2006/chart" xmlns:ns1="http://schemas.openxmlformats.org/drawingml/2006/main" '
              'xmlns:ns2="http://schemas.microsoft.com/office/drawing/2012/chart" xmlns:ns3="http://schemas.microsoft.com/office/drawing/2014/chart"><ns0:idx '
              'val="2" /><ns0:order val="2" /><ns0:tx><ns0:strRef><ns0:f>\'Gráfico 30\'!$B$9</ns0:f><ns0:strCache><ns0:ptCount val="1" /><ns0:pt '
              'idx="0"><ns0:v>Gasto intereses</ns0:v></ns0:pt></ns0:strCache></ns0:strRef></ns0:tx><ns0:spPr><ns1:solidFill><ns1:schemeClr val="accent6" '
              '/></ns1:solidFill><ns1:ln><ns1:noFill /></ns1:ln><ns1:effectLst /></ns0:spPr><ns0:invertIfNegative val="0" /><ns0:dLbls><ns0:spPr><ns1:noFill '
              '/><ns1:ln><ns1:noFill /></ns1:ln><ns1:effectLst /></ns0:spPr><ns0:txPr><ns1:bodyPr rot="0" spcFirstLastPara="1" vertOverflow="ellipsis" '
              'vert="horz" wrap="square" lIns="38100" tIns="19050" rIns="38100" bIns="19050" anchor="ctr" anchorCtr="1"><ns1:spAutoFit '
              '/></ns1:bodyPr><ns1:lstStyle /><ns1:p><ns1:pPr><ns1:defRPr sz="1050" b="0" i="0" u="none" strike="noStrike" kern="1200" '
              'baseline="0"><ns1:solidFill><ns1:schemeClr val="accent5" /></ns1:solidFill><ns1:latin typeface="Gill Sans MT" panose="020B0502020104020203" '
              'pitchFamily="34" charset="0" /><ns1:ea typeface="+mn-ea" /><ns1:cs typeface="+mn-cs" /></ns1:defRPr></ns1:pPr><ns1:endParaRPr lang="es-ES" '
              '/></ns1:p></ns0:txPr><ns0:showLegendKey val="0" /><ns0:showVal val="1" /><ns0:showCatName val="0" /><ns0:showSerName val="0" /><ns0:showPercent '
              'val="0" /><ns0:showBubbleSize val="0" /><ns0:showLeaderLines val="0" /><ns0:extLst><ns0:ext '
              'uri="{CE6537A1-D6FC-4f65-9D91-7224C49458BB}"><ns2:showLeaderLines val="1" /><ns2:leaderLines><ns0:spPr><ns1:ln w="9525" cap="flat" cmpd="sng" '
              'algn="ctr"><ns1:solidFill><ns1:schemeClr val="tx1"><ns1:lumMod val="35000" /><ns1:lumOff val="65000" '
              '/></ns1:schemeClr></ns1:solidFill><ns1:round /></ns1:ln><ns1:effectLst '
              "/></ns0:spPr></ns2:leaderLines></ns0:ext></ns0:extLst></ns0:dLbls><ns0:val><ns0:numRef><ns0:f>'Gráfico "
              '30\'!$C$9</ns0:f><ns0:numCache><ns0:formatCode>#,##0.0</ns0:formatCode><ns0:ptCount val="1" /><ns0:pt '
              'idx="0"><ns0:v>-5.2257679432943718</ns0:v></ns0:pt></ns0:numCache></ns0:numRef></ns0:val><ns0:extLst><ns0:ext '
              'uri="{C3380CC4-5D6E-409C-BE32-E72D297353CC}"><ns3:uniqueId val="{00000001-722B-482F-96C6-FF4AFB22CA97}" /></ns0:ext></ns0:extLst></ns0:ser>',
  'root_axes': ['<ns0:valAx xmlns:ns0="http://schemas.openxmlformats.org/drawingml/2006/chart" '
                'xmlns:ns1="http://schemas.openxmlformats.org/drawingml/2006/main"><ns0:axId val="180534160" /><ns0:scaling><ns0:orientation val="minMax" '
                '/></ns0:scaling><ns0:delete val="0" /><ns0:axPos val="l" /><ns0:numFmt formatCode="0" sourceLinked="0" /><ns0:majorTickMark val="none" '
                '/><ns0:minorTickMark val="none" /><ns0:tickLblPos val="nextTo" /><ns0:spPr><ns1:noFill /><ns1:ln><ns1:noFill /></ns1:ln><ns1:effectLst '
                '/></ns0:spPr><ns0:txPr><ns1:bodyPr rot="-60000000" spcFirstLastPara="1" vertOverflow="ellipsis" vert="horz" wrap="square" anchor="ctr" '
                'anchorCtr="1" /><ns1:lstStyle /><ns1:p><ns1:pPr><ns1:defRPr sz="1000" b="0" i="0" u="none" strike="noStrike" kern="1200" '
                'baseline="0"><ns1:solidFill><ns1:schemeClr val="tx1"><ns1:lumMod val="65000" /><ns1:lumOff val="35000" '
                '/></ns1:schemeClr></ns1:solidFill><ns1:latin typeface="Gill Sans MT" panose="020B0502020104020203" pitchFamily="34" charset="0" /><ns1:ea '
                'typeface="+mn-ea" /><ns1:cs typeface="+mn-cs" /></ns1:defRPr></ns1:pPr><ns1:endParaRPr lang="es-ES" /></ns1:p></ns0:txPr><ns0:crossAx '
                'val="180525840" /><ns0:crosses val="autoZero" /><ns0:crossBetween val="between" /></ns0:valAx>'],
  'value_ref': "'Gráfico 30'!$C$9",
  'category_ref': None,
  'name_ref': "'Gráfico 30'!$B$9",
  'physical_value_cells': ['C9'],
  'physical_category_cells': []},
 {'book': '2026',
  'block': 'Gráfico 30',
  'sid': 'chart38_s1',
  'headers': [('B3', 'GRÁFICO 30. DIFERENCIA CON OPINIÓN DE 2025 EN LA RATIO DE DEUDA EN 2050 (P.P)'),
              ('B7', 'Diferencia de la ratio a en 2050'),
              ('B8', 'Mejor punto de partida 2025'),
              ('B9', 'Gasto intereses'),
              ('B10', 'Crecimiento Real'),
              ('B11', 'Inflación'),
              ('B12', 'Déficit primario'),
              ('B13', 'Aj. stock flujo')],
  'keys': [(5077, 'C8', 'chart38_s1_p0')],
  'mode': 'normal',
  'unit': 'percentage_points_gdp',
  'multiplier': '1',
  'pdf': [70, ['DIFERENCIA', '2050']],
  'role': 'published_debt_ratio_difference_component',
  'scope': 'difference_vs_Opinion2025_at_2050',
  'horizon': '2050',
  'part': 'xl/charts/chart38.xml',
  'refs': ["'Gráfico 30'!$B$8", "'Gráfico 30'!$C$8"],
  'node_xml': '<ns0:ser xmlns:ns0="http://schemas.openxmlformats.org/drawingml/2006/chart" xmlns:ns1="http://schemas.openxmlformats.org/drawingml/2006/main" '
              'xmlns:ns2="http://schemas.microsoft.com/office/drawing/2014/chart"><ns0:idx val="1" /><ns0:order val="1" /><ns0:tx><ns0:strRef><ns0:f>\'Gráfico '
              '30\'!$B$8</ns0:f><ns0:strCache><ns0:ptCount val="1" /><ns0:pt idx="0"><ns0:v>Mejor punto de partida '
              '2025</ns0:v></ns0:pt></ns0:strCache></ns0:strRef></ns0:tx><ns0:spPr><ns1:solidFill><ns1:srgbClr val="FFC000" '
              '/></ns1:solidFill><ns1:ln><ns1:noFill /></ns1:ln><ns1:effectLst /></ns0:spPr><ns0:invertIfNegative val="0" '
              '/><ns0:val><ns0:numRef><ns0:f>\'Gráfico 30\'!$C$8</ns0:f><ns0:numCache><ns0:formatCode>#,##0.0</ns0:formatCode><ns0:ptCount val="1" /><ns0:pt '
              'idx="0"><ns0:v>0.13134695602559532</ns0:v></ns0:pt></ns0:numCache></ns0:numRef></ns0:val><ns0:extLst><ns0:ext '
              'uri="{C3380CC4-5D6E-409C-BE32-E72D297353CC}"><ns2:uniqueId val="{00000000-722B-482F-96C6-FF4AFB22CA97}" /></ns0:ext></ns0:extLst></ns0:ser>',
  'root_axes': ['<ns0:valAx xmlns:ns0="http://schemas.openxmlformats.org/drawingml/2006/chart" '
                'xmlns:ns1="http://schemas.openxmlformats.org/drawingml/2006/main"><ns0:axId val="180534160" /><ns0:scaling><ns0:orientation val="minMax" '
                '/></ns0:scaling><ns0:delete val="0" /><ns0:axPos val="l" /><ns0:numFmt formatCode="0" sourceLinked="0" /><ns0:majorTickMark val="none" '
                '/><ns0:minorTickMark val="none" /><ns0:tickLblPos val="nextTo" /><ns0:spPr><ns1:noFill /><ns1:ln><ns1:noFill /></ns1:ln><ns1:effectLst '
                '/></ns0:spPr><ns0:txPr><ns1:bodyPr rot="-60000000" spcFirstLastPara="1" vertOverflow="ellipsis" vert="horz" wrap="square" anchor="ctr" '
                'anchorCtr="1" /><ns1:lstStyle /><ns1:p><ns1:pPr><ns1:defRPr sz="1000" b="0" i="0" u="none" strike="noStrike" kern="1200" '
                'baseline="0"><ns1:solidFill><ns1:schemeClr val="tx1"><ns1:lumMod val="65000" /><ns1:lumOff val="35000" '
                '/></ns1:schemeClr></ns1:solidFill><ns1:latin typeface="Gill Sans MT" panose="020B0502020104020203" pitchFamily="34" charset="0" /><ns1:ea '
                'typeface="+mn-ea" /><ns1:cs typeface="+mn-cs" /></ns1:defRPr></ns1:pPr><ns1:endParaRPr lang="es-ES" /></ns1:p></ns0:txPr><ns0:crossAx '
                'val="180525840" /><ns0:crosses val="autoZero" /><ns0:crossBetween val="between" /></ns0:valAx>'],
  'value_ref': "'Gráfico 30'!$C$8",
  'category_ref': None,
  'name_ref': "'Gráfico 30'!$B$8",
  'physical_value_cells': ['C8'],
  'physical_category_cells': []},
 {'book': '2026',
  'block': 'Gráfico 30',
  'sid': 'chart38_s5',
  'headers': [('B3', 'GRÁFICO 30. DIFERENCIA CON OPINIÓN DE 2025 EN LA RATIO DE DEUDA EN 2050 (P.P)'),
              ('B7', 'Diferencia de la ratio a en 2050'),
              ('B8', 'Mejor punto de partida 2025'),
              ('B9', 'Gasto intereses'),
              ('B10', 'Crecimiento Real'),
              ('B11', 'Inflación'),
              ('B12', 'Déficit primario'),
              ('B13', 'Aj. stock flujo')],
  'keys': [(5081, 'C12', 'chart38_s5_p0')],
  'mode': 'normal',
  'unit': 'percentage_points_gdp',
  'multiplier': '1',
  'pdf': [70, ['DIFERENCIA', '2050']],
  'role': 'published_debt_ratio_difference_component',
  'scope': 'difference_vs_Opinion2025_at_2050',
  'horizon': '2050',
  'part': 'xl/charts/chart38.xml',
  'refs': ["'Gráfico 30'!$B$12", "'Gráfico 30'!$C$12"],
  'node_xml': '<ns0:ser xmlns:ns0="http://schemas.openxmlformats.org/drawingml/2006/chart" xmlns:ns1="http://schemas.openxmlformats.org/drawingml/2006/main" '
              'xmlns:ns2="http://schemas.microsoft.com/office/drawing/2014/chart" xmlns:ns3="http://schemas.microsoft.com/office/drawing/2012/chart"><ns0:idx '
              'val="5" /><ns0:order val="5" /><ns0:tx><ns0:strRef><ns0:f>\'Gráfico 30\'!$B$12</ns0:f><ns0:strCache><ns0:ptCount val="1" /><ns0:pt '
              'idx="0"><ns0:v>Déficit primario</ns0:v></ns0:pt></ns0:strCache></ns0:strRef></ns0:tx><ns0:spPr><ns1:solidFill><ns1:schemeClr val="accent1" '
              '/></ns1:solidFill><ns1:ln w="25400"><ns1:noFill /></ns1:ln><ns1:effectLst /></ns0:spPr><ns0:invertIfNegative val="0" '
              '/><ns0:dLbls><ns0:dLbl><ns0:idx val="0" /><ns0:layout><ns0:manualLayout><ns0:x val="4.9382716049382715E-3" /><ns0:y '
              'val="-7.0319664815767277E-2" /></ns0:manualLayout></ns0:layout><ns0:dLblPos val="ctr" /><ns0:showLegendKey val="0" /><ns0:showVal val="1" '
              '/><ns0:showCatName val="0" /><ns0:showSerName val="0" /><ns0:showPercent val="0" /><ns0:showBubbleSize val="0" /><ns0:extLst><ns0:ext '
              'uri="{CE6537A1-D6FC-4f65-9D91-7224C49458BB}" /><ns0:ext uri="{C3380CC4-5D6E-409C-BE32-E72D297353CC}"><ns2:uniqueId '
              'val="{00000004-722B-482F-96C6-FF4AFB22CA97}" /></ns0:ext></ns0:extLst></ns0:dLbl><ns0:spPr><ns1:noFill /><ns1:ln><ns1:noFill '
              '/></ns1:ln><ns1:effectLst /></ns0:spPr><ns0:txPr><ns1:bodyPr rot="0" spcFirstLastPara="1" vertOverflow="ellipsis" vert="horz" wrap="square" '
              'lIns="38100" tIns="19050" rIns="38100" bIns="19050" anchor="ctr" anchorCtr="1"><ns1:spAutoFit /></ns1:bodyPr><ns1:lstStyle '
              '/><ns1:p><ns1:pPr><ns1:defRPr sz="1050" b="0" i="0" u="none" strike="noStrike" kern="1200" baseline="0"><ns1:solidFill><ns1:schemeClr '
              'val="accent1"><ns1:lumMod val="20000" /><ns1:lumOff val="80000" /></ns1:schemeClr></ns1:solidFill><ns1:latin typeface="Gill Sans MT" '
              'panose="020B0502020104020203" pitchFamily="34" charset="0" /><ns1:ea typeface="+mn-ea" /><ns1:cs typeface="+mn-cs" '
              '/></ns1:defRPr></ns1:pPr><ns1:endParaRPr lang="es-ES" /></ns1:p></ns0:txPr><ns0:dLblPos val="inBase" /><ns0:showLegendKey val="0" '
              '/><ns0:showVal val="1" /><ns0:showCatName val="0" /><ns0:showSerName val="0" /><ns0:showPercent val="0" /><ns0:showBubbleSize val="0" '
              '/><ns0:showLeaderLines val="0" /><ns0:extLst><ns0:ext uri="{CE6537A1-D6FC-4f65-9D91-7224C49458BB}"><ns3:showLeaderLines val="1" '
              '/><ns3:leaderLines><ns0:spPr><ns1:ln w="9525" cap="flat" cmpd="sng" algn="ctr"><ns1:solidFill><ns1:schemeClr val="tx1"><ns1:lumMod val="35000" '
              '/><ns1:lumOff val="65000" /></ns1:schemeClr></ns1:solidFill><ns1:round /></ns1:ln><ns1:effectLst '
              "/></ns0:spPr></ns3:leaderLines></ns0:ext></ns0:extLst></ns0:dLbls><ns0:val><ns0:numRef><ns0:f>'Gráfico "
              '30\'!$C$12</ns0:f><ns0:numCache><ns0:formatCode>#,##0.0</ns0:formatCode><ns0:ptCount val="1" /><ns0:pt '
              'idx="0"><ns0:v>-5.3134502425695516</ns0:v></ns0:pt></ns0:numCache></ns0:numRef></ns0:val><ns0:extLst><ns0:ext '
              'uri="{C3380CC4-5D6E-409C-BE32-E72D297353CC}"><ns2:uniqueId val="{00000005-722B-482F-96C6-FF4AFB22CA97}" /></ns0:ext></ns0:extLst></ns0:ser>',
  'root_axes': ['<ns0:valAx xmlns:ns0="http://schemas.openxmlformats.org/drawingml/2006/chart" '
                'xmlns:ns1="http://schemas.openxmlformats.org/drawingml/2006/main"><ns0:axId val="180534160" /><ns0:scaling><ns0:orientation val="minMax" '
                '/></ns0:scaling><ns0:delete val="0" /><ns0:axPos val="l" /><ns0:numFmt formatCode="0" sourceLinked="0" /><ns0:majorTickMark val="none" '
                '/><ns0:minorTickMark val="none" /><ns0:tickLblPos val="nextTo" /><ns0:spPr><ns1:noFill /><ns1:ln><ns1:noFill /></ns1:ln><ns1:effectLst '
                '/></ns0:spPr><ns0:txPr><ns1:bodyPr rot="-60000000" spcFirstLastPara="1" vertOverflow="ellipsis" vert="horz" wrap="square" anchor="ctr" '
                'anchorCtr="1" /><ns1:lstStyle /><ns1:p><ns1:pPr><ns1:defRPr sz="1000" b="0" i="0" u="none" strike="noStrike" kern="1200" '
                'baseline="0"><ns1:solidFill><ns1:schemeClr val="tx1"><ns1:lumMod val="65000" /><ns1:lumOff val="35000" '
                '/></ns1:schemeClr></ns1:solidFill><ns1:latin typeface="Gill Sans MT" panose="020B0502020104020203" pitchFamily="34" charset="0" /><ns1:ea '
                'typeface="+mn-ea" /><ns1:cs typeface="+mn-cs" /></ns1:defRPr></ns1:pPr><ns1:endParaRPr lang="es-ES" /></ns1:p></ns0:txPr><ns0:crossAx '
                'val="180525840" /><ns0:crosses val="autoZero" /><ns0:crossBetween val="between" /></ns0:valAx>'],
  'value_ref': "'Gráfico 30'!$C$12",
  'category_ref': None,
  'name_ref': "'Gráfico 30'!$B$12",
  'physical_value_cells': ['C12'],
  'physical_category_cells': []},
 {'book': '2025',
  'block': 'Gráfico 72',
  'sid': 'chart125_s3',
  'headers': [('B2', 'Gráfico 72. DIFERENCIA CON OPINIÓN DE 2023'),
              ('B4', 'Diferencia de la ratio de deuda en 2070'),
              ('B5', 'Mejor punto de partida 2024'),
              ('B6', 'Mayor acumulación de deuda hasta 2070')],
  'keys': [(36607, 'C4', 'chart125_s3_p0')],
  'mode': 'normal',
  'unit': 'percentage_points_gdp',
  'multiplier': '1',
  'pdf': [131, ['5 puntos', '2070']],
  'role': 'published_debt_ratio_difference_component',
  'scope': 'difference_vs_Opinion2023_at_2070',
  'horizon': '2070',
  'part': 'xl/charts/chart125.xml',
  'refs': ["'Gráfico 72'!$B$4", "'Gráfico 72'!$C$4"],
  'node_xml': '<ns0:ser xmlns:ns0="http://schemas.openxmlformats.org/drawingml/2006/chart" xmlns:ns1="http://schemas.openxmlformats.org/drawingml/2006/main" '
              'xmlns:ns2="http://schemas.microsoft.com/office/drawing/2014/chart" xmlns:ns3="http://schemas.microsoft.com/office/drawing/2012/chart"><ns0:idx '
              'val="0" /><ns0:order val="0" /><ns0:tx><ns0:strRef><ns0:f>\'Gráfico 72\'!$B$4</ns0:f><ns0:strCache><ns0:ptCount val="1" /><ns0:pt '
              'idx="0"><ns0:v>Diferencia de la ratio de deuda en 2070</ns0:v></ns0:pt></ns0:strCache></ns0:strRef></ns0:tx><ns0:spPr><ns1:ln w="28575" '
              'cap="rnd"><ns1:noFill /><ns1:round /></ns1:ln><ns1:effectLst /></ns0:spPr><ns0:marker><ns0:symbol val="diamond" /><ns0:size val="8" '
              '/><ns0:spPr><ns1:solidFill><ns1:schemeClr val="tx1" /></ns1:solidFill><ns1:ln w="9525"><ns1:noFill /></ns1:ln><ns1:effectLst '
              '/></ns0:spPr></ns0:marker><ns0:dLbls><ns0:dLbl><ns0:idx val="0" /><ns0:layout><ns0:manualLayout><ns0:x val="9.9227646006077256E-3" /><ns0:y '
              'val="1.5013541633988881E-2" /></ns0:manualLayout></ns0:layout><ns0:showLegendKey val="0" /><ns0:showVal val="1" /><ns0:showCatName val="0" '
              '/><ns0:showSerName val="0" /><ns0:showPercent val="0" /><ns0:showBubbleSize val="0" /><ns0:extLst><ns0:ext '
              'uri="{CE6537A1-D6FC-4f65-9D91-7224C49458BB}" /><ns0:ext uri="{C3380CC4-5D6E-409C-BE32-E72D297353CC}"><ns2:uniqueId '
              'val="{00000003-6C44-4976-A5F1-BDB1C5C23A10}" /></ns0:ext></ns0:extLst></ns0:dLbl><ns0:spPr><ns1:noFill /><ns1:ln><ns1:noFill '
              '/></ns1:ln><ns1:effectLst /></ns0:spPr><ns0:txPr><ns1:bodyPr rot="0" spcFirstLastPara="1" vertOverflow="ellipsis" vert="horz" wrap="square" '
              'lIns="38100" tIns="19050" rIns="38100" bIns="19050" anchor="ctr" anchorCtr="1"><ns1:spAutoFit /></ns1:bodyPr><ns1:lstStyle '
              '/><ns1:p><ns1:pPr><ns1:defRPr sz="1050" b="1" i="0" u="none" strike="noStrike" kern="1200" baseline="0"><ns1:solidFill><ns1:schemeClr '
              'val="tx1"><ns1:lumMod val="75000" /><ns1:lumOff val="25000" /></ns1:schemeClr></ns1:solidFill><ns1:latin typeface="Gill Sans MT" '
              'panose="020B0502020104020203" pitchFamily="34" charset="0" /><ns1:ea typeface="+mn-ea" /><ns1:cs typeface="+mn-cs" '
              '/></ns1:defRPr></ns1:pPr><ns1:endParaRPr lang="es-ES" /></ns1:p></ns0:txPr><ns0:showLegendKey val="0" /><ns0:showVal val="1" /><ns0:showCatName '
              'val="0" /><ns0:showSerName val="0" /><ns0:showPercent val="0" /><ns0:showBubbleSize val="0" /><ns0:showLeaderLines val="0" '
              '/><ns0:extLst><ns0:ext uri="{CE6537A1-D6FC-4f65-9D91-7224C49458BB}"><ns3:showLeaderLines val="1" /><ns3:leaderLines><ns0:spPr><ns1:ln w="9525" '
              'cap="flat" cmpd="sng" algn="ctr"><ns1:solidFill><ns1:schemeClr val="tx1"><ns1:lumMod val="35000" /><ns1:lumOff val="65000" '
              '/></ns1:schemeClr></ns1:solidFill><ns1:round /></ns1:ln><ns1:effectLst '
              "/></ns0:spPr></ns3:leaderLines></ns0:ext></ns0:extLst></ns0:dLbls><ns0:val><ns0:numRef><ns0:f>'Gráfico "
              '72\'!$C$4</ns0:f><ns0:numCache><ns0:formatCode>#,##0</ns0:formatCode><ns0:ptCount val="1" /><ns0:pt '
              'idx="0"><ns0:v>-5.4727172348318476</ns0:v></ns0:pt></ns0:numCache></ns0:numRef></ns0:val><ns0:smooth val="0" /><ns0:extLst><ns0:ext '
              'uri="{C3380CC4-5D6E-409C-BE32-E72D297353CC}"><ns2:uniqueId val="{00000004-6C44-4976-A5F1-BDB1C5C23A10}" /></ns0:ext></ns0:extLst></ns0:ser>',
  'root_axes': ['<ns0:valAx xmlns:ns0="http://schemas.openxmlformats.org/drawingml/2006/chart" '
                'xmlns:ns1="http://schemas.openxmlformats.org/drawingml/2006/main"><ns0:axId val="180534160" /><ns0:scaling><ns0:orientation val="minMax" '
                '/></ns0:scaling><ns0:delete val="0" /><ns0:axPos val="l" /><ns0:numFmt formatCode="0" sourceLinked="0" /><ns0:majorTickMark val="none" '
                '/><ns0:minorTickMark val="none" /><ns0:tickLblPos val="nextTo" /><ns0:spPr><ns1:noFill /><ns1:ln><ns1:noFill /></ns1:ln><ns1:effectLst '
                '/></ns0:spPr><ns0:txPr><ns1:bodyPr rot="-60000000" spcFirstLastPara="1" vertOverflow="ellipsis" vert="horz" wrap="square" anchor="ctr" '
                'anchorCtr="1" /><ns1:lstStyle /><ns1:p><ns1:pPr><ns1:defRPr sz="1000" b="0" i="0" u="none" strike="noStrike" kern="1200" '
                'baseline="0"><ns1:solidFill><ns1:schemeClr val="tx1"><ns1:lumMod val="65000" /><ns1:lumOff val="35000" '
                '/></ns1:schemeClr></ns1:solidFill><ns1:latin typeface="Gill Sans MT" panose="020B0502020104020203" pitchFamily="34" charset="0" /><ns1:ea '
                'typeface="+mn-ea" /><ns1:cs typeface="+mn-cs" /></ns1:defRPr></ns1:pPr><ns1:endParaRPr lang="es-ES" /></ns1:p></ns0:txPr><ns0:crossAx '
                'val="180525840" /><ns0:crosses val="autoZero" /><ns0:crossBetween val="between" /></ns0:valAx>'],
  'value_ref': "'Gráfico 72'!$C$4",
  'category_ref': None,
  'name_ref': "'Gráfico 72'!$B$4",
  'physical_value_cells': ['C4'],
  'physical_category_cells': []},
 {'book': '2025',
  'block': 'Gráfico 72',
  'sid': 'chart125_s2',
  'headers': [('B2', 'Gráfico 72. DIFERENCIA CON OPINIÓN DE 2023'),
              ('B4', 'Diferencia de la ratio de deuda en 2070'),
              ('B5', 'Mejor punto de partida 2024'),
              ('B6', 'Mayor acumulación de deuda hasta 2070')],
  'keys': [(36606, 'C6', 'chart125_s2_p0')],
  'mode': 'normal',
  'unit': 'percentage_points_gdp',
  'multiplier': '1',
  'pdf': [131, ['5 puntos', '2070']],
  'role': 'published_debt_ratio_difference_component',
  'scope': 'difference_vs_Opinion2023_at_2070',
  'horizon': '2070',
  'part': 'xl/charts/chart125.xml',
  'refs': ["'Gráfico 72'!$B$6", "'Gráfico 72'!$C$6"],
  'node_xml': '<ns0:ser xmlns:ns0="http://schemas.openxmlformats.org/drawingml/2006/chart" xmlns:ns1="http://schemas.openxmlformats.org/drawingml/2006/main" '
              'xmlns:ns2="http://schemas.microsoft.com/office/drawing/2012/chart" xmlns:ns3="http://schemas.microsoft.com/office/drawing/2014/chart"><ns0:idx '
              'val="2" /><ns0:order val="2" /><ns0:tx><ns0:strRef><ns0:f>\'Gráfico 72\'!$B$6</ns0:f><ns0:strCache><ns0:ptCount val="1" /><ns0:pt '
              'idx="0"><ns0:v>Mayor acumulación de deuda hasta '
              '2070</ns0:v></ns0:pt></ns0:strCache></ns0:strRef></ns0:tx><ns0:spPr><ns1:solidFill><ns1:schemeClr val="accent3" '
              '/></ns1:solidFill><ns1:ln><ns1:noFill /></ns1:ln><ns1:effectLst /></ns0:spPr><ns0:invertIfNegative val="0" /><ns0:dLbls><ns0:spPr><ns1:noFill '
              '/><ns1:ln><ns1:noFill /></ns1:ln><ns1:effectLst /></ns0:spPr><ns0:txPr><ns1:bodyPr rot="0" spcFirstLastPara="1" vertOverflow="ellipsis" '
              'vert="horz" wrap="square" lIns="38100" tIns="19050" rIns="38100" bIns="19050" anchor="ctr" anchorCtr="1"><ns1:spAutoFit '
              '/></ns1:bodyPr><ns1:lstStyle /><ns1:p><ns1:pPr><ns1:defRPr sz="1000" b="0" i="0" u="none" strike="noStrike" kern="1200" '
              'baseline="0"><ns1:solidFill><ns1:schemeClr val="bg1" /></ns1:solidFill><ns1:latin typeface="Gill Sans MT" panose="020B0502020104020203" '
              'pitchFamily="34" charset="0" /><ns1:ea typeface="+mn-ea" /><ns1:cs typeface="+mn-cs" /></ns1:defRPr></ns1:pPr><ns1:endParaRPr lang="es-ES" '
              '/></ns1:p></ns0:txPr><ns0:dLblPos val="inEnd" /><ns0:showLegendKey val="0" /><ns0:showVal val="1" /><ns0:showCatName val="0" /><ns0:showSerName '
              'val="0" /><ns0:showPercent val="0" /><ns0:showBubbleSize val="0" /><ns0:showLeaderLines val="0" /><ns0:extLst><ns0:ext '
              'uri="{CE6537A1-D6FC-4f65-9D91-7224C49458BB}"><ns2:showLeaderLines val="1" /><ns2:leaderLines><ns0:spPr><ns1:ln w="9525" cap="flat" cmpd="sng" '
              'algn="ctr"><ns1:solidFill><ns1:schemeClr val="tx1"><ns1:lumMod val="35000" /><ns1:lumOff val="65000" '
              '/></ns1:schemeClr></ns1:solidFill><ns1:round /></ns1:ln><ns1:effectLst '
              "/></ns0:spPr></ns2:leaderLines></ns0:ext></ns0:extLst></ns0:dLbls><ns0:val><ns0:numRef><ns0:f>'Gráfico "
              '72\'!$C$6</ns0:f><ns0:numCache><ns0:formatCode>#,##0</ns0:formatCode><ns0:ptCount val="1" /><ns0:pt '
              'idx="0"><ns0:v>1.9555321256998894</ns0:v></ns0:pt></ns0:numCache></ns0:numRef></ns0:val><ns0:extLst><ns0:ext '
              'uri="{C3380CC4-5D6E-409C-BE32-E72D297353CC}"><ns3:uniqueId val="{00000002-6C44-4976-A5F1-BDB1C5C23A10}" /></ns0:ext></ns0:extLst></ns0:ser>',
  'root_axes': ['<ns0:valAx xmlns:ns0="http://schemas.openxmlformats.org/drawingml/2006/chart" '
                'xmlns:ns1="http://schemas.openxmlformats.org/drawingml/2006/main"><ns0:axId val="180534160" /><ns0:scaling><ns0:orientation val="minMax" '
                '/></ns0:scaling><ns0:delete val="0" /><ns0:axPos val="l" /><ns0:numFmt formatCode="0" sourceLinked="0" /><ns0:majorTickMark val="none" '
                '/><ns0:minorTickMark val="none" /><ns0:tickLblPos val="nextTo" /><ns0:spPr><ns1:noFill /><ns1:ln><ns1:noFill /></ns1:ln><ns1:effectLst '
                '/></ns0:spPr><ns0:txPr><ns1:bodyPr rot="-60000000" spcFirstLastPara="1" vertOverflow="ellipsis" vert="horz" wrap="square" anchor="ctr" '
                'anchorCtr="1" /><ns1:lstStyle /><ns1:p><ns1:pPr><ns1:defRPr sz="1000" b="0" i="0" u="none" strike="noStrike" kern="1200" '
                'baseline="0"><ns1:solidFill><ns1:schemeClr val="tx1"><ns1:lumMod val="65000" /><ns1:lumOff val="35000" '
                '/></ns1:schemeClr></ns1:solidFill><ns1:latin typeface="Gill Sans MT" panose="020B0502020104020203" pitchFamily="34" charset="0" /><ns1:ea '
                'typeface="+mn-ea" /><ns1:cs typeface="+mn-cs" /></ns1:defRPr></ns1:pPr><ns1:endParaRPr lang="es-ES" /></ns1:p></ns0:txPr><ns0:crossAx '
                'val="180525840" /><ns0:crosses val="autoZero" /><ns0:crossBetween val="between" /></ns0:valAx>'],
  'value_ref': "'Gráfico 72'!$C$6",
  'category_ref': None,
  'name_ref': "'Gráfico 72'!$B$6",
  'physical_value_cells': ['C6'],
  'physical_category_cells': []}]
def dumps(x):return json.dumps(x,ensure_ascii=False,sort_keys=True,default=str)
def digest(x):return hashlib.sha256(dumps(x).encode()).hexdigest()
PROFILES={(p['book'],p['block'],p['sid']):p for p in PROFILES_LIST}
BASE_OBSERVATION_TABLES=['blocked_observations','bulk_observations','context_observations','debt_observations','extended_family_observations','family_observations','final_family_observations','formula_observations','grid_observations','header_observations','indexed_observations','labour_observations','next_observations','pending_family_observations','quantity_observations','ready_observations','reconciled_observations','remaining_observations','resolved_observations','reviewed_family_observations','scale_observations','source_observations','table_observations']
def load_inventory():
 require(sha(INVENTORY)==INVENTORY_SHA,'Frozen closure inventory changed');x=json.loads(INVENTORY.read_text(encoding='utf8'));require(x['counts']==dict(usable=145,excluded=421,internal_validated=168,blocked=4115),'Closure disposition counts changed');rows=x['records'];require(len(rows)==4849 and len({(a['book'],a['row_number'])for a in rows})==4849 and all(a['coverage_complete']and not a['admission_performed']for a in rows),'Closure inventory incomplete');return rows
class ClosureResolver(ReviewedResolver):
 def __init__(self):super().__init__();self.expected_payloads={(a['book'],a['row_number']):a['payload_sha256']for a in load_inventory()if a['disposition']=='usable'}
 def profile(self,row):
  p=PROFILES[row['book'],row['bloque'],row['serie_id']];s=self.sources[row['book']];require(all(s.text(p['block'],a)==v for a,v in p['headers']),'Closure physical scope/header changed');return p,s
 def point(self,row):
  require(digest(row)==self.expected_payloads.get((row['book'],row['row_number'])),'Closure point original metadata/approved key changed')
  p,s=self.profile(row);sh=p['block'];require(any(n==row['row_number']and rid==row['id']for n,cell,rid in p['keys']),'Closure unapproved point');proof=dict(headers=p['headers'],universal_series=row['serie_literal'],universal_category=row['periodo_categoria_literal'],universal_metric=row['metrica_literal'],unit=p['unit'],pdf=self.pdf(row['book'],*p['pdf']),semantic_scope=p.get('scope'));i=None;cell=row['celda_valor'];category=row['periodo_categoria_literal'];label=row['serie_literal'];role=p.get('role','published_quantity_as_source_labelled');mult=Decimal(p['multiplier']);unit=p['unit'];value_ref=category_ref=name_ref=None;axis_id=axis_format=None
  if p['mode']=='published_table':
   require(row['serie_id']is None and row['hoja_valor']==sh,'Closure table identity changed');rr=int(re.search('[0-9]+',cell)[0]);col=re.match('[A-Z]+',cell)[0];require((rr==7 and col in'DEFG')or(8<=rr<=13 and col in'EFG'),'Cuadro14 unpublished cell');label=s.text(sh,'C'+str(rr));category=s.text(sh,col+'6');unit='percent_interest_rate_new_issues'if rr==7 else'percentage_points_interest_rate_new_issues';role='published_base_new_issue_interest_rate'if rr==7 else'published_scenario_interest_rate_difference_from_base';raw=s.number(sh,cell);require(raw==Decimal(row['decimal_XML_texto']),'Cuadro14 XML/raw mismatch');rounded=raw.quantize(Decimal('.1'),rounding=ROUND_HALF_UP);published={'D7':'3.3','E7':'3.6','F7':'4.2','G7':'4.1','E8':'.2','F8':'.1','G8':'0','E9':'.2','F9':'.6','G9':'.3','E10':'0','F10':'.4','G10':'.1','E11':'.5','F11':'.1','G11':'-.1','E12':'.1','F12':'.4','G12':'1.6','E13':'.1','F13':'.2','G13':'1'};require(rounded==Decimal(published[cell]),'Cuadro14 published rounding mismatch');proof.update(pdf_display_value=published[cell],rounding='Decimal ROUND_HALF_UP to one decimal; signed zero compares numerically',auxiliary_levels_not_certified=True,source_metric=s.text(sh,'D5'),period_cell=col+'6',scenario_cell='C'+str(rr));metric=s.text(sh,'D5')
  else:
   require(row['parte_grafico_imagen']==p['part'],'Closure chart part changed');root=ET.fromstring(s.xml.parts[p['part']]);node=root.findall('.//c:ser',C)[int(row['serie_id'].split('_s')[1])-1];require(ET.tostring(node,encoding='unicode')==p['node_xml'],'Closure literal/name/ref/cache vector changed');require([ET.tostring(a,encoding='unicode')for a in root.findall('.//c:valAx',C)]==p['root_axes'],'Closure chart unit axis changed');m=re.fullmatch(re.escape(row['serie_id'])+r'_p([0-9]+)',row['id']or'');require(m is not None,'Closure point identity changed');i=int(m[1]);val=node.find('c:val/c:numRef',C);vp=points(val.find('c:numCache',C))if val is not None else points(node.find('c:val/c:numLit',C));require(i in vp and i<len(p['physical_value_cells']),'Closure point outside physical vector');cell=p['physical_value_cells'][i];raw=s.number(sh,cell);require(raw==Decimal(vp[i])==Decimal(row['decimal_XML_texto']),'Closure source/cache/universal raw mismatch');value_ref=p['value_ref'];category_ref=p['category_ref'];name_ref=p['name_ref']
   for j,a in enumerate(p['physical_value_cells']):
    n=s.number(sh,a)
    if j in vp:require(n is not None and n==Decimal(vp[j]),'Closure full physical vector disagrees with chart cache')
    else:require(n is None,'Closure chart omits physical numeric point')
   if p['physical_category_cells']:
    cc=p['physical_category_cells'];category=self.category(s,sh,cc[i]);cat=node.find('c:cat/c:numRef/c:numCache',C)
    if cat is None:cat=node.find('c:cat/c:strRef/c:strCache',C)
    if cat is None:cat=node.find('c:cat/c:numLit',C)
    if cat is None:cat=node.find('c:cat/c:strLit',C)
    cp=points(cat);require(all(j in cp and self.category(s,sh,cc[j])==cp[j]for j in vp),'Closure physical categories disagree with chart cache');require(row['periodo_categoria_literal']==category,'Closure universal category changed')
   else:
    require(row['periodo_categoria_literal']is None,'Closure scalar unexpectedly categorised');category=p['horizon'];proof['explicit_horizon']=p['horizon']
   label=s.text(sh,p['label_cell'])if p.get('label_cell')else row['serie_literal'];metric=row['metrica_literal'];proof.update(value_ref=value_ref,category_ref=category_ref,name_ref=name_ref,physical_value_cell=cell,physical_category_cell=p['physical_category_cells'][i]if p['physical_category_cells']else None,source_label_cell=p.get('label_cell'))
   parent=next(a for a in root.iter()if node in list(a));ids={a.get('val')for a in parent.findall('c:axId',C)};axes=[a for a in root.findall('.//c:valAx',C)if a.find('c:axId',C).get('val')in ids];require(len(axes)==1,'Closure ambiguous value axis');axis=axes[0];axis_id=axis.find('c:axId',C).get('val');axis_format=axis.find('c:numFmt',C).get('formatCode');require(axis.find('c:dispUnits',C)is None,'Closure unsupported display divisor');proof.update(axis_id=axis_id,axis_format=axis_format)
   if p['mode']=='nawru_literal':
    require(node.find('c:tx/c:v',C).text=='Previsión'and s.text(sh,'C6')=='Variable'and s.text(sh,'D6')=='Previsión','NAWRU bounded name conflict changed');marker=root.findall('.//c:ser',C)[0];mv=points(marker.find('c:val/c:numLit',C));require(all(Decimal(mv[j])==s.number(sh,'D'+str(7+j))for j in range(71)),'NAWRU marker D vector changed');require(set(vp)==set(range(71))and set(mv)==set(vp),'NAWRU vector domain changed');proof.update(bounded_name_override=dict(original_literal='Previsión',source_variable_header='Variable',resolved_metric=s.text(sh,'B3'),reason='unique full C vector is the NAWRU measure by source/PDF caption; D vector is separate plotting marker',forecast_classification_not_inferred=True));label=s.text(sh,'B3');metric=s.text(sh,'B3')
   if p['mode']=='debt_decomposition':
    require(s.text(sh,'B5')=='2008-2014'and cell in['C5','D5','E5'],'Only reviewed debt period eligible');v=[s.number(sh,a)for a in['C5','D5','E5']];require(abs(v[0]-v[1]-v[2])<=Decimal('1e-12'),'Debt period decomposition fails');role='published_period_debt_change_total'if cell=='C5'else'published_period_debt_change_contribution';proof['period_triplet']=dict(cells=['C5','D5','E5'],residual=str(v[0]-v[1]-v[2]),tolerance='1e-12')
   if sh=='Gráfico RE_2.4.':proof.update(first_registration_year={0:2024,1:2023,2:2022}[i],currency_price_year=2024,not_longitudinal_panel=True)
   if sh=='Gráfico RE 2.1':proof['not_error_bar_endpoint']=True
   if sh in['Gráfico 72','Gráfico 30']:
    total=s.number(sh,'C4'if row['book']=='2025'else'C7');parts=[s.number(sh,'C'+str(j))for j in([5,6]if row['book']=='2025'else range(8,14))];require(abs(total-sum(parts))<=Decimal('1e-12'),'Debt difference decomposition changed');role='published_debt_ratio_difference_total_not_extra_sum'if cell in['C4','C7']else'published_debt_ratio_difference_component';proof['total_not_added_as_component']=True
  require(raw is not None,'Closure missing raw source number');shadow=dict(row,hoja_valor=sh,celda_valor=cell);proof.update(unit=unit,multiplier=str(mult),divisor='1',original_value_sheet=row['hoja_valor'],original_value_cell=row['celda_valor']);out=self.result(shadow,raw,raw*mult,unit,metric,label,category,role,proof,point_index=i,value_ref=value_ref,category_ref=category_ref,name_ref=name_ref,axis_id=axis_id,axis_format=axis_format,multiplier=mult,binding_mode='closure_exact_published_source_contract',unit_text=unit,category_basis='explicit_source_cell_or_published_horizon',semantic_scope=p['scope']);d=dict(zip(FIELDS,out));d.update(universal_series=row['serie_literal'],universal_category=row['periodo_categoria_literal'],universal_value_sheet=row['hoja_valor'],universal_value_cell=row['celda_valor']);return tuple(d[n]for n in FIELDS)
 def review_closure(self,row):
  try:return 'usable_closure_family',self.point(row)
  except(ValueError,ArithmeticError,KeyError,IndexError,AttributeError,TypeError,StopIteration)as e:return 'closure_evidence_review:'+str(e),None
def make_quarantine(c,r):
 q=c.execute("SELECT * FROM indexed_observations WHERE book='2025'AND series_id IN('chart87_s1','chart87_s2')ORDER BY row_number");ns=[a[0]for a in q.description];old={a['row_number']:a for a in[dict(zip(ns,v))for v in q.fetchall()]};require(len(old)==98,'Approved withdrawal set changed');q=c.execute("SELECT * FROM staging_records WHERE book='2025'AND serie_id IN('chart87_s1','chart87_s2')ORDER BY row_number");ns=[a[0]for a in q.description];stage=[dict(zip(ns,v))for v in q.fetchall()];require(len(stage)==100,'Chart87 quarantine domain changed');out=[];s=r.sources['2025'];root=ET.fromstring(s.xml.parts['xl/charts/chart87.xml'])
 for a in stage:
  i=int(re.search(r'_p([0-9]+)$',a['id'])[1]);node=root.findall('.//c:ser',C)[int(a['serie_id'].split('_s')[1])-1];cref=node.find('c:cat/c:numRef/c:f',C).text;vref=node.find('c:val/c:numRef/c:f',C).text;require(cref=="'Gráfico 49'!$B$20:$AY$20"and vref in["'Gráfico 49'!$C$22:$AZ$22","'Gráfico 49'!$C$23:$AZ$23"],'Chart87 containment references changed');cs,cc=ref_cells(cref);vs,vc=ref_cells(vref);require(vc[i]==a['celda_valor']and s.number(vs,vc[i])==Decimal(a['decimal_XML_texto']),'Quarantine value changed');chart_year=r.category(s,cs,cc[i]);col=re.match('[A-Z]+',vc[i])[0];physical_year=s.text(vs,col+'20');require(chart_year is None if i==0 else int(chart_year)+1==physical_year,'Approved year conflict no longer matches');proof=dict(chart_year=chart_year,physical_column_year=str(physical_year),physical_column_year_cell=col+'20',category_ref=cref,value_ref=vref,raw_xml=a['decimal_XML_texto'],point_index=i,original_series=a['serie_literal'],no_automatic_realignment=True,chart86_untouched=True);out.append(('2025',a['row_number'],a['row_number']in old,chart_year,str(physical_year),cref,vref,a['celda_valor'],a['decimal_XML_texto'],dumps(proof)))
 return out
def make_overlays(r):
 s=r.sources['2026'];pdf=r.pdf('2026',47,['BASE MEDIA ANUAL','2024']);out=[]
 for i,cell in enumerate(['C7','D7','E7']):
  proof=dict(quantity_role='lower_edge_of_demographic_profile_range_not_mean_or_baseline_scenario',original_series='Base',source_cell=cell,original_raw=str(s.number('Gráfico RE_2.4.',cell)),first_registration_year=2024-i,currency_price_year=2024,not_longitudinal_panel=True,not_confidence_interval=True,derived_upper_edge_not_observation=True,pdf=pdf,legacy_unit_preserved='thousand_euros',safe_unit_context='thousand_current_2024_euros');out.append(('2026',1831+i,dumps(proof)))
 return out
def diagnostics(c,r,quarantine):
 out=[(b,n,'chart_physical_year_conflict',ev)for b,n,was,cy,py,cr,vr,cell,raw,ev in quarantine];q=c.execute("SELECT book,row_number,celda_valor,decimal_XML_texto FROM reviewed_family_numeric_queue WHERE book='2025'AND bloque='Cuadro 14'AND celda_valor IS NOT NULL ORDER BY row_number");records=q.fetchall();require(len(records)==50,'Cuadro14 incident domain changed');s=r.sources['2025'];basis=dict(metric=s.text('Cuadro 14','D5'),published_base=[str(s.number('Cuadro 14',a))for a in['D7','E7','F7','G7']],auxiliary_base=[str(s.number('Cuadro 14',a))for a in['I7','J7','K7','L7']],published22_verified_to_pdf154=True,auxiliary28_not_certified=True,no_auxiliary_reconstruction=True)
 out.extend((b,n,'published_auxiliary_interest_rate_base_conflict',dumps(dict(basis,source_cell=cell,raw_xml=raw)))for b,n,cell,raw in records)
 v=[s.number('Gráfico 64',a)for a in['C6','D6','E6']];res=v[0]-v[1]-v[2];require(abs(res)>Decimal('1e-12'),'Unreviewed debt-period conflict changed');q=c.execute("SELECT row_number,celda_valor FROM staging_records WHERE book='2025'AND bloque='Gráfico 64'AND celda_valor IN('C6','D6','E6')");out.extend(('2025',n,'debt_period_decomposition_conflict',dumps(dict(period='2015-2024',source_cell=cell,residual=str(res),raw_cells=['C6','D6','E6'],raw_values=[str(x)for x in v],values_not_balanced=True)))for n,cell in q.fetchall())
 q=c.execute("SELECT row_number FROM staging_records WHERE book='2026'AND serie_id='chart25_s2'");out.extend(('2026',n,'bounded_nawru_source_name_override',dumps(dict(literal_series='Previsión',physical_variable_header='Variable',metric_header=r.sources['2026'].text('Gráfico 17','B3'),source_measure_vector='C7:C77',separate_marker_vector='D7:D77',pdf_page=60,no_general_alias=True)))for(n,)in q.fetchall());s26=r.sources['2026'];v=[s26.number('Gráfico 30','C'+str(i))for i in range(7,14)];res=v[0]-sum(v[1:]);require(abs(res)>Decimal('1e-12'),'Debt2050 period conflict changed');q=c.execute("SELECT row_number,celda_valor FROM reviewed_family_numeric_queue WHERE book='2026'AND bloque='Gráfico 30'");out.extend(('2026',n,'debt_2050_decomposition_conflict',dumps(dict(period='2050',source_cell=cell,total_cell='C7',component_cells=['C8','C9','C10','C11','C12','C13'],raw_values=[str(x)for x in v],residual=str(res),no_value_balancing=True)))for n,cell in q.fetchall());require(len(out)==231,'Closure incident count changed');return out
def collect(c):
 require(sha(PREVIOUS)==PREVIOUS_SHA,'Previous database changed');require(c.execute('SELECT count(*)FROM reviewed_family_usable_keys').fetchone()[0]==42462,'Previous usable count changed');inv=load_inventory();q=c.execute('SELECT * FROM reviewed_family_numeric_queue ORDER BY book,row_number');ns=[a[0]for a in q.description];queue={};
 for v in q.fetchall():
  a=dict(zip(ns,v));queue[a['book'],a['row_number']]=a
 require(len(queue)==4751,'Previous pending domain changed');r=ClosureResolver();rows=[];review=[]
 try:
  for d in inv:
   if not d['original_queue']:continue
   a=queue[d['book'],d['row_number']];require(digest(a)==d['payload_sha256'],'Closure original payload changed')
   if d['disposition']=='usable':
    status,result=r.review_closure(a);review.append((a['book'],a['row_number'],status))
    if result is not None:rows.append(result)
  quarantine=make_quarantine(c,r);overlays=make_overlays(r);diag=diagnostics(c,r,quarantine);pages=[(b,p,PDF_HASHES[b],t)for(b,p),t in sorted(r.pdf_pages.items())]
 finally:r.close()
 decisions=[(a['book'],a['row_number'],a['book']+':'+str(a['row_number']),a['block'],a['series_id'],a['disposition'],a['reason'],dumps(a['evidence']),a['missing_evidence'],a['next_action'],a['payload_sha256'],a['source_xlsx_sha256'],a['source_pdf_sha256'],a['coverage_complete'],a['original_queue'],dumps(a))for a in inv];return rows,review,pages,diag,decisions,quarantine,overlays

def expressions():
 main="SELECT book,row_number FROM reviewed_family_usable_keys EXCEPT SELECT book,row_number FROM closure_quarantine WHERE was_admitted UNION ALL SELECT book,row_number FROM closure_family_observations"
 queue="SELECT s.*,CASE WHEN q.was_admitted THEN 'chart_physical_year_conflict' ELSE p.reviewed_family_reason END closure_family_reason FROM staging_records s JOIN(SELECT book,row_number FROM reviewed_family_usable_keys UNION SELECT book,row_number FROM reviewed_family_numeric_queue)u USING(book,row_number)LEFT JOIN reviewed_family_numeric_queue p USING(book,row_number)LEFT JOIN closure_quarantine q USING(book,row_number)WHERE NOT EXISTS(SELECT 1 FROM closure_family_usable_keys k WHERE k.book=s.book AND k.row_number=s.row_number)"
 union=' UNION ALL BY NAME '.join("SELECT *,'"+t+"' source_layer FROM "+t for t in BASE_OBSERVATION_TABLES+['closure_family_observations']);public='SELECT a.* FROM('+union+')a JOIN closure_family_usable_keys k USING(book,row_number)';return[('closure_family_usable_keys',main),('closure_family_numeric_queue',queue),('closure_family_public_observations',public)]
def build(output):
 output=Path(output).resolve();require(output.is_relative_to(ROOT)and not output.exists(),'Output must be a new project directory');require(sha(PREVIOUS)==PREVIOUS_SHA,'Previous database changed');output.mkdir(parents=True);db=output/'closure_families.duckdb';shutil.copyfile(PREVIOUS,db)
 with duckdb.connect(str(db))as c:
  require(not c.execute("SELECT table_name FROM information_schema.tables WHERE table_name LIKE 'closure%'").fetchall(),'Closure namespace collision');data=collect(c);rows,review,pages,diag,decisions,quarantine,overlays=data;c.execute('CREATE TABLE closure_family_observations AS SELECT * FROM reviewed_family_observations WHERE false');c.execute('ALTER TABLE closure_family_observations ADD PRIMARY KEY(book,row_number)')
  if rows:c.executemany('INSERT INTO closure_family_observations VALUES('+','.join('?'for _ in FIELDS)+')',rows)
  schemas=[('closure_family_review','book VARCHAR,row_number INTEGER,status VARCHAR',review),('closure_family_pdf_evidence','book VARCHAR,page INTEGER,pdf_sha256 VARCHAR,text VARCHAR',pages),('closure_family_diagnostics','book VARCHAR,row_number INTEGER,status VARCHAR,evidence VARCHAR',diag),('closure_family_decisions','book VARCHAR,row_number INTEGER,sourcekey VARCHAR,block VARCHAR,series_id VARCHAR,disposition VARCHAR,reason VARCHAR,evidence VARCHAR,missing_evidence VARCHAR,next_action VARCHAR,payload_sha256 VARCHAR,source_xlsx_sha256 VARCHAR,source_pdf_sha256 VARCHAR,coverage_complete BOOLEAN,original_queue BOOLEAN,decision_json VARCHAR',decisions),('closure_quarantine','book VARCHAR,row_number INTEGER,was_admitted BOOLEAN,chart_year VARCHAR,physical_column_year VARCHAR,category_ref VARCHAR,value_ref VARCHAR,source_cell VARCHAR,raw_xml VARCHAR,evidence VARCHAR',quarantine),('closure_metadata_overlays','book VARCHAR,row_number INTEGER,evidence VARCHAR',overlays)]
  for name,schema,values in schemas:
   pk='book,page'if name=='closure_family_pdf_evidence'else'book,row_number';c.execute('CREATE TABLE '+name+'('+schema+',PRIMARY KEY('+pk+'))')
   if values:c.executemany('INSERT INTO '+name+' VALUES('+','.join('?'for _ in values[0])+')',values)
  for name,sql in expressions():c.execute('CREATE VIEW '+name+' AS '+sql)
 report=dict(status='built_pending_validation',added=len(rows),withdrawn=98,admitted=42462-98+len(rows),numeric_pending=4751+98-len(rows),review_domain=4849,decision_counts=dict(usable=145,excluded=421,internal_validated=168,blocked=4115),new_literal_diagnostics=len(diag),previous_sha256=PREVIOUS_SHA,inventory_sha256=INVENTORY_SHA,code_sha256=sha(Path(__file__)),db_sha256=sha(db),internal_excluded_from_public=True,legacy_queries_not_current_safe_api=True,rag_integrated=False);(output/'manifest.json').write_text(dumps(report),encoding='utf8');return report
def verify_exclusions(c,inv):
 r=SourceResolver();counts={}
 try:
  for a in inv:
   if a['disposition']!='excluded':continue
   sh=a['block'];sid=a['series_id'];b=a['book'];s=r.sources[b];reason=a['queue_reason'];counts[reason]=counts.get(reason,0)+1
   if 'duplicate_pair'in a:
    d=a['duplicate_pair'];v=c.execute('SELECT metric,series_literal,category_literal,unit,raw_value,scope,role,source_sheet,source_cell FROM table_observations WHERE book=?AND row_number=?',[d['canonical_book'],d['canonical_row_number']]).fetchone();require(v is not None,'Excluded duplicate canonical missing');require(v[0]==d['metric_alias'][1]and v[1]==d['series']and v[2]==d['category']and v[3]==d['unit_alias'][1]and v[4]==Decimal(d['raw_decimal'])and v[5]==d['canonical_scope']and v[6]==d['canonical_role']and(v[7],v[8])==(d['source_sheet'],d['source_cell']),'Excluded duplicate semantic payload changed');require(c.execute('SELECT count(*)FROM reviewed_family_usable_keys WHERE book=?AND row_number=?',[b,d['canonical_row_number']]).fetchone()[0]==1,'Excluded duplicate canonical not admitted')
   elif b=='2025'and sid in['chart55_s2','chart56_s2']:
    require(s.text(sh,'F4')=='Auxiliar1'and s.text(sh,'G4')=='Auxiliar2','Diagonal helper source labels changed');require(all(s.number(sh,'F'+str(i))==s.number(sh,'G'+str(i))for i in range(5,36)),'Diagonal helper no longer diagonal')
   elif b=='2026'and sh=='Gráfico 15':require(s.text(sh,'E6')=='Cumulative'and s.text(sh,'G6')=='Before'and s.text(sh,'H6')=='After','Waterfall template identity changed');require(abs(s.number(sh,'E8')-s.number(sh,'D7')-s.number(sh,'D8'))<=Decimal('1e-14'),'Waterfall template cumulative relation changed')
   elif b=='2026'and sh=='Gráfico RE_2.4.':
    root=ET.fromstring(s.xml.parts['xl/charts/chart17.xml']);require(root.find('.//c:areaChart/c:grouping',C).get('val')=='stacked'and s.text(sh,'B7')=='Base'and s.text(sh,'B8')=='Rango perfiles demográficos'and s.text(sh,'B9')=='Media ponderada','Demographic range thickness source role changed')
   else:require(reason in['forecast_marker_not_measure','drawing_helper_not_economic_series'],'Excluded role lacks positive source proof')
 finally:r.close()
 require(sum(counts.values())==421,'Excluded count changed')
def check_new(c):
 rows,review,pages,diag,decisions,quarantine,overlays=collect(c);require(len(rows)==145,'Closure approved admissions incomplete');verify_exclusions(c,load_inventory())
 for table,expected in[('closure_family_observations',rows),('closure_family_review',review),('closure_family_pdf_evidence',pages),('closure_family_diagnostics',diag),('closure_family_decisions',decisions),('closure_quarantine',quarantine),('closure_metadata_overlays',overlays)]:
  actual=c.execute('SELECT * FROM '+table).fetchall();require(len(actual)==len(expected)and set(actual)==set(expected),'Closure persisted evidence mismatch: '+table)
 for name,sql in expressions():
  require(c.execute('SELECT * FROM '+name+' LIMIT0'.replace('LIMIT0','LIMIT 0')).description==c.execute(sql+' LIMIT 0').description,'Closure view columns changed')
  for a,b in[(name,'('+sql+')'),('('+sql+')',name)]:require(not c.execute('SELECT * FROM '+a+' EXCEPT ALL SELECT * FROM '+b).fetchall(),'Closure view content/multiplicity changed')
 require(c.execute('SELECT count(*)FROM closure_family_usable_keys').fetchone()[0]==42509 and c.execute('SELECT count(*)FROM closure_family_numeric_queue').fetchone()[0]==4704,'Closure candidate counts changed');require(not c.execute('SELECT * FROM closure_family_usable_keys INTERSECT SELECT book,row_number FROM closure_family_numeric_queue').fetchall(),'Closure public/pending overlap');require(c.execute('SELECT count(*)FROM(SELECT * FROM closure_family_usable_keys UNION SELECT book,row_number FROM closure_family_numeric_queue)').fetchone()[0]==47213,'Closure universe changed');require(not c.execute('SELECT * FROM closure_family_usable_keys GROUP BY ALL HAVING count(*)<>1').fetchall(),'Closure admission collision');require(c.execute('SELECT count(*)FROM closure_family_public_observations').fetchone()[0]==42509 and not c.execute('SELECT book,row_number FROM closure_family_public_observations GROUP BY ALL HAVING count(*)<>1').fetchall(),'Safe public source multiplicity changed');require(not c.execute('SELECT book,row_number FROM closure_family_public_observations EXCEPT SELECT * FROM closure_family_usable_keys').fetchall(),'Unsafe public key');require(not c.execute('SELECT * FROM closure_family_usable_keys EXCEPT SELECT book,row_number FROM closure_family_public_observations').fetchall(),'Missing safe public key');require(not c.execute('SELECT * FROM remaining_internal_keys INTERSECT SELECT * FROM closure_family_usable_keys').fetchall(),'Internal model input publicly admitted');require(not c.execute('SELECT book,row_number FROM ready_quarantine INTERSECT SELECT * FROM closure_family_usable_keys').fetchall(),'Legacy quarantine publicly admitted');require(not c.execute('SELECT book,row_number FROM closure_quarantine INTERSECT SELECT * FROM closure_family_usable_keys').fetchall(),'New quarantine publicly admitted');require(c.execute('SELECT count(*)FROM closure_quarantine WHERE was_admitted').fetchone()[0]==98,'Withdrawal count changed');return rows

def validate(db):
 previous=validate_previous(db)
 with duckdb.connect(str(db),read_only=True)as c:rows=check_new(c);stats=c.execute('SELECT book,block,unit,quantity_role,count(*)FROM closure_family_observations GROUP BY ALL ORDER BY book,block').fetchall();counts=dict(c.execute('SELECT disposition,count(*)FROM closure_family_decisions GROUP BY ALL').fetchall())
 return dict(status='pass',previous_validation=previous['status'],added=len(rows),withdrawn=98,admitted=42509,numeric_pending=4704,stats=stats,decision_counts=counts,review_domain=4849,coverage_complete=True,internal_verified_separately=168,new_literal_diagnostics=231,previous_sha256=PREVIOUS_SHA,inventory_sha256=INVENTORY_SHA,db_sha256=sha(Path(db)),code_sha256=sha(Path(__file__)),legacy_queries_not_current_safe_api=True,rag_integrated=False)
def query_safe(db,*,book,row_number):
 require(book in['2025','2026']and type(row_number)is int and row_number>0,'Exact safe source key required')
 with duckdb.connect(str(db),read_only=True)as c:
  q=c.execute('SELECT p.*,s.xlsx_path,s.xlsx_sha256,s.pdf_path,s.pdf_sha256 FROM closure_family_public_observations p JOIN sources s USING(book)WHERE book=?AND row_number=?',[book,row_number]);ns=[a[0]for a in q.description];data=q.fetchall();overlay=c.execute('SELECT evidence FROM closure_metadata_overlays WHERE book=?AND row_number=?',[book,row_number]).fetchone()
 if len(data)!=1:raise LookupError('No unique currently safe economic observation; no legacy fallback')
 result={n:str(v)if isinstance(v,Decimal)else v for n,v in zip(ns,data[0])};result['safe_current_view']=True
 if overlay is not None:result['metadata_overlay']=json.loads(overlay[0]);result['safe_quantity_role']=result['metadata_overlay']['quantity_role'];result['safe_unit_context']=result['metadata_overlay']['safe_unit_context']
 if result['unit']=='thousand_persons_signed_plot':
  proof=json.loads(result['quantity_evidence']);result.update(population_thousands=proof['population_thousands'],plot_value_thousands=proof['plot_value_thousands'],sex=proof['sex'],reference_year=proof['reference_year'],age_group=proof['age_group'],population_scope_literal=proof['population_scope_literal'],value_is_population_count=False)
 return result
def query(db,*,book,block,series_id,point_index):
 require(book in['2025','2026']and all(isinstance(x,str)and x.strip()for x in[block,series_id])and type(point_index)is int and point_index>=0,'Exact new chart identity required')
 with duckdb.connect(str(db),read_only=True)as c:rows=c.execute('SELECT row_number FROM closure_family_observations JOIN closure_family_usable_keys USING(book,row_number)WHERE book=?AND block=?AND series_id=?AND point_index=?',[book,block,series_id,point_index]).fetchall()
 if len(rows)!=1:raise LookupError('No unique new chart point; no fallback')
 return query_safe(db,book=book,row_number=rows[0][0])
def query_table(db,*,book,block,row_number):
 require(book in['2025','2026']and isinstance(block,str)and block.strip()and type(row_number)is int and row_number>0,'Exact new table key required')
 with duckdb.connect(str(db),read_only=True)as c:rows=c.execute('SELECT row_number FROM closure_family_observations JOIN closure_family_usable_keys USING(book,row_number)WHERE book=?AND block=?AND row_number=?AND series_id IS NULL',[book,block,row_number]).fetchall()
 if len(rows)!=1:raise LookupError('No unique new table point; no fallback')
 return query_safe(db,book=book,row_number=row_number)
def main():
 p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='command',required=True);b=s.add_parser('build');b.add_argument('--output-dir',type=Path,required=True);b=s.add_parser('validate');b.add_argument('--db',type=Path,required=True);b.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.command=='build':report=build(a.output_dir)
 else:
  require(not a.output.exists(),'Report exists')
  try:report=validate(a.db)
  except Exception as e:report=dict(status='fail',error=str(e))
  a.output.write_text(dumps(report),encoding='utf8')
 print(dumps(report));return int(report['status']=='fail')
if __name__=='__main__':raise SystemExit(main())
