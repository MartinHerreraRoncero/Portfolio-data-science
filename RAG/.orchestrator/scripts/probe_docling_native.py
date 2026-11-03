import json, time
from pathlib import Path
from docling.datamodel.base_models import InputFormat, ConversionStatus
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableStructureOptions, TableFormerMode
from docling.document_converter import DocumentConverter, PdfFormatOption
folder = Path('evaluation/document_layout_2026-10-27')
opts = PdfPipelineOptions(do_ocr=False, do_table_structure=True, enable_remote_services=False, artifacts_path=Path('.cache-docling/models'), accelerator_options={'device': 'cpu', 'num_threads': 4}, table_structure_options=TableStructureOptions(mode=TableFormerMode.ACCURATE, do_cell_matching=True))
converter = DocumentConverter(allowed_formats=[InputFormat.PDF], format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)})
started = time.perf_counter()
r = converter.convert(Path('Airef/Opinion sobre la Sostenibilidad de las AAPP a largo plazo Marzo 2025.pdf'), page_range=(95, 95))
assert r.status == ConversionStatus.SUCCESS, r.status
r.document.save_as_json(folder / 'probe95.native.json')
(folder / 'probe95.md').write_text(r.document.export_to_markdown(), encoding='utf-8')
summary = dict(status=str(r.status), wall_seconds=time.perf_counter()-started, pages=list(r.document.pages), tables=[dict(ref=t.self_ref, rows=t.data.num_rows, cols=t.data.num_cols, cells=len(t.data.table_cells), pages=[p.page_no for p in t.prov]) for t in r.document.tables])
(folder / 'probe95.summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
print(json.dumps(summary))
