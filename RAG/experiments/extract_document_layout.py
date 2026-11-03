"""CLI opt-in para extracción reproducible de layout y tablas estructuradas."""

import argparse
from pathlib import Path
import sys

from rag_pensiones.document_layout import extract_document, write_bundle


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Extracción opt-in de layout de documentos y tablas con Docling y perfiles revisados."
    )
    parser.add_argument("--input", required=True, type=Path, help="Ruta al archivo PDF original.")
    parser.add_argument("--published-at", required=True, type=str, help="Fecha ISO de publicación (YYYY-MM-DD).")
    parser.add_argument("--start-page", type=int, default=1, help="Página inicial física (1-indexed).")
    parser.add_argument("--end-page", type=int, default=None, help="Página final física (1-indexed).")
    parser.add_argument(
        "--artifacts-path",
        type=Path,
        default=None,
        help="Ruta local con modelos de Docling en caché para modo offline.",
    )
    parser.add_argument(
        "--regions",
        type=Path,
        default=None,
        help="Ruta opcional al archivo JSON de regiones geométricas revisadas.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Carpeta exclusiva para los artefactos generados.",
    )
    parser.add_argument(
        "--text-view",
        choices=["legacy", "layout"],
        default="legacy",
        help="Vista textual para pages.jsonl (legacy por defecto, layout como candidata).",
    )
    parser.add_argument(
        "--no-duckdb",
        action="store_true",
        help="Omite la generación de la base de datos DuckDB tables.duckdb.",
    )
    
    args = parser.parse_args(argv)
    
    input_path = args.input.resolve()
    if not input_path.exists():
        print(f"Error: input file not found at {input_path}", file=sys.stderr)
        return 1
    
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        print(
            f"Error: output directory already exists and is not empty: {output_dir}. Refusing to overwrite.",
            file=sys.stderr,
        )
        return 1
    
    try:
        print(f"Iniciando extracción local de layout para {input_path.name}...")
        bundle = extract_document(
            pdf_path=input_path,
            published_at=args.published_at,
            start_page=args.start_page,
            end_page=args.end_page,
            artifacts_path=args.artifacts_path,
            regions_path=args.regions,
        )
        
        result = write_bundle(
            bundle=bundle,
            output_dir=output_dir,
            with_duckdb=not args.no_duckdb,
            text_view=args.text_view,
        )
        print(f"Extracción completada exitosamente en {result['output_dir']}")
        print(f"Archivos generados: {list(result['files'].keys())}")
        return 0
    except Exception as exc:
        print(f"Error durante la extracción: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
