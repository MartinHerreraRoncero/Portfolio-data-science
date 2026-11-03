"""Static analysis of catalog source object sizes and distribution.

Distinguishes semantic catalog card rows from Excel physical rows:
- Excel physical row: Raw grid row in an official spreadsheet containing
  individual observations / period cells.
- Catalog card row: A unified semantic series (CatalogCard) representing an
  entire temporal trajectory for a specific metric, series, unit, and scope.
- Source object: A structured tabular or chart entity (identified by book,
  source_path, and block) containing multiple card rows.
"""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path

from rag_pensiones.graph_catalog import _source_object_key
from rag_pensiones.structured_retrieval import ROOT


def percentile(data: list[int | float], p: float) -> float:
    """Compute percentile using linear interpolation between closest ranks."""
    if not data:
        return 0.0
    k = (len(data) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(data[int(k)])
    return float(data[f] * (c - k) + data[c] * (k - f))


def compact_card_metadata(card: dict) -> dict:
    """Compact metadata representation using period range and count, never observation values."""
    periods = card.get('periods', [])
    return dict(
        card_id=card['card_id'],
        book=card.get('book'),
        block=card.get('block'),
        metric=card.get('metric'),
        series=card.get('series'),
        unit=card.get('unit'),
        period_kind=card.get('period_kind'),
        quantity_role=card.get('quantity_role'),
        scope=card.get('scope'),
        title=card.get('title'),
        period_count=len(periods),
        period_range=[min(periods), max(periods)] if periods else [],
        source_path=card.get('source_path'),
    )


def compute_source_object_sizes(catalog: dict) -> dict:
    cards = catalog.get('cards', [])
    grouped = defaultdict(list)
    for card in cards:
        k = _source_object_key(card)
        grouped[k].append(card)

    counts = sorted([len(group) for group in grouped.values()])
    total_objs = len(grouped)
    total_card_rows = len(cards)

    p50 = percentile(counts, 50)
    p90 = percentile(counts, 90)
    p95 = percentile(counts, 95)
    max_val = max(counts) if counts else 0
    min_val = min(counts) if counts else 0
    mean_val = (sum(counts) / total_objs) if total_objs else 0.0

    sorted_groups = sorted(
        grouped.items(),
        key=lambda item: (-len(item[1]), repr(item[0]))
    )

    top15 = []
    for rank, (k, group_cards) in enumerate(sorted_groups[:15], 1):
        compact_meta = [compact_card_metadata(c) for c in group_cards]
        meta_json = json.dumps(compact_meta, ensure_ascii=False, separators=(',', ':'))
        meta_bytes = len(meta_json.encode('utf-8'))

        book = group_cards[0].get('book')
        path = group_cards[0].get('source_path')
        block = group_cards[0].get('block')

        top15.append(dict(
            rank=rank,
            book=book,
            path=path,
            block=block,
            card_row_count=len(group_cards),
            metadata_utf8_bytes=meta_bytes,
        ))

    report = dict(
        report_name="source_object_sizes",
        date="2026-11-03",
        concept_distinction=dict(
            card_row="A semantic catalog series card (CatalogCard) representing an entire time-series for a specific metric, series, unit, and scope within a source object block.",
            excel_physical_row="A physical row in an official source Excel spreadsheet containing individual observations/periods. Multiple observation rows reconcile into card rows.",
            source_object="A certified tabular or graphical block within a source document, grouping related semantic card rows."
        ),
        total_source_objects=total_objs,
        total_card_rows=total_card_rows,
        catalog_hash=catalog.get('catalog_hash'),
        metadata_size_note='Unpacked descriptive metadata estimate, not actual compressed model request size.',
        stats=dict(
            count=total_objs,
            p50=round(p50, 2),
            p90=round(p90, 2),
            p95=round(p95, 2),
            max=max_val,
            min=min_val,
            mean=round(mean_val, 2),
        ),
        top15=top15,
    )
    return report


def main():
    parser = argparse.ArgumentParser(description="Generate static catalog source object sizes report.")
    parser.add_argument("--output", default="experiments/results/source_object_sizes_2026-11-03.json",
                        help="Path to write the output JSON report.")
    args = parser.parse_args()

    catalog_path = ROOT / "data/processed/langgraph_9b_assets/catalog.json"
    if catalog_path.is_file():
        with catalog_path.open("r", encoding="utf-8") as f:
            catalog = json.load(f)
    else:
        from rag_pensiones.graph_catalog import build_catalog
        catalog = build_catalog()

    report = compute_source_object_sizes(catalog)
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Report written to {output_path}")


if __name__ == "__main__":
    main()
