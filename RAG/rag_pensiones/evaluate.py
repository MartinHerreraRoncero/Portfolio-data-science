"""Evaluación de recuperación semántica previa a la generación sobre preguntas de desarrollo."""

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import re
import sys

from rag_pensiones.embeddings import OllamaEmbedder
from rag_pensiones.qwen_embeddings import QwenEmbedder, MODEL as QWEN_EMBEDDING_MODEL
from rag_pensiones.ingest import PROJECT_ROOT, validate_project_path
from rag_pensiones.search import (
    DEFAULT_INDEX_PATH,
    SearchHit,
    load_index,
    search,
    validate_index,
)

DEFAULT_DATASET_PATH = Path("evaluation/pensiones_dev.json")
DATASET_SCHEMA_VERSION = "retrieval-dev-v1"
EVALUATION_SCHEMA_VERSION = "retrieval-eval-v1"


def _reject_constant(val: str) -> None:
    raise ValueError(f"Invalid constant {val!r} found in JSON")


def collapse_spaces(text: str) -> str:
    """Colapsa secuencias de espacios en blanco a un único espacio sin alterar mayúsculas ni signos."""
    return re.sub(r"\s+", " ", text).strip()


def compute_canonical_sha256(data: dict) -> str:
    """Calcula el hash SHA-256 canónico de un diccionario JSON ordenado y compacto."""
    canonical_bytes = json.dumps(
        data,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical_bytes).hexdigest()


def validate_dataset(dataset: dict) -> None:
    """Valida exhaustivamente el esquema, metadatos y consistencia del dataset de evaluación."""
    if not isinstance(dataset, dict):
        raise ValueError(f"Dataset must be a JSON object (dict), got {type(dataset).__name__}")
    json.dumps(dataset, allow_nan=False)

    schema_version = dataset.get("schema_version")
    if schema_version != DATASET_SCHEMA_VERSION:
        raise ValueError(
            f"Invalid schema_version: expected {DATASET_SCHEMA_VERSION!r}, got {schema_version!r}"
        )

    desc = dataset.get("description")
    if not isinstance(desc, str) or not desc.strip():
        raise ValueError("description must be a non-empty string")

    cases = dataset.get("cases")
    if not isinstance(cases, list) or len(cases) == 0:
        raise ValueError("cases must be a non-empty list")

    seen_ids: set[str] = set()
    required_keys = {"id", "question", "answerable", "expected_answer", "evidence"}
    required_evidence_keys = {"doc_id", "page_number", "quotes"}

    for idx, c in enumerate(cases):
        if not isinstance(c, dict):
            raise ValueError(f"Case at index {idx} must be a dict, got {type(c).__name__}")

        missing = required_keys - set(c.keys())
        if missing:
            raise ValueError(f"Case at index {idx} missing required fields: {sorted(missing)}")

        case_id = c["id"]
        if not isinstance(case_id, str) or not case_id.strip():
            raise ValueError(f"Case at index {idx} has invalid id: {case_id!r}")
        if case_id in seen_ids:
            raise ValueError(f"Duplicate case id detected at index {idx}: {case_id!r}")
        seen_ids.add(case_id)

        question = c["question"]
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"Case {case_id!r} has empty or non-string question")

        answerable = c["answerable"]
        if type(answerable) is not bool:
            raise ValueError(
                f"Case {case_id!r} field 'answerable' must be strictly a boolean, got {answerable!r}"
            )

        expected_answer = c["expected_answer"]
        if not isinstance(expected_answer, str) or not expected_answer.strip():
            raise ValueError(f"Case {case_id!r} has empty or non-string expected_answer")

        evidence = c["evidence"]
        if not isinstance(evidence, list):
            raise ValueError(f"Case {case_id!r} field 'evidence' must be a list")

        if answerable:
            if len(evidence) == 0:
                raise ValueError(
                    f"Case {case_id!r} is answerable (answerable=True) but has an empty evidence list"
                )
            for ev_idx, ev in enumerate(evidence):
                if not isinstance(ev, dict):
                    raise ValueError(
                        f"Evidence {ev_idx} in case {case_id!r} must be a dict, got {type(ev).__name__}"
                    )
                ev_missing = required_evidence_keys - set(ev.keys())
                if ev_missing:
                    raise ValueError(
                        f"Evidence {ev_idx} in case {case_id!r} missing required fields: {sorted(ev_missing)}"
                    )

                doc_id = ev["doc_id"]
                if not isinstance(doc_id, str) or not doc_id.strip():
                    raise ValueError(
                        f"Evidence {ev_idx} in case {case_id!r} has invalid doc_id: {doc_id!r}"
                    )

                page_num = ev["page_number"]
                if type(page_num) is not int or page_num < 1:
                    raise ValueError(
                        f"Evidence {ev_idx} in case {case_id!r} has invalid page_number (must be positive int, not bool): {page_num!r}"
                    )

                quotes = ev["quotes"]
                if not isinstance(quotes, list) or len(quotes) == 0:
                    raise ValueError(
                        f"Evidence {ev_idx} in case {case_id!r} must have a non-empty list of quotes"
                    )
                for q_idx, quote in enumerate(quotes):
                    if not isinstance(quote, str) or not quote.strip():
                        raise ValueError(
                            f"Quote {q_idx} in evidence {ev_idx} of case {case_id!r} must be a non-empty string"
                        )
        else:
            if len(evidence) != 0:
                raise ValueError(
                    f"Case {case_id!r} requires abstention (answerable=False) and must have an empty evidence list, got {len(evidence)} items"
                )


def load_dataset(path: Path) -> dict:
    """Carga y valida un dataset de evaluación JSON respetando los límites del proyecto."""
    resolved_path = validate_project_path(Path(path))
    if not resolved_path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {resolved_path}")

    raw_bytes = resolved_path.read_bytes()
    if not raw_bytes.strip():
        raise ValueError("Dataset file is empty")

    try:
        text = raw_bytes.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError(f"Dataset file is not valid UTF-8: {exc}") from exc

    try:
        data = json.loads(text, parse_constant=_reject_constant)
    except Exception as exc:
        raise ValueError(f"Malformed JSON in dataset file: {exc}") from exc

    validate_dataset(data)
    return data


def evaluate(
    index: dict,
    dataset: dict,
    top_k: int = 3,
    client: OllamaEmbedder | None = None,
) -> dict:
    """Evalúa la recuperación semántica sobre un dataset de desarrollo y calcula métricas."""
    if type(top_k) is not int or top_k <= 0:
        raise ValueError(f"top_k must be a positive integer (not bool), got {top_k!r}")

    # 1. Validaciones previas completas antes de contactar con la red
    validate_index(index)
    validate_dataset(dataset)

    # 2. Verificar que todo documento/página citado en las evidencias exista en el índice
    index_doc_pages = {(c["doc_id"], c["page_number"]) for c in index["chunks"]}
    for case in dataset["cases"]:
        if case["answerable"]:
            for ev in case["evidence"]:
                doc_page_key = (ev["doc_id"], ev["page_number"])
                if doc_page_key not in index_doc_pages:
                    raise ValueError(
                        f"Gold evidence in case {case['id']!r} references document {ev['doc_id']!r} "
                        f"page {ev['page_number']}, which is not present in the index"
                    )

    if client is None:
        client = QwenEmbedder() if index["model"] == QWEN_EMBEDDING_MODEL else OllamaEmbedder()

    # Pre-indexar fragmentos del corpus colapsados para verificar disponibilidad gold
    corpus_chunks_collapsed: list[tuple[str, int, str]] = [
        (c["doc_id"], c["page_number"], collapse_spaces(c["text"]))
        for c in index["chunks"]
    ]

    chunking_strategy = index["chunks"][0].get("strategy", "unknown")
    dataset_sha256 = compute_canonical_sha256(dataset)

    n_answerable = 0
    n_unanswerable = 0
    hit_at_1_sum = 0
    hit_at_k_sum = 0
    mrr_at_k_sum = 0.0

    case_reports: list[dict] = []

    for case in dataset["cases"]:
        hits = search(
            index=index,
            query=case["question"],
            top_k=top_k,
            client=client,
        )

        if case["answerable"]:
            n_answerable += 1

            # Comprobar si existe al menos un chunk en el corpus que contenga TODAS las citas
            # de al menos una evidencia alternativa (disponibilidad de gold)
            gold_available = False
            for ev in case["evidence"]:
                ev_quotes_collapsed = [collapse_spaces(q) for q in ev["quotes"]]
                for c_doc, c_page, c_text in corpus_chunks_collapsed:
                    if c_doc == ev["doc_id"] and c_page == ev["page_number"]:
                        if all(q in c_text for q in ev_quotes_collapsed):
                            gold_available = True
                            break
                if gold_available:
                    break

            first_relevant_rank: int | None = None
            if gold_available:
                for hit in hits:
                    hit_chunk = hit.chunk
                    hit_text_collapsed = collapse_spaces(hit_chunk.text)
                    hit_is_relevant = False
                    for ev in case["evidence"]:
                        if (
                            hit_chunk.doc_id == ev["doc_id"]
                            and hit_chunk.page_number == ev["page_number"]
                        ):
                            ev_quotes_collapsed = [collapse_spaces(q) for q in ev["quotes"]]
                            if all(q in hit_text_collapsed for q in ev_quotes_collapsed):
                                hit_is_relevant = True
                                break
                    if hit_is_relevant:
                        first_relevant_rank = hit.rank
                        break

            hit_at_1_val = 1 if first_relevant_rank == 1 else 0
            hit_at_k_val = 1 if (first_relevant_rank is not None and first_relevant_rank <= top_k) else 0
            rr_val = (1.0 / first_relevant_rank) if (first_relevant_rank is not None and first_relevant_rank <= top_k) else 0.0

            hit_at_1_sum += hit_at_1_val
            hit_at_k_sum += hit_at_k_val
            mrr_at_k_sum += rr_val

            case_reports.append({
                "id": case["id"],
                "question": case["question"],
                "answerable": True,
                "expected_answer": case["expected_answer"],
                "gold_available": gold_available,
                "first_relevant_rank": first_relevant_rank,
                "hit_at_1": hit_at_1_val,
                "hit_at_k": hit_at_k_val,
                "reciprocal_rank": rr_val,
                "hits": [asdict(h) for h in hits],
            })
        else:
            n_unanswerable += 1
            case_reports.append({
                "id": case["id"],
                "question": case["question"],
                "answerable": False,
                "expected_answer": case["expected_answer"],
                "gold_available": None,
                "status": "not_evaluated_no_generator",
                "reason": (
                    "Pregunta de abstención: la evaluación de abstención requiere el generador (LLM) "
                    "y no debe inferirse de la similitud del recuperador."
                ),
                "first_relevant_rank": None,
                "hit_at_1": None,
                "hit_at_k": None,
                "reciprocal_rank": None,
                "hits": [asdict(h) for h in hits],
            })

    if n_answerable > 0:
        overall_metrics = {
            "hit_at_1": hit_at_1_sum / n_answerable,
            "hit_at_k": hit_at_k_sum / n_answerable,
            "mrr_at_k": mrr_at_k_sum / n_answerable,
            "denominator": n_answerable,
        }
    else:
        overall_metrics = {
            "hit_at_1": None,
            "hit_at_k": None,
            "mrr_at_k": None,
            "denominator": 0,
        }

    return {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "dataset_sha256": dataset_sha256,
        "index_input_sha256": index.get("input_sha256"),
        "model": index.get("model"),
        "model_digest": index.get("model_digest"),
        "dimension": index.get("dimension"),
        "document_prefix": index.get("document_prefix"),
        "query_prefix": index.get("query_prefix"),
        "chunking_strategy": chunking_strategy,
        "top_k_requested": top_k,
        "corpus_size": len(index["chunks"]),
        "n_answerable": n_answerable,
        "n_unanswerable": n_unanswerable,
        "metrics": overall_metrics,
        "cases": case_reports,
    }


def _format_text_report(report: dict, dataset: dict) -> str:
    """Genera un informe estructurado y legible en español a partir del dict de resultados."""
    lines: list[str] = [
        "=" * 72,
        "EVALUACIÓN DE RECUPERACIÓN SEMÁNTICA (RETRIEVAL DEV)",
        "=" * 72,
        f"Modelo: {report.get('model')} (dimensión {report.get('dimension')})",
        f"Model digest: {report.get('model_digest')}",
        f"Estrategia de fragmentación: {report.get('chunking_strategy')}",
        f"Tamaño del corpus indexado: {report.get('corpus_size')} fragmentos",
        f"Top-k evaluado: {report.get('top_k_requested')}",
        f"Casos respondibles: {report.get('n_answerable')} | Casos de abstención: {report.get('n_unanswerable')}",
        f"Dataset SHA-256: {report.get('dataset_sha256')}",
        "",
        "-" * 72,
        "RESULTADOS POR CASO (RESPONDIBLES)",
        "-" * 72,
        f"{'ID':<6} | {'Pág. Esp.':<9} | {'1ª Pos. Relevante':<18} | {'Hit@1':<5} | {'Hit@k':<5} | {'RR@k':<6}",
        "-" * 72,
    ]

    case_dict = {c["id"]: c for c in dataset["cases"]}
    for c in report["cases"]:
        if not c["answerable"]:
            continue
        orig = case_dict.get(c["id"], {})
        ev_pages = sorted(list({ev["page_number"] for ev in orig.get("evidence", [])}))
        page_str = ", ".join(str(p) for p in ev_pages) if ev_pages else "-"

        if not c["gold_available"]:
            rank_str = "Gold no disp. (Fallo)"
        elif c["first_relevant_rank"] is not None:
            rank_str = f"Posición {c['first_relevant_rank']}"
        else:
            rank_str = "No recuperado"

        hit1 = str(c["hit_at_1"])
        hitk = str(c["hit_at_k"])
        rr = f"{c['reciprocal_rank']:.4f}"
        lines.append(f"{c['id']:<6} | {page_str:<9} | {rank_str:<18} | {hit1:<5} | {hitk:<5} | {rr:<6}")

    lines.append("-" * 72)
    lines.append("")
    lines.append("-" * 72)
    lines.append("MÉTRICAS GLOBALES DE RECUPERACIÓN")
    lines.append("-" * 72)

    metrics = report["metrics"]
    denom = metrics["denominator"]
    if denom > 0:
        h1 = metrics["hit_at_1"]
        hk = metrics["hit_at_k"]
        mrr = metrics["mrr_at_k"]
        h1_pct = (h1 * 100.0) if h1 is not None else 0.0
        hk_pct = (hk * 100.0) if hk is not None else 0.0
        lines.append(f"  Denominador (casos respondibles): {denom}")
        lines.append(f"  Hit@1: {h1:.4f} ({h1_pct:.1f}%) [{int(round(h1 * denom))}/{denom}]")
        lines.append(f"  Hit@{report['top_k_requested']}: {hk:.4f} ({hk_pct:.1f}%) [{int(round(hk * denom))}/{denom}]")
        lines.append(f"  MRR@{report['top_k_requested']}: {mrr:.4f} (rango recíproco medio truncado a k={report['top_k_requested']})")
    else:
        lines.append("  Denominador: 0 casos respondibles (métricas no aplicables: null)")

    lines.append("")
    lines.append("-" * 72)
    lines.append("PREGUNTAS DE ABSTENCIÓN (Pendientes de evaluar con generador LLM)")
    lines.append("-" * 72)
    for c in report["cases"]:
        if c["answerable"]:
            continue
        lines.append(f"[{c['id']}] {c['question']}")
        lines.append(f"  Estado: {c['status']}")
        lines.append(f"  Motivo: {c['reason']}")
        lines.append(f"  Fragmentos devueltos por el buscador en top-{report['top_k_requested']}: {len(c['hits'])}")
        lines.append("")

    lines.append("-" * 72)
    lines.append("ADVERTENCIA METODOLÓGICA")
    lines.append("-" * 72)
    lines.append(
        "Advertencia: este conjunto contiene solo cinco casos de desarrollo para validar la recuperación "
        "antes de la generación; no constituye un test reservado ni permite generalizar la calidad global del sistema."
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Punto de entrada de la CLI de evaluación de recuperación."""
    parser = argparse.ArgumentParser(
        description="Evaluación de recuperación semántica sobre preguntas de desarrollo de pensiones."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="Ruta al archivo JSON del dataset de desarrollo.",
    )
    parser.add_argument(
        "--index",
        type=Path,
        default=DEFAULT_INDEX_PATH,
        help="Ruta al archivo JSON con el índice de embeddings.",
    )
    parser.add_argument(
        "--top-k",
        dest="top_k",
        type=int,
        default=3,
        help="Número de fragmentos principales a recuperar y evaluar.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emitir informe completo en formato JSON UTF-8.",
    )

    try:
        args = parser.parse_args(argv)

        if type(args.top_k) is not int or args.top_k <= 0:
            sys.stderr.write(f"Error: top-k must be a positive integer, got {args.top_k!r}\n")
            return 1

        dataset_path = validate_project_path(args.dataset)
        index_path = validate_project_path(args.index)

        dataset = load_dataset(dataset_path)
        index = load_index(index_path)

        report = evaluate(
            index=index,
            dataset=dataset,
            top_k=args.top_k,
            client=None,
        )

        if args.json:
            sys.stdout.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        else:
            text_output = _format_text_report(report, dataset)
            sys.stdout.write(text_output + "\n")

        return 0
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 1
    except Exception as exc:
        sys.stderr.write(f"Error: {exc}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
