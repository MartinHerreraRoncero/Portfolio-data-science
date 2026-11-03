"""Evaluación reproducible offline de pares afirmación-evidencias para verificación semántica."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import time

from rag_pensiones.answer import (
    ALLOWED_MODELS,
    DEFAULT_GENERATION_MODEL,
    _reject_constant,
    validate_chat_stats,
)
from rag_pensiones.grounding import (
    OllamaGrounder,
    SHA256_HEX_PATTERN,
    validate_verification_payload,
)
from rag_pensiones.ingest import validate_project_path

DEFAULT_PAIRS_DATASET_PATH = Path("evaluation/grounding_pairs.json")
DATASET_SCHEMA_VERSION = "grounding-pairs-v1"
EVALUATION_SCHEMA_VERSION = "grounding-pairs-eval-v1"

ALLOWED_SPLITS = ("dev", "reserved")
ALLOWED_EXPECTED = ("supported", "contradicted", "insufficient_evidence")
ALLOWED_SOURCE_KINDS = ("sintetico", "extracto_publico")

MAX_CLAIM_CHARS = 300
MIN_EVIDENCE_ITEMS = 1
MAX_EVIDENCE_ITEMS = 4
MIN_QUOTE_CHARS = 12
MAX_QUOTE_CHARS = 700
MAX_EVIDENCE_BYTES = 2400


class EvaluationError(RuntimeError):
    """Error técnico durante la evaluación de pares que provoca terminación inmediata (fail-fast)."""

    def __init__(self, message: str, report: dict | None = None) -> None:
        super().__init__(message)
        self.report = report


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


def validate_pairs_dataset(dataset: dict) -> dict:
    """Valida exhaustivamente el esquema, tipos y restricciones de todos los casos del dataset."""
    if isinstance(dataset, bool) or not isinstance(dataset, dict):
        raise ValueError(f"Dataset must be a dict, got {type(dataset).__name__}")

    # Rechazar NaN e infinitos en JSON
    json.dumps(dataset, allow_nan=False)

    allowed_top_keys = {"schema_version", "annotation", "cases"}
    if set(dataset.keys()) != allowed_top_keys:
        extra = set(dataset.keys()) - allowed_top_keys
        missing = allowed_top_keys - set(dataset.keys())
        raise ValueError(
            f"Dataset keys mismatch. Missing: {sorted(missing)}, Extra: {sorted(extra)}"
        )

    schema_version = dataset["schema_version"]
    if schema_version != DATASET_SCHEMA_VERSION:
        raise ValueError(
            f"Invalid schema_version: expected {DATASET_SCHEMA_VERSION!r}, got {schema_version!r}"
        )

    annotation = dataset["annotation"]
    if not isinstance(annotation, str) or not annotation.strip():
        raise ValueError("annotation must be a non-empty string")

    cases = dataset["cases"]
    if not isinstance(cases, list) or len(cases) == 0:
        raise ValueError("cases must be a non-empty list of case objects")

    allowed_case_keys = {
        "id",
        "split",
        "kind",
        "claim",
        "evidence",
        "expected",
        "source_kind",
    }
    seen_ids: set[str] = set()

    for idx, c in enumerate(cases):
        if isinstance(c, bool) or not isinstance(c, dict):
            raise ValueError(f"Case at index {idx} must be a dict, got {type(c).__name__}")

        if set(c.keys()) != allowed_case_keys:
            extra = set(c.keys()) - allowed_case_keys
            missing = allowed_case_keys - set(c.keys())
            raise ValueError(
                f"Case at index {idx} keys mismatch. Missing: {sorted(missing)}, Extra: {sorted(extra)}"
            )

        case_id = c["id"]
        if not isinstance(case_id, str) or not case_id.strip():
            raise ValueError(f"Case at index {idx} has invalid or empty id: {case_id!r}")
        if case_id in seen_ids:
            raise ValueError(f"Duplicate case id detected at index {idx}: {case_id!r}")
        seen_ids.add(case_id)

        split = c["split"]
        if split not in ALLOWED_SPLITS:
            raise ValueError(
                f"Case {case_id!r} has invalid split {split!r}; expected one of {sorted(ALLOWED_SPLITS)}"
            )

        kind = c["kind"]
        if not isinstance(kind, str) or not kind.strip():
            raise ValueError(f"Case {case_id!r} has empty or non-string kind")

        claim = c["claim"]
        if not isinstance(claim, str) or not claim.strip():
            raise ValueError(f"Case {case_id!r} has empty or non-string claim")
        if len(claim) > MAX_CLAIM_CHARS:
            raise ValueError(
                f"Case {case_id!r} claim length ({len(claim)}) exceeds maximum allowed {MAX_CLAIM_CHARS} chars"
            )

        evidence = c["evidence"]
        if not isinstance(evidence, list):
            raise ValueError(f"Case {case_id!r} evidence must be a list of quote strings")
        if not (MIN_EVIDENCE_ITEMS <= len(evidence) <= MAX_EVIDENCE_ITEMS):
            raise ValueError(
                f"Case {case_id!r} evidence list must contain between {MIN_EVIDENCE_ITEMS} and "
                f"{MAX_EVIDENCE_ITEMS} items, got {len(evidence)}"
            )

        total_bytes = 0
        for q_idx, quote in enumerate(evidence):
            if not isinstance(quote, str) or not quote.strip():
                raise ValueError(
                    f"Evidence quote {q_idx} in case {case_id!r} must be a non-empty string"
                )
            if not (MIN_QUOTE_CHARS <= len(quote) <= MAX_QUOTE_CHARS):
                raise ValueError(
                    f"Evidence quote {q_idx} in case {case_id!r} length ({len(quote)}) must be between "
                    f"{MIN_QUOTE_CHARS} and {MAX_QUOTE_CHARS} chars"
                )
            total_bytes += len(quote.encode("utf-8"))

        if total_bytes > MAX_EVIDENCE_BYTES:
            raise ValueError(
                f"Case {case_id!r} total evidence bytes ({total_bytes}) exceeds budget of "
                f"{MAX_EVIDENCE_BYTES} bytes UTF-8"
            )

        expected = c["expected"]
        if expected not in ALLOWED_EXPECTED:
            raise ValueError(
                f"Case {case_id!r} has invalid expected verdict {expected!r}; expected one of {sorted(ALLOWED_EXPECTED)}"
            )

        source_kind = c["source_kind"]
        if source_kind not in ALLOWED_SOURCE_KINDS:
            raise ValueError(
                f"Case {case_id!r} has invalid source_kind {source_kind!r}; expected one of {sorted(ALLOWED_SOURCE_KINDS)}"
            )

    return dataset


def load_pairs_dataset(path: Path | str) -> tuple[dict, str]:
    """Carga y valida el dataset de pares calculando la huella SHA-256 de los bytes crudos."""
    resolved_path = validate_project_path(Path(path))
    if not resolved_path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {resolved_path}")

    raw_bytes = resolved_path.read_bytes()
    if not raw_bytes.strip():
        raise ValueError("Dataset file is empty")

    file_sha256 = hashlib.sha256(raw_bytes).hexdigest()

    try:
        text = raw_bytes.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError(f"Dataset file is not valid UTF-8: {exc}") from exc

    try:
        data = json.loads(text, parse_constant=_reject_constant)
    except Exception as exc:
        raise ValueError(f"Malformed JSON in dataset file: {exc}") from exc

    validate_pairs_dataset(data)
    return data, file_sha256


def compute_pair_metrics(results: list[dict]) -> dict:
    """Calcula la matriz de confusión 3-clases y métricas con denominadores explícitos."""
    classes = list(ALLOWED_EXPECTED)
    confusion_matrix: dict[str, dict[str, int]] = {
        gold: {pred: 0 for pred in classes} for gold in classes
    }

    for idx, r in enumerate(results):
        if not isinstance(r, dict):
            raise ValueError(f"Result at index {idx} must be a dict")
        gold = r.get("expected")
        pred = r.get("predicted", r.get("verdict"))
        if gold not in ALLOWED_EXPECTED:
            raise ValueError(f"Invalid expected verdict {gold!r} at index {idx}")
        if pred not in ALLOWED_EXPECTED:
            raise ValueError(f"Invalid predicted verdict {pred!r} at index {idx}")
        confusion_matrix[gold][pred] += 1

    total_cases = len(results)
    correct = sum(confusion_matrix[c][c] for c in classes)

    def _make_metric(numerator: int, denominator: int) -> dict:
        val = (numerator / denominator) if denominator > 0 else None
        return {
            "numerator": numerator,
            "denominator": denominator,
            "value": val,
        }

    # Falsa aceptación de no apoyadas: predijo supported cuando gold es contradicted o insufficient_evidence
    fa_num = (
        confusion_matrix["contradicted"]["supported"]
        + confusion_matrix["insufficient_evidence"]["supported"]
    )
    fa_den = sum(confusion_matrix["contradicted"].values()) + sum(
        confusion_matrix["insufficient_evidence"].values()
    )

    # Falso rechazo de apoyadas: predijo no supported cuando gold es supported
    fr_num = (
        confusion_matrix["supported"]["contradicted"]
        + confusion_matrix["supported"]["insufficient_evidence"]
    )
    fr_den = sum(confusion_matrix["supported"].values())

    # Precisión de aceptadas: aciertos supported / total predichas supported
    prec_num = confusion_matrix["supported"]["supported"]
    prec_den = sum(confusion_matrix[g]["supported"] for g in classes)

    # Cobertura de apoyadas (recall): aciertos supported / total gold supported
    rec_num = confusion_matrix["supported"]["supported"]
    rec_den = sum(confusion_matrix["supported"].values())

    # Tasa de aceptación: total predichas supported / total casos válidos
    acc_num = sum(confusion_matrix[g]["supported"] for g in classes)

    return {
        "confusion_matrix": confusion_matrix,
        "total_cases": total_cases,
        "accuracy": _make_metric(correct, total_cases),
        "false_accept_unsupported": _make_metric(fa_num, fa_den),
        "false_reject_supported": _make_metric(fr_num, fr_den),
        "accepted_precision": _make_metric(prec_num, prec_den),
        "supported_recall": _make_metric(rec_num, rec_den),
        "acceptance_rate": _make_metric(acc_num, total_cases),
    }


def _aggregate_stats(cases: list[dict]) -> dict:
    """Agrega sumas de estadísticas sobre los casos válidos completados."""
    agg: dict[str, int] = {
        "prompt_eval_count": sum(c["stats"].get("prompt_eval_count", 0) for c in cases),
        "eval_count": sum(c["stats"].get("eval_count", 0) for c in cases),
        "eval_duration": sum(c["stats"].get("eval_duration", 0) for c in cases),
        "total_duration": sum(c["stats"].get("total_duration", 0) for c in cases),
    }
    if any("load_duration" in c["stats"] for c in cases):
        agg["load_duration"] = sum(
            c["stats"].get("load_duration", 0)
            for c in cases
            if "load_duration" in c["stats"]
        )
    if any("prompt_eval_duration" in c["stats"] for c in cases):
        agg["prompt_eval_duration"] = sum(
            c["stats"].get("prompt_eval_duration", 0)
            for c in cases
            if "prompt_eval_duration" in c["stats"]
        )
    return agg


def evaluate_pairs(
    dataset: dict,
    split: str = "dev",
    grounder: OllamaGrounder | None = None,
    model: str = DEFAULT_GENERATION_MODEL,
    dataset_sha256: str | None = None,
) -> dict:
    """Evalúa la verificación de pares sobre una división del dataset con fail-fast y métricas."""
    if split not in ALLOWED_SPLITS:
        raise ValueError(
            f"Invalid split {split!r}; expected one of {sorted(ALLOWED_SPLITS)}"
        )

    if model not in ALLOWED_MODELS:
        raise ValueError(
            f"Model {model!r} not in ALLOWED_MODELS: {sorted(ALLOWED_MODELS)}"
        )

    validate_pairs_dataset(dataset)

    # Validar dataset_sha256 si se aporta como str hexadecimal64 antes de inicializar cliente
    if dataset_sha256 is not None:
        if (
            isinstance(dataset_sha256, bool)
            or not isinstance(dataset_sha256, str)
            or len(dataset_sha256) != 64
            or not SHA256_HEX_PATTERN.fullmatch(dataset_sha256)
        ):
            raise ValueError(
                f"Invalid dataset_sha256: {dataset_sha256!r}. Must be a 64-character lowercase hexadecimal string."
            )
        dataset_sha256_kind = "rawbytes"
    else:
        dataset_sha256 = compute_canonical_sha256(dataset)
        dataset_sha256_kind = "canonical_json"

    split_cases = [c for c in dataset["cases"] if c["split"] == split]
    if len(split_cases) == 0:
        raise ValueError(f"No cases found for split {split!r}")

    dataset_count = len(split_cases)

    # Inicializar o validar el grounder inyectado
    if grounder is None:
        grounder = OllamaGrounder(model=model)
    else:
        if getattr(grounder, "model", None) != model:
            raise ValueError(
                f"Injected grounder model ({getattr(grounder, 'model', None)!r}) "
                f"does not match requested model ({model!r})"
            )
        if getattr(grounder, "generator", None) is not None:
            if getattr(grounder.generator, "model", None) != model:
                raise ValueError(
                    f"Injected grounder generator model ({getattr(grounder.generator, 'model', None)!r}) "
                    f"does not match requested model ({model!r})"
                )

    if grounder.generator is None:
        raise RuntimeError("Grounder does not have an initialized generator")

    pinned_digest = grounder.generator.get_digest()
    if not isinstance(pinned_digest, str) or not pinned_digest.strip():
        raise RuntimeError("Unable to obtain model digest from generator")

    case_results: list[dict] = []
    total_start = time.perf_counter()

    for case in split_cases:
        # Aislar rigurosamente la entrada al modelo: SOLO claims y catálogo necesario
        # NUNCA transmitir expected, kind, split, annotation ni source_kind al LLM
        claims = [
            {
                "claim_id": "C1",
                "text": case["claim"],
                "evidence_ids": [f"E{i + 1}" for i in range(len(case["evidence"]))],
            }
        ]
        evidence = [
            {
                "evidence_id": f"E{i + 1}",
                "source_id": f"S{i + 1}",
                "quote": q,
            }
            for i, q in enumerate(case["evidence"])
        ]

        t0 = time.perf_counter()
        try:
            try:
                verify_res = grounder.verify(
                    claims=claims,
                    evidence=evidence,
                    expected_digest=pinned_digest,
                )
            except Exception as exc:
                raise RuntimeError(f"Verify invocation failed: {exc}") from exc

            case_latency = time.perf_counter() - t0

            # Validar exhaustivamente la respuesta del verify inyectado
            if isinstance(verify_res, bool) or not isinstance(verify_res, dict):
                raise ValueError(
                    f"Invalid verify response type: {type(verify_res).__name__}"
                )

            required_keys = {"verdicts", "digest", "prompt_sha256", "stats"}
            if set(verify_res.keys()) != required_keys:
                extra = set(verify_res.keys()) - required_keys
                missing = required_keys - set(verify_res.keys())
                raise ValueError(
                    f"Verify response keys mismatch. Missing: {sorted(missing)}, Extra: {sorted(extra)}"
                )

            if verify_res["digest"] != pinned_digest:
                raise ValueError(
                    f"Digest mismatch in verify response: expected {pinned_digest!r}, got {verify_res['digest']!r}"
                )

            p_hash = verify_res["prompt_sha256"]
            if (
                isinstance(p_hash, bool)
                or not isinstance(p_hash, str)
                or len(p_hash) != 64
                or not SHA256_HEX_PATTERN.fullmatch(p_hash)
            ):
                raise ValueError(f"Invalid prompt_sha256 in verify response: {p_hash!r}")

            # Validar estadísticas usando validate_chat_stats
            validated_stats = validate_chat_stats(verify_res["stats"])

            # Validar veredictos usando validate_verification_payload con entradas exactas
            validated_payload = validate_verification_payload(
                {"verdicts": verify_res["verdicts"]},
                claims=claims,
                evidence=evidence,
            )

            v0 = validated_payload["verdicts"][0]
            pred_verdict = v0["verdict"]

            case_results.append(
                {
                    "id": case["id"],
                    "kind": case["kind"],
                    "source_kind": case["source_kind"],
                    "claim": case["claim"],
                    "expected": case["expected"],
                    "predicted": pred_verdict,
                    "verdict_reason": v0["reason"],
                    "evidence_verdicts": v0["evidence_verdicts"],
                    "correct": (pred_verdict == case["expected"]),
                    "latency_seconds": case_latency,
                    "prompt_sha256": p_hash,
                    "stats": validated_stats,
                }
            )
        except Exception as exc:
            # Fail-fast: construir reporte parcial exacto según contrato
            completed_count = len(case_results)
            error_count = 1
            unprocessed_count = dataset_count - completed_count - error_count
            partial_duration = time.perf_counter() - total_start
            partial_metrics = compute_pair_metrics(case_results)
            partial_stats = _aggregate_stats(case_results)
            clean_message = str(exc)

            partial_report = {
                "schema_version": EVALUATION_SCHEMA_VERSION,
                "run_status": "error",
                "split": split,
                "model": model,
                "model_digest": pinned_digest,
                "dataset_sha256": dataset_sha256,
                "dataset_sha256_kind": dataset_sha256_kind,
                "dataset_count": dataset_count,
                "completed_count": completed_count,
                "error_count": error_count,
                "unprocessed_count": unprocessed_count,
                "cases_count": completed_count,
                "duration_seconds": partial_duration,
                "error": {
                    "case_id": case["id"],
                    "message": clean_message,
                },
                "metrics": partial_metrics,
                "stats": partial_stats,
                "cases": list(case_results),
            }
            # Validar que no contiene NaN/Inf
            json.dumps(partial_report, allow_nan=False)
            raise EvaluationError(
                f"Evaluation failed at case {case['id']!r}: {clean_message}",
                report=partial_report,
            ) from exc

    total_duration = time.perf_counter() - total_start
    metrics = compute_pair_metrics(case_results)
    total_stats = _aggregate_stats(case_results)

    success_report = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "run_status": "completed",
        "split": split,
        "model": model,
        "model_digest": pinned_digest,
        "dataset_sha256": dataset_sha256,
        "dataset_sha256_kind": dataset_sha256_kind,
        "dataset_count": dataset_count,
        "completed_count": dataset_count,
        "error_count": 0,
        "unprocessed_count": 0,
        "cases_count": dataset_count,
        "duration_seconds": total_duration,
        "error": None,
        "metrics": metrics,
        "stats": total_stats,
        "cases": case_results,
    }
    json.dumps(success_report, allow_nan=False)
    return success_report


def _format_text_report(report: dict) -> str:
    """Formatea el reporte de evaluación en texto claro para inspección humana."""
    dataset_count = report.get("dataset_count", report.get("cases_count", 0))
    completed_count = report.get("completed_count", report.get("cases_count", 0))
    error_count = report.get("error_count", 0)
    unprocessed_count = report.get("unprocessed_count", 0)

    lines = [
        "=" * 72,
        "EVALUACIÓN DE PARES AFIRMACIÓN-EVIDENCIAS (GROUNDING)",
        "=" * 72,
        f"Versión de esquema : {report['schema_version']}",
        f"Estado de ejecución : {report['run_status']}",
        f"División evaluada  : {report['split']}",
        f"Modelo             : {report['model']}",
        f"Digest del modelo  : {report['model_digest']}",
        f"SHA-256 dataset    : {report['dataset_sha256']} ({report.get('dataset_sha256_kind', 'unknown')})",
        f"Casos totales      : {dataset_count}",
        f"Casos completados  : {completed_count}",
        f"Casos con error    : {error_count}",
        f"Casos no procesados: {unprocessed_count}",
        f"Tiempo total (s)   : {report['duration_seconds']:.2f}",
    ]

    if report.get("error") is not None:
        err = report["error"]
        lines.extend([
            "",
            "!" * 72,
            f"ERROR EN CASO: {err.get('case_id', 'unknown')}",
            f"Mensaje      : {err.get('message', 'Sin detalle')}",
            "!" * 72,
        ])

    lines.extend([
        "",
        "-" * 72,
        "MATRIZ DE CONFUSIÓN 3-CLASES (Filas: Gold, Columnas: Predicción)",
        "-" * 72,
    ])

    cm = report["metrics"]["confusion_matrix"]
    classes = list(ALLOWED_EXPECTED)
    header = f"{'Gold / Pred':<24} | {'supported':<11} | {'contradicted':<12} | {'insufficient':<12}"
    lines.append(header)
    lines.append("-" * len(header))
    for gold in classes:
        row = (
            f"{gold:<24} | "
            f"{cm[gold]['supported']:<11} | "
            f"{cm[gold]['contradicted']:<12} | "
            f"{cm[gold]['insufficient_evidence']:<12}"
        )
        lines.append(row)

    lines.append("")
    lines.append("-" * 72)
    lines.append("MÉTRICAS DE RENDIMIENTO")
    lines.append("-" * 72)

    def _fmt_m(m: dict) -> str:
        num, den, val = m["numerator"], m["denominator"], m["value"]
        if val is None or den == 0:
            return f"N/A ({num}/{den})"
        return f"{val * 100:.2f}% ({num}/{den})"

    metrics = report["metrics"]
    lines.append(f"  Exactitud global (Accuracy)        : {_fmt_m(metrics['accuracy'])}")
    lines.append(f"  Falsa aceptación no apoyadas (FA) : {_fmt_m(metrics['false_accept_unsupported'])}")
    lines.append(f"  Falso rechazo apoyadas (FR)       : {_fmt_m(metrics['false_reject_supported'])}")
    lines.append(f"  Precisión de aceptadas (Precision) : {_fmt_m(metrics['accepted_precision'])}")
    lines.append(f"  Exhaustividad apoyadas (Recall)    : {_fmt_m(metrics['supported_recall'])}")
    lines.append(f"  Tasa de aceptación (Acceptance)   : {_fmt_m(metrics['acceptance_rate'])}")

    lines.append("")
    lines.append("-" * 72)
    lines.append("CONSUMO DE TOKENS Y ESTADÍSTICAS")
    lines.append("-" * 72)
    stats = report["stats"]
    lines.append(f"  Prompt tokens evaluados : {stats.get('prompt_eval_count', 0)}")
    lines.append(f"  Tokens generados        : {stats.get('eval_count', 0)}")
    lines.append(f"  Duración eval (ms)      : {stats.get('eval_duration', 0) // 1_000_000}")
    lines.append(f"  Duración total (ms)     : {stats.get('total_duration', 0) // 1_000_000}")
    if "load_duration" in stats:
        lines.append(f"  Duración de carga (ms)  : {stats['load_duration'] // 1_000_000}")
    if "prompt_eval_duration" in stats:
        lines.append(f"  Duración prompt eval(ms): {stats['prompt_eval_duration'] // 1_000_000}")

    lines.append("")
    lines.append("-" * 72)
    lines.append("ADVERTENCIA METODOLÓGICA")
    lines.append("-" * 72)
    lines.append(
        "Advertencia metodológica:\n"
        "1. Las etiquetas de referencia gold fueron curadas por Codex sin adjudicación humana independiente.\n"
        "2. El conjunto reservado de pares sintéticos no permite generalizaciones estadísticas amplias sobre el corpus.\n"
        "3. El modelo verificador comparte familia de pesos (Qwen 9B), por lo que sus errores pueden estar correlacionados con el generador.\n"
        "4. Esta evaluación mide la coherencia de verificación afirmación-cita sobre pares aislados, no la calidad del pipeline RAG completo."
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Punto de entrada de la CLI de evaluación de pares de fundamentación."""
    parser = argparse.ArgumentParser(
        description="Evaluación reproducible de pares afirmación-evidencias para verificación semántica."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_PAIRS_DATASET_PATH,
        help="Ruta al archivo JSON del dataset de pares.",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=list(ALLOWED_SPLITS),
        default="dev",
        help="División a evaluar ('dev' por defecto, 'reserved' para conjunto final).",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=list(ALLOWED_MODELS),
        default=DEFAULT_GENERATION_MODEL,
        help=f"Modelo Ollama para verificación (por defecto {DEFAULT_GENERATION_MODEL}).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emitir informe completo en formato JSON UTF-8 por stdout.",
    )

    args = None
    try:
        args = parser.parse_args(argv)
        dataset_path = validate_project_path(args.dataset)
        dataset, dataset_sha256 = load_pairs_dataset(dataset_path)

        report = evaluate_pairs(
            dataset=dataset,
            split=args.split,
            grounder=None,
            model=args.model,
            dataset_sha256=dataset_sha256,
        )

        if args.json:
            sys.stdout.write(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        else:
            sys.stdout.write(_format_text_report(report) + "\n")

        return 0
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 1
    except EvaluationError as exc:
        if exc.report is not None and args is not None:
            if getattr(args, "json", False):
                sys.stdout.write(json.dumps(exc.report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
            else:
                sys.stdout.write(_format_text_report(exc.report) + "\n")
        else:
            sys.stderr.write(f"Error: {exc}\n")
        return 1
    except Exception as exc:
        sys.stderr.write(f"Error: {exc}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
