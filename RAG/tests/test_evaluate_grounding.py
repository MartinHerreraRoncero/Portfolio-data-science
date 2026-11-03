"""Tests unitarios para la evaluación de pares de fundamentación (rag_pensiones.evaluate_grounding)."""

from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from rag_pensiones.ingest import PROJECT_ROOT
from rag_pensiones.evaluate_grounding import (
    ALLOWED_EXPECTED,
    ALLOWED_SOURCE_KINDS,
    ALLOWED_SPLITS,
    DATASET_SCHEMA_VERSION,
    EVALUATION_SCHEMA_VERSION,
    EvaluationError,
    _format_text_report,
    compute_canonical_sha256,
    compute_pair_metrics,
    evaluate_pairs,
    load_pairs_dataset,
    main,
    validate_pairs_dataset,
)


class MockGenerator:
    """Generador simulado para pruebas unitarias sin llamadas a Ollama."""

    def __init__(self, model: str = "qwen3.5:9b", digest: str = "sha256:mock_model_digest_12345"):
        self.model = model
        self.digest = digest

    def get_digest(self) -> str:
        return self.digest


class MockGrounder:
    """Grounder simulado para pruebas de evaluación de pares sin red externa."""

    def __init__(
        self,
        model: str = "qwen3.5:9b",
        generator: MockGenerator | None = None,
        responses: list[dict] | None = None,
    ):
        self.model = model
        self.generator = generator or MockGenerator(model=model)
        self.responses = list(responses) if responses else []
        self.recorded_calls: list[dict] = []

    def verify(self, claims: list[dict], evidence: list[dict], expected_digest: str) -> dict:
        self.recorded_calls.append(
            {
                "claims": claims,
                "evidence": evidence,
                "expected_digest": expected_digest,
            }
        )
        if self.responses:
            resp = self.responses.pop(0)
            if isinstance(resp, Exception):
                raise resp
            return resp
        # Respuesta por defecto supported
        return {
            "verdicts": [
                {
                    "claim_id": "C1",
                    "verdict": "supported",
                    "reason": "Soporte completo verificado",
                    "evidence_verdicts": [
                        {"evidence_id": f"E{i + 1}", "relation": "supports"}
                        for i in range(len(evidence))
                    ],
                }
            ],
            "digest": expected_digest,
            "prompt_sha256": "a" * 64,
            "stats": {
                "prompt_eval_count": 120,
                "eval_count": 35,
                "eval_duration": 40_000_000,
                "total_duration": 50_000_000,
            },
        }


def make_valid_dataset() -> dict:
    """Genera un dataset sintético válido para pruebas unitarias sin consultar gold real."""
    return {
        "schema_version": "grounding-pairs-v1",
        "annotation": "Dataset sintético para pruebas unitarias de aislamiento",
        "cases": [
            {
                "id": "pair_dev_01",
                "split": "dev",
                "kind": "direct_quote",
                "claim": "El gasto en pensiones alcanzará su pico histórico en el año 2050.",
                "evidence": [
                    "El gasto en pensiones alcanzará su cota máxima en el año 2050 según las proyecciones."
                ],
                "expected": "supported",
                "source_kind": "sintetico",
            },
            {
                "id": "pair_dev_02",
                "split": "dev",
                "kind": "contradiction_year",
                "claim": "El informe prevé que el gasto máximo de pensiones se producirá en 2030.",
                "evidence": [
                    "El gasto en pensiones alcanzará su cota máxima en el año 2050 según las proyecciones."
                ],
                "expected": "contradicted",
                "source_kind": "sintetico",
            },
            {
                "id": "pair_dev_03",
                "split": "dev",
                "kind": "insufficient_context",
                "claim": "La tasa de sustitución se reducirá al 40% en 2070.",
                "evidence": [
                    "El gasto en pensiones alcanzará su cota máxima en el año 2050 según las proyecciones."
                ],
                "expected": "insufficient_evidence",
                "source_kind": "sintetico",
            },
            {
                "id": "pair_res_01",
                "split": "reserved",
                "kind": "reserved_synthetic",
                "claim": "El factor demográfico presiona al alza el gasto total de jubilación.",
                "evidence": [
                    "La demografía desfavorable y la jubilación de la generación baby-boom elevan el gasto."
                ],
                "expected": "supported",
                "source_kind": "sintetico",
            },
        ],
    }


class TestValidatePairsDataset(unittest.TestCase):
    """Pruebas de validación exhaustiva previa del dataset de pares."""

    def test_valid_dataset_passes(self):
        data = make_valid_dataset()
        validated = validate_pairs_dataset(data)
        self.assertIs(validated, data)

    def test_non_dict_rejected(self):
        with self.assertRaises(ValueError):
            validate_pairs_dataset(["not", "a", "dict"])
        with self.assertRaises(ValueError):
            validate_pairs_dataset(True)

    def test_invalid_schema_version_rejected(self):
        data = make_valid_dataset()
        data["schema_version"] = "invalid-v1"
        with self.assertRaises(ValueError):
            validate_pairs_dataset(data)

    def test_missing_or_extra_top_keys_rejected(self):
        data = make_valid_dataset()
        data["extra_key"] = 123
        with self.assertRaises(ValueError):
            validate_pairs_dataset(data)

        data = make_valid_dataset()
        del data["annotation"]
        with self.assertRaises(ValueError):
            validate_pairs_dataset(data)

    def test_empty_cases_rejected(self):
        data = make_valid_dataset()
        data["cases"] = []
        with self.assertRaises(ValueError):
            validate_pairs_dataset(data)

    def test_duplicate_case_id_rejected(self):
        data = make_valid_dataset()
        data["cases"].append(dict(data["cases"][0]))
        with self.assertRaises(ValueError):
            validate_pairs_dataset(data)

    def test_invalid_split_rejected(self):
        data = make_valid_dataset()
        data["cases"][0]["split"] = "eval"
        with self.assertRaises(ValueError):
            validate_pairs_dataset(data)

    def test_claim_length_limits(self):
        data = make_valid_dataset()
        data["cases"][0]["claim"] = "A" * 301
        with self.assertRaises(ValueError):
            validate_pairs_dataset(data)

        data = make_valid_dataset()
        data["cases"][0]["claim"] = "   "
        with self.assertRaises(ValueError):
            validate_pairs_dataset(data)

    def test_evidence_item_count_limits(self):
        data = make_valid_dataset()
        data["cases"][0]["evidence"] = []
        with self.assertRaises(ValueError):
            validate_pairs_dataset(data)

        data = make_valid_dataset()
        data["cases"][0]["evidence"] = [
            "Evidencia número uno válida para test.",
            "Evidencia número dos válida para test.",
            "Evidencia número tres válida para test.",
            "Evidencia número cuatro válida para test.",
            "Evidencia número cinco que excede límite.",
        ]
        with self.assertRaises(ValueError):
            validate_pairs_dataset(data)

    def test_quote_length_limits(self):
        data = make_valid_dataset()
        data["cases"][0]["evidence"] = ["Corto"]  # 5 chars < 12
        with self.assertRaises(ValueError):
            validate_pairs_dataset(data)

        data = make_valid_dataset()
        data["cases"][0]["evidence"] = ["A" * 701]
        with self.assertRaises(ValueError):
            validate_pairs_dataset(data)

    def test_total_evidence_bytes_budget(self):
        data = make_valid_dataset()
        # 4 citas de 650 chars = 2600 bytes UTF-8 > 2400
        data["cases"][0]["evidence"] = ["A" * 650 for _ in range(4)]
        with self.assertRaises(ValueError):
            validate_pairs_dataset(data)

    def test_invalid_expected_or_source_kind(self):
        data = make_valid_dataset()
        data["cases"][0]["expected"] = "partially_supported"
        with self.assertRaises(ValueError):
            validate_pairs_dataset(data)

        data = make_valid_dataset()
        data["cases"][0]["source_kind"] = "wikipedia"
        with self.assertRaises(ValueError):
            validate_pairs_dataset(data)

    def test_reject_nan_values(self):
        data = make_valid_dataset()
        data["cases"][0]["claim"] = float("nan")
        with self.assertRaises((ValueError, TypeError)):
            validate_pairs_dataset(data)


class TestComputePairMetrics(unittest.TestCase):
    """Pruebas de la matriz de confusión 3-clases y cálculo de métricas con denominadores explícitos."""

    def test_known_confusion_matrix_arithmetic(self):
        # Escenario conocido de 8 casos:
        # Gold supported: 2 pred supported, 1 pred contradicted (total 3)
        # Gold contradicted: 1 pred supported, 2 pred contradicted (total 3)
        # Gold insufficient: 1 pred supported, 1 pred insufficient (total 2)
        results = [
            {"expected": "supported", "predicted": "supported"},
            {"expected": "supported", "predicted": "supported"},
            {"expected": "supported", "predicted": "contradicted"},
            {"expected": "contradicted", "predicted": "supported"},
            {"expected": "contradicted", "predicted": "contradicted"},
            {"expected": "contradicted", "predicted": "contradicted"},
            {"expected": "insufficient_evidence", "predicted": "supported"},
            {"expected": "insufficient_evidence", "predicted": "insufficient_evidence"},
        ]
        metrics = compute_pair_metrics(results)

        cm = metrics["confusion_matrix"]
        self.assertEqual(cm["supported"]["supported"], 2)
        self.assertEqual(cm["supported"]["contradicted"], 1)
        self.assertEqual(cm["supported"]["insufficient_evidence"], 0)
        self.assertEqual(cm["contradicted"]["supported"], 1)
        self.assertEqual(cm["contradicted"]["contradicted"], 2)
        self.assertEqual(cm["insufficient_evidence"]["supported"], 1)
        self.assertEqual(cm["insufficient_evidence"]["insufficient_evidence"], 1)

        # Accuracy: 5/8 = 0.625
        self.assertEqual(metrics["accuracy"]["numerator"], 5)
        self.assertEqual(metrics["accuracy"]["denominator"], 8)
        self.assertAlmostEqual(metrics["accuracy"]["value"], 0.625)

        # False Accept Unsupported: (1 contradicted + 1 insufficient) / 5 total unsupported = 2/5 = 0.4
        self.assertEqual(metrics["false_accept_unsupported"]["numerator"], 2)
        self.assertEqual(metrics["false_accept_unsupported"]["denominator"], 5)
        self.assertAlmostEqual(metrics["false_accept_unsupported"]["value"], 0.4)

        # False Reject Supported: 1 / 3 = 0.3333...
        self.assertEqual(metrics["false_reject_supported"]["numerator"], 1)
        self.assertEqual(metrics["false_reject_supported"]["denominator"], 3)
        self.assertAlmostEqual(metrics["false_reject_supported"]["value"], 1.0 / 3.0)

        # Accepted Precision: 2 correct supported / 4 total pred supported = 0.5
        self.assertEqual(metrics["accepted_precision"]["numerator"], 2)
        self.assertEqual(metrics["accepted_precision"]["denominator"], 4)
        self.assertAlmostEqual(metrics["accepted_precision"]["value"], 0.5)

        # Supported Recall: 2 correct supported / 3 gold supported = 2/3
        self.assertEqual(metrics["supported_recall"]["numerator"], 2)
        self.assertEqual(metrics["supported_recall"]["denominator"], 3)
        self.assertAlmostEqual(metrics["supported_recall"]["value"], 2.0 / 3.0)

        # Acceptance Rate: 4 pred supported / 8 total = 0.5
        self.assertEqual(metrics["acceptance_rate"]["numerator"], 4)
        self.assertEqual(metrics["acceptance_rate"]["denominator"], 8)
        self.assertAlmostEqual(metrics["acceptance_rate"]["value"], 0.5)

    def test_zero_denominators_return_none(self):
        # Lista vacía
        metrics = compute_pair_metrics([])
        self.assertEqual(metrics["total_cases"], 0)
        for key in (
            "accuracy",
            "false_accept_unsupported",
            "false_reject_supported",
            "accepted_precision",
            "supported_recall",
            "acceptance_rate",
        ):
            self.assertEqual(metrics[key]["denominator"], 0)
            self.assertIsNone(metrics[key]["value"])

    def test_trivial_all_insufficient_baseline(self):
        # Control estricto de baseline trivial:
        # Predecir siempre insufficient_evidence frente a dataset mixto
        results = [
            {"expected": "supported", "predicted": "insufficient_evidence"},
            {"expected": "supported", "predicted": "insufficient_evidence"},
            {"expected": "contradicted", "predicted": "insufficient_evidence"},
            {"expected": "insufficient_evidence", "predicted": "insufficient_evidence"},
        ]
        metrics = compute_pair_metrics(results)

        # false_accept_unsupported = 0.0 (0 predichas supported)
        self.assertEqual(metrics["false_accept_unsupported"]["numerator"], 0)
        self.assertEqual(metrics["false_accept_unsupported"]["denominator"], 2)
        self.assertEqual(metrics["false_accept_unsupported"]["value"], 0.0)

        # false_reject_supported = 1.0 (todas las supported rechazadas)
        self.assertEqual(metrics["false_reject_supported"]["numerator"], 2)
        self.assertEqual(metrics["false_reject_supported"]["denominator"], 2)
        self.assertEqual(metrics["false_reject_supported"]["value"], 1.0)

        # accepted_precision = None (0 predichas supported, denominador 0)
        self.assertEqual(metrics["accepted_precision"]["numerator"], 0)
        self.assertEqual(metrics["accepted_precision"]["denominator"], 0)
        self.assertIsNone(metrics["accepted_precision"]["value"])

        # supported_recall = 0.0 (0 de 2 supported recuperadas)
        self.assertEqual(metrics["supported_recall"]["numerator"], 0)
        self.assertEqual(metrics["supported_recall"]["denominator"], 2)
        self.assertEqual(metrics["supported_recall"]["value"], 0.0)

        # acceptance_rate = 0.0
        self.assertEqual(metrics["acceptance_rate"]["numerator"], 0)
        self.assertEqual(metrics["acceptance_rate"]["denominator"], 4)
        self.assertEqual(metrics["acceptance_rate"]["value"], 0.0)


class TestEvaluatePairs(unittest.TestCase):
    """Pruebas del runner de evaluación de pares con aislamiento de prompts y fail-fast."""

    def test_split_filtering_dev_vs_reserved(self):
        data = make_valid_dataset()
        grounder_dev = MockGrounder()
        rep_dev = evaluate_pairs(data, split="dev", grounder=grounder_dev)
        self.assertEqual(rep_dev["cases_count"], 3)
        self.assertEqual(len(grounder_dev.recorded_calls), 3)
        self.assertEqual(rep_dev["split"], "dev")

        grounder_res = MockGrounder()
        rep_res = evaluate_pairs(data, split="reserved", grounder=grounder_res)
        self.assertEqual(rep_res["cases_count"], 1)
        self.assertEqual(len(grounder_res.recorded_calls), 1)
        self.assertEqual(rep_res["split"], "reserved")

    def test_prompt_isolation_no_gold_leakage(self):
        data = make_valid_dataset()
        grounder = MockGrounder()
        evaluate_pairs(data, split="dev", grounder=grounder)

        self.assertEqual(len(grounder.recorded_calls), 3)
        forbidden_keys = {"expected", "kind", "split", "annotation", "source_kind"}

        for call in grounder.recorded_calls:
            # Inspeccionar claims
            claims = call["claims"]
            self.assertEqual(len(claims), 1)
            c = claims[0]
            self.assertEqual(set(c.keys()), {"claim_id", "text", "evidence_ids"})
            self.assertEqual(c["claim_id"], "C1")
            for k in forbidden_keys:
                self.assertNotIn(k, c)

            # Inspeccionar evidence
            evidence = call["evidence"]
            self.assertGreaterEqual(len(evidence), 1)
            for ev in evidence:
                self.assertEqual(set(ev.keys()), {"evidence_id", "source_id", "quote"})
                for k in forbidden_keys:
                    self.assertNotIn(k, ev)

    def test_fail_fast_on_technical_error(self):
        data = make_valid_dataset()
        # El grounder falla en la segunda llamada
        grounder = MockGrounder(
            responses=[
                # Primer caso OK
                {
                    "verdicts": [
                        {
                            "claim_id": "C1",
                            "verdict": "supported",
                            "reason": "OK",
                            "evidence_verdicts": [{"evidence_id": "E1", "relation": "supports"}],
                        }
                    ],
                    "digest": "sha256:mock_model_digest_12345",
                    "prompt_sha256": "b" * 64,
                    "stats": {"prompt_eval_count": 10, "eval_count": 5, "eval_duration": 1000, "total_duration": 2000},
                },
                # Segundo caso error técnico de conexión
                RuntimeError("Ollama connection reset by peer"),
            ]
        )

        with self.assertRaises(EvaluationError) as ctx:
            evaluate_pairs(data, split="dev", grounder=grounder)

        self.assertIn("Ollama connection reset by peer", str(ctx.exception))
        # Solo se intentaron 2 llamadas; no se continuó al tercer caso
        self.assertEqual(len(grounder.recorded_calls), 2)
        # El reporte parcial está disponible en ctx.exception.report
        self.assertIsNotNone(ctx.exception.report)
        self.assertEqual(ctx.exception.report["run_status"], "error")
        self.assertEqual(ctx.exception.report["completed_count"], 1)
        self.assertEqual(ctx.exception.report["error_count"], 1)
        self.assertEqual(ctx.exception.report["unprocessed_count"], 1)

    def test_invalid_verify_report_raises_error(self):
        data = make_valid_dataset()
        # Respuesta con digest incorrecto
        bad_resp = {
            "verdicts": [{"claim_id": "C1", "verdict": "supported", "reason": "OK", "evidence_verdicts": []}],
            "digest": "sha256:wrong_digest",
            "prompt_sha256": "c" * 64,
            "stats": {"prompt_eval_count": 1, "eval_count": 1, "eval_duration": 1, "total_duration": 1},
        }
        grounder = MockGrounder(responses=[bad_resp])

        with self.assertRaises(EvaluationError):
            evaluate_pairs(data, split="dev", grounder=grounder)

    def test_unknown_evidence_id_in_verification_verdicts_fails_fast(self):
        data = make_valid_dataset()
        bad_resp = {
            "verdicts": [
                {
                    "claim_id": "C1",
                    "verdict": "supported",
                    "reason": "OK",
                    "evidence_verdicts": [{"evidence_id": "E999", "relation": "supports"}],
                }
            ],
            "digest": "sha256:mock_model_digest_12345",
            "prompt_sha256": "c" * 64,
            "stats": {"prompt_eval_count": 100, "eval_count": 10, "eval_duration": 10, "total_duration": 20},
        }
        grounder = MockGrounder(responses=[bad_resp])

        with self.assertRaises(EvaluationError) as ctx:
            evaluate_pairs(data, split="dev", grounder=grounder)

        report = ctx.exception.report
        self.assertIsNotNone(report)
        self.assertEqual(report["run_status"], "error")
        self.assertEqual(report["error"]["case_id"], "pair_dev_01")
        self.assertIn("E999", report["error"]["message"])

    def test_inconsistent_supported_and_contradicts_fails_fast(self):
        data = make_valid_dataset()
        bad_resp = {
            "verdicts": [
                {
                    "claim_id": "C1",
                    "verdict": "supported",
                    "reason": "Inconsistente",
                    "evidence_verdicts": [{"evidence_id": "E1", "relation": "contradicts"}],
                }
            ],
            "digest": "sha256:mock_model_digest_12345",
            "prompt_sha256": "c" * 64,
            "stats": {"prompt_eval_count": 100, "eval_count": 10, "eval_duration": 10, "total_duration": 20},
        }
        grounder = MockGrounder(responses=[bad_resp])

        with self.assertRaises(EvaluationError) as ctx:
            evaluate_pairs(data, split="dev", grounder=grounder)

        report = ctx.exception.report
        self.assertIsNotNone(report)
        self.assertEqual(report["run_status"], "error")
        self.assertIn("Inconsistent verdict", report["error"]["message"])

    def test_prompt_eval_count_exceeding_budget_6913_fails_fast(self):
        data = make_valid_dataset()
        bad_resp = {
            "verdicts": [
                {
                    "claim_id": "C1",
                    "verdict": "supported",
                    "reason": "OK",
                    "evidence_verdicts": [{"evidence_id": "E1", "relation": "supports"}],
                }
            ],
            "digest": "sha256:mock_model_digest_12345",
            "prompt_sha256": "c" * 64,
            "stats": {"prompt_eval_count": 6913, "eval_count": 10, "eval_duration": 10, "total_duration": 20},
        }
        grounder = MockGrounder(responses=[bad_resp])

        with self.assertRaises(EvaluationError) as ctx:
            evaluate_pairs(data, split="dev", grounder=grounder)

        report = ctx.exception.report
        self.assertIsNotNone(report)
        self.assertEqual(report["run_status"], "error")
        self.assertIn("exceeds maximum allowed budget", report["error"]["message"])

    def test_load_duration_nan_fails_fast(self):
        data = make_valid_dataset()
        bad_resp = {
            "verdicts": [
                {
                    "claim_id": "C1",
                    "verdict": "supported",
                    "reason": "OK",
                    "evidence_verdicts": [{"evidence_id": "E1", "relation": "supports"}],
                }
            ],
            "digest": "sha256:mock_model_digest_12345",
            "prompt_sha256": "c" * 64,
            "stats": {
                "prompt_eval_count": 100,
                "eval_count": 10,
                "eval_duration": 10,
                "total_duration": 20,
                "load_duration": float("nan"),
            },
        }
        grounder = MockGrounder(responses=[bad_resp])

        with self.assertRaises(EvaluationError) as ctx:
            evaluate_pairs(data, split="dev", grounder=grounder)

        report = ctx.exception.report
        self.assertIsNotNone(report)
        self.assertEqual(report["run_status"], "error")
        self.assertEqual(report["error_count"], 1)

    def test_extra_stat_outside_whitelist_fails_fast(self):
        data = make_valid_dataset()
        bad_resp = {
            "verdicts": [
                {
                    "claim_id": "C1",
                    "verdict": "supported",
                    "reason": "OK",
                    "evidence_verdicts": [{"evidence_id": "E1", "relation": "supports"}],
                }
            ],
            "digest": "sha256:mock_model_digest_12345",
            "prompt_sha256": "c" * 64,
            "stats": {
                "prompt_eval_count": 100,
                "eval_count": 10,
                "eval_duration": 10,
                "total_duration": 20,
                "extra_metric": 999,
            },
        }
        grounder = MockGrounder(responses=[bad_resp])

        with self.assertRaises(EvaluationError) as ctx:
            evaluate_pairs(data, split="dev", grounder=grounder)

        report = ctx.exception.report
        self.assertIsNotNone(report)
        self.assertEqual(report["run_status"], "error")
        self.assertIn("extra_metric", report["error"]["message"])

    def test_dataset_sha256_nan_or_invalid_rejected_before_client(self):
        data = make_valid_dataset()
        grounder = MockGrounder()
        for invalid_hash in ["not_a_valid_hash", "12345", "z" * 64, float("nan")]:
            with self.subTest(invalid_hash=invalid_hash):
                with self.assertRaises(ValueError):
                    evaluate_pairs(data, split="dev", grounder=grounder, dataset_sha256=invalid_hash)
        # El cliente no debe haber sido llamado ni una vez
        self.assertEqual(len(grounder.recorded_calls), 0)

    def test_dataset_sha256_kind_recording(self):
        data = make_valid_dataset()
        grounder = MockGrounder()
        # Con sha256 proporcionado -> 'rawbytes'
        valid_raw_sha = "f" * 64
        rep1 = evaluate_pairs(data, split="dev", grounder=grounder, dataset_sha256=valid_raw_sha)
        self.assertEqual(rep1["dataset_sha256_kind"], "rawbytes")
        self.assertEqual(rep1["dataset_sha256"], valid_raw_sha)

        # Sin sha256 proporcionado -> 'canonical_json'
        grounder2 = MockGrounder()
        rep2 = evaluate_pairs(data, split="dev", grounder=grounder2)
        self.assertEqual(rep2["dataset_sha256_kind"], "canonical_json")
        self.assertEqual(rep2["dataset_sha256"], compute_canonical_sha256(data))

    def test_fail_fast_accounting_case_2_of_3_fails(self):
        data = make_valid_dataset()
        # 3 casos en split 'dev'
        valid_resp = {
            "verdicts": [
                {
                    "claim_id": "C1",
                    "verdict": "supported",
                    "reason": "OK",
                    "evidence_verdicts": [{"evidence_id": "E1", "relation": "supports"}],
                }
            ],
            "digest": "sha256:mock_model_digest_12345",
            "prompt_sha256": "c" * 64,
            "stats": {"prompt_eval_count": 100, "eval_count": 10, "eval_duration": 10, "total_duration": 20},
        }
        grounder = MockGrounder(responses=[valid_resp, RuntimeError("Fallo en caso 2")])

        with self.assertRaises(EvaluationError) as ctx:
            evaluate_pairs(data, split="dev", grounder=grounder)

        report = ctx.exception.report
        self.assertIsNotNone(report)
        self.assertEqual(report["dataset_count"], 3)
        self.assertEqual(report["completed_count"], 1)
        self.assertEqual(report["error_count"], 1)
        self.assertEqual(report["unprocessed_count"], 1)
        self.assertEqual(report["completed_count"] + report["error_count"] + report["unprocessed_count"], 3)
        self.assertEqual(len(report["cases"]), 1)
        self.assertEqual(report["cases"][0]["id"], "pair_dev_01")
        self.assertEqual(report["error"]["case_id"], "pair_dev_02")
        self.assertEqual(report["metrics"]["total_cases"], 1)

    def test_fail_fast_accounting_case_1_of_3_fails(self):
        data = make_valid_dataset()
        grounder = MockGrounder(responses=[RuntimeError("Fallo inmediato en caso 1")])

        with self.assertRaises(EvaluationError) as ctx:
            evaluate_pairs(data, split="dev", grounder=grounder)

        report = ctx.exception.report
        self.assertIsNotNone(report)
        self.assertEqual(report["dataset_count"], 3)
        self.assertEqual(report["completed_count"], 0)
        self.assertEqual(report["error_count"], 1)
        self.assertEqual(report["unprocessed_count"], 2)
        self.assertEqual(report["completed_count"] + report["error_count"] + report["unprocessed_count"], 3)
        self.assertEqual(len(report["cases"]), 0)
        self.assertEqual(report["metrics"]["total_cases"], 0)

    def test_model_mismatch_rejected_before_calls(self):
        data = make_valid_dataset()
        grounder = MockGrounder(model="qwen3.5:9b")
        with self.assertRaises(ValueError):
            evaluate_pairs(data, split="dev", grounder=grounder, model="qwen3.5:4b-q4_K_M")


class TestFormatTextReport(unittest.TestCase):
    """Pruebas del formateo legible de resultados."""

    def test_format_text_report_contains_key_sections(self):
        data = make_valid_dataset()
        grounder = MockGrounder()
        report = evaluate_pairs(data, split="dev", grounder=grounder)
        text = _format_text_report(report)

        self.assertIn("EVALUACIÓN DE PARES AFIRMACIÓN-EVIDENCIAS (GROUNDING)", text)
        self.assertIn("MATRIZ DE CONFUSIÓN 3-CLASES", text)
        self.assertIn("MÉTRICAS DE RENDIMIENTO", text)
        self.assertIn("CONSUMO DE TOKENS Y ESTADÍSTICAS", text)
        self.assertIn("ADVERTENCIA METODOLÓGICA", text)
        self.assertIn("Exactitud global (Accuracy)", text)
        self.assertIn("Falsa aceptación no apoyadas (FA)", text)


class TestMainCLI(unittest.TestCase):
    """Pruebas de la interfaz de línea de comandos de evaluación de pares."""

    def test_main_cli_with_synthetic_dataset_json(self):
        data = make_valid_dataset()
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as tmpdir:
            temp_path = Path(tmpdir) / "test_pairs.json"
            temp_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

            mock_g = MockGrounder()
            with patch("rag_pensiones.evaluate_grounding.OllamaGrounder", return_value=mock_g):
                with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
                    exit_code = main(["--dataset", str(temp_path), "--split", "dev", "--json"])
                    self.assertEqual(exit_code, 0)
                    output = mock_stdout.getvalue()
                    parsed = json.loads(output)
                    self.assertEqual(parsed["schema_version"], EVALUATION_SCHEMA_VERSION)
                    self.assertEqual(parsed["split"], "dev")
                    self.assertEqual(parsed["cases_count"], 3)
                    self.assertIn("metrics", parsed)
                    self.assertIn("stats", parsed)

    def test_main_cli_with_synthetic_dataset_text(self):
        data = make_valid_dataset()
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as tmpdir:
            temp_path = Path(tmpdir) / "test_pairs.json"
            temp_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

            mock_g = MockGrounder()
            with patch("rag_pensiones.evaluate_grounding.OllamaGrounder", return_value=mock_g):
                with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
                    exit_code = main(["--dataset", str(temp_path), "--split", "dev"])
                    self.assertEqual(exit_code, 0)
                    output = mock_stdout.getvalue()
                    self.assertIn("MATRIZ DE CONFUSIÓN", output)
                    self.assertIn("MÉTRICAS DE RENDIMIENTO", output)

    def test_main_cli_nonexistent_dataset_exits_code_1(self):
        with patch("sys.stderr", new_callable=StringIO) as mock_stderr:
            exit_code = main(["--dataset", "evaluation/nonexistent_dataset_12345.json"])
            self.assertEqual(exit_code, 1)
            err = mock_stderr.getvalue()
            self.assertIn("Error:", err)

    def test_main_cli_evaluation_error_json_outputs_partial_report_and_exits_1(self):
        data = make_valid_dataset()
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as tmpdir:
            temp_path = Path(tmpdir) / "test_pairs.json"
            temp_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

            valid_resp = {
                "verdicts": [
                    {
                        "claim_id": "C1",
                        "verdict": "supported",
                        "reason": "OK",
                        "evidence_verdicts": [{"evidence_id": "E1", "relation": "supports"}],
                    }
                ],
                "digest": "sha256:mock_model_digest_12345",
                "prompt_sha256": "c" * 64,
                "stats": {"prompt_eval_count": 100, "eval_count": 10, "eval_duration": 10, "total_duration": 20},
            }
            mock_g = MockGrounder(responses=[valid_resp, RuntimeError("Fallo simulado caso 2")])
            with patch("rag_pensiones.evaluate_grounding.OllamaGrounder", return_value=mock_g):
                with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
                    exit_code = main(["--dataset", str(temp_path), "--split", "dev", "--json"])
                    self.assertEqual(exit_code, 1)
                    output = mock_stdout.getvalue()
                    parsed = json.loads(output)
                    self.assertEqual(parsed["run_status"], "error")
                    self.assertEqual(parsed["dataset_count"], 3)
                    self.assertEqual(parsed["completed_count"], 1)
                    self.assertEqual(parsed["error_count"], 1)
                    self.assertEqual(parsed["unprocessed_count"], 1)
                    self.assertEqual(parsed["error"]["case_id"], "pair_dev_02")

    def test_main_cli_evaluation_error_text_outputs_partial_report_and_exits_1(self):
        data = make_valid_dataset()
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as tmpdir:
            temp_path = Path(tmpdir) / "test_pairs.json"
            temp_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

            valid_resp = {
                "verdicts": [
                    {
                        "claim_id": "C1",
                        "verdict": "supported",
                        "reason": "OK",
                        "evidence_verdicts": [{"evidence_id": "E1", "relation": "supports"}],
                    }
                ],
                "digest": "sha256:mock_model_digest_12345",
                "prompt_sha256": "c" * 64,
                "stats": {"prompt_eval_count": 100, "eval_count": 10, "eval_duration": 10, "total_duration": 20},
            }
            mock_g = MockGrounder(responses=[valid_resp, RuntimeError("Fallo simulado caso 2")])
            with patch("rag_pensiones.evaluate_grounding.OllamaGrounder", return_value=mock_g):
                with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
                    exit_code = main(["--dataset", str(temp_path), "--split", "dev"])
                    self.assertEqual(exit_code, 1)
                    output = mock_stdout.getvalue()
                    self.assertIn("Estado de ejecución : error", output)
                    self.assertIn("ERROR EN CASO: pair_dev_02", output)


if __name__ == "__main__":
    unittest.main()
