"""
Evaluación de Rendimiento de Target Dual: Causalidad, Transiciones Puras y Persistencia de Distress.
Proyecto TFM: Predicción de Insolvencia en Empresas del S&P 500 y Mercado de EE.UU.

Este script desglosa el rendimiento del modelo líder LightGBM en tres poblaciones clave:
1. Panel Global (N = 393,331 / N_total)
2. Subconjunto A: Transiciones Puras (firmas sanas en t: is_distress_event == 0, evaluando transición a quiebra/distress en t+12m)
3. Subconjunto B: Persistencia de Distress (firmas ya en distress en t: is_distress_event == 1, evaluando persistencia a 12m)

Métricas computadas:
- N, Positivos, Tasa Base (%)
- PR-AUC, ROC-AUC
- Sensibilidad / Recall, Precisión, Especificidad
- Falsos Negativos (FN), Falsos Positivos (FP), Verdaderos Positivos (TP), Verdaderos Negativos (TN)
- Coste Financiero en USD ($FN * 6.0M + $FP * 250k) tanto al umbral de producción (tau=0.150) como al umbral óptimo (tau*).

Exporta los resultados a:
- data/processed/sec_dataset/dual_target_evaluation_metrics.json
- data/processed/sec_dataset/dual_target_evaluation_metrics.csv
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, List

import duckdb
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score, confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import DEFAULT_PORTFOLIO_CONFIG, OFFICIAL_THRESHOLDS_12M

# Parametros Globales de Exposicion Financiera unificados desde src.config
COST_FN_USD = DEFAULT_PORTFOLIO_CONFIG.cost_fn_usd
COST_FP_USD = DEFAULT_PORTFOLIO_CONFIG.cost_fp_usd
PROFIT_TN_USD = DEFAULT_PORTFOLIO_CONFIG.profit_tn_usd
AVOIDED_TP_USD = DEFAULT_PORTFOLIO_CONFIG.avoided_tp_usd
PRODUCTION_THRESHOLD = OFFICIAL_THRESHOLDS_12M.get('XGBoost', 0.158)  # Umbral canonico oficial (0.158) para XGBoost

DATASET_PATH = Path("data/processed/sec_dataset/v2_sec_financials_pivoted_clean.parquet")
MODEL_PATH = Path("models/best_xgboost_model.pkl")
PREPROCESSOR_PATH = Path("models/preprocessor_pipeline.joblib")
OUTPUT_DIR = Path("data/processed/sec_dataset")
OUTPUT_CSV = OUTPUT_DIR / "dual_target_evaluation_metrics.csv"
OUTPUT_JSON = OUTPUT_DIR / "dual_target_evaluation_metrics.json"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("DualTargetEvaluator")


def load_artifacts():
    """Carga el modelo líder LightGBM y el pipeline de preprocesamiento serializado."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"No se encontró el modelo LightGBM en: {MODEL_PATH}")
    if not PREPROCESSOR_PATH.exists():
        raise FileNotFoundError(f"No se encontró el preprocesador en: {PREPROCESSOR_PATH}")

    logger.info(f"Cargando preprocesador desde {PREPROCESSOR_PATH}...")
    preprocessor = joblib.load(PREPROCESSOR_PATH)

    logger.info(f"Cargando modelo LightGBM desde {MODEL_PATH}...")
    model = joblib.load(MODEL_PATH)

    # Configurar para inferencia en CPU si aplica
    if hasattr(model, "set_params"):
        try:
            model.set_params(device="cpu")
        except Exception:
            pass

    return model, preprocessor


def evaluate_dual_target_performance():
    """Ejecuta la evaluación empírica y causal en las tres particiones metodológicas."""
    print("=" * 110)
    print("EVALUACIÓN METODOLÓGICA DE TARGET DUAL: CAUSALIDAD Y PERSISTENCIA DE DISTRESS (LIGHTGBM SOTA)")
    print(f"Parámetros Financieros: Coste FN = ${COST_FN_USD:,.0f} USD | Coste FP = ${COST_FP_USD:,.0f} USD (Ratio 24.0x)")
    print(f"Umbral de Operación en Producción: tau = {PRODUCTION_THRESHOLD:.3f}")
    print("=" * 110)

    # 1. Cargar artefactos
    model, preprocessor = load_artifacts()
    feature_names = preprocessor["feature_names"]
    imputer = preprocessor["imputer"]
    scaler = preprocessor["scaler"]

    # 2. Cargar dataset maestro
    logger.info(f"Cargando dataset maestro desde {DATASET_PATH}...")
    con = duckdb.connect()
    # Extraer columnas necesarias
    required_cols = list(set(feature_names + ["target_bankrupt_12m", "is_distress_event", "cik", "filed_date"]))
    query = f"SELECT {', '.join(required_cols)} FROM '{DATASET_PATH.as_posix()}' ORDER BY filed_date ASC;"
    df = con.execute(query).df()
    con.close()

    total_rows = len(df)
    logger.info(f"Dataset cargado con éxito. Total observaciones: {total_rows:,}")

    # 3. Preprocesar características e inferir probabilidades
    logger.info("Transformando matriz de características e infiriendo probabilidades con LightGBM...")
    X_raw = df[feature_names]
    X_imp = imputer.transform(X_raw)
    X_scaled = scaler.transform(X_imp)

    y_probs = model.predict_proba(X_scaled)[:, 1]
    y_true = df["target_bankrupt_12m"].values.astype(int)
    is_distress = df["is_distress_event"].values.astype(int)

    # 4. Definición de Subconjuntos
    subsets_def = {
        "Panel Global (Total Observaciones)": {
            "mask": np.ones(total_rows, dtype=bool),
            "description": "Total del universo de estados financieros auditados."
        },
        "Subconjunto A: Transiciones Puras (Sanas en t -> Quiebra 12m)": {
            "mask": (is_distress == 0),
            "description": "Firmas sanas en t (is_distress_event == 0), evaluando la predicción causal de transición ex-ante a insolvencia."
        },
        "Subconjunto B: Persistencia de Distress (Distress en t -> Distress 12m)": {
            "mask": (is_distress == 1),
            "description": "Firmas ya en distress técnico en t (is_distress_event == 1), evaluando la persistencia estructural a 12m."
        }
    }

    results_table: List[Dict[str, Any]] = []
    json_export_data: Dict[str, Any] = {
        "metadata": {
            "dataset": str(DATASET_PATH),
            "model": str(MODEL_PATH),
            "n_total_observations": total_rows,
            "cost_fn_usd": COST_FN_USD,
            "cost_fp_usd": COST_FP_USD,
            "production_threshold": PRODUCTION_THRESHOLD,
            "evaluation_timestamp": pd.Timestamp.utcnow().isoformat()
        },
        "subsets": {}
    }

    # Rejilla para umbral óptimo
    th_grid = np.linspace(0.01, 0.99, 99)

    for subset_name, info in subsets_def.items():
        mask = info["mask"]
        y_sub = y_true[mask]
        p_sub = y_probs[mask]

        n_sub = len(y_sub)
        positives_sub = int(np.sum(y_sub))
        base_rate_sub = float(positives_sub / n_sub) if n_sub > 0 else 0.0

        # PR-AUC y ROC-AUC
        prec_curve, rec_curve, _ = precision_recall_curve(y_sub, p_sub)
        pr_auc_val = float(auc(rec_curve, prec_curve))
        try:
            roc_auc_val = float(roc_auc_score(y_sub, p_sub))
        except Exception:
            roc_auc_val = 0.5

        # 4.1. Evaluación al Umbral de Producción (tau = 0.150)
        y_pred_prod = (p_sub >= PRODUCTION_THRESHOLD).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_sub, y_pred_prod, labels=[0, 1]).ravel()
        recall_prod = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        precision_prod = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        specificity_prod = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        cost_usd_prod = float((fn * COST_FN_USD) + (fp * COST_FP_USD))
        pnl_neto_usd_prod = float((tn * PROFIT_TN_USD) + (tp * AVOIDED_TP_USD) - (fp * COST_FP_USD) - (fn * COST_FN_USD))

        # 4.2. Búsqueda de Umbral Óptimo Financiero (tau*)
        best_cost = float("inf")
        best_tau = 0.5
        best_tn, best_fp, best_fn, best_tp = 0, 0, 0, 0
        best_rec, best_prec, best_spec = 0.0, 0.0, 0.0

        for th in th_grid:
            y_pred_th = (p_sub >= th).astype(int)
            c_tn, c_fp, c_fn, c_tp = confusion_matrix(y_sub, y_pred_th, labels=[0, 1]).ravel()
            th_cost = (c_fn * COST_FN_USD) + (c_fp * COST_FP_USD)
            if th_cost < best_cost:
                best_cost = th_cost
                best_tau = float(th)
                best_tn, best_fp, best_fn, best_tp = int(c_tn), int(c_fp), int(c_fn), int(c_tp)
                best_rec = float(c_tp / (c_tp + c_fn)) if (c_tp + c_fn) > 0 else 0.0
                best_prec = float(c_tp / (c_tp + c_fp)) if (c_tp + c_fp) > 0 else 0.0
                best_spec = float(c_tn / (c_tn + c_fp)) if (c_tn + c_fp) > 0 else 0.0

        best_pnl_usd = float((best_tn * PROFIT_TN_USD) + (best_tp * AVOIDED_TP_USD) - (best_fp * COST_FP_USD) - (best_fn * COST_FN_USD))

        # Registrar fila resumida para CSV
        row_summary = {
            "Subconjunto": subset_name,
            "N": n_sub,
            "Positivos": positives_sub,
            "Tasa Base (%)": round(base_rate_sub * 100, 2),
            "PR-AUC": round(pr_auc_val, 4),
            "ROC-AUC": round(roc_auc_val, 4),
            "Sensitivity (Recall) @tau=0.15": round(recall_prod, 4),
            "Precision @tau=0.15": round(precision_prod, 4),
            "Specificity @tau=0.15": round(specificity_prod, 4),
            "FN @tau=0.15": int(fn),
            "FP @tau=0.15": int(fp),
            "Coste Financiero USD @tau=0.15": round(cost_usd_prod, 2),
            "Coste Financiero ($M) @tau=0.15": round(cost_usd_prod / 1e6, 2),
            "P&L Neto USD @tau=0.15": round(pnl_neto_usd_prod, 2),
            "Umbral Óptimo (tau*)": round(best_tau, 3),
            "Sensitivity (Recall) @tau*": round(best_rec, 4),
            "Precision @tau*": round(best_prec, 4),
            "Specificity @tau*": round(best_spec, 4),
            "FN @tau*": int(best_fn),
            "FP @tau*": int(best_fp),
            "Coste Financiero USD @tau*": round(best_cost, 2),
            "Coste Financiero ($M) @tau*": round(best_cost / 1e6, 2),
            "P&L Neto USD @tau*": round(best_pnl_usd, 2)
        }
        results_table.append(row_summary)

        # Estructura detallada para JSON
        json_export_data["subsets"][subset_name] = {
            "description": info["description"],
            "sample_size_N": n_sub,
            "positive_events": positives_sub,
            "negative_events": n_sub - positives_sub,
            "base_rate": base_rate_sub,
            "base_rate_pct": round(base_rate_sub * 100, 2),
            "ranking_metrics": {
                "pr_auc": pr_auc_val,
                "roc_auc": roc_auc_val
            },
            "production_threshold_metrics": {
                "threshold_tau": PRODUCTION_THRESHOLD,
                "confusion_matrix": {
                    "true_negatives": int(tn),
                    "false_positives": int(fp),
                    "false_negatives": int(fn),
                    "true_positives": int(tp)
                },
                "classification_metrics": {
                    "sensitivity_recall": recall_prod,
                    "precision": precision_prod,
                    "specificity": specificity_prod
                },
                "financial_impact": {
                    "fn_loss_usd": float(fn * COST_FN_USD),
                    "fp_loss_usd": float(fp * COST_FP_USD),
                    "total_cost_usd": cost_usd_prod,
                    "total_cost_usd_millions": round(cost_usd_prod / 1e6, 2),
                    "net_pnl_usd": pnl_neto_usd_prod,
                    "net_pnl_usd_millions": round(pnl_neto_usd_prod / 1e6, 2)
                }
            },
            "optimal_threshold_metrics": {
                "optimal_threshold_tau": best_tau,
                "confusion_matrix": {
                    "true_negatives": best_tn,
                    "false_positives": best_fp,
                    "false_negatives": best_fn,
                    "true_positives": best_tp
                },
                "classification_metrics": {
                    "sensitivity_recall": best_rec,
                    "precision": best_prec,
                    "specificity": best_spec
                },
                "financial_impact": {
                    "fn_loss_usd": float(best_fn * COST_FN_USD),
                    "fp_loss_usd": float(best_fp * COST_FP_USD),
                    "total_cost_usd": best_cost,
                    "total_cost_usd_millions": round(best_cost / 1e6, 2),
                    "net_pnl_usd": best_pnl_usd,
                    "net_pnl_usd_millions": round(best_pnl_usd / 1e6, 2)
                }
            }
        }

    # 5. Exportar Resultados a CSV y JSON
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df_results = pd.DataFrame(results_table)
    df_results.to_csv(OUTPUT_CSV, index=False)
    logger.info(f"Resultados exportados a CSV: {OUTPUT_CSV}")

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(json_export_data, f, indent=2, ensure_ascii=False)
    logger.info(f"Resultados exportados a JSON: {OUTPUT_JSON}")

    # 6. Mostrar Informe en Pantalla
    print("\n" + "=" * 115)
    print("TABLA RESUMEN DE RENDIMIENTO DUAL: GLOBAL VS TRANSICIONES PURAS VS PERSISTENCIA")
    print("=" * 115)
    display_cols = [
        "Subconjunto", "N", "Positivos", "Tasa Base (%)", "PR-AUC", "ROC-AUC",
        "Sensitivity (Recall) @tau=0.15", "Precision @tau=0.15", "FN @tau=0.15", "FP @tau=0.15",
        "Coste Financiero ($M) @tau=0.15"
    ]
    print(df_results[display_cols].to_string(index=False))
    print("=" * 115)

    print("\n" + "=" * 115)
    print("COMPARACIÓN CON UMBRAL ÓPTIMO (MINIMIZACIÓN DE COSTE FINANCIERO)")
    print("=" * 115)
    opt_display_cols = [
        "Subconjunto", "Umbral Óptimo (tau*)", "Sensitivity (Recall) @tau*", "Precision @tau*",
        "FN @tau*", "FP @tau*", "Coste Financiero ($M) @tau*"
    ]
    print(df_results[opt_display_cols].to_string(index=False))
    print("=" * 115)

    return df_results, json_export_data


if __name__ == "__main__":
    evaluate_dual_target_performance()
