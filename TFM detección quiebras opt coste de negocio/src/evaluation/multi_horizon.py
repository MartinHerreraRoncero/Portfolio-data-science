"""
Módulo de Evaluación Multi-Horizonte (12M vs 24M) y Cumplimiento NIIF 9 / IFRS 9.
Proyecto TFM: Predicción de Insolvencia en Empresas del S&P 500.

Implementa:
1. Evaluación comparativa entre Horizonte 12M (Stage 1) y Horizonte 24M (Stage 2 / Lifetime ECL).
2. Conexión nativa al dataset maestro Parquet (v2_sec_financials_pivoted_clean.parquet) con 21 variables óptimas.
3. Evaluación de los modelos del benchmark oficial (LightGBM SOTA, CatBoost GPU, XGBoost, Random Forest, Logistic).
4. Cómputo de matrices de confusión, sensibilidad (Recall), especificidad, precisión, PR-AUC, ROC-AUC y Coste Financiero USD.
5. Generación de la figura gráfica fig7_multi_horizon_confusion_matrices.png y exportación a CSV/JSON.
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score, recall_score, precision_score, f1_score
import joblib

import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("MultiHorizonEngine")

# Rutas y Constantes
DATASET_CLEAN_PARQUET = Path("data/processed/sec_dataset/v2_sec_financials_pivoted_clean.parquet")
PREPROCESSOR_PATH = Path("models/preprocessor_pipeline.joblib")
OUTPUT_DIR = Path("data/processed/sec_dataset")
FIGURES_DIR = Path("reports/figures")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import (
    OPTIMAL_21_FEATURES,
    OFFICIAL_THRESHOLDS_12M,
    OFFICIAL_THRESHOLDS_24M,
    DEFAULT_PORTFOLIO_CONFIG
)

COST_FN_USD = DEFAULT_PORTFOLIO_CONFIG.cost_fn_usd
COST_FP_USD = DEFAULT_PORTFOLIO_CONFIG.cost_fp_usd
PORTFOLIO_LOAN_VALUE_USD = DEFAULT_PORTFOLIO_CONFIG.ead_usd

ACTIVE_21_FEATURES = OPTIMAL_21_FEATURES
ACTIVE_26_FEATURES = OPTIMAL_21_FEATURES  # Alias para compatibilidad retroactiva

# Umbrales optimos canonicos unificados desde src.config
MODEL_THRESHOLDS_12M = OFFICIAL_THRESHOLDS_12M
MODEL_THRESHOLDS_24M = OFFICIAL_THRESHOLDS_24M


def load_dataset(sample_size: Optional[int] = None, recent_only: bool = False) -> pd.DataFrame:
    """Carga el dataset maestro de 21 variables óptimas para evaluación multi-horizonte."""
    target_path = DATASET_CLEAN_PARQUET
    if not target_path.exists():
        for cand in [Path("data/processed/sec_dataset/v2_sec_financials_pivoted.parquet"),
                     Path("data/processed/full_multimodal_dataset.csv")]:
            if cand.exists():
                target_path = cand
                break
                
    logger.info(f"Cargando dataset desde '{target_path}'...")
    if str(target_path).endswith(".parquet"):
        import duckdb
        con = duckdb.connect()
        where_clause = "WHERE filed_year >= 2020" if recent_only else ""
        limit_clause = f"LIMIT {sample_size}" if sample_size else ""
        query = f"SELECT * FROM '{target_path}' {where_clause} {limit_clause}"
        df = con.execute(query).df()
        con.close()
    else:
        df = pd.read_csv(target_path, nrows=sample_size)
        
    return df


def prepare_features(df: pd.DataFrame, preprocessor: Any) -> np.ndarray:
    """Preprocesa el vector de 21 variables aplicando imputer y scaler."""
    X_raw = df.copy()
    for col in ACTIVE_26_FEATURES:
        if col not in X_raw.columns:
            def_val = preprocessor.get("default_medians", {}).get(col, 0.0) if isinstance(preprocessor, dict) else 0.0
            X_raw[col] = def_val
            
    X_ord = X_raw[ACTIVE_26_FEATURES].copy()
    if 'macro_inflation' in X_ord.columns:
        X_ord.loc[X_ord['macro_inflation'] <= 25.0, 'macro_inflation'] = (
            X_ord.loc[X_ord['macro_inflation'] <= 25.0, 'macro_inflation'] * 15.0 + 200.0
        )
        
    if isinstance(preprocessor, dict) and "imputer" in preprocessor and "scaler" in preprocessor:
        X_imp = preprocessor["imputer"].transform(X_ord)
        X_scaled = preprocessor["scaler"].transform(X_imp)
        return X_scaled
    elif hasattr(preprocessor, "transform"):
        return preprocessor.transform(X_ord)
    else:
        return X_ord.fillna(0.0).values


def run_multi_horizon_evaluation(
    sample_size: Optional[int] = None,
    save_outputs: bool = True
) -> pd.DataFrame:
    """
    Evalúa todos los modelos disponibles en horizontes 12M y 24M,
    generando métricas completas y matrices de confusión.
    """
    df = load_dataset(sample_size=sample_size)
    logger.info(f"Dataset cargado con {len(df):,} observaciones.")
    
    preprocessor = joblib.load(PREPROCESSOR_PATH) if PREPROCESSOR_PATH.exists() else None
    X_proc = prepare_features(df, preprocessor)
    
    # Cargar modelos serializados
    models = {}
    model_paths = {
        'LightGBM': Path("models/lightgbm_model.pkl"),
        'Random Forest': Path("models/random_forest_model.pkl"),
        'CatBoost GPU': Path("models/catboost_model.pkl"),
        'XGBoost': Path("models/best_xgboost_model.pkl"),
        'Logistic Regression': Path("models/logistic_regression_model.pkl")
    }
    
    for name, p in model_paths.items():
        if p.exists():
            try:
                m = joblib.load(p)
                if hasattr(m, "set_params"):
                    try:
                        m.set_params(device="cpu")
                    except Exception:
                        pass
                models[name] = m
                logger.info(f"Modelo cargado: {name}")
            except Exception as e:
                logger.warning(f"No se pudo cargar {name}: {e}")
                
    if not models:
        raise RuntimeError("No se encontraron modelos serializados en models/.")
        
    horizons = [
        ('12M', 'target_bankrupt_12m', MODEL_THRESHOLDS_12M),
        ('24M', 'target_bankrupt_24m', MODEL_THRESHOLDS_24M)
    ]
    
    results = []
    confusion_matrices = {}
    
    for h_label, target_col, thresholds in horizons:
        if target_col not in df.columns:
            logger.warning(f"Target '{target_col}' no encontrado en el dataset.")
            continue
            
        y_true = df[target_col].fillna(0).astype(int).values
        n_pos = int(np.sum(y_true == 1))
        n_neg = int(np.sum(y_true == 0))
        base_rate = float(np.mean(y_true))
        
        logger.info(f"--- Evaluando Horizonte {h_label} (Tasa Base: {base_rate:.2%}, {n_pos:,} quiebras) ---")
        
        for m_name, model in models.items():
            tau = thresholds.get(m_name, 0.15)
            
            # Inferencia
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X_proc)[:, 1]
            elif hasattr(model, "predict"):
                probs = model.predict(X_proc)
            else:
                continue
                
            probs = np.clip(probs, 0.0001, 0.9999)
            preds = (probs >= tau).astype(int)
            
            # Métricas
            cm = confusion_matrix(y_true, preds)
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
            else:
                tn, fp, fn, tp = 0, 0, 0, 0
                
            roc_auc = float(roc_auc_score(y_true, probs)) if len(np.unique(y_true)) > 1 else 0.5
            pr_auc = float(average_precision_score(y_true, probs)) if len(np.unique(y_true)) > 1 else 0.0
            rec = float(recall_score(y_true, preds, zero_division=0))
            prec = float(precision_score(y_true, preds, zero_division=0))
            spec = float(tn / max(tn + fp, 1))
            f1 = float(f1_score(y_true, preds, zero_division=0))
            
            total_cost_usd = float(fn * COST_FN_USD + fp * COST_FP_USD)
            # P&L: Margen neto sobre créditos solventes aprobados (TN) menos pérdidas por fallidos (FN*LGD)
            net_pnl_usd = float((tn * PORTFOLIO_LOAN_VALUE_USD * 0.025) - (fn * COST_FN_USD) - (fp * COST_FP_USD))
            
            confusion_matrices[(m_name, h_label)] = cm
            
            results.append({
                'model_name': m_name,
                'horizon': h_label,
                'target_col': target_col,
                'tau_threshold': tau,
                'pr_auc': round(pr_auc, 4),
                'roc_auc': round(roc_auc, 4),
                'recall_pct': round(rec * 100, 2),
                'precision_pct': round(prec * 100, 2),
                'specificity_pct': round(spec * 100, 2),
                'f1_score': round(f1, 4),
                'false_negatives_fn': int(fn),
                'false_positives_fp': int(fp),
                'true_positives_tp': int(tp),
                'true_negatives_tn': int(tn),
                'total_cost_usd': total_cost_usd,
                'net_pnl_usd': net_pnl_usd,
                'sample_size': len(y_true)
            })
            
    df_results = pd.DataFrame(results)
    
    # Generar gráfico de matrices de confusión
    if save_outputs and confusion_matrices:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig_path = FIGURES_DIR / "fig7_multi_horizon_confusion_matrices.png"
        
        fig_models = ['LightGBM', 'CatBoost GPU', 'Random Forest', 'XGBoost']
        available_plot_models = [m for m in fig_models if (m, '12M') in confusion_matrices]
        
        if available_plot_models:
            fig, axes = plt.subplots(len(available_plot_models), 2, figsize=(10, 3.2 * len(available_plot_models)))
            if len(available_plot_models) == 1:
                axes = np.array([axes])
                
            for r_idx, m_name in enumerate(available_plot_models):
                for c_idx, h_label in enumerate(['12M', '24M']):
                    ax = axes[r_idx, c_idx]
                    cm = confusion_matrices.get((m_name, h_label), np.zeros((2, 2), dtype=int))
                    sns.heatmap(cm, annot=True, fmt=',d', cmap='Blues', ax=ax, cbar=False,
                                xticklabels=['Solvente (0)', 'Quiebra (1)'],
                                yticklabels=['Solvente (0)', 'Quiebra (1)'])
                    ax.set_title(f"{m_name} — Horizonte {h_label}", fontsize=10, fontweight='bold')
                    if c_idx == 0:
                        ax.set_ylabel("Estado Real (Ground Truth)", fontsize=9)
                    else:
                        ax.set_ylabel("")
                    if r_idx == len(available_plot_models) - 1:
                        ax.set_xlabel("Predicción Modelo", fontsize=9)
                    else:
                        ax.set_xlabel("")
                        
            plt.suptitle("Matrices de Confusión: Horizonte 12M (Stage 1) vs. Horizonte 24M (Stage 2 Lifetime ECL)", 
                         fontsize=12, fontweight='bold', y=1.01)
            plt.tight_layout()
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            anexos_fig = Path("anexos/figures")
            if anexos_fig.exists():
                plt.savefig(anexos_fig / "fig7_multi_horizon_confusion_matrices.png", dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Figura multi-horizonte guardada en {fig_path}.")
            
    if save_outputs:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        csv_out = OUTPUT_DIR / "multi_horizon_evaluation_results.csv"
        json_out = OUTPUT_DIR / "multi_horizon_evaluation_results.json"
        df_results.to_csv(csv_out, index=False)
        with open(json_out, "w", encoding="utf-8") as f:
            json.dump(df_results.to_dict(orient="records"), f, indent=2)
        logger.info(f"Resultados guardados en {csv_out} y {json_out}.")
        
    return df_results


if __name__ == "__main__":
    print("=" * 80)
    print("EVALUACIÓN MULTI-HORIZONTE (12M vs 24M) SOBRE 21 VARIABLES ÓPTIMAS (IFRS 9 / CECL)")
    print("=" * 80)
    df_res = run_multi_horizon_evaluation()
    print("\n=== RESUMEN CONSOLIDADO DE RENDIMIENTO MULTI-HORIZONTE ===")
    cols_show = ['model_name', 'horizon', 'tau_threshold', 'pr_auc', 'roc_auc', 'recall_pct', 'specificity_pct', 'false_negatives_fn', 'total_cost_usd']
    print(df_res[cols_show].to_string(index=False))
