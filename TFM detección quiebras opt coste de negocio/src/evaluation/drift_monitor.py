"""
Módulo de Monitorización de Data Drift y Concept/Prediction Drift:
Population Stability Index (PSI) y Kolmogorov-Smirnov (KS).
Proyecto TFM: Predicción de Insolvencia en Empresas del S&P 500.
Conforme a las directrices de gobernanza de modelos de la Reserva Federal (Fed SR 11-7)
y del Banco Central Europeo (EBA Guidelines).

Implementa:
1. Population Stability Index (PSI) y test bi-muestral de Kolmogorov-Smirnov (KS).
2. Monitorización de Feature Drift sobre el espacio canónico de 21 variables óptimas.
3. Monitorización de Prediction / Score Drift sobre las probabilidades predichas por LightGBM SOTA.
4. Conexión nativa al dataset maestro Parquet (v2_sec_financials_pivoted_clean.parquet).
5. Exportación de métricas a CSV y JSON en data/processed/sec_dataset/.
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
import joblib
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("DriftMonitor")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import OPTIMAL_21_FEATURES

ACTIVE_21_FEATURES = OPTIMAL_21_FEATURES
ACTIVE_26_FEATURES = OPTIMAL_21_FEATURES  # Alias para compatibilidad retroactiva

DATASET_CLEAN_PARQUET = Path("data/processed/sec_dataset/v2_sec_financials_pivoted_clean.parquet")
PREPROCESSOR_PATH = Path("models/preprocessor_pipeline.joblib")
MODEL_PATH = Path("models/best_xgboost_model.pkl")
OUTPUT_DIR = Path("data/processed/sec_dataset")


def calculate_psi(expected: np.ndarray, actual: np.ndarray, num_buckets: int = 10, eps: float = 1e-4) -> Tuple[float, str]:
    """
    Calcula el Population Stability Index (PSI) entre la distribución de referencia (Expected)
    y la distribución observada (Actual).
    
    Criterio Regulatorio Estándar:
    - PSI < 0.10: Población estable (Sin acción).
    - 0.10 <= PSI <= 0.25: Deriva moderada (Monitoreo reforzado).
    - PSI > 0.25: Deriva severa / Crítica (Reentrenamiento mandatario).
    """
    expected = np.asarray(expected)
    actual = np.asarray(actual)
    
    # Filtrar infinitos y nulos
    expected = expected[np.isfinite(expected)]
    actual = actual[np.isfinite(actual)]
    
    if len(expected) == 0 or len(actual) == 0:
        return 0.0, "Sin Datos Suficientes"
        
    percentiles = np.linspace(0, 100, num_buckets + 1)
    raw_buckets = np.percentile(expected, percentiles)
    buckets = np.unique(raw_buckets)
    
    if len(buckets) < 2:
        val = buckets[0]
        buckets = np.array([val - 1e-3, val + 1e-3])
        
    buckets[0] = -np.inf
    buckets[-1] = np.inf
    
    expected_counts = np.histogram(expected, bins=buckets)[0]
    actual_counts = np.histogram(actual, bins=buckets)[0]
    
    expected_pct = (expected_counts / len(expected)) + eps
    actual_pct = (actual_counts / len(actual)) + eps
    
    # Normalización para suma unitaria
    expected_pct = expected_pct / np.sum(expected_pct)
    actual_pct = actual_pct / np.sum(actual_pct)
    
    psi_value = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    psi_value = max(float(psi_value), 0.0)
    
    if psi_value < 0.10:
        action = "Estable (Sin Acción)"
    elif 0.10 <= psi_value <= 0.25:
        action = "Deriva Moderada (Monitorear)"
    else:
        action = "CRÍTICO: Data Drift Severo (Reentrenamiento Requerido)"
        
    return float(psi_value), action


def monitor_feature_drift(df_train: pd.DataFrame, df_test: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """
    Monitoriza la degradación de distribuciones entre Train (Referencia) y Test OOT (Monitoreo).
    Aplica KS-Test y PSI sobre cada característica del espacio predictivo.
    """
    drift_report = []
    
    for col in feature_cols:
        if col not in df_train.columns or col not in df_test.columns:
            continue
            
        expected = df_train[col].dropna().values
        actual = df_test[col].dropna().values
        
        if len(expected) == 0 or len(actual) == 0:
            continue
            
        psi, action = calculate_psi(expected, actual)
        
        # KS-Test para variables continuas
        if len(np.unique(expected)) > 2 and len(np.unique(actual)) > 2:
            ks_stat, ks_pvalue = ks_2samp(expected, actual)
            ks_stat_val = float(ks_stat)
            ks_pval_val = float(ks_pvalue)
        else:
            ks_stat_val = 0.0
            ks_pval_val = 1.0
            
        is_drift = bool(psi >= 0.10 or (ks_pval_val < 0.01 and ks_stat_val > 0.05))
        
        drift_report.append({
            'feature': col,
            'psi_value': round(psi, 4),
            'ks_statistic': round(ks_stat_val, 4),
            'ks_p_value': float(f"{ks_pval_val:.4e}"),
            'status': action,
            'drift_detected': is_drift,
            'train_mean': float(np.mean(expected)),
            'test_mean': float(np.mean(actual)),
            'train_std': float(np.std(expected)),
            'test_std': float(np.std(actual))
        })
        
    df_res = pd.DataFrame(drift_report)
    if not df_res.empty:
        df_res = df_res.sort_values(by='psi_value', ascending=False).reset_index(drop=True)
    return df_res


def monitor_prediction_drift(
    df_train: pd.DataFrame, 
    df_test: pd.DataFrame, 
    model=None, 
    preprocessor=None,
    feature_cols: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Monitoriza el Concept Drift / Prediction Drift evaluando la estabilidad de las
    probabilidades de quiebra predichas por el modelo SOTA entre los dos periodos.
    """
    features = feature_cols or ACTIVE_26_FEATURES
    
    if preprocessor is None and PREPROCESSOR_PATH.exists():
        preprocessor = joblib.load(PREPROCESSOR_PATH)
    if model is None and MODEL_PATH.exists():
        model = joblib.load(MODEL_PATH)
        
    if preprocessor is None or model is None:
        logger.warning("Modelo o preprocesador no disponibles para Score Drift.")
        return {}
        
    X_train_raw = df_train.copy()
    X_test_raw = df_test.copy()
    
    # Rellenar columnas faltantes
    for col in features:
        if col not in X_train_raw.columns:
            def_val = preprocessor.get("default_medians", {}).get(col, 0.0) if isinstance(preprocessor, dict) else 0.0
            X_train_raw[col] = def_val
        if col not in X_test_raw.columns:
            def_val = preprocessor.get("default_medians", {}).get(col, 0.0) if isinstance(preprocessor, dict) else 0.0
            X_test_raw[col] = def_val
            
    X_train_ordered = X_train_raw[features].copy()
    X_test_ordered = X_test_raw[features].copy()
    
    # Manejo de macro_inflation para evitar skew
    for d in [X_train_ordered, X_test_ordered]:
        if 'macro_inflation' in d.columns:
            d.loc[d['macro_inflation'] <= 25.0, 'macro_inflation'] = (
                d.loc[d['macro_inflation'] <= 25.0, 'macro_inflation'] * 15.0 + 200.0
            )
            
    # Transformación con el preprocesador oficial
    if isinstance(preprocessor, dict) and "imputer" in preprocessor and "scaler" in preprocessor:
        X_tr_imp = preprocessor["imputer"].transform(X_train_ordered)
        X_tr_scaled = preprocessor["scaler"].transform(X_tr_imp)
        X_te_imp = preprocessor["imputer"].transform(X_test_ordered)
        X_te_scaled = preprocessor["scaler"].transform(X_te_imp)
    elif hasattr(preprocessor, "transform"):
        X_tr_scaled = preprocessor.transform(X_train_ordered)
        X_te_scaled = preprocessor.transform(X_test_ordered)
    else:
        X_tr_scaled = X_train_ordered.fillna(0.0).values
        X_te_scaled = X_test_ordered.fillna(0.0).values
    
    # Inferencia
    if hasattr(model, "predict_proba"):
        probs_train = model.predict_proba(X_tr_scaled)[:, 1]
        probs_test = model.predict_proba(X_te_scaled)[:, 1]
    else:
        probs_train = model.predict(X_tr_scaled)
        probs_test = model.predict(X_te_scaled)
        
    psi_score, action_score = calculate_psi(probs_train, probs_test)
    ks_stat, ks_pval = ks_2samp(probs_train, probs_test)
    
    return {
        'prediction_psi': round(psi_score, 4),
        'prediction_ks_stat': round(float(ks_stat), 4),
        'prediction_ks_pval': float(f"{ks_pval:.4e}"),
        'status': action_score,
        'train_avg_pd': round(float(np.mean(probs_train)), 4),
        'test_avg_pd': round(float(np.mean(probs_test)), 4),
        'train_median_pd': round(float(np.median(probs_train)), 4),
        'test_median_pd': round(float(np.median(probs_test)), 4),
    }


def run_drift_monitoring(
    dataset_path: Optional[Path] = None,
    split_year: int = 2021,
    sample_size: Optional[int] = None,
    save_reports: bool = True
) -> Dict[str, Any]:
    """
    Ejecuta el pipeline completo de auditoría de Data Drift y Prediction Drift
    sobre el dataset maestro de 21 variables óptimas.
    """
    target_path = dataset_path or DATASET_CLEAN_PARQUET
    if not Path(target_path).exists():
        for cand in [Path("data/processed/sec_dataset/v2_sec_financials_pivoted.parquet"),
                     Path("data/processed/full_multimodal_dataset.csv")]:
            if cand.exists():
                target_path = cand
                break
                
    logger.info(f"Cargando dataset para monitorización de drift desde '{target_path}'...")
    if str(target_path).endswith(".parquet"):
        import duckdb
        con = duckdb.connect()
        limit_clause = f"LIMIT {sample_size}" if sample_size else ""
        query = f"SELECT * FROM '{target_path}' {limit_clause}"
        df = con.execute(query).df()
        con.close()
    else:
        df = pd.read_csv(target_path, nrows=sample_size)
        
    # Verificar columnas
    available_features = [c for c in ACTIVE_21_FEATURES if c in df.columns]
    logger.info(f"Variables activas encontradas: {len(available_features)} / {len(ACTIVE_21_FEATURES)}")
    
    # Partición temporal
    year_col = 'filed_year' if 'filed_year' in df.columns else 'year'
    if year_col not in df.columns and 'filed_date' in df.columns:
        df[year_col] = pd.to_datetime(df['filed_date']).dt.year
        
    df_train = df[df[year_col] <= split_year].copy()
    df_test = df[df[year_col] > split_year].copy()
    
    logger.info(f"Población Referencia (Train <= {split_year}): {len(df_train):,} filas")
    logger.info(f"Población Monitoreo (Test OOT > {split_year}): {len(df_test):,} filas")
    
    if len(df_train) == 0 or len(df_test) == 0:
        # Fallback si no hay año: división cronológica 80/20
        split_idx = int(len(df) * 0.80)
        df_train = df.iloc[:split_idx].copy()
        df_test = df.iloc[split_idx:].copy()
        logger.info("Partición temporal por fecha no disponible; usando partición secuencial 80/20.")
        
    # 1. Feature Drift
    logger.info("Calculando Population Stability Index (PSI) y KS-Test para cada variable...")
    feature_report = monitor_feature_drift(df_train, df_test, available_features)
    
    # 2. Prediction Drift
    logger.info("Calculando Prediction Drift / Score Drift con modelo LightGBM...")
    score_report = monitor_prediction_drift(df_train, df_test, feature_cols=available_features)
    
    # 3. Métricas Globales de Gobernanza
    stable_count = int(np.sum(feature_report['psi_value'] < 0.10))
    moderate_count = int(np.sum((feature_report['psi_value'] >= 0.10) & (feature_report['psi_value'] <= 0.25)))
    severe_count = int(np.sum(feature_report['psi_value'] > 0.25))
    
    summary = {
        'total_features_monitored': len(feature_report),
        'stable_features_psi_under_010': stable_count,
        'moderate_drift_psi_010_to_025': moderate_count,
        'severe_drift_psi_over_025': severe_count,
        'max_psi_feature': feature_report.iloc[0]['feature'] if not feature_report.empty else None,
        'max_psi_value': float(feature_report.iloc[0]['psi_value']) if not feature_report.empty else 0.0,
        'prediction_drift': score_report,
        'governance_verdict': "MODELO APTO PARA PRODUCCIÓN (Sin Deriva Crítica Global)" if severe_count <= 2 and score_report.get('prediction_psi', 0) < 0.25 else "ALERTA: REVISIÓN DE GOBERNANZA REQUERIDA"
    }
    
    if save_reports:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        csv_path = OUTPUT_DIR / "data_drift_psi_ks_report.csv"
        json_path = OUTPUT_DIR / "data_drift_psi_ks_report.json"
        
        feature_report.to_csv(csv_path, index=False)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({
                'summary': summary,
                'feature_drift': feature_report.to_dict(orient='records'),
                'score_drift': score_report
            }, f, indent=2)
        logger.info(f"Informes de drift guardados en {csv_path} y {json_path}.")
        
    return {
        'summary': summary,
        'feature_report': feature_report,
        'score_report': score_report
    }


if __name__ == "__main__":
    print("=" * 80)
    print("MONITORIZACIÓN DE DATA DRIFT Y PREDICTION DRIFT (FED SR 11-7 / EBA)")
    print("=" * 80)
    res = run_drift_monitoring(sample_size=None, split_year=2021)
    
    print("\n=== RESUMEN EJECUTIVO DE GOBERNANZA DE MODELOS ===")
    print(f"Total Variables Auditadas:     {res['summary']['total_features_monitored']}")
    print(f"Variables Estables (PSI < 0.10): {res['summary']['stable_features_psi_under_010']}")
    print(f"Deriva Moderada (0.10-0.25):     {res['summary']['moderate_drift_psi_010_to_025']}")
    print(f"Deriva Severa (PSI > 0.25):      {res['summary']['severe_drift_psi_over_025']}")
    print(f"Score Drift (Prediction PSI):   {res['score_report'].get('prediction_psi', 'N/A')} ({res['score_report'].get('status', 'N/A')})")
    print(f"Dictamen Final:                  {res['summary']['governance_verdict']}")
    print("-" * 80)
    print("\n=== TOP 10 VARIABLES POR POPULATION STABILITY INDEX (PSI) ===")
    print(res['feature_report'][['feature', 'psi_value', 'ks_statistic', 'ks_p_value', 'status']].head(10).to_string(index=False))
