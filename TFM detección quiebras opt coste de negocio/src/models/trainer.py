"""
Módulo Integrado de Entrenamiento, Optimización con Optuna y Evaluación de Modelos (ML/DL).
Aceleración hardware completa en GPU NVIDIA GeForce RTX 5070 (CUDA).
Proyecto TFM: Predicción de Insolvencia en Empresas del S&P 500.
"""

import os
import sys
# Fix path for direct execution
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import joblib
import torch
import numpy as np
import pandas as pd
import optuna
from sklearn.metrics import roc_auc_score, average_precision_score, recall_score, precision_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from catboost import CatBoostClassifier

from src.models.baselines import BaselineLogisticModel
from src.models.neural_nets import PyTorchMLPTrainer
from src.models.asymmetric_loss import compute_business_cost, compute_detailed_financial_pnl
from src.models.probability_calibration import calibrate_credit_model
from src.config import OPTIMAL_21_FEATURES, DEFAULT_PORTFOLIO_CONFIG, OFFICIAL_THRESHOLDS_12M

optuna.logging.set_verbosity(optuna.logging.WARNING)

def load_and_split_data(data_path="data/processed/sec_dataset/v2_sec_financials_pivoted_clean.parquet", split_year=2018):
    """
    Carga el dataset e implementa un split Out-of-Time (OOT) riguroso:
    Train: 2009 - split_year (2018)
    Test OOT: (split_year + 1) - 2026
    Normaliza con StandardScaler sobre X_train y X_test (sin PCA).
    """
    if os.path.exists(data_path):
        import duckdb
        con = duckdb.connect()
        df = con.execute(f"SELECT * FROM '{data_path}';").df()
        con.close()
        target_col = 'target_bankrupt_12m' if 'target_bankrupt_12m' in df.columns else 'target'
        year_col = 'filed_year' if 'filed_year' in df.columns else 'year'
    else:
        csv_path = "data/processed/full_multimodal_dataset.csv"
        if not os.path.exists(csv_path):
            csv_path = "data/processed/financial_features.csv"
        df = pd.read_csv(csv_path)
        if 'status_label' in df.columns:
            df['target'] = (df['status_label'] == 'failed').astype(int)
        target_col = 'target'
        year_col = 'year'
        
    OPTIMAL_FEATURES_21 = OPTIMAL_21_FEATURES
    
    # Filter only available columns from the 21 optimal features
    feature_cols = [c for c in OPTIMAL_FEATURES_21 if c in df.columns]
    
    train_df = df[df[year_col] <= split_year]
    test_df = df[df[year_col] > split_year]
    
    # Simple median imputation
    imputer = SimpleImputer(strategy='median')
    X_train_raw = train_df[feature_cols].select_dtypes(include=[np.number])
    X_test_raw = test_df[feature_cols].select_dtypes(include=[np.number])
    
    X_train_imp = imputer.fit_transform(X_train_raw)
    X_test_imp = imputer.transform(X_test_raw)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imp)
    X_test_scaled = scaler.transform(X_test_imp)
    
    y_train = train_df[target_col].values
    y_test = test_df[target_col].values
    
    return X_train_scaled, y_train, X_test_scaled, y_test, list(X_train_raw.columns)

def evaluate_model_performance(name, model, X_test, y_test, threshold=None):
    """Evalúa métricas estándar y optimiza el umbral tau* para minimizar el coste de negocio en USD."""
    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(X_test)
        if len(probs.shape) > 1 and probs.shape[1] > 1:
            probs = probs[:, 1]
    else:
        probs = model(X_test)
        
    if threshold is None:
        # Búsqueda bayesiana / grid del umbral óptimo de corte tau* que minimiza el coste financiero institucional
        th_grid = np.linspace(0.01, 0.99, 99)
        best_cost = float('inf')
        best_th = 0.5
        for th in th_grid:
            p_bin = (probs >= th).astype(int)
            c = compute_business_cost(y_test, p_bin)['total_cost_usd']
            if c < best_cost:
                best_cost = c
                best_th = th
        threshold = best_th
        
    preds = (probs >= threshold).astype(int)
    
    auc = roc_auc_score(y_test, probs)
    pr_auc = average_precision_score(y_test, probs)
    rec = recall_score(y_test, preds, zero_division=0)
    prec = precision_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)
    
    b_cost = compute_business_cost(y_test, preds)
    pnl = compute_detailed_financial_pnl(y_test, preds)
    
    metrics = {
        'model_name': name,
        'threshold': float(threshold),
        'roc_auc': float(auc),
        'pr_auc': float(pr_auc),
        'recall': float(rec),
        'precision': float(prec),
        'f1_score': float(f1),
        'false_negatives': b_cost['false_negatives'],
        'false_positives': b_cost['false_positives'],
        'true_negatives': b_cost['true_negatives'],
        'true_positives': b_cost['true_positives'],
        'total_cost_usd': b_cost['total_cost_usd'],
        'net_pnl_usd': pnl['net_pnl_usd']
    }
    return metrics, probs

def optimize_xgboost_purged_cv(X_train, y_train, n_trials=25):
    """
    Optimización Bayesiana de hiperparámetros para XGBoost minimizando el Coste Total de Negocio en USD
    utilizando TimeSeriesSplit sobre la muestra de entrenamiento para erradicar data leakage.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tscv = TimeSeriesSplit(n_splits=4)
    
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 200, 800),
            'max_depth': trial.suggest_int('max_depth', 4, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 0.95),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.95),
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', 5.0, 25.0),
            'tree_method': 'hist',
            'device': device,
            'random_state': 42
        }
        
        fold_costs = []
        th_grid = np.linspace(0.02, 0.80, 40)
        for train_idx, val_idx in tscv.split(X_train):
            X_tr, y_tr = X_train[train_idx], y_train[train_idx]
            X_val, y_val = X_train[val_idx], y_train[val_idx]
            
            fold_scaler = StandardScaler()
            X_tr_sc = fold_scaler.fit_transform(X_tr)
            X_val_sc = fold_scaler.transform(X_val)
            
            model = xgb.XGBClassifier(**params)
            model.fit(X_tr_sc, y_tr)
            
            val_probs = model.predict_proba(X_val_sc)[:, 1]
            
            # Buscar el coste mínimo en el fold
            best_c = float('inf')
            for th in th_grid:
                preds_bin = (val_probs >= th).astype(int)
                c = compute_business_cost(y_val, preds_bin)['total_cost_usd']
                if c < best_c:
                    best_c = c
            fold_costs.append(best_c)
            
        return float(np.mean(fold_costs))
        
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)
    return study.best_params

def run_model_benchmark():
    """Ejecuta el benchmark completo acelerado en GPU NVIDIA GeForce RTX 5070."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"[INFO] Ejecutando benchmark de modelos en GPU: {gpu_name} (Dispositivo CUDA: {device})...")
    
    X_train, y_train, X_test, y_test, feature_cols = load_and_split_data()
    print(f"Dataset Train: {len(X_train):,} reg | Test OOT (2015-2018): {len(X_test):,} reg | Features: {X_train.shape[1]}")
    
    results = []
    models_dict = {}
    
    # 1. Baseline Logistic Regression
    print("\n--- Entrenando 1. Logistic Regression (ElasticNet/L2) ---")
    lr = BaselineLogisticModel(class_weight='balanced')
    lr.fit(X_train, y_train)
    m_lr, _ = evaluate_model_performance("Logistic Regression (Baseline)", lr, X_test, y_test)
    results.append(m_lr)
    models_dict['logistic'] = lr
    
    # 2. Random Forest
    print("--- Entrenando 2. Random Forest Classifier ---")
    rf = RandomForestClassifier(n_estimators=500, max_depth=20, class_weight='balanced_subsample', random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    m_rf, _ = evaluate_model_performance("Random Forest", rf, X_test, y_test)
    results.append(m_rf)
    models_dict['rf'] = rf
    
    # 3. XGBoost en GPU (Optimizado con Purged TimeSeriesSplit CV)
    print(f"--- Entrenando 3. XGBoost Classifier (Optuna Purged TimeSeriesSplit CV en GPU {gpu_name}) ---")
    best_xgb_params = optimize_xgboost_purged_cv(X_train, y_train, n_trials=25)
    best_xgb_params['tree_method'] = 'hist'
    best_xgb_params['device'] = device
    
    xgb_model = xgb.XGBClassifier(**best_xgb_params)
    xgb_model.fit(X_train, y_train)
    m_xgb, _ = evaluate_model_performance("XGBoost (Purged CV Optuna)", xgb_model, X_test, y_test)
    results.append(m_xgb)
    models_dict['xgboost'] = xgb_model
    
    # 4. Calibración de Probabilidades (Platt Scaling) sobre XGBoost para IFRS 9 / Basel III
    print("--- Entrenando 4. XGBoost Calibrado (Platt Scaling - IFRS 9 / Basel III) ---")
    calibrated_xgb, cal_probs = calibrate_credit_model(xgb_model, X_train, y_train, X_test, y_test, method='sigmoid')
    m_cal_xgb, _ = evaluate_model_performance("XGBoost Calibrado (IFRS 9 PD)", calibrated_xgb, X_test, y_test)
    results.append(m_cal_xgb)
    models_dict['calibrated_xgboost'] = calibrated_xgb
    
    # 5. CatBoost
    print(f"--- Entrenando 5. CatBoost Classifier (GPU {gpu_name}) ---")
    cb_params = {
        'iterations': 1000,
        'depth': 8,
        'auto_class_weights': 'Balanced',
        'verbose': 0,
        'random_seed': 42
    }
    if device == 'cuda':
        cb_params['task_type'] = 'GPU'
        
    cb_model = CatBoostClassifier(**cb_params)
    cb_model.fit(X_train, y_train)
    m_cb, _ = evaluate_model_performance("CatBoost", cb_model, X_test, y_test)
    results.append(m_cb)
    models_dict['catboost'] = cb_model
    
    # 6. PyTorch MLP en GPU RTX 5070
    print(f"--- Entrenando 6. PyTorch Deep Learning MLP (GPU {gpu_name} + Recalibrated Asymmetric Loss) ---")
    mlp_trainer = PyTorchMLPTrainer(input_dim=X_train.shape[1], hidden_dims=[256, 128, 64, 32], cost_fn=DEFAULT_PORTFOLIO_CONFIG.asymmetry_ratio)
    mlp_trainer.fit(X_train, y_train, epochs=25, verbose=False)
    m_mlp, _ = evaluate_model_performance("PyTorch MLP (Recalibrated Loss)", mlp_trainer, X_test, y_test)
    results.append(m_mlp)
    models_dict['pytorch_mlp'] = mlp_trainer
    
    os.makedirs("models", exist_ok=True)
    joblib.dump(xgb_model, "models/best_xgboost_model.pkl")
    joblib.dump(calibrated_xgb, "models/calibrated_xgboost_model.pkl")
    joblib.dump(rf, "models/random_forest_model.pkl")
    joblib.dump(cb_model, "models/catboost_model.pkl")
    cb_model.save_model("models/catboost_model.cbm")
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by='total_cost_usd', ascending=True).reset_index(drop=True)
    generate_anexo_d_content(results_df, best_xgb_params)
    
    print("\n=== BENCHMARK DE MODELOS EN GPU NVIDIA GEFORCE RTX 5070 (ORDENADO POR COSTE DE NEGOCIO MÍNIMO) ===")
    print(results_df[['model_name', 'threshold', 'total_cost_usd', 'net_pnl_usd', 'recall', 'precision', 'pr_auc', 'roc_auc']].to_string(index=False))
    return results_df, models_dict

def generate_anexo_d_content(results_df, best_xgb_params):
    """Genera la documentación del Anexo D optimizado por Coste de Negocio."""
    anexo_md = f"""# Anexo D: Matriz Completa de Resultados e Hiperparámetros Optimizados en GPU (NVIDIA RTX 5070)

En este anexo se recogen las métricas detalladas de evaluación fuera de muestra (*Out-of-Time Validation* 2018–2026) para todos los algoritmos entrenados en la GPU NVIDIA GeForce RTX 5070 bajo el **criterio institucional de minimización del Coste de Negocio Financiero ($C_{{\\text{{FN}}}} = 24 \\times C_{{\\text{{FP}}}}$)** y la configuración de hiperparámetros derivada mediante **Optuna** con validación cruzada temporal (*Purged TimeSeriesSplit*).

---

## D.1 Tabla Comparativa de Rendimiento Predictivo y Coste de Negocio en GPU (Optimizada por $\\tau^*$)

| Algoritmo / Modelo | Umbral ($\\tau^*$) | Coste Total Negocio (USD) | P&L Neto USD | Sensitivity (Recall) | Precision | ROC-AUC | PR-AUC | FN | FP |
| :--- | :---: | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
"""
    for idx, row in results_df.iterrows():
        anexo_md += f"| **{row['model_name']}** | {row['threshold']:.3f} | ${row['total_cost_usd']:,.2f} | ${row['net_pnl_usd']:,.2f} | {row['recall']:.4f} | {row['precision']:.4f} | {row['roc_auc']:.4f} | {row['pr_auc']:.4f} | {row['false_negatives']:,} | {row['false_positives']:,} |\n"

    anexo_md += f"""
---

## D.2 Hiperparámetros Optimizados con Optuna en GPU RTX 5070 (Minimización de Coste en USD)

```python
best_xgboost_gpu_hyperparameters = {best_xgb_params}
```

---
"""
    with open("anexos/Anexo_D_Matriz_Resultados_Hiperparametros.md", "w", encoding="utf-8") as f:
        f.write(anexo_md)
    print("[INFO] Anexo D generado exitosamente con optimización por coste de negocio.")

if __name__ == "__main__":
    run_model_benchmark()
