"""
Módulo de Optimización de Hiperparámetros con Optuna en GPU.
Proyecto TFM: Predicción de Insolvencia en Empresas del S&P 500.

Objetivo de Negocio: Minimizar el Coste Total Financiero en USD
- CFN = $6,000,000 USD (LGD = 60%, EAD = $10M)
- CFP = $250,000 USD (NIM = 2.5%, EAD = $10M)
- Ratio Asimétrico = 24.0x
- Esquema de Validación: 5-Fold Purged GroupTimeSeriesSplit CV (365 días de purgado)
"""

import os
import sys

# Fix path for direct execution
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import json
import duckdb
import optuna
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from catboost import CatBoostClassifier
import xgboost as xgb
from src.models.neural_nets import PyTorchMLPTrainer
from src.models.asymmetric_loss import COST_FN_USD, COST_FP_USD

# Optuna verbosity
optuna.logging.set_verbosity(optuna.logging.WARNING)

PARQUET_PATH_V21 = Path("data/processed/sec_dataset/v2.1_master_financials_dataset.parquet")
PARQUET_PATH_V2 = Path("data/processed/sec_dataset/v2_sec_financials_pivoted_clean.parquet")
OUTPUT_HYPERPARAMS_JSON = Path("data/processed/sec_dataset/optuna_best_hyperparameters.json")


def load_dataset():
    target_path = PARQUET_PATH_V2 if PARQUET_PATH_V2.exists() else PARQUET_PATH_V21
    con = duckdb.connect()
    df = con.execute(f"SELECT * FROM '{target_path}';").df()
    con.close()
    return df



def get_purged_folds(df, date_col='filed_date', n_splits=5, purge_days=365):
    df_sorted = df.sort_values(date_col).reset_index(drop=True)
    dates = pd.to_datetime(df_sorted[date_col])
    unique_dates = np.sort(dates.unique())
    date_splits = np.array_split(unique_dates, n_splits + 1)
    folds = []

    for i in range(n_splits):
        train_date_end = pd.to_datetime(date_splits[i][-1])
        purge_cutoff_date = train_date_end + pd.Timedelta(days=purge_days)
        val_date_start = pd.to_datetime(date_splits[i + 1][0])
        if val_date_start <= purge_cutoff_date:
            val_date_start = purge_cutoff_date
        val_date_end = pd.to_datetime(date_splits[i + 1][-1])

        if val_date_start > val_date_end:
            continue

        train_mask = (dates <= train_date_end)
        val_mask = (dates >= val_date_start) & (dates <= val_date_end)

        train_indices = np.where(train_mask)[0]
        val_indices = np.where(val_mask)[0]

        if len(train_indices) > 0 and len(val_indices) > 0:
            folds.append((train_indices, val_indices))

    return df_sorted, folds


import re

def prepare_feature_matrix(df):
    target_cols = ['is_bankrupt_event', 'is_distress_event', 'target_bankrupt_12m', 'target_bankrupt_24m']
    meta_cols = ['adsh', 'cik', 'company_name', 'period_date', 'filed_date', 'fy', 'fp', 'filed_year', 'filed_month', 'ticker', 'target_str', 'sic', 'sic_description']

    feature_cols = [c for c in df.columns if c not in target_cols and c not in meta_cols]
    
    # Categoricals to encode (only low-cardinality categoricals)
    cat_cols = ['sector_division', 'form_type', 'macro_regime_name', 'corp_archetype_name']
    
    # Filter out any other high-cardinality text/object columns
    numeric_or_allowed = [c for c in feature_cols if c in cat_cols or df[c].dtype != 'object']
    df_filtered = df[numeric_or_allowed]

    df_encoded = pd.get_dummies(df_filtered, columns=[c for c in cat_cols if c in df_filtered.columns], drop_first=True)

    # Clean special JSON / bracket / comma characters from column names for LightGBM/XGBoost
    clean_cols = [re.sub(r'[^a-zA-Z0-9_]', '_', str(c)) for c in df_encoded.columns]
    df_encoded.columns = clean_cols

    X = df_encoded.fillna(0.0).astype(np.float32)
    return X, list(X.columns)



def evaluate_model_cost_on_folds(model_builder_fn, X, y, folds, scale_features=False):
    fold_costs = []

    for train_idx, val_idx in folds:
        X_train, y_train = X.iloc[train_idx].values, y[train_idx]
        X_val, y_val = X.iloc[val_idx].values, y[val_idx]

        if len(np.unique(y_train)) < 2 or len(np.unique(y_val)) < 2:
            continue

        if scale_features:
            scaler = StandardScaler()
            X_train_proc = scaler.fit_transform(X_train)
            X_val_proc = scaler.transform(X_val)
        else:
            X_train_proc = X_train
            X_val_proc = X_val

        model, is_pytorch = model_builder_fn(X_train_proc.shape[1], y_train)

        if is_pytorch:
            model.fit(X_train_proc, y_train, epochs=15, batch_size=1024)
            preds_proba = model.predict_proba_positive(X_val_proc)
        else:
            model.fit(X_train_proc, y_train)
            preds_proba = model.predict_proba(X_val_proc)[:, 1]

        # Find best threshold for minimum cost
        thresholds = np.linspace(0.01, 0.99, 50)
        best_cost = float('inf')

        for th in thresholds:
            preds_bin = (preds_proba >= th).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_val, preds_bin, labels=[0, 1]).ravel()
            cost = (fn * COST_FN_USD) + (fp * COST_FP_USD)
            if cost < best_cost:
                best_cost = cost


        fold_costs.append(best_cost)

    return float(np.mean(fold_costs))


def optimize_all_models(n_trials_per_model=15):
    print("=" * 100)
    print("OPTIMIZACIÓN DE HIPERPARÁMETROS CON OPTUNA EN GPU (MINIMIZACIÓN DE COSTE TOTAL EN USD)")
    print(f"  - Matriz de Costes: FN = $6,000,000 USD | FP = $250,000 USD | EAD = $10M")
    print(f"  - Trials por Modelo: {n_trials_per_model}")
    print("=" * 100)

    df_raw = load_dataset()
    df_sorted, folds = get_purged_folds(df_raw, n_splits=5, purge_days=365)
    X, _ = prepare_feature_matrix(df_sorted)
    y = df_sorted['target_bankrupt_12m'].values

    best_params_all = {}

    # 1. CatBoost GPU
    print("\n[Optuna GPU] 1/6 Optimizando CatBoost GPU...")
    def objective_catboost(trial):
        params = {
            'iterations': trial.suggest_int('iterations', 150, 400, step=50),
            'depth': trial.suggest_int('depth', 4, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0),
            'random_strength': trial.suggest_float('random_strength', 0.1, 5.0),
            'task_type': 'GPU',
            'verbose': 0,
            'random_seed': 42
        }

        def builder(input_dim, y_tr):
            return CatBoostClassifier(**params), False

        return evaluate_model_cost_on_folds(builder, X, y, folds)

    study_cb = optuna.create_study(direction="minimize")
    study_cb.optimize(objective_catboost, n_trials=n_trials_per_model)
    best_params_all['CatBoost GPU'] = study_cb.best_params
    print(f"  -> CatBoost GPU Mínimo Coste: ${study_cb.best_value:,.2f} USD")
    print(f"  -> Mejoress Hiperparámetros: {study_cb.best_params}")

    # 2. XGBoost GPU
    print("\n[Optuna GPU] 2/6 Optimizando XGBoost GPU...")
    def objective_xgboost(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 150, 400, step=50),
            'max_depth': trial.suggest_int('max_depth', 4, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.0, 10.0),
            'tree_method': 'hist',
            'device': 'cuda',
            'random_state': 42
        }

        def builder(input_dim, y_tr):
            return xgb.XGBClassifier(**params), False

        return evaluate_model_cost_on_folds(builder, X, y, folds)

    study_xgb = optuna.create_study(direction="minimize")
    study_xgb.optimize(objective_xgboost, n_trials=n_trials_per_model)
    best_params_all['XGBoost'] = study_xgb.best_params
    print(f"  -> XGBoost Mínimo Coste: ${study_xgb.best_value:,.2f} USD")
    print(f"  -> Mejores Hiperparámetros: {study_xgb.best_params}")

    # 3. LightGBM
    print("\n[Optuna GPU] 3/6 Optimizando LightGBM...")
    def objective_lightgbm(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 150, 400, step=50),
            'max_depth': trial.suggest_int('max_depth', 4, 10),
            'num_leaves': trial.suggest_int('num_leaves', 15, 127),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.0, 10.0),
            'random_state': 42,
            'verbose': -1,
            'n_jobs': -1
        }

        def builder(input_dim, y_tr):
            return lgb.LGBMClassifier(**params), False

        return evaluate_model_cost_on_folds(builder, X, y, folds)

    study_lgb = optuna.create_study(direction="minimize")
    study_lgb.optimize(objective_lightgbm, n_trials=n_trials_per_model)
    best_params_all['LightGBM'] = study_lgb.best_params
    print(f"  -> LightGBM Mínimo Coste: ${study_lgb.best_value:,.2f} USD")
    print(f"  -> Mejores Hiperparámetros: {study_lgb.best_params}")

    # 4. Random Forest
    print("\n[Optuna GPU] 4/6 Optimizando Random Forest...")
    def objective_rf(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 250, step=50),
            'max_depth': trial.suggest_int('max_depth', 8, 16),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
            'class_weight': trial.suggest_categorical('class_weight', ['balanced', 'balanced_subsample', None]),
            'random_state': 42,
            'n_jobs': -1
        }

        def builder(input_dim, y_tr):
            return RandomForestClassifier(**params), False

        return evaluate_model_cost_on_folds(builder, X, y, folds)

    study_rf = optuna.create_study(direction="minimize")
    study_rf.optimize(objective_rf, n_trials=n_trials_per_model)
    best_params_all['Random Forest'] = study_rf.best_params
    print(f"  -> Random Forest Mínimo Coste: ${study_rf.best_value:,.2f} USD")
    print(f"  -> Mejores Hiperparámetros: {study_rf.best_params}")

    # 5. PyTorch MLP (Custom Loss)
    print("\n[Optuna GPU] 5/6 Optimizando PyTorch MLP (Custom Loss)...")
    def objective_pytorch(trial):
        hidden_opt = trial.suggest_categorical('hidden_dims', ['256-128-64', '512-256-128-64', '128-64-32'])
        hidden_dims = [int(h) for h in hidden_opt.split('-')]
        lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
        dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.4)
        cost_fn = trial.suggest_float('cost_fn', 10.0, 40.0)

        def builder(input_dim, y_tr):
            trainer = PyTorchMLPTrainer(
                input_dim=input_dim,
                hidden_dims=hidden_dims,
                dropout_rate=dropout_rate,
                lr=lr,
                cost_fn=cost_fn
            )
            return trainer, True

        return evaluate_model_cost_on_folds(builder, X, y, folds, scale_features=True)

    study_pt = optuna.create_study(direction="minimize")
    study_pt.optimize(objective_pytorch, n_trials=5)
    best_params_all['PyTorch MLP (Custom Loss)'] = study_pt.best_params
    print(f"  -> PyTorch MLP Mínimo Coste: ${study_pt.best_value:,.2f} USD")
    print(f"  -> Mejores Hiperparámetros: {study_pt.best_params}")

    # 6. Logistic Regression
    print("\n[Optuna GPU] 6/6 Optimizando Logistic Regression...")
    def objective_lr(trial):
        C = trial.suggest_float('C', 0.01, 10.0, log=True)
        class_weight = trial.suggest_categorical('class_weight', ['balanced', None])
        params = {'C': C, 'class_weight': class_weight, 'max_iter': 1000, 'random_state': 42}

        def builder(input_dim, y_tr):
            return LogisticRegression(**params), False

        return evaluate_model_cost_on_folds(builder, X, y, folds, scale_features=True)

    study_lr = optuna.create_study(direction="minimize")
    study_lr.optimize(objective_lr, n_trials=5)

    best_params_all['Logistic Regression'] = study_lr.best_params
    print(f"  -> Logistic Regression Mínimo Coste: ${study_lr.best_value:,.2f} USD")
    print(f"  -> Mejores Hiperparámetros: {study_lr.best_params}")

    # Save to JSON
    with open(OUTPUT_HYPERPARAMS_JSON, "w") as f:
        json.dump(best_params_all, f, indent=2)

    print("\n" + "=" * 100)
    print(f"[Optuna GPU] TODOS LOS HIPERPARÁMETROS OPTIMIZADOS Y GUARDADOS EN {OUTPUT_HYPERPARAMS_JSON}")
    print("=" * 100)


if __name__ == "__main__":
    optimize_all_models()
