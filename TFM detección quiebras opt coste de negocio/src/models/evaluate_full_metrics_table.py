import os
import sys
import json
import duckdb
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    precision_recall_curve, auc, roc_auc_score, confusion_matrix
)

# Fix path for direct execution
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from catboost import CatBoostClassifier
import xgboost as xgb
from src.models.neural_nets import PyTorchMLPTrainer
from src.models.asymmetric_loss import (
    COST_FN_USD, COST_FP_USD, PROFIT_TN_USD, AVOIDED_TP_USD
)

# Project paths
PARQUET_PATH = Path("data/processed/sec_dataset/v2_sec_financials_pivoted_clean.parquet")
OUTPUT_CSV = Path("data/processed/sec_dataset/benchmark_models_full_metrics.csv")
OUTPUT_JSON = Path("data/processed/sec_dataset/benchmark_models_full_metrics.json")
OPTUNA_JSON = Path("data/processed/sec_dataset/optuna_best_hyperparameters.json")


def load_dataset():
    con = duckdb.connect()
    df = con.execute(f"SELECT * FROM '{PARQUET_PATH}';").df()
    con.close()
    return df


def load_optuna_hyperparams():
    if OPTUNA_JSON.exists():
        with open(OPTUNA_JSON, 'r') as f:
            return json.load(f)
    return {}


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


def prepare_feature_matrix(df):
    meta_cols = {'adsh', 'cik', 'period_date', 'filed_date', 'form_type', 'is_distress_event',
                 'target_bankrupt_12m', 'target_bankrupt_24m', 'company_name', 'sic', 'fy',
                 'filed_year', 'filed_month', 'ticker',
                 'tag_LiabilitiesAndStockholdersEquity', 'macro_real_gdp', 'news_sentiment_std_6m',
                 'tag_CommonStockSharesIssued', 'tag_CommonStockSharesOutstanding', 'tag_LiabilitiesCurrent',
                 'macro_yield_curve'}
    feature_cols = [c for c in df.columns if c not in meta_cols]

    X = df[feature_cols].copy()
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    # Fill NaNs cleanly
    X[num_cols] = X[num_cols].fillna(0.0)

    # Encode categorical
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    if cat_cols:
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

    return X, feature_cols


def run_full_evaluation():
    print("=" * 125)
    print("EVALUACIÓN MULTIMODELO TFM CON HIPERPARÁMETROS OPTIMIZADOS EN OPTUNA GPU & VOTING CLASSIFIER")
    print(f"  - EAD: $10,000,000 USD | NIM: 2.5% ($250k) | LGD: 60.0% ($6.0M) | Loss Ratio: 24.0x")
    print("=" * 125)

    df_raw = load_dataset()
    df_sorted = df_raw.sort_values('filed_date').reset_index(drop=True)
    X, feature_cols = prepare_feature_matrix(df_sorted)

    optuna_params = load_optuna_hyperparams()

    # Apply 365 days purge for 12M horizon, and 730 days (2 full years) for 24M horizon
    horizons = [('12M', 'target_bankrupt_12m', 365), ('24M', 'target_bankrupt_24m', 730)]

    base_model_names = [
        'Logistic Regression',
        'Random Forest',
        'LightGBM',
        'CatBoost GPU',
        'XGBoost',
        'PyTorch MLP (Custom Loss)'
    ]

    all_model_names = base_model_names + ['Ensemble (PR-AUC Weighted Voting)']

    full_results = []

    for horizon_label, target_col, purge_days in horizons:
        print(f"\n--- Evaluando Horizonte: {horizon_label} ({target_col}) | Purga: {purge_days} días ---")
        _, folds = get_purged_folds(df_sorted, date_col='filed_date', n_splits=5, purge_days=purge_days)
        y = df_sorted[target_col].values

        model_fold_metrics = {m: {
            'tn': [], 'fp': [], 'fn': [], 'tp': [],
            'pr_auc': [], 'roc_auc': [], 'rec': [], 'spec': [], 'prec': [], 'tau': []
        } for m in all_model_names}

        for fold_idx, (train_idx, val_idx) in enumerate(folds):
            print(f"\n  [Fold {fold_idx + 1}/{len(folds)}] Entrenando modelos con mejores hiperparámetros Optuna GPU...")
            X_train, y_train = X.iloc[train_idx].values, y[train_idx]
            X_val, y_val = X.iloc[val_idx].values, y[val_idx]

            if len(np.unique(y_train)) < 2 or len(np.unique(y_val)) < 2:
                continue

            imputer = SimpleImputer(strategy='median')
            scaler = StandardScaler()

            X_train_proc = scaler.fit_transform(imputer.fit_transform(X_train))
            X_val_proc = scaler.transform(imputer.transform(X_val))

            fold_base_probs = {}
            fold_base_pr_aucs = {}

            # Train all 6 base models using best Optuna GPU parameters
            for name in base_model_names:
                params = optuna_params.get(name, {})

                if name == 'Logistic Regression':
                    lr_c = params.get('C', 5.57)
                    lr_cw = params.get('class_weight', None)
                    model = LogisticRegression(C=lr_c, class_weight=lr_cw, max_iter=1000, random_state=42)
                    model.fit(X_train_proc, y_train)
                    probs = model.predict_proba(X_val_proc)[:, 1]

                elif name == 'Random Forest':
                    rf_n = params.get('n_estimators', 200)
                    rf_d = params.get('max_depth', 14)
                    rf_mps = params.get('min_samples_split', 10)
                    rf_mpl = params.get('min_samples_leaf', 4)
                    rf_cw = params.get('class_weight', 'balanced_subsample')
                    model = RandomForestClassifier(
                        n_estimators=rf_n, max_depth=rf_d, min_samples_split=rf_mps,
                        min_samples_leaf=rf_mpl, class_weight=rf_cw, random_state=42, n_jobs=-1
                    )
                    model.fit(X_train_proc, y_train)
                    probs = model.predict_proba(X_val_proc)[:, 1]

                elif name == 'LightGBM':
                    lgb_n = params.get('n_estimators', 350)
                    lgb_d = params.get('max_depth', 8)
                    lgb_nl = params.get('num_leaves', 112)
                    lgb_lr = params.get('learning_rate', 0.0102)
                    lgb_ss = params.get('subsample', 0.827)
                    lgb_cs = params.get('colsample_bytree', 0.602)
                    lgb_spw = params.get('scale_pos_weight', 7.90)
                    model = lgb.LGBMClassifier(
                        n_estimators=lgb_n, max_depth=lgb_d, num_leaves=lgb_nl,
                        learning_rate=lgb_lr, subsample=lgb_ss, colsample_bytree=lgb_cs,
                        scale_pos_weight=lgb_spw, random_state=42, verbose=-1, n_jobs=-1
                    )
                    model.fit(X_train_proc, y_train)
                    probs = model.predict_proba(X_val_proc)[:, 1]

                elif name == 'CatBoost GPU':
                    cb_iter = params.get('iterations', 300)
                    cb_d = params.get('depth', 8)
                    cb_lr = params.get('learning_rate', 0.0393)
                    cb_l2 = params.get('l2_leaf_reg', 8.008)
                    cb_rs = params.get('random_strength', 4.036)
                    model = CatBoostClassifier(
                        iterations=cb_iter, depth=cb_d, learning_rate=cb_lr,
                        l2_leaf_reg=cb_l2, random_strength=cb_rs, task_type='GPU', verbose=0, random_seed=42
                    )
                    model.fit(X_train_proc, y_train)
                    probs = model.predict_proba(X_val_proc)[:, 1]

                elif name == 'XGBoost':
                    xgb_n = params.get('n_estimators', 250)
                    xgb_d = params.get('max_depth', 5)
                    xgb_lr = params.get('learning_rate', 0.0747)
                    xgb_ss = params.get('subsample', 0.758)
                    xgb_cs = params.get('colsample_bytree', 0.865)
                    xgb_mcw = params.get('min_child_weight', 6)
                    xgb_spw = params.get('scale_pos_weight', 9.86)
                    model = xgb.XGBClassifier(
                        n_estimators=xgb_n, max_depth=xgb_d, learning_rate=xgb_lr,
                        subsample=xgb_ss, colsample_bytree=xgb_cs, min_child_weight=xgb_mcw,
                        scale_pos_weight=xgb_spw, tree_method='hist', device='cuda', random_state=42
                    )
                    model.fit(X_train_proc, y_train)
                    probs = model.predict_proba(X_val_proc)[:, 1]

                elif name == 'PyTorch MLP (Custom Loss)':
                    hidden_opt = params.get('hidden_dims', '256-128-64')
                    hidden_dims = [int(h) for h in hidden_opt.split('-')]
                    pt_lr = params.get('lr', 0.00261)
                    pt_drop = params.get('dropout_rate', 0.158)
                    pt_cfn = params.get('cost_fn', 24.0)

                    model = PyTorchMLPTrainer(
                        input_dim=X_train_proc.shape[1], hidden_dims=hidden_dims,
                        dropout_rate=pt_drop, lr=pt_lr, cost_fn=pt_cfn
                    )
                    model.fit(X_train_proc, y_train, epochs=25, batch_size=512)
                    probs = model.predict_proba_positive(X_val_proc)

                fold_base_probs[name] = probs

                # Compute PR-AUC for weight allocation
                p_c, r_c, _ = precision_recall_curve(y_val, probs)
                pr_auc_val = auc(r_c, p_c)
                fold_base_pr_aucs[name] = pr_auc_val

            # Compute Weighted Voting Ensemble Probability for this fold
            total_pr_auc = sum(fold_base_pr_aucs.values())
            normalized_weights = {m: fold_base_pr_aucs[m] / total_pr_auc for m in base_model_names}

            ensemble_probs = np.zeros_like(y_val, dtype=np.float32)
            for m in base_model_names:
                ensemble_probs += normalized_weights[m] * fold_base_probs[m]

            fold_base_probs['Ensemble (PR-AUC Weighted Voting)'] = ensemble_probs

            # Evaluate threshold optimization for all 7 models in this fold
            thresholds = np.linspace(0.01, 0.99, 99)

            for m in all_model_names:
                preds_proba = fold_base_probs[m]
                p_c, r_c, _ = precision_recall_curve(y_val, preds_proba)
                pr_auc_val = auc(r_c, p_c)
                roc_auc_val = roc_auc_score(y_val, preds_proba)

                best_cost = float('inf')
                best_tau = 0.5
                best_tn, best_fp, best_fn, best_tp = 0, 0, 0, 0
                best_rec, best_spec, best_prec = 0.0, 0.0, 0.0

                for th in thresholds:
                    preds_bin = (preds_proba >= th).astype(int)
                    tn, fp, fn, tp = confusion_matrix(y_val, preds_bin, labels=[0, 1]).ravel()
                    cost = (fn * COST_FN_USD) + (fp * COST_FP_USD)

                    if cost < best_cost:
                        best_cost = cost
                        best_tau = th
                        best_tn, best_fp, best_fn, best_tp = tn, fp, fn, tp
                        best_rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                        best_spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                        best_prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0

                m_dict = model_fold_metrics[m]
                m_dict['tau'].append(best_tau)
                m_dict['tn'].append(best_tn)
                m_dict['fp'].append(best_fp)
                m_dict['fn'].append(best_fn)
                m_dict['tp'].append(best_tp)
                m_dict['pr_auc'].append(pr_auc_val)
                m_dict['roc_auc'].append(roc_auc_val)
                m_dict['rec'].append(best_rec)
                m_dict['spec'].append(best_spec)
                m_dict['prec'].append(best_prec)

        # Consolidate average metrics across folds for each model
        for name in all_model_names:
            m_dict = model_fold_metrics[name]
            avg_tau = round(float(np.mean(m_dict['tau'])), 3)
            avg_tn = int(np.mean(m_dict['tn']))
            avg_fp = int(np.mean(m_dict['fp']))
            avg_fn = int(np.mean(m_dict['fn']))
            avg_tp = int(np.mean(m_dict['tp']))

            total_cost_usd = (avg_fn * COST_FN_USD) + (avg_fp * COST_FP_USD)
            net_pnl_usd = (avg_tn * PROFIT_TN_USD) + (avg_tp * AVOIDED_TP_USD) - (avg_fp * COST_FP_USD) - (avg_fn * COST_FN_USD)

            res_entry = {
                "Modelo / Algoritmo": name,
                "Horizonte": horizon_label,
                "Umbral (tau*)": avg_tau,
                "PR-AUC": round(float(np.mean(m_dict['pr_auc'])), 4),
                "ROC-AUC": round(float(np.mean(m_dict['roc_auc'])), 4),
                "Recall (Sensibilidad)": round(float(np.mean(m_dict['rec'])), 4),
                "Specificity": round(float(np.mean(m_dict['spec'])), 4),
                "Precision": round(float(np.mean(m_dict['prec'])), 4),
                "TN": avg_tn,
                "FP": avg_fp,
                "FN": avg_fn,
                "TP": avg_tp,
                "Coste Total USD": f"${total_cost_usd:,.2f}",
                "P&L Neto USD": f"${net_pnl_usd:,.2f}"
            }

            full_results.append(res_entry)

    df_res = pd.DataFrame(full_results)

    # Print Table
    print("\n" + "=" * 125)
    print("TABLA OFICIAL DE EVALUACIÓN MULTIMODELO TFM (HIPERPARÁMETROS OPTIMIZADOS EN OPTUNA GPU & VOTING CLASSIFIER)")
    print("=" * 125)
    print(df_res.to_string(index=False))
    print("=" * 125)

    # Save to CSV and JSON
    df_res.to_csv(OUTPUT_CSV, index=False)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(full_results, f, indent=2)

    print(f"\n[Evaluator Engine] Full metrics table saved successfully to:")
    print(f"  - CSV:  {OUTPUT_CSV}")
    print(f"  - JSON: {OUTPUT_JSON}")


if __name__ == "__main__":
    run_full_evaluation()
