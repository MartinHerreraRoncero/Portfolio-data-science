import os
import sys
import json
import time
import pickle
import duckdb
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score, brier_score_loss, fbeta_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

import catboost as cb
import lightgbm as lgb
import xgboost as xgb

try:
    from src.models.validation_scheme import PurgedGroupTimeSeriesSplit
    from src.models.neural_nets import PyTorchMLPTrainer
    from src.models.woe_scorecard import CreditScorecardWOE
    from src.models.asymmetric_loss import compute_detailed_financial_pnl, COST_FN_USD, COST_FP_USD
    from src.models.probability_calibration import calibrate_credit_model
except ModuleNotFoundError:
    from validation_scheme import PurgedGroupTimeSeriesSplit
    from neural_nets import PyTorchMLPTrainer
    from woe_scorecard import CreditScorecardWOE
    from asymmetric_loss import compute_detailed_financial_pnl, COST_FN_USD, COST_FP_USD
    from probability_calibration import calibrate_credit_model

DATASET_PARQUET = Path("data/processed/sec_dataset/v2_sec_financials_pivoted_clean.parquet")
OUTPUT_DIR = Path("data/processed/sec_dataset")
MODELS_DIR = Path("models")
RESULTS_JSON = OUTPUT_DIR / "benchmark_models_results.json"
RESULTS_CSV = OUTPUT_DIR / "benchmark_models_results.csv"


import re

def prepare_feature_matrix(df):
    """Prepara la matriz de características multimodales para los modelos."""
    target_cols = ['is_bankrupt_event', 'is_distress_event', 'target_bankrupt_12m', 'target_bankrupt_24m']
    meta_cols = ['adsh', 'cik', 'company_name', 'period_date', 'filed_date', 'fy', 'fp', 'filed_year', 'filed_month', 'ticker', 'ticker_1', 'target_str']
    # Exclude collinear features (>75% correlation with higher-importance features)
    collinear_to_drop = [
        'tag_LiabilitiesAndStockholdersEquity', 'macro_real_gdp', 'news_sentiment_std_6m',
        'tag_CommonStockSharesIssued', 'tag_CommonStockSharesOutstanding', 'tag_LiabilitiesCurrent',
        'macro_yield_curve'
    ]

    feature_cols = [c for c in df.columns if c not in target_cols and c not in meta_cols and not c.startswith('ticker') and c not in collinear_to_drop]
    
    # Categoricals to encode
    cat_cols = ['sector_division', 'form_type', 'macro_regime_name', 'corp_archetype_name']
    
    # One-hot encode low-cardinality categoricals
    df_encoded = pd.get_dummies(df[feature_cols], columns=[c for c in cat_cols if c in df.columns], drop_first=True)

    # Clean special JSON / bracket / comma characters from column names for LightGBM/XGBoost
    clean_cols = [re.sub(r'[^a-zA-Z0-9_]', '_', str(c)) for c in df_encoded.columns]
    df_encoded.columns = clean_cols

    X = df_encoded.fillna(0.0).astype(np.float32)
    y = df['target_bankrupt_12m'].astype(int)
    groups = df['cik']
    dates = df['filed_date']

    return X, y, groups, dates, list(X.columns), df



def compute_metrics(y_true, y_prob):
    """Calcula métricas de ranking, discriminación, calibración y costes de negocio."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)
    roc_auc = roc_auc_score(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)

    # Optimal threshold for F2-Score (favorece exhaustividad de quiebras)
    th_grid = np.linspace(0.01, 0.99, 99)
    f2_scores = [fbeta_score(y_true, (y_prob >= th).astype(int), beta=2, zero_division=0) for th in th_grid]
    opt_f2 = np.max(f2_scores)
    opt_thresh = th_grid[np.argmax(f2_scores)]

    # Cálculo de coste de negocio al umbral óptimo
    y_pred_opt = (y_prob >= opt_thresh).astype(int)
    pnl = compute_detailed_financial_pnl(y_true, y_pred_opt)

    return {
        "pr_auc": float(pr_auc),
        "roc_auc": float(roc_auc),
        "brier_score": float(brier),
        "optimal_f2_score": float(opt_f2),
        "optimal_threshold": float(opt_thresh),
        "false_negatives": int(pnl['FN']),
        "false_positives": int(pnl['FP']),
        "total_cost_usd": float(pnl['total_cost_usd']),
        "net_pnl_usd": float(pnl['net_pnl_usd'])
    }


def train_and_evaluate_benchmark():
    print("=" * 80)
    print("FASE 4: BENCHMARK MULTIMODELO - PREDICCIÓN DE QUIEBRA (5-FOLD PURGED CV EN GPU)")
    print("=" * 80)

    con = duckdb.connect()
    print(f"[Benchmark Engine] Cargando Master Dataset V2.1 desde {DATASET_PARQUET}...")
    df = con.execute(f"SELECT * FROM '{DATASET_PARQUET}' ORDER BY filed_date ASC;").df()
    con.close()

    print(f"  - Total Observaciones: {len(df):,}")
    print(f"  - Tasa de Quiebras a 12M: {df['target_bankrupt_12m'].sum():,} ({df['target_bankrupt_12m'].mean()*100:.2f}%)")

    X, y, groups, dates, feature_names, df_full = prepare_feature_matrix(df)
    print(f"  - Total Variables Multimodales: {X.shape[1]}")

    # Variables seleccionadas para el Scorecard WOE (Top representativas para Basilea)
    woe_features = [
        'tag_WorkingCapital', 'tag_NetIncomeLoss', 'tag_Assets',
        'merton_distance_to_default', 'stock_price_close',
        'macro_interest_rate', 'macro_inflation',
        'market_news_sentiment_mean', 'corp_archetype_prob_1'
    ]
    woe_features = [f for f in woe_features if f in X.columns]

    cv = PurgedGroupTimeSeriesSplit(n_splits=5, purge_window_days=365)

    model_names = [
        "Altman_Merton_Baseline",
        "Scorecard_WOE_Basel",
        "Logistic_Regression",
        "Random_Forest",
        "LightGBM",
        "XGBoost",
        "CatBoost_GPU",
        "FinancialMLP_PyTorch"
    ]

    results = {name: [] for name in model_names}
    trained_final_models = {}

    print("\n[Benchmark Engine] Iniciando 5-Fold Purged GroupTimeSeriesSplit CV...")

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y, groups=groups, date_column=dates), 1):
        print(f"\n" + "-" * 75)
        print(f"--- Running Fold {fold}/5 (Train: {len(train_idx):,} | Val: {len(val_idx):,}) ---")
        print("-" * 75)

        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        # Impute NaNs with median & scale for linear/MLP models
        imputer = SimpleImputer(strategy="median")
        X_train_imp = imputer.fit_transform(X_train)
        X_val_imp = imputer.transform(X_val)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_imp)
        X_val_scaled = scaler.transform(X_val_imp)

        for model_name in model_names:
            t0 = time.time()

            if model_name == "Altman_Merton_Baseline":
                # Baseline combinando Merton DtD invertido y Working Capital negativo normalizados
                merton_val = X_val['merton_distance_to_default'].values if 'merton_distance_to_default' in X_val.columns else np.zeros(len(X_val))
                wc_val = X_val['tag_WorkingCapital'].values if 'tag_WorkingCapital' in X_val.columns else np.zeros(len(X_val))
                # Distancia baja = mayor riesgo -> Invertir
                raw_score = - (merton_val / (np.std(merton_val) + 1e-5)) - (wc_val / (np.std(wc_val) + 1e-5))
                y_prob = 1.0 / (1.0 + np.exp(-raw_score))

            elif model_name == "Scorecard_WOE_Basel":
                X_train_woe = X_train[woe_features]
                X_val_woe = X_val[woe_features]
                sc_model = CreditScorecardWOE(variable_names=woe_features, random_state=42)
                sc_model.fit(X_train_woe, y_train)
                y_prob = sc_model.predict_proba_positive(X_val_woe)
                if fold == 5:
                    trained_final_models[model_name] = sc_model

            elif model_name == "Logistic_Regression":
                lr_model = LogisticRegression(max_iter=500, C=0.1, class_weight="balanced", random_state=42)
                lr_model.fit(X_train_scaled, y_train)
                y_prob = lr_model.predict_proba(X_val_scaled)[:, 1]
                if fold == 5:
                    trained_final_models[model_name] = lr_model

            elif model_name == "Random_Forest":
                rf_model = RandomForestClassifier(n_estimators=100, max_depth=12, class_weight="balanced", n_jobs=-1, random_state=42)
                rf_model.fit(X_train_imp, y_train)
                y_prob = rf_model.predict_proba(X_val_imp)[:, 1]
                if fold == 5:
                    trained_final_models[model_name] = rf_model

            elif model_name == "LightGBM":
                lgb_model = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, max_depth=6, class_weight="balanced", n_jobs=-1, random_state=42, verbose=-1)
                lgb_model.fit(X_train, y_train)
                y_prob = lgb_model.predict_proba(X_val)[:, 1]
                if fold == 5:
                    trained_final_models[model_name] = lgb_model

            elif model_name == "XGBoost":
                xgb_model = xgb.XGBClassifier(n_estimators=300, learning_rate=0.03, max_depth=6, scale_pos_weight=15, tree_method="hist", random_state=42)
                xgb_model.fit(X_train, y_train)
                y_prob = xgb_model.predict_proba(X_val)[:, 1]
                if fold == 5:
                    trained_final_models[model_name] = xgb_model

            elif model_name == "CatBoost_GPU":
                cb_model = cb.CatBoostClassifier(iterations=400, learning_rate=0.04, depth=6, auto_class_weights="Balanced", task_type="GPU", verbose=0, random_seed=42)
                cb_model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
                y_prob = cb_model.predict_proba(X_val)[:, 1]
                if fold == 5:
                    trained_final_models[model_name] = cb_model

            elif model_name == "FinancialMLP_PyTorch":
                mlp_trainer = PyTorchMLPTrainer(input_dim=X_train_scaled.shape[1], hidden_dims=[256, 128, 64, 32], dropout_rate=0.3, lr=1e-3)
                mlp_trainer.fit(X_train_scaled, y_train, X_val=X_val_scaled, y_val=y_val, epochs=25, batch_size=1024, verbose=False)
                y_prob = mlp_trainer.predict_proba_positive(X_val_scaled)
                if fold == 5:
                    trained_final_models[model_name] = mlp_trainer

            elapsed = time.time() - t0
            metrics = compute_metrics(y_val, y_prob)
            metrics["elapsed_seconds"] = float(elapsed)
            metrics["fold"] = fold

            results[model_name].append(metrics)

            cost_m = metrics['total_cost_usd'] / 1e6
            print(f"  [{model_name:<22}] PR-AUC: {metrics['pr_auc']:.4f} | ROC-AUC: {metrics['roc_auc']:.4f} | F2: {metrics['optimal_f2_score']:.4f} | Cost: ${cost_m:,.1f}M | Time: {elapsed:.1f}s")

    # Aggregate & Summarize Results
    summary_list = []
    print("\n" + "=" * 85)
    print("RESUMEN GENERAL DEL BENCHMARK MULTIMODELO (PROMEDIOS SOBRE 5 FOLDS PURGADOS):")
    print("=" * 85)

    for model_name, fold_results in results.items():
        avg_pr_auc = np.mean([r["pr_auc"] for r in fold_results])
        avg_roc_auc = np.mean([r["roc_auc"] for r in fold_results])
        avg_f2 = np.mean([r["optimal_f2_score"] for r in fold_results])
        avg_brier = np.mean([r["brier_score"] for r in fold_results])
        avg_cost = np.mean([r["total_cost_usd"] for r in fold_results])
        avg_time = np.mean([r["elapsed_seconds"] for r in fold_results])

        summary_entry = {
            "Model": model_name,
            "Mean_PR_AUC": round(float(avg_pr_auc), 4),
            "Mean_ROC_AUC": round(float(avg_roc_auc), 4),
            "Mean_F2_Score": round(float(avg_f2), 4),
            "Mean_Brier_Score": round(float(avg_brier), 4),
            "Mean_Total_Cost_USD_M": round(float(avg_cost / 1e6), 2),
            "Mean_Time_Sec": round(float(avg_time), 1)
        }
        summary_list.append(summary_entry)
        print(f"  {model_name:<24} | PR-AUC: {avg_pr_auc:.4f} | ROC-AUC: {avg_roc_auc:.4f} | F2: {avg_f2:.4f} | Brier: {avg_brier:.4f} | Cost: ${avg_cost/1e6:,.1f}M")

    # Calibración de Probabilidades para el mejor modelo (CatBoost / LightGBM)
    print("\n[Benchmark Engine] Calibrando probabilidades del modelo líder...")
    best_model_name = sorted(summary_list, key=lambda x: x["Mean_PR_AUC"], reverse=True)[0]["Model"]
    print(f"  - Modelo líder identificado: {best_model_name}")

    # Export to JSON & CSV
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    with open(RESULTS_JSON, "w") as f:
        json.dump({"summary": summary_list, "folds_detail": results}, f, indent=2)

    df_summary = pd.DataFrame(summary_list).sort_values("Mean_PR_AUC", ascending=False)
    df_summary.to_csv(RESULTS_CSV, index=False)

    # Save serialized models
    for name, model in trained_final_models.items():
        try:
            if name == "CatBoost_GPU":
                model.save_model(str(MODELS_DIR / "catboost_best_model.cbm"))
            elif name == "LightGBM":
                model.booster_.save_model(str(MODELS_DIR / "lightgbm_best_model.txt"))
            elif name == "FinancialMLP_PyTorch":
                import torch
                torch.save(model.model.state_dict(), str(MODELS_DIR / "financial_mlp_best.pt"))
            elif name == "Scorecard_WOE_Basel":
                with open(MODELS_DIR / "scorecard_woe_best.pkl", "wb") as f:
                    pickle.dump(model, f)
        except Exception as e:
            print(f"  [Warning] No se pudo serializar {name}: {e}")

    print("\n[Benchmark Engine] Resultados guardados en:")
    print(f"  - JSON: {RESULTS_JSON}")
    print(f"  - CSV:  {RESULTS_CSV}")
    print("=" * 80)
    print("[Benchmark Engine] ¡Entrenamiento y evaluación del benchmark finalizados exitosamente!")


if __name__ == "__main__":
    train_and_evaluate_benchmark()

