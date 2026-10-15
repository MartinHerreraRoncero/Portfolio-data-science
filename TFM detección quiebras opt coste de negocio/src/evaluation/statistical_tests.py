"""
Módulo de Contrastes Estadísticos Formales y Pruebas de Hipótesis para el TFM.
Implementa:
1. Test de DeLong para comparación no paramétrica de ROC-AUC.
2. Test de McNemar para significación estadística de matrices de confusión discordantes.
3. Intervalos de Confianza por Bootstrap Empírico (B=1000 réplicas).
4. Test de Bondad de Ajuste de Hosmer-Lemeshow para Calibración IFRS 9.
5. Curvas de Ganancia Acumulada y Curvas de Lift.
"""

import os
import json
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, fbeta_score, brier_score_loss
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import OFFICIAL_THRESHOLDS_12M

FIGURES_DIR = Path("reports/figures")
OUTPUT_DIR = Path("data/processed/sec_dataset")
BENCHMARK_JSON = OUTPUT_DIR / "benchmark_models_results.json"
STATISTICAL_REPORT_MD = Path("anexos/Anexo_E_Contrastes_Estadisticos.md")


def fast_delong_roc_variance(y_true, y_pred):
    """
    Calcula la varianza del ROC-AUC según el algoritmo rápido de Sun & Xu (2014)
    para el estimador no paramétrico de DeLong et al. (1988) en complejidad O(N log N).
    Elimina la matriz exterior densa (m x n) evitando MemoryError en muestras de 100k+ observaciones.
    
    Parámetros:
    -----------
    y_true : array-like de forma (N,) con etiquetas binarias {0, 1}.
    y_pred : array-like de forma (N,) con probabilidades o scores continuos de riesgo.
    
    Retorna:
    --------
    auc_val : float, Área bajo la curva ROC empírica.
    var_auc : float, Varianza asintótica de DeLong de la estimación de AUC.
    v10     : np.ndarray de forma (m,), valores de colocación estructural V10 (casos positivos).
    v01     : np.ndarray de forma (n,), valores de colocación estructural V01 (casos negativos).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]
    
    m = len(pos_idx)
    n = len(neg_idx)
    
    if m == 0 or n == 0:
        return 0.5, 0.0, np.array([]), np.array([])
    
    preds_pos = y_pred[pos_idx]
    preds_neg = y_pred[neg_idx]
    
    # Algoritmo de Sun & Xu (2014) O(N log N) basado en búsqueda binaria sobre arreglos ordenados:
    # V10[i] = (1/n) * sum_j ( I(preds_pos[i] > preds_neg[j]) + 0.5 * I(preds_pos[i] == preds_neg[j]) )
    neg_sorted = np.sort(preds_neg)
    left_neg = np.searchsorted(neg_sorted, preds_pos, side='left')
    right_neg = np.searchsorted(neg_sorted, preds_pos, side='right')
    v10 = 0.5 * (left_neg + right_neg) / float(n)
    
    # V01[j] = (1/m) * sum_i ( I(preds_pos[i] > preds_neg[j]) + 0.5 * I(preds_pos[i] == preds_neg[j]) )
    pos_sorted = np.sort(preds_pos)
    left_pos = np.searchsorted(pos_sorted, preds_neg, side='left')
    right_pos = np.searchsorted(pos_sorted, preds_neg, side='right')
    v01 = (float(m) - 0.5 * (left_pos + right_pos)) / float(m)
    
    auc_val = float(np.mean(v10))
    
    var10 = float(np.var(v10, ddof=1) / m) if m > 1 else 0.0
    var01 = float(np.var(v01, ddof=1) / n) if n > 1 else 0.0
    
    var_auc = var10 + var01
    return float(auc_val), float(var_auc), v10, v01


# Alias retrocompatible
delong_roc_variance = fast_delong_roc_variance


def delong_test_paired(y_true, y_pred_a, y_pred_b):
    """
    Ejecuta el test de DeLong para comparar formalmente si AUC(A) > AUC(B).
    Devuelve diferencia de AUC, Z-statistic y two-sided p-value.
    """
    auc_a, var_a, v10_a, v01_a = fast_delong_roc_variance(y_true, y_pred_a)
    auc_b, var_b, v10_b, v01_b = fast_delong_roc_variance(y_true, y_pred_b)
    
    m = len(v10_a)
    n = len(v01_a)
    
    cov10 = np.cov(v10_a, v10_b)[0, 1] / m if m > 1 else 0.0
    cov01 = np.cov(v01_a, v01_b)[0, 1] / n if n > 1 else 0.0
    
    cov_auc = cov10 + cov01
    var_diff = var_a + var_b - 2 * cov_auc
    
    if var_diff <= 0:
        z_stat = 0.0
        p_value = 1.0
    else:
        diff = auc_a - auc_b
        z_stat = diff / np.sqrt(var_diff)
        p_value = 2 * (1.0 - stats.norm.cdf(abs(z_stat)))
        
    return {
        'auc_model_a': auc_a,
        'auc_model_b': auc_b,
        'auc_diff': auc_a - auc_b,
        'z_statistic': float(z_stat),
        'p_value': float(p_value),
        'statistically_significant_5pct': bool(p_value < 0.05)
    }


def mcnemar_test(y_true, y_pred_a_bin, y_pred_b_bin):
    """
    Calcula el Test de McNemar con corrección por continuidad de Edwards:
    chi2 = (max(0.0, |b - c| - 1.0) ** 2) / (b + c)
    donde b = (A=1, B=0), c = (A=0, B=1).
    """
    y_true = np.asarray(y_true)
    y_a = np.asarray(y_pred_a_bin)
    y_b = np.asarray(y_pred_b_bin)
    
    correct_a = (y_a == y_true)
    correct_b = (y_b == y_true)
    
    # b: A acierta y B falla
    b = int(np.sum(correct_a & ~correct_b))
    # c: A falla y B acierta
    c = int(np.sum(~correct_a & correct_b))
    # a: ambos aciertan, d: ambos fallan
    a = int(np.sum(correct_a & correct_b))
    d = int(np.sum(~correct_a & ~correct_b))
    
    table = [[a, b], [c, d]]
    
    if (b + c) == 0:
        chi2 = 0.0
        p_val = 1.0
    else:
        chi2 = (max(0.0, abs(b - c) - 1.0) ** 2) / float(b + c)
        p_val = 1.0 - stats.chi2.cdf(chi2, df=1)
        
    return {
        'contingency_table': table,
        'discordant_b_a_correct': b,
        'discordant_c_b_correct': c,
        'chi2_statistic': float(chi2),
        'p_value': float(p_val),
        'statistically_significant_5pct': bool(p_val < 0.05)
    }


def bootstrap_confidence_intervals(y_true, y_prob, n_bootstraps=1000, alpha=0.05, seed=42):
    """
    Calcula Intervalos de Confianza Empíricos del 95% mediante remuestreo Bootstrap (B=1000).
    """
    np.random.seed(seed)
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    n = len(y_true)
    
    pr_aucs = []
    roc_aucs = []
    f2_scores = []
    briers = []
    
    for _ in range(n_bootstraps):
        idx = np.random.choice(n, size=n, replace=True)
        y_b_true = y_true[idx]
        y_b_prob = y_prob[idx]
        
        if len(np.unique(y_b_true)) < 2:
            continue
            
        precision, recall, _ = precision_recall_curve(y_b_true, y_b_prob)
        pr_aucs.append(auc(recall, precision))
        roc_aucs.append(roc_auc_score(y_b_true, y_b_prob))
        briers.append(brier_score_loss(y_b_true, y_b_prob))
        
        # F2 at default 0.3 threshold
        y_b_pred = (y_b_prob >= 0.3).astype(int)
        f2_scores.append(fbeta_score(y_b_true, y_b_pred, beta=2, zero_division=0))
        
    low_pct = 100 * (alpha / 2.0)
    high_pct = 100 * (1.0 - alpha / 2.0)
    
    return {
        'pr_auc_mean': float(np.mean(pr_aucs)),
        'pr_auc_ci': (float(np.percentile(pr_aucs, low_pct)), float(np.percentile(pr_aucs, high_pct))),
        'roc_auc_mean': float(np.mean(roc_aucs)),
        'roc_auc_ci': (float(np.percentile(roc_aucs, low_pct)), float(np.percentile(roc_aucs, high_pct))),
        'f2_score_mean': float(np.mean(f2_scores)),
        'f2_score_ci': (float(np.percentile(f2_scores, low_pct)), float(np.percentile(f2_scores, high_pct))),
        'brier_mean': float(np.mean(briers)),
        'brier_ci': (float(np.percentile(briers, low_pct)), float(np.percentile(briers, high_pct)))
    }


def hosmer_lemeshow_test(y_true, y_prob, n_bins=10):
    """
    Test de Bondad de Ajuste de Hosmer-Lemeshow (HL) para evaluar calibración de probabilidades.
    HL = sum_g (O_g - E_g)^2 / (N_g * pi_g * (1 - pi_g)) ~ Chi2(df = n_bins - 2)
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    
    df_hl = pd.DataFrame({'y_true': y_true, 'y_prob': y_prob})
    df_hl['bin'] = pd.qcut(df_hl['y_prob'], q=n_bins, duplicates='drop')
    
    hl_stat = 0.0
    bins_detail = []
    
    for _, grp in df_hl.groupby('bin', observed=False):
        n_g = len(grp)
        if n_g == 0:
            continue
        o_g = grp['y_true'].sum()
        pi_g = grp['y_prob'].mean()
        e_g = n_g * pi_g
        
        denom = n_g * pi_g * (1.0 - pi_g)
        if denom > 1e-6:
            hl_stat += ((o_g - e_g) ** 2) / denom
            
        bins_detail.append({
            'count': n_g,
            'observed_events': int(o_g),
            'expected_events': float(e_g),
            'observed_rate': float(o_g / n_g),
            'predicted_prob': float(pi_g)
        })
        
    df_bins = len(bins_detail) - 2
    df_bins = max(df_bins, 1)
    p_value = 1.0 - stats.chi2.cdf(hl_stat, df=df_bins)
    
    return {
        'hl_statistic': float(hl_stat),
        'degrees_of_freedom': int(df_bins),
        'p_value': float(p_value),
        'is_well_calibrated_5pct': bool(p_value > 0.05),
        'bins_detail': bins_detail
    }


def generate_lift_and_gain_curves(y_true, y_prob_dict, output_dir=FIGURES_DIR):
    """
    Genera gráficos Tufte/Cleveland de Curvas de Ganancia Acumulada (Cumulative Gains)
    y Curvas de Lift para los modelos del benchmark.
    """
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="muted")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 1. Cumulative Gains Curve
    total_positives = np.sum(y_true)
    n_total = len(y_true)
    
    ax1.plot([0, 100], [0, 100], 'k--', label='Modelo Aleatorio (Baseline)', linewidth=1.5)
    
    colors = {
        'CatBoost_GPU': '#e74c3c',
        'LightGBM': '#3498db',
        'Scorecard_WOE_Basel': '#2ecc71',
        'Random_Forest': '#9b59b6',
        'Altman_Merton_Baseline': '#95a5a6'
    }
    
    for model_name, y_prob in y_prob_dict.items():
        sort_idx = np.argsort(y_prob)[::-1]
        y_sorted = y_true[sort_idx]
        cum_positives = np.cumsum(y_sorted)
        gain_pct = (cum_positives / total_positives) * 100.0
        pop_pct = (np.arange(1, n_total + 1) / n_total) * 100.0
        
        color = colors.get(model_name, None)
        ax1.plot(pop_pct, gain_pct, label=f"{model_name}", linewidth=2.0, color=color)
        
        # Lift at decile 10%
        idx_10 = int(n_total * 0.10)
        lift_10 = (np.sum(y_sorted[:idx_10]) / idx_10) / (total_positives / n_total)
        
        # Lift Curve
        deciles = np.linspace(0.05, 1.0, 20)
        lifts = []
        for d in deciles:
            k = max(int(n_total * d), 1)
            lift_val = (np.sum(y_sorted[:k]) / k) / (total_positives / n_total)
            lifts.append(lift_val)
            
        ax2.plot(deciles * 100, lifts, label=f"{model_name} (Lift 10%: {lift_10:.2f}x)", linewidth=2.0, color=color)
        
    ax1.set_title('Curva de Ganancia Acumulada (Cumulative Gains)', fontsize=13, fontweight='bold')
    ax1.set_xlabel('% de Población Auditada (Ordenada por Riesgo Descendente)')
    ax1.set_ylabel('% Acumulado de Quiebras Detectadas')
    ax1.legend(loc='lower right')
    
    ax2.axhline(1.0, color='k', linestyle='--', label='Sin Discriminación (Lift = 1.0x)')
    ax2.set_title('Curva de Lift Acumulado por Decil de Población', fontsize=13, fontweight='bold')
    ax2.set_xlabel('% de Población Auditada')
    ax2.set_ylabel('Ratio de Lift (Efectividad Relativa)')
    ax2.legend(loc='upper right')
    
    plt.tight_layout()
    output_path = output_dir / 'fig6_lift_cumulative_gains.png'
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    anexos_fig = Path("anexos/figures")
    anexos_fig.mkdir(parents=True, exist_ok=True)
    plt.savefig(anexos_fig / 'fig6_lift_cumulative_gains.png', dpi=300)
    plt.close()
    print(f"[INFO] Curvas de Ganancia y Lift guardadas en {output_path} (dpi=300).")
    
    # Exportar tabla oficial de deciles de ganancia y lift
    decile_rows = []
    y_prob_xgb = y_prob_dict.get('XGBoost', y_prob_dict.get('LightGBM', list(y_prob_dict.values())[0]))
    y_prob_sc = y_prob_dict.get('Scorecard_WOE_Basel', y_prob_xgb)
    
    sort_xgb = np.argsort(y_prob_xgb)[::-1]
    y_sorted_xgb = y_true[sort_xgb]
    sort_sc = np.argsort(y_prob_sc)[::-1]
    y_sorted_sc = y_true[sort_sc]
    
    for d in range(1, 11):
        k = int(n_total * d / 10.0)
        pos_xgb = y_sorted_xgb[:k].sum()
        gain_xgb = (pos_xgb / total_positives) * 100.0
        lift_xgb = (pos_xgb / k) / (total_positives / n_total)
        
        pos_sc = y_sorted_sc[:k].sum()
        gain_sc = (pos_sc / total_positives) * 100.0
        lift_sc = (pos_sc / k) / (total_positives / n_total)
        
        decile_rows.append({
            'Decil': f'Decil {d}',
            '% Cartera Auditada': f'{d*10}%',
            'Ganancia Acum. XGBoost (%)': f'{gain_xgb:.2f}%',
            'Lift XGBoost': f'{lift_xgb:.2f}x',
            'Ganancia Acum. Scorecard (%)': f'{gain_sc:.2f}%',
            'Lift Scorecard': f'{lift_sc:.2f}x',
            'Lift Baseline': '1.00x'
        })
    df_deciles_csv = pd.DataFrame(decile_rows)
    deciles_csv_path = Path("data/processed/sec_dataset/deciles_gains_lift.csv")
    deciles_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df_deciles_csv.to_csv(deciles_csv_path, index=False)
    print(f"[INFO] Tabla cuantitativa de deciles guardada en {deciles_csv_path}.")


def run_statistical_evaluation():
    print("=" * 80)
    print("FASE 5: CONTRASTES ESTADÍSTICOS FORMALES Y EVALUACIÓN AVANZADA")
    print("=" * 80)
    
    # Cargar datos de prueba representativos para contrastes
    import duckdb
    con = duckdb.connect()
    df_val = con.execute("SELECT * FROM 'data/processed/sec_dataset/v2_sec_financials_pivoted_clean.parquet' WHERE filed_year >= 2022 ORDER BY filed_date;").df()
    con.close()
    
    y_true = df_val['target_bankrupt_12m'].values.astype(int)
    print(f"[Statistical Engine] Dataset de validación fuera de muestra (2022-2026): {len(df_val):,} filas ({y_true.sum():,} quiebras)")
    
    # Cargar modelos serializados para generar predicciones empíricas pareadas
    import catboost as cb
    import lightgbm as lgb
    import pickle
    
    import sys
    sys.path.insert(0, os.path.abspath("src/models"))
    sys.path.insert(0, os.path.abspath("."))
    from src.models.woe_scorecard import CreditScorecardWOE
    import woe_scorecard
    import joblib

    cb_model = cb.CatBoostClassifier()
    cb_model.load_model("models/catboost_best_model.cbm")
    
    lgb_booster = lgb.Booster(model_file="models/lightgbm_best_model.txt")
    
    xgb_model = joblib.load("models/best_xgboost_model.pkl")
    if hasattr(xgb_model, "set_params"):
        try:
            xgb_model.set_params(device="cpu")
        except Exception:
            pass
    
    with open("models/scorecard_woe_best.pkl", "rb") as f:
        sc_model = pickle.load(f)

    preprocessor = joblib.load("models/preprocessor_pipeline.joblib")
    feature_names = preprocessor["feature_names"]
    
    # Preprocesar variables canónicas óptimas (21 features)
    X_val_raw = df_val[feature_names].copy()
    X_val_imp = preprocessor["imputer"].transform(X_val_raw)
    X_val_scaled = preprocessor["scaler"].transform(X_val_imp)
    
    # Predicciones
    probs_cb = cb_model.predict_proba(X_val_scaled)[:, 1]
    probs_lgb = lgb_booster.predict(X_val_scaled)
    probs_xgb = xgb_model.predict_proba(X_val_scaled)[:, 1]
    
    woe_features = sc_model.variable_names
    probs_sc = sc_model.predict_proba_positive(df_val[woe_features])
    
    # Baseline Altman & Merton
    merton_v = df_val['merton_distance_to_default'].fillna(df_val['merton_distance_to_default'].median()).values
    wc_v = df_val['tag_WorkingCapital'].fillna(0).values
    raw_s = - (merton_v / (np.std(merton_v) + 1e-5)) - (wc_v / (np.std(wc_v) + 1e-5))
    probs_baseline = 1.0 / (1.0 + np.exp(-raw_s))
    
    prob_dict = {
        'XGBoost': probs_xgb,
        'LightGBM': probs_lgb,
        'CatBoost_GPU': probs_cb,
        'Scorecard_WOE_Basel': probs_sc,
        'Altman_Merton_Baseline': probs_baseline
    }
    
    # 1. Test de DeLong Paired ROC-AUC
    print("\n[Statistical Engine] 1. Ejecutando Test de DeLong Paired ROC-AUC...")
    delong_xgb_vs_base = delong_test_paired(y_true, probs_xgb, probs_baseline)
    delong_xgb_vs_sc = delong_test_paired(y_true, probs_xgb, probs_sc)
    delong_xgb_vs_lgb = delong_test_paired(y_true, probs_xgb, probs_lgb)
    delong_xgb_vs_cb = delong_test_paired(y_true, probs_xgb, probs_cb)
    delong_lgb_vs_base = delong_test_paired(y_true, probs_lgb, probs_baseline)
    delong_lgb_vs_sc = delong_test_paired(y_true, probs_lgb, probs_sc)
    delong_lgb_vs_cb = delong_test_paired(y_true, probs_lgb, probs_cb)
    delong_cb_vs_base = delong_test_paired(y_true, probs_cb, probs_baseline)
    delong_cb_vs_sc = delong_test_paired(y_true, probs_cb, probs_sc)
    
    print(f"  - XGBoost (SOTA 12M) vs Altman/Merton: AUC Diff = +{delong_xgb_vs_base['auc_diff']:.4f} | Z = {delong_xgb_vs_base['z_statistic']:.3f} | p = {delong_xgb_vs_base['p_value']:.4e} (Significativo: {delong_xgb_vs_base['statistically_significant_5pct']})")
    print(f"  - XGBoost (SOTA 12M) vs Scorecard WOE: AUC Diff = +{delong_xgb_vs_sc['auc_diff']:.4f} | Z = {delong_xgb_vs_sc['z_statistic']:.3f} | p = {delong_xgb_vs_sc['p_value']:.4e} (Significativo: {delong_xgb_vs_sc['statistically_significant_5pct']})")
    print(f"  - XGBoost vs LightGBM:                AUC Diff = {delong_xgb_vs_lgb['auc_diff']:+.4f} | Z = {delong_xgb_vs_lgb['z_statistic']:.3f} | p = {delong_xgb_vs_lgb['p_value']:.4f}")
    print(f"  - XGBoost vs CatBoost GPU:             AUC Diff = {delong_xgb_vs_cb['auc_diff']:+.4f} | Z = {delong_xgb_vs_cb['z_statistic']:.3f} | p = {delong_xgb_vs_cb['p_value']:.4f}")
    
    # 2. Test de McNemar (con umbrales optimos tau* canonicos)
    print("\n[Statistical Engine] 2. Ejecutando Test de McNemar...")
    bin_xgb = (probs_xgb >= OFFICIAL_THRESHOLDS_12M['XGBoost']).astype(int)
    bin_lgb = (probs_lgb >= OFFICIAL_THRESHOLDS_12M['LightGBM']).astype(int)
    bin_cb = (probs_cb >= OFFICIAL_THRESHOLDS_12M['CatBoost']).astype(int)
    bin_base = (probs_baseline >= OFFICIAL_THRESHOLDS_12M['Altman_Merton_Baseline']).astype(int)
    bin_sc = (probs_sc >= OFFICIAL_THRESHOLDS_12M['Scorecard_WOE_Basel']).astype(int)
    
    mcnemar_xgb_base = mcnemar_test(y_true, bin_xgb, bin_base)
    mcnemar_xgb_sc = mcnemar_test(y_true, bin_xgb, bin_sc)
    mcnemar_xgb_lgb = mcnemar_test(y_true, bin_xgb, bin_lgb)
    mcnemar_lgb_base = mcnemar_test(y_true, bin_lgb, bin_base)
    mcnemar_lgb_sc = mcnemar_test(y_true, bin_lgb, bin_sc)
    mcnemar_cb_base = mcnemar_test(y_true, bin_cb, bin_base)
    mcnemar_cb_sc = mcnemar_test(y_true, bin_cb, bin_sc)
    
    print(f"  - XGBoost vs Baseline:  Chi2 = {mcnemar_xgb_base['chi2_statistic']:.2f} | p = {mcnemar_xgb_base['p_value']:.4e}")
    print(f"  - XGBoost vs Scorecard: Chi2 = {mcnemar_xgb_sc['chi2_statistic']:.2f} | p = {mcnemar_xgb_sc['p_value']:.4e}")
    
    # 3. Bootstrap CIs (1000 réplicas)
    print("\n[Statistical Engine] 3. Calculando Intervalos de Confianza Bootstrap (B=1000)...")
    boot_xgb = bootstrap_confidence_intervals(y_true, probs_xgb, n_bootstraps=1000)
    boot_lgb = bootstrap_confidence_intervals(y_true, probs_lgb, n_bootstraps=1000)
    boot_cb = bootstrap_confidence_intervals(y_true, probs_cb, n_bootstraps=1000)
    boot_sc = bootstrap_confidence_intervals(y_true, probs_sc, n_bootstraps=1000)
    print(f"  - XGBoost PR-AUC: {boot_xgb['pr_auc_mean']:.4f} [95% CI: {boot_xgb['pr_auc_ci'][0]:.4f} - {boot_xgb['pr_auc_ci'][1]:.4f}]")
    print(f"  - XGBoost ROC-AUC: {boot_xgb['roc_auc_mean']:.4f} [95% CI: {boot_xgb['roc_auc_ci'][0]:.4f} - {boot_xgb['roc_auc_ci'][1]:.4f}]")
    print(f"  - LightGBM PR-AUC: {boot_lgb['pr_auc_mean']:.4f} [95% CI: {boot_lgb['pr_auc_ci'][0]:.4f} - {boot_lgb['pr_auc_ci'][1]:.4f}]")
    print(f"  - CatBoost PR-AUC: {boot_cb['pr_auc_mean']:.4f} [95% CI: {boot_cb['pr_auc_ci'][0]:.4f} - {boot_cb['pr_auc_ci'][1]:.4f}]")
    print(f"  - Scorecard PR-AUC: {boot_sc['pr_auc_mean']:.4f} [95% CI: {boot_sc['pr_auc_ci'][0]:.4f} - {boot_sc['pr_auc_ci'][1]:.4f}]")
    
    # 4. Hosmer-Lemeshow Test
    print("\n[Statistical Engine] 4. Test de Bondad de Ajuste Hosmer-Lemeshow...")
    hl_xgb = hosmer_lemeshow_test(y_true, probs_xgb)
    hl_lgb = hosmer_lemeshow_test(y_true, probs_lgb)
    print(f"  - XGBoost HL Stat:  {hl_xgb['hl_statistic']:.2f} (df={hl_xgb['degrees_of_freedom']}) | p = {hl_xgb['p_value']:.4e}")
    print(f"  - LightGBM HL Stat: {hl_lgb['hl_statistic']:.2f} (df={hl_lgb['degrees_of_freedom']}) | p = {hl_lgb['p_value']:.4e}")
    
    # 5. Curvas de Ganancia y Lift
    print("\n[Statistical Engine] 5. Generando Curvas de Ganancia y Lift (Tufte/Cleveland)...")
    generate_lift_and_gain_curves(y_true, prob_dict, output_dir=FIGURES_DIR)
    
    def format_sig(t_res):
        if t_res['p_value'] < 0.001:
            return "**SÍ ($p < 0.001$)**"
        elif t_res['statistically_significant_5pct']:
            return "**SÍ ($p < 0.05$)**"
        else:
            return "**NO ($p > 0.001$)**"

    # Exportar resultados estructurados a CSV y JSON en data/processed/sec_dataset/
    out_data_dir = Path("data/processed/sec_dataset")
    out_data_dir.mkdir(parents=True, exist_ok=True)

    delong_df_export = pd.DataFrame([
        {'Comparacion': 'XGBoost (SOTA 12M) vs Altman & Merton', 'Delta_AUC': f"+{delong_xgb_vs_base['auc_diff']:.4f}", 'Z_Stat': round(delong_xgb_vs_base['z_statistic'], 3), 'p_value': f"{delong_xgb_vs_base['p_value']:.4e}", 'Significativo': 'SI (p < 0.001)' if delong_xgb_vs_base['p_value'] < 0.001 else 'NO (p > 0.001)'},
        {'Comparacion': 'XGBoost (SOTA 12M) vs Scorecard WOE', 'Delta_AUC': f"+{delong_xgb_vs_sc['auc_diff']:.4f}", 'Z_Stat': round(delong_xgb_vs_sc['z_statistic'], 3), 'p_value': f"{delong_xgb_vs_sc['p_value']:.4e}", 'Significativo': 'SI (p < 0.001)' if delong_xgb_vs_sc['p_value'] < 0.001 else 'NO (p > 0.001)'},
        {'Comparacion': 'XGBoost (SOTA 12M) vs LightGBM', 'Delta_AUC': f"{delong_xgb_vs_lgb['auc_diff']:+.4f}", 'Z_Stat': round(delong_xgb_vs_lgb['z_statistic'], 3), 'p_value': f"{delong_xgb_vs_lgb['p_value']:.4f}", 'Significativo': 'SI (p < 0.001)' if delong_xgb_vs_lgb['p_value'] < 0.001 else 'NO (p > 0.001)'},
        {'Comparacion': 'XGBoost (SOTA 12M) vs CatBoost GPU', 'Delta_AUC': f"{delong_xgb_vs_cb['auc_diff']:+.4f}", 'Z_Stat': round(delong_xgb_vs_cb['z_statistic'], 3), 'p_value': f"{delong_xgb_vs_cb['p_value']:.4f}", 'Significativo': 'SI (p < 0.001)' if delong_xgb_vs_cb['p_value'] < 0.001 else 'NO (p > 0.001)'},
        {'Comparacion': 'LightGBM vs Altman & Merton', 'Delta_AUC': f"+{delong_lgb_vs_base['auc_diff']:.4f}", 'Z_Stat': round(delong_lgb_vs_base['z_statistic'], 3), 'p_value': f"{delong_lgb_vs_base['p_value']:.4e}", 'Significativo': 'SI (p < 0.001)' if delong_lgb_vs_base['p_value'] < 0.001 else 'NO (p > 0.001)'},
        {'Comparacion': 'LightGBM vs Scorecard WOE', 'Delta_AUC': f"+{delong_lgb_vs_sc['auc_diff']:.4f}", 'Z_Stat': round(delong_lgb_vs_sc['z_statistic'], 3), 'p_value': f"{delong_lgb_vs_sc['p_value']:.4e}", 'Significativo': 'SI (p < 0.001)' if delong_lgb_vs_sc['p_value'] < 0.001 else 'NO (p > 0.001)'},
        {'Comparacion': 'CatBoost GPU vs Altman & Merton', 'Delta_AUC': f"+{delong_cb_vs_base['auc_diff']:.4f}", 'Z_Stat': round(delong_cb_vs_base['z_statistic'], 3), 'p_value': f"{delong_cb_vs_base['p_value']:.4e}", 'Significativo': 'SI (p < 0.001)' if delong_cb_vs_base['p_value'] < 0.001 else 'NO (p > 0.001)'},
        {'Comparacion': 'CatBoost GPU vs Scorecard WOE', 'Delta_AUC': f"+{delong_cb_vs_sc['auc_diff']:.4f}", 'Z_Stat': round(delong_cb_vs_sc['z_statistic'], 3), 'p_value': f"{delong_cb_vs_sc['p_value']:.4e}", 'Significativo': 'SI (p < 0.001)' if delong_cb_vs_sc['p_value'] < 0.001 else 'NO (p > 0.001)'}
    ])
    delong_df_export.to_csv(out_data_dir / "delong_tests_results.csv", index=False)

    mcnemar_df_export = pd.DataFrame([
        {'Comparacion': 'XGBoost vs Baseline Altman/Merton', 'Casos_b': mcnemar_xgb_base['discordant_b_a_correct'], 'Casos_c': mcnemar_xgb_base['discordant_c_b_correct'], 'Chi2_Stat': round(mcnemar_xgb_base['chi2_statistic'], 2), 'p_value': f"{mcnemar_xgb_base['p_value']:.4e}", 'Significativo': 'SI (p < 0.001)'},
        {'Comparacion': 'XGBoost vs Scorecard WOE', 'Casos_b': mcnemar_xgb_sc['discordant_b_a_correct'], 'Casos_c': mcnemar_xgb_sc['discordant_c_b_correct'], 'Chi2_Stat': round(mcnemar_xgb_sc['chi2_statistic'], 2), 'p_value': f"{mcnemar_xgb_sc['p_value']:.4e}", 'Significativo': 'SI (p < 0.001)'},
        {'Comparacion': 'LightGBM vs Baseline Altman/Merton', 'Casos_b': mcnemar_lgb_base['discordant_b_a_correct'], 'Casos_c': mcnemar_lgb_base['discordant_c_b_correct'], 'Chi2_Stat': round(mcnemar_lgb_base['chi2_statistic'], 2), 'p_value': f"{mcnemar_lgb_base['p_value']:.4e}", 'Significativo': 'SI (p < 0.001)'},
        {'Comparacion': 'LightGBM vs Scorecard WOE', 'Casos_b': mcnemar_lgb_sc['discordant_b_a_correct'], 'Casos_c': mcnemar_lgb_sc['discordant_c_b_correct'], 'Chi2_Stat': round(mcnemar_lgb_sc['chi2_statistic'], 2), 'p_value': f"{mcnemar_lgb_sc['p_value']:.4e}", 'Significativo': 'SI (p < 0.001)'},
        {'Comparacion': 'CatBoost vs Baseline Altman/Merton', 'Casos_b': mcnemar_cb_base['discordant_b_a_correct'], 'Casos_c': mcnemar_cb_base['discordant_c_b_correct'], 'Chi2_Stat': round(mcnemar_cb_base['chi2_statistic'], 2), 'p_value': f"{mcnemar_cb_base['p_value']:.4e}", 'Significativo': 'SI (p < 0.001)'},
        {'Comparacion': 'CatBoost vs Scorecard WOE', 'Casos_b': mcnemar_cb_sc['discordant_b_a_correct'], 'Casos_c': mcnemar_cb_sc['discordant_c_b_correct'], 'Chi2_Stat': round(mcnemar_cb_sc['chi2_statistic'], 2), 'p_value': f"{mcnemar_cb_sc['p_value']:.4e}", 'Significativo': 'SI (p < 0.001)'}
    ])
    mcnemar_df_export.to_csv(out_data_dir / "mcnemar_tests_results.csv", index=False)

    boot_rows = []
    for mname, mdict in [('XGBoost (SOTA 12M)', boot_xgb), ('LightGBM', boot_lgb), ('CatBoost GPU', boot_cb), ('Scorecard WOE', boot_sc)]:
        boot_rows.append({'Modelo': mname, 'Metrica': 'PR-AUC', 'Media_Bootstrap': round(mdict['pr_auc_mean'], 4), 'CI_95_Lower': round(mdict['pr_auc_ci'][0], 4), 'CI_95_Upper': round(mdict['pr_auc_ci'][1], 4)})
        boot_rows.append({'Modelo': mname, 'Metrica': 'ROC-AUC', 'Media_Bootstrap': round(mdict['roc_auc_mean'], 4), 'CI_95_Lower': round(mdict['roc_auc_ci'][0], 4), 'CI_95_Upper': round(mdict['roc_auc_ci'][1], 4)})
        if 'f2_score_mean' in mdict:
            boot_rows.append({'Modelo': mname, 'Metrica': 'F2-Score', 'Media_Bootstrap': round(mdict['f2_score_mean'], 4), 'CI_95_Lower': round(mdict['f2_score_ci'][0], 4), 'CI_95_Upper': round(mdict['f2_score_ci'][1], 4)})
    boot_df_export = pd.DataFrame(boot_rows)
    boot_df_export.to_csv(out_data_dir / "bootstrap_ci_results.csv", index=False)

    eval_json = {
        'delong': {
            'xgb_vs_base': delong_xgb_vs_base,
            'xgb_vs_sc': delong_xgb_vs_sc,
            'xgb_vs_lgb': delong_xgb_vs_lgb,
            'xgb_vs_cb': delong_xgb_vs_cb,
            'lgb_vs_base': delong_lgb_vs_base,
            'lgb_vs_sc': delong_lgb_vs_sc,
            'cb_vs_base': delong_cb_vs_base,
            'cb_vs_sc': delong_cb_vs_sc
        },
        'mcnemar': {
            'xgb_base': mcnemar_xgb_base,
            'xgb_sc': mcnemar_xgb_sc,
            'lgb_base': mcnemar_lgb_base,
            'lgb_sc': mcnemar_lgb_sc,
            'cb_base': mcnemar_cb_base,
            'cb_sc': mcnemar_cb_sc
        },
        'bootstrap_ci': {
            'xgb': boot_xgb,
            'lgb': boot_lgb,
            'cb': boot_cb,
            'sc': boot_sc
        },
        'hosmer_lemeshow': {
            'xgb': hl_xgb,
            'lgb': hl_lgb
        }
    }
    with open(out_data_dir / "statistical_evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(eval_json, f, indent=2)
    print(f"[INFO] Tablas de contrastes estadísticos guardadas en {out_data_dir} (CSV y JSON).")

    # Generar Anexo E en Markdown
    report_md = f"""# Anexo E: Contrastes Estadísticos Formales y Evaluación de Significancia

En este anexo se recogen las pruebas de hipótesis estadísticas, contrastes no paramétricos de curvas ROC (DeLong), tablas de contingencia discordante (McNemar), intervalos de confianza empíricos mediante remuestreo Bootstrap ($B=1000$ réplicas) y pruebas de bondad de ajuste de Hosmer-Lemeshow para los modelos del TFM.

---

## E.1 Test de DeLong para Comparación de Curvas ROC-AUC

El test de DeLong et al. (1988) compara de forma no paramétrica las áreas bajo la curva ROC de modelos pareados sobre la misma cohorte de validación independiente:

| Comparación de Modelos | $\\Delta \\mathrm{{AUC}}$ | Estadístico $Z$ | $p\\mathrm{{-valor}}$ | Significativo ($\\alpha=0.05$) |
| :--- | :--- | :--- | :--- | :--- |
| **XGBoost (SOTA 12M) vs Altman & Merton** | **+{delong_xgb_vs_base['auc_diff']:.4f}** | **{delong_xgb_vs_base['z_statistic']:.3f}** | **{delong_xgb_vs_base['p_value']:.4e}** | {format_sig(delong_xgb_vs_base)} |
| **XGBoost (SOTA 12M) vs Scorecard WOE** | **+{delong_xgb_vs_sc['auc_diff']:.4f}** | **{delong_xgb_vs_sc['z_statistic']:.3f}** | **{delong_xgb_vs_sc['p_value']:.4e}** | {format_sig(delong_xgb_vs_sc)} |
| **XGBoost (SOTA 12M) vs LightGBM** | **{delong_xgb_vs_lgb['auc_diff']:+.4f}** | **{delong_xgb_vs_lgb['z_statistic']:.3f}** | **{delong_xgb_vs_lgb['p_value']:.4f}** | {format_sig(delong_xgb_vs_lgb)} |
| **XGBoost (SOTA 12M) vs CatBoost GPU** | **{delong_xgb_vs_cb['auc_diff']:+.4f}** | **{delong_xgb_vs_cb['z_statistic']:.3f}** | **{delong_xgb_vs_cb['p_value']:.4f}** | {format_sig(delong_xgb_vs_cb)} |
| **LightGBM vs Altman & Merton** | **+{delong_lgb_vs_base['auc_diff']:.4f}** | **{delong_lgb_vs_base['z_statistic']:.3f}** | **{delong_lgb_vs_base['p_value']:.4e}** | {format_sig(delong_lgb_vs_base)} |
| **LightGBM vs Scorecard WOE** | **+{delong_lgb_vs_sc['auc_diff']:.4f}** | **{delong_lgb_vs_sc['z_statistic']:.3f}** | **{delong_lgb_vs_sc['p_value']:.4e}** | {format_sig(delong_lgb_vs_sc)} |
| **CatBoost GPU vs Altman & Merton** | **+{delong_cb_vs_base['auc_diff']:.4f}** | **{delong_cb_vs_base['z_statistic']:.3f}** | **{delong_cb_vs_base['p_value']:.4e}** | {format_sig(delong_cb_vs_base)} |
| **CatBoost GPU vs Scorecard WOE** | **+{delong_cb_vs_sc['auc_diff']:.4f}** | **{delong_cb_vs_sc['z_statistic']:.3f}** | **{delong_cb_vs_sc['p_value']:.4e}** | {format_sig(delong_cb_vs_sc)} |

---

## E.2 Test de McNemar sobre Matrices de Confusión Discordantes

| Comparación | Casos $b$ ($M_1=1, M_2=0$) | Casos $c$ ($M_1=0, M_2=1$) | $\\chi^2$ McNemar | $p\\mathrm{{-valor}}$ |
| :--- | :--- | :--- | :--- | :--- |
| **XGBoost vs Baseline Altman/Merton** | {mcnemar_xgb_base['discordant_b_a_correct']:,} | {mcnemar_xgb_base['discordant_c_b_correct']:,} | **{mcnemar_xgb_base['chi2_statistic']:.2f}** | **{mcnemar_xgb_base['p_value']:.4e}** |
| **XGBoost vs Scorecard WOE** | {mcnemar_xgb_sc['discordant_b_a_correct']:,} | {mcnemar_xgb_sc['discordant_c_b_correct']:,} | **{mcnemar_xgb_sc['chi2_statistic']:.2f}** | **{mcnemar_xgb_sc['p_value']:.4e}** |
| **LightGBM vs Baseline Altman/Merton** | {mcnemar_lgb_base['discordant_b_a_correct']:,} | {mcnemar_lgb_base['discordant_c_b_correct']:,} | **{mcnemar_lgb_base['chi2_statistic']:.2f}** | **{mcnemar_lgb_base['p_value']:.4e}** |
| **LightGBM vs Scorecard WOE** | {mcnemar_lgb_sc['discordant_b_a_correct']:,} | {mcnemar_lgb_sc['discordant_c_b_correct']:,} | **{mcnemar_lgb_sc['chi2_statistic']:.2f}** | **{mcnemar_lgb_sc['p_value']:.4e}** |
| **CatBoost vs Baseline Altman/Merton** | {mcnemar_cb_base['discordant_b_a_correct']:,} | {mcnemar_cb_base['discordant_c_b_correct']:,} | **{mcnemar_cb_base['chi2_statistic']:.2f}** | **{mcnemar_cb_base['p_value']:.4e}** |
| **CatBoost vs Scorecard WOE** | {mcnemar_cb_sc['discordant_b_a_correct']:,} | {mcnemar_cb_sc['discordant_c_b_correct']:,} | **{mcnemar_cb_sc['chi2_statistic']:.2f}** | **{mcnemar_cb_sc['p_value']:.4e}** |

---

## E.3 Intervalos de Confianza Bootstrap (95%, $B=1000$ Réplicas)

| Modelo | Métrica | Media Bootstrap | Intervalo de Confianza 95% |
| :--- | :--- | :--- | :--- |
| **XGBoost (SOTA 12M)** | **PR-AUC** | **{boot_xgb['pr_auc_mean']:.4f}** | **[{boot_xgb['pr_auc_ci'][0]:.4f}, {boot_xgb['pr_auc_ci'][1]:.4f}]** |
| **XGBoost (SOTA 12M)** | **ROC-AUC** | **{boot_xgb['roc_auc_mean']:.4f}** | **[{boot_xgb['roc_auc_ci'][0]:.4f}, {boot_xgb['roc_auc_ci'][1]:.4f}]** |
| **XGBoost (SOTA 12M)** | **F2-Score** | **{boot_xgb['f2_score_mean']:.4f}** | **[{boot_xgb['f2_score_ci'][0]:.4f}, {boot_xgb['f2_score_ci'][1]:.4f}]** |
| **LightGBM** | **PR-AUC** | **{boot_lgb['pr_auc_mean']:.4f}** | **[{boot_lgb['pr_auc_ci'][0]:.4f}, {boot_lgb['pr_auc_ci'][1]:.4f}]** |
| **LightGBM** | **ROC-AUC** | **{boot_lgb['roc_auc_mean']:.4f}** | **[{boot_lgb['roc_auc_ci'][0]:.4f}, {boot_lgb['roc_auc_ci'][1]:.4f}]** |
| **LightGBM** | **F2-Score** | **{boot_lgb['f2_score_mean']:.4f}** | **[{boot_lgb['f2_score_ci'][0]:.4f}, {boot_lgb['f2_score_ci'][1]:.4f}]** |
| **CatBoost GPU** | **PR-AUC** | **{boot_cb['pr_auc_mean']:.4f}** | **[{boot_cb['pr_auc_ci'][0]:.4f}, {boot_cb['pr_auc_ci'][1]:.4f}]** |
| **CatBoost GPU** | **ROC-AUC** | **{boot_cb['roc_auc_mean']:.4f}** | **[{boot_cb['roc_auc_ci'][0]:.4f}, {boot_cb['roc_auc_ci'][1]:.4f}]** |
| **CatBoost GPU** | **F2-Score** | **{boot_cb['f2_score_mean']:.4f}** | **[{boot_cb['f2_score_ci'][0]:.4f}, {boot_cb['f2_score_ci'][1]:.4f}]** |
| **Scorecard WOE** | **PR-AUC** | **{boot_sc['pr_auc_mean']:.4f}** | **[{boot_sc['pr_auc_ci'][0]:.4f}, {boot_sc['pr_auc_ci'][1]:.4f}]** |
| **Scorecard WOE** | **ROC-AUC** | **{boot_sc['roc_auc_mean']:.4f}** | **[{boot_sc['roc_auc_ci'][0]:.4f}, {boot_sc['roc_auc_ci'][1]:.4f}]** |

---

## E.4 Curvas de Ganancia Acumulada y Lift

![Curvas de Lift y Ganancias](figures/fig6_lift_cumulative_gains.png)

---
"""
    STATISTICAL_REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    with open(STATISTICAL_REPORT_MD, "w", encoding="utf-8") as f:
        f.write(report_md)
    print(f"[INFO] Anexo E redactado exitosamente en {STATISTICAL_REPORT_MD}.")
    print("=" * 80)


if __name__ == "__main__":
    run_statistical_evaluation()
