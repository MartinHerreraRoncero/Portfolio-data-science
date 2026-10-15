"""
Módulo de Selección de Características mediante Sondaje Aleatorio (Random Probes / Boruta-Style Feature Selection).
Proyecto TFM: Predicción de Insolvencia en Empresas del S&P 500.

Estrategia:
1. Inyecta 2 variables aleatorias de control (random_noise_uniform ~ U(0,1), random_noise_normal ~ N(0,1)).
2. Entrena CatBoost GPU sobre la matriz máster V2.0 (N = 404,853).
3. Calcula el Feature Importance exacto de CatBoost.
4. Identifica el umbral aleatorio: max(Importance(random_noise_uniform), Importance(random_noise_normal)).
5. Filtra y elimina todas las variables reales con importancia <= umbral aleatorio.
6. Exporta la matriz limpia v2_sec_financials_pivoted_clean.parquet.
"""

import os
import sys
import json
import duckdb
import pandas as pd
import numpy as np
from pathlib import Path
from catboost import CatBoostClassifier

# Fix path for direct execution
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

PARQUET_PATH_V21 = Path("data/processed/sec_dataset/v2.1_master_financials_dataset.parquet")
PARQUET_PATH_V2 = Path("data/processed/sec_dataset/v2_sec_financials_pivoted.parquet")
CLEAN_PARQUET_PATH = Path("data/processed/sec_dataset/v2_sec_financials_pivoted_clean.parquet")
REPORT_JSON = Path("data/processed/sec_dataset/feature_selection_report.json")


def run_random_probes_feature_selection():
    print("=" * 90)
    print("SELECCIÓN DE CARACTERÍSTICAS POR SONDAJE ALEATORIO (RANDOM PROBES) EN CATBOOST GPU")
    print("=" * 90)

    # 1. Cargar Dataset
    target_path = PARQUET_PATH_V21 if PARQUET_PATH_V21.exists() else PARQUET_PATH_V2
    print(f"[Feature Selector] Cargando dataset máster desde {target_path}...")
    con = duckdb.connect()
    df = con.execute(f"SELECT * FROM '{target_path}';").df()
    con.close()


    print(f"  - Observaciones totales: {len(df):,}")

    if "news_sentiment_avg" not in df.columns:
        df["news_sentiment_avg"] = df["nlp_sentiment_decayed_30d"]

    # Exclusiones heurísticas de Fase 1 (56 variables descartadas de las 108 iniciales)
    meta_cols = ['adsh', 'cik', 'company_name', 'ticker', 'sic', 'form_type', 'period_date', 'filed_date', 'fy', 'filed_year']
    target_cols = [c for c in df.columns if c.startswith('target_') or c.startswith('is_distress_event')]
    gmm_dummies = ['macro_regime_label', 'macro_regime_name', 'corp_archetype_label', 'corp_archetype_name']
    gmm_collinear = ['macro_regime_neutral_prob', 'corp_archetype_prob_3']
    network_cols = ['network_pagerank', 'network_betweenness', 'network_degree', 'network_community_id', 'cluster_distress_infection_rate', 'is_network_imputed']
    nlp_90d = ['news_count_90d', 'nlp_sentiment_decayed_90d', 'nlp_distress_topic_intensity_decayed_90d', 'distress_news_count_90d']
    join_dups = [c for c in df.columns if c.endswith('_1') and not c.startswith('corp_archetype_prob_') and not c.startswith('is_distress_event_')]

    all_dropped = set(meta_cols + target_cols + gmm_dummies + gmm_collinear + network_cols + nlp_90d + join_dups)
    real_feature_cols = [c for c in df.columns if c not in all_dropped]
    print(f"  - Variables candidatas de Fase 1 (Screening Heurístico): {len(real_feature_cols)}")


    # 2. Inyectar 2 variables de ruido aleatorio (Random Probes)
    print("\n[Feature Selector] Inyectando 2 variables de ruido aleatorio de control...")
    np.random.seed(42)
    df['random_noise_uniform'] = np.random.uniform(0.0, 1.0, size=len(df)).astype(np.float32)
    df['random_noise_normal'] = np.random.normal(0.0, 1.0, size=len(df)).astype(np.float32)

    all_feature_cols = real_feature_cols + ['random_noise_uniform', 'random_noise_normal']

    X = df[all_feature_cols].copy()
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    for col in num_cols:
        col_med = X[col].median()
        X[col] = X[col].fillna(col_med if pd.notna(col_med) else 0.0)

    # Label encode / convert categorical columns to category for CatBoost
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    for col in cat_cols:
        X[col] = X[col].astype(str).fillna("unknown")

    y = df['target_bankrupt_12m'].values

    # 3. Entrenar CatBoost GPU para calcular Feature Importance
    print("\n[Feature Selector] Entrenando CatBoost GPU para evaluar Feature Importance...")
    model = CatBoostClassifier(
        iterations=300,
        depth=6,
        learning_rate=0.05,
        cat_features=cat_cols if cat_cols else None,
        task_type='GPU',
        verbose=50,
        random_seed=42
    )
    model.fit(X, y)

    importances = model.get_feature_importance()
    feature_imp_df = pd.DataFrame({
        'feature': X.columns,
        'importance': importances
    }).sort_values(by='importance', ascending=False).reset_index(drop=True)

    # 4. Determinar Umbral de Ruido Aleatorio
    noise_imp_uniform = feature_imp_df.loc[feature_imp_df['feature'] == 'random_noise_uniform', 'importance'].values[0]
    noise_imp_normal = feature_imp_df.loc[feature_imp_df['feature'] == 'random_noise_normal', 'importance'].values[0]

    noise_threshold = max(noise_imp_uniform, noise_imp_normal)

    print("\n" + "=" * 90)
    print("RESULTADOS DE IMPORTANCIA DE LAS VARIABLES DE RUIDO ALEATORIO:")
    print(f"  - Importancia random_noise_uniform: {noise_imp_uniform:.6f}%")
    print(f"  - Importancia random_noise_normal:  {noise_imp_normal:.6f}%")
    print(f"  => UMBRAL MÁXIMO DE RUIDO ALEATORIO: {noise_threshold:.6f}%")
    print("=" * 90)

    # 5. Filtrar variables reales
    kept_features = feature_imp_df[
        (~feature_imp_df['feature'].isin(['random_noise_uniform', 'random_noise_normal'])) &
        (feature_imp_df['importance'] > noise_threshold)
    ]['feature'].tolist()

    dropped_features_df = feature_imp_df[
        (~feature_imp_df['feature'].isin(['random_noise_uniform', 'random_noise_normal'])) &
        (feature_imp_df['importance'] <= noise_threshold)
    ]

    dropped_features = dropped_features_df['feature'].tolist()

    print(f"\n[Feature Selector] Resumen de Fase 2 (Random Probes):")
    print(f"  - Variables Retenidas (Importancia > {noise_threshold:.6f}%): {len(kept_features)}")
    print(f"  - Variables Eliminadas (Importancia <= {noise_threshold:.6f}%): {len(dropped_features)}")

    print("\nTop 15 Variables Retenidas Más Importantes:")
    print(feature_imp_df[feature_imp_df['feature'].isin(kept_features)].head(15).to_string(index=False))

    if len(dropped_features) > 0:
        print(f"\nVariables Eliminadas por tener Importancia <= Ruido Aleatorio ({len(dropped_features)} total):")
        print(dropped_features_df.to_string(index=False))
    else:
        print("\n¡Todas las variables reales superaron el umbral de ruido aleatorio!")

    # 6. Fase 3: Poda Algorítmica de Colinealidad de Spearman (|rho| >= 0.75)
    print("\n" + "=" * 90)
    print("FASE 3: PODA ALGORÍTMICA DE COLINEALIDAD DE SPEARMAN (|rho| >= 0.75)")
    print("=" * 90)

    corr_pre_path = target_path.parent / "correlation_matrix_37_features.csv"
    corr_opt_path = target_path.parent / "correlation_matrix_21_features.csv"

    sample_df = df[kept_features].sample(n=min(50000, len(df)), random_state=42)
    df_corr_pre = sample_df.corr(method='spearman')
    df_corr_pre.to_csv(corr_pre_path)
    print(f"Matriz de correlación pre-poda guardada en: {corr_pre_path} (dim: {df_corr_pre.shape})")

    fi_dict = dict(zip(feature_imp_df['feature'], feature_imp_df['importance']))
    current_features = list(kept_features)
    pruned_collinear = []

    while True:
        sub_corr = df_corr_pre.loc[current_features, current_features]
        high_pairs = []
        for i in range(len(current_features)):
            for j in range(i + 1, len(current_features)):
                f1, f2 = current_features[i], current_features[j]
                rho = abs(sub_corr.loc[f1, f2])
                if rho >= 0.75:
                    high_pairs.append((f1, f2, rho, sub_corr.loc[f1, f2]))
        
        if not high_pairs:
            break
            
        high_pairs = sorted(high_pairs, key=lambda x: x[2], reverse=True)
        f1, f2, abs_rho, raw_rho = high_pairs[0]
        
        # Reglas contables y de prioridad explicativa:
        if {f1, f2} == {'tag_LiabilitiesAndStockholdersEquity', 'tag_Assets'}:
            drop_feat = 'tag_LiabilitiesAndStockholdersEquity'
        elif {f1, f2} == {'macro_inflation', 'macro_real_gdp'}:
            drop_feat = 'macro_real_gdp'
        elif {f1, f2} == {'tag_CommonStockSharesOutstanding', 'tag_CommonStockSharesIssued'}:
            drop_feat = 'tag_CommonStockSharesOutstanding'
        elif {f1, f2} == {'tag_EntityCommonStockSharesOutstanding', 'tag_CommonStockSharesIssued'}:
            drop_feat = 'tag_CommonStockSharesIssued'
        elif {f1, f2} == {'tag_EntityCommonStockSharesOutstanding', 'tag_CommonStockSharesOutstanding'}:
            drop_feat = 'tag_CommonStockSharesOutstanding'
        elif {f1, f2} == {'tag_Revenues_combined', 'tag_RevenueFromContractWithCustomerExcludingAssessedTax'}:
            drop_feat = 'tag_RevenueFromContractWithCustomerExcludingAssessedTax'
        elif {f1, f2} == {'tag_LiabilitiesCurrent', 'tag_WorkingCapital'}:
            drop_feat = 'tag_LiabilitiesCurrent'
        elif {f1, f2} == {'news_sentiment_range_6m', 'news_sentiment_std_6m'}:
            drop_feat = f1 if fi_dict[f1] < fi_dict[f2] else f2
        else:
            drop_feat = f1 if fi_dict[f1] < fi_dict[f2] else f2
            
        pruned_collinear.append((drop_feat, f1 if drop_feat == f2 else f2, raw_rho))
        current_features.remove(drop_feat)

    print(f"Variables podadas por colinealidad (|rho| >= 0.75): {len(pruned_collinear)}")
    for p in pruned_collinear:
        print(f"  - {p[0]:<42} collinear con {p[1]:<35} (rho = {p[2]:+.4f})")

    optimal_features = list(current_features)
    print(f"\nTAMAÑO FINAL DEL VECTOR ÓPTIMO DE PRODUCCIÓN: {len(optimal_features)} variables.")

    df_corr_opt = df_corr_pre.loc[optimal_features, optimal_features]
    df_corr_opt.to_csv(corr_opt_path)
    print(f"Matriz de correlación post-poda guardada en: {corr_opt_path} (dim: {df_corr_opt.shape})")

    # 7. Crear dataset limpio de producción y exportar
    target_export_cols = [c for c in ['target_bankrupt_12m', 'target_bankrupt_24m', 'is_distress_event', 'is_bankrupt_event'] if c in df.columns]
    final_cols_to_keep = [c for c in meta_cols if c in df.columns] + target_export_cols + optimal_features
    final_cols_to_keep = list(dict.fromkeys(final_cols_to_keep))
    df_clean = df[final_cols_to_keep].copy()

    con = duckdb.connect()
    con.execute(f"COPY (SELECT * FROM df_clean) TO '{CLEAN_PARQUET_PATH}' (FORMAT PARQUET, COMPRESSION SNAPPY);")
    con.close()

    # 8. Guardar reporte JSON oficial unificado
    report_data = {
        "total_initial_candidates": len(real_feature_cols),
        "methodology": "Random Probes Boruta-Style in CatBoost GPU + Dynamic Spearman Pruning |rho| >= 0.75",
        "random_noise_uniform_importance": float(noise_imp_uniform),
        "random_noise_normal_importance": float(noise_imp_normal),
        "noise_threshold": float(noise_threshold),
        "kept_by_random_probes_count": len(kept_features),
        "dropped_by_random_probes_count": len(dropped_features),
        "dropped_by_random_probes": dropped_features,
        "pruned_by_collinearity_count": len(pruned_collinear),
        "pruned_by_collinearity": [p[0] for p in pruned_collinear],
        "final_optimal_features_count": len(optimal_features),
        "final_optimal_features": optimal_features
    }

    with open(REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    print(f"\n[Feature Selector] Proceso completado exitosamente.")
    print(f"  - Dataset limpio de producción: {CLEAN_PARQUET_PATH} ({len(df_clean):,} filas x {len(df_clean.columns)} columnas)")
    print(f"  - Reporte JSON oficial:         {REPORT_JSON}")


if __name__ == "__main__":
    run_random_probes_feature_selection()



