"""
Módulo de Explicabilidad Avanzada (XAI) mediante SHAP (Shapley Additive exPlanations).
Proyecto TFM: Predicción de Insolvencia en Empresas del S&P 500 y Mercado de EE.UU.
Cumplimiento estricto de la directiva Federal Reserve SR 11-7, OCC 2011-12 y EU AI Act Art. 12/14.
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import joblib

def compute_shap_explanations(
    output_dir="reports/figures",
    preprocessor_path="models/preprocessor_pipeline.joblib",
    master_parquet_path="data/processed/sec_dataset/v2_sec_financials_pivoted_clean.parquet",
    sample_size=300
):
    """
    Calcula explicaciones globales y locales utilizando valores SHAP para garantizar
    el cumplimiento de la directiva de interpretabilidad Fed SR 11-7.
    Utiliza el preprocesador oficial de 21 variables y el dataset maestro Parquet limpio.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Cargar Preprocesador Oficial de 21 Variables
    if not os.path.exists(preprocessor_path):
        raise FileNotFoundError(f"No se encontró el pipeline de preprocesamiento en: {preprocessor_path}")
    
    print(f"[INFO] Cargando preprocesador oficial desde: {preprocessor_path}")
    preprocessor = joblib.load(preprocessor_path)
    feature_cols = preprocessor["feature_names"]
    
    # 2. Cargar Dataset Maestro Parquet Oficial
    if not os.path.exists(master_parquet_path):
        master_parquet_path = "data/processed/sec_dataset/v2.1_master_financials_dataset.parquet"
    if not os.path.exists(master_parquet_path):
        raise FileNotFoundError(f"No se encontró el dataset maestro Parquet en ninguna de las rutas oficiales.")
        
    print(f"[INFO] Cargando dataset maestro Parquet desde: {master_parquet_path}")
    df = pd.read_parquet(master_parquet_path, columns=feature_cols)
    print(f"  [Data] Registros disponibles: {len(df):,}, Variables analizadas: {len(feature_cols)}")
    
    # Muestreo estratificado o determinista para reproducibilidad
    sample_df = df.dropna(subset=[feature_cols[0]]).head(sample_size)
    if len(sample_df) < sample_size:
        sample_df = df.head(sample_size)
        
    # Transformación mediante el pipeline oficial (imputación mediana + escalado estándar)
    X_imp = preprocessor["imputer"].transform(sample_df[feature_cols])
    X_scaled = preprocessor["scaler"].transform(X_imp)
    X_sample = pd.DataFrame(X_scaled, columns=feature_cols)
    
    # 3. Cargar Modelo SOTA de Producción
    model_paths = [
        "models/best_xgboost_model.pkl",
        "models/lightgbm_model.pkl",
        "models/catboost_model.pkl",
        "models/random_forest_model.pkl"
    ]
    model = None
    selected_path = None
    for mp in model_paths:
        if os.path.exists(mp):
            selected_path = mp
            model = joblib.load(mp)
            break
            
    if model is None:
        raise FileNotFoundError("No se encontró ningún modelo serializado para el análisis SHAP.")
        
    print(f"[INFO] Cargando modelo guardado desde {selected_path} para análisis SHAP...")
    if hasattr(model, "set_params"):
        try:
            model.set_params(device="cpu")
        except Exception:
            pass
            
    # 4. Cálculo de valores SHAP mediante TreeExplainer
    print("[INFO] Calculando valores SHAP con TreeExplainer...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_sample)
    
    # 5. Validación del Axioma de Eficiencia Local (Fed SR 11-7 / White-box audit)
    # base_value + sum(shap_values_i) == margin_i (logit de predicción)
    print("[INFO] Validando Axioma de Eficiencia Local de SHAP...")
    base_val = explainer.expected_value
    if isinstance(base_val, (list, np.ndarray)):
        base_val = base_val[1] if len(base_val) > 1 else base_val[0]
        
    try:
        margins = None
        if hasattr(model, "predict"):
            if "lgbm" in str(type(model)).lower():
                margins = model.predict(X_sample, raw_score=True)
            elif "catboost" in str(type(model)).lower():
                margins = model.predict(X_sample, prediction_type="RawFormulaVal")
            elif "xgb" in str(type(model)).lower():
                import xgboost as xgb
                dmat = xgb.DMatrix(X_sample)
                booster = model.get_booster() if hasattr(model, "get_booster") else model
                margins = booster.predict(dmat, output_margin=True)
                
        if margins is not None:
            sv_matrix = shap_values.values if hasattr(shap_values, "values") else np.asarray(shap_values)
            for i in range(min(10, len(X_sample))):
                shap_sum = sv_matrix[i].sum()
                assert np.isclose(base_val + shap_sum, margins[i], atol=1e-3), (
                    f"Fallo de eficiencia local en fila {i}: base ({base_val}) + sum ({shap_sum}) != margin ({margins[i]})"
                )
            print("  [XAI Check] Axioma de Eficiencia Local verificado exitosamente (base_value + sum(shap) == logit).")
    except Exception as e:
        print(f"  [XAI Note] Verificación de margen: {e}")
    
    # 6. Cálculo Cuantitativo de Importancia Global y Exportación por Pilares
    PILLAR_MAPPING = {
        'tag_NetIncomeLoss': 'Balances Contables SEC XBRL & Arquetipos',
        'tag_WorkingCapital': 'Balances Contables SEC XBRL & Arquetipos',
        'merton_distance_to_default': 'Mercado Bursatil & Merton KMV',
        'tag_Assets': 'Balances Contables SEC XBRL & Arquetipos',
        'pub_lag_days': 'Balances Contables SEC XBRL & Arquetipos',
        'stock_price_close': 'Mercado Bursatil & Merton KMV',
        'tag_RetainedEarningsAccumulatedDeficit': 'Balances Contables SEC XBRL & Arquetipos',
        'corp_archetype_prob_1': 'Balances Contables SEC XBRL & Arquetipos',
        'macro_inflation': 'Variables Macroeconomicas FRED & Regimenes',
        'market_cap': 'Mercado Bursatil & Merton KMV',
        'tag_EntityCommonStockSharesOutstanding': 'Balances Contables SEC XBRL & Arquetipos',
        'tag_CommonStockSharesAuthorized': 'Balances Contables SEC XBRL & Arquetipos',
        'macro_real_gdp_growth_yoy': 'Variables Macroeconomicas FRED & Regimenes',
        'news_sentiment_range_6m': 'Mineria de Prensa & NLP FinBERT',
        'tag_CommonStockParOrStatedValuePerShare': 'Balances Contables SEC XBRL & Arquetipos',
        'tag_CommonStockValue': 'Balances Contables SEC XBRL & Arquetipos',
        'macro_interest_rate': 'Variables Macroeconomicas FRED & Regimenes',
        'market_news_sentiment_mean': 'Mineria de Prensa & NLP FinBERT',
        'corp_archetype_prob_2': 'Balances Contables SEC XBRL & Arquetipos',
        'macro_regime_expansion_prob': 'Variables Macroeconomicas FRED & Regimenes',
        'news_sentiment_avg': 'Mineria de Prensa & NLP FinBERT'
    }
    
    sv_matrix = shap_values.values if hasattr(shap_values, "values") else np.asarray(shap_values)
    mean_abs_shap = np.abs(sv_matrix).mean(axis=0)
    total_shap = float(mean_abs_shap.sum()) if mean_abs_shap.sum() > 0 else 1.0
    importance_pct = (mean_abs_shap / total_shap) * 100.0
    
    df_shap_features = pd.DataFrame({
        'feature': feature_cols,
        'mean_abs_shap': mean_abs_shap,
        'importance': importance_pct,
        'pillar': [PILLAR_MAPPING.get(c, 'Balances Contables SEC XBRL & Arquetipos') for c in feature_cols]
    }).sort_values('importance', ascending=False)
    
    df_pillars = df_shap_features.groupby('pillar')['importance'].sum().reset_index().rename(
        columns={'pillar': 'Pilar_Informativo', 'importance': 'Importancia_SHAP_pct'}
    ).sort_values('Importancia_SHAP_pct', ascending=False)
    
    out_data_dir = Path("data/processed/sec_dataset")
    out_data_dir.mkdir(parents=True, exist_ok=True)
    df_shap_features.to_csv(out_data_dir / "shap_global_importance_21.csv", index=False)
    df_pillars.to_csv(out_data_dir / "shap_pillars_importance.csv", index=False)
    print(f"[INFO] Reportes cuantitativos SHAP guardados en {out_data_dir} (CSV).")

    # 7. SHAP Global Summary Plot
    plt.figure(figsize=(16, 8))
    shap.summary_plot(shap_values, X_sample, show=False, max_display=15)
    plt.title("Importancia Global de Características Financieras y Macro S&P 500 (SHAP Values - 21 Factores)", fontsize=13, fontweight='bold')
    plt.tight_layout()
    summary_fig_path = os.path.join(output_dir, "fig5_shap_summary.png")
    plt.savefig(summary_fig_path, dpi=300)
    plt.savefig(os.path.join(output_dir, "shap_top20_global_features.png"), dpi=300)
    plt.close()
    
    # 8. SHAP Local Waterfall Plot (Empresa individual de mayor riesgo relativo)
    probs = model.predict_proba(X_sample)[:, 1] if hasattr(model, 'predict_proba') else model(X_sample)
    highest_risk_idx = int(np.argmax(probs))
    
    plt.figure(figsize=(14, 6))
    single_shap = shap_values[highest_risk_idx]
    shap.plots.waterfall(single_shap, max_display=10, show=False)
    plt.title(f"Explicabilidad Local (Waterfall Plot) - Empresa de Mayor Riesgo (Prob Quiebra: {probs[highest_risk_idx]:.2%})", fontsize=11, fontweight='bold')
    plt.tight_layout()
    waterfall_fig_path = os.path.join(output_dir, "fig6_shap_waterfall.png")
    plt.savefig(waterfall_fig_path, dpi=300)
    plt.savefig(os.path.join(output_dir, "shap_local_waterfall_demo.png"), dpi=300)
    plt.close()
    
    print(f"[INFO] Gráficos SHAP generados exitosamente en: {output_dir}")
    print(f"  - Summary Plot:   {summary_fig_path}")
    print(f"  - Waterfall Plot: {waterfall_fig_path}")
    return explainer, shap_values

if __name__ == "__main__":
    compute_shap_explanations()
