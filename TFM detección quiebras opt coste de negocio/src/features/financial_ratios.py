"""
Módulo de Ingeniería de Características Financieras con Nombres Descriptivos, Ratios, Altman Z-Score, Merton KMV y Deltas Temporales.
Proyecto TFM: Predicción de Insolvencia en Empresas del S&P 500.
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import norm
from src.data.quality_audit import load_raw_data

def compute_financial_ratios(df):
    """
    Calcula ratios financieros estructurales utilizando nombres descriptivos de variables contables.
    """
    data = df.copy()
    eps = 1e-5  # Evitar división por cero
    
    # 1. Ratios del Altman Z-Score Clásico para empresas cotizadas
    # R1: Working Capital / Total Assets
    data['working_capital'] = data['current_assets'] - data['total_current_liabilities']
    data['R1_WC_TA'] = data['working_capital'] / (data['total_assets'] + eps)
    
    # R2: Retained Earnings / Total Assets
    data['R2_RE_TA'] = data['retained_earnings'] / (data['total_assets'] + eps)
    
    # R3: EBIT / Total Assets (ROA Operativo)
    data['R3_EBIT_TA'] = data['ebit'] / (data['total_assets'] + eps)
    
    # R4: Market Value of Equity / Total Liabilities
    data['R4_MVE_TL'] = data['market_value'] / (data['total_liabilities'] + eps)
    
    # R5: Total Revenue / Total Assets (Rotación de Activos)
    data['R5_Sales_TA'] = data['total_revenue'] / (data['total_assets'] + eps)
    
    # Altman Z-Score Modificado
    data['altman_z_score'] = (
        1.2 * data['R1_WC_TA'] +
        1.4 * data['R2_RE_TA'] +
        3.3 * data['R3_EBIT_TA'] +
        0.6 * data['R4_MVE_TL'] +
        0.999 * data['R5_Sales_TA']
    )
    
    # 2. Ratios Adicionales de Liquidez, Solvencia y Cobertura
    data['R6_Current_Ratio'] = data['current_assets'] / (data['total_current_liabilities'] + eps)
    data['R7_Quick_Ratio'] = (data['current_assets'] - data['inventory']) / (data['total_current_liabilities'] + eps)
    data['R8_Debt_to_Assets'] = data['total_liabilities'] / (data['total_assets'] + eps)
    data['R9_Net_Margin'] = data['net_income'] / (data['total_revenue'] + eps)
    data['R10_EBITDA_Margin'] = data['ebitda'] / (data['total_revenue'] + eps)
    
    return data

def compute_advanced_financial_ratios(df):
    """
    Calcula ratios financieros avanzados: Merton KMV Distance-to-Default proxy,
    intereses, volatilidad rolling y métricas de estructura de capital.
    """
    data = df.copy()
    eps = 1e-5
    
    # 1. Proxies de Modelo Estructural de Merton KMV (Distance to Default)
    # Valor de la firma approx = Valor de Mercado + Deuda Total
    data['firm_value'] = data['market_value'] + data['total_liabilities']
    # Punto de Default (Punto de quiebra contable) = Pasivo Corriente + 0.5 * Deuda a Largo Plazo
    data['default_point'] = data['total_current_liabilities'] + 0.5 * data['total_long_term_debt']
    
    # Volatilidad móvil de la empresa (calculada a 3 años por empresa)
    df_sorted = data.sort_values(by=['company_name', 'year']).copy()
    df_sorted['roa_operating'] = df_sorted['ebit'] / (df_sorted['total_assets'] + eps)
    df_sorted['roa_volatility_3y'] = (
        df_sorted.groupby('company_name')['roa_operating']
        .transform(lambda x: x.rolling(window=3, min_periods=1).std())
        .fillna(0.01)
    )
    data['roa_volatility_3y'] = df_sorted['roa_volatility_3y']
    
    # Distancia a la Quiebra (Distance-to-Default DD)
    v_over_d = np.log((data['firm_value'] + eps) / (data['default_point'] + eps))
    vol = data['roa_volatility_3y'] + eps
    data['merton_distance_to_default'] = v_over_d / vol
    data['merton_pd_proxy'] = norm.cdf(-data['merton_distance_to_default'])
    
    # 2. Cobertura de Intereses y Carga Estructural de Deuda
    data['interest_coverage_ratio'] = data['ebitda'] / (data['total_long_term_debt'] * 0.05 + eps)
    data['cash_flow_to_total_debt'] = (data['net_income'] + data['depreciation_and_amortization']) / (data['total_liabilities'] + eps)
    data['working_capital_to_sales'] = (data['current_assets'] - data['total_current_liabilities']) / (data['total_revenue'] + eps)
    
    return data

def compute_velocity_deltas(df):
    """
    Calcula las variables de velocidad de cambio temporal (Delta = X_t - X_{t-1})
    ordenando las observaciones por empresa y año.
    """
    df_sorted = df.sort_values(by=['company_name', 'year']).copy()
    
    key_features = [
        'R1_WC_TA', 'R3_EBIT_TA', 'R4_MVE_TL', 'R8_Debt_to_Assets',
        'altman_z_score', 'ebitda', 'net_income', 'merton_distance_to_default'
    ]
    
    for feat in key_features:
        if feat in df_sorted.columns:
            delta_name = f'delta_{feat}'
            df_sorted[delta_name] = df_sorted.groupby('company_name')[feat].diff().fillna(0)
    
    return df_sorted

def run_feature_engineering_pipeline(output_path="data/processed/financial_features.csv"):
    """Pipeline principal de ingeniería de características financieras con nombres descriptivos."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df = load_raw_data()
    
    print("[INFO] Calculando ratios financieros estructurales y Altman Z-Score con nombres descriptivos...")
    df_ratios = compute_financial_ratios(df)
    
    print("[INFO] Calculando ratios avanzados de Merton KMV y volatilidad estructural...")
    df_advanced = compute_advanced_financial_ratios(df_ratios)
    
    print("[INFO] Calculando variables de velocidad de deterioro temporal (Deltas)...")
    df_features = compute_velocity_deltas(df_advanced)
    
    df_features.to_csv(output_path, index=False)
    print(f"[INFO] Dataset con {df_features.shape[1]} características guardado con éxito en: {output_path}")
    return df_features

if __name__ == "__main__":
    run_feature_engineering_pipeline()
