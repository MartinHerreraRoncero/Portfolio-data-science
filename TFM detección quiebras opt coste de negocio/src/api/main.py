"""
Servidor RESTful FastAPI con Interfaz Grafica Interactiva (GUI), Soporte para 6 Modelos
(Voting Classifier SOTA Rank 1 y 5 Modelos Individuales) y Descarga Automatica de Estados Contables.
Proyecto TFM: Prediccion de Insolvencia en Empresas del S&P 500 y Mercado de EE.UU.
Cumplimiento EU AI Act Art. 12/14 y Federal Reserve SR 11-7.
"""

import os
import re
import sys
import json
import time
import random
import logging
from datetime import datetime
from pathlib import Path
from contextlib import asynccontextmanager
from typing import List, Optional, Dict, Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../models")))

from src.api.schemas import (
    CompanyFinancialPayload,
    PredictionResponse,
    BatchPredictionResponse,
    ModelInfoResponse,
    ScorecardResponse
)

CONFIG_PATH = "config/production.json"
config = {
    "version": "2.3.0",
    "environment": "production",
    "default_model": "voting_classifier",
    "preprocessor_path": "models/preprocessor_pipeline.joblib",
    "scorecard_path": "models/scorecard_woe_best.pkl",
    "log_file": "logs/audit_logs.jsonl"
}

MODEL_THRESHOLDS = {
    "voting_classifier": 0.158,
    "lightgbm": 0.224,
    "random_forest": 0.182,
    "catboost": 0.038,
    "xgboost": 0.158,
    "logistic_regression": 0.170
}

MODEL_DISPLAY_NAMES = {
    "voting_classifier": "Voting Classifier Ensemble (Top 1)",
    "lightgbm": "LightGBM SOTA (Top 2)",
    "random_forest": "Random Forest (Top 3)",
    "catboost": "CatBoost GPU (Top 4)",
    "xgboost": "XGBoost GPU (Top 5)",
    "logistic_regression": "Logistic Regression (Baseline)"
}

if os.path.exists(CONFIG_PATH):
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg_data = json.load(f)
            config.update(cfg_data)
            if "models" in cfg_data:
                for m_key, m_val in cfg_data["models"].items():
                    if "threshold" in m_val:
                        MODEL_THRESHOLDS[m_key] = m_val["threshold"]
                    if "name" in m_val:
                        MODEL_DISPLAY_NAMES[m_key] = m_val["name"]
    except Exception as e:
        print(f"[WARN] No se pudo leer {CONFIG_PATH}: {e}")

audit_logger = logging.getLogger("eu_ai_act_audit_logger")
audit_logger.setLevel(logging.INFO)
if not audit_logger.handlers:
    log_file = config.get("log_file", "logs/audit_logs.jsonl")
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(message)s"))
    audit_logger.addHandler(fh)

loaded_models = {}
preprocessor = None
scorecard = None
sample_bank = None
distribution_stats = None

def set_cpu_device(model_obj):
    if hasattr(model_obj, "set_params"):
        try:
            model_obj.set_params(device="cpu")
        except Exception:
            pass
    if hasattr(model_obj, "get_booster"):
        try:
            model_obj.get_booster().set_param({"device": "cpu"})
        except Exception:
            pass
    if hasattr(model_obj, "calibrated_classifiers_"):
        for cc in model_obj.calibrated_classifiers_:
            base_est = getattr(cc, "estimator", getattr(cc, "base_estimator", None))
            if base_est is not None:
                set_cpu_device(base_est)
    for attr in ["lgb", "rf", "cb", "xgb"]:
        if hasattr(model_obj, attr):
            set_cpu_device(getattr(model_obj, attr))

def load_artifacts():
    global loaded_models, preprocessor, scorecard, sample_bank, distribution_stats
    
    # 1. Cargar Preprocesador
    prep_path = config.get("preprocessor_path", "models/preprocessor_pipeline.joblib")
    if os.path.exists(prep_path):
        try:
            preprocessor = joblib.load(prep_path)
            print(f"[INFO] Preprocesador cargado desde: {prep_path}")
        except Exception as e:
            print(f"[ERROR] Preprocesador fallo: {e}")

    # 2. Cargar los Modelos de Produccion
    model_configs = {
        "voting_classifier": "models/voting_classifier.pkl",
        "lightgbm": "models/lightgbm_model.pkl",
        "random_forest": "models/random_forest_model.pkl",
        "catboost": "models/catboost_model.pkl",
        "xgboost": "models/best_xgboost_model.pkl",
        "logistic_regression": "models/logistic_regression_model.pkl"
    }
    for m_key, m_path in model_configs.items():
        if os.path.exists(m_path):
            try:
                m = joblib.load(m_path)
                set_cpu_device(m)
                loaded_models[m_key] = m
                print(f"[INFO] Modelo [{m_key}] cargado con exito.")
            except Exception as e:
                print(f"[WARN] No se pudo cargar modelo {m_key}: {e}")

    # 3. Cargar Scorecard WoE
    sc_path = config.get("scorecard_path", "models/scorecard_woe_best.pkl")
    if os.path.exists(sc_path):
        try:
            scorecard = joblib.load(sc_path)
            print(f"[INFO] Scorecard WoE cargado desde: {sc_path}")
        except Exception as e:
            print(f"[WARN] Scorecard WoE no disponible: {e}")

    # 4. Cargar banco de muestras
    bank_path = "models/sample_companies_bank.joblib"
    if os.path.exists(bank_path):
        try:
            sample_bank = joblib.load(bank_path)
            print("[INFO] Banco de empresas de muestra cargado.")
        except Exception as e:
            print(f"[WARN] Banco de muestras no disponible: {e}")

    # 5. Cargar estadisticas
    dist_path = "models/distribution_stats.joblib"
    if os.path.exists(dist_path):
        try:
            distribution_stats = joblib.load(dist_path)
            print("[INFO] Estadisticas multivariantes cargadas.")
        except Exception as e:
            print(f"[WARN] Estadisticas no disponibles: {e}")

load_artifacts()

@asynccontextmanager
async def lifespan(app: FastAPI):
    if not loaded_models or preprocessor is None:
        load_artifacts()
    yield
    print("[INFO] Servicio finalizado.")

app = FastAPI(
    title="S&P 500 Corporate Bankruptcy & Credit Risk API",
    description="Microservicio de Inferencia de Riesgo Crediticio con soporte multimodelo, XAI, Descarga Automatica por Ticker y Dashboard Interactivo.",
    version=config.get("version", "2.3.0"),
    lifespan=lifespan
)

ALLOWED_ORIGINS = [
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:8501",
    "http://127.0.0.1:8501"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

FEATURE_HUMAN_LABELS = {
    # --- Pilar Contable (SEC) ---
    "tag_NetIncomeLoss": ("Resultado neto contable (Net Income)", "rentabilidad contable y generación de beneficios"),
    "tag_WorkingCapital": ("Fondo de maniobra neto (Working Capital)", "liquidez operativa corriente"),
    "tag_Assets": ("Activos totales consolidados", "base total de activos corporativos"),
    "pub_lag_days": ("Retardo de publicación contable SEC 10-K/10-Q", "puntualidad y transparencia contable"),
    "tag_RetainedEarningsAccumulatedDeficit": ("Reservas y beneficios acumulados / Déficit", "estabilidad patrimonial histórica"),
    "tag_EntityCommonStockSharesOutstanding": ("Acciones comunes en circulación", "base de accionistas y dilución"),
    "tag_CommonStockSharesAuthorized": ("Acciones comunes autorizadas", "estructura de capital social autorizada"),
    "tag_CommonStockParOrStatedValuePerShare": ("Valor facial por acción común", "denominación de acciones"),
    "tag_CommonStockValue": ("Valor nominal de acciones comunes", "capital social emitido"),

    # --- Pilar Bursátil y Estructural ---
    "merton_distance_to_default": ("Distancia estructural de Merton a la quiebra (DD)", "riesgo de insolvencia estructural"),
    "stock_price_close": ("Precio de cotización bursátil", "valoración de mercado"),
    "market_cap": ("Capitalización bursátil total", "valor de capitalización y tamaño"),

    # --- Pilar Macroeconómico y Regímenes ---
    "macro_inflation": ("Nivel de inflación macroeconómica (CPI)", "presión inflacionaria de costes"),
    "macro_real_gdp_growth_yoy": ("Crecimiento interanual PIB real", "ciclo económico y actividad"),
    "macro_interest_rate": ("Tipo de interés de referencia (Fed Funds)", "coste macroeconómico de crédito"),
    "macro_regime_expansion_prob": ("Probabilidad de régimen macroeconómico de expansión (GMM)", "estabilidad del entorno macroeconómico"),

    # --- Pilar Sentimiento NLP y Noticias ---
    "news_sentiment_range_6m": ("Volatilidad del sentimiento mediático (6m)", "estabilidad informativa corporativa"),
    "market_news_sentiment_mean": ("Sentimiento medio global del mercado", "clima global de mercado"),
    "news_sentiment_avg": ("Sentimiento medio de noticias corporativas (FinBERT)", "percepción mediática corporativa"),

    # --- Pilar Arquetipos Corporativos ---
    "corp_archetype_prob_1": ("Probabilidad de pertenencia al Arquetipo Corporativo 1 (GMM)", "perfil estructural de madurez / escala"),
    "corp_archetype_prob_2": ("Probabilidad de pertenencia al Arquetipo Corporativo 2 (GMM)", "perfil estructural de liquidez / apalancamiento"),

    # --- Campos legacy auxiliares ---
    "tag_CashAndCashEquivalentsAtCarryingValue": ("Tesorería y equivalentes de efectivo", "cobertura de pasivos inmediatos"),
    "tag_AssetsCurrent": ("Activo corriente / circulante", "capacidad operativa a corto plazo"),
    "tag_StockholdersEquity": ("Fondos propios y patrimonio neto", "solvencia y respaldo patrimonial"),
    "tag_InterestExpense": ("Gastos financieros por intereses de deuda", "carga y coste financiero"),
    "tag_Revenues_combined": ("Ingresos operacionales consolidados", "flujo de ingresos"),
    "tag_RevenueFromContractWithCustomerExcludingAssessedTax": ("Ingresos netos por contratos con clientes", "ventas operativas directas"),
    "nlp_sentiment_decayed_30d": ("Sentimiento de noticias FinBERT (30d)", "percepción mediática del riesgo"),
    "news_sentiment_max": ("Pico de sentimiento mediático", "picos de noticias"),
    "macro_unemployment_rate": ("Tasa macroeconómica de desempleo", "entorno laboral macro")
}

shap_explainers: Dict[str, Any] = {}

def get_tree_explainer(model_name: str = "voting_classifier"):
    global shap_explainers
    if model_name == "voting_classifier":
        return get_tree_explainer("xgboost")
    if model_name not in shap_explainers:
        m_obj = loaded_models.get(model_name) or loaded_models.get("xgboost") or loaded_models.get("lightgbm")
        if m_obj is not None:
            try:
                import shap
                shap_explainers[model_name] = shap.TreeExplainer(m_obj)
            except Exception as e:
                print(f"[WARN] No se pudo inicializar SHAP TreeExplainer para {model_name}: {e}")
                shap_explainers[model_name] = None
        else:
            shap_explainers[model_name] = None
    return shap_explainers.get(model_name)

def format_metric_value(val: Any) -> str:
    if not isinstance(val, (int, float)) or isinstance(val, bool):
        return str(val)
    sign = "-" if val < 0 else ""
    abs_v = abs(val)
    if abs_v >= 1e9:
        return f"{sign}${abs_v/1e9:,.2f}B"
    elif abs_v >= 1e6:
        return f"{sign}${abs_v/1e6:,.2f}M"
    elif abs_v >= 1e3:
        return f"{sign}${abs_v/1e3:,.2f}K"
    elif abs_v == 0.0:
        return "$0.00"
    return f"{val:.2f}"

def normalize_macro_inflation(val: Any, base_cpi: float = 236.468) -> float:
    """
    Normaliza macro_inflation para erradicar el Training-Serving Skew.
    
    El pipeline de entrenamiento ajustó el StandardScaler sobre el nivel absoluto del
    Consumer Price Index (CPIAUCSL, nivel base ~236.468 en 2014-2018, std=7.76).
    Si el cliente suministra la inflación como tasa porcentual anual (ej. 2.5% o valor <= 25.0),
    pasar directamente ese valor generaría una distorsión de -30 sigma ((2.5 - 236.65) / 7.76 = -30.16).
    
    Esta función detecta tasas porcentuales (val <= 25.0) y las proyecta coherentemente
    al nivel de índice CPI esperado: CPI = base_cpi * (1.0 + val / 100.0).
    Si el valor ya viene en la escala de índice CPI (> 25.0), se preserva intacto.
    """
    if val is None:
        return base_cpi
    try:
        f_val = float(val)
        if np.isnan(f_val):
            return base_cpi
    except (ValueError, TypeError):
        return base_cpi
    if -50.0 <= f_val <= 25.0:
        return base_cpi * (1.0 + f_val / 100.0)
    return f_val

def compute_approximate_altman_z(payload_dict: dict) -> float:
    ta = payload_dict.get("tag_Assets") or payload_dict.get("total_assets") or 1.0
    if ta <= 0:
        ta = 1.0
    wc = payload_dict.get("tag_WorkingCapital") or 0.0
    re_val = payload_dict.get("tag_RetainedEarningsAccumulatedDeficit") or payload_dict.get("retained_earnings") or 0.0
    ebit = payload_dict.get("ebit") or payload_dict.get("tag_NetIncomeLoss") or 0.0
    mve = payload_dict.get("market_cap") or payload_dict.get("market_value") or ta
    tl = payload_dict.get("total_liabilities") or (ta - (payload_dict.get("tag_StockholdersEquity") or 0.0))
    if tl <= 0:
        tl = ta * 0.5
    sales = payload_dict.get("tag_Revenues_combined") or payload_dict.get("total_revenue") or ta

    r1 = wc / ta
    r2 = re_val / ta
    r3 = ebit / ta
    r4 = mve / tl
    r5 = sales / ta
    z = 1.2 * r1 + 1.4 * r2 + 3.3 * r3 + 0.6 * r4 + 0.999 * r5
    return float(z)

def generate_adverse_action_notices(
    payload_dict: dict,
    prob: float,
    tau: float,
    model_name: str = "lightgbm",
    scaled_x: Optional[np.ndarray] = None
) -> List[str]:
    """
    Generación dinámica de Cartas de Acción Adversa conforme a ECOA (Equal Credit
    Opportunity Act), CFPB Circular 2022-03 y EU AI Act Art. 12/14.
    
    En lugar de reglas fijas o textos genéricos, extrae los principales factores
    de riesgo que empujaron el modelo hacia la insolvencia mediante SHAP TreeExplainer
    (contribución log-odds positiva) o análisis de desviaciones relativas del vector de 26 variables.
    """
    notices = []
    if prob < tau:
        return notices

    if preprocessor is None:
        return ["Riesgo crediticio elevado detectado por el modelo multivariante de producción."]

    feature_names = preprocessor["feature_names"]
    defaults = preprocessor["default_medians"]

    # 1. Intentar extracción dinámica de factores de riesgo con SHAP TreeExplainer
    explainer = get_tree_explainer(model_name)
    if explainer is not None:
        try:
            if scaled_x is None:
                scaled_x = prepare_feature_vector(payload_dict)
            X_df = pd.DataFrame(scaled_x, columns=feature_names)
            sv = explainer(X_df)
            shap_vals = sv.values[0] if hasattr(sv, "values") else np.asarray(sv)[0]
            if hasattr(shap_vals, "ndim") and shap_vals.ndim > 1 and shap_vals.shape[-1] == 2:
                shap_vals = shap_vals[..., 1]
            
            # Seleccionar características cuya contribución SHAP incrementa el riesgo de quiebra (shap > 0)
            adverse_indices = [i for i in np.argsort(shap_vals)[::-1] if shap_vals[i] > 0.01][:5]
            
            for idx in adverse_indices:
                feat = feature_names[idx]
                label, area = FEATURE_HUMAN_LABELS.get(feat, (feat, "factor contable/financiero"))
                raw_val = payload_dict.get(feat, defaults.get(feat, 0.0))
                val_str = format_metric_value(raw_val)
                impact = shap_vals[idx]
                notices.append(
                    f"{label}: nivel reportado en {val_str} incrementa severamente el riesgo de quiebra "
                    f"(impacto SHAP: +{impact:.2f} logit en {area})"
                )
        except Exception as e:
            print(f"[WARN] Error en extracción dinámica SHAP: {e}")

    # 2. Si SHAP no produjo factores suficientes (ej. modelo no-árbol o error), usar desviaciones relativas
    if not notices:
        try:
            if scaled_x is None:
                scaled_x = prepare_feature_vector(payload_dict)
            adverse_high_cols = {
                "macro_interest_rate", "pub_lag_days", "macro_inflation",
                "news_sentiment_range_6m", "tag_InterestExpense", "macro_unemployment_rate"
            }
            scores = []
            for i, col in enumerate(feature_names):
                z = scaled_x[0, i]
                risk_dir = z if col in adverse_high_cols else -z
                scores.append((risk_dir, i, col))
            scores.sort(reverse=True)
            for risk_dir, idx, feat in scores[:5]:
                if risk_dir > 0.1:
                    label, area = FEATURE_HUMAN_LABELS.get(feat, (feat, "factor financiero"))
                    raw_val = payload_dict.get(feat, defaults.get(feat, 0.0))
                    val_str = format_metric_value(raw_val)
                    notices.append(
                        f"{label}: nivel reportado en {val_str} desvía negativamente el perfil de solvencia "
                        f"({risk_dir:+.2f} sigma en {area})"
                    )
        except Exception as e:
            print(f"[WARN] Error en cálculo de desviaciones relativas: {e}")

    # 3. Incorporar indicador complementario estructural de Merton o Altman Z si está en zona de estrés
    mdd = payload_dict.get("merton_distance_to_default")
    if mdd is not None and mdd < 1.5:
        mdd_notice = f"Distancia estructural de Merton en zona crítica ({mdd:.2f} desviaciones < 1.50 umbral regulatorio)"
        if not any("Merton" in n for n in notices):
            notices.append(mdd_notice)

    altman_z = compute_approximate_altman_z(payload_dict)
    if altman_z < 1.81 and not any("Altman" in n for n in notices):
        notices.append(f"Altman Z-Score en zona crítica de estrés financiero ({altman_z:.2f} < 1.81)")

    return notices

def prepare_feature_vector(payload_dict: dict) -> np.ndarray:
    if preprocessor is None:
        raise HTTPException(status_code=503, detail="Preprocesador de produccion no disponible.")
    feature_names = preprocessor["feature_names"]
    defaults = preprocessor["default_medians"]
    base_cpi = defaults.get("macro_inflation", 236.468)

    row = []
    for col in feature_names:
        val = payload_dict.get(col)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            val = defaults.get(col, 0.0)
        elif col == "macro_inflation":
            val = normalize_macro_inflation(val, base_cpi=base_cpi)
        row.append(float(val))

    raw_df = pd.DataFrame([row], columns=feature_names, dtype=np.float64)
    imp_mat = preprocessor["imputer"].transform(raw_df)
    scaled_mat = preprocessor["scaler"].transform(imp_mat)
    return scaled_mat

def fetch_company_data(ticker_or_name: str, force_refresh: bool = False, defaults: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Obtiene los datos más frescos de la empresa aplicando la Estrategia A (Cache-Aside):
    - Balances contables (Caché SQLite 90d / DuckDB SEC / yfinance).
    - Cotización bursátil, market cap y recálculo de Merton DD diarios (24h).
    - Sentimiento de noticias reciente analizado con FinBERT diario (24h).
    - Series macroeconómicas de FRED API diarias (24h).
    """
    from src.api.live_data_service import get_freshest_company_data
    if defaults is None and preprocessor:
        defaults = preprocessor.get("default_medians", {})
    return get_freshest_company_data(ticker_or_name, force_refresh=force_refresh, defaults=defaults)

@app.get("/health")
async def health_check():
    return {
        "status": "online",
        "api_version": config.get("version", "2.3.0"),
        "model_loaded": len(loaded_models) > 0,
        "preprocessor_loaded": preprocessor is not None,
        "scorecard_loaded": scorecard is not None,
        "available_models": list(loaded_models.keys()),
        "default_model": config.get("default_model", "voting_classifier"),
        "environment": config.get("environment", "production")
    }

@app.get("/model_info", response_model=ModelInfoResponse)
async def model_info():
    feature_names = preprocessor["feature_names"] if preprocessor else []
    def_model = config.get("default_model", "voting_classifier")
    return ModelInfoResponse(
        model_name="Voting Classifier Ensemble (Rank 1 SOTA) + Full 6-Model Benchmark",
        version=config.get("version", "2.3.0"),
        features_count=len(feature_names),
        features=feature_names,
        optimal_decision_threshold=MODEL_THRESHOLDS.get(def_model, 0.158),
        framework="Voting Classifier (25% c/u) + LightGBM + Random Forest + CatBoost + XGBoost + Logistic Regression",
        hardware_acceleration="NVIDIA GPU RTX 5070 (Inference: Low-Latency Multi-Threaded CPU)",
        regulatory_compliance=[
            "Federal Reserve SR 11-7 (Model Risk Management)",
            "Basel III / IV IRB Advanced Standard",
            "EU AI Act (Regulation EU 2024/1689, Articles 12 & 14)",
            "Fair Credit Reporting Act (FCRA Adverse Action Notices)"
        ]
    )

@app.get("/sample_company")
async def get_sample_company(mode: str = Query("random", enum=["solvent", "distressed", "random", "synthetic"])):
    if sample_bank is not None and mode in ["solvent", "distressed", "random"]:
        if mode == "solvent":
            raw_solv = sample_bank.get("solvent_samples", [])
            healthy_solv = [
                s for s in raw_solv
                if float(s.get("tag_WorkingCapital") or 0.0) > 0
                and float(s.get("tag_NetIncomeLoss") or 0.0) > 0
                and float(s.get("tag_RetainedEarningsAccumulatedDeficit") or 0.0) > 0
                and float(s.get("merton_distance_to_default") or 0.0) > 1.5
            ]
            pool = healthy_solv if healthy_solv else raw_solv
        elif mode == "distressed":
            pool = sample_bank.get("distressed_samples", [])
        elif mode == "random":
            solv_p = sample_bank.get("solvent_samples", [])
            dist_p = sample_bank.get("distressed_samples", [])
            if random.random() < 0.12 and dist_p:
                pool = dist_p
            else:
                pool = solv_p if solv_p else dist_p
        else:
            pool = []
        if pool:
            raw_rec = dict(random.choice(pool))
            defaults = preprocessor["default_medians"] if preprocessor else {}
            clean_rec = {}
            for k, v in raw_rec.items():
                if k in ["raw_features", "scaled_features", "scaled_vector"]:
                    continue
                if isinstance(v, (float, np.floating)):
                    if np.isnan(v) or np.isinf(v):
                        clean_rec[k] = float(defaults.get(k, 0.0))
                    else:
                        clean_rec[k] = float(v)
                elif isinstance(v, (int, np.integer)):
                    clean_rec[k] = int(v)
                elif isinstance(v, (str, bool)) or v is None:
                    clean_rec[k] = v
            if "year" not in clean_rec and "filed_year" in clean_rec:
                clean_rec["year"] = int(clean_rec["filed_year"])
            return {
                "source": f"Empirical Pool ({mode.upper()})",
                "company_data": clean_rec
            }

    if distribution_stats is not None:
        features = distribution_stats["features"]
        base = distribution_stats["distressed_median"] if mode == "distressed" else (
            distribution_stats["solvent_median"] if mode == "solvent" else distribution_stats["mean"]
        )
        sample = {
            "company_name": f"Synthetic_{mode.capitalize()}_Corp",
            "year": 2024,
            "ticker": f"SYN_{random.randint(100, 999)}"
        }
        for f in features:
            med = base[f]
            pcts = distribution_stats["percentiles"][f]
            iqr = max(abs(pcts[3] - pcts[1]), abs(med) * 0.15, 1.0)
            val = med + np.random.normal(0, iqr * 0.10)
            if any(k in f for k in ["Assets", "Cash", "stock_price", "market_cap", "unemployment", "inflation"]):
                val = max(0.01, val)
            sample[f] = round(float(val), 2)
        return {
            "source": f"Multivariate Synthetic ({mode.upper()})",
            "company_data": sample
        }

    return {
        "source": "Default Fallback",
        "company_data": {
            "company_name": "Demo_Corp",
            "year": 2024,
            "tag_NetIncomeLoss": 2500000.0,
            "tag_WorkingCapital": 10000000.0,
            "merton_distance_to_default": 3.5,
            "tag_Assets": 50000000.0,
            "pub_lag_days": 40.0,
            "stock_price_close": 45.0,
            "tag_RetainedEarningsAccumulatedDeficit": 8000000.0,
            "corp_archetype_prob_1": 0.05,
            "macro_inflation": 2.8,
            "market_cap": 150000000.0,
            "tag_EntityCommonStockSharesOutstanding": 5000000.0,
            "tag_CommonStockSharesAuthorized": 10000000.0,
            "macro_real_gdp_growth_yoy": 2.2,
            "news_sentiment_range_6m": 0.20,
            "tag_CommonStockParOrStatedValuePerShare": 0.01,
            "tag_CommonStockValue": 50000.0,
            "macro_interest_rate": 4.5,
            "market_news_sentiment_mean": 0.50,
            "corp_archetype_prob_2": 0.01,
            "macro_regime_expansion_prob": 0.80,
            "news_sentiment_avg": 0.25
        }
    }

@app.get("/fetch_company")
async def get_company_by_ticker(
    ticker: str = Query(..., description="Ticker bursatil o nombre (ej: BA, AAPL, TSLA, NVDA, F, BBBY)"),
    force_refresh: bool = Query(False, description="Forzar actualizacion en vivo de mercado, contabilidad y noticias")
):
    res = fetch_company_data(ticker, force_refresh=force_refresh)
    if not res.get("found"):
        raise HTTPException(
            status_code=404,
            detail=res.get("error", f"Empresa o ticker '{ticker}' no encontrado en los registros oficiales.")
        )
    return res

@app.post("/fetch_and_predict")
async def fetch_and_predict(
    ticker: str = Query(..., description="Ticker bursatil o nombre de la empresa"),
    model_name: Optional[str] = Query("voting_classifier", enum=["voting_classifier", "lightgbm", "random_forest", "catboost", "xgboost", "logistic_regression"]),
    force_refresh: bool = Query(False, description="Forzar actualizacion en vivo de mercado, contabilidad y noticias")
):
    fetched = fetch_company_data(ticker, force_refresh=force_refresh)
    if not fetched.get("found"):
        raise HTTPException(
            status_code=404,
            detail=fetched.get("error", f"Empresa o ticker '{ticker}' no encontrado en los registros oficiales.")
        )
    c_data = fetched.get("company_data", {})
    valid_keys = set(CompanyFinancialPayload.model_fields.keys())
    filtered_payload = {k: v for k, v in c_data.items() if k in valid_keys}
    
    payload = CompanyFinancialPayload(**filtered_payload)
    pred = await predict_bankruptcy(payload, model_name=model_name)
    comp = await predict_comparison(payload)

    return {
        "company_info": {
            "ticker": fetched.get("ticker"),
            "company_name": fetched.get("company_name"),
            "source": fetched.get("source"),
            "cache_metadata": fetched.get("cache_metadata")
        },
        "financial_data": c_data,
        "prediction": pred,
        "comparison": comp
    }

@app.get("/cache_status")
async def get_api_cache_status():
    from src.api.live_data_service import get_cache_stats
    return {
        "status": "online",
        "cache_strategy": "Strategy A: Cache-Aside with Multi-Level TTL",
        "ttl_policies": {
            "accounting_statements": "90 days (Quarterly 10-Q/10-K)",
            "market_quotes_merton": "24 hours (Daily)",
            "nlp_news_sentiment": "24 hours (Daily with FinBERT)",
            "macro_indicators": "24 hours (Global FRED API)"
        },
        "statistics": get_cache_stats()
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_bankruptcy(
    payload: CompanyFinancialPayload,
    model_name: Optional[str] = Query("voting_classifier", enum=["voting_classifier", "lightgbm", "random_forest", "catboost", "xgboost", "logistic_regression"])
):
    start_time = time.time()
    payload_dict = payload.model_dump()

    target_model = loaded_models.get(model_name) or loaded_models.get("voting_classifier") or loaded_models.get("lightgbm")
    if target_model is None or preprocessor is None:
        raise HTTPException(status_code=503, detail="Modelo o preprocesador de inferencia no cargado.")

    scaled_x = prepare_feature_vector(payload_dict)
    prob = float(target_model.predict_proba(scaled_x)[:, 1][0])

    tau = MODEL_THRESHOLDS.get(model_name, MODEL_THRESHOLDS.get("voting_classifier", 0.158))
    is_bankrupt = prob >= tau

    if prob >= tau:
        category = "HIGH RISK / DISTRESS"
    elif prob >= (tau * 0.4):
        category = "MEDIUM RISK / WATCHLIST"
    else:
        category = "LOW RISK / SOLVENT"

    adverse_notices = generate_adverse_action_notices(
        payload_dict=payload_dict,
        prob=prob,
        tau=tau,
        model_name=model_name or "voting_classifier",
        scaled_x=scaled_x
    )
    altman_z = compute_approximate_altman_z(payload_dict)
    mdd = payload_dict.get("merton_distance_to_default")
    latency = (time.time() - start_time) * 1000.0

    audit_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "model_used": model_name,
        "company_name": payload.company_name,
        "year": payload.year,
        "input_vector": {col: payload_dict.get(col) for col in (preprocessor["feature_names"] if preprocessor else [])},
        "probability": round(prob, 5),
        "threshold": tau,
        "is_predicted_distress": is_bankrupt,
        "risk_category": category,
        "adverse_action_notices": adverse_notices,
        "latency_ms": round(latency, 2),
        "compliance": "EU AI Act Art. 12 (Technical Record-Keeping)"
    }
    audit_logger.info(json.dumps(audit_entry))

    return PredictionResponse(
        company_name=payload.company_name,
        year=payload.year,
        bankruptcy_probability=round(prob, 5),
        is_bankruptcy_predicted=is_bankrupt,
        risk_category=category,
        decision_threshold=tau,
        altman_z_score=round(altman_z, 4),
        merton_distance_to_default=round(mdd, 4) if mdd is not None else None,
        adverse_action_notices=adverse_notices,
        latency_ms=round(latency, 2)
    )

@app.post("/predict_comparison")
async def predict_comparison(payload: CompanyFinancialPayload):
    start_time = time.time()
    payload_dict = payload.model_dump()
    scaled_x = prepare_feature_vector(payload_dict)

    comparisons = {}
    for m_name in ["voting_classifier", "lightgbm", "random_forest", "catboost", "xgboost", "logistic_regression"]:
        m_obj = loaded_models.get(m_name)
        if m_obj is not None:
            try:
                p = float(m_obj.predict_proba(scaled_x)[:, 1][0])
                m_tau = MODEL_THRESHOLDS.get(m_name, 0.150)
                comparisons[m_name] = {
                    "display_name": MODEL_DISPLAY_NAMES.get(m_name, m_name),
                    "probability": round(p, 5),
                    "threshold": m_tau,
                    "prediction": "DISTRESS" if p >= m_tau else "SOLVENT",
                    "risk_percent": round(p * 100, 2)
                }
            except Exception as e:
                comparisons[m_name] = {"error": str(e)}

    sc_res = await evaluate_scorecard(payload)
    total_time = (time.time() - start_time) * 1000.0

    primary_tau = MODEL_THRESHOLDS.get("voting_classifier", 0.158)
    primary_prob = comparisons.get("voting_classifier", {}).get("probability", 0.0)

    return {
        "company_name": payload.company_name,
        "models": comparisons,
        "scorecard": {
            "credit_score": sc_res.credit_score,
            "rating_band": sc_res.rating_band,
            "implied_pd": sc_res.implied_pd
        },
        "altman_z_score": round(compute_approximate_altman_z(payload_dict), 2),
        "adverse_action_notices": generate_adverse_action_notices(
            payload_dict=payload_dict,
            prob=primary_prob,
            tau=primary_tau,
            model_name="voting_classifier",
            scaled_x=scaled_x
        ),
        "latency_ms": round(total_time, 2)
    }

@app.post("/predict_batch", response_model=BatchPredictionResponse)
async def predict_bankruptcy_batch(
    payloads: List[CompanyFinancialPayload],
    model_name: Optional[str] = Query("voting_classifier", enum=["voting_classifier", "lightgbm", "random_forest", "catboost", "xgboost", "logistic_regression"])
):
    start_time = time.time()
    if not payloads:
        return BatchPredictionResponse(total_processed=0, mean_latency_ms=0.0, total_latency_ms=0.0, predictions=[])

    target_model = loaded_models.get(model_name) or loaded_models.get("voting_classifier") or loaded_models.get("lightgbm")
    if target_model is None or preprocessor is None:
        raise HTTPException(status_code=503, detail="Modelo o preprocesador de inferencia no cargado.")

    feature_names = preprocessor["feature_names"]
    defaults = preprocessor["default_medians"]
    base_cpi = defaults.get("macro_inflation", 236.468)
    tau = MODEL_THRESHOLDS.get(model_name, MODEL_THRESHOLDS.get("voting_classifier", 0.158))

    records = []
    payload_dicts = []
    for p in payloads:
        p_dict = p.model_dump()
        payload_dicts.append(p_dict)
        row = []
        for col in feature_names:
            val = p_dict.get(col)
            if val is None or (isinstance(val, float) and np.isnan(val)):
                val = defaults.get(col, 0.0)
            elif col == "macro_inflation":
                val = normalize_macro_inflation(val, base_cpi=base_cpi)
            row.append(float(val))
        records.append(row)

    raw_df = pd.DataFrame(records, columns=feature_names, dtype=np.float64)
    imp_mat = preprocessor["imputer"].transform(raw_df)
    scaled_mat = preprocessor["scaler"].transform(imp_mat)

    probs = target_model.predict_proba(scaled_mat)[:, 1]

    predictions = []
    for i, (p_dict, prob_val) in enumerate(zip(payload_dicts, probs)):
        prob = float(prob_val)
        is_bankrupt = prob >= tau

        if prob >= tau:
            category = "HIGH RISK / DISTRESS"
        elif prob >= (tau * 0.4):
            category = "MEDIUM RISK / WATCHLIST"
        else:
            category = "LOW RISK / SOLVENT"

        notices = generate_adverse_action_notices(
            payload_dict=p_dict,
            prob=prob,
            tau=tau,
            model_name=model_name or "voting_classifier",
            scaled_x=scaled_mat[i:i+1]
        )
        altman_z = compute_approximate_altman_z(p_dict)
        mdd = p_dict.get("merton_distance_to_default")

        predictions.append(
            PredictionResponse(
                company_name=p_dict.get("company_name", "Unknown"),
                year=p_dict.get("year", 2024),
                bankruptcy_probability=round(prob, 5),
                is_bankruptcy_predicted=is_bankrupt,
                risk_category=category,
                decision_threshold=tau,
                altman_z_score=round(altman_z, 4),
                merton_distance_to_default=round(mdd, 4) if mdd is not None else None,
                adverse_action_notices=notices,
                latency_ms=0.0
            )
        )

        # Registro de auditoría individual conforme a EU AI Act Art. 12 (Trazabilidad y registro de vectores)
        batch_audit_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "endpoint": "/predict_batch",
            "model_used": model_name,
            "company_name": p_dict.get("company_name", f"Batch_Item_{i}"),
            "year": p_dict.get("year", 2024),
            "input_vector": {col: p_dict.get(col) for col in feature_names},
            "probability": round(prob, 5),
            "threshold": tau,
            "is_predicted_distress": is_bankrupt,
            "risk_category": category,
            "adverse_action_notices": notices,
            "compliance": "EU AI Act Art. 12 (Technical Record-Keeping)"
        }
        audit_logger.info(json.dumps(batch_audit_entry))

    total_time = (time.time() - start_time) * 1000.0
    mean_lat = total_time / max(len(payloads), 1)

    batch_summary_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "endpoint": "/predict_batch_summary",
        "model_used": model_name,
        "total_processed": len(payloads),
        "mean_latency_ms": round(mean_lat, 3),
        "total_latency_ms": round(total_time, 2),
        "distress_count": sum(1 for p in predictions if p.is_bankruptcy_predicted),
        "compliance": "EU AI Act Art. 12 (Batch Execution Log)"
    }
    audit_logger.info(json.dumps(batch_summary_entry))

    return BatchPredictionResponse(
        total_processed=len(payloads),
        mean_latency_ms=round(mean_lat, 3),
        total_latency_ms=round(total_time, 2),
        predictions=predictions
    )

@app.post("/scorecard", response_model=ScorecardResponse)
async def evaluate_scorecard(payload: CompanyFinancialPayload):
    start_time = time.time()
    payload_dict = payload.model_dump()

    score_val = 650.0
    if scorecard is not None and hasattr(scorecard, "score"):
        try:
            var_names = getattr(scorecard, "variable_names", [])
            df_in = pd.DataFrame([{col: payload_dict.get(col, 0.0) for col in var_names}])
            s = scorecard.score(df_in)
            score_val = float(s[0]) if hasattr(s, "__len__") else float(s)
        except Exception:
            scaled_x = prepare_feature_vector(payload_dict)
            model_ref = loaded_models.get("lightgbm") or loaded_models.get("xgboost")
            prob = float(model_ref.predict_proba(scaled_x)[:, 1][0]) if model_ref else 0.05
            score_val = 850.0 - (prob * 550.0)
    else:
        scaled_x = prepare_feature_vector(payload_dict)
        model_ref = loaded_models.get("lightgbm") or loaded_models.get("xgboost")
        prob = float(model_ref.predict_proba(scaled_x)[:, 1][0]) if model_ref else 0.05
        score_val = 850.0 - (prob * 550.0)

    score_val = max(300.0, min(850.0, score_val))

    if score_val >= 780:
        band = "AAA"
    elif score_val >= 720:
        band = "AA"
    elif score_val >= 660:
        band = "A"
    elif score_val >= 600:
        band = "BBB"
    elif score_val >= 540:
        band = "BB"
    elif score_val >= 480:
        band = "B"
    elif score_val >= 400:
        band = "CCC"
    else:
        band = "D"

    implied_pd = max(0.0001, min(0.9999, (850.0 - score_val) / 550.0))
    latency = (time.time() - start_time) * 1000.0

    return ScorecardResponse(
        company_name=payload.company_name,
        year=payload.year,
        credit_score=round(score_val, 1),
        rating_band=band,
        implied_pd=round(implied_pd, 4),
        latency_ms=round(latency, 2)
    )

DASHBOARD_HTML_PATH = "src/api/static/dashboard.html"

@app.get("/", response_class=HTMLResponse)
@app.get("/ui", response_class=HTMLResponse)
async def serve_dashboard():
    # Resolver ruta absoluta a dashboard.html independientemente del CWD
    candidates = [
        Path(__file__).resolve().parent / "static" / "dashboard.html",
        Path(__file__).resolve().parent.parent / "api" / "static" / "dashboard.html",
        Path(DASHBOARD_HTML_PATH)
    ]
    for p in candidates:
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                return HTMLResponse(
                    content=f.read(),
                    headers={
                        "Cache-Control": "no-cache, no-store, must-revalidate, max-age=0",
                        "Pragma": "no-cache",
                        "Expires": "0"
                    }
                )
    return HTMLResponse(content="<h1>Dashboard no encontrado. Por favor genere src/api/static/dashboard.html</h1>", status_code=404)