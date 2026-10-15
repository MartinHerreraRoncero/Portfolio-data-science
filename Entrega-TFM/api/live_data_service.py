"""
Servicio de Datos en Vivo y Gestión de Caché Inteligente (Cache-Aside con TTL).
Proyecto TFM: Predicción de Insolvencia en Empresas del S&P 500.
Entregable API - Cumplimiento normativo y resolución dinámica relativa a ENTREGABLE_DIR.

Estrategia Multinivel por Volatilidad:
1. Macroeconomía (FRED API): Actualización diaria global (Fed Funds, CPI, Desempleo, PIB).
2. Mercado (yfinance Fast Info): Actualización diaria de cotización, capitalización y recálculo de Merton DD.
3. Noticias y Sentimiento NLP (yfinance News + FinBERT): Análisis de titulares recientes con decaimiento temporal.
4. Estados Contables (SEC EDGAR / DuckDB / yfinance): Caché trimestral (90 días) con soporte para forzar refresco.
"""

import os
import re
import sys
import json
import time
import sqlite3
import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger("live_data_service")
logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Resolución Dinámica de Rutas Relativas a ENTREGABLE_DIR
# ---------------------------------------------------------------------------
CURRENT_FILE = Path(__file__).resolve()
API_DIR = CURRENT_FILE.parent
ENTREGABLE_DIR = API_DIR.parent

if str(ENTREGABLE_DIR) not in sys.path:
    sys.path.insert(0, str(ENTREGABLE_DIR))
if str(API_DIR) not in sys.path:
    sys.path.insert(0, str(API_DIR))

# Configuración de Rutas y TTL
CACHE_DIR = ENTREGABLE_DIR / "data" / "processed" / "live_cache"
SQLITE_DB_PATH = CACHE_DIR / "live_data_cache.sqlite"
PARQUET_SEC_PATH = ENTREGABLE_DIR / "data" / "processed" / "sec_dataset" / "v2_sec_financials_pivoted_clean.parquet"

# TTL en segundos
TTL_ACCOUNTING_SECONDS = 90 * 86400   # 90 días (trimestral)
TTL_MARKET_SECONDS = 24 * 3600        # 24 horas (diario)
TTL_NEWS_SECONDS = 24 * 3600          # 24 horas (diario)
TTL_MACRO_SECONDS = 24 * 3600         # 24 horas (diario global)

FRED_API_KEY = os.getenv("FED_API_KEY", "3db6890dafdd5256345433bc73dfebe1")

# Singletons para Modelos NLP y GMM
_finbert_tokenizer = None
_finbert_model = None
_finbert_device = None
_macro_gmm_detector = None
_corp_gmm_detector = None

# Defaults canónicos de las 21 variables óptimas según preprocessor_pipeline.joblib
CANONICAL_21_DEFAULTS = {
    'tag_NetIncomeLoss': 0.0,
    'tag_WorkingCapital': 16498000.0,
    'merton_distance_to_default': 4.534358,
    'tag_Assets': 243633500.0,
    'pub_lag_days': 43.0,
    'stock_price_close': 45.0,
    'tag_RetainedEarningsAccumulatedDeficit': -451008.0,
    'corp_archetype_prob_1': 0.0,
    'macro_inflation': 236.468,
    'market_cap': 100000000.0,
    'tag_EntityCommonStockSharesOutstanding': 34679114.5,
    'tag_CommonStockSharesAuthorized': 130000000.0,
    'macro_real_gdp_growth_yoy': 2.403336,
    'news_sentiment_range_6m': 0.071805,
    'tag_CommonStockParOrStatedValuePerShare': 0.01,
    'tag_CommonStockValue': 152000.0,
    'macro_interest_rate': 0.142581,
    'market_news_sentiment_mean': 0.052079,
    'corp_archetype_prob_2': 0.000232,
    'macro_regime_expansion_prob': 5.533757e-05,
    'news_sentiment_avg': 0.052079
}


def get_db_connection() -> sqlite3.Connection:
    """Crea o conecta a la base de datos SQLite con WAL mode para concurrencia."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(SQLITE_DB_PATH), timeout=10.0)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL;")
    return con


def init_cache_database():
    """Inicializa las tablas del esquema de caché en SQLite con soporte para las 21 variables."""
    with get_db_connection() as con:
        con.executescript("""
            CREATE TABLE IF NOT EXISTS global_macro_cache (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                interest_rate REAL,
                inflation REAL,
                unemployment_rate REAL,
                gdp_growth REAL,
                regime_expansion_prob REAL,
                source TEXT,
                updated_at REAL
            );

            CREATE TABLE IF NOT EXISTS company_accounting_cache (
                ticker TEXT PRIMARY KEY,
                company_name TEXT,
                filed_year INTEGER,
                data_json TEXT,
                source TEXT,
                updated_at REAL
            );

            CREATE TABLE IF NOT EXISTS company_market_cache (
                ticker TEXT PRIMARY KEY,
                stock_price_close REAL,
                market_cap REAL,
                shares_outstanding REAL,
                merton_distance_to_default REAL,
                source TEXT,
                updated_at REAL
            );

            CREATE TABLE IF NOT EXISTS company_news_cache (
                ticker TEXT PRIMARY KEY,
                nlp_sentiment_decayed_30d REAL,
                news_sentiment_range_6m REAL,
                news_sentiment_max REAL,
                market_news_sentiment_mean REAL,
                news_sentiment_avg REAL,
                article_count INTEGER,
                source TEXT,
                updated_at REAL
            );
        """)
        # Migraciones seguras para bases de datos SQLite preexistentes
        try:
            con.execute("ALTER TABLE global_macro_cache ADD COLUMN regime_expansion_prob REAL;")
        except Exception:
            pass
        try:
            con.execute("ALTER TABLE company_news_cache ADD COLUMN news_sentiment_avg REAL;")
        except Exception:
            pass
        con.commit()


# Inicializar base de datos al importar el módulo
init_cache_database()


def get_macro_gmm_detector():
    """Carga perezosa del detector GMM de regímenes macroeconómicos."""
    global _macro_gmm_detector
    if _macro_gmm_detector is None:
        try:
            import __main__
            from src.features.macro_gmm_regimes import MacroGMMRegimeDetector
            setattr(__main__, 'MacroGMMRegimeDetector', MacroGMMRegimeDetector)
            import joblib
            candidates = [
                ENTREGABLE_DIR / "models" / "macro_gmm_regimes_model.pkl",
                Path("models/macro_gmm_regimes_model.pkl"),
                ENTREGABLE_DIR.parent / "models" / "macro_gmm_regimes_model.pkl"
            ]
            for p in candidates:
                if p.exists():
                    _macro_gmm_detector = joblib.load(str(p))
                    logger.info(f"[Macro GMM] Detector cargado exitosamente desde {p}")
                    break
        except Exception as e:
            logger.warning(f"[Macro GMM] Fallback heurístico activado: {e}")
            _macro_gmm_detector = False
    return _macro_gmm_detector if _macro_gmm_detector is not False else None


def get_corp_gmm_detector():
    """Carga perezosa del detector GMM de arquetipos corporativos."""
    global _corp_gmm_detector
    if _corp_gmm_detector is None:
        try:
            import __main__
            from src.features.macro_gmm_regimes import CorporateArchetypeGMMDetector
            setattr(__main__, 'CorporateArchetypeGMMDetector', CorporateArchetypeGMMDetector)
            import joblib
            candidates = [
                ENTREGABLE_DIR / "models" / "corp_archetypes_gmm_model.pkl",
                Path("models/corp_archetypes_gmm_model.pkl"),
                ENTREGABLE_DIR.parent / "models" / "corp_archetypes_gmm_model.pkl"
            ]
            for p in candidates:
                if p.exists():
                    _corp_gmm_detector = joblib.load(str(p))
                    logger.info(f"[Corp GMM] Detector cargado exitosamente desde {p}")
                    break
        except Exception as e:
            logger.warning(f"[Corp GMM] Fallback heurístico activado: {e}")
            _corp_gmm_detector = False
    return _corp_gmm_detector if _corp_gmm_detector is not False else None


def get_finbert():
    """Carga perezosa (lazy load) en memoria del modelo FinBERT para inferencia NLP."""
    global _finbert_tokenizer, _finbert_model, _finbert_device
    if _finbert_model is None:
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForSequenceClassification

            device = "cuda" if torch.cuda.is_available() else "cpu"
            model_name = "ProsusAI/finbert"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
            model.eval()

            _finbert_tokenizer = tokenizer
            _finbert_model = model
            _finbert_device = device
            logger.info(f"[FinBERT] Modelo cargado exitosamente en dispositivo: {device}")
        except Exception as e:
            logger.warning(f"[FinBERT] No se pudo cargar FinBERT: {e}. Se utilizará fallback heurístico.")
            _finbert_model = False
    return _finbert_tokenizer, _finbert_model, _finbert_device


# ==============================================================================
# 1. CAPA MACROECONÓMICA (FRED API)
# ==============================================================================

def fetch_live_macro(force_refresh: bool = False, defaults: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Obtiene las variables macroeconómicas actuales desde FRED API con caché de 24h
    y computa la probabilidad de régimen macroeconómico de expansión vía Macro GMM.
    Series:
    - DFF: Federal Funds Rate (tipo de interés interbancario de la Fed)
    - CPIAUCSL: Consumer Price Index for All Urban Consumers (nivel de índice de precios)
    - UNRATE: Civilian Unemployment Rate (tasa de desempleo %)
    - A191RL1Q225SBEA: Real Gross Domestic Product (crecimiento anualizado del PIB real %)
    """
    defaults = defaults or CANONICAL_21_DEFAULTS
    now_ts = time.time()
    
    # 1. Comprobar caché en SQLite
    if not force_refresh:
        try:
            with get_db_connection() as con:
                row = con.execute("SELECT * FROM global_macro_cache WHERE id = 1").fetchone()
                if row and (now_ts - row["updated_at"] < TTL_MACRO_SECONDS):
                    reg_prob = row["regime_expansion_prob"] if "regime_expansion_prob" in row.keys() and row["regime_expansion_prob"] is not None else defaults.get("macro_regime_expansion_prob", CANONICAL_21_DEFAULTS["macro_regime_expansion_prob"])
                    return {
                        "macro_interest_rate": float(row["interest_rate"]),
                        "macro_inflation": float(row["inflation"]),
                        "macro_unemployment_rate": float(row["unemployment_rate"]),
                        "macro_real_gdp_growth_yoy": float(row["gdp_growth"]),
                        "macro_regime_expansion_prob": float(reg_prob),
                        "_macro_source": row["source"],
                        "_macro_cached": True
                    }
        except Exception as e:
            logger.warning(f"[Macro Cache] Error leyendo SQLite: {e}")

    # 2. Consultar FRED API
    series_map = {
        "DFF": "interest_rate",
        "CPIAUCSL": "inflation",
        "UNRATE": "unemployment_rate",
        "A191RL1Q225SBEA": "gdp_growth"
    }
    macro_vals = {
        "interest_rate": 3.63,
        "inflation": 332.8,
        "unemployment_rate": 4.1,
        "gdp_growth": 1.5
    }
    success_count = 0

    for sid, k in series_map.items():
        try:
            url = f"https://api.stlouisfed.org/fred/series/observations?series_id={sid}&api_key={FRED_API_KEY}&file_type=json&sort_order=desc&limit=1"
            res = requests.get(url, timeout=4.0)
            if res.status_code == 200:
                obs = res.json().get("observations", [])
                if obs and obs[0].get("value") not in (None, ".", ""):
                    macro_vals[k] = float(obs[0]["value"])
                    success_count += 1
        except Exception as e:
            logger.warning(f"[FRED API] No se pudo obtener serie {sid}: {e}")

    # 3. Calcular macro_regime_expansion_prob con detector GMM si está disponible
    regime_prob = float(defaults.get("macro_regime_expansion_prob", CANONICAL_21_DEFAULTS["macro_regime_expansion_prob"]))
    detector = get_macro_gmm_detector()
    if detector is not None:
        try:
            macro_input = {
                "macro_interest_rate": macro_vals["interest_rate"],
                "macro_yield_curve": 0.50,
                "macro_real_gdp_growth_yoy": macro_vals["gdp_growth"],
                "macro_inflation": macro_vals["inflation"],
                "macro_unemployment_rate": macro_vals["unemployment_rate"]
            }
            df_res = detector.transform(pd.DataFrame([macro_input]))
            if "macro_regime_expansion_prob" in df_res.columns:
                regime_prob = float(df_res["macro_regime_expansion_prob"].iloc[0])
        except Exception as e:
            logger.warning(f"[Macro GMM] Error al inferir probabilidad de régimen: {e}")

    # 4. Guardar en SQLite
    source_label = f"FRED Live API ({datetime.now(timezone.utc).strftime('%Y-%m-%d')})" if success_count > 0 else "FRED Cache Fallback"
    try:
        with get_db_connection() as con:
            con.execute("""
                INSERT OR REPLACE INTO global_macro_cache (id, interest_rate, inflation, unemployment_rate, gdp_growth, regime_expansion_prob, source, updated_at)
                VALUES (1, ?, ?, ?, ?, ?, ?, ?)
            """, (
                macro_vals["interest_rate"],
                macro_vals["inflation"],
                macro_vals["unemployment_rate"],
                macro_vals["gdp_growth"],
                regime_prob,
                source_label,
                now_ts
            ))
            con.commit()
    except Exception as e:
        logger.warning(f"[Macro Cache] Error escribiendo SQLite: {e}")

    return {
        "macro_interest_rate": macro_vals["interest_rate"],
        "macro_inflation": macro_vals["inflation"],
        "macro_unemployment_rate": macro_vals["unemployment_rate"],
        "macro_real_gdp_growth_yoy": macro_vals["gdp_growth"],
        "macro_regime_expansion_prob": regime_prob,
        "_macro_source": source_label,
        "_macro_cached": False
    }


# ==============================================================================
# 2. CAPA DE MERCADO (yfinance Fast Info + Recálculo Merton DD)
# ==============================================================================

def calculate_merton_dd(market_cap: float, total_liabilities: float, sigma_asset: float = 0.25) -> float:
    """
    Calcula la Distancia a la Quiebra Estructural de Merton:
    DD = (ln(V / D) + (mu - 0.5 * sigma_asset^2)) / sigma_asset
    donde V = Market Cap + Pasivo Total (Debt), D = Pasivo Total.
    """
    debt = max(float(total_liabilities or 0.0), 1000.0)
    v_firm = max(float(market_cap or 0.0), 1000.0) + debt
    v_over_d = max(0.01, v_firm / debt)
    mu = 0.05
    dd = (np.log(v_over_d) + (mu - 0.5 * (sigma_asset ** 2))) / sigma_asset
    return float(round(max(-2.0, min(10.0, dd)), 4))


def fetch_live_market(
    ticker: str,
    total_assets: float = 0.0,
    total_equity: float = 0.0,
    total_liabilities: Optional[float] = None,
    force_refresh: bool = False
) -> Dict[str, Any]:
    """
    Obtiene los datos bursátiles más recientes (precio, market cap, acciones) y recalcula
    la distancia estructural de Merton con los pasivos más recientes. TTL: 24 horas.
    """
    clean_ticker = ticker.strip().upper()
    now_ts = time.time()

    # 1. Comprobar caché de mercado en SQLite
    if not force_refresh:
        try:
            with get_db_connection() as con:
                row = con.execute("SELECT * FROM company_market_cache WHERE ticker = ?", (clean_ticker,)).fetchone()
                if row and (now_ts - row["updated_at"] < TTL_MARKET_SECONDS):
                    return {
                        "stock_price_close": float(row["stock_price_close"]),
                        "market_cap": float(row["market_cap"]),
                        "tag_EntityCommonStockSharesOutstanding": float(row["shares_outstanding"]),
                        "merton_distance_to_default": float(row["merton_distance_to_default"]),
                        "_market_source": row["source"],
                        "_market_cached": True
                    }
        except Exception as e:
            logger.warning(f"[Market Cache] Error leyendo SQLite para {clean_ticker}: {e}")

    # 2. Consultar yfinance Fast Info
    price = 50.0
    mcap = 50000000.0
    shares = 1000000.0
    got_live = False

    try:
        import yfinance as yf
        t = yf.Ticker(clean_ticker)
        fi = getattr(t, "fast_info", None)
        if fi:
            p_val = getattr(fi, "last_price", None) or getattr(fi, "regular_market_previous_close", None)
            mc_val = getattr(fi, "market_cap", None)
            sh_val = getattr(fi, "shares", None)
            if p_val and p_val > 0:
                price = float(p_val)
                got_live = True
            if mc_val and mc_val > 0:
                mcap = float(mc_val)
                got_live = True
            if sh_val and sh_val > 0:
                shares = float(sh_val)
            elif mcap > 0 and price > 0:
                shares = mcap / price

        if not got_live:
            info = getattr(t, "info", {}) or {}
            price = float(info.get("currentPrice") or info.get("regularMarketPrice") or price)
            mcap = float(info.get("marketCap") or mcap)
            shares = float(info.get("sharesOutstanding") or (mcap / max(price, 1.0)))
            if "marketCap" in info or "currentPrice" in info:
                got_live = True
    except Exception as e:
        logger.warning(f"[Market Live] Error consultando yfinance para {clean_ticker}: {e}")

    # 3. Recalcular Pasivo Total y Merton Distance to Default
    if total_liabilities is not None and total_liabilities > 0:
        debt = total_liabilities
    elif total_assets > 0 and total_equity > 0:
        debt = max(0.0, total_assets - total_equity)
    elif total_assets > 0:
        debt = total_assets * 0.65
    else:
        debt = mcap * 0.5

    merton_dd = calculate_merton_dd(mcap, debt)
    source_label = f"Yahoo Finance Live ({datetime.now(timezone.utc).strftime('%Y-%m-%d')})" if got_live else "Market Fallback Baseline"

    # 4. Guardar en SQLite
    try:
        with get_db_connection() as con:
            con.execute("""
                INSERT OR REPLACE INTO company_market_cache (ticker, stock_price_close, market_cap, shares_outstanding, merton_distance_to_default, source, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (clean_ticker, price, mcap, shares, merton_dd, source_label, now_ts))
            con.commit()
    except Exception as e:
        logger.warning(f"[Market Cache] Error guardando en SQLite para {clean_ticker}: {e}")

    return {
        "stock_price_close": round(price, 2),
        "market_cap": round(mcap, 2),
        "tag_EntityCommonStockSharesOutstanding": round(shares, 0),
        "merton_distance_to_default": merton_dd,
        "_market_source": source_label,
        "_market_cached": False
    }


# ==============================================================================
# 3. CAPA DE NOTICIAS Y SENTIMIENTO NLP (yfinance News + FinBERT)
# ==============================================================================

def fetch_live_news_sentiment(ticker: str, force_refresh: bool = False, defaults: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Descarga titulares financieros recientes de la empresa y calcula el índice de sentimiento
    FinBERT con ponderación temporal decaída y media agregada (news_sentiment_avg). TTL: 24 horas.
    """
    defaults = defaults or CANONICAL_21_DEFAULTS
    clean_ticker = ticker.strip().upper()
    now_ts = time.time()

    # 1. Comprobar caché en SQLite
    if not force_refresh:
        try:
            with get_db_connection() as con:
                row = con.execute("SELECT * FROM company_news_cache WHERE ticker = ?", (clean_ticker,)).fetchone()
                if row and (now_ts - row["updated_at"] < TTL_NEWS_SECONDS):
                    avg_sentiment = row["news_sentiment_avg"] if "news_sentiment_avg" in row.keys() and row["news_sentiment_avg"] is not None else float(row["nlp_sentiment_decayed_30d"])
                    return {
                        "nlp_sentiment_decayed_30d": float(row["nlp_sentiment_decayed_30d"]),
                        "news_sentiment_range_6m": float(row["news_sentiment_range_6m"]),
                        "news_sentiment_max": float(row["news_sentiment_max"]),
                        "market_news_sentiment_mean": float(row["market_news_sentiment_mean"]),
                        "news_sentiment_avg": float(avg_sentiment),
                        "_news_count": int(row["article_count"]),
                        "_news_source": row["source"],
                        "_news_cached": True
                    }
        except Exception as e:
            logger.warning(f"[News Cache] Error leyendo SQLite para {clean_ticker}: {e}")

    # 2. Descargar titulares de noticias
    articles = []
    try:
        import yfinance as yf
        t = yf.Ticker(clean_ticker)
        raw_news = getattr(t, "news", []) or []
        for item in raw_news:
            content = item.get("content", item)
            title = content.get("title", item.get("title", ""))
            summary = content.get("summary", "")
            pub_date = content.get("pubDate", item.get("providerPublishTime", ""))
            if title:
                text = f"{title}. {summary}".strip() if summary and summary.lower() != title.lower() else title
                articles.append({"text": text, "pub_date": pub_date})
    except Exception as e:
        logger.warning(f"[News Live] Error descargando noticias para {clean_ticker}: {e}")

    # 3. Analizar sentimiento con FinBERT
    sentiment_scores = []
    if articles:
        tok, model, device = get_finbert()
        if model and tok:
            try:
                import torch
                texts = [a["text"] for a in articles[:15]]
                inputs = tok(texts, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
                with torch.no_grad():
                    outputs = model(**inputs)
                    probs = torch.nn.functional.softmax(outputs.logits, dim=-1).cpu().numpy()
                # ProsusAI/finbert clases: 0=positive, 1=negative, 2=neutral
                # score = pos - neg (en rango [-1.0, 1.0])
                for p in probs:
                    s_idx = float(p[0] - p[1])
                    sentiment_scores.append(s_idx)
            except Exception as e:
                logger.warning(f"[FinBERT] Error en inferencia: {e}")

    # 4. Calcular Métricas Agregadas
    if sentiment_scores:
        decay_weights = np.exp(-0.05 * np.arange(len(sentiment_scores)))
        decayed_30d = float(np.sum(np.array(sentiment_scores) * decay_weights) / np.sum(decay_weights))
        s_max = float(np.max(sentiment_scores))
        s_range = float(np.max(sentiment_scores) - np.min(sentiment_scores))
        s_avg = float(np.mean(sentiment_scores))
        mkt_mean = 0.052079
        source_label = f"FinBERT Live ({len(sentiment_scores)} titulares, {datetime.now(timezone.utc).strftime('%Y-%m-%d')})"
    else:
        decayed_30d = float(defaults.get("news_sentiment_avg", CANONICAL_21_DEFAULTS["news_sentiment_avg"]))
        s_max = 0.20
        s_range = float(defaults.get("news_sentiment_range_6m", CANONICAL_21_DEFAULTS["news_sentiment_range_6m"]))
        s_avg = float(defaults.get("news_sentiment_avg", CANONICAL_21_DEFAULTS["news_sentiment_avg"]))
        mkt_mean = float(defaults.get("market_news_sentiment_mean", CANONICAL_21_DEFAULTS["market_news_sentiment_mean"]))
        source_label = "FinBERT Historical Prior Baseline"

    # 5. Guardar en SQLite
    try:
        with get_db_connection() as con:
            con.execute("""
                INSERT OR REPLACE INTO company_news_cache (ticker, nlp_sentiment_decayed_30d, news_sentiment_range_6m, news_sentiment_max, market_news_sentiment_mean, news_sentiment_avg, article_count, source, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (clean_ticker, round(decayed_30d, 4), round(s_range, 4), round(s_max, 4), round(mkt_mean, 4), round(s_avg, 4), len(sentiment_scores), source_label, now_ts))
            con.commit()
    except Exception as e:
        logger.warning(f"[News Cache] Error guardando SQLite para {clean_ticker}: {e}")

    return {
        "nlp_sentiment_decayed_30d": round(decayed_30d, 4),
        "news_sentiment_range_6m": round(s_range, 4),
        "news_sentiment_max": round(s_max, 4),
        "market_news_sentiment_mean": round(mkt_mean, 4),
        "news_sentiment_avg": round(s_avg, 4),
        "_news_count": len(sentiment_scores),
        "_news_source": source_label,
        "_news_cached": False
    }


# ==============================================================================
# 4. CAPA CONTABLE Y ENSAMBLADOR MAESTRO DE 26 VARIABLES
# ==============================================================================

def fetch_accounting_data(
    ticker_or_name: str,
    clean_ticker: str,
    clean_name: str,
    force_refresh: bool = False,
    defaults: Optional[Dict[str, Any]] = None
) -> Tuple[bool, str, str, int, Dict[str, Any]]:
    """
    Obtiene los balances contables de la empresa comprobando SQLite Cache (90d),
    DuckDB SEC Dataset y fallback a yfinance Balance Sheet.
    """
    defaults = defaults or {}
    now_ts = time.time()

    # 1. Comprobar caché contable en SQLite
    if not force_refresh:
        try:
            with get_db_connection() as con:
                row = con.execute("SELECT * FROM company_accounting_cache WHERE ticker = ?", (clean_ticker,)).fetchone()
                if row and (now_ts - row["updated_at"] < TTL_ACCOUNTING_SECONDS):
                    data = json.loads(row["data_json"])
                    return True, row["source"], row["company_name"], row["filed_year"], data
        except Exception as e:
            logger.warning(f"[Accounting Cache] Error leyendo SQLite para {clean_ticker}: {e}")

    # 2. Si force_refresh=True o no está en SQLite, intentar yfinance Live Balance Sheet primero si es ticker
    if force_refresh and len(clean_ticker) <= 5:
        live_acc = _try_yfinance_accounting(clean_ticker)
        if live_acc is not None:
            c_name, f_year, acc_dict = live_acc
            _save_accounting_cache(clean_ticker, c_name, f_year, acc_dict, f"Yahoo Finance Live 10-Q/10-K ({f_year})")
            return True, f"Yahoo Finance Live 10-Q/10-K ({f_year})", c_name, f_year, acc_dict

    # 3. Consultar DuckDB Parquet Dataset
    duck_res = _try_duckdb_accounting(clean_ticker, clean_name, defaults)
    if duck_res is not None:
        c_name, f_year, acc_dict = duck_res
        source_label = f"SEC EDGAR Dataset ({f_year})"
        _save_accounting_cache(clean_ticker, c_name, f_year, acc_dict, source_label)
        return True, source_label, c_name, f_year, acc_dict

    # 4. Si no estaba en DuckDB, intentar yfinance
    live_acc = _try_yfinance_accounting(clean_ticker)
    if live_acc is not None:
        c_name, f_year, acc_dict = live_acc
        source_label = f"Yahoo Finance Live ({f_year})"
        _save_accounting_cache(clean_ticker, c_name, f_year, acc_dict, source_label)
        return True, source_label, c_name, f_year, acc_dict

    return False, "", clean_name or clean_ticker, 2024, {}


def _save_accounting_cache(ticker: str, company_name: str, filed_year: int, data: dict, source: str):
    try:
        with get_db_connection() as con:
            con.execute("""
                INSERT OR REPLACE INTO company_accounting_cache (ticker, company_name, filed_year, data_json, source, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (ticker, company_name, filed_year, json.dumps(data, default=str), source, time.time()))
            con.commit()
    except Exception as e:
        logger.warning(f"[Accounting Cache] Error guardando en SQLite: {e}")


def _try_duckdb_accounting(clean_ticker: str, clean_name: str, defaults: dict) -> Optional[Tuple[str, int, dict]]:
    parquet_path = PARQUET_SEC_PATH
    if not parquet_path.exists():
        fallback = ENTREGABLE_DIR.parent / "data" / "processed" / "sec_dataset" / "v2_sec_financials_pivoted_clean.parquet"
        if fallback.exists():
            parquet_path = fallback
        else:
            return None
    try:
        import duckdb
        con = duckdb.connect()
        name_variants = [
            f"'{clean_name}'",
            f"'{clean_name} CO'",
            f"'{clean_name} CORP'",
            f"'{clean_name} INC'",
            f"'{clean_name} LTD'",
            f"'{clean_name} COMPANY'",
            f"'{clean_name} CORPORATION'"
        ]
        variants_sql = ", ".join(name_variants)
        parquet_posix = parquet_path.resolve().as_posix()
        sql = f"""
            SELECT *,
                CASE 
                    WHEN UPPER(TRIM(ticker)) = '{clean_ticker}' THEN 1
                    WHEN UPPER(TRIM(company_name)) = '{clean_name}' THEN 2
                    WHEN UPPER(TRIM(company_name)) IN ({variants_sql}) THEN 3
                    ELSE 4
                END as match_priority
            FROM '{parquet_posix}'
            WHERE UPPER(TRIM(ticker)) = '{clean_ticker}' 
               OR UPPER(TRIM(company_name)) IN ({variants_sql})
            ORDER BY match_priority ASC, filed_year DESC
            LIMIT 1;
        """
        df = con.execute(sql).df()
        con.close()
        if not df.empty:
            raw_rec = df.iloc[0].to_dict()
            clean_rec = {}
            for k, v in raw_rec.items():
                if isinstance(v, (float, np.floating)):
                    clean_rec[k] = float(defaults.get(k, 0.0)) if (np.isnan(v) or np.isinf(v)) else float(v)
                else:
                    clean_rec[k] = v
            c_name = clean_rec.get("company_name") or clean_name
            f_year = int(clean_rec.get("filed_year") or 2024)
            return c_name, f_year, clean_rec
    except Exception as e:
        logger.warning(f"[DuckDB] Error en consulta: {e}")
    return None


def _try_yfinance_accounting(clean_ticker: str) -> Optional[Tuple[str, int, dict]]:
    try:
        import yfinance as yf
        t = yf.Ticker(clean_ticker)
        bs = t.balance_sheet
        inc = t.financials
        info = getattr(t, "info", {}) or {}

        if bs is None or bs.empty:
            return None

        c_name = info.get("longName") or info.get("shortName") or clean_ticker

        def get_val(df_obj, row_candidates):
            if df_obj is None or df_obj.empty:
                return 0.0
            for rc in row_candidates:
                if rc in df_obj.index:
                    s = df_obj.loc[rc]
                    if hasattr(s, "iloc") and len(s) > 0:
                        v = s.iloc[0]
                        if not pd.isna(v):
                            return float(v)
            return 0.0

        assets = get_val(bs, ["Total Assets"])
        ca = get_val(bs, ["Current Assets"])
        wc = get_val(bs, ["Working Capital"])
        if wc == 0.0 and ca > 0:
            cl = get_val(bs, ["Current Liabilities"])
            wc = ca - cl
        cash = get_val(bs, ["Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments"])
        equity = get_val(bs, ["Stockholders Equity", "Total Equity Gross Minority Interest"])
        re_val = get_val(bs, ["Retained Earnings"])

        rev = get_val(inc, ["Total Revenue", "Operating Revenue"])
        ni = get_val(inc, ["Net Income", "Net Income Common Stockholders"])
        interest = get_val(inc, ["Interest Expense", "Interest Expense Non Operating"])

        year_val = datetime.now(timezone.utc).year
        if hasattr(bs.columns, "__iter__") and len(bs.columns) > 0:
            try:
                first_col = bs.columns[0]
                if hasattr(first_col, "year"):
                    year_val = first_col.year
            except Exception:
                pass

        shares = float(info.get("sharesOutstanding") or 1000000.0)

        acc_dict = {
            "tag_Assets": assets,
            "tag_AssetsCurrent": ca,
            "tag_LiabilitiesCurrent": cl if 'cl' in locals() and cl > 0 else max(0.0, assets - equity),
            "tag_WorkingCapital": wc,
            "tag_CashAndCashEquivalentsAtCarryingValue": cash,
            "tag_StockholdersEquity": equity,
            "tag_RetainedEarningsAccumulatedDeficit": re_val,
            "tag_Revenues_combined": rev,
            "tag_NetIncomeLoss": ni,
            "tag_InterestExpense": interest,
            "tag_RevenueFromContractWithCustomerExcludingAssessedTax": rev,
            "tag_EntityCommonStockSharesOutstanding": shares,
            "tag_CommonStockSharesAuthorized": shares * 2.0,
            "tag_CommonStockValue": shares * 0.01,
            "tag_CommonStockParOrStatedValuePerShare": 0.01,
            "pub_lag_days": 43.0
        }
        return c_name, year_val, acc_dict
    except Exception as e:
        logger.warning(f"[yfinance Accounting] Error: {e}")
    return None


def compute_corp_archetypes(acc_dict: dict, defaults: Optional[dict] = None) -> Tuple[float, float]:
    """
    Computa las probabilidades de pertenencia a los arquetipos corporativos 1 y 2
    utilizando el modelo CorporateArchetypeGMMDetector ($K=4$), o defaults canónicos.
    """
    defaults = defaults or CANONICAL_21_DEFAULTS
    p1_def = float(defaults.get("corp_archetype_prob_1", CANONICAL_21_DEFAULTS["corp_archetype_prob_1"]))
    p2_def = float(defaults.get("corp_archetype_prob_2", CANONICAL_21_DEFAULTS["corp_archetype_prob_2"]))

    detector = get_corp_gmm_detector()
    if detector is not None:
        try:
            req_cols = getattr(detector, "corp_cols", ['tag_WorkingCapital', 'tag_NetIncomeLoss', 'tag_AssetsCurrent', 'tag_LiabilitiesCurrent'])
            row = {}
            for col in req_cols:
                v = acc_dict.get(col)
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    v = defaults.get(col, 0.0)
                row[col] = float(v)
            df_in = pd.DataFrame([row])
            df_out = detector.transform(df_in)
            if "corp_archetype_prob_1" in df_out.columns:
                p1_def = float(df_out["corp_archetype_prob_1"].iloc[0])
            if "corp_archetype_prob_2" in df_out.columns:
                p2_def = float(df_out["corp_archetype_prob_2"].iloc[0])
        except Exception as e:
            logger.warning(f"[Corp GMM] Error al inferir arquetipos corporativos: {e}")
    return round(p1_def, 6), round(p2_def, 6)


def get_freshest_company_data(
    ticker_or_name: str,
    force_refresh: bool = False,
    defaults: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Orquestador de Frescura Total (Estrategia A):
    Combina:
    - Balances contables (Caché 90d / DuckDB / yfinance)
    - Mercado diario (Precio, Capitalización, Merton DD actualizado a fecha de hoy)
    - Sentimiento de noticias NLP diario (FinBERT analizando titulares recientes)
    - Macroeconomía global diaria (FRED API y Macro GMM)
    - Segmentación GMM de Arquetipos Corporativos
    Ensambla el vector completo y canónico de 21 Variables Óptimas.
    """
    defaults = defaults or CANONICAL_21_DEFAULTS
    q = ticker_or_name.strip().upper()
    clean_ticker = re.sub(r'[^a-zA-Z0-9\.\-]', '', q)
    clean_name = q.replace("'", "''")

    # 1. Obtener Balances Contables
    found, acc_source, c_name, f_year, acc_dict = fetch_accounting_data(
        ticker_or_name=q,
        clean_ticker=clean_ticker,
        clean_name=clean_name,
        force_refresh=force_refresh,
        defaults=defaults
    )

    if not found:
        return {
            "found": False,
            "ticker": clean_ticker,
            "company_name": None,
            "error": f"Empresa o ticker '{ticker_or_name}' no encontrado en los registros oficiales.",
            "company_data": None
        }

    # 2. Obtener Datos de Mercado en Vivo (Cotización, Capitalización y Merton DD con pasivos actuales)
    assets = acc_dict.get("tag_Assets", 0.0)
    equity = acc_dict.get("tag_StockholdersEquity", 0.0)
    market_data = fetch_live_market(
        ticker=clean_ticker,
        total_assets=assets,
        total_equity=equity,
        force_refresh=force_refresh
    )

    # 3. Obtener Sentimiento de Noticias en Vivo con FinBERT
    news_data = fetch_live_news_sentiment(
        ticker=clean_ticker,
        force_refresh=force_refresh,
        defaults=defaults
    )

    # 4. Obtener Variables Macroeconómicas Globales de FRED
    macro_data = fetch_live_macro(force_refresh=force_refresh, defaults=defaults)

    # 5. Obtener Probabilidades de Arquetipos Corporativos con GMM
    prob_arch1, prob_arch2 = compute_corp_archetypes(acc_dict, defaults=defaults)

    # 6. Ensamblar el Vector Completo de 21 Variables Canónicas
    assembled_data = dict(acc_dict)
    assembled_data["company_name"] = c_name
    assembled_data["ticker"] = clean_ticker
    assembled_data["year"] = f_year

    # Inyectar mercado
    for k in ["stock_price_close", "market_cap", "tag_EntityCommonStockSharesOutstanding", "merton_distance_to_default"]:
        if k in market_data:
            assembled_data[k] = market_data[k]

    # Inyectar noticias NLP
    for k in ["news_sentiment_range_6m", "market_news_sentiment_mean", "news_sentiment_avg", "nlp_sentiment_decayed_30d", "news_sentiment_max"]:
        if k in news_data:
            assembled_data[k] = news_data[k]

    # Inyectar macroeconomía
    for k in ["macro_interest_rate", "macro_inflation", "macro_real_gdp_growth_yoy", "macro_regime_expansion_prob", "macro_unemployment_rate"]:
        if k in macro_data:
            assembled_data[k] = macro_data[k]

    # Inyectar arquetipos corporativos GMM
    assembled_data["corp_archetype_prob_1"] = prob_arch1
    assembled_data["corp_archetype_prob_2"] = prob_arch2

    # Defaults de seguridad para que las 21 variables óptimas nunca contengan NaN o None
    for feat_name, def_val in defaults.items():
        if feat_name not in assembled_data or pd.isna(assembled_data[feat_name]):
            assembled_data[feat_name] = def_val

    for feat_name, def_val in CANONICAL_21_DEFAULTS.items():
        if feat_name not in assembled_data or pd.isna(assembled_data[feat_name]):
            assembled_data[feat_name] = def_val

    composite_source = f"{acc_source} + {market_data['_market_source']} + {news_data['_news_source']}"

    return {
        "found": True,
        "source": composite_source,
        "ticker": clean_ticker,
        "company_name": c_name,
        "company_data": assembled_data,
        "cache_metadata": {
            "accounting_source": acc_source,
            "market_source": market_data.get("_market_source"),
            "news_source": news_data.get("_news_source"),
            "macro_source": macro_data.get("_macro_source"),
            "market_cached": market_data.get("_market_cached", False),
            "news_cached": news_data.get("_news_cached", False),
            "macro_cached": macro_data.get("_macro_cached", False)
        }
    }


def get_cache_stats() -> Dict[str, Any]:
    """Retorna métricas de telemetría y conteo del estado del caché SQLite."""
    stats = {}
    try:
        with get_db_connection() as con:
            stats["accounting_cached_count"] = con.execute("SELECT COUNT(*) FROM company_accounting_cache").fetchone()[0]
            stats["market_cached_count"] = con.execute("SELECT COUNT(*) FROM company_market_cache").fetchone()[0]
            stats["news_cached_count"] = con.execute("SELECT COUNT(*) FROM company_news_cache").fetchone()[0]
            macro_row = con.execute("SELECT * FROM global_macro_cache WHERE id = 1").fetchone()
            if macro_row:
                macro_dict = {
                    "source": macro_row["source"],
                    "updated_at": datetime.fromtimestamp(macro_row["updated_at"], tz=timezone.utc).isoformat(),
                    "interest_rate": macro_row["interest_rate"],
                    "inflation": macro_row["inflation"],
                    "unemployment": macro_row["unemployment_rate"],
                    "gdp_growth": macro_row["gdp_growth"]
                }
                if "regime_expansion_prob" in macro_row.keys():
                    macro_dict["regime_expansion_prob"] = macro_row["regime_expansion_prob"]
                stats["macro_status"] = macro_dict
            else:
                stats["macro_status"] = "No inicializado"
    except Exception as e:
        stats["error"] = str(e)
    return stats
