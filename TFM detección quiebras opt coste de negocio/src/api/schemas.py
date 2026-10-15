"""
Esquemas Pydantic v2 para la API RESTful de Prediccion de Insolvencia S&P 500 / SEC EDGAR.
Cumplimiento EU AI Act Art. 12/14 y Federal Reserve SR 11-7.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, model_validator

class CompanyFinancialPayload(BaseModel):
    company_name: str = Field("Corp_Alpha_Inc", json_schema_extra={"example": "Corp_Alpha_Inc"})
    year: int = Field(2024, json_schema_extra={"example": 2024})
    cik: Optional[str] = Field(None, json_schema_extra={"example": "0000320193"})
    ticker: Optional[str] = Field(None, json_schema_extra={"example": "AAPL"})

    # --- 21 Variables Óptimas SEC EDGAR + Merton + NLP + Macro + GMM ---
    tag_NetIncomeLoss: Optional[float] = Field(
        None,
        description="Resultado neto contable / pérdidas (Net Income)",
        json_schema_extra={"example": 1200000.0}
    )
    tag_WorkingCapital: Optional[float] = Field(
        None,
        description="Fondo de maniobra neto (Working Capital)",
        json_schema_extra={"example": 4000000.0}
    )
    merton_distance_to_default: Optional[float] = Field(
        None,
        description="Distancia estructural de Merton KMV a la quiebra (DD)",
        json_schema_extra={"example": 4.2}
    )
    tag_Assets: Optional[float] = Field(
        None,
        description="Activos totales consolidados",
        json_schema_extra={"example": 18000000.0}
    )
    pub_lag_days: Optional[float] = Field(
        None,
        description="Retardo de publicación contable SEC 10-K/10-Q (días)",
        json_schema_extra={"example": 45.0}
    )
    stock_price_close: Optional[float] = Field(
        None,
        description="Precio de cotización bursátil de cierre",
        json_schema_extra={"example": 45.5}
    )
    tag_RetainedEarningsAccumulatedDeficit: Optional[float] = Field(
        None,
        description="Reservas y beneficios acumulados / Déficit",
        json_schema_extra={"example": 6000000.0}
    )
    corp_archetype_prob_1: Optional[float] = Field(
        None,
        description="Probabilidad de pertenencia al Arquetipo Corporativo 1 (GMM)",
        json_schema_extra={"example": 0.05}
    )
    macro_inflation: Optional[float] = Field(
        None,
        description=(
            "Nivel de inflación macroeconómica. Si se introduce como tasa porcentual interanual "
            "(ej. 2.8 para 2.8% o cualquier valor <= 25.0), la API la proyecta coherentemente "
            "a la escala del índice CPI (~236.468) requerida por el StandardScaler para erradicar el "
            "Training-Serving Skew y evitar outliers numéricos de -30 sigma. Si se proporciona directamente "
            "el nivel de índice CPI (ej. 236.5 o 313.0), se procesa directamente."
        ),
        json_schema_extra={"example": 2.8}
    )
    market_cap: Optional[float] = Field(
        None,
        description="Capitalización bursátil total",
        json_schema_extra={"example": 2275000000.0}
    )
    tag_EntityCommonStockSharesOutstanding: Optional[float] = Field(
        None,
        description="Acciones comunes en circulación",
        json_schema_extra={"example": 50000000.0}
    )
    tag_CommonStockSharesAuthorized: Optional[float] = Field(
        None,
        description="Acciones comunes autorizadas",
        json_schema_extra={"example": 100000000.0}
    )
    macro_real_gdp_growth_yoy: Optional[float] = Field(
        None,
        description="Crecimiento interanual PIB real (%)",
        json_schema_extra={"example": 2.1}
    )
    news_sentiment_range_6m: Optional[float] = Field(
        None,
        description="Volatilidad del sentimiento mediático (6m)",
        json_schema_extra={"example": 0.35}
    )
    tag_CommonStockParOrStatedValuePerShare: Optional[float] = Field(
        None,
        description="Valor facial por acción común",
        json_schema_extra={"example": 0.01}
    )
    tag_CommonStockValue: Optional[float] = Field(
        None,
        description="Valor nominal de acciones comunes",
        json_schema_extra={"example": 500000.0}
    )
    macro_interest_rate: Optional[float] = Field(
        None,
        description="Tipo de interés de referencia (Fed Funds)",
        json_schema_extra={"example": 4.5}
    )
    market_news_sentiment_mean: Optional[float] = Field(
        None,
        description="Sentimiento medio global del mercado",
        json_schema_extra={"example": 0.52}
    )
    corp_archetype_prob_2: Optional[float] = Field(
        None,
        description="Probabilidad de pertenencia al Arquetipo Corporativo 2 (GMM)",
        json_schema_extra={"example": 0.01}
    )
    macro_regime_expansion_prob: Optional[float] = Field(
        None,
        description="Probabilidad de régimen macroeconómico de expansión (GMM)",
        json_schema_extra={"example": 0.75}
    )
    news_sentiment_avg: Optional[float] = Field(
        None,
        description="Sentimiento medio agregado de noticias (FinBERT)",
        json_schema_extra={"example": 0.25}
    )

    # Campos legacy para compatibilidad retroactiva
    tag_CashAndCashEquivalentsAtCarryingValue: Optional[float] = None
    tag_AssetsCurrent: Optional[float] = None
    tag_Revenues_combined: Optional[float] = None
    tag_InterestExpense: Optional[float] = None
    tag_RevenueFromContractWithCustomerExcludingAssessedTax: Optional[float] = None
    tag_StockholdersEquity: Optional[float] = None
    news_sentiment_max: Optional[float] = None
    nlp_sentiment_decayed_30d: Optional[float] = None
    macro_unemployment_rate: Optional[float] = None
    current_assets: Optional[float] = None
    total_assets: Optional[float] = None
    net_income: Optional[float] = None
    retained_earnings: Optional[float] = None
    market_value: Optional[float] = None
    total_revenue: Optional[float] = None
    net_sales: Optional[float] = None
    cost_of_goods_sold: Optional[float] = None
    ebitda: Optional[float] = None
    ebit: Optional[float] = None
    inventory: Optional[float] = None
    total_receivables: Optional[float] = None
    total_current_liabilities: Optional[float] = None
    total_liabilities: Optional[float] = None
    total_long_term_debt: Optional[float] = None
    gross_profit: Optional[float] = None
    operating_expenses: Optional[float] = None
    depreciation_and_amortization: Optional[float] = None

    @model_validator(mode="before")
    @classmethod
    def map_legacy_fields(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        # Convertir CIK a string si viene como entero
        if data.get("cik") is not None:
            data["cik"] = str(data["cik"])

        # Mapeo de campos legacy a tags si no vienen informados
        if data.get("tag_AssetsCurrent") is None and data.get("current_assets") is not None:
            data["tag_AssetsCurrent"] = data["current_assets"]
        if data.get("tag_Assets") is None and data.get("total_assets") is not None:
            data["tag_Assets"] = data["total_assets"]
        if data.get("tag_NetIncomeLoss") is None and data.get("net_income") is not None:
            data["tag_NetIncomeLoss"] = data["net_income"]
        if data.get("tag_RetainedEarningsAccumulatedDeficit") is None and data.get("retained_earnings") is not None:
            data["tag_RetainedEarningsAccumulatedDeficit"] = data["retained_earnings"]
        if data.get("market_cap") is None and data.get("market_value") is not None:
            data["market_cap"] = data["market_value"]
        if data.get("tag_Revenues_combined") is None:
            rev = data.get("total_revenue") or data.get("net_sales")
            if rev is not None:
                data["tag_Revenues_combined"] = rev
        if data.get("tag_WorkingCapital") is None:
            ca = data.get("current_assets") or data.get("tag_AssetsCurrent")
            cl = data.get("total_current_liabilities")
            if ca is not None and cl is not None:
                data["tag_WorkingCapital"] = ca - cl
        if data.get("tag_StockholdersEquity") is None:
            ta = data.get("total_assets") or data.get("tag_Assets")
            tl = data.get("total_liabilities")
            if ta is not None and tl is not None:
                data["tag_StockholdersEquity"] = ta - tl

        # Mapeo de sentimiento legacy hacia news_sentiment_avg
        if data.get("news_sentiment_avg") is None:
            if data.get("nlp_sentiment_decayed_30d") is not None:
                data["news_sentiment_avg"] = data["nlp_sentiment_decayed_30d"]
            elif data.get("market_news_sentiment_mean") is not None:
                data["news_sentiment_avg"] = data["market_news_sentiment_mean"]

        return data

class PredictionResponse(BaseModel):
    company_name: str
    year: int
    bankruptcy_probability: float
    is_bankruptcy_predicted: bool
    risk_category: str
    decision_threshold: float
    altman_z_score: Optional[float] = None
    merton_distance_to_default: Optional[float] = None
    adverse_action_notices: List[str] = Field(default_factory=list, description="Razones de riesgo conforme a EU AI Act y FCRA")
    latency_ms: float

class BatchPredictionResponse(BaseModel):
    total_processed: int
    mean_latency_ms: float
    total_latency_ms: float
    predictions: List[PredictionResponse]

class ModelInfoResponse(BaseModel):
    model_name: str
    version: str
    features_count: int
    features: List[str]
    optimal_decision_threshold: float
    framework: str
    hardware_acceleration: str
    regulatory_compliance: List[str]

class ScorecardResponse(BaseModel):
    company_name: str
    year: int
    credit_score: float = Field(..., description="Score regulatorio WoE (300 - 850 pts)")
    rating_band: str = Field(..., description="Banda de calificacion de riesgo (AAA a D)")
    implied_pd: float = Field(..., description="Probabilidad de Incumplimiento estimada por el Scorecard")
    latency_ms: float