"""
Esquemas Pydantic v2 unificados para el Microservicio RESTful y el Cockpit Interactivo de Predicción de Quiebra S&P 500.
Cumplimiento normativo y gobernanza algorítmica:
- Federal Reserve SR 11-7 (Model Risk Management)
- Basilea III / IV IRB Advanced Standard (Credit WoE Scorecard 300-850 pts)
- NIIF 9 / IFRS 9 (Expected Credit Loss - ECL)
- EU AI Act (Regulation EU 2024/1689, Articles 12, 13 & 14)
- Equal Credit Opportunity Act (ECOA) & Fair Credit Reporting Act (FCRA Adverse Action Notices)
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, model_validator

# ---------------------------------------------------------------------------
# Definición Canónica de las 21 Características Óptimas
# ---------------------------------------------------------------------------
OPTIMAL_21_FEATURES: List[str] = [
    'tag_NetIncomeLoss',
    'tag_WorkingCapital',
    'merton_distance_to_default',
    'tag_Assets',
    'pub_lag_days',
    'stock_price_close',
    'tag_RetainedEarningsAccumulatedDeficit',
    'corp_archetype_prob_1',
    'macro_inflation',
    'market_cap',
    'tag_EntityCommonStockSharesOutstanding',
    'tag_CommonStockSharesAuthorized',
    'macro_real_gdp_growth_yoy',
    'news_sentiment_range_6m',
    'tag_CommonStockParOrStatedValuePerShare',
    'tag_CommonStockValue',
    'macro_interest_rate',
    'market_news_sentiment_mean',
    'corp_archetype_prob_2',
    'macro_regime_expansion_prob',
    'news_sentiment_avg'
]

# ---------------------------------------------------------------------------
# Taxonomía Oficial de los 5 Pilares de Riesgo Financiero
# ---------------------------------------------------------------------------
TAXONOMIA_PILARES: Dict[str, List[str]] = {
    'Pilar Contable (SEC)': [
        'tag_NetIncomeLoss',
        'tag_WorkingCapital',
        'tag_Assets',
        'pub_lag_days',
        'tag_RetainedEarningsAccumulatedDeficit',
        'tag_EntityCommonStockSharesOutstanding',
        'tag_CommonStockSharesAuthorized',
        'tag_CommonStockParOrStatedValuePerShare',
        'tag_CommonStockValue'
    ],
    "Pilar Bursatil y Estructural": [
        "merton_distance_to_default",
        "stock_price_close",
        "market_cap"
    ],
    "Pilar Macroeconomico y Regimenes": [
        "macro_inflation",
        "macro_real_gdp_growth_yoy",
        "macro_interest_rate",
        "macro_regime_expansion_prob"
    ],
    'Pilar Sentimiento NLP y Noticias': [
        'news_sentiment_range_6m',
        'market_news_sentiment_mean',
        'news_sentiment_avg'
    ],
    'Pilar Arquetipos Corporativos': [
        'corp_archetype_prob_1',
        'corp_archetype_prob_2'
    ]
}

FEATURE_TO_PILLAR: Dict[str, str] = {}
for p_name, p_feats in TAXONOMIA_PILARES.items():
    for f in p_feats:
        FEATURE_TO_PILLAR[f] = p_name


# ---------------------------------------------------------------------------
# Entrada Financiera Corporativa (Payload)
# ---------------------------------------------------------------------------
class CompanyFinancialPayload(BaseModel):
    company_name: str = Field("Corp_Alpha_Inc", json_schema_extra={"example": "Corp_Alpha_Inc"}, description="Razón social o denominación de la empresa")
    year: int = Field(2024, json_schema_extra={"example": 2024}, description="Año fiscal del reporte")
    cik: Optional[str] = Field(None, json_schema_extra={"example": "0000320193"}, description="Central Index Key SEC")
    ticker: Optional[str] = Field(None, json_schema_extra={"example": "AAPL"}, description="Ticker bursátil")

    # --- 21 Variables Óptimas Canónicas ---
    tag_NetIncomeLoss: Optional[float] = Field(
        None,
        description="Resultado neto contable / pérdidas (USD)",
        json_schema_extra={"example": 1200000.0}
    )
    tag_WorkingCapital: Optional[float] = Field(
        None,
        description="Fondo de maniobra neto (Working Capital, USD)",
        json_schema_extra={"example": 16498000.0}
    )
    merton_distance_to_default: Optional[float] = Field(
        None,
        description="Distancia estructural de Merton KMV a la quiebra (desviaciones estándar)",
        json_schema_extra={"example": 4.53}
    )
    tag_Assets: Optional[float] = Field(
        None,
        description="Activos totales consolidados (USD)",
        json_schema_extra={"example": 243633500.0}
    )
    pub_lag_days: Optional[float] = Field(
        None,
        description="Retardo de publicación contable SEC 10-K/10-Q (días)",
        json_schema_extra={"example": 43.0}
    )
    stock_price_close: Optional[float] = Field(
        None,
        description="Precio de cotización bursátil de cierre (USD)",
        json_schema_extra={"example": 45.0}
    )
    tag_RetainedEarningsAccumulatedDeficit: Optional[float] = Field(
        None,
        description="Reservas y beneficios acumulados / Déficit (USD)",
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
            "Training-Serving Skew. Si se proporciona directamente el nivel de índice CPI (ej. 236.5 o 313.0), se procesa directamente."
        ),
        json_schema_extra={"example": 2.8}
    )
    market_cap: Optional[float] = Field(
        None,
        description="Capitalización bursátil total (USD)",
        json_schema_extra={"example": 2275000000.0}
    )
    tag_EntityCommonStockSharesOutstanding: Optional[float] = Field(
        None,
        description="Acciones comunes en circulación",
        json_schema_extra={"example": 34679114.0}
    )
    tag_CommonStockSharesAuthorized: Optional[float] = Field(
        None,
        description="Acciones comunes autorizadas",
        json_schema_extra={"example": 130000000.0}
    )
    macro_real_gdp_growth_yoy: Optional[float] = Field(
        None,
        description="Crecimiento interanual PIB real (%)",
        json_schema_extra={"example": 2.40}
    )
    news_sentiment_range_6m: Optional[float] = Field(
        None,
        description="Volatilidad del sentimiento mediático a 6 meses",
        json_schema_extra={"example": 0.35}
    )
    tag_CommonStockParOrStatedValuePerShare: Optional[float] = Field(
        None,
        description="Valor facial por acción común (USD)",
        json_schema_extra={"example": 0.01}
    )
    tag_CommonStockValue: Optional[float] = Field(
        None,
        description="Valor nominal de acciones comunes en balance (USD)",
        json_schema_extra={"example": 152000.0}
    )
    macro_interest_rate: Optional[float] = Field(
        None,
        description="Tipo de interés oficial (Fed Funds Rate, %)",
        json_schema_extra={"example": 4.5}
    )
    market_news_sentiment_mean: Optional[float] = Field(
        None,
        description="Sentimiento medio global del mercado",
        json_schema_extra={"example": 0.052}
    )
    corp_archetype_prob_2: Optional[float] = Field(
        None,
        description="Probabilidad de pertenencia al Arquetipo Corporativo 2 (GMM)",
        json_schema_extra={"example": 0.0002}
    )
    macro_regime_expansion_prob: Optional[float] = Field(
        None,
        description="Probabilidad de régimen macroeconómico de expansión (GMM)",
        json_schema_extra={"example": 0.75}
    )
    news_sentiment_avg: Optional[float] = Field(
        None,
        description="Sentimiento medio agregado de noticias de la empresa (FinBERT)",
        json_schema_extra={"example": 0.05}
    )

    # --- Campos Legacy y Auxiliares para Garantizar Compatibilidad Retroactiva ---
    tag_CashAndCashEquivalentsAtCarryingValue: Optional[float] = None
    tag_AssetsCurrent: Optional[float] = None
    tag_LiabilitiesCurrent: Optional[float] = None
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
    working_capital: Optional[float] = None
    operating_income: Optional[float] = None
    stockholders_equity: Optional[float] = None
    current_liabilities: Optional[float] = None
    shares_outstanding: Optional[float] = None
    shares_authorized: Optional[float] = None
    stock_price: Optional[float] = None
    distance_to_default: Optional[float] = None
    merton_dd: Optional[float] = None
    inflation: Optional[float] = None
    gdp_growth: Optional[float] = None
    interest_rate: Optional[float] = None
    news_range: Optional[float] = None
    news_sentiment: Optional[float] = None
    market_news_sentiment: Optional[float] = None
    ratio_working_capital_to_assets: Optional[float] = None
    ratio_current_ratio: Optional[float] = None
    ratio_leverage: Optional[float] = None
    altman_z_score: Optional[float] = None
    stock_volatility_30d: Optional[float] = None
    macro_yield_curve: Optional[float] = None
    macro_credit_spread: Optional[float] = None
    news_sentiment_std_3m: Optional[float] = None

    @model_validator(mode="before")
    @classmethod
    def map_legacy_fields(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        # Convertir CIK a string si viene como número
        if data.get("cik") is not None:
            data["cik"] = str(data["cik"])

        # Mapeo de Activos y Balance
        if data.get("tag_Assets") is None and data.get("total_assets") is not None:
            data["tag_Assets"] = data["total_assets"]
        if data.get("tag_AssetsCurrent") is None and data.get("current_assets") is not None:
            data["tag_AssetsCurrent"] = data["current_assets"]
        if data.get("tag_NetIncomeLoss") is None and data.get("net_income") is not None:
            data["tag_NetIncomeLoss"] = data["net_income"]
        if data.get("tag_RetainedEarningsAccumulatedDeficit") is None and data.get("retained_earnings") is not None:
            data["tag_RetainedEarningsAccumulatedDeficit"] = data["retained_earnings"]
        if data.get("tag_EntityCommonStockSharesOutstanding") is None and data.get("shares_outstanding") is not None:
            data["tag_EntityCommonStockSharesOutstanding"] = data["shares_outstanding"]
        if data.get("tag_CommonStockSharesAuthorized") is None and data.get("shares_authorized") is not None:
            data["tag_CommonStockSharesAuthorized"] = data["shares_authorized"]

        # Deducción de Fondo de Maniobra
        if data.get("tag_WorkingCapital") is None:
            if data.get("working_capital") is not None:
                data["tag_WorkingCapital"] = data["working_capital"]
            else:
                ca = data.get("current_assets") or data.get("tag_AssetsCurrent")
                cl = data.get("current_liabilities") or data.get("total_current_liabilities") or data.get("tag_LiabilitiesCurrent")
                if ca is not None and cl is not None:
                    data["tag_WorkingCapital"] = float(ca) - float(cl)

        # Deducción de Fondos Propios
        if data.get("tag_StockholdersEquity") is None:
            if data.get("stockholders_equity") is not None:
                data["tag_StockholdersEquity"] = data["stockholders_equity"]
            else:
                ta = data.get("total_assets") or data.get("tag_Assets")
                tl = data.get("total_liabilities")
                if ta is not None and tl is not None:
                    data["tag_StockholdersEquity"] = float(ta) - float(tl)

        # Mapeo de Ingresos
        if data.get("tag_Revenues_combined") is None:
            rev = data.get("total_revenue") or data.get("net_sales")
            if rev is not None:
                data["tag_Revenues_combined"] = rev

        # Mapeo Bursátil
        if data.get("market_cap") is None and data.get("market_value") is not None:
            data["market_cap"] = data["market_value"]
        if data.get("stock_price_close") is None and data.get("stock_price") is not None:
            data["stock_price_close"] = data.get("stock_price")
        if data.get("merton_distance_to_default") is None:
            mdd = data.get("distance_to_default") or data.get("merton_dd")
            if mdd is not None:
                data["merton_distance_to_default"] = mdd

        # Mapeo Macroeconómico
        if data.get("macro_inflation") is None and data.get("inflation") is not None:
            data["macro_inflation"] = data["inflation"]
        if data.get("macro_real_gdp_growth_yoy") is None and data.get("gdp_growth") is not None:
            data["macro_real_gdp_growth_yoy"] = data["gdp_growth"]
        if data.get("macro_interest_rate") is None and data.get("interest_rate") is not None:
            data["macro_interest_rate"] = data["interest_rate"]

        # Mapeo Sentimiento NLP
        if data.get("news_sentiment_range_6m") is None and data.get("news_range") is not None:
            data["news_sentiment_range_6m"] = data["news_range"]
        if data.get("market_news_sentiment_mean") is None and data.get("market_news_sentiment") is not None:
            data["market_news_sentiment_mean"] = data["market_news_sentiment"]
        if data.get("news_sentiment_avg") is None:
            if data.get("news_sentiment") is not None:
                data["news_sentiment_avg"] = data["news_sentiment"]
            elif data.get("nlp_sentiment_decayed_30d") is not None:
                data["news_sentiment_avg"] = data["nlp_sentiment_decayed_30d"]
            elif data.get("market_news_sentiment_mean") is not None:
                data["news_sentiment_avg"] = data["market_news_sentiment_mean"]

        return data


# Alias canónico
CompanyFinancialInput = CompanyFinancialPayload


# ---------------------------------------------------------------------------
# Entrada por Lotes (Batch Input)
# ---------------------------------------------------------------------------
class BatchPredictionInput(BaseModel):
    companies: List[CompanyFinancialPayload] = Field(
        ...,
        description="Lista de entidades corporativas para evaluación crediticia por lotes"
    )


# ---------------------------------------------------------------------------
# Salida de Predicción Individual (Prediction Response)
# ---------------------------------------------------------------------------
class PredictionResponse(BaseModel):
    company_name: str = Field(..., description="Nombre corporativo o ticker de la entidad")
    year: int = Field(2024, description="Año fiscal evaluado")
    bankruptcy_probability: float = Field(..., description="Probabilidad calibrada de insolvencia a 12 meses (PD)")
    is_bankruptcy_predicted: bool = Field(..., description="Alerta de insolvencia activa (probabilidad >= umbral)")
    risk_category: str = Field(..., description="Categoría institucional de riesgo")
    decision_threshold: float = Field(..., description="Umbral óptimo tau* del clasificador")
    altman_z_score: Optional[float] = Field(None, description="Puntuación Z-Score de Altman aproximada")
    merton_distance_to_default: Optional[float] = Field(None, description="Distancia estructural Merton al default (DD)")
    adverse_action_notices: List[str] = Field(default_factory=list, description="Razones determinantes de riesgo conforme a EU AI Act y FCRA")
    latency_ms: float = Field(0.0, description="Tiempo de inferencia en milisegundos")

    # Campos de paridad completa con el framework de gobernanza
    model_version: Optional[str] = Field("Voting Classifier Ensemble", description="Denominación formal del modelo")
    model_rank: Optional[str] = Field("Rank 1 SOTA", description="Posición en el benchmark competitivo")
    operational_threshold: Optional[float] = Field(None, description="Umbral de decisión operativa (tau*)")
    distress_decision: Optional[int] = Field(None, description="Decisión binaria de alerta (1=Alerta, 0=Solvente)")
    risk_tier: Optional[str] = Field(None, description="Nivel regulatorio de riesgo crediticio")
    estimated_cost_usd: Optional[float] = Field(None, description="Coste institucional esperado según matriz asimétrica 24x")
    regulatory_framework: Optional[str] = Field("Fed SR 11-7, Basel III, IFRS 9, EU AI Act", description="Marcos regulatorios cumplidos")
    pillar_contributions: Optional[Dict[str, float]] = Field(None, description="Aporte agregado de los 5 pilares de riesgo")

    @model_validator(mode="before")
    @classmethod
    def sync_governance_fields(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        if data.get("operational_threshold") is None:
            data["operational_threshold"] = data.get("decision_threshold")
        if data.get("distress_decision") is None:
            is_dist = data.get("is_bankruptcy_predicted")
            if is_dist is not None:
                data["distress_decision"] = 1 if is_dist else 0
            else:
                prob = data.get("bankruptcy_probability", 0.0)
                tau = data.get("decision_threshold", 0.158)
                data["distress_decision"] = 1 if prob >= tau else 0
        if data.get("risk_tier") is None:
            data["risk_tier"] = data.get("risk_category")
        return data


# Alias canónico
PredictionOutput = PredictionResponse


# ---------------------------------------------------------------------------
# Salida de Predicción por Lotes (Batch Response)
# ---------------------------------------------------------------------------
class BatchPredictionResponse(BaseModel):
    total_processed: int = Field(..., description="Total de entidades evaluadas en el lote")
    mean_latency_ms: float = Field(0.0, description="Latencia media por empresa en ms")
    total_latency_ms: float = Field(0.0, description="Latencia total del procesamiento por lote en ms")
    total_alerts: Optional[int] = Field(0, description="Total de empresas con alerta de insolvencia activa")
    portfolio_average_pd: Optional[float] = Field(0.0, description="Probabilidad media de insolvencia de la cartera")
    total_estimated_cost_usd: Optional[float] = Field(0.0, description="Coste total estimado de la cartera en USD")
    predictions: List[PredictionResponse] = Field(default_factory=list, description="Predicciones individuales de la cartera")


# Alias canónico
BatchPredictionOutput = BatchPredictionResponse


# ---------------------------------------------------------------------------
# Metadatos del Modelo y Catálogo (Model Info)
# ---------------------------------------------------------------------------
class ModelInfoResponse(BaseModel):
    model_name: str = Field(..., description="Denominación del modelo principal")
    version: str = Field(..., description="Versión de la API y modelo")
    features_count: int = Field(..., description="Número de variables predictivas canónicas")
    features: List[str] = Field(..., description="Lista ordenada de características predictivas")
    optimal_decision_threshold: float = Field(..., description="Umbral óptimo de decisión financiera")
    framework: str = Field(..., description="Arquitectura de los modelos y ensambles")
    hardware_acceleration: str = Field(..., description="Entorno de aceleración hardware")
    regulatory_compliance: List[str] = Field(..., description="Directivas y normativas certificadas")


class ModelInfoItem(BaseModel):
    model_id: str
    model_name: str
    model_rank: str
    threshold_12m: float
    threshold_24m: Optional[float] = None
    pr_auc: float
    roc_auc: float
    description: str
    status: str = "ready"


class ModelCatalogResponse(BaseModel):
    default_model: str = "voting_classifier"
    total_models: int
    models: List[ModelInfoItem]
    regulatory_frameworks: List[str] = Field(
        default_factory=lambda: ["Fed SR 11-7", "Basel III", "IFRS 9", "EU AI Act"]
    )


class FeatureItem(BaseModel):
    feature_name: str
    pillar: str
    description: str
    default_median: Optional[float] = None


class FeaturesInfoResponse(BaseModel):
    total_features: int = 21
    features: List[FeatureItem]
    pillars: Dict[str, List[str]]


# ---------------------------------------------------------------------------
# Scorecard Crediticio Basilea III (WoE)
# ---------------------------------------------------------------------------
class ScorecardResponse(BaseModel):
    company_name: str = Field(..., description="Nombre corporativo o ticker")
    year: int = Field(2024, description="Año evaluado")
    credit_score: float = Field(..., description="Score regulatorio WoE en escala 300 a 850 puntos")
    rating_band: str = Field(..., description="Banda de calificación crediticia (AAA a D)")
    implied_pd: float = Field(..., description="Probabilidad de Incumplimiento implícita en el Scorecard")
    pd_probability: Optional[float] = Field(None, description="Probabilidad de default equivalente")
    decision: Optional[str] = Field(None, description="Resolución crediticia de admisión")
    points_breakdown: Optional[Dict[str, float]] = None
    regulatory_framework: Optional[str] = Field(
        "Basel III / IFRS 9 Weight-of-Evidence Scorecard",
        description="Normativa aplicable al scorecard"
    )
    latency_ms: float = Field(0.0, description="Latencia en ms")

    @model_validator(mode="before")
    @classmethod
    def sync_scorecard_fields(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        if data.get("pd_probability") is None and data.get("implied_pd") is not None:
            data["pd_probability"] = data["implied_pd"]
        elif data.get("implied_pd") is None and data.get("pd_probability") is not None:
            data["implied_pd"] = data["pd_probability"]
        return data


# ---------------------------------------------------------------------------
# Explicabilidad XAI (SHAP TreeExplainer por Pilares)
# ---------------------------------------------------------------------------
class ExplanationFactor(BaseModel):
    feature_name: str = Field(..., description="Variable financiera analizada")
    shap_value: float = Field(..., description="Contribución aditiva SHAP local en escala logit")
    feature_value: Optional[float] = Field(None, description="Valor contable o de mercado reportado")
    information_pillar: str = Field(..., description="Pilar de riesgo financiero correspondiente")
    risk_impact: str = Field(..., description="Efecto: 'Incrementa Riesgo' o 'Mitiga Riesgo'")


class ExplanationOutput(BaseModel):
    company_name: str = Field(..., description="Nombre corporativo")
    model_version: str = Field(..., description="Modelo de árboles utilizado para SHAP")
    base_value_logit: float = Field(..., description="Valor logit base de referencia del modelo")
    final_score_logit: float = Field(..., description="Logit total acumulado")
    bankruptcy_probability: float = Field(..., description="Probabilidad resultante de insolvencia")
    top_risk_drivers: List[ExplanationFactor] = Field(..., description="Factores determinantes ordenados por magnitud SHAP")
    pillar_summary: Optional[Dict[str, float]] = Field(None, description="Impacto neto agrupado por pilar de riesgo")
    regulatory_compliance: str = Field(
        "Federal Reserve SR 11-7 / ECOA Adverse Action Notice / EU AI Act Art. 13-14",
        description="Certificación de cumplimiento normativo XAI"
    )
