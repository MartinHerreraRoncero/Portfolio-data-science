"""
Modulo de Especificacion Canonica de Caracteristicas (Features) y Taxonomia de Pilares.
"""
OPTIMAL_21_FEATURES = [
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

TAXONOMIA_PILARES = {
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
    'Pilar Bursatil y Estructural': [
        'merton_distance_to_default',
        'stock_price_close',
        'market_cap'
    ],
    'Pilar Macroeconomico y Regimenes': [
        'macro_inflation',
        'macro_real_gdp_growth_yoy',
        'macro_interest_rate',
        'macro_regime_expansion_prob'
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

TARGET_COLS = ['target_bankrupt_12m', 'target_bankrupt_24m']
META_COLS = ['adsh', 'cik', 'ticker', 'filed_date', 'filed_year', 'filed_month', 'sic']

# Imputaciones Canonicas de Redes Complejas
DEFAULT_NETWORK_PAGERANK = 0.000330
DEFAULT_NETWORK_BETWEENNESS = 0.000885
