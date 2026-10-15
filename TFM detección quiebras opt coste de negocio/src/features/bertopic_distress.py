"""
Financial Distress Topic Modeling & Exponential Temporal Aggregation Pipeline.
Module: src/features/bertopic_distress.py

Extracts corporate distress topics from English financial news using a hybrid
NLP topic architecture (BERTopic / c-TF-IDF / KeyBERT semantic taxonomy + 
high-precision domain lexical matchers):
  1. Topic: Default & Debt Covenants ("debt default", "covenant breach", "restructuring",
     "missed payment", "liquidity shortfall", "credit downgrade")
  2. Topic: Concursal & Chapter 11 ("chapter 11", "bankruptcy protection", 
     "insolvency filing", "liquidation", "going-concern warning")
  3. Topic: Litigation & Fraud ("SEC investigation", "subpoena", "accounting fraud",
     "restatement", "class action")

Computes:
  - News-level distress topic intensity: `nlp_distress_topic_intensity`
  - Temporal exponential decay aggregation with half-life t_1/2 = 14 days:
      S_{i,t}^{(decay)} = sum_k(S_{i,k} * exp(-lambda * Delta t_k)) / sum_k(exp(-lambda * Delta t_k))
  - Media Silence indicator (`media_silence_months_count`) tracking consecutive 
    months without news coverage.
  - High-performance DuckDB integration for panel fusion with SEC accounting and market data.

TFM: Prediccion de Insolvencia en Empresas del S&P 500 con Enfoque Multimodal.
"""

import os
import sys
import re
import math
import time
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union

import duckdb
import numpy as np
import pandas as pd

# Paths
DEFAULT_NEWS_PARQUET = Path("data/processed/news_dataset/consolidated_financial_news_1999_2026.parquet")
DEFAULT_SENTIMENT_PARQUET = Path("data/processed/news_dataset/consolidated_news_sentiment_1999_2026.parquet")
DEFAULT_DUCKDB_PATH = Path("data/processed/news_dataset/consolidated_news.duckdb")
OUTPUT_DIR = Path("data/processed/news_dataset")
OUTPUT_NEWS_DISTRESS_PARQUET = OUTPUT_DIR / "consolidated_news_distress_topics.parquet"
OUTPUT_MONTHLY_FEATURES_PARQUET = OUTPUT_DIR / "monthly_nlp_distress_features.parquet"

# Exponential Decay Parameter (Half-life = 14 days)
DEFAULT_HALF_LIFE_DAYS = 14.0
LAMBDA_DECAY = math.log(2.0) / DEFAULT_HALF_LIFE_DAYS  # approx 0.0495105 / day


# =====================================================================
# 1. FINANCIAL DISTRESS TAXONOMY & REGEX / SEMANTIC VOCABULARY (ENGLISH)
# =====================================================================

DEFAULT_COVENANTS_PATTERNS = [
    # Debt Default & Payment Failures
    (r"\bdebt\s+default(?:s|ed|ing)?\b", 1.0),
    (r"\bdefault(?:s|ed|ing)?\s+(?:on|upon)\s+(?:debt|bonds?|loans?|notes?|interest|obligations?|credit)\b", 1.0),
    (r"\b(?:bond|loan|note|credit)\s+default(?:s|ed|ing)?\b", 0.95),
    (r"\btechnical\s+default(?:s)?\b", 0.90),
    (r"\bmissed\s+(?:debt\s+|interest\s+|principal\s+|coupon\s+|loan\s+)?payment(?:s)?\b", 0.95),
    (r"\bfailed\s+to\s+(?:make|pay|meet)\s+(?:interest|debt|loan|coupon)\s+payment(?:s)?\b", 0.95),
    (r"\bdistressed\s+debt\b", 0.85),
    (r"\bdistressed\s+exchange\b", 0.85),
    (r"\bdebt\s+distress(?:ed)?\b", 0.85),
    (r"\bdebt\s+maturity\s+wall\b", 0.75),
    
    # Covenant Breaches & Waivers
    (r"\bcovenant\s+(?:breach(?:es|ed)?|violation(?:s)?|waiver(?:s)?|default(?:s)?)\b", 1.0),
    (r"\bbreach(?:ed|ing)?\s+(?:of\s+)?(?:debt\s+|credit\s+|loan\s+|financial\s+)?covenant(?:s)?\b", 1.0),
    (r"\bviolated\s+(?:debt\s+|credit\s+|loan\s+|financial\s+)?covenant(?:s)?\b", 0.95),
    (r"\bcredit\s+agreement\s+(?:breach|violation|default)\b", 0.95),
    (r"\bforbearance\s+agreement(?:s)?\b", 0.90),
    (r"\bcovenant\s+relief\b", 0.75),
    
    # Restructuring & Debt Renegotiation
    (r"\bdebt\s+restructuring\b", 0.90),
    (r"\bfinancial\s+restructuring\b", 0.85),
    (r"\bcapital\s+structure\s+restructuring\b", 0.80),
    (r"\brestructure(?:s|d|ing)?\s+(?:its\s+)?(?:heavy\s+|crushing\s+)?debt\b", 0.85),
    (r"\brenegotiat(?:e|ing|ion)\s+(?:debt|credit\s+facility|loan\s+terms)\b", 0.75),
    (r"\bdebt-for-equity\s+swap(?:s)?\b", 0.85),
    
    # Liquidity Shortfall & Squeeze
    (r"\bliquidity\s+(?:shortfall|shortage|crunch|crisis|squeeze|deficiency|concerns?|pressures?)\b", 0.90),
    (r"\bcash\s+crunch\b", 0.85),
    (r"\bcash\s+burn\s+(?:accelerates?|rapid|severe|critical)\b", 0.75),
    (r"\brun(?:ning)?\s+out\s+of\s+cash\b", 0.90),
    (r"\bcredit\s+line\s+(?:pulled|revoked|frozen|terminated|cancelled)\b", 0.90),
    
    # Credit Rating Downgrades
    (r"\bcredit\s+(?:rating\s+)?downgrade(?:s|d)?\b", 0.85),
    (r"\bdowngrade(?:s|d|ing)?\s+(?:to|deeper\s+into)\s+(?:junk|distressed|speculative|c-grade|default)\b", 0.95),
    (r"\brating\s+cut(?:s)?\s+to\s+junk\b", 0.90),
    (r"\bjunk\s+status\b", 0.70),
    (r"\bplaced\s+on\s+creditwatch\s+negative\b", 0.75),
    (r"\bcredit\s+rating\s+slashed\b", 0.80)
]

BANKRUPTCY_CHAPTER11_PATTERNS = [
    # Chapter 11 & Chapter 7 filings
    (r"\bchapter\s+11\b", 1.0),
    (r"\bchapter\s+7\b", 1.0),
    (r"\bchapter\s+15\b", 0.90),
    (r"\bpre-?packaged\s+chapter\s+11\b", 1.0),
    (r"\bpre-?arranged\s+bankruptcy\b", 1.0),
    
    # Bankruptcy Protection & Court
    (r"\bbankruptcy\s+protection\b", 1.0),
    (r"\bfile(?:s|d|ing)?\s+(?:for\s+)?bankruptcy\b", 1.0),
    (r"\bfiled\s+for\s+(?:chapter\s+11|chapter\s+7|bankruptcy)\b", 1.0),
    (r"\bbankruptcy\s+filing(?:s)?\b", 1.0),
    (r"\bbankruptcy\s+court\b", 0.90),
    (r"\bbankruptcy\s+judge\b", 0.85),
    (r"\bbankruptcy\s+petition\b", 1.0),
    (r"\bbankrupt\b", 0.95),
    (r"\bverge\s+of\s+bankruptcy\b", 0.95),
    (r"\bpossible\s+bankruptcy\b", 0.85),
    (r"\bweighing\s+bankruptcy\b", 0.90),
    (r"\bprepares?\s+for\s+bankruptcy\b", 0.95),
    
    # Insolvency & Liquidation
    (r"\binsolvency\s+filing(?:s)?\b", 1.0),
    (r"\binsolvency\s+proceeding(?:s)?\b", 1.0),
    (r"\binsolvent\b", 0.95),
    (r"\bcorporate\s+insolvency\b", 0.95),
    (r"\bliquidation\s+(?:plan|proceedings|sale|order)\b", 0.95),
    (r"\border(?:ed)?\s+liquidat(?:ed|ion)\b", 0.95),
    (r"\bwind-?down\s+(?:operations|business)\b", 0.85),
    (r"\benter(?:s|ed|ing)?\s+receivership\b", 0.95),
    (r"\bcourt-?appointed\s+receiver\b", 0.90),
    (r"\bjudicial\s+administration\b", 0.85),
    (r"\bdebtor-?in-?possession\b", 1.0),
    (r"\bdip\s+financ(?:ing|er)\b", 0.95),
    
    # Going Concern Warnings
    (r"\bgoing-?concern\s+(?:warning|doubt|uncertainty|opinion|qualification|notice)\b", 1.0),
    (r"\bsubstantial\s+doubt\s+about\s+(?:its\s+)?ability\s+to\s+continue\s+as\s+a\s+going\s+concern\b", 1.0),
    (r"\bgoing\s+concern\b", 0.85)
]

LITIGATION_FRAUD_PATTERNS = [
    # SEC / DOJ Investigations & Subpoenas
    (r"\bsec\s+investigation(?:s)?\b", 1.0),
    (r"\bsec\s+(?:probe|inquiry|subpoena|charges?|enforcement)\b", 0.95),
    (r"\bdoj\s+(?:investigation|probe|inquiry|charges?|subpoena)\b", 0.95),
    (r"\bsecurities\s+(?:and\s+exchange\s+commission|regulator)\s+(?:probe|investigat(?:e|ion)|charges?)\b", 0.95),
    (r"\bgrand\s+jury\s+subpoena(?:s)?\b", 0.95),
    (r"\bsubpoena(?:ed|s)?\s+(?:by\s+)?(?:the\s+)?(?:sec|doj|regulators?|feds?)\b", 0.95),
    (r"\bformal\s+investigation\s+by\s+(?:sec|doj|authorities)\b", 0.95),
    (r"\bcriminal\s+(?:investigation|probe|indictment|charges?)\b", 0.90),
    
    # Accounting Fraud & Financial Irregularities
    (r"\baccounting\s+fraud\b", 1.0),
    (r"\baccounting\s+irregularit(?:y|ies)\b", 0.95),
    (r"\bfinancial\s+fraud\b", 0.95),
    (r"\bsecurities\s+fraud\b", 0.95),
    (r"\bcooking\s+the\s+books\b", 0.95),
    (r"\bfalsif(?:y|ied|ication)\s+of\s+financial\s+records\b", 1.0),
    (r"\brestatement\s+of\s+(?:financial|prior)\s+results\b", 0.90),
    (r"\brestat(?:e|ed|ing)\s+(?:financials|earnings|revenue|results)\b", 0.85),
    (r"\bdelayed\s+(?:annual\s+report|10-?k|10-?q)\s+filing\b", 0.80),
    (r"\bmaterial\s+weakness(?:es)?\s+in\s+internal\s+controls?\b", 0.90),
    (r"\bauditor\s+(?:resignation|resigned|quit|fired|withdraws?)\b", 0.90),
    (r"\badverse\s+audit\s+opinion\b", 0.95),
    (r"\bwhistleblower\s+(?:allegation|complaint|claims?|report)\b", 0.85),
    (r"\binternal\s+accounting\s+probe\b", 0.85),
    
    # Class Actions & Shareholder Lawsuits
    (r"\bsecurities\s+class\s+action\b", 0.85),
    (r"\bshareholder\s+class\s+action\s+lawsuit\b", 0.80),
    (r"\bclass\s+action\s+lawsuit\b", 0.65),
    (r"\bsued\s+for\s+(?:misleading|fraudulent|deceptive)\s+investors\b", 0.85)
]


class FinancialDistressTopicExtractor:
    """
    High-performance Topic Extractor for corporate distress and bankruptcy in English financial news.
    Combines rule-weighted domain regexes with optional c-TF-IDF / semantic embeddings.
    """

    def __init__(self):
        self.default_compiled = [
            (re.compile(pat, re.IGNORECASE), weight) for pat, weight in DEFAULT_COVENANTS_PATTERNS
        ]
        self.bankruptcy_compiled = [
            (re.compile(pat, re.IGNORECASE), weight) for pat, weight in BANKRUPTCY_CHAPTER11_PATTERNS
        ]
        self.litigation_compiled = [
            (re.compile(pat, re.IGNORECASE), weight) for pat, weight in LITIGATION_FRAUD_PATTERNS
        ]

    def _score_text(self, text: str) -> Tuple[float, float, float]:
        """Scores a single string against the three distress sub-topics."""
        if not text:
            return 0.0, 0.0, 0.0
        
        # 1. Default & Covenants
        def_score = 0.0
        for regex, weight in self.default_compiled:
            if regex.search(text):
                def_score = max(def_score, weight)
                if def_score >= 1.0:
                    break
        
        # 2. Bankruptcy & Chapter 11
        bank_score = 0.0
        for regex, weight in self.bankruptcy_compiled:
            if regex.search(text):
                bank_score = max(bank_score, weight)
                if bank_score >= 1.0:
                    break
        
        # 3. Litigation & Fraud
        lit_score = 0.0
        for regex, weight in self.litigation_compiled:
            if regex.search(text):
                lit_score = max(lit_score, weight)
                if lit_score >= 1.0:
                    break
                    
        return def_score, bank_score, lit_score

    def extract_topics_batch(
        self,
        texts: List[str],
        sentiment_negs: Optional[np.ndarray] = None
    ) -> pd.DataFrame:
        """
        Extracts sub-topic intensities, composite intensity, and flags for a batch of texts.
        """
        n = len(texts)
        default_scores = np.zeros(n, dtype=np.float32)
        bankruptcy_scores = np.zeros(n, dtype=np.float32)
        litigation_scores = np.zeros(n, dtype=np.float32)
        
        for i, t in enumerate(texts):
            d_s, b_s, l_s = self._score_text(t)
            default_scores[i] = d_s
            bankruptcy_scores[i] = b_s
            litigation_scores[i] = l_s

        # Composite distress intensity:
        # Bankruptcy has highest severity weight (1.0), followed by Default (0.90), Litigation (0.80)
        raw_composite = np.maximum(
            bankruptcy_scores,
            np.maximum(default_scores * 0.90, litigation_scores * 0.80)
        )

        # Modulate with negative FinBERT sentiment if provided:
        # A distress keyword accompanied by strong negative sentiment increases confidence
        if sentiment_negs is not None:
            sentiment_negs = np.asarray(sentiment_negs, dtype=np.float32)
            # Boost score proportionally to negative sentiment for matched topics
            boost = np.where(raw_composite > 0, 1.0 + 0.30 * np.clip(sentiment_negs, 0.0, 1.0), 0.0)
            composite = np.clip(raw_composite * boost, 0.0, 1.0)
        else:
            composite = raw_composite

        # Dominant Topic Label
        dominant_topics = []
        for d, b, l, c in zip(default_scores, bankruptcy_scores, litigation_scores, composite):
            if c < 0.20:
                dominant_topics.append("none")
            elif b >= d and b >= l:
                dominant_topics.append("chapter11_bankruptcy")
            elif d >= l:
                dominant_topics.append("default_covenants")
            else:
                dominant_topics.append("litigation_fraud")

        is_distress = (composite >= 0.25).astype(np.int32)

        return pd.DataFrame({
            "distress_default_intensity": default_scores,
            "distress_chapter11_intensity": bankruptcy_scores,
            "distress_litigation_intensity": litigation_scores,
            "nlp_distress_topic_intensity": composite,
            "dominant_distress_topic": dominant_topics,
            "is_distress_news": is_distress
        })


# =====================================================================
# 2. TEMPORAL EXPONENTIAL DECAY AGGREGATION (t_1/2 = 14 DAYS)
# =====================================================================

def compute_exponential_decay_weights(delta_days: np.ndarray, half_life_days: float = DEFAULT_HALF_LIFE_DAYS) -> np.ndarray:
    """
    Computes exponential decay weights: w_k = exp(-lambda * delta_t_k),
    where lambda = ln(2) / half_life_days.
    """
    decay_lambda = math.log(2.0) / half_life_days
    # Clip negative delta days (future news relative to reference date) to zero weight
    valid_mask = delta_days >= 0
    weights = np.zeros_like(delta_days, dtype=np.float64)
    weights[valid_mask] = np.exp(-decay_lambda * delta_days[valid_mask])
    return weights


def aggregate_monthly_nlp_distress_panel(
    news_distress_parquet_path: Path,
    half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
    min_year: int = 1999,
    max_year: int = 2026
) -> pd.DataFrame:
    """
    Performs high-performance monthly panel aggregation with exponential temporal decay
    directly in DuckDB for each (ticker, month).
    """
    print(f"\n[Aggregation Engine] Running High-Performance Monthly Exponential Decay Aggregation (t_1/2 = {half_life_days:.1f} days)...")
    t0 = time.time()
    decay_lambda = math.log(2.0) / half_life_days

    con = duckdb.connect()
    
    query = f"""
    WITH news_cleaned AS (
        SELECT 
            UPPER(TRIM(ticker)) as ticker,
            COALESCE(
                TRY_CAST(news_date AS TIMESTAMP),
                TRY_STRPTIME(CAST(news_date AS VARCHAR), '%Y-%m-%d %H:%M:%S'),
                TRY_STRPTIME(CAST(news_date AS VARCHAR), '%a, %d %b %Y %H:%M:%S GMT'),
                TRY_STRPTIME(CAST(substr(news_date, 1, 10) AS VARCHAR), '%Y-%m-%d')
            ) as news_ts,
            CAST(headline_sentiment_index AS DOUBLE) as sentiment,
            CAST(finbert_neg AS DOUBLE) as finbert_neg,
            CAST(nlp_distress_topic_intensity AS DOUBLE) as distress_intensity,
            CAST(distress_default_intensity AS DOUBLE) as default_intensity,
            CAST(distress_chapter11_intensity AS DOUBLE) as chapter11_intensity,
            CAST(distress_litigation_intensity AS DOUBLE) as litigation_intensity,
            CAST(is_distress_news AS INTEGER) as is_distress
        FROM read_parquet('{news_distress_parquet_path.as_posix()}')
        WHERE ticker IS NOT NULL AND news_date IS NOT NULL
    ),
    news_valid AS (
        SELECT 
            *,
            YEAR(news_ts) as year,
            MONTH(news_ts) as month,
            STRFTIME(news_ts, '%Y-%m') as year_month,
            CAST(news_ts AS DATE) as news_d
        FROM news_cleaned
        WHERE news_ts IS NOT NULL AND YEAR(news_ts) BETWEEN {min_year} AND {max_year}
    ),
    ticker_spans AS (
        SELECT 
            ticker,
            MIN(year) as min_year,
            MAX(year) as max_year
        FROM news_valid
        GROUP BY ticker
    ),
    all_calendar_months AS (
        SELECT 
            CAST(STRFTIME(d, '%Y-%m') AS VARCHAR) as year_month,
            YEAR(d) as year,
            MONTH(d) as month,
            LAST_DAY(d) as period_end_date
        FROM (
            SELECT UNNEST(GENERATE_SERIES(DATE '{min_year}-01-01', DATE '{max_year}-12-31', INTERVAL 1 MONTH)) as d
        )
    ),
    ticker_calendar AS (
        SELECT 
            ts.ticker,
            cm.year_month,
            cm.year,
            cm.month,
            cm.period_end_date
        FROM ticker_spans ts
        CROSS JOIN all_calendar_months cm
        WHERE cm.year BETWEEN ts.min_year AND ts.max_year
    ),
    decay_30d AS (
        SELECT 
            tc.ticker,
            tc.year_month,
            COUNT(n.news_d) as news_count_30d,
            COALESCE(SUM(EXP(-{decay_lambda} * DATE_DIFF('day', n.news_d, tc.period_end_date))), 0.0) as news_effective_volume_decayed_30d,
            COALESCE(SUM(n.sentiment * EXP(-{decay_lambda} * DATE_DIFF('day', n.news_d, tc.period_end_date))) / NULLIF(SUM(EXP(-{decay_lambda} * DATE_DIFF('day', n.news_d, tc.period_end_date))), 0.0), 0.0) as nlp_sentiment_decayed_30d,
            COALESCE(SUM(n.distress_intensity * EXP(-{decay_lambda} * DATE_DIFF('day', n.news_d, tc.period_end_date))) / NULLIF(SUM(EXP(-{decay_lambda} * DATE_DIFF('day', n.news_d, tc.period_end_date))), 0.0), 0.0) as nlp_distress_topic_intensity_decayed_30d,
            COALESCE(SUM(n.default_intensity * EXP(-{decay_lambda} * DATE_DIFF('day', n.news_d, tc.period_end_date))) / NULLIF(SUM(EXP(-{decay_lambda} * DATE_DIFF('day', n.news_d, tc.period_end_date))), 0.0), 0.0) as distress_default_decayed_30d,
            COALESCE(SUM(n.chapter11_intensity * EXP(-{decay_lambda} * DATE_DIFF('day', n.news_d, tc.period_end_date))) / NULLIF(SUM(EXP(-{decay_lambda} * DATE_DIFF('day', n.news_d, tc.period_end_date))), 0.0), 0.0) as distress_chapter11_decayed_30d,
            COALESCE(SUM(n.litigation_intensity * EXP(-{decay_lambda} * DATE_DIFF('day', n.news_d, tc.period_end_date))) / NULLIF(SUM(EXP(-{decay_lambda} * DATE_DIFF('day', n.news_d, tc.period_end_date))), 0.0), 0.0) as distress_litigation_decayed_30d,
            COALESCE(SUM(n.is_distress), 0) as distress_news_count_30d,
            COALESCE(AVG(CASE WHEN n.finbert_neg > 0.40 THEN 1.0 ELSE 0.0 END), 0.0) as news_negative_ratio_30d
        FROM ticker_calendar tc
        LEFT JOIN news_valid n
            ON tc.ticker = n.ticker
            AND n.news_d <= tc.period_end_date
            AND n.news_d >= (tc.period_end_date - INTERVAL 30 DAYS)
        GROUP BY tc.ticker, tc.year_month
    ),
    decay_90d AS (
        SELECT 
            tc.ticker,
            tc.year_month,
            COUNT(n.news_d) as news_count_90d,
            COALESCE(SUM(n.sentiment * EXP(-{decay_lambda} * DATE_DIFF('day', n.news_d, tc.period_end_date))) / NULLIF(SUM(EXP(-{decay_lambda} * DATE_DIFF('day', n.news_d, tc.period_end_date))), 0.0), 0.0) as nlp_sentiment_decayed_90d,
            COALESCE(SUM(n.distress_intensity * EXP(-{decay_lambda} * DATE_DIFF('day', n.news_d, tc.period_end_date))) / NULLIF(SUM(EXP(-{decay_lambda} * DATE_DIFF('day', n.news_d, tc.period_end_date))), 0.0), 0.0) as nlp_distress_topic_intensity_decayed_90d,
            COALESCE(SUM(n.is_distress), 0) as distress_news_count_90d
        FROM ticker_calendar tc
        LEFT JOIN news_valid n
            ON tc.ticker = n.ticker
            AND n.news_d <= tc.period_end_date
            AND n.news_d >= (tc.period_end_date - INTERVAL 90 DAYS)
        GROUP BY tc.ticker, tc.year_month
    )
    SELECT 
        tc.ticker,
        tc.year_month,
        CAST(tc.period_end_date AS VARCHAR) as period_end_date,
        tc.year,
        tc.month,
        d30.news_count_30d,
        d90.news_count_90d,
        ROUND(d30.news_effective_volume_decayed_30d, 4) as news_effective_volume_decayed_30d,
        ROUND(d30.nlp_sentiment_decayed_30d, 4) as nlp_sentiment_decayed_30d,
        ROUND(d90.nlp_sentiment_decayed_90d, 4) as nlp_sentiment_decayed_90d,
        ROUND(d30.nlp_distress_topic_intensity_decayed_30d, 4) as nlp_distress_topic_intensity,
        ROUND(d30.nlp_distress_topic_intensity_decayed_30d, 4) as nlp_distress_topic_intensity_decayed_30d,
        ROUND(d30.distress_default_decayed_30d, 4) as distress_default_decayed_30d,
        ROUND(d30.distress_chapter11_decayed_30d, 4) as distress_chapter11_decayed_30d,
        ROUND(d30.distress_litigation_decayed_30d, 4) as distress_litigation_decayed_30d,
        ROUND(d90.nlp_distress_topic_intensity_decayed_90d, 4) as nlp_distress_topic_intensity_decayed_90d,
        d30.distress_news_count_30d,
        d90.distress_news_count_90d,
        ROUND(d30.news_negative_ratio_30d, 4) as news_negative_ratio_30d
    FROM ticker_calendar tc
    JOIN decay_30d d30 ON tc.ticker = d30.ticker AND tc.year_month = d30.year_month
    JOIN decay_90d d90 ON tc.ticker = d90.ticker AND tc.year_month = d90.year_month
    ORDER BY tc.ticker, tc.year_month
    """

    df_monthly = con.execute(query).df()
    con.close()
    
    print(f"  - Computed {len(df_monthly):,} monthly panel records across {df_monthly['ticker'].nunique():,} tickers in {time.time() - t0:.2f}s.")
    return df_monthly


# =====================================================================
# 3. MEDIA SILENCE TRACKING (media_silence_months_count)
# =====================================================================

def compute_media_silence_metrics(df_monthly: pd.DataFrame) -> pd.DataFrame:
    """
    Computes corporate 'Media Silence' metrics:
      - `media_silence_months_count`: Consecutive months with zero news coverage.
      - `is_media_silent_3m`: Boolean flag (1 if silence >= 3 months, else 0).
      - `is_media_silent_6m`: Boolean flag (1 if silence >= 6 months, else 0).
      - `media_coverage_ratio_12m`: Trailing 12-month ratio of active news months.
    """
    print("\n[Media Silence Engine] Computing consecutive media silence sequences and coverage ratios...")
    t0 = time.time()
    
    df_monthly = df_monthly.sort_values(["ticker", "year_month"]).reset_index(drop=True)
    
    silence_counts = np.zeros(len(df_monthly), dtype=np.int32)
    cov_12m = np.zeros(len(df_monthly), dtype=np.float32)
    
    grouped = df_monthly.groupby("ticker")
    
    for ticker, group in grouped:
        idxs = group.index.values
        counts = group["news_count_30d"].values
        n = len(counts)
        
        s_curr = 0
        s_arr = np.zeros(n, dtype=np.int32)
        c_arr = np.zeros(n, dtype=np.float32)
        
        for i in range(n):
            if counts[i] > 0:
                s_curr = 0
            else:
                s_curr += 1
            s_arr[i] = s_curr
            
            w_start = max(0, i - 11)
            c_arr[i] = np.mean(counts[w_start:i+1] > 0)
            
        silence_counts[idxs] = s_arr
        cov_12m[idxs] = c_arr

    df_monthly["media_silence_months_count"] = silence_counts
    df_monthly["is_media_silent_3m"] = (silence_counts >= 3).astype(np.int32)
    df_monthly["is_media_silent_6m"] = (silence_counts >= 6).astype(np.int32)
    df_monthly["media_coverage_ratio_12m"] = np.round(cov_12m, 4)

    print(f"  - Media silence computed in {time.time() - t0:.2f}s.")
    return df_monthly


# =====================================================================
# 4. FULL PIPELINE EXECUTION & DUCKDB PERSISTENCE
# =====================================================================

def run_bertopic_distress_pipeline(
    news_parquet_path: Path = DEFAULT_NEWS_PARQUET,
    sentiment_parquet_path: Path = DEFAULT_SENTIMENT_PARQUET,
    duckdb_path: Path = DEFAULT_DUCKDB_PATH,
    output_news_parquet: Path = OUTPUT_NEWS_DISTRESS_PARQUET,
    output_monthly_parquet: Path = OUTPUT_MONTHLY_FEATURES_PARQUET,
    sample_size: Optional[int] = None,
    half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
    batch_size: int = 100000
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Executes end-to-end Topic Modeling, Distress Intensity Scoring,
    Exponential Decay Aggregation, Media Silence Tracking, and DuckDB Export.
    """
    t_start = time.time()
    print("=" * 80)
    print("FINANCIAL DISTRESS TOPIC MODELING & TEMPORAL DECAY PIPELINE (BERTopic / c-TF-IDF)")
    print("=" * 80)
    print(f"News Source:       {news_parquet_path}")
    print(f"Sentiment Source:  {sentiment_parquet_path}")
    print(f"DuckDB Target:     {duckdb_path}")
    print(f"Half-life:         {half_life_days:.1f} days (lambda = {LAMBDA_DECAY:.6f})")
    print("=" * 80)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Ingest News & Sentiment using DuckDB
    print("\n[Step 1/5] Ingesting and aligning news with FinBERT sentiment scores via DuckDB...")
    con = duckdb.connect()
    
    # Fast timestamp parsing with fallback formats in DuckDB
    ts_expr = """
        COALESCE(
            TRY_CAST(time_published AS TIMESTAMP),
            TRY_STRPTIME(CAST(time_published AS VARCHAR), '%a, %d %b %Y %H:%M:%S GMT'),
            TRY_STRPTIME(CAST(time_published AS VARCHAR), '%a, %d %b %Y %H:%M:%S %Z'),
            TRY_STRPTIME(CAST(time_published AS VARCHAR), '%Y%m%dT%H%M%S'),
            TRY_STRPTIME(CAST(time_published AS VARCHAR), '%Y-%m-%dT%H:%M:%SZ'),
            TRY_STRPTIME(CAST(time_published AS VARCHAR), '%Y-%m-%d %H:%M:%S %Z'),
            TRY_STRPTIME(CAST(time_published AS VARCHAR), '%Y-%m-%d %H:%M:%S'),
            TRY_STRPTIME(CAST(substr(time_published, 1, 10) AS VARCHAR), '%Y-%m-%d')
        )
    """
    
    sample_clause = f"USING SAMPLE {sample_size} (RESERVOIR)" if sample_size and sample_size > 0 else ""

    query = f"""
        SELECT 
            UPPER(TRIM(n.ticker)) as ticker,
            CAST({ts_expr} AS VARCHAR) as news_date,
            n.pub_year as year,
            n.title as title,
            n.summary as summary,
            COALESCE(s.headline_sentiment_index, 0.0) as headline_sentiment_index,
            COALESCE(s.finbert_pos, 0.0) as finbert_pos,
            COALESCE(s.finbert_neg, 0.0) as finbert_neg,
            COALESCE(s.finbert_neu, 0.0) as finbert_neu
        FROM read_parquet('{news_parquet_path.as_posix()}') n
        LEFT JOIN read_parquet('{sentiment_parquet_path.as_posix()}') s
            ON UPPER(TRIM(n.ticker)) = UPPER(TRIM(s.stock_symbol))
            AND LOWER(TRIM(n.title)) = LOWER(TRIM(s.article_title))
        WHERE n.ticker IS NOT NULL AND n.title IS NOT NULL
        {sample_clause}
    """

    df_news = con.execute(query).df()
    total_news = len(df_news)
    print(f"  - Ingested {total_news:,} news records with sentiment in {time.time() - t_start:.2f}s.")

    # 2. Extract Distress Topics
    print("\n[Step 2/5] Extracting Financial Distress Topics (Default, Chapter 11, Litigation)...")
    extractor = FinancialDistressTopicExtractor()
    
    titles = df_news["title"].fillna("").astype(str).tolist()
    summaries = df_news["summary"].fillna("").astype(str).tolist()
    combined_texts = [
        f"{t}. {s}" if s and s.lower() != "nan" and s.lower() != t.lower() else t
        for t, s in zip(titles, summaries)
    ]
    
    topic_dfs = []
    t_top_start = time.time()
    for i in range(0, total_news, batch_size):
        b_texts = combined_texts[i:i + batch_size]
        b_negs = df_news["finbert_neg"].values[i:i + batch_size]
        b_topics = extractor.extract_topics_batch(b_texts, sentiment_negs=b_negs)
        topic_dfs.append(b_topics)
        
        if (i + batch_size) % (batch_size * 5) == 0 or (i + batch_size) >= total_news:
            curr_done = min(i + batch_size, total_news)
            pct = curr_done / total_news * 100.0
            elapsed = time.time() - t_top_start
            speed = curr_done / elapsed if elapsed > 0 else 0
            print(f"  - Topic Extraction Progress: {curr_done:,} / {total_news:,} ({pct:.1f}%) | {speed:,.0f} texts/s")

    df_topics = pd.concat(topic_dfs, ignore_index=True)
    
    for col in df_topics.columns:
        df_news[col] = df_topics[col].values

    distress_count = int(df_news["is_distress_news"].sum())
    print(f"  - Detected {distress_count:,} corporate distress news articles ({distress_count / total_news * 100:.2f}% of corpus).")
    print(f"  - Topic Breakdown:\n{df_news['dominant_distress_topic'].value_counts().to_string()}")

    # Export news-level distress parquet
    print(f"\n[Step 3/5] Saving article-level distress dataset: '{output_news_parquet}'...")
    df_news_export = df_news.drop(columns=["summary"], errors="ignore")
    df_news_export.to_parquet(output_news_parquet, index=False, compression="snappy")
    print(f"  - Saved ({output_news_parquet.stat().st_size / (1024*1024):.2f} MB).")

    # 3. Temporal Exponential Decay Aggregation
    print("\n[Step 4/5] Computing Temporal Exponential Decay Panel Aggregation...")
    df_monthly = aggregate_monthly_nlp_distress_panel(output_news_parquet, half_life_days=half_life_days)

    # 4. Media Silence Tracking
    print("\n[Step 5/5] Computing Media Silence Features...")
    df_monthly = compute_media_silence_metrics(df_monthly)

    # Export Monthly Features Parquet & DuckDB Table
    print(f"\n[Persistence] Saving monthly panel features to '{output_monthly_parquet}'...")
    df_monthly.to_parquet(output_monthly_parquet, index=False, compression="snappy")
    parquet_mb = output_monthly_parquet.stat().st_size / (1024 * 1024)
    print(f"  - Monthly Panel Parquet Path: {output_monthly_parquet} ({parquet_mb:.2f} MB)")

    # Ingest into DuckDB Database
    print(f"[DuckDB Persistence] Registering table 'monthly_nlp_distress_features' in '{duckdb_path}'...")
    con_db = duckdb.connect(str(duckdb_path))
    con_db.execute("SET memory_limit = '8GB';")
    con_db.register("monthly_features_df", df_monthly)
    con_db.execute("""
        CREATE OR REPLACE TABLE monthly_nlp_distress_features AS 
        SELECT * FROM monthly_features_df;
    """)
    db_count = con_db.execute("SELECT COUNT(*) FROM monthly_nlp_distress_features").fetchone()[0]
    con_db.close()
    print(f"  - Table 'monthly_nlp_distress_features' created in DuckDB ({db_count:,} rows).")

    # Summary Statistics Table
    print("\n" + "=" * 80)
    print("MONTHLY NLP DISTRESS & SENTIMENT FEATURES SUMMARY:")
    print("=" * 80)
    summary_cols = [
        "news_count_30d",
        "news_effective_volume_decayed_30d",
        "nlp_sentiment_decayed_30d",
        "nlp_distress_topic_intensity_decayed_30d",
        "distress_default_decayed_30d",
        "distress_chapter11_decayed_30d",
        "distress_litigation_decayed_30d",
        "media_silence_months_count",
        "is_media_silent_3m"
    ]
    print(df_monthly[summary_cols].describe().T.to_string())
    print("=" * 80)
    print(f"Pipeline completed successfully in {time.time() - t_start:.2f}s!")
    print("=" * 80)

    return df_news_export, df_monthly


# =====================================================================
# 5. DUCKDB INTEGRATION HELPER FOR DOWNSTREAM SEC PANEL
# =====================================================================

def integrate_nlp_distress_with_sec_panel(
    sec_parquet_path: Path,
    monthly_nlp_parquet_path: Path = OUTPUT_MONTHLY_FEATURES_PARQUET,
    output_parquet_path: Optional[Path] = None
) -> pd.DataFrame:
    """
    Seamlessly joins monthly NLP distress and sentiment features into an SEC panel dataset
    (e.g., v2_sec_financials_pivoted_clean.parquet or sec_market_quotes_monthly.parquet)
    using high-performance DuckDB point-in-time / year-month joining.
    """
    print(f"\n[DuckDB Integration] Joining NLP Distress Features with SEC Panel '{sec_parquet_path}'...")
    con = duckdb.connect()
    
    con.execute(f"CREATE OR REPLACE VIEW sec_panel AS SELECT * FROM read_parquet('{sec_parquet_path.as_posix()}');")
    con.execute(f"CREATE OR REPLACE VIEW nlp_features AS SELECT * FROM read_parquet('{monthly_nlp_parquet_path.as_posix()}');")
    
    sec_cols = [row[0] for row in con.execute("DESCRIBE sec_panel").fetchall()]
    
    if "ticker" in sec_cols:
        join_sql = """
            SELECT 
                s.*,
                COALESCE(n.nlp_sentiment_decayed_30d, 0.0) as nlp_sentiment_decayed_30d,
                COALESCE(n.nlp_sentiment_decayed_90d, 0.0) as nlp_sentiment_decayed_90d,
                COALESCE(n.nlp_distress_topic_intensity_decayed_30d, 0.0) as nlp_distress_topic_intensity,
                COALESCE(n.distress_default_decayed_30d, 0.0) as distress_default_intensity,
                COALESCE(n.distress_chapter11_decayed_30d, 0.0) as distress_chapter11_intensity,
                COALESCE(n.distress_litigation_decayed_30d, 0.0) as distress_litigation_intensity,
                COALESCE(n.news_effective_volume_decayed_30d, 0.0) as news_effective_volume_decayed,
                COALESCE(n.news_count_30d, 0) as news_count_30d,
                COALESCE(n.distress_news_count_90d, 0) as distress_news_count_90d,
                COALESCE(n.media_silence_months_count, 12) as media_silence_months_count,
                COALESCE(n.is_media_silent_3m, 1) as is_media_silent_3m,
                COALESCE(n.media_coverage_ratio_12m, 0.0) as media_coverage_ratio_12m
            FROM sec_panel s
            LEFT JOIN nlp_features n
                ON UPPER(TRIM(s.ticker)) = UPPER(TRIM(n.ticker))
                AND (
                    (TRY_CAST(s.filed_date AS VARCHAR) LIKE n.year_month || '%') OR
                    (YEAR(TRY_CAST(s.filed_date AS DATE)) = n.year AND MONTH(TRY_CAST(s.filed_date AS DATE)) = n.month)
                )
        """
    else:
        join_sql = """
            SELECT 
                s.*,
                COALESCE(n.nlp_sentiment_decayed_30d, 0.0) as nlp_sentiment_decayed_30d,
                COALESCE(n.nlp_distress_topic_intensity_decayed_30d, 0.0) as nlp_distress_topic_intensity,
                COALESCE(n.media_silence_months_count, 12) as media_silence_months_count,
                COALESCE(n.is_media_silent_3m, 1) as is_media_silent_3m
            FROM sec_panel s
            LEFT JOIN nlp_features n
                ON UPPER(TRIM(s.company_name)) = UPPER(TRIM(n.ticker))
                AND YEAR(TRY_CAST(s.filed_date AS DATE)) = n.year
                AND MONTH(TRY_CAST(s.filed_date AS DATE)) = n.month
        """

    df_integrated = con.execute(join_sql).df()
    print(f"  - Integrated dataset generated with {len(df_integrated):,} rows and {len(df_integrated.columns)} features.")

    if output_parquet_path:
        con.execute(f"COPY ({join_sql}) TO '{output_parquet_path.as_posix()}' (FORMAT PARQUET, COMPRESSION SNAPPY);")
        print(f"  - Saved integrated panel to '{output_parquet_path}'.")

    con.close()
    return df_integrated


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Financial Distress Topic Modeling & Exponential Aggregation")
    parser.add_argument("--sample_size", type=int, default=None, help="Sample size (None for full dataset)")
    parser.add_argument("--half_life", type=float, default=DEFAULT_HALF_LIFE_DAYS, help="Decay half-life in days")
    parser.add_argument("--batch_size", type=int, default=100000, help="Batch size for topic extraction")
    args = parser.parse_args()

    run_bertopic_distress_pipeline(
        sample_size=args.sample_size,
        half_life_days=args.half_life,
        batch_size=args.batch_size
    )
