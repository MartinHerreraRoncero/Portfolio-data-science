import os
import sys
import duckdb
import pandas as pd
import numpy as np
from pathlib import Path

# Paths
INPUT_QUOTES_PARQUET = Path("data/raw/market_quotes/stock_prices_2009_2026.parquet")
SEC_PIVOTED_PARQUET = Path("data/processed/sec_dataset/v2_sec_financials_pivoted.parquet")
CIK_TICKER_MAP_PARQUET = Path("data/raw/sec_metadata/cik_ticker_map.parquet")

OUTPUT_DIR = Path("data/processed/sec_dataset")
PROCESSED_QUOTES_PARQUET = OUTPUT_DIR / "sec_market_quotes_monthly.parquet"


def process_market_quotes_and_features():
    print("=" * 65)
    print("PROCESSING MARKET QUOTES & DERIVING EQUITY/MERTON FEATURES")
    print("=" * 65)

    if not INPUT_QUOTES_PARQUET.exists():
        print(f"[Error] Raw market quotes Parquet not found at '{INPUT_QUOTES_PARQUET}'!")
        return

    con = duckdb.connect()
    con.execute("SET memory_limit = '8GB';")

    # 1. Calculate Daily Returns and 30-day Rolling Volatility per Ticker
    print("[Market Engine] Calculating daily stock returns, 30d annualized volatility & momentum...")
    con.execute(f"""
    CREATE OR REPLACE TABLE stock_quotes_features AS
    SELECT 
        ticker,
        TRY_CAST(trading_date AS DATE) as trading_date,
        close_price as stock_price_close,
        volume,
        -- Daily Return
        (close_price - LAG(close_price, 1) OVER (PARTITION BY ticker ORDER BY trading_date)) / 
            NULLIF(LAG(close_price, 1) OVER (PARTITION BY ticker ORDER BY trading_date), 0.0) as daily_return,
        -- Returns 1m (21 trading days), 3m (63 trading days), 12m (252 trading days)
        (close_price - LAG(close_price, 21) OVER (PARTITION BY ticker ORDER BY trading_date)) / 
            NULLIF(LAG(close_price, 21) OVER (PARTITION BY ticker ORDER BY trading_date), 0.0) as stock_return_1m,
        (close_price - LAG(close_price, 63) OVER (PARTITION BY ticker ORDER BY trading_date)) / 
            NULLIF(LAG(close_price, 63) OVER (PARTITION BY ticker ORDER BY trading_date), 0.0) as stock_return_3m,
        (close_price - LAG(close_price, 252) OVER (PARTITION BY ticker ORDER BY trading_date)) / 
            NULLIF(LAG(close_price, 252) OVER (PARTITION BY ticker ORDER BY trading_date), 0.0) as stock_return_12m
    FROM '{INPUT_QUOTES_PARQUET}';
    """)

    print("[Market Engine] Calculating 30-day annualized rolling volatility (sigma_equity)...")
    con.execute("""
    CREATE OR REPLACE TABLE stock_quotes_volatility AS
    SELECT *,
        COALESCE(STDDEV(daily_return) OVER (
            PARTITION BY ticker ORDER BY trading_date ROWS BETWEEN 20 PRECEDING AND CURRENT ROW
        ) * SQRT(252), 0.20) as stock_volatility_30d
    FROM stock_quotes_features;
    """)

    # 2. CIK <-> Ticker Map
    con.execute("""
    CREATE OR REPLACE TABLE cik_ticker_map AS
    SELECT DISTINCT 
        CAST(ISSUERCIK AS INTEGER) as cik, 
        UPPER(TRIM(ISSUERTRADINGSYMBOL)) as ticker
    FROM read_csv_auto($1, ignore_errors=True, header=True, all_varchar=True)
    WHERE ISSUERTRADINGSYMBOL IS NOT NULL AND ISSUERCIK IS NOT NULL;
    """, [str(SUBMISSION_TSV)])

    # 3. ASOF JOIN: Match SEC filing date with the exact latest stock quote available on or before filed_date
    print("[Market Engine] Executing ASOF JOIN between SEC filings and daily stock quotes...")
    con.execute(f"""
    CREATE OR REPLACE TABLE sec_market_matched AS
    SELECT 
        f.adsh,
        f.cik,
        TRY_CAST(f.filed_date AS DATE) as filed_date,
        f.filed_year,
        EXTRACT(month FROM TRY_CAST(f.filed_date AS DATE))::INTEGER as filed_month,
        f.sic,
        (f.sic // 100)::INTEGER as sic_2digit,
        f.tag_Assets,
        f.tag_StockholdersEquity,
        f.tag_LiabilitiesCurrent,
        f.tag_EntityCommonStockSharesOutstanding,
        f.tag_CommonStockSharesOutstanding,
        f.tag_EntityPublicFloat,
        m.ticker,
        q.stock_price_close,
        q.stock_return_1m,
        q.stock_return_3m,
        q.stock_return_12m,
        q.stock_volatility_30d,
        q.volume as stock_volume
    FROM '{SEC_PIVOTED_PARQUET}' f
    LEFT JOIN cik_ticker_map m ON f.cik = m.cik
    ASOF LEFT JOIN stock_quotes_volatility q
      ON m.ticker = q.ticker AND TRY_CAST(f.filed_date AS DATE) >= q.trading_date;
    """)

    # 4. Calculate Market Cap, Price-to-Book & Merton Distance to Default Proxy
    print("[Market Engine] Deriving Market Cap & Corrected Merton Distance to Default Proxy (No Double Counting)...")
    con.execute("""
    CREATE OR REPLACE TABLE sec_market_quotes_monthly AS
    WITH market_cap_calc AS (
        SELECT 
            adsh,
            cik,
            filed_date,
            filed_year,
            filed_month,
            sic,
            sic_2digit,
            ticker,
            COALESCE(stock_price_close, 0.0) as stock_price_close,
            COALESCE(stock_return_1m, 0.0) as stock_return_1m,
            COALESCE(stock_return_3m, 0.0) as stock_return_3m,
            COALESCE(stock_return_12m, 0.0) as stock_return_12m,
            COALESCE(stock_volatility_30d, 0.0) as stock_volatility_30d,
            COALESCE(stock_volume, 0) as stock_volume,
            tag_Assets,
            tag_StockholdersEquity,
            tag_LiabilitiesCurrent,
            -- Market Capitalization from Price * Shares, with fallback to SEC Public Float
            COALESCE(
                NULLIF(stock_price_close * COALESCE(tag_EntityCommonStockSharesOutstanding, tag_CommonStockSharesOutstanding), 0.0),
                tag_EntityPublicFloat,
                0.0
            ) as market_cap,
            -- Market-wide average volatility as dynamic fallback
            AVG(NULLIF(stock_volatility_30d, 0.0)) OVER (PARTITION BY filed_year, filed_month) as market_avg_volatility,
            -- Robust sector and market median debt benchmarks for null / zero liabilities fallback
            MEDIAN(NULLIF(GREATEST(tag_LiabilitiesCurrent, 0.0), 0.0)) OVER (PARTITION BY filed_year, sic_2digit) as med_debt_sector,
            MEDIAN(NULLIF(GREATEST(tag_LiabilitiesCurrent, 0.0), 0.0)) OVER (PARTITION BY filed_year) as med_debt_market,
            MEDIAN(NULLIF(GREATEST(tag_LiabilitiesCurrent, 0.0), 0.0)) OVER () as med_debt_global,
            -- Sector median debt-to-assets ratio for scaling when firm assets are reported
            MEDIAN(CASE WHEN tag_LiabilitiesCurrent > 0 AND tag_Assets > 0 THEN tag_LiabilitiesCurrent / tag_Assets END) OVER (PARTITION BY filed_year, sic_2digit) as med_debt_to_assets_sector
        FROM sec_market_matched
    ),
    merton_calc AS (
        SELECT *,
            -- Price to Book (Market Cap / Stockholders Equity)
            CASE 
                WHEN tag_StockholdersEquity IS NOT NULL AND tag_StockholdersEquity > 0 AND market_cap > 0
                THEN market_cap / tag_StockholdersEquity 
                ELSE 1.0 
            END as price_to_book,
            -- Total Debt Face Value D: when tag_LiabilitiesCurrent is null or <= 0,
            -- use NULLIF and fallback to sector/market median or reasonable asset proportion
            -- to avoid forcing 1e-3 (which inflates +19.7 sigma the distance to default)
            COALESCE(
                NULLIF(GREATEST(tag_LiabilitiesCurrent, 0.0), 0.0),
                CASE 
                    WHEN tag_Assets IS NOT NULL AND tag_Assets > 0 
                    THEN tag_Assets * COALESCE(med_debt_to_assets_sector, 0.25)
                    ELSE NULL 
                END,
                med_debt_sector,
                med_debt_market,
                med_debt_global,
                10000.0
            ) as debt_face_value,
            -- Total Firm Asset Value V = Total Debt + Market Cap (No double-counting Equity)
            GREATEST(
                COALESCE(
                    NULLIF(GREATEST(tag_LiabilitiesCurrent, 0.0), 0.0),
                    CASE 
                        WHEN tag_Assets IS NOT NULL AND tag_Assets > 0 
                        THEN tag_Assets * COALESCE(med_debt_to_assets_sector, 0.25)
                        ELSE NULL 
                    END,
                    med_debt_sector,
                    med_debt_market,
                    med_debt_global,
                    10000.0
                ) + market_cap,
                COALESCE(tag_Assets, 0.0),
                10000.0
            ) as firm_asset_value,
            -- Robust Asset Volatility (sigma_A) using dynamic market/sector fallback
            GREATEST(
                COALESCE(NULLIF(stock_volatility_30d, 0.0), market_avg_volatility, 0.35),
                0.05
            ) as sigma_asset
        FROM market_cap_calc
    )
    SELECT 
        adsh,
        cik,
        filed_date,
        filed_year,
        filed_month,
        ticker,
        stock_price_close,
        stock_return_1m,
        stock_return_3m,
        stock_return_12m,
        stock_volatility_30d,
        stock_volume,
        market_cap,
        price_to_book,
        -- Structural Merton Distance to Default: DD = (ln(V/D) + (mu - 0.5*sigma^2)*T) / (sigma*sqrt(T))
        -- T = 1.0 year, mu = 0.05 (risk-free / expected drift)
        CASE 
            WHEN firm_asset_value > 0 AND debt_face_value > 0
            THEN (
                LN(firm_asset_value / debt_face_value) + (0.05 - 0.5 * POW(sigma_asset, 2))
            ) / (sigma_asset)
            ELSE 0.0
        END as merton_distance_to_default
    FROM merton_calc;
    """)

    # Export Processed Market Quotes
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    con.execute(f"COPY sec_market_quotes_monthly TO '{PROCESSED_QUOTES_PARQUET}' (FORMAT PARQUET, COMPRESSION SNAPPY);")
    
    parquet_mb = PROCESSED_QUOTES_PARQUET.stat().st_size / (1024 * 1024)
    total_records = con.execute("SELECT COUNT(*) FROM sec_market_quotes_monthly").fetchone()[0]
    matched_quotes = con.execute("SELECT COUNT(*) FROM sec_market_quotes_monthly WHERE stock_price_close > 0").fetchone()[0]

    print("\n" + "=" * 65)
    print("PROCESSED MARKET QUOTES SUMMARY:")
    print("=" * 65)
    print(f"  - Total SEC Submissions: {total_records:,}")
    print(f"  - Matched Submissions with Market Quotes: {matched_quotes:,} ({(matched_quotes/total_records)*100:.2f}%)")
    print(f"  - Derived Features: stock_price_close, stock_return_1m/3m/12m, stock_volatility_30d, market_cap, price_to_book, merton_distance_to_default")
    print(f"  - Output Parquet Path: {PROCESSED_QUOTES_PARQUET} ({parquet_mb:.2f} MB)")
    print("=" * 65)

    con.close()
    print("[Market Engine] Stock quotes feature pipeline completed successfully!")


if __name__ == "__main__":
    process_market_quotes_and_features()
