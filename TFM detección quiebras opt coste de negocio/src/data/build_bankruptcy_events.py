import os
import sys
import duckdb
import pandas as pd
from pathlib import Path

# Paths
NEWS_PARQUET = Path("data/processed/news_dataset/consolidated_financial_news_1999_2026.parquet")
QUOTES_PARQUET = Path("data/raw/market_quotes/stock_prices_2009_2026.parquet")
CIK_TICKER_MAP_PARQUET = Path("data/raw/sec_metadata/cik_ticker_map.parquet")
SUBMISSION_TSV = Path("scratch/sample_insider_extracted/SUBMISSION.tsv")
SEC_PIVOTED_PARQUET = Path("data/processed/sec_dataset/v2_sec_financials_pivoted.parquet")

OUTPUT_DIR = Path("data/processed/sec_dataset")
BANKRUPT_EVENTS_PARQUET = OUTPUT_DIR / "sec_bankruptcy_events.parquet"


def build_bankruptcy_events():
    print("=" * 65)
    print("BUILDING CORPORATE BANKRUPTCY EVENTS DATASET (2009-2026)")
    print("=" * 65)

    con = duckdb.connect()

    # 1. CIK <-> Ticker Mapping
    print("[Bankruptcy Engine] Loading CIK to Ticker map...")
    if CIK_TICKER_MAP_PARQUET.exists():
        con.execute(f"CREATE OR REPLACE TABLE cik_ticker_map AS SELECT cik, ticker FROM '{CIK_TICKER_MAP_PARQUET}';")
    else:
        con.execute("""
        CREATE OR REPLACE TABLE cik_ticker_map AS
        SELECT DISTINCT 
            CAST(ISSUERCIK AS INTEGER) as cik, 
            UPPER(TRIM(ISSUERTRADINGSYMBOL)) as ticker
        FROM read_csv_auto($1, ignore_errors=True, header=True, all_varchar=True)
        WHERE ISSUERTRADINGSYMBOL IS NOT NULL AND ISSUERCIK IS NOT NULL;
        """, [str(SUBMISSION_TSV)])

    # 2. Extract Bankruptcy Events from News Data (Chapter 11 / Bankruptcy Headlines)
    print("[Bankruptcy Engine] Extracting bankruptcy events from NLP news dataset...")
    con.execute(f"""
    CREATE OR REPLACE TABLE news_bankruptcy_events AS
    SELECT 
        UPPER(TRIM(ticker)) as ticker,
        TRY_CAST(COALESCE(
            TRY_CAST(time_published AS TIMESTAMP),
            TRY_STRPTIME(CAST(time_published AS VARCHAR), '%a, %d %b %Y %H:%M:%S GMT'),
            TRY_STRPTIME(CAST(time_published AS VARCHAR), '%Y-%m-%d %H:%M:%S %Z'),
            TRY_STRPTIME(CAST(time_published AS VARCHAR), '%Y-%m-%d %H:%M:%S')
        ) AS DATE) as bankruptcy_date,
        'News_Chapter11_Headline' as bankruptcy_source,
        title as bankruptcy_headline
    FROM '{NEWS_PARQUET}'
    WHERE (
        LOWER(title) LIKE '%filed for chapter 11%' OR 
        LOWER(title) LIKE '%filed for bankruptcy%' OR 
        LOWER(title) LIKE '%chapter 11 bankruptcy%' OR
        LOWER(title) LIKE '%files for chapter 11%' OR
        LOWER(title) LIKE '%files for bankruptcy%' OR
        LOWER(title) LIKE '%filed chapter 11%' OR
        LOWER(title) LIKE '%chapter 7 liquidation%' OR
        LOWER(title) LIKE '%bankruptcy protection%'
    ) AND ticker IS NOT NULL;
    """)

    # 3. Extract Delisting Bankruptcy Events from Market Quotes (Tickers ending in 'Q')
    print("[Bankruptcy Engine] Extracting delisting bankruptcy events from Stock Quotes (Suffix Q)...")
    if QUOTES_PARQUET.exists():
        con.execute(f"""
        CREATE OR REPLACE TABLE quotes_bankruptcy_events AS
        SELECT 
            UPPER(TRIM(REGEXP_REPLACE(ticker, 'Q$', ''))) as ticker,
            MIN(TRY_CAST(trading_date AS DATE)) as bankruptcy_date,
            'FINRA_Delisting_Suffix_Q' as bankruptcy_source,
            CONCAT('Delisted Ticker Suffix Q: ', ANY_VALUE(ticker)) as bankruptcy_headline
        FROM '{QUOTES_PARQUET}'
        WHERE ticker LIKE '%Q' AND LENGTH(ticker) >= 4
        GROUP BY UPPER(TRIM(REGEXP_REPLACE(ticker, 'Q$', '')));
        """)
    else:
        con.execute("CREATE OR REPLACE TABLE quotes_bankruptcy_events AS SELECT NULL as ticker, NULL as bankruptcy_date, NULL as bankruptcy_source, NULL as bankruptcy_headline WHERE 1=0;")

    # 4. Consolidate Master Bankruptcy Events Dataset per CIK
    print("[Bankruptcy Engine] Consolidating Master Bankruptcy Events Dataset per CIK...")
    con.execute("""
    CREATE OR REPLACE TABLE master_bankruptcy_events AS
    SELECT 
        m.cik,
        e.ticker,
        MIN(e.bankruptcy_date) as bankruptcy_date,
        FIRST(e.bankruptcy_source) as bankruptcy_source,
        FIRST(e.bankruptcy_headline) as bankruptcy_headline
    FROM (
        SELECT ticker, bankruptcy_date, bankruptcy_source, bankruptcy_headline FROM news_bankruptcy_events
        UNION ALL
        SELECT ticker, bankruptcy_date, bankruptcy_source, bankruptcy_headline FROM quotes_bankruptcy_events
    ) e
    JOIN cik_ticker_map m ON e.ticker = m.ticker
    WHERE e.bankruptcy_date IS NOT NULL
    GROUP BY m.cik, e.ticker;
    """)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    con.execute(f"COPY master_bankruptcy_events TO '{BANKRUPT_EVENTS_PARQUET}' (FORMAT PARQUET, COMPRESSION SNAPPY);")

    total_events = con.execute("SELECT COUNT(*) FROM master_bankruptcy_events").fetchone()[0]
    unique_ciks = con.execute("SELECT COUNT(DISTINCT cik) FROM master_bankruptcy_events").fetchone()[0]

    print("\n" + "=" * 65)
    print("BANKRUPTCY EVENTS SUMMARY:")
    print("=" * 65)
    print(f"  - Total Bankrupt CIKs Identified: {unique_ciks:,}")
    print(f"  - Total Bankruptcy Events: {total_events:,}")
    print(f"  - Output Parquet Path: {BANKRUPT_EVENTS_PARQUET}")
    print("=" * 65)

    # Display sample
    df_sample = con.execute("SELECT cik, ticker, bankruptcy_date, bankruptcy_source FROM master_bankruptcy_events LIMIT 10").df()
    print(df_sample.to_string(index=False))

    con.close()
    print("[Bankruptcy Engine] Dataset construction finished successfully!")


if __name__ == "__main__":
    build_bankruptcy_events()
