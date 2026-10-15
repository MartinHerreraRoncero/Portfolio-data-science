import os
import sys
import duckdb
import pandas as pd
from pathlib import Path

FNSPID_CSV_PATH = Path("data/raw/FNSPID/fnspid_financial_news.csv")
RECENT_NEWS_PARQUET_PATH = Path("data/raw/news_api/all_financial_news.parquet")

OUTPUT_DIR = Path("data/processed/news_dataset")
CONSOLIDATED_PARQUET_PATH = OUTPUT_DIR / "consolidated_financial_news_1999_2026.parquet"
CONSOLIDATED_DUCKDB_PATH = OUTPUT_DIR / "consolidated_news.duckdb"


def merge_news_datasets():
    """
    Merges historical FNSPID news dataset (1999-2020) with recent News API dataset (2021-2026)
    into a unified high-performance Parquet & DuckDB database.
    """
    print("=" * 60)
    print("MERGING HISTORICAL FNSPID (1999-2020) & RECENT NEWS (2021-2026)")
    print("=" * 60)

    if not FNSPID_CSV_PATH.exists():
        print(f"[Error] FNSPID CSV not found at '{FNSPID_CSV_PATH}'.")
        return

    if not RECENT_NEWS_PARQUET_PATH.exists():
        print(f"[Error] Recent News Parquet not found at '{RECENT_NEWS_PARQUET_PATH}'.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[Merge Engine] Connecting to DuckDB database at '{CONSOLIDATED_DUCKDB_PATH}'...")
    con = duckdb.connect(str(CONSOLIDATED_DUCKDB_PATH))
    con.execute("SET memory_limit = '8GB';")
    con.execute("SET preserve_insertion_order = false;")

    # 1. Ingest historical FNSPID dataset (1999 to 2020)
    print(f"[Merge Engine] Processing historical FNSPID dataset (1999-2020) from '{FNSPID_CSV_PATH}'...")
    con.execute("""
    CREATE OR REPLACE TABLE fnspid_historical AS
    SELECT 
        UPPER(TRIM(stock_symbol)) as ticker,
        CAST(NULL AS VARCHAR) as company_name,
        TRIM(article_title) as title,
        CAST(NULL AS VARCHAR) as url,
        date as time_published,
        YEAR(COALESCE(
            TRY_CAST(date AS TIMESTAMP),
            TRY_STRPTIME(CAST(date AS VARCHAR), '%Y-%m-%d %H:%M:%S %Z'),
            TRY_STRPTIME(CAST(substr(date, 1, 10) AS VARCHAR), '%Y-%m-%d')
        )) as pub_year,
        TRIM(publisher) as authors,
        TRIM(article_title) as summary,
        'FNSPID' as source_dataset
    FROM read_csv_auto($1, ignore_errors=True, header=True, all_varchar=True)
    WHERE stock_symbol IS NOT NULL 
      AND LENGTH(TRIM(stock_symbol)) > 0
      AND article_title IS NOT NULL;
    """, [str(FNSPID_CSV_PATH)])

    fnspid_count = con.execute("SELECT COUNT(*) FROM fnspid_historical WHERE pub_year BETWEEN 1999 AND 2020").fetchone()[0]
    print(f"  - Historical FNSPID valid records (1999-2020): {fnspid_count:,}")

    # 2. Ingest recent News API dataset (2021 to 2026)
    print(f"[Merge Engine] Processing recent News datasets (2021-2026)...")
    hist_2021_dir = Path("data/raw/news_historical_2021_2024")
    
    csv_glob = str(hist_2021_dir / "news_*.csv") if hist_2021_dir.exists() and list(hist_2021_dir.glob("news_*.csv")) else None

    if csv_glob:
        con.execute("""
        CREATE OR REPLACE TABLE recent_news AS
        SELECT * FROM (
            SELECT 
                UPPER(TRIM(ticker)) as ticker,
                company_name,
                TRIM(title) as title,
                url,
                time_published,
                YEAR(COALESCE(
                    TRY_CAST(time_published AS TIMESTAMP),
                    TRY_STRPTIME(CAST(time_published AS VARCHAR), '%a, %d %b %Y %H:%M:%S GMT'),
                    TRY_STRPTIME(CAST(time_published AS VARCHAR), '%a, %d %b %Y %H:%M:%S %Z'),
                    TRY_STRPTIME(CAST(time_published AS VARCHAR), '%Y%m%dT%H%M%S'),
                    TRY_STRPTIME(CAST(time_published AS VARCHAR), '%Y-%m-%dT%H:%M:%SZ'),
                    TRY_STRPTIME(CAST(time_published AS VARCHAR), '%Y-%m-%d %H:%M:%S %Z')
                )) as pub_year,
                authors,
                summary,
                'NewsAPI_Google_yfinance' as source_dataset
            FROM read_parquet($1)
            WHERE ticker IS NOT NULL AND title IS NOT NULL

            UNION ALL

            SELECT 
                UPPER(TRIM(ticker)) as ticker,
                company_name,
                TRIM(title) as title,
                url,
                time_published,
                YEAR(COALESCE(
                    TRY_CAST(time_published AS TIMESTAMP),
                    TRY_STRPTIME(CAST(time_published AS VARCHAR), '%a, %d %b %Y %H:%M:%S GMT'),
                    TRY_STRPTIME(CAST(time_published AS VARCHAR), '%a, %d %b %Y %H:%M:%S %Z'),
                    TRY_STRPTIME(CAST(time_published AS VARCHAR), '%Y%m%dT%H%M%S'),
                    TRY_STRPTIME(CAST(time_published AS VARCHAR), '%Y-%m-%dT%H:%M:%SZ'),
                    TRY_STRPTIME(CAST(time_published AS VARCHAR), '%Y-%m-%d %H:%M:%S %Z')
                )) as pub_year,
                authors,
                summary,
                'GoogleNewsRSS_Targeted' as source_dataset
            FROM read_csv_auto($2, ignore_errors=True, header=True, all_varchar=True, union_by_name=True)
            WHERE ticker IS NOT NULL AND title IS NOT NULL
        );
        """, [str(RECENT_NEWS_PARQUET_PATH), csv_glob])
    else:
        con.execute("""
        CREATE OR REPLACE TABLE recent_news AS
        SELECT 
            UPPER(TRIM(ticker)) as ticker,
            company_name,
            TRIM(title) as title,
            url,
            time_published,
            YEAR(COALESCE(
                TRY_CAST(time_published AS TIMESTAMP),
                TRY_STRPTIME(CAST(time_published AS VARCHAR), '%a, %d %b %Y %H:%M:%S GMT'),
                TRY_STRPTIME(CAST(time_published AS VARCHAR), '%a, %d %b %Y %H:%M:%S %Z'),
                TRY_STRPTIME(CAST(time_published AS VARCHAR), '%Y%m%dT%H%M%S'),
                TRY_STRPTIME(CAST(time_published AS VARCHAR), '%Y-%m-%dT%H:%M:%SZ'),
                TRY_STRPTIME(CAST(time_published AS VARCHAR), '%Y-%m-%d %H:%M:%S %Z')
            )) as pub_year,
            authors,
            summary,
            'NewsAPI_Google_yfinance' as source_dataset
        FROM read_parquet($1)
        WHERE ticker IS NOT NULL AND title IS NOT NULL;
        """, [str(RECENT_NEWS_PARQUET_PATH)])

    recent_count = con.execute("SELECT COUNT(*) FROM recent_news WHERE pub_year >= 2021 OR pub_year IS NULL").fetchone()[0]
    print(f"  - Recent News valid records (2021-2026): {recent_count:,}")

    # 3. Union and Deduplicate
    print("[Merge Engine] Combining and deduplicating news articles by (ticker, title, pub_year)...")
    con.execute("""
    CREATE OR REPLACE TABLE consolidated_financial_news AS
    SELECT 
        ticker,
        company_name,
        title,
        url,
        time_published,
        pub_year,
        authors,
        summary,
        source_dataset
    FROM (
        SELECT *,
            ROW_NUMBER() OVER(PARTITION BY ticker, LOWER(TRIM(title)), pub_year ORDER BY time_published DESC) as rn
        FROM (
            SELECT * FROM fnspid_historical WHERE pub_year BETWEEN 1999 AND 2020
            UNION ALL
            SELECT * FROM recent_news WHERE pub_year >= 2021 OR pub_year IS NULL
        )
    )
    WHERE rn = 1;
    """)

    total_consolidated = con.execute("SELECT COUNT(*) FROM consolidated_financial_news").fetchone()[0]
    unique_tickers = con.execute("SELECT COUNT(DISTINCT ticker) FROM consolidated_financial_news").fetchone()[0]

    # 4. Export to Parquet
    print(f"[Merge Engine] Exporting master dataset to Parquet: '{CONSOLIDATED_PARQUET_PATH}'...")
    con.execute(f"COPY consolidated_financial_news TO '{CONSOLIDATED_PARQUET_PATH}' (FORMAT PARQUET, COMPRESSION SNAPPY);")
    
    parquet_mb = CONSOLIDATED_PARQUET_PATH.stat().st_size / (1024 * 1024)

    # 5. Yearly Summary
    yearly_df = con.execute("""
        SELECT 
            pub_year as year,
            source_dataset,
            COUNT(*) as total_news,
            COUNT(DISTINCT ticker) as unique_tickers
        FROM consolidated_financial_news
        GROUP BY year, source_dataset
        ORDER BY year ASC
    """).df()

    print("\n" + "=" * 65)
    print("CONSOLIDATED NEWS DATASET SUMMARY (1999-2026):")
    print("=" * 65)
    print(yearly_df.to_string(index=False))
    print("=" * 65)
    print(f"  - Total Consolidated Articles: {total_consolidated:,}")
    print(f"  - Total Unique Tickers Covered: {unique_tickers:,}")
    print(f"  - Master Parquet Path: {CONSOLIDATED_PARQUET_PATH} ({parquet_mb:.2f} MB)")
    print(f"  - Master DuckDB Path: {CONSOLIDATED_DUCKDB_PATH}")
    print("=" * 65)

    con.close()
    print("[Merge Engine] Consolidation completed successfully!")


if __name__ == "__main__":
    merge_news_datasets()
