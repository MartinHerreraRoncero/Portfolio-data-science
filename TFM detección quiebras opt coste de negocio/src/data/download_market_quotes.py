import os
import sys
import time
import duckdb
import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path
from tqdm import tqdm

OUTPUT_DIR = Path("data/raw/market_quotes")
DAILY_QUOTES_PARQUET = OUTPUT_DIR / "stock_prices_2009_2026.parquet"

SEC_PIVOTED_PARQUET = Path("data/processed/sec_dataset/v2_sec_financials_pivoted.parquet")
CIK_TICKER_MAP_PARQUET = Path("data/raw/sec_metadata/cik_ticker_map.parquet")
SUBMISSION_TSV = Path("scratch/sample_insider_extracted/SUBMISSION.tsv")

BATCH_SIZE = 100  # Chunks of 100 tickers per bulk HTTP request
PAUSE_BETWEEN_BATCHES = 2.0  # 2 seconds pause between batches


def get_all_tickers():
    """Extracts all unique tickers from SEC dataset and CIK-Ticker map."""
    print("[Market Quotes] Fetching unique ticker symbols from SEC dataset...")
    tickers = set()
    con = duckdb.connect()

    # 1. From CIK-Ticker map (persistent parquet) or SUBMISSION.tsv fallback
    if CIK_TICKER_MAP_PARQUET.exists():
        df_sub = con.execute(f"SELECT DISTINCT ticker FROM '{CIK_TICKER_MAP_PARQUET}' WHERE ticker IS NOT NULL AND TRIM(ticker) != '';").df()
        tickers.update(df_sub['ticker'].tolist())
    elif SUBMISSION_TSV.exists():
        df_sub = con.execute("""
            SELECT DISTINCT UPPER(TRIM(ISSUERTRADINGSYMBOL)) as ticker
            FROM read_csv_auto($1, ignore_errors=True, header=True, all_varchar=True)
            WHERE ISSUERTRADINGSYMBOL IS NOT NULL AND TRIM(ISSUERTRADINGSYMBOL) != '';
        """, [str(SUBMISSION_TSV)]).df()
        tickers.update(df_sub['ticker'].tolist())

    # 2. From consolidated news dataset
    news_parquet = Path("data/processed/news_dataset/consolidated_financial_news_1999_2026.parquet")
    if news_parquet.exists():
        df_news = con.execute("""
            SELECT DISTINCT UPPER(TRIM(ticker)) as ticker
            FROM read_parquet($1)
            WHERE ticker IS NOT NULL AND TRIM(ticker) != '';
        """, [str(news_parquet)]).df()
        tickers.update(df_news['ticker'].tolist())

    con.close()

    # Filter valid ticker strings (alphanumeric or with dot/dash, max len 7)
    valid_tickers = sorted([t for t in tickers if t and len(t) <= 7 and (t.isalnum() or '.' in t or '-' in t)])
    print(f"  - Total unique valid tickers identified: {len(valid_tickers):,}")
    return valid_tickers


def download_batch_yfinance(batch_tickers):
    """Downloads historical daily quotes for a batch of tickers using yfinance bulk download."""
    try:
        # Convert tickers with dot to dash for yfinance (e.g. BRK.A -> BRK-A)
        yf_tickers = [t.replace('.', '-') for t in batch_tickers]
        ticker_map = {t.replace('.', '-'): t for t in batch_tickers}

        df_batch = yf.download(
            yf_tickers, 
            start="2008-01-01", 
            end="2026-12-31", 
            group_by="ticker", 
            threads=True, 
            auto_adjust=True,
            progress=False
        )

        if df_batch.empty:
            return None

        records = []
        
        # If single ticker in batch
        if len(yf_tickers) == 1:
            t = yf_tickers[0]
            orig_t = ticker_map.get(t, t)
            df_single = df_batch.reset_index()
            if 'Close' in df_single.columns and 'Date' in df_single.columns:
                sub = df_single[['Date', 'Close', 'Volume']].dropna(subset=['Close']).copy()
                sub['ticker'] = orig_t
                sub['trading_date'] = pd.to_datetime(sub['Date']).dt.strftime('%Y-%m-%d')
                sub = sub.rename(columns={'Close': 'close_price', 'Volume': 'volume'})
                return sub[['trading_date', 'ticker', 'close_price', 'volume']]
            return None

        # Multi-ticker DataFrame
        for t in yf_tickers:
            orig_t = ticker_map.get(t, t)
            if t in df_batch.columns.levels[0]:
                sub = df_batch[t].reset_index()
                if 'Close' in sub.columns and 'Date' in sub.columns:
                    clean_sub = sub[['Date', 'Close', 'Volume']].dropna(subset=['Close']).copy()
                    if not clean_sub.empty:
                        clean_sub['ticker'] = orig_t
                        clean_sub['trading_date'] = pd.to_datetime(clean_sub['Date']).dt.strftime('%Y-%m-%d')
                        clean_sub = clean_sub.rename(columns={'Close': 'close_price', 'Volume': 'volume'})
                        clean_sub['close_price'] = pd.to_numeric(clean_sub['close_price'], errors='coerce')
                        clean_sub['volume'] = pd.to_numeric(clean_sub['volume'], errors='coerce')
                        records.append(clean_sub[['trading_date', 'ticker', 'close_price', 'volume']].dropna(subset=['close_price']))

        if records:
            return pd.concat(records, ignore_index=True)

    except Exception:
        pass
    return None


def download_market_quotes_batch():
    print("=" * 65)
    print("BULK BATCH DOWNLOAD: DAILY STOCK QUOTES (2008-2026) VIA YFINANCE")
    print("=" * 65)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_tickers = get_all_tickers()

    if not all_tickers:
        print("[Error] No tickers found to process!")
        return

    # Divide tickers into batches of 100
    num_batches = int(np.ceil(len(all_tickers) / BATCH_SIZE))
    print(f"\n[yfinance Batch Engine] Processing {len(all_tickers):,} tickers across {num_batches} batches (100 tickers/batch)...")

    rescued_dfs = []

    for b_idx in tqdm(range(num_batches), desc="Downloading Quotes Batches", unit="batch"):
        batch = all_tickers[b_idx * BATCH_SIZE : (b_idx + 1) * BATCH_SIZE]
        df_res = download_batch_yfinance(batch)
        if df_res is not None and not df_res.empty:
            rescued_dfs.append(df_res)

        time.sleep(PAUSE_BETWEEN_BATCHES)

    if not rescued_dfs:
        print("[Warning] No market quotes downloaded.")
        return

    print("\n[Processing] Combining downloaded daily quotes...")
    df_all = pd.concat(rescued_dfs, ignore_index=True)

    print(f"  - Total Daily Quotes Records: {len(df_all):,}")
    print(f"  - Unique Tickers Downloaded: {df_all['ticker'].nunique():,}")

    # Save to Master Parquet
    con = duckdb.connect()
    con.register("quotes_temp", df_all)
    con.execute(f"COPY quotes_temp TO '{DAILY_QUOTES_PARQUET}' (FORMAT PARQUET, COMPRESSION SNAPPY);")
    con.close()

    parquet_mb = DAILY_QUOTES_PARQUET.stat().st_size / (1024 * 1024)
    print(f"\n[Market Quotes] Exported daily quotes Parquet to '{DAILY_QUOTES_PARQUET}' ({parquet_mb:.2f} MB).")


if __name__ == "__main__":
    download_market_quotes_batch()
