import os
import sys
import duckdb
import requests
import pandas as pd
import numpy as np
import yfinance as yf
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "805POS4LAV5IQECM")

OUTPUT_DIR = Path("data/raw/market_quotes")
DAILY_QUOTES_PARQUET = OUTPUT_DIR / "stock_prices_2009_2026.parquet"
RESCUED_QUOTES_PARQUET = OUTPUT_DIR / "stock_prices_rescued_2009_2026.parquet"

SEC_PIVOTED_PARQUET = Path("data/processed/sec_dataset/v2_sec_financials_pivoted.parquet")
CIK_TICKER_MAP_PARQUET = Path("data/raw/sec_metadata/cik_ticker_map.parquet")
SUBMISSION_TSV = Path("scratch/sample_insider_extracted/SUBMISSION.tsv")


def get_failed_tickers():
    """Identifies all tickers that failed in the first yfinance pass."""
    con = duckdb.connect()
    
    # Existing downloaded tickers
    existing_tickers = set()
    if DAILY_QUOTES_PARQUET.exists():
        df_exist = con.execute(f"SELECT DISTINCT ticker FROM '{DAILY_QUOTES_PARQUET}'").df()
        existing_tickers = set(df_exist['ticker'].tolist())

    # All dataset tickers
    all_tickers = set()
    if CIK_TICKER_MAP_PARQUET.exists():
        df_sub = con.execute(f"SELECT DISTINCT ticker FROM '{CIK_TICKER_MAP_PARQUET}' WHERE ticker IS NOT NULL AND TRIM(ticker) != '';").df()
        all_tickers.update(df_sub['ticker'].tolist())
    elif SUBMISSION_TSV.exists():
        df_sub = con.execute("""
            SELECT DISTINCT UPPER(TRIM(ISSUERTRADINGSYMBOL)) as ticker
            FROM read_csv_auto($1, ignore_errors=True, header=True, all_varchar=True)
            WHERE ISSUERTRADINGSYMBOL IS NOT NULL AND TRIM(ISSUERTRADINGSYMBOL) != '';
        """, [str(SUBMISSION_TSV)]).df()
        all_tickers.update(df_sub['ticker'].tolist())

    news_parquet = Path("data/processed/news_dataset/consolidated_financial_news_1999_2026.parquet")
    if news_parquet.exists():
        df_news = con.execute("""
            SELECT DISTINCT UPPER(TRIM(ticker)) as ticker
            FROM read_parquet($1)
            WHERE ticker IS NOT NULL AND TRIM(ticker) != '';
        """, [str(news_parquet)]).df()
        all_tickers.update(df_news['ticker'].tolist())

    con.close()

    failed = sorted([t for t in all_tickers if t and len(t) <= 6 and t.isalnum() and t not in existing_tickers])
    print(f"[Rescue Engine] Identified {len(failed):,} failed/unquoted tickers out of {len(all_tickers):,} total tickers.")
    return failed


def rescue_ticker_yfinance(orig_ticker):
    """
    Attempts ticker transformation variants:
    1. Replacing '.' with '-' (e.g. BRK.A -> BRK-A)
    2. Appending 'Q' for bankrupt Chapter 11 firms (e.g. AVYA -> AVYAQ)
    3. Appending '.PK' for Pink Sheets / OTC delisted firms (e.g. ENE -> ENE.PK)
    """
    variants = [
        orig_ticker.replace('.', '-'),
        f"{orig_ticker}Q",
        f"{orig_ticker}.PK",
        f"{orig_ticker}.OTC"
    ]
    
    for var in variants:
        try:
            df = yf.download(var, start="2008-01-01", end="2026-12-31", progress=False, auto_adjust=True)
            if not df.empty:
                df = df.reset_index()
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]

                df['ticker'] = orig_ticker  # Save under original ticker
                cols_map = {'Date': 'trading_date', 'Close': 'close_price', 'Volume': 'volume'}
                df = df.rename(columns=cols_map)
                
                needed_cols = ['trading_date', 'ticker', 'close_price', 'volume']
                avail_cols = [c for c in needed_cols if c in df.columns]
                
                res = df[avail_cols].copy()
                res['trading_date'] = pd.to_datetime(res['trading_date']).dt.strftime('%Y-%m-%d')
                res['close_price'] = pd.to_numeric(res['close_price'], errors='coerce')
                res['volume'] = pd.to_numeric(res['volume'], errors='coerce')
                
                clean_res = res.dropna(subset=['close_price'])
                if not clean_res.empty:
                    return clean_res
        except Exception:
            continue
            
    return None


def rescue_ticker_alpha_vantage(orig_ticker):
    """Fallback query to Alpha Vantage API."""
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={orig_ticker}&apikey={ALPHA_VANTAGE_API_KEY}"
    try:
        res = requests.get(url, timeout=8)
        if res.status_code == 200:
            data = res.json()
            if 'Information' in data: # Quota reached
                return 'QUOTA_EXCEEDED'
            
            series = data.get('Time Series (Daily)')
            if series:
                rows = []
                for d, vals in series.items():
                    close_p = vals.get('4. close') or vals.get('5. adjusted close')
                    vol = vals.get('6. volume') or 0
                    if close_p:
                        rows.append({
                            'trading_date': d,
                            'ticker': orig_ticker,
                            'close_price': float(close_p),
                            'volume': float(vol)
                        })
                if rows:
                    return pd.DataFrame(rows)
    except Exception:
        pass
    return None


def run_rescue_pipeline():
    print("=" * 65)
    print("RESCUE PIPELINE: BANKRUPT SUFFIXES (Q, PK), ALPHA VANTAGE & SEC FALLBACK")
    print("=" * 65)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    failed_tickers = get_failed_tickers()

    if not failed_tickers:
        print("[Rescue Engine] No failed tickers to rescue!")
        return

    rescued_dfs = []
    alpha_vantage_active = True

    print(f"\n[Rescue Engine] Step 1: Rescuing {len(failed_tickers):,} tickers via Bankruptcy (Q, PK) variants & Alpha Vantage...")
    
    # Process batch in parallel
    with ThreadPoolExecutor(max_workers=12) as executor:
        futures = {executor.submit(rescue_ticker_yfinance, t): t for t in failed_tickers}
        for future in tqdm(as_completed(futures), total=len(failed_tickers), desc="Rescuing Bankruptcy Tickers", unit="ticker"):
            orig_t = futures[future]
            res = future.result()
            if res is not None and not res.empty:
                rescued_dfs.append(res)
            elif alpha_vantage_active:
                # Try Alpha Vantage Fallback
                av_res = rescue_ticker_alpha_vantage(orig_t)
                if av_res == 'QUOTA_EXCEEDED':
                    alpha_vantage_active = False
                elif av_res is not None and isinstance(av_res, pd.DataFrame) and not av_res.empty:
                    rescued_dfs.append(av_res)

    if rescued_dfs:
        print("\n[Rescue Engine] Consolidating newly rescued tickers...")
        df_rescued = pd.concat(rescued_dfs, ignore_index=True)
        print(f"  - Rescued Tickers Count: {df_rescued['ticker'].nunique():,}")
        print(f"  - Rescued Daily Quotes Rows: {len(df_rescued):,}")

        con = duckdb.connect()
        con.register("rescued_temp", df_rescued)
        
        # Combine with existing stock_prices Parquet
        if DAILY_QUOTES_PARQUET.exists():
            con.execute(f"""
            CREATE OR REPLACE TABLE master_quotes AS
            SELECT * FROM '{DAILY_QUOTES_PARQUET}'
            UNION ALL
            SELECT * FROM rescued_temp;
            """)
        else:
            con.execute("CREATE OR REPLACE TABLE master_quotes AS SELECT * FROM rescued_temp;")

        # Save Master Rescued Parquet
        con.execute(f"COPY master_quotes TO '{DAILY_QUOTES_PARQUET}' (FORMAT PARQUET, COMPRESSION SNAPPY);")
        con.close()

        parquet_mb = DAILY_QUOTES_PARQUET.stat().st_size / (1024 * 1024)
        print(f"  - Updated Master Stock Quotes Parquet: '{DAILY_QUOTES_PARQUET}' ({parquet_mb:.2f} MB).")
    else:
        print("[Rescue Engine] No additional quotes rescued via Yahoo/Alpha Vantage variants.")

    print("\n[Rescue Engine] Step 2: Proceeding to SEC Public Float Fallback & Feature Processing...")


if __name__ == "__main__":
    run_rescue_pipeline()
