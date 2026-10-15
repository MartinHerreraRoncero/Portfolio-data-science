import os
import sys
import time
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm
import requests

HISTORICAL_NEWS_DIR = Path("data/raw/news_historical_2021_2024")
CONSOLIDATED_PARQUET_PATH = Path("data/processed/news_dataset/consolidated_financial_news_1999_2026.parquet")

ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "805POS4LAV5IQECM")
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

TARGET_YEARS = [2021, 2022, 2023, 2024]


def load_dataset_tickers(max_tickers: Optional[int] = None) -> List[Tuple[str, str, int]]:
    """Loads tickers, company names, and CIKs from SEC and Bankruptcy dataset."""
    tickers_list = []
    
    # 1. SEC Insider CIK-Ticker map (persistent parquet or scratch fallback)
    cik_map_pq = Path("data/raw/sec_metadata/cik_ticker_map.parquet")
    sample_sub = Path("scratch/sample_insider_extracted/SUBMISSION.tsv")
    if cik_map_pq.exists():
        try:
            df_map = pd.read_parquet(cik_map_pq)
            for _, r in df_map.iterrows():
                ticker = str(r['ticker']).strip().upper()
                cik = int(r['cik'])
                if len(ticker) <= 6 and ticker.isalnum():
                    tickers_list.append((ticker, ticker, cik))
        except Exception:
            pass
    elif sample_sub.exists():
        try:
            df_sub = pd.read_csv(sample_sub, sep='\t', low_memory=False)
            df_clean = df_sub[['ISSUERTRADINGSYMBOL', 'ISSUERNAME', 'ISSUERCIK']].dropna().drop_duplicates(subset=['ISSUERTRADINGSYMBOL'])
            for _, r in df_clean.iterrows():
                ticker = str(r['ISSUERTRADINGSYMBOL']).strip().upper()
                name = str(r['ISSUERNAME']).strip()
                cik = int(r['ISSUERCIK'])
                if len(ticker) <= 6 and ticker.isalnum():
                    tickers_list.append((ticker, name, cik))
        except Exception:
            pass

    # 2. Bankrupt companies dataset
    bankrupt_csv = Path("data/raw/american_bankruptcy.csv")
    if bankrupt_csv.exists():
        try:
            df_b = pd.read_csv(bankrupt_csv)
            if 'company_name' in df_b.columns:
                bankrupt_names = df_b[df_b['status_label'] == 1]['company_name'].unique()
                for name in bankrupt_names:
                    clean_t = "".join([c for c in name.split()[0] if c.isalpha()]).upper()
                    if len(clean_t) >= 2:
                        tickers_list.append((clean_t, name, 0))
        except Exception:
            pass

    seen = set()
    unique_tickers = []
    for t, name, cik in tickers_list:
        if t not in seen:
            seen.add(t)
            unique_tickers.append((t, name, cik))

    if max_tickers:
        unique_tickers = unique_tickers[:max_tickers]
    return unique_tickers


def fetch_google_news_rss_year(ticker: str, company_name: str, year: int) -> List[Dict]:
    """Fetches historical financial news for a specific target year via Google News RSS operators."""
    articles = []
    try:
        query = f"{ticker} {company_name} stock OR earnings after:{year}-01-01 before:{year+1}-01-01"
        encoded_query = urllib.parse.quote(query)
        rss_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"
        
        req = urllib.request.Request(rss_url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=10) as response:
            xml_content = response.read()
            
        root = ET.fromstring(xml_content)
        for item in root.findall(".//item"):
            title = item.find("title").text if item.find("title") is not None else ""
            link = item.find("link").text if item.find("link") is not None else ""
            pub_date = item.find("pubDate").text if item.find("pubDate") is not None else ""
            source = item.find("source").text if item.find("source") is not None else "GoogleNewsRSS"
            
            if title:
                articles.append({
                    "ticker": ticker,
                    "source_api": "GoogleNewsRSS_Yearly",
                    "title": title,
                    "url": link,
                    "time_published": pub_date,
                    "authors": source,
                    "summary": title,
                    "target_year": year
                })
    except Exception:
        pass
    return articles


def fetch_finnhub_news_year(ticker: str, year: int, api_key: str = FINNHUB_API_KEY) -> List[Dict]:
    """Fetches historical company news for a specific year from Finnhub /company-news."""
    if not api_key:
        return []
    url = "https://finnhub.io/api/v1/company-news"
    params = {
        "symbol": ticker,
        "from": f"{year}-01-01",
        "to": f"{year}-12-31",
        "token": api_key
    }
    articles = []
    try:
        resp = requests.get(url, params=params, timeout=15)
        if resp.status_code == 200:
            items = resp.json()
            if isinstance(items, list):
                for item in items:
                    pub_time = time.strftime('%Y-%m-%dT%H:%M:%S', time.gmtime(item.get('datetime', 0)))
                    articles.append({
                        "ticker": ticker,
                        "source_api": "Finnhub_Yearly",
                        "title": item.get("headline", ""),
                        "url": item.get("url", ""),
                        "time_published": pub_time,
                        "authors": item.get("source", ""),
                        "summary": item.get("summary", ""),
                        "target_year": year
                    })
    except Exception:
        pass
    return articles


def download_historical_news_2021_2024(max_tickers_to_process: Optional[int] = None):
    """
    Downloads historical news for target years 2021, 2022, 2023, and 2024
    across all tickers using year-bounded RSS and API queries.
    """
    HISTORICAL_NEWS_DIR.mkdir(parents=True, exist_ok=True)
    tickers = load_dataset_tickers(max_tickers=max_tickers_to_process)

    print(f"\n[Historical News 2021-2024] Starting targeted historical download for {len(tickers)} tickers across years {TARGET_YEARS}...")
    print(f"  - Output Directory: '{HISTORICAL_NEWS_DIR}'\n")

    total_articles = 0
    pbar = tqdm(tickers, desc="Processing Tickers 2021-2024", unit="ticker")

    for i, (ticker, comp_name, cik) in enumerate(pbar, 1):
        ticker_articles = []
        
        for yr in TARGET_YEARS:
            csv_yr_path = HISTORICAL_NEWS_DIR / f"news_{ticker}_{yr}.csv"
            
            # Resume check
            if csv_yr_path.exists() and csv_yr_path.stat().st_size > 50:
                continue

            # Fetch for specific year
            yr_arts = fetch_google_news_rss_year(ticker, comp_name, yr)
            if FINNHUB_API_KEY:
                yr_arts.extend(fetch_finnhub_news_year(ticker, yr))

            if yr_arts:
                df_yr = pd.DataFrame(yr_arts)
                df_yr["cik"] = cik
                df_yr["company_name"] = comp_name
                df_yr.to_csv(csv_yr_path, index=False, encoding="utf-8")
                total_articles += len(yr_arts)
                ticker_articles.extend(yr_arts)
            else:
                pd.DataFrame(columns=["ticker", "company_name", "title", "url", "time_published", "summary"]).to_csv(csv_yr_path, index=False)

            # 0.15s rate limit pause
            time.sleep(0.15)

        pbar.set_postfix_str(f"Last: {ticker} (+{len(ticker_articles)} arts)")

    print("\n[Historical News 2021-2024] Download pipeline completed successfully!")


if __name__ == "__main__":
    download_historical_news_2021_2024()
