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

RAW_NEWS_DIR = Path("data/raw/news_api")
CONSOLIDATED_CSV_PATH = RAW_NEWS_DIR / "all_financial_news.csv"
CONSOLIDATED_PARQUET_PATH = RAW_NEWS_DIR / "all_financial_news.parquet"

ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "805POS4LAV5IQECM")
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}


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


def fetch_yfinance_news(ticker: str) -> List[Dict]:
    """Fetches financial news via yfinance API."""
    articles = []
    try:
        import yfinance as yf
        t_obj = yf.Ticker(ticker)
        raw_news = t_obj.news
        if raw_news:
            for item in raw_news:
                content = item.get("content", item)
                title = content.get("title", item.get("title", ""))
                url = content.get("canonicalUrl", {}).get("url", item.get("link", ""))
                pub_date = content.get("pubDate", item.get("publisher", ""))
                summary = content.get("summary", "")
                if title:
                    articles.append({
                        "ticker": ticker,
                        "source_api": "yfinance",
                        "title": title,
                        "url": url,
                        "time_published": pub_date,
                        "authors": "yfinance",
                        "summary": summary
                    })
    except Exception:
        pass
    return articles


def fetch_google_news_rss(ticker: str, company_name: str = "") -> List[Dict]:
    """Fetches historical financial news via Google News RSS Feed (fast, free, unlimited)."""
    articles = []
    try:
        query = f"{ticker} {company_name} stock OR bankruptcy OR earnings"
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
            source = item.find("source").text if item.find("source") is not None else "GoogleNews"
            
            if title:
                articles.append({
                    "ticker": ticker,
                    "source_api": "GoogleNewsRSS",
                    "title": title,
                    "url": link,
                    "time_published": pub_date,
                    "authors": source,
                    "summary": title
                })
    except Exception:
        pass
    return articles


def fetch_alpha_vantage_news(ticker: str, api_key: str) -> List[Dict]:
    """Fetches news from Alpha Vantage if valid non-demo API Key provided."""
    if not api_key or api_key.upper() == "DEMO":
        return []
        
    url = "https://www.alphavantage.co/query"
    params = {
        "function": "NEWS_SENTIMENT",
        "tickers": ticker,
        "limit": 1000,
        "apikey": api_key
    }
    articles = []
    try:
        resp = requests.get(url, params=params, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            feed = data.get("feed", [])
            for item in feed:
                articles.append({
                    "ticker": ticker,
                    "source_api": "AlphaVantage",
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "time_published": item.get("time_published", ""),
                    "authors": ", ".join(item.get("authors", [])),
                    "summary": item.get("summary", "")
                })
    except Exception:
        pass
    return articles


def fetch_news_multi_source(ticker: str, company_name: str = "") -> List[Dict]:
    """Combines YFinance, Google News RSS, and API sources with deduplication."""
    articles = []
    
    # 1. YFinance
    articles.extend(fetch_yfinance_news(ticker))
    
    # 2. Google News RSS
    articles.extend(fetch_google_news_rss(ticker, company_name=company_name))
    
    # 3. Alpha Vantage if key provided
    if ALPHA_VANTAGE_API_KEY and ALPHA_VANTAGE_API_KEY.upper() != "DEMO":
        articles.extend(fetch_alpha_vantage_news(ticker, ALPHA_VANTAGE_API_KEY))

    # Deduplicate by title
    seen = set()
    unique_articles = []
    for art in articles:
        clean_t = art["title"].strip().lower()
        if clean_t and clean_t not in seen:
            seen.add(clean_t)
            unique_articles.append(art)

    return unique_articles


def download_news_dataset(max_tickers_to_process: Optional[int] = None):
    """Fast, multi-source news downloader over all tickers in dataset."""
    RAW_NEWS_DIR.mkdir(parents=True, exist_ok=True)

    tickers = load_dataset_tickers(max_tickers=max_tickers_to_process)
    print(f"\n[News Downloader] Starting fast multi-source news collection for {len(tickers)} tickers...")
    print(f"  - Output Directory: '{RAW_NEWS_DIR}'\n")

    total_articles = 0
    already_done = 0

    pbar = tqdm(tickers, desc="Processing Tickers", unit="ticker")
    for i, (ticker, comp_name, cik) in enumerate(pbar, 1):
        ticker_csv = RAW_NEWS_DIR / f"news_{ticker}.csv"
        
        # Resume capability
        if ticker_csv.exists() and ticker_csv.stat().st_size > 50:
            already_done += 1
            continue

        arts = fetch_news_multi_source(ticker, company_name=comp_name)
        if arts:
            df_t = pd.DataFrame(arts)
            df_t["cik"] = cik
            df_t["company_name"] = comp_name
            df_t.to_csv(ticker_csv, index=False, encoding="utf-8")
            total_articles += len(arts)
            pbar.set_postfix_str(f"Last: {ticker} ({len(arts)} news)")
        else:
            pd.DataFrame(columns=["ticker", "company_name", "title", "url", "time_published", "summary"]).to_csv(ticker_csv, index=False)
            pbar.set_postfix_str(f"Last: {ticker} (0 news)")

        # Fast 0.2s pause between Google News RSS calls
        time.sleep(0.2)

    # Master Consolidation
    print("\n[News Downloader] Consolidating all ticker news files into master dataset...")
    all_csvs = list(RAW_NEWS_DIR.glob("news_*.csv"))
    dfs = []
    for f in all_csvs:
        try:
            d = pd.read_csv(f, dtype=str)
            if not d.empty and len(d.columns) > 2:
                dfs.append(d)
        except Exception:
            pass

    if dfs:
        master_df = pd.concat(dfs, ignore_index=True)
        master_df.to_csv(CONSOLIDATED_CSV_PATH, index=False, encoding="utf-8")
        try:
            master_df.to_parquet(CONSOLIDATED_PARQUET_PATH, compression="snappy", index=False)
            mb = CONSOLIDATED_PARQUET_PATH.stat().st_size / (1024 * 1024)
            print(f"  - Master Parquet exported: {CONSOLIDATED_PARQUET_PATH} ({mb:.2f} MB)")
        except Exception as e:
            print(f"  - [Warning] Parquet export failed: {e}")

        print("\n" + "=" * 60)
        print("NEWS INGESTION PIPELINE COMPLETE:")
        print("=" * 60)
        print(f"  - Total Tickers Processed: {len(tickers)}")
        print(f"  - Total Master Articles Collected: {len(master_df):,}")
        print(f"  - Master CSV Path: {CONSOLIDATED_CSV_PATH}")
        print("=" * 60)
    else:
        print("[News Downloader] No articles downloaded.")


if __name__ == "__main__":
    download_news_dataset()
