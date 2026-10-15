import os
import sys
import duckdb
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Inputs and Outputs
CONSOLIDATED_NEWS_PARQUET = Path("data/processed/news_dataset/consolidated_financial_news_1999_2026.parquet")
HISTORICAL_FNSPID_SENTIMENT_CSV = Path("data/processed/sp500_real_news_sentiment_dataset.csv")

OUTPUT_DIR = Path("data/processed/news_dataset")
RECENT_SENTIMENT_PARQUET = OUTPUT_DIR / "news_sentiment_2021_2026.parquet"
MASTER_SENTIMENT_PARQUET = OUTPUT_DIR / "consolidated_news_sentiment_1999_2026.parquet"
MASTER_SENTIMENT_CSV = OUTPUT_DIR / "consolidated_news_sentiment_1999_2026.csv"

# Batching & GPU Setup
BATCH_SIZE = 512
MAX_LENGTH = 128
MODEL_NAME = "ProsusAI/finbert"


def format_news_text(row):
    """Concatenates title and summary cleanly."""
    title = str(row.get('title', '') or '').strip()
    summary = str(row.get('summary', '') or '').strip()
    
    if not summary or summary.lower() == 'nan' or summary.lower() == title.lower():
        return title
    
    return f"{title}. {summary}"


def run_finbert_gpu_inference():
    """
    Runs FinBERT sentiment analysis on NVIDIA GeForce RTX 5070 for 2021-2026 news,
    concatenating title and summary.
    """
    print("=" * 65)
    print("FINBERT SENTIMENT ANALYSIS ON NVIDIA GEFORCE RTX 5070 (2021-2026)")
    print("=" * 65)

    if not torch.cuda.is_available():
        print("[Error] CUDA is not available! GPU required.")
        return

    device = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    print(f"[GPU Setup] Device: {gpu_name} ({vram_gb:.2f} GB VRAM)")

    # 1. Load 2021-2026 news articles
    print(f"\n[Data Loader] Reading 2021-2026 news articles from '{CONSOLIDATED_NEWS_PARQUET}'...")
    con = duckdb.connect()
    df_recent = con.execute("""
        SELECT 
            ticker as stock_symbol,
            company_name,
            title as article_title,
            url,
            time_published as date,
            pub_year as year,
            authors as publisher,
            summary
        FROM read_parquet($1)
        WHERE pub_year >= 2021 OR pub_year IS NULL;
    """, [str(CONSOLIDATED_NEWS_PARQUET)]).df()

    con.close()
    print(f"  - Loaded {len(df_recent):,} news articles for 2021-2026 sentiment processing.")

    if df_recent.empty:
        print("[Warning] No 2021-2026 news records found to process.")
        return

    # 2. Concatenate Title + Summary
    print("[Preprocessing] Concatenating title and summary for each article...")
    texts = [format_news_text(row) for row in df_recent.to_dict('records')]
    print(f"  - Sample input text: '{texts[0][:120]}...'")

    # 3. Load FinBERT Model & Tokenizer
    print(f"\n[FinBERT Engine] Loading model '{MODEL_NAME}' onto RTX 5070 VRAM...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    # 4. GPU Batch Inference with PyTorch AMP FP16
    print(f"\n[FinBERT Engine] Starting CUDA inference (batch_size={BATCH_SIZE}, FP16 AMP)...")
    pos_probs = []
    neg_probs = []
    neu_probs = []

    num_batches = int(np.ceil(len(texts) / BATCH_SIZE))
    
    with torch.no_grad():
        for b_idx in tqdm(range(num_batches), desc="RTX 5070 FinBERT Inference", unit="batch"):
            batch_texts = texts[b_idx * BATCH_SIZE : (b_idx + 1) * BATCH_SIZE]
            
            inputs = tokenizer(
                batch_texts, 
                padding=True, 
                truncation=True, 
                max_length=MAX_LENGTH, 
                return_tensors="pt"
            ).to(device)

            with torch.cuda.amp.autocast():
                outputs = model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

            # ProsusAI/finbert order: [positive, negative, neutral]
            probs_cpu = probs.cpu().numpy()
            pos_probs.extend(probs_cpu[:, 0])
            neg_probs.extend(probs_cpu[:, 1])
            neu_probs.extend(probs_cpu[:, 2])

    df_recent['finbert_pos'] = pos_probs
    df_recent['finbert_neg'] = neg_probs
    df_recent['finbert_neu'] = neu_probs
    df_recent['headline_sentiment_index'] = df_recent['finbert_pos'] - df_recent['finbert_neg']

    # Clean intermediate summary column before final output
    df_recent_output = df_recent.drop(columns=['summary', 'company_name', 'url'], errors='ignore')

    # 5. Export 2021-2026 Sentiment Dataset
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df_recent_output.to_parquet(RECENT_SENTIMENT_PARQUET, index=False, compression="snappy")
    print(f"\n[FinBERT Engine] Saved 2021-2026 sentiment predictions to '{RECENT_SENTIMENT_PARQUET}' ({RECENT_SENTIMENT_PARQUET.stat().st_size / (1024*1024):.2f} MB).")

    # 6. Merge with Historical FNSPID Sentiment Dataset
    merge_with_historical_fnspid(df_recent_output)


def merge_with_historical_fnspid(df_recent_sentiment):
    """
    Merges historical FNSPID sentiment dataset (1999-2020) from sp500_real_news_sentiment_dataset.csv
    with recent 2021-2026 sentiment dataset into master unified Parquet & CSV.
    """
    print("\n" + "=" * 65)
    print("MERGING HISTORICAL FNSPID SENTIMENT (1999-2020) & RECENT SENTIMENT (2021-2026)")
    print("=" * 65)

    if not HISTORICAL_FNSPID_SENTIMENT_CSV.exists():
        print(f"[Warning] Historical FNSPID sentiment CSV not found at '{HISTORICAL_FNSPID_SENTIMENT_CSV}'. Exporting recent sentiment only.")
        df_recent_sentiment.to_parquet(MASTER_SENTIMENT_PARQUET, index=False, compression="snappy")
        return

    print(f"[Merge Engine] Reading historical FNSPID sentiment dataset from '{HISTORICAL_FNSPID_SENTIMENT_CSV}'...")
    con = duckdb.connect()
    
    con.execute("""
    CREATE OR REPLACE TABLE fnspid_sentiment AS
    SELECT 
        date,
        article_title,
        stock_symbol,
        publisher,
        TRY_CAST(year AS INTEGER) as year,
        TRY_CAST(finbert_pos AS DOUBLE) as finbert_pos,
        TRY_CAST(finbert_neg AS DOUBLE) as finbert_neg,
        TRY_CAST(finbert_neu AS DOUBLE) as finbert_neu,
        TRY_CAST(headline_sentiment_index AS DOUBLE) as headline_sentiment_index
    FROM read_csv_auto($1, ignore_errors=True, header=True)
    WHERE stock_symbol IS NOT NULL AND article_title IS NOT NULL;
    """, [str(HISTORICAL_FNSPID_SENTIMENT_CSV)])

    fnspid_rows = con.execute("SELECT COUNT(*) FROM fnspid_sentiment").fetchone()[0]
    print(f"  - Historical FNSPID sentiment records (1999-2020): {fnspid_rows:,}")

    con.register("recent_sentiment_temp", df_recent_sentiment)

    print("[Merge Engine] Unifying historical FNSPID and 2021-2026 sentiment predictions...")
    con.execute("""
    CREATE OR REPLACE TABLE master_sentiment AS
    SELECT 
        date,
        article_title,
        stock_symbol,
        publisher,
        year,
        finbert_pos,
        finbert_neg,
        finbert_neu,
        headline_sentiment_index
    FROM (
        SELECT *, ROW_NUMBER() OVER(PARTITION BY stock_symbol, LOWER(TRIM(article_title)), year ORDER BY date DESC) as rn
        FROM (
            SELECT * FROM fnspid_sentiment WHERE year <= 2020 OR year IS NULL
            UNION ALL
            SELECT date, article_title, stock_symbol, publisher, year, finbert_pos, finbert_neg, finbert_neu, headline_sentiment_index FROM recent_sentiment_temp
        )
    )
    WHERE rn = 1;
    """)

    total_master = con.execute("SELECT COUNT(*) FROM master_sentiment").fetchone()[0]
    unique_symbols = con.execute("SELECT COUNT(DISTINCT stock_symbol) FROM master_sentiment").fetchone()[0]

    # Export Master Parquet & CSV
    print(f"\n[Merge Engine] Exporting Master Sentiment Parquet: '{MASTER_SENTIMENT_PARQUET}'...")
    con.execute(f"COPY master_sentiment TO '{MASTER_SENTIMENT_PARQUET}' (FORMAT PARQUET, COMPRESSION SNAPPY);")
    
    parquet_mb = MASTER_SENTIMENT_PARQUET.stat().st_size / (1024 * 1024)

    # Yearly breakdown
    yearly_df = con.execute("""
        SELECT 
            year,
            COUNT(*) as total_news_sentiment,
            COUNT(DISTINCT stock_symbol) as unique_tickers,
            ROUND(AVG(headline_sentiment_index), 4) as avg_sentiment_index
        FROM master_sentiment
        WHERE year IS NOT NULL
        GROUP BY year
        ORDER BY year ASC
    """).df()

    print("\n" + "=" * 65)
    print("MASTER SENTIMENT DATASET SUMMARY (1999-2026):")
    print("=" * 65)
    print(yearly_df.to_string(index=False))
    print("=" * 65)
    print(f"  - Total Master Sentiment Records: {total_master:,}")
    print(f"  - Total Unique Tickers Covered: {unique_symbols:,}")
    print(f"  - Master Parquet Path: {MASTER_SENTIMENT_PARQUET} ({parquet_mb:.2f} MB)")
    print("=" * 65)

    con.close()
    print("[Merge Engine] FinBERT GPU sentiment pipeline finished successfully!")


if __name__ == "__main__":
    run_finbert_gpu_inference()
