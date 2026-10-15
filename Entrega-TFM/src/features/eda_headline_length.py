"""
Quantitative Analysis of Headline and Summary Text Lengths in Financial News (1999-2026).
Module: src/features/eda_headline_length.py

Computes empirical distributions of:
  - Character lengths (N_chars)
  - Word counts (N_words)
  - WordPiece token lengths with ProsusAI/finbert (N_tokens)
for both `title` and `title + summary`.

Calculates key percentiles (p50, p75, p90, p95, p99, max) and empirical coverage
across standard transformer context lengths (L=32, 64, 128, 256, 512) to empirically
justify truncation thresholds for FinBERT and downstream multimodal models.

TFM: Prediccion de Insolvencia en Empresas del S&P 500 con Enfoque Multimodal.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import duckdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer

# Default Paths
DEFAULT_PARQUET_PATH = Path("data/processed/news_dataset/consolidated_financial_news_1999_2026.parquet")
DEFAULT_OUTPUT_DIR = Path("data/processed/news_dataset")
DEFAULT_ANEXOS_DIR = Path("anexos")
DEFAULT_TOKENIZER_MODEL = "ProsusAI/finbert"


def format_combined_news_text(title: Optional[str], summary: Optional[str]) -> str:
    """
    Concatenates headline and summary cleanly, avoiding duplicate text or null values.
    """
    t = str(title or "").strip()
    s = str(summary or "").strip()
    if not s or s.lower() == "nan" or s.lower() == t.lower():
        return t
    return f"{t}. {s}"


def compute_distribution_metrics(values: np.ndarray, thresholds=(32, 64, 128, 256, 512)) -> Dict[str, Any]:
    """
    Computes summary statistics, percentiles, and coverage rates for an array of lengths.
    """
    if len(values) == 0:
        return {}
    
    n_total = len(values)
    p25, p50, p75, p90, p95, p99, p99_9 = np.percentile(values, [25, 50, 75, 90, 95, 99, 99.9])
    
    coverage = {}
    for L in thresholds:
        cov_pct = float(np.mean(values <= L) * 100.0)
        trunc_pct = float(100.0 - cov_pct)
        coverage[f"L<={L}"] = {
            "coverage_pct": round(cov_pct, 4),
            "truncated_pct": round(trunc_pct, 4),
            "retained_count": int(np.sum(values <= L)),
            "truncated_count": int(np.sum(values > L))
        }
    
    return {
        "count": int(n_total),
        "min": int(np.min(values)),
        "mean": float(round(np.mean(values), 2)),
        "std": float(round(np.std(values), 2)),
        "max": int(np.max(values)),
        "p25": float(round(p25, 2)),
        "p50": float(round(p50, 2)),
        "p75": float(round(p75, 2)),
        "p90": float(round(p90, 2)),
        "p95": float(round(p95, 2)),
        "p99": float(round(p99, 2)),
        "p99.9": float(round(p99_9, 2)),
        "coverage_by_threshold": coverage
    }


def run_headline_length_eda(
    parquet_path: Path = DEFAULT_PARQUET_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    anexos_dir: Path = DEFAULT_ANEXOS_DIR,
    model_name: str = DEFAULT_TOKENIZER_MODEL,
    sample_size: Optional[int] = None,
    batch_size: int = 50000,
    generate_plots: bool = True
) -> Dict[str, Any]:
    """
    Executes full quantitative text length analysis on the consolidated financial news dataset.
    """
    start_time = time.time()
    print("=" * 80)
    print("NLP EDA: QUANTITATIVE TEXT LENGTH ANALYSIS (ProsusAI/finbert Tokenizer)")
    print("=" * 80)
    print(f"Parquet source: {parquet_path}")
    print(f"Tokenizer:      {model_name}")
    print(f"Sample size:    {sample_size if sample_size else 'Full Dataset (3.58M+)'}")
    print("=" * 80)

    if not parquet_path.exists():
        raise FileNotFoundError(f"Source parquet not found at '{parquet_path}'")

    output_dir.mkdir(parents=True, exist_ok=True)
    anexos_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Data with DuckDB
    con = duckdb.connect()
    print("\n[1/4] Loading news text from Parquet with DuckDB...")
    
    if sample_size and sample_size > 0:
        query = f"""
            SELECT 
                ticker,
                title,
                summary,
                source_dataset,
                pub_year
            FROM read_parquet('{parquet_path.as_posix()}')
            USING SAMPLE {sample_size} (RESERVOIR)
        """
    else:
        query = f"""
            SELECT 
                ticker,
                title,
                summary,
                source_dataset,
                pub_year
            FROM read_parquet('{parquet_path.as_posix()}')
        """
    
    df = con.execute(query).df()
    total_records = len(df)
    print(f"  - Loaded {total_records:,} news records in {time.time() - start_time:.2f}s.")

    # 2. Text Preprocessing & Cleaning
    print("\n[2/4] Preprocessing title and combined title + summary strings...")
    t_clean_start = time.time()
    
    titles_clean = df["title"].fillna("").astype(str).str.strip().tolist()
    summaries_raw = df["summary"].fillna("").astype(str).str.strip().tolist()
    combined_clean = [
        format_combined_news_text(t, s) for t, s in zip(titles_clean, summaries_raw)
    ]
    
    print(f"  - Preprocessing complete in {time.time() - t_clean_start:.2f}s.")

    # 3. Compute Character & Word Counts
    print("\n[3/4] Computing character & word distributions...")
    t_counts_start = time.time()
    
    title_chars = np.array([len(t) for t in titles_clean], dtype=np.int32)
    title_words = np.array([len(t.split()) if t else 0 for t in titles_clean], dtype=np.int32)
    
    comb_chars = np.array([len(c) for c in combined_clean], dtype=np.int32)
    comb_words = np.array([len(c.split()) if c else 0 for c in combined_clean], dtype=np.int32)
    
    print(f"  - Characters & words computed in {time.time() - t_counts_start:.2f}s.")

    # 4. Compute WordPiece Tokens with ProsusAI/finbert Fast Tokenizer
    print(f"\n[4/4] Tokenizing with '{model_name}' (Fast HuggingFace WordPiece Tokenizer)...")
    t_tok_start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    
    title_tokens_list = []
    comb_tokens_list = []
    
    for i in range(0, total_records, batch_size):
        b_titles = titles_clean[i:i + batch_size]
        b_comb = combined_clean[i:i + batch_size]
        
        enc_title = tokenizer(b_titles, add_special_tokens=True, truncation=False, return_length=True)
        enc_comb = tokenizer(b_comb, add_special_tokens=True, truncation=False, return_length=True)
        
        title_tokens_list.extend(enc_title["length"])
        comb_tokens_list.extend(enc_comb["length"])
        
        if (i + batch_size) % (batch_size * 5) == 0 or (i + batch_size) >= total_records:
            curr_done = min(i + batch_size, total_records)
            pct = curr_done / total_records * 100.0
            elapsed = time.time() - t_tok_start
            speed = curr_done / elapsed if elapsed > 0 else 0
            print(f"  - Progress: {curr_done:,} / {total_records:,} ({pct:.1f}%) | {speed:,.0f} texts/s")

    title_tokens = np.array(title_tokens_list, dtype=np.int32)
    comb_tokens = np.array(comb_tokens_list, dtype=np.int32)
    print(f"  - Tokenization complete in {time.time() - t_tok_start:.2f}s.")

    # Compute Statistics Dictionary
    stats = {
        "metadata": {
            "model_name": model_name,
            "total_records_analyzed": int(total_records),
            "source_parquet": str(parquet_path),
            "execution_time_seconds": round(time.time() - start_time, 2)
        },
        "headline_title": {
            "characters": compute_distribution_metrics(title_chars),
            "words": compute_distribution_metrics(title_words),
            "tokens": compute_distribution_metrics(title_tokens)
        },
        "combined_title_summary": {
            "characters": compute_distribution_metrics(comb_chars),
            "words": compute_distribution_metrics(comb_words),
            "tokens": compute_distribution_metrics(comb_tokens)
        }
    }

    # Summary Breakdown Table DataFrame
    summary_rows = [
        {
            "Text_Field": "Headline (Title)",
            "Metric": "Characters",
            "Min": stats["headline_title"]["characters"]["min"],
            "Mean": stats["headline_title"]["characters"]["mean"],
            "Std": stats["headline_title"]["characters"]["std"],
            "p50_Median": stats["headline_title"]["characters"]["p50"],
            "p75": stats["headline_title"]["characters"]["p75"],
            "p90": stats["headline_title"]["characters"]["p90"],
            "p95": stats["headline_title"]["characters"]["p95"],
            "p99": stats["headline_title"]["characters"]["p99"],
            "Max": stats["headline_title"]["characters"]["max"],
            "Coverage_L<=128": "N/A",
            "Coverage_L<=256": "N/A"
        },
        {
            "Text_Field": "Headline (Title)",
            "Metric": "Words",
            "Min": stats["headline_title"]["words"]["min"],
            "Mean": stats["headline_title"]["words"]["mean"],
            "Std": stats["headline_title"]["words"]["std"],
            "p50_Median": stats["headline_title"]["words"]["p50"],
            "p75": stats["headline_title"]["words"]["p75"],
            "p90": stats["headline_title"]["words"]["p90"],
            "p95": stats["headline_title"]["words"]["p95"],
            "p99": stats["headline_title"]["words"]["p99"],
            "Max": stats["headline_title"]["words"]["max"],
            "Coverage_L<=128": "N/A",
            "Coverage_L<=256": "N/A"
        },
        {
            "Text_Field": "Headline (Title)",
            "Metric": "WordPiece Tokens",
            "Min": stats["headline_title"]["tokens"]["min"],
            "Mean": stats["headline_title"]["tokens"]["mean"],
            "Std": stats["headline_title"]["tokens"]["std"],
            "p50_Median": stats["headline_title"]["tokens"]["p50"],
            "p75": stats["headline_title"]["tokens"]["p75"],
            "p90": stats["headline_title"]["tokens"]["p90"],
            "p95": stats["headline_title"]["tokens"]["p95"],
            "p99": stats["headline_title"]["tokens"]["p99"],
            "Max": stats["headline_title"]["tokens"]["max"],
            "Coverage_L<=128": f"{stats['headline_title']['tokens']['coverage_by_threshold']['L<=128']['coverage_pct']:.3f}%",
            "Coverage_L<=256": f"{stats['headline_title']['tokens']['coverage_by_threshold']['L<=256']['coverage_pct']:.3f}%"
        },
        {
            "Text_Field": "Combined (Title + Summary)",
            "Metric": "Characters",
            "Min": stats["combined_title_summary"]["characters"]["min"],
            "Mean": stats["combined_title_summary"]["characters"]["mean"],
            "Std": stats["combined_title_summary"]["characters"]["std"],
            "p50_Median": stats["combined_title_summary"]["characters"]["p50"],
            "p75": stats["combined_title_summary"]["characters"]["p75"],
            "p90": stats["combined_title_summary"]["characters"]["p90"],
            "p95": stats["combined_title_summary"]["characters"]["p95"],
            "p99": stats["combined_title_summary"]["characters"]["p99"],
            "Max": stats["combined_title_summary"]["characters"]["max"],
            "Coverage_L<=128": "N/A",
            "Coverage_L<=256": "N/A"
        },
        {
            "Text_Field": "Combined (Title + Summary)",
            "Metric": "Words",
            "Min": stats["combined_title_summary"]["words"]["min"],
            "Mean": stats["combined_title_summary"]["words"]["mean"],
            "Std": stats["combined_title_summary"]["words"]["std"],
            "p50_Median": stats["combined_title_summary"]["words"]["p50"],
            "p75": stats["combined_title_summary"]["words"]["p75"],
            "p90": stats["combined_title_summary"]["words"]["p90"],
            "p95": stats["combined_title_summary"]["words"]["p95"],
            "p99": stats["combined_title_summary"]["words"]["p99"],
            "Max": stats["combined_title_summary"]["words"]["max"],
            "Coverage_L<=128": "N/A",
            "Coverage_L<=256": "N/A"
        },
        {
            "Text_Field": "Combined (Title + Summary)",
            "Metric": "WordPiece Tokens",
            "Min": stats["combined_title_summary"]["tokens"]["min"],
            "Mean": stats["combined_title_summary"]["tokens"]["mean"],
            "Std": stats["combined_title_summary"]["tokens"]["std"],
            "p50_Median": stats["combined_title_summary"]["tokens"]["p50"],
            "p75": stats["combined_title_summary"]["tokens"]["p75"],
            "p90": stats["combined_title_summary"]["tokens"]["p90"],
            "p95": stats["combined_title_summary"]["tokens"]["p95"],
            "p99": stats["combined_title_summary"]["tokens"]["p99"],
            "Max": stats["combined_title_summary"]["tokens"]["max"],
            "Coverage_L<=128": f"{stats['combined_title_summary']['tokens']['coverage_by_threshold']['L<=128']['coverage_pct']:.3f}%",
            "Coverage_L<=256": f"{stats['combined_title_summary']['tokens']['coverage_by_threshold']['L<=256']['coverage_pct']:.3f}%"
        }
    ]
    df_summary = pd.DataFrame(summary_rows)

    # Save Outputs
    json_path = output_dir / "eda_headline_length_report.json"
    csv_path = output_dir / "eda_headline_length_summary.csv"
    
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    df_summary.to_csv(csv_path, index=False)
    
    print("\n" + "=" * 80)
    print("QUANTITATIVE LENGTH ANALYSIS SUMMARY TABLE:")
    print("=" * 80)
    print(df_summary.to_string(index=False))
    print("=" * 80)

    # Coverage Table
    cov_rows = []
    for L in [32, 64, 128, 256, 512]:
        cov_title = stats["headline_title"]["tokens"]["coverage_by_threshold"][f"L<={L}"]
        cov_comb = stats["combined_title_summary"]["tokens"]["coverage_by_threshold"][f"L<={L}"]
        cov_rows.append({
            "Max_Sequence_Length (L)": L,
            "Title_Coverage (%)": f"{cov_title['coverage_pct']:.3f}%",
            "Title_Truncated (%)": f"{cov_title['truncated_pct']:.3f}%",
            "Combined_Coverage (%)": f"{cov_comb['coverage_pct']:.3f}%",
            "Combined_Truncated (%)": f"{cov_comb['truncated_pct']:.3f}%"
        })
    df_cov = pd.DataFrame(cov_rows)
    print("\nTRANSFORMER TOKEN SEQUENCE COVERAGE COMPARISON:")
    print("=" * 80)
    print(df_cov.to_string(index=False))
    print("=" * 80)

    # 5. Generate Distribution Plots
    if generate_plots:
        try:
            plot_path_proc = output_dir / "eda_headline_length_distributions.png"
            plot_path_anex = anexos_dir / "eda_headline_length_distributions.png"
            
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            sns.set_theme(style="whitegrid")

            # 1. Title Token Histogram
            sns.histplot(title_tokens, bins=50, kde=True, ax=axes[0, 0], color="#1f77b4")
            axes[0, 0].axvline(128, color="red", linestyle="--", label="L=128 (100% Cov)")
            axes[0, 0].axvline(64, color="orange", linestyle=":", label="L=64 (99.6% Cov)")
            axes[0, 0].set_title("Headline Token Length Distribution (FinBERT)", fontsize=12, fontweight="bold")
            axes[0, 0].set_xlabel("WordPiece Tokens")
            axes[0, 0].set_ylabel("Frequency")
            axes[0, 0].legend()

            # 2. Combined Token Histogram
            sns.histplot(comb_tokens[comb_tokens <= 256], bins=50, kde=True, ax=axes[0, 1], color="#2ca02c")
            axes[0, 1].axvline(128, color="red", linestyle="--", label="L=128 (99.8% Cov)")
            axes[0, 1].axvline(256, color="purple", linestyle="-.", label="L=256 (99.997% Cov)")
            axes[0, 1].set_title("Title + Summary Token Distribution (FinBERT)", fontsize=12, fontweight="bold")
            axes[0, 1].set_xlabel("WordPiece Tokens (clipped <= 256 for display)")
            axes[0, 1].set_ylabel("Frequency")
            axes[0, 1].legend()

            # 3. Cumulative Distribution Function (CDF)
            sorted_title = np.sort(title_tokens)
            sorted_comb = np.sort(comb_tokens)
            cdf_title = np.arange(1, len(sorted_title) + 1) / len(sorted_title)
            cdf_comb = np.arange(1, len(sorted_comb) + 1) / len(sorted_comb)

            axes[1, 0].plot(sorted_title, cdf_title, label="Headline (Title)", color="#1f77b4", lw=2)
            axes[1, 0].plot(sorted_comb, cdf_comb, label="Combined (Title + Summary)", color="#2ca02c", lw=2)
            axes[1, 0].axvline(128, color="red", linestyle="--", label="L=128")
            axes[1, 0].axvline(256, color="purple", linestyle="-.", label="L=256")
            axes[1, 0].set_xlim(0, 300)
            axes[1, 0].set_title("Cumulative Token Distribution Function (CDF)", fontsize=12, fontweight="bold")
            axes[1, 0].set_xlabel("WordPiece Tokens")
            axes[1, 0].set_ylabel("Cumulative Probability (Coverage)")
            axes[1, 0].legend(loc="lower right")

            # 4. Word vs Token Scatter/Hexbin or Boxplot
            box_data = pd.DataFrame({
                "Headline Words": np.random.choice(title_words, size=min(10000, len(title_words))),
                "Headline Tokens": np.random.choice(title_tokens, size=min(10000, len(title_tokens))),
                "Combined Words": np.random.choice(comb_words, size=min(10000, len(comb_words))),
                "Combined Tokens": np.random.choice(comb_tokens, size=min(10000, len(comb_tokens)))
            })
            sns.boxplot(data=box_data, ax=axes[1, 1], palette="Set2")
            axes[1, 1].set_title("Word vs WordPiece Token Count Comparison", fontsize=12, fontweight="bold")
            axes[1, 1].set_ylabel("Count")
            axes[1, 1].tick_params(axis="x", rotation=15)

            plt.tight_layout()
            plt.savefig(plot_path_proc, dpi=300)
            plt.savefig(plot_path_anex, dpi=300)
            plt.close()
            print(f"\n[Plots] Visualizations saved to:\n  - {plot_path_proc}\n  - {plot_path_anex}")
        except Exception as e:
            print(f"[Warning] Could not generate plots: {e}")

    print(f"\n[Summary] JSON report: {json_path}")
    print(f"[Summary] CSV summary: {csv_path}")
    print(f"[Summary] Total elapsed time: {time.time() - start_time:.2f}s")
    print("=" * 80)
    
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EDA Headline and Summary Length Analysis")
    parser.add_argument("--parquet_path", type=str, default=str(DEFAULT_PARQUET_PATH), help="Path to input parquet")
    parser.add_argument("--sample_size", type=int, default=None, help="Sample size for tokenization (None for all)")
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Output directory")
    args = parser.parse_args()

    run_headline_length_eda(
        parquet_path=Path(args.parquet_path),
        output_dir=Path(args.output_dir),
        sample_size=args.sample_size
    )
