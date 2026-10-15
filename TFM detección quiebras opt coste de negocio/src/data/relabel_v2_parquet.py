import os
import sys
import duckdb
import pandas as pd
from pathlib import Path

PARQUET_PATH = Path("data/processed/sec_dataset/v2_sec_financials_pivoted.parquet")

def relabel_v2_with_triple_trigger_distress():
    print("=" * 70)
    print("[Relabel Engine] Updating V2 Parquet with Triple-Trigger SEC Distress Target...")
    print("=" * 70)

    con = duckdb.connect()

    # 1. Strict chronological filtering and restatement-bias-aware deduplication
    print("Filtering inverted dates (period_date > filed_date) and deduplicating filings...")
    print("Restricting restatement bias: amendments (>180d lag) cannot overwrite original historical filings...")
    con.execute(f"""
    CREATE OR REPLACE TABLE master_dedup AS
    SELECT * EXCLUDE (row_num, base_form)
    FROM (
        SELECT *, 
               CASE 
                   WHEN form_type IN ('10-K', '10-K/A') THEN '10-K'
                   WHEN form_type IN ('10-Q', '10-Q/A') THEN '10-Q'
                   ELSE form_type 
               END as base_form,
               ROW_NUMBER() OVER (
                   PARTITION BY cik, period_date, 
                       CASE 
                           WHEN form_type IN ('10-K', '10-K/A') THEN '10-K'
                           WHEN form_type IN ('10-Q', '10-Q/A') THEN '10-Q'
                           ELSE form_type 
                       END
                   ORDER BY 
                       -- Restatement bias prevention:
                       -- Timely filings & amendments within 0-180 days have priority 1 (timely reporting window).
                       -- Late amendments (pub_lag_days > 180) are demoted to priority 2, preventing them from overwriting original balances.
                       CASE 
                           WHEN form_type IN ('10-K/A', '10-Q/A') AND (pub_lag_days > 180 OR pub_lag_days < 0) THEN 2
                           ELSE 1 
                       END ASC,
                       -- Within the same priority, take the most recent timely filing
                       filed_date DESC, 
                       adsh DESC
               ) as row_num
        FROM '{PARQUET_PATH}'
        -- Strict filter: eliminate 16 inverted date records where period_date > filed_date (pub_lag_days < 0)
        WHERE filed_date >= period_date
    )
    WHERE row_num = 1;
    """)

    dedup_count = con.execute("SELECT COUNT(*) FROM master_dedup;").fetchone()[0]
    print(f"  - Clean deduplicated row count: {dedup_count:,}")

    # 2. Calculate Puntual Triple-Trigger Technical Distress Flag (C1 + C2 + C3)
    # C1: StockholdersEquity <= 0 OR Liabilities >= Assets
    # C2: Merton Distance to Default <= 0.5
    # C3: Working Capital < 0 AND (Operating Income < Interest Expense OR Operating Income < 0)
    print("Calculating Puntual Triple-Trigger Technical Distress Flag (C1 + C2 + C3)...")
    con.execute("""
    CREATE OR REPLACE TABLE master_flagged AS
    SELECT *,
        CASE WHEN 
            -- C1: Patrimonio Neto <= 0 OR Pasivo Total >= Activo Total
            (tag_StockholdersEquity <= 0 OR (COALESCE(tag_LiabilitiesAndStockholdersEquity - tag_StockholdersEquity, tag_LiabilitiesCurrent, tag_Assets - tag_StockholdersEquity) >= tag_Assets)) AND
            -- C2: Merton Distance to Default <= 0.5
            (merton_distance_to_default <= 0.5) AND
            -- C3: Working Capital < 0 AND (Net Income < Interest Expense OR Net Income < 0)
            (tag_WorkingCapital < 0 AND (tag_NetIncomeLoss < COALESCE(tag_InterestExpense, 0) OR tag_NetIncomeLoss < 0))
        THEN 1 ELSE 0 END as is_distress_event
    FROM master_dedup;
    """)

    # 3. Calculate Prospective 12M and 24M Targets per CIK (Resolving Terminal Liquidation Paradox)
    print("Computing prospective target_bankrupt_12m and target_bankrupt_24m (Resolving Terminal Liquidation Paradox)...")
    con.execute("""
    CREATE OR REPLACE TABLE master_relabeled AS
    SELECT 
        f.* EXCLUDE (target_bankrupt_12m, target_bankrupt_24m),
        CASE WHEN EXISTS (
            SELECT 1 FROM master_flagged e
            WHERE e.cik = f.cik
              AND e.is_distress_event = 1
              AND e.filed_date > f.filed_date
              AND e.filed_date <= f.filed_date + INTERVAL 12 MONTH
        ) OR (
            f.is_distress_event = 1
            AND NOT EXISTS (
                SELECT 1 FROM master_flagged e
                WHERE e.cik = f.cik
                  AND e.filed_date > f.filed_date
            )
        ) THEN 1 ELSE 0 END as target_bankrupt_12m,
        CASE WHEN EXISTS (
            SELECT 1 FROM master_flagged e
            WHERE e.cik = f.cik
              AND e.is_distress_event = 1
              AND e.filed_date > f.filed_date
              AND e.filed_date <= f.filed_date + INTERVAL 24 MONTH
        ) OR (
            f.is_distress_event = 1
            AND NOT EXISTS (
                SELECT 1 FROM master_flagged e
                WHERE e.cik = f.cik
                  AND e.filed_date > f.filed_date
            )
        ) THEN 1 ELSE 0 END as target_bankrupt_24m
    FROM master_flagged f;
    """)

    # 4. Print Statistics
    stats = con.execute("""
    SELECT 
        COUNT(*) as total_obs,
        SUM(is_distress_event) as total_distress_events,
        SUM(target_bankrupt_12m) as pos_12m,
        AVG(target_bankrupt_12m) * 100 as rate_12m_pct,
        SUM(target_bankrupt_24m) as pos_24m,
        AVG(target_bankrupt_24m) * 100 as rate_24m_pct
    FROM master_relabeled;
    """).df()

    print("\n[Relabel Engine] Triple-Trigger SEC Distress Target Statistics:")
    print(f"  - Total Observations: {stats['total_obs'][0]:,}")
    print(f"  - Total Puntual Distress Events (t): {stats['total_distress_events'][0]:,}")
    print(f"  - Target 12M Distress (+): {stats['pos_12m'][0]:,} ({stats['rate_12m_pct'][0]:.2f}%)")
    print(f"  - Target 24M Distress (+): {stats['pos_24m'][0]:,} ({stats['rate_24m_pct'][0]:.2f}%)")

    # Sample distribution by CIK
    df_ciks = con.execute("""
    SELECT cik, company_name, SUM(target_bankrupt_12m) as pos_12m, SUM(target_bankrupt_24m) as pos_24m
    FROM master_relabeled
    WHERE target_bankrupt_12m = 1
    GROUP BY cik, company_name
    ORDER BY pos_12m DESC
    LIMIT 15;
    """).df()

    print("\nTop 15 Distressed CIKs (Target 12M):")
    print(df_ciks.to_string(index=False))

    # Overwrite master Parquet and Clean Parquet
    CLEAN_PARQUET_PATH = Path("data/processed/sec_dataset/v2_sec_financials_pivoted_clean.parquet")
    con.execute(f"COPY master_relabeled TO '{PARQUET_PATH.as_posix()}' (FORMAT PARQUET, COMPRESSION SNAPPY);")
    con.execute(f"COPY master_relabeled TO '{CLEAN_PARQUET_PATH.as_posix()}' (FORMAT PARQUET, COMPRESSION SNAPPY);")
    con.close()

    print(f"\n[Relabel Engine] Master V2 Parquets updated successfully at {PARQUET_PATH} and {CLEAN_PARQUET_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    relabel_v2_with_triple_trigger_distress()
