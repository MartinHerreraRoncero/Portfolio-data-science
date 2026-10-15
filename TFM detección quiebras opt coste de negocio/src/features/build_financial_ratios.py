import os
import sys
import duckdb
import pandas as pd
from pathlib import Path

PIVOTED_PARQUET = Path("data/processed/sec_dataset/v2_sec_financials_pivoted.parquet")
OUTPUT_PIVOTED_RATIOS_PARQUET = Path("data/processed/sec_dataset/v2_sec_financials_pivoted.parquet")


def build_financial_ratios():
    print("=" * 65)
    print("FASE 3: FEATURE ENGINEERING - RATIOS FINANCIEROS Y MODELOS DE QUEBRA")
    print("=" * 65)

    if not PIVOTED_PARQUET.exists():
        print(f"[Error] Source parquet dataset not found at {PIVOTED_PARQUET}")
        return

    con = duckdb.connect()

    print("[Ratio Engine] Calculating classic insolvency scores (Altman, Ohlson, Springate, Zmijewski) and financial ratios...")

    con.execute(f"""
    CREATE OR REPLACE TABLE v2_sec_financials_with_ratios AS
    SELECT 
        f.*,
        
        -- Derived Total Liabilities (Liabilities & Equity - Equity)
        COALESCE(
            NULLIF(f.tag_LiabilitiesAndStockholdersEquity - f.tag_StockholdersEquity, 0.0),
            f.tag_LiabilitiesCurrent,
            0.0
        ) as derived_total_liabilities,

        -- 1. Solvency & Liquidity Ratios
        COALESCE(NULLIF(f.tag_WorkingCapital / NULLIF(f.tag_Assets, 0.0), 0.0), 0.0) as ratio_working_capital_to_assets,
        COALESCE(NULLIF(COALESCE(NULLIF(f.tag_LiabilitiesAndStockholdersEquity - f.tag_StockholdersEquity, 0.0), f.tag_LiabilitiesCurrent, 0.0) / NULLIF(f.tag_Assets, 0.0), 0.0), 0.0) as ratio_leverage,
        COALESCE(NULLIF(f.tag_AssetsCurrent / NULLIF(f.tag_LiabilitiesCurrent, 0.0), 0.0), 0.0) as ratio_current_ratio,
        COALESCE(NULLIF(f.tag_CashAndCashEquivalentsAtCarryingValue / NULLIF(f.tag_LiabilitiesCurrent, 0.0), 0.0), 0.0) as ratio_quick_cash_ratio,
        COALESCE(NULLIF(COALESCE(NULLIF(f.tag_LiabilitiesAndStockholdersEquity - f.tag_StockholdersEquity, 0.0), f.tag_LiabilitiesCurrent, 0.0) / NULLIF(f.tag_StockholdersEquity, 0.0), 0.0), 0.0) as ratio_debt_to_equity,
        
        -- 2. Profitability Ratios
        COALESCE(NULLIF(f.tag_NetIncomeLoss / NULLIF(f.tag_Assets, 0.0), 0.0), 0.0) as ratio_roa,
        COALESCE(NULLIF(f.tag_NetIncomeLoss / NULLIF(f.tag_StockholdersEquity, 0.0), 0.0), 0.0) as ratio_roe,
        COALESCE(NULLIF(f.tag_EBIT_combined / NULLIF(f.tag_Assets, 0.0), 0.0), 0.0) as ratio_ebit_to_assets,
        COALESCE(NULLIF(f.tag_EBIT_combined / NULLIF(f.tag_Revenues_combined, 0.0), 0.0), 0.0) as ratio_operating_margin,
        COALESCE(NULLIF(f.tag_Revenues_combined / NULLIF(f.tag_Assets, 0.0), 0.0), 0.0) as ratio_asset_turnover,

        -- 3. Cash Flow Ratios
        COALESCE(NULLIF(f.tag_NetCashProvidedByUsedInOperatingActivities / NULLIF(COALESCE(NULLIF(f.tag_LiabilitiesAndStockholdersEquity - f.tag_StockholdersEquity, 0.0), f.tag_LiabilitiesCurrent, 0.0), 0.0), 0.0), 0.0) as ratio_ocf_to_liabilities,

        -- 4. Altman Z-Score (1968 / 1993 Model)
        -- Z = 1.2(WC/Assets) + 1.4(RE/Assets) + 3.3(EBIT/Assets) + 0.6(Market Cap/Liabilities) + 0.999(Sales/Assets)
        (
            1.2 * COALESCE(f.tag_WorkingCapital / NULLIF(f.tag_Assets, 0.0), 0.0) +
            1.4 * COALESCE(f.tag_RetainedEarningsAccumulatedDeficit / NULLIF(f.tag_Assets, 0.0), 0.0) +
            3.3 * COALESCE(f.tag_EBIT_combined / NULLIF(f.tag_Assets, 0.0), 0.0) +
            0.6 * COALESCE(f.market_cap / NULLIF(COALESCE(NULLIF(f.tag_LiabilitiesAndStockholdersEquity - f.tag_StockholdersEquity, 0.0), f.tag_LiabilitiesCurrent, 0.0), 0.0), 0.0) +
            0.999 * COALESCE(f.tag_Revenues_combined / NULLIF(f.tag_Assets, 0.0), 0.0)
        ) as altman_z_score,

        -- 5. Ohlson O-Score (1980 Logit Proxy)
        (
            -1.32 
            - 0.407 * LN(GREATEST(1.0, f.tag_Assets / 1000.0))
            + 6.03 * COALESCE(COALESCE(NULLIF(f.tag_LiabilitiesAndStockholdersEquity - f.tag_StockholdersEquity, 0.0), f.tag_LiabilitiesCurrent, 0.0) / NULLIF(f.tag_Assets, 0.0), 0.0)
            - 1.43 * COALESCE(f.tag_WorkingCapital / NULLIF(f.tag_Assets, 0.0), 0.0)
            + 0.0757 * COALESCE(f.tag_LiabilitiesCurrent / NULLIF(f.tag_AssetsCurrent, 0.0), 0.0)
            - 1.72 * CASE WHEN COALESCE(NULLIF(f.tag_LiabilitiesAndStockholdersEquity - f.tag_StockholdersEquity, 0.0), f.tag_LiabilitiesCurrent, 0.0) > f.tag_Assets THEN 1.0 ELSE 0.0 END
            - 2.37 * COALESCE(f.tag_NetIncomeLoss / NULLIF(f.tag_Assets, 0.0), 0.0)
            - 1.83 * COALESCE(f.tag_NetCashProvidedByUsedInOperatingActivities / NULLIF(COALESCE(NULLIF(f.tag_LiabilitiesAndStockholdersEquity - f.tag_StockholdersEquity, 0.0), f.tag_LiabilitiesCurrent, 0.0), 0.0), 0.0)
        ) as ohlson_o_score,

        -- 6. Springate Score (1978 Discriminant Model)
        (
            1.03 * COALESCE(f.tag_WorkingCapital / NULLIF(f.tag_Assets, 0.0), 0.0) +
            3.07 * COALESCE(f.tag_EBIT_combined / NULLIF(f.tag_Assets, 0.0), 0.0) +
            0.66 * COALESCE(f.tag_NetIncomeLoss / NULLIF(f.tag_LiabilitiesCurrent, 0.0), 0.0) +
            0.40 * COALESCE(f.tag_Revenues_combined / NULLIF(f.tag_Assets, 0.0), 0.0)
        ) as springate_score,

        -- 7. Zmijewski Score (1984 Probit Model)
        (
            -4.336
            - 4.513 * COALESCE(f.tag_NetIncomeLoss / NULLIF(f.tag_Assets, 0.0), 0.0)
            + 5.679 * COALESCE(COALESCE(NULLIF(f.tag_LiabilitiesAndStockholdersEquity - f.tag_StockholdersEquity, 0.0), f.tag_LiabilitiesCurrent, 0.0) / NULLIF(f.tag_Assets, 0.0), 0.0)
            + 0.004 * COALESCE(f.tag_AssetsCurrent / NULLIF(f.tag_LiabilitiesCurrent, 0.0), 0.0)
        ) as zmijewski_score

    FROM read_parquet('{PIVOTED_PARQUET}') f;
    """)

    # Overwrite master Parquet
    con.execute(f"COPY v2_sec_financials_with_ratios TO '{OUTPUT_PIVOTED_RATIOS_PARQUET}' (FORMAT PARQUET, COMPRESSION SNAPPY);")

    total_rows = con.execute("SELECT COUNT(*) FROM v2_sec_financials_with_ratios").fetchone()[0]
    total_cols = con.execute("SELECT COUNT(*) FROM information_schema.columns WHERE table_name = 'v2_sec_financials_with_ratios'").fetchone()[0]
    parquet_mb = OUTPUT_PIVOTED_RATIOS_PARQUET.stat().st_size / (1024 * 1024)

    print("\n" + "=" * 65)
    print("FINANCIAL RATIOS FEATURE ENGINEERING SUMMARY:")
    print("=" * 65)
    print(f"  - Total Observations: {total_rows:,}")
    print(f"  - Total Columns in Parquet: {total_cols} (1 ID adsh + {total_cols-1} Multimodal Features)")
    print(f"  - Parquet Size: {parquet_mb:.2f} MB")
    print(f"  - Calculated Scores: Altman Z-Score, Ohlson O-Score, Springate Score, Zmijewski Score + 11 Solvency/Profitability Ratios")
    print("=" * 65)

    # Display sample scores
    df_sample = con.execute("""
        SELECT company_name, filed_date, altman_z_score, ohlson_o_score, springate_score, zmijewski_score, target_bankrupt_12m 
        FROM v2_sec_financials_with_ratios 
        WHERE altman_z_score != 0 
        LIMIT 10
    """).df()
    print(df_sample.to_string(index=False))

    con.close()
    print("[Ratio Engine] Feature engineering completed successfully!")


if __name__ == "__main__":
    build_financial_ratios()
