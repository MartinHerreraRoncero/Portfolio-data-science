import os
import sys
import glob
import shutil
import zipfile
import duckdb
from pathlib import Path
import pandas as pd
from tqdm import tqdm

ZIPS_DIR = Path("data/raw/sec_insider_zips")
TEMP_EXTRACT_DIR = Path("data/interim/temp_insider_extract")
PROCESSED_DIR = Path("data/processed/sec_dataset")
DUCKDB_PATH = PROCESSED_DIR / "sec_insider_monthly.duckdb"
PARQUET_PATH = PROCESSED_DIR / "sec_insider_monthly.parquet"


def process_sec_insider_streaming():
    """Processes SEC Insider Transactions (Forms 3, 4, 5) ZIP archives into DuckDB in streaming mode."""
    zip_files = sorted(list(ZIPS_DIR.glob("*.zip")))
    if not zip_files:
        print("[Error] No ZIP files found in data/raw/sec_insider_zips.")
        return

    print(f"[Insider ETL] Found {len(zip_files)} ZIP files to process.")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_EXTRACT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[Insider ETL] Connecting to DuckDB database at '{DUCKDB_PATH}'...")
    con = duckdb.connect(str(DUCKDB_PATH))
    con.execute("SET preserve_insertion_order = false;")
    con.execute("SET memory_limit = '6GB';")

    con.execute("""
    CREATE OR REPLACE TABLE sec_insider_raw (
        adsh VARCHAR,
        cik INTEGER,
        filing_date DATE,
        year INTEGER,
        month INTEGER,
        rptowner_cik INTEGER,
        is_ceo_cfo INTEGER,
        trans_code VARCHAR,
        trans_acquired_disp VARCHAR,
        trans_shares DOUBLE,
        trans_price DOUBLE,
        trans_value DOUBLE
    );
    """)

    insert_query = """
    INSERT INTO sec_insider_raw
    SELECT 
        s.ACCESSION_NUMBER as adsh,
        TRY_CAST(s.ISSUERCIK AS INTEGER) as cik,
        COALESCE(
            TRY_CAST(STRPTIME(CAST(s.FILING_DATE AS VARCHAR), '%d-%b-%Y') AS DATE),
            TRY_CAST(STRPTIME(CAST(s.FILING_DATE AS VARCHAR), '%Y-%m-%d') AS DATE),
            TRY_CAST(STRPTIME(CAST(s.FILING_DATE AS VARCHAR), '%Y%m%d') AS DATE)
        ) as filing_date,
        YEAR(COALESCE(
            TRY_CAST(STRPTIME(CAST(s.FILING_DATE AS VARCHAR), '%d-%b-%Y') AS DATE),
            TRY_CAST(STRPTIME(CAST(s.FILING_DATE AS VARCHAR), '%Y-%m-%d') AS DATE),
            TRY_CAST(STRPTIME(CAST(s.FILING_DATE AS VARCHAR), '%Y%m%d') AS DATE)
        )) as year,
        MONTH(COALESCE(
            TRY_CAST(STRPTIME(CAST(s.FILING_DATE AS VARCHAR), '%d-%b-%Y') AS DATE),
            TRY_CAST(STRPTIME(CAST(s.FILING_DATE AS VARCHAR), '%Y-%m-%d') AS DATE),
            TRY_CAST(STRPTIME(CAST(s.FILING_DATE AS VARCHAR), '%Y%m%d') AS DATE)
        )) as month,
        TRY_CAST(o.RPTOWNERCIK AS INTEGER) as rptowner_cik,
        CASE WHEN (
            UPPER(COALESCE(o.RPTOWNER_RELATIONSHIP, '')) LIKE '%CEO%' OR 
            UPPER(COALESCE(o.RPTOWNER_RELATIONSHIP, '')) LIKE '%CFO%' OR 
            UPPER(COALESCE(o.RPTOWNER_TITLE, '')) LIKE '%CEO%' OR 
            UPPER(COALESCE(o.RPTOWNER_TITLE, '')) LIKE '%CFO%' OR
            UPPER(COALESCE(o.RPTOWNER_TITLE, '')) LIKE '%CHIEF EXECUTIVE%' OR
            UPPER(COALESCE(o.RPTOWNER_TITLE, '')) LIKE '%CHIEF FINANCIAL%'
        ) THEN 1 ELSE 0 END as is_ceo_cfo,
        t.TRANS_CODE as trans_code,
        t.TRANS_ACQUIRED_DISP_CD as trans_acquired_disp,
        TRY_CAST(t.TRANS_SHARES AS DOUBLE) as trans_shares,
        TRY_CAST(t.TRANS_PRICEPERSHARE AS DOUBLE) as trans_price,
        (TRY_CAST(t.TRANS_SHARES AS DOUBLE) * COALESCE(TRY_CAST(t.TRANS_PRICEPERSHARE AS DOUBLE), 0.0)) as trans_value
    FROM read_csv_auto($1, delim='\t', header=True, all_varchar=True, ignore_errors=True, strict_mode=False) s
    JOIN read_csv_auto($2, delim='\t', header=True, all_varchar=True, ignore_errors=True, strict_mode=False) o
      ON s.ACCESSION_NUMBER = o.ACCESSION_NUMBER
    JOIN read_csv_auto($3, delim='\t', header=True, all_varchar=True, ignore_errors=True, strict_mode=False) t
      ON s.ACCESSION_NUMBER = t.ACCESSION_NUMBER
    WHERE s.ISSUERCIK IS NOT NULL 
      AND t.TRANS_SHARES IS NOT NULL;
    """

    success_zips = 0
    for zip_path in tqdm(zip_files, desc="Streaming Insider ZIPs into DuckDB"):
        if TEMP_EXTRACT_DIR.exists():
            shutil.rmtree(TEMP_EXTRACT_DIR, ignore_errors=True)
        TEMP_EXTRACT_DIR.mkdir(parents=True, exist_ok=True)

        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                namelist = zf.namelist()
                sub_f = "SUBMISSION.tsv" if "SUBMISSION.tsv" in namelist else ("SUBMISSION.txt" if "SUBMISSION.txt" in namelist else None)
                own_f = "REPORTINGOWNER.tsv" if "REPORTINGOWNER.tsv" in namelist else ("REPORTINGOWNER.txt" if "REPORTINGOWNER.txt" in namelist else None)
                trn_f = "NONDERIV_TRANS.tsv" if "NONDERIV_TRANS.tsv" in namelist else ("NONDERIV_TRANS.txt" if "NONDERIV_TRANS.txt" in namelist else None)

                if sub_f and own_f and trn_f:
                    zf.extract(sub_f, TEMP_EXTRACT_DIR)
                    zf.extract(own_f, TEMP_EXTRACT_DIR)
                    zf.extract(trn_f, TEMP_EXTRACT_DIR)

                    sub_path = str(TEMP_EXTRACT_DIR / sub_f)
                    own_path = str(TEMP_EXTRACT_DIR / own_f)
                    trn_path = str(TEMP_EXTRACT_DIR / trn_f)

                    con.execute(insert_query, [sub_path, own_path, trn_path])
                    success_zips += 1
                else:
                    print(f"[Warning] Skipping {zip_path.name}: TSV tables not found.")

        except Exception as e:
            print(f"[Error] Failed to process {zip_path.name}: {e}")
        finally:
            if TEMP_EXTRACT_DIR.exists():
                shutil.rmtree(TEMP_EXTRACT_DIR, ignore_errors=True)

    print("\n[Insider ETL] Aggregating monthly insider metrics by (CIK, Year, Month)...")
    
    con.execute("""
    CREATE OR REPLACE TABLE sec_insider_monthly AS
    SELECT 
        cik,
        year as filed_year,
        month as filed_month,
        SUM(CASE WHEN trans_acquired_disp = 'A' THEN trans_shares ELSE -trans_shares END) as insider_net_shares,
        SUM(CASE WHEN trans_acquired_disp = 'A' THEN trans_value ELSE -trans_value END) as insider_net_value_usd,
        ROUND(SUM(CASE WHEN trans_acquired_disp = 'A' THEN 1.0 ELSE 0.0 END) / NULLIF(COUNT(*), 0), 4) as insider_buy_ratio,
        COUNT(DISTINCT CASE WHEN trans_acquired_disp = 'D' THEN rptowner_cik END) as insider_num_sellers,
        MAX(CASE WHEN is_ceo_cfo = 1 AND trans_acquired_disp = 'D' THEN 1 ELSE 0 END) as insider_ceo_cfo_sale_flag
    FROM sec_insider_raw
    WHERE cik IS NOT NULL AND year IS NOT NULL AND month IS NOT NULL
    GROUP BY cik, year, month;
    """)

    monthly_rows = con.execute("SELECT COUNT(*) FROM sec_insider_monthly").fetchone()[0]
    unique_ciks = con.execute("SELECT COUNT(DISTINCT cik) FROM sec_insider_monthly").fetchone()[0]

    print("\n" + "=" * 60)
    print("INSIDER ETL SUMMARY STATISTICS:")
    print("=" * 60)
    print(f"  - Total ZIP archives processed: {success_zips}/{len(zip_files)}")
    print(f"  - Monthly Insider Aggregations: {monthly_rows:,}")
    print(f"  - Unique Companies (CIKs) with Insider Activity: {unique_ciks:,}")

    print(f"\n[Insider ETL] Exporting consolidated insider dataset to Parquet: {PARQUET_PATH}...")
    con.execute(f"COPY sec_insider_monthly TO '{PARQUET_PATH}' (FORMAT PARQUET, COMPRESSION SNAPPY);")
    
    parquet_mb = PARQUET_PATH.stat().st_size / (1024 * 1024)
    print(f"  - Parquet export completed! Size: {parquet_mb:.2f} MB")

    con.close()
    print("[Insider ETL] Pipeline finished successfully!")


def main():
    print("=" * 60)
    print("SEC INSIDER TRANSACTIONS (FORMS 3, 4, 5) STREAMING ETL PIPELINE")
    print("=" * 60)
    
    process_sec_insider_streaming()


if __name__ == "__main__":
    main()
