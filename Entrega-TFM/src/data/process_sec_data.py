import os
import sys
import glob
import shutil
import zipfile
import duckdb
from pathlib import Path
import pandas as pd
from tqdm import tqdm

ZIPS_DIR = Path("data/raw/sec_zips")
TEMP_EXTRACT_DIR = Path("data/interim/temp_extract")
PROCESSED_DIR = Path("data/processed/sec_dataset")
DUCKDB_PATH = PROCESSED_DIR / "sec_financials.duckdb"
PARQUET_PATH = PROCESSED_DIR / "sec_financials_2009_2026.parquet"


def process_sec_data_streaming():
    """Processes 79 SEC ZIP datasets sequentially into DuckDB to minimize disk space footprint."""
    zip_files = sorted(list(ZIPS_DIR.glob("*.zip")))
    if not zip_files:
        print("[Error] No ZIP files found in data/raw/sec_zips.")
        return

    print(f"[Streaming ETL] Found {len(zip_files)} ZIP files to process.")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_EXTRACT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[Streaming ETL] Connecting to DuckDB database at '{DUCKDB_PATH}'...")
    con = duckdb.connect(str(DUCKDB_PATH))
    
    con.execute("SET preserve_insertion_order = false;")
    con.execute("SET memory_limit = '6GB';")

    # Create target table if not exists
    con.execute("""
    CREATE OR REPLACE TABLE sec_financials (
        adsh VARCHAR,
        cik INTEGER,
        company_name VARCHAR,
        sic INTEGER,
        form VARCHAR,
        period_date DATE,
        filed_date DATE,
        pub_lag_days INTEGER,
        fy INTEGER,
        fp VARCHAR,
        tag VARCHAR,
        version VARCHAR,
        ddate DATE,
        qtrs INTEGER,
        uom VARCHAR,
        value DOUBLE
    );
    """)

    insert_query = """
    INSERT INTO sec_financials
    SELECT 
        s.adsh,
        TRY_CAST(s.cik AS INTEGER) as cik,
        s.name as company_name,
        TRY_CAST(s.sic AS INTEGER) as sic,
        s.form,
        TRY_CAST(STRPTIME(CAST(s.period AS VARCHAR), '%Y%m%d') AS DATE) as period_date,
        TRY_CAST(STRPTIME(CAST(s.filed AS VARCHAR), '%Y%m%d') AS DATE) as filed_date,
        DATEDIFF('day', 
                 TRY_CAST(STRPTIME(CAST(s.period AS VARCHAR), '%Y%m%d') AS DATE), 
                 TRY_CAST(STRPTIME(CAST(s.filed AS VARCHAR), '%Y%m%d') AS DATE)) as pub_lag_days,
        TRY_CAST(s.fy AS INTEGER) as fy,
        s.fp,
        n.tag,
        n.version,
        TRY_CAST(STRPTIME(CAST(n.ddate AS VARCHAR), '%Y%m%d') AS DATE) as ddate,
        TRY_CAST(n.qtrs AS INTEGER) as qtrs,
        n.uom,
        TRY_CAST(n.value AS DOUBLE) as value
    FROM read_csv_auto($1, delim='\t', header=True, all_varchar=True, ignore_errors=True, strict_mode=False) s
    JOIN read_csv_auto($2, delim='\t', header=True, all_varchar=True, ignore_errors=True, strict_mode=False) n
      ON s.adsh = n.adsh
    WHERE s.form IN ('10-K', '10-Q', '10-K/A', '10-Q/A')
      AND n.value IS NOT NULL;
    """

    success_zips = 0
    for zip_path in tqdm(zip_files, desc="Streaming ZIP datasets into DuckDB"):
        if TEMP_EXTRACT_DIR.exists():
            shutil.rmtree(TEMP_EXTRACT_DIR, ignore_errors=True)
        TEMP_EXTRACT_DIR.mkdir(parents=True, exist_ok=True)

        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                namelist = zf.namelist()
                
                # Determine sub filename (.tsv or .txt)
                sub_name = "sub.tsv" if "sub.tsv" in namelist else ("sub.txt" if "sub.txt" in namelist else None)
                num_name = "num.tsv" if "num.tsv" in namelist else ("num.txt" if "num.txt" in namelist else None)
                
                if sub_name and num_name:
                    zf.extract(sub_name, TEMP_EXTRACT_DIR)
                    zf.extract(num_name, TEMP_EXTRACT_DIR)
                    sub_file = TEMP_EXTRACT_DIR / sub_name
                    num_file = TEMP_EXTRACT_DIR / num_name
                    
                    con.execute(insert_query, [str(sub_file), str(num_file)])
                    success_zips += 1
                else:
                    print(f"[Warning] Skipping {zip_path.name}: sub or num file not found in archive.")

        except Exception as e:
            print(f"[Error] Failed to process {zip_path.name}: {e}")
        finally:
            if TEMP_EXTRACT_DIR.exists():
                shutil.rmtree(TEMP_EXTRACT_DIR, ignore_errors=True)

    print("\n" + "=" * 60)
    print("ETL SUMMARY STATISTICS:")
    print("=" * 60)
    
    row_count = con.execute("SELECT COUNT(*) FROM sec_financials").fetchone()[0]
    cik_count = con.execute("SELECT COUNT(DISTINCT cik) FROM sec_financials").fetchone()[0]
    min_filed = con.execute("SELECT MIN(filed_date) FROM sec_financials").fetchone()[0]
    max_filed = con.execute("SELECT MAX(filed_date) FROM sec_financials").fetchone()[0]
    avg_lag_res = con.execute("SELECT AVG(pub_lag_days) FROM sec_financials WHERE pub_lag_days >= 0 AND pub_lag_days < 365").fetchone()[0]
    avg_lag = avg_lag_res if avg_lag_res is not None else 0.0

    print(f"  - Total processed ZIP archives: {success_zips}/{len(zip_files)}")
    print(f"  - Total consolidated rows: {row_count:,}")
    print(f"  - Unique CIKs (Companies): {cik_count:,}")
    print(f"  - SEC Filing Date Range: {min_filed} to {max_filed}")
    print(f"  - Average Publication Lag: {avg_lag:.1f} days")

    if row_count > 0:
        print(f"\n[Streaming ETL] Exporting consolidated dataset to Parquet: {PARQUET_PATH}...")
        con.execute(f"COPY sec_financials TO '{PARQUET_PATH}' (FORMAT PARQUET, COMPRESSION SNAPPY);")
        parquet_size_mb = PARQUET_PATH.stat().st_size / (1024 * 1024)
        print(f"  - Parquet export completed! Size: {parquet_size_mb:.2f} MB")

    con.close()
    print("[Streaming ETL] Pipeline finished successfully!")


def main():
    print("=" * 60)
    print("SEC FINANCIAL STATEMENT DATA SETS (2009-2026) STREAMING ETL PIPELINE")
    print("=" * 60)
    
    process_sec_data_streaming()


if __name__ == "__main__":
    main()
