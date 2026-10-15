import os
import sys
import duckdb
import pandas as pd
import requests
from pathlib import Path
from tqdm import tqdm

DUCKDB_PATH = Path("data/processed/sec_dataset/sec_financials.duckdb")
OUTPUT_PIVOT_PARQUET = Path("data/processed/sec_dataset/v2_sec_financials_pivoted.parquet")
OUTPUT_CLEAN_PARQUET = Path("data/processed/sec_dataset/v2_sec_financials_pivoted_clean.parquet")
OUTPUT_MASTER_V21_PARQUET = Path("data/processed/sec_dataset/v2.1_master_financials_dataset.parquet")

NET_FEATURES_PARQUET = Path("data/processed/sec_dataset/network_features_monthly.parquet")
NLP_DISTRESS_PARQUET = Path("data/processed/news_dataset/monthly_nlp_distress_features.parquet")
MARKET_QUOTES_PARQUET = Path("data/processed/sec_dataset/sec_market_quotes_monthly.parquet")
BANKRUPTCY_PARQUET = Path("data/processed/sec_dataset/sec_bankruptcy_events.parquet")


def build_master_v21_dataset():
    print("=" * 80)
    print("FASE 2: CONSTRUCCIÓN DEL MASTER DATASET V2.1 MULTIMODAL EN DUCKDB")
    print("=" * 80)

    con = duckdb.connect()
    con.execute("SET memory_limit = '10GB';")

    print("\n[DuckDB Assembler] 1. Leyendo datos financieros base SEC con GMM...")
    if OUTPUT_PIVOT_PARQUET.exists():
        sec_table = f"'{OUTPUT_PIVOT_PARQUET}'"
    else:
        raise FileNotFoundError(f"No se encontró {OUTPUT_PIVOT_PARQUET}")

    print("[DuckDB Assembler] 2. Verificando fuentes de datos complementarias:")
    print(f"  - Redes Complejas: {NET_FEATURES_PARQUET.exists()} ({NET_FEATURES_PARQUET})")
    print(f"  - NLP & BERTopic Distress: {NLP_DISTRESS_PARQUET.exists()} ({NLP_DISTRESS_PARQUET})")

    # Assemble unified master view
    print("\n[DuckDB Assembler] 3. Ejecutando ensamble relacional de alta velocidad en DuckDB...")
    query = f"""
    CREATE OR REPLACE TABLE master_v21_dataset AS
    WITH sec AS (
        SELECT * FROM {sec_table}
    ),
    net AS (
        SELECT 
            adsh,
            ticker,
            network_pagerank,
            network_betweenness,
            network_degree,
            network_community_id,
            cluster_distress_infection_rate,
            is_network_imputed
        FROM '{NET_FEATURES_PARQUET}'
    ),
    nlp AS (
        SELECT 
            ticker,
            year,
            month,
            news_count_30d,
            news_count_90d,
            news_effective_volume_decayed_30d,
            nlp_sentiment_decayed_30d,
            nlp_sentiment_decayed_90d,
            nlp_distress_topic_intensity,
            nlp_distress_topic_intensity_decayed_30d,
            distress_default_decayed_30d,
            distress_chapter11_decayed_30d,
            distress_litigation_decayed_30d,
            nlp_distress_topic_intensity_decayed_90d,
            distress_news_count_30d,
            distress_news_count_90d,
            news_negative_ratio_30d,
            media_silence_months_count,
            is_media_silent_3m,
            is_media_silent_6m,
            media_coverage_ratio_12m
        FROM '{NLP_DISTRESS_PARQUET}'
    )
    SELECT 
        s.*,
        n.ticker,
        COALESCE(n.network_pagerank, 0.000330) as network_pagerank,
        COALESCE(n.network_betweenness, 0.000885) as network_betweenness,
        COALESCE(n.network_degree, 1.0) as network_degree,
        COALESCE(n.network_community_id, -1) as network_community_id,
        COALESCE(n.cluster_distress_infection_rate, 0.0) as cluster_distress_infection_rate,
        COALESCE(n.is_network_imputed, 1) as is_network_imputed,
        COALESCE(nlp.news_count_30d, 0) as news_count_30d,
        COALESCE(nlp.news_count_90d, 0) as news_count_90d,
        COALESCE(nlp.news_effective_volume_decayed_30d, 0.0) as news_effective_volume_decayed_30d,
        COALESCE(nlp.nlp_sentiment_decayed_30d, s.market_news_sentiment_mean, 0.05) as nlp_sentiment_decayed_30d,
        COALESCE(nlp.nlp_sentiment_decayed_90d, s.market_news_sentiment_mean, 0.05) as nlp_sentiment_decayed_90d,
        COALESCE(nlp.nlp_distress_topic_intensity, 0.0) as nlp_distress_topic_intensity,
        COALESCE(nlp.nlp_distress_topic_intensity_decayed_30d, 0.0) as nlp_distress_topic_intensity_decayed_30d,
        COALESCE(nlp.distress_default_decayed_30d, 0.0) as distress_default_decayed_30d,
        COALESCE(nlp.distress_chapter11_decayed_30d, 0.0) as distress_chapter11_decayed_30d,
        COALESCE(nlp.distress_litigation_decayed_30d, 0.0) as distress_litigation_decayed_30d,
        COALESCE(nlp.nlp_distress_topic_intensity_decayed_90d, 0.0) as nlp_distress_topic_intensity_decayed_90d,
        COALESCE(nlp.distress_news_count_30d, 0.0) as distress_news_count_30d,
        COALESCE(nlp.distress_news_count_90d, 0.0) as distress_news_count_90d,
        COALESCE(nlp.news_negative_ratio_30d, 0.0) as news_negative_ratio_30d,
        COALESCE(nlp.media_silence_months_count, 12) as media_silence_months_count,
        COALESCE(nlp.is_media_silent_3m, 1) as is_media_silent_3m,
        COALESCE(nlp.is_media_silent_6m, 1) as is_media_silent_6m,
        COALESCE(nlp.media_coverage_ratio_12m, 0.0) as media_coverage_ratio_12m
    FROM sec s
    LEFT JOIN net n ON s.adsh = n.adsh
    -- Point-in-time strictly aligned: join with closed month t-1 relative to filed_date to prevent intra-month look-ahead leakage
    LEFT JOIN nlp ON n.ticker = nlp.ticker 
        AND YEAR(s.filed_date - INTERVAL 1 MONTH) = nlp.year 
        AND MONTH(s.filed_date - INTERVAL 1 MONTH) = nlp.month;
    """
    con.execute(query)

    row_count = con.execute("SELECT COUNT(*) FROM master_v21_dataset").fetchone()[0]
    col_count = con.execute("SELECT COUNT(*) FROM (DESCRIBE master_v21_dataset)").fetchone()[0]
    print(f"[DuckDB Assembler] Master V2.1 creado exitosamente: {row_count:,} registros, {col_count} columnas.")

    # Write to Parquets
    print(f"\n[DuckDB Assembler] 4. Exportando a Parquet máster...")
    con.execute(f"COPY master_v21_dataset TO '{OUTPUT_PIVOT_PARQUET}' (FORMAT PARQUET, COMPRESSION SNAPPY);")
    con.execute(f"COPY master_v21_dataset TO '{OUTPUT_CLEAN_PARQUET}' (FORMAT PARQUET, COMPRESSION SNAPPY);")
    con.execute(f"COPY master_v21_dataset TO '{OUTPUT_MASTER_V21_PARQUET}' (FORMAT PARQUET, COMPRESSION SNAPPY);")

    size_mb = OUTPUT_PIVOT_PARQUET.stat().st_size / (1024 * 1024)
    print(f"  - Guardado en {OUTPUT_PIVOT_PARQUET} ({size_mb:.2f} MB)")
    print(f"  - Guardado en {OUTPUT_CLEAN_PARQUET}")
    print(f"  - Guardado en {OUTPUT_MASTER_V21_PARQUET}")

    con.close()
    print("\n[DuckDB Assembler] ¡Fase 2 completada con éxito!")


if __name__ == "__main__":
    build_master_v21_dataset()
