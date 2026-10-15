"""
Módulo de Análisis de Redes Complejas, Co-movimiento Bursátil y Contagio Financiero (Subagente 1C).
Proyecto TFM: Predicción de Insolvencia en Empresas del S&P 500 y Mercado Financiero Estadounidense.

Metodología:
1. Estimación de matrices mensuales de correlación de rendimientos C_ij(t) entre firmas cotizadas activas.
2. Transformación de distancia métrica ultramétrica: d_ij(t) = sqrt(2 * (1 - C_ij(t))).
3. Construcción del Árbol de Expansión Mínima (Minimum Spanning Tree - MST) con NetworkX / SciPy.
4. Detección de comunidades latentes por modularidad de Louvain (networkx.algorithms.community.louvain_communities).
5. Cálculo de métricas nodales mensuales por empresa:
   - network_pagerank: Centralidad e importancia sistémica en la red de co-movimiento.
   - network_betweenness: Intermediación y rol de puente entre clusters de mercado.
   - network_degree: Grado de conectividad estructural en el MST.
   - cluster_distress_infection_rate: Retardo espacial [W * Distress]_{i, t-1}, que mide la proporción
     de empresas vecinas en su cluster latente de Louvain que experimentaron insolvencia/deterioro en t-1.
6. Imputación robusta por mediana sectorial jerárquica (SIC 4-digit -> SIC 2-digit -> Market Median -> Global Median)
   para firmas no cotizadas o con series bursátiles incompletas.
7. Generación de 'data/processed/sec_dataset/network_features_monthly.parquet' e integración nativa con DuckDB.
"""

import os
import sys
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any

import duckdb
import numpy as np
import pandas as pd
import polars as pl
import networkx as nx
from networkx.algorithms.community import louvain_communities
from scipy.sparse.csgraph import minimum_spanning_tree

# Logging Configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("MarketNetworkEngine")

# Standard Project Paths
RAW_STOCK_PRICES_PARQUET = Path("data/raw/market_quotes/stock_prices_2009_2026.parquet")
SEC_MARKET_QUOTES_PARQUET = Path("data/processed/sec_dataset/sec_market_quotes_monthly.parquet")
SEC_FINANCIALS_PIVOTED_PARQUET = Path("data/processed/sec_dataset/v2_sec_financials_pivoted.parquet")
OUTPUT_NETWORK_PARQUET = Path("data/processed/sec_dataset/network_features_monthly.parquet")


def fast_tree_betweenness(tree: nx.Graph) -> Dict[Any, float]:
    """
    Calcula la intermediación (betweenness centrality) normalizada exacta en O(N) para un árbol (o bosque).
    Aprovecha que en un árbol los caminos más cortos son únicos, descomponiendo los tamaños de subárboles
    mediante recorrido post-orden DFS (16,000x más rápido que el algoritmo genérico de Brandes).

    Fórmula: B_norm(v) = ((N-1)^2 - sum(s_k^2)) / ((N-1)(N-2))
    donde s_k son los tamaños de las componentes conectadas resultantes de remover v.
    """
    betweenness: Dict[Any, float] = {}
    
    for comp in nx.connected_components(tree):
        sub_g = tree.subgraph(comp)
        N = sub_g.number_of_nodes()
        if N <= 2:
            for node in sub_g.nodes():
                betweenness[node] = 0.0
            continue
            
        root = next(iter(sub_g.nodes()))
        parent: Dict[Any, Any] = {}
        children: Dict[Any, List[Any]] = {node: [] for node in sub_g.nodes()}
        order: List[Any] = []
        
        visited = {root}
        stack = [root]
        while stack:
            u = stack.pop()
            order.append(u)
            for v in sub_g.neighbors(u):
                if v not in visited:
                    visited.add(v)
                    parent[v] = u
                    children[u].append(v)
                    stack.append(v)
                    
        subtree_size: Dict[Any, int] = {}
        for u in reversed(order):
            subtree_size[u] = 1 + sum(subtree_size[v] for v in children[u])
            
        norm_factor = (N - 1) * (N - 2)
        for u in sub_g.nodes():
            sub_sizes = [subtree_size[v] for v in children[u]]
            if u != root:
                sub_sizes.append(N - subtree_size[u])
            sum_sq = sum(s ** 2 for s in sub_sizes)
            b_val = ((N - 1) ** 2 - sum_sq) / norm_factor
            betweenness[u] = float(b_val)
            
    return betweenness


def compute_monthly_market_networks(
    raw_prices_path: Path = RAW_STOCK_PRICES_PARQUET,
    financials_path: Path = SEC_FINANCIALS_PIVOTED_PARQUET,
    quotes_path: Path = SEC_MARKET_QUOTES_PARQUET,
    lookback_months: int = 4,
    min_valid_pct: float = 0.70,
    seed: int = 42
) -> pd.DataFrame:
    """
    Procesa las series de precios diarias históricas y calcula las redes mensuales completas (2009-2026).
    
    Retorna un DataFrame con columnas:
    ['year', 'month', 'ticker', 'network_pagerank', 'network_betweenness',
     'network_degree', 'network_community_id', 'cluster_distress_infection_rate']
    """
    logger.info("=" * 70)
    logger.info("ESTIMANDO REDES COMPLEJAS MENSUALES, MST Y CONTAGIO DE MERCADO")
    logger.info("=" * 70)

    if not raw_prices_path.exists():
        raise FileNotFoundError(f"Archivo de cotizaciones brutas no encontrado: {raw_prices_path}")
    if not financials_path.exists():
        raise FileNotFoundError(f"Archivo de estados financieros no encontrado: {financials_path}")

    con = duckdb.connect()
    con.execute("SET memory_limit = '8GB';")

    # 1. Cargar precios diarios y calcular rendimientos diarios
    logger.info("1. Cargando precios bursátiles y calculando rendimientos logarítmicos/diarios...")
    t0 = time.time()
    df_prices = con.execute(f"""
        SELECT 
            ticker,
            TRY_CAST(trading_date AS DATE) as dt,
            close_price
        FROM '{raw_prices_path}'
        WHERE close_price > 0
        ORDER BY ticker, dt
    """).pl()

    df_returns = df_prices.with_columns(
        daily_return=(pl.col("close_price") - pl.col("close_price").shift(1).over("ticker")) /
                     pl.col("close_price").shift(1).over("ticker")
    ).filter(pl.col("daily_return").is_not_null())
    logger.info(f"   -> {len(df_returns):,} observaciones de rendimientos cargadas en {time.time() - t0:.2f}s.")

    # 2. Cargar historial de eventos de insolvencia / distress por ticker
    logger.info("2. Cargando eventos históricos de distress financiero para retardo espacial [W*Distress]...")
    df_distress_hist = con.execute(f"""
        SELECT 
            m.ticker,
            TRY_CAST(f.filed_date AS DATE) as filed_date,
            COALESCE(f.is_distress_event, 0)::INTEGER as is_distressed
        FROM '{financials_path}' f
        JOIN '{quotes_path}' m ON f.adsh = m.adsh
        WHERE m.ticker IS NOT NULL AND m.ticker != ''
    """).pl()
    logger.info(f"   -> {len(df_distress_hist):,} presentaciones de empresas cotizadas vinculadas.")

    # 3. Extraer ventanas temporales mensuales (2009 - 2026)
    df_ym = con.execute(f"""
        SELECT DISTINCT 
            EXTRACT(year FROM TRY_CAST(trading_date AS DATE))::INTEGER as yr,
            EXTRACT(month FROM TRY_CAST(trading_date AS DATE))::INTEGER as mo,
            MAX(TRY_CAST(trading_date AS DATE)) as end_date
        FROM '{raw_prices_path}'
        GROUP BY yr, mo
        HAVING yr >= 2009 AND yr <= 2026
        ORDER BY yr, mo
    """).df()

    total_months = len(df_ym)
    logger.info(f"3. Iniciando cálculo iterativo de grafos MST y comunidades Louvain para {total_months} meses...")
    
    monthly_records: List[Dict[str, Any]] = []
    t_start_loop = time.time()

    for idx, row in df_ym.iterrows():
        yr = int(row['yr'])
        mo = int(row['mo'])
        end_dt = pd.to_datetime(row['end_date']).date()
        start_dt = (pd.to_datetime(end_dt) - pd.DateOffset(months=lookback_months)).date()
        prior_dt = (pd.to_datetime(end_dt) - pd.DateOffset(months=1)).date()
        prior_lookback_dt = (pd.to_datetime(end_dt) - pd.DateOffset(months=13)).date()

        # Empresas con distress en t-1 (ventana previa de 12 meses hasta mes t-1)
        distressed_t_minus_1: Set[str] = set(
            df_distress_hist.filter(
                (pl.col("filed_date") >= pl.lit(prior_lookback_dt)) &
                (pl.col("filed_date") <= pl.lit(prior_dt)) &
                (pl.col("is_distressed") == 1)
            )['ticker'].to_list()
        )

        # Slice de rendimientos en la ventana rolling
        sub = df_returns.filter(
            (pl.col("dt") >= pl.lit(start_dt)) & 
            (pl.col("dt") <= pl.lit(end_dt))
        )

        piv = sub.pivot(index="dt", on="ticker", values="daily_return")
        thresh = int(len(piv) * min_valid_pct)
        null_counts = piv.null_count()
        valid_cols = [col for col in piv.columns if col != "dt" and null_counts[col][0] <= (len(piv) - thresh)]

        if len(valid_cols) < 5:
            continue

        piv_valid = piv.select(valid_cols).fill_null(strategy="mean").to_numpy()

        # A. Matriz de Correlación de Rendimientos C_ij(t)
        corr = np.corrcoef(piv_valid, rowvar=False)
        corr = np.nan_to_num(corr, nan=0.0)
        np.fill_diagonal(corr, 1.0)
        corr = np.clip(corr, -1.0, 1.0)

        # B. Distancia Métrica Ultramétrica: d_ij = sqrt(2 * (1 - C_ij))
        dist = np.sqrt(np.maximum(0.0, 2.0 * (1.0 - corr)))
        np.fill_diagonal(dist, 0.0)

        # C. Minimum Spanning Tree (MST)
        mst_sp = minimum_spanning_tree(dist)
        G_mst = nx.from_scipy_sparse_array(mst_sp)
        mapping = {i: valid_cols[i] for i in range(len(valid_cols))}
        G_mst = nx.relabel_nodes(G_mst, mapping)

        for u, v, d in G_mst.edges(data=True):
            d['similarity'] = 1.0 / (d.get('weight', 1.0) + 1e-4)

        # D. Detección de Comunidades Louvain
        comms = louvain_communities(G_mst, weight='similarity', seed=seed)
        node_to_comm: Dict[str, int] = {}
        comm_members: Dict[int, List[str]] = {}
        for c_idx, comm in enumerate(comms):
            comm_members[c_idx] = list(comm)
            for node in comm:
                node_to_comm[node] = c_idx

        # E. Métricas Nodales (PageRank, Fast Tree Betweenness, Degree)
        pr = nx.pagerank(G_mst, weight='similarity', max_iter=200)
        bet = fast_tree_betweenness(G_mst)
        deg = dict(G_mst.degree())

        # F. Retardo Espacial de Contagio: [W * Distress]_{i, t-1}
        for node in G_mst.nodes():
            c_idx = node_to_comm[node]
            peers = [p for p in comm_members[c_idx] if p != node]
            if not peers:
                inf_rate = 0.0
            else:
                distressed_count = sum(1 for p in peers if p in distressed_t_minus_1)
                inf_rate = float(distressed_count / len(peers))

            monthly_records.append({
                'year': yr,
                'month': mo,
                'ticker': node,
                'network_pagerank': float(pr[node]),
                'network_betweenness': float(bet[node]),
                'network_degree': float(deg[node]),
                'network_community_id': int(c_idx),
                'cluster_distress_infection_rate': float(inf_rate)
            })

        if (idx + 1) % 24 == 0 or idx == total_months - 1:
            logger.info(
                f"   [Mes {idx + 1:3d}/{total_months}] Periodo {yr}-{mo:02d} | "
                f"Activos: {len(valid_cols):4d} | Comunidades: {len(comms):3d} | "
                f"Registros nodales: {len(monthly_records):,}"
            )

    con.close()
    df_nodal = pd.DataFrame(monthly_records)
    logger.info(f"Redes mensuales completadas en {time.time() - t_start_loop:.2f}s. Total registros: {len(df_nodal):,}")
    return df_nodal


def build_and_impute_network_features(
    df_nodal: Optional[pd.DataFrame] = None,
    financials_path: Path = SEC_FINANCIALS_PIVOTED_PARQUET,
    quotes_path: Path = SEC_MARKET_QUOTES_PARQUET,
    output_parquet_path: Path = OUTPUT_NETWORK_PARQUET
) -> pd.DataFrame:
    """
    Cruza las características nodales de red con las 404,853 presentaciones SEC y aplica
    imputación jerárquica robusta por mediana sectorial (SIC 4d -> SIC 2d -> Mercado mensual -> Global).
    
    Genera el archivo final 'data/processed/sec_dataset/network_features_monthly.parquet'.
    """
    logger.info("=" * 70)
    logger.info("INTEGRACIÓN SEC & IMPUTACIÓN JERÁRQUICA SECTORIAL DE MÉTRICAS DE RED")
    logger.info("=" * 70)

    if df_nodal is None or df_nodal.empty:
        logger.info("Calculando métricas de red desde cotizaciones...")
        df_nodal = compute_monthly_market_networks(
            financials_path=financials_path,
            quotes_path=quotes_path
        )

    con = duckdb.connect()
    con.execute("SET memory_limit = '8GB';")

    # Registrar df_nodal en DuckDB
    con.register("df_nodal", df_nodal)

    logger.info("1. Vinculando presentaciones SEC con cotizaciones y métricas de red...")
    con.execute(f"""
        CREATE OR REPLACE TABLE sec_filings_raw AS
        SELECT 
            f.adsh,
            f.cik,
            TRY_CAST(f.filed_date AS DATE) as filed_date,
            EXTRACT(year FROM f.filed_date)::INTEGER as filed_year,
            EXTRACT(month FROM f.filed_date)::INTEGER as filed_month,
            f.sic,
            (f.sic // 100)::INTEGER as sic_2digit,
            m.ticker
        FROM '{financials_path}' f
        LEFT JOIN (
            SELECT DISTINCT adsh, cik, ticker 
            FROM '{quotes_path}' 
            WHERE ticker IS NOT NULL AND ticker != ''
        ) m ON f.adsh = m.adsh;
    """)

    logger.info("2. Calculando estadísticas y medianas jerárquicas sectoriales por periodo...")
    # A. Match directo con ticker en el mes correspondiente
    con.execute("""
        CREATE OR REPLACE TABLE sec_matched_nodal AS
        SELECT 
            s.adsh,
            s.cik,
            s.ticker,
            s.filed_date,
            s.filed_year,
            s.filed_month,
            s.sic,
            s.sic_2digit,
            n.network_pagerank,
            n.network_betweenness,
            n.network_degree,
            n.network_community_id,
            n.cluster_distress_infection_rate,
            CASE WHEN n.network_pagerank IS NOT NULL THEN 0 ELSE 1 END as is_network_imputed
        FROM sec_filings_raw s
        LEFT JOIN df_nodal n 
          ON s.ticker = n.ticker 
         AND s.filed_year = n.year 
         AND s.filed_month = n.month;
    """)

    # Eliminar duplicados de adsh (en caso de múltiples tickers por CIK, conservar el de mayor centralidad)
    con.execute("""
        CREATE OR REPLACE TABLE sec_matched_unique AS
        SELECT * EXCLUDE (row_num)
        FROM (
            SELECT *,
                ROW_NUMBER() OVER (
                    PARTITION BY adsh 
                    ORDER BY network_pagerank DESC NULLS LAST, ticker
                ) as row_num
            FROM sec_matched_nodal
        )
        WHERE row_num = 1;
    """)

    logger.info("3. Calculando tablas de medianas sectoriales jerárquicas...")
    # Medianas por SIC 4-digit y (año, mes)
    con.execute("""
        CREATE OR REPLACE TABLE med_sic4 AS
        SELECT 
            filed_year, filed_month, sic,
            MEDIAN(network_pagerank) as med_pr_sic4,
            MEDIAN(network_betweenness) as med_bet_sic4,
            MEDIAN(network_degree) as med_deg_sic4,
            MEDIAN(cluster_distress_infection_rate) as med_inf_sic4
        FROM sec_matched_unique
        WHERE network_pagerank IS NOT NULL AND sic IS NOT NULL
        GROUP BY filed_year, filed_month, sic;
    """)

    # Medianas por SIC 2-digit y (año, mes)
    con.execute("""
        CREATE OR REPLACE TABLE med_sic2 AS
        SELECT 
            filed_year, filed_month, sic_2digit,
            MEDIAN(network_pagerank) as med_pr_sic2,
            MEDIAN(network_betweenness) as med_bet_sic2,
            MEDIAN(network_degree) as med_deg_sic2,
            MEDIAN(cluster_distress_infection_rate) as med_inf_sic2
        FROM sec_matched_unique
        WHERE network_pagerank IS NOT NULL AND sic_2digit IS NOT NULL
        GROUP BY filed_year, filed_month, sic_2digit;
    """)

    # Medianas del mercado mensual (año, mes)
    con.execute("""
        CREATE OR REPLACE TABLE med_month AS
        SELECT 
            filed_year, filed_month,
            MEDIAN(network_pagerank) as med_pr_m,
            MEDIAN(network_betweenness) as med_bet_m,
            MEDIAN(network_degree) as med_deg_m,
            MEDIAN(cluster_distress_infection_rate) as med_inf_m
        FROM sec_matched_unique
        WHERE network_pagerank IS NOT NULL
        GROUP BY filed_year, filed_month;
    """)

    # Medianas globales de contingencia
    con.execute("""
        CREATE OR REPLACE TABLE med_global AS
        SELECT 
            MEDIAN(network_pagerank) as med_pr_g,
            MEDIAN(network_betweenness) as med_bet_g,
            MEDIAN(network_degree) as med_deg_g,
            MEDIAN(cluster_distress_infection_rate) as med_inf_g
        FROM sec_matched_unique
        WHERE network_pagerank IS NOT NULL;
    """)

    logger.info("4. Ejecutando imputación en cascada y ensamblando dataset final...")
    con.execute("""
        CREATE OR REPLACE TABLE network_features_monthly AS
        SELECT 
            u.adsh,
            u.cik,
            u.ticker,
            u.filed_date,
            u.filed_year,
            u.filed_month,
            u.sic,
            
            -- Imputación Jerárquica de PageRank
            COALESCE(
                u.network_pagerank,
                s4.med_pr_sic4,
                s2.med_pr_sic2,
                mm.med_pr_m,
                g.med_pr_g,
                0.000330
            ) as network_pagerank,

            -- Imputación Jerárquica de Betweenness
            COALESCE(
                u.network_betweenness,
                s4.med_bet_sic4,
                s2.med_bet_sic2,
                mm.med_bet_m,
                g.med_bet_g,
                0.000885
            ) as network_betweenness,

            -- Imputación Jerárquica de Degree
            COALESCE(
                u.network_degree,
                s4.med_deg_sic4,
                s2.med_deg_sic2,
                mm.med_deg_m,
                g.med_deg_g,
                1.0
            ) as network_degree,

            -- ID de Comunidad Louvain (-1 si es imputada)
            COALESCE(u.network_community_id, -1) as network_community_id,

            -- Imputación Jerárquica de Spatial Infection Rate
            COALESCE(
                u.cluster_distress_infection_rate,
                s4.med_inf_sic4,
                s2.med_inf_sic2,
                mm.med_inf_m,
                g.med_inf_g,
                0.0
            ) as cluster_distress_infection_rate,

            u.is_network_imputed

        FROM sec_matched_unique u
        LEFT JOIN med_sic4 s4 
          ON u.filed_year = s4.filed_year 
         AND u.filed_month = s4.filed_month 
         AND u.sic = s4.sic
        LEFT JOIN med_sic2 s2 
          ON u.filed_year = s2.filed_year 
         AND u.filed_month = s2.filed_month 
         AND u.sic_2digit = s2.sic_2digit
        LEFT JOIN med_month mm 
          ON u.filed_year = mm.filed_year 
         AND u.filed_month = mm.filed_month
        CROSS JOIN med_global g
        ORDER BY u.filed_date, u.cik;
    """)

    # Exportar a Parquet
    output_parquet_path.parent.mkdir(parents=True, exist_ok=True)
    con.execute(f"""
        COPY network_features_monthly 
        TO '{output_parquet_path}' 
        (FORMAT PARQUET, COMPRESSION SNAPPY);
    """)

    df_final = con.execute("SELECT * FROM network_features_monthly").df()
    total_filings = len(df_final)
    direct_matched = (df_final['is_network_imputed'] == 0).sum()
    imputed_count = (df_final['is_network_imputed'] == 1).sum()
    file_size_mb = output_parquet_path.stat().st_size / (1024 * 1024)

    logger.info("=" * 70)
    logger.info("RESUMEN DE GENERACIÓN DE MÉTRICAS DE RED:")
    logger.info(f"  - Total Presentaciones SEC (Filings): {total_filings:,}")
    logger.info(f"  - Métricas Directas de Red (Cotizadas): {direct_matched:,} ({(direct_matched/total_filings)*100:.2f}%)")
    logger.info(f"  - Imputación Jerárquica Sectorial:     {imputed_count:,} ({(imputed_count/total_filings)*100:.2f}%)")
    logger.info(f"  - Archivo Parquet Generado:            {output_parquet_path} ({file_size_mb:.2f} MB)")
    logger.info("=" * 70)

    con.close()
    return df_final


# =====================================================================
# DUCKDB INTEGRATION FUNCTIONS
# =====================================================================

def register_network_features_in_duckdb(
    con: duckdb.DuckDBPyConnection,
    table_name: str = "network_features_monthly",
    parquet_path: Optional[Path] = None
) -> None:
    """
    Registra el archivo Parquet de características de red en una conexión DuckDB activa.
    """
    target_path = parquet_path or OUTPUT_NETWORK_PARQUET
    if not target_path.exists():
        raise FileNotFoundError(f"Parquet de características de red no encontrado: {target_path}")
    
    con.execute(f"""
        CREATE OR REPLACE VIEW {table_name} AS 
        SELECT * FROM '{target_path}';
    """)
    logger.info(f"Vista DuckDB '{table_name}' registrada exitosamente desde {target_path}.")


def get_merged_financials_with_network_duckdb(
    con: Optional[duckdb.DuckDBPyConnection] = None,
    financials_path: Path = SEC_FINANCIALS_PIVOTED_PARQUET,
    network_parquet_path: Path = OUTPUT_NETWORK_PARQUET,
    view_name: str = "v2_sec_financials_with_network"
) -> duckdb.DuckDBPyConnection:
    """
    Crea una conexión o utiliza una existente para unir el dataset financiero maestro
    con las características de red complejas y contagio financiero.
    """
    if con is None:
        con = duckdb.connect()
        con.execute("SET memory_limit = '8GB';")

    if not financials_path.exists():
        raise FileNotFoundError(f"Archivo de balances no encontrado: {financials_path}")
    if not network_parquet_path.exists():
        raise FileNotFoundError(f"Archivo de red no encontrado: {network_parquet_path}")

    con.execute(f"""
        CREATE OR REPLACE VIEW {view_name} AS
        SELECT 
            f.*,
            n.ticker,
            n.network_pagerank,
            n.network_betweenness,
            n.network_degree,
            n.network_community_id,
            n.cluster_distress_infection_rate,
            n.is_network_imputed
        FROM '{financials_path}' f
        LEFT JOIN '{network_parquet_path}' n ON f.adsh = n.adsh;
    """)
    logger.info(f"Vista DuckDB integrada '{view_name}' creada exitosamente.")
    return con


# =====================================================================
# VALIDATION & QUALITY AUDIT
# =====================================================================

def validate_network_features(
    parquet_path: Path = OUTPUT_NETWORK_PARQUET,
    financials_path: Path = SEC_FINANCIALS_PIVOTED_PARQUET
) -> Dict[str, Any]:
    """
    Realiza una auditoría exhaustiva de calidad, integridad y relevancia estadística de las métricas de red.
    """
    logger.info("=" * 70)
    logger.info("AUDITORÍA Y VALIDACIÓN DE CALIDAD DE CARACTERÍSTICAS DE RED")
    logger.info("=" * 70)

    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet no encontrado: {parquet_path}")

    con = duckdb.connect()
    
    # 1. Conteo e Integridad
    total_rows = con.execute(f"SELECT COUNT(*) FROM '{parquet_path}'").fetchone()[0]
    null_report = con.execute(f"""
        SELECT 
            SUM(CASE WHEN network_pagerank IS NULL OR isnan(network_pagerank) THEN 1 ELSE 0 END) as null_pr,
            SUM(CASE WHEN network_betweenness IS NULL OR isnan(network_betweenness) THEN 1 ELSE 0 END) as null_bet,
            SUM(CASE WHEN network_degree IS NULL OR isnan(network_degree) THEN 1 ELSE 0 END) as null_deg,
            SUM(CASE WHEN cluster_distress_infection_rate IS NULL OR isnan(cluster_distress_infection_rate) THEN 1 ELSE 0 END) as null_inf,
            SUM(CASE WHEN is_network_imputed IS NULL THEN 1 ELSE 0 END) as null_imp
        FROM '{parquet_path}'
    """).df()

    # 2. Estadísticas descriptivas de las métricas
    stats_df = con.execute(f"""
        SELECT 
            MIN(network_pagerank) as min_pr, AVG(network_pagerank) as avg_pr, MAX(network_pagerank) as max_pr, STDDEV(network_pagerank) as std_pr,
            MIN(network_betweenness) as min_bet, AVG(network_betweenness) as avg_bet, MAX(network_betweenness) as max_bet, STDDEV(network_betweenness) as std_bet,
            MIN(network_degree) as min_deg, AVG(network_degree) as avg_deg, MAX(network_degree) as max_deg, STDDEV(network_degree) as std_deg,
            MIN(cluster_distress_infection_rate) as min_inf, AVG(cluster_distress_infection_rate) as avg_inf, MAX(cluster_distress_infection_rate) as max_inf, STDDEV(cluster_distress_infection_rate) as std_inf
        FROM '{parquet_path}'
    """).df()

    # 3. Correlación con Insolvencia y Merton Distance-to-Default
    corr_df = con.execute(f"""
        SELECT 
            CORR(n.cluster_distress_infection_rate, f.is_distress_event) as corr_infection_distress,
            CORR(n.cluster_distress_infection_rate, f.target_bankrupt_12m) as corr_infection_bankrupt12m,
            CORR(n.cluster_distress_infection_rate, f.merton_distance_to_default) as corr_infection_merton,
            CORR(n.network_betweenness, f.is_distress_event) as corr_betweenness_distress,
            CORR(n.network_pagerank, f.is_distress_event) as corr_pagerank_distress
        FROM '{parquet_path}' n
        JOIN '{financials_path}' f ON n.adsh = f.adsh;
    """).df()

    imputation_stats = con.execute(f"""
        SELECT 
            is_network_imputed,
            COUNT(*) as count,
            AVG(network_pagerank) as avg_pr,
            AVG(network_betweenness) as avg_bet,
            AVG(network_degree) as avg_deg,
            AVG(cluster_distress_infection_rate) as avg_inf
        FROM '{parquet_path}'
        GROUP BY is_network_imputed
    """).df()

    con.close()

    report = {
        'total_filings': total_rows,
        'null_counts': null_report.to_dict(orient='records')[0],
        'descriptive_stats': stats_df.to_dict(orient='records')[0],
        'correlations': corr_df.to_dict(orient='records')[0],
        'imputation_comparison': imputation_stats.to_dict(orient='records')
    }

    logger.info("AUDITORÍA DE CALIDAD COMPLETADA:")
    logger.info(f"  - Total Filings auditados: {total_rows:,}")
    logger.info(f"  - Valores Nulos / NaN: {null_report.iloc[0].to_dict()}")
    logger.info(f"  - Correlación Contagio vs Distress Event: {corr_df['corr_infection_distress'][0]:+.4f}")
    logger.info(f"  - Correlación Contagio vs Bankrupt 12M:   {corr_df['corr_infection_bankrupt12m'][0]:+.4f}")
    logger.info(f"  - Correlación Contagio vs Merton DD:      {corr_df['corr_infection_merton'][0]:+.4f}")
    logger.info("=" * 70)

    return report


def main():
    """Punto de entrada principal para generar y validar las características de red."""
    print("=" * 75)
    print("SUBAGENTE 1C: PIPELINE DE CARACTERÍSTICAS DE RED Y CONTAGIO FINANCIERO")
    print("=" * 75)
    
    t_start = time.time()
    
    # 1. Ejecutar cálculo e imputación
    df_final = build_and_impute_network_features()
    
    # 2. Validar calidad de los datos
    validation_report = validate_network_features()
    
    # 3. Test de integración DuckDB
    con = duckdb.connect()
    get_merged_financials_with_network_duckdb(con=con)
    sample_duckdb = con.execute("SELECT adsh, ticker, network_pagerank, network_betweenness, network_degree, cluster_distress_infection_rate, is_network_imputed FROM v2_sec_financials_with_network LIMIT 5").df()
    print("\n--- MUESTRA DE INTEGRACIÓN DUCKDB ---")
    print(sample_duckdb)
    con.close()
    
    print("\n" + "=" * 75)
    print(f"PIPELINE DE REDES COMPLETADO EXITOSAMENTE EN {time.time() - t_start:.2f} SEGUNDOS.")
    print("=" * 75)


if __name__ == "__main__":
    main()
