"""
Script de Medicion de Latencia y Benchmarking para la API RESTful (FastAPI).
Proyecto TFM: Prediccion de Insolvencia en Empresas del S&P 500 y Mercado de EE.UU.
"""

import os
import sys
import time

# Asegurar path raiz del proyecto
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

def run_api_latency_benchmark():
    print("=== INICIANDO BENCHMARK DE LATENCIA DE LA API RESTFUL DE PRODUCCION ===")

    r = client.get("/health")
    print(f"Health Check: {r.json()}")

    r_info = client.get("/model_info")
    print(f"Model Info: {r_info.json()['model_name']} | Version: {r_info.json()['version']}")

    payload_26 = {
        "company_name": "Corp_Benchmark_Inc",
        "year": 2024,
        "ticker": "CBK",
        "tag_CashAndCashEquivalentsAtCarryingValue": 12000000.0,
        "tag_CommonStockSharesAuthorized": 100000000.0,
        "tag_NetIncomeLoss": 5400000.0,
        "tag_AssetsCurrent": 35000000.0,
        "tag_Revenues_combined": 85000000.0,
        "tag_InterestExpense": 1200000.0,
        "tag_RevenueFromContractWithCustomerExcludingAssessedTax": 85000000.0,
        "tag_CommonStockValue": 2500000.0,
        "tag_CommonStockParOrStatedValuePerShare": 0.01,
        "news_sentiment_range_6m": 0.28,
        "tag_StockholdersEquity": 45000000.0,
        "tag_Assets": 75000000.0,
        "tag_WorkingCapital": 18000000.0,
        "news_sentiment_max": 0.82,
        "nlp_sentiment_decayed_30d": 0.55,
        "macro_inflation": 2.8,
        "stock_price_close": 52.4,
        "merton_distance_to_default": 4.1,
        "tag_EntityCommonStockSharesOutstanding": 40000000.0,
        "market_news_sentiment_mean": 0.50,
        "macro_unemployment_rate": 3.8,
        "tag_RetainedEarningsAccumulatedDeficit": 28000000.0,
        "macro_real_gdp_growth_yoy": 2.2,
        "market_cap": 2096000000.0,
        "macro_interest_rate": 4.5,
        "pub_lag_days": 42.0
    }

    # 1. Warm-up
    for _ in range(5):
        client.post("/predict", json=payload_26)

    # 2. Single Request Inferences (50 peticiones consecutivas)
    latencies_single = []
    for _ in range(50):
        t0 = time.time()
        res = client.post("/predict", json=payload_26)
        lat = (time.time() - t0) * 1000.0
        latencies_single.append(lat)

    avg_single = sum(latencies_single) / len(latencies_single)
    p95_single = sorted(latencies_single)[int(len(latencies_single) * 0.95)]
    print(f"Latencia Media Inferencia Unitaria (/predict): {avg_single:.2f} ms (p95: {p95_single:.2f} ms)")

    # 3. Scorecard Request Inferences
    latencies_sc = []
    for _ in range(20):
        t0 = time.time()
        res = client.post("/scorecard", json=payload_26)
        lat = (time.time() - t0) * 1000.0
        latencies_sc.append(lat)
    avg_sc = sum(latencies_sc) / len(latencies_sc)
    print(f"Latencia Media Scorecard (/scorecard): {avg_sc:.2f} ms")

    # 4. Batch Request Inferences Vectorizadas (Lote 100 empresas)
    batch_payload = [dict(payload_26, company_name=f"Batch_Corp_{i}") for i in range(100)]
    t0 = time.time()
    res_batch = client.post("/predict_batch", json=batch_payload)
    batch_total_time = (time.time() - t0) * 1000.0
    per_item_lat = batch_total_time / 100.0

    print(f"Latencia Total Lote (Batch 100 empresas vectorizado): {batch_total_time:.2f} ms")
    print(f"Latencia Promedio por Elemento en Lote: {per_item_lat:.3f} ms / empresa")

    print("\n=== BENCHMARK COMPLETADO EXITOSAMENTE ===")
    return {
        'single_request_avg_ms': round(avg_single, 2),
        'single_request_p95_ms': round(p95_single, 2),
        'scorecard_avg_ms': round(avg_sc, 2),
        'batch_total_ms': round(batch_total_time, 2),
        'batch_per_item_ms': round(per_item_lat, 3)
    }

if __name__ == "__main__":
    run_api_latency_benchmark()