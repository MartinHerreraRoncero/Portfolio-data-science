"""
Modulo de resolucion canonica de rutas del proyecto TFM.
Garantiza anclaje determinista relativo a PROJECT_ROOT.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed" / "sec_dataset"
RAW_DATA_DIR = DATA_DIR / "raw"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
ANEXOS_DIR = PROJECT_ROOT / "anexos"

MASTER_DATASET_PARQUET = PROCESSED_DATA_DIR / "v2.1_master_financials_dataset.parquet"
PIVOTED_CLEAN_PARQUET = PROCESSED_DATA_DIR / "v2_sec_financials_pivoted_clean.parquet"
PIVOTED_RAW_PARQUET = PROCESSED_DATA_DIR / "v2_sec_financials_pivoted.parquet"
NETWORK_PARQUET = PROCESSED_DATA_DIR / "network_features_monthly.parquet"
MARKET_QUOTES_PARQUET = PROCESSED_DATA_DIR / "sec_market_quotes_monthly.parquet"
BANKRUPTCY_EVENTS_PARQUET = PROCESSED_DATA_DIR / "sec_bankruptcy_events.parquet"

CIK_TICKER_MAP_PARQUET = RAW_DATA_DIR / "sec_metadata" / "cik_ticker_map.parquet"

BENCHMARK_METRICS_CSV = PROCESSED_DATA_DIR / "benchmark_models_full_metrics.csv"
BENCHMARK_METRICS_JSON = PROCESSED_DATA_DIR / "benchmark_models_full_metrics.json"
BENCHMARK_RESULTS_JSON = PROCESSED_DATA_DIR / "benchmark_models_results.json"
STATISTICAL_EVALUATION_JSON = PROCESSED_DATA_DIR / "statistical_evaluation_results.json"
DECILES_GAINS_LIFT_CSV = PROCESSED_DATA_DIR / "deciles_gains_lift.csv"
DUAL_TARGET_METRICS_CSV = PROCESSED_DATA_DIR / "dual_target_evaluation_metrics.csv"
DATA_DRIFT_REPORT_CSV = PROCESSED_DATA_DIR / "data_drift_psi_ks_report.csv"
