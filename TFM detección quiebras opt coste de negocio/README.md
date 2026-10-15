# Estructura del Repositorio

```text
Entrega-TFM/
├── TFM Martin Herrera Roncero.pdf
├── README.md
├── requirements.txt
├── verificar_entorno.py
├── run_api.bat
├── run_api.sh
│
├── notebooks/
│   ├── 01_EDA_FE.ipynb
│   ├── 02_Feature_selection.ipynb
│   ├── 03_Model_Training_GPU_and_Benchmark.ipynb
│   └── 04_Explainability_SHAP_and_Stress_Testing.ipynb
│
├── api/
│   ├── main.py
│   ├── schemas.py
│   ├── service.py
│   ├── run_api.py
│   ├── live_data_service.py
│   ├── test_api.py
│   ├── README_API.md
│   └── static/
│       └── dashboard.html
│
├── models/
│   ├── best_xgboost_model.pkl
│   ├── voting_classifier.pkl
│   ├── lightgbm_model.pkl
│   ├── lightgbm_best_model.txt
│   ├── random_forest_model.pkl
│   ├── catboost_model.pkl
│   ├── catboost_best_model.cbm
│   ├── logistic_regression_model.pkl
│   ├── scorecard_woe_best.pkl
│   ├── financial_mlp_best.pt
│   ├── preprocessor_pipeline.joblib
│   ├── distribution_stats.joblib
│   ├── sample_companies_bank.joblib
│   └── README_MODELS.md
│
├── data/
│   ├── processed/
│   │   ├── sec_dataset/
│   │   │   ├── v2_sec_financials_pivoted_clean.parquet
│   │   │   ├── v2.1_master_financials_dataset.parquet
│   │   │   ├── v2_sec_financials_pivoted.parquet
│   │   │   ├── network_features_monthly.parquet
│   │   │   ├── sec_bankruptcy_events.parquet
│   │   │   ├── sec_insider_monthly.parquet
│   │   │   ├── sec_market_quotes_monthly.parquet
│   │   │   └── *.csv / *.json
│   │   └── live_cache/
│   │       └── live_data_cache.sqlite
│   └── raw/
│       └── sec_metadata/
│           └── cik_ticker_map.parquet
│
├── src/
│   ├── config/
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── evaluation/
│   ├── explainability/
│   └── api/
│
├── reports/
│   └── figures/
│
└── config/
    └── production.json
```
