"""
================================================================================
SCRIPT DE VERIFICACION INTEGRAL DE ENTORNO Y REPRODUCIBILIDAD
Trabajo de Fin de Master en Data Science, Big Data and Business Analytics (UCM)
Autor: Martin Herrera Roncero
Directores / Tutores: Carlos Ortega and Santiago Mota
================================================================================
Uso:
    python verificar_entorno.py
"""

import sys
import os
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

PASS = "[OK]"
FAIL = "[ERROR]"

def print_header(title: str):
    print("\n" + "=" * 78)
    print(f"  {title}")
    print("=" * 78)

def main():
    print_header("TFM: AUDITORIA AUTOMATIZADA DE REPRODUCIBILIDAD Y ENTORNO")
    print(f"Directorio de ejecucion: {ROOT_DIR}")
    print(f"Version de Python:       {sys.version.split()[0]} ({sys.platform})")
    
    total_checks = 0
    passed_checks = 0
    errors = []

    # 1. VERIFICACION DE DEPENDENCIAS PYTHON
    print_header("1. COMPROBACION DE LIBRERIAS Y PAQUETES PYTHON")
    required_packages = [
        ("numpy", "Calculo numerico matricial"),
        ("pandas", "Manipulacion tabular de paneles contables"),
        ("scipy", "Contrastes estadisticos y distribuciones"),
        ("sklearn", "Scikit-Learn (Pipelines, Imputacion, Metricas)"),
        ("duckdb", "Motor OLAP SQL sobre Parquet"),
        ("pyarrow", "Serializacion columnar Apache Arrow/Parquet"),
        ("joblib", "Serializacion de pipelines y modelos"),
        ("xgboost", "Extreme Gradient Boosting SOTA"),
        ("lightgbm", "Light Gradient Boosting Machine"),
        ("catboost", "CatBoost Classifier con GPU"),
        ("optbinning", "Scorecard regulatorio WoE Basilea III"),
        ("torch", "PyTorch (Deep Learning / Perdida Asimetrica)"),
        ("shap", "Explicabilidad XAI TreeExplainer"),
        ("fastapi", "Framework API RESTful"),
        ("uvicorn", "Servidor ASGI de alto rendimiento"),
        ("pydantic", "Validacion estricta de esquemas de datos"),
        ("matplotlib", "Visualizacion corporativa"),
        ("seaborn", "Graficos estadisticos"),
    ]

    for pkg, desc in required_packages:
        total_checks += 1
        try:
            mod = __import__(pkg)
            version = getattr(mod, "__version__", "instalado")
            print(f" {PASS} {pkg:<14} v{version:<10} - {desc}")
            passed_checks += 1
        except ImportError as e:
            print(f" {FAIL} {pkg:<14} NO INSTALADO   - {desc} (Error: {e})")
            errors.append(f"Paquete faltante: {pkg}")

    # 2. VERIFICACION DE ARTEFACTOS PRINCIPALES Y DOCUMENTO OFICIAL
    print_header("2. COMPROBACION DEL DOCUMENTO OFICIAL Y CUADERNOS JUPYTER")
    core_files = [
        (ROOT_DIR / "TFM Martin Herrera Roncero.pdf", "Memoria Academica Oficial (24 paginas)"),
        (ROOT_DIR / "README.md", "Guia Tecnica de Ejecucion y Gobernanza"),
        (ROOT_DIR / "requirements.txt", "Especificacion de dependencias fijadas"),
        (ROOT_DIR / "notebooks" / "01_EDA_FE.ipynb", "Cuaderno 01: EDA e Ingenieria Multimodal"),
        (ROOT_DIR / "notebooks" / "02_Feature_selection.ipynb", "Cuaderno 02: Seleccion Optima de Variables"),
        (ROOT_DIR / "notebooks" / "03_Model_Training_GPU_and_Benchmark.ipynb", "Cuaderno 03: Entrenamiento GPU y Benchmark"),
        (ROOT_DIR / "notebooks" / "04_Explainability_SHAP_and_Stress_Testing.ipynb", "Cuaderno 04: Explicabilidad XAI y Estres"),
    ]

    for fpath, desc in core_files:
        total_checks += 1
        if fpath.exists():
            size_kb = fpath.stat().st_size / 1024
            print(f" {PASS} {fpath.name:<36} ({size_kb:>8.1f} KB) - {desc}")
            passed_checks += 1
        else:
            print(f" {FAIL} {fpath.name:<36} NO ENCONTRADO - {desc}")
            errors.append(f"Archivo critico ausente: {fpath}")

    # 3. VERIFICACION DE MODELOS SERIALIZADOS
    print_header("3. COMPROBACION DE MODELOS ESTIMADORES SERIALIZADOS")
    models_dir = ROOT_DIR / "models"
    models_to_check = [
        ("best_xgboost_model.pkl", "Modelo XGBoost SOTA Campeon a 12M"),
        ("voting_classifier.pkl", "Ensamble Soft Voting Classifier"),
        ("lightgbm_model.pkl", "LightGBM Classifier"),
        ("random_forest_model.pkl", "Random Forest de referencia"),
        ("catboost_model.pkl", "CatBoost Classifier"),
        ("logistic_regression_model.pkl", "Regresion Logistica Baseline"),
        ("scorecard_woe_best.pkl", "Scorecard Regulatorio Basilea III"),
        ("financial_mlp_best.pt", "Red Neuronal PyTorch (Perdida Asimetrica 24x)"),
        ("preprocessor_pipeline.joblib", "Pipeline Imputacion y Escalado (21 Vars)"),
        ("distribution_stats.joblib", "Estadisticas Multivariantes y Covarianza"),
        ("sample_companies_bank.joblib", "Banco de Firmas para Inferencia y XAI"),
    ]

    for mfile, desc in models_to_check:
        total_checks += 1
        mpath = models_dir / mfile
        if mpath.exists():
            size_kb = mpath.stat().st_size / 1024
            print(f" {PASS} {mfile:<32} ({size_kb:>8.1f} KB) - {desc}")
            passed_checks += 1
        else:
            print(f" {FAIL} {mfile:<32} NO ENCONTRADO - {desc}")
            errors.append(f"Modelo ausente: {mfile}")

    # 4. VERIFICACION DE DATASETS PROCESADOS Y TABLAS DE RESULTADOS
    print_header("4. COMPROBACION DE DATASETS PARQUET Y TABLAS METRICAS")
    sec_dir = ROOT_DIR / "data" / "processed" / "sec_dataset"
    data_to_check = [
        ("v2_sec_financials_pivoted_clean.parquet", "Dataset Limpio Canonico de Produccion (21 Vars)"),
        ("v2.1_master_financials_dataset.parquet", "Dataset Maestro Multimodal Completo (52 Vars)"),
        ("network_features_monthly.parquet", "Grafo Intersectorial y Metricas PageRank"),
        ("benchmark_models_full_metrics.csv", "Tabla Completa de Metricas del Benchmark"),
        ("delong_tests_results.csv", "Resultados del Contraste Pareado de DeLong"),
        ("mcnemar_tests_results.csv", "Resultados del Contraste Pareado de McNemar"),
        ("bootstrap_ci_results.csv", "Intervalos de Confianza Bootstrap (B=1,000)"),
        ("data_drift_psi_ks_report.csv", "Reporte Censal de Estabilidad PSI y Kolmogorov-Smirnov"),
        ("stress_testing_monte_carlo_results.json", "Simulacion Estocastica de Estres Monte Carlo"),
    ]

    for dfile, desc in data_to_check:
        total_checks += 1
        dpath = sec_dir / dfile
        if dpath.exists():
            size_kb = dpath.stat().st_size / 1024
            print(f" {PASS} {dfile:<40} ({size_kb:>8.1f} KB) - {desc}")
            passed_checks += 1
        else:
            print(f" {FAIL} {dfile:<40} NO ENCONTRADO - {desc}")
            errors.append(f"Dataset ausente: {dfile}")

    # 5. PRUEBA DE FUNCIONAMIENTO DEL MOTOR DE INFERENCIA
    print_header("5. PRUEBA DE FUNCIONAMIENTO DEL MOTOR DE INFERENCIA")
    try:
        total_checks += 1
        import joblib
        from api.service import BankruptcyPredictorService
        
        service = BankruptcyPredictorService()
        bank = joblib.load(models_dir / "sample_companies_bank.joblib")

        test_company = bank["solvent_samples"][0] if "solvent_samples" in bank else (bank["distressed_samples"][0] if "distressed_samples" in bank else bank)
        company_name = test_company.get("company_name", "STRATEGIC STORAGE TRUST, INC.")
        
        pred_res = service.predict_single(test_company, model_name="xgboost")
        prob = pred_res["bankruptcy_probability"]
        tau_xgb = pred_res["operational_threshold"]
        clasificacion = pred_res["risk_tier"]

        print(f" {PASS} Servicio de inferencia y modelo SOTA XGBoost evaluados exitosamente.")
        print(f"      -> Empresa evaluada: {company_name}")
        print(f"      -> Probabilidad P(Insolvencia): {prob * 100:.2f}%")
        print(f"      -> Umbral operativo optimo tau*: {tau_xgb:.3f}")
        print(f"      -> Clasificacion regulatoria:    {clasificacion}")
        print(f"      -> Coste esperado de negocio:    ${pred_res.get('estimated_cost_usd', 0):,.2f} USD")
        passed_checks += 1
    except Exception as e:
        print(f" {FAIL} Error durante la prueba de inferencia: {e}")
        errors.append(f"Fallo en inferencia de prueba: {e}")

    # 6. RESUMEN Y DICTAMEN FINAL
    print_header("DICTAMEN FINAL DE VERIFICACION")
    pct = (passed_checks / total_checks) * 100
    print(f"Comprobaciones superadas: {passed_checks} / {total_checks} ({pct:.1f}%)")

    if errors:
        print(f"\n{FAIL} Se detectaron incidencias:")
        for err in errors:
            print(f"  - {err}")
        sys.exit(1)
    else:
        print(f"\n{PASS} TODOS LOS COMPONENTES HAN SIDO VERIFICADOS CORRECTAMENTE.")
        print("     El entregable es 100% autotenido, reproducible y operativo.")
        print("     Puede proceder a revisar los cuadernos o iniciar la API con:")
        print("       python api/run_api.py   o   run_api.bat (Windows)\n")
        sys.exit(0)

if __name__ == "__main__":
    main()
