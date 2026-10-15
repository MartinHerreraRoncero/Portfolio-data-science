"""
Suite exhaustiva de pruebas unitarias y de integración para la API de Predicción de Quiebra Corporativa.
Valida endpoints, gobernanza regulatoria, inferencia con los 6 modelos, explicabilidad SHAP,
Scorecard de Basilea III y compatibilidad retroactiva con esquemas contables legacy.
"""

import sys
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

# Configuración de rutas para ejecución autónoma
CURRENT_FILE = Path(__file__).resolve()
API_DIR = CURRENT_FILE.parent
ENTREGABLE_DIR = API_DIR.parent

if str(ENTREGABLE_DIR) not in sys.path:
    sys.path.insert(0, str(ENTREGABLE_DIR))
if str(API_DIR) not in sys.path:
    sys.path.insert(0, str(API_DIR))

from api.main import app
from src.config import OPTIMAL_21_FEATURES, TAXONOMIA_PILARES

client = TestClient(app)


# ---------------------------------------------------------------------------
# 1. Pruebas de Diagnóstico y Estado
# ---------------------------------------------------------------------------
def test_root_endpoint():
    """Verifica el estado del servicio, versión 2.4.0 y marcos regulatorios."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert data["version"] == "2.4.0"
    assert data["features_count"] == 21
    assert "Fed SR 11-7" in data["frameworks"]
    assert "xgboost" in data["active_models"]
    assert data["default_model"] == "xgboost"


def test_health_endpoint():
    """Verifica que los modelos y el preprocesador oficial estén cargados en memoria."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["preprocessor_loaded"] is True
    assert data["scorecard_status"] == "loaded"
    assert data["total_models_loaded"] >= 6
    assert data["available_models"]["xgboost"] is True
    assert data["available_models"]["lightgbm"] is True
    assert data["default_model"] == "xgboost"


# ---------------------------------------------------------------------------
# 2. Pruebas de Catálogo de Modelos y Taxonomía de Características
# ---------------------------------------------------------------------------
def test_models_catalog_endpoint():
    """Verifica el catálogo completo de los 6 modelos y el Scorecard con sus tau*."""
    response = client.get("/models")
    assert response.status_code == 200
    data = response.json()
    assert data["default_model"] == "xgboost"
    assert data["total_models"] >= 7
    model_ids = [m["model_id"] for m in data["models"]]
    expected_ids = [
        "voting_classifier",
        "lightgbm",
        "random_forest",
        "catboost",
        "xgboost",
        "logistic_regression",
        "scorecard"
    ]
    for expected in expected_ids:
        assert expected in model_ids

    # Comprobación de umbrales tau* oficiales (12M)
    model_dict = {m["model_id"]: m for m in data["models"]}
    assert model_dict["voting_classifier"]["threshold_12m"] == 0.220
    assert model_dict["lightgbm"]["threshold_12m"] == 0.224
    assert model_dict["random_forest"]["threshold_12m"] == 0.182
    assert model_dict["catboost"]["threshold_12m"] == 0.038
    assert model_dict["xgboost"]["threshold_12m"] == 0.158
    assert model_dict["logistic_regression"]["threshold_12m"] == 0.170
    assert model_dict["scorecard"]["threshold_12m"] == 0.110


def test_features_info_endpoint():
    """Verifica las 21 características canónicas y su mapeo a los 5 pilares."""
    response = client.get("/features")
    assert response.status_code == 200
    data = response.json()
    assert data["total_features"] == 21
    assert len(data["features"]) == 21

    feature_names = [f["feature_name"] for f in data["features"]]
    for opt_feat in OPTIMAL_21_FEATURES:
        assert opt_feat in feature_names

    # Comprobar que los 5 pilares regulatorios existen
    expected_pillars = list(TAXONOMIA_PILARES.keys())
    for p in expected_pillars:
        assert p in data["pillars"]


# ---------------------------------------------------------------------------
# 3. Pruebas de Muestras Empíricas
# ---------------------------------------------------------------------------
def test_sample_company_endpoint():
    """Verifica la generación de muestras corporativas reales (solvente, distressed, random)."""
    for mode in ["solvent", "distressed", "random"]:
        response = client.get(f"/sample_company?mode={mode}")
        assert response.status_code == 200
        data = response.json()
        assert "company_name" in data
        assert "year" in data
        assert "tag_WorkingCapital" in data
        assert "tag_NetIncomeLoss" in data
        assert "merton_distance_to_default" in data


# ---------------------------------------------------------------------------
# 4. Pruebas de Inferencia Predictiva (Single Prediction)
# ---------------------------------------------------------------------------
def test_predict_solvent_company():
    """Verifica que una empresa solvente no dispare alertas de quiebra."""
    sample_res = client.get("/sample_company?mode=solvent")
    sample_data = sample_res.json()

    response = client.post("/predict", json=sample_data)
    assert response.status_code == 200
    pred = response.json()
    assert pred["distress_decision"] == 0
    assert pred["bankruptcy_probability"] < pred["operational_threshold"]
    assert pred["risk_tier"] in [
        "Grado de Inversión (Alta Calidad)",
        "Vigilancia Preventiva (Riesgo Moderado)"
    ]
    assert pred["estimated_cost_usd"] > 0
    assert "Pilar Contable (SEC)" in pred["pillar_contributions"]


def test_predict_distressed_company():
    """Verifica que una empresa en dificultades financieras active la alerta (decision = 1)."""
    sample_res = client.get("/sample_company?mode=distressed")
    sample_data = sample_res.json()

    response = client.post("/predict", json=sample_data)
    assert response.status_code == 200
    pred = response.json()
    assert pred["distress_decision"] == 1
    assert pred["bankruptcy_probability"] >= pred["operational_threshold"]
    assert pred["risk_tier"] == "Alto Riesgo de Insolvencia (Quiebra / Distress)"
    assert pred["estimated_cost_usd"] > 0


def test_predict_across_all_six_models():
    """Verifica la inferencia correcta en los 6 modelos oficiales de producción."""
    sample_res = client.get("/sample_company?mode=distressed")
    sample_data = sample_res.json()

    models = [
        "voting_classifier",
        "lightgbm",
        "random_forest",
        "catboost",
        "xgboost",
        "logistic_regression"
    ]
    for model_name in models:
        response = client.post(f"/predict?model_name={model_name}", json=sample_data)
        assert response.status_code == 200, f"Fallo al predecir con {model_name}"
        pred = response.json()
        assert 0.0 <= pred["bankruptcy_probability"] <= 1.0
        assert pred["operational_threshold"] > 0.0
        assert pred["distress_decision"] in [0, 1]
        assert "estimated_cost_usd" in pred
        assert "model_version" in pred


# ---------------------------------------------------------------------------
# 5. Pruebas de Inferencia por Lotes (Batch Prediction)
# ---------------------------------------------------------------------------
def test_batch_prediction():
    """Verifica la evaluación por lotes y las métricas agregadas de cartera."""
    solv = client.get("/sample_company?mode=solvent").json()
    dist = client.get("/sample_company?mode=distressed").json()

    batch_payload = {"companies": [solv, dist]}
    response = client.post("/predict/batch", json=batch_payload)
    assert response.status_code == 200
    batch_res = response.json()
    assert batch_res["total_processed"] == 2
    assert batch_res["total_alerts"] >= 1
    assert 0.0 <= batch_res["portfolio_average_pd"] <= 1.0
    assert batch_res["total_estimated_cost_usd"] > 0.0
    assert len(batch_res["predictions"]) == 2


# ---------------------------------------------------------------------------
# 6. Pruebas de Explicabilidad XAI (Valores SHAP Reales y Pilares)
# ---------------------------------------------------------------------------
def test_explain_endpoint_real_shap():
    """Verifica la descomposición aditiva SHAP con TreeExplainer y mapeo de pilares."""
    sample_res = client.get("/sample_company?mode=distressed")
    sample_data = sample_res.json()

    response = client.post("/explain?model_name=lightgbm", json=sample_data)
    assert response.status_code == 200
    explain_data = response.json()

    assert "base_value_logit" in explain_data
    assert "final_score_logit" in explain_data
    assert "bankruptcy_probability" in explain_data
    assert len(explain_data["top_risk_drivers"]) > 0

    # Validar estructura y pilares de los factores explicativos
    valid_pillars = set(TAXONOMIA_PILARES.keys())
    for factor in explain_data["top_risk_drivers"]:
        assert factor["feature_name"] in OPTIMAL_21_FEATURES
        assert isinstance(factor["shap_value"], float)
        assert factor["information_pillar"] in valid_pillars
        assert factor["risk_impact"] in ["Incrementa Riesgo", "Mitiga Riesgo"]

    # Validar resumen de pilares
    assert "pillar_summary" in explain_data
    assert "Pilar Contable (SEC)" in explain_data["pillar_summary"]
    assert "Federal Reserve SR 11-7" in explain_data["regulatory_compliance"]


# ---------------------------------------------------------------------------
# 7. Pruebas del Scorecard Basilea III (WoE)
# ---------------------------------------------------------------------------
def test_scorecard_endpoints():
    """Verifica el cálculo de puntos del Scorecard de Basilea III (rango 300 - 850)."""
    # GET con muestra solvente
    res_solv = client.get("/scorecard?mode=solvent")
    assert res_solv.status_code == 200
    data_solv = res_solv.json()
    assert 300.0 <= data_solv["credit_score"] <= 850.0
    assert data_solv["rating_band"] in ["AAA", "AA", "A", "BBB", "BB", "B", "CCC", "D"]

    # GET con muestra en dificultades
    res_dist = client.get("/scorecard?mode=distressed")
    assert res_dist.status_code == 200
    data_dist = res_dist.json()
    assert 300.0 <= data_dist["credit_score"] <= 850.0
    # La empresa en quiebra debe obtener una puntuación inferior a la solvente
    assert data_dist["credit_score"] < data_solv["credit_score"]

    # POST con payload completo
    sample_payload = client.get("/sample_company?mode=solvent").json()
    res_post = client.post("/scorecard", json=sample_payload)
    assert res_post.status_code == 200
    assert 300.0 <= res_post.json()["credit_score"] <= 850.0


# ---------------------------------------------------------------------------
# 8. Pruebas de Compatibilidad Retroactiva (Campos Legacy)
# ---------------------------------------------------------------------------
def test_legacy_payload_backwards_compatibility():
    """Verifica que la API procese payloads con nomenclatura contable legacy."""
    legacy_payload = {
        "company_name": "Legacy Enterprise Inc",
        "year": 2023,
        "current_assets": 8500000.0,
        "total_current_liabilities": 3500000.0,
        "total_assets": 25000000.0,
        "net_income": 2100000.0,
        "retained_earnings": 5000000.0,
        "market_value": 45000000.0,
        "distance_to_default": 4.8,
        "inflation": 2.8,
        "interest_rate": 4.5
    }

    response = client.post("/predict", json=legacy_payload)
    assert response.status_code == 200
    pred = response.json()
    assert pred["company_name"] == "Legacy Enterprise Inc"
    assert pred["distress_decision"] in [0, 1]
    assert 0.0 <= pred["bankruptcy_probability"] <= 1.0


# ---------------------------------------------------------------------------
# 9. Pruebas de Normalización Anti-Sesgo de Inflación
# ---------------------------------------------------------------------------
def test_macro_inflation_anti_skew_normalization():
    """Verifica que tanto porcentajes (2.5%) como índices CPI brutos (244.5) se procesen sin error."""
    base_sample = client.get("/sample_company?mode=solvent").json()

    # Caso A: Inflación enviada como tasa porcentual anual
    sample_pct = dict(base_sample)
    sample_pct["macro_inflation"] = 2.8
    res_pct = client.post("/predict", json=sample_pct)
    assert res_pct.status_code == 200

    # Caso B: Inflación enviada como nivel de índice CPI
    sample_cpi = dict(base_sample)
    sample_cpi["macro_inflation"] = 243.1
    res_cpi = client.post("/predict", json=sample_cpi)
    assert res_cpi.status_code == 200

    # Ambas predicciones deben arrojar valores coherentes dentro del espacio muestral
    assert abs(res_pct.json()["bankruptcy_probability"] - res_cpi.json()["bankruptcy_probability"]) < 0.10
