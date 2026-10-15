# Microservicio RESTful: Predicción de Quiebra Corporativa y Solvencia S&P 500

Servicio de alto rendimiento para la evaluación de riesgo crediticio institucional, clasificación de insolvencia y explicabilidad aditiva XAI, desarrollado con **FastAPI** y validación formal **Pydantic v2**.

Cumple con las exigencias normativas de la **Federal Reserve (Fed SR 11-7)**, los estándares de capital regulatorio de **Basilea III**, la provisión por pérdidas esperadas de **NIIF 9 / IFRS 9** y los requisitos de transparencia del **Reglamento de Inteligencia Artificial de la Unión Europea (EU AI Act, Arts. 12, 13 y 14)**.

---

## 1. Puesta en Marcha

### Requisitos Previos
- Python 3.11+
- Entorno virtual con dependencias (`fastapi`, `uvicorn`, `pydantic`, `joblib`, `scikit-learn`, `lightgbm`, `xgboost`, `catboost`, `shap`, `pandas`, `numpy`).

### Ejecución Local

```bash
# Opción A: Lanzador directo con banner informativo
python Entregable/api/run_api.py

# Opción B: Mediante Uvicorn CLI
uvicorn api.main:app --app-dir Entregable/api --host 127.0.0.1 --port 8000 --reload
```

Una vez iniciado el microservicio:
- **Documentación Interactiva Swagger UI:** `http://127.0.0.1:8000/docs`
- **Documentación Alternativa ReDoc:** `http://127.0.0.1:8000/redoc`

---

## 2. Catálogo Oficial de Modelos y Umbrales Óptimos ($\tau^*$)

El servicio implementa un catálogo con **6 modelos calibrados** más el **Scorecard WOE de Basilea III**. Cada arquitectura cuenta con su umbral óptimo operativo ($\tau^*$) calibrado mediante optimización asimétrica de costes institucionales bajo una penalización de **24.0x** ($C_{\mathrm{FN}} = \$6,000,000$ USD vs $C_{\mathrm{FP}} = \$250,000$ USD para una exposición de \$10M USD):

| Modelo | ID API | Rango | $\tau^*$ (12M) | $\tau^*$ (24M) | PR-AUC | ROC-AUC | Coste 12M | Arquitectura / Rol |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **XGBoost Classifier** | `xgboost` | **Rank 1 SOTA** | **0.158** | 0.184 | **0.8551** | **0.9763** | **$2,380.25M** | Campeón oficial 12M con GPU, mínima pérdida institucional y mayor PR-AUC |
| **LightGBM Classifier** | `lightgbm` | **Rank 2** | **0.224** | 0.256 | **0.8501** | **0.9759** | **$2,383.00M** | Campeón a 24M y segundo a 12M con mínimos FN (solo 127 fallos no detectados) |
| **Random Forest Classifier** | `random_forest` | **Rank 3** | **0.182** | 0.202 | **0.8450** | **0.9751** | **$2,410.25M** | Máxima precisión (45.90%) y especificidad (84.78%), reduciendo FP a 6,089 |
| **Voting Classifier Ensemble** | `voting_classifier` | **Rank 4** | **0.220** | 0.258 | **0.8449** | **0.9751** | **$2,432.75M** | Soft Voting uniforme (25% c/u). No consigue mejorar a los modelos de árboles |
| **CatBoost Classifier** | `catboost` | **Rank 5** | **0.038** | 0.082 | **0.8462** | **0.9744** | **$2,577.50M** | Árboles simétricos regulares en GPU con calibración en baja probabilidad |
| **Logistic Regression ElasticNet** | `logistic_regression` | **Rank 6** | **0.170** | 0.224 | **0.4193** | **0.7964** | **$6,064.25M** | Regresión logística regularizada lineal (Baseline analítico clásico) |
| **Basel III WoE Scorecard** | `scorecard` | **Regulatorio** | **0.110** | — | **0.6500** | **0.8950** | — | Scorecard crediticio monótono interpretable (escala 300 a 850 puntos) |

---

## 3. Vector Canónico de 21 Características y Taxonomía de Pilares

El pipeline oficial de preprocesamiento (`models/preprocessor_pipeline.joblib`) normaliza e imputa el vector de **21 variables óptimas** estructurado en **5 pilares de información**:

### Pilar 1: Contable y Balance SEC EDGAR (9 Variables)
- `tag_NetIncomeLoss`: Resultado neto consolidado del ejercicio (USD).
- `tag_WorkingCapital`: Fondo de maniobra o capital circulante neto ($AC - PC$, USD).
- `tag_Assets`: Activo total consolidado (USD).
- `pub_lag_days`: Días naturales transcurridos entre cierre fiscal y depósito en SEC.
- `tag_RetainedEarningsAccumulatedDeficit`: Beneficios retenidos o déficit acumulado (USD).
- `tag_EntityCommonStockSharesOutstanding`: Acciones comunes en circulación.
- `tag_CommonStockSharesAuthorized`: Acciones comunes autorizadas statutariamente.
- `tag_CommonStockParOrStatedValuePerShare`: Valor nominal por acción común (USD/acc).
- `tag_CommonStockValue`: Capital social común registrado en balance (USD).

### Pilar 2: Bursátil y Estructural Merton-KMV (3 Variables)
- `merton_distance_to_default`: Distancia al default estructural ($\sigma$).
- `stock_price_close`: Precio de cotización de cierre bursátil (USD).
- `market_cap`: Capitalización bursátil total de mercado (USD).

### Pilar 3: Macroeconómico y Regímenes FRED (4 Variables)
- `macro_inflation`: Inflación macroeconómica FRED. **Normalización Anti-Sesgo Automática**: si se envía una tasa anual $\le 25.0$ (ej. $2.8\%$), se proyecta al nivel base del índice CPI ($\sim 236.468 \times (1 + \pi / 100)$) evitando anomalías de $-30\sigma$ en inferencia.
- `macro_real_gdp_growth_yoy`: Crecimiento interanual del PIB real de EE.UU. (%).
- `macro_interest_rate`: Tipo de interés oficial de la Reserva Federal (Fed Funds Rate, %).
- `macro_regime_expansion_prob`: Probabilidad de régimen expansivo (modelo GMM).

### Pilar 4: Sentimiento NLP y Noticias FinBERT (3 Variables)
- `news_sentiment_range_6m`: Rango y dispersión semántica del sentimiento a 6 meses.
- `market_news_sentiment_mean`: Sentimiento agregado de los medios sobre el mercado.
- `news_sentiment_avg`: Sentimiento medio específico sobre las noticias de la compañía.

### Pilar 5: Arquetipos Corporativos GMM (2 Variables)
- `corp_archetype_prob_1`: Probabilidad de pertenencia al Arquetipo Corporativo 1.
- `corp_archetype_prob_2`: Probabilidad de pertenencia al Arquetipo Corporativo 2.

> **Compatibilidad Retroactiva:** El esquema `CompanyFinancialInput` admite campos tradicionales (`current_assets`, `total_assets`, `net_income`, `retained_earnings`, `market_value`, `working_capital`, etc.), mapeándolos automáticamente a sus tags canónicos.

---

## 4. Endpoints RESTful Disponibles

### Diagnóstico y Catálogo
- `GET /`: Metadatos generales, versión 2.4.0, marcos regulatorios y modelos activos (o Cockpit HTML en navegador).
- `GET /ui`: Cockpit Dashboard Interactivo GUI en HTML5/Tailwind/SVG con medidor de probabilidad y visualización en tiempo real.
- `GET /health`: Estado de salud, verificación de artefactos en memoria y diagnóstico de los 6 modelos.
- `GET /model_info`: Metadatos oficiales del benchmark, umbrales y directivas (Fed SR 11-7, Basilea III, EU AI Act, FCRA).
- `GET /models`: Catálogo exhaustivo de modelos con métricas de rendimiento y umbrales.
- `GET /features`: Descripción de las 21 variables agrupadas por sus 5 pilares regulatorios.
- `GET /sample_company?mode=solvent|distressed|random`: Recupera una empresa real de SEC EDGAR para pruebas.
- `GET /cache_status`: Telemetría del estado de la caché SQLite multinivel y políticas TTL.

### Datos en Vivo (Live Data)
- `GET /fetch_company?ticker=AAPL`: Consulta en vivo por ticker bursátil con caché multinivel (Strategy A: Cache-Aside).
- `POST /fetch_and_predict?ticker=AAPL`: Descarga en vivo de balance, mercado (Merton DD) y noticias FinBERT + predicción individual y comparativa multimodelo.

### Inferencia
- `POST /predict?model_name=voting_classifier`: Predicción individual con cálculo de probabilidad calibrada, clasificación de riesgo, coste asimétrico institucional 24x y cartas de acción adversa ECOA/FCRA.
- `POST /predict_comparison`: Comparación simultánea de los 6 modelos oficiales y el Scorecard de Basilea III.
- `POST /predict_batch` o `POST /predict/batch`: Evaluación por lotes con inferencia vectorizada y auditoría EU AI Act Art. 12 en `logs/audit_logs.jsonl`.

### Explicabilidad XAI
- `POST /explain?model_name=voting_classifier`: Explicación local aditiva con valores SHAP reales (`shap.TreeExplainer`) desglosados por los 5 pilares regulatorios para justificación de cartas de acción adversa (*ECOA / FCRA / EU AI Act*).

### Scorecard Regulatorio Basilea III
- `GET /scorecard?mode=distressed`: Cálculo de puntuación crediticia (300 a 850 puntos) y calificación de riesgo (AAA a D).
- `POST /scorecard`: Evaluación de scorecard enviando un payload contable completo.

---

## 5. Ejemplos de Petición y Respuesta

### 5.1. Inferencia Individual (`POST /predict`)

```bash
curl -X POST "http://127.0.0.1:8000/predict?model_name=voting_classifier" \
     -H "Content-Type: application/json" \
     -d '{
       "company_name": "Apex Energy Inc",
       "year": 2024,
       "tag_WorkingCapital": -18000000.0,
       "tag_NetIncomeLoss": -7500000.0,
       "merton_distance_to_default": 0.85,
       "tag_Assets": 25000000.0,
       "macro_inflation": 3.2,
       "macro_interest_rate": 5.25
     }'
```

**Respuesta JSON:**
```json
{
  "company_name": "Apex Energy Inc",
  "year": 2024,
  "model_version": "Voting Classifier Ensemble",
  "model_rank": "Rank 1 SOTA (Champion)",
  "bankruptcy_probability": 0.2854,
  "operational_threshold": 0.22,
  "distress_decision": 1,
  "risk_tier": "Alto Riesgo de Insolvencia (Quiebra / Distress)",
  "estimated_cost_usd": 178650.0,
  "regulatory_framework": "Fed SR 11-7, Basel III, IFRS 9, EU AI Act",
  "pillar_contributions": {
    "Pilar Contable (SEC)": 4.12,
    "Pilar Bursatil y Estructural": 2.85,
    "Pilar Macroeconomico y Regimenes": 1.45,
    "Pilar Sentimiento NLP y Noticias": 0.32,
    "Pilar Arquetipos Corporativos": 0.88
  }
}
```

### 5.2. Explicabilidad Local SHAP (`POST /explain`)

```bash
curl -X POST "http://127.0.0.1:8000/explain?model_name=lightgbm" \
     -H "Content-Type: application/json" \
     -d '{
       "company_name": "Apex Energy Inc",
       "tag_WorkingCapital": -18000000.0,
       "tag_NetIncomeLoss": -7500000.0,
       "merton_distance_to_default": 0.85
     }'
```

**Respuesta JSON:**
```json
{
  "company_name": "Apex Energy Inc",
  "model_version": "TreeExplainer (LIGHTGBM)",
  "base_value_logit": -3.1407,
  "final_score_logit": -0.8542,
  "bankruptcy_probability": 0.2985,
  "top_risk_drivers": [
    {
      "feature_name": "tag_WorkingCapital",
      "shap_value": 0.8421,
      "feature_value": -18000000.0,
      "information_pillar": "Pilar Contable (SEC)",
      "risk_impact": "Incrementa Riesgo"
    },
    {
      "feature_name": "merton_distance_to_default",
      "shap_value": 0.6512,
      "feature_value": 0.85,
      "information_pillar": "Pilar Bursatil y Estructural",
      "risk_impact": "Incrementa Riesgo"
    }
  ],
  "pillar_summary": {
    "Pilar Contable (SEC)": 1.245,
    "Pilar Bursatil y Estructural": 0.651,
    "Pilar Macroeconomico y Regimenes": 0.124,
    "Pilar Sentimiento NLP y Noticias": 0.052,
    "Pilar Arquetipos Corporativos": 0.214
  },
  "regulatory_compliance": "Federal Reserve SR 11-7 / ECOA Adverse Action Notice / EU AI Act Art. 13-14"
}
```

### 5.3. Scorecard de Crédito Basilea III (`GET /scorecard`)

```bash
curl -X GET "http://127.0.0.1:8000/scorecard?mode=distressed"
```

**Respuesta JSON:**
```json
{
  "company_name": "GRYPHON DIGITAL MINING, INC.",
  "credit_score": 420.95,
  "rating_band": "D",
  "pd_probability": 0.6241,
  "decision": "Rechazado / Alerta de Insolvencia",
  "points_breakdown": {
    "tag_WorkingCapital": -18115000.0,
    "tag_NetIncomeLoss": -7698285.71,
    "merton_distance_to_default": 2.83,
    "macro_interest_rate": 5.33,
    "market_news_sentiment_mean": 0.08
  },
  "regulatory_framework": "Basel III / IFRS 9 Weight-of-Evidence Scorecard"
}
```

---

## 6. Ejecución de la Suite de Pruebas Automatizadas

El microservicio incluye una batería de pruebas unitarias y de integración que verifica la conformidad de cada endpoint y componente algorítmico:

```bash
pytest Entregable/api/test_api.py -v
```
