# Martín Herrera Roncero | Portfolio de Data Science e Inteligencia Artificial

Soy economista con formación de máster en Finanzas y en Inteligencia Artificial, Data Science y Business Intelligence. Mi trabajo combina análisis económico, estadística y programación para abordar problemas de riesgo financiero, inversión y predicción.

Este portfolio reúne proyectos en Python que abarcan preparación de datos, ingeniería de variables, comparación de modelos, explicabilidad y desarrollo de servicios de inferencia. Mi interés se centra en conectar el rendimiento de los modelos con las decisiones y los costes del problema que resuelven.

## Proyectos

### Detección temprana de quiebras y optimización del coste de negocio

**[Explorar el TFM](./TFM%20detecci%C3%B3n%20quiebras%20opt%20coste%20de%20negocio/)**

Sistema de predicción de insolvencia corporativa que integra información contable de SEC EDGAR, datos bursátiles, variables macroeconómicas y sentimiento de noticias. El criterio de decisión incorpora el coste asimétrico de los falsos negativos y los falsos positivos.

- Preparación de datos financieros, construcción y selección de variables y evaluación temporal.
- Comparación de XGBoost, LightGBM, CatBoost, Random Forest, regresión logística y un ensamble de votación.
- Ajuste de umbrales orientado al coste de negocio y evaluación a distintos horizontes.
- Explicabilidad con SHAP, pruebas de estrés y análisis de deriva de datos y predicciones.
- API con FastAPI y panel web para consultar predicciones y comparar modelos.

La carpeta incluye la memoria del TFM, notebooks, código fuente, modelos y resultados de evaluación.

### Detección de fraude en tarjetas de crédito

**[Explorar el proyecto](./ML_Credit_Card_Fraud_Detection/)**

Clasificación de transacciones fraudulentas en un contexto de clases desbalanceadas, con atención al equilibrio entre detección de fraude, falsas alertas y tiempo de inferencia.

- Ingeniería de variables temporales y de frecuencia.
- Comparación de XGBoost y CatBoost y ajuste del umbral de clasificación.
- Evaluación de precision, recall y latencia de inferencia.
- Interpretación de predicciones individuales mediante SHAP.

### Optimización de carteras con Deep Reinforcement Learning

**[Explorar el TFM](./TFM_Gestor_Carteras_DRL/)**

Investigación sobre aprendizaje por refuerzo aplicado a la asignación de activos, con comparación frente al S&P 500 y enfoques clásicos de optimización de carteras.

- Evaluación de algoritmos PPO, DDPG y SAC.
- Desarrollo de una arquitectura que combina Transformer y LSTM.
- Incorporación de indicadores técnicos, variables macroeconómicas y señales de cointegración mediante VECM.
- Análisis de rentabilidad y riesgo, junto con una reflexión sobre la complejidad de los modelos.

Incluye el código y la memoria del trabajo.

### Análisis de sentimiento y redes en retransmisiones del Mundial 2026

**[Explorar el proyecto](./world-cup-sentiment-analysis/)**

Pipeline de análisis de audio de retransmisiones de la selección española para estudiar la evolución emocional del relato y las relaciones entre jugadores mencionados.

- Transcripción con faster-whisper y normalización temporal por períodos.
- Identificación de jugadores mediante alias y expresiones regulares.
- Análisis de sentimiento en español con pysentimiento.
- Construcción de grafos de coocurrencia y cálculo de centralidades con NetworkX.

### Pump it Up: estado operativo de bombas de agua

**[Explorar el proyecto](./ML_Pump_It_Up/)**

Resolución de un problema de clasificación multiclase de la competición de DrivenData: predecir si una bomba de agua es funcional, no funcional o necesita reparación.

- Limpieza de datos e ingeniería de variables como la antigüedad de la bomba.
- Imputación mediante KNN y codificación de variables categóricas de alta cardinalidad.
- Tratamiento del desbalanceo y combinación de Random Forest y XGBoost mediante votación suave.

### Minería de datos y modelización predictiva

**[Explorar el proyecto](./Mineria_Datos_Predictiva/)**

Análisis de un conjunto de datos agrícola centrado en la depuración, la justificación estadística de las transformaciones y la aplicación de modelos de regresión y clasificación.

El notebook documenta el proceso analítico. Su ejecución completa requiere un módulo académico de funciones propias que no está incluido, tal como se indica en la documentación del proyecto.

## Tecnologías

| Área | Herramientas utilizadas en los proyectos |
| --- | --- |
| Análisis y tratamiento de datos | Python, pandas, NumPy, SciPy, DuckDB, PyArrow |
| Machine Learning | scikit-learn, XGBoost, LightGBM, CatBoost, imbalanced-learn, Optuna |
| Deep Learning y NLP | PyTorch, Transformers, faster-whisper, pysentimiento |
| Econometría y aprendizaje por refuerzo | statsmodels, VECM, PPO, DDPG, SAC |
| Explicabilidad y visualización | SHAP, Matplotlib, Seaborn, NetworkX |
| Servicios y experimentación | FastAPI, Pydantic, Uvicorn, Jupyter |

## Cómo consultar el portfolio

Cada proyecto está organizado en su propia carpeta. Consulta su documentación para conocer los datos, la metodología, los resultados y los requisitos de ejecución. Las dependencias y los artefactos disponibles varían entre proyectos; los que incluyen una memoria en PDF permiten profundizar en el planteamiento y las conclusiones.
