# 💳 Sistema de Detección de Fraude en Tarjetas de Crédito

Proyecto de *Machine Learning* aplicado a la detección de fraude transaccional. Este desarrollo abarca desde el análisis exploratorio y la ingeniería de variables hasta el despliegue de un modelo robusto con monitorización de latencia y explicabilidad bajo estándares regulatorios.

## 🚀 Descripción del Proyecto
El objetivo es identificar patrones de fraude en un entorno altamente desbalanceado. El proyecto aplica técnicas de econometría y *Data Science* para minimizar el riesgo financiero del banco, equilibrando la precisión en la detección con la eficiencia operativa.

**Dataset fuente:** [Credit Card Transactions Dataset - Kaggle](https://www.kaggle.com/datasets/priyamchoksi/credit-card-transactions-dataset/data)

## 🛠 Arquitectura y Metodología
- **Ingeniería de Características:** Transformaciones cíclicas (trigonometría para variables temporales) y análisis de frecuencias.
- **Modelado:** Comparativa entre **XGBoost**  y **CatBoost**  mediante test A/B.
- **Optimización:** *Threshold Tuning* orientado a negocio (Cumplimiento de umbrales de *Recall*).
- **Latencia:** Benchmark de inferencia con latencia P99 sub-milisegundo.
- **Explicabilidad:** Auditoría de decisiones individuales mediante **SHAP** para cumplimiento normativo.
- **MLOps:** Serialización de modelos en formato binario nativo (`.cbm`) listo para despliegue en producción.

## 📊 Resultados Clave
| Métrica | XGBoost  | CatBoost  |
| :--- | :--- | :--- |
| **Recall (Sensibilidad)** | 97.00% | 97.87% |
| **Precision** | 32.03% | 26.59% |
| **Latencia P99** | 2.79 ms | 0.56 ms |

## 💡 Conclusión Estratégica
La selección definitiva debe ponderar la política de riesgos de la entidad frente a los costes de implementación:

1. **Prioridad en la Experiencia de Usuario:** Si el objetivo es minimizar la fricción del cliente y optimizar costes de *Call Center*, **XGBoost** es la opción que mejor equilibra precisión y operatividad.
2. **Política de Tolerancia Cero al Fraude:** Si el coste marginal de investigar más de alertas debido a la reducción de 6 p.p. en precisión es inferior al capital salvado por elevar la sensibilidad al **97,87%**, **CatBoost** es el candidato a implementar.
3. **Restricciones Críticas de Latencia:** En infraestructuras con cuellos de botella extremos (ej. pagos *contactless* masivos), la arquitectura de **CatBoost** es la elección necesaria por viabilidad técnica.

---