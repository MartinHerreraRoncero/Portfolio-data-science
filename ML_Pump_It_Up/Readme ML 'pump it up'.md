# Machine Learning Competition: Pump it Up - Data Mining the Water Table

## Descripción del Proyecto
Este proyecto contiene la resolución de la competición de Machine Learning "Pump it Up", organizada por DrivenData. El objetivo de la tarea es predecir el estado operativo de diversas bombas de agua a partir de un conjunto de datos complejo con múltiples variables numéricas y categóricas.

🔗 **Enlace a la competición:** [Pump it up: Data Mining the Water Table](https://www.drivendata.org/competitions/7/pump-it-up-data-mining-the-water-table/).

## Retos Técnicos y Enfoque
La predicción es un problema de clasificación multiclase donde la variable objetivo ("status_group") no está balanceada y se divide en tres estados: funcional, no funcional y necesita reparación. 

Para abordar el problema, el pipeline de datos implementa:
* **Ingeniería de Características:** Limpieza de datos inconsistentes y creación de nuevas variables como `pump_age`.
* **Imputación de Valores Nulos:** Uso de algoritmos como `KNNImputer` basado en similitud geoespacial.
* **Encoding de Variables Categóricas:** Manejo de alta cardinalidad utilizando `TargetEncoder` (con suavizado) para variables complejas y `LabelEncoder` para el resto.
* **Modelado y Ensamblado:** Entrenamiento de un `VotingClassifier` mediante "soft voting", combinando la robustez de un modelo `RandomForestClassifier` y un `XGBClassifier`. Se aplicó también la técnica de Random Over Sampling para corregir el desbalanceo severo de clases.
