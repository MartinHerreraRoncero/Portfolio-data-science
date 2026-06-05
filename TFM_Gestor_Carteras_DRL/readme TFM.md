# Inteligencia Artificial Aplicada a los Mercados Financieros: Optimización de Carteras mediante Deep Reinforcement Learning

## Descripción del Proyecto
Este proyecto es mi Trabajo Fin de Máster (TFM) y se enfoca en construir un modelo avanzado de optimización de carteras utilizando técnicas de Deep Learning y Deep Reinforcement Learning. El objetivo principal es superar las deficiencias de los métodos tradicionales de optimización (como el modelo de Markowitz), eliminando el supuesto de linealidad y mejorando la capacidad de adaptación en entornos de mercado dinámicos.

## Enfoque Técnico
* **Algoritmos DRL:** Se evaluaron algoritmos de vanguardia basados en arquitecturas actor-crítico, concretamente PPO, DDPG y SAC.
* **Arquitectura de Redes:** El modelo óptimo desarrollado combina una red neuronal Transformer como extractora de características globales y una red LSTM para el modelado de dependencias temporales, alimentando finalmente al algoritmo PPO.
* **Variables Explicativas:** El entorno del agente se construyó utilizando el índice S&P 500, incorporando indicadores de análisis técnico, variables macroeconómicas y términos de corrección de error derivados de análisis de cointegración (VECM).

## Resultados
El modelo desarrollado logra superar de manera significativa el rendimiento del índice S&P 500 y las optimizaciones clásicas de Markowitz, maximizando la rentabilidad ajustada al riesgo de la cartera.

## ⚠️ Nota del Autor
Este proyecto supuso mi inmersión inicial en Python y mi primer contacto real con el desarrollo de Data Science a un nivel técnico profundo. Como economista adentrándome en la ingeniería de datos y el aprendizaje automático, abordé el modelado, la arquitectura de redes y el entrenamiento de los agentes de manera completamente autodidacta y sin supervisión técnica directa. 

El código y la implementación algorítmica presenten áreas de optimización y mejora que he detectado conforme mi formación en data science ha ido avanzando. No obstante, los resultados obtenidos reflejan un trabajo analítico sólido y son una muestra fehaciente de mi capacidad de aprendizaje autónomo y resolución de problemas complejos en el ámbito de la Inteligencia Artificial aplicada a las finanzas. Por último comentar que el algoritmo es demasiado complejo y no representa ganancia respecto a otros mas ligeros como muestro en el TFM, aun sabiendo esto quise desarrollarlo con un alto grado de complejidad para aprender y ponerme a prueba de lo que era capaz de hacer.

> **Documentación:** En este directorio se incluye el documento PDF completo con la memoria del TFM detallando la fundamentación matemática, la metodología y los resultados obtenidos.
