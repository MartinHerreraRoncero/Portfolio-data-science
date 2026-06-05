# Inteligencia Artificial Aplicada a los Mercados Financieros: Optimización de Carteras mediante Deep Reinforcement Learning

## Descripción del Proyecto
Este proyecto es mi Trabajo Fin de Máster (TFM) y se enfoca en construir un modelo avanzado de optimización de carteras utilizando técnicas de Deep Learning y Deep Reinforcement Learning. El objetivo principal es superar las deficiencias de los métodos tradicionales de optimización (como el modelo de Markowitz), eliminando el supuesto de linealidad y mejorando la adaptación en entornos dinámicos.

## Enfoque Técnico
* **Algoritmos DRL:** Se evaluaron algoritmos como PPO, DDPG y SAC.
* **Arquitectura de Redes:** El modelo óptimo desarrollado combina una red neuronal Transformer como extractora de características globales y una red LSTM para el modelado de dependencias temporales, alimentando finalmente al algoritmo PPO.
* **Variables Explicativas:** El entorno del agente se construyó utilizando el índice S&P 500, incorporando indicadores de análisis técnico, variables macroeconómicas y términos de corrección de error derivados de análisis de cointegración (VECM).

## Resultados
El modelo desarrollado logra superar de manera significativa el rendimiento del índice S&P 500 y las optimizaciones clásicas de Markowitz, maximizando la rentabilidad ajustada al riesgo de la cartera.

> **Nota:** En este directorio se incluye el documento PDF completo con la memoria del TFM detallando la fundamentación matemática, la metodología y los resultados obtenidos.
