# ⚽ Copa Mundial 2026: Análisis Narrativo, Emocional y de Redes de las Retransmisiones de España

Este proyecto procesa y analiza las retransmisiones en audio de la Selección Española en el Mundial 2026 mediante un *pipeline* *end-to-end* en Python. El flujo transforma el audio bruto en series temporales de sentimiento y grafos de co-ocurrencia narrativa.

### 🛠️ Metodología
* **ASR (Transcripción):** Procesamiento de audio con `faster-whisper` (`large-v3`) con control estricto de alucinaciones.
* **Normalización Temporal:** Filtrado de intervalos efectivos y cálculo del tiempo relativo ($t_{rel}$) por período.
* **NER & Análisis de Sentimiento:** Mapeo de alias de jugadores mediante Regex ordenadas y cuantificación de la polaridad neta ($S_t = P(\text{POS})_t - P(\text{NEG})_t$) con `pysentimiento` (RoBERTa).
* **Modelado de Redes:** Construcción de grafos no dirigidos $G=(V,E)$ con `NetworkX` para extraer métricas de centralidad (*Betweenness*, *Eigenvector* y *Degree*).

### 📊 Resultados Principales
* **Dinámica Emocional:** Series temporales que capturan los picos de euforia y pánico a lo largo de cada partido.
* **Perfil por Jugador:** Ranking de polaridad neta que identifica a los creadores de ritmo positivo y a los jugadores expuestos a mayor desgaste/tensión.
* **Topología Narrativa:** Mapeo de los circuitos de juego principales y los nodos conector en la retransmisión.

### 🚀 Próximos Pasos
* **Series Temporales:** Causalidad de Granger y *Markov Switching*.
* **Grafos Dinámicos:** Métricas en ventanas móviles (*rolling windows*).
* **Fusión Multimodal & Deep Learning:** Extracción de energía acústica (RMS con `librosa`) e integración con redes LSTM.
