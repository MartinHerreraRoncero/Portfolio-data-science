# Evaluación inicial de Ollama — 21 de octubre de 2026

GPU: RTX 5070, 12227 MiB de VRAM. Uso inicial observado: 1962 MiB. Ollama 0.34.4.
Modelos instalados: qwen3.5:4b-q4_K_M y qwen3.5:9b, ambos Q4_K_M.
No existe wrapper de delegación en la carpeta; se aplicó la excepción de AGENTS.md y Codex realizó las mediciones.

## Método
Una solicitud secuencial, think=false, temperatura 0, semilla 42, salida limitada a 256 tokens, sin cambios persistentes en Ollama. Dos ejecuciones por modelo y ventana con una entrada de 43 tokens. Una prueba adicional por modelo con 5611 tokens sintéticos repetitivos y ventana de 16384. La velocidad es eval_count/(eval_duration/1e9); no incluye carga ni procesamiento de entrada. GPU medida después de responder, no pico de memoria. size y size_vram de /api/ps coincidieron en todas las pruebas: carga completa en GPU según Ollama. Modelos descargados de memoria al terminar.

| Modelo | Ventana | Generación corta (tokens/s) | VRAM total observada (MiB) |
|---|---:|---:|---:|
| 4B | 4096 | 139.45–139.96 | 5625–5641 |
| 4B | 8192 | 134.30–140.87 | 5779–5844 |
| 4B | 16384 | 137.58–137.75 | 6131–6132 |
| 9B | 4096 | 90.71–91.00 | 8376–8404 |
| 9B | 8192 | 90.78–91.21 | 8437–8446 |
| 9B | 16384 | 90.11–90.65 | 8635–8660 |

Con 5611 tokens: 4B, 130.51 tokens/s, entrada en 1.011 s, VRAM total 6057 MiB; 9B, 88.82 tokens/s, entrada en 1.532 s, VRAM total 8646 MiB. Estos tiempos no son tiempo al primer token. La entrada repetitiva no representa diversidad documental ni evalúa calidad. No se ha llenado la ventana de 16K ni probado concurrencia, visión o ventanas superiores. El tamaño en disco no equivale a VRAM de ejecución.

Recomendación: 9B, contexto inicial 8192, una solicitud, think=false; conservar 4B como alternativa rápida y baseline. 16K cabe en la configuración probada. Comparar calidad con documentos reales antes de elegir definitivamente. El índice de documentos puede vivir en RAM/disco; solo los fragmentos recuperados entran en el contexto. Presupuestar embeddings/reranker por separado antes de cargarlos simultáneamente.

Datos y scripts reproducibles: work/ollama-benchmark.json, work/ollama-long-context.json, work/benchmark.ps1, work/benchmark-long.ps1.

## Embeddings locales — primera integración

Modelo instalado: embeddinggemma:300m, BF16, 307,58 millones de parámetros, contexto de 2.048 tokens y salida de 768 dimensiones. Digest: 85462619ee721b466c5927d109d4cb765861907d5417b9109caebc4e614679f1. Ollama reportó size=size_vram=681417113 bytes, unos 649,8 MiB: carga completa en GPU según el servidor, no medida del pico de VRAM.

La CLI generó 13 vectores de los chunks reales en 11,598 segundos de pared, incluyendo inicio de Python, peticiones HTTP e inferencia. Una llamada posterior para tres consultas con el modelo cargado tardó 0,077 segundos. No son benchmarks ni latencias garantizadas. Comprobados dimensiones, valores finitos, norma L2 unitaria, correspondencia de cada chunk y huella del archivo de entrada. truncate=false evita recortes silenciosos por exceso de contexto.

Evidencias: work/embeddinggemma-smoke.json, work/embeddings-integration-metrics.json y work/embeddings-smoke-results.json. Artefacto: data/processed/airef_2025_embeddings.json. No se ha probado co-residencia con Qwen; no inferir esta propiedad del consumo individual.
