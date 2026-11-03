# Reranker experimental

Implementado Qwen/Qwen3-Reranker-0.6B con revisión fija `e61197ed45024b0ed8a2d74b80b4d909f1255473`.
Documentación del modelo: https://huggingface.co/Qwen/Qwen3-Reranker-0.6B

Se recuperan hasta 50 candidatos por publicación y se intercalan hasta un pool de 50 por necesidad. El reranker puntúa cada pareja petición/ficha antes del recorte final a diez. Conserva las puntuaciones originales y añade puntuación, rango, tamaño del pool e identidad del reranker. La puntuación no acredita hechos ni sustituye las validaciones de citas o cálculos. El pool podría seguir omitiendo fuentes.

El worker usa CPU, dos hilos, lote uno y un límite de 2048 tokens por pareja sin truncación silenciosa. Se ejecuta en un subproceso para liberar la memoria después; no usa la GPU ni descarga modelos durante consultas. Los archivos locales se verifican contra el manifiesto del snapshot descargado. No se entrena ningún modelo.

## Estado y resultados

**Desactivado por defecto.** Se activa únicamente con `RAG_ENABLE_RERANKER=1` al iniciar el servidor. El caso real de 2070 falla, por lo que no se adopta este cambio como mejora validada. No se cambia la baseline aceptada.

Cinco casos con inferencia real: **4/5**. Escenario explícito, cobertura de año, agregado frente a componente y pool ajeno pasan. La recuperación de un nivel de gasto para 2070 falla: las variantes del cuadro de diferencias siguen ocupando los primeros puestos. Ese caso tarda aproximadamente54 segundos; los pequeños tardan entre6 y7 segundos, incluida la carga del worker. No se llamó al chatbot ni se generaron embeddings nuevos: el caso real reutiliza el vector de una traza y verifica la identidad del índice. Las cinco comprobaciones técnicas y12 tests de catálogo pasan. Estos resultados no certifican calidad conversacional ni ausencia de regresiones generales.

Resultados completos: `experiments/results/reranker_five_cases_2026-11-03.json`.

## Reproducir

Instalar `requirements-reranker.txt` en un entorno CPU separado. Aquí ya existían las dependencias en `.venv-docling`; no se modificó el entorno principal. El ejecutable del worker puede indicarse con `RAG_RERANKER_PYTHON`; por defecto usa `.venv-docling/Scripts/python.exe`.

Desde la raíz:

```powershell
.venv-docling\Scripts\python.exe -m experiments.prepare_graph_reranker
.venv-langgraph\Scripts\python.exe -m unittest discover -s tests -p test_graph_reranker.py
.venv-langgraph\Scripts\python.exe -m experiments.test_reranker_five_cases --output evaluation/reranker_nuevo/five_cases.json
```

El replay real necesita los assets locales y la traza `outputs/prototipo/5c5362d9-5219-4270-9bf7-bbc25346d63b.json`; no están publicados junto al código. El informe conserva el resultado negativo. Modelos, fuentes y trazas se excluyen de Git.
