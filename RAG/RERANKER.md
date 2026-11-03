# Reranker experimental

Implementado Qwen/Qwen3-Reranker-0.6B con revisión fija `e61197ed45024b0ed8a2d74b80b4d909f1255473`.
Documentación del modelo: https://huggingface.co/Qwen/Qwen3-Reranker-0.6B

Se recuperan hasta 50 candidatos por publicación y se intercalan hasta un pool de 50 por necesidad. El reranker puntúa cada pareja petición/ficha antes del recorte final a diez. Conserva las puntuaciones originales y añade puntuación, rango, tamaño del pool e identidad del reranker. La puntuación no acredita hechos ni sustituye las validaciones de citas o cálculos. El pool podría seguir omitiendo fuentes.

El worker usa CPU, dos hilos, lote uno y un límite de 2048 tokens por pareja sin truncación silenciosa. Se ejecuta en un subproceso para liberar la memoria después; no usa la GPU ni descarga modelos durante consultas. Los archivos locales se verifican contra el manifiesto del snapshot descargado. No se entrena ningún modelo.

## Primera evaluación (histórica)

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

## Comparación posterior: cinco casos reales

La petición interpretada completa llega ahora al scorer, incluido el tipo de resultado solicitado. Se aplica diversidad genérica por publicación/archivo/objeto: primero la fila mejor puntuada de cada objeto y después relleno por puntuación. No hay reglas específicas de pensiones. Se conservan IDs, scores y rangos antes/después; las condiciones no se truncan. El pool sigue siendo50 y la salida10 por necesidad. Una selección diversa puede desplazar filas útiles del mismo objeto: no garantiza pertinencia ni debe adoptarse sin evaluación.

Se reutilizaron cinco preguntas reales y sus embeddings guardados, con golds fijados antes de ejecutar. Se verificaron13 celdas de las fuentes originales y los SHA256 de sus libros. Se evaluaron cuatro brazos sobre el mismo pool amplio: orden original, diversidad sin modelo, reranker puro y reranker con diversidad. El buscador habitual se compara adicionalmente con su recorte original. No hubo llamadas al chatbot, embeddings nuevos ni cambio de baseline.

| Caso | Buscador actual | Diversidad sola | Reranker solo | Reranker + diversidad |
|---|---:|---:|---:|---:|
| 2070 que fallaba | Fuera | Fuera | Fuera | 3 |
| 2070 que funcionaba | 1 | 1 | 1 | 1 |
| Gasto2050 | 8 | Fuera | Fuera | Fuera |
| Déficit público2050 | 1 | 1 | 2 | 2 |
| IRPF2050 | 2 | 2 | 1 | 1 |

Las posiciones corresponden al primer testigo de fuente esperado dentro de10, no a exactitud de respuesta. Cobertura:4/5 con el buscador y4/5 con la combinación; una ganancia y una pérdida. El reranker puro y la diversidad sola obtienen3/5. La combinación resuelve la recuperación de2070, pero empeora un control que funcionaba y baja el déficit del puesto1 al2. Añade48–74 segundos de workerCPU/carga por caso, sin incluir el chatbot. Siete controles sintéticos PASS. No se ha ejecutado una evaluación mixta amplia ni se certifica calidad conversacional. El caso2070 ya se usó durante desarrollo y no es un test independiente de generalización.

**Sigue desactivado por defecto.** El criterio de mejora sin pérdida de cobertura no se cumple; tampoco se afirma ausencia de regresiones de posición. No se ajustaron expectativas después de observar resultados. Los resultados anteriores negativos se conservan.

Resumen reproducible: `experiments/results/reranker_real_comparison_2026-11-03.json`. Protocolo congelado, scores completos y comprobación de fuentes: `evaluation/reranker_real_comparison_2026-11-03_v1/`. Para repetir en una carpeta nueva:

```powershell
.venv-langgraph\Scripts\python.exe -m experiments.compare_reranker_real_cases --output-dir evaluation/reranker_comparacion_nueva
```


## Revisión retrospectiva de equivalencias (sin repetir la ejecución)

La evaluación anterior no reconocía algunas fuentes semánticamente equivalentes con etiquetas diferentes. Se sustituye la ampliación automática por coincidencia de nombres/valores por un motor genérico de grupos de fuentes revisadas. El registro exige magnitud, ámbito, publicación, escenario, unidad, periodo, tipo de cantidad, justificación y procedencia de cada observación. Coincidir numéricamente no acredita equivalencia. Los valores se cotejan con su propia celda para detectar cambios de procedencia; no se usan para decidir equivalencia entre fuentes. Si no hay una equivalencia revisada, no se inventa. Las anotaciones de este corpus son datos de evaluación, no reglas del buscador ni del reranker.

Se revisaron14celdas originales y hashes de sus libros. Sólo se reinterpretaron los IDs y rankings guardados: cero inferencia, embeddings o búsquedas nuevas, cero tests. Los archivos históricos de comparación y protocolo permanecen intactos y sus hashes se registran en el nuevo informe. No se cambia la baseline aceptada. Esta anotación retrospectiva está informada por resultados ya vistos; no constituye un benchmark independiente.

| Caso | Buscador | Diversidad sola | Reranker solo | Reranker + diversidad |
|---|---:|---:|---:|---:|
| 2070 que fallaba | Fuera | 9 | Fuera | 3 |
| 2070 que funcionaba | 1 | 1 | 1 | 1 |
| Gasto2050 | 8 | 6 | 7 | 6 |
| Déficit2050, contrato2025* | 1 | 1 | 2 | 2 |
| IRPF2050 | 2 | 2 | 1 | 1 |

Sobre los contratos originales, la cobertura revisada es4/5,5/5,4/5,5/5 respectivamente. En2050, cuadro12 y gráficos1/4/12 contienen la misma proyección pública de la AIReF2026. En2070, el gráficoRE4.4, filaOpinión2025, también reproduce la proyección base: diversidad sola ya lo había recuperado en el puesto9. El cuadro9 de2025 presenta una pequeña discrepancia numérica documentada; su aceptación para cobertura conceptual no certifica precisión numérica ni permite intercambiar valores sin explicación.

*La referencia de déficit fijada en2025 contradice la preferencia latest_supported de la conversación, pues la publicación2026 contiene una revisión diferente (gráficos3/32). No se mezclan ambas publicaciones como equivalentes ni se sustituye silenciosamente el objetivo original. Este caso se marca publication_target_conflict y no se usa para afirmar calidad de respuesta a la pregunta original. Sin ese contrato pendiente, las coberturas son3/4,4/4,3/4,4/4. El evaluador todavía mide presencia de evidencia entre10, no selección final ni exactitud de respuesta.

**Reranker desactivado.** La revisión no demuestra cobertura adicional respecto a diversidad sola. Mejora la posición de algunas fuentes, pero requiere el tiempo CPU ya medido; no se ha repetido esa medición. La diversidad tampoco se activa por esta revisión retrospectiva.

Motor: `experiments/source_equivalence.py`. Registro revisado: `experiments/fixtures/source_equivalence_2026-11-03.json`. Informe: `experiments/results/reranker_equivalence_review_2026-11-03.json`. Revisión detallada y comprobación de fuentes: `evaluation/reranker_equivalence_review_2026-11-03/`. El comando `python -m experiments.review_saved_reranker --saved-dir <carpeta_con_resultados> --output <archivo_nuevo>` sólo lee resultados guardados, no ejecuta modelos ni búsquedas.
# Estado actual — 2026-11-03

Diversificación sin modelo activada por defecto en la recuperación de datos: pool de 50, representantes por `(book, source_path, block)` y relleno hasta diez resultados por necesidad siguiendo el orden de recuperación. Los registros sin bloque se mantienen como objetos independientes. Las trazas separan esta selección del reranker. El modelo reranker continúa siendo opt-in mediante `RAG_ENABLE_RERANKER=1`; no se ha activado. Esta activación se apoya en la revisión retrospectiva descrita al final del documento; no se ejecutan nuevas pruebas ni se promueve una línea base. Las secciones anteriores de experimentación que describen ambas opciones desactivadas corresponden al estado previo.
