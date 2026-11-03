# RAG local sobre pensiones en España

## Pregunta del proyecto
¿Qué proyectan los informes oficiales sobre el gasto en pensiones, bajo qué supuestos y qué cambia entre publicaciones?

Usuario inicial: analista que prepara una nota de seguimiento documental. Salida: respuesta breve con evidencias localizables. Las respuestas describen escenarios de las fuentes; no prueban causalidad ni predicen pensiones individuales.

## Primera entrega
Responder preguntas sobre un solo informe de AIReF, con citas de página y fragmento. Empezar por unas páginas seleccionadas del apartado de pensiones de la segunda Opinión de sostenibilidad de 2025. No comenzar mezclando instituciones o versiones.

Fuentes oficiales candidatas, localizadas el 21 de octubre de 2026:
- Informe principal 2025: https://www.airef.es/es/centro-documental/opiniones/segunda-opinion-sobre-la-sostenibilidad-de-las-administraciones-publicas-a-largo-plazo/
- Metodología de estimación del gasto: https://www.airef.es/es/centro-documental/documentos-tecnicos/metodologia-de-la-estimacion-del-gasto-en-pensiones/
- Ampliación temporal 2026: https://www.airef.es/es/estudios/estudio-sobre-la-regla-de-gasto-de-pensiones/

Antes de incorporar cada documento: verificar PDF, fecha, versión, condiciones de reutilización y calidad de extracción. Acceso público no implica permiso para redistribuir el PDF. Las fuentes están localizadas; todavía no se ha inspeccionado ni descargado su contenido completo.

## Incrementos didácticos
1. Leer manualmente unas páginas y redactar 5 preguntas con respuesta y evidencias; añadir 2 preguntas que esas páginas no permitan responder.
2. Extraer texto por página, conservar documento, fecha, página PDF y número impreso cuando exista; revisar extracción frente al original.
3. Dividir por secciones y fragmentos; inspeccionar ejemplos y mantener identificadores estables. Tablas complejas se excluyen inicialmente o se revisan por separado.
4. Crear embeddings locales y recuperar fragmentos, sin generador. Evaluar si aparece la evidencia esperada.
5. Entregar contexto al Qwen 9B mediante Ollama y generar respuestas con citas y posibilidad de abstención. Comparar contra el 4B.
6. Ampliar el conjunto de evaluación, separar desarrollo y evaluación reservada, y comparar búsqueda semántica, léxica e híbrida.
7. Añadir tablas verificadas, cálculos en Python y comparación de publicaciones con control de fecha y escenario.

## Criterios de aceptación iniciales
- Cada respuesta sustantiva permite localizar su evidencia en las páginas originales.
- Si las páginas no contienen la respuesta, el sistema lo indica.
- No confunde año de publicación con año proyectado, datos observados con proyecciones, porcentaje con puntos porcentuales ni gasto bruto con gasto neto.
- Se evalúan por separado recuperación, respuesta, citas y abstención; los umbrales se fijarán tras revisar los primeros casos y antes de comparar mejoras.
- Se registran modelo, parámetros, versión del corpus, fragmentos recuperados y tiempos de entrada/generación.

## Ejecución local
Punto de partida: qwen3.5:9b, num_ctx=8192, think=false, una solicitud simultánea. qwen3.5:4b-q4_K_M como comparación rápida. Modelo de embeddings pendiente de seleccionar y medir; no asumir que los generadores ya cumplen esa función. Índice en RAM/disco. Ninguna API de pago.

## Diferenciación prevista
Control temporal, escenario y unidad como metadatos; comparación de publicaciones; citas verificables; evaluación reproducible y análisis de errores; comparación de modelos locales. Cada ampliación debe justificar una mejora medida. La interfaz se incorpora después de validar el recorrido documental.
