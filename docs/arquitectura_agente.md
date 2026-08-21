# Arquitectura del Agente PaperStats

## 1. Objetivo de la arquitectura

PaperStats está estructurado como un **agente de consultoría estadística**: una herramienta de apoyo al análisis estadístico, a la investigación científica y a la ciencia de datos. No es un generador de código ni un sistema autónomo que decide por sí mismo qué análisis ejecutar.

El orquestador central es **`PaperStatsFlow`**, que controla el flujo de trabajo de principio a fin: diagnóstico de los datos, decisión sobre datos faltantes, preparación de los datos e invocación de análisis estadísticos **bajo demanda**. El agente traduce los resultados técnicos producidos por los motores estadísticos a una presentación comprensible para el usuario, sin modificar ni recalcular dichos resultados.

El sistema se apoya sobre los motores estadísticos existentes (pruebas de hipótesis, imputación de datos faltantes, análisis exploratorio) y sobre una capa de presentación independiente. No incorpora en su flujo principal capacidades que no estén implementadas.

## 2. Diagrama completo del flujo

El flujo actual del agente es el siguiente:

```text
Usuario
   │
   ▼
PaperStatsFlow
   │
   ▼
diagnose()
   │
   ▼
diagnóstico de datos (perfil/QC + detección y diagnóstico de faltantes + recomendación)
   │
   ├── Sin datos faltantes ──► estado: sin_faltantes
   │        │
   │        └──► entregable_inicial() (análisis descriptivo/exploratorio)
   │                    │
   │                    └──► espera de solicitud explícita de análisis
   │
   ├── Con datos faltantes ──► estado: esperando_decision
   │        │
   │        └──► decisión del usuario
   │                 │
   │                 └──► imputación opt-in (imputar con method_override o accept_recommendation)
   │                          │
   │                          └──► datos preparados
   │
   ▼
análisis solicitado (bajo demanda)
   │
   ▼
resultado técnico (del motor estadístico)
   │
   ▼
Deliverable (modelo neutral de presentación)
   │
   ▼
renderización (Markdown y PDF)
   │
   ▼
PDF (informe)
   │
   ▼
espera de nueva decisión del usuario
```

Puntos donde P-FLOW se detiene y espera una decisión:

- Tras `diagnose()`, si hay datos con faltantes, se detiene en `esperando_decision` y espera que el usuario decida si imputa y con qué método.
- Tras `entregable_inicial()`, con datos limpios, se detiene y espera que el usuario solicite explícitamente un análisis estadístico.
- Tras la generación de un `Deliverable`, espera el siguiente comando o decisión del usuario.

No hay llamadas a componentes que no estén demostradas por el código; el flujo se compone únicamente de los módulos descritos en las secciones siguientes.

## 3. Estados de P-FLOW y puntos explícitos de parada

P-FLOW mantiene una máquina de estados con los estados reales existentes. Cada estado define una condición de entrada, la acción del agente, el punto de parada y, cuando corresponde, la decisión que debe tomar el usuario.

| Estado | Condición de entrada | Acción del agente | Punto de parada | Decisión del usuario |
|---|---|---|---|---|
| `sin_diagnostico` | Inicio o reinicio del flujo | `diagnose()`: perfil/QC, detección de faltantes y recomendación inicial | Agente listo para recibir los datos | Ninguna; se procede a diagnosticar |
| `esperando_decision` | La detección encuentra datos faltantes | Presenta la recomendación y espera la decisión de imputación | El agente se detiene | El usuario debe decidir si imputar y con qué método |
| `sin_faltantes` | No se detectaron datos faltantes | Construye el entregable inicial descriptivo/exploratorio | El agente cierra ofreciendo análisis | El usuario debe solicitar un análisis concreto |
| `datos_preparados` | Tras aceptar la imputación o tener datos limpios | Permite ejecutar análisis bajo demanda (`analizar(**kwargs)`) | El agente se detiene hasta recibir una solicitud | El usuario solicita el análisis que desea |
| `revisar` | La validación posterior a la imputación (E6) devuelve un veredicto de revisión | Lanza `ValidationRevisionError` y detiene el flujo | Punto de detención; no continúa silenciosamente | El usuario resuelve la validación o revisa la imputación |

**Comportamiento de `ValidationRevisionError`:** cuando la validación posterior a la imputación (E6) concluye que los datos imputados requieren revisión (por ejemplo, si quedan valores imposibles o faltantes residuales), el flujo se detiene de forma explícita. Con el modo estricto activo, se lanza `ValidationRevisionError` para impedir continuar silenciosamente hacia el análisis mientras exista un problema pendiente. El agente no avanza hasta que el usuario resuelva la situación.

Ningún estado ejecuta análisis que no hayan sido solicitados por el usuario.

## 4. Rutas de decisión

### 4.1 Datos sin faltantes

Cuando no se detectan datos faltantes, el flujo procede de la siguiente manera:

1. `diagnose()` construye el diagnóstico (perfil y control de calidad).
2. Al no existir datos faltantes, el estado pasa a `sin_faltantes`.
3. `entregable_inicial()` construye el entregable descriptivo/exploratorio (control de calidad, estadísticos descriptivos, frecuencias y hallazgos exploratorios marcados como tales, sin inferencia).
4. El agente cierra y espera que el usuario **solicite explícitamente** un análisis estadístico.

No se ejecuta ningún análisis inferencial en esta etapa; solo se presentan resultados descriptivos.

### 4.2 Datos con faltantes

Cuando el diagnóstico detecta datos faltantes, el flujo se detiene en `esperando_decision` y la imputación **requiere una decisión explícita** del usuario. `imputar()` no actúa de forma automática; exige una de estas condiciones:

- **`method_override`**: el usuario indica qué método usar para cada variable con faltantes, prevaleciendo sobre la recomendación automática.
- **`accept_recommendation=True`**: el usuario acepta explícitamente la recomendación generada.

Es correcto afirmar que **no existe imputación automática**: `imputar()` no avanza sin una de estas dos condiciones. Una vez el usuario decide, los datos quedan preparados y el flujo puede pasar a `datos_preparados`.

## 5. Componentes que actúan como herramientas

Los componentes que participan realmente en el flujo P-FLOW son los siguientes:

| Componente | Responsabilidad real |
|---|---|
| **`PaperStatsFlow`** | Orquestador central. Controla los estados, las decisiones del usuario y la transición entre fases. No contiene la lógica estadística principal; delega en los motores. Construye o coordina los entregables. |
| **`DatasetStatisticalAnalyzer`** | Motor estadístico que ejecuta pruebas de hipótesis, calcula estadísticos y evalúa supuestos cuando el usuario solicita un análisis (`analizar(**kwargs)`). |
| **`MissingDataPipeline`** | Motor de datos faltantes. Detecta, diagnostica e imputa datos faltantes. La imputación es opt-in y se inyecta con decisión explícita. Produce los reportes de evaluación y validación. |
| **`DeliverableGenerator`** | Capa de presentación. Traduce los resultados técnicos de los motores a un modelo neutral entregable (`Deliverable` → `Section` → `Item`). No invoca motores estadísticos ni recalcula resultados. |
| **`renderers`** | Capa de presentación. Representan el modelo neutral `Deliverable` en Markdown o PDF. Consumen el `Deliverable` y no calculan estadísticas. |

**Aclaración sobre el orquestador:** `PaperStatsFlow` organiza la secuencia y coordina los componentes, pero la lógica estadística principal (pruebas, imputación, validación) reside en los motores. El orquestador no duplica cálculos estadísticos.

## 6. Dirección de dependencias

La dirección conceptual de las dependencias es una sola vía:

```text
PaperStatsFlow
   │
   ▼
motores estadísticos / pipeline de datos  (DatasetStatisticalAnalyzer, MissingDataPipeline)
   │
   ▼
resultados técnicos
   │
   ▼
Deliverable  (modelo neutral de presentación)
   │
   ▼
renderers  (Markdown / PDF)
   │
   ▼
presentación (informe)
```

Así funciona en el código:

- El orquestador (`PaperStatsFlow`) invoca a los motores estadísticos para obtener resultados técnicos.
- Estos resultados se traducen a un `Deliverable` mediante `DeliverableGenerator`.
- La capa de presentación (`DeliverableGenerator` y `renderers`) consume información de los resultados, pero **no debe llamar a los motores estadísticos**, ni recalcular estadísticas ni tomar decisiones analíticas.

Esta separación mantiene que la presentación no reintroduce ni altera la lógica estadística.

## 7. Guardarraíles

El comportamiento actual del agente implementa una serie de guardarraíles permanentes:

1. **No imputación automática:** `imputar()` exige una decisión explícita del usuario (`method_override` o `accept_recommendation=True`). El comportamiento por defecto (`impute=False`) solo diagnostica; no imputa por su cuenta.

2. **No análisis no solicitado:** el agente no ejecuta análisis estadísticos que el usuario no haya pedido. Tras `entregable_inicial()` el flujo cierra y espera una solicitud explícita.

3. **Separación entre hallazgo exploratorio y análisis inferencial:** los hallazgos del entregable inicial y los módulos de EDA se marcan como exploratorios (sin pruebas de hipótesis), y se distinguen de los análisis solicitados, que sí incluyen método, supuestos, resultado e interpretación.

4. **No causalidad:** los resultados de asociación se presentan como correlaciones o asociaciones, no como relaciones causales. La arquitectura advierte que la significancia estadística no implica causalidad.

5. **No modificación de resultados:** el agente no modifica ni recalcula los valores estadísticos producidos por los motores. La capa de presentación los consume tal como se generaron.

6. **No generación de resultados estadísticos inexistentes:** el agente no inventa ni genera resultados que no provengan de los motores o del análisis disponible.

## 8. Verificación de salida

La verificación de la salida se apoya en el modelo neutral de presentación y en el seguimiento del último entregable:

- **Construcción del `Deliverable`:** `DeliverableGenerator` construye un modelo neutral a partir de los resultados técnicos, sin modificar los valores. Cada `Item` tiene un tipo (`text`, `bullets`, `table`, `hallazgo`) y los datos que conforman el resultado presentado.
- **Uso de `_last_deliverable`:** `PaperStatsFlow` guarda el último `Deliverable` generado por cualquiera de los métodos de entregable (`entregable_inicial`, `entregable_missing`, `entregable_analisis`). El comando `informe()` se apoya en este registro.
- **Los entregables se construyen antes que el informe:** `informe()` requiere que exista previamente un `Deliverable`. Sin él, no hay nada que exportar.
- **`informe()` requiere un `Deliverable` previo:** si `_last_deliverable` no existe, `informe()` lanza un `ValueError`.
- **Markdown y PDF parten del `Deliverable`:** ambas representaciones (texto Markdown y PDF) se generan a partir del mismo modelo neutral.
- **El PDF no se genera a partir del Markdown:** el PDF se construye directamente desde el `Deliverable`, no transformando el Markdown.
- **La capa de renderización no debe volver a calcular resultados estadísticos:** los renderers solo representan el contenido del `Deliverable`; no ejecutan ningún cálculo ni verificación estadística nueva.

No se debe afirmar que el renderer realiza una validación estadística que no exista en el código actual.

## 9. Manejo de errores

El manejo de errores documenta comportamientos observables comprobados en el flujo:

- **`ValueError` cuando `informe()` no tiene un `Deliverable` previo:** si el usuario solicita "Informe" sin haber generado antes un entregable, `informe()` no tiene qué exportar y lanza un `ValueError` con una indicación de que primero se genere un entregable (`entregable_inicial`, `entregable_missing` o `entregable_analisis`).
- **`ValidationRevisionError` cuando la validación requiere revisión:** si la validación posterior a la imputación (E6) devuelve un veredicto de "Revisar", el flujo se detiene explícitamente lanzándolo, lo que impide continuar silenciosamente con un proceso revisado.
- **`build_analisis_secciones` ante información incompleta:** cuando el resultado de un análisis solicitado no contiene todas las claves esperadas (por ejemplo, faltan `estadístico`, `p_valor`, `interpretación`), el entregable se construye con la información disponible sin producir un error fatal ni lanzar una excepción, ya que el modelo de presentación puede omitir secciones incompletas.
- **Ante tipos de `Item` no reconocidos en los renderers:** según el comportamiento comprobado en la capa de renderizado, un `Item` cuyo tipo no es uno de los soportados (`text`, `bullets`, `table`, `hallazgo`) no se representa (se omite) sin que el renderer genere una excepción a causa de ese tipo.

**IMPORTANTE:** el incumplimiento de un supuesto estadístico no debe interpretarse como una conclusión estadística ("No rechazar H0"). Un supuesto no satisfecho no es un resultado de prueba de hipótesis; es solo el reflejo de una condición del modelo que no se cumple. Esta distinción evita que se lea un incumplimiento de supuestos como un resultado inferencial del análisis.

### 9.1 Aplicabilidad del manejo de errores

La arquitectura actual de PaperStats es **100 % local** (sin llamadas HTTP, sin `requests`/`httpx`/`urllib`/`socket`, sin parsing de JSON de respuestas externas, sin API remota, `BaseLLMClient`/`RuleBasedLLMClient` sin red; verificado por `grep -rn` vacío en `src/` y `requirements.txt` sin cliente HTTP). Por lo tanto:

- **Timeout / `ConnectionError` / `429` / `JSONDecodeError` de servicios externos: NO APLICA** a la arquitectura actual porque no existen operaciones de red ni respuestas JSON externas. Implementar `except TimeoutError`/`ConnectionError`/`RateLimitError`/`JSONDecodeError` para HTTP sería código inalcanzable y artificial. Queda fuera del alcance hasta que se integre un servicio externo (Ollama), en cuyo caso se evaluaría con criterios de `docs/arquitectura_agente.md §12`.
- **Errores de procesamiento local: SÍ APLICA y deben manejarse por tipo.** Operaciones reales pueden levantar `ValueError` (parámetros o datos insuficientes, p. ej. `IterativeImputer` sin filas completas, `regresion` sin columna predictora), `TypeError` (tipo de DataFrame), `KeyError` (columna inexistente) y `numpy.linalg.LinAlgError` (fallo SVD en `MICE`), además de `ValueError`/`TypeError` en correlaciones `pearsonr`/`spearmanr`. Estos se capturan de forma específica en `src/missing_data/pipeline.py` (E4 y bucle de imputación) y `src/missing_data/evaluation.py` y `src/analysis/dataset_analyzer.py` sin cambiar el comportamiento estadístico.
- **Degradación controlada: SÍ APLICA y ya existe en `MissingDataPipeline`.** Si falla la evaluación opcional E4 (`pipeline.py:289-302`), el pipeline registra `logger.warning` y continúa con `evaluation_report=None` hacia E5. Si falla la imputación de una variable (`pipeline.py:340-354`), se registra en `skipped_variables` con `logger.warning` y continúa; la validación E6 decide el veredicto final. En `evaluation.py:278-282` el fallo de un método se registra como `error` y no se propaga.

## Aplicabilidad de la ingeniería de prompts

Con evidencia concreta del código actual:

1. **El flujo principal P-FLOW no consume prompts ni depende de un LLM.** `src/orchestration/flow.py:28-42` importa solo `DatasetStatisticalAnalyzer`, `MissingDataPipeline`, `DeliverableGenerator` y `ReportGenerator`; no importa `src/llm`. El diagrama de dependencias de `§6` es `PaperStatsFlow → motores estadísticos → resultados → Deliverable → renderers`, sin rama LLM. `docs/roadmap.md:112-115` y `AGENTS.md:172` confirman "Ollama pendiente (no implementar aún)".

2. **`src/llm/base.py` contiene una interfaz abstracta preparada para desacoplamiento, pero no un sistema de prompts.** `src/llm/base.py:14-21` define `class BaseLLMClient(ABC)` con `generate(self, prompt: str)` abstracto como contrato futuro para un eventual `OllamaLLMClient`; es una capa de abstracción, no un repositorio de prompts versionados, sin `system`/`user`/`template`/`jinja` ni contrato de salida.

3. **`RuleBasedLLMClient` es determinista y heurístico; su método `generate(prompt)` no utiliza el contenido del prompt.** `src/llm/base.py:42-51` implementa `generate` como `return f"[Respuesta Heurística] Procesado prompt de {len(prompt)} caracteres."` — solo calcula `len(prompt)`, no lo interpreta como instrucción.

4. **`extract_metadata()` y `analyze_methodology()` funcionan mediante regex, keywords y reglas, no mediante prompts.** `src/llm/base.py:53-137` usa `re.findall`, `re.search`, `re.sub` y listas de `keywords`/`software_keywords`; `src/llm/base.py:139-201` usa coincidencias `in text_lower` sobre listas `possible_tests`/`possible_models`. `src/article/extractor.py:18-36` y `src/article/analyzer.py:19-34` inyectan `RuleBasedLLMClient` y delegan en esos métodos heurísticos, sin construir ni enviar prompts.

5. **No existen archivos de prompts, templates Jinja, system prompts, contratos JSON de salida ni versionado v1/v2.** Verificado: `ls src/llm/` solo contiene `base.py` y `__init__.py`; `grep -RIn "prompt" src --include="*.py"` solo retorna las dos firmas `generate(self, prompt: str)` vacías; `grep -RIn "template|jinja|system.*role" src docs` → 0 hits; no existe directorio `prompts/`.

6. **Crear prompts v1/v2 en este momento sería introducir artefactos sin consumidor real.** No hay operación en P-FLOW ni en `RuleBasedLLMClient` que lea `prompts/v1` o `v2`; serían plantillas muertas, sin llamada, sin test que las ejercite y sin métrica que las compare, violando `AGENTS.md:172` y `docs/roadmap.md:112-115`.

7. **Por tanto, la ingeniería de prompts queda declarada como NO APLICABLE a la arquitectura actual.** La evaluación del criterio debe considerar que no hay componente generativo que consuma prompts, tal como `§12` documenta "Actualmente no existe un modelo LLM generativo integrado" y `§10 llm/` "No participa en las decisiones estadísticas de P-FLOW".

8. **Si posteriormente se incorpora un componente generativo que consuma prompts,** entonces sí deberán implementarse: prompts versionados (ej. `prompts/extraction/v1.j2`, `v2.j2`); `role` (system); contrato de salida (ej. JSON schema); restricciones; iteración `v1 → v2`; y justificación del cambio mediante una métrica (ej. precisión/recall/F1 de extracción de los 19 puntos sobre un set de artículos). Hasta entonces, queda como criterio futuro, igual que manejo de `timeout`/`429` documentado en `§9.1`.

## 10. Componentes fuera del flujo principal

Además del núcleo del flujo P-FLOW existen otros módulos que forman parte del repositorio pero **no participan** en el flujo principal de análisis estadístico del agente. No se debe afirmar que el flujo los invoca si el código no lo demuestra.

### visualization/

Proporciona funciones para generar gráficos estadísticos (histograma, boxplot, dispersión, Q-Q, matriz de correlación). Estas capacidades existen y pueden invocarse de forma independiente, pero no forman parte de la ruta automática del flujo P-FLOW, ni los gráficos se generan automáticamente como parte de un entregable o del informe.

### article/

Constituye un flujo separado de procesamiento y análisis de artículos científicos (ingesta de PDF, extracción de los 19 puntos clave y análisis metodológico). Es un flujo independiente que no forma parte de la ruta principal de análisis de datos de P-FLOW; el agente puede tomarlos como referencia, pero no los procesa como parte de la rutina estadística de datos.

### llm/

Contiene `BaseLLMClient` y `RuleBasedLLMClient`. **No existe actualmente un modelo LLM generativo integrado al flujo principal de P-FLOW.**

`RuleBasedLLMClient` es un cliente **heurístico/determinista**: está diseñado para procesar textos (asociados a la extracción de artículos) sin depender de un servicio externo. Pertenece al flujo separado relacionado con artículos y **no participa en las decisiones estadísticas de P-FLOW**. `BaseLLMClient` es una interfaz abstracta de la que se derivarían futuros clientes.

### reports/

Genera informes técnicos de Markdown para los módulos técnicos (artículo, dataset y reporte de datos faltantes E1–E6). Estos corresponden al nivel de desarrollo técnico. Se diferencian claramente de **`deliverables/`**, que produce los entregables destinados al usuario final (mediante `DeliverableGenerator` y los renderers). Los reportes técnicos y los entregables de usuario no deben confundirse.

### Sobre integraciones

No se documentan integraciones entre estos módulos externos y el flujo principal de P-FLOW más allá de las que el código demuestra. No se debe afirmar que `visualization/`, `article/` o `llm/` invocan al flujo principal de P-FLOW ni que son invocados sin evidencia de código.

## 11. Comparación arquitectónica

Esta sección es una **interpretación arquitectónica del sistema actual**, no una funcionalidad implementada o medible del propio código.

El tipo de sistema **agente estadístico** (el papel actual de PaperStats) se caracteriza por:

- Orquestación explícita y control del flujo.
- Máquina de estados que representa el progreso del análisis.
- Decisiones del usuario (imputación, análisis) tomadas de forma opt-in.
- Motores estadísticos especializados para producir los resultados.
- Separación entre la capa de cálculo y la capa de presentación.
- Guardarraíles frente a la imputación automática.
- Guardarraíles frente a la ejecución de análisis no solicitados.

En comparación:

- **Script tradicional:** normalmente consiste en una secuencia lineal de instrucciones y ofrece menos mecanismos explícitos para representar estados y decisiones del usuario. Resulta menos adecuado para un flujo de consultoría en el que el usuario debe decidir, por ejemplo, cómo tratar los datos faltantes o qué análisis ejecutar.
- **Chatbot general:** está orientado principalmente a la interacción conversacional y no necesariamente incorpora una máquina de estados, motores estadísticos especializados ni restricciones explícitas sobre imputación y ejecución de análisis.
- **RAG (retrieval-augmented generation):** arquitectura que recupera documentos y genera texto tomándolos de base. Podría ser complementaria en una futura arquitectura para recuperación de literatura o documentación, pero no es el mecanismo central actual de PaperStats, cuyo problema es el análisis estadístico de los datos propios del usuario.

La arquitectura actual de PaperStats corresponde al problema de consultoría y análisis estadístico porque:

- Permite al usuario tomar decisiones (por ejemplo, cómo tratar los datos faltantes o qué análisis ejecutar).
- Presenta información técnicamente fundamentada por los motores estadísticos.
- Separa la capa de presentación de la capa de cálculo estadístico.
- Reduce el riesgo de introducir resultados estadísticos que no provengan de los motores, al mantener separado el cálculo estadístico de la presentación.

## 12. Decisión de modelo

### Estado actual

- **Actualmente no existe un modelo LLM generativo integrado al flujo principal de P-FLOW.**
- Las operaciones estadísticas se realizan mediante los motores estadísticos existentes (pruebas de hipótesis, análisis descriptivo, imputación y validación de datos, todos basados en bibliotecas de Python).
- `RuleBasedLLMClient` es un cliente **heurístico/determinista** que pertenece a un flujo separado (relacionado con la extracción de artículos científicos) y **no participa en las decisiones estadísticas de P-FLOW**.
- **Ollama no está integrado actualmente.** Los motores estadísticos no dependen de ningún modelo de lenguaje.

### Criterios de selección futura

Si en el futuro se decide incorporar un modelo de lenguaje, debería evaluarse según criterios de selección explícitos:

- **Desacoplamiento del motor estadístico:** el núcleo estadístico debe seguir funcionando de forma independiente al proveedor de modelo de lenguaje.
- **Opt-in:** cualquier integración debe exigir autorización explícita del usuario, igual que la imputación.
- **Privacidad/seguridad:** los datos y las claves no deben filtrarse; gestionarse mediante variables de entorno o de forma segura.
- **Reproducibilidad:** la elección debe permitir reproducir los análisis.
- **Costo computacional:** evaluar el costo de procesos y de cómputo.
- **Latencia y contexto:** evaluar el tiempo de respuesta y la ventana de contexto necesarios.

No se selecciona ni se recomienda aquí un modelo LLM particular, ni se recomienda utilizar Ollama en el estado actual. No se inventan ni se describen como ya existentes las arquitecturas futuras. El documento describe únicamente el estado actual y estos criterios de decisión futura.