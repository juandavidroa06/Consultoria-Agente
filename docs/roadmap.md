# ROADMAP Y DECISIONES — PAPERSTATS

Documento on-demand con el plan futuro, el historial de decisiones y las instrucciones de fases ya ejecutadas. No contiene reglas operativas permanentes del agente.

---

## 1. ESTADO ACTUAL (AGOSTO 2026)

El núcleo del proyecto está implementado y probado (**405 tests**):

- **Fase 1 — Núcleo de análisis de artículos científicos**: completada.
  - Extracción de texto de PDF/TXT/MD (`src/article/parser.py`).
  - Identificación de los 19 puntos clave (heurístico, `src/article/extractor.py` + `src/llm/base.py`).
  - Análisis metodológico de supuestos (`src/article/analyzer.py`).
  - Generación de informes Markdown con regla anti-alucinaciones (`src/reports/generator.py`).
  - Interfaz `BaseLLMClient` lista para conectar Ollama localmente.

- **Fase 2 — Análisis exploratorio y estadístico de datos**: completada.
  - Carga de datos CSV/Excel (`src/data/loader.py`).
  - Validación de calidad (`src/data/validator.py`).
  - EDA (`src/analysis/eda.py`).
  - 22 pruebas de hipótesis paramétricas y no paramétricas (`src/analysis/hypothesis.py`): 10 originales + 12 adicionales (Fase 9, E9-C).
  - Visualización estadística (`src/visualization/plots.py`).

- **Fase 3 — Analizador estadístico inteligente**: completada.
  - Coordinador autónomo `DatasetStatisticalAnalyzer` (`src/analysis/dataset_analyzer.py`).
  - Diagnóstico contextual de normalidad, criterios integrales de correlación y gestión explícita de supuestos.

- **Fase 4 — Datos faltantes E1–E3 (detección, diagnóstico, imputación)**: completada.
  - E1 detección robusta + placeholders (`src/missing_data/detection.py`, 29 tests).
  - E2 diagnóstico del mecanismo MCAR/MAR/MNAR (`src/missing_data/diagnostics.py`, 30 tests).
  - E3 registro y 10 métodos de imputación (`src/missing_data/methods.py`, `registry.py`, 45+17 tests).

- **Fase 5 — Datos faltantes E4 (evaluación artificial)**: completada.
  - `induce_missing` + `ArtificialMissingnessEvaluator` (RMSE/MAE/accuracy) (`src/missing_data/evaluation.py`, 25 tests).

- **Fase 6 — Datos faltantes E5 (selección de métodos)**: completada.
  - `ImputationSelector`, pesos `DEFAULT_WEIGHTS`, puertas duras y blending `0.5·perfil + 0.5·evidencia E4` (`src/missing_data/selector.py`, 32 tests).
  - `DatasetProfile`/`build_dataset_profile` creado en `src/analysis/profile.py`.

- **Fase 7 — Datos faltantes E6 (validación post-imputación)**: completada.
  - `ImputationValidator` con 7 comprobaciones y veredicto "Aceptable"/"Revisar" (`src/missing_data/validation.py`, 18 tests).

- **Fase 8 — Datos faltantes E7 (integración del pipeline)**: completada.
  - `MissingDataPipeline` (`impute` opt-in, `ValidationRevisionError`), integración con `DatasetStatisticalAnalyzer.analyze()` (sección `missing_data`) y `ReportGenerator.generate_missing_data_report()` (`src/missing_data/pipeline.py`, 19 tests + 3 de regresión de importación).
  - Verificación real sobre `data/raw/Drug Price.xlsx` en ambos modos: aprobada (todas las comprobaciones OK).

- **Fase 9 — E9 (auditoría y pruebas estadísticas adicionales §2.2)**: completada.
  - E9-A: tests de regresión de hallazgos de auditoría A1–A11 (`tests/test_audit_e9.py`, 24 tests).
  - E9-B: correcciones de los 11 hallazgos (contratos A1–A11) en `src/analysis/`, `src/article/`, `src/data/`, `src/llm/`, `src/reports/`. Cierre: guard de tamaño muestral en normalidad (n<3) y contrato `alpha`→`reject_h0`/`decision` en correlaciones.
  - E9-C: 12 pruebas estadísticas adicionales del §2.2 implementadas en `src/analysis/hypothesis.py` (`kolmogorov_smirnov_1samp_test`, `kolmogorov_smirnov_2samp_test`, `lilliefors_test`, `bartlett_test`, `breusch_pagan_test`, `white_test`, `durbin_watson_test`, `breusch_godfrey_test`, `reset_test`, `chi_square_test`, `tukey_hsd_test`, `permutation_test`) + `tests/test_hypothesis_additional.py` (51 tests). Exports públicos en `src/analysis/__init__.py`.
  - E9-D: integración de las pruebas con contexto claro en `DatasetStatisticalAnalyzer` — Lilliefors (diagnóstico complementario de normalidad, n≥4), Bartlett (solo bajo normalidad contextual), Tukey HSD (post hoc solo tras ANOVA de un factor significativa), chi-cuadrado de independencia (modo exploratorio, pares de categóricas) y permutaciones (verificación complementaria en comparación de 2 grupos no normal, semilla fija). Sin cambios en las decisiones paramétricas/no paramétricas existentes. `tests/test_dataset_analyzer_e9d.py` (12 tests).

- **Fase 10 — P-FLOW (flujo principal del agente)**: completada.
  - `PaperStatsFlow` (`src/orchestration/flow.py`, `tests/test_orchestration_flow.py`, 17 tests): orquestador de alto nivel que controla el flujo `DATASET → diagnose() → [ESPERAR DECISIÓN] → imputar() → analizar()`, sin duplicar lógica estadística.
  - `diagnose()` es estrictamente diagnóstica: perfilamiento/QC, E1–E2–E3 y recomendación de imputación E5; NO ejecuta EDA, inferencia ni recomendaciones de análisis (no existen claves `executed_test_results`/`recommendations`/`eda` en su resultado).
  - La imputación es explícita: `imputar()` exige `method_override` o `accept_recommendation=True`; nunca se imputa en silencio.
  - `MissingDataPipeline.run(impute=False)` ahora produce la recomendación E5 sin imputar (cambio aditivo). `method_override` (variable → método) permite aplicar la decisión del usuario (validado contra el registro).
  - `DatasetStatisticalAnalyzer` NO se modificó; `analizar()` delega en él solo sobre datos preparados.

- **Fase 11 — Capa de entregables de usuario (`src/deliverables/`)**: completada.
  - Separación formal entre el nivel técnico de desarrollo y el entregable que recibe el usuario. `src/deliverables/` es una capa de PRESENTACIÓN pura: consume los resultados técnicos de los motores y los traduce a un modelo neutral (`Deliverable`→`Section`→`Item` con `kind` text/bullets/table/hallazgo), sin recalcular, sin seleccionar pruebas/modelos, sin decidir análisis, sin imputar y sin introducir reglas estadísticas.
  - Módulos: `generator.py` (modelo neutral + coordinador `DeliverableGenerator` + `render_markdown`, primera representación aislada para futuros renderers HTML/PDF/tablas), `quality.py`, `eda.py` (con **hallazgos exploratorios** marcados y sin pruebas inferenciales), `missing.py` (reporte de faltantes + diagnóstico del mecanismo + recomendación de imputación en lenguaje de usuario, sin exponer pesos/scores/componentes/umbrales), `analysis.py` (resultado de un análisis ya solicitado). `tests/test_deliverables.py` (17 tests).
  - `PaperStatsFlow` solo gana métodos de presentación ADITIVOS: `entregable_inicial()`, `entregable_missing()`, `entregable_analisis()` (+ `_resultados_eda()` que orquesta los motores de EDA). `diagnose()`, `imputar()`, `analizar()` y la máquina de estados quedan intactos.
  - Dirección de dependencia: `PaperStatsFlow → motores → resultados técnicos → deliverables → representación`. Deliverables nunca invoca motores.
  - El entregable inicial no recomienda el siguiente análisis: cierra con "Los datos están listos. ¿Qué análisis deseas realizar?" y espera la decisión del usuario. `DatasetStatisticalAnalyzer`, `MissingDataPipeline`, `ImputationSelector`, `ReportGenerator`, `hypothesis.py` y motores EDA intactos.
  - Suite: 378 tests (pre-Phase 12; tras la Fase 12: 394).

- **Fase 12 — Orden "Informe" y renderizado PDF de entregables**: completada.
  - La orden "Informe" es un comando de PRESENTACIÓN/EXPORTACIÓN (no un análisis): `PaperStatsFlow.informe()` genera el PDF del **último** `Deliverable` generado (rastreado en `_last_deliverable` por los métodos `entregable_*`), sin recalcular estadísticas, sin seleccionar métodos ni ejecutar nuevos análisis, y sin alterar la máquina de estados.
  - Nuevo subpaquete `src/deliverables/renderers/`: dispatcher `render(deliverable, formato="markdown"|"pdf", output_path=None)` (formato inválido → `ValueError`), `markdown.py` (lógica extraída de `render_markdown`, que ahora delega) y `pdf.py` (reportlab). El PDF se genera DIRECTAMENTE del modelo neutral `Deliverable`, no del Markdown, consumiendo las cadenas verbatim de los items.
  - Tipografía: registro de las 4 variantes TTF de Times New Roman (`/usr/share/fonts/truetype/msttcorefonts/`) con fallback a la familia Type1 `Times-Roman` de reportlab (métricamente idéntica); todos los estilos, tablas y pie de página usan esa familia. Jerarquía: título (Times-Bold 18 centrado) → archivo (Times-Italic 10) → regla → secciones (Times-Bold 13) → cuerpo (Times 10.5) → bullets → tablas (Times 9-10, cabecera negrita sobre gris, grid, `repeatRows=1`, celdas `Paragraph`, `colWidths` proporcionales al ancho útil A4−4 cm sin desborde) → hallazgos (negrita + detalle itálico) → cierre (Times-Bold 11); pie con número de página.
  - `DeliverableGenerator.render_markdown()` y `.render_pdf()` pasan a ser wrappers que delegan en `renderers`. La capa de builders y todos los motores quedan intactos.
  - Ruta por defecto del informe: `outputs/reports/<dataset>_informe_<marca>.pdf`. Sin entregable previo → `ValueError` ("ejecute primero entregable_inicial/entregable_missing/entregable_analisis").
  - Dependencia añadida: `reportlab>=4.0.0` en `requirements.txt`. `tests/test_deliverables_pdf.py` (16 tests). Suite: 394 tests.

---

## 2. PENDIENTE / PLAN FUTURO

> IMPORTANTE: no implementar todavía. Las siguientes capacidades están planificadas pero fuera del alcance actual.

### 2.1 Modelos estadísticos (`src/models/`)
- Regresión lineal y múltiple.
- Regresión logística.
- Regresión Poisson y binomial negativa.
- Modelos lineales generalizados (GLM).
- Modelos mixtos.
- Análisis de supervivencia (Kaplan-Meier, modelo de Cox).
- Series de tiempo (ARIMA, ARCH, GARCH, EGARCH).
- Clustering, PCA, FAMD, análisis factorial, reducción de dimensión.
- Modelos de clasificación y métodos de Machine Learning.

### 2.2 Pruebas estadísticas adicionales
> Implementado y probado en la Fase 9 (E9-C): `src/analysis/hypothesis.py` + `tests/test_hypothesis_additional.py` (51 tests). Ver §1.
- Kolmogorov-Smirnov (1 y 2 muestras), Lilliefors, Bartlett, Breusch-Pagan, White.
- Durbin-Watson, Breusch-Godfrey, RESET.
- Chi-cuadrado (independencia y bondad de ajuste), Tukey HSD, permutaciones.
- Integrado en `DatasetStatisticalAnalyzer` (Fase 9, E9-D): Lilliefors, Bartlett (bajo normalidad), Tukey HSD (post hoc tras ANOVA significativa), chi-cuadrado (modo exploratorio) y permutaciones (rama no paramétrica de 2 grupos). KS 1/2 muestras y los diagnósticos de regresión (BP, White, DW, BG, RESET) permanecen disponibles como API directa pero no integrados: los diagnósticos de regresión requieren un modelo lineal ajustado (corresponderán a `src/models/`).

### 2.3 Datos faltantes — extensiones futuras
- E1–E7 están implementados y probados (ver §1, Fases 4–8).
- Posibles extensiones (no implementar aún): análisis formal de sensibilidad de la imputación, indicador de ausencia como característica, más métodos (p. ej. bayesianos), validación cruzada de E4 más robusta. Pendiente documentado: el hallazgo cosmético de `generate_missing_data_report` (ver §3).

### 2.4 Estadística bayesiana
- Priors (conjugadas, no informativas, débilmente informativas), posterior, intervalos creíbles, predicción posterior, MCMC.
- Solo si se justifica la elección sobre la aproximación frecuentista.

### 2.5 Riesgo y actuaría
- Frecuencia/severidad, distribuciones de pérdidas, VaR, Expected Shortfall, credibilidad, tablas de mortalidad, series financieras y volatilidad.

### 2.6 Ollama (IA local)
- El proyecto ya está preparado: `BaseLLMClient` es una capa de abstracción desacoplada del proveedor.
- Implementar `OllamaLLMClient` sobre `BaseLLMClient` cuando se decida.
- No acoplar a un único proveedor; no incluir claves API; no enviar información privada a servicios externos por defecto.

### 2.7 Streamlit (interfaz)
- Interfaz para subir PDFs/CSV/Excel, seleccionar análisis, ejecutar, visualizar resultados y descargar informes/gráficos/código reproducible.
- Construir solo después de que el núcleo esté correctamente consolidado.

### 2.8 Reproducibilidad
- Carpeta `notebooks/` con ejemplos de análisis reproducible.
- Scripts de ejemplo y documentación de uso.

---

## 3. HISTORIAL DE DECISIONES

- **Optimización de contexto (2026-08)**: AGENTS.md condensado a reglas permanentes; el conocimiento metodológico completo se movió a `docs/metodologia.md`; estructura a `docs/project_map.md`; plan futuro a `docs/roadmap.md`. Estrategia de contexto por capas (localizar antes de leer, leer rangos necesarios, módulos relacionados, tests como especificación).
- **Refactor de `src/analysis/hypothesis.py` (2026-08)**: la construcción repetida del dict de resultados se centralizó en el helper interno `_build_test_result`. Sin cambios de API ni de valores calculados (39 tests).
- **Exportación de `DatasetStatisticalAnalyzer` (2026-08)**: se agregó a `__all__` en `src/analysis/__init__.py`.
- **Dependencias (2026-08)**: se agregaron `statsmodels>=0.14.2` y `scikit-learn>=1.5.0` a `requirements.txt` e instaladas en `.venv`.
- **Fases E1–E3 (2026-08)**: detección, diagnóstico y métodos de imputación implementados; pesos y puertas duras definidos; formato de reportes JSON-serializable y reproducible.
- **E4 (2026-08)**: evaluación artificial con MCAR/MAR; la evidencia se incorpora a E5 como `0.5·perfil + 0.5·evidencia E4`; advertencia explícita cuando E4 se ejecuta sobre muestra pequeña (casos completos).
- **E5 (2026-08)**: `DatasetProfile` creado en `src/analysis/profile.py` (no existía previamente); nueva puerta estructural: `regresion` queda excluida si no existe columna numérica con `missing_count == 0` (hallazgo de la revisión real sobre `Drug Price.xlsx`).
- **E6 (2026-08)**: 7 comprobaciones de validación; veredicto "Revisar" solo si hay error (faltantes residuales o valores imposibles); umbrales configurables documentados.
- **E7 (2026-08)**: pipeline integrado con imputación **opt-in**; `ValidationRevisionError` con `strict=True` para impedir continuar silenciosamente tras "Revisar"; integración con `DatasetStatisticalAnalyzer` (sección `missing_data`, sin imputar) y `generate_missing_data_report`.
- **Bug de importación circular corregido (2026-08)**: `import src.missing_data` como primer import en un proceso limpio fallaba (ciclo `selector → analysis.profile → analysis/__init__ → dataset_analyzer → pipeline → selector`). Corregido con importación perezosa del pipeline dentro de `DatasetStatisticalAnalyzer.analyze()`. Tests de regresión en `tests/test_import_regression.py` (3). Sin cambios de comportamiento.
- **Hallazgo cosmético PENDIENTE (2026-08)**: `ReportGenerator.generate_missing_data_report` muestra "No evaluado" como mecanismo porque `MechanismAssessment` no tiene campo `mechanism` (la conclusión está en `evidence`). No corregido aún; pendiente de revisión.
- **E9-C (2026-08)**: 12 pruebas estadísticas adicionales del §2.2 en `src/analysis/hypothesis.py` (Kolmo-gorov-Smirnov 1/2 muestras, Lilliefors, Bartlett, Breusch-Pagan, White, Durbin-Watson, Breusch-Godfrey, RESET, Chi-cuadrado, Tukey HSD, permutaciones) + `tests/test_hypothesis_additional.py` (51 tests). Extensiones de esquema documentadas: `f_statistic`/`f_p_value` (BP, White, BG) y `pairwise_comparisons` (Tukey). Contrato DW: sin p-valor exacto, decisión por regla práctica (DW<1.5 o DW>2.5). Permutaciones con corrección +1 y `seed` reproducible. No integradas aún en `DatasetStatisticalAnalyzer`.
- **Revisión de E9-C (2026-08)**: se corrigió la consistencia de los 4 diagnósticos de regresión con el helper `_prepare_diagnostic_inputs` (acepta `exog` 1D/2D y elimina filas con NaN/inf conjuntamente); guardas de tamaño mínimo de la regresión auxiliar (BP: n≥k+2; White: n≥k+k(k+1)/2+2); validaciones claras en chi-cuadrado (sin marginales nulas, sin celdas negativas, esperados estrictamente positivos). Sin cambios de contrato de las 10 pruebas originales. Suite: 361 tests.
- **E9-D (2026-08)**: integración en `DatasetStatisticalAnalyzer` de las 5 pruebas con contexto claro. Decisiones de diseño: (1) la decisión paramétrica/no paramétrica existente NO se toca — las pruebas nuevas son complementarias o post hoc; (2) Bartlett solo bajo `is_normal_contextual` (asume normalidad); (3) Tukey solo tras ANOVA de un factor con H0 rechazada (homocedasticidad garantiza su validez); (4) chi-cuadrado en modo exploratorio (mismo guard que correlaciones) con omisión diagnóstica de pares sin tabla válida; (5) permutaciones con `seed=42` fija para reproducibilidad; (6) `_analyze_group_comparison` retorna `additional_executed_tests` (clave aditiva) para no romper `executed_test`; (7) las pruebas nuevas van en `try/except ValueError` con diagnóstico de omisión para no introducir rutas de crash. Suite: 361 tests.
- **P-FLOW (2026-08)**: el primer punto de entrada del sistema pasa a ser `PaperStatsFlow.diagnose()`, no `DatasetStatisticalAnalyzer.analyze()` (que generaba recomendaciones automáticas de correlación sobre datos incompletos). Decisiones de diseño: (1) `diagnose()` es estrictamente diagnóstica — perfil/QC, E1–E3 y recomendación E5, sin EDA ni inferencia; (2) la imputación exige decisión explícita del usuario (`method_override` o `accept_recommendation=True`); (3) `MissingDataPipeline.run(impute=False)` ahora produce `selection_report` (E5 sin imputar) — **cambio de contrato intencional**: `tests/test_pipeline.py::test_con_faltantes_impute_false_no_imputa` se actualizó para afirmar `selection_report is not None`; (4) `method_override` opcional en `run()` (validado contra el registro, prevalece sobre la recomendación, registrado en `reproduction`); (5) trazabilidad de la recomendación vía `all_scores`/`components`/`reasons`/`caveats` y síntesis por variable con `decision ∈ {metodo_unico, comparar_alternativas, sin_recomendacion}` (umbral `SCORE_GAP_SENSITIVITY_THRESHOLD` reutilizado, no duplicado); (6) `DatasetStatisticalAnalyzer` no se tocó. Suite: 361 tests.

---

## 4. NOTA HISTÓRICA — PRIMERA FASE (COMPLETADA)

La instrucción original de "primera fase" pedía: analizar la carpeta actual, identificar archivos/carpetas/entorno virtual/código previo, no eliminar nada, proponer la arquitectura definitiva y explicar integraciones futuras (Ollama y Streamlit), deteniéndose a esperar aprobación antes de implementar la aplicación completa.

Ese proceso se completó y la arquitectura fue aprobada. Esta sección se conserva únicamente como historial.
