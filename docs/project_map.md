# MAPA DEL PROYECTO — PAPERSTATS

Documento on-demand de referencia estructural. Se consulta cuando se necesita ubicar módulos, API pública o tests. La fuente de verdad del comportamiento es el código en `src/`.

## 1. ESTRUCTURA GENERAL

```
consultoría/
│
├── AGENTS.md                 # Reglas permanentes del agente (condensado)
├── README.md                 # Documentación general y guía de uso
├── requirements.txt          # Dependencias de Python
├── .gitignore                # Filtro de archivos
│
├── articles/                 # Artículos PDF de entrada
├── data/
│   ├── raw/                  # Datos originales (no modificar)
│   └── processed/            # Datos procesados
├── docs/                     # Documentación on-demand (este mapa, metodología, roadmap)
│
├── src/                      # Código fuente modular
│   ├── article/              # Ingesta y análisis de artículos científicos
│   ├── data/                 # Carga y validación de datos
│   ├── analysis/             # EDA, pruebas de hipótesis, perfil estructural y analizador de datasets
│   ├── missing_data/         # Datos faltantes: detección → diagnóstico → imputación → validación (E1–E7)
│   ├── orchestration/        # P-FLOW: orquestador de alto nivel del flujo principal
│   ├── deliverables/         # Capa de presentación: entregables de usuario (modelo neutral + builders)
│   ├── models/               # (futuro) Modelos estadísticos
│   ├── visualization/        # Gráficos estadísticos
│   ├── llm/                  # Capa de abstracción LLM (Ollama pendiente)
│   ├── reports/              # Generación de informes Markdown
│   └── utils/                # Utilidades (logger)
│
├── outputs/
│   ├── figures/              # Gráficos generados
│   ├── tables/               # Tablas generadas
│   └── reports/              # Informes Markdown generados
│
├── tests/                    # Suite de pruebas unitarias (405 tests)
└── notebooks/                # (futuro) Notebooks de análisis reproducible
```

Nota: `PaperStat/` es un directorio vacío heredado de un nombre anterior del proyecto. No eliminarlo sin autorización explícita.

## 2. MAPA DE MÓDULOS → API PÚBLICA → TESTS

| Módulo | Clases / Funciones públicas | Archivo de tests |
|---|---|---|
| `src/article/parser.py` | `ArticleParser(file_path)` → `.parse()` → dict(text, pages, num_pages, file_name, file_path). Formatos: .pdf, .txt, .md | `tests/test_parser.py` (3) |
| `src/article/extractor.py` | `ArticleExtractor(llm_client=None)` → `.extract(article_data)` → metadatos estructurados (19+ puntos) | `tests/test_extractor.py` (1) |
| `src/article/analyzer.py` | `StatisticalMethodologyAnalyzer(llm_client=None)` → `.analyze(metadata, full_text)` → tests/modelos/supuestos detectados + `variable_classification` | `tests/test_analyzer.py` (1) |
| `src/data/loader.py` | `load_data(file_path, sheet_name=0, **kwargs)` → pd.DataFrame (CSV/Excel) | `tests/test_data_loader.py` (4) |
| `src/data/validator.py` | `DataValidator` (métodos estáticos): `identify_variable_types`, `detect_missing_values`, `detect_duplicates`, `summarize_data_quality` | `tests/test_data_validator.py` (4) |
| `src/analysis/eda.py` | `describe_numerical`, `describe_categorical`, `detect_outliers_iqr`, `calculate_correlation_matrix` | `tests/test_eda.py` (4) |
| `src/analysis/hypothesis.py` | `shapiro_wilk_test`, `levene_test`, `t_test_1samp`, `t_test_ind`, `t_test_rel`, `wilcoxon_signed_rank`, `mann_whitney_test`, `anova_one_way`, `welch_anova`, `kruskal_wallis_test`, `kolmogorov_smirnov_1samp_test`, `kolmogorov_smirnov_2samp_test`, `lilliefors_test`, `bartlett_test`, `breusch_pagan_test`, `white_test`, `durbin_watson_test`, `breusch_godfrey_test`, `reset_test`, `chi_square_test`, `tukey_hsd_test`, `permutation_test`. Helpers internos: `_clean_sample`, `_build_test_result` | `tests/test_hypothesis.py` (10) + `tests/test_hypothesis_additional.py` (51) |
| `src/analysis/profile.py` | `DatasetProfile`, `build_dataset_profile(df, target, temporal, datetime_columns, identifier_columns)` → perfil estructural (n, tipos de variable, estructura temporal, identificadores). Lo consume E5 | `tests/test_imputation_selector.py` (32) |
| `src/analysis/dataset_analyzer.py` | `DatasetStatisticalAnalyzer(data)` → `.analyze(target_col, group_col, paired_col, popmean, alpha)` → dict integral **+ sección `missing_data`** (E1–E3 sin imputar). Integra E9-D: Lilliefors (normalidad), Bartlett (bajo normalidad), Tukey HSD (post hoc), chi-cuadrado (modo exploratorio) y permutaciones (rama no paramétrica) | `tests/test_dataset_analyzer.py` (6) + `tests/test_dataset_analyzer_e9d.py` (12) |
| `src/visualization/plots.py` | `plot_histogram`, `plot_boxplot`, `plot_scatter`, `plot_qq`, `plot_correlation_matrix` → plt.Figure | `tests/test_plots.py` (5) |
| `src/llm/base.py` | `BaseLLMClient` (ABC: `generate`, `extract_metadata`, `analyze_methodology`) + `RuleBasedLLMClient` (heurístico, determinista) | — |
| `src/reports/generator.py` | `ReportGenerator.generate(metadata, analysis, output_path)` (informe de artículo), `generate_dataset_report(analysis_results, output_path)` y `generate_missing_data_report(pipeline_result, output_path)` (informe de datos faltantes E1–E6) | `tests/test_generator.py` (1) |
| `src/utils/logger.py` | `setup_logger(name, log_level)` | — |
| `src/missing_data/detection.py` | `MissingDataDetector` → `.detect(df)` → `MissingReport`; `MissingVariableInfo`; `convert_placeholders_to_na` (E1) | `tests/test_missing_data_detector.py` (29) |
| `src/missing_data/diagnostics.py` | `MissingDataDiagnostics` → `.diagnose(df)` → `MissingnessDiagnosticsReport` (asociaciones Mann-Whitney/chi-cuadrado, `MechanismAssessment` MCAR/MAR/MNAR) (E2) | `tests/test_missing_data_diagnostics.py` (30) |
| `src/missing_data/methods.py` | `ImputationMethod` (ABC: `fit`/`transform`/`impute`) + `MeanImputation`, `MedianImputation`, `ModeImputation`, `ConstantImputation`, `KNNImputation`, `IterativeImputation`, `MICEImputation`, `RegressionImputation`, `LinearInterpolationImputation`, `LOCFImputation` (E3) | `tests/test_imputation_methods.py` (45) |
| `src/missing_data/registry.py` | `ImputationRegistry`, `default_registry`, `register`, `get`, `names`, `candidates_for`, `summary` (E3) | `tests/test_imputation_registry.py` (17) |
| `src/missing_data/evaluation.py` | `induce_missing`, `ArtificialMissingnessEvaluator` → `.evaluate(df, columns, methods, fraction, mechanism, predictor)` → `ImputationEvaluationReport` (RMSE/MAE/accuracy + ranking) (E4) | `tests/test_imputation_evaluation.py` (25) |
| `src/missing_data/selector.py` | `ImputationSelector` → `.select(profile, missing_report, diagnostics, evaluation)` → `ImputationSelectionReport`; `DEFAULT_WEIGHTS`, `MethodScore`, `VariableRecommendation` (E5) | `tests/test_imputation_selector.py` (32) |
| `src/missing_data/validation.py` | `ImputationValidator` → `.validate(original, imputed)` → `ImputationValidationReport` (7 checks, verdict "Aceptable"/"Revisar") (E6) | `tests/test_imputation_validation.py` (18) |
| `src/missing_data/pipeline.py` | `MissingDataPipeline` → `.run(df, impute=False, ..., method_override=None)` → `MissingDataPipelineResult`; con `impute=False` recomienda (E5) sin imputar; con `impute=True` aplica la decisión del usuario (`method_override`) + E4–E6; `ValidationRevisionError`; `run_pipeline` (E7) | `tests/test_pipeline.py` (19) |
| `src/orchestration/flow.py` | `PaperStatsFlow(data)` → `.diagnose()` (perfil/QC + E1–E3 + recomendación E5, sin EDA/inferencia) → `.imputar(method_override\|accept_recommendation)` → `.analizar(**kwargs)` (delega en `DatasetStatisticalAnalyzer` solo sobre datos preparados); `generar_informe_missing`; `state`/`imputed_df`; métodos de presentación aditivos `entregable_inicial()`/`entregable_missing()`/`entregable_analisis()` (rastrean `_last_deliverable`); orden de exportación `informe(output_path, formato="pdf")` (+ `_ruta_informe_default()` → `outputs/reports/<dataset>_informe_<marca>.pdf`) (P-FLOW) | `tests/test_orchestration_flow.py` (17) + `tests/test_deliverables_pdf.py` |
| `src/deliverables/generator.py` | Modelo neutral `Deliverable`→`Section`→`Item` (kind text/bullets/table/hallazgo); `DeliverableGenerator.build_inicial(diagnose_result, eda_results)` / `.build_missing(diagnose_result)` / `.build_analisis(pregunta, resultado)`; wrappers `render_markdown(deliverable)` / `render_pdf(deliverable, output_path)` que delegan en `renderers` | `tests/test_deliverables.py` (17) + `tests/test_deliverables_pdf.py` |
| `src/deliverables/quality.py` | `build_quality_secciones(diagnose_result)` — control de calidad y estado de los datos | `tests/test_deliverables.py` |
| `src/deliverables/eda.py` | `build_eda_secciones(eda_results)` — descriptivos, frecuencias y hallazgos exploratorios (sin inferencia ni recomendaciones) | `tests/test_deliverables.py` |
| `src/deliverables/missing.py` | `build_missing_secciones(diagnose_result)` — faltantes, diagnóstico del mecanismo y recomendación de imputación en lenguaje de usuario (sin pesos/scores/umbrales) | `tests/test_deliverables.py` |
| `src/deliverables/analysis.py` | `build_analisis_secciones(pregunta, resultado)` — método, supuestos, resultado e interpretación de un análisis solicitado | `tests/test_deliverables.py` |
| `src/deliverables/renderers/__init__.py` | Dispatcher `render(deliverable, formato="markdown"\|"pdf", output_path=None)`; formato inválido → `ValueError`; `FORMATOS` | `tests/test_deliverables_pdf.py` |
| `src/deliverables/renderers/markdown.py` | `render_markdown(deliverable)` — representación Markdown (lógica extraída de `generator.render_markdown`) | `tests/test_deliverables_pdf.py` |
| `src/deliverables/renderers/pdf.py` | `render_pdf(deliverable, output_path=None)` — PDF reportlab en Times New Roman (TTF con fallback Type1 Times); jerarquía, tablas con `Paragraph`/`colWidths` proporcionales, pie de página; `_build_story`, `_tabla`, `_col_widths`, `_ESTILOS`, `_FUENTES` | `tests/test_deliverables_pdf.py` (16) |

Total de tests: 405.

## 3. API EXPORTADA (src/*/__init__.py)

- `src/`: versión del paquete `__version__ = "0.1.0"`.
- `src.article`: `ArticleParser`, `ArticleExtractor`, `StatisticalMethodologyAnalyzer`.
- `src.data`: `load_data`, `DataValidator`.
- `src.analysis`: `describe_numerical`, `describe_categorical`, `detect_outliers_iqr`, `calculate_correlation_matrix`, `shapiro_wilk_test`, `levene_test`, `t_test_1samp`, `t_test_ind`, `t_test_rel`, `wilcoxon_signed_rank`, `mann_whitney_test`, `anova_one_way`, `welch_anova`, `kruskal_wallis_test`, `DatasetStatisticalAnalyzer`, `DatasetProfile`, `build_dataset_profile`, `NUMERIC_TYPES`, `CATEGORICAL_TYPES`, `DATETIME_TYPE`.
- `src.visualization`: `plot_histogram`, `plot_boxplot`, `plot_scatter`, `plot_qq`, `plot_correlation_matrix`.
- `src.llm`: `BaseLLMClient`, `RuleBasedLLMClient`.
- `src.reports`: `ReportGenerator`.
- `src.utils`: `setup_logger`.
- `src.missing_data`: `MissingDataDetector`, `MissingReport`, `MissingVariableInfo`, `MissingDataDiagnostics`, `MissingnessDiagnosticsReport`, `MissingnessAssociation`, `MechanismAssessment`, `ImputationMethod`, `MethodCapabilities`, `MeanImputation`, `MedianImputation`, `ModeImputation`, `ConstantImputation`, `KNNImputation`, `IterativeImputation`, `MICEImputation`, `RegressionImputation`, `LinearInterpolationImputation`, `LOCFImputation`, `ImputationRegistry`, `default_registry`, `register`, `get`, `names`, `summary`, `candidates_for`, `convert_placeholders_to_na`, `induce_missing`, `ArtificialMissingnessEvaluator`, `MethodEvaluation`, `ImputationEvaluationReport`, `ImputationSelector`, `ImputationSelectionReport`, `VariableRecommendation`, `MethodScore`, `DEFAULT_WEIGHTS`, `ImputationValidator`, `ImputationValidationReport`, `ValidationCheck`, `MissingDataPipeline`, `MissingDataPipelineResult`, `ValidationRevisionError`, `MIN_COMPLETE_CASES_FOR_EVALUATION`, `run_pipeline`.

## 4. FORMATO ESTÁNDAR DE RESULTADOS DE PRUEBAS DE HIPÓTESIS

Toda prueba en `src/analysis/hypothesis.py` retorna un dict con la misma estructura (construido por `_build_test_result`):

```
test_name, statistic, p_value, alpha,
null_hypothesis, alt_hypothesis, decision,
reject_h0, interpretation
```

`decision` es "Rechazar H0" / "No rechazar H0" derivado de `reject_h0`. No cambiar este formato: lo consumen los tests y `DatasetStatisticalAnalyzer`.

## 5. IMPLEMENTADO vs PENDIENTE

**Implementado y probado (405 tests):** ingestión de artículos (PDF/TXT/MD), extracción heurística de 17+ puntos, análisis metodológico, carga CSV/Excel, validación de calidad, EDA, 22 pruebas de hipótesis (10 originales + 12 adicionales del §2.2), analizador autónomo de datasets (con sección `missing_data`), perfil estructural (`DatasetProfile`), 5 visualizaciones, informes Markdown (artículo, dataset y datos faltantes), capa de abstracción LLM (solo heurística), el **módulo de datos faltantes E1–E7** (detección, diagnóstico, 10 métodos de imputación, evaluación artificial, selección, validación y pipeline integrado), el **orquestador P-FLOW** (`PaperStatsFlow`: diagnóstico → decisión de imputación → análisis bajo demanda), la **capa de entregables de usuario** (`src/deliverables/`: presentación pura con modelo neutral y representación Markdown desacoplada) y la **orden "Informe"** (`src/deliverables/renderers/`: PDF en Times New Roman generado directamente del modelo neutral, sin recalcular).

**Pendiente (ver `docs/roadmap.md`):** `src/models/`, bayesiano, riesgo/actuaría, series de tiempo, ML, PCA/clustering, Ollama real, Streamlit, notebooks.

## 6. ARQUITECTURA DEL PIPELINE DE DATOS FALTANTES (E1–E7)

El módulo `src/missing_data/` implementa un flujo en etapas orquestado por `MissingDataPipeline` (`src/missing_data/pipeline.py`):

```
LOAD (copia, nunca se muta el original)
  ↓
E1 DETECTION          MissingDataDetector.detect(df) → MissingReport
  ↓
  ├── sin faltantes → status="sin_faltantes", continued=True → EDA/inferencia
  └── con faltantes
        ↓
E2 DIAGNÓSTICO        MissingDataDiagnostics.diagnose(df) → MissingnessDiagnosticsReport
        ↓
E3 CANDIDATOS         candidates_for(df, temporal) → métodos compatibles del registro
        ↓
        ├── impute=False (opt-in, default) → status="con_faltantes", sin E4–E6
        └── impute=True
              ↓
E4 EVALUACIÓN         ArtificialMissingnessEvaluator.evaluate(casos_completos, ...)
              ↓
E5 SELECCIÓN          build_dataset_profile + ImputationSelector.select(...)
              ↓
              IMPUTACIÓN EXPLÍCITA   (variable a variable con el método recomendado)
              ↓
E6 VALIDACIÓN         ImputationValidator.validate(original, imputed)
              ↓
              ├── "Aceptable" → continued=True → EDA/inferencia
              └── "Revisar"   → continued=False; strict=True lanza ValidationRevisionError
```

**Contratos de entrada/salida relevantes**
- `MissingDataPipeline.run(df, *, impute=False, target, temporal, datetime_columns, identifier_columns, evaluation_fraction=0.2, evaluation_mechanism="MCAR", strict=True)` → `MissingDataPipelineResult` con `status` (`sin_faltantes`/`con_faltantes`/`imputado`), `continued`, los reportes E1–E6 disponibles como atributos, `applied_methods` (método seleccionado + reasons/caveats por variable), `skipped_variables`, `imputed_df`, `n_imputed_cells` y `reproduction`.
- `MissingDataPipelineResult.to_dict()` es JSON-serializable (excluye `imputed_df`).
- `ValidationRevisionError` se lanza con `strict=True` (default) cuando E6 devuelve "Revisar"; impide continuar silenciosamente.

**Integración**
- `DatasetStatisticalAnalyzer.analyze()` incluye la clave `missing_data` (estado E1–E3 vía pipeline con `impute=False`; nunca imputa).
- `ReportGenerator.generate_missing_data_report(pipeline_result, output_path)` genera el informe Markdown de datos faltantes.

## 7. BUENAS PRÁCTICAS DE PROGRAMACIÓN

- Código modular y con funciones; nombres descriptivos; manejo de errores; validación de entradas.
- `pathlib` cuando sea apropiado.
- Separar datos, código y resultados (carpetas `data/`, `src/`, `outputs/`).
- Dependencias mantenidas en `requirements.txt`.
- Evitar archivos gigantescos; usar entorno virtual (`.venv`).
- No incluir claves API en el código.
- No modificar los datos originales.
