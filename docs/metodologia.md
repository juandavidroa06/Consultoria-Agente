# METODOLOGÍA ESTADÍSTICA — PAPERSTATS

Documento on-demand con el inventario metodológico completo. Se consulta cuando el problema requiere decisión metodológica, análisis de artículos o análisis de datos. Contiene el conocimiento estadístico que en versiones anteriores vivía en AGENTS.md; se conserva íntegro para no perder capacidad. La fuente de verdad de la implementación es el código en `src/`.

---

## 1. ANÁLISIS DE ARTÍCULOS CIENTÍFICOS

Cuando el usuario proporcione un artículo científico, PDF o documento académico, identifica y analiza:

- Título.
- Autores.
- Año.
- Revista.
- Objetivo.
- Pregunta de investigación.
- Hipótesis.
- Población.
- Muestra.
- Diseño del estudio.
- Método de muestreo.
- Variables.
- Métodos estadísticos.
- Modelos estadísticos.
- Pruebas de hipótesis.
- Intervalos de confianza.
- Medidas de efecto.
- Resultados.
- Software utilizado.
- Paquetes o librerías.
- Limitaciones.
- Conclusiones.

---

## 2. ANÁLISIS DE VARIABLES

Identifica y clasifica las variables como:

- Cuantitativas continuas.
- Cuantitativas discretas.
- Cualitativas nominales.
- Cualitativas ordinales.
- Binarias.
- Variables de tiempo.
- Variables dependientes.
- Variables independientes.
- Covariables.

Explica el papel de cada variable dentro del modelo o análisis.

---

## 3. METODOLOGÍA ESTADÍSTICA

Para cada método estadístico utilizado en un artículo, explica:

- Nombre.
- Objetivo.
- Tipo de problema que resuelve.
- Tipo de variables requeridas.
- Supuestos.
- Hipótesis, cuando corresponda.
- Estadístico utilizado.
- Fórmula matemática relevante.
- Interpretación.
- Razón por la que puede ser apropiado.
- Posibles alternativas.
- Limitaciones.
- Implementación en Python.
- Implementación en R.

No debes recomendar métodos únicamente porque sean posibles. Debes justificar estadísticamente su utilización.

---

## 4. PRUEBAS DE HIPÓTESIS

Cuando identifiques una prueba estadística, analiza:

- Hipótesis nula.
- Hipótesis alternativa.
- Nivel de significancia.
- Estadístico de prueba.
- Distribución del estadístico.
- p-value.
- Región crítica cuando sea relevante.
- Decisión.
- Interpretación.

Cuando sea pertinente considera pruebas como:

- Shapiro-Wilk.
- Kolmogorov-Smirnov.
- Lilliefors.
- Mann-Whitney.
- Wilcoxon.
- Levene.
- Bartlett.
- Breusch-Pagan.
- White.
- Durbin-Watson.
- Breusch-Godfrey.
- RESET.
- Chi-cuadrado.
- t de Student.
- ANOVA.
- Tukey.
- Permutaciones.

No utilices automáticamente todas estas pruebas. Selecciona únicamente las apropiadas al problema.

Implementadas actualmente en `src/analysis/hypothesis.py` (22): Shapiro-Wilk, Levene, t de Student (1 muestra / independientes / pareadas), Wilcoxon (1 muestra / pareado), Mann-Whitney U, ANOVA de un factor, ANOVA de Welch, Kruskal-Wallis H, Kolmogorov-Smirnov (1 y 2 muestras), Lilliefors, Bartlett, Breusch-Pagan, White, Durbin-Watson, Breusch-Godfrey, RESET de Ramsey, Chi-cuadrado (independencia y bondad de ajuste), Tukey HSD y permutaciones.

---

## 5. SUPUESTOS

Evalúa los supuestos relevantes de cada modelo. Entre ellos:

- Normalidad.
- Independencia.
- Homocedasticidad.
- Linealidad.
- Ausencia de multicolinealidad.
- Independencia de errores.
- Estacionariedad.
- Proporcionalidad de riesgos.
- Ausencia de separación perfecta.
- Representatividad.

Cuando sea apropiado, recomienda diagnósticos o pruebas para comprobarlos.

---

## 6. MODELOS ESTADÍSTICOS

El agente debe reconocer y trabajar cuando sea pertinente con:

- Regresión lineal.
- Regresión múltiple.
- Regresión logística.
- Regresión Poisson.
- Regresión binomial negativa.
- Modelos lineales generalizados.
- Modelos mixtos.
- Análisis de supervivencia.
- Kaplan-Meier.
- Modelo de Cox.
- ARIMA.
- ARCH.
- GARCH.
- EGARCH.
- Modelos bayesianos.
- Clustering.
- PCA.
- FAMD.
- Análisis factorial.
- Métodos de reducción de dimensión.
- Modelos de clasificación.
- Métodos de Machine Learning.

---

## 7. ESTADÍSTICA BAYESIANA

Cuando corresponda, trabaja con:

- Distribuciones previas.
- Verosimilitud.
- Distribución posterior.
- Priors conjugadas.
- Priors no informativas.
- Priors débilmente informativas.
- Inferencia posterior.
- Intervalos creíbles.
- Predicción posterior.
- Simulación Monte Carlo.
- MCMC.

Explica claramente la diferencia entre inferencia frecuentista e inferencia bayesiana.

No utilices métodos bayesianos sin justificar su elección.

---

## 8. RIESGO Y ACTUARÍA

Cuando el problema corresponda a riesgo o actuaría, considera:

- Riesgo financiero.
- Riesgo actuarial.
- Frecuencia.
- Severidad.
- Distribuciones de pérdidas.
- VaR.
- Expected Shortfall.
- Modelos de frecuencia.
- Modelos de severidad.
- Teoría de credibilidad.
- Supervivencia.
- Tablas de mortalidad.
- Modelos de vida.
- Modelos de riesgo.
- Series financieras.
- Volatilidad.
- Dependencia.

Explica los fundamentos matemáticos cuando sean necesarios.

---

## 9. ANÁLISIS EXPLORATORIO DE DATOS (EDA)

Cuando el usuario proporcione un CSV o Excel:

1. Lee los datos.
2. Identifica dimensiones.
3. Identifica tipos de variables.
4. Revisa valores faltantes.
5. Revisa duplicados.
6. Identifica posibles inconsistencias.
7. Detecta posibles valores atípicos.
8. Calcula estadísticas descriptivas.
9. Analiza variables numéricas.
10. Analiza variables categóricas.
11. Analiza relaciones entre variables.
12. Genera visualizaciones apropiadas.

Nunca modifiques los datos originales.

---

## 10. DATOS FALTANTES E IMPUTACIÓN (E1–E7)

Módulo `src/missing_data/`, orquestado por `MissingDataPipeline`. La imputación es **opt-in**: el comportamiento predeterminado (`impute=False`) solo detecta y diagnostica; nunca modifica los datos ni imputa automáticamente.

### E1 — Detección (MCAR/MAR/MNAR, base)
`MissingDataDetector.detect(df)` produce `MissingReport`: total y porcentaje de faltantes, grado global, casos completos, variables con/sin faltantes, distribuciones y **placeholders candidatos** (representaciones no-NaN de ausencia: vacíos, "N/A", "?", "-", etc.), convertibles con `convert_placeholders_to_na`. No se modifica el DataFrame de entrada.

### E2 — Diagnóstico del mecanismo de ausencia
`MissingDataDiagnostics.diagnose(df)` produce `MissingnessDiagnosticsReport`. Compara el patrón de ausencia (indicador 0/1) de cada variable con faltantes contra las demás:

- Variable numérica → **Mann-Whitney U** (independencia del indicador de ausencia).
- Variable categórica → **chi-cuadrado de independencia** (Fisher exacto para muestras pequeñas).

La conclusión sobre el mecanismo es cautelosa:

- **MCAR**: la ausencia de asociaciones significativas no la confirma, solo no la cuestiona.
- **MAR**: no puede demostrarse ni descartarse únicamente con los datos observados.
- **MNAR**: no puede inferirse únicamente con los datos observados.

La decisión final de imputación/exclusión depende del objetivo y contexto del estudio, no solo de estas pruebas. La selección E5 penaliza la robustez cuando hay evidencia contra MCAR.

### E3 — Métodos de imputación y registro
`src/missing_data/methods.py` define `ImputationMethod` (ABC con `fit`/`transform`/`impute`; `impute` devuelve una copia, nunca muta) y `MethodCapabilities` (soporte numérico/categórico, estructura temporal, dependencia de otras columnas). Métodos implementados (registrados por nombre en `default_registry`):

- Univariados: **media** (`MeanImputation`), **mediana** (`MedianImputation`), **moda** (`ModeImputation`), **constante** (`ConstantImputation`).
- Multivariados: **kNN** (`KNNImputation`), **iterativo** (`IterativeImputation`), **MICE** (`MICEImputation`), **regresión** (`RegressionImputation`).
- Temporales: **interpolación lineal** (`LinearInterpolationImputation`), **LOCF** (`LOCFImputation`).

`candidates_for(df, temporal)` devuelve los métodos compatibles con el DataFrame.

### E4 — Evaluación artificial (ADVERTENCIA: muestra pequeña)
`induce_missing` oculta celdas completas al azar (MCAR) o condicionadas a un predictor numérico completo (MAR). `ArtificialMissingnessEvaluator.evaluate(df, methods, fraction, mechanism, predictor)` imputa con cada método y mide **RMSE / MAE** (numéricas) y **accuracy** (categóricas), con `n_repeats` repeticiones y ranking agregado.

> [!WARNING]
> En el pipeline E7, E4 se ejecuta sobre los **casos completos** (`df.dropna()`), con umbral mínimo `MIN_COMPLETE_CASES_FOR_EVALUATION = 5`. Si la muestra de casos completos es pequeña (p. ej. 15 filas en `Drug Price.xlsx`), la evidencia de E4 es **exploratoria**: los rankings deben interpretarse con cautela. E5 lo advierte explícitamente cuando incorpora evidencia de E4, y la robustez se calcula como `0.5·perfil + 0.5·evidencia E4`.

### E5 — Selección de métodos
`build_dataset_profile` (`src/analysis/profile.py`) describe tamaño muestral, tipos de variable, estructura temporal e identificadores. `ImputationSelector.select(profile, missing_report, diagnostics, evaluation)` puntúa cada método con pesos `DEFAULT_WEIGHTS` (type_fit 0.20, missing_pct_fit 0.20, sample_size_fit 0.15, structure_fit 0.10, relationship_exploitation 0.15, robustness 0.15, complexity_cost 0.05) y puertas duras:

- Tipo de variable no soportado.
- Método temporal sin estructura temporal.
- Métodos que requieren otras variables con n < mínimo.
- Variables **identificadoras** (una por fila): no deben imputarse.
- `regresion` sin columna numérica completamente observada (`missing_count == 0`): puerta estructural añadida en E5.

Advertencias: faltantes >50% (imputación especulativa), n reducido, estructura temporal, evidencia E4 exploratoria, gaps de score < `SCORE_GAP_SENSITIVITY_THRESHOLD` (0.02). Produce `ImputationSelectionReport` (recomendación por variable + alternativas + razones + caveats + ranking por grupo).

### E6 — Validación post-imputación
`ImputationValidator.validate(original, imputed)` ejecuta 7 comprobaciones:

1. **Faltantes residuales** (NaN que quedan en el imputado; error).
2. **Preservación de dtype** (warn si cambia).
3. **Valores imposibles** (fuera del dominio observado: rango [min,max] numérico / categorías observadas; error; "dominio desconocido" si no hay observados).
4. **Cambios de distribución** (Kolmogorov-Smirnov dos muestras observados vs. imputados + estadísticas descriptivas; umbral cambio relativo 0.10; warn; mínimo 5 por muestra).
5. **Cambios en correlaciones** (Pearson antes/después; umbral Δ|r| 0.15; warn).
6. **Proporción imputada** (por columna y global; umbral 0.50; warn).
7. **Comparación imputados vs. observados** (sesgo de media relativo 0.10 / moda para categóricas; warn).

**Veredicto**: `"Aceptable"` salvo que haya un error (faltantes residuales o valores imposibles), en cuyo caso `"Revisar"`. Los `warn` no cambian el veredicto por sí solos. Todo es JSON-serializable y reproducible (sin aleatoriedad).

### E7 — Pipeline integrado (imputación opt-in)
`MissingDataPipeline.run(df, *, impute=False, ...)`:

- `impute=False` (default): E1 → E2 → E3; no imputa.
- `impute=True`: E1 → E6 completo con imputación explícita variable a variable según la recomendación de E5, y validación E6.
- E6 `"Aceptable"` → `continued=True`. E6 `"Revisar"` → `continued=False` y, con `strict=True` (default), lanza `ValidationRevisionError` (no continúa silenciosamente).
- Nunca muta el DataFrame original; conserva todos los reportes E1–E6; registra método seleccionado y razones (`applied_methods`) y variables omitidas (`skipped_variables`).

`DatasetStatisticalAnalyzer.analyze()` integra el pipeline con `impute=False` en la sección `missing_data` del resultado. `ReportGenerator.generate_missing_data_report()` produce el informe Markdown.

---

## 11. VISUALIZACIONES

Utiliza las visualizaciones apropiadas según el problema. Entre ellas:

- Histogramas.
- Boxplots.
- Violin plots.
- Gráficos de dispersión.
- Matrices de correlación.
- Gráficos de barras.
- Densidades.
- QQ-plots.
- Series temporales.
- Gráficos de residuos.
- Curvas ROC.
- Curvas de supervivencia.
- Forest plots.
- Gráficos diagnósticos.

Cada gráfico debe tener títulos, etiquetas y una interpretación estadística cuando corresponda.

---

## 12. REPRODUCIBILIDAD

Cuando sea posible reproducir el análisis de un artículo:

1. Identifica los datos disponibles.
2. Identifica qué resultados pueden reproducirse.
3. Identifica qué información falta.
4. Genera código reproducible.
5. Ejecuta el análisis cuando sea posible.
6. Compara los resultados con los publicados.
7. Explica cualquier diferencia.

El código debe poder ejecutarse nuevamente por otra persona.

---

## 13. PYTHON Y R

Utiliza Python como lenguaje principal. Considera:

- pandas
- numpy
- scipy
- statsmodels
- scikit-learn
- matplotlib
- seaborn cuando sea apropiado
- openpyxl

Utiliza R cuando sea especialmente conveniente. Considera:

- tidyverse
- ggplot2
- stats
- car
- lmtest
- forecast
- survival
- MASS
- otras librerías justificadas

---

## 14. INFORMES

Cuando el usuario solicite un informe, genera una estructura académica que pueda incluir:

1. Descripción del problema.
2. Datos.
3. Metodología.
4. Análisis exploratorio.
5. Métodos estadísticos.
6. Resultados.
7. Diagnóstico de supuestos.
8. Visualizaciones.
9. Interpretación.
10. Conclusiones.
11. Limitaciones.
12. Código reproducible.
