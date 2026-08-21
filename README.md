# PaperStats — Agente de Consultoría Estadística

PaperStats es un agente especializado en consultoría estadística, investigación científica y ciencia de datos. Su objetivo es ayudar a analizar artículos científicos, conjuntos de datos y problemas estadísticos de manera rigurosa, reproducible y comprensible.

---

## Requisitos

- **Python 3.11+** (probado con **Python 3.14.4** — ver `.venv/pyvenv.cfg`). Recomendado: Python 3.14.
- `pip` >= 23 y `venv` (incluido en la distribución estándar de Python).
- Sistema operativo: Linux / macOS / Windows (comandos abajo para bash; en Windows ver sección de activación: PowerShell `.\.venv\Scripts\Activate.ps1` o CMD ` .venv\Scripts\activate.bat`).

---

## Instalación desde cero en otra máquina

### 1. Obtener el proyecto

**Opción A — Clonar (si dispone de la URL del repositorio remoto):**

```bash
git clone <URL-del-repositorio> consultoria
cd consultoria
```

**Opción B — Carpeta comprimida (si recibió el proyecto como ZIP, sin URL remota):**

```bash
# Descomprima el archivo recibido y entre a la carpeta que contiene requirements.txt y src/
unzip consultoria.zip   # o descompresión manual con el explorador
cd consultoria
```

> En ambos casos debe quedar en la carpeta `consultoria/` que contiene `requirements.txt` y `src/`.

### 2. Crear el entorno virtual

```bash
python3 -m venv .venv
```

Esto crea la carpeta `.venv/` (ignorada por `.gitignore`). No instales paquetes globalmente.

### 3. Activar el entorno virtual

En Linux / macOS:

```bash
source .venv/bin/activate
```

En Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
# Si PowerShell bloquea la ejecución: Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

En Windows CMD:

```cmd
.venv\Scripts\activate.bat
```

Verifica que el prompt muestre `(.venv)`.

### 4. Actualizar pip e instalar dependencias fijadas

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

`requirements.txt` contiene versiones **fijadas con `==`** (ej. `pandas==3.0.5`, `numpy==2.5.1`) obtenidas con `pip freeze` el 2026-08-20 para instalación reproducible. No uses rangos `>=` si buscas reproducibilidad exacta.

### 5. Verificar la instalación

```bash
python --version   # debe mostrar 3.11+ (probado: 3.14.4)
pip list | grep -E "pandas|numpy|scipy|scikit-learn|reportlab"
```

En Windows (sin `grep`):

```powershell
pip list | findstr "pandas numpy scipy scikit-learn reportlab"
# alternativa multiplataforma:
pip show pandas numpy scipy scikit-learn reportlab
```

---

## Ejecución de la suite de tests

Con el entorno activado, desde la raíz del proyecto:

```bash
./.venv/bin/pytest -v
# o, con el entorno activado:
pytest -v
# ejecución resumida:
pytest -q
```

**Resultado esperado:** `394 passed` (sin fallos). Warnings esperados: 2 (deprecation de `seaborn`/`matplotlib` si aparecen, no afectan el resultado). Ver `docs/project_map.md` para el mapeo módulo → tests.

Comando de referencia según `AGENTS.md §7`: `./.venv/bin/pytest -v`.

---

## Ejemplo reproducible del flujo principal (P-FLOW)

El punto de entrada es `PaperStatsFlow.diagnose()` (`src/orchestration/flow.py`). Flujo: `diagnose() → decisión del usuario → imputar() → analizar()` con máquina de estados `sin_diagnostico → sin_faltantes / esperando_decision → datos_preparados / revisar`.

### Ejemplo 1 — Dataset sin faltantes (in-memory)

```python
import pandas as pd
from src.orchestration.flow import PaperStatsFlow

# Dataset mínimo reproducible (sin faltantes)
df = pd.DataFrame({
    "edad": [25, 30, 35, 40, 45],
    "ingreso": [1000, 1500, 2000, 2500, 3000],
    "grupo": ["A", "A", "B", "B", "B"],
})

flow = PaperStatsFlow(df)

# 1) Diagnóstico — estrictamente diagnóstico (perfil/QC + E1-E3 + recomendación E5)
diag = flow.diagnose()
print(diag["estado"])  # -> "sin_faltantes"
print(diag["recomendacion_imputacion"]["resumen"])

# 2) Entregable inicial (descriptivo/exploratorio, sin inferencia)
entregable = flow.entregable_inicial()
print(entregable.titulo)

# 3) Análisis bajo demanda (solo cuando estado es sin_faltantes o datos_preparados)
resultado = flow.analizar(target_col="ingreso", group_col="grupo")
print(resultado["executed_test_results"])  # dict de pruebas ejecutadas según supuestos
# ejemplo: print(list(resultado["executed_test_results"].keys()))
```

**Resultado esperado:** `estado == "sin_faltantes"`, entregable con secciones de calidad y EDA (hallazgos marcados como exploratorios), y `resultado` con clave `executed_test_results` + `missing_data` (E1-E3 sin imputar).

### Ejemplo 2 — Dataset con faltantes (requiere decisión explícita)

```python
import pandas as pd
from src.orchestration.flow import PaperStatsFlow

df = pd.DataFrame({
    "edad": [25, None, 35, 40, 45],
    "ingreso": [1000, 1500, None, 2500, 3000],
    "grupo": ["A", "A", "B", "B", "B"],
})

flow = PaperStatsFlow(df)
diag = flow.diagnose()
print(diag["estado"])  # -> "esperando_decision"
print(diag["recomendacion_imputacion"]["por_variable"].keys())

# La imputación es opt-in: requiere method_override o accept_recommendation=True
imp = flow.imputar(accept_recommendation=True)
print(imp["estado"])              # -> "datos_preparados" o "revisar"
print(imp["validation_verdict"])   # -> "Aceptable" o "Revisar"

if flow.state == "datos_preparados":
    resultado = flow.analizar(target_col="ingreso", group_col="grupo")
    entregable = flow.entregable_analisis("¿Difiere el ingreso por grupo?", resultado)
    # Exportar informe PDF (orden de presentación, sin recalcular)
    ruta_pdf = flow.informe()  # -> outputs/reports/<dataset>_informe_<marca>.pdf
    print(ruta_pdf)
```

**Resultado esperado:** con faltantes, `diagnose()` se detiene en `esperando_decision` y no imputa; `imputar()` exige decisión explícita o lanza `ValueError`; si E6 valida `"Aceptable"` → `datos_preparados` y `analizar()` delega en `DatasetStatisticalAnalyzer`; si E6 es `"Revisar"` → estado `revisar` y `ValidationRevisionError` (con `strict=True`) impide continuar silenciosamente. `informe()` genera PDF en Times New Roman sin recalcular.

### Ejemplo 3 — Con archivo CSV/Excel

```python
from src.orchestration.flow import PaperStatsFlow

flow = PaperStatsFlow("data/raw/Drug Price.xlsx")  # o "data/raw/Trabajadores.xlsx"
diag = flow.diagnose()
print(diag["estado"], diag["dataset"])
```

Los datos originales nunca se modifican; `MissingDataPipeline` trabaja sobre copia.

---

## Estructura del Proyecto (actual)

```
consultoria/
│
├── AGENTS.md                   # Reglas permanentes del agente (condensado)
├── README.md                   # Esta guía — instrucciones reproducibles
├── requirements.txt            # Dependencias fijadas con == (reproducibilidad)
├── .gitignore                  # Excluye .venv/, outputs/, data/raw/, articles/
│
├── articles/                   # Artículos PDF de entrada (no versionados)
├── data/
│   ├── raw/                    # Datos originales (no modificar) — incluye Drug Price.xlsx de ejemplo
│   └── processed/              # Datos procesados
├── docs/                       # Documentación on-demand
│   ├── project_map.md          # Mapa módulos → API pública → tests (fuente estructural)
│   ├── metodologia.md          # Inventario metodológico completo
│   ├── arquitectura_agente.md  # Flujo P-FLOW, estados y guardarraíles
│   └── roadmap.md              # Estado fases 1-12 (394 tests) y plan futuro
│
├── benchmarks/                 # Benchmarks reproducibles (Criterio 3)
│   ├── benchmark_imputation.py # Benchmark media vs knn (time.perf_counter + tracemalloc)
│   └── README.md               # Documentación del benchmark
│
├── src/                        # Código fuente modular
│   ├── __init__.py             # __version__ = "0.1.0"
│   ├── article/                # Ingesta y análisis de artículos PDF/TXT/MD
│   │   ├── parser.py           # ArticleParser
│   │   ├── extractor.py        # ArticleExtractor (19 puntos)
│   │   └── analyzer.py         # StatisticalMethodologyAnalyzer
│   ├── data/                   # Carga y validación
│   │   ├── loader.py           # load_data (CSV/Excel)
│   │   └── validator.py        # DataValidator
│   ├── analysis/               # EDA, pruebas de hipótesis, perfil y analizador
│   │   ├── eda.py              # describe_numerical/categorical, outliers, correlación
│   │   ├── hypothesis.py       # 22 pruebas (Shapiro, Levene, t, Wilcoxon, ANOVA, KS, Lilliefors, Bartlett, BP, White, DW, BG, RESET, chi², Tukey, permutaciones)
│   │   ├── profile.py          # DatasetProfile / build_dataset_profile
│   │   └── dataset_analyzer.py # DatasetStatisticalAnalyzer (sin imputar, sección missing_data)
│   ├── missing_data/           # Datos faltantes E1–E7
│   │   ├── detection.py        # MissingDataDetector (E1)
│   │   ├── diagnostics.py      # MissingDataDiagnostics (E2, MCAR/MAR/MNAR)
│   │   ├── methods.py          # 10 métodos de imputación (E3)
│   │   ├── registry.py         # ImputationRegistry (E3)
│   │   ├── evaluation.py       # ArtificialMissingnessEvaluator (E4)
│   │   ├── selector.py         # ImputationSelector (E5, DEFAULT_WEIGHTS)
│   │   ├── validation.py       # ImputationValidator (E6, 7 checks)
│   │   └── pipeline.py         # MissingDataPipeline, run_pipeline (E7, opt-in, ValidationRevisionError)
│   ├── orchestration/          # P-FLOW — orquestador principal
│   │   └── flow.py             # PaperStatsFlow (diagnose → imputar → analizar, estados sin_diagnostico/sin_faltantes/esperando_decision/datos_preparados/revisar)
│   ├── deliverables/           # Capa de presentación (no recalcula)
│   │   ├── generator.py        # Deliverable→Section→Item, DeliverableGenerator, wrappers render_*
│   │   ├── quality.py          # build_quality_secciones
│   │   ├── eda.py              # build_eda_secciones (hallazgos exploratorios)
│   │   ├── missing.py          # build_missing_secciones
│   │   ├── analysis.py         # build_analisis_secciones
│   │   └── renderers/
│   │       ├── __init__.py     # dispatcher render(formato="markdown"|"pdf")
│   │       ├── markdown.py     # render_markdown
│   │       └── pdf.py          # render_pdf (reportlab, Times New Roman)
│   ├── visualization/          # Gráficos estadísticos
│   │   └── plots.py            # histogram, boxplot, scatter, qq, correlation
│   ├── llm/                    # Capa de abstracción LLM (Ollama pendiente, no integrado)
│   │   └── base.py             # BaseLLMClient (ABC) + RuleBasedLLMClient (heurístico)
│   ├── reports/                # Informes técnicos Markdown
│   │   └── generator.py        # ReportGenerator (artículo, dataset, missing_data)
│   └── utils/                  # Utilidades
│       └── logger.py           # setup_logger
│
├── outputs/
│   ├── benchmarks/             # Resultados JSON/MD de benchmarks (Criterio 3)
│   ├── figures/                # Gráficos generados
│   ├── tables/                 # Tablas generadas
│   └── reports/                # Informes PDF/Markdown generados (ej. <dataset>_informe_<marca>.pdf)
│
└── tests/                      # Suite de pruebas unitarias (394 tests)
    ├── test_parser.py / test_extractor.py / test_analyzer.py / test_generator.py
    ├── test_data_loader.py / test_data_validator.py
    ├── test_eda.py / test_hypothesis.py / test_hypothesis_additional.py
    ├── test_plots.py / test_dataset_analyzer.py / test_dataset_analyzer_e9d.py
    ├── test_missing_data_detector.py / test_missing_data_diagnostics.py
    ├── test_imputation_methods.py / test_imputation_registry.py / test_imputation_evaluation.py
    ├── test_imputation_selector.py / test_imputation_validation.py / test_pipeline.py
    ├── test_orchestration_flow.py / test_deliverables.py / test_deliverables_pdf.py
    └── test_audit_e9.py / test_import_regression.py
```

Nota: `PaperStat/` es un directorio vacío heredado (no eliminar sin autorización).

---

## Fases del Proyecto

### Fase 1: Núcleo de Análisis de Artículos Científicos
- Extracción de texto de artículos científicos (PDF, TXT, MD).
- Identificación de los 19 puntos clave (autores, objetivos, muestra, metodología, etc.).
- Análisis metodológico de supuestos estadísticos.
- Generación de informes en Markdown con regla anti-alucinaciones.
- Interfaz `BaseLLMClient` lista para conectar Ollama localmente (no integrado al flujo principal).

### Fase 2: Análisis Exploratorio y Estadístico de Datos
- **Carga de Datos (`src/data/loader.py`)**: Lectura automática de archivos CSV y Excel (`.csv`, `.xlsx`, `.xls`).
- **Validación de Calidad (`src/data/validator.py`)**: Clasificación automática de variables, detección de nulos y duplicados.
- **Análisis Exploratorio EDA (`src/analysis/eda.py`)**: Estadísticos descriptivos, tablas de frecuencia, matriz de correlación y outliers IQR.
- **Pruebas de Hipótesis (`src/analysis/hypothesis.py`)**: 22 pruebas (Shapiro-Wilk, Levene, t 1/ind/pareada, Wilcoxon, Mann-Whitney, ANOVA, Welch, Kruskal-Wallis, KS 1/2, Lilliefors, Bartlett, Breusch-Pagan, White, Durbin-Watson, Breusch-Godfrey, RESET, chi², Tukey HSD, permutaciones).
- **Visualización Estadística (`src/visualization/plots.py`)**: Histogramas (KDE), Boxplots, Dispersión, Q-Q plots y Mapas de Calor.

### Fase 3: Analizador Estadístico Inteligente y Coordinador Autónomo
- **Coordinador de Dataset (`src/analysis/dataset_analyzer.py`)**: `DatasetStatisticalAnalyzer` realiza análisis autónomos e integrales de datasets.
- **Diagnóstico Contextual de Normalidad**: Evaluación combinada de tamaño de muestra ($n$), Shapiro-Wilk, asimetría y atípicos sin caer en reglas binarias rígidas. Distingue entre la normalidad de los datos originales, de los residuos y de la distribución muestral de la media (TLC).
- **Criterios Integrales de Correlación**: Evaluación de linealidad, monotonicidad, atípicos e independencia (Pearson vs. Spearman).
- **Gestión Explícita de Supuestos**: Marca los supuestos no automatizables (como Independencia de Observaciones) como `"Supuesto no evaluado / Pendiente de verificación"`.
- **Desacoplamiento e Inferencia**: Explicaciones pedagógicas y advertencia de que significancia no implica causalidad.

### Fases 4–8: Datos faltantes E1–E7
Ver `docs/roadmap.md` y `docs/metodologia.md §10`: detección (E1), diagnóstico del mecanismo (E2), 10 métodos + registro (E3), evaluación artificial (E4), selección con `DEFAULT_WEIGHTS` (E5), validación con 7 checks (E6) y pipeline opt-in con `ValidationRevisionError` (E7).

### Fases 10–12: P-FLOW y entregables
`PaperStatsFlow` (diagnóstico → decisión → imputación → análisis bajo demanda) + `src/deliverables/` (modelo neutral `Deliverable` → renderers Markdown/PDF) + orden `informe()` (PDF Times New Roman sin recalcular).

---

## Resultado esperado de la ejecución

1. **Tests:** `pytest -q` → `394 passed` en ~25–30 s. Código de salida 0. Los tests son la especificación del comportamiento (ver `docs/project_map.md §2`).
2. **Ejemplo P-FLOW sin faltantes:** `diagnose()` retorna `estado="sin_faltantes"`, `entregable_inicial()` produce un `Deliverable` con secciones de calidad y EDA (sin pruebas inferenciales), `analizar()` retorna dict con `executed_test_results` y `missing_data` (E1-E3).
3. **Ejemplo con faltantes:** `diagnose()` retorna `esperando_decision`; `imputar()` sin decisión lanza `ValueError`; con `accept_recommendation=True` o `method_override` imputa y valida (E6); si `"Aceptable"` → `datos_preparados`, si `"Revisar"` con `strict=True` → `ValidationRevisionError`.
4. **Informe PDF:** `flow.informe()` genera `outputs/reports/<dataset>_informe_<YYYYMMDD_HHMMSS>.pdf` en Times New Roman (TTF con fallback Type1), sin recalcular estadísticas. Sin `Deliverable` previo → `ValueError`.

Toda ejecución es local y determinista salvo E4 (evaluación artificial con `fraction=0.2` sobre casos completos, advertida como exploratoria si `n` pequeño). Los datos originales nunca se mutan.

---

## Documentación on-demand

- `docs/project_map.md` — estructura, API pública y tests.
- `docs/metodologia.md` — inventario metodológico completo.
- `docs/roadmap.md` — estado, plan futuro y decisiones.
- `docs/arquitectura_agente.md` — diagrama P-FLOW, estados y guardarraíles.

---

## Notas de reproducibilidad

- Datos: `data/raw/Drug Price.xlsx` está versionado como dataset de ejemplo (excepción en `.gitignore`); el resto de `data/raw/`, `data/processed/` y `outputs/` no se versiona (ver `.gitignore`); usa `.gitkeep` para preservar carpetas vacías.
- No hay claves API ni envío a servicios externos por defecto (`AGENTS.md §2`).
- Para citar versiones exactas, guarda la salida de `pip freeze` junto al informe: `pip freeze > outputs/reports/versiones_$(date +%Y%m%d).txt`.
