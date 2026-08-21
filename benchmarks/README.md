# Benchmark de recursos — Imputación

Objetivo del Criterio 3 (medición de recursos): generar evidencia cuantitativa y reproducible del uso de tiempo y memoria de dos métodos de imputación ya implementados en PaperStats, sin modificar `src/`, `tests/`, `README.md` ni `requirements.txt`.

## Métodos comparados

Se eligieron **dos métodos existentes y verificados** sobre el mismo dataset real `data/raw/Drug Price.xlsx`:

1. **`media` — `MeanImputation` (`src/missing_data/methods.py:149-172`)**
   - `capabilities = supports_numeric=True`, no temporal, no necesita otras columnas.
   - Univariado: imputa cada columna con su media aritmética. Complejidad O(n).
   - `uses_random_state = False` — semilla no aplica.

2. **`knn` — `KNNImputation` (`src/missing_data/methods.py:288-332`)**
   - `capabilities = supports_numeric=True, needs_other_columns=True`.
   - Multivariado por distancia: `sklearn.impute.KNNImputer(n_neighbors=5)`. Complejidad O(n log n) por distancia.
   - `uses_random_state = False` — semilla no aplica (determinista).

**Por qué estos dos:** ambos están registrados en `src/missing_data/registry.py:29-40` (`"media"`, `"knn"`), ambos superan `validate_input()` sobre `Drug Price.xlsx` (73 filas x 8 columnas, 6 numéricas con 159 celdas faltantes), y ambos ejecutan sin fallo (verificado: `media` 0.011s / `knn` 0.040s en inspección previa). Métodos descartados para esta comparación: `regresion` (requiere columna numérica completa — falla en este dataset), `mice` (SVD did not converge en este dataset), `interpolacion_lineal`/`locf` (temporal_only, no aplica).

Alternativa válida documentada: `iterativo` vs `mice` (ambos con `uses_random_state=True`, `random_state=42`) sería igualmente reproducible, pero `mice` falla en este dataset, por lo que `media vs knn` es la comparación más robusta y clara (baseline univariado vs multivariado).

## Dataset

- **Archivo:** `data/raw/Drug Price.xlsx` (16775 bytes, Excel 2007+)
- **Tamaño:** 73 filas x 8 columnas (`Region`, `Country` categóricas sin faltantes; 6 numéricas `Amphetamine`, `Cannabis`, `Cocaine`, `Hallucinogens`, `Opioids`, `Offences` con faltantes)
- **Faltantes:** 159 celdas en 6 variables (verificado con `df.isna().sum()`). Es el dataset de ejemplo del proyecto (usado en `docs/roadmap.md` Fase 8).

Mismo `DataFrame` copiado con `df.copy(deep=True)` por cada corrida.

## Métricas

- **Tiempo:** `time.perf_counter()` — wall-clock por corrida, instancia fresca del método.
- **Memoria pico:** `tracemalloc.get_traced_memory()` peak KB (stdlib, sin `psutil` — verificado no instalado en `.venv`). Complementado con `resource.getrusage(RUSAGE_SELF).ru_maxrss` delta KB (RSS máximo del proceso, disponible en Linux).
- **Repeticiones:** `n_runs = 5` por defecto (configurable con `--runs`), estadísticas mean/median/min/max/stdev.

Mismas condiciones: mismo dataset, misma máquina, ejecución secuencial sin paralelismo, misma semilla cuando aplique (aquí `null` para ambos; si aplicase, `random_state=42`).

## Archivos creados

- `benchmarks/benchmark_imputation.py` — script aislado (este benchmark). Solo importa `src.missing_data.methods`, no modifica producción.
- `benchmarks/README.md` — este archivo (documentación del benchmark).
- `outputs/benchmarks/results.json` — payload cuantitativo (generado al ejecutar).
- `outputs/benchmarks/results.md` — tabla comparativa (generado al ejecutar).

No se modifica `src/`, `tests/`, `README.md`, `requirements.txt` ni la arquitectura.

## Cómo ejecutar (reproducible)

Desde la raíz del proyecto, con el entorno activado:

```bash
source .venv/bin/activate
python benchmarks/benchmark_imputation.py
# o sin activar:
.venv/bin/python benchmarks/benchmark_imputation.py

# con parámetros:
.venv/bin/python benchmarks/benchmark_imputation.py --runs 7 --dataset data/raw/Drug\ Price.xlsx
```

Salidas esperadas:

```
Dataset: data/raw/Drug Price.xlsx — 73 filas x 8 columnas, 159 celdas faltantes...
Benchmark media (MeanImputation)...
Benchmark knn (KNNImputation, n_neighbors=5)...
JSON guardado: outputs/benchmarks/results.json
Markdown guardado: outputs/benchmarks/results.md
```

Ver `outputs/benchmarks/results.json` para campos: `method`, `elapsed_mean_sec`, `peak_mean_kb`, `rss_delta_mean_kb`, `config`, `dataset {rows, columns, n_missing_cells}`, `seed`, `comparison`.

## Verificación

Tras ejecutar, verificar que `src/`, `tests/`, `README.md`, `requirements.txt` no fueron tocados:

```bash
find src tests -type f -mmin -10  # debe estar vacío
ls -la outputs/benchmarks/results.json outputs/benchmarks/results.md
cat outputs/benchmarks/results.json | python -m json.tool | head -n 40
```

El benchmark es determinista salvo variación de scheduler/OS; las ratios `knn/media` son estables (>1 para tiempo y pico en este dataset/maquina).
