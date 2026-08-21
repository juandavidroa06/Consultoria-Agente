# Benchmark de recursos — Imputación (media vs knn)

Generado: 2026-08-21T09:25:31-0500 — Python 3.14.4

## Configuración

- **Dataset:** `data/raw/Drug Price.xlsx` — 73 filas x 8 columnas, 159 celdas faltantes en 6 variables
- **Repeticiones por método:** 5
- **Semilla:** no aplica (`uses_random_state=False` para ambos; documentada como `null`; si se comparasen `iterativo`/`mice` se fijaría `random_state=42`)
- **Medición tiempo:** `time.perf_counter()` por corrida sobre copia fresca del DataFrame
- **Medición memoria:** `tracemalloc` peak KB (stdlib) + `resource.getrusage` RSS delta KB
- **Condiciones:** mismo dataset, misma máquina, ejecución secuencial, instancia fresca por corrida

## Resultados

| método | clase | tiempo medio (s) | tiempo mediana (s) | tiempo min–max (s) | peak memoria medio (KB) | peak memoria max (KB) | RSS delta medio (KB) | n_runs | seed |
|---|---|---|---|---|---|---|---|---|---|
| media | MeanImputation | 0.0069 | 0.0065 | 0.0063–0.0083 | 22.8 | 24.0 | 25.6 | 5 | — |
| knn | KNNImputation (k=5) | 0.0179 | 0.0162 | 0.0159–0.0249 | 160.9 | 199.5 | 141.6 | 5 | — |

## Comparación

- **Más rápido (tiempo medio):** `media` (ratio knn/media = 2.59x)
- **Menor pico de memoria (tracemalloc):** `media` (ratio knn/media = 7.07x)

## Reproducibilidad

```bash
.venv/bin/python benchmarks/benchmark_imputation.py --runs 5 --dataset data/raw/Drug Price.xlsx
```

Payload JSON completo: `outputs/benchmarks/results.json`
