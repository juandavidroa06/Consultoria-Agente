# Benchmark de recursos — Imputación (media vs knn)

Generado: 2026-08-23T13:47:31-0500 — Python 3.14.7

## Configuración

- **Dataset:** `data\raw\Drug Price.xlsx` — 73 filas x 8 columnas, 159 celdas faltantes en 6 variables
- **Repeticiones por método:** 2
- **Semilla:** no aplica (`uses_random_state=False` para ambos; documentada como `null`; si se comparasen `iterativo`/`mice` se fijaría `random_state=42`)
- **Medición tiempo:** `time.perf_counter()` por corrida sobre copia fresca del DataFrame
- **Medición memoria:** `tracemalloc` peak KB (stdlib); RSS delta no disponible en esta plataforma (módulo `resource` ausente en Windows)
- **Condiciones:** mismo dataset, misma máquina, ejecución secuencial, instancia fresca por corrida

## Resultados

| método | clase | tiempo medio (s) | tiempo mediana (s) | tiempo min–max (s) | peak memoria medio (KB) | peak memoria max (KB) | RSS delta medio (KB) | n_runs | seed |
|---|---|---|---|---|---|---|---|---|---|
| media | MeanImputation | 0.0079 | 0.0079 | 0.0073–0.0085 | 23.5 | 24.0 | N/A | 2 | — |
| knn | KNNImputation (k=5) | 0.0229 | 0.0229 | 0.0145–0.0313 | 175.4 | 199.1 | N/A | 2 | — |

## Comparación

- **Más rápido (tiempo medio):** `media` (ratio knn/media = 2.89x)
- **Menor pico de memoria (tracemalloc):** `media` (ratio knn/media = 7.46x)

## Reproducibilidad

```bash
# Linux/macOS:
.venv/bin/python benchmarks/benchmark_imputation.py --runs 2 --dataset "data\raw\Drug Price.xlsx"
# Windows (PowerShell):
.venv\Scripts\python benchmarks/benchmark_imputation.py --runs 2 --dataset "data\raw\Drug Price.xlsx"
```

Payload JSON completo: `outputs/benchmarks/results.json`
