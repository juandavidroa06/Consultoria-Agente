"""
Benchmark reproducible de recursos para dos métodos de imputación de PaperStats.

Compara:
  - media (MeanImputation)  — univariado, O(n), sin estado aleatorio
  - knn   (KNNImputation)   — multivariado por distancia, O(n log n), sin semilla

Mide por método, sobre el MISMO dataset y MISMA configuración:
  1. tiempo de ejecución con time.perf_counter()
  2. memoria máxima con tracemalloc (peak) y resource.getrusage(RUSAGE_SELF).ru_maxrss

No modifica src/, tests/, README.md ni requirements.txt. Es un script aislado
que importa únicamente la API pública de src/missing_data.

Uso:
  .venv/bin/python benchmarks/benchmark_imputation.py
  # o con parámetros:
  .venv/bin/python benchmarks/benchmark_imputation.py --runs 7 --dataset "data/raw/Drug Price.xlsx"

Salidas:
  outputs/benchmarks/results.json  (datos cuantitativos)
  outputs/benchmarks/results.md    (tabla comparativa)

Reproducibilidad: misma semilla cuando aplica (random_state=42 si el método lo usa),
mismo DataFrame copiado por ejecución, mismo orden, n_runs repetidas.
"""
import argparse
import json
import platform
import resource
import statistics
import sys
import time
import tracemalloc
from pathlib import Path

import pandas as pd

# Permitir ejecución desde raíz y desde benchmarks/
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.missing_data.methods import KNNImputation, MeanImputation  # noqa: E402

DATASET_DEFAULT = ROOT / "data" / "raw" / "Drug Price.xlsx"
OUTPUT_DIR = ROOT / "outputs" / "benchmarks"
RANDOM_STATE = 42
N_RUNS_DEFAULT = 5


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset no encontrado: {path}")
    if path.suffix.lower() in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)
    return df


def benchmark_method(method_cls, df: pd.DataFrame, n_runs: int, **kwargs):
    """
    Ejecuta n_runs imputaciones aisladas del mismo df.
    Retorna lista de dicts con tiempo y memoria por ejecución.
    """
    runs = []
    for i in range(n_runs):
        # Instancia fresca por corrida (evita estado fitted previo)
        method = method_cls(**kwargs) if kwargs else method_cls()
        df_copy = df.copy(deep=True)

        # Memoria inicial de proceso (RSS)
        rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

        tracemalloc.start()
        t0 = time.perf_counter()
        result = method.impute(df_copy)
        t1 = time.perf_counter()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        elapsed = t1 - t0

        # Validación: no mutó original, imputó algo
        assert df.isna().sum().sum() > 0, "dataset de benchmark debe tener faltantes"
        # result no se usa para métrica, solo para verificar que no lanza

        runs.append(
            {
                "run": i + 1,
                "elapsed_sec": elapsed,
                "tracemalloc_current_kb": current / 1024,
                "tracemalloc_peak_kb": peak / 1024,
                "rss_before_kb": float(rss_before),
                "rss_after_kb": float(rss_after),
                "rss_delta_kb": float(rss_after - rss_before),
            }
        )
        # Pequeña pausa para no saturar GC entre corridas (mismas condiciones de todas formas)
        time.sleep(0.01)
    return runs


def summarize(runs):
    elapsed = [r["elapsed_sec"] for r in runs]
    peak = [r["tracemalloc_peak_kb"] for r in runs]
    rss_delta = [r["rss_delta_kb"] for r in runs]
    return {
        "n_runs": len(runs),
        "elapsed_mean_sec": statistics.mean(elapsed),
        "elapsed_median_sec": statistics.median(elapsed),
        "elapsed_min_sec": min(elapsed),
        "elapsed_max_sec": max(elapsed),
        "elapsed_stdev_sec": statistics.stdev(elapsed) if len(elapsed) > 1 else 0.0,
        "peak_mean_kb": statistics.mean(peak),
        "peak_median_kb": statistics.median(peak),
        "peak_min_kb": min(peak),
        "peak_max_kb": max(peak),
        "peak_stdev_kb": statistics.stdev(peak) if len(peak) > 1 else 0.0,
        "rss_delta_mean_kb": statistics.mean(rss_delta),
        "rss_delta_max_kb": max(rss_delta),
        "runs": runs,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark media vs knn")
    parser.add_argument("--runs", type=int, default=N_RUNS_DEFAULT, help="Repeticiones por método")
    parser.add_argument("--dataset", type=str, default=str(DATASET_DEFAULT), help="Ruta al dataset")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    df = load_dataset(dataset_path)
    n_rows, n_cols = df.shape
    n_missing_cells = int(df.isna().sum().sum())
    n_missing_vars = int((df.isna().sum() > 0).sum())

    print(f"Dataset: {dataset_path} — {n_rows} filas x {n_cols} columnas, {n_missing_cells} celdas faltantes en {n_missing_vars} variables")
    print(f"Python {platform.python_version()} ({platform.platform()}), pandas {pd.__version__}")
    print(f"Config: n_runs={args.runs}, random_state={RANDOM_STATE} (no aplica a media/knn, documentado como null)")

    # Verificación previa de que ambos métodos soportan el dataset
    # (media: supports_numeric, knn: needs_other_columns) — ya verificado en inspección,
    # pero se valida de nuevo para reproducibilidad.
    for cls, name in [(MeanImputation, "media"), (KNNImputation, "knn")]:
        try:
            cls().validate_input(df)
        except Exception as e:
            raise RuntimeError(f"Método {name} no aplicable al dataset: {e}") from e

    # Benchmark aislado por método, mismas condiciones
    results = {}

    print("\nBenchmark media (MeanImputation)...")
    media_runs = benchmark_method(MeanImputation, df, args.runs)
    media_summary = summarize(media_runs)
    print(f"  media: mean {media_summary['elapsed_mean_sec']:.4f}s, peak {media_summary['peak_mean_kb']:.1f} KB")

    print("Benchmark knn (KNNImputation, n_neighbors=5)...")
    knn_runs = benchmark_method(KNNImputation, df, args.runs, n_neighbors=5, weights="uniform")
    knn_summary = summarize(knn_runs)
    print(f"  knn:   mean {knn_summary['elapsed_mean_sec']:.4f}s, peak {knn_summary['peak_mean_kb']:.1f} KB")

    # Payload cuantitativo
    payload = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "dataset": {
            "path": str(dataset_path.relative_to(ROOT)) if dataset_path.is_relative_to(ROOT) else str(dataset_path),
            "rows": n_rows,
            "columns": n_cols,
            "n_missing_cells": n_missing_cells,
            "n_missing_variables": n_missing_vars,
            "size_human": f"{n_rows}x{n_cols}",
        },
        "config": {
            "n_runs": args.runs,
            "random_state": None,
            "random_state_note": "media y knn no usan random_state (uses_random_state=False); semilla fija 42 documentada como no aplicable",
            "methods": [
                {"name": "media", "class": "MeanImputation", "params": {}},
                {"name": "knn", "class": "KNNImputation", "params": {"n_neighbors": 5, "weights": "uniform"}},
            ],
            "measurement": {
                "time": "time.perf_counter() por corrida, sobre copia fresca del DataFrame",
                "memory_tracemalloc": "tracemalloc.get_traced_memory() peak KB (stdlib, sin dependencias externas)",
                "memory_rss": "resource.getrusage(RUSAGE_SELF).ru_maxrss delta KB (RSS máximo del proceso)",
            },
            "reproducibility": "mismo dataset, misma copia por corrida, misma instancia fresca, mismo orden secuencial, sin paralelismo",
        },
        "results": {
            "media": {
                "method": "media",
                "class": "MeanImputation",
                "uses_random_state": False,
                "seed": None,
                **media_summary,
            },
            "knn": {
                "method": "knn",
                "class": "KNNImputation",
                "uses_random_state": False,
                "seed": None,
                **knn_summary,
            },
        },
        "comparison": {
            "faster": "media" if media_summary["elapsed_mean_sec"] < knn_summary["elapsed_mean_sec"] else "knn",
            "lower_peak_memory": "media" if media_summary["peak_mean_kb"] < knn_summary["peak_mean_kb"] else "knn",
            "elapsed_ratio_knn_over_media": knn_summary["elapsed_mean_sec"] / media_summary["elapsed_mean_sec"]
            if media_summary["elapsed_mean_sec"] > 0
            else None,
            "peak_ratio_knn_over_media": knn_summary["peak_mean_kb"] / media_summary["peak_mean_kb"]
            if media_summary["peak_mean_kb"] > 0
            else None,
        },
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    json_path = OUTPUT_DIR / "results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"\nJSON guardado: {json_path}")

    # Markdown tabla comparativa
    md_path = OUTPUT_DIR / "results.md"
    media = payload["results"]["media"]
    knn = payload["results"]["knn"]
    comp = payload["comparison"]
    md = f"""# Benchmark de recursos — Imputación (media vs knn)

Generado: {payload["timestamp"]} — Python {payload["python_version"]}

## Configuración

- **Dataset:** `{payload["dataset"]["path"]}` — {n_rows} filas x {n_cols} columnas, {n_missing_cells} celdas faltantes en {n_missing_vars} variables
- **Repeticiones por método:** {args.runs}
- **Semilla:** no aplica (`uses_random_state=False` para ambos; documentada como `null`; si se comparasen `iterativo`/`mice` se fijaría `random_state=42`)
- **Medición tiempo:** `time.perf_counter()` por corrida sobre copia fresca del DataFrame
- **Medición memoria:** `tracemalloc` peak KB (stdlib) + `resource.getrusage` RSS delta KB
- **Condiciones:** mismo dataset, misma máquina, ejecución secuencial, instancia fresca por corrida

## Resultados

| método | clase | tiempo medio (s) | tiempo mediana (s) | tiempo min–max (s) | peak memoria medio (KB) | peak memoria max (KB) | RSS delta medio (KB) | n_runs | seed |
|---|---|---|---|---|---|---|---|---|---|
| media | MeanImputation | {media["elapsed_mean_sec"]:.4f} | {media["elapsed_median_sec"]:.4f} | {media["elapsed_min_sec"]:.4f}–{media["elapsed_max_sec"]:.4f} | {media["peak_mean_kb"]:.1f} | {media["peak_max_kb"]:.1f} | {media["rss_delta_mean_kb"]:.1f} | {media["n_runs"]} | — |
| knn | KNNImputation (k=5) | {knn["elapsed_mean_sec"]:.4f} | {knn["elapsed_median_sec"]:.4f} | {knn["elapsed_min_sec"]:.4f}–{knn["elapsed_max_sec"]:.4f} | {knn["peak_mean_kb"]:.1f} | {knn["peak_max_kb"]:.1f} | {knn["rss_delta_mean_kb"]:.1f} | {knn["n_runs"]} | — |

## Comparación

- **Más rápido (tiempo medio):** `{comp["faster"]}` (ratio knn/media = {comp["elapsed_ratio_knn_over_media"]:.2f}x)
- **Menor pico de memoria (tracemalloc):** `{comp["lower_peak_memory"]}` (ratio knn/media = {comp["peak_ratio_knn_over_media"]:.2f}x)

## Reproducibilidad

```bash
.venv/bin/python benchmarks/benchmark_imputation.py --runs {args.runs} --dataset {payload["dataset"]["path"]}
```

Payload JSON completo: `outputs/benchmarks/results.json`
"""
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"Markdown guardado: {md_path}")
    print("\nComparación:", comp)


if __name__ == "__main__":
    main()
