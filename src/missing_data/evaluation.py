"""
Evaluación artificial de métodos de imputación (Etapa E4).

Procedimiento estándar de validación por ocultamiento:
  1. Tomar un DataFrame con columnas completas (la "verdad" es conocida).
  2. Inducir faltantes artificialmente (MCAR o MAR) con una fracción conocida.
  3. Imputar con cada método del registro E3.
  4. Comparar las imputaciones contra los valores verdaderos con métricas de error
     (MAE y RMSE para numéricas; precisión/accuracy para categóricas).

La inducción de faltantes no modifica el DataFrame original. Con la misma semilla,
la evaluación completa es reproducible (incluso para métodos estocásticos como
iterativo y MICE). Esta etapa no selecciona un método: solo compara.

Notas de diseño:
  - Las columnas a evaluar deben estar completas en el DataFrame de entrada.
  - Mecanismo MAR: la probabilidad de ausencia de una variable depende de una
    variable predictora observada; se induce ocultando las filas de mayor valor
    del predictor (tras una permutación reproducible por repetición).
  - MNAR no se simula: requiere censura dependiente del propio valor faltante y
    su evaluación por error contra la "verdad" es menos informativa.
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from src.missing_data.methods import (
    ImputationMethod,
    _is_categorical_series,
    _is_numeric_series,
)
from src.missing_data.registry import get as registry_get
from src.missing_data.registry import names as registry_names
from src.utils.logger import setup_logger

logger = setup_logger("ArtificialMissingnessEvaluator")

SUPPORTED_MECHANISMS = ("MCAR", "MAR")


def induce_missing(
    df: pd.DataFrame,
    *,
    columns: Optional[List[Any]] = None,
    fraction: float,
    mechanism: str = "MCAR",
    predictor: Optional[Any] = None,
    random_state: Optional[int] = None,
) -> pd.DataFrame:
    """
    Devuelve una copia del DataFrame con faltantes inducidos artificialmente.

    Args:
        df: DataFrame con columnas completas.
        columns: Columnas en las que inducir faltantes (por defecto todas).
        fraction: Proporción de celdas ocultadas por columna (0, 1).
        mechanism: "MCAR" o "MAR".
        predictor: Columna numérica completa que condiciona la ausencia (solo MAR);
            nunca se oculta.
        random_state: Semilla para reproducibilidad.

    Returns:
        Copia del DataFrame con NaN inducidos.

    Raises:
        TypeError: Si df no es un DataFrame de pandas.
        ValueError: Si fraction, mechanism o predictor no son válidos.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Se requiere un DataFrame de pandas.")
    if not (0.0 < fraction < 1.0):
        raise ValueError("fraction debe estar en el intervalo abierto (0, 1).")
    if mechanism not in SUPPORTED_MECHANISMS:
        raise ValueError(
            f"mechanism debe ser uno de {SUPPORTED_MECHANISMS}, no '{mechanism}'."
        )

    target_columns = list(df.columns) if columns is None else list(columns)
    for col in target_columns:
        if col not in df.columns:
            raise KeyError(f"La columna '{col}' no existe en el DataFrame.")

    rng = np.random.default_rng(random_state)
    result = df.copy()
    n = len(df)
    if n == 0:
        return result

    if mechanism == "MCAR":
        for col in target_columns:
            if not (_is_numeric_series(df[col]) or _is_categorical_series(df[col])):
                continue
            k = int(fraction * n)
            idx = rng.choice(n, size=k, replace=False)
            _set_nan(result, col, idx)
    else:
        if predictor is None:
            raise ValueError("El mecanismo MAR requiere una columna 'predictor'.")
        if predictor not in df.columns:
            raise KeyError(f"La columna predictora '{predictor}' no existe.")
        if not _is_numeric_series(df[predictor]) or df[predictor].isna().any():
            raise ValueError(
                "El 'predictor' de MAR debe ser una columna numérica sin faltantes."
            )
        order = rng.permutation(n)
        z = df[predictor].to_numpy(dtype=float)[order]
        order_by_z = order[np.argsort(z)]
        k = int(fraction * n)
        miss_idx = order_by_z[-k:]
        for col in target_columns:
            if col == predictor:
                continue
            if not (_is_numeric_series(df[col]) or _is_categorical_series(df[col])):
                continue
            _set_nan(result, col, miss_idx)

    return result


def _set_nan(result: pd.DataFrame, col: Any, positions: np.ndarray) -> None:
    """Pone NaN en las posiciones indicadas de una columna (posicional)."""
    if len(positions) == 0:
        return
    pos = result.columns.get_loc(col)
    result.iloc[positions, pos] = np.nan


@dataclass
class MethodEvaluation:
    """Resultado de la evaluación de un método sobre faltantes artificiales."""

    method: str
    error: Optional[str]
    n_induced: int
    n_repeats: int
    per_column: Dict[str, Dict[str, float]]
    global_metrics: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        """Devuelve la evaluación como diccionario JSON-serializable."""
        return asdict(self)


@dataclass
class ImputationEvaluationReport:
    """Reporte estructurado y reproducible de la evaluación artificial."""

    mechanism: str
    fraction: float
    predictor: Optional[Any]
    n_repeats: int
    random_state: Optional[int]
    numeric_columns: List[str]
    categorical_columns: List[str]
    methods: List[MethodEvaluation]
    ranking: Dict[str, float]
    reproduction: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Devuelve el reporte como diccionario JSON-serializable."""
        return asdict(self)


class ArtificialMissingnessEvaluator:
    """
    Evalúa métodos de imputación inducendo faltantes artificiales.

    Los métodos pueden pasarse como nombres registrados (se instancian por
    repetición con semilla derivada para garantizar reproducibilidad) o como
    instancias de `ImputationMethod` (se usan tal cual).
    """

    def __init__(
        self,
        random_state: Optional[int] = 42,
        n_repeats: int = 1,
    ) -> None:
        self.random_state = random_state
        if int(n_repeats) < 1:
            raise ValueError("n_repeats debe ser al menos 1.")
        self.n_repeats = int(n_repeats)

    def evaluate(
        self,
        df: pd.DataFrame,
        *,
        columns: Optional[List[Any]] = None,
        methods: Optional[Union[str, List[Union[str, ImputationMethod]]]] = None,
        fraction: float = 0.2,
        mechanism: str = "MCAR",
        predictor: Optional[Any] = None,
    ) -> ImputationEvaluationReport:
        """
        Evalúa métodos de imputación con faltantes inducidos.

        Args:
            df: DataFrame con columnas completas (verdad conocida).
            columns: Columnas a evaluar (por defecto todas las completas).
            methods: Nombres registrados, instancias de ImputationMethod o None
                (por defecto todos los métodos no temporales compatibles).
            fraction: Fracción de celdas ocultadas por columna.
            mechanism: "MCAR" o "MAR".
            predictor: Predictor numérico completo para MAR.

        Returns:
            ImputationEvaluationReport con métricas por método y ranking.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Se requiere un DataFrame de pandas.")
        if not (0.0 < fraction < 1.0):
            raise ValueError("fraction debe estar en el intervalo abierto (0, 1).")
        if mechanism not in SUPPORTED_MECHANISMS:
            raise ValueError(
                f"mechanism debe ser uno de {SUPPORTED_MECHANISMS}, no '{mechanism}'."
            )

        target_columns = list(df.columns) if columns is None else list(columns)
        for col in target_columns:
            if col not in df.columns:
                raise KeyError(f"La columna '{col}' no existe en el DataFrame.")
            if df[col].isna().any():
                raise ValueError(
                    f"La columna '{col}' debe estar completa para evaluar "
                    "(la verdad debe ser conocida)."
                )
        target_columns = [str(col) for col in target_columns]

        numeric_cols = [col for col in target_columns if _is_numeric_series(df[col])]
        categorical_cols = [
            col for col in target_columns if _is_categorical_series(df[col])
        ]
        if not numeric_cols and not categorical_cols:
            raise ValueError(
                "Las columnas objetivo no contienen variables numéricas ni categóricas."
            )

        if mechanism == "MAR":
            if predictor is None:
                raise ValueError("El mecanismo MAR requiere un 'predictor'.")
            if predictor not in df.columns:
                raise KeyError(f"La columna predictora '{predictor}' no existe.")
            if not _is_numeric_series(df[predictor]) or df[predictor].isna().any():
                raise ValueError(
                    "El 'predictor' de MAR debe ser una columna numérica sin faltantes."
                )
            predictor = str(predictor)

        method_specs = self._resolve_methods(methods, numeric_cols, categorical_cols)

        logger.info(
            f"Evaluando {len(method_specs)} métodos sobre {len(target_columns)} "
            f"columnas (mechanism={mechanism}, fraction={fraction}, "
            f"n_repeats={self.n_repeats})."
        )

        acc: Dict[str, Dict[Any, Dict[str, list]]] = {}
        for name, _spec in method_specs:
            acc[name] = {col: _new_accumulator() for col in target_columns}

        for r in range(self.n_repeats):
            seed = None if self.random_state is None else self.random_state + r
            masked = induce_missing(
                df,
                columns=target_columns,
                fraction=fraction,
                mechanism=mechanism,
                predictor=predictor,
                random_state=seed,
            )
            for name, spec in method_specs:
                if isinstance(spec, type):
                    method = spec(random_state=seed)
                else:
                    method = spec
                try:
                    imputed = method.impute(masked)
                except (ValueError, TypeError, KeyError, np.linalg.LinAlgError) as exc:
                    acc[name]["_error"] = str(exc)
                    continue

                supported = _supported_columns(method, numeric_cols, categorical_cols)
                for col in supported:
                    if col == predictor:
                        continue
                    induced = masked[col].isna().to_numpy()
                    if not induced.any():
                        continue
                    true = df[col].to_numpy()[induced]
                    pred = imputed[col].to_numpy()[induced]
                    _accumulate(acc[name][col], true, pred)

        methods_result: List[MethodEvaluation] = []
        for name, _spec in method_specs:
            entry = acc[name].pop("_error", None)
            methods_result.append(
                _build_method_evaluation(name, entry, acc[name], self.n_repeats)
            )

        ranking = _build_ranking(methods_result, numeric_cols)

        reproduction = {
            "random_state": self.random_state,
            "n_repeats": self.n_repeats,
            "fraction": fraction,
            "mechanism": mechanism,
            "predictor": predictor,
            "module": "src.missing_data.evaluation",
            "version": "1.0",
        }

        return ImputationEvaluationReport(
            mechanism=mechanism,
            fraction=fraction,
            predictor=predictor,
            n_repeats=self.n_repeats,
            random_state=self.random_state,
            numeric_columns=numeric_cols,
            categorical_columns=categorical_cols,
            methods=methods_result,
            ranking=ranking,
            reproduction=reproduction,
        )

    def _resolve_methods(
        self,
        methods: Optional[Union[str, List[Union[str, ImputationMethod]]]],
        numeric_cols: List[str],
        categorical_cols: List[str],
    ) -> List[Tuple[str, Union[type, ImputationMethod]]]:
        if methods is None:
            specs: List[Tuple[str, Union[type, ImputationMethod]]] = []
            for name in registry_names():
                caps = registry_get(name).capabilities
                if caps.temporal_only:
                    continue
                if (caps.supports_numeric and numeric_cols) or (
                    caps.supports_categorical and categorical_cols
                ):
                    specs.append((name, registry_get(name)))
            return specs

        if isinstance(methods, str):
            methods = [methods]
        specs: List[Tuple[str, Union[type, ImputationMethod]]] = []
        seen: set = set()
        for item in methods:
            if isinstance(item, str):
                method_cls = registry_get(item)
                key = item
                entry = (item, method_cls)
            elif isinstance(item, ImputationMethod):
                name = getattr(item, "name", type(item).__name__)
                key = name
                entry = (name, item)
            else:
                raise TypeError(
                    "methods debe contener nombres registrados o instancias de ImputationMethod."
                )
            if key in seen:
                continue
            seen.add(key)
            specs.append(entry)
        return specs


def _new_accumulator() -> Dict[str, list]:
    return {"mae": [], "rmse": [], "accuracy": [], "n_induced": [], "n_unimputed": []}


def _supported_columns(
    method: ImputationMethod, numeric_cols: List[str], categorical_cols: List[str]
) -> List[str]:
    caps = method.capabilities
    result = []
    if caps.supports_numeric:
        result.extend(numeric_cols)
    if caps.supports_categorical:
        result.extend(categorical_cols)
    return result


def _accumulate(acc: Dict[str, list], true: np.ndarray, pred: np.ndarray) -> None:
    valid = ~pd.isna(pred)
    n_induced = int(len(true))
    n_unimputed = int((~valid).sum())
    acc["n_induced"].append(n_induced)
    acc["n_unimputed"].append(n_unimputed)
    if n_unimputed == n_induced:
        return
    t = true[valid]
    p = pred[valid]

    if pd.api.types.is_numeric_dtype(p) and pd.api.types.is_numeric_dtype(t):
        diff = np.abs(p.astype(float) - t.astype(float))
        acc["mae"].append(float(np.mean(diff)))
        acc["rmse"].append(float(np.sqrt(np.mean(diff ** 2))))
    else:
        acc["accuracy"].append(float(np.mean(p == t)))


def _build_method_evaluation(
    name: str,
    error: Optional[str],
    acc: Dict[Any, Dict[str, list]],
    n_repeats: int,
) -> MethodEvaluation:
    per_column: Dict[str, Dict[str, float]] = {}
    n_induced_total = 0
    for col, col_acc in acc.items():
        if not col_acc["n_induced"]:
            continue
        entry: Dict[str, float] = {}
        if col_acc["mae"]:
            entry["mae"] = float(np.mean(col_acc["mae"]))
        if col_acc["rmse"]:
            entry["rmse"] = float(np.mean(col_acc["rmse"]))
        if col_acc["accuracy"]:
            entry["accuracy"] = float(np.mean(col_acc["accuracy"]))
        entry["n_induced"] = int(np.sum(col_acc["n_induced"]))
        entry["n_unimputed"] = int(np.sum(col_acc["n_unimputed"]))
        n_induced_total += entry["n_induced"]
        per_column[str(col)] = entry

    global_metrics: Dict[str, float] = {}
    mae_vals = [e["mae"] for e in per_column.values() if "mae" in e]
    rmse_vals = [e["rmse"] for e in per_column.values() if "rmse" in e]
    acc_vals = [e["accuracy"] for e in per_column.values() if "accuracy" in e]
    if mae_vals:
        global_metrics["mae"] = float(np.mean(mae_vals))
    if rmse_vals:
        global_metrics["rmse"] = float(np.mean(rmse_vals))
    if acc_vals:
        global_metrics["accuracy"] = float(np.mean(acc_vals))

    return MethodEvaluation(
        method=name,
        error=error,
        n_induced=n_induced_total,
        n_repeats=n_repeats,
        per_column=per_column,
        global_metrics=global_metrics,
    )


def _build_ranking(
    evaluations: List[MethodEvaluation], numeric_cols: List[str]
) -> Dict[str, float]:
    ranking: Dict[str, float] = {}
    for ev in evaluations:
        if ev.error is not None or not ev.global_metrics:
            continue
        if numeric_cols and "rmse" in ev.global_metrics:
            score = ev.global_metrics["rmse"]
        elif "accuracy" in ev.global_metrics:
            score = 1.0 - ev.global_metrics["accuracy"]
        else:
            continue
        ranking[ev.method] = float(score)
    return dict(sorted(ranking.items(), key=lambda kv: kv[1]))