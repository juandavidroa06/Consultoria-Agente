"""
Métodos de imputación de datos faltantes (Etapa E3).

Cada método implementa la interfaz `ImputationMethod` y declara sus
`MethodCapabilities` (tipos de variables soportados, si es exclusivo de
series temporales y si necesita otras columnas como predictores).

Reglas transversales:
  - Nunca se modifica el DataFrame original: se devuelve una copia.
  - La política de redondeo aplica únicamente a columnas de dtype entero:
    tras imputar se redondea y se restaura el dtype entero original.
  - Las columnas con el 100% de valores faltantes no pueden imputarse con
    métodos que requieren valores observados (media, KNN, iterativo, MICE,
    regresión, interpolación, LOCF); quedan intactas (NaN) y el hecho se
    documenta en la salida. Solo el método 'constante' puede imputarlas.
  - Los métodos temporales (interpolación lineal y LOCF) se registran con
    `temporal_only=True` y solo se ofrecen en contextos temporales.
"""

import warnings as _warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.linear_model import LinearRegression
from statsmodels.imputation.mice import MICEData

with _warnings.catch_warnings():
    _warnings.simplefilter("ignore")
    from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer

from src.utils.logger import setup_logger

logger = setup_logger("ImputationMethods")


@dataclass(frozen=True)
class MethodCapabilities:
    """Capacidades declaradas de un método de imputación."""

    supports_numeric: bool = False
    supports_categorical: bool = False
    temporal_only: bool = False
    needs_other_columns: bool = False


def _is_numeric_series(series: pd.Series) -> bool:
    return (
        pd.api.types.is_numeric_dtype(series.dtype)
        and not pd.api.types.is_bool_dtype(series.dtype)
        and not pd.api.types.is_datetime64_any_dtype(series.dtype)
        and not pd.api.types.is_timedelta64_dtype(series.dtype)
    )


def _is_categorical_series(series: pd.Series) -> bool:
    return not (
        _is_numeric_series(series)
        or pd.api.types.is_datetime64_any_dtype(series.dtype)
        or pd.api.types.is_timedelta64_dtype(series.dtype)
    )


def _restore_integer_rounding(series: pd.Series, original_dtype: Any) -> pd.Series:
    """Redondea y restaura el dtype entero original si corresponde."""
    if not pd.api.types.is_integer_dtype(original_dtype):
        return series
    rounded = series.round()
    if rounded.isna().any():
        return rounded.astype("Float64")
    try:
        return rounded.astype(original_dtype)
    except (TypeError, ValueError):
        return rounded.astype("Float64")


def _numeric_columns_with_missing(df: pd.DataFrame) -> List[Any]:
    return [col for col in df.columns if _is_numeric_series(df[col]) and df[col].isna().any()]


class ImputationMethod(ABC):
    """
    Interfaz común de un método de imputación.

    El flujo recomendado es `impute(df)`, que valida la entrada, ajusta el
    método sobre el DataFrame (fit) y devuelve una copia imputada (transform).
    """

    name: ClassVar[str] = ""
    description: ClassVar[str] = ""
    capabilities: ClassVar[MethodCapabilities] = MethodCapabilities()
    uses_random_state: ClassVar[bool] = False

    def __init__(self, random_state: Optional[int] = None) -> None:
        self.random_state = random_state
        self._fitted = False

    def validate_input(self, df: pd.DataFrame) -> None:
        """Rechaza de forma explícita las entradas que el método no soporta."""
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Se requiere un DataFrame de pandas.")
        has_numeric = any(_is_numeric_series(df[col]) for col in df.columns)
        has_categorical = any(_is_categorical_series(df[col]) for col in df.columns)
        if self.capabilities.supports_numeric and not self.capabilities.supports_categorical:
            if not has_numeric:
                raise ValueError(
                    f"El método '{self.name}' solo soporta columnas numéricas y el "
                    "DataFrame no contiene columnas numéricas."
                )
        if self.capabilities.supports_categorical and not self.capabilities.supports_numeric:
            if not has_categorical:
                raise ValueError(
                    f"El método '{self.name}' solo soporta columnas categóricas y el "
                    "DataFrame no contiene columnas categóricas."
                )

    @abstractmethod
    def fit(self, df: pd.DataFrame) -> "ImputationMethod":
        """Ajusta el método sobre el DataFrame (estadísticos o modelos)."""

    @abstractmethod
    def _apply(self, result: pd.DataFrame) -> pd.DataFrame:
        """Aplica la imputación sobre una copia (positional-safe)."""

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Devuelve una copia del DataFrame con la imputación aplicada."""
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Se requiere un DataFrame de pandas.")
        if not self._fitted:
            self.fit(df)
        result = df.copy()
        return self._apply(result)

    def impute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Valida, ajusta y devuelve una copia imputada del DataFrame."""
        self.validate_input(df)
        return self.transform(df)


# ---------------------------------------------------------------------------
# Métodos de ubicación y constante
# ---------------------------------------------------------------------------


class MeanImputation(ImputationMethod):
    """Imputa con la media de cada columna numérica."""

    name = "media"
    description = "Imputa cada columna numérica con su media aritmética."
    capabilities = MethodCapabilities(supports_numeric=True)

    def fit(self, df: pd.DataFrame) -> "MeanImputation":
        self._means: Dict[Any, float] = {}
        self._orig_dtypes: Dict[Any, Any] = {}
        for col in _numeric_columns_with_missing(df):
            mean = df[col].mean()
            if not pd.isna(mean):
                self._means[col] = float(mean)
            self._orig_dtypes[col] = df[col].dtype
        self._fitted = True
        return self

    def _apply(self, result: pd.DataFrame) -> pd.DataFrame:
        for col, mean in self._means.items():
            result[col] = _restore_integer_rounding(
                result[col].astype(float).fillna(mean), self._orig_dtypes[col]
            )
        return result


class MedianImputation(ImputationMethod):
    """Imputa con la mediana de cada columna numérica."""

    name = "mediana"
    description = "Imputa cada columna numérica con su mediana."
    capabilities = MethodCapabilities(supports_numeric=True)

    def fit(self, df: pd.DataFrame) -> "MedianImputation":
        self._medians: Dict[Any, float] = {}
        self._orig_dtypes: Dict[Any, Any] = {}
        for col in _numeric_columns_with_missing(df):
            median = df[col].median()
            if not pd.isna(median):
                self._medians[col] = float(median)
            self._orig_dtypes[col] = df[col].dtype
        self._fitted = True
        return self

    def _apply(self, result: pd.DataFrame) -> pd.DataFrame:
        for col, median in self._medians.items():
            result[col] = _restore_integer_rounding(
                result[col].astype(float).fillna(median), self._orig_dtypes[col]
            )
        return result


class ModeImputation(ImputationMethod):
    """Imputa con la moda de cada columna (numérica o categórica)."""

    name = "moda"
    description = "Imputa cada columna con su valor modal (numérica o categórica)."
    capabilities = MethodCapabilities(supports_numeric=True, supports_categorical=True)

    def fit(self, df: pd.DataFrame) -> "ModeImputation":
        self._modes: Dict[Any, Any] = {}
        self._orig_dtypes: Dict[Any, Any] = {}
        for col in df.columns:
            series = df[col]
            if not (_is_numeric_series(series) or _is_categorical_series(series)):
                continue
            if not series.isna().any():
                continue
            mode = series.dropna().mode()
            if not mode.empty:
                self._modes[col] = mode.iloc[0]
            self._orig_dtypes[col] = series.dtype
        self._fitted = True
        return self

    def _apply(self, result: pd.DataFrame) -> pd.DataFrame:
        for col, mode in self._modes.items():
            if _is_numeric_series(result[col]):
                result[col] = _restore_integer_rounding(
                    result[col].astype(float).fillna(mode), self._orig_dtypes[col]
                )
            else:
                result[col] = result[col].fillna(mode)
        return result


class ConstantImputation(ImputationMethod):
    """Imputa con un valor constante y opcionalmente añade un indicador de ausencia."""

    name = "constante"
    description = (
        "Imputa con un valor constante y añade una columna indicadora "
        "de qué observaciones fueron imputadas."
    )
    capabilities = MethodCapabilities(supports_numeric=True, supports_categorical=True)

    def __init__(
        self,
        constant_value: Any = 0,
        add_indicator: bool = True,
        random_state: Optional[int] = None,
    ) -> None:
        super().__init__(random_state=random_state)
        self.constant_value = constant_value
        self.add_indicator = add_indicator

    def fit(self, df: pd.DataFrame) -> "ConstantImputation":
        self._targets: List[Any] = []
        self._orig_dtypes: Dict[Any, Any] = {}
        for col in df.columns:
            series = df[col]
            if not (_is_numeric_series(series) or _is_categorical_series(series)):
                continue
            if series.isna().any():
                self._targets.append(col)
                self._orig_dtypes[col] = series.dtype
        self._fitted = True
        return self

    def _apply(self, result: pd.DataFrame) -> pd.DataFrame:
        for col in self._targets:
            was_missing = result[col].isna()
            if self.add_indicator and was_missing.any():
                result[f"{col}_was_missing"] = was_missing.astype(int)
            if _is_numeric_series(result[col]):
                result[col] = _restore_integer_rounding(
                    result[col].astype(float).fillna(self.constant_value),
                    self._orig_dtypes[col],
                )
            else:
                result[col] = result[col].fillna(self.constant_value)
        return result


# ---------------------------------------------------------------------------
# Métodos basados en modelos
# ---------------------------------------------------------------------------


class KNNImputation(ImputationMethod):
    """Imputación por k-vecinos más cercanos (scikit-learn KNNImputer)."""

    name = "knn"
    description = "Imputa valores numéricos con k vecinos más cercanos (KNNImputer)."
    capabilities = MethodCapabilities(supports_numeric=True, needs_other_columns=True)

    def __init__(
        self,
        n_neighbors: int = 5,
        weights: str = "uniform",
        random_state: Optional[int] = None,
    ) -> None:
        super().__init__(random_state=random_state)
        self.n_neighbors = n_neighbors
        self.weights = weights

    def fit(self, df: pd.DataFrame) -> "KNNImputation":
        self._features: List[Any] = [
            col for col in df.columns
            if _is_numeric_series(df[col]) and df[col].notna().sum() > 0
        ]
        self._targets: List[Any] = [
            col for col in self._features if df[col].isna().any()
        ]
        self._orig_dtypes: Dict[Any, Any] = {
            col: df[col].dtype for col in self._features
        }
        self._fitted = True
        return self

    def _apply(self, result: pd.DataFrame) -> pd.DataFrame:
        if not self._targets or not self._features:
            return result
        X = np.array(result[self._features].astype(float).to_numpy(copy=True))
        imputer = KNNImputer(
            n_neighbors=self.n_neighbors, weights=self.weights, keep_empty_features=True
        )
        X_imp = imputer.fit_transform(X)
        for col in self._targets:
            idx = self._features.index(col)
            result[col] = _restore_integer_rounding(
                pd.Series(X_imp[:, idx], index=result.index), self._orig_dtypes[col]
            )
        return result


class IterativeImputation(ImputationMethod):
    """Imputación iterativa multivariada (scikit-learn IterativeImputer)."""

    name = "iterativo"
    description = "Imputación iterativa multivariada por modelos condicionales."
    capabilities = MethodCapabilities(supports_numeric=True, needs_other_columns=True)
    uses_random_state = True

    def __init__(
        self,
        max_iter: int = 10,
        initial_strategy: str = "mean",
        random_state: Optional[int] = None,
    ) -> None:
        super().__init__(random_state=random_state)
        self.max_iter = max_iter
        self.initial_strategy = initial_strategy

    def fit(self, df: pd.DataFrame) -> "IterativeImputation":
        self._features: List[Any] = [
            col for col in df.columns
            if _is_numeric_series(df[col]) and df[col].notna().sum() > 0
        ]
        self._targets: List[Any] = [
            col for col in self._features if df[col].isna().any()
        ]
        self._orig_dtypes: Dict[Any, Any] = {
            col: df[col].dtype for col in self._features
        }
        self._fitted = True
        return self

    def _apply(self, result: pd.DataFrame) -> pd.DataFrame:
        if not self._targets or not self._features:
            return result
        X = np.array(result[self._features].astype(float).to_numpy(copy=True))
        imputer = IterativeImputer(
            max_iter=self.max_iter,
            initial_strategy=self.initial_strategy,
            random_state=self.random_state,
            keep_empty_features=True,
        )
        try:
            X_imp = imputer.fit_transform(X)
        except ValueError as exc:
            raise ValueError(
                f"El método 'iterativo' no pudo ajustarse sobre estos datos "
                f"({exc}). Revise que existan filas con valores observados en "
                "más de una variable."
            ) from exc
        for col in self._targets:
            idx = self._features.index(col)
            result[col] = _restore_integer_rounding(
                pd.Series(X_imp[:, idx], index=result.index), self._orig_dtypes[col]
            )
        return result


class MICEImputation(ImputationMethod):
    """Imputación múltiple por ecuaciones encadenadas (statsmodels MICEData)."""

    name = "mice"
    description = "Imputación múltiple por ecuaciones encadenadas (MICE) con statsmodels."
    capabilities = MethodCapabilities(supports_numeric=True, needs_other_columns=True)
    uses_random_state = True

    def __init__(self, n_iter: int = 10, random_state: Optional[int] = None) -> None:
        super().__init__(random_state=random_state)
        self.n_iter = n_iter

    def fit(self, df: pd.DataFrame) -> "MICEImputation":
        self._numeric_cols: List[Any] = [
            col for col in df.columns if _is_numeric_series(df[col])
        ]
        self._targets: List[Any] = [
            col for col in self._numeric_cols
            if df[col].isna().any() and df[col].notna().sum() > 0
        ]
        self._orig_dtypes: Dict[Any, Any] = {
            col: df[col].dtype for col in self._numeric_cols
        }
        if self._numeric_cols:
            empty_rows = df[self._numeric_cols].isna().all(axis=1)
            if empty_rows.any():
                raise ValueError(
                    "El método 'mice' no puede procesar filas completamente vacías "
                    "en las columnas numéricas; imputelas antes con 'media' o elimine esas filas."
                )
        self._fitted = True
        return self

    def _apply(self, result: pd.DataFrame) -> pd.DataFrame:
        if not self._targets:
            return result
        data = result[self._numeric_cols].astype(float).copy()
        data.columns = [str(c) for c in data.columns]
        if self.random_state is not None:
            np.random.seed(self.random_state)
        mice = MICEData(data)
        mice.update_all(n_iter=self.n_iter)
        imputed = mice.data
        for i, col in enumerate(self._numeric_cols):
            if col in self._targets:
                result[col] = _restore_integer_rounding(
                    pd.Series(imputed.iloc[:, i].to_numpy(), index=result.index),
                    self._orig_dtypes[col],
                )
        return result


class RegressionImputation(ImputationMethod):
    """Imputación por regresión lineal sobre columnas numéricas completas."""

    name = "regresion"
    description = (
        "Predice cada valor faltante con regresión lineal usando las columnas "
        "numéricas completas como predictoras."
    )
    capabilities = MethodCapabilities(supports_numeric=True, needs_other_columns=True)

    def fit(self, df: pd.DataFrame) -> "RegressionImputation":
        self._targets: List[Any] = []
        self._predictors: List[Any] = []
        self._orig_dtypes: Dict[Any, Any] = {}
        for col in df.columns:
            if not _is_numeric_series(df[col]):
                continue
            self._orig_dtypes[col] = df[col].dtype
            if df[col].isna().any():
                if df[col].notna().sum() > 0:
                    self._targets.append(col)
            elif not df[col].isna().any():
                self._predictors.append(col)
        if self._targets and not self._predictors:
            raise ValueError(
                "El método 'regresion' requiere al menos una columna numérica "
                "completa para usarse como predictora."
            )
        self._fitted = True
        return self

    def _apply(self, result: pd.DataFrame) -> pd.DataFrame:
        if not self._targets or not self._predictors:
            return result
        predictors = result[self._predictors].astype(float).to_numpy(copy=True)
        for col in self._targets:
            values = result[col].to_numpy(dtype=float, copy=True)
            missing = pd.isna(values)
            if missing.sum() == 0 or (~missing).sum() < 2:
                continue
            model = LinearRegression()
            model.fit(predictors[~missing], values[~missing])
            values[missing] = model.predict(predictors[missing])
            result[col] = _restore_integer_rounding(
                pd.Series(values, index=result.index), self._orig_dtypes[col]
            )
        return result


# ---------------------------------------------------------------------------
# Métodos temporales
# ---------------------------------------------------------------------------


class LinearInterpolationImputation(ImputationMethod):
    """Interpolación lineal entre valores observados (requiere orden temporal)."""

    name = "interpolacion_lineal"
    description = (
        "Interpola linealmente entre valores observados siguiendo el orden de las filas."
    )
    capabilities = MethodCapabilities(supports_numeric=True, temporal_only=True)

    def fit(self, df: pd.DataFrame) -> "LinearInterpolationImputation":
        self._targets: List[Any] = []
        self._orig_dtypes: Dict[Any, Any] = {}
        for col in _numeric_columns_with_missing(df):
            if df[col].notna().sum() >= 2:
                self._targets.append(col)
                self._orig_dtypes[col] = df[col].dtype
        self._fitted = True
        return self

    def _apply(self, result: pd.DataFrame) -> pd.DataFrame:
        for col in self._targets:
            result[col] = _restore_integer_rounding(
                result[col].interpolate(method="linear", limit_direction="both"),
                self._orig_dtypes[col],
            )
        return result


class LOCFImputation(ImputationMethod):
    """Last Observation Carried Forward con relleno hacia atrás para el inicio."""

    name = "locf"
    description = (
        "Lleva hacia adelante la última observación (LOCF) y rellena hacia atrás "
        "las observaciones iniciales sin precedente."
    )
    capabilities = MethodCapabilities(supports_numeric=True, temporal_only=True)

    def fit(self, df: pd.DataFrame) -> "LOCFImputation":
        self._targets: List[Any] = []
        self._orig_dtypes: Dict[Any, Any] = {}
        for col in _numeric_columns_with_missing(df):
            if df[col].notna().sum() >= 1:
                self._targets.append(col)
                self._orig_dtypes[col] = df[col].dtype
        self._fitted = True
        return self

    def _apply(self, result: pd.DataFrame) -> pd.DataFrame:
        for col in self._targets:
            result[col] = _restore_integer_rounding(
                result[col].ffill().bfill(), self._orig_dtypes[col]
            )
        return result