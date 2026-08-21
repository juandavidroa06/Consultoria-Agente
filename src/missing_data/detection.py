"""
Módulo de detección de datos faltantes (Etapa E1).

Detecta valores faltantes reales (NaN, None, pd.NA y equivalentes reconocidos por pandas)
y placeholders textuales candidatos, diferenciándolos claramente en el reporte.
La conversión de placeholders a NA es una operación explícita y configurable.
"""

from dataclasses import asdict, dataclass
from typing import Any, Dict, FrozenSet, List, Optional, Union

import numpy as np
import pandas as pd

from src.data.validator import DataValidator
from src.utils.logger import setup_logger

logger = setup_logger("MissingDataDetector")

DEFAULT_PLACEHOLDERS = frozenset({"n/a", "na", "?", "-", "", "unknown"})


@dataclass
class MissingVariableInfo:
    """Información de datos faltantes para una variable individual."""

    variable: str
    variable_type: str
    dtype: str
    missing_count: int
    missing_percentage: float
    placeholder_count: int
    missing_grade: str

    def to_dict(self) -> Dict[str, Any]:
        """Devuelve la información como diccionario JSON-serializable."""
        return asdict(self)


@dataclass
class MissingReport:
    """Reporte estructurado y reproducible de datos faltantes de un DataFrame."""

    status: str
    total_observations: int
    total_variables: int
    total_missing_values: int
    overall_missing_percentage: float
    overall_missing_grade: str
    complete_cases: int
    complete_cases_percentage: float
    incomplete_cases: int
    rows_completely_empty: int
    variables_with_missing: List[str]
    variables_without_missing: List[str]
    by_variable: Dict[str, MissingVariableInfo]
    observations_missing_distribution: Dict[int, int]
    total_placeholders: int
    has_placeholders: bool
    reproduction: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Devuelve el reporte como diccionario JSON-serializable."""
        return asdict(self)


class MissingDataDetector:
    """
    Detector robusto y reproducible de datos faltantes.

    Clasifica los valores faltantes reales (NaN, None, pd.NA y equivalentes reconocidos
    por pandas) de forma separada de los placeholders textuales candidatos (p. ej. "N/A",
    "-", cadena vacía). Nunca trata valores numéricos, "0", "no" ni "false" como faltantes.
    """

    DEFAULT_PLACEHOLDERS = DEFAULT_PLACEHOLDERS

    def __init__(
        self,
        placeholder_tokens: Optional[Union[List[str], FrozenSet[str]]] = None,
        detect_placeholders: bool = True,
        random_state: Optional[int] = None,
    ) -> None:
        tokens = self.DEFAULT_PLACEHOLDERS if placeholder_tokens is None else set(placeholder_tokens)
        self.placeholder_tokens = frozenset(str(t).strip().lower() for t in tokens)
        self.detect_placeholders = detect_placeholders
        self.random_state = random_state

    def detect(self, df: pd.DataFrame) -> MissingReport:
        """
        Detecta y resume los datos faltantes del DataFrame.

        Args:
            df: DataFrame de pandas a analizar.

        Returns:
            MissingReport con resumen estructurado de datos faltantes.

        Raises:
            TypeError: Si df no es un DataFrame de pandas.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Se requiere un DataFrame de pandas.")

        logger.info(
            f"Detectando datos faltantes en DataFrame de {df.shape[0]} filas x {df.shape[1]} columnas."
        )

        total_observations = int(df.shape[0])
        total_variables = int(df.shape[1])

        missing_mask = df.isna()
        missing_counts = missing_mask.sum()

        total_missing = int(missing_counts.sum())
        total_cells = total_observations * total_variables
        overall_pct = round(total_missing / total_cells * 100, 2) if total_cells > 0 else 0.0

        per_row_missing = missing_mask.sum(axis=1)
        complete_cases = int((per_row_missing == 0).sum())
        incomplete_cases = int((per_row_missing >= 1).sum())
        rows_completely_empty = (
            int((per_row_missing == total_variables).sum()) if total_variables > 0 else 0
        )
        complete_pct = (
            round(complete_cases / total_observations * 100, 2) if total_observations > 0 else 0.0
        )

        distribution = per_row_missing.value_counts().sort_index()
        observations_missing_distribution = {int(k): int(v) for k, v in distribution.items()}

        variable_types = DataValidator.identify_variable_types(df)

        variables_with_missing: List[str] = []
        variables_without_missing: List[str] = []
        by_variable: Dict[str, MissingVariableInfo] = {}
        total_placeholders = 0

        for col in df.columns:
            series = df[col]
            count = int(missing_counts[col])
            pct = round(count / total_observations * 100, 2) if total_observations > 0 else 0.0
            placeholder_count = self._count_placeholders(series) if self.detect_placeholders else 0
            total_placeholders += placeholder_count

            by_variable[col] = MissingVariableInfo(
                variable=str(col),
                variable_type=variable_types.get(col, "Desconocido"),
                dtype=str(series.dtype),
                missing_count=count,
                missing_percentage=pct,
                placeholder_count=placeholder_count,
                missing_grade=_classify_missing_percentage(pct),
            )

            if count > 0:
                variables_with_missing.append(str(col))
            else:
                variables_without_missing.append(str(col))

        status = "con_faltantes" if total_missing > 0 else "sin_faltantes"

        reproduction = {
            "random_state": self.random_state,
            "placeholder_tokens": sorted(self.placeholder_tokens) if self.detect_placeholders else [],
            "pandas_version": pd.__version__,
            "numpy_version": np.__version__,
            "module": "src.missing_data.detection",
            "version": "1.0",
        }

        report = MissingReport(
            status=status,
            total_observations=total_observations,
            total_variables=total_variables,
            total_missing_values=total_missing,
            overall_missing_percentage=overall_pct,
            overall_missing_grade=_classify_missing_percentage(overall_pct),
            complete_cases=complete_cases,
            complete_cases_percentage=complete_pct,
            incomplete_cases=incomplete_cases,
            rows_completely_empty=rows_completely_empty,
            variables_with_missing=variables_with_missing,
            variables_without_missing=variables_without_missing,
            by_variable=by_variable,
            observations_missing_distribution=observations_missing_distribution,
            total_placeholders=total_placeholders,
            has_placeholders=total_placeholders > 0,
            reproduction=reproduction,
        )

        logger.info(
            f"Detección completada: {total_missing} faltantes reales, "
            f"{total_placeholders} placeholders candidatos."
        )
        return report

    def convert_placeholders_to_na(
        self, df: pd.DataFrame, columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Devuelve una copia del DataFrame con los placeholders textuales convertidos a pd.NA.

        Operación explícita y configurable: no modifica el DataFrame original.

        Args:
            df: DataFrame de pandas.
            columns: Columnas de texto a procesar (por defecto todas las de tipo texto).

        Returns:
            Copia del DataFrame con los placeholders reemplazados por pd.NA.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Se requiere un DataFrame de pandas.")

        result = df.copy()
        cols = list(df.columns) if columns is None else columns

        for col in cols:
            if col not in df.columns:
                raise KeyError(f"La columna '{col}' no existe en el DataFrame.")
            series = df[col]
            if not self._is_text_like(series):
                continue
            mask = series.map(self._is_placeholder)
            result.loc[mask, col] = pd.NA

        return result

    def _is_placeholder(self, value: Any) -> bool:
        if isinstance(value, str):
            return value.strip().lower() in self.placeholder_tokens
        return False

    def _count_placeholders(self, series: pd.Series) -> int:
        if not self._is_text_like(series):
            return 0
        return sum(1 for value in series if self._is_placeholder(value))

    def _is_text_like(self, series: pd.Series) -> bool:
        return pd.api.types.is_string_dtype(series.dtype) or isinstance(
            series.dtype, pd.CategoricalDtype
        )


def convert_placeholders_to_na(
    df: pd.DataFrame,
    placeholder_tokens: Optional[List[str]] = None,
    columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Función de conveniencia para convertir placeholders a NA de forma explícita."""
    return MissingDataDetector(placeholder_tokens=placeholder_tokens).convert_placeholders_to_na(
        df, columns=columns
    )


def _classify_missing_percentage(pct: float) -> str:
    """Clasificación descriptiva del porcentaje de faltantes. No es criterio de selección de métodos."""
    if pct == 0.0:
        return "Sin faltantes"
    if pct < 1.0:
        return "Muy baja"
    if pct < 5.0:
        return "Baja"
    if pct < 20.0:
        return "Moderada"
    if pct < 50.0:
        return "Alta"
    return "Muy alta"