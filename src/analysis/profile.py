"""
Perfil estructural de un conjunto de datos (DatasetProfile).

Describe la información de alto nivel que los módulos de selección de métodos
(por ejemplo, la selección de métodos de imputación de E5) necesitan: tamaño
muestral, tipos de variables, estructura temporal e identificadores.

Los tipos de variable provienen de `DataValidator.identify_variable_types`:
  - "Binaria"
  - "Cuantitativa discreta"
  - "Cuantitativa continua"
  - "Variable de tiempo"
  - "Cualitativa nominal"
"""

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

import pandas as pd

from src.data.validator import DataValidator
from src.utils.logger import setup_logger

logger = setup_logger("DatasetProfile")

NUMERIC_TYPES = frozenset({"Cuantitativa discreta", "Cuantitativa continua"})
CATEGORICAL_TYPES = frozenset({"Binaria", "Cualitativa nominal"})
DATETIME_TYPE = "Variable de tiempo"

PROFILE_VERSION = "1.0"


@dataclass(frozen=True)
class DatasetProfile:
    """Descripción estructural y reproducible de un conjunto de datos."""

    n_observations: int
    n_variables: int
    variable_types: Dict[str, str]
    temporal: bool
    datetime_columns: List[str]
    identifier_columns: List[str]
    target_variable: Optional[str]
    reproduction: Dict[str, Any]

    def is_numeric(self, variable: Any) -> bool:
        """True si la variable es numérica (discreta o continua)."""
        return self.variable_types.get(variable) in NUMERIC_TYPES

    def is_categorical(self, variable: Any) -> bool:
        """True si la variable es categórica (binaria o nominal)."""
        return self.variable_types.get(variable) in CATEGORICAL_TYPES

    def is_datetime(self, variable: Any) -> bool:
        """True si la variable es de tiempo."""
        return self.variable_types.get(variable) == DATETIME_TYPE

    def to_dict(self) -> Dict[str, Any]:
        """Devuelve el perfil como diccionario JSON-serializable."""
        return asdict(self)


def build_dataset_profile(
    df: pd.DataFrame,
    *,
    target: Optional[Any] = None,
    temporal: Optional[bool] = None,
    datetime_columns: Optional[List[Any]] = None,
    identifier_columns: Optional[List[Any]] = None,
) -> DatasetProfile:
    """
    Construye un DatasetProfile a partir de un DataFrame.

    Args:
        df: DataFrame de datos.
        target: Variable objetivo del análisis, si existe.
        temporal: Indica si los datos tienen orden temporal. Si es None, se
            infiere: hay estructura temporal si existe al menos una columna de
            fecha/hora.
        datetime_columns: Columnas de fecha/hora. Si es None, se detectan por
            dtype.
        identifier_columns: Columnas identificadoras (una por fila). Si es None,
            se detectan heurísticamente: columnas de texto cuyo número de valores
            únicos coincide con el número de observaciones.

    Returns:
        DatasetProfile.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Se requiere un DataFrame de pandas.")

    variable_types = DataValidator.identify_variable_types(df)

    if datetime_columns is None:
        datetime_columns = [
            col
            for col in df.columns
            if pd.api.types.is_datetime64_any_dtype(df[col].dtype)
        ]
    datetime_columns = [str(c) for c in datetime_columns]

    if identifier_columns is None:
        n = len(df)
        identifier_columns = []
        for col in df.columns:
            series = df[col].dropna()
            if n > 0 and series.nunique() == n and not pd.api.types.is_numeric_dtype(df[col].dtype):
                identifier_columns.append(str(col))
        identifier_auto = True
    else:
        identifier_columns = [str(c) for c in identifier_columns]
        identifier_auto = False

    temporal_inferred = temporal is None
    if temporal is None:
        temporal = bool(datetime_columns)

    reproduction = {
        "module": "src.analysis.profile",
        "version": PROFILE_VERSION,
        "n_observations": len(df),
        "n_variables": df.shape[1],
        "identifier_detection": "auto" if identifier_auto else "explícito",
        "temporal_inference": temporal_inferred,
    }

    logger.info(
        f"Perfil construido: {len(df)} filas, {df.shape[1]} variables, "
        f"temporal={temporal}, identificadores={identifier_columns}."
    )

    return DatasetProfile(
        n_observations=len(df),
        n_variables=df.shape[1],
        variable_types={str(c): t for c, t in variable_types.items()},
        temporal=bool(temporal),
        datetime_columns=datetime_columns,
        identifier_columns=identifier_columns,
        target_variable=None if target is None else str(target),
        reproduction=reproduction,
    )
