"""
Registro de métodos de imputación (Etapa E3).

El registro es explícito: los métodos deben registrarse con `register()` antes
de poder obtenerse con `get()`. Se provee un registro por defecto con los
métodos incluidos en el módulo `methods`.
"""

from typing import Dict, List, Type

import pandas as pd

from src.missing_data.methods import (
    ConstantImputation,
    ImputationMethod,
    IterativeImputation,
    KNNImputation,
    LinearInterpolationImputation,
    LOCFImputation,
    MICEImputation,
    MeanImputation,
    MedianImputation,
    ModeImputation,
    RegressionImputation,
    _is_categorical_series,
    _is_numeric_series,
)

DEFAULT_METHOD_NAMES = [
    "media",
    "mediana",
    "moda",
    "constante",
    "knn",
    "iterativo",
    "mice",
    "regresion",
    "interpolacion_lineal",
    "locf",
]


class ImputationRegistry:
    """Registro explícito de métodos de imputación."""

    def __init__(self) -> None:
        self._methods: Dict[str, Type[ImputationMethod]] = {}

    def register(
        self,
        name: str,
        method_cls: Type[ImputationMethod],
        *,
        overwrite: bool = False,
    ) -> None:
        """Registra una clase de método bajo un nombre."""
        if not isinstance(name, str) or not name.strip():
            raise ValueError("El nombre del método debe ser una cadena no vacía.")
        name = name.strip()
        if not (isinstance(method_cls, type) and issubclass(method_cls, ImputationMethod)):
            raise TypeError("method_cls debe ser una subclase de ImputationMethod.")
        if name in self._methods and not overwrite:
            raise ValueError(
                f"El método '{name}' ya está registrado; use overwrite=True para reemplazarlo."
            )
        self._methods[name] = method_cls

    def get(self, name: str) -> Type[ImputationMethod]:
        """Devuelve la clase del método registrado bajo `name`."""
        if name not in self._methods:
            raise KeyError(
                f"El método '{name}' no está registrado. "
                f"Métodos disponibles: {sorted(self.names())}"
            )
        return self._methods[name]

    def names(self) -> List[str]:
        """Devuelve los nombres de los métodos registrados, en orden de registro."""
        return list(self._methods)

    def candidates_for(self, df: pd.DataFrame, *, temporal: bool = False) -> List[str]:
        """
        Devuelve los métodos compatibles con los tipos de variables del DataFrame
        que tienen al menos un dato faltante.

        Los métodos temporales solo se ofrecen si `temporal=True`.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Se requiere un DataFrame de pandas.")
        if not df.isna().any().any():
            return []
        has_numeric = any(_is_numeric_series(df[col]) for col in df.columns)
        has_categorical = any(_is_categorical_series(df[col]) for col in df.columns)

        result: List[str] = []
        for name, method_cls in self._methods.items():
            caps = method_cls.capabilities
            if caps.temporal_only and not temporal:
                continue
            if caps.supports_numeric and has_numeric:
                result.append(name)
            elif caps.supports_categorical and has_categorical:
                result.append(name)
        return result

    def summary(self) -> List[Dict[str, object]]:
        """Resumen de los métodos registrados con sus capacidades."""
        return [
            {
                "name": name,
                "description": method_cls.description,
                "supports_numeric": method_cls.capabilities.supports_numeric,
                "supports_categorical": method_cls.capabilities.supports_categorical,
                "temporal_only": method_cls.capabilities.temporal_only,
                "needs_other_columns": method_cls.capabilities.needs_other_columns,
                "uses_random_state": method_cls.uses_random_state,
            }
            for name, method_cls in self._methods.items()
        ]


default_registry = ImputationRegistry()

for _name, _cls in zip(
    DEFAULT_METHOD_NAMES,
    [
        MeanImputation,
        MedianImputation,
        ModeImputation,
        ConstantImputation,
        KNNImputation,
        IterativeImputation,
        MICEImputation,
        RegressionImputation,
        LinearInterpolationImputation,
        LOCFImputation,
    ],
):
    default_registry.register(_name, _cls)


def register(
    name: str, method_cls: Type[ImputationMethod], *, overwrite: bool = False
) -> None:
    """Registra un método en el registro por defecto."""
    default_registry.register(name, method_cls, overwrite=overwrite)


def get(name: str) -> Type[ImputationMethod]:
    """Obtiene un método del registro por defecto."""
    return default_registry.get(name)


def names() -> List[str]:
    """Nombres de los métodos del registro por defecto."""
    return default_registry.names()


def candidates_for(df: pd.DataFrame, *, temporal: bool = False) -> List[str]:
    """Métodos del registro por defecto compatibles con el DataFrame."""
    return default_registry.candidates_for(df, temporal=temporal)


def summary() -> List[Dict[str, object]]:
    """Resumen de los métodos del registro por defecto."""
    return default_registry.summary()