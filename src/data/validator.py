"""
Módulo para la validación, identificación de tipos de variables y diagnóstico de calidad de datos.
"""

from typing import Dict, Any, List
import pandas as pd
import numpy as np
from src.utils.logger import setup_logger

logger = setup_logger("DataValidator")


class DataValidator:
    """
    Clase encargada de analizar la calidad de los datos, identificar tipos de variables,
    contabilizar valores faltantes y detectar filas duplicadas.
    """

    @staticmethod
    def identify_variable_types(df: pd.DataFrame) -> Dict[str, str]:
        """
        Clasifica cada columna del DataFrame según su tipo estadístico:
        - Binaria
        - Cuantitativa discreta
        - Cuantitativa continua
        - Cualitativa nominal / categórica

        Args:
            df: DataFrame de datos.

        Returns:
            Dict con el nombre de la columna y su clasificación estadística.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Se requiere un DataFrame de pandas.")

        types = {}
        for col in df.columns:
            series = df[col].dropna()
            n_unique = series.nunique()
            n_non_null = df[col].notna().sum()

            if n_non_null == 0:
                types[col] = "Sin datos"
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                types[col] = "Variable de tiempo"
            elif n_unique == 2:
                types[col] = "Binaria"
            elif pd.api.types.is_numeric_dtype(df[col]):
                # Si todos los valores son enteros o el número de únicos es pequeño
                if pd.api.types.is_integer_dtype(df[col]) or (n_unique < 15 and (series % 1 == 0).all()):
                    types[col] = "Cuantitativa discreta"
                else:
                    types[col] = "Cuantitativa continua"
            else:
                types[col] = "Cualitativa nominal"

        return types

    @staticmethod
    def detect_missing_values(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detecta el número y porcentaje de valores faltantes por columna y en todo el DataFrame.

        Args:
            df: DataFrame de datos.

        Returns:
            Dict con la cantidad de faltantes por columna, porcentajes y totales.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Se requiere un DataFrame de pandas.")

        missing_counts = df.isnull().sum()
        total_cells = df.size
        total_missing = missing_counts.sum()

        columns_info = {}
        for col in df.columns:
            count = int(missing_counts[col])
            pct = float((count / len(df)) * 100) if len(df) > 0 else 0.0
            columns_info[col] = {
                "missing_count": count,
                "missing_percentage": round(pct, 2),
            }

        return {
            "total_missing_values": int(total_missing),
            "overall_missing_percentage": round(float((total_missing / total_cells) * 100), 2) if total_cells > 0 else 0.0,
            "columns": columns_info,
        }

    @staticmethod
    def detect_duplicates(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detecta la cantidad y porcentaje de filas duplicadas en el DataFrame.

        Args:
            df: DataFrame de datos.

        Returns:
            Dict con 'duplicate_count' y 'duplicate_percentage'.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Se requiere un DataFrame de pandas.")

        num_duplicates = int(df.duplicated().sum())
        total_rows = len(df)
        pct = float((num_duplicates / total_rows) * 100) if total_rows > 0 else 0.0

        return {
            "duplicate_count": num_duplicates,
            "duplicate_percentage": round(pct, 2),
        }

    @classmethod
    def summarize_data_quality(cls, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Genera un resumen integral de calidad de datos.

        Args:
            df: DataFrame de datos.

        Returns:
            Dict con dimensiones, tipos de variable, faltantes y duplicados.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Se requiere un DataFrame de pandas.")

        logger.info("Generando resumen de calidad de datos.")

        return {
            "dimensions": {
                "rows": int(df.shape[0]),
                "columns": int(df.shape[1]),
            },
            "variable_types": cls.identify_variable_types(df),
            "missing_values": cls.detect_missing_values(df),
            "duplicates": cls.detect_duplicates(df),
        }
