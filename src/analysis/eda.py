"""
Módulo para análisis exploratorio de datos (EDA).
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from scipy import stats
from src.utils.logger import setup_logger

logger = setup_logger("EDA")


def describe_numerical(
    df: pd.DataFrame, columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Calcula estadísticas descriptivas para variables cuantitativas:
    conteo, media, desviación estándar, mediana, IQR, min, max, asimetría y curtosis.

    Args:
        df: DataFrame de datos.
        columns: Lista opcional de nombres de columnas numéricas a analizar.

    Returns:
        pd.DataFrame con los estadísticos calculados por variable.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Se requiere un DataFrame de pandas.")

    if columns is not None:
        non_numeric = [
            col for col in columns if not pd.api.types.is_numeric_dtype(df[col])
        ]
        if non_numeric:
            raise ValueError(
                "Las siguientes columnas no son numéricas: " + ", ".join(non_numeric)
            )

    num_df = df.select_dtypes(include=[np.number]) if columns is None else df[columns]

    if num_df.empty:
        logger.warning("No se encontraron columnas numéricas para analizar.")
        return pd.DataFrame()

    summary = []
    for col in num_df.columns:
        series = num_df[col].dropna()
        if series.empty:
            continue

        q25 = series.quantile(0.25)
        q75 = series.quantile(0.75)
        iqr = q75 - q25

        summary.append({
            "variable": col,
            "count": int(series.count()),
            "mean": float(series.mean()),
            "std": float(series.std()) if len(series) > 1 else float("nan"),
            "median": float(series.median()),
            "iqr": float(iqr),
            "min": float(series.min()),
            "max": float(series.max()),
            "skewness": float(series.skew()) if len(series) > 2 else float("nan"),
            "kurtosis": float(series.kurtosis()) if len(series) > 3 else float("nan"),
        })

    if not summary:
        logger.warning("No se encontraron columnas numéricas para analizar.")
        return pd.DataFrame()

    result_df = pd.DataFrame(summary).set_index("variable")
    return result_df


def describe_categorical(
    df: pd.DataFrame, columns: Optional[List[str]] = None
) -> Dict[str, pd.DataFrame]:
    """
    Calcula frecuencias absolutas y relativas para variables cualitativas / categóricas.

    Args:
        df: DataFrame de datos.
        columns: Lista opcional de columnas categóricas a analizar.

    Returns:
        Dict con el nombre de cada columna y su tabla de frecuencias en un DataFrame.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Se requiere un DataFrame de pandas.")

    cat_df = df.select_dtypes(exclude=[np.number]) if columns is None else df[columns]

    results = {}
    for col in cat_df.columns:
        series = cat_df[col].dropna()
        counts = series.value_counts()
        percentages = (series.value_counts(normalize=True) * 100).round(2)

        freq_table = pd.DataFrame({
            "frecuencia": counts,
            "porcentaje": percentages,
        })
        results[col] = freq_table

    return results


def detect_outliers_iqr(
    df: pd.DataFrame, columns: Optional[List[str]] = None, factor: float = 1.5
) -> Dict[str, Any]:
    """
    Detecta valores atípicos (outliers) utilizando el método del Rango Intercuartílico (IQR).

    Args:
        df: DataFrame de datos.
        columns: Lista opcional de columnas a analizar.
        factor: Factor multiplicador del IQR (por defecto 1.5 para atípicos moderados).

    Returns:
        Dict con cantidad de atípicos, índices y límites inferior y superior por columna.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Se requiere un DataFrame de pandas.")

    num_cols = df.select_dtypes(include=[np.number]).columns if columns is None else columns

    outlier_info = {}
    for col in num_cols:
        series = df[col].dropna()
        if series.empty:
            continue

        q25 = series.quantile(0.25)
        q75 = series.quantile(0.75)
        iqr = q75 - q25

        lower_bound = q25 - (factor * iqr)
        upper_bound = q75 + (factor * iqr)

        outliers_mask = (series < lower_bound) | (series > upper_bound)
        outlier_indices = series[outliers_mask].index.tolist()

        outlier_info[col] = {
            "outlier_count": int(outliers_mask.sum()),
            "outlier_percentage": round(float((outliers_mask.sum() / len(series)) * 100), 2),
            "lower_bound": float(lower_bound),
            "upper_bound": float(upper_bound),
            "outlier_indices": outlier_indices,
        }

    return outlier_info


def calculate_correlation_matrix(
    df: pd.DataFrame, columns: Optional[List[str]] = None, method: str = "pearson"
) -> pd.DataFrame:
    """
    Calcula la matriz de correlación para variables cuantitativas.

    Args:
        df: DataFrame de datos.
        columns: Lista opcional de columnas numéricas.
        method: Método de correlación ('pearson', 'spearman', 'kendall').

    Returns:
        pd.DataFrame con la matriz de correlación.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Se requiere un DataFrame de pandas.")

    num_df = df.select_dtypes(include=[np.number]) if columns is None else df[columns]

    if num_df.empty or num_df.shape[1] < 2:
        logger.warning("Se requieren al menos 2 columnas numéricas para calcular la matriz de correlación.")
        return pd.DataFrame()

    corr_matrix = num_df.corr(method=method)
    return corr_matrix
