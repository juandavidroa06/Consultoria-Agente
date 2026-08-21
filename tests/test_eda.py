"""
Pruebas unitarias para las funciones de EDA.
"""

import pytest
import pandas as pd
import numpy as np
from src.analysis.eda import (
    describe_numerical,
    describe_categorical,
    detect_outliers_iqr,
    calculate_correlation_matrix,
)


@pytest.fixture
def eda_df():
    np.random.seed(42)
    return pd.DataFrame({
        "num1": [10, 12, 11, 13, 100],  # 100 es outlier
        "num2": [5, 6, 7, 8, 9],
        "cat1": ["Grupo 1", "Grupo 1", "Grupo 2", "Grupo 2", "Grupo 1"],
    })


def test_describe_numerical(eda_df):
    num_summary = describe_numerical(eda_df)
    assert "num1" in num_summary.index
    assert "mean" in num_summary.columns
    assert "median" in num_summary.columns
    assert num_summary.loc["num2", "mean"] == 7.0


def test_describe_categorical(eda_df):
    cat_summary = describe_categorical(eda_df)
    assert "cat1" in cat_summary
    assert cat_summary["cat1"].loc["Grupo 1", "frecuencia"] == 3


def test_detect_outliers_iqr(eda_df):
    outliers = detect_outliers_iqr(eda_df, columns=["num1"])
    assert outliers["num1"]["outlier_count"] == 1
    assert 4 in outliers["num1"]["outlier_indices"]  # Índice de 100


def test_calculate_correlation_matrix(eda_df):
    corr = calculate_correlation_matrix(eda_df)
    assert corr.shape == (2, 2)
    assert "num1" in corr.columns
