"""
Pruebas unitarias para el módulo de visualizaciones estadísticas.
"""

import pytest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.visualization.plots import (
    plot_histogram,
    plot_boxplot,
    plot_scatter,
    plot_qq,
    plot_correlation_matrix,
)


@pytest.fixture
def plot_df():
    np.random.seed(42)
    return pd.DataFrame({
        "x": np.random.normal(10, 2, 30),
        "y": np.random.normal(5, 1, 30),
        "grupo": ["A"] * 15 + ["B"] * 15,
    })


def test_plot_histogram(plot_df):
    fig = plot_histogram(plot_df["x"], column_name="Variable X")
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_boxplot(plot_df):
    fig = plot_boxplot(plot_df, y_col="x", x_col="grupo")
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_scatter(plot_df):
    fig = plot_scatter(plot_df, x_col="x", y_col="y")
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_qq(plot_df):
    fig = plot_qq(plot_df["x"], column_name="Variable X")
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_correlation_matrix(plot_df):
    corr = plot_df[["x", "y"]].corr()
    fig = plot_correlation_matrix(corr)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)
