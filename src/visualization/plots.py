"""
Módulo para generación de gráficos estadísticos en matplotlib y seaborn.
"""

from typing import Optional, Union
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats
from src.utils.logger import setup_logger

logger = setup_logger("Visualization")

# Configuración del estilo visual académico
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")


def plot_histogram(
    data: Union[pd.Series, np.ndarray, list],
    column_name: str = "Variable",
    bins: int = 20,
    kde: bool = True,
) -> plt.Figure:
    """
    Genera un histograma con estimación de densidad de kernel (KDE) opcional.

    Args:
        data: Serie o vector de datos numéricos.
        column_name: Nombre de la variable para etiquetas.
        bins: Número de intervalos (bins).
        kde: Si se incluye la curva de densidad KDE.

    Returns:
        Objeto matplotlib.figure.Figure.
    """
    series = pd.Series(data).dropna()
    fig, ax = plt.subplots(figsize=(8, 5))

    sns.histplot(series, bins=bins, kde=kde, ax=ax, color="#1f77b4", edgecolor="black", alpha=0.6)
    ax.set_title(f"Distribución de {column_name}", fontsize=13, fontweight="bold")
    ax.set_xlabel(column_name, fontsize=11)
    ax.set_ylabel("Frecuencia", fontsize=11)
    fig.tight_layout()

    return fig


def plot_boxplot(
    df: pd.DataFrame,
    y_col: str,
    x_col: Optional[str] = None,
    title: Optional[str] = None,
) -> plt.Figure:
    """
    Genera un diagrama de caja (Boxplot) para una variable continua, opcionalmente agrupada por una categórica.

    Args:
        df: DataFrame de datos.
        y_col: Nombre de la columna continua.
        x_col: Nombre de la columna categórica para agrupación opcional.
        title: Título opcional del gráfico.

    Returns:
        Objeto matplotlib.figure.Figure.
    """
    if y_col not in df.columns:
        raise KeyError(f"La columna '{y_col}' no existe en el DataFrame.")

    fig, ax = plt.subplots(figsize=(8, 5))

    if x_col:
        sns.boxplot(data=df, x=x_col, y=y_col, hue=x_col, legend=False, ax=ax, palette="Set2")
    else:
        sns.boxplot(data=df, y=y_col, ax=ax, color="#2ca02c")

    chart_title = title or (f"Boxplot de {y_col}" + (f" por {x_col}" if x_col else ""))
    ax.set_title(chart_title, fontsize=13, fontweight="bold")
    ax.set_ylabel(y_col, fontsize=11)
    if x_col:
        ax.set_xlabel(x_col, fontsize=11)

    fig.tight_layout()
    return fig


def plot_scatter(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: Optional[str] = None,
) -> plt.Figure:
    """
    Genera un gráfico de dispersión (Scatter Plot) entre dos variables continuas.

    Args:
        df: DataFrame de datos.
        x_col: Nombre de la variable independiente (eje X).
        y_col: Nombre de la variable dependiente (eje Y).
        title: Título opcional.

    Returns:
        Objeto matplotlib.figure.Figure.
    """
    if x_col not in df.columns or y_col not in df.columns:
        raise KeyError("Ambas columnas deben existir en el DataFrame.")

    fig, ax = plt.subplots(figsize=(8, 5))

    sns.scatterplot(data=df, x=x_col, y=y_col, ax=ax, color="#d62728", s=60, alpha=0.8)
    sns.regplot(data=df, x=x_col, y=y_col, ax=ax, scatter=False, color="black", line_kws={"linestyle": "--"})

    chart_title = title or f"Dispersión: {y_col} vs {x_col}"
    ax.set_title(chart_title, fontsize=13, fontweight="bold")
    ax.set_xlabel(x_col, fontsize=11)
    ax.set_ylabel(y_col, fontsize=11)

    fig.tight_layout()
    return fig


def plot_qq(
    data: Union[pd.Series, np.ndarray, list],
    column_name: str = "Variable",
) -> plt.Figure:
    """
    Genera un gráfico cuantil-cuantil (Q-Q Plot) para evaluar la normalidad teórica.

    Args:
        data: Datos numéricos.
        column_name: Nombre de la variable para el título.

    Returns:
        Objeto matplotlib.figure.Figure.
    """
    series = pd.Series(data).dropna()
    fig, ax = plt.subplots(figsize=(7, 5))

    stats.probplot(series, dist="norm", plot=ax)
    ax.set_title(f"Q-Q Plot (Normalidad) — {column_name}", fontsize=13, fontweight="bold")
    ax.get_lines()[0].set_markerfacecolor("#9467bd")
    ax.get_lines()[0].set_markeredgecolor("#9467bd")

    fig.tight_layout()
    return fig


def plot_correlation_matrix(
    corr_matrix: pd.DataFrame,
    title: str = "Matriz de Correlación",
) -> plt.Figure:
    """
    Genera un mapa de calor (Heatmap) para una matriz de correlación.

    Args:
        corr_matrix: DataFrame de matriz de correlación.
        title: Título del gráfico.

    Returns:
        Objeto matplotlib.figure.Figure.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        square=True,
        ax=ax,
        cbar_kws={"shrink": 0.8},
    )

    ax.set_title(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig
