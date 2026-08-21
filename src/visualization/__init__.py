"""
Módulo para visualización de datos estadísticas.
"""

from .plots import (
    plot_histogram,
    plot_boxplot,
    plot_scatter,
    plot_qq,
    plot_correlation_matrix,
)

__all__ = [
    "plot_histogram",
    "plot_boxplot",
    "plot_scatter",
    "plot_qq",
    "plot_correlation_matrix",
]
