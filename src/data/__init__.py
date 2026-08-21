"""
Módulo de carga y validación de datos.
"""

from .loader import load_data
from .validator import DataValidator

__all__ = ["load_data", "DataValidator"]
