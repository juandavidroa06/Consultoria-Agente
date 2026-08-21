"""
Capa de orquestación del flujo principal de PaperStats (P-FLOW).

El orquestador de alto nivel (`PaperStatsFlow`) controla las etapas del flujo:

    DATASET → diagnose() → [ESPERAR DECISIÓN] → imputar() → analizar()

sin duplicar la lógica estadística de los módulos de análisis y datos faltantes.
"""

from src.orchestration.flow import PaperStatsFlow

__all__ = ["PaperStatsFlow"]