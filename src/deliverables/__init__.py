"""
Capa de entregables de usuario (capa de presentación).

Consume exclusivamente los resultados técnicos producidos por los motores
estadísticos (PaperStatsFlow, análisis, EDA, pipeline de datos faltantes) y los
transforma en una estructura neutral (`Deliverable`), independiente del formato
final de salida (Markdown hoy; HTML/PDF/tablas en el futuro).

Reglas:
  - Es una capa de presentación pura: NO recalcula estadísticas, NO selecciona
    pruebas ni modelos, NO decide qué análisis realizar, NO modifica datos,
    NO imputa, NO diagnostica mecanismos y NO introduce reglas estadísticas
    nuevas.
  - Dirección de dependencia:
        PaperStatsFlow -> motores -> resultados técnicos -> deliverables -> representación
    deliverables NUNCA invoca motores estadísticos.
"""

from src.deliverables.generator import (
    Deliverable,
    DeliverableGenerator,
    Item,
    Section,
)

__all__ = ["Deliverable", "DeliverableGenerator", "Item", "Section"]