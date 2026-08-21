"""
Entregable de control de calidad y estado de los datos.

Capa de presentación: consume `diagnose_result` (ya calculado por el flujo) y
no recalcula nada. No invoca motores estadísticos.
"""

from typing import Any, Dict, List

from src.deliverables.generator import (
    Section,
    item_table,
    item_text,
)


def _fmt(value: Any, sufijo: str = "", dec: int = 2) -> str:
    try:
        return f"{float(value):.{dec}f}{sufijo}"
    except (TypeError, ValueError):
        return "N/D"


def build_quality_secciones(diagnose_result: Dict[str, Any]) -> List[Section]:
    """Construye las secciones de control de calidad del entregable inicial."""
    dataset = (diagnose_result or {}).get("dataset") or {}
    calidad = (diagnose_result or {}).get("calidad") or {}
    clasificacion = (diagnose_result or {}).get("clasificacion_variables") or {}

    items = []
    items.append(
        item_text(
            f"{dataset.get('rows', 'N/D')} registros, "
            f"{dataset.get('columns', 'N/D')} variables."
        )
    )

    missing = (calidad.get("missing_values") or {})
    total_missing = missing.get("total_missing_values", 0)
    overall_pct = missing.get("overall_missing_percentage")
    if total_missing == 0:
        items.append(item_text("Sin valores faltantes."))
    else:
        items.append(
            item_text(
                f"Valores faltantes: {total_missing} "
                f"({_fmt(overall_pct, '%')} del total)."
            )
        )

    dup_count = ((calidad.get("duplicates")) or {}).get("duplicate_count", 0)
    items.append(item_text(f"Filas duplicadas: {dup_count}."))

    if clasificacion:
        items.append(
            item_table(
                ["Variable", "Tipo"],
                [[str(col), str(tipo)] for col, tipo in clasificacion.items()],
            )
        )

    return [Section(titulo="Control de calidad", items=items)]