"""
Entregable de datos faltantes.

Presenta la detección, el diagnóstico del mecanismo de ausencia y la
recomendación de imputación en lenguaje de usuario, SIN exponer pesos,
puntajes, componentes, umbrales ni otros detalles internos del selector o del
pipeline (esos viven en el reporte técnico de desarrollo).
"""

from typing import Any, Dict, List

from src.deliverables.generator import (
    Section,
    item_bullets,
    item_hallazgo,
    item_table,
    item_text,
)

_DECISION_METODO_UNICO = "metodo_unico"
_DECISION_COMPARAR_ALTERNATIVAS = "comparar_alternativas"
_DECISION_SIN_RECOMENDACION = "sin_recomendacion"


def _fmt(value: Any, sufijo: str = "", dec: int = 2) -> str:
    try:
        return f"{float(value):.{dec}f}{sufijo}"
    except (TypeError, ValueError):
        return "N/D"


def build_missing_secciones(diagnose_result: Dict[str, Any]) -> List[Section]:
    """Construye las secciones del entregable de datos faltantes."""
    md = (diagnose_result or {}).get("missing_data") or {}
    detection = md.get("detection_report") or {}
    diagnostics = md.get("diagnostics_report") or {}
    recomendacion = (diagnose_result or {}).get("recomendacion_imputacion") or {}

    sections: List[Section] = []

    # --- Resumen de datos faltantes (E1) ---
    items = [
        item_text(
            f"Valores faltantes: {detection.get('total_missing_values', 0)} "
            f"({_fmt(detection.get('overall_missing_percentage'), '%')} del total)."
        ),
        item_text(
            f"Grado de ausencia global: {detection.get('overall_missing_grade', 'N/D')}."
        ),
        item_text(f"Casos completos: {detection.get('complete_cases', 0)}."),
    ]
    by_variable = detection.get("by_variable") or {}
    if by_variable:
        rows = []
        for var, info in by_variable.items():
            rows.append(
                [
                    str(var),
                    str(info.get("missing_count", 0)),
                    _fmt(info.get("missing_percentage"), "%"),
                    str(info.get("missing_grade", "N/D")),
                ]
            )
        items.append(item_table(["Variable", "Faltantes", "%", "Grado"], rows))
    sections.append(Section(titulo="Datos faltantes", items=items))

    # --- Diagnóstico del mecanismo (E2) ---
    mechanism = diagnostics.get("mechanism") or {}
    mech_items: List[Any] = []
    significantes = mechanism.get("significant_comparisons") or []
    if significantes:
        mech_items.append(
            item_hallazgo(
                "Se detectó evidencia estadística de que la ausencia de "
                "valores no es completamente aleatoria.",
                tipo="advertencia",
                detalle=[
                    f"Variables implicadas: {', '.join(str(s) for s in significantes)}."
                ],
            )
        )
    else:
        mech_items.append(
            item_text(
                "No se encontró evidencia estadística en contra del supuesto "
                "de ausencia completamente aleatoria (MCAR)."
            )
        )
    recomendacion_mecanismo = mechanism.get("recommendation")
    if recomendacion_mecanismo:
        mech_items.append(
            item_text(f"Recomendación del diagnóstico: {recomendacion_mecanismo}")
        )
    sections.append(
        Section(titulo="Diagnóstico del mecanismo de ausencia", items=mech_items)
    )

    # --- Recomendación de imputación (E5, en lenguaje de usuario) ---
    rec_items: List[Any] = []
    por_variable = recomendacion.get("por_variable") or {}
    for var, vr in por_variable.items():
        decision = vr.get("decision")
        metodo = vr.get("metodo_recomendado")
        alternativo = vr.get("metodo_alternativo")

        encabezado = item_text(
            f"**{var}** ({vr.get('missing_count', '?')} faltantes, "
            f"{_fmt(vr.get('missing_percentage'), '%')}):"
        )
        detalles: List[str] = []
        if decision == _DECISION_METODO_UNICO and metodo:
            detalles.append(f"Método recomendado: '{metodo}'.")
        elif decision == _DECISION_COMPARAR_ALTERNATIVAS:
            if metodo and alternativo:
                detalles.append(
                    f"No hay un método claramente superior entre '{metodo}' "
                    f"y '{alternativo}'; conviene comparar ambas alternativas "
                    "antes de imputar."
                )
            elif metodo:
                detalles.append(
                    f"No hay un método claramente superior; conviene comparar "
                    f"alternativas alrededor de '{metodo}'."
                )
        elif decision == _DECISION_SIN_RECOMENDACION:
            detalles.append("No se dispone de un método único recomendado para esta variable.")
        else:
            detalles.append("No se dispone de una recomendación de imputación para esta variable.")

        advertencias = vr.get("advertencias") or []
        if advertencias:
            detalles.append("Advertencias: " + "; ".join(str(a) for a in advertencias))

        rec_items.append(encabezado)
        if detalles:
            rec_items.append(item_bullets(detalles))

    if not rec_items:
        rec_items.append(
            item_text("No hay recomendación de imputación disponible.")
        )
    sections.append(
        Section(titulo="Recomendación de imputación", items=rec_items)
    )

    return sections