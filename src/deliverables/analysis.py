"""
Entregable de análisis estadístico solicitado por el usuario.

Presenta un análisis YA ejecutado por los motores (método, supuestos,
resultado, interpretación). No selecciona pruebas ni recalcula resultados.

Contrato de `resultado` (dict producido por el motor tras ejecutar el análisis
solicitado):
  - dataset (opcional): nombre del archivo.
  - objetivo (opcional): texto del objetivo; si falta se usa `pregunta`.
  - metodo: nombre del método aplicado.
  - justificacion_metodo: por qué se eligió ese método.
  - supuestos: lista de {"supuesto", "evaluacion", "cumple": True|False|None}.
  - resultado: {"estadistico", "p_valor", "hipotesis", "decision"}.
  - interpretacion: texto de interpretación del resultado.
  - advertencias: lista de limitaciones/caveats.
"""

from typing import Any, Dict, List

from src.deliverables.generator import (
    Section,
    item_bullets,
    item_text,
)


def build_analisis_secciones(
    pregunta: str, resultado: Dict[str, Any]
) -> List[Section]:
    """Construye las secciones del entregable de un análisis solicitado."""
    sections: List[Section] = []

    objetivo = resultado.get("objetivo") or pregunta or "Análisis solicitado"
    sections.append(
        Section(titulo="Pregunta u objetivo", items=[item_text(str(objetivo))])
    )

    metodo = resultado.get("metodo")
    justificacion = resultado.get("justificacion_metodo")
    if metodo or justificacion:
        items = []
        if metodo:
            items.append(item_text(f"Método aplicado: {metodo}."))
        if justificacion:
            items.append(item_text(f"Justificación: {justificacion}"))
        sections.append(Section(titulo="Método aplicado", items=items))

    supuestos = resultado.get("supuestos") or []
    if supuestos:
        items = []
        for s in supuestos:
            nombre = s.get("supuesto", "Supuesto")
            cumple = s.get("cumple")
            if cumple is True:
                etiqueta = "se cumple"
            elif cumple is False:
                etiqueta = "NO se cumple"
            else:
                etiqueta = "no evaluado"
            evaluacion = s.get("evaluacion", "")
            texto = f"{nombre} ({etiqueta})"
            if evaluacion:
                texto += f": {evaluacion}"
            items.append(item_bullets([texto]))
        sections.append(Section(titulo="Supuestos evaluados", items=items))

    res = resultado.get("resultado") or {}
    if res:
        items = []
        if res.get("estadistico"):
            items.append(item_text(f"Estadístico: {res['estadistico']}"))
        if res.get("p_valor"):
            items.append(item_text(f"p-valor: {res['p_valor']}"))
        if res.get("hipotesis"):
            items.append(item_text(f"Hipótesis: {res['hipotesis']}"))
        if res.get("decision"):
            items.append(item_text(f"Decisión: {res['decision']}"))
        sections.append(Section(titulo="Resultado", items=items))

    interpretacion = resultado.get("interpretacion")
    if interpretacion:
        sections.append(
            Section(
                titulo="Interpretación", items=[item_text(str(interpretacion))]
            )
        )

    advertencias = resultado.get("advertencias") or []
    if advertencias:
        sections.append(
            Section(
                titulo="Advertencias y limitaciones",
                items=[item_bullets([str(a) for a in advertencias])],
            )
        )

    return sections