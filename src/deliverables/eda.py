"""
Entregable descriptivo/exploratorio.

Presenta los resultados de EDA ya calculados por los motores y destaca
HALLAZGOS EXPLORATORIOS (patrones o problemas observables) marcados como tales.
No ejecuta pruebas inferenciales ni recomienda el siguiente análisis.

Los umbrales de presentación (`HALLAZGO_*`) son elecciones de la capa de
presentación para decidir qué patrones se destacan; no son reglas estadísticas
ni deciden ningún análisis.
"""

from typing import Any, Dict, List

import pandas as pd

from src.deliverables.generator import (
    Item,
    Section,
    item_hallazgo,
    item_table,
    item_text,
)

HALLAZGO_CORRELACION_UMBRAL = 0.5
HALLAZGO_ASIMETRIA_UMBRAL = 1.0

_COLUMNAS_DESCRIPTIVOS = [
    ("count", "n"),
    ("mean", "Media"),
    ("median", "Mediana"),
    ("std", "Desv."),
    ("min", "Mín."),
    ("max", "Máx."),
]


def _fmt(value: Any, dec: int = 2) -> str:
    try:
        return f"{float(value):.{dec}f}"
    except (TypeError, ValueError):
        return "N/D"


def build_eda_secciones(eda_results: Dict[str, Any]) -> List[Section]:
    """Construye las secciones descriptivas/exploratorias del entregable inicial."""
    sections: List[Section] = []

    numerical = eda_results.get("numerical")
    if numerical is not None and not numerical.empty:
        rows = []
        for var, r in numerical.iterrows():
            fila = [str(var)]
            for clave, _ in _COLUMNAS_DESCRIPTIVOS:
                fila.append(_fmt(r.get(clave)))
            rows.append(fila)
        sections.append(
            Section(
                titulo="Análisis descriptivo — variables numéricas",
                items=[
                    item_table(
                        ["Variable"]
                        + [nombre for _, nombre in _COLUMNAS_DESCRIPTIVOS],
                        rows,
                    )
                ],
            )
        )

    categorical = eda_results.get("categorical") or {}
    if categorical:
        items = []
        for col, tab in categorical.items():
            if isinstance(tab, pd.DataFrame) and not tab.empty:
                partes = ", ".join(
                    f"{etiqueta}={int(frec)} ({_fmt(pct, 1)}%)"
                    for etiqueta, frec, pct in zip(
                        tab.index, tab["frecuencia"], tab["porcentaje"]
                    )
                )
                items.append(item_text(f"{col}: {partes}"))
        if items:
            sections.append(
                Section(titulo="Frecuencias — variables categóricas", items=items)
            )

    hallazgos = _hallazgos_exploratorios(eda_results)
    if hallazgos:
        sections.append(
            Section(titulo="Hallazgos exploratorios", items=hallazgos)
        )

    return sections


def _hallazgos_exploratorios(eda_results: Dict[str, Any]) -> List[Item]:
    """Destaca patrones observables en los resultados de EDA (sin inferencia)."""
    hallazgos: List[Item] = []
    aviso = "Es un patrón exploratorio descriptivo; no se ejecutó ninguna prueba de hipótesis."

    corr = eda_results.get("correlation")
    if corr is not None and not corr.empty:
        for i, c1 in enumerate(corr.columns):
            for c2 in corr.columns[i + 1:]:
                r = corr.loc[c1, c2]
                if abs(r) >= HALLAZGO_CORRELACION_UMBRAL:
                    sentido = "positiva" if r > 0 else "negativa"
                    hallazgos.append(
                        item_hallazgo(
                            f"'{c1}' y '{c2}' presentan una asociación "
                            f"{sentido} (r = {r:+.3f}).",
                            tipo="patrón",
                            detalle=[aviso],
                        )
                    )

    outliers = eda_results.get("outliers") or {}
    for var, info in outliers.items():
        if info.get("outlier_count", 0) > 0:
            hallazgos.append(
                item_hallazgo(
                    f"'{var}' presenta {info['outlier_count']} valores atípicos "
                    f"({info.get('outlier_percentage', 0)}% de sus observaciones).",
                    tipo="problema potencial",
                    detalle=[aviso],
                )
            )

    numerical = eda_results.get("numerical")
    if numerical is not None and not numerical.empty:
        for var, r in numerical.iterrows():
            skew = r.get("skewness")
            if skew is not None and abs(skew) >= HALLAZGO_ASIMETRIA_UMBRAL:
                lado = "a la derecha" if skew > 0 else "a la izquierda"
                hallazgos.append(
                    item_hallazgo(
                        f"'{var}' tiene una distribución asimétrica {lado} "
                        f"(asimetría {skew:+.2f}).",
                        tipo="patrón",
                        detalle=[aviso],
                    )
                )

    return hallazgos