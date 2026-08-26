"""
Coordinador principal de entregables.

Define el modelo neutral de presentación (`Deliverable`, `Section`, `Item`) y
coordina la construcción de los entregables de usuario a partir de los
resultados técnicos que producen los motores. No recalcula ni decide nada.

La representación final está aislada en `render_markdown`, la primera forma de
salida soportada; un futuro módulo de renderizado (HTML, PDF, tablas, gráficos)
puede operar sobre el mismo modelo neutral sin tocar los motores estadísticos.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Union

# --- Modelo neutral de presentación (independiente del formato final) ---------

ESTADO_SIN_FALTANTES = "sin_faltantes"
ESTADO_ESPERANDO_DECISION = "esperando_decision"


@dataclass
class Item:
    """Unidad de contenido de una sección.

    `kind` define el tipo de contenido: "text", "bullets", "table" o "hallazgo".
    """

    kind: str
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Section:
    """Sección de un entregable: título + items."""

    titulo: str
    items: List[Item] = field(default_factory=list)


@dataclass
class Deliverable:
    """Entregable de usuario en forma neutral (sin markdown embebido)."""

    titulo: str
    dataset: str = ""
    secciones: List[Section] = field(default_factory=list)
    cierre: str = ""


# --- Fábricas de items (conveniencia para los builders) -----------------------


def item_text(text: str) -> Item:
    return Item(kind="text", data={"text": str(text)})


def item_bullets(bullets: List[str]) -> Item:
    return Item(kind="bullets", data={"bullets": [str(b) for b in bullets]})


def item_table(headers: List[str], rows: List[List[Any]]) -> Item:
    return Item(
        kind="table",
        data={
            "headers": [str(h) for h in headers],
            "rows": [[str(c) for c in row] for row in rows],
        },
    )


def item_hallazgo(descripcion: str, tipo: str, detalle: List[str] = None) -> Item:
    return Item(
        kind="hallazgo",
        data={
            "descripcion": str(descripcion),
            "tipo": str(tipo),
            "detalle": [str(d) for d in (detalle or [])],
        },
    )


# --- Coordinador principal -----------------------------------------------------


class DeliverableGenerator:
    """Construye entregables de usuario desde resultados técnicos.

    La capa de presentación nunca invoca motores estadísticos: recibe los
    resultados ya calculados y solo los transforma en `Deliverable`.
    """

    def build_inicial(
        self,
        diagnose_result: Dict[str, Any],
        eda_results: Dict[str, Any] = None,
    ) -> Deliverable:
        """Entregable de la etapa inicial.

        Si los datos tienen faltantes, delega en el entregable de datos
        faltantes; en caso contrario, presenta control de calidad + EDA
        descriptivo/exploratorio y espera la decisión del usuario.
        """
        nombre = self._dataset_name(diagnose_result)
        estado = (diagnose_result or {}).get("estado")
        if estado == ESTADO_ESPERANDO_DECISION:
            return self.build_missing(diagnose_result)
        if estado != ESTADO_SIN_FALTANTES:
            raise ValueError(
                "El entregable inicial requiere estado "
                f"'{ESTADO_SIN_FALTANTES}' u '{ESTADO_ESPERANDO_DECISION}', "
                f"se recibió '{estado}'."
            )
        if eda_results is None:
            raise ValueError(
                "Se requieren los resultados de EDA (eda_results) para "
                "construir el entregable inicial sin datos faltantes."
            )
        from src.deliverables.eda import build_eda_secciones
        from src.deliverables.quality import build_quality_secciones

        secciones = build_quality_secciones(diagnose_result)
        secciones += build_eda_secciones(eda_results)
        return Deliverable(
            titulo="Informe de datos",
            dataset=nombre,
            secciones=secciones,
            cierre="Los datos están listos. ¿Qué análisis deseas realizar?",
        )

    def build_missing(self, diagnose_result: Dict[str, Any]) -> Deliverable:
        """Entregable de datos faltantes (reporte + diagnóstico + recomendación)."""
        from src.deliverables.missing import build_missing_secciones

        return Deliverable(
            titulo="Datos faltantes",
            dataset=self._dataset_name(diagnose_result),
            secciones=build_missing_secciones(diagnose_result),
            cierre="Esperando tu decisión sobre el método de imputación.",
        )

    def build_analisis(
        self, pregunta: str, resultado: Dict[str, Any]
    ) -> Deliverable:
        """Entregable de un análisis estadístico ya ejecutado por el motor."""
        from src.deliverables.analysis import build_analisis_secciones

        nombre = (
            (resultado or {}).get("dataset")
            if isinstance(resultado, dict)
            else None
        )
        return Deliverable(
            titulo="Resultado del análisis solicitado",
            dataset=str(nombre) if nombre else "",
            secciones=build_analisis_secciones(pregunta, resultado or {}),
            cierre="",
        )

    @staticmethod
    def render_markdown(deliverable: "Deliverable") -> str:
        """Representación Markdown del entregable (delega en renderers)."""
        from src.deliverables.renderers.markdown import render_markdown

        return render_markdown(deliverable)

    @staticmethod
    def render_pdf(
        deliverable: "Deliverable",
        output_path: Union[str, Path, None] = None,
    ) -> bytes:
        """Representación PDF (Times New Roman) del entregable.

        Delega en la capa de renderers; consume el `Deliverable` sin recalcular
        ni modificar valores estadísticos.
        """
        from src.deliverables.renderers.pdf import render_pdf

        return render_pdf(deliverable, output_path)

    @staticmethod
    def _dataset_name(diagnose_result: Dict[str, Any]) -> str:
        return str(
            ((diagnose_result or {}).get("dataset") or {}).get(
                "file_name", "Dataset"
            )
        )