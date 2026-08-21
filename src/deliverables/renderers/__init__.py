"""
Renderizado de entregables (representaciones del modelo neutral `Deliverable`).

Cada formato es una representación independiente del MISMO `Deliverable`; los
renderers no invocan motores ni recalcular nada. Formatos disponibles:
- markdown: texto (primera representación).
- pdf: archivo PDF con Times New Roman (representación de exportación).

Futuros formatos (HTML, etc.) se añaden como módulos en este paquete sin tocar
la capa de builders ni los motores.
"""

from src.deliverables.renderers.markdown import render_markdown
from src.deliverables.renderers.pdf import render_pdf

FORMATOS = {"markdown": "texto/markdown", "pdf": "application/pdf"}


def render(deliverable, formato: str = "markdown", output_path=None):
    """Representa un `Deliverable` en el formato solicitado.

    Args:
        deliverable: Modelo neutral producido por los builders.
        formato: "markdown" o "pdf".
        output_path: Ruta opcional para escribir el archivo (solo pdf).

    Returns:
        str para markdown; bytes para pdf.
    """
    if formato == "markdown":
        return render_markdown(deliverable)
    if formato == "pdf":
        return render_pdf(deliverable, output_path)
    raise ValueError(
        f"Formato de entregable no soportado: {formato!r}. "
        f"Formatos disponibles: {sorted(FORMATOS)}."
    )


__all__ = ["render", "render_markdown", "render_pdf", "FORMATOS"]