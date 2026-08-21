"""
Representación Markdown del modelo neutral `Deliverable`.

Lógica extraída de `DeliverableGenerator.render_markdown` (que ahora delega aquí).
No recalcula nada; solo serializa los items del `Deliverable` como texto Markdown.
"""

from typing import List


def render_markdown(deliverable) -> str:
    """Primera representación del entregable: Markdown."""
    lines: List[str] = [f"# {deliverable.titulo}"]
    if deliverable.dataset:
        lines.append("")
        lines.append(f"**Archivo**: {deliverable.dataset}")

    for sec in deliverable.secciones:
        lines.append("")
        lines.append(f"## {sec.titulo}")
        for it in sec.items:
            data = it.data
            if it.kind == "text":
                lines.append(str(data.get("text", "")))
            elif it.kind == "bullets":
                for b in data.get("bullets", []):
                    lines.append(f"- {b}")
            elif it.kind == "table":
                headers = data.get("headers", [])
                rows = data.get("rows", [])
                lines.append("| " + " | ".join(headers) + " |")
                lines.append("|" + "|".join("---" for _ in headers) + "|")
                for row in rows:
                    lines.append("| " + " | ".join(row) + " |")
            elif it.kind == "hallazgo":
                tipo = data.get("tipo", "patrón")
                lines.append(
                    f"- **Hallazgo exploratorio ({tipo})**: "
                    f"{data.get('descripcion', '')}"
                )
                for det in data.get("detalle", []):
                    lines.append(f"  - {det}")

    if deliverable.cierre:
        lines.append("")
        lines.append(f"**{deliverable.cierre}**")
    return "\n".join(lines)