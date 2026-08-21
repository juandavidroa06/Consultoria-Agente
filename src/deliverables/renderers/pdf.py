"""
Renderizado de entregables a PDF con tipografía Times New Roman.

Segunda representación del modelo neutral `Deliverable` (la primera es
Markdown). El PDF se genera DIRECTAMENTE desde el `Deliverable`, sin pasar por
Markdown y sin recalcular ni reformatear valores estadísticos: consume las
cadenas ya presentes en los items producidos por los builders.

Garantía de tipografía: se registran las 4 variantes TTF de Times New Roman
(Regular, Bold, Italic, Bold Italic) si están disponibles en el sistema; en
caso contrario se usa la familia Type1 'Times' de reportlab (métricamente
idéntica a Times New Roman). Todos los estilos y tablas usan esa familia.
"""

from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    HRFlowable,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

FAMILIA = "TimesNewRoman"

# Rutas candidatas de las 4 variantes de Times New Roman (Linux msttcorefonts).
_RUTAS_TNR = {
    "normal": "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf",
    "bold": "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman_Bold.ttf",
    "italic": "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman_Italic.ttf",
    "boldItalic": "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman_Bold_Italic.ttf",
}


def _resolver_fuentes() -> Dict[str, str]:
    """Nombre de fuente concreto por rol (Times New Roman TTF o Times Type1)."""
    disponibles = all(Path(p).is_file() for p in _RUTAS_TNR.values())
    if not disponibles:
        return {
            "normal": "Times-Roman",
            "bold": "Times-Bold",
            "italic": "Times-Italic",
            "boldItalic": "Times-BoldItalic",
        }
    nombres = {
        "normal": FAMILIA,
        "bold": f"{FAMILIA}-Bold",
        "italic": f"{FAMILIA}-Italic",
        "boldItalic": f"{FAMILIA}-BoldItalic",
    }
    for rol, ruta in _RUTAS_TNR.items():
        pdfmetrics.registerFont(TTFont(nombres[rol], ruta))
    pdfmetrics.registerFontFamily(
        FAMILIA,
        normal=nombres["normal"],
        bold=nombres["bold"],
        italic=nombres["italic"],
        boldItalic=nombres["boldItalic"],
    )
    return nombres


_FUENTES = _resolver_fuentes()
_MARGEN = 2 * cm
_ANCHO_UTIL = A4[0] - 2 * _MARGEN


def _estilo(nombre: str, **kw) -> ParagraphStyle:
    return ParagraphStyle(nombre, **kw)


_ESTILOS = {
    "titulo": _estilo(
        "TNR_Titulo",
        fontName=_FUENTES["bold"],
        fontSize=18,
        leading=22,
        alignment=TA_CENTER,
        spaceAfter=4,
    ),
    "archivo": _estilo(
        "TNR_Archivo",
        fontName=_FUENTES["italic"],
        fontSize=10,
        leading=13,
        alignment=TA_CENTER,
        spaceAfter=6,
    ),
    "seccion": _estilo(
        "TNR_Seccion",
        fontName=_FUENTES["bold"],
        fontSize=13,
        leading=16,
        spaceBefore=10,
        spaceAfter=4,
    ),
    "texto": _estilo(
        "TNR_Texto",
        fontName=_FUENTES["normal"],
        fontSize=10.5,
        leading=14,
        spaceBefore=2,
        spaceAfter=2,
    ),
    "bullet": _estilo(
        "TNR_Bullet",
        fontName=_FUENTES["normal"],
        fontSize=10.5,
        leading=14,
        leftIndent=16,
        bulletIndent=2,
        bulletFontName=_FUENTES["normal"],
        bulletFontSize=10.5,
        spaceBefore=1,
        spaceAfter=1,
    ),
    "hallazgo": _estilo(
        "TNR_Hallazgo",
        fontName=_FUENTES["normal"],
        fontSize=10.5,
        leading=14,
        spaceBefore=2,
        spaceAfter=1,
    ),
    "hallazgo_detalle": _estilo(
        "TNR_HallazgoDetalle",
        fontName=_FUENTES["italic"],
        fontSize=9.5,
        leading=12.5,
        leftIndent=16,
        spaceBefore=0.5,
        spaceAfter=1,
    ),
    "celda_cab": _estilo(
        "TNR_CeldaCab",
        fontName=_FUENTES["bold"],
        fontSize=9,
        leading=11.5,
    ),
    "celda": _estilo(
        "TNR_Celda",
        fontName=_FUENTES["normal"],
        fontSize=9,
        leading=11.5,
    ),
    "cierre": _estilo(
        "TNR_Cierre",
        fontName=_FUENTES["bold"],
        fontSize=11,
        leading=14,
        spaceBefore=10,
        spaceAfter=2,
    ),
}


def _pie_de_pagina(canvas, doc) -> None:
    """Pie de página con número de página en Times New Roman."""
    canvas.saveState()
    canvas.setFont(_FUENTES["normal"], 8)
    canvas.drawCentredString(
        A4[0] / 2, 0.9 * cm, f"PaperStats — Página {doc.page}"
    )
    canvas.restoreState()


def _col_widths(headers: List[str], rows: List[List[str]]) -> List[float]:
    """Ancho de columnas proporcional al contenido, sin exceder el ancho útil."""
    if not headers:
        return []
    pesos = []
    for i, h in enumerate(headers):
        max_len = max(
            [len(str(h))]
            + [len(str(r[i])) for r in rows if i < len(r)]
        )
        pesos.append(max(1.0, float(max_len)))
    total = sum(pesos)
    return [_ANCHO_UTIL * p / total for p in pesos]


def _tabla(headers: List[str], rows: List[List[str]]) -> Table:
    data = [[Paragraph(str(h), _ESTILOS["celda_cab"]) for h in headers]]
    for r in rows:
        data.append([Paragraph(str(c), _ESTILOS["celda"]) for c in r])
    tabla = Table(data, colWidths=_col_widths(headers, rows), repeatRows=1)
    tabla.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, -1), _FUENTES["normal"]),
                ("FONTNAME", (0, 0), (-1, 0), _FUENTES["bold"]),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("BACKGROUND", (0, 0), (-1, 0), colors.Color(0.93, 0.93, 0.93)),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.Color(0.6, 0.6, 0.6)),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return tabla


def _hallazgo_flowables(item) -> List[Any]:
    data = item.data
    tipo = data.get("tipo", "patrón")
    descripcion = data.get("descripcion", "")
    flowables = [
        Paragraph(
            f"<b>Hallazgo exploratorio ({tipo}):</b> {descripcion}",
            _ESTILOS["hallazgo"],
        )
    ]
    for det in data.get("detalle", []):
        flowables.append(Paragraph(str(det), _ESTILOS["hallazgo_detalle"]))
    return flowables


def _build_story(deliverable) -> List[Any]:
    """Convierte el `Deliverable` en la lista de flowables del PDF (sin recalcular)."""
    story: List[Any] = []
    story.append(Paragraph(deliverable.titulo, _ESTILOS["titulo"]))
    if deliverable.dataset:
        story.append(
            Paragraph(f"Archivo: {deliverable.dataset}", _ESTILOS["archivo"])
        )
    story.append(
        HRFlowable(
            width="100%",
            thickness=0.8,
            color=colors.Color(0.35, 0.35, 0.35),
            spaceAfter=4,
        )
    )
    for sec in deliverable.secciones:
        story.append(Paragraph(sec.titulo, _ESTILOS["seccion"]))
        for it in sec.items:
            if it.kind == "text":
                story.append(
                    Paragraph(str(it.data.get("text", "")), _ESTILOS["texto"])
                )
            elif it.kind == "bullets":
                for b in it.data.get("bullets", []):
                    story.append(
                        Paragraph(str(b), _ESTILOS["bullet"], bulletText="•")
                    )
            elif it.kind == "table":
                story.append(
                    _tabla(it.data.get("headers", []), it.data.get("rows", []))
                )
                story.append(Spacer(1, 4))
            elif it.kind == "hallazgo":
                story.extend(_hallazgo_flowables(it))
    if deliverable.cierre:
        story.append(Spacer(1, 2))
        story.append(Paragraph(deliverable.cierre, _ESTILOS["cierre"]))
    return story


def render_pdf(deliverable, output_path: Optional[Union[str, Path]] = None) -> bytes:
    """Representa el `Deliverable` como PDF (Times New Roman).

    No recalcula ni formatea valores: usa las cadenas del `Deliverable` tal cual.
    """
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=_MARGEN,
        rightMargin=_MARGEN,
        topMargin=_MARGEN,
        bottomMargin=_MARGEN,
        title=deliverable.titulo,
        author="PaperStats",
    )
    doc.build(_build_story(deliverable), onFirstPage=_pie_de_pagina, onLaterPages=_pie_de_pagina)
    data = buffer.getvalue()
    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(data)
    return data