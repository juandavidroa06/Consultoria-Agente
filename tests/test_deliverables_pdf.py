"""
Pruebas del renderizado PDF de entregables y de la orden de presentación
"Informe" (`PaperStatsFlow.informe`).

Verifican:
  - PDF en Times New Roman generado directamente del modelo neutral
    `Deliverable` (no del Markdown);
  - preservación verbatim de los valores estadísticos (sin recalcular);
  - jerarquía visual, tablas sin desborde y estilos con la familia Times;
  - que "Informe" es solo presentación/exportación (no ejecuta análisis).
"""

import pytest

from pathlib import Path

import pandas as pd

from src.deliverables.generator import (
    Deliverable,
    DeliverableGenerator,
    Section,
    item_bullets,
    item_hallazgo,
    item_table,
    item_text,
)
from src.deliverables.renderers import render
from src.deliverables.renderers.markdown import render_markdown
from src.deliverables.renderers.pdf import (
    _ANCHO_UTIL,
    _ESTILOS,
    _FUENTES,
    _RUTAS_TNR,
    _build_story,
    _tabla,
    render_pdf,
)
from src.orchestration.flow import PaperStatsFlow


def _ttf_disponibles() -> bool:
    return all(Path(p).is_file() for p in _RUTAS_TNR.values())


def _entregable_ejemplo() -> Deliverable:
    return Deliverable(
        titulo="Resultado del análisis solicitado",
        dataset="datos_sinteticos.csv",
        secciones=[
            Section(
                "Pregunta u objetivo",
                [item_text("Determinar si existe relación entre escolaridad e ingresos.")],
            ),
            Section(
                "Método aplicado",
                [item_text("Método aplicado: Kruskal-Wallis H.")],
            ),
            Section(
                "Supuestos evaluados",
                [item_bullets(["Normalidad (NO se cumple): p=3.7e-03"])],
            ),
            Section(
                "Resultado",
                [
                    item_text("Estadístico: H = 29.30"),
                    item_text("p-valor: 4.34e-07"),
                ],
            ),
            Section(
                "Tabla",
                [item_table(["Variable", "Mediana"], [["Técnico", "3.30"], ["Posgrado", "4.98"]])],
            ),
            Section(
                "Hallazgos",
                [item_hallazgo("Asociación positiva.", "patrón")],
            ),
        ],
        cierre="Esperando tu siguiente decisión.",
    )


def _textos_de_story(story):
    """Extrae el texto de todos los flowables (incluye celdas de tablas)."""
    textos = []
    for fl in story:
        if hasattr(fl, "text"):
            textos.append(str(fl.text))
        elif hasattr(fl, "_cellvalues"):
            for fila in fl._cellvalues:
                for celda in fila:
                    if hasattr(celda, "text"):
                        textos.append(str(celda.text))
    return textos


# ---------------------------------------------------------------------------
# Generación del PDF
# ---------------------------------------------------------------------------

def test_render_pdf_devuelve_bytes_pdf():
    datos = render_pdf(_entregable_ejemplo())
    assert isinstance(datos, bytes)
    assert datos.startswith(b"%PDF")


def test_render_pdf_escribe_archivo(tmp_path):
    destino = tmp_path / "informe.pdf"
    datos = render_pdf(_entregable_ejemplo(), output_path=destino)
    assert destino.exists()
    assert destino.read_bytes() == datos
    assert destino.read_bytes().startswith(b"%PDF")


def test_pdf_usa_familia_times_resuelta():
    datos = render_pdf(_entregable_ejemplo())
    assert datos.startswith(b"%PDF")
    if _ttf_disponibles():
        assert b"TimesNewRoman" in datos
        assert _FUENTES["normal"] == "TimesNewRoman"
        assert _FUENTES["bold"] == "TimesNewRoman-Bold"
    else:
        assert set(_FUENTES.values()) == {
            "Times-Roman",
            "Times-Bold",
            "Times-Italic",
            "Times-BoldItalic",
        }


# ---------------------------------------------------------------------------
# Preservación de valores y jerarquía
# ---------------------------------------------------------------------------

def test_story_preserva_valores_verbatim():
    story = _build_story(_entregable_ejemplo())
    textos = _textos_de_story(story)
    assert any("H = 29.30" in t for t in textos)
    assert any("p-valor: 4.34e-07" in t for t in textos)
    assert any("Técnico" in t for t in textos) and any("3.30" in t for t in textos)


def test_story_jerarquia():
    story = _build_story(_entregable_ejemplo())
    assert story[0].text == "Resultado del análisis solicitado"
    textos = [f.text for f in story if hasattr(f, "text")]
    assert any(t == "Pregunta u objetivo" for t in textos)
    assert any("Esperando tu siguiente decisión." in t for t in textos)
    assert any(isinstance(f, type(_tabla(["a"], [["b"]]))) for f in story)
    assert story[-1].text == "Esperando tu siguiente decisión."


def test_tabla_no_excede_ancho_util():
    from src.deliverables.renderers.pdf import _col_widths

    headers = ["Variable", "Mediana"]
    rows = [["Técnico", "3.30"], ["Pregrado", "4.44"], ["Posgrado", "4.98"]]
    anchos = _col_widths(headers, rows)
    assert sum(anchos) <= _ANCHO_UTIL + 1e-6
    tabla = _tabla(headers, rows)
    assert sum(tabla._colWidths) <= _ANCHO_UTIL + 1e-6


def test_estilos_usan_familia_times():
    familia = set(_FUENTES.values())
    for est in _ESTILOS.values():
        assert est.fontName in familia, est.name
    assert _ESTILOS["celda_cab"].fontName == _FUENTES["bold"]
    assert _ESTILOS["titulo"].fontName == _FUENTES["bold"]
    assert _ESTILOS["archivo"].fontName == _FUENTES["italic"]


# ---------------------------------------------------------------------------
# Dispatcher y compatibilidad
# ---------------------------------------------------------------------------

def test_dispatcher_markdown_y_pdf():
    md = render(_entregable_ejemplo(), formato="markdown")
    assert isinstance(md, str) and "## Pregunta u objetivo" in md
    pdf = render(_entregable_ejemplo(), formato="pdf")
    assert isinstance(pdf, bytes) and pdf.startswith(b"%PDF")


def test_dispatcher_formato_invalido():
    with pytest.raises(ValueError):
        render(_entregable_ejemplo(), formato="html")


def test_render_markdown_retrocompatible():
    assert "## Resultado" in DeliverableGenerator.render_markdown(
        _entregable_ejemplo()
    )


def test_render_pdf_wrapper_generator():
    assert DeliverableGenerator.render_pdf(_entregable_ejemplo()).startswith(b"%PDF")


def test_render_markdown_modulo_renderers():
    assert "## Resultado" in render_markdown(_entregable_ejemplo())


# ---------------------------------------------------------------------------
# Orden "Informe" en PaperStatsFlow
# ---------------------------------------------------------------------------

def _flujo_sintetico() -> PaperStatsFlow:
    df = pd.DataFrame(
        {
            "edad": [25, 30, 35, 40, 45],
            "ingreso": [1000.0, 1500.0, 2000.0, 2500.0, 3000.0],
            "grupo": ["A", "A", "B", "B", "B"],
        }
    )
    return PaperStatsFlow(df)


def test_informe_sin_deliverable_error():
    flujo = PaperStatsFlow(pd.DataFrame({"A": [1.0, 2.0], "B": [3.0, 4.0]}))
    with pytest.raises(ValueError):
        flujo.informe()


def test_informe_genera_pdf_del_ultimo_entregable(tmp_path):
    flujo = _flujo_sintetico()
    flujo.entregable_analisis(
        "Primera pregunta",
        {"metodo": "X", "resultado": {"estadistico": "v1"}},
    )
    ruta1 = flujo.informe(output_path=tmp_path / "a.pdf")
    flujo.entregable_analisis(
        "Segunda pregunta",
        {"metodo": "Y", "resultado": {"estadistico": "v2"}},
    )
    ruta2 = flujo.informe(output_path=tmp_path / "b.pdf")
    assert ruta1.endswith("a.pdf") and ruta2.endswith("b.pdf")
    for p in (ruta1, ruta2):
        assert open(p, "rb").read().startswith(b"%PDF")
    textos = _textos_de_story(_build_story(flujo._last_deliverable))
    assert any("Segunda pregunta" in t for t in textos)


def test_informe_no_recalcula_ni_ejecuta_analisis(monkeypatch, tmp_path):
    import src.analysis.eda as eda_engine

    def _boom(*args, **kwargs):
        raise AssertionError("'Informe' no debe ejecutar motores.")

    for fn in (
        "describe_numerical",
        "describe_categorical",
        "detect_outliers_iqr",
        "calculate_correlation_matrix",
    ):
        monkeypatch.setattr(eda_engine, fn, _boom)

    flujo = _flujo_sintetico()
    flujo.entregable_analisis(
        "¿Existe relación entre escolaridad e ingresos?",
        {"metodo": "Kruskal-Wallis", "resultado": {"estadistico": "H = 29.30"}},
    )
    ruta = flujo.informe(output_path=tmp_path / "informe.pdf")
    assert open(ruta, "rb").read().startswith(b"%PDF")
    assert flujo.state == "sin_diagnostico"


def test_informe_formato_invalido():
    flujo = _flujo_sintetico()
    flujo.entregable_analisis("P", {"metodo": "X"})
    with pytest.raises(ValueError):
        flujo.informe(formato="html")