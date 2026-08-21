"""
Pruebas de la capa de entregables de usuario (`src/deliverables`).

Verifican que la capa es de presentación pura:
  - los builders no invocan motores estadísticos ni recalcan nada;
  - los entregables no exponen detalles técnicos internos;
  - los hallazgos exploratorios se distinguen de análisis inferenciales;
  - la representación (Markdown) está desacoplada del modelo neutral.
"""

import pytest

import pandas as pd

from src.deliverables.generator import (
    Deliverable,
    DeliverableGenerator,
    Item,
    Section,
    item_bullets,
    item_hallazgo,
    item_table,
    item_text,
)
from src.deliverables.eda import build_eda_secciones
from src.deliverables.quality import build_quality_secciones
from src.deliverables.missing import build_missing_secciones
from src.deliverables.analysis import build_analisis_secciones
from src.orchestration.flow import PaperStatsFlow


# ---------------------------------------------------------------------------
# Fixtures (resultados técnicos simulados; los builders solo los consumen)
# ---------------------------------------------------------------------------

def _diagnose_sin_faltantes():
    return {
        "estado": "sin_faltantes",
        "dataset": {
            "file_name": "test.csv",
            "rows": 5,
            "columns": 3,
            "target_variable": None,
        },
        "calidad": {
            "missing_values": {
                "total_missing_values": 0,
                "overall_missing_percentage": 0.0,
            },
            "duplicates": {"duplicate_count": 0, "duplicate_percentage": 0.0},
        },
        "clasificacion_variables": {
            "X": "Cuantitativa continua",
            "Grupo": "Cualitativa nominal",
        },
    }


def _diagnose_con_faltantes():
    return {
        "estado": "esperando_decision",
        "dataset": {
            "file_name": "test.csv",
            "rows": 5,
            "columns": 2,
            "target_variable": None,
        },
        "calidad": {
            "missing_values": {
                "total_missing_values": 3,
                "overall_missing_percentage": 30.0,
            },
            "duplicates": {"duplicate_count": 0, "duplicate_percentage": 0.0},
        },
        "clasificacion_variables": {"Y": "Cuantitativa continua"},
        "missing_data": {
            "detection_report": {
                "total_missing_values": 3,
                "overall_missing_percentage": 30.0,
                "overall_missing_grade": "Media",
                "complete_cases": 2,
                "by_variable": {
                    "Y": {
                        "variable": "Y",
                        "variable_type": "Cuantitativa continua",
                        "dtype": "float64",
                        "missing_count": 3,
                        "missing_percentage": 60.0,
                        "placeholder_count": 0,
                        "missing_grade": "Alta",
                    }
                },
            },
            "diagnostics_report": {
                "mechanism": {
                    "significant_comparisons": [],
                    "recommendation": (
                        "Mantener la ausencia completamente aleatoria como "
                        "hipótesis de trabajo."
                    ),
                    "evidence": "Sin asociaciones significativas tras corrección FDR.",
                },
            },
        },
        "recomendacion_imputacion": {
            "por_variable": {
                "Y": {
                    "variable_type": "Cuantitativa continua",
                    "missing_count": 3,
                    "missing_percentage": 60.0,
                    "decision": "comparar_alternativas",
                    "metodo_recomendado": "mediana",
                    "metodo_alternativo": "media",
                    "score_recomendado": 0.7575,
                    "metodos_considerados": [],
                    "advertencias": ["Sin evidencia empírica de E4."],
                    "justificacion": (
                        "La brecha entre 'mediana' (score=0.7575) y 'media' "
                        "(score=0.7500) es de 0.0075, menor al umbral de "
                        "sensibilidad (0.02)."
                    ),
                }
            },
            "advertencias_globales": [],
            "resumen": "1 variable(s) con datos faltantes; 0 con método único recomendado.",
        },
    }


def _eda_results_fixture():
    return {
        "numerical": pd.DataFrame(
            {
                "count": [5, 5],
                "mean": [10.0, 20.0],
                "std": [2.0, 3.0],
                "median": [9.5, 19.0],
                "iqr": [3.0, 4.0],
                "min": [7.0, 15.0],
                "max": [13.0, 26.0],
                "skewness": [0.4, 1.3],
                "kurtosis": [0.0, 1.0],
            },
            index=["X", "Y"],
        ),
        "categorical": {
            "Grupo": pd.DataFrame(
                {"frecuencia": [3, 2], "porcentaje": [60.0, 40.0]},
                index=["A", "B"],
            )
        },
        "outliers": {
            "X": {
                "outlier_count": 0,
                "outlier_percentage": 0.0,
                "lower_bound": 1.0,
                "upper_bound": 5.0,
            },
            "Y": {
                "outlier_count": 1,
                "outlier_percentage": 20.0,
                "lower_bound": 0.0,
                "upper_bound": 10.0,
            },
        },
        "correlation": pd.DataFrame(
            {"X": [1.0, 0.7], "Y": [0.7, 1.0]}, index=["X", "Y"]
        ),
        "numerical_columns": ["X", "Y"],
        "categorical_columns": ["Grupo"],
    }


# ---------------------------------------------------------------------------
# Modelo neutral y representación
# ---------------------------------------------------------------------------

def test_modelo_neutral_no_contiene_markdown():
    d = Deliverable(
        titulo="T",
        dataset="d.csv",
        secciones=[Section("S", [item_text("hola"), item_bullets(["a"])])],
        cierre="¿Qué análisis deseas realizar?",
    )
    assert isinstance(d, Deliverable)
    assert d.secciones[0].items[0].kind == "text"
    assert d.secciones[0].items[1].kind == "bullets"
    assert "##" not in repr(d.secciones) and "# " not in d.titulo


def test_render_markdown_estructura():
    d = Deliverable(
        titulo="Informe",
        dataset="d.csv",
        secciones=[
            Section("Control de calidad", [item_text("Sin faltantes.")]),
            Section("Tabla", [item_table(["A", "B"], [[1, 2], [3, 4]])]),
            Section("Hallazgos", [item_hallazgo("Asociación positiva.", "patrón")]),
        ],
        cierre="Los datos están listos. ¿Qué análisis deseas realizar?",
    )
    md = DeliverableGenerator.render_markdown(d)
    assert md.startswith("# Informe")
    assert "## Control de calidad" in md
    assert "## Tabla" in md
    assert "| A | B |" in md
    assert "**Hallazgo exploratorio (patrón)**: Asociación positiva." in md
    assert "**Los datos están listos. ¿Qué análisis deseas realizar?**" in md


# ---------------------------------------------------------------------------
# Los builders son presentación pura (no recalculan ni invocan motores)
# ---------------------------------------------------------------------------

def test_builders_no_invocan_motores(monkeypatch):
    import src.analysis.eda as eda_engine

    def _boom(*args, **kwargs):
        raise AssertionError("La capa de presentación no debe invocar motores.")

    for fn in (
        "describe_numerical",
        "describe_categorical",
        "detect_outliers_iqr",
        "calculate_correlation_matrix",
    ):
        monkeypatch.setattr(eda_engine, fn, _boom)

    q = build_quality_secciones(_diagnose_sin_faltantes())
    e = build_eda_secciones(_eda_results_fixture())
    m = build_missing_secciones(_diagnose_con_faltantes())
    a = build_analisis_secciones("¿Relación X~Y?", {"metodo": "Spearman"})
    assert q and e and m and a


def test_build_inicial_sin_faltantes_estructura():
    d = DeliverableGenerator().build_inicial(
        _diagnose_sin_faltantes(), _eda_results_fixture()
    )
    titulos = [s.titulo for s in d.secciones]
    assert "Control de calidad" in titulos
    assert "Análisis descriptivo — variables numéricas" in titulos
    assert "Frecuencias — variables categóricas" in titulos
    assert "Hallazgos exploratorios" in titulos
    assert d.cierre == "Los datos están listos. ¿Qué análisis deseas realizar?"


def test_build_inicial_con_faltantes_delega_en_missing():
    d = DeliverableGenerator().build_inicial(_diagnose_con_faltantes())
    assert d.titulo == "Datos faltantes"
    assert any(s.titulo == "Datos faltantes" for s in d.secciones)


def test_build_inicial_requiere_eda_sin_faltantes():
    with pytest.raises(ValueError):
        DeliverableGenerator().build_inicial(_diagnose_sin_faltantes())


def test_build_inicial_estado_invalido():
    with pytest.raises(ValueError):
        DeliverableGenerator().build_inicial({"estado": "revisar"})


# ---------------------------------------------------------------------------
# Hallazgo exploratorio vs análisis inferencial
# ---------------------------------------------------------------------------

def test_hallazgo_exploratorio_no_es_inferencial():
    secciones = build_eda_secciones(_eda_results_fixture())
    hallazgos = [
        it for s in secciones for it in s.items if it.kind == "hallazgo"
    ]
    descripciones = [h.data["descripcion"] for h in hallazgos]
    assert any("'X' y 'Y' presentan una asociación positiva" in d for d in descripciones)
    assert any("valores atípicos" in d for d in descripciones)
    assert any("asimétrica" in d for d in descripciones)
    for h in hallazgos:
        assert "p-valor" not in h.data["descripcion"]
    texto = DeliverableGenerator.render_markdown(
        Deliverable("t", secciones=secciones)
    )
    assert "Spearman" not in texto
    assert "Pearson" not in texto
    assert "no se ejecutó ninguna prueba" in texto


def test_deliverable_no_recomienda_siguiente_analisis():
    d = DeliverableGenerator().build_inicial(
        _diagnose_sin_faltantes(), _eda_results_fixture()
    )
    texto = DeliverableGenerator.render_markdown(d)
    assert "recomienda" not in texto.lower()
    assert "siguiente análisis" not in texto.lower()
    assert "¿Qué análisis deseas realizar?" in texto


# ---------------------------------------------------------------------------
# Entregable de datos faltantes (sin exponer detalles técnicos)
# ---------------------------------------------------------------------------

def test_missing_deliverable_no_expone_internos():
    d = DeliverableGenerator().build_missing(_diagnose_con_faltantes())
    texto = DeliverableGenerator.render_markdown(d)
    for interno in ("score", "weights", "components", "umbral", "0.02", "brecha"):
        assert interno.lower() not in texto.lower(), f"Filtró interno: {interno}"
    assert "0.7575" not in texto


def test_missing_deliverable_comparar_alternativas():
    d = DeliverableGenerator().build_missing(_diagnose_con_faltantes())
    texto = DeliverableGenerator.render_markdown(d)
    assert "comparar ambas alternativas" in texto
    assert "mediana" in texto and "media" in texto


def test_missing_deliverable_mecanismo_sin_evidencia():
    secciones = build_missing_secciones(_diagnose_con_faltantes())
    mech = next(s for s in secciones if s.titulo == "Diagnóstico del mecanismo de ausencia")
    texto = "\n".join(
        it.data.get("text", "") for it in mech.items if it.kind == "text"
    )
    assert "no se encontró evidencia estadística" in texto.lower()


# ---------------------------------------------------------------------------
# Entregable de análisis solicitado
# ---------------------------------------------------------------------------

def test_build_analisis_secciones():
    resultado = {
        "dataset": "test.csv",
        "metodo": "Correlación de Spearman",
        "justificacion_metodo": "Datos no normales y con atípicos.",
        "supuestos": [
            {"supuesto": "Normalidad", "evaluacion": "p < 0.05", "cumple": False},
            {"supuesto": "Relación monótona", "evaluacion": "Se cumple", "cumple": True},
        ],
        "resultado": {
            "estadistico": "rho = +0.5813",
            "p_valor": "1.45e-16",
            "hipotesis": "H0: no hay asociación monótona.",
            "decision": "Se rechaza H0.",
        },
        "interpretacion": "Asociación positiva moderada-fuerte.",
        "advertencias": ["La significancia no implica causalidad."],
    }
    d = DeliverableGenerator().build_analisis("¿Relación Ingresos~Egresos?", resultado)
    titulos = [s.titulo for s in d.secciones]
    for t in ("Pregunta u objetivo", "Método aplicado", "Supuestos evaluados",
              "Resultado", "Interpretación", "Advertencias y limitaciones"):
        assert t in titulos
    texto = DeliverableGenerator.render_markdown(d)
    assert "¿Relación Ingresos~Egresos?" in texto
    assert "rho = +0.5813" in texto
    assert "NO se cumple" in texto
    assert d.cierre == "Esperando tu siguiente decisión."


# ---------------------------------------------------------------------------
# Integración con PaperStatsFlow (los motores los orquesta el flujo)
# ---------------------------------------------------------------------------

def _df_sin_faltantes():
    return pd.DataFrame(
        {
            "Edad": [20, 25, 30, 35, 40, 45, 50, 55],
            "Ingresos": [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            "Egresos": [1.5, 2.4, 3.3, 4.2, 5.1, 6.0, 6.9, 7.8],
            "Sexo": ["M", "F", "M", "F", "M", "F", "M", "F"],
        }
    )


def _df_con_faltantes():
    return pd.DataFrame(
        {
            "X": [1.0, 2.0, float("nan"), 4.0, 5.0, float("nan")],
            "Y": [1.0, float("nan"), 3.0, float("nan"), 5.0, 6.0],
            "G": ["a", "b", "a", "b", "a", "b"],
        }
    )


def test_flujo_entregable_inicial_sin_faltantes():
    flujo = PaperStatsFlow(_df_sin_faltantes())
    d = flujo.entregable_inicial()
    assert isinstance(d, Deliverable)
    titulos = [s.titulo for s in d.secciones]
    assert "Control de calidad" in titulos
    assert "Análisis descriptivo — variables numéricas" in titulos
    assert "Frecuencias — variables categóricas" in titulos
    assert flujo.state == "sin_faltantes"
    res = flujo.analizar(target_col="Edad", group_col="Sexo")
    assert res["dataset_summary"]["target_variable"] == "Edad"


def test_flujo_entregable_inicial_con_faltantes():
    flujo = PaperStatsFlow(_df_con_faltantes())
    d = flujo.entregable_inicial()
    assert d.titulo == "Datos faltantes"
    assert any(s.titulo == "Datos faltantes" for s in d.secciones)
    assert any(s.titulo == "Recomendación de imputación" for s in d.secciones)
    assert flujo.state == "esperando_decision"
    with pytest.raises(ValueError):
        flujo.analizar()


def test_flujo_entregable_missing_sin_diagnose():
    flujo = PaperStatsFlow(_df_con_faltantes())
    with pytest.raises(ValueError):
        flujo.entregable_missing()


def test_flujo_entregable_analisis():
    flujo = PaperStatsFlow(_df_sin_faltantes())
    d = flujo.entregable_analisis(
        "¿Relación Ingresos~Egresos?",
        {"metodo": "Spearman", "resultado": {"estadistico": "rho=0.9"}},
    )
    assert any(s.titulo == "Pregunta u objetivo" for s in d.secciones)
    assert "¿Relación Ingresos~Egresos?" in DeliverableGenerator.render_markdown(d)
