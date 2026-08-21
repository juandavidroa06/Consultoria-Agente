"""
Tests de P-FLOW: orquestador PaperStatsFlow y el flujo principal.

Verifican que el primer punto de entrada sea el diagnóstico, que la etapa
inicial sea estrictamente diagnóstica (sin inferencia/EDA), que la imputación
requiera decisión explícita y que el análisis posterior se delegue en
DatasetStatisticalAnalyzer sin modificarlo.
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import src.orchestration.flow as flow_mod
from src.missing_data.pipeline import MissingDataPipeline
from src.orchestration.flow import PaperStatsFlow

CLAVES_DIAGNOSE = (
    "estado",
    "dataset",
    "perfil",
    "calidad",
    "clasificacion_variables",
    "missing_data",
    "recomendacion_imputacion",
    "mensaje_estado",
)


@pytest.fixture
def df_con_faltantes():
    rng = np.random.default_rng(42)
    n = 30
    df = pd.DataFrame(
        {
            "x": rng.normal(0, 1, n),
            "g": ["A", "B", "C"] * (n // 3),
        }
    )
    df.loc[[1, 2, 10], "x"] = np.nan
    df.loc[[5], "g"] = np.nan
    return df


@pytest.fixture
def df_completo():
    rng = np.random.default_rng(7)
    return pd.DataFrame(
        {"x": rng.normal(0, 1, 20), "y": rng.normal(0, 1, 20)}
    )


# ---------------------------------------------------------------------------
# diagnose(): diagnóstico estrictamente diagnóstica
# ---------------------------------------------------------------------------


def test_diagnose_con_faltantes_estado_esperando_decision(df_con_faltantes):
    result = PaperStatsFlow(df_con_faltantes).diagnose()

    assert result["estado"] == "esperando_decision"
    for key in CLAVES_DIAGNOSE:
        assert key in result
    assert result["missing_data"]["status"] == "con_faltantes"
    assert "faltantes" in result["mensaje_estado"]


def test_diagnose_es_estrictamente_diagnostica(df_con_faltantes):
    result = PaperStatsFlow(df_con_faltantes).diagnose()

    assert "executed_test_results" not in result
    assert "recommendations" not in result
    assert "eda" not in result
    assert "diagnostics" not in result
    assert "assumptions_status" not in result


def test_diagnose_sin_faltantes_estado_sin_faltantes(df_completo):
    flujo = PaperStatsFlow(df_completo)
    result = flujo.diagnose()

    assert result["estado"] == "sin_faltantes"
    assert result["missing_data"]["status"] == "sin_faltantes"
    assert result["recomendacion_imputacion"]["por_variable"] == {}
    assert flujo.state == "sin_faltantes"


def test_diagnose_acepta_ruta_de_archivo_csv(tmp_path, df_con_faltantes):
    csv_path = tmp_path / "datos.csv"
    df_con_faltantes.to_csv(csv_path, index=False)

    result = PaperStatsFlow(str(csv_path)).diagnose()

    assert result["dataset"]["file_name"] == "datos.csv"
    assert result["dataset"]["rows"] == len(df_con_faltantes)
    assert result["dataset"]["columns"] == len(df_con_faltantes.columns)


# ---------------------------------------------------------------------------
# Trazabilidad de la recomendación de imputación
# ---------------------------------------------------------------------------


def test_trazabilidad_recomendacion(df_con_faltantes):
    result = PaperStatsFlow(df_con_faltantes).diagnose()
    rec = result["recomendacion_imputacion"]

    assert rec["resumen"]
    assert isinstance(rec["pesos"], dict) and rec["pesos"]
    assert rec["por_variable"]

    for var, info in rec["por_variable"].items():
        assert info["variable_type"]
        assert info["missing_count"] >= 1
        assert info["missing_percentage"] > 0
        assert info["decision"] in (
            "metodo_unico",
            "comparar_alternativas",
            "sin_recomendacion",
        )
        assert info["justificacion"]
        assert isinstance(info["advertencias"], list)
        assert info["metodos_considerados"], "deben listarse los métodos considerados"
        for m in info["metodos_considerados"]:
            assert m["method"]
            assert isinstance(m["components"], dict)
            assert isinstance(m["reasons"], list)
            assert isinstance(m["excluded"], bool)


def test_decision_comparar_cuando_brecha_menor_al_umbral(df_con_faltantes, monkeypatch):
    # Un umbral enorme fuerza la rama "comparar_alternativas" en todas las
    # variables con al menos dos métodos no excluidos.
    monkeypatch.setattr(flow_mod, "SCORE_GAP_SENSITIVITY_THRESHOLD", 1000.0)

    result = PaperStatsFlow(df_con_faltantes).diagnose()
    decisiones = {
        info["decision"]
        for info in result["recomendacion_imputacion"]["por_variable"].values()
    }
    assert "comparar_alternativas" in decisiones


def test_sin_recomendacion_cuando_todo_excluido(df_con_faltantes):
    # Síntesis con un resultado del pipeline sin método recomendado (todas las
    # alternativas excluidas): no debe inventarse una certeza.
    vr = SimpleNamespace(
        recommended=None,
        variable_type="Cuantitativa continua",
        missing_count=3,
        missing_percentage=10.0,
        alternatives=[],
        all_scores=[],
        warnings=["todas las alternativas fueron excluidas."],
    )
    sel = SimpleNamespace(
        weights={"perfil": 0.5},
        warnings=["advertencia global"],
        variables={"x": vr},
    )
    pipeline_result = SimpleNamespace(status="con_faltantes", selection_report=sel)

    flujo = PaperStatsFlow(df_con_faltantes)
    rec = flujo._sintetizar_recomendacion(pipeline_result)

    info = rec["por_variable"]["x"]
    assert info["decision"] == "sin_recomendacion"
    assert info["metodo_recomendado"] is None
    assert "no hay método no excluido" in info["justificacion"].lower()


# ---------------------------------------------------------------------------
# Compuertas del flujo: imputación explícita y análisis posterior
# ---------------------------------------------------------------------------


def test_analizar_requiere_decision_antes_de_imputar(df_con_faltantes):
    flujo = PaperStatsFlow(df_con_faltantes)
    flujo.diagnose()

    with pytest.raises(ValueError, match="imputar"):
        flujo.analizar(target_col="x")


def test_imputar_requiere_decision_explicita(df_con_faltantes):
    flujo = PaperStatsFlow(df_con_faltantes)
    flujo.diagnose()

    with pytest.raises(ValueError, match="decisión explícita"):
        flujo.imputar()

    resultado = flujo.imputar(accept_recommendation=True)
    assert resultado["estado"] == "datos_preparados"
    assert resultado["validation_verdict"] == "Aceptable"
    assert resultado["n_imputed_cells"] >= 1
    imputed = resultado["imputed_df"]
    assert imputed["x"].isna().sum() == 0
    assert flujo.imputed_df is not None


def test_imputar_con_method_override(df_con_faltantes):
    flujo = PaperStatsFlow(df_con_faltantes)
    flujo.diagnose()

    resultado = flujo.imputar(method_override={"x": "mediana"})

    assert resultado["applied_methods"]["x"]["method"] == "mediana"
    assert resultado["estado"] == "datos_preparados"


def test_imputar_method_override_desconocido_raise(df_con_faltantes):
    flujo = PaperStatsFlow(df_con_faltantes)
    flujo.diagnose()

    with pytest.raises(ValueError, match="desconocido"):
        flujo.imputar(method_override={"x": "metodo_inexistente"})


def test_imputar_requiere_diagnose_previo(df_con_faltantes):
    flujo = PaperStatsFlow(df_con_faltantes)
    with pytest.raises(ValueError, match="diagnose"):
        flujo.imputar(accept_recommendation=True)


def test_analizar_tras_imputar_delega_en_analyzer(df_con_faltantes):
    flujo = PaperStatsFlow(df_con_faltantes)
    flujo.diagnose()
    flujo.imputar(accept_recommendation=True)

    analisis = flujo.analizar(target_col="x", group_col="g")

    assert "executed_test_results" in analisis
    assert "recommendations" in analisis
    assert "missing_data" in analisis


def test_analizar_sin_faltantes_permitido(df_completo):
    flujo = PaperStatsFlow(df_completo)
    flujo.diagnose()

    analisis = flujo.analizar(target_col="x")

    assert "executed_test_results" in analisis


# ---------------------------------------------------------------------------
# Informe de datos faltantes
# ---------------------------------------------------------------------------


def test_informe_missing_se_escribe(df_con_faltantes, tmp_path):
    flujo = PaperStatsFlow(df_con_faltantes)
    flujo.diagnose()

    out = tmp_path / "informe_missing.md"
    markdown = flujo.generar_informe_missing(out)

    assert "DATOS FALTANTES" in markdown
    assert out.exists()
    assert "faltantes" in out.read_text(encoding="utf-8").lower()


def test_informe_missing_requiere_diagnose(df_con_faltantes):
    flujo = PaperStatsFlow(df_con_faltantes)
    with pytest.raises(ValueError, match="diagnose"):
        flujo.generar_informe_missing()


# ---------------------------------------------------------------------------
# Cambio aditivo en MissingDataPipeline: recomendación sin imputar
# ---------------------------------------------------------------------------


def test_pipeline_impute_false_incluye_selection_report(df_con_faltantes):
    result = MissingDataPipeline().run(df_con_faltantes, impute=False)

    assert result.status == "con_faltantes"
    assert result.selection_report is not None
    assert result.imputed_df is None
    assert result.n_imputed_cells == 0

    serializado = result.to_dict()
    assert serializado["selection_report"] is not None
    assert serializado["selection_report"]["variables"]