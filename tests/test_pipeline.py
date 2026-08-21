"""
Pruebas de integración del pipeline E1–E6 (Etapa E7).
"""

import json

import numpy as np
import pandas as pd
import pytest

from src.analysis.dataset_analyzer import DatasetStatisticalAnalyzer
from src.missing_data import (
    MissingDataPipeline,
    ValidationRevisionError,
    induce_missing,
    run_pipeline,
)
from src.reports.generator import ReportGenerator

REPORT_KEYS = [
    "dataset_summary",
    "variable_classification",
    "data_quality",
    "eda",
    "diagnostics",
    "assumptions_status",
    "recommendations",
    "executed_test_results",
    "pedagogical_explanation",
]


@pytest.fixture
def df_completo():
    rng = np.random.default_rng(0)
    n = 60
    return pd.DataFrame({
        "x": rng.normal(50, 5, n),
        "y": 2.0 * rng.normal(50, 5, n) + rng.normal(0, 3, n),
        "g": ["A", "B"] * 30,
    })


@pytest.fixture
def df_con_faltantes(df_completo):
    return induce_missing(df_completo, columns=["x", "y"], fraction=0.2, random_state=1)


# ---------------------------------------------------------------------------
# Flujo principal
# ---------------------------------------------------------------------------


def test_sin_faltantes_continua_hacia_eda(df_completo):
    result = MissingDataPipeline().run(df_completo, impute=False)
    assert result.status == "sin_faltantes"
    assert result.continued is True
    assert result.detection_report is not None
    assert result.diagnostics_report is None
    assert result.candidate_methods == []
    assert result.imputed_df is None
    assert result.n_imputed_cells == 0


def test_sin_faltantes_con_impute_true_no_imputa(df_completo):
    result = MissingDataPipeline().run(df_completo, impute=True)
    assert result.status == "sin_faltantes"
    assert result.imputed_df is None
    assert result.validation_report is None


def test_con_faltantes_impute_false_no_imputa(df_con_faltantes):
    result = MissingDataPipeline().run(df_con_faltantes, impute=False)
    assert result.status == "con_faltantes"
    assert result.detection_report is not None
    assert result.diagnostics_report is not None
    assert result.candidate_methods
    assert result.evaluation_report is None
    # P-FLOW: con impute=False se recomienda el método de imputación (E5)
    # pero NO se imputa.
    assert result.selection_report is not None
    assert result.validation_report is None
    assert result.imputed_df is None


def test_con_faltantes_impute_true_ejecuta_e1_a_e6(df_con_faltantes):
    result = MissingDataPipeline().run(df_con_faltantes, impute=True)
    assert result.status == "imputado"
    assert result.detection_report is not None
    assert result.diagnostics_report is not None
    assert result.candidate_methods
    assert result.evaluation_report is not None
    assert result.selection_report is not None
    assert result.validation_report is not None
    assert result.validation_verdict in ("Aceptable", "Revisar")
    assert result.imputed_df is not None
    assert result.n_imputed_cells > 0


def test_imputacion_aceptable_continua(df_con_faltantes):
    result = MissingDataPipeline().run(df_con_faltantes, impute=True)
    assert result.validation_verdict == "Aceptable"
    assert result.continued is True


def test_validacion_revisar_no_continua_silenciosamente(df_con_faltantes):
    df = df_con_faltantes.copy()
    with pytest.raises(ValidationRevisionError):
        MissingDataPipeline().run(df, impute=True, identifier_columns=["x"])
    result = MissingDataPipeline().run(
        df, impute=True, identifier_columns=["x"], strict=False
    )
    assert result.validation_verdict == "Revisar"
    assert result.continued is False
    assert result.validation_report is not None


# ---------------------------------------------------------------------------
# No mutación del DataFrame original
# ---------------------------------------------------------------------------


def test_dataframe_original_no_se_modifica(df_con_faltantes):
    antes = df_con_faltantes.copy()
    MissingDataPipeline().run(df_con_faltantes, impute=True)
    pd.testing.assert_frame_equal(df_con_faltantes, antes)
    MissingDataPipeline().run(df_con_faltantes, impute=False)
    pd.testing.assert_frame_equal(df_con_faltantes, antes)


# ---------------------------------------------------------------------------
# Reportes y trazabilidad
# ---------------------------------------------------------------------------


def test_todos_los_reportes_e1_a_e6_disponibles(df_con_faltantes):
    result = MissingDataPipeline().run(df_con_faltantes, impute=True)
    assert result.detection_report is not None
    assert result.diagnostics_report is not None
    assert result.evaluation_report is not None
    assert result.selection_report is not None
    assert result.validation_report is not None
    assert isinstance(result.candidate_methods, list) and result.candidate_methods


def test_registra_metodo_seleccionado_y_por_que(df_con_faltantes):
    result = MissingDataPipeline().run(df_con_faltantes, impute=True)
    assert result.applied_methods
    for info in result.applied_methods.values():
        assert info["method"]
        assert info["reasons"]


def test_registra_variables_omitidas(df_con_faltantes):
    result = MissingDataPipeline().run(
        df_con_faltantes, impute=True, identifier_columns=["x"], strict=False
    )
    assert "x" in result.skipped_variables


# ---------------------------------------------------------------------------
# DatasetStatisticalAnalyzer
# ---------------------------------------------------------------------------


def test_analyzer_incluye_seccion_missing_data(df_con_faltantes):
    results = DatasetStatisticalAnalyzer(df_con_faltantes).analyze()
    assert "missing_data" in results
    md = results["missing_data"]
    assert md["status"] == "con_faltantes"
    assert md["detection"] is not None
    assert md["diagnostics"] is not None
    assert md["candidate_methods"]
    assert md["imputed"] is False


def test_analyzer_sin_faltantes(df_completo):
    results = DatasetStatisticalAnalyzer(df_completo).analyze()
    md = results["missing_data"]
    assert md["status"] == "sin_faltantes"
    assert md["detection"]["total_missing_values"] == 0


def test_analyzer_conserva_comportamiento_anterior(df_con_faltantes):
    results = DatasetStatisticalAnalyzer(df_con_faltantes).analyze(
        target_col="x", group_col="g"
    )
    for key in REPORT_KEYS:
        assert key in results
    assert results["dataset_summary"]["rows"] == len(df_con_faltantes)
    assert results["dataset_summary"]["columns"] == len(df_con_faltantes.columns)


# ---------------------------------------------------------------------------
# Serialización y reproducibilidad
# ---------------------------------------------------------------------------


def test_serializacion_json(df_completo, df_con_faltantes):
    json.dumps(MissingDataPipeline().run(df_completo, impute=False).to_dict())
    json.dumps(MissingDataPipeline().run(df_con_faltantes, impute=False).to_dict())
    payload = json.dumps(MissingDataPipeline().run(df_con_faltantes, impute=True).to_dict())
    assert isinstance(payload, str)
    loaded = json.loads(payload)
    assert loaded["status"] == "imputado"
    assert loaded["validation_verdict"] in ("Aceptable", "Revisar")
    assert "imputed_df" not in loaded


def test_reproducibilidad(df_con_faltantes):
    r1 = MissingDataPipeline().run(df_con_faltantes, impute=True)
    r2 = MissingDataPipeline().run(df_con_faltantes, impute=True)
    assert r1.to_dict() == r2.to_dict()


def test_reproducibilidad_sin_imputacion(df_con_faltantes):
    r1 = MissingDataPipeline().run(df_con_faltantes, impute=False)
    r2 = MissingDataPipeline().run(df_con_faltantes, impute=False)
    assert r1.to_dict() == r2.to_dict()


# ---------------------------------------------------------------------------
# ReportGenerator
# ---------------------------------------------------------------------------


def test_reporte_missing_data(df_con_faltantes, tmp_path):
    result = MissingDataPipeline().run(df_con_faltantes, impute=True)
    report_file = tmp_path / "faltantes.md"
    md_text = ReportGenerator().generate_missing_data_report(
        result.to_dict(), output_path=report_file
    )
    assert report_file.exists()
    assert "INFORME DE DATOS FALTANTES" in md_text
    assert "VALIDACIÓN (E6)" in md_text


def test_reporte_missing_data_sin_faltantes(df_completo, tmp_path):
    result = MissingDataPipeline().run(df_completo, impute=False)
    md_text = ReportGenerator().generate_missing_data_report(result.to_dict())
    assert "sin_faltantes" in md_text


# ---------------------------------------------------------------------------
# run_pipeline
# ---------------------------------------------------------------------------


def test_run_pipeline_conveniencia(df_con_faltantes):
    result = run_pipeline(df_con_faltantes, impute=True)
    assert result.status == "imputado"
    assert result.detection_report is not None
