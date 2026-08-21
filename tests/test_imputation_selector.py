"""
Pruebas unitarias para la selección de métodos de imputación (Etapa E5).

Se construyen los reportes E1/E2 con las clases reales del módulo y, cuando
procede, un reporte E4 construido manualmente para probar la integración
explícita de la evidencia empírica.
"""

import json

import numpy as np
import pandas as pd
import pytest

from src.analysis.profile import DatasetProfile, build_dataset_profile
from src.missing_data import (
    DEFAULT_WEIGHTS,
    ImputationEvaluationReport,
    ImputationSelectionReport,
    ImputationSelector,
    MethodEvaluation,
    MissingDataDetector,
    MissingDataDiagnostics,
    VariableRecommendation,
)
from src.missing_data.diagnostics import (
    MechanismAssessment,
    MissingnessAssociation,
    MissingnessDiagnosticsReport,
)
from src.missing_data.methods import MethodCapabilities


def _reports(df, *, target=None, temporal=None, identifier_columns=None):
    profile = build_dataset_profile(
        df,
        target=target,
        temporal=temporal,
        identifier_columns=identifier_columns,
    )
    missing_report = MissingDataDetector().detect(df)
    diagnostics = MissingDataDiagnostics().diagnose(df)
    return profile, missing_report, diagnostics


def _numeric_df(n=100, missing_col="y", pct=0.2, seed=3):
    rng = np.random.default_rng(seed)
    x = rng.normal(50, 10, n)
    y = 2.0 * x + rng.normal(0, 5, n)
    y[np.arange(n) < int(pct * n)] = np.nan
    df = pd.DataFrame({"x": x, "y": y})
    return df


def _mixed_df():
    n = 80
    rng = np.random.default_rng(1)
    df = pd.DataFrame(
        {
            "num1": rng.normal(0, 1, n),
            "num2": rng.normal(5, 2, n),
            "cat": rng.choice(["a", "b", "c"], n),
        }
    )
    df.loc[0:15, "num1"] = np.nan
    df.loc[0:25, "cat"] = np.nan
    return df


def _e4_report(extra_error_method=None):
    methods = [
        MethodEvaluation(
            method="media",
            error=None,
            n_induced=40,
            n_repeats=1,
            per_column={"y": {}},
            global_metrics={"mae": 10.0, "rmse": 12.0},
        ),
        MethodEvaluation(
            method="mice",
            error=None,
            n_induced=40,
            n_repeats=1,
            per_column={"y": {}},
            global_metrics={"mae": 1.0, "rmse": 1.5},
        ),
    ]
    if extra_error_method:
        methods.append(
            MethodEvaluation(
                method=extra_error_method,
                error="no se pudo ajustar",
                n_induced=40,
                n_repeats=1,
                per_column={"y": {}},
                global_metrics={},
            )
        )
    return ImputationEvaluationReport(
        mechanism="MCAR",
        fraction=0.2,
        predictor=None,
        n_repeats=1,
        random_state=42,
        numeric_columns=["y"],
        categorical_columns=[],
        methods=methods,
        ranking={"media": 12.0, "mice": 1.5},
        reproduction={},
    )


# ---------------------------------------------------------------------------
# Estructura y puntaje
# ---------------------------------------------------------------------------


def test_select_variable_con_recomendacion_y_alternativas():
    df = _numeric_df()
    profile, missing_report, diagnostics = _reports(df)
    report = ImputationSelector().select(profile, missing_report, diagnostics)

    assert isinstance(report, ImputationSelectionReport)
    assert "y" in report.variables
    rec = report.variables["y"]
    assert isinstance(rec, VariableRecommendation)
    assert rec.recommended is not None
    assert rec.recommended.score is not None
    assert 0.0 <= rec.recommended.score <= 1.0
    scored = [s for s in rec.all_scores if not s.excluded]
    assert rec.alternatives == scored[1:]
    assert rec.recommended == scored[0]


def test_puntajes_en_rango_y_orden_descendente():
    df = _mixed_df()
    profile, missing_report, diagnostics = _reports(df)
    report = ImputationSelector().select(profile, missing_report, diagnostics)
    for var, rec in report.variables.items():
        scores = [s.score for s in rec.all_scores if not s.excluded]
        assert all(0.0 <= s <= 1.0 for s in scores)
        assert scores == sorted(scores, reverse=True)


def test_cada_metodo_no_excluido_tiene_siete_razones():
    df = _numeric_df()
    profile, missing_report, diagnostics = _reports(df)
    report = ImputationSelector().select(profile, missing_report, diagnostics)
    for var, rec in report.variables.items():
        for s in rec.all_scores:
            if s.excluded:
                continue
            assert not s.excluded
            assert len(s.reasons) == 7
            assert all(reason.startswith(tuple(DEFAULT_WEIGHTS.keys())) for reason in s.reasons)
            assert s.exclusion_reason is None


def test_metodos_excluidos_tienen_razon():
    df = _numeric_df()
    profile, missing_report, diagnostics = _reports(df)
    report = ImputationSelector().select(profile, missing_report, diagnostics)
    rec = report.variables["y"]
    scored = {s.method for s in rec.all_scores if not s.excluded}
    excluded = {s.method for s in rec.all_scores if s.excluded}
    assert scored
    assert excluded  # métodos temporales quedan fuera sin estructura temporal


# ---------------------------------------------------------------------------
# Puertas duras
# ---------------------------------------------------------------------------


def test_puerta_dura_temporal():
    df = _numeric_df()
    profile, missing_report, diagnostics = _reports(df, temporal=False)
    report = ImputationSelector().select(profile, missing_report, diagnostics)
    rec = report.variables["y"]
    by_method = {s.method: s for s in rec.all_scores}
    assert "interpolacion_lineal" in by_method
    assert "locf" in by_method
    for name in ("interpolacion_lineal", "locf"):
        assert by_method[name].excluded
        assert "temporal" in by_method[name].exclusion_reason
    for name, s in by_method.items():
        if s.method not in ("interpolacion_lineal", "locf"):
            assert not s.excluded


def test_puerta_dura_tipo_categorica():
    df = _mixed_df()
    profile, missing_report, diagnostics = _reports(df)
    report = ImputationSelector().select(profile, missing_report, diagnostics)
    rec = report.variables["cat"]
    assert rec.recommended is not None
    assert rec.recommended.method in {"moda", "constante"}
    for s in rec.all_scores:
        if not s.excluded:
            assert s.method in {"moda", "constante"}


def test_puerta_dura_una_sola_variable():
    df = pd.DataFrame({"a": [1.0, 2.0, np.nan, 4.0, 5.0] * 20})
    profile, missing_report, diagnostics = _reports(df)
    assert profile.n_variables == 1
    report = ImputationSelector().select(profile, missing_report, diagnostics)
    rec = report.variables["a"]
    assert rec.recommended is not None
    assert rec.recommended.method in {"media", "mediana", "moda", "constante"}
    for s in rec.all_scores:
        if s.excluded:
            if s.method in ("interpolacion_lineal", "locf"):
                assert "temporal" in s.exclusion_reason
            else:
                assert "predictoras" in s.exclusion_reason
        else:
            assert s.method in {"media", "mediana", "moda", "constante"}


def test_variable_identificadora_excluida():
    df = pd.DataFrame(
        {
            "id": ["r0", None, "r2", "r3", "r4"] * 10,
            "val": np.r_[np.array([np.nan] * 10), np.arange(40.0)],
        }
    )
    profile, missing_report, diagnostics = _reports(df, identifier_columns=["id"])
    report = ImputationSelector().select(profile, missing_report, diagnostics)
    assert "id" in report.variables
    rec = report.variables["id"]
    assert rec.recommended is None
    assert rec.alternatives == []
    assert all(s.excluded for s in rec.all_scores)
    assert "identificador" in rec.all_scores[0].exclusion_reason


def test_variable_tiempo_excluida():
    df = pd.DataFrame(
        {
            "fecha": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03", None, "2020-01-05"] * 10),
            "val": np.arange(50.0),
        }
    )
    profile, missing_report, diagnostics = _reports(df)
    assert profile.temporal
    report = ImputationSelector().select(profile, missing_report, diagnostics)
    assert "fecha" in report.variables
    rec = report.variables["fecha"]
    assert rec.recommended is None
    assert rec.alternatives == []
    assert all(s.excluded for s in rec.all_scores)
    assert "tiempo" in rec.all_scores[0].exclusion_reason.lower()


# ---------------------------------------------------------------------------
# Uso de E2
# ---------------------------------------------------------------------------


def test_e2_cuestiona_mcar_penaliza_valor_unico():
    df = _numeric_df(n=100)
    profile, missing_report, _ = _reports(df)
    associations = [
        MissingnessAssociation(
            variable="y",
            associated_with="x",
            associated_variable_type="Cuantitativa continua",
            test="Mann-Whitney U",
            statistic=1.0,
            p_value=0.001,
            adjusted_p_value=0.002,
            n_observed=80,
            n_missing=20,
            conclusion="Evidencia contra MCAR.",
        )
    ]
    diagnostics = MissingnessDiagnosticsReport(
        status="ok",
        total_observations=100,
        total_variables=2,
        variables_with_missing=["y"],
        co_missing_counts={},
        systematic_patterns=[],
        observations_missing_distribution={},
        associations=associations,
        skipped_comparisons=[],
        multiple_comparisons={},
        mechanism=MechanismAssessment(
            evidence="Se detectó 1 asociación significativa. Esto constituye evidencia en contra del supuesto MCAR.",
            tests_performed=1,
            significant_comparisons=["y | x"],
            limitations=[],
            cannot_infer=[],
            recommendation="Considere modelar el mecanismo de ausencia.",
        ),
        reproduction={},
    )
    report = ImputationSelector().select(profile, missing_report, diagnostics)
    assert any("MCAR" in w for w in report.warnings)
    rec = report.variables["y"]
    scores = {s.method: s for s in rec.all_scores}
    assert scores["mice"].components["robustness"] > scores["media"].components["robustness"]


def test_e2_sin_evidencia_no_adviere_mcar():
    df = _numeric_df()
    profile, missing_report, diagnostics = _reports(df)
    report = ImputationSelector().select(profile, missing_report, diagnostics)
    assert not any("en contra del supuesto MCAR" in w for w in report.warnings)


# ---------------------------------------------------------------------------
# Uso de E4
# ---------------------------------------------------------------------------


def test_e4_mejora_la_robustez_del_metodo_mejor():
    df = _numeric_df()
    profile, missing_report, diagnostics = _reports(df)
    e4 = _e4_report()
    report = ImputationSelector().select(profile, missing_report, diagnostics, evaluation=e4)
    rec = report.variables["y"]
    scores = {s.method: s for s in rec.all_scores}
    mice = scores["mice"]
    assert mice.e4_evidence is not None
    assert mice.e4_evidence["available"] is True
    assert mice.e4_evidence["global_metrics"]["rmse"] == 1.5
    assert mice.components["robustness"] > scores["media"].components["robustness"]


def test_e4_error_se_registra_como_caveat():
    df = _numeric_df()
    profile, missing_report, diagnostics = _reports(df)
    e4 = _e4_report(extra_error_method="knn")
    report = ImputationSelector().select(profile, missing_report, diagnostics, evaluation=e4)
    scores = {s.method: s for s in report.variables["y"].all_scores}
    assert any("E4" in c and "falló" in c for c in scores["knn"].caveats)


def test_e4_variable_no_evaluada_queda_registrada():
    df = _numeric_df()
    profile, missing_report, diagnostics = _reports(df)
    e4 = _e4_report()
    report = ImputationSelector().select(profile, missing_report, diagnostics, evaluation=e4)
    scores = {s.method: s for s in report.variables["y"].all_scores}
    assert scores["media"].e4_evidence is not None
    assert scores["media"].e4_evidence["available"] is True
    assert any("E4 disponible" in c for c in scores["media"].caveats)


# ---------------------------------------------------------------------------
# Advertencias
# ---------------------------------------------------------------------------


def test_advertencia_mayor_50_porciento():
    df = pd.DataFrame({"z": np.r_[np.array([np.nan] * 60), np.arange(40.0)]})
    profile, missing_report, diagnostics = _reports(df)
    report = ImputationSelector().select(profile, missing_report, diagnostics)
    rec = report.variables["z"]
    assert any("50%" in w for w in rec.warnings)
    assert any("50%" in w for w in report.warnings)
    assert rec.recommended is not None


def test_advertencia_target():
    df = _numeric_df()
    profile, missing_report, diagnostics = _reports(df, target="y")
    report = ImputationSelector().select(profile, missing_report, diagnostics)
    rec = report.variables["y"]
    assert any("objetivo" in w.lower() for w in rec.warnings)
    assert any("objetivo" in c.lower() for c in rec.recommended.caveats)


def test_advertencia_muestra_pequena():
    df = _numeric_df(n=15)
    profile, missing_report, diagnostics = _reports(df)
    report = ImputationSelector().select(profile, missing_report, diagnostics)
    assert any("muestral" in w.lower() for w in report.warnings)


# ---------------------------------------------------------------------------
# Ranking por grupo y reproducibilidad
# ---------------------------------------------------------------------------


def test_ranking_por_grupo():
    df = _mixed_df()
    profile, missing_report, diagnostics = _reports(df)
    report = ImputationSelector().select(profile, missing_report, diagnostics)
    assert "Numericas" in report.group_ranking
    assert "Categoricas" in report.group_ranking
    for group, entries in report.group_ranking.items():
        means = [e["mean_score"] for e in entries]
        assert means == sorted(means, reverse=True)
        for e in entries:
            assert set(e) == {"method", "mean_score", "n_variables", "best_for"}
            assert 0.0 <= e["mean_score"] <= 1.0


def test_reproducibilidad():
    df = _mixed_df()
    profile, missing_report, diagnostics = _reports(df)
    r1 = ImputationSelector().select(profile, missing_report, diagnostics)
    r2 = ImputationSelector().select(profile, missing_report, diagnostics)
    assert r1.to_dict() == r2.to_dict()


def test_reporte_json_serializable():
    df = _mixed_df()
    profile, missing_report, diagnostics = _reports(df)
    report = ImputationSelector().select(profile, missing_report, diagnostics)
    payload = json.dumps(report.to_dict())
    assert isinstance(payload, str)


# ---------------------------------------------------------------------------
# Pesos configurables
# ---------------------------------------------------------------------------


def test_pesos_configurables_afectan_score():
    df = _numeric_df()
    profile, missing_report, diagnostics = _reports(df)
    weights = dict(DEFAULT_WEIGHTS)
    weights["type_fit"] = 1.0
    for k in weights:
        if k != "type_fit":
            weights[k] = 0.0
    report = ImputationSelector(weights=weights).select(profile, missing_report, diagnostics)
    for s in report.variables["y"].all_scores:
        if not s.excluded:
            assert s.score == pytest.approx(1.0)


def test_pesos_invalidos():
    df = _numeric_df()
    profile, missing_report, diagnostics = _reports(df)
    with pytest.raises(ValueError):
        ImputationSelector(weights={})
    bad = dict(DEFAULT_WEIGHTS)
    bad["robustness"] = -0.1
    with pytest.raises(ValueError):
        ImputationSelector(weights=bad)
    zero = {k: 0.0 for k in DEFAULT_WEIGHTS}
    with pytest.raises(ValueError):
        ImputationSelector(weights=zero)
    with pytest.raises(TypeError):
        ImputationSelector(weights=[0.1] * 7)


# ---------------------------------------------------------------------------
# Validación de entradas
# ---------------------------------------------------------------------------


def test_validacion_entradas():
    df = _numeric_df()
    profile, missing_report, diagnostics = _reports(df)
    with pytest.raises(TypeError):
        ImputationSelector().select(df, missing_report, diagnostics)
    with pytest.raises(TypeError):
        ImputationSelector().select(profile, df, diagnostics)
    with pytest.raises(TypeError):
        ImputationSelector().select(profile, missing_report, df)
    with pytest.raises(TypeError):
        ImputationSelector().select(profile, missing_report, diagnostics, evaluation=df)
    with pytest.raises(ValueError):
        ImputationSelector(alpha=1.5)


def test_sin_variables_con_faltantes():
    df = pd.DataFrame({"a": np.arange(50.0), "b": np.arange(50.0) * 2})
    profile, missing_report, diagnostics = _reports(df)
    assert missing_report.variables_with_missing == []
    report = ImputationSelector().select(profile, missing_report, diagnostics)
    assert report.variables == {}
    assert report.group_ranking == {}


def test_score_usa_todos_los_pesos_documentados():
    df = _numeric_df()
    profile, missing_report, diagnostics = _reports(df)
    report = ImputationSelector().select(profile, missing_report, diagnostics)
    assert report.weights == dict(DEFAULT_WEIGHTS)
    assert abs(sum(report.weights.values()) - 1.0) < 1e-9
    for s in report.variables["y"].all_scores:
        if s.excluded:
            continue
        expected = sum(
            report.weights[c] * s.components[c] for c in report.weights
        )
        assert s.score == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Puerta estructural de 'regresion' (predictora numérica completa requerida)
# ---------------------------------------------------------------------------


def _sin_predictora_completa_df():
    return pd.DataFrame(
        {
            "a": [1.0, np.nan, 3.0, 4.0, np.nan, 6.0],
            "b": [np.nan, 2.0, np.nan, 4.0, 5.0, np.nan],
        }
    )


def test_regresion_excluida_sin_predictora_numérica_completa():
    df = _sin_predictora_completa_df()
    profile, missing_report, diagnostics = _reports(df)
    report = ImputationSelector().select(profile, missing_report, diagnostics)
    for var in ("a", "b"):
        by_method = {s.method: s for s in report.variables[var].all_scores}
        assert by_method["regresion"].excluded
        assert "missing_count == 0" in by_method["regresion"].exclusion_reason
        assert not any(
            s.method == "regresion" for s in report.variables[var].all_scores
            if not s.excluded
        )


def test_regresion_candidata_con_predictora_completa():
    df = _numeric_df()  # x completa, y con faltantes
    profile, missing_report, diagnostics = _reports(df)
    report = ImputationSelector().select(profile, missing_report, diagnostics)
    by_method = {s.method: s for s in report.variables["y"].all_scores}
    assert not by_method["regresion"].excluded
    assert any(
        s.method == "regresion" for s in report.variables["y"].all_scores
        if not s.excluded
    )


def test_regresion_no_excluida_con_predictora_completa_aun_con_otras_faltantes():
    df = pd.DataFrame(
        {
            "x": np.arange(20.0),
            "y": np.r_[np.array([np.nan] * 5), np.arange(15.0)],
            "z": np.r_[np.array([np.nan] * 8), np.arange(12.0)],
        }
    )
    profile, missing_report, diagnostics = _reports(df)
    report = ImputationSelector().select(profile, missing_report, diagnostics)
    for var in ("y", "z"):
        by_method = {s.method: s for s in report.variables[var].all_scores}
        assert not by_method["regresion"].excluded
        assert by_method["regresion"].exclusion_reason is None


def test_razon_exclusion_regresion_clara_y_serializable():
    df = _sin_predictora_completa_df()
    profile, missing_report, diagnostics = _reports(df)
    report = ImputationSelector().select(profile, missing_report, diagnostics)
    score = next(
        s for s in report.variables["a"].all_scores if s.method == "regresion"
    )
    assert score.excluded
    assert isinstance(score.exclusion_reason, str)
    assert len(score.exclusion_reason) > 20
    assert "regresion" in score.exclusion_reason
    assert "completamente observada" in score.exclusion_reason
    payload = json.dumps(score.to_dict())
    assert isinstance(payload, str)


def test_regresion_excluida_se_mantiene_en_all_scores():
    df = _sin_predictora_completa_df()
    profile, missing_report, diagnostics = _reports(df)
    report = ImputationSelector().select(profile, missing_report, diagnostics)
    rec = report.variables["a"]
    names = {s.method for s in rec.all_scores}
    assert "regresion" in names
    assert all(
        s.method == "regresion" for s in rec.all_scores if s.method == "regresion"
    ) and rec.recommended.method != "regresion"


# ---------------------------------------------------------------------------
# Caveats documentados de E5 (evidencia E4 exploratoria y sensibilidad a pesos)
# ---------------------------------------------------------------------------


def test_caveat_e4_exploratorio_en_advertencias():
    df = _numeric_df()
    profile, missing_report, diagnostics = _reports(df)
    e4 = _e4_report()
    report = ImputationSelector().select(profile, missing_report, diagnostics, evaluation=e4)
    assert any("exploratoria" in w for w in report.warnings)


def test_advertencia_sensibilidad_a_pesos_con_diferencias_pequeñas():
    df = _numeric_df()
    profile, missing_report, diagnostics = _reports(df)
    report = ImputationSelector().select(profile, missing_report, diagnostics)
    rec = report.variables["y"]
    assert any("sensible a los pesos" in w for w in rec.warnings)