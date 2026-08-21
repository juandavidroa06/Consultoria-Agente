"""
Pruebas unitarias para la evaluación artificial de imputación (Etapa E4).

Los valores esperados de las métricas se verifican de forma independiente
(reimplementando la inducción MCAR con numpy), no replicando la implementación.
"""

import json

import numpy as np
import pandas as pd
import pytest

from src.missing_data import (
    ArtificialMissingnessEvaluator,
    ImputationEvaluationReport,
    MethodEvaluation,
    KNNImputation,
    induce_missing,
)


@pytest.fixture
def evaluator():
    return ArtificialMissingnessEvaluator(random_state=42, n_repeats=1)


@pytest.fixture
def df_num():
    return pd.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "y": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0],
        }
    )


# ---------------------------------------------------------------------------
# Inducción de faltantes artificiales
# ---------------------------------------------------------------------------


def test_induce_missing_mcar_conteo():
    df = pd.DataFrame({"a": list(range(20)), "b": list(range(20))})
    result = induce_missing(df, fraction=0.2, mechanism="MCAR", random_state=7)
    assert int(result["a"].isna().sum()) == 4
    assert int(result["b"].isna().sum()) == 4


def test_induce_missing_mcar_reproducible():
    df = pd.DataFrame({"a": list(range(30))})
    r1 = induce_missing(df, fraction=0.3, random_state=5)
    r2 = induce_missing(df, fraction=0.3, random_state=5)
    pd.testing.assert_frame_equal(r1, r2)


def test_induce_missing_no_modifica_original():
    df = pd.DataFrame({"a": list(range(10))})
    original = df.copy()
    induce_missing(df, fraction=0.2, random_state=1)
    pd.testing.assert_frame_equal(df, original)


def test_induce_missing_mcar_categorico():
    df = pd.DataFrame({"z": ["a", "b", "c", "d"] * 5})
    result = induce_missing(df, fraction=0.25, random_state=3)
    assert result["z"].isna().sum() == 5


def test_induce_missing_mar_concentra_en_predictor():
    n = 10
    x = np.arange(1, n + 1, dtype=float)
    df = pd.DataFrame({"x": x, "y": x * 2})
    result = induce_missing(df, fraction=0.3, mechanism="MAR", predictor="x", random_state=7)
    miss = result["y"].isna()
    assert miss.sum() == 3
    assert miss.iloc[n - 1]  # la fila de mayor x siempre queda oculta
    assert not miss.iloc[0]  # la de menor x no


def test_induce_missing_mar_reproducible():
    n = 20
    x = np.arange(n, dtype=float)
    df = pd.DataFrame({"x": x, "y": x * 3})
    r1 = induce_missing(df, fraction=0.3, mechanism="MAR", predictor="x", random_state=9)
    r2 = induce_missing(df, fraction=0.3, mechanism="MAR", predictor="x", random_state=9)
    pd.testing.assert_frame_equal(r1, r2)


def test_induce_missing_columns_restrictivo():
    df = pd.DataFrame({"a": list(range(10)), "b": list(range(10))})
    result = induce_missing(df, columns=["a"], fraction=0.3, random_state=2)
    assert result["a"].isna().sum() > 0
    assert result["b"].isna().sum() == 0


def test_induce_missing_validaciones():
    df = pd.DataFrame({"a": list(range(10))})
    with pytest.raises(ValueError):
        induce_missing(df, fraction=0.0)
    with pytest.raises(ValueError):
        induce_missing(df, fraction=1.0)
    with pytest.raises(ValueError):
        induce_missing(df, fraction=0.2, mechanism="MNAR")
    with pytest.raises(ValueError):
        induce_missing(df, fraction=0.2, mechanism="MAR")
    with pytest.raises(KeyError):
        induce_missing(df, fraction=0.2, mechanism="MAR", predictor="nope")
    with pytest.raises(KeyError):
        induce_missing(df, columns=["nope"], fraction=0.2)
    with pytest.raises(TypeError):
        induce_missing([1, 2, 3], fraction=0.2)


# ---------------------------------------------------------------------------
# Evaluador: métricas y comportamiento
# ---------------------------------------------------------------------------


def test_evaluate_media_mae_esperado(evaluator, df_num):
    df = df_num[["x"]]
    seed = 42
    n = len(df)
    rng = np.random.default_rng(seed)
    k = int(0.2 * n)
    idx = rng.choice(n, size=k, replace=False)
    true = df["x"].to_numpy(dtype=float)[idx]
    observed = np.delete(df["x"].to_numpy(dtype=float), idx)
    expected_mae = float(np.mean(np.abs(true - observed.mean())))

    report = evaluator.evaluate(df, methods=["media"], fraction=0.2)
    assert isinstance(report, ImputationEvaluationReport)
    ev = report.methods[0]
    assert isinstance(ev, MethodEvaluation)
    assert ev.method == "media"
    assert ev.error is None
    assert ev.global_metrics["mae"] == pytest.approx(expected_mae)


def test_evaluate_regresion_exacta():
    n = 30
    x = np.arange(n, dtype=float)
    y = 2.0 * x + 1.0
    df = pd.DataFrame({"x": x, "y": y})
    report = ArtificialMissingnessEvaluator(random_state=10).evaluate(
        df, columns=["y"], methods=["regresion"], fraction=0.3
    )
    ev = next(m for m in report.methods if m.method == "regresion")
    assert ev.global_metrics["rmse"] < 1e-6
    assert ev.global_metrics["mae"] < 1e-6


def test_evaluate_media_peor_que_regresion():
    n = 40
    x = np.arange(n, dtype=float)
    y = 3.0 * x + 5.0
    df = pd.DataFrame({"x": x, "y": y})
    report = ArtificialMissingnessEvaluator(random_state=3).evaluate(
        df, columns=["y"], methods=["media", "regresion"], fraction=0.3
    )
    metrics = {m.method: m.global_metrics["rmse"] for m in report.methods}
    assert metrics["media"] > metrics["regresion"]


def test_evaluate_ranking_ordenado():
    n = 30
    x = np.arange(n, dtype=float)
    y = 2.0 * x
    df = pd.DataFrame({"x": x, "y": y})
    report = ArtificialMissingnessEvaluator(random_state=5).evaluate(
        df, columns=["y"], methods=["media", "regresion"], fraction=0.3
    )
    assert report.ranking["regresion"] < report.ranking["media"]
    assert list(report.ranking) == sorted(report.ranking, key=report.ranking.get)


def test_evaluate_categorico_accuracy():
    z = ["a", "b"] * 20
    df = pd.DataFrame({"z": z})
    report = ArtificialMissingnessEvaluator(random_state=1).evaluate(
        df, columns=["z"], methods=["moda"], fraction=0.3
    )
    ev = report.methods[0]
    assert 0.0 < ev.global_metrics["accuracy"] <= 1.0


def test_evaluate_no_modifica_original(evaluator, df_num):
    original = df_num.copy()
    evaluator.evaluate(df_num, methods=["media", "knn"], fraction=0.2)
    pd.testing.assert_frame_equal(df_num, original)


def test_evaluate_reproducible_iterativo_y_mice():
    df = pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "b": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        }
    )
    r1 = ArtificialMissingnessEvaluator(random_state=11).evaluate(
        df, methods=["iterativo", "mice"], fraction=0.3
    )
    r2 = ArtificialMissingnessEvaluator(random_state=11).evaluate(
        df, methods=["iterativo", "mice"], fraction=0.3
    )
    assert r1.to_dict() == r2.to_dict()


def test_evaluate_metodos_por_defecto_no_temporales(evaluator, df_num):
    report = evaluator.evaluate(df_num, fraction=0.2)
    names = {m.method for m in report.methods}
    assert {"media", "mediana", "moda", "constante", "knn", "iterativo", "mice", "regresion"} <= names
    assert "interpolacion_lineal" not in names
    assert "locf" not in names


def test_evaluate_registra_error_sin_propagarlo():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0]})
    report = ArtificialMissingnessEvaluator(random_state=1).evaluate(
        df, methods=["regresion"], fraction=0.3
    )
    ev = report.methods[0]
    assert ev.error is not None
    assert ev.method == "regresion"


def test_evaluate_instancia_personalizada(evaluator, df_num):
    report = evaluator.evaluate(df_num, methods=[KNNImputation(n_neighbors=2)], fraction=0.2)
    assert [m.method for m in report.methods] == ["knn"]


def test_evaluate_metodo_inexistente(evaluator, df_num):
    with pytest.raises(KeyError):
        evaluator.evaluate(df_num, methods=["metodo_fantasma"], fraction=0.2)


def test_evaluate_n_repeats_acumula():
    df = pd.DataFrame({"a": [float(i) for i in range(20)]})
    report = ArtificialMissingnessEvaluator(random_state=7, n_repeats=3).evaluate(
        df, methods=["media"], fraction=0.3
    )
    ev = report.methods[0]
    assert ev.n_repeats == 3
    assert "mae" in ev.global_metrics
    assert ev.n_induced == 3 * int(0.3 * 20)


def test_evaluate_mar_con_predictor():
    n = 40
    x = np.arange(n, dtype=float)
    y = 5.0 * x
    df = pd.DataFrame({"x": x, "y": y})
    report = ArtificialMissingnessEvaluator(random_state=4).evaluate(
        df, columns=["y"], methods=["regresion"], mechanism="MAR", predictor="x", fraction=0.3
    )
    ev = report.methods[0]
    assert ev.error is None
    assert report.mechanism == "MAR"
    assert report.predictor == "x"


def test_evaluate_metricas_no_negativas_y_finitas(evaluator, df_num):
    report = evaluator.evaluate(df_num, methods=["media", "knn", "mediana"], fraction=0.2)
    for ev in report.methods:
        assert ev.error is None
        assert ev.global_metrics["mae"] >= 0
        assert ev.global_metrics["rmse"] >= 0
        assert np.isfinite(ev.global_metrics["mae"])
        assert np.isfinite(ev.global_metrics["rmse"])


def test_reporte_json_serializable(evaluator, df_num):
    report = evaluator.evaluate(df_num, methods=["media"], fraction=0.2)
    payload = json.dumps(report.to_dict())
    assert isinstance(payload, str)


def test_evaluate_validaciones():
    with pytest.raises(TypeError):
        ArtificialMissingnessEvaluator().evaluate([1, 2, 3])

    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [np.nan, 5.0, 6.0]})
    with pytest.raises(ValueError):
        ArtificialMissingnessEvaluator().evaluate(df, methods=["media"], fraction=0.2)

    df2 = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    with pytest.raises(KeyError):
        ArtificialMissingnessEvaluator().evaluate(df2, columns=["no_existe"], fraction=0.2)
    with pytest.raises(ValueError):
        ArtificialMissingnessEvaluator().evaluate(df2, methods=["media"], fraction=0.0)
    with pytest.raises(ValueError):
        ArtificialMissingnessEvaluator().evaluate(df2, methods=["media"], mechanism="MNAR")
    with pytest.raises(ValueError):
        ArtificialMissingnessEvaluator().evaluate(df2, methods=["media"], mechanism="MAR")
    with pytest.raises(ValueError):
        ArtificialMissingnessEvaluator(n_repeats=0)


def test_evaluate_metodos_duplicados_se_agrupan(evaluator, df_num):
    report = evaluator.evaluate(
        df_num, methods=["media", "media"], fraction=0.2
    )
    assert len(report.methods) == 1
    assert report.methods[0].method == "media"