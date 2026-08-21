"""
Tests de regresión E9-D: integración en DatasetStatisticalAnalyzer de las
pruebas estadísticas adicionales (Lilliefors, Bartlett, Tukey HSD,
chi-cuadrado de independencia y permutaciones), según el mapa aprobado.
"""

import pytest
import numpy as np
import pandas as pd

from src.analysis.dataset_analyzer import DatasetStatisticalAnalyzer


@pytest.fixture
def normal_two_groups():
    rng = np.random.default_rng(11)
    g1 = rng.normal(0, 1, 40)
    g2 = rng.normal(1, 1, 40)
    return pd.DataFrame({"t": np.concatenate([g1, g2]), "g": ["a"] * 40 + ["b"] * 40})


@pytest.fixture
def non_normal_two_groups():
    rng = np.random.default_rng(13)
    e1 = rng.exponential(1, 40)
    e2 = rng.exponential(1, 40)
    return pd.DataFrame({"t": np.concatenate([e1, e2]), "g": ["a"] * 40 + ["b"] * 40})


@pytest.fixture
def three_groups_significant():
    rng = np.random.default_rng(7)
    n = 30
    return pd.DataFrame({
        "t": np.concatenate([rng.normal(0, 1, n), rng.normal(2, 1, n), rng.normal(2.2, 1, n)]),
        "g": ["a"] * n + ["b"] * n + ["c"] * n,
    })


@pytest.fixture
def three_groups_not_significant():
    rng = np.random.default_rng(8)
    n = 30
    return pd.DataFrame({
        "t": np.concatenate([rng.normal(0, 2, n), rng.normal(0.1, 2, n), rng.normal(0.2, 2, n)]),
        "g": ["a"] * n + ["b"] * n + ["c"] * n,
    })


def _normality_diagnostic(result):
    return next(
        d for d in result["diagnostics"]
        if d.get("type") == "Diagnóstico de Normalidad Contextual"
    )


# ---------------------------------------------------------------------------
# Lilliefors: diagnóstico complementario de normalidad
# ---------------------------------------------------------------------------


def test_lilliefors_presente_con_n_suficiente():
    rng = np.random.default_rng(5)
    df = pd.DataFrame({"t": rng.normal(0, 1, 30)})
    result = DatasetStatisticalAnalyzer(df).analyze(target_col="t")

    norm_diag = _normality_diagnostic(result)
    lr = norm_diag["lilliefors_result"]
    for field in ("test_name", "statistic", "p_value", "reject_h0", "decision"):
        assert field in lr
    assert lr["test_name"].startswith("Lilliefors")
    assert isinstance(lr["reject_h0"], bool)
    assert lr["decision"] == ("Rechazar H0" if lr["reject_h0"] else "No rechazar H0")


def test_lilliefors_ausente_con_n_menor_a_4():
    df = pd.DataFrame({"t": [5.0, 6.0, 7.0]})
    result = DatasetStatisticalAnalyzer(df).analyze(target_col="t")

    norm_diag = _normality_diagnostic(result)
    assert norm_diag["sample_size"] == 3
    assert "lilliefors_result" not in norm_diag


# ---------------------------------------------------------------------------
# Bartlett: homogeneidad de varianzas bajo normalidad contextual
# ---------------------------------------------------------------------------


def test_bartlett_se_ejecuta_bajo_normalidad(normal_two_groups):
    result = DatasetStatisticalAnalyzer(normal_two_groups).analyze(
        target_col="t", group_col="g"
    )
    executed = result["executed_test_results"]
    assert "Bartlett (Homogeneidad de Varianzas)" in executed
    assert executed["Bartlett (Homogeneidad de Varianzas)"]["test_name"].startswith("Bartlett")
    assert any(
        d.get("type") == "Diagnóstico de Homocedasticidad (Bartlett)"
        for d in result["diagnostics"]
    )


def test_bartlett_no_se_ejecuta_sin_normalidad(non_normal_two_groups):
    result = DatasetStatisticalAnalyzer(non_normal_two_groups).analyze(
        target_col="t", group_col="g"
    )
    assert "Bartlett (Homogeneidad de Varianzas)" not in result["executed_test_results"]


# ---------------------------------------------------------------------------
# Tukey HSD: post hoc únicamente tras ANOVA de un factor significativa
# ---------------------------------------------------------------------------


def test_tukey_post_hoc_cuando_anova_significativa(three_groups_significant):
    result = DatasetStatisticalAnalyzer(three_groups_significant).analyze(
        target_col="t", group_col="g"
    )
    executed = result["executed_test_results"]
    assert "Tukey HSD (Comparaciones Múltiples)" in executed
    tukey = executed["Tukey HSD (Comparaciones Múltiples)"]
    assert "pairwise_comparisons" in tukey
    assert any(d.get("type") == "Post hoc (Tukey HSD)" for d in result["diagnostics"])


def test_tukey_no_se_ejecuta_cuando_anova_no_significativa(three_groups_not_significant):
    result = DatasetStatisticalAnalyzer(three_groups_not_significant).analyze(
        target_col="t", group_col="g"
    )
    assert "Tukey HSD (Comparaciones Múltiples)" not in result["executed_test_results"]


# ---------------------------------------------------------------------------
# Permutaciones: verificación complementaria en comparación no paramétrica
# ---------------------------------------------------------------------------


def test_permutaciones_en_rama_no_parametrica(non_normal_two_groups):
    result = DatasetStatisticalAnalyzer(non_normal_two_groups).analyze(
        target_col="t", group_col="g"
    )
    executed = result["executed_test_results"]
    assert "Permutaciones (Diferencia de Medias)" in executed
    perm = executed["Permutaciones (Diferencia de Medias)"]
    assert 0.0 <= perm["p_value"] <= 1.0
    assert isinstance(perm["reject_h0"], bool)


def test_permutaciones_no_se_ejecuta_en_rama_parametrica(normal_two_groups):
    result = DatasetStatisticalAnalyzer(normal_two_groups).analyze(
        target_col="t", group_col="g"
    )
    assert "Permutaciones (Diferencia de Medias)" not in result["executed_test_results"]


def test_permutaciones_reproducible(non_normal_two_groups):
    r1 = DatasetStatisticalAnalyzer(non_normal_two_groups).analyze(
        target_col="t", group_col="g"
    )
    r2 = DatasetStatisticalAnalyzer(non_normal_two_groups).analyze(
        target_col="t", group_col="g"
    )
    p1 = r1["executed_test_results"]["Permutaciones (Diferencia de Medias)"]["p_value"]
    p2 = r2["executed_test_results"]["Permutaciones (Diferencia de Medias)"]["p_value"]
    assert p1 == p2


# ---------------------------------------------------------------------------
# Chi-cuadrado de independencia (modo exploratorio)
# ---------------------------------------------------------------------------


def test_chi_cuadrado_ejecuta_asociacion():
    c1 = ["A"] * 30 + ["B"] * 30
    c2 = ["X"] * 28 + ["Y"] * 2 + ["X"] * 2 + ["Y"] * 28
    df = pd.DataFrame({"c1": c1, "c2": c2})
    result = DatasetStatisticalAnalyzer(df).analyze()

    recs = [r for r in result["recommendations"] if "Chi-cuadrado" in r["recommended_test"]]
    assert recs, "Se esperaba una recomendación chi-cuadrado de independencia."
    assert recs[0]["variables"] == ["c1", "c2"]

    key = "Chi-cuadrado (Independencia)"
    assert key in result["executed_test_results"]
    chi = result["executed_test_results"][key]
    assert chi["test_name"].startswith("Chi-cuadrado")
    assert 0.0 <= chi["p_value"] <= 1.0
    assert isinstance(chi["reject_h0"], bool)
    assert chi["decision"] == ("Rechazar H0" if chi["reject_h0"] else "No rechazar H0")


def test_chi_cuadrado_una_sola_categorica_sin_recomendacion():
    df = pd.DataFrame({"c1": ["A", "B"] * 30})
    result = DatasetStatisticalAnalyzer(df).analyze()
    recs = [r for r in result["recommendations"] if "Chi-cuadrado" in r["recommended_test"]]
    assert recs == []
    assert "Chi-cuadrado (Independencia)" not in result["executed_test_results"]


def test_chi_cuadrado_marginal_nula_se_omite(monkeypatch):
    # Rama defensiva: con pd.crosstab estándar las marginales nulas no se
    # producen (las categorías sin co-ocurrencias válidas se eliminan). Se
    # inyecta una tabla con una fila de suma cero para verificar la omisión.
    df = pd.DataFrame({"c1": ["A", "B"] * 30, "c2": ["X", "Y"] * 30})
    analyzer = DatasetStatisticalAnalyzer(df)

    def fake_crosstab(index, columns, **kwargs):
        return pd.DataFrame({"X": [0, 30], "Y": [0, 30]}, index=["A", "B"])

    monkeypatch.setattr(pd, "crosstab", fake_crosstab)
    out = analyzer._analyze_categorical_association(["c1", "c2"], alpha=0.05)

    assert out["recommendations"] == []
    assert out["executed_tests"] == {}
    omitted = [d for d in out["diagnostics"] if d.get("type") == "Asociación categórica omitida"]
    assert omitted and "suma total cero" in omitted[0]["summary"]
