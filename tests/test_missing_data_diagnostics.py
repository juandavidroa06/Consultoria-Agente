"""
Pruebas unitarias para MissingDataDiagnostics (Etapa E2).

Los valores esperados de las pruebas estadísticas se calculan de forma
independiente con scipy.stats y statsmodels, sin reproducir la implementación.
"""

import json

import numpy as np
import pandas as pd
import pytest
from scipy.stats import MonteCarloMethod, chi2_contingency, fisher_exact, mannwhitneyu
from statsmodels.stats.multitest import multipletests

from src.missing_data import (
    MechanismAssessment,
    MissingDataDiagnostics,
    MissingnessAssociation,
    MissingnessDiagnosticsReport,
)


@pytest.fixture
def diagnostics():
    return MissingDataDiagnostics()


# ---------------------------------------------------------------------------
# Sin datos faltantes
# ---------------------------------------------------------------------------


def test_sin_faltantes(diagnostics):
    df = pd.DataFrame({"num": [1.0, 2.0, 3.0], "cat": ["a", "b", "a"]})
    report = diagnostics.diagnose(df)

    assert isinstance(report, MissingnessDiagnosticsReport)
    assert report.status == "sin_faltantes"
    assert report.variables_with_missing == []
    assert report.co_missing_counts == {}
    assert report.associations == []
    assert report.multiple_comparisons["n_comparisons"] == 0
    assert isinstance(report.mechanism, MechanismAssessment)
    assert report.mechanism.tests_performed == 0


def test_sin_faltantes_evidencia_estadistica(diagnostics):
    df = pd.DataFrame({"a": [1, 2, 3], "b": [1.0, 2.0, 3.0]})
    report = diagnostics.diagnose(df)

    assert "no presenta valores faltantes" in report.mechanism.evidence.lower()


# ---------------------------------------------------------------------------
# Co-ausencia y patrones sistemáticos
# ---------------------------------------------------------------------------


def test_co_missing_counts(diagnostics):
    df = pd.DataFrame(
        {
            "x": [1.0, np.nan, 3.0, np.nan, 5.0],
            "y": [1.0, 2.0, np.nan, 4.0, 5.0],
        }
    )
    report = diagnostics.diagnose(df)

    assert report.co_missing_counts == {"x": 2, "y": 1}


def test_patron_univariado(diagnostics):
    df = pd.DataFrame({"x": [1.0, np.nan, np.nan, 4.0], "y": [1.0, 2.0, 3.0, 4.0]})
    report = diagnostics.diagnose(df)

    assert any("univariado" in p.lower() for p in report.systematic_patterns)


def test_patron_en_bloque(diagnostics):
    df = pd.DataFrame(
        {
            "x": [np.nan, np.nan, np.nan, np.nan, np.nan, 6.0, 7.0],
            "y": [np.nan, np.nan, np.nan, np.nan, np.nan, 6.0, 7.0],
            "z": list(range(7)),
        }
    )
    report = diagnostics.diagnose(df)

    assert any("en bloque" in p.lower() for p in report.systematic_patterns)


def test_patron_monotono(diagnostics):
    df = pd.DataFrame(
        {
            "x": [np.nan, np.nan, np.nan, np.nan, np.nan, 6.0, 7.0],
            "y": [np.nan, np.nan, 3.0, 4.0, 5.0, 6.0, 7.0],
            "z": list(range(7)),
        }
    )
    report = diagnostics.diagnose(df)

    assert any("monótono" in p.lower() for p in report.systematic_patterns)


def test_patron_dominante(diagnostics):
    n = 10
    x = list(range(n))
    x[0:8] = [np.nan] * 8
    df = pd.DataFrame({"x": x, "y": list(range(n))})
    report = diagnostics.diagnose(df)

    assert any("dominante" in p.lower() for p in report.systematic_patterns)


# ---------------------------------------------------------------------------
# Mann-Whitney U (variable numérica)
# ---------------------------------------------------------------------------


def test_mann_whitney_valor_esperado(diagnostics):
    n = 20
    y = np.arange(n, dtype=float)
    x = y.copy()
    x[10:15] = np.nan
    df = pd.DataFrame({"x": x, "y": y})

    missing = np.zeros(n, dtype=bool)
    missing[10:15] = True
    expected_stat, expected_p = mannwhitneyu(
        y[missing], y[~missing], alternative="two-sided"
    )

    report = diagnostics.diagnose(df)
    assert len(report.associations) == 1
    assoc = report.associations[0]
    assert isinstance(assoc, MissingnessAssociation)
    assert assoc.variable == "x"
    assert assoc.associated_with == "y"
    assert assoc.test == "Mann-Whitney U"
    assert assoc.n_missing == 5
    assert assoc.n_observed == 15
    assert assoc.statistic == pytest.approx(expected_stat)
    assert assoc.p_value == pytest.approx(expected_p)


def test_mann_whitney_variable_binaria_numerica(diagnostics):
    n = 12
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=float)
    x = y.copy()
    x[0:5] = np.nan
    df = pd.DataFrame({"x": x, "y": y})

    missing = np.zeros(n, dtype=bool)
    missing[0:5] = True
    expected_stat, expected_p = mannwhitneyu(
        y[missing], y[~missing], alternative="two-sided"
    )

    report = diagnostics.diagnose(df)
    assoc = report.associations[0]
    assert assoc.test == "Mann-Whitney U"
    assert assoc.statistic == pytest.approx(expected_stat)
    assert assoc.p_value == pytest.approx(expected_p)


def test_mann_whitney_grupo_demasiado_pequeno(diagnostics):
    df = pd.DataFrame(
        {
            "x": [np.nan, np.nan, 3.0, 4.0, 5.0, 6.0, 7.0],
            "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        }
    )
    report = diagnostics.diagnose(df)

    assert len(report.associations) == 1
    assoc = report.associations[0]
    assert assoc.test == "Mann-Whitney U"
    assert assoc.p_value is None
    assert assoc.adjusted_p_value is None
    assert "Prueba no realizada" in assoc.conclusion
    assert report.multiple_comparisons["n_comparisons"] == 0


def test_mann_whitney_grupo_pequeno_limita_potencia(diagnostics):
    df = pd.DataFrame(
        {
            "x": [np.nan, np.nan, 3.0, 4.0, 5.0, 6.0, 7.0],
            "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        }
    )
    report = diagnostics.diagnose(df)

    assert any("potencia de la prueba es limitada" in lim for lim in report.mechanism.limitations)


def test_variable_numerica_constante(diagnostics):
    n = 10
    x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    x[0:5] = [np.nan] * 5
    df = pd.DataFrame({"x": x, "y": [5.0] * n})
    report = diagnostics.diagnose(df)

    assert report.associations == []


# ---------------------------------------------------------------------------
# Chi-cuadrado / Fisher exacto (variable categórica)
# ---------------------------------------------------------------------------


def test_chi_cuadrado_valor_esperado(diagnostics):
    n = 20
    z = ["a"] * 10 + ["b"] * 10
    x = [1.0] * n
    x[10:20] = [np.nan] * 10
    df = pd.DataFrame({"x": x, "z": z})

    expected = chi2_contingency([[10, 0], [0, 10]], correction=False)

    report = diagnostics.diagnose(df)
    assert len(report.associations) == 1
    assoc = report.associations[0]
    assert assoc.test == "Chi-cuadrado"
    assert assoc.associated_variable_type == "Binaria"
    assert assoc.statistic == pytest.approx(expected.statistic)
    assert assoc.p_value == pytest.approx(expected.pvalue)


def test_fisher_exacto_valor_esperado(diagnostics):
    z = np.array(["a"] * 6 + ["b"] * 6)
    x = np.array([np.nan, np.nan, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
    df = pd.DataFrame({"x": x, "z": z})

    table = pd.crosstab(pd.isna(x).astype(int), pd.Series(z)).to_numpy()
    assert table.shape == (2, 2)
    assert table.min() < 5
    expected_stat, expected_p = fisher_exact(table, alternative="two-sided")

    report = diagnostics.diagnose(df)
    assert len(report.associations) == 1
    assoc = report.associations[0]
    assert assoc.test == "Fisher exacto"
    assert assoc.statistic == pytest.approx(expected_stat)
    assert assoc.p_value == pytest.approx(expected_p)


def test_chi_cuadrado_monte_carlo_reproducible(diagnostics):
    z = ["a"] * 4 + ["b"] * 4 + ["c"] * 4
    x = [np.nan, np.nan, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
    df = pd.DataFrame({"x": x, "z": z})

    report = diagnostics.diagnose(df)
    assert len(report.associations) == 1
    assoc = report.associations[0]
    assert assoc.test == "Chi-cuadrado (Monte Carlo)"
    assert assoc.p_value is not None

    report2 = diagnostics.diagnose(df)
    assert report2.associations[0].p_value == assoc.p_value


def test_variable_categorica_constante(diagnostics):
    df = pd.DataFrame(
        {
            "x": [np.nan, np.nan, np.nan, 4.0, 5.0],
            "z": ["a"] * 5,
        }
    )
    report = diagnostics.diagnose(df)

    assert report.associations == []


def test_columna_fecha_no_soportada(diagnostics):
    df = pd.DataFrame(
        {
            "x": [np.nan, np.nan, 3.0, 4.0, 5.0],
            "fecha": pd.to_datetime(["2020-01-01"] * 5),
        }
    )
    report = diagnostics.diagnose(df)

    assert report.associations == []
    assert any(
        "tipo de variable no soportado" in skipped["reason"]
        for skipped in report.skipped_comparisons
    )


# ---------------------------------------------------------------------------
# Múltiples comparaciones y FDR
# ---------------------------------------------------------------------------


def test_fdr_valores_ajustados_esperados(diagnostics):
    n = 20
    y1 = np.arange(n, dtype=float)
    y2 = 100.0 - np.arange(n, dtype=float)
    x = y1.copy()
    x[10:15] = np.nan
    df = pd.DataFrame({"x": x, "y1": y1, "y2": y2})

    missing = np.zeros(n, dtype=bool)
    missing[10:15] = True
    p1 = mannwhitneyu(y1[missing], y1[~missing], alternative="two-sided").pvalue
    p2 = mannwhitneyu(y2[missing], y2[~missing], alternative="two-sided").pvalue
    _, expected_adj, _, _ = multipletests([p1, p2], alpha=0.05, method="fdr_bh")

    report = diagnostics.diagnose(df)
    assert report.multiple_comparisons["method"] == "fdr_bh"
    assert report.multiple_comparisons["n_comparisons"] == 2
    assert report.multiple_comparisons["n_significant"] == int(
        sum(1 for p in expected_adj if p < 0.05)
    )

    actual_adj = sorted(a.adjusted_p_value for a in report.associations)
    assert actual_adj == pytest.approx(sorted(float(p) for p in expected_adj))

    for assoc in report.associations:
        assert assoc.adjusted_p_value is not None
        assert assoc.conclusion != ""


def test_sin_comparaciones_por_faltante_total(diagnostics):
    df = pd.DataFrame(
        {
            "x": [np.nan] * 10,
            "y": list(range(10)),
        }
    )
    report = diagnostics.diagnose(df)

    assert report.variables_with_missing == ["x"]
    assert report.associations == []
    assert report.multiple_comparisons["n_comparisons"] == 0
    assert "no se pudo realizar ninguna comparación" in report.mechanism.evidence.lower()


# ---------------------------------------------------------------------------
# Evaluación del mecanismo (MCAR / MAR / MNAR)
# ---------------------------------------------------------------------------


def test_evidencia_contra_mcar(diagnostics):
    n = 20
    z = ["a"] * 10 + ["b"] * 10
    x = [1.0] * n
    x[10:20] = [np.nan] * 10
    df = pd.DataFrame({"x": x, "z": z})

    report = diagnostics.diagnose(df)
    assert len(report.mechanism.significant_comparisons) == 1
    assert report.mechanism.significant_comparisons[0] == "x | z"
    assert "evidencia en contra del supuesto mcar" in report.mechanism.evidence.lower()
    assert any("cuestiona el supuesto MCAR" in a.conclusion for a in report.associations)


def test_sin_evidencia_no_confirma_mcar(diagnostics):
    n = 20
    y = np.tile([1.0, 2.0], n // 2)
    x = y.copy()
    x[10:15] = np.nan
    df = pd.DataFrame({"x": x, "y": y})

    report = diagnostics.diagnose(df)
    assert report.mechanism.significant_comparisons == []
    assert report.multiple_comparisons["n_significant"] == 0
    assert "no se detectaron asociaciones significativas" in report.mechanism.evidence.lower()
    assert "esto no lo confirma" in report.mechanism.evidence.lower()


def test_no_se_infiere_mar_mnar(diagnostics):
    df = pd.DataFrame(
        {
            "x": [np.nan, np.nan, np.nan, 4.0, 5.0, 6.0],
            "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )
    report = diagnostics.diagnose(df)

    assert any("mar no puede demostrarse" in c.lower() for c in report.mechanism.cannot_infer)
    assert any("mnar no puede inferirse" in c.lower() for c in report.mechanism.cannot_infer)


def test_mecanismo_sin_faltantes_no_infiera():
    diag = MissingDataDiagnostics()
    report = diag.diagnose(pd.DataFrame({"a": [1, 2, 3]}))
    assert any("mar" in c.lower() for c in report.mechanism.cannot_infer)
    assert any("mnar" in c.lower() for c in report.mechanism.cannot_infer)


# ---------------------------------------------------------------------------
# Serialización, reproducibilidad y validación
# ---------------------------------------------------------------------------


def test_reporte_json_serializable(diagnostics):
    df = pd.DataFrame(
        {
            "x": [np.nan, 2.0, np.nan, 4.0, 5.0],
            "y": [1.0, 2.0, 3.0, 4.0, 5.0],
            "z": ["a", "b", "a", "b", "a"],
        }
    )
    report = diagnostics.diagnose(df)
    payload = json.dumps(report.to_dict())
    assert isinstance(payload, str)


def test_reproducibilidad(diagnostics):
    n = 30
    y = np.arange(n, dtype=float)
    x = y.copy()
    x[5:12] = np.nan
    df = pd.DataFrame({"x": x, "y": y})

    r1 = diagnostics.diagnose(df)
    r2 = diagnostics.diagnose(df)
    assert r1.to_dict() == r2.to_dict()


def test_no_se_modifica_el_dataframe_original(diagnostics):
    df = pd.DataFrame({"x": [np.nan, 2.0, 3.0], "y": [1.0, 2.0, 3.0]})
    df_copy = df.copy()
    diagnostics.diagnose(df)
    pd.testing.assert_frame_equal(df, df_copy)


def test_diagnose_rechaza_no_dataframe(diagnostics):
    with pytest.raises(TypeError):
        diagnostics.diagnose([1, 2, 3])


def test_parametros_configurables():
    diag = MissingDataDiagnostics(alpha=0.10, min_group_size=3, mc_n_resamples=500)
    assert diag.alpha == 0.10
    assert diag.min_group_size == 3
    assert diag.mc_n_resamples == 500

    df = pd.DataFrame(
        {
            "x": [np.nan, np.nan, np.nan, 4.0],
            "y": [1.0, 2.0, 3.0, 4.0],
        }
    )
    report = diag.diagnose(df)
    assert report.associations[0].p_value is not None


def test_alpha_reflejado_en_metadatos():
    diag = MissingDataDiagnostics(alpha=0.01)
    report = diag.diagnose(pd.DataFrame({"a": [1, 2, 3]}))
    assert report.multiple_comparisons["alpha"] == 0.01


def test_dataframe_vacio_sin_faltantes(diagnostics):
    report = diagnostics.diagnose(pd.DataFrame({"a": pd.Series(dtype="float64")}))
    assert report.status == "sin_faltantes"
    assert report.associations == []
