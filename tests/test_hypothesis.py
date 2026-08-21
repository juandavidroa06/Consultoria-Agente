"""
Pruebas unitarias para el módulo de pruebas de hipótesis.
"""

import pytest
import numpy as np
from src.analysis.hypothesis import (
    shapiro_wilk_test,
    levene_test,
    t_test_1samp,
    t_test_ind,
    t_test_rel,
    wilcoxon_signed_rank,
    mann_whitney_test,
    anova_one_way,
    welch_anova,
    kruskal_wallis_test,
)


@pytest.fixture
def normal_data():
    np.random.seed(42)
    return np.random.normal(loc=10, scale=2, size=50)


@pytest.fixture
def group_samples():
    np.random.seed(42)
    g1 = np.random.normal(loc=10, scale=2, size=30)
    g2 = np.random.normal(loc=12, scale=2, size=30)
    g3 = np.random.normal(loc=15, scale=2.5, size=30)
    return g1, g2, g3


def test_shapiro_wilk_test(normal_data):
    res = shapiro_wilk_test(normal_data)
    assert res["test_name"].startswith("Shapiro-Wilk")
    assert "statistic" in res
    assert "p_value" in res
    assert res["reject_h0"] is False  # Datos normales, no rechaza H0


def test_levene_test(group_samples):
    g1, g2, _ = group_samples
    res = levene_test(g1, g2)
    assert res["test_name"].startswith("Levene")
    assert "p_value" in res


def test_t_test_1samp(normal_data):
    res = t_test_1samp(normal_data, popmean=10.0)
    assert res["test_name"].startswith("t de Student (1 Muestra)")
    assert res["reject_h0"] is False


def test_t_test_ind(group_samples):
    g1, g2, _ = group_samples
    res = t_test_ind(g1, g2)
    assert res["test_name"].startswith("t de Student")
    assert res["reject_h0"] is True  # Medias 10 y 12 deben diferir


def test_t_test_rel(group_samples):
    g1, g2, _ = group_samples
    res = t_test_rel(g1, g2)
    assert res["test_name"].startswith("t de Student (Muestras Pareadas)")
    assert res["reject_h0"] is True


def test_wilcoxon_signed_rank(group_samples):
    g1, g2, _ = group_samples
    res = wilcoxon_signed_rank(g1, g2)
    assert res["test_name"].startswith("Wilcoxon Pareado")
    assert res["reject_h0"] is True


def test_mann_whitney_test(group_samples):
    g1, g2, _ = group_samples
    res = mann_whitney_test(g1, g2)
    assert res["test_name"].startswith("Mann-Whitney")
    assert res["reject_h0"] is True


def test_anova_one_way(group_samples):
    g1, g2, g3 = group_samples
    res = anova_one_way(g1, g2, g3)
    assert res["test_name"].startswith("ANOVA")
    assert res["reject_h0"] is True


def test_welch_anova(group_samples):
    g1, g2, g3 = group_samples
    res = welch_anova(g1, g2, g3)
    assert res["test_name"].startswith("ANOVA de Welch")
    assert res["reject_h0"] is True


def test_kruskal_wallis_test(group_samples):
    g1, g2, g3 = group_samples
    res = kruskal_wallis_test(g1, g2, g3)
    assert res["test_name"].startswith("Kruskal-Wallis")
    assert res["reject_h0"] is True
