"""
Pruebas unitarias para las pruebas estadísticas adicionales
(Roadmap §2.2): Kolmogorov-Smirnov, Lilliefors, Bartlett,
Breusch-Pagan, White, Durbin-Watson, Breusch-Godfrey, RESET,
Chi-cuadrado, Tukey HSD y prueba de permutaciones.
"""

import pytest
import numpy as np
from src.analysis.hypothesis import (
    kolmogorov_smirnov_1samp_test,
    kolmogorov_smirnov_2samp_test,
    lilliefors_test,
    bartlett_test,
    breusch_pagan_test,
    white_test,
    durbin_watson_test,
    breusch_godfrey_test,
    reset_test,
    chi_square_test,
    tukey_hsd_test,
    permutation_test,
)


@pytest.fixture
def normal_sample():
    return np.random.default_rng(42).normal(loc=0, scale=1, size=80)


@pytest.fixture
def group_a():
    return np.random.default_rng(42).normal(loc=0, scale=1, size=50)


@pytest.fixture
def group_b():
    return np.random.default_rng(43).normal(loc=1.2, scale=1, size=50)


@pytest.fixture
def reg_data():
    rng = np.random.default_rng(7)
    n = 120
    x = rng.normal(0, 1, n)
    eps = rng.normal(0, 0.5, n)
    y = 2 + 3 * x + eps
    return {"x": x, "y": y, "resid": y - (2 + 3 * x)}


@pytest.fixture
def reg_data_het():
    rng = np.random.default_rng(11)
    n = 200
    x = rng.uniform(0.5, 2.0, n)
    eps = rng.normal(0, 0.3, n)
    y = 2 + 3 * x + x * eps
    return {"x": x, "y": y, "resid": y - (2 + 3 * x)}


@pytest.fixture
def autocorrelated_resid():
    rng = np.random.default_rng(7)
    n = 120
    ar = np.zeros(n)
    for i in range(1, n):
        ar[i] = 0.95 * ar[i - 1] + rng.normal(0, 0.2)
    return ar


def _assert_schema(res):
    for key in (
        "test_name",
        "statistic",
        "p_value",
        "alpha",
        "null_hypothesis",
        "alt_hypothesis",
        "decision",
        "reject_h0",
        "interpretation",
    ):
        assert key in res


def test_kolmogorov_smirnov_1samp_matches_normal(normal_sample):
    res = kolmogorov_smirnov_1samp_test(normal_sample, cdf="norm")
    _assert_schema(res)
    assert res["test_name"].startswith("Kolmogorov-Smirnov (1 Muestra)")
    assert res["reject_h0"] is False


def test_kolmogorov_smirnov_1samp_rejects_non_normal():
    data = np.random.default_rng(5).uniform(0, 1, size=80)
    res = kolmogorov_smirnov_1samp_test(data, cdf="norm")
    assert res["reject_h0"] is True


def test_kolmogorov_smirnov_1samp_validation():
    with pytest.raises(ValueError):
        kolmogorov_smirnov_1samp_test([], cdf="norm")


def test_kolmogorov_smirnov_2samp_different(group_a, group_b):
    res = kolmogorov_smirnov_2samp_test(group_a, group_b)
    _assert_schema(res)
    assert res["test_name"].startswith("Kolmogorov-Smirnov (2 Muestras)")
    assert res["reject_h0"] is True


def test_kolmogorov_smirnov_2samp_same(normal_sample):
    other = np.random.default_rng(99).normal(loc=0, scale=1, size=100)
    res = kolmogorov_smirnov_2samp_test(normal_sample, other)
    assert res["reject_h0"] is False


def test_kolmogorov_smirnov_2samp_validation(normal_sample):
    with pytest.raises(ValueError):
        kolmogorov_smirnov_2samp_test(normal_sample, [])


def test_lilliefors_normal(normal_sample):
    res = lilliefors_test(normal_sample)
    _assert_schema(res)
    assert res["test_name"].startswith("Lilliefors")
    assert res["reject_h0"] is False


def test_lilliefors_exponential():
    data = np.random.default_rng(6).exponential(scale=1.0, size=60)
    res = lilliefors_test(data)
    assert res["reject_h0"] is True


def test_lilliefors_validation():
    with pytest.raises(ValueError):
        lilliefors_test([1.0, 2.0, 3.0])


def test_bartlett_homogeneous():
    rng = np.random.default_rng(8)
    s1 = rng.normal(0, 1, 40)
    s2 = rng.normal(1, 1, 40)
    res = bartlett_test(s1, s2)
    _assert_schema(res)
    assert res["test_name"].startswith("Bartlett")
    assert res["reject_h0"] is False


def test_bartlett_heterogeneous():
    rng = np.random.default_rng(8)
    s1 = rng.normal(0, 1, 40)
    s2 = rng.normal(0, 10, 40)
    res = bartlett_test(s1, s2)
    assert res["reject_h0"] is True


def test_bartlett_validation_single_group():
    with pytest.raises(ValueError):
        bartlett_test([1.0, 2.0, 3.0])


def test_bartlett_validation_constant(normal_sample):
    with pytest.raises(ValueError):
        bartlett_test(normal_sample, [5.0, 5.0, 5.0])


def test_breusch_pagan_homoscedastic(reg_data):
    res = breusch_pagan_test(reg_data["resid"], reg_data["x"].reshape(-1, 1))
    _assert_schema(res)
    assert res["test_name"].startswith("Breusch-Pagan")
    assert res["reject_h0"] is False


def test_breusch_pagan_heteroscedastic(reg_data_het):
    res = breusch_pagan_test(reg_data_het["resid"], reg_data_het["x"].reshape(-1, 1))
    assert res["reject_h0"] is True


def test_breusch_pagan_extra_keys(reg_data_het):
    res = breusch_pagan_test(reg_data_het["resid"], reg_data_het["x"].reshape(-1, 1))
    assert "f_statistic" in res
    assert "f_p_value" in res


def test_breusch_pagan_validation_length(reg_data):
    with pytest.raises(ValueError):
        breusch_pagan_test(reg_data["resid"][:10], reg_data["x"].reshape(-1, 1))


def test_white_homoscedastic(reg_data):
    res = white_test(reg_data["resid"], reg_data["x"].reshape(-1, 1))
    _assert_schema(res)
    assert res["test_name"].startswith("White")
    assert res["reject_h0"] is False


def test_white_heteroscedastic(reg_data_het):
    res = white_test(reg_data_het["resid"], reg_data_het["x"].reshape(-1, 1))
    assert res["reject_h0"] is True
    assert "f_statistic" in res
    assert "f_p_value" in res


def test_white_validation_length(reg_data):
    with pytest.raises(ValueError):
        white_test(reg_data["resid"][:10], reg_data["x"].reshape(-1, 1))


def test_durbin_watson_iid(reg_data):
    res = durbin_watson_test(reg_data["resid"])
    _assert_schema(res)
    assert res["test_name"].startswith("Durbin-Watson")
    assert res["p_value"] is None
    assert 1.5 < res["statistic"] < 2.5
    assert res["reject_h0"] is False


def test_durbin_watson_autocorrelated(autocorrelated_resid):
    res = durbin_watson_test(autocorrelated_resid)
    assert res["statistic"] < 1.5
    assert res["reject_h0"] is True


def test_durbin_watson_validation():
    with pytest.raises(ValueError):
        durbin_watson_test([1.0, 2.0])


def test_breusch_godfrey_iid(reg_data):
    res = breusch_godfrey_test(reg_data["y"], reg_data["x"].reshape(-1, 1))
    _assert_schema(res)
    assert res["test_name"].startswith("Breusch-Godfrey")
    assert res["reject_h0"] is False
    assert "f_statistic" in res
    assert "f_p_value" in res


def test_breusch_godfrey_autocorrelated(autocorrelated_resid):
    rng = np.random.default_rng(7)
    x = rng.normal(0, 1, len(autocorrelated_resid))
    y = 2 + 3 * x + autocorrelated_resid
    res = breusch_godfrey_test(y, x.reshape(-1, 1))
    assert res["reject_h0"] is True


def test_breusch_godfrey_validation_nlags(reg_data):
    with pytest.raises(ValueError):
        breusch_godfrey_test(reg_data["y"], reg_data["x"].reshape(-1, 1), nlags=0)


def test_reset_linear_specification(reg_data):
    res = reset_test(reg_data["y"], reg_data["x"].reshape(-1, 1))
    _assert_schema(res)
    assert res["test_name"].startswith("RESET")
    assert res["reject_h0"] is False


def test_reset_nonlinear_specification():
    rng = np.random.default_rng(9)
    n = 120
    x = rng.normal(0, 1, n)
    y = 2 + 3 * x + 1.5 * x ** 2 + rng.normal(0, 0.3, n)
    res = reset_test(y, x.reshape(-1, 1))
    assert res["reject_h0"] is True


def test_reset_validation_length(reg_data):
    with pytest.raises(ValueError):
        reset_test(reg_data["y"][:10], reg_data["x"].reshape(-1, 1))


def test_chi_square_independence():
    observed = np.array([[40, 40], [40, 40]])
    res = chi_square_test(observed)
    _assert_schema(res)
    assert res["test_name"].startswith("Chi-cuadrado (Independencia)")
    assert res["reject_h0"] is False


def test_chi_square_association():
    observed = np.array([[90, 10], [10, 90]])
    res = chi_square_test(observed)
    assert res["reject_h0"] is True


def test_chi_square_goodness_of_fit_match():
    res = chi_square_test(np.array([10, 10, 10, 10]), expected=np.array([10, 10, 10, 10]))
    assert res["test_name"].startswith("Chi-cuadrado (Bondad de Ajuste)")
    assert res["reject_h0"] is False


def test_chi_square_goodness_of_fit_mismatch():
    res = chi_square_test(np.array([20, 0, 0, 0]), expected=np.array([5, 5, 5, 5]))
    assert res["reject_h0"] is True


def test_chi_square_validation():
    with pytest.raises(ValueError):
        chi_square_test(np.array([10, 10]))


def test_tukey_hsd_significant():
    rng = np.random.default_rng(10)
    g1 = rng.normal(0, 1, 30)
    g2 = rng.normal(2.0, 1, 30)
    g3 = rng.normal(2.1, 1, 30)
    res = tukey_hsd_test(g1, g2, g3)
    _assert_schema(res)
    assert res["test_name"].startswith("Tukey HSD")
    assert res["reject_h0"] is True
    assert len(res["pairwise_comparisons"]) == 3
    assert all(
        set(p.keys())
        >= {
            "group1",
            "group2",
            "mean_difference",
            "p_adjusted",
            "ci_lower",
            "ci_upper",
            "significant",
        }
        for p in res["pairwise_comparisons"]
    )


def test_tukey_hsd_validation(group_a):
    with pytest.raises(ValueError):
        tukey_hsd_test(group_a)


def test_permutation_test_different(group_a, group_b):
    res = permutation_test(group_a, group_b, n_permutations=800, seed=3)
    _assert_schema(res)
    assert res["test_name"].startswith("Permutaciones")
    assert res["reject_h0"] is True
    assert 0.0 < res["p_value"] <= 1.0


def test_permutation_test_same():
    rng = np.random.default_rng(12)
    s1 = rng.normal(0, 1, 40)
    s2 = rng.normal(0, 1, 40)
    res = permutation_test(s1, s2, n_permutations=800, seed=3)
    assert res["reject_h0"] is False


def test_permutation_test_reproducible(group_a, group_b):
    r1 = permutation_test(group_a, group_b, n_permutations=800, seed=3)
    r2 = permutation_test(group_a, group_b, n_permutations=800, seed=3)
    assert r1["p_value"] == r2["p_value"]
    assert r1["statistic"] == r2["statistic"]


def test_permutation_test_validation(group_a):
    with pytest.raises(ValueError):
        permutation_test(group_a, group_a, n_permutations=0)


def test_breusch_pagan_exog_1d(reg_data):
    res_1d = breusch_pagan_test(reg_data["resid"], reg_data["x"])
    res_2d = breusch_pagan_test(reg_data["resid"], reg_data["x"].reshape(-1, 1))
    assert res_1d["statistic"] == res_2d["statistic"]
    assert res_1d["p_value"] == res_2d["p_value"]


def test_white_exog_1d(reg_data):
    res_1d = white_test(reg_data["resid"], reg_data["x"])
    res_2d = white_test(reg_data["resid"], reg_data["x"].reshape(-1, 1))
    assert res_1d["statistic"] == res_2d["statistic"]
    assert res_1d["p_value"] == res_2d["p_value"]


def test_breusch_godfrey_exog_1d(reg_data):
    res_1d = breusch_godfrey_test(reg_data["y"], reg_data["x"])
    res_2d = breusch_godfrey_test(reg_data["y"], reg_data["x"].reshape(-1, 1))
    assert res_1d["statistic"] == res_2d["statistic"]
    assert res_1d["p_value"] == res_2d["p_value"]


def test_reset_exog_1d(reg_data):
    res_1d = reset_test(reg_data["y"], reg_data["x"])
    res_2d = reset_test(reg_data["y"], reg_data["x"].reshape(-1, 1))
    assert res_1d["statistic"] == res_2d["statistic"]
    assert res_1d["p_value"] == res_2d["p_value"]


def test_breusch_pagan_nan_rows_cleaned(reg_data):
    resid = reg_data["resid"].copy()
    x = reg_data["x"].copy()
    resid[5] = np.nan
    x[20] = np.nan
    res = breusch_pagan_test(resid, x)
    mask = ~(np.isnan(resid) | np.isnan(x))
    ref = breusch_pagan_test(resid[mask], x[mask])
    assert res["statistic"] == ref["statistic"]
    assert res["p_value"] == ref["p_value"]


def test_white_nan_rows_cleaned(reg_data):
    resid = reg_data["resid"].copy()
    x = reg_data["x"].copy()
    resid[5] = np.nan
    x[20] = np.nan
    res = white_test(resid, x)
    mask = ~(np.isnan(resid) | np.isnan(x))
    ref = white_test(resid[mask], x[mask])
    assert res["statistic"] == ref["statistic"]
    assert res["p_value"] == ref["p_value"]


def test_breusch_pagan_insufficient_observations(reg_data):
    with pytest.raises(ValueError):
        breusch_pagan_test(reg_data["resid"][:2], reg_data["x"][:2])


def test_white_insufficient_observations(reg_data):
    with pytest.raises(ValueError):
        white_test(reg_data["resid"][:3], reg_data["x"][:3])


def test_chi_square_zero_marginal():
    with pytest.raises(ValueError):
        chi_square_test(np.array([[0, 0], [10, 10]]))


def test_chi_square_negative_cell():
    with pytest.raises(ValueError):
        chi_square_test(np.array([[1, -1], [1, 1]]))


def test_chi_square_nonpositive_expected():
    with pytest.raises(ValueError):
        chi_square_test(np.array([10, 10]), expected=np.array([10, 0]))
