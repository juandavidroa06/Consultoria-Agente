"""
Pruebas unitarias para la validación post-imputación (Etapa E6).
"""

import json

import numpy as np
import pandas as pd
import pytest

from src.missing_data import (
    ImputationValidationReport,
    ImputationValidator,
    MeanImputation,
    ValidationCheck,
    induce_missing,
)


def _base_df(n=60, seed=7):
    rng = np.random.default_rng(seed)
    x = rng.normal(50, 5, n)
    y = 2.0 * x + rng.normal(0, 5, n)
    return pd.DataFrame({"x": x, "y": y})


def _validator(**kwargs):
    return ImputationValidator(**kwargs)


def _check_by_name(report, name):
    return next(c for c in report.checks if c.name == name)


# ---------------------------------------------------------------------------
# Faltantes residuales
# ---------------------------------------------------------------------------


def test_detecta_faltantes_residuales():
    original = _base_df()
    imputed = original.copy()
    imputed.loc[0, "x"] = np.nan
    report = _validator().validate(original, imputed)
    check = _check_by_name(report, "faltantes_residuales")
    assert check.status == "error"
    assert check.details["columnas"]["x"] == 1
    assert report.verdict == "Revisar"


def test_faltantes_residuales_por_columna_ausente():
    original = _base_df()
    imputed = original.drop(columns=["y"]).copy()
    report = _validator().validate(original, imputed)
    check = _check_by_name(report, "faltantes_residuales")
    assert check.status == "error"
    assert "y" in check.details["columnas"]


# ---------------------------------------------------------------------------
# Cambio de dtype
# ---------------------------------------------------------------------------


def test_detecta_cambio_de_dtype():
    original = pd.DataFrame({"a": [1, 2, 3, 4], "b": [1.0, 2.0, 3.0, 4.0]})
    imputed = original.astype({"a": "float64"}).copy()
    report = _validator().validate(original, imputed)
    check = _check_by_name(report, "dtype_preservacion")
    assert check.status == "warn"
    assert check.details["columnas"]["a"]["antes"] == "int64"
    assert check.details["columnas"]["a"]["despues"] == "float64"


# ---------------------------------------------------------------------------
# Valores imposibles
# ---------------------------------------------------------------------------


def test_detecta_valores_imposibles_numericos():
    original = pd.DataFrame({"v": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, np.nan, np.nan]})
    imputed = original.copy()
    imputed.loc[6, "v"] = 100.0
    imputed.loc[7, "v"] = -3.0
    report = _validator().validate(original, imputed)
    check = _check_by_name(report, "valores_imposibles")
    assert check.status == "error"
    assert check.details["columnas"]["v"]["n_violaciones"] == 2
    assert report.verdict == "Revisar"


def test_detecta_valores_imposibles_categoricos():
    original = pd.DataFrame({"c": ["a", "b", "a", "b", None, None]})
    imputed = original.copy()
    imputed.loc[4, "c"] = "zz"
    imputed.loc[5, "c"] = "a"
    report = _validator().validate(original, imputed)
    check = _check_by_name(report, "valores_imposibles")
    assert check.status == "error"
    assert check.details["columnas"]["c"]["n_violaciones"] == 1


# ---------------------------------------------------------------------------
# Cambios de distribución (KS + descriptivas)
# ---------------------------------------------------------------------------


def test_detecta_cambio_importante_de_distribucion():
    rng = np.random.default_rng(1)
    n = 60
    observed = rng.normal(0, 1, n)
    observed[0:10] = np.nan
    original = pd.DataFrame({"v": observed})
    imputed = original.copy()
    imputed.loc[0:9, "v"] = 50.0 + rng.normal(0, 1, 10)
    report = _validator().validate(original, imputed)
    check = _check_by_name(report, "distribucion_ks")
    assert check.status == "warn"
    entry = check.details["columnas"]["v"]
    assert entry["ks_pvalue"] < 0.05


def test_distribucion_sin_cambios_ok():
    original = _base_df()
    imputed = MeanImputation().impute(original)
    report = _validator().validate(original, imputed)
    check = _check_by_name(report, "distribucion_ks")
    assert check.status in ("ok", "warn")


# ---------------------------------------------------------------------------
# Cambios en correlaciones
# ---------------------------------------------------------------------------


def test_detecta_cambio_importante_en_correlaciones():
    n = 50
    x = np.arange(n, dtype=float)
    y = 2.0 * x + 10.0
    original = pd.DataFrame({"x": x, "y": y})
    original.loc[0:20, "x"] = np.nan
    imputed = original.copy()
    imputed.loc[0:20, "x"] = 50.0
    report = _validator().validate(original, imputed)
    check = _check_by_name(report, "correlaciones")
    assert check.status == "warn"
    assert len(check.details["pares_cambios"]) >= 1


# ---------------------------------------------------------------------------
# Proporción de celdas imputadas
# ---------------------------------------------------------------------------


def test_proporcion_imputada_alta_adviere():
    original = pd.DataFrame({"v": np.r_[np.array([np.nan] * 40), np.arange(10.0)]})
    imputed = original.copy()
    imputed["v"] = imputed["v"].fillna(5.0)
    report = _validator().validate(original, imputed)
    check = _check_by_name(report, "proporcion_imputada")
    assert check.status == "warn"
    assert check.details["n_imputadas"] == 40


# ---------------------------------------------------------------------------
# Veredictos
# ---------------------------------------------------------------------------


def test_resultado_correcto_aceptable():
    original = _base_df()
    original = induce_missing(original, columns=["x"], fraction=0.2, random_state=3)
    imputed = MeanImputation().impute(original)
    report = _validator().validate(original, imputed)
    assert report.verdict == "Aceptable"
    assert _check_by_name(report, "faltantes_residuales").status == "ok"
    assert _check_by_name(report, "valores_imposibles").status == "ok"
    assert report.n_imputed_cells > 0


def test_resultado_problematico_revisar():
    original = _base_df()
    imputed = original.copy()
    imputed.loc[0, "y"] = np.nan
    imputed.loc[1, "y"] = 1e6
    report = _validator().validate(original, imputed)
    assert report.verdict == "Revisar"
    assert report.warnings


def test_verdict_revisar_si_error_aunque_no_haya_warnings():
    original = _base_df()
    imputed = original.copy()
    imputed.loc[0, "x"] = np.nan
    report = _validator().validate(original, imputed)
    assert report.verdict == "Revisar"


# ---------------------------------------------------------------------------
# Estructura del reporte
# ---------------------------------------------------------------------------


def test_checks_completos():
    original = _base_df()
    imputed = MeanImputation().impute(original)
    report = _validator().validate(original, imputed)
    names = [c.name for c in report.checks]
    assert names == [
        "faltantes_residuales",
        "dtype_preservacion",
        "valores_imposibles",
        "distribucion_ks",
        "correlaciones",
        "proporcion_imputada",
        "comparacion_imputados_vs_observados",
    ]
    assert all(isinstance(c, ValidationCheck) for c in report.checks)


def test_serializacion_json():
    original = _base_df()
    imputed = MeanImputation().impute(original)
    report = _validator().validate(original, imputed)
    payload = json.dumps(report.to_dict())
    assert isinstance(payload, str)
    loaded = json.loads(payload)
    assert loaded["verdict"] == report.verdict


def test_reproducibilidad():
    original = _base_df()
    imputed = MeanImputation().impute(original)
    r1 = _validator().validate(original, imputed)
    r2 = _validator().validate(original, imputed)
    assert r1.to_dict() == r2.to_dict()


def test_validacion_entradas():
    df = _base_df()
    with pytest.raises(TypeError):
        _validator().validate("nope", df)
    with pytest.raises(TypeError):
        _validator().validate(df, "nope")
    with pytest.raises(ValueError):
        _validator().validate(df, df.iloc[:5])
    with pytest.raises(ValueError):
        _validator(ks_alpha=1.5)
    with pytest.raises(ValueError):
        _validator(corr_threshold=-0.1)


def test_reproduccion_documenta_umbrales():
    original = _base_df()
    imputed = original.copy()
    report = _validator().validate(original, imputed)
    assert report.reproduction["thresholds"]["ks_alpha"] == 0.05
    assert report.reproduction["module"] == "src.missing_data.validation"


def test_sin_imputacion_reporte_trivial():
    original = _base_df()
    report = _validator().validate(original, original.copy())
    assert report.n_imputed_cells == 0
    assert report.verdict == "Aceptable"