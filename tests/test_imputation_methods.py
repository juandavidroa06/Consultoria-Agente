"""
Pruebas unitarias para los métodos de imputación (Etapa E3).

Los valores esperados se verifican de forma independiente (cálculo directo con
pandas/scikit-learn), no replicando la implementación.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.impute import KNNImputer
from sklearn.linear_model import LinearRegression

from src.missing_data.methods import (
    ConstantImputation,
    IterativeImputation,
    KNNImputation,
    LinearInterpolationImputation,
    LOCFImputation,
    MICEImputation,
    MeanImputation,
    MedianImputation,
    ModeImputation,
    RegressionImputation,
)


def _df_original_intacto(df, result):
    pd.testing.assert_frame_equal(df, df.copy())


# ---------------------------------------------------------------------------
# Media, mediana, moda, constante
# ---------------------------------------------------------------------------


def test_media_valor_esperado():
    df = pd.DataFrame({"x": [1.0, 2.0, np.nan, 4.0]})
    result = MeanImputation().impute(df)
    assert result["x"].iloc[2] == pytest.approx(7 / 3)
    _df_original_intacto(df, result)


def test_mediana_valor_esperado():
    df = pd.DataFrame({"x": [1.0, 2.0, np.nan, 4.0, 100.0]})
    result = MedianImputation().impute(df)
    assert result["x"].iloc[2] == pytest.approx(3.0)
    _df_original_intacto(df, result)


def test_moda_numerica():
    df = pd.DataFrame({"x": [1, 1, np.nan, 2, 2, 2]})
    result = ModeImputation().impute(df)
    assert result["x"].iloc[2] == 2


def test_moda_categorica():
    df = pd.DataFrame({"z": ["a", "a", None, "b", "a"]})
    result = ModeImputation().impute(df)
    assert result["z"].iloc[2] == "a"
    assert pd.api.types.is_string_dtype(result["z"].dtype)


def test_constante_con_indicador():
    df = pd.DataFrame({"x": [1.0, np.nan, 3.0, np.nan]})
    result = ConstantImputation(constant_value=0.0).impute(df)
    assert result["x"].iloc[1] == 0.0
    assert result["x"].iloc[3] == 0.0
    assert "x_was_missing" in result.columns
    assert result["x_was_missing"].tolist() == [0, 1, 0, 1]


def test_constante_sin_indicador():
    df = pd.DataFrame({"x": [1.0, np.nan, 3.0]})
    result = ConstantImputation(constant_value=-1.0, add_indicator=False).impute(df)
    assert result["x"].iloc[1] == -1.0
    assert "x_was_missing" not in result.columns


def test_constante_categorica():
    df = pd.DataFrame({"z": ["a", None, "b"]})
    result = ConstantImputation(constant_value="desconocido").impute(df)
    assert result["z"].iloc[1] == "desconocido"


# ---------------------------------------------------------------------------
# Redondeo de columnas enteras y preservación de tipos
# ---------------------------------------------------------------------------


def test_media_respeta_politica_redondeo_entero():
    df = pd.DataFrame({"x": pd.Series([1, 2, pd.NA, 4], dtype="Int64")})
    result = MeanImputation().impute(df)
    assert result["x"].isna().sum() == 0
    assert pd.api.types.is_integer_dtype(result["x"].dtype)
    assert result["x"].iloc[2] == 2


def test_redondeo_bancario_media_entera():
    df = pd.DataFrame({"x": pd.Series([1, 2, pd.NA], dtype="Int64")})
    result = MeanImputation().impute(df)
    assert pd.api.types.is_integer_dtype(result["x"].dtype)
    assert result["x"].iloc[2] == 2


def test_columna_float_se_preserva_como_float():
    df = pd.DataFrame({"x": [1.0, 2.0, np.nan, 4.0]})
    result = MeanImputation().impute(df)
    assert pd.api.types.is_float_dtype(result["x"].dtype)


def test_mediana_redondea_enteros():
    df = pd.DataFrame({"x": pd.Series([1, 2, 3, pd.NA, 5], dtype="Int64")})
    result = MedianImputation().impute(df)
    assert pd.api.types.is_integer_dtype(result["x"].dtype)
    assert result["x"].iloc[3] == 2


# ---------------------------------------------------------------------------
# KNN, iterativo, MICE, regresión
# ---------------------------------------------------------------------------


def test_knn_valor_esperado():
    df = pd.DataFrame(
        {"a": [1.0, 2.0, 3.0, 4.0], "b": [10.0, np.nan, 30.0, 40.0]}
    )
    expected = KNNImputer(n_neighbors=2, weights="uniform").fit_transform(
        df[["a", "b"]].astype(float).to_numpy()
    )
    result = KNNImputation(n_neighbors=2).impute(df)
    assert result["b"].iloc[1] == pytest.approx(expected[1, 1])


def test_knn_determinista():
    df = pd.DataFrame(
        {"a": [1.0, 2.0, 3.0, 4.0, 5.0], "b": [10.0, np.nan, 30.0, np.nan, 50.0]}
    )
    r1 = KNNImputation(n_neighbors=2).impute(df)
    r2 = KNNImputation(n_neighbors=2).impute(df)
    pd.testing.assert_frame_equal(r1, r2)


def test_iterativo_determinista():
    df = pd.DataFrame(
        {
            "a": [1.0, 2.0, np.nan, 4.0, 5.0],
            "b": [10.0, np.nan, 30.0, 40.0, 50.0],
            "c": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )
    r1 = IterativeImputation(random_state=42).impute(df)
    r2 = IterativeImputation(random_state=42).impute(df)
    pd.testing.assert_frame_equal(r1, r2)
    assert r1.isna().sum().sum() == 0


def test_iterativo_sin_faltantes():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    result = IterativeImputation(random_state=42).impute(df)
    assert result.isna().sum().sum() == 0


def test_mice_determinista():
    df = pd.DataFrame(
        {
            "a": [1.0, 2.0, np.nan, 4.0, 5.0],
            "b": [10.0, np.nan, 30.0, 40.0, 50.0],
            "c": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )
    r1 = MICEImputation(random_state=7).impute(df)
    r2 = MICEImputation(random_state=7).impute(df)
    pd.testing.assert_frame_equal(r1, r2)
    assert r1.isna().sum().sum() == 0


def test_mice_una_sola_columna_con_faltantes():
    df = pd.DataFrame(
        {"a": [1.0, np.nan, 3.0, 4.0, 5.0], "b": [1.0, 2.0, 3.0, 4.0, 5.0]}
    )
    result = MICEImputation(random_state=1).impute(df)
    assert result.isna().sum().sum() == 0


def test_regresion_valor_esperado():
    df = pd.DataFrame(
        {"x": [1.0, 2.0, 3.0, 4.0, 5.0], "y": [2.0, 4.0, np.nan, 8.0, 10.0]}
    )
    x = df["x"].to_numpy()
    y = df["y"].to_numpy()
    missing = pd.isna(y)
    model = LinearRegression()
    model.fit(x[~missing].reshape(-1, 1), y[~missing])
    expected = model.predict(x[missing].reshape(-1, 1))[0]

    result = RegressionImputation().impute(df)
    assert result["y"].iloc[2] == pytest.approx(expected)


def test_regresion_requiere_predictora_completa():
    df = pd.DataFrame({"a": [1.0, np.nan, 3.0]})
    with pytest.raises(ValueError):
        RegressionImputation().impute(df)


# ---------------------------------------------------------------------------
# Métodos temporales
# ---------------------------------------------------------------------------


def test_interpolacion_lineal():
    df = pd.DataFrame({"t": [1.0, np.nan, np.nan, 4.0, np.nan]})
    result = LinearInterpolationImputation().impute(df)
    assert result["t"].tolist() == [1.0, 2.0, 3.0, 4.0, 4.0]


def test_interpolacion_extremo_inicial():
    df = pd.DataFrame({"t": [np.nan, 2.0, 3.0]})
    result = LinearInterpolationImputation().impute(df)
    assert result["t"].tolist() == [2.0, 2.0, 3.0]


def test_locf():
    df = pd.DataFrame({"t": [1.0, np.nan, np.nan, 4.0, np.nan]})
    result = LOCFImputation().impute(df)
    assert result["t"].tolist() == [1.0, 1.0, 1.0, 4.0, 4.0]


def test_locf_extremo_inicial_bfill():
    df = pd.DataFrame({"t": [np.nan, 2.0, 3.0]})
    result = LOCFImputation().impute(df)
    assert result["t"].tolist() == [2.0, 2.0, 3.0]


# ---------------------------------------------------------------------------
# Casos problemáticos
# ---------------------------------------------------------------------------


def test_columna_completamente_faltante_media():
    df = pd.DataFrame({"x": [np.nan, np.nan], "y": [1.0, 2.0]})
    result = MeanImputation().impute(df)
    assert result["x"].isna().all()
    assert result["y"].iloc[0] == 1.0


def test_columna_completamente_faltante_knn():
    df = pd.DataFrame({"x": [np.nan, np.nan, np.nan], "y": [1.0, 2.0, 3.0]})
    result = KNNImputation(n_neighbors=2).impute(df)
    assert result["x"].isna().all()


def test_columna_completamente_faltante_mice():
    df = pd.DataFrame({"x": [np.nan, np.nan, np.nan], "y": [1.0, 2.0, 3.0]})
    result = MICEImputation(random_state=1).impute(df)
    assert result["x"].isna().all()


def test_columna_completamente_faltante_constante():
    df = pd.DataFrame({"x": [np.nan, np.nan], "y": [1.0, 2.0]})
    result = ConstantImputation(constant_value=0.0).impute(df)
    assert result["x"].tolist() == [0.0, 0.0]
    assert "x_was_missing" in result.columns


def test_fila_completamente_vacia_media():
    df = pd.DataFrame({"a": [np.nan, 2.0], "b": [np.nan, 4.0]})
    result = MeanImputation().impute(df)
    assert result.isna().sum().sum() == 0


def test_fila_completamente_vacia_knn():
    df = pd.DataFrame({"a": [np.nan, 2.0, 3.0], "b": [np.nan, 4.0, 5.0]})
    result = KNNImputation(n_neighbors=2).impute(df)
    assert result.isna().sum().sum() == 0


def test_fila_completamente_vacia_mice_rechaza():
    df = pd.DataFrame({"a": [np.nan, 2.0, 3.0], "b": [np.nan, 4.0, 5.0]})
    with pytest.raises(ValueError):
        MICEImputation(random_state=1).impute(df)


def test_interpolacion_una_observacion():
    df = pd.DataFrame({"t": [1.0, np.nan, np.nan]})
    result = LinearInterpolationImputation().impute(df)
    assert result["t"].iloc[0] == 1.0
    assert result["t"].iloc[1:] .isna().all()


# ---------------------------------------------------------------------------
# Rechazo explícito de entradas no soportadas
# ---------------------------------------------------------------------------


def test_metodo_numerico_rechaza_solo_categorico():
    df = pd.DataFrame({"z": ["a", "b", None]})
    with pytest.raises(ValueError):
        MeanImputation().impute(df)
    with pytest.raises(ValueError):
        KNNImputation().impute(df)


def test_metodo_numerico_acepta_mixto_solo_numerico():
    df = pd.DataFrame({"x": [1.0, np.nan, 3.0], "z": ["a", "b", "c"]})
    result = MeanImputation().impute(df)
    assert result["x"].isna().sum() == 0


def test_impute_rechaza_no_dataframe():
    with pytest.raises(TypeError):
        MeanImputation().impute([1, 2, 3])


# ---------------------------------------------------------------------------
# Determinismo y no modificación del original
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "method_factory",
    [
        lambda: MeanImputation(),
        lambda: MedianImputation(),
        lambda: ModeImputation(),
        lambda: ConstantImputation(),
        lambda: KNNImputation(n_neighbors=2),
        lambda: IterativeImputation(random_state=42),
        lambda: RegressionImputation(),
        lambda: LinearInterpolationImputation(),
        lambda: LOCFImputation(),
    ],
)
def test_no_modifica_original_ni_misma_semilla(method_factory):
    df = pd.DataFrame(
        {
            "a": [1.0, 2.0, np.nan, 4.0, 5.0],
            "b": [10.0, np.nan, 30.0, 40.0, 50.0],
            "c": [1.0, 2.0, 3.0, 4.0, 5.0],
            "z": ["u", "v", "u", "v", "u"],
        }
    )
    original = df.copy()
    result = method_factory().impute(df)
    pd.testing.assert_frame_equal(df, original)


def test_metodos_estocasticos_reproducibles_con_misma_semilla():
    df = pd.DataFrame(
        {
            "a": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0],
            "b": [10.0, np.nan, 30.0, 40.0, 50.0, 60.0],
            "c": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )
    r1 = IterativeImputation(random_state=123).impute(df)
    r2 = IterativeImputation(random_state=123).impute(df)
    pd.testing.assert_frame_equal(r1, r2)

    r3 = MICEImputation(random_state=5).impute(df)
    r4 = MICEImputation(random_state=5).impute(df)
    pd.testing.assert_frame_equal(r3, r4)


def test_mice_resultado_depende_de_la_semilla():
    df = pd.DataFrame(
        {
            "a": [1.0, 2.0, np.nan, 4.0, 5.0],
            "b": [10.0, np.nan, 30.0, 40.0, 50.0],
            "c": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )
    r1 = MICEImputation(random_state=1).impute(df)
    r2 = MICEImputation(random_state=2).impute(df)
    assert not r1.equals(r2)