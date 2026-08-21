"""
Pruebas unitarias para MissingDataDetector (Etapa E1).
"""

import json

import numpy as np
import pandas as pd
import pytest

from src.missing_data.detection import (
    MissingDataDetector,
    MissingReport,
    convert_placeholders_to_na,
)


@pytest.fixture
def detector():
    return MissingDataDetector()


def test_dataframe_sin_faltantes(detector):
    df = pd.DataFrame({"num": [1.0, 2.0, 3.0, 4.0], "cat": ["a", "b", "a", "b"]})
    report = detector.detect(df)

    assert isinstance(report, MissingReport)
    assert report.status == "sin_faltantes"
    assert report.total_missing_values == 0
    assert report.complete_cases == 4
    assert report.incomplete_cases == 0
    assert report.total_placeholders == 0
    assert report.variables_without_missing == ["num", "cat"]
    assert report.variables_with_missing == []


def test_deteccion_nan(detector):
    df = pd.DataFrame({"x": [1.0, np.nan, 3.0, np.nan]})
    report = detector.detect(df)

    assert report.total_missing_values == 2
    assert report.by_variable["x"].missing_count == 2
    assert report.by_variable["x"].missing_percentage == 50.0
    assert report.status == "con_faltantes"


def test_deteccion_none(detector):
    df = pd.DataFrame({"x": [1, None, 3, None]})
    report = detector.detect(df)

    assert report.total_missing_values == 2
    assert report.by_variable["x"].missing_count == 2


def test_deteccion_pd_na(detector):
    df = pd.DataFrame({"x": pd.Series([1, pd.NA, 3, pd.NA], dtype="Int64")})
    report = detector.detect(df)

    assert report.total_missing_values == 2
    assert report.by_variable["x"].missing_count == 2
    assert report.by_variable["x"].dtype == "Int64"


def test_mezcla_de_tipos(detector):
    df = pd.DataFrame({
        "num": [1.0, np.nan, 3.0, 4.0],
        "cat": ["a", "b", None, "d"],
        "bool": [True, False, True, pd.NA],
        "fecha": pd.to_datetime(["2024-01-01", "2024-01-02", None, "2024-01-04"]),
    })
    report = detector.detect(df)

    assert report.total_missing_values == 4
    assert report.overall_missing_percentage == 25.0
    assert set(report.variables_with_missing) == {"num", "cat", "bool", "fecha"}
    assert report.by_variable["fecha"].dtype.startswith("datetime64")


def test_placeholders_detectados_y_separados(detector):
    df = pd.DataFrame({
        "texto": ["N/A", "NA", "A", "-", "", "unknown", "B", "C"],
        "num": [1, 2, 3, 4, 5, 6, 7, 8],
    })
    report = detector.detect(df)

    assert report.total_missing_values == 0
    assert report.by_variable["texto"].missing_count == 0
    assert report.by_variable["texto"].placeholder_count == 5
    assert report.total_placeholders == 5
    assert report.has_placeholders is True
    assert report.status == "sin_faltantes"


def test_placeholders_desactivados_con_tokens_vacios():
    detector_sin_tokens = MissingDataDetector(placeholder_tokens=set())
    df = pd.DataFrame({"texto": ["N/A", "NA", "A", "-", "", "B"]})
    report = detector_sin_tokens.detect(df)

    assert report.total_placeholders == 0
    assert report.by_variable["texto"].placeholder_count == 0


def test_placeholders_desactivados_con_flag():
    detector_sin_flag = MissingDataDetector(detect_placeholders=False)
    df = pd.DataFrame({"texto": ["N/A", "A", "-", "B"]})
    report = detector_sin_flag.detect(df)

    assert report.total_placeholders == 0
    assert report.by_variable["texto"].placeholder_count == 0


def test_placeholder_no_es_faltante_real(detector):
    df = pd.DataFrame({"texto": ["N/A", None, "A", "B"]})
    report = detector.detect(df)

    assert report.by_variable["texto"].missing_count == 1
    assert report.by_variable["texto"].placeholder_count == 1


def test_valor_cero_no_se_convierte_en_faltante(detector):
    df = pd.DataFrame({"num": [0, 0.0, 1, 2]})
    report = detector.detect(df)

    assert report.total_missing_values == 0
    assert report.by_variable["num"].placeholder_count == 0
    assert report.status == "sin_faltantes"


def test_valor_no_no_se_convierte_en_faltante(detector):
    df = pd.DataFrame({"resp": ["no", "si", "no", "si"]})
    report = detector.detect(df)

    assert report.total_missing_values == 0
    assert report.total_placeholders == 0
    assert report.status == "sin_faltantes"


def test_fila_completamente_vacia(detector):
    df = pd.DataFrame({
        "a": [1.0, np.nan],
        "b": ["x", np.nan],
        "c": [True, np.nan],
    })
    report = detector.detect(df)

    assert report.total_missing_values == 3
    assert report.rows_completely_empty == 1
    assert report.incomplete_cases == 1
    assert report.complete_cases == 1


def test_columna_completamente_vacia(detector):
    df = pd.DataFrame({"a": [1.0, 2.0], "b": [np.nan, np.nan]})
    report = detector.detect(df)

    assert report.by_variable["b"].missing_count == 2
    assert report.by_variable["b"].missing_percentage == 100.0
    assert report.by_variable["b"].missing_grade == "Muy alta"
    assert report.variables_with_missing == ["b"]
    assert report.variables_without_missing == ["a"]


def test_dataframe_vacio_sin_columnas(detector):
    df = pd.DataFrame()
    report = detector.detect(df)

    assert report.total_observations == 0
    assert report.total_variables == 0
    assert report.total_missing_values == 0
    assert report.overall_missing_percentage == 0.0
    assert report.complete_cases == 0
    assert report.rows_completely_empty == 0
    assert report.status == "sin_faltantes"


def test_dataframe_vacio_con_columnas(detector):
    df = pd.DataFrame(columns=["a", "b"])
    report = detector.detect(df)

    assert report.total_observations == 0
    assert report.total_variables == 2
    assert report.total_missing_values == 0
    assert report.variables_without_missing == ["a", "b"]
    assert report.overall_missing_percentage == 0.0


def test_calculo_correcto_de_porcentajes(detector):
    df = pd.DataFrame({
        "x": [1.0, np.nan, 3.0, np.nan, 5.0],
        "y": [1, 2, 3, 4, 5],
    })
    report = detector.detect(df)

    assert report.by_variable["x"].missing_percentage == 40.0
    assert report.by_variable["y"].missing_percentage == 0.0
    assert report.overall_missing_percentage == 20.0


def test_casos_completos(detector):
    df = pd.DataFrame({
        "x": [1.0, np.nan, 3.0, 4.0, 5.0],
        "y": [np.nan, 2, 3, 4, 5],
    })
    report = detector.detect(df)

    assert report.complete_cases == 3
    assert report.complete_cases_percentage == 60.0
    assert report.incomplete_cases == 2


def test_distribucion_de_faltantes_por_observacion(detector):
    df = pd.DataFrame({
        "a": [1.0, np.nan, 1.0, np.nan],
        "b": [2.0, 2.0, np.nan, np.nan],
    })
    report = detector.detect(df)

    assert report.observations_missing_distribution == {0: 1, 1: 2, 2: 1}


def test_reproducibilidad_del_reporte(detector):
    df = pd.DataFrame({
        "num": [1.0, np.nan, 3.0, None],
        "cat": ["N/A", "a", "-", "b"],
    })
    report1 = detector.detect(df)
    report2 = detector.detect(df)

    assert report1.to_dict() == report2.to_dict()


def test_clasificacion_de_tipos_de_variable(detector):
    df = pd.DataFrame({
        "cont": [1.5, 2.5, 3.5, 4.5],
        "ent": [1, 2, 3, 4],
        "cat": ["a", "b", "c", "a"],
    })
    report = detector.detect(df)

    assert report.by_variable["cont"].variable_type == "Cuantitativa continua"
    assert report.by_variable["cont"].dtype == "float64"
    assert report.by_variable["ent"].variable_type == "Cuantitativa discreta"
    assert report.by_variable["ent"].dtype == "int64"
    assert report.by_variable["cat"].variable_type == "Cualitativa nominal"


def test_grados_descriptivos_del_porcentaje(detector):
    n = 200
    df = pd.DataFrame({
        "sin_faltantes": np.full(n, 1.0),
        "muy_baja": np.full(n, 1.0),
        "baja": np.full(n, 1.0),
        "moderada": np.full(n, 1.0),
        "alta": np.full(n, 1.0),
        "muy_alta": np.full(n, 1.0),
    })
    df.loc[:0, "muy_baja"] = np.nan
    df.loc[:1, "baja"] = np.nan
    df.loc[:19, "moderada"] = np.nan
    df.loc[:59, "alta"] = np.nan
    df.loc[:149, "muy_alta"] = np.nan
    report = detector.detect(df)

    assert report.by_variable["sin_faltantes"].missing_grade == "Sin faltantes"
    assert report.by_variable["muy_baja"].missing_grade == "Muy baja"
    assert report.by_variable["baja"].missing_grade == "Baja"
    assert report.by_variable["moderada"].missing_grade == "Moderada"
    assert report.by_variable["alta"].missing_grade == "Alta"
    assert report.by_variable["muy_alta"].missing_grade == "Muy alta"


def test_conversion_explicita_de_placeholders(detector):
    df = pd.DataFrame({"texto": ["N/A", "A", "-", "B"], "num": [1, 2, 3, 4]})
    converted = detector.convert_placeholders_to_na(df)

    assert converted["texto"].isna().sum() == 2
    assert converted["num"].isna().sum() == 0
    assert df["texto"].isna().sum() == 0
    assert df["num"].isna().sum() == 0


def test_conversion_de_placeholders_por_defecto():
    df = pd.DataFrame({"texto": ["unknown", "N/A", "C", "D"], "num": [1, 2, 3, 4]})
    converted = convert_placeholders_to_na(df)

    assert converted["texto"].isna().sum() == 2


def test_conversion_no_toca_valores_no_placeholders(detector):
    df = pd.DataFrame({"resp": ["no", "si", "0", "false"]})
    converted = detector.convert_placeholders_to_na(df)

    assert converted["resp"].isna().sum() == 0


def test_reporte_json_serializable(detector):
    df = pd.DataFrame({
        "num": [1.0, np.nan, 3.0, None],
        "cat": ["N/A", "a", "-", "b"],
    })
    payload = json.dumps(detector.detect(df).to_dict())
    parsed = json.loads(payload)

    assert parsed["total_missing_values"] == 2
    assert parsed["by_variable"]["num"]["missing_count"] == 2
    assert parsed["reproduction"]["module"] == "src.missing_data.detection"


def test_detect_requiere_dataframe(detector):
    with pytest.raises(TypeError):
        detector.detect([1, 2, 3])


def test_conversion_requiere_dataframe(detector):
    with pytest.raises(TypeError):
        detector.convert_placeholders_to_na({"a": [1, 2]})


def test_conversion_columna_inexistente(detector):
    df = pd.DataFrame({"a": [1, 2]})
    with pytest.raises(KeyError):
        detector.convert_placeholders_to_na(df, columns=["no_existe"])


def test_detect_no_muta_el_dataframe(detector):
    df = pd.DataFrame({"texto": ["N/A", "A", "-", "B"]})
    original = df.copy()
    detector.detect(df)

    pd.testing.assert_frame_equal(df, original)