"""
Pruebas unitarias para el registro de métodos de imputación (Etapa E3).
"""

import pandas as pd
import pytest

from src.missing_data.methods import (
    ConstantImputation,
    ImputationMethod,
    MeanImputation,
    ModeImputation,
)
from src.missing_data.registry import (
    DEFAULT_METHOD_NAMES,
    ImputationRegistry,
    default_registry,
    candidates_for,
    get,
    names,
    register,
    summary,
)


@pytest.fixture
def registry():
    return ImputationRegistry()


def test_registro_por_defecto_contiene_todos_los_metodos():
    assert names() == DEFAULT_METHOD_NAMES


def test_get_devuelve_clase_correcta():
    assert get("media") is MeanImputation
    assert get("constante") is ConstantImputation


def test_get_metodo_no_registrado_lanza_keyerror():
    with pytest.raises(KeyError):
        get("metodo_inexistente")


def test_summary_estructura_correcta():
    entries = summary()
    assert len(entries) == 10
    entry = next(e for e in entries if e["name"] == "media")
    assert entry["supports_numeric"] is True
    assert entry["supports_categorical"] is False
    assert entry["temporal_only"] is False
    assert entry["needs_other_columns"] is False
    assert entry["uses_random_state"] is False


def test_summary_capacidades_por_metodo():
    entries = {e["name"]: e for e in summary()}
    assert entries["moda"]["supports_categorical"] is True
    assert entries["moda"]["supports_numeric"] is True
    assert entries["interpolacion_lineal"]["temporal_only"] is True
    assert entries["locf"]["temporal_only"] is True
    assert entries["knn"]["needs_other_columns"] is True
    assert entries["mice"]["uses_random_state"] is True
    assert entries["iterativo"]["uses_random_state"] is True


def test_candidates_for_numerico_sin_temporal():
    df = pd.DataFrame({"x": [1.0, 2.0, None], "y": [1, 2, 3]})
    candidatos = candidates_for(df)
    assert "media" in candidatos
    assert "knn" in candidatos
    assert "moda" in candidatos
    assert "interpolacion_lineal" not in candidatos
    assert "locf" not in candidatos


def test_candidates_for_con_temporal():
    df = pd.DataFrame({"x": [1.0, None, 3.0]})
    candidatos = candidates_for(df, temporal=True)
    assert "interpolacion_lineal" in candidatos
    assert "locf" in candidatos


def test_candidates_for_sin_faltantes_devuelve_vacio():
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    assert candidates_for(df) == []


def test_candidates_for_solo_categorico():
    df = pd.DataFrame({"z": ["a", "b", None]})
    candidatos = candidates_for(df)
    assert "moda" in candidatos
    assert "constante" in candidatos
    assert "media" not in candidatos
    assert "knn" not in candidatos


def test_candidates_for_rechaza_no_dataframe():
    with pytest.raises(TypeError):
        candidates_for([1, 2, 3])


def test_register_metodo_custom(registry):
    class CustomMethod(ImputationMethod):
        name = "custom"
        capabilities = ImputationMethod.capabilities

        def fit(self, df):
            self._fitted = True
            return self

        def _apply(self, result):
            return result

    registry.register("custom", CustomMethod)
    assert "custom" in registry.names()
    assert registry.get("custom") is CustomMethod


def test_register_duplicado_lanza_valorerror(registry):
    class CustomMethod(ImputationMethod):
        name = "custom"
        capabilities = ImputationMethod.capabilities

        def fit(self, df):
            self._fitted = True
            return self

        def _apply(self, result):
            return result

    registry.register("dup", CustomMethod)
    with pytest.raises(ValueError):
        registry.register("dup", CustomMethod)


def test_register_overwrite_permitido(registry):
    registry.register("media", MeanImputation, overwrite=True)
    assert "media" in registry.names()


def test_register_nombre_vacio_lanza_valorerror(registry):
    with pytest.raises(ValueError):
        registry.register("", MeanImputation)


def test_register_clase_invalida_lanza_typeerror(registry):
    with pytest.raises(TypeError):
        registry.register("mal", MeanImputation.__init__)


def test_registro_por_funcion_global():
    class DummyMethod(ImputationMethod):
        name = "dummy_e3_test"
        capabilities = ImputationMethod.capabilities

        def fit(self, df):
            self._fitted = True
            return self

        def _apply(self, result):
            return result

    register("dummy_e3_test", DummyMethod)
    try:
        assert get("dummy_e3_test") is DummyMethod
    finally:
        default_registry._methods.pop("dummy_e3_test", None)


def test_summary_metodo_custom(registry):
    class Dummy2(ImputationMethod):
        name = "dummy2"
        capabilities = ModeImputation.capabilities

        def fit(self, df):
            self._fitted = True
            return self

        def _apply(self, result):
            return result

    registry.register("dummy2", Dummy2)
    entry = next(e for e in registry.summary() if e["name"] == "dummy2")
    assert entry["supports_categorical"] is True
