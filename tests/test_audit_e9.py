"""
Tests de regresión de la auditoría E9 (hallazgos A1-A11).

Estos tests codifican el comportamiento esperado (contrato) que el código
actual viola. No corrigen el código: documentan la regresión para su
corrección en la fase E9-B.

Fase E9-A: SOLO tests. No se modifica código de producción.
"""

import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from src.analysis.dataset_analyzer import DatasetStatisticalAnalyzer
from src.analysis.eda import describe_numerical
from src.analysis.hypothesis import (
    _clean_sample,
    t_test_1samp,
    t_test_rel,
    wilcoxon_signed_rank,
)
from src.analysis.profile import build_dataset_profile
from src.article.analyzer import StatisticalMethodologyAnalyzer
from src.data.loader import load_data
from src.data.validator import DataValidator
from src.llm.base import RuleBasedLLMClient
from src.reports.generator import ReportGenerator

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _paired_aligned(a, b):
    """Alinea dos muestras por índice eliminando pares con valores nulos."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = ~(np.isnan(a) | np.isnan(b))
    return a[mask], b[mask]


# ---------------------------------------------------------------------------
# A1: pruebas de muestras pareadas no alinean los pares por índice
# ---------------------------------------------------------------------------


class TestA1AlineacionPares:
    def test_t_test_rel_alinea_pares_por_indice(self):
        a = np.array([1.0, np.nan, 3.0, 4.0, 5.0])
        b = np.array([2.0, 2.0, np.nan, 4.0, 6.0])
        ea, eb = _paired_aligned(a, b)
        expected = stats.ttest_rel(ea, eb).statistic
        result = t_test_rel(a, b)
        assert result["statistic"] == pytest.approx(expected)

    def test_wilcoxon_signed_rank_alinea_pares_por_indice(self):
        a = np.array([1.0, np.nan, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        b = np.array([2.0, 2.0, np.nan, 4.0, 6.0, 8.0, 10.0, 12.0])
        ea, eb = _paired_aligned(a, b)
        diff = ea - eb
        diff = diff[diff != 0]
        expected_stat = stats.wilcoxon(diff).statistic
        expected_p = stats.wilcoxon(diff).pvalue
        result = wilcoxon_signed_rank(a, b)
        assert result["statistic"] == pytest.approx(expected_stat)
        assert result["p_value"] == pytest.approx(expected_p)


# ---------------------------------------------------------------------------
# A2: el analizador colapsa con entradas degeneradas
# ---------------------------------------------------------------------------


class TestA2EntradasDegeneradas:
    def test_target_100_porciento_nan_no_colapsa(self):
        df = pd.DataFrame({"t": [np.nan] * 5, "g": ["a", "b", "a", "b", "a"]})
        result = DatasetStatisticalAnalyzer(df).analyze(
            target_col="t", group_col="g"
        )
        assert isinstance(result, dict)
        assert "dataset_summary" in result

    def test_target_con_1_observacion_valida_no_colapsa(self):
        df = pd.DataFrame({"t": [5.0, np.nan, np.nan, np.nan]})
        result = DatasetStatisticalAnalyzer(df).analyze(target_col="t")
        assert isinstance(result, dict)
        assert "dataset_summary" in result
        guard = [
            d
            for d in result["diagnostics"]
            if d.get("insufficient_sample") and d.get("sample_size") == 1
        ]
        assert guard, "Se esperaba diagnóstico de normalidad no evaluable (n = 1)."
        assert "al menos 3" in guard[0]["summary"]

    def test_target_con_2_observaciones_validas_no_colapsa(self):
        df = pd.DataFrame({"t": [5.0, 5.0, np.nan, np.nan]})
        result = DatasetStatisticalAnalyzer(df).analyze(target_col="t")
        assert isinstance(result, dict)
        assert "dataset_summary" in result
        guard = [
            d
            for d in result["diagnostics"]
            if d.get("insufficient_sample") and d.get("sample_size") == 2
        ]
        assert guard, "Se esperaba diagnóstico de normalidad no evaluable (n = 2)."
        assert "al menos 3" in guard[0]["summary"]

    def test_grupo_de_una_sola_categoria_no_colapsa(self):
        df = pd.DataFrame({"t": [1, 2, 3, 4, 5, 6], "g": ["a"] * 6})
        result = DatasetStatisticalAnalyzer(df).analyze(
            target_col="t", group_col="g"
        )
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# A3: _clean_sample no elimina infinitos y las muestras constantes
#     devuelven p-valores nan silenciosos
# ---------------------------------------------------------------------------


class TestA3ManejoValoresDegenerados:
    def test_clean_sample_elimina_infinitos(self):
        cleaned = _clean_sample([1.0, 2.0, np.inf])
        assert not np.isinf(cleaned).any()

    def test_t_test_1samp_muestra_constante_lanza_error(self):
        with pytest.raises(ValueError):
            t_test_1samp([5.0, 5.0, 5.0], popmean=5.0)


# ---------------------------------------------------------------------------
# A4: popmean genera recomendaciones de correlación y los análisis de
#     correlación no registran resultados ejecutados
# ---------------------------------------------------------------------------


class TestA4CoherenciaRecomendaciones:
    def test_popmean_no_genera_recomendaciones_de_correlacion(self):
        rng = np.random.default_rng(0)
        df = pd.DataFrame(
            {"x": rng.normal(10, 2, 40), "y": rng.normal(20, 3, 40)}
        )
        result = DatasetStatisticalAnalyzer(df).analyze(
            target_col="x", popmean=10.0
        )
        recs = [r["recommended_test"] for r in result["recommendations"]]
        assert len(recs) == 1
        assert all("Correlación" not in r for r in recs)

    def test_correlacion_registra_resultado_ejecutado(self):
        rng = np.random.default_rng(1)
        df = pd.DataFrame(
            {
                "x": rng.normal(0, 1, 50),
                "y": 2 * rng.normal(0, 1, 50) + rng.normal(0, 0.5, 50),
            }
        )
        result = DatasetStatisticalAnalyzer(df).analyze()
        corr_recs = [
            r
            for r in result["recommendations"]
            if "Correlación" in r["recommended_test"]
        ]
        assert corr_recs, "Se esperaban recomendaciones de correlación."
        executed = result["executed_test_results"]
        for rec in corr_recs:
            assert rec["recommended_test"] in executed, (
                f"Falta resultado ejecutado para {rec['recommended_test']}"
            )
            assert 0.0 <= executed[rec["recommended_test"]]["p_value"] <= 1.0

    def test_correlacion_alpha_determina_significancia(self):
        rng = np.random.default_rng(15)
        x = rng.normal(0, 1, 40)
        y = 0.3 * x + rng.normal(0, 1, 40)
        df = pd.DataFrame({"x": x, "y": y})
        res_lo = DatasetStatisticalAnalyzer(df).analyze(alpha=0.05)
        res_hi = DatasetStatisticalAnalyzer(df).analyze(alpha=0.01)

        recs_lo = [
            r
            for r in res_lo["recommendations"]
            if "Correlación" in r["recommended_test"]
        ]
        assert recs_lo, "Se esperaban recomendaciones de correlación."
        key = recs_lo[0]["recommended_test"]

        exec_lo = res_lo["executed_test_results"][key]
        exec_hi = res_hi["executed_test_results"][key]

        schema_fields = (
            "test_name", "statistic", "p_value", "alpha",
            "null_hypothesis", "alt_hypothesis", "decision",
            "reject_h0", "interpretation",
        )
        for field in schema_fields:
            assert field in exec_lo, f"Falta campo '{field}' en el resultado ejecutado."

        assert exec_lo["statistic"] == pytest.approx(exec_hi["statistic"])
        assert exec_lo["p_value"] == pytest.approx(exec_hi["p_value"])

        assert exec_lo["alpha"] == 0.05 and exec_lo["reject_h0"] is True
        assert exec_hi["alpha"] == 0.01 and exec_hi["reject_h0"] is False
        assert exec_lo["decision"] == "Rechazar H0"
        assert exec_hi["decision"] == "No rechazar H0"


# ---------------------------------------------------------------------------
# A5: describe_numerical no valida columnas no numéricas y reporta
#     std/skew como 0 (no NaN) con muestras pequeñas
# ---------------------------------------------------------------------------


class TestA5DescripcionNumerica:
    def test_valida_columnas_no_numericas(self):
        df = pd.DataFrame({"c": ["a", "b", "a"]})
        with pytest.raises(ValueError):
            describe_numerical(df, columns=["c"])

    def test_muestra_unica_std_es_nan(self):
        result = describe_numerical(pd.DataFrame({"x": [5.0]}))
        assert np.isnan(result.loc["x", "std"])

    def test_muestra_doble_skew_es_nan(self):
        result = describe_numerical(pd.DataFrame({"x": [1.0, 2.0]}))
        assert np.isnan(result.loc["x", "skewness"])


# ---------------------------------------------------------------------------
# A6: build_dataset_profile registra temporal_inference siempre False
# ---------------------------------------------------------------------------


class TestA6PerfilTemporal:
    def test_registra_inferencia_temporal(self):
        df = pd.DataFrame(
            {
                "t": pd.to_datetime(
                    ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04"]
                ),
                "v": [1, 2, 3, 4],
            }
        )
        profile = build_dataset_profile(df)
        assert profile.temporal is True
        assert profile.reproduction["temporal_inference"] is True


# ---------------------------------------------------------------------------
# A7: identificación de tipos: datetime con 2 únicos se etiqueta "Binaria"
#     y columnas totalmente nulas se etiquetan "Cuantitativa discreta"
# ---------------------------------------------------------------------------


class TestA7TiposDeVariables:
    def test_datetime_con_dos_valores_es_variable_de_tiempo(self):
        df = pd.DataFrame(
            {
                "t": pd.to_datetime(
                    ["2020-01-01", "2020-01-02", "2020-01-01", "2020-01-02"]
                )
            }
        )
        types = DataValidator.identify_variable_types(df)
        assert types["t"] == "Variable de tiempo"

    def test_columna_totalmente_nan_no_es_discreta(self):
        df = pd.DataFrame({"v": [np.nan] * 5, "x": [1, 2, 3, 4, 5]})
        types = DataValidator.identify_variable_types(df)
        assert types["v"] != "Cuantitativa discreta"


# ---------------------------------------------------------------------------
# A8: generate_missing_data_report no tolera overall_missing_percentage
#     no numérico
# ---------------------------------------------------------------------------


class TestA8ReporteFaltantes:
    def _payload(self):
        return {
            "status": "imputado",
            "continued": True,
            "detection_report": {
                "total_missing_values": 1,
                "overall_missing_percentage": None,
                "overall_missing_grade": "Alta",
                "complete_cases": 0,
                "variables_with_missing": [],
            },
            "diagnostics_report": {},
            "candidate_methods": [],
            "selection_report": {},
            "validation_report": {},
            "validation_verdict": "Aceptable",
            "applied_methods": {},
            "skipped_variables": {},
            "n_imputed_cells": 0,
        }

    def test_tolera_missing_percentage_no_numerico(self):
        markdown = ReportGenerator().generate_missing_data_report(
            self._payload()
        )
        assert isinstance(markdown, str)
        assert markdown.strip()


# ---------------------------------------------------------------------------
# A9: extracción de metadatos LLM
# ---------------------------------------------------------------------------


class TestA9MetadatosLLM:
    def test_extrae_tamano_muestral_con_notacion_n(self):
        metadata = RuleBasedLLMClient().extract_metadata(
            "Pacientes evaluados: n = 1500 en total.\n\n"
            "Método: encuesta transversal.\n\n"
        )
        assert "1500" in metadata["sample"]

    def test_no_falso_positivo_software_r(self):
        metadata = RuleBasedLLMClient().extract_metadata(
            "El R^2 fue 0.93 y los análisis se realizaron con Python.\n\n"
        )
        assert "R" not in metadata["software"]

    def test_analisis_metodologico_determinista_entre_procesos(self):
        text = (
            "Se aplicó regresión lineal y t de Student. "
            "Análisis de varianza incluido.\n\n"
        )
        code = (
            "import logging; logging.disable(logging.CRITICAL);"
            "import src.analysis;"
            "from src.llm.base import RuleBasedLLMClient;"
            "import json;"
            f"r = RuleBasedLLMClient().analyze_methodology({{}}, {text!r});"
            "print(json.dumps(r['assumptions_required']))"
        )
        outputs = set()
        for seed in [0, 1, 2]:
            env = dict(os.environ)
            env["PYTHONHASHSEED"] = str(seed)
            proc = subprocess.run(
                [sys.executable, "-c", code],
                env=env,
                capture_output=True,
                text=True,
                cwd=PROJECT_ROOT,
            )
            assert proc.returncode == 0, proc.stderr
            outputs.add(proc.stdout.strip())
        assert len(outputs) == 1, (
            "El orden de assumptions_required varía entre procesos: "
            f"{sorted(outputs)}"
        )


# ---------------------------------------------------------------------------
# A10: errores de carga de datos no siguen el contrato documentado
# ---------------------------------------------------------------------------


class TestA10CargaDatos:
    def test_csv_malformado_lanza_runtimeerror(self, tmp_path):
        path = tmp_path / "malo.csv"
        path.write_text('a,b\n"1,2\n3,4')
        with pytest.raises(RuntimeError):
            load_data(path)


# ---------------------------------------------------------------------------
# A11: las covariables nunca se clasifican y variables=None colapsa el análisis
# ---------------------------------------------------------------------------


class TestA11ClasificacionVariables:
    def test_clasifica_covariables(self):
        metadata = {
            "variables": (
                "Variable dependiente: ingreso. "
                "Independientes: edad, educación. "
                "Covariables: sexo, región."
            )
        }
        result = StatisticalMethodologyAnalyzer().analyze(
            metadata, full_text=""
        )
        covariables = result["variable_classification"]["covariables"]
        assert "sexo" in covariables
        assert covariables != "No se especifica en el artículo."

    def test_variables_none_no_colapsa(self):
        metadata = {"variables": None}
        result = StatisticalMethodologyAnalyzer().analyze(
            metadata, full_text=""
        )
        assert "variable_classification" in result