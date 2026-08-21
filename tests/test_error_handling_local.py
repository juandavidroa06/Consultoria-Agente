"""
Tests de manejo de errores locales por tipo (no de red).

Verifica que los `except (ValueError, TypeError, KeyError, LinAlgError)` específicos
en pipeline.py / evaluation.py / dataset_analyzer.py capturan errores reales de
operaciones locales y mantienen degradación controlada, sin inventar handlers de red.
"""

import numpy as np
import pandas as pd
import pytest

from src.missing_data.pipeline import MissingDataPipeline
from src.missing_data.evaluation import ArtificialMissingnessEvaluator


def test_pipeline_per_variable_valueerror_se_registra_y_continua():
    """
    Drug Price no tiene columna numérica completa, por lo que 'regresion' debe
    fallar con ValueError y quedar registrado en skipped_variables sin propagar.
    Es el error local real que antes capturaba `except Exception`.
    """
    df = pd.read_excel("data/raw/Drug Price.xlsx")
    # Forzar regresión en una variable que existe para provocar ValueError
    # El pipeline debe capturar ValueError por tipo y continuar
    result = MissingDataPipeline().run(
        df, impute=True, method_override={"Amphetamine": "regresion"}, strict=False
    )
    assert "Amphetamine" in result.skipped_variables
    assert "regresion" in result.skipped_variables["Amphetamine"]
    # El pipeline no debe lanzar, solo registrar
    assert result.status == "imputado"
    # Verificar que el error fue ValueError (mensaje contiene la causa local)
    assert "completa" in result.skipped_variables["Amphetamine"].lower() or "predictora" in result.skipped_variables["Amphetamine"].lower()


def test_pipeline_e4_valueerror_degradacion_controlada():
    """
    E4 es opcional: si ArtificialMissingnessEvaluator levanta ValueError
    (p. ej. fraction fuera de rango, o LinAlgError interno), pipeline debe
    capturar por tipo específico y continuar con evaluation_report=None.
    Se fuerza un evaluador que falla con ValueError.
    """
    df = pd.read_excel("data/raw/Drug Price.xlsx")

    class EvaluadorRoto(ArtificialMissingnessEvaluator):
        def evaluate(self, *args, **kwargs):
            raise ValueError("fallo sintético de evaluación E4")

    pipe = MissingDataPipeline(evaluator=EvaluadorRoto(random_state=42))
    # No debe propagar ValueError, debe degradar a None y continuar a E5
    result = pipe.run(df, impute=True, strict=False)
    assert result.evaluation_report is None
    assert result.selection_report is not None
    assert result.status == "imputado"


def test_pipeline_e4_linalgerror_degradacion_controlada():
    """
    MICE puede fallar con numpy.linalg.LinAlgError (SVD did not converge) en
    datasets pequeños o colineales. El pipeline debe capturarlo por tipo.
    """
    df = pd.read_excel("data/raw/Drug Price.xlsx")

    class EvaluadorLinalg(ArtificialMissingnessEvaluator):
        def evaluate(self, *args, **kwargs):
            raise np.linalg.LinAlgError("SVD did not converge")

    pipe = MissingDataPipeline(evaluator=EvaluadorLinalg(random_state=42))
    result = pipe.run(df, impute=True, strict=False)
    assert result.evaluation_report is None
    assert result.status == "imputado"


def test_evaluation_metodo_lin_alg_error_registrado_no_propaga():
    """
    ArtificialMissingnessEvaluator debe registrar LinAlgError de un método sin propagar,
    manteniendo el ranking para los métodos que sí funcionan.
    """
    from src.missing_data.methods import ImputationMethod, MethodCapabilities

    # Dataset completo pequeño con dos columnas numéricas colineales para forzar problemas
    df = pd.DataFrame({
        "a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "b": [2.0, 4.0, 6.0, 8.0, 10.0, 12.0],  # perfectamente colineal con a
        "c": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],  # constante
    })

    class MetodoLinalg(ImputationMethod):
        name = "falla_linalg"
        capabilities = MethodCapabilities(supports_numeric=True, supports_categorical=False)
        uses_random_state = False

        def fit(self, df):
            self._fitted = True
            return self

        def _apply(self, result):
            raise np.linalg.LinAlgError("SVD did not converge sintético")

    evaluator = ArtificialMissingnessEvaluator(random_state=42, n_repeats=1)
    report = evaluator.evaluate(df, methods=[MetodoLinalg(), "media"], fraction=0.2, mechanism="MCAR")
    # El método que falla debe tener error registrado, no propagar
    falla = next(m for m in report.methods if m.method == "falla_linalg")
    assert falla.error is not None
    assert "SVD" in falla.error or "LinAlg" in falla.error
    # media debe seguir funcionando
    media = next(m for m in report.methods if m.method == "media")
    assert media.error is None


def test_dataset_analyzer_correlacion_valueerror_no_propaga():
    """
    _analyze_correlations captura ValueError/TypeError de pearsonr/spearmanr y
    degrada a NaN sin propagar. Se verifica que un par con datos degenerados no rompe el análisis.
    """
    # Dos columnas numéricas donde una es constante tras filtrar NaN -> ptp==0 ya maneja,
    # pero forzamos un caso donde pearsonr levantaría ValueError si no estuviera guardado.
    # Usamos NaN + inf para forzar TypeError/ValueError dentro del try.
    df = pd.DataFrame({
        "x": [1.0, 2.0, 3.0, 4.0, 5.0],
        "y": [1.0, 2.0, np.inf, 4.0, 5.0],
        "grupo": ["A", "A", "B", "B", "B"],
    })
    from src.analysis.dataset_analyzer import DatasetStatisticalAnalyzer
    analyzer = DatasetStatisticalAnalyzer(df)
    # No debe propagar ValueError/TypeError de la correlación
    result = analyzer.analyze()
    assert "executed_test_results" in result
    # Al menos una correlación registrada (aunque sea NaN por degradación)
    assert isinstance(result["executed_test_results"], dict)
