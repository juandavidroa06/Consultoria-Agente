"""
Pruebas unitarias para StatisticalMethodologyAnalyzer.
"""

from src.article.analyzer import StatisticalMethodologyAnalyzer
from src.llm.base import RuleBasedLLMClient


def test_analyzer_detects_tests_and_models():
    sample_text = """
    Se evaluó la normalidad con la prueba de Shapiro-Wilk y la homocedasticidad con Levene.
    Se ajustó un modelo de Regresión Lineal OLS y ANOVA para comparar los tres grupos.
    """

    metadata = {
        "variables": "Variable dependiente: Rendimiento. Variables independientes: Tiempo de estudio y Edad.",
        "statistical_methods": "Shapiro-Wilk, Levene, ANOVA, Regresión Lineal",
    }

    analyzer = StatisticalMethodologyAnalyzer(llm_client=RuleBasedLLMClient())
    analysis = analyzer.analyze(metadata, full_text=sample_text)

    assert "Shapiro-Wilk" in analysis["tests_detected"]
    assert "Levene" in analysis["tests_detected"]
    assert "ANOVA" in analysis["tests_detected"]
    assert "Regresión Lineal" in analysis["models_detected"]
    assert len(analysis["assumptions_required"]) > 0
