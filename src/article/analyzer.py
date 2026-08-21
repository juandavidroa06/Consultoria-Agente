"""
Módulo para el análisis crítico de la metodología estadística empleada en un artículo.
"""

from typing import Dict, Any, Optional
from src.llm.base import BaseLLMClient, RuleBasedLLMClient
from src.utils.logger import setup_logger

logger = setup_logger("StatisticalMethodologyAnalyzer")


class StatisticalMethodologyAnalyzer:
    """
    Analizador formal de la metodología estadística.
    Evalúa la congruencia del diseño, los modelos estadísticos utilizados,
    las pruebas de hipótesis aplicadas y los supuestos estadísticos requeridos.
    """

    def __init__(self, llm_client: Optional[BaseLLMClient] = None):
        self.llm_client = llm_client or RuleBasedLLMClient()

    def analyze(self, metadata: Dict[str, Any], full_text: str = "") -> Dict[str, Any]:
        """
        Ejecuta el análisis metodológico riguroso.

        Args:
            metadata: Diccionario de metadatos extraídos por ArticleExtractor.
            full_text: Texto completo del artículo para análisis contextual.

        Returns:
            Diccionario con análisis de métodos, pruebas, modelos, supuestos y recomendaciones.
        """
        logger.info("Analizando metodología estadística y supuestos del artículo.")
        analysis = self.llm_client.analyze_methodology(metadata, full_text)

        # Clasificación y justificación de variables
        analysis["variable_classification"] = self._classify_variables(metadata.get("variables", ""))

        return analysis

    def _classify_variables(self, variables_text: str) -> Dict[str, str]:
        """
        Clasifica las variables reportadas según su naturaleza estadística.
        """
        if not variables_text:
            variables_text = ""
        text_lower = variables_text.lower()
        classification = {
            "dependiente": "No se especifica en el artículo.",
            "independientes": "No se especifica en el artículo.",
            "covariables": "No se especifica en el artículo.",
            "naturaleza": "Variables cuantitativas o cualitativas según el contexto del estudio.",
        }

        if "dependiente" in text_lower or "outcome" in text_lower or "respuesta" in text_lower:
            classification["dependiente"] = variables_text[:150]
        if "independiente" in text_lower or "predictor" in text_lower or "explicativa" in text_lower:
            classification["independientes"] = variables_text[:200]
        if "covariable" in text_lower:
            classification["covariables"] = variables_text[:200]

        return classification
