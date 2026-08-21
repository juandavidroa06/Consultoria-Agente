"""
Módulo para la extracción de metadatos e información clave de artículos científicos.
"""

from typing import Dict, Any, Optional
from src.llm.base import BaseLLMClient, RuleBasedLLMClient
from src.utils.logger import setup_logger

logger = setup_logger("ArticleExtractor")


class ArticleExtractor:
    """
    Extractor estructurado de metadatos y secciones científicas.
    Extrae título, autores, año, objetivos, muestra, hipótesis, métodos estadísticos, resultados, etc.
    """

    def __init__(self, llm_client: Optional[BaseLLMClient] = None):
        self.llm_client = llm_client or RuleBasedLLMClient()

    def extract(self, article_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extrae metadatos estructurados a partir del texto procesado por ArticleParser.

        Args:
            article_data: Diccionario retornado por ArticleParser.parse().

        Returns:
            Diccionario estructurado con los 19 elementos clave del artículo.
        """
        text = article_data.get("text", "")
        if not text:
            logger.warning("Se recibió un artículo sin texto para extraer metadatos.")

        logger.info("Iniciando extracción de metadatos del artículo.")
        metadata = self.llm_client.extract_metadata(text)

        # Asignar metadatos adicionales del archivo
        metadata["file_name"] = article_data.get("file_name", "Desconocido")
        metadata["num_pages"] = article_data.get("num_pages", 0)

        return metadata
