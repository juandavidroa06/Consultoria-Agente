"""
Pruebas unitarias para ArticleExtractor.
"""

from src.article.extractor import ArticleExtractor
from src.llm.base import RuleBasedLLMClient


def test_extractor_extracts_metadata():
    sample_text = """
    Análisis del Riesgo Financiero mediante Regresión Logística
    Autor: Laura Gómez, 2024
    Objetivo: Evaluar la probabilidad de default en crédito vehicular.
    Muestra: Se analizaron n = 1500 clientes.
    Métodos estadísticos: Se utilizó Regresión Logística y prueba de Chi-cuadrado.
    Software: El análisis se realizó en R y Python.
    Resultados: El modelo obtuvo un AUC de 0.88.
    Conclusiones: Las variables de ingreso y scoring son altamente significativas.
    """

    article_data = {
        "file_name": "riesgo_crediticio.pdf",
        "num_pages": 5,
        "text": sample_text,
    }

    extractor = ArticleExtractor(llm_client=RuleBasedLLMClient())
    metadata = extractor.extract(article_data)

    assert metadata["file_name"] == "riesgo_crediticio.pdf"
    assert metadata["year"] == "2024"
    assert "R" in metadata["software"]
    assert "Python" in metadata["software"]
    assert metadata["title"] != ""
