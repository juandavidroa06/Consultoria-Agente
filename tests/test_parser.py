"""
Pruebas unitarias para ArticleParser.
"""

import pytest
from pathlib import Path
from src.article.parser import ArticleParser


def test_parser_file_not_found():
    with pytest.raises(FileNotFoundError):
        ArticleParser("non_existent_file.pdf")


def test_parser_text_file(tmp_path):
    sample_file = tmp_path / "test_article.txt"
    sample_file.write_text("Título: Estudio Estadístico\nObjetivo: Analizar datos.", encoding="utf-8")

    parser = ArticleParser(sample_file)
    result = parser.parse()

    assert result["file_name"] == "test_article.txt"
    assert result["num_pages"] == 1
    assert "Estudio Estadístico" in result["text"]


def test_parser_unsupported_extension(tmp_path):
    invalid_file = tmp_path / "test.xyz"
    invalid_file.write_text("contenido", encoding="utf-8")

    parser = ArticleParser(invalid_file)
    with pytest.raises(ValueError, match="Formato de archivo no soportado"):
        parser.parse()
