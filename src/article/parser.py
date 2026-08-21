"""
Módulo para la carga y extracción de texto desde archivos PDF o texto plano.
"""

from pathlib import Path
from typing import Dict, Any, Union
import pypdf
from src.utils.logger import setup_logger

logger = setup_logger("ArticleParser")


class ArticleParser:
    """
    Parser encargado de verificar la existencia del archivo, validar su extensión y
    extraer el contenido textual por páginas.
    """

    def __init__(self, file_path: Union[str, Path]):
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"No se encontró el archivo: {self.file_path}")

    def parse(self) -> Dict[str, Any]:
        """
        Lee el archivo especificado y extrae su texto.

        Returns:
            Dict con 'text' (texto completo), 'num_pages' y 'file_name'.
        """
        ext = self.file_path.suffix.lower()
        logger.info(f"Procesando archivo: {self.file_path.name} ({ext})")

        if ext == ".pdf":
            return self._parse_pdf()
        elif ext in [".txt", ".md"]:
            return self._parse_text()
        else:
            raise ValueError(f"Formato de archivo no soportado: {ext}. Formatos aceptados: .pdf, .txt, .md")

    def _parse_pdf(self) -> Dict[str, Any]:
        pages_text = []
        try:
            reader = pypdf.PdfReader(self.file_path)
            num_pages = len(reader.pages)
            logger.info(f"PDF cargado con {num_pages} páginas.")

            for i, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                pages_text.append(text)

            full_text = "\n".join(pages_text).strip()
            if not full_text:
                logger.warning("El PDF no contiene texto extraíble (posiblemente sea una imagen escaneada).")

            return {
                "file_name": self.file_path.name,
                "file_path": str(self.file_path),
                "num_pages": num_pages,
                "text": full_text,
                "pages": pages_text,
            }
        except Exception as e:
            logger.error(f"Error al leer el archivo PDF {self.file_path}: {e}")
            raise RuntimeError(f"Fallo en la lectura del PDF: {e}") from e

    def _parse_text(self) -> Dict[str, Any]:
        try:
            with open(self.file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

            return {
                "file_name": self.file_path.name,
                "file_path": str(self.file_path),
                "num_pages": 1,
                "text": content.strip(),
                "pages": [content.strip()],
            }
        except Exception as e:
            logger.error(f"Error al leer el archivo de texto {self.file_path}: {e}")
            raise RuntimeError(f"Fallo en la lectura del archivo de texto: {e}") from e
