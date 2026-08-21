"""
Sistema de registro de logs centralizado para PaperStats.
"""

import logging
import sys
from pathlib import Path


def setup_logger(name: str = "PaperStats", log_level: int = logging.INFO) -> logging.Logger:
    """
    Configura y devuelve un logger estandarizado.

    Args:
        name: Nombre del logger.
        log_level: Nivel de registro (logging.INFO, logging.DEBUG, etc.).

    Returns:
        logging.Logger configurado.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(log_level)
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # Handler de consola
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger
