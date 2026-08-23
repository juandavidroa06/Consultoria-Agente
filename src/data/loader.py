"""
Módulo para la carga de datos desde archivos CSV y Excel.
"""

import zipfile
from pathlib import Path
from typing import Union, Optional

import pandas as pd

from src.utils.logger import setup_logger

logger = setup_logger("DataLoader")


def load_data(
    file_path: Union[str, Path],
    sheet_name: Union[str, int, None] = 0,
    **kwargs
) -> pd.DataFrame:
    """
    Carga un conjunto de datos desde un archivo CSV o Excel (.xlsx, .xls)
    y lo retorna como un DataFrame de pandas.

    Args:
        file_path: Ruta al archivo CSV o Excel.
        sheet_name: Nombre o índice de la hoja para archivos Excel (por defecto 0).
        **kwargs: Argumentos adicionales pasados a pd.read_csv o pd.read_excel.

    Returns:
        pd.DataFrame con los datos cargados.

    Raises:
        FileNotFoundError: Si el archivo especificado no existe.
        ValueError: Si el formato del archivo no es soportado.
        RuntimeError: Si ocurre un error durante la lectura del archivo
            (CSV malformado, Excel corrupto, hoja inexistente, fallo de E/S).
    """
    path = Path(file_path)

    if not path.exists():
        logger.error(f"Archivo no encontrado: {path}")
        raise FileNotFoundError(f"No se encontró el archivo de datos: {path}")

    ext = path.suffix.lower()
    logger.info(f"Cargando archivo: {path.name} (extensión: {ext})")

    if ext not in [".csv", ".xlsx", ".xls"]:
        msg = f"Extensión de archivo '{ext}' no soportada. Use archivos .csv, .xlsx o .xls."
        logger.error(msg)
        raise ValueError(msg)

    try:
        if ext == ".csv":
            df = pd.read_csv(path, **kwargs)
        else:
            df = pd.read_excel(path, sheet_name=sheet_name, **kwargs)

        if not isinstance(df, pd.DataFrame):
            # En caso de que se lea un dict de hojas de Excel
            if isinstance(df, dict):
                first_key = list(df.keys())[0]
                df = df[first_key]

        logger.info(f"Datos cargados exitosamente: {df.shape[0]} filas, {df.shape[1]} columnas.")
        return df

    except FileNotFoundError:
        raise
    except zipfile.BadZipFile as e:
        msg = f"El archivo '{path.name}' no es un Excel válido (contenedor dañado): {e}"
        logger.error(msg)
        raise RuntimeError(msg) from e
    except pd.errors.ParserError as e:
        msg = f"El CSV '{path.name}' está mal formado (filas con número inconsistente de campos): {e}"
        logger.error(msg)
        raise RuntimeError(msg) from e
    except KeyError as e:
        msg = f"Hoja inexistente en '{path.name}': {e}. Verifique sheet_name."
        logger.error(msg)
        raise RuntimeError(msg) from e
    except UnicodeDecodeError as e:
        msg = f"No se pudo decodificar '{path.name}' como texto: {e}"
        logger.error(msg)
        raise RuntimeError(msg) from e
    except OSError as e:
        msg = f"Fallo de E/S al leer '{path.name}': {e}"
        logger.error(msg)
        raise RuntimeError(msg) from e
    except ValueError as e:
        msg = f"Contenido inválido al leer '{path.name}': {e}"
        logger.error(msg)
        raise RuntimeError(msg) from e
