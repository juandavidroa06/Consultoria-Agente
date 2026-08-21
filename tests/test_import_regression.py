"""
Tests de regresión de importaciones.

Reproducen el defecto detectado: `import src.missing_data` como PRIMER import
en un proceso limpio fallaba por un ciclo selector → profile → analysis →
dataset_analyzer → pipeline → selector.
"""

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _run(code: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )


def test_import_src_missing_data_primero_en_proceso_limpio():
    result = _run("import src.missing_data")
    assert result.returncode == 0, result.stderr


def test_import_src_missing_data_primero_usa_exports():
    result = _run(
        "import src.missing_data\n"
        "from src.missing_data import (MissingDataPipeline, MissingDataPipelineResult, "
        "ValidationRevisionError, run_pipeline)"
    )
    assert result.returncode == 0, result.stderr


def test_import_src_analysis_y_luego_missing_data():
    result = _run(
        "import src.analysis\n"
        "from src.missing_data import MissingDataPipeline, run_pipeline"
    )
    assert result.returncode == 0, result.stderr