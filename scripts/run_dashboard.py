# -*- coding: utf-8 -*-
"""Lanzador del dashboard interactivo.

Ejecuta la aplicación Streamlit ubicada en src/deliverables/dashboard.py
usando el intérprete del entorno virtual del proyecto (.venv).

Uso:
    python run_dashboard.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DASHBOARD_PATH = PROJECT_ROOT / "src" / "deliverables" / "dashboard.py"


def main() -> int:
    """Lanza `streamlit run` sobre el dashboard y devuelve el código de salida."""
    if not DASHBOARD_PATH.exists():
        print(f"[ERROR] No se encontró el dashboard en: {DASHBOARD_PATH}")
        return 1

    comando = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(DASHBOARD_PATH),
        "--theme.primaryColor",
        "#2E6E8E",
    ]
    print("Iniciando dashboard... (se abrirá en el navegador; Ctrl+C para detener)")
    try:
        return subprocess.call(comando)
    except KeyboardInterrupt:
        print("\nDashboard detenido por el usuario.")
        return 0
    except OSError as exc:
        print(f"[ERROR] No se pudo iniciar Streamlit: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
