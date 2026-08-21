"""
Pruebas unitarias para load_data.
"""

import pytest
import pandas as pd
from pathlib import Path
from src.data.loader import load_data


def test_load_csv_success(tmp_path):
    csv_file = tmp_path / "datos.csv"
    csv_file.write_text("edad,ingreso\n25,3000\n30,4500\n35,5000\n", encoding="utf-8")

    df = load_data(csv_file)
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (3, 2)
    assert "edad" in df.columns


def test_load_excel_success(tmp_path):
    excel_file = tmp_path / "datos.xlsx"
    df_sample = pd.DataFrame({"grupo": ["A", "B"], "valor": [10, 20]})
    df_sample.to_excel(excel_file, index=False)

    df = load_data(excel_file)
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (2, 2)


def test_load_data_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_data("archivo_inexistente.csv")


def test_load_data_invalid_extension(tmp_path):
    invalid_file = tmp_path / "datos.json"
    invalid_file.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="no soportada"):
        load_data(invalid_file)
