"""
Pruebas unitarias para DataValidator.
"""

import pytest
import pandas as pd
import numpy as np
from src.data.validator import DataValidator


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "binaria": [0, 1, 0, 1, 0],
        "discreta": [1, 2, 3, 2, 1],
        "continua": [10.5, 20.3, np.nan, 40.1, 50.8],
        "categoria": ["A", "B", "A", "C", "B"],
    })


def test_identify_variable_types(sample_df):
    types = DataValidator.identify_variable_types(sample_df)
    assert types["binaria"] == "Binaria"
    assert types["discreta"] == "Cuantitativa discreta"
    assert types["continua"] == "Cuantitativa continua"
    assert types["categoria"] == "Cualitativa nominal"


def test_detect_missing_values(sample_df):
    missing = DataValidator.detect_missing_values(sample_df)
    assert missing["total_missing_values"] == 1
    assert missing["columns"]["continua"]["missing_count"] == 1
    assert missing["columns"]["binaria"]["missing_count"] == 0


def test_detect_duplicates():
    df = pd.DataFrame({"a": [1, 2, 1], "b": [3, 4, 3]})
    dups = DataValidator.detect_duplicates(df)
    assert dups["duplicate_count"] == 1


def test_summarize_data_quality(sample_df):
    summary = DataValidator.summarize_data_quality(sample_df)
    assert summary["dimensions"]["rows"] == 5
    assert summary["dimensions"]["columns"] == 4
    assert "variable_types" in summary
