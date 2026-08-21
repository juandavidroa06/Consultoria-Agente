"""
Pruebas unitarias para DatasetStatisticalAnalyzer.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from src.analysis.dataset_analyzer import DatasetStatisticalAnalyzer
from src.reports.generator import ReportGenerator


@pytest.fixture
def sample_dataset():
    np.random.seed(42)
    n = 60
    return pd.DataFrame({
        "ingreso": np.random.normal(loc=3000, scale=500, size=n),
        "gasto": np.random.normal(loc=1500, scale=300, size=n),
        "grupo_edu": ["Secundaria"] * 20 + ["Grado"] * 20 + ["Posgrado"] * 20,
        "tratamiento": ["Control"] * 30 + ["Tratado"] * 30,
    })


def test_analyzer_init(sample_dataset):
    analyzer = DatasetStatisticalAnalyzer(sample_dataset)
    assert analyzer.file_name == "DataFrame_en_memoria"
    assert "ingreso" in analyzer.variable_types


def test_analyze_two_groups(sample_dataset):
    analyzer = DatasetStatisticalAnalyzer(sample_dataset)
    results = analyzer.analyze(target_col="ingreso", group_col="tratamiento")

    assert "dataset_summary" in results
    assert "recommendations" in results
    assert len(results["recommendations"]) == 1

    rec = results["recommendations"][0]
    assert rec["recommended_test"] in [
        "t de Student (Muestras Independientes)",
        "t de Welch (Varianzas Desiguales)",
        "Mann-Whitney U (No Paramétrica)",
    ]
    assert "NO implica una relación de causalidad" in rec["causality_disclaimer"]


def test_analyze_three_groups(sample_dataset):
    analyzer = DatasetStatisticalAnalyzer(sample_dataset)
    results = analyzer.analyze(target_col="ingreso", group_col="grupo_edu")

    rec = results["recommendations"][0]
    assert rec["recommended_test"] in [
        "ANOVA de un factor (One-Way ANOVA)",
        "ANOVA de Welch (Varianzas Heterogéneas)",
        "Kruskal-Wallis H (No Paramétrica)",
    ]


def test_analyze_paired_samples(sample_dataset):
    analyzer = DatasetStatisticalAnalyzer(sample_dataset)
    results = analyzer.analyze(target_col="ingreso", paired_col="gasto")

    rec = results["recommendations"][0]
    assert rec["recommended_test"] in [
        "t de Student (Muestras Pareadas)",
        "Wilcoxon Pareado (No Paramétrica)",
    ]


def test_unevaluated_assumptions_flag(sample_dataset):
    analyzer = DatasetStatisticalAnalyzer(sample_dataset)
    results = analyzer.analyze(target_col="ingreso", group_col="tratamiento")

    assumptions = results["assumptions_status"]
    ind_ass = next(a for a in assumptions if a["assumption"] == "Independencia de las observaciones")

    assert ind_ass["status"] == "Supuesto no evaluado / Pendiente de verificación"
    assert ind_ass["evaluated"] is False


def test_dataset_report_generation(sample_dataset, tmp_path):
    analyzer = DatasetStatisticalAnalyzer(sample_dataset)
    results = analyzer.analyze(target_col="ingreso", group_col="tratamiento")

    report_file = tmp_path / "reporte_dataset.md"
    generator = ReportGenerator()
    md_text = generator.generate_dataset_report(results, output_path=report_file)

    assert report_file.exists()
    assert "# INFORME DE CONSULTORÍA ESTADÍSTICA DE DATASET" in md_text
    assert "Supuesto no evaluado / Pendiente de verificación" in md_text
