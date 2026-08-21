"""
Pruebas unitarias para ReportGenerator.
"""

from src.reports.generator import ReportGenerator


def test_generator_creates_markdown_report(tmp_path):
    metadata = {
        "file_name": "articulo_ejemplo.pdf",
        "num_pages": 10,
        "title": "Evaluación de Modelos de Riesgo Actuarial",
        "authors": "Laura Martínez",
        "year": "2023",
        "journal": "Revista de Estadística y Actuaría",
        "objective": "Analizar la severidad de reclamos.",
        "research_question": "¿Cuál distribución ajusta mejor las pérdidas?",
        "hypothesis": "La distribución Log-Normal presenta un mejor ajuste.",
        "population": "Asegurados de póliza de automóvil.",
        "sample": "n = 5000 reclamos.",
        "study_design": "Observacional retrospectivo.",
        "sampling_method": "Muestreo aleatorio simple.",
        "variables": "Frecuencia y Severidad.",
        "methodology": "Ajuste de distribuciones de pérdidas mediante Máxima Verosimilitud.",
        "software": "R (fitdistrplus)",
        "results": "La distribución Gumbel logró menor AIC.",
        "limitations": "Muestra limitada a una sola región.",
        "conclusions": "Se sugiere utilizar Gumbel para estimación de VaR.",
    }

    analysis = {
        "tests_detected": ["Kolmogorov-Smirnov"],
        "models_detected": ["GLM Poisson"],
        "assumptions_required": ["Independencia de reclamos"],
        "justification_evaluation": "Ajuste validado mediante test KS.",
        "variable_classification": {
            "dependiente": "Severidad del reclamo",
            "independientes": "Edad del conductor",
            "covariables": "Antigüedad del vehículo",
        },
    }

    output_file = tmp_path / "reporte_test.md"
    generator = ReportGenerator()
    report_text = generator.generate(metadata, analysis, output_path=output_file)

    assert output_file.exists()
    assert "# INFORME DE CONSULTORÍA ESTADÍSTICA" in report_text
    assert "Evaluación de Modelos de Riesgo Actuarial" in report_text
    assert "No se especifica en el artículo." in report_text or "Toda la información contenida" in report_text
