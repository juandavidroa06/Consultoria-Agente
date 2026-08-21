"""
Generador de informes estadísticos estructurados en formato Markdown.
"""

from pathlib import Path
from typing import Dict, Any, Union, Optional
from src.utils.logger import setup_logger

logger = setup_logger("ReportGenerator")


class ReportGenerator:
    """
    Generador de informes rigurosos en Markdown basados en la extracción
    y análisis metodológico de artículos científicos o conjuntos de datos.
    """

    def generate(self, metadata: Dict[str, Any], analysis: Dict[str, Any], output_path: Union[str, Path] = None) -> str:
        """
        Compila el informe estructurado de un artículo científico en formato Markdown.

        Args:
            metadata: Metadatos extraídos por ArticleExtractor.
            analysis: Resultados del análisis estadístico por StatisticalMethodologyAnalyzer.
            output_path: Ruta opcional para guardar el archivo Markdown.

        Returns:
            String en Markdown con el informe completo.
        """
        logger.info("Generando informe estadístico de artículo.")

        def get_val(key, default="No se especifica en el artículo."):
            val = metadata.get(key)
            return val if val and str(val).strip() else default

        tests_list = "\n".join([f"- {test}" for test in analysis.get("tests_detected", [])])
        models_list = "\n".join([f"- {model}" for model in analysis.get("models_detected", [])])
        assumptions_list = "\n".join([f"- {ass}" for ass in analysis.get("assumptions_required", [])])

        report_md = f"""# INFORME DE CONSULTORÍA ESTADÍSTICA — ANÁLISIS DE ARTÍCULO CIENTÍFICO

**Nombre del Archivo**: `{metadata.get('file_name', 'Desconocido')}`  
**Páginas**: {metadata.get('num_pages', 'N/A')}  

---

## 1. FICHA TÉCNICA DEL ARTÍCULO

- **Título**: {get_val('title')}
- **Autores**: {get_val('authors')}
- **Año**: {get_val('year')}
- **Revista**: {get_val('journal')}

---

## 2. OBJETIVOS E HIPÓTESIS

- **Objetivo del Estudio**: {get_val('objective')}
- **Pregunta de Investigación**: {get_val('research_question')}
- **Hipótesis Planteada**: {get_val('hypothesis')}

---

## 3. POBLACIÓN Y MUESTREO

- **Población Objetivo**: {get_val('population')}
- **Muestra y Tamaño Muestral**: {get_val('sample')}
- **Diseño del Estudio**: {get_val('study_design')}
- **Método de Muestreo**: {get_val('sampling_method')}

---

## 4. ANÁLISIS DE VARIABLES

- **Descripción General de Variables**: {get_val('variables')}
- **Variable Dependiente / Respuesta**: {analysis.get('variable_classification', {}).get('dependiente', 'No se especifica en el artículo.')}
- **Variables Independientes / Predictores**: {analysis.get('variable_classification', {}).get('independientes', 'No se especifica en el artículo.')}
- **Covariables / Control**: {analysis.get('variable_classification', {}).get('covariables', 'No se especifica en el artículo.')}

---

## 5. METODOLOGÍA Y MODELOS ESTADÍSTICOS

### Descripción Metodológica
{get_val('methodology')}

### Pruebas de Hipótesis Identificadas
{tests_list}

### Modelos Estadísticos Identificados
{models_list}

### Software y Librerías Utilizadas
- **Software Reportado**: {get_val('software')}

---

## 6. EVALUACIÓN Y DIAGNÓSTICO DE SUPUESTOS

### Supuestos Estadísticos Requeridos
{assumptions_list}

### Evaluación Metodológica de PaperStats
> [!NOTE]
> {analysis.get('justification_evaluation', 'Evaluación no disponible.')}

---

## 7. RESULTADOS Y CONCLUSIONES DEL ESTUDIO

- **Resultados Principales**: {get_val('results')}
- **Limitaciones Declaradas**: {get_val('limitations')}
- **Conclusiones de los Autores**: {get_val('conclusions')}

---

## 8. RESTRICCIÓN SOBRE INFORMACIÓN Y ALUCINACIONES

> [!IMPORTANT]
> *Toda la información contenida en las secciones anteriores refleja estrictamente lo reportado por los autores o la interpretación metodológica explícita de PaperStats. Cualquier dato o método no mencionado expresamente en el texto original se marca como "No se especifica en el artículo."*
"""

        if output_path:
            out_file = Path(output_path)
            out_file.parent.mkdir(parents=True, exist_ok=True)
            with open(out_file, "w", encoding="utf-8") as f:
                f.write(report_md)
            logger.info(f"Informe guardado en: {out_file}")

        return report_md

    def generate_dataset_report(self, analysis_results: Dict[str, Any], output_path: Optional[Union[str, Path]] = None) -> str:
        """
        Compila el informe de consultoría estadística para un conjunto de datos en formato Markdown.

        Args:
            analysis_results: Resultados retornados por DatasetStatisticalAnalyzer.analyze().
            output_path: Ruta opcional para guardar el informe.

        Returns:
            String en Markdown con el informe estructurado.
        """
        logger.info("Generando informe estadístico para conjunto de datos.")

        sum_data = analysis_results.get("dataset_summary", {})
        quality = analysis_results.get("data_quality", {})
        assumptions = analysis_results.get("assumptions_status", [])
        recommendations = analysis_results.get("recommendations", [])
        explanation = analysis_results.get("pedagogical_explanation", "")

        assumptions_md = "\n".join([
            f"- **{a['assumption']}**: `{a['status']}`" for a in assumptions
        ])

        recs_md = ""
        for r in recommendations:
            rec_title = r.get("recommended_test", "Recomendación")
            just = r.get("statistical_justification", "")
            disclaimer = r.get("causality_disclaimer", "")
            recs_md += f"### {rec_title}\n- **Justificación Estadística**: {just}\n- **Advertencia**: *{disclaimer}*\n\n"

        report_md = f"""# INFORME DE CONSULTORÍA ESTADÍSTICA DE DATASET — PAPERSTATS

**Archivo**: `{sum_data.get('file_name', 'DataFrame')}`  
**Dimensiones**: {sum_data.get('rows', 0)} filas, {sum_data.get('columns', 0)} columnas  

---

## 1. RESUMEN Y CALIDAD DE DATOS

- **Valores Faltantes Totales**: {quality.get('missing_values', {}).get('total_missing_values', 0)}
- **Filas Duplicadas**: {quality.get('duplicates', {}).get('duplicate_count', 0)}

---

## 2. EVALUACIÓN Y ESTADO DE SUPUESTOS ESTADÍSTICOS

{assumptions_md}

---

## 3. RECOMENDACIONES METODOLÓGICAS Y JUSTIFICACIÓN

{recs_md}

---

## 4. EXPLICACIÓN PEDAGÓGICA Y RECOMENDACIONES DE ESTUDIO

{explanation}

---

> [!IMPORTANT]
> *Los análisis cuantitativos indican patrones y diferencias estadísticamente significativas en los datos analizados. La significancia estadística no implica causalidad.*
"""

        if output_path:
            out_file = Path(output_path)
            out_file.parent.mkdir(parents=True, exist_ok=True)
            with open(out_file, "w", encoding="utf-8") as f:
                f.write(report_md)
            logger.info(f"Informe de dataset guardado en: {out_file}")

        return report_md

    def generate_missing_data_report(
        self,
        pipeline_result: Dict[str, Any],
        output_path: Optional[Union[str, Path]] = None,
    ) -> str:
        """
        Compila el informe de datos faltantes del pipeline E1–E6 en Markdown.

        Args:
            pipeline_result: Resultado serializado de MissingDataPipeline
                (MissingDataPipelineResult.to_dict()).
            output_path: Ruta opcional para guardar el informe.

        Returns:
            String en Markdown con el informe estructurado.
        """
        logger.info("Generando informe de datos faltantes.")

        status = pipeline_result.get("status", "desconocido")
        continued = pipeline_result.get("continued")
        detection = pipeline_result.get("detection_report") or {}
        diagnostics = pipeline_result.get("diagnostics_report") or {}
        candidates = pipeline_result.get("candidate_methods") or []
        selection = pipeline_result.get("selection_report") or {}
        validation = pipeline_result.get("validation_report") or {}
        verdict = pipeline_result.get("validation_verdict")
        applied = pipeline_result.get("applied_methods") or {}
        skipped = pipeline_result.get("skipped_variables") or {}

        mechanism = (diagnostics.get("mechanism") or {}).get("mechanism", "No evaluado")

        vars_missing = detection.get("variables_with_missing") or []
        missing_lines = "\n".join(
            [f"- `{v}`" for v in vars_missing]
        ) or "- Ninguna"

        overall_pct = detection.get("overall_missing_percentage")
        try:
            overall_pct_str = (
                f"{float(overall_pct):.2f}%" if overall_pct is not None else "N/A"
            )
        except (TypeError, ValueError):
            overall_pct_str = "N/A"

        applied_lines = "\n".join(
            f"- **{var}**: `{info.get('method')}` (score={info.get('score')})\n"
            f"  - Razones: {'; '.join(info.get('reasons') or [])}"
            for var, info in applied.items()
        ) or "- Ninguna variable imputada"

        skipped_lines = "\n".join(
            f"- **{var}**: {reason}" for var, reason in skipped.items()
        ) or "- Ninguna variable omitida"

        report_md = f"""# INFORME DE DATOS FALTANTES — PIPELINE PAPERSTATS

**Estado del pipeline**: `{status}`  
**Continúa hacia análisis**: `{continued}`  
**Mecanismo de ausencia (E2)**: {mechanism}

---

## 1. DETECCIÓN (E1)

- **Valores faltantes totales**: {detection.get('total_missing_values', 0)}
- **Porcentaje global**: {overall_pct_str}
- **Casos completos**: {detection.get('complete_cases', 0)}
- **Grado global**: {detection.get('overall_missing_grade', 'N/A')}

### Variables con faltantes
{missing_lines}

---

## 2. CANDIDATOS (E3)

- **Métodos candidatos**: {', '.join(candidates) if candidates else 'N/A'}

---

## 3. SELECCIÓN E IMPUTACIÓN (E5)

### Métodos aplicados
{applied_lines}

### Variables omitidas
{skipped_lines}

---

## 4. VALIDACIÓN (E6)

- **Veredicto**: `{verdict if verdict else 'No ejecutada'}`
- **Celdas imputadas**: {pipeline_result.get('n_imputed_cells', 0)}
- **Advertencias**: {len(validation.get('warnings') or [])}

> [!IMPORTANT]
> *Si la validación devuelve "Revisar", la imputación no es aceptable y el pipeline no continúa silenciosamente hacia EDA/inferencia.*
"""

        if output_path:
            out_file = Path(output_path)
            out_file.parent.mkdir(parents=True, exist_ok=True)
            with open(out_file, "w", encoding="utf-8") as f:
                f.write(report_md)
            logger.info(f"Informe de datos faltantes guardado en: {out_file}")

        return report_md
