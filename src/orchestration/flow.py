"""
Orquestador de alto nivel del flujo principal de PaperStats (P-FLOW).

Flujo objetivo:

    DATASET
      → diagnose()   perfilamiento/QC + detección de faltantes (E1)
                     + diagnóstico del mecanismo (E2) + candidatos (E3)
                     + recomendación del método de imputación (E5)
      → [ESPERAR DECISIÓN DEL USUARIO]    (si hay datos faltantes)
      → imputar()    imputación explícita (E4–E5–E6) con decisión del usuario
      → analizar()   EDA/inferencia/modelamiento bajo demanda, sobre datos
                     preparados (delega en DatasetStatisticalAnalyzer)

Reglas:
  - `diagnose()` es estrictamente diagnóstica: NO ejecuta EDA, pruebas
    inferenciales ni recomendaciones de análisis sobre el dataset incompleto.
  - La imputación nunca se ejecuta en silencio: requiere una decisión explícita
    del usuario (`method_override` o `accept_recommendation=True`).
  - `DatasetStatisticalAnalyzer` no se modifica; se reutiliza tal cual.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Union

import pandas as pd

from src.analysis.dataset_analyzer import DatasetStatisticalAnalyzer
from src.analysis.eda import (
    calculate_correlation_matrix,
    describe_categorical,
    describe_numerical,
    detect_outliers_iqr,
)
from src.analysis.profile import build_dataset_profile
from src.data.loader import load_data
from src.data.validator import DataValidator
from src.deliverables.generator import Deliverable, DeliverableGenerator
from src.missing_data.pipeline import MissingDataPipeline
from src.missing_data.selector import SCORE_GAP_SENSITIVITY_THRESHOLD
from src.reports.generator import ReportGenerator
from src.utils.logger import setup_logger

logger = setup_logger("PaperStatsFlow")

# --- Estados del flujo -------------------------------------------------------
ESTADO_SIN_DIAGNOSTICO = "sin_diagnostico"
ESTADO_SIN_FALTANTES = "sin_faltantes"
ESTADO_ESPERANDO_DECISION = "esperando_decision"
ESTADO_DATOS_PREPARADOS = "datos_preparados"
ESTADO_REVISAR = "revisar"

# --- Decisiones de recomendación de imputación -------------------------------
DECISION_METODO_UNICO = "metodo_unico"
DECISION_COMPARAR_ALTERNATIVAS = "comparar_alternativas"
DECISION_SIN_RECOMENDACION = "sin_recomendacion"

_ANALIZABLES = (ESTADO_SIN_FALTANTES, ESTADO_DATOS_PREPARADOS)

_VERSION = "1.0"


class PaperStatsFlow:
    """Orquestador del flujo principal de PaperStats."""

    def __init__(self, data: Union[str, Path, pd.DataFrame]) -> None:
        if isinstance(data, pd.DataFrame):
            self.df = data.copy()
            self.file_name = "DataFrame_en_memoria"
        else:
            self.df = load_data(data)
            self.file_name = Path(data).name
        self._state = ESTADO_SIN_DIAGNOSTICO
        self._pipeline_result = None
        self._missing_result: Optional[Dict[str, Any]] = None
        self._imputed_df: Optional[pd.DataFrame] = None
        self._last_diagnose: Optional[Dict[str, Any]] = None
        self._last_deliverable: Optional[Deliverable] = None

    # ------------------------------------------------------------------ API
    @property
    def state(self) -> str:
        """Estado actual del flujo (máquina de estados simple)."""
        return self._state

    @property
    def imputed_df(self) -> Optional[pd.DataFrame]:
        """DataFrame imputado tras `imputar()`, o None si aún no se imputó."""
        return self._imputed_df

    def diagnose(
        self,
        *,
        target: Optional[Any] = None,
        temporal: Optional[bool] = None,
        datetime_columns: Optional[list] = None,
        identifier_columns: Optional[list] = None,
    ) -> Dict[str, Any]:
        """
        ETAPA 1: diagnóstico del dataset. Estrictamente diagnóstica.

        Perfila los datos, reporta la calidad, detecta y diagnostica los datos
        faltantes (E1–E2–E3) y recomienda el método de imputación (E5) SIN
        imputar ni ejecutar ninguna prueba inferencial o EDA.

        Devuelve un dict con `estado` = "sin_faltantes" | "esperando_decision".
        """
        variable_types = DataValidator.identify_variable_types(self.df)
        calidad = DataValidator.summarize_data_quality(self.df)
        perfil = build_dataset_profile(
            self.df,
            target=target,
            temporal=temporal,
            datetime_columns=datetime_columns,
            identifier_columns=identifier_columns,
        )

        pipeline_result = MissingDataPipeline().run(
            self.df,
            impute=False,
            target=target,
            temporal=temporal,
            datetime_columns=datetime_columns,
            identifier_columns=identifier_columns,
        )
        self._pipeline_result = pipeline_result
        self._missing_result = pipeline_result.to_dict()
        self._imputed_df = None

        if pipeline_result.status == "con_faltantes":
            self._state = ESTADO_ESPERANDO_DECISION
        else:
            self._state = ESTADO_SIN_FALTANTES

        recomendacion = self._sintetizar_recomendacion(pipeline_result)

        mensaje = (
            "El dataset presenta datos faltantes. PaperStats se detiene aquí: "
            "revise el diagnóstico del mecanismo y la recomendación de "
            "imputación, y decida cómo imputar (method_override o "
            "accept_recommendation) antes de continuar."
            if self._state == ESTADO_ESPERANDO_DECISION
            else "Sin datos faltantes: los datos están listos para análisis "
            "bajo demanda (analizar())."
        )

        result = {
            "estado": self._state,
            "dataset": {
                "file_name": self.file_name,
                "rows": int(self.df.shape[0]),
                "columns": int(self.df.shape[1]),
                "target_variable": target,
            },
            "perfil": perfil.to_dict(),
            "calidad": calidad,
            "clasificacion_variables": variable_types,
            "missing_data": self._missing_result,
            "recomendacion_imputacion": recomendacion,
            "mensaje_estado": mensaje,
        }
        self._last_diagnose = result
        logger.info(
            f"diagnose() completado: estado={self._state}, "
            f"{self.df.shape[0]} filas x {self.df.shape[1]} columnas."
        )
        return result

    def generar_informe_missing(self, output_path: Optional[Union[str, Path]] = None) -> str:
        """Genera el informe Markdown de datos faltantes (E1–E3 + E5)."""
        if self._missing_result is None:
            raise ValueError("Debe ejecutar diagnose() antes de generar el informe.")
        return ReportGenerator().generate_missing_data_report(
            self._missing_result, output_path
        )

    def imputar(
        self,
        *,
        method_override: Optional[Dict[str, str]] = None,
        accept_recommendation: bool = False,
        target: Optional[Any] = None,
        temporal: Optional[bool] = None,
        evaluation_fraction: float = 0.2,
        evaluation_mechanism: str = "MCAR",
        strict: bool = True,
    ) -> Dict[str, Any]:
        """
        ETAPA 2: imputación explícita (E4–E5–E6).

        Requiere una decisión explícita del usuario:
          - `method_override` (variable -> método), o
          - `accept_recommendation=True` (aceptar las recomendaciones del
            selector).

        La imputación nunca se ejecuta en silencio.
        """
        if self._state == ESTADO_SIN_DIAGNOSTICO:
            raise ValueError("Debe ejecutar diagnose() antes de imputar.")

        if self._state == ESTADO_SIN_FALTANTES:
            self._imputed_df = self.df.copy()
            self._state = ESTADO_DATOS_PREPARADOS
            return {
                "estado": self._state,
                "mensaje": "El dataset no tenía datos faltantes; no se imputó nada.",
                "validation_verdict": None,
                "n_imputed_cells": 0,
                "applied_methods": {},
                "skipped_variables": {},
                "imputed_df": self._imputed_df,
                "missing_data": self._missing_result,
            }

        if self._state == ESTADO_DATOS_PREPARADOS:
            raise ValueError(
                "Los datos ya están preparados; ejecute analizar()."
            )

        if not method_override and not accept_recommendation:
            raise ValueError(
                "La imputación requiere una decisión explícita del usuario: "
                "proporcione `method_override` (variable -> método) o "
                "`accept_recommendation=True` para aceptar las recomendaciones "
                "del selector."
            )

        pipeline_result = MissingDataPipeline().run(
            self.df,
            impute=True,
            method_override=method_override,
            target=target,
            temporal=temporal,
            evaluation_fraction=evaluation_fraction,
            evaluation_mechanism=evaluation_mechanism,
            strict=strict,
        )
        self._pipeline_result = pipeline_result
        self._missing_result = pipeline_result.to_dict()
        self._imputed_df = pipeline_result.imputed_df

        if pipeline_result.validation_verdict == "Aceptable":
            self._state = ESTADO_DATOS_PREPARADOS
            mensaje = "Imputación validada (E6 = Aceptable): datos preparados."
        else:
            self._state = ESTADO_REVISAR
            mensaje = (
                "La validación E6 devolvió 'Revisar': revise la imputación antes "
                "de continuar."
            )
        logger.info(
            f"imputar() completado: estado={self._state}, "
            f"{pipeline_result.n_imputed_cells} celdas imputadas."
        )

        return {
            "estado": self._state,
            "mensaje": mensaje,
            "validation_verdict": pipeline_result.validation_verdict,
            "n_imputed_cells": pipeline_result.n_imputed_cells,
            "applied_methods": pipeline_result.applied_methods,
            "skipped_variables": pipeline_result.skipped_variables,
            "imputed_df": pipeline_result.imputed_df,
            "missing_data": self._missing_result,
        }

    def analizar(self, **kwargs: Any) -> Dict[str, Any]:
        """
        ETAPA 3: análisis bajo demanda sobre datos preparados.

        Delega en `DatasetStatisticalAnalyzer.analyze()` (no se modifica).
        Solo se permite cuando los datos están listos ("sin_faltantes" o
        "datos_preparados").
        """
        if self._state == ESTADO_SIN_DIAGNOSTICO:
            raise ValueError("Debe ejecutar diagnose() antes de analizar.")
        if self._state == ESTADO_ESPERANDO_DECISION:
            raise ValueError(
                "Los datos tienen faltantes: ejecute imputar() antes de "
                "solicitar análisis estadístico."
            )
        if self._state == ESTADO_REVISAR:
            raise ValueError(
                "La imputación fue marcada como 'Revisar': resuelva la "
                "validación antes de analizar."
            )

        df = self._imputed_df if self._state == ESTADO_DATOS_PREPARADOS else self.df
        logger.info(f"analizar(): análisis bajo demanda sobre '{self.file_name}'.")
        return DatasetStatisticalAnalyzer(df).analyze(**kwargs)

    # ------------------------------------------- entregables de usuario
    def entregable_inicial(
        self, eda_results: Optional[Dict[str, Any]] = None
    ) -> Deliverable:
        """
        Entregable de usuario de la etapa inicial.

        Método de PRESENTACIÓN: orquesta los motores (EDA cuando los datos no
        tienen faltantes) y delega la construcción del entregable en la capa
        `deliverables`. No altera la máquina de estados ni la lógica
        estadística existente.

        Con datos faltantes, delega en el entregable de datos faltantes y se
        detiene esperando la decisión del usuario.
        """
        if self._last_diagnose is None:
            self.diagnose()
        if self._state == ESTADO_ESPERANDO_DECISION:
            entregable = DeliverableGenerator().build_missing(self._last_diagnose)
        elif self._state == ESTADO_SIN_FALTANTES:
            if eda_results is None:
                eda_results = self._resultados_eda()
            entregable = DeliverableGenerator().build_inicial(
                self._last_diagnose, eda_results
            )
        else:
            raise ValueError(
                "No se puede generar el entregable inicial en el estado "
                f"actual: '{self._state}'."
            )
        self._last_deliverable = entregable
        return entregable

    def entregable_missing(self) -> Deliverable:
        """Entregable de usuario de datos faltantes (requiere diagnose())."""
        if self._last_diagnose is None:
            raise ValueError(
                "Debe ejecutar diagnose() antes de generar el entregable de "
                "datos faltantes."
            )
        entregable = DeliverableGenerator().build_missing(self._last_diagnose)
        self._last_deliverable = entregable
        return entregable

    def entregable_analisis(
        self, pregunta: str, resultado: Dict[str, Any]
    ) -> Deliverable:
        """
        Entregable de usuario de un análisis estadístico solicitado.

        `resultado` es el dict producido por el motor tras ejecutar el análisis
        pedido por el usuario; la capa de presentación solo lo traduce.
        """
        entregable = DeliverableGenerator().build_analisis(pregunta, resultado)
        self._last_deliverable = entregable
        return entregable

    def informe(
        self,
        output_path: Optional[Union[str, Path]] = None,
        formato: str = "pdf",
    ) -> Any:
        """
        Orden de PRESENTACIÓN/EXPORTACIÓN "Informe".

        Genera la representación solicitada (PDF por ahora) del ÚLTIMO
        `Deliverable` generado, sin recalcular estadísticas, sin seleccionar
        métodos ni ejecutar nuevos análisis. No altera la máquina de estados.

        Args:
            output_path: Ruta del archivo PDF. Si no se indica, se usa
                `outputs/reports/<dataset>_informe_<marca>.pdf`.
            formato: "pdf" (único implementado; "markdown"/"html" futuros).

        Returns:
            Ruta del archivo generado (formato "pdf").
        """
        if self._last_deliverable is None:
            raise ValueError(
                "No hay un entregable disponible para generar el informe: "
                "ejecute primero entregable_inicial(), entregable_missing() o "
                "un análisis (entregable_analisis) antes de la orden 'Informe'."
            )
        from src.deliverables.renderers import render as _render

        if formato == "pdf":
            if output_path is None:
                output_path = self._ruta_informe_default()
            _render(self._last_deliverable, formato="pdf", output_path=output_path)
            ruta = str(Path(output_path))
            logger.info(f"informe(): PDF generado en '{ruta}'.")
            return ruta
        return _render(self._last_deliverable, formato=formato)

    def _ruta_informe_default(self) -> Path:
        """Ruta por defecto del informe PDF en `outputs/reports/`."""
        from datetime import datetime

        nombre = Path(self.file_name).stem or "dataset"
        marca = datetime.now().strftime("%Y%m%d_%H%M%S")
        ruta = Path("outputs") / "reports" / f"{nombre}_informe_{marca}.pdf"
        ruta.parent.mkdir(parents=True, exist_ok=True)
        return ruta

    def _resultados_eda(self) -> Dict[str, Any]:
        """
        Orquesta los motores de EDA y entrega sus resultados técnicos.

        Es orquestación (no presentación): los cálculos los realizan los
        motores de `src/analysis/eda`; la capa `deliverables` solo los presenta.
        """
        numericas = self.df.select_dtypes(include=["number"]).columns.tolist()
        categoricas = [c for c in self.df.columns if c not in numericas]
        return {
            "numerical": describe_numerical(self.df),
            "categorical": describe_categorical(self.df),
            "outliers": detect_outliers_iqr(self.df),
            "correlation": calculate_correlation_matrix(self.df),
            "numerical_columns": numericas,
            "categorical_columns": categoricas,
        }

    # ----------------------------------------------------- métodos internos
    def _sintetizar_recomendacion(self, pipeline_result: Any) -> Dict[str, Any]:
        """
        Sintetiza la recomendación de imputación a partir del selection_report
        de E5 (fuente única de verdad en el selector). No recalcula puntajes.
        """
        sel = getattr(pipeline_result, "selection_report", None)
        if pipeline_result.status != "con_faltantes" or sel is None:
            return {
                "pesos": {},
                "por_variable": {},
                "advertencias_globales": [],
                "resumen": "Sin datos faltantes: no hay recomendación de imputación.",
            }

        por_variable = {}
        for var, vr in sel.variables.items():
            rec = vr.recommended
            considerados = [
                {
                    "method": s.method,
                    "score": s.score,
                    "components": dict(s.components),
                    "reasons": list(s.reasons),
                    "caveats": list(s.caveats),
                    "e4_evidence": s.e4_evidence,
                    "excluded": bool(s.excluded),
                    "exclusion_reason": s.exclusion_reason,
                }
                for s in vr.all_scores
            ]

            if rec is None:
                decision = DECISION_SIN_RECOMENDACION
                metodo = None
                score = None
                gap = None
                justificacion = (
                    "No hay método no excluido para esta variable: "
                    + ("; ".join(vr.warnings) if vr.warnings else "todas las "
                       "alternativas fueron excluidas.")
                )
            else:
                metodo = rec.method
                score = rec.score
                alt = vr.alternatives[0] if vr.alternatives else None
                gap = None
                if (
                    alt is not None
                    and rec.score is not None
                    and alt.score is not None
                ):
                    gap = round(float(rec.score - alt.score), 4)

                if (
                    alt is not None
                    and gap is not None
                    and gap < SCORE_GAP_SENSITIVITY_THRESHOLD
                ):
                    decision = DECISION_COMPARAR_ALTERNATIVAS
                    justificacion = (
                        f"La brecha entre '{rec.method}' (score={rec.score:.4f}) "
                        f"y '{alt.method}' (score={alt.score:.4f}) es de {gap:.4f}, "
                        f"menor al umbral de sensibilidad "
                        f"({SCORE_GAP_SENSITIVITY_THRESHOLD:.2f}): no hay evidencia "
                        "suficiente para un único método. Considere comparar ambos "
                        "métodos (p. ej. mediante E4) antes de imputar."
                    )
                else:
                    decision = DECISION_METODO_UNICO
                    if alt is not None and gap is not None:
                        justificacion = (
                            f"Se recomienda '{rec.method}' (score={rec.score:.4f}) "
                            f"con una brecha de {gap:.4f} sobre '{alt.method}' "
                            f"(score={alt.score:.4f}). Motivos: "
                            + "; ".join(list(rec.reasons)[:3])
                        )
                    else:
                        justificacion = (
                            f"Se recomienda '{rec.method}' (score={rec.score:.4f}). "
                            "Motivos: " + "; ".join(list(rec.reasons)[:3])
                        )

            por_variable[str(var)] = {
                "variable_type": vr.variable_type,
                "missing_count": vr.missing_count,
                "missing_percentage": vr.missing_percentage,
                "decision": decision,
                "metodo_recomendado": metodo,
                "score_recomendado": score,
                "metodo_alternativo": (
                    alt.method if decision == DECISION_COMPARAR_ALTERNATIVAS and alt is not None else None
                ),
                "brecha_puntaje": gap,
                "metodos_considerados": considerados,
                "advertencias": list(vr.warnings),
                "justificacion": justificacion,
            }

        n_unico = sum(
            1
            for v in por_variable.values()
            if v["decision"] == DECISION_METODO_UNICO
        )
        return {
            "pesos": dict(sel.weights),
            "por_variable": por_variable,
            "advertencias_globales": list(sel.warnings),
            "resumen": (
                f"{len(por_variable)} variable(s) con datos faltantes; "
                f"{n_unico} con método único recomendado."
            ),
        }