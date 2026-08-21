"""
Pipeline de integración E1–E6 para datos faltantes.

Orquestador del flujo:

    LOAD → E1 DETECTION
      ├── sin faltantes → continúa hacia EDA / inferencia / modelos
      └── con faltantes → E2 diagnóstico → E3 candidatos → E5 recomendación
            ├── impute=False → detecta, diagnostica y recomienda el método
            │                  sin imputar (la imputación queda opt-in)
            └── impute=True  → E4 evaluación → E5 selección → imputación
                               explícita → E6 validación
                                    ├── "Aceptable" → continúa
                                    └── "Revisar" → detiene (o lanza error)

La imputación es siempre opt-in: el comportamiento predeterminado
(`impute=False`) no modifica los datos ni imputa automáticamente. El
DataFrame original nunca se muta: todas las operaciones operan sobre copias.

Este módulo únicamente orquesta los módulos E1–E6 sin duplicar su lógica
estadística interna.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.analysis.profile import build_dataset_profile
from src.missing_data.detection import MissingDataDetector, MissingReport
from src.missing_data.diagnostics import (
    MissingDataDiagnostics,
    MissingnessDiagnosticsReport,
)
from src.missing_data.evaluation import (
    ArtificialMissingnessEvaluator,
    ImputationEvaluationReport,
)
from src.missing_data.registry import candidates_for, get, names
from src.missing_data.selector import ImputationSelectionReport, ImputationSelector
from src.missing_data.validation import (
    ImputationValidationReport,
    ImputationValidator,
)
from src.utils.logger import setup_logger

logger = setup_logger("MissingDataPipeline")

PIPELINE_VERSION = "1.0"

# Umbral mínimo de casos completos para ejecutar E4 (evaluación artificial).
# Por debajo de este umbral la evidencia empírica sería especulativa.
MIN_COMPLETE_CASES_FOR_EVALUATION = 5


class ValidationRevisionError(RuntimeError):
    """
    Error lanzado cuando la validación E6 devuelve 'Revisar' y `strict=True`.

    Evita que el pipeline continúe silenciosamente hacia EDA/inferencia con
    una imputación no aceptable.
    """


@dataclass
class MissingDataPipelineResult:
    """Resultado estructurado del pipeline con todos los reportes E1–E6."""

    status: str
    continued: bool
    detection_report: Optional[MissingReport]
    diagnostics_report: Optional[MissingnessDiagnosticsReport]
    candidate_methods: List[str]
    evaluation_report: Optional[ImputationEvaluationReport]
    selection_report: Optional[ImputationSelectionReport]
    validation_report: Optional[ImputationValidationReport]
    validation_verdict: Optional[str]
    applied_methods: Dict[str, Dict[str, Any]]
    skipped_variables: Dict[str, str]
    imputed_df: Optional[pd.DataFrame]
    n_imputed_cells: int
    reproduction: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """
        Devuelve el resultado como diccionario JSON-serializable.

        El DataFrame imputado (`imputed_df`) se excluye de la serialización
        por ser un objeto no JSON; está disponible como atributo.
        """
        return {
            "status": self.status,
            "continued": self.continued,
            "detection_report": (
                self.detection_report.to_dict() if self.detection_report else None
            ),
            "diagnostics_report": (
                self.diagnostics_report.to_dict() if self.diagnostics_report else None
            ),
            "candidate_methods": list(self.candidate_methods),
            "evaluation_report": (
                self.evaluation_report.to_dict() if self.evaluation_report else None
            ),
            "selection_report": (
                self.selection_report.to_dict() if self.selection_report else None
            ),
            "validation_report": (
                self.validation_report.to_dict() if self.validation_report else None
            ),
            "validation_verdict": self.validation_verdict,
            "applied_methods": self.applied_methods,
            "skipped_variables": self.skipped_variables,
            "n_imputed_cells": self.n_imputed_cells,
            "reproduction": self.reproduction,
        }


class MissingDataPipeline:
    """
    Orquestador del flujo E1–E6 para datos faltantes.
    """

    def __init__(
        self,
        *,
        detector: Optional[MissingDataDetector] = None,
        diagnostics: Optional[MissingDataDiagnostics] = None,
        evaluator: Optional[ArtificialMissingnessEvaluator] = None,
        selector: Optional[ImputationSelector] = None,
        validator: Optional[ImputationValidator] = None,
        random_state: int = 42,
    ) -> None:
        self.random_state = random_state
        self.detector = detector or MissingDataDetector()
        self.diagnostics = diagnostics or MissingDataDiagnostics()
        self.evaluator = evaluator or ArtificialMissingnessEvaluator(
            random_state=random_state
        )
        self.selector = selector or ImputationSelector()
        self.validator = validator or ImputationValidator()

    def run(
        self,
        df: pd.DataFrame,
        *,
        impute: bool = False,
        target: Optional[Any] = None,
        temporal: Optional[bool] = None,
        datetime_columns: Optional[List[Any]] = None,
        identifier_columns: Optional[List[Any]] = None,
        evaluation_fraction: float = 0.2,
        evaluation_mechanism: str = "MCAR",
        strict: bool = True,
        method_override: Optional[Dict[str, str]] = None,
    ) -> MissingDataPipelineResult:
        """
        Ejecuta el pipeline completo de datos faltantes.

        Args:
            df: DataFrame de entrada (no se modifica).
            impute: Si es True ejecuta E4, E5, imputación explícita y E6.
                Por defecto False: solo detecta (E1), diagnostica (E2, E3) y
                recomienda el método de imputación (E5) sin imputar.
            target: Variable objetivo (para el perfil de E5).
            temporal: Indica estructura temporal (por defecto auto-detectado).
            datetime_columns: Columnas de tiempo.
            identifier_columns: Columnas identificadoras (excluidas de imputación).
            evaluation_fraction: Fracción ocultada en E4 (default 0.2).
            evaluation_mechanism: Mecanismo de E4 (default "MCAR").
            strict: Si True y E6 devuelve "Revisar", lanza
                ValidationRevisionError (impide continuar silenciosamente).
            method_override: Mapeo opcional variable -> método de imputación a
                aplicar en lugar de la recomendación del selector (decisión
                explícita del usuario). Solo se aplica con impute=True.

        Returns:
            MissingDataPipelineResult con todos los reportes disponibles.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Se requiere un DataFrame de pandas.")
        original = df.copy()

        if method_override:
            known = set(names())
            unknown = sorted(set(method_override.values()) - known)
            if unknown:
                raise ValueError(
                    f"Método(s) de imputación desconocido(s): {unknown}. "
                    f"Métodos disponibles: {sorted(known)}."
                )
            if not all(isinstance(v, str) for v in method_override.values()):
                raise ValueError(
                    "method_override debe mapear cada variable a un nombre de "
                    "método de imputación (str)."
                )

        reproduction = {
            "impute": bool(impute),
            "target": str(target) if target is not None else None,
            "temporal": temporal,
            "datetime_columns": [str(c) for c in (datetime_columns or [])],
            "identifier_columns": [str(c) for c in (identifier_columns or [])],
            "evaluation_fraction": float(evaluation_fraction),
            "evaluation_mechanism": str(evaluation_mechanism),
            "strict": bool(strict),
            "method_override": dict(method_override or {}),
            "random_state": self.random_state,
            "module": "src.missing_data.pipeline",
            "version": PIPELINE_VERSION,
        }

        # ---- E1 DETECTION -------------------------------------------------
        detection_report = self.detector.detect(original)
        logger.info(
            f"E1: {detection_report.status} — total_missing="
            f"{detection_report.total_missing_values}."
        )

        if not detection_report.variables_with_missing:
            logger.info("Sin datos faltantes: se continúa hacia EDA/inferencia.")
            return MissingDataPipelineResult(
                status="sin_faltantes",
                continued=True,
                detection_report=detection_report,
                diagnostics_report=None,
                candidate_methods=[],
                evaluation_report=None,
                selection_report=None,
                validation_report=None,
                validation_verdict=None,
                applied_methods={},
                skipped_variables={},
                imputed_df=None,
                n_imputed_cells=0,
                reproduction=reproduction,
            )

        # ---- E2 DIAGNÓSTICO ----------------------------------------------
        diagnostics_report = self.diagnostics.diagnose(original)
        logger.info("E2: diagnóstico del mecanismo de ausencia completado.")

        # ---- E3 CANDIDATOS ------------------------------------------------
        candidate_methods = candidates_for(original, temporal=bool(temporal))
        logger.info(f"E3: {len(candidate_methods)} métodos candidatos.")

        # ---- PERFIL ESTRUCTURAL (requerido por E5, también sin imputar) ----
        profile = build_dataset_profile(
            original,
            target=target,
            temporal=temporal,
            datetime_columns=datetime_columns,
            identifier_columns=identifier_columns,
        )

        if not impute:
            # ---- E5 RECOMENDACIÓN (sin imputar) ----------------------------
            # La recomendación del método se produce sin ejecutar la imputación:
            # la decisión de imputar siempre queda en manos del usuario.
            selection_report = self.selector.select(
                profile,
                detection_report,
                diagnostics_report,
                evaluation=None,
            )
            logger.info(
                "impute=False: se detecta, diagnostica y recomienda sin imputar."
            )
            return MissingDataPipelineResult(
                status="con_faltantes",
                continued=True,
                detection_report=detection_report,
                diagnostics_report=diagnostics_report,
                candidate_methods=candidate_methods,
                evaluation_report=None,
                selection_report=selection_report,
                validation_report=None,
                validation_verdict=None,
                applied_methods={},
                skipped_variables={},
                imputed_df=None,
                n_imputed_cells=0,
                reproduction=reproduction,
            )

        # ---- E4 EVALUACIÓN (solo sobre casos completos) --------------------
        evaluation_report = None
        complete = original.dropna()
        if len(complete) >= MIN_COMPLETE_CASES_FOR_EVALUATION and candidate_methods:
            try:
                evaluation_report = self.evaluator.evaluate(
                    complete,
                    methods=list(candidate_methods),
                    fraction=evaluation_fraction,
                    mechanism=evaluation_mechanism,
                )
                logger.info(
                    "E4: evaluación artificial completada sobre "
                    f"{len(complete)} casos completos."
                )
            except (ValueError, TypeError, KeyError, np.linalg.LinAlgError) as exc:
                logger.warning(f"E4: evaluación artificial no disponible: {exc}")
                evaluation_report = None
        else:
            logger.info(
                "E4: omitida (casos completos insuficientes para una evaluación "
                "empírica estable)."
            )

        # ---- E5 SELECCIÓN (con evidencia E4) ------------------------------
        selection_report = self.selector.select(
            profile,
            detection_report,
            diagnostics_report,
            evaluation=evaluation_report,
        )
        logger.info("E5: selección de métodos completada.")

        # ---- IMPUTACIÓN EXPLÍCITA (variable a variable) --------------------
        imputed = original.copy()
        applied_methods: Dict[str, Dict[str, Any]] = {}
        skipped_variables: Dict[str, str] = {}

        for var in selection_report.variables:
            rec = selection_report.variables[var]
            recommended = rec.recommended

            method_name = None
            if method_override and str(var) in method_override:
                # Decisión explícita del usuario: prevalece sobre la recomendación.
                method_name = method_override[str(var)]
            elif recommended is not None and not recommended.excluded:
                method_name = recommended.method

            if method_name is None:
                reason = "sin método recomendado (todas las alternativas excluidas)."
                if recommended is not None and recommended.exclusion_reason:
                    reason = recommended.exclusion_reason
                skipped_variables[str(var)] = reason
                continue
            try:
                method = get(method_name)(random_state=self.random_state)
                imputed[var] = method.impute(imputed)[var]
                applied_methods[str(var)] = {
                    "method": method_name,
                    "score": recommended.score if recommended is not None else None,
                    "reasons": list(recommended.reasons) if recommended is not None else [],
                    "caveats": list(recommended.caveats) if recommended is not None else [],
                    "components": recommended.components if recommended is not None else {},
                }
            except (ValueError, TypeError, KeyError, np.linalg.LinAlgError) as exc:
                skipped_variables[str(var)] = (
                    f"fallo al aplicar '{method_name}': {exc}"
                )
                logger.warning(f"Imputación fallida para '{var}' ({method_name}): {exc}")

        # ---- E6 VALIDACIÓN -------------------------------------------------
        validation_report = self.validator.validate(original, imputed)
        validation_verdict = validation_report.verdict
        n_imputed_cells = int((original.isna() & imputed.notna()).sum().sum())
        logger.info(
            f"E6: validación = {validation_verdict} "
            f"({n_imputed_cells} celdas imputadas)."
        )

        result = MissingDataPipelineResult(
            status="imputado",
            continued=(validation_verdict == "Aceptable"),
            detection_report=detection_report,
            diagnostics_report=diagnostics_report,
            candidate_methods=candidate_methods,
            evaluation_report=evaluation_report,
            selection_report=selection_report,
            validation_report=validation_report,
            validation_verdict=validation_verdict,
            applied_methods=applied_methods,
            skipped_variables=skipped_variables,
            imputed_df=imputed,
            n_imputed_cells=n_imputed_cells,
            reproduction=reproduction,
        )

        if validation_verdict == "Revisar":
            logger.warning(
                "E6 devolvió 'Revisar': la imputación no es aceptable y el "
                "pipeline no continúa de forma silenciosa."
            )
            if strict:
                raise ValidationRevisionError(
                    "La validación E6 devolvió 'Revisar' (faltantes residuales o "
                    "valores imposibles). Revise la imputación antes de continuar "
                    "hacia EDA/inferencia/modelos."
                )

        return result


def run_pipeline(
    df: pd.DataFrame,
    *,
    impute: bool = False,
    **kwargs: Any,
) -> MissingDataPipelineResult:
    """Ejecuta el pipeline con la configuración por defecto."""
    return MissingDataPipeline().run(df, impute=impute, **kwargs)
