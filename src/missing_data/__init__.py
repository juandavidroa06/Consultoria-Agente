"""
Módulo de detección y manejo de datos faltantes.

Etapa E1: detección robusta de valores faltantes y placeholders candidatos.
Etapa E2: diagnóstico estadístico del mecanismo de ausencia (MCAR/MAR/MNAR).
Etapa E3: registro y métodos de imputación.
Etapa E4: evaluación artificial de métodos de imputación.
Etapa E5: selección estadística de métodos de imputación.
Etapa E6: validación post-imputación.
Etapa E7: integración del pipeline E1–E6.
"""

from .detection import (
    MissingDataDetector,
    MissingReport,
    MissingVariableInfo,
    convert_placeholders_to_na,
)
from .diagnostics import (
    MechanismAssessment,
    MissingDataDiagnostics,
    MissingnessAssociation,
    MissingnessDiagnosticsReport,
)
from .evaluation import (
    ArtificialMissingnessEvaluator,
    ImputationEvaluationReport,
    MethodEvaluation,
    induce_missing,
)
from .methods import (
    ConstantImputation,
    ImputationMethod,
    IterativeImputation,
    KNNImputation,
    LinearInterpolationImputation,
    LOCFImputation,
    MICEImputation,
    MeanImputation,
    MedianImputation,
    MethodCapabilities,
    ModeImputation,
    RegressionImputation,
)
from .registry import (
    ImputationRegistry,
    default_registry,
    get,
    candidates_for,
    names,
    register,
    summary,
)
from .selector import (
    DEFAULT_WEIGHTS,
    ImputationSelectionReport,
    ImputationSelector,
    MethodScore,
    VariableRecommendation,
)
from .validation import (
    ImputationValidationReport,
    ImputationValidator,
    ValidationCheck,
)
from .pipeline import (
    MIN_COMPLETE_CASES_FOR_EVALUATION,
    MissingDataPipeline,
    MissingDataPipelineResult,
    ValidationRevisionError,
    run_pipeline,
)

__all__ = [
    "MissingDataDetector",
    "MissingReport",
    "MissingVariableInfo",
    "convert_placeholders_to_na",
    "MissingDataDiagnostics",
    "MissingnessDiagnosticsReport",
    "MissingnessAssociation",
    "MechanismAssessment",
    "ImputationMethod",
    "MethodCapabilities",
    "MeanImputation",
    "MedianImputation",
    "ModeImputation",
    "ConstantImputation",
    "KNNImputation",
    "IterativeImputation",
    "MICEImputation",
    "RegressionImputation",
    "LinearInterpolationImputation",
    "LOCFImputation",
    "ImputationRegistry",
    "default_registry",
    "register",
    "get",
    "names",
    "candidates_for",
    "summary",
    "induce_missing",
    "ArtificialMissingnessEvaluator",
    "ImputationEvaluationReport",
    "MethodEvaluation",
    "ImputationSelector",
    "ImputationSelectionReport",
    "VariableRecommendation",
    "MethodScore",
    "DEFAULT_WEIGHTS",
    "ImputationValidator",
    "ImputationValidationReport",
    "ValidationCheck",
    "MissingDataPipeline",
    "MissingDataPipelineResult",
    "ValidationRevisionError",
    "MIN_COMPLETE_CASES_FOR_EVALUATION",
    "run_pipeline",
]