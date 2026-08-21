"""
Selección estadística de métodos de imputación (Etapa E5).

El selector evalúa los métodos candidatos de E3 para cada variable con datos
faltantes y produce una recomendación reproducible, explicable y por grupo.

Criterios del puntaje (pesos configurables, suma = 1):

  type_fit                  (0.20) compatibilidad del método con el tipo de variable.
  missing_pct_fit           (0.20) idoneidad del método según el % de faltantes.
  sample_size_fit           (0.15) idoneidad según el tamaño muestral.
  structure_fit             (0.10) compatibilidad con la estructura de los datos
                                  (orden temporal).
  relationship_exploitation (0.15) capacidad de explotar relaciones entre variables,
                                  incluida la evidencia de asociación de E2.
  robustness                (0.15) robustez a priori del método ajustada por el
                                  mecanismo de ausencia (E2) y, si existe, por la
                                  evidencia empírica de E4.
  complexity_cost           (0.05) costo computacional (mayor puntaje = más barato).

Puertas duras (excluyen al método, sin puntaje):
  1. El método no soporta el tipo de la variable (type_fit = 0).
  2. El método requiere orden temporal y los datos no son temporales
     (structure_fit = 0).
  3. El método requiere otras variables predictoras y el conjunto solo tiene una
     variable (needs_other_columns sin predictores).
  4. La variable es identificadora (una por fila) o de tiempo: no se imputa.

El puntaje final es la suma ponderada de los componentes en [0, 1]; con pesos
que suman 1, el puntaje queda en [0, 1].

Integración explícita de E4: cuando se proporciona un ImputationEvaluationReport,
la componente `robustness` se calcula como un promedio ponderado 50 % del perfil
a priori del método y 50 % de la evidencia empírica de E4 (1 - RMSE normalizado
para numéricas; accuracy para categóricas). La evidencia de E4 utilizada queda
registrada en `MethodScore.e4_evidence` y en las razones/caveats.

Cada componente devuelve una razón textual que documenta la fórmula o el umbral
utilizado. No se ocultan decisiones estadísticas dentro de heurísticas no
documentadas.

Puerta estructural de 'regresion': este método requiere al menos una columna
numérica completamente observada (missing_count == 0) para usarse como
predictora. Si ninguna existe en el conjunto (todas las columnas numéricas
tienen faltantes), 'regresion' queda excluida con `excluded=True` y un
`exclusion_reason` explícito. La condición es estrictamente missing_count == 0:
una variable con pocos faltantes NO cuenta como predictora completa.

Caveats documentados:
  - La evidencia de E4 es empírica y su fiabilidad depende del tamaño de la
    muestra y del número de repeticiones. Si E4 se ejecutó sobre una muestra
    pequeña (por ejemplo, únicamente los casos completos), dicha evidencia es
    exploratoria y el ranking debe interpretarse con cautela.
  - El ranking es sensible a los pesos cuando las diferencias de puntaje entre
    métodos son pequeñas (umbral documentado SCORE_GAP_SENSITIVITY_THRESHOLD).
    Cuando la diferencia entre el mejor método y su alternativa es inferior a
    ese umbral se emite una advertencia y se recomienda considerar alternativas.
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from src.analysis.profile import (
    CATEGORICAL_TYPES,
    DATETIME_TYPE,
    DatasetProfile,
    NUMERIC_TYPES,
)
from src.missing_data.detection import MissingReport
from src.missing_data.diagnostics import (
    MissingnessAssociation,
    MissingnessDiagnosticsReport,
)
from src.missing_data.evaluation import ImputationEvaluationReport
from src.missing_data.methods import ImputationMethod, MethodCapabilities
from src.missing_data.registry import ImputationRegistry, default_registry
from src.utils.logger import setup_logger

logger = setup_logger("ImputationSelector")

SELECTOR_VERSION = "1.0"

DEFAULT_WEIGHTS: Dict[str, float] = {
    "type_fit": 0.20,
    "missing_pct_fit": 0.20,
    "sample_size_fit": 0.15,
    "structure_fit": 0.10,
    "relationship_exploitation": 0.15,
    "robustness": 0.15,
    "complexity_cost": 0.05,
}

_COMPONENT_NAMES = tuple(DEFAULT_WEIGHTS.keys())

HIGH_MISSINGNESS_THRESHOLD = 0.5
SMALL_SAMPLE_THRESHOLD = 30
NEEDS_OTHER_MIN_VARIABLES = 2
E4_BLEND_WEIGHT = 0.5
SCORE_GAP_SENSITIVITY_THRESHOLD = 0.02


# ---------------------------------------------------------------------------
# Perfil a priori de cada método (valores base documentados).
# ---------------------------------------------------------------------------
# robustness : robustez ante la incertidumbre/desviación del mecanismo.
# complexity : inversa del costo computacional (1 = trivial, 0 = muy costoso).
# relationship: capacidad de explotar relaciones inter-variable (0 a 1).
# needs_other: el método requiere otras variables predictoras.
_METHOD_PROFILES: Dict[str, Dict[str, float]] = {
    "media": {"robustness": 0.50, "complexity": 1.00, "relationship": 0.30, "needs_other": 0.0},
    "mediana": {"robustness": 0.55, "complexity": 1.00, "relationship": 0.30, "needs_other": 0.0},
    "moda": {"robustness": 0.50, "complexity": 1.00, "relationship": 0.30, "needs_other": 0.0},
    "constante": {"robustness": 0.35, "complexity": 1.00, "relationship": 0.25, "needs_other": 0.0},
    "knn": {"robustness": 0.65, "complexity": 0.70, "relationship": 0.60, "needs_other": 1.0},
    "iterativo": {"robustness": 0.75, "complexity": 0.50, "relationship": 0.70, "needs_other": 1.0},
    "mice": {"robustness": 0.85, "complexity": 0.40, "relationship": 0.80, "needs_other": 1.0},
    "regresion": {"robustness": 0.60, "complexity": 0.70, "relationship": 0.65, "needs_other": 1.0},
    "interpolacion_lineal": {"robustness": 0.60, "complexity": 0.90, "relationship": 0.50, "needs_other": 0.0},
    "locf": {"robustness": 0.40, "complexity": 1.00, "relationship": 0.45, "needs_other": 0.0},
}


def _profile_for(name: str, caps: MethodCapabilities) -> Dict[str, float]:
    """Perfil a priori del método; si no está catalogado se deriva de sus capacidades."""
    if name in _METHOD_PROFILES:
        return dict(_METHOD_PROFILES[name])
    return {
        "robustness": 0.60,
        "complexity": 0.60,
        "relationship": 0.50 if caps.needs_other_columns else 0.35,
        "needs_other": 1.0 if caps.needs_other_columns else 0.0,
    }


def _is_numeric_type(variable_type: str) -> bool:
    return variable_type in NUMERIC_TYPES


def _is_categorical_type(variable_type: str) -> bool:
    return variable_type in CATEGORICAL_TYPES


def _group_for(variable_type: str) -> str:
    if _is_numeric_type(variable_type):
        return "Numericas"
    if _is_categorical_type(variable_type):
        return "Categoricas"
    if variable_type == DATETIME_TYPE:
        return "Tiempo"
    return "Otras"


# ---------------------------------------------------------------------------
# Componentes del puntaje (cada uno devuelve puntaje y razón explícita).
# ---------------------------------------------------------------------------


def _type_fit(caps: MethodCapabilities, variable_type: str) -> tuple:
    if variable_type == DATETIME_TYPE:
        return 0.0, "La variable es de tiempo; los métodos de imputación E3 no la soportan."
    if _is_numeric_type(variable_type):
        if caps.supports_numeric:
            return 1.0, "El método soporta variables numéricas."
        return 0.0, "El método no soporta variables numéricas (puerta dura de tipo)."
    if _is_categorical_type(variable_type):
        if caps.supports_categorical:
            return 1.0, "El método soporta variables categóricas."
        return 0.0, "El método no soporta variables categóricas (puerta dura de tipo)."
    return 0.0, f"Tipo de variable no soportado por ningún método E3 ('{variable_type}')."


def _structure_fit(caps: MethodCapabilities, temporal: bool) -> tuple:
    if caps.temporal_only and not temporal:
        return 0.0, "Método temporal; requiere estructura temporal en los datos (puerta dura de estructura)."
    if caps.temporal_only:
        return 1.0, "El método explota el orden temporal de las observaciones."
    return 1.0, "El método no depende de la estructura temporal de los datos."


def _missing_pct_fit(prof: Dict[str, float], pct: float) -> tuple:
    simple = prof["needs_other"] == 0.0
    tier = "simple" if simple else "multivariado"
    if pct <= 0.10:
        score, detail = 1.0, "faltantes <= 10%: riesgo mínimo para todos los métodos."
    elif pct <= 0.30:
        score, detail = (1.0 if simple else 0.85), (
            "10% < faltantes <= 30%: los métodos multivariados requieren una fracción "
            "observada suficiente y pierden algo de idoneidad."
        )
    elif pct <= 0.50:
        score, detail = (0.85 if simple else 0.60), (
            "30% < faltantes <= 50%: la evidencia observada se reduce; se penaliza a los "
            "métodos que necesitan estimar modelos sobre otras variables."
        )
    elif pct <= 0.80:
        score, detail = (0.65 if simple else 0.30), (
            "50% < faltantes <= 80%: la imputación es especulativa; los métodos simples "
            "siguen siendo aplicables pero los multivariados pierden estabilidad."
        )
    else:
        score, detail = (0.45 if simple else 0.10), (
            "faltantes > 80%: casi no hay señal observada; solo los métodos simples "
            "conservan idoneidad (con alta incertidumbre)."
        )
    return score, f"Método {tier}: {detail}"


def _sample_size_fit(prof: Dict[str, float], n: int) -> tuple:
    simple = prof["needs_other"] == 0.0
    if n < 30:
        score, detail = (1.0 if simple else 0.30), (
            "n < 30: los métodos multivariados no tienen datos suficientes para estimar "
            "parámetros de forma estable."
        )
    elif n < 100:
        score, detail = (1.0 if simple else 0.70), (
            "30 <= n < 100: los métodos multivariados funcionan con precaución."
        )
    else:
        score, detail = (1.0 if simple else 0.95), "n >= 100: tamaño muestral suficiente."
    return score, f"Tamaño muestral n={n}: {detail}"


def _relationship_exploitation(
    prof: Dict[str, float],
    caps: MethodCapabilities,
    n_variables: int,
    e2_significant_for_var: bool,
) -> tuple:
    base = prof["relationship"]
    notes: List[str] = []
    if caps.needs_other_columns:
        others = max(0, n_variables - 1)
        availability = min(1.0, others / 3.0)
        base *= availability
        notes.append(
            f"disponibilidad de predictores ajustada a {others} otra(s) variable(s) "
            f"(factor {availability:.2f})."
        )
        if e2_significant_for_var:
            base = min(1.0, base + 0.15)
            notes.append("E2 detectó asociaciones significativas que involucran a la variable (+0.15).")
        else:
            notes.append("E2 no detectó asociaciones significativas para esta variable.")
    else:
        notes.append("método univariado: no explota relaciones entre variables.")
    if caps.temporal_only:
        base = min(1.0, base + 0.10)
        notes.append("explota la dependencia temporal de la serie (+0.10).")
    base = max(0.0, min(1.0, base))
    return base, "Explotación de relaciones: " + "; ".join(notes)


def _e4_evidence_for(
    method: str,
    variable: str,
    evaluation: Optional[ImputationEvaluationReport],
) -> Optional[Dict[str, Any]]:
    """Evidencia de E4 para (método, variable) o None si no hay evaluación."""
    if evaluation is None:
        return None
    if variable not in evaluation.numeric_columns and variable not in evaluation.categorical_columns:
        return {"available": False, "reason": "la variable no fue evaluada en E4"}
    method_eval = next((m for m in evaluation.methods if m.method == method), None)
    if method_eval is None:
        return {"available": False, "reason": f"el método '{method}' no fue evaluado en E4"}
    if method_eval.error:
        return {"available": True, "error": method_eval.error}
    ranked = sorted(evaluation.ranking.keys(), key=lambda k: evaluation.ranking[k])
    rank = ranked.index(method) + 1 if method in ranked else None
    return {
        "available": True,
        "global_metrics": dict(method_eval.global_metrics),
        "rank": rank,
    }


def _e4_robustness_score(
    e4: Optional[Dict[str, Any]],
    variable: str,
    evaluation: Optional[ImputationEvaluationReport],
) -> Optional[float]:
    """Puntaje de robustez derivado de E4 en [0, 1] o None sin evidencia."""
    if e4 is None or not e4.get("available") or e4.get("error"):
        return None
    if evaluation is None:
        return None
    gm = e4["global_metrics"]
    if variable in evaluation.numeric_columns:
        rmse = gm.get("rmse")
        if rmse is None:
            return None
        rmse_list = [
            m.global_metrics.get("rmse")
            for m in evaluation.methods
            if not m.error and m.global_metrics.get("rmse") is not None
        ]
        max_rmse = max(rmse_list) if rmse_list else None
        if max_rmse is None or max_rmse <= 0:
            return 1.0 if rmse == 0 else None
        return max(0.0, min(1.0, 1.0 - rmse / max_rmse))
    accuracy = gm.get("accuracy")
    if accuracy is None:
        return None
    return max(0.0, min(1.0, float(accuracy)))


def _robustness(
    prof: Dict[str, float],
    e2_not_mcar: bool,
    e4_score: Optional[float],
) -> tuple:
    base = prof["robustness"]
    notes: List[str] = []
    if e2_not_mcar:
        if prof["needs_other"] == 1.0:
            base += 0.10
            notes.append("E2 cuestiona MCAR: los métodos multivariados modelan la relación (+0.10).")
        else:
            base -= 0.20
            notes.append("E2 cuestiona MCAR: el valor único puede sesgar la distribución (-0.20).")
    if e4_score is not None:
        base = E4_BLEND_WEIGHT * base + E4_BLEND_WEIGHT * e4_score
        notes.append(
            f"robustez final = 0.5*perfil + 0.5*evidencia de E4 ({e4_score:.3f})."
        )
    base = max(0.0, min(1.0, base))
    return base, "Robustez: " + "; ".join(notes) if notes else "Robustez a priori del método."


def _complexity_cost(prof: Dict[str, float]) -> tuple:
    score = prof["complexity"]
    if score >= 0.9:
        detail = "costo computacional trivial."
    elif score >= 0.6:
        detail = "costo computacional bajo (modelos sencillos o distancia entre vecinos)."
    else:
        detail = "costo computacional alto (múltiples iteraciones o cadenas imputadas)."
    return score, f"Costo computacional: {detail}"


# ---------------------------------------------------------------------------
# Estructuras de salida
# ---------------------------------------------------------------------------


@dataclass
class MethodScore:
    """Puntaje y evidencia de un método para una variable."""

    method: str
    variable: str
    description: str
    score: Optional[float]
    components: Dict[str, float]
    reasons: List[str]
    caveats: List[str]
    e4_evidence: Optional[Dict[str, Any]]
    excluded: bool
    exclusion_reason: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        """Devuelve el puntaje como diccionario JSON-serializable."""
        return asdict(self)


@dataclass
class VariableRecommendation:
    """Recomendación por variable, con alternativas y advertencias."""

    variable: str
    variable_type: str
    missing_count: int
    missing_percentage: float
    recommended: Optional[MethodScore]
    alternatives: List[MethodScore]
    all_scores: List[MethodScore]
    warnings: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Devuelve la recomendación como diccionario JSON-serializable."""
        return asdict(self)


@dataclass
class ImputationSelectionReport:
    """Reporte estructurado y reproducible de la selección de métodos."""

    weights: Dict[str, float]
    variables: Dict[str, VariableRecommendation]
    group_ranking: Dict[str, List[Dict[str, Any]]]
    warnings: List[str]
    selector_version: str
    reproduction: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Devuelve el reporte como diccionario JSON-serializable."""
        return asdict(self)


class ImputationSelector:
    """
    Selecciona métodos de imputación por variable con puntaje reproducible.

    Requiere un DatasetProfile, un MissingReport (E1) y un
    MissingnessDiagnosticsReport (E2). Opcionalmente recibe un
    ImputationEvaluationReport (E4) cuya evidencia se integra de forma explícita
    y transparente en la componente de robustez.
    """

    def __init__(
        self,
        *,
        weights: Optional[Dict[str, float]] = None,
        alpha: float = 0.05,
        registry: Optional[ImputationRegistry] = None,
    ) -> None:
        self.weights = self._validate_weights(weights)
        if not (0.0 < alpha < 1.0):
            raise ValueError("alpha debe estar en el intervalo abierto (0, 1).")
        self.alpha = float(alpha)
        self.registry = registry if registry is not None else default_registry

    @staticmethod
    def _validate_weights(weights: Optional[Dict[str, float]]) -> Dict[str, float]:
        if weights is None:
            weights = DEFAULT_WEIGHTS
        if not isinstance(weights, dict):
            raise TypeError("weights debe ser un diccionario.")
        missing = set(_COMPONENT_NAMES) - set(weights)
        if missing:
            raise ValueError(
                f"weights debe incluir las componentes {set(_COMPONENT_NAMES)}; "
                f"faltan: {sorted(missing)}."
            )
        for name in _COMPONENT_NAMES:
            value = weights[name]
            if not isinstance(value, (int, float)) or not (0.0 <= value <= 1.0):
                raise ValueError(f"El peso '{name}' debe estar en [0, 1].")
        if sum(weights[name] for name in _COMPONENT_NAMES) <= 0:
            raise ValueError("La suma de los pesos debe ser positiva.")
        return {name: float(weights[name]) for name in _COMPONENT_NAMES}

    def select(
        self,
        profile: DatasetProfile,
        missing_report: MissingReport,
        diagnostics: MissingnessDiagnosticsReport,
        *,
        evaluation: Optional[ImputationEvaluationReport] = None,
    ) -> ImputationSelectionReport:
        """
        Selecciona el método de imputación recomendado para cada variable.

        Args:
            profile: Perfil estructural del conjunto de datos.
            missing_report: Reporte E1 de datos faltantes.
            diagnostics: Reporte E2 del mecanismo de ausencia.
            evaluation: Reporte E4 de evaluación artificial (opcional).

        Returns:
            ImputationSelectionReport con recomendaciones, ranking por grupo y
            advertencias.
        """
        self._validate_inputs(profile, missing_report, diagnostics, evaluation)

        e2_not_mcar = self._e2_questions_mcar(diagnostics)
        e2_sig_for_var = self._e2_significant_per_variable(diagnostics)

        warnings: List[str] = []
        if e2_not_mcar:
            warnings.append(
                "E2 encontró evidencia en contra del supuesto MCAR; los métodos de valor "
                "único (media, mediana, moda, constante) pueden sesgar la imputación."
            )
        if profile.n_observations < SMALL_SAMPLE_THRESHOLD:
            warnings.append(
                f"Tamaño muestral reducido (n={profile.n_observations}): la imputación "
                "multivariada es poco estable; valide cualquier recomendación con "
                "análisis de sensibilidad."
            )
        if profile.temporal:
            warnings.append(
                "El conjunto tiene estructura temporal: considere métodos temporales "
                "(interpolacion_lineal, locf) si la variable sigue un orden."
            )
        if evaluation is not None:
            warnings.append(
                "Se incorporó evidencia empírica de E4 a la robustez. Si E4 se ejecutó "
                "sobre una muestra pequeña (por ejemplo, únicamente los casos "
                "completos), esa evidencia es exploratoria y el ranking debe "
                "interpretarse con cautela."
            )

        has_complete_numeric_predictor = self._has_complete_numeric_predictor(
            profile, missing_report
        )

        variables: Dict[str, VariableRecommendation] = {}
        for var in missing_report.variables_with_missing:
            info = missing_report.by_variable[var]
            var_type = info.variable_type
            pct = info.missing_percentage / 100.0

            var_warnings: List[str] = []
            if pct > HIGH_MISSINGNESS_THRESHOLD:
                message = (
                    f"La variable '{var}' tiene {info.missing_percentage:.1f}% de "
                    "faltantes (>50%): la imputación es especulativa y puede no ser "
                    "representativa; considere un indicador de ausencia o la exclusión."
                )
                var_warnings.append(message)
                warnings.append(message)
            if profile.target_variable and str(var) == profile.target_variable:
                var_warnings.append(
                    f"La variable '{var}' es la variable objetivo: imputarla puede "
                    "distorsionar el análisis predictivo o inferencial. Evalúe mantener "
                    "un indicador de ausencia y validar con E4 si procede."
                )

            scores = [
                self._score_method(
                    name,
                    var=var,
                    variable_type=var_type,
                    pct=pct,
                    profile=profile,
                    e2_not_mcar=e2_not_mcar,
                    e2_significant_for_var=e2_sig_for_var.get(str(var), False),
                    has_complete_numeric_predictor=has_complete_numeric_predictor,
                    evaluation=evaluation,
                )
                for name in self.registry.names()
            ]

            non_excluded = [s for s in scores if not s.excluded]
            non_excluded.sort(key=lambda s: s.score, reverse=True)
            if len(non_excluded) >= 2:
                gap = non_excluded[0].score - non_excluded[1].score
                if gap < SCORE_GAP_SENSITIVITY_THRESHOLD:
                    var_warnings.append(
                        f"La diferencia entre el método recomendado "
                        f"('{non_excluded[0].method}') y la alternativa "
                        f"('{non_excluded[1].method}') es de {gap:.4f} puntos, menor al "
                        f"umbral de sensibilidad ({SCORE_GAP_SENSITIVITY_THRESHOLD:.2f}): "
                        "el ranking es sensible a los pesos; considere las alternativas."
                    )
            excluded = [s for s in scores if s.excluded]
            all_scores = non_excluded + excluded
            recommended = non_excluded[0] if non_excluded else None
            alternatives = non_excluded[1:]

            variables[str(var)] = VariableRecommendation(
                variable=str(var),
                variable_type=var_type,
                missing_count=info.missing_count,
                missing_percentage=info.missing_percentage,
                recommended=recommended,
                alternatives=alternatives,
                all_scores=all_scores,
                warnings=var_warnings,
            )

        group_ranking = self._group_ranking(variables)

        reproduction = {
            "module": "src.missing_data.selector",
            "version": SELECTOR_VERSION,
            "weights": self.weights,
            "alpha": self.alpha,
            "registry": list(self.registry.names()),
            "evaluation_provided": evaluation is not None,
        }

        logger.info(
            f"Selección completada: {len(variables)} variables con faltantes, "
            f"{sum(1 for v in variables.values() if v.recommended is not None)} "
            "variables con método recomendado."
        )

        return ImputationSelectionReport(
            weights=self.weights,
            variables=variables,
            group_ranking=group_ranking,
            warnings=warnings,
            selector_version=SELECTOR_VERSION,
            reproduction=reproduction,
        )

    # ------------------------------------------------------------------
    # Métodos internos
    # ------------------------------------------------------------------

    def _validate_inputs(
        self,
        profile: DatasetProfile,
        missing_report: MissingReport,
        diagnostics: MissingnessDiagnosticsReport,
        evaluation: Optional[ImputationEvaluationReport],
    ) -> None:
        if not isinstance(profile, DatasetProfile):
            raise TypeError("profile debe ser un DatasetProfile.")
        if not isinstance(missing_report, MissingReport):
            raise TypeError("missing_report debe ser un MissingReport (E1).")
        if not isinstance(diagnostics, MissingnessDiagnosticsReport):
            raise TypeError("diagnostics debe ser un MissingnessDiagnosticsReport (E2).")
        if evaluation is not None and not isinstance(evaluation, ImputationEvaluationReport):
            raise TypeError("evaluation debe ser un ImputationEvaluationReport (E4) o None.")

    def _score_method(
        self,
        name: str,
        *,
        var: str,
        variable_type: str,
        pct: float,
        profile: DatasetProfile,
        e2_not_mcar: bool,
        e2_significant_for_var: bool,
        has_complete_numeric_predictor: bool,
        evaluation: Optional[ImputationEvaluationReport],
    ) -> MethodScore:
        method_cls = self.registry.get(name)
        caps = method_cls.capabilities
        prof = _profile_for(name, caps)
        description = method_cls.description

        if str(var) in profile.identifier_columns:
            return MethodScore(
                method=name,
                variable=var,
                description=description,
                score=None,
                components={},
                reasons=[],
                caveats=["La variable es un identificador (una por fila); no debe imputarse."],
                e4_evidence=None,
                excluded=True,
                exclusion_reason=(
                    "La variable es identificadora (una por fila); no debe imputarse."
                ),
            )

        type_score, type_reason = _type_fit(caps, variable_type)
        if type_score == 0:
            return self._excluded(name, var, description, type_reason)

        structure_score, structure_reason = _structure_fit(caps, profile.temporal)
        if structure_score == 0:
            return self._excluded(name, var, description, structure_reason)

        if caps.needs_other_columns and profile.n_variables < NEEDS_OTHER_MIN_VARIABLES:
            return self._excluded(
                name,
                var,
                description,
                "El método requiere otras variables predictoras y el conjunto solo "
                f"tiene {profile.n_variables} variable(s) (puerta dura de estructura).",
            )

        if name == "regresion" and not has_complete_numeric_predictor:
            return self._excluded(
                name,
                var,
                description,
                "El método 'regresion' requiere al menos una columna numérica "
                "completamente observada (sin faltantes, missing_count == 0) para "
                "usarse como predictora; ninguna existe en el conjunto "
                "(puerta dura de estructura).",
            )

        mp_score, mp_reason = _missing_pct_fit(prof, pct)
        ss_score, ss_reason = _sample_size_fit(prof, profile.n_observations)
        rel_score, rel_reason = _relationship_exploitation(
            prof, caps, profile.n_variables, e2_significant_for_var
        )
        e4 = _e4_evidence_for(name, str(var), evaluation)
        e4_score = _e4_robustness_score(e4, str(var), evaluation)
        rob_score, rob_reason = _robustness(prof, e2_not_mcar, e4_score)
        cc_score, cc_reason = _complexity_cost(prof)

        components = {
            "type_fit": type_score,
            "missing_pct_fit": mp_score,
            "sample_size_fit": ss_score,
            "structure_fit": structure_score,
            "relationship_exploitation": rel_score,
            "robustness": rob_score,
            "complexity_cost": cc_score,
        }
        score = sum(self.weights[c] * components[c] for c in _COMPONENT_NAMES)
        score = max(0.0, min(1.0, score))

        reasons = [
            f"type_fit: {type_reason}",
            f"missing_pct_fit: {mp_reason}",
            f"sample_size_fit: {ss_reason}",
            f"structure_fit: {structure_reason}",
            f"relationship_exploitation: {rel_reason}",
            f"robustness: {rob_reason}",
            f"complexity_cost: {cc_reason}",
        ]

        caveats: List[str] = []
        if pct > HIGH_MISSINGNESS_THRESHOLD:
            caveats.append(
                f"Variable con {pct:.0%} de faltantes: imputación especulativa."
            )
        if profile.target_variable and str(var) == profile.target_variable:
            caveats.append(
                "La variable es la variable objetivo: revise las implicaciones de imputarla."
            )
        if e4 is None:
            caveats.append("Sin evidencia empírica de E4: el puntaje usa solo el perfil a priori.")
        elif not e4.get("available"):
            caveats.append(f"E4: {e4.get('reason', 'sin evidencia disponible.')}")
        elif e4.get("error"):
            caveats.append(f"E4: el método falló en la evaluación artificial ({e4['error']}).")
        else:
            caveats.append(
                f"E4 disponible: RMSE/MAE/accuracy incorporados a la robustez "
                f"(rank E4={e4.get('rank')})."
            )
        if caps.temporal_only:
            caveats.append("Método temporal: asume que el orden de filas tiene sentido.")

        return MethodScore(
            method=name,
            variable=var,
            description=description,
            score=score,
            components=components,
            reasons=reasons,
            caveats=caveats,
            e4_evidence=e4,
            excluded=False,
            exclusion_reason=None,
        )

    @staticmethod
    def _excluded(
        name: str, var: str, description: str, reason: str
    ) -> MethodScore:
        return MethodScore(
            method=name,
            variable=var,
            description=description,
            score=None,
            components={},
            reasons=[],
            caveats=[reason],
            e4_evidence=None,
            excluded=True,
            exclusion_reason=reason,
        )

    @staticmethod
    def _e2_questions_mcar(diagnostics: MissingnessDiagnosticsReport) -> bool:
        if diagnostics.mechanism.significant_comparisons:
            return True
        evidence = diagnostics.mechanism.evidence or ""
        return "en contra del supuesto MCAR" in evidence

    @staticmethod
    def _has_complete_numeric_predictor(
        profile: DatasetProfile,
        missing_report: MissingReport,
    ) -> bool:
        """True si existe al menos una columna numérica con missing_count == 0.

        Las columnas identificadoras se excluyen como predictoras. La condición
        es estrictamente missing_count == 0 (usar `variables_without_missing`);
        una variable con pocos faltantes NO cuenta como predictora completa.
        """
        for col in missing_report.variables_without_missing:
            col_str = str(col)
            if col_str in profile.identifier_columns:
                continue
            if profile.is_numeric(col_str):
                return True
        return False

    def _e2_significant_per_variable(
        self, diagnostics: MissingnessDiagnosticsReport
    ) -> Dict[str, bool]:
        result: Dict[str, bool] = {}
        for assoc in diagnostics.associations:
            if assoc.adjusted_p_value is not None and assoc.adjusted_p_value < self.alpha:
                result[assoc.variable] = True
                result[assoc.associated_with] = True
        return result

    @staticmethod
    def _group_ranking(
        variables: Dict[str, VariableRecommendation],
    ) -> Dict[str, List[Dict[str, Any]]]:
        groups: Dict[str, List[str]] = {}
        for var, rec in variables.items():
            groups.setdefault(_group_for(rec.variable_type), []).append(var)

        ranking: Dict[str, List[Dict[str, Any]]] = {}
        for group, var_list in groups.items():
            agg: Dict[str, List[float]] = {}
            best_for: Dict[str, List[str]] = {}
            for var in var_list:
                rec = variables[var]
                for s in rec.all_scores:
                    if s.excluded or s.score is None:
                        continue
                    agg.setdefault(s.method, []).append(s.score)
                if rec.recommended is not None:
                    best_for.setdefault(rec.recommended.method, []).append(var)
            entries: List[Dict[str, Any]] = []
            for method, scores in agg.items():
                entries.append(
                    {
                        "method": method,
                        "mean_score": sum(scores) / len(scores),
                        "n_variables": len(scores),
                        "best_for": sorted(best_for.get(method, [])),
                    }
                )
            entries.sort(key=lambda e: e["mean_score"], reverse=True)
            ranking[group] = entries
        return ranking
