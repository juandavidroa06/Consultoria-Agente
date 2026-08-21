"""
Diagnóstico estadístico de datos faltantes (Etapa E2).

Evalúa el mecanismo de ausencia mediante:
  - Patrones sistemáticos de ausencia (co-ausencia, bloques, monotonía).
  - Comparaciones estadísticas entre el indicador de ausencia de cada variable con
    datos faltantes y el resto de las variables (Mann-Whitney U para numéricas,
    chi-cuadrado / Fisher exacto para categóricas).
  - Corrección por múltiples comparaciones (FDR de Benjamini-Hochberg).

El reporte distingue explícitamente la evidencia estadística de la inferencia:
una asociación significativa es evidencia en contra del supuesto MCAR; nunca se
declara MAR o MNAR como un hecho. Imputación y simulación quedan fuera del alcance.
"""

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import MonteCarloMethod, chi2_contingency, fisher_exact, mannwhitneyu
from statsmodels.stats.multitest import multipletests

from src.data.validator import DataValidator
from src.missing_data.detection import MissingDataDetector
from src.utils.logger import setup_logger

logger = setup_logger("MissingDataDiagnostics")

DEFAULT_ALPHA = 0.05
DEFAULT_MC_N_RESAMPLES = 9999
DEFAULT_MIN_GROUP_SIZE = 5


@dataclass
class MissingnessAssociation:
    """Resultado de una comparación estadística entre un indicador de ausencia y una variable."""

    variable: str
    associated_with: str
    associated_variable_type: str
    test: str
    statistic: Optional[float]
    p_value: Optional[float]
    adjusted_p_value: Optional[float]
    n_observed: int
    n_missing: int
    conclusion: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Devuelve la asociación como diccionario JSON-serializable."""
        return asdict(self)


@dataclass
class MechanismAssessment:
    """Evaluación de la evidencia estadística sobre el mecanismo de ausencia."""

    evidence: str
    tests_performed: int
    significant_comparisons: List[str]
    limitations: List[str]
    cannot_infer: List[str]
    recommendation: str

    def to_dict(self) -> Dict[str, Any]:
        """Devuelve la evaluación como diccionario JSON-serializable."""
        return asdict(self)


@dataclass
class MissingnessDiagnosticsReport:
    """Reporte estructurado y reproducible del diagnóstico de datos faltantes."""

    status: str
    total_observations: int
    total_variables: int
    variables_with_missing: List[str]
    co_missing_counts: Dict[str, int]
    systematic_patterns: List[str]
    observations_missing_distribution: Dict[int, int]
    associations: List[MissingnessAssociation]
    skipped_comparisons: List[Dict[str, Any]]
    multiple_comparisons: Dict[str, Any]
    mechanism: MechanismAssessment
    reproduction: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Devuelve el reporte como diccionario JSON-serializable."""
        return asdict(self)


class MissingDataDiagnostics:
    """
    Diagnóstico estadístico reproducible del mecanismo de datos faltantes.

    Realiza comparaciones estadísticas entre el patrón de ausencia (indicador 0/1)
    de cada variable con faltantes y las demás variables del DataFrame. La selección
    de la prueba depende del tipo de variable:

      - Numérica  -> Mann-Whitney U (independencia del indicador de ausencia).
      - Categórica (texto, booleana, category) -> chi-cuadrado; Fisher exacto si la
        tabla es 2x2 y alguna frecuencia esperada es insuficiente; Monte Carlo
        reproducible si la tabla es mayor que 2x2 y hay esperadas insuficientes.
      - Fecha/hora -> se omite (prueba no soportada).

    Los p-valores se corrigen por múltiples comparaciones con FDR
    (Benjamini-Hochberg). Las conclusiones se expresan de forma conservadora:
    la evidencia cuestiona o no el supuesto MCAR; no se afirma MAR/MNAR.
    """

    def __init__(
        self,
        alpha: float = DEFAULT_ALPHA,
        random_state: Optional[int] = 42,
        min_group_size: int = DEFAULT_MIN_GROUP_SIZE,
        mc_n_resamples: int = DEFAULT_MC_N_RESAMPLES,
        detect_placeholders: bool = True,
    ) -> None:
        self.alpha = float(alpha)
        self.random_state = random_state
        self.min_group_size = int(min_group_size)
        self.mc_n_resamples = int(mc_n_resamples)
        self.detector = MissingDataDetector(detect_placeholders=detect_placeholders)

    def diagnose(self, df: pd.DataFrame) -> MissingnessDiagnosticsReport:
        """
        Ejecuta el diagnóstico completo de datos faltantes del DataFrame.

        Args:
            df: DataFrame de pandas a diagnosticar.

        Returns:
            MissingnessDiagnosticsReport con co-ausencia, patrones sistemáticos,
            asociaciones, corrección por múltiples comparaciones y evaluación del mecanismo.

        Raises:
            TypeError: Si df no es un DataFrame de pandas.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Se requiere un DataFrame de pandas.")

        df = df.reset_index(drop=True)

        logger.info(
            f"Iniciando diagnóstico de datos faltantes en DataFrame de "
            f"{df.shape[0]} filas x {df.shape[1]} columnas."
        )

        report = self.detector.detect(df)
        total_observations = report.total_observations
        total_variables = report.total_variables
        variables_with_missing = report.variables_with_missing

        if report.status == "sin_faltantes":
            status = "sin_faltantes"
            co_missing_counts: Dict[str, int] = {}
            systematic_patterns = ["No hay variables con datos faltantes."]
            associations: List[MissingnessAssociation] = []
            skipped: List[Dict[str, Any]] = []
            multiple_comparisons = {
                "method": "fdr_bh",
                "alpha": self.alpha,
                "n_comparisons": 0,
                "n_significant": 0,
            }
            mechanism = self._assess_mechanism(
                associations=associations,
                limitations=[],
                no_missing=True,
            )
        else:
            status = "con_faltantes"
            co_missing_counts = self._co_missing_counts(df, variables_with_missing)
            systematic_patterns = self._systematic_patterns(df, variables_with_missing)

            associations, skipped = self._test_associations(df, variables_with_missing)

            pvals = [a.p_value for a in associations if a.p_value is not None]
            multiple_comparisons = {
                "method": "fdr_bh",
                "alpha": self.alpha,
                "n_comparisons": len(pvals),
            }
            if pvals:
                _, pvals_corrected, _, _ = multipletests(
                    pvals, alpha=self.alpha, method="fdr_bh"
                )
                for assoc, padj in zip(associations, pvals_corrected):
                    assoc.adjusted_p_value = float(padj)
                    assoc.conclusion = self._conclusion_for(assoc, padj)
                multiple_comparisons["n_significant"] = int(
                    sum(1 for p in pvals_corrected if p < self.alpha)
                )
            else:
                multiple_comparisons["n_significant"] = 0

            limitations = [
                assoc for assoc in associations
                if assoc.test == "Mann-Whitney U" and assoc.n_missing < self.min_group_size
            ]
            limitations_text = [
                f"Para '{lim.variable}' el grupo con datos faltantes es pequeño "
                f"(n={lim.n_missing} < {self.min_group_size}); la potencia de la prueba es limitada."
                for lim in limitations
            ]

            mechanism = self._assess_mechanism(
                associations=associations,
                limitations=limitations_text,
                no_missing=False,
            )

        reproduction = {
            "random_state": self.random_state,
            "alpha": self.alpha,
            "min_group_size": self.min_group_size,
            "mc_n_resamples": self.mc_n_resamples,
            "module": "src.missing_data.diagnostics",
            "version": "1.0",
        }

        diag_report = MissingnessDiagnosticsReport(
            status=status,
            total_observations=total_observations,
            total_variables=total_variables,
            variables_with_missing=variables_with_missing,
            co_missing_counts=co_missing_counts,
            systematic_patterns=systematic_patterns,
            observations_missing_distribution=report.observations_missing_distribution,
            associations=associations,
            skipped_comparisons=skipped,
            multiple_comparisons=multiple_comparisons,
            mechanism=mechanism,
            reproduction=reproduction,
        )

        logger.info(
            f"Diagnóstico completado: {len(associations)} comparaciones realizadas, "
            f"{len(skipped)} omitidas."
        )
        return diag_report

    # ------------------------------------------------------------------
    # Patrones sistemáticos
    # ------------------------------------------------------------------

    def _co_missing_counts(
        self, df: pd.DataFrame, variables_with_missing: List[str]
    ) -> Dict[str, int]:
        """Número de filas en las que cada variable con faltantes está ausente."""
        counts: Dict[str, int] = {}
        for var in variables_with_missing:
            counts[str(var)] = int(df[var].isna().sum())
        return counts

    def _systematic_patterns(
        self, df: pd.DataFrame, variables_with_missing: List[str]
    ) -> List[str]:
        """Identifica patrones de ausencia sistemática (bloques y monotonía)."""
        if not variables_with_missing:
            return []

        missing_mask = df[variables_with_missing].isna()
        n_obs = missing_mask.shape[0]
        patterns: List[str] = []

        # Patrón dominante: alguna variable ausente en más del 70% de las filas.
        col_missing = missing_mask.sum(axis=0).to_dict()
        for var, cnt in col_missing.items():
            if n_obs > 0 and cnt / n_obs > 0.7:
                patterns.append(
                    f"Patrón de ausencia dominante: '{var}' presenta datos faltantes "
                    f"en más del 70% de las observaciones."
                )

        if len(variables_with_missing) >= 2:
            # Co-ausencia por pares mayor a la esperada bajo independencia (bloque).
            block_evidence = False
            for i, var_i in enumerate(variables_with_missing):
                for var_j in variables_with_missing[i + 1 :]:
                    miss_i = df[var_i].isna()
                    miss_j = df[var_j].isna()
                    both = int((miss_i & miss_j).sum())
                    n_i, n_j = int(miss_i.sum()), int(miss_j.sum())
                    if n_i > 0 and n_j > 0 and n_obs > 0:
                        expected = n_i * n_j / n_obs
                        if expected > 0 and both > expected * 1.5:
                            block_evidence = True
                            break
                if block_evidence:
                    break

            if block_evidence:
                patterns.append(
                    "Patrón de ausencia en bloque: la co-ausencia entre pares de "
                    "variables supera lo esperado bajo independencia."
                )
            else:
                patterns.append(
                    "No se detectó co-ausencia en bloque entre pares de variables "
                    "con datos faltantes."
                )

            # Monotonía: la ausencia de una variable implica la ausencia de otra.
            monotone_pairs = []
            for i, var_i in enumerate(variables_with_missing):
                for var_j in variables_with_missing[i + 1 :]:
                    miss_i = df[var_i].isna()
                    miss_j = df[var_j].isna()
                    if miss_i.sum() > 0 and miss_j.sum() > 0:
                        if (miss_i & ~miss_j).sum() == 0:
                            monotone_pairs.append(f"{var_i} -> {var_j}")
                        elif (miss_j & ~miss_i).sum() == 0:
                            monotone_pairs.append(f"{var_j} -> {var_i}")

            if monotone_pairs:
                patterns.append(
                    "Patrón de ausencia monótono: la ausencia de una variable implica "
                    "la ausencia de otra (" + ", ".join(monotone_pairs) + ")."
                )

        if len(variables_with_missing) == 1:
            var = variables_with_missing[0]
            n_miss = int(df[var].isna().sum())
            patterns.append(
                f"Patrón univariado: solo '{var}' presenta datos faltantes "
                f"({n_miss} observaciones ausentes)."
            )

        if not patterns:
            patterns.append(
                "No se identificaron patrones sistemáticos evidentes de ausencia."
            )

        return patterns

    # ------------------------------------------------------------------
    # Comparaciones estadísticas
    # ------------------------------------------------------------------

    def _test_associations(
        self,
        df: pd.DataFrame,
        variables_with_missing: List[str],
    ) -> Tuple[List[MissingnessAssociation], List[Dict[str, Any]]]:
        """Compara el indicador de ausencia de cada variable con las demás."""
        variable_types = DataValidator.identify_variable_types(df)
        associations: List[MissingnessAssociation] = []
        skipped: List[Dict[str, Any]] = []

        for var in variables_with_missing:
            indicator = df[var].isna().astype(int)
            n_missing = int(indicator.sum())

            for other in df.columns:
                if other == var:
                    continue
                other_type = variable_types.get(other, "Desconocido")

                if pd.api.types.is_numeric_dtype(df[other].dtype):
                    assoc = self._test_numeric(var, df, other, other_type, indicator, n_missing)
                elif pd.api.types.is_datetime64_any_dtype(df[other].dtype):
                    skipped.append(
                        {
                            "variable": str(var),
                            "associated_with": str(other),
                            "associated_variable_type": other_type,
                            "reason": "tipo de variable no soportado (fecha/hora)",
                        }
                    )
                    continue
                else:
                    assoc = self._test_categorical(var, df, other, other_type, indicator)

                if assoc is None:
                    continue
                associations.append(assoc)

        return associations, skipped

    def _test_numeric(
        self,
        var: Any,
        df: pd.DataFrame,
        other: Any,
        other_type: str,
        indicator: pd.Series,
        n_missing: int,
    ) -> Optional[MissingnessAssociation]:
        """Mann-Whitney U entre la variable numérica y el indicador de ausencia."""
        observed = df[other].dropna()
        if observed.nunique() <= 1:
            return None

        n_observed = int((indicator == 0).sum())
        mask = indicator.to_numpy().astype(bool)
        values = df[other].to_numpy()
        group_missing = pd.Series(values[mask]).dropna()
        group_observed = pd.Series(values[~mask]).dropna()

        if len(group_missing) == 0 or len(group_observed) == 0:
            return None
        if n_missing < self.min_group_size:
            return MissingnessAssociation(
                variable=str(var),
                associated_with=str(other),
                associated_variable_type=other_type,
                test="Mann-Whitney U",
                statistic=None,
                p_value=None,
                adjusted_p_value=None,
                n_observed=n_observed,
                n_missing=n_missing,
                conclusion=(
                    "Prueba no realizada: el grupo con datos faltantes es demasiado "
                    f"pequeño (n={n_missing} < {self.min_group_size})."
                ),
            )

        stat, p = mannwhitneyu(group_missing, group_observed, alternative="two-sided")
        return MissingnessAssociation(
            variable=str(var),
            associated_with=str(other),
            associated_variable_type=other_type,
            test="Mann-Whitney U",
            statistic=float(stat),
            p_value=float(p),
            adjusted_p_value=None,
            n_observed=n_observed,
            n_missing=n_missing,
            conclusion="",
        )

    def _test_categorical(
        self,
        var: Any,
        df: pd.DataFrame,
        other: Any,
        other_type: str,
        indicator: pd.Series,
    ) -> Optional[MissingnessAssociation]:
        """Chi-cuadrado o Fisher exacto entre la variable categórica y el indicador de ausencia."""
        col = df[other]
        if col.nunique() <= 1:
            return None

        table = pd.crosstab(indicator, col)
        table_arr = table.to_numpy(dtype=int)

        if table_arr.size == 0 or table_arr.sum() == 0:
            return None

        if (indicator == 0).sum() == 0 or (indicator == 1).sum() == 0:
            return None

        try:
            expected = _expected_freq(table_arr)
        except ValueError:
            return None

        min_expected = float(expected.min()) if expected.size else float("inf")

        if min_expected >= 5:
            res = chi2_contingency(table_arr, correction=False)
            statistic, p = float(res.statistic), float(res.pvalue)
            test = "Chi-cuadrado"
        elif table_arr.shape == (2, 2):
            statistic, p = fisher_exact(table_arr, alternative="two-sided")
            test = "Fisher exacto"
        else:
            method = MonteCarloMethod(n_resamples=self.mc_n_resamples, rng=self.random_state)
            res = chi2_contingency(table_arr, correction=False, method=method)
            statistic, p = float(res.statistic), float(res.pvalue)
            test = "Chi-cuadrado (Monte Carlo)"

        n_observed = int((indicator == 0).sum())
        n_missing = int((indicator == 1).sum())
        return MissingnessAssociation(
            variable=str(var),
            associated_with=str(other),
            associated_variable_type=other_type,
            test=test,
            statistic=statistic,
            p_value=float(p),
            adjusted_p_value=None,
            n_observed=n_observed,
            n_missing=n_missing,
            conclusion="",
        )

    # ------------------------------------------------------------------
    # Conclusión y evaluación del mecanismo
    # ------------------------------------------------------------------

    def _conclusion_for(self, assoc: MissingnessAssociation, adjusted_p: float) -> str:
        if adjusted_p < self.alpha:
            return (
                f"Se encontró evidencia de asociación entre el patrón de ausencia de "
                f"'{assoc.variable}' y '{assoc.associated_with}' "
                f"(p ajustado = {adjusted_p:.4g}), lo que cuestiona el supuesto MCAR."
            )
        return (
            f"No se encontró evidencia estadísticamente significativa de asociación "
            f"entre el patrón de ausencia de '{assoc.variable}' y '{assoc.associated_with}' "
            f"(p ajustado = {adjusted_p:.4g})."
        )

    def _assess_mechanism(
        self,
        associations: List[MissingnessAssociation],
        limitations: List[str],
        no_missing: bool,
    ) -> MechanismAssessment:
        if no_missing:
            return MechanismAssessment(
                evidence=(
                    "El conjunto de datos no presenta valores faltantes; el mecanismo de "
                    "ausencia no es aplicable."
                ),
                tests_performed=0,
                significant_comparisons=[],
                limitations=["Sin datos faltantes, no hay mecanismo que evaluar."],
                cannot_infer=[
                    "MAR no puede demostrarse ni descartarse sin datos faltantes.",
                    "MNAR no puede inferirse sin datos faltantes ni información externa.",
                ],
                recommendation=(
                    "No se requiere tratamiento de datos faltantes. Proceda con el análisis "
                    "estadístico principal."
                ),
            )

        tests_performed = sum(
            1 for a in associations if a.p_value is not None
        )
        significant = [
            a for a in associations
            if a.adjusted_p_value is not None and a.adjusted_p_value < self.alpha
        ]
        significant_terms = [
            f"{a.variable} | {a.associated_with}" for a in significant
        ]

        if significant:
            evidence = (
                f"Se detectaron {len(significant)} asociaciones significativas (FDR "
                f"alpha={self.alpha}) entre el patrón de ausencia y otras variables: "
                + "; ".join(significant_terms)
                + ". Esto constituye evidencia en contra del supuesto MCAR."
            )
        else:
            evidence = (
                "No se detectaron asociaciones significativas entre el patrón de ausencia "
                "y las variables comparadas. No se encontró evidencia que cuestione el "
                "supuesto MCAR; sin embargo, esto no lo confirma."
            )

        cannot_infer = [
            "MAR no puede demostrarse ni descartarse con los datos observados.",
            "MNAR no puede inferirse únicamente con los datos observados.",
            "La ausencia de asociación no prueba que los datos sean MCAR.",
        ]

        if tests_performed == 0:
            evidence = (
                "No se pudo realizar ninguna comparación estadística del patrón de "
                "ausencia; la evaluación del mecanismo queda limitada a los patrones "
                "descriptivos."
            )

        recommendation = (
            "Si la evidencia cuestiona MCAR, considere modelar explícitamente el "
            "mecanismo de ausencia o analizar la sensibilidad de los resultados "
            "principales. La decisión final de imputación o exclusión debe basarse "
            "en el objetivo y el contexto del estudio, no únicamente en estas pruebas."
        )

        return MechanismAssessment(
            evidence=evidence,
            tests_performed=tests_performed,
            significant_comparisons=significant_terms,
            limitations=limitations,
            cannot_infer=cannot_infer,
            recommendation=recommendation,
        )


def _expected_freq(table: np.ndarray) -> np.ndarray:
    """Frecuencias esperadas bajo independencia (mismos márgenes que chi2_contingency)."""
    table = np.asarray(table, dtype=float)
    if table.ndim != 2:
        raise ValueError("La tabla de contingencia debe ser bidimensional.")
    if table.size == 0:
        return np.array([])
    if np.any(table < 0):
        raise ValueError("Las frecuencias observadas no pueden ser negativas.")
    row_tot = table.sum(axis=1, keepdims=True)
    col_tot = table.sum(axis=0, keepdims=True)
    grand = float(table.sum())
    if grand == 0:
        raise ValueError("La tabla de contingencia no puede tener total cero.")
    return row_tot @ col_tot / grand