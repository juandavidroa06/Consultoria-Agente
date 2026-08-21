"""
Validación post-imputación (Etapa E6).

Compara un DataFrame imputado contra el original (con faltantes) y emite un
veredicto reproducible: "Aceptable" o "Revisar".

Comprobaciones realizadas (en orden):
  1. faltantes_residuales: celdas que siguen siendo NaN tras la imputación.
  2. dtype_preservacion: cambio del dtype de una columna tras la imputación.
  3. valores_imposibles: valores imputados fuera del dominio observado
     (rango [min, max] para numéricas; categorías observadas para categóricas).
  4. distribucion_ks: prueba de Kolmogorov-Smirnov (dos muestras) entre los
     valores observados y los imputados de cada columna numérica, junto con
     estadísticas descriptivas (media y desviación) antes y después.
  5. correlaciones: cambio en la matriz de correlación de Pearson entre las
     columnas numéricas (antes con datos observados, después con la imputación).
  6. proporcion_imputada: proporción de celdas imputadas (global y por columna).
  7. comparacion_imputados_vs_observados: sesgo entre los valores imputados y
     los observados de la misma variable (media para numéricas; moda para
     categóricas).

Estados posibles de una comprobación:
  - "ok":   no se detectó ningún problema.
  - "warn": se detectó un cambio o situación que requiere atención.
  - "error": se detectó un problema que compromete la imputación.

Veredicto: "Revisar" si existe alguna comprobación con estado "error"
(faltantes residuales o valores imposibles); en otro caso "Aceptable".
Las advertencias ("warn") se reportan pero no cambian el veredicto por sí solas.

Umbrales documentados y configurables:
  - ks_alpha: 0.05 (significancia de la prueba KS).
  - corr_threshold: 0.15 (cambio absoluto de una correlación que se señala).
  - distribution_rel_change: 0.10 (cambio relativo de media/desviación señalado).
  - high_imputation_threshold: 0.50 (proporción de imputación señalada por columna).
  - bias_threshold: 0.10 (sesgo relativo de media señalado).
  - min_ks_sample: 5 (tamaño mínimo de cada muestra para la prueba KS).

No hay aleatoriedad en la validación: los resultados son reproducibles.
"""

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from src.utils.logger import setup_logger

logger = setup_logger("ImputationValidator")

VALIDATION_VERSION = "1.0"

DEFAULT_KS_ALPHA = 0.05
DEFAULT_CORR_THRESHOLD = 0.15
DEFAULT_DISTRIBUTION_REL_CHANGE = 0.10
DEFAULT_HIGH_IMPUTATION_THRESHOLD = 0.50
DEFAULT_BIAS_THRESHOLD = 0.10
DEFAULT_MIN_KS_SAMPLE = 5

VERDICT_ACEPTABLE = "Aceptable"
VERDICT_REVISAR = "Revisar"


def _is_numeric_series(series: pd.Series) -> bool:
    return (
        pd.api.types.is_numeric_dtype(series.dtype)
        and not pd.api.types.is_bool_dtype(series.dtype)
        and not pd.api.types.is_datetime64_any_dtype(series.dtype)
        and not pd.api.types.is_timedelta64_dtype(series.dtype)
    )


def _is_categorical_series(series: pd.Series) -> bool:
    return not (
        _is_numeric_series(series)
        or pd.api.types.is_datetime64_any_dtype(series.dtype)
        or pd.api.types.is_timedelta64_dtype(series.dtype)
    )


def _relative_change(before: Optional[float], after: Optional[float]) -> Optional[float]:
    """Cambio relativo |después - antes| / |antes|; None si antes es 0 o indefinido."""
    if before is None or after is None:
        return None
    before = float(before)
    after = float(after)
    if before == 0:
        return None
    return abs(after - before) / abs(before)


@dataclass
class ValidationCheck:
    """Resultado de una comprobación individual de la validación."""

    name: str
    status: str
    passed: bool
    message: str
    details: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        """Devuelve la comprobación como diccionario JSON-serializable."""
        return asdict(self)


@dataclass
class ImputationValidationReport:
    """Reporte estructurado y reproducible de la validación post-imputación."""

    verdict: str
    n_imputed_cells: int
    n_imputed_percentage: float
    residual_missing: Dict[str, int]
    checks: List[ValidationCheck]
    warnings: List[str]
    reproduction: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Devuelve el reporte como diccionario JSON-serializable."""
        return asdict(self)


class ImputationValidator:
    """
    Valida un DataFrame imputado contra el original con faltantes.

    La máscara de celdas imputadas se deriva de `original.isna()` (celdas que
    estaban ausentes y ahora están completas); no requiere un MissingReport.
    Original e imputado deben tener el mismo índice y el mismo número de filas.
    """

    def __init__(
        self,
        *,
        ks_alpha: float = DEFAULT_KS_ALPHA,
        corr_threshold: float = DEFAULT_CORR_THRESHOLD,
        distribution_rel_change: float = DEFAULT_DISTRIBUTION_REL_CHANGE,
        high_imputation_threshold: float = DEFAULT_HIGH_IMPUTATION_THRESHOLD,
        bias_threshold: float = DEFAULT_BIAS_THRESHOLD,
        min_ks_sample: int = DEFAULT_MIN_KS_SAMPLE,
    ) -> None:
        if not (0.0 < ks_alpha < 1.0):
            raise ValueError("ks_alpha debe estar en el intervalo abierto (0, 1).")
        if corr_threshold < 0.0:
            raise ValueError("corr_threshold no puede ser negativo.")
        if distribution_rel_change < 0.0:
            raise ValueError("distribution_rel_change no puede ser negativo.")
        if not (0.0 < high_imputation_threshold <= 1.0):
            raise ValueError("high_imputation_threshold debe estar en (0, 1].")
        if bias_threshold < 0.0:
            raise ValueError("bias_threshold no puede ser negativo.")
        if int(min_ks_sample) < 1:
            raise ValueError("min_ks_sample debe ser al menos 1.")
        self.ks_alpha = float(ks_alpha)
        self.corr_threshold = float(corr_threshold)
        self.distribution_rel_change = float(distribution_rel_change)
        self.high_imputation_threshold = float(high_imputation_threshold)
        self.bias_threshold = float(bias_threshold)
        self.min_ks_sample = int(min_ks_sample)

    def validate(
        self,
        original: pd.DataFrame,
        imputed: pd.DataFrame,
    ) -> ImputationValidationReport:
        """
        Ejecuta la validación completa de la imputación.

        Args:
            original: DataFrame antes de imputar (con datos faltantes).
            imputed: DataFrame después de imputar.

        Returns:
            ImputationValidationReport con veredicto, comprobaciones y advertencias.
        """
        if not isinstance(original, pd.DataFrame):
            raise TypeError("original debe ser un DataFrame de pandas.")
        if not isinstance(imputed, pd.DataFrame):
            raise TypeError("imputed debe ser un DataFrame de pandas.")
        if imputed.shape[0] != original.shape[0]:
            raise ValueError(
                "original e imputado deben tener el mismo número de filas "
                f"({original.shape[0]} vs {imputed.shape[0]})."
            )
        if not imputed.index.equals(original.index):
            raise ValueError("Los índices de original e imputado deben coincidir.")

        present_cols = [c for c in original.columns if c in imputed.columns]
        mask = original[present_cols].isna() & imputed[present_cols].notna()
        n_imputed = int(mask.to_numpy().sum())
        n_total = original.size
        n_imputed_pct = (n_imputed / n_total) * 100 if n_total else 0.0

        checks: List[ValidationCheck] = [
            self._check_residual_missing(original, imputed),
            self._check_dtype_preservation(original, imputed),
            self._check_impossible_values(original, imputed, mask),
            self._check_distribution(original, imputed, mask),
            self._check_correlations(original, imputed),
            self._check_proportion(original, imputed, mask),
            self._check_imputed_vs_observed(original, imputed, mask),
        ]

        warnings: List[str] = []
        for check in checks:
            if check.status != "ok":
                warnings.append(f"[{check.name}] {check.message}")

        has_error = any(c.status == "error" for c in checks)
        verdict = VERDICT_REVISAR if has_error else VERDICT_ACEPTABLE

        residual = {
            col: int(imputed[col].isna().sum())
            for col in original.columns
            if col in imputed.columns and imputed[col].isna().any()
        }
        missing_cols = [c for c in original.columns if c not in imputed.columns]
        for col in missing_cols:
            residual[col] = len(original)

        new_columns = [c for c in imputed.columns if c not in original.columns]

        reproduction = {
            "module": "src.missing_data.validation",
            "version": VALIDATION_VERSION,
            "n_observations": len(original),
            "n_variables_original": original.shape[1],
            "n_variables_imputado": imputed.shape[1],
            "new_columns_in_imputed": new_columns,
            "thresholds": {
                "ks_alpha": self.ks_alpha,
                "corr_threshold": self.corr_threshold,
                "distribution_rel_change": self.distribution_rel_change,
                "high_imputation_threshold": self.high_imputation_threshold,
                "bias_threshold": self.bias_threshold,
                "min_ks_sample": self.min_ks_sample,
            },
        }

        logger.info(
            f"Validación completada: verdict={verdict}, celdas imputadas="
            f"{n_imputed} ({n_imputed_pct:.2f}%), "
            f"checks: {sum(1 for c in checks if c.status == 'ok')} ok, "
            f"{sum(1 for c in checks if c.status == 'warn')} warn, "
            f"{sum(1 for c in checks if c.status == 'error')} error."
        )

        return ImputationValidationReport(
            verdict=verdict,
            n_imputed_cells=n_imputed,
            n_imputed_percentage=round(n_imputed_pct, 4),
            residual_missing=residual,
            checks=checks,
            warnings=warnings,
            reproduction=reproduction,
        )

    # ------------------------------------------------------------------
    # Comprobaciones
    # ------------------------------------------------------------------

    @staticmethod
    def _check_residual_missing(original: pd.DataFrame, imputed: pd.DataFrame) -> ValidationCheck:
        residual: Dict[str, int] = {}
        for col in original.columns:
            if col in imputed.columns:
                count = int(imputed[col].isna().sum())
            else:
                count = len(original)
            if count > 0:
                residual[col] = count
        if residual:
            total = sum(residual.values())
            return ValidationCheck(
                name="faltantes_residuales",
                status="error",
                passed=False,
                message=(
                    f"Hay {total} celdas con faltantes residuales tras la imputación "
                    f"en {len(residual)} columna(s)."
                ),
                details={"columnas": residual},
            )
        return ValidationCheck(
            name="faltantes_residuales",
            status="ok",
            passed=True,
            message="No quedan celdas con faltantes tras la imputación.",
            details={"columnas": {}},
        )

    @staticmethod
    def _check_dtype_preservation(original: pd.DataFrame, imputed: pd.DataFrame) -> ValidationCheck:
        changed: Dict[str, Dict[str, str]] = {}
        for col in original.columns:
            if col not in imputed.columns:
                continue
            before = str(original[col].dtype)
            after = str(imputed[col].dtype)
            if before != after:
                changed[col] = {"antes": before, "despues": after}
        if changed:
            return ValidationCheck(
                name="dtype_preservacion",
                status="warn",
                passed=False,
                message=(
                    f"El dtype cambió en {len(changed)} columna(s) tras la imputación."
                ),
                details={"columnas": changed},
            )
        return ValidationCheck(
            name="dtype_preservacion",
            status="ok",
            passed=True,
            message="Todos los dtypes de las columnas se preservaron.",
            details={"columnas": {}},
        )

    def _check_impossible_values(
        self,
        original: pd.DataFrame,
        imputed: pd.DataFrame,
        mask: pd.DataFrame,
    ) -> ValidationCheck:
        violations: Dict[str, Any] = {}
        for col in original.columns:
            if col not in imputed.columns:
                continue
            series = original[col]
            n_imputed = int(mask[col].sum())
            if n_imputed == 0:
                continue
            observed = series.dropna()
            if _is_numeric_series(series):
                if observed.empty:
                    violations[col] = {
                        "n_violaciones": 0,
                        "nota": "Sin valores observados: dominio desconocido, no verificable.",
                        "ejemplos": [],
                    }
                    continue
                lo = float(observed.min())
                hi = float(observed.max())
                values = imputed[col][mask[col]].astype(float)
                bad = values[(values < lo) | (values > hi)]
                if len(bad) > 0:
                    violations[col] = {
                        "n_violaciones": int(len(bad)),
                        "rango_observado": [lo, hi],
                        "ejemplos": [
                            {"row": int(i), "value": float(v)} for i, v in bad.head(5).items()
                        ],
                    }
            elif _is_categorical_series(series):
                observed_values = set(observed.astype(str))
                if not observed_values:
                    violations[col] = {
                        "n_violaciones": 0,
                        "nota": "Sin valores observados: dominio desconocido, no verificable.",
                        "ejemplos": [],
                    }
                    continue
                values = imputed[col][mask[col]].astype(str)
                bad = values[~values.isin(observed_values)]
                if len(bad) > 0:
                    violations[col] = {
                        "n_violaciones": int(len(bad)),
                        "ejemplos": [
                            {"row": int(i), "value": str(v)} for i, v in bad.head(5).items()
                        ],
                    }
        if violations:
            return ValidationCheck(
                name="valores_imposibles",
                status="error",
                passed=False,
                message=(
                    f"Se detectaron valores imputados imposibles en {len(violations)} "
                    "columna(s) (fuera del dominio observado)."
                ),
                details={"columnas": violations},
            )
        return ValidationCheck(
            name="valores_imposibles",
            status="ok",
            passed=True,
            message="No se detectaron valores imputados fuera del dominio observado.",
            details={"columnas": {}},
        )

    def _check_distribution(
        self,
        original: pd.DataFrame,
        imputed: pd.DataFrame,
        mask: pd.DataFrame,
    ) -> ValidationCheck:
        column_details: Dict[str, Any] = {}
        any_warn = False
        for col in original.columns:
            if col not in imputed.columns:
                continue
            series = original[col]
            n_imputed = int(mask[col].sum())
            if n_imputed == 0:
                continue
            if not _is_numeric_series(series):
                continue
            observed = series.dropna().astype(float)
            imputed_vals = imputed[col][mask[col]].astype(float)
            col_warns: List[str] = []
            entry: Dict[str, Any] = {}

            if len(observed) >= self.min_ks_sample and len(imputed_vals) >= self.min_ks_sample:
                ks = ks_2samp(observed.to_numpy(), imputed_vals.to_numpy())
                entry["ks_statistic"] = float(ks.statistic)
                entry["ks_pvalue"] = float(ks.pvalue)
                if ks.pvalue < self.ks_alpha:
                    col_warns.append(
                        f"La distribución de los valores imputados difiere de la observada "
                        f"(KS p={ks.pvalue:.4g} < {self.ks_alpha})."
                    )
                    any_warn = True
            else:
                entry["ks_statistic"] = None
                entry["ks_pvalue"] = None
                entry["nota"] = (
                    "Muestra insuficiente para la prueba KS "
                    f"(observados={len(observed)}, imputados={len(imputed_vals)})."
                )

            mean_before = float(observed.mean())
            mean_after = float(imputed[col].astype(float).mean())
            std_before = float(observed.std())
            std_after = float(imputed[col].astype(float).std())
            rel_mean = _relative_change(mean_before, mean_after)
            rel_std = _relative_change(std_before, std_after)
            entry.update(
                {
                    "mean_observados": mean_before,
                    "mean_columnas_despues": mean_after,
                    "std_observados": std_before,
                    "std_columnas_despues": std_after,
                    "rel_change_mean": rel_mean,
                    "rel_change_std": rel_std,
                }
            )
            if rel_mean is not None and rel_mean > self.distribution_rel_change:
                col_warns.append(
                    f"La media de la columna cambió {rel_mean:.1%} tras la imputación "
                    f"(umbral {self.distribution_rel_change:.0%})."
                )
                any_warn = True
            if rel_std is not None and rel_std > self.distribution_rel_change:
                col_warns.append(
                    f"La desviación estándar cambió {rel_std:.1%} tras la imputación "
                    f"(umbral {self.distribution_rel_change:.0%})."
                )
                any_warn = True
            entry["warnings"] = col_warns
            column_details[col] = entry

        if any_warn:
            return ValidationCheck(
                name="distribucion_ks",
                status="warn",
                passed=False,
                message=(
                    "Se detectaron cambios importantes de distribución en al menos una "
                    "columna numérica (KS y/o estadísticas descriptivas)."
                ),
                details={"columnas": column_details},
            )
        return ValidationCheck(
            name="distribucion_ks",
            status="ok",
            passed=True,
            message="No se detectaron cambios importantes de distribución.",
            details={"columnas": column_details},
        )

    def _check_correlations(
        self,
        original: pd.DataFrame,
        imputed: pd.DataFrame,
    ) -> ValidationCheck:
        numeric = [
            col for col in original.columns
            if col in imputed.columns and _is_numeric_series(original[col])
        ]
        if len(numeric) < 2:
            return ValidationCheck(
                name="correlaciones",
                status="ok",
                passed=True,
                message="No hay suficientes columnas numéricas para comparar correlaciones.",
                details={"n_pares": 0, "pares_cambios": [], "max_abs_diff": None,
                         "mean_abs_diff": None},
            )
        before = original[numeric].corr()
        after = imputed[numeric].corr()
        changes: List[Dict[str, Any]] = []
        diffs: List[float] = []
        n_pairs = 0
        for i in range(len(numeric)):
            for j in range(i + 1, len(numeric)):
                r1 = before.iloc[i, j]
                r2 = after.iloc[i, j]
                if pd.isna(r1) or pd.isna(r2):
                    continue
                n_pairs += 1
                diff = abs(float(r1) - float(r2))
                diffs.append(diff)
                if diff > self.corr_threshold:
                    changes.append(
                        {
                            "par": f"{numeric[i]} | {numeric[j]}",
                            "r_antes": float(r1),
                            "r_despues": float(r2),
                            "diferencia": round(diff, 6),
                        }
                    )
        max_diff = round(max(diffs), 6) if diffs else None
        mean_diff = round(sum(diffs) / len(diffs), 6) if diffs else None
        if changes:
            return ValidationCheck(
                name="correlaciones",
                status="warn",
                passed=False,
                message=(
                    f"Se detectaron cambios importantes en {len(changes)} par(es) de "
                    f"correlaciones (umbral {self.corr_threshold})."
                ),
                details={
                    "n_pares": n_pairs,
                    "pares_cambios": changes,
                    "max_abs_diff": max_diff,
                    "mean_abs_diff": mean_diff,
                },
            )
        return ValidationCheck(
            name="correlaciones",
            status="ok",
            passed=True,
            message="No se detectaron cambios importantes en las correlaciones.",
            details={
                "n_pares": n_pairs,
                "pares_cambios": changes,
                "max_abs_diff": max_diff,
                "mean_abs_diff": mean_diff,
            },
        )

    def _check_proportion(
        self,
        original: pd.DataFrame,
        imputed: pd.DataFrame,
        mask: pd.DataFrame,
    ) -> ValidationCheck:
        n_imputed = int(mask.to_numpy().sum())
        n_total = original.size
        overall = (n_imputed / n_total) if n_total else 0.0
        per_column = {
            col: round(int(mask[col].sum()) / len(original), 6)
            for col in original.columns
            if col in mask.columns and int(mask[col].sum()) > 0
        }
        high = [
            col for col, p in per_column.items()
            if p > self.high_imputation_threshold
        ]
        if high:
            return ValidationCheck(
                name="proporcion_imputada",
                status="warn",
                passed=False,
                message=(
                    f"La proporción de celdas imputadas supera el {self.high_imputation_threshold:.0%} "
                    f"en {len(high)} columna(s); la imputación es especulativa."
                ),
                details={
                    "n_imputadas": n_imputed,
                    "n_total": n_total,
                    "proporcion_global": round(overall, 6),
                    "por_columna": per_column,
                },
            )
        return ValidationCheck(
            name="proporcion_imputada",
            status="ok",
            passed=True,
            message="La proporción de celdas imputadas no supera el umbral en ninguna columna.",
            details={
                "n_imputadas": n_imputed,
                "n_total": n_total,
                "proporcion_global": round(overall, 6),
                "por_columna": per_column,
            },
        )

    def _check_imputed_vs_observed(
        self,
        original: pd.DataFrame,
        imputed: pd.DataFrame,
        mask: pd.DataFrame,
    ) -> ValidationCheck:
        column_details: Dict[str, Any] = {}
        any_warn = False
        for col in original.columns:
            if col not in imputed.columns:
                continue
            n_imputed = int(mask[col].sum())
            if n_imputed == 0:
                continue
            observed = original[col].dropna()
            imputed_vals = imputed[col][mask[col]]
            entry: Dict[str, Any] = {}
            col_warns: List[str] = []

            if _is_numeric_series(original[col]):
                mean_obs = float(observed.astype(float).mean())
                mean_imp = float(imputed_vals.astype(float).mean())
                rel_bias = _relative_change(mean_obs, mean_imp)
                entry["mean_observados"] = mean_obs
                entry["mean_imputados"] = mean_imp
                entry["rel_bias_mean"] = rel_bias
                if rel_bias is not None and rel_bias > self.bias_threshold:
                    col_warns.append(
                        f"La media de los valores imputados difiere {rel_bias:.1%} de la "
                        f"de los observados (umbral {self.bias_threshold:.0%})."
                    )
                    any_warn = True
            elif _is_categorical_series(original[col]):
                mode_obs = str(observed.astype(str).mode().iloc[0]) if not observed.empty else None
                mode_imp = (
                    str(imputed_vals.astype(str).mode().iloc[0])
                    if not imputed_vals.empty else None
                )
                entry["moda_observados"] = mode_obs
                entry["moda_imputados"] = mode_imp
                if mode_obs is not None and mode_imp is not None and mode_obs != mode_imp:
                    col_warns.append(
                        f"La moda de los valores imputados ('{mode_imp}') difiere de la "
                        f"observada ('{mode_obs}')."
                    )
                    any_warn = True
            entry["warnings"] = col_warns
            column_details[col] = entry

        if any_warn:
            return ValidationCheck(
                name="comparacion_imputados_vs_observados",
                status="warn",
                passed=False,
                message=(
                    "Los valores imputados difieren sustancialmente de los observados "
                    "en al menos una columna."
                ),
                details={"columnas": column_details},
            )
        return ValidationCheck(
            name="comparacion_imputados_vs_observados",
            status="ok",
            passed=True,
            message="Los valores imputados son coherentes con los observados.",
            details={"columnas": column_details},
        )
