"""
Módulo para pruebas de hipótesis estadísticas frecuentistas.
"""

from typing import Dict, Any, Union, List, Tuple, Optional
import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.api import add_constant
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.diagnostic import (
    het_breuschpagan,
    het_white,
    acorr_breusch_godfrey,
    linear_reset,
    lilliefors as sm_lilliefors,
)
from statsmodels.stats.stattools import durbin_watson as sm_durbin_watson
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from src.utils.logger import setup_logger

logger = setup_logger("HypothesisTests")


def _clean_sample(sample: Union[pd.Series, np.ndarray, list]) -> np.ndarray:
    """Función auxiliar para limpiar datos eliminando valores nulos o infinitos."""
    if isinstance(sample, pd.Series):
        arr = sample.replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
    else:
        arr = np.asarray(sample)
        arr = arr[~np.isnan(arr) & ~np.isinf(arr)]
    return arr


def _align_paired_samples(
    sample1: Union[pd.Series, np.ndarray, list],
    sample2: Union[pd.Series, np.ndarray, list],
) -> Tuple[np.ndarray, np.ndarray]:
    """Alinea dos muestras pareadas por índice, eliminando conjuntamente los
    pares con valores nulos o infinitos para preservar el emparejamiento."""
    if isinstance(sample1, pd.Series):
        s1 = sample1.to_numpy()
    else:
        s1 = np.asarray(sample1)
    if isinstance(sample2, pd.Series):
        s2 = sample2.to_numpy()
    else:
        s2 = np.asarray(sample2)

    if len(s1) != len(s2):
        raise ValueError("Las muestras pareadas deben tener la misma cantidad de observaciones.")

    valid = ~(np.isnan(s1) | np.isnan(s2) | np.isinf(s1) | np.isinf(s2))
    return s1[valid], s2[valid]


def _ensure_positive_variance(sample: np.ndarray, test_name: str) -> None:
    """Valida que la muestra tenga varianza positiva (no sea constante)."""
    if len(sample) > 0 and np.ptp(sample) == 0:
        raise ValueError(
            f"La prueba {test_name} requiere varianza positiva; la muestra es constante."
        )


def _prepare_diagnostic_inputs(
    dependent: Union[pd.Series, np.ndarray, list],
    exog: Union[pd.DataFrame, np.ndarray, list],
) -> Tuple[np.ndarray, np.ndarray]:
    """Normaliza los datos de los diagnósticos de regresión (Breusch-Pagan, White,
    Breusch-Godfrey, RESET): convierte a float, acepta `exog` 1D o 2D y elimina
    conjuntamente las filas con valores nulos o infinitos para preservar la
    alineación entre la variable dependiente y las explicativas.
    """
    dependent_arr = np.asarray(dependent, dtype=float)
    exog_arr = np.asarray(exog, dtype=float)
    if exog_arr.ndim == 1:
        exog_arr = exog_arr.reshape(-1, 1)

    if len(dependent_arr) != len(exog_arr):
        raise ValueError(
            "La variable dependiente y las variables explicativas deben tener la misma longitud."
        )

    valid = ~(
        np.isnan(dependent_arr)
        | np.isinf(dependent_arr)
        | np.isnan(exog_arr).any(axis=1)
        | np.isinf(exog_arr).any(axis=1)
    )
    return dependent_arr[valid], exog_arr[valid]


def _build_test_result(
    test_name: str,
    statistic: float,
    p_value: float,
    alpha: float,
    null_hypothesis: str,
    alt_hypothesis: str,
    reject_h0: bool,
    interpretation: str,
) -> Dict[str, Any]:
    """Construye el diccionario de resultados estandarizado de una prueba de hipótesis."""
    decision = "Rechazar H0" if reject_h0 else "No rechazar H0"

    return {
        "test_name": test_name,
        "statistic": statistic,
        "p_value": p_value,
        "alpha": alpha,
        "null_hypothesis": null_hypothesis,
        "alt_hypothesis": alt_hypothesis,
        "decision": decision,
        "reject_h0": reject_h0,
        "interpretation": interpretation,
    }


def shapiro_wilk_test(
    data: Union[pd.Series, np.ndarray, list], alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Prueba de Shapiro-Wilk para evaluar el supuesto de normalidad en una muestra.

    H0: La muestra proviene de una distribución normal.
    H1: La muestra no proviene de una distribución normal.

    Args:
        data: Serie, array o lista con los datos numéricos.
        alpha: Nivel de significancia (por defecto 0.05).

    Returns:
        Dict estructurado con el estadístico, p-value, decisión e interpretación.
    """
    arr = _clean_sample(data)
    if len(arr) < 3:
        raise ValueError("La prueba de Shapiro-Wilk requiere al menos 3 observaciones.")

    stat, p_val = stats.shapiro(arr)
    reject_h0 = bool(p_val < alpha)

    interpretation = (
        f"Con un p-valor de {p_val:.4e} y un nivel de significancia alpha={alpha}, "
        f"{'existe' if reject_h0 else 'no existe'} evidencia suficiente para rechazar la hipótesis de normalidad."
    )

    return _build_test_result(
        test_name="Shapiro-Wilk (Normalidad)",
        statistic=float(stat),
        p_value=float(p_val),
        alpha=float(alpha),
        null_hypothesis="H0: La muestra proviene de una población con distribución normal.",
        alt_hypothesis="H1: La muestra no proviene de una población con distribución normal.",
        reject_h0=reject_h0,
        interpretation=interpretation,
    )


def levene_test(
    *samples: Union[pd.Series, np.ndarray, list],
    center: str = "median",
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Prueba de Levene para evaluar la homogeneidad de varianzas (homocedasticidad) entre dos o más muestras.

    H0: Las varianzas poblacionales de los grupos son iguales.
    H1: Al menos una varianza poblacional es diferente.

    Args:
        *samples: Dos o más muestras numéricas.
        center: Método de centralidad ('median' recomendado, 'mean', 'trimmed').
        alpha: Nivel de significancia (por defecto 0.05).

    Returns:
        Dict estructurado con el resultado de la prueba.
    """
    cleaned_samples = [_clean_sample(s) for s in samples]
    if len(cleaned_samples) < 2:
        raise ValueError("La prueba de Levene requiere al menos 2 muestras.")

    stat, p_val = stats.levene(*cleaned_samples, center=center)
    reject_h0 = bool(p_val < alpha)

    interpretation = (
        f"Con un p-valor de {p_val:.4e} y alpha={alpha}, "
        f"{'existe' if reject_h0 else 'no existe'} evidencia estadística de heterocedasticidad (diferencia de varianzas)."
    )

    return _build_test_result(
        test_name="Levene (Homogeneidad de Varianzas)",
        statistic=float(stat),
        p_value=float(p_val),
        alpha=float(alpha),
        null_hypothesis="H0: Las varianzas de los grupos son iguales (homocedasticidad).",
        alt_hypothesis="H1: Al menos una varianza difiere (heterocedasticidad).",
        reject_h0=reject_h0,
        interpretation=interpretation,
    )


def t_test_1samp(
    sample: Union[pd.Series, np.ndarray, list],
    popmean: float,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Prueba t de Student para 1 muestra respecto a una media poblacional de referencia.

    H0: La media poblacional es igual a popmean (mu = popmean).
    H1: La media poblacional es diferente de popmean (mu != popmean).
    """
    s = _clean_sample(sample)
    if len(s) < 2:
        raise ValueError("La prueba t de 1 muestra requiere al menos 2 observaciones.")
    _ensure_positive_variance(s, "t de Student de 1 muestra")

    stat, p_val = stats.ttest_1samp(s, popmean)
    reject_h0 = bool(p_val < alpha)

    interpretation = (
        f"Con un p-valor de {p_val:.4e} y alpha={alpha}, "
        f"{'existe' if reject_h0 else 'no existe'} evidencia estadística de que la media difiere de {popmean}."
    )

    return _build_test_result(
        test_name="t de Student (1 Muestra)",
        statistic=float(stat),
        p_value=float(p_val),
        alpha=float(alpha),
        null_hypothesis=f"H0: La media poblacional es igual a {popmean}.",
        alt_hypothesis=f"H1: La media poblacional difiere de {popmean}.",
        reject_h0=reject_h0,
        interpretation=interpretation,
    )


def t_test_ind(
    sample1: Union[pd.Series, np.ndarray, list],
    sample2: Union[pd.Series, np.ndarray, list],
    equal_var: bool = True,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Prueba t de Student para dos muestras independientes.

    H0: Las medias de ambos grupos son iguales (mu1 = mu2).
    H1: Las medias de ambos grupos son diferentes (mu1 != mu2).
    """
    s1 = _clean_sample(sample1)
    s2 = _clean_sample(sample2)

    if len(s1) < 2 or len(s2) < 2:
        raise ValueError("Cada muestra debe tener al menos 2 observaciones para la prueba t.")
    _ensure_positive_variance(s1, "t de Student de dos muestras")
    _ensure_positive_variance(s2, "t de Student de dos muestras")

    stat, p_val = stats.ttest_ind(s1, s2, equal_var=equal_var)
    reject_h0 = bool(p_val < alpha)

    test_type = "t de Student" if equal_var else "t de Welch"
    interpretation = (
        f"Con un p-valor de {p_val:.4e} y alpha={alpha}, "
        f"{'existe' if reject_h0 else 'no existe'} evidencia estadísticamente significativa para afirmar que las medias difieren."
    )

    return _build_test_result(
        test_name=f"{test_type} (Muestras Independientes)",
        statistic=float(stat),
        p_value=float(p_val),
        alpha=float(alpha),
        null_hypothesis="H0: Las medias poblacionales de ambos grupos son iguales.",
        alt_hypothesis="H1: Las medias poblacionales de ambos grupos son diferentes.",
        reject_h0=reject_h0,
        interpretation=interpretation,
    )


def t_test_rel(
    sample1: Union[pd.Series, np.ndarray, list],
    sample2: Union[pd.Series, np.ndarray, list],
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Prueba t de Student para dos muestras pareadas o relacionadas.

    H0: La diferencia media entre las observaciones pareadas es cero.
    H1: La diferencia media difiere de cero.
    """
    s1, s2 = _align_paired_samples(sample1, sample2)

    if len(s1) < 2:
        raise ValueError("Se requieren al menos 2 parejas válidas de datos.")
    _ensure_positive_variance(s1 - s2, "t de Student de muestras pareadas")

    stat, p_val = stats.ttest_rel(s1, s2)
    reject_h0 = bool(p_val < alpha)

    interpretation = (
        f"Con un p-valor de {p_val:.4e} y alpha={alpha}, "
        f"{'existe' if reject_h0 else 'no existe'} evidencia estadística significativa de diferencia media entre las mediciones pareadas."
    )

    return _build_test_result(
        test_name="t de Student (Muestras Pareadas)",
        statistic=float(stat),
        p_value=float(p_val),
        alpha=float(alpha),
        null_hypothesis="H0: La diferencia media entre las observaciones pareadas es cero.",
        alt_hypothesis="H1: La diferencia media difiere de cero.",
        reject_h0=reject_h0,
        interpretation=interpretation,
    )


def wilcoxon_signed_rank(
    sample1: Union[pd.Series, np.ndarray, list],
    sample2: Optional[Union[pd.Series, np.ndarray, list]] = None,
    popmean: float = 0.0,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Prueba no paramétrica de rangos con signo de Wilcoxon (1 muestra o 2 muestras pareadas).

    H0: La mediana (o diferencia de medianas) es igual a cero (o popmean).
    H1: La mediana difiere de cero (o popmean).
    """
    if sample2 is not None:
        s1, s2 = _align_paired_samples(sample1, sample2)
        diff = s1 - s2
    else:
        s1 = _clean_sample(sample1)
        diff = s1 - popmean

    diff = diff[diff != 0]  # Eliminar diferencias nulas según estándar Wilcoxon
    if len(diff) < 5:
        raise ValueError("La prueba de Wilcoxon requiere al menos 5 diferencias no nulas.")

    stat, p_val = stats.wilcoxon(diff)
    reject_h0 = bool(p_val < alpha)

    test_label = "Wilcoxon Pareado" if sample2 is not None else "Wilcoxon de 1 Muestra"
    interpretation = (
        f"Con un p-valor de {p_val:.4e} y alpha={alpha}, "
        f"{'existe' if reject_h0 else 'no existe'} evidencia estadística significativa de diferencia en la mediana."
    )

    return _build_test_result(
        test_name=f"{test_label} (No Paramétrica)",
        statistic=float(stat),
        p_value=float(p_val),
        alpha=float(alpha),
        null_hypothesis="H0: La mediana de las diferencias es igual a cero.",
        alt_hypothesis="H1: La mediana de las diferencias difiere de cero.",
        reject_h0=reject_h0,
        interpretation=interpretation,
    )


def mann_whitney_test(
    sample1: Union[pd.Series, np.ndarray, list],
    sample2: Union[pd.Series, np.ndarray, list],
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Prueba no paramétrica de Mann-Whitney U para dos muestras independientes.
    """
    s1 = _clean_sample(sample1)
    s2 = _clean_sample(sample2)

    if len(s1) < 1 or len(s2) < 1:
        raise ValueError("Cada muestra debe tener al menos 1 observación para Mann-Whitney.")

    stat, p_val = stats.mannwhitneyu(s1, s2, alternative="two-sided")
    reject_h0 = bool(p_val < alpha)

    interpretation = (
        f"Con un p-valor de {p_val:.4e} y alpha={alpha}, "
        f"{'existe' if reject_h0 else 'no existe'} evidencia estadística significativa de diferencia en las distribuciones de ambos grupos."
    )

    return _build_test_result(
        test_name="Mann-Whitney U (No Paramétrica)",
        statistic=float(stat),
        p_value=float(p_val),
        alpha=float(alpha),
        null_hypothesis="H0: Las distribuciones poblacionales de ambos grupos son idénticas.",
        alt_hypothesis="H1: Las distribuciones poblacionales difieren.",
        reject_h0=reject_h0,
        interpretation=interpretation,
    )


def anova_one_way(
    *samples: Union[pd.Series, np.ndarray, list], alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Prueba de Análisis de Varianza de un factor (One-Way ANOVA) para k muestras independientes.
    """
    cleaned_samples = [_clean_sample(s) for s in samples]
    if len(cleaned_samples) < 2:
        raise ValueError("ANOVA requiere al menos 2 grupos para comparar.")

    stat, p_val = stats.f_oneway(*cleaned_samples)
    reject_h0 = bool(p_val < alpha)

    interpretation = (
        f"Con un p-valor de {p_val:.4e} y alpha={alpha}, "
        f"{'existe' if reject_h0 else 'no existe'} evidencia estadística suficiente de que al menos una media grupal difiere."
    )

    return _build_test_result(
        test_name="ANOVA de un factor (One-Way ANOVA)",
        statistic=float(stat),
        p_value=float(p_val),
        alpha=float(alpha),
        null_hypothesis="H0: Las medias poblacionales de todos los grupos son iguales.",
        alt_hypothesis="H1: Al menos la media de un grupo es diferente.",
        reject_h0=reject_h0,
        interpretation=interpretation,
    )


def welch_anova(
    *samples: Union[pd.Series, np.ndarray, list], alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Prueba de ANOVA de Welch de 1 factor para k muestras independientes con varianzas desiguales.

    H0: Las medias de todos los grupos son iguales.
    H1: Al menos la media de un grupo es diferente.
    """
    cleaned = [_clean_sample(s) for s in samples]
    k = len(cleaned)
    if k < 2:
        raise ValueError("ANOVA de Welch requiere al menos 2 grupos.")

    ns = np.array([len(s) for s in cleaned])
    vars_ = np.array([np.var(s, ddof=1) for s in cleaned])
    means = np.array([np.mean(s) for s in cleaned])

    if any(ns < 2):
        raise ValueError("Cada grupo debe tener al menos 2 observaciones.")
    if any(vars_ == 0):
        raise ValueError("Varianzas nulas detectadas en uno de los grupos.")

    weights = ns / vars_
    w_total = np.sum(weights)
    mean_w = np.sum(weights * means) / w_total

    num = np.sum(weights * (means - mean_w) ** 2) / (k - 1)
    lambda_term = np.sum(((1 - weights / w_total) ** 2) / (ns - 1))
    den = 1 + (2 * (k - 2) / (k ** 2 - 1)) * lambda_term

    f_stat = num / den
    df1 = k - 1
    df2 = (k ** 2 - 1) / (3 * lambda_term)

    p_val = float(stats.f.sf(f_stat, df1, df2))
    reject_h0 = bool(p_val < alpha)

    interpretation = (
        f"Con un p-valor de {p_val:.4e} y alpha={alpha}, "
        f"{'existe' if reject_h0 else 'no existe'} evidencia estadística de diferencia en las medias mediante el ajuste de Welch."
    )

    return _build_test_result(
        test_name="ANOVA de Welch (Varianzas Heterogéneas)",
        statistic=float(f_stat),
        p_value=p_val,
        alpha=float(alpha),
        null_hypothesis="H0: Las medias poblacionales de todos los grupos son iguales.",
        alt_hypothesis="H1: Al menos la media de un grupo es diferente.",
        reject_h0=reject_h0,
        interpretation=interpretation,
    )


def kruskal_wallis_test(
    *samples: Union[pd.Series, np.ndarray, list], alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Prueba no paramétrica de Kruskal-Wallis H para k muestras independientes.
    """
    cleaned_samples = [_clean_sample(s) for s in samples]
    if len(cleaned_samples) < 2:
        raise ValueError("Kruskal-Wallis requiere al menos 2 grupos para comparar.")

    stat, p_val = stats.kruskal(*cleaned_samples)
    reject_h0 = bool(p_val < alpha)

    interpretation = (
        f"Con un p-valor de {p_val:.4e} y alpha={alpha}, "
        f"{'existe' if reject_h0 else 'no existe'} evidencia estadística significativa de diferencia entre los grupos."
    )

    return _build_test_result(
        test_name="Kruskal-Wallis H (No Paramétrica)",
        statistic=float(stat),
        p_value=float(p_val),
        alpha=float(alpha),
        null_hypothesis="H0: Las medianas poblacionales de todos los grupos son iguales.",
        alt_hypothesis="H1: Al menos la mediana de un grupo es diferente.",
        reject_h0=reject_h0,
        interpretation=interpretation,
    )


# ============================================================================
# Pruebas estadísticas adicionales (Roadmap §2.2)
# ============================================================================


def kolmogorov_smirnov_1samp_test(
    data: Union[pd.Series, np.ndarray, list],
    cdf: str = "norm",
    args: Tuple[float, ...] = (),
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Prueba de Kolmogorov-Smirnov de 1 muestra (bondad de ajuste).

    H0: Los datos siguen la distribución continua especificada por `cdf`.
    H1: Los datos no siguen dicha distribución.
    """
    arr = _clean_sample(data)
    if len(arr) < 1:
        raise ValueError("Kolmogorov-Smirnov de 1 muestra requiere al menos 1 observación.")

    stat, p_val = stats.kstest(arr, cdf, args=args)
    reject_h0 = bool(p_val < alpha)

    interpretation = (
        f"Con un p-valor de {p_val:.4e} y alpha={alpha}, "
        f"{'existe' if reject_h0 else 'no existe'} evidencia suficiente para afirmar que los datos "
        f"no siguen la distribución {cdf}."
    )

    return _build_test_result(
        test_name="Kolmogorov-Smirnov (1 Muestra)",
        statistic=float(stat),
        p_value=float(p_val),
        alpha=float(alpha),
        null_hypothesis=f"H0: Los datos siguen la distribución {cdf}.",
        alt_hypothesis=f"H1: Los datos no siguen la distribución {cdf}.",
        reject_h0=reject_h0,
        interpretation=interpretation,
    )


def kolmogorov_smirnov_2samp_test(
    sample1: Union[pd.Series, np.ndarray, list],
    sample2: Union[pd.Series, np.ndarray, list],
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Prueba de Kolmogorov-Smirnov de 2 muestras independientes.

    H0: Ambas muestras provienen de la misma distribución continua.
    H1: Las muestras provienen de distribuciones distintas.
    """
    s1 = _clean_sample(sample1)
    s2 = _clean_sample(sample2)

    if len(s1) < 1 or len(s2) < 1:
        raise ValueError("Kolmogorov-Smirnov de 2 muestras requiere al menos 1 observación por muestra.")

    stat, p_val = stats.ks_2samp(s1, s2)
    reject_h0 = bool(p_val < alpha)

    interpretation = (
        f"Con un p-valor de {p_val:.4e} y alpha={alpha}, "
        f"{'existe' if reject_h0 else 'no existe'} evidencia estadística de que las distribuciones de ambas muestras difieren."
    )

    return _build_test_result(
        test_name="Kolmogorov-Smirnov (2 Muestras)",
        statistic=float(stat),
        p_value=float(p_val),
        alpha=float(alpha),
        null_hypothesis="H0: Ambas muestras provienen de la misma distribución.",
        alt_hypothesis="H1: Las muestras provienen de distribuciones distintas.",
        reject_h0=reject_h0,
        interpretation=interpretation,
    )


def lilliefors_test(
    data: Union[pd.Series, np.ndarray, list], alpha: float = 0.05
) -> Dict[str, Any]:
    """Prueba de Lilliefors de normalidad (Kolmogorov-Smirnov con parámetros estimados).

    H0: Los datos provienen de una distribución normal.
    H1: Los datos no provienen de una distribución normal.
    """
    arr = _clean_sample(data)
    if len(arr) < 4:
        raise ValueError("La prueba de Lilliefors requiere al menos 4 observaciones.")

    stat, p_val = sm_lilliefors(arr, dist="norm")
    reject_h0 = bool(p_val < alpha)

    interpretation = (
        f"Con un p-valor de {p_val:.4e} y alpha={alpha}, "
        f"{'existe' if reject_h0 else 'no existe'} evidencia suficiente para rechazar la normalidad de los datos."
    )

    return _build_test_result(
        test_name="Lilliefors (Normalidad)",
        statistic=float(stat),
        p_value=float(p_val),
        alpha=float(alpha),
        null_hypothesis="H0: Los datos provienen de una distribución normal.",
        alt_hypothesis="H1: Los datos no provienen de una distribución normal.",
        reject_h0=reject_h0,
        interpretation=interpretation,
    )


def bartlett_test(
    *samples: Union[pd.Series, np.ndarray, list], alpha: float = 0.05
) -> Dict[str, Any]:
    """Prueba de Bartlett para la homogeneidad de varianzas de k muestras.

    H0: Las varianzas poblacionales de todos los grupos son iguales.
    H1: Al menos una varianza difiere.

    Supone normalidad de los datos; es sensible a desviaciones de este supuesto
    (preferir Levene si no se cumple normalidad).
    """
    cleaned = [_clean_sample(s) for s in samples]
    if len(cleaned) < 2:
        raise ValueError("La prueba de Bartlett requiere al menos 2 grupos.")

    for s in cleaned:
        if len(s) < 2:
            raise ValueError("Cada grupo debe tener al menos 2 observaciones para la prueba de Bartlett.")
        _ensure_positive_variance(s, "Bartlett")

    stat, p_val = stats.bartlett(*cleaned)
    reject_h0 = bool(p_val < alpha)

    interpretation = (
        f"Con un p-valor de {p_val:.4e} y alpha={alpha}, "
        f"{'existe' if reject_h0 else 'no existe'} evidencia estadística de que las varianzas de los grupos difieren."
    )

    return _build_test_result(
        test_name="Bartlett (Homogeneidad de Varianzas)",
        statistic=float(stat),
        p_value=float(p_val),
        alpha=float(alpha),
        null_hypothesis="H0: Las varianzas poblacionales de todos los grupos son iguales.",
        alt_hypothesis="H1: Al menos una varianza difiere.",
        reject_h0=reject_h0,
        interpretation=interpretation,
    )


def breusch_pagan_test(
    residuals: Union[pd.Series, np.ndarray, list],
    exog: Union[pd.DataFrame, np.ndarray, list],
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Prueba de Breusch-Pagan para heterocedasticidad en una regresión lineal.

    H0: Homocedasticidad (las varianzas de los errores son constantes).
    H1: Heterocedasticidad.

    `exog` corresponde a las variables explicativas del modelo (sin constante;
    esta se añade internamente). Las filas con valores nulos o infinitos en los
    residuos o en `exog` se eliminan conjuntamente. Retorna además el estadístico
    F y su p-valor en las claves extendidas ``f_statistic`` y ``f_p_value``.
    """
    resid, exog_arr = _prepare_diagnostic_inputs(residuals, exog)
    k = exog_arr.shape[1]

    if len(resid) < k + 2:
        raise ValueError(
            f"Observaciones completas insuficientes para Breusch-Pagan: se requieren al menos "
            f"{k + 2} (k + 2) con {k} variable(s) explicativa(s)."
        )

    lm, lmpval, fval, fpval = het_breuschpagan(resid, add_constant(exog_arr))
    reject_h0 = bool(lmpval < alpha)

    interpretation = (
        f"Con un p-valor de {lmpval:.4e} y alpha={alpha}, "
        f"{'existe' if reject_h0 else 'no existe'} evidencia estadística de heterocedasticidad "
        f"(estadístico F: {fval:.4e}, p-valor F: {fpval:.4e})."
    )

    result = _build_test_result(
        test_name="Breusch-Pagan (Heterocedasticidad)",
        statistic=float(lm),
        p_value=float(lmpval),
        alpha=float(alpha),
        null_hypothesis="H0: Homocedasticidad (varianzas de los errores constantes).",
        alt_hypothesis="H1: Heterocedasticidad.",
        reject_h0=reject_h0,
        interpretation=interpretation,
    )
    result["f_statistic"] = float(fval)
    result["f_p_value"] = float(fpval)
    return result


def white_test(
    residuals: Union[pd.Series, np.ndarray, list],
    exog: Union[pd.DataFrame, np.ndarray, list],
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Prueba de White para heterocedasticidad (general, sin suponer una forma funcional).

    H0: Homocedasticidad.
    H1: Heterocedasticidad.

    `exog` corresponde a las variables explicativas del modelo (sin constante;
    esta se añade internamente). Las filas con valores nulos o infinitos en los
    residuos o en `exog` se eliminan conjuntamente. Retorna además el estadístico
    F y su p-valor en las claves extendidas ``f_statistic`` y ``f_p_value``.
    """
    resid, exog_arr = _prepare_diagnostic_inputs(residuals, exog)
    k = exog_arr.shape[1]
    min_obs = k + k * (k + 1) // 2 + 2

    if len(resid) < min_obs:
        raise ValueError(
            f"Observaciones completas insuficientes para la prueba de White: se requieren al menos "
            f"{min_obs} con {k} variable(s) explicativa(s) (la regresión auxiliar añade cuadrados y productos)."
        )

    lm, lmpval, fval, fpval = het_white(resid, add_constant(exog_arr))
    reject_h0 = bool(lmpval < alpha)

    interpretation = (
        f"Con un p-valor de {lmpval:.4e} y alpha={alpha}, "
        f"{'existe' if reject_h0 else 'no existe'} evidencia estadística de heterocedasticidad "
        f"(estadístico F: {fval:.4e}, p-valor F: {fpval:.4e})."
    )

    result = _build_test_result(
        test_name="White (Heterocedasticidad)",
        statistic=float(lm),
        p_value=float(lmpval),
        alpha=float(alpha),
        null_hypothesis="H0: Homocedasticidad.",
        alt_hypothesis="H1: Heterocedasticidad.",
        reject_h0=reject_h0,
        interpretation=interpretation,
    )
    result["f_statistic"] = float(fval)
    result["f_p_value"] = float(fpval)
    return result


def durbin_watson_test(
    residuals: Union[pd.Series, np.ndarray, list], alpha: float = 0.05
) -> Dict[str, Any]:
    """Estadístico de Durbin-Watson para autocorrelación de primer orden en residuos.

    H0: No hay autocorrelación de primer orden (DW ≈ 2).
    H1: Existe autocorrelación de primer orden.

    Esta implementación no calcula un p-valor exacto; la decisión se basa en la
    regla práctica estándar DW < 1.5 o DW > 2.5 (para muestras moderadas).
    """
    resid = _clean_sample(residuals)
    if len(resid) < 3:
        raise ValueError("Se requieren al menos 3 residuos para Durbin-Watson.")

    dw = float(sm_durbin_watson(resid))
    reject_h0 = bool(dw < 1.5 or dw > 2.5)

    interpretation = (
        f"El estadístico de Durbin-Watson es {dw:.4f}. Valores cercanos a 2 sugieren ausencia de "
        f"autocorrelación de primer orden; por regla práctica se {'rechaza' if reject_h0 else 'no rechaza'} "
        f"H0 (se rechaza si DW < 1.5 o DW > 2.5) con alpha={alpha}."
    )

    return _build_test_result(
        test_name="Durbin-Watson (Autocorrelación de Primer Orden)",
        statistic=dw,
        p_value=None,
        alpha=float(alpha),
        null_hypothesis="H0: No hay autocorrelación de primer orden.",
        alt_hypothesis="H1: Existe autocorrelación de primer orden.",
        reject_h0=reject_h0,
        interpretation=interpretation,
    )


def breusch_godfrey_test(
    y: Union[pd.Series, np.ndarray, list],
    exog: Union[pd.DataFrame, np.ndarray, list],
    nlags: int = 1,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Prueba de Breusch-Godfrey para autocorrelación de errores de orden `nlags`.

    H0: No hay autocorrelación serial de orden hasta `nlags`.
    H1: Existe autocorrelación serial.

    `exog` corresponde a las variables explicativas del modelo (sin constante;
    esta se añade internamente y el modelo se ajusta por MCO). Las filas con
    valores nulos o infinitos en `y` o en `exog` se eliminan conjuntamente.
    Retorna además el estadístico F y su p-valor en las claves extendidas
    ``f_statistic`` y ``f_p_value``.
    """
    if nlags < 1:
        raise ValueError("nlags debe ser al menos 1.")

    y_arr, exog_arr = _prepare_diagnostic_inputs(y, exog)

    if len(y_arr) < nlags + 3:
        raise ValueError("Observaciones completas insuficientes para Breusch-Godfrey con nlags dado.")

    model = OLS(y_arr, add_constant(exog_arr)).fit()
    lm, lmpval, fval, fpval = acorr_breusch_godfrey(model, nlags=nlags)
    reject_h0 = bool(lmpval < alpha)

    interpretation = (
        f"Con un p-valor de {lmpval:.4e} y alpha={alpha}, "
        f"{'existe' if reject_h0 else 'no existe'} evidencia estadística de autocorrelación serial "
        f"hasta el orden {nlags} (estadístico F: {fval:.4e}, p-valor F: {fpval:.4e})."
    )

    result = _build_test_result(
        test_name=f"Breusch-Godfrey (Autocorrelación, Orden {nlags})",
        statistic=float(lm),
        p_value=float(lmpval),
        alpha=float(alpha),
        null_hypothesis=f"H0: No hay autocorrelación serial de orden hasta {nlags}.",
        alt_hypothesis="H1: Existe autocorrelación serial.",
        reject_h0=reject_h0,
        interpretation=interpretation,
    )
    result["f_statistic"] = float(fval)
    result["f_p_value"] = float(fpval)
    return result


def reset_test(
    y: Union[pd.Series, np.ndarray, list],
    exog: Union[pd.DataFrame, np.ndarray, list],
    power: int = 2,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Prueba de especificación RESET (Ramsey) sobre una regresión lineal.

    H0: El modelo lineal está correctamente especificado.
    H1: Hay términos de orden superior omitidos (error de especificación).

    `exog` corresponde a las variables explicativas (sin constante; esta se añade
    internamente). `power` indica la potencia máxima de los valores ajustados que
    se añaden como términos de prueba. Las filas con valores nulos o infinitos en
    `y` o en `exog` se eliminan conjuntamente.
    """
    if power < 2:
        raise ValueError("power debe ser al menos 2.")

    y_arr, exog_arr = _prepare_diagnostic_inputs(y, exog)

    if len(y_arr) < 3:
        raise ValueError("Observaciones completas insuficientes para la prueba RESET.")

    model = OLS(y_arr, add_constant(exog_arr)).fit()
    reset_res = linear_reset(model, power=power, use_f=True)
    fval = float(reset_res.fvalue)
    p_val = float(reset_res.pvalue)
    reject_h0 = bool(p_val < alpha)

    interpretation = (
        f"Con un p-valor de {p_val:.4e} y alpha={alpha}, "
        f"{'existe' if reject_h0 else 'no existe'} evidencia estadística de error de especificación "
        f"(términos de orden superior omitidos)."
    )

    return _build_test_result(
        test_name="RESET de Ramsey (Especificación)",
        statistic=float(fval),
        p_value=float(p_val),
        alpha=float(alpha),
        null_hypothesis="H0: El modelo lineal está correctamente especificado.",
        alt_hypothesis="H1: Existen términos de orden superior omitidos.",
        reject_h0=reject_h0,
        interpretation=interpretation,
    )


def chi_square_test(
    observed: Union[pd.DataFrame, np.ndarray, list],
    expected: Optional[Union[np.ndarray, list]] = None,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Prueba Chi-cuadrado.

    - Si `expected` es ``None`` y `observed` es una tabla de contingencia 2D:
      prueba de independencia (H0: las variables son independientes).
    - Si `expected` se proporciona (vector 1D): prueba de bondad de ajuste
      (H0: las frecuencias observadas se ajustan a las esperadas).
    """
    observed_arr = np.asarray(observed)

    if expected is None:
        if observed_arr.ndim != 2:
            raise ValueError("Para la prueba de independencia se requiere una tabla de contingencia 2D.")
        if (observed_arr < 0).any():
            raise ValueError("La tabla de contingencia no debe contener valores negativos.")
        if (observed_arr.sum(axis=0) == 0).any() or (observed_arr.sum(axis=1) == 0).any():
            raise ValueError("La tabla de contingencia no debe tener filas ni columnas con suma total cero.")
        chi2, p_val, dof, _ = stats.chi2_contingency(observed_arr)
        test_name = "Chi-cuadrado (Independencia)"
        null_hypothesis = "H0: Las variables son independientes."
        alt_hypothesis = "H1: Las variables no son independientes."
    else:
        expected_arr = np.asarray(expected, dtype=float)
        if observed_arr.ndim != 1 or expected_arr.ndim != 1:
            raise ValueError("Para la bondad de ajuste se requieren vectores 1D observados y esperados.")
        if len(observed_arr) != len(expected_arr):
            raise ValueError("Los vectores observados y esperados deben tener la misma longitud.")
        if (expected_arr <= 0).any():
            raise ValueError("Las frecuencias esperadas deben ser estrictamente positivas para la bondad de ajuste.")
        chi2, p_val = stats.chisquare(observed_arr, f_exp=expected_arr)
        test_name = "Chi-cuadrado (Bondad de Ajuste)"
        null_hypothesis = "H0: Las frecuencias observadas se ajustan a las esperadas."
        alt_hypothesis = "H1: Las frecuencias observadas difieren de las esperadas."

    reject_h0 = bool(p_val < alpha)

    interpretation = (
        f"Con un p-valor de {p_val:.4e} y alpha={alpha}, "
        f"{'existe' if reject_h0 else 'no existe'} evidencia estadística para rechazar la hipótesis nula."
    )

    return _build_test_result(
        test_name=test_name,
        statistic=float(chi2),
        p_value=float(p_val),
        alpha=float(alpha),
        null_hypothesis=null_hypothesis,
        alt_hypothesis=alt_hypothesis,
        reject_h0=reject_h0,
        interpretation=interpretation,
    )


def tukey_hsd_test(
    *samples: Union[pd.Series, np.ndarray, list], alpha: float = 0.05
) -> Dict[str, Any]:
    """Prueba post hoc de Tukey HSD para comparaciones múltiples de medias.

    H0: Todas las medias de los pares son iguales.
    H1: Al menos un par de medias difiere.

    El resultado extiende el esquema estándar con la clave ``pairwise_comparisons``,
    que contiene el detalle de cada comparación por pares.
    """
    cleaned = [_clean_sample(s) for s in samples]
    if len(cleaned) < 2:
        raise ValueError("Tukey HSD requiere al menos 2 grupos.")
    for s in cleaned:
        if len(s) < 2:
            raise ValueError("Cada grupo debe tener al menos 2 observaciones para Tukey HSD.")

    endog = np.concatenate(cleaned)
    groups = np.concatenate([np.full(len(s), f"grupo_{i}") for i, s in enumerate(cleaned)])
    res = pairwise_tukeyhsd(endog, groups, alpha=alpha)

    summary_table = res.summary()
    frame = pd.DataFrame(summary_table.data[1:], columns=summary_table.data[0])

    pairwise = []
    for _, row in frame.iterrows():
        pairwise.append(
            {
                "group1": str(row["group1"]),
                "group2": str(row["group2"]),
                "mean_difference": float(row["meandiff"]),
                "p_adjusted": float(row["p-adj"]),
                "ci_lower": float(row["lower"]),
                "ci_upper": float(row["upper"]),
                "significant": bool(row["reject"]),
            }
        )

    significant_pairs = [p for p in pairwise if p["significant"]]
    reject_h0 = bool(significant_pairs)
    min_p = min(p["p_adjusted"] for p in pairwise) if pairwise else float("nan")
    max_abs_diff = max(abs(p["mean_difference"]) for p in pairwise) if pairwise else float("nan")

    interpretation = (
        f"Se compararon {len(cleaned)} grupos por pares (Tukey HSD, alpha={alpha}); "
        f"{len(significant_pairs)} de {len(pairwise)} pares presentaron diferencias estadísticamente significativas."
    )

    result = _build_test_result(
        test_name="Tukey HSD (Comparaciones Múltiples)",
        statistic=float(max_abs_diff),
        p_value=float(min_p),
        alpha=float(alpha),
        null_hypothesis="H0: Todas las medias de los pares son iguales.",
        alt_hypothesis="H1: Al menos un par de medias difiere.",
        reject_h0=reject_h0,
        interpretation=interpretation,
    )
    result["pairwise_comparisons"] = pairwise
    return result


def permutation_test(
    sample1: Union[pd.Series, np.ndarray, list],
    sample2: Union[pd.Series, np.ndarray, list],
    n_permutations: int = 1000,
    alpha: float = 0.05,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Prueba no paramétrica de permutaciones para la diferencia de medias de 2 muestras independientes.

    H0: Las muestras provienen de la misma distribución (diferencia de medias nula).
    H1: La diferencia de medias poblacionales es distinta de cero.

    El p-valor incluye la corrección estándar (+1 en numerador y denominador) para
    incluir el estadístico observado en la distribución nula. `seed` permite fijar
    la semilla del generador para reproducibilidad.
    """
    s1 = _clean_sample(sample1)
    s2 = _clean_sample(sample2)

    if len(s1) < 1 or len(s2) < 1:
        raise ValueError("La prueba de permutaciones requiere al menos 1 observación por muestra.")
    if n_permutations < 1:
        raise ValueError("n_permutations debe ser al menos 1.")

    n1 = len(s1)
    pooled = np.concatenate([s1, s2])
    obs_stat = float(np.mean(s1) - np.mean(s2))

    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_permutations):
        perm = rng.permutation(pooled)
        perm_stat = float(np.mean(perm[:n1]) - np.mean(perm[n1:]))
        if abs(perm_stat) >= abs(obs_stat):
            count += 1

    p_val = (count + 1) / (n_permutations + 1)
    reject_h0 = bool(p_val < alpha)

    interpretation = (
        f"Con un p-valor de {p_val:.4e} (basado en {n_permutations} permutaciones) y alpha={alpha}, "
        f"{'existe' if reject_h0 else 'no existe'} evidencia estadística de diferencia de medias entre los grupos."
    )

    return _build_test_result(
        test_name="Permutaciones (Diferencia de Medias)",
        statistic=obs_stat,
        p_value=p_val,
        alpha=float(alpha),
        null_hypothesis="H0: La diferencia de medias poblacionales es nula.",
        alt_hypothesis="H1: La diferencia de medias poblacionales es distinta de cero.",
        reject_h0=reject_h0,
        interpretation=interpretation,
    )
