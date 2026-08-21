"""
Coordinador y Analizador Estadístico Inteligente para conjuntos de datos (datasets).
Integrador de diagnósticos, supuestos, recomendaciones y ejecución autónoma.
"""

from typing import Dict, Any, Union, Optional, List
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats

from src.data.loader import load_data
from src.data.validator import DataValidator
from src.analysis.eda import (
    describe_numerical,
    describe_categorical,
    detect_outliers_iqr,
    calculate_correlation_matrix,
)
from src.analysis.hypothesis import (
    shapiro_wilk_test,
    levene_test,
    t_test_1samp,
    t_test_ind,
    t_test_rel,
    wilcoxon_signed_rank,
    mann_whitney_test,
    anova_one_way,
    welch_anova,
    kruskal_wallis_test,
    lilliefors_test,
    bartlett_test,
    tukey_hsd_test,
    chi_square_test,
    permutation_test,
)
from src.utils.logger import setup_logger

logger = setup_logger("DatasetStatisticalAnalyzer")


class DatasetStatisticalAnalyzer:
    """
    Coordinador de consultoría estadística que analiza autónoma y rigurosamente
    un conjunto de datos (CSV, Excel o DataFrame).
    """

    def __init__(self, data: Union[str, Path, pd.DataFrame]):
        if isinstance(data, (str, Path)):
            self.df = load_data(data)
            self.file_name = Path(data).name
        elif isinstance(data, pd.DataFrame):
            self.df = data.copy()
            self.file_name = "DataFrame_en_memoria"
        else:
            raise TypeError("Se requiere un archivo CSV/Excel o un DataFrame de pandas.")

        self.quality_summary = DataValidator.summarize_data_quality(self.df)
        self.variable_types = self.quality_summary["variable_types"]

    def analyze(
        self,
        target_col: Optional[str] = None,
        group_col: Optional[str] = None,
        paired_col: Optional[str] = None,
        popmean: Optional[float] = None,
        alpha: float = 0.05,
    ) -> Dict[str, Any]:
        """
        Ejecuta el análisis autónomo de datos, evaluando calidad, EDA, diagnósticos
        de supuestos y recomendaciones metodológicas con justificación formal.
        """
        logger.info(f"Iniciando análisis inteligente para '{self.file_name}'.")

        eda_res = {
            "numerical_summary": describe_numerical(self.df).to_dict(orient="index"),
            "categorical_summary": {
                k: v.to_dict(orient="index") for k, v in describe_categorical(self.df).items()
            },
            "outliers": detect_outliers_iqr(self.df),
            "correlation_matrix": calculate_correlation_matrix(self.df).to_dict(),
        }

        diagnostics: List[Dict[str, Any]] = []
        assumptions: List[Dict[str, Any]] = []
        recommendations: List[Dict[str, Any]] = []
        executed_tests: Dict[str, Any] = {}

        # Supuesto universal que no puede ser verificado automáticamente
        assumptions.append({
            "assumption": "Independencia de las observaciones",
            "status": "Supuesto no evaluado / Pendiente de verificación",
            "evaluated": False,
            "justification": "La independencia depende del diseño del estudio, del método de muestreo y del protocolo de recolección de datos, por lo que debe ser confirmada por el investigador."
        })

        if target_col and target_col in self.df.columns:
            target_series = self.df[target_col].dropna()
            var_type = self.variable_types.get(target_col, "Desconocido")

            if "Cuantitativa" in var_type:
                # Diagnóstico contextual de normalidad de la variable cuantitativa
                norm_diag = self._evaluate_normality_context(target_series, target_col, alpha)
                diagnostics.append(norm_diag)

                if norm_diag.get("insufficient_sample"):
                    norm_status = f"No evaluada: tamaño muestral insuficiente (n = {norm_diag['sample_size']})."
                    norm_evaluated = False
                else:
                    norm_status = f"Evaluado mediante diagnóstico contextual ({norm_diag['summary']})"
                    norm_evaluated = True

                assumptions.append({
                    "assumption": f"Normalidad de la variable '{target_col}'",
                    "status": norm_status,
                    "evaluated": norm_evaluated,
                    "is_satisfied": norm_diag["is_normal_contextual"],
                })

                # CASO 1: Comparación de grupos independientes (si group_col existe)
                if group_col and group_col in self.df.columns:
                    group_res = self._analyze_group_comparison(
                        target_col, group_col, target_series, norm_diag, alpha
                    )
                    diagnostics.extend(group_res["diagnostics"])
                    assumptions.extend(group_res["assumptions"])
                    if group_res["recommendation"] is not None:
                        recommendations.append(group_res["recommendation"])
                        executed_tests[group_res["recommendation"]["recommended_test"]] = group_res["executed_test"]
                        executed_tests.update(group_res.get("additional_executed_tests", {}))

                # CASO 2: Comparación pareada (si paired_col existe)
                elif paired_col and paired_col in self.df.columns:
                    paired_res = self._analyze_paired_comparison(
                        target_col, paired_col, norm_diag, alpha
                    )
                    diagnostics.extend(paired_res["diagnostics"])
                    assumptions.extend(paired_res["assumptions"])
                    if paired_res["recommendation"] is not None:
                        recommendations.append(paired_res["recommendation"])
                        executed_tests[paired_res["recommendation"]["recommended_test"]] = paired_res["executed_test"]

                # CASO 3: 1 muestra contra media poblacional (si popmean existe)
                elif popmean is not None:
                    samp1_res = self._analyze_one_sample(
                        target_col, target_series, popmean, norm_diag, alpha
                    )
                    if samp1_res["recommendation"] is not None:
                        recommendations.append(samp1_res["recommendation"])
                        executed_tests[samp1_res["recommendation"]["recommended_test"]] = samp1_res["executed_test"]

        # Evaluaciones de asociación / correlación (si existen al menos 2 variables numéricas y no hay target explícito)
        num_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        if (
            len(num_cols) >= 2
            and not group_col
            and not paired_col
            and popmean is None
        ):
            corr_recommendations = self._analyze_correlations(num_cols, alpha)
            recommendations.extend(corr_recommendations["recommendations"])
            assumptions.extend(corr_recommendations["assumptions"])
            executed_tests.update(corr_recommendations["executed_tests"])

        # Asociación entre variables categóricas (chi-cuadrado de independencia),
        # únicamente en modo exploratorio (mismo guard que las correlaciones).
        cat_cols = [
            c
            for c, t in self.variable_types.items()
            if t in ("Binaria", "Cualitativa nominal")
        ]
        if (
            len(cat_cols) >= 2
            and not group_col
            and not paired_col
            and popmean is None
        ):
            cat_association = self._analyze_categorical_association(cat_cols, alpha)
            recommendations.extend(cat_association["recommendations"])
            diagnostics.extend(cat_association["diagnostics"])
            executed_tests.update(cat_association["executed_tests"])

        explanation = self._build_pedagogical_explanation(
            recommendations, assumptions, diagnostics
        )

        # Sección de datos faltantes (pipeline E1–E3, sin imputación).
        # La imputación es opt-in: analyze() nunca modifica ni imputa datos.
        # Importación perezosa para evitar el ciclo selector → profile →
        # analysis → dataset_analyzer → pipeline → selector.
        from src.missing_data.pipeline import MissingDataPipeline

        missing_result = MissingDataPipeline().run(self.df, impute=False)
        missing_data = {
            "status": missing_result.status,
            "detection": (
                missing_result.detection_report.to_dict()
                if missing_result.detection_report
                else None
            ),
            "diagnostics": (
                missing_result.diagnostics_report.to_dict()
                if missing_result.diagnostics_report
                else None
            ),
            "candidate_methods": missing_result.candidate_methods,
            "imputed": False,
        }

        return {
            "dataset_summary": {
                "file_name": self.file_name,
                "rows": int(self.df.shape[0]),
                "columns": int(self.df.shape[1]),
                "target_variable": target_col,
                "group_variable": group_col,
            },
            "variable_classification": self.variable_types,
            "data_quality": self.quality_summary,
            "eda": eda_res,
            "diagnostics": diagnostics,
            "assumptions_status": assumptions,
            "recommendations": recommendations,
            "executed_test_results": executed_tests,
            "pedagogical_explanation": explanation,
            "missing_data": missing_data,
        }

    def _evaluate_normality_context(
        self, series: pd.Series, col_name: str, alpha: float
    ) -> Dict[str, Any]:
        """
        Realiza un diagnóstico contextual riguroso de la normalidad.
        Combina Shapiro-Wilk, tamaño de muestra (n), presencia de outliers y asimetría.
        """
        n = len(series)
        if n < 3:
            summary = (
                f"Normalidad no evaluable: se requieren al menos 3 observaciones para "
                f"Shapiro-Wilk (n = {n} en '{col_name}')."
            )
            return {
                "type": "Diagnóstico de Normalidad Contextual",
                "tool": "Shapiro-Wilk + Atípicos IQR + Asimetría + Evaluación de n",
                "variable": col_name,
                "sample_size": n,
                "shapiro_p_value": None,
                "outliers_count": 0,
                "skewness": float("nan"),
                "is_normal_contextual": False,
                "insufficient_sample": True,
                "summary": summary,
            }

        outliers_dict = detect_outliers_iqr(pd.DataFrame({col_name: series}), columns=[col_name])
        outliers_count = outliers_dict[col_name]["outlier_count"]
        skewness = float(series.skew())

        sw_res = shapiro_wilk_test(series, alpha=alpha)
        p_val = sw_res["p_value"]

        # Criterio evaluativo contextual (no binario ni estricto por p-valor)
        is_normal_stat = not sw_res["reject_h0"]
        has_outliers = outliers_count > 0
        is_large_sample = n >= 30

        if is_normal_stat and not has_outliers:
            summary = "Shapiro-Wilk no rechaza normalidad (p > 0.05) y no hay atípicos severos."
            is_normal_contextual = True
        elif is_large_sample and abs(skewness) < 1.0 and outliers_count < (0.05 * n):
            summary = (
                f"El tamaño muestral (n = {n}) proporciona mayor robustez para la inferencia sobre la media frente a "
                f"desviaciones moderadas de normalidad, aunque no implica que los datos originales sean estrictamente normales."
            )
            is_normal_contextual = True
        else:
            summary = (
                f"Se observan desviaciones de normalidad (Shapiro-Wilk p = {p_val:.4e}) "
                f"o presencia de atípicos ({outliers_count} valores) con asimetría de {skewness:.2f}."
            )
            is_normal_contextual = False

        result = {
            "type": "Diagnóstico de Normalidad Contextual",
            "tool": f"Shapiro-Wilk + Atípicos IQR + Asimetría + Evaluación de n = {n}",
            "variable": col_name,
            "sample_size": n,
            "shapiro_p_value": p_val,
            "outliers_count": outliers_count,
            "skewness": skewness,
            "is_normal_contextual": is_normal_contextual,
            "summary": summary,
        }

        # Diagnóstico complementario de Lilliefors (requiere al menos 4 observaciones)
        if n >= 4:
            result["lilliefors_result"] = lilliefors_test(series, alpha=alpha)

        return result

    def _analyze_group_comparison(
        self,
        target_col: str,
        group_col: str,
        target_series: pd.Series,
        norm_diag: Dict[str, Any],
        alpha: float,
    ) -> Dict[str, Any]:
        """
        Evalúa y ejecuta comparación de grupos independientes (2 grupos o k > 2 grupos).
        """
        groups = self.df[group_col].dropna().unique()
        k = len(groups)

        sample_groups = [
            self.df[self.df[group_col] == g][target_col].dropna() for g in groups
        ]

        diagnostics = []
        assumptions = []

        # Comparación de grupos requiere al menos 2 categorías y
        # observaciones suficientes en cada una para Levene y las pruebas.
        if k < 2 or any(len(g) < 2 for g in sample_groups):
            reason = (
                f"La variable '{group_col}' tiene {k} categoría(s) válida(s) y "
                "cada grupo requiere al menos 2 observaciones para comparar "
                "grupos independientes."
            )
            diagnostics.append({
                "type": "Comparación de grupos omitida",
                "tool": None,
                "statistic": None,
                "p_value": None,
                "summary": reason,
            })
            assumptions.append({
                "assumption": f"Comparabilidad de grupos de '{group_col}'",
                "status": "No evaluada: categorías u observaciones insuficientes.",
                "evaluated": False,
                "is_satisfied": False,
            })
            return {
                "diagnostics": diagnostics,
                "assumptions": assumptions,
                "recommendation": None,
                "executed_test": None,
                "additional_executed_tests": {},
            }

        # Evaluación de homocedasticidad
        lev_res = levene_test(*sample_groups, alpha=alpha)
        homocedastic = not lev_res["reject_h0"]

        diagnostics.append({
            "type": "Diagnóstico de Homocedasticidad",
            "tool": "Prueba de Levene",
            "statistic": lev_res["statistic"],
            "p_value": lev_res["p_value"],
            "summary": "Varianzas de los grupos homogéneas" if homocedastic else "Heterocedasticidad detectada (varianzas desiguales)",
        })

        assumptions.append({
            "assumption": f"Homocedasticidad entre grupos de '{group_col}'",
            "status": f"Evaluado mediante prueba de Levene (p = {lev_res['p_value']:.4e})",
            "evaluated": True,
            "is_satisfied": homocedastic,
        })

        additional_executed_tests: Dict[str, Any] = {}

        # Diagnóstico complementario de Bartlett (solo bajo normalidad contextual:
        # la prueba de Bartlett asume normalidad en los grupos)
        if norm_diag["is_normal_contextual"]:
            try:
                bart_res = bartlett_test(*sample_groups, alpha=alpha)
                additional_executed_tests["Bartlett (Homogeneidad de Varianzas)"] = bart_res
                diagnostics.append({
                    "type": "Diagnóstico de Homocedasticidad (Bartlett)",
                    "tool": "Prueba de Bartlett",
                    "statistic": bart_res["statistic"],
                    "p_value": bart_res["p_value"],
                    "summary": (
                        "Varianzas homogéneas según Bartlett (complementa a Levene bajo normalidad)"
                        if not bart_res["reject_h0"]
                        else "Heterocedasticidad según Bartlett (complementa a Levene bajo normalidad)"
                    ),
                })
            except ValueError as e:
                diagnostics.append({
                    "type": "Diagnóstico de Homocedasticidad (Bartlett)",
                    "tool": "Prueba de Bartlett",
                    "statistic": None,
                    "p_value": None,
                    "summary": f"Bartlett no ejecutable: {e}",
                })

        if k == 2:
            s1, s2 = sample_groups[0], sample_groups[1]
            if norm_diag["is_normal_contextual"] and homocedastic:
                rec_test = "t de Student (Muestras Independientes)"
                alt_test = "Mann-Whitney U"
                is_param = True
                justification = (
                    f"Se comparan 2 grupos independientes de '{group_col}'. Ambos grupos presentan un comportamiento "
                    f"adecuado de normalidad/robustez muestral y varianzas homogéneas (Levene p = {lev_res['p_value']:.4e}). "
                    f"La prueba t de Student es la alternativa paramétrica más potente."
                )
                exec_res = t_test_ind(s1, s2, equal_var=True, alpha=alpha)
            elif norm_diag["is_normal_contextual"] and not homocedastic:
                rec_test = "t de Welch (Varianzas Desiguales)"
                alt_test = "Mann-Whitney U"
                is_param = True
                justification = (
                    f"Se comparan 2 grupos independientes de '{group_col}'. Aunque el supuesto de normalidad/robustez "
                    f"se mantiene, existe heterocedasticidad significativa (Levene p = {lev_res['p_value']:.4e}). "
                    f"Se recomienda la prueba t de Welch, que ajusta los grados de libertad para varianzas desiguales."
                )
                exec_res = t_test_ind(s1, s2, equal_var=False, alpha=alpha)
            else:
                rec_test = "Mann-Whitney U (No Paramétrica)"
                alt_test = "t de Welch / Student"
                is_param = False
                justification = (
                    f"Se comparan 2 grupos independientes de '{group_col}'. Debido a desviaciones sustanciales en la "
                    f"normalidad o a la presencia de atípicos en la muestra, se recomienda la prueba no paramétrica "
                    f"de Mann-Whitney U para evaluar diferencias en la ordenación por rangos/mediana."
                )
                exec_res = mann_whitney_test(s1, s2, alpha=alpha)

        else:  # k > 2 grupos
            if norm_diag["is_normal_contextual"] and homocedastic:
                rec_test = "ANOVA de un factor (One-Way ANOVA)"
                alt_test = "Kruskal-Wallis H"
                is_param = True
                justification = (
                    f"Se comparan {k} grupos independientes. Las varianzas resultaron homogéneas (Levene p = {lev_res['p_value']:.4e}) "
                    f"y los grupos no presentan atípicos extremos. ANOVA es la prueba paramétrica óptima."
                )
                exec_res = anova_one_way(*sample_groups, alpha=alpha)
            elif norm_diag["is_normal_contextual"] and not homocedastic:
                rec_test = "ANOVA de Welch (Varianzas Heterogéneas)"
                alt_test = "Kruskal-Wallis H"
                is_param = True
                justification = (
                    f"Se comparan {k} grupos independientes. Existe evidencia de heterocedasticidad (Levene p = {lev_res['p_value']:.4e}). "
                    f"Se recomienda ANOVA de Welch, la cual corrige la ponderación de las varianzas entre grupos."
                )
                exec_res = welch_anova(*sample_groups, alpha=alpha)
            else:
                rec_test = "Kruskal-Wallis H (No Paramétrica)"
                alt_test = "ANOVA de Welch"
                is_param = False
                justification = (
                    f"Se comparan {k} grupos independientes. Al no cumplirse la normalidad o presentar atípicos severos, "
                    f"se recomienda la prueba no paramétrica de Kruskal-Wallis sobre los rangos de las observaciones."
                )
                exec_res = kruskal_wallis_test(*sample_groups, alpha=alpha)

        # Post hoc de Tukey HSD: únicamente tras un ANOVA de un factor significativo
        # (normalidad contextual + homocedasticidad), que es el contexto en el que
        # Tukey HSD es la comparación múltiple paramétrica adecuada.
        if (
            k > 2
            and rec_test == "ANOVA de un factor (One-Way ANOVA)"
            and exec_res["reject_h0"]
        ):
            try:
                tukey_res = tukey_hsd_test(*sample_groups, alpha=alpha)
                additional_executed_tests["Tukey HSD (Comparaciones Múltiples)"] = tukey_res
                significant_pairs = [
                    p for p in tukey_res["pairwise_comparisons"] if p["significant"]
                ]
                diagnostics.append({
                    "type": "Post hoc (Tukey HSD)",
                    "tool": "Tukey HSD",
                    "statistic": tukey_res["statistic"],
                    "p_value": tukey_res["p_value"],
                    "summary": (
                        f"ANOVA rechazó H0; Tukey HSD identificó {len(significant_pairs)} par(es) "
                        "con diferencias significativas."
                    ),
                })
            except ValueError as e:
                diagnostics.append({
                    "type": "Post hoc (Tukey HSD)",
                    "tool": "Tukey HSD",
                    "statistic": None,
                    "p_value": None,
                    "summary": f"Tukey HSD no ejecutable: {e}",
                })

        # Verificación complementaria por permutaciones en la comparación no
        # paramétrica de 2 grupos independientes (semilla fija para reproducibilidad).
        if k == 2 and not norm_diag["is_normal_contextual"]:
            try:
                perm_res = permutation_test(
                    sample_groups[0], sample_groups[1], alpha=alpha, seed=42
                )
                additional_executed_tests["Permutaciones (Diferencia de Medias)"] = perm_res
                diagnostics.append({
                    "type": "Verificación complementaria (Permutaciones)",
                    "tool": "Prueba de permutaciones",
                    "statistic": perm_res["statistic"],
                    "p_value": perm_res["p_value"],
                    "summary": (
                        f"Verificación complementaria por permutaciones (p = {perm_res['p_value']:.4e}, "
                        "1000 permutaciones, semilla fija)."
                    ),
                })
            except ValueError as e:
                diagnostics.append({
                    "type": "Verificación complementaria (Permutaciones)",
                    "tool": "Prueba de permutaciones",
                    "statistic": None,
                    "p_value": None,
                    "summary": f"Permutaciones no ejecutable: {e}",
                })

        recommendation = {
            "target_variable": target_col,
            "group_variable": group_col,
            "recommended_test": rec_test,
            "alternative_test": alt_test,
            "is_parametric": is_param,
            "statistical_justification": justification,
            "causality_disclaimer": "Nota: La existencia de diferencias estadísticamente significativas entre grupos NO implica una relación de causalidad.",
        }

        return {
            "diagnostics": diagnostics,
            "assumptions": assumptions,
            "recommendation": recommendation,
            "executed_test": exec_res,
            "additional_executed_tests": additional_executed_tests,
        }

    def _analyze_paired_comparison(
        self,
        target_col: str,
        paired_col: str,
        norm_diag: Dict[str, Any],
        alpha: float,
    ) -> Dict[str, Any]:
        """
        Evalúa y ejecuta comparación de 2 mediciones pareadas / relacionadas.
        """
        s1 = self.df[target_col].dropna()
        s2 = self.df[paired_col].dropna()
        common_idx = s1.index.intersection(s2.index)

        s1_c = s1.loc[common_idx]
        s2_c = s2.loc[common_idx]
        diff = s1_c - s2_c

        diff_norm = self._evaluate_normality_context(diff, f"Diferencia({target_col}-{paired_col})", alpha)

        if diff_norm.get("insufficient_sample"):
            diagnostics = [diff_norm]
            assumptions = [{
                "assumption": f"Normalidad de las diferencias ({target_col} - {paired_col})",
                "status": f"No evaluada: tamaño muestral insuficiente (n = {diff_norm['sample_size']}).",
                "evaluated": False,
                "is_satisfied": False,
            }]
            return {
                "diagnostics": diagnostics,
                "assumptions": assumptions,
                "recommendation": None,
                "executed_test": None,
            }

        if diff_norm["is_normal_contextual"]:
            rec_test = "t de Student (Muestras Pareadas)"
            alt_test = "Wilcoxon Pareado"
            is_param = True
            justification = (
                f"Se analizan diferencias pareadas entre '{target_col}' y '{paired_col}'. La distribución de las "
                f"diferencias no muestra violaciones severas de normalidad. Se recomienda la prueba t pareada."
            )
            exec_res = t_test_rel(s1_c, s2_c, alpha=alpha)
        else:
            rec_test = "Wilcoxon Pareado (No Paramétrica)"
            alt_test = "t de Student Pareada"
            is_param = False
            justification = (
                f"Se analizan diferencias pareadas entre '{target_col}' y '{paired_col}'. Las diferencias muestran "
                f"asimetría o atípicos significativos. Se recomienda la prueba no paramétrica de rangos con signo de Wilcoxon."
            )
            exec_res = wilcoxon_signed_rank(s1_c, s2_c, alpha=alpha)

        recommendation = {
            "target_variable": target_col,
            "paired_variable": paired_col,
            "recommended_test": rec_test,
            "alternative_test": alt_test,
            "is_parametric": is_param,
            "statistical_justification": justification,
            "causality_disclaimer": "Nota: La diferencia observada no demuestra causalidad directa sin control experimental adecuado.",
        }

        return {
            "diagnostics": [diff_norm],
            "assumptions": [{
                "assumption": f"Normalidad de las diferencias ({target_col} - {paired_col})",
                "status": f"Evaluado ({diff_norm['summary']})",
                "evaluated": True,
                "is_satisfied": diff_norm["is_normal_contextual"],
            }],
            "recommendation": recommendation,
            "executed_test": exec_res,
        }

    def _analyze_one_sample(
        self,
        target_col: str,
        target_series: pd.Series,
        popmean: float,
        norm_diag: Dict[str, Any],
        alpha: float,
    ) -> Dict[str, Any]:
        """
        Evalúa y ejecuta inferencia para 1 muestra respecto a una constante de referencia (popmean).
        """
        if norm_diag.get("insufficient_sample"):
            return {
                "recommendation": None,
                "executed_test": None,
            }

        if norm_diag["is_normal_contextual"]:
            rec_test = "t de Student (1 Muestra)"
            alt_test = "Wilcoxon de 1 Muestra"
            is_param = True
            justification = f"Se evalúa la media de '{target_col}' frente al valor de referencia {popmean} mediante t de Student."
            exec_res = t_test_1samp(target_series, popmean=popmean, alpha=alpha)
        else:
            rec_test = "Wilcoxon de 1 Muestra (No Paramétrica)"
            alt_test = "t de Student (1 Muestra)"
            is_param = False
            justification = f"Debido a la asimetría o presencia de atípicos en '{target_col}', se evalúa la mediana frente a {popmean} mediante Wilcoxon."
            exec_res = wilcoxon_signed_rank(target_series, popmean=popmean, alpha=alpha)

        recommendation = {
            "target_variable": target_col,
            "reference_mean": popmean,
            "recommended_test": rec_test,
            "alternative_test": alt_test,
            "is_parametric": is_param,
            "statistical_justification": justification,
            "causality_disclaimer": "Nota: La prueba evalúa concordancia estadística con un parámetro de referencia, no relaciones causales.",
        }

        return {
            "recommendation": recommendation,
            "executed_test": exec_res,
        }

    def _analyze_correlations(
        self, num_cols: List[str], alpha: float
    ) -> Dict[str, Any]:
        """
        Evalúa y recomienda pruebas de correlación (Pearson vs. Spearman) entre variables cuantitativas.
        """
        recommendations = []
        assumptions = []
        executed_tests = {}

        assumptions.append({
            "assumption": "Linealidad y Normalidad Bivariada para Pearson",
            "status": "Supuesto no evaluado / Pendiente de verificación",
            "evaluated": False,
            "justification": "El coeficiente de Pearson asume una relación lineal estricta y normalidad bivariada. Si la relación es monótona pero no lineal, Spearman es más apropiado."
        })

        for i in range(len(num_cols)):
            for j in range(i + 1, len(num_cols)):
                col1, col2 = num_cols[i], num_cols[j]
                s1 = self.df[col1].dropna()
                s2 = self.df[col2].dropna()

                outliers1 = detect_outliers_iqr(pd.DataFrame({col1: s1}), columns=[col1])[col1]["outlier_count"]
                outliers2 = detect_outliers_iqr(pd.DataFrame({col2: s2}), columns=[col2])[col2]["outlier_count"]

                has_outliers = (outliers1 > 0) or (outliers2 > 0)

                if not has_outliers:
                    rec_test = "Correlación de Pearson"
                    alt_test = "Correlación de Spearman"
                    justification = (
                        f"Se analiza la asociación entre '{col1}' y '{col2}'. Al ser ambas cuantitativas continuas y no registrar "
                        f"atípicos severos, Pearson evalúa la intensidad de la asociación lineal."
                    )
                else:
                    rec_test = "Correlación de Spearman (No Paramétrica)"
                    alt_test = "Correlación de Pearson"
                    justification = (
                        f"Se detectaron valores atípicos en las variables cuantitativas ({col1} / {col2}). "
                        f"Se recomienda el coeficiente de Spearman sobre rangos para evaluar asociación monótona sin sensibilidad a atípicos."
                    )

                recommendations.append({
                    "variables": [col1, col2],
                    "recommended_test": rec_test,
                    "alternative_test": alt_test,
                    "statistical_justification": justification,
                    "causality_disclaimer": f"Nota: Una correlación significativa entre '{col1}' y '{col2}' demuestra asociación matemática, NO una relación de causa y efecto.",
                })

                # Ejecución de la correlación sobre pares alineados por índice
                mask = ~(
                    np.isnan(self.df[col1].to_numpy(dtype=float))
                    | np.isnan(self.df[col2].to_numpy(dtype=float))
                )
                x = self.df[col1].to_numpy(dtype=float)[mask]
                y = self.df[col2].to_numpy(dtype=float)[mask]

                if len(x) < 2 or np.ptp(x) == 0 or np.ptp(y) == 0:
                    stat_val, p_val = float("nan"), float("nan")
                    reject_h0 = False
                else:
                    try:
                        if rec_test.startswith("Correlación de Pearson"):
                            stat_val, p_val = stats.pearsonr(x, y)
                        else:
                            stat_val, p_val = stats.spearmanr(x, y)
                        reject_h0 = bool(p_val < alpha)
                    except (ValueError, TypeError) as exc:
                        logger.warning(f"Correlación no calculable para '{col1}'/'{col2}': {exc}")
                        stat_val, p_val = float("nan"), float("nan")
                        reject_h0 = False

                decision = "Rechazar H0" if reject_h0 else "No rechazar H0"
                executed_tests[rec_test] = {
                    "test_name": rec_test,
                    "statistic": float(stat_val),
                    "p_value": float(p_val),
                    "alpha": float(alpha),
                    "null_hypothesis": f"H0: No existe asociación (correlación = 0) entre '{col1}' y '{col2}'.",
                    "alt_hypothesis": f"H1: Existe asociación entre '{col1}' y '{col2}'.",
                    "decision": decision,
                    "reject_h0": reject_h0,
                    "interpretation": (
                        f"Con un p-valor de {p_val:.4e} y alpha={alpha}, "
                        f"{'existe' if reject_h0 else 'no existe'} evidencia estadística de asociación "
                        f"entre '{col1}' y '{col2}'."
                    ),
                }

        return {
            "recommendations": recommendations,
            "assumptions": assumptions,
            "executed_tests": executed_tests,
        }

    def _analyze_categorical_association(
        self, cat_cols: List[str], alpha: float
    ) -> Dict[str, Any]:
        """
        Evalúa la asociación entre pares de variables categóricas mediante la
        prueba chi-cuadrado de independencia. Se omiten los pares sin tabla de
        contingencia válida (una sola categoría, observaciones insuficientes o
        marginal nula), registrando el motivo de forma diagnóstica.
        """
        recommendations = []
        diagnostics = []
        executed_tests = {}

        for i in range(len(cat_cols)):
            for j in range(i + 1, len(cat_cols)):
                col1, col2 = cat_cols[i], cat_cols[j]
                mask = self.df[col1].notna() & self.df[col2].notna()

                if mask.sum() < 2:
                    diagnostics.append({
                        "type": "Asociación categórica omitida",
                        "tool": "Chi-cuadrado (Independencia)",
                        "statistic": None,
                        "p_value": None,
                        "summary": (
                            f"Chi-cuadrado omitido para '{col1}' y '{col2}': observaciones "
                            "válidas insuficientes para construir una tabla de contingencia."
                        ),
                    })
                    continue

                table = pd.crosstab(self.df.loc[mask, col1], self.df.loc[mask, col2])

                if table.shape[0] < 2 or table.shape[1] < 2:
                    diagnostics.append({
                        "type": "Asociación categórica omitida",
                        "tool": "Chi-cuadrado (Independencia)",
                        "statistic": None,
                        "p_value": None,
                        "summary": (
                            f"Chi-cuadrado omitido para '{col1}' y '{col2}': una de las variables "
                            "presenta una sola categoría en los datos válidos."
                        ),
                    })
                    continue

                if (table.sum(axis=0) == 0).any() or (table.sum(axis=1) == 0).any():
                    diagnostics.append({
                        "type": "Asociación categórica omitida",
                        "tool": "Chi-cuadrado (Independencia)",
                        "statistic": None,
                        "p_value": None,
                        "summary": (
                            f"Chi-cuadrado omitido para '{col1}' y '{col2}': la tabla de contingencia "
                            "presenta una fila o columna con suma total cero."
                        ),
                    })
                    continue

                chi_res = chi_square_test(table.to_numpy(), alpha=alpha)
                rec_test = "Chi-cuadrado (Independencia)"

                recommendations.append({
                    "variables": [col1, col2],
                    "recommended_test": rec_test,
                    "alternative_test": "Prueba exacta de Fisher (para tablas 2x2 con frecuencias esperadas bajas)",
                    "statistical_justification": (
                        f"Se evalúa la asociación entre las variables categóricas '{col1}' y '{col2}' "
                        "mediante la prueba chi-cuadrado de independencia."
                    ),
                    "causality_disclaimer": (
                        f"Nota: Una asociación significativa entre '{col1}' y '{col2}' indica dependencia "
                        "estadística, NO causalidad."
                    ),
                })
                executed_tests[rec_test] = chi_res

        return {
            "recommendations": recommendations,
            "diagnostics": diagnostics,
            "executed_tests": executed_tests,
        }

    def _build_pedagogical_explanation(
        self,
        recommendations: List[Dict[str, Any]],
        assumptions: List[Dict[str, Any]],
        diagnostics: List[Dict[str, Any]],
    ) -> str:
        """
        Genera una explicación pedagógica clara dirigida a un estudiante de Estadística.
        """
        lines = [
            "### Explicación Metodológica para el Estudiante de Estadística\n",
            "1. **Enfoque de Decisión Metodológica**:",
            "   En consultoría estadística, las pruebas no se seleccionan mediante reglas rígidas de p-valor.",
            "   Se evalúa en conjunto el tamaño muestral (n), la distribución de los datos, los valores atípicos y los supuestos teóricos.\n",
            "2. **Teorema del Límite Central (TLC)**:",
            "   Recordatorio fundamental: El TLC establece que para muestras de tamaño adecuado, la distribución de la media muestral",
            "   tiende a ser normal. Esto otorga mayor robustez a pruebas paramétricas frente a desviaciones moderadas, pero NO implica",
            "   que los datos originales sean normales ni justifica ignorar atípicos severos.\n",
            "3. **Evaluación de Supuestos**:",
        ]

        for ass in assumptions:
            lines.append(f"   - **{ass['assumption']}**: {ass['status']}.")

        lines.append("\n4. **Recomendaciones y Causalidad**:")
        for rec in recommendations:
            test_name = rec.get("recommended_test", "Prueba")
            just = rec.get("statistical_justification", "")
            lines.append(f"   - **{test_name}**: {just}")
            lines.append(f"     *{rec.get('causality_disclaimer', '')}*\n")

        return "\n".join(lines)
