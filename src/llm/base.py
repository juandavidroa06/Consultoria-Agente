"""
Capa de abstracción e interfaz base para integración con Modelos de Lenguaje (LLM).
Permite mantener el núcleo estadístico completamente desacoplado del proveedor del LLM (ej. Ollama).
"""

from abc import ABC, abstractmethod
import re
from typing import Dict, Any
from src.utils.logger import setup_logger

logger = setup_logger("LLMBase")


class BaseLLMClient(ABC):
    """
    Clase base abstracta para clientes LLM.
    Cualquier proveedor posterior (como OllamaLLMClient) heredará de esta interfaz.
    """

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Genera una respuesta de texto a partir de un prompt.
        """
        pass

    @abstractmethod
    def extract_metadata(self, article_text: str) -> Dict[str, Any]:
        """
        Extrae la información estructurada relevante del texto del artículo científico.
        """
        pass

    @abstractmethod
    def analyze_methodology(self, metadata: Dict[str, Any], article_text: str) -> Dict[str, Any]:
        """
        Analiza estadísticamente la metodología, diseño y supuestos del artículo.
        """
        pass


class RuleBasedLLMClient(BaseLLMClient):
    """
    Cliente heurístico/reglado por defecto para la Fase 1.
    Permite procesar artículos sin depender de un servicio LLM externo activo,
    garantizando la ejecución local y determinista.
    """

    def generate(self, prompt: str, **kwargs) -> str:
        logger.info("Ejecutando generación de respuesta reglada/heurística.")
        return f"[Respuesta Heurística] Procesado prompt de {len(prompt)} caracteres."

    def extract_metadata(self, article_text: str) -> Dict[str, Any]:
        """
        Extrae los 17 puntos clave del artículo utilizando expresiones regulares y heurísticas de texto.
        """
        logger.info("Extrayendo metadatos de artículo mediante heurísticas textuales.")

        lines = [line.strip() for line in article_text.split("\n") if line.strip()]
        full_text_lower = article_text.lower()

        # Heurística para título (primera línea no vacía relevante)
        title = lines[0] if lines else "No se especifica en el artículo."
        if len(title) > 200:
            title = title[:200] + "..."

        # Extracción de año mediante regex de 4 dígitos
        years = re.findall(r"\b(19\d\d|20\d\d)\b", article_text)
        year = years[0] if years else "No se especifica en el artículo."

        # Búsqueda de autores
        authors = "No se especifica en el artículo."
        for line in lines[1:5]:
            if any(keyword in line.lower() for keyword in ["author", "autor", "by", "por", "dpto", "university", "universidad"]):
                authors = line
                break

        # Búsqueda de secciones clave mediante patrones de lenguaje
        def extract_section(keywords, default="No se especifica en el artículo."):
            pattern = re.compile(
                r"(?::|--|\n)\s*([^\n]+(?:\n[^\n]+){0,3})", re.IGNORECASE
            )
            for kw in keywords:
                prefix = r"\b" if (kw[0].isalnum() or kw[0] == "_") else r""
                suffix = r"\b" if (kw[-1].isalnum() or kw[-1] == "_") else r""
                match = re.search(prefix + re.escape(kw) + suffix + r"\s*[:\-\n]?\s*(.*?)(?=\n\n|\Z)", article_text, re.IGNORECASE | re.DOTALL)
                if match:
                    snippet = match.group(1).strip()
                    # Tomar los primeros 300 caracteres limpios
                    clean_snippet = re.sub(r"\s+", " ", snippet)[:350].strip()
                    if clean_snippet:
                        return clean_snippet
            return default

        objective = extract_section(["objetivo", "objective", "aim", "purpose", "propósito"])
        research_question = extract_section(["pregunta de investigación", "research question", "pregunta"])
        hypothesis = extract_section(["hipótesis", "hypothesis", "hipotesis"])
        population = extract_section(["población", "population", "target population"])
        sample = extract_section(["muestra", "sample", "sample size", "tamaño muestral", "n ="])
        methodology = extract_section(["metodología", "methodology", "methods", "métodos", "diseño del estudio"])
        results = extract_section(["resultados", "results", "hallazgos", "findings"])
        limitations = extract_section(["limitaciones", "limitations", "debilidades"])
        conclusions = extract_section(["conclusiones", "conclusions", "conclusión"])

        # Identificación de software estadístico
        software_found = []
        software_keywords = ["R", "Python", "SPSS", "Stata", "SAS", "Mplus", "G*Power", "MATLAB", "EViews", "JASP"]
        for sw in software_keywords:
            if sw == "R":
                pattern = r"(?<![\w])R(?![\w²^])"
            else:
                pattern = r"\b" + re.escape(sw) + r"\b"
            if re.search(pattern, article_text, re.IGNORECASE):
                software_found.append(sw)
        software_str = ", ".join(software_found) if software_found else "No se especifica en el artículo."

        return {
            "title": title,
            "authors": authors,
            "year": year,
            "journal": "No se especifica en el artículo.",
            "objective": objective,
            "research_question": research_question,
            "hypothesis": hypothesis,
            "population": population,
            "sample": sample,
            "study_design": extract_section(["diseño", "design", "estudio observacional", "experimento"]),
            "sampling_method": extract_section(["muestreo", "sampling", "aleatorio", "conveniencia"]),
            "variables": extract_section(["variables", "variable dependiente", "covariables"]),
            "methodology": methodology,
            "statistical_methods": extract_section(["métodos estadísticos", "statistical methods", "análisis estadístico"]),
            "statistical_models": extract_section(["modelos", "models", "regresión", "regression"]),
            "results": results,
            "software": software_str,
            "limitations": limitations,
            "conclusions": conclusions,
        }

    def analyze_methodology(self, metadata: Dict[str, Any], article_text: str) -> Dict[str, Any]:
        """
        Analiza los métodos estadísticos identificados y clasifica variables y supuestos.
        """
        logger.info("Analizando metodología estadística mediante regla heurística.")

        text_lower = article_text.lower()

        # Detección de pruebas de hipótesis comunes
        tests_detected = []
        possible_tests = [
            ("Shapiro-Wilk", ["shapiro", "shapiro-wilk"]),
            ("Kolmogorov-Smirnov", ["kolmogorov", "ks test"]),
            ("t de Student", ["t-test", "t de student", "prueba t"]),
            ("ANOVA", ["anova", "análisis de varianza"]),
            ("Chi-cuadrado", ["chi-cuadrado", "chi-square", "chi2"]),
            ("Mann-Whitney", ["mann-whitney", "wilcoxon rank-sum"]),
            ("Wilcoxon", ["wilcoxon"]),
            ("Levene", ["levene"]),
            ("Breusch-Pagan", ["breusch-pagan"]),
            ("Durbin-Watson", ["durbin-watson"]),
            ("Kruskal-Wallis", ["kruskal-wallis"]),
            ("Tukey HSD", ["tukey"]),
        ]
        for name, kws in possible_tests:
            if any(kw in text_lower for kw in kws):
                tests_detected.append(name)

        # Detección de modelos estadísticos
        models_detected = []
        possible_models = [
            ("Regresión Lineal", ["regresión lineal", "linear regression", "ols"]),
            ("Regresión Logística", ["regresión logística", "logistic regression", "logit"]),
            ("GLM (Modelos Lineales Generalizados)", ["glm", "generalized linear"]),
            ("Regresión Poisson", ["poisson regression", "regresión poisson"]),
            ("Modelos Mixtos", ["mixed models", "multilevel", "modelos mixtos"]),
            ("Análisis de Supervivencia / Cox", ["cox", "kaplan-meier", "survival"]),
            ("ARIMA / Series de Tiempo", ["arima", "time series", "series de tiempo"]),
            ("PCA / Análisis Factorial", ["pca", "componentes principales", "factor analysis"]),
            ("Clustering", ["cluster", "k-means"]),
        ]
        for name, kws in possible_models:
            if any(kw in text_lower for kw in kws):
                models_detected.append(name)

        # Supuestos requeridos según métodos detectados
        assumptions_to_check = []
        if any(m in models_detected for m in ["Regresión Lineal", "ANOVA"]) or "t de Student" in tests_detected:
            assumptions_to_check.extend(["Normalidad de residuos / datos", "Homocedasticidad (Igualdad de varianzas)", "Independencia de observaciones"])
        if "Regresión Lineal" in models_detected:
            assumptions_to_check.extend(["Linealidad en parámetros", "Ausencia de multicolinealidad (VIF)"])
        if "Regresión Logística" in models_detected:
            assumptions_to_check.extend(["Ausencia de separación perfecta", "Linealidad del logit con covariables continuas"])

        return {
            "tests_detected": tests_detected if tests_detected else ["No se especifican pruebas explícitas."],
            "models_detected": models_detected if models_detected else ["No se especifican modelos formales explícitos."],
            "assumptions_required": list(dict.fromkeys(assumptions_to_check)) if assumptions_to_check else ["Normalidad y representatividad de la muestra."],
            "justification_evaluation": (
                "Se recomienda verificar si los supuestos del modelo fueron evaluados formalmente en la investigación. "
                "En estudios con datos continuos y regresión, la verificación de normalidad e independencia es indispensable."
            ),
        }
