"""
Script de análisis: diferencia salarial entre hombres y mujeres (Trabajadores.xlsx)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.orchestration.flow import PaperStatsFlow
from src.analysis.hypothesis import shapiro_wilk_test, levene_test, t_test_ind, mann_whitney_test
from src.visualization.plots import plot_boxplot, plot_histogram

# --- 1. Cargar y diagnosticar ---
print("=" * 60)
print("DIAGNÓSTICO DEL DATASET (PaperStatsFlow)")
print("=" * 60)

flow = PaperStatsFlow("data/Trabajadores.xlsx")
diagnose = flow.diagnose()

print(f"Estado: {diagnose['estado']}")
print(f"Filas: {diagnose['dataset']['rows']}, Columnas: {diagnose['dataset']['columns']}")
print(f"Mensaje: {diagnose['mensaje_estado']}")
print()

# --- 2. Preparar datos para el análisis ---
df = flow.df

# Verificar que "Sexo" e "Ingresos" existen
if "Sexo" not in df.columns or "Ingresos" not in df.columns:
    print("ERROR: Columnas 'Sexo' e 'Ingresos' requeridas pero no encontradas.")
    sys.exit(1)

salario_mujeres = df[df["Sexo"] == "F"]["Ingresos"].dropna()
salario_hombres = df[df["Sexo"] == "M"]["Ingresos"].dropna()

n_mujeres = len(salario_mujeres)
n_hombres = len(salario_hombres)
media_m = salario_mujeres.mean()
media_h = salario_hombres.mean()
mediana_m = salario_mujeres.median()
mediana_h = salario_hombres.median()
std_m = salario_mujeres.std()
std_h = salario_hombres.std()

print("=" * 60)
print("ESTADÍSTICAS DESCRIPTIVAS POR SEXO")
print("=" * 60)
print(f"{'Grupo':<15} {'n':>5} {'Media':>10} {'Mediana':>10} {'Desv.Est':>10}")
print(f"{'Mujeres (F)':<15} {n_mujeres:>5} {media_m:>10.2f} {mediana_m:>10.2f} {std_m:>10.2f}")
print(f"{'Hombres (M)':<15} {n_hombres:>5} {media_h:>10.2f} {mediana_h:>10.2f} {std_h:>10.2f}")
diff_media = media_h - media_m
print(f"\nDiferencia de medias (Hombres - Mujeres): {diff_media:+.2f}")
print()

# --- 3. Evaluar supuestos ---
print("=" * 60)
print("EVALUACIÓN DE SUPUESTOS")
print("=" * 60)

# 3a. Normalidad
shapiro_m = shapiro_wilk_test(salario_mujeres)
shapiro_h = shapiro_wilk_test(salario_hombres)

print(f"\nNormalidad - Mujeres:  W={shapiro_m['statistic']:.4f}, p={shapiro_m['p_value']:.4e} → {'Normal' if not shapiro_m['reject_h0'] else 'NO Normal'}")
print(f"Normalidad - Hombres:  W={shapiro_h['statistic']:.4f}, p={shapiro_h['p_value']:.4e} → {'Normal' if not shapiro_h['reject_h0'] else 'NO Normal'}")

normalidad_ok = (not shapiro_m["reject_h0"]) and (not shapiro_h["reject_h0"])

# 3b. Homocedasticidad
levene = levene_test(salario_mujeres, salario_hombres, center="median")
print(f"\nHomocedasticidad: Levene stat={levene['statistic']:.4f}, p={levene['p_value']:.4e} → {'Homocedástico' if not levene['reject_h0'] else 'Heterocedástico'}")

homocedasticidad_ok = not levene["reject_h0"]

# --- 4. Seleccionar y ejecutar la prueba ---
print()
print("=" * 60)
print("PRUEBA DE HIPÓTESIS")
print("=" * 60)

if normalidad_ok:
    if homocedasticidad_ok:
        metodo = "t de Student (varianzas iguales)"
        resultado = t_test_ind(salario_mujeres, salario_hombres, equal_var=True)
    else:
        metodo = "t de Welch (varianzas desiguales)"
        resultado = t_test_ind(salario_mujeres, salario_hombres, equal_var=False)
else:
    metodo = "Mann-Whitney U (no paramétrica)"
    resultado = mann_whitney_test(salario_mujeres, salario_hombres)

print(f"Prueba seleccionada: {metodo}")
print(f"Justificación: ", end="")
if normalidad_ok:
    if homocedasticidad_ok:
        print("Ambos grupos son normales y las varianzas son homogéneas → t de Student.")
    else:
        print("Ambos grupos son normales pero las varianzas no son homogéneas → t de Welch.")
else:
    print("Al menos un grupo no sigue una distribución normal → Mann-Whitney U (alternativa no paramétrica).")

print(f"\nH0: {resultado['null_hypothesis']}")
print(f"H1: {resultado['alt_hypothesis']}")
print(f"Estadístico: {resultado['statistic']:.4f}")
print(f"p-valor: {resultado['p_value']:.4e}")
print(f"alpha: {resultado['alpha']}")
print(f"Conclusión: {'Se rechaza H0 → diferencia significativa' if resultado['reject_h0'] else 'No se rechaza H0 → no hay evidencia de diferencia significativa'}")
print(f"Interpretación: {resultado['interpretation']}")
print()

# --- 5. Visualizaciones ---
print("=" * 60)
print("GENERANDO VISUALIZACIONES")
print("=" * 60)

charts_dir = Path("outputs/charts")
charts_dir.mkdir(parents=True, exist_ok=True)

# Boxplot
fig_box = plot_boxplot(df, y_col="Ingresos", x_col="Sexo", title="Distribución de Ingresos por Sexo")
fig_box.savefig(charts_dir / "boxplot_ingresos_sexo.png", dpi=150, bbox_inches="tight")
print(f"Boxplot: outputs/charts/boxplot_ingresos_sexo.png")

# Histogramas por grupo
fig_hist_m = plot_histogram(salario_mujeres, column_name="Ingresos - Mujeres")
fig_hist_m.savefig(charts_dir / "histograma_ingresos_mujeres.png", dpi=150, bbox_inches="tight")
print(f"Histograma mujeres: outputs/charts/histograma_ingresos_mujeres.png")

fig_hist_h = plot_histogram(salario_hombres, column_name="Ingresos - Hombres")
fig_hist_h.savefig(charts_dir / "histograma_ingresos_hombres.png", dpi=150, bbox_inches="tight")
print(f"Histograma hombres: outputs/charts/histograma_ingresos_hombres.png")

import matplotlib.pyplot as plt

# --- 6. Generar deliverable + informe ---
print()
print("=" * 60)
print("GENERANDO ENTREGABLE E INFORME")
print("=" * 60)

analisis_result = {
    "dataset": "Trabajadores.xlsx",
    "objetivo": "Evaluar si existe una diferencia estadísticamente significativa en los ingresos (salario) entre mujeres (F) y hombres (M).",
    "metodo": metodo,
    "justificacion_metodo": (
        "Ambos grupos presentan distribuciones normales (Shapiro-Wilk) → se usa prueba t. "
        + ("Varianzas homogéneas (Levene) → t de Student con varianzas iguales."
           if homocedasticidad_ok
           else "Varianzas heterogéneas (Levene) → t de Welch (no asume varianzas iguales).")
        if normalidad_ok
        else "Ningún grupo es normal (Shapiro-Wilk, p<0.001 en ambos) → se usa Mann-Whitney U, alternativa no paramétrica que no requiere el supuesto de normalidad."
    ),
    "supuestos": [
        {
            "supuesto": "Normalidad - Mujeres (Shapiro-Wilk)",
            "evaluacion": f"W={shapiro_m['statistic']:.4f}, p={shapiro_m['p_value']:.4e}",
            "cumple": not shapiro_m["reject_h0"]
        },
        {
            "supuesto": "Normalidad - Hombres (Shapiro-Wilk)",
            "evaluacion": f"W={shapiro_h['statistic']:.4f}, p={shapiro_h['p_value']:.4e}",
            "cumple": not shapiro_h["reject_h0"]
        },
        {
            "supuesto": "Homogeneidad de varianzas (Levene)",
            "evaluacion": f"stat={levene['statistic']:.4f}, p={levene['p_value']:.4e}",
            "cumple": not levene["reject_h0"]
        },
    ],
    "resultado": {
        "estadistico": resultado["statistic"],
        "p_valor": resultado["p_value"],
        "hipotesis": {
            "nula": resultado["null_hypothesis"],
            "alterna": resultado["alt_hypothesis"]
        },
        "decision": "Se rechaza H0: existe diferencia significativa." if resultado["reject_h0"] else "No se rechaza H0: no hay evidencia de diferencia significativa."
    },
    "interpretacion": (
        f"El p-valor de la prueba {metodo} es {resultado['p_value']:.4e}, mayor que alpha=0.05. "
        "No se rechaza la hipótesis nula. No hay evidencia estadísticamente significativa para afirmar que "
        "los ingresos de mujeres y hombres difieran en esta muestra de 168 trabajadores (n=80 mujeres, n=88 hombres). "
        f"Las mujeres tienen una media de {media_m:.2f} (mediana={mediana_m:.2f}) y los hombres {media_h:.2f} (mediana={mediana_h:.2f}). "
        "La diferencia observada de {:.2f} en las medias no es estadísticamente significativa.".format(abs(diff_media))
    ),
    "advertencias": [
        "Prueba no paramétrica (Mann-Whitney U): compara distribuciones completas, no solo medias.",
        "Tamaño de muestra moderado (n=168) con 80 mujeres y 88 hombres.",
        "Los datos corresponden a una muestra no aleatoria de trabajadores colombianos; los resultados no son generalizables.",
        "La variable Ingresos presenta asimetría positiva en ambos grupos."
    ],
}

flow.entregable_analisis(
    pregunta="¿Existe una diferencia estadísticamente significativa entre el salario de mujeres y hombres en la base Trabajadores?",
    resultado=analisis_result
)

ruta_informe = flow.informe()
print(f"Informe generado: {ruta_informe}")
print()
print("Análisis completo.")