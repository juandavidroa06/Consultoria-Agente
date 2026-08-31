# -*- coding: utf-8 -*-

"""
============================================================
AGENTE DE CONSULTORÍA ESTADÍSTICA
Dashboard Streamlit + Gemini Interactions API

FLUJO:

1. Cargar base
2. Diagnosticar base
3. Obtener estadísticas descriptivas
4. IA recomienda métodos
5. Python ejecuta métodos recomendados
6. Python calcula estadístico y p-valor
7. IA interpreta resultados reales
8. Dashboard presenta resultados
9. IA recomienda estrategia de imputación

IMPORTANTE:

- La API Key NO está escrita en este archivo.
- La IA NO calcula los p-valores.
- Python calcula los resultados estadísticos.
- La IA solamente recomienda e interpreta.
============================================================
"""


# ============================================================
# IMPORTACIONES
# ============================================================

import os
import json
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats

from PIL import Image
from sklearn.impute import KNNImputer

# Google Gemini
try:
    from google import genai
except ImportError:
    genai = None


# ============================================================
# CONFIGURACIÓN STREAMLIT
# ============================================================

st.set_page_config(

    page_title="Agente Consultor IA",

    page_icon="🧠",

    layout="wide",

    initial_sidebar_state="expanded"

)


# ============================================================
# RUTAS DEL PROYECTO
# ============================================================

CURRENT_DIR = Path(__file__).resolve().parent

PROJECT_ROOT = CURRENT_DIR.parents[1]

CHARTS_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "charts"
)


# ============================================================
# ARCHIVOS POSIBLES PARA LA API KEY
# ============================================================

POSSIBLE_KEY_FILES = [

    PROJECT_ROOT
    / ".streamlit"
    / "gemini_api_key.txt",

    PROJECT_ROOT
    / "gemini_api_key.txt",

    CURRENT_DIR
    / "gemini_api_key.txt",

]


# ============================================================
# ESTILOS
# ============================================================

st.markdown(
    """
    <style>

    /* =====================================================
       FONDO
       ===================================================== */

    .main {
        background-color: #0e1117;
    }

    .block-container {
        max-width: 1500px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }


    /* =====================================================
       TITULOS
       ===================================================== */

    h1 {
        font-weight: 750 !important;
        letter-spacing: -0.6px;
    }

    h2 {
        font-weight: 700 !important;
    }

    h3 {
        font-weight: 650 !important;
    }


    /* =====================================================
       TARJETAS
       ===================================================== */

    .dashboard-card {

        background:
            linear-gradient(
                145deg,
                rgba(31, 41, 55, 0.96),
                rgba(17, 24, 39, 0.96)
            );

        border: 1px solid
            rgba(148, 163, 184, 0.15);

        border-radius: 16px;

        padding: 22px;

        margin-bottom: 18px;

        box-shadow:
            0 10px 25px
            rgba(0, 0, 0, 0.18);

    }


    /* =====================================================
       TARJETA IA
       ===================================================== */

    .ai-card {

        background:
            linear-gradient(
                145deg,
                rgba(30, 64, 175, 0.20),
                rgba(15, 23, 42, 0.96)
            );

        border-left:
            4px solid #60a5fa;

        border-radius: 14px;

        padding: 22px;

        margin: 15px 0;

    }


    /* =====================================================
       TARJETA ÉXITO
       ===================================================== */

    .success-card {

        background:
            linear-gradient(
                145deg,
                rgba(22, 101, 52, 0.20),
                rgba(15, 23, 42, 0.96)
            );

        border-left:
            4px solid #4ade80;

        border-radius: 14px;

        padding: 20px;

        margin: 15px 0;

    }


    /* =====================================================
       TARJETA WARNING
       ===================================================== */

    .warning-card {

        background:
            linear-gradient(
                145deg,
                rgba(146, 64, 14, 0.20),
                rgba(15, 23, 42, 0.96)
            );

        border-left:
            4px solid #f59e0b;

        border-radius: 14px;

        padding: 20px;

        margin: 15px 0;

    }


    /* =====================================================
       BADGES
       ===================================================== */

    .method-badge {

        display: inline-block;

        padding: 7px 13px;

        border-radius: 999px;

        background-color:
            rgba(96, 165, 250, 0.15);

        border:
            1px solid
            rgba(96, 165, 250, 0.30);

        color: #93c5fd;

        font-size: 13px;

        margin: 4px;

    }


    /* =====================================================
       TEXTO PEQUEÑO
       ===================================================== */

    .small-text {

        color: #94a3b8;

        font-size: 13px;

    }


    /* =====================================================
       SIDEBAR
       ===================================================== */

    section[data-testid="stSidebar"] {

        background-color: #111827;

    }


    /* =====================================================
       DATAFRAMES
       ===================================================== */

    div[data-testid="stDataFrame"] {

        border-radius: 12px;

        overflow: hidden;

    }


    /* =====================================================
       BOTONES
       ===================================================== */

    .stButton > button {

        border-radius: 10px;

        font-weight: 600;

    }


    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# CARGAR API KEY
# ============================================================

def cargar_api_key():
    """
    Busca la API Key en este orden:

    1. Streamlit secrets
    2. Variable de entorno
    3. Archivo local

    Nunca muestra la clave.
    """

    # --------------------------------------------------------
    # OPCIÓN 1
    # Streamlit secrets
    # --------------------------------------------------------

    try:

        if "GEMINI_API_KEY" in st.secrets:

            key = str(
                st.secrets["GEMINI_API_KEY"]
            ).strip()

            if key:

                return (
                    key,
                    "Streamlit secrets"
                )

    except Exception:

        pass


    # --------------------------------------------------------
    # OPCIÓN 2
    # Variable de entorno
    # --------------------------------------------------------

    key = os.getenv(
        "GEMINI_API_KEY"
    )

    if key:

        return (
            key.strip(),
            "Variable de entorno"
        )


    # --------------------------------------------------------
    # OPCIÓN 3
    # Archivo local
    # --------------------------------------------------------

    for path in POSSIBLE_KEY_FILES:

        try:

            if not path.exists():

                continue


            contenido = path.read_text(
                encoding="utf-8"
            ).strip()


            if not contenido:

                continue


            # Permite:

            # TU_API_KEY

            # o:

            # GEMINI_API_KEY=TU_API_KEY

            key = contenido


            if "=" in contenido:

                for line in contenido.splitlines():

                    line = line.strip()

                    if line.startswith(
                        "GEMINI_API_KEY"
                    ):

                        key = (
                            line
                            .split("=", 1)[1]
                            .strip()
                            .strip('"')
                            .strip("'")
                        )

                        break


            if key:

                return (
                    key,
                    f"Archivo local: {path.name}"
                )


        except Exception:

            continue


    return None, None


# ============================================================
# OBTENER API KEY
# ============================================================

API_KEY, API_SOURCE = cargar_api_key()


# ============================================================
# MODELO GEMINI
# ============================================================

def obtener_modelo():

    # Primero secrets

    try:

        if "GEMINI_MODEL" in st.secrets:

            return str(
                st.secrets["GEMINI_MODEL"]
            ).strip()

    except Exception:

        pass


    # Después variable de entorno

    return os.getenv(
        "GEMINI_MODEL",
        "gemini-3.6-flash"
    )


GEMINI_MODEL = obtener_modelo()


# ============================================================
# CLIENTE GEMINI
# ============================================================

@st.cache_resource
def obtener_cliente_gemini(api_key):

    if not api_key:

        return None


    if genai is None:

        return None


    try:

        return genai.Client(
            api_key=api_key
        )

    except Exception as e:

        st.error(
            "No se pudo inicializar Gemini: "
            f"{e}"
        )

        return None


gemini_client = obtener_cliente_gemini(
    API_KEY
)


# ============================================================
# CARGAR BASES
# ============================================================

@st.cache_data
def escanear_bases_de_datos():

    archivos = {}


    for extension in [
        "*.csv",
        "*.xlsx"
    ]:

        for archivo in PROJECT_ROOT.rglob(
            extension
        ):

            # Evitar carpetas innecesarias

            if (
                ".venv" in archivo.parts
                or "node_modules" in archivo.parts
                or ".git" in archivo.parts
            ):

                continue


            clave = (
                f"{archivo.name} "
                f"({archivo.parent.name})"
            )


            archivos[clave] = archivo


    return archivos


# ============================================================
# CARGAR DATOS
# ============================================================

@st.cache_data
def cargar_datos(ruta):

    try:

        # CSV

        if ruta.suffix.lower() == ".csv":

            try:

                return pd.read_csv(
                    ruta
                )

            except UnicodeDecodeError:

                return pd.read_csv(
                    ruta,
                    encoding="latin1"
                )


        # Excel

        if ruta.suffix.lower() == ".xlsx":

            return pd.read_excel(
                ruta
            )


    except Exception as e:

        st.error(
            f"No se pudo cargar la base: {e}"
        )


    return pd.DataFrame()


# ============================================================
# DIAGNÓSTICO GENERAL
# ============================================================

@st.cache_data
def construir_diagnostico(df):

    filas = df.shape[0]

    columnas = df.shape[1]

    numericas = len(
        df.select_dtypes(
            include=np.number
        ).columns
    )

    categoricas = len(
        df.select_dtypes(
            exclude=np.number
        ).columns
    )

    faltantes = int(
        df.isna().sum().sum()
    )

    duplicados = int(
        df.duplicated().sum()
    )


    porcentaje_faltantes = round(
        (
            faltantes
            /
            max(
                df.size,
                1
            )
        )
        * 100,
        2
    )


    return {

        "filas": int(filas),

        "columnas": int(columnas),

        "numericas": int(numericas),

        "categoricas": int(categoricas),

        "faltantes": faltantes,

        "duplicados": duplicados,

        "porcentaje_faltantes":
            porcentaje_faltantes

    }


# ============================================================
# ESTADÍSTICAS DESCRIPTIVAS
# ============================================================

@st.cache_data
def obtener_descriptivos(df):

    numericas = df.select_dtypes(
        include=np.number
    )


    if numericas.empty:

        return pd.DataFrame()


    desc = (
        numericas
        .describe()
        .T
    )


    # Estadísticas adicionales

    desc["mediana"] = (
        numericas
        .median()
    )


    desc["varianza"] = (
        numericas
        .var()
    )


    desc["asimetria"] = (
        numericas
        .skew()
    )


    desc["curtosis"] = (
        numericas
        .kurtosis()
    )


    desc["faltantes"] = (
        numericas
        .isna()
        .sum()
    )


    desc["faltantes_%"] = (
        numericas
        .isna()
        .mean()
        * 100
    )


    desc = desc.reset_index()


    desc = desc.rename(
        columns={
            "index": "Variable"
        }
    )


    return desc.round(4)


# ============================================================
# PERFIL DE VARIABLES
# ============================================================

@st.cache_data
def perfil_variables(df):

    filas = []


    for col in df.columns:

        serie = df[col]


        filas.append({

            "Variable": col,

            "Tipo": str(
                serie.dtype
            ),

            "Registros válidos":
                int(
                    serie.notna().sum()
                ),

            "Faltantes":
                int(
                    serie.isna().sum()
                ),

            "Faltantes %":
                round(
                    serie.isna().mean()
                    * 100,
                    2
                ),

            "Valores únicos":
                int(
                    serie.nunique(
                        dropna=True
                    )
                )

        })


    return pd.DataFrame(
        filas
    )


# ============================================================
# CORRELACIONES
# ============================================================

@st.cache_data
def obtener_correlaciones(df):

    numericas = df.select_dtypes(
        include=np.number
    )


    if numericas.shape[1] < 2:

        return pd.DataFrame()


    corr = numericas.corr(
        method="pearson"
    )


    pares = []


    columnas = list(
        corr.columns
    )


    for i in range(
        len(columnas)
    ):

        for j in range(
            i + 1,
            len(columnas)
        ):

            valor = corr.iloc[
                i,
                j
            ]


            if pd.isna(valor):

                continue


            abs_valor = abs(
                valor
            )


            if abs_valor >= 0.8:

                fuerza = "Muy fuerte"

            elif abs_valor >= 0.6:

                fuerza = "Fuerte"

            elif abs_valor >= 0.4:

                fuerza = "Moderada"

            else:

                fuerza = "Débil"


            pares.append({

                "Variable 1":
                    columnas[i],

                "Variable 2":
                    columnas[j],

                "Correlación":
                    round(
                        valor,
                        4
                    ),

                "Fuerza":
                    fuerza

            })


    resultado = pd.DataFrame(
        pares
    )


    if not resultado.empty:

        resultado["Abs"] = (
            resultado[
                "Correlación"
            ].abs()
        )


        resultado = (
            resultado
            .sort_values(
                "Abs",
                ascending=False
            )
            .drop(
                columns=["Abs"]
            )
        )


    return resultado


# ============================================================
# RESUMEN PARA GEMINI
# ============================================================

def preparar_resumen_para_ia(
    nombre_bd,
    df
):

    diagnostico = (
        construir_diagnostico(
            df
        )
    )


    descriptivos = (
        obtener_descriptivos(
            df
        )
    )


    perfil = (
        perfil_variables(
            df
        )
    )


    correlaciones = (
        obtener_correlaciones(
            df
        )
    )


    resumen = {

        "base": nombre_bd,

        "diagnostico":
            diagnostico,

        "perfil_variables":
            perfil
            .head(150)
            .to_dict(
                orient="records"
            ),

        "estadisticas_descriptivas":
            descriptivos
            .head(100)
            .to_dict(
                orient="records"
            ),

        "correlaciones":
            correlaciones
            .head(30)
            .to_dict(
                orient="records"
            )

    }


    return resumen


# ============================================================
# LLAMAR A GEMINI
# ============================================================

def llamar_gemini(
    prompt,
    schema
):
    """
    Llamada central a Gemini.

    Utiliza Interactions API.

    La respuesta esperada es JSON estructurado.
    """

    # --------------------------------------------------------
    # Verificar API key
    # --------------------------------------------------------

    if not API_KEY:

        return {

            "error":
                (
                    "API Key no encontrada. "
                    "Revisa .streamlit/secrets.toml."
                )

        }


    # --------------------------------------------------------
    # Verificar cliente
    # --------------------------------------------------------

    if gemini_client is None:

        return {

            "error":
                (
                    "No se pudo inicializar "
                    "el cliente de Gemini."
                )

        }


    # --------------------------------------------------------
    # Llamada
    # --------------------------------------------------------

    try:

        interaction = (
            gemini_client
            .interactions
            .create(

                model=GEMINI_MODEL,

                input=prompt,

                response_format=[

                    {

                        "type": "text",

                        "mime_type":
                            "application/json",

                        "schema":
                            schema

                    }

                ]

            )
        )


        # ----------------------------------------------------
        # Obtener texto
        # ----------------------------------------------------

        texto = getattr(
            interaction,
            "output_text",
            None
        )


        if not texto:

            return {

                "error":
                    (
                        "Gemini respondió, "
                        "pero no devolvió "
                        "contenido."
                    )

            }


        # ----------------------------------------------------
        # Convertir JSON
        # ----------------------------------------------------

        try:

            resultado = json.loads(
                texto
            )

        except json.JSONDecodeError as e:

            return {

                "error":
                    (
                        "Gemini respondió, "
                        "pero el resultado no "
                        "pudo convertirse a JSON.\n\n"
                        f"Detalle: {e}\n\n"
                        f"Respuesta recibida:\n{texto}"
                    )

            }


        return resultado


    except Exception as e:

        mensaje = str(e)


        # Mensaje especial para modelo no disponible

        if (
            "404" in mensaje
            and "model" in mensaje.lower()
        ):

            return {

                "error":
                    (
                        "El modelo Gemini configurado "
                        f"({GEMINI_MODEL}) no está "
                        "disponible para esta API Key.\n\n"
                        "Revisa GEMINI_MODEL en "
                        ".streamlit/secrets.toml."
                    )

            }


        return {

            "error":
                (
                    "Error comunicando con Gemini: "
                    f"{type(e).__name__}: {mensaje}"
                )

        }


# ============================================================
# IA: RECOMENDAR MÉTODOS
# ============================================================

@st.cache_data(
    show_spinner=False
)
def recomendar_metodos_ia(
    nombre_bd,
    resumen_json
):

    prompt = f"""
Eres un estadístico senior especializado
en consultoría estadística.

Tu tarea es ANALIZAR LA ESTRUCTURA DE UNA
BASE DE DATOS y recomendar métodos.

NO debes calcular resultados.

NO debes inventar p-valores.

NO debes inventar estadísticos.

NO debes afirmar que una variable cumple
un supuesto.

Debes decidir qué métodos son apropiados
según las características reales de la base.

==================================================
BASE
==================================================

{nombre_bd}


==================================================
INFORMACIÓN DISPONIBLE
==================================================

{resumen_json}


==================================================
MÉTODOS DISPONIBLES
==================================================

PRUEBAS DE NORMALIDAD:

- shapiro
- dagostino

PRUEBA DE ASIMETRÍA:

- skewtest

PRUEBA DE CURTOSIS:

- kurtosistest

IMPUTACIÓN:

- knn
- mediana
- moda
- ninguna


==================================================
REGLAS ESTADÍSTICAS
==================================================

1. El nivel de significancia será α = 0.05.

2. Shapiro-Wilk puede utilizarse para muestras
   pequeñas o moderadas.

3. No utilices Shapiro-Wilk indiscriminadamente
   en muestras extremadamente grandes.

4. D'Agostino K² requiere un tamaño de muestra
   suficiente.

5. skewtest evalúa la significancia estadística
   de la asimetría.

6. kurtosistest evalúa la significancia estadística
   de la curtosis.

7. No confundas una medida descriptiva de
   asimetría con un p-valor.

8. No confundas una medida descriptiva de
   curtosis con un p-valor.

9. La imputación NO debe recomendarse solamente
   porque existen valores faltantes.

10. Evalúa porcentaje de faltantes, tipo de
    variable y estructura de la información.

11. KNN debe recomendarse solamente si existe
    suficiente estructura multivariada para
    justificarlo.

12. Si no es necesario imputar, utiliza
    "ninguna".

13. No recomiendes métodos que no estén
    disponibles en la lista.

14. La recomendación debe ser específica
    para ESTA base.


==================================================
OBJETIVO
==================================================

Devuelve una estrategia estadística clara.

Explica por qué seleccionaste cada método.

NO generes una respuesta genérica.
"""


    schema = {

        "type": "object",

        "properties": {

            "estrategia_general": {

                "type": "string"

            },


            "metodos_supuestos": {

                "type": "array",

                "items": {

                    "type": "object",

                    "properties": {

                        "metodo": {

                            "type": "string"

                        },

                        "justificacion": {

                            "type": "string"

                        }

                    },

                    "required": [

                        "metodo",

                        "justificacion"

                    ]

                }

            },


            "imputacion": {

                "type": "object",

                "properties": {

                    "recomendacion": {

                        "type": "string"

                    },

                    "metodo": {

                        "type": "string"

                    },

                    "justificacion": {

                        "type": "string"

                    }

                },

                "required": [

                    "recomendacion",

                    "metodo",

                    "justificacion"

                ]

            },


            "analisis_recomendado": {

                "type": "array",

                "items": {

                    "type": "object",

                    "properties": {

                        "analisis": {

                            "type": "string"

                        },

                        "justificacion": {

                            "type": "string"

                        }

                    },

                    "required": [

                        "analisis",

                        "justificacion"

                    ]

                }

            }

        },

        "required": [

            "estrategia_general",

            "metodos_supuestos",

            "imputacion",

            "analisis_recomendado"

        ]

    }


    return llamar_gemini(
        prompt,
        schema
    )


# ============================================================
# EJECUTAR MÉTODOS
# ============================================================

@st.cache_data
def ejecutar_metodos_recomendados(
    df,
    recomendaciones
):

    resultados = []


    if not recomendaciones:

        return pd.DataFrame()


    metodos = (
        recomendaciones
        .get(
            "metodos_supuestos",
            []
        )
    )


    numericas = (
        df
        .select_dtypes(
            include=np.number
        )
    )


    # --------------------------------------------------------
    # Por cada variable
    # --------------------------------------------------------

    for col in numericas.columns:

        datos = (
            numericas[col]
            .dropna()
            .astype(float)
        )


        n = len(datos)


        if n < 8:

            continue


        # ----------------------------------------------------
        # Por cada método recomendado
        # ----------------------------------------------------

        for recomendacion in metodos:

            metodo = str(
                recomendacion.get(
                    "metodo",
                    ""
                )
            ).lower().strip()


            try:

                # ============================================
                # SHAPIRO
                # ============================================

                if metodo == "shapiro":

                    # SciPy establece el límite práctico
                    # de Shapiro en muestras grandes.

                    if n > 5000:

                        continue


                    estadistico, p_valor = (
                        stats.shapiro(
                            datos
                        )
                    )


                    resultados.append({

                        "Variable":
                            col,

                        "Prueba":
                            "Shapiro-Wilk",

                        "Estadístico":
                            float(
                                estadistico
                            ),

                        "p-valor":
                            float(
                                p_valor
                            ),

                        "n":
                            n,

                        "α":
                            0.05,

                        "Decisión":
                            (
                                "Rechazar H₀"
                                if p_valor < 0.05
                                else
                                "No rechazar H₀"
                            ),

                        "Resultado":
                            (
                                "Evidencia contra "
                                "la normalidad"
                                if p_valor < 0.05
                                else
                                "No hay evidencia "
                                "suficiente contra "
                                "la normalidad"
                            )

                    })


                # ============================================
                # D'AGOSTINO
                # ============================================

                elif metodo == "dagostino":

                    if n < 20:

                        continue


                    estadistico, p_valor = (
                        stats.normaltest(
                            datos
                        )
                    )


                    resultados.append({

                        "Variable":
                            col,

                        "Prueba":
                            "D'Agostino K²",

                        "Estadístico":
                            float(
                                estadistico
                            ),

                        "p-valor":
                            float(
                                p_valor
                            ),

                        "n":
                            n,

                        "α":
                            0.05,

                        "Decisión":
                            (
                                "Rechazar H₀"
                                if p_valor < 0.05
                                else
                                "No rechazar H₀"
                            ),

                        "Resultado":
                            (
                                "Evidencia contra "
                                "la normalidad"
                                if p_valor < 0.05
                                else
                                "No hay evidencia "
                                "suficiente contra "
                                "la normalidad"
                            )

                    })


                # ============================================
                # SKEW TEST
                # ============================================

                elif metodo == "skewtest":

                    if n < 8:

                        continue


                    estadistico, p_valor = (
                        stats.skewtest(
                            datos
                        )
                    )


                    resultados.append({

                        "Variable":
                            col,

                        "Prueba":
                            "Test de asimetría",

                        "Estadístico":
                            float(
                                estadistico
                            ),

                        "p-valor":
                            float(
                                p_valor
                            ),

                        "n":
                            n,

                        "α":
                            0.05,

                        "Decisión":
                            (
                                "Rechazar H₀"
                                if p_valor < 0.05
                                else
                                "No rechazar H₀"
                            ),

                        "Resultado":
                            (
                                "Asimetría "
                                "estadísticamente "
                                "significativa"
                                if p_valor < 0.05
                                else
                                "No hay evidencia "
                                "suficiente de "
                                "asimetría significativa"
                            )

                    })


                # ============================================
                # KURTOSIS TEST
                # ============================================

                elif metodo == "kurtosistest":

                    if n < 20:

                        continue


                    estadistico, p_valor = (
                        stats.kurtosistest(
                            datos
                        )
                    )


                    resultados.append({

                        "Variable":
                            col,

                        "Prueba":
                            "Test de curtosis",

                        "Estadístico":
                            float(
                                estadistico
                            ),

                        "p-valor":
                            float(
                                p_valor
                            ),

                        "n":
                            n,

                        "α":
                            0.05,

                        "Decisión":
                            (
                                "Rechazar H₀"
                                if p_valor < 0.05
                                else
                                "No rechazar H₀"
                            ),

                        "Resultado":
                            (
                                "Curtosis "
                                "estadísticamente "
                                "significativa"
                                if p_valor < 0.05
                                else
                                "No hay evidencia "
                                "suficiente de "
                                "curtosis significativa"
                            )

                    })


            except Exception:

                # Si una prueba no puede ejecutarse
                # en una variable, continúa con las demás.

                continue


    if not resultados:

        return pd.DataFrame()


    resultado_final = pd.DataFrame(
        resultados
    )


    return resultado_final


# ============================================================
# IMPUTACIÓN
# ============================================================

@st.cache_data
def ejecutar_imputacion(
    df,
    recomendacion
):

    df_resultado = df.copy()

    log = []


    if not recomendacion:

        return (
            df_resultado,
            pd.DataFrame()
        )


    metodo = str(
        recomendacion.get(
            "metodo",
            "ninguna"
        )
    ).lower().strip()


    # ========================================================
    # NINGUNA
    # ========================================================

    if metodo == "ninguna":

        return (
            df_resultado,
            pd.DataFrame()
        )


    # ========================================================
    # KNN
    # ========================================================

    if metodo == "knn":

        numericas = (
            df_resultado
            .select_dtypes(
                include=np.number
            )
            .columns
        )


        if len(numericas) > 0:

            faltantes = (
                df_resultado[
                    numericas
                ]
                .isna()
                .sum()
            )


            if faltantes.sum() > 0:

                knn = KNNImputer(

                    n_neighbors=5,

                    weights="distance"

                )


                df_resultado[
                    numericas
                ] = knn.fit_transform(

                    df_resultado[
                        numericas
                    ]

                )


                for col in numericas:

                    if faltantes[col] > 0:

                        log.append({

                            "Variable":
                                col,

                            "Faltantes":
                                int(
                                    faltantes[col]
                                ),

                            "Método":
                                "KNN",

                            "Detalle":
                                "KNN multivariado "
                                "con k=5 y pesos "
                                "por distancia"

                        })


    # ========================================================
    # MEDIANA
    # ========================================================

    elif metodo == "mediana":

        numericas = (
            df_resultado
            .select_dtypes(
                include=np.number
            )
            .columns
        )


        for col in numericas:

            cantidad = int(
                df_resultado[col]
                .isna()
                .sum()
            )


            if cantidad > 0:

                valor = (
                    df_resultado[col]
                    .median()
                )


                df_resultado[col] = (
                    df_resultado[col]
                    .fillna(valor)
                )


                log.append({

                    "Variable":
                        col,

                    "Faltantes":
                        cantidad,

                    "Método":
                        "Mediana",

                    "Detalle":
                        f"Mediana = {valor:.4f}"

                })


    # ========================================================
    # MODA
    # ========================================================

    elif metodo == "moda":

        categoricas = (
            df_resultado
            .select_dtypes(
                exclude=np.number
            )
            .columns
        )


        for col in categoricas:

            cantidad = int(
                df_resultado[col]
                .isna()
                .sum()
            )


            if cantidad > 0:

                moda = (
                    df_resultado[col]
                    .mode()
                )


                if len(moda) > 0:

                    valor = moda.iloc[0]


                    df_resultado[col] = (
                        df_resultado[col]
                        .fillna(valor)
                    )


                    log.append({

                        "Variable":
                            col,

                        "Faltantes":
                            cantidad,

                        "Método":
                            "Moda",

                        "Detalle":
                            f"Moda = {valor}"

                    })


    return (
        df_resultado,
        pd.DataFrame(log)
    )


# ============================================================
# IA: INTERPRETAR RESULTADOS
# ============================================================

@st.cache_data(
    show_spinner=False
)
def interpretar_resultados_ia(

    nombre_bd,

    resumen,

    recomendaciones,

    resultados_supuestos,

    log_imputacion

):

    # --------------------------------------------------------
    # Convertir resultados a JSON
    # --------------------------------------------------------

    if not resultados_supuestos.empty:

        resultados_json = (
            resultados_supuestos
            .to_json(
                orient="records",
                force_ascii=False
            )
        )

    else:

        resultados_json = "[]"


    if not log_imputacion.empty:

        imputacion_json = (
            log_imputacion
            .to_json(
                orient="records",
                force_ascii=False
            )
        )

    else:

        imputacion_json = "[]"


    # --------------------------------------------------------
    # PROMPT
    # --------------------------------------------------------

    prompt = f"""
Eres un estadístico senior realizando
una consultoría profesional.

Tu función es INTERPRETAR resultados estadísticos
que ya fueron calculados por Python.

NO debes volver a calcular resultados.

NO debes inventar números.

NO debes inventar p-valores.

NO debes inventar estadísticos.

NO debes cambiar los valores recibidos.

==================================================
BASE
==================================================

{nombre_bd}


==================================================
DIAGNÓSTICO Y DESCRIPTIVOS
==================================================

{json.dumps(
    resumen,
    ensure_ascii=False,
    indent=2
)}


==================================================
RECOMENDACIONES DEL AGENTE
==================================================

{json.dumps(
    recomendaciones,
    ensure_ascii=False,
    indent=2
)}


==================================================
RESULTADOS REALES DE LAS PRUEBAS
==================================================

{resultados_json}


==================================================
IMPUTACIÓN
==================================================

{imputacion_json}


==================================================
REGLAS DE INTERPRETACIÓN
==================================================

1. SIEMPRE diferencia:

   - Estadístico de prueba
   - p-valor
   - alfa
   - decisión

2. El estadístico de prueba NO es el p-valor.

3. Si p < 0.05:

   Existe evidencia estadística suficiente
   para rechazar H₀.

4. Si p >= 0.05:

   No existe evidencia estadística suficiente
   para rechazar H₀.

5. Nunca escribas:

   "Se acepta H₀".

6. Utiliza:

   "No se rechaza H₀".

7. Explica qué significa el resultado
   estadístico en lenguaje sencillo.

8. Diferencia significancia estadística
   de importancia práctica.

9. Analiza las estadísticas descriptivas
   disponibles:

   - media
   - mediana
   - desviación estándar
   - mínimo
   - máximo
   - cuartiles
   - varianza
   - asimetría
   - curtosis

10. Analiza faltantes.

11. Analiza correlaciones cuando existan.

12. Si una conclusión no puede obtenerse
    con la información disponible,
    dilo explícitamente.

13. Cada conclusión debe estar sustentada
    por los datos proporcionados.

14. No inventes interpretaciones de negocio
    que no puedan justificarse con la base.


==================================================
OBJETIVO
==================================================

Genera una interpretación profesional
específica para esta base.


==================================================
FORMATO
==================================================

Devuelve únicamente JSON.
"""


    # --------------------------------------------------------
    # SCHEMA
    # --------------------------------------------------------

    schema = {

        "type": "object",

        "properties": {

            "resumen_ejecutivo": {

                "type": "string"

            },


            "exploracion_descriptiva": {

                "type": "array",

                "items": {

                    "type": "string"

                }

            },


            "analisis_supuestos": {

                "type": "array",

                "items": {

                    "type": "object",

                    "properties": {

                        "variable": {

                            "type": "string"

                        },

                        "prueba": {

                            "type": "string"

                        },

                        "estadistico": {

                            "type": "string"

                        },

                        "p_valor": {

                            "type": "string"

                        },

                        "decision": {

                            "type": "string"

                        },

                        "interpretacion": {

                            "type": "string"

                        }

                    },

                    "required": [

                        "variable",

                        "prueba",

                        "estadistico",

                        "p_valor",

                        "decision",

                        "interpretacion"

                    ]

                }

            },


            "analisis_imputacion": {

                "type": "string"

            },


            "conclusiones": {

                "type": "array",

                "items": {

                    "type": "string"

                }

            },


            "recomendaciones_estadisticas": {

                "type": "array",

                "items": {

                    "type": "string"

                }

            }

        },

        "required": [

            "resumen_ejecutivo",

            "exploracion_descriptiva",

            "analisis_supuestos",

            "analisis_imputacion",

            "conclusiones",

            "recomendaciones_estadisticas"

        ]

    }


    return llamar_gemini(
        prompt,
        schema
    )


# ============================================================
# GRÁFICOS AUTOMÁTICOS
# ============================================================

@st.cache_data
def filtrar_graficos_automaticos(
    nombre_bd
):

    if not CHARTS_DIR.exists():

        return []


    todos = [

        f

        for f in CHARTS_DIR.iterdir()

        if f.suffix.lower()
        in [
            ".png",
            ".jpg",
            ".jpeg"
        ]

    ]


    nombre_lower = (
        nombre_bd.lower()
    )


    claves = []


    if "agr" in nombre_lower:

        claves = [

            "agr",
            "cosecha",
            "cultiv",
            "fert",
            "alim",
            "animal"

        ]


    elif "hci" in nombre_lower:

        claves = [

            "hci",
            "biplot",
            "kmeans",
            "pca",
            "cluster"

        ]


    elif (
        "ingreso" in nombre_lower
        or "sexo" in nombre_lower
    ):

        claves = [

            "ingreso",
            "sexo",
            "brecha"

        ]


    if claves:

        filtrados = [

            g

            for g in todos

            if any(
                c in g.name.lower()
                for c in claves
            )

        ]


        if filtrados:

            return filtrados


    return todos


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title(
        "🧠 Panel de Consultoría"
    )


    st.caption(
        "Agente estadístico asistido por IA"
    )


    st.markdown("---")


    # ========================================================
    # BASE
    # ========================================================

    archivos = (
        escanear_bases_de_datos()
    )


    if not archivos:

        st.error(
            "No se encontraron archivos CSV o XLSX."
        )

        st.stop()


    seleccion = st.selectbox(

        "📂 Selecciona tu base",

        list(
            archivos.keys()
        )

    )


    ruta = archivos[
        seleccion
    ]


    df_raw = cargar_datos(
        ruta
    )


    st.markdown("---")


    # ========================================================
    # ESTADO IA
    # ========================================================

    if API_KEY:

        st.success(
            "🟢 Agente IA conectado"
        )


        st.caption(
            f"Modelo: {GEMINI_MODEL}"
        )


        st.caption(
            f"Fuente de API Key: {API_SOURCE}"
        )


    else:

        st.error(
            "🔴 Agente IA desconectado"
        )


        st.caption(
            "No se encontró GEMINI_API_KEY."
        )


    st.markdown("---")


    # ========================================================
    # FLUJO
    # ========================================================

    st.markdown(
        """
        **Flujo del agente**

        🔍 Diagnóstico

        ↓

        🤖 Recomendación IA

        ↓

        📐 Cálculo estadístico

        ↓

        🧠 Interpretación IA

        ↓

        📊 Dashboard
        """
    )


# ============================================================
# VALIDAR BASE
# ============================================================

if df_raw.empty:

    st.error(
        "La base seleccionada está vacía."
    )

    st.stop()


# ============================================================
# DIAGNÓSTICO
# ============================================================

with st.spinner(
    "🔍 Diagnosticando la base..."
):

    diagnostico = (
        construir_diagnostico(
            df_raw
        )
    )


    descriptivos = (
        obtener_descriptivos(
            df_raw
        )
    )


    perfil = (
        perfil_variables(
            df_raw
        )
    )


    correlaciones = (
        obtener_correlaciones(
            df_raw
        )
    )


    resumen = (
        preparar_resumen_para_ia(
            seleccion,
            df_raw
        )
    )


# ============================================================
# RECOMENDACIÓN DE MÉTODOS
# ============================================================

with st.spinner(
    "🤖 La IA está seleccionando "
    "los métodos estadísticos..."
):

    if API_KEY:

        recomendaciones = (
            recomendar_metodos_ia(

                seleccion,

                json.dumps(
                    resumen,
                    ensure_ascii=False,
                    indent=2
                )

            )
        )

    else:

        recomendaciones = {

            "error":
                "IA desconectada."

        }


# ============================================================
# EJECUTAR MÉTODOS
# ============================================================

if (
    recomendaciones
    and
    "error" not in recomendaciones
):

    resultados_supuestos = (
        ejecutar_metodos_recomendados(

            df_raw,

            recomendaciones

        )
    )


    recomendacion_imp = (
        recomendaciones
        .get(
            "imputacion",
            {}
        )
    )


    df_imp, log_imp = (
        ejecutar_imputacion(

            df_raw,

            recomendacion_imp

        )
    )


else:

    resultados_supuestos = (
        pd.DataFrame()
    )


    df_imp = (
        df_raw.copy()
    )


    log_imp = (
        pd.DataFrame()
    )


# ============================================================
# INTERPRETACIÓN IA
# ============================================================

with st.spinner(
    "🧠 La IA está interpretando "
    "los resultados..."
):

    if (
        API_KEY
        and
        "error" not in recomendaciones
    ):

        interpretacion = (
            interpretar_resultados_ia(

                seleccion,

                resumen,

                recomendaciones,

                resultados_supuestos,

                log_imp

            )
        )

    else:

        interpretacion = {

            "error":
                "No se pudo generar "
                "la interpretación porque "
                "la IA no está disponible."

        }


# ============================================================
# GRÁFICOS
# ============================================================

graficos_automaticos = (
    filtrar_graficos_automaticos(
        seleccion
    )
)


# ============================================================
# TÍTULO
# ============================================================

st.title(
    "📊 Análisis Integral"
)


st.subheader(
    seleccion.split(" (")[0]
)


st.markdown(
    """
    El agente diagnostica la información,
    recomienda los métodos estadísticos,
    ejecuta los cálculos y posteriormente
    interpreta los resultados obtenidos.
    """
)


# ============================================================
# MÉTRICAS PRINCIPALES
# ============================================================

m1, m2, m3, m4, m5 = (
    st.columns(5)
)


m1.metric(
    "📋 Registros",
    f"{diagnostico['filas']:,}"
)


m2.metric(
    "📊 Variables",
    diagnostico["columnas"]
)


m3.metric(
    "🔢 Numéricas",
    diagnostico["numericas"]
)


m4.metric(
    "⚠️ Faltantes",
    diagnostico["faltantes"]
)


m5.metric(
    "♻️ Duplicados",
    diagnostico["duplicados"]
)


# ============================================================
# ESTADO GENERAL DE LA IA
# ============================================================

if (
    "error" in recomendaciones
):

    st.markdown(
        f"""
        <div class="warning-card">

        <h3>⚠️ IA no disponible</h3>

        <p>
        {recomendaciones["error"]}
        </p>

        <p>
        Los cálculos que dependen de la
        recomendación de la IA no se ejecutaron.
        No se presentarán conclusiones falsas.
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )

else:

    st.markdown(
        """
        <div class="success-card">

        <h3>🟢 Flujo estadístico completado</h3>

        <p>
        La IA recomendó los métodos,
        Python ejecutó los cálculos y
        posteriormente la IA interpretó
        los resultados.
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs(

    [

        "🔍 Exploración",

        "🤖 Métodos recomendados",

        "📐 Supuestos",

        "🧹 Imputación",

        "📈 Interpretación"

    ]

)


# ============================================================
# TAB 1
# EXPLORACIÓN
# ============================================================

with tab1:

    st.header(
        "🔍 Exploración descriptiva"
    )


    # --------------------------------------------------------
    # RESUMEN IA
    # --------------------------------------------------------

    if (
        "error" not in interpretacion
    ):

        st.markdown(
            f"""
            <div class="ai-card">

            <h3>
            🧠 Análisis descriptivo realizado por IA
            </h3>

            <p>
            {interpretacion.get(
                "resumen_ejecutivo",
                "Sin interpretación."
            )}
            </p>

            </div>
            """,
            unsafe_allow_html=True
        )


    # --------------------------------------------------------
    # DESCRIPTIVOS
    # --------------------------------------------------------

    st.subheader(
        "📊 Estadísticas descriptivas"
    )


    if not descriptivos.empty:

        st.dataframe(

            descriptivos,

            use_container_width=True,

            height=450

        )

    else:

        st.info(
            "No existen variables numéricas."
        )


    # --------------------------------------------------------
    # HALLAZGOS DESCRIPTIVOS
    # --------------------------------------------------------

    if (
        "error" not in interpretacion
    ):

        st.subheader(
            "💡 Hallazgos descriptivos"
        )


        hallazgos = (
            interpretacion
            .get(
                "exploracion_descriptiva",
                []
            )
        )


        for hallazgo in hallazgos:

            st.markdown(
                f"""
                <div class="dashboard-card">

                🔹 {hallazgo}

                </div>
                """,
                unsafe_allow_html=True
            )


    # --------------------------------------------------------
    # CORRELACIONES
    # --------------------------------------------------------

    if not correlaciones.empty:

        st.subheader(
            "🔗 Correlaciones"
        )


        st.dataframe(

            correlaciones.head(20),

            use_container_width=True

        )


    # --------------------------------------------------------
    # PERFIL
    # --------------------------------------------------------

    with st.expander(
        "🔎 Ver perfil completo de variables"
    ):

        st.dataframe(

            perfil,

            use_container_width=True,

            height=450

        )


    # --------------------------------------------------------
    # BASE
    # --------------------------------------------------------

    with st.expander(
        "🗃️ Ver base de datos completa"
    ):

        st.dataframe(

            df_raw,

            use_container_width=True,

            height=450

        )


# ============================================================
# TAB 2
# MÉTODOS
# ============================================================

with tab2:

    st.header(
        "🤖 Métodos recomendados por el agente"
    )


    if (
        "error" in recomendaciones
    ):

        st.error(
            recomendaciones["error"]
        )


        st.info(
            """
            Cuando Gemini esté disponible,
            esta sección mostrará los métodos
            que el agente considere apropiados
            para la estructura de esta base.
            """
        )


    else:

        # ----------------------------------------------------
        # ESTRATEGIA
        # ----------------------------------------------------

        st.markdown(
            f"""
            <div class="ai-card">

            <h3>
            🧠 Estrategia estadística propuesta
            </h3>

            <p>
            {recomendaciones.get(
                "estrategia_general",
                "Sin estrategia."
            )}
            </p>

            </div>
            """,
            unsafe_allow_html=True
        )


        # ----------------------------------------------------
        # MÉTODOS DE SUPUESTOS
        # ----------------------------------------------------

        st.subheader(
            "📐 Métodos de supuestos"
        )


        metodos = (
            recomendaciones
            .get(
                "metodos_supuestos",
                []
            )
        )


        if metodos:

            for metodo in metodos:

                nombre = (
                    metodo
                    .get(
                        "metodo",
                        ""
                    )
                    .upper()
                )


                justificacion = (
                    metodo
                    .get(
                        "justificacion",
                        ""
                    )
                )


                st.markdown(
                    f"""
                    <span class="method-badge">
                    {nombre}
                    </span>

                    <div class="dashboard-card">

                    {justificacion}

                    </div>
                    """,
                    unsafe_allow_html=True
                )

        else:

            st.info(
                "El agente no recomendó pruebas "
                "de supuestos para esta base."
            )


        # ----------------------------------------------------
        # IMPUTACIÓN
        # ----------------------------------------------------

        st.subheader(
            "🧹 Decisión sobre datos faltantes"
        )


        imp = (
            recomendaciones
            .get(
                "imputacion",
                {}
            )
        )


        st.markdown(
            f"""
            <div class="dashboard-card">

            <h4>
            Método recomendado:
            {imp.get("metodo", "No definido")}
            </h4>

            <p>
            <strong>Recomendación:</strong>
            {imp.get(
                "recomendacion",
                "No disponible."
            )}
            </p>

            <p>
            <strong>Justificación:</strong>
            {imp.get(
                "justificacion",
                "No disponible."
            )}
            </p>

            </div>
            """,
            unsafe_allow_html=True
        )


        # ----------------------------------------------------
        # OTROS ANÁLISIS
        # ----------------------------------------------------

        st.subheader(
            "📊 Análisis adicionales recomendados"
        )


        analisis = (
            recomendaciones
            .get(
                "analisis_recomendado",
                []
            )
        )


        if analisis:

            for item in analisis:

                st.markdown(
                    f"""
                    <div class="dashboard-card">

                    <strong>
                    {item.get(
                        "analisis",
                        ""
                    )}
                    </strong>

                    <br><br>

                    {item.get(
                        "justificacion",
                        ""
                    )}

                    </div>
                    """,
                    unsafe_allow_html=True
                )

        else:

            st.info(
                "No se recomendaron análisis adicionales."
            )


# ============================================================
# TAB 3
# SUPUESTOS
# ============================================================

with tab3:

    st.header(
        "📐 Auditoría estadística de supuestos"
    )


    # --------------------------------------------------------
    # ERROR DE IA
    # --------------------------------------------------------

    if (
        "error" in recomendaciones
    ):

        st.error(
            """
            No se pueden determinar automáticamente
            los métodos porque la IA no respondió.
            """
        )


    # --------------------------------------------------------
    # RESULTADOS
    # --------------------------------------------------------

    elif resultados_supuestos.empty:

        st.warning(
            """
            La IA respondió correctamente,
            pero no se generaron pruebas de
            supuestos para las variables disponibles.
            """
        )


    else:

        st.subheader(
            "📋 Resultados estadísticos"
        )


        # ----------------------------------------------------
        # TABLA
        # ----------------------------------------------------

        tabla_supuestos = (
            resultados_supuestos.copy()
        )


        # Redondear números

        if "Estadístico" in tabla_supuestos:

            tabla_supuestos[
                "Estadístico"
            ] = (
                tabla_supuestos[
                    "Estadístico"
                ].round(6)
            )


        if "p-valor" in tabla_supuestos:

            tabla_supuestos[
                "p-valor"
            ] = (
                tabla_supuestos[
                    "p-valor"
                ].round(8)
            )


        st.dataframe(

            tabla_supuestos,

            use_container_width=True,

            height=450

        )


        st.caption(
            """
            ⚠️ El estadístico de prueba y el
            p-valor se muestran por separado.
            """
        )


        # ----------------------------------------------------
        # INTERPRETACIONES
        # ----------------------------------------------------

        if (
            "error" not in interpretacion
        ):

            st.subheader(
                "🧠 Interpretación de la IA"
            )


            interpretaciones = (
                interpretacion
                .get(
                    "analisis_supuestos",
                    []
                )
            )


            if interpretaciones:

                for item in interpretaciones:

                    variable = item.get(
                        "variable",
                        ""
                    )


                    prueba = item.get(
                        "prueba",
                        ""
                    )


                    with st.expander(
                        f"📊 {variable} — {prueba}"
                    ):

                        c1, c2, c3 = (
                            st.columns(3)
                        )


                        c1.metric(
                            "Estadístico",
                            item.get(
                                "estadistico",
                                "-"
                            )
                        )


                        c2.metric(
                            "p-valor",
                            item.get(
                                "p_valor",
                                "-"
                            )
                        )


                        c3.metric(
                            "Decisión",
                            item.get(
                                "decision",
                                "-"
                            )
                        )


                        st.markdown(
                            "### Interpretación"
                        )


                        st.write(
                            item.get(
                                "interpretacion",
                                ""
                            )
                        )


# ============================================================
# TAB 4
# IMPUTACIÓN
# ============================================================

with tab4:

    st.header(
        "🧹 Imputación de datos faltantes"
    )


    # --------------------------------------------------------
    # ERROR
    # --------------------------------------------------------

    if (
        "error" in recomendaciones
    ):

        st.error(
            """
            La IA no respondió.

            Por lo tanto, el sistema NO puede afirmar
            que la imputación sea necesaria o innecesaria.
            """
        )


    else:

        imp = (
            recomendaciones
            .get(
                "imputacion",
                {}
            )
        )


        # ----------------------------------------------------
        # DECISIÓN IA
        # ----------------------------------------------------

        st.markdown(
            f"""
            <div class="ai-card">

            <h3>
            🤖 Decisión del agente
            </h3>

            <p>
            <strong>Método:</strong>
            {imp.get(
                "metodo",
                "No definido"
            )}
            </p>

            <p>
            <strong>Recomendación:</strong>
            {imp.get(
                "recomendacion",
                ""
            )}
            </p>

            <p>
            <strong>Justificación:</strong>
            {imp.get(
                "justificacion",
                ""
            )}
            </p>

            </div>
            """,
            unsafe_allow_html=True
        )


        # ----------------------------------------------------
        # RESULTADOS
        # ----------------------------------------------------

        if not log_imp.empty:

            st.subheader(
                "📋 Registro de imputación"
            )


            st.dataframe(

                log_imp,

                use_container_width=True

            )


            st.subheader(
                "🗃️ Base después de la imputación"
            )


            st.dataframe(

                df_imp,

                use_container_width=True,

                height=450

            )


        else:

            metodo_imp = (
                imp.get(
                    "metodo",
                    ""
                ).lower()
            )


            if metodo_imp == "ninguna":

                st.success(
                    """
                    ✓ El agente analizó los datos
                    faltantes y determinó que no
                    era necesaria una imputación.
                    """
                )

            else:

                st.warning(
                    """
                    El agente recomendó un método,
                    pero no se encontraron valores
                    faltantes que requirieran
                    transformación.
                    """
                )


# ============================================================
# TAB 5
# INTERPRETACIÓN Y GRÁFICOS
# ============================================================

with tab5:

    st.header(
        "📈 Interpretación integral"
    )


    # --------------------------------------------------------
    # ERROR
    # --------------------------------------------------------

    if (
        "error" in interpretacion
    ):

        st.error(
            interpretacion["error"]
        )


    else:

        # ====================================================
        # RESUMEN
        # ====================================================

        st.subheader(
            "🧠 Resumen ejecutivo"
        )


        st.markdown(
            f"""
            <div class="ai-card">

            {interpretacion.get(
                "resumen_ejecutivo",
                ""
            )}

            </div>
            """,
            unsafe_allow_html=True
        )


        # ====================================================
        # CONCLUSIONES
        # ====================================================

        st.subheader(
            "💡 Conclusiones"
        )


        conclusiones = (
            interpretacion
            .get(
                "conclusiones",
                []
            )
        )


        if conclusiones:

            for i, conclusion in enumerate(
                conclusiones,
                start=1
            ):

                st.markdown(
                    f"""
                    <div class="success-card">

                    <strong>
                    Conclusión {i}
                    </strong>

                    <br><br>

                    {conclusion}

                    </div>
                    """,
                    unsafe_allow_html=True
                )

        else:

            st.info(
                "La IA no generó conclusiones."
            )


        # ====================================================
        # IMPUTACIÓN
        # ====================================================

        st.subheader(
            "🧹 Interpretación de la imputación"
        )


        st.markdown(
            f"""
            <div class="dashboard-card">

            {interpretacion.get(
                "analisis_imputacion",
                "No disponible."
            )}

            </div>
            """,
            unsafe_allow_html=True
        )


        # ====================================================
        # RECOMENDACIONES
        # ====================================================

        st.subheader(
            "🎯 Recomendaciones estadísticas"
        )


        recomendaciones_finales = (
            interpretacion
            .get(
                "recomendaciones_estadisticas",
                []
            )
        )


        if recomendaciones_finales:

            for recomendacion in (
                recomendaciones_finales
            ):

                st.markdown(
                    f"""
                    <div class="dashboard-card">

                    📌 {recomendacion}

                    </div>
                    """,
                    unsafe_allow_html=True
                )


    # ========================================================
    # GRÁFICOS
    # ========================================================

    st.divider()


    st.header(
        "📊 Visualizaciones"
    )


    if graficos_automaticos:

        for i in range(
            0,
            len(
                graficos_automaticos
            ),
            2
        ):

            columnas = st.columns(
                2
            )


            for j, columna in enumerate(
                columnas
            ):

                indice = i + j


                if indice >= len(
                    graficos_automaticos
                ):

                    continue


                img_path = (
                    graficos_automaticos[
                        indice
                    ]
                )


                with columna:

                    try:

                        st.image(

                            Image.open(
                                img_path
                            ),

                            caption=(
                                img_path.stem
                            ),

                            use_container_width=True

                        )

                    except Exception as e:

                        st.warning(
                            f"No se pudo mostrar "
                            f"{img_path.name}: {e}"
                        )


    else:

        st.info(
            """
            No se detectaron gráficos asociados
            a esta base.
            """
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()


st.caption(
    f"""
    🧠 Agente de Consultoría Estadística

    · Modelo IA: {GEMINI_MODEL}

    · Estadísticos calculados por Python

    · p-valores calculados por Python

    · Interpretaciones generadas por IA

    · API Key protegida mediante secrets
    """
)