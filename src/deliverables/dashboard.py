import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from src.data.loader import load_data

st.set_page_config(
    page_title="PaperStats Dashboard",
    page_icon="📊",
    layout="wide"
)

# ─────────────────────────────────────────────
# SIDEBAR: selector de archivos
# ─────────────────────────────────────────────
st.sidebar.header("📂 Fuente de datos")

DATA_RAW = Path("data/raw")
archivos_disponibles = []
if DATA_RAW.exists():
    archivos_disponibles = [
        f for f in DATA_RAW.iterdir()
        if f.suffix.lower() in {".csv", ".xlsx", ".xls"}
    ]

opcion_fuente = st.sidebar.radio(
    "¿De dónde cargar los datos?",
    ["Archivos del proyecto (data/raw/)", "Subir archivo desde mi PC"],
)

df = None
nombre_dataset = ""

if opcion_fuente == "Archivos del proyecto (data/raw/)":
    if not archivos_disponibles:
        st.sidebar.warning("No hay archivos CSV/Excel en data/raw/")
    else:
        archivo_elegido = st.sidebar.selectbox(
            "Selecciona un archivo:",
            archivos_disponibles,
            format_func=lambda p: p.name,
        )
        if archivo_elegido:
            try:
                df = load_data(str(archivo_elegido))
                nombre_dataset = archivo_elegido.name
            except Exception as e:
                st.sidebar.error(f"No se pudo cargar: {e}")

else:  # Subir archivo
    archivo_subido = st.sidebar.file_uploader(
        "Arrastra o selecciona tu archivo:",
        type=["csv", "xlsx", "xls"],
    )
    if archivo_subido is not None:
        try:
            if archivo_subido.name.endswith(".csv"):
                df = pd.read_csv(archivo_subido)
            else:
                df = pd.read_excel(archivo_subido)
            nombre_dataset = archivo_subido.name
        except Exception as e:
            st.sidebar.error(f"Error al leer: {e}")

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.title("📊 PaperStats — Dashboard Interactivo")
if nombre_dataset:
    st.subheader(f"Dataset: {nombre_dataset}")
else:
    st.info("👈 Selecciona o sube un archivo en la barra lateral para comenzar.")
    st.stop()

# ─────────────────────────────────────────────
# FILTROS
# ─────────────────────────────────────────────
st.sidebar.header("🔍 Filtros")
categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
df_filtered = df.copy()

if categorical_cols:
    filtro_col = st.sidebar.selectbox(
        "Filtrar por variable categórica:",
        [None] + categorical_cols,
    )
    if filtro_col:
        valores = st.sidebar.multiselect(
            f"Valores de {filtro_col}:",
            options=sorted(df[filtro_col].dropna().unique()),
            default=sorted(df[filtro_col].dropna().unique()),
        )
        if valores:
            df_filtered = df_filtered[df_filtered[filtro_col].isin(valores)]

# ─────────────────────────────────────────────
# MÉTRICAS
# ─────────────────────────────────────────────
st.header("📈 Resumen del dataset")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Registros", len(df_filtered))
with col2:
    st.metric("Variables", len(df_filtered.columns))
with col3:
    numeric_cols = df_filtered.select_dtypes(include=["number"]).columns
    st.metric("Variables numéricas", len(numeric_cols))
with col4:
    st.metric("Valores faltantes", int(df_filtered.isnull().sum().sum()))

# ─────────────────────────────────────────────
# GRÁFICOS
# ─────────────────────────────────────────────
st.header("📊 Visualizaciones")
tipo_grafico = st.selectbox(
    "Tipo de gráfico:",
    ["Histograma", "Boxplot", "Dispersión", "Correlación"],
)

if tipo_grafico == "Histograma" and len(numeric_cols) > 0:
    var = st.selectbox("Variable:", numeric_cols)
    fig = px.histogram(df_filtered, x=var, nbins=30, title=f"Distribución de {var}")
    st.plotly_chart(fig, use_container_width=True)

elif tipo_grafico == "Boxplot" and len(numeric_cols) > 0:
    var = st.selectbox("Variable:", numeric_cols)
    color = st.selectbox("Agrupar por (opcional):", [None] + categorical_cols)
    fig = px.box(df_filtered, y=var, color=color, title=f"Boxplot de {var}")
    st.plotly_chart(fig, use_container_width=True)

elif tipo_grafico == "Dispersión" and len(numeric_cols) >= 2:
    col_x = st.selectbox("Eje X:", numeric_cols, index=0)
    col_y = st.selectbox("Eje Y:", numeric_cols, index=1)
    color = st.selectbox("Color por (opcional):", [None] + categorical_cols)
    fig = px.scatter(
        df_filtered, x=col_x, y=col_y,
        color=color if color else None,
        title=f"{col_x} vs {col_y}",
    )
    st.plotly_chart(fig, use_container_width=True)

elif tipo_grafico == "Correlación" and len(numeric_cols) >= 2:
    corr = df_filtered[numeric_cols].corr()
    fig = px.imshow(
        corr,
        title="Matriz de correlación (Pearson)",
        color_continuous_scale="RdBu_r",
        aspect="auto",
    )
    st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────
# TABLA + DESCRIPTIVAS
# ─────────────────────────────────────────────
st.header("📋 Explorar datos")
st.dataframe(df_filtered, use_container_width=True)

st.header("📐 Estadísticas descriptivas")
if len(numeric_cols) > 0:
    st.write(df_filtered[numeric_cols].describe())

# ─────────────────────────────────────────────
# EXPORTAR
# ─────────────────────────────────────────────
st.sidebar.header("💾 Exportar")
csv = df_filtered.to_csv(index=False).encode("utf-8")
st.sidebar.download_button(
    label="Descargar CSV filtrado",
    data=csv,
    file_name=f"{Path(nombre_dataset).stem}_filtrado.csv",
    mime="text/csv",
)