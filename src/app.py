from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# Rutas y configuración del modelo -----------------------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT_DIR / "models"
DATA_PATH = ROOT_DIR / "data" / "raw" / "housing.csv"

FEATURES = ["Latitude", "Longitude", "MedInc"]

CLUSTER_LABELS = {
    0: "Zona centro-interior · ingreso bajo",
    1: "Zona sur · ingreso alto",
    2: "Zona centro-sur · ingreso muy alto",
    3: "Zona norte-interior · ingreso bajo",
    4: "Zona norte · ingreso alto",
    5: "Zona sur · ingreso bajo",
}

CLUSTER_COLORS = {
    "Zona centro-interior · ingreso bajo": "#0071e3",
    "Zona sur · ingreso alto": "#34c759",
    "Zona centro-sur · ingreso muy alto": "#af52de",
    "Zona norte-interior · ingreso bajo": "#ff9f0a",
    "Zona norte · ingreso alto": "#ff2d55",
    "Zona sur · ingreso bajo": "#00c7be",
}


# Carga de datos y modelo --------------------------------------------------------------------------------------------

@st.cache_resource
def load_artifacts():
    scaler = joblib.load(MODEL_DIR / "scaler.pkl")
    kmeans = joblib.load(MODEL_DIR / "kmeans.pkl")
    return scaler, kmeans


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


# Lógica de predicción -----------------------------------------------------------------------------------------------

def predict_cluster(samples: pd.DataFrame, scaler, kmeans) -> np.ndarray:
    scaled_samples = scaler.transform(samples[FEATURES])
    scaled_samples = pd.DataFrame(scaled_samples, columns=FEATURES)
    return kmeans.predict(scaled_samples)


# Estilos de la página -----------------------------------------------------------------------------------------------

def add_page_styles():
    st.markdown(
        """
        <style>
        :root {
            --apple-blue: #0066cc;
            --apple-blue-focus: #0071e3;
            --apple-blue-dark: #2997ff;
            --apple-ink: #1d1d1f;
            --apple-muted: #6e6e73;
            --apple-canvas: #ffffff;
            --apple-parchment: #f5f5f7;
            --apple-pearl: #fafafc;
            --apple-hairline: #e0e0e0;
            --apple-black: #000000;
        }

        .stApp {
            background: var(--apple-parchment);
            color: var(--apple-ink);
            font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        }

        .block-container {
            max-width: 1180px;
            padding-top: 3.25rem;
            padding-bottom: 4rem;
        }

        .hero-card {
            background: linear-gradient(180deg, #ffffff 0%, #fafafc 100%);
            color: var(--apple-ink);
            border-radius: 28px;
            padding: clamp(3rem, 8vw, 5.5rem) clamp(1.5rem, 5vw, 4rem);
            margin: 0 0 1.5rem 0;
            text-align: center;
            box-shadow: rgba(0, 0, 0, 0.04) 0 12px 36px 0;
        }

        .eyebrow {
            color: var(--apple-blue-focus);
            font-size: 0.78rem;
            font-weight: 600;
            letter-spacing: -0.12px;
            margin-bottom: 1rem;
        }

        .hero-title {
            color: var(--apple-ink);
            font-size: clamp(2.5rem, 6vw, 4.5rem);
            line-height: 1.07;
            letter-spacing: -0.28px;
            font-weight: 600;
            max-width: 920px;
            margin: 0 auto 1rem auto;
        }

        .hero-subtitle {
            color: var(--apple-muted);
            font-size: clamp(1.15rem, 2vw, 1.65rem);
            font-weight: 400;
            line-height: 1.25;
            letter-spacing: 0.196px;
            max-width: 820px;
            margin: 0 auto;
        }

        .section-title {
            color: var(--apple-ink);
            font-size: clamp(2rem, 4vw, 2.75rem);
            font-weight: 600;
            line-height: 1.1;
            letter-spacing: -0.28px;
            text-align: center;
            margin: 3rem 0 0.75rem 0;
        }

        .soft-note {
            color: var(--apple-muted);
            font-size: 1.06rem;
            line-height: 1.47;
            letter-spacing: -0.374px;
            max-width: 760px;
            margin: 0 auto 1.75rem auto;
            text-align: center;
            text-wrap: balance;
        }

        .section-copy {
            display: block;
            width: min(100%, 760px);
            margin-left: auto !important;
            margin-right: auto !important;
            text-align: center !important;
        }

        .section-copy,
        .section-copy * {
            text-align: center !important;
        }

        .soft-note code {
            color: var(--apple-ink);
            background: var(--apple-pearl);
            border: 1px solid var(--apple-hairline);
            border-radius: 9999px;
            padding: 0.16rem 0.45rem;
            font-size: 0.9rem;
        }

        .info-card {
            background: transparent;
            color: var(--apple-muted);
            border-top: 1px solid var(--apple-hairline);
            border-bottom: 1px solid var(--apple-hairline);
            padding: 1rem 0;
            margin: 1.5rem auto 1.5rem auto;
            max-width: 860px;
            font-size: 1rem;
            line-height: 1.47;
            letter-spacing: -0.224px;
            text-align: center;
        }

        .info-card strong {
            color: var(--apple-ink);
            font-weight: 600;
        }

        .cluster-card {
            background: var(--apple-canvas);
            border-radius: 22px;
            padding: 2rem 2.25rem;
            min-height: 240px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            box-shadow: rgba(0, 0, 0, 0.06) 0 10px 30px 0;
        }

        .cluster-label {
            color: var(--apple-muted);
            font-size: 0.88rem;
            font-weight: 600;
            letter-spacing: -0.224px;
            margin-bottom: 0.75rem;
        }

        .cluster-value {
            color: var(--apple-ink);
            font-size: clamp(1.8rem, 3.1vw, 2.45rem);
            font-weight: 600;
            line-height: 1.12;
            letter-spacing: -0.28px;
            margin-bottom: 1rem;
            text-wrap: balance;
            overflow-wrap: anywhere;
        }

        .cluster-help {
            color: var(--apple-muted);
            font-size: 0.92rem;
            line-height: 1.45;
            letter-spacing: -0.224px;
            max-width: 30rem;
        }

        div[data-testid="stMetric"] {
            background: var(--apple-canvas);
            border-radius: 18px;
            padding: 1.25rem 1.35rem;
            box-shadow: rgba(0, 0, 0, 0.04) 0 8px 24px 0;
        }

        div[data-testid="stMetric"],
        div[data-testid="stMetric"] * {
            color: var(--apple-ink) !important;
        }

        div[data-testid="stMetricLabel"],
        div[data-testid="stMetricLabel"] *,
        div[data-testid="stMetric"] p {
            color: var(--apple-muted) !important;
            font-size: 0.88rem;
            letter-spacing: -0.224px;
        }

        div[data-testid="stMetricValue"],
        div[data-testid="stMetricValue"] * {
            color: var(--apple-ink) !important;
            font-weight: 600;
            letter-spacing: -0.28px;
        }

        div[data-testid="stAlert"] {
            background: var(--apple-canvas) !important;
            color: var(--apple-ink) !important;
            border: 0 !important;
            border-radius: 18px;
            box-shadow: rgba(0, 0, 0, 0.04) 0 8px 24px 0;
        }

        div[data-testid="stAlert"] *,
        div[data-testid="stAlert"] p,
        div[data-testid="stAlert"] div {
            color: var(--apple-ink) !important;
        }

        div[data-testid="stAlert"] svg {
            fill: var(--apple-blue-focus) !important;
        }

        div[data-testid="stVerticalBlockBorderWrapper"] {
            background: var(--apple-canvas);
            border: 0 !important;
            border-radius: 18px;
            padding: 1.5rem;
            box-shadow: rgba(0, 0, 0, 0.04) 0 8px 24px 0;
        }

        .stSlider label {
            color: var(--apple-ink) !important;
            font-weight: 600;
            letter-spacing: -0.224px;
        }

        .stSlider {
            --primary-color: var(--apple-blue-focus) !important;
            --secondary-background-color: #d2d2d7 !important;
        }

        .stSlider [data-baseweb="slider"] [role="slider"] {
            background-color: var(--apple-blue-focus) !important;
            border-color: var(--apple-blue-focus) !important;
            box-shadow: none !important;
        }

        .stSlider [data-baseweb="slider"] div,
        .stSlider [data-baseweb="slider"] span {
            accent-color: var(--apple-blue-focus) !important;
        }

        .stSlider [data-baseweb="slider"] p,
        .stSlider [data-testid="stTickBar"] *,
        .stSlider [data-testid="stThumbValue"] * {
            color: var(--apple-blue-focus) !important;
        }


        .feature-order-card {
            background: var(--apple-canvas);
            color: var(--apple-ink);
            border: 1px solid var(--apple-hairline);
            border-radius: 18px;
            padding: 1rem 1.25rem;
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
            font-size: 0.98rem;
            font-weight: 600;
            letter-spacing: -0.224px;
            box-shadow: rgba(0, 0, 0, 0.04) 0 8px 24px 0;
        }

        .feature-order-card span {
            color: var(--apple-blue-focus);
        }

        .stCodeBlock {
            border-radius: 18px;
            overflow: hidden;
        }

        .js-plotly-plot {
            border-radius: 18px;
            overflow: hidden;
            box-shadow: rgba(0, 0, 0, 0.06) 0 10px 30px 0;
        }

        a, button, [role="button"] {
            accent-color: var(--apple-blue-focus);
        }

        </style>
        """,
        unsafe_allow_html=True,
    )


# Configuración de la app --------------------------------------------------------------------------------------------

st.set_page_config(
    page_title="Clusters de viviendas en California",
    layout="wide",
)

add_page_styles()

scaler, kmeans = load_artifacts()
housing = load_data()


# Rangos de los sliders ----------------------------------------------------------------------------------------------

feature_ranges = {
    feature: {
        "min": float(housing[feature].min()),
        "max": float(housing[feature].max()),
        "median": float(housing[feature].median()),
    }
    for feature in FEATURES
}


# Encabezado ---------------------------------------------------------------------------------------------------------

st.markdown(
    """
    <section class="hero-card">
        <div class="eyebrow">Machine Learning · K-Means · Streamlit</div>
        <h1 class="hero-title">Explorador de clusters de viviendas en California</h1>
        <p class="hero-subtitle">
            Ajustá la ubicación y el ingreso medio de una zona para ver a qué cluster de
            K-Means pertenece. El mapa muestra cómo se distribuyen los grupos sobre California.
        </p>
    </section>
    """,
    unsafe_allow_html=True,
)


# Métricas resumen ---------------------------------------------------------------------------------------------------

st.markdown(
    """
    <div class="info-card">
        <strong>Importante:</strong> el resultado es un cluster artificial de K-Means.
        No es una predicción del precio de una vivienda, un barrio real ni una categoría oficial.
    </div>
    """,
    unsafe_allow_html=True,
)

metric_col1, metric_col2, metric_col3 = st.columns(3)

metric_col1.metric("Registros visualizados", f"{len(housing):,}".replace(",", "."))
metric_col2.metric("Clusters del modelo", len(kmeans.cluster_centers_))
metric_col3.metric("Variables usadas", len(FEATURES))


# Entradas del usuario -----------------------------------------------------------------------------------------------

st.markdown(
    '<h2 class="section-title">Configurar una vivienda</h2>',
    unsafe_allow_html=True,
)

st.markdown(
    '<p class="soft-note section-copy"><code>Latitude</code> y <code>Longitude</code> representan la ubicación. '
    '<code>MedInc</code> representa el ingreso medio de la zona. Los sliders usan los rangos reales del dataset.</p>',
    unsafe_allow_html=True,
)

with st.container(border=True):
    col1, col2, col3 = st.columns(3)

    with col1:
        latitude = st.slider(
            "Latitud",
            min_value=feature_ranges["Latitude"]["min"],
            max_value=feature_ranges["Latitude"]["max"],
            value=feature_ranges["Latitude"]["median"],
            step=0.01,
        )

    with col2:
        longitude = st.slider(
            "Longitud",
            min_value=feature_ranges["Longitude"]["min"],
            max_value=feature_ranges["Longitude"]["max"],
            value=feature_ranges["Longitude"]["median"],
            step=0.01,
        )

    with col3:
        med_inc = st.slider(
            "Ingreso medio",
            min_value=feature_ranges["MedInc"]["min"],
            max_value=feature_ranges["MedInc"]["max"],
            value=feature_ranges["MedInc"]["median"],
            step=0.10,
        )


# Predicción del cluster ---------------------------------------------------------------------------------------------

sample = pd.DataFrame(
    [[latitude, longitude, med_inc]],
    columns=FEATURES,
)

cluster = int(predict_cluster(sample, scaler, kmeans)[0])
cluster_label = CLUSTER_LABELS[cluster]


# Resultado de la predicción -----------------------------------------------------------------------------------------

st.markdown(
    '<h2 class="section-title">Resultado</h2>',
    unsafe_allow_html=True,
)

result_col, explanation_col = st.columns([1.25, 2], gap="large")

with result_col:
    st.markdown(
        f"""
        <div class="cluster-card">
            <div class="cluster-label">Cluster predicho</div>
            <div class="cluster-value">{cluster_label}</div>
            <div class="cluster-help">Cluster técnico: {cluster}. Asignación al centroide más cercano después de escalar las variables.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with explanation_col:
    st.write("La app conserva el orden exacto de variables usado en el entrenamiento:")
    st.markdown(
        '<div class="feature-order-card">Latitude <span>→</span> Longitude <span>→</span> MedInc</div>',
        unsafe_allow_html=True,
    )
    st.write(
        "Cuando mueves los sliders, la muestra se escala con el scaler.pkl entrenado "
        "y se compara contra los centroides guardados en kmeans.pkl."
    )


# Mapa interactivo ------------------------------------------------------------------------------------------------

st.markdown(
    '<h2 class="section-title">Mapa interactivo</h2>',
    unsafe_allow_html=True,
)

st.markdown(
    '<p class="soft-note section-copy">Los puntos del dataset aparecen coloreados por cluster. '
    'El marcador negro indica la vivienda configurada con los sliders.</p>',
    unsafe_allow_html=True,
)

map_data = housing[FEATURES].copy()
map_data["Cluster"] = predict_cluster(map_data, scaler, kmeans)
map_data["Grupo"] = map_data["Cluster"].map(CLUSTER_LABELS)

fig = px.scatter_map(
    map_data,
    lat="Latitude",
    lon="Longitude",
    color="Grupo",
    color_discrete_map=CLUSTER_COLORS,
    hover_data={
        "Cluster": True,
        "Grupo": True,
        "Latitude": ":.2f",
        "Longitude": ":.2f",
        "MedInc": ":.2f",
    },
    zoom=5,
    height=680,
    title="Viviendas de California coloreadas por grupo de K-Means",
)

fig.update_traces(
    marker={
        "size": 7,
        "opacity": 0.58,
    }
)

fig.add_trace(
    go.Scattermap(
        lat=[latitude],
        lon=[longitude],
        mode="markers",
        name="Punto introducido",
        marker={
            "size": 20,
            "color": "black",
        },
        text=[f"Punto introducido — {cluster_label} — cluster técnico {cluster}"],
        hoverinfo="text",
    )
)

fig.update_layout(
    map_style="carto-positron",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font={"color": "#1d1d1f"},
    legend={
        "bgcolor": "rgba(255,255,255,0.96)",
        "bordercolor": "rgba(0,0,0,0.10)",
        "borderwidth": 1,
        "font": {"color": "#1d1d1f", "size": 12},
        "title": {"font": {"color": "#1d1d1f", "size": 13}},
    },
)

st.plotly_chart(fig, width="stretch")
