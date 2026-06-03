from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT_DIR / "models"
DATA_PATH = ROOT_DIR / "data" / "raw" / "housing.csv"
FEATURES = ["Latitude", "Longitude", "MedInc"]


@st.cache_resource
def load_artifacts():
    scaler = joblib.load(MODEL_DIR / "scaler.pkl")
    kmeans = joblib.load(MODEL_DIR / "kmeans.pkl")
    return scaler, kmeans


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


def predict_cluster(samples: pd.DataFrame, scaler, kmeans) -> np.ndarray:
    """Predict K-Means cluster using the trained centroids."""
    scaled_samples = scaler.transform(samples[FEATURES])
    distances = np.linalg.norm(
        scaled_samples[:, None, :] - kmeans.cluster_centers_[None, :, :],
        axis=2,
    )
    return distances.argmin(axis=1)


def add_page_styles():
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(94, 106, 210, 0.22), transparent 34rem),
                radial-gradient(circle at top right, rgba(16, 185, 129, 0.12), transparent 28rem),
                #08090a;
        }
        .block-container {
            padding-top: 3rem;
            max-width: 1180px;
        }
        .hero-card {
            border: 1px solid rgba(255,255,255,0.10);
            border-radius: 28px;
            padding: 2.25rem;
            margin-bottom: 1.5rem;
            background: linear-gradient(135deg, rgba(255,255,255,0.075), rgba(255,255,255,0.025));
            box-shadow: 0 24px 80px rgba(0,0,0,0.35);
        }
        .eyebrow {
            color: #9aa4ff;
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            margin-bottom: 0.75rem;
        }
        .hero-title {
            color: #f7f8f8;
            font-size: clamp(2.35rem, 5vw, 4.4rem);
            line-height: 0.98;
            letter-spacing: -0.06em;
            font-weight: 800;
            margin: 0 0 1rem 0;
        }
        .hero-subtitle {
            color: #d0d6e0;
            font-size: 1.08rem;
            max-width: 780px;
            line-height: 1.7;
            margin: 0;
        }
        .pill-row {
            display: flex;
            gap: 0.75rem;
            flex-wrap: wrap;
            margin-top: 1.5rem;
        }
        .pill {
            border: 1px solid rgba(255,255,255,0.12);
            background: rgba(255,255,255,0.055);
            border-radius: 999px;
            padding: 0.45rem 0.8rem;
            color: #d0d6e0;
            font-size: 0.88rem;
        }
        .section-title {
            color: #f7f8f8;
            letter-spacing: -0.04em;
            font-size: 1.85rem;
            font-weight: 750;
            margin: 1.8rem 0 0.75rem 0;
        }
        .soft-note {
            color: #9aa0aa;
            line-height: 1.6;
            margin-bottom: 1rem;
        }
        .cluster-card {
            border: 1px solid rgba(113,112,255,0.35);
            border-radius: 22px;
            padding: 1.4rem;
            background: linear-gradient(135deg, rgba(113,112,255,0.20), rgba(255,255,255,0.035));
        }
        .cluster-label {
            color: #b9bbff;
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.10em;
            text-transform: uppercase;
        }
        .cluster-value {
            color: #ffffff;
            font-size: 4rem;
            line-height: 1;
            font-weight: 850;
            margin-top: 0.5rem;
        }
        .cluster-help {
            color: #c9cbe8;
            margin-top: 0.75rem;
            line-height: 1.5;
        }
        div[data-testid="stMetric"] {
            border: 1px solid rgba(255,255,255,0.10);
            border-radius: 18px;
            padding: 1rem;
            background: rgba(255,255,255,0.045);
        }
        div[data-testid="stSlider"] {
            padding: 0.7rem 0.2rem 0.9rem 0.2rem;
        }
        code {
            color: #89f7b5 !important;
            background: rgba(16,185,129,0.10) !important;
            border-radius: 6px !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


st.set_page_config(page_title="Clusters de viviendas en California", layout="wide")
add_page_styles()

scaler, kmeans = load_artifacts()
housing = load_data()
feature_ranges = {
    feature: {
        "min": float(housing[feature].min()),
        "max": float(housing[feature].max()),
        "median": float(housing[feature].median()),
    }
    for feature in FEATURES
}

st.markdown(
    """
    <section class="hero-card">
        <div class="eyebrow">Machine Learning · K-Means · Streamlit</div>
        <h1 class="hero-title">Explorador de clusters de viviendas en California</h1>
        <p class="hero-subtitle">
            Ajustá la ubicación y el ingreso medio de una zona para ver a qué cluster de
            K-Means pertenece. El mapa muestra cómo se distribuyen los grupos sobre California.
        </p>
        <div class="pill-row">
            <span class="pill">Modelo no supervisado</span>
            <span class="pill">Mapa interactivo</span>
            <span class="pill">Predicción en tiempo real</span>
        </div>
    </section>
    """,
    unsafe_allow_html=True,
)

st.info(
    "Importante: el resultado es un cluster artificial de K-Means. No es una "
    "predicción del precio de una vivienda, un barrio real ni una categoría oficial."
)

metric_col1, metric_col2, metric_col3 = st.columns(3)
metric_col1.metric("Registros visualizados", f"{len(housing):,}".replace(",", "."))
metric_col2.metric("Clusters del modelo", len(kmeans.cluster_centers_))
metric_col3.metric("Variables usadas", len(FEATURES))

st.markdown('<h2 class="section-title">Configurar una vivienda</h2>', unsafe_allow_html=True)
st.markdown(
    '<p class="soft-note"><code>Latitude</code> y <code>Longitude</code> representan la ubicación. '
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

sample = pd.DataFrame([[latitude, longitude, med_inc]], columns=FEATURES)
cluster = int(predict_cluster(sample, scaler, kmeans)[0])

st.markdown('<h2 class="section-title">Resultado</h2>', unsafe_allow_html=True)
result_col, explanation_col = st.columns([1, 2])
with result_col:
    st.markdown(
        f"""
        <div class="cluster-card">
            <div class="cluster-label">Cluster predicho</div>
            <div class="cluster-value">{cluster}</div>
            <div class="cluster-help">Asignación al centroide más cercano después de escalar las variables.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with explanation_col:
    st.write("La app conserva el orden exacto de variables usado en el entrenamiento:")
    st.code("Latitude → Longitude → MedInc", language="text")
    st.write(
        "Cuando movés los sliders, la muestra se escala con el `scaler.pkl` entrenado "
        "y se compara contra los centroides guardados en `kmeans.pkl`."
    )

st.markdown('<h2 class="section-title">Mapa interactivo</h2>', unsafe_allow_html=True)
st.markdown(
    '<p class="soft-note">Los puntos del dataset aparecen coloreados por cluster. '
    'El marcador negro indica la vivienda configurada con los sliders.</p>',
    unsafe_allow_html=True,
)

map_data = housing[FEATURES].copy()
map_data["Cluster"] = predict_cluster(map_data, scaler, kmeans).astype(str)

fig = px.scatter_map(
    map_data,
    lat="Latitude",
    lon="Longitude",
    color="Cluster",
    hover_data={"Latitude": ":.2f", "Longitude": ":.2f", "MedInc": ":.2f"},
    zoom=5,
    height=680,
    title="Viviendas de California coloreadas por cluster de K-Means",
)
fig.update_traces(marker={"size": 7, "opacity": 0.58})
fig.add_trace(
    go.Scattermap(
        lat=[latitude],
        lon=[longitude],
        mode="markers",
        name="Punto introducido",
        marker={"size": 20, "color": "black"},
        text=[f"Punto introducido — cluster {cluster}"],
        hoverinfo="text",
    )
)
fig.update_layout(
    map_style="carto-positron",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font={"color": "#f7f8f8"},
    legend={"bgcolor": "rgba(8,9,10,0.72)", "bordercolor": "rgba(255,255,255,0.10)", "borderwidth": 1},
)
st.plotly_chart(fig, width="stretch")
