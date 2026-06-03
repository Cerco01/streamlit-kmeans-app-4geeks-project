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
    """Predict K-Means cluster using the trained centroids.

    This avoids a Windows-specific threadpoolctl issue triggered by
    KMeans.predict(), while preserving the same K-Means decision rule:
    assign each scaled sample to the nearest trained centroid.
    """
    scaled_samples = scaler.transform(samples[FEATURES])
    distances = np.linalg.norm(
        scaled_samples[:, None, :] - kmeans.cluster_centers_[None, :, :],
        axis=2,
    )
    return distances.argmin(axis=1)


st.set_page_config(page_title="App de clusters de viviendas en California", layout="wide")

st.title("Predicción de clusters de viviendas en California")
st.write(
    "Esta aplicación usa un modelo K-Means entrenado para agrupar registros "
    "de viviendas de California según su ubicación e ingreso medio."
)
st.info(
    "Importante: el resultado es un cluster artificial de K-Means. No es una "
    "predicción del precio de una vivienda, un barrio real ni una categoría oficial."
)

scaler, kmeans = load_artifacts()
housing = load_data()

st.header("Introduce las características de la vivienda")
st.write(
    "`Latitude` y `Longitude` representan la ubicación. `MedInc` representa "
    "el ingreso medio de la zona."
)

col1, col2, col3 = st.columns(3)
with col1:
    latitude = st.number_input(
        "Latitude",
        min_value=32.54,
        max_value=41.95,
        value=34.05,
        step=0.01,
    )
with col2:
    longitude = st.number_input(
        "Longitude",
        min_value=-124.35,
        max_value=-114.31,
        value=-118.25,
        step=0.01,
    )
with col3:
    med_inc = st.number_input(
        "MedInc",
        min_value=0.50,
        max_value=15.00,
        value=4.50,
        step=0.10,
    )

sample = pd.DataFrame(
    [[latitude, longitude, med_inc]],
    columns=FEATURES,
)

cluster = int(predict_cluster(sample, scaler, kmeans)[0])

st.header("Predicción")
st.metric("Cluster predicho", cluster)
st.caption(
    "La app conserva el orden de variables usado en el entrenamiento: Latitude, Longitude, MedInc."
)

st.header("Mapa de clusters de viviendas en California")
map_data = housing[FEATURES].copy()
map_data["Cluster"] = predict_cluster(map_data, scaler, kmeans).astype(str)

fig = px.scatter_map(
    map_data,
    lat="Latitude",
    lon="Longitude",
    color="Cluster",
    hover_data=["MedInc"],
    zoom=5,
    height=650,
    title="Viviendas de California coloreadas por cluster de K-Means",
)
fig.update_traces(marker={"size": 7, "opacity": 0.55})
fig.add_trace(
    go.Scattermap(
        lat=[latitude],
        lon=[longitude],
        mode="markers",
        name="Punto introducido",
        marker={"size": 18, "color": "black"},
        text=[f"Punto introducido — cluster {cluster}"],
        hoverinfo="text",
    )
)
fig.update_layout(map_style="carto-positron")
st.plotly_chart(fig, width="stretch")
