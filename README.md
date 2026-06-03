# California Housing Cluster App

Streamlit web app for predicting California Housing K-Means clusters from three features:

- `Latitude`
- `Longitude`
- `MedInc`

The app loads the trained artifacts from `models/` and visualizes the California housing records on an interactive Plotly map.

## Important limitation

This project does not predict house prices.

The output is an artificial K-Means cluster. It is not a real neighborhood, an official category, or a price estimate.

## Local usage

Install dependencies from the repository root:

```bash
pip install -r requirements.txt
```

Run the Streamlit app from the repository root:

```bash
streamlit run src/app.py
```

## Render deployment

Target repository:

```text
https://github.com/Cerco01/streamlit-kmeans-app-4geeks-project
```

Render settings:

```text
Root directory: repository root
Build command: pip install -r requirements.txt
Start command: streamlit run src/app.py
```

After deploying, open the public Render URL once before submitting to verify that the app wakes up and loads correctly. Render Free services may take around one minute to wake up after inactivity.

## Project structure

```text
src/app.py              Streamlit app
src/explore.ipynb       Original K-Means notebook/model provenance
data/raw/housing.csv    California Housing data used for the map
models/scaler.pkl       Trained scaler
models/kmeans.pkl       Trained K-Means model
models/supervised.pkl   Extra supervised model kept from the previous project
requirements.txt        Runtime dependencies
```

## Notes for submission

The `theory/` directory contains local class material and is not part of the final 4Geeks delivery.
