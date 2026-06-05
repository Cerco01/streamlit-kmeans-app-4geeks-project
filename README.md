# Clusters de viviendas en California — Streamlit + Render

## Contexto

Este proyecto toma el modelo de K-Means entrenado en el proyecto anterior (California Housing) y lo expone como una aplicación web interactiva. El modelo original está en [20-k-means-4geeks-project](https://github.com/Cerco01/20-k-means-4geeks-project).

La app permite configurar una vivienda con tres variables y ver a qué cluster del modelo pertenece, además de mostrar todos los registros del dataset sobre un mapa.

La app está desplegada en Render:

```text
https://streamlit-kmeans-app-4geeks-project.onrender.com/
```

![Vista de la app](docs/img/app-preview.png)

## Dataset

El dataset es California Housing, ya usado en el proyecto anterior. La app no lo entrena de nuevo, solo lo lee para alimentar el mapa interactivo y los sliders.

Ubicación local:

```text
data/raw/housing.csv
```

El archivo se mantiene fuera del repositorio si pesa demasiado; en este caso ya estaba versionado del proyecto anterior.

## Variables que usa el modelo

El modelo consume tres variables en este orden:

- `Latitude`
- `Longitude`
- `MedInc`

Ese orden está fijado en `src/app.py` como `FEATURES`. El modelo lo respeta, así que cualquier cambio en el `DataFrame` que entra a `predict()` debe mantenerlo.

## Qué incluye la app

- Carga del scaler y del modelo entrenado desde `models/`.
- Carga del dataset para alimentar el mapa y los rangos de los sliders.
- Sliders de Latitud, Longitud e Ingreso medio, con rangos tomados del dataset.
- Predicción del cluster con `scaler.transform()` seguido de `kmeans.predict()`.
- Tarjeta de resultado con etiqueta interpretativa del cluster.
- Mapa interactivo de Plotly con los registros coloreados por grupo y el punto introducido marcado encima.
- Aviso visible en la app: el resultado no predice precios ni barrios oficiales.

## Limitación importante

Este proyecto **no predice precios de viviendas**. El resultado es un cluster artificial generado por K-Means. No representa un barrio real, una categoría oficial ni una estimación de precio. El aviso se muestra dentro de la app para que quede explícito al usuario.

## Cómo usar este proyecto

1. Clonar el repositorio.
2. Crear o activar un entorno de Python compatible.
3. Instalar las dependencias.

```bash
pip install -r requirements.txt
```

4. Ejecutar la app localmente.

```bash
streamlit run src/app.py
```

5. Para el deploy en Render, la configuración es:

```text
Root directory: raíz del repositorio
Build command: pip install -r requirements.txt
Start command: streamlit run src/app.py
Python:      3.11.9 (fijado desde el panel de Render)
```

## Archivos principales

- `src/app.py`: aplicación de Streamlit.
- `src/explore.ipynb`: notebook original del modelo de K-Means.
- `models/scaler.pkl`: escalador entrenado.
- `models/kmeans.pkl`: modelo K-Means entrenado.
- `models/supervised.pkl`: modelo supervisado adicional del proyecto anterior.
- `data/raw/housing.csv`: dataset usado para el mapa y los sliders.
- `docs/img/app-preview.png`: captura de la app para este README.
- `.streamlit/config.toml`: tema de Streamlit (azul Apple, fondo claro).
- `requirements.txt`: dependencias de ejecución.

## Despliegue

La app está desplegada en Render. Fecha del último despliegue verificado: 2026-06-05.

El servicio corre en el plan gratuito de Render, que pone la app en reposo tras periodos de inactividad. La primera carga después de un reposo puede tardar alrededor de un minuto en despertar.

El despliegue se realizó como parte de la entrega de este proyecto. Es posible que el servicio deje de estar disponible en el futuro cuando el autor desactive la cuenta gratuita de Render o el servicio quede fuera de servicio. En ese caso, la app puede ejecutarse localmente siguiendo los pasos de la sección anterior.

## Recursos externos

- Streamlit: https://docs.streamlit.io/
- Plotly Express (`px.scatter_map`): https://plotly.com/python/scattermapbox/
- Scikit-learn (KMeans, StandardScaler): https://scikit-learn.org/
- `getdesign` para la referencia visual Apple: https://getdesign.md/

## Mejora futura

- Reentrenar el modelo con más variables del dataset y comparar la coherencia de los clusters.
- Sustituir `kmeans.predict()` por una función manual de distancias para tener un fallback documentado en el código si Render se queja de BLAS/MKL.
- Sustituir el plan gratuito de Render por uno de pago para evitar el delay de despertar tras inactividad.
- Añadir tests unitarios sobre `predict_cluster()` y sobre el orden de `FEATURES`.

## Limitaciones de hardware y próximos pasos

Tres cosas que quedaron pendientes:

- El `kmeans.predict()` local en Windows puede fallar con un error raro de `threadpoolctl`/BLAS/MKL. En Render con `python-3.11.9` no se reproduce, pero tenerlo documentado evita búsquedas futuras.
- El tema Apple se aplica vía `.streamlit/config.toml` (`primaryColor`). El CSS propio en `src/app.py` refuerza sliders, métricas y cards. Si en el futuro se actualiza Streamlit, los selectores CSS pueden romperse.
- El deploy se hizo con plan gratuito. Cualquier freeze o sleep del servicio impacta al reviewer.

## Créditos

Este proyecto fue realizado como parte del [Bootcamp de Data Science y Machine Learning de 4Geeks](https://4geeksacademy.com/en/career-programs/data-science-ml).

El enunciado original pertenece a [4Geeks Academy](https://github.com/4GeeksAcademy).
