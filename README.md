# Aplicación de clusters de viviendas en California

Aplicación web desarrollada con Streamlit para predecir clusters de K-Means del dataset California Housing a partir de tres variables:

- `Latitude`
- `Longitude`
- `MedInc`

La aplicación carga los artefactos entrenados desde `models/` y visualiza los registros de viviendas de California en un mapa interactivo de Plotly.

## Limitación importante

Este proyecto no predice precios de viviendas.

El resultado es un cluster artificial generado por K-Means. No representa un barrio real, una categoría oficial ni una estimación de precio.

## Uso local

Instalar las dependencias desde la raíz del repositorio:

```bash
pip install -r requirements.txt
```

Ejecutar la aplicación de Streamlit desde la raíz del repositorio:

```bash
streamlit run src/app.py
```

## Despliegue en Render

URL pública de la aplicación:

```text
https://streamlit-kmeans-app-4geeks-project.onrender.com/
```

Repositorio objetivo:

```text
https://github.com/Cerco01/streamlit-kmeans-app-4geeks-project
```

Configuración usada en Render:

```text
Root directory: raíz del repositorio
Build command: pip install -r requirements.txt
Start command: streamlit run src/app.py
```

Después del despliegue, se abrió la URL pública de Render para verificar que la aplicación carga correctamente. En el plan gratuito de Render, el servicio puede tardar alrededor de un minuto en despertarse después de un periodo de inactividad.

## Estructura del proyecto

```text
src/app.py              Aplicación de Streamlit
src/explore.ipynb       Notebook original de K-Means y procedencia del modelo
data/raw/housing.csv    Datos de California Housing usados para el mapa
models/scaler.pkl       Escalador entrenado
models/kmeans.pkl       Modelo K-Means entrenado
models/supervised.pkl   Modelo supervisado adicional conservado del proyecto anterior
requirements.txt        Dependencias de ejecución
runtime.txt             Versión de Python usada por Render
```

## Notas para la entrega

El directorio `theory/` contiene material local de clase y no forma parte de la entrega final de 4Geeks.
