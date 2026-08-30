# Instalación

## Requisitos y dependencias

Se necesita Python 3.11 o superior, [`uv`](https://docs.astral.sh/uv/) y, para la opción
containerizada, Docker con Compose. Desde la raíz del repositorio:

```bash
uv sync --all-extras
uv run starwars-autocalls validate-data
```

`--all-extras` instala en el entorno bloqueado por `uv.lock` las dependencias de ejecución, modelos de
boosting, EDA, explicabilidad, tests, MkDocs y Streamlit.

## Servicios con `uv run`

Ejecuta cada servicio que necesites en una terminal distinta:

```bash
# FastAPI
uv run starwars-autocalls serve --host 127.0.0.1 --port 8000

# Streamlit
uv run streamlit run app/streamlit_app.py --server.port 8501

# MLflow
uv run mlflow server --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns --host 127.0.0.1 --port 5050 \
  --allowed-hosts 127.0.0.1:5050,localhost:5050

# MkDocs
uv run mkdocs serve --dev-addr 127.0.0.1:8001
```

Streamlit usa FastAPI para la inferencia; levanta ambos servicios para probar ese flujo. Las interfaces
quedan disponibles en:

| Servicio | URL |
| --- | --- |
| FastAPI (Swagger) | `http://127.0.0.1:8000/docs` |
| Streamlit | `http://127.0.0.1:8501` |
| MLflow | `http://127.0.0.1:5050` |
| MkDocs | `http://127.0.0.1:8001` |

## Servicios con Docker Compose

Para levantar los cuatro servicios:

```bash
docker compose up --build
```

Para iniciar sólo uno, usa su nombre; Compose añadirá automáticamente sus dependencias declaradas:

```bash
docker compose up --build api
docker compose up --build streamlit
docker compose up --build mlflow
docker compose up --build docs
```

Las URL son las mismas que en la tabla anterior. Pueden cambiarse copiando `.env.example` a `.env` y
editando `API_PORT`, `STREAMLIT_PORT`, `MLFLOW_PORT` o `DOCS_PORT`.

```bash
docker compose down
```

MLflow persiste su base de datos y artefactos en el volumen `starwars-autocalls_mlflow_data`; el comando
anterior lo conserva.

## Ficheros necesarios

```text
data/raw/rfqs.csv
data/raw/daily_volatility.csv
data/raw/underlyings_reference.csv
config/final_model.json
artifacts/model.joblib
artifacts/model.joblib.sha256
artifacts/model_metadata.json
```

El repositorio ya incluye el artefacto de inferencia. Joblib sólo debe cargarse desde una fuente
confiable; el checksum detecta corrupción accidental, no hace segura una serialización ajena.

## Comprobaciones

```bash
uv run mkdocs build --strict
uv run ruff check src tests scripts
uv run pytest
```
