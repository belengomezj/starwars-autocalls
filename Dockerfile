FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_NO_CACHE=1
WORKDIR /app

RUN pip install --no-cache-dir uv==0.12.5 \
    && groupadd --system app \
    && useradd --system --gid app --home-dir /app app \
    && chown app:app /app

COPY pyproject.toml uv.lock ./


FROM base AS runtime-dependencies
RUN uv sync --frozen --no-dev --no-install-project


FROM runtime-dependencies AS project-runtime
COPY README.md ./README.md
COPY config ./config
COPY src ./src
RUN uv sync --frozen --no-dev


FROM project-runtime AS api-dependencies
RUN uv sync --frozen --no-dev --extra boosting


FROM api-dependencies AS api
COPY --chown=app:app artifacts ./artifacts
COPY --chown=app:app data/raw ./data/raw

USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD [".venv/bin/python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/ready', timeout=3)"]
CMD [".venv/bin/starwars-autocalls", "serve", "--host", "0.0.0.0", "--port", "8000"]


FROM api-dependencies AS demo
RUN uv sync --frozen --no-dev --extra boosting --extra demo
COPY --chown=app:app app ./app
COPY --chown=app:app artifacts ./artifacts
COPY --chown=app:app data/raw ./data/raw
COPY --chown=app:app docs/assets/images ./docs/assets/images

USER app
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD [".venv/bin/python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=3)"]
CMD [".venv/bin/streamlit", "run", "app/streamlit_app.py", "--server.address", "0.0.0.0", "--server.port", "8501"]


FROM runtime-dependencies AS mlflow
RUN mkdir -p /mlflow/artifacts \
    && chown -R app:app /mlflow

USER app
EXPOSE 5000
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=5 \
  CMD [".venv/bin/python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:5000/health', timeout=3)"]
CMD [".venv/bin/mlflow", "server", "--host", "0.0.0.0", "--port", "5000"]


FROM runtime-dependencies AS docs
RUN uv sync --frozen --no-dev --extra dev --no-install-project
COPY --chown=app:app mkdocs.yml ./mkdocs.yml
COPY --chown=app:app docs ./docs
COPY --chown=app:app scripts/serve_docs.py ./scripts/serve_docs.py

USER app
EXPOSE 8000
CMD [".venv/bin/python", "scripts/serve_docs.py"]


# Preserve `docker build .` as the production API image.
FROM api AS final
