FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    BIKEFLOW_ROOT=/app \
    BIKEFLOW_MODEL_PATH=/models/model.joblib

WORKDIR /app

COPY requirements/runtime-py311.lock ./requirements/runtime-py311.lock
RUN python -m pip install --upgrade pip && \
    python -m pip install --constraint requirements/runtime-py311.lock \
        torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu && \
    python -m pip install --constraint requirements/runtime-py311.lock \
        evidently==0.7.23 fastapi==0.115.6 joblib==1.4.2 numpy==2.2.1 pandas==2.2.3 \
        prometheus-client==0.26.0 pydantic-settings==2.7.1 PyYAML==6.0.2 \
        scikit-learn==1.6.1 uvicorn==0.34.0

COPY pyproject.toml README.md params.yaml ./
COPY src ./src
RUN python -m pip install --no-deps . && python -m pip check

FROM base AS training

RUN python -m pip install --constraint requirements/runtime-py311.lock \
        matplotlib==3.10.0 mlflow==3.16.0 pyarrow==18.1.0 requests==2.32.3 tabulate==0.9.0

CMD ["python", "-m", "bikeflow.ml", "train", "--no-figures"]

FROM base AS runtime

# Retraining runs inside the API: it needs the MLflow client to read and register
# models, and requests to fetch the dataset the challenger learns from.
RUN python -m pip install --constraint requirements/runtime-py311.lock \
        mlflow==3.16.0 requests==2.32.3

# The API runs as an unprivileged user and cannot write under /app, so the
# prediction journal and the dataset retraining downloads get their own
# directories owned by that user.
RUN addgroup --system bikeflow && adduser --system --ingroup bikeflow bikeflow && \
    mkdir -p /var/lib/bikeflow /app/data && \
    chown bikeflow:bikeflow /var/lib/bikeflow /app/data

ENV BIKEFLOW_DB_PATH=/var/lib/bikeflow/predictions.db \
    BIKEFLOW_MONITORING_DIR=/var/lib/bikeflow/monitoring \
    BIKEFLOW_RETRAINING_DIR=/var/lib/bikeflow/retraining

USER bikeflow
EXPOSE 8000

CMD ["uvicorn", "bikeflow.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

FROM base AS ui

# The interface is a plain HTTP client of the API, so it needs neither the model
# nor the journal. mlflow-skinny is the MLflow client without its server, enough
# for the experiments tab to list runs from a remote tracking server.
RUN python -m pip install --constraint requirements/runtime-py311.lock \
        mlflow-skinny==3.16.0 requests==2.32.3 streamlit==1.55.0

ENV BIKEFLOW_API_URL=http://api:8000

EXPOSE 8501

CMD ["streamlit", "run", "src/bikeflow/ui/app.py", \
     "--server.address", "0.0.0.0", "--server.port", "8501", "--server.headless", "true"]
