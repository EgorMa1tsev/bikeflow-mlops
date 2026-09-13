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

# The API runs as an unprivileged user and cannot write under /app, so the
# prediction journal gets its own directory owned by that user.
RUN addgroup --system bikeflow && adduser --system --ingroup bikeflow bikeflow && \
    mkdir -p /var/lib/bikeflow && chown bikeflow:bikeflow /var/lib/bikeflow

ENV BIKEFLOW_DB_PATH=/var/lib/bikeflow/predictions.db \
    BIKEFLOW_MONITORING_DIR=/var/lib/bikeflow/monitoring

USER bikeflow
EXPOSE 8000

CMD ["uvicorn", "bikeflow.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
