.PHONY: install install-ml lint format test run docker-build data cv train evaluate

install:
	python -m pip install -e ".[dev]"
	pre-commit install

install-ml:
	python -m pip install --constraint requirements/runtime-py311.lock torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
	python -m pip install --constraint requirements/runtime-py311.lock -e ".[dev,ml,mlp]"

lint:
	ruff check .
	ruff format --check .

format:
	ruff check --fix .
	ruff format .

test:
	pytest

run:
	uvicorn bikeflow.api.main:app --reload --host 0.0.0.0 --port 8000

docker-build:
	docker build --tag bikeflow:local .

# --- data / model pipeline (participant A) ----------------------------------
# Requires the `ml` extra: make install-ml

data:
	python -m bikeflow.ml download
	python -m bikeflow.ml preprocess
	python -m bikeflow.ml split

cv:
	python -m bikeflow.ml cv

train:
	python -m bikeflow.ml train

evaluate:
	python -m bikeflow.ml evaluate --split test
