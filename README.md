# BikeFlow

BikeFlow — публичный учебный MLOps-проект для прогноза почасового спроса на
велопрокат в Сеуле. Первый вертикальный срез проходит весь путь: официальный
датасет UCI → хронологическая проверка → rolling-origin model selection → единый
MLP-артефакт preprocessing + model → FastAPI `/predict` → Docker.

Участник A ([EgorMa1tsev](https://github.com/EgorMa1tsev)) отвечает за данные,
признаки, обучение и оценку. Участница B
([daya-alexandra](https://github.com/daya-alexandra)) — за репозиторий, API,
интеграцию, тесты, CI и Docker.

## Production-модель и путь запроса

Production-модель — PyTorch **MLP embedding**. HGB и seasonal median остаются
сравниваемыми моделями. Выбор сделан по среднему MAE на трёх rolling-origin folds:

| Модель | fold 1 | fold 2 | fold 3 | средний MAE |
| --- | ---: | ---: | ---: | ---: |
| **MLP embedding** | 409.0 | 376.4 | 381.6 | **389.0** |
| HGB | 582.0 | 456.3 | 211.5 | 416.6 |
| MLP one-hot | 475.2 | 357.7 | 431.1 | 421.3 |
| seasonal median | 689.9 | 867.3 | 644.1 | 733.8 |

MAE — основная метрика; WAPE публикуется как дополнительная. Test не участвует
в выборе и оценивается только после фиксации победителя. В закреплённой
интеграционной среде MLP embedding получила validation MAE/WAPE
`184.303 / 0.190213`, test MAE/WAPE `191.095 / 0.224749`.

`JSON` → валидация → перевод времени в `Asia/Seoul` → календарные признаки →
`InferencePipeline` из `BIKEFLOW_MODEL_PATH` → MLP embedding → прогноз и версия.
Погоду пока передаёт пользователь; внешнего weather API нет.

## Установка и обучение

Требуется Python 3.11. Прямые версии закреплены в `pyproject.toml`, runtime lock —
в `requirements/runtime-py311.lock`. Для Linux CPU wheel PyTorch ставится из
официального CPU-only index:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\Activate.ps1
python -m pip install --constraint requirements/runtime-py311.lock \
  torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install --constraint requirements/runtime-py311.lock -e ".[dev,ml,mlp]"
python -m bikeflow.ml download
python -m bikeflow.ml preprocess
python -m bikeflow.ml split
python -m bikeflow.ml train --no-figures
```

`python -m bikeflow.ml cv` отдельно повторяет rolling-origin сравнение. Загрузчик
проверяет SHA256 исходного CSV. Данные и `.joblib`-артефакты Git игнорирует.

Seed и версии зависимостей фиксируются и пишутся в metadata. Это повышает
воспроизводимость, но проект не обещает bit-for-bit совпадение двух независимых
обучений на разных системах.

## API

Путь по умолчанию — `models/model.joblib`; его можно изменить:

```bash
BIKEFLOW_MODEL_PATH=models/model.joblib uvicorn bikeflow.api.main:app \
  --host 127.0.0.1 --port 8000
```

Модель загружается лениво после успешной валидации первого запроса и затем
переиспользуется; API не обучает модель при запросах.

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "prediction_time":"2026-07-15T08:00:00+09:00",
    "temperature_c":24.5,
    "humidity_pct":61,
    "wind_speed_m_s":1.8,
    "visibility_10m":1800,
    "dew_point_c":16.4,
    "solar_radiation_mj_m2":1.2,
    "rainfall_mm":0,
    "snowfall_cm":0,
    "holiday":false,
    "functioning_day":true
  }'
```

`prediction_time` обязан содержать timezone; время нормализуется в
`Asia/Seoul`. Неверное тело и значения вне общего API/ML-контракта получают
`422`. Swagger UI: <http://127.0.0.1:8000/docs>.

## Docker и проверки

Serving-образ — `python:3.11-slim` с CPU-only PyTorch и без training/reporting
зависимостей. Compose монтирует локальный артефакт read-only:

```bash
docker compose up --build
```

```bash
ruff check .
ruff format --check .
pytest
docker build --tag bikeflow:local .
```

CI выполняет `lint`, `tests`, `ml-tests`, `docker-build` и
`docker-runtime-smoke`. Smoke-тест обучает небольшую настоящую MLP embedding,
монтирует bundle и вызывает `/predict` по HTTP. Stub используется только как
injected test double.

## Ограничения

Нет автоматического получения погоды, DVC, MLflow, оркестратора, drift
monitoring, автоматического переобучения и UI. Данные охватывают один город и
один год; test целиком осенний, пики и дождь остаются сложными режимами.
Подробности: [`docs/model/model_card.md`](docs/model/model_card.md),
[`docs/contracts/model_api.md`](docs/contracts/model_api.md) и
[`docs/teacher_demo_ru.md`](docs/teacher_demo_ru.md).
