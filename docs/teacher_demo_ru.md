# Демонстрация первого вертикального среза BikeFlow

BikeFlow прогнозирует суммарный спрос на велосипеды в Сеуле на конкретный час и
заданную погоду. Репозиторий публичный: <https://github.com/daya-alexandra/bikeflow-mlops>.
Погода пока вводится пользователем; внешнего weather API нет.

## Что показать

1. В PR №7 показать сохранённый коммит участника A с rolling-origin CV и
   отдельные интеграционные коммиты API/Docker/CI.
2. В `reports/model_selection.md` показать выбор MLP embedding: средний CV MAE
   `388.982` против `416.625` у HGB. MAE — основная метрика, WAPE — дополнительная.
3. Объяснить, что HGB выигрывает только Aug–Sep fold, поэтому остаётся
   сравнительной, а не production-моделью. Test не участвовал в выборе.
4. В `InferencePipeline` показать preprocessing и MLP в одном `.joblib` bundle.
5. В API показать обязательную timezone, нормализацию в `Asia/Seoul`, ручные
   погодные поля и путь `BIKEFLOW_MODEL_PATH`.
6. Выполнить `pytest` и `docker compose up --build`, затем вызвать `/predict`.
7. На вкладке Checks показать зелёные `lint`, `tests`, `ml-tests`,
   `docker-build`, `docker-runtime-smoke`.

## Команды

```bash
python -m pip install --constraint requirements/runtime-py311.lock \
  torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install --constraint requirements/runtime-py311.lock -e ".[dev,ml,mlp]"
python -m bikeflow.ml download
python -m bikeflow.ml preprocess
python -m bikeflow.ml split
python -m bikeflow.ml cv
python -m bikeflow.ml train --no-figures
pytest
docker compose up --build
```

Serving-образ использует Python 3.11 slim и CPU-only PyTorch. Training/reporting
зависимости в финальный stage не попадают. Compose монтирует
`./models/model.joblib` в `/models/model.joblib:ro`.

## Контрольный запрос

```json
{
  "prediction_time": "2026-07-15T08:00:00+09:00",
  "temperature_c": 24.5,
  "humidity_pct": 61,
  "wind_speed_m_s": 1.8,
  "visibility_10m": 1800,
  "dew_point_c": 16.4,
  "solar_radiation_mj_m2": 1.2,
  "rainfall_mm": 0,
  "snowfall_cm": 0,
  "holiday": false,
  "functioning_day": true
}
```

Ответ содержит неотрицательный `predicted_rentals`, нормализованный timestamp и
`model_version`, начинающийся с `mlp_embedding-`. Неверная timezone или диапазон
поля дают `422`. Predictor лениво загружается один раз; запросы не запускают fit.

## Честные оговорки

Seed и версии закреплены, но bit-for-bit совпадение независимых обучений не
обещается. Датасет и production-артефакт не хранятся в Git. Нет автоматической
погоды, мониторинга, drift detection, retraining, DVC/MLflow и UI; данные — один
город и один год, test только осенний.
