# Контракт model/API

`POST /predict` принимает один объект. `prediction_time` обязан содержать UTC
offset/timezone и перед построением календарных признаков переводится в
`Asia/Seoul`. Погодные признаки передаются непосредственно клиентом.

| Поле API | Тип | Единица | Диапазон |
| --- | --- | --- | --- |
| `temperature_c` | number | °C | −40…50 |
| `humidity_pct` | number | % | 0…100 |
| `wind_speed_m_s` | number | m/s | 0…50 |
| `visibility_10m` | number | 10 m | 0…2000 |
| `dew_point_c` | number | °C | −40…40 |
| `solar_radiation_mj_m2` | number | MJ/m² | 0…10 |
| `rainfall_mm` | number | mm | 0…200 |
| `snowfall_cm` | number | cm | 0…100 |
| `holiday` | boolean | — | по умолчанию `false` |
| `functioning_day` | boolean | — | по умолчанию `true` |

Фактический единственный источник этой таблицы — `FEATURE_CONTRACT` в
`src/bikeflow/ml/features.py`. Pydantic-поля API и canonical mapping строятся из
него. `hour`, `day_of_week` и `season` выводятся из нормализованного времени.

Неверные данные отклоняются с HTTP `422`. Неизвестные поля запрещены. Ответ:

```json
{
  "prediction_id": 17,
  "prediction_time": "2026-07-15T08:00:00+09:00",
  "predicted_rentals": 512.3,
  "model_version": "mlp_embedding-373339b7-..."
}
```

## Журнал прогнозов

Каждый успешный прогноз записывается в SQLite-журнал (`BIKEFLOW_DB_PATH`): входные
признаки вместе с выведенными `hour`, `day_of_week` и `season`, прогноз, версия модели
и время запроса. Отклонённые запросы не записываются.

Фактический спрос становится известен только после окончания часа, поэтому он
досылается отдельно по `prediction_id`:

| Запрос | Тело | Ответ |
| --- | --- | --- |
| `POST /predictions/{prediction_id}/actual` | `{"actual_rentals": 498}`, число `>= 0` | запись журнала; `404`, если id неизвестен |
| `GET /predictions?limit=100` | — | последние записи, новые первыми; `limit` от 1 до 1000 |

Запись журнала:

```json
{
  "prediction_id": 17,
  "created_at": "2026-09-13T10:00:00+00:00",
  "prediction_time": "2026-07-15T08:00:00+09:00",
  "features": {"temperature_c": 24.5, "hour": 8, "season": "Summer", "...": "..."},
  "predicted_rentals": 512.3,
  "model_version": "mlp_embedding-373339b7-...",
  "actual_rentals": 498.0,
  "absolute_error": 14.3
}
```

Пока факт не прислан, `actual_rentals` и `absolute_error` равны `null`.

FastAPI получает настоящий `BikeflowPredictor`, который лениво загружает путь из
`BIKEFLOW_MODEL_PATH`. Ленивая загрузка позволяет вернуть `422` за неверное тело
до обращения к диску. После первой загрузки тот же MLP predictor используется для
всех запросов без переобучения. Stub разрешён только как injected test double.

## Мониторинг дрейфа

Проверка сравнивает последние `monitoring.window_hours` журнала — записи текущей версии модели с
известным фактом, только рабочие часы — с эталоном из артефакта модели.

| Запрос | Ответ |
| --- | --- |
| `POST /drift/check` | результат проверки; `409`, если у модели нет эталона; `422`, если записей меньше `monitoring.min_rows` |
| `GET /drift/latest` | последний результат; `404`, если проверок не было |
| `GET /drift/report` | HTML-отчёт Evidently; `404`, если отчёта ещё нет |

Результат проверки:

```json
{
  "check_id": 12,
  "model_version": "mlp_embedding-373339b7-...",
  "window_start": "2018-11-08T00:00:00+09:00",
  "window_end": "2018-11-14T23:00:00+09:00",
  "rows": 168,
  "data_drift": true,
  "drifted_feature_share": 0.875,
  "drifted_features": ["temperature", "humidity", "..."],
  "target_drift": true,
  "target_drift_score": 0.48,
  "concept_drift": true,
  "current_mae": 385.0,
  "reference_mae": 183.1,
  "mae_ratio": 2.1,
  "thresholds": {"drift_threshold": 0.1, "data_drift_share": 0.5, "concept_drift_mae_ratio": 1.6}
}
```

`concept_drift` — сигнал к переобучению. `data_drift` и `target_drift` описывают изменение входов и
спроса и сами по себе переобучения не требуют.

Production bundle содержит preprocessing и PyTorch MLP embedding. Погода сейчас передаётся
пользователем. Обучение и переобучение не входят в inference-контракт.
