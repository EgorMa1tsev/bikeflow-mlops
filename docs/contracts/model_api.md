# Контракт model/API

Статус: **provisional**, подготовлен для обсуждения участников A и B.

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
  "prediction_time": "2026-07-15T08:00:00+09:00",
  "predicted_rentals": 512.3,
  "model_version": "mlp_embedding-373339b7-..."
}
```

FastAPI получает настоящий `BikeflowPredictor`, который лениво загружает путь из
`BIKEFLOW_MODEL_PATH`. Ленивая загрузка позволяет вернуть `422` за неверное тело
до обращения к диску. После первой загрузки тот же MLP predictor используется для
всех запросов без переобучения. Stub разрешён только как injected test double.

Production bundle содержит preprocessing и PyTorch MLP embedding. Погода сейчас
передаётся пользователем. Обучение, внешний weather API, drift detection и
retraining не входят в inference-контракт.
