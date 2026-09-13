# Models

Бинарные артефакты не хранятся в Git. `dvc repro` или `python -m bikeflow.ml train` создаёт:

- `model.joblib` — serving-модель, которую загружает API: копия лучшей модели по скользящей
  валидации (сейчас `mlp_embedding`);
- `mlp_embedding.joblib`, `mlp_onehot.joblib` — нейросети с эмбеддингами и с one-hot кодированием;
- `hgb.joblib` — градиентный бустинг;
- `seasonal_median.joblib` — baseline.

Формат bundle:

```python
{
  "kind": "mlp_embedding",
  "pipeline": InferencePipeline(model),   # предобработка и модель вместе
  "feature_spec": {...},
  "target": "rented_bike_count",
  "metadata": {
    "model_version": ...,
    "git_commit": ...,
    "training_params": ...,
    "data_sha256": ...,
    "config_sha256": ...,
    "seed": ...,
    "python": ...,
    "numpy": ...,
    "pandas": ...,
    "scikit-learn": ...,
    "torch": ...
  },
  "metrics": {"train": {...}, "validation": {...}, "test": {...}},
  "reference": DataFrame                  # эталон для мониторинга дрейфа
}
```

`reference` — период validation с прогнозами этой модели: погода, час, фактический и
предсказанный спрос. API сравнивает с ним журнал прогнозов, а `metrics.validation.mae` служит
эталонной ошибкой для concept drift. Модели, обученные до появления мониторинга, эталона не
содержат — проверка дрейфа для них отвечает `409` и просит переобучить модель.

При загрузке проверяются структура bundle и совместимость feature contract.
