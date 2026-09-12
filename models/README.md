# Models

Бинарные артефакты не хранятся в Git. Команда
`python -m bikeflow.ml train --no-figures` создаёт:

- `hgb.joblib` — обученный HGB;
- `seasonal_median.joblib` — baseline;
- `model.joblib` — копия выбранной serving-модели (сейчас HGB).

С `--include-mlp` дополнительно создаются экспериментальные MLP-артефакты.

Формат bundle:

```python
{
  "kind": "hgb",
  "pipeline": InferencePipeline(model),
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
    "scikit-learn": ...
  },
  "metrics": {...}
}
```

При загрузке проверяются структура bundle и совместимость feature contract.
