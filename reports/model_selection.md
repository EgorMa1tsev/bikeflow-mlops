# Выбор основной модели

Сгенерировано: 2026-09-08T11:31:57+00:00

Production-модель — **MLP embedding (`mlp_embedding`)**. Она выбрана по
минимальному среднему **MAE** на трёх rolling-origin folds; WAPE —
дополнительная метрика. Test не участвовал в выборе и был рассчитан только
после фиксации победителя, только для MLP embedding.

## Rolling-origin cross-validation

| model           | metric   |   mean_score |   worst_score |   best_score |
|:----------------|:---------|-------------:|--------------:|-------------:|
| mlp_embedding   | mae      |      388.982 |       408.967 |      376.356 |
| hgb             | mae      |      416.625 |       582.045 |      211.527 |
| mlp_onehot      | mae      |      421.341 |       475.249 |      357.696 |
| seasonal_median | mae      |      733.769 |       867.318 |      644.086 |

## Метрики фиксированного train/validation и финального test

| model           |   train_mae |   train_wape |   validation_mae |   validation_wape | test_mae   | test_wape   |
|:----------------|------------:|-------------:|-----------------:|------------------:|:-----------|:------------|
| seasonal_median |     410.472 |     0.636434 |          548.828 |          0.566426 | —          | —           |
| hgb             |      53.151 |     0.08241  |          161.085 |          0.16625  | —          | —           |
| mlp_onehot      |      90.734 |     0.140682 |          211.841 |          0.218633 | —          | —           |
| mlp_embedding   |      96.777 |     0.150053 |          184.303 |          0.190213 | 191.095    | 0.224749    |

- MLP embedding validation MAE: **184.303**
- MLP embedding validation WAPE: **0.190213**
- MLP embedding test MAE: **191.095**
- MLP embedding test WAPE: **0.224749**

HGB и seasonal median сохранены как сравниваемые модели, но не promoted.
Seed и точные версии зависимостей фиксируются; это делает запуск контролируемым,
но не обещает bit-for-bit совпадение между независимыми средами и обучениями.

## Параметры сетей

- `mlp_onehot`: 14,081 параметров, лучшая эпоха 126 из 151
- `mlp_embedding`: 10,878 параметров, лучшая эпоха 156 из 181
