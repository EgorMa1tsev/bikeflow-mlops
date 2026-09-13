# Выбор основной модели

Сгенерировано: 2026-09-13T03:26:15+03:00

Production-модель — **MLP embedding (`mlp_embedding`)**. Она выбрана по
минимальному среднему **MAE** на трёх rolling-origin folds; WAPE —
дополнительная метрика. Test не участвовал в выборе и был рассчитан только
после фиксации победителя, только для MLP embedding.

## Rolling-origin cross-validation

| model           | metric   |   mean_score |   worst_score |   best_score |
|:----------------|:---------|-------------:|--------------:|-------------:|
| mlp_embedding   | mae      |      386.753 |       418.806 |      370.094 |
| hgb             | mae      |      416.625 |       582.045 |      211.527 |
| mlp_onehot      | mae      |      422.094 |       475.128 |      359.94  |
| seasonal_median | mae      |      733.769 |       867.318 |      644.086 |

## Метрики фиксированного train/validation и финального test

| model           |   train_mae |   train_wape |   validation_mae |   validation_wape | test_mae   | test_wape   |
|:----------------|------------:|-------------:|-----------------:|------------------:|:-----------|:------------|
| seasonal_median |     410.472 |     0.636434 |          548.828 |          0.566426 | —          | —           |
| hgb             |      53.151 |     0.08241  |          161.085 |          0.16625  | —          | —           |
| mlp_onehot      |      96.566 |     0.149725 |          213.215 |          0.220051 | —          | —           |
| mlp_embedding   |      99.05  |     0.153576 |          183.139 |          0.189011 | 189.509    | 0.222884    |

- MLP embedding validation MAE: **183.139**
- MLP embedding validation WAPE: **0.189011**
- MLP embedding test MAE: **189.509**
- MLP embedding test WAPE: **0.222884**

HGB и seasonal median сохранены как сравниваемые модели, но не promoted.
Seed и точные версии зависимостей фиксируются; это делает запуск контролируемым,
но не обещает bit-for-bit совпадение между независимыми средами и обучениями.

## Параметры сетей

- `mlp_onehot`: 14,081 параметров, лучшая эпоха 111 из 136
- `mlp_embedding`: 10,878 параметров, лучшая эпоха 156 из 181
