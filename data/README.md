# Data

No dataset is committed to Git. Everything here is fetched and derived locally by `dvc repro` or `make data`.

```
data/raw/SeoulBikeData.csv     downloaded from UCI, cp1252, dates DD/MM/YYYY
data/raw/dataset_meta.json     source URL, sha256, size, download timestamp
data/processed/dataset.parquet canonical hourly frame, contract-checked
data/processed/train.parquet
data/processed/validation.parquet
data/processed/test.parquet
```

The download is idempotent and records a sha256 that every trained artifact carries in its
metadata, so a model can always be traced back to the exact bytes it was trained on.

`preprocess` fails loudly if the data violates any assumption the pipeline relies on: row count,
missing values, duplicate or non-contiguous timestamps, hour range, negative targets, unknown
seasons, and the rule that a non-operating hour always has zero rentals.

Full description of columns, units, split boundaries and known limitations:
[`docs/model/data_card.md`](../docs/model/data_card.md).

The same steps are the first three stages of the DVC pipeline in `dvc.yaml`: `dvc repro` runs
them and reruns only what changed.
