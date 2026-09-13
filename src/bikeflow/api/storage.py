"""SQLite journal of served predictions and the actual demand that arrives later.

Every `/predict` call is recorded with its input, prediction and model version.
The true number of rentals is only known once the hour is over, so it is added
to the same row afterwards. This journal feeds the prediction table in the UI,
drift detection and retraining.

The standard-library `sqlite3` is used on purpose: no extra dependency, and the
API is the only process that owns this file. A connection is opened per
operation, which keeps FastAPI's worker threads independent.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    prediction_time TEXT NOT NULL,
    features TEXT NOT NULL,
    predicted_rentals REAL NOT NULL,
    model_version TEXT NOT NULL,
    actual_rentals REAL,
    actual_recorded_at TEXT
);
CREATE INDEX IF NOT EXISTS predictions_by_time ON predictions (prediction_time);
CREATE TABLE IF NOT EXISTS drift_checks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    checked_at TEXT NOT NULL,
    result TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS retrainings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    finished_at TEXT NOT NULL,
    result TEXT NOT NULL
);
"""


@dataclass(frozen=True)
class StoredPrediction:
    """One journal row."""

    id: int
    created_at: datetime
    prediction_time: datetime
    features: dict[str, Any]
    predicted_rentals: float
    model_version: str
    actual_rentals: float | None
    actual_recorded_at: datetime | None

    @property
    def absolute_error(self) -> float | None:
        if self.actual_rentals is None:
            return None
        return abs(self.predicted_rentals - self.actual_rentals)


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _row_to_prediction(row: sqlite3.Row) -> StoredPrediction:
    return StoredPrediction(
        id=row["id"],
        created_at=datetime.fromisoformat(row["created_at"]),
        prediction_time=datetime.fromisoformat(row["prediction_time"]),
        features=json.loads(row["features"]),
        predicted_rentals=row["predicted_rentals"],
        model_version=row["model_version"],
        actual_rentals=row["actual_rentals"],
        actual_recorded_at=(
            datetime.fromisoformat(row["actual_recorded_at"]) if row["actual_recorded_at"] else None
        ),
    )


class PredictionStore:
    """Read and write the prediction journal."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connection() as connection:
            connection.executescript(SCHEMA)

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def add(
        self,
        prediction_time: datetime,
        features: Mapping[str, Any],
        predicted_rentals: float,
        model_version: str,
    ) -> int:
        """Record a served prediction and return its id."""
        with self._connection() as connection:
            cursor = connection.execute(
                "INSERT INTO predictions "
                "(created_at, prediction_time, features, predicted_rentals, model_version) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    _now(),
                    prediction_time.isoformat(),
                    json.dumps(dict(features), ensure_ascii=False, sort_keys=True),
                    float(predicted_rentals),
                    model_version,
                ),
            )
            return int(cursor.lastrowid)

    def get(self, prediction_id: int) -> StoredPrediction | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM predictions WHERE id = ?", (prediction_id,)
            ).fetchone()
        return _row_to_prediction(row) if row else None

    def record_actual(self, prediction_id: int, actual_rentals: float) -> StoredPrediction | None:
        """Attach the observed demand to a prediction. None if the id is unknown."""
        with self._connection() as connection:
            cursor = connection.execute(
                "UPDATE predictions SET actual_rentals = ?, actual_recorded_at = ? WHERE id = ?",
                (float(actual_rentals), _now(), prediction_id),
            )
            if cursor.rowcount == 0:
                return None
        return self.get(prediction_id)

    def recent(self, limit: int = 100) -> list[StoredPrediction]:
        """The most recently served predictions, newest first."""
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT * FROM predictions ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [_row_to_prediction(row) for row in rows]

    def monitoring_window(self, model_version: str, hours: int) -> list[StoredPrediction]:
        """Entries with an actual demand from one model, over its latest `hours`.

        The window ends at the latest predicted hour rather than the wall clock, so
        replayed historical traffic is monitored the same way as live traffic.
        Prediction times are stored normalised to Asia/Seoul, so ISO strings
        compare in time order.
        """
        with self._connection() as connection:
            latest = connection.execute(
                "SELECT MAX(prediction_time) FROM predictions "
                "WHERE model_version = ? AND actual_rentals IS NOT NULL",
                (model_version,),
            ).fetchone()[0]
            if latest is None:
                return []
            start = (datetime.fromisoformat(latest) - timedelta(hours=hours)).isoformat()
            rows = connection.execute(
                "SELECT * FROM predictions WHERE model_version = ? "
                "AND actual_rentals IS NOT NULL AND prediction_time > ? "
                "ORDER BY prediction_time",
                (model_version, start),
            ).fetchall()
        return [_row_to_prediction(row) for row in rows]

    def training_rows(self) -> list[StoredPrediction]:
        """Every entry with an actual demand, oldest first — the data for retraining."""
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT * FROM predictions WHERE actual_rentals IS NOT NULL "
                "ORDER BY prediction_time, id"
            ).fetchall()
        return [_row_to_prediction(row) for row in rows]

    def add_retraining(self, result: Mapping[str, Any]) -> int:
        """Store the outcome of a retraining and return its id."""
        with self._connection() as connection:
            cursor = connection.execute(
                "INSERT INTO retrainings (finished_at, result) VALUES (?, ?)",
                (_now(), json.dumps(dict(result), ensure_ascii=False)),
            )
            return int(cursor.lastrowid)

    def latest_retraining(self) -> dict[str, Any] | None:
        """The most recent finished retraining, or None."""
        with self._connection() as connection:
            row = connection.execute(
                "SELECT result FROM retrainings ORDER BY id DESC LIMIT 1"
            ).fetchone()
        return json.loads(row["result"]) if row else None

    def add_drift_check(self, result: Mapping[str, Any]) -> int:
        """Store the outcome of a drift check and return its id."""
        with self._connection() as connection:
            cursor = connection.execute(
                "INSERT INTO drift_checks (checked_at, result) VALUES (?, ?)",
                (_now(), json.dumps(dict(result), ensure_ascii=False)),
            )
            return int(cursor.lastrowid)

    def latest_drift_check(self) -> dict[str, Any] | None:
        """The most recent drift check, with its id, or None if there was none."""
        with self._connection() as connection:
            row = connection.execute(
                "SELECT id, result FROM drift_checks ORDER BY id DESC LIMIT 1"
            ).fetchone()
        if row is None:
            return None
        return {"check_id": row["id"], **json.loads(row["result"])}
