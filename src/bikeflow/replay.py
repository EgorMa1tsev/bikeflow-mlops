"""Replay historical hours through the API as if they were arriving live.

For every hour of the chosen period the replay sends the weather to `/predict`,
exactly as a client would, and reports the actual demand for that hour only
after a delay: in reality the number of rentals is known once the hour is over.
This fills the prediction journal with realistic traffic for monitoring, drift
detection and retraining.

By default the test period (October–November 2018) is replayed: the model has
never seen it. `--evening-boost` multiplies evening demand from a chosen date,
imitating a promotion. The weather stays the same, so the model's error grows
without any change in its inputs — a controlled concept drift.

    python -m bikeflow.replay --hours 72
    python -m bikeflow.replay --interval 0.5 --evening-boost 1.6 --boost-from 2018-11-01

The API can run the same scenario in its own background thread: `POST /replay`.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from bikeflow.ml.data.preprocess import load_raw, to_canonical
from bikeflow.ml.data.split import split_bounds
from bikeflow.ml.features import TARGET, api_input_contract

SEOUL = "Asia/Seoul"
EVENING_HOURS = range(17, 22)

Post = Callable[[str, dict[str, Any]], dict[str, Any]]
Get = Callable[[str], dict[str, Any]]


class ReplayError(RuntimeError):
    """Raised when the API refuses a replayed request."""


def http_post(base_url: str) -> Post:
    """POST JSON to the API with the standard library only."""

    def post(path: str, body: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            base_url.rstrip("/") + path,
            data=json.dumps(body).encode("utf-8"),
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")
            raise ReplayError(f"POST {path} -> HTTP {exc.code}: {detail}") from exc

    return post


def http_get(base_url: str) -> Get:
    """GET JSON from the API with the standard library only."""

    def get(path: str) -> dict[str, Any]:
        with urllib.request.urlopen(base_url.rstrip("/") + path, timeout=30) as response:
            return json.load(response)

    return get


def load_hours(start: str | None = None, end: str | None = None) -> pd.DataFrame:
    """Canonical hourly rows for the period, by default the test split."""
    test_start, test_end = split_bounds()["test"]
    first = pd.Timestamp(start) if start else test_start
    last = pd.Timestamp(end) if end else test_end
    frame = to_canonical(load_raw())
    window = frame["timestamp"].between(first, last + pd.Timedelta(hours=23))
    return frame.loc[window].sort_values("timestamp").reset_index(drop=True)


def to_request(row: pd.Series) -> dict[str, Any]:
    """Turn one canonical row into a `/predict` request body."""
    casts = {"float": float, "int": int, "bool": bool}
    body: dict[str, Any] = {
        public: casts[spec["kind"]](row[spec["canonical_name"]])
        for public, spec in api_input_contract().items()
    }
    body["prediction_time"] = pd.Timestamp(row["timestamp"]).tz_localize(SEOUL).isoformat()
    return body


def observed_demand(row: pd.Series, boost: float, boost_from: pd.Timestamp | None) -> float:
    """Actual rentals, optionally inflated in the evening to simulate concept drift."""
    demand = float(row[TARGET])
    boosted = (
        boost != 1.0
        and int(row["hour"]) in EVENING_HOURS
        and (boost_from is None or pd.Timestamp(row["timestamp"]) >= boost_from)
    )
    return float(round(demand * boost)) if boosted else demand


@dataclass
class ReplayStats:
    predictions: int = 0
    actuals: int = 0
    absolute_errors: list[float] = field(default_factory=list)
    drift_checks: int = 0
    concept_drift_alerts: int = 0
    retrainings_started: int = 0
    retrainings_promoted: int = 0

    @property
    def mae(self) -> float | None:
        if not self.absolute_errors:
            return None
        return sum(self.absolute_errors) / len(self.absolute_errors)


def replay(
    hours: Iterable[pd.Series],
    post: Post,
    delay_hours: int = 1,
    interval: float = 0.0,
    boost: float = 1.0,
    boost_from: pd.Timestamp | None = None,
    report_every: int = 24,
    check_every: int = 0,
    log: Callable[[str], None] = print,
    get: Get | None = None,
    retraining_timeout: float = 600.0,
) -> ReplayStats:
    """Send each hour to the API and report its actual demand `delay_hours` later.

    With `check_every` set, a drift check runs after every that many reported
    hours, the way a scheduler would trigger it in production.

    When a drift check starts a retraining and `get` is given, the replay waits
    for it to finish. Replayed hours pass in milliseconds while training takes
    seconds; in real time a retraining is instant next to an hour of traffic,
    and waiting keeps that proportion, so the new model serves the hours after it.
    """

    def wait_for_retraining() -> None:
        deadline = time.monotonic() + retraining_timeout
        status = get("/retrain/status")
        while status["state"] == "running" and time.monotonic() < deadline:
            time.sleep(1.0)
            status = get("/retrain/status")
        if status["state"] == "running":
            log("[retrain] переобучение идёт дольше таймаута, поток продолжается")
            return
        if status["state"] == "failed":
            log(f"[retrain] переобучение упало: {status['error']}")
            return
        result = status["result"]
        verdict = (
            "gate пройден, в работе новая модель"
            if result["promoted"]
            else "gate не пройден, модель прежняя"
        )
        log(
            f"[retrain] MAE на отложенных часах {result['champion_mae']:.0f} -> "
            f"{result['challenger_mae']:.0f} ({100 * result['improvement']:+.1f}%): {verdict}"
        )
        stats.retrainings_promoted += bool(result["promoted"])

    stats = ReplayStats()
    pending: deque[tuple[pd.Timestamp, int, float, float]] = deque()
    delay = pd.Timedelta(hours=delay_hours)

    def report(until: pd.Timestamp | None) -> None:
        while pending and (until is None or pending[0][0] <= until):
            _, prediction_id, predicted, actual = pending.popleft()
            post(f"/predictions/{prediction_id}/actual", {"actual_rentals": actual})
            stats.actuals += 1
            stats.absolute_errors.append(abs(predicted - actual))

    def maybe_check_drift(force: bool = False) -> None:
        if not check_every or not (
            force or stats.actuals >= check_every * (stats.drift_checks + 1)
        ):
            return
        stats.drift_checks += 1
        try:
            result = post("/drift/check", {})
        except ReplayError as exc:
            if "HTTP 422" in str(exc):
                log("[drift] мало данных для проверки")
                return
            raise
        stats.concept_drift_alerts += bool(result["concept_drift"])
        flag = "ДРЕЙФ МОДЕЛИ" if result["concept_drift"] else "модель в норме"
        log(
            f"[drift] до {result['window_end'][:16]}: {flag} "
            f"(MAE {result['current_mae']:.0f} = {result['mae_ratio']:.2f} × эталона); "
            f"data drift {'да' if result['data_drift'] else 'нет'}, "
            f"target drift {'да' if result['target_drift'] else 'нет'}"
        )
        if result.get("retraining_started"):
            stats.retrainings_started += 1
            log("[drift] запущено переобучение")
            if get is not None:
                wait_for_retraining()

    for row in hours:
        moment = pd.Timestamp(row["timestamp"])
        response = post("/predict", to_request(row))
        stats.predictions += 1
        pending.append(
            (
                moment,
                int(response["prediction_id"]),
                float(response["predicted_rentals"]),
                observed_demand(row, boost, boost_from),
            )
        )
        report(moment - delay)
        maybe_check_drift()

        if report_every and stats.predictions % report_every == 0:
            mae = f"{stats.mae:.1f}" if stats.mae is not None else "—"
            log(
                f"[replay] {moment:%Y-%m-%d %H:00}  прогнозов {stats.predictions}, "
                f"фактов {stats.actuals}, MAE {mae}"
            )
        if interval:
            time.sleep(interval)

    report(None)
    maybe_check_drift(force=True)
    return stats


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m bikeflow.replay",
        description="Replay historical hours through the BikeFlow API.",
    )
    parser.add_argument(
        "--api",
        default=os.environ.get("BIKEFLOW_API_URL", "http://127.0.0.1:8000"),
        help="API base URL (default: $BIKEFLOW_API_URL or http://127.0.0.1:8000)",
    )
    parser.add_argument("--start", help="first day, YYYY-MM-DD (default: start of test split)")
    parser.add_argument("--end", help="last day, YYYY-MM-DD (default: end of test split)")
    parser.add_argument("--hours", type=int, help="replay at most this many hours")
    parser.add_argument(
        "--delay", type=int, default=1, help="hours before the actual demand is reported"
    )
    parser.add_argument("--interval", type=float, default=0.0, help="seconds to wait between hours")
    parser.add_argument(
        "--evening-boost",
        type=float,
        default=1.0,
        help="multiply 17:00-21:00 demand by this factor (concept drift)",
    )
    parser.add_argument("--boost-from", help="apply the boost from this day, YYYY-MM-DD")
    parser.add_argument(
        "--check-every",
        type=int,
        default=0,
        help="run a drift check after every N reported hours (default: off)",
    )
    return parser


def run_scenario(
    api: str,
    start: str | None = None,
    end: str | None = None,
    hours: int | None = None,
    delay: int = 1,
    interval: float = 0.0,
    evening_boost: float = 1.0,
    boost_from: str | None = None,
    check_every: int = 0,
    log: Callable[[str], None] = print,
) -> ReplayStats:
    """Load the period and replay it against the API at `api`, logging progress."""
    frame = load_hours(start, end)
    if hours is not None:
        frame = frame.head(hours)

    boost_start = pd.Timestamp(boost_from) if boost_from else None
    log(
        f"[replay] {len(frame)} ч: {frame['timestamp'].min():%Y-%m-%d %H:00} .. "
        f"{frame['timestamp'].max():%Y-%m-%d %H:00} -> {api}"
    )
    if evening_boost != 1.0:
        since = f" с {boost_start:%Y-%m-%d}" if boost_start is not None else ""
        log(f"[replay] вечерний спрос ×{evening_boost}{since}")

    stats = replay(
        (row for _, row in frame.iterrows()),
        http_post(api),
        delay_hours=delay,
        interval=interval,
        boost=evening_boost,
        boost_from=boost_start,
        check_every=check_every,
        log=log,
        get=http_get(api),
    )
    mae = f"{stats.mae:.1f}" if stats.mae is not None else "—"
    log(f"[replay] готово: прогнозов {stats.predictions}, фактов {stats.actuals}, MAE {mae}")
    if stats.drift_checks:
        log(
            f"[drift] проверок {stats.drift_checks}, "
            f"из них с дрейфом модели {stats.concept_drift_alerts}, "
            f"переобучений {stats.retrainings_started}, "
            f"новая модель введена в работу {stats.retrainings_promoted} раз"
        )
    return stats


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_scenario(
        args.api,
        start=args.start,
        end=args.end,
        hours=args.hours,
        delay=args.delay,
        interval=args.interval,
        evening_boost=args.evening_boost,
        boost_from=args.boost_from,
        check_every=args.check_every,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
