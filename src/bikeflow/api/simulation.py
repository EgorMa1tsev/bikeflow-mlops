"""The replay scenario started from the API, in a background thread.

Replaying the test period with drift checks and retrainings takes minutes, so it
runs in a thread like retraining does. The thread is an ordinary API client: it
sends the hours to this same API over HTTP, exactly as `python -m bikeflow.replay`
would. The manager refuses a second start and keeps the log of the last run.
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger("bikeflow.api.simulation")

Log = Callable[[str], None]
Job = Callable[[dict[str, Any], Log], dict[str, Any]]
LOG_LINES = 200


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class ReplayManager:
    """Run one replay at a time in a background thread and report its progress."""

    def __init__(self, job: Job) -> None:
        self._job = job
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._log: deque[str] = deque(maxlen=LOG_LINES)
        self._state: dict[str, Any] = {
            "state": "idle",
            "params": None,
            "started_at": None,
            "finished_at": None,
            "result": None,
            "error": None,
        }

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, params: dict[str, Any]) -> bool:
        """Start a replay unless one is already running. Returns whether it started."""
        with self._lock:
            if self.running:
                return False
            self._log.clear()
            self._state = {
                "state": "running",
                "params": params,
                "started_at": _now(),
                "finished_at": None,
                "result": None,
                "error": None,
            }
            self._thread = threading.Thread(
                target=self._run, args=(params,), name="bikeflow-replay", daemon=True
            )
            self._thread.start()
            logger.info("replay_started params=%s", params)
            return True

    def _append(self, line: str) -> None:
        with self._lock:
            self._log.append(line)

    def _run(self, params: dict[str, Any]) -> None:
        try:
            result = self._job(params, self._append)
        except Exception as exc:  # noqa: BLE001 - reported through the status endpoint
            logger.error("replay_failed error=%s", exc)
            update = {"state": "failed", "error": f"{type(exc).__name__}: {exc}"}
        else:
            logger.info("replay_finished result=%s", result)
            update = {"state": "finished", "result": result}
        with self._lock:
            self._state.update(update, finished_at=_now())

    def wait(self, timeout: float | None = None) -> None:
        """Block until the current run ends; used by tests."""
        thread = self._thread
        if thread is not None:
            thread.join(timeout)

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {**self._state, "log": list(self._log)}
