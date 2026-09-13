"""One retraining at a time, in the background of the API process.

Training takes seconds to a minute, too long to hold an HTTP request open, so it
runs in a thread. The manager refuses a second start while one is running and
remembers the state of the last run for the status endpoint.
"""

from __future__ import annotations

import logging
import threading
import traceback
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger("bikeflow.api.retraining")

Job = Callable[[str], dict[str, Any]]


class RetrainingManager:
    """Start retraining jobs in a background thread and report their state."""

    def __init__(self, job: Job, enabled: bool = True) -> None:
        self._job = job
        self.enabled = enabled
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._state: dict[str, Any] = {
            "state": "idle",
            "trigger": None,
            "started_at": None,
            "finished_at": None,
            "result": None,
            "error": None,
        }

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, trigger: str) -> bool:
        """Start a retraining unless one is already running. Returns whether it started."""
        with self._lock:
            if self.running:
                return False
            self._state = {
                "state": "running",
                "trigger": trigger,
                "started_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "finished_at": None,
                "result": None,
                "error": None,
            }
            self._thread = threading.Thread(
                target=self._run, args=(trigger,), name="bikeflow-retraining", daemon=True
            )
            self._thread.start()
            logger.info("retraining_started trigger=%s", trigger)
            return True

    def _run(self, trigger: str) -> None:
        try:
            result = self._job(trigger)
        except Exception as exc:  # noqa: BLE001 - reported through the status endpoint
            logger.error("retraining_failed trigger=%s error=%s", trigger, exc)
            update = {"state": "failed", "error": f"{type(exc).__name__}: {exc}"}
            logger.debug(traceback.format_exc())
        else:
            logger.info(
                "retraining_finished trigger=%s promoted=%s", trigger, result.get("promoted")
            )
            update = {"state": "finished", "result": result}
        with self._lock:
            self._state.update(update, finished_at=datetime.now(UTC).isoformat(timespec="seconds"))

    def wait(self, timeout: float | None = None) -> None:
        """Block until the current run ends; used by tests and scripts."""
        thread = self._thread
        if thread is not None:
            thread.join(timeout)

    def status(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._state)
