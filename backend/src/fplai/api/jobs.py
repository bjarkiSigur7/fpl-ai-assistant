"""Background jobs for the API: the refresh pipeline and per-entry my-team solves.

Both are *expensive* (the refresh pipeline pulls + rebuilds everything; a my-team solve
runs ``SquadState.from_entry`` + ``build_recommendation`` — minutes of MILP), so requests
never run them inline.  :class:`JobRegistry` gives each named job single-flight
semantics: starting an already-running job is a no-op reported to the caller, who
answers 409 (refresh) or 202 (my-team).

The workers are module-level functions (:func:`run_refresh`, :func:`solve_my_team`)
resolved from module globals at call time — tests monkeypatch them with fast fakes.
Refresh output (the pipeline's rich-console prints + ``fplai`` log records) is captured
into a :class:`LogBuffer` served by ``GET /api/refresh/status``.
"""

from __future__ import annotations

import contextlib
import json
import logging
import sys
import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import IO, Any

__all__ = [
    "Job",
    "JobRegistry",
    "LogBuffer",
    "my_team_cache_path",
    "refresh_log",
    "registry",
    "run_refresh",
    "run_refresh_captured",
    "solve_my_team",
]

REFRESH_JOB = "refresh"


# --------------------------------------------------------------------------------------
# Log capture
# --------------------------------------------------------------------------------------


class LogBuffer:
    """Thread-safe line buffer with a file-like ``write`` interface.

    Collects the refresh pipeline's console output (rich resolves ``sys.stdout``
    dynamically, so ``contextlib.redirect_stdout`` reaches it) plus ``fplai`` logger
    records.  Keeps the last ``maxlen`` complete lines.
    """

    def __init__(self, maxlen: int = 2000) -> None:
        self._lines: deque[str] = deque(maxlen=maxlen)
        self._partial = ""
        self._lock = threading.Lock()

    def write(self, text: str) -> int:
        """File-protocol write: split into lines, keep the trailing partial."""
        with self._lock:
            self._partial += text
            *complete, self._partial = self._partial.split("\n")
            self._lines.extend(line.rstrip() for line in complete)
        return len(text)

    def flush(self) -> None:
        """File-protocol flush (no-op; lines are committed on newline)."""

    def isatty(self) -> bool:
        """Not a terminal — keeps rich in plain-text mode."""
        return False

    def tail(self, n: int = 40) -> list[str]:
        """The last ``n`` complete lines (plus any trailing partial line)."""
        with self._lock:
            lines = list(self._lines)
            if self._partial.strip():
                lines.append(self._partial.rstrip())
        return lines[-n:]

    def clear(self) -> None:
        """Drop all buffered output (called at the start of each refresh run)."""
        with self._lock:
            self._lines.clear()
            self._partial = ""


class _Tee:
    """Duplicate writes to the real stdout and a :class:`LogBuffer`."""

    def __init__(self, primary: IO[str], buffer: LogBuffer) -> None:
        self._primary = primary
        self._buffer = buffer

    def write(self, text: str) -> int:
        with contextlib.suppress(Exception):  # never let a broken stdout kill the job
            self._primary.write(text)
        return self._buffer.write(text)

    def flush(self) -> None:
        with contextlib.suppress(Exception):
            self._primary.flush()

    def isatty(self) -> bool:
        return False


#: The (single-flight) refresh job's log buffer, served by GET /api/refresh/status.
refresh_log = LogBuffer()


# --------------------------------------------------------------------------------------
# Single-flight job registry
# --------------------------------------------------------------------------------------


@dataclass
class Job:
    """One named background job and its latest outcome."""

    name: str
    state: str = "idle"  # idle | running | done | error
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error: str | None = None
    result: Any = None
    thread: threading.Thread | None = field(default=None, repr=False)


class JobRegistry:
    """Named single-flight background jobs running in daemon threads."""

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def get(self, name: str) -> Job | None:
        """The job record for ``name`` (None if it never ran)."""
        with self._lock:
            return self._jobs.get(name)

    def start(self, name: str, target: Any) -> tuple[Job, bool]:
        """Start ``target()`` in a daemon thread unless ``name`` is already running.

        Returns ``(job, started)`` — ``started=False`` means the caller lost the
        single-flight race and should answer 409/202 with the running job's status.
        """
        with self._lock:
            job = self._jobs.setdefault(name, Job(name=name))
            if job.state == "running":
                return job, False
            job.state = "running"
            job.started_at = datetime.now(UTC)
            job.finished_at = None
            job.error = None

            def _run() -> None:
                try:
                    result = target()
                except Exception as exc:  # noqa: BLE001 - surfaced via job.error
                    logging.getLogger(__name__).exception("job %s failed", name)
                    job.error = f"{type(exc).__name__}: {exc}"
                    job.state = "error"
                else:
                    job.result = result
                    job.state = "done"
                finally:
                    job.finished_at = datetime.now(UTC)

            thread = threading.Thread(target=_run, name=f"fplai-job-{name}", daemon=True)
            job.thread = thread
        thread.start()
        return job, True

    def reset(self) -> None:
        """Forget every job (test hook; does not stop running threads)."""
        with self._lock:
            self._jobs.clear()


#: Process-wide singleton used by the API endpoints.
registry = JobRegistry()


# --------------------------------------------------------------------------------------
# Workers (module-level seams — tests monkeypatch these)
# --------------------------------------------------------------------------------------


def run_refresh() -> None:
    """Run the full refresh pipeline (proxy to :func:`fplai.pipeline.run_refresh`)."""
    from fplai import pipeline

    pipeline.run_refresh()


def run_refresh_captured() -> None:
    """Run :func:`run_refresh` with stdout + ``fplai`` logs captured into ``refresh_log``.

    ``redirect_stdout`` is process-global, so any concurrent stdout writes land in the
    buffer too — acceptable for a single-flight background job on a local tool.
    """
    refresh_log.clear()
    tee = _Tee(sys.stdout, refresh_log)
    handler = logging.StreamHandler(refresh_log)
    handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    fplai_logger = logging.getLogger("fplai")
    fplai_logger.addHandler(handler)
    try:
        with contextlib.redirect_stdout(tee):
            run_refresh()
    finally:
        fplai_logger.removeHandler(handler)


def my_team_cache_path(entry_id: int) -> Any:
    """Per-entry recommendation cache file (under ``config.CACHE_DIR`` — ours to write)."""
    from fplai import config

    return config.CACHE_DIR / "api" / f"my_team_{entry_id}.json"


def solve_my_team(entry_id: int) -> dict[str, Any]:
    """Build SquadState + Recommendation for one entry (EXPENSIVE — minutes of MILP).

    Reads the squad from the public FPL API (``SquadState.from_entry``), takes the xp
    frame from ``predictions_gw.parquet`` (season/GW window of the state), builds the
    prices frame from the latest per-player row, then runs
    ``optimizer.plans.build_recommendation``.  The JSON-safe payload is written to the
    per-entry cache file and returned (also kept on the job record in memory).
    """
    from fplai import config
    from fplai.api.cache import cache
    from fplai.optimizer.plans import build_recommendation
    from fplai.optimizer.state import SquadState

    state = SquadState.from_entry(entry_id)
    gw_df = cache.load_parquet(config.PROCESSED_DIR / "predictions_gw.parquet")
    window = gw_df[(gw_df["season"] == state.season) & (gw_df["gw"] >= state.current_gw)]
    if window.empty:
        raise RuntimeError(
            f"no predictions for season {state.season} GW>={state.current_gw} — "
            "run `fplai predict` (or `fplai refresh`) first"
        )
    xp = window[["season", "gw", "player_code", "xp", "q0"]].reset_index(drop=True)
    price_cols = ["player_code", "price", "position", "team_code", "web_name"]
    prices = (
        window.sort_values(["player_code", "gw"], kind="stable")
        .groupby("player_code", as_index=False)
        .tail(1)[price_cols]
        .reset_index(drop=True)
    )
    now = datetime.now(UTC)
    recommendation = build_recommendation(state, xp, prices, as_of=now)
    payload: dict[str, Any] = {
        "entry_id": entry_id,
        "generated_at": now.isoformat(),
        "squad_state": state.model_dump(mode="json"),
        "recommendation": recommendation.model_dump(mode="json"),
    }
    path = my_team_cache_path(entry_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    return payload
