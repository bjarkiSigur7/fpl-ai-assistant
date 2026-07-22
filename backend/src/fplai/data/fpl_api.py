"""FPL API client: throttled polite HTTP, on-disk cache, snapshot archiver, launch detector.

This module owns *all* HTTP etiquette for the project (MODEL_DESIGN_INPUTS §5.3):

- ``polite_get`` is the shared throttled fetch helper every data client imports —
  per-host minimum request interval (monotonic clock), browser User-Agent, tenacity
  retries with exponential backoff on 429/5xx/transport errors, and an on-disk cache
  under ``config.CACHE_DIR/http`` keyed by URL hash.
- ``FplApiClient`` wraps the public endpoints at https://fantasy.premierleague.com/api/.
- ``take_snapshot`` archives today's ``bootstrap-static`` + ``fixtures`` to
  ``RAW_DIR/fpl_api/snapshots/{YYYY-MM-DD}/`` — the append-only archive that must run
  daily from 2026-27 launch day (unrecoverable if missed).
- ``season_state`` / ``parse_season_state`` implement the launch detector from the
  FPL_KNOWLEDGE re-verification protocol: when ``static_content_url`` flips to
  ``2026_27`` the pipeline re-verifies every rule against the live API.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
import re
import threading
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx
from tenacity import (
    Retrying,
    before_sleep_log,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from fplai import config

logger = logging.getLogger(__name__)

BASE_URL = "https://fantasy.premierleague.com/api/"
BROWSER_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)
DEFAULT_TIMEOUT_S = 30.0
RETRY_ATTEMPTS = 5
RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})

# Cache TTLs (seconds) per endpoint family.
TTL_BOOTSTRAP = 3600  # 1h
TTL_FIXTURES = 6 * 3600  # 6h
TTL_ELEMENT_SUMMARY = 12 * 3600  # 12h
TTL_EVENT_LIVE = 60  # live scores move fast
TTL_EVENT_STATUS = 300
TTL_ENTRY = 300  # 5min for all entry/* endpoints
TTL_SET_PIECE = 6 * 3600

# Patchable clock/sleep hooks — tests monkeypatch these module attributes.
_monotonic = time.monotonic
_wall = time.time
_sleep = time.sleep

# Per-host throttle state. `_last_request_at` maps host -> monotonic timestamp of the
# last *reserved* request slot; guarded by a lock so threaded sweeps stay polite.
_throttle_lock = threading.Lock()
_last_request_at: dict[str, float] = {}

_client_lock = threading.Lock()
_shared_client: httpx.Client | None = None


class RetryableStatusError(Exception):
    """A 429/5xx response — retried with backoff, raised when attempts are exhausted."""

    def __init__(self, response: httpx.Response) -> None:
        super().__init__(f"HTTP {response.status_code} from {response.request.url}")
        self.response = response


def _get_shared_client() -> httpx.Client:
    """Return the lazily-created module-level ``httpx.Client``."""
    global _shared_client
    with _client_lock:
        if _shared_client is None or _shared_client.is_closed:
            _shared_client = httpx.Client(timeout=DEFAULT_TIMEOUT_S, follow_redirects=True)
        return _shared_client


def _throttle(host: str, min_interval_s: float) -> None:
    """Block until at least ``min_interval_s`` has passed since the last request to ``host``."""
    if min_interval_s <= 0:
        return
    with _throttle_lock:
        now = _monotonic()
        earliest = _last_request_at.get(host, now - min_interval_s) + min_interval_s
        start = max(now, earliest)
        _last_request_at[host] = start  # reserve the slot before sleeping
        wait = start - now
    if wait > 0:
        _sleep(wait)


def _cache_paths(cache_dir: Path, url: str) -> tuple[Path, Path]:
    """Return (meta_path, body_path) for ``url`` under ``cache_dir``."""
    key = hashlib.sha256(url.encode("utf-8")).hexdigest()[:32]
    return cache_dir / f"{key}.meta.json", cache_dir / f"{key}.body"


def _cache_read(cache_dir: Path, url: str, cache_ttl_s: float) -> httpx.Response | None:
    """Return a synthesized cached response for ``url`` if fresh, else ``None``."""
    meta_path, body_path = _cache_paths(cache_dir, url)
    if not (meta_path.exists() and body_path.exists()):
        return None
    try:
        meta = json.loads(meta_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None  # corrupt cache entry -> treat as a miss
    if _wall() - float(meta.get("fetched_at", 0.0)) > cache_ttl_s:
        return None
    return httpx.Response(
        status_code=int(meta.get("status_code", 200)),
        content=body_path.read_bytes(),
        headers={
            "content-type": meta.get("content_type", "application/json"),
            "x-fplai-cache": "hit",
        },
        request=httpx.Request("GET", url),
    )


def _cache_write(cache_dir: Path, url: str, response: httpx.Response) -> None:
    """Persist a successful response body + metadata to the on-disk cache."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    meta_path, body_path = _cache_paths(cache_dir, url)
    _atomic_write_bytes(body_path, response.content)
    meta = {
        "url": url,
        "fetched_at": _wall(),
        "status_code": response.status_code,
        "content_type": response.headers.get("content-type", "application/json"),
    }
    _atomic_write_bytes(meta_path, json.dumps(meta).encode("utf-8"))


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write ``data`` to ``path`` via a temp file + rename (crash-safe)."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


def _atomic_write_json(path: Path, obj: Any, *, pretty: bool) -> None:
    """Serialize ``obj`` as JSON to ``path`` atomically."""
    if pretty:
        text = json.dumps(obj, indent=2, ensure_ascii=False)
    else:
        text = json.dumps(obj, separators=(",", ":"), ensure_ascii=False)
    _atomic_write_bytes(path, text.encode("utf-8"))


def polite_get(
    url: str,
    *,
    min_interval_s: float = 1.0,
    cache_ttl_s: float = 0.0,
    headers: Mapping[str, str] | None = None,
    cache_dir: Path | None = None,
    client: httpx.Client | None = None,
) -> httpx.Response:
    """Throttled, retried, cached HTTP GET — the shared fetch helper for all data modules.

    Args:
        url: Absolute URL to fetch.
        min_interval_s: Minimum spacing between requests to the same host (monotonic
            clock; shared module-wide, so every caller collectively stays polite).
        cache_ttl_s: Serve from the on-disk cache when the entry is younger than this
            many seconds. ``0`` bypasses the cache read (a fresh fetch still refreshes
            the cache for other callers).
        headers: Extra request headers; a browser User-Agent is always set.
        cache_dir: Cache directory (default ``config.CACHE_DIR / "http"``).
        client: Optional ``httpx.Client`` override (tests inject a MockTransport).

    Returns:
        The ``httpx.Response`` (status 200; cached responses carry ``x-fplai-cache: hit``).

    Raises:
        RetryableStatusError: 429/5xx persisted through all retry attempts.
        httpx.HTTPStatusError: non-retryable error status (e.g. 404).
        httpx.TransportError: network failure persisted through all retry attempts.
    """
    cache_dir = cache_dir if cache_dir is not None else config.CACHE_DIR / "http"
    if cache_ttl_s > 0:
        cached = _cache_read(cache_dir, url, cache_ttl_s)
        if cached is not None:
            return cached

    request_headers = {"User-Agent": BROWSER_UA, "Accept": "application/json;q=0.9,*/*;q=0.8"}
    if headers:
        request_headers.update(headers)
    host = urlsplit(url).netloc
    http = client if client is not None else _get_shared_client()

    def _fetch() -> httpx.Response:
        _throttle(host, min_interval_s)
        response = http.get(url, headers=request_headers)
        if response.status_code in RETRYABLE_STATUS:
            raise RetryableStatusError(response)
        return response

    retryer = Retrying(
        retry=retry_if_exception_type((httpx.TransportError, RetryableStatusError)),
        wait=wait_exponential(multiplier=1.0, min=1.0, max=30.0),
        stop=stop_after_attempt(RETRY_ATTEMPTS),
        sleep=_sleep,
        reraise=True,
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    response = retryer(_fetch)
    response.raise_for_status()
    _cache_write(cache_dir, url, response)
    return response


_SEASON_RE = re.compile(r"(20\d{2})_(\d{2})")


@dataclass(frozen=True)
class SeasonState:
    """Parsed launch/season status from ``bootstrap-static`` (the launch detector)."""

    season: int
    """Season start year (2025 = the 2025-26 season), from ``static_content_url``."""
    is_live_2026_27: bool
    """True once ``static_content_url`` has flipped to ``2026_27`` — triggers re-verification."""
    current_gw: int | None
    """Event id with ``is_current`` (None pre-season)."""
    next_gw: int | None
    """Event id with ``is_next`` (None when the season has fully finished)."""
    next_deadline_utc: dt.datetime | None
    """UTC deadline of the next event, if any."""
    total_players: int
    """Registered managers — sanity signal that the payload is real."""
    static_content_url: str
    """The raw URL the season was parsed from."""


def parse_season_state(bootstrap: Mapping[str, Any]) -> SeasonState:
    """Parse a ``bootstrap-static`` payload into a :class:`SeasonState`.

    The season comes from ``game_config.settings.static_content_url`` (observed live
    2026-07-22), with fallbacks to ``game_settings`` / top-level for older payload shapes.
    """
    game_config = bootstrap.get("game_config") or {}
    url = (
        (game_config.get("settings") or {}).get("static_content_url")
        or (bootstrap.get("game_settings") or {}).get("static_content_url")
        or bootstrap.get("static_content_url")
    )
    if not url:
        raise ValueError("bootstrap payload has no static_content_url — schema changed?")
    matches = _SEASON_RE.findall(url)
    if not matches:
        raise ValueError(f"cannot parse season from static_content_url: {url!r}")
    season = int(matches[-1][0])

    events: list[Mapping[str, Any]] = bootstrap.get("events") or []
    current_gw = next((int(e["id"]) for e in events if e.get("is_current")), None)
    next_event = next((e for e in events if e.get("is_next")), None)
    next_gw = int(next_event["id"]) if next_event else None
    next_deadline_utc: dt.datetime | None = None
    if next_event and next_event.get("deadline_time"):
        next_deadline_utc = dt.datetime.fromisoformat(str(next_event["deadline_time"]))
        if next_deadline_utc.tzinfo is None:
            next_deadline_utc = next_deadline_utc.replace(tzinfo=dt.UTC)
        else:
            next_deadline_utc = next_deadline_utc.astimezone(dt.UTC)

    return SeasonState(
        season=season,
        is_live_2026_27=season >= 2026,
        current_gw=current_gw,
        next_gw=next_gw,
        next_deadline_utc=next_deadline_utc,
        total_players=int(bootstrap.get("total_players") or 0),
        static_content_url=str(url),
    )


class FplApiClient:
    """Client for the public FPL API with polite throttling and per-endpoint cache TTLs."""

    def __init__(
        self,
        *,
        base_url: str = BASE_URL,
        min_interval_s: float = 1.0,
        cache_dir: Path | None = None,
        raw_dir: Path | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        """Create a client.

        Args:
            base_url: API root (default the real FPL API).
            min_interval_s: Per-host request spacing (≤1 req/s sustained per etiquette).
            cache_dir: HTTP cache directory (default ``config.CACHE_DIR / "http"``).
            raw_dir: Raw archive root (default ``config.RAW_DIR / "fpl_api"``).
            client: Optional ``httpx.Client`` override for tests.
        """
        self.base_url = base_url.rstrip("/") + "/"
        self.min_interval_s = min_interval_s
        self.cache_dir = cache_dir if cache_dir is not None else config.CACHE_DIR / "http"
        self.raw_dir = raw_dir if raw_dir is not None else config.RAW_DIR / "fpl_api"
        self._client = client

    def get_json(self, path: str, *, cache_ttl_s: float) -> Any:
        """GET ``base_url + path`` through :func:`polite_get` and decode JSON."""
        response = polite_get(
            self.base_url + path.lstrip("/"),
            min_interval_s=self.min_interval_s,
            cache_ttl_s=cache_ttl_s,
            cache_dir=self.cache_dir,
            client=self._client,
        )
        return response.json()

    # -- endpoints ---------------------------------------------------------------

    def bootstrap(self) -> dict[str, Any]:
        """``bootstrap-static/`` — prices, ownership, rules, events, chips (~2 MB, TTL 1h)."""
        return self.get_json("bootstrap-static/", cache_ttl_s=TTL_BOOTSTRAP)

    def fixtures(self) -> list[dict[str, Any]]:
        """``fixtures/`` — full-season schedule with per-fixture stats (TTL 6h)."""
        return self.get_json("fixtures/", cache_ttl_s=TTL_FIXTURES)

    def element_summary(self, element_id: int) -> dict[str, Any]:
        """``element-summary/{id}/`` — per-GW player history + remaining fixtures (TTL 12h)."""
        return self.get_json(f"element-summary/{element_id}/", cache_ttl_s=TTL_ELEMENT_SUMMARY)

    def event_live(self, gw: int) -> dict[str, Any]:
        """``event/{gw}/live/`` — live points and ``explain`` breakdown (TTL 60s)."""
        return self.get_json(f"event/{gw}/live/", cache_ttl_s=TTL_EVENT_LIVE)

    def event_status(self) -> dict[str, Any]:
        """``event-status/`` — bonus-added / league-update status for the current GW."""
        return self.get_json("event-status/", cache_ttl_s=TTL_EVENT_STATUS)

    def entry(self, entry_id: int) -> dict[str, Any]:
        """``entry/{id}/`` — public manager summary (TTL 5min)."""
        return self.get_json(f"entry/{entry_id}/", cache_ttl_s=TTL_ENTRY)

    def entry_history(self, entry_id: int) -> dict[str, Any]:
        """``entry/{id}/history/`` — per-GW and past-season history for a manager (TTL 5min)."""
        return self.get_json(f"entry/{entry_id}/history/", cache_ttl_s=TTL_ENTRY)

    def entry_picks(self, entry_id: int, gw: int) -> dict[str, Any]:
        """``entry/{id}/event/{gw}/picks/`` — a manager's squad for a GW (public post-deadline)."""
        return self.get_json(f"entry/{entry_id}/event/{gw}/picks/", cache_ttl_s=TTL_ENTRY)

    def entry_transfers(self, entry_id: int) -> list[dict[str, Any]]:
        """``entry/{id}/transfers/`` — a manager's full transfer log (TTL 5min)."""
        return self.get_json(f"entry/{entry_id}/transfers/", cache_ttl_s=TTL_ENTRY)

    def set_piece_notes(self) -> dict[str, Any]:
        """``team/set-piece-notes/`` — official per-team set-piece taker notes (TTL 6h)."""
        return self.get_json("team/set-piece-notes/", cache_ttl_s=TTL_SET_PIECE)

    # -- higher-level helpers ------------------------------------------------------

    def season_state(self) -> SeasonState:
        """Fetch ``bootstrap-static`` and parse the current :class:`SeasonState`."""
        return parse_season_state(self.bootstrap())

    def take_snapshot(self, *, force: bool = False, date: dt.date | None = None) -> Path:
        """Archive today's ``bootstrap.json`` + ``fixtures.json`` (append-only daily archive).

        Writes pretty-printed JSON to ``{raw_dir}/snapshots/{YYYY-MM-DD}/``, bypassing
        the HTTP cache so the archive is always a genuine live payload. Idempotent:
        if both files already exist for ``date`` the network is not touched unless
        ``force=True``.

        Args:
            force: Re-fetch and overwrite even if today's snapshot exists.
            date: Snapshot date (default: today, UTC).

        Returns:
            The snapshot directory path.
        """
        day = date if date is not None else dt.datetime.now(dt.UTC).date()
        snap_dir = self.raw_dir / "snapshots" / day.isoformat()
        bootstrap_path = snap_dir / "bootstrap.json"
        fixtures_path = snap_dir / "fixtures.json"
        if not force and bootstrap_path.exists() and fixtures_path.exists():
            logger.info("snapshot for %s already exists at %s — skipping", day, snap_dir)
            return snap_dir
        snap_dir.mkdir(parents=True, exist_ok=True)
        bootstrap = polite_get(
            self.base_url + "bootstrap-static/",
            min_interval_s=self.min_interval_s,
            cache_ttl_s=0,
            cache_dir=self.cache_dir,
            client=self._client,
        ).json()
        fixtures = polite_get(
            self.base_url + "fixtures/",
            min_interval_s=self.min_interval_s,
            cache_ttl_s=0,
            cache_dir=self.cache_dir,
            client=self._client,
        ).json()
        _atomic_write_json(bootstrap_path, bootstrap, pretty=True)
        _atomic_write_json(fixtures_path, fixtures, pretty=True)
        logger.info("snapshot written to %s", snap_dir)
        return snap_dir

    def element_summary_sweep(
        self,
        ids: Iterable[int],
        on_progress: Callable[[int, int], None] | None = None,
    ) -> list[Path]:
        """Nightly full sweep of ``element-summary/{id}/`` (~850 req at 1 req/s ≈ 15 min).

        Honors the client's per-host throttle and writes each payload to
        ``{raw_dir}/element_summary/{season}/{id}.json``.

        Args:
            ids: Element ids to sweep (typically every ``element.id`` in bootstrap).
            on_progress: Optional callback invoked after each element as
                ``on_progress(done, total)``.

        Returns:
            The list of file paths written, in input order.
        """
        id_list = [int(i) for i in ids]
        season = self.season_state().season
        out_dir = self.raw_dir / "element_summary" / str(season)
        out_dir.mkdir(parents=True, exist_ok=True)
        paths: list[Path] = []
        for done, element_id in enumerate(id_list, start=1):
            payload = self.element_summary(element_id)
            path = out_dir / f"{element_id}.json"
            _atomic_write_json(path, payload, pretty=False)
            paths.append(path)
            if on_progress is not None:
                on_progress(done, len(id_list))
        return paths
