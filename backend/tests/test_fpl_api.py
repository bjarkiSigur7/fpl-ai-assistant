"""Tests for fplai.data.fpl_api — offline via recorded trimmed fixtures; live tests marked."""

from __future__ import annotations

import copy
import datetime as dt
import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from fplai.data import fpl_api
from fplai.data.fpl_api import (
    FplApiClient,
    RetryableStatusError,
    parse_season_state,
    polite_get,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"
BOOTSTRAP: dict[str, Any] = json.loads((FIXTURES_DIR / "fpl_api_bootstrap.json").read_text())
FIXTURES_PAYLOAD: list[dict[str, Any]] = json.loads(
    (FIXTURES_DIR / "fpl_api_fixtures.json").read_text()
)
ENTRY_PICKS: dict[str, Any] = json.loads((FIXTURES_DIR / "fpl_api_entry_picks.json").read_text())


def _make_transport(counter: dict[str, int]) -> httpx.MockTransport:
    """MockTransport serving the recorded fixtures; counts hits per URL path."""

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        counter[path] = counter.get(path, 0) + 1
        if path.endswith("/bootstrap-static/"):
            return httpx.Response(200, json=BOOTSTRAP)
        if path.endswith("/fixtures/"):
            return httpx.Response(200, json=FIXTURES_PAYLOAD)
        if "/element-summary/" in path:
            element_id = int(path.rstrip("/").rsplit("/", 1)[-1])
            return httpx.Response(
                200, json={"id": element_id, "history": [], "fixtures": [], "history_past": []}
            )
        if path.endswith("/entry/1/event/38/picks/"):
            return httpx.Response(200, json=ENTRY_PICKS)
        return httpx.Response(404, json={"detail": "not found"})

    return httpx.MockTransport(handler)


@pytest.fixture()
def offline(tmp_path: Path) -> tuple[FplApiClient, dict[str, int]]:
    """An FplApiClient backed by the mock transport, isolated cache/raw dirs, no throttle."""
    counter: dict[str, int] = {}
    client = FplApiClient(
        min_interval_s=0,
        cache_dir=tmp_path / "cache",
        raw_dir=tmp_path / "raw",
        client=httpx.Client(transport=_make_transport(counter)),
    )
    return client, counter


# --- season_state / launch detector -------------------------------------------------


def test_parse_season_state_2025() -> None:
    state = parse_season_state(BOOTSTRAP)
    assert state.season == 2025
    assert state.is_live_2026_27 is False
    assert state.current_gw == 38  # finished 2025-26: event 38 is_current, nothing is_next
    assert state.next_gw is None
    assert state.next_deadline_utc is None
    assert state.total_players == 13_107_732
    assert "2025_26" in state.static_content_url


def test_parse_season_state_flips_to_2026_27() -> None:
    bootstrap = copy.deepcopy(BOOTSTRAP)
    settings = bootstrap["game_config"]["settings"]
    settings["static_content_url"] = settings["static_content_url"].replace("2025_26", "2026_27")
    for event in bootstrap["events"]:
        event["is_current"] = False
        event["is_next"] = event["id"] == 1
    state = parse_season_state(bootstrap)
    assert state.season == 2026
    assert state.is_live_2026_27 is True
    assert state.current_gw is None
    assert state.next_gw == 1
    assert state.next_deadline_utc is not None
    assert state.next_deadline_utc.tzinfo is not None
    # GW1 2025-26 deadline is 2025-08-15T17:30:00Z in the recorded fixture.
    assert state.next_deadline_utc == dt.datetime(2025, 8, 15, 17, 30, tzinfo=dt.UTC)


def test_parse_season_state_missing_url_raises() -> None:
    with pytest.raises(ValueError, match="static_content_url"):
        parse_season_state({"events": [], "total_players": 0})


def test_client_season_state(offline: tuple[FplApiClient, dict[str, int]]) -> None:
    client, _ = offline
    assert client.season_state().season == 2025


# --- polite_get: cache ----------------------------------------------------------------


def test_cache_roundtrip(tmp_path: Path) -> None:
    counter: dict[str, int] = {}
    http = httpx.Client(transport=_make_transport(counter))
    url = "https://fantasy.premierleague.com/api/bootstrap-static/"
    kwargs: dict[str, Any] = {"min_interval_s": 0, "cache_dir": tmp_path, "client": http}

    first = polite_get(url, cache_ttl_s=3600, **kwargs)
    assert first.status_code == 200
    assert counter["/api/bootstrap-static/"] == 1

    second = polite_get(url, cache_ttl_s=3600, **kwargs)
    assert counter["/api/bootstrap-static/"] == 1  # served from disk, no network
    assert second.headers.get("x-fplai-cache") == "hit"
    assert second.json() == first.json()


def test_cache_ttl_zero_bypasses(tmp_path: Path) -> None:
    counter: dict[str, int] = {}
    http = httpx.Client(transport=_make_transport(counter))
    url = "https://fantasy.premierleague.com/api/fixtures/"
    kwargs: dict[str, Any] = {"min_interval_s": 0, "cache_dir": tmp_path, "client": http}

    polite_get(url, cache_ttl_s=3600, **kwargs)
    polite_get(url, cache_ttl_s=0, **kwargs)  # ttl=0 must skip the cache read
    assert counter["/api/fixtures/"] == 2


def test_cache_expired_refetches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    counter: dict[str, int] = {}
    http = httpx.Client(transport=_make_transport(counter))
    url = "https://fantasy.premierleague.com/api/fixtures/"
    kwargs: dict[str, Any] = {"min_interval_s": 0, "cache_dir": tmp_path, "client": http}

    polite_get(url, cache_ttl_s=100, **kwargs)
    real_wall = fpl_api._wall
    monkeypatch.setattr(fpl_api, "_wall", lambda: real_wall() + 101)  # entry now stale
    polite_get(url, cache_ttl_s=100, **kwargs)
    assert counter["/api/fixtures/"] == 2


# --- polite_get: throttle (mocked clock) ---------------------------------------------


class _FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def test_throttle_enforces_min_interval_per_host(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = _FakeClock()
    monkeypatch.setattr(fpl_api, "_monotonic", clock.monotonic)
    monkeypatch.setattr(fpl_api, "_sleep", clock.sleep)
    monkeypatch.setattr(fpl_api, "_last_request_at", {})

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"ok": True})

    http = httpx.Client(transport=httpx.MockTransport(handler))
    kwargs: dict[str, Any] = {
        "min_interval_s": 1.0,
        "cache_ttl_s": 0,
        "cache_dir": tmp_path,
        "client": http,
    }

    polite_get("https://example.com/a", **kwargs)
    assert clock.sleeps == []  # first request to a host never waits

    polite_get("https://example.com/b", **kwargs)
    assert clock.sleeps == [pytest.approx(1.0)]  # same host, zero elapsed -> full interval

    polite_get("https://other.example.org/c", **kwargs)
    assert len(clock.sleeps) == 1  # different host is not throttled by example.com

    clock.now += 0.4  # 0.4s passes; 0.6s of the interval remains
    polite_get("https://example.com/d", **kwargs)
    assert clock.sleeps[-1] == pytest.approx(0.6)


# --- polite_get: retries --------------------------------------------------------------


def test_retry_recovers_from_5xx(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fpl_api, "_sleep", lambda _s: None)  # instant backoff
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] <= 2:
            return httpx.Response(503)
        return httpx.Response(200, json={"ok": True})

    http = httpx.Client(transport=httpx.MockTransport(handler))
    response = polite_get(
        "https://example.com/flaky", min_interval_s=0, cache_ttl_s=0,
        cache_dir=tmp_path, client=http,
    )
    assert response.status_code == 200
    assert calls["n"] == 3


def test_retry_exhausted_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fpl_api, "_sleep", lambda _s: None)
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(429)

    http = httpx.Client(transport=httpx.MockTransport(handler))
    with pytest.raises(RetryableStatusError):
        polite_get(
            "https://example.com/limited", min_interval_s=0, cache_ttl_s=0,
            cache_dir=tmp_path, client=http,
        )
    assert calls["n"] == fpl_api.RETRY_ATTEMPTS


def test_non_retryable_4xx_raises_immediately(tmp_path: Path) -> None:
    counter: dict[str, int] = {}
    http = httpx.Client(transport=_make_transport(counter))
    with pytest.raises(httpx.HTTPStatusError):
        polite_get(
            "https://fantasy.premierleague.com/api/nope/", min_interval_s=0, cache_ttl_s=0,
            cache_dir=tmp_path, client=http,
        )
    assert counter["/api/nope/"] == 1  # no retries on 404


# --- endpoints ------------------------------------------------------------------------


def test_bootstrap_uses_cache(offline: tuple[FplApiClient, dict[str, int]]) -> None:
    client, counter = offline
    first = client.bootstrap()
    second = client.bootstrap()
    assert counter["/api/bootstrap-static/"] == 1  # TTL 1h -> one network hit
    assert first == second == BOOTSTRAP


def test_entry_picks(offline: tuple[FplApiClient, dict[str, int]]) -> None:
    client, _ = offline
    picks = client.entry_picks(1, 38)
    assert len(picks["picks"]) == 15
    assert {"element", "position", "multiplier", "is_captain"} <= set(picks["picks"][0])


def test_fixtures_payload(offline: tuple[FplApiClient, dict[str, int]]) -> None:
    client, _ = offline
    fixtures = client.fixtures()
    assert {"event", "kickoff_time", "team_h", "team_a", "stats"} <= set(fixtures[0])


# --- snapshot archiver ----------------------------------------------------------------


def test_take_snapshot_writes_and_is_idempotent(
    offline: tuple[FplApiClient, dict[str, int]]
) -> None:
    client, counter = offline
    day = dt.date(2026, 7, 22)

    snap_dir = client.take_snapshot(date=day)
    assert snap_dir == client.raw_dir / "snapshots" / "2026-07-22"
    bootstrap_text = (snap_dir / "bootstrap.json").read_text()
    assert json.loads(bootstrap_text) == BOOTSTRAP
    assert "\n" in bootstrap_text  # pretty-printed
    assert json.loads((snap_dir / "fixtures.json").read_text()) == FIXTURES_PAYLOAD
    assert counter["/api/bootstrap-static/"] == 1

    client.take_snapshot(date=day)  # idempotent: files exist -> no network
    assert counter["/api/bootstrap-static/"] == 1
    assert counter["/api/fixtures/"] == 1

    client.take_snapshot(date=day, force=True)  # force -> refetch
    assert counter["/api/bootstrap-static/"] == 2
    assert counter["/api/fixtures/"] == 2


def test_take_snapshot_bypasses_http_cache(offline: tuple[FplApiClient, dict[str, int]]) -> None:
    client, counter = offline
    client.bootstrap()  # warms the HTTP cache
    client.take_snapshot(date=dt.date(2026, 7, 23))
    assert counter["/api/bootstrap-static/"] == 2  # snapshot fetched fresh, not from cache


# --- element summary sweep ------------------------------------------------------------


def test_element_summary_sweep(offline: tuple[FplApiClient, dict[str, int]]) -> None:
    client, counter = offline
    progress: list[tuple[int, int]] = []

    paths = client.element_summary_sweep([7, 8, 9], on_progress=lambda d, t: progress.append((d, t)))

    assert progress == [(1, 3), (2, 3), (3, 3)]
    assert [p.name for p in paths] == ["7.json", "8.json", "9.json"]
    assert paths[0].parent == client.raw_dir / "element_summary" / "2025"
    assert json.loads(paths[0].read_text())["id"] == 7
    assert counter["/api/element-summary/7/"] == 1


# --- live tests (run explicitly with -m live) ----------------------------------------


@pytest.mark.live
def test_live_bootstrap_and_season_state(tmp_path: Path) -> None:
    client = FplApiClient(cache_dir=tmp_path / "cache", raw_dir=tmp_path / "raw")
    bootstrap = client.bootstrap()
    assert {"events", "teams", "elements", "game_settings", "game_config"} <= set(bootstrap)
    state = parse_season_state(bootstrap)
    assert state.season in (2025, 2026)
    assert state.total_players > 1_000_000
    if state.is_live_2026_27:
        assert state.next_deadline_utc is None or state.next_deadline_utc.year == 2026


@pytest.mark.live
def test_live_entry_picks(tmp_path: Path) -> None:
    client = FplApiClient(cache_dir=tmp_path / "cache", raw_dir=tmp_path / "raw")
    picks = client.entry_picks(1, 38)
    assert len(picks["picks"]) == 15
