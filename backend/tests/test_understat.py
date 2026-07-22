"""Tests for fplai.data.understat.

Offline tests run from trimmed real payloads recorded 2026-07-22
(tests/fixtures/understat_league.json, tests/fixtures/understat_player.json).
Live tests are marked ``live`` and excluded by default (``-m "not live"``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from fplai.data.understat import (
    PLAYER_MATCH_COLUMNS,
    UnderstatClient,
    season_dir,
    to_player_match_frame,
)

FIXTURES = Path(__file__).parent / "fixtures"
LEAGUE_FIXTURE = FIXTURES / "understat_league.json"
PLAYER_FIXTURE = FIXTURES / "understat_player.json"
HAALAND_ID = 8260
SEASON = 2025


class FakeFetch:
    """URL -> fixture payload, counting calls (network stand-in)."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def __call__(self, url: str) -> Any:
        self.calls.append(url)
        if "getLeagueData/EPL/" in url:
            return json.loads(LEAGUE_FIXTURE.read_text())
        if f"getPlayerData/{HAALAND_ID}/" in url:
            # Exercise the str-payload normalisation path too.
            return PLAYER_FIXTURE.read_text()
        raise AssertionError(f"unexpected url {url}")


@pytest.fixture()
def client(tmp_path: Path) -> UnderstatClient:
    return UnderstatClient(raw_dir=tmp_path, fetch=FakeFetch())


def test_league_data_shape(client: UnderstatClient) -> None:
    data = client.league_data(SEASON)
    assert set(data) >= {"players", "teams", "dates"}
    assert isinstance(data["players"], list) and data["players"]
    first = data["players"][0]
    assert {"id", "player_name", "xG", "xA", "npxG", "shots", "key_passes"} <= set(first)
    # teams is a dict keyed by team-id string, each with a per-match history
    team = next(iter(data["teams"].values()))
    assert {"id", "title", "history"} <= set(team)
    assert {"xG", "xGA", "npxG", "date"} <= set(team["history"][0])
    # dates carry the season's fixtures with match ids
    assert {"id", "h", "a", "datetime"} <= set(data["dates"][0])


def test_player_data_shape(client: UnderstatClient) -> None:
    data = client.player_data(HAALAND_ID)
    assert int(data["player"]["id"]) == HAALAND_ID
    match = data["matches"][0]
    assert {"id", "season", "date", "h_team", "a_team", "xG", "xA", "npxG"} <= set(match)
    assert {"shots", "key_passes", "time", "goals", "assists"} <= set(match)


def test_fetch_league_season_writes_and_skips(client: UnderstatClient) -> None:
    path = client.fetch_league_season(SEASON)
    assert path == season_dir(SEASON, client.raw_dir) / "league.json"
    assert json.loads(path.read_text())["players"]
    n_calls = len(client._fetch.calls)  # type: ignore[attr-defined]
    assert client.fetch_league_season(SEASON) == path  # skip-if-exists
    assert len(client._fetch.calls) == n_calls  # type: ignore[attr-defined]
    client.fetch_league_season(SEASON, force=True)  # force re-fetches
    assert len(client._fetch.calls) == n_calls + 1  # type: ignore[attr-defined]


def test_fetch_players_writes_and_skips(client: UnderstatClient) -> None:
    paths = client.fetch_players([HAALAND_ID], SEASON)
    assert paths == [season_dir(SEASON, client.raw_dir) / "players" / f"{HAALAND_ID}.json"]
    assert paths[0].exists()
    n_calls = len(client._fetch.calls)  # type: ignore[attr-defined]
    assert client.fetch_players([HAALAND_ID], SEASON) == paths  # skip-if-exists
    assert len(client._fetch.calls) == n_calls  # type: ignore[attr-defined]


def test_to_player_match_frame_schema_and_filtering(client: UnderstatClient) -> None:
    client.fetch_league_season(SEASON)
    client.fetch_players([HAALAND_ID], SEASON)
    frame = to_player_match_frame(SEASON, raw_dir=client.raw_dir)

    assert list(frame.columns) == list(PLAYER_MATCH_COLUMNS)
    for col, dtype in PLAYER_MATCH_COLUMNS.items():
        assert str(frame[col].dtype) == dtype, f"{col}: {frame[col].dtype} != {dtype}"

    # Fixture has 3 matches: 2 EPL 2025-26 + 1 from another season, which must
    # be dropped because its match id is absent from the league's dates.
    assert len(frame) == 2
    assert (frame["understat_id"] == HAALAND_ID).all()
    assert (frame["season"] == SEASON).all()
    # values converted from Understat's string encoding
    assert frame["us_xg"].between(0, 5).all()
    assert frame["minutes"].between(0, 120).all()
    assert frame["date"].dt.year.isin([2025, 2026]).all()
    row = frame.loc[frame["match_id"] == 29138].iloc[0]
    assert row["h_team"] == "Bournemouth" and row["a_team"] == "Manchester City"
    assert row["us_shots"] == 2 and row["us_key_passes"] == 1 and row["minutes"] == 90

    # frame is persisted next to the raw JSON (never into processed/)
    parquet = season_dir(SEASON, client.raw_dir) / "player_matches.parquet"
    assert parquet.exists()
    pd.testing.assert_frame_equal(pd.read_parquet(parquet), frame)


def test_to_player_match_frame_missing_league_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        to_player_match_frame(SEASON, raw_dir=tmp_path)


def test_to_player_match_frame_no_players_is_empty_with_schema(
    client: UnderstatClient,
) -> None:
    client.fetch_league_season(SEASON)
    frame = to_player_match_frame(SEASON, raw_dir=client.raw_dir)
    assert frame.empty
    assert list(frame.columns) == list(PLAYER_MATCH_COLUMNS)


# -- live tests (excluded by default; run with `-m live`) ---------------------


@pytest.mark.live
def test_live_league_data_most_recent_completed_season(tmp_path: Path) -> None:
    """2025 (== 2025-26, completed May 2026) must return a full player list."""
    client = UnderstatClient(raw_dir=tmp_path)
    data = client.league_data(2025)
    assert len(data["players"]) > 400
    assert len(data["dates"]) == 380
    assert data["dates"][0]["datetime"].startswith("2025-08")  # year == season start year


@pytest.mark.live
def test_live_player_data_star_player(tmp_path: Path) -> None:
    client = UnderstatClient(raw_dir=tmp_path)
    data = client.player_data(HAALAND_ID)
    assert data["player"]["name"] == "Erling Haaland"
    epl_2025 = [m for m in data["matches"] if m["season"] == "2025"]
    assert len(epl_2025) >= 30
    assert float(epl_2025[0]["xG"]) >= 0.0
