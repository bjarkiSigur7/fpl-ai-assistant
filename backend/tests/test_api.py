"""Offline tests for the FastAPI layer (fplai.api).

Every endpoint is exercised with FastAPI's TestClient against fixture parquets/JSONs in
a tmp path (``fplai.config`` paths monkeypatched): 200 paths, 404 paths, the refresh
single-flight 409 and the my-team 202 background-solve flow.  The expensive seams
(``fplai.api.jobs.run_refresh`` / ``fplai.api.jobs.solve_my_team``) are monkeypatched
with deterministic fakes — no network, no MILP.
"""

from __future__ import annotations

import json
import os
import threading
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from fplai import config
from fplai.api import jobs
from fplai.api.app import GW1_DEADLINE_2026_27, app
from fplai.api.cache import cache as file_cache
from fplai.optimizer.plans import Recommendation
from fplai.optimizer.state import OwnedPlayer, SquadState

SEASON = 2025
GW = 34

XP_COMPONENTS = (
    "xp_appearance",
    "xp_goals",
    "xp_assists",
    "xp_cs",
    "xp_concede",
    "xp_saves",
    "xp_defcon",
    "xp_bonus",
    "xp_cards",
    "xp_other",
)

#: (player_code, web_name, position, team_code, price, xp, q0, n_fixtures_gw34)
PLAYERS = [
    (1001, "Salah", "MID", 1, 130, 8.5, 0.05, 1),
    (1002, "Haaland", "FWD", 2, 145, 7.9, 0.04, 1),
    (1003, "Gabriel", "DEF", 3, 62, 5.2, 0.10, 1),
    (1004, "Raya", "GKP", 3, 55, 4.0, 0.02, 1),
    (1005, "Bowen", "MID", 2, 78, 6.1, 0.06, 2),  # DGW in GW34
    (1006, None, "DEF", 1, 45, 2.2, 0.30, 1),  # web_name joined from players.parquet
]


# --------------------------------------------------------------------------------------
# Fixture data builders
# --------------------------------------------------------------------------------------


def _with_components(row: dict[str, Any], xp: float) -> dict[str, Any]:
    row["xp"] = xp
    for i, comp in enumerate(XP_COMPONENTS):
        row[comp] = round(xp / 20 + i * 0.01, 4)
    return row


def make_predictions_gw() -> pd.DataFrame:
    rows = []
    for code, name, pos, team, price, xp, q0, n_fx in PLAYERS:
        rows.append(
            _with_components(
                {
                    "season": SEASON,
                    "gw": GW,
                    "player_code": code,
                    "n_fixtures": n_fx,
                    "q0": q0,
                    "team_code": team,
                    "position": pos,
                    "price": price,
                    "web_name": name,
                },
                xp,
            )
        )
    for code, name, pos, team, price, xp, q0, _ in PLAYERS[:2]:  # GW35: subset
        rows.append(
            _with_components(
                {
                    "season": SEASON,
                    "gw": GW + 1,
                    "player_code": code,
                    "n_fixtures": 1,
                    "q0": q0,
                    "team_code": team,
                    "position": pos,
                    "price": price,
                    "web_name": name,
                },
                xp - 0.5,
            )
        )
    df = pd.DataFrame(rows)
    df["web_name"] = df["web_name"].astype("string")
    return df


def make_predictions() -> pd.DataFrame:
    rows = []
    fixture_id = 340
    for code, _name, pos, team, price, xp, q0, n_fx in PLAYERS:
        for leg in range(n_fx):
            rows.append(
                _with_components(
                    {
                        "season": SEASON,
                        "gw": GW,
                        "player_code": code,
                        "fpl_fixture_id": fixture_id,
                        "team_code": team,
                        "opponent_code": (team % 3) + 1,
                        "was_home": leg == 0,
                        "position": pos,
                        "price": price,
                        "q0": q0,
                        "q1": 0.15,
                        "q2": round(1.0 - q0 - 0.15, 4),
                        "mu1": 25.0,
                        "mu2": 88.0,
                    },
                    xp / n_fx,
                )
            )
            fixture_id += 1
    for code, _name, pos, team, price, xp, q0, _ in PLAYERS[:2]:  # GW35
        rows.append(
            _with_components(
                {
                    "season": SEASON,
                    "gw": GW + 1,
                    "player_code": code,
                    "fpl_fixture_id": fixture_id,
                    "team_code": team,
                    "opponent_code": (team % 3) + 1,
                    "was_home": True,
                    "position": pos,
                    "price": price,
                    "q0": q0,
                    "q1": 0.15,
                    "q2": round(1.0 - q0 - 0.15, 4),
                    "mu1": 25.0,
                    "mu2": 88.0,
                },
                xp - 0.5,
            )
        )
        fixture_id += 1
    return pd.DataFrame(rows)


def make_players() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "player_code": [c for c, *_ in PLAYERS],
            "web_name": ["Salah", "Haaland", "Gabriel", "Raya", "Bowen", "Fallback"],
            "first_name": ["Mo", "Erling", "G.", "David", "Jarrod", "No"],
            "second_name": ["Salah", "Haaland", "Magalhaes", "Raya", "Bowen", "Name"],
            "understat_id": pd.array([1250, 8260, None, None, 7430, None], dtype="Int64"),
            "opta_code": pd.array(["p1", "p2", None, "p4", "p5", None], dtype="string"),
        }
    )


def make_teams() -> pd.DataFrame:
    rows = []
    for season in (SEASON - 1, SEASON):
        for tid, (code, name, short) in enumerate(
            [(1, "Liverpool", "LIV"), (2, "West Ham", "WHU"), (3, "Arsenal", "ARS")], start=1
        ):
            rows.append(
                {
                    "season": season,
                    "fpl_team_id": tid,
                    "team_code": code,
                    "name": name,
                    "short_name": short,
                    "understat_name": name,
                    "clubelo_name": name,
                    "footballdata_name": name,
                }
            )
    return pd.DataFrame(rows)


def make_player_gw() -> pd.DataFrame:
    rows = []
    for season, gws in ((SEASON - 1, (1, 2)), (SEASON, (1, 2, 3))):
        for gw in gws:
            rows.append(
                {
                    "season": season,
                    "gw": gw,
                    "player_code": 1001,
                    "fpl_element_id": 300,
                    "team_code": 1,
                    "position": "MID",
                    "n_fixtures": 1,
                    "minutes": 90,
                    "total_points": 8 + gw,
                    "goals_scored": 1,
                    "assists": 0,
                    "clean_sheets": 0,
                    "goals_conceded": 1,
                    "saves": 0,
                    "penalties_saved": 0,
                    "penalties_missed": 0,
                    "yellow_cards": 0,
                    "red_cards": 0,
                    "own_goals": 0,
                    "bonus": 2,
                    "bps": 30,
                    "starts": 1 if gw != 2 else None,
                    "defensive_contribution": None,
                    "tackles": None,
                    "recoveries": None,
                    "clearances_blocks_interceptions": None,
                    "xg": 0.6,
                    "xa": 0.2,
                    "xgc": 1.1,
                    "value": 128 + gw,
                    "selected_by_percent": None,
                }
            )
    df = pd.DataFrame(rows)
    for col in ("starts", "defensive_contribution", "tackles", "recoveries",
                "clearances_blocks_interceptions"):
        df[col] = pd.array(df[col], dtype="Int64")
    df["selected_by_percent"] = pd.array(df["selected_by_percent"], dtype="Float64")
    return df


def make_dream_team_dict() -> dict[str, Any]:
    squad = list(range(1001, 1016))
    return {
        "season": SEASON,
        "gw": GW,
        "squad": squad,
        "lineup": squad[:11],
        "bench_order": squad[11:],
        "captain": 1001,
        "vice": 1002,
        "formation": "3-4-3",
        "expected_points": 58.3,
        "objective": 61.0,
        "total_cost": 998,
        "solve_seconds": 1.5,
    }


def make_recommendation_dict(gw: int = GW) -> dict[str, Any]:
    rec = Recommendation(
        season=SEASON,
        gw=gw,
        as_of=datetime(2026, 7, 22, 12, 0, tzinfo=UTC),
        action="hold",
        captain=1001,
        vice=1002,
        lineup=list(range(1001, 1012)),
        bench_order=list(range(1012, 1016)),
        squad=list(range(1001, 1016)),
        expected_points=54.2,
        objective=310.5,
        rationale=["Hold: the optimal plan makes 0 transfers this GW."],
    )
    return rec.model_dump(mode="json")


def make_squad_state_dict() -> dict[str, Any]:
    squad = [
        OwnedPlayer(player_code=c, purchase_price=50, current_price=52)
        for c in range(1001, 1016)
    ]
    state = SquadState(
        season=SEASON,
        current_gw=GW,
        squad=squad,
        bank=15,
        free_transfers=2,
        chips_available=["wc2", "bb1", "tc1"],
    )
    return state.model_dump(mode="json")


def make_bootstrap(season: int = SEASON, *, live: bool = False) -> dict[str, Any]:
    url = f"https://fantasy.premierleague.com/static/{season}_{(season + 1) % 100:02d}/"
    events: list[dict[str, Any]] = [{"id": 38, "finished": True, "is_current": True}]
    if live:
        events = [
            {
                "id": 1,
                "finished": False,
                "is_next": True,
                "deadline_time": "2026-08-21T17:30:00Z",
            }
        ]
    return {
        "game_config": {"settings": {"static_content_url": url}},
        "events": events,
        "total_players": 11_000_000,
    }


# --------------------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------------------


@pytest.fixture()
def api_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Tmp data dirs with fixture artifacts; config paths monkeypatched; caches reset."""
    processed = tmp_path / "processed"
    models = tmp_path / "models"
    raw = tmp_path / "raw"
    cache_dir = tmp_path / "cache"
    for d in (processed, models, raw, cache_dir):
        d.mkdir(parents=True)
    monkeypatch.setattr(config, "PROCESSED_DIR", processed)
    monkeypatch.setattr(config, "MODELS_DIR", models)
    monkeypatch.setattr(config, "RAW_DIR", raw)
    monkeypatch.setattr(config, "CACHE_DIR", cache_dir)
    file_cache.clear()
    jobs.registry.reset()
    jobs.refresh_log.clear()

    make_predictions_gw().to_parquet(processed / "predictions_gw.parquet", index=False)
    make_predictions().to_parquet(processed / "predictions.parquet", index=False)
    make_players().to_parquet(processed / "players.parquet", index=False)
    make_teams().to_parquet(processed / "teams.parquet", index=False)
    make_player_gw().to_parquet(processed / "player_gw.parquet", index=False)

    (models / "manifest.json").write_text(
        json.dumps({"schema": 1, "created_utc": "2026-07-22T15:02:36+00:00", "components": {}})
    )
    snap = raw / "fpl_api" / "snapshots" / "2026-07-21"
    snap.mkdir(parents=True)
    (snap / "bootstrap.json").write_text(json.dumps(make_bootstrap()))
    (snap / "fixtures.json").write_text("[]")

    yield SimpleNamespace(
        processed=processed, models=models, raw=raw, cache_dir=cache_dir, snap=snap
    )
    file_cache.clear()
    jobs.registry.reset()


@pytest.fixture()
def client(api_env: SimpleNamespace) -> TestClient:
    return TestClient(app)


def _wait_job(name: str, timeout: float = 10.0) -> jobs.Job:
    job = jobs.registry.get(name)
    assert job is not None and job.thread is not None
    job.thread.join(timeout=timeout)
    assert job.state in ("done", "error"), f"job {name} still {job.state}"
    return job


# --------------------------------------------------------------------------------------
# /health and /state
# --------------------------------------------------------------------------------------


class TestHealthAndState:
    def test_health(self, client: TestClient) -> None:
        resp = client.get("/api/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["version"]
        assert body["time_utc"]

    def test_state_pre_launch(self, client: TestClient) -> None:
        resp = client.get("/api/state")
        assert resp.status_code == 200
        body = resp.json()
        assert body["pre_launch"] is True
        ss = body["season_state"]
        assert ss["season"] == SEASON
        assert ss["is_live_2026_27"] is False
        assert ss["next_gw"] is None
        assert body["gw1_deadline_utc"] == GW1_DEADLINE_2026_27.isoformat().replace(
            "+00:00", "Z"
        )
        assert body["seconds_to_gw1_deadline"] is not None
        assert body["days_to_gw1_deadline"] is not None
        fresh = body["freshness"]
        assert fresh["latest_snapshot"]["date"] == "2026-07-21"
        assert "predictions_gw.parquet" in fresh["processed"]
        assert fresh["models_trained_utc"] is not None
        assert body["model_manifest"]["schema"] == 1

    def test_state_live_mode(self, api_env: SimpleNamespace, client: TestClient) -> None:
        live_snap = api_env.raw / "fpl_api" / "snapshots" / "2026-08-01"
        live_snap.mkdir(parents=True)
        (live_snap / "bootstrap.json").write_text(
            json.dumps(make_bootstrap(2026, live=True))
        )
        body = client.get("/api/state").json()
        assert body["pre_launch"] is False
        assert body["season_state"]["is_live_2026_27"] is True
        assert body["season_state"]["next_gw"] == 1
        assert body["gw1_deadline_utc"] == "2026-08-21T17:30:00Z"

    def test_state_without_snapshot(
        self, api_env: SimpleNamespace, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(config, "RAW_DIR", api_env.raw / "empty")
        body = client.get("/api/state").json()
        assert body["season_state"] is None
        assert body["pre_launch"] is True
        assert body["freshness"]["latest_snapshot"] is None


# --------------------------------------------------------------------------------------
# /predictions
# --------------------------------------------------------------------------------------


class TestPredictions:
    def test_defaults_latest_season_earliest_gw(self, client: TestClient) -> None:
        body = client.get("/api/predictions").json()
        assert (body["season"], body["gw"]) == (SEASON, GW)
        assert body["total"] == len(PLAYERS)
        xps = [p["xp"] for p in body["players"]]
        assert xps == sorted(xps, reverse=True)  # default sort: xp desc
        top = body["players"][0]
        assert top["web_name"] == "Salah"
        assert top["team_name"] == "Liverpool"
        assert top["team_short_name"] == "LIV"
        for comp in XP_COMPONENTS:
            assert comp in top
        assert top["q0"] == pytest.approx(0.05)

    def test_web_name_joined_from_players_parquet(self, client: TestClient) -> None:
        body = client.get("/api/predictions?season=2025&gw=34&limit=100").json()
        by_code = {p["player_code"]: p for p in body["players"]}
        assert by_code[1006]["web_name"] == "Fallback"

    def test_fixture_breakdown_and_dgw(self, client: TestClient) -> None:
        body = client.get("/api/predictions?season=2025&gw=34&limit=100").json()
        by_code = {p["player_code"]: p for p in body["players"]}
        dgw = by_code[1005]
        assert dgw["n_fixtures"] == 2
        assert len(dgw["fixtures"]) == 2
        fx = dgw["fixtures"][0]
        assert fx["q1"] == pytest.approx(0.15)
        assert fx["mu2"] == pytest.approx(88.0)
        assert fx["opponent_short_name"] == "ARS"  # team 2's opponent is team 3
        assert len(by_code[1001]["fixtures"]) == 1

    def test_position_filter_and_sort(self, client: TestClient) -> None:
        body = client.get("/api/predictions?position=MID&sort=price&order=asc").json()
        assert body["total"] == 2
        assert [p["player_code"] for p in body["players"]] == [1005, 1001]
        assert all(p["position"] == "MID" for p in body["players"])

    def test_pagination(self, client: TestClient) -> None:
        page1 = client.get("/api/predictions?limit=2&offset=0").json()
        page2 = client.get("/api/predictions?limit=2&offset=2").json()
        assert page1["total"] == page2["total"] == len(PLAYERS)
        assert len(page1["players"]) == 2
        codes1 = {p["player_code"] for p in page1["players"]}
        codes2 = {p["player_code"] for p in page2["players"]}
        assert not codes1 & codes2

    def test_unknown_gw_404(self, client: TestClient) -> None:
        resp = client.get("/api/predictions?season=2025&gw=9")
        assert resp.status_code == 404
        assert "available GWs" in resp.json()["detail"]

    def test_unknown_season_404(self, client: TestClient) -> None:
        resp = client.get("/api/predictions?season=2019")
        assert resp.status_code == 404

    def test_missing_file_404(
        self, api_env: SimpleNamespace, client: TestClient
    ) -> None:
        (api_env.processed / "predictions_gw.parquet").unlink()
        resp = client.get("/api/predictions")
        assert resp.status_code == 404
        assert "fplai predict" in resp.json()["detail"]

    def test_mtime_reload(self, api_env: SimpleNamespace, client: TestClient) -> None:
        """The parquet cache reloads when the file changes under the API's feet."""
        assert client.get("/api/predictions").json()["total"] == len(PLAYERS)
        df = make_predictions_gw()
        df = df[df["player_code"] != 1006]
        path = api_env.processed / "predictions_gw.parquet"
        df.to_parquet(path, index=False)
        os.utime(path, (time.time() + 5, time.time() + 5))  # force a new mtime
        assert client.get("/api/predictions").json()["total"] == len(PLAYERS) - 1


# --------------------------------------------------------------------------------------
# /dream-team, /recommendation, /chip-curves
# --------------------------------------------------------------------------------------


class TestArtifactEndpoints:
    def test_dream_team_ok(self, api_env: SimpleNamespace, client: TestClient) -> None:
        (api_env.processed / "dream_team.json").write_text(json.dumps(make_dream_team_dict()))
        body = client.get("/api/dream-team?season=2025&gw=34").json()
        assert body["season"] == SEASON
        assert body["gw"] == GW
        assert len(body["squad"]) == 15
        assert body["captain"] == 1001
        assert body["formation"] == "3-4-3"

    def test_dream_team_wrong_gw_404(
        self, api_env: SimpleNamespace, client: TestClient
    ) -> None:
        (api_env.processed / "dream_team.json").write_text(json.dumps(make_dream_team_dict()))
        resp = client.get("/api/dream-team?gw=35")
        assert resp.status_code == 404
        assert "GW34" in resp.json()["detail"]

    def test_dream_team_missing_404(self, client: TestClient) -> None:
        resp = client.get("/api/dream-team")
        assert resp.status_code == 404
        assert "dream_team.json not found" in resp.json()["detail"]

    def test_dream_team_stale_404(
        self, api_env: SimpleNamespace, client: TestClient
    ) -> None:
        dream_path = api_env.processed / "dream_team.json"
        dream_path.write_text(json.dumps(make_dream_team_dict()))
        old = time.time() - 3600
        os.utime(dream_path, (old, old))  # predictions_gw.parquet is now newer
        resp = client.get("/api/dream-team")
        assert resp.status_code == 404
        assert "stale" in resp.json()["detail"]

    def test_recommendation_ok(self, api_env: SimpleNamespace, client: TestClient) -> None:
        (api_env.processed / "recommendation.json").write_text(
            json.dumps(make_recommendation_dict())
        )
        body = client.get("/api/recommendation").json()
        assert body["action"] == "hold"
        assert body["season"] == SEASON
        assert body["captain"] == 1001
        assert Recommendation.model_validate(body).gw == GW  # round-trips the contract

    def test_recommendation_missing_404(self, client: TestClient) -> None:
        resp = client.get("/api/recommendation")
        assert resp.status_code == 404
        assert "recommendation.json not found" in resp.json()["detail"]

    def test_chip_curves_ok(self, api_env: SimpleNamespace, client: TestClient) -> None:
        pd.DataFrame(
            {
                "chip": ["bb1", "bb1", "tc1"],
                "gw": [34, 35, 34],
                "objective": [310.0, 312.5, 309.0],
                "delta_vs_no_chip": [1.2, 3.7, 0.2],
            }
        ).to_parquet(api_env.processed / "chip_curves.parquet", index=False)
        body = client.get("/api/chip-curves").json()
        assert body["generated_at"]
        assert len(body["curves"]) == 3
        assert body["curves"][1] == {
            "chip": "bb1",
            "gw": 35,
            "objective": 312.5,
            "delta_vs_no_chip": 3.7,
            # optional passthrough columns (absent from this minimal fixture file)
            "evaluated": None,
            "skip_reason": None,
            "window_last_gw": None,
            "urgency": None,
        }

    def test_chip_curves_missing_404(self, client: TestClient) -> None:
        resp = client.get("/api/chip-curves")
        assert resp.status_code == 404
        assert "chip_curves.parquet" in resp.json()["detail"]


# --------------------------------------------------------------------------------------
# /players/{player_code}
# --------------------------------------------------------------------------------------


class TestPlayerDetail:
    def test_identity_history_upcoming(self, client: TestClient) -> None:
        body = client.get("/api/players/1001").json()
        assert body["player"]["web_name"] == "Salah"
        assert body["player"]["understat_id"] == 1250
        assert body["seasons"] == [SEASON - 1, SEASON]
        assert body["season"] == SEASON  # defaults to the latest season
        assert body["team_name"] == "Liverpool"
        assert body["position"] == "MID"
        assert [h["gw"] for h in body["history"]] == [1, 2, 3]
        assert body["history"][0]["total_points"] == 9
        assert body["history"][1]["starts"] is None  # nullable Int64 survives
        assert len(body["upcoming"]) == 2  # GW34 + GW35 fixture rows
        assert body["upcoming"][0]["gw"] == GW
        assert body["upcoming"][0]["xp_goals"] is not None

    def test_explicit_season(self, client: TestClient) -> None:
        body = client.get(f"/api/players/1001?season={SEASON - 1}").json()
        assert body["season"] == SEASON - 1
        assert [h["gw"] for h in body["history"]] == [1, 2]

    def test_season_without_history_404(self, client: TestClient) -> None:
        resp = client.get("/api/players/1001?season=2020")
        assert resp.status_code == 404

    def test_unknown_player_404(self, client: TestClient) -> None:
        resp = client.get("/api/players/99999")
        assert resp.status_code == 404
        assert "99999" in resp.json()["detail"]


# --------------------------------------------------------------------------------------
# /refresh (single-flight + log tail)
# --------------------------------------------------------------------------------------


class TestRefresh:
    def test_single_flight_409_and_log_tail(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        release = threading.Event()

        def fake_refresh() -> None:
            print("snapshot: archived")
            print("build: 5 tables")
            assert release.wait(timeout=10)

        monkeypatch.setattr(jobs, "run_refresh", fake_refresh)

        resp = client.post("/api/refresh")
        assert resp.status_code == 202
        assert resp.json()["state"] == "running"

        dup = client.post("/api/refresh")
        assert dup.status_code == 409
        assert "already running" in dup.json()["detail"]

        deadline = time.time() + 5  # wait for the worker to emit both lines
        while time.time() < deadline:
            status = client.get("/api/refresh/status").json()
            if len(status["log_tail"]) >= 2:
                break
            time.sleep(0.02)
        assert status["state"] == "running"
        assert "snapshot: archived" in status["log_tail"]
        assert "build: 5 tables" in status["log_tail"]

        release.set()
        _wait_job(jobs.REFRESH_JOB)
        done = client.get("/api/refresh/status").json()
        assert done["state"] == "done"
        assert done["error"] is None
        assert done["finished_at"] is not None

    def test_refresh_error_reported(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def boom() -> None:
            raise RuntimeError("upstream exploded")

        monkeypatch.setattr(jobs, "run_refresh", boom)
        assert client.post("/api/refresh").status_code == 202
        _wait_job(jobs.REFRESH_JOB)
        status = client.get("/api/refresh/status").json()
        assert status["state"] == "error"
        assert "upstream exploded" in status["error"]

    def test_refresh_restarts_after_done(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(jobs, "run_refresh", lambda: print("quick run"))
        assert client.post("/api/refresh").status_code == 202
        _wait_job(jobs.REFRESH_JOB)
        assert client.post("/api/refresh").status_code == 202  # not 409 once finished
        _wait_job(jobs.REFRESH_JOB)

    def test_status_idle_before_any_run(self, client: TestClient) -> None:
        status = client.get("/api/refresh/status").json()
        assert status["state"] == "idle"
        assert status["started_at"] is None


# --------------------------------------------------------------------------------------
# /my-team/{entry_id} (202 background-solve flow; solver monkeypatched)
# --------------------------------------------------------------------------------------


def _fake_payload(entry_id: int, generated_at: datetime | None = None) -> dict[str, Any]:
    ts = generated_at if generated_at is not None else datetime.now(UTC)
    return {
        "entry_id": entry_id,
        "generated_at": ts.isoformat(),
        "squad_state": make_squad_state_dict(),
        "recommendation": make_recommendation_dict(),
    }


class TestMyTeam:
    ENTRY = 777

    def test_202_then_200_flow(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        release = threading.Event()
        calls: list[int] = []

        def fake_solve(entry_id: int) -> dict[str, Any]:
            calls.append(entry_id)
            assert release.wait(timeout=10)
            return _fake_payload(entry_id)

        monkeypatch.setattr(jobs, "solve_my_team", fake_solve)

        t0 = time.monotonic()
        resp = client.get(f"/api/my-team/{self.ENTRY}")
        assert time.monotonic() - t0 < 5.0  # never blocks on the solver
        assert resp.status_code == 202
        body = resp.json()
        assert body["state"] == "running"
        assert body["status_url"] == f"/api/my-team/{self.ENTRY}/status"

        again = client.get(f"/api/my-team/{self.ENTRY}")
        assert again.status_code == 202  # single-flight: no second solve kicked
        status = client.get(f"/api/my-team/{self.ENTRY}/status").json()
        assert status["state"] == "running"

        release.set()
        _wait_job(f"my-team:{self.ENTRY}")
        assert calls == [self.ENTRY]

        status = client.get(f"/api/my-team/{self.ENTRY}/status").json()
        assert status["state"] == "done"

        final = client.get(f"/api/my-team/{self.ENTRY}")
        assert final.status_code == 200
        body = final.json()
        assert body["entry_id"] == self.ENTRY
        assert body["source"] == "memory"
        assert body["recommendation"]["action"] == "hold"
        assert body["squad_state"]["current_gw"] == GW
        assert len(body["squad_state"]["squad"]) == 15

    def test_solver_error_then_retry(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def boom(entry_id: int) -> dict[str, Any]:
            raise RuntimeError("no predictions on disk")

        monkeypatch.setattr(jobs, "solve_my_team", boom)
        assert client.get(f"/api/my-team/{self.ENTRY}").status_code == 202
        _wait_job(f"my-team:{self.ENTRY}")
        status = client.get(f"/api/my-team/{self.ENTRY}/status").json()
        assert status["state"] == "error"
        assert "no predictions on disk" in status["error"]

        monkeypatch.setattr(jobs, "solve_my_team", _fake_payload)
        retry = client.get(f"/api/my-team/{self.ENTRY}")
        assert retry.status_code == 202
        assert "previous solve failed" in retry.json()["detail"]
        _wait_job(f"my-team:{self.ENTRY}")
        assert client.get(f"/api/my-team/{self.ENTRY}").status_code == 200

    def test_fresh_cache_file_served_without_solve(
        self, api_env: SimpleNamespace, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def must_not_run(entry_id: int) -> dict[str, Any]:  # pragma: no cover
            raise AssertionError("solver must not be called when the cache is fresh")

        monkeypatch.setattr(jobs, "solve_my_team", must_not_run)
        cache_path = api_env.cache_dir / "api" / f"my_team_{self.ENTRY}.json"
        cache_path.parent.mkdir(parents=True)
        cache_path.write_text(json.dumps(_fake_payload(self.ENTRY)))

        resp = client.get(f"/api/my-team/{self.ENTRY}")
        assert resp.status_code == 200
        assert resp.json()["source"] == "cache"
        assert jobs.registry.get(f"my-team:{self.ENTRY}") is None

    def test_stale_cache_kicks_background_solve(
        self, api_env: SimpleNamespace, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        stale = datetime.now(UTC) - timedelta(hours=7)
        cache_path = api_env.cache_dir / "api" / f"my_team_{self.ENTRY}.json"
        cache_path.parent.mkdir(parents=True)
        cache_path.write_text(json.dumps(_fake_payload(self.ENTRY, generated_at=stale)))
        monkeypatch.setattr(jobs, "solve_my_team", _fake_payload)

        resp = client.get(f"/api/my-team/{self.ENTRY}")
        assert resp.status_code == 202  # >6h old is not served
        _wait_job(f"my-team:{self.ENTRY}")
        assert client.get(f"/api/my-team/{self.ENTRY}").json()["source"] == "memory"

    def test_shared_recommendation_wrapped(
        self, api_env: SimpleNamespace, client: TestClient
    ) -> None:
        payload = {"entry_id": 888, "recommendation": make_recommendation_dict()}
        (api_env.processed / "recommendation.json").write_text(json.dumps(payload))
        resp = client.get("/api/my-team/888")
        assert resp.status_code == 200
        body = resp.json()
        assert body["source"] == "shared"
        assert body["squad_state"] is None
        assert body["recommendation"]["gw"] == GW

    def test_shared_recommendation_bare_uses_settings_entry(
        self,
        api_env: SimpleNamespace,
        client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        (api_env.processed / "recommendation.json").write_text(
            json.dumps(make_recommendation_dict())
        )
        monkeypatch.setattr(config.settings, "entry_id", 999)
        assert client.get("/api/my-team/999").status_code == 200
        # a different entry must NOT get the configured user's recommendation
        monkeypatch.setattr(jobs, "solve_my_team", _fake_payload)
        assert client.get("/api/my-team/123").status_code == 202
        _wait_job("my-team:123")


# --------------------------------------------------------------------------------------
# Real solve_my_team plumbing (solver + FPL API monkeypatched at the optimizer seams)
# --------------------------------------------------------------------------------------


class TestSolveMyTeamPlumbing:
    def test_solver_receives_xp_window_and_writes_cache(
        self, api_env: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: dict[str, Any] = {}
        state = SquadState.model_validate(make_squad_state_dict())

        monkeypatch.setattr(
            "fplai.optimizer.state.SquadState.from_entry",
            classmethod(lambda cls, entry_id, **kw: state),
        )

        def fake_build_recommendation(
            st: SquadState, xp: pd.DataFrame, prices: pd.DataFrame, *, as_of: datetime, **kw: Any
        ) -> Recommendation:
            captured["state"] = st
            captured["xp"] = xp
            captured["prices"] = prices
            return Recommendation.model_validate(make_recommendation_dict())

        monkeypatch.setattr(
            "fplai.optimizer.plans.build_recommendation", fake_build_recommendation
        )

        payload = jobs.solve_my_team(4242)
        assert captured["state"].current_gw == GW
        assert set(captured["xp"]["gw"]) == {GW, GW + 1}
        assert list(captured["xp"].columns) == ["season", "gw", "player_code", "xp", "q0"]
        prices = captured["prices"]
        assert {"player_code", "price", "position", "team_code", "web_name"} <= set(
            prices.columns
        )
        assert prices["player_code"].is_unique
        assert payload["entry_id"] == 4242
        cache_file = api_env.cache_dir / "api" / "my_team_4242.json"
        assert cache_file.exists()
        assert json.loads(cache_file.read_text())["recommendation"]["action"] == "hold"


# --------------------------------------------------------------------------------------
# OpenAPI completeness (the frontend generates its TS types from these shapes)
# --------------------------------------------------------------------------------------


class TestOpenApi:
    def test_every_endpoint_has_a_response_schema(self, client: TestClient) -> None:
        spec = client.get("/openapi.json").json()
        paths = spec["paths"]
        expected = {
            "/api/health",
            "/api/state",
            "/api/predictions",
            "/api/dream-team",
            "/api/recommendation",
            "/api/chip-curves",
            "/api/players/{player_code}",
            "/api/my-team/{entry_id}",
            "/api/my-team/{entry_id}/status",
            "/api/refresh",
            "/api/refresh/status",
        }
        assert expected <= set(paths)
        for route, ops in paths.items():
            for op in ops.values():
                assert "200" in op["responses"] or "202" in op["responses"], route
        schemas = spec["components"]["schemas"]
        # optimizer contracts are exposed verbatim
        assert "Recommendation" in schemas
        assert "DreamTeam" in schemas
        assert "SquadState" in schemas
        rec_fields = set(schemas["Recommendation"]["properties"])
        assert {"action", "transfers", "chip_advice", "dream_team", "plan", "rationale"} <= (
            rec_fields
        )
        assert "202" in paths["/api/my-team/{entry_id}"]["get"]["responses"]
