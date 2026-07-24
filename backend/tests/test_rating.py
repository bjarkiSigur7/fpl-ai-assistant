"""Offline tests for the squad-rating module and POST /api/rate-team.

Unit tests exercise :mod:`fplai.optimizer.rating` on synthetic frames: legality
validation (every violation reported at once), the greedy best-XI (formation bounds,
captain bonus, tie-breaks), score calibration (floor=0 / optimal=100 anchors,
monotonicity, clamp-over-100) and the no-prediction-rows path.  API tests run the
endpoint through FastAPI's TestClient against tmp fixture parquets/JSON with
``fplai.config`` paths monkeypatched (the established ``tests/test_api.py`` pattern).
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from fplai import config
from fplai.api.app import app
from fplai.api.cache import cache as file_cache
from fplai.optimizer.rating import (
    RatingResult,
    TeamValidationError,
    best_xi,
    cheapest_legal_squad,
    rate_team,
    validate_squad,
)

SEASON = 2026
GWS = (1, 2)
Q0_BY_GW = {1: 0.05, 2: 0.5}

#: (player_code, web_name, position, team_code, price, xp per GW; None = no rows)
ROSTER_ROWS: list[tuple[int, str, str, int, int, float | None]] = [
    # -- floor: price-40 players with 0 xP (a full legal 15 of them exists) -----------
    (101, "GkFloorA", "GKP", 1, 40, 0.0),
    (102, "GkFloorB", "GKP", 2, 40, 0.0),
    (201, "DefFloorA", "DEF", 3, 40, 0.0),
    (202, "DefFloorB", "DEF", 4, 40, 0.0),
    (203, "DefFloorC", "DEF", 5, 40, 0.0),
    (204, "DefFloorD", "DEF", 6, 40, 0.0),
    (205, "DefFloorE", "DEF", 7, 40, 0.0),
    (301, "MidFloorA", "MID", 8, 40, 0.0),
    (302, "MidFloorB", "MID", 9, 40, 0.0),
    (303, "MidFloorC", "MID", 10, 40, 0.0),
    (304, "MidFloorD", "MID", 1, 40, 0.0),
    (305, "MidFloorE", "MID", 2, 40, 0.0),
    (401, "FwdFloorA", "FWD", 3, 40, 0.0),
    (402, "FwdFloorB", "FWD", 4, 40, 0.0),
    (403, "FwdFloorC", "FWD", 5, 40, 0.0),
    # -- stars: the dream-team squad ---------------------------------------------------
    (111, "GkStarA", "GKP", 6, 50, 3.0),
    (112, "GkStarB", "GKP", 7, 45, 2.0),
    (211, "DefStarA", "DEF", 8, 45, 5.5),
    (212, "DefStarB", "DEF", 9, 50, 5.0),
    (213, "DefStarC", "DEF", 10, 50, 4.5),
    (214, "DefStarD", "DEF", 1, 55, 4.0),
    (215, "DefStarE", "DEF", 2, 55, 3.5),
    (311, "MidStarA", "MID", 3, 55, 8.0),
    (312, "MidStarB", "MID", 4, 55, 7.0),
    (313, "MidStarC", "MID", 5, 60, 6.5),
    (314, "MidStarD", "MID", 6, 60, 6.0),
    (315, "MidStarE", "MID", 7, 65, 5.5),
    (411, "FwdStarA", "FWD", 8, 60, 7.5),
    (412, "FwdStarB", "FWD", 9, 65, 6.0),
    (413, "FwdStarC", "FWD", 10, 70, 5.0),
    # -- tier / edge-case players ------------------------------------------------------
    (121, "GkTierA", "GKP", 3, 45, 2.5),
    (122, "GkTierB", "GKP", 4, 45, 2.8),
    (321, "MidTierA", "MID", 8, 55, 2.5),
    (331, "Luxury", "MID", 10, 300, 9.0),
    (500, "NewSigning", "FWD", 1, 55, None),  # no prediction rows: legal, 0 xP
]

FLOOR = [101, 102, 201, 202, 203, 204, 205, 301, 302, 303, 304, 305, 401, 402, 403]
STARS = [111, 112, 211, 212, 213, 214, 215, 311, 312, 313, 314, 315, 411, 412, 413]

#: STARS GW1 greedy XI: 111 / 211,212,213 / 311,312,313,314,315 / 411,412 (3-5-2).
STARS_XI_XP = 64.5
STARS_CAPTAIN_XP = 8.0  # 311


def make_roster() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "season": SEASON,
                "player_code": code,
                "web_name": name,
                "position": pos,
                "team_code": team,
                "price": price,
            }
            for code, name, pos, team, price, _xp in ROSTER_ROWS
        ]
    )


def make_predictions() -> pd.DataFrame:
    rows = [
        {"season": SEASON, "gw": gw, "player_code": code, "xp": xp, "q0": Q0_BY_GW[gw]}
        for code, _name, _pos, _team, _price, xp in ROSTER_ROWS
        if xp is not None
        for gw in GWS
    ]
    return pd.DataFrame(rows)


def _swap(squad: list[int], out_code: int, in_code: int) -> list[int]:
    return [in_code if c == out_code else c for c in squad]


def _rate(
    codes: list[int],
    *,
    dream: list[int] | None = None,
    gw: int = 1,
    horizon: int = 2,
) -> RatingResult:
    return rate_team(
        codes,
        make_predictions(),
        make_roster(),
        list(dream if dream is not None else STARS),
        season=SEASON,
        from_gw=gw,
        horizon=horizon,
    )


# --------------------------------------------------------------------------------------
# Legality validation: every violated rule reported at once
# --------------------------------------------------------------------------------------


class TestValidation:
    def test_valid_squad_has_no_problems(self) -> None:
        assert validate_squad(STARS, make_roster()) == []

    def test_all_violations_reported_at_once(self) -> None:
        # 3 GKP, 4 DEF, 5 MID, 2 FWD + one unknown code = 15 entries, 4 broken rules.
        bad = [101, 102, 111, 201, 202, 203, 204, 301, 302, 303, 304, 305, 401, 402, 999]
        with pytest.raises(TeamValidationError) as ei:
            _rate(bad)
        problems = ei.value.problems
        assert len(problems) == 4
        assert any("unknown player_code 999" in p for p in problems)
        assert any("need 2 GKP, got 3" in p for p in problems)
        assert any("need 5 DEF, got 4" in p for p in problems)
        assert any("need 3 FWD, got 2" in p for p in problems)

    def test_wrong_size_and_duplicate_reported(self) -> None:
        bad = [*STARS[:13], 111]  # 14 entries, 111 twice
        problems = validate_squad(bad, make_roster())
        assert any("need exactly 15 players, got 14" in p for p in problems)
        assert any("duplicate player_code 111" in p for p in problems)

    def test_club_limit_violation(self) -> None:
        # team_code 8 four times: 211 (DEF), 301+321 (MID), 411 (FWD).
        bad = [101, 102, 211, 201, 202, 203, 204, 301, 321, 302, 303, 304, 411, 401, 402]
        problems = validate_squad(bad, make_roster())
        assert problems == ["club limit: max 3 players per club, got 4 from team_code 8"]

    def test_budget_violation(self) -> None:
        bad = _swap(STARS, 315, 331)  # Luxury @ 300 pushes the total to 1075
        problems = validate_squad(bad, make_roster())
        assert problems == ["total price 1075 exceeds budget 1000"]

    def test_unknown_code_raises(self) -> None:
        with pytest.raises(TeamValidationError) as ei:
            _rate(_swap(STARS, 413, 99999))
        assert any("unknown player_code 99999" in p for p in ei.value.problems)


# --------------------------------------------------------------------------------------
# Greedy best XI
# --------------------------------------------------------------------------------------


class TestBestXi:
    def test_formation_maxes_honored_with_def_heavy_profile(self) -> None:
        # 6 monster DEFs and a high-xP backup GK: the XI must still cap at 5 DEF/1 GKP.
        entries = [
            (901, "GKP", 8.0),
            (902, "GKP", 7.9),
            *[(911 + i, "DEF", 9.9 - i * 0.1) for i in range(6)],  # 911..916
            *[(921 + i, "MID", 1.4 - i * 0.1) for i in range(4)],  # 921..924
            *[(931 + i, "FWD", 1.0 - i * 0.1) for i in range(3)],  # 931..933
        ]
        sel = best_xi(entries)
        assert len(sel.xi) == 11
        positions = dict((c, p) for c, p, _ in entries)
        counts = {"GKP": 0, "DEF": 0, "MID": 0, "FWD": 0}
        for c in sel.xi:
            counts[positions[c]] += 1
        assert counts == {"GKP": 1, "DEF": 5, "MID": 4, "FWD": 1}
        assert sel.formation == "5-4-1"
        assert 916 not in sel.xi  # 6th-best DEF benched despite outscoring every MID/FWD
        assert 902 not in sel.xi  # backup GK benched despite xP 7.9
        assert sel.captain == 911  # highest xP in the XI (9.9)

    def test_captain_bonus_is_xi_max_and_ties_break_low_code(self) -> None:
        entries = [
            (1, "GKP", 1.0),
            (11, "DEF", 5.0),
            (12, "DEF", 4.0),
            (13, "DEF", 3.0),
            (21, "MID", 5.0),  # ties 11 for the XI max -> lower code 11 is captain
            (22, "MID", 2.0),
            (31, "FWD", 2.0),
            (14, "DEF", 1.0),
            (23, "MID", 1.0),
            (24, "MID", 1.0),
            (32, "FWD", 1.0),
        ]
        sel = best_xi(entries)
        assert sel.captain == 11
        assert sel.captain_xp == pytest.approx(5.0)

    def test_infeasible_xi_raises(self) -> None:
        with pytest.raises(ValueError, match="need 1 GKP"):
            best_xi([(11, "DEF", 1.0), (21, "MID", 1.0)])


# --------------------------------------------------------------------------------------
# rate_team scoring
# --------------------------------------------------------------------------------------


class TestRateTeam:
    def test_dream_squad_scores_100(self) -> None:
        r = _rate(STARS)
        assert r.score == 100.0
        assert r.verdict == "ELITE"
        assert (r.season, r.from_gw, r.horizon) == (SEASON, 1, 2)
        assert r.team_xp_horizon == pytest.approx((STARS_XI_XP + STARS_CAPTAIN_XP) * 2)
        assert r.optimal_xp_horizon == pytest.approx(r.team_xp_horizon)
        assert r.floor_xp_horizon == pytest.approx(0.0)
        assert r.formation_gw1 == "3-5-2"
        assert r.suggested_captain == 311
        assert set(r.best_xi_gw1) == {111, 211, 212, 213, 311, 312, 313, 314, 315, 411, 412}
        assert [p.player_code for p in r.player_ratings] == STARS  # input order kept
        assert sum(p.in_best_xi_gw1 for p in r.player_ratings) == 11
        assert [w.player_code for w in r.weakest] == [112, 111, 215]  # lowest xp_horizon
        assert r.weakest[0].xp_horizon == pytest.approx(4.0)

    def test_floor_squad_scores_0(self) -> None:
        r = _rate(FLOOR)
        assert r.score == 0.0
        assert r.verdict == "FODDER"
        assert r.team_xp_horizon == pytest.approx(0.0)

    def test_cheapest_legal_squad_is_the_floor(self) -> None:
        assert cheapest_legal_squad(make_roster()) == FLOOR

    def test_captain_bonus_included_in_gw_term(self) -> None:
        r = _rate(STARS, horizon=1)
        assert r.team_xp_gw1 == pytest.approx(STARS_XI_XP + STARS_CAPTAIN_XP)
        assert r.team_xp_horizon == pytest.approx(r.team_xp_gw1)

    def test_score_monotonic_in_player_quality(self) -> None:
        weaker = _swap(STARS, 111, 121)  # GK 3.0 -> 2.5 (always in the XI)
        better = _swap(STARS, 111, 122)  # GK 3.0 -> 2.8
        s_weaker, s_better = _rate(weaker).score, _rate(better).score
        assert 0.0 < s_weaker < s_better < 100.0

    def test_clamp_when_team_beats_the_dream_squad(self) -> None:
        # The dream team optimizes a decayed+chips objective, so a squad CAN beat it
        # on this raw metric — the score clamps to 100 instead of exceeding it.
        weaker_dream = _swap(STARS, 111, 121)
        r = _rate(STARS, dream=weaker_dream)
        assert r.team_xp_horizon > r.optimal_xp_horizon
        assert r.score == 100.0
        assert r.verdict == "ELITE"

    def test_player_without_prediction_rows_contributes_zero(self) -> None:
        # NewSigning is in the roster but has no prediction rows; swapping it for 412
        # (a GW1 XI member) must drop the score below 100 and bottom the weakest list.
        squad = _swap(STARS, 412, 500)
        r = _rate(squad)
        row = next(p for p in r.player_ratings if p.player_code == 500)
        assert row.xp_horizon == 0.0
        assert row.xp_gw1 == 0.0
        assert row.q0 is None
        assert not row.in_best_xi_gw1
        assert r.weakest[0].player_code == 500
        assert 0.0 < r.score < 100.0

    def test_q0_taken_from_first_window_gw(self) -> None:
        assert _rate(STARS).player_ratings[0].q0 == pytest.approx(Q0_BY_GW[1])
        assert _rate(STARS, gw=2, horizon=1).player_ratings[0].q0 == pytest.approx(Q0_BY_GW[2])


# --------------------------------------------------------------------------------------
# POST /api/rate-team (TestClient against tmp fixture artifacts)
# --------------------------------------------------------------------------------------

CONTRACT_KEYS = {
    "score",
    "verdict",
    "season",
    "from_gw",
    "horizon",
    "team_xp_gw1",
    "team_xp_horizon",
    "optimal_xp_horizon",
    "floor_xp_horizon",
    "best_xi_gw1",
    "formation_gw1",
    "suggested_captain",
    "player_ratings",
    "weakest",
}
PLAYER_RATING_KEYS = {
    "player_code",
    "web_name",
    "position",
    "team_short",
    "price",
    "xp_gw1",
    "xp_horizon",
    "q0",
    "in_best_xi_gw1",
}
WEAKEST_KEYS = {"player_code", "web_name", "xp_horizon"}


def make_teams() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "season": SEASON,
                "fpl_team_id": t,
                "team_code": t,
                "name": f"Team {t}",
                "short_name": f"T{t}",
            }
            for t in range(1, 11)
        ]
    )


def make_dream_team_dict() -> dict[str, object]:
    return {
        "season": SEASON,
        "gw": 1,
        "squad": STARS,
        "lineup": STARS[:11],
        "bench_order": STARS[11:],
        "captain": 311,
        "vice": 312,
        "formation": "3-5-2",
        "expected_points": 72.5,
        "objective": 145.0,
        "total_cost": 840,
        "solve_seconds": 0.1,
    }


@pytest.fixture()
def api_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Tmp processed dir with fixture artifacts; config paths monkeypatched; cache reset."""
    processed = tmp_path / "processed"
    processed.mkdir(parents=True)
    monkeypatch.setattr(config, "PROCESSED_DIR", processed)
    file_cache.clear()

    make_predictions().to_parquet(processed / "predictions_gw.parquet", index=False)
    make_roster().to_parquet(processed / "live_roster.parquet", index=False)
    make_teams().to_parquet(processed / "teams.parquet", index=False)
    (processed / "dream_team.json").write_text(json.dumps(make_dream_team_dict()))

    yield SimpleNamespace(processed=processed)
    file_cache.clear()


@pytest.fixture()
def client(api_env: SimpleNamespace) -> TestClient:
    return TestClient(app)


class TestRateTeamEndpoint:
    def test_happy_path_matches_contract_shape(self, client: TestClient) -> None:
        resp = client.post("/api/rate-team", json={"player_codes": STARS})
        assert resp.status_code == 200
        body = resp.json()
        assert set(body) == CONTRACT_KEYS
        assert body["score"] == 100.0
        assert body["verdict"] == "ELITE"
        assert body["season"] == SEASON
        assert body["from_gw"] == 1  # defaults to the window's first GW
        assert body["horizon"] == 2
        assert body["team_xp_gw1"] == pytest.approx(STARS_XI_XP + STARS_CAPTAIN_XP)
        assert body["team_xp_horizon"] == pytest.approx((STARS_XI_XP + STARS_CAPTAIN_XP) * 2)
        assert body["optimal_xp_horizon"] == pytest.approx(body["team_xp_horizon"])
        assert body["floor_xp_horizon"] == pytest.approx(0.0)
        assert len(body["best_xi_gw1"]) == 11
        assert body["formation_gw1"] == "3-5-2"
        assert body["suggested_captain"] == 311
        assert len(body["player_ratings"]) == 15
        for entry in body["player_ratings"]:
            assert set(entry) == PLAYER_RATING_KEYS
        first = body["player_ratings"][0]
        assert first["player_code"] == 111  # input order preserved
        assert first["web_name"] == "GkStarA"
        assert first["position"] == "GKP"
        assert first["team_short"] == "T6"  # joined from teams.parquet
        assert first["price"] == 50
        assert first["q0"] == pytest.approx(Q0_BY_GW[1])
        assert first["in_best_xi_gw1"] is True
        assert len(body["weakest"]) == 3
        for entry in body["weakest"]:
            assert set(entry) == WEAKEST_KEYS
        assert body["weakest"][0] == {
            "player_code": 112,
            "web_name": "GkStarB",
            "xp_horizon": pytest.approx(4.0),
        }

    def test_explicit_gw_narrows_the_window(self, client: TestClient) -> None:
        body = client.post(
            "/api/rate-team", json={"player_codes": STARS, "season": SEASON, "gw": 2}
        ).json()
        assert (body["from_gw"], body["horizon"]) == (2, 1)
        assert body["team_xp_gw1"] == pytest.approx(body["team_xp_horizon"])
        assert body["player_ratings"][0]["q0"] == pytest.approx(Q0_BY_GW[2])

    def test_422_lists_every_violated_rule(self, client: TestClient) -> None:
        bad = [101, 102, 111, 201, 202, 203, 204, 301, 302, 303, 304, 305, 401, 402, 999]
        resp = client.post("/api/rate-team", json={"player_codes": bad})
        assert resp.status_code == 422
        detail = resp.json()["detail"]
        assert isinstance(detail, list)
        assert all(isinstance(d, str) for d in detail)
        assert len(detail) == 4
        assert any("unknown player_code 999" in d for d in detail)
        assert any("need 2 GKP, got 3" in d for d in detail)
        assert any("need 5 DEF, got 4" in d for d in detail)
        assert any("need 3 FWD, got 2" in d for d in detail)

    def test_422_wrong_size(self, client: TestClient) -> None:
        resp = client.post("/api/rate-team", json={"player_codes": STARS[:14]})
        assert resp.status_code == 422
        assert any("need exactly 15 players, got 14" in d for d in resp.json()["detail"])

    def test_422_unknown_code(self, client: TestClient) -> None:
        resp = client.post(
            "/api/rate-team", json={"player_codes": _swap(STARS, 413, 99999)}
        )
        assert resp.status_code == 422
        assert any("unknown player_code 99999" in d for d in resp.json()["detail"])

    def test_404_predictions_missing(
        self, api_env: SimpleNamespace, client: TestClient
    ) -> None:
        (api_env.processed / "predictions_gw.parquet").unlink()
        resp = client.post("/api/rate-team", json={"player_codes": STARS})
        assert resp.status_code == 404
        assert "fplai predict" in resp.json()["detail"]

    def test_404_roster_missing(self, api_env: SimpleNamespace, client: TestClient) -> None:
        (api_env.processed / "live_roster.parquet").unlink()
        resp = client.post("/api/rate-team", json={"player_codes": STARS})
        assert resp.status_code == 404
        assert "live_roster.parquet" in resp.json()["detail"]

    def test_404_dream_team_missing(
        self, api_env: SimpleNamespace, client: TestClient
    ) -> None:
        (api_env.processed / "dream_team.json").unlink()
        resp = client.post("/api/rate-team", json={"player_codes": STARS})
        assert resp.status_code == 404
        assert "dream_team.json not found" in resp.json()["detail"]

    def test_404_gw_without_predictions(self, client: TestClient) -> None:
        resp = client.post("/api/rate-team", json={"player_codes": STARS, "gw": 9})
        assert resp.status_code == 404
        assert "available GWs" in resp.json()["detail"]

    def test_openapi_exposes_the_route_and_model(self, client: TestClient) -> None:
        spec = client.get("/openapi.json").json()
        assert "post" in spec["paths"]["/api/rate-team"]
        assert "RatingResult" in spec["components"]["schemas"]
        assert set(spec["components"]["schemas"]["RatingResult"]["properties"]) == (
            CONTRACT_KEYS
        )
