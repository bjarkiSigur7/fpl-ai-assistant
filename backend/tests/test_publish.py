"""Offline tests for the static site-data bundle publisher (:mod:`fplai.publish`).

A tmp processed/ dir is populated with synthetic parquet/JSON artifacts mirroring the
real pipeline outputs; tests build the bundle and parse every file back against the
shared static-data contract: schema round-trip, byte-identical determinism, graceful
degradation of optional artifacts, required-artifact errors, the size guard, 3 dp
float rounding everywhere, and rating.json anchors matching the *exact*
:mod:`fplai.optimizer.rating` metric functions over the shipped (rounded) xP table.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from fplai import publish
from fplai.optimizer.rating import cheapest_legal_squad, squad_metric
from fplai.publish import build_bundle

SEASON = 2026
GWS = (1, 2)
Q0_BY_GW = {1: 0.05, 2: 0.5}

#: (player_code, web_name, position, team_code, price, base_xp) — clubs 1..10, ≤3 each.
ROSTER_ROWS: list[tuple[int, str, str, int, int, float]] = [
    # -- floor: price-40 players with small xP (a full legal 15 of them exists) -------
    (101, "GkFloorA", "GKP", 1, 40, 0.5),
    (102, "GkFloorB", "GKP", 2, 40, 0.5),
    (201, "DefFloorA", "DEF", 3, 40, 0.5),
    (202, "DefFloorB", "DEF", 4, 40, 0.5),
    (203, "DefFloorC", "DEF", 5, 40, 0.5),
    (204, "DefFloorD", "DEF", 6, 40, 0.5),
    (205, "DefFloorE", "DEF", 7, 40, 0.5),
    (301, "MidFloorA", "MID", 8, 40, 0.5),
    (302, "MidFloorB", "MID", 9, 40, 0.5),
    (303, "MidFloorC", "MID", 10, 40, 0.5),
    (304, "MidFloorD", "MID", 1, 40, 0.5),
    (305, "MidFloorE", "MID", 2, 40, 0.5),
    (401, "FwdFloorA", "FWD", 3, 40, 0.5),
    (402, "FwdFloorB", "FWD", 4, 40, 0.5),
    (403, "FwdFloorC", "FWD", 5, 40, 0.5),
    # -- stars: the dream-team squad --------------------------------------------------
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
]
#: Predictions only in GW1 — exercises the 0-fill in xp.json's columnar table.
ONE_GW_PLAYER = (555, "OneGw", "FWD", 6, 45, 4.0)

FLOOR = [101, 102, 201, 202, 203, 204, 205, 301, 302, 303, 304, 305, 401, 402, 403]
STARS = [111, 112, 211, 212, 213, 214, 215, 311, 312, 313, 314, 315, 411, 412, 413]

ALL_ROWS = [*ROSTER_ROWS, ONE_GW_PLAYER]


def _xp(base: float, gw: int) -> float:
    """High-precision per-GW xP so the 3 dp rounding actually does something."""
    return base + 0.000456789 if gw == 1 else base * 0.75 + 0.000123456


def _write_processed(processed: Path) -> None:
    processed.mkdir(parents=True, exist_ok=True)

    gw_rows, fixture_rows = [], []
    for code, name, pos, team, price, base in ALL_ROWS:
        for gw in GWS:
            if code == ONE_GW_PLAYER[0] and gw != 1:
                continue
            xp = _xp(base, gw)
            gw_rows.append(
                {
                    "season": SEASON, "gw": gw, "player_code": code, "n_fixtures": 1,
                    "xp": xp, "xp_appearance": xp * 0.4, "xp_goals": xp * 0.3,
                    "xp_assists": xp * 0.1, "xp_cs": xp * 0.1, "xp_concede": -0.0512345,
                    "xp_saves": 0.0, "xp_defcon": 0.0, "xp_bonus": xp * 0.1,
                    "xp_cards": -0.0123456, "xp_other": 0.0, "q0": Q0_BY_GW[gw],
                    "team_code": team, "position": pos, "price": price, "web_name": name,
                }
            )
            fixture_rows.append(
                {
                    "season": SEASON, "gw": gw, "player_code": code,
                    "fpl_fixture_id": gw * 100 + team, "xp": xp,
                    "team_code": team, "opponent_code": (team % 10) + 1,
                    "was_home": code % 2 == 0, "position": pos, "price": price,
                    "q0": Q0_BY_GW[gw], "q1": 0.1, "q2": 1.0 - Q0_BY_GW[gw] - 0.1,
                    "mu1": 30.0, "mu2": 85.1234567,
                }
            )
    pd.DataFrame(gw_rows).to_parquet(processed / "predictions_gw.parquet")
    pd.DataFrame(fixture_rows).to_parquet(processed / "predictions.parquet")

    pd.DataFrame(
        [
            {
                "season": SEASON, "player_code": code, "web_name": name, "position": pos,
                "team_code": team, "price": price,
                "status": "i" if code == ONE_GW_PLAYER[0] else "a",
                "selected_by_percent": 25.512345 if code == 311 else 1.0,
                "news": "Knee injury - Expected back 23 Aug"
                if code == ONE_GW_PLAYER[0]
                else "",
                "penalties_order": 1 if code == 311 else None,
                "penalties_text": "Shares duties" if code == 311 else "",
                "direct_freekicks_order": None,
                "direct_freekicks_text": "",
                "corners_and_indirect_freekicks_order": 2 if code == 311 else None,
                "corners_and_indirect_freekicks_text": "",
            }
            for code, name, pos, team, price, _ in ALL_ROWS
        ]
    ).to_parquet(processed / "live_roster.parquet")

    (processed / f"availability_{SEASON}.json").write_text(
        json.dumps(
            {
                "season": SEASON,
                "returns": {
                    str(ONE_GW_PLAYER[0]): {
                        "return_date": "2026-08-23", "kind": "injury", "return_gw": 2,
                    }
                },
            }
        )
    )

    pd.DataFrame(
        [
            {
                "season": SEASON, "gw": gw, "fpl_fixture_id": gw * 100 + h,
                "kickoff_utc": pd.Timestamp("2026-08-21 19:00:00+00:00"),
                "home_team_code": h, "away_team_code": (h % 10) + 1,
                "home_lambda": 1.6123456, "away_lambda": 1.1,
                "p_cs_home": 0.31234567, "p_cs_away": 0.2,
                "p_home_win": 0.5, "p_draw": 0.25, "p_away_win": 0.25,
                "odds_blended": gw == 1,
            }
            for gw in GWS
            for h in (1, 3)
        ]
    ).to_parquet(processed / "team_fixtures.parquet")

    pd.DataFrame(
        [
            {
                "season": SEASON, "gw": 1, "player_code": 311, "xp": 8.0004567,
                "mean_pts": 7.912345, "sd_pts": 4.1, "p_haul": 0.3312345,
                "p_blank": 0.15, "p_best": 0.41, "p_beats_top": np.nan,
                "n_draws": 2000,
            },
            {
                "season": SEASON, "gw": 1, "player_code": 411, "xp": 7.5,
                "mean_pts": 7.4, "sd_pts": 4.0, "p_haul": 0.30,
                "p_blank": 0.17, "p_best": 0.35, "p_beats_top": 0.44123456,
                "n_draws": 2000,
            },
        ]
    ).to_parquet(processed / "captaincy.parquet")

    pd.DataFrame(
        [
            {
                "season": SEASON, "fpl_team_id": i, "team_code": i,
                "name": f"Team {i}", "short_name": f"T{i}",
            }
            for i in range(1, 11)
        ]
    ).to_parquet(processed / "teams.parquet")

    (processed / "dream_team.json").write_text(
        json.dumps(
            {
                "season": SEASON, "gw": 1, "squad": STARS, "lineup": STARS[:11],
                "bench_order": STARS[11:], "captain": 311, "vice": 411,
                "formation": "3-4-3", "expected_points": 65.13459599443708,
                "objective": 232.60234809282025, "total_cost": 995,
                "solve_seconds": 8.50185083411634,
            }
        )
    )
    (processed / "recommendation.json").write_text(
        json.dumps(
            {
                "season": SEASON, "gw": 1, "action": "initial-squad", "captain": 311,
                "objective": 232.60234809282025, "squad": STARS,
                "chip_advice": [
                    {"chip": "bb1", "verdict": "hold", "ev_now": 6.585123456789}
                ],
                "rationale": ["build the dream team"],
            }
        )
    )

    _write_chip_curves(processed, v2=True)
    (processed / "chip_sim_report.json").write_text(
        json.dumps(
            {
                "season": SEASON, "n_rollouts": 1000,
                "stats": [
                    {"chip": "wc1", "gw": 2, "sd": 15.2779711,
                     "p_best_week": 0.045, "p_beats_hold": 0.049},
                    {"chip": "bb1", "gw": 1, "sd": 4.4441563,
                     "p_best_week": 0.0999166, "p_beats_hold": 0.11},
                ],
            }
        )
    )


def _write_chip_curves(processed: Path, *, v2: bool) -> None:
    rows = [
        {"chip": "wc1", "gw": 1, "objective": np.nan, "delta_vs_no_chip": np.nan,
         "evaluated": False, "skip_reason": "outside_window", "window_last_gw": 19,
         "urgency": np.nan, "sd": np.nan, "p_best_week": np.nan, "p_beats_hold": np.nan,
         "n_rollouts": 1000},
        {"chip": "wc1", "gw": 2, "objective": 279.92791666, "delta_vs_no_chip": 2.46855,
         "evaluated": True, "skip_reason": None, "window_last_gw": 19,
         "urgency": -4.7226, "sd": 15.2779711, "p_best_week": 0.045,
         "p_beats_hold": 0.049, "n_rollouts": 1000},
        {"chip": "bb1", "gw": 1, "objective": 270.0, "delta_vs_no_chip": 1.23456789,
         "evaluated": True, "skip_reason": None, "window_last_gw": 19,
         "urgency": 0.5, "sd": 4.4441563, "p_best_week": 0.0999166,
         "p_beats_hold": 0.11, "n_rollouts": 1000},
    ]
    df = pd.DataFrame(rows)
    if not v2:
        df = df.drop(columns=["sd", "p_best_week", "p_beats_hold", "n_rollouts"])
    df.to_parquet(processed / "chip_curves.parquet")


@pytest.fixture
def env(tmp_path: Path) -> SimpleNamespace:
    processed = tmp_path / "processed"
    _write_processed(processed)
    return SimpleNamespace(processed=processed, out=tmp_path / "site-data", tmp=tmp_path)


def _load(out: Path, name: str):
    return json.loads((out / name).read_text(encoding="utf-8"))


# --------------------------------------------------------------------------------------
# Round-trip: every file parses and matches the shared contract
# --------------------------------------------------------------------------------------


def test_round_trip_contract(env: SimpleNamespace) -> None:
    report = build_bundle(env.processed, env.out)

    assert (env.out / "history").is_dir()
    for name in publish._BUNDLE_FILES:
        assert (env.out / name).exists(), name
    assert report.missing == ()
    assert report.season == SEASON
    assert report.window == (1, 2)

    meta = _load(env.out, "meta.json")
    assert set(meta) == {
        "generated_utc", "season", "next_gw", "next_deadline_utc",
        "model_version", "window", "missing",
    }
    assert meta["season"] == SEASON
    assert meta["next_gw"] == 1
    assert meta["window"] == {"from_gw": 1, "through_gw": 2}
    assert meta["missing"] == []
    assert meta["next_deadline_utc"] is None  # no raw_dir given
    assert meta["model_version"] is None
    assert meta["generated_utc"].endswith("Z") and "T" in meta["generated_utc"]

    players = _load(env.out, "players.json")
    assert len(players) == len(ALL_ROWS)
    assert [p["player_code"] for p in players] == sorted(r[0] for r in ALL_ROWS)
    first = next(p for p in players if p["player_code"] == 101)
    assert set(first) == {
        "player_code", "web_name", "position", "team_short", "team_code",
        "price", "status", "q0_gw1", "ownership", "news", "return_gw", "pen_order",
    }
    assert first == {
        "player_code": 101, "web_name": "GkFloorA", "position": "GKP",
        "team_short": "T1", "team_code": 1, "price": 40, "status": "a", "q0_gw1": 0.05,
        "ownership": 1.0, "news": None, "return_gw": None, "pen_order": None,
    }
    one_gw = next(p for p in players if p["player_code"] == 555)
    assert one_gw["status"] == "i" and one_gw["q0_gw1"] == 0.05
    assert one_gw["news"] == "Knee injury - Expected back 23 Aug"
    assert one_gw["return_gw"] == 2  # from availability_{season}.json
    star = next(p for p in players if p["player_code"] == 311)
    assert star["ownership"] == 25.512 and star["pen_order"] == 1

    xp = _load(env.out, "xp.json")
    assert xp["gws"] == [1, 2]
    assert set(xp["players"]) == {str(r[0]) for r in ALL_ROWS}
    assert all(len(v) == 2 for v in xp["players"].values())
    assert xp["players"]["555"] == [round(_xp(4.0, 1), 3), 0.0]  # 0-filled second GW
    assert xp["players"]["311"] == [round(_xp(8.0, 1), 3), round(_xp(8.0, 2), 3)]

    gw1 = _load(env.out, "predictions_gw1.json")
    assert len(gw1) == len(ALL_ROWS)
    assert gw1[0]["player_code"] == 311  # sorted by xp desc
    row = gw1[0]
    for key in (
        "web_name", "position", "team_code", "team_short", "price", "n_fixtures",
        "xp", "xp_appearance", "xp_goals", "xp_assists", "xp_cs", "xp_concede",
        "xp_saves", "xp_defcon", "xp_bonus", "xp_cards", "xp_other", "q0", "fixtures",
    ):
        assert key in row, key
    assert "season" not in row and "gw" not in row
    leg = row["fixtures"][0]
    assert leg["fpl_fixture_id"] == 103 and leg["opponent_code"] == 4
    assert leg["opponent_short"] == "T4"
    assert set(leg) == {
        "fpl_fixture_id", "opponent_code", "opponent_short", "was_home",
        "xp", "q0", "q1", "q2", "mu1", "mu2",
    }

    rec = _load(env.out, "recommendation.json")
    src = json.loads((env.processed / "recommendation.json").read_text())
    assert set(rec) == set(src)  # structure verbatim
    assert rec["objective"] == 232.602 and rec["squad"] == STARS
    assert rec["chip_advice"][0]["ev_now"] == 6.585

    dream = _load(env.out, "dream_team.json")
    assert dream["expected_points"] == 65.135 and dream["squad"] == STARS

    curves = _load(env.out, "chip_curves.json")
    assert [(c["chip"], c["gw"]) for c in curves] == [("bb1", 1), ("wc1", 1), ("wc1", 2)]
    skipped = curves[1]
    assert skipped["objective"] is None and skipped["skip_reason"] == "outside_window"
    assert skipped["sd"] is None and skipped["evaluated"] is False
    assert curves[2]["sd"] == 15.278 and curves[2]["n_rollouts"] == 1000

    rating = _load(env.out, "rating.json")
    assert set(rating) == {"floor_metric", "optimal_metric", "window_gws", "captain_bonus"}
    assert rating["captain_bonus"] is True
    assert rating["window_gws"] == [1, 2]
    assert 0 < rating["floor_metric"] < rating["optimal_metric"]

    fixtures = _load(env.out, "fixtures.json")
    assert len(fixtures) == 4  # 2 GWs x 2 fixtures
    fx = fixtures[0]
    assert set(fx) == {
        "gw", "fpl_fixture_id", "kickoff_utc", "home_code", "home_short",
        "away_code", "away_short", "home_xg", "away_xg", "p_cs_home", "p_cs_away",
        "p_home_win", "p_draw", "p_away_win", "odds_blended",
    }
    assert fx["gw"] == 1 and fx["home_short"] == "T1" and fx["away_short"] == "T2"
    assert fx["home_xg"] == 1.612 and fx["p_cs_home"] == 0.312
    assert fx["odds_blended"] is True and fixtures[-1]["odds_blended"] is False
    assert fx["kickoff_utc"] == "2026-08-21T19:00:00Z"

    set_pieces = _load(env.out, "set_pieces.json")
    assert len(set_pieces) == 1
    sp = set_pieces[0]
    assert sp["player_code"] == 311 and sp["pen"] == 1 and sp["corner"] == 2
    assert sp["pen_note"] == "Shares duties"
    assert "fk" not in sp

    captaincy = _load(env.out, "captaincy.json")
    assert [c["player_code"] for c in captaincy] == [311, 411]  # xp desc
    top = captaincy[0]
    assert top["p_haul"] == 0.331 and top["p_beats_top"] is None
    assert captaincy[1]["p_beats_top"] == 0.441


def test_rating_anchors_reuse_exact_engine(env: SimpleNamespace) -> None:
    """floor/optimal must equal fplai.optimizer.rating's functions over the SHIPPED
    (3 dp-rounded) xP table — a client re-running the formula reproduces them."""
    build_bundle(env.processed, env.out)
    rating = _load(env.out, "rating.json")

    roster = pd.read_parquet(env.processed / "live_roster.parquet")
    pos_of = {int(r.player_code): str(r.position) for r in roster.itertuples(index=False)}
    xp_of = {
        (gw, code): round(_xp(base, gw), 3)
        for code, _, _, _, _, base in ALL_ROWS
        for gw in GWS
        if not (code == ONE_GW_PLAYER[0] and gw != 1)
    }
    assert cheapest_legal_squad(roster) == FLOOR
    assert rating["floor_metric"] == round(squad_metric(FLOOR, pos_of, xp_of, [1, 2]), 3)
    assert rating["optimal_metric"] == round(squad_metric(STARS, pos_of, xp_of, [1, 2]), 3)


# --------------------------------------------------------------------------------------
# Determinism, degradation, required artifacts, size guard
# --------------------------------------------------------------------------------------


def test_determinism_byte_identical(env: SimpleNamespace) -> None:
    out2 = env.tmp / "site-data-2"
    build_bundle(env.processed, env.out)
    build_bundle(env.processed, out2)
    build_bundle(env.processed, env.out)  # rebuild in place too
    for name in publish._BUNDLE_FILES:
        assert (env.out / name).read_bytes() == (out2 / name).read_bytes(), name


def test_graceful_degradation_optional_artifacts(env: SimpleNamespace) -> None:
    for name in publish._OPTIONAL:
        (env.processed / name).unlink()
    report = build_bundle(env.processed, env.out)

    assert report.missing == tuple(sorted(publish._OPTIONAL))
    meta = _load(env.out, "meta.json")
    assert meta["missing"] == sorted(publish._OPTIONAL)
    assert _load(env.out, "chip_curves.json") == []
    assert _load(env.out, "recommendation.json") is None
    assert _load(env.out, "dream_team.json") is None
    rating = _load(env.out, "rating.json")
    assert rating["optimal_metric"] is None
    assert rating["floor_metric"] > 0
    assert all(p["team_short"] is None for p in _load(env.out, "players.json"))


def test_missing_required_artifacts_raise(env: SimpleNamespace) -> None:
    (env.processed / "predictions_gw.parquet").unlink()
    (env.processed / "live_roster.parquet").unlink()
    with pytest.raises(FileNotFoundError, match="predictions_gw.parquet") as exc_info:
        build_bundle(env.processed, env.out)
    assert "live_roster.parquet" in str(exc_info.value)
    assert "fplai predict" in str(exc_info.value)


def test_size_guard_reports_and_enforces(
    env: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = build_bundle(env.processed, env.out)
    assert tuple(report.files) == publish._BUNDLE_FILES
    for name, size in report.files.items():
        assert size == (env.out / name).stat().st_size, name
    assert report.total_bytes == sum(report.files.values())
    assert report.total_bytes < publish.MAX_BUNDLE_BYTES

    monkeypatch.setattr(publish, "MAX_BUNDLE_BYTES", 10)
    with pytest.raises(ValueError, match="budget"):
        build_bundle(env.processed, env.out)


def test_v1_curves_backfilled_from_sim_report(env: SimpleNamespace) -> None:
    _write_chip_curves(env.processed, v2=False)
    build_bundle(env.processed, env.out)
    curves = _load(env.out, "chip_curves.json")
    bb1 = next(c for c in curves if c["chip"] == "bb1")
    assert bb1["sd"] == 4.444 and bb1["p_beats_hold"] == 0.11
    assert bb1["n_rollouts"] == 1000
    wc1_skip = next(c for c in curves if c["chip"] == "wc1" and c["gw"] == 1)
    assert wc1_skip["sd"] is None  # no report stat for the skipped GW


def _walk_floats(obj) -> list[float]:
    if isinstance(obj, bool):
        return []
    if isinstance(obj, float):
        return [obj]
    if isinstance(obj, dict):
        return [f for v in obj.values() for f in _walk_floats(v)]
    if isinstance(obj, list):
        return [f for v in obj for f in _walk_floats(v)]
    return []


def test_every_float_rounded_to_3dp(env: SimpleNamespace) -> None:
    build_bundle(env.processed, env.out)
    for name in publish._BUNDLE_FILES:
        for x in _walk_floats(_load(env.out, name)):
            assert x == round(x, 3), f"{name}: {x!r} not 3 dp"


# --------------------------------------------------------------------------------------
# meta.json enrichment from raw snapshots + model manifest
# --------------------------------------------------------------------------------------


def test_meta_deadline_and_model_version(env: SimpleNamespace) -> None:
    raw = env.tmp / "raw"
    snap = raw / "fpl_api" / "snapshots" / "2026-07-24"
    snap.mkdir(parents=True)
    (snap / "bootstrap.json").write_text(
        json.dumps(
            {
                "game_config": {
                    "settings": {"static_content_url": "https://pl.example/2026_27/"}
                },
                "events": [
                    {"id": 1, "is_next": True, "deadline_time": "2026-08-21T17:30:00Z"}
                ],
                "total_players": 5,
            }
        )
    )
    models = env.tmp / "models"
    models.mkdir()
    (models / "manifest.json").write_text(
        json.dumps({"schema": 1, "created_utc": "2026-07-22T15:02:36+00:00"})
    )

    build_bundle(env.processed, env.out, raw_dir=raw, models_dir=models)
    meta = _load(env.out, "meta.json")
    assert meta["next_deadline_utc"] == "2026-08-21T17:30:00Z"
    assert meta["model_version"] == "2026-07-22T15:02:36+00:00"


# --------------------------------------------------------------------------------------
# CLI: fplai publish-static
# --------------------------------------------------------------------------------------


def test_cli_publish_static(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fplai import config
    from fplai.cli import app

    processed = tmp_path / "processed"
    _write_processed(processed)
    monkeypatch.setattr(config, "PROCESSED_DIR", processed)
    monkeypatch.setattr(config, "RAW_DIR", tmp_path / "raw")
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(config, "REPO_ROOT", tmp_path)

    result = CliRunner().invoke(app, ["publish-static"])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "site-data" / "meta.json").exists()
    assert "total" in result.output

    out_dir = tmp_path / "elsewhere"
    result = CliRunner().invoke(app, ["publish-static", "--out", str(out_dir)])
    assert result.exit_code == 0, result.output
    assert (out_dir / "meta.json").exists()


def test_cli_publish_static_missing_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fplai import config
    from fplai.cli import app

    empty = tmp_path / "processed"
    empty.mkdir()
    monkeypatch.setattr(config, "PROCESSED_DIR", empty)
    monkeypatch.setattr(config, "RAW_DIR", tmp_path / "raw")
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(config, "REPO_ROOT", tmp_path)

    result = CliRunner().invoke(app, ["publish-static"])
    assert result.exit_code == 1
    assert "publish-static failed" in result.output
