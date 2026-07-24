"""Regression guards for the 2026-07-24 degenerate-incumbent failure.

The all-chips MILP timed out holding only HiGHS's trivial feasibility incumbent
(cheapest legal squad, GW1 12.2 xP vs 65+ attainable) and the pipeline published
it. Two independent guards now prevent this: solve_plan's two-phase floor (the
chip solve can never return worse than the always-feasible no-chips restriction)
and run_optimize's degenerate-plan gate (refuses to write artifacts when the
plan's first-GW xP collapses versus what the predictions obviously support).
"""

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from fplai.optimizer.milp import SolveParams, solve_plan

SEASON = 2026

POOL: list[tuple[int, str, int, int]] = [
    (1, "GKP", 45, 1),
    (2, "GKP", 40, 2),
    (11, "DEF", 45, 3),
    (12, "DEF", 45, 4),
    (13, "DEF", 45, 5),
    (14, "DEF", 40, 6),
    (15, "DEF", 40, 7),
    (21, "MID", 60, 8),
    (22, "MID", 60, 9),
    (23, "MID", 55, 10),
    (24, "MID", 50, 11),
    (25, "MID", 50, 12),
    (31, "FWD", 65, 13),
    (32, "FWD", 60, 14),
    (33, "FWD", 55, 15),
    (34, "FWD", 70, 16),
    (26, "MID", 70, 17),
    (16, "DEF", 55, 18),
]


def _prices() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"player_code": c, "position": pos, "price": price, "team_code": club}
            for c, pos, price, club in POOL
        ]
    )


def _xp(gws: list[int]) -> pd.DataFrame:
    per_player = {c: 2.0 + (c % 7) * 0.5 for c, *_ in POOL}
    return pd.DataFrame(
        [
            {"season": SEASON, "gw": gw, "player_code": c, "xp": v}
            for gw in gws
            for c, v in per_player.items()
        ]
    )


def test_chip_solve_never_below_no_chips_floor() -> None:
    xp, prices = _xp([30, 31, 32]), _prices()
    floor = solve_plan(
        xp, prices, None, horizon=3, params=SolveParams(no_chips=True, time_limit_s=30)
    )
    full = solve_plan(xp, prices, None, horizon=3, params=SolveParams(time_limit_s=30))
    assert full.objective >= floor.objective - 1e-6


def test_two_phase_disabled_still_solves() -> None:
    xp, prices = _xp([30, 31]), _prices()
    plan = solve_plan(
        xp, prices, None, horizon=2, params=SolveParams(two_phase=False, time_limit_s=30)
    )
    assert len(plan.gws[0].squad) == 15


def test_run_optimize_refuses_degenerate_plan(tmp_path: Path, monkeypatch: Any) -> None:
    from fplai import pipeline
    from fplai.optimizer.plans import Recommendation

    rows = [
        {
            "season": SEASON,
            "gw": 1,
            "player_code": 1000 + i,
            "xp": 5.0,
            "q0": 0.05,
            "price": 60,
            "position": "MID",
            "team_code": 1 + i,
            "web_name": f"p{i}",
        }
        for i in range(20)
    ]
    pred = pd.DataFrame(rows)
    pred.to_parquet(tmp_path / "predictions_gw.parquet", index=False)

    import datetime as dt

    degenerate = Recommendation(
        action="initial-squad",
        season=SEASON,
        gw=1,
        as_of=dt.datetime(2026, 7, 24, tzinfo=dt.UTC),
        expected_points=6.0,  # far under 0.4 * (11 * 5.0)
        rationale=[],
    )
    from fplai.optimizer import plans as plans_mod

    monkeypatch.setattr(plans_mod, "build_recommendation", lambda *a, **k: degenerate)
    monkeypatch.setattr(pipeline, "_load_chip_sim_report", lambda *a, **k: None)
    with pytest.raises(RuntimeError, match="degenerate plan"):
        pipeline.run_optimize(
            season=SEASON,
            gw=1,
            run_chips=False,
            run_stability=False,
            processed_dir=tmp_path,
            out_dir=tmp_path,
        )
    assert not (tmp_path / "recommendation.json").exists()


def test_run_optimize_gate_allows_healthy_plan(tmp_path: Path, monkeypatch: Any) -> None:
    from fplai import pipeline
    from fplai.optimizer.plans import Recommendation

    rows = [
        {
            "season": SEASON,
            "gw": 1,
            "player_code": 1000 + i,
            "xp": 5.0,
            "q0": 0.05,
            "price": 60,
            "position": "MID",
            "team_code": 1 + i,
            "web_name": f"p{i}",
        }
        for i in range(20)
    ]
    pd.DataFrame(rows).to_parquet(tmp_path / "predictions_gw.parquet", index=False)

    import datetime as dt

    healthy = Recommendation(
        action="initial-squad",
        season=SEASON,
        gw=1,
        as_of=dt.datetime(2026, 7, 24, tzinfo=dt.UTC),
        expected_points=40.0,
        rationale=[],
    )
    from fplai.optimizer import plans as plans_mod

    monkeypatch.setattr(plans_mod, "build_recommendation", lambda *a, **k: healthy)
    monkeypatch.setattr(pipeline, "_load_chip_sim_report", lambda *a, **k: None)
    rec = pipeline.run_optimize(
        season=SEASON,
        gw=1,
        run_chips=False,
        run_stability=False,
        processed_dir=tmp_path,
        out_dir=tmp_path,
    )
    assert rec.expected_points == 40.0
    written = json.loads((tmp_path / "recommendation.json").read_text())
    assert written["expected_points"] == 40.0
