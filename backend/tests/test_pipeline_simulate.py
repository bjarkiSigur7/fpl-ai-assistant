"""Offline tests for the season-simulation pipeline wiring (stage 6 integration).

``season_sim.simulate_chip_plans`` is monkeypatched with a stub returning a
hand-built (real) ``ChipSimReport``; the simulator itself is covered by
``test_season_sim.py`` and the sampler by ``test_sampler.py``. These tests pin
the integrator surface: ``fplai simulate`` artifacts, the recommendation.json
fold-in, ``fplai optimize`` consuming a fresh report, and the ``--through-gw``
prediction-window plumbing.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from typer.testing import CliRunner

from fplai import pipeline
from fplai.cli import app
from fplai.optimizer import plans
from fplai.optimizer import season_sim as season_sim_mod
from fplai.optimizer.plans import ChipAdvice, Recommendation
from fplai.optimizer.season_sim import (
    BackboneQuality,
    ChipGwStat,
    ChipSimReport,
    ChipVerdict,
    JointSchedule,
)

runner = CliRunner()

SEASON = 2025
GWS = (34, 35, 36, 37, 38)
CODES = (101, 102, 103, 104, 105, 106)
POSITIONS = {101: "GKP", 102: "DEF", 103: "DEF", 104: "MID", 105: "MID", 106: "FWD"}


def _pred_gw_frame() -> pd.DataFrame:
    rows = []
    for gw in GWS:
        for i, code in enumerate(CODES):
            rows.append(
                {
                    "season": SEASON,
                    "gw": gw,
                    "player_code": code,
                    "xp": 2.0 + 0.5 * i + 0.01 * gw,
                    "q0": 0.1,
                    "team_code": 100 + i,
                    "position": POSITIONS[code],
                    "price": 40 + 5 * i,
                    "web_name": f"P{code}",
                }
            )
    return pd.DataFrame(rows)


def _write_predictions(processed: Path) -> None:
    frame = _pred_gw_frame()
    frame.to_parquet(processed / "predictions_gw.parquet", index=False)
    fx = frame.assign(fpl_fixture_id=1000, was_home=True, opponent_code=999)
    fx.to_parquet(processed / "predictions.parquet", index=False)


def _report(*, window_first_gw: int = 34, window_last_gw: int = 38) -> ChipSimReport:
    gws = range(window_first_gw, window_last_gw + 1)
    stats = []
    for chip in ("bb2", "tc2"):
        for g in gws:
            best = 36 if chip == "bb2" else 37
            stats.append(
                ChipGwStat(
                    chip=chip,
                    gw=g,
                    xp_gain=5.0,
                    e_gain=8.0 if g == best else 4.0,
                    sd=5.0,
                    p_best_week=0.4 if g == best else 0.15,
                    p_beats_hold=0.9 if g == best else 0.3,
                    evaluated=True,
                    window_last_gw=window_last_gw,
                )
            )
    return ChipSimReport(
        season=SEASON,
        window_first_gw=window_first_gw,
        window_last_gw=window_last_gw,
        n_rollouts=200,
        seed=0,
        chips=["bb2", "tc2"],
        stats=stats,
        verdicts=[
            ChipVerdict(
                chip="bb2", recommended_gw=36, e_gain=8.0, sd=5.0,
                confidence=0.4, p_beats_hold=0.9,
            ),
            ChipVerdict(
                chip="tc2", recommended_gw=37, e_gain=8.0, sd=5.0,
                confidence=0.4, p_beats_hold=0.9,
            ),
        ],
        schedules=[JointSchedule(placements={"bb2": 36, "tc2": 37}, e_total_gain=16.0, p_best=0.5)],
        backbone=BackboneQuality(
            objective=100.0, gap=0.01, solve_seconds=1.0, status="optimal",
            horizon=len(list(gws)),
        ),
        assumptions=list(season_sim_mod.ASSUMPTIONS),
        appearance_source="sample_minutes",
        n_segment_solves=4,
        sim_seconds=2.0,
    )


def _rec(gw: int = 34) -> Recommendation:
    return Recommendation(
        season=SEASON,
        gw=gw,
        as_of=dt.datetime(2026, 7, 22, tzinfo=dt.UTC),
        action="hold",
        squad=list(CODES),
        lineup=list(CODES),
        expected_points=50.0,
        objective=200.0,
        chip_advice=[
            ChipAdvice(chip="bb2", verdict="play", planned_gw=34, ev_now=6.0,
                       best_gw=34, best_ev=6.0),
            ChipAdvice(chip="tc2", verdict="hold", best_gw=35, best_ev=3.0),
        ],
        rationale=[
            "Hold: the optimal 5-GW plan makes 0 transfers this GW — bank the free transfer.",
            "Best future chip window: Triple Captain (tc2) in GW35 (+3.0 xP vs no chip) — "
            "hold for now.",
        ],
    )


# ---------------------------------------------------------------------------
# run_simulate: artifacts + fold-in
# ---------------------------------------------------------------------------


def test_run_simulate_writes_artifacts_and_folds_recommendation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_predictions(tmp_path)
    (tmp_path / "recommendation.json").write_text(_rec().model_dump_json(indent=2))
    report = _report()
    calls: list[dict[str, Any]] = []

    def stub(xp: pd.DataFrame, prices: pd.DataFrame, state: Any, **kwargs: Any) -> ChipSimReport:
        calls.append({"xp": xp, "prices": prices, "state": state, **kwargs})
        return report

    monkeypatch.setattr(season_sim_mod, "simulate_chip_plans", stub)

    got = pipeline.run_simulate(rollouts=200, seed=7, processed_dir=tmp_path)
    assert got is report

    call = calls[0]
    assert call["state"] is None
    # 2025 set-2 chip window runs to GW38; the next GW on disk is 34.
    assert list(call["window"]) == [34, 35, 36, 37, 38]
    assert call["n_rollouts"] == 200
    assert call["seed"] == 7
    assert {"player_code", "price", "position", "team_code"} <= set(call["prices"].columns)

    # v2 curve artifacts: chip_sim.parquet + chip_curves.parquet share the frame
    for name in ("chip_sim.parquet", "chip_curves.parquet"):
        frame = pd.read_parquet(tmp_path / name)
        assert {"sd", "p_best_week", "p_beats_hold", "n_rollouts"} <= set(frame.columns)
        assert (frame["n_rollouts"] == 200).all()
    # the durable report round-trips
    roundtrip = ChipSimReport.model_validate_json(
        (tmp_path / "chip_sim_report.json").read_text()
    )
    assert roundtrip == report

    # recommendation.json re-verdicted: sim overlays advice + probability bullets
    rec = Recommendation.model_validate(
        json.loads((tmp_path / "recommendation.json").read_text())
    )
    bb2 = next(a for a in rec.chip_advice if a.chip == "bb2")
    assert bb2.verdict == "hold"  # p_beats_hold at GW34 is 0.3 -> hold
    assert bb2.p_beats_hold == pytest.approx(0.3)
    assert bb2.recommended_gw == 36
    assert bb2.n_rollouts == 200
    assert any("now beats holding in 30%" in b for b in rec.rationale)
    assert not any(b.startswith("Best future chip window:") for b in rec.rationale)


def test_run_simulate_skips_fold_on_gw_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_predictions(tmp_path)
    stale = _rec(gw=30)
    (tmp_path / "recommendation.json").write_text(stale.model_dump_json(indent=2))
    monkeypatch.setattr(
        season_sim_mod, "simulate_chip_plans", lambda *a, **k: _report()
    )

    pipeline.run_simulate(rollouts=200, processed_dir=tmp_path)

    on_disk = json.loads((tmp_path / "recommendation.json").read_text())
    assert Recommendation.model_validate(on_disk) == stale  # untouched
    assert "not folding" in capsys.readouterr().out.replace("\n", " ")


def test_run_simulate_missing_predictions_raises_with_pointer(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="--through-gw"):
        pipeline.run_simulate(processed_dir=tmp_path)


def test_cli_simulate_fails_cleanly_without_predictions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fplai import config

    monkeypatch.setattr(config, "PROCESSED_DIR", tmp_path)
    result = runner.invoke(app, ["simulate"])
    assert result.exit_code == 1
    assert "simulate failed" in result.output


# ---------------------------------------------------------------------------
# run_optimize: consuming a fresh chip_sim_report.json
# ---------------------------------------------------------------------------


class _BuildRecSpy:
    def __init__(self, rec: Recommendation) -> None:
        self.rec = rec
        self.calls: list[dict[str, Any]] = []

    def __call__(
        self, state: Any, xp: pd.DataFrame, prices: pd.DataFrame, **kwargs: Any
    ) -> Recommendation:
        self.calls.append({"state": state, "xp": xp, "prices": prices, **kwargs})
        return self.rec


def test_run_optimize_consumes_matching_chip_sim_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_predictions(tmp_path)
    report = _report()
    (tmp_path / "chip_sim_report.json").write_text(report.model_dump_json())
    spy = _BuildRecSpy(_rec())
    monkeypatch.setattr(plans, "build_recommendation", spy)

    pipeline.run_optimize(processed_dir=tmp_path)

    chip_sim = spy.calls[0]["chip_sim"]
    assert isinstance(chip_sim, ChipSimReport)
    assert chip_sim.window_first_gw == 34
    # chip_curves.parquet comes from the sim report (v2), not the captured v1 curves
    curves = pd.read_parquet(tmp_path / "chip_curves.parquet")
    assert {"p_best_week", "p_beats_hold"} <= set(curves.columns)
    assert set(curves["chip"]) == {"bb2", "tc2"}


def test_run_optimize_ignores_stale_chip_sim_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_predictions(tmp_path)
    stale = _report(window_first_gw=35)  # window starts past the next GW on disk (34)
    (tmp_path / "chip_sim_report.json").write_text(stale.model_dump_json())
    spy = _BuildRecSpy(_rec())
    monkeypatch.setattr(plans, "build_recommendation", spy)

    pipeline.run_optimize(processed_dir=tmp_path)

    assert spy.calls[0]["chip_sim"] is None
    assert "stale" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# run_predict --through-gw plumbing (validation; live path needs real tables)
# ---------------------------------------------------------------------------


def test_run_predict_through_gw_rejects_backtest_mode() -> None:
    with pytest.raises(ValueError, match="live-mode"):
        pipeline.run_predict(2025, 30, through_gw=19)


def test_run_predict_through_gw_excludes_horizon() -> None:
    with pytest.raises(ValueError, match="not both"):
        pipeline.run_predict(horizon=8, through_gw=19)


def test_cli_predict_through_gw_conflict_is_a_usage_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = runner.invoke(app, ["predict", "--horizon", "8", "--through-gw", "19"])
    assert result.exit_code != 0
    assert "not both" in result.output


# ---------------------------------------------------------------------------
# plans.apply_sim_to_recommendation (the fold-in primitive)
# ---------------------------------------------------------------------------


def test_apply_sim_to_recommendation_overrides_and_is_idempotent() -> None:
    rec = _rec()
    report = _report()

    once = plans.apply_sim_to_recommendation(rec, report)
    bb2 = next(a for a in once.chip_advice if a.chip == "bb2")
    assert bb2.verdict == "hold"
    assert bb2.planned_gw == 34  # the plan's schedule is kept for context
    assert bb2.confidence is not None
    sim_bullets = [b for b in once.rationale if "now beats holding" in b]
    assert len(sim_bullets) == 2  # bb2 + tc2
    assert not any(b.startswith("Best future chip window:") for b in once.rationale)
    # non-chip bullets survive
    assert any(b.startswith("Hold:") for b in once.rationale)

    twice = plans.apply_sim_to_recommendation(once, report)
    assert twice.chip_advice == once.chip_advice
    assert twice.rationale == once.rationale
