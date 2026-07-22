"""Offline tests for the optimizer/backtest pipeline wiring (no network, no MILP).

``plans.build_recommendation`` / ``plans.chip_ev_curves`` are monkeypatched with
deterministic stubs (their documented test seams); the real MILP path is covered
by the optimizer's own tests and by the manual `fplai optimize` run in STATUS.md.
"""

from __future__ import annotations

import datetime as dt
import json
import sys
import types
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from typer.testing import CliRunner

from fplai import pipeline
from fplai.cli import app
from fplai.optimizer import plans
from fplai.optimizer import state as state_mod
from fplai.optimizer.plans import DreamTeam, Recommendation
from fplai.optimizer.state import OwnedPlayer, SquadState

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
    _pred_gw_frame().to_parquet(processed / "predictions_gw.parquet", index=False)


def _stub_rec(gw: int = 34, action: str = "initial-squad") -> Recommendation:
    return Recommendation(
        season=SEASON,
        gw=gw,
        as_of=dt.datetime(2026, 7, 22, tzinfo=dt.UTC),
        action=action,
        squad=list(CODES),
        lineup=list(CODES),
        expected_points=50.0,
        objective=200.0,
        dream_team=DreamTeam(
            season=SEASON,
            gw=gw,
            squad=list(CODES),
            lineup=list(CODES),
            bench_order=[],
            captain=106,
            vice=105,
            expected_points=55.0,
            total_cost=990,
        ),
        rationale=["stub bullet"],
    )


class _BuildRecSpy:
    """Records every ``build_recommendation`` call and returns a canned result."""

    def __init__(self, rec: Recommendation) -> None:
        self.rec = rec
        self.calls: list[dict[str, Any]] = []

    def __call__(
        self, state: Any, xp: pd.DataFrame, prices: pd.DataFrame, **kwargs: Any
    ) -> Recommendation:
        self.calls.append({"state": state, "xp": xp, "prices": prices, **kwargs})
        return self.rec


def _fake_state(current_gw: int) -> SquadState:
    squad = [
        OwnedPlayer(player_code=c, purchase_price=50, current_price=50) for c in CODES
    ]
    return SquadState(
        season=SEASON,
        current_gw=current_gw,
        squad=squad,
        bank=10,
        free_transfers=2,
        chips_available=["bb2", "tc2"],
    )


# ---------------------------------------------------------------------------
# run_optimize: None-state (initial-squad) demo path + artifacts
# ---------------------------------------------------------------------------


def test_run_optimize_none_state_writes_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_predictions(tmp_path)
    spy = _BuildRecSpy(_stub_rec())
    monkeypatch.setattr(plans, "build_recommendation", spy)

    rec = pipeline.run_optimize(processed_dir=tmp_path)

    assert rec.action == "initial-squad"
    call = spy.calls[0]
    assert call["state"] is None
    assert call["horizon"] == 5  # min(settings 8, 5 available GWs)
    assert sorted(call["xp"]["gw"].unique().tolist()) == list(GWS)
    prices = call["prices"]
    assert {"player_code", "price", "position", "team_code", "web_name"} <= set(prices.columns)
    assert sorted(prices["player_code"]) == list(CODES)
    assert prices["price"].dtype.kind in "iu"

    saved = json.loads((tmp_path / "recommendation.json").read_text())
    assert saved["action"] == "initial-squad" and saved["gw"] == 34
    dream = json.loads((tmp_path / "dream_team.json").read_text())
    assert dream["gw"] == 34 and len(dream["squad"]) == len(CODES)
    out = capsys.readouterr().out
    assert "demo" in out  # no fixtures.parquet -> pre-launch demo note


def test_run_optimize_demo_window_and_horizon_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_predictions(tmp_path)
    spy = _BuildRecSpy(_stub_rec(gw=36))
    monkeypatch.setattr(plans, "build_recommendation", spy)

    pipeline.run_optimize(None, SEASON, 36, processed_dir=tmp_path)
    call = spy.calls[0]
    assert int(call["xp"]["gw"].min()) == 36
    assert call["horizon"] == 3  # GWs 36..38 available

    pipeline.run_optimize(None, SEASON, 36, horizon=2, processed_dir=tmp_path)
    assert spy.calls[1]["horizon"] == 2  # explicit cap wins when smaller


def test_run_optimize_errors_cleanly(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(FileNotFoundError, match="fplai predict"):
        pipeline.run_optimize(processed_dir=tmp_path)
    _write_predictions(tmp_path)
    monkeypatch.setattr(plans, "build_recommendation", _BuildRecSpy(_stub_rec()))
    with pytest.raises(ValueError, match="GW>=39"):
        pipeline.run_optimize(None, SEASON, 39, processed_dir=tmp_path)
    with pytest.raises(ValueError, match="together"):
        pipeline.run_optimize(None, SEASON, None, processed_dir=tmp_path)


# ---------------------------------------------------------------------------
# run_optimize: entry-state path + fallbacks
# ---------------------------------------------------------------------------


def test_run_optimize_entry_state_drives_the_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_predictions(tmp_path)
    spy = _BuildRecSpy(_stub_rec(gw=35, action="transfer"))
    monkeypatch.setattr(plans, "build_recommendation", spy)
    fake = _fake_state(current_gw=35)
    monkeypatch.setattr(state_mod, "from_entry", lambda entry_id: fake)

    pipeline.run_optimize(4242, processed_dir=tmp_path)
    call = spy.calls[0]
    assert call["state"] is fake
    assert int(call["xp"]["gw"].min()) == 35  # window starts at the entry's next GW
    assert call["horizon"] == 4  # GWs 35..38


def test_run_optimize_falls_back_when_from_entry_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_predictions(tmp_path)
    spy = _BuildRecSpy(_stub_rec())
    monkeypatch.setattr(plans, "build_recommendation", spy)

    def boom(entry_id: int) -> SquadState:
        raise RuntimeError("api down")

    monkeypatch.setattr(state_mod, "from_entry", boom)
    pipeline.run_optimize(4242, processed_dir=tmp_path)
    assert spy.calls[0]["state"] is None
    out = " ".join(capsys.readouterr().out.split())  # undo rich's line wrapping
    assert "falling back to initial-squad" in out


def test_run_optimize_falls_back_on_uncovered_entry_gw(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_predictions(tmp_path)
    spy = _BuildRecSpy(_stub_rec())
    monkeypatch.setattr(plans, "build_recommendation", spy)
    monkeypatch.setattr(state_mod, "from_entry", lambda entry_id: _fake_state(current_gw=10))

    pipeline.run_optimize(4242, processed_dir=tmp_path)
    assert spy.calls[0]["state"] is None  # GW10 not in the predictions window
    assert "do not cover" in " ".join(capsys.readouterr().out.split())


def test_run_optimize_uses_settings_entry_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fplai import config

    _write_predictions(tmp_path)
    monkeypatch.setattr(plans, "build_recommendation", _BuildRecSpy(_stub_rec()))
    monkeypatch.setattr(config.settings, "entry_id", 777)
    seen: list[int] = []

    def record(entry_id: int) -> SquadState:
        seen.append(entry_id)
        raise RuntimeError("stop here")  # fall back to None-state after recording

    monkeypatch.setattr(state_mod, "from_entry", record)
    pipeline.run_optimize(processed_dir=tmp_path)
    assert seen == [777]


# ---------------------------------------------------------------------------
# prices re-derivation + chip-curve capture
# ---------------------------------------------------------------------------


def test_optimizer_prices_rederive_from_player_gw_without_lookahead(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_predictions(tmp_path)
    pd.DataFrame(
        {
            "season": [SEASON] * 3,
            "gw": [33, 34, 38],
            "player_code": [101, 101, 101],
            "team_code": [100] * 3,
            "position": ["GKP"] * 3,
            "value": [59, 60, 99],
        }
    ).to_parquet(tmp_path / "player_gw.parquet", index=False)
    spy = _BuildRecSpy(_stub_rec())
    monkeypatch.setattr(plans, "build_recommendation", spy)

    pipeline.run_optimize(None, SEASON, 34, processed_dir=tmp_path)
    prices = spy.calls[0]["prices"].set_index("player_code")
    assert prices.loc[101, "price"] == 60  # latest player_gw value at/before GW34
    assert prices.loc[102, "price"] == 45  # not in player_gw -> predictions context


def test_run_optimize_captures_and_writes_chip_curves(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_predictions(tmp_path)
    curves = pd.DataFrame(
        {"chip": ["bb2"], "gw": [34], "objective": [100.0], "delta_vs_no_chip": [1.5]}
    )

    def fake_curves(*args: Any, **kwargs: Any) -> pd.DataFrame:
        return curves

    monkeypatch.setattr(plans, "chip_ev_curves", fake_curves)

    def fake_build(
        state: Any, xp: pd.DataFrame, prices: pd.DataFrame, **kwargs: Any
    ) -> Recommendation:
        plans.chip_ev_curves(xp, prices, state, ["bb2"], range(34, 39))
        return _stub_rec()

    monkeypatch.setattr(plans, "build_recommendation", fake_build)
    pipeline.run_optimize(processed_dir=tmp_path)

    saved = pd.read_parquet(tmp_path / "chip_curves.parquet")
    assert saved["chip"].tolist() == ["bb2"]
    assert plans.chip_ev_curves is fake_curves  # the capture wrapper was restored


# ---------------------------------------------------------------------------
# run_backtest wiring (harness is a stage-4 module; wired lazily)
# ---------------------------------------------------------------------------


def test_run_backtest_without_harness_raises_clear_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "fplai.backtest", None)  # force ImportError
    with pytest.raises(RuntimeError, match="not available yet"):
        pipeline.run_backtest(2025)


def test_run_backtest_calls_harness_run(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, Any] = {}
    harness = types.ModuleType("fplai.backtest.harness")

    def run(*, season: int, gws: list[int] | None = None) -> dict[str, int]:
        calls["season"] = season
        calls["gws"] = gws
        return {"season": season}

    harness.run = run  # type: ignore[attr-defined]
    pkg = types.ModuleType("fplai.backtest")
    pkg.harness = harness  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "fplai.backtest", pkg)
    monkeypatch.setitem(sys.modules, "fplai.backtest.harness", harness)

    assert pipeline.run_backtest(2025, [30, 31]) == {"season": 2025}
    assert calls == {"season": 2025, "gws": [30, 31]}


def test_parse_gws() -> None:
    assert pipeline.parse_gws(None) is None
    assert pipeline.parse_gws(" ") is None
    assert pipeline.parse_gws("30..32,38") == [30, 31, 32, 38]
    for bad in ("0", "39", "x", "8..3"):
        with pytest.raises(ValueError):
            pipeline.parse_gws(bad)


# ---------------------------------------------------------------------------
# CLI smoke (typer runner)
# ---------------------------------------------------------------------------


def test_cli_optimize_passes_args(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[tuple[Any, ...]] = []

    def fake(
        entry_id: int | None = None,
        season: int | None = None,
        gw: int | None = None,
        *,
        horizon: int | None = None,
        run_chips: bool = True,
        run_stability: bool = True,
        stability_n: int = 30,
        **kwargs: Any,
    ) -> None:
        seen.append((entry_id, season, gw, horizon, run_chips, run_stability, stability_n))

    monkeypatch.setattr(pipeline, "run_optimize", fake)
    result = runner.invoke(
        app,
        [
            "optimize",
            "--entry-id", "42",
            "--season", "2025",
            "--gw", "34",
            "--horizon", "3",
            "--no-chips",
            "--no-stability",
            "--stability-n", "10",
        ],
    )
    assert result.exit_code == 0
    assert seen == [(42, 2025, 34, 3, False, False, 10)]


def test_cli_optimize_rejects_half_specified_window() -> None:
    result = runner.invoke(app, ["optimize", "--season", "2025"])
    assert result.exit_code != 0


def test_cli_optimize_reports_missing_predictions(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake(*args: Any, **kwargs: Any) -> None:
        raise FileNotFoundError("predictions_gw.parquet is missing")

    monkeypatch.setattr(pipeline, "run_optimize", fake)
    result = runner.invoke(app, ["optimize"])
    assert result.exit_code == 1
    assert "missing" in result.output


def test_cli_backtest_passes_parsed_gws(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[tuple[int, list[int] | None]] = []
    monkeypatch.setattr(
        pipeline, "run_backtest", lambda season, gws=None: seen.append((season, gws))
    )
    result = runner.invoke(app, ["backtest", "--season", "2025", "--gws", "30..32"])
    assert result.exit_code == 0
    assert seen == [(2025, [30, 31, 32])]


def test_cli_backtest_missing_harness_exits_nonzero(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "fplai.backtest", None)
    result = runner.invoke(app, ["backtest", "--season", "2025"])
    assert result.exit_code == 1
    assert "not available" in result.output


def test_cli_backtest_rejects_bad_gws_spec(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        pipeline, "run_backtest", lambda *a, **k: pytest.fail("must not run on bad spec")
    )
    result = runner.invoke(app, ["backtest", "--season", "2025", "--gws", "39"])
    assert result.exit_code != 0


# ---------------------------------------------------------------------------
# refresh: optimize is best-effort; launch watch printed prominently
# ---------------------------------------------------------------------------


def test_refresh_runs_optimize_best_effort_and_prints_launch_watch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from fplai import config
    from fplai.data.fpl_api import SeasonState

    state = SeasonState(
        season=2025,
        is_live_2026_27=False,
        current_gw=38,
        next_gw=None,
        next_deadline_utc=None,
        total_players=1,
        static_content_url="x",
    )
    order: list[str] = []
    monkeypatch.setattr(pipeline, "run_snapshot", lambda: state)
    monkeypatch.setattr(pipeline, "_refresh_pulls", lambda s: None)
    monkeypatch.setattr(pipeline, "run_build", lambda: {})
    models = tmp_path / "models"
    models.mkdir()
    (models / "manifest.json").write_text("{}")
    monkeypatch.setattr(config, "MODELS_DIR", models)
    monkeypatch.setattr(pipeline, "run_predict", lambda: order.append("predict"))

    def opt_boom() -> None:
        raise RuntimeError("optimize blew up")

    monkeypatch.setattr(pipeline, "run_optimize", opt_boom)
    pipeline.run_refresh()  # must not raise despite the optimize failure

    out = " ".join(capsys.readouterr().out.split())  # undo rich's line wrapping
    assert order == ["predict"]
    assert "optimize blew up" in out
    assert "launch watch" in out
    assert "NOT LIVE" in out
