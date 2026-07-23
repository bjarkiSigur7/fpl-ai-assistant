"""Offline tests for the pipeline orchestration and CLI wiring (no network)."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from fplai import pipeline
from fplai.cli import app
from fplai.data.fpl_api import SeasonState

runner = CliRunner()


def _state(*, season: int = 2025, next_gw: int | None = None) -> SeasonState:
    return SeasonState(
        season=season,
        is_live_2026_27=season >= 2026,
        current_gw=38,
        next_gw=next_gw,
        next_deadline_utc=dt.datetime(2026, 8, 21, 17, 30, tzinfo=dt.UTC),
        total_players=13_000_000,
        static_content_url=f"https://fantasy.premierleague.com/dist/{season}_{(season + 1) % 100}",
    )


# ---------------------------------------------------------------------------
# parse_seasons
# ---------------------------------------------------------------------------


def test_parse_seasons_none_and_empty() -> None:
    assert pipeline.parse_seasons(None) is None
    assert pipeline.parse_seasons("") is None
    assert pipeline.parse_seasons("  ") is None


def test_parse_seasons_singles_and_ranges() -> None:
    assert pipeline.parse_seasons("2024") == [2024]
    assert pipeline.parse_seasons("2024,2025") == [2024, 2025]
    assert pipeline.parse_seasons("2016..2018") == [2016, 2017, 2018]
    assert pipeline.parse_seasons("2016..2017, 2025") == [2016, 2017, 2025]
    # duplicates collapse, output sorted
    assert pipeline.parse_seasons("2025,2024..2025") == [2024, 2025]


@pytest.mark.parametrize("bad", ["banana", "2018..2016", "2015", "2027", "20a4", "2016..x"])
def test_parse_seasons_rejects_bad_specs(bad: str) -> None:
    with pytest.raises(ValueError):
        pipeline.parse_seasons(bad)


# ---------------------------------------------------------------------------
# run_build: partial-season builds and odds skip logic
# ---------------------------------------------------------------------------


def _write_parquet(path: Path, rows: int) -> Path:
    pd.DataFrame({"x": range(rows)}).to_parquet(path, index=False)
    return path


def test_run_build_skips_odds_without_raw_e0(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fplai.data import build, crosswalk, football_data

    tables = {name: _write_parquet(tmp_path / f"{name}.parquet", 3) for name in ("teams", "players")}
    monkeypatch.setattr(crosswalk, "resolve_seasons", lambda seasons: sorted(seasons or [2024]))
    monkeypatch.setattr(build, "build_all", lambda seasons: dict(tables))
    monkeypatch.setattr(football_data, "e0_path", lambda season: tmp_path / f"missing/{season}.csv")
    called: list[object] = []
    monkeypatch.setattr(football_data, "build_odds_table", lambda *a, **k: called.append(a))

    paths = pipeline.run_build([2024])
    assert "odds" not in paths
    assert called == []  # never invoked without raw E0 files
    assert set(paths) == {"teams", "players"}


def test_run_build_builds_odds_for_available_seasons_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fplai.data import build, crosswalk, football_data

    tables = {"teams": _write_parquet(tmp_path / "teams.parquet", 40)}
    odds_path = _write_parquet(tmp_path / "odds.parquet", 380)
    e0 = tmp_path / "2024" / "E0.csv"
    e0.parent.mkdir()
    e0.write_text("Div,\n")

    monkeypatch.setattr(crosswalk, "resolve_seasons", lambda seasons: sorted(seasons))
    monkeypatch.setattr(build, "build_all", lambda seasons: dict(tables))
    monkeypatch.setattr(
        football_data,
        "e0_path",
        lambda season: e0 if season == 2024 else tmp_path / f"missing/{season}.csv",
    )
    seen: list[list[int]] = []

    def fake_build_odds(seasons: list[int]) -> Path:
        seen.append(list(seasons))
        return odds_path

    monkeypatch.setattr(football_data, "build_odds_table", fake_build_odds)

    paths = pipeline.run_build([2024, 2025])
    assert paths["odds"] == odds_path
    assert seen == [[2024]]  # 2025 has no raw E0.csv, so it is excluded


# ---------------------------------------------------------------------------
# run_refresh: sequencing + exit-zero guarantee for stub stages
# ---------------------------------------------------------------------------


def test_run_refresh_sequences_and_does_not_exit_nonzero(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    from fplai import config

    order: list[str] = []
    monkeypatch.setattr(
        pipeline, "run_snapshot", lambda: order.append("snapshot") or _state(next_gw=None)
    )
    monkeypatch.setattr(pipeline, "_refresh_pulls", lambda state: order.append("pulls"))
    monkeypatch.setattr(pipeline, "run_build", lambda: order.append("build") or {})
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "models")  # no manifest -> skip

    pipeline.run_refresh()  # must not raise SystemExit

    assert order == ["snapshot", "pulls", "build"]
    out = capsys.readouterr().out
    for stage in ("train", "predict", "optimize"):
        assert f"{stage}" in out
    assert "skip" in out  # model stages degrade without artifacts


def test_run_refresh_predicts_when_artifacts_exist(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from fplai import config

    order: list[str] = []
    monkeypatch.setattr(pipeline, "run_snapshot", lambda: _state(next_gw=None))
    monkeypatch.setattr(pipeline, "_refresh_pulls", lambda state: None)
    monkeypatch.setattr(pipeline, "run_build", lambda: {})
    models = tmp_path / "models"
    models.mkdir()
    (models / "manifest.json").write_text("{}")
    monkeypatch.setattr(config, "MODELS_DIR", models)
    monkeypatch.setattr(pipeline, "run_predict", lambda: order.append("predict"))
    monkeypatch.setattr(pipeline, "run_optimize", lambda: order.append("optimize"))

    pipeline.run_refresh()
    assert order == ["predict", "optimize"]


def test_run_refresh_stays_exit_zero_when_predict_fails(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    from fplai import config

    monkeypatch.setattr(pipeline, "run_snapshot", lambda: _state(next_gw=None))
    monkeypatch.setattr(pipeline, "_refresh_pulls", lambda state: None)
    monkeypatch.setattr(pipeline, "run_build", lambda: {})
    models = tmp_path / "models"
    models.mkdir()
    (models / "manifest.json").write_text("{}")
    monkeypatch.setattr(config, "MODELS_DIR", models)

    def boom() -> None:
        raise RuntimeError("predict blew up")

    monkeypatch.setattr(pipeline, "run_predict", boom)
    monkeypatch.setattr(pipeline, "run_optimize", lambda: None)
    pipeline.run_refresh()  # must not raise
    assert "predict blew up" in capsys.readouterr().out


def test_refresh_pulls_survive_aux_source_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    """A single upstream outage must not break the refresh data portion."""
    from fplai.data import clubelo, football_data, understat, vaastav

    def boom(*args: object, **kwargs: object) -> None:
        raise RuntimeError("upstream down")

    monkeypatch.setattr(vaastav, "download_season", boom)
    monkeypatch.setattr(football_data, "download_e0", boom)
    monkeypatch.setattr(understat.UnderstatClient, "fetch_league_season", boom)
    monkeypatch.setattr(clubelo.EloClient, "pl_snapshot", boom)

    pipeline._refresh_pulls(_state(season=2025, next_gw=1))  # must not raise


def test_refresh_pulls_force_only_in_season(monkeypatch: pytest.MonkeyPatch) -> None:
    from fplai.data import clubelo, football_data, understat, vaastav

    calls: dict[str, object] = {}
    monkeypatch.setattr(
        vaastav,
        "download_season",
        lambda season, force=False: calls.__setitem__("vaastav", (season, force)),
    )
    monkeypatch.setattr(
        football_data,
        "download_e0",
        lambda season, force=False: calls.__setitem__("e0", (season, force)),
    )
    monkeypatch.setattr(
        understat.UnderstatClient,
        "fetch_league_season",
        lambda self, year, force=False: calls.__setitem__("understat", (year, force)),
    )
    monkeypatch.setattr(
        clubelo.EloClient, "pl_snapshot", lambda self, date: calls.__setitem__("elo", date)
    )

    pipeline._refresh_pulls(_state(season=2025, next_gw=None))  # season finished
    assert calls["vaastav"] == (2025, False)
    assert calls["e0"] == (2025, False)
    assert calls["understat"] == (2025, False)

    pipeline._refresh_pulls(_state(season=2025, next_gw=3))  # season in progress
    assert calls["vaastav"] == (2025, True)
    assert calls["e0"] == (2025, True)
    assert calls["understat"] == (2025, True)


# ---------------------------------------------------------------------------
# CLI wiring
# ---------------------------------------------------------------------------


def test_cli_snapshot_invokes_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    called: list[str] = []
    monkeypatch.setattr(pipeline, "run_snapshot", lambda: called.append("snap") or _state())
    result = runner.invoke(app, ["snapshot"])
    assert result.exit_code == 0
    assert called == ["snap"]


def test_cli_backfill_passes_parsed_seasons(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[object] = []
    monkeypatch.setattr(pipeline, "run_backfill", lambda seasons: seen.append(seasons))
    result = runner.invoke(app, ["backfill", "--seasons", "2024..2025"])
    assert result.exit_code == 0
    assert seen == [[2024, 2025]]


def test_cli_build_default_seasons_is_none(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[object] = []
    monkeypatch.setattr(pipeline, "run_build", lambda seasons: seen.append(seasons) or {})
    result = runner.invoke(app, ["build"])
    assert result.exit_code == 0
    assert seen == [None]


def test_cli_bad_seasons_spec_is_a_usage_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        pipeline, "run_backfill", lambda seasons: pytest.fail("must not run on bad spec")
    )
    result = runner.invoke(app, ["backfill", "--seasons", "202x"])
    assert result.exit_code != 0


def test_cli_refresh_exit_zero_without_artifacts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from fplai import config

    monkeypatch.setattr(pipeline, "run_snapshot", lambda: _state(next_gw=None))
    monkeypatch.setattr(pipeline, "_refresh_pulls", lambda state: None)
    monkeypatch.setattr(pipeline, "run_build", lambda: {})
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "models")
    result = runner.invoke(app, ["refresh"])
    assert result.exit_code == 0
    assert "skip" in result.output


def test_cli_train_passes_options(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[object] = []
    monkeypatch.setattr(
        pipeline,
        "run_train",
        lambda seasons, before_season=None, before_gw=None: seen.append(
            (seasons, before_season, before_gw)
        ),
    )
    result = runner.invoke(
        app,
        ["train", "--seasons", "2024..2025", "--before-season", "2025", "--before-gw", "30"],
    )
    assert result.exit_code == 0
    assert seen == [([2024, 2025], 2025, 30)]


def test_cli_train_rejects_before_gw_without_before_season() -> None:
    result = runner.invoke(app, ["train", "--before-gw", "30"])
    assert result.exit_code != 0


def test_cli_predict_passes_backtest_args(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[object] = []
    monkeypatch.setattr(
        pipeline,
        "run_predict",
        lambda season=None, gw=None, horizon=None, use_odds=True: seen.append(
            (season, gw, horizon, use_odds)
        ),
    )
    result = runner.invoke(
        app, ["predict", "--season", "2025", "--gw", "30", "--horizon", "9", "--no-odds"]
    )
    assert result.exit_code == 0
    assert seen == [(2025, 30, 9, False)]


def test_cli_predict_rejects_gw_without_season() -> None:
    result = runner.invoke(app, ["predict", "--gw", "30"])
    assert result.exit_code != 0


def test_run_predict_rejects_half_specified_backtest() -> None:
    with pytest.raises(ValueError, match="together"):
        pipeline.run_predict(2025, None)


# ---------------------------------------------------------------------------
# model-stage helpers
# ---------------------------------------------------------------------------


def test_cutoff_mask_strictly_before() -> None:
    df = pd.DataFrame({"season": [2024, 2025, 2025, 2025], "gw": [38, 29, 30, 31]})
    assert pipeline._cutoff_mask(df, None, None).all()
    assert pipeline._cutoff_mask(df, 2025, None).tolist() == [True, False, False, False]
    assert pipeline._cutoff_mask(df, 2025, 30).tolist() == [True, True, False, False]


def test_team_train_end_is_first_excluded_kickoff() -> None:
    fx = pd.DataFrame(
        {
            "season": [2025, 2025, 2025],
            "gw": [29, 30, 31],
            "kickoff_utc": pd.to_datetime(
                ["2026-03-01", "2026-03-08", "2026-03-15"], utc=True
            ),
        }
    )
    assert pipeline._team_train_end(fx, 2025, 30) == pd.Timestamp("2026-03-08", tz="UTC")
    assert pipeline._team_train_end(fx, None, None) is None
    assert pipeline._team_train_end(fx, 2026, None) is None


def _mini_player_match() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "season": [2025, 2025, 2025, 2024],
            "gw": [37, 38, 38, 38],
            "fpl_fixture_id": [370, 380, 380, 999],
            "player_code": [1, 1, 2, 3],
            "fpl_element_id": [11, 11, 22, 33],
            "team_code": [100, 100, 200, 300],
            "opponent_code": [200, 200, 100, 100],
            "was_home": [True, True, False, True],
            "position": ["MID", "MID", "DEF", "FWD"],
            "price": [80, 81, 45, 60],
            "minutes": [90, 90, 60, 45],
            "starts": [1, 1, 1, 0],
            "empty_stadium": [False] * 4,
            "void_gw": [False] * 4,
            "subs_regime": [5] * 4,
            "stint_id": [0] * 4,
        }
    )


def test_future_player_match_rolls_rosters_forward() -> None:
    pm = _mini_player_match()
    target = pd.DataFrame(
        {
            "season": [2026],
            "gw": [1],
            "fpl_fixture_id": [1],
            "home_team_code": [100],
            "away_team_code": [200],
        }
    )
    synth = pipeline._future_player_match(pm, target)
    assert set(synth["player_code"]) == {1, 2}  # player 3's team is not playing
    assert list(synth.columns) == list(pm.columns)
    row1 = synth[synth["player_code"] == 1].iloc[0]
    assert row1["season"] == 2026 and row1["gw"] == 1 and row1["fpl_fixture_id"] == 1
    assert row1["price"] == 81 and bool(row1["was_home"])  # latest row rolls forward
    assert pd.isna(row1["minutes"])  # outcomes unknown
    row2 = synth[synth["player_code"] == 2].iloc[0]
    assert not bool(row2["was_home"]) and row2["opponent_code"] == 100


def test_future_player_match_drops_long_departed_players() -> None:
    pm = _mini_player_match()
    target = pd.DataFrame(
        {
            "season": [2026],
            "gw": [1],
            "fpl_fixture_id": [2],
            "home_team_code": [300],
            "away_team_code": [200],
        }
    )
    synth = pipeline._future_player_match(pm, target)
    # player 3 last seen 2024 < 2026-1 -> excluded; team 300 has no roster.
    assert set(synth["player_code"]) == {2}


def test_run_predict_live_mode_degrades_without_upcoming_fixtures(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    fixtures = pd.DataFrame(
        {
            "season": [2025],
            "gw": [38],
            "fpl_fixture_id": [380],
            "kickoff_utc": pd.to_datetime(["2026-05-24"], utc=True),
            "home_team_code": [100],
            "away_team_code": [200],
            "home_goals": [1.0],
            "away_goals": [0.0],
            "finished": [True],
            "void": [False],
        }
    )
    monkeypatch.setattr(
        pipeline, "load_processed_tables", lambda processed_dir=None: {"fixtures": fixtures}
    )
    # Pin the no-snapshot path: the repo's real raw dir may hold a live snapshot.
    monkeypatch.setattr(pipeline, "_live_context", lambda: None)
    assert pipeline.run_predict() is None
    assert "no upcoming fixtures" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# train + predict end-to-end on real processed data (auto-skips without data)
# ---------------------------------------------------------------------------

_PM_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "player_match.parquet"


@pytest.mark.skipif(not _PM_PATH.exists(), reason="processed tables not built")
def test_train_predict_roundtrip_on_real_data(tmp_path: Path) -> None:
    from fplai.models.assemble import COMPONENT_COLUMNS, XP_COLUMNS

    manifest = pipeline.run_train([2024, 2025], models_dir=tmp_path / "models")
    assert manifest["train_window"]["seasons"] == [2024, 2025]
    assert (tmp_path / "models" / "manifest.json").exists()

    paths = pipeline.run_predict(
        2025, 37, horizon=2, models_dir=tmp_path / "models", out_dir=tmp_path / "out"
    )
    assert paths is not None
    pred = pd.read_parquet(paths["predictions"])
    gw_pred = pd.read_parquet(paths["predictions_gw"])
    assert set(XP_COLUMNS) <= set(pred.columns)
    assert {"q0", "q1", "q2", "mu1", "mu2", "position", "price"} <= set(pred.columns)
    assert sorted(pred["gw"].unique()) == [37, 38]
    # components sum to xp; probabilities proper
    comp_sum = pred[list(COMPONENT_COLUMNS)].sum(axis=1)
    assert (comp_sum - pred["xp"]).abs().max() < 1e-6
    assert ((pred[["q0", "q1", "q2"]].sum(axis=1) - 1).abs() < 1e-6).all()
    # per-GW aggregation covers the same players
    assert {"xp", "q0", "web_name", "n_fixtures"} <= set(gw_pred.columns)
    assert len(gw_pred) == pred.groupby(["season", "gw", "player_code"]).ngroups
    # a mid-table xp sanity band: best player of a normal GW sits in ~[4, 20]
    best = gw_pred[gw_pred["gw"] == 37]["xp"].max()
    assert 4.0 < best < 25.0


@pytest.mark.skipif(not _PM_PATH.exists(), reason="processed tables not built")
def test_predict_live_mode_synthesizes_upcoming_fixtures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Simulate an in-season state: GW38 2025 not yet played -> live predict covers it."""
    pipeline.run_train([2024, 2025], before_season=2025, before_gw=38,
                       models_dir=tmp_path / "models")

    tables = pipeline.load_processed_tables()
    # Live refreshes persist upcoming 2026-27 fixtures into fixtures.parquet;
    # drop them so the simulated 2025 GW38 is the only upcoming window here.
    fx = tables["fixtures"]
    fx = fx[fx["season"] <= 2025].copy()
    mask = (fx["season"] == 2025) & (fx["gw"] == 38)
    assert mask.any()
    fx.loc[mask, "finished"] = False
    fx.loc[mask, ["home_goals", "away_goals"]] = pd.NA
    fx.loc[mask, "kickoff_utc"] = pd.Timestamp.now(tz="UTC") + pd.Timedelta("7D")
    pm = tables["player_match"]
    tables["fixtures"] = fx
    tables["player_match"] = pm[~((pm["season"] == 2025) & (pm["gw"] == 38))].copy()
    monkeypatch.setattr(pipeline, "load_processed_tables", lambda processed_dir=None: tables)
    # This test exercises the pre-relaunch roll-forward synthesizer; disable the
    # live 2026-27 snapshot the repo's raw dir now carries.
    monkeypatch.setattr(pipeline, "_live_context", lambda: None)

    paths = pipeline.run_predict(models_dir=tmp_path / "models", out_dir=tmp_path / "out")
    assert paths is not None
    pred = pd.read_parquet(paths["predictions"])
    assert (pred["season"] == 2025).all() and (pred["gw"] == 38).all()
    assert len(pred) > 400  # both squads of all 10 fixtures
    assert pred["xp"].between(-2, 25).all()
    # the synthesized pool must rank plausibly: xp correlates with price
    assert pred["xp"].corr(pred["price"].astype(float)) > 0.3
