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
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    order: list[str] = []
    monkeypatch.setattr(
        pipeline, "run_snapshot", lambda: order.append("snapshot") or _state(next_gw=None)
    )
    monkeypatch.setattr(pipeline, "_refresh_pulls", lambda state: order.append("pulls"))
    monkeypatch.setattr(pipeline, "run_build", lambda: order.append("build") or {})

    pipeline.run_refresh()  # must not raise SystemExit

    assert order == ["snapshot", "pulls", "build"]
    out = capsys.readouterr().out
    for stage in ("train", "predict", "optimize"):
        assert f"{stage}" in out and "not yet implemented" in out


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


def test_cli_refresh_exit_zero_with_stub_model_stages(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pipeline, "run_snapshot", lambda: _state(next_gw=None))
    monkeypatch.setattr(pipeline, "_refresh_pulls", lambda state: None)
    monkeypatch.setattr(pipeline, "run_build", lambda: {})
    result = runner.invoke(app, ["refresh"])
    assert result.exit_code == 0
    assert "not yet implemented" in result.output


def test_cli_train_stub_exits_nonzero() -> None:
    result = runner.invoke(app, ["train"])
    assert result.exit_code == 1
