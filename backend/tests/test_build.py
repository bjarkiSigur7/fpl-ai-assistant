"""Tests for fplai.data.build and fplai.data.crosswalk on a synthetic vaastav mirror.

The fixture tree under tests/fixtures/vaastav_synth/vaastav/ covers, with a handful
of rows: a DGW (two fixtures in one GW), the 2019-20 COVID event remap (39 -> 30)
with a stub-event row to drop, the 2022-23 void GW7, a mid-season club switch
(stint_id increment), GK -> GKP position normalization, manager-element filtering,
the rich-stats era (2016-17) including fixture reconstruction without fixtures.csv,
and the id_dict.csv Understat map.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
import pytest

from fplai.data import build, crosswalk

RAW_ROOT = Path(__file__).parent / "fixtures" / "vaastav_synth"
REAL_TEAMS_CSV = Path(__file__).parent / "fixtures" / "vaastav_real_teams_2025-26.csv"


@pytest.fixture(scope="module")
def teams_xw() -> pd.DataFrame:
    return crosswalk.build_teams_crosswalk(raw_root=RAW_ROOT)


@pytest.fixture(scope="module")
def fixtures_df(teams_xw: pd.DataFrame) -> pd.DataFrame:
    return build.build_fixtures(raw_root=RAW_ROOT, teams=teams_xw, write=False)


@pytest.fixture(scope="module")
def pm(fixtures_df: pd.DataFrame) -> pd.DataFrame:
    return build.build_player_match(raw_root=RAW_ROOT, fixtures=fixtures_df, write=False)


@pytest.fixture(scope="module")
def pgw(pm: pd.DataFrame) -> pd.DataFrame:
    return build.build_player_gw(player_match=pm, write=False)


def _one(df: pd.DataFrame, **criteria: object) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    for col, val in criteria.items():
        mask &= df[col] == val
    rows = df[mask]
    assert len(rows) == 1, f"expected 1 row for {criteria}, got {len(rows)}"
    return rows.iloc[0]


# --- crosswalk --------------------------------------------------------------------


def test_normalize_name() -> None:
    assert crosswalk.normalize_name("Ødegaard, Martin") == "odegaard martin"
    assert crosswalk.normalize_name("João Félix") == "joao felix"
    assert crosswalk.normalize_name("  N'Golo   Kanté ") == "n golo kante"


def test_teams_crosswalk_schema_and_backfill(teams_xw: pd.DataFrame) -> None:
    assert list(teams_xw.columns) == [
        "season",
        "fpl_team_id",
        "team_code",
        "name",
        "short_name",
        "understat_name",
        "clubelo_name",
        "footballdata_name",
    ]
    # 2016-17 has no teams.csv upstream: derived from players_raw + master_team_list,
    # short_name backfilled by stable team_code from later seasons' teams.csv.
    arsenal_2016 = _one(teams_xw, season=2016, fpl_team_id=1)
    assert arsenal_2016["team_code"] == 3
    assert arsenal_2016["name"] == "Arsenal"
    assert arsenal_2016["short_name"] == "ARS"
    bou_2016 = _one(teams_xw, season=2016, fpl_team_id=2)
    assert bou_2016["team_code"] == 91
    assert bou_2016["short_name"] == "BOU"
    assert teams_xw["understat_name"].isna().all()
    assert str(teams_xw["understat_name"].dtype) == "string"
    # per-season ids differ but team_code is stable: MCI is id 2 in 2020, id 3 in 2022
    assert _one(teams_xw, season=2020, team_code=43)["fpl_team_id"] == 2
    assert _one(teams_xw, season=2022, team_code=43)["fpl_team_id"] == 3


def test_teams_crosswalk_parses_real_2025_26_payload(tmp_path: Path) -> None:
    dest = tmp_path / "vaastav" / "2025-26" / "teams.csv"
    dest.parent.mkdir(parents=True)
    shutil.copy(REAL_TEAMS_CSV, dest)
    teams = crosswalk.build_teams_crosswalk([2025], raw_root=tmp_path)
    assert len(teams) == 20
    assert set(teams["fpl_team_id"]) == set(range(1, 21))
    assert "Sunderland" in set(teams["name"])
    assert _one(teams, name="Arsenal")["short_name"] == "ARS"


def test_players_crosswalk(tmp_path: Path) -> None:
    players = crosswalk.build_players_crosswalk(raw_root=RAW_ROOT)
    assert list(players.columns) == [
        "player_code",
        "web_name",
        "first_name",
        "second_name",
        "understat_id",
        "opta_code",
    ]
    assert set(players["player_code"]) == {500, 600, 700, 800, 850, 950, 960}
    assert 900 not in set(players["player_code"])  # 2024-25-style manager element
    assert str(players["understat_id"].dtype) == "Int64"
    assert _one(players, player_code=500)["understat_id"] == 111  # via id_dict.csv
    assert _one(players, player_code=600)["understat_id"] == 222
    assert pd.isna(_one(players, player_code=700)["understat_id"])  # no map shipped
    assert _one(players, player_code=950)["web_name"] == "Özil"
    assert players["opta_code"].isna().all()  # column absent in these season dumps


# --- fixtures ---------------------------------------------------------------------


def test_fixtures_schema(fixtures_df: pd.DataFrame) -> None:
    assert list(fixtures_df.columns) == list(build.FIXTURES_COLS)
    assert str(fixtures_df["kickoff_utc"].dtype) == "datetime64[ns, UTC]"
    assert str(fixtures_df["home_goals"].dtype) == "Int64"
    assert fixtures_df["finished"].dtype == bool
    assert fixtures_df["void"].dtype == bool


def test_fixtures_2019_remap(fixtures_df: pd.DataFrame) -> None:
    s2019 = fixtures_df[fixtures_df["season"] == 2019]
    assert set(s2019["gw"]) == {29, 30}  # API event 39 -> canonical GW 30
    assert _one(s2019, fpl_fixture_id=200)["gw"] == 30
    assert not s2019["void"].any()


def test_fixtures_void_gw7_kept_and_flagged(fixtures_df: pd.DataFrame) -> None:
    void_row = _one(fixtures_df, season=2022, fpl_fixture_id=61)
    assert void_row["void"]
    assert not void_row["finished"]
    assert pd.isna(void_row["home_goals"])
    assert not _one(fixtures_df, season=2022, fpl_fixture_id=60)["void"]


def test_fixtures_reconstructed_without_fixtures_csv(fixtures_df: pd.DataFrame) -> None:
    # 2016-17 ships no fixtures.csv: home/away inferred from merged_gw opponent_team.
    row = _one(fixtures_df, season=2016, fpl_fixture_id=400)
    assert row["gw"] == 1
    assert row["home_team_code"] == 3
    assert row["away_team_code"] == 91
    assert row["home_goals"] == 2
    assert row["away_goals"] == 0
    assert row["finished"]


# --- player_match -----------------------------------------------------------------


def test_player_match_schema_and_dtypes(pm: pd.DataFrame, tmp_path: Path) -> None:
    out = tmp_path / "player_match.parquet"
    build.build_player_match(raw_root=RAW_ROOT, out_path=out)
    on_disk = pd.read_parquet(out)
    assert list(on_disk.columns) == list(build.PLAYER_MATCH_COLS)
    for col in ("_kickoff_utc", "_transfers_in", "_transfers_out"):
        assert col not in on_disk.columns  # private pass-through columns not persisted
    assert pm["gw"].dtype == "int64"
    assert pm["was_home"].dtype == bool
    assert str(pm["position"].dtype) == "string"
    assert str(pm["starts"].dtype) == "Int64"
    assert str(pm["xg"].dtype) == "Float64"
    assert str(pm["us_xg"].dtype) == "Float64"
    assert pm["us_xg"].isna().all()  # populated later by the Understat join
    # key uniqueness at the player-fixture grain
    assert not pm.duplicated(["season", "gw", "fpl_fixture_id", "player_code"]).any()


def test_dgw_keeps_one_row_per_fixture(pm: pd.DataFrame) -> None:
    dgw = pm[(pm["season"] == 2022) & (pm["player_code"] == 600) & (pm["gw"] == 2)]
    assert len(dgw) == 2
    assert sorted(dgw["fpl_fixture_id"]) == [21, 29]
    assert sorted(dgw["minutes"]) == [45, 90]


def test_2019_stub_drop_and_remap(pm: pd.DataFrame) -> None:
    luiz = pm[(pm["season"] == 2019) & (pm["player_code"] == 800)]
    assert sorted(luiz["gw"]) == [29, 30]  # stub GW33 row dropped; event 39 -> 30
    assert not ((pm["season"] == 2019) & (pm["gw"] == 33)).any()


def test_empty_stadium_flags(pm: pd.DataFrame) -> None:
    assert not _one(pm, season=2019, gw=29)["empty_stadium"]
    assert _one(pm, season=2019, gw=30)["empty_stadium"]  # COVID restart
    assert pm.loc[pm["season"] == 2020, "empty_stadium"].all()  # whole season
    assert not pm.loc[pm["season"] == 2016, "empty_stadium"].any()
    assert not pm.loc[pm["season"] == 2022, "empty_stadium"].any()


def test_void_gw7_flagged_on_player_rows(pm: pd.DataFrame) -> None:
    void_row = _one(pm, season=2022, player_code=700, gw=7)
    assert void_row["void_gw"]
    assert void_row["minutes"] == 0
    assert not _one(pm, season=2022, player_code=700, gw=20)["void_gw"]
    assert not pm.loc[pm["season"] != 2022, "void_gw"].any()


def test_stint_increments_on_mid_season_club_switch(pm: pd.DataFrame) -> None:
    # Semenyo-style switch: BOU (team_code 91) -> MCI (43) within 2022-23.
    switcher = pm[(pm["season"] == 2022) & (pm["player_code"] == 500)].sort_values("gw")
    assert list(switcher["team_code"]) == [91, 43]
    assert list(switcher["stint_id"]) == [0, 1]
    # non-movers stay at stint 0 (including across a DGW)
    assert (pm.loc[pm["player_code"] == 600, "stint_id"] == 0).all()
    assert (pm.loc[pm["player_code"] == 700, "stint_id"] == 0).all()


def test_position_normalization_and_manager_filter(pm: pd.DataFrame) -> None:
    assert _one(pm, player_code=850)["position"] == "GKP"  # "GK" in 2020-21 file
    assert _one(pm, season=2022, player_code=700, gw=20)["position"] == "GKP"
    assert _one(pm, player_code=950)["position"] == "MID"  # element_type map (2016-17)
    assert _one(pm, player_code=960)["position"] == "DEF"
    assert 900 not in set(pm["player_code"])  # element_type 5 / "AM" dropped
    assert set(pm["position"]) <= {"GKP", "DEF", "MID", "FWD"}


def test_subs_regime(pm: pd.DataFrame) -> None:
    regimes = pm.groupby("season")["subs_regime"].unique().to_dict()
    assert {s: list(v) for s, v in regimes.items()} == {
        2016: [3],
        2019: [3],
        2020: [3],
        2022: [5],
    }


def test_rich_stats_and_era_nullables(pm: pd.DataFrame) -> None:
    # 2016-18 rich per-match stats are kept where available
    francis = _one(pm, player_code=960)
    assert francis["tackles"] == 4
    assert francis["clearances_blocks_interceptions"] == 7
    assert francis["recoveries"] == 8
    assert pd.isna(francis["starts"]) and pd.isna(francis["xg"])
    # 2019-2024 have neither rich stats nor (pre-2022) starts/xg
    leno = _one(pm, player_code=850)
    assert pd.isna(leno["tackles"]) and pd.isna(leno["starts"]) and pd.isna(leno["xg"])
    # 2022+ has starts and the xG family
    odegaard = _one(pm, season=2022, player_code=600, fpl_fixture_id=21)
    assert odegaard["starts"] == 1
    assert odegaard["xg"] == pytest.approx(0.5)
    assert odegaard["xgc"] == pytest.approx(0.4)


def test_price_and_home_away(pm: pd.DataFrame) -> None:
    ozil = _one(pm, player_code=950)
    assert ozil["price"] == 95  # observed `value` column, 0.1m units
    assert ozil["was_home"]
    assert ozil["team_code"] == 3
    assert ozil["opponent_code"] == 91
    francis = _one(pm, player_code=960)
    assert not francis["was_home"]
    assert francis["team_code"] == 91
    assert francis["opponent_code"] == 3


# --- player_gw --------------------------------------------------------------------


def test_player_gw_schema(pgw: pd.DataFrame) -> None:
    expected = [
        "season",
        "gw",
        "player_code",
        "fpl_element_id",
        "team_code",
        "position",
        "n_fixtures",
        *build.CORE_OUTCOME_COLS,
        *build.NULLABLE_INT_OUTCOME_COLS,
        *build.XG_COLS.values(),
        "value",
        "selected_by_percent",
        "transfers_in_event",
        "transfers_out_event",
    ]
    assert list(pgw.columns) == expected
    assert not pgw.duplicated(["season", "gw", "player_code"]).any()
    assert str(pgw["selected_by_percent"].dtype) == "Float64"
    assert pgw["selected_by_percent"].isna().all()  # vaastav ships counts, not percent


def test_player_gw_sums_dgw_fixtures(pgw: pd.DataFrame) -> None:
    row = _one(pgw, season=2022, player_code=600, gw=2)
    assert row["n_fixtures"] == 2
    assert row["minutes"] == 135
    assert row["total_points"] == 9
    assert row["goals_scored"] == 1
    assert row["starts"] == 1
    assert row["xg"] == pytest.approx(0.7)
    assert row["value"] == 84
    assert row["transfers_in_event"] == 1000
    assert row["transfers_out_event"] == 50


def test_player_gw_single_fixture_and_switch(pgw: pd.DataFrame) -> None:
    assert _one(pgw, season=2022, player_code=700, gw=7)["n_fixtures"] == 1
    assert _one(pgw, season=2022, player_code=500, gw=20)["team_code"] == 43
    leno = _one(pgw, player_code=850)
    assert leno["saves"] == 4
    assert pd.isna(leno["starts"])  # all-NA sums stay NA (min_count=1)
    assert pd.isna(leno["xg"])


# --- build_all --------------------------------------------------------------------


def test_build_all_writes_everything_deterministically(tmp_path: Path) -> None:
    out1 = tmp_path / "run1"
    out2 = tmp_path / "run2"
    paths1 = build.build_all(raw_root=RAW_ROOT, processed_dir=out1)
    paths2 = build.build_all(raw_root=RAW_ROOT, processed_dir=out2)
    assert set(paths1) == {"teams", "players", "fixtures", "player_match", "player_gw"}
    for name, p1 in paths1.items():
        p2 = paths2[name]
        assert p1.exists() and p2.exists()
        pd.testing.assert_frame_equal(pd.read_parquet(p1), pd.read_parquet(p2))
        assert p1.read_bytes() == p2.read_bytes(), f"{name} parquet not byte-identical"
    # idempotent: re-running into the same directory is safe
    build.build_all(raw_root=RAW_ROOT, processed_dir=out1)
    pm = pd.read_parquet(paths1["player_match"])
    assert list(pm.columns) == list(build.PLAYER_MATCH_COLS)
