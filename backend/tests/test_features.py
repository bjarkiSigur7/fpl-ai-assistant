"""Tests for fplai.features.windows on a synthetic mini-league.

Synthetic setup: 3 players, seasons 2024 (5 GWs, DGW in GW3 for team 1) and 2025
(3 GWs, team 4 newly promoted), plus a 2023 teams row so 2024 promoted flags are
knowable. Hand-computed window values cover: strictly-prior-GW windows (DGW legs
share one information set), season-boundary carry-over with f_new_season, per-season
team-window resets, nullable stats staying NaN (starts/xg pre-availability), defcon
count reconstruction, schedule context (days-rest/congestion), promoted flags and
the availability join. A perturbation test asserts no future information reaches
any earlier row's f_* values.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from fplai.features import windows as fw

UTC = "UTC"
PROCESSED = Path(__file__).resolve().parents[2] / "data" / "processed"


def _ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(f"{s} 15:00:00", tz=UTC)


def _fixture_row(
    season: int, gw: int, fid: int, ko: str, home: int, away: int, hg: int, ag: int
) -> dict[str, Any]:
    return {
        "season": season,
        "gw": gw,
        "fpl_fixture_id": fid,
        "kickoff_utc": _ts(ko),
        "home_team_code": home,
        "away_team_code": away,
        "home_goals": hg,
        "away_goals": ag,
        "finished": True,
        "void": False,
    }


def _fixtures() -> pd.DataFrame:
    rows = [
        # season 2024 — team 1 plays every GW; GW3 is a DGW for teams 1 and 3.
        _fixture_row(2024, 1, 1, "2024-08-09", 1, 2, 2, 1),
        _fixture_row(2024, 2, 2, "2024-08-17", 2, 1, 0, 3),
        _fixture_row(2024, 3, 3, "2024-08-24", 1, 3, 1, 1),
        _fixture_row(2024, 3, 4, "2024-08-27", 3, 1, 0, 2),
        _fixture_row(2024, 4, 5, "2024-08-31", 1, 2, 4, 0),
        _fixture_row(2024, 5, 6, "2024-09-07", 2, 1, 2, 2),
        # season 2025 — team 4 promoted.
        _fixture_row(2025, 1, 101, "2025-08-09", 1, 4, 1, 0),
        _fixture_row(2025, 2, 102, "2025-08-16", 4, 1, 2, 2),
        _fixture_row(2025, 3, 103, "2025-08-23", 1, 2, 0, 1),
    ]
    df = pd.DataFrame(rows)
    df["gw"] = df["gw"].astype("Int64")
    df["home_goals"] = df["home_goals"].astype("Int64")
    df["away_goals"] = df["away_goals"].astype("Int64")
    return df


def _teams() -> pd.DataFrame:
    rows = []
    for season, codes in [(2023, (1, 2, 3)), (2024, (1, 2, 3)), (2025, (1, 2, 4))]:
        for i, code in enumerate(codes, start=1):
            rows.append(
                {
                    "season": season,
                    "fpl_team_id": i,
                    "team_code": code,
                    "name": f"Team {code}",
                    "short_name": f"T{code}",
                }
            )
    return pd.DataFrame(rows)


_PM_DEFAULTS: dict[str, Any] = {
    "fpl_element_id": 0,
    "minutes": 0,
    "total_points": 0,
    "goals_scored": 0,
    "assists": 0,
    "clean_sheets": 0,
    "goals_conceded": 0,
    "saves": 0,
    "penalties_saved": 0,
    "penalties_missed": 0,
    "yellow_cards": 0,
    "red_cards": 0,
    "own_goals": 0,
    "bonus": 0,
    "bps": 0,
    "starts": None,
    "defensive_contribution": None,
    "tackles": None,
    "recoveries": None,
    "clearances_blocks_interceptions": None,
    "xg": None,
    "xa": None,
    "xgc": None,
    "us_xg": None,
    "us_xa": None,
    "us_npxg": None,
    "us_shots": None,
    "us_key_passes": None,
    "empty_stadium": False,
    "void_gw": False,
    "price": 50,
    "subs_regime": 5,
    "stint_id": 0,
}

_INT64_COLS = (
    "starts",
    "defensive_contribution",
    "tackles",
    "recoveries",
    "clearances_blocks_interceptions",
)
_FLOAT64_COLS = ("xg", "xa", "xgc", "us_xg", "us_xa", "us_npxg", "us_shots", "us_key_passes")


def _pm_row(
    season: int,
    gw: int,
    fid: int,
    code: int,
    team: int,
    opp: int,
    home: bool,
    pos: str,
    **stats: Any,
) -> dict[str, Any]:
    row = dict(_PM_DEFAULTS)
    row.update(
        season=season,
        gw=gw,
        fpl_fixture_id=fid,
        player_code=code,
        team_code=team,
        opponent_code=opp,
        was_home=home,
        position=pos,
    )
    row.update(stats)
    return row


def _player_match() -> pd.DataFrame:
    r = _pm_row
    rows = [
        # P101 — MID, team 1 both seasons (form carries over the boundary).
        r(2024, 1, 1, 101, 1, 2, True, "MID", total_points=5, minutes=90, goals_scored=1),
        r(2024, 2, 2, 101, 1, 2, False, "MID", total_points=2, minutes=60),
        r(2024, 3, 3, 101, 1, 3, True, "MID", total_points=8, minutes=90, goals_scored=2),
        r(2024, 3, 4, 101, 1, 3, False, "MID", total_points=3, minutes=45),
        r(2024, 4, 5, 101, 1, 2, True, "MID", total_points=10, minutes=90, goals_scored=3),
        r(2024, 5, 6, 101, 1, 2, False, "MID", total_points=2, minutes=90),
        r(2025, 1, 101, 101, 1, 4, True, "MID", total_points=6, minutes=90, goals_scored=1,
          starts=1, xg=0.5),
        r(2025, 2, 102, 101, 1, 4, False, "MID", total_points=4, minutes=90, goals_scored=1,
          starts=1, xg=0.7),
        r(2025, 3, 103, 101, 1, 2, True, "MID", total_points=1, minutes=30, starts=0, xg=0.1),
        # P102 — FWD, team 2 in 2024, moves to team 1 for 2025 (cross-season club change).
        r(2024, 1, 1, 102, 2, 1, False, "FWD", total_points=2, minutes=90),
        r(2024, 2, 2, 102, 2, 1, True, "FWD", total_points=6, minutes=90, goals_scored=1),
        r(2024, 4, 5, 102, 2, 1, False, "FWD", total_points=1, minutes=20),
        r(2024, 5, 6, 102, 2, 1, True, "FWD", total_points=9, minutes=90, goals_scored=2),
        r(2025, 1, 101, 102, 1, 4, True, "FWD", total_points=3, minutes=90, starts=1),
        r(2025, 2, 102, 102, 1, 4, False, "FWD", total_points=7, minutes=90, starts=1),
        r(2025, 3, 103, 102, 1, 2, True, "FWD", total_points=2, minutes=60, starts=1),
        # P103 — DEF, team 1 GW1-2 then mid-season move to team 3 (stint 0 -> 1);
        # rich defensive stats present for the defcon reconstruction (DEF: CBI + tackles).
        r(2024, 1, 1, 103, 1, 2, True, "DEF", total_points=6, minutes=90,
          clearances_blocks_interceptions=5, tackles=3, recoveries=4),
        r(2024, 2, 2, 103, 1, 2, False, "DEF", total_points=1, minutes=90,
          clearances_blocks_interceptions=2, tackles=1, recoveries=6),
        r(2024, 3, 3, 103, 3, 1, False, "DEF", total_points=2, minutes=90, stint_id=1,
          clearances_blocks_interceptions=4, tackles=2, recoveries=1),
        r(2024, 3, 4, 103, 3, 1, True, "DEF", total_points=0, minutes=45, stint_id=1,
          clearances_blocks_interceptions=1, tackles=0, recoveries=2),
    ]
    df = pd.DataFrame(rows)
    for c in _INT64_COLS:
        df[c] = df[c].astype("Int64")
    for c in _FLOAT64_COLS:
        df[c] = df[c].astype("Float64")
    df["position"] = df["position"].astype("string")
    return df


def _availability() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"season": 2025, "gw": 1, "player_code": 101, "status": "d",
             "chance_of_playing": 75},
            {"season": 2025, "gw": 2, "player_code": 101, "status": "a",
             "chance_of_playing": 100},
        ]
    )


@pytest.fixture(scope="module")
def tables() -> dict[str, pd.DataFrame]:
    return {
        "player_match": _player_match(),
        "fixtures": _fixtures(),
        "teams": _teams(),
        "availability": _availability(),
    }


@pytest.fixture(scope="module")
def frame(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    return fw.build_feature_frame(tables, target="match")


def _row(df: pd.DataFrame, code: int, season: int, gw: int, fid: int | None = None) -> pd.Series:
    m = (df["player_code"] == code) & (df["season"] == season) & (df["gw"] == gw)
    if fid is not None:
        m &= df["fpl_fixture_id"] == fid
    sub = df.loc[m]
    assert len(sub) == 1, f"expected 1 row, got {len(sub)}"
    return sub.iloc[0]


class TestShapeAndSchema:
    def test_one_row_per_player_fixture(self, frame: pd.DataFrame,
                                        tables: dict[str, pd.DataFrame]) -> None:
        assert len(frame) == len(tables["player_match"])
        keys = ["season", "gw", "fpl_fixture_id", "player_code"]
        assert not frame.duplicated(keys).any()

    def test_feature_columns_are_float(self, frame: pd.DataFrame) -> None:
        fcols = [c for c in frame.columns if c.startswith(fw.FEATURE_PREFIX)]
        assert fcols, "no feature columns built"
        for c in fcols:
            assert frame[c].dtype == np.float64, f"{c} is {frame[c].dtype}"

    def test_labels_present_and_disjoint_from_features(self, frame: pd.DataFrame) -> None:
        labels = fw.get_label_columns()
        assert set(labels) <= set(frame.columns)
        assert not any(c.startswith(fw.FEATURE_PREFIX) for c in labels)

    def test_expected_feature_families(self, frame: pd.DataFrame) -> None:
        for stem in ("points", "minutes", "goals", "xg", "us_xg", "shots", "saves", "defcon"):
            for w in fw.WINDOWS:
                assert f"f_{stem}_mean_{w}" in frame.columns
        for w in fw.STARTS_WINDOWS:
            assert f"f_starts_share_{w}" in frame.columns
        for col in ("f_was_home", "f_days_rest", "f_congestion_21d", "f_season_phase",
                    "f_promoted", "f_opp_promoted", "f_price", "f_empty_stadium",
                    "f_new_season", "f_new_club", "f_n_fixtures", "f_pos_MID",
                    "f_team_gf_mean_5", "f_opp_ga_mean_5", "f_status_a",
                    "f_chance_of_playing", "f_days_since_last_match"):
            assert col in frame.columns, col


class TestPlayerFormWindows:
    def test_first_ever_match_all_nan(self, frame: pd.DataFrame) -> None:
        row = _row(frame, 101, 2024, 1)
        for w in fw.WINDOWS:
            assert np.isnan(row[f"f_points_mean_{w}"])
        assert np.isnan(row["f_new_season"])
        assert np.isnan(row["f_days_since_last_match"])

    def test_gw2_windows(self, frame: pd.DataFrame) -> None:
        row = _row(frame, 101, 2024, 2)
        assert row["f_points_mean_1"] == 5.0
        assert row["f_points_mean_3"] == 5.0
        assert row["f_minutes_mean_1"] == 90.0
        assert row["f_goals_mean_38"] == 1.0
        assert row["f_days_since_last_match"] == 8.0  # Aug 17 - Aug 9

    def test_dgw_legs_share_information_set(self, frame: pd.DataFrame) -> None:
        leg1 = _row(frame, 101, 2024, 3, fid=3)
        leg2 = _row(frame, 101, 2024, 3, fid=4)
        # Prior matches are GW1+GW2 only for BOTH legs — leg 2 must not see leg 1.
        for row in (leg1, leg2):
            assert row["f_points_mean_1"] == 2.0  # GW2 match, not the DGW sibling
            assert row["f_points_mean_3"] == pytest.approx(3.5)  # mean(5, 2)
            assert row["f_n_fixtures"] == 2.0
        form_cols = [f"f_points_mean_{w}" for w in fw.WINDOWS] + ["f_minutes_mean_5"]
        for c in form_cols:
            assert leg1[c] == leg2[c] or (np.isnan(leg1[c]) and np.isnan(leg2[c]))
        # Context still differs per fixture.
        assert leg1["f_was_home"] == 1.0 and leg2["f_was_home"] == 0.0
        # days-since-last is anchored at the prior-GW boundary (GW2 match on Aug 17).
        assert leg1["f_days_since_last_match"] == 7.0
        assert leg2["f_days_since_last_match"] == 10.0

    def test_post_dgw_windows(self, frame: pd.DataFrame) -> None:
        row = _row(frame, 101, 2024, 4)
        assert row["f_points_mean_1"] == 3.0  # last match = second DGW leg
        assert row["f_points_mean_3"] == pytest.approx((2 + 8 + 3) / 3)
        assert row["f_points_mean_5"] == pytest.approx((5 + 2 + 8 + 3) / 4)
        assert row["f_minutes_mean_5"] == pytest.approx((90 + 60 + 90 + 45) / 4)

    def test_season_boundary_carry_over(self, frame: pd.DataFrame) -> None:
        row = _row(frame, 101, 2025, 1)
        assert row["f_points_mean_5"] == pytest.approx((2 + 8 + 3 + 10 + 2) / 5)
        assert row["f_points_mean_38"] == pytest.approx((5 + 2 + 8 + 3 + 10 + 2) / 6)
        assert row["f_new_season"] == 1.0
        assert row["f_new_club"] == 0.0
        assert row["f_days_since_last_match"] == pytest.approx(336.0)  # Sep 7 24 -> Aug 9 25
        mid = _row(frame, 101, 2025, 2)
        assert mid["f_new_season"] == 0.0

    def test_nullable_stats_stay_nan_not_zero(self, frame: pd.DataFrame) -> None:
        row = _row(frame, 101, 2025, 1)
        # 6 prior matches exist but all pre-date starts/xg availability.
        for w in fw.STARTS_WINDOWS:
            assert np.isnan(row[f"f_starts_share_{w}"])
        for w in fw.WINDOWS:
            assert np.isnan(row[f"f_xg_mean_{w}"])
            assert np.isnan(row[f"f_us_xg_mean_{w}"])

    def test_nan_aware_means_over_mixed_windows(self, frame: pd.DataFrame) -> None:
        gw2 = _row(frame, 101, 2025, 2)
        assert gw2["f_xg_mean_3"] == pytest.approx(0.5)  # only 2025 GW1 non-NaN
        assert gw2["f_starts_share_3"] == 1.0
        gw3 = _row(frame, 101, 2025, 3)
        assert gw3["f_xg_mean_3"] == pytest.approx(0.6)  # mean(NaN, 0.5, 0.7)
        assert gw3["f_starts_share_3"] == 1.0
        assert gw3["f_points_mean_1"] == 4.0

    def test_defcon_reconstruction_for_def(self, frame: pd.DataFrame) -> None:
        # DEF counts CBI + tackles: GW1 -> 8, GW2 -> 3.
        gw2 = _row(frame, 103, 2024, 2)
        assert gw2["f_defcon_mean_1"] == 8.0
        gw3 = _row(frame, 103, 2024, 3, fid=3)
        assert gw3["f_defcon_mean_3"] == pytest.approx((8 + 3) / 2)

    def test_mid_season_club_change_flag(self, frame: pd.DataFrame) -> None:
        for fid in (3, 4):
            row = _row(frame, 103, 2024, 3, fid=fid)
            assert row["f_new_club"] == 1.0
            assert row["f_new_season"] == 0.0
        cross = _row(frame, 102, 2025, 1)
        assert cross["f_new_club"] == 1.0 and cross["f_new_season"] == 1.0


class TestTeamAndOpponentForm:
    def test_team_windows(self, frame: pd.DataFrame) -> None:
        row = _row(frame, 101, 2024, 4)
        # Team 1 prior fixtures: gf [2, 3, 1, 2], ga [1, 0, 1, 0].
        assert row["f_team_gf_mean_3"] == pytest.approx(2.0)
        assert row["f_team_gf_mean_5"] == pytest.approx(2.0)
        assert row["f_team_ga_mean_3"] == pytest.approx(1 / 3)
        # Opponent (team 2) prior fixtures: gf [1, 0], ga [2, 3].
        assert row["f_opp_gf_mean_5"] == pytest.approx(0.5)
        assert row["f_opp_ga_mean_5"] == pytest.approx(2.5)

    def test_dgw_team_windows_exclude_same_gw(self, frame: pd.DataFrame) -> None:
        leg2 = _row(frame, 101, 2024, 3, fid=4)
        # Prior team fixtures are GW1-2 only, even for the second DGW leg.
        assert leg2["f_team_gf_mean_5"] == pytest.approx(2.5)  # mean(2, 3)
        assert leg2["f_team_ga_mean_5"] == pytest.approx(0.5)

    def test_team_windows_reset_at_season_boundary(self, frame: pd.DataFrame) -> None:
        gw1 = _row(frame, 101, 2025, 1)
        for w in fw.WINDOWS:
            assert np.isnan(gw1[f"f_team_gf_mean_{w}"])  # no 2024 carry-over
        gw2 = _row(frame, 101, 2025, 2)
        assert gw2["f_team_gf_mean_5"] == 1.0  # only 2025 GW1 (1-0)
        assert gw2["f_team_ga_mean_5"] == 0.0
        assert gw2["f_opp_gf_mean_5"] == 0.0  # team 4 conceded 1, scored 0 in GW1
        assert gw2["f_opp_ga_mean_5"] == 1.0
        gw3 = _row(frame, 101, 2025, 3)
        for w in fw.WINDOWS:
            assert np.isnan(gw3[f"f_opp_gf_mean_{w}"])  # team 2's first 2025 fixture


class TestMatchContext:
    def test_days_rest_and_congestion(self, frame: pd.DataFrame) -> None:
        assert np.isnan(_row(frame, 101, 2024, 1)["f_days_rest"])
        assert _row(frame, 101, 2024, 1)["f_congestion_21d"] == 0.0
        assert _row(frame, 101, 2024, 2)["f_days_rest"] == 8.0
        leg1 = _row(frame, 101, 2024, 3, fid=3)
        leg2 = _row(frame, 101, 2024, 3, fid=4)
        # Schedule context uses the true kickoff sequence: the DGW sibling counts.
        assert leg1["f_days_rest"] == 7.0
        assert leg2["f_days_rest"] == 3.0
        assert leg1["f_congestion_21d"] == 2.0  # Aug 9, Aug 17
        assert leg2["f_congestion_21d"] == 3.0  # Aug 9, 17, 24
        gw4 = _row(frame, 101, 2024, 4)
        assert gw4["f_days_rest"] == 4.0
        assert gw4["f_congestion_21d"] == 3.0  # Aug 17, 24, 27 (Aug 9 outside 21d)
        assert np.isnan(_row(frame, 101, 2025, 1)["f_days_rest"])  # resets per season

    def test_static_context(self, frame: pd.DataFrame) -> None:
        row = _row(frame, 101, 2024, 4)
        assert row["f_was_home"] == 1.0
        assert row["f_season_phase"] == pytest.approx(4 / 38)
        assert row["f_price"] == 50.0
        assert row["f_pos_MID"] == 1.0 and row["f_pos_DEF"] == 0.0
        assert row["f_empty_stadium"] == 0.0
        d = _row(frame, 103, 2024, 2)
        assert d["f_pos_DEF"] == 1.0

    def test_promoted_flags(self, frame: pd.DataFrame) -> None:
        assert _row(frame, 101, 2024, 1)["f_promoted"] == 0.0  # 2023 teams known
        gw1_25 = _row(frame, 101, 2025, 1)
        assert gw1_25["f_promoted"] == 0.0
        assert gw1_25["f_opp_promoted"] == 1.0  # team 4 not in 2024

    def test_promoted_unknowable_is_nan(self, tables: dict[str, pd.DataFrame]) -> None:
        t = {**tables, "teams": tables["teams"].query("season >= 2024")}
        df = fw.build_feature_frame(t, target="match")
        assert np.isnan(_row(df, 101, 2024, 1)["f_promoted"])  # 2023 absent
        assert _row(df, 101, 2025, 1)["f_opp_promoted"] == 1.0


class TestAvailability:
    def test_joined_when_present(self, frame: pd.DataFrame) -> None:
        row = _row(frame, 101, 2025, 1)
        assert row["f_chance_of_playing"] == 75.0
        assert row["f_status_d"] == 1.0 and row["f_status_a"] == 0.0
        gw2 = _row(frame, 101, 2025, 2)
        assert gw2["f_status_a"] == 1.0 and gw2["f_chance_of_playing"] == 100.0

    def test_nan_where_no_snapshot(self, frame: pd.DataFrame) -> None:
        row = _row(frame, 101, 2024, 1)
        assert np.isnan(row["f_chance_of_playing"])
        assert np.isnan(row["f_status_a"])

    def test_all_nan_without_table(self, tables: dict[str, pd.DataFrame]) -> None:
        t = {k: v for k, v in tables.items() if k != "availability"}
        df = fw.build_feature_frame(t, target="match")
        assert df["f_chance_of_playing"].isna().all()
        assert df["f_status_a"].isna().all()


class TestGwTarget:
    def test_aggregation(self, tables: dict[str, pd.DataFrame]) -> None:
        gw = fw.build_feature_frame(tables, target="gw")
        assert not gw.duplicated(["season", "gw", "player_code"]).any()
        dgw = gw.loc[
            (gw["player_code"] == 101) & (gw["season"] == 2024) & (gw["gw"] == 3)
        ].iloc[0]
        assert dgw["n_fixtures"] == 2
        assert dgw["total_points"] == 11  # 8 + 3, labels summed
        assert dgw["minutes"] == 135
        assert dgw["goals_scored"] == 2
        assert dgw["f_points_mean_3"] == pytest.approx(3.5)  # form identical across legs
        assert dgw["f_was_home"] == 0.5  # one home + one away leg
        assert dgw["f_n_fixtures"] == 2.0
        single = gw.loc[
            (gw["player_code"] == 101) & (gw["season"] == 2024) & (gw["gw"] == 4)
        ].iloc[0]
        assert single["n_fixtures"] == 1
        assert single["total_points"] == 10


class TestLeakage:
    def test_future_perturbation_changes_nothing_earlier(
        self, tables: dict[str, pd.DataFrame], frame: pd.DataFrame
    ) -> None:
        t2 = {k: v.copy() for k, v in tables.items()}
        pm = t2["player_match"]
        m = (pm["season"] == 2025) & (pm["gw"] == 3) & (pm["player_code"] == 101)
        pm.loc[m, "total_points"] += 100
        pm.loc[m, "goals_scored"] += 5
        pm.loc[m, "minutes"] = 90
        fx = t2["fixtures"]
        fx.loc[(fx["season"] == 2025) & (fx["fpl_fixture_id"] == 103), "home_goals"] = 9
        rebuilt = fw.build_feature_frame(t2, target="match")
        fcols = sorted(c for c in frame.columns if c.startswith(fw.FEATURE_PREFIX))
        # (2025, GW3) is the final GW: no row anywhere may see the perturbation —
        # including the GW3 rows themselves (own-GW outcomes are post-deadline).
        pd.testing.assert_frame_equal(frame[fcols], rebuilt[fcols])

    def test_mid_history_perturbation_respects_gw_boundary(
        self, tables: dict[str, pd.DataFrame], frame: pd.DataFrame
    ) -> None:
        t2 = {k: v.copy() for k, v in tables.items()}
        pm = t2["player_match"]
        m = (pm["season"] == 2024) & (pm["fpl_fixture_id"] == 4) & (pm["player_code"] == 101)
        pm.loc[m, "total_points"] = 50
        rebuilt = fw.build_feature_frame(t2, target="match")
        fcols = sorted(c for c in frame.columns if c.startswith(fw.FEATURE_PREFIX))
        early = (frame["season"] < 2024) | ((frame["season"] == 2024) & (frame["gw"] <= 3))
        pd.testing.assert_frame_equal(
            frame.loc[early, fcols].reset_index(drop=True),
            rebuilt.loc[early, fcols].reset_index(drop=True),
        )
        # ... and the perturbation DOES propagate forward (sanity check).
        after = _row(rebuilt, 101, 2024, 4)
        assert after["f_points_mean_1"] == 50.0

    def test_assert_no_leakage_passes(
        self, frame: pd.DataFrame, tables: dict[str, pd.DataFrame]
    ) -> None:
        fw.assert_no_leakage(frame, tables, target="match", n_samples=len(frame), seed=1)

    def test_assert_no_leakage_catches_corruption(
        self, frame: pd.DataFrame, tables: dict[str, pd.DataFrame]
    ) -> None:
        bad = frame.copy()
        row = bad.loc[
            (bad["player_code"] == 101) & (bad["season"] == 2024) & (bad["gw"] == 4)
        ].index[0]
        bad.loc[row, "f_points_mean_1"] = 99.0
        with pytest.raises(AssertionError, match="f_points_mean_1"):
            fw.assert_no_leakage(bad, tables, target="match", n_samples=len(bad), seed=1)


class TestErrors:
    def test_bad_target(self, tables: dict[str, pd.DataFrame]) -> None:
        with pytest.raises(ValueError, match="target"):
            fw.build_feature_frame(tables, target="week")  # type: ignore[arg-type]

    def test_missing_table(self, tables: dict[str, pd.DataFrame]) -> None:
        with pytest.raises(ValueError, match="fixtures"):
            fw.build_feature_frame({"player_match": tables["player_match"]}, target="match")


@pytest.mark.skipif(
    not (PROCESSED / "player_match.parquet").exists(),
    reason="processed parquet tables not on disk",
)
class TestRealData:
    """Full build over whatever seasons exist on disk: timing, shape, leakage spot-check."""

    def test_full_build(self) -> None:
        tables = {
            name: pd.read_parquet(PROCESSED / f"{name}.parquet")
            for name in ("player_match", "fixtures", "teams")
        }
        t0 = time.perf_counter()
        df = fw.build_feature_frame(tables, target="match")
        elapsed = time.perf_counter() - t0
        n_expected = int((~tables["player_match"]["void_gw"]).sum())
        assert len(df) == n_expected
        fcols = [c for c in df.columns if c.startswith(fw.FEATURE_PREFIX)]
        assert len(fcols) > 100
        assert all(df[c].dtype == np.float64 for c in fcols)
        # Form features should be broadly populated once history exists.
        late = df[(df["season"] > int(df["season"].min())) | (df["gw"] > 10)]
        assert late["f_points_mean_5"].notna().mean() > 0.95
        assert elapsed < 120, f"full build took {elapsed:.1f}s (budget 120s)"
        fw.assert_no_leakage(df, tables, target="match", n_samples=15, seed=7)
        print(
            f"\nreal build: {len(df)} rows x {len(fcols)} features in {elapsed:.1f}s "
            f"(seasons {sorted(df['season'].unique())})"
        )
