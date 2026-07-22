"""Offline tests for fplai.models.bonus — BPS reconstruction and the bonus rank sim."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fplai import rules
from fplai.models import bonus

PLAYER_MATCH_PARQUET = (
    Path(__file__).resolve().parents[2] / "data" / "processed" / "player_match.parquet"
)

#: Deterministic draws: zero noise makes the Monte Carlo an exact rank calculation.
ZERO_NOISE = bonus.BonusCalibration(bias={}, sigma_intercept=0.0, sigma_slope=0.0, sigma_floor=0.0)


def make_profile(rows: list[dict], fixture: int = 100) -> pd.DataFrame:
    """Build an event profile from partial row dicts (fills keys and zeros)."""
    frame = pd.DataFrame(rows)
    if "position" not in frame.columns:
        frame["position"] = "MID"
    if "fpl_fixture_id" not in frame.columns:
        frame["fpl_fixture_id"] = fixture
    frame["season"] = 2026
    frame["gw"] = 1
    frame["player_code"] = np.arange(1, len(frame) + 1)
    return frame


class TestBpsMatrix:
    def test_v4_is_rules_bps_2026(self) -> None:
        assert bonus.bps_matrix(2026) == {k: float(v) for k, v in rules.BPS_2026.items()}

    def test_gk_penalty_save_chain(self) -> None:
        # FPL_KNOWLEDGE §2.1: 15 (v1) -> 9 (v2) -> 8 (v3) -> 7 (v4).
        assert bonus.bps_matrix(2017)["gk_penalty_save"] == 15.0
        assert bonus.bps_matrix(2023)["gk_penalty_save"] == 15.0
        assert bonus.bps_matrix(2024)["gk_penalty_save"] == 9.0
        assert bonus.bps_matrix(2025)["gk_penalty_save"] == 8.0
        assert bonus.bps_matrix(2026)["gk_penalty_save"] == 7.0

    def test_cbi_divisor_change(self) -> None:
        for season in (2016, 2024, 2025):
            m = bonus.bps_matrix(season)
            assert m["cbi_per_2"] == 1.0 and "cbi_per_3" not in m
        m26 = bonus.bps_matrix(2026)
        assert m26["cbi_per_3"] == 1.0 and "cbi_per_2" not in m26

    def test_tackled_against_removed_in_v4(self) -> None:
        assert bonus.bps_matrix(2025)["tackled_against"] == -1.0
        assert "tackled_against" not in bonus.bps_matrix(2026)

    def test_gk_save_categories_per_version(self) -> None:
        v2 = bonus.bps_matrix(2024)
        assert v2["gk_save"] == 2.0 and "gk_save_inside_box" not in v2
        v3 = bonus.bps_matrix(2025)
        assert v3["gk_save_inside_box"] == 3.0 and v3["gk_save_outside_box"] == 2.0
        assert "gk_save" not in v3
        v4 = bonus.bps_matrix(2026)
        assert v4["gk_save"] == 2.0
        assert v4["gk_save_inside_box_extra"] == 1.0 and v4["gk_big_chance_save_extra"] == 1.0

    def test_v2_additions_absent_in_v1(self) -> None:
        v1 = bonus.bps_matrix(2023)
        for key in ("goal_conceded_gkp_def", "goal_line_clearance", "foul_won", "shot_on_target"):
            assert key not in v1, key
        v2 = bonus.bps_matrix(2024)
        assert v2["goal_conceded_gkp_def"] == -4.0
        assert v2["goal_line_clearance"] == 3.0  # 9.0 only from v3

    def test_penalty_goal_category_from_v3(self) -> None:
        assert "penalty_goal" not in bonus.bps_matrix(2024)
        assert bonus.bps_matrix(2025)["penalty_goal"] == 12.0

    def test_matches_rules_deltas(self) -> None:
        # Where BPS_DELTAS pins an explicit value, the reconstruction must agree.
        assert (
            bonus.bps_matrix(2025)["gk_penalty_save"] == rules.BPS_DELTAS["v3"]["gk_penalty_save"]
        )
        assert (
            bonus.bps_matrix(2024)["gk_penalty_save"] == rules.BPS_DELTAS["v2"]["gk_penalty_save"]
        )
        assert (
            bonus.bps_matrix(2024)["goal_line_clearance"]
            == rules.BPS_DELTAS["v2"]["goal_line_clearance"]
        )

    def test_bad_season_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown season"):
            bonus.bps_matrix(1999)


class TestExpectedBps:
    def test_goal_bps_by_position_2026(self) -> None:
        profile = make_profile(
            [
                {"position": "GKP", "q2": 1.0, "e_goals": 1.0},
                {"position": "DEF", "q2": 1.0, "e_goals": 1.0},
                {"position": "MID", "q2": 1.0, "e_goals": 1.0},
                {"position": "FWD", "q2": 1.0, "e_goals": 1.0},
            ]
        )
        ebps = bonus.expected_bps(profile, 2026)
        # 6 (60+ minutes) + positional goal BPS (12/12/18/24).
        assert ebps.tolist() == [18.0, 18.0, 24.0, 30.0]

    def test_gk_save_value_by_version(self) -> None:
        profile = make_profile([{"position": "GKP", "q2": 1.0, "e_saves": 3.0}])
        # v2 flat: 6 + 3*2 = 12.
        assert bonus.expected_bps(profile, 2024).iloc[0] == pytest.approx(12.0)
        # v3 in/out box: 6 + 3*(0.7*3 + 0.3*2) = 14.1.
        assert bonus.expected_bps(profile, 2025).iloc[0] == pytest.approx(14.1)
        # v4: 6 + 3*(2 + 0.7*1 + 0.15*1) = 14.55.
        assert bonus.expected_bps(profile, 2026).iloc[0] == pytest.approx(14.55)

    def test_cbi_divisor_flows_through(self) -> None:
        profile = make_profile([{"position": "DEF", "q2": 1.0, "e_cbi": 6.0}])
        assert bonus.expected_bps(profile, 2025).iloc[0] == pytest.approx(6.0 + 6.0 / 2)
        assert bonus.expected_bps(profile, 2026).iloc[0] == pytest.approx(6.0 + 6.0 / 3)

    def test_cs_and_concede_only_gkp_def(self) -> None:
        profile = make_profile(
            [
                {"position": "DEF", "q2": 1.0, "e_cs": 1.0, "e_goals_conceded": 1.0},
                {"position": "MID", "q2": 1.0, "e_cs": 1.0, "e_goals_conceded": 1.0},
            ]
        )
        ebps = bonus.expected_bps(profile, 2026)
        assert ebps.iloc[0] == pytest.approx(6.0 + 12.0 - 4.0)
        assert ebps.iloc[1] == pytest.approx(6.0)  # neither category applies to MID

    def test_saves_only_gkp(self) -> None:
        profile = make_profile([{"position": "DEF", "q2": 1.0, "e_saves": 3.0}])
        assert bonus.expected_bps(profile, 2026).iloc[0] == pytest.approx(6.0)

    def test_hand_computed_full_row_2025(self) -> None:
        # DEF, 60+: 6; goal 12; assist 9; CS 12; conceded -4; cbi 4/2=2;
        # recoveries 6/3=2; tackles 2*2=4; yellow -3*0.1.
        profile = make_profile(
            [
                {
                    "position": "DEF",
                    "q2": 1.0,
                    "e_goals": 1.0,
                    "e_assists": 1.0,
                    "e_cs": 1.0,
                    "e_goals_conceded": 1.0,
                    "e_cbi": 4.0,
                    "e_recoveries": 6.0,
                    "e_tackles": 2.0,
                    "e_yellow": 0.1,
                }
            ]
        )
        expected = 6 + 12 + 9 + 12 - 4 + 2 + 2 + 4 - 0.3
        assert bonus.expected_bps(profile, 2025).iloc[0] == pytest.approx(expected)

    def test_missing_and_nan_columns_are_zero(self) -> None:
        profile = make_profile([{"position": "MID", "q2": 1.0, "e_goals": np.nan}])
        assert bonus.expected_bps(profile, 2026).iloc[0] == pytest.approx(6.0)

    def test_unknown_position_raises(self) -> None:
        profile = make_profile([{"position": "XXX", "q2": 1.0}])
        with pytest.raises(ValueError, match="unknown positions"):
            bonus.expected_bps(profile, 2026)

    def test_missing_position_raises(self) -> None:
        frame = pd.DataFrame({"fpl_fixture_id": [1], "q2": [1.0]})
        with pytest.raises(ValueError, match="position"):
            bonus.expected_bps(frame, 2026)


def _mid(ebps_extra: dict | None = None) -> dict:
    """A 60+-minute MID row (EBPS 6) plus optional extra expected counts."""
    row = {"position": "MID", "q2": 1.0}
    row.update(ebps_extra or {})
    return row


class TestTieRules:
    """Official tie rules on deterministic (zero-noise) draws.

    EBPS building blocks (2026, MID): 60+ minutes = 6; +goal = 24; +assist = 15.
    """

    def run(self, rows: list[dict]) -> list[float]:
        profile = make_profile(rows)
        out = bonus.expected_bonus(profile, 2026, calibration=ZERO_NOISE, n_draws=8, seed=1)
        return out.tolist()

    def test_distinct_3_2_1(self) -> None:
        assert self.run([_mid({"e_goals": 1.0}), _mid({"e_assists": 1.0}), _mid()]) == [3, 2, 1]

    def test_tie_for_first_3_3_1(self) -> None:
        assert self.run([_mid({"e_goals": 1.0}), _mid({"e_goals": 1.0}), _mid()]) == [3, 3, 1]

    def test_tie_for_second_3_2_2(self) -> None:
        rows = [_mid({"e_goals": 1.0}), _mid({"e_assists": 1.0}), _mid({"e_assists": 1.0})]
        assert self.run(rows) == [3, 2, 2]

    def test_tie_for_third_3_2_1_1(self) -> None:
        rows = [_mid({"e_goals": 1.0}), _mid({"e_assists": 1.0}), _mid(), _mid()]
        assert self.run(rows) == [3, 2, 1, 1]

    def test_three_way_tie_for_first(self) -> None:
        rows = [_mid({"e_goals": 1.0})] * 3 + [_mid()]
        assert self.run(rows) == [3, 3, 3, 0]


class TestExpectedBonus:
    def test_grouping_is_per_fixture(self) -> None:
        rows_a = [_mid({"e_goals": 1.0}), _mid()]
        rows_b = [_mid({"e_assists": 1.0}), _mid()]
        profile = pd.concat(
            [make_profile(rows_a, fixture=1), make_profile(rows_b, fixture=2)],
            ignore_index=True,
        )
        out = bonus.expected_bonus(profile, 2026, calibration=ZERO_NOISE, n_draws=8, seed=1)
        assert out.tolist() == [3, 2, 3, 2]  # ranked within each fixture separately

    def test_deterministic_given_seed(self) -> None:
        rows = [_mid({"e_goals": 0.6}), _mid({"e_assists": 0.4}), _mid(), _mid({"q2": 0.5})]
        profile = make_profile(rows)
        a = bonus.expected_bonus(profile, 2026, seed=7)
        b = bonus.expected_bonus(profile, 2026, seed=7)
        pd.testing.assert_series_equal(a, b)
        c = bonus.expected_bonus(profile, 2026, seed=8)
        assert not np.allclose(a.to_numpy(), c.to_numpy())

    def test_total_bonus_per_fixture_is_about_six(self) -> None:
        # With noise, each draw awards >= 6 points (ties only add); with 8
        # players total E[bonus] stays close to 6.
        rows = [_mid({"e_goals": g / 10}) for g in range(8)]
        profile = make_profile(rows)
        total = bonus.expected_bonus(profile, 2026, n_draws=400, seed=3).sum()
        assert 5.99 <= total <= 7.0

    def test_symmetric_players_split_top_two(self) -> None:
        cal = bonus.BonusCalibration(bias={}, sigma_intercept=5.0, sigma_slope=0.0, sigma_floor=5.0)
        rows = [_mid({"e_goals": 1.0}), _mid({"e_goals": 1.0}), _mid()]
        profile = make_profile(rows)
        out = bonus.expected_bonus(profile, 2026, calibration=cal, n_draws=400, seed=11)
        assert abs(out.iloc[0] - out.iloc[1]) < 0.25  # symmetric by construction
        assert out.iloc[0] > 2.3 and out.iloc[1] > 2.3
        assert 0.8 <= out.iloc[2] <= 1.2  # nearly always third

    def test_non_players_excluded(self) -> None:
        rows = [_mid({"e_goals": 1.0}), _mid(), {"position": "MID", "q1": 0.0, "q2": 0.0}]
        profile = make_profile(rows)
        out = bonus.expected_bonus(profile, 2026, calibration=ZERO_NOISE, n_draws=8, seed=1)
        assert out.tolist() == [3, 2, 0]  # the q=0 row neither scores nor competes

    def test_empty_profile(self) -> None:
        profile = make_profile([_mid()]).iloc[:0]
        out = bonus.expected_bonus(profile, 2026)
        assert out.empty

    def test_missing_fixture_id_raises(self) -> None:
        frame = pd.DataFrame({"position": ["MID"], "q2": [1.0]})
        with pytest.raises(ValueError, match="fpl_fixture_id"):
            bonus.expected_bonus(frame, 2026)


class TestRealizedProfileAndCalibration:
    def make_player_match(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "season": [2025] * 3,
                "gw": [1] * 3,
                "player_code": [1, 2, 3],
                "fpl_fixture_id": [10, 10, 10],
                "position": ["GKP", "DEF", "MID"],
                "minutes": [90, 45, 0],
                "goals_scored": [0, 1, 0],
                "assists": [0, 0, 0],
                "clean_sheets": [1, 0, 0],
                "goals_conceded": [0, 1, 0],
                "saves": [4, 0, 0],
                "penalties_saved": [0, 0, 0],
                "penalties_missed": [0, 0, 0],
                "yellow_cards": [0, 1, 0],
                "red_cards": [0, 0, 0],
                "own_goals": [0, 0, 0],
                "tackles": [0, 2, 0],
                "recoveries": [6, 3, 0],
                "clearances_blocks_interceptions": [1, 8, 0],
                "bps": [28, 15, 0],
                "bonus": [3, 0, 0],
            }
        )

    def test_event_profile_from_realized(self) -> None:
        profile = bonus.event_profile_from_realized(self.make_player_match())
        assert profile["q2"].tolist() == [1.0, 0.0, 0.0]
        assert profile["q1"].tolist() == [0.0, 1.0, 0.0]
        assert profile["e_saves"].tolist() == [4.0, 0.0, 0.0]
        assert profile["e_cbi"].tolist() == [1.0, 8.0, 0.0]
        assert profile["e_goals"].tolist() == [0.0, 1.0, 0.0]

    def test_missing_rich_columns_become_zero(self) -> None:
        pm = self.make_player_match().drop(
            columns=["tackles", "recoveries", "clearances_blocks_interceptions"]
        )
        profile = bonus.event_profile_from_realized(pm)
        assert profile["e_cbi"].tolist() == [0.0, 0.0, 0.0]

    def test_calibrate_recovers_known_bias_and_sigma(self) -> None:
        rng = np.random.default_rng(42)
        n = 1200
        pm = pd.DataFrame(
            {
                "season": 2025,
                "gw": 1,
                "player_code": np.arange(n),
                "fpl_fixture_id": np.arange(n) // 30,
                "position": np.where(np.arange(n) % 2 == 0, "DEF", "FWD"),
                "minutes": 90,
                "goals_scored": rng.binomial(1, 0.1, n),
                "assists": rng.binomial(1, 0.1, n),
                "clean_sheets": rng.binomial(1, 0.3, n),
                "goals_conceded": rng.poisson(1.2, n),
                "saves": 0,
                "penalties_saved": 0,
                "penalties_missed": 0,
                "yellow_cards": 0,
                "red_cards": 0,
                "own_goals": 0,
                "tackles": rng.poisson(1.5, n),
                "recoveries": rng.poisson(4.0, n),
                "clearances_blocks_interceptions": rng.poisson(4.0, n),
            }
        )
        true_bias, true_sigma = 2.0, 4.0
        ebps = bonus.expected_bps(bonus.event_profile_from_realized(pm), 2025)
        pm["bps"] = (ebps + true_bias + rng.normal(0, true_sigma, n)).round()
        cal = bonus.calibrate(pm, 2025)
        assert cal.bias["DEF"] == pytest.approx(true_bias, abs=0.8)
        assert cal.bias["FWD"] == pytest.approx(true_bias, abs=0.8)
        mid_center = float(np.median(ebps + true_bias))
        assert cal.sigma(np.array([mid_center]))[0] == pytest.approx(true_sigma, abs=1.0)

    def test_calibrate_too_few_rows_raises(self) -> None:
        with pytest.raises(ValueError, match="not enough"):
            bonus.calibrate(self.make_player_match(), 2025)


@pytest.fixture(scope="module")
def player_match() -> pd.DataFrame:
    if not PLAYER_MATCH_PARQUET.exists():
        pytest.skip("processed data not built")
    return pd.read_parquet(PLAYER_MATCH_PARQUET)


@pytest.mark.skipif(not PLAYER_MATCH_PARQUET.exists(), reason="processed data not built")
class TestRealData2025:
    """BPS -> bonus mapping validation on realized 2025-26 data (offline: local parquet)."""

    def test_total_bonus_conserved(self, player_match: pd.DataFrame) -> None:
        d = player_match[(player_match["season"] == 2025) & (player_match["minutes"] > 0)]
        profile = bonus.event_profile_from_realized(d)
        pred = bonus.expected_bonus(profile, 2025)
        actual = d["bonus"].sum()
        assert abs(pred.sum() - actual) / actual < 0.05

    def test_decile_table_tracks_actual(self, player_match: pd.DataFrame) -> None:
        table = bonus.bonus_validation_table(player_match, 2025)
        assert len(table) == 10
        top = table.iloc[-1]
        assert top["mean_actual"] > 1.0
        assert top["mean_predicted"] == pytest.approx(top["mean_actual"], rel=0.15)
        # Actual bonus should rise across the last three predicted deciles.
        tail = table["mean_actual"].iloc[-3:].to_numpy()
        assert tail[0] < tail[1] < tail[2]

    def test_calibration_close_to_default(self, player_match: pd.DataFrame) -> None:
        cal = bonus.calibrate(player_match, 2025)
        for pos_name, value in bonus.DEFAULT_CALIBRATION.bias.items():
            assert cal.bias[pos_name] == pytest.approx(value, abs=0.5), pos_name


def test_independent_floor_identity() -> None:
    """E[floor] approximations used for CBI/recoveries are within the doc'd bias."""
    # E[x]/3 vs E[floor(x/3)] for a Poisson(4) count: bias < 0.5 (documented approx).
    lam = 4.0
    exact = sum(math.exp(-lam) * lam**k / math.factorial(k) * (k // 3) for k in range(60))
    assert abs(lam / 3 - exact) < 0.5
