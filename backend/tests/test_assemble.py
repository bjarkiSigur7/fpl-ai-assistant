"""Offline tests for fplai.models.assemble — the xP decomposition (FPL_KNOWLEDGE §1.1)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from fplai.models import assemble, bonus


def poisson_floor_ev(lam: float, divisor: int, k_max: int = 60) -> float:
    """Independent reference: E[floor(X / divisor)], X ~ Poisson(lam)."""
    return sum(
        math.exp(-lam) * lam**k / math.factorial(k) * (k // divisor) for k in range(k_max + 1)
    )


def nb_tail(mean: float, r: int, threshold: int) -> float:
    """Independent reference: P(X >= threshold), X ~ NB(size=r, mean=mean), integer r."""
    p = r / (r + mean)
    cdf = sum(math.comb(k + r - 1, k) * p**r * (1 - p) ** k for k in range(threshold))
    return 1.0 - cdf


# --------------------------------------------------------------------------------------
# Synthetic DGW scenario: star striker (team 100, two fixtures) vs bench defender
# --------------------------------------------------------------------------------------

SEASON = 2025


def dgw_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    features = pd.DataFrame(
        {
            "season": [SEASON] * 3,
            "gw": [30] * 3,
            "player_code": [1, 1, 2],
            "fpl_fixture_id": [900, 901, 900],
            "team_code": [100, 100, 200],
            "was_home": [True, False, False],
            "position": ["FWD", "FWD", "DEF"],
        }
    )
    minutes = pd.DataFrame(
        {
            "season": [SEASON] * 2,
            "gw": [30] * 2,
            "player_code": [1, 2],
            "q0": [0.05, 0.7],
            "q1": [0.10, 0.2],
            "q2": [0.85, 0.1],
            "mu1": [30.0, 20.0],
            "mu2": [90.0, 90.0],
        }
    )
    rates = pd.DataFrame(
        {
            "season": [SEASON] * 2,
            "gw": [30] * 2,
            "player_code": [1, 2],
            "lam_goal": [0.6, 0.02],
            "lam_assist": [0.2, 0.03],
            "lam_saves": [0.0, 0.0],
            "lam_defcon": [2.0, 9.0],
            "defcon_disp": [8.0, 6.0],
            "p_yellow": [0.1, 0.05],
            "p_red": [0.01, 0.005],
            "lam_og": [0.005, 0.01],
        }
    )
    team = pd.DataFrame(
        {
            "season": [SEASON] * 2,
            "fpl_fixture_id": [900, 901],
            "home_lambda": [2.0, 1.5],
            "away_lambda": [1.0, 1.2],
            "p_cs_home": [0.35, 0.30],
            "p_cs_away": [0.15, 0.25],
        }
    )
    return minutes, team, rates, features


@pytest.fixture(scope="module")
def dgw_xp() -> pd.DataFrame:
    minutes, team, rates, features = dgw_inputs()
    return assemble.assemble_xp(minutes, team, rates, features, SEASON)


class TestDgwScenario:
    """Hand-computed xP for a star striker across a DGW vs a bench defender."""

    @pytest.fixture()
    def xp(self, dgw_xp: pd.DataFrame) -> pd.DataFrame:
        return dgw_xp

    def test_output_schema(self, xp: pd.DataFrame) -> None:
        assert list(xp.columns) == list(assemble.XP_COLUMNS)
        assert len(xp) == 3

    def test_striker_appearance(self, xp: pd.DataFrame) -> None:
        # E = 1*q1 + 2*q2 = 0.10 + 2*0.85, identical in both fixtures.
        striker = xp[xp["player_code"] == 1]
        assert striker["xp_appearance"].tolist() == pytest.approx([1.8, 1.8])

    def test_striker_goals_with_fixture_multiplier(self, xp: pd.DataFrame) -> None:
        # E[min] = 0.1*30 + 0.85*90 = 79.5; share = 79.5/90.
        # Team 100 lambdas: 2.0 (home, f900) and 1.2 (away, f901); avg 1.6.
        # F_att = 1.25 / 0.75. FWD goal = 4 pts.
        share = 79.5 / 90
        f900 = xp[(xp["player_code"] == 1) & (xp["fpl_fixture_id"] == 900)].iloc[0]
        f901 = xp[(xp["player_code"] == 1) & (xp["fpl_fixture_id"] == 901)].iloc[0]
        assert f900["xp_goals"] == pytest.approx(4 * 0.6 * share * (2.0 / 1.6))
        assert f901["xp_goals"] == pytest.approx(4 * 0.6 * share * (1.2 / 1.6))
        assert f900["xp_assists"] == pytest.approx(3 * 0.2 * share * (2.0 / 1.6))

    def test_striker_no_cs_concede_saves(self, xp: pd.DataFrame) -> None:
        striker = xp[xp["player_code"] == 1]
        assert striker["xp_cs"].tolist() == [0.0, 0.0]  # FWD CS = 0 pts
        assert striker["xp_concede"].tolist() == [0.0, 0.0]  # FWD concede = 0 pts
        assert striker["xp_saves"].tolist() == [0.0, 0.0]

    def test_striker_cards_and_og(self, xp: pd.DataFrame) -> None:
        f900 = xp[(xp["player_code"] == 1) & (xp["fpl_fixture_id"] == 900)].iloc[0]
        p_play = 0.95
        assert f900["xp_cards"] == pytest.approx(-1 * 0.1 * p_play + -3 * 0.01 * p_play)
        assert f900["xp_other"] == pytest.approx(-2 * 0.005 * (79.5 / 90))

    def test_defender_cs_and_concede(self, xp: pd.DataFrame) -> None:
        row = xp[xp["player_code"] == 2].iloc[0]
        assert row["xp_cs"] == pytest.approx(4 * 0.1 * 0.15)  # pts_CS * q2 * P(CS)
        # Conceded: opponent (home) lambda 2.0 scaled by E[min]/90 = 13/90.
        lam_on = 2.0 * 13.0 / 90.0
        assert row["xp_concede"] == pytest.approx(-1 * poisson_floor_ev(lam_on, 2), abs=1e-9)

    def test_defender_defcon_reference(self, xp: pd.DataFrame) -> None:
        row = xp[xp["player_code"] == 2].iloc[0]
        mean = 9.0 * 13.0 / 90.0
        assert row["xp_defcon"] == pytest.approx(2 * nb_tail(mean, 6, 10), abs=1e-9)

    def test_components_sum_to_xp(self, xp: pd.DataFrame) -> None:
        np.testing.assert_allclose(
            xp[list(assemble.COMPONENT_COLUMNS)].sum(axis=1), xp["xp"], rtol=1e-12
        )

    def test_aggregate_gw_sums_dgw(self, xp: pd.DataFrame) -> None:
        agg = assemble.aggregate_gw(xp)
        assert len(agg) == 2
        striker = agg[agg["player_code"] == 1].iloc[0]
        defender = agg[agg["player_code"] == 2].iloc[0]
        assert striker["n_fixtures"] == 2 and defender["n_fixtures"] == 1
        per_fixture = xp[xp["player_code"] == 1]["xp"].sum()
        assert striker["xp"] == pytest.approx(per_fixture)
        # Aggregated components still sum to aggregated xp.
        np.testing.assert_allclose(
            agg[list(assemble.COMPONENT_COLUMNS)].sum(axis=1), agg["xp"], rtol=1e-12
        )

    def test_deterministic(self) -> None:
        minutes, team, rates, features = dgw_inputs()
        a = assemble.assemble_xp(minutes, team, rates, features, SEASON, seed=5)
        b = assemble.assemble_xp(minutes, team, rates, features, SEASON, seed=5)
        pd.testing.assert_frame_equal(a, b)


# --------------------------------------------------------------------------------------
# Single-fixture builder for focused component tests
# --------------------------------------------------------------------------------------


def single_fixture_inputs(
    season: int, players: list[dict]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """One home fixture (team 1 vs 2, lambdas 1.5/1.0); players are all on team 1.

    Each player dict supplies position and optional rate/minutes overrides.
    Defaults: certain 90 minutes, zero rates.
    """
    n = len(players)
    features = pd.DataFrame(
        {
            "season": season,
            "gw": 1,
            "player_code": np.arange(1, n + 1),
            "fpl_fixture_id": 500,
            "team_code": 1,
            "was_home": True,
            "position": [p["position"] for p in players],
        }
    )
    minutes = pd.DataFrame(
        {
            "season": season,
            "gw": 1,
            "player_code": np.arange(1, n + 1),
            "q0": [p.get("q0", 0.0) for p in players],
            "q1": [p.get("q1", 0.0) for p in players],
            "q2": [p.get("q2", 1.0) for p in players],
            "mu1": [p.get("mu1", 45.0) for p in players],
            "mu2": [p.get("mu2", 90.0) for p in players],
        }
    )
    rate_defaults = {
        "lam_goal": 0.0,
        "lam_assist": 0.0,
        "lam_saves": 0.0,
        "lam_defcon": 0.0,
        "defcon_disp": 10.0,
        "p_yellow": 0.0,
        "p_red": 0.0,
        "lam_og": 0.0,
    }
    rates = pd.DataFrame(
        {
            "season": season,
            "gw": 1,
            "player_code": np.arange(1, n + 1),
            **{k: [p.get(k, v) for p in players] for k, v in rate_defaults.items()},
        }
    )
    team = pd.DataFrame(
        {
            "season": [season],
            "fpl_fixture_id": [500],
            "home_lambda": [1.5],
            "away_lambda": [1.0],
            "p_cs_home": [0.3],
            "p_cs_away": [0.2],
        }
    )
    return minutes, team, rates, features


class TestDefconNegativeBinomial:
    def test_def_2025_known_nb_params(self) -> None:
        # DEF, full 90 (m_share = 1): NB(mean=8, r=4) vs CBIT threshold 10.
        minutes, team, rates, features = single_fixture_inputs(
            2025, [{"position": "DEF", "lam_defcon": 8.0, "defcon_disp": 4.0}]
        )
        xp = assemble.assemble_xp(minutes, team, rates, features, 2025)
        expected = 2 * nb_tail(8.0, 4, 10)
        assert xp["xp_defcon"].iloc[0] == pytest.approx(expected, abs=1e-9)
        assert 0.05 < expected < 0.9  # the reference itself is a non-trivial number

    def test_mid_threshold_is_12(self) -> None:
        minutes, team, rates, features = single_fixture_inputs(
            2025,
            [
                {"position": "MID", "lam_defcon": 8.0, "defcon_disp": 4.0},
                {"position": "DEF", "lam_defcon": 8.0, "defcon_disp": 4.0},
            ],
        )
        xp = assemble.assemble_xp(minutes, team, rates, features, 2025)
        assert xp["xp_defcon"].iloc[0] == pytest.approx(2 * nb_tail(8.0, 4, 12), abs=1e-9)
        assert xp["xp_defcon"].iloc[0] < xp["xp_defcon"].iloc[1]

    def test_gkp_ineligible(self) -> None:
        minutes, team, rates, features = single_fixture_inputs(
            2025, [{"position": "GKP", "lam_defcon": 20.0, "defcon_disp": 4.0}]
        )
        xp = assemble.assemble_xp(minutes, team, rates, features, 2025)
        assert xp["xp_defcon"].iloc[0] == 0.0

    def test_no_defcon_before_2025(self) -> None:
        minutes, team, rates, features = single_fixture_inputs(
            2024, [{"position": "DEF", "lam_defcon": 20.0, "defcon_disp": 4.0}]
        )
        xp = assemble.assemble_xp(minutes, team, rates, features, 2024)
        assert xp["xp_defcon"].iloc[0] == 0.0

    def test_poisson_fallback_when_dispersion_missing(self) -> None:
        minutes, team, rates, features = single_fixture_inputs(
            2025, [{"position": "DEF", "lam_defcon": 8.0, "defcon_disp": np.nan}]
        )
        xp = assemble.assemble_xp(minutes, team, rates, features, 2025)
        lam = 8.0
        p_poisson = 1.0 - sum(math.exp(-lam) * lam**k / math.factorial(k) for k in range(10))
        assert xp["xp_defcon"].iloc[0] == pytest.approx(2 * p_poisson, abs=1e-9)


class TestSavesAndConcede:
    def test_gk_saves_poisson_grid(self) -> None:
        minutes, team, rates, features = single_fixture_inputs(
            2025, [{"position": "GKP", "lam_saves": 4.5}]
        )
        xp = assemble.assemble_xp(minutes, team, rates, features, 2025)
        assert xp["xp_saves"].iloc[0] == pytest.approx(poisson_floor_ev(4.5, 3), abs=1e-9)

    def test_gk_concede_poisson_grid(self) -> None:
        minutes, team, rates, features = single_fixture_inputs(2025, [{"position": "GKP"}])
        xp = assemble.assemble_xp(minutes, team, rates, features, 2025)
        # Opponent (away) lambda 1.0 at full minutes.
        assert xp["xp_concede"].iloc[0] == pytest.approx(-poisson_floor_ev(1.0, 2), abs=1e-9)

    def test_outfield_no_save_points(self) -> None:
        minutes, team, rates, features = single_fixture_inputs(
            2025, [{"position": "MID", "lam_saves": 4.5}]
        )
        xp = assemble.assemble_xp(minutes, team, rates, features, 2025)
        assert xp["xp_saves"].iloc[0] == 0.0


class TestSeasonScoringDifferences:
    def gk_goal_xp(self, season: int) -> float:
        minutes, team, rates, features = single_fixture_inputs(
            season, [{"position": "GKP", "lam_goal": 0.3}]
        )
        xp = assemble.assemble_xp(minutes, team, rates, features, season)
        return float(xp["xp_goals"].iloc[0])

    def test_gk_goal_6_in_2023_vs_10_in_2024(self) -> None:
        xp23, xp24 = self.gk_goal_xp(2023), self.gk_goal_xp(2024)
        assert xp23 == pytest.approx(6 * 0.3)
        assert xp24 == pytest.approx(10 * 0.3)
        assert xp24 / xp23 == pytest.approx(10 / 6)

    def test_mid_goal_unchanged(self) -> None:
        for season in (2023, 2024):
            minutes, team, rates, features = single_fixture_inputs(
                season, [{"position": "MID", "lam_goal": 0.3}]
            )
            xp = assemble.assemble_xp(minutes, team, rates, features, season)
            assert xp["xp_goals"].iloc[0] == pytest.approx(5 * 0.3), season


class TestInvariantsAndValidation:
    def test_components_sum_on_random_frame(self) -> None:
        rng = np.random.default_rng(0)
        players = [
            {
                "position": rng.choice(["GKP", "DEF", "MID", "FWD"]),
                "q1": 0.2,
                "q2": 0.7,
                "lam_goal": float(rng.uniform(0, 0.8)),
                "lam_assist": float(rng.uniform(0, 0.5)),
                "lam_saves": float(rng.uniform(0, 5)),
                "lam_defcon": float(rng.uniform(0, 12)),
                "defcon_disp": float(rng.uniform(2, 20)),
                "p_yellow": float(rng.uniform(0, 0.3)),
                "p_red": float(rng.uniform(0, 0.03)),
                "lam_og": float(rng.uniform(0, 0.02)),
            }
            for _ in range(24)
        ]
        minutes, team, rates, features = single_fixture_inputs(2025, players)
        xp = assemble.assemble_xp(minutes, team, rates, features, 2025)
        np.testing.assert_allclose(
            xp[list(assemble.COMPONENT_COLUMNS)].sum(axis=1), xp["xp"], rtol=1e-12
        )
        # A realistic 24-player pool competes for ~6 bonus points.
        assert 5.9 <= xp["xp_bonus"].sum() <= 7.5

    def test_bonus_calibration_passthrough(self) -> None:
        minutes, team, rates, features = single_fixture_inputs(
            2025,
            [
                {"position": "MID", "lam_goal": 1.0},
                {"position": "MID", "lam_assist": 1.0},
                {"position": "MID"},
            ],
        )
        zero_noise = bonus.BonusCalibration(
            bias={}, sigma_intercept=0.0, sigma_slope=0.0, sigma_floor=0.0
        )
        xp = assemble.assemble_xp(
            minutes, team, rates, features, 2025, bonus_calibration=zero_noise
        )
        assert xp["xp_bonus"].tolist() == [3.0, 2.0, 1.0]

    def test_wrong_season_rows_raise(self) -> None:
        minutes, team, rates, features = single_fixture_inputs(2025, [{"position": "MID"}])
        with pytest.raises(ValueError, match="outside season"):
            assemble.assemble_xp(minutes, team, rates, features, 2024)

    def test_unmatched_minutes_row_raises(self) -> None:
        minutes, team, rates, features = single_fixture_inputs(2025, [{"position": "MID"}])
        with pytest.raises(ValueError, match="minutes"):
            assemble.assemble_xp(minutes.iloc[:0], team, rates, features, 2025)

    def test_unmatched_team_row_raises(self) -> None:
        minutes, team, rates, features = single_fixture_inputs(2025, [{"position": "MID"}])
        with pytest.raises(ValueError, match="team"):
            assemble.assemble_xp(minutes, team.iloc[:0], rates, features, 2025)

    def test_missing_feature_columns_raise(self) -> None:
        minutes, team, rates, features = single_fixture_inputs(2025, [{"position": "MID"}])
        with pytest.raises(ValueError, match="missing required columns"):
            assemble.assemble_xp(minutes, team, rates, features.drop(columns=["team_code"]), 2025)

    def test_unknown_position_raises(self) -> None:
        minutes, team, rates, features = single_fixture_inputs(2025, [{"position": "MID"}])
        features["position"] = "MNG"
        with pytest.raises(ValueError, match="unknown positions"):
            assemble.assemble_xp(minutes, team, rates, features, 2025)

    def test_zero_minutes_player_scores_zero(self) -> None:
        minutes, team, rates, features = single_fixture_inputs(
            2025,
            [
                {"position": "FWD", "q0": 1.0, "q1": 0.0, "q2": 0.0, "lam_goal": 1.0},
                {"position": "MID", "lam_goal": 0.5},
            ],
        )
        xp = assemble.assemble_xp(minutes, team, rates, features, 2025)
        row = xp.iloc[0]
        assert row["xp"] == pytest.approx(0.0, abs=1e-12)
