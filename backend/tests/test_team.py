"""Offline tests for the Dixon-Coles team model (fplai.models.team).

All tests run on synthetic fixtures conforming to the fixtures.parquet /
odds.parquet contracts (ARCHITECTURE.md); no network, deterministic seeds.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import check_grad
from scipy.stats import spearmanr

from fplai.models.team import (
    MAX_GOALS,
    TeamModel,
    _grid_outcomes,
    _scoreline_grid,
    attach_fixture_ids,
    demargin_1x2,
    empty_stadium_mask,
)

PROCESSED = Path(__file__).resolve().parents[2] / "data" / "processed"

N_TEAMS = 10
CODES = list(range(101, 101 + N_TEAMS))
TRUE_LOG_A = np.linspace(0.4, -0.4, N_TEAMS)  # code 101 strongest attack
TRUE_LOG_B = np.linspace(-0.3, 0.3, N_TEAMS) + np.log(1.2)  # code 101 best defence
TRUE_GAMMA = 1.3


def synthetic_fixtures(
    *,
    n_rounds: int = 4,
    seed: int = 42,
    gamma: float = TRUE_GAMMA,
    log_a: np.ndarray = TRUE_LOG_A,
    log_b: np.ndarray = TRUE_LOG_B,
    start_season: int = 2020,
    empty: bool = False,
) -> pd.DataFrame:
    """Simulate independent-Poisson double round-robins with planted strengths.

    One "round" is a full home-and-away round-robin treated as one season of data.
    """
    rng = np.random.default_rng(seed)
    rows = []
    fid = 0
    for rep in range(n_rounds):
        for i in range(N_TEAMS):
            for j in range(N_TEAMS):
                if i == j:
                    continue
                lam = np.exp(log_a[i] + log_b[j]) * gamma
                mu = np.exp(log_a[j] + log_b[i])
                fid += 1
                kickoff = pd.Timestamp("2020-08-01", tz="UTC") + pd.Timedelta(
                    days=365 * rep + (i * N_TEAMS + j) % 280
                )
                rows.append(
                    {
                        "season": start_season + rep,
                        "gw": 1 + (i * N_TEAMS + j) % 38,
                        "fpl_fixture_id": fid,
                        "kickoff_utc": kickoff,
                        "home_team_code": CODES[i],
                        "away_team_code": CODES[j],
                        "home_goals": int(rng.poisson(lam)),
                        "away_goals": int(rng.poisson(mu)),
                        "finished": True,
                        "void": False,
                        "empty_stadium": empty,
                    }
                )
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def fitted() -> tuple[TeamModel, pd.DataFrame]:
    fx = synthetic_fixtures()
    model = TeamModel(decay_halflife_days=1e6, empty_gamma="zero").fit(fx)
    return model, fx


# ------------------------------------------------------------------ fitting


def test_fit_recovers_planted_ordering(fitted: tuple[TeamModel, pd.DataFrame]) -> None:
    model, _ = fitted
    order = [model.team_codes_.index(c) for c in CODES]
    attack_corr = spearmanr(model.log_a_[order], TRUE_LOG_A).statistic
    defence_corr = spearmanr(model.log_b_[order], TRUE_LOG_B).statistic
    assert attack_corr > 0.85
    assert defence_corr > 0.85
    # identifiability constraint
    assert abs(model.log_a_.sum()) < 1e-8
    # home advantage recovered within a reasonable band
    assert 1.15 < np.exp(model.log_gamma_) < 1.5


def test_gradient_matches_finite_differences() -> None:
    rng = np.random.default_rng(7)
    n = 6
    pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
    h_idx = np.array([p[0] for p in pairs])
    a_idx = np.array([p[1] for p in pairs])
    m = len(pairs)
    x = rng.poisson(1.4, m)
    y = rng.poisson(1.1, m)
    w = rng.uniform(0.3, 1.0, m)
    empty = rng.random(m) < 0.25
    for fit_empty in (True, False):
        n_params = (n - 1) + n + 1 + (1 if fit_empty else 0) + 1
        theta = rng.normal(0, 0.2, n_params)
        theta[-1] = -0.08
        args = (n, h_idx, a_idx, x, y, w, empty, fit_empty)
        err = check_grad(
            lambda t: TeamModel._nll_and_grad(t, *args)[0],
            lambda t: TeamModel._nll_and_grad(t, *args)[1],
            theta,
            epsilon=1e-7,
        )
        assert err < 1e-4


def test_decay_downweights_old_seasons() -> None:
    """Team flips from strong to weak; short half-life must track the recent regime."""
    strong = TRUE_LOG_A.copy()
    weak = TRUE_LOG_A.copy()
    weak[0], weak[-1] = TRUE_LOG_A[-1], TRUE_LOG_A[0]  # team 101 becomes worst
    old = synthetic_fixtures(n_rounds=2, seed=1, log_a=strong, start_season=2018)
    recent = synthetic_fixtures(n_rounds=2, seed=2, log_a=weak, start_season=2020)
    recent["fpl_fixture_id"] += old["fpl_fixture_id"].max()
    recent["kickoff_utc"] += pd.Timedelta(days=2 * 365)
    fx = pd.concat([old, recent], ignore_index=True)

    short = TeamModel(decay_halflife_days=180.0, empty_gamma="zero").fit(fx)
    flat = TeamModel(decay_halflife_days=1e6, empty_gamma="zero").fit(fx)
    i = short.team_codes_.index(101)
    # with strong decay, team 101's rating reflects the recent (weak) regime
    assert short.log_a_[i] < flat.log_a_[i] - 0.1
    rank_short = (short.log_a_ > short.log_a_[i]).sum()
    assert rank_short >= N_TEAMS - 3  # near the bottom under decay


def test_fit_respects_train_end_cutoff() -> None:
    fx = synthetic_fixtures()
    cutoff = pd.Timestamp("2021-08-01", tz="UTC")
    model = TeamModel(decay_halflife_days=1e6, empty_gamma="zero").fit(fx, train_end=cutoff)
    assert model.n_matches_ == int((fx["kickoff_utc"] < cutoff).sum())
    assert model.as_of_ == cutoff


def test_empty_stadium_gamma_fitted_separately() -> None:
    normal = synthetic_fixtures(n_rounds=2, seed=5, gamma=1.35)
    empty = synthetic_fixtures(n_rounds=2, seed=6, gamma=1.0, start_season=2022, empty=True)
    empty["fpl_fixture_id"] += normal["fpl_fixture_id"].max()
    empty["kickoff_utc"] += pd.Timedelta(days=2 * 365)
    fx = pd.concat([normal, empty], ignore_index=True)
    model = TeamModel(decay_halflife_days=1e6, empty_gamma="fit").fit(fx)
    assert np.exp(model.log_gamma_) > np.exp(model.log_gamma_empty_) + 0.1


def test_empty_stadium_mask_from_rules() -> None:
    fx = pd.DataFrame(
        {
            "season": [2019, 2019, 2020, 2021],
            "gw": [29, 31, 5, 5],
            "kickoff_utc": pd.to_datetime(
                ["2020-03-07", "2020-06-20", "2020-10-01", "2021-10-01"], utc=True
            ),
        }
    )
    mask = empty_stadium_mask(fx)
    assert mask.tolist() == [False, True, True, False]


# ------------------------------------------------------------------ grid / predictions


def test_tau_correction_moves_low_score_mass() -> None:
    lam = np.array([1.4])
    mu = np.array([1.1])
    base = _scoreline_grid(lam, mu, rho=0.0)
    corrected = _scoreline_grid(lam, mu, rho=-0.08)
    # negative rho boosts 0-0 and 1-1 at the expense of 0-1 and 1-0
    assert corrected[0, 0, 0] > base[0, 0, 0]
    assert corrected[0, 1, 1] > base[0, 1, 1]
    assert corrected[0, 0, 1] < base[0, 0, 1]
    assert corrected[0, 1, 0] < base[0, 1, 0]
    assert corrected.sum() == pytest.approx(1.0)


def test_predict_fixtures_contract(fitted: tuple[TeamModel, pd.DataFrame]) -> None:
    model, fx = fitted
    pred = model.predict_fixtures(fx.head(30))
    expected_cols = [
        "season",
        "fpl_fixture_id",
        "home_lambda",
        "away_lambda",
        "p_cs_home",
        "p_cs_away",
        "p_home_win",
        "p_draw",
        "p_away_win",
    ]
    assert list(pred.columns) == expected_cols
    onextwo = pred[["p_home_win", "p_draw", "p_away_win"]].to_numpy()
    assert np.allclose(onextwo.sum(axis=1), 1.0)
    assert ((onextwo >= 0) & (onextwo <= 1)).all()
    assert (pred["home_lambda"] > 0).all() and (pred["away_lambda"] > 0).all()
    # CS probabilities consistent with the Poisson rates (τ shifts them only slightly)
    approx_cs_home = np.exp(-pred["away_lambda"].to_numpy())
    assert np.abs(pred["p_cs_home"].to_numpy() - approx_cs_home).max() < 0.05
    assert ((pred["p_cs_home"] > 0) & (pred["p_cs_home"] < 1)).all()


def test_cs_prob_decreases_with_opponent_attack(fitted: tuple[TeamModel, pd.DataFrame]) -> None:
    model, _ = fitted
    strongest, weakest = CODES[0], CODES[-1]
    neutral = CODES[N_TEAMS // 2]
    frame = pd.DataFrame(
        {
            "season": [2024, 2024],
            "gw": [1, 1],
            "fpl_fixture_id": [90001, 90002],
            "home_team_code": [neutral, neutral],
            "away_team_code": [strongest, weakest],
        }
    )
    pred = model.predict_fixtures(frame)
    assert pred.loc[0, "p_cs_home"] < pred.loc[1, "p_cs_home"]
    assert pred.loc[0, "p_home_win"] < pred.loc[1, "p_home_win"]


def test_grid_outcomes_partition() -> None:
    grid = _scoreline_grid(np.array([1.8, 0.9]), np.array([1.2, 1.4]), rho=-0.05)
    out = _grid_outcomes(grid)
    total = out["p_home_win"] + out["p_draw"] + out["p_away_win"]
    assert np.allclose(total, 1.0)
    assert grid.shape == (2, MAX_GOALS + 1, MAX_GOALS + 1)


# ------------------------------------------------------------------ promoted seeding


def test_promoted_prior_for_unseen_team(fitted: tuple[TeamModel, pd.DataFrame]) -> None:
    model, _ = fitted
    unseen = 999
    frame = pd.DataFrame(
        {
            "season": [2024],
            "gw": [1],
            "fpl_fixture_id": [90010],
            "home_team_code": [CODES[0]],
            "away_team_code": [unseen],
        }
    )
    pred = model.predict_fixtures(frame)  # must not raise
    assert np.isfinite(pred.loc[0, "home_lambda"]) and np.isfinite(pred.loc[0, "away_lambda"])

    # §3.6 hook: a stronger multiplier raises the unseen team's attack and defence
    boosted = TeamModel(
        decay_halflife_days=1e6, empty_gamma="zero", promoted_strength={unseen: 1.4}
    )
    boosted.team_codes_ = model.team_codes_
    boosted.log_a_ = model.log_a_
    boosted.log_b_ = model.log_b_
    boosted.log_gamma_ = model.log_gamma_
    boosted.log_gamma_empty_ = model.log_gamma_empty_
    boosted.rho_ = model.rho_
    boosted.promoted_prior_ = model.promoted_prior_
    pred_boost = boosted.predict_fixtures(frame)
    assert pred_boost.loc[0, "away_lambda"] > pred.loc[0, "away_lambda"]  # more attack
    assert pred_boost.loc[0, "home_lambda"] < pred.loc[0, "home_lambda"]  # tighter defence


def test_promoted_codes_detected_across_seasons() -> None:
    fx = synthetic_fixtures(n_rounds=2, seed=11)
    # replace team 110 with a new code 555 in the second season only -> promoted
    second = fx["season"] == fx["season"].max()
    for col in ("home_team_code", "away_team_code"):
        fx.loc[second & (fx[col] == CODES[-1]), col] = 555
    model = TeamModel(decay_halflife_days=1e6, empty_gamma="zero").fit(fx)
    assert model.promoted_codes_ == [555]
    i = model.team_codes_.index(555)
    assert model.promoted_prior_ == (float(model.log_a_[i]), float(model.log_b_[i]))


# ------------------------------------------------------------------ odds


def make_odds(
    pred: pd.DataFrame, p_h: float, p_d: float, p_a: float, *, margin: float = 1.05
) -> pd.DataFrame:
    """Bookmaker odds frame implying (p_h, p_d, p_a) after de-margining."""
    return pd.DataFrame(
        {
            "season": pred["season"].to_numpy(),
            "fpl_fixture_id": pred["fpl_fixture_id"].to_numpy(),
            "odds_h": 1.0 / (p_h * margin),  # inverse-prob odds with the margin baked in
            "odds_d": 1.0 / (p_d * margin),
            "odds_a": 1.0 / (p_a * margin),
        }
    )


def test_demargin_1x2_normalises() -> None:
    odds = pd.DataFrame({"odds_h": [2.0], "odds_d": [3.6], "odds_a": [4.0]})
    p = demargin_1x2(odds)
    assert p[["p_h", "p_d", "p_a"]].to_numpy().sum() == pytest.approx(1.0)
    assert p.loc[0, "p_h"] > p.loc[0, "p_d"] > p.loc[0, "p_a"]
    assert np.isnan(p.loc[0, "p_over25"])


def test_blend_moves_probabilities_toward_odds(fitted: tuple[TeamModel, pd.DataFrame]) -> None:
    model, fx = fitted
    pred = model.predict_fixtures(fx.head(1))
    dc_ph = float(pred.loc[0, "p_home_win"])
    target_h = min(dc_ph + 0.2, 0.9)
    rest = 1.0 - target_h
    odds = make_odds(pred, target_h, rest * 0.5, rest * 0.5)
    blended = model.blend_odds(pred, odds, weight=0.7)
    assert bool(blended.loc[0, "odds_blended"])
    new_ph = float(blended.loc[0, "p_home_win"])
    expect = 0.7 * target_h + 0.3 * dc_ph
    assert new_ph == pytest.approx(expect, abs=0.01)
    assert dc_ph < new_ph  # moved toward the (stronger) odds-implied home prob
    # lambdas re-solved: stronger home favourite => bigger rate gap
    assert blended.loc[0, "home_lambda"] > pred.loc[0, "home_lambda"]


def test_blend_weight_extremes(fitted: tuple[TeamModel, pd.DataFrame]) -> None:
    model, fx = fitted
    pred = model.predict_fixtures(fx.head(1))
    odds = make_odds(pred, 0.6, 0.25, 0.15)
    w0 = model.blend_odds(pred, odds, weight=0.0)
    assert float(w0.loc[0, "p_home_win"]) == pytest.approx(
        float(pred.loc[0, "p_home_win"]), abs=1e-3
    )
    w1 = model.blend_odds(pred, odds, weight=1.0)
    assert float(w1.loc[0, "p_home_win"]) == pytest.approx(0.6, abs=0.01)
    assert float(w1.loc[0, "p_away_win"]) == pytest.approx(0.15, abs=0.01)


def test_blend_leaves_unmatched_fixtures_alone(fitted: tuple[TeamModel, pd.DataFrame]) -> None:
    model, fx = fitted
    pred = model.predict_fixtures(fx.head(2))
    odds = make_odds(pred.head(1), 0.5, 0.3, 0.2)
    blended = model.blend_odds(pred, odds, weight=0.7)
    assert bool(blended.loc[0, "odds_blended"])
    assert not bool(blended.loc[1, "odds_blended"])
    assert float(blended.loc[1, "p_home_win"]) == pytest.approx(float(pred.loc[1, "p_home_win"]))


def test_blend_with_totals_market(fitted: tuple[TeamModel, pd.DataFrame]) -> None:
    model, fx = fitted
    pred = model.predict_fixtures(fx.head(1))
    odds = make_odds(pred, 0.5, 0.28, 0.22)
    odds["odds_over25"] = 1.0 / (0.65 * 1.03)
    odds["odds_under25"] = 1.0 / (0.35 * 1.03)
    blended = model.blend_odds(pred, odds, weight=1.0)
    total = blended.loc[0, "home_lambda"] + blended.loc[0, "away_lambda"]
    grid = _scoreline_grid(
        np.array([blended.loc[0, "home_lambda"]]),
        np.array([blended.loc[0, "away_lambda"]]),
        model.rho_,
    )
    p_over = float(_grid_outcomes(grid)["p_over25"][0])
    # three residuals / two unknowns -> approximate match, but must move toward 0.65
    base_over = float(
        _grid_outcomes(
            _scoreline_grid(
                pred["home_lambda"].to_numpy(), pred["away_lambda"].to_numpy(), model.rho_
            )
        )["p_over25"][0]
    )
    assert abs(p_over - 0.65) < abs(base_over - 0.65) + 1e-9
    assert total > 0


def test_attach_fixture_ids_by_name() -> None:
    fixtures = pd.DataFrame(
        {
            "season": [2024, 2024],
            "gw": [1, 1],
            "fpl_fixture_id": [11, 12],
            "home_team_code": [1, 6],
            "away_team_code": [6, 1],
        }
    )
    teams = pd.DataFrame(
        {
            "season": [2024, 2024],
            "fpl_team_id": [1, 2],
            "team_code": [1, 6],
            "name": ["Man Utd", "Spurs"],
            "short_name": ["MUN", "TOT"],
        }
    )
    odds = pd.DataFrame(
        {
            "season": [2024, 2024, 2024],
            "home_footballdata_name": ["Man United", "Tottenham", "Chelsea"],
            "away_footballdata_name": ["Tottenham", "Man United", "Arsenal"],
            "odds_h": [2.0, 2.0, 2.0],
            "odds_d": [3.4, 3.4, 3.4],
            "odds_a": [3.8, 3.8, 3.8],
        }
    )
    attached = attach_fixture_ids(odds, fixtures, teams)
    assert attached["fpl_fixture_id"].tolist()[:2] == [11, 12]
    assert pd.isna(attached["fpl_fixture_id"].iloc[2])


# ------------------------------------------------------------------ evaluate / persistence


def test_evaluate_beats_uniform_on_synthetic(fitted: tuple[TeamModel, pd.DataFrame]) -> None:
    model, _ = fitted
    holdout = synthetic_fixtures(n_rounds=1, seed=99, start_season=2024)
    metrics = model.evaluate(holdout)
    assert metrics["log_loss_uniform"] == pytest.approx(np.log(3.0))
    assert metrics["brier_uniform"] == pytest.approx(2.0 / 3.0)
    assert metrics["log_loss"] < metrics["log_loss_uniform"]
    assert metrics["brier"] < metrics["brier_uniform"]


def test_evaluate_odds_benchmark(fitted: tuple[TeamModel, pd.DataFrame]) -> None:
    model, _ = fitted
    holdout = synthetic_fixtures(n_rounds=1, seed=100, start_season=2024)
    pred = model.predict_fixtures(holdout)
    odds = make_odds(pred, 0.45, 0.3, 0.25)
    metrics = model.evaluate(holdout, odds=odds)
    assert metrics["n_odds"] == metrics["n"]
    for key in ("log_loss_odds", "brier_odds", "log_loss_model_on_odds_subset"):
        assert np.isfinite(metrics[key])


def test_save_load_roundtrip(tmp_path: Path, fitted: tuple[TeamModel, pd.DataFrame]) -> None:
    model, fx = fitted
    model.save(tmp_path)
    restored = TeamModel.load(tmp_path)
    pred_a = model.predict_fixtures(fx.head(20))
    pred_b = restored.predict_fixtures(fx.head(20))
    pd.testing.assert_frame_equal(pred_a, pred_b)
    assert restored.team_codes_ == model.team_codes_
    assert restored.rho_ == pytest.approx(model.rho_)


def test_fit_is_deterministic() -> None:
    fx = synthetic_fixtures(n_rounds=2, seed=3)
    m1 = TeamModel(decay_halflife_days=1e6, empty_gamma="zero").fit(fx)
    m2 = TeamModel(decay_halflife_days=1e6, empty_gamma="zero").fit(fx)
    assert np.array_equal(m1.log_a_, m2.log_a_)
    assert m1.rho_ == m2.rho_


# ------------------------------------------------------------------ real data (optional)


@pytest.mark.skipif(
    not (PROCESSED / "fixtures.parquet").exists(), reason="processed tables not built"
)
def test_real_data_close_to_odds_benchmark() -> None:
    """Train on seasons <= 2024, hold out 2025; DC must beat uniform and sit near odds."""
    fixtures = pd.read_parquet(PROCESSED / "fixtures.parquet")
    if not {2024, 2025} <= set(fixtures["season"].unique()):
        pytest.skip("need 2024 train and 2025 holdout seasons on disk")
    train = fixtures[fixtures["season"] <= 2024]
    holdout = fixtures[fixtures["season"] == 2025]
    odds = None
    teams = None
    if (PROCESSED / "odds.parquet").exists() and (PROCESSED / "teams.parquet").exists():
        odds = pd.read_parquet(PROCESSED / "odds.parquet")
        teams = pd.read_parquet(PROCESSED / "teams.parquet")
    model = TeamModel().fit(train, odds=odds, teams=teams)
    metrics = model.evaluate(holdout, teams=teams)
    assert metrics["log_loss"] < metrics["log_loss_uniform"]
    assert metrics["brier"] < metrics["brier_uniform"]
    # sensible home advantage; empty-stadium γ near 1 (2019 restart + 2020-21 regime)
    assert 1.0 < np.exp(model.log_gamma_) < 1.6
    assert 0.8 < np.exp(model.log_gamma_empty_) < 1.15
    if "log_loss_odds" in metrics:
        # the bar: close to closing odds (beating them is not expected)
        assert metrics["log_loss_model_on_odds_subset"] < metrics["log_loss_odds"] + 0.10
