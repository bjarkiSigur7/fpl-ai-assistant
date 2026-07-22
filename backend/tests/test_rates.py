"""Offline tests for fplai.models.rates — synthetic frames with planted rates.

All frames conform to the ARCHITECTURE feature-frame contract: key columns
(season, gw, player_code[, fpl_fixture_id]), a position column, label columns and
numeric ``f_*`` feature columns. No network, no real data; every RNG is seeded.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fplai.models.rates import (
    KEY_COLUMNS,
    PREDICTION_COLUMNS,
    RatesModel,
    defcon_counts,
    nb_dispersion_mom,
    poisson_deviance,
)

#: Small-but-honest LightGBM settings so tests stay fast yet actually learn.
FAST = {
    "min_rows": 100,
    "min_events": 10,
    "lgbm_params": {"n_estimators": 150, "learning_rate": 0.1, "min_child_samples": 20},
}


def _frame(
    n: int,
    *,
    position: str = "MID",
    minutes: np.ndarray,
    season: int = 2024,
    start_code: int = 0,
    **overrides: np.ndarray,
) -> pd.DataFrame:
    """Contract-conformant synthetic frame; labels default to zero, ``f_*`` via overrides."""
    zeros = np.zeros(n, dtype=int)
    df = pd.DataFrame(
        {
            "season": season,
            "gw": (np.arange(n) % 38) + 1,
            "fpl_fixture_id": np.arange(n),
            "player_code": start_code + np.arange(n),
            "position": position,
            "minutes": minutes,
            "goals_scored": zeros,
            "assists": zeros,
            "saves": zeros,
            "yellow_cards": zeros,
            "red_cards": zeros,
            "own_goals": zeros,
        }
    )
    for col, values in overrides.items():
        df[col] = values
    return df


# ---------------------------------------------------------------------- exposure handling


def test_exposure_offset_recovers_per90_rates() -> None:
    """A 45-min player with 1 goal must resolve to a HIGHER per-90 rate than a 90-min one.

    Two groups separated by a feature, identical per-90 rate evidence would be violated if
    exposure were ignored: group A plays 45 min at true rate 2.0/90, group B plays 90 min at
    true rate 1.0/90 — both average ~1 goal per match.
    """
    rng = np.random.default_rng(0)
    n = 800
    frames = []
    for i, (mins, rate, f_x) in enumerate([(45, 2.0, 1.0), (90, 1.0, 0.0)]):
        minutes = np.full(n, mins)
        frames.append(
            _frame(
                n,
                minutes=minutes,
                start_code=i * n,
                goals_scored=rng.poisson(rate * mins / 90.0, size=n),
                f_x=np.full(n, f_x),
            )
        )
    df = pd.concat(frames, ignore_index=True)
    model = RatesModel(**FAST).fit(df)
    pred = model.predict(df)
    lam_a = float(pred.loc[df["f_x"] == 1.0, "lam_goal"].mean())
    lam_b = float(pred.loc[df["f_x"] == 0.0, "lam_goal"].mean())
    assert 1.6 < lam_a < 2.5, lam_a
    assert 0.75 < lam_b < 1.25, lam_b
    assert lam_a > 1.5 * lam_b


def test_planted_rate_recovery_from_feature() -> None:
    """Feature-driven per-90 goal rates are recovered (correlation + low relative error)."""
    rng = np.random.default_rng(1)
    n = 8000
    x = rng.uniform(0.0, 1.0, size=n)
    lam_true = np.exp(-1.5 + 1.2 * x)  # per-90, ~0.22 .. 0.74
    minutes = rng.integers(45, 91, size=n)
    goals = rng.poisson(lam_true * minutes / 90.0)
    df = _frame(n, minutes=minutes, goals_scored=goals, f_x=x)
    # single smooth feature -> keep the trees tiny so the GBM can't chase Poisson noise
    params = {
        "min_rows": 100,
        "min_events": 10,
        "lgbm_params": {
            "n_estimators": 200,
            "learning_rate": 0.1,
            "num_leaves": 7,
            "min_child_samples": 200,
        },
    }
    model = RatesModel(**params).fit(df)
    lam_hat = model.predict(df)["lam_goal"].to_numpy()
    assert np.corrcoef(lam_hat, lam_true)[0, 1] > 0.8
    assert float(np.mean(np.abs(lam_hat - lam_true) / lam_true)) < 0.35


def test_thin_position_falls_back_to_pooled_rate() -> None:
    """A position below min_rows gets the pooled empirical rate, not a (noisy) GBM."""
    rng = np.random.default_rng(2)
    n = 50  # < min_rows=100
    minutes = np.full(n, 90)
    goals = rng.poisson(0.5, size=n)
    df = _frame(n, position="FWD", minutes=minutes, goals_scored=goals, f_x=rng.normal(size=n))
    model = RatesModel(**FAST).fit(df)
    pred = model.predict(df)
    lam = pred["lam_goal"].to_numpy()
    assert np.allclose(lam, lam[0])  # constant fallback
    assert lam[0] == pytest.approx(goals.sum() / n, rel=1e-6)


# ---------------------------------------------------------------------- DefCon


def test_defcon_counts_label_resolution() -> None:
    """Native 2025 label wins; 2016-18 raw stats reconstruct CBIT (DEF) / CBIRT (MID/FWD)."""
    df = pd.DataFrame(
        {
            "position": ["DEF", "MID", "GKP", "DEF", "FWD"],
            "defensive_contribution": pd.array([pd.NA, pd.NA, pd.NA, 9, pd.NA], dtype="Int64"),
            "tackles": pd.array([3, 3, 1, 100, 2], dtype="Int64"),
            "clearances_blocks_interceptions": pd.array([4, 4, 0, 100, 1], dtype="Int64"),
            "recoveries": pd.array([5, 5, 2, 100, 3], dtype="Int64"),
        }
    )
    out = defcon_counts(df)
    assert out.tolist() == [7.0, 12.0, 0.0, 9.0, 6.0]


def test_defcon_counts_without_recoveries() -> None:
    """No recoveries column: DEF still reconstructs CBIT; MID/FWD stay NaN (no CBIRT)."""
    df = pd.DataFrame(
        {
            "position": ["DEF", "MID"],
            "tackles": pd.array([2, 2], dtype="Int64"),
            "clearances_blocks_interceptions": pd.array([5, 5], dtype="Int64"),
        }
    )
    out = defcon_counts(df)
    assert out.iloc[0] == 7.0
    assert np.isnan(out.iloc[1])


def test_nb_dispersion_mom_direct() -> None:
    """Moment estimator recovers NB2 alpha on overdispersed counts, ~0 on Poisson counts."""
    rng = np.random.default_rng(3)
    n = 4000
    mu = np.full(n, 8.0)
    alpha = 0.4
    gamma_mix = rng.gamma(shape=1.0 / alpha, scale=alpha, size=n)
    y_nb = rng.poisson(mu * gamma_mix)
    assert 0.25 < nb_dispersion_mom(y_nb, mu) < 0.6
    y_pois = rng.poisson(mu)
    assert nb_dispersion_mom(y_pois, mu) < 0.05


def test_defcon_model_dispersion_on_overdispersed_counts() -> None:
    """Full model: DefCon NB alpha per position is sane on planted overdispersed counts."""
    rng = np.random.default_rng(4)
    n = 4000
    x = rng.uniform(0.0, 1.0, size=n)
    lam_true = np.exp(2.0 + 0.5 * x)  # per-90 CBIT ~7.4 .. 12.2 (defender-like)
    minutes = rng.integers(60, 91, size=n)
    mu = lam_true * minutes / 90.0
    alpha = 0.3
    counts = rng.poisson(mu * rng.gamma(shape=1.0 / alpha, scale=alpha, size=n))
    df = _frame(
        n,
        position="DEF",
        season=2025,
        minutes=minutes,
        defensive_contribution=pd.array(counts, dtype="Int64"),
        f_x=x,
    )
    model = RatesModel(**FAST).fit(df)
    pred = model.predict(df)
    disp = float(pred["defcon_disp"].iloc[0])
    assert 0.15 < disp < 0.6, disp
    # rate model itself should track the planted rate too
    assert np.corrcoef(pred["lam_defcon"].to_numpy(), lam_true)[0, 1] > 0.6


# ---------------------------------------------------------------------- cards & own goals


def test_cards_empirical_bayes_shrinkage() -> None:
    """Player card rates shrink toward the position prior in proportion to exposure."""
    rng = np.random.default_rng(5)
    frames = []
    # 200 background players, 38 full matches each, heterogeneous true rates
    # (Gamma with mean 0.15/90 — real populations vary, which is what sets the EB
    # prior strength; a homogeneous population would legitimately shrink everything).
    true_rates = rng.gamma(shape=2.25, scale=0.15 / 2.25, size=200)
    for code in range(200):
        n = 38
        frames.append(
            _frame(
                n,
                minutes=np.full(n, 90),
                start_code=0,
                yellow_cards=rng.poisson(true_rates[code], size=n),
            ).assign(player_code=code)
        )
    # A genuinely dirty player: 15 yellows in 38 matches (raw 0.39/90).
    dirty = _frame(38, minutes=np.full(38, 90)).assign(player_code=500)
    dirty.loc[dirty.index[:15], "yellow_cards"] = 1
    # A one-match, one-yellow player (raw 1.0/90 — must be shrunk hard).
    lucky = _frame(1, minutes=np.array([90])).assign(player_code=501)
    lucky["yellow_cards"] = 1
    df = pd.concat([*frames, dirty, lucky], ignore_index=True)
    model = RatesModel(**FAST).fit(df)

    query = _frame(3, minutes=np.full(3, 90)).assign(player_code=[500, 501, 502])  # 502 unseen
    pred = model.predict(query).set_index("player_code")
    p_dirty = float(pred.loc[500, "p_yellow"])
    p_lucky = float(pred.loc[501, "p_yellow"])
    p_prior = float(pred.loc[502, "p_yellow"])
    assert 0.0 < p_prior < 0.35
    assert p_dirty > 1.5 * p_prior  # real signal survives shrinkage
    assert p_prior < p_lucky < 0.5 * (1.0 - np.exp(-1.0))  # heavy shrinkage of 1-match noise
    assert float(pred["p_red"].max()) < 0.02  # no reds in training -> tiny prior
    assert (pred["p_yellow"] < 1.0).all() and (pred["p_yellow"] > 0.0).all()


def test_own_goal_prior_is_tiny() -> None:
    rng = np.random.default_rng(6)
    n = 2000
    df = _frame(n, position="DEF", minutes=np.full(n, 90), own_goals=rng.poisson(0.008, size=n))
    model = RatesModel(**FAST).fit(df)
    lam_og = model.predict(df)["lam_og"]
    assert (lam_og > 0.0).all()
    assert float(lam_og.max()) < 0.05


# ---------------------------------------------------------------------- contract & lifecycle


def _mixed_frame(seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    parts = []
    for i, pos in enumerate(("GKP", "DEF", "MID", "FWD")):
        n = 300
        minutes = rng.integers(30, 91, size=n)
        e = minutes / 90.0
        part = _frame(
            n,
            position=pos,
            minutes=minutes,
            start_code=i * 10_000,
            goals_scored=rng.poisson((0.0 if pos == "GKP" else 0.2) * e),
            assists=rng.poisson(0.1 * e),
            saves=rng.poisson((3.0 if pos == "GKP" else 0.0) * e),
            yellow_cards=rng.poisson(0.15 * e),
            red_cards=rng.poisson(0.005 * e),
            own_goals=rng.poisson(0.004 * e),
            f_form=rng.normal(size=n),
            f_xg_p90_10=rng.uniform(0, 0.6, size=n),
        )
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def test_predict_contract_columns_and_position_zeros() -> None:
    df = _mixed_frame()
    model = RatesModel(**FAST).fit(df)
    pred = model.predict(df)
    expected = [*KEY_COLUMNS, "fpl_fixture_id", *PREDICTION_COLUMNS]
    assert pred.columns.tolist() == expected
    assert len(pred) == len(df)
    values = pred[list(PREDICTION_COLUMNS)]
    assert np.isfinite(values.to_numpy(dtype="float64")).all()
    gkp = df["position"] == "GKP"
    assert (pred.loc[gkp.to_numpy(), "lam_defcon"] == 0.0).all()
    assert (pred.loc[gkp.to_numpy(), "defcon_disp"] == 0.0).all()
    assert (pred.loc[(~gkp).to_numpy(), "lam_saves"] == 0.0).all()
    assert (pred.loc[gkp.to_numpy(), "lam_saves"] > 0.5).all()  # GK saves rate learned
    assert (pred.loc[gkp.to_numpy(), "lam_goal"] < 0.05).all()  # GK goals ~ never
    assert pred["p_yellow"].between(0.0, 1.0, inclusive="neither").all()
    assert pred["p_red"].between(0.0, 1.0, inclusive="left").all()


def test_save_load_roundtrip(tmp_path) -> None:
    df = _mixed_frame()
    model = RatesModel(**FAST).fit(df)
    pred = model.predict(df)
    model.save(tmp_path / "rates")
    reloaded = RatesModel.load(tmp_path / "rates")
    pred2 = reloaded.predict(df)
    pd.testing.assert_frame_equal(pred, pred2)


def test_evaluate_reports_deviance_and_calibration() -> None:
    df = _mixed_frame()
    model = RatesModel(**FAST).fit(df)
    results = model.evaluate(df)
    for name in ("goal", "assist", "saves", "yellow", "red"):
        dev = results[f"poisson_deviance_{name}"]
        assert np.isfinite(dev) and dev >= 0.0
    calib = results["calibration_goal"]
    assert 1 <= len(calib) <= 10
    for row in calib:
        assert set(row) == {"decile", "pred_rate", "obs_rate", "n"}
        assert row["n"] > 0
    # no DefCon labels in this frame -> no defcon deviance key
    assert "poisson_deviance_defcon" not in results


def test_error_paths() -> None:
    df = _mixed_frame()
    model = RatesModel(**FAST)
    with pytest.raises(RuntimeError):
        model.predict(df)
    with pytest.raises(KeyError):
        RatesModel(**FAST).fit(df.drop(columns=["saves"]))
    model.fit(df)
    with pytest.raises(KeyError):
        model.predict(df.drop(columns=["player_code"]))
    with pytest.raises(KeyError):
        model.predict(df.drop(columns=["position"]))


def test_poisson_deviance_zero_for_perfect_fit() -> None:
    y = np.array([0.0, 1.0, 2.0, 5.0])
    assert poisson_deviance(y, np.where(y > 0, y, 1e-9)) == pytest.approx(0.0, abs=1e-6)
