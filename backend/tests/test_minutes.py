"""Offline tests for fplai.models.minutes (heuristic v0 + LightGBM v1)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fplai.models.minutes import (
    HeuristicMinutesModel,
    LgbMinutesModel,
    build_minutes_feature_frame,
)

PROCESSED = Path(__file__).resolve().parents[2] / "data" / "processed"
PLAYER_MATCH = PROCESSED / "player_match.parquet"


# ---------------------------------------------------------------------------
# Synthetic inputs conforming to the features contract (f_* columns + labels)
# ---------------------------------------------------------------------------


def heuristic_frame() -> pd.DataFrame:
    """Five archetypes: nailed starter, rotation, super-sub, injured, 50% doubt."""
    return pd.DataFrame(
        {
            "season": [2025] * 5,
            "gw": [10] * 5,
            "player_code": [1, 2, 3, 4, 5],
            "fpl_fixture_id": [100] * 5,
            "position": ["DEF", "MID", "FWD", "MID", "FWD"],
            "price": [55, 75, 60, 80, 90],
            "f_price": [55.0, 75.0, 60.0, 80.0, 90.0],
            "f_start_share_5": [1.0, 0.5, 0.0, 1.0, 1.0],
            "f_cameo_share_5": [0.0, 0.2, 0.8, 0.0, 0.0],
            "f_start_minutes_mean_5": [90.0, 78.0, np.nan, 90.0, 88.0],
            "f_cameo_minutes_mean_10": [np.nan, 25.0, 18.0, np.nan, np.nan],
            "f_avail_chance": [np.nan, np.nan, np.nan, np.nan, 0.5],
            "f_avail_is_out": [0.0, 0.0, 0.0, 1.0, 0.0],
            "f_avail_is_doubt": [0.0, 0.0, 0.0, 0.0, 1.0],
        }
    )


def synthetic_training_frame(
    n_players: int = 240, seasons: tuple[int, ...] = (2022, 2023, 2024), n_gws: int = 30
) -> pd.DataFrame:
    """Contract-conforming f_* frame with known latent start propensities.

    Each player has a stable P(start); features are noisy views of it, labels
    are drawn from the generative process the two-stage model assumes.
    """
    rng = np.random.default_rng(7)
    start_prop = rng.beta(1.2, 1.8, size=n_players)
    cameo_prop = rng.beta(1.5, 3.0, size=n_players) * (1.0 - start_prop)
    positions = rng.choice(["GKP", "DEF", "MID", "FWD"], size=n_players, p=[0.1, 0.35, 0.35, 0.2])
    # Attackers get hooked more often -> stage B has position signal to learn.
    hook_p = np.where(np.isin(positions, ["MID", "FWD"]), 0.11, 0.03)
    rows: list[dict[str, object]] = []
    for season in seasons:
        for gw in range(1, n_gws + 1):
            starts = rng.random(n_players) < start_prop
            cameos = (~starts) & (rng.random(n_players) < cameo_prop / (1 - start_prop + 1e-9))
            minutes = np.zeros(n_players)
            hooked = rng.random(n_players) < hook_p
            minutes[starts] = np.where(
                hooked[starts], rng.integers(20, 60, starts.sum()), rng.integers(60, 91, starts.sum())
            )
            minutes[cameos] = rng.integers(1, 35, cameos.sum())
            # Independent noise per feature: combining views beats any single one.
            noise = rng.normal(0, 0.15, n_players)
            noise2 = rng.normal(0, 0.15, n_players)
            noise3 = rng.normal(0, 12.0, n_players)
            for p in range(n_players):
                rows.append(
                    {
                        "season": season,
                        "gw": gw,
                        "player_code": p + 1,
                        "fpl_fixture_id": gw * 1000 + p,
                        "position": positions[p],
                        "price": 45 + int(60 * start_prop[p]),
                        "void_gw": False,
                        "minutes": int(minutes[p]),
                        "starts": int(starts[p]),
                        "f_start_share_5": float(np.clip(start_prop[p] + noise[p], 0, 1)),
                        "f_cameo_share_5": float(np.clip(cameo_prop[p] + noise2[p], 0, 1)),
                        "f_minutes_mean_5": float(
                            np.clip(90 * start_prop[p] + 20 * cameo_prop[p] + noise3[p], 0, 90)
                        ),
                        "f_price": 45.0 + 60 * start_prop[p],
                        "f_subs_regime": 5.0,
                        "f_gw_phase": gw / 38.0,
                        "f_pos_GKP": float(positions[p] == "GKP"),
                        "f_pos_DEF": float(positions[p] == "DEF"),
                        "f_pos_MID": float(positions[p] == "MID"),
                        "f_pos_FWD": float(positions[p] == "FWD"),
                    }
                )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Heuristic (v0)
# ---------------------------------------------------------------------------


def test_heuristic_known_patterns() -> None:
    pred = HeuristicMinutesModel().predict(heuristic_frame())
    assert list(pred.columns[:4]) == ["season", "gw", "player_code", "fpl_fixture_id"]
    nailed, rotation, supersub, injured, doubt = pred.to_dict("records")
    assert nailed["q2"] > 0.85
    assert 0.3 < rotation["q2"] < 0.6
    assert supersub["q1"] > 0.6 and supersub["q2"] < 0.1
    assert injured["q0"] > 0.95
    # 50% chance-of-playing roughly halves q2 vs the equivalent nailed starter
    assert doubt["q2"] == pytest.approx(nailed["q2"] * 0.5, rel=0.1)


def test_heuristic_probabilities_sum_and_mu_ranges() -> None:
    pred = HeuristicMinutesModel().predict(heuristic_frame())
    np.testing.assert_allclose(pred[["q0", "q1", "q2"]].sum(axis=1), 1.0, atol=1e-9)
    assert ((pred["mu1"] >= 1) & (pred["mu1"] <= 59)).all()
    assert ((pred["mu2"] >= 60) & (pred["mu2"] <= 90)).all()


def test_heuristic_defaults_for_unseen_players() -> None:
    """No history at all -> position x price-band calibrated defaults."""
    df = pd.DataFrame(
        {
            "season": [2025, 2025],
            "gw": [1, 1],
            "player_code": [11, 12],
            "position": ["GKP", "MID"],
            "price": [60, 45],
            "f_price": [60.0, 45.0],
        }
    )
    pred = HeuristicMinutesModel().predict(df)
    # mid-priced GKP: heavy favourite to start; cheap MID: fodder
    assert pred.loc[0, "q2"] > 0.6
    assert pred.loc[1, "q2"] < 0.15
    np.testing.assert_allclose(pred[["q0", "q1", "q2"]].sum(axis=1), 1.0, atol=1e-9)


def test_heuristic_fatigue_dampening() -> None:
    model = HeuristicMinutesModel()
    base = model.predict(heuristic_frame())
    damp = model.predict(heuristic_frame(), fatigue_dampening={1: 0.5})
    # Dampened player: q2 halved pre-renormalization, so strictly lower share.
    assert damp.loc[0, "q2"] < base.loc[0, "q2"] * 0.75
    assert damp.loc[0, "q0"] > base.loc[0, "q0"]
    np.testing.assert_allclose(damp[["q0", "q1", "q2"]].sum(axis=1), 1.0, atol=1e-9)
    # Other players untouched.
    pd.testing.assert_frame_equal(base.iloc[1:], damp.iloc[1:])


def test_fatigue_dampening_validation() -> None:
    with pytest.raises(ValueError):
        HeuristicMinutesModel().predict(heuristic_frame(), fatigue_dampening={1: 1.5})


def test_heuristic_evaluate_contract() -> None:
    df = synthetic_training_frame(n_players=80, seasons=(2024,), n_gws=10)
    result = HeuristicMinutesModel().evaluate(df)
    assert set(result) >= {"n", "bucket_log_loss", "brier_p0", "calibration"}
    assert 0.0 < result["bucket_log_loss"] < 2.0  # sane on very noisy features
    cal = result["calibration"]
    assert set(cal.columns) == {"bucket", "decile", "n", "p_pred_mean", "p_realized"}
    assert set(cal["bucket"]) == {"q0", "q1", "q2"}
    assert cal["n"].sum() == 3 * result["n"]


# ---------------------------------------------------------------------------
# LightGBM (v1)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def synthetic_split() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = synthetic_training_frame()
    return df[df["season"] <= 2023], df[df["season"] == 2024]


@pytest.fixture(scope="module")
def fitted_lgb(synthetic_split: tuple[pd.DataFrame, pd.DataFrame]) -> LgbMinutesModel:
    train, _ = synthetic_split
    return LgbMinutesModel(n_estimators=120).fit(train)


def test_lgb_beats_heuristic_on_synthetic(
    synthetic_split: tuple[pd.DataFrame, pd.DataFrame], fitted_lgb: LgbMinutesModel
) -> None:
    _, test = synthetic_split
    lgb_ll = fitted_lgb.evaluate(test)["bucket_log_loss"]
    heur_ll = HeuristicMinutesModel().evaluate(test)["bucket_log_loss"]
    assert lgb_ll < heur_ll
    assert lgb_ll < np.log(3)


def test_lgb_output_contract(
    synthetic_split: tuple[pd.DataFrame, pd.DataFrame], fitted_lgb: LgbMinutesModel
) -> None:
    _, test = synthetic_split
    pred = fitted_lgb.predict(test)
    assert {"season", "gw", "player_code", "q0", "q1", "q2", "mu1", "mu2"} <= set(pred.columns)
    assert len(pred) == len(test)
    np.testing.assert_allclose(pred[["q0", "q1", "q2"]].sum(axis=1), 1.0, atol=1e-9)
    assert ((pred["mu1"] >= 1) & (pred["mu1"] <= 59)).all()
    assert ((pred["mu2"] >= 60) & (pred["mu2"] <= 90)).all()


def test_lgb_fatigue_hook(
    synthetic_split: tuple[pd.DataFrame, pd.DataFrame], fitted_lgb: LgbMinutesModel
) -> None:
    _, test = synthetic_split
    one = test.head(50)
    code = int(one.iloc[0]["player_code"])
    base = fitted_lgb.predict(one)
    damp = fitted_lgb.predict(one, fatigue_dampening={code: 0.0})
    hit = damp["player_code"] == code
    assert (damp.loc[hit, "q2"] < 0.01).all()
    assert (damp.loc[hit, "q2"] < base.loc[hit, "q2"]).all()
    np.testing.assert_allclose(damp[["q0", "q1", "q2"]].sum(axis=1), 1.0, atol=1e-9)


def test_lgb_missing_feature_columns_tolerated(
    synthetic_split: tuple[pd.DataFrame, pd.DataFrame], fitted_lgb: LgbMinutesModel
) -> None:
    _, test = synthetic_split
    reduced = test.drop(columns=["f_minutes_mean_5", "f_cameo_share_5"]).head(20)
    pred = fitted_lgb.predict(reduced)
    np.testing.assert_allclose(pred[["q0", "q1", "q2"]].sum(axis=1), 1.0, atol=1e-9)


def test_lgb_save_load_roundtrip(
    synthetic_split: tuple[pd.DataFrame, pd.DataFrame],
    fitted_lgb: LgbMinutesModel,
    tmp_path: Path,
) -> None:
    _, test = synthetic_split
    fitted_lgb.save(tmp_path / "minutes")
    loaded = LgbMinutesModel.load(tmp_path / "minutes")
    pd.testing.assert_frame_equal(fitted_lgb.predict(test), loaded.predict(test))


def test_lgb_determinism(synthetic_split: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    train, test = synthetic_split
    a = LgbMinutesModel(n_estimators=60).fit(train).predict(test.head(100))
    b = LgbMinutesModel(n_estimators=60).fit(train).predict(test.head(100))
    pd.testing.assert_frame_equal(a, b)


def test_lgb_requires_features() -> None:
    df = pd.DataFrame({"season": [2024], "gw": [1], "player_code": [1], "minutes": [90]})
    with pytest.raises(ValueError, match="f_"):
        LgbMinutesModel().fit(df)


# ---------------------------------------------------------------------------
# Fallback feature builder — leakage discipline
# ---------------------------------------------------------------------------


def test_feature_builder_is_strictly_lagged() -> None:
    pm = pd.DataFrame(
        {
            "season": [2024] * 4,
            "gw": [1, 2, 3, 4],
            "fpl_fixture_id": [10, 20, 30, 40],
            "player_code": [7] * 4,
            "position": ["MID"] * 4,
            "price": [50] * 4,
            "was_home": [True, False, True, False],
            "subs_regime": [5] * 4,
            "void_gw": [False] * 4,
            "minutes": [90, 0, 30, 90],
            "starts": [1, 0, 0, 1],
        }
    )
    feats = build_minutes_feature_frame(pm)
    # Row for GW g reflects only matches strictly before g.
    assert np.isnan(feats.loc[0, "f_start_share_1"])  # no history before GW1
    assert feats.loc[1, "f_start_share_1"] == 1.0  # GW1 start
    assert feats.loc[2, "f_start_share_1"] == 0.0  # GW2 no-show
    assert feats.loc[3, "f_cameo_share_3"] == pytest.approx(1 / 3)
    assert feats.loc[2, "f_missed_streak"] == 1.0  # missed GW2
    assert feats.loc[3, "f_missed_streak"] == 0.0  # played (cameo) in GW3
    assert feats.loc[3, "f_minutes_mean_3"] == pytest.approx(40.0)


def test_feature_builder_infers_starts_when_missing() -> None:
    pm = pd.DataFrame(
        {
            "season": [2018] * 3,
            "gw": [1, 2, 3],
            "fpl_fixture_id": [1, 2, 3],
            "player_code": [9] * 3,
            "position": ["FWD"] * 3,
            "price": [70] * 3,
            "subs_regime": [3] * 3,
            "void_gw": [False] * 3,
            "minutes": [85, 12, 90],
            "starts": pd.array([None, None, None], dtype="Int64"),
        }
    )
    feats = build_minutes_feature_frame(pm)
    assert feats.loc[1, "f_start_share_1"] == 1.0  # 85 min -> inferred start
    assert feats.loc[2, "f_start_share_1"] == 0.0  # 12 min cameo


# ---------------------------------------------------------------------------
# Real processed data (skipped when parquet tables are absent)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not PLAYER_MATCH.exists(), reason="processed data not built")
def test_lgb_on_real_data_beats_heuristic() -> None:
    pm = pd.read_parquet(PLAYER_MATCH)
    pm = pm[pm["season"] >= 2023]  # keep the test fast; full eval done offline
    feats = build_minutes_feature_frame(pm)
    train = feats[feats["season"] <= 2024]
    test = feats[feats["season"] == 2025]
    model = LgbMinutesModel(n_estimators=150).fit(train)
    lgb_eval = model.evaluate(test)
    heur_eval = HeuristicMinutesModel().evaluate(test)
    assert lgb_eval["bucket_log_loss"] < heur_eval["bucket_log_loss"]
    assert lgb_eval["brier_p0"] < heur_eval["brier_p0"]
    pred = model.predict(test)
    np.testing.assert_allclose(pred[["q0", "q1", "q2"]].sum(axis=1), 1.0, atol=1e-9)
    assert ((pred["mu1"] >= 1) & (pred["mu1"] <= 59)).all()
    assert ((pred["mu2"] >= 60) & (pred["mu2"] <= 90)).all()
