"""Offline tests for the distributional captaincy artifact (pipeline._captaincy_frame)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fplai.pipeline import _captaincy_frame

FIXTURE = Path(__file__).parent / "fixtures" / "predictions_2026_gw1.parquet"


@pytest.fixture(scope="module")
def pred() -> pd.DataFrame:
    return pd.read_parquet(FIXTURE)


@pytest.fixture(scope="module")
def gw_pred(pred: pd.DataFrame) -> pd.DataFrame:
    return pred.groupby(["season", "gw", "player_code"], as_index=False).agg(
        xp=("xp", "sum"), q0=("q0", "prod")
    )


def test_captaincy_frame_contract(pred: pd.DataFrame, gw_pred: pd.DataFrame) -> None:
    cap = _captaincy_frame(pred, gw_pred, n_draws=400, seed=3, top_k=6)
    assert cap is not None
    assert list(cap.columns) == [
        "season", "gw", "player_code", "xp", "mean_pts", "sd_pts",
        "p_haul", "p_blank", "p_best", "p_beats_top", "n_draws",
    ]
    assert len(cap) == 6
    # Sorted by xp desc; the top pick has no p_beats_top (it IS the reference).
    assert cap["xp"].is_monotonic_decreasing
    assert np.isnan(cap["p_beats_top"].iloc[0])
    assert cap["p_beats_top"].iloc[1:].between(0, 1).all()
    # P(best) is a proper distribution over the candidate set (ties split evenly).
    assert cap["p_best"].sum() == pytest.approx(1.0, abs=1e-9)
    assert ((cap["p_haul"] >= 0) & (cap["p_haul"] <= 1)).all()
    # The sampled mean tracks the analytic xP within Monte Carlo noise.
    assert (cap["mean_pts"] - cap["xp"]).abs().max() < 1.5


def test_captaincy_frame_deterministic(pred: pd.DataFrame, gw_pred: pd.DataFrame) -> None:
    a = _captaincy_frame(pred, gw_pred, n_draws=200, seed=11)
    b = _captaincy_frame(pred, gw_pred, n_draws=200, seed=11)
    assert a is not None and b is not None
    pd.testing.assert_frame_equal(a, b)


def test_captaincy_frame_needs_two_candidates(pred: pd.DataFrame) -> None:
    one = pred[pred["player_code"] == pred["player_code"].iloc[0]]
    gw_one = one.groupby(["season", "gw", "player_code"], as_index=False).agg(
        xp=("xp", "sum"), q0=("q0", "prod")
    )
    assert _captaincy_frame(one, gw_one, n_draws=50) is None
