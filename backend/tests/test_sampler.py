"""Offline tests for fplai.models.sampler — the stage-6 Monte Carlo points sampler.

Calibration is validated against a *recorded real slice* of the live 2026-27 GW1
predictions artifact (``tests/fixtures/predictions_2026_gw1.parquet``, 554
player-fixture rows over 10 fixtures): the mean of the sampled points per row
must converge to the analytic ``xp`` column that ``models/assemble.py`` produced
— this is the check that the sampler's inversion + joint sampling reproduces the
assembly maths. Coherence (shared scorelines) is validated on a synthetic
fixture built with the same maths so every parameter is known exactly.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fplai.models import sampler as sampler_mod
from fplai.models.sampler import (
    REQUIRED_COLUMNS,
    PointsSampler,
    chip_window_end,
    horizon_through_gw,
)

FIXTURE = Path(__file__).parent / "fixtures" / "predictions_2026_gw1.parquet"

SEASON = 2026

#: Explicit dispersion so tests never depend on the (gitignored) rates artifact.
DISP = {"GKP": 0.0, "DEF": 0.07, "MID": 0.05, "FWD": 0.07}


def poisson_floor_ev(lam: float, divisor: int, k_max: int = 80) -> float:
    """Independent reference: E[floor(X / divisor)], X ~ Poisson(lam)."""
    return sum(
        math.exp(-lam) * lam**k / math.factorial(k) * (k // divisor) for k in range(k_max + 1)
    )


# --------------------------------------------------------------------------------------
# Real-slice fixtures (shared draws: sampling 4000 rollouts once keeps the suite fast)
# --------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def real_pred() -> pd.DataFrame:
    return pd.read_parquet(FIXTURE)


@pytest.fixture(scope="module")
def real_draws(real_pred: pd.DataFrame) -> sampler_mod.SampleDraws:
    return PointsSampler(defcon_disp=DISP).sample_draws(real_pred, 4000, seed=7)


# --------------------------------------------------------------------------------------
# Synthetic single-fixture frame with exactly known generative parameters
# --------------------------------------------------------------------------------------

LAM_HOME = 1.7  # E[home goals]  (= away side's conceded mean)
LAM_AWAY = 1.1
P_CS_HOME = 0.40  # P(away scores 0)
P_CS_AWAY = 0.25
FWD_GOALS = 0.8
SAVES_H, SAVES_A = 2.8, 3.2


def _row(
    code: int,
    team: int,
    home: bool,
    pos: str,
    *,
    q: tuple[float, float, float] = (0.0, 0.0, 1.0),
    mu: tuple[float, float] = (25.0, 90.0),
    gw: int = 3,
    fixture: int = 900,
    opponent: int = 0,
    **xp: float,
) -> dict[str, float | int | bool | str]:
    base = {
        "season": SEASON,
        "gw": gw,
        "player_code": code,
        "fpl_fixture_id": fixture,
        "team_code": team,
        "opponent_code": opponent,
        "was_home": home,
        "position": pos,
        "q0": q[0],
        "q1": q[1],
        "q2": q[2],
        "mu1": mu[0],
        "mu2": mu[1],
    }
    comps = {
        c: 0.0
        for c in (
            "xp_appearance",
            "xp_goals",
            "xp_assists",
            "xp_cs",
            "xp_concede",
            "xp_saves",
            "xp_defcon",
            "xp_bonus",
            "xp_cards",
            "xp_other",
        )
    }
    comps.update(xp)
    p_play = q[1] + q[2]
    comps["xp_appearance"] = q[1] * 1.0 + q[2] * 2.0
    del p_play
    base.update(comps)
    base["xp"] = sum(comps.values())
    return base


def synth_frame() -> pd.DataFrame:
    """One fixture (home 10 v away 20) assembled with the analytic maths."""
    f2h = poisson_floor_ev(LAM_AWAY, 2)  # conceded floor EV for home GKP/DEF (90 min)
    f2a = poisson_floor_ev(LAM_HOME, 2)
    rows = [
        _row(1, 10, True, "GKP", xp_cs=4 * P_CS_HOME, xp_concede=-f2h,
             xp_saves=poisson_floor_ev(SAVES_H, 3)),
        _row(2, 10, True, "DEF", xp_cs=4 * P_CS_HOME, xp_concede=-f2h),
        _row(3, 10, True, "DEF", xp_cs=4 * P_CS_HOME, xp_concede=-f2h),
        _row(
            4, 10, True, "MID",
            q=(0.2, 0.3, 0.5), mu=(25.0, 85.0),
            xp_goals=5 * 0.30, xp_assists=3 * 0.20, xp_cs=1 * 0.5 * P_CS_HOME,
            xp_defcon=2 * 0.10, xp_cards=-0.15, xp_other=-0.004,
        ),
        _row(5, 10, True, "FWD", q=(1.0, 0.0, 0.0)),  # never plays
        _row(6, 20, False, "GKP", xp_cs=4 * P_CS_AWAY, xp_concede=-f2a,
             xp_saves=poisson_floor_ev(SAVES_A, 3)),
        _row(7, 20, False, "DEF", xp_cs=4 * P_CS_AWAY, xp_concede=-f2a),
        _row(8, 20, False, "FWD", xp_goals=4 * FWD_GOALS),
    ]
    return pd.DataFrame(rows)


def synth_sampler(**kwargs: object) -> PointsSampler:
    kwargs.setdefault("defcon_disp", DISP)
    kwargs.setdefault("include_bonus", False)
    return PointsSampler(**kwargs)  # type: ignore[arg-type]


# --------------------------------------------------------------------------------------
# Contract: shapes, dtypes, determinism, validation
# --------------------------------------------------------------------------------------


def test_shapes_dtypes_and_determinism(real_pred: pd.DataFrame) -> None:
    s = PointsSampler(defcon_disp=DISP)
    a = s.sample(real_pred, 50, seed=1)
    assert a.shape == (50, len(real_pred))
    assert a.dtype == np.int16
    d = s.sample_draws(real_pred, 50, seed=1)
    assert np.array_equal(d.points, a)  # same seed -> identical
    assert d.minutes.shape == a.shape and d.minutes.dtype == np.int16
    b = s.sample(real_pred, 50, seed=2)
    assert not np.array_equal(a, b)  # different seed -> different draws


def test_row_order_invariance() -> None:
    frame = synth_frame()
    s = synth_sampler()
    base = s.sample(frame, 200, seed=3)
    perm = np.random.default_rng(0).permutation(len(frame))
    shuffled = s.sample(frame.iloc[perm].reset_index(drop=True), 200, seed=3)
    assert np.array_equal(base[:, perm], shuffled)


def test_input_validation(real_pred: pd.DataFrame) -> None:
    s = PointsSampler(defcon_disp=DISP)
    with pytest.raises(ValueError, match="missing required columns"):
        s.sample(real_pred.drop(columns=["q0"]), 5, seed=1)
    with pytest.raises(ValueError, match="one season"):
        two = pd.concat([real_pred.head(5), real_pred.head(5).assign(season=2025)])
        s.sample(two, 5, seed=1)
    with pytest.raises(ValueError, match="empty"):
        s.sample(real_pred.head(0), 5, seed=1)
    with pytest.raises(ValueError, match="n must be"):
        s.sample(real_pred, 0, seed=1)
    assert set(REQUIRED_COLUMNS) <= set(real_pred.columns)


# --------------------------------------------------------------------------------------
# Calibration against the analytic xp (the assembly-maths validation)
# --------------------------------------------------------------------------------------


def test_calibration_total_within_2pct(
    real_pred: pd.DataFrame, real_draws: sampler_mod.SampleDraws
) -> None:
    mean = real_draws.points.mean(axis=0)
    total_xp = real_pred["xp"].sum()
    assert abs(mean.sum() - total_xp) / total_xp < 0.02


def test_calibration_per_row(
    real_pred: pd.DataFrame, real_draws: sampler_mod.SampleDraws
) -> None:
    mean = real_draws.points.mean(axis=0)
    xp = real_pred["xp"].to_numpy()
    assert np.corrcoef(mean, xp)[0, 1] > 0.99
    big = xp > 2.0
    dev = np.abs(mean - xp)[big]
    # Slack covers the documented bonus-realization deviation + MC noise.
    assert (dev <= np.maximum(0.30, 0.15 * xp[big])).all()


def test_calibration_no_bonus_component(real_pred: pd.DataFrame) -> None:
    """Without the bonus competition the sampler mean is essentially unbiased."""
    s = PointsSampler(defcon_disp=DISP, include_bonus=False)
    mean = s.sample(real_pred, 3000, seed=11).mean(axis=0)
    base = (real_pred["xp"] - real_pred["xp_bonus"]).to_numpy()
    assert abs(mean.sum() - base.sum()) / base.sum() < 0.01
    assert np.abs(mean - base).max() < 0.30  # per-row: MC noise scale at n=3000


def test_bonus_totals_conserved(real_pred: pd.DataFrame) -> None:
    """Sampled bonus per fixture stays near the 3+2+1 (+ties) analytic mass."""
    n = 2000
    with_b = PointsSampler(defcon_disp=DISP, chunk_draws=4096).sample(real_pred, n, seed=5)
    without = PointsSampler(
        defcon_disp=DISP, include_bonus=False, chunk_draws=4096
    ).sample(real_pred, n, seed=5)
    # Single chunk + same seed: the non-bonus rng stream is identical, so the
    # difference isolates the sampled bonus points exactly.
    bonus = (with_b - without).astype(np.int32)
    assert (bonus >= 0).all() and (bonus <= 3).all()
    total_sampled = bonus.mean(axis=0).sum()
    total_xp = real_pred["xp_bonus"].sum()
    assert abs(total_sampled - total_xp) / total_xp < 0.15
    per_fx = (
        pd.DataFrame({"fx": real_pred["fpl_fixture_id"], "b": bonus.mean(axis=0)})
        .groupby("fx")["b"]
        .sum()
    )
    assert ((per_fx > 5.9) & (per_fx < 7.6)).all()  # >= 6 by the rules, + tie inflation


# --------------------------------------------------------------------------------------
# Variance sanity: fat right tails for haulers, mass at zero for q0-heavy rows
# --------------------------------------------------------------------------------------


def test_hauler_right_tail(
    real_pred: pd.DataFrame, real_draws: sampler_mod.SampleDraws
) -> None:
    i = int(real_pred["xp_goals"].idxmax())
    pts = real_draws.points[:, i].astype(float)
    mean = pts.mean()
    assert (pts >= 10).mean() > 0.005  # double-digit hauls happen
    assert (pts >= 2 * mean).mean() > 0.005
    skew = ((pts - mean) ** 3).mean() / pts.std() ** 3
    assert skew > 0.3


def test_zero_heavy_rows_mass_at_zero(
    real_pred: pd.DataFrame, real_draws: sampler_mod.SampleDraws
) -> None:
    q0 = real_pred["q0"].to_numpy()
    rows = np.flatnonzero(q0 > 0.8)
    assert len(rows) > 10
    p_zero = (real_draws.points[:, rows] == 0).mean(axis=0)
    assert (p_zero >= 0.95 * q0[rows]).all()


def test_minutes_draws(real_pred: pd.DataFrame, real_draws: sampler_mod.SampleDraws) -> None:
    minutes = real_draws.minutes
    mu1 = np.rint(real_pred["mu1"].to_numpy())
    mu2 = np.rint(real_pred["mu2"].to_numpy())
    e_min = (real_pred["q1"] * real_pred["mu1"] + real_pred["q2"] * real_pred["mu2"]).to_numpy()
    for j in (int(np.argmax(e_min)), 0, len(real_pred) - 1):
        assert set(np.unique(minutes[:, j])) <= {0, mu1[j], mu2[j]}
    assert np.abs(minutes.mean(axis=0) - e_min).max() < 4.0  # MC noise at n=4000


# --------------------------------------------------------------------------------------
# Shared-scoreline coherence (synthetic fixture, all parameters known)
# --------------------------------------------------------------------------------------


def test_shared_scoreline_coherence() -> None:
    frame = synth_frame()
    d = synth_sampler().sample_draws(frame, 12000, seed=13)
    pts = d.points.astype(int)
    def_a, def_b = pts[:, 1], pts[:, 2]  # two home DEFs, always 90 minutes
    # A home DEF scores 6 = 2 (appearance) + 4 (CS) iff the away side scored 0;
    # the CS event is the shared scoreline draw, so it is identical for both.
    cs_a, cs_b = def_a == 6, def_b == 6
    assert np.array_equal(cs_a, cs_b)
    assert abs(cs_a.mean() - P_CS_HOME) < 0.04
    # Both defenders share the conceded draw too: points co-move strongly.
    assert np.corrcoef(def_a, def_b)[0, 1] > 0.9
    # The away FWD's goals are thinned from the SAME away-goals draw: he can
    # never have scored in a rollout where the home defence kept a clean sheet.
    fwd = pts[:, 7]
    goals_fwd = (fwd - 2) // 4
    assert (goals_fwd[cs_a] == 0).all()
    assert abs(goals_fwd.mean() - FWD_GOALS) < 0.05
    # Never-playing row: zero points, zero minutes, always.
    assert (pts[:, 4] == 0).all() and (d.minutes[:, 4] == 0).all()
    # Calibration on the synthetic frame (row means vs the assembled xp;
    # tolerance is >4 sigma of the MC noise at n=12000).
    dev = np.abs(pts.mean(axis=0) - frame["xp"].to_numpy())
    assert dev.max() < 0.15


def test_synthetic_goalkeeper_components() -> None:
    frame = synth_frame()
    pts = synth_sampler().sample(frame, 12000, seed=17).astype(int)
    gkp = pts[:, 0]
    # Mean matches the assembled xp; saves make the right tail (7+ = CS + 2 save pts).
    assert abs(gkp.mean() - frame.loc[0, "xp"]) < 0.1
    assert (gkp >= 7).mean() > 0.01


# --------------------------------------------------------------------------------------
# GW aggregation (DGWs)
# --------------------------------------------------------------------------------------


def dgw_frame() -> pd.DataFrame:
    """Two GW7 fixtures; player 77 (DEF, team 10) plays in both."""
    a = synth_frame().assign(gw=7, fpl_fixture_id=910)
    b = synth_frame().assign(gw=7, fpl_fixture_id=911)
    b["team_code"] = b["team_code"].map({10: 30, 20: 10})
    b["was_home"] = ~b["was_home"].astype(bool)
    b["player_code"] = b["player_code"] + 100
    # player 77 = home DEF (code 2) in fixture 910 and away DEF (code 107) in 911
    a.loc[a["player_code"] == 2, "player_code"] = 77
    b.loc[b["player_code"] == 107, "player_code"] = 77
    return pd.concat([a, b], ignore_index=True)


def test_sample_gw_aggregates_dgw() -> None:
    frame = dgw_frame()
    s = synth_sampler()
    gw = s.sample_gw(frame, 300, seed=19)
    draws = s.sample_draws(frame, 300, seed=19)  # same seed -> same fixture draws
    assert list(gw.index.columns) == ["season", "gw", "player_code", "n_fixtures"]
    assert gw.points.shape == (300, len(gw.index))
    dgw_pos = gw.index.index[gw.index["player_code"] == 77]
    assert len(dgw_pos) == 1
    j = int(dgw_pos[0])
    assert int(gw.index.loc[j, "n_fixtures"]) == 2
    rows = np.flatnonzero(frame["player_code"].to_numpy() == 77)
    manual = draws.points[:, rows].sum(axis=1)
    assert np.array_equal(gw.points[:, j], manual.astype(np.int16))
    manual_min = draws.minutes[:, rows].sum(axis=1)
    assert np.array_equal(gw.minutes[:, j], manual_min.astype(np.int16))
    singles = gw.index["player_code"] != 77
    assert (gw.index.loc[singles, "n_fixtures"] == 1).all()


# --------------------------------------------------------------------------------------
# Performance smoke + config knobs + horizon helpers
# --------------------------------------------------------------------------------------


def test_performance_smoke(real_pred: pd.DataFrame) -> None:
    """Scaled-down bar: the full 1000 x ~11k-row frame runs ~3 s / <1 GB on an
    M1 Pro (measured); here 500 draws x 554 rows must stay well under 10 s."""
    start = time.perf_counter()
    PointsSampler(defcon_disp=DISP).sample(real_pred, 500, seed=23)
    assert time.perf_counter() - start < 10.0


def test_defcon_disp_and_chunking_knobs(real_pred: pd.DataFrame) -> None:
    small_chunks = PointsSampler(defcon_disp=DISP, chunk_draws=64).sample(real_pred, 130, seed=29)
    one_chunk = PointsSampler(defcon_disp=DISP, chunk_draws=1024).sample(real_pred, 130, seed=29)
    assert small_chunks.shape == one_chunk.shape == (130, len(real_pred))
    # Different dispersion changes the DefCon draws deterministically.
    other = PointsSampler(defcon_disp={"GKP": 0.0, "DEF": 5.0, "MID": 5.0, "FWD": 5.0})
    assert other.defcon_disp["DEF"] == 5.0
    assert other.sample(real_pred, 30, seed=29).shape == (30, len(real_pred))
    with pytest.raises(ValueError, match="chunk_draws"):
        PointsSampler(chunk_draws=0)
    with pytest.raises(ValueError, match="red_card_ev_share"):
        PointsSampler(red_card_ev_share=1.5)


def test_bonus_variance_match_flag(real_pred: pd.DataFrame) -> None:
    """Event-driven mode redistributes star bonus downward vs the analytic xp."""
    star = int(real_pred["xp_bonus"].idxmax())
    matched = PointsSampler(defcon_disp=DISP).sample(real_pred, 800, seed=31)
    raw = PointsSampler(defcon_disp=DISP, bonus_variance_match=False).sample(
        real_pred, 800, seed=31
    )
    assert raw[:, star].mean() < matched[:, star].mean()


def test_horizon_helpers() -> None:
    assert chip_window_end(2026, 1) == 19
    assert chip_window_end(2026, 19) == 19
    assert chip_window_end(2026, 20) == 38
    assert chip_window_end(2026, 38) == 38
    assert horizon_through_gw(1, 19) == 19
    assert horizon_through_gw(5, 19) == 15
    assert horizon_through_gw(19, 19) == 1
    with pytest.raises(ValueError):
        horizon_through_gw(20, 19)
    with pytest.raises(ValueError):
        chip_window_end(2026, 0)
