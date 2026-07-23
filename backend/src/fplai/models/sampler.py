"""Vectorized Monte Carlo sampler of realized FPL points per prediction row (stage 6).

Implements the ARCHITECTURE "Season-simulation contracts" ``PointsSampler``: joint
draws of realized points for every player-fixture row of a predictions frame (the
``assemble_xp`` + predict-path artifact: ``season, gw, player_code, fpl_fixture_id,
team_code, was_home, position, q0, q1, q2, mu1, mu2`` plus the ``xp_*`` component
columns). The correlation structure is the point: within one sampled rollout a
fixture gets ONE scoreline draw shared by every player in it, so CS, conceded and
the team-goals context for attacking involvement cohere (BB/DEF-stack and TC
tails come out right).

Generative-parameter recovery ("inversion" design)
--------------------------------------------------
The predictions frame carries expected-points components, not the raw model rates,
so the sampler *inverts the assembly maths* (``models/assemble.py``) to recover the
generative parameters. Linear components invert exactly:

* ``e_goals = xp_goals / pts_goal(pos)``, ``e_assists = xp_assists / 3``.
* ``p_cs = xp_cs / (pts_cs(pos) * q2)`` for GKP/DEF/MID rows.
* Cards: ``|xp_cards| = e_yellow + 3 e_red`` cannot be split from one column; a
  documented constant EV share (``red_card_ev_share``, default 8% of the cards EV
  mass, the empirical red-card share) splits it. The split preserves the mean for
  any share value; only the tail shape (-3 vs -1) depends on it.
* ``e_own_goals = xp_other / -2``.

The *nonlinear* components (floor divisions, the DefCon threshold) are where
assemble's convention and a faithful per-rollout simulation genuinely differ:
assemble evaluates them once at the **expected** minutes (including the q0 mass),
while a sampled season realizes them only in rollouts where the player actually
plays — a large Jensen gap for rotation-risk players (e.g. a q0=0.7 defender
concedes at the full-match rate in the 30% of rollouts he plays; assemble's
Poisson at 0.3x the rate says almost nothing is conceded). To stay calibrated to
the analytic ``xp`` *by construction*, the sampler inverts these components
**through its own conditional law**: it solves per row (vectorized bisection, the
functions are monotone) for the exposure rate whose *bucket-mixture* expectation
matches the target —

* saves rate: ``q1 E[floor(Pois(r mu1)/3)] + q2 E[floor(Pois(r mu2)/3)] = xp_saves``,
* DefCon rate: ``q1 P(NB(r mu1) >= T) + q2 P(NB(r mu2) >= T) = xp_defcon / 2``
  with the *stored* per-position NB dispersion from the rates artifact (Poisson
  fallback exactly as assemble),
* concede thinning ``alpha``: the hurdle-thinned ``E[floor(C/2)]`` (closed form
  via the parity identity ``E[floor(C/2)] = (E[C] - P(C odd))/2`` and the pgf of
  the shared team-goals draw) matched to ``-xp_concede``.

The recovered rates therefore reproduce the analytic component means exactly
under the sampler's distribution; the corresponding realized *counts* are scaled
relative to a purist simulation (that is assemble's approximation, inherited
deliberately — the season-sim must be an unbiased estimator of the xP the
optimizer plans on). The scoreline means themselves are unaffected: the team-goal
mean recovers exactly from the concede floor-EV inversion because assemble's
``E[C_on] = lam_opp * e_min / 90`` identity is linear.

Fixture-side scoreline parameters are aggregated from the recovered row-level
values: for each fixture direction (home-goals / away-goals), the *defending*
side's GKP/DEF rows give the mean (``lam`` from the concede inversion) and its
GKP/DEF/MID rows give ``P(0 goals)`` (the CS probability, which the team model may
have odds-blended — so ``P(0)`` is honoured separately from the mean). Sides with
no usable rows fall back to ``-ln(p_cs)`` / ``exp(-lam)`` / a league-average prior.

Sampling scheme (one rollout)
-----------------------------
* Minutes bucket ~ Categorical(q0, q1, q2) independently per player-fixture row
  (simplification: bucket draws ignore cross-player rotation/lineup correlation
  within a team); minutes within a bucket are deterministic at the conditional
  means ``mu1``/``mu2`` (variance inside a bucket is not modelled — downstream
  autosubs only need the 0 / 1-59 / 60+ distinction).
* Team goals per fixture direction ~ hurdle distribution with ``P(0) = p_cs`` and
  mean ``lam``: 0 w.p. ``p_cs`` else ``1 + Poisson(lam/(1-p_cs) - 1)``. Both the
  zero mass (CS) and the mean (concede/attack) are matched exactly; higher moments
  differ slightly from the Poisson grid assemble integrates over (documented bias,
  see below). One draw per direction per rollout, shared by all rows.
* Attacking involvement is *conditioned on the sampled team total* via binomial
  thinning: ``goals_i ~ Binomial(G_team, e_goals_i / lam_team * m_i / e_min_i)``
  (assists likewise). Marginal means are exact; if the team scores 0 nobody
  scores; a haul requires a big team score. Bias: player totals need not sum to
  ``G_team`` (draws are independent given the total), and the thinning probability
  is clipped at 1 for extreme rows.
* Conceded-while-on ~ Binomial(G_opp, minutes/90) (thinning by sampled minutes);
  points ``-floor(C/2)`` for GKP/DEF. Saves ~ Poisson(rate x minutes/90), GKP.
* DefCon count ~ NB(mean = rate x minutes/90, size = stored dispersion) via the
  gamma-Poisson mixture, threshold per :data:`fplai.rules.DEFCON_THRESHOLDS`.
* Cards ~ Bernoulli given played; own goals ~ Poisson(rate x minutes/90).
* Bonus reuses the ``models/bonus.py`` machinery. The analytic ``xp_bonus`` ranks
  ``Normal(center, sigma)`` draws within the fixture (``center``/``sigma`` from
  :func:`fplai.models.bonus.expected_bps` + the
  :class:`~fplai.models.bonus.BonusCalibration`); to keep per-row bonus means
  aligned with that while still *coupling bonus to the sampled events*, the
  sampler ranks a **variance-matched** statistic
  ``center + kappa * dev + Normal(0, sigma_res)`` where ``dev`` is the realized
  BPS deviation of the sampled counts (season BPS-matrix coefficients), and
  ``kappa``/``sigma_res`` are chosen per row so the statistic's total variance
  equals the analytic ``sigma**2`` (``kappa <= 1``). Full event-driven ranking
  (``kappa = 1``, residual noise only) is available via
  ``bonus_variance_match=False`` — more realistic tails, but per-row bonus means
  then redistribute within fixtures relative to ``xp_bonus`` (a blanking star
  drops, role players rise). The official tie rules
  (:func:`fplai.models.bonus._tie_rule_bonus`) award 3/2/1 per draw among the
  players who played in that draw. The DefCon count deviation is excluded from
  the statistic (the stored NB dispersion produces unphysically wide count draws
  — it calibrates the threshold indicator, not the count) — bonus couples to
  scoreline/goals/assists/CS/conceded/saves/cards instead.

Documented biases vs the analytic ``xp``
----------------------------------------
Row means of sampled points converge to ``xp`` (validated <2% xp-weighted, and
per-row on the real GW1 slice, in ``tests/test_sampler.py``) up to:

1. *Hurdle vs Poisson team goals*: only ``P(0)`` and the mean of the scoreline
   draw are matched to the team model; higher moments follow the shifted-Poisson
   hurdle (the concede bisection absorbs this for the concede EV itself).
2. *Bonus realization*: per-draw participation (players sit out of the ranking
   when their sampled bucket is 0) and the non-Normal shape of the shrunk event
   deviation shift per-row bonus means by O(0.1). Fixture bonus totals are
   conserved at 3+2+1 + ties (~6.3-6.5 sampled), while the analytic
   ``expected_bonus`` inflates ties to ~7.0-7.7 per fixture on flat-q GWs
   (rounded tight normals tie often) — the sampler sits nearer the real game's
   ~6.1 average, and this tie-rate gap is the only measurable aggregate bias:
   -0.5% of total xp on the GW1 slice, -1.1% over GWs 1-19 (the no-bonus path
   is unbiased to 0.01%).
3. *Unreachable targets*: rows whose analytic component EV exceeds what the
   bucket-mixture law can produce at the bisection bound clamp there (not
   observed on real frames).

Determinism: fully reproducible for a given ``(frame, n, seed)``; row order of the
input does not matter (rows are canonically sorted internally and results are
returned in input order). Memory is bounded by processing draws in chunks.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

from fplai import config, rules
from fplai.models import bonus as bonus_model
from fplai.models.assemble import (
    COMPONENT_COLUMNS,
    DEFCON_STAT_SHARES,
    _nb_tail_prob,
    _poisson_floor_ev,
)
from fplai.models.bonus import _tie_rule_bonus

__all__ = [
    "REQUIRED_COLUMNS",
    "GwSample",
    "PointsSampler",
    "SampleDraws",
    "chip_window_end",
    "horizon_through_gw",
]

#: Input contract: identity/minutes columns + every xp component from assemble.
REQUIRED_COLUMNS: Final[tuple[str, ...]] = (
    "season",
    "gw",
    "player_code",
    "fpl_fixture_id",
    "team_code",
    "was_home",
    "position",
    "q0",
    "q1",
    "q2",
    "mu1",
    "mu2",
    *COMPONENT_COLUMNS,
)

#: Default RNG seed (matches the bonus module's convention).
DEFAULT_SEED: Final[int] = bonus_model.DEFAULT_SEED

#: Fallback NB dispersion when no rates artifact is on disk (mirrors
#: ``rates._DEFAULT_DEFCON_DISP``; GKP is DefCon-ineligible). Any positive value
#: is self-consistent (the tail inversion and the sampler share it), the stored
#: artifact value only improves the minutes-conditional shape.
_FALLBACK_DEFCON_DISP: Final[dict[str, float]] = {"GKP": 0.0, "DEF": 0.3, "MID": 0.3, "FWD": 0.3}

#: League-average expected goals per side, used only when a fixture side has no
#: recoverable concede/CS rows at all (should not happen on real predict frames).
_DEFAULT_TEAM_LAMBDA: Final[float] = 1.30

_MIN_MINUTES: Final[float] = 1e-6  # e_min floor below which a row never plays


def _stored_defcon_disp(models_dir: Path) -> dict[str, float] | None:
    """The per-position NB dispersion saved by ``RatesModel.save`` (None if absent)."""
    meta_path = Path(models_dir) / "rates" / "meta.json"
    if not meta_path.exists():
        return None
    try:
        raw = json.loads(meta_path.read_text())["defcon_disp"]
        return {str(k): float(v) for k, v in raw.items()}
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _grid_inverse(ys: np.ndarray, xs: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Invert a monotone-increasing tabulated function: find x with f(x) = target.

    ``ys = f(xs)`` must be non-decreasing; targets outside the range clamp to the
    grid ends (``np.interp`` semantics).
    """
    return np.interp(targets, ys, xs)


def _solve_monotone(
    fn: Callable[[np.ndarray], np.ndarray],
    targets: np.ndarray,
    lo: float,
    hi: float,
    iters: int = 40,
) -> np.ndarray:
    """Vectorized bisection: per-element x in [lo, hi] with monotone fn(x) = target.

    Unreachable targets converge to the nearer bound (callers document the clamp).
    """
    lo_arr = np.full(targets.shape, float(lo))
    hi_arr = np.full(targets.shape, float(hi))
    for _ in range(iters):
        mid = 0.5 * (lo_arr + hi_arr)
        below = fn(mid) < targets
        lo_arr = np.where(below, mid, lo_arr)
        hi_arr = np.where(below, hi_arr, mid)
    return 0.5 * (lo_arr + hi_arr)


# --------------------------------------------------------------------------------------
# Prepared, sorted view of a predictions frame
# --------------------------------------------------------------------------------------


@dataclass
class _Prepared:
    """Row-level generative parameters in canonical sorted order (internal)."""

    n_rows: int
    order: np.ndarray  # original index of sorted row j (out[:, order] = sorted result)
    keys: pd.DataFrame  # season, gw, player_code, fpl_fixture_id in sorted order
    # minutes machinery
    q0: np.ndarray
    q01: np.ndarray  # q0 + q1
    q1: np.ndarray
    q2: np.ndarray
    mu1: np.ndarray
    mu2: np.ndarray
    e_min: np.ndarray
    # scoring values per row
    pts_short: np.ndarray
    pts_long: np.ndarray
    pts_goal: np.ndarray
    pts_assist: np.ndarray
    pts_cs: np.ndarray
    pts_concede: np.ndarray  # -1 for GKP/DEF else 0 (per 2 conceded)
    pts_save: np.ndarray  # 1 for GKP else 0 (per 3 saves)
    pts_defcon: np.ndarray
    pts_yellow: np.ndarray
    pts_red: np.ndarray
    pts_og: np.ndarray
    pts_bonus: np.ndarray
    # recovered expectations (match-level, minutes-integrated)
    e_goals: np.ndarray
    e_assists: np.ndarray
    e_conc_on: np.ndarray
    e_saves_on: np.ndarray
    e_defcon_on: np.ndarray
    e_cs: np.ndarray  # q2 * p_cs (bonus profile)
    e_yellow: np.ndarray
    e_red: np.ndarray
    e_og: np.ndarray
    # per-draw sampling rates (multiply by sampled minutes / thin the shared draw)
    goal_share_pm: np.ndarray  # thinning probability per sampled minute
    assist_share_pm: np.ndarray
    conc_alpha: np.ndarray  # conceded thinning: p = alpha * minutes / 90
    rate_saves_pm: np.ndarray  # solved per-minute rates (bucket-mixture calibrated)
    rate_defcon_pm: np.ndarray
    rate_og_pm: np.ndarray
    e_conc_dev: np.ndarray  # sampler-consistent count means (BPS deviation centers)
    e_saves_dev: np.ndarray
    p_yellow: np.ndarray  # per-match given played
    p_red: np.ndarray
    defcon_r: np.ndarray  # NB size per row (<=0 or non-finite -> Poisson)
    defcon_nb: np.ndarray  # bool: NB path
    defcon_threshold: np.ndarray
    # fixture bookkeeping
    n_dir: int
    dir_own: np.ndarray  # per row: index of own team's goals direction
    dir_opp: np.ndarray
    dir_p0: np.ndarray  # per direction: P(0 goals)
    dir_lam2: np.ndarray  # per direction: Poisson rate of the (count - 1 | count > 0) part
    fixture_slices: list[slice]
    # bonus machinery
    center: np.ndarray
    sigma_res: np.ndarray  # residual Normal scale after variance matching
    kappa: np.ndarray  # shrink factor on the realized-event BPS deviation
    bps_minute1: float
    bps_minute2: float
    bps_goal: np.ndarray
    bps_assist: float
    bps_cs: np.ndarray
    bps_conc: np.ndarray
    bps_save: np.ndarray
    bps_yellow: float
    bps_red: float
    bps_og: float


@dataclass(frozen=True)
class SampleDraws:
    """Per-fixture-row draws: ``points`` and ``minutes`` are int16 ``[n, rows]``
    aligned to the input frame's row order."""

    points: np.ndarray
    minutes: np.ndarray


@dataclass(frozen=True)
class GwSample:
    """Per player-GW aggregation of a fixture-level sample.

    ``index`` has one row per ``(season, gw, player_code)`` (plus ``n_fixtures``);
    ``points``/``minutes`` are int16 ``[n, len(index)]`` sums over the player's
    fixtures in that GW (DGWs sum both legs; minutes sum accordingly).
    """

    index: pd.DataFrame
    points: np.ndarray
    minutes: np.ndarray


class PointsSampler:
    """Vectorized sampler of realized FPL points from a predictions frame.

    Parameters
    ----------
    defcon_disp:
        Per-position NB dispersion (the value assemble fed to ``_nb_tail_prob``).
        Default: the stored ``rates/meta.json`` dispersion under ``models_dir``
        (:data:`fplai.config.MODELS_DIR`), falling back to 0.30 per outfield
        position when no artifact exists.
    include_bonus:
        Disable to skip the within-fixture bonus competition (synthetic-frame
        tests, or callers that model bonus separately).
    bonus_variance_match:
        Match the ranking statistic's variance to the analytic bonus machinery
        (default; keeps per-row bonus means aligned with ``xp_bonus``). ``False``
        ranks fully event-driven BPS — realistic tails, redistributed means.
    red_card_ev_share:
        Share of the ``|xp_cards|`` EV mass attributed to red cards (mean is
        preserved for any value; only the -3 tail frequency changes).
    bonus_calibration:
        Override the :class:`~fplai.models.bonus.BonusCalibration`; default is
        the module default assemble used.
    chunk_draws:
        Draw-axis chunk size bounding peak memory (~15 float64 arrays of
        ``chunk_draws x rows``).
    """

    def __init__(
        self,
        *,
        defcon_disp: Mapping[str, float] | None = None,
        include_bonus: bool = True,
        bonus_variance_match: bool = True,
        red_card_ev_share: float = 0.08,
        bonus_calibration: bonus_model.BonusCalibration | None = None,
        chunk_draws: int = 256,
        models_dir: Path | None = None,
    ) -> None:
        if not 0.0 <= red_card_ev_share < 1.0:
            raise ValueError("red_card_ev_share must be in [0, 1)")
        if chunk_draws < 1:
            raise ValueError("chunk_draws must be >= 1")
        root = Path(models_dir) if models_dir is not None else config.MODELS_DIR
        disp = dict(defcon_disp) if defcon_disp is not None else _stored_defcon_disp(root)
        self.defcon_disp: dict[str, float] = disp if disp is not None else dict(
            _FALLBACK_DEFCON_DISP
        )
        self.include_bonus = include_bonus
        self.bonus_variance_match = bonus_variance_match
        self.red_card_ev_share = red_card_ev_share
        self.bonus_calibration = (
            bonus_calibration if bonus_calibration is not None else bonus_model.DEFAULT_CALIBRATION
        )
        self.chunk_draws = chunk_draws

    # ------------------------------------------------------------------ public API

    def sample(self, predictions: pd.DataFrame, n: int, seed: int = DEFAULT_SEED) -> np.ndarray:
        """Draw ``n`` joint rollouts of realized points; int16 array ``[n, rows]``.

        Column ``j`` corresponds to ``predictions`` row ``j`` (positional).
        Deterministic for a given ``(predictions, n, seed)`` regardless of the
        frame's row order.
        """
        return self.sample_draws(predictions, n, seed).points

    def sample_draws(
        self, predictions: pd.DataFrame, n: int, seed: int = DEFAULT_SEED
    ) -> SampleDraws:
        """Like :meth:`sample` but also returns sampled minutes (autosub input).

        Minutes take values ``{0, round(mu1), round(mu2)}`` per row — the bucket
        draw is the modelled randomness; within-bucket minutes are the
        conditional means (documented simplification).
        """
        if n < 1:
            raise ValueError("n must be >= 1")
        prep = self._prepare(predictions)
        points_sorted = np.empty((n, prep.n_rows), dtype=np.int16)
        minutes_sorted = np.empty((n, prep.n_rows), dtype=np.int16)
        rng = np.random.default_rng(seed)
        for lo in range(0, n, self.chunk_draws):
            hi = min(lo + self.chunk_draws, n)
            self._sample_chunk(prep, rng, points_sorted[lo:hi], minutes_sorted[lo:hi])
        points = np.empty_like(points_sorted)
        minutes = np.empty_like(minutes_sorted)
        points[:, prep.order] = points_sorted
        minutes[:, prep.order] = minutes_sorted
        return SampleDraws(points=points, minutes=minutes)

    def sample_gw(
        self, predictions: pd.DataFrame, n: int, seed: int = DEFAULT_SEED
    ) -> GwSample:
        """Sample per fixture, then sum each player's fixtures within a GW.

        The per-rollout sums preserve the within-GW correlation (a DGW player's
        two legs share nothing but their own draws; every fixture's scoreline is
        shared by its players). Returns a :class:`GwSample`.
        """
        draws = self.sample_draws(predictions, n, seed)
        keys = predictions[["season", "gw", "player_code"]].reset_index(drop=True)
        order = np.lexsort(
            (
                keys["player_code"].to_numpy(),
                keys["gw"].to_numpy(),
                keys["season"].to_numpy(),
            )
        )
        sorted_keys = keys.iloc[order].reset_index(drop=True)
        boundary = np.ones(len(sorted_keys), dtype=bool)
        if len(sorted_keys) > 1:
            same = (
                sorted_keys[["season", "gw", "player_code"]].iloc[1:].to_numpy()
                == sorted_keys[["season", "gw", "player_code"]].iloc[:-1].to_numpy()
            ).all(axis=1)
            boundary[1:] = ~same
        starts = np.flatnonzero(boundary)
        index = sorted_keys.iloc[starts].reset_index(drop=True)
        counts = np.diff(np.append(starts, len(sorted_keys)))
        index["n_fixtures"] = counts
        points = np.add.reduceat(
            draws.points[:, order].astype(np.int32), starts, axis=1
        ).astype(np.int16)
        minutes = np.add.reduceat(
            draws.minutes[:, order].astype(np.int32), starts, axis=1
        ).astype(np.int16)
        return GwSample(index=index, points=points, minutes=minutes)

    # ------------------------------------------------------------------ preparation

    def _prepare(self, predictions: pd.DataFrame) -> _Prepared:
        missing = [c for c in REQUIRED_COLUMNS if c not in predictions.columns]
        if missing:
            raise ValueError(f"predictions frame is missing required columns: {missing}")
        if predictions.empty:
            raise ValueError("predictions frame is empty")
        seasons = predictions["season"].unique()
        if len(seasons) != 1:
            raise ValueError(f"predictions frame must cover one season, got {sorted(seasons)}")
        season = int(seasons[0])
        scoring = rules.get_scoring(season)

        order = np.lexsort(
            (
                predictions["player_code"].to_numpy(),
                predictions["fpl_fixture_id"].to_numpy(),
                predictions["gw"].to_numpy(),
            )
        )
        d = predictions.iloc[order].reset_index(drop=True)
        n_rows = len(d)
        pos = d["position"].astype(str)
        unknown = set(pos.unique()) - set(rules.POSITIONS)
        if unknown:
            raise ValueError(f"unknown positions in predictions: {sorted(unknown)}")

        def fcol(name: str) -> np.ndarray:
            return np.nan_to_num(d[name].to_numpy(dtype=float), nan=0.0)

        def pts(action: str) -> np.ndarray:
            return pos.map(scoring[action]).to_numpy(dtype=float)

        q0, q1, q2 = fcol("q0"), fcol("q1"), fcol("q2")
        mu1, mu2 = fcol("mu1"), fcol("mu2")
        e_min = q1 * mu1 + q2 * mu2
        m_share = e_min / 90.0
        p_play = q1 + q2
        playable = e_min > _MIN_MINUTES

        pts_short, pts_long = pts("short_play"), pts("long_play")
        pts_goal, pts_assist = pts("goals_scored"), pts("assists")
        pts_cs, pts_concede = pts("clean_sheets"), pts("goals_conceded")
        pts_save = pts("saves")
        pts_yellow, pts_red = pts("yellow_cards"), pts("red_cards")
        pts_og, pts_bonus = pts("own_goals"), pts("bonus")
        if "defensive_contribution" in scoring:
            pts_defcon = pts("defensive_contribution")
        else:
            pts_defcon = np.zeros(n_rows)

        # --- invert the assembly maths back to generative parameters -------------
        e_goals = fcol("xp_goals") / pts_goal
        e_assists = fcol("xp_assists") / pts_assist

        e_floor_c = np.where(pts_concede != 0, fcol("xp_concede") / np.where(
            pts_concede != 0, pts_concede, 1.0
        ), 0.0)
        grid_c = np.linspace(0.0, 15.0, 3001)
        e_conc_on = _grid_inverse(_poisson_floor_ev(grid_c, 2, grid_max=15), grid_c, e_floor_c)
        e_conc_on = np.where(pts_concede != 0, e_conc_on, 0.0)

        e_floor_s = np.where(pts_save > 0, fcol("xp_saves") / np.where(
            pts_save > 0, pts_save, 1.0
        ), 0.0)
        grid_s = np.linspace(0.0, 20.0, 4001)
        e_saves_on = _grid_inverse(_poisson_floor_ev(grid_s, 3, grid_max=30), grid_s, e_floor_s)
        e_saves_on = np.where(pts_save > 0, e_saves_on, 0.0)

        cs_ok = (pts_cs > 0) & (q2 > 1e-9)
        p_cs_row = np.where(
            cs_ok,
            fcol("xp_cs") / np.where(cs_ok, pts_cs * np.clip(q2, 1e-9, None), 1.0),
            np.nan,
        )
        p_cs_row = np.clip(p_cs_row, 0.0, 0.98)

        defcon_r = pos.map(
            lambda p: self.defcon_disp.get(p, _FALLBACK_DEFCON_DISP.get(p, 0.3))
        ).to_numpy(dtype=float)
        defcon_threshold = pos.map(rules.DEFCON_THRESHOLDS).to_numpy(dtype=float)
        defcon_threshold = np.nan_to_num(defcon_threshold, nan=1e9)  # GKP: never
        p_defcon = np.where(pts_defcon > 0, fcol("xp_defcon") / np.where(
            pts_defcon > 0, pts_defcon, 1.0
        ), 0.0)
        e_defcon_on = np.zeros(n_rows)
        eligible = (p_defcon > 0) & (defcon_threshold < 1e6)
        if eligible.any():
            # Invert per (position) group through the exact assemble tail function.
            grid_mu = np.concatenate(
                [np.linspace(0.0, 20.0, 2001), np.linspace(20.02, 300.0, 1500)]
            )
            for p in np.unique(pos.to_numpy()[eligible]):
                mask = eligible & (pos == p).to_numpy()
                r = float(self.defcon_disp.get(p, _FALLBACK_DEFCON_DISP.get(p, 0.3)))
                thr = float(rules.DEFCON_THRESHOLDS[p])
                ys = _nb_tail_prob(
                    grid_mu, np.full_like(grid_mu, r), np.full_like(grid_mu, thr)
                )
                e_defcon_on[mask] = _grid_inverse(ys, grid_mu, p_defcon[mask])
        defcon_nb = np.isfinite(defcon_r) & (defcon_r > 0) & (defcon_threshold < 1e6)

        cards_ev = np.clip(-fcol("xp_cards"), 0.0, None)  # e_yellow + 3 * e_red
        e_red = self.red_card_ev_share * cards_ev / 3.0
        e_yellow = (1.0 - self.red_card_ev_share) * cards_ev
        with np.errstate(divide="ignore", invalid="ignore"):
            p_yellow = np.clip(np.where(p_play > 1e-9, e_yellow / p_play, 0.0), 0.0, 1.0)
            p_red = np.clip(np.where(p_play > 1e-9, e_red / p_play, 0.0), 0.0, 1.0)
        e_og = np.clip(fcol("xp_other") / -2.0, 0.0, None)

        # --- fixture directions: shared scoreline parameters ---------------------
        was_home = d["was_home"].to_numpy(dtype=bool)
        fx_key = d["gw"].to_numpy(dtype=np.int64) * 10**9 + d["fpl_fixture_id"].to_numpy(
            dtype=np.int64
        )
        uniq, fx_idx = np.unique(fx_key, return_inverse=True)
        n_fx = len(uniq)
        n_dir = 2 * n_fx
        # Direction 2f = home goals, 2f+1 = away goals. A row's own attack is its
        # side's direction; its conceded goals come from the opposite direction.
        dir_own = 2 * fx_idx + (~was_home).astype(np.int64)
        dir_opp = 2 * fx_idx + was_home.astype(np.int64)

        with np.errstate(divide="ignore", invalid="ignore"):
            lam_opp_row = np.where(m_share > 1e-9, e_conc_on / m_share, np.nan)
        lam_usable = (pts_concede != 0) & (e_min > 5.0) & np.isfinite(lam_opp_row)
        lam_dir = _weighted_dir_mean(dir_opp, lam_opp_row, e_min, lam_usable, n_dir)
        cs_usable = np.isfinite(p_cs_row) & (e_min > 5.0)
        p0_dir = _weighted_dir_mean(dir_opp, p_cs_row, e_min, cs_usable, n_dir)

        lam_nan, p0_nan = ~np.isfinite(lam_dir), ~np.isfinite(p0_dir)
        lam_dir[lam_nan & ~p0_nan] = -np.log(np.clip(p0_dir[lam_nan & ~p0_nan], 1e-3, None))
        lam_dir[lam_nan & p0_nan] = _DEFAULT_TEAM_LAMBDA
        p0_dir[p0_nan] = np.exp(-lam_dir[p0_nan])
        lam_dir = np.clip(lam_dir, 0.05, 6.0)
        p0_dir = np.clip(p0_dir, 0.005, 0.98)
        # Hurdle: 0 w.p. p0 else 1 + Poisson(lam2); lam2 matches the mean exactly
        # when lam >= 1 - p0 (else the positive part floors at 1: tiny upward bias).
        dir_lam2 = np.clip(lam_dir / (1.0 - p0_dir) - 1.0, 0.0, None)
        dir_mean = (1.0 - p0_dir) * (1.0 + dir_lam2)  # exact hurdle mean (clip-aware)

        team_mean_row = dir_mean[dir_own]
        with np.errstate(divide="ignore", invalid="ignore"):
            goal_share_pm = np.where(
                playable, e_goals / (team_mean_row * np.clip(e_min, _MIN_MINUTES, None)), 0.0
            )
            assist_share_pm = np.where(
                playable, e_assists / (team_mean_row * np.clip(e_min, _MIN_MINUTES, None)), 0.0
            )
            per_min = np.where(playable, 1.0 / np.clip(e_min, _MIN_MINUTES, None), 0.0)
        rate_og_pm = e_og * per_min

        # --- solve sampling exposures through the bucket-mixture law -------------
        # (see module docstring: matches each nonlinear component's mean under the
        # sampler's own conditional distribution to the analytic assemble target)
        rate_saves_pm = np.zeros(n_rows)
        save_mask = (pts_save > 0) & (e_floor_s > 1e-12) & playable
        if save_mask.any():
            m1s, m2s = mu1[save_mask], mu2[save_mask]
            q1s, q2s = q1[save_mask], q2[save_mask]

            def ev_saves(rate: np.ndarray) -> np.ndarray:
                return q1s * _poisson_floor_ev(rate * m1s, 3, grid_max=40) + q2s * (
                    _poisson_floor_ev(rate * m2s, 3, grid_max=40)
                )

            rate_saves_pm[save_mask] = _solve_monotone(ev_saves, e_floor_s[save_mask], 0.0, 0.4)

        rate_defcon_pm = np.zeros(n_rows)
        dc_mask = eligible & playable
        if dc_mask.any():
            m1d, m2d = mu1[dc_mask], mu2[dc_mask]
            q1d, q2d = q1[dc_mask], q2[dc_mask]
            rd, td = defcon_r[dc_mask], defcon_threshold[dc_mask]
            rd = np.where(defcon_nb[dc_mask], rd, np.nan)  # Poisson fallback path

            def ev_defcon(rate: np.ndarray) -> np.ndarray:
                return q1d * _nb_tail_prob(rate * m1d, rd, td) + q2d * (
                    _nb_tail_prob(rate * m2d, rd, td)
                )

            rate_defcon_pm[dc_mask] = _solve_monotone(ev_defcon, p_defcon[dc_mask], 0.0, 5.0)

        conc_alpha = np.zeros(n_rows)
        conc_mask = (pts_concede != 0) & (e_floor_c > 1e-12) & playable
        if conc_mask.any():
            m1c, m2c = mu1[conc_mask], mu2[conc_mask]
            q1c, q2c = q1[conc_mask], q2[conc_mask]
            p0c = p0_dir[dir_opp][conc_mask]
            lam2c = dir_lam2[dir_opp][conc_mask]
            meanc = dir_mean[dir_opp][conc_mask]

            def ev_conc(alpha: np.ndarray) -> np.ndarray:
                # E[floor(C/2)] = (E[C] - P(C odd)) / 2 with C = Binomial(G, p)
                # thinning of the shared hurdle draw G; P(C odd) via the pgf.
                ev = np.zeros_like(alpha)
                for mb, qb in ((m1c, q1c), (m2c, q2c)):
                    p = np.clip(alpha * mb / 90.0, 0.0, 1.0)
                    t = 1.0 - 2.0 * p
                    pgf = p0c + (1.0 - p0c) * t * np.exp(lam2c * (t - 1.0))
                    ev = ev + qb * 0.5 * (p * meanc - 0.5 * (1.0 - pgf))
                return ev

            conc_alpha[conc_mask] = _solve_monotone(ev_conc, e_floor_c[conc_mask], 0.0, 8.0)

        # Sampler-consistent count means (BPS deviation centers).
        p1c = np.clip(conc_alpha * mu1 / 90.0, 0.0, 1.0)
        p2c = np.clip(conc_alpha * mu2 / 90.0, 0.0, 1.0)
        e_conc_dev = (q1 * p1c + q2 * p2c) * dir_mean[dir_opp]
        e_saves_dev = rate_saves_pm * e_min

        # Fixture group slices (rows are sorted by (gw, fixture, player)).
        starts = np.flatnonzero(np.r_[True, fx_idx[1:] != fx_idx[:-1]])
        ends = np.r_[starts[1:], n_rows]
        fixture_slices = [slice(int(a), int(b)) for a, b in zip(starts, ends, strict=True)]

        # --- bonus: expected BPS center/sigma + per-count coefficients -----------
        e_cs = np.nan_to_num(p_cs_row, nan=0.0) * q2
        # DEF CS-eligibility for BPS covers GKP too; e_cs is 0 where pts_cs == 0
        # (FWD), matching assemble's profile (it passed q2 * p_cs for all rows but
        # expected_bps zeroes non-GKP/DEF CS BPS; MID e_cs only feeds FPL points).
        shares = pos.map(DEFCON_STAT_SHARES)
        share_cbi = shares.map(lambda s: s[0]).to_numpy(dtype=float)
        share_tackles = shares.map(lambda s: s[1]).to_numpy(dtype=float)
        share_rec = shares.map(lambda s: s[2]).to_numpy(dtype=float)
        profile = pd.DataFrame(
            {
                "season": d["season"],
                "gw": d["gw"],
                "player_code": d["player_code"],
                "fpl_fixture_id": d["fpl_fixture_id"],
                "position": pos,
                "q1": q1,
                "q2": q2,
                "e_goals": e_goals,
                "e_assists": e_assists,
                "e_cs": e_cs,
                "e_goals_conceded": e_conc_on,
                "e_saves": e_saves_on,
                "e_cbi": e_defcon_on * share_cbi,
                "e_tackles": e_defcon_on * share_tackles,
                "e_recoveries": e_defcon_on * share_rec,
                "e_yellow": e_yellow,
                "e_red": e_red,
                "e_own_goals": e_og,
            }
        )
        ebps = bonus_model.expected_bps(profile, season).to_numpy()
        cal = self.bonus_calibration
        center = cal.center(ebps, pos.to_numpy())
        sigma = cal.sigma(center)

        m = bonus_model.bps_matrix(season)
        is_gk = (pos == "GKP").to_numpy()
        is_gkdef = is_gk | (pos == "DEF").to_numpy()
        bps_goal = np.select(
            [pos == "MID", pos == "FWD"], [m["goal_mid"], m["goal_fwd"]], default=m["goal_gkp_def"]
        )
        if "gk_save_inside_box_extra" in m:  # v4 (mirrors bonus.expected_bps)
            save_val = (
                m["gk_save"]
                + bonus_model.IN_BOX_SAVE_SHARE * m["gk_save_inside_box_extra"]
                + bonus_model.BIG_CHANCE_SAVE_SHARE * m["gk_big_chance_save_extra"]
            )
        elif "gk_save_inside_box" in m:  # v3
            save_val = (
                bonus_model.IN_BOX_SAVE_SHARE * m["gk_save_inside_box"]
                + (1.0 - bonus_model.IN_BOX_SAVE_SHARE) * m["gk_save_outside_box"]
            )
        else:  # v1/v2
            save_val = m["gk_save"]
        bps_minute1 = float(m["minutes_1_60"])
        bps_minute2 = float(m["minutes_over_60"])
        bps_cs_arr = np.where(is_gkdef, float(m["clean_sheet_gkp_def"]), 0.0)
        bps_conc_arr = np.where(is_gkdef, float(m.get("goal_conceded_gkp_def", 0.0)), 0.0)
        bps_save_arr = np.where(is_gk, float(save_val), 0.0)
        bps_goal_arr = bps_goal.astype(float)
        bps_assist = float(m["assist"])
        bps_yellow = float(m["yellow_card"])
        bps_red = float(m["red_card"])
        bps_og = float(m["own_goal"])

        # Variance of the realized-event BPS deviation (independence approx; the
        # DefCon count is excluded — see module docstring).
        v_event = (
            bps_minute1**2 * q1 * (1.0 - q1)
            + bps_minute2**2 * q2 * (1.0 - q2)
            - 2.0 * bps_minute1 * bps_minute2 * q1 * q2
            + bps_goal_arr**2 * e_goals
            + bps_assist**2 * e_assists
            + bps_cs_arr**2 * e_cs * (1.0 - e_cs)
            + bps_conc_arr**2 * e_conc_dev
            + bps_save_arr**2 * e_saves_dev
            + bps_yellow**2 * e_yellow
            + bps_red**2 * e_red
            + bps_og**2 * e_og
        )
        if self.bonus_variance_match:
            floor2 = float(cal.sigma_floor) ** 2
            excess = np.clip(sigma**2 - floor2, 0.0, None)
            kappa = np.clip(np.sqrt(excess / np.clip(v_event, 1e-9, None)), 0.0, 1.0)
            sigma_res = np.sqrt(np.clip(sigma**2 - kappa**2 * v_event, floor2, None))
        else:
            kappa = np.ones(n_rows)
            sigma_res = sigma.copy()

        return _Prepared(
            n_rows=n_rows,
            order=order,
            keys=d[["season", "gw", "player_code", "fpl_fixture_id"]],
            q0=q0,
            q01=q0 + q1,
            q1=q1,
            q2=q2,
            mu1=mu1,
            mu2=mu2,
            e_min=e_min,
            pts_short=pts_short.astype(np.int16),
            pts_long=pts_long.astype(np.int16),
            pts_goal=pts_goal.astype(np.int16),
            pts_assist=pts_assist.astype(np.int16),
            pts_cs=pts_cs.astype(np.int16),
            pts_concede=pts_concede.astype(np.int16),
            pts_save=pts_save.astype(np.int16),
            pts_defcon=pts_defcon.astype(np.int16),
            pts_yellow=pts_yellow.astype(np.int16),
            pts_red=pts_red.astype(np.int16),
            pts_og=pts_og.astype(np.int16),
            pts_bonus=pts_bonus.astype(np.int16),
            e_goals=e_goals,
            e_assists=e_assists,
            e_conc_on=e_conc_on,
            e_saves_on=e_saves_on,
            e_defcon_on=e_defcon_on,
            e_cs=e_cs,
            e_yellow=e_yellow,
            e_red=e_red,
            e_og=e_og,
            goal_share_pm=goal_share_pm,
            assist_share_pm=assist_share_pm,
            conc_alpha=conc_alpha,
            rate_saves_pm=rate_saves_pm,
            rate_defcon_pm=rate_defcon_pm,
            rate_og_pm=rate_og_pm,
            e_conc_dev=e_conc_dev,
            e_saves_dev=e_saves_dev,
            p_yellow=p_yellow,
            p_red=p_red,
            defcon_r=defcon_r,
            defcon_nb=defcon_nb,
            defcon_threshold=defcon_threshold,
            n_dir=n_dir,
            dir_own=dir_own,
            dir_opp=dir_opp,
            dir_p0=p0_dir,
            dir_lam2=dir_lam2,
            fixture_slices=fixture_slices,
            center=center,
            sigma_res=sigma_res,
            kappa=kappa,
            bps_minute1=bps_minute1,
            bps_minute2=bps_minute2,
            bps_goal=bps_goal_arr,
            bps_assist=bps_assist,
            bps_cs=bps_cs_arr,
            bps_conc=bps_conc_arr,
            bps_save=bps_save_arr,
            bps_yellow=bps_yellow,
            bps_red=bps_red,
            bps_og=bps_og,
        )

    # ------------------------------------------------------------------ chunk kernel

    def _sample_chunk(
        self,
        prep: _Prepared,
        rng: np.random.Generator,
        pts_out: np.ndarray,
        minutes_out: np.ndarray,
    ) -> None:
        """Fill one chunk of draws (views into the sorted output arrays)."""
        c = pts_out.shape[0]
        shape = (c, prep.n_rows)

        # Minutes bucket ~ Categorical(q0, q1, q2); minutes = conditional mean.
        u = rng.random(shape)
        bucket = (u >= prep.q0).astype(np.int8) + (u >= prep.q01).astype(np.int8)
        played = bucket > 0
        minutes = np.where(bucket == 2, prep.mu2, np.where(bucket == 1, prep.mu1, 0.0))

        # One scoreline draw per fixture direction, shared by all its rows.
        z = rng.random((c, prep.n_dir))
        extra = rng.poisson(prep.dir_lam2, (c, prep.n_dir))
        goals_dir = np.where(z < prep.dir_p0, 0, 1 + extra).astype(np.int64)
        g_own = goals_dir[:, prep.dir_own]
        g_opp = goals_dir[:, prep.dir_opp]

        # Attacking involvement: binomial thinning of the sampled team total.
        goals = rng.binomial(g_own, np.clip(prep.goal_share_pm * minutes, 0.0, 1.0))
        assists = rng.binomial(g_own, np.clip(prep.assist_share_pm * minutes, 0.0, 1.0))

        cs = (bucket == 2) & (g_opp == 0)
        conceded_on = rng.binomial(g_opp, np.clip(prep.conc_alpha * minutes / 90.0, 0.0, 1.0))
        saves = rng.poisson(prep.rate_saves_pm * minutes)

        # DefCon count: NB via gamma-Poisson mixture where a positive stored
        # dispersion exists (X | G ~ Poisson(G * mu / r), G ~ Gamma(r, 1)),
        # else plain Poisson — exactly assemble's fallback split.
        mu_d = prep.rate_defcon_pm * minutes
        gam = rng.gamma(np.clip(prep.defcon_r, 1e-9, None), 1.0, shape)
        lam_nb = np.where(
            prep.defcon_nb, gam * mu_d / np.clip(prep.defcon_r, 1e-9, None), mu_d
        )
        defcon_count = rng.poisson(lam_nb)
        defcon_hit = defcon_count >= prep.defcon_threshold

        yellow = played & (rng.random(shape) < prep.p_yellow)
        red = played & (rng.random(shape) < prep.p_red)
        own_goals = rng.poisson(prep.rate_og_pm * minutes)

        pts = (
            np.where(bucket == 1, prep.pts_short, 0).astype(np.int32)
            + np.where(bucket == 2, prep.pts_long, 0)
            + goals * prep.pts_goal
            + assists * prep.pts_assist
            + cs * prep.pts_cs
            + (conceded_on // 2) * prep.pts_concede
            + (saves // 3) * prep.pts_save
            + defcon_hit * prep.pts_defcon
            + yellow * prep.pts_yellow
            + red * prep.pts_red
            + own_goals * prep.pts_og
        )

        if self.include_bonus:
            # Ranking statistic: analytic center + (variance-matched) realized
            # event deviation + residual noise. See module docstring.
            dev = (
                prep.bps_minute1 * ((bucket == 1) - prep.q1)
                + prep.bps_minute2 * ((bucket == 2) - prep.q2)
                + prep.bps_goal * (goals - prep.e_goals)
                + prep.bps_assist * (assists - prep.e_assists)
                + prep.bps_cs * (cs - prep.e_cs)
                + prep.bps_conc * (conceded_on - prep.e_conc_dev)
                + prep.bps_save * (saves - prep.e_saves_dev)
                + prep.bps_yellow * (yellow - prep.e_yellow)
                + prep.bps_red * (red - prep.e_red)
                + prep.bps_og * (own_goals - prep.e_og)
            )
            bps = prep.center + prep.kappa * dev + rng.normal(0.0, 1.0, shape) * prep.sigma_res
            bps = np.rint(bps)
            bps[~played] = -1e4  # non-players never rank; masked below regardless
            for sl in prep.fixture_slices:
                bonus = _tie_rule_bonus(bps[:, sl].T).T
                pts[:, sl] += bonus * played[:, sl] * prep.pts_bonus[sl]

        pts_out[:] = pts.astype(np.int16)
        minutes_out[:] = np.rint(minutes).astype(np.int16)


def _weighted_dir_mean(
    dir_index: np.ndarray,
    values: np.ndarray,
    weights: np.ndarray,
    usable: np.ndarray,
    n_dir: int,
) -> np.ndarray:
    """Weight-averaged per-direction estimate from usable rows (NaN when none)."""
    idx = dir_index[usable]
    w = weights[usable]
    v = values[usable]
    wsum = np.bincount(idx, weights=w * v, minlength=n_dir)
    wtot = np.bincount(idx, weights=w, minlength=n_dir)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(wtot > 0, wsum / np.where(wtot > 0, wtot, 1.0), np.nan)


# --------------------------------------------------------------------------------------
# Predict-horizon helpers (module-layer hooks for `fplai predict --through-gw`)
# --------------------------------------------------------------------------------------


def chip_window_end(season: int, next_gw: int) -> int:
    """Last GW of the chip window containing ``next_gw`` (19 or 38 for 2025+).

    The default prediction horizon for season simulation: set-1 chips expire at
    the GW19 deadline, so pre-GW20 planning wants predictions through GW19.
    """
    if not 1 <= next_gw <= 38:
        raise ValueError(f"next_gw must be 1..38, got {next_gw}")
    ends = sorted({w.last_gw for w in rules.chip_windows(season) if w.last_gw >= next_gw})
    return ends[0] if ends else 38


def horizon_through_gw(next_gw: int, through_gw: int) -> int:
    """Horizon count for ``run_predict(horizon=...)`` covering ``next_gw..through_gw``.

    Integrator hook for ``fplai predict --through-gw N``: the pipeline's
    ``run_predict``/``fplai predict`` already accept ``horizon`` (the "8" is
    ``config.settings.horizon_gws``); pass
    ``horizon_through_gw(state.next_gw, chip_window_end(season, state.next_gw))``
    to extend live predictions to the end of the current chip window (GW19).
    """
    if through_gw < next_gw:
        raise ValueError(f"through_gw {through_gw} is before next_gw {next_gw}")
    return through_gw - next_gw + 1
