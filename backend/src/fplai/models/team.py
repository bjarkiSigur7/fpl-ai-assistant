"""Dixon-Coles team strength model (MODEL_DESIGN_INPUTS §1.2).

Fits per-team attack ``a_i`` and defence ``b_i`` ratings, a global home advantage ``γ``
(with an optional separately-fitted ``γ_empty`` for empty-stadium fixtures — FPL_KNOWLEDGE
§2.4 regime flags), and the Dixon-Coles low-score correlation ``τ(ρ)``, by weighted maximum
likelihood (scipy L-BFGS-B) with exponential time decay on match weights.

Scoreline model for a fixture with home team ``h`` and away team ``a``::

    λ_home = a_h · b_a · γ        λ_away = a_a · b_h
    p(x, y) = τ(x, y; λ_home, λ_away, ρ) · Pois(x | λ_home) · Pois(y | λ_away)

Identifiability is fixed by the constraint ``Σ_i log a_i = 0`` (defence absorbs the goal
scale). Predictions come from the τ-corrected scoreline grid (goals 0..10). De-margined
bookmaker odds can be blended in via :meth:`TeamModel.blend_odds` — odds take priority per
the §1.2 blending order (odds > Dixon-Coles > FDR-never).

Promoted teams (no PL history in the training window) are seeded from the mean fitted
``(log a, log b)`` of historically promoted teams (teams appearing in a season but absent
the prior one), with a FPL_KNOWLEDGE §3.6 strength-multiplier hook
(``promoted_strength: dict[team_code, float]``).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import optimize, stats

from fplai.rules import SEASON_FLAGS

__all__ = ["TeamModel", "attach_fixture_ids", "demargin_1x2", "empty_stadium_mask"]

#: Scoreline grid extends to this many goals per side (inclusive).
MAX_GOALS = 10

#: Bound on the Dixon-Coles low-score correlation parameter ρ.
_RHO_BOUND = 0.15

#: football-data.co.uk names that normalise differently from FPL names.
_FD_ALIASES = {
    "manunited": "manutd",
    "sheffieldunited": "sheffieldutd",
    "tottenham": "spurs",
}

_EMPTY_RESTART_CUTOFF = pd.Timestamp("2020-06-01", tz="UTC")


def _norm_name(name: str) -> str:
    """Normalise a club name for cross-source joining (lowercase, letters only)."""
    key = re.sub(r"[^a-z]", "", str(name).lower())
    return _FD_ALIASES.get(key, key)


def empty_stadium_mask(fixtures: pd.DataFrame) -> pd.Series:
    """Boolean mask of fixtures played behind closed doors (FPL_KNOWLEDGE §2.4).

    Uses an ``empty_stadium`` column when present, otherwise derives the flag from
    ``rules.SEASON_FLAGS``: all of 2020-21, plus the 2019-20 restart (canonical
    ``gw >= 30``; falls back on kickoff date if ``gw`` is unavailable).
    """
    if "empty_stadium" in fixtures.columns:
        return fixtures["empty_stadium"].fillna(False).astype(bool)
    mask = pd.Series(False, index=fixtures.index)
    for season, sub in fixtures.groupby("season"):
        flag = SEASON_FLAGS.get(int(season))
        regime = flag.empty_stadiums if flag is not None else None
        if regime == "all":
            mask.loc[sub.index] = True
        elif regime == "events_39_47":
            if "gw" in sub.columns and sub["gw"].notna().all():
                mask.loc[sub.index] = (sub["gw"].astype(int) >= 30).to_numpy()
            elif "kickoff_utc" in sub.columns:
                mask.loc[sub.index] = (sub["kickoff_utc"] >= _EMPTY_RESTART_CUTOFF).to_numpy()
    return mask


def attach_fixture_ids(
    odds: pd.DataFrame, fixtures: pd.DataFrame, teams: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Resolve ``fpl_fixture_id`` on an odds frame via team-name matching.

    Joins on ``(season, home_team_code, away_team_code)`` — unique in a single
    round-robin league — after mapping ``*_footballdata_name`` to team codes using
    the per-season ``teams`` table (``name``/``short_name``, normalised, with the
    known football-data aliases). Rows already carrying a non-null id are kept as-is;
    unresolvable rows keep a null id.

    Args:
        odds: odds.parquet-shaped frame (``season``, ``home_footballdata_name``,
            ``away_footballdata_name``, odds/probability columns).
        fixtures: fixtures.parquet-shaped frame covering the odds' seasons.
        teams: teams.parquet-shaped frame; required unless every odds row already
            has a resolved ``fpl_fixture_id``.

    Returns:
        A copy of ``odds`` with ``fpl_fixture_id`` (Int64) filled where resolvable.
    """
    out = odds.copy()
    if "fpl_fixture_id" not in out.columns:
        out["fpl_fixture_id"] = pd.Series(pd.NA, index=out.index, dtype="Int64")
    out["fpl_fixture_id"] = out["fpl_fixture_id"].astype("Int64")
    unresolved = out["fpl_fixture_id"].isna()
    if not unresolved.any():
        return out
    if teams is None:
        raise ValueError("teams frame required to resolve odds rows without fpl_fixture_id")

    name_to_code: dict[tuple[int, str], int] = {}
    for row in teams.itertuples(index=False):
        for col in ("name", "short_name"):
            value = getattr(row, col, None)
            if value is not None and not pd.isna(value):
                name_to_code[(int(row.season), _norm_name(value))] = int(row.team_code)

    sub = out.loc[unresolved]
    h_codes = [
        name_to_code.get((int(s), _norm_name(n)))
        for s, n in zip(sub["season"], sub["home_footballdata_name"], strict=True)
    ]
    a_codes = [
        name_to_code.get((int(s), _norm_name(n)))
        for s, n in zip(sub["season"], sub["away_footballdata_name"], strict=True)
    ]
    key_to_fid = {
        (int(r.season), int(r.home_team_code), int(r.away_team_code)): int(r.fpl_fixture_id)
        for r in fixtures.itertuples(index=False)
    }
    fids = [
        key_to_fid.get((int(s), h, a)) if h is not None and a is not None else None
        for s, h, a in zip(sub["season"], h_codes, a_codes, strict=True)
    ]
    out.loc[unresolved, "fpl_fixture_id"] = pd.array(fids, dtype="Int64")
    return out


def demargin_1x2(odds: pd.DataFrame) -> pd.DataFrame:
    """De-margined 1X2 (and over/under 2.5 where present) probabilities per odds row.

    Prefers pre-computed ``p_h/p_d/p_a`` (and ``p_over25/p_under25``) columns; falls back
    to proportional normalisation of inverse decimal odds. Rows without usable 1X2 data
    get NaN probabilities.
    """
    n = len(odds)
    out = pd.DataFrame(index=odds.index)
    cols = odds.columns
    if {"p_h", "p_d", "p_a"} <= set(cols):
        probs = odds[["p_h", "p_d", "p_a"]].astype(float).to_numpy().copy()
    else:
        probs = np.full((n, 3), np.nan)
    if {"odds_h", "odds_d", "odds_a"} <= set(cols):
        inv = 1.0 / odds[["odds_h", "odds_d", "odds_a"]].astype(float).to_numpy()
        fallback = inv / inv.sum(axis=1, keepdims=True)
        missing = np.isnan(probs).any(axis=1)
        probs[missing] = fallback[missing]
    out[["p_h", "p_d", "p_a"]] = probs

    over = np.full(n, np.nan)
    if {"p_over25", "p_under25"} <= set(cols):
        over = odds["p_over25"].astype(float).to_numpy().copy()
    if {"odds_over25", "odds_under25"} <= set(cols):
        inv_o = 1.0 / odds["odds_over25"].astype(float).to_numpy()
        inv_u = 1.0 / odds["odds_under25"].astype(float).to_numpy()
        fb = inv_o / (inv_o + inv_u)
        missing_o = np.isnan(over)
        over[missing_o] = fb[missing_o]
    out["p_over25"] = over
    return out


def _scoreline_grid(
    lam_home: np.ndarray, lam_away: np.ndarray, rho: float, max_goals: int = MAX_GOALS
) -> np.ndarray:
    """τ-corrected scoreline probability grids, shape ``(n, G+1, G+1)``, each summing to 1.

    Axis 1 is home goals, axis 2 away goals. The four Dixon-Coles low-score cells are
    multiplied by τ and the (truncated) grid renormalised.
    """
    lam_home = np.asarray(lam_home, dtype=float)
    lam_away = np.asarray(lam_away, dtype=float)
    goals = np.arange(max_goals + 1)
    ph = stats.poisson.pmf(goals[None, :], lam_home[:, None])
    pa = stats.poisson.pmf(goals[None, :], lam_away[:, None])
    grid = ph[:, :, None] * pa[:, None, :]
    grid[:, 0, 0] *= np.clip(1.0 - lam_home * lam_away * rho, 0.0, None)
    grid[:, 0, 1] *= np.clip(1.0 + lam_home * rho, 0.0, None)
    grid[:, 1, 0] *= np.clip(1.0 + lam_away * rho, 0.0, None)
    grid[:, 1, 1] *= np.clip(1.0 - rho, 0.0, None)
    grid /= grid.sum(axis=(1, 2), keepdims=True)
    return grid


def _grid_outcomes(grid: np.ndarray) -> dict[str, np.ndarray]:
    """1X2, clean-sheet and over-2.5 probabilities from scoreline grids."""
    size = grid.shape[1]
    home_win = np.tril(np.ones((size, size)), k=-1)
    draw = np.eye(size)
    away_win = np.triu(np.ones((size, size)), k=1)
    x = np.arange(size)
    under25 = ((x[:, None] + x[None, :]) <= 2).astype(float)
    return {
        "p_home_win": np.einsum("nij,ij->n", grid, home_win),
        "p_draw": np.einsum("nij,ij->n", grid, draw),
        "p_away_win": np.einsum("nij,ij->n", grid, away_win),
        "p_cs_home": grid[:, :, 0].sum(axis=1),
        "p_cs_away": grid[:, 0, :].sum(axis=1),
        "p_over25": 1.0 - np.einsum("nij,ij->n", grid, under25),
    }


class TeamModel:
    """Dixon-Coles team goals model with time decay, odds blending and promoted priors.

    Lifecycle per ARCHITECTURE "Model layer contracts": ``fit -> predict_fixtures``
    (``blend_odds`` optionally after) ``-> save / load``.

    Args:
        decay_halflife_days: half-life of the exponential match-weight decay
            (ξ = ln 2 / half-life; ~1 year per §1.2).
        max_goals: scoreline grid extent per side.
        empty_gamma: ``"fit"`` fits a separate home advantage for empty-stadium
            fixtures (needs >= ``min_empty_matches`` such matches, else falls back);
            ``"zero"`` forces γ_empty = 1 (no home advantage behind closed doors).
        promoted_strength: FPL_KNOWLEDGE §3.6 hook — ``{team_code: multiplier}``
            strength multipliers applied on top of the promoted-team prior for teams
            unseen in training (multiplier m: attack ×m, defence weakness ÷m).
        min_empty_matches: minimum empty-stadium matches required to fit γ_empty.
    """

    _PARAMS_FILE = "team_model.json"

    def __init__(
        self,
        *,
        decay_halflife_days: float = 365.0,
        max_goals: int = MAX_GOALS,
        empty_gamma: str = "fit",
        promoted_strength: dict[int, float] | None = None,
        min_empty_matches: int = 10,
    ) -> None:
        if empty_gamma not in ("fit", "zero"):
            raise ValueError("empty_gamma must be 'fit' or 'zero'")
        self.decay_halflife_days = float(decay_halflife_days)
        self.max_goals = int(max_goals)
        self.empty_gamma = empty_gamma
        self.promoted_strength = dict(promoted_strength or {})
        self.min_empty_matches = int(min_empty_matches)

        # Fitted state
        self.team_codes_: list[int] = []
        self.log_a_: np.ndarray = np.empty(0)
        self.log_b_: np.ndarray = np.empty(0)
        self.log_gamma_: float = 0.0
        self.log_gamma_empty_: float = 0.0
        self.rho_: float = 0.0
        self.promoted_codes_: list[int] = []
        self.promoted_prior_: tuple[float, float] = (0.0, 0.0)
        self.as_of_: pd.Timestamp | None = None
        self.n_matches_: int = 0
        self._odds_lookup: pd.DataFrame | None = None

    # ------------------------------------------------------------------ fitting

    def fit(
        self,
        fixtures: pd.DataFrame,
        odds: pd.DataFrame | None = None,
        *,
        train_end: pd.Timestamp | None = None,
        teams: pd.DataFrame | None = None,
    ) -> TeamModel:
        """Fit ratings by weighted MLE on finished, non-void fixtures.

        Args:
            fixtures: fixtures.parquet-shaped frame (any span of seasons; only rows
                with ``finished`` and both goals present are used).
            odds: optional odds.parquet-shaped frame. NOT used in the likelihood —
                stored (with fixture ids attached where resolvable) as the default
                bookmaker benchmark/blend source for :meth:`evaluate`.
            train_end: leakage guard — only fixtures with ``kickoff_utc`` strictly
                before this instant are used, and decay ages are measured from it.
                Defaults to just after the last used kickoff.
            teams: teams.parquet-shaped frame, used only to resolve odds fixture ids.

        Returns:
            self.
        """
        fx = fixtures.loc[
            fixtures["finished"]
            & ~fixtures.get("void", pd.Series(False, index=fixtures.index)).astype(bool)
            & fixtures["home_goals"].notna()
            & fixtures["away_goals"].notna()
        ].copy()
        if train_end is not None:
            train_end = pd.Timestamp(train_end)
            if train_end.tzinfo is None:
                train_end = train_end.tz_localize("UTC")
            fx = fx.loc[fx["kickoff_utc"] < train_end]
        if fx.empty:
            raise ValueError("no finished fixtures to fit on")

        as_of = train_end if train_end is not None else fx["kickoff_utc"].max() + pd.Timedelta("1D")
        age_days = (as_of - fx["kickoff_utc"]).dt.total_seconds().to_numpy() / 86400.0
        weights = np.power(0.5, age_days / self.decay_halflife_days)

        codes = sorted(
            set(fx["home_team_code"].astype(int)) | set(fx["away_team_code"].astype(int))
        )
        code_idx = {c: i for i, c in enumerate(codes)}
        n = len(codes)
        h_idx = fx["home_team_code"].astype(int).map(code_idx).to_numpy()
        a_idx = fx["away_team_code"].astype(int).map(code_idx).to_numpy()
        x = fx["home_goals"].astype(int).to_numpy()
        y = fx["away_goals"].astype(int).to_numpy()
        empty = empty_stadium_mask(fx).to_numpy()

        fit_empty = self.empty_gamma == "fit" and int(empty.sum()) >= self.min_empty_matches
        theta0, bounds = self._initial_params(n, fit_empty, x, y)
        result = optimize.minimize(
            self._nll_and_grad,
            theta0,
            args=(n, h_idx, a_idx, x, y, weights, empty, fit_empty),
            jac=True,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 1000, "ftol": 1e-10, "gtol": 1e-8},
        )
        if not result.success:
            raise RuntimeError(f"Dixon-Coles MLE did not converge: {result.message}")
        la, lb, lg, lg_e, rho = self._unpack(result.x, n, fit_empty)
        if not fit_empty and self.empty_gamma == "fit" and empty.any():
            lg_e = 0.0  # too few empty matches to identify — no home advantage

        self.team_codes_ = codes
        self.log_a_ = la
        self.log_b_ = lb
        self.log_gamma_ = float(lg)
        self.log_gamma_empty_ = float(lg_e) if self.empty_gamma == "fit" else 0.0
        self.rho_ = float(rho)
        self.as_of_ = as_of
        self.n_matches_ = len(fx)
        self._set_promoted_prior(fx)
        self._odds_lookup = None
        if odds is not None:
            attached = attach_fixture_ids(odds, fixtures, teams) if teams is not None else odds
            self._odds_lookup = attached
        return self

    def _initial_params(
        self, n: int, fit_empty: bool, x: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, list[tuple[float, float]]]:
        """Starting parameter vector and L-BFGS-B bounds."""
        mean_goals = max(float(np.mean(np.concatenate([x, y]))), 0.1)
        n_params = (n - 1) + n + 1 + (1 if fit_empty else 0) + 1
        theta0 = np.zeros(n_params)
        theta0[n - 1 : 2 * n - 1] = np.log(mean_goals)  # defence absorbs the goal scale
        theta0[2 * n - 1] = 0.2  # home advantage ~1.22
        bounds: list[tuple[float, float]] = [(-3.0, 3.0)] * (n - 1)
        bounds += [(-3.0, 3.0)] * n
        bounds += [(-1.0, 1.0)]
        if fit_empty:
            bounds += [(-1.0, 1.0)]
        bounds += [(-_RHO_BOUND, _RHO_BOUND)]
        return theta0, bounds

    @staticmethod
    def _unpack(
        theta: np.ndarray, n: int, fit_empty: bool
    ) -> tuple[np.ndarray, np.ndarray, float, float, float]:
        """Split the flat parameter vector; reconstruct log_a with Σ log a = 0."""
        la_free = theta[: n - 1]
        la = np.append(la_free, -la_free.sum())
        lb = theta[n - 1 : 2 * n - 1]
        lg = float(theta[2 * n - 1])
        idx = 2 * n
        lg_e = float(theta[idx]) if fit_empty else 0.0
        rho = float(theta[-1])
        return la, lb, lg, lg_e, rho

    @staticmethod
    def _nll_and_grad(
        theta: np.ndarray,
        n: int,
        h_idx: np.ndarray,
        a_idx: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        w: np.ndarray,
        empty: np.ndarray,
        fit_empty: bool,
    ) -> tuple[float, np.ndarray]:
        """Weighted negative log-likelihood and analytic gradient (vectorised)."""
        la, lb, lg, lg_e, rho = TeamModel._unpack(theta, n, fit_empty)
        home_adv = np.where(empty, lg_e if fit_empty else 0.0, lg)
        log_lam = la[h_idx] + lb[a_idx] + home_adv
        log_mu = la[a_idx] + lb[h_idx]
        lam = np.exp(log_lam)
        mu = np.exp(log_mu)

        ll = x * log_lam - lam + y * log_mu - mu
        gl = x - lam  # d ll / d log λ
        gm = y - mu  # d ll / d log μ
        gr = np.zeros_like(lam)  # d ll / d ρ

        eps = 1e-10
        m00 = (x == 0) & (y == 0)
        d = np.clip(1.0 - lam[m00] * mu[m00] * rho, eps, None)
        ll[m00] += np.log(d)
        gl[m00] += -(lam[m00] * mu[m00] * rho) / d
        gm[m00] += -(lam[m00] * mu[m00] * rho) / d
        gr[m00] = -(lam[m00] * mu[m00]) / d

        m01 = (x == 0) & (y == 1)
        d = np.clip(1.0 + lam[m01] * rho, eps, None)
        ll[m01] += np.log(d)
        gl[m01] += (lam[m01] * rho) / d
        gr[m01] = lam[m01] / d

        m10 = (x == 1) & (y == 0)
        d = np.clip(1.0 + mu[m10] * rho, eps, None)
        ll[m10] += np.log(d)
        gm[m10] += (mu[m10] * rho) / d
        gr[m10] = mu[m10] / d

        m11 = (x == 1) & (y == 1)
        d = max(1.0 - rho, eps)
        ll[m11] += np.log(d)
        gr[m11] = -1.0 / d

        nll = -float(np.sum(w * ll))
        g_la = -(np.bincount(h_idx, w * gl, n) + np.bincount(a_idx, w * gm, n))
        g_lb = -(np.bincount(a_idx, w * gl, n) + np.bincount(h_idx, w * gm, n))
        # γ multiplies λ only on non-empty fixtures (empty ones use γ_empty or none at all)
        g_lg = -float(np.sum(w[~empty] * gl[~empty]))
        grad = np.concatenate(
            [
                g_la[: n - 1] - g_la[n - 1],  # reduced parameterisation, Σ log a = 0
                g_lb,
                [g_lg],
                [-float(np.sum(w[empty] * gl[empty]))] if fit_empty else [],
                [-float(np.sum(w * gr))],
            ]
        )
        return nll, grad

    def _set_promoted_prior(self, fx: pd.DataFrame) -> None:
        """Derive the promoted-team prior from the fitted ratings.

        Promoted team-seasons are teams appearing in a season but not the prior one
        (within the training window). The prior is the mean fitted ``(log a, log b)``
        over those teams; with no detectable promotions (single-season window) it
        falls back to a weak-team proxy (20th-pct attack, 80th-pct defence weakness).
        """
        by_season: dict[int, set[int]] = {
            int(s): set(sub["home_team_code"].astype(int)) | set(sub["away_team_code"].astype(int))
            for s, sub in fx.groupby("season")
        }
        promoted: set[int] = set()
        for season, members in by_season.items():
            prev = by_season.get(season - 1)
            if prev is not None:
                promoted |= members - prev
        self.promoted_codes_ = sorted(promoted)
        if promoted:
            idx = [self.team_codes_.index(c) for c in self.promoted_codes_]
            self.promoted_prior_ = (
                float(self.log_a_[idx].mean()),
                float(self.log_b_[idx].mean()),
            )
        else:
            self.promoted_prior_ = (
                float(np.quantile(self.log_a_, 0.2)),
                float(np.quantile(self.log_b_, 0.8)),
            )

    # ------------------------------------------------------------------ prediction

    def _ratings_for(self, codes: pd.Series) -> tuple[np.ndarray, np.ndarray]:
        """(log a, log b) per team code; unseen codes get the promoted prior + §3.6 hook."""
        idx = {c: i for i, c in enumerate(self.team_codes_)}
        la = np.empty(len(codes))
        lb = np.empty(len(codes))
        prior_a, prior_b = self.promoted_prior_
        for j, code in enumerate(codes.astype(int)):
            i = idx.get(int(code))
            if i is not None:
                la[j], lb[j] = self.log_a_[i], self.log_b_[i]
            else:
                shift = np.log(self.promoted_strength.get(int(code), 1.0))
                la[j] = prior_a + shift
                lb[j] = prior_b - shift
        return la, lb

    def _lambdas(self, fixtures: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Poisson rates (λ_home, λ_away) for a fixtures frame."""
        if not self.team_codes_:
            raise RuntimeError("TeamModel is not fitted")
        la_h, lb_h = self._ratings_for(fixtures["home_team_code"])
        la_a, lb_a = self._ratings_for(fixtures["away_team_code"])
        empty = empty_stadium_mask(fixtures).to_numpy()
        home_adv = np.where(empty, self.log_gamma_empty_, self.log_gamma_)
        lam_home = np.exp(la_h + lb_a + home_adv)
        lam_away = np.exp(la_a + lb_h)
        return lam_home, lam_away

    def predict_fixtures(self, fixtures: pd.DataFrame) -> pd.DataFrame:
        """Predict fixture-level goal rates and outcome probabilities.

        Returns the ARCHITECTURE contract frame: one row per fixture with
        ``season, fpl_fixture_id, home_lambda, away_lambda, p_cs_home, p_cs_away,
        p_home_win, p_draw, p_away_win`` — all probabilities from the τ-corrected
        scoreline grid (goals 0..``max_goals``), 1X2 summing to 1.
        """
        lam_home, lam_away = self._lambdas(fixtures)
        grid = _scoreline_grid(lam_home, lam_away, self.rho_, self.max_goals)
        stats_ = _grid_outcomes(grid)
        return pd.DataFrame(
            {
                "season": fixtures["season"].astype(int).to_numpy(),
                "fpl_fixture_id": fixtures["fpl_fixture_id"].astype(int).to_numpy(),
                "home_lambda": lam_home,
                "away_lambda": lam_away,
                "p_cs_home": stats_["p_cs_home"],
                "p_cs_away": stats_["p_cs_away"],
                "p_home_win": stats_["p_home_win"],
                "p_draw": stats_["p_draw"],
                "p_away_win": stats_["p_away_win"],
            }
        )

    # ------------------------------------------------------------------ odds blending

    def blend_odds(
        self, pred: pd.DataFrame, odds: pd.DataFrame, weight: float = 0.7
    ) -> pd.DataFrame:
        """Recalibrate predictions toward de-margined bookmaker odds where markets exist.

        For each prediction row with a matching odds row (joined on
        ``season, fpl_fixture_id`` — use :func:`attach_fixture_ids` first if the odds
        frame lacks ids), the target outcome probabilities are the blend
        ``weight · odds + (1 − weight) · model`` of the de-margined 1X2 (plus over/under
        2.5 when present), and (λ_home, λ_away) are re-solved by least squares so the
        τ-corrected grid (ρ kept from the fit) reproduces the targets
        (moment-matching on P(home win), P(away win)[, P(over 2.5)]). Fixtures without
        odds keep their Dixon-Coles values — odds take priority, DC fills the rest
        (MODEL_DESIGN_INPUTS §1.2).

        Returns a copy of ``pred`` with recomputed rate/probability columns and a
        boolean ``odds_blended`` column.
        """
        if not 0.0 <= weight <= 1.0:
            raise ValueError("weight must be in [0, 1]")
        out = pred.copy()
        out["odds_blended"] = False
        if "fpl_fixture_id" not in odds.columns or odds["fpl_fixture_id"].isna().all():
            raise ValueError(
                "odds frame has no resolved fpl_fixture_id — call attach_fixture_ids() first"
            )
        market = odds.loc[odds["fpl_fixture_id"].notna()].copy()
        probs = demargin_1x2(market)
        market_key = {
            (int(s), int(f)): (float(h), float(d), float(a), float(o) if pd.notna(o) else None)
            for s, f, h, d, a, o in zip(
                market["season"],
                market["fpl_fixture_id"],
                probs["p_h"],
                probs["p_d"],
                probs["p_a"],
                probs["p_over25"],
                strict=True,
            )
            if pd.notna(h) and pd.notna(d) and pd.notna(a)
        }

        for i in out.index:
            key = (int(out.at[i, "season"]), int(out.at[i, "fpl_fixture_id"]))
            entry = market_key.get(key)
            if entry is None:
                continue
            po_h, po_d, po_a, po_over = entry
            lam0 = float(out.at[i, "home_lambda"])
            mu0 = float(out.at[i, "away_lambda"])
            dc = _grid_outcomes(
                _scoreline_grid(np.array([lam0]), np.array([mu0]), self.rho_, self.max_goals)
            )
            t_h = weight * po_h + (1 - weight) * float(dc["p_home_win"][0])
            t_a = weight * po_a + (1 - weight) * float(dc["p_away_win"][0])
            t_over = (
                weight * po_over + (1 - weight) * float(dc["p_over25"][0])
                if po_over is not None
                else None
            )
            lam, mu = self._solve_lambdas(t_h, t_a, t_over, lam0, mu0)
            g = _grid_outcomes(
                _scoreline_grid(np.array([lam]), np.array([mu]), self.rho_, self.max_goals)
            )
            out.at[i, "home_lambda"] = lam
            out.at[i, "away_lambda"] = mu
            out.at[i, "p_cs_home"] = float(g["p_cs_home"][0])
            out.at[i, "p_cs_away"] = float(g["p_cs_away"][0])
            out.at[i, "p_home_win"] = float(g["p_home_win"][0])
            out.at[i, "p_draw"] = float(g["p_draw"][0])
            out.at[i, "p_away_win"] = float(g["p_away_win"][0])
            out.at[i, "odds_blended"] = True
        return out

    def _solve_lambdas(
        self, t_h: float, t_a: float, t_over: float | None, lam0: float, mu0: float
    ) -> tuple[float, float]:
        """Root-find (λ_home, λ_away) whose τ-grid matches the target probabilities."""

        def residuals(z: np.ndarray) -> np.ndarray:
            lam = np.exp(z)
            g = _grid_outcomes(_scoreline_grid(lam[:1], lam[1:], self.rho_, self.max_goals))
            r = [float(g["p_home_win"][0]) - t_h, float(g["p_away_win"][0]) - t_a]
            if t_over is not None:
                r.append(float(g["p_over25"][0]) - t_over)
            return np.array(r)

        z0 = np.log(np.clip([lam0, mu0], 0.05, 8.0))
        result = optimize.least_squares(
            residuals, z0, bounds=(np.log(0.05), np.log(8.0)), xtol=1e-12, ftol=1e-12
        )
        lam, mu = np.exp(result.x)
        return float(lam), float(mu)

    # ------------------------------------------------------------------ evaluation

    def evaluate(
        self,
        fixtures_holdout: pd.DataFrame,
        odds: pd.DataFrame | None = None,
        *,
        teams: pd.DataFrame | None = None,
    ) -> dict[str, float]:
        """1X2 log-loss and Brier score vs uniform and de-margined closing odds.

        The bar (MODEL_DESIGN_INPUTS §1.2/§6): get close to the odds benchmark;
        beating closing odds is not expected.

        Args:
            fixtures_holdout: finished fixtures to score (must postdate the training
                window — the caller owns walk-forward discipline).
            odds: optional odds frame for the bookmaker benchmark; defaults to the
                odds passed to :meth:`fit`.
            teams: optional teams.parquet-shaped frame — when given and the odds
                frame lacks resolved ids, fixture ids are attached against the
                holdout fixtures by team-name matching.

        Returns:
            dict with ``n``, ``log_loss``, ``brier``, ``log_loss_uniform``,
            ``brier_uniform`` and — where odds cover part of the holdout —
            ``n_odds``, ``log_loss_odds``, ``brier_odds``,
            ``log_loss_model_on_odds_subset``, ``brier_model_on_odds_subset``.
        """
        fx = fixtures_holdout.loc[
            fixtures_holdout["finished"]
            & fixtures_holdout["home_goals"].notna()
            & fixtures_holdout["away_goals"].notna()
        ]
        if "void" in fx.columns:
            fx = fx.loc[~fx["void"].astype(bool)]
        if fx.empty:
            raise ValueError("no finished fixtures in holdout")
        pred = self.predict_fixtures(fx)
        outcome = np.select(
            [
                fx["home_goals"].astype(int).to_numpy() > fx["away_goals"].astype(int).to_numpy(),
                fx["home_goals"].astype(int).to_numpy() == fx["away_goals"].astype(int).to_numpy(),
            ],
            [0, 1],
            default=2,
        )
        model_p = pred[["p_home_win", "p_draw", "p_away_win"]].to_numpy()
        uniform_p = np.full_like(model_p, 1.0 / 3.0)
        metrics: dict[str, float] = {
            "n": float(len(fx)),
            "log_loss": _log_loss(model_p, outcome),
            "brier": _brier(model_p, outcome),
            "log_loss_uniform": _log_loss(uniform_p, outcome),
            "brier_uniform": _brier(uniform_p, outcome),
        }

        odds_frame = odds if odds is not None else self._odds_lookup
        if odds_frame is not None:
            needs_ids = (
                "fpl_fixture_id" not in odds_frame.columns
                or odds_frame["fpl_fixture_id"].isna().all()
            )
            has_names = {"home_footballdata_name", "away_footballdata_name"} <= set(
                odds_frame.columns
            )
            if needs_ids and has_names and teams is not None:
                odds_frame = attach_fixture_ids(odds_frame, fx, teams)
            if (
                "fpl_fixture_id" in odds_frame.columns
                and odds_frame["fpl_fixture_id"].notna().any()
            ):
                op = demargin_1x2(odds_frame)
                op = op.assign(
                    season=odds_frame["season"].astype(int),
                    fpl_fixture_id=odds_frame["fpl_fixture_id"],
                ).dropna(subset=["fpl_fixture_id", "p_h", "p_d", "p_a"])
                op["fpl_fixture_id"] = op["fpl_fixture_id"].astype(int)
                joined = pd.DataFrame(
                    {
                        "season": fx["season"].astype(int).to_numpy(),
                        "fpl_fixture_id": fx["fpl_fixture_id"].astype(int).to_numpy(),
                        "outcome": outcome,
                        "m_h": model_p[:, 0],
                        "m_d": model_p[:, 1],
                        "m_a": model_p[:, 2],
                    }
                ).merge(op, on=["season", "fpl_fixture_id"], how="inner")
                if len(joined):
                    odds_p = joined[["p_h", "p_d", "p_a"]].to_numpy(dtype=float)
                    sub_out = joined["outcome"].to_numpy()
                    model_sub = joined[["m_h", "m_d", "m_a"]].to_numpy(dtype=float)
                    metrics.update(
                        {
                            "n_odds": float(len(joined)),
                            "log_loss_odds": _log_loss(odds_p, sub_out),
                            "brier_odds": _brier(odds_p, sub_out),
                            "log_loss_model_on_odds_subset": _log_loss(model_sub, sub_out),
                            "brier_model_on_odds_subset": _brier(model_sub, sub_out),
                        }
                    )
        return metrics

    # ------------------------------------------------------------------ persistence

    def save(self, dir: Path) -> None:
        """Persist fitted parameters (JSON) under ``dir``."""
        dir = Path(dir)
        dir.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "decay_halflife_days": self.decay_halflife_days,
            "max_goals": self.max_goals,
            "empty_gamma": self.empty_gamma,
            "promoted_strength": {str(k): v for k, v in self.promoted_strength.items()},
            "min_empty_matches": self.min_empty_matches,
            "team_codes": self.team_codes_,
            "log_a": self.log_a_.tolist(),
            "log_b": self.log_b_.tolist(),
            "log_gamma": self.log_gamma_,
            "log_gamma_empty": self.log_gamma_empty_,
            "rho": self.rho_,
            "promoted_codes": self.promoted_codes_,
            "promoted_prior": list(self.promoted_prior_),
            "as_of": self.as_of_.isoformat() if self.as_of_ is not None else None,
            "n_matches": self.n_matches_,
        }
        (dir / self._PARAMS_FILE).write_text(json.dumps(payload, indent=2))

    @classmethod
    def load(cls, dir: Path) -> TeamModel:
        """Restore a fitted model saved with :meth:`save`."""
        payload = json.loads((Path(dir) / cls._PARAMS_FILE).read_text())
        model = cls(
            decay_halflife_days=payload["decay_halflife_days"],
            max_goals=payload["max_goals"],
            empty_gamma=payload["empty_gamma"],
            promoted_strength={int(k): v for k, v in payload["promoted_strength"].items()},
            min_empty_matches=payload["min_empty_matches"],
        )
        model.team_codes_ = [int(c) for c in payload["team_codes"]]
        model.log_a_ = np.array(payload["log_a"], dtype=float)
        model.log_b_ = np.array(payload["log_b"], dtype=float)
        model.log_gamma_ = float(payload["log_gamma"])
        model.log_gamma_empty_ = float(payload["log_gamma_empty"])
        model.rho_ = float(payload["rho"])
        model.promoted_codes_ = [int(c) for c in payload["promoted_codes"]]
        model.promoted_prior_ = (
            float(payload["promoted_prior"][0]),
            float(payload["promoted_prior"][1]),
        )
        model.as_of_ = pd.Timestamp(payload["as_of"]) if payload["as_of"] else None
        model.n_matches_ = int(payload["n_matches"])
        return model


def _log_loss(probs: np.ndarray, outcome: np.ndarray) -> float:
    """Mean negative log probability of the realised outcome (columns H/D/A)."""
    p = np.clip(probs[np.arange(len(outcome)), outcome], 1e-12, 1.0)
    return float(-np.mean(np.log(p)))


def _brier(probs: np.ndarray, outcome: np.ndarray) -> float:
    """Multiclass Brier score: mean squared distance to the one-hot outcome."""
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(outcome)), outcome] = 1.0
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))
