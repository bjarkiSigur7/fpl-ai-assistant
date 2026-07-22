"""Per-90 event-rate models: goals, assists, GK saves, DefCon, cards, own goals.

Implements the ``models/rates.py`` contract from ``docs/ARCHITECTURE.md`` and the maths of
``docs/MODEL_DESIGN_INPUTS.md`` §1.1 items 2 (attacking returns), 4 (saves), 5 (defensive
contribution) and 7 (cards / own goals).

Design notes (load-bearing — read before touching)
--------------------------------------------------
**Exposure mechanism.** Goals/assists/saves/DefCon are per-90 Poisson intensities but the
training rows are raw per-match counts with unequal minutes. We use the classic GLM *offset*:
LightGBM is fitted with ``objective="poisson"`` on the raw counts ``y`` with
``init_score = log(minutes/90 * base)`` where ``base`` is the pooled per-90 rate of that
(target, position) cell. The booster therefore learns ``F(x) = log E[y] - init_score``, a log
multiplicative correction to the position rate. At predict time LightGBM does **not** add the
``init_score`` offset, so the per-90 rate is ``base * booster.predict(X) == base * exp(F(x))``.
Centering on ``base`` matters: with ``init_score`` given, LightGBM disables
``boost_from_average``, and a heavily regularized booster started from rate=1.0/90 can fail to
recover the intercept (observed on real data). This offset also makes exposure count as
evidence: a 45-minute player with one goal resolves to a *higher* per-90 rate than a 90-minute
player with one goal (verified in ``tests/test_rates.py``).

**Fixture adjustment.** Rates are conditioned only on the ``f_*`` feature columns (which may
legitimately include opponent-strength windows built strictly pre-deadline). The Dixon-Coles
fixture multiplier ``F_att(m)`` from ``models/team.py`` is applied downstream in
``models/assemble.py`` — ``RatesModel`` must NOT multiply it in, or it would be double-counted.

**Minutes.** All outputs are per-90 (or per-90 probabilities, see below); scaling by expected
minutes / minute-bucket probabilities also happens in ``assemble`` via ``models/minutes.py``.

**DefCon labels.** ``defensive_contribution`` (2025+) *is* the position-relevant count: CBIT
(clearances+blocks+interceptions+tackles) for DEF, CBIRT (CBIT+recoveries) for MID/FWD, 0 for
GKP (verified against raw columns on 2025 data). For 2016-18 the rich per-match stats allow an
exact reconstruction from ``tackles``/``clearances_blocks_interceptions``/``recoveries``.
Seasons 2019-24 have no label and are simply absent from DefCon training (approximation: the
2016-18 rows come from the 3-subs, pre-VAR era; Opta definitions are assumed stable).
``defcon_disp`` is the NB2 dispersion alpha (``Var = mu + alpha * mu^2``; NB "size"
``r = 1/alpha``), estimated per position by method of moments on training residuals; thin
positions fall back to the pooled estimate.

**Cards.** ``p_yellow`` / ``p_red`` are probabilities *per full 90 played*
(``1 - exp(-lam_card)`` with ``lam_card`` a per-90 rate); ``assemble`` scales by expected
minutes. Player rates use empirical-Bayes Gamma-Poisson shrinkage: a per-position Gamma prior
``(a, b)`` fitted by method of moments (prior mean = position rate, prior strength ``b`` in
units of 90s derived from the between-player variance), posterior mean
``(cards + a) / (nineties + b)``. Unseen players get the position prior mean.

**Own goals** are a tiny per-position pooled per-90 prior (``lam_og`` ~ 0.002-0.008).

**Leakage.** This module never builds features; it consumes the ``f_*`` columns of
``features/windows.py`` (contract: computed strictly from pre-deadline information) plus label
columns. Fitting on a frame that satisfies that contract cannot leak.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

#: Feature-column prefix, mirroring the ``features/windows.py`` contract.
FEATURE_PREFIX = "f_"

#: Mandatory key columns on every prediction frame (player-level contract).
KEY_COLUMNS: tuple[str, ...] = ("season", "gw", "player_code")

#: Optional key carried through when present (match-level frames).
OPTIONAL_KEY_COLUMNS: tuple[str, ...] = ("fpl_fixture_id",)

#: Output columns of :meth:`RatesModel.predict`, in contract order.
PREDICTION_COLUMNS: tuple[str, ...] = (
    "lam_goal",
    "lam_assist",
    "lam_saves",
    "lam_defcon",
    "defcon_disp",
    "p_yellow",
    "p_red",
    "lam_og",
)

_GBM_TARGETS: dict[str, tuple[str, ...]] = {
    "goal": ("DEF", "MID", "FWD"),
    "assist": ("DEF", "MID", "FWD"),
    "saves": ("GKP",),
    "defcon": ("DEF", "MID", "FWD"),
}
_TARGET_LABELS: dict[str, str] = {"goal": "goals_scored", "assist": "assists", "saves": "saves"}
#: Per-90 rate clip ranges applied to every prediction (sanity bounds, not tuning knobs).
_RATE_CLIPS: dict[str, tuple[float, float]] = {
    "goal": (1e-6, 3.0),
    "assist": (1e-6, 3.0),
    "saves": (1e-6, 12.0),
    "defcon": (1e-6, 40.0),
}
_REQUIRED_LABELS: tuple[str, ...] = (
    "minutes",
    "goals_scored",
    "assists",
    "saves",
    "yellow_cards",
    "red_cards",
    "own_goals",
)
# Defaults picked on real data (train <=2024, eval 2025): heavily regularized trees beat
# looser ones on holdout Poisson deviance for goals/assists/saves alike.
_DEFAULT_LGBM_PARAMS: dict[str, Any] = {
    "n_estimators": 600,
    "learning_rate": 0.02,
    "num_leaves": 7,
    "min_child_samples": 250,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.7,
    "reg_lambda": 5.0,
}
#: Dispersion used for positions with no DefCon training data at all (mild overdispersion).
_DEFAULT_DEFCON_DISP = 0.30


def _num(s: pd.Series) -> np.ndarray:
    """Series -> float64 ndarray with pandas NA mapped to NaN (handles nullable dtypes)."""
    return pd.to_numeric(s, errors="coerce").astype("float64").to_numpy()


def _matrix(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    """Feature matrix as float64 with NaN for missing (LightGBM handles NaN natively)."""
    return df[cols].apply(pd.to_numeric, errors="coerce").astype("float64").to_numpy()


def _position_series(df: pd.DataFrame) -> pd.Series:
    """Resolve the position column (``position`` preferred, ``f_position`` fallback)."""
    for col in ("position", "f_position"):
        if col in df.columns:
            return df[col].astype(str).str.upper()
    raise KeyError("frame must carry a 'position' (or 'f_position') column")


def defcon_counts(df: pd.DataFrame) -> pd.Series:
    """Position-aware DefCon count label: CBIT for DEF, CBIRT for MID/FWD, 0 for GKP.

    Sources, in priority order per row: an explicit ``defcon_count`` column; the native
    ``defensive_contribution`` column (2025+ — verified to equal CBIT/CBIRT); reconstruction
    from raw ``tackles`` + ``clearances_blocks_interceptions`` (+ ``recoveries`` for MID/FWD),
    available 2016-18. Rows with no source stay NaN and are excluded from DefCon training.
    """
    pos = _position_series(df)
    out = pd.Series(np.nan, index=df.index, dtype="float64")
    if "defcon_count" in df.columns:
        out = pd.Series(_num(df["defcon_count"]), index=df.index)
    if "defensive_contribution" in df.columns:
        out = out.fillna(pd.Series(_num(df["defensive_contribution"]), index=df.index))
    if {"tackles", "clearances_blocks_interceptions"}.issubset(df.columns):
        cbit = _num(df["tackles"]) + _num(df["clearances_blocks_interceptions"])
        if "recoveries" in df.columns:
            cbirt = cbit + _num(df["recoveries"])
        else:
            cbirt = np.full(len(df), np.nan)
        recon = pd.Series(np.where(pos.to_numpy() == "DEF", cbit, cbirt), index=df.index)
        out = out.fillna(recon)
    out[pos == "GKP"] = 0.0
    return out


def poisson_deviance(y: np.ndarray, mu: np.ndarray) -> float:
    """Mean Poisson deviance ``2 * mean(y*log(y/mu) - (y - mu))`` (y*log(y/mu) := 0 at y=0)."""
    y = np.asarray(y, dtype="float64")
    mu = np.clip(np.asarray(mu, dtype="float64"), 1e-9, None)
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(y > 0, y * np.log(np.where(y > 0, y, 1.0) / mu), 0.0)
    return float(np.mean(2.0 * (term - (y - mu))))


def nb_dispersion_mom(y: np.ndarray, mu: np.ndarray) -> float:
    """Method-of-moments NB2 dispersion alpha for ``Var = mu + alpha * mu^2``.

    ``alpha = sum((y - mu)^2 - mu) / sum(mu^2)``, floored at 0 (pure Poisson) and capped at 10.
    """
    y = np.asarray(y, dtype="float64")
    mu = np.asarray(mu, dtype="float64")
    den = float(np.sum(mu**2))
    if den <= 0.0:
        return 0.0
    alpha = float(np.sum((y - mu) ** 2 - mu) / den)
    return float(np.clip(alpha, 0.0, 10.0))


def _eb_gamma_prior(y: np.ndarray, e: np.ndarray) -> tuple[float, float]:
    """Method-of-moments Gamma prior ``(a, b)`` for per-90 Poisson rates with exposures.

    ``y``/``e`` are per-player event totals and exposures (nineties). The prior mean is the
    pooled rate ``m``; the prior strength ``b`` (in nineties) comes from the excess of the
    observed between-player rate variance over the variance Poisson noise alone would produce.
    ``b`` is clamped to [5, 5000]: near-zero between-variance (e.g. red cards) collapses to a
    strong prior instead of a degenerate one. Posterior mean rate = ``(y_p + a) / (e_p + b)``.
    """
    y = np.asarray(y, dtype="float64")
    e = np.asarray(e, dtype="float64")
    total_e = float(e.sum())
    m = float(y.sum() / total_e) if total_e > 0 else 1e-4
    m = max(m, 1e-6)
    mask = e >= 3.0
    if int(mask.sum()) < 10:
        b = 50.0
        return m * b, b
    rates = y[mask] / e[mask]
    w = e[mask]
    var_obs = float(np.average((rates - m) ** 2, weights=w))
    var_poisson = m * float(np.average(1.0 / e[mask], weights=w))
    tau2 = max(var_obs - var_poisson, m**2 * 1e-4)
    b = float(np.clip(m / tau2, 5.0, 5000.0))
    return m * b, b


class RatesModel:
    """Per-90 event-rate models per player-fixture (ARCHITECTURE ``models/rates.py`` contract).

    Lifecycle: ``fit(features_with_labels) -> self``, ``predict(features) -> DataFrame``,
    ``evaluate(holdout) -> dict``, ``save(dir)`` / ``RatesModel.load(dir)``.

    Parameters
    ----------
    seed:
        Seed for LightGBM (``deterministic=True``, ``n_jobs=1`` are forced — fits are
        bit-reproducible).
    lgbm_params:
        Overrides merged over :data:`_DEFAULT_LGBM_PARAMS` (objective/seed keys are forced).
    min_rows, min_events:
        Per (target, position) thresholds below which the GBM is skipped and the pooled
        empirical position rate is used instead (the fallback is always fitted).
    """

    def __init__(
        self,
        *,
        seed: int = 7,
        lgbm_params: dict[str, Any] | None = None,
        min_rows: int = 300,
        min_events: int = 25,
    ) -> None:
        self.seed = seed
        self.lgbm_params: dict[str, Any] = {**_DEFAULT_LGBM_PARAMS, **(lgbm_params or {})}
        self.min_rows = min_rows
        self.min_events = min_events
        self._feature_cols: list[str] = []
        self._boosters: dict[tuple[str, str], lgb.Booster] = {}
        self._fallback_rates: dict[str, dict[str, float]] = {}
        self._defcon_disp: dict[str, float] = {}
        self._card_priors: dict[str, dict[str, float]] = {}
        self._card_players: pd.DataFrame | None = None
        self._og_rates: dict[str, float] = {}
        self._fitted = False

    # ------------------------------------------------------------------ fitting

    def fit(self, features_with_labels: pd.DataFrame) -> RatesModel:
        """Fit all rate components on per-match rows (only ``minutes > 0`` rows are used).

        The frame must carry ``position`` (or ``f_position``), the label columns
        ``minutes, goals_scored, assists, saves, yellow_cards, red_cards, own_goals`` and any
        number of numeric ``f_*`` feature columns. DefCon labels are resolved per row by
        :func:`defcon_counts`; rows without a resolvable count are excluded from DefCon
        training only. Void-GW rows (0 minutes for everyone) drop out automatically.
        """
        missing = [c for c in _REQUIRED_LABELS if c not in features_with_labels.columns]
        if missing:
            raise KeyError(f"fit frame is missing label columns: {missing}")
        df = features_with_labels[_num(features_with_labels["minutes"]) > 0].copy()
        if df.empty:
            raise ValueError("no rows with minutes > 0 to fit on")
        pos = _position_series(df)
        exposure = _num(df["minutes"]) / 90.0
        self._feature_cols = sorted(
            c
            for c in df.columns
            if c.startswith(FEATURE_PREFIX) and pd.api.types.is_numeric_dtype(df[c])
        )

        labels: dict[str, np.ndarray] = {
            t: _num(df[col]) for t, col in _TARGET_LABELS.items()
        }
        labels["defcon"] = defcon_counts(df).to_numpy()

        for target, positions in _GBM_TARGETS.items():
            self._fallback_rates[target] = {}
            for p in positions:
                mask = (pos == p).to_numpy() & ~np.isnan(labels[target])
                self._fit_one(target, p, df.loc[mask], labels[target][mask], exposure[mask])
            # Positions outside the GBM set still need a fallback (e.g. GKP goals/assists).
            for p in ("GKP", "DEF", "MID", "FWD"):
                if p not in self._fallback_rates[target]:
                    mask = (pos == p).to_numpy() & ~np.isnan(labels[target])
                    self._fallback_rates[target][p] = self._pooled_rate(
                        labels[target][mask], exposure[mask]
                    )

        self._fit_defcon_dispersion(df, pos, labels["defcon"], exposure)
        self._fit_cards(df, pos, exposure)
        self._og_rates = {
            p: self._pooled_rate(_num(df.loc[(pos == p).to_numpy(), "own_goals"]),
                                 exposure[(pos == p).to_numpy()])
            for p in ("GKP", "DEF", "MID", "FWD")
        }
        self._fitted = True
        return self

    @staticmethod
    def _pooled_rate(y: np.ndarray, e: np.ndarray) -> float:
        """Pooled empirical per-90 rate with a tiny floor (safe for empty subsets)."""
        total_e = float(np.nansum(e))
        if total_e <= 0.0:
            return 1e-5
        return max(float(np.nansum(y)) / total_e, 1e-5)

    def _fit_one(
        self,
        target: str,
        position: str,
        sub: pd.DataFrame,
        y: np.ndarray,
        e: np.ndarray,
    ) -> None:
        """Fit one (target, position) cell: fallback rate always, GBM when data allows."""
        base = self._pooled_rate(y, e)
        self._fallback_rates[target][position] = base
        enough = len(sub) >= self.min_rows and float(np.sum(y)) >= self.min_events
        if not enough or not self._feature_cols:
            return
        model = lgb.LGBMRegressor(
            objective="poisson",
            random_state=self.seed,
            deterministic=True,
            force_row_wise=True,
            n_jobs=1,
            verbosity=-1,
            **self.lgbm_params,
        )
        # Offset = log(exposure * pooled rate): the booster learns a multiplicative
        # correction to the position rate (see module docstring for why not log(e) alone).
        model.fit(_matrix(sub, self._feature_cols), y, init_score=np.log(e * base))
        self._boosters[(target, position)] = model.booster_

    def _fit_defcon_dispersion(
        self, df: pd.DataFrame, pos: pd.Series, y: np.ndarray, e: np.ndarray
    ) -> None:
        """Per-position NB2 alpha by method of moments on in-sample DefCon predictions."""
        alphas: dict[str, float] = {}
        pooled_y: list[np.ndarray] = []
        pooled_mu: list[np.ndarray] = []
        for p in _GBM_TARGETS["defcon"]:
            mask = (pos == p).to_numpy() & ~np.isnan(y)
            if int(mask.sum()) < 30:
                continue
            lam = self._predict_target("defcon", df.loc[mask], pd.Series(p, index=df.index[mask]))
            mu = lam * e[mask]
            alphas[p] = nb_dispersion_mom(y[mask], mu)
            pooled_y.append(y[mask])
            pooled_mu.append(mu)
        pooled = (
            nb_dispersion_mom(np.concatenate(pooled_y), np.concatenate(pooled_mu))
            if pooled_y
            else _DEFAULT_DEFCON_DISP
        )
        self._defcon_disp = {
            p: alphas.get(p, pooled) for p in _GBM_TARGETS["defcon"]
        }
        self._defcon_disp["GKP"] = 0.0

    def _fit_cards(self, df: pd.DataFrame, pos: pd.Series, e: np.ndarray) -> None:
        """Empirical-Bayes per-player yellow/red per-90 rates with per-position Gamma priors."""
        work = pd.DataFrame(
            {
                "player_code": df["player_code"].to_numpy(),
                "position": pos.to_numpy(),
                "yellow": _num(df["yellow_cards"]),
                "red": _num(df["red_cards"]),
                "e": e,
            }
        )
        # Last observed position per player decides which prior applies.
        if {"season", "gw"}.issubset(df.columns):
            order = df[["season", "gw"]].apply(pd.to_numeric, errors="coerce")
            work = work.iloc[np.lexsort((order["gw"].to_numpy(), order["season"].to_numpy()))]
        per_player = work.groupby("player_code").agg(
            yellow=("yellow", "sum"),
            red=("red", "sum"),
            e=("e", "sum"),
            position=("position", "last"),
        )
        self._card_priors = {}
        for p in ("GKP", "DEF", "MID", "FWD"):
            grp = per_player[per_player["position"] == p]
            if grp.empty:
                grp = per_player  # global fallback for positions absent from training
            ay, by = _eb_gamma_prior(grp["yellow"].to_numpy(), grp["e"].to_numpy())
            ar, br = _eb_gamma_prior(grp["red"].to_numpy(), grp["e"].to_numpy())
            self._card_priors[p] = {"a_y": ay, "b_y": by, "a_r": ar, "b_r": br}
        prior = per_player["position"].map(lambda p: self._card_priors[p])
        per_player = per_player.assign(
            lam_yellow=[
                (row.yellow + pr["a_y"]) / (row.e + pr["b_y"])
                for row, pr in zip(per_player.itertuples(), prior, strict=True)
            ],
            lam_red=[
                (row.red + pr["a_r"]) / (row.e + pr["b_r"])
                for row, pr in zip(per_player.itertuples(), prior, strict=True)
            ],
        )
        self._card_players = per_player[["lam_yellow", "lam_red"]].copy()

    # ------------------------------------------------------------------ prediction

    def _predict_target(self, target: str, df: pd.DataFrame, pos: pd.Series) -> np.ndarray:
        """Per-90 rate for one target: GBM per position where trained, fallback otherwise."""
        global_fb = float(np.mean(list(self._fallback_rates[target].values()) or [1e-5]))
        out = np.array(
            pos.map(lambda p: self._fallback_rates[target].get(p, global_fb)),
            dtype="float64",
        )
        for (t, p), booster in self._boosters.items():
            if t != target:
                continue
            mask = (pos == p).to_numpy()
            if not mask.any():
                continue
            missing = [c for c in self._feature_cols if c not in df.columns]
            if missing:
                raise KeyError(f"predict frame is missing feature columns: {missing}")
            base = self._fallback_rates[target].get(p, global_fb)
            out[mask] = base * booster.predict(_matrix(df.loc[mask], self._feature_cols))
        lo, hi = _RATE_CLIPS[target]
        return np.clip(out, lo, hi)

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        """Per-90 intensities per row of ``features`` (ARCHITECTURE contract columns).

        Returns ``[keys..., lam_goal, lam_assist, lam_saves, lam_defcon, defcon_disp,
        p_yellow, p_red, lam_og]`` with the index of ``features`` preserved. ``lam_saves`` is 0
        for outfielders; ``lam_defcon``/``defcon_disp`` are 0 for GKP. The TeamModel fixture
        multiplier is *not* applied here (see module docstring).
        """
        if not self._fitted:
            raise RuntimeError("RatesModel.predict called before fit/load")
        for c in KEY_COLUMNS:
            if c not in features.columns:
                raise KeyError(f"predict frame is missing key column {c!r}")
        pos = _position_series(features)
        keys = list(KEY_COLUMNS) + [c for c in OPTIONAL_KEY_COLUMNS if c in features.columns]
        out = features[keys].copy()

        out["lam_goal"] = self._predict_target("goal", features, pos)
        out["lam_assist"] = self._predict_target("assist", features, pos)
        out["lam_saves"] = np.where(
            (pos == "GKP").to_numpy(), self._predict_target("saves", features, pos), 0.0
        )
        gkp = (pos == "GKP").to_numpy()
        out["lam_defcon"] = np.where(gkp, 0.0, self._predict_target("defcon", features, pos))
        out["defcon_disp"] = pos.map(
            lambda p: self._defcon_disp.get(p, _DEFAULT_DEFCON_DISP)
        ).to_numpy(dtype="float64")

        lam_y, lam_r = self._card_rates_for(features["player_code"], pos)
        out["p_yellow"] = 1.0 - np.exp(-lam_y)
        out["p_red"] = 1.0 - np.exp(-lam_r)
        global_og = float(np.mean(list(self._og_rates.values()) or [1e-5]))
        out["lam_og"] = pos.map(lambda p: self._og_rates.get(p, global_og)).to_numpy(
            dtype="float64"
        )
        return out

    def _card_rates_for(
        self, player_codes: pd.Series, pos: pd.Series
    ) -> tuple[np.ndarray, np.ndarray]:
        """Per-90 yellow/red rates: player posterior where seen in training, else prior mean."""
        prior_y = pos.map(
            lambda p: self._card_priors[p]["a_y"] / self._card_priors[p]["b_y"]
        ).to_numpy(dtype="float64")
        prior_r = pos.map(
            lambda p: self._card_priors[p]["a_r"] / self._card_priors[p]["b_r"]
        ).to_numpy(dtype="float64")
        assert self._card_players is not None
        lam_y = (
            player_codes.map(self._card_players["lam_yellow"])
            .astype("float64")
            .fillna(pd.Series(prior_y, index=player_codes.index))
            .to_numpy()
        )
        lam_r = (
            player_codes.map(self._card_players["lam_red"])
            .astype("float64")
            .fillna(pd.Series(prior_r, index=player_codes.index))
            .to_numpy()
        )
        return lam_y, lam_r

    # ------------------------------------------------------------------ evaluation

    def evaluate(self, holdout: pd.DataFrame) -> dict[str, Any]:
        """Poisson deviance per target + per-90 rate calibration by decile on a holdout frame.

        The holdout must carry the same label columns as :meth:`fit`. Only ``minutes > 0``
        rows are scored; DefCon is scored only where a count label resolves; saves only on
        GKP rows. Calibration entries are exposure-weighted (predicted mean rate vs observed
        events per 90) per decile of the predicted rate.
        """
        df = holdout[_num(holdout["minutes"]) > 0].copy()
        if df.empty:
            raise ValueError("no rows with minutes > 0 to evaluate on")
        pos = _position_series(df)
        e = _num(df["minutes"]) / 90.0
        pred = self.predict(df)
        results: dict[str, Any] = {}
        outfield = (pos != "GKP").to_numpy()
        dc = defcon_counts(df).to_numpy()
        specs: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {
            "goal": (_num(df["goals_scored"]), pred["lam_goal"].to_numpy(), outfield),
            "assist": (_num(df["assists"]), pred["lam_assist"].to_numpy(), outfield),
            "saves": (_num(df["saves"]), pred["lam_saves"].to_numpy(), ~outfield),
            "defcon": (dc, pred["lam_defcon"].to_numpy(), outfield & ~np.isnan(dc)),
            "yellow": (
                _num(df["yellow_cards"]),
                -np.log1p(-pred["p_yellow"].to_numpy()),
                np.ones(len(df), dtype=bool),
            ),
            "red": (
                _num(df["red_cards"]),
                -np.log1p(-pred["p_red"].to_numpy()),
                np.ones(len(df), dtype=bool),
            ),
        }
        for name, (y, lam, mask) in specs.items():
            if int(mask.sum()) == 0:
                continue
            mu = lam[mask] * e[mask]
            results[f"poisson_deviance_{name}"] = poisson_deviance(y[mask], mu)
            if name in _GBM_TARGETS:
                results[f"calibration_{name}"] = _calibration_by_decile(
                    lam[mask], y[mask], e[mask]
                )
        return results

    # ------------------------------------------------------------------ persistence

    def save(self, directory: Path) -> None:
        """Serialize to ``directory``: boosters as text, card table as parquet, meta as JSON."""
        if not self._fitted:
            raise RuntimeError("RatesModel.save called before fit")
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        for (target, position), booster in self._boosters.items():
            booster.save_model(str(directory / f"booster_{target}_{position}.txt"))
        assert self._card_players is not None
        self._card_players.reset_index().to_parquet(directory / "cards.parquet", index=False)
        meta = {
            "seed": self.seed,
            "lgbm_params": self.lgbm_params,
            "min_rows": self.min_rows,
            "min_events": self.min_events,
            "feature_cols": self._feature_cols,
            "boosters": sorted([t, p] for (t, p) in self._boosters),
            "fallback_rates": self._fallback_rates,
            "defcon_disp": self._defcon_disp,
            "card_priors": self._card_priors,
            "og_rates": self._og_rates,
        }
        (directory / "meta.json").write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, directory: Path) -> RatesModel:
        """Reload a model saved by :meth:`save`."""
        directory = Path(directory)
        meta = json.loads((directory / "meta.json").read_text())
        model = cls(
            seed=meta["seed"],
            lgbm_params=meta["lgbm_params"],
            min_rows=meta["min_rows"],
            min_events=meta["min_events"],
        )
        model._feature_cols = list(meta["feature_cols"])
        model._fallback_rates = meta["fallback_rates"]
        model._defcon_disp = meta["defcon_disp"]
        model._card_priors = meta["card_priors"]
        model._og_rates = meta["og_rates"]
        for target, position in meta["boosters"]:
            path = directory / f"booster_{target}_{position}.txt"
            model._boosters[(target, position)] = lgb.Booster(model_file=str(path))
        cards = pd.read_parquet(directory / "cards.parquet")
        model._card_players = cards.set_index("player_code")[["lam_yellow", "lam_red"]]
        model._fitted = True
        return model


def _calibration_by_decile(
    lam: np.ndarray, y: np.ndarray, e: np.ndarray
) -> list[dict[str, float]]:
    """Exposure-weighted calibration table over deciles of the predicted per-90 rate."""
    frame = pd.DataFrame({"lam": lam, "y": y, "e": e})
    # rank(method="first") makes qcut well-defined even with heavily tied predictions.
    n_bins = min(10, len(frame))
    frame["decile"] = pd.qcut(
        frame["lam"].rank(method="first"), n_bins, labels=False, duplicates="drop"
    )
    rows: list[dict[str, float]] = []
    for decile, grp in frame.groupby("decile"):
        total_e = float(grp["e"].sum())
        rows.append(
            {
                "decile": int(decile),
                "pred_rate": float(np.average(grp["lam"], weights=grp["e"])),
                "obs_rate": float(grp["y"].sum() / total_e) if total_e > 0 else float("nan"),
                "n": int(len(grp)),
            }
        )
    return rows
