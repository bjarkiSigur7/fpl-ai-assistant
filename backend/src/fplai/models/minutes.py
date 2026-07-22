"""Minutes model — the highest-leverage component (MODEL_DESIGN_INPUTS §2).

Predicts, per player-fixture, the FPL-relevant minutes-bucket distribution

    ``q0 = P(0 min)``, ``q1 = P(1-59 min)``, ``q2 = P(60+ min)``  (q0+q1+q2 = 1)

plus conditional expected minutes ``mu1 = E[min | 1-59]`` (in [1, 59]) and
``mu2 = E[min | 60+]`` (in [60, 90]).

Two implementations, one contract (ARCHITECTURE "Model layer contracts"):

* :class:`HeuristicMinutesModel` (v0) — availability-gated recent-start-share
  heuristic. No fit required; calibrated defaults by position x price band for
  players with no history.
* :class:`LgbMinutesModel` (v1) — two-stage LightGBM: Stage A P(start),
  Stage B P(60+ | start) + P(cameo | bench) + cameo/start minutes expectations,
  with isotonic probability calibration on a temporal validation split.

Both accept any DataFrame carrying ``f_*`` feature columns (the
``features/windows.py`` contract) plus key columns; training additionally needs
the ``minutes`` label (and optionally ``starts``, ``void_gw``). Because the
features module may not exist yet, :func:`build_minutes_feature_frame` builds a
minimal, strictly leakage-safe ``f_*`` frame directly from the canonical
``player_match.parquet`` table (every window feature is computed from matches
*strictly before* the row's own match via a per-player ``shift(1)``).

WC2026 hook (FPL_KNOWLEDGE §3.3): ``predict(..., fatigue_dampening={code: d})``
multiplies q2 by ``d`` in [0, 1], shifts the removed mass to q1/q0 (70/30 —
early hooks and bench cameos dominate over full rests) and renormalizes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.isotonic import IsotonicRegression

FEATURE_PREFIX = "f_"
KEY_COLUMNS = ("season", "gw", "player_code")
_EPS = 1e-4
_MODEL_FILE = "model.joblib"

#: P(plays 60+ | started), pooled 2022-25 seasons (label-cleaned; see module tests).
P60_GIVEN_START = 0.93

#: Fallback E[minutes | 60+] by position (pooled 2016-25 player_match data).
MU2_BY_POSITION = {"GKP": 89.9, "DEF": 88.3, "MID": 84.2, "FWD": 83.1}
MU2_DEFAULT = 85.5
#: Fallback E[minutes | 1-59] (cameo mean, stable across 3- and 5-sub regimes).
MU1_DEFAULT = 22.0

#: Price-band edges in 0.1m units -> band labels used by the defaults table.
_PRICE_BAND_EDGES = (45, 55, 70, 90)
_PRICE_BAND_LABELS = ("le45", "46_55", "56_70", "71_90", "gt90")

#: Calibrated (start_share, cameo_share) defaults for players with no history,
#: by (position, price band). Computed from player_match seasons 2022-2024.
DEFAULT_SHARES: dict[tuple[str, str], tuple[float, float]] = {
    ("GKP", "le45"): (0.13, 0.01),
    ("GKP", "46_55"): (0.66, 0.01),
    ("GKP", "56_70"): (0.77, 0.01),
    ("GKP", "71_90"): (0.77, 0.01),
    ("GKP", "gt90"): (0.77, 0.01),
    ("DEF", "le45"): (0.24, 0.08),
    ("DEF", "46_55"): (0.49, 0.11),
    ("DEF", "56_70"): (0.56, 0.10),
    ("DEF", "71_90"): (0.50, 0.13),
    ("DEF", "gt90"): (0.55, 0.11),
    ("MID", "le45"): (0.06, 0.09),
    ("MID", "46_55"): (0.32, 0.22),
    ("MID", "56_70"): (0.44, 0.23),
    ("MID", "71_90"): (0.60, 0.15),
    ("MID", "gt90"): (0.63, 0.12),
    ("FWD", "le45"): (0.03, 0.07),
    ("FWD", "46_55"): (0.19, 0.26),
    ("FWD", "56_70"): (0.37, 0.26),
    ("FWD", "71_90"): (0.44, 0.20),
    ("FWD", "gt90"): (0.48, 0.12),
}
_GLOBAL_DEFAULT_SHARES = (0.27, 0.13)

#: Bucket probabilities forced when a player is flagged out (status i/s/u).
_OUT_BUCKETS = (0.98, 0.015, 0.005)

#: Share of fatigue-removed q2 mass shifted to q1 (rest to q0): a fatigued
#: starter is more often early-hooked/bench-cameoed than fully rested.
_FATIGUE_SHIFT_TO_Q1 = 0.7


def price_band(price: float) -> str:
    """Map a price in 0.1m units to a defaults-table band label."""
    if np.isnan(price):
        return "46_55"
    for edge, label in zip(_PRICE_BAND_EDGES, _PRICE_BAND_LABELS, strict=False):
        if price <= edge:
            return label
    return _PRICE_BAND_LABELS[-1]


def _feature_columns(df: pd.DataFrame) -> list[str]:
    """Numeric ``f_*`` columns of ``df``, sorted for determinism."""
    cols = []
    for c in sorted(df.columns):
        if c.startswith(FEATURE_PREFIX) and pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def _first_present(df: pd.DataFrame, candidates: list[str]) -> pd.Series:
    """Row-wise coalesce over the candidate columns (all-NaN if none exist)."""
    out = pd.Series(np.nan, index=df.index, dtype=float)
    for c in candidates:
        if c in df.columns:
            out = out.fillna(pd.to_numeric(df[c], errors="coerce"))
    return out


def _started_label(df: pd.DataFrame) -> np.ndarray:
    """Boolean start label, cleaned against upstream ``starts`` glitches.

    Where ``starts`` is present it is OR-ed with ``minutes >= 60`` (a small
    fraction of upstream rows record 60+ minutes with ``starts == 0``, which is
    an ingestion artifact — genuine 60+ substitute appearances are rare); where
    ``starts`` is null (pre-2022 seasons) the ``minutes >= 60`` proxy is used.
    """
    minutes = pd.to_numeric(df["minutes"], errors="coerce").to_numpy(dtype=float)
    sixty = minutes >= 60
    if "starts" in df.columns:
        starts = pd.to_numeric(df["starts"], errors="coerce").to_numpy(dtype=float)
        return np.where(np.isnan(starts), sixty, (starts == 1) | sixty)
    return sixty


def _minutes_bucket(minutes: np.ndarray) -> np.ndarray:
    """Map raw minutes to bucket index 0 (=0), 1 (1-59), 2 (60+)."""
    return np.where(minutes <= 0, 0, np.where(minutes < 60, 1, 2)).astype(int)


class MinutesModelBase:
    """Shared lifecycle + output-shaping for minutes models.

    Subclasses implement :meth:`_predict_raw` returning arrays
    ``(q0, q1, q2, mu1, mu2)`` aligned with ``features``; this base class
    handles clipping, renormalization, the WC2026 fatigue hook, evaluation and
    save/load.
    """

    def fit(self, features: pd.DataFrame) -> "MinutesModelBase":
        """Fit the model (no-op by default). Returns ``self``."""
        return self

    def _predict_raw(
        self, features: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        raise NotImplementedError

    def predict(
        self,
        features: pd.DataFrame,
        fatigue_dampening: dict[int, float] | None = None,
    ) -> pd.DataFrame:
        """Predict the minutes-bucket distribution per row of ``features``.

        Args:
            features: frame with ``f_*`` feature columns plus the key columns
                ``season, gw, player_code`` (``fpl_fixture_id`` passed through
                when present).
            fatigue_dampening: optional ``{player_code: d}`` with ``d`` in
                [0, 1]; q2 is multiplied by ``d``, the removed mass is shifted
                to q1/q0 (70/30) and the row renormalized (WC2026 GW1-4 hook,
                FPL_KNOWLEDGE §3.3).

        Returns:
            DataFrame with keys plus ``q0, q1, q2, mu1, mu2``;
            ``q0+q1+q2 == 1`` per row, ``mu1`` in [1, 59], ``mu2`` in [60, 90].
        """
        q0, q1, q2, mu1, mu2 = self._predict_raw(features)
        q = np.clip(np.stack([q0, q1, q2], axis=1), _EPS, 1.0)

        if fatigue_dampening:
            codes = pd.to_numeric(features["player_code"], errors="coerce").to_numpy()
            damp = np.array(
                [float(fatigue_dampening.get(int(c), 1.0)) if not np.isnan(c) else 1.0
                 for c in codes]
            )
            if ((damp < 0) | (damp > 1)).any():
                raise ValueError("fatigue_dampening values must be in [0, 1]")
            removed = q[:, 2] * (1.0 - damp)
            q[:, 2] -= removed
            q[:, 1] += _FATIGUE_SHIFT_TO_Q1 * removed
            q[:, 0] += (1.0 - _FATIGUE_SHIFT_TO_Q1) * removed

        q = np.clip(q, _EPS, None)
        q /= q.sum(axis=1, keepdims=True)

        keys = [c for c in (*KEY_COLUMNS, "fpl_fixture_id") if c in features.columns]
        out = features[keys].copy().reset_index(drop=True)
        out["q0"], out["q1"], out["q2"] = q[:, 0], q[:, 1], q[:, 2]
        out["mu1"] = np.clip(mu1, 1.0, 59.0)
        out["mu2"] = np.clip(mu2, 60.0, 90.0)
        return out

    def evaluate(self, features: pd.DataFrame) -> dict[str, Any]:
        """Evaluate against realized ``minutes`` labels in ``features``.

        Returns a dict with ``n``, ``bucket_log_loss`` (mean negative log
        probability of the realized bucket), ``brier_p0`` (Brier score of q0
        vs the played-0-minutes indicator), and ``calibration`` — a DataFrame
        of realized vs predicted bucket frequencies by predicted-probability
        decile (columns: bucket, decile, n, p_pred_mean, p_realized).
        """
        labelled = features[pd.to_numeric(features["minutes"], errors="coerce").notna()]
        pred = self.predict(labelled)
        minutes = pd.to_numeric(labelled["minutes"], errors="coerce").to_numpy(dtype=float)
        bucket = _minutes_bucket(minutes)
        q = pred[["q0", "q1", "q2"]].to_numpy()

        p_true = q[np.arange(len(q)), bucket]
        log_loss = float(-np.log(np.clip(p_true, 1e-9, 1.0)).mean())
        brier_p0 = float(((q[:, 0] - (bucket == 0)) ** 2).mean())

        rows: list[dict[str, Any]] = []
        for b in (0, 1, 2):
            p_b = pd.Series(q[:, b])
            realized = pd.Series((bucket == b).astype(float))
            deciles = pd.qcut(p_b, 10, labels=False, duplicates="drop")
            grouped = pd.DataFrame({"decile": deciles, "p": p_b, "y": realized}).groupby(
                "decile", observed=True
            )
            for decile, grp in grouped:
                rows.append(
                    {
                        "bucket": f"q{b}",
                        "decile": int(decile),
                        "n": int(len(grp)),
                        "p_pred_mean": float(grp["p"].mean()),
                        "p_realized": float(grp["y"].mean()),
                    }
                )
        return {
            "n": int(len(q)),
            "bucket_log_loss": log_loss,
            "brier_p0": brier_p0,
            "calibration": pd.DataFrame(rows),
        }

    def save(self, directory: Path) -> None:
        """Serialize the model into ``directory`` (created if needed)."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, directory / _MODEL_FILE)

    @classmethod
    def load(cls, directory: Path) -> "MinutesModelBase":
        """Load a model previously written by :meth:`save`."""
        obj = joblib.load(Path(directory) / _MODEL_FILE)
        if not isinstance(obj, cls):
            raise TypeError(f"{directory} holds a {type(obj).__name__}, expected {cls.__name__}")
        return obj


class HeuristicMinutesModel(MinutesModelBase):
    """v0 minutes model: availability-gated recent start share (no fit needed).

    ``q2 ~= start_share x availability x P(60+|start)``; ``q1`` from the recent
    cameo share; players flagged out (``f_avail_is_out``) get ``q0 ~= 1``.
    Players without history fall back to calibrated defaults by
    position x price band (:data:`DEFAULT_SHARES`).
    """

    START_SHARE_COLS = ["f_start_share_5", "f_start_share_3", "f_start_share_10", "f_start_share_1"]
    CAMEO_SHARE_COLS = ["f_cameo_share_5", "f_cameo_share_3", "f_cameo_share_10"]
    MU1_COLS = ["f_cameo_minutes_mean_5", "f_cameo_minutes_mean_10"]
    MU2_COLS = ["f_start_minutes_mean_5", "f_start_minutes_mean_10"]

    def _default_shares(self, features: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Per-row (start_share, cameo_share) defaults by position + price band."""
        positions = (
            features["position"].astype(str)
            if "position" in features.columns
            else pd.Series("MID", index=features.index)
        )
        price = _first_present(features, ["f_price", "price"])
        start_d = np.empty(len(features))
        cameo_d = np.empty(len(features))
        for i, (pos, prc) in enumerate(zip(positions.to_numpy(), price.to_numpy(), strict=True)):
            start_d[i], cameo_d[i] = DEFAULT_SHARES.get(
                (pos, price_band(float(prc))), _GLOBAL_DEFAULT_SHARES
            )
        return start_d, cameo_d

    def _predict_raw(
        self, features: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        n = len(features)
        start_default, cameo_default = self._default_shares(features)
        start_share = _first_present(features, self.START_SHARE_COLS).to_numpy()
        cameo_share = _first_present(features, self.CAMEO_SHARE_COLS).to_numpy()
        start_share = np.where(np.isnan(start_share), start_default, start_share)
        cameo_share = np.where(np.isnan(cameo_share), cameo_default, cameo_share)
        start_share = np.clip(start_share, 0.0, 1.0)
        cameo_share = np.clip(cameo_share, 0.0, 1.0)

        chance = _first_present(features, ["f_avail_chance"]).to_numpy()
        is_doubt = _first_present(features, ["f_avail_is_doubt"]).fillna(0).to_numpy() > 0
        is_out = _first_present(features, ["f_avail_is_out"]).fillna(0).to_numpy() > 0
        avail = np.where(np.isnan(chance), np.where(is_doubt, 0.5, 1.0), chance)
        avail = np.clip(avail, 0.0, 1.0)

        q2 = np.clip(start_share * avail * P60_GIVEN_START, _EPS, 0.98)
        q1 = np.clip(avail * (start_share * (1.0 - P60_GIVEN_START) + cameo_share), _EPS, None)
        q1 = np.minimum(q1, 1.0 - q2 - _EPS)
        q0 = 1.0 - q1 - q2

        q0 = np.where(is_out, _OUT_BUCKETS[0], q0)
        q1 = np.where(is_out, _OUT_BUCKETS[1], q1)
        q2 = np.where(is_out, _OUT_BUCKETS[2], q2)

        mu1 = _first_present(features, self.MU1_COLS).fillna(MU1_DEFAULT).to_numpy()
        mu2_default = (
            features["position"].astype(str).map(MU2_BY_POSITION).fillna(MU2_DEFAULT).to_numpy()
            if "position" in features.columns
            else np.full(n, MU2_DEFAULT)
        )
        mu2 = _first_present(features, self.MU2_COLS).to_numpy()
        mu2 = np.where(np.isnan(mu2), mu2_default, mu2)
        return q0, q1, q2, mu1, mu2


class _ConstantProb:
    """Degenerate stand-in for a classifier trained on a single class."""

    def __init__(self, p: float) -> None:
        self.p = float(p)

    def predict(self, x: np.ndarray) -> np.ndarray:  # noqa: ARG002 - sklearn-like API
        return np.full(len(x), self.p)


class LgbMinutesModel(MinutesModelBase):
    """v1 minutes model: two-stage LightGBM with isotonic calibration.

    Stage A: P(start). Stage B: P(60+ | start); Stage C: P(cameo | bench).
    Combined: ``q2 = pA*pB``, ``q1 = pA*(1-pB) + (1-pA)*pC``,
    ``q0 = (1-pA)*(1-pC)``. Small LightGBM regressors give the cameo-minutes
    expectation (mu1) and started-minutes expectation (mu2).

    Calibration (when >= 2 seasons of training data): base classifiers are fit
    on all seasons strictly before the latest training season; isotonic
    calibrators are fit on that held-out latest season and frozen. Void GWs are
    excluded from all training. The 3-vs-5 subs regime enters through the
    ``f_subs_regime`` feature column.
    """

    def __init__(
        self,
        *,
        n_estimators: int = 400,
        learning_rate: float = 0.05,
        num_leaves: int = 63,
        min_child_samples: int = 40,
        random_state: int = 42,
        calibrate: bool = True,
    ) -> None:
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.min_child_samples = min_child_samples
        self.random_state = random_state
        self.calibrate = calibrate
        self.feature_names_: list[str] = []
        self.clf_start_: LGBMClassifier | _ConstantProb | None = None
        self.clf_p60_: LGBMClassifier | _ConstantProb | None = None
        self.clf_cameo_: LGBMClassifier | _ConstantProb | None = None
        self.cal_start_: IsotonicRegression | None = None
        self.cal_p60_: IsotonicRegression | None = None
        self.cal_cameo_: IsotonicRegression | None = None
        self.reg_mu1_: LGBMRegressor | None = None
        self.reg_mu2_: LGBMRegressor | None = None

    # -- internals ---------------------------------------------------------

    def _lgb_params(self) -> dict[str, Any]:
        return {
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "num_leaves": self.num_leaves,
            "min_child_samples": self.min_child_samples,
            "random_state": self.random_state,
            "deterministic": True,
            "force_row_wise": True,
            "verbose": -1,
        }

    def _fit_classifier(
        self, x: np.ndarray, y: np.ndarray
    ) -> LGBMClassifier | _ConstantProb:
        if len(np.unique(y)) < 2:
            return _ConstantProb(float(y.mean()) if len(y) else 0.5)
        clf = LGBMClassifier(**self._lgb_params())
        clf.fit(x, y)
        return clf

    @staticmethod
    def _proba(clf: LGBMClassifier | _ConstantProb, x: np.ndarray) -> np.ndarray:
        if isinstance(clf, _ConstantProb):
            return clf.predict(x)
        idx = int(np.where(clf.classes_ == 1)[0][0])
        return clf.predict_proba(x)[:, idx]

    @staticmethod
    def _fit_isotonic(p: np.ndarray, y: np.ndarray) -> IsotonicRegression | None:
        if len(p) < 100 or len(np.unique(y)) < 2:
            return None
        iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        iso.fit(p, y)
        return iso

    @staticmethod
    def _apply_cal(cal: IsotonicRegression | None, p: np.ndarray) -> np.ndarray:
        return p if cal is None else np.asarray(cal.predict(p))

    def _matrix(self, features: pd.DataFrame) -> np.ndarray:
        cols = pd.DataFrame(index=features.index)
        for c in self.feature_names_:
            cols[c] = (
                pd.to_numeric(features[c], errors="coerce")
                if c in features.columns
                else np.nan
            )
        return cols.to_numpy(dtype=float)

    # -- lifecycle ---------------------------------------------------------

    def fit(self, features: pd.DataFrame) -> "LgbMinutesModel":
        """Train on rows where the ``minutes`` label exists, excluding void GWs.

        ``features`` must carry numeric ``f_*`` columns, ``season`` and
        ``minutes``; ``starts`` (nullable) and ``void_gw`` are used when
        present.
        """
        df = features
        if "void_gw" in df.columns:
            df = df[~df["void_gw"].fillna(False).astype(bool)]
        df = df[pd.to_numeric(df["minutes"], errors="coerce").notna()]
        if df.empty:
            raise ValueError("no labelled rows to train on")

        self.feature_names_ = _feature_columns(df)
        if not self.feature_names_:
            raise ValueError("no f_* feature columns found")

        minutes = pd.to_numeric(df["minutes"], errors="coerce").to_numpy(dtype=float)
        started = _started_label(df)
        p60_y = (minutes >= 60).astype(int)
        cameo_y = (minutes > 0).astype(int)
        x_all = self._matrix(df)

        seasons = np.sort(pd.to_numeric(df["season"], errors="coerce").unique())
        use_cal = self.calibrate and len(seasons) >= 2
        if use_cal:
            cal_season = seasons[-1]
            base = (df["season"] < cal_season).to_numpy()
        else:
            base = np.ones(len(df), dtype=bool)

        xb, sb = x_all[base], started[base]
        self.clf_start_ = self._fit_classifier(xb, sb.astype(int))
        self.clf_p60_ = self._fit_classifier(xb[sb], p60_y[base][sb])
        self.clf_cameo_ = self._fit_classifier(xb[~sb], cameo_y[base][~sb])

        self.cal_start_ = self.cal_p60_ = self.cal_cameo_ = None
        if use_cal:
            hold = ~base
            xh, sh = x_all[hold], started[hold]
            self.cal_start_ = self._fit_isotonic(
                self._proba(self.clf_start_, xh), sh.astype(int)
            )
            self.cal_p60_ = self._fit_isotonic(
                self._proba(self.clf_p60_, xh[sh]), p60_y[hold][sh]
            )
            self.cal_cameo_ = self._fit_isotonic(
                self._proba(self.clf_cameo_, xh[~sh]), cameo_y[hold][~sh]
            )

        reg_params = {**self._lgb_params(), "n_estimators": 200}
        cameo_rows = (minutes > 0) & (minutes < 60)
        start60_rows = minutes >= 60
        self.reg_mu1_ = LGBMRegressor(**reg_params)
        if cameo_rows.any():
            self.reg_mu1_.fit(x_all[cameo_rows], minutes[cameo_rows])
        else:
            self.reg_mu1_ = None
        self.reg_mu2_ = LGBMRegressor(**reg_params)
        if start60_rows.any():
            self.reg_mu2_.fit(x_all[start60_rows], minutes[start60_rows])
        else:
            self.reg_mu2_ = None
        return self

    def _predict_raw(
        self, features: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self.clf_start_ is None:
            raise RuntimeError("LgbMinutesModel.predict called before fit")
        x = self._matrix(features)
        p_start = np.clip(
            self._apply_cal(self.cal_start_, self._proba(self.clf_start_, x)), _EPS, 1 - _EPS
        )
        p60 = np.clip(
            self._apply_cal(self.cal_p60_, self._proba(self.clf_p60_, x)), _EPS, 1 - _EPS
        )
        p_cameo = np.clip(
            self._apply_cal(self.cal_cameo_, self._proba(self.clf_cameo_, x)), _EPS, 1 - _EPS
        )
        q2 = p_start * p60
        q1 = p_start * (1.0 - p60) + (1.0 - p_start) * p_cameo
        q0 = (1.0 - p_start) * (1.0 - p_cameo)
        mu1 = (
            self.reg_mu1_.predict(x) if self.reg_mu1_ is not None else np.full(len(x), MU1_DEFAULT)
        )
        mu2 = (
            self.reg_mu2_.predict(x) if self.reg_mu2_ is not None else np.full(len(x), MU2_DEFAULT)
        )
        return q0, q1, q2, mu1, mu2


def build_minutes_feature_frame(
    player_match: pd.DataFrame, fixtures: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Build a minimal leakage-safe ``f_*`` feature frame from ``player_match``.

    Interim stand-in until ``features/windows.py`` lands: one row per input
    row (player-fixture), with every rolling feature computed from that
    player's matches *strictly before* the current one (per-player
    ``shift(1)`` before any window), so no label-time information leaks
    (MODEL_DESIGN_INPUTS §7.2). Availability columns are emitted as nulls —
    historical deadline-timestamped snapshots do not exist pre-2026.

    Args:
        player_match: the canonical ``player_match.parquet`` frame (or any
            subset with its columns).
        fixtures: optional ``fixtures.parquet`` frame; joins ``kickoff_utc``
            to compute a days-rest feature.

    Returns:
        Frame with key columns, ``position``/``price``/``void_gw`` metadata,
        label columns (``minutes``, ``starts``) and ``f_*`` features, sorted
        by (player_code, season, gw, fpl_fixture_id).
    """
    pm = player_match.copy()
    if fixtures is not None and "kickoff_utc" in fixtures.columns:
        pm = pm.merge(
            fixtures[["season", "fpl_fixture_id", "kickoff_utc"]],
            on=["season", "fpl_fixture_id"],
            how="left",
        )
    pm = pm.sort_values(["player_code", "season", "gw", "fpl_fixture_id"]).reset_index(drop=True)

    minutes = pd.to_numeric(pm["minutes"], errors="coerce").astype(float)
    started = pd.Series(_started_label(pm).astype(float), index=pm.index)
    cameo = ((minutes > 0) & (minutes < 60)).astype(float)
    played = (minutes > 0).astype(float)
    grp = pm["player_code"]

    def lagged(series: pd.Series) -> pd.Series:
        """Per-player one-step lag (strictly-prior value)."""
        return series.groupby(grp).shift(1)

    def roll_mean(series: pd.Series, window: int) -> pd.Series:
        """Per-player rolling mean over the previous ``window`` matches."""
        shifted = lagged(series)
        rolled = shifted.groupby(grp).rolling(window, min_periods=1).mean()
        return rolled.reset_index(level=0, drop=True)

    out = pm[
        [c for c in ("season", "gw", "fpl_fixture_id", "player_code", "position", "price",
                     "void_gw", "minutes", "starts") if c in pm.columns]
    ].copy()

    for w in (1, 3, 5, 10):
        out[f"f_start_share_{w}"] = roll_mean(started, w)
    for w in (3, 5, 10):
        out[f"f_cameo_share_{w}"] = roll_mean(cameo, w)
        out[f"f_minutes_mean_{w}"] = roll_mean(minutes, w)
    out["f_played_share_5"] = roll_mean(played, 5)
    out["f_minutes_last"] = lagged(minutes)

    start_minutes = minutes.where(started == 1.0)
    cameo_minutes = minutes.where(cameo == 1.0)
    out["f_start_minutes_mean_5"] = roll_mean(start_minutes, 5)
    out["f_cameo_minutes_mean_10"] = roll_mean(cameo_minutes, 10)

    # Consecutive prior matches with 0 minutes (injury/out-of-favour proxy).
    zero = (minutes <= 0).astype(int)
    reset_id = played.groupby(grp).cumsum()
    consec = zero.groupby([grp, reset_id]).cumsum()
    out["f_missed_streak"] = lagged(consec).fillna(0.0)

    if "kickoff_utc" in pm.columns:
        prev_kick = pm.groupby("player_code")["kickoff_utc"].shift(1)
        rest = (pm["kickoff_utc"] - prev_kick).dt.total_seconds() / 86400.0
        out["f_days_rest"] = rest.clip(lower=0.0, upper=60.0)

    out["f_gw_phase"] = pd.to_numeric(pm["gw"], errors="coerce") / 38.0
    out["f_price"] = pd.to_numeric(pm["price"], errors="coerce").astype(float)
    if "was_home" in pm.columns:
        out["f_was_home"] = pm["was_home"].astype(float)
    if "subs_regime" in pm.columns:
        out["f_subs_regime"] = pd.to_numeric(pm["subs_regime"], errors="coerce").astype(float)
    for pos in ("GKP", "DEF", "MID", "FWD"):
        out[f"f_pos_{pos}"] = (pm["position"].astype(str) == pos).astype(float)

    # Availability: no historical deadline snapshots — nullable per contract.
    out["f_avail_chance"] = np.nan
    out["f_avail_is_out"] = 0.0
    out["f_avail_is_doubt"] = 0.0
    return out
