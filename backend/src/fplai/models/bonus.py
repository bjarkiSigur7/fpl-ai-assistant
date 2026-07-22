"""Expected bonus points from expected BPS via rank-within-match simulation.

Two public entry points (ARCHITECTURE "Model layer contracts"):

* :func:`expected_bps` — map a per-player-fixture *event profile* (expected event
  counts) to expected BPS under a season's BPS matrix version.
* :func:`expected_bonus` — map expected BPS to E[bonus] via a seeded Monte Carlo
  over the within-fixture BPS ranking with the official tie rules
  (FPL_KNOWLEDGE §1.2: tie for 1st -> 3,3,1; tie for 2nd -> 3,2,2; tie for 3rd -> 3,2,1,1).

Event profile contract
----------------------
``event_profile`` is one row per player-fixture. Required columns:
``fpl_fixture_id`` (the rank-competition group), ``position`` (GKP/DEF/MID/FWD).
``season``/``gw``/``player_code`` are carried when present (grouping is per
``(season, gw, fpl_fixture_id)`` where available). All remaining columns are
*expected realized counts for the match* — already integrated over minutes — and
every one of them is optional (absent or NaN => 0):

``q1``, ``q2``                  minutes-bucket probabilities (1-59 / 60+)
``e_goals``, ``e_assists``      expected goals / assists
``e_cs``                        P(clean sheet awarded) — i.e. q2 x P(CS | on 60+)
``e_goals_conceded``            expected goals conceded while on pitch
``e_saves``, ``e_pen_saves``    GK shot saves / penalty saves
``e_pen_misses``                penalty misses
``e_cbi``, ``e_recoveries``, ``e_tackles``, ``e_tackled``   defensive counts
``e_yellow``, ``e_red``, ``e_own_goals``                    discipline / OG
``e_key_passes``, ``e_shots_on_target``, ``e_shots_off_target``  rich-era extras

BPS reconstruction and documented approximations
------------------------------------------------
The per-version BPS matrices are rebuilt from :data:`fplai.rules.BPS_2026` by
undoing :data:`fplai.rules.BPS_DELTAS` (v4 -> v3 -> v2 -> v1); values removed by
a delta are restored from the FPL_KNOWLEDGE §1.2/§2.1 documentation. Categories
we cannot expect (pass-completion bands, dribbles, crosses, fouls, big chances,
errors, offsides, match-winning goal, penalty-goal recategorisation) are absorbed
into an empirical per-position bias term in :class:`BonusCalibration` rather than
modelled from counts. Split of GK saves into in-box / big-chance categories uses
fixed share priors (:data:`IN_BOX_SAVE_SHARE`, :data:`BIG_CHANCE_SAVE_SHARE`).
Floor divisions (per 3 CBI etc.) are approximated by ``E[x] / divisor``.

Historical completeness: 2016-18 ("rich stats" era) and 2025+ profiles built
from realized data include CBI/tackles/recoveries; 2019-24 core-stats seasons
reconstruct from goals/assists/CS/concede/saves/cards/minutes only — the
per-position bias absorbs the (larger) missing mass there, so calibrate
per-era with :func:`calibrate` if precision on old seasons matters.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from fplai import rules

__all__ = [
    "BIG_CHANCE_SAVE_SHARE",
    "DEFAULT_CALIBRATION",
    "IN_BOX_SAVE_SHARE",
    "BonusCalibration",
    "bonus_validation_table",
    "bps_matrix",
    "calibrate",
    "event_profile_from_realized",
    "expected_bonus",
    "expected_bps",
]

#: Default number of Monte Carlo draws for the bonus rank simulation.
DEFAULT_N_DRAWS: int = 200

#: Default RNG seed — every stochastic step in this module is seeded.
DEFAULT_SEED: int = 2026

#: Prior share of GK saves made inside the box (drives the v3 in/out-box split and
#: the v4 "+1 in-box extra"). Community shot-location priors; not season-fitted.
IN_BOX_SAVE_SHARE: float = 0.70

#: Prior share of GK saves that are "big chance" saves (v4 "+1 big-chance extra").
BIG_CHANCE_SAVE_SHARE: float = 0.15


# --------------------------------------------------------------------------------------
# BPS matrix reconstruction per season
# --------------------------------------------------------------------------------------


def bps_matrix(season: int) -> dict[str, float]:
    """Return the BPS matrix in force in ``season``, keyed like :data:`rules.BPS_2026`.

    Reconstructed by undoing :data:`rules.BPS_DELTAS` from the v4 (2026-27) matrix.
    Values removed by a forward delta are restored from FPL_KNOWLEDGE §1.2/§2.1:
    "being tackled" -1 (v1-v3), +1 per 2 CBI (v1-v3), flat +2 GK save (v1-v2),
    GK penalty save 15 in v1 ("pen save 15->9" in the 2024-25 changelog row).

    Semantic caveats not expressible as matrix values: pre-v3 ``tackle_won`` was
    *net* of tackles lost; pre-v3 penalty goals scored the scorer's positional
    goal BPS (no separate ``penalty_goal`` category).
    """
    if season not in rules.SEASON_FLAGS:
        raise ValueError(f"unknown season {season!r}; expected one of {sorted(rules.SEASON_FLAGS)}")
    version = rules.SEASON_FLAGS[season].bps_version
    m: dict[str, float] = {k: float(v) for k, v in rules.BPS_2026.items()}
    if version == "v4":
        return m
    # v4 -> v3: undo the four official 2026-27 changes (rules.BPS_DELTAS["v4"]).
    del m["cbi_per_3"]
    m["cbi_per_2"] = 1.0  # +1 per 2 CBI through 2025-26 (FPL_KNOWLEDGE §1.2)
    m["tackled_against"] = -1.0  # "being tackled" -1 each, removed in v4
    del m["gk_save"], m["gk_save_inside_box_extra"], m["gk_big_chance_save_extra"]
    m["gk_save_inside_box"] = float(rules.BPS_DELTAS["v3"]["gk_save_inside_box"])  # type: ignore[arg-type]
    m["gk_save_outside_box"] = float(rules.BPS_DELTAS["v3"]["gk_save_outside_box"])  # type: ignore[arg-type]
    m["gk_penalty_save"] = float(rules.BPS_DELTAS["v3"]["gk_penalty_save"])  # type: ignore[arg-type]
    if version == "v3":
        return m
    # v3 -> v2: undo the 2025-26 changes (rules.BPS_DELTAS["v3"]).
    del m["penalty_goal"]  # pre-v3: penalty goals score the position's goal BPS
    del m["gk_save_inside_box"], m["gk_save_outside_box"]
    m["gk_save"] = 2.0  # flat per-save category pre-v3
    m["gk_penalty_save"] = float(rules.BPS_DELTAS["v2"]["gk_penalty_save"])  # type: ignore[arg-type]
    m["goal_line_clearance"] = float(rules.BPS_DELTAS["v2"]["goal_line_clearance"])  # type: ignore[arg-type]
    if version == "v2":
        return m
    # v2 -> v1: undo the 2024-25 changes (rules.BPS_DELTAS["v2"]).
    m["gk_penalty_save"] = 15.0  # "pen save 15->9" (FPL_KNOWLEDGE §2.1, 2024-25 row)
    for new_in_v2 in ("goal_conceded_gkp_def", "goal_line_clearance", "foul_won", "shot_on_target"):
        del m[new_in_v2]
    return m


# --------------------------------------------------------------------------------------
# Expected BPS from an event profile
# --------------------------------------------------------------------------------------


def _col(df: pd.DataFrame, name: str) -> np.ndarray:
    """Return column ``name`` as a float array, zeros when absent, NaN -> 0."""
    if name not in df.columns:
        return np.zeros(len(df))
    return np.nan_to_num(df[name].to_numpy(dtype=float, na_value=0.0), nan=0.0)


def expected_bps(event_profile: pd.DataFrame, season: int) -> pd.Series:
    """Return expected BPS per player-fixture row under ``season``'s BPS matrix.

    ``event_profile`` follows the module-level event-profile contract; missing
    count columns contribute zero. Goal BPS uses the positional category
    (penalty-goal and match-winning-goal refinements are absorbed by the
    calibration bias, see module docstring). Returns a float Series aligned to
    ``event_profile.index``, named ``"expected_bps"``.
    """
    if "position" not in event_profile.columns:
        raise ValueError("event_profile must have a 'position' column")
    m = bps_matrix(season)
    pos = event_profile["position"].astype(str).to_numpy()
    unknown = set(pos) - set(rules.POSITIONS)
    if unknown:
        raise ValueError(f"unknown positions in event_profile: {sorted(unknown)}")
    is_gk = pos == "GKP"
    is_gkdef = is_gk | (pos == "DEF")

    goal_bps = np.select(
        [pos == "MID", pos == "FWD"], [m["goal_mid"], m["goal_fwd"]], default=m["goal_gkp_def"]
    )
    # GK save value per expected save, per BPS version (see module docstring).
    if "gk_save_inside_box_extra" in m:  # v4
        save_val = (
            m["gk_save"]
            + IN_BOX_SAVE_SHARE * m["gk_save_inside_box_extra"]
            + BIG_CHANCE_SAVE_SHARE * m["gk_big_chance_save_extra"]
        )
        pen_save_val = m["gk_penalty_save"] + m["gk_big_chance_save_extra"]
    elif "gk_save_inside_box" in m:  # v3
        save_val = (
            IN_BOX_SAVE_SHARE * m["gk_save_inside_box"]
            + (1.0 - IN_BOX_SAVE_SHARE) * m["gk_save_outside_box"]
        )
        pen_save_val = m["gk_penalty_save"]
    else:  # v1 / v2
        save_val = m["gk_save"]
        pen_save_val = m["gk_penalty_save"]
    cbi_divisor, cbi_val = (3.0, m["cbi_per_3"]) if "cbi_per_3" in m else (2.0, m["cbi_per_2"])

    ebps = (
        _col(event_profile, "q1") * m["minutes_1_60"]
        + _col(event_profile, "q2") * m["minutes_over_60"]
        + _col(event_profile, "e_goals") * goal_bps
        + _col(event_profile, "e_assists") * m["assist"]
        + _col(event_profile, "e_cs") * m["clean_sheet_gkp_def"] * is_gkdef
        + _col(event_profile, "e_goals_conceded") * m.get("goal_conceded_gkp_def", 0.0) * is_gkdef
        + _col(event_profile, "e_saves") * save_val * is_gk
        + _col(event_profile, "e_pen_saves") * pen_save_val * is_gk
        + _col(event_profile, "e_pen_misses") * m["penalty_miss"]
        + _col(event_profile, "e_cbi") / cbi_divisor * cbi_val
        + _col(event_profile, "e_recoveries") / 3.0 * m["recoveries_per_3"]
        + _col(event_profile, "e_tackles") * m["tackle_won"]
        + _col(event_profile, "e_tackled") * m.get("tackled_against", 0.0)
        + _col(event_profile, "e_yellow") * m["yellow_card"]
        + _col(event_profile, "e_red") * m["red_card"]
        + _col(event_profile, "e_own_goals") * m["own_goal"]
        + _col(event_profile, "e_key_passes") * m["key_pass"]
        + _col(event_profile, "e_shots_on_target") * m.get("shot_on_target", 0.0)
        + _col(event_profile, "e_shots_off_target") * m["shot_off_target"]
    )
    return pd.Series(ebps, index=event_profile.index, name="expected_bps")


# --------------------------------------------------------------------------------------
# Calibration: EBPS bias + residual noise scale, fitted on realized data
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class BonusCalibration:
    """Empirical mapping from reconstructed expected BPS to the realized-BPS scale.

    ``bias`` is the per-position mean of ``realized_bps - expected_bps`` (absorbs
    BPS categories our profiles cannot expect: pass completion, dribbles, fouls,
    key passes pre-rich-era, errors, ...). ``sigma = clip(sigma_intercept +
    sigma_slope * center, sigma_floor, None)`` is the residual noise scale used
    for the Normal draws in :func:`expected_bonus`, with ``center = ebps + bias``.
    """

    bias: dict[str, float] = field(default_factory=dict)
    sigma_intercept: float = 5.0
    sigma_slope: float = 0.15
    sigma_floor: float = 2.0

    def center(self, ebps: np.ndarray, position: np.ndarray) -> np.ndarray:
        """Bias-corrected expected BPS per row."""
        bias = np.array([self.bias.get(p, 0.0) for p in position])
        return ebps + bias

    def sigma(self, center: np.ndarray) -> np.ndarray:
        """Residual noise scale per row (floored)."""
        return np.clip(self.sigma_intercept + self.sigma_slope * center, self.sigma_floor, None)


#: Fitted by :func:`calibrate` on the full 2025-26 season of realized player_match
#: data (profiles via :func:`event_profile_from_realized`, minutes > 0 rows,
#: n=11,492). Positive DEF/MID bias reflects unmodelled pass-completion bands and
#: key passes/crosses; negative FWD bias reflects unmodelled negative categories
#: (shots off target, offsides, fouls, big chances missed). Sigma grows mildly
#: with expected BPS. Refit with :func:`calibrate` for other stat eras.
DEFAULT_CALIBRATION: BonusCalibration = BonusCalibration(
    bias={"GKP": -0.043, "DEF": 1.341, "MID": 0.608, "FWD": -1.773},
    sigma_intercept=3.111,
    sigma_slope=0.083,
    sigma_floor=2.0,
)


def event_profile_from_realized(player_match: pd.DataFrame) -> pd.DataFrame:
    """Build a perfect-foresight event profile from realized ``player_match`` rows.

    Used for calibration/validation of the BPS->bonus mapping (isolates the
    mapping from event-forecast error). Missing stat columns (e.g. CBI/tackles/
    recoveries in 2019-24) become zeros and are absorbed by the calibration bias.
    """
    out = pd.DataFrame(index=player_match.index)
    for key in ("season", "gw", "player_code", "fpl_fixture_id", "position"):
        if key in player_match.columns:
            out[key] = player_match[key]
    minutes = _col(player_match, "minutes")
    out["q1"] = ((minutes > 0) & (minutes < 60)).astype(float)
    out["q2"] = (minutes >= 60).astype(float)
    realized_map = {
        "e_goals": "goals_scored",
        "e_assists": "assists",
        "e_cs": "clean_sheets",
        "e_goals_conceded": "goals_conceded",
        "e_saves": "saves",
        "e_pen_saves": "penalties_saved",
        "e_pen_misses": "penalties_missed",
        "e_cbi": "clearances_blocks_interceptions",
        "e_recoveries": "recoveries",
        "e_tackles": "tackles",
        "e_yellow": "yellow_cards",
        "e_red": "red_cards",
        "e_own_goals": "own_goals",
    }
    for profile_col, source_col in realized_map.items():
        out[profile_col] = _col(player_match, source_col)
    return out


def calibrate(
    player_match: pd.DataFrame, season: int, *, n_bins: int = 10, sigma_floor: float = 2.0
) -> BonusCalibration:
    """Fit a :class:`BonusCalibration` from realized BPS residuals for ``season``.

    Rows with ``minutes > 0`` from ``player_match`` (canonical schema) are turned
    into perfect-foresight profiles; residuals ``bps - expected_bps`` give the
    per-position bias, and the bias-corrected residual standard deviation is
    fitted linearly across ``n_bins`` quantile bins of centered EBPS.
    """
    d = player_match[(player_match["season"] == season) & (player_match["minutes"] > 0)]
    if len(d) < 100:
        raise ValueError(f"not enough season={season} rows to calibrate ({len(d)})")
    profile = event_profile_from_realized(d)
    ebps = expected_bps(profile, season).to_numpy()
    bps = d["bps"].to_numpy(dtype=float)
    pos = d["position"].astype(str).to_numpy()
    resid = bps - ebps
    bias = {p: float(resid[pos == p].mean()) for p in rules.POSITIONS if (pos == p).any()}
    center = ebps + np.array([bias.get(p, 0.0) for p in pos])
    centered_resid = bps - center
    # Quantile-binned residual std, then a least-squares line through the bins.
    qs = np.quantile(center, np.linspace(0, 1, n_bins + 1))
    qs[-1] += 1e-9
    mids, stds = [], []
    for lo, hi in zip(qs[:-1], qs[1:], strict=True):
        mask = (center >= lo) & (center < hi)
        if mask.sum() >= 20:
            mids.append(float(center[mask].mean()))
            stds.append(float(centered_resid[mask].std()))
    slope, intercept = np.polyfit(np.array(mids), np.array(stds), 1)
    return BonusCalibration(
        bias=bias,
        sigma_intercept=float(intercept),
        sigma_slope=float(slope),
        sigma_floor=sigma_floor,
    )


# --------------------------------------------------------------------------------------
# E[bonus] via Monte Carlo over the within-fixture BPS ranking
# --------------------------------------------------------------------------------------


def _tie_rule_bonus(draws: np.ndarray) -> np.ndarray:
    """Bonus points per (player, draw) from integer BPS draws of one fixture.

    Official tie rules reduce to: a player's bonus depends only on how many
    players in the match have strictly higher BPS — 0 higher -> 3, 1 -> 2,
    2 -> 1, else 0 (reproduces 3,3,1 / 3,2,2 / 3,2,1,1 exactly).
    """
    higher = (draws[None, :, :] > draws[:, None, :]).sum(axis=1)
    return np.select([higher == 0, higher == 1, higher == 2], [3, 2, 1], default=0)


def expected_bonus(
    event_profile: pd.DataFrame,
    season: int,
    *,
    n_draws: int = DEFAULT_N_DRAWS,
    seed: int = DEFAULT_SEED,
    calibration: BonusCalibration | None = None,
) -> pd.Series:
    """Return E[bonus] per player-fixture via the rank-within-match Monte Carlo.

    Per fixture: draw ``n_draws`` integer BPS samples per player from
    ``Normal(center, sigma)`` (center/sigma from ``calibration``, default
    :data:`DEFAULT_CALIBRATION`), award 3/2/1 per draw under the official tie
    rules, and average. Rows that cannot play (``q1 + q2 == 0`` when the q
    columns are present) are excluded from the competition and get 0. The
    simulation is deterministic for a given ``seed`` and row order (fixtures
    are processed in sorted key order).
    """
    if "fpl_fixture_id" not in event_profile.columns:
        raise ValueError("event_profile must have an 'fpl_fixture_id' column")
    cal = calibration if calibration is not None else DEFAULT_CALIBRATION
    result = pd.Series(0.0, index=event_profile.index, name="expected_bonus")
    if event_profile.empty:
        return result

    ebps = expected_bps(event_profile, season).to_numpy()
    pos = event_profile["position"].astype(str).to_numpy()
    center_all = cal.center(ebps, pos)
    sigma_all = cal.sigma(center_all)
    if {"q1", "q2"} <= set(event_profile.columns):
        can_play = (_col(event_profile, "q1") + _col(event_profile, "q2")) > 1e-9
    else:
        can_play = np.ones(len(event_profile), dtype=bool)

    group_keys = [k for k in ("season", "gw", "fpl_fixture_id") if k in event_profile.columns]
    rng = np.random.default_rng(seed)
    groups = event_profile.groupby(group_keys, sort=True).indices
    for key in sorted(groups):
        idx = np.asarray(groups[key])
        idx = idx[can_play[idx]]
        if idx.size == 0:
            continue
        center = center_all[idx]
        sigma = sigma_all[idx]
        draws = rng.normal(center[:, None], sigma[:, None], size=(idx.size, n_draws))
        bonus_pts = _tie_rule_bonus(np.rint(draws))
        result.iloc[idx] = bonus_pts.mean(axis=1)
    return result


# --------------------------------------------------------------------------------------
# Validation: predicted vs actual bonus by predicted decile
# --------------------------------------------------------------------------------------


def bonus_validation_table(
    player_match: pd.DataFrame,
    season: int,
    *,
    n_draws: int = DEFAULT_N_DRAWS,
    seed: int = DEFAULT_SEED,
    calibration: BonusCalibration | None = None,
) -> pd.DataFrame:
    """Predicted-vs-actual bonus by predicted decile on realized ``season`` data.

    Builds perfect-foresight profiles from played rows, runs the full rank
    simulation, and returns one row per decile of predicted E[bonus]
    (``decile``, ``n``, ``mean_predicted``, ``mean_actual``) — a direct check
    of the EBPS -> bonus mapping, holding event forecasts at ground truth.
    """
    d = player_match[(player_match["season"] == season) & (player_match["minutes"] > 0)]
    if d.empty:
        raise ValueError(f"no season={season} played rows in player_match")
    profile = event_profile_from_realized(d)
    pred = expected_bonus(profile, season, n_draws=n_draws, seed=seed, calibration=calibration)
    frame = pd.DataFrame({"predicted": pred.to_numpy(), "actual": d["bonus"].to_numpy(dtype=float)})
    frame["decile"] = pd.qcut(frame["predicted"].rank(method="first"), 10, labels=False) + 1
    table = (
        frame.groupby("decile")
        .agg(
            n=("actual", "size"),
            mean_predicted=("predicted", "mean"),
            mean_actual=("actual", "mean"),
        )
        .reset_index()
    )
    return table
