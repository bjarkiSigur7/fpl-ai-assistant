"""xP assembly: combine minutes/team/rates model outputs into per-fixture expected points.

Implements FPL_KNOWLEDGE §1.1 exactly via :func:`fplai.rules.get_scoring` (no
hardcoded point values) following the MODEL_DESIGN_INPUTS §1.1 decomposition:

``xp = appearance + goals + assists + cs + concede + saves + defcon + bonus + cards + other``

* appearance   ``1*q1 + 2*q2`` (per-position values from the scoring table)
* attacking    ``pts(pos) * lam * (E[min]/90) * F_att`` where the fixture
  multiplier ``F_att = lam_team(fixture) / mean(lam_team over the team's
  fixtures present in the input)`` re-anchors the per-90 rates (which are
  fitted vs an average opponent) to this fixture's expected team goals.
* clean sheet  ``pts_CS(pos) * q2 * P(CS)`` with P(CS) from the team model
  (approximates P(CS | on for 60+) by the full-match clean-sheet probability).
* conceded     ``pts_concede(pos) * E[floor(C/2)]`` over a Poisson grid with
  ``C ~ Poisson(lam_opponent * E[min]/90)`` (GKP/DEF only via the table).
* saves        ``pts_save(pos) * E[floor(S/3)]`` over a Poisson grid with
  ``S ~ Poisson(lam_saves * E[min]/90)`` (lam_saves used directly per contract).
* defcon       ``pts_defcon(pos) * P(NB(lam_defcon * E[min]/90, defcon_disp) >=
  threshold(pos))`` with thresholds from :data:`fplai.rules.DEFCON_THRESHOLDS`;
  zero for GKP and for seasons without DefCon scoring.
* bonus        :func:`fplai.models.bonus.expected_bonus` on an event profile
  built from the same expectations (seeded Monte Carlo rank simulation).
* cards        yellow/red EVs from per-match probabilities scaled by P(plays).
* other        own-goal EV (``lam_og * E[min]/90``).

Captain/vice multipliers are NOT applied here — that is the optimizer's job.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd
from scipy import stats

from fplai import rules
from fplai.models import bonus as bonus_model

__all__ = [
    "COMPONENT_COLUMNS",
    "DEFCON_STAT_SHARES",
    "XP_COLUMNS",
    "aggregate_gw",
    "assemble_xp",
]

#: Component columns; they sum exactly to ``xp`` (tested invariant).
COMPONENT_COLUMNS: Final[tuple[str, ...]] = (
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

#: Full output schema of :func:`assemble_xp` (the ARCHITECTURE contract).
XP_COLUMNS: Final[tuple[str, ...]] = (
    "season",
    "gw",
    "player_code",
    "fpl_fixture_id",
    "xp",
    *COMPONENT_COLUMNS,
)

#: Split of ``lam_defcon`` (the position's threshold stat: CBIT for DEF, CBIRT
#: for MID/FWD) into (cbi, tackles, recoveries) expected counts for the bonus
#: event profile. Measured on 2025-26 realized data, 60+ minute rows. For DEF
#: the recoveries entry is *additional to* lam_defcon (CBIT excludes
#: recoveries); for MID/FWD/GKP the three shares partition CBIRT.
DEFCON_STAT_SHARES: Final[dict[str, tuple[float, float, float]]] = {
    "GKP": (0.13, 0.00, 0.87),
    "DEF": (0.78, 0.22, 0.48),
    "MID": (0.27, 0.21, 0.52),
    "FWD": (0.36, 0.16, 0.48),
}

_FEATURE_COLUMNS_REQUIRED: Final[tuple[str, ...]] = (
    "season",
    "gw",
    "player_code",
    "fpl_fixture_id",
    "team_code",
    "was_home",
    "position",
)
_MINUTES_COLUMNS: Final[tuple[str, ...]] = ("q0", "q1", "q2", "mu1", "mu2")
_TEAM_COLUMNS: Final[tuple[str, ...]] = (
    "season",
    "fpl_fixture_id",
    "home_lambda",
    "away_lambda",
    "p_cs_home",
    "p_cs_away",
)
_RATES_COLUMNS: Final[tuple[str, ...]] = (
    "lam_goal",
    "lam_assist",
    "lam_saves",
    "lam_defcon",
    "defcon_disp",
    "p_yellow",
    "p_red",
    "lam_og",
)
_PLAYER_KEYS: Final[tuple[str, ...]] = ("season", "gw", "player_code", "fpl_fixture_id")


def _require(df: pd.DataFrame, columns: tuple[str, ...], name: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"{name} frame is missing required columns: {missing}")


def _merge_player_frame(
    base: pd.DataFrame, other: pd.DataFrame, value_columns: tuple[str, ...], name: str
) -> pd.DataFrame:
    """Left-join a player-level prediction frame on its available key columns.

    Uses the intersection of ``_PLAYER_KEYS`` with ``other``'s columns, so both
    GW-level frames (``season, gw, player_code`` — one row reused across a DGW's
    fixtures) and match-level frames (``... , fpl_fixture_id``) join correctly.
    Raises if any base row fails to match (leakage-safe: no silent zero-filling).
    """
    keys = [k for k in _PLAYER_KEYS if k in other.columns]
    if not {"season", "gw", "player_code"} <= set(keys):
        raise ValueError(f"{name} frame must carry season, gw, player_code keys")
    _require(other, value_columns, name)
    merged = base.merge(other[[*keys, *value_columns]], on=keys, how="left", validate="many_to_one")
    unmatched = int(merged[value_columns[0]].isna().sum())
    if unmatched:
        raise ValueError(f"{unmatched} feature rows have no matching {name} prediction row")
    return merged


def _poisson_floor_ev(lam: np.ndarray, divisor: int, grid_max: int = 30) -> np.ndarray:
    """E[floor(X / divisor)] for X ~ Poisson(lam), on a truncated grid 0..grid_max."""
    lam = np.clip(np.nan_to_num(lam, nan=0.0), 0.0, None)
    ks = np.arange(grid_max + 1)
    pmf = stats.poisson.pmf(ks[None, :], lam[:, None])
    return (pmf * (ks // divisor)[None, :]).sum(axis=1)


def _nb_tail_prob(mean: np.ndarray, disp: np.ndarray, threshold: np.ndarray) -> np.ndarray:
    """P(X >= threshold) for X ~ NegBinom(mean, dispersion) per row.

    Parameterisation: dispersion r is the NB size (``Var = mean + mean^2 / r``),
    i.e. ``X ~ nbinom(n=r, p=r / (r + mean))``. Rows with missing/non-finite or
    non-positive dispersion fall back to the Poisson limit (r -> inf). Rows with
    ``threshold <= 0`` return 0 (used to disable ineligible positions).
    """
    mean = np.clip(np.nan_to_num(mean, nan=0.0), 1e-12, None)
    threshold = np.nan_to_num(threshold, nan=0.0)
    out = np.zeros(len(mean))
    eligible = threshold > 0
    if not eligible.any():
        return out
    disp = np.asarray(disp, dtype=float)
    nb = eligible & np.isfinite(disp) & (disp > 0)
    po = eligible & ~nb
    if nb.any():
        r = disp[nb]
        out[nb] = stats.nbinom.sf(threshold[nb] - 1, r, r / (r + mean[nb]))
    if po.any():
        out[po] = stats.poisson.sf(threshold[po] - 1, mean[po])
    return out


def _pos_points(scoring: dict[str, dict[str, int]], action: str, position: pd.Series) -> np.ndarray:
    """Per-row point value for ``action`` looked up by position from the scoring table."""
    return position.map(scoring[action]).to_numpy(dtype=float)


def assemble_xp(
    minutes: pd.DataFrame,
    team: pd.DataFrame,
    rates: pd.DataFrame,
    features: pd.DataFrame,
    season: int,
    *,
    n_draws: int = bonus_model.DEFAULT_N_DRAWS,
    seed: int = bonus_model.DEFAULT_SEED,
    bonus_calibration: bonus_model.BonusCalibration | None = None,
) -> pd.DataFrame:
    """Assemble per-player-fixture expected points from the component model outputs.

    Parameters
    ----------
    minutes:
        ``MinutesModel.predict`` output: keys + ``q0, q1, q2, mu1, mu2``.
    team:
        ``TeamModel.predict_fixtures`` output: ``season, fpl_fixture_id,
        home_lambda, away_lambda, p_cs_home, p_cs_away, ...``. Pass a frame
        spanning several GWs where possible — the fixture attack multiplier
        ``F_att`` is anchored on each team's average lambda over the fixtures
        present here (a single-fixture frame degrades gracefully to
        ``F_att = 1``, i.e. unadjusted rates).
    rates:
        ``RatesModel.predict`` output: keys + per-90 ``lam_goal, lam_assist,
        lam_saves, lam_defcon, defcon_disp, p_yellow, p_red, lam_og``
        (``p_yellow``/``p_red`` are per-match probabilities given play).
    features:
        Match-level feature frame; only the identity columns
        ``season, gw, player_code, fpl_fixture_id, team_code, was_home,
        position`` are used here. The bonus simulation ranks players *within
        each fixture among the rows provided* — pass the complete player pool
        for every fixture (both squads), otherwise E[bonus] is overstated
        (with k < 4 candidates everyone lands in the top 3 by construction).
    season:
        Season start year; selects the scoring table and BPS version. All
        input rows must belong to this season.

    Returns a DataFrame with exactly :data:`XP_COLUMNS`, one row per input
    feature row; component columns sum to ``xp``. Deterministic for a given
    ``seed``. Captain/vice multipliers are not applied.
    """
    _require(features, _FEATURE_COLUMNS_REQUIRED, "features")
    _require(team, _TEAM_COLUMNS, "team")
    if len(features) and not (features["season"] == season).all():
        raise ValueError(f"features frame contains rows outside season={season}")

    base = features.loc[:, list(_FEATURE_COLUMNS_REQUIRED)].reset_index(drop=True)
    base = _merge_player_frame(base, minutes, _MINUTES_COLUMNS, "minutes")
    base = _merge_player_frame(base, rates, _RATES_COLUMNS, "rates")
    team_cols = [c for c in _TEAM_COLUMNS if c != "season"]
    merged = base.merge(
        team[["season", *team_cols]].drop_duplicates(subset=["season", "fpl_fixture_id"]),
        on=["season", "fpl_fixture_id"],
        how="left",
        validate="many_to_one",
    )
    if int(merged["home_lambda"].isna().sum()):
        n = int(merged["home_lambda"].isna().sum())
        raise ValueError(f"{n} feature rows have no matching team prediction row")

    scoring = rules.get_scoring(season)
    pos = merged["position"].astype(str)
    unknown = set(pos.unique()) - set(rules.POSITIONS)
    if unknown:
        raise ValueError(f"unknown positions in features: {sorted(unknown)}")
    was_home = merged["was_home"].to_numpy(dtype=bool)
    q1 = merged["q1"].to_numpy(dtype=float)
    q2 = merged["q2"].to_numpy(dtype=float)
    p_play = q1 + q2
    e_min = q1 * merged["mu1"].to_numpy(dtype=float) + q2 * merged["mu2"].to_numpy(dtype=float)
    m_share = e_min / 90.0

    # Team-side lambdas and clean-sheet probability for this player's team.
    lam_team = np.where(was_home, merged["home_lambda"], merged["away_lambda"]).astype(float)
    lam_opp = np.where(was_home, merged["away_lambda"], merged["home_lambda"]).astype(float)
    p_cs = np.where(was_home, merged["p_cs_home"], merged["p_cs_away"]).astype(float)

    # Fixture attack multiplier: this fixture's expected team goals over the
    # team's average across the fixtures present in the joined frame.
    sides = pd.DataFrame(
        {
            "fpl_fixture_id": merged["fpl_fixture_id"],
            "team_code": merged["team_code"],
            "lam_team": lam_team,
        }
    ).drop_duplicates(subset=["fpl_fixture_id", "team_code"])
    team_avg = sides.groupby("team_code")["lam_team"].mean()
    avg_lam = merged["team_code"].map(team_avg).to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        f_att = np.where(avg_lam > 0, lam_team / avg_lam, 1.0)

    # --- components (FPL_KNOWLEDGE §1.1, per-unit values from the scoring table) ---
    xp_appearance = q1 * _pos_points(scoring, "short_play", pos) + q2 * _pos_points(
        scoring, "long_play", pos
    )

    lam_goal = np.nan_to_num(merged["lam_goal"].to_numpy(dtype=float), nan=0.0)
    lam_assist = np.nan_to_num(merged["lam_assist"].to_numpy(dtype=float), nan=0.0)
    e_goals = lam_goal * m_share * f_att
    e_assists = lam_assist * m_share * f_att
    xp_goals = _pos_points(scoring, "goals_scored", pos) * e_goals
    xp_assists = _pos_points(scoring, "assists", pos) * e_assists

    xp_cs = _pos_points(scoring, "clean_sheets", pos) * q2 * p_cs

    # Conceded: -1 per floor(C/2) for GKP/DEF; C over the on-pitch Poisson grid.
    e_conceded_on = lam_opp * m_share
    e_floor_c = _poisson_floor_ev(e_conceded_on, rules.CONCEDED_PER_POINT, grid_max=15)
    xp_concede = _pos_points(scoring, "goals_conceded", pos) * e_floor_c

    # Saves: +1 per floor(S/3), GKP only via the table; lam_saves used directly.
    lam_saves = np.nan_to_num(merged["lam_saves"].to_numpy(dtype=float), nan=0.0)
    e_saves_on = lam_saves * m_share
    e_floor_s = _poisson_floor_ev(e_saves_on, rules.SAVES_PER_POINT, grid_max=30)
    xp_saves = _pos_points(scoring, "saves", pos) * e_floor_s

    # DefCon: threshold on NB(lam_defcon scaled by minutes, defcon_disp). Only
    # from 2025-26 (scoring table carries the action) and never for GKP.
    lam_defcon = np.nan_to_num(merged["lam_defcon"].to_numpy(dtype=float), nan=0.0)
    if "defensive_contribution" in scoring:
        thresholds = pos.map(rules.DEFCON_THRESHOLDS).to_numpy(dtype=float)
        p_defcon = _nb_tail_prob(
            lam_defcon * m_share, merged["defcon_disp"].to_numpy(dtype=float), thresholds
        )
        xp_defcon = _pos_points(scoring, "defensive_contribution", pos) * p_defcon
    else:
        xp_defcon = np.zeros(len(merged))

    # Cards and own goals: per-match probabilities/rates scaled by participation.
    e_yellow = np.nan_to_num(merged["p_yellow"].to_numpy(dtype=float), nan=0.0) * p_play
    e_red = np.nan_to_num(merged["p_red"].to_numpy(dtype=float), nan=0.0) * p_play
    e_og = np.nan_to_num(merged["lam_og"].to_numpy(dtype=float), nan=0.0) * m_share
    xp_cards = (
        _pos_points(scoring, "yellow_cards", pos) * e_yellow
        + _pos_points(scoring, "red_cards", pos) * e_red
    )
    xp_other = _pos_points(scoring, "own_goals", pos) * e_og

    # Bonus: rank-within-match Monte Carlo on the expected event profile.
    shares = pos.map(DEFCON_STAT_SHARES)
    share_cbi = shares.map(lambda s: s[0]).to_numpy(dtype=float)
    share_tackles = shares.map(lambda s: s[1]).to_numpy(dtype=float)
    share_rec = shares.map(lambda s: s[2]).to_numpy(dtype=float)
    defcon_on = lam_defcon * m_share
    profile = pd.DataFrame(
        {
            "season": merged["season"],
            "gw": merged["gw"],
            "player_code": merged["player_code"],
            "fpl_fixture_id": merged["fpl_fixture_id"],
            "position": pos,
            "q1": q1,
            "q2": q2,
            "e_goals": e_goals,
            "e_assists": e_assists,
            "e_cs": q2 * p_cs,
            "e_goals_conceded": e_conceded_on,
            "e_saves": e_saves_on,
            "e_cbi": defcon_on * share_cbi,
            "e_tackles": defcon_on * share_tackles,
            "e_recoveries": defcon_on * share_rec,
            "e_yellow": e_yellow,
            "e_red": e_red,
            "e_own_goals": e_og,
        }
    )
    xp_bonus = bonus_model.expected_bonus(
        profile, season, n_draws=n_draws, seed=seed, calibration=bonus_calibration
    ).to_numpy() * _pos_points(scoring, "bonus", pos)

    out = merged.loc[:, list(_PLAYER_KEYS)].copy()
    out["xp_appearance"] = xp_appearance
    out["xp_goals"] = xp_goals
    out["xp_assists"] = xp_assists
    out["xp_cs"] = xp_cs
    out["xp_concede"] = xp_concede
    out["xp_saves"] = xp_saves
    out["xp_defcon"] = xp_defcon
    out["xp_bonus"] = xp_bonus
    out["xp_cards"] = xp_cards
    out["xp_other"] = xp_other
    out["xp"] = out[list(COMPONENT_COLUMNS)].sum(axis=1)
    return out.loc[:, list(XP_COLUMNS)]


def aggregate_gw(xp: pd.DataFrame) -> pd.DataFrame:
    """Sum a player's per-fixture xP rows within each GW (the optimizer's view).

    DGWs sum both fixtures, blank GWs simply have no rows. Returns one row per
    ``(season, gw, player_code)`` with ``n_fixtures``, ``xp`` and every
    component column summed; components still sum to ``xp``.
    """
    _require(xp, XP_COLUMNS, "xp")
    named_aggs: dict[str, tuple[str, str]] = {
        "n_fixtures": ("fpl_fixture_id", "count"),
        **{col: (col, "sum") for col in ("xp", *COMPONENT_COLUMNS)},
    }
    return xp.groupby(["season", "gw", "player_code"], as_index=False).agg(**named_aggs)  # type: ignore[call-overload]
