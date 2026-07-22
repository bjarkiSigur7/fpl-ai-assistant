"""Leakage-safe multi-horizon feature builder (OpenFPL-style windows).

Builds one row per (player_code, season, gw[, fpl_fixture_id]) with ``f_*`` feature
columns computed STRICTLY from information available before that GW's deadline:

* rows for GW ``g`` may only use match outcomes with ``(season, gw)`` lexicographically
  smaller than the row's — so the second leg of a double gameweek never sees the first
  leg (both are predicted from the same pre-deadline information set);
* fixture *schedule* (kickoff times, venue, number of fixtures this GW) is known before
  the deadline and is therefore usable for the current GW's context features
  (days-rest, congestion, ``f_n_fixtures``, ``f_was_home``).

Feature categories (all prefixed with :data:`FEATURE_PREFIX`):

* **player form** — NaN-aware means of points/minutes/goals/assists/bps/bonus/xg/xa/
  us_xg/us_xa/shots/saves/defcon-counts over the previous 1/3/5/10/38 matches, plus
  starts-share over the previous 3/5/10 matches. Windows are computed per
  ``player_code`` over the match sequence ordered by ``(season, gw, kickoff)`` and
  carry over season boundaries and club changes (form travels with the player);
  ``f_new_season`` / ``f_new_club`` flag those discontinuities (the ``stint_id``
  split is surfaced through ``f_new_club`` rather than a hard window reset) and
  ``f_days_since_last_match`` gives the model a continuous decay signal.
* **team / opponent form** — goals for/against and team xG for/against over the same
  windows, computed per ``(season, team_code)`` (team windows reset each season:
  squads and team_code semantics change). Team xG per fixture is the sum of player
  FPL ``xg`` (2022+) with Understat ``us_xg`` as fallback; NaN where neither exists.
* **match context** — venue, team days-rest and 21-day congestion (schedule-derived,
  computed over the real kickoff sequence including same-GW siblings — schedule is
  pre-deadline information), season phase, promoted flags, position one-hot, price,
  empty-stadium regime flag, fixtures-this-GW count.
* **availability** — optional ``tables["availability"]`` frame with columns
  ``season, gw, player_code, status, chance_of_playing`` (deadline-timestamped
  snapshots); features are all-NaN when the table is absent (pre-2026 history).

Void GWs (2019-20 events 30-38, 2022-23 GW7) are excluded from both the output rows
and the rolling history — they carry no information and would poison form windows.

Nullable inputs stay NaN (never 0-filled): ``starts`` pre-2022, FPL xg pre-2022,
Understat fields where unmatched, rich defensive stats outside 2016-18/2025+.
LightGBM consumes NaN natively.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

#: Every feature column starts with this prefix; everything else is a key/meta/label.
FEATURE_PREFIX = "f_"

#: Multi-horizon window lengths (previous N matches), per the OpenFPL template.
WINDOWS: tuple[int, ...] = (1, 3, 5, 10, 38)

#: Windows used for the starts-share features (minutes-model inputs).
STARTS_WINDOWS: tuple[int, ...] = (3, 5, 10)

#: Player-form stats: feature stem -> player_match column ("defcon" is derived).
PLAYER_STATS: dict[str, str] = {
    "points": "total_points",
    "minutes": "minutes",
    "goals": "goals_scored",
    "assists": "assists",
    "bps": "bps",
    "bonus": "bonus",
    "xg": "xg",
    "xa": "xa",
    "us_xg": "us_xg",
    "us_xa": "us_xa",
    "shots": "us_shots",
    "saves": "saves",
    "defcon": "_defcon_count",
}

#: Team-form stats computed on the team-fixture frame.
TEAM_STATS: tuple[str, ...] = ("gf", "ga", "xg", "xga")

#: Outcome (label) columns carried through for training joins — never features.
LABEL_COLUMNS: tuple[str, ...] = (
    "minutes",
    "total_points",
    "goals_scored",
    "assists",
    "clean_sheets",
    "goals_conceded",
    "saves",
    "penalties_saved",
    "penalties_missed",
    "yellow_cards",
    "red_cards",
    "own_goals",
    "bonus",
    "bps",
    "starts",
    "defensive_contribution",
    "tackles",
    "recoveries",
    "clearances_blocks_interceptions",
    "xg",
    "xa",
    "xgc",
    "us_xg",
    "us_xa",
    "us_npxg",
    "us_shots",
    "us_key_passes",
)

_STATUSES: tuple[str, ...] = ("a", "d", "i", "s", "u")
_POSITIONS: tuple[str, ...] = ("GKP", "DEF", "MID", "FWD")


def get_label_columns() -> list[str]:
    """Label (non-feature) outcome columns present in the built frame.

    These are realized match/GW outcomes for training joins; key columns
    (``season, gw, player_code[, fpl_fixture_id]``) and identity/meta columns
    (``fpl_element_id, team_code, opponent_code, position, was_home, price,
    kickoff_utc, stint_id, n_fixtures``) are neither labels nor features.
    """
    return list(LABEL_COLUMNS)


# --------------------------------------------------------------------------------------
# Vectorized rolling machinery
# --------------------------------------------------------------------------------------


def _as_float(s: pd.Series) -> np.ndarray:
    """Series (incl. nullable Int64/Float64/bool) -> float64 ndarray with NaN."""
    return pd.array(s, dtype="Float64").to_numpy(dtype="float64", na_value=np.nan)


def _naive_utc(s: pd.Series) -> pd.Series:
    """Kickoff series -> tz-naive UTC datetime64[ns]."""
    if isinstance(s.dtype, pd.DatetimeTZDtype):
        return s.dt.tz_convert("UTC").dt.tz_localize(None)
    return pd.to_datetime(s)


def _group_shift(arr: np.ndarray, gid: np.ndarray, w: int) -> np.ndarray:
    """``arr`` shifted by ``w`` within contiguous groups ``gid``; NaN outside."""
    n = arr.shape[0]
    out = np.full(n, np.nan)
    if 0 < w < n:
        out[w:] = arr[: n - w]
        same = np.zeros(n, dtype=bool)
        same[w:] = gid[w:] == gid[: n - w]
        out[~same] = np.nan
    return out


def _prior_boundary(
    df: pd.DataFrame, group_cols: list[str], block_cols: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Positional index of the last row strictly before each row's GW-block.

    ``df`` must be sorted by ``group_cols`` then ``block_cols`` (then kickoff) with a
    RangeIndex, so rows of a group are contiguous and GW blocks are contiguous runs.
    Returns ``(prior_abs, valid)``: ``prior_abs[i]`` is the positional index of the
    final row of the previous GW block within the same group; ``valid[i]`` is False
    when the row's GW block is the group's first (no prior information).
    """
    pos = df.groupby(group_cols, sort=False).cumcount().to_numpy()
    within = df.groupby(group_cols + block_cols, sort=False).cumcount().to_numpy()
    prior_abs = np.arange(len(df)) - within - 1
    valid = (pos - within) >= 1
    return prior_abs, valid


def _take_prior(arr: np.ndarray, prior_abs: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """``arr[prior_abs]`` where valid, NaN elsewhere (float64 output)."""
    idx = np.clip(prior_abs, 0, max(len(arr) - 1, 0))
    picked = arr[idx].astype("float64", copy=False)
    return np.where(valid, picked, np.nan)


def _window_means(
    gid: np.ndarray, values: np.ndarray, windows: tuple[int, ...]
) -> dict[int, np.ndarray]:
    """NaN-aware rolling means over the last ``w`` rows per group, INCLUSIVE of the
    current row (callers re-anchor to the prior-GW boundary via :func:`_take_prior`).

    ``min_periods=1`` semantics: mean over however many non-NaN values fall in the
    window; NaN when none do.
    """
    finite = np.isfinite(values)
    v0 = np.where(finite, values, 0.0)
    csum = pd.Series(v0).groupby(gid).cumsum().to_numpy()
    ccnt = pd.Series(finite.astype("float64")).groupby(gid).cumsum().to_numpy()
    out: dict[int, np.ndarray] = {}
    for w in windows:
        s_lag = _group_shift(csum, gid, w)
        k_lag = _group_shift(ccnt, gid, w)
        s_lag = np.where(np.isnan(s_lag), 0.0, s_lag)
        k_lag = np.where(np.isnan(k_lag), 0.0, k_lag)
        cnt = ccnt - k_lag
        out[w] = np.where(cnt > 0, (csum - s_lag) / np.where(cnt > 0, cnt, 1.0), np.nan)
    return out


def _defcon_count(pm: pd.DataFrame) -> np.ndarray:
    """Defensive-contribution count: the API column (2025+) or the CBIT/CBIRT
    reconstruction from rich stats (2016-18): DEF counts CBI + tackles, everyone
    else CBI + recoveries + tackles. NaN where components are unavailable."""
    dc = _as_float(pm["defensive_contribution"])
    cbi = _as_float(pm["clearances_blocks_interceptions"])
    tkl = _as_float(pm["tackles"])
    rec = _as_float(pm["recoveries"])
    is_def = (pm["position"] == "DEF").to_numpy()
    recon = cbi + tkl + np.where(is_def, 0.0, rec)
    return np.where(np.isfinite(dc), dc, recon)


# --------------------------------------------------------------------------------------
# Component builders
# --------------------------------------------------------------------------------------


def _player_form_features(pm: pd.DataFrame) -> pd.DataFrame:
    """Player-form windows + sequence flags on the sorted player_match frame."""
    gid = pm.groupby(["player_code"], sort=False).ngroup().to_numpy()
    prior_abs, valid = _prior_boundary(pm, ["player_code"], ["season", "gw"])

    feats: dict[str, np.ndarray] = {}
    for stem, col in PLAYER_STATS.items():
        values = _defcon_count(pm) if col == "_defcon_count" else _as_float(pm[col])
        means = _window_means(gid, values, WINDOWS)
        for w in WINDOWS:
            feats[f"f_{stem}_mean_{w}"] = _take_prior(means[w], prior_abs, valid)

    starts_means = _window_means(gid, _as_float(pm["starts"]), STARTS_WINDOWS)
    for w in STARTS_WINDOWS:
        feats[f"f_starts_share_{w}"] = _take_prior(starts_means[w], prior_abs, valid)

    season = pm["season"].to_numpy(dtype="float64")
    team = pm["team_code"].to_numpy(dtype="float64")
    prev_season = _take_prior(season, prior_abs, valid)
    prev_team = _take_prior(team, prior_abs, valid)
    with np.errstate(invalid="ignore"):
        feats["f_new_season"] = np.where(
            np.isnan(prev_season), np.nan, (season > prev_season).astype("float64")
        )
        feats["f_new_club"] = np.where(
            np.isnan(prev_team), np.nan, (team != prev_team).astype("float64")
        )

    ko_ns = _naive_utc(pm["kickoff_utc"]).to_numpy(dtype="datetime64[ns]").astype("float64")
    ko_ns[pm["kickoff_utc"].isna().to_numpy()] = np.nan
    prev_ko = _take_prior(ko_ns, prior_abs, valid)
    feats["f_days_since_last_match"] = (ko_ns - prev_ko) / (86_400 * 1e9)

    return pd.DataFrame(feats, index=pm.index)


def _team_fixture_frame(pm: pd.DataFrame, fx: pd.DataFrame) -> pd.DataFrame:
    """Long team-fixture frame: one row per (season, fpl_fixture_id, team_code) with
    gf/ga (fixtures) and team xg/xga (player sums; FPL 2022+ else Understat)."""
    home = fx.rename(
        columns={"home_team_code": "team_code", "away_team_code": "opp_code"}
    ).assign(gf=lambda d: _as_float(d["home_goals"]), ga=lambda d: _as_float(d["away_goals"]))
    away = fx.rename(
        columns={"away_team_code": "team_code", "home_team_code": "opp_code"}
    ).assign(gf=lambda d: _as_float(d["away_goals"]), ga=lambda d: _as_float(d["home_goals"]))
    cols = ["season", "gw", "fpl_fixture_id", "kickoff_utc", "team_code", "opp_code", "gf", "ga"]
    tm = pd.concat([home[cols], away[cols]], ignore_index=True)

    pxg = (
        pm.groupby(["season", "fpl_fixture_id", "team_code"], sort=False)[["xg", "us_xg"]]
        .sum(min_count=1)
        .reset_index()
    )
    pxg["team_xg"] = pxg["xg"].astype("Float64").fillna(pxg["us_xg"].astype("Float64"))
    tm = tm.merge(
        pxg[["season", "fpl_fixture_id", "team_code", "team_xg"]],
        on=["season", "fpl_fixture_id", "team_code"],
        how="left",
        validate="1:1",
    )
    opp_xg = tm[["season", "fpl_fixture_id", "team_code", "team_xg"]].rename(
        columns={"team_code": "opp_code", "team_xg": "team_xga"}
    )
    tm = tm.merge(opp_xg, on=["season", "fpl_fixture_id", "opp_code"], how="left", validate="1:1")
    tm["xg"] = _as_float(tm["team_xg"])
    tm["xga"] = _as_float(tm["team_xga"])
    return tm.drop(columns=["team_xg", "team_xga"])


def _team_form_features(tm: pd.DataFrame) -> pd.DataFrame:
    """Team-form windows (reset per season) + schedule context per team-fixture.

    Returns a frame keyed by (season, fpl_fixture_id, team_code) with ``f_team_*``
    window means and schedule-derived ``f_days_rest`` / ``f_congestion_21d``.
    """
    tm = tm.sort_values(
        ["season", "team_code", "gw", "kickoff_utc", "fpl_fixture_id"], kind="stable"
    ).reset_index(drop=True)
    gid = tm.groupby(["season", "team_code"], sort=False).ngroup().to_numpy()
    prior_abs, valid = _prior_boundary(tm, ["season", "team_code"], ["gw"])

    feats: dict[str, np.ndarray] = {}
    for stem in TEAM_STATS:
        means = _window_means(gid, _as_float(tm[stem]), WINDOWS)
        for w in WINDOWS:
            feats[f"f_team_{stem}_mean_{w}"] = _take_prior(means[w], prior_abs, valid)
    form = pd.concat(
        [tm[["season", "fpl_fixture_id", "team_code"]], pd.DataFrame(feats, index=tm.index)],
        axis=1,
    )

    # Schedule features on the true kickoff order (pre-deadline information, so the
    # current GW's earlier fixtures DO count for rest/congestion of the later ones).
    sched = tm[["season", "fpl_fixture_id", "team_code", "kickoff_utc"]].copy()
    sched["_ko"] = _naive_utc(sched["kickoff_utc"])
    sched = sched.sort_values(
        ["season", "team_code", "_ko", "fpl_fixture_id"], kind="stable"
    ).reset_index(drop=True)
    grp = sched.groupby(["season", "team_code"], sort=False)
    sched["f_days_rest"] = grp["_ko"].diff().dt.total_seconds().to_numpy() / 86_400

    def _congestion(ko: pd.Series) -> pd.Series:
        arr = ko.to_numpy(dtype="datetime64[ns]")
        lo = np.searchsorted(arr, arr - np.timedelta64(21, "D"), side="left")
        return pd.Series(np.arange(len(arr), dtype="float64") - lo, index=ko.index)

    sched["f_congestion_21d"] = grp["_ko"].transform(_congestion)
    return form.merge(
        sched[["season", "fpl_fixture_id", "team_code", "f_days_rest", "f_congestion_21d"]],
        on=["season", "fpl_fixture_id", "team_code"],
        how="left",
        validate="1:1",
    )


def _promoted_frame(teams: pd.DataFrame) -> pd.DataFrame:
    """(season, team_code) -> promoted flag: 1.0 if the club was not in the league the
    previous season, 0.0 if it was, NaN when the previous season is absent from the
    teams table (unknowable, e.g. the first backfilled season)."""
    t = teams[["season", "team_code"]].drop_duplicates()
    prev = t.assign(season=t["season"] + 1, _prev=1.0)
    out = t.merge(prev, on=["season", "team_code"], how="left")
    seasons = set(t["season"].unique())
    known = out["season"].map(lambda s: (s - 1) in seasons)
    out["promoted"] = np.where(known, np.where(out["_prev"].notna(), 0.0, 1.0), np.nan)
    return out[["season", "team_code", "promoted"]]


def _availability_features(pm: pd.DataFrame, avail: pd.DataFrame | None) -> pd.DataFrame:
    """Availability features joined on (season, gw, player_code); all-NaN columns when
    no snapshot table is provided (history before our daily archive)."""
    cols = [f"f_status_{s}" for s in _STATUSES] + ["f_chance_of_playing"]
    if avail is None or avail.empty:
        return pd.DataFrame(
            {c: np.full(len(pm), np.nan) for c in cols}, index=pm.index
        )
    required = {"season", "gw", "player_code"}
    missing = required - set(avail.columns)
    if missing:
        raise ValueError(f"availability table missing columns: {sorted(missing)}")
    a = avail.drop_duplicates(["season", "gw", "player_code"]).copy()
    status = a["status"] if "status" in a.columns else pd.Series(pd.NA, index=a.index)
    for s in _STATUSES:
        eq = (status == s).fillna(False).astype("float64")
        a[f"f_status_{s}"] = np.where(status.isna(), np.nan, eq)
    chance_col = next(
        (c for c in ("chance_of_playing", "chance_of_playing_this_round") if c in a.columns),
        None,
    )
    a["f_chance_of_playing"] = (
        _as_float(a[chance_col]) if chance_col is not None else np.nan
    )
    joined = pm[["season", "gw", "player_code"]].merge(
        a[["season", "gw", "player_code", *cols]],
        on=["season", "gw", "player_code"],
        how="left",
        validate="m:1",
    )
    joined.index = pm.index
    return joined[cols]


# --------------------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------------------


def build_feature_frame(
    tables: dict[str, pd.DataFrame], *, target: Literal["match", "gw"]
) -> pd.DataFrame:
    """Build the leakage-safe multi-horizon feature frame.

    Parameters
    ----------
    tables:
        Canonical processed tables. Required: ``"player_match"``, ``"fixtures"``,
        ``"teams"``. Optional: ``"availability"`` (columns ``season, gw, player_code,
        status, chance_of_playing`` from deadline-timestamped snapshots).
    target:
        ``"match"`` — one row per (player_code, season, gw, fpl_fixture_id);
        ``"gw"`` — one row per (player_code, season, gw): labels summed over the GW's
        fixtures, features averaged (form features are identical across a GW's
        fixtures by construction; per-fixture context averages out).

    Returns
    -------
    DataFrame with key columns, meta columns, :data:`LABEL_COLUMNS` and ``f_*``
    features (all float64; NaN where information does not exist yet — LightGBM
    handles NaN natively). Every ``f_*`` value for a row in GW ``g`` is computable
    from matches with ``(season, gw) < (row.season, g)`` plus pre-deadline schedule
    and static info only.
    """
    if target not in ("match", "gw"):
        raise ValueError(f"target must be 'match' or 'gw', got {target!r}")
    for name in ("player_match", "fixtures", "teams"):
        if name not in tables:
            raise ValueError(f"tables is missing required table {name!r}")

    pm = tables["player_match"]
    fx = tables["fixtures"]
    teams = tables["teams"]

    pm = pm.loc[~pm["void_gw"].astype(bool)].copy()
    fx = fx.loc[~fx["void"].astype(bool)].copy()
    fx["gw"] = fx["gw"].astype("int64")

    pm = pm.merge(
        fx[["season", "fpl_fixture_id", "kickoff_utc"]],
        on=["season", "fpl_fixture_id"],
        how="left",
        validate="m:1",
    )
    pm = pm.sort_values(
        ["player_code", "season", "gw", "kickoff_utc", "fpl_fixture_id"], kind="stable"
    ).reset_index(drop=True)

    player_feats = _player_form_features(pm)
    team_form = _team_form_features(_team_fixture_frame(pm, fx))
    promoted = _promoted_frame(teams)
    avail_feats = _availability_features(pm, tables.get("availability"))

    out = pd.concat([pm, player_feats, avail_feats], axis=1)

    out = out.merge(
        team_form, on=["season", "fpl_fixture_id", "team_code"], how="left", validate="m:1"
    )
    opp_form = team_form.rename(
        columns={
            "team_code": "opponent_code",
            **{f"f_team_{s}_mean_{w}": f"f_opp_{s}_mean_{w}" for s in TEAM_STATS for w in WINDOWS},
        }
    ).drop(columns=["f_days_rest", "f_congestion_21d"])
    out = out.merge(
        opp_form, on=["season", "fpl_fixture_id", "opponent_code"], how="left", validate="m:1"
    )

    out = out.merge(promoted.rename(columns={"promoted": "f_promoted"}),
                    on=["season", "team_code"], how="left", validate="m:1")
    out = out.merge(
        promoted.rename(columns={"team_code": "opponent_code", "promoted": "f_opp_promoted"}),
        on=["season", "opponent_code"],
        how="left",
        validate="m:1",
    )

    # Match context (all pre-deadline: schedule, venue, price, static season info).
    out["f_was_home"] = out["was_home"].astype("float64")
    out["f_empty_stadium"] = out["empty_stadium"].astype("float64")
    out["f_season_phase"] = out["gw"].astype("float64") / 38.0
    out["f_price"] = out["price"].astype("float64")
    for p in _POSITIONS:
        out[f"f_pos_{p}"] = (out["position"] == p).astype("float64")
    out["f_n_fixtures"] = (
        out.groupby(["season", "gw", "player_code"], sort=False)["fpl_fixture_id"]
        .transform("size")
        .astype("float64")
    )

    feature_cols = sorted(c for c in out.columns if c.startswith(FEATURE_PREFIX))
    label_cols = [c for c in LABEL_COLUMNS if c in out.columns]

    if target == "match":
        keys = ["season", "gw", "fpl_fixture_id", "player_code"]
        meta = [
            "fpl_element_id",
            "team_code",
            "opponent_code",
            "position",
            "was_home",
            "price",
            "kickoff_utc",
            "stint_id",
        ]
        out = out[keys + meta + label_cols + feature_cols]
        return out.sort_values(keys, kind="stable").reset_index(drop=True)

    grp = out.groupby(["season", "gw", "player_code"], sort=False)
    labels = grp[label_cols].sum(min_count=1)
    feats = grp[feature_cols].mean()
    meta_gw = grp[["fpl_element_id", "team_code", "position", "price"]].first()
    kickoff = grp["kickoff_utc"].min().rename("kickoff_utc")
    n_fixtures = grp.size().rename("n_fixtures")
    gw_out = pd.concat([meta_gw, kickoff, n_fixtures, labels, feats], axis=1).reset_index()
    return gw_out.sort_values(["season", "gw", "player_code"], kind="stable").reset_index(
        drop=True
    )


def assert_no_leakage(
    df: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
    *,
    target: Literal["match", "gw"] = "match",
    n_samples: int = 25,
    seed: int = 0,
) -> None:
    """Debug guard: recompute sampled ``f_*`` values from ONLY strictly-prior matches.

    For ``n_samples`` random rows of ``df``, player-form window means, starts-share
    and (for ``target="match"``, or single-fixture GWs) team/opponent goals-for means
    are recomputed naively from ``tables`` restricted to matches with
    ``(season, gw) < (row.season, row.gw)``, and compared to the built values.
    Raises :class:`AssertionError` on any mismatch (tolerance 1e-9, NaN == NaN).
    """
    rng = np.random.default_rng(seed)
    pm = tables["player_match"].loc[~tables["player_match"]["void_gw"].astype(bool)]
    fx = tables["fixtures"].loc[~tables["fixtures"]["void"].astype(bool)].copy()
    fx["gw"] = fx["gw"].astype("int64")
    pm = pm.merge(
        fx[["season", "fpl_fixture_id", "kickoff_utc"]],
        on=["season", "fpl_fixture_id"],
        how="left",
        validate="m:1",
    )

    def _close(a: float, b: float, name: str, row: pd.Series) -> None:
        ok = (np.isnan(a) and np.isnan(b)) or bool(np.isclose(a, b, atol=1e-9, equal_nan=True))
        assert ok, (
            f"leakage/recompute mismatch for {name} at season={row['season']} "
            f"gw={row['gw']} player_code={row['player_code']}: built={a!r} recomputed={b!r}"
        )

    idx = rng.choice(len(df), size=min(n_samples, len(df)), replace=False)
    for i in idx:
        row = df.iloc[int(i)]
        season, gw, code = int(row["season"]), int(row["gw"]), int(row["player_code"])
        hist = pm.loc[
            (pm["player_code"] == code)
            & ((pm["season"] < season) | ((pm["season"] == season) & (pm["gw"] < gw)))
        ].sort_values(["season", "gw", "kickoff_utc", "fpl_fixture_id"], kind="stable")
        dc = _defcon_count(hist) if len(hist) else np.array([])
        for stem, col in PLAYER_STATS.items():
            vals = dc if col == "_defcon_count" else _as_float(hist[col])
            for w in WINDOWS:
                tail = vals[-w:] if len(vals) else vals
                expect = float(np.nanmean(tail)) if np.isfinite(tail).any() else np.nan
                _close(float(row[f"f_{stem}_mean_{w}"]), expect, f"f_{stem}_mean_{w}", row)
        starts = _as_float(hist["starts"])
        for w in STARTS_WINDOWS:
            tail = starts[-w:] if len(starts) else starts
            expect = float(np.nanmean(tail)) if np.isfinite(tail).any() else np.nan
            _close(float(row[f"f_starts_share_{w}"]), expect, f"f_starts_share_{w}", row)

        if target == "gw" and row.get("n_fixtures", 1) != 1:
            continue  # team context was averaged over the GW's fixtures; skip
        team_code = int(row["team_code"])
        th = fx.loc[
            (fx["season"] == season)
            & (fx["gw"] < gw)
            & ((fx["home_team_code"] == team_code) | (fx["away_team_code"] == team_code))
        ].sort_values(["gw", "kickoff_utc", "fpl_fixture_id"], kind="stable")
        gf = np.where(
            th["home_team_code"] == team_code,
            _as_float(th["home_goals"]),
            _as_float(th["away_goals"]),
        )
        for w in WINDOWS:
            tail = gf[-w:] if len(gf) else gf
            expect = float(np.nanmean(tail)) if np.isfinite(tail).any() else np.nan
            _close(float(row[f"f_team_gf_mean_{w}"]), expect, f"f_team_gf_mean_{w}", row)
