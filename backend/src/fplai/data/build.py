"""Raw vaastav files -> canonical processed parquet tables.

Implements every FPL_KNOWLEDGE.md §2.4 hard rule for the training corpus:

* **2019-20 COVID remap** — API events 39-47 are remapped to canonical GWs 30-38;
  the original events 30-38 exist upstream only as header-only stubs (any stray rows
  are dropped).  Applied to both fixtures and player rows.
* **2022-23 GW7 void** — the gameweek rolled over after Queen Elizabeth II's death;
  merged_gw ships GW7 rows with 0 minutes for everyone.  Fixture rows are kept with
  ``void=True``; player rows carry ``void_gw=True`` so minutes/appearance models can
  exclude them.
* **Empty-stadium regime** — ``empty_stadium=True`` for 2019-20 canonical GWs 30-38
  (the behind-closed-doors restart, API events 39-47) and for all of 2020-21.
* **Subs regime** — ``subs_regime`` 3 for seasons <= 2021-22, 5 from 2022-23 (PL
  moved to five substitutions).
* **Mid-season club switches** — ``stint_id`` starts at 0 per (season, player) and
  increments whenever the player's ``team_code`` changes within the season (2025-26
  reference cases: Semenyo BOU->MCI, Guéhi CRY->MCI in Jan 2026).
* **Positions** are normalized to GKP/DEF/MID/FWD across all seasons (older files say
  "GK"; pre-2020-21 files have no position column at all and are joined from
  ``players_raw.csv`` ``element_type``).  2024-25 assistant-manager elements
  (element_type 5 / position "AM") are dropped from player tables.
* **Prices** are the observed per-GW ``value`` column (0.1m units) — never simulated.

merged_gw.csv is one row per player per *fixture* (so double gameweeks produce two
rows for one GW); ``player_match.parquet`` keeps that grain and
``player_gw.parquet`` sums over each GW's fixtures.

For 2016-17/2017-18 the upstream repo ships no ``fixtures.csv``; fixtures are
reconstructed from merged_gw (each fixture's home team id is the ``opponent_team``
of its away players and vice versa — every fixture has rows from both sides because
old merged_gw files include all elements every GW).

All builders are deterministic and idempotent: same raw inputs -> byte-identical
sorted parquet outputs; safe to re-run.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from fplai import config
from fplai.data import crosswalk, vaastav

logger = logging.getLogger(__name__)

# --- schema constants -------------------------------------------------------------

#: Outcome columns present in every season of merged_gw (int64, non-null).
CORE_OUTCOME_COLS: tuple[str, ...] = (
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
)

#: Nullable Int64 outcome columns: present only in some seasons (see §2.3 matrix).
#: starts: 2022-23+.  defensive_contribution: 2025-26+.  tackles / recoveries /
#: clearances_blocks_interceptions: rich-stats era 2016-17..2018-19 AND 2025-26+.
NULLABLE_INT_OUTCOME_COLS: tuple[str, ...] = (
    "starts",
    "defensive_contribution",
    "tackles",
    "recoveries",
    "clearances_blocks_interceptions",
)

#: FPL-API xG family (2022-23+), renamed from the expected_* merged_gw columns.
XG_COLS: dict[str, str] = {
    "expected_goals": "xg",
    "expected_assists": "xa",
    "expected_goals_conceded": "xgc",
}

#: Understat-sourced columns — all-NA placeholders here; populated by the Understat
#: client's join (aux-source agent) for all seasons where players are matched.
UNDERSTAT_COLS: tuple[str, ...] = ("us_xg", "us_xa", "us_npxg", "us_shots", "us_key_passes")

PLAYER_MATCH_COLS: tuple[str, ...] = (
    "season",
    "gw",
    "fpl_fixture_id",
    "player_code",
    "fpl_element_id",
    "team_code",
    "opponent_code",
    "was_home",
    "position",
    "price",
    *CORE_OUTCOME_COLS,
    *NULLABLE_INT_OUTCOME_COLS,
    *XG_COLS.values(),
    *UNDERSTAT_COLS,
    "empty_stadium",
    "void_gw",
    "subs_regime",
    "stint_id",
)

FIXTURES_COLS: tuple[str, ...] = (
    "season",
    "gw",
    "fpl_fixture_id",
    "kickoff_utc",
    "home_team_code",
    "away_team_code",
    "home_goals",
    "away_goals",
    "finished",
    "void",
)

#: Position spellings across seasons -> canonical. Older merged_gw files use "GK";
#: "AM" (2024-25 assistant managers) is intentionally unmapped and dropped upstream.
POSITION_MAP: dict[str, str] = {"GK": "GKP", "GKP": "GKP", "DEF": "DEF", "MID": "MID", "FWD": "FWD"}

_COVID_SEASON = 2019
_COVID_STUB_GWS = range(30, 39)  # header-only stub events upstream — rows dropped
_COVID_REMAP_OFFSET = 9  # API events 39-47 -> canonical 30-38
_VOID_SEASON, _VOID_GW = 2022, 7  # Queen Elizabeth II rollover
_FIVE_SUBS_FROM = 2022


# --- small coercion helpers -------------------------------------------------------


def _to_bool(s: pd.Series, *, what: str) -> pd.Series:
    """Coerce a True/False column that may arrive as bool or string."""
    if s.dtype == bool:
        return s
    mapped = s.map(
        {True: True, False: False, "True": True, "False": False, "true": True, "false": False}
    )
    if mapped.isna().any():
        raise ValueError(f"Unparseable {what} values: {s[mapped.isna()].unique()[:5]}")
    return mapped.astype(bool)


def _canonical_gw(season: int, gw: pd.Series, *, what: str) -> pd.Series:
    """Apply the 2019-20 COVID event remap; drop nothing (caller drops stubs first)."""
    gw = gw.astype("Int64")
    if season == _COVID_SEASON:
        remap = gw.ge(39).fillna(False).astype(bool)
        gw = gw.mask(remap, gw - _COVID_REMAP_OFFSET)
        logger.debug("%s 2019-20: remapped %d rows from events 39-47", what, int(remap.sum()))
    return gw


def _drop_covid_stub_rows(season: int, df: pd.DataFrame, gw_col: str, what: str) -> pd.DataFrame:
    if season != _COVID_SEASON:
        return df
    stubs = df[gw_col].isin(list(_COVID_STUB_GWS)).fillna(False).astype(bool)
    if stubs.any():
        logger.info("2019-20 %s: dropping %d stub rows for void events 30-38", what, stubs.sum())
    return df[~stubs]


def _empty_stadium(season: int, gw: pd.Series) -> pd.Series:
    """§2.4.4 home-advantage regime: 2019-20 restart (canonical GWs 30+) + all 2020-21."""
    if season == 2020:
        return pd.Series(True, index=gw.index)
    if season == _COVID_SEASON:
        return gw.ge(30).fillna(False).astype(bool)
    return pd.Series(False, index=gw.index)


def _subs_regime(season: int) -> int:
    return 5 if season >= _FIVE_SUBS_FROM else 3


def _to_utc_ns(s: pd.Series) -> pd.Series:
    """Parse kickoff strings to tz-aware UTC datetimes at a stable ns resolution."""
    return pd.to_datetime(s, utc=True, errors="coerce").dt.as_unit("ns")


# --- fixtures ---------------------------------------------------------------------


def _season_fixtures_from_csv(season: int, path: Path) -> pd.DataFrame:
    fx = vaastav.read_csv_tolerant(path)
    out = pd.DataFrame(
        {
            "gw": fx["event"],
            "fpl_fixture_id": fx["id"].astype("int64"),
            "kickoff_utc": _to_utc_ns(fx["kickoff_time"]),
            "home_team_id": fx["team_h"].astype("int64"),
            "away_team_id": fx["team_a"].astype("int64"),
            "home_goals": fx["team_h_score"].astype("Int64"),
            "away_goals": fx["team_a_score"].astype("Int64"),
            "finished": _to_bool(fx["finished"], what="finished"),
        }
    )
    return out


def _season_fixtures_from_merged_gw(season: int, mg: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct fixtures for 2016-17/2017-18 (no fixtures.csv upstream).

    Home team id = ``opponent_team`` of the fixture's away-player rows (and vice
    versa); kickoff/scores/GW are constant within a fixture.
    """
    was_home = _to_bool(mg["was_home"], what="was_home")
    per_fixture = (
        mg.assign(_was_home=was_home)
        .groupby("fixture", sort=True)
        .agg(
            gw=("GW", "first"),
            kickoff=("kickoff_time", "first"),
            home_goals=("team_h_score", "first"),
            away_goals=("team_a_score", "first"),
        )
    )
    home_ids = mg.loc[~was_home].groupby("fixture")["opponent_team"].first()
    away_ids = mg.loc[was_home].groupby("fixture")["opponent_team"].first()
    per_fixture["home_team_id"] = home_ids
    per_fixture["away_team_id"] = away_ids
    if per_fixture[["home_team_id", "away_team_id"]].isna().any().any():
        bad = per_fixture[per_fixture[["home_team_id", "away_team_id"]].isna().any(axis=1)]
        raise ValueError(
            f"{vaastav.season_label(season)}: cannot infer both teams for fixtures "
            f"{list(bad.index[:5])}"
        )
    out = pd.DataFrame(
        {
            "gw": per_fixture["gw"],
            "fpl_fixture_id": per_fixture.index.astype("int64"),
            "kickoff_utc": _to_utc_ns(per_fixture["kickoff"]),
            "home_team_id": per_fixture["home_team_id"].astype("int64"),
            "away_team_id": per_fixture["away_team_id"].astype("int64"),
            "home_goals": per_fixture["home_goals"].astype("Int64"),
            "away_goals": per_fixture["away_goals"].astype("Int64"),
        }
    ).reset_index(drop=True)
    out["finished"] = out["home_goals"].notna() & out["away_goals"].notna()
    return out


def _read_merged_gw(season: int, raw_root: Path | None) -> pd.DataFrame:
    path = vaastav.season_dir(season, raw_root) / "merged_gw.csv"
    mg = vaastav.read_csv_tolerant(path, low_memory=False)
    if "GW" not in mg.columns and "round" in mg.columns:  # defensive: GW always present
        mg = mg.rename(columns={"round": "GW"})
    return mg


def build_fixtures(
    seasons: Iterable[int] | None = None,
    raw_root: Path | None = None,
    *,
    teams: pd.DataFrame | None = None,
    out_path: Path | None = None,
    write: bool = True,
) -> pd.DataFrame:
    """Build ``fixtures.parquet`` (one row per fixture, canonical remapped GWs)."""
    seasons = crosswalk.resolve_seasons(seasons, raw_root)
    if teams is None:
        teams = crosswalk.build_teams_crosswalk(seasons, raw_root)

    frames = []
    for season in seasons:
        fixtures_csv = vaastav.season_dir(season, raw_root) / "fixtures.csv"
        if fixtures_csv.exists():
            fx = _season_fixtures_from_csv(season, fixtures_csv)
        else:
            fx = _season_fixtures_from_merged_gw(season, _read_merged_gw(season, raw_root))
        fx = _drop_covid_stub_rows(season, fx, "gw", "fixtures")
        fx["gw"] = _canonical_gw(season, fx["gw"], what="fixtures")
        fx.insert(0, "season", season)

        code_by_id = (
            teams.loc[teams["season"] == season]
            .set_index("fpl_team_id")["team_code"]
            .astype("int64")
        )
        for side in ("home", "away"):
            codes = fx[f"{side}_team_id"].map(code_by_id)
            if codes.isna().any():
                missing = fx.loc[codes.isna(), f"{side}_team_id"].unique()
                raise ValueError(
                    f"{vaastav.season_label(season)}: unknown {side} team ids {missing[:5]}"
                )
            fx[f"{side}_team_code"] = codes.astype("int64")
        fx["void"] = (season == _VOID_SEASON) & (fx["gw"] == _VOID_GW).fillna(False).astype(bool)
        frames.append(fx[list(FIXTURES_COLS)])

    fixtures = (
        pd.concat(frames, ignore_index=True)
        .sort_values(["season", "gw", "fpl_fixture_id"])
        .reset_index(drop=True)
    )
    if write:
        out = out_path or (config.PROCESSED_DIR / "fixtures.parquet")
        out.parent.mkdir(parents=True, exist_ok=True)
        fixtures.to_parquet(out, index=False)
        logger.info("wrote %s (%d rows)", out, len(fixtures))
    return fixtures


# --- player_match -----------------------------------------------------------------


def _season_player_match(
    season: int,
    raw_root: Path | None,
    fixtures: pd.DataFrame,
) -> pd.DataFrame:
    """One season of player-fixture rows (includes _transfers/_selected private cols)."""
    label = vaastav.season_label(season)
    mg = _read_merged_gw(season, raw_root)
    mg = mg[mg["element"].notna()].copy()
    mg = _drop_covid_stub_rows(season, mg, "GW", "merged_gw")
    mg["gw"] = _canonical_gw(season, mg["GW"], what="merged_gw")
    if mg["gw"].isna().any():
        raise ValueError(f"{label}: {int(mg['gw'].isna().sum())} merged_gw rows without a GW")

    # element -> stable code + element_type from the season's bootstrap dump.
    players = vaastav.read_csv_tolerant(
        vaastav.season_dir(season, raw_root) / "players_raw.csv"
    )[["id", "code", "element_type"]]
    n_before = len(mg)
    mg = mg.merge(players, left_on="element", right_on="id", how="inner", suffixes=("", "_pr"))
    if len(mg) < n_before:
        logger.warning(
            "%s: dropped %d merged_gw rows with element ids missing from players_raw",
            label,
            n_before - len(mg),
        )
    # 2024-25 assistant-manager elements (element_type 5, position "AM") are not players.
    mg = mg[mg["element_type"].astype(int) <= 4]

    # Deduplicate defensively on the output key (vaastav has shipped dup rows before).
    key = ["gw", "fixture", "element"]
    dups = mg.duplicated(subset=key)
    if dups.any():
        logger.warning("%s: dropping %d duplicate merged_gw rows", label, int(dups.sum()))
        mg = mg[~dups]

    # Fixture join: teams-of-record and kickoff come from the fixtures table (already
    # remapped + team ids resolved to stable codes). NOTE: gw comes from merged_gw, not
    # the fixture — 2022-23 GW7 void rows reference fixtures rescheduled to later GWs.
    fx = fixtures.loc[
        fixtures["season"] == season,
        ["fpl_fixture_id", "home_team_code", "away_team_code", "kickoff_utc"],
    ]
    mg = mg.merge(fx, left_on="fixture", right_on="fpl_fixture_id", how="left")
    if mg["home_team_code"].isna().any():
        missing = mg.loc[mg["home_team_code"].isna(), "fixture"].unique()
        raise ValueError(f"{label}: merged_gw references unknown fixtures {missing[:5]}")

    was_home = _to_bool(mg["was_home"], what="was_home")
    team_code = np.where(was_home, mg["home_team_code"], mg["away_team_code"])
    opponent_code = np.where(was_home, mg["away_team_code"], mg["home_team_code"])

    # Position: per-GW column where vaastav ships one (2020-21+, spelled "GK" etc.),
    # else the season's element_type. Both normalized to GKP/DEF/MID/FWD.
    if "position" in mg.columns:
        position = mg["position"].map(POSITION_MAP)
    else:
        position = mg["element_type"].astype(int).map(crosswalk.ELEMENT_TYPE_TO_POSITION)
    if position.isna().any():
        bad = mg.loc[position.isna(), "position" if "position" in mg.columns else "element_type"]
        raise ValueError(f"{label}: unmappable positions {bad.unique()[:5]}")

    out = pd.DataFrame(
        {
            "season": season,
            "gw": mg["gw"].astype("int64"),
            "fpl_fixture_id": mg["fixture"].astype("int64"),
            "player_code": mg["code"].astype("int64"),
            "fpl_element_id": mg["element"].astype("int64"),
            "team_code": pd.Series(team_code, dtype="int64"),
            "opponent_code": pd.Series(opponent_code, dtype="int64"),
            "was_home": was_home.to_numpy(),
            "position": position.astype("string"),
            "price": pd.to_numeric(mg["value"]).round().astype("int64"),
        }
    )
    for col in CORE_OUTCOME_COLS:
        out[col] = pd.to_numeric(mg[col]).fillna(0).round().astype("int64")
    for col in NULLABLE_INT_OUTCOME_COLS:
        if col in mg.columns:
            out[col] = pd.to_numeric(mg[col]).round().astype("Int64")
        else:
            out[col] = pd.Series(pd.NA, index=out.index, dtype="Int64")
    for src, dst in XG_COLS.items():
        if src in mg.columns:
            out[dst] = pd.to_numeric(mg[src]).astype("Float64")
        else:
            out[dst] = pd.Series(pd.NA, index=out.index, dtype="Float64")
    for col in UNDERSTAT_COLS:
        out[col] = pd.Series(pd.NA, index=out.index, dtype="Float64")

    out["empty_stadium"] = _empty_stadium(season, out["gw"]).to_numpy()
    out["void_gw"] = (season == _VOID_SEASON) & (out["gw"] == _VOID_GW)
    out["subs_regime"] = _subs_regime(season)
    out["_kickoff_utc"] = mg["kickoff_utc"].to_numpy()
    out["_transfers_in"] = pd.to_numeric(mg["transfers_in"]).astype("Int64")
    out["_transfers_out"] = pd.to_numeric(mg["transfers_out"]).astype("Int64")
    return out


def _assign_stints(pm: pd.DataFrame) -> pd.DataFrame:
    """stint_id: 0 within (season, player), +1 at every mid-season team_code change."""
    pm = pm.sort_values(
        ["season", "player_code", "gw", "_kickoff_utc", "fpl_fixture_id"], kind="mergesort"
    ).reset_index(drop=True)
    grp = [pm["season"], pm["player_code"]]
    prev_team = pm.groupby(["season", "player_code"], sort=False)["team_code"].shift()
    changed = prev_team.notna() & (pm["team_code"] != prev_team)
    pm["stint_id"] = changed.groupby(grp).cumsum().astype("int64")
    return pm


def build_player_match(
    seasons: Iterable[int] | None = None,
    raw_root: Path | None = None,
    *,
    fixtures: pd.DataFrame | None = None,
    teams: pd.DataFrame | None = None,
    out_path: Path | None = None,
    write: bool = True,
) -> pd.DataFrame:
    """Build ``player_match.parquet`` (one row per player per fixture opportunity).

    The returned frame additionally carries private ``_kickoff_utc`` /
    ``_transfers_in`` / ``_transfers_out`` columns consumed by
    :func:`build_player_gw`; they are excluded from the parquet output.
    """
    seasons = crosswalk.resolve_seasons(seasons, raw_root)
    if fixtures is None:
        fixtures = build_fixtures(seasons, raw_root, teams=teams, write=False)
    pm = pd.concat(
        [_season_player_match(season, raw_root, fixtures) for season in seasons],
        ignore_index=True,
    )
    pm = _assign_stints(pm)
    pm = pm.sort_values(
        ["season", "gw", "fpl_fixture_id", "player_code"], kind="mergesort"
    ).reset_index(drop=True)
    if write:
        out = out_path or (config.PROCESSED_DIR / "player_match.parquet")
        out.parent.mkdir(parents=True, exist_ok=True)
        pm[list(PLAYER_MATCH_COLS)].to_parquet(out, index=False)
        logger.info("wrote %s (%d rows)", out, len(pm))
    return pm


# --- player_gw --------------------------------------------------------------------


def build_player_gw(
    seasons: Iterable[int] | None = None,
    raw_root: Path | None = None,
    *,
    player_match: pd.DataFrame | None = None,
    out_path: Path | None = None,
    write: bool = True,
) -> pd.DataFrame:
    """Build ``player_gw.parquet`` — per player per GW, summed over the GW's fixtures.

    ``player_match`` may be passed through from :func:`build_player_match` (it must
    include the private ``_transfers_*`` columns, i.e. come from that function, not
    from the parquet file).  ``value`` is the observed price that GW;
    ``selected_by_percent`` is left NA (vaastav ships raw ``selected`` counts, not
    percentages — our own daily snapshots carry the true percent from 2026-27 on).
    """
    if player_match is None:
        player_match = build_player_match(seasons, raw_root, write=False)
    pm = player_match.sort_values(
        ["season", "player_code", "gw", "_kickoff_utc", "fpl_fixture_id"], kind="mergesort"
    )
    keys = ["season", "gw", "player_code"]
    g = pm.groupby(keys, sort=True)

    ident = g.agg(
        fpl_element_id=("fpl_element_id", "first"),
        team_code=("team_code", "last"),  # a mid-GW club switch keeps the latest club
        position=("position", "first"),
        n_fixtures=("fpl_fixture_id", "size"),
        value=("price", "first"),
        transfers_in_event=("_transfers_in", "first"),
        transfers_out_event=("_transfers_out", "first"),
    )
    summed_core = g[list(CORE_OUTCOME_COLS)].sum()
    # min_count=1 keeps all-NA groups NA instead of coercing to 0 (e.g. `starts`
    # before 2022-23, xG before 2022-23, DefCon before 2025-26).
    nullable_cols = list(NULLABLE_INT_OUTCOME_COLS) + list(XG_COLS.values())
    summed_nullable = g[nullable_cols].sum(min_count=1)

    out = pd.concat([ident, summed_core, summed_nullable], axis=1).reset_index()
    out["n_fixtures"] = out["n_fixtures"].astype("int64")
    out["selected_by_percent"] = pd.Series(pd.NA, index=out.index, dtype="Float64")
    cols = [
        *keys,
        "fpl_element_id",
        "team_code",
        "position",
        "n_fixtures",
        *CORE_OUTCOME_COLS,
        *NULLABLE_INT_OUTCOME_COLS,
        *XG_COLS.values(),
        "value",
        "selected_by_percent",
        "transfers_in_event",
        "transfers_out_event",
    ]
    out = out[cols].sort_values(keys, kind="mergesort").reset_index(drop=True)
    if write:
        out_file = out_path or (config.PROCESSED_DIR / "player_gw.parquet")
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(out_file, index=False)
        logger.info("wrote %s (%d rows)", out_file, len(out))
    return out


# --- top level --------------------------------------------------------------------


def build_all(
    seasons: Iterable[int] | None = None,
    raw_root: Path | None = None,
    processed_dir: Path | None = None,
) -> dict[str, Path]:
    """Build every processed table this module owns from the raw vaastav mirror.

    Writes ``teams.parquet``, ``players.parquet``, ``fixtures.parquet``,
    ``player_match.parquet`` and ``player_gw.parquet`` to ``processed_dir``
    (default ``config.PROCESSED_DIR``).  ``odds.parquet`` is owned by the
    football-data client.  Deterministic and idempotent.
    """
    seasons = crosswalk.resolve_seasons(seasons, raw_root)
    processed = processed_dir or config.PROCESSED_DIR
    processed.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    teams = crosswalk.build_teams_crosswalk(seasons, raw_root)
    paths["teams"] = processed / "teams.parquet"
    teams.to_parquet(paths["teams"], index=False)

    players = crosswalk.build_players_crosswalk(seasons, raw_root)
    paths["players"] = processed / "players.parquet"
    players.to_parquet(paths["players"], index=False)

    paths["fixtures"] = processed / "fixtures.parquet"
    fixtures = build_fixtures(seasons, raw_root, teams=teams, out_path=paths["fixtures"])

    paths["player_match"] = processed / "player_match.parquet"
    pm = build_player_match(seasons, raw_root, fixtures=fixtures, out_path=paths["player_match"])

    paths["player_gw"] = processed / "player_gw.parquet"
    build_player_gw(player_match=pm, out_path=paths["player_gw"])

    logger.info("build_all complete: %s", ", ".join(str(p) for p in paths.values()))
    return paths
