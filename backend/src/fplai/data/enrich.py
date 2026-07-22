"""Post-build enrichment: the identity joins left open by the base build (STATUS gaps 2-4, 6).

Runs automatically at the end of :func:`fplai.data.build.build_all` and is safe to
re-run standalone: every step recomputes its columns from scratch (idempotent) and
tolerates any subset of raw seasons on disk (partial-season safe, fully offline).

Steps, in order:

1. **Team aux names** — fill ``understat_name`` / ``clubelo_name`` /
   ``footballdata_name`` on ``teams.parquet`` from the deterministic maps in
   :mod:`fplai.data.crosswalk`.
2. **Understat player identity** — fill ``players.understat_id`` for all seasons via
   the fuzzy name+team matcher (:func:`crosswalk.match_players`), seeded from
   vaastav's ``id_dict.csv`` ids already present.  Unmatched players are written to
   ``processed/unmatched_understat.csv``.
3. **Understat per-match stats** — join ``us_xg/us_xa/us_npxg/us_shots/us_key_passes``
   into ``player_match.parquet``.  Source per season: vaastav's per-player understat
   CSVs (``raw/vaastav/{season}/understat/{Name}_{id}.csv`` — shipped upstream for
   2021-22..2024-25) first, falling back to ``raw/understat/{season}/
   player_matches.parquet`` (built by ``understat.to_player_match_frame`` from the
   JSON endpoints).  Matched on understat id + match date against fixture kickoff
   dates (Europe/London calendar day, with a ±1-day second pass).
4. **Odds fixture ids** — resolve ``odds.fpl_fixture_id`` by joining football-data
   rows to fixtures on (season, date, home team, away team) through the
   ``footballdata_name`` crosswalk.

:func:`backfill_clubelo` (network!) extends the ClubElo raw archive to every club in
the crosswalk; it is *not* part of :func:`enrich_all` — the pipeline's backfill stage
calls it explicitly.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd

from fplai import config
from fplai.data import build, crosswalk, vaastav
from fplai.data.clubelo import EloClient, clubelo_url_name

logger = logging.getLogger(__name__)

#: Minimum season minutes for the headline match-rate metric (regular players).
MATCH_RATE_MINUTES = 900

_PLAYER_CSV_RE = re.compile(r"_(\d+)\.csv$")

#: Column mapping from vaastav per-player understat CSVs to the us_* contract.
_VAASTAV_US_COLS: dict[str, str] = {
    "xG": "us_xg",
    "xA": "us_xa",
    "npxG": "us_npxg",
    "shots": "us_shots",
    "key_passes": "us_key_passes",
}


# --- small helpers ------------------------------------------------------------------


def _raw_dir(raw_root: Path | None) -> Path:
    return raw_root if raw_root is not None else config.RAW_DIR


def _local_date(kickoff_utc: pd.Series) -> pd.Series:
    """Europe/London calendar date (naive midnight ns timestamps) of UTC kickoffs."""
    local = kickoff_utc.dt.tz_convert("Europe/London")
    return local.dt.normalize().dt.tz_localize(None).dt.as_unit("ns")


def _read_required(processed: Path, name: str) -> pd.DataFrame:
    path = processed / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing — run build.build_all() first")
    return pd.read_parquet(path)


# --- understat player lists ---------------------------------------------------------


def understat_league_players(season: int, raw_root: Path | None = None) -> pd.DataFrame | None:
    """One season's Understat player list for identity matching, or None if no raw data.

    Prefers ``raw/understat/{season}/league.json`` (clean names) and falls back to
    vaastav's ``understat/understat_player.csv`` (HTML-escaped names).  Returns
    columns ``understat_id`` (int), ``player_name`` (str), ``teams``
    (frozenset[str] of Understat team titles) and ``minutes`` (int).
    """
    raw = _raw_dir(raw_root)
    league_path = raw / "understat" / str(season) / "league.json"
    if league_path.exists():
        league = json.loads(league_path.read_text(encoding="utf-8"))
        rows = [
            {
                "understat_id": int(p["id"]),
                "player_name": str(p["player_name"]),
                "teams": frozenset(str(p.get("team_title", "")).split(",")) - {""},
                "minutes": int(float(p.get("time", 0) or 0)),
            }
            for p in league["players"]
        ]
        return pd.DataFrame(rows).sort_values("understat_id").reset_index(drop=True)

    agg_path = vaastav.season_dir(season, raw_root) / "understat" / "understat_player.csv"
    if agg_path.exists():
        agg = vaastav.read_csv_tolerant(agg_path)
        return pd.DataFrame(
            {
                "understat_id": agg["id"].astype(int),
                "player_name": agg["player_name"].astype(str),
                "teams": [
                    frozenset(str(t).split(",")) - {""} for t in agg["team_title"]
                ],
                "minutes": pd.to_numeric(agg["time"], errors="coerce").fillna(0).astype(int),
            }
        ).sort_values("understat_id").reset_index(drop=True)
    return None


def _season_rosters(
    players: pd.DataFrame, player_match: pd.DataFrame, teams: pd.DataFrame
) -> pd.DataFrame:
    """Per-(season, player_code) matching inputs derived from the processed tables.

    Columns: ``season``, ``player_code``, ``variants`` (list[str] name renderings),
    ``teams`` (frozenset[str] Understat-style titles) and ``minutes`` (season sum).
    """
    title_by_code = {
        (int(s), int(c)): t
        for s, c, t in zip(teams["season"], teams["team_code"], teams["understat_name"])
        if pd.notna(t)
    }
    grouped = (
        player_match.groupby(["season", "player_code"])
        .agg(minutes=("minutes", "sum"), team_codes=("team_code", "unique"))
        .reset_index()
    )
    grouped["teams"] = [
        frozenset(
            t
            for c in codes
            if (t := title_by_code.get((int(season), int(c)))) is not None
        )
        for season, codes in zip(grouped["season"], grouped["team_codes"])
    ]

    def _s(value: object) -> str:
        return "" if pd.isna(value) else str(value)

    names = players.set_index("player_code")[["web_name", "first_name", "second_name"]]
    variants_by_code: dict[int, list[str]] = {}
    for code, row in names.iterrows():
        first, second, web = _s(row["first_name"]), _s(row["second_name"]), _s(row["web_name"])
        raw_variants = [f"{first} {second}".strip(), web, f"{first} {web}".strip()]
        variants_by_code[int(code)] = sorted({v for v in raw_variants if v})
    grouped["variants"] = [
        variants_by_code.get(int(c), []) for c in grouped["player_code"]
    ]
    return grouped[["season", "player_code", "variants", "teams", "minutes"]]


def match_understat_ids(
    players: pd.DataFrame,
    player_match: pd.DataFrame,
    teams: pd.DataFrame,
    raw_root: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Fill ``players.understat_id`` for all seasons with raw Understat data on disk.

    Existing non-null ids (vaastav ``id_dict.csv`` seeds) are kept and never
    overridden; the fuzzy matcher adds the rest season by season, then the results
    are consolidated one-to-one across seasons (best score wins, later seasons break
    ties, id_dict seeds always win).

    Returns:
        ``(players_updated, unmatched_report, stats)`` where ``stats`` carries the
        per-season and headline (>= 900 season minutes) match rates.
    """
    rosters = _season_rosters(players, player_match, teams)
    seeds = {
        int(c): int(u)
        for c, u in zip(players["player_code"], players["understat_id"])
        if pd.notna(u)
    }

    proposals: list[dict[str, Any]] = []
    us_ids_by_season: dict[int, set[int]] = {}
    for season in sorted(rosters["season"].unique()):
        us = understat_league_players(int(season), raw_root)
        if us is None:
            logger.info("understat match: no raw player list for season %d, skipping", season)
            continue
        us_ids_by_season[int(season)] = set(us["understat_id"].astype(int))
        fpl = rosters.loc[rosters["season"] == season, ["player_code", "variants", "teams"]]
        for m in crosswalk.match_players(fpl, us, seeds=seeds):
            proposals.append(
                {
                    "season": int(season),
                    "player_code": m.player_code,
                    "understat_id": m.understat_id,
                    "score": m.score,
                    "method": m.method,
                }
            )

    stats: dict[str, Any] = {"seasons_with_data": sorted(us_ids_by_season)}
    if proposals:
        prop = pd.DataFrame(proposals)
        prop["is_seed"] = prop["method"] == "seed"
        prop = prop.sort_values(
            ["is_seed", "score", "season"], ascending=[False, False, False], kind="mergesort"
        )
        # One id per player, then one player per id (injective both ways).
        prop = prop.drop_duplicates("player_code").drop_duplicates("understat_id")
        chosen = dict(zip(prop["player_code"].astype(int), prop["understat_id"].astype(int)))
    else:
        chosen = {}
    final_map = {**chosen, **seeds}  # seeds always win

    players_updated = players.copy()
    players_updated["understat_id"] = (
        players_updated["player_code"].map(final_map).astype("Int64")
    )

    # Per-season match verification: a matched player's id must exist in that
    # season's Understat list (it always does when the player actually appeared).
    rows = []
    for _, r in rosters.iterrows():
        season = int(r["season"])
        if season not in us_ids_by_season:
            continue
        us_id = final_map.get(int(r["player_code"]))
        rows.append(
            {
                "season": season,
                "player_code": int(r["player_code"]),
                "minutes": int(r["minutes"]),
                "matched": us_id is not None and us_id in us_ids_by_season[season],
            }
        )
    verify = pd.DataFrame(rows)
    if not verify.empty:
        regs = verify[verify["minutes"] >= MATCH_RATE_MINUTES]
        stats["match_rate_900min"] = float(regs["matched"].mean()) if len(regs) else float("nan")
        stats["match_rate_900min_by_season"] = {
            int(s): float(g["matched"].mean())
            for s, g in regs.groupby("season")
        }
        stats["n_players_900min"] = int(len(regs))
        unmatched_keys = verify.loc[~verify["matched"], ["season", "player_code", "minutes"]]
    else:
        unmatched_keys = pd.DataFrame(columns=["season", "player_code", "minutes"])

    name_cols = players.set_index("player_code")[["web_name", "first_name", "second_name"]]
    unmatched = unmatched_keys.merge(
        name_cols, left_on="player_code", right_index=True, how="left"
    ).sort_values(["season", "minutes"], ascending=[True, False]).reset_index(drop=True)
    return players_updated, unmatched, stats


# --- understat per-match join -------------------------------------------------------


def vaastav_understat_matches(
    season: int, raw_root: Path | None = None, understat_titles: set[str] | None = None
) -> pd.DataFrame | None:
    """Parse vaastav per-player understat CSVs for one season, or None if not on disk.

    Files are ``raw/vaastav/{label}/understat/{Name}_{understat_id}.csv`` (one row
    per match across the player's whole career); rows are filtered to this league
    season via the ``season`` column and, when ``understat_titles`` is given, to
    matches involving a Premier League club (drops e.g. a January signing's foreign
    league rows).  Returns ``understat_id``, ``date`` and the ``us_*`` columns.
    """
    us_dir = vaastav.season_dir(season, raw_root) / "understat"
    if not us_dir.is_dir():
        return None
    frames: list[pd.DataFrame] = []
    for path in sorted(us_dir.iterdir()):
        m = _PLAYER_CSV_RE.search(path.name)
        if m is None or path.name.startswith("understat_"):
            continue
        df = vaastav.read_csv_tolerant(path)
        if df.empty or "season" not in df.columns:
            continue
        df = df[pd.to_numeric(df["season"], errors="coerce") == season]
        if understat_titles:
            df = df[df["h_team"].isin(understat_titles) | df["a_team"].isin(understat_titles)]
        if df.empty:
            continue
        out = pd.DataFrame(
            {
                "understat_id": int(m.group(1)),
                "date": pd.to_datetime(df["date"], errors="coerce").dt.normalize(),
            }
        )
        for src, dst in _VAASTAV_US_COLS.items():
            out[dst] = pd.to_numeric(df[src], errors="coerce").astype("Float64")
        frames.append(out)
    if not frames:
        return None
    joined = pd.concat(frames, ignore_index=True).dropna(subset=["date"])
    joined["date"] = joined["date"].dt.as_unit("ns")
    return joined.sort_values(["understat_id", "date"]).reset_index(drop=True)


def understat_parquet_matches(season: int, raw_root: Path | None = None) -> pd.DataFrame | None:
    """Read ``raw/understat/{season}/player_matches.parquet`` (fallback source)."""
    path = _raw_dir(raw_root) / "understat" / str(season) / "player_matches.parquet"
    if not path.exists():
        return None
    frame = pd.read_parquet(path)
    out = frame[["understat_id", "date", "us_xg", "us_xa", "us_npxg", "us_shots",
                 "us_key_passes"]].copy()
    out["date"] = pd.to_datetime(out["date"]).dt.normalize().dt.as_unit("ns")
    for col in build.UNDERSTAT_COLS:
        out[col] = out[col].astype("Float64")
    return out.sort_values(["understat_id", "date"]).reset_index(drop=True)


def _season_us_matches(
    season: int, raw_root: Path | None, understat_titles: set[str]
) -> pd.DataFrame | None:
    """Per-match us_* rows for one season, or None when neither source is on disk.

    vaastav's extracts are the primary source; rows from the JSON-endpoint parquet
    fill in (player, date) pairs vaastav does not cover.
    """
    primary = vaastav_understat_matches(season, raw_root, understat_titles)
    fallback = understat_parquet_matches(season, raw_root)
    if primary is None:
        return fallback
    if fallback is None:
        return primary
    merged = pd.concat([primary, fallback[primary.columns]], ignore_index=True)
    merged = merged.drop_duplicates(["understat_id", "date"], keep="first")
    return merged.sort_values(["understat_id", "date"]).reset_index(drop=True)


def join_understat(
    player_match: pd.DataFrame,
    fixtures: pd.DataFrame,
    players: pd.DataFrame,
    teams: pd.DataFrame,
    raw_root: Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fill the ``us_*`` columns of ``player_match`` from per-match Understat data.

    Join key: player identity via ``players.understat_id``, match identity via the
    fixture's Europe/London kickoff date vs Understat's match date (exact, then a
    ±1-day pass for kickoff/date-rollover stragglers, applied only where
    unambiguous).  2022-23 void-GW7 rows are excluded — they reference rescheduled
    fixtures whose date belongs to the make-up game.  Recomputes from scratch:
    idempotent.

    Returns the updated frame and per-season coverage stats over minutes>0 rows.
    """
    code_by_us_id: dict[int, int] = {
        int(u): int(c)
        for c, u in zip(players["player_code"], players["understat_id"])
        if pd.notna(u)
    }
    titles_by_season: dict[int, set[str]] = {
        int(s): set(g.dropna())
        for s, g in teams.groupby("season")["understat_name"]
    }

    us_frames: list[pd.DataFrame] = []
    for season in sorted(player_match["season"].unique()):
        frame = _season_us_matches(int(season), raw_root, titles_by_season.get(int(season), set()))
        if frame is None:
            logger.info("understat join: no per-match data for season %d", season)
            continue
        frame = frame.copy()
        frame["player_code"] = frame["understat_id"].map(code_by_us_id).astype("Int64")
        frame = frame.dropna(subset=["player_code"])
        frame["player_code"] = frame["player_code"].astype("int64")
        frame.insert(0, "season", int(season))
        us_frames.append(frame[["season", "player_code", "date", *build.UNDERSTAT_COLS]])

    pm = player_match.copy()
    for col in build.UNDERSTAT_COLS:
        pm[col] = pd.Series(pd.NA, index=pm.index, dtype="Float64")
    stats: dict[str, Any] = {"us_coverage_by_season": {}, "seasons_with_data": []}
    if not us_frames:
        return pm, stats

    us_all = pd.concat(us_frames, ignore_index=True)
    us_all = us_all.drop_duplicates(["season", "player_code", "date"])
    stats["seasons_with_data"] = sorted(us_all["season"].unique().tolist())

    kick = fixtures[["season", "fpl_fixture_id", "kickoff_utc"]].copy()
    kick["date_local"] = _local_date(kick["kickoff_utc"])
    pm_key = pm[["season", "fpl_fixture_id", "player_code", "void_gw"]].reset_index()
    pm_key = pm_key.merge(kick[["season", "fpl_fixture_id", "date_local"]],
                          on=["season", "fpl_fixture_id"], how="left")
    pm_key = pm_key[~pm_key["void_gw"] & pm_key["date_local"].notna()]

    hits = pm_key.merge(
        us_all,
        left_on=["season", "player_code", "date_local"],
        right_on=["season", "player_code", "date"],
        how="inner",
    )

    # ±1-day pass for rows still unmatched on both sides (unambiguous only).
    open_pm = pm_key[~pm_key["index"].isin(hits["index"])]
    matched_us = set(zip(hits["season"], hits["player_code"], hits["date"]))
    open_us = us_all[
        ~pd.Series(list(zip(us_all["season"], us_all["player_code"], us_all["date"])),
                   index=us_all.index).isin(matched_us)
    ]
    near_frames = []
    for offset in (pd.Timedelta(days=1), pd.Timedelta(days=-1)):
        shifted = open_us.copy()
        shifted["date_local"] = shifted["date"] + offset
        near_frames.append(
            open_pm.merge(shifted, on=["season", "player_code", "date_local"], how="inner")
        )
    near = pd.concat(near_frames, ignore_index=True)
    near = near[~near.duplicated("index", keep=False)]
    near = near[~near.duplicated(["season", "player_code", "date"], keep=False)]

    filled = pd.concat([hits, near], ignore_index=True)
    filled = filled[~filled.duplicated("index", keep=False)].set_index("index")
    for col in build.UNDERSTAT_COLS:
        pm.loc[filled.index, col] = filled[col].astype("Float64")

    played = pm[(pm["minutes"] > 0) & ~pm["void_gw"]]
    for season, g in played.groupby("season"):
        stats["us_coverage_by_season"][int(season)] = float(g["us_xg"].notna().mean())
    return pm, stats


# --- odds -> fixture join -----------------------------------------------------------


def join_odds(
    odds: pd.DataFrame, fixtures: pd.DataFrame, teams: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Resolve ``odds.fpl_fixture_id`` via (season, date, home, away) against fixtures.

    Team identity goes through the ``footballdata_name`` column of the teams
    crosswalk.  The primary join is on the exact Europe/London kickoff date; odds
    rows still unresolved get a ±1-day pass (kick-off moved across midnight or a
    source recorded the wrong day) when the (season, home, away) pairing is unique.
    Recomputes from scratch: idempotent.
    """
    code_by_fd = {
        (int(s), str(n)): int(c)
        for s, n, c in zip(teams["season"], teams["footballdata_name"], teams["team_code"])
        if pd.notna(n)
    }
    out = odds.copy()
    seasons = out["season"].astype(int)
    out["_home_code"] = pd.Series(
        [code_by_fd.get((s, str(n))) for s, n in zip(seasons, out["home_footballdata_name"])],
        index=out.index, dtype="Int64",
    )
    out["_away_code"] = pd.Series(
        [code_by_fd.get((s, str(n))) for s, n in zip(seasons, out["away_footballdata_name"])],
        index=out.index, dtype="Int64",
    )

    fx = fixtures[["season", "fpl_fixture_id", "home_team_code", "away_team_code",
                   "kickoff_utc"]].copy()
    fx["season"] = fx["season"].astype("Int64")
    fx["home_team_code"] = fx["home_team_code"].astype("Int64")
    fx["away_team_code"] = fx["away_team_code"].astype("Int64")
    fx["date_local"] = _local_date(fx["kickoff_utc"])
    fx = fx.dropna(subset=["date_local"])

    left = pd.DataFrame(
        {
            "_row": out.index,
            "season": out["season"].astype("Int64"),
            "date_local": pd.to_datetime(out["date"]).dt.normalize().dt.as_unit("ns"),
            "home_team_code": out["_home_code"],
            "away_team_code": out["_away_code"],
        }
    )
    exact_keys = ["season", "date_local", "home_team_code", "away_team_code"]
    exact_fx = fx.drop_duplicates(subset=exact_keys, keep=False)
    hit = left.merge(exact_fx[[*exact_keys, "fpl_fixture_id"]], on=exact_keys, how="inner")
    resolved = pd.Series(pd.NA, index=out.index, dtype="Int64")
    resolved.loc[hit["_row"]] = hit["fpl_fixture_id"].astype("Int64").to_numpy()

    # ±1-day fallback for the rest: same (season, home, away) pairing within a day of
    # the odds date, kept only when unambiguous on both sides and the fixture is not
    # already claimed by the exact pass.
    pair_keys = ["season", "home_team_code", "away_team_code"]
    open_left = left[~left["_row"].isin(hit["_row"])]
    open_fx = fx[~fx["fpl_fixture_id"].isin(hit["fpl_fixture_id"])]
    near = open_left.merge(
        open_fx[[*pair_keys, "date_local", "fpl_fixture_id"]],
        on=pair_keys, how="inner", suffixes=("", "_fx"),
    )
    near = near[(near["date_local"] - near["date_local_fx"]).abs() <= pd.Timedelta(days=1)]
    near = near[~near.duplicated("_row", keep=False)]
    near = near[~near.duplicated("fpl_fixture_id", keep=False)]
    resolved.loc[near["_row"]] = near["fpl_fixture_id"].astype("Int64").to_numpy()

    out["fpl_fixture_id"] = resolved
    out = out.drop(columns=["_home_code", "_away_code"])
    fixture_seasons = set(fixtures["season"].astype(int).unique())
    stats: dict[str, Any] = {"odds_join_rate_by_season": {}}
    for season, g in out.groupby("season"):
        if int(season) in fixture_seasons:
            stats["odds_join_rate_by_season"][int(season)] = float(
                g["fpl_fixture_id"].notna().mean()
            )
    return out, stats


# --- raw-source backfills (network) -------------------------------------------------


def download_vaastav_understat_players(
    seasons: Iterable[int] | None = None,
    *,
    raw_root: Path | None = None,
    fetch: vaastav.FetchFn | None = None,
) -> dict[int, int]:
    """Download vaastav's per-player understat CSVs for seasons that ship them (network!).

    Upstream layout: ``data/{label}/understat/{Name}_{understat_id}.csv`` where
    ``Name`` is the raw Understat player name with spaces replaced by underscores
    (HTML entities and accents preserved: ``N&#039;Golo_Kanté_751.csv``).  Filenames
    are reconstructed from the season's ``understat_player.csv`` aggregate, so that
    file must already be on disk (``vaastav.download_season``).  Seasons without the
    aggregate (2016-17..2018-19, 2025-26 — no understat dir upstream) and players
    404ing upstream are skipped with a log.  Skip-if-exists per file: idempotent.

    Returns a mapping of season -> number of files newly downloaded.
    """
    from urllib.parse import quote

    fetch = fetch or vaastav._polite_fetch
    chosen = crosswalk.resolve_seasons(seasons, raw_root)
    downloaded: dict[int, int] = {}
    for season in chosen:
        us_dir = vaastav.season_dir(season, raw_root) / "understat"
        agg_path = us_dir / "understat_player.csv"
        if not agg_path.exists():
            logger.info("vaastav understat players %d: no understat_player.csv, skipping",
                        season)
            continue
        agg = vaastav.read_csv_tolerant(agg_path)
        label = vaastav.season_label(season)
        n_new = 0
        for name, us_id in zip(agg["player_name"], agg["id"], strict=True):
            fname = f"{str(name).replace(' ', '_')}_{int(us_id)}.csv"
            dest = us_dir / fname
            if dest.exists():
                continue
            url = f"{vaastav.VAASTAV_BASE_URL}/{label}/understat/{quote(fname)}"
            try:
                content = fetch(url)
            except FileNotFoundError:
                logger.info("vaastav understat players %d: %s missing upstream", season, fname)
                continue
            tmp = dest.with_name(dest.name + ".part")
            tmp.write_bytes(content)
            tmp.replace(dest)
            n_new += 1
        downloaded[season] = n_new
        logger.info("vaastav understat players %d: %d new files", season, n_new)
    return downloaded


# --- clubelo backfill (network) -----------------------------------------------------


def backfill_clubelo(
    teams: pd.DataFrame | None = None,
    *,
    raw_dir: Path | None = None,
    client: EloClient | None = None,
) -> list[str]:
    """Archive ClubElo history for every crosswalk club missing a raw CSV (network!).

    Covers clubs relegated before the current season (West Brom, Huddersfield, ...)
    that the snapshot-driven backfill skips.  Skip-if-exists per club; returns the
    club names actually fetched.
    """
    if teams is None:
        teams = _read_required(config.PROCESSED_DIR, "teams")
    if "clubelo_name" not in teams.columns or teams["clubelo_name"].isna().all():
        teams = crosswalk.apply_team_name_maps(teams)
    dest_dir = raw_dir if raw_dir is not None else config.RAW_DIR / "clubelo"
    client = client if client is not None else EloClient(raw_dir=dest_dir)
    fetched: list[str] = []
    for name in sorted(teams["clubelo_name"].dropna().unique()):
        if (dest_dir / f"{clubelo_url_name(name)}.csv").exists():
            continue
        client.team_history(name)
        fetched.append(name)
        logger.info("clubelo backfill: archived %s", name)
    return fetched


# --- top level ----------------------------------------------------------------------


def enrich_all(
    processed_dir: Path | None = None, raw_root: Path | None = None
) -> dict[str, Any]:
    """Run every enrichment step over the processed tables; write results in place.

    Reads ``teams/players/fixtures/player_match`` (required) and ``odds`` (optional)
    from ``processed_dir``, enriches them from whatever raw data exists under
    ``raw_root``, rewrites the parquet files, and writes the unmatched-players
    report to ``processed/unmatched_understat.csv``.  Fully offline, deterministic,
    idempotent and partial-season safe.

    Returns a stats dict: understat match rates, us_* coverage per season and odds
    join rates per season.
    """
    processed = processed_dir if processed_dir is not None else config.PROCESSED_DIR
    teams = _read_required(processed, "teams")
    players = _read_required(processed, "players")
    fixtures = _read_required(processed, "fixtures")
    player_match = _read_required(processed, "player_match")

    teams = crosswalk.apply_team_name_maps(teams)
    teams.to_parquet(processed / "teams.parquet", index=False)

    players, unmatched, match_stats = match_understat_ids(
        players, player_match, teams, raw_root
    )
    players.to_parquet(processed / "players.parquet", index=False)
    unmatched.to_csv(processed / "unmatched_understat.csv", index=False)

    player_match, us_stats = join_understat(player_match, fixtures, players, teams, raw_root)
    player_match[list(build.PLAYER_MATCH_COLS)].to_parquet(
        processed / "player_match.parquet", index=False
    )

    stats: dict[str, Any] = {"understat_match": match_stats, "understat_join": us_stats}
    odds_path = processed / "odds.parquet"
    if odds_path.exists():
        odds, odds_stats = join_odds(pd.read_parquet(odds_path), fixtures, teams)
        odds.to_parquet(odds_path, index=False)
        stats["odds"] = odds_stats
    else:
        logger.info("enrich: no odds.parquet at %s, skipping odds join", odds_path)

    logger.info("enrich_all complete: %s", stats)
    return stats
