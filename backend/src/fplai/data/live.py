"""Bootstrap-grounded live layer for the 2026-27 season.

Once the FPL API serves 2026-27 (a snapshot directory whose ``bootstrap.json``
parses to ``season >= 2026``), live prediction must stop rolling rosters
forward from last-season appearances and instead ground everything in the day's
bootstrap: real squads (promoted-team players, summer signings), real launch
prices, the 2026-27 position reclassifications (free via ``element_type``),
availability flags and the 2026-27 fixture calendar.

This module owns that grounding:

* :func:`load_live_context` — parse the newest live snapshot into a
  :class:`LiveContext` (roster / teams / fixtures frames conforming to the
  ARCHITECTURE parquet schemas).
* :func:`bootstrap_player_match` — synthesize future ``player_match`` rows for
  upcoming fixtures from the bootstrap roster (replaces the roll-forward
  synthesizer in live mode).
* :func:`augment_tables` / :func:`persist_live_tables` — splice the live
  season's fixtures/teams/players (and odds) into the processed tables,
  in-memory and on disk.
* :class:`RatePriors`, :func:`live_minutes_predict`,
  :func:`apply_cold_start_rates` — the cold-start policy (below).
* :func:`match_fatigue_watchlist` / :func:`build_adjustments` — the WC2026
  fatigue dampening list (FPL_KNOWLEDGE §3.3) fed into the minutes model's
  ``fatigue_dampening`` hook for GWs 1-4.
* :func:`seed_team_model` — promoted-team strength seeding (§3.6) through the
  TeamModel's ``promoted_strength`` hook plus new-manager uncertainty
  shrinkage (§3.5 / MODEL_DESIGN_INPUTS §1.2).
* :func:`fetch_live_odds` — one cached the-odds-api snapshot (h2h + totals)
  mapped onto the odds-frame shape the TeamModel blend consumes.

Cold-start policy (documented contract)
---------------------------------------
A *cold-start* player is a bootstrap-roster ``player_code`` with **no**
``player_match`` history at all (promoted-team players, new signings from
abroad, academy debutants).  Players who merely changed clubs keep their full
history — features are keyed by ``player_code`` and form travels with the
player, while team/opponent context comes from the NEW club's fixtures.  For
cold-start rows the models' NaN-driven outputs are replaced with explicit
priors:

* **minutes** — :class:`~fplai.models.minutes.HeuristicMinutesModel`, whose
  no-history fallback is the calibrated ``DEFAULT_SHARES`` table by
  position x price band (a cheap promoted GKP starts far less often than a
  premium one);
* **per-90 rates** — pooled empirical rates by position x price decile
  (:class:`RatePriors`, fitted on modern-era history with mild shrinkage to
  the position rate), replacing ``lam_goal/lam_assist/lam_saves/lam_defcon``;
  cards/own-goals already fall back to position priors inside ``RatesModel``.

Availability (status/news/chance_of_playing from the bootstrap — our first
deadline-timestamped signal) additionally gates the minutes distribution for
*all* live rows: flagged players lose q1/q2 mass to q0 in the next GW, with a
linear recovery over the following 4 GWs (injuries heal; the flag is a
snapshot, not a season sentence).
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from fplai import config
from fplai.data import crosswalk
from fplai.data.fpl_api import parse_season_state

if TYPE_CHECKING:
    from fplai.models.team import TeamModel

logger = logging.getLogger(__name__)

#: First season the live layer applies to (the relaunch this module was built for).
LIVE_FROM_SEASON = 2026

#: GWs 1-4 get WC2026 fatigue dampening (FPL_KNOWLEDGE §3.3).
FATIGUE_GWS: tuple[int, ...] = (1, 2, 3, 4)
#: q2 multiplier for the highest-load WC2026 players (played until 14-19 July).
FATIGUE_HIGH = 0.85
#: q2 multiplier for the lighter-load flags (§3.3 "Lighter Spain loads").
FATIGUE_LIGHT = 0.95

#: Minimum name-similarity score to accept a watchlist -> roster match.
FATIGUE_MATCH_THRESHOLD = 0.72

#: FPL_KNOWLEDGE §3.3 WC2026 fatigue watchlist: (display name, allowed FPL club
#: short names, q2 dampening).  Names are matched fuzzily against the bootstrap
#: roster (web_name + full name) within the listed club(s); misses are reported
#: in ``fatigue_2026.json`` rather than raised (squads move until 1 Sep).
WC2026_FATIGUE_WATCHLIST: tuple[tuple[str, frozenset[str], float], ...] = (
    ("Rodri", frozenset({"MCI"}), FATIGUE_HIGH),
    ("Pedro Porro", frozenset({"TOT"}), FATIGUE_HIGH),
    ("Mikel Merino", frozenset({"ARS"}), FATIGUE_HIGH),
    ("E. Martínez", frozenset({"AVL"}), FATIGUE_HIGH),
    ("Romero", frozenset({"TOT"}), FATIGUE_HIGH),
    ("L. Martínez", frozenset({"MUN"}), FATIGUE_HIGH),
    ("Enzo Fernández", frozenset({"CHE"}), FATIGUE_HIGH),
    ("Mac Allister", frozenset({"LIV"}), FATIGUE_HIGH),
    ("Rice", frozenset({"ARS"}), FATIGUE_HIGH),
    ("Saka", frozenset({"ARS"}), FATIGUE_HIGH),
    ("Eze", frozenset({"ARS"}), FATIGUE_HIGH),
    ("Madueke", frozenset({"ARS"}), FATIGUE_HIGH),
    ("Guéhi", frozenset({"MCI"}), FATIGUE_HIGH),
    ("Stones", frozenset({"MCI"}), FATIGUE_HIGH),
    ("O'Reilly", frozenset({"MCI"}), FATIGUE_HIGH),
    ("Trafford", frozenset({"MCI"}), FATIGUE_HIGH),
    ("Elliot Anderson", frozenset({"MCI"}), FATIGUE_HIGH),
    ("Pickford", frozenset({"EVE"}), FATIGUE_HIGH),
    ("Rashford", frozenset({"MUN"}), FATIGUE_HIGH),
    ("Mainoo", frozenset({"MUN"}), FATIGUE_HIGH),
    ("Watkins", frozenset({"AVL"}), FATIGUE_HIGH),
    ("Konsa", frozenset({"AVL"}), FATIGUE_HIGH),
    ("Burn", frozenset({"NEW"}), FATIGUE_HIGH),
    ("Chalobah", frozenset({"CHE"}), FATIGUE_HIGH),
    ("Reece James", frozenset({"CHE"}), FATIGUE_HIGH),
    ("Spence", frozenset({"TOT"}), FATIGUE_HIGH),
    ("Dean Henderson", frozenset({"CRY"}), FATIGUE_HIGH),
    ("Jordan Henderson", frozenset({"BRE"}), FATIGUE_HIGH),
    # §3.4: Rogers AVL->CHE reported — accept either club.
    ("Morgan Rogers", frozenset({"AVL", "CHE"}), FATIGUE_HIGH),
    ("Konaté", frozenset({"LIV"}), FATIGUE_HIGH),
    ("Digne", frozenset({"AVL"}), FATIGUE_HIGH),
    ("Gusto", frozenset({"CHE"}), FATIGUE_HIGH),
    ("Lacroix", frozenset({"CRY"}), FATIGUE_HIGH),
    ("Mateta", frozenset({"CRY"}), FATIGUE_HIGH),
    ("Cherki", frozenset({"MCI"}), FATIGUE_HIGH),
    ("Zubimendi", frozenset({"ARS"}), FATIGUE_LIGHT),
    ("Raya", frozenset({"ARS"}), FATIGUE_LIGHT),
    ("Muñoz", frozenset({"LIV"}), FATIGUE_LIGHT),
)

#: FPL_KNOWLEDGE §3.6 promoted-club strength multipliers over the TeamModel's
#: promoted prior (weight Coventry > Ipswich > Hull; Championship-xG-informed
#: ordering, not a flat relegation-fodder constant).  Keys are bootstrap names.
PROMOTED_STRENGTH_2026: dict[str, float] = {
    "Coventry City": 1.08,
    "Ipswich Town": 1.00,
    "Hull City": 0.92,
}

#: FPL_KNOWLEDGE §3.5 clubs with a new manager vs the fitted history — their
#: fitted ratings are shrunk toward the league mean (style uncertainty).
#: Ipswich Town appears in the promoted set and is seeded there instead.
NEW_MANAGER_TEAMS_2026: tuple[str, ...] = (
    "Chelsea",
    "Man City",
    "Liverpool",
    "Nott'm Forest",
    "Spurs",
    "Crystal Palace",
    "Bournemouth",
    "Fulham",
    "Man Utd",
    "Ipswich Town",
)

#: Shrinkage factor for new-manager clubs: ratings move this fraction toward
#: the league mean (a point-estimate proxy for a widened predictive interval).
NEW_MANAGER_SHRINK = 0.15

#: the-odds-api event team names -> FPL bootstrap names (non-identity cases).
ODDS_API_TEAM_NAMES: dict[str, str] = {
    "AFC Bournemouth": "Bournemouth",
    "Brighton and Hove Albion": "Brighton",
    "Leeds United": "Leeds",
    "Leicester City": "Leicester",
    "Luton Town": "Luton",
    "Manchester City": "Man City",
    "Manchester United": "Man Utd",
    "Newcastle United": "Newcastle",
    "Nottingham Forest": "Nott'm Forest",
    "Sheffield United": "Sheffield Utd",
    "Tottenham Hotspur": "Spurs",
    "West Bromwich Albion": "West Brom",
    "West Ham United": "West Ham",
    "Wolverhampton Wanderers": "Wolves",
}

#: Availability factor for out-flagged players (status i/s/u) with no
#: chance_of_playing percentage from the API.
_OUT_NO_CHANCE_FACTOR = 0.05
#: Availability factor for doubtful players (status d) with no percentage.
_DOUBT_NO_CHANCE_FACTOR = 0.5
#: GWs over which an availability flag linearly recovers to full fitness.
_AVAILABILITY_RECOVERY_GWS = 4

#: Minutes factor in a dated injury's RETURN GW (sharpness/bench risk on comeback).
_RETURN_GW_FACTOR = 0.65
#: The FPL news grammar's dated fragments (verified live 2026-08-11):
#: "<Type> injury - Expected back 23 Aug" / "Suspended until 6 Sep".
_NEWS_RETURN_RE = re.compile(
    r"(?P<kind>Expected back|Suspended until)\s+(?P<day>\d{1,2})\s+(?P<mon>[A-Za-z]{3})",
    re.IGNORECASE,
)
_MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}

_MINUTES_KEYS = ("season", "gw", "player_code", "fpl_fixture_id")
_Q_COLUMNS = ("q0", "q1", "q2", "mu1", "mu2")


# --------------------------------------------------------------------------------------
# Snapshot parsing
# --------------------------------------------------------------------------------------


@dataclass
class LiveContext:
    """Parsed live snapshot: everything the pipeline needs for a live season."""

    snap_dir: Path
    season: int
    next_gw: int | None
    roster: pd.DataFrame
    """One row per bootstrap element (players only) — see :func:`parse_roster`."""
    teams: pd.DataFrame
    """teams.parquet-schema rows for ``season`` with aux source names filled."""
    fixtures: pd.DataFrame
    """fixtures.parquet-schema rows for ``season``."""


def latest_live_snapshot_dir(snapshots_root: Path | None = None) -> Path | None:
    """Newest dated snapshot dir whose bootstrap parses to a live (>= 2026) season.

    Returns ``None`` when no snapshot on disk is from the relaunched game (the
    caller then stays in roll-forward / pre-launch behavior).
    """
    root = (
        Path(snapshots_root)
        if snapshots_root is not None
        else config.RAW_DIR / "fpl_api" / "snapshots"
    )
    if not root.exists():
        return None
    for snap_dir in sorted((d for d in root.iterdir() if d.is_dir()), reverse=True):
        bootstrap_path = snap_dir / "bootstrap.json"
        if not bootstrap_path.exists() or not (snap_dir / "fixtures.json").exists():
            continue
        try:
            state = parse_season_state(json.loads(bootstrap_path.read_text()))
        except (ValueError, json.JSONDecodeError) as exc:
            logger.warning("skipping unparseable snapshot %s: %s", snap_dir, exc)
            continue
        if state.season >= LIVE_FROM_SEASON:
            return snap_dir
    return None


def load_live_context(
    snapshots_root: Path | None = None, *, snap_dir: Path | None = None
) -> LiveContext | None:
    """Load the newest live snapshot (or an explicit ``snap_dir``) into a context.

    Returns ``None`` when no live snapshot exists.  Raises on a snapshot that
    exists but cannot be parsed (a real schema problem the caller must see).
    """
    if snap_dir is None:
        snap_dir = latest_live_snapshot_dir(snapshots_root)
        if snap_dir is None:
            return None
    bootstrap = json.loads((snap_dir / "bootstrap.json").read_text())
    fixtures_json = json.loads((snap_dir / "fixtures.json").read_text())
    state = parse_season_state(bootstrap)
    season = int(state.season)
    return LiveContext(
        snap_dir=snap_dir,
        season=season,
        next_gw=state.next_gw,
        roster=parse_roster(bootstrap),
        teams=parse_live_teams(bootstrap, season),
        fixtures=parse_live_fixtures(bootstrap, fixtures_json, season),
    )


def _team_code_by_id(bootstrap: dict[str, Any]) -> dict[int, int]:
    return {int(t["id"]): int(t["code"]) for t in bootstrap["teams"]}


def _validate_element_types(bootstrap: dict[str, Any]) -> None:
    """Guard: the 2026-27 element_type ids must still mean GKP/DEF/MID/FWD."""
    expected = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}
    for et in bootstrap.get("element_types", []):
        et_id = int(et["id"])
        short = str(et.get("singular_name_short", ""))
        if et_id in expected and short and short != expected[et_id]:
            raise ValueError(
                f"element_type {et_id} is {short!r}, expected {expected[et_id]!r} — "
                "the FPL position scheme changed; re-verify rules before predicting"
            )


def parse_roster(bootstrap: dict[str, Any]) -> pd.DataFrame:
    """One row per bootstrap player element (assistant managers excluded).

    Columns: ``player_code`` (= ``element.code``, THE stable key),
    ``fpl_element_id`` (per-season id), ``web_name``, ``first_name``,
    ``second_name``, ``position`` (from ``element_type`` — the 2026-27
    reclassifications come for free), ``team_code``, ``fpl_team_id``, ``price``
    (``now_cost``, 0.1m units), ``status``, ``news``, ``chance_of_playing``
    (next round, falling back to this round), ``selected_by_percent``,
    ``can_select`` and ``opta_code``.
    """
    _validate_element_types(bootstrap)
    code_by_id = _team_code_by_id(bootstrap)
    rows = [e for e in bootstrap["elements"] if int(e["element_type"]) <= 4]
    if not rows:
        raise ValueError("bootstrap has no player elements")

    def _chance(e: dict[str, Any]) -> float | None:
        for key in ("chance_of_playing_next_round", "chance_of_playing_this_round"):
            if e.get(key) is not None:
                return float(e[key])
        return None

    out = pd.DataFrame(
        {
            "player_code": [int(e["code"]) for e in rows],
            "fpl_element_id": [int(e["id"]) for e in rows],
            "web_name": pd.array([str(e["web_name"]) for e in rows], dtype="string"),
            "first_name": pd.array([str(e.get("first_name", "")) for e in rows], dtype="string"),
            "second_name": pd.array([str(e.get("second_name", "")) for e in rows], dtype="string"),
            "position": pd.array(
                [crosswalk.ELEMENT_TYPE_TO_POSITION[int(e["element_type"])] for e in rows],
                dtype="string",
            ),
            "team_code": [int(e["team_code"]) for e in rows],
            "fpl_team_id": [int(e["team"]) for e in rows],
            "price": [int(e["now_cost"]) for e in rows],
            "status": pd.array([str(e.get("status", "a")) for e in rows], dtype="string"),
            "news": pd.array([str(e.get("news", "") or "") for e in rows], dtype="string"),
            "chance_of_playing": pd.array([_chance(e) for e in rows], dtype="Float64"),
            "selected_by_percent": pd.array(
                [
                    float(e["selected_by_percent"])
                    if e.get("selected_by_percent") is not None
                    else None
                    for e in rows
                ],
                dtype="Float64",
            ),
            "can_select": [bool(e.get("can_select", True)) for e in rows],
            "opta_code": pd.array(
                [e.get("opta_code") for e in rows], dtype="string"
            ),
            # 2026-08 additions: injury-news timestamp (return-date year inference),
            # set-piece duty orders (Scout-informed editorial data shipped IN the
            # bootstrap) and the official price-predictor progress field.
            "news_added": pd.array(
                [e.get("news_added") for e in rows], dtype="string"
            ),
            "penalties_order": pd.array(
                [e.get("penalties_order") for e in rows], dtype="Int64"
            ),
            "penalties_text": pd.array(
                [str(e.get("penalties_text", "") or "") for e in rows], dtype="string"
            ),
            "direct_freekicks_order": pd.array(
                [e.get("direct_freekicks_order") for e in rows], dtype="Int64"
            ),
            "direct_freekicks_text": pd.array(
                [str(e.get("direct_freekicks_text", "") or "") for e in rows], dtype="string"
            ),
            "corners_and_indirect_freekicks_order": pd.array(
                [e.get("corners_and_indirect_freekicks_order") for e in rows], dtype="Int64"
            ),
            "corners_and_indirect_freekicks_text": pd.array(
                [str(e.get("corners_and_indirect_freekicks_text", "") or "") for e in rows],
                dtype="string",
            ),
            "price_change_percent": pd.array(
                [
                    float(e["price_change_percent"])
                    if e.get("price_change_percent") not in (None, "")
                    else None
                    for e in rows
                ],
                dtype="Float64",
            ),
        }
    )
    mismatched = out["team_code"].to_numpy() != out["fpl_team_id"].map(code_by_id).to_numpy()
    if mismatched.any():
        raise ValueError(
            f"{int(mismatched.sum())} elements have team_code inconsistent with their "
            "team id — bootstrap teams block changed shape?"
        )
    return out.sort_values("player_code").reset_index(drop=True)


def parse_live_teams(bootstrap: dict[str, Any], season: int) -> pd.DataFrame:
    """teams.parquet-schema rows for the live season, aux source names filled.

    Aux names come from the deterministic crosswalk maps; newly promoted clubs
    absent from a map are logged loudly by ``apply_team_name_maps`` (extend the
    maps rather than letting a silent NA propagate).
    """
    teams = pd.DataFrame(
        {
            "season": season,
            "fpl_team_id": [int(t["id"]) for t in bootstrap["teams"]],
            "team_code": [int(t["code"]) for t in bootstrap["teams"]],
            "name": pd.array([str(t["name"]) for t in bootstrap["teams"]], dtype="string"),
            "short_name": pd.array(
                [str(t["short_name"]) for t in bootstrap["teams"]], dtype="string"
            ),
        }
    )
    for col in ("understat_name", "clubelo_name", "footballdata_name"):
        teams[col] = pd.Series(pd.NA, index=teams.index, dtype="string")
    teams = crosswalk.apply_team_name_maps(teams)
    return teams.sort_values("fpl_team_id").reset_index(drop=True)


def parse_live_fixtures(
    bootstrap: dict[str, Any], fixtures_json: list[dict[str, Any]], season: int
) -> pd.DataFrame:
    """fixtures.parquet-schema rows for the live season.

    ``gw`` comes from the API ``event`` (no remaps apply to 2026-27); team ids
    are resolved to stable codes via the bootstrap teams block.  Fixtures with
    no scheduled event yet (postponements) are dropped with a warning — they
    re-enter once the API assigns them a GW.
    """
    code_by_id = _team_code_by_id(bootstrap)
    scheduled = [f for f in fixtures_json if f.get("event") is not None]
    dropped = len(fixtures_json) - len(scheduled)
    if dropped:
        logger.warning("dropping %d unscheduled fixtures (event=null)", dropped)
    if not scheduled:
        raise ValueError("no scheduled fixtures in the live snapshot")
    out = pd.DataFrame(
        {
            "season": season,
            "gw": pd.array([int(f["event"]) for f in scheduled], dtype="Int64"),
            "fpl_fixture_id": [int(f["id"]) for f in scheduled],
            "kickoff_utc": pd.to_datetime(
                [f.get("kickoff_time") for f in scheduled], utc=True
            ).as_unit("ns"),
            "home_team_code": [code_by_id[int(f["team_h"])] for f in scheduled],
            "away_team_code": [code_by_id[int(f["team_a"])] for f in scheduled],
            "home_goals": pd.array([f.get("team_h_score") for f in scheduled], dtype="Int64"),
            "away_goals": pd.array([f.get("team_a_score") for f in scheduled], dtype="Int64"),
            "finished": [bool(f.get("finished", False)) for f in scheduled],
            "void": False,
        }
    )
    return (
        out.sort_values(["season", "gw", "fpl_fixture_id"])
        .reset_index(drop=True)
    )


# --------------------------------------------------------------------------------------
# Table augmentation + persistence
# --------------------------------------------------------------------------------------


def _replace_season_rows(
    existing: pd.DataFrame, fresh: pd.DataFrame, season: int
) -> pd.DataFrame:
    """Drop ``season`` rows from ``existing`` and append ``fresh`` (idempotent)."""
    keep = existing[existing["season"] != season]
    return pd.concat([keep, fresh], ignore_index=True)


def _merged_players(players: pd.DataFrame, roster: pd.DataFrame) -> pd.DataFrame:
    """players.parquet with roster identities merged in.

    Existing rows keep their ``understat_id``; names refresh to the bootstrap
    spelling; codes unseen in any historical season are appended (new signings,
    promoted-team players).
    """
    players = players.copy()
    name_cols = ("web_name", "first_name", "second_name")
    maps = {c: dict(zip(roster["player_code"], roster[c], strict=True)) for c in name_cols}
    mask = players["player_code"].isin(set(roster["player_code"]))
    for col in name_cols:
        players.loc[mask, col] = players.loc[mask, "player_code"].map(maps[col])
    new = roster[~roster["player_code"].isin(set(players["player_code"]))]
    if len(new):
        add = new[["player_code", *name_cols, "opta_code"]].copy()
        add["understat_id"] = pd.Series(pd.NA, index=add.index, dtype="Int64")
        players = pd.concat([players, add[list(players.columns)]], ignore_index=True)
    return players.sort_values("player_code").reset_index(drop=True)


def augment_tables(
    tables: dict[str, pd.DataFrame],
    ctx: LiveContext,
    *,
    odds: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    """Splice the live season into the processed tables (returns a new dict).

    ``fixtures`` / ``teams`` get their live-season rows replaced from the
    snapshot; ``players`` gains new roster identities; ``odds`` (optional live
    odds frame) replaces any live-season odds rows.  ``player_match`` /
    ``player_gw`` are untouched — history stays history; future rows are
    synthesized separately by :func:`bootstrap_player_match`.
    """
    out = dict(tables)
    out["fixtures"] = (
        _replace_season_rows(tables["fixtures"], ctx.fixtures, ctx.season)
        .sort_values(["season", "gw", "fpl_fixture_id"])
        .reset_index(drop=True)
    )
    out["teams"] = (
        _replace_season_rows(tables["teams"], ctx.teams, ctx.season)
        .sort_values(["season", "fpl_team_id"])
        .reset_index(drop=True)
    )
    out["players"] = _merged_players(tables["players"], ctx.roster)
    if odds is not None and len(odds):
        base = tables.get("odds")
        out["odds"] = (
            odds.reset_index(drop=True)
            if base is None or not len(base)
            else _replace_season_rows(base, odds, ctx.season).reset_index(drop=True)
        )
    return out


def persist_live_tables(
    ctx: LiveContext,
    tables: dict[str, pd.DataFrame],
    processed_dir: Path | None = None,
) -> dict[str, Path]:
    """Write the live-augmented fixtures/teams/players back to parquet, plus the
    roster snapshot (``live_roster.parquet``) the optimizer prices from.

    The processed tables are rebuilt from raw sources by ``fplai build``, which
    knows nothing about the live season; persisting here keeps downstream
    stages (optimize's live detection, the API's freshness/fixture routes)
    consistent with what predict just used.  Idempotent — live-season rows are
    replaced, never duplicated.
    """
    processed = Path(processed_dir) if processed_dir is not None else config.PROCESSED_DIR
    processed.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for name in ("fixtures", "teams", "players"):
        paths[name] = processed / f"{name}.parquet"
        tables[name].to_parquet(paths[name], index=False)
    roster = ctx.roster.copy()
    roster.insert(0, "season", ctx.season)
    paths["live_roster"] = processed / "live_roster.parquet"
    roster.to_parquet(paths["live_roster"], index=False)
    return paths


# --------------------------------------------------------------------------------------
# Future player_match synthesis + availability
# --------------------------------------------------------------------------------------


def _selectable_roster(roster: pd.DataFrame) -> pd.DataFrame:
    """Roster rows that can actually appear: drops departed/unselectable players."""
    keep = (roster["status"].astype(str) != "u") & roster["can_select"].astype(bool)
    return roster.loc[keep]


def bootstrap_player_match(
    ctx: LiveContext, target_fx: pd.DataFrame, pm: pd.DataFrame
) -> pd.DataFrame:
    """Synthesize ``player_match`` rows for unplayed live fixtures from the roster.

    One row per (fixture, rostered player of either side) with the bootstrap's
    position/price/element id and the fixture's team/opponent/venue context.
    Outcome columns are NaN — the feature builder ignores them, so the rows
    contribute nothing to anyone's history.  Unlike the roll-forward
    synthesizer this sees promoted-team players, summer signings and real
    prices, and never resurrects players who left the league.
    """
    from fplai import rules

    roster = _selectable_roster(ctx.roster)
    flags = rules.SEASON_FLAGS.get(ctx.season)
    frames: list[pd.DataFrame] = []
    for fx_row in target_fx.itertuples():
        for team_code, opponent_code, was_home in (
            (fx_row.home_team_code, fx_row.away_team_code, True),
            (fx_row.away_team_code, fx_row.home_team_code, False),
        ):
            side = roster[roster["team_code"] == int(team_code)]
            if side.empty:
                continue
            frames.append(
                pd.DataFrame(
                    {
                        "season": int(fx_row.season),
                        "gw": int(fx_row.gw),
                        "fpl_fixture_id": int(fx_row.fpl_fixture_id),
                        "player_code": side["player_code"].to_numpy(),
                        "fpl_element_id": side["fpl_element_id"].to_numpy(),
                        "team_code": int(team_code),
                        "opponent_code": int(opponent_code),
                        "was_home": was_home,
                        "position": side["position"].astype(str).to_numpy(),
                        "price": side["price"].to_numpy(),
                        "empty_stadium": False,
                        "void_gw": False,
                        "subs_regime": flags.subs_regime if flags is not None else 5,
                        "stint_id": 0,
                    }
                )
            )
    if not frames:
        return pm.iloc[0:0].copy()
    out = pd.concat(frames, ignore_index=True)
    for col in pm.columns:
        if col not in out.columns:
            out[col] = np.nan
    return out[list(pm.columns)]


def availability_frame(ctx: LiveContext, gws: list[int]) -> pd.DataFrame:
    """Deadline-snapshot availability rows for ``features/windows.py``.

    One row per (season, gw, player_code) over the target ``gws`` with the
    bootstrap's ``status`` and ``chance_of_playing`` — the flags apply to the
    upcoming GWs (the per-GW recovery happens later in the minutes gating, not
    here, so the feature columns reflect the raw snapshot).
    """
    base = ctx.roster[["player_code", "status", "chance_of_playing"]]
    frames = []
    for gw in gws:
        rows = base.copy()
        rows.insert(0, "gw", int(gw))
        rows.insert(0, "season", ctx.season)
        frames.append(rows)
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------------------
# Availability v2 — dated return gates parsed from the news strings
# --------------------------------------------------------------------------------------


def parse_news_return(
    news: str, news_added: str | None, fallback: dt.date | None = None
) -> tuple[dt.date, str] | None:
    """Extract (return date, kind) from an FPL news string; ``None`` when undated.

    The live news grammar is a small closed vocabulary (verified 2026-08-11):
    ``"<Type> injury - Expected back 23 Aug"`` and ``"Suspended until 6 Sep"``
    carry dates; ``"… - Unknown return date"`` / ``"… - 75% chance of playing"``
    / departure notices do not.  ``kind`` is ``"injury"`` or ``"suspension"``.

    The year is inferred from ``news_added``: the parsed day-month is placed in
    the announcement's year unless that lands more than 45 days BEFORE the
    announcement, in which case it rolls to the next year (Dec -> Jan wrap).
    A return date in the recent past (stale news) parses fine and simply gates
    nothing — the caller compares kickoffs against it.
    """
    m = _NEWS_RETURN_RE.search(news or "")
    if m is None:
        return None
    mon = _MONTHS.get(m.group("mon").lower())
    if mon is None:
        return None
    added: dt.date | None = None
    if news_added:
        try:
            added = dt.datetime.fromisoformat(str(news_added).replace("Z", "+00:00")).date()
        except ValueError:
            added = None
    if added is None:
        added = fallback if fallback is not None else dt.datetime.now(dt.UTC).date()
    try:
        candidate = dt.date(added.year, mon, int(m.group("day")))
    except ValueError:
        return None
    if candidate < added - dt.timedelta(days=45):
        candidate = dt.date(added.year + 1, mon, candidate.day)
    kind = "suspension" if m.group("kind").lower().startswith("suspended") else "injury"
    return candidate, kind


def availability_overrides(
    ctx: LiveContext,
) -> tuple[dict[tuple[int, int], float], dict[int, dict[str, Any]]]:
    """Per-(player_code, gw) minutes-availability factors from dated news.

    For every roster player whose news carries a parseable return date, the
    player's OWN team's first kickoff in each GW decides the gate:

    * kickoff before the return date -> out: ``0.0`` for suspensions (a ban is
      a hard rule, not a fitness state), :data:`_OUT_NO_CHANCE_FACTOR` for
      injuries;
    * the first GW at/after the return date -> ``1.0`` for suspensions
      (suspended players are match-fit), :data:`_RETURN_GW_FACTOR` for injuries
      (comeback sharpness/bench risk);
    * later GWs -> ``1.0`` — overriding the generic linear-recovery heuristic
      in both directions.

    Returns ``(overrides, report)`` where the report maps player_code ->
    ``{return_date, kind, return_gw}`` for observability/publishing.
    """
    fx = ctx.fixtures
    ko = (
        pd.concat(
            [
                fx[["gw", "home_team_code", "kickoff_utc"]].rename(
                    columns={"home_team_code": "team_code"}
                ),
                fx[["gw", "away_team_code", "kickoff_utc"]].rename(
                    columns={"away_team_code": "team_code"}
                ),
            ],
            ignore_index=True,
        )
        .groupby(["team_code", "gw"])["kickoff_utc"]
        .min()
    )
    # Year-inference anchor when news_added is absent: the snapshot's own date
    # (directory name), so offline runs are reproducible regardless of wall clock.
    snap_date: dt.date | None = None
    try:
        snap_date = dt.date.fromisoformat(ctx.snap_dir.name)
    except ValueError:
        earliest = ctx.fixtures["kickoff_utc"].min()
        if pd.notna(earliest):
            snap_date = earliest.date()
    overrides: dict[tuple[int, int], float] = {}
    report: dict[int, dict[str, Any]] = {}
    for row in ctx.roster.itertuples(index=False):
        news = row.news if isinstance(row.news, str) else ""
        added = getattr(row, "news_added", None)
        added = added if isinstance(added, str) else None
        parsed = parse_news_return(news, added, fallback=snap_date)
        if parsed is None:
            continue
        return_date, kind = parsed
        team_code = int(row.team_code)
        if team_code not in ko.index.get_level_values(0):
            continue
        code = int(row.player_code)
        team_ko = ko.loc[team_code]
        return_gw: int | None = None
        for gw, kickoff in team_ko.items():
            gw = int(gw)
            if pd.isna(kickoff):
                continue
            if kickoff.date() < return_date:
                overrides[(code, gw)] = 0.0 if kind == "suspension" else _OUT_NO_CHANCE_FACTOR
            elif return_gw is None:
                return_gw = gw
                overrides[(code, gw)] = 1.0 if kind == "suspension" else _RETURN_GW_FACTOR
            else:
                overrides[(code, gw)] = 1.0
        report[code] = {
            "return_date": return_date.isoformat(),
            "kind": kind,
            "return_gw": return_gw,
        }
    return overrides, report


# --------------------------------------------------------------------------------------
# Cold-start priors
# --------------------------------------------------------------------------------------


@dataclass
class RatePriors:
    """Pooled per-90 rate priors by position x price decile (cold-start policy).

    Fitted on modern-era history (``season >= min_season``; falls back to all
    seasons when that leaves nothing) on played rows only.  Each (position,
    decile) cell rate is shrunk toward the position rate with
    ``shrink_nineties`` pseudo-90s so thin cells stay sane.  ``lookup``
    resolves a new player's decile from the fitted per-position price edges.
    """

    n_deciles: int = 10
    min_season: int = 2022
    shrink_nineties: float = 50.0
    _edges: dict[str, np.ndarray] = field(default_factory=dict, repr=False)
    _cells: dict[tuple[str, int], dict[str, float]] = field(default_factory=dict, repr=False)
    _position_rates: dict[str, dict[str, float]] = field(default_factory=dict, repr=False)

    _TARGETS = ("lam_goal", "lam_assist", "lam_saves", "lam_defcon")

    def fit(self, player_match: pd.DataFrame) -> RatePriors:
        """Fit price-decile cells from historical ``player_match`` rows."""
        from fplai.models.rates import defcon_counts

        pm = player_match[pd.to_numeric(player_match["minutes"], errors="coerce") > 0]
        modern = pm[pm["season"] >= self.min_season]
        if len(modern):
            pm = modern
        if pm.empty:
            raise ValueError("no played rows to fit rate priors on")
        minutes = pd.to_numeric(pm["minutes"], errors="coerce").to_numpy(dtype=float)
        exposure = minutes / 90.0
        price = pd.to_numeric(pm["price"], errors="coerce").to_numpy(dtype=float)
        pos = pm["position"].astype(str).to_numpy()
        labels = {
            "lam_goal": pd.to_numeric(pm["goals_scored"], errors="coerce").to_numpy(float),
            "lam_assist": pd.to_numeric(pm["assists"], errors="coerce").to_numpy(float),
            "lam_saves": pd.to_numeric(pm["saves"], errors="coerce").to_numpy(float),
            "lam_defcon": defcon_counts(pm).to_numpy(),
        }

        def _rate(y: np.ndarray, e: np.ndarray) -> float:
            ok = np.isfinite(y) & np.isfinite(e)
            total = float(e[ok].sum())
            return float(y[ok].sum() / total) if total > 0 else 0.0

        for p in ("GKP", "DEF", "MID", "FWD"):
            mask = pos == p
            if not mask.any():
                self._edges[p] = np.array([])
                self._position_rates[p] = {t: 0.0 for t in self._TARGETS}
                continue
            quantiles = np.linspace(0.0, 1.0, self.n_deciles + 1)[1:-1]
            edges = np.unique(np.nanquantile(price[mask], quantiles))
            self._edges[p] = edges
            self._position_rates[p] = {
                t: _rate(labels[t][mask], exposure[mask]) for t in self._TARGETS
            }
            deciles = np.searchsorted(edges, price[mask], side="right")
            for d in range(len(edges) + 1):
                cell = mask.copy()
                cell[mask] = deciles == d
                cell_rates: dict[str, float] = {}
                for t in self._TARGETS:
                    ok = cell & np.isfinite(labels[t]) & np.isfinite(exposure)
                    y_sum = float(labels[t][ok].sum())
                    e_sum = float(exposure[ok].sum())
                    prior = self._position_rates[p][t]
                    k = self.shrink_nineties
                    cell_rates[t] = (y_sum + k * prior) / (e_sum + k) if (e_sum + k) > 0 else prior
                self._cells[(p, d)] = cell_rates
        return self

    def lookup(self, positions: np.ndarray, prices: np.ndarray) -> dict[str, np.ndarray]:
        """Vectorized cell lookup -> ``{target: per-90 rate array}``.

        GKP rows get 0 for goal/assist/defcon rates; outfield rows get 0 saves
        (mirroring the RatesModel output conventions).
        """
        n = len(positions)
        out = {t: np.zeros(n) for t in self._TARGETS}
        for i in range(n):
            p = str(positions[i])
            edges = self._edges.get(p)
            if edges is None:
                rates = {t: 0.0 for t in self._TARGETS}
            else:
                price = float(prices[i]) if np.isfinite(prices[i]) else float("nan")
                d = int(np.searchsorted(edges, price, side="right")) if np.isfinite(price) else 0
                rates = self._cells.get((p, d), self._position_rates.get(p, {}))
            for t in self._TARGETS:
                out[t][i] = rates.get(t, 0.0)
        gkp = np.asarray([str(p) == "GKP" for p in positions])
        out["lam_saves"] = np.where(gkp, out["lam_saves"], 0.0)
        out["lam_defcon"] = np.where(gkp, 0.0, out["lam_defcon"])
        return out


@dataclass
class LiveAdjustments:
    """Everything the prediction path applies on top of the trained models."""

    season: int
    first_gw: int
    fatigue: dict[int, float]
    """player_code -> q2 dampening for GWs 1-4 (FPL_KNOWLEDGE §3.3 hook input)."""
    cold_codes: frozenset[int]
    """Roster codes with no player_match history at all (cold-start policy)."""
    rate_priors: RatePriors
    avail_overrides: dict[tuple[int, int], float] = field(default_factory=dict)
    """(player_code, gw) -> minutes factor from dated news (availability v2);
    overrides the generic flag+linear-recovery heuristic where present."""

    @property
    def fatigue_gws(self) -> tuple[int, ...]:
        return FATIGUE_GWS


def match_fatigue_watchlist(
    roster: pd.DataFrame, teams: pd.DataFrame
) -> tuple[dict[int, float], dict[str, Any]]:
    """Match the §3.3 watchlist names against the bootstrap roster.

    Fuzzy name matching (via the crosswalk similarity machinery) constrained to
    the listed club(s); the strongest dampening wins if a player matches twice.
    Returns ``(dampening_by_code, report)`` where the report carries matched
    and unmatched entries for ``fatigue_2026.json``.
    """
    short_by_code = {
        int(c): str(s) for c, s in zip(teams["team_code"], teams["short_name"], strict=True)
    }
    club = roster["team_code"].map(lambda c: short_by_code.get(int(c), ""))
    dampening: dict[int, float] = {}
    matched: list[dict[str, Any]] = []
    unmatched: list[str] = []
    for name, clubs, damp in WC2026_FATIGUE_WATCHLIST:
        candidates = roster.loc[club.isin(clubs)]
        best_code: int | None = None
        best_row: Any = None
        best_score = 0.0
        for row in candidates.itertuples():
            variants = [str(row.web_name), f"{row.first_name} {row.second_name}"]
            score = max(crosswalk.name_similarity([name], v) for v in variants)
            if score > best_score or (
                score == best_score and best_code is not None and row.player_code < best_code
            ):
                best_score, best_code, best_row = score, int(row.player_code), row
        if best_code is not None and best_score >= FATIGUE_MATCH_THRESHOLD:
            dampening[best_code] = min(damp, dampening.get(best_code, 1.0))
            matched.append(
                {
                    "name": name,
                    "clubs": sorted(clubs),
                    "player_code": best_code,
                    "web_name": str(best_row.web_name),
                    "score": round(best_score, 3),
                    "dampening": dampening[best_code],
                }
            )
        else:
            unmatched.append(name)
    report = {
        "matched": matched,
        "unmatched": unmatched,
        "n_watchlist": len(WC2026_FATIGUE_WATCHLIST),
    }
    return dampening, report


def build_adjustments(
    ctx: LiveContext,
    player_match: pd.DataFrame,
    *,
    processed_dir: Path | None = None,
) -> LiveAdjustments:
    """Assemble the live prediction adjustments and write ``fatigue_2026.json``.

    ``player_match`` must be the *historical* table (no synthesized future
    rows) — cold-start codes are roster codes absent from it entirely.
    """
    seen = set(int(c) for c in player_match["player_code"].unique())
    cold = frozenset(
        int(c) for c in ctx.roster["player_code"] if int(c) not in seen
    )
    fatigue, report = match_fatigue_watchlist(ctx.roster, ctx.teams)
    processed = Path(processed_dir) if processed_dir is not None else config.PROCESSED_DIR
    processed.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "season": ctx.season,
        "gws": list(FATIGUE_GWS),
        "dampening": {str(code): d for code, d in sorted(fatigue.items())},
        **report,
    }
    (processed / f"fatigue_{ctx.season}.json").write_text(json.dumps(payload, indent=1))
    overrides, returns_report = availability_overrides(ctx)
    (processed / f"availability_{ctx.season}.json").write_text(
        json.dumps(
            {
                "generated_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
                "season": ctx.season,
                "returns": {str(c): r for c, r in sorted(returns_report.items())},
            },
            indent=1,
        )
    )
    priors = RatePriors().fit(player_match)
    return LiveAdjustments(
        season=ctx.season,
        first_gw=int(ctx.next_gw or 1),
        fatigue=fatigue,
        cold_codes=cold,
        rate_priors=priors,
        avail_overrides=overrides,
    )


# --------------------------------------------------------------------------------------
# Live prediction adjustments (minutes + rates)
# --------------------------------------------------------------------------------------


def _neutral_avail_features(features: pd.DataFrame) -> pd.DataFrame:
    """Copy of ``features`` with heuristic-model availability inputs neutralized.

    The heuristic cold-start pass must not double-apply availability — the
    uniform gating in :func:`live_minutes_predict` handles it for every row.
    """
    out = features.copy()
    out["f_avail_chance"] = np.nan
    out["f_avail_is_out"] = 0.0
    out["f_avail_is_doubt"] = 0.0
    return out


def _predict_with_fatigue(model: Any, features: pd.DataFrame, adj: LiveAdjustments) -> pd.DataFrame:
    """Model predict with the §3.3 ``fatigue_dampening`` hook on GWs 1-4 rows only."""
    if not adj.fatigue:
        return model.predict(features)
    in_window = (features["season"] == adj.season) & features["gw"].isin(adj.fatigue_gws)
    parts: list[pd.DataFrame] = []
    if in_window.any():
        parts.append(model.predict(features.loc[in_window], fatigue_dampening=adj.fatigue))
    if (~in_window).any():
        parts.append(model.predict(features.loc[~in_window]))
    return pd.concat(parts, ignore_index=True)


def _availability_factors(features: pd.DataFrame, adj: LiveAdjustments) -> np.ndarray:
    """Per-row minutes availability factor in [0, 1] with per-GW recovery.

    Derived from the snapshot's ``f_status_*`` / ``f_chance_of_playing``
    feature columns (NaN -> fully available): out-flagged (i/s/u) players play
    at ``chance/100`` (or :data:`_OUT_NO_CHANCE_FACTOR`), doubts (d) at
    ``chance/100`` (or 0.5).  The factor linearly recovers to 1.0 over
    :data:`_AVAILABILITY_RECOVERY_GWS` GWs after ``adj.first_gw`` — a snapshot
    flag is information about the next deadline, not the whole horizon.
    """

    def col(name: str) -> np.ndarray:
        if name in features.columns:
            return pd.to_numeric(features[name], errors="coerce").to_numpy(dtype=float)
        return np.full(len(features), np.nan)

    is_out = np.nansum(np.stack([col("f_status_i"), col("f_status_s"), col("f_status_u")]), axis=0)
    is_doubt = col("f_status_d")
    chance = col("f_chance_of_playing") / 100.0
    factor = np.ones(len(features))
    out_mask = is_out > 0
    doubt_mask = (is_doubt > 0) & ~out_mask
    factor[out_mask] = np.where(
        np.isfinite(chance[out_mask]), chance[out_mask], _OUT_NO_CHANCE_FACTOR
    )
    factor[doubt_mask] = np.where(
        np.isfinite(chance[doubt_mask]), chance[doubt_mask], _DOUBT_NO_CHANCE_FACTOR
    )
    gw = pd.to_numeric(features["gw"], errors="coerce").to_numpy(dtype=float)
    season = pd.to_numeric(features["season"], errors="coerce").to_numpy(dtype=float)
    offset = np.where(season == adj.season, gw - adj.first_gw, np.inf)
    recovery = np.clip(offset / _AVAILABILITY_RECOVERY_GWS, 0.0, 1.0)
    result = np.clip(factor + (1.0 - factor) * recovery, 0.0, 1.0)
    if adj.avail_overrides:
        # Dated-news gates (availability v2) replace the heuristic outright:
        # suspensions are hard zeros until the return GW, dated injuries are
        # floored until it — in BOTH directions vs the linear recovery.
        codes = pd.to_numeric(features["player_code"], errors="coerce").to_numpy(dtype=float)
        live_mask = season == adj.season
        for i in np.flatnonzero(live_mask):
            override = adj.avail_overrides.get((int(codes[i]), int(gw[i])))
            if override is not None:
                result[i] = override
    return result


def live_minutes_predict(
    model: Any, features: pd.DataFrame, adj: LiveAdjustments
) -> pd.DataFrame:
    """Minutes prediction with the full live policy applied.

    1. Base model predict, with WC2026 ``fatigue_dampening`` on GWs 1-4 rows.
    2. Cold-start rows (codes in ``adj.cold_codes``) replaced by the
       :class:`~fplai.models.minutes.HeuristicMinutesModel` — its no-history
       fallback is the position x price-band ``DEFAULT_SHARES`` prior, exactly
       the documented cold-start policy (see module docstring).
    3. Availability gating for every row: q1/q2 scaled by the per-row factor,
       removed mass to q0 (players flagged out at the deadline cannot score).
    """
    from fplai.models.minutes import HeuristicMinutesModel

    base = _predict_with_fatigue(model, features, adj)
    keys = [k for k in _MINUTES_KEYS if k in base.columns]
    cold_mask = features["player_code"].isin(adj.cold_codes)
    if cold_mask.any():
        heuristic = HeuristicMinutesModel()
        cold_pred = _predict_with_fatigue(
            heuristic, _neutral_avail_features(features.loc[cold_mask]), adj
        )
        base = base.set_index(keys)
        cold_pred = cold_pred.set_index(keys)
        base.loc[cold_pred.index, list(_Q_COLUMNS)] = cold_pred[list(_Q_COLUMNS)]
        base = base.reset_index()

    factors = features[keys].copy()
    factors["_avail"] = _availability_factors(features, adj)
    base = base.merge(factors.drop_duplicates(keys), on=keys, how="left")
    f = base.pop("_avail").fillna(1.0).to_numpy(dtype=float)
    q1 = base["q1"].to_numpy() * f
    q2 = base["q2"].to_numpy() * f
    base["q1"], base["q2"] = q1, q2
    base["q0"] = np.clip(1.0 - q1 - q2, 0.0, 1.0)
    total = base[["q0", "q1", "q2"]].to_numpy().sum(axis=1)
    base[["q0", "q1", "q2"]] = base[["q0", "q1", "q2"]].to_numpy() / total[:, None]
    return base


def apply_cold_start_rates(
    rates_pred: pd.DataFrame, features: pd.DataFrame, adj: LiveAdjustments
) -> pd.DataFrame:
    """Replace GBM rate outputs with position x price-decile priors for cold rows.

    Only ``lam_goal / lam_assist / lam_saves / lam_defcon`` are overridden —
    cards, own goals and dispersion already degrade to position priors inside
    ``RatesModel`` for unseen player codes.  ``rates_pred`` must be
    index-aligned with ``features`` (the RatesModel contract preserves it).
    """
    from fplai.models.rates import _RATE_CLIPS

    mask = features["player_code"].isin(adj.cold_codes).to_numpy()
    if not mask.any():
        return rates_pred
    out = rates_pred.copy()
    positions = features["position"].astype(str).to_numpy()[mask]
    prices = pd.to_numeric(features["price"], errors="coerce").to_numpy(dtype=float)[mask]
    priors = adj.rate_priors.lookup(positions, prices)
    idx = features.index[mask]
    for target, col in (
        ("goal", "lam_goal"),
        ("assist", "lam_assist"),
        ("saves", "lam_saves"),
        ("defcon", "lam_defcon"),
    ):
        lo, hi = _RATE_CLIPS[target]
        out.loc[idx, col] = np.clip(priors[col], lo, hi)
    return out


# --------------------------------------------------------------------------------------
# Team model seeding (promoted priors + new-manager uncertainty)
# --------------------------------------------------------------------------------------


def seed_team_model(team_model: TeamModel, ctx: LiveContext) -> dict[str, Any]:
    """Feed the TeamModel's promoted-team hooks and widen new-manager clubs.

    Promoted clubs (:data:`PROMOTED_STRENGTH_2026`): any stale fitted rating
    (e.g. Hull's 2016-17-era parameters, decay-weighted to nothing yet still
    present) is dropped so the club routes through the promoted-team prior,
    then the §3.6 strength multiplier is set on ``promoted_strength`` (the
    documented hook: attack x m, defence weakness / m on top of the prior).

    New-manager clubs (:data:`NEW_MANAGER_TEAMS_2026`, §3.5): fitted
    ``(log a, log b)`` shrink :data:`NEW_MANAGER_SHRINK` of the way toward the
    league mean — a point-estimate widening of the style uncertainty (the
    model regresses what it thinks it knows about how those teams played).

    Mutates the loaded model in place (in-memory only — artifacts on disk are
    untouched) and returns a report dict.
    """
    code_by_name = {
        str(n): int(c) for n, c in zip(ctx.teams["name"], ctx.teams["team_code"], strict=True)
    }
    report: dict[str, Any] = {"promoted": [], "new_manager": [], "unmatched": []}

    for name, multiplier in PROMOTED_STRENGTH_2026.items():
        code = code_by_name.get(name)
        if code is None:
            report["unmatched"].append(name)
            continue
        dropped_stale = False
        if code in team_model.team_codes_:
            i = team_model.team_codes_.index(code)
            team_model.team_codes_ = (
                team_model.team_codes_[:i] + team_model.team_codes_[i + 1 :]
            )
            team_model.log_a_ = np.delete(team_model.log_a_, i)
            team_model.log_b_ = np.delete(team_model.log_b_, i)
            dropped_stale = True
        team_model.promoted_strength[code] = float(multiplier)
        report["promoted"].append(
            {"name": name, "team_code": code, "strength": multiplier,
             "dropped_stale_rating": dropped_stale}
        )

    mean_b = float(team_model.log_b_.mean()) if len(team_model.log_b_) else 0.0
    for name in NEW_MANAGER_TEAMS_2026:
        code = code_by_name.get(name)
        if code is None or code not in team_model.team_codes_:
            continue  # promoted clubs were re-routed above; relocated clubs absent
        i = team_model.team_codes_.index(code)
        team_model.log_a_[i] *= 1.0 - NEW_MANAGER_SHRINK
        team_model.log_b_[i] = mean_b + (1.0 - NEW_MANAGER_SHRINK) * (
            team_model.log_b_[i] - mean_b
        )
        report["new_manager"].append({"name": name, "team_code": code})
    return report


# --------------------------------------------------------------------------------------
# Live odds (the-odds-api snapshot -> TeamModel blend input)
# --------------------------------------------------------------------------------------


def _odds_api_to_fpl_name(raw: str, fpl_names: set[str]) -> str | None:
    """Resolve a the-odds-api team name to a bootstrap team name (None if unknown)."""
    if raw in fpl_names:
        return raw
    mapped = ODDS_API_TEAM_NAMES.get(raw)
    if mapped in fpl_names:
        return mapped
    norm = crosswalk.normalize_name(raw)
    by_norm = {crosswalk.normalize_name(n): n for n in fpl_names}
    if norm in by_norm:
        return by_norm[norm]
    return None


def odds_frame_from_events(
    events: list[dict[str, Any]], ctx: LiveContext
) -> pd.DataFrame:
    """the-odds-api events -> odds-frame rows the TeamModel blend consumes.

    Median decimal odds across bookmakers for 1X2 (``h2h``) and the 2.5-goal
    totals line.  Team names are emitted as the FPL bootstrap names — the
    downstream ``attach_fixture_ids`` join normalizes against the live teams
    table, so the round trip is exact.  Unmappable events are logged, not fatal.
    """
    fpl_names = {str(n) for n in ctx.teams["name"]}
    rows: list[dict[str, Any]] = []
    for event in events:
        home = _odds_api_to_fpl_name(str(event.get("home_team", "")), fpl_names)
        away = _odds_api_to_fpl_name(str(event.get("away_team", "")), fpl_names)
        if home is None or away is None:
            logger.warning(
                "skipping odds event with unmapped teams: %r vs %r",
                event.get("home_team"), event.get("away_team"),
            )
            continue
        h2h: dict[str, list[float]] = {"home": [], "draw": [], "away": []}
        totals: dict[str, list[float]] = {"over": [], "under": []}
        for bookmaker in event.get("bookmakers", []):
            for market in bookmaker.get("markets", []):
                if market.get("key") == "h2h":
                    for outcome in market.get("outcomes", []):
                        name, price = outcome.get("name"), outcome.get("price")
                        if price is None:
                            continue
                        if name == event.get("home_team"):
                            h2h["home"].append(float(price))
                        elif name == event.get("away_team"):
                            h2h["away"].append(float(price))
                        elif name == "Draw":
                            h2h["draw"].append(float(price))
                elif market.get("key") == "totals":
                    for outcome in market.get("outcomes", []):
                        if outcome.get("point") != 2.5 or outcome.get("price") is None:
                            continue
                        side = str(outcome.get("name", "")).lower()
                        if side in totals:
                            totals[side].append(float(outcome["price"]))
        if not (h2h["home"] and h2h["draw"] and h2h["away"]):
            continue
        commence = str(event.get("commence_time", ""))[:10] or None
        rows.append(
            {
                "season": ctx.season,
                "date": commence,
                "home_footballdata_name": home,
                "away_footballdata_name": away,
                "odds_h": float(np.median(h2h["home"])),
                "odds_d": float(np.median(h2h["draw"])),
                "odds_a": float(np.median(h2h["away"])),
                "odds_over25": float(np.median(totals["over"])) if totals["over"] else np.nan,
                "odds_under25": float(np.median(totals["under"])) if totals["under"] else np.nan,
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "season", "date", "home_footballdata_name", "away_footballdata_name",
            "odds_h", "odds_d", "odds_a", "odds_over25", "odds_under25",
        ],
    )


def fetch_live_odds(
    ctx: LiveContext,
    *,
    raw_dir: Path | None = None,
    client: Any | None = None,
    use_network: bool = True,
) -> pd.DataFrame | None:
    """One budget-aware h2h+totals snapshot from the-odds-api (cached on disk).

    Today's raw payload is cached under ``raw/odds_api/epl_odds_{date}.json``;
    a repeat call the same day never touches the network (~2 credits/day per
    the MODEL_DESIGN_INPUTS §5.2 budget).  Without an API key (or on a fetch
    failure) the newest cached payload is used instead; returns ``None`` when
    no odds are obtainable at all — the blend then simply doesn't run.
    """
    root = Path(raw_dir) if raw_dir is not None else config.RAW_DIR / "odds_api"
    root.mkdir(parents=True, exist_ok=True)
    today_path = root / f"epl_odds_{dt.datetime.now(dt.UTC).date().isoformat()}.json"
    events: list[dict[str, Any]] | None = None
    if today_path.exists():
        events = json.loads(today_path.read_text())
        logger.info("live odds: using today's cached snapshot %s", today_path)
    elif use_network:
        from fplai.data.odds_api import TheOddsApiClient

        api = client if client is not None else TheOddsApiClient()
        try:
            events = api.epl_odds()
        except Exception as exc:  # noqa: BLE001 - odds are optional enrichment
            logger.warning("live odds fetch failed: %s", exc)
            events = None
        if events is not None:
            today_path.write_text(json.dumps(events))
    if events is None:
        cached = sorted(root.glob("epl_odds_*.json"), reverse=True)
        if not cached:
            return None
        events = json.loads(cached[0].read_text())
        logger.info("live odds: falling back to cached snapshot %s", cached[0])
    frame = odds_frame_from_events(events, ctx)
    return frame if len(frame) else None
