"""Played-GW outcome ingestion for the live season.

The historical backfill (``fplai.data.vaastav`` -> ``build.py``) covers finished
seasons; nothing feeds *this* season's played gameweeks back into
``player_match.parquet`` / ``player_gw.parquet`` — and the vaastav repo no longer
guarantees weekly in-season updates (README, verified 2026-08-11).  Without those
rows every form window (last-1/3/5 matches …) starves from GW2 onward.

This module closes the gap from the primary source, the FPL API itself:

* **Source**: ``element-summary/{id}/`` ``history`` rows — one row *per fixture*
  per element with the full outcome set (minutes, points, goals, BPS, starts,
  DefCon components, FPL xG/xA/xGC) **and** the per-GW price (``value``), which
  makes them a drop-in for the canonical ``player_match`` schema and DGW-safe
  (unlike ``event/{gw}/live``, whose ``stats`` block is a GW total).  The fetch
  reuses :meth:`~fplai.data.fpl_api.FplApiClient.element_summary_sweep`.
* **Trigger**: a GW is *ingestable* once any of its fixtures is finished in the
  snapshot; it is *frozen* once the bootstrap event's ``data_checked`` flag is
  true and its rows are on disk — frozen GWs are never re-fetched, so the daily
  cron does **zero** element-summary requests between gameweeks.
* **Splice**: rows replace any existing (season, gw) rows in the processed
  parquets — idempotent, safe to re-run while bonus/BPS still settle.

Ordering contract: ``fplai build`` rebuilds ``player_match`` from raw (vaastav)
sources and therefore DROPS live-season rows; ``run_refresh`` runs
``ingest`` *after* ``build`` and *before* ``predict`` so features always see the
played rows.  ``fplai ingest`` re-runs it manually.

Known approximations (documented judgment):

* ``us_*`` (Understat) columns are left NA — the Understat splice arrives with
  the normal build once that season's league JSON exists.
* ``stint_id`` is 0 for live-season rows: an intra-season club move gives a
  player a NEW element id and the roster maps ``player_code`` to the new club,
  so features keyed by player_code stay correct; stint boundaries are
  reconstructed properly when the finished season is backfilled.
* ``player_gw.selected_by_percent`` is NA (element-summary carries counts, not
  percentages); ``transfers_in_event``/``transfers_out_event`` take the first
  history row's value per GW (they are GW-level numbers repeated per fixture).
"""

from __future__ import annotations

import datetime as dt
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from fplai import config, rules

if TYPE_CHECKING:
    from fplai.data.fpl_api import FplApiClient
    from fplai.data.live import LiveContext

logger = logging.getLogger(__name__)

__all__ = ["IngestReport", "ingest_played", "build_played_tables", "load_state", "STATE_FILE"]

#: Per-season ingest bookkeeping file under the processed dir.
STATE_FILE = "ingest_state.json"

#: element-summary history key -> player_match column (1:1 int outcomes).
_INT_STATS = (
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

#: Nullable-Int64 outcomes (2025+ fields; tolerate absence).
_NULLABLE_INT_STATS = (
    "starts",
    "defensive_contribution",
    "tackles",
    "recoveries",
    "clearances_blocks_interceptions",
)

#: element-summary float stats -> player_match columns.
_FLOAT_STATS = (
    ("expected_goals", "xg"),
    ("expected_assists", "xa"),
    ("expected_goals_conceded", "xgc"),
)

#: Understat columns the live splice cannot fill (NA until the season backfill).
_US_COLS = ("us_xg", "us_xa", "us_npxg", "us_shots", "us_key_passes")


@dataclass
class IngestReport:
    """What one ingest run did."""

    season: int
    gws: list[int] = field(default_factory=list)
    frozen_gws: list[int] = field(default_factory=list)
    n_rows: int = 0
    n_players: int = 0
    n_swept: int = 0
    unmatched_elements: list[int] = field(default_factory=list)


# --------------------------------------------------------------------------------------
# State (which GWs are final on disk)
# --------------------------------------------------------------------------------------


def load_state(processed_dir: Path | None = None) -> dict[str, Any]:
    """Read the ingest state file (``{}`` when absent/corrupt)."""
    processed = Path(processed_dir) if processed_dir is not None else config.PROCESSED_DIR
    path = processed / STATE_FILE
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def _save_state(state: dict[str, Any], processed_dir: Path | None = None) -> None:
    processed = Path(processed_dir) if processed_dir is not None else config.PROCESSED_DIR
    processed.mkdir(parents=True, exist_ok=True)
    (processed / STATE_FILE).write_text(json.dumps(state, indent=1, sort_keys=True))


def _frozen_gws(state: dict[str, Any], season: int) -> set[int]:
    frozen = state.get(str(season), {}).get("frozen_gws", [])
    return {int(g) for g in frozen} if isinstance(frozen, list) else set()


# --------------------------------------------------------------------------------------
# GW selection
# --------------------------------------------------------------------------------------


def _bootstrap_events(ctx: LiveContext) -> list[dict[str, Any]]:
    """The snapshot bootstrap's ``events`` array (``[]`` when unreadable)."""
    path = ctx.snap_dir / "bootstrap.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    events = data.get("events")
    return events if isinstance(events, list) else []


def ingestable_gws(ctx: LiveContext, state: dict[str, Any]) -> tuple[list[int], set[int]]:
    """(GWs to ingest now, GWs whose data is checked/final) from the snapshot.

    A GW is ingestable when at least one of its fixtures is ``finished`` and it
    is not already frozen in ``state``.  Finality comes from the bootstrap
    event's ``data_checked`` flag (bonus awarded, points settled).
    """
    fx = ctx.fixtures
    played = {
        int(g)
        for g in fx.loc[fx["finished"].astype(bool), "gw"].unique()
    }
    checked = {
        int(e["id"])
        for e in _bootstrap_events(ctx)
        if e.get("data_checked") is True and e.get("id") is not None
    }
    frozen = _frozen_gws(state, ctx.season)
    todo = sorted(played - frozen)
    return todo, checked


# --------------------------------------------------------------------------------------
# Row building
# --------------------------------------------------------------------------------------


def _team_code_by_fpl_id(ctx: LiveContext) -> dict[int, int]:
    teams = ctx.teams
    return {
        int(i): int(c)
        for i, c in zip(teams["fpl_team_id"], teams["team_code"], strict=True)
    }


def build_played_tables(
    ctx: LiveContext,
    payloads: dict[int, dict[str, Any]],
    gws: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """element-summary payloads -> canonical (player_match, player_gw) rows.

    Args:
        ctx: The live snapshot context (roster + teams for identity mapping).
        payloads: ``{fpl_element_id: element-summary payload}``.
        gws: Only ``history`` rows whose ``round`` is in this list are kept.

    Returns:
        Two frames conforming to the ARCHITECTURE ``player_match`` /
        ``player_gw`` schemas (Understat columns NA).
    """
    roster = ctx.roster.set_index("fpl_element_id")
    opp_code = _team_code_by_fpl_id(ctx)
    keep = {int(g) for g in gws}
    flags = rules.SEASON_FLAGS.get(ctx.season)
    subs_regime = flags.subs_regime if flags is not None else 5

    rows: list[dict[str, Any]] = []
    for element_id, payload in payloads.items():
        history = payload.get("history") or []
        if int(element_id) not in roster.index:
            continue
        who = roster.loc[int(element_id)]
        for h in history:
            gw = h.get("round")
            if gw is None or int(gw) not in keep:
                continue
            opponent = h.get("opponent_team")
            row: dict[str, Any] = {
                "season": ctx.season,
                "gw": int(gw),
                "fpl_fixture_id": int(h["fixture"]),
                "player_code": int(who["player_code"]),
                "fpl_element_id": int(element_id),
                "team_code": int(who["team_code"]),
                "opponent_code": opp_code.get(int(opponent), -1) if opponent is not None else -1,
                "was_home": bool(h.get("was_home", False)),
                "position": str(who["position"]),
                "price": int(h.get("value") or who["price"]),
                "empty_stadium": False,
                "void_gw": False,
                "subs_regime": int(subs_regime),
                "stint_id": 0,
                "_transfers_in": h.get("transfers_in"),
                "_transfers_out": h.get("transfers_out"),
            }
            for key in _INT_STATS:
                row[key] = int(h.get(key) or 0)
            for key in _NULLABLE_INT_STATS:
                row[key] = h.get(key)
            for src, dst in _FLOAT_STATS:
                v = h.get(src)
                row[dst] = float(v) if v is not None else None
            rows.append(row)

    if not rows:
        empty = pd.DataFrame()
        return empty, empty

    pm = pd.DataFrame(rows).sort_values(
        ["season", "gw", "fpl_fixture_id", "player_code"], kind="stable"
    )
    transfers = pm[["season", "gw", "player_code", "_transfers_in", "_transfers_out"]]
    pm = pm.drop(columns=["_transfers_in", "_transfers_out"])
    for col in _NULLABLE_INT_STATS:
        pm[col] = pd.array(pm[col], dtype="Int64")
    for _, dst in _FLOAT_STATS:
        pm[dst] = pd.array(pm[dst], dtype="Float64")
    for col in _US_COLS:
        pm[col] = pd.array([None] * len(pm), dtype="Float64")
    pm["position"] = pd.array(pm["position"], dtype="string")

    # ---- player_gw: sum fixtures within the GW ---------------------------------------
    keys = ["season", "gw", "player_code"]
    sums = (
        pm.groupby(keys, as_index=False)
        .agg(
            fpl_element_id=("fpl_element_id", "first"),
            team_code=("team_code", "first"),
            position=("position", "first"),
            n_fixtures=("fpl_fixture_id", "count"),
            **{c: (c, "sum") for c in _INT_STATS},
            **{c: (c, "sum") for c in _NULLABLE_INT_STATS},
            xg=("xg", "sum"),
            xa=("xa", "sum"),
            xgc=("xgc", "sum"),
            value=("price", "last"),
        )
    )
    first_transfers = transfers.groupby(keys, as_index=False).first()
    sums = sums.merge(first_transfers, on=keys, how="left")
    sums["selected_by_percent"] = pd.array([None] * len(sums), dtype="Float64")
    sums["transfers_in_event"] = pd.array(
        pd.to_numeric(sums.pop("_transfers_in"), errors="coerce"), dtype="Int64"
    )
    sums["transfers_out_event"] = pd.array(
        pd.to_numeric(sums.pop("_transfers_out"), errors="coerce"), dtype="Int64"
    )
    for col in _NULLABLE_INT_STATS:
        sums[col] = pd.array(sums[col], dtype="Int64")
    gw_cols = [
        *keys,
        "fpl_element_id",
        "team_code",
        "position",
        "n_fixtures",
        *_INT_STATS,
        *_NULLABLE_INT_STATS,
        "xg",
        "xa",
        "xgc",
        "value",
        "selected_by_percent",
        "transfers_in_event",
        "transfers_out_event",
    ]
    return pm.reset_index(drop=True), sums[gw_cols]


def _splice(base: pd.DataFrame, new: pd.DataFrame, season: int, gws: list[int]) -> pd.DataFrame:
    """Replace ``base``'s (season, gw in gws) rows with ``new`` (schema-aligned)."""
    drop = (base["season"] == season) & (base["gw"].isin(gws))
    aligned = new.copy()
    for col in base.columns:
        if col not in aligned.columns:
            aligned[col] = np.nan
    aligned = aligned[list(base.columns)]
    out = pd.concat([base.loc[~drop], aligned], ignore_index=True)
    return out.sort_values(
        ["season", "gw", "player_code"], kind="stable"
    ).reset_index(drop=True)


# --------------------------------------------------------------------------------------
# The stage
# --------------------------------------------------------------------------------------


def ingest_played(
    client: FplApiClient,
    ctx: LiveContext,
    *,
    processed_dir: Path | None = None,
    on_progress: Any | None = None,
) -> IngestReport | None:
    """Fetch + splice played-GW outcomes for the live season; ``None`` when idle.

    Fast path: when every played GW is already frozen on disk (or nothing has
    been played), no network request is made at all.
    """
    processed = Path(processed_dir) if processed_dir is not None else config.PROCESSED_DIR
    state = load_state(processed)
    todo, checked = ingestable_gws(ctx, state)
    if not todo:
        return None

    # Only elements whose team has a finished fixture in the target GWs.
    fx = ctx.fixtures
    played_fx = fx[fx["finished"].astype(bool) & fx["gw"].isin(todo)]
    team_codes = set(played_fx["home_team_code"].astype(int)) | set(
        played_fx["away_team_code"].astype(int)
    )
    roster = ctx.roster
    ids = sorted(
        int(i)
        for i, tc in zip(roster["fpl_element_id"], roster["team_code"], strict=True)
        if int(tc) in team_codes
    )
    if not ids:
        return None

    logger.info("ingest: GWs %s — sweeping %d element summaries", todo, len(ids))
    paths = client.element_summary_sweep(ids, on_progress=on_progress)
    payloads: dict[int, dict[str, Any]] = {}
    for path in paths:
        try:
            payloads[int(Path(path).stem)] = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            logger.warning("ingest: unreadable element summary %s (%s)", path, exc)

    pm_new, gw_new = build_played_tables(ctx, payloads, todo)
    if pm_new.empty:
        logger.warning("ingest: no history rows for GWs %s yet — API still settling", todo)
        return IngestReport(season=ctx.season, gws=[], n_swept=len(ids))

    pm_path = processed / "player_match.parquet"
    gw_path = processed / "player_gw.parquet"
    pm = pd.read_parquet(pm_path)
    pgw = pd.read_parquet(gw_path)
    ingested = sorted(int(g) for g in pm_new["gw"].unique())
    _splice(pm, pm_new, ctx.season, ingested).to_parquet(pm_path, index=False)
    _splice(pgw, gw_new, ctx.season, ingested).to_parquet(gw_path, index=False)

    # Freeze the ingested GWs whose bootstrap event is data_checked.
    newly_frozen = sorted(set(ingested) & checked)
    season_state = state.setdefault(str(ctx.season), {})
    frozen = sorted(_frozen_gws(state, ctx.season) | set(newly_frozen))
    season_state["frozen_gws"] = frozen
    season_state["last_run_utc"] = dt.datetime.now(dt.UTC).isoformat(timespec="seconds")
    _save_state(state, processed)

    roster_ids = {int(i) for i in roster["fpl_element_id"]}
    unmatched = sorted(set(payloads) - roster_ids)
    return IngestReport(
        season=ctx.season,
        gws=ingested,
        frozen_gws=newly_frozen,
        n_rows=len(pm_new),
        n_players=int(pm_new["player_code"].nunique()),
        n_swept=len(ids),
        unmatched_elements=unmatched,
    )
