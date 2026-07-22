"""End-to-end pipeline orchestration: snapshot -> backfill/pulls -> build -> models.

Each stage is implemented in its own module; this file only sequences them and
is the single place the CLI and API call into.

Data stages (snapshot / backfill / build / the data portion of refresh) are
fully wired.  Model stages (train / predict / optimize) are stubs until the
feature, model and optimizer modules land.
"""

from __future__ import annotations

import datetime as dt
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from rich.console import Console

if TYPE_CHECKING:
    from fplai.data.fpl_api import SeasonState

console = Console()
logger = logging.getLogger(__name__)


def parse_seasons(spec: str | None) -> list[int] | None:
    """Parse a CLI season spec into a sorted list of season start years.

    Accepts comma-separated entries where each entry is either a single start
    year (``"2024"``) or an inclusive range (``"2016..2025"``).  Returns
    ``None`` for ``None``/empty input (meaning "use the stage's default").

    >>> parse_seasons("2016..2018,2025")
    [2016, 2017, 2018, 2025]

    Raises:
        ValueError: on malformed entries or years outside 2016..2026.
    """
    if spec is None or not spec.strip():
        return None
    seasons: set[int] = set()
    for entry in spec.split(","):
        entry = entry.strip()
        if not entry:
            continue
        if ".." in entry:
            lo_s, _, hi_s = entry.partition("..")
            try:
                lo, hi = int(lo_s), int(hi_s)
            except ValueError as exc:
                raise ValueError(f"bad season range {entry!r} (want e.g. 2016..2025)") from exc
            if lo > hi:
                raise ValueError(f"bad season range {entry!r}: start > end")
            seasons.update(range(lo, hi + 1))
        else:
            try:
                seasons.add(int(entry))
            except ValueError as exc:
                raise ValueError(f"bad season {entry!r} (want e.g. 2024 or 2016..2025)") from exc
    for s in seasons:
        if not 2016 <= s <= 2026:
            raise ValueError(f"season {s} out of supported range 2016..2026")
    return sorted(seasons)


def _table_row_counts(paths: dict[str, Path]) -> dict[str, int]:
    """Row counts of the given parquet tables (from parquet metadata; cheap)."""
    import pyarrow.parquet as pq

    return {name: pq.ParquetFile(path).metadata.num_rows for name, path in sorted(paths.items())}


# ---------------------------------------------------------------------------
# snapshot
# ---------------------------------------------------------------------------


def run_snapshot() -> SeasonState:
    """Archive today's FPL API snapshot and report season state (launch detector)."""
    from fplai.data.fpl_api import FplApiClient

    client = FplApiClient()
    snap_dir = client.take_snapshot()
    state = client.season_state()

    console.print(f"[bold]snapshot[/bold]: archived to {snap_dir}")
    console.print(
        f"  season={state.season} ({state.season}-{(state.season + 1) % 100:02d})  "
        f"current_gw={state.current_gw}  next_gw={state.next_gw}  "
        f"players={state.total_players:,}"
    )
    if state.next_deadline_utc is not None:
        console.print(f"  next deadline (UTC): {state.next_deadline_utc:%Y-%m-%d %H:%M}")
    if state.is_live_2026_27:
        console.print(
            "[bold red]  2026-27 IS LIVE[/bold red] — re-verify scoring/rules assumptions "
            "against the new bootstrap (see FPL_KNOWLEDGE UNCERTAIN items)"
        )
    else:
        console.print(f"  2026-27 launch check: not live yet (API still serving {state.season})")
    return state


# ---------------------------------------------------------------------------
# backfill
# ---------------------------------------------------------------------------


def run_backfill(seasons: Sequence[int] | None = None) -> None:
    """Download historical raw data: vaastav, football-data odds, Understat, ClubElo.

    Args:
        seasons: Season start years to fetch (default: every vaastav season,
            2016..2025).  Idempotent — already-downloaded files are skipped.
    """
    from fplai.data import clubelo, football_data, understat, vaastav

    chosen: list[int] = sorted(seasons) if seasons is not None else list(vaastav.SEASONS)

    console.print(f"[bold]backfill[/bold]: seasons {', '.join(str(s) for s in chosen)}")

    console.print("  vaastav historical CSVs ...")
    vaastav.download_all(chosen)

    console.print("  football-data.co.uk E0.csv odds ...")
    for season in chosen:
        football_data.download_e0(season)

    console.print("  Understat league JSON ...")
    us_client = understat.UnderstatClient()
    for season in chosen:
        if season <= understat.HISTORY_END_YEAR:
            us_client.fetch_league_season(season)
        else:
            console.print(f"    [yellow]skipping Understat {season} (not yet served)[/yellow]")

    console.print("  ClubElo snapshot + current PL club histories ...")
    elo = clubelo.EloClient()
    today = dt.date.today()
    snap = elo.pl_snapshot(today)
    for club in snap["Club"]:
        elo.team_history(str(club))
    console.print(f"    {len(snap)} PL clubs archived under raw/clubelo/")

    console.print("[bold]backfill[/bold]: done")


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------


def run_build(seasons: Sequence[int] | None = None) -> dict[str, Path]:
    """Build every processed parquet table from raw data already on disk.

    Builds teams/players/fixtures/player_match/player_gw over the requested
    seasons (default: every season with a raw vaastav directory), plus
    odds.parquet over whichever of those seasons have a raw E0.csv.  Never
    touches the network.  Returns table name -> written path.
    """
    from fplai.data import build, crosswalk, football_data

    resolved = crosswalk.resolve_seasons(seasons)
    console.print(
        f"[bold]build[/bold]: seasons {', '.join(str(s) for s in resolved)} "
        "(from raw vaastav dirs)"
    )
    paths = build.build_all(resolved)

    odds_seasons = [s for s in resolved if football_data.e0_path(s).exists()]
    if odds_seasons:
        paths["odds"] = football_data.build_odds_table(odds_seasons)
    else:
        console.print(
            "  [yellow]no raw football-data E0.csv files for these seasons — "
            "skipping odds.parquet (run `fplai backfill` first)[/yellow]"
        )

    for name, count in _table_row_counts(paths).items():
        console.print(f"  {name}.parquet: {count:,} rows")
    return paths


# ---------------------------------------------------------------------------
# refresh (snapshot -> incremental pulls -> build -> model stages)
# ---------------------------------------------------------------------------


def _refresh_pulls(state: SeasonState) -> None:
    """Incremental data pulls for the current season (best-effort per source).

    The FPL snapshot and the build are hard requirements of ``refresh``; the
    auxiliary sources (vaastav mirror, Understat, football-data, ClubElo) are
    refreshed best-effort with a visible warning on failure so a single
    upstream outage doesn't take down the daily refresh.
    """
    from fplai.data import clubelo, football_data, understat, vaastav

    season = state.season
    in_progress = state.next_gw is not None

    if season in vaastav.SEASONS:
        try:
            console.print(
                f"  vaastav {season} ({'in-season force refresh' if in_progress else 'fill-if-missing'}) ..."
            )
            vaastav.download_season(season, force=in_progress)
        except Exception as exc:  # noqa: BLE001 - best-effort aux pull, warned loudly
            console.print(f"  [yellow]vaastav refresh failed: {exc}[/yellow]")
    else:
        console.print(f"  [yellow]vaastav has no {season} mirror yet — skipping[/yellow]")

    try:
        console.print(f"  football-data E0 {season} ...")
        football_data.download_e0(season, force=in_progress)
    except Exception as exc:  # noqa: BLE001
        console.print(f"  [yellow]football-data refresh failed: {exc}[/yellow]")

    try:
        console.print(f"  Understat league {season} ...")
        understat.UnderstatClient().fetch_league_season(season, force=in_progress)
    except Exception as exc:  # noqa: BLE001
        console.print(f"  [yellow]Understat refresh failed: {exc}[/yellow]")

    try:
        console.print("  ClubElo snapshot ...")
        clubelo.EloClient().pl_snapshot(dt.date.today())
    except Exception as exc:  # noqa: BLE001
        console.print(f"  [yellow]ClubElo refresh failed: {exc}[/yellow]")


def run_refresh() -> None:
    """Pull fresh data from all sources, rebuild tables, then (stub) model stages.

    The data portion (snapshot -> incremental pulls -> build) is fully wired
    and must succeed; the model stages print "not yet implemented" without
    failing so `fplai refresh` stays exit-0 until they land.
    """
    state = run_snapshot()
    console.print("[bold]refresh[/bold]: incremental pulls")
    _refresh_pulls(state)
    run_build()
    console.print("[bold]train[/bold]: not yet implemented (skipping)")
    console.print("[bold]predict[/bold]: not yet implemented (skipping)")
    console.print("[bold]optimize[/bold]: not yet implemented (skipping)")
    console.print("[bold]refresh[/bold]: data refresh complete")


# ---------------------------------------------------------------------------
# model-stage stubs (stage 2+)
# ---------------------------------------------------------------------------


def run_train() -> None:
    """Retrain all models on the full historical dataset (stage 2 — not built yet)."""
    console.print("[bold]train[/bold]: model training not yet implemented")
    raise SystemExit(1)


def run_predict() -> None:
    """Generate expected-points predictions (stage 2 — not built yet)."""
    console.print("[bold]predict[/bold]: prediction not yet implemented")
    raise SystemExit(1)


def run_optimize() -> None:
    """Run the squad optimizer and chip planner (stage 3 — not built yet)."""
    console.print("[bold]optimize[/bold]: optimization not yet implemented")
    raise SystemExit(1)
