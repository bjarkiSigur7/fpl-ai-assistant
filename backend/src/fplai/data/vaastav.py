"""Historical backfill from the vaastav/Fantasy-Premier-League GitHub repository.

Downloads per-season CSV extracts (merged gameweek data, raw player dumps, teams,
fixtures and the bundled Understat id maps) into ``RAW_DIR/vaastav/{season-label}/``.

Observed per-season availability / drift in the upstream repo (probed 2026-07-22):

===========  ==========  =========  ============  ============  =====================
season       teams.csv   fixtures   id_dict.csv   understat/    merged_gw notes
===========  ==========  =========  ============  ============  =====================
2016-17      absent      absent     absent        absent        rich per-match stats
2017-18      absent      absent     absent        absent        rich per-match stats
2018-19      absent      present    absent        absent        rich stats; latin-1!
2019-20      present     present    absent        player+team   rich stats dropped
2020-21      present     present    absent        player only   +position/team/xP cols
2021-22      present     present    present       player only   same as 2020-21
2022-23      present     present    present       player only   +starts, expected_* cols
2023-24      present     present    absent        player only   same as 2022-23
2024-25      present     present    absent        player only   +mng_* cols, AM elements
2025-26      present     present    absent        absent        +CBI/DefCon/recoveries/
                                                                 tackles cols
===========  ==========  =========  ============  ============  =====================

Further drift notes (all verified against the live raw files):

* ``merged_gw.csv`` (under ``gws/`` upstream) always has ``element``, ``fixture``,
  ``opponent_team``, ``round``/``GW``, ``value``, ``was_home`` and the core scoring
  stats.  ``position``/``team`` (name) columns only exist from 2020-21; the position
  strings are ``GK``/``DEF``/``MID``/``FWD`` (plus ``AM`` managers in 2024-25) — never
  ``GKP``.  2016-17/2017-18 files quote every field; 2018-19 files are latin-1 encoded
  (invalid UTF-8 bytes in player names) — see :func:`read_csv_tolerant`.
* ``players_raw.csv`` (bootstrap elements dump) always has ``id``, ``code``,
  ``element_type``, ``team``, ``team_code`` and name columns; ``opta_code`` appears in
  recent seasons only.  2024-25 includes manager elements (``element_type == 5``).
* ``fixtures.csv`` is the FPL fixtures endpoint dump (``id``, ``event``,
  ``kickoff_time``, ``team_h``/``team_a``, scores, ``finished``).  For 2016-17/2017-18
  the file is absent — fixtures are reconstructed from merged_gw in ``build.py``.
* ``id_dict.csv`` (Understat <-> FPL id map; header has spaces after commas) only
  ships for 2021-22 and 2022-23.
* ``master_team_list.csv`` lives at the repo data root (season label -> team id ->
  team name for every season) and is needed to name teams for 2016-17..2018-19.

All HTTP goes through :func:`fplai.data.fpl_api.polite_get` (the shared throttled
fetch helper) with a long cache TTL; a custom ``fetch`` callable may be injected for
testing.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import httpx
import pandas as pd

from fplai import config

logger = logging.getLogger(__name__)

VAASTAV_BASE_URL = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"

#: Seasons shipped by the upstream repo, as start years (2016-17 .. 2025-26).
FIRST_SEASON = 2016
LAST_SEASON = 2025
SEASONS: tuple[int, ...] = tuple(range(FIRST_SEASON, LAST_SEASON + 1))

#: GitHub raw is a static CDN — be polite but not paranoid; cache for 30 days.
MIN_INTERVAL_S = 0.5
CACHE_TTL_S = 30 * 24 * 3600

FetchFn = Callable[[str], bytes]
"""Fetch callable contract: ``fetch(url) -> bytes``; raises ``FileNotFoundError`` on 404."""


@dataclass(frozen=True)
class SeasonFile:
    """One downloadable file within a season directory of the vaastav repo."""

    source: str
    """Path relative to ``data/{season-label}/`` upstream (e.g. ``gws/merged_gw.csv``)."""
    dest: str
    """Path relative to ``RAW_DIR/vaastav/{season-label}/`` locally."""
    required: bool = False
    """If True, a missing upstream file is an error; otherwise it is skipped with a log."""


#: Default per-season manifest (see the availability table in the module docstring).
DEFAULT_FILES: tuple[SeasonFile, ...] = (
    SeasonFile("gws/merged_gw.csv", "merged_gw.csv", required=True),
    SeasonFile("players_raw.csv", "players_raw.csv", required=True),
    SeasonFile("teams.csv", "teams.csv", required=False),  # absent 2016-17..2018-19
    SeasonFile("fixtures.csv", "fixtures.csv", required=False),  # absent 2016-17/2017-18
    SeasonFile("id_dict.csv", "id_dict.csv", required=False),  # only 2021-22/2022-23
    SeasonFile("understat/understat_player.csv", "understat/understat_player.csv", required=False),
    SeasonFile("understat/understat_team.csv", "understat/understat_team.csv", required=False),
)

MASTER_TEAM_LIST = "master_team_list.csv"


def season_label(season: int) -> str:
    """Return the display label (``2016 -> "2016-17"``) for a start-year season int.

    Delegates to ``fplai.rules.season_label`` when available (the canonical helper per
    ARCHITECTURE.md); falls back to a local implementation so this module works while
    ``rules.py`` is being built in parallel.
    """
    try:  # pragma: no cover - exercised only once rules.py lands
        from fplai import rules

        return rules.season_label(season)
    except (ImportError, AttributeError):
        return f"{season}-{(season + 1) % 100:02d}"


def vaastav_root(raw_root: Path | None = None) -> Path:
    """Local root of the vaastav mirror (``{raw_root}/vaastav``)."""
    return (raw_root or config.RAW_DIR) / "vaastav"


def season_dir(season: int, raw_root: Path | None = None) -> Path:
    """Local directory holding one season's downloaded files."""
    return vaastav_root(raw_root) / season_label(season)


def _payload_bytes(resp: object, url: str) -> bytes:
    """Normalize a ``polite_get`` return value (bytes / str / Response-like) to bytes."""
    if isinstance(resp, bytes | bytearray):
        content = bytes(resp)
    elif isinstance(resp, str):
        content = resp.encode("utf-8")
    else:
        status = getattr(resp, "status_code", None)
        if status == 404:
            raise FileNotFoundError(url)
        if status is not None and int(status) >= 400:
            raise RuntimeError(f"HTTP {status} fetching {url}")
        body = getattr(resp, "content", None)
        if body is None:
            raise TypeError(f"Unsupported fetch return type {type(resp)!r} for {url}")
        content = bytes(body)
    # raw.githubusercontent.com serves a tiny "404: Not Found" body for missing paths;
    # guard against a fetch helper that does not surface the status code.
    if len(content) < 64 and content.lstrip().startswith(b"404"):
        raise FileNotFoundError(url)
    return content


def _polite_fetch(url: str) -> bytes:
    """Default fetch: the shared throttled helper from ``fplai.data.fpl_api``."""
    from fplai.data import fpl_api  # deferred: built by the API-client agent

    try:
        resp = fpl_api.polite_get(url, min_interval_s=MIN_INTERVAL_S, cache_ttl_s=CACHE_TTL_S)
    except httpx.HTTPStatusError as exc:
        if exc.response is not None and exc.response.status_code == 404:
            raise FileNotFoundError(url) from exc
        raise
    return _payload_bytes(resp, url)


def _write_atomic(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    tmp.write_bytes(content)
    tmp.replace(path)


def _as_season_files(files: Sequence[SeasonFile | str] | None) -> tuple[SeasonFile, ...]:
    if files is None:
        return DEFAULT_FILES
    out: list[SeasonFile] = []
    for f in files:
        if isinstance(f, SeasonFile):
            out.append(f)
        else:
            dest = f.removeprefix("gws/")
            out.append(SeasonFile(source=f, dest=dest, required=True))
    return tuple(out)


def download_season(
    season: int,
    files: Sequence[SeasonFile | str] | None = None,
    *,
    raw_root: Path | None = None,
    fetch: FetchFn | None = None,
    force: bool = False,
) -> list[Path]:
    """Download one season's files into ``{raw_root}/vaastav/{label}/``.

    Args:
        season: Season start year (2016 .. 2025).
        files: Manifest override; plain strings are treated as required files whose
            local name drops any leading ``gws/`` component.
        raw_root: Override for ``config.RAW_DIR`` (mainly for tests).
        fetch: Injectable ``fetch(url) -> bytes`` (defaults to the shared
            ``fpl_api.polite_get`` helper). Must raise ``FileNotFoundError`` on 404.
        force: Re-download even when the local file already exists (skip-if-exists
            is the default so backfills are cheap to re-run).

    Returns:
        Paths of all files present locally after the call (downloaded or pre-existing).
    """
    label = season_label(season)
    fetch = fetch or _polite_fetch
    dest_dir = season_dir(season, raw_root)
    results: list[Path] = []
    for sf in _as_season_files(files):
        dest = dest_dir / sf.dest
        if dest.exists() and not force:
            logger.debug("vaastav %s: %s already present, skipping", label, sf.dest)
            results.append(dest)
            continue
        url = f"{VAASTAV_BASE_URL}/{label}/{sf.source}"
        try:
            content = fetch(url)
        except FileNotFoundError:
            if sf.required:
                raise FileNotFoundError(f"Required vaastav file missing upstream: {url}") from None
            logger.info("vaastav %s: %s not shipped upstream, skipping", label, sf.source)
            continue
        _write_atomic(dest, content)
        logger.info("vaastav %s: downloaded %s (%d bytes)", label, sf.dest, len(content))
        results.append(dest)
    return results


def download_master_team_list(
    *,
    raw_root: Path | None = None,
    fetch: FetchFn | None = None,
    force: bool = False,
) -> Path:
    """Download the repo-level ``master_team_list.csv`` (team names for all seasons)."""
    fetch = fetch or _polite_fetch
    dest = vaastav_root(raw_root) / MASTER_TEAM_LIST
    if dest.exists() and not force:
        return dest
    content = fetch(f"{VAASTAV_BASE_URL}/{MASTER_TEAM_LIST}")
    _write_atomic(dest, content)
    logger.info("vaastav: downloaded %s (%d bytes)", MASTER_TEAM_LIST, len(content))
    return dest


def download_all(
    seasons: Iterable[int] | None = None,
    files: Sequence[SeasonFile | str] | None = None,
    *,
    raw_root: Path | None = None,
    fetch: FetchFn | None = None,
    force: bool = False,
) -> dict[int, list[Path]]:
    """Download every season (2016-17 .. 2025-26 by default) plus the master team list.

    Returns a mapping of season start year to the local paths present for it.
    """
    download_master_team_list(raw_root=raw_root, fetch=fetch, force=force)
    out: dict[int, list[Path]] = {}
    for season in seasons if seasons is not None else SEASONS:
        out[season] = download_season(
            season, files, raw_root=raw_root, fetch=fetch, force=force
        )
    return out


def read_csv_tolerant(path: Path, **read_csv_kwargs: object) -> pd.DataFrame:
    """``pd.read_csv`` with a latin-1 fallback for the 2018-19 encoding quirk.

    The 2018-19 season files upstream contain latin-1 bytes (accented player names)
    that are invalid UTF-8; every other season is plain UTF-8.
    """
    try:
        return pd.read_csv(path, encoding="utf-8", **read_csv_kwargs)  # type: ignore[arg-type]
    except UnicodeDecodeError:
        logger.debug("%s is not UTF-8; falling back to latin-1", path)
        return pd.read_csv(path, encoding="latin-1", **read_csv_kwargs)  # type: ignore[arg-type]
