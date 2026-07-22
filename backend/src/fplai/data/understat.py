"""Understat client (post-Dec-2025 JSON endpoints).

Understat re-architected in Dec 2025: league and player pages are now thin HTML
shells whose JS fetches plain JSON from:

- ``GET https://understat.com/getLeagueData/EPL/{year}``
- ``GET https://understat.com/getPlayerData/{understat_id}/``

Reality check (verified live 2026-07-22, deviations from the research notes):

- The load-bearing requirement is the ``X-Requested-With: XMLHttpRequest``
  header — without it both endpoints return a 404 HTML page. A browser
  ``User-Agent`` alone is NOT sufficient, and a ``Referer`` is NOT required.
- The trailing slash on ``getPlayerData`` is NOT actually required (both forms
  return 200); we keep it anyway since the site's own JS uses it.
- ``{year}`` is the season *start* year: ``2025`` == the 2025-26 season
  (its ``dates`` begin 2025-08-15). This matches the repo-wide convention.
- History is kept: seasons 2016..present all serve full data (2016 verified
  live: 524 players / 380 dates).

Payload shapes (as observed live; numeric values arrive as *strings* unless
noted):

``getLeagueData/EPL/{year}`` -> dict with keys:
    ``players``: list[dict] — season aggregates per player: ``id``,
        ``player_name``, ``games``, ``time``, ``goals``, ``xG``, ``assists``,
        ``xA``, ``shots``, ``key_passes``, ``yellow_cards``, ``red_cards``,
        ``position``, ``team_title``, ``npg``, ``npxG``, ``xGChain``,
        ``xGBuildup``.
    ``teams``: dict keyed by team-id string -> ``{id, title, history}`` where
        ``history`` is a per-match list (floats, not strings): ``h_a``, ``xG``,
        ``xGA``, ``npxG``, ``npxGA``, ``ppda``, ``ppda_allowed``, ``deep``,
        ``deep_allowed``, ``scored``, ``missed``, ``xpts``, ``result``,
        ``date``, ``wins``, ``draws``, ``loses``, ``pts``, ``npxGD``.
    ``dates``: list[dict] — one per fixture: ``id`` (match id), ``isResult``
        (bool), ``h``/``a`` (``{id, title, short_title}``), ``goals``, ``xG``,
        ``datetime``, ``forecast``.

``getPlayerData/{id}/`` -> dict with keys:
    ``player``: ``{id, name, favorite_position}``.
    ``matches``: list[dict] — per-match history across ALL leagues/seasons the
        player appears in (e.g. Haaland's Bundesliga games are included):
        ``goals``, ``shots``, ``xG``, ``time`` (minutes), ``position``,
        ``h_team``, ``a_team``, ``h_goals``, ``a_goals``, ``date``
        (YYYY-MM-DD), ``id`` (match id, joins to league ``dates[].id``),
        ``season`` (start year as str), ``roster_id``, ``xA``, ``assists``,
        ``key_passes``, ``npg``, ``npxG``, ``xGChain``, ``xGBuildup``.
    ``groups``: aggregates by season/position/situation/shotZones/shotTypes.
    ``positionsList``, ``minMaxPlayerStats``, ``shots`` (shot events),
    ``lastMatch``.

Etiquette: ~1 request / 2 s (``MIN_INTERVAL_S``), gzip accepted, browser UA.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from fplai.config import RAW_DIR

logger = logging.getLogger(__name__)

BASE_URL = "https://understat.com"
MIN_INTERVAL_S = 2.0
CACHE_TTL_S = 6 * 3600.0

#: Understat's endpoints 404 without this header (verified live 2026-07-22);
#: the browser UA is polite convention, the XHR header is the hard requirement.
REQUIRED_HEADERS: dict[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
    ),
    "X-Requested-With": "XMLHttpRequest",
    "Accept": "application/json",
}

#: Understat serves EPL history from 2014; we backfill 2016+ to match the FPL
#: training corpus (vaastav's bundled understat extracts cover the same span).
HISTORY_START_YEAR = 2016
HISTORY_END_YEAR = 2025

#: Column order/dtypes of the tidy per-player-per-match frame (the contract
#: consumed by build.py via the identity crosswalk).
PLAYER_MATCH_COLUMNS: dict[str, str] = {
    "season": "int64",
    "match_id": "int64",
    "understat_id": "int64",
    "date": "datetime64[ns]",
    "h_team": "str",
    "a_team": "str",
    "us_xg": "float64",
    "us_xa": "float64",
    "us_npxg": "float64",
    "us_shots": "int64",
    "us_key_passes": "int64",
    "minutes": "int64",
}

_last_request_monotonic: float = 0.0


def _throttle(min_interval_s: float = MIN_INTERVAL_S) -> None:
    """Sleep so consecutive fallback requests are >= ``min_interval_s`` apart."""
    global _last_request_monotonic
    elapsed = time.monotonic() - _last_request_monotonic
    if elapsed < min_interval_s:
        time.sleep(min_interval_s - elapsed)
    _last_request_monotonic = time.monotonic()


@retry(
    retry=retry_if_exception_type((httpx.TransportError, httpx.HTTPStatusError)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=2, max=30),
    reraise=True,
)
def _fallback_get(url: str) -> dict[str, Any]:
    """Throttled, retrying GET used when ``fpl_api.polite_get`` is unavailable.

    Sends the browser UA + ``X-Requested-With`` headers Understat requires and
    returns the parsed JSON body. httpx transparently decompresses gzip.
    """
    _throttle()
    logger.info("GET %s", url)
    resp = httpx.get(url, headers=REQUIRED_HEADERS, timeout=30.0, follow_redirects=True)
    resp.raise_for_status()
    return resp.json()


def _as_json(payload: Any) -> dict[str, Any]:
    """Normalise whatever the fetch layer returns into a parsed JSON dict.

    ``polite_get``'s return type is not pinned by the interface contract, so we
    accept a dict (already parsed), an httpx.Response-like object (has
    ``.json()``), or raw ``bytes``/``str``.
    """
    if isinstance(payload, dict):
        return payload
    if hasattr(payload, "json") and callable(payload.json):
        return payload.json()  # type: ignore[no-any-return]
    if isinstance(payload, bytes):
        payload = payload.decode("utf-8")
    if isinstance(payload, str):
        return json.loads(payload)  # type: ignore[no-any-return]
    raise TypeError(f"cannot interpret fetch result of type {type(payload)!r} as JSON")


def _resolve_fetch() -> Callable[[str], Any]:
    """Return the shared ``polite_get`` bound with Understat's headers, else a fallback.

    The architecture routes all HTTP through ``fpl_api.polite_get(url, *,
    min_interval_s, cache_ttl_s)``. Understat additionally needs custom
    headers; we try passing ``headers=`` (not part of the guaranteed
    signature) and degrade to the local ``_fallback_get`` if ``polite_get``
    is missing or rejects the kwarg.
    """
    try:
        from fplai.data.fpl_api import polite_get
    except ImportError:  # fpl_api not present (e.g. partial checkout)
        logger.debug("fpl_api.polite_get unavailable; using local fallback fetch")
        return _fallback_get

    def fetch(url: str) -> Any:
        try:
            return polite_get(
                url,
                min_interval_s=MIN_INTERVAL_S,
                cache_ttl_s=CACHE_TTL_S,
                headers=REQUIRED_HEADERS,
            )
        except TypeError:
            # polite_get without a headers kwarg cannot satisfy Understat's
            # X-Requested-With requirement — use the self-contained fallback.
            return _fallback_get(url)

    return fetch


def season_dir(year: int, raw_dir: Path = RAW_DIR) -> Path:
    """Raw-data directory for one Understat season (``raw/understat/{year}``)."""
    return raw_dir / "understat" / str(year)


class UnderstatClient:
    """Polite client for Understat's JSON endpoints, with raw-JSON persistence.

    Parameters
    ----------
    raw_dir:
        Root raw-data directory (defaults to ``fplai.config.RAW_DIR``).
    fetch:
        Optional ``url -> payload`` callable (tests inject fixtures here).
        Defaults to the shared ``fpl_api.polite_get`` when available, else a
        local throttled httpx fallback. The payload may be a dict, a
        Response-like object, ``str`` or ``bytes`` — it is normalised.
    """

    def __init__(
        self,
        raw_dir: Path = RAW_DIR,
        fetch: Callable[[str], Any] | None = None,
    ) -> None:
        self.raw_dir = raw_dir
        self._fetch = fetch if fetch is not None else _resolve_fetch()

    # -- endpoint wrappers ---------------------------------------------------

    def league_data(self, year: int, league: str = "EPL") -> dict[str, Any]:
        """Fetch one league season: dict with ``players``, ``teams``, ``dates``.

        ``year`` is the season start year (2025 == 2025-26). See the module
        docstring for the exact payload shape.
        """
        data = _as_json(self._fetch(f"{BASE_URL}/getLeagueData/{league}/{year}"))
        missing = {"players", "teams", "dates"} - data.keys()
        if missing:
            raise ValueError(f"unexpected league payload for {league}/{year}: missing {missing}")
        return data

    def player_data(self, understat_id: int) -> dict[str, Any]:
        """Fetch one player's full history: ``player``, ``matches``, ``groups``, ``shots``, ...

        ``matches`` spans every league/season the player appears in; filter by
        match id against a league's ``dates`` to isolate EPL games. See the
        module docstring for the exact payload shape.
        """
        # Trailing slash kept to mirror the site's own JS (not strictly required).
        data = _as_json(self._fetch(f"{BASE_URL}/getPlayerData/{understat_id}/"))
        missing = {"player", "matches"} - data.keys()
        if missing:
            raise ValueError(f"unexpected player payload for id {understat_id}: missing {missing}")
        return data

    # -- persistence ---------------------------------------------------------

    def fetch_league_season(self, year: int, *, force: bool = False) -> Path:
        """Fetch and persist one season's league JSON to ``raw/understat/{year}/league.json``.

        Skips the network round-trip if the file already exists (pass
        ``force=True`` to re-fetch, e.g. nightly during the season).
        """
        path = season_dir(year, self.raw_dir) / "league.json"
        if path.exists() and not force:
            logger.debug("league.json for %s exists, skipping", year)
            return path
        data = self.league_data(year)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data), encoding="utf-8")
        logger.info("wrote %s (%d players)", path, len(data["players"]))
        return path

    def fetch_players(self, ids: Iterable[int], year: int) -> list[Path]:
        """Fetch and persist player JSONs to ``raw/understat/{year}/players/{id}.json``.

        Skip-if-exists per player, so re-runs only fetch what is missing.
        Returns the paths (existing and newly written).
        """
        players_dir = season_dir(year, self.raw_dir) / "players"
        players_dir.mkdir(parents=True, exist_ok=True)
        paths: list[Path] = []
        for understat_id in ids:
            path = players_dir / f"{int(understat_id)}.json"
            if not path.exists():
                data = self.player_data(int(understat_id))
                path.write_text(json.dumps(data), encoding="utf-8")
                logger.info("wrote %s", path)
            paths.append(path)
        return paths

    def fetch_season(self, year: int, *, force: bool = False) -> Path:
        """Fetch a season's league JSON plus every player listed in it."""
        league_path = self.fetch_league_season(year, force=force)
        league = json.loads(league_path.read_text(encoding="utf-8"))
        self.fetch_players((int(p["id"]) for p in league["players"]), year)
        return season_dir(year, self.raw_dir)

    def fetch_history(
        self,
        start_year: int = HISTORY_START_YEAR,
        end_year: int = HISTORY_END_YEAR,
        *,
        include_players: bool = False,
    ) -> list[Path]:
        """Backfill league JSON for all seasons ``start_year..end_year`` inclusive.

        Understat keeps full history, so pre-2022 xG can come straight from
        here (vaastav's bundled understat extracts remain an offline
        alternative). With ``include_players=True`` also fetches every player
        file per season (~500 requests/season at 1 req/2s — slow; league-level
        files alone are enough to seed the crosswalk).
        """
        paths = []
        for year in range(start_year, end_year + 1):
            if include_players:
                paths.append(self.fetch_season(year))
            else:
                paths.append(self.fetch_league_season(year))
        return paths


def to_player_match_frame(season: int, raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Parse raw league + player JSON into a tidy per-player-per-match frame.

    Reads ``raw/understat/{season}/league.json`` and every
    ``raw/understat/{season}/players/*.json``, keeps only matches belonging to
    that league season (player ``matches`` span all leagues/seasons, so rows
    are filtered by match id against the league's ``dates``), and returns one
    row per player per match with columns/dtypes per
    :data:`PLAYER_MATCH_COLUMNS`:

    ``season, match_id, understat_id, date, h_team, a_team, us_xg, us_xa,
    us_npxg, us_shots, us_key_passes, minutes``

    The frame is also saved to ``raw/understat/{season}/player_matches.parquet``.
    build.py joins it into ``player_match.parquet`` via the identity crosswalk;
    this function never writes processed tables itself.
    """
    sdir = season_dir(season, raw_dir)
    league_path = sdir / "league.json"
    if not league_path.exists():
        raise FileNotFoundError(
            f"{league_path} not found — run UnderstatClient.fetch_league_season({season}) first"
        )
    league = json.loads(league_path.read_text(encoding="utf-8"))
    season_match_ids = {str(d["id"]) for d in league["dates"]}

    rows: list[dict[str, Any]] = []
    players_dir = sdir / "players"
    player_files = sorted(players_dir.glob("*.json")) if players_dir.exists() else []
    for path in player_files:
        data = json.loads(path.read_text(encoding="utf-8"))
        understat_id = int(data["player"]["id"])
        for match in data["matches"]:
            if str(match["id"]) not in season_match_ids:
                continue  # other league, other season, or cup game
            rows.append(
                {
                    "season": season,
                    "match_id": int(match["id"]),
                    "understat_id": understat_id,
                    "date": match["date"],
                    "h_team": match["h_team"],
                    "a_team": match["a_team"],
                    "us_xg": float(match["xG"]),
                    "us_xa": float(match["xA"]),
                    "us_npxg": float(match["npxG"]),
                    "us_shots": int(match["shots"]),
                    "us_key_passes": int(match["key_passes"]),
                    "minutes": int(match["time"]),
                }
            )

    frame = pd.DataFrame(rows, columns=list(PLAYER_MATCH_COLUMNS))
    # to_datetime first (astype str->datetime is deprecated), then pin the unit
    # to ns — pandas >= 3 otherwise infers datetime64[us] for date-only strings.
    frame["date"] = pd.to_datetime(frame["date"], format="%Y-%m-%d")
    frame = frame.astype(PLAYER_MATCH_COLUMNS)
    frame = frame.sort_values(["date", "match_id", "understat_id"]).reset_index(drop=True)

    out_path = sdir / "player_matches.parquet"
    frame.to_parquet(out_path, index=False)
    logger.info("wrote %s (%d rows from %d player files)", out_path, len(frame), len(player_files))
    return frame
