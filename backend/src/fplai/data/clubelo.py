"""ClubElo API client (http://api.clubelo.com — free CSV, no key, HTTP only).

Endpoints:

* ``/{ClubName}`` — full Elo history for one club, one row per rating period.
* ``/{YYYY-MM-DD}`` — Elo snapshot of every ranked club on that date.

Both return CSV with columns ``Rank,Club,Country,Level,Elo,From,To`` (``Rank``
is the literal string ``None`` outside the global top ranks). Raw responses are
archived under ``RAW_DIR/clubelo/``. The site has no HTTPS endpoint (port 443
closed, verified July 2026), hence the http:// base URL.

Club-name spellings were recorded from the real ``/2026-07-01`` snapshot (all
2025-26/2026-27 PL clubs) plus per-club probes for recent PL sides — see
:data:`PL_CLUBELO_NAMES`. URLs use the club name with spaces/dots/apostrophes
removed (``Man City`` -> ``/ManCity``, ``Sheffield Weds`` -> ``/SheffieldWeds``).
"""

from __future__ import annotations

import datetime as dt
import io
import logging
import re
from pathlib import Path

import httpx
import pandas as pd
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from fplai.config import RAW_DIR

logger = logging.getLogger(__name__)

BASE_URL = "http://api.clubelo.com"

#: FPL club name (bootstrap ``team.name`` and common variants) -> ClubElo ``Club``
#: value. Non-identity mappings verified against the real 2026-07-01 snapshot and
#: per-club history probes (Luton/Cardiff/Huddersfield). Identity-mapped clubs are
#: included so the dict doubles as the roster of known-good spellings.
PL_CLUBELO_NAMES: dict[str, str] = {
    # 2026-07-01 snapshot, ENG level 1 (20 clubs)
    "Arsenal": "Arsenal",
    "Aston Villa": "Aston Villa",
    "Bournemouth": "Bournemouth",
    "Brentford": "Brentford",
    "Brighton": "Brighton",
    "Burnley": "Burnley",
    "Chelsea": "Chelsea",
    "Crystal Palace": "Crystal Palace",
    "Everton": "Everton",
    "Fulham": "Fulham",
    "Leeds": "Leeds",
    "Liverpool": "Liverpool",
    "Man City": "Man City",
    "Man Utd": "Man United",  # FPL short form
    "Man United": "Man United",
    "Newcastle": "Newcastle",
    "Nott'm Forest": "Forest",  # FPL spelling -> ClubElo 'Forest'
    "Nottingham Forest": "Forest",
    "Spurs": "Tottenham",  # FPL name -> ClubElo 'Tottenham'
    "Tottenham": "Tottenham",
    "Sunderland": "Sunderland",
    "West Ham": "West Ham",
    "Wolves": "Wolves",
    # Recent PL clubs (backfill seasons 2016-17+), from the same snapshot (ENG
    # level 2/3) or verified per-club probes
    "Leicester": "Leicester",
    "Southampton": "Southampton",
    "Ipswich": "Ipswich",
    "Luton": "Luton",
    "Sheffield Utd": "Sheffield United",  # FPL short form
    "Sheffield United": "Sheffield United",
    "West Brom": "West Brom",
    "Norwich": "Norwich",
    "Watford": "Watford",
    "Middlesbrough": "Middlesbrough",
    "Swansea": "Swansea",
    "Stoke": "Stoke",
    "Hull": "Hull",
    "Cardiff": "Cardiff",
    "Huddersfield": "Huddersfield",
}


def fpl_to_clubelo(name: str) -> str:
    """Map an FPL club name to its ClubElo ``Club`` spelling (identity if unknown)."""
    return PL_CLUBELO_NAMES.get(name, name)


def clubelo_url_name(club: str) -> str:
    """Return the URL path segment for a ClubElo club name (``Man City`` -> ``ManCity``)."""
    return re.sub(r"[ .'’]", "", club)


def _coerce_bytes(resp: object) -> bytes:
    """Normalise ``polite_get``'s return value (bytes/str/httpx.Response) to bytes."""
    if isinstance(resp, bytes):
        return resp
    if isinstance(resp, bytearray):
        return bytes(resp)
    if isinstance(resp, str):
        return resp.encode("utf-8")
    content = getattr(resp, "content", None)
    if isinstance(content, (bytes, bytearray)):
        return bytes(content)
    raise TypeError(f"polite_get returned unsupported type {type(resp)!r}")


@retry(
    retry=retry_if_exception_type((httpx.TransportError, ConnectionError, TimeoutError, OSError)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, max=20),
    reraise=True,
    before_sleep=before_sleep_log(logger, logging.WARNING),
)
def _fetch(url: str, *, min_interval_s: float = 1.0, cache_ttl_s: float = 6 * 3600) -> bytes:
    """Fetch ``url`` through the shared throttled helper in ``fplai.data.fpl_api``."""
    from fplai.data.fpl_api import polite_get  # deferred: module owned by another agent

    return _coerce_bytes(polite_get(url, min_interval_s=min_interval_s, cache_ttl_s=cache_ttl_s))


def _parse_clubelo_csv(payload: bytes) -> pd.DataFrame:
    """Parse a ClubElo CSV payload into a typed DataFrame.

    ``Rank`` becomes nullable Int64 (the API sends the string ``None``), ``Elo``
    float64, ``From``/``To`` datetime64. Trailing blank lines are dropped.
    """
    df = pd.read_csv(io.BytesIO(payload))
    expected = {"Rank", "Club", "Country", "Level", "Elo", "From", "To"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"unexpected ClubElo CSV; missing columns {sorted(missing)}")
    df = df.dropna(subset=["Club"]).reset_index(drop=True)
    df["Rank"] = pd.to_numeric(df["Rank"], errors="coerce").astype("Int64")
    df["Level"] = pd.to_numeric(df["Level"], errors="coerce").astype("Int64")
    df["Elo"] = pd.to_numeric(df["Elo"], errors="coerce").astype("float64")
    df["From"] = pd.to_datetime(df["From"], format="%Y-%m-%d", errors="coerce")
    df["To"] = pd.to_datetime(df["To"], format="%Y-%m-%d", errors="coerce")
    return df


class EloClient:
    """Client for api.clubelo.com; archives raw CSVs under ``RAW_DIR/clubelo/``."""

    def __init__(self, raw_dir: Path | None = None) -> None:
        """Args: raw_dir: override for ``RAW_DIR/clubelo`` (tests)."""
        self.raw_dir = raw_dir if raw_dir is not None else RAW_DIR / "clubelo"

    def _get_and_archive(self, path_segment: str, dest_name: str) -> bytes:
        payload = _fetch(f"{BASE_URL}/{path_segment}")
        dest = self.raw_dir / dest_name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(payload)
        return payload

    def team_history(self, club_name: str) -> pd.DataFrame:
        """Full Elo history for one club (one row per rating period).

        Args:
            club_name: FPL or ClubElo club name (``'Spurs'``, ``'Man City'``, ...).

        Returns:
            DataFrame with Rank/Club/Country/Level/Elo/From/To, oldest first.
            Raw CSV is saved to ``RAW_DIR/clubelo/{UrlName}.csv``.

        Raises:
            ValueError: if the response has no rows (unknown club names return
                an empty CSV rather than a 404).
        """
        url_name = clubelo_url_name(fpl_to_clubelo(club_name))
        df = _parse_clubelo_csv(self._get_and_archive(url_name, f"{url_name}.csv"))
        if df.empty:
            raise ValueError(f"ClubElo returned no history for {club_name!r} (/{url_name})")
        return df

    def snapshot(self, date: dt.date | str) -> pd.DataFrame:
        """Elo snapshot of every ranked club on ``date`` (``YYYY-MM-DD`` or date).

        Raw CSV is saved to ``RAW_DIR/clubelo/{YYYY-MM-DD}.csv``.
        """
        if isinstance(date, str):
            date = dt.date.fromisoformat(date)
        iso = date.isoformat()
        df = _parse_clubelo_csv(self._get_and_archive(iso, f"{iso}.csv"))
        if df.empty:
            raise ValueError(f"ClubElo returned an empty snapshot for {iso}")
        return df

    def pl_snapshot(self, date: dt.date | str) -> pd.DataFrame:
        """English top-flight rows of :meth:`snapshot` (Country ENG, Level 1)."""
        snap = self.snapshot(date)
        mask = (snap["Country"] == "ENG") & (snap["Level"] == 1)
        return snap.loc[mask].reset_index(drop=True)
