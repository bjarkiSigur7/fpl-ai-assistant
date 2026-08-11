"""football-data.co.uk downloader and odds parser (E0 = Premier League).

Downloads per-season ``E0.csv`` files from
``https://www.football-data.co.uk/mmz4281/{yy}{yy}/E0.csv`` (e.g. ``2526`` for
2025-26) into ``RAW_DIR/football_data/{season}/E0.csv`` and parses them into the
``odds.parquet`` schema defined in ARCHITECTURE.md.

Column availability (inspected on real files, July 2026):

* 1993-94 .. 1999-00: match results only, no odds columns.
* 2000-01 .. 2004-05: bookmaker 1X2 odds (WH/LB/GB/IW/SB), no Bet365, no O/U.
* 2005-06 .. 2015-16: ``B365H/D/A`` plus Betbrain aggregates ``BbAvH/D/A`` and
  over/under 2.5 as ``BbAv>2.5`` / ``BbAv<2.5``. (``B365H`` already present 2002-03+.)
* 2016-17 .. 2018-19: adds Pinnacle ``PSH/PSD/PSA`` and *closing* ``PSCH/PSCD/PSCA``;
  O/U still only via Betbrain averages.
* 2019-20 onward: full closing set — ``PSCH/PSCD/PSCA``, ``B365CH/...``,
  ``AvgCH/...``, and closing O/U 2.5 ``PC>2.5`` / ``PC<2.5``, ``B365C>2.5`` etc.

Chosen 1X2 source: **Pinnacle closing (PSCH/PSCD/PSCA)** first — sharpest book,
present for the default training window 2016-17+ — falling back through Bet365
closing, market-average closing, then pre-closing Pinnacle/Bet365/averages for
older seasons. O/U 2.5 analogously (``PC>2.5`` first). Fallbacks are applied
triple-wise per row so home/draw/away always come from the same bookmaker.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
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

from fplai.config import PROCESSED_DIR, RAW_DIR

logger = logging.getLogger(__name__)

BASE_URL = "https://www.football-data.co.uk/mmz4281/{code}/E0.csv"

#: First season available on football-data.co.uk (1993-94).
FIRST_SEASON = 1993
#: Latest season start year we accept (2026 == 2026-27; appears each August).
LAST_SEASON = 2026
#: Default backfill window: 2016-17 .. 2025-26 (matches the FPL training corpus).
DEFAULT_SEASONS: tuple[int, ...] = tuple(range(2016, 2026))

# Candidate (home, draw, away) column triples for closing 1X2 odds, best first.
CLOSING_1X2_CANDIDATES: tuple[tuple[str, str, str], ...] = (
    ("PSCH", "PSCD", "PSCA"),  # Pinnacle closing, 2016-17+
    ("B365CH", "B365CD", "B365CA"),  # Bet365 closing, 2019-20+
    ("AvgCH", "AvgCD", "AvgCA"),  # market-average closing, 2019-20+
    ("PSH", "PSD", "PSA"),  # Pinnacle (pre-closing capture), ~2012-13+
    ("B365H", "B365D", "B365A"),  # Bet365, ~2002-03+
    ("BbAvH", "BbAvD", "BbAvA"),  # Betbrain average, ~2005-06..2018-19
    ("AvgH", "AvgD", "AvgA"),  # market average, 2019-20+
    ("WHH", "WHD", "WHA"),  # William Hill, 2000-01+
    ("LBH", "LBD", "LBA"),  # Ladbrokes, early 2000s
)

# Candidate (over, under) column pairs for over/under 2.5 goals, best first.
OU25_CANDIDATES: tuple[tuple[str, str], ...] = (
    ("PC>2.5", "PC<2.5"),  # Pinnacle closing, 2019-20+
    ("B365C>2.5", "B365C<2.5"),  # Bet365 closing, 2019-20+
    ("AvgC>2.5", "AvgC<2.5"),  # market-average closing, 2019-20+
    ("P>2.5", "P<2.5"),  # Pinnacle, 2019-20+
    ("B365>2.5", "B365<2.5"),  # Bet365
    ("Avg>2.5", "Avg<2.5"),  # market average
    ("BbAv>2.5", "BbAv<2.5"),  # Betbrain average, ~2005-06..2018-19
)

#: Column order of the odds.parquet contract (implied probabilities appended by
#: :func:`build_odds_table`; ``fpl_fixture_id`` is filled by build/crosswalk).
ODDS_COLUMNS: tuple[str, ...] = (
    "season",
    "date",
    "home_footballdata_name",
    "away_footballdata_name",
    "odds_h",
    "odds_d",
    "odds_a",
    "odds_over25",
    "odds_under25",
)


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


def season_code(season: int) -> str:
    """Return the football-data URL code for a season start year, e.g. 2025 -> ``'2526'``."""
    _validate_season(season)
    return f"{season % 100:02d}{(season + 1) % 100:02d}"


def _validate_season(season: int) -> None:
    if not FIRST_SEASON <= season <= LAST_SEASON:
        raise ValueError(
            f"season must be a start year in {FIRST_SEASON}..{LAST_SEASON}, got {season!r}"
        )


def e0_url(season: int) -> str:
    """Return the download URL for a season's Premier League (E0) CSV."""
    return BASE_URL.format(code=season_code(season))


def e0_path(season: int, raw_dir: Path | None = None) -> Path:
    """Return the on-disk path for a season's raw E0.csv."""
    base = raw_dir if raw_dir is not None else RAW_DIR / "football_data"
    return base / str(season) / "E0.csv"


def download_e0(season: int, *, raw_dir: Path | None = None, force: bool = False) -> Path:
    """Download one season's E0.csv into the raw data dir; idempotent unless ``force``.

    Args:
        season: Season start year (1993..2026).
        raw_dir: Override for ``RAW_DIR/football_data`` (tests).
        force: Re-download even if the file already exists.

    Returns:
        Path of the written (or pre-existing) CSV.
    """
    dest = e0_path(season, raw_dir)
    if dest.exists() and not force:
        logger.debug("E0.csv for %s already present at %s", season, dest)
        return dest
    url = e0_url(season)
    logger.info("Downloading %s -> %s", url, dest)
    payload = _fetch(url)
    if not payload.lstrip(b"\xef\xbb\xbf").startswith(b"Div,"):
        raise ValueError(f"unexpected E0.csv payload from {url} (first bytes: {payload[:40]!r})")
    # Guard against football-data's pre-publication redirect: before a season's E0.csv
    # exists the URL 301s to another division's file (observed live 2026-08-11:
    # mmz4281/2627/E0.csv -> EC.csv, the National League) whose header also starts
    # with "Div," — so require the DATA rows to actually be division E0.
    body = payload.lstrip(b"\xef\xbb\xbf")
    data_lines = [ln for ln in body.splitlines()[1:] if ln.strip()]
    if data_lines and not all(ln.split(b",", 1)[0] == b"E0" for ln in data_lines):
        divisions = sorted({ln.split(b",", 1)[0].decode("latin-1") for ln in data_lines})
        raise ValueError(
            f"payload from {url} is not division E0 (rows carry {divisions}) — the "
            "season's file is likely not published yet and the request was redirected"
        )
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(payload)
    return dest


def _read_e0_csv(path: Path) -> pd.DataFrame:
    """Read a raw E0.csv robustly (BOM on recent files, latin-1 on some old ones)."""
    try:
        df = pd.read_csv(path, encoding="utf-8-sig", on_bad_lines="skip")
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="latin-1", on_bad_lines="skip")
    # Drop trailing blank rows (old files end with lines of bare commas).
    df = df.dropna(subset=["Date", "HomeTeam", "AwayTeam"]).reset_index(drop=True)
    return df


def _parse_dates(raw: pd.Series) -> pd.Series:
    """Parse football-data dates: dd/mm/yyyy (2017-18+) or dd/mm/yy (older)."""
    dates = pd.to_datetime(raw, format="%d/%m/%Y", errors="coerce")
    fallback = pd.to_datetime(raw, format="%d/%m/%y", errors="coerce")
    dates = dates.fillna(fallback)
    if dates.isna().all():
        raise ValueError("could not parse any Date values from E0.csv")
    return dates


def _coalesce_triples(
    df: pd.DataFrame, candidates: Sequence[tuple[str, ...]]
) -> list[pd.Series]:
    """Row-wise coalesce over candidate column tuples, keeping tuples intact per row.

    For each row the first candidate tuple whose columns are all present *and*
    non-null wins, so e.g. home/draw/away odds always come from one bookmaker.
    """
    n = len(candidates[0])
    out = [pd.Series(pd.NA, index=df.index, dtype="Float64") for _ in range(n)]
    for cols in candidates:
        if not all(c in df.columns for c in cols):
            continue
        vals = [pd.to_numeric(df[c], errors="coerce").astype("Float64") for c in cols]
        complete = pd.concat(vals, axis=1).notna().all(axis=1)
        fill = out[0].isna() & complete
        for i in range(n):
            out[i] = out[i].mask(fill, vals[i])
    return out


def parse_odds(season: int, *, raw_dir: Path | None = None) -> pd.DataFrame:
    """Parse one season's E0.csv into the odds.parquet schema (downloads if missing).

    Returns a DataFrame with columns :data:`ODDS_COLUMNS`: ``season``, ``date``,
    ``home_footballdata_name``, ``away_footballdata_name``, closing 1X2
    ``odds_h/odds_d/odds_a`` and O/U 2.5 ``odds_over25/odds_under25`` (nullable
    Float64 — all-NA for seasons predating the market).
    """
    path = download_e0(season, raw_dir=raw_dir)
    df = _read_e0_csv(path)
    odds_h, odds_d, odds_a = _coalesce_triples(df, CLOSING_1X2_CANDIDATES)
    odds_over, odds_under = _coalesce_triples(df, OU25_CANDIDATES)
    out = pd.DataFrame(
        {
            "season": pd.Series(season, index=df.index, dtype="int64"),
            "date": _parse_dates(df["Date"]),
            "home_footballdata_name": df["HomeTeam"].astype("string"),
            "away_footballdata_name": df["AwayTeam"].astype("string"),
            "odds_h": odds_h,
            "odds_d": odds_d,
            "odds_a": odds_a,
            "odds_over25": odds_over,
            "odds_under25": odds_under,
        }
    )
    return out.sort_values("date", kind="stable").reset_index(drop=True)


def demargin(odds: pd.DataFrame, cols: Sequence[str], prefix: str = "p_") -> pd.DataFrame:
    """Add proportionally de-margined implied probabilities for an odds column set.

    ``p_i = (1/o_i) / sum_j (1/o_j)`` — the basic proportional (multiplicative)
    de-margin. Output columns are ``prefix`` + the column name stripped of its
    ``odds_`` prefix (``odds_h`` -> ``p_h``). Rows with any missing odds get NA.
    """
    inv = pd.concat([1.0 / odds[c] for c in cols], axis=1)
    total = inv.sum(axis=1, skipna=False)
    out = odds.copy()
    for c in cols:
        out[prefix + c.removeprefix("odds_")] = (1.0 / odds[c]) / total
    return out


def build_odds_table(
    seasons: Sequence[int] | None = None,
    *,
    raw_dir: Path | None = None,
    out_path: Path | None = None,
) -> Path:
    """Build ``processed/odds.parquet`` across seasons, with implied probabilities.

    Downloads any missing raw files, parses each season via :func:`parse_odds`,
    appends de-margined probabilities ``p_h/p_d/p_a`` and ``p_over25/p_under25``,
    and an all-NA nullable ``fpl_fixture_id`` column (resolved later by the
    build/crosswalk stage). Deterministic and idempotent.

    Args:
        seasons: Season start years; defaults to :data:`DEFAULT_SEASONS` (2016..2025).
        raw_dir: Override for ``RAW_DIR/football_data`` (tests).
        out_path: Override for ``PROCESSED_DIR/odds.parquet`` (tests).

    Returns:
        Path of the written parquet file.
    """
    chosen = tuple(seasons) if seasons is not None else DEFAULT_SEASONS
    for s in chosen:
        _validate_season(s)
    frames = [parse_odds(s, raw_dir=raw_dir) for s in chosen]
    table = pd.concat(frames, ignore_index=True)
    table = demargin(table, ["odds_h", "odds_d", "odds_a"])
    table = demargin(table, ["odds_over25", "odds_under25"])
    table["fpl_fixture_id"] = pd.Series(pd.NA, index=table.index, dtype="Int64")
    table = table.sort_values(["season", "date"], kind="stable").reset_index(drop=True)
    dest = out_path if out_path is not None else PROCESSED_DIR / "odds.parquet"
    dest.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(dest, index=False)
    logger.info("Wrote %d rows across %d seasons to %s", len(table), len(chosen), dest)
    return dest
