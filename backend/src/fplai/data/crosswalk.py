"""Identity resolution across data sources.

Builds the two canonical identity tables from the raw vaastav mirror:

* ``players.parquet`` — one row per player, keyed by the cross-season-stable FPL
  ``element.code`` (``player_code``), with the Understat id where vaastav shipped an
  id map (``id_dict.csv``, 2021-22/2022-23 only) and ``opta_code`` where the season
  dump carries it.
* ``teams.parquet`` — one row per team-season with the per-season FPL team id and the
  stable ``team_code``.  For 2016-17..2018-19 (no ``teams.csv`` upstream) the rows are
  derived from ``players_raw.csv`` (which carries ``team``/``team_code`` per player)
  plus the repo-level ``master_team_list.csv`` for names; short names are backfilled
  by ``team_code`` from seasons that do ship ``teams.csv``.

The ``understat_name`` / ``clubelo_name`` / ``footballdata_name`` columns are left
nullable here — the auxiliary-source clients join those in later using
:func:`normalize_name` for fuzzy matching.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from fplai.data import vaastav

logger = logging.getLogger(__name__)

_NON_ALNUM = re.compile(r"[^a-z0-9]+")

#: Stroked/ligature letters that NFKD does not decompose to ASCII (Ødegaard, Højlund,
#: Łukasz, Đorđe ...) — translated before the diacritic strip.
_SPECIAL_LETTERS = str.maketrans(
    {
        "Ø": "O",
        "ø": "o",
        "Æ": "AE",
        "æ": "ae",
        "Œ": "OE",
        "œ": "oe",
        "Ð": "D",
        "ð": "d",
        "Þ": "Th",
        "þ": "th",
        "ß": "ss",
        "Đ": "D",
        "đ": "d",
        "Ħ": "H",
        "ħ": "h",
        "Ł": "L",
        "ł": "l",
        "Ŧ": "T",
        "ŧ": "t",
    }
)

#: FPL element_type -> canonical position string. 5 (2024-25 assistant managers) is
#: intentionally absent: manager elements are not players and are filtered out.
ELEMENT_TYPE_TO_POSITION: dict[int, str] = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}


def normalize_name(name: str) -> str:
    """Normalize a person/team name for fuzzy joins across sources.

    Strips unicode diacritics (NFKD -> ASCII), lowercases, and collapses every run of
    non-alphanumeric characters to a single space (``"Ødegaard, Martin" ->
    "odegaard martin"``).  Used later to seed Understat/ClubElo/football-data joins.
    """
    translated = name.translate(_SPECIAL_LETTERS)
    ascii_ = unicodedata.normalize("NFKD", translated).encode("ascii", "ignore").decode("ascii")
    return _NON_ALNUM.sub(" ", ascii_.lower()).strip()


def discover_seasons(raw_root: Path | None = None) -> list[int]:
    """Season start years that have a downloaded vaastav directory locally."""
    root = vaastav.vaastav_root(raw_root)
    seasons: list[int] = []
    if not root.exists():
        return seasons
    for entry in sorted(root.iterdir()):
        if entry.is_dir() and re.fullmatch(r"\d{4}-\d{2}", entry.name):
            seasons.append(int(entry.name[:4]))
    return seasons


def resolve_seasons(seasons: Iterable[int] | None, raw_root: Path | None = None) -> list[int]:
    """Sorted season list: the given seasons, or all locally downloaded ones."""
    resolved = sorted(seasons) if seasons is not None else discover_seasons(raw_root)
    if not resolved:
        raise FileNotFoundError(
            f"No vaastav season directories under {vaastav.vaastav_root(raw_root)}; "
            "run vaastav.download_all() first"
        )
    return resolved


def _read_players_raw(season: int, raw_root: Path | None) -> pd.DataFrame:
    path = vaastav.season_dir(season, raw_root) / "players_raw.csv"
    return vaastav.read_csv_tolerant(path)


def _teams_from_teams_csv(season: int, path: Path) -> pd.DataFrame:
    teams = vaastav.read_csv_tolerant(path)
    return pd.DataFrame(
        {
            "season": season,
            "fpl_team_id": teams["id"].astype("int64"),
            "team_code": teams["code"].astype("int64"),
            "name": teams["name"].astype("string"),
            "short_name": teams["short_name"].astype("string"),
        }
    )


def _teams_from_players_raw(season: int, raw_root: Path | None) -> pd.DataFrame:
    """Derive team-season rows for seasons without teams.csv (2016-17..2018-19)."""
    players = _read_players_raw(season, raw_root)
    teams = (
        players[["team", "team_code"]]
        .drop_duplicates()
        .rename(columns={"team": "fpl_team_id"})
        .astype("int64")
        .sort_values("fpl_team_id")
        .reset_index(drop=True)
    )
    teams.insert(0, "season", season)

    master_path = vaastav.vaastav_root(raw_root) / vaastav.MASTER_TEAM_LIST
    if not master_path.exists():
        raise FileNotFoundError(
            f"{master_path} is required to name {vaastav.season_label(season)} teams; "
            "run vaastav.download_master_team_list() first"
        )
    master = vaastav.read_csv_tolerant(master_path)
    label = vaastav.season_label(season)
    names = master.loc[master["season"] == label, ["team", "team_name"]]
    name_map = dict(zip(names["team"].astype(int), names["team_name"], strict=True))
    teams["name"] = teams["fpl_team_id"].map(name_map).astype("string")
    teams["short_name"] = pd.Series(pd.NA, index=teams.index, dtype="string")
    return teams


def build_teams_crosswalk(
    seasons: Iterable[int] | None = None,
    raw_root: Path | None = None,
) -> pd.DataFrame:
    """Build the canonical team-season table (``teams.parquet`` schema).

    Columns: ``season``, ``fpl_team_id`` (per-season), ``team_code`` (stable),
    ``name``, ``short_name``, plus nullable ``understat_name`` / ``clubelo_name`` /
    ``footballdata_name`` placeholders for the aux-source joins.
    """
    frames = []
    for season in resolve_seasons(seasons, raw_root):
        teams_csv = vaastav.season_dir(season, raw_root) / "teams.csv"
        if teams_csv.exists():
            frames.append(_teams_from_teams_csv(season, teams_csv))
        else:
            frames.append(_teams_from_players_raw(season, raw_root))
    teams = pd.concat(frames, ignore_index=True)

    # Backfill names/short names by stable team_code from seasons that have them
    # (e.g. Arsenal 2016-17 gets "ARS" from its 2019-20+ teams.csv rows).
    for col in ("name", "short_name"):
        known = teams.dropna(subset=[col]).sort_values("season")
        fill = known.groupby("team_code")[col].last()
        teams[col] = teams[col].fillna(teams["team_code"].map(fill)).astype("string")

    for col in ("understat_name", "clubelo_name", "footballdata_name"):
        teams[col] = pd.Series(pd.NA, index=teams.index, dtype="string")

    return teams.sort_values(["season", "fpl_team_id"]).reset_index(drop=True)


def _understat_map(season: int, raw_root: Path | None) -> dict[int, int]:
    """player_code -> understat_id from one season's id_dict.csv (where shipped)."""
    id_dict_path = vaastav.season_dir(season, raw_root) / "id_dict.csv"
    if not id_dict_path.exists():
        return {}
    # Header upstream is "Understat_ID, FPL_ID, Understat_Name, FPL_Name" (note the
    # spaces after commas) — hence skipinitialspace + column-name strip.
    id_dict = vaastav.read_csv_tolerant(id_dict_path, skipinitialspace=True)
    id_dict.columns = [c.strip() for c in id_dict.columns]
    players = _read_players_raw(season, raw_root)
    code_by_element = dict(
        zip(players["id"].astype(int), players["code"].astype(int), strict=True)
    )
    out: dict[int, int] = {}
    for fpl_id, us_id in zip(id_dict["FPL_ID"], id_dict["Understat_ID"], strict=True):
        code = code_by_element.get(int(fpl_id))
        if code is not None:
            out[code] = int(us_id)
    return out


def build_players_crosswalk(
    seasons: Iterable[int] | None = None,
    raw_root: Path | None = None,
) -> pd.DataFrame:
    """Build the canonical player identity table (``players.parquet`` schema).

    One row per ``player_code`` (FPL ``element.code`` — the only stable cross-season
    player key); names come from the most recent season the player appears in.
    ``understat_id`` is filled from vaastav's ``id_dict.csv`` maps where shipped and
    left NA otherwise (the Understat client fuzzy-joins the rest later).
    ``opta_code`` is taken from season dumps that carry the column.
    """
    resolved = resolve_seasons(seasons, raw_root)
    frames = []
    for season in resolved:
        players = _read_players_raw(season, raw_root)
        # 2024-25 ships assistant-manager elements (element_type 5) — not players.
        players = players[players["element_type"].astype(int) <= 4]
        frame = pd.DataFrame(
            {
                "player_code": players["code"].astype("int64"),
                "web_name": players["web_name"].astype("string"),
                "first_name": players["first_name"].astype("string"),
                "second_name": players["second_name"].astype("string"),
                "opta_code": (
                    players["opta_code"].astype("string")
                    if "opta_code" in players.columns
                    else pd.Series(pd.NA, index=players.index, dtype="string")
                ),
            }
        )
        frame["season"] = season
        frames.append(frame)
    all_rows = pd.concat(frames, ignore_index=True)
    latest = (
        all_rows.sort_values(["player_code", "season"], kind="mergesort")
        .groupby("player_code", as_index=False)
        .last()
        .drop(columns=["season"])
    )

    understat: dict[int, int] = {}
    for season in resolved:  # later seasons win
        understat.update(_understat_map(season, raw_root))
    latest["understat_id"] = latest["player_code"].map(understat).astype("Int64")

    cols = ["player_code", "web_name", "first_name", "second_name", "understat_id", "opta_code"]
    return latest[cols].sort_values("player_code").reset_index(drop=True)
