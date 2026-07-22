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

import html
import logging
import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd

from fplai.data import vaastav
from fplai.data.clubelo import PL_CLUBELO_NAMES

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


# --- team-name maps across sources -------------------------------------------------
#
# Deterministic FPL team ``name`` -> per-source spelling, verified against the real
# files on disk (2026-07-22): every distinct ``HomeTeam`` in football-data E0.csv
# 2016-17..2025-26, every ``teams[].title`` in Understat league JSON 2016..2025, and
# the ClubElo names probed live by the clubelo module.  Identity mappings are listed
# explicitly so each dict doubles as the roster of verified spellings; FPL names
# absent from a map (e.g. clubs promoted after 2025-26) resolve to NA and are logged.

#: FPL bootstrap ``team.name`` -> football-data.co.uk ``HomeTeam``/``AwayTeam``.
FOOTBALLDATA_TEAM_NAMES: dict[str, str] = {
    "Arsenal": "Arsenal",
    "Aston Villa": "Aston Villa",
    "Bournemouth": "Bournemouth",
    "Brentford": "Brentford",
    "Brighton": "Brighton",
    "Burnley": "Burnley",
    "Cardiff": "Cardiff",
    "Chelsea": "Chelsea",
    "Crystal Palace": "Crystal Palace",
    "Everton": "Everton",
    "Fulham": "Fulham",
    "Huddersfield": "Huddersfield",
    "Hull": "Hull",
    "Ipswich": "Ipswich",
    "Leeds": "Leeds",
    "Leicester": "Leicester",
    "Liverpool": "Liverpool",
    "Luton": "Luton",
    "Man City": "Man City",
    "Man Utd": "Man United",
    "Middlesbrough": "Middlesbrough",
    "Newcastle": "Newcastle",
    "Norwich": "Norwich",
    "Nott'm Forest": "Nott'm Forest",
    "Sheffield Utd": "Sheffield United",
    "Southampton": "Southampton",
    "Spurs": "Tottenham",
    "Stoke": "Stoke",
    "Sunderland": "Sunderland",
    "Swansea": "Swansea",
    "Watford": "Watford",
    "West Brom": "West Brom",
    "West Ham": "West Ham",
    "Wolves": "Wolves",
}

#: FPL bootstrap ``team.name`` -> Understat ``teams[].title`` / ``team_title``.
UNDERSTAT_TEAM_NAMES: dict[str, str] = {
    "Arsenal": "Arsenal",
    "Aston Villa": "Aston Villa",
    "Bournemouth": "Bournemouth",
    "Brentford": "Brentford",
    "Brighton": "Brighton",
    "Burnley": "Burnley",
    "Cardiff": "Cardiff",
    "Chelsea": "Chelsea",
    "Crystal Palace": "Crystal Palace",
    "Everton": "Everton",
    "Fulham": "Fulham",
    "Huddersfield": "Huddersfield",
    "Hull": "Hull",
    "Ipswich": "Ipswich",
    "Leeds": "Leeds",
    "Leicester": "Leicester",
    "Liverpool": "Liverpool",
    "Luton": "Luton",
    "Man City": "Manchester City",
    "Man Utd": "Manchester United",
    "Middlesbrough": "Middlesbrough",
    "Newcastle": "Newcastle United",
    "Norwich": "Norwich",
    "Nott'm Forest": "Nottingham Forest",
    "Sheffield Utd": "Sheffield United",
    "Southampton": "Southampton",
    "Spurs": "Tottenham",
    "Stoke": "Stoke",
    "Sunderland": "Sunderland",
    "Swansea": "Swansea",
    "Watford": "Watford",
    "West Brom": "West Bromwich Albion",
    "West Ham": "West Ham",
    "Wolves": "Wolverhampton Wanderers",
}

#: FPL bootstrap ``team.name`` -> ClubElo ``Club`` (delegated to the clubelo client's
#: verified roster, which also accepts common variants).
CLUBELO_TEAM_NAMES: dict[str, str] = dict(PL_CLUBELO_NAMES)

_AUX_NAME_MAPS: dict[str, dict[str, str]] = {
    "understat_name": UNDERSTAT_TEAM_NAMES,
    "clubelo_name": CLUBELO_TEAM_NAMES,
    "footballdata_name": FOOTBALLDATA_TEAM_NAMES,
}


def apply_team_name_maps(teams: pd.DataFrame) -> pd.DataFrame:
    """Fill ``understat_name`` / ``clubelo_name`` / ``footballdata_name`` on a teams frame.

    Returns a copy with each aux column recomputed from the deterministic maps above
    (existing values are overwritten — the maps are the source of truth).  FPL names
    missing from a map are left NA and logged once per name, so newly promoted clubs
    surface loudly instead of joining silently wrong.
    """
    out = teams.copy()
    for col, mapping in _AUX_NAME_MAPS.items():
        out[col] = out["name"].map(mapping).astype("string")
        unknown = sorted(out.loc[out[col].isna(), "name"].dropna().unique())
        if unknown:
            logger.warning("apply_team_name_maps: no %s mapping for %s", col, unknown)
    return out


# --- player fuzzy matching (FPL <-> Understat) --------------------------------------


def name_tokens(name: str) -> frozenset[str]:
    """Normalized token set of a person name (order/hyphen/accent-insensitive).

    HTML entities are unescaped first — vaastav's understat extracts ship names like
    ``"N&#039;Golo Kanté"``.
    """
    return frozenset(normalize_name(html.unescape(name)).split())


def _tokens_match(a: str, b: str) -> bool:
    """Soft equality of two single name tokens.

    Handles the systematic FPL <-> Understat renderings: initials from web_names
    (``j`` ~ ``jota``), diminutives that are prefixes (``ben`` ~ ``benjamin``,
    ``max`` ~ ``maximilian``), shared 4+ letter stems (``matty`` ~ ``matthew``)
    and transliteration drift (``nayef`` ~ ``naif``).
    """
    if a == b:
        return True
    if len(a) == 1 or len(b) == 1:
        return a[0] == b[0]
    shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
    if len(shorter) >= 3 and longer.startswith(shorter):
        return True
    if len(shorter) >= 4 and shorter[:4] == longer[:4]:
        return True
    return SequenceMatcher(None, a, b).ratio() >= 0.85


def _soft_matches(a: frozenset[str], b: frozenset[str]) -> int:
    """Count of one-to-one soft token pairings between two token sets (greedy)."""
    remaining = sorted(b)
    count = 0
    for tok in sorted(a):
        exact = tok in remaining
        chosen = tok if exact else next((r for r in remaining if _tokens_match(tok, r)), None)
        if chosen is not None:
            remaining.remove(chosen)
            count += 1
    return count


def _token_score(a: frozenset[str], b: frozenset[str]) -> float:
    """Similarity of two token sets in [0, 1]; 1.0 only for exact set equality."""
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    m = _soft_matches(a, b)
    smaller = min(len(a), len(b))
    jaccard = m / (len(a) + len(b) - m) if m else 0.0
    containment = m / smaller
    if m >= 2 and m == smaller:
        # One name is (softly) a multi-token subset of the other ("Bruno Fernandes"
        # vs "Bruno Miguel Borges Fernandes", "Diogo J." vs "Diogo Jota").
        return 0.9 + 0.1 * jaccard
    if m == 1 and smaller == 1:
        (single,) = a if len(a) == 1 else b
        if len(single) >= 4 and single in (a & b):
            # Understat mononym exactly contained in the FPL full name ("Ederson"
            # in "Ederson Santana de Moraes") — decisive within a team.
            return 0.75 + 0.1 * jaccard
    seq = SequenceMatcher(None, " ".join(sorted(a)), " ".join(sorted(b))).ratio()
    return 0.5 * jaccard + 0.2 * containment + 0.3 * seq


def name_similarity(fpl_variants: Iterable[str], understat_name: str) -> float:
    """Best token-set similarity between any FPL name variant and an Understat name.

    ``fpl_variants`` are alternative renderings of the same player (full name,
    ``web_name``, ...); the max over variants is returned so display names like
    ``"Son"`` or ``"B.Fernandes"`` get their fair chance.
    """
    target = name_tokens(understat_name)
    return max((_token_score(name_tokens(v), target) for v in fpl_variants), default=0.0)


@dataclass(frozen=True)
class PlayerMatch:
    """One accepted FPL-player <-> Understat-player match within a season."""

    player_code: int
    understat_id: int
    score: float
    method: str
    """``seed`` | ``exact`` | ``team_fuzzy`` | ``global_fuzzy``"""


#: Accept a same-team fuzzy candidate at or above this score.
TEAM_FUZZY_THRESHOLD = 0.72
#: Accept a cross-team (transfer window / team-name miss) candidate only here.
GLOBAL_FUZZY_THRESHOLD = 0.92


def match_players(
    fpl: pd.DataFrame,
    understat: pd.DataFrame,
    *,
    seeds: dict[int, int] | None = None,
) -> list[PlayerMatch]:
    """Match one season's FPL roster to its Understat player list, deterministically.

    Args:
        fpl: One row per FPL player with columns ``player_code`` (int),
            ``variants`` (list[str] of name renderings) and ``teams``
            (set[str] of Understat-style team titles the player appeared for).
        understat: One row per Understat player with columns ``understat_id`` (int),
            ``player_name`` (str) and ``teams`` (set[str] of team titles).
        seeds: Known-good ``player_code -> understat_id`` pairs (vaastav id_dict,
            prior-season matches).  Consumed first and never overridden.

    Returns:
        Accepted matches, one-to-one on both sides, ordered deterministically.

    The passes, in order: seeds; unique exact normalized-name equality (any team —
    robust to mid-season transfers); same-team fuzzy (greedy best-first, threshold
    :data:`TEAM_FUZZY_THRESHOLD`); global fuzzy at :data:`GLOBAL_FUZZY_THRESHOLD`.
    """
    fpl = fpl.sort_values("player_code").reset_index(drop=True)
    understat = understat.sort_values("understat_id").reset_index(drop=True)
    us_ids = set(understat["understat_id"].astype(int))
    matched: list[PlayerMatch] = []
    taken_fpl: set[int] = set()
    taken_us: set[int] = set()

    fpl_codes = set(fpl["player_code"].astype(int))
    for code, us_id in sorted((seeds or {}).items()):
        if code not in fpl_codes:
            continue
        # A seeded player is out of the fuzzy pool even when their id is absent from
        # this season's Understat list (they simply did not appear that season).
        taken_fpl.add(code)
        if us_id in us_ids:
            matched.append(PlayerMatch(code, us_id, 1.0, "seed"))
            taken_us.add(us_id)

    # Pass 2: unique exact normalized-name matches, team-independent.
    fpl_by_tokens: dict[frozenset[str], list[int]] = {}
    for code, variants in zip(fpl["player_code"].astype(int), fpl["variants"], strict=True):
        if code in taken_fpl:
            continue
        for v in variants:
            toks = name_tokens(v)
            if toks:
                fpl_by_tokens.setdefault(toks, []).append(code)
    us_by_tokens: dict[frozenset[str], list[int]] = {}
    for us_id, name in zip(understat["understat_id"].astype(int), understat["player_name"],
                           strict=True):
        if us_id in taken_us:
            continue
        toks = name_tokens(str(name))
        if toks:
            us_by_tokens.setdefault(toks, []).append(us_id)
    for toks in sorted(us_by_tokens, key=lambda t: " ".join(sorted(t))):
        us_side = us_by_tokens[toks]
        fpl_side = sorted(set(fpl_by_tokens.get(toks, [])))
        if len(us_side) == 1 and len(fpl_side) == 1:
            code, us_id = fpl_side[0], us_side[0]
            if code not in taken_fpl and us_id not in taken_us:
                matched.append(PlayerMatch(code, us_id, 1.0, "exact"))
                taken_fpl.add(code)
                taken_us.add(us_id)

    # Passes 3 & 4: scored candidates, greedy best-first (deterministic tie-break).
    def _greedy(pairs: list[tuple[float, int, int]], threshold: float, method: str) -> None:
        for score, code, us_id in sorted(pairs, key=lambda p: (-p[0], p[1], p[2])):
            if score < threshold:
                break
            if code in taken_fpl or us_id in taken_us:
                continue
            matched.append(PlayerMatch(code, us_id, score, method))
            taken_fpl.add(code)
            taken_us.add(us_id)

    fpl_open = fpl[~fpl["player_code"].astype(int).isin(taken_fpl)]
    us_open = understat[~understat["understat_id"].astype(int).isin(taken_us)]
    team_pairs: list[tuple[float, int, int]] = []
    for us_id, name, us_teams in zip(
        us_open["understat_id"].astype(int), us_open["player_name"], us_open["teams"],
        strict=True,
    ):
        for code, variants, fpl_teams in zip(
            fpl_open["player_code"].astype(int), fpl_open["variants"], fpl_open["teams"],
            strict=True,
        ):
            if not (set(us_teams) & set(fpl_teams)):
                continue
            score = name_similarity(variants, str(name))
            if score >= TEAM_FUZZY_THRESHOLD:
                team_pairs.append((score, code, us_id))
    _greedy(team_pairs, TEAM_FUZZY_THRESHOLD, "team_fuzzy")

    # Pass 3b: unique surname within a team.  Catches nickname first names the soft
    # token rules cannot bridge ("Joe"/"Joseph" Gomez, "Tino"/"Valentino"
    # Livramento, "Kiko" Femenía): the Understat surname (last name token, 4+
    # letters) appears in exactly one open FPL candidate's tokens on that team, and
    # no other open Understat player on the team shares it.
    fpl_open = fpl[~fpl["player_code"].astype(int).isin(taken_fpl)]
    us_open = understat[~understat["understat_id"].astype(int).isin(taken_us)]
    surname_pairs: list[tuple[float, int, int]] = []
    for us_id, name, us_teams in zip(
        us_open["understat_id"].astype(int), us_open["player_name"], us_open["teams"],
        strict=True,
    ):
        toks = normalize_name(html.unescape(str(name))).split()
        if not toks or len(toks[-1]) < 4:
            continue
        surname = toks[-1]
        us_rivals = sum(
            1
            for other_name, other_teams in zip(us_open["player_name"], us_open["teams"],
                                               strict=True)
            if (set(other_teams) & set(us_teams))
            and normalize_name(html.unescape(str(other_name))).split()[-1:] == [surname]
        )
        if us_rivals != 1:  # itself only
            continue
        candidates = [
            code
            for code, variants, fpl_teams in zip(
                fpl_open["player_code"].astype(int), fpl_open["variants"], fpl_open["teams"],
                strict=True,
            )
            if (set(us_teams) & set(fpl_teams))
            and any(surname in name_tokens(v) for v in variants)
        ]
        if len(candidates) == 1:
            surname_pairs.append((0.7, candidates[0], us_id))
    _greedy(surname_pairs, 0.7, "surname")

    fpl_open = fpl[~fpl["player_code"].astype(int).isin(taken_fpl)]
    us_open = understat[~understat["understat_id"].astype(int).isin(taken_us)]
    global_pairs: list[tuple[float, int, int]] = []
    for us_id, name in zip(us_open["understat_id"].astype(int), us_open["player_name"],
                           strict=True):
        for code, variants in zip(fpl_open["player_code"].astype(int), fpl_open["variants"],
                                  strict=True):
            score = name_similarity(variants, str(name))
            if score >= GLOBAL_FUZZY_THRESHOLD:
                global_pairs.append((score, code, us_id))
    _greedy(global_pairs, GLOBAL_FUZZY_THRESHOLD, "global_fuzzy")

    return sorted(matched, key=lambda m: (m.player_code, m.understat_id))
