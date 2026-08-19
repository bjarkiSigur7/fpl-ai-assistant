"""Resolve Gemini-recognized squad entries onto live-roster player codes.

Pure matching logic (no I/O, no network) so it stays unit-testable: the API layer
feeds it the recognized :class:`~fplai.data.gemini.SeenPlayer` list plus the
``live_roster`` frame and the team-short lookup, and gets back one
:class:`MatchedPlayer` per recognized card.

Matching is fuzzy by necessity — screenshots print FPL ``web_name`` ("M.Salah",
"Van Dijk"), sometimes with OCR wobble ("Odegaard" for "Ødegaard") — so each
seen/roster pair is scored on normalized-name similarity with small nudges from
the position row (reliable in screenshots), the printed price and the shirt-derived
club. Codes are assigned greedily best-score-first, one seen card per code, and a
pair below :data:`ACCEPT_SCORE` stays unmatched rather than guessing.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import TYPE_CHECKING

from fplai.data.gemini import SeenPlayer

if TYPE_CHECKING:
    import pandas as pd

#: A pair must reach this score to claim a player code.
ACCEPT_SCORE = 0.72
#: Name similarity below this never enters the candidate pool.
MIN_NAME_SCORE = 0.55

#: Characters NFKD won't decompose to ASCII — mapped by hand before stripping marks.
_CHAR_MAP = str.maketrans(
    {
        "ø": "o",
        "Ø": "O",
        "đ": "d",
        "Đ": "D",
        "ł": "l",
        "Ł": "L",
        "ß": "ss",
        "æ": "ae",
        "Æ": "AE",
        "œ": "oe",
        "Œ": "OE",
    }
)


@dataclass(frozen=True)
class MatchedPlayer:
    """One recognized card resolved (or not) against the roster."""

    seen: SeenPlayer
    player_code: int | None
    web_name: str | None
    team_short: str | None
    score: float


def normalize_name(name: str) -> str:
    """Lowercase ASCII words: strip accents, drop punctuation, collapse spaces."""
    decomposed = unicodedata.normalize("NFKD", name.translate(_CHAR_MAP))
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    cleaned = "".join(c if c.isalnum() else " " for c in stripped.lower())
    return " ".join(cleaned.split())


def _name_score(seen_norm: str, web_norm: str, full_norm: str) -> float:
    if not seen_norm:
        return 0.0
    if seen_norm in (web_norm, full_norm):
        return 1.0
    best = max(
        SequenceMatcher(None, seen_norm, web_norm).ratio(),
        SequenceMatcher(None, seen_norm, full_norm).ratio(),
    )
    # Whole-token containment ("Salah" ⊂ "mohamed salah") beats raw edit distance.
    for target in (web_norm, full_norm):
        if f" {seen_norm} " in f" {target} ":
            best = max(best, 0.92)
    return best


def _pair_score(
    seen: SeenPlayer, name_score: float, position: str, price: int, short: str | None
) -> float:
    score = name_score
    if seen.position is not None:
        score += 0.08 if seen.position == position else -0.25
    if seen.price is not None:
        diff = abs(price / 10 - seen.price)
        if diff <= 0.2:
            score += 0.08
        elif diff > 1.5:
            score -= 0.15
    if seen.club is not None and short is not None:
        score += 0.08 if seen.club.strip().upper() == short.upper() else -0.05
    return score


def match_squad(
    seen: list[SeenPlayer], roster: pd.DataFrame, shorts: dict[int, str]
) -> list[MatchedPlayer]:
    """Assign each seen card the best-scoring unclaimed roster code (greedy, unique)."""
    rows = [
        {
            "code": int(r.player_code),
            "web_name": str(r.web_name),
            "web_norm": normalize_name(str(r.web_name)),
            "full_norm": normalize_name(f"{r.first_name} {r.second_name}"),
            "position": str(r.position),
            "price": int(r.price),
            "short": shorts.get(int(r.team_code)),
        }
        for r in roster.itertuples(index=False)
    ]

    candidates: list[tuple[float, int, int]] = []  # (score, seen_idx, row_idx)
    for i, sp in enumerate(seen):
        sp_norm = normalize_name(sp.name)
        for j, row in enumerate(rows):
            name_score = _name_score(sp_norm, row["web_norm"], row["full_norm"])
            if name_score < MIN_NAME_SCORE:
                continue
            score = _pair_score(sp, name_score, row["position"], row["price"], row["short"])
            if score >= ACCEPT_SCORE:
                candidates.append((score, i, j))

    assigned: dict[int, tuple[float, dict]] = {}  # seen_idx -> (score, roster row)
    used_codes: set[int] = set()
    for score, i, j in sorted(candidates, key=lambda t: t[0], reverse=True):
        row = rows[j]
        if i in assigned or row["code"] in used_codes:
            continue
        assigned[i] = (score, row)
        used_codes.add(row["code"])

    out: list[MatchedPlayer] = []
    for i, sp in enumerate(seen):
        if i in assigned:
            score, row = assigned[i]
            out.append(
                MatchedPlayer(
                    seen=sp,
                    player_code=row["code"],
                    web_name=row["web_name"],
                    team_short=row["short"],
                    score=round(min(score, 1.0), 3),
                )
            )
        else:
            out.append(
                MatchedPlayer(seen=sp, player_code=None, web_name=None, team_short=None, score=0.0)
            )
    return out
