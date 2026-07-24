"""Rate an arbitrary 15-man squad against the dream-team benchmark (stage-5 support).

:func:`rate_team` scores a submitted squad over the live prediction window with a pure,
deterministic metric::

    metric(squad) = sum over the window's GWs of (best-legal-XI xP sum + captain bonus)

The best XI for a GW is chosen greedily — formation minimums (1 GKP/3 DEF/2 MID/1 FWD)
by xP first, then filled to 11 by xP respecting the maxes (1 GKP/5 DEF/5 MID/3 FWD) —
and the captain bonus is the highest xP within that XI.  The score normalises the
metric between two anchors evaluated with the SAME metric::

    score = clamp01((metric(team) - metric(floor)) / (metric(optimal) - metric(floor))) * 100

``optimal`` is the squad from ``data/processed/dream_team.json`` and ``floor`` is the
cheapest legal 15 from the live roster.  A user squad CAN outscore the dream team on
this metric — the dream team is solved against a decayed, chip-aware, bench-weighted
MILP objective over its own horizon, not this undecayed XI-sum — so the score is
clamped to 100 rather than treated as an error.

Everything here is pure and deterministic: no I/O, no clock, every tie broken by
``player_code``.  Players with no prediction rows in the window contribute 0 xP but
remain legal picks (e.g. brand-new signings the model has not scored yet); ``q0`` is
taken from the player's first prediction row within the window, where present.
"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, Literal, NamedTuple

import pandas as pd
from pydantic import BaseModel, Field

from fplai import rules
from fplai.optimizer.milp import (
    CLUB_LIMIT,
    FORMATION_MAX,
    FORMATION_MIN,
    INITIAL_BUDGET,
    POSITION_QUOTA,
    SQUAD_SIZE,
    XI_SIZE,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

__all__ = [
    "PlayerRating",
    "RatingResult",
    "TeamValidationError",
    "VERDICT_BANDS",
    "WeakestPlayer",
    "XiSelection",
    "best_xi",
    "cheapest_legal_squad",
    "rate_team",
    "squad_metric",
    "validate_squad",
]

#: Score bands, checked in order: first threshold the score reaches wins.
VERDICT_BANDS: tuple[tuple[float, str], ...] = (
    (95.0, "ELITE"),
    (85.0, "STRONG"),
    (70.0, "SOLID"),
    (50.0, "ROUGH"),
)

Verdict = Literal["ELITE", "STRONG", "SOLID", "ROUGH", "FODDER"]


class TeamValidationError(ValueError):
    """The submitted squad is illegal; ``problems`` lists EVERY violated rule."""

    def __init__(self, problems: list[str]) -> None:
        self.problems = problems
        super().__init__("; ".join(problems))


# --------------------------------------------------------------------------------------
# Response models (mirror the POST /api/rate-team contract field-for-field)
# --------------------------------------------------------------------------------------


class PlayerRating(BaseModel):
    """One squad member's contribution over the rating window."""

    player_code: int
    web_name: str
    position: str
    team_short: str | None = None
    price: int = Field(description="Current price, £0.1m units")
    xp_gw1: float = Field(description="xP at the window's first GW (0 when no row)")
    xp_horizon: float = Field(description="Summed xP over the window's GWs")
    q0: float | None = Field(
        default=None, description="P(0 minutes) from the player's first window row"
    )
    in_best_xi_gw1: bool


class WeakestPlayer(BaseModel):
    """One of the (up to 3) lowest-xP squad members over the window."""

    player_code: int
    web_name: str
    xp_horizon: float


class RatingResult(BaseModel):
    """POST /api/rate-team response: the squad's score against the benchmark anchors.

    ``team_xp_gw1`` is the first window GW's metric term (best-XI xP + captain bonus);
    ``team_xp_horizon`` is the full-window metric, so the per-GW terms sum to it.
    """

    score: float = Field(ge=0.0, le=100.0)
    verdict: Verdict
    season: int
    from_gw: int
    horizon: int = Field(description="Number of GWs in the rating window")
    team_xp_gw1: float
    team_xp_horizon: float
    optimal_xp_horizon: float = Field(description="metric(dream-team squad), same metric")
    floor_xp_horizon: float = Field(description="metric(cheapest legal 15), same metric")
    best_xi_gw1: list[int]
    formation_gw1: str = Field(description="DEF-MID-FWD, e.g. '3-5-2'")
    suggested_captain: int = Field(description="Highest-xP member of the first GW's XI")
    player_ratings: list[PlayerRating]
    weakest: list[WeakestPlayer]


class XiSelection(NamedTuple):
    """A greedy best-XI pick for one GW."""

    xi: list[int]
    formation: str
    xi_xp: float
    captain: int
    captain_xp: float


# --------------------------------------------------------------------------------------
# Squad legality
# --------------------------------------------------------------------------------------


def validate_squad(player_codes: Sequence[int], roster: pd.DataFrame) -> list[str]:
    """Every squad-legality rule the submitted codes violate (empty = legal).

    Rules per FPL_KNOWLEDGE §1.3: exactly 15 distinct players, quotas 2 GKP/5 DEF/
    5 MID/3 FWD, at most 3 per club, total price within the £100m budget, and every
    code present in the live roster.  All checks run — nothing short-circuits — so
    the caller can report every violation at once.
    """
    problems: list[str] = []
    codes = [int(c) for c in player_codes]
    if len(codes) != SQUAD_SIZE:
        problems.append(f"need exactly {SQUAD_SIZE} players, got {len(codes)}")
    for code in sorted(c for c, n in Counter(codes).items() if n > 1):
        problems.append(f"duplicate player_code {code}")

    info: dict[int, tuple[str, int, int]] = {
        int(r.player_code): (str(r.position), int(r.team_code), int(r.price))
        for r in roster.itertuples(index=False)
    }
    known = [c for c in codes if c in info]
    for code in sorted(set(codes) - set(known)):
        problems.append(f"unknown player_code {code}")

    counts = Counter(info[c][0] for c in known)
    for pos in rules.POSITIONS:
        if counts.get(pos, 0) != POSITION_QUOTA[pos]:
            problems.append(
                f"position quota: need {POSITION_QUOTA[pos]} {pos}, got {counts.get(pos, 0)}"
            )
    for club, n in sorted(Counter(info[c][1] for c in known).items()):
        if n > CLUB_LIMIT:
            problems.append(
                f"club limit: max {CLUB_LIMIT} players per club, got {n} from team_code {club}"
            )
    total_price = sum(info[c][2] for c in known)
    if total_price > INITIAL_BUDGET:
        problems.append(f"total price {total_price} exceeds budget {INITIAL_BUDGET}")
    return problems


# --------------------------------------------------------------------------------------
# The metric: greedy best XI + captain bonus, summed over the window
# --------------------------------------------------------------------------------------


def best_xi(entries: Iterable[tuple[int, str, float]]) -> XiSelection:
    """Greedy best legal XI from ``(player_code, position, xp)`` entries.

    Formation minimums (1 GKP/3 DEF/2 MID/1 FWD) by xP first, then filled to 11 by
    xP respecting the maxes (1 GKP/5 DEF/5 MID/3 FWD).  Every tie breaks by
    ``player_code`` for determinism; the captain is the XI's highest-xP player.

    Raises:
        ValueError: when the entries cannot field a formation-legal XI.
    """
    by_pos: dict[str, list[tuple[int, float]]] = {pos: [] for pos in rules.POSITIONS}
    for code, pos, xp in entries:
        if pos not in by_pos:
            raise ValueError(f"unknown position {pos!r} for player {code}")
        by_pos[pos].append((int(code), float(xp)))
    for group in by_pos.values():
        group.sort(key=lambda e: (-e[1], e[0]))

    chosen: list[tuple[int, str, float]] = []
    counts: dict[str, int] = {}
    for pos in rules.POSITIONS:
        need = FORMATION_MIN[pos]
        if len(by_pos[pos]) < need:
            raise ValueError(f"cannot field a legal XI: need {need} {pos}, have {len(by_pos[pos])}")
        chosen.extend((code, pos, xp) for code, xp in by_pos[pos][:need])
        counts[pos] = need
    rest = sorted(
        (
            (code, pos, xp)
            for pos in rules.POSITIONS
            for code, xp in by_pos[pos][FORMATION_MIN[pos] :]
        ),
        key=lambda e: (-e[2], e[0]),
    )
    for code, pos, xp in rest:
        if len(chosen) == XI_SIZE:
            break
        if counts[pos] >= FORMATION_MAX[pos]:
            continue
        chosen.append((code, pos, xp))
        counts[pos] += 1
    if len(chosen) < XI_SIZE:
        raise ValueError(f"cannot field a legal XI: only {len(chosen)} eligible players")

    captain, captain_xp = max(((c, xp) for c, _, xp in chosen), key=lambda e: (e[1], -e[0]))
    order = {pos: i for i, pos in enumerate(rules.POSITIONS)}
    xi = [c for c, _, _ in sorted(chosen, key=lambda e: (order[e[1]], -e[2], e[0]))]
    return XiSelection(
        xi=xi,
        formation=f"{counts['DEF']}-{counts['MID']}-{counts['FWD']}",
        xi_xp=sum(xp for _, _, xp in chosen),
        captain=captain,
        captain_xp=captain_xp,
    )


def squad_metric(
    codes: Sequence[int],
    pos_of: Mapping[int, str],
    xp_of: Mapping[tuple[int, int], float],
    window: Iterable[int],
) -> float:
    """``sum over window GWs of (best-XI xP + captain bonus)`` for one 15-man squad.

    ``xp_of`` maps ``(gw, player_code) -> xp``; players without a row contribute 0.
    """
    total = 0.0
    for gw in window:
        sel = best_xi((c, pos_of[c], xp_of.get((gw, c), 0.0)) for c in codes)
        total += sel.xi_xp + sel.captain_xp
    return total


def cheapest_legal_squad(roster: pd.DataFrame) -> list[int]:
    """The floor squad: cheapest quota-satisfying, club-legal 15 from the roster.

    Greedy per position (cheapest first, ties by ``player_code``), skipping players
    whose club is already at the limit — the same warm-start greedy the MILP uses.

    Raises:
        ValueError: when the roster cannot field a legal 15.
    """
    chosen: list[int] = []
    club_n: Counter[int] = Counter()
    for pos, quota in POSITION_QUOTA.items():
        group = roster[roster["position"] == pos].sort_values(
            ["price", "player_code"], kind="stable"
        )
        taken = 0
        for row in group.itertuples(index=False):
            if club_n[int(row.team_code)] >= CLUB_LIMIT:
                continue
            chosen.append(int(row.player_code))
            club_n[int(row.team_code)] += 1
            taken += 1
            if taken == quota:
                break
        if taken < quota:
            raise ValueError(f"roster cannot field a legal 15: only {taken} {pos} available")
    return chosen


# --------------------------------------------------------------------------------------
# rate_team
# --------------------------------------------------------------------------------------


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _verdict(score: float) -> Verdict:
    for threshold, verdict in VERDICT_BANDS:
        if score >= threshold:
            return verdict  # type: ignore[return-value]
    return "FODDER"


def rate_team(
    player_codes: Sequence[int],
    predictions_gw: pd.DataFrame,
    roster: pd.DataFrame,
    dream_squad: list[int],
    *,
    season: int,
    from_gw: int,
    horizon: int,
) -> RatingResult:
    """Score a 15-man squad 0-100 between the floor and dream-team anchors.

    Pure and deterministic — see the module docstring for the metric and the
    clamp-to-100 rationale.  ``roster`` is the legality pool (an optional
    ``team_short`` column feeds the per-player ratings); ``predictions_gw`` needs
    ``season, gw, player_code, xp`` (+ ``q0`` where available).

    Raises:
        TeamValidationError: with EVERY violated legality rule (API turns this
            into the contract's 422 ``{"detail": [...]}``).
        ValueError: when the dream squad or roster cannot be evaluated at all.
    """
    problems = validate_squad(player_codes, roster)
    if problems:
        raise TeamValidationError(problems)
    codes = [int(c) for c in player_codes]
    window = range(from_gw, from_gw + horizon)

    preds = predictions_gw[
        (predictions_gw["season"] == season) & predictions_gw["gw"].isin(window)
    ]
    xp_of: dict[tuple[int, int], float] = {
        (int(gw), int(code)): float(xp)
        for gw, code, xp in zip(preds["gw"], preds["player_code"], preds["xp"], strict=True)
    }
    q0_of: dict[int, float] = {}
    if "q0" in preds.columns:
        first_rows = preds.sort_values(["player_code", "gw"], kind="stable")
        for row in first_rows.itertuples(index=False):
            code = int(row.player_code)
            if code not in q0_of and pd.notna(row.q0):
                q0_of[code] = float(row.q0)

    pos_of: dict[int, str] = {
        int(r.player_code): str(r.position) for r in roster.itertuples(index=False)
    }
    missing = [c for c in dream_squad if c not in pos_of]
    if missing and "position" in preds.columns:
        for row in preds[preds["player_code"].isin(missing)].itertuples(index=False):
            pos_of.setdefault(int(row.player_code), str(row.position))
        missing = [c for c in dream_squad if c not in pos_of]
    if missing:
        raise ValueError(f"dream squad players unknown to roster/predictions: {missing}")

    team_metric = squad_metric(codes, pos_of, xp_of, window)
    optimal_metric = squad_metric([int(c) for c in dream_squad], pos_of, xp_of, window)
    floor_metric = squad_metric(cheapest_legal_squad(roster), pos_of, xp_of, window)

    denom = optimal_metric - floor_metric
    if denom > 0:
        score = _clamp01((team_metric - floor_metric) / denom) * 100.0
    else:  # degenerate anchors (e.g. empty window): fall back to a pass/fail split
        score = 100.0 if team_metric >= optimal_metric else 0.0

    sel_gw1 = best_xi((c, pos_of[c], xp_of.get((from_gw, c), 0.0)) for c in codes)
    xi_gw1 = set(sel_gw1.xi)

    xp_horizon_of = {c: sum(xp_of.get((gw, c), 0.0) for gw in window) for c in codes}
    detail: dict[int, tuple[str, str | None, int]] = {}
    for row in roster.itertuples(index=False):
        code = int(row.player_code)
        if code in xp_horizon_of:
            short = getattr(row, "team_short", None)
            detail[code] = (
                str(row.web_name),
                str(short) if pd.notna(short) else None,
                int(row.price),
            )
    ratings = [
        PlayerRating(
            player_code=c,
            web_name=detail[c][0],
            position=pos_of[c],
            team_short=detail[c][1],
            price=detail[c][2],
            xp_gw1=xp_of.get((from_gw, c), 0.0),
            xp_horizon=xp_horizon_of[c],
            q0=q0_of.get(c),
            in_best_xi_gw1=c in xi_gw1,
        )
        for c in codes
    ]
    weakest = [
        WeakestPlayer(player_code=c, web_name=detail[c][0], xp_horizon=xp_horizon_of[c])
        for c in sorted(codes, key=lambda c: (xp_horizon_of[c], c))[:3]
    ]

    return RatingResult(
        score=score,
        verdict=_verdict(score),
        season=season,
        from_gw=from_gw,
        horizon=horizon,
        team_xp_gw1=sel_gw1.xi_xp + sel_gw1.captain_xp,
        team_xp_horizon=team_metric,
        optimal_xp_horizon=optimal_metric,
        floor_xp_horizon=floor_metric,
        best_xi_gw1=sel_gw1.xi,
        formation_gw1=sel_gw1.formation,
        suggested_captain=sel_gw1.captain,
        player_ratings=ratings,
        weakest=weakest,
    )
