"""Weekly-verdict orchestrator: distill MILP solver outputs into one Recommendation.

Implements the ``optimizer/plans.py`` stage-3 contract from ``docs/ARCHITECTURE.md``:
:func:`build_recommendation` runs (a) the user-state multi-GW MILP plan
(``fplai.optimizer.milp.solve_plan``), (b) a None-state "dream team" benchmark solve for
the same GW, (c) chip EV curves (``fplai.optimizer.chips.chip_ev_curves``) and (d)
plan-stability noise re-solves (``fplai.optimizer.sensitivity.plan_stability``), and
distills them into a single :class:`Recommendation` — the weekly verdict the API layer
returns verbatim. Rationale bullets are template-generated from the numbers (xP deltas,
fixture runs, chip EV windows, stability support, ``q0`` availability flags) — no LLM.

Input contracts (ARCHITECTURE "Optimizer contracts"): ``xp`` is the ``predictions_gw``
frame (``season, gw, player_code, xp`` plus a per-player ``q0`` column = P(0 minutes)
used for vice weighting and availability flags); ``prices`` maps ``player_code`` to the
current ``price`` (0.1m units), ``position`` and ``team_code``. An optional ``web_name``
column in ``prices`` makes rationale bullets human-readable.

The heavy lifting lives in sibling modules that are built in parallel, so they are
imported lazily through thin proxy functions (:func:`solve_plan`,
:func:`chip_ev_curves`, :func:`plan_stability`). Tests monkeypatch these seams with
deterministic fakes; the proxies resolve the real implementations at call time.

Timestamps are always passed in via ``as_of`` — this module never calls
``datetime.now()`` — and every pydantic model here serializes cleanly to JSON via
``model_dump(mode="json")`` / ``model_dump_json()``.
"""

from __future__ import annotations

import dataclasses
import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel

from fplai import rules

if TYPE_CHECKING:  # pragma: no cover — sibling module is built in parallel
    from fplai.optimizer.state import SquadState

__all__ = [
    "CHIP_NAMES",
    "CONFIDENCE_BANDS",
    "ChipAdvice",
    "DreamTeam",
    "Recommendation",
    "StabilityEntry",
    "TransferPair",
    "apply_sim_to_recommendation",
    "build_chip_advice_from_sim",
    "build_recommendation",
    "chip_ev_curves",
    "confidence_from_p_beats_hold",
    "dream_team",
    "merge_chip_advice",
    "plan_stability",
    "solve_plan",
]

# --------------------------------------------------------------------------------------
# Lazy proxies to the sibling optimizer modules (monkeypatch seams for tests)
# --------------------------------------------------------------------------------------


def solve_plan(*args: Any, **kwargs: Any) -> Any:
    """Proxy to :func:`fplai.optimizer.milp.solve_plan` (lazy import; test seam)."""
    from fplai.optimizer.milp import solve_plan as _solve_plan

    return _solve_plan(*args, **kwargs)


def chip_ev_curves(*args: Any, **kwargs: Any) -> pd.DataFrame:
    """Proxy to :func:`fplai.optimizer.chips.chip_ev_curves` (lazy import; test seam)."""
    from fplai.optimizer.chips import chip_ev_curves as _chip_ev_curves

    return _chip_ev_curves(*args, **kwargs)


def plan_stability(*args: Any, **kwargs: Any) -> pd.DataFrame:
    """Proxy to :func:`fplai.optimizer.sensitivity.plan_stability` (lazy import; test seam)."""
    from fplai.optimizer.sensitivity import plan_stability as _plan_stability

    return _plan_stability(*args, **kwargs)


# --------------------------------------------------------------------------------------
# Pydantic output models (the API layer returns these verbatim)
# --------------------------------------------------------------------------------------

#: Human names for chip-id prefixes (chip ids are e.g. ``"wc1"``, ``"bb2"``).
CHIP_NAMES: dict[str, str] = {
    "wc": "Wildcard",
    "fh": "Free Hit",
    "bb": "Bench Boost",
    "tc": "Triple Captain",
    "aoa": "All Out Attack",
    "am": "Assistant Manager",
}

#: q0 (P(0 minutes)) at or above which a player is flagged as an availability risk.
AVAILABILITY_Q0_FLAG: float = 0.35

#: |price_change_percent| at/above which an execution bullet flags an imminent
#: price move. The field is the OFFICIAL predictor's progress-toward-change
#: number shipped in bootstrap elements (2026-27+); observed "0" for everyone
#: while prices are frozen pre-GW1. Sign semantics are assumed positive=rise /
#: negative=fall and treated defensively — a bullet is advisory, never a solve
#: input.
PRICE_TREND_FLAG_PCT: float = 75.0

_CHIP_ID_RE = re.compile(r"^([a-z]+?)(\d+)$")


class TransferPair(BaseModel):
    """One in/out transfer pair for the current GW, with horizon xP deltas."""

    player_in: int | None = None
    player_in_name: str | None = None
    player_out: int | None = None
    player_out_name: str | None = None
    position: str | None = None
    xp_in: float = 0.0  # in-player xP summed over the horizon
    xp_out: float = 0.0  # out-player xP summed over the horizon
    xp_delta: float = 0.0  # xp_in - xp_out
    out_q0: float | None = None  # out-player P(0 minutes) this GW (availability flag)
    support_pct: float | None = None  # % of noise re-solves agreeing with this move


class ChipAdvice(BaseModel):
    """Play/hold verdict for one available chip instance, with EV context.

    The simulation block (``e_gain_now`` … ``assumptions``) is populated only after a
    Monte Carlo season simulation has run (the ``optimizer/season_sim`` stage-6
    contract — see :func:`build_chip_advice_from_sim`). Every simulation field is
    Optional and defaults to ``None``, so pre-simulation flows and artifacts on disk
    stay valid unchanged.
    """

    chip: str  # chip id, e.g. "bb1"
    verdict: Literal["play", "hold"]
    planned_gw: int | None = None  # GW the optimal plan schedules this chip (if any)
    ev_now: float | None = None  # delta_vs_no_chip if forced this GW
    best_gw: int | None = None  # best GW for this chip within the horizon
    best_ev: float | None = None  # delta_vs_no_chip at best_gw
    # --- season-simulation fields (None until `fplai simulate` has run) ------------
    e_gain_now: float | None = None  # E[gain] if played this GW (simulation mean)
    sd: float | None = None  # sd of that gain across rollouts
    p_best_week_now: float | None = None  # P(this GW is the chip's best week)
    p_beats_hold: float | None = None  # P(playing now beats holding the chip)
    recommended_gw: int | None = None  # simulation-recommended GW to play the chip
    p_best_week_reco: float | None = None  # P(recommended_gw is the best week)
    confidence: str | None = None  # "high" | "medium" | "low" (p_beats_hold bands)
    n_rollouts: int | None = None  # simulated seasons behind these numbers
    assumptions: list[str] | None = None  # simulation assumptions, human-readable


class DreamTeam(BaseModel):
    """The fresh-£100m benchmark squad from a None-state solve for one GW."""

    season: int
    gw: int
    squad: list[int]
    lineup: list[int]
    bench_order: list[int]
    captain: int | None = None
    vice: int | None = None
    formation: str | None = None  # e.g. "3-4-3" (DEF-MID-FWD)
    expected_points: float = 0.0  # this GW's expected points
    objective: float = 0.0  # full-horizon solve objective
    total_cost: int = 0  # squad cost in 0.1m units
    solve_seconds: float | None = None


class StabilityEntry(BaseModel):
    """One this-GW move with its support across noise re-solves (0-100 scale)."""

    move: str
    support_pct: float


class Recommendation(BaseModel):
    """The weekly verdict: action, transfers, chip advice, benchmark and rationale.

    ``action`` is ``"hold"`` (optimal plan makes 0 transfers this GW), ``"transfer"``,
    ``"chip:<id>"`` (plan plays a chip this GW) or ``"initial-squad"`` (built with
    ``state=None`` — the pre-GW1 product). ``plan`` carries the full
    ``PlanResult`` as a JSON-safe dict; ``dream_team`` is the fresh-£100m benchmark.
    """

    season: int
    gw: int
    as_of: datetime
    action: str
    transfers: list[TransferPair] = []
    hits: int = 0
    captain: int | None = None
    vice: int | None = None
    lineup: list[int] = []
    bench_order: list[int] = []
    squad: list[int] = []
    formation: str | None = None
    expected_points: float = 0.0  # this GW, from the optimal plan
    objective: float = 0.0  # full-horizon plan objective
    chip_advice: list[ChipAdvice] = []
    dream_team: DreamTeam | None = None
    plan: dict[str, Any] = {}
    stability: list[StabilityEntry] = []
    rationale: list[str] = []


# --------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------


def _get(obj: Any, name: str, default: Any = None) -> Any:
    """Attribute-or-mapping access, tolerant of pydantic models, dataclasses and dicts."""
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _to_jsonable(obj: Any) -> Any:
    """Recursively convert ``obj`` to JSON-safe python (dicts/lists/str/int/float/None)."""
    if isinstance(obj, BaseModel):
        return _to_jsonable(obj.model_dump(mode="json"))
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return _to_jsonable(dataclasses.asdict(obj))
    if isinstance(obj, Mapping):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return _to_jsonable(float(obj))
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, float):
        return None if math.isnan(obj) or math.isinf(obj) else obj
    if obj is None or isinstance(obj, (str, int, bool)):
        return obj
    return str(obj)


def _plan_dump(plan: Any) -> dict[str, Any]:
    """Full ``PlanResult`` as a JSON-safe dict (whatever concrete type the solver used)."""
    dumped = _to_jsonable(plan)
    if isinstance(dumped, dict):
        return dumped
    return {"plan": dumped}


def _xp_window(xp: pd.DataFrame, season: int, gw: int, horizon: int) -> pd.DataFrame:
    """Rows of ``xp`` for ``season`` and GWs in ``[gw, gw + horizon)``."""
    mask = (xp["season"] == season) & (xp["gw"] >= gw) & (xp["gw"] < gw + horizon)
    return xp.loc[mask]


def _sum_xp(window: pd.DataFrame, code: int) -> float:
    """Total xP for ``code`` over the window (0.0 if absent)."""
    vals = window.loc[window["player_code"] == code, "xp"]
    return float(vals.sum()) if len(vals) else 0.0


def _gw_xp(window: pd.DataFrame, code: int, gw: int) -> float:
    """xP for ``code`` in a single GW (0.0 if absent)."""
    vals = window.loc[(window["player_code"] == code) & (window["gw"] == gw), "xp"]
    return float(vals.sum()) if len(vals) else 0.0


def _next3_avg(window: pd.DataFrame, code: int, gw: int) -> float | None:
    """Mean xP/GW for ``code`` over GWs ``gw .. gw+2`` (None if absent)."""
    vals = window.loc[(window["player_code"] == code) & (window["gw"] < gw + 3), "xp"]
    return float(vals.mean()) if len(vals) else None


def _q0(xp: pd.DataFrame, season: int, gw: int, code: int) -> float | None:
    """P(0 minutes) for ``code`` this GW from the xp frame's ``q0`` column (if present)."""
    if "q0" not in xp.columns:
        return None
    mask = (xp["season"] == season) & (xp["gw"] == gw) & (xp["player_code"] == code)
    vals = xp.loc[mask, "q0"].dropna()
    if not len(vals):
        return None
    val = float(vals.mean())
    return None if math.isnan(val) else val


def _name_map(prices: pd.DataFrame) -> dict[int, str]:
    if "web_name" not in prices.columns:
        return {}
    return {
        int(code): str(name)
        for code, name in zip(prices["player_code"], prices["web_name"], strict=True)
    }


def _pos_map(prices: pd.DataFrame) -> dict[int, str]:
    if "position" not in prices.columns:
        return {}
    return {
        int(code): str(pos)
        for code, pos in zip(prices["player_code"], prices["position"], strict=True)
    }


def _price_map(prices: pd.DataFrame) -> dict[int, int]:
    if "price" not in prices.columns:
        return {}
    return {
        int(code): int(price)
        for code, price in zip(prices["player_code"], prices["price"], strict=True)
    }


def _name(code: int | None, names: Mapping[int, str]) -> str:
    if code is None:
        return "(none)"
    return names.get(code, f"player {code}")


def _formation(lineup: Sequence[int], positions: Mapping[int, str]) -> str | None:
    """DEF-MID-FWD formation string for the XI, or None when positions are unknown."""
    if not lineup or any(code not in positions for code in lineup):
        return None
    counts = Counter(positions[code] for code in lineup)
    formation = f"{counts.get('DEF', 0)}-{counts.get('MID', 0)}-{counts.get('FWD', 0)}"
    if len(lineup) == 15:  # Bench Boost week: all 15 count, an XI formation would mislead
        return f"BB · {formation}"
    return formation


def _chip_name(chip_id: str) -> str:
    """Human name for a chip id like ``"bb1"`` -> ``"Bench Boost"``."""
    match = _CHIP_ID_RE.match(chip_id)
    prefix = match.group(1) if match else chip_id
    return CHIP_NAMES.get(prefix, chip_id.upper())


def _chip_windows_by_id(season: int) -> dict[str, rules.ChipWindow]:
    """Map chip ids (``"wc1"`` etc.) to their :class:`~fplai.rules.ChipWindow`."""
    return {f"{w.name.lower()}{w.set}": w for w in rules.chip_windows(season)}


def _normalize_stability(stability: pd.DataFrame | None) -> list[StabilityEntry]:
    """DataFrame[move, support_pct] -> entries on a 0-100 scale (accepts 0-1 input)."""
    if stability is None or len(stability) == 0:
        return []
    support = stability["support_pct"].astype(float)
    if float(support.max()) <= 1.0:
        support = support * 100.0
    return [
        StabilityEntry(move=str(move), support_pct=float(pct))
        for move, pct in zip(stability["move"], support, strict=True)
    ]


def _support_for(entries: Sequence[StabilityEntry], *codes: int | None) -> float | None:
    """Best support % among moves mentioning the given player codes.

    Prefers moves mentioning ALL codes (the exact in/out pair); falls back to any move
    mentioning at least one. Codes are matched as whole tokens to avoid substring hits.
    """
    wanted = [c for c in codes if c is not None]
    if not entries or not wanted:
        return None
    patterns = [re.compile(rf"(?<!\d){c}(?!\d)") for c in wanted]
    all_hits = [e.support_pct for e in entries if all(p.search(e.move) for p in patterns)]
    if all_hits:
        return max(all_hits)
    any_hits = [e.support_pct for e in entries if any(p.search(e.move) for p in patterns)]
    return max(any_hits) if any_hits else None


def _pair_codes(
    ins: Sequence[int], outs: Sequence[int], positions: Mapping[int, str]
) -> list[tuple[int | None, int | None]]:
    """Pair transfer-in codes with transfer-out codes, matching by position first."""
    ins_left = [int(c) for c in ins]
    outs_left = [int(c) for c in outs]
    pairs: list[tuple[int | None, int | None]] = []
    for code_in in list(ins_left):
        match = next(
            (o for o in outs_left if positions.get(o) == positions.get(code_in)), None
        )
        if match is not None:
            pairs.append((code_in, match))
            ins_left.remove(code_in)
            outs_left.remove(match)
    n_common = min(len(ins_left), len(outs_left))
    pairs.extend(zip(ins_left[:n_common], outs_left[:n_common], strict=True))
    pairs.extend((c, None) for c in ins_left[n_common:])
    pairs.extend((None, c) for c in outs_left[n_common:])
    return pairs


def _call_solve(
    xp: pd.DataFrame,
    prices: pd.DataFrame,
    state: SquadState | None,
    horizon: int,
    params: Any | None,
) -> Any:
    """Invoke the (proxied) solver, only passing ``params`` when the caller supplied it."""
    kwargs: dict[str, Any] = {"horizon": horizon}
    if params is not None:
        kwargs["params"] = params
    return solve_plan(xp, prices, state, **kwargs)


def _plan_gw(plan: Any, gw: int) -> Any:
    """The GwPlan for ``gw`` (falls back to the first GW of the plan)."""
    gws = _get(plan, "gws") or []
    if not gws:
        raise ValueError("solve_plan returned a plan with no gameweeks")
    return next((g for g in gws if int(_get(g, "gw", -1)) == gw), gws[0])


# --------------------------------------------------------------------------------------
# Dream team (standalone None-state benchmark)
# --------------------------------------------------------------------------------------


def dream_team(
    xp: pd.DataFrame,
    prices: pd.DataFrame,
    gw: int,
    season: int,
    *,
    horizon: int = 1,
    params: Any | None = None,
) -> DreamTeam:
    """Solve the fresh-£100m "perfect squad" benchmark for one GW (None-state solve).

    Runs the MILP with ``state=None`` (GW ``gw``, £100.0m, unlimited transfers) over
    ``horizon`` GWs of ``xp`` starting at ``gw`` and returns the first GW's squad, XI,
    captaincy and expected points — the benchmark squad the UI shows every GW.
    ``horizon=1`` is the pure single-GW dream team; ``horizon>1`` makes the pick
    horizon-aware (the ``objective`` field then covers the full horizon).
    """
    window = _xp_window(xp, season, gw, horizon)
    if len(window) == 0:
        raise ValueError(f"no xp rows for season={season} gw={gw} (horizon={horizon})")
    plan = _call_solve(window, prices, None, horizon, params)
    plan_first = _plan_gw(plan, gw)
    positions = _pos_map(prices)
    price_of = _price_map(prices)
    squad = [int(c) for c in _get(plan_first, "squad", []) or []]
    lineup = [int(c) for c in _get(plan_first, "lineup", []) or []]
    captain = _get(plan_first, "captain")
    vice = _get(plan_first, "vice")
    return DreamTeam(
        season=season,
        gw=int(_get(plan_first, "gw", gw)),
        squad=squad,
        lineup=lineup,
        bench_order=[int(c) for c in _get(plan_first, "bench_order", []) or []],
        captain=int(captain) if captain is not None else None,
        vice=int(vice) if vice is not None else None,
        formation=_formation(lineup, positions),
        expected_points=float(_get(plan_first, "expected_points", 0.0) or 0.0),
        objective=float(_get(plan, "objective", 0.0) or 0.0),
        total_cost=sum(price_of.get(c, 0) for c in squad),
        solve_seconds=_get(plan, "solve_seconds"),
    )


def _dream_from_plan(
    plan: Any, plan_first: Any, season: int, gw: int, prices: pd.DataFrame
) -> DreamTeam:
    """Build a :class:`DreamTeam` from an already-solved None-state plan (no re-solve)."""
    positions = _pos_map(prices)
    price_of = _price_map(prices)
    squad = [int(c) for c in _get(plan_first, "squad", []) or []]
    lineup = [int(c) for c in _get(plan_first, "lineup", []) or []]
    captain = _get(plan_first, "captain")
    vice = _get(plan_first, "vice")
    return DreamTeam(
        season=season,
        gw=gw,
        squad=squad,
        lineup=lineup,
        bench_order=[int(c) for c in _get(plan_first, "bench_order", []) or []],
        captain=int(captain) if captain is not None else None,
        vice=int(vice) if vice is not None else None,
        formation=_formation(lineup, positions),
        expected_points=float(_get(plan_first, "expected_points", 0.0) or 0.0),
        objective=float(_get(plan, "objective", 0.0) or 0.0),
        total_cost=sum(price_of.get(c, 0) for c in squad),
        solve_seconds=_get(plan, "solve_seconds"),
    )


# --------------------------------------------------------------------------------------
# Chip advice
# --------------------------------------------------------------------------------------


def _chip_advice(
    chip_ids: Sequence[str],
    curves: pd.DataFrame | None,
    plan: Any,
    gw: int,
) -> list[ChipAdvice]:
    """Distill the chip EV curves + the plan's chip schedule into play/hold advice."""
    planned: dict[str, int] = {}
    for gwp in _get(plan, "gws") or []:
        chip = _get(gwp, "chip")
        if chip is not None and chip not in planned:
            planned[str(chip)] = int(_get(gwp, "gw", gw))
    chip_now = next((c for c, g in planned.items() if g == gw), None)

    advice: list[ChipAdvice] = []
    for cid in chip_ids:
        ev_now: float | None = None
        best_gw: int | None = None
        best_ev: float | None = None
        if curves is not None and len(curves):
            sub = curves.loc[curves["chip"] == cid]
            if len(sub):
                now_rows = sub.loc[sub["gw"] == gw, "delta_vs_no_chip"].dropna()
                if len(now_rows):
                    ev_now = float(now_rows.max())
                deltas = sub["delta_vs_no_chip"].astype(float)
                if deltas.notna().any():
                    best_idx = deltas.idxmax()
                    best_gw = int(sub.loc[best_idx, "gw"])
                    best_ev = float(deltas.loc[best_idx])
        advice.append(
            ChipAdvice(
                chip=cid,
                verdict="play" if cid == chip_now else "hold",
                planned_gw=planned.get(cid),
                ev_now=ev_now,
                best_gw=best_gw,
                best_ev=best_ev,
            )
        )
    return advice


# --------------------------------------------------------------------------------------
# Chip advice v2: Monte Carlo season-simulation overlay (optimizer/season_sim contract)
# --------------------------------------------------------------------------------------

#: Confidence bands on |p_beats_hold - 0.5| (distance from a coin flip):
#: >= 0.20 -> "high", >= 0.10 -> "medium", else "low".
CONFIDENCE_BANDS: tuple[tuple[float, str], ...] = ((0.20, "high"), (0.10, "medium"))

#: Gain-column aliases accepted in a ChipSimReport frame, in preference order.
_SIM_GAIN_COLUMNS: tuple[str, ...] = ("delta_vs_no_chip", "e_gain", "gain")

#: ChipAdvice fields owned by the simulation (copied wholesale on merge).
_SIM_ADVICE_FIELDS: tuple[str, ...] = (
    "e_gain_now",
    "sd",
    "p_best_week_now",
    "p_beats_hold",
    "recommended_gw",
    "p_best_week_reco",
    "confidence",
    "n_rollouts",
    "assumptions",
)


def confidence_from_p_beats_hold(p_beats_hold: float) -> str:
    """Confidence wording for a play/hold verdict from P(playing now beats holding).

    The verdict is decisive when the probability sits far from 0.5 in either
    direction (a confident *hold* is a low probability): the :data:`CONFIDENCE_BANDS`
    thresholds on ``|p - 0.5|`` give ``"high"`` / ``"medium"`` / ``"low"``.
    """
    margin = abs(p_beats_hold - 0.5)
    for threshold, label in CONFIDENCE_BANDS:
        if margin >= threshold:
            return label
    return "low"


def _sim_frame(report: Any) -> pd.DataFrame:
    """The per-(chip, gw) frame behind a ChipSimReport-shaped object (duck-typed)."""
    frame = report
    to_frame = getattr(report, "to_frame", None)
    if callable(to_frame):
        frame = to_frame()
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(
            "chip-sim report must be a DataFrame or expose to_frame() -> DataFrame, "
            f"got {type(report).__name__}"
        )
    missing = [col for col in ("chip", "gw") if col not in frame.columns]
    if missing:
        raise ValueError(f"chip-sim frame missing required column(s) {missing}")
    return frame


def build_chip_advice_from_sim(report: Any) -> list[ChipAdvice]:
    """Distill a season-simulation report into per-chip :class:`ChipAdvice` (v2).

    Duck-typed against the ``optimizer/season_sim`` stage-6 contract
    (``ChipSimReport``): ``report`` either *is* the chip-curves v2 frame or exposes
    ``.to_frame()`` returning one — columns ``chip``, ``gw``, a gain column
    (``delta_vs_no_chip`` preferred; ``e_gain``/``gain`` accepted) and the nullable
    v2 additions ``sd``, ``p_best_week``, ``p_beats_hold``, ``n_rollouts``.
    Report-level attributes are used when present: ``current_gw`` /
    ``window_first_gw`` (defaults to the frame's min GW), ``n_rollouts``,
    ``assumptions`` and per-chip ``verdicts`` entries (objects with ``chip``,
    ``recommended_gw`` and a float ``confidence`` = P(best week) at the recommended
    GW); per-row ``recommended_gw`` / ``confidence`` columns are honored too.

    Per chip: ``verdict`` is ``"play"`` iff P(playing now beats holding) > 0.5;
    ``recommended_gw`` comes from the report's verdicts when given, else the argmax
    of ``p_best_week`` (falling back to the gain curve; earliest GW wins ties);
    ``confidence`` (the string band) comes from :func:`confidence_from_p_beats_hold`.
    ``ev_now``/``best_gw``/``best_ev`` are filled from the gain curve so pre-v2
    consumers keep working.
    """
    frame = _sim_frame(report)
    if len(frame) == 0:
        return []
    gain_col = next((c for c in _SIM_GAIN_COLUMNS if c in frame.columns), None)

    now_raw = _get(report, "current_gw")
    if now_raw is None:
        now_raw = _get(report, "window_first_gw")
    now = int(now_raw) if now_raw is not None else int(frame["gw"].min())
    report_rollouts = _get(report, "n_rollouts")
    assumptions_raw = _get(report, "assumptions")
    assumptions = [str(a) for a in assumptions_raw] if assumptions_raw else None
    verdict_by_chip: dict[str, Any] = {}
    for entry in _get(report, "verdicts") or []:
        chip_id = _get(entry, "chip")
        if chip_id is not None:
            verdict_by_chip[str(chip_id)] = entry

    advice: list[ChipAdvice] = []
    for cid in dict.fromkeys(str(c) for c in frame["chip"]):  # first-seen chip order
        sub = frame.loc[frame["chip"].astype(str) == cid].sort_values("gw", kind="stable")

        def _at(col: str, gw: int | None, sub: pd.DataFrame = sub) -> float | None:
            if gw is None or col not in sub.columns:
                return None
            vals = sub.loc[sub["gw"] == gw, col].dropna()
            return float(vals.iloc[0]) if len(vals) else None

        def _argmax_gw(col: str | None, sub: pd.DataFrame = sub) -> int | None:
            if col is None or col not in sub.columns:
                return None
            vals = sub[col].astype(float)
            if not vals.notna().any():
                return None
            return int(sub.loc[vals.idxmax(), "gw"])  # stable sort -> earliest tie wins

        best_gw = _argmax_gw(gain_col)
        best_ev = _at(gain_col, best_gw) if gain_col is not None else None

        report_verdict = verdict_by_chip.get(cid)
        reco_raw = (
            _get(report_verdict, "recommended_gw") if report_verdict is not None else None
        )
        reco_col = sub["recommended_gw"].dropna() if "recommended_gw" in sub.columns else None
        if reco_raw is not None:
            recommended_gw: int | None = int(reco_raw)
        elif reco_col is not None and len(reco_col):
            recommended_gw = int(reco_col.iloc[0])
        else:
            recommended_gw = _argmax_gw("p_best_week")
            if recommended_gw is None:
                recommended_gw = best_gw

        p_now = _at("p_beats_hold", now)
        conf_col = sub["confidence"].dropna() if "confidence" in sub.columns else None
        if conf_col is not None and len(conf_col):
            confidence: str | None = str(conf_col.iloc[0])
        else:
            confidence = confidence_from_p_beats_hold(p_now) if p_now is not None else None

        rollouts_val = _at("n_rollouts", now)
        if rollouts_val is None and "n_rollouts" in sub.columns:
            col_vals = sub["n_rollouts"].dropna()
            rollouts_val = float(col_vals.iloc[0]) if len(col_vals) else None
        if rollouts_val is None and report_rollouts is not None:
            rollouts_val = float(report_rollouts)

        p_best_reco = _at("p_best_week", recommended_gw)
        if p_best_reco is None and report_verdict is not None:
            # season_sim's ChipVerdict.confidence IS p_best_week at the recommended GW
            conf_val = _get(report_verdict, "confidence")
            if isinstance(conf_val, (int, float)):
                p_best_reco = float(conf_val)

        e_gain_now = _at(gain_col, now) if gain_col is not None else None
        advice.append(
            ChipAdvice(
                chip=cid,
                verdict="play" if p_now is not None and p_now > 0.5 else "hold",
                planned_gw=None,  # the MILP plan's schedule is merged in separately
                ev_now=e_gain_now,
                best_gw=best_gw,
                best_ev=best_ev,
                e_gain_now=e_gain_now,
                sd=_at("sd", now),
                p_best_week_now=_at("p_best_week", now),
                p_beats_hold=p_now,
                recommended_gw=recommended_gw,
                p_best_week_reco=p_best_reco,
                confidence=confidence,
                n_rollouts=int(rollouts_val) if rollouts_val is not None else None,
                assumptions=assumptions,
            )
        )
    return advice


def merge_chip_advice(
    base: Sequence[ChipAdvice], sim: Sequence[ChipAdvice]
) -> list[ChipAdvice]:
    """Overlay simulation-derived advice onto plan-derived advice, chip by chip.

    ``base`` (from :func:`_chip_advice` — the squad state's chip inventory) fixes
    membership and order. For chips also present in ``sim`` the simulation fields
    are copied over and the simulation *verdict* wins — it prices opportunity cost
    over the full chip window, which the greedy within-horizon MILP cannot.
    ``planned_gw`` and the forced-resolve EV numbers are kept from ``base`` unless
    the simulation supplies its own.
    """
    sim_by_chip = {a.chip: a for a in sim}
    merged: list[ChipAdvice] = []
    for entry in base:
        sim_entry = sim_by_chip.get(entry.chip)
        if sim_entry is None:
            merged.append(entry)
            continue
        update: dict[str, Any] = {
            name: getattr(sim_entry, name)
            for name in _SIM_ADVICE_FIELDS
            if getattr(sim_entry, name) is not None
        }
        update["verdict"] = sim_entry.verdict
        if sim_entry.ev_now is not None:
            update["ev_now"] = sim_entry.ev_now
        if sim_entry.best_gw is not None:
            update["best_gw"] = sim_entry.best_gw
            update["best_ev"] = sim_entry.best_ev
        merged.append(entry.model_copy(update=update))
    return merged


def apply_sim_to_recommendation(rec: Recommendation, chip_sim: Any) -> Recommendation:
    """Fold season-simulation verdicts into an already-built :class:`Recommendation`.

    The post-hoc counterpart of ``build_recommendation(..., chip_sim=)`` for the
    pipeline's ``fplai simulate`` stage, which runs *after* ``optimize`` has written
    ``recommendation.json``: chip advice is merged per :func:`merge_chip_advice` (the
    simulation verdict wins) and the rationale's chip bullets are replaced by the
    probability-speaking simulation bullets. Everything else (plan, transfers,
    captaincy, stability) is untouched — the simulation re-verdicts chips only.
    """
    sim_advice = build_chip_advice_from_sim(chip_sim)
    if not sim_advice:
        return rec
    merged = merge_chip_advice(rec.chip_advice, sim_advice)
    # Replace the chip-speaking bullets (the held-chip EV-window bullet and any stale
    # simulation bullets from a previous fold) with fresh probability bullets; the
    # action/transfer/captain bullets stay — same shape build_recommendation(chip_sim=)
    # produces, and idempotent across repeated simulate runs.
    rationale = [
        bullet
        for bullet in rec.rationale
        if not bullet.startswith("Best future chip window:")
        and " now beats holding in " not in bullet
    ]
    rationale.extend(_sim_chip_bullet(a) for a in merged if a.p_beats_hold is not None)
    return rec.model_copy(update={"chip_advice": merged, "rationale": rationale})


# --------------------------------------------------------------------------------------
# Rationale (template-generated from the numbers; no LLM)
# --------------------------------------------------------------------------------------


def _transfer_bullet(pair: TransferPair, window: pd.DataFrame, gw: int) -> str:
    in_avg3 = (
        _next3_avg(window, pair.player_in, gw) if pair.player_in is not None else None
    )
    in_part = pair.player_in_name or "(none)"
    if in_avg3 is not None:
        in_part += f" ({in_avg3:.1f} xP/GW next 3)"
    if pair.player_out is None:
        # Initial-squad / draft mode: there is no outgoing player to name.
        text = f"Draft {in_part}: {pair.xp_delta:+.1f} xP over the horizon"
    else:
        out_part = pair.player_out_name or "(none)"
        if pair.out_q0 is not None and pair.out_q0 >= AVAILABILITY_Q0_FLAG:
            out_part += f" (q0 {pair.out_q0 * 100:.0f}%, availability risk)"
        text = f"Bring in {in_part} for {out_part}: {pair.xp_delta:+.1f} xP over the horizon"
    if pair.support_pct is not None:
        text += f", {pair.support_pct:.0f}% of noise re-solves agree"
    return text + "."


def _sim_chip_bullet(advice: ChipAdvice) -> str:
    """Probability-speaking chip bullet when simulation data is present.

    Example: ``"Bench Boost (bb1) now beats holding in 34% of 1,000 simulated
    seasons — hold; best window GW7 (P(best)=0.28)."``
    """
    assert advice.p_beats_hold is not None  # caller filters on this
    seasons = f"{advice.n_rollouts:,}" if advice.n_rollouts else "the"
    text = (
        f"{_chip_name(advice.chip)} ({advice.chip}) now beats holding in "
        f"{advice.p_beats_hold * 100:.0f}% of {seasons} simulated seasons — "
        f"{advice.verdict}"
    )
    if advice.recommended_gw is not None:
        text += f"; best window GW{advice.recommended_gw}"
        if advice.p_best_week_reco is not None:
            text += f" (P(best)={advice.p_best_week_reco:.2f})"
    return text + "."


def _execution_bullets(
    action: str,
    pairs: Sequence[TransferPair],
    price_trends: Mapping[int, float] | None,
    next_deadline_utc: datetime | None,
) -> list[str]:
    """WHEN to execute this GW's moves — the campaign-handover guidance.

    Doctrine: information dominates price — transfers wait for the final press
    conferences (typically the day before the deadline) unless the official
    price predictor says a target is about to move tonight, in which case the
    early-lock trade-off is flagged explicitly. The bullets also anchor the
    re-check cadence (daily rebuilds + the pre-deadline run), so a manager
    following the site blindly acts on the freshest verdict, not a stale one.
    Emits nothing without a known deadline (pre-launch/backtest windows).
    """
    if next_deadline_utc is None:
        return []
    when = next_deadline_utc.strftime("%a %d %b %H:%M UTC")
    bullets: list[str] = []
    if action == "hold":
        bullets.append(
            f"EXECUTION: no transfers this GW — nothing to do before the deadline ({when}); "
            "re-check the final pre-deadline verdict (~3h out) in case late news flips it."
        )
    else:
        what = "confirm the squad" if action == "initial-squad" else "make these moves"
        bullets.append(
            f"EXECUTION: {what} AFTER the final press conferences — ideally the evening "
            f"before or on deadline day ({when}). Re-check the latest verdict first: the "
            "model rebuilds ~05:30/11:30/17:30 UTC daily and again ~3h before the deadline."
        )
    if price_trends:
        flagged = 0
        for pair in pairs:
            if flagged >= 3:
                break
            t_in = price_trends.get(pair.player_in) if pair.player_in is not None else None
            t_out = price_trends.get(pair.player_out) if pair.player_out is not None else None
            if t_in is not None and t_in >= PRICE_TREND_FLAG_PCT:
                bullets.append(
                    f"PRICE: {pair.player_in_name} is {t_in:.0f}% toward a rise (official "
                    "predictor) — buying earlier locks the price, at the cost of acting "
                    "before the final team news."
                )
                flagged += 1
            elif t_in is not None and t_in <= -PRICE_TREND_FLAG_PCT:
                bullets.append(
                    f"PRICE: {pair.player_in_name} is {abs(t_in):.0f}% toward a FALL — "
                    "waiting past tonight's change buys £0.1m cheaper."
                )
                flagged += 1
            if t_out is not None and t_out <= -PRICE_TREND_FLAG_PCT and flagged < 3:
                bullets.append(
                    f"PRICE: {pair.player_out_name} is {abs(t_out):.0f}% toward a FALL — "
                    "selling before tonight's change protects the sale price."
                )
                flagged += 1
    return bullets


def _build_rationale(
    *,
    action: str,
    state: SquadState | None,
    season: int,
    gw: int,
    horizon: int,
    pairs: Sequence[TransferPair],
    hits: int,
    captain: int | None,
    vice: int | None,
    captain_q0: float | None,
    expected_points: float,
    objective: float,
    dream: DreamTeam | None,
    advice: Sequence[ChipAdvice],
    stability_entries: Sequence[StabilityEntry],
    names: Mapping[int, str],
    window: pd.DataFrame,
    squad_cost: int,
) -> list[str]:
    """Crisp, number-driven bullets covering the WHY of the verdict."""
    bullets: list[str] = []

    if action == "initial-squad":
        bank = max(0, 1000 - squad_cost)
        bullets.append(
            f"Initial squad for GW{gw}: 15 players, £{squad_cost / 10:.1f}m spent, "
            f"£{bank / 10:.1f}m in the bank; expected {expected_points:.1f} xP in GW{gw}."
        )
    elif action == "hold":
        ft_now = int(_get(state, "free_transfers", 1) or 1)
        try:
            ft_next = rules.next_free_transfers(ft_now, 0, season, False)
        except ValueError:
            ft_next = min(ft_now + 1, 5)
        bullets.append(
            f"Hold: the optimal {horizon}-GW plan makes 0 transfers this GW — bank the "
            f"free transfer ({ft_now} → {ft_next})."
        )
    elif action.startswith("chip:"):
        cid = action.removeprefix("chip:")
        chip_adv = next((a for a in advice if a.chip == cid), None)
        text = f"Play {_chip_name(cid)} ({cid}) this GW"
        if chip_adv is not None and chip_adv.ev_now is not None:
            text += f": {chip_adv.ev_now:+.1f} xP vs the best no-chip plan"
            if (
                chip_adv.best_gw is not None
                and chip_adv.best_gw != gw
                and chip_adv.best_ev is not None
            ):
                text += f" (best future window GW{chip_adv.best_gw}: {chip_adv.best_ev:+.1f})"
        else:
            text += " per the optimal plan"
        bullets.append(text + ".")

    if action != "hold":
        bullets.extend(_transfer_bullet(pair, window, gw) for pair in pairs)

    if hits > 0:
        n_hits = hits // 4
        plural = "s" if n_hits != 1 else ""
        bullets.append(
            f"Plan takes {n_hits} hit{plural} (−{hits} pts) this GW; horizon "
            f"objective {objective:.1f} already nets the cost."
        )

    if captain is not None:
        cap_xp = _gw_xp(window, captain, gw)
        text = f"Captain {_name(captain, names)} ({cap_xp:.1f} xP this GW)"
        if vice is not None:
            text += f"; vice {_name(vice, names)}"
            if captain_q0 is not None:
                text += f" (captain q0 {captain_q0 * 100:.0f}%)"
        bullets.append(text + ".")

    # Simulation-backed chips speak probability; the rest keep the EV-window bullet.
    sim_backed = [a for a in advice if a.p_beats_hold is not None]
    bullets.extend(_sim_chip_bullet(a) for a in sim_backed)

    if not action.startswith("chip:"):
        held = [
            a
            for a in advice
            if a.p_beats_hold is None
            and a.verdict == "hold"
            and a.best_ev is not None
            and a.best_ev > 0
        ]
        if held:
            best = max(held, key=lambda a: a.best_ev or 0.0)
            bullets.append(
                f"Best future chip window: {_chip_name(best.chip)} ({best.chip}) in "
                f"GW{best.best_gw} ({best.best_ev:+.1f} xP vs no chip) — hold for now."
            )

    if dream is not None and action != "initial-squad":
        gap = expected_points - dream.expected_points
        bullets.append(
            f"Fresh-£100m benchmark: {dream.expected_points:.1f} xP this GW vs your "
            f"{expected_points:.1f} ({gap:+.1f})."
        )

    if stability_entries:
        supports = [p.support_pct for p in pairs if p.support_pct is not None]
        if supports:
            mean_support = sum(supports) / len(supports)
            bullets.append(
                f"Stability: {mean_support:.0f}% of noise re-solves agree with this "
                f"GW's moves."
            )
        else:
            top = max(stability_entries, key=lambda e: e.support_pct)
            bullets.append(
                f"Most stable this-GW move across noise re-solves: {top.move} "
                f"({top.support_pct:.0f}%)."
            )

    return bullets


# --------------------------------------------------------------------------------------
# The weekly verdict
# --------------------------------------------------------------------------------------


def build_recommendation(
    state: SquadState | None,
    xp: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    as_of: datetime,
    horizon: int = 8,
    params: Any | None = None,
    dream_horizon: int = 1,
    stability_n: int = 30,
    stability_seed: int = 0,
    run_chips: bool = True,
    run_stability: bool = True,
    chip_sim: Any | None = None,
    price_trends: Mapping[int, float] | None = None,
    next_deadline_utc: datetime | None = None,
) -> Recommendation:
    """Build the weekly verdict for a user squad state (or the initial-squad product).

    Runs (a) the user-state multi-GW MILP plan, (b) the None-state dream-team benchmark
    for the same GW, (c) chip EV curves over the horizon for every available chip and
    (d) plan-stability noise re-solves, then distills everything into a single
    :class:`Recommendation`.

    ``state=None`` is the pre-GW1 initial-squad mode: the returned action is
    ``"initial-squad"`` with the full 15, XI, captaincy, bench order and rationale (the
    plan itself doubles as the dream-team benchmark — no second solve). Otherwise the
    action is ``"hold"`` when the optimal plan makes 0 transfers this GW,
    ``"chip:<id>"`` when it plays a chip this GW, and ``"transfer"`` otherwise.

    ``as_of`` is the caller-supplied timestamp stamped onto the result (this function
    never reads the clock). ``run_chips`` / ``run_stability`` let callers skip the
    chip-curve and noise re-solves for speed; ``params`` is forwarded to
    ``solve_plan`` untouched when provided.

    ``chip_sim`` is the season-simulation integration hook: pass a
    ``ChipSimReport``-shaped object (or its v2 frame) from
    ``optimizer.season_sim.simulate_chip_plans`` and the simulation verdicts,
    uncertainty and confidence are folded into ``chip_advice`` (via
    :func:`build_chip_advice_from_sim` + :func:`merge_chip_advice`) and the chip
    rationale speaks simulation probability instead of raw point deltas.
    """
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    season = int(_get(state, "season") if state is not None else xp["season"].min())
    gw = int(
        _get(state, "current_gw")
        if state is not None
        else xp.loc[xp["season"] == season, "gw"].min()
    )

    plan = _call_solve(xp, prices, state, horizon, params)
    plan_first = _plan_gw(plan, gw)
    gw = int(_get(plan_first, "gw", gw))
    window = _xp_window(xp, season, gw, horizon)

    names = _name_map(prices)
    positions = _pos_map(prices)
    price_of = _price_map(prices)

    squad = [int(c) for c in _get(plan_first, "squad", []) or []]
    lineup = [int(c) for c in _get(plan_first, "lineup", []) or []]
    bench_order = [int(c) for c in _get(plan_first, "bench_order", []) or []]
    captain_raw = _get(plan_first, "captain")
    vice_raw = _get(plan_first, "vice")
    captain = int(captain_raw) if captain_raw is not None else None
    vice = int(vice_raw) if vice_raw is not None else None
    transfers_in = [int(c) for c in _get(plan_first, "transfers_in", []) or []]
    transfers_out = [int(c) for c in _get(plan_first, "transfers_out", []) or []]
    hits = int(_get(plan_first, "hit_points", 0) or 0)
    chip_now = _get(plan_first, "chip")
    expected_points = float(_get(plan_first, "expected_points", 0.0) or 0.0)
    objective = float(_get(plan, "objective", 0.0) or 0.0)

    # ---- (d) stability -----------------------------------------------------------
    stability_entries: list[StabilityEntry] = []
    if run_stability:
        stability_df = plan_stability(xp, prices, state, n=stability_n, seed=stability_seed)
        stability_entries = _normalize_stability(stability_df)

    # ---- transfer pairs with horizon xP deltas -----------------------------------
    pairs: list[TransferPair] = []
    for code_in, code_out in _pair_codes(transfers_in, transfers_out, positions):
        xp_in = _sum_xp(window, code_in) if code_in is not None else 0.0
        xp_out = _sum_xp(window, code_out) if code_out is not None else 0.0
        pairs.append(
            TransferPair(
                player_in=code_in,
                player_in_name=_name(code_in, names) if code_in is not None else None,
                player_out=code_out,
                player_out_name=_name(code_out, names) if code_out is not None else None,
                position=positions.get(code_in if code_in is not None else code_out),
                xp_in=xp_in,
                xp_out=xp_out,
                xp_delta=xp_in - xp_out,
                out_q0=_q0(xp, season, gw, code_out) if code_out is not None else None,
                support_pct=_support_for(stability_entries, code_in, code_out),
            )
        )

    # ---- (c) chip EV curves ------------------------------------------------------
    if state is not None:
        chip_ids = [str(c) for c in _get(state, "chips_available", []) or []]
    else:
        chip_ids = sorted(_chip_windows_by_id(season))
    windows_by_id = _chip_windows_by_id(season)
    gw_range = range(gw, gw + horizon)
    active_ids = [
        cid
        for cid in chip_ids
        if cid not in windows_by_id
        or (windows_by_id[cid].first_gw <= gw_range[-1] and windows_by_id[cid].last_gw >= gw)
    ]
    curves: pd.DataFrame | None = None
    if run_chips and active_ids:
        curves = chip_ev_curves(xp, prices, state, active_ids, gw_range)
    advice = _chip_advice(chip_ids, curves, plan, gw)
    if chip_sim is not None:
        advice = merge_chip_advice(advice, build_chip_advice_from_sim(chip_sim))

    # ---- (b) dream-team benchmark ------------------------------------------------
    if state is None:
        dream = _dream_from_plan(plan, plan_first, season, gw, prices)
    else:
        dream = dream_team(xp, prices, gw, season, horizon=dream_horizon, params=params)

    # ---- action selection --------------------------------------------------------
    if state is None:
        action = "initial-squad"
    elif chip_now is not None:
        action = f"chip:{chip_now}"
    elif not transfers_in and not transfers_out:
        action = "hold"
    else:
        action = "transfer"

    squad_cost = sum(price_of.get(c, 0) for c in squad)
    rationale = _build_rationale(
        action=action,
        state=state,
        season=season,
        gw=gw,
        horizon=horizon,
        pairs=pairs,
        hits=hits,
        captain=captain,
        vice=vice,
        captain_q0=_q0(xp, season, gw, captain) if captain is not None else None,
        expected_points=expected_points,
        objective=objective,
        dream=dream,
        advice=advice,
        stability_entries=stability_entries,
        names=names,
        window=window,
        squad_cost=squad_cost,
    )
    rationale.extend(
        _execution_bullets(action, pairs, price_trends, next_deadline_utc)
    )

    return Recommendation(
        season=season,
        gw=gw,
        as_of=as_of,
        action=action,
        transfers=pairs,
        hits=hits,
        captain=captain,
        vice=vice,
        lineup=lineup,
        bench_order=bench_order,
        squad=squad,
        formation=_formation(lineup, positions),
        expected_points=expected_points,
        objective=objective,
        chip_advice=advice,
        dream_team=dream,
        plan=_plan_dump(plan),
        stability=stability_entries,
        rationale=rationale,
    )
