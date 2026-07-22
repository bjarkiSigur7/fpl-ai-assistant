"""Chip EV curves: forced-chip re-solves vs a no-chip baseline (MODEL_DESIGN_INPUTS §4).

Chip EV 4-8 GWs out is dominated by projection uncertainty, so a single joint MILP solve
is not trusted for chip timing. Instead :func:`chip_ev_curves` sweeps each candidate GW,
re-solving the plan with the chip *forced* at that GW (and all other chips banned) and
compares the objective against a no-chip baseline solve.

Coordination with ``optimizer.milp`` (built in parallel): chip directives are passed on
the ``SolveParams`` object via the field names in :data:`FORCED_CHIPS_FIELD` /
:data:`BANNED_CHIPS_FIELD` — ``forced_chips`` maps ``gw -> chip_id`` (e.g. ``{5: "bb1"}``)
and ``banned_chips`` is a set of chip ids the solver may not play. If the milp agent's
parameter shape differs, adapt these two constants / :func:`copy_params_with` — the only
call-site glue. Chip ids follow ``optimizer.state``: ``"wc1"``, ``"fh1"``, ``"bb1"``,
``"tc1"``, ``"wc2"``, ... (name + half-season set number).
"""

from __future__ import annotations

import copy
import dataclasses
import re
from collections.abc import Callable, Iterable, Sequence
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

from fplai import rules
from fplai.optimizer.milp import InfeasiblePlanError

#: Attribute on the params object carrying ``{gw: chip_id}`` forced-chip directives.
FORCED_CHIPS_FIELD: str = "forced_chips"
#: Attribute on the params object carrying the set of banned chip ids.
BANNED_CHIPS_FIELD: str = "banned_chips"

#: ``solve_plan``-shaped callable: ``(xp, prices, state, *, horizon, params) -> PlanResult``.
SolvePlanFn = Callable[..., Any]

_CHIP_ID_RE = re.compile(r"^(wc|fh|bb|tc)([12])$")
_CHIP_NAMES = {"wc": "WC", "fh": "FH", "bb": "BB", "tc": "TC"}


# --------------------------------------------------------------------------------------
# Solver glue (shared with optimizer.sensitivity)
# --------------------------------------------------------------------------------------


def resolve_solver(solve_fn: SolvePlanFn | None) -> SolvePlanFn:
    """Return ``solve_fn`` or lazily import ``fplai.optimizer.milp.solve_plan``."""
    if solve_fn is not None:
        return solve_fn
    try:
        from fplai.optimizer import milp
    except ImportError as exc:  # pragma: no cover - depends on parallel build state
        raise ImportError("optimizer.milp is not available; pass solve_fn= explicitly") from exc
    return milp.solve_plan


def default_solve_params() -> Any | None:
    """Best-effort default ``SolveParams`` from ``optimizer.milp`` (None if unavailable)."""
    try:
        from fplai.optimizer import milp
    except ImportError:
        return None
    params_cls = getattr(milp, "SolveParams", None)
    if params_cls is None:
        return None
    try:
        return params_cls()
    except Exception:  # pragma: no cover - depends on parallel build state
        return None


def copy_params_with(params: Any | None, **updates: Any) -> Any:
    """Return a copy of ``params`` with ``updates`` applied (pydantic/dataclass/plain).

    ``None`` becomes a :class:`types.SimpleNamespace` carrying only the updates, so stub
    solvers (and a not-yet-landed milp) can still be exercised.
    """
    if params is None:
        return SimpleNamespace(**updates)
    if hasattr(params, "model_copy"):  # pydantic v2
        return params.model_copy(update=dict(updates))
    if dataclasses.is_dataclass(params) and not isinstance(params, type):
        try:
            return dataclasses.replace(params, **updates)
        except TypeError:
            pass  # unknown fields -> fall through to attribute copy
    clone = copy.copy(params)
    for key, value in updates.items():
        setattr(clone, key, value)
    return clone


def squad_codes(state: Any | None) -> list[int]:
    """Extract owned player codes from a ``SquadState``-shaped object (or [])."""
    if state is None:
        return []
    squad = getattr(state, "squad", None) or []
    codes: list[int] = []
    for entry in squad:
        if isinstance(entry, int):
            codes.append(entry)
        elif hasattr(entry, "player_code"):
            codes.append(int(entry.player_code))
        else:
            codes.append(int(entry["player_code"]))
    return codes


# --------------------------------------------------------------------------------------
# Chip windows
# --------------------------------------------------------------------------------------


def _chip_window(season: int, chip_id: str) -> rules.ChipWindow:
    """Map a chip id like ``"bb1"`` to its :class:`fplai.rules.ChipWindow` for ``season``."""
    match = _CHIP_ID_RE.match(chip_id.lower())
    if match is None:
        raise ValueError(f"unrecognized chip id {chip_id!r} (expected e.g. 'wc1', 'bb2')")
    name, set_no = _CHIP_NAMES[match.group(1)], int(match.group(2))
    for window in rules.chip_windows(season):
        if window.name == name and window.set == set_no:
            return window
    raise ValueError(f"chip {chip_id!r} does not exist in season {season}")


def season_chip_ids(season: int) -> list[str]:
    """All chip ids playable in ``season`` (e.g. ``["wc1", "wc2", "fh1", ...]``)."""
    return [
        f"{window.name.lower()}{window.set}"
        for window in rules.chip_windows(season)
        if window.name in _CHIP_NAMES.values()  # WC/FH/BB/TC only (no AOA/AM)
    ]


# --------------------------------------------------------------------------------------
# Dominance heuristics (skip flags, never silent drops)
# --------------------------------------------------------------------------------------


def _bb_bench_signal(xp_gw: pd.DataFrame, squad: list[int]) -> float:
    """Proxy for bench xP at a GW: sum of the 4 lowest squad xPs (or pool ranks 12-15)."""
    if squad:
        vals = np.sort(
            xp_gw.set_index("player_code")["xp"].reindex(squad).fillna(0.0).to_numpy(dtype=float)
        )
        return float(vals[:4].sum())
    top = xp_gw["xp"].nlargest(15).to_numpy(dtype=float)
    return float(top[11:15].sum()) if len(top) > 11 else 0.0


def _tc_captain_signal(xp_gw: pd.DataFrame) -> float:
    """Proxy for TC value at a GW: the best single-player xP in the pool."""
    return float(xp_gw["xp"].max()) if len(xp_gw) else 0.0


# --------------------------------------------------------------------------------------
# Chip EV curves
# --------------------------------------------------------------------------------------


def chip_ev_curves(
    xp: pd.DataFrame,
    prices: pd.DataFrame,
    state: Any | None,
    chips: Sequence[str],
    gw_range: Iterable[int],
    *,
    horizon: int = 8,
    params: Any | None = None,
    solve_fn: SolvePlanFn | None = None,
    skip_dominated: bool = True,
    expiry_proximity: int = 5,
) -> pd.DataFrame:
    """EV curve per chip: forced-at-GW re-solves vs one no-chip baseline solve.

    Parameters
    ----------
    xp, prices:
        The frames ``solve_plan`` consumes (``season, gw, player_code, xp[, q0]`` /
        ``player_code, price, position, team_code``).
    state:
        ``SquadState`` (or None for the initial-build state).
    chips:
        Chip ids to sweep (e.g. ``["bb1", "tc1"]``).
    gw_range:
        Candidate GWs. Cells outside the chip's window/horizon are returned flagged,
        not silently dropped.
    horizon, params, solve_fn:
        Forwarded to the solver; ``params`` copies get the chip directives set (see
        module docstring); ``solve_fn`` defaults to ``optimizer.milp.solve_plan``.
    skip_dominated:
        Skip obviously-dominated cells with a flag: BB on GWs whose bench-xP proxy is
        bottom-quartile, TC on GWs whose best-captain xP is bottom-quartile (only when
        >= 4 candidate GWs exist; WC/FH cells are never skipped).
    expiry_proximity:
        A chip whose window ends within this many GWs of ``current_gw`` (or inside
        ``gw_range``) gets an urgency annotation.

    Returns
    -------
    pandas.DataFrame
        Columns ``chip, gw, objective, delta_vs_no_chip, evaluated, skip_reason,
        window_last_gw, urgency``. ``objective``/``delta_vs_no_chip`` are NaN for
        unevaluated cells (``evaluated`` False + ``skip_reason`` says why, so the UI
        can distinguish "not evaluated" from "negative EV"). ``urgency`` (near-expiry
        chips only) is ``delta[gw] - max(0, best delta at any later evaluated GW in
        the window)`` — positive means playing now beats every remaining alternative.
        ``df.attrs["baseline_objective"]`` holds the no-chip objective.
    """
    solver = resolve_solver(solve_fn)
    if params is None:
        params = default_solve_params()

    if state is not None:
        season = int(state.season)
        current_gw = int(state.current_gw)
    else:
        season = int(xp["season"].iloc[0]) if len(xp) else 2026
        current_gw = int(xp["gw"].min()) if len(xp) else 1

    gws = sorted({int(g) for g in gw_range})
    all_chip_ids = set(season_chip_ids(season)) | set(chips)
    available = getattr(state, "chips_available", None)
    available_set = set(available) if available is not None else None

    baseline_params = copy_params_with(
        params,
        **{FORCED_CHIPS_FIELD: {}, BANNED_CHIPS_FIELD: frozenset(all_chip_ids)},
    )
    baseline = solver(xp, prices, state, horizon=horizon, params=baseline_params)
    baseline_obj = float(baseline.objective)

    xp_by_gw = {g: xp[xp["gw"] == g] for g in gws}
    squad = squad_codes(state)

    rows: list[dict[str, Any]] = []
    for chip in chips:
        window = _chip_window(season, chip)
        candidates = [
            g
            for g in gws
            if window.first_gw <= g <= window.last_gw and current_gw <= g < current_gw + horizon
        ]

        skip_gws: dict[int, str] = {}
        if skip_dominated and len(candidates) >= 4:
            if window.name == "BB":
                signals = {g: _bb_bench_signal(xp_by_gw[g], squad) for g in candidates}
                reason = "dominated:bb_bench_bottom_quartile"
            elif window.name == "TC":
                signals = {g: _tc_captain_signal(xp_by_gw[g]) for g in candidates}
                reason = "dominated:tc_captain_bottom_quartile"
            else:
                signals, reason = {}, ""
            if signals:
                threshold = float(np.quantile(list(signals.values()), 0.25))
                skip_gws = {g: reason for g, s in signals.items() if s < threshold}

        chip_rows: list[dict[str, Any]] = []
        for g in gws:
            row: dict[str, Any] = {
                "chip": chip,
                "gw": g,
                "objective": np.nan,
                "delta_vs_no_chip": np.nan,
                "evaluated": False,
                "skip_reason": None,
                "window_last_gw": window.last_gw,
                "urgency": np.nan,
            }
            if available_set is not None and chip not in available_set:
                row["skip_reason"] = "not_available"
            elif not window.first_gw <= g <= window.last_gw:
                row["skip_reason"] = "outside_window"
            elif not current_gw <= g < current_gw + horizon:
                row["skip_reason"] = "outside_horizon"
            elif g in skip_gws:
                row["skip_reason"] = skip_gws[g]
            elif state is None and g == current_gw and chip[:2] in ("wc", "fh"):
                # Transfer chips at the first GW of a None-state initial build are
                # degenerate and refused by the MILP — don't even attempt the force.
                row["skip_reason"] = "initial_build_first_gw"
            else:
                forced_params = copy_params_with(
                    params,
                    **{
                        FORCED_CHIPS_FIELD: {g: chip},
                        BANNED_CHIPS_FIELD: frozenset(all_chip_ids - {chip}),
                    },
                )
                try:
                    result = solver(xp, prices, state, horizon=horizon, params=forced_params)
                except InfeasiblePlanError as exc:
                    # A force the model refuses (window/state conflicts) is a skipped
                    # cell, not a crashed sweep.
                    row["skip_reason"] = f"infeasible: {exc}"
                else:
                    row["objective"] = float(result.objective)
                    row["delta_vs_no_chip"] = float(result.objective) - baseline_obj
                    row["evaluated"] = True
            chip_rows.append(row)

        near_expiry = (window.last_gw - current_gw <= expiry_proximity) or (
            gws and window.last_gw <= max(gws)
        )
        if near_expiry:
            evaluated = [r for r in chip_rows if r["evaluated"]]
            for row in evaluated:
                later = [
                    r["delta_vs_no_chip"]
                    for r in evaluated
                    if row["gw"] < r["gw"] <= window.last_gw
                ]
                best_alternative = max(0.0, max(later, default=0.0))
                row["urgency"] = row["delta_vs_no_chip"] - best_alternative
        rows.extend(chip_rows)

    df = pd.DataFrame(
        rows,
        columns=[
            "chip",
            "gw",
            "objective",
            "delta_vs_no_chip",
            "evaluated",
            "skip_reason",
            "window_last_gw",
            "urgency",
        ],
    )
    df.attrs["baseline_objective"] = baseline_obj
    return df
