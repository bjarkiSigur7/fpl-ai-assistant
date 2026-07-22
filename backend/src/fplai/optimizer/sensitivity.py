"""Plan stability under projection noise (MODEL_DESIGN_INPUTS §3.4).

Re-solve the plan ``n`` times with perturbed projections and report how often each
this-GW decision (transfer buys/sells, captain, chip — or "hold") appears in the
optimal plan. The §3.4 community noise model is applied exactly:

    Pts'[p, w] = Pts[p, w] + strength * Pts[p, w] * (92 - xMins[p, w]) / 134 * N(0, 1)

with ``xMins = 90 * (q1 + q2) = 90 * (1 - q0)`` approximated from the minutes model's
``q0`` column carried on the xp frame (relative noise, inflated for low-minutes
players). Each re-solve draws from an independent child seed of one
:class:`numpy.random.SeedSequence`, so runs are distinct but the whole study is
deterministic for a given ``seed``.

To keep ``n = 30`` re-solves tractable the player pool is pruned ONCE from the
unperturbed projections (top total-xP players per position + the current squad) and the
same reduced ``xp``/``prices`` frames are reused for every run — the solver's own §3.5
pruning then operates on an already-small pool.
"""

from __future__ import annotations

import warnings
from collections import Counter
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from fplai.optimizer.chips import (
    SolvePlanFn,
    default_solve_params,
    resolve_solver,
    squad_codes,
)

#: Per-position floor on warm-pool size — keeps every squad shape feasible
#: (quotas 2/5/5/3 need spare cover beyond the owned squad).
_POOL_POSITION_MINIMA: dict[str, int] = {"GKP": 5, "DEF": 15, "MID": 15, "FWD": 10}


def perturb_xp(
    xp: pd.DataFrame,
    *,
    strength: float,
    rng: np.random.Generator,
    q0_col: str = "q0",
) -> pd.DataFrame:
    """Return a copy of ``xp`` with the §3.4 relative-noise model applied to ``xp["xp"]``.

    ``xMins = 90 * (1 - q0)`` from ``q0_col``; when the column is missing every player
    is treated as nailed (``q0 = 0`` -> xMins 90 -> minimal noise) with a warning.
    """
    out = xp.copy()
    base = xp["xp"].to_numpy(dtype=float)
    if q0_col in xp.columns:
        q0 = xp[q0_col].astype(float).clip(0.0, 1.0).to_numpy()
    else:
        warnings.warn(
            f"xp frame has no {q0_col!r} column; assuming q0=0 (minimal noise inflation)",
            stacklevel=2,
        )
        q0 = np.zeros(len(xp))
    xmins = 90.0 * (1.0 - q0)
    scale = strength * (92.0 - xmins) / 134.0
    out["xp"] = base + base * scale * rng.standard_normal(len(xp))
    return out


def _warm_pool(
    xp: pd.DataFrame,
    prices: pd.DataFrame,
    squad: list[int],
    pool_size: int,
) -> set[int]:
    """Pick the reusable player pool: squad + per-position top-xP + overall top-xP fill."""
    totals = xp.groupby("player_code", sort=False)["xp"].sum().sort_values(ascending=False)
    position: Mapping[int, str] = (
        prices.drop_duplicates("player_code").set_index("player_code")["position"].to_dict()
    )
    pool: set[int] = set(squad)
    by_position: dict[str, list[int]] = {p: [] for p in _POOL_POSITION_MINIMA}
    for code in totals.index:
        pos = position.get(int(code))
        if pos in by_position:
            by_position[pos].append(int(code))
    for pos, minimum in _POOL_POSITION_MINIMA.items():
        pool.update(by_position[pos][:minimum])
    for code in totals.index:  # fill by overall rank up to pool_size
        if len(pool) >= pool_size:
            break
        pool.add(int(code))
    return pool


def _plan_moves(result: Any) -> set[str]:
    """Extract this-GW decision labels from a ``PlanResult``-shaped object."""
    gws = result["gws"] if isinstance(result, Mapping) else getattr(result, "gws", None)
    if not gws:
        raise ValueError("solver result has no gws — cannot extract this-GW moves")
    plan = gws[0]

    def get(name: str) -> Any:
        return plan.get(name) if isinstance(plan, Mapping) else getattr(plan, name, None)

    moves = {f"buy:{int(c)}" for c in (get("transfers_in") or [])}
    moves |= {f"sell:{int(c)}" for c in (get("transfers_out") or [])}
    if not moves:
        moves.add("hold")
    captain = get("captain")
    if captain is not None:
        moves.add(f"captain:{int(captain)}")
    chip = get("chip")
    moves.add(f"chip:{chip}" if chip else "chip:none")
    return moves


def plan_stability(
    xp: pd.DataFrame,
    prices: pd.DataFrame,
    state: Any | None,
    n: int = 30,
    strength: float = 1.0,
    seed: int = 0,
    *,
    horizon: int = 8,
    params: Any | None = None,
    solve_fn: SolvePlanFn | None = None,
    pool_size: int = 150,
    q0_col: str = "q0",
) -> pd.DataFrame:
    """Support percentage of each this-GW move across ``n`` noise re-solves.

    Parameters
    ----------
    xp, prices, state:
        As for ``optimizer.milp.solve_plan``; ``xp`` should carry the minutes model's
        ``q0`` (used both for vice weighting and, here, the noise inflation).
    n:
        Number of perturbed re-solves (distinct child seeds of ``seed``).
    strength:
        Noise strength multiplier in the §3.4 model (0 = no noise; 1 = community
        default).
    seed:
        Seed for the whole study — deterministic output for a fixed seed.
    horizon, params, solve_fn:
        Forwarded to the solver (``solve_fn`` defaults to
        ``optimizer.milp.solve_plan``).
    pool_size:
        Warm player-pool size reused across all runs (plus per-position minima and the
        current squad, which are always kept).
    q0_col:
        Column on ``xp`` holding q0 = P(0 minutes).

    Returns
    -------
    pandas.DataFrame
        Columns ``move, support_pct, count`` sorted by support (then move name):
        ``move`` is ``buy:<code>``, ``sell:<code>``, ``hold``, ``captain:<code>``,
        ``chip:<id>`` or ``chip:none``; ``support_pct`` = 100 * count / n.
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    solver = resolve_solver(solve_fn)
    if params is None:
        params = default_solve_params()

    squad = squad_codes(state)
    pool = _warm_pool(xp, prices, squad, pool_size)
    xp_pool = xp[xp["player_code"].isin(pool)].reset_index(drop=True)
    prices_pool = prices[prices["player_code"].isin(pool)].reset_index(drop=True)

    counter: Counter[str] = Counter()
    for child in np.random.SeedSequence(seed).spawn(n):
        rng = np.random.default_rng(child)
        noisy = perturb_xp(xp_pool, strength=strength, rng=rng, q0_col=q0_col)
        result = solver(noisy, prices_pool, state, horizon=horizon, params=params)
        counter.update(_plan_moves(result))

    df = pd.DataFrame(
        [{"move": move, "count": count} for move, count in counter.items()],
        columns=["move", "count"],
    )
    df["support_pct"] = 100.0 * df["count"] / n
    df = df.sort_values(["support_pct", "move"], ascending=[False, True]).reset_index(drop=True)
    return df[["move", "support_pct", "count"]]
