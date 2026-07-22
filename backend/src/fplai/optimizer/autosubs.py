"""Automatic-substitution Monte Carlo: true bench weights per MODEL_DESIGN_INPUTS §4.

FPL processes autosubs at lockdown (FPL_KNOWLEDGE §1.5):

* a starter with 0 minutes in the GW is replaced by the highest-priority bench player
  whose entry keeps the formation legal (GKP exactly 1, DEF >= 3, MID >= 2, FWD >= 1);
* the starting GK is only ever replaced by the bench GK (and the bench GK never covers
  an outfield absence);
* players who played but scored <= 0 points are NOT substituted — only non-appearance
  triggers a sub.

:func:`bench_weights_mc` estimates, by vectorized Monte Carlo over independent
Bernoulli(q0) absences (starters and bench alike), the probability that each bench
slot's points end up counting *conditional on that bench player appearing*. Those
conditional probabilities are the correct multiplier for a bench player's xP, because
xP already embeds his own appearance probability (xp = (1 - q0) * E[pts | plays]);
multiplying xP by the unconditional sub-in probability would double-count his q0.
Under Bench Boost every bench slot counts, so the weights are exactly 1.0 — consistent
with this conditional semantics.

Tie-break approximation: when a bench player could legally replace absent starters of
several positions, FPL uses the first absent starter in team-sheet order (GK, DEF, MID,
FWD). We replicate that as position order DEF -> MID -> FWD, which matches team-sheet
order at position granularity (within a position group the choice never changes which
bench slots get used).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

#: Minimum on-pitch players per position for a legal formation (FPL_KNOWLEDGE §1.3/§1.5).
#: Upper bounds (DEF/MID <= 5, FWD <= 3) are implied by the 2/5/5/3 squad quotas and
#: never need checking during substitution.
FORMATION_MIN: dict[str, int] = {"GKP": 1, "DEF": 3, "MID": 2, "FWD": 1}

_OUTFIELD: tuple[str, ...] = ("DEF", "MID", "FWD")


def _validate_q0(name: str, q0: Mapping[int, float]) -> None:
    for code, q in q0.items():
        if not 0.0 <= float(q) <= 1.0:
            raise ValueError(f"{name}[{code}] must be a probability in [0, 1], got {q!r}")


def bench_weights_mc(
    lineup_q0: Mapping[int, float],
    bench_q0: Mapping[int, float],
    formation: Mapping[str, int],
    *,
    positions: Mapping[int, str],
    bench_order: Sequence[int] | None = None,
    n: int = 2000,
    seed: int = 0,
    bench_boost: bool = False,
) -> np.ndarray:
    """Estimate per-bench-slot autosub probabilities via Monte Carlo.

    Parameters
    ----------
    lineup_q0:
        ``player_code -> q0`` (P(0 minutes) from the minutes model) for the 11 starters.
    bench_q0:
        ``player_code -> q0`` for the 4 bench players. Outfield priority defaults to
        this mapping's insertion order; pass ``bench_order`` to be explicit.
    formation:
        Starting-XI position counts, e.g. ``{"GKP": 1, "DEF": 4, "MID": 4, "FWD": 2}``.
        Must match the positions of ``lineup_q0`` and be a legal formation.
    positions:
        ``player_code -> "GKP"|"DEF"|"MID"|"FWD"`` covering all 15 players.
    bench_order:
        The 3 outfield bench player codes in priority order (slots 1-3). Slot 0 is
        always the bench GK (inferred from ``positions``).
    n:
        Monte Carlo draws (default 2000).
    seed:
        RNG seed — results are deterministic for a given seed.
    bench_boost:
        If True (a BB week) every bench slot scores: returns all-ones immediately.

    Returns
    -------
    numpy.ndarray
        Shape ``(4,)``: for slots ``[GK, 1, 2, 3]``, P(the slot's points count | the
        bench player appears) — the multiplier to apply to each bench player's xP.
        The unconditional sub-in probability is ``weights * (1 - q0_bench)``.
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if len(lineup_q0) != 11:
        raise ValueError(f"lineup_q0 must have 11 starters, got {len(lineup_q0)}")
    if len(bench_q0) != 4:
        raise ValueError(f"bench_q0 must have 4 bench players, got {len(bench_q0)}")
    _validate_q0("lineup_q0", lineup_q0)
    _validate_q0("bench_q0", bench_q0)

    counts: dict[str, int] = {p: 0 for p in FORMATION_MIN}
    for code in lineup_q0:
        pos = positions[code]
        if pos not in counts:
            raise ValueError(f"unknown position {pos!r} for player {code}")
        counts[pos] += 1
    for pos in FORMATION_MIN:
        if counts[pos] != int(formation.get(pos, 0)):
            raise ValueError(f"formation {dict(formation)} does not match lineup counts {counts}")
    if counts["GKP"] != 1 or any(counts[p] < FORMATION_MIN[p] for p in _OUTFIELD):
        raise ValueError(f"starting formation {counts} is not legal")

    bench_gks = [c for c in bench_q0 if positions[c] == "GKP"]
    if len(bench_gks) != 1:
        raise ValueError(f"bench must contain exactly one GKP, got {len(bench_gks)}")
    gk_bench = bench_gks[0]
    if bench_order is None:
        outfield_bench = [c for c in bench_q0 if c != gk_bench]
    else:
        outfield_bench = list(bench_order)
    if sorted(outfield_bench) != sorted(c for c in bench_q0 if c != gk_bench):
        raise ValueError("bench_order must be a permutation of the outfield bench players")

    if bench_boost:
        return np.ones(4)

    starters = list(lineup_q0)
    q0_start = np.array([float(lineup_q0[c]) for c in starters])
    q0_bench = np.array([float(bench_q0[c]) for c in (gk_bench, *outfield_bench)])

    rng = np.random.default_rng(seed)
    absent = rng.random((n, 11)) < q0_start[None, :]  # starter missed the whole GW
    plays = rng.random((n, 4)) < (1.0 - q0_bench)[None, :]  # bench player appeared

    gk_col = next(i for i, c in enumerate(starters) if positions[c] == "GKP")
    pos_idx = {p: [i for i, c in enumerate(starters) if positions[c] == p] for p in _OUTFIELD}

    # Per-draw state: unreplaced absent starters and current XI position counts.
    remaining = {p: absent[:, pos_idx[p]].sum(axis=1).astype(np.int64) for p in _OUTFIELD}
    xi_counts = {p: np.full(n, counts[p], dtype=np.int64) for p in _OUTFIELD}

    subbed = np.zeros((n, 4), dtype=bool)
    subbed[:, 0] = absent[:, gk_col] & plays[:, 0]  # GK-for-GK only

    for slot, code in enumerate(outfield_bench, start=1):
        b_pos = positions[code]
        eligible = plays[:, slot].copy()
        for p in _OUTFIELD:  # team-sheet order tie-break: DEF -> MID -> FWD
            if not eligible.any():
                break
            new = {q: xi_counts[q] - int(q == p) + int(q == b_pos) for q in _OUTFIELD}
            legal = np.ones(n, dtype=bool)
            for q in _OUTFIELD:
                legal &= new[q] >= FORMATION_MIN[q]
            take = eligible & (remaining[p] > 0) & legal
            if take.any():
                step = take.astype(np.int64)
                remaining[p] = remaining[p] - step
                xi_counts[p] = xi_counts[p] - step
                xi_counts[b_pos] = xi_counts[b_pos] + step
                subbed[:, slot] |= take
                eligible &= ~take

    p_sub = subbed.mean(axis=0)  # unconditional P(slot subbed in)
    appears = 1.0 - q0_bench
    weights = np.divide(p_sub, appears, out=np.zeros(4), where=appears > 0.0)
    return np.clip(weights, 0.0, 1.0)


def expected_autosub_points(
    bench_xp: Sequence[float],
    probs: Sequence[float] | np.ndarray,
) -> float:
    """Expected points recovered from the bench: ``sum(bench_xp * probs)``.

    ``bench_xp`` and ``probs`` are aligned to bench slots ``[GK, 1, 2, 3]``; ``probs``
    are the conditional weights from :func:`bench_weights_mc` (xP already embeds each
    bench player's own appearance probability).
    """
    xp_arr = np.asarray(bench_xp, dtype=float)
    pr_arr = np.asarray(probs, dtype=float)
    if xp_arr.shape != pr_arr.shape:
        raise ValueError(f"shape mismatch: bench_xp {xp_arr.shape} vs probs {pr_arr.shape}")
    return float(np.dot(xp_arr, pr_arr))
