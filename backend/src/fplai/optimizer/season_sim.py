"""Full-window Monte Carlo chip planner (ARCHITECTURE "Season-simulation contracts", stage 6).

Fixes the greedy-chip-within-horizon problem: instead of letting an 8-GW MILP cluster all
set-1 chips into the first weeks, :func:`simulate_chip_plans` evaluates every candidate
chip placement over the FULL remaining chip window (e.g. GWs 1-19 for set-1 chips) and
quantifies the value of *holding* with Monte Carlo rollouts. Three layers:

1. **Backbone**: one no-chip MILP over the whole window (all chips banned, aggressive
   player-pool pruning, hard time cap). Its squad/lineup trajectory is the "hold"
   behavior every chip placement is measured against.
2. **Candidate placements**: BB/TC gains are analytic per backbone GW (bench-xP sum /
   captain xP). WC/FH gains come from bounded segment re-solves: for candidate GW ``g``
   only ``[g, min(g + segment_len - 1, window_end)]`` is re-solved with the chip forced,
   starting from the backbone squad at ``g``, against a no-chip re-solve of the same
   segment from the same state.
3. **Monte Carlo**: a ``models.sampler.PointsSampler`` draws ``n_rollouts`` realized
   seasons; every candidate placement is scored per rollout (autosubs and captain->vice
   fallback resolved from the sampled realization), yielding E[gain], sd, P(best week),
   P(playing now beats holding) and a top-k joint-schedule ranking.

Documented approximations (surfaced verbatim in ``ChipSimReport.assumptions``):

* **Fixed backbone** — chip gains are measured on a single no-chip plan; transfer
  strategy is not re-optimized jointly with chip timing (except locally inside WC/FH
  segments). A full joint solve over 19 GWs x all chips is beyond a sane time budget
  and its point estimate would be spuriously precise anyway; the MC layer is where the
  honest uncertainty lives.
* **Segment approximation** — WC/FH deterministic gains are objective deltas over a
  short segment, so squad value carried beyond the segment (the main reason to WC) is
  understated; treat WC/FH numbers as lower bounds on timing value, not total value.
* **Independence across chips** — joint schedules sum per-chip gains within a rollout;
  playing one chip does not change the squad another chip sees.
* **BB/autosub overlap** — BB gain counts all four bench players' sampled points and
  does not net out the autosub points the bench would have recovered anyway on a
  no-chip week (slight upward bias when starters blank).
* **Appearance proxy** — when the sampler does not expose ``sample_minutes``,
  "appeared" is inferred from nonzero sampled points (a played-but-net-zero GW counts
  as absent for autosub/vice resolution; rare, slightly bench-generous).
* **Points-only rollouts** — realized gains score lineup points (with autosubs, vice
  fallback and hit costs); the FT-continuation and ITB shadow values participate only
  in the deterministic segment objectives.

Determinism: one seed drives the sampler; the MILP is seeded and single-threaded by
default; every downstream aggregation is pure numpy — same inputs + seed = same report.
"""

from __future__ import annotations

import logging
import re
import time
from itertools import product
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from fplai import rules
from fplai.optimizer.autosubs import FORMATION_MIN
from fplai.optimizer.chips import (
    BANNED_CHIPS_FIELD,
    FORCED_CHIPS_FIELD,
    SolvePlanFn,
    copy_params_with,
    resolve_solver,
    season_chip_ids,
)
from fplai.optimizer.milp import InfeasiblePlanError, SolveParams
from fplai.optimizer.state import OwnedPlayer, SquadState

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

logger = logging.getLogger(__name__)

_CHIP_ID_RE = re.compile(r"^(wc|fh|bb|tc)([12])$")
_OUTFIELD: tuple[str, ...] = ("DEF", "MID", "FWD")

#: Backbone solve overrides: chips banned, hard time cap, aggressive §3.5 pruning.
BACKBONE_TIME_LIMIT_S: float = 120.0
BACKBONE_MIP_REL_GAP: float = 1e-3
BACKBONE_PRUNE: dict[str, Any] = {
    "prune": True,
    "prune_keep_frac": 0.06,
    "prune_value_frac": 0.03,
    "prune_keep_cheapest": 3,
}
#: Segment re-solve overrides: short horizon, short cap, an even smaller pool.
SEGMENT_TIME_LIMIT_S: float = 20.0
SEGMENT_MIP_REL_GAP: float = 5e-3
SEGMENT_PRUNE: dict[str, Any] = {
    "prune": True,
    "prune_keep_frac": 0.05,
    "prune_value_frac": 0.02,
    "prune_keep_cheapest": 3,
}

#: The approximations baked into this simulator, for the UI to surface (module docstring
#: has the long-form discussion of each).
ASSUMPTIONS: tuple[str, ...] = (
    "fixed_backbone: chip gains are measured on one no-chip backbone plan over the full "
    "window; transfers are not re-optimized jointly with chip timing outside WC/FH segments",
    "segment_approximation: WC/FH gains are objective deltas of bounded [g, g+K-1] re-solves "
    "vs a no-chip re-solve from the same backbone squad state; squad value carried beyond "
    "the segment is not credited, so WC/FH value is understated (lower bound)",
    "independence: joint schedules sum per-chip gains within each rollout; playing one chip "
    "does not change the squad another chip sees",
    "bb_autosub_overlap: BB gain counts all four bench players' sampled points and does not "
    "net out autosub points the bench would have recovered on a no-chip week",
    "appearance_proxy: without sampler.sample_draws/sample_minutes, appearance is inferred "
    "from nonzero sampled points (played-but-net-zero counts as absent for autosubs/vice)",
    "static_prices: synthetic segment squad states value players at current prices "
    "(purchase == current); price changes inside the window are ignored",
    "points_only_rollouts: realized gains score lineup points + autosubs + vice fallback - "
    "hits; FT-continuation/ITB values only enter the deterministic segment objectives",
    "frozen_predictions: the whole window is scored on today's predictions; news, form and "
    "price moves inside the window are not resampled",
)


# --------------------------------------------------------------------------------------
# Report models (pydantic at the module boundary, per conventions)
# --------------------------------------------------------------------------------------


class ChipGwStat(BaseModel):
    """Simulation statistics for one (chip, gw) cell of the window."""

    model_config = ConfigDict(frozen=True)

    chip: str
    gw: int
    xp_gain: float | None = Field(
        default=None,
        description="Deterministic candidate-layer gain: bench-xP sum (BB), captain xP "
        "(TC), or segment objective delta (WC/FH)",
    )
    e_gain: float | None = Field(default=None, description="Monte Carlo mean realized gain")
    sd: float | None = None
    p_best_week: float | None = Field(
        default=None,
        description="P(this GW has the strictly-or-jointly best realized gain of all "
        "candidate GWs for this chip); ties split evenly, so the column sums to 1",
    )
    p_beats_hold: float | None = Field(
        default=None,
        description="P(realized gain at this GW >= the best realized gain at any LATER "
        "candidate GW, floored at 0 = never playing the chip)",
    )
    evaluated: bool = False
    skip_reason: str | None = None
    window_last_gw: int


class ChipVerdict(BaseModel):
    """Per-chip recommendation: the E[gain]-argmax GW plus its confidence numbers."""

    model_config = ConfigDict(frozen=True)

    chip: str
    recommended_gw: int | None
    e_gain: float | None = None
    sd: float | None = None
    confidence: float | None = Field(
        default=None, description="p_best_week at the recommended GW"
    )
    p_beats_hold: float | None = None


class JointSchedule(BaseModel):
    """One joint placement of the simulated chips (at most one chip per GW)."""

    model_config = ConfigDict(frozen=True)

    placements: dict[str, int] = Field(description="chip id -> GW; unplaced chips omitted")
    e_total_gain: float
    p_best: float = Field(
        description="P(this schedule has the best realized total gain of the enumerated "
        "schedules); ties split evenly"
    )


class BackboneQuality(BaseModel):
    """Honesty block: how good the no-chip backbone solve actually was."""

    model_config = ConfigDict(frozen=True)

    objective: float
    gap: float
    solve_seconds: float
    status: str
    horizon: int
    pool_hint: str | None = Field(
        default=None, description="Free-text note on pruning/pool size where known"
    )


class ChipSimReport(BaseModel):
    """Output of :func:`simulate_chip_plans` (feeds chip_sim.parquet / chip_curves v2)."""

    model_config = ConfigDict(frozen=True)

    season: int
    window_first_gw: int
    window_last_gw: int
    n_rollouts: int
    seed: int
    chips: list[str]
    stats: list[ChipGwStat]
    verdicts: list[ChipVerdict]
    schedules: list[JointSchedule]
    backbone: BackboneQuality
    assumptions: list[str]
    appearance_source: str = Field(
        description='"sample_minutes" (sampled minutes, via sample_draws or the '
        'sample_minutes duck-type) or "points_nonzero" (the fallback proxy)'
    )
    n_segment_solves: int = 0
    sim_seconds: float = 0.0

    def to_frame(self) -> pd.DataFrame:
        """The chip_curves **v2** frame: v1 columns + ``sd, p_best_week, p_beats_hold,
        n_rollouts`` (additive — consumers of the v1 schema keep working).

        ``delta_vs_no_chip`` is the Monte Carlo E[gain]; ``objective`` is the backbone
        objective plus that gain (the v1 identity ``objective = baseline + delta``);
        ``urgency`` follows chips.py: ``e_gain - max(0, best later e_gain)``.
        ``df.attrs["baseline_objective"]`` carries the backbone objective (attrs do not
        survive parquet round-trips; the report itself is the durable source).
        """
        rows: list[dict[str, Any]] = []
        for chip in self.chips:
            chip_stats = sorted((s for s in self.stats if s.chip == chip), key=lambda s: s.gw)
            for s in chip_stats:
                urgency = np.nan
                objective = np.nan
                if s.evaluated and s.e_gain is not None:
                    later = [
                        t.e_gain for t in chip_stats if t.gw > s.gw and t.e_gain is not None
                    ]
                    urgency = s.e_gain - max(0.0, max(later, default=0.0))
                    objective = self.backbone.objective + s.e_gain
                rows.append(
                    {
                        "chip": chip,
                        "gw": s.gw,
                        "objective": objective,
                        "delta_vs_no_chip": np.nan if s.e_gain is None else s.e_gain,
                        "evaluated": s.evaluated,
                        "skip_reason": s.skip_reason,
                        "window_last_gw": s.window_last_gw,
                        "urgency": urgency,
                        "sd": np.nan if s.sd is None else s.sd,
                        "p_best_week": np.nan if s.p_best_week is None else s.p_best_week,
                        "p_beats_hold": np.nan if s.p_beats_hold is None else s.p_beats_hold,
                        "n_rollouts": self.n_rollouts,
                    }
                )
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
                "sd",
                "p_best_week",
                "p_beats_hold",
                "n_rollouts",
            ],
        )
        df.attrs["baseline_objective"] = self.backbone.objective
        return df


# --------------------------------------------------------------------------------------
# Sampling adapter
# --------------------------------------------------------------------------------------


def _default_sampler() -> Any:
    """Lazily construct the real ``models.sampler.PointsSampler`` (built in parallel)."""
    try:
        from fplai.models.sampler import PointsSampler
    except ImportError as exc:  # pragma: no cover - depends on parallel build state
        raise ImportError(
            "fplai.models.sampler.PointsSampler is unavailable; pass sampler= explicitly"
        ) from exc
    return PointsSampler()


def _draw_samples(
    sampler: Any, pred: pd.DataFrame, n: int, seed: int
) -> tuple[np.ndarray, np.ndarray, str]:
    """Draw realized points (and appearance) per prediction row.

    Returns ``(points[n, rows], appeared[n, rows] bool, appearance_source)``. Appearance
    prefers the real sampler's ``sample_draws(pred, n=, seed=)`` (one pass returning
    jointly-drawn points AND minutes — the documented ``models.sampler`` contract), then
    a duck-typed ``sampler.sample_minutes(pred, n=, seed=)`` (same shape, minutes > 0 =
    appeared); otherwise nonzero points are the proxy (see module docstring).
    """
    sample_draws = getattr(sampler, "sample_draws", None)
    if callable(sample_draws):
        try:
            draws = sample_draws(pred, n=n, seed=seed)
            points = np.asarray(draws.points, dtype=float)
            minutes = np.asarray(draws.minutes, dtype=float)
        except Exception:  # noqa: BLE001 - fall back to .sample, never kill the sim
            logger.warning("sampler.sample_draws failed; falling back to sample()")
        else:
            if points.shape == (n, len(pred)) and minutes.shape == points.shape:
                return points, minutes > 0.0, "sample_minutes"
            logger.warning(
                "sampler.sample_draws shapes points=%s minutes=%s != %s; falling back",
                points.shape,
                minutes.shape,
                (n, len(pred)),
            )
    points = np.asarray(sampler.sample(pred, n=n, seed=seed), dtype=float)
    if points.shape != (n, len(pred)):
        raise ValueError(
            f"sampler.sample returned shape {points.shape}, expected {(n, len(pred))} "
            "(contract: ndarray[n, rows] aligned to the predictions frame)"
        )
    if hasattr(sampler, "sample_minutes"):
        try:
            minutes = np.asarray(sampler.sample_minutes(pred, n=n, seed=seed), dtype=float)
        except Exception:  # noqa: BLE001 - a parallel-built extension must not kill the sim
            logger.warning("sampler.sample_minutes failed; inferring appearance from points")
        else:
            if minutes.shape == points.shape:
                return points, minutes > 0.0, "sample_minutes"
            logger.warning(
                "sampler.sample_minutes shape %s != %s; inferring appearance from points",
                minutes.shape,
                points.shape,
            )
    return points, points != 0.0, "points_nonzero"


class _SampledWindow:
    """Per-(player_code, gw) sampled points/appearance, lazily aggregated over fixtures."""

    def __init__(self, pred: pd.DataFrame, points: np.ndarray, appeared: np.ndarray) -> None:
        self.n = points.shape[0]
        self._points = points
        self._appeared = appeared
        grouped = pred.groupby(["player_code", "gw"], sort=False).indices
        self._idx: dict[tuple[int, int], np.ndarray] = {
            (int(code), int(gw)): np.asarray(ix) for (code, gw), ix in grouped.items()
        }
        self._pts_cache: dict[tuple[int, int], np.ndarray] = {}
        self._app_cache: dict[tuple[int, int], np.ndarray] = {}
        self._zeros = np.zeros(self.n)
        self._false = np.zeros(self.n, dtype=bool)

    def points(self, code: int, gw: int) -> np.ndarray:
        """Sampled GW points (fixtures summed); zeros for players without predictions."""
        key = (int(code), int(gw))
        got = self._pts_cache.get(key)
        if got is None:
            ix = self._idx.get(key)
            got = self._zeros if ix is None else self._points[:, ix].sum(axis=1)
            self._pts_cache[key] = got
        return got

    def appeared(self, code: int, gw: int) -> np.ndarray:
        """Sampled appearance in the GW (any fixture); False for unknown players."""
        key = (int(code), int(gw))
        got = self._app_cache.get(key)
        if got is None:
            ix = self._idx.get(key)
            got = self._false if ix is None else self._appeared[:, ix].any(axis=1)
            self._app_cache[key] = got
        return got


# --------------------------------------------------------------------------------------
# Realized scoring of a GwPlan under one sampled window (vectorized over rollouts)
# --------------------------------------------------------------------------------------


def _autosub_points(
    gw: int,
    lineup: Sequence[int],
    bench_order: Sequence[int],
    sim: _SampledWindow,
    pos_of: Mapping[int, str],
) -> np.ndarray:
    """Points recovered by FPL autosubs per rollout (FPL_KNOWLEDGE §1.5 semantics).

    Mirrors ``optimizer.autosubs`` (bench priority order, GK-for-GK only, formation
    legality, DEF->MID->FWD team-sheet tie-break) but on *realized* appearances, adding
    each subbed-in bench player's sampled points (which may be negative — FPL applies
    the sub regardless).
    """
    if len(bench_order) < 4:
        return np.zeros(sim.n)  # BB week (bench empty) or malformed plan: nothing to do
    starters = list(lineup)
    absent = np.column_stack([~sim.appeared(p, gw) for p in starters])
    added = np.zeros(sim.n)

    gk_cols = [i for i, p in enumerate(starters) if pos_of.get(p) == "GKP"]
    gk_bench = bench_order[0]
    if gk_cols and pos_of.get(gk_bench) == "GKP":
        take = absent[:, gk_cols[0]] & sim.appeared(gk_bench, gw)
        if take.any():
            added += np.where(take, sim.points(gk_bench, gw), 0.0)

    cols = {p: [i for i, c in enumerate(starters) if pos_of.get(c) == p] for p in _OUTFIELD}
    remaining = {p: absent[:, cols[p]].sum(axis=1).astype(np.int64) for p in _OUTFIELD}
    xi_counts = {p: np.full(sim.n, len(cols[p]), dtype=np.int64) for p in _OUTFIELD}

    for code in bench_order[1:]:
        b_pos = pos_of.get(code)
        if b_pos not in _OUTFIELD:
            continue
        eligible = sim.appeared(code, gw).copy()
        for p in _OUTFIELD:  # team-sheet order tie-break
            if not eligible.any():
                break
            new = {q: xi_counts[q] - int(q == p) + int(q == b_pos) for q in _OUTFIELD}
            legal = np.ones(sim.n, dtype=bool)
            for q in _OUTFIELD:
                legal &= new[q] >= FORMATION_MIN[q]
            take = eligible & (remaining[p] > 0) & legal
            if take.any():
                step = take.astype(np.int64)
                remaining[p] = remaining[p] - step
                xi_counts[p] = xi_counts[p] - step
                xi_counts[b_pos] = xi_counts[b_pos] + step
                added += np.where(take, sim.points(code, gw), 0.0)
                eligible &= ~take
    return added


def _realized_gw_score(
    plan_gw: Any, sim: _SampledWindow, pos_of: Mapping[int, str]
) -> np.ndarray:
    """Realized FPL score of one GwPlan per rollout.

    XI sampled points + autosub recoveries + captain extra (vice fallback when the
    captain does not appear; nothing when both miss) − hit points. On BB weeks all 15
    squad members count and autosubs do not apply; on TC weeks the extra is doubled
    (3x vs the 1x already in the sum). WC/FH change nothing score-side.
    """
    gw = int(plan_gw.gw)
    chip = (plan_gw.chip or "")[:2]
    total = np.zeros(sim.n)
    if chip == "bb":
        for p in plan_gw.squad:
            total += sim.points(p, gw)
    else:
        for p in plan_gw.lineup:
            total += sim.points(p, gw)
        total += _autosub_points(gw, plan_gw.lineup, plan_gw.bench_order, sim, pos_of)
    cap, vice = int(plan_gw.captain), int(plan_gw.vice)
    extra = np.where(
        sim.appeared(cap, gw),
        sim.points(cap, gw),
        np.where(sim.appeared(vice, gw), sim.points(vice, gw), 0.0),
    )
    total += 2.0 * extra if chip == "tc" else extra
    total -= float(plan_gw.hit_points)
    return total


def _effective_captain_points(
    gw: int, captain: int, vice: int, sim: _SampledWindow
) -> np.ndarray:
    """The extra points one additional captain multiple earns (with vice fallback)."""
    return np.where(
        sim.appeared(captain, gw),
        sim.points(captain, gw),
        np.where(sim.appeared(vice, gw), sim.points(vice, gw), 0.0),
    )


# --------------------------------------------------------------------------------------
# Monte Carlo aggregation
# --------------------------------------------------------------------------------------


def _mc_stats(
    gains: dict[int, np.ndarray],
) -> dict[int, tuple[float, float, float, float]]:
    """Per candidate GW: ``(e_gain, sd, p_best_week, p_beats_hold)``.

    ``p_best_week`` splits exact ties evenly (the column sums to 1 over candidates);
    ``p_beats_hold`` uses the weak inequality against ``max(0, best later gain)`` —
    holding past the last candidate GW forfeits the chip (gain 0).
    """
    if not gains:
        return {}
    gws = sorted(gains)
    matrix = np.stack([np.asarray(gains[g], dtype=float) for g in gws])
    n = matrix.shape[1]
    e = matrix.mean(axis=1)
    sd = matrix.std(axis=1, ddof=1) if n > 1 else np.zeros(len(gws))
    best = matrix.max(axis=0)
    is_best = matrix == best[None, :]
    p_best = (is_best / is_best.sum(axis=0)[None, :]).sum(axis=1) / n
    out: dict[int, tuple[float, float, float, float]] = {}
    for i, g in enumerate(gws):
        later = matrix[i + 1 :]
        hold = np.maximum(later.max(axis=0), 0.0) if len(later) else np.zeros(n)
        p_beats = float((matrix[i] >= hold).mean())
        out[g] = (float(e[i]), float(sd[i]), float(p_best[i]), p_beats)
    return out


def _rank_joint_schedules(
    mc_gains: Mapping[str, Mapping[int, np.ndarray]],
    n_rollouts: int,
    *,
    top_k: int,
    per_chip: int,
) -> list[JointSchedule]:
    """Enumerate and rank joint chip schedules (≤ 1 chip per GW, windows already applied).

    Each chip contributes its ``per_chip`` best candidate GWs by E[gain] plus the "don't
    play" option; schedules placing two chips in one GW are excluded. E and P(best) are
    computed on summed per-rollout gains (see the independence assumption).
    """
    chips = sorted(c for c in mc_gains if mc_gains[c])
    if not chips:
        return []
    options: dict[str, list[int | None]] = {}
    for chip in chips:
        gains = mc_gains[chip]
        e = {g: float(np.mean(gains[g])) for g in gains}
        top = sorted(gains, key=lambda g: (-e[g], g))[:per_chip]
        options[chip] = [None, *sorted(top)]

    placements_list: list[dict[str, int]] = []
    totals: list[np.ndarray] = []
    for combo in product(*(options[c] for c in chips)):
        placed = [(c, g) for c, g in zip(chips, combo, strict=True) if g is not None]
        gws_used = [g for _c, g in placed]
        if len(set(gws_used)) != len(gws_used):
            continue  # two chips in one GW — illegal (≤ 1 chip per GW)
        total = np.zeros(n_rollouts)
        for c, g in placed:
            total = total + np.asarray(mc_gains[c][g], dtype=float)
        placements_list.append(dict(placed))
        totals.append(total)
    if not totals:
        return []
    matrix = np.stack(totals)
    e_total = matrix.mean(axis=1)
    best = matrix.max(axis=0)
    is_best = matrix == best[None, :]
    p_best = (is_best / is_best.sum(axis=0)[None, :]).sum(axis=1) / n_rollouts
    order = sorted(
        range(len(totals)),
        key=lambda i: (
            -e_total[i],
            -p_best[i],
            tuple(sorted(placements_list[i].items())),
        ),
    )
    return [
        JointSchedule(
            placements=placements_list[i],
            e_total_gain=float(e_total[i]),
            p_best=float(p_best[i]),
        )
        for i in order[:top_k]
    ]


# --------------------------------------------------------------------------------------
# Windows, candidates and segment states
# --------------------------------------------------------------------------------------


def _chip_window(season: int, chip_id: str) -> rules.ChipWindow:
    match = _CHIP_ID_RE.match(chip_id.lower())
    if match is None:
        raise ValueError(f"unrecognized chip id {chip_id!r} (expected e.g. 'wc1', 'bb2')")
    name, set_no = match.group(1).upper(), int(match.group(2))
    for window in rules.chip_windows(season):
        if window.name == name and window.set == set_no:
            return window
    raise ValueError(f"chip {chip_id!r} does not exist in season {season}")


def _window_gws(window: Any) -> list[int]:
    gws = [int(g) for g in window]
    if not gws:
        raise ValueError("empty simulation window")
    if gws != list(range(gws[0], gws[-1] + 1)):
        raise ValueError(f"window must be a contiguous ascending GW range, got {gws}")
    if gws[0] < 1 or gws[-1] > 38:
        raise ValueError(f"window {gws[0]}..{gws[-1]} outside GWs 1..38")
    return gws


def _candidate_gws(
    chip: str,
    window_gws: Sequence[int],
    season: int,
    state: SquadState | None,
) -> tuple[list[int], dict[int, str]]:
    """Candidate GWs for a chip within the window, plus skip reasons for the rest."""
    win = _chip_window(season, chip)
    candidates: list[int] = []
    skips: dict[int, str] = {}
    for g in window_gws:
        if not win.first_gw <= g <= win.last_gw:
            skips[g] = "outside_window"
        elif state is None and g == window_gws[0] and chip[:2] in ("wc", "fh"):
            skips[g] = "initial_build_first_gw"
        elif state is not None and state.active_chip is not None and g == window_gws[0]:
            skips[g] = "active_chip_conflict"  # ≤ 1 chip per GW; one is already committed
        else:
            candidates.append(g)
    return candidates, skips


def _segment_state(
    g: int,
    chip: str | None,
    *,
    window_start: int,
    season: int,
    state: SquadState | None,
    plan_by_gw: Mapping[int, Any],
    price_of: Mapping[int, int],
) -> SquadState:
    """The squad state entering GW ``g``, fixed from the backbone (or the real state).

    Synthetic states value every player at current price (purchase == current — the
    static_prices assumption). If a Free Hit is *active* at the window start, the squad
    entering ``start + 1`` is the real (reverted) squad, not the backbone's FH squad.
    """
    if g == window_start:
        if state is None:
            raise InfeasiblePlanError(
                "cannot re-solve the initial-build GW as a segment", ["state is None"]
            )
        return SquadState(
            season=season,
            current_gw=g,
            squad=state.squad,
            bank=state.bank,
            free_transfers=state.free_transfers,
            chips_available=[chip] if chip in set(state.chips_available) else [],
            active_chip=state.active_chip,
        )
    prev = plan_by_gw[g - 1]
    codes = list(prev.squad)
    if (
        state is not None
        and state.active_chip is not None
        and state.active_chip[:2] == "fh"
        and g - 1 == window_start
    ):
        codes = [op.player_code for op in state.squad]  # the FH squad reverted
    cur_ft = plan_by_gw[g].free_transfers
    return SquadState(
        season=season,
        current_gw=g,
        squad=[
            OwnedPlayer(player_code=c, purchase_price=price_of[c], current_price=price_of[c])
            for c in codes
        ],
        bank=int(prev.bank),
        free_transfers=int(cur_ft) if cur_ft is not None else 1,
        chips_available=[chip] if chip is not None else [],
        active_chip=None,
    )


# --------------------------------------------------------------------------------------
# The simulator
# --------------------------------------------------------------------------------------


def simulate_chip_plans(  # noqa: PLR0915 - the orchestration is deliberately one story
    xp: pd.DataFrame,
    prices: pd.DataFrame,
    state: SquadState | None,
    *,
    window: range,
    n_rollouts: int = 1000,
    seed: int = 0,
    params: SolveParams | None = None,
    sampler: Any | None = None,
    solve_fn: SolvePlanFn | None = None,
    chips: Sequence[str] | None = None,
    backbone_time_limit_s: float = BACKBONE_TIME_LIMIT_S,
    segment_time_limit_s: float = SEGMENT_TIME_LIMIT_S,
    segment_len: int = 4,
    top_k: int = 10,
    joint_gws_per_chip: int = 4,
) -> ChipSimReport:
    """Monte Carlo chip-timing simulation over a full chip window (stage-6 contract).

    Args:
        xp: Predictions per the ``solve_plan`` contract — ``season, gw, player_code,
            xp`` (+ optional ``q0``); a per-fixture frame (e.g. predictions.parquet)
            works too and is what the sampler wants (fixture-level correlation).
            Must cover every GW of ``window``.
        prices: ``player_code, price, position, team_code`` (one row per player).
        state: The manager's :class:`SquadState`, or ``None`` for initial-build mode.
        window: Contiguous candidate GWs, e.g. ``range(1, 20)`` = GWs 1..19 (the set-1
            chip window). Must start at ``state.current_gw`` when a state is given.
        n_rollouts: Monte Carlo rollouts (default 1000).
        seed: Drives the sampler (and is forwarded to the solver params unchanged).
        params: Base :class:`SolveParams`; backbone/segment overrides are applied on
            copies (chips banned/forced, time caps, aggressive pruning).
        sampler: A ``PointsSampler``-shaped object (``.sample(pred, n=, seed=) ->
            ndarray[n, rows]``); defaults to the real ``models.sampler.PointsSampler``.
        solve_fn: ``solve_plan``-shaped callable (test seam); defaults to the real MILP.
        chips: Chip ids to simulate; default = every season chip whose window overlaps
            ``window`` (chips a state has already used are reported ``not_available``).
        backbone_time_limit_s / segment_time_limit_s: Solver caps per layer.
        segment_len: WC/FH re-solve span ``[g, min(g + segment_len - 1, window_end)]``.
        top_k: Joint schedules to report.
        joint_gws_per_chip: Candidate GWs per chip entering the joint enumeration.

    Returns:
        A :class:`ChipSimReport`; ``.to_frame()`` yields the chip_curves-v2 frame.

    Raises:
        ValueError: on malformed inputs, a window/state mismatch, or missing GW
            coverage in ``xp`` (run ``fplai predict --through-gw`` first).
    """
    t0 = time.perf_counter()
    if n_rollouts < 1:
        raise ValueError(f"n_rollouts must be >= 1, got {n_rollouts}")
    window_gws = _window_gws(window)
    horizon = len(window_gws)

    required = {"player_code", "price", "position", "team_code"}
    missing_cols = required - set(prices.columns)
    if missing_cols:
        raise ValueError(f"prices frame missing columns: {sorted(missing_cols)}")
    price_of = dict(
        zip(prices["player_code"].astype(int), prices["price"].astype(int), strict=True)
    )
    pos_of = dict(
        zip(prices["player_code"].astype(int), prices["position"].astype(str), strict=True)
    )

    if state is not None:
        season = int(state.season)
        if int(state.current_gw) != window_gws[0]:
            raise ValueError(
                f"window starts at GW{window_gws[0]} but state.current_gw is "
                f"{state.current_gw}; the simulation must start at the next deadline"
            )
    else:
        seasons = sorted(set(xp["season"].astype(int)))
        if len(seasons) != 1:
            raise ValueError(f"xp frame must cover exactly one season, got {seasons}")
        season = seasons[0]

    xp_win = xp[(xp["season"] == season) & xp["gw"].isin(window_gws)].copy()
    covered = {int(g) for g in xp_win["gw"].unique()}
    missing_gws = [g for g in window_gws if g not in covered]
    if missing_gws:
        raise ValueError(
            f"xp frame has no rows for GWs {missing_gws} of the window — extend "
            "predictions first (fplai predict --through-gw)"
        )

    solver = resolve_solver(solve_fn)
    base_params = params if params is not None else SolveParams()
    all_ids = season_chip_ids(season)
    if chips is None:
        chips = [
            cid
            for cid in all_ids
            if _chip_window(season, cid).first_gw <= window_gws[-1]
            and _chip_window(season, cid).last_gw >= window_gws[0]
        ]
    else:
        unknown = [c for c in chips if _CHIP_ID_RE.match(str(c).lower()) is None]
        if unknown:
            raise ValueError(f"unknown chip ids: {unknown}")
        chips = [str(c).lower() for c in chips]
    available = set(state.chips_available) if state is not None else set(all_ids)

    # -- layer 1: the no-chip backbone over the full window ----------------------------
    backbone_params = copy_params_with(
        base_params,
        **{
            FORCED_CHIPS_FIELD: {},
            BANNED_CHIPS_FIELD: frozenset(all_ids),
            "time_limit_s": float(backbone_time_limit_s),
            "mip_rel_gap": BACKBONE_MIP_REL_GAP,
            **BACKBONE_PRUNE,
        },
    )
    backbone = solver(xp_win, prices, state, horizon=horizon, params=backbone_params)
    plan_by_gw = {int(gp.gw): gp for gp in backbone.gws}
    backbone_quality = BackboneQuality(
        objective=float(backbone.objective),
        gap=float(getattr(backbone, "gap", float("nan"))),
        solve_seconds=float(getattr(backbone, "solve_seconds", float("nan"))),
        status=str(getattr(backbone, "status", "unknown")),
        horizon=horizon,
        pool_hint=(
            f"prune_keep_frac={BACKBONE_PRUNE['prune_keep_frac']}, "
            f"time_cap={backbone_time_limit_s:.0f}s"
        ),
    )
    logger.info(
        "backbone: %d GWs, objective=%.2f gap=%.3g in %.1fs (%s)",
        horizon,
        backbone_quality.objective,
        backbone_quality.gap,
        backbone_quality.solve_seconds,
        backbone_quality.status,
    )

    # -- layer 3 prep: one sampler pass for the whole window ---------------------------
    sort_cols = ["gw", "player_code"] + (
        ["fpl_fixture_id"] if "fpl_fixture_id" in xp_win.columns else []
    )
    pred = xp_win.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
    if sampler is None:
        sampler = _default_sampler()
    points, appeared, appearance_source = _draw_samples(sampler, pred, n_rollouts, seed)
    sim = _SampledWindow(pred, points, appeared)
    xp_lookup = {
        (int(c), int(g)): float(v)
        for (c, g), v in pred.groupby(["player_code", "gw"], sort=False)["xp"].sum().items()
    }

    # -- layer 2 + 3: candidate placements and their rollout gains ---------------------
    seg_updates_base = {
        "time_limit_s": float(segment_time_limit_s),
        "mip_rel_gap": SEGMENT_MIP_REL_GAP,
        **SEGMENT_PRUNE,
    }
    seg_baseline_cache: dict[int, Any] = {}
    seg_baseline_realized: dict[int, np.ndarray] = {}
    n_segment_solves = 0

    def _solve_segment(g: int, chip: str | None) -> Any:
        nonlocal n_segment_solves
        seg_end = min(g + segment_len - 1, window_gws[-1])
        seg_state = _segment_state(
            g,
            chip,
            window_start=window_gws[0],
            season=season,
            state=state,
            plan_by_gw=plan_by_gw,
            price_of=price_of,
        )
        updates: dict[str, Any] = dict(seg_updates_base)
        if chip is None:
            updates[FORCED_CHIPS_FIELD] = {}
            updates[BANNED_CHIPS_FIELD] = frozenset(all_ids)
        else:
            updates[FORCED_CHIPS_FIELD] = {g: chip}
            updates[BANNED_CHIPS_FIELD] = frozenset(set(all_ids) - {chip})
        seg_params = copy_params_with(base_params, **updates)
        n_segment_solves += 1
        return solver(xp_win, prices, seg_state, horizon=seg_end - g + 1, params=seg_params)

    stats: list[ChipGwStat] = []
    verdicts: list[ChipVerdict] = []
    mc_gains: dict[str, dict[int, np.ndarray]] = {}

    for chip in chips:
        win = _chip_window(season, chip)
        if chip not in available:
            stats.extend(
                ChipGwStat(
                    chip=chip, gw=g, skip_reason="not_available", window_last_gw=win.last_gw
                )
                for g in window_gws
            )
            verdicts.append(ChipVerdict(chip=chip, recommended_gw=None))
            mc_gains[chip] = {}
            continue

        candidates, skips = _candidate_gws(chip, window_gws, season, state)
        kind = chip[:2]
        det_gain: dict[int, float] = {}
        gains: dict[int, np.ndarray] = {}

        if kind == "bb":
            for g in candidates:
                gp = plan_by_gw.get(g)
                if gp is None or len(gp.bench_order) < 4:
                    skips[g] = "no_backbone_bench"
                    continue
                det_gain[g] = sum(xp_lookup.get((int(p), g), 0.0) for p in gp.bench_order)
                bench_pts = np.zeros(n_rollouts)
                for p in gp.bench_order:
                    bench_pts = bench_pts + sim.points(int(p), g)
                gains[g] = bench_pts
        elif kind == "tc":
            for g in candidates:
                gp = plan_by_gw.get(g)
                if gp is None:
                    skips[g] = "no_backbone_gw"
                    continue
                det_gain[g] = xp_lookup.get((int(gp.captain), g), 0.0)
                gains[g] = _effective_captain_points(g, int(gp.captain), int(gp.vice), sim)
        else:  # wc / fh: bounded segment re-solves against a no-chip segment baseline
            for g in candidates:
                try:
                    if g not in seg_baseline_cache:
                        seg_baseline_cache[g] = _solve_segment(g, None)
                    base_plan = seg_baseline_cache[g]
                    forced_plan = _solve_segment(g, chip)
                except (InfeasiblePlanError, RuntimeError) as exc:
                    skips[g] = f"solver: {str(exc).splitlines()[0][:120]}"
                    continue
                det_gain[g] = float(forced_plan.objective) - float(base_plan.objective)
                if g not in seg_baseline_realized:
                    realized = np.zeros(n_rollouts)
                    for gp in base_plan.gws:
                        realized = realized + _realized_gw_score(gp, sim, pos_of)
                    seg_baseline_realized[g] = realized
                delta = -seg_baseline_realized[g]
                for gp in forced_plan.gws:
                    delta = delta + _realized_gw_score(gp, sim, pos_of)
                gains[g] = delta

        mc_gains[chip] = gains
        stat_by_gw = _mc_stats(gains)
        for g in window_gws:
            if g in stat_by_gw:
                e, sd, p_best, p_beats = stat_by_gw[g]
                stats.append(
                    ChipGwStat(
                        chip=chip,
                        gw=g,
                        xp_gain=det_gain.get(g),
                        e_gain=e,
                        sd=sd,
                        p_best_week=p_best,
                        p_beats_hold=p_beats,
                        evaluated=True,
                        window_last_gw=win.last_gw,
                    )
                )
            else:
                stats.append(
                    ChipGwStat(
                        chip=chip,
                        gw=g,
                        skip_reason=skips.get(g, "not_evaluated"),
                        window_last_gw=win.last_gw,
                    )
                )
        if stat_by_gw:
            ordered = sorted(stat_by_gw)
            e_vals = [stat_by_gw[g][0] for g in ordered]
            rec = ordered[int(np.argmax(e_vals))]
            e, sd, p_best, p_beats = stat_by_gw[rec]
            verdicts.append(
                ChipVerdict(
                    chip=chip,
                    recommended_gw=rec,
                    e_gain=e,
                    sd=sd,
                    confidence=p_best,
                    p_beats_hold=p_beats,
                )
            )
        else:
            verdicts.append(ChipVerdict(chip=chip, recommended_gw=None))

    schedules = _rank_joint_schedules(
        mc_gains, n_rollouts, top_k=top_k, per_chip=joint_gws_per_chip
    )
    elapsed = time.perf_counter() - t0
    logger.info(
        "simulate_chip_plans: %d chips x GWs %d-%d, %d rollouts, %d segment solves in %.1fs",
        len(chips),
        window_gws[0],
        window_gws[-1],
        n_rollouts,
        n_segment_solves,
        elapsed,
    )
    return ChipSimReport(
        season=season,
        window_first_gw=window_gws[0],
        window_last_gw=window_gws[-1],
        n_rollouts=n_rollouts,
        seed=seed,
        chips=list(chips),
        stats=stats,
        verdicts=verdicts,
        schedules=schedules,
        backbone=backbone_quality,
        assumptions=list(ASSUMPTIONS),
        appearance_source=appearance_source,
        n_segment_solves=n_segment_solves,
        sim_seconds=elapsed,
    )
