"""Walk-forward policy backtester: replay a season GW by GW (MODEL_DESIGN_INPUTS §7).

For each backtested GW ``t`` the harness (a) obtains xP predictions for GWs ``t..t+H-1``
from model artifacts, (b) solves the weekly plan with the stage-3 MILP
(:func:`fplai.optimizer.milp.solve_plan`) against the rolled-forward squad state, (c)
applies REALIZED points from ``player_gw`` to the chosen XI/captain/bench with the real
autosub + vice-captain rules (FPL_KNOWLEDGE §1.5/§1.4), subtracts hits, and (d) rolls all
state forward: squad, bank with sell-price arithmetic (:func:`fplai.rules.sell_price`) on
realized price movements (``player_gw.value``), the FT bank
(:func:`fplai.rules.next_free_transfers`), and the chip inventory.

Prediction modes (the §7.1 walk-forward requirement, labelled on the result)
---------------------------------------------------------------------------
* **coarse-pretrained** (``retrain_every=None``, the default): ONE artifact set (from
  ``BacktestParams.models_dir``, default ``config.MODELS_DIR``) is reused for every GW.
  Features per target GW still only use matches strictly before that GW (enforced by
  ``features/windows.py``), but the artifacts must have been trained strictly before the
  backtested window for the season points to be honest — the harness reads the artifact
  ``manifest.json`` and emits a loud ``leakage_warnings`` entry when the train window
  overlaps the backtest (the §7.2 trap 5 "own-model feedback" check).
* **retrain-every-N** (``retrain_every=N``): artifacts are retrained with
  ``pipeline.run_train(before_season=season, before_gw=block_start)`` at every N-GW block
  boundary (cached under ``<models_dir>/backtest/``; SLOW — ~70 s per retrain). All
  predictions consumed by decisions inside a block come from that block's artifacts, so
  every decision at GW ``t`` uses artifacts trained strictly before ``t``.

Remaining leakage caveats (documented, and repeated in ``BacktestResult.leakage_warnings``)
-------------------------------------------------------------------------------------------
* **Horizon look-ahead**: within a decision at GW ``t``, the xP rows for GWs
  ``t+1..t+H-1`` are built from matches played before *their own* GW, not before ``t``
  (per-target-GW causal, not per-decision causal). The points banked at ``t`` come from
  strictly pre-``t`` predictions; only the multi-GW *planning* input peeks. A strict
  per-decision rebuild would multiply prediction cost ~H-fold and is left to a follow-up.
* **Closing odds**: the odds blend uses closing odds that can post-date a GW deadline
  (STATUS.md caveat 4), so ``BacktestParams.use_odds`` defaults to ``False`` here.

Chip policy (light forced sweep, per the stage-4 brief)
-------------------------------------------------------
Every weekly base solve runs with ``no_chips=True`` (small model, fast). A chip is only
*evaluated* when a cheap heuristic trigger fires, and then via at most
``max_chip_solves_per_gw`` (default 2) extra forced-chip-now solves compared against the
base objective; the chip is played when the delta exceeds ``chip_play_margin``. Triggers:

* **BB**: base plan's bench xP this GW >= ``bb_bench_ev_threshold``, or a DGW this GW
  (any ``n_fixtures >= 2`` row), or the instance is within ``chip_expiry_within`` GWs of
  its window end.
* **TC**: top xP this GW >= ``tc_captain_xp_threshold``, or DGW this GW, or near expiry.
* **WC**: base plan takes >= ``wc_hit_points_trigger`` hit points this GW, or near expiry.
* **FH**: >= ``fh_blank_squad_threshold`` owned players without an xP row this GW (blank
  GW proxy), or near expiry.

This deliberately trades a little chip-timing optimality for a ~2-solves-per-GW budget
(the full ``optimizer.chips.chip_ev_curves`` sweep is the accurate-but-slow alternative).
Because base solves ban chips, the solver never *plans ahead* for a chip week — a further
documented approximation of the light version.

Baselines (run in the same replay loop) and reference constants (§7.3 metric 3)
-------------------------------------------------------------------------------
* ``last5``: last-5-average realized points as xP (the §6 "Last-5 baseline"), same solver,
  same chip policy (DGW triggers disabled — the frame has no fixture counts).
* ``set_and_forget``: initial None-state solve, then no transfers/chips ever; the XI and
  bench stay fixed and only the captain re-picks each GW (highest model xP in the XI);
  real autosubs still apply.
* Reference constants :data:`AVERAGE_MANAGER_POINTS` / :data:`TOP_10K_POINTS` (sources in
  their comments) are attached to every result for context.

Performance: per-GW no-chips solves with tight settings (default 15 s limit, 1% gap,
default §3.5 pool pruning, horizon 5) — a 38-GW season replay (model + both baselines)
lands in roughly the 10-20 minute bracket on a laptop. Use ``gws=range(a, b)`` subsets
for anything iterative. Everything is deterministic for fixed seeds/params.
"""

from __future__ import annotations

import json
import logging
import time
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from fplai import rules
from fplai.optimizer.autosubs import FORMATION_MIN
from fplai.optimizer.milp import (
    INITIAL_BUDGET,
    InfeasiblePlanError,
    PlanResult,
    SolveParams,
    solve_plan,
)
from fplai.optimizer.state import OwnedPlayer, SquadState, all_chip_ids, chip_instance_for

logger = logging.getLogger(__name__)

__all__ = [
    "AVERAGE_MANAGER_POINTS",
    "TOP_10K_POINTS",
    "BacktestParams",
    "BacktestResult",
    "GwLedger",
    "PolicyLedger",
    "ScoredGw",
    "apply_autosubs",
    "compute_xp_metrics",
    "last5_xp_frame",
    "replay",
    "run",
    "run_backtest",
    "score_gw",
]

# --------------------------------------------------------------------------------------
# Reference constants (§7.3 metric 3 — decision quality anchors)
# --------------------------------------------------------------------------------------

#: MODEL_DESIGN_INPUTS §7.3: "average human ≈ 2,200-2,300 pts" per season. Midpoint.
AVERAGE_MANAGER_POINTS: int = 2250

#: MODEL_DESIGN_INPUTS §7.3: "top-10k ≈ 2,550-2,700 (varies by season)". Midpoint.
TOP_10K_POINTS: int = 2625

#: MODEL_DESIGN_INPUTS §7.3: hindsight optimum ≈ 4,984 — AlpsCode's perfect-foresight
#: solve of 2019-20 (alpscode.com hindsight-optimization post). Unreachable variance.
HINDSIGHT_OPTIMUM_POINTS_2019: int = 4984

_CHIP_KINDS: tuple[str, ...] = ("wc", "fh", "bb", "tc")
_OUTFIELD_ORDER: tuple[str, ...] = ("DEF", "MID", "FWD")  # team-sheet order tie-break
_LAST_GW: int = 38

#: Return categories for xP-accuracy metrics (§6/§7.3): name -> (lo, hi) realized points.
RETURN_CATEGORIES: dict[str, tuple[int, int]] = {
    "zeros": (0, 0),
    "blanks": (1, 2),
    "tickers": (3, 4),
    "haulers": (5, 10**9),
}


# --------------------------------------------------------------------------------------
# Parameters and result models
# --------------------------------------------------------------------------------------


class BacktestParams(BaseModel):
    """Policy + engineering knobs of the walk-forward backtest (module docstring)."""

    model_config = ConfigDict(frozen=True)

    horizon: int = Field(default=5, ge=1, le=15, description="Planning horizon per solve")
    time_limit_s: float = Field(default=15.0, gt=0.0, description="Per-solve MILP limit")
    mip_rel_gap: float = Field(default=0.01, ge=0.0, description="Per-solve MIP gap")
    seed: int = Field(default=0, ge=0)
    use_odds: bool = Field(
        default=False,
        description="Blend closing odds into team predictions (leaky pre-2026; see docstring)",
    )
    models_dir: Path | None = Field(
        default=None, description="Artifact root (default config.MODELS_DIR)"
    )
    solver_overrides: dict[str, Any] = Field(
        default_factory=dict, description="Extra SolveParams fields layered on the defaults"
    )
    # chip policy (light forced sweep)
    evaluate_chips: bool = True
    max_chip_solves_per_gw: int = Field(default=2, ge=0)
    chip_play_margin: float = Field(default=1.5, ge=0.0)
    bb_bench_ev_threshold: float = 10.0
    tc_captain_xp_threshold: float = 8.0
    wc_hit_points_trigger: int = 8
    fh_blank_squad_threshold: int = 4
    chip_expiry_within: int = Field(default=3, ge=0)
    # baselines
    run_baselines: bool = True
    last5_window: int = Field(default=5, ge=1)


class GwLedger(BaseModel):
    """One gameweek of a replayed policy. Player fields are ``player_code`` values."""

    gw: int
    points: int = Field(description="Net banked points (gross - hit_points)")
    gross_points: int = Field(description="XI + captain (+chip) points before hits")
    bench_points: int = Field(description="Realized points left on the bench (0 on BB)")
    hit_points: int
    n_transfers: int
    transfers_in: list[int] = []
    transfers_out: list[int] = []
    chip: str | None = None
    bank: int = Field(description="After this GW's transfers, £0.1m units")
    squad_value: int = Field(description="Owned squad at current prices, £0.1m units")
    team_value: int = Field(description="bank + owned squad at SELL prices, £0.1m units")
    free_transfers: int | None = Field(
        description="FTs going into this GW; None = unlimited (initial build)"
    )
    captain: int | None = None
    vice: int | None = None
    effective_captain: int | None = Field(
        default=None, description="Who the multiplier landed on (None = neither played)"
    )
    captain_points: int = Field(default=0, description="Effective captain pts x multiplier")
    captain_success: bool = Field(
        default=False, description="Effective captain was the (joint-)top scorer of the XI"
    )
    autosubs: list[tuple[int, int]] = Field(
        default_factory=list, description="(starter out, bench in) pairs applied at lockdown"
    )
    lineup: list[int] = Field(
        default_factory=list,
        description="Players whose points counted: final XI after autosubs (all 15 on BB)",
    )
    squad: list[int] = Field(
        default_factory=list, description="Owned (permanent) squad after transfers"
    )
    expected_points: float = Field(default=0.0, description="Solver xP for this GW")
    solve_seconds: float = 0.0


class PolicyLedger(BaseModel):
    """A replayed policy: per-GW ledger rows plus season totals."""

    name: str
    description: str = ""
    rows: list[GwLedger] = []
    total_points: int = 0
    total_gross_points: int = 0
    total_hit_points: int = 0
    total_transfers: int = 0
    chips_played: list[str] = []


class BacktestResult(BaseModel):
    """Season replay output: policies, references, xP-accuracy metrics, provenance."""

    season: int
    gws: list[int]
    mode: str = Field(description='"coarse-pretrained" | "retrain-every-N" | "injected"')
    params: dict[str, Any] = {}
    policies: dict[str, PolicyLedger] = {}
    references: dict[str, int] = {}
    metrics: dict[str, Any] = {}
    leakage_warnings: list[str] = []
    artifact_train_window: dict[str, Any] | None = None

    def totals(self) -> dict[str, int]:
        """Season points per policy."""
        return {name: ledger.total_points for name, ledger in self.policies.items()}

    def to_frame(self) -> pd.DataFrame:
        """Scalar ledger columns as one tidy frame (one row per policy per GW)."""
        records: list[dict[str, Any]] = []
        for name, ledger in self.policies.items():
            for row in ledger.rows:
                records.append(
                    {
                        "season": self.season,
                        "policy": name,
                        "gw": row.gw,
                        "points": row.points,
                        "gross_points": row.gross_points,
                        "bench_points": row.bench_points,
                        "hit_points": row.hit_points,
                        "n_transfers": row.n_transfers,
                        "chip": row.chip,
                        "bank": row.bank,
                        "squad_value": row.squad_value,
                        "team_value": row.team_value,
                        "free_transfers": row.free_transfers,
                        "captain": row.captain,
                        "captain_points": row.captain_points,
                        "captain_success": row.captain_success,
                        "n_autosubs": len(row.autosubs),
                        "expected_points": row.expected_points,
                        "solve_seconds": row.solve_seconds,
                    }
                )
        return pd.DataFrame.from_records(records)

    def save(self, out_dir: Path) -> dict[str, Path]:
        """Write ``result.json`` (full detail) + ``ledger.parquet`` (tidy frame)."""
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        paths = {"result": out / "result.json", "ledger": out / "ledger.parquet"}
        paths["result"].write_text(self.model_dump_json(indent=1))
        self.to_frame().to_parquet(paths["ledger"], index=False)
        return paths


# --------------------------------------------------------------------------------------
# Realized-data access (points, minutes, deadline prices)
# --------------------------------------------------------------------------------------

_REALIZED_COLUMNS = ("gw", "player_code", "minutes", "total_points", "value", "position",
                     "team_code")


class _Realized:
    """Per-GW realized outcomes and deadline-time prices for one season's ``player_gw``."""

    def __init__(self, season_pg: pd.DataFrame, season: int) -> None:
        missing = set(_REALIZED_COLUMNS) - set(season_pg.columns)
        if missing:
            raise ValueError(f"player_gw is missing columns: {sorted(missing)}")
        self.season = season
        self._df = season_pg.sort_values(["gw", "player_code"], kind="stable").reset_index(
            drop=True
        )
        self._ctx_cache: dict[int, dict[int, tuple[int, str, int]]] = {}
        self._pts_cache: dict[int, dict[int, int]] = {}
        self._min_cache: dict[int, dict[int, int]] = {}

    def context_at(self, t: int) -> dict[int, tuple[int, str, int]]:
        """``player_code -> (value, position, team_code)`` from the latest row with gw <= t.

        ``player_gw.value`` is the price snapshot for that GW (known at its deadline), so
        using the row AT ``t`` for decisions at ``t`` is deadline-safe (§7.2 trap 1).
        """
        if t not in self._ctx_cache:
            sub = self._df[self._df["gw"] <= t].groupby("player_code", sort=False).tail(1)
            self._ctx_cache[t] = {
                int(r.player_code): (int(r.value), str(r.position), int(r.team_code))
                for r in sub.itertuples()
            }
        return self._ctx_cache[t]

    def value_at(self, code: int, t: int) -> int | None:
        """Last known price of ``code`` at GW ``t`` (0.1m units), or None if never seen."""
        entry = self.context_at(t).get(code)
        return entry[0] if entry is not None else None

    def gw_points(self, t: int) -> dict[int, int]:
        """Realized FPL points per player for GW ``t`` (players absent -> not in the map)."""
        if t not in self._pts_cache:
            sub = self._df[self._df["gw"] == t]
            self._pts_cache[t] = {
                int(p): int(v) for p, v in zip(sub["player_code"], sub["total_points"])
            }
        return self._pts_cache[t]

    def gw_minutes(self, t: int) -> dict[int, int]:
        """Realized minutes per player for GW ``t``."""
        if t not in self._min_cache:
            sub = self._df[self._df["gw"] == t]
            self._min_cache[t] = {
                int(p): int(v) for p, v in zip(sub["player_code"], sub["minutes"])
            }
        return self._min_cache[t]


def _prices_frame(
    realized: _Realized,
    t: int,
    universe: set[int],
    fallback: pd.DataFrame | None,
    owned_meta: Mapping[int, tuple[int, str, int]],
) -> pd.DataFrame:
    """Solver prices frame at GW ``t``: ``player_code, price, position, team_code``.

    Prices come from the latest ``player_gw.value`` at or before ``t``; players without a
    realized row yet fall back to the predictions frame's price/position/team columns
    (when present), then to ``owned_meta`` (price/position/club captured when a squad
    member was bought) so owned players can always be valued.
    """
    ctx = realized.context_at(t)
    rows: list[dict[str, int | str]] = []
    missing: list[int] = []
    for code in sorted(universe):
        entry = ctx.get(code)
        if entry is not None:
            value, pos, club = entry
            rows.append({"player_code": code, "price": value, "position": pos,
                         "team_code": club})
        else:
            missing.append(code)
    fb_cols = {"player_code", "price", "position", "team_code"}
    if missing and fallback is not None and fb_cols <= set(fallback.columns):
        fb = fallback[fallback["player_code"].isin(missing)].dropna(subset=["price"])
        fb = fb.sort_values("gw").groupby("player_code", sort=False).head(1)
        for r in fb.itertuples():
            rows.append(
                {"player_code": int(r.player_code), "price": int(r.price),
                 "position": str(r.position), "team_code": int(r.team_code)}
            )
            missing.remove(int(r.player_code))
    for code in list(missing):
        if code in owned_meta:
            price, pos, club = owned_meta[code]
            rows.append({"player_code": code, "price": price, "position": pos,
                         "team_code": club})
            missing.remove(code)
    if missing:
        logger.warning("GW%d: %d unpriceable players dropped from the pool", t, len(missing))
    return pd.DataFrame(rows, columns=["player_code", "price", "position", "team_code"])


# --------------------------------------------------------------------------------------
# Realized scoring: autosubs (FPL_KNOWLEDGE §1.5), captaincy (§1.4), chips
# --------------------------------------------------------------------------------------


def apply_autosubs(
    lineup: Sequence[int],
    bench_order: Sequence[int],
    positions: Mapping[int, str],
    minutes: Mapping[int, int],
) -> tuple[list[int], list[tuple[int, int]]]:
    """Deterministic lockdown autosubs per FPL_KNOWLEDGE §1.5.

    * A starter with 0 minutes in the GW is replaced by the highest-priority bench player
      whose entry keeps the formation legal (GKP exactly 1, DEF >= 3, MID >= 2, FWD >= 1).
    * The starting GK is only ever replaced by the bench GK (and vice versa).
    * Players who played but scored <= 0 points are NOT substituted.
    * Position-order tie-break (DEF -> MID -> FWD, then lineup order) mirrors
      ``optimizer.autosubs`` — team-sheet order at position granularity.

    Returns ``(final_xi, [(starter_out, bench_in), ...])``. ``bench_order`` is slot order
    (reserve GK first); an empty bench (BB week) returns the lineup unchanged.
    """
    xi = list(lineup)
    subs: list[tuple[int, int]] = []
    if not bench_order:
        return xi, subs

    bench_gks = [c for c in bench_order if positions[c] == "GKP"]
    outfield_bench = [c for c in bench_order if positions[c] != "GKP"]

    start_gk = next((c for c in xi if positions[c] == "GKP"), None)
    if start_gk is not None and minutes.get(start_gk, 0) == 0 and bench_gks:
        bench_gk = bench_gks[0]
        if minutes.get(bench_gk, 0) > 0:
            xi[xi.index(start_gk)] = bench_gk
            subs.append((start_gk, bench_gk))

    for bench_player in outfield_bench:
        if minutes.get(bench_player, 0) == 0:
            continue
        counts = Counter(positions[c] for c in xi)
        b_pos = positions[bench_player]
        replaced = False
        for pos in _OUTFIELD_ORDER:
            if replaced:
                break
            new_counts = dict(counts)
            new_counts[pos] = new_counts.get(pos, 0) - 1
            new_counts[b_pos] = new_counts.get(b_pos, 0) + 1
            if any(new_counts.get(p, 0) < FORMATION_MIN[p] for p in _OUTFIELD_ORDER):
                continue
            for starter in [c for c in lineup if c in xi and positions[c] == pos]:
                if minutes.get(starter, 0) != 0:
                    continue
                xi[xi.index(starter)] = bench_player
                subs.append((starter, bench_player))
                replaced = True
                break
    return xi, subs


@dataclass(frozen=True, slots=True)
class ScoredGw:
    """Realized outcome of one GW's picks after autosubs/captaincy/chip effects."""

    gross_points: int
    bench_points: int
    final_lineup: list[int]
    autosubs: list[tuple[int, int]]
    effective_captain: int | None
    captain_points: int
    captain_success: bool


def score_gw(
    *,
    squad: Sequence[int],
    lineup: Sequence[int],
    bench_order: Sequence[int],
    captain: int,
    vice: int | None,
    chip: str | None,
    points: Mapping[int, int],
    minutes: Mapping[int, int],
    positions: Mapping[int, str],
) -> ScoredGw:
    """Score one GW's picks with realized points under the real FPL rules.

    ``chip`` is a chip instance id (``"bb1"``) or None. Bench Boost counts all 15 of
    ``squad`` (no autosubs); Triple Captain applies x3 (falling to the vice on the
    captain's non-appearance, like the regular x2 — FPL_KNOWLEDGE §1.4). Players missing
    from ``points``/``minutes`` (no fixture / postponed) count as 0-minute non-appearers.
    """
    kind = (chip or "")[:2]
    if kind == "bb":
        counted: list[int] = list(dict.fromkeys(squad))
        final_xi, subs = counted, []  # all 15 count on BB weeks (matches milp's 15-man lineup)
    else:
        final_xi, subs = apply_autosubs(lineup, bench_order, positions, minutes)
        counted = final_xi

    base = sum(points.get(c, 0) for c in counted)
    multiplier = 3 if kind == "tc" else 2
    if minutes.get(captain, 0) > 0:
        effective = captain
    elif vice is not None and minutes.get(vice, 0) > 0:
        effective = vice
    else:
        effective = None
    extra = points.get(effective, 0) * (multiplier - 1) if effective is not None else 0

    bench_points = (
        0 if kind == "bb" else sum(points.get(b, 0) for b in bench_order if b not in final_xi)
    )
    top_xi = max((points.get(c, 0) for c in final_xi), default=0)
    return ScoredGw(
        gross_points=base + extra,
        bench_points=bench_points,
        final_lineup=final_xi,
        autosubs=subs,
        effective_captain=effective,
        captain_points=(points.get(effective, 0) * multiplier) if effective is not None else 0,
        captain_success=effective is not None and points.get(effective, 0) >= top_xi,
    )


def _apply_transfers(
    squad: Sequence[OwnedPlayer],
    bank: int,
    transfers_in: Sequence[int],
    transfers_out: Sequence[int],
    value_of: Callable[[int], int | None],
) -> tuple[list[OwnedPlayer], int]:
    """Apply permanent transfers: sells credit the §1.8 sell price, buys debit full price."""
    new_squad = list(squad)
    for code in transfers_out:
        owned = next((p for p in new_squad if p.player_code == code), None)
        if owned is None:
            raise RuntimeError(f"solver sold player {code} who is not owned")
        new_squad.remove(owned)
        now = value_of(code)
        bank += rules.sell_price(owned.purchase_price, now if now is not None
                                 else owned.current_price)
    for code in transfers_in:
        price = value_of(code)
        if price is None:
            raise RuntimeError(f"transfer-in {code} has no known price")
        bank -= price
        new_squad.append(
            OwnedPlayer(player_code=code, purchase_price=price, current_price=price)
        )
    if bank < 0:
        raise RuntimeError(f"bank went negative ({bank}) applying transfers — price mismatch")
    return new_squad, bank


# --------------------------------------------------------------------------------------
# Solver-parameter and chip-decision helpers
# --------------------------------------------------------------------------------------


def _solver_params(params: BacktestParams, **updates: Any) -> SolveParams:
    """Tight per-GW SolveParams (§3.5 engineering defaults for the replay loop)."""
    base: dict[str, Any] = {
        "time_limit_s": params.time_limit_s,
        "mip_rel_gap": params.mip_rel_gap,
        "seed": params.seed,
        "no_chips": True,
    }
    base.update(params.solver_overrides)
    base.update(updates)
    return SolveParams(**base)


def _chip_windows_by_id(season: int) -> dict[str, rules.ChipWindow]:
    return {f"{w.name.lower()}{w.set}": w for w in rules.chip_windows(season)}


def _chip_candidates(
    *,
    t: int,
    season: int,
    window: pd.DataFrame,
    base_plan: PlanResult,
    owned_codes: Sequence[int],
    chips: Sequence[str],
    params: BacktestParams,
) -> list[str]:
    """Chip kinds whose trigger fires at GW ``t``, best-first (module docstring heuristics)."""
    xp_t = window[window["gw"] == t]
    xp_of = {int(p): float(x) for p, x in zip(xp_t["player_code"], xp_t["xp"])}
    gw0 = base_plan.gws[0]
    bench_ev = sum(xp_of.get(b, 0.0) for b in gw0.bench_order)
    top_xp = float(xp_t["xp"].max()) if len(xp_t) else 0.0
    dgw_now = "n_fixtures" in xp_t.columns and bool((xp_t["n_fixtures"] >= 2).any())
    n_blank = sum(1 for c in owned_codes if c not in xp_of)
    windows_by_id = _chip_windows_by_id(season)

    scored: list[tuple[float, str]] = []
    for kind in _CHIP_KINDS:
        instance = chip_instance_for(season, kind, t, chips)
        if instance is None:
            continue
        gws_left = windows_by_id[instance].last_gw - t
        near_expiry = gws_left <= params.chip_expiry_within
        fired, magnitude = False, 0.0
        if kind == "bb":
            fired = bench_ev >= params.bb_bench_ev_threshold or dgw_now
            magnitude = bench_ev
        elif kind == "tc":
            fired = top_xp >= params.tc_captain_xp_threshold or dgw_now
            magnitude = top_xp
        elif kind == "wc":
            fired = gw0.hit_points >= params.wc_hit_points_trigger
            magnitude = float(gw0.hit_points)
        elif kind == "fh":
            fired = n_blank >= params.fh_blank_squad_threshold
            magnitude = 2.0 * n_blank
        if fired or near_expiry:
            scored.append((magnitude + (5.0 if near_expiry else 0.0), kind))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [kind for _, kind in scored[: params.max_chip_solves_per_gw]]


def _decide_chips(
    *,
    t: int,
    season: int,
    window: pd.DataFrame,
    prices: pd.DataFrame,
    state: SquadState | None,
    chips: Sequence[str],
    base_plan: PlanResult,
    horizon: int,
    params: BacktestParams,
    solve_fn: Callable[..., PlanResult],
) -> tuple[PlanResult, int]:
    """Light forced sweep: chip-now solves for fired triggers vs the no-chip base plan.

    Returns ``(chosen_plan, n_extra_solves)``. The chip plays only when its forced-now
    objective beats the base by more than ``chip_play_margin``.
    """
    owned = [p.player_code for p in state.squad] if state is not None else []
    kinds = _chip_candidates(
        t=t, season=season, window=window, base_plan=base_plan, owned_codes=owned,
        chips=chips, params=params,
    )
    best_plan, best_delta, n_solves = base_plan, 0.0, 0
    for kind in kinds:
        instance = chip_instance_for(season, kind, t, chips)
        if instance is None:  # pragma: no cover — _chip_candidates already filtered
            continue
        banned = {k for k in _CHIP_KINDS if k != kind}
        banned |= {c for c in chips if c[:2] == kind and c != instance}
        forced = _solver_params(
            params,
            no_chips=False,
            forced_chips={t: kind},
            banned_chips=frozenset(banned),
        )
        try:
            plan = solve_fn(window, prices, state, horizon=horizon, params=forced)
        except (InfeasiblePlanError, ValueError, RuntimeError) as exc:
            logger.warning("GW%d forced-%s solve failed: %s", t, kind, exc)
            continue
        n_solves += 1
        delta = plan.objective - base_plan.objective
        logger.info("GW%d chip %s forced-now delta: %+.2f", t, kind, delta)
        if delta > params.chip_play_margin and delta > best_delta:
            best_plan, best_delta = plan, delta
    return best_plan, n_solves


# --------------------------------------------------------------------------------------
# Policy replays
# --------------------------------------------------------------------------------------


def _refresh_prices(squad: Sequence[OwnedPlayer], realized: _Realized, t: int) -> list[OwnedPlayer]:
    """Mark owned players to their GW-``t`` market prices (realized price movements)."""
    refreshed: list[OwnedPlayer] = []
    for p in squad:
        now = realized.value_at(p.player_code, t)
        refreshed.append(
            OwnedPlayer(
                player_code=p.player_code,
                purchase_price=p.purchase_price,
                current_price=now if now is not None else p.current_price,
            )
        )
    return refreshed


def _roll_free_transfers(ft: int | None, used: int, season: int, wc_or_fh: bool, t: int) -> int:
    """FT bank going into GW ``t+1`` (rules.next_free_transfers + the 2025 AFCON top-up)."""
    flags = rules.SEASON_FLAGS[season]
    nxt = 1 if ft is None else rules.next_free_transfers(ft, used, season, wc_or_fh)
    if flags.amnesty == "topped_to_5_at_GW16" and t + 1 == 16:
        nxt = flags.max_free_transfers  # 2025-26 AFCON one-off top-up (mirrors state.py)
    return nxt


def _sell_value(squad: Sequence[OwnedPlayer]) -> int:
    return sum(p.sell_price() for p in squad)


def _replay_solver_policy(
    *,
    name: str,
    description: str,
    season: int,
    gws: Sequence[int],
    window_of: Callable[[int], pd.DataFrame],
    realized: _Realized,
    params: BacktestParams,
    solve_fn: Callable[..., PlanResult] = solve_plan,
    fallback_prices: pd.DataFrame | None = None,
) -> PolicyLedger:
    """Replay one solver-driven policy over ``gws``, rolling all state forward."""
    squad: list[OwnedPlayer] = []
    owned_meta: dict[int, tuple[int, str, int]] = {}
    bank = 0
    ft: int | None = None
    chips = all_chip_ids(season)
    rows: list[GwLedger] = []

    for t in gws:
        if squad:
            squad = _refresh_prices(squad, realized, t)
            state: SquadState | None = SquadState(
                season=season,
                current_gw=t,
                squad=squad,
                bank=bank,
                free_transfers=ft if ft is not None else 1,
                chips_available=list(chips),
            )
        else:
            state = None

        window = window_of(t)
        horizon = params.horizon
        if len(window):
            horizon = max(1, min(params.horizon, int(window["gw"].max()) - t + 1))
        universe = {int(c) for c in window["player_code"]} | {p.player_code for p in squad}
        prices = _prices_frame(realized, t, universe, fallback_prices, owned_meta)

        started = time.perf_counter()
        plan = solve_fn(window, prices, state, horizon=horizon, params=_solver_params(params))
        if params.evaluate_chips and params.max_chip_solves_per_gw > 0:
            plan, _extra = _decide_chips(
                t=t, season=season, window=window, prices=prices, state=state,
                chips=chips, base_plan=plan, horizon=horizon, params=params,
                solve_fn=solve_fn,
            )
        solve_seconds = time.perf_counter() - started

        gw0 = plan.gws[0]
        if gw0.gw != t:
            raise RuntimeError(f"solver returned first GW {gw0.gw}, expected {t}")
        chip_id = gw0.chip
        kind = (chip_id or "")[:2]
        price_map = {int(r.player_code): int(r.price) for r in prices.itertuples()}
        pos_map = {int(r.player_code): str(r.position) for r in prices.itertuples()}
        club_map = {int(r.player_code): int(r.team_code) for r in prices.itertuples()}

        if state is None:  # initial squad: None-state build, unlimited free transfers
            squad = [
                OwnedPlayer(
                    player_code=c, purchase_price=price_map[c], current_price=price_map[c]
                )
                for c in gw0.squad
            ]
            bank = INITIAL_BUDGET - sum(price_map[c] for c in gw0.squad)
            if bank < 0:
                raise RuntimeError(f"initial squad cost exceeds budget (bank={bank})")
            transfers_in: list[int] = []
            transfers_out: list[int] = []
        else:
            transfers_in = list(gw0.transfers_in)
            transfers_out = list(gw0.transfers_out)
            if kind != "fh" and (transfers_in or transfers_out):
                def _value_of(code: int, gw: int = t) -> int | None:
                    value = realized.value_at(code, gw)
                    return value if value is not None else price_map.get(code)

                squad, bank = _apply_transfers(
                    squad, bank, transfers_in, transfers_out, _value_of
                )
        for p in squad:  # remember how to price owned players if they leave the data
            owned_meta[p.player_code] = (
                p.current_price,
                pos_map.get(p.player_code, owned_meta.get(p.player_code, (0, "MID", 0))[1]),
                club_map.get(p.player_code, owned_meta.get(p.player_code, (0, "MID", 0))[2]),
            )

        scored = score_gw(
            squad=gw0.squad,
            lineup=gw0.lineup,
            bench_order=gw0.bench_order,
            captain=gw0.captain,
            vice=gw0.vice,
            chip=chip_id,
            points=realized.gw_points(t),
            minutes=realized.gw_minutes(t),
            positions=pos_map,
        )
        hit_points = int(gw0.hit_points)
        rows.append(
            GwLedger(
                gw=t,
                points=scored.gross_points - hit_points,
                gross_points=scored.gross_points,
                bench_points=scored.bench_points,
                hit_points=hit_points,
                n_transfers=len(transfers_in),
                transfers_in=transfers_in,
                transfers_out=transfers_out,
                chip=chip_id,
                bank=bank,
                squad_value=sum(p.current_price for p in squad),
                team_value=bank + _sell_value(squad),
                free_transfers=ft,
                captain=gw0.captain,
                vice=gw0.vice,
                effective_captain=scored.effective_captain,
                captain_points=scored.captain_points,
                captain_success=scored.captain_success,
                autosubs=scored.autosubs,
                lineup=scored.final_lineup,
                squad=[p.player_code for p in squad],
                expected_points=float(gw0.expected_points),
                solve_seconds=round(solve_seconds, 3),
            )
        )

        if chip_id is not None:
            if chip_id in chips:
                chips.remove(chip_id)
            else:  # pragma: no cover — solver only plays chips from the state inventory
                logger.warning("played chip %s was not in the inventory", chip_id)
        ft = _roll_free_transfers(ft, len(transfers_in), season, kind in ("wc", "fh"), t)

    return _finish_ledger(name, description, rows)


def _replay_set_and_forget(
    *,
    season: int,
    gws: Sequence[int],
    window_of: Callable[[int], pd.DataFrame],
    realized: _Realized,
    params: BacktestParams,
    solve_fn: Callable[..., PlanResult] = solve_plan,
    fallback_prices: pd.DataFrame | None = None,
) -> PolicyLedger:
    """Set-and-forget baseline: initial solve, never transfer, autocaptain highest xP.

    The squad, XI and bench order stay fixed from the initial None-state solve; only the
    captain/vice re-pick each GW (highest model xP within the fixed XI, ties broken on
    player_code). Real autosubs still apply, so 0-minute starters are covered by the
    fixed bench. No chips, no hits.
    """
    t0 = gws[0]
    window = window_of(t0)
    horizon = params.horizon
    if len(window):
        horizon = max(1, min(params.horizon, int(window["gw"].max()) - t0 + 1))
    universe = {int(c) for c in window["player_code"]}
    prices = _prices_frame(realized, t0, universe, fallback_prices, {})
    plan = solve_fn(window, prices, None, horizon=horizon, params=_solver_params(params))
    gw0 = plan.gws[0]
    if gw0.gw != t0:
        raise RuntimeError(f"solver returned first GW {gw0.gw}, expected {t0}")

    price_map = {int(r.player_code): int(r.price) for r in prices.itertuples()}
    pos_map = {int(r.player_code): str(r.position) for r in prices.itertuples()}
    squad = [
        OwnedPlayer(player_code=c, purchase_price=price_map[c], current_price=price_map[c])
        for c in gw0.squad
    ]
    bank = INITIAL_BUDGET - sum(price_map[c] for c in gw0.squad)
    lineup, bench_order = list(gw0.lineup), list(gw0.bench_order)

    ft: int | None = None
    rows: list[GwLedger] = []
    for t in gws:
        squad = _refresh_prices(squad, realized, t)
        gw_window = window_of(t)
        xp_t = gw_window[gw_window["gw"] == t]
        xp_of = {int(p): float(x) for p, x in zip(xp_t["player_code"], xp_t["xp"])}
        ranked = sorted(lineup, key=lambda c: (-xp_of.get(c, 0.0), c))
        captain, vice = ranked[0], ranked[1]
        scored = score_gw(
            squad=[p.player_code for p in squad],
            lineup=lineup,
            bench_order=bench_order,
            captain=captain,
            vice=vice,
            chip=None,
            points=realized.gw_points(t),
            minutes=realized.gw_minutes(t),
            positions=pos_map,
        )
        rows.append(
            GwLedger(
                gw=t,
                points=scored.gross_points,
                gross_points=scored.gross_points,
                bench_points=scored.bench_points,
                hit_points=0,
                n_transfers=0,
                chip=None,
                bank=bank,
                squad_value=sum(p.current_price for p in squad),
                team_value=bank + _sell_value(squad),
                free_transfers=ft,
                captain=captain,
                vice=vice,
                effective_captain=scored.effective_captain,
                captain_points=scored.captain_points,
                captain_success=scored.captain_success,
                autosubs=scored.autosubs,
                lineup=scored.final_lineup,
                squad=[p.player_code for p in squad],
                expected_points=sum(xp_of.get(c, 0.0) for c in lineup)
                + xp_of.get(captain, 0.0),
                solve_seconds=0.0,
            )
        )
        ft = _roll_free_transfers(ft, 0, season, False, t)

    return _finish_ledger(
        "set_and_forget",
        "Initial None-state solve; no transfers or chips; autocaptain = highest model xP "
        "in the fixed XI; real autosubs applied.",
        rows,
    )


def _finish_ledger(name: str, description: str, rows: list[GwLedger]) -> PolicyLedger:
    return PolicyLedger(
        name=name,
        description=description,
        rows=rows,
        total_points=sum(r.points for r in rows),
        total_gross_points=sum(r.gross_points for r in rows),
        total_hit_points=sum(r.hit_points for r in rows),
        total_transfers=sum(r.n_transfers for r in rows),
        chips_played=[r.chip for r in rows if r.chip is not None],
    )


# --------------------------------------------------------------------------------------
# Baseline xP: last-5 average (§6 "Last-5 baseline")
# --------------------------------------------------------------------------------------


def last5_xp_frame(
    player_gw: pd.DataFrame,
    season: int,
    gws: Sequence[int],
    *,
    window: int = 5,
) -> pd.DataFrame:
    """Last-``window``-average baseline xP frame for the requested GWs of ``season``.

    ``xp`` for (player, GW) = mean realized FPL points over the player's previous
    up-to-``window`` player-GW rows STRICTLY before that GW (crossing season boundaries
    when ``player_gw`` spans seasons — no same-GW leak); ``q0`` = share of those rows
    with 0 minutes (1.0 with no history, matching "never seen playing"). Rows are
    emitted for every player with a ``player_gw`` row at each requested GW.
    """
    cols = ["season", "gw", "player_code", "total_points", "minutes"]
    pg = (
        player_gw[cols]
        .sort_values(["player_code", "season", "gw"], kind="stable")
        .reset_index(drop=True)
    )
    codes = pg["player_code"]
    rolled_pts = (
        pg.groupby("player_code", sort=False)["total_points"]
        .rolling(window, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    zero_min = (pg["minutes"] == 0).astype(float)
    rolled_q0 = (
        zero_min.groupby(codes, sort=False)
        .rolling(window, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    pg["xp"] = rolled_pts.groupby(codes, sort=False).shift(1)
    pg["q0"] = rolled_q0.groupby(codes, sort=False).shift(1)
    mask = (pg["season"] == season) & pg["gw"].isin(set(int(g) for g in gws))
    out = pg.loc[mask, ["season", "gw", "player_code"]].copy()
    out["xp"] = pg.loc[mask, "xp"].fillna(0.0).astype(float)
    out["q0"] = pg.loc[mask, "q0"].fillna(1.0).astype(float)
    return out.reset_index(drop=True)


# --------------------------------------------------------------------------------------
# xP-accuracy metrics (§7.3 metric 2)
# --------------------------------------------------------------------------------------


def compute_xp_metrics(
    predictions: pd.DataFrame, season_pg: pd.DataFrame, gws: Sequence[int]
) -> dict[str, Any]:
    """RMSE/MAE by return category, per position, and within-GW Spearman.

    Predictions are left-joined to realized ``player_gw`` points (players predicted but
    absent from the realized table count as 0 points — they did not play). Categories per
    §6: Zeros (0), Blanks (1-2), Tickers (3-4), Haulers (>=5).
    """
    from scipy.stats import spearmanr

    keys = ["season", "gw", "player_code"]
    wanted = set(int(g) for g in gws)
    pred_cols = keys + ["xp"] + (["position"] if "position" in predictions.columns else [])
    pred = predictions.loc[predictions["gw"].isin(wanted), pred_cols].copy()
    merged = pred.merge(season_pg[keys + ["total_points"]], on=keys, how="left")
    merged["total_points"] = merged["total_points"].fillna(0.0)
    err = merged["xp"].astype(float) - merged["total_points"].astype(float)

    def _stats(mask: pd.Series) -> dict[str, float | int]:
        sub = err[mask]
        if not len(sub):
            return {"rmse": float("nan"), "mae": float("nan"), "n": 0}
        return {
            "rmse": round(float((sub**2).mean() ** 0.5), 4),
            "mae": round(float(sub.abs().mean()), 4),
            "n": int(len(sub)),
        }

    pts = merged["total_points"]
    by_category = {
        name: _stats((pts >= lo) & (pts <= hi))
        for name, (lo, hi) in RETURN_CATEGORIES.items()
    }
    by_position: dict[str, dict[str, float | int]] = {}
    if "position" in merged.columns:
        by_position = {
            str(pos): _stats(merged["position"] == pos)
            for pos in sorted(merged["position"].dropna().unique())
        }
    rhos: list[float] = []
    for _, grp in merged.groupby("gw"):
        if len(grp) >= 3 and grp["xp"].nunique() > 1 and grp["total_points"].nunique() > 1:
            rho = spearmanr(grp["xp"], grp["total_points"]).statistic
            if pd.notna(rho):
                rhos.append(float(rho))
    return {
        "overall": _stats(pd.Series(True, index=err.index)),
        "by_category": by_category,
        "by_position": by_position,
        "spearman_within_gw": round(sum(rhos) / len(rhos), 4) if rhos else None,
        "mean_xp": round(float(merged["xp"].mean()), 4) if len(merged) else None,
        "mean_points": round(float(pts.mean()), 4) if len(merged) else None,
    }


# --------------------------------------------------------------------------------------
# The replay orchestrator (frame-injection seam — what the tests drive)
# --------------------------------------------------------------------------------------


def _normalize_gws(gws: Iterable[int] | None, available: Sequence[int]) -> list[int]:
    """Validate/resolve the backtested GW list: contiguous, ascending, data-backed."""
    if gws is None:
        chosen = [int(g) for g in available]
    else:
        chosen = [int(g) for g in gws]
        unknown = sorted(set(chosen) - set(available))
        if unknown:
            raise ValueError(f"no player_gw rows for GWs {unknown}")
    if not chosen:
        raise ValueError("empty GW list")
    if any(b - a != 1 for a, b in zip(chosen, chosen[1:])):
        raise ValueError(f"gws must be contiguous ascending, got {chosen}")
    return chosen


def replay(
    *,
    season: int,
    gws: Iterable[int] | None,
    predictions: pd.DataFrame,
    player_gw: pd.DataFrame,
    params: BacktestParams | None = None,
    solve_fn: Callable[..., PlanResult] = solve_plan,
    mode: str = "injected",
    leakage_warnings: Sequence[str] | None = None,
    artifact_train_window: dict[str, Any] | None = None,
    window_provider: Callable[[int], pd.DataFrame] | None = None,
) -> BacktestResult:
    """Replay a season slice from already-materialized frames (the testable core).

    Args:
        season: Season start year.
        gws: Contiguous GWs to replay (None = every GW with ``player_gw`` rows).
        predictions: Model xP frame (``season, gw, player_code, xp[, q0, n_fixtures,
            position, price, team_code]``) covering the backtested GWs (+ horizon tail
            ideally). Also the source for metrics and set-and-forget captaincy.
        player_gw: Realized per-GW rows. May span multiple seasons — earlier seasons feed
            the last-5 baseline's early-GW history.
        params: :class:`BacktestParams` (None = defaults).
        solve_fn: MILP entry point (test seam; default the real solver).
        mode: Label recorded on the result (``run_backtest`` sets the real modes).
        leakage_warnings / artifact_train_window: Provenance recorded on the result.
        window_provider: Optional override returning the xP window for a decision GW
            (used by retrain mode, where each block has its own prediction frame).

    Returns:
        A :class:`BacktestResult` with the model policy, the baselines (when enabled),
        reference constants and xP-accuracy metrics.
    """
    params = params if params is not None else BacktestParams()
    season_pg = player_gw[player_gw["season"] == season]
    if season_pg.empty:
        raise ValueError(f"player_gw has no rows for season {season}")
    available = sorted(int(g) for g in season_pg["gw"].unique())
    gws_list = _normalize_gws(gws, available)
    realized = _Realized(season_pg, season)

    def _slice_window(frame: pd.DataFrame) -> Callable[[int], pd.DataFrame]:
        def window(t: int) -> pd.DataFrame:
            mask = (
                (frame["season"] == season)
                & (frame["gw"] >= t)
                & (frame["gw"] <= min(t + params.horizon - 1, _LAST_GW))
            )
            return frame.loc[mask]

        return window

    model_window = window_provider if window_provider is not None else _slice_window(predictions)

    policies: dict[str, PolicyLedger] = {}
    policies["model"] = _replay_solver_policy(
        name="model",
        description="Weekly re-solve on model xP with the light chip sweep.",
        season=season,
        gws=gws_list,
        window_of=model_window,
        realized=realized,
        params=params,
        solve_fn=solve_fn,
        fallback_prices=predictions,
    )
    if params.run_baselines:
        last5 = last5_xp_frame(
            player_gw, season, list(range(gws_list[0], min(gws_list[-1] + params.horizon,
                                                           _LAST_GW + 1))),
            window=params.last5_window,
        )
        policies["last5"] = _replay_solver_policy(
            name="last5",
            description="Last-5-average realized points as xP; same solver and chip "
            "policy (§6 baseline).",
            season=season,
            gws=gws_list,
            window_of=_slice_window(last5),
            realized=realized,
            params=params,
            solve_fn=solve_fn,
            fallback_prices=predictions,
        )
        policies["set_and_forget"] = _replay_set_and_forget(
            season=season,
            gws=gws_list,
            window_of=model_window,
            realized=realized,
            params=params,
            solve_fn=solve_fn,
            fallback_prices=predictions,
        )

    return BacktestResult(
        season=season,
        gws=gws_list,
        mode=mode,
        params=params.model_dump(mode="json"),
        policies=policies,
        references={
            "average_manager": AVERAGE_MANAGER_POINTS,
            "top_10k": TOP_10K_POINTS,
        },
        metrics=compute_xp_metrics(predictions, season_pg, gws_list),
        leakage_warnings=list(leakage_warnings or []),
        artifact_train_window=artifact_train_window,
    )


# --------------------------------------------------------------------------------------
# Real-data entry points (predictions from artifacts; pipeline.py wires the CLI)
# --------------------------------------------------------------------------------------


def _artifact_overlap_warnings(
    train_window: dict[str, Any] | None, season: int, first_gw: int
) -> list[str]:
    """§7.2 trap 5 check: warn when the artifacts saw the evaluation window in training."""
    if not train_window:
        return [
            "no artifact manifest train_window found — cannot verify the artifacts were "
            "trained strictly before the backtest window (§7.2 own-model feedback trap)"
        ]
    before_season = train_window.get("before_season")
    before_gw = train_window.get("before_gw")
    seasons = train_window.get("seasons") or []
    if before_season is not None:
        clean = before_season < season or (
            before_season == season and (before_gw is None or before_gw <= first_gw)
        )
    else:
        clean = bool(seasons) and max(seasons) < season
    if clean:
        return []
    return [
        f"LEAKY ARTIFACTS: manifest train_window {train_window} overlaps the backtest "
        f"({season} GW{first_gw}+). Season points are optimistic — retrain with "
        f"`fplai train --before-season {season} --before-gw {first_gw}` (§7.2 trap 5)."
    ]


_HORIZON_LOOKAHEAD_NOTE = (
    "horizon look-ahead: xP rows for GWs t+1..t+H-1 inside each decision are built from "
    "matches before their own GW, not before GW t (planning-only peek; the scored GW is "
    "strictly pre-deadline — see module docstring)"
)


def _predict_gws(
    tables: dict[str, pd.DataFrame],
    models_dir: Path,
    season: int,
    pred_gws: Sequence[int],
    use_odds: bool,
    features: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Per-GW xP predictions for historical GWs from saved artifacts.

    NOTE for the integrator: reuses ``pipeline._load_artifacts`` / ``_predict_frames``
    (the private seams behind ``fplai predict --season --gw``) to avoid re-implementing
    the prediction assembly; promote them to public pipeline API if preferred.
    """
    from fplai.features.windows import build_feature_frame
    from fplai.pipeline import _load_artifacts, _predict_frames

    feats = features if features is not None else build_feature_frame(tables, target="match")
    rows = feats[(feats["season"] == season) & feats["gw"].isin(set(pred_gws))]
    if rows.empty:
        raise ValueError(f"no feature rows for season {season} GWs {list(pred_gws)}")
    _manifest, minutes_model, team_model, rates_model, calibration = _load_artifacts(models_dir)
    _pred, gw_pred, _team_fx = _predict_frames(
        rows,
        tables["fixtures"],
        tables,
        (minutes_model, team_model, rates_model, calibration),
        use_odds,
    )
    return gw_pred


def run_backtest(
    season: int,
    *,
    gws: Iterable[int] | None = None,
    policy_params: BacktestParams | None = None,
    retrain_every: int | None = None,
) -> BacktestResult:
    """Walk-forward season replay on the real processed tables and model artifacts.

    Args:
        season: Season start year (e.g. 2025 for 2025-26).
        gws: Contiguous GW subset to replay (None = the whole season). Use
            ``range(30, 33)`` style subsets for quick runs.
        policy_params: :class:`BacktestParams` (None = defaults; see the module
            docstring for the coarse-mode and chip-policy semantics).
        retrain_every: None = coarse mode (one pretrained artifact set, with the
            manifest leakage check); an integer N retrains artifacts strictly before
            every N-GW block (slow — ~70 s per block — but per-decision clean).

    Returns:
        A labelled :class:`BacktestResult` (``mode`` records which prediction regime
        produced it; ``leakage_warnings`` records every known caveat).
    """
    from fplai import config
    from fplai.features.windows import build_feature_frame
    from fplai.pipeline import load_processed_tables

    if season not in rules.SEASONS:
        raise ValueError(f"season {season} outside supported range {rules.SEASONS}")
    params = policy_params if policy_params is not None else BacktestParams()
    tables = load_processed_tables()
    player_gw = tables["player_gw"]
    season_pg = player_gw[player_gw["season"] == season]
    if season_pg.empty:
        raise ValueError(f"no player_gw rows for season {season} — run `fplai build` first")
    available = sorted(int(g) for g in season_pg["gw"].unique())
    gws_list = _normalize_gws(gws, available)
    first, last = gws_list[0], gws_list[-1]
    pred_last = min(last + params.horizon - 1, _LAST_GW)
    models_dir = Path(params.models_dir) if params.models_dir is not None else config.MODELS_DIR

    warnings: list[str] = [_HORIZON_LOOKAHEAD_NOTE]
    if params.use_odds:
        warnings.append(
            "closing-odds blend enabled: odds for matches later in a GW post-date its "
            "deadline (STATUS.md caveat 4)"
        )

    features = build_feature_frame(tables, target="match")
    window_provider: Callable[[int], pd.DataFrame] | None = None
    train_window: dict[str, Any] | None = None

    if retrain_every is None:
        manifest_path = models_dir / "manifest.json"
        if manifest_path.exists():
            train_window = json.loads(manifest_path.read_text()).get("train_window")
        warnings += _artifact_overlap_warnings(train_window, season, first)
        predictions = _predict_gws(
            tables, models_dir, season, list(range(first, pred_last + 1)), params.use_odds,
            features=features,
        )
        mode = "coarse-pretrained"
    else:
        if retrain_every < 1:
            raise ValueError(f"retrain_every must be >= 1, got {retrain_every}")
        from fplai.pipeline import run_train

        mode = f"retrain-every-{int(retrain_every)}"
        blocks: list[tuple[int, int, pd.DataFrame]] = []
        for block_start in range(first, last + 1, retrain_every):
            block_end = min(block_start + retrain_every - 1, last)
            block_dir = models_dir / "backtest" / f"{season}-before-gw{block_start:02d}"
            if not (block_dir / "manifest.json").exists():
                logger.info("retraining artifacts before %d GW%d ...", season, block_start)
                run_train(
                    before_season=season, before_gw=block_start, models_dir=block_dir
                )
            block_pred_gws = list(
                range(block_start, min(block_end + params.horizon - 1, _LAST_GW) + 1)
            )
            blocks.append(
                (
                    block_start,
                    block_end,
                    _predict_gws(
                        tables, block_dir, season, block_pred_gws, params.use_odds,
                        features=features,
                    ),
                )
            )
        # Decisions inside a block read that block's frame (horizon rows included), so
        # every xP a decision at GW t consumes comes from artifacts trained before t.
        def _block_window(t: int) -> pd.DataFrame:
            frame = next(fr for b0, b1, fr in blocks if b0 <= t <= b1)
            mask = (frame["gw"] >= t) & (frame["gw"] <= min(t + params.horizon - 1, _LAST_GW))
            return frame.loc[mask]

        window_provider = _block_window
        # Metrics/set-and-forget frame: each scored GW from its own block (clean).
        parts = [fr[(fr["gw"] >= b0) & (fr["gw"] <= b1)] for b0, b1, fr in blocks]
        parts.append(blocks[-1][2][blocks[-1][2]["gw"] > blocks[-1][1]])
        predictions = pd.concat(parts, ignore_index=True)
        train_window = {"retrain_every": retrain_every, "before_season": season}

    return replay(
        season=season,
        gws=gws_list,
        predictions=predictions,
        player_gw=player_gw,
        params=params,
        mode=mode,
        leakage_warnings=warnings,
        artifact_train_window=train_window,
        window_provider=window_provider,
    )


def run(
    season: int,
    *,
    gws: Iterable[int] | None = None,
    retrain_every: int | None = None,
    policy_params: BacktestParams | None = None,
    out_dir: Path | None = None,
    save: bool = True,
) -> BacktestResult:
    """CLI-ready entry point: run the backtest, print a summary, save the artifacts.

    ``pipeline.py`` wires ``fplai backtest`` to this function. Results are saved under
    ``<out_dir or DATA_DIR/backtests>/<season>-gw<first>-<last>/`` (``result.json`` +
    ``ledger.parquet``).
    """
    from rich.console import Console

    from fplai import config

    console = Console()
    result = run_backtest(
        season, gws=gws, policy_params=policy_params, retrain_every=retrain_every
    )
    first, last = result.gws[0], result.gws[-1]
    console.print(
        f"[bold]backtest[/bold]: {rules.season_label(season)} GW{first}-{last} "
        f"mode={result.mode}"
    )
    for name, ledger in result.policies.items():
        chips = ",".join(ledger.chips_played) or "none"
        console.print(
            f"  {name}: [bold]{ledger.total_points}[/bold] pts "
            f"(gross {ledger.total_gross_points}, hits -{ledger.total_hit_points}, "
            f"transfers {ledger.total_transfers}, chips {chips})"
        )
    console.print(
        f"  references (full 38-GW seasons, §7.3): average manager ≈ "
        f"{result.references['average_manager']}, top-10k ≈ {result.references['top_10k']}"
    )
    overall = result.metrics.get("overall", {})
    console.print(
        f"  xP accuracy: RMSE {overall.get('rmse')} MAE {overall.get('mae')} "
        f"(n={overall.get('n')}), within-GW Spearman "
        f"{result.metrics.get('spearman_within_gw')}"
    )
    for warning in result.leakage_warnings:
        console.print(f"  [yellow]caveat: {warning}[/yellow]")
    if save:
        target = (
            Path(out_dir)
            if out_dir is not None
            else config.DATA_DIR / "backtests" / f"{season}-gw{first:02d}-{last:02d}"
        )
        paths = result.save(target)
        console.print(f"  saved: {paths['result']} + {paths['ledger']}")
    return result
