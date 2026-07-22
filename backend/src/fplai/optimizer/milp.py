"""Multi-gameweek FPL squad-planning MILP on HiGHS (MODEL_DESIGN_INPUTS.md §3).

Implemented from the transcribed mathematical formulation only (§3.1 variables,
§3.2 constraints, §3.3 objective, §3.5 engineering) — no external solver code was copied.

Formulation summary
-------------------
Binary ``squad``/``lineup``/``captain``/``vicecap``/``bench[slot]``/``transfer_in``/
``transfer_out_{first,regular}`` per player-week, a separate Free-Hit squad, per-player
Triple-Captain binaries, chip binaries per week, continuous in-the-bank, and an FT state
machine (integer ``fts`` in 1..5 with a one-hot ``fts_state`` and big-M(=20) clamps).
The objective is decayed weekly xP with captain/vice/TC weighting, per-slot bench weights,
-4/hit, an incremental FT continuation value, and an in-the-bank value.

Deviation from the §3.2 one-liner, documented: the literal transcription
``raw_ft = fts - transfer_count + 1 - use_wc - use_fh`` contradicts the authoritative
game rule (FPL_KNOWLEDGE §1.6: WC/FH pass the FT bank through with the +1 accrual). We
keep the exact one-hot/big-M machinery but feed it an *effective* FT-consuming transfer
count: on 2024-25+ WC/FH weeks it may drop to 0 (bank pass-through; the canonical value
is always weakly optimal, and reported trajectories are replayed through
``rules.next_free_transfers`` at extraction — see ``_add_ft_machine``/``extract``); on
pre-2024 WC/FH weeks it is pinned to ``fts[w]`` exactly, reproducing the old
chips-WIPE-the-bank rule. The bank cap is season-aware
(``rules.SEASON_FLAGS[season].max_free_transfers``: 2 through 2023-24, 5 after). Both
regimes are property-verified against ``rules.next_free_transfers`` in tests.

Game constants used here that ``fplai.rules`` does not export (squad size 15, quotas
2/5/5/3, club limit 3, formation bounds, £100.0m budget, 20-transfer cap) are encoded
from FPL_KNOWLEDGE §1.3/§1.6 as module constants.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import highspy
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from fplai import rules
from fplai.optimizer.state import ChipId, SquadState, all_chip_ids, chip_instance_for

logger = logging.getLogger(__name__)

# -- game constants not exported by fplai.rules (FPL_KNOWLEDGE §1.3 / §1.6) -------------
SQUAD_SIZE: int = 15
XI_SIZE: int = 11
POSITION_QUOTA: dict[str, int] = {"GKP": 2, "DEF": 5, "MID": 5, "FWD": 3}
FORMATION_MIN: dict[str, int] = {"GKP": 1, "DEF": 3, "MID": 2, "FWD": 1}
FORMATION_MAX: dict[str, int] = {"GKP": 1, "DEF": 5, "MID": 5, "FWD": 3}
CLUB_LIMIT: int = 3
INITIAL_BUDGET: int = 1000  # £100.0m in £0.1m units
TRANSFER_CAP: int = 20  # per-GW hard cap outside chip weeks (§1.6)
BENCH_SLOTS: tuple[int, ...] = (0, 1, 2, 3)  # 0 = reserve GK
_FT_BIG_M: int = 20  # big-M of the FT state machine, per §3.2
_CHIP_KINDS: tuple[str, ...] = ("wc", "fh", "bb", "tc")

LAST_GW: int = 38


class InfeasiblePlanError(RuntimeError):
    """The MILP (or the input state) admits no legal plan.

    ``problems`` lists the violated/binding constraint groups detected.
    """

    def __init__(self, message: str, problems: list[str] | None = None) -> None:
        self.problems = problems or []
        details = ("\n  - " + "\n  - ".join(self.problems)) if self.problems else ""
        super().__init__(message + details)


class SolveParams(BaseModel):
    """Tunable knobs of :func:`solve_plan`, with MODEL_DESIGN_INPUTS §3.3/§3.5 defaults."""

    model_config = ConfigDict(frozen=True)

    # objective (§3.3)
    decay: float = Field(default=0.84, gt=0.0, le=1.0, description="Decay base β")
    bench_weights: dict[int, float] = Field(
        default={0: 0.03, 1: 0.21, 2: 0.06, 3: 0.002},
        description="Static per-slot bench weights (slot 0 = GK)",
    )
    bench_weights_by_gw: dict[int, dict[int, float]] = Field(
        default_factory=dict,
        description="Per-GW per-slot overrides (fed by optimizer.autosubs)",
    )
    vice_weight: float = Field(default=0.1, ge=0.0, le=1.0)
    use_q0_vice_weight: bool = Field(
        default=True,
        description="Weight the vice by the presumptive captain's q0 (per-week scalar; "
        "falls back to vice_weight when q0 is absent)",
    )
    ft_value: float | tuple[float, float, float, float, float] = Field(
        default=1.5,
        description="FT continuation value: points per saved FT, or per-state increments "
        "for states 1..5 (V(s) = Σ increments)",
    )
    itb_value: float = Field(default=0.08, ge=0.0, description="Points per £1.0m per GW")
    hit_cost: float = Field(default=4.0, ge=0.0, description="Points per extra transfer (§1.6)")
    # solver engineering (§3.5)
    time_limit_s: float = Field(default=60.0, gt=0.0, le=1200.0)
    mip_rel_gap: float = Field(default=1e-6, ge=0.0)
    threads: int = Field(default=1, ge=0, description="0 = automatic")
    seed: int = Field(default=0, ge=0)
    verbose: bool = False
    # player-pool pruning (§3.5)
    prune: bool = True
    prune_keep_frac: float = Field(default=0.10, gt=0.0, le=1.0)
    prune_value_frac: float = Field(default=0.05, ge=0.0, le=1.0)
    prune_keep_cheapest: int = Field(default=5, ge=0)
    prune_min_per_position: int | None = Field(
        default=None, description="Per-position floor; None = quota + 3"
    )
    # optional constraint hooks (§3.2)
    banned_players: set[int] = Field(default_factory=set)
    locked_players: set[int] = Field(default_factory=set)
    forced_chips: dict[int, str] = Field(
        default_factory=dict, description='gw -> chip kind ("wc") or instance id ("wc1")'
    )
    banned_chips: frozenset[str] = Field(
        default_factory=frozenset,
        description='Chip instance ids ("bb1") or kinds ("bb") the solver may not play; '
        "an already-active chip from the state stays committed regardless "
        "(it cannot be un-played). This is the optimizer.chips coordination field.",
    )
    banned_chip_gws: dict[int, frozenset[str]] = Field(
        default_factory=dict,
        description='gw -> chip kinds ("wc") or instance ids ("wc1") not playable that '
        "specific GW (e.g. transfer chips at the first GW of an initial build, where a "
        "Free Hit is degenerate: there is no prior squad to revert to)",
    )
    no_chips: bool = False
    max_hits_per_gw: int | None = Field(
        default=None, ge=0, description="Max penalized transfers (count, not points) per GW"
    )
    max_total_hits: int | None = Field(default=None, ge=0)
    no_transfer_gws: set[int] = Field(default_factory=set)
    transfer_cap: int = Field(default=TRANSFER_CAP, ge=1)


class GwPlan(BaseModel):
    """One gameweek of the plan. All player fields are ``player_code`` values."""

    gw: int
    squad: list[int] = Field(description="Active 15 that GW (the FH squad on FH weeks)")
    lineup: list[int]
    bench_order: list[int] = Field(description="Slot 0 (GK) first; empty on BB weeks")
    captain: int
    vice: int
    transfers_in: list[int] = Field(description="Permanent transfers (empty on FH weeks)")
    transfers_out: list[int]
    hit_points: int
    chip: ChipId | None = None
    expected_points: float = Field(
        description="Σ xP over the lineup + captain (+TC) — no bench/vice weighting"
    )
    bank: int
    free_transfers: int | None = Field(
        description="FTs going into this GW; None = unlimited (initial build)"
    )


class PlanResult(BaseModel):
    """Solved plan plus solver diagnostics (ARCHITECTURE stage-3 contract)."""

    objective: float
    gws: list[GwPlan]
    solve_seconds: float
    gap: float
    status: str
    n_variables: int = 0
    n_constraints: int = 0


# --------------------------------------------------------------------------------------
# Thin helper over highspy: named variables, sums, constraint adder, solve diagnostics
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SolveDiagnostics:
    """Outcome of one HiGHS solve."""

    status: str
    has_solution: bool
    objective: float
    gap: float
    seconds: float


class HighsModel:
    """A small named-variable modeling layer over :class:`highspy.Highs`.

    highspy's expression API is used directly for arithmetic; this class adds a name
    registry (duplicate detection, debuggability), empty-sum handling, constraint
    counting and a one-call solve returning tidy diagnostics.
    """

    def __init__(self, name: str = "milp", *, verbose: bool = False) -> None:
        self.name = name
        self.h = highspy.Highs()
        if not verbose:
            self.h.silent()
        self._vars: dict[str, Any] = {}
        self._n_constraints = 0

    # -- variables ---------------------------------------------------------------------

    def _register(self, name: str, var: Any) -> Any:
        if name in self._vars:
            raise ValueError(f"duplicate variable name {name!r}")
        self._vars[name] = var
        return var

    def binary(self, name: str) -> Any:
        """Add a {0,1} variable."""
        return self._register(name, self.h.addBinary(name=name))

    def integer(self, name: str, *, lb: float, ub: float) -> Any:
        """Add a bounded integer variable."""
        var = self.h.addVariable(lb=lb, ub=ub, type=highspy.HighsVarType.kInteger, name=name)
        return self._register(name, var)

    def continuous(self, name: str, *, lb: float = 0.0, ub: float | None = None) -> Any:
        """Add a continuous variable (default ≥ 0)."""
        var = self.h.addVariable(lb=lb, ub=highspy.kHighsInf if ub is None else ub, name=name)
        return self._register(name, var)

    # -- expressions & constraints -----------------------------------------------------

    def sum(self, terms: Iterable[Any]) -> Any:
        """Sum of variables/linear expressions; 0.0 for an empty iterable."""
        items = list(terms)
        if not items:
            return 0.0
        return self.h.qsum(items)

    def add(self, constraint: Any, name: str) -> None:
        """Add a relational linear constraint built with highspy expressions."""
        self.h.addConstr(constraint, name=name)
        self._n_constraints += 1

    @property
    def n_variables(self) -> int:
        """Number of registered variables."""
        return len(self._vars)

    @property
    def n_constraints(self) -> int:
        """Number of added constraints."""
        return self._n_constraints

    # -- solving -----------------------------------------------------------------------

    def set_start(self, assignment: Mapping[Any, float]) -> None:
        """Provide a MIP-start solution; variables absent from ``assignment`` are 0.

        Must be called after the objective is set: any model modification (including
        ``setObjective``) invalidates a previously supplied solution.
        """
        values = [0.0] * int(self.h.getNumCol())
        for var, val in assignment.items():
            values[var.index] = float(val)
        sol = highspy.HighsSolution()
        sol.col_value = values
        self.h.setSolution(sol)

    def solve(
        self,
        objective: Any,
        *,
        time_limit_s: float,
        mip_rel_gap: float,
        seed: int = 0,
        threads: int = 1,
        start: Mapping[Any, float] | None = None,
    ) -> SolveDiagnostics:
        """Maximize ``objective`` and return diagnostics (presolve/parallel per §3.5).

        ``start`` is an optional MIP-start assignment (see :meth:`set_start`) applied
        after the objective is installed so HiGHS keeps it as the incumbent seed.
        """
        h = self.h
        h.setOptionValue("presolve", "on")
        # HiGHS checks its clock between nodes and can overshoot the limit by a
        # few hundredths of a second; hand it a slightly smaller internal limit
        # so the solve genuinely respects the caller's budget.
        margin = min(1.0, max(0.05, 0.01 * float(time_limit_s)))
        h.setOptionValue("time_limit", max(0.05, float(time_limit_s) - margin))
        h.setOptionValue("mip_rel_gap", float(mip_rel_gap))
        h.setOptionValue("random_seed", int(seed))
        h.setOptionValue("threads", int(threads))
        if threads != 1:
            h.setOptionValue("parallel", "on")
        h.setObjective(objective, highspy.ObjSense.kMaximize)
        if start is not None:
            self.set_start(start)
        h.solve()
        status = h.getModelStatus()
        if status == highspy.HighsModelStatus.kOptimal:
            status_str = "optimal"
        elif status == highspy.HighsModelStatus.kInfeasible:
            status_str = "infeasible"
        elif status == highspy.HighsModelStatus.kTimeLimit:
            status_str = "time_limit"
        else:
            status_str = h.modelStatusToString(status).lower().replace(" ", "_")
        info = h.getInfo()
        has_solution = int(info.primal_solution_status) == 2  # kSolutionStatusFeasible
        return SolveDiagnostics(
            status=status_str,
            has_solution=has_solution,
            objective=float(h.getObjectiveValue()) if has_solution else math.nan,
            gap=float(info.mip_gap) if has_solution else math.inf,
            seconds=float(h.getRunTime()),
        )

    def value(self, var: Any) -> float:
        """Solution value of a variable."""
        return float(self.h.val(var))


# --------------------------------------------------------------------------------------
# Input validation and player-pool pruning
# --------------------------------------------------------------------------------------


def _index_prices(prices: pd.DataFrame) -> tuple[dict[int, int], dict[int, str], dict[int, int]]:
    """Validate the prices frame and return (price, position, club) dicts by player_code."""
    required = {"player_code", "price", "position", "team_code"}
    missing = required - set(prices.columns)
    if missing:
        raise ValueError(f"prices frame missing columns: {sorted(missing)}")
    if prices["player_code"].duplicated().any():
        dupes = prices.loc[prices["player_code"].duplicated(), "player_code"].tolist()
        raise ValueError(f"prices frame has duplicate player_code rows: {dupes[:5]}")
    bad_pos = set(prices["position"]) - set(rules.POSITIONS)
    if bad_pos:
        raise ValueError(f"prices frame has unknown positions: {sorted(bad_pos)}")
    codes = prices["player_code"].astype(int)
    price = dict(zip(codes, prices["price"].astype(int), strict=True))
    pos = dict(zip(codes, prices["position"].astype(str), strict=True))
    club = dict(zip(codes, prices["team_code"].astype(int), strict=True))
    return price, pos, club


def _index_xp(
    xp: pd.DataFrame, season: int, weeks: list[int]
) -> tuple[dict[tuple[int, int], float], dict[tuple[int, int], float]]:
    """Return xp and q0 lookups keyed by (player_code, gw), restricted to the horizon.

    Accepts either the per-GW ``predictions_gw`` frame (one row per player-GW) or a
    per-fixture frame: duplicate (player_code, gw) rows are summed over ``xp`` (a DGW
    player's fixtures add up, per the assemble contract) and their ``q0`` combined
    multiplicatively (missing the whole GW = missing every fixture, independence
    approximation — q0 only feeds the vice-weight heuristic).
    """
    required = {"season", "gw", "player_code", "xp"}
    missing = required - set(xp.columns)
    if missing:
        raise ValueError(f"xp frame missing columns: {sorted(missing)}")
    sub = xp[(xp["season"] == season) & xp["gw"].isin(weeks)].copy()
    if sub.duplicated(["player_code", "gw"]).any():
        logger.debug("xp frame has per-fixture rows; summing within (player_code, gw)")
    sub["player_code"] = sub["player_code"].astype(int)
    sub["gw"] = sub["gw"].astype(int)
    sub["xp"] = sub["xp"].astype(float)
    grouped = sub.groupby(["player_code", "gw"], sort=False)
    xp_map = {
        (int(p), int(g)): float(v) for (p, g), v in grouped["xp"].sum().items()
    }
    q0_map: dict[tuple[int, int], float] = {}
    if "q0" in sub.columns:
        sub["q0"] = sub["q0"].astype(float)
        q0_prod = sub.groupby(["player_code", "gw"], sort=False)["q0"].prod(min_count=1)
        q0_map = {
            (int(p), int(g)): float(v)
            for (p, g), v in q0_prod.items()
            if not math.isnan(v)
        }
    return xp_map, q0_map


def validate_state(
    state: SquadState, pos_of: Mapping[int, str], price_of: Mapping[int, int]
) -> list[str]:
    """Check a squad state against squad-legality rules; return violated-group messages."""
    problems: list[str] = []
    codes = [p.player_code for p in state.squad]
    if len(codes) != SQUAD_SIZE:
        problems.append(f"squad-size: {len(codes)} players, must be exactly {SQUAD_SIZE}")
    if len(set(codes)) != len(codes):
        problems.append("squad-duplicates: repeated player_code in squad")
    unknown = [c for c in codes if c not in pos_of or c not in price_of]
    if unknown:
        problems.append(f"unknown-players: not in prices frame: {unknown[:5]}")
    else:
        counts = {pos: 0 for pos in rules.POSITIONS}
        for c in codes:
            counts[pos_of[c]] += 1
        if counts != POSITION_QUOTA:
            problems.append(f"position-quota: have {counts}, need {POSITION_QUOTA}")
    ft_cap = rules.SEASON_FLAGS[state.season].max_free_transfers
    if state.free_transfers > ft_cap:
        problems.append(
            f"free-transfers: {state.free_transfers} exceeds the season "
            f"{state.season} bank cap of {ft_cap}"
        )
    known_ids = set(all_chip_ids(state.season))
    bad_chips = [c for c in state.chips_available if c not in known_ids]
    if bad_chips:
        problems.append(f"chip-ids: unknown for season {state.season}: {bad_chips}")
    if state.active_chip is not None and state.active_chip[:2] not in _CHIP_KINDS:
        problems.append(f"active-chip: {state.active_chip!r} is not plannable (wc/fh/bb/tc)")
    return problems


def _prune_pool(
    candidates: list[int],
    score: Mapping[int, float],
    price_of: Mapping[int, int],
    pos_of: Mapping[int, str],
    must_keep: set[int],
    params: SolveParams,
) -> list[int]:
    """Player-pool pruning per §3.5: per-position top-EV + EV-per-price + cheap enablers.

    Always keeps ``must_keep`` (current squad, locked players). Deterministic: ties break
    on player_code.
    """
    keep: set[int] = set(must_keep)
    for pos in rules.POSITIONS:
        group = [p for p in candidates if pos_of[p] == pos]
        if not group:
            continue
        floor = params.prune_min_per_position
        if floor is None:
            floor = POSITION_QUOTA[pos] + 3
        n_top = max(math.ceil(params.prune_keep_frac * len(group)), floor)
        by_ev = sorted(group, key=lambda p: (-score[p], p))
        keep.update(by_ev[:n_top])
        n_val = math.ceil(params.prune_value_frac * len(group))
        by_value = sorted(group, key=lambda p: (-score[p] / max(price_of[p], 1), p))
        keep.update(by_value[:n_val])
        by_price = sorted(group, key=lambda p: (price_of[p], p))
        keep.update(by_price[: params.prune_keep_cheapest])
    return sorted(keep & set(candidates) | must_keep)


# --------------------------------------------------------------------------------------
# The model builder
# --------------------------------------------------------------------------------------


class _PlanModel:
    """Builds, solves and extracts the multi-GW plan MILP for one call of solve_plan."""

    def __init__(
        self,
        *,
        pool: list[int],
        weeks: list[int],
        season: int,
        state: SquadState | None,
        params: SolveParams,
        price_of: Mapping[int, int],
        pos_of: Mapping[int, str],
        club_of: Mapping[int, int],
        xp_map: Mapping[tuple[int, int], float],
        q0_map: Mapping[tuple[int, int], float],
    ) -> None:
        self.pool = pool
        self.weeks = weeks
        self.season = season
        self.state = state
        self.params = params
        self.price_of = price_of
        self.pos_of = pos_of
        self.club_of = club_of
        self.xp_map = xp_map
        self.q0_map = q0_map

        self.initial_mode = state is None
        self.start = weeks[0]
        self.end = weeks[-1] + 1  # terminal FT-state key
        flags = rules.SEASON_FLAGS[season]
        #: season FT regime: bank cap (2 through 2023-24, 5 after) and whether WC/FH
        #: wipe the banked FTs to 1 (pre-2024) instead of passing them through.
        self.ft_cap: int = flags.max_free_transfers
        self.ft_wipe: bool = flags.chips_wipe_banked_fts

        # ownership & sell prices (§1.8 sell formula against the authoritative price frame)
        self.own0: dict[int, int] = {}
        self.sell_first: dict[int, int] = {}
        if state is not None:
            for op in state.squad:
                self.own0[op.player_code] = 1
                sell = rules.sell_price(op.purchase_price, price_of[op.player_code])
                if sell != price_of[op.player_code]:
                    self.sell_first[op.player_code] = sell
        #: sale valuation used for the FH affordability budget (first-sale price for
        #: price-modified originals, current price otherwise — mild approximation if an
        #: original is sold and rebought before the FH week).
        self.sale_value: dict[int, int] = {
            p: self.sell_first.get(p, self.price_of[p]) for p in pool
        }

        self._resolve_chips()

        self.m = HighsModel("fpl_plan", verbose=params.verbose)
        self._build_vars()

    # -- chip availability -------------------------------------------------------------

    def _resolve_chips(self) -> None:
        """Compute per-week chip availability and per-instance windows from rules."""
        params, weeks = self.params, self.weeks
        windows = {
            f"{w.name.lower()}{w.set}": w
            for w in rules.chip_windows(self.season)
            if w.name.lower() in _CHIP_KINDS
        }
        if self.state is None:
            available = [cid for cid in all_chip_ids(self.season) if cid in windows]
        else:
            available = [cid for cid in self.state.chips_available if cid in windows]
            skipped = set(self.state.chips_available) - set(available)
            if skipped:
                logger.warning("ignoring unplannable chips %s", sorted(skipped))
        if params.no_chips:
            if params.forced_chips or (self.state and self.state.active_chip):
                raise ValueError("no_chips=True conflicts with forced_chips/active_chip")
            available = []
        if params.banned_chips:
            banned = {str(b).lower() for b in params.banned_chips}
            known = set(all_chip_ids(self.season)) | set(_CHIP_KINDS) | {"aoa", "am"}
            unknown = banned - known
            if unknown:
                raise ValueError(f"banned_chips has unknown entries: {sorted(unknown)}")
            available = [c for c in available if c not in banned and c[:2] not in banned]
        #: instance id -> weeks of the horizon it covers
        self.instance_weeks: dict[str, list[int]] = {}
        for cid in available:
            win = windows[cid]
            ws = [w for w in weeks if win.first_gw <= w <= win.last_gw]
            if ws:
                self.instance_weeks[cid] = ws
        #: kind -> weeks where at least one instance allows it
        self.allowed: dict[str, set[int]] = {kind: set() for kind in _CHIP_KINDS}
        for cid, ws in self.instance_weeks.items():
            self.allowed[cid[:2]].update(ws)
        # per-GW bans; applied before the active-chip add so an active chip always wins
        for gw, chips in params.banned_chip_gws.items():
            for chip in chips:
                kind = str(chip)[:2].lower()
                if kind in self.allowed:
                    self.allowed[kind].discard(gw)
        # None-state (initial build): transfer chips at the first GW are degenerate —
        # a Free Hit has no prior squad to revert to and a Wildcard adds nothing to the
        # unlimited initial transfers — so they are never playable there (an explicit
        # forced_chips entry for that GW raises InfeasiblePlanError below, on purpose).
        if self.state is None:
            for kind in ("wc", "fh"):
                self.allowed[kind].discard(self.start)
        # an already-active chip is playable at start even though its instance was consumed
        self.active_kind: str | None = None
        if self.state is not None and self.state.active_chip is not None:
            kind = self.state.active_chip[:2]
            if kind in _CHIP_KINDS:
                self.active_kind = kind
                self.allowed[kind].add(self.start)
        # forced chips: normalize gw -> kind
        self.forced: dict[int, str] = {}
        for gw, chip in params.forced_chips.items():
            kind = chip[:2].lower()
            if kind not in _CHIP_KINDS:
                raise ValueError(f"forced_chips[{gw}]={chip!r}: unknown chip kind")
            if gw not in weeks:
                raise ValueError(f"forced_chips[{gw}]: gw outside horizon {weeks[0]}-{weeks[-1]}")
            if gw not in self.allowed[kind]:
                raise InfeasiblePlanError(
                    f"forced chip {chip!r} at GW{gw} is not available there",
                    [f"chip-windows: {kind} has no available instance covering GW{gw}"],
                )
            self.forced[gw] = kind

    # -- variables ---------------------------------------------------------------------

    def _build_vars(self) -> None:
        m, pool, weeks = self.m, self.pool, self.weeks
        self.fh_weeks = sorted(self.allowed["fh"])
        self.tc_weeks = sorted(self.allowed["tc"])
        gk = {p for p in pool if self.pos_of[p] == "GKP"}

        self.sq = {(p, w): m.binary(f"squad[{p},{w}]") for p in pool for w in weeks}
        self.fh_sq = {
            (p, w): m.binary(f"squad_fh[{p},{w}]") for p in pool for w in self.fh_weeks
        }
        self.lin = {(p, w): m.binary(f"lineup[{p},{w}]") for p in pool for w in weeks}
        self.cap = {(p, w): m.binary(f"captain[{p},{w}]") for p in pool for w in weeks}
        self.vic = {(p, w): m.binary(f"vicecap[{p},{w}]") for p in pool for w in weeks}
        self.bench = {
            (p, w, o): m.binary(f"bench[{p},{w},{o}]")
            for p in pool
            for w in weeks
            for o in BENCH_SLOTS
            if (o == 0) == (p in gk)
        }
        self.inn = {(p, w): m.binary(f"transfer_in[{p},{w}]") for p in pool for w in weeks}
        self.out_r = {
            (p, w): m.binary(f"transfer_out_regular[{p},{w}]") for p in pool for w in weeks
        }
        self.out_f = {
            (p, w): m.binary(f"transfer_out_first[{p},{w}]")
            for p in self.sell_first
            for w in weeks
        }
        self.itb = {w: m.continuous(f"in_the_bank[{w}]") for w in weeks}
        self.count = {w: m.integer(f"transfer_count[{w}]", lb=0, ub=SQUAD_SIZE) for w in weeks}
        self.use_wc = {w: m.binary(f"use_wc[{w}]") for w in weeks}
        self.use_fh = {w: m.binary(f"use_fh[{w}]") for w in weeks}
        self.use_bb = {w: m.binary(f"use_bb[{w}]") for w in weeks}
        self.use_tc = {
            (p, w): m.binary(f"use_tc[{p},{w}]") for p in pool for w in self.tc_weeks
        }
        # FT machine weeks: every horizon week except the unlimited initial-build week
        self.machine_weeks = [w for w in weeks if not (self.initial_mode and w == self.start)]
        self.pen = {
            w: m.integer(f"penalized_transfers[{w}]", lb=0, ub=SQUAD_SIZE)
            for w in self.machine_weeks
        }
        self.tc_eff = {
            w: m.integer(f"ft_consuming_transfers[{w}]", lb=0, ub=SQUAD_SIZE)
            for w in self.machine_weeks
        }
        self.ft_above = {w: m.binary(f"ft_above_ub[{w}]") for w in self.machine_weeks}
        self.ft_below = {w: m.binary(f"ft_below_lb[{w}]") for w in self.machine_weeks}
        # FT state vars exist for every week after start, plus the terminal state
        self.ft_var_weeks = [w for w in weeks if w != self.start] + [self.end]
        self.fts = {
            w: m.integer(f"fts[{w}]", lb=1, ub=self.ft_cap) for w in self.ft_var_weeks
        }
        self.fts_state = {
            (w, s): m.binary(f"fts_state[{w},{s}]")
            for w in self.ft_var_weeks
            for s in range(self.ft_cap + 1)
        }

    # -- expression helpers ------------------------------------------------------------

    def _prev_sq(self, p: int, w: int) -> Any:
        """squad[p, w-1]: a variable, or the initial-ownership constant at the start."""
        if w == self.start:
            return float(self.own0.get(p, 0))
        return self.sq[(p, w - 1)]

    def _prev_itb(self, w: int) -> Any:
        """in_the_bank[w-1]: a variable, or the starting bank/budget constant."""
        if w == self.start:
            return float(INITIAL_BUDGET if self.state is None else self.state.bank)
        return self.itb[w - 1]

    def _fts_expr(self, w: int) -> Any:
        """fts[w]: a variable, or the state's FT count constant at the start."""
        if w == self.start:
            return float(self.state.free_transfers) if self.state is not None else 1.0
        return self.fts[w]

    def _out_total(self, p: int, w: int) -> Any:
        of = self.out_f.get((p, w))
        return self.out_r[(p, w)] + of if of is not None else self.out_r[(p, w)]

    def _bench_of(self, p: int, w: int) -> list[Any]:
        return [self.bench[(p, w, o)] for o in BENCH_SLOTS if (p, w, o) in self.bench]

    # -- constraints (§3.2) ------------------------------------------------------------

    def build(self) -> None:
        """Add every constraint group and set the objective."""
        self._add_squad_composition()
        self._add_continuity_and_budget()
        self._add_transfer_counts()
        self._add_ft_machine()
        self._add_lineup()
        self._add_chips()
        self._add_optional_hooks()
        self._objective = self._build_objective()
        logger.info(
            "MILP built: %d players x %d weeks -> %d variables, %d constraints",
            len(self.pool),
            len(self.weeks),
            self.m.n_variables,
            self.m.n_constraints,
        )

    def _add_squad_composition(self) -> None:
        m, pool = self.m, self.pool
        clubs = sorted({self.club_of[p] for p in pool})
        for w in self.weeks:
            m.add(m.sum(self.sq[(p, w)] for p in pool) == SQUAD_SIZE, f"squad_size[{w}]")
            for pos, quota in POSITION_QUOTA.items():
                members = [self.sq[(p, w)] for p in pool if self.pos_of[p] == pos]
                m.add(m.sum(members) == quota, f"quota[{pos},{w}]")
            for club in clubs:
                members = [self.sq[(p, w)] for p in pool if self.club_of[p] == club]
                if len(members) > CLUB_LIMIT:
                    m.add(m.sum(members) <= CLUB_LIMIT, f"club_limit[{club},{w}]")
        # the FH squad mirrors all composition constraints x use_fh[w]
        for w in self.fh_weeks:
            fh = self.use_fh[w]
            m.add(
                m.sum(self.fh_sq[(p, w)] for p in pool) - SQUAD_SIZE * fh == 0,
                f"fh_squad_size[{w}]",
            )
            for pos, quota in POSITION_QUOTA.items():
                members = [self.fh_sq[(p, w)] for p in pool if self.pos_of[p] == pos]
                m.add(m.sum(members) - quota * fh == 0, f"fh_quota[{pos},{w}]")
            for club in clubs:
                members = [self.fh_sq[(p, w)] for p in pool if self.club_of[p] == club]
                if len(members) > CLUB_LIMIT:
                    m.add(m.sum(members) - CLUB_LIMIT * fh <= 0, f"fh_club_limit[{club},{w}]")
            # FH affordability against the previous squad's sale value + bank
            spend = m.sum(self.price_of[p] * self.fh_sq[(p, w)] for p in pool)
            funds = m.sum(self.sale_value[p] * self._prev_sq(p, w) for p in pool)
            m.add(spend - funds - self._prev_itb(w) <= 0, f"fh_budget[{w}]")

    def _add_continuity_and_budget(self) -> None:
        m, pool = self.m, self.pool
        for w in self.weeks:
            for p in pool:
                m.add(
                    self.sq[(p, w)]
                    - self._prev_sq(p, w)
                    - self.inn[(p, w)]
                    + self._out_total(p, w)
                    == 0,
                    f"continuity[{p},{w}]",
                )
                m.add(
                    self.inn[(p, w)] + self._out_total(p, w) <= 1, f"no_churn[{p},{w}]"
                )
            sold = m.sum(
                [self.sell_first[p] * self.out_f[(p, w)] for p in self.sell_first]
                + [self.price_of[p] * self.out_r[(p, w)] for p in pool]
            )
            bought = m.sum(self.price_of[p] * self.inn[(p, w)] for p in pool)
            m.add(
                self.itb[w] - self._prev_itb(w) - sold + bought == 0, f"itb_flow[{w}]"
            )
        # first/regular sale split for price-modified originals (multiple-sell fix):
        # the first sale realizes the §1.8 sell price and can happen at most once; a
        # regular (current-price) sale requires an earlier first sale + rebuy.
        for p in self.sell_first:
            m.add(
                m.sum(self.out_f[(p, w)] for w in self.weeks) <= 1, f"first_sale_once[{p}]"
            )
            for w in self.weeks:
                earlier = m.sum(self.out_f[(p, ww)] for ww in self.weeks if ww <= w)
                m.add(self.out_r[(p, w)] - earlier <= 0, f"regular_after_first[{p},{w}]")

    def _add_transfer_counts(self) -> None:
        m, pool = self.m, self.pool
        cap = self.params.transfer_cap
        for w in self.weeks:
            m.add(
                self.count[w] - m.sum(self.inn[(p, w)] for p in pool) == 0,
                f"count_def[{w}]",
            )
            if self.initial_mode and w == self.start:
                continue  # GW1 semantics: unlimited transfers into the first squad
            # no transfers in FH weeks; ≤ cap per non-chip GW (WC exempt, §1.6)
            m.add(
                self.count[w] + SQUAD_SIZE * self.use_fh[w] <= SQUAD_SIZE,
                f"no_transfers_fh[{w}]",
            )
            m.add(
                self.count[w] - (SQUAD_SIZE - cap) * self.use_wc[w] <= cap,
                f"transfer_cap[{w}]",
            )
            # hits: penalized ≥ count − fts − M·use_wc (FH weeks have count 0). §3.2
            # writes M=15; M=14 is valid (count ≤ 15, fts ≥ 1) and tightens the LP.
            m.add(
                self.pen[w]
                - self.count[w]
                + self._fts_expr(w)
                + (SQUAD_SIZE - 1) * self.use_wc[w]
                >= 0,
                f"hits[{w}]",
            )
        for w in self.params.no_transfer_gws:
            if w in self.weeks and not (self.initial_mode and w == self.start):
                m.add(self.count[w] <= 0, f"no_transfer_gw[{w}]")

    def _add_ft_machine(self) -> None:
        """FT state machine (§3.2): one-hot states, big-M(=20) clamps to [1, cap].

        ``raw_ft[w] = fts[w] − eff[w] + 1`` where ``eff`` (the FT-consuming transfer
        count) equals ``transfer_count`` on plain weeks. On WC/FH weeks in the
        2024-25+ regime ``eff`` is only bounded (``count − 15 ≤ eff ≤ count``): the
        canonical value 0 (bank pass-through with +1 accrual per §1.6) is always
        weakly optimal because a higher FT state never hurts, so the *solution* is
        unaffected — and :meth:`extract` reports the trajectory from a
        ``rules.next_free_transfers`` replay, never from the (possibly degenerate
        when ``ft_value=0``) ``fts`` variables. Pinning ``eff`` to 0 with an extra
        row was tried and measurably wrecked HiGHS's search on realistic all-chips
        instances. In the pre-2024 regime WC/FH WIPE the bank, which does change
        feasibility, so there ``eff = fts[w]`` is linearized exactly. Then
        raw > cap ⇒ fts[w+1] = cap; raw ≤ 0 ⇒ fts[w+1] = 1; else fts[w+1] = raw —
        matching ``rules.next_free_transfers`` in both regimes.
        """
        m = self.m
        big = _FT_BIG_M
        cap = self.ft_cap
        n_states = cap + 1
        # one-hot linkage for every FT state variable (s ∈ 0..cap; §3.1 uses 0..5)
        for w in self.ft_var_weeks:
            states = [self.fts_state[(w, s)] for s in range(n_states)]
            m.add(m.sum(states) == 1, f"fts_onehot[{w}]")
            m.add(
                m.sum(s * self.fts_state[(w, s)] for s in range(1, n_states))
                - self.fts[w]
                == 0,
                f"fts_link[{w}]",
            )
        if self.initial_mode and len(self.weeks) >= 1:
            # after the unlimited GW1-style week the manager holds exactly 1 FT
            first_state = self.ft_var_weeks[0]
            m.add(self.fts[first_state] == 1, "fts_initial")
        for w in self.machine_weeks:
            nxt = self.end if w == self.weeks[-1] else w + 1
            chips = self.use_wc[w] + self.use_fh[w]
            eff = self.tc_eff[w]
            # eff = count on plain weeks (both regimes); M = 15 is valid (count <= 15,
            # eff >= 0 / = fts >= 1 on chip weeks) and tightens the LP relaxation
            m.add(eff - self.count[w] + SQUAD_SIZE * chips >= 0, f"ft_eff_count_lb[{w}]")
            if self.ft_wipe:
                # pre-2024: eff = fts on chip weeks (wipes the bank -> raw = 1)
                m.add(eff - self.count[w] - big * chips <= 0, f"ft_eff_count_ub[{w}]")
                fts_w = self._fts_expr(w)
                m.add(eff - fts_w + big * (1 - chips) >= 0, f"ft_eff_wipe_lb[{w}]")
                m.add(eff - fts_w - big * (1 - chips) <= 0, f"ft_eff_wipe_ub[{w}]")
            else:
                # 2024-25+: eff may drop to 0 on chip weeks (canonical: pass-through)
                m.add(eff - self.count[w] <= 0, f"ft_eff_count_ub[{w}]")
            raw = self._fts_expr(w) - eff + 1
            a, b = self.ft_above[w], self.ft_below[w]
            m.add(raw - big * a <= cap, f"ft_raw_ub[{w}]")
            m.add(raw + big * (1 - a) >= cap + 1, f"ft_above_iff[{w}]")
            m.add(raw + big * b >= 1, f"ft_raw_lb[{w}]")
            m.add(raw - big * (1 - b) <= 0, f"ft_below_iff[{w}]")
            m.add(a + b <= 1, f"ft_clamp_excl[{w}]")
            m.add(self.fts[nxt] - cap * a >= 0, f"ft_clamp_hi[{w}]")
            m.add(self.fts[nxt] - big * (1 - b) <= 1, f"ft_clamp_lo[{w}]")
            m.add(self.fts[nxt] - raw - big * (a + b) <= 0, f"ft_follow_ub[{w}]")
            m.add(self.fts[nxt] - raw + big * (a + b) >= 0, f"ft_follow_lb[{w}]")

    def _add_lineup(self) -> None:
        m, pool = self.m, self.pool
        for w in self.weeks:
            bb, fh = self.use_bb[w], self.use_fh[w]
            m.add(
                m.sum(self.lin[(p, w)] for p in pool) - 4 * bb == XI_SIZE,
                f"xi_size[{w}]",
            )
            for pos in rules.POSITIONS:
                members = [self.lin[(p, w)] for p in pool if self.pos_of[p] == pos]
                lo, hi, quota = FORMATION_MIN[pos], FORMATION_MAX[pos], POSITION_QUOTA[pos]
                m.add(
                    m.sum(members) - (quota - lo) * bb >= lo, f"formation_min[{pos},{w}]"
                )
                m.add(
                    m.sum(members) - (quota - hi) * bb <= hi, f"formation_max[{pos},{w}]"
                )
            # bench slots: exactly one GK in slot 0, one outfielder per slot 1-3 (no BB)
            gk_bench = [self.bench[k] for k in self.bench if k[1] == w and k[2] == 0]
            m.add(m.sum(gk_bench) + bb == 1, f"bench_gk[{w}]")
            for o in (1, 2, 3):
                slot = [self.bench[k] for k in self.bench if k[1] == w and k[2] == o]
                m.add(m.sum(slot) + bb == 1, f"bench_slot[{o},{w}]")
            # membership: picked (lineup or bench) once, and only from the active squad
            for p in pool:
                picked = self.lin[(p, w)] + m.sum(self._bench_of(p, w))
                m.add(picked <= 1, f"picked_once[{p},{w}]")
                m.add(picked - self.sq[(p, w)] - fh <= 0, f"picked_in_squad[{p},{w}]")
                if w in self.fh_weeks:
                    m.add(
                        picked - self.fh_sq[(p, w)] + fh <= 1,
                        f"picked_in_fh_squad[{p},{w}]",
                    )
                m.add(self.cap[(p, w)] - self.lin[(p, w)] <= 0, f"cap_in_xi[{p},{w}]")
                m.add(self.vic[(p, w)] - self.lin[(p, w)] <= 0, f"vice_in_xi[{p},{w}]")
                m.add(self.cap[(p, w)] + self.vic[(p, w)] <= 1, f"cap_ne_vice[{p},{w}]")
            m.add(m.sum(self.cap[(p, w)] for p in pool) == 1, f"one_captain[{w}]")
            m.add(m.sum(self.vic[(p, w)] for p in pool) == 1, f"one_vice[{w}]")

    def _add_chips(self) -> None:
        m, pool = self.m, self.pool
        use = {"wc": self.use_wc, "fh": self.use_fh, "bb": self.use_bb}
        for w in self.weeks:
            tc_sum = m.sum(self.use_tc[(p, w)] for p in pool if (p, w) in self.use_tc)
            m.add(
                self.use_wc[w] + self.use_fh[w] + self.use_bb[w] + tc_sum <= 1,
                f"one_chip[{w}]",
            )
            for kind in ("wc", "fh", "bb"):
                if w not in self.allowed[kind]:
                    m.add(use[kind][w] <= 0, f"chip_window[{kind},{w}]")
        for (p, w), var in self.use_tc.items():
            m.add(var - self.cap[(p, w)] <= 0, f"tc_is_captain[{p},{w}]")
        # per-instance count limits over the horizon (windows are disjoint 2025+; for
        # historical overlapping windows this is mildly loose — documented)
        for cid, ws in self.instance_weeks.items():
            kind = cid[:2]
            if kind == "tc":
                total = m.sum(
                    self.use_tc[(p, w)] for p in pool for w in ws if (p, w) in self.use_tc
                )
            else:
                total = m.sum(use[kind][w] for w in ws)
            m.add(total <= 1, f"chip_count[{cid}]")
        # an already-active chip this GW (e.g. a WC played pre-deadline) is committed
        if self.active_kind is not None:
            if self.active_kind == "tc":
                m.add(
                    m.sum(
                        self.use_tc[(p, self.start)]
                        for p in pool
                        if (p, self.start) in self.use_tc
                    )
                    == 1,
                    "active_chip",
                )
            else:
                m.add(use[self.active_kind][self.start] == 1, "active_chip")
        for gw, kind in self.forced.items():
            if kind == "tc":
                m.add(
                    m.sum(self.use_tc[(p, gw)] for p in pool if (p, gw) in self.use_tc) == 1,
                    f"forced_chip[{gw}]",
                )
            else:
                m.add(use[kind][gw] == 1, f"forced_chip[{gw}]")

    def _add_optional_hooks(self) -> None:
        m = self.m
        params = self.params
        overlap = params.banned_players & params.locked_players
        if overlap:
            raise ValueError(f"players both banned and locked: {sorted(overlap)}")
        for p in params.banned_players:
            for w in self.weeks:
                if (p, w) in self.sq:
                    m.add(self.sq[(p, w)] <= 0, f"banned[{p},{w}]")
                    m.add(self.inn[(p, w)] <= 0, f"banned_in[{p},{w}]")
                if (p, w) in self.fh_sq:
                    m.add(self.fh_sq[(p, w)] <= 0, f"banned_fh[{p},{w}]")
        for p in params.locked_players:
            if (p, self.start) not in self.sq:
                raise ValueError(f"locked player {p} is not in the player pool/prices")
            for w in self.weeks:
                m.add(self.sq[(p, w)] >= 1, f"locked[{p},{w}]")
        if params.max_hits_per_gw is not None:
            for w in self.machine_weeks:
                m.add(self.pen[w] <= params.max_hits_per_gw, f"max_hits[{w}]")
        if params.max_total_hits is not None:
            m.add(
                m.sum(self.pen[w] for w in self.machine_weeks) <= params.max_total_hits,
                "max_total_hits",
            )

    # -- objective (§3.3) --------------------------------------------------------------

    def _ft_values(self) -> list[float]:
        """Cumulative state values V(0..5) from the ft_value increments."""
        fv = self.params.ft_value
        inc = [float(fv)] * 5 if isinstance(fv, int | float) else [float(x) for x in fv]
        values = [0.0]
        for i in range(5):
            values.append(values[-1] + inc[i])
        return values

    def _vice_weight(self, w: int) -> float:
        """Per-week vice weight: presumptive captain's q0 when enabled, else the default."""
        if not self.params.use_q0_vice_weight or not self.q0_map:
            return self.params.vice_weight
        best = max(self.pool, key=lambda p: (self.xp_map.get((p, w), 0.0), -p))
        q0 = self.q0_map.get((best, w))
        return self.params.vice_weight if q0 is None else min(max(q0, 0.0), 1.0)

    def _bench_weight(self, w: int, o: int) -> float:
        per_gw = self.params.bench_weights_by_gw.get(w)
        if per_gw is not None and o in per_gw:
            return per_gw[o]
        return self.params.bench_weights.get(o, 0.0)

    def _build_objective(self) -> Any:
        m, params = self.m, self.params
        beta = params.decay
        itb_coef = params.itb_value / 10.0  # per £0.1m unit
        values = self._ft_values()
        terms: list[Any] = []
        for w in self.weeks:
            disc = beta ** (w - self.start)
            vw = self._vice_weight(w)
            for p in self.pool:
                xp = self.xp_map.get((p, w), 0.0)
                if xp == 0.0:
                    continue
                terms.append(disc * xp * self.lin[(p, w)])
                terms.append(disc * xp * self.cap[(p, w)])
                terms.append(disc * vw * xp * self.vic[(p, w)])
                tc = self.use_tc.get((p, w))
                if tc is not None:
                    terms.append(disc * xp * tc)
                for o in BENCH_SLOTS:
                    var = self.bench.get((p, w, o))
                    if var is not None:
                        bw = self._bench_weight(w, o)
                        if bw:
                            terms.append(disc * bw * xp * var)
            if w in self.pen:
                terms.append(-disc * params.hit_cost * self.pen[w])
            terms.append(disc * itb_coef * self.itb[w])
            # FT continuation value, incremental: disc · (V(fts[w+1]) − V(fts[w]))
            if w in self.machine_weeks:
                nxt = self.end if w == self.weeks[-1] else w + 1
                for s in range(self.ft_cap + 1):
                    if values[s]:
                        terms.append(disc * values[s] * self.fts_state[(nxt, s)])
                if w == self.start and self.state is not None:
                    pass  # V(fts[start]) is a constant; constants don't affect argmax
                else:
                    for s in range(self.ft_cap + 1):
                        if values[s]:
                            terms.append(-disc * values[s] * self.fts_state[(w, s)])
        return m.sum(terms)

    # -- warm start (guaranteed incumbent) ---------------------------------------------

    def _greedy_initial_squad(self) -> set[int] | None:
        """Cheapest quota-satisfying, club-legal 15 from the pool (None if greedy fails)."""
        chosen: set[int] = set()
        club_n: dict[int, int] = {}
        cost = 0
        for pos, quota in POSITION_QUOTA.items():
            group = sorted(
                (p for p in self.pool if self.pos_of[p] == pos),
                key=lambda p: (self.price_of[p], p),
            )
            taken = 0
            for p in group:
                club = self.club_of[p]
                if club_n.get(club, 0) >= CLUB_LIMIT:
                    continue
                chosen.add(p)
                club_n[club] = club_n.get(club, 0) + 1
                cost += self.price_of[p]
                taken += 1
                if taken == quota:
                    break
            if taken < quota:
                return None
        if cost > INITIAL_BUDGET:
            return None
        return chosen

    def _greedy_xi(self, own: set[int], w: int, *, bb: bool) -> list[int]:
        """A legal (formation-feasible) XI from ``own``, greedy by xp; all 15 under BB."""
        if bb:
            return sorted(own)
        by_pos: dict[str, list[int]] = {pos: [] for pos in rules.POSITIONS}
        for p in sorted(own, key=lambda p: (-self.xp_map.get((p, w), 0.0), p)):
            by_pos[self.pos_of[p]].append(p)
        xi: list[int] = []
        counts: dict[str, int] = {}
        for pos in rules.POSITIONS:
            take = FORMATION_MIN[pos]
            xi.extend(by_pos[pos][:take])
            counts[pos] = take
        leftovers = sorted(
            (
                p
                for pos in ("DEF", "MID", "FWD")
                for p in by_pos[pos][FORMATION_MIN[pos] :]
            ),
            key=lambda p: (-self.xp_map.get((p, w), 0.0), p),
        )
        for p in leftovers:
            if len(xi) == XI_SIZE:
                break
            pos = self.pos_of[p]
            if counts[pos] < FORMATION_MAX[pos]:
                xi.append(p)
                counts[pos] += 1
        return xi

    def warm_start(self) -> dict[Any, float] | None:
        """Build a trivially feasible MIP-start incumbent: the hold plan.

        Current squad kept (or a greedy cheapest legal 15 in initial mode), zero
        transfers, greedy XI/captaincy/bench each week, forced/active chips honoured,
        FT trajectory replayed via ``rules.next_free_transfers``. This guarantees the
        solver returns an incumbent even when the time limit bites before its own
        heuristics find one (the all-chips LP relaxation is weak, §3.5). Returns
        None (skipped) for setups the hold plan may violate: locked non-owned or
        banned owned players, an unaffordable forced/active FH re-pick, or an
        initial-mode pool with no greedy-feasible squad.
        """
        params = self.params
        if self.initial_mode:
            greedy = self._greedy_initial_squad()
            if greedy is None:
                return None
            own = greedy
            bank = float(INITIAL_BUDGET - sum(self.price_of[p] for p in own))
        else:
            assert self.state is not None
            own = set(self.own0)
            bank = float(self.state.bank)
        if params.locked_players - own or params.banned_players & own:
            return None

        chip_of: dict[int, str] = dict(self.forced)
        if self.active_kind is not None:
            chip_of[self.start] = self.active_kind
        for w, kind in chip_of.items():
            if kind == "fh" and not (self.initial_mode and w == self.start):
                spend = sum(self.price_of[p] for p in own)
                funds = sum(self.sale_value.get(p, self.price_of[p]) for p in own)
                if spend > funds + bank:
                    return None  # cannot re-pick the own squad under the FH budget

        assign: dict[Any, float] = {}
        for w in self.weeks:
            kind = chip_of.get(w)
            bb = kind == "bb"
            for p in own:
                assign[self.sq[(p, w)]] = 1.0
                if kind == "fh":
                    assign[self.fh_sq[(p, w)]] = 1.0
            xi = self._greedy_xi(own, w, bb=bb)
            for p in xi:
                assign[self.lin[(p, w)]] = 1.0
            ranked = sorted(xi, key=lambda p: (-self.xp_map.get((p, w), 0.0), p))
            cap_p, vice_p = ranked[0], ranked[1]
            assign[self.cap[(cap_p, w)]] = 1.0
            assign[self.vic[(vice_p, w)]] = 1.0
            if not bb:
                bench = sorted(own - set(xi))
                gk_bench = next(p for p in bench if self.pos_of[p] == "GKP")
                outfield = sorted(
                    (p for p in bench if p != gk_bench),
                    key=lambda p: (-self.xp_map.get((p, w), 0.0), p),
                )
                assign[self.bench[(gk_bench, w, 0)]] = 1.0
                for slot, p in enumerate(outfield, start=1):
                    assign[self.bench[(p, w, slot)]] = 1.0
            if kind == "wc":
                assign[self.use_wc[w]] = 1.0
            elif kind == "fh":
                assign[self.use_fh[w]] = 1.0
            elif kind == "bb":
                assign[self.use_bb[w]] = 1.0
            elif kind == "tc":
                assign[self.use_tc[(cap_p, w)]] = 1.0
            assign[self.itb[w]] = bank
        if self.initial_mode:
            for p in own:
                assign[self.inn[(p, self.start)]] = 1.0
            assign[self.count[self.start]] = float(len(own))

        # FT trajectory replay (zero transfers; chips per chip_of)
        cap = self.ft_cap
        fts_val: dict[int, int] = {}
        f = 1 if self.state is None else self.state.free_transfers
        for w in self.machine_weeks:
            chipflag = chip_of.get(w) in ("wc", "fh")
            eff = float(f) if (self.ft_wipe and chipflag) else 0.0
            raw = f - eff + 1
            assign[self.ft_above[w]] = 1.0 if raw >= cap + 1 else 0.0
            assign[self.ft_below[w]] = 1.0 if raw <= 0 else 0.0
            assign[self.tc_eff[w]] = eff
            f = rules.next_free_transfers(f, 0, self.season, chipflag)
            nxt = self.end if w == self.weeks[-1] else w + 1
            fts_val[nxt] = f
        for w in self.ft_var_weeks:
            v = fts_val.get(w, 1)  # unset states only occur where fts is pinned to 1
            assign[self.fts[w]] = float(v)
            assign[self.fts_state[(w, v)]] = 1.0
        return assign

    # -- solve & extraction ------------------------------------------------------------

    def solve(self) -> SolveDiagnostics:
        """Run HiGHS on the built model, guaranteeing an incumbent where one exists.

        Two phases: a COLD solve first (HiGHS's feasibility-jump/diving heuristics
        only hunt for a first incumbent when none was supplied, and measurably find
        far better ones than a seeded search on realistic all-chips instances), then
        — only if the time limit expired with no feasible incumbent — a short warm
        re-solve seeded with the hold plan (:meth:`warm_start`), so callers always
        get a legal plan back instead of a RuntimeError. A slice of the time budget
        (≤ 10s, ≤ 20%) is reserved for the fallback whenever a hold plan exists.
        """
        params = self.params
        try:
            start = self.warm_start()
        except Exception:  # defensive: a warm-start edge case must never kill a solve
            logger.exception("warm-start construction failed; no incumbent fallback")
            start = None
        reserve = min(10.0, 0.2 * params.time_limit_s) if start is not None else 0.0
        diag = self.m.solve(
            self._objective,
            time_limit_s=params.time_limit_s - reserve,
            mip_rel_gap=params.mip_rel_gap,
            seed=params.seed,
            threads=params.threads,
        )
        if diag.has_solution or start is None or diag.status == "infeasible":
            return diag
        logger.info(
            "no incumbent within %.1fs; re-solving warm-started with the hold plan",
            params.time_limit_s - reserve,
        )
        diag2 = self.m.solve(
            self._objective,
            time_limit_s=reserve,
            mip_rel_gap=params.mip_rel_gap,
            seed=params.seed,
            threads=params.threads,
            start=start,
        )
        return SolveDiagnostics(
            status=diag2.status,
            has_solution=diag2.has_solution,
            objective=diag2.objective,
            gap=diag2.gap,
            seconds=diag.seconds + diag2.seconds,
        )

    def _on(self, var: Any) -> bool:
        return self.m.value(var) > 0.5

    def _chip_played(self, w: int) -> str | None:
        if self._on(self.use_wc[w]):
            kind = "wc"
        elif self._on(self.use_fh[w]):
            kind = "fh"
        elif self._on(self.use_bb[w]):
            kind = "bb"
        elif any(
            self._on(self.use_tc[(p, w)]) for p in self.pool if (p, w) in self.use_tc
        ):
            kind = "tc"
        else:
            return None
        if w == self.start and self.active_kind == kind and self.state is not None:
            return self.state.active_chip
        avail = [cid for cid in self.instance_weeks if cid.startswith(kind)]
        return chip_instance_for(self.season, kind, w, avail)

    def extract(self, diag: SolveDiagnostics) -> PlanResult:
        """Turn the incumbent solution into a :class:`PlanResult`.

        ``free_transfers`` is reported from a ``rules.next_free_transfers`` replay of
        the extracted transfers/chips (see ``_add_ft_machine``: the model's ``fts``
        variables can sit below the canonical trajectory on WC/FH weeks when
        ``ft_value=0`` makes them objective-neutral).
        """
        pool = self.pool
        gws: list[GwPlan] = []
        ft_replay: int | None = (
            None if self.initial_mode else self.state.free_transfers  # type: ignore[union-attr]
        )
        for w in self.weeks:
            fh_active = w in self.fh_weeks and self._on(self.use_fh[w])
            squad_vars = self.fh_sq if fh_active else self.sq
            squad = sorted(p for p in pool if self._on(squad_vars[(p, w)]))
            lineup = sorted(p for p in pool if self._on(self.lin[(p, w)]))
            bench_order = [
                p
                for o in BENCH_SLOTS
                for p in pool
                if (p, w, o) in self.bench and self._on(self.bench[(p, w, o)])
            ]
            captain = next(p for p in pool if self._on(self.cap[(p, w)]))
            vice = next(p for p in pool if self._on(self.vic[(p, w)]))
            tc_extra = sum(
                self.xp_map.get((p, w), 0.0)
                for p in pool
                if (p, w) in self.use_tc and self._on(self.use_tc[(p, w)])
            )
            expected = (
                sum(self.xp_map.get((p, w), 0.0) for p in lineup)
                + self.xp_map.get((captain, w), 0.0)
                + tc_extra
            )
            pen_val = round(self.m.value(self.pen[w])) if w in self.pen else 0
            transfers_in = sorted(p for p in pool if self._on(self.inn[(p, w)]))
            chip = self._chip_played(w)
            gws.append(
                GwPlan(
                    gw=w,
                    squad=squad,
                    lineup=lineup,
                    bench_order=bench_order,
                    captain=captain,
                    vice=vice,
                    transfers_in=transfers_in,
                    transfers_out=sorted(
                        p
                        for p in pool
                        if self._on(self.out_r[(p, w)])
                        or ((p, w) in self.out_f and self._on(self.out_f[(p, w)]))
                    ),
                    hit_points=int(round(self.params.hit_cost * pen_val)),
                    chip=chip,
                    expected_points=expected,
                    bank=round(self.m.value(self.itb[w])),
                    free_transfers=ft_replay,
                )
            )
            if ft_replay is None:  # unlimited initial-build week -> 1 FT next
                ft_replay = 1
            else:
                ft_replay = rules.next_free_transfers(
                    ft_replay,
                    len(transfers_in),
                    self.season,
                    (chip or "")[:2] in ("wc", "fh"),
                )
        return PlanResult(
            objective=diag.objective,
            gws=gws,
            solve_seconds=diag.seconds,
            gap=diag.gap,
            status=diag.status,
            n_variables=self.m.n_variables,
            n_constraints=self.m.n_constraints,
        )


# --------------------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------------------

_INFEASIBLE_GROUPS = [
    "squad-composition (size 15, quotas 2/5/5/3)",
    "club-limit (≤3 per club)",
    "budget / in-the-bank ≥ 0 (check purchase & current prices)",
    "transfer continuity & first-sale split",
    "FT state machine (free_transfers must be 1-5)",
    "chip windows / forced or active chips",
    "locked/banned players",
    "max-hits / no-transfer restrictions",
]


def solve_plan(
    xp: pd.DataFrame,
    prices: pd.DataFrame,
    state: SquadState | None,
    *,
    horizon: int = 8,
    params: SolveParams | None = None,
) -> PlanResult:
    """Solve the multi-GW transfer/lineup/chip plan MILP (MODEL_DESIGN_INPUTS §3).

    Args:
        xp: Per-GW predictions per the ``models/assemble`` contract: columns
            ``season, gw, player_code, xp`` plus an optional ``q0`` column
            (P(0 minutes), used for vice weighting).
        prices: One row per player: ``player_code, price, position, team_code``
            (price in £0.1m units; position in GKP/DEF/MID/FWD).
        state: The manager's :class:`SquadState`, or ``None`` for initial-squad
            mode (GW1 semantics: £100.0m budget, unlimited transfers into the first
            horizon GW, all chips available per their windows).
        horizon: Number of GWs to plan (clipped at GW38).
        params: :class:`SolveParams`; defaults follow §3.3/§3.5.

    Returns:
        A :class:`PlanResult` with the per-GW plan and solver diagnostics.

    Raises:
        InfeasiblePlanError: when the state cannot form a legal squad or the MILP is
            infeasible; the message lists the constraint groups to check.
        ValueError: on malformed inputs.
        RuntimeError: when the time limit expires with no feasible incumbent.
    """
    params = params if params is not None else SolveParams()
    price_of, pos_of, club_of = _index_prices(prices)

    if state is not None:
        season, start = state.season, state.current_gw
        problems = validate_state(state, pos_of, price_of)
        if problems:
            raise InfeasiblePlanError(
                "SquadState cannot form a legal plan; violated constraint groups:", problems
            )
    else:
        seasons = sorted(set(xp["season"].astype(int)))
        if len(seasons) != 1:
            raise ValueError(f"xp frame must cover exactly one season, got {seasons}")
        season = seasons[0]
        start = int(xp["gw"].min())

    weeks = list(range(start, min(start + horizon, LAST_GW + 1)))
    if not weeks:
        raise ValueError(f"empty horizon: start={start}, horizon={horizon}")
    xp_map, q0_map = _index_xp(xp, season, weeks)

    owned_codes = {p.player_code for p in state.squad} if state is not None else set()
    must_keep = owned_codes | params.locked_players
    for p in params.locked_players:
        if p not in price_of:
            raise ValueError(f"locked player {p} not in prices frame")
    candidates = sorted(
        (set(price_of) & ({p for (p, _w) in xp_map} | must_keep)) - (
            params.banned_players - owned_codes
        )
    )
    if params.prune:
        score = {
            p: sum(
                params.decay ** (w - start) * xp_map.get((p, w), 0.0) for w in weeks
            )
            for p in candidates
        }
        pool = _prune_pool(candidates, score, price_of, pos_of, must_keep, params)
    else:
        pool = candidates
    if not pool:
        raise ValueError("empty player pool: no xp rows match the horizon/prices")

    model = _PlanModel(
        pool=pool,
        weeks=weeks,
        season=season,
        state=state,
        params=params,
        price_of=price_of,
        pos_of=pos_of,
        club_of=club_of,
        xp_map=xp_map,
        q0_map=q0_map,
    )
    model.build()
    diag = model.solve()
    if diag.status == "infeasible" or (
        not diag.has_solution and diag.status not in ("time_limit",)
    ):
        raise InfeasiblePlanError(
            f"MILP infeasible (status={diag.status}); constraint groups to check:",
            _INFEASIBLE_GROUPS,
        )
    if not diag.has_solution:
        raise RuntimeError(
            f"time limit {params.time_limit_s}s expired with no feasible incumbent; "
            "raise time_limit_s or prune harder"
        )
    logger.info(
        "solved: status=%s objective=%.3f gap=%.2e in %.2fs",
        diag.status,
        diag.objective,
        diag.gap,
        diag.seconds,
    )
    return model.extract(diag)
