"""Squad optimizer (stage 3): multi-GW MILP planning on HiGHS.

Public surface per ``docs/ARCHITECTURE.md``: :class:`SquadState`/:func:`from_entry`
(state reconstruction), :func:`solve_plan` (the core MILP), plus the parameter/result
models. ``chips``, ``sensitivity``, ``autosubs`` and ``plans`` build on these.
"""

from fplai.optimizer.milp import (
    GwPlan,
    HighsModel,
    InfeasiblePlanError,
    PlanResult,
    SolveParams,
    solve_plan,
)
from fplai.optimizer.state import (
    ChipId,
    OwnedPlayer,
    SquadState,
    all_chip_ids,
    chip_instance_for,
    from_entry,
)

__all__ = [
    "ChipId",
    "GwPlan",
    "HighsModel",
    "InfeasiblePlanError",
    "OwnedPlayer",
    "PlanResult",
    "SolveParams",
    "SquadState",
    "all_chip_ids",
    "chip_instance_for",
    "from_entry",
    "solve_plan",
]
