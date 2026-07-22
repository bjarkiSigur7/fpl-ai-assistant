"""Per-GW chip bans: the banned_chip_gws hook and the None-state first-GW guard.

A Free Hit (or Wildcard) at the first GW of a None-state initial build is degenerate —
there is no prior squad to revert to and transfers into the build are already unlimited —
so the MILP never allows transfer chips there (and an explicit force raises).
"""

from typing import Any

import pandas as pd
import pytest

from fplai.optimizer.milp import InfeasiblePlanError, SolveParams, solve_plan
from fplai.optimizer.plans import TransferPair, _transfer_bullet
from fplai.optimizer.state import OwnedPlayer, SquadState

SEASON = 2025

# 2 GKP / 5 DEF / 5 MID / 3 FWD, all distinct clubs where it matters, cheap.
POOL: list[tuple[int, str, int, int]] = [
    (1, "GKP", 45, 1),
    (2, "GKP", 40, 2),
    (11, "DEF", 45, 3),
    (12, "DEF", 45, 4),
    (13, "DEF", 45, 5),
    (14, "DEF", 40, 6),
    (15, "DEF", 40, 7),
    (21, "MID", 60, 8),
    (22, "MID", 60, 9),
    (23, "MID", 55, 10),
    (24, "MID", 50, 11),
    (25, "MID", 50, 12),
    (31, "FWD", 65, 13),
    (32, "FWD", 60, 14),
    (33, "FWD", 55, 15),
]


def _prices() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"player_code": c, "position": pos, "price": price, "team_code": club}
            for c, pos, price, club in POOL
        ]
    )


def _xp(gws: list[int], per_player: dict[int, float] | None = None) -> pd.DataFrame:
    per_player = per_player or {c: 3.0 for c, *_ in POOL}
    return pd.DataFrame(
        [
            {"season": SEASON, "gw": gw, "player_code": c, "xp": v}
            for gw in gws
            for c, v in per_player.items()
        ]
    )


def _params(**overrides: Any) -> SolveParams:
    defaults: dict[str, Any] = {"time_limit_s": 30.0, "mip_rel_gap": 1e-6}
    defaults.update(overrides)
    return SolveParams(**defaults)


def test_none_state_never_plays_transfer_chip_at_first_gw() -> None:
    plan = solve_plan(_xp([30, 31, 32]), _prices(), None, horizon=3, params=_params())
    assert plan.gws[0].chip not in ("wc1", "wc2", "fh1", "fh2")


def test_none_state_forced_fh_at_first_gw_raises() -> None:
    with pytest.raises(InfeasiblePlanError):
        solve_plan(
            _xp([30, 31, 32]),
            _prices(),
            None,
            horizon=3,
            params=_params(forced_chips={30: "fh"}),
        )


def test_none_state_fh_still_allowed_after_first_gw() -> None:
    plan = solve_plan(
        _xp([30, 31, 32]),
        _prices(),
        None,
        horizon=3,
        params=_params(forced_chips={31: "fh"}),
    )
    assert plan.gws[1].chip in ("fh1", "fh2")


def test_banned_chip_gws_shifts_bench_boost() -> None:
    state = SquadState(
        season=SEASON,
        current_gw=30,
        squad=[
            OwnedPlayer(player_code=c, purchase_price=p, current_price=p)
            for c, _pos, p, _club in POOL
        ],
        bank=0,
        free_transfers=1,
        chips_available=["bb2"],
    )
    xp = _xp([30, 31, 32])
    free = solve_plan(xp, _prices(), state, horizon=3, params=_params())
    assert any(g.chip == "bb2" for g in free.gws)  # uniform xp: BB is free value somewhere
    banned = solve_plan(
        xp,
        _prices(),
        state,
        horizon=3,
        params=_params(banned_chip_gws={30: frozenset({"bb"})}),
    )
    assert banned.gws[0].chip != "bb2"


def test_draft_bullet_wording_for_initial_squad() -> None:
    pair = TransferPair(
        player_in=21,
        player_in_name="B.Fernandes",
        player_out=None,
        player_out_name=None,
        position="MID",
        xp_in=31.0,
        xp_out=0.0,
        xp_delta=31.0,
        out_q0=None,
        support_pct=97.0,
    )
    text = _transfer_bullet(pair, pd.DataFrame(columns=["season", "gw", "player_code", "xp"]), 30)
    assert text.startswith("Draft B.Fernandes")
    assert "(none)" not in text
