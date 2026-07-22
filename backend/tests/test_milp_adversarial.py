"""Adversarial verification tests for the stage-3 optimizer (milp/autosubs/chips seam).

Written by the verifier pass. Probes, per the verification brief:

(a) the FT state machine against ``rules.next_free_transfers`` — a 200-seed property
    test over randomized 8-GW usage sequences on the *transcribed constraint system*
    (kept in sync with ``milp._add_ft_machine``), plus real solves whose reported FT
    trajectories must match a rules replay (including with ``ft_value=0``, which used
    to leave the machine under-constrained, and the pre-2024 wipe regime);
(b) §1.8 sell-price arithmetic with exact bank evolution;
(c) DGW handling: per-fixture xp rows sum within the GW and captaining doubles the sum;
(d) chip-window enforcement (set-1 expiry, set-2 opening, TC⊆captain, ≤1 chip/GW);
(e) a greedy-vs-optimal instance where FT banking beats the myopic move;
(f) a by-hand objective audit (decay, hits, bench weights, FT value, ITB);
(g) autosub Monte Carlo convergence against exact enumeration;
plus a regression test for the ``banned_chips`` chips.py <-> milp.py coordination field.

Everything is offline, deterministic and hand-checkable.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
import pytest

from fplai import rules
from fplai.optimizer import (
    InfeasiblePlanError,
    OwnedPlayer,
    PlanResult,
    SolveParams,
    SquadState,
    solve_plan,
)
from fplai.optimizer.autosubs import bench_weights_mc
from fplai.optimizer.chips import chip_ev_curves

SEASON = 2025

# --------------------------------------------------------------------------------------
# Instance builders (kept local: tests are not a package)
# --------------------------------------------------------------------------------------

#: The standard 15-man squad: (code, position, price, club) — distinct clubs 1-15.
STD_SQUAD: list[tuple[int, str, int, int]] = [
    (101, "GKP", 45, 1),
    (102, "GKP", 40, 2),
    (201, "DEF", 45, 3),
    (202, "DEF", 45, 4),
    (203, "DEF", 50, 5),
    (204, "DEF", 50, 6),
    (205, "DEF", 40, 7),
    (301, "MID", 60, 8),
    (302, "MID", 65, 9),
    (303, "MID", 55, 10),
    (304, "MID", 50, 11),
    (305, "MID", 50, 12),
    (401, "FWD", 70, 13),
    (402, "FWD", 60, 14),
    (403, "FWD", 45, 15),
]


def make_prices(players: list[tuple[int, str, int, int]]) -> pd.DataFrame:
    """Prices frame from (code, position, price, club) tuples."""
    return pd.DataFrame(
        [
            {"player_code": c, "position": pos, "price": price, "team_code": club}
            for c, pos, price, club in players
        ]
    )


def make_xp(
    xp_by_gw: dict[int, dict[int, float]], *, season: int = SEASON
) -> pd.DataFrame:
    """xp frame from {gw: {player_code: xp}}."""
    return pd.DataFrame(
        [
            {"season": season, "gw": gw, "player_code": code, "xp": xp}
            for gw, per_player in xp_by_gw.items()
            for code, xp in per_player.items()
        ]
    )


def make_state(
    players: list[tuple[int, str, int, int]],
    *,
    season: int = SEASON,
    gw: int = 10,
    bank: int = 0,
    fts: int = 1,
    chips: list[str] | None = None,
    purchase_override: dict[int, int] | None = None,
) -> SquadState:
    """SquadState owning ``players`` at purchase == current price unless overridden."""
    overrides = purchase_override or {}
    return SquadState(
        season=season,
        current_gw=gw,
        squad=[
            OwnedPlayer(
                player_code=c, purchase_price=overrides.get(c, price), current_price=price
            )
            for c, _pos, price, _club in players
        ],
        bank=bank,
        free_transfers=fts,
        chips_available=chips or [],
    )


def tiny_params(**overrides: Any) -> SolveParams:
    """Deterministic fast params for hand-checked instances."""
    defaults: dict[str, Any] = {"time_limit_s": 30.0, "mip_rel_gap": 1e-9}
    defaults.update(overrides)
    return SolveParams(**defaults)


def replay_ft_trajectory(plan: PlanResult, state: SquadState) -> None:
    """Assert the plan's reported FT states match a rules.next_free_transfers replay."""
    for prev, nxt in itertools.pairwise(plan.gws):
        assert prev.free_transfers is not None and nxt.free_transfers is not None
        wc_fh = (prev.chip or "")[:2] in ("wc", "fh")
        expected = rules.next_free_transfers(
            prev.free_transfers, len(prev.transfers_in), state.season, wc_fh
        )
        assert nxt.free_transfers == expected, (
            f"GW{nxt.gw}: reported {nxt.free_transfers}, rules say {expected}"
        )


# ======================================================================================
# (a) FT state machine vs rules.next_free_transfers — property test, 200 seeds
# ======================================================================================


def _feasible_next_states(fts: int, count: int, chip: bool, season: int) -> set[int]:
    """All next-FT values feasible under the milp FT constraint system.

    This transcribes the inequalities of ``milp._PlanModel._add_ft_machine`` verbatim
    (KEEP IN SYNC when editing that method) and enumerates every feasible
    ``(eff, a, b, next)`` assignment. The machine is correct iff, for every reachable
    input, the CANONICAL rules value is the maximum of the feasible set (a higher FT
    state is always weakly better, and extraction replays the rules), and the set is
    a singleton everywhere the game state is actually pinned — i.e. everywhere except
    2024-25+ WC/FH weeks, where the model deliberately leaves ``eff`` slack.
    """
    flags = rules.SEASON_FLAGS[season]
    cap = flags.max_free_transfers
    wipe = flags.chips_wipe_banked_fts
    big = 20
    squad_size = 15
    chips = 1 if chip else 0
    feasible: set[int] = set()
    for eff in range(0, squad_size + 1):
        if not (eff - count + squad_size * chips >= 0):
            continue
        if wipe:
            if not (eff - count - big * chips <= 0):
                continue
            if not (eff - fts + big * (1 - chips) >= 0):
                continue
            if not (eff - fts - big * (1 - chips) <= 0):
                continue
        else:
            if not (eff - count <= 0):
                continue
        raw = fts - eff + 1
        for a, b in itertools.product((0, 1), repeat=2):
            if not (raw - big * a <= cap):
                continue
            if not (raw + big * (1 - a) >= cap + 1):
                continue
            if not (raw + big * b >= 1):
                continue
            if not (raw - big * (1 - b) <= 0):
                continue
            if a + b > 1:
                continue
            for nxt in range(1, cap + 1):
                if not (nxt - cap * a >= 0):
                    continue
                if not (nxt - big * (1 - b) <= 1):
                    continue
                if not (nxt - raw - big * (a + b) <= 0):
                    continue
                if not (nxt - raw + big * (a + b) >= 0):
                    continue
                feasible.add(nxt)
    return feasible


@pytest.mark.parametrize("seed", range(200))
def test_ft_machine_matches_rules_on_random_sequences(seed: int) -> None:
    """8-GW random usage sequences: constraint system == rules replay, both regimes."""
    rng = np.random.default_rng(seed)
    season = int(rng.choice([2026, 2025, 2024, 2022, 2019]))
    flags = rules.SEASON_FLAGS[season]
    cap = flags.max_free_transfers
    fts = int(rng.integers(1, cap + 1))
    for _ in range(8):
        count = int(rng.integers(0, 16)) if rng.random() < 0.3 else int(rng.integers(0, 4))
        chip = bool(rng.random() < 0.25)
        expected = rules.next_free_transfers(fts, count, season, chip)
        feasible = _feasible_next_states(fts, count, chip, season)
        label = f"season={season} fts={fts} count={count} chip={chip}"
        assert feasible, f"{label}: constraint system infeasible"
        assert max(feasible) == expected, (
            f"{label}: canonical next state {max(feasible)} != rules {expected}"
        )
        if not (chip and not flags.chips_wipe_banked_fts):
            # everywhere the game state is pinned, the machine admits exactly one value
            assert feasible == {expected}, (
                f"{label}: constraints admit {sorted(feasible)}, rules say {expected}"
            )
        fts = expected


def test_ft_trajectory_pinned_even_with_zero_ft_value() -> None:
    """ft_value=0 removes the objective pressure on the fts variables — the REPORTED
    trajectory must still be rules-exact (extraction replays rules.next_free_transfers;
    regression: it used to read the then-degenerate fts variables directly)."""
    upgrades = [(501, "MID", 50, 16), (502, "MID", 50, 17)]
    pool = STD_SQUAD + upgrades
    prices = make_prices(pool)
    xp_by_gw: dict[int, dict[int, float]] = {10: {}, 11: {}, 12: {}}
    for i, (c, *_rest) in enumerate(STD_SQUAD):
        for w in (10, 11, 12):
            xp_by_gw[w][c] = 3.0 + i * 0.01
    for w in (10, 11, 12):
        xp_by_gw[w][304] = 0.0
        xp_by_gw[w][305] = 0.0
        xp_by_gw[w][501] = 8.0
        xp_by_gw[w][502] = 8.0
    state = make_state(STD_SQUAD, fts=2, chips=["wc1"])
    plan = solve_plan(
        make_xp(xp_by_gw),
        prices,
        state,
        horizon=3,
        params=tiny_params(ft_value=0.0, itb_value=0.0, forced_chips={10: "wc"}),
    )
    assert plan.gws[0].chip == "wc1"
    replay_ft_trajectory(plan, state)
    # WC passes the bank through with +1 accrual: 2 -> 3 regardless of WC transfers
    assert plan.gws[1].free_transfers == 3


def test_pre_2024_regime_wildcard_wipes_bank_and_cap_is_two() -> None:
    """Season 2022: FT bank caps at 2 and a WC week resets the bank to exactly 1."""
    season = 2022
    prices = make_prices(STD_SQUAD)
    xp_by_gw: dict[int, dict[int, float]] = {}
    for w in (10, 11, 12, 13):
        xp_by_gw[w] = {c: 2.0 + i * 0.01 for i, (c, *_rest) in enumerate(STD_SQUAD)}
    state = make_state(STD_SQUAD, season=season, fts=2, chips=["wc1"])
    plan = solve_plan(
        make_xp(xp_by_gw, season=season),
        prices,
        state,
        horizon=4,
        params=tiny_params(forced_chips={11: "wc"}),
    )
    assert plan.gws[1].chip == "wc1"
    replay_ft_trajectory(plan, state)
    # 2 (hold: capped at 2, not 3) -> WC wipes -> 1 -> 2
    assert [g.free_transfers for g in plan.gws] == [2, 2, 1, 2]


def test_state_ft_above_season_cap_rejected() -> None:
    """A 2022 state claiming 4 banked FTs (cap 2) is flagged, not silently clamped."""
    state = make_state(STD_SQUAD, season=2022, fts=4)
    xp_of = {c: 2.0 for c, *_ in STD_SQUAD}
    with pytest.raises(InfeasiblePlanError, match="free-transfers"):
        solve_plan(
            make_xp({10: xp_of}, season=2022),
            make_prices(STD_SQUAD),
            state,
            horizon=1,
            params=tiny_params(),
        )


# ======================================================================================
# (b) Sell-price arithmetic and bank evolution
# ======================================================================================


def test_sell_price_rise_bank_evolution() -> None:
    """Bought at 50, now 56 -> sells at 53 (half profit floored); bank tracks exactly.

    Every other owned MID is cheapened so no alternative sale can fund either target:
    the only route to the 53-priced target is realizing 305's §1.8 sell price, and the
    54-priced decoy is affordable only under the WRONG (now-price 56) model.
    """
    mid_price = {301: 48, 302: 46, 303: 45, 304: 44, 305: 56}
    squad = [
        (c, pos, mid_price.get(c, price), club) for c, pos, price, club in STD_SQUAD
    ]
    pool = squad + [(501, "MID", 53, 16), (502, "MID", 54, 17)]
    prices = make_prices(pool)
    state = make_state(squad, bank=0, purchase_override={305: 50})
    xp_of = {c: 3.0 + i * 0.01 for i, (c, *_rest) in enumerate(STD_SQUAD)}
    xp_of[305] = 0.0
    xp_of[501] = 15.0
    xp_of[502] = 20.0  # better — but 54 > sell(305) = 53 + bank 0: unaffordable
    plan = solve_plan(
        make_xp({10: xp_of}),
        prices,
        state,
        horizon=1,
        params=tiny_params(max_hits_per_gw=0, itb_value=0.0),
    )
    g = plan.gws[0]
    assert g.transfers_out == [305]
    assert g.transfers_in == [501]  # 502 only fits under the wrong now-price sale model
    assert g.bank == 0  # sold 53 (NOT 56), bought 53: bank unchanged
    assert g.hit_points == 0


def test_sell_price_fall_borne_in_full() -> None:
    """Bought at 50, now 44 -> sells at 44 (full fall borne); bank tracks exactly.

    The 45-priced decoy is affordable only under a purchase-price (50) sale model —
    the correct model sells at the fallen price 44.
    """
    mid_price = {301: 43, 302: 42, 303: 41, 304: 40, 305: 44}
    squad = [
        (c, pos, mid_price.get(c, price), club) for c, pos, price, club in STD_SQUAD
    ]
    pool = squad + [(501, "MID", 44, 16), (502, "MID", 45, 17)]
    prices = make_prices(pool)
    state = make_state(squad, bank=0, purchase_override={305: 50})
    xp_of = {c: 3.0 + i * 0.01 for i, (c, *_rest) in enumerate(STD_SQUAD)}
    xp_of[305] = 0.0
    xp_of[501] = 15.0
    xp_of[502] = 20.0  # better — but 45 > sell(305) = 44: unaffordable
    plan = solve_plan(
        make_xp({10: xp_of}),
        prices,
        state,
        horizon=1,
        params=tiny_params(max_hits_per_gw=0, itb_value=0.0),
    )
    g = plan.gws[0]
    assert g.transfers_out == [305]
    assert g.transfers_in == [501]
    assert g.bank == 0  # sold 44 (no purchase-price refund), bought 44
    assert g.hit_points == 0


# ======================================================================================
# (c) DGW: per-fixture rows sum; captaining a DGW player doubles the sum
# ======================================================================================


def test_dgw_per_fixture_rows_sum_and_captain_doubles() -> None:
    prices = make_prices(STD_SQUAD)
    rows = []
    for i, (c, *_rest) in enumerate(STD_SQUAD):
        rows.append({"season": SEASON, "gw": 10, "player_code": c, "xp": 2.0 + i * 0.01})
    # player 301 has TWO fixtures in GW10: 5.0 + 4.0 must combine to 9.0
    rows.append({"season": SEASON, "gw": 10, "player_code": 301, "xp": 4.0})
    xp = pd.DataFrame(rows)
    xp.loc[(xp.player_code == 301) & (xp.xp != 4.0), "xp"] = 5.0
    state = make_state(STD_SQUAD)
    params = tiny_params(
        ft_value=0.0,
        itb_value=0.0,
        vice_weight=0.0,
        use_q0_vice_weight=False,
        bench_weights={0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0},
    )
    plan = solve_plan(xp, prices, state, horizon=1, params=params)
    g = plan.gws[0]
    assert g.captain == 301
    xp_of = {c: 2.0 + i * 0.01 for i, (c, *_rest) in enumerate(STD_SQUAD)}
    xp_of[301] = 9.0  # the summed DGW value
    # best legal XI given one GK and the zero bench weights: drop 102 (GK2) and the
    # three cheapest-xp outfielders (201, 202, 203 have the lowest values after 101)
    best_xi_xp = max(
        sum(xp_of[p] for p in xi)
        for xi in itertools.combinations([c for c, *_ in STD_SQUAD], 11)
        if _legal_xi(xi)
    )
    assert plan.objective == pytest.approx(best_xi_xp + 9.0, abs=1e-6)
    assert g.expected_points == pytest.approx(best_xi_xp + 9.0, abs=1e-6)


def _legal_xi(xi: Sequence[int]) -> bool:
    pos_of = {c: pos for c, pos, *_ in STD_SQUAD}
    counts: dict[str, int] = {}
    for p in xi:
        counts[pos_of[p]] = counts.get(pos_of[p], 0) + 1
    return (
        counts.get("GKP", 0) == 1
        and 3 <= counts.get("DEF", 0) <= 5
        and 2 <= counts.get("MID", 0) <= 5
        and 1 <= counts.get("FWD", 0) <= 3
    )


# ======================================================================================
# (d) Chip-window enforcement
# ======================================================================================


def _flat_xp(weeks: Sequence[int]) -> pd.DataFrame:
    xp_by_gw: dict[int, dict[int, float]] = {}
    for w in weeks:
        xp_by_gw[w] = {c: 5.0 + i * 0.01 for i, (c, *_rest) in enumerate(STD_SQUAD)}
    return make_xp(xp_by_gw)


def test_set1_chip_dead_after_expiry_gw() -> None:
    """bb1 (window GW1-19) at GW20+: never played despite a huge incentive; forcing raises."""
    state = make_state(STD_SQUAD, gw=20, chips=["bb1"])
    xp = _flat_xp([20, 21])
    plan = solve_plan(xp, make_prices(STD_SQUAD), state, horizon=2, params=tiny_params())
    assert all(g.chip is None for g in plan.gws)
    with pytest.raises(InfeasiblePlanError, match="chip"):
        solve_plan(
            xp,
            make_prices(STD_SQUAD),
            state,
            horizon=2,
            params=tiny_params(forced_chips={20: "bb"}),
        )


def test_set2_chip_not_playable_before_gw20() -> None:
    """bb2 (window GW20-38) with GWs 18-20 in horizon: only playable at 20."""
    state = make_state(STD_SQUAD, gw=18, chips=["bb2"])
    xp = _flat_xp([18, 19, 20])
    plan = solve_plan(
        xp, make_prices(STD_SQUAD), state, horizon=3, params=tiny_params()
    )
    assert plan.gws[0].chip is None and plan.gws[1].chip is None
    assert plan.gws[2].chip == "bb2"  # equal xp every week -> discounting alone would
    # favour GW18; only the window constraint pushes it to GW20
    with pytest.raises(InfeasiblePlanError, match="chip"):
        solve_plan(
            xp,
            make_prices(STD_SQUAD),
            state,
            horizon=3,
            params=tiny_params(forced_chips={18: "bb"}),
        )


def test_tc_lands_exactly_on_the_captain() -> None:
    """A forced TC triples the captain (never any other player): xP arithmetic proves it."""
    state = make_state(STD_SQUAD, chips=["tc1"])
    xp_of = {c: 2.0 + i * 0.01 for i, (c, *_rest) in enumerate(STD_SQUAD)}
    xp_of[301] = 9.0
    plan = solve_plan(
        make_xp({10: xp_of}),
        make_prices(STD_SQUAD),
        state,
        horizon=1,
        params=tiny_params(forced_chips={10: "tc"}),
    )
    g = plan.gws[0]
    assert g.chip == "tc1"
    assert g.captain == 301
    xi_xp = sum(xp_of[p] for p in g.lineup)
    # XI + captain double + TC extra — the extra equals the CAPTAIN's xp exactly
    assert g.expected_points == pytest.approx(xi_xp + 2 * 9.0, abs=1e-9)


def test_at_most_one_chip_per_gw_even_when_both_look_great() -> None:
    """Huge bench AND huge captain in one GW: BB and TC cannot stack."""
    bench4 = {102, 205, 305, 403}
    xp_of: dict[int, float] = {}
    for i, (c, *_rest) in enumerate(STD_SQUAD):
        xp_of[c] = 20.0 if c in bench4 else 2.0 + i * 0.01
    xp_of[301] = 30.0  # monster captain
    state = make_state(STD_SQUAD, chips=["bb1", "tc1"])
    plan = solve_plan(
        make_xp({10: xp_of}),
        make_prices(STD_SQUAD),
        state,
        horizon=1,
        params=tiny_params(),
    )
    g = plan.gws[0]
    assert g.chip in ("bb1", "tc1")
    if g.chip == "bb1":
        # all 15 score + captain double, NO additional TC tripling
        assert g.expected_points == pytest.approx(sum(xp_of.values()) + 30.0, abs=1e-9)
    else:
        xi_xp = sum(xp_of[p] for p in g.lineup)
        assert g.expected_points == pytest.approx(xi_xp + 2 * 30.0, abs=1e-9)


# ======================================================================================
# (e) Greedy-vs-optimal: banking FTs for a monster DGW beats the myopic move
# ======================================================================================


#: Bench weights for the banking instance: slot 3 bumped so buying a 0-xp monster
#: early (parking him on the bench) is STRICTLY worse than banking the FT — this
#: breaks the spread-the-buys/bank-the-FTs degeneracy without touching the maths
#: under test (greedy myopia vs multi-GW planning).
_BANK_BENCH_WEIGHTS = {0: 0.03, 1: 0.21, 2: 0.06, 3: 0.05}


def _banking_instance() -> tuple[pd.DataFrame, pd.DataFrame, SquadState]:
    extras = [
        (501, "MID", 50, 16),  # C: the myopic upgrade on the weakest starter
        (601, "MID", 50, 17),  # the three GW12 DGW monsters
        (602, "MID", 50, 18),
        (603, "FWD", 45, 19),
    ]
    pool = STD_SQUAD + extras
    prices = make_prices(pool)
    xp_by_gw: dict[int, dict[int, float]] = {10: {}, 11: {}, 12: {}}
    fodder = {304, 305, 403}  # weak-but-playing fodder the monsters will replace
    for i, (c, *_rest) in enumerate(STD_SQUAD):
        for w in (10, 11, 12):
            if c == 102:
                xp_by_gw[w][c] = 0.0
            elif c in fodder:
                xp_by_gw[w][c] = 2.0
            else:
                xp_by_gw[w][c] = 3.0 + i * 0.01
    for w in (10, 11, 12):
        xp_by_gw[w][501] = 3.3  # beats the weakest 3.0x starter a little, every week
        for c in (601, 602, 603):
            xp_by_gw[w][c] = 12.0 if w == 12 else 0.0
    state = make_state(STD_SQUAD, fts=1)
    return make_xp(xp_by_gw), prices, state


def test_greedy_single_gw_takes_the_small_upgrade() -> None:
    xp, prices, state = _banking_instance()
    greedy = solve_plan(
        xp[xp.gw == 10],
        prices,
        state,
        horizon=1,
        params=tiny_params(
            ft_value=0.0, itb_value=0.0, bench_weights=_BANK_BENCH_WEIGHTS
        ),
    )
    assert greedy.gws[0].transfers_in == [501]  # myopically correct...


def test_milp_banks_fts_for_the_monster_dgw() -> None:
    """...but the 3-GW MILP holds, banks to 3 FTs and lands all three DGW monsters."""
    xp, prices, state = _banking_instance()
    plan = solve_plan(
        xp,
        prices,
        state,
        horizon=3,
        params=tiny_params(
            ft_value=0.0, itb_value=0.0, bench_weights=_BANK_BENCH_WEIGHTS
        ),
    )
    assert plan.gws[0].transfers_in == []  # holds despite the positive myopic move
    assert plan.gws[1].transfers_in == []
    assert sorted(plan.gws[2].transfers_in) == [601, 602, 603]
    assert plan.gws[2].free_transfers == 3
    assert sum(g.hit_points for g in plan.gws) == 0  # all three came free
    assert not any(501 in g.squad for g in plan.gws)  # the myopic buy never happens
    replay_ft_trajectory(plan, state)


# ======================================================================================
# (f) Objective audit: recompute the reported objective from the returned plan
# ======================================================================================


def _audit_objective(
    plan: PlanResult,
    xp_map: Mapping[tuple[int, int], float],
    params: SolveParams,
    state: SquadState,
) -> float:
    """Recompute the §3.3 objective from the extracted plan, by hand."""
    beta = params.decay
    ft_value = float(params.ft_value) if isinstance(params.ft_value, int | float) else None
    assert ft_value is not None, "audit only supports scalar ft_value"

    def v(s: int) -> float:
        return ft_value * s

    total = 0.0
    fts_seq: list[int] = [g.free_transfers or 0 for g in plan.gws]
    last = plan.gws[-1]
    fts_seq.append(
        rules.next_free_transfers(
            last.free_transfers or 0,
            len(last.transfers_in),
            state.season,
            (last.chip or "")[:2] in ("wc", "fh"),
        )
    )
    for i, g in enumerate(plan.gws):
        disc = beta**i
        xi = sum(xp_map[(p, g.gw)] for p in g.lineup)
        cap_xp = xp_map[(g.captain, g.gw)]
        vice_xp = params.vice_weight * xp_map[(g.vice, g.gw)]
        tc_xp = cap_xp if (g.chip or "").startswith("tc") else 0.0
        bench = sum(
            params.bench_weights[slot] * xp_map[(p, g.gw)]
            for slot, p in enumerate(g.bench_order)
        )
        itb = params.itb_value / 10.0 * g.bank
        total += disc * (xi + cap_xp + vice_xp + tc_xp + bench - g.hit_points + itb)
        # FT continuation increment: disc * (V(fts[w+1]) - V(fts[w])), the V(fts[start])
        # term being a constant the solver drops
        total += disc * v(fts_seq[i + 1])
        if i > 0:
            total -= disc * v(fts_seq[i])
    return total


def test_objective_audit_transfers_hit_itb_ft_value() -> None:
    """Full-fat defaults (decay/bench/FT-value/ITB) + a hit: audit matches to 1e-4."""
    extras = [(501, "MID", 48, 16), (502, "FWD", 58, 17)]
    pool = STD_SQUAD + extras
    prices = make_prices(pool)
    xp_by_gw: dict[int, dict[int, float]] = {10: {}, 11: {}}
    for i, (c, *_rest) in enumerate(STD_SQUAD):
        xp_by_gw[10][c] = 2.0 + i * 0.01
        xp_by_gw[11][c] = 2.0 + i * 0.01
    xp_by_gw[10][304] = 0.0
    xp_by_gw[11][304] = 0.0
    xp_by_gw[10][402] = 0.0
    xp_by_gw[11][402] = 0.0
    for w in (10, 11):
        xp_by_gw[w][501] = 12.0  # worth a hit alongside the free 502 upgrade
        xp_by_gw[w][502] = 11.0
    xp = make_xp(xp_by_gw)
    state = make_state(STD_SQUAD, fts=1, bank=3)
    params = tiny_params(use_q0_vice_weight=False)  # everything else at §3.3 defaults
    plan = solve_plan(xp, prices, state, horizon=2, params=params)
    assert sorted(plan.gws[0].transfers_in) == [501, 502]
    assert plan.gws[0].hit_points == 4
    xp_map = {
        (int(r.player_code), int(r.gw)): float(r.xp) for r in xp.itertuples()
    }
    audited = _audit_objective(plan, xp_map, params, state)
    assert plan.objective == pytest.approx(audited, abs=1e-4)


def test_objective_audit_with_tc_chip() -> None:
    """Audit across a TC week (per-player chip weighting enters the objective)."""
    prices = make_prices(STD_SQUAD)
    xp_by_gw: dict[int, dict[int, float]] = {10: {}, 11: {}}
    for i, (c, *_rest) in enumerate(STD_SQUAD):
        xp_by_gw[10][c] = 2.0 + i * 0.01
        xp_by_gw[11][c] = 2.5 + i * 0.01
    xp_by_gw[11][301] = 13.0
    xp = make_xp(xp_by_gw)
    state = make_state(STD_SQUAD, fts=2, bank=7, chips=["tc1"])
    params = tiny_params(use_q0_vice_weight=False)
    plan = solve_plan(xp, prices, state, horizon=2, params=params)
    assert plan.gws[1].chip == "tc1" and plan.gws[1].captain == 301
    xp_map = {
        (int(r.player_code), int(r.gw)): float(r.xp) for r in xp.itertuples()
    }
    audited = _audit_objective(plan, xp_map, params, state)
    assert plan.objective == pytest.approx(audited, abs=1e-4)


# ======================================================================================
# (g) Autosub Monte Carlo convergence vs exact enumeration (3 risky starters)
# ======================================================================================


def _exact_autosub_probs(
    lineup_q0: dict[int, float],
    bench_q0: dict[int, float],
    positions: dict[int, str],
    bench_order: list[int],
) -> np.ndarray:
    """Exact per-slot P(slot scores | bench player appears) by full enumeration.

    Independent re-implementation of the FPL_KNOWLEDGE §1.5 algorithm (bench priority
    order, GK-for-GK only, XI list must keep DEF>=3 / MID>=2 / FWD>=1 / GKP==1).
    """
    starters = list(lineup_q0)
    gk_start = next(c for c in starters if positions[c] == "GKP")
    gk_bench = next(c for c in bench_q0 if positions[c] == "GKP")
    slots = [gk_bench, *bench_order]
    p_score = np.zeros(4)
    risky = [c for c in starters if lineup_q0[c] > 0.0]
    for absent_set in itertools.chain.from_iterable(
        itertools.combinations(risky, k) for k in range(len(risky) + 1)
    ):
        p_abs = 1.0
        for c in starters:
            q = lineup_q0[c]
            p_abs *= q if c in absent_set else (1.0 - q)
        if p_abs == 0.0:
            continue
        for plays_mask in itertools.product([False, True], repeat=4):
            p_bench = 1.0
            for slot_code, plays in zip(slots, plays_mask, strict=True):
                q = bench_q0[slot_code]
                p_bench *= (1.0 - q) if plays else q
            if p_bench == 0.0:
                continue
            prob = p_abs * p_bench
            # simulate FPL autosubs on the XI *list* (nominal positions)
            xi = list(starters)
            unfilled = [c for c in absent_set]
            used = [False] * 4
            # GK slot: GK-for-GK only
            if gk_start in unfilled and plays_mask[0]:
                xi.remove(gk_start)
                xi.append(gk_bench)
                unfilled.remove(gk_start)
                used[0] = True
            for slot in (1, 2, 3):
                if not plays_mask[slot]:
                    continue
                code = slots[slot]
                for absent in list(unfilled):
                    if positions[absent] == "GKP":
                        continue  # only the bench GK covers the GK
                    trial = [c for c in xi if c != absent] + [code]
                    counts: dict[str, int] = {}
                    for c in trial:
                        counts[positions[c]] = counts.get(positions[c], 0) + 1
                    if (
                        counts.get("GKP", 0) == 1
                        and counts.get("DEF", 0) >= 3
                        and counts.get("MID", 0) >= 2
                        and counts.get("FWD", 0) >= 1
                    ):
                        xi = trial
                        unfilled.remove(absent)
                        used[slot] = True
                        break
            for slot in range(4):
                if used[slot]:
                    p_score[slot] += prob
    appears = np.array([1.0 - bench_q0[c] for c in slots])
    return p_score / appears


def test_autosub_mc_converges_to_exact_enumeration() -> None:
    """n=20000 MC within CI (3.5 sigma < 0.02) of exact probabilities, 3+GK risky case."""
    positions = {
        1: "GKP", 2: "DEF", 3: "DEF", 4: "DEF", 5: "DEF",
        6: "MID", 7: "MID", 8: "MID", 9: "MID", 10: "FWD", 11: "FWD",
        12: "GKP", 13: "DEF", 14: "MID", 15: "FWD",
    }
    lineup_q0 = {
        1: 0.15, 2: 0.4, 3: 0.0, 4: 0.0, 5: 0.0,
        6: 0.5, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.3, 11: 0.0,
    }
    bench_q0 = {12: 0.1, 13: 0.2, 14: 0.1, 15: 0.25}
    formation = {"GKP": 1, "DEF": 4, "MID": 4, "FWD": 2}
    bench_order = [13, 14, 15]
    exact = _exact_autosub_probs(lineup_q0, bench_q0, positions, bench_order)
    mc = bench_weights_mc(
        lineup_q0,
        bench_q0,
        formation,
        positions=positions,
        bench_order=bench_order,
        n=20000,
        seed=123,
    )
    assert exact[0] == pytest.approx(0.15, abs=1e-12)  # GK slot is analytic
    assert np.all(exact[1:] > 0.0)  # the case genuinely exercises every slot
    np.testing.assert_allclose(mc, exact, atol=0.02)


# ======================================================================================
# banned_chips: the chips.py <-> milp.py coordination field must actually bind
# ======================================================================================


def test_banned_chips_param_is_enforced() -> None:
    bench4 = {102, 205, 305, 403}
    xp_of = {
        c: 15.0 if c in bench4 else 2.0 + i * 0.01
        for i, (c, *_rest) in enumerate(STD_SQUAD)
    }
    xp = make_xp({10: xp_of})
    prices = make_prices(STD_SQUAD)
    state = make_state(STD_SQUAD, chips=["bb1"])
    free = solve_plan(xp, prices, state, horizon=1, params=tiny_params())
    assert free.gws[0].chip == "bb1"  # the bench is huge: BB is optimal when allowed
    banned = solve_plan(
        xp, prices, state, horizon=1, params=tiny_params(banned_chips=frozenset({"bb1"}))
    )
    assert banned.gws[0].chip is None
    assert banned.objective < free.objective - 1.0
    by_kind = solve_plan(
        xp, prices, state, horizon=1, params=tiny_params(banned_chips=frozenset({"bb"}))
    )
    assert by_kind.gws[0].chip is None
    with pytest.raises(ValueError, match="unknown"):
        solve_plan(
            xp, prices, state, horizon=1, params=tiny_params(banned_chips=frozenset({"xx9"}))
        )


def test_chip_ev_curves_baseline_is_truly_chipless() -> None:
    """End-to-end chips.py -> milp.py: the baseline bans every chip, so a monster-bench
    BB week shows up as a POSITIVE delta (was ~0 when banned_chips was ignored)."""
    bench4 = {102, 205, 305, 403}
    xp_by_gw: dict[int, dict[int, float]] = {}
    for w in (10, 11):
        xp_by_gw[w] = {
            c: (10.0 if w == 11 else 1.0) if c in bench4 else 3.0 + i * 0.01
            for i, (c, *_rest) in enumerate(STD_SQUAD)
        }
    xp = make_xp(xp_by_gw)
    prices = make_prices(STD_SQUAD)
    state = make_state(STD_SQUAD, chips=["bb1"])
    curves = chip_ev_curves(
        xp,
        prices,
        state,
        ["bb1"],
        [10, 11],
        horizon=2,
        params=tiny_params(),
        skip_dominated=False,
    )
    evaluated = curves[curves["evaluated"]].set_index("gw")
    assert set(evaluated.index) == {10, 11}
    # baseline never plays a chip, so forcing BB on the loaded week must show a clearly
    # positive delta (the 4 weakest squad players' xp, ~12, discounted and net of the
    # forgone bench weights) — before the banned_chips fix the baseline itself played
    # BB and the delta collapsed to ~0
    assert evaluated.loc[11, "delta_vs_no_chip"] > 5.0
    assert evaluated.loc[11, "delta_vs_no_chip"] > evaluated.loc[10, "delta_vs_no_chip"]
