"""Tests for the multi-GW MILP optimizer (fplai.optimizer.milp / .state).

All solver tests are tiny hand-solvable instances (5-8 players per position, 1-3 GWs)
with planted optima, plus one realistic ~840-player instance built from real 2025-26
prices with synthetic (seeded) xP. Everything is offline and deterministic except the
one ``@pytest.mark.live`` test.
"""

from __future__ import annotations

import itertools
import time
from typing import Any

import numpy as np
import pandas as pd
import pytest

from fplai import config, rules
from fplai.optimizer import (
    InfeasiblePlanError,
    OwnedPlayer,
    PlanResult,
    SolveParams,
    SquadState,
    solve_plan,
)

SEASON = 2025

# --------------------------------------------------------------------------------------
# Instance builders
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


def make_xp(xp_by_gw: dict[int, dict[int, float]], q0: float | None = None) -> pd.DataFrame:
    """xp frame from {gw: {player_code: xp}} (season fixed)."""
    rows = []
    for gw, per_player in xp_by_gw.items():
        for code, xp in per_player.items():
            row = {"season": SEASON, "gw": gw, "player_code": code, "xp": xp}
            if q0 is not None:
                row["q0"] = q0
            rows.append(row)
    return pd.DataFrame(rows)


def make_state(
    players: list[tuple[int, str, int, int]],
    *,
    gw: int = 10,
    bank: int = 0,
    fts: int = 1,
    chips: list[str] | None = None,
    purchase_override: dict[int, int] | None = None,
) -> SquadState:
    """SquadState owning ``players`` at purchase == current price unless overridden."""
    overrides = purchase_override or {}
    return SquadState(
        season=SEASON,
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


def assert_plan_legal(plan: PlanResult, prices: pd.DataFrame) -> None:
    """Squad-legality invariants every plan must satisfy (FPL_KNOWLEDGE §1.3)."""
    pos_of = dict(zip(prices.player_code, prices.position, strict=True))
    club_of = dict(zip(prices.player_code, prices.team_code, strict=True))
    for g in plan.gws:
        assert len(g.squad) == 15
        counts: dict[str, int] = {}
        clubs: dict[int, int] = {}
        for p in g.squad:
            counts[pos_of[p]] = counts.get(pos_of[p], 0) + 1
            clubs[club_of[p]] = clubs.get(club_of[p], 0) + 1
        assert counts == {"GKP": 2, "DEF": 5, "MID": 5, "FWD": 3}, f"GW{g.gw}: {counts}"
        assert max(clubs.values()) <= 3, f"GW{g.gw}: club counts {clubs}"
        assert set(g.lineup) <= set(g.squad)
        expected_xi = 15 if g.chip in ("bb1", "bb2") else 11
        assert len(g.lineup) == expected_xi
        xi_counts: dict[str, int] = {}
        for p in g.lineup:
            xi_counts[pos_of[p]] = xi_counts.get(pos_of[p], 0) + 1
        if expected_xi == 11:
            assert xi_counts["GKP"] == 1
            assert 3 <= xi_counts["DEF"] <= 5
            assert 2 <= xi_counts["MID"] <= 5
            assert 1 <= xi_counts["FWD"] <= 3
        assert g.captain in g.lineup and g.vice in g.lineup
        assert g.captain != g.vice
        assert g.bank >= 0
        assert set(g.bench_order) == set(g.squad) - set(g.lineup)


def assert_ft_trajectory(plan: PlanResult, state: SquadState) -> None:
    """The plan's FT states must match a rules.next_free_transfers replay exactly."""
    for prev, nxt in itertools.pairwise(plan.gws):
        assert prev.free_transfers is not None and nxt.free_transfers is not None
        wc_fh = (prev.chip or "")[:2] in ("wc", "fh")
        expected = rules.next_free_transfers(
            prev.free_transfers, len(prev.transfers_in), state.season, wc_fh
        )
        assert nxt.free_transfers == expected, f"GW{nxt.gw}: {nxt.free_transfers} != {expected}"


# --------------------------------------------------------------------------------------
# Core correctness: XI/captain/bench versus brute-force enumeration
# --------------------------------------------------------------------------------------


def _brute_force_single_gw(
    xp_of: dict[int, float], pos_of: dict[int, str]
) -> tuple[float, frozenset[int], int, list[int]]:
    """Enumerate every legal XI/captain/vice/bench-order for a fixed 15-man squad."""
    codes = sorted(xp_of)
    best: tuple[float, frozenset[int], int, list[int]] | None = None
    for xi in itertools.combinations(codes, 11):
        counts: dict[str, int] = {}
        for p in xi:
            counts[pos_of[p]] = counts.get(pos_of[p], 0) + 1
        if counts.get("GKP", 0) != 1:
            continue
        if not (3 <= counts.get("DEF", 0) <= 5):
            continue
        if not (2 <= counts.get("MID", 0) <= 5):
            continue
        if not (1 <= counts.get("FWD", 0) <= 3):
            continue
        xi_xp = sorted((xp_of[p] for p in xi), reverse=True)
        bench = [p for p in codes if p not in xi]
        bench_gk = next(p for p in bench if pos_of[p] == "GKP")
        outfield = sorted(
            (p for p in bench if p != bench_gk), key=lambda p: -xp_of[p]
        )
        score = (
            sum(xi_xp)
            + xi_xp[0]  # captain doubles
            + 0.1 * xi_xp[1]  # vice weight
            + 0.03 * xp_of[bench_gk]
            + 0.21 * xp_of[outfield[0]]
            + 0.06 * xp_of[outfield[1]]
            + 0.002 * xp_of[outfield[2]]
        )
        if best is None or score > best[0]:
            cap = max(xi, key=lambda p: xp_of[p])
            best = (score, frozenset(xi), cap, [bench_gk, *outfield])
    assert best is not None
    return best


def test_xi_captain_bench_match_bruteforce() -> None:
    rng = np.random.default_rng(7)
    xp_of = {c: round(float(x), 3) for (c, *_), x in zip(STD_SQUAD, rng.uniform(1, 10, 15))}
    assert len(set(xp_of.values())) == 15  # unique optimum
    prices = make_prices(STD_SQUAD)
    state = make_state(STD_SQUAD)
    params = tiny_params(ft_value=0.0, itb_value=0.0, use_q0_vice_weight=False)
    plan = solve_plan(make_xp({10: xp_of}), prices, state, horizon=1, params=params)
    assert_plan_legal(plan, prices)

    pos_of = {c: pos for c, pos, *_ in STD_SQUAD}
    best_obj, best_xi, best_cap, best_bench = _brute_force_single_gw(xp_of, pos_of)
    g = plan.gws[0]
    assert plan.objective == pytest.approx(best_obj, abs=1e-6)
    assert frozenset(g.lineup) == best_xi
    assert g.captain == best_cap
    assert g.bench_order == best_bench
    assert g.transfers_in == [] and g.hit_points == 0


# --------------------------------------------------------------------------------------
# Transfers, sell-price arithmetic, hits, FT banking
# --------------------------------------------------------------------------------------


def test_must_transfer_respects_sell_price_budget() -> None:
    """Star is affordable only via the §1.8 sell price; decoy only via (wrong) now-price."""
    # cheapen the other MIDs so no alternative sale can fund either target
    mid_price = {301: 50, 302: 48, 303: 45, 304: 44, 305: 56}
    squad = [
        (c, pos, mid_price.get(c, price), club) for c, pos, price, club in STD_SQUAD
    ]
    pool = squad + [(501, "MID", 55, 16), (502, "MID", 58, 17)]
    prices = make_prices(pool)
    state = make_state(
        squad,
        bank=2,
        purchase_override={305: 50},  # bought 50, now 56 -> sell = 50 + (56-50)//2 = 53
    )
    xp_of = {c: 2.0 + i * 0.01 for i, (c, *_rest) in enumerate(STD_SQUAD)}
    xp_of[305] = 0.5
    xp_of[501] = 20.0
    xp_of[502] = 25.0
    plan = solve_plan(
        make_xp({10: xp_of}),
        prices,
        state,
        horizon=1,
        # 1 FT + no hits allowed -> exactly one transfer: isolates the budget arithmetic
        params=tiny_params(use_q0_vice_weight=False, max_hits_per_gw=0),
    )
    assert_plan_legal(plan, prices)
    g = plan.gws[0]
    # funds = sell(305)=53 + bank 2 = 55: star (55) affordable, decoy (58, higher xp) is
    # NOT — a model pricing the sale at now-cost 56 would wrongly afford it (56+2=58)
    assert g.transfers_in == [501]
    assert g.transfers_out == [305]
    assert g.bank == 0
    assert g.hit_points == 0
    assert g.captain == 501


def test_two_transfer_week_with_one_ft_costs_exactly_four() -> None:
    xi_xp = {
        101: 4.0, 201: 4.1, 202: 4.2, 203: 4.3, 204: 4.4,
        301: 30.0, 302: 4.5, 303: 4.6, 401: 4.8, 402: 0.0, 304: 0.0,
        102: 0.0, 205: 0.0, 305: 0.0, 403: 0.0,  # permanent dead-weight bench
    }
    pool = STD_SQUAD + [(501, "MID", 50, 16), (502, "FWD", 60, 17)]
    prices = make_prices(pool)
    xp_of = dict(xi_xp)
    xp_of[501] = 10.0  # upgrade on 304 (price-equal)
    xp_of[502] = 5.0  # upgrade on 402 (price-equal)
    xp = make_xp({10: xp_of})
    state = make_state(STD_SQUAD, fts=1)
    base = {"ft_value": 0.0, "itb_value": 0.0, "use_q0_vice_weight": False}

    no_hit = solve_plan(xp, prices, state, horizon=1, params=tiny_params(**base, max_hits_per_gw=0))
    free = solve_plan(xp, prices, state, horizon=1, params=tiny_params(**base))
    assert no_hit.gws[0].transfers_in == [501]  # only the better upgrade fits 1 FT
    assert no_hit.gws[0].hit_points == 0
    assert sorted(free.gws[0].transfers_in) == [501, 502]
    assert free.gws[0].hit_points == 4
    # the extra transfer adds xp 5 and costs exactly one -4 hit
    assert free.objective - no_hit.objective == pytest.approx(5.0 - 4.0, abs=1e-6)


def test_no_transfer_last_gws_tail_guard() -> None:
    """The tail guard bans transfers in the horizon's final N GWs (edge-churn fix)."""
    pool = STD_SQUAD + [(501, "MID", 50, 16)]
    prices = make_prices(pool)
    flat = {c: 2.0 for c, _p, _pr, _cl in STD_SQUAD}
    # A juicy upgrade appears only in the LAST horizon GW (11): without the guard
    # the solver buys 501 there; with no_transfer_last_gws=1 it must not.
    xp = make_xp({10: dict(flat), 11: {**flat, 501: 12.0, 304: 0.0}})
    state = make_state(STD_SQUAD, fts=2)
    base = {"ft_value": 0.0, "itb_value": 0.0, "use_q0_vice_weight": False}

    free = solve_plan(xp, prices, state, horizon=2, params=tiny_params(**base))
    assert 501 in free.gws[1].transfers_in

    guarded = solve_plan(
        xp, prices, state, horizon=2, params=tiny_params(**base, no_transfer_last_gws=1)
    )
    assert guarded.gws[1].transfers_in == []
    assert_plan_legal(guarded, prices)


def test_ft_banking_holds_transfer_for_future_value() -> None:
    """With one FT and all value next week, banking beats moving now (decay + FT value)."""
    pool = STD_SQUAD + [(501, "MID", 50, 16), (502, "MID", 50, 17)]
    prices = make_prices(pool)
    xp_by_gw: dict[int, dict[int, float]] = {10: {}, 11: {}}
    for i, (c, *_rest) in enumerate(STD_SQUAD):
        steady = 3.0 + i * 0.01
        xp_by_gw[10][c] = steady
        xp_by_gw[11][c] = steady
    for w in (10, 11):
        xp_by_gw[w][304] = 0.0
        xp_by_gw[w][305] = 0.0
    xp_by_gw[10][501] = 0.0
    xp_by_gw[10][502] = 0.0
    xp_by_gw[11][501] = 6.0
    xp_by_gw[11][502] = 6.0
    state = make_state(STD_SQUAD, fts=1)
    plan = solve_plan(
        make_xp(xp_by_gw), prices, state, horizon=2, params=tiny_params(itb_value=0.0)
    )
    assert_plan_legal(plan, prices)
    assert plan.gws[0].transfers_in == []  # held: banked the FT
    assert sorted(plan.gws[1].transfers_in) == [501, 502]
    assert len(plan.gws[1].transfers_out) == 2
    assert plan.gws[0].hit_points == 0 and plan.gws[1].hit_points == 0
    assert plan.gws[1].free_transfers == 2  # banked
    assert_ft_trajectory(plan, state)


# --------------------------------------------------------------------------------------
# Squad-construction constraints (initial mode): club limit and quotas bind
# --------------------------------------------------------------------------------------


def test_club_limit_and_position_quotas_bind() -> None:
    players: list[tuple[int, str, int, int]] = []
    club = 1
    for i in range(4):
        players.append((100 + i, "GKP", 50, club))
        club += 1
    for i in range(8):
        players.append((200 + i, "DEF", 50, club))
        club += 1
    stars = [601, 602, 603, 604]  # 4 great same-club MIDs -> only 3 can be squadded
    for code in stars:
        players.append((code, "MID", 50, 50))
    for i in range(4):
        players.append((300 + i, "MID", 50, club))
        club += 1
    for i in range(6):  # 6 great FWDs -> quota keeps exactly 3
        players.append((400 + i, "FWD", 50, club))
        club += 1
    prices = make_prices(players)
    xp_of = {c: 2.0 + (c % 10) * 0.01 for c, *_ in players}
    for code in stars:
        xp_of[code] = 10.0
    for c, pos, *_rest in players:
        if pos == "FWD":
            xp_of[c] = 9.0 + (c % 6) * 0.1
    plan = solve_plan(
        make_xp({1: xp_of}), prices, None, horizon=1, params=tiny_params()
    )
    assert_plan_legal(plan, prices)
    g = plan.gws[0]
    assert sum(1 for p in g.squad if p in stars) == 3  # club limit binds
    club_of = dict(zip(prices.player_code, prices.team_code, strict=True))
    assert sum(1 for p in g.squad if club_of[p] == 50) == 3
    assert g.bank == 1000 - 15 * 50
    assert g.free_transfers is None  # unlimited initial build


def test_infeasible_16_man_squad_raises_clear_error() -> None:
    squad16 = STD_SQUAD + [(501, "MID", 50, 16)]
    prices = make_prices(squad16)
    state = make_state(squad16)
    xp_of = {c: 2.0 for c, *_ in squad16}
    with pytest.raises(InfeasiblePlanError) as err:
        solve_plan(make_xp({10: xp_of}), prices, state, horizon=1, params=tiny_params())
    msg = str(err.value)
    assert "squad-size" in msg and "16" in msg and "15" in msg


# --------------------------------------------------------------------------------------
# Chips
# --------------------------------------------------------------------------------------


def test_wildcard_many_transfers_no_hits_ft_bank_preserved() -> None:
    upgrades = [
        (501, "MID", 60, 16), (502, "MID", 65, 17), (503, "MID", 55, 18),
        (511, "DEF", 45, 19), (512, "DEF", 45, 20), (513, "DEF", 50, 21),
        (521, "FWD", 70, 22), (522, "FWD", 60, 23),
    ]
    pool = STD_SQUAD + upgrades
    prices = make_prices(pool)
    xp_by_gw: dict[int, dict[int, float]] = {10: {}, 11: {}}
    for i, (c, *_rest) in enumerate(STD_SQUAD):
        xp_by_gw[10][c] = 1.0 + i * 0.001
        xp_by_gw[11][c] = 1.0 + i * 0.001
    for c, *_rest in upgrades:
        xp_by_gw[10][c] = 6.0
        xp_by_gw[11][c] = 6.0
    state = make_state(STD_SQUAD, fts=1, chips=["wc1"])
    plan = solve_plan(make_xp(xp_by_gw), prices, state, horizon=2, params=tiny_params())
    assert_plan_legal(plan, prices)
    g0, g1 = plan.gws
    assert g0.chip == "wc1"
    assert len(g0.transfers_in) == 8  # every planted upgrade squadded
    assert g0.hit_points == 0 and g1.hit_points == 0
    assert g1.free_transfers == 2  # WC passes the FT bank through with +1 accrual
    # money-bank consistency: purchase == current for all owned, so sales realize price
    price_of = dict(zip(prices.player_code, prices.price, strict=True))
    expected_bank = sum(price_of[p] for p in g0.transfers_out) - sum(
        price_of[p] for p in g0.transfers_in
    )
    assert g0.bank == expected_bank >= 0
    assert g1.bank == g0.bank
    assert g1.chip is None and g1.transfers_in == []
    assert_ft_trajectory(plan, state)


def test_free_hit_squad_reverts_next_gw() -> None:
    owned = [(c, pos, 62, club) for c, pos, _pr, club in STD_SQUAD]
    stars = [
        (601, "GKP", 60, 21), (602, "GKP", 60, 22),
        (611, "DEF", 60, 23), (612, "DEF", 60, 24), (613, "DEF", 60, 25),
        (614, "DEF", 60, 26), (615, "DEF", 60, 27),
        (621, "MID", 60, 28), (622, "MID", 60, 29), (623, "MID", 60, 30),
        (624, "MID", 60, 31), (625, "MID", 60, 32),
        (631, "FWD", 60, 33), (632, "FWD", 60, 34), (633, "FWD", 60, 35),
    ]
    pool = owned + stars
    prices = make_prices(pool)
    xp_by_gw: dict[int, dict[int, float]] = {10: {}, 11: {}, 12: {}}
    for i, (c, *_rest) in enumerate(owned):
        for w in (10, 11, 12):
            xp_by_gw[w][c] = 2.0 + i * 0.01
    for c, *_rest in stars:
        xp_by_gw[10][c] = 0.0
        xp_by_gw[11][c] = 15.0
        xp_by_gw[12][c] = 0.0
    state = make_state(owned, bank=20, fts=1, chips=["fh1"])
    plan = solve_plan(make_xp(xp_by_gw), prices, state, horizon=3, params=tiny_params())
    assert_plan_legal(plan, prices)
    g0, g1, g2 = plan.gws
    assert g1.chip == "fh1"
    star_codes = {c for c, *_rest in stars}
    owned_codes = {c for c, *_rest in owned}
    assert set(g1.squad) == star_codes  # FH budget: 15*62 + 20 = 950 >= 15*60
    assert set(g0.squad) == owned_codes
    assert set(g2.squad) == owned_codes  # squad reverts: continuity through the FH week
    for g in plan.gws:
        assert g.transfers_in == [] and g.transfers_out == []  # no permanent transfers
        assert g.bank == 20
    assert g1.free_transfers == 2  # banked in GW10
    assert g2.free_transfers == 3  # FH passes the FT bank through with +1
    assert_ft_trajectory(plan, state)


def test_bench_boost_adds_bench_xp_for_one_gw() -> None:
    prices = make_prices(STD_SQUAD)
    bench4 = {102, 205, 305, 403}
    xp_by_gw: dict[int, dict[int, float]] = {10: {}, 11: {}}
    for i, (c, *_rest) in enumerate(STD_SQUAD):
        xp_by_gw[10][c] = 0.5 if c in bench4 else 4.0 + i * 0.01
        xp_by_gw[11][c] = 5.0 + i * 0.01
    state = make_state(STD_SQUAD, chips=["bb1"])
    plan = solve_plan(make_xp(xp_by_gw), prices, state, horizon=2, params=tiny_params())
    assert_plan_legal(plan, prices)
    g0, g1 = plan.gws
    assert g0.chip is None and g1.chip == "bb1"
    assert len(g1.lineup) == 15 and g1.bench_order == []
    assert len(g0.lineup) == 11 and len(g0.bench_order) == 4
    all_squad_xp = sum(xp_by_gw[11].values())
    cap_xp = max(xp_by_gw[11].values())
    assert g1.expected_points == pytest.approx(all_squad_xp + cap_xp, abs=1e-9)


def test_triple_captain_triples_exactly_the_captain() -> None:
    prices = make_prices(STD_SQUAD)
    xp_by_gw: dict[int, dict[int, float]] = {10: {}, 11: {}}
    for i, (c, *_rest) in enumerate(STD_SQUAD):
        xp_by_gw[10][c] = 2.0 + i * 0.01
        xp_by_gw[11][c] = 2.0 + i * 0.01
    xp_by_gw[10][301] = 8.0
    xp_by_gw[11][301] = 12.0
    state = make_state(STD_SQUAD, chips=["tc1"])
    plan = solve_plan(make_xp(xp_by_gw), prices, state, horizon=2, params=tiny_params())
    assert_plan_legal(plan, prices)
    g0, g1 = plan.gws
    assert g0.chip is None and g1.chip == "tc1"  # 12*0.84 > 8: play it on the bigger week
    assert g1.captain == 301
    xi_xp = sum(xp_by_gw[11][p] for p in g1.lineup)
    assert g1.expected_points == pytest.approx(xi_xp + 2 * 12.0, abs=1e-9)
    assert g0.expected_points == pytest.approx(
        sum(xp_by_gw[10][p] for p in g0.lineup) + 8.0, abs=1e-9
    )


def test_forced_chip_hook() -> None:
    prices = make_prices(STD_SQUAD)
    xp_of = {c: 2.0 + i * 0.01 for i, (c, *_rest) in enumerate(STD_SQUAD)}
    state = make_state(STD_SQUAD, chips=["bb1"])
    plan = solve_plan(
        make_xp({10: xp_of, 11: xp_of}),
        prices,
        state,
        horizon=2,
        params=tiny_params(forced_chips={10: "bb"}),
    )
    assert plan.gws[0].chip == "bb1"
    assert plan.gws[1].chip is None


def test_banned_and_locked_players() -> None:
    pool = STD_SQUAD + [(501, "MID", 50, 16), (502, "MID", 50, 17)]
    prices = make_prices(pool)
    xp_of = {c: 2.0 + i * 0.01 for i, (c, *_rest) in enumerate(pool)}
    xp_of[501] = 30.0  # would be the obvious buy — but banned
    xp_of[305] = 0.0
    state = make_state(STD_SQUAD, fts=1)
    plan = solve_plan(
        make_xp({10: xp_of}),
        prices,
        state,
        horizon=1,
        params=tiny_params(banned_players={501}, locked_players={305}),
    )
    g = plan.gws[0]
    assert 501 not in g.squad
    assert 305 in g.squad  # locked despite 0 xp


# --------------------------------------------------------------------------------------
# SquadState.from_entry (offline stub + live)
# --------------------------------------------------------------------------------------


class _StubClient:
    """Duck-typed FplApiClient serving fixed payloads."""

    def __init__(self) -> None:
        self.elements = [
            {
                "id": i,
                "code": 1000 + i,
                "now_cost": 50 + (i % 10),
                "cost_change_start": i % 2,  # purchase-since-GW1 = now - this
            }
            for i in range(1, 26)
        ]
        events = []
        for e in range(1, 39):
            events.append(
                {
                    "id": e,
                    "finished": e <= 10,
                    "is_current": e == 10,
                    "is_next": e == 11,
                    "deadline_time": f"2025-10-{min(e + 10, 28):02d}T17:30:00Z",
                }
            )
        self._bootstrap = {
            "game_config": {
                "settings": {
                    "static_content_url": "https://fantasy.premierleague.com/static/2025_26/"
                }
            },
            "events": events,
            "elements": self.elements,
            "total_players": 12345,
        }

    def bootstrap(self) -> dict[str, Any]:
        return self._bootstrap

    def entry_history(self, entry_id: int) -> dict[str, Any]:
        transfers_per_gw = {2: 0, 3: 1, 4: 0, 5: 3, 6: 0, 7: 1, 8: 0, 9: 2, 10: 0}
        return {
            "current": [
                {"event": g, "bank": 7 if g == 10 else 3, "event_transfers": n}
                for g, n in transfers_per_gw.items()
            ],
            "chips": [
                {"name": "wildcard", "event": 5},
                {"name": "bboost", "event": 8},
            ],
        }

    def entry_picks(self, entry_id: int, gw: int) -> dict[str, Any]:
        assert gw == 10
        ids = [1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20]  # 5 was sold for 20
        return {"picks": [{"element": i} for i in ids]}

    def entry_transfers(self, entry_id: int) -> list[dict[str, Any]]:
        return [  # newest first
            {
                "element_in": 21, "element_in_cost": 55,
                "element_out": 4, "element_out_cost": 48, "event": 11,  # pending
            },
            {
                "element_in": 20, "element_in_cost": 60,
                "element_out": 5, "element_out_cost": 50, "event": 7,
            },
        ]


def test_from_entry_offline_reconstruction() -> None:
    state = SquadState.from_entry(99, client=_StubClient())  # type: ignore[arg-type]
    assert state.season == SEASON
    assert state.current_gw == 11
    codes = {p.player_code for p in state.squad}
    expected_ids = {1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20, 21}
    assert codes == {1000 + i for i in expected_ids}  # pending 4 -> 21 applied
    by_code = {p.player_code: p for p in state.squad}
    assert by_code[1020].purchase_price == 60  # from the GW7 transfer log
    assert by_code[1021].purchase_price == 55  # from the pending transfer
    stub_el = {e["id"]: e for e in _StubClient().elements}
    assert by_code[1001].purchase_price == stub_el[1]["now_cost"] - stub_el[1][
        "cost_change_start"
    ]  # held since GW1: start price
    assert state.bank == 7 + 48 - 55  # picks-GW bank adjusted for the pending transfer
    # FT replay: 1->2->2->3->WC(4)->5->5->5->4->5, minus the 1 pending GW11 transfer
    # already confirmed (it consumed an FT the moment it was made)
    assert state.free_transfers == 4
    assert set(state.chips_available) == {"wc2", "fh1", "fh2", "bb2", "tc1", "tc2"}
    assert state.active_chip is None


@pytest.mark.live
def test_from_entry_live_entry_1() -> None:
    state = SquadState.from_entry(1)
    assert len(state.squad) == 15
    assert 1 <= state.free_transfers <= 5
    assert state.bank >= 0
    assert set(state.chips_available) <= set(
        f"{w.name.lower()}{w.set}" for w in rules.chip_windows(state.season)
    )


# --------------------------------------------------------------------------------------
# Realistic instance: real 2025-26 prices/positions, synthetic seeded xP, 8 GWs
# --------------------------------------------------------------------------------------

_PLAYER_GW = config.PROCESSED_DIR / "player_gw.parquet"


@pytest.mark.skipif(not _PLAYER_GW.exists(), reason="processed data not built")
def test_realistic_full_pool_eight_gw_under_time_budget() -> None:
    pg = pd.read_parquet(
        _PLAYER_GW, columns=["season", "gw", "player_code", "position", "team_code", "value"]
    )
    last = (
        pg[(pg.season == 2025) & (pg.gw == 38) & pg.position.isin(rules.POSITIONS)]
        .drop_duplicates("player_code")
        .rename(columns={"value": "price"})
    )
    prices = last[["player_code", "price", "position", "team_code"]].reset_index(drop=True)
    n = len(prices)
    assert n > 600  # a real full pool

    rng = np.random.default_rng(42)
    weeks = list(range(10, 18))
    strength = (prices.price - prices.price.min()) / (prices.price.max() - prices.price.min())
    # realistic xp structure: stable price-driven quality + small weekly fixture wiggle
    base = np.clip(rng.normal(1.2 + 4.8 * strength, 0.5), 0.0, None)
    xp_rows = []
    for gw in weeks:
        vals = np.clip(base + rng.normal(0.0, 0.3, n), 0.0, None)
        q0 = np.clip(rng.beta(2, 9, n), 0.0, 1.0)
        for code, b, q in zip(prices.player_code, vals, q0, strict=True):
            xp_rows.append(
                {"season": SEASON, "gw": gw, "player_code": int(code), "xp": float(b),
                 "q0": float(q)}
            )
    xp = pd.DataFrame(xp_rows)

    # a plausible mid-priced legal squad from the real pool
    squad: list[OwnedPlayer] = []
    club_count: dict[int, int] = {}
    for pos, quota in (("GKP", 2), ("DEF", 5), ("MID", 5), ("FWD", 3)):
        group = prices[prices.position == pos].sort_values(["price", "player_code"])
        picked = 0
        for row in group.iloc[15:].itertuples():  # skip the ultra-cheap fodder
            if club_count.get(row.team_code, 0) >= 3:
                continue
            club_count[row.team_code] = club_count.get(row.team_code, 0) + 1
            squad.append(
                OwnedPlayer(
                    player_code=int(row.player_code),
                    purchase_price=int(row.price),
                    current_price=int(row.price),
                )
            )
            picked += 1
            if picked == quota:
                break
    cost = sum(p.current_price for p in squad)
    assert len(squad) == 15 and cost <= 1000
    state = SquadState(
        season=SEASON,
        current_gw=10,
        squad=squad,
        bank=1000 - cost,
        free_transfers=2,
        chips_available=["wc1", "fh1", "bb1", "tc1", "wc2", "fh2", "bb2", "tc2"],
    )

    # (a) transfers-only planning solves to proven optimality inside the 60s budget
    t0 = time.perf_counter()
    plan_nochip = solve_plan(
        xp,
        prices,
        state,
        horizon=8,
        params=SolveParams(time_limit_s=60.0, mip_rel_gap=1e-3, threads=0, no_chips=True),
    )
    wall_nochip = time.perf_counter() - t0
    print(
        f"\n[realistic no-chips] pool={n} vars={plan_nochip.n_variables} "
        f"cons={plan_nochip.n_constraints} status={plan_nochip.status} "
        f"obj={plan_nochip.objective:.2f} gap={plan_nochip.gap:.2e} "
        f"solver_s={plan_nochip.solve_seconds:.2f} wall_s={wall_nochip:.2f}"
    )
    assert plan_nochip.status == "optimal"
    assert plan_nochip.solve_seconds < 60.0
    assert_plan_legal(plan_nochip, prices)

    # (b) the full chip-enabled solve must deliver a legal incumbent within the budget
    # (proving optimality with all 4 chips open needs the §3.5 long time limit on HiGHS)
    t0 = time.perf_counter()
    plan = solve_plan(
        xp,
        prices,
        state,
        horizon=8,
        params=SolveParams(time_limit_s=60.0, mip_rel_gap=1e-3, threads=0),
    )
    wall = time.perf_counter() - t0
    print(
        f"[realistic all-chips] pool={n} vars={plan.n_variables} "
        f"cons={plan.n_constraints} status={plan.status} obj={plan.objective:.2f} "
        f"gap={plan.gap:.2e} solver_s={plan.solve_seconds:.2f} wall_s={wall:.2f}"
    )
    assert plan.status in ("optimal", "time_limit")
    assert plan.solve_seconds < 62.0  # returned within the configured budget
    assert plan.objective >= plan_nochip.objective - 1e-6  # chips can only help
    assert_plan_legal(plan, prices)
    assert len(plan.gws) == 8
    assert_ft_trajectory(plan, state)
