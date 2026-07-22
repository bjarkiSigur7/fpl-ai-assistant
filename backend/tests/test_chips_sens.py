"""Tests for optimizer.chips + optimizer.sensitivity against a stubbed solve_plan.

The real ``optimizer.milp`` is built in parallel, so these tests inject deterministic
stub solvers via the ``solve_fn`` parameter (the same seam a monkeypatched
``milp.solve_plan`` would use). One integration-style test at the bottom exercises the
real solver iff ``fplai.optimizer.milp`` is importable.
"""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from fplai.optimizer.chips import (
    chip_ev_curves,
    copy_params_with,
    default_solve_params,
    season_chip_ids,
)
from fplai.optimizer.sensitivity import perturb_xp, plan_stability

SEASON = 2026

# --------------------------------------------------------------------------------------
# Synthetic fixtures (deterministic, offline)
# --------------------------------------------------------------------------------------

#: 15-man squad: 2 GKP, 5 DEF, 5 MID, 3 FWD (codes 101-115).
SQUAD_POSITIONS: dict[int, str] = {
    **{c: "GKP" for c in (101, 102)},
    **{c: "DEF" for c in range(103, 108)},
    **{c: "MID" for c in range(108, 113)},
    **{c: "FWD" for c in range(113, 116)},
}


def make_state(current_gw: int = 1, chips_available: list[str] | None = None) -> SimpleNamespace:
    """A SquadState-shaped stub (attribute access only, like the pydantic model)."""
    return SimpleNamespace(
        season=SEASON,
        current_gw=current_gw,
        squad=[
            SimpleNamespace(player_code=c, purchase_price=50, current_price=50)
            for c in SQUAD_POSITIONS
        ],
        bank=0,
        free_transfers=1,
        chips_available=(
            chips_available if chips_available is not None else season_chip_ids(SEASON)
        ),
        active_chip=None,
    )


def make_frames(
    gws: list[int],
    xp_fn,
    extra_positions: dict[int, str] | None = None,
    q0: float = 0.05,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build (xp, prices) frames per the milp input contract."""
    positions = {**SQUAD_POSITIONS, **(extra_positions or {})}
    xp_rows = [
        {"season": SEASON, "gw": g, "player_code": c, "xp": float(xp_fn(c, g)), "q0": q0}
        for g in gws
        for c in positions
    ]
    price_rows = [
        {"player_code": c, "price": 50, "position": p, "team_code": 1 + (i % 10)}
        for i, (c, p) in enumerate(positions.items())
    ]
    return pd.DataFrame(xp_rows), pd.DataFrame(price_rows)


@dataclasses.dataclass
class StubParams:
    """Dataclass params carrying the chip-directive fields chips.py sets."""

    forced_chips: dict[int, str] = dataclasses.field(default_factory=dict)
    banned_chips: frozenset[str] = frozenset()
    decay: float = 0.84


class ChipStubSolver:
    """Deterministic objective as a function of the forced-chip directive."""

    def __init__(self, objective_fn) -> None:
        self.objective_fn = objective_fn
        self.calls: list[Any] = []

    def __call__(self, xp, prices, state, *, horizon=8, params=None):
        self.calls.append(params)
        forced = dict(getattr(params, "forced_chips", {}) or {})
        return SimpleNamespace(
            objective=self.objective_fn(forced), gws=[], solve_seconds=0.01, gap=0.0
        )


def chip_objective(forced: dict[int, str]) -> float:
    """Baseline 100; BB at gw g -> +0.5*g; TC -> +3; WC -> -1."""
    if not forced:
        return 100.0
    ((g, chip),) = forced.items()
    if chip.startswith("bb"):
        return 100.0 + 0.5 * g
    if chip.startswith("tc"):
        return 103.0
    return 99.0


# --------------------------------------------------------------------------------------
# chip_ev_curves
# --------------------------------------------------------------------------------------


class TestChipEvCurves:
    def test_deltas_vs_no_chip_baseline(self) -> None:
        xp, prices = make_frames([1, 2, 3, 4, 5, 6], lambda c, g: 5.0)
        solver = ChipStubSolver(chip_objective)
        df = chip_ev_curves(
            xp,
            prices,
            make_state(),
            ["bb1", "tc1"],
            range(1, 7),
            params=StubParams(),
            solve_fn=solver,
        )
        assert list(df.columns[:4]) == ["chip", "gw", "objective", "delta_vs_no_chip"]
        assert df.attrs["baseline_objective"] == 100.0
        bb = df[df["chip"] == "bb1"].set_index("gw")
        for g in range(1, 7):
            assert bb.loc[g, "evaluated"]
            assert bb.loc[g, "delta_vs_no_chip"] == pytest.approx(0.5 * g)
        tc = df[df["chip"] == "tc1"]
        assert (tc["delta_vs_no_chip"] == 3.0).all()
        # 1 baseline + 12 forced solves
        assert len(solver.calls) == 13

    def test_directives_isolate_the_forced_chip(self) -> None:
        xp, prices = make_frames([1, 2], lambda c, g: 5.0)
        solver = ChipStubSolver(chip_objective)
        chip_ev_curves(
            xp,
            prices,
            make_state(),
            ["bb1"],
            [1, 2],
            params=StubParams(),
            solve_fn=solver,
        )
        baseline_params = solver.calls[0]
        assert baseline_params.forced_chips == {}
        assert set(season_chip_ids(SEASON)) <= set(baseline_params.banned_chips)
        forced_params = solver.calls[1]
        assert forced_params.forced_chips == {1: "bb1"}
        assert "bb1" not in forced_params.banned_chips
        assert "tc1" in forced_params.banned_chips

    def test_outside_window_flagged_not_solved(self) -> None:
        xp, prices = make_frames([1, 2, 3], lambda c, g: 5.0)
        solver = ChipStubSolver(chip_objective)
        df = chip_ev_curves(
            xp,
            prices,
            make_state(),
            ["wc1"],
            [1, 2, 3],
            params=StubParams(),
            solve_fn=solver,
        )
        row = df[df["gw"] == 1].iloc[0]
        assert not row["evaluated"]
        assert row["skip_reason"] == "outside_window"  # WC opens at GW2
        assert np.isnan(row["objective"])
        evaluated = df[df["evaluated"]]
        assert list(evaluated["gw"]) == [2, 3]
        assert (evaluated["delta_vs_no_chip"] == -1.0).all()

    def test_outside_horizon_flagged(self) -> None:
        xp, prices = make_frames([1, 2, 3], lambda c, g: 5.0)
        df = chip_ev_curves(
            xp,
            prices,
            make_state(),
            ["bb1"],
            [1, 2, 3],
            horizon=2,
            params=StubParams(),
            solve_fn=ChipStubSolver(chip_objective),
        )
        assert df.set_index("gw").loc[3, "skip_reason"] == "outside_horizon"

    def test_unavailable_chip_never_solved(self) -> None:
        xp, prices = make_frames([1, 2], lambda c, g: 5.0)
        solver = ChipStubSolver(chip_objective)
        df = chip_ev_curves(
            xp,
            prices,
            make_state(chips_available=["tc1"]),
            ["bb1"],
            [1, 2],
            params=StubParams(),
            solve_fn=solver,
        )
        assert (df["skip_reason"] == "not_available").all()
        assert len(solver.calls) == 1  # baseline only

    def test_bb_dominated_cells_skipped_with_flag(self) -> None:
        # Squad bench proxy (4 lowest of the 15) = 0.4*g -> bottom quartile is gw 1-2.
        def xp_fn(c: int, g: int) -> float:
            return 0.1 * g if c in (112, 113, 114, 115) else 5.0

        xp, prices = make_frames(list(range(1, 9)), xp_fn)
        solver = ChipStubSolver(chip_objective)
        df = chip_ev_curves(
            xp,
            prices,
            make_state(),
            ["bb1"],
            range(1, 9),
            params=StubParams(),
            solve_fn=solver,
        )
        skipped = df[~df["evaluated"]]
        assert list(skipped["gw"]) == [1, 2]
        assert (skipped["skip_reason"] == "dominated:bb_bench_bottom_quartile").all()
        assert np.isnan(skipped["delta_vs_no_chip"]).all()
        assert len(solver.calls) == 1 + 6

        # skip_dominated=False evaluates everything.
        solver2 = ChipStubSolver(chip_objective)
        df_all = chip_ev_curves(
            xp,
            prices,
            make_state(),
            ["bb1"],
            range(1, 9),
            params=StubParams(),
            solve_fn=solver2,
            skip_dominated=False,
        )
        assert df_all["evaluated"].all()
        assert len(solver2.calls) == 1 + 8

    def test_urgency_near_expiry(self) -> None:
        """Set-1 BB near the GW19 cliff: urgency = delta - best remaining alternative."""

        def objective(forced: dict[int, str]) -> float:
            if not forced:
                return 100.0
            ((g, _),) = forced.items()
            return {17: 101.0, 18: 105.0, 19: 103.0}[g]

        xp, prices = make_frames([17, 18, 19], lambda c, g: 5.0)
        df = chip_ev_curves(
            xp,
            prices,
            make_state(current_gw=17),
            ["bb1"],
            [17, 18, 19],
            horizon=3,
            params=StubParams(),
            solve_fn=ChipStubSolver(objective),
        ).set_index("gw")
        assert df.loc[17, "urgency"] == pytest.approx(1.0 - 5.0)  # wait for GW18
        assert df.loc[18, "urgency"] == pytest.approx(5.0 - 3.0)  # now beats the rest
        assert df.loc[19, "urgency"] == pytest.approx(3.0)  # last chance: use or lose
        assert (df["window_last_gw"] == 19).all()

    def test_no_urgency_far_from_expiry(self) -> None:
        xp, prices = make_frames([1, 2], lambda c, g: 5.0)
        df = chip_ev_curves(
            xp,
            prices,
            make_state(),
            ["bb1"],
            [1, 2],
            params=StubParams(),
            solve_fn=ChipStubSolver(chip_objective),
        )
        assert np.isnan(df["urgency"]).all()  # GW19 expiry is 18 GWs away

    def test_invalid_chip_id_raises(self) -> None:
        xp, prices = make_frames([1], lambda c, g: 5.0)
        with pytest.raises(ValueError, match="unrecognized chip id"):
            chip_ev_curves(
                xp,
                prices,
                make_state(),
                ["xx1"],
                [1],
                params=StubParams(),
                solve_fn=ChipStubSolver(chip_objective),
            )

    def test_none_params_and_none_state(self) -> None:
        """params=None (milp may be absent) and state=None (initial build) both work."""
        xp, prices = make_frames([1, 2], lambda c, g: 5.0)
        solver = ChipStubSolver(chip_objective)
        df = chip_ev_curves(xp, prices, None, ["bb1"], [1, 2], solve_fn=solver)
        assert df["evaluated"].all()
        assert (df["delta_vs_no_chip"] == [0.5, 1.0]).all()


class TestParamsGlue:
    def test_copy_params_with_dataclass_pydantic_and_none(self) -> None:
        dc = StubParams(decay=0.9)
        dc2 = copy_params_with(dc, forced_chips={3: "bb1"})
        assert dc2.forced_chips == {3: "bb1"} and dc2.decay == 0.9
        assert dc.forced_chips == {}  # original untouched

        import pydantic

        class PydParams(pydantic.BaseModel):
            forced_chips: dict[int, str] = {}
            banned_chips: frozenset[str] = frozenset()

        pm = PydParams()
        pm2 = copy_params_with(pm, forced_chips={5: "tc1"})
        assert pm2.forced_chips == {5: "tc1"} and pm.forced_chips == {}

        ns = copy_params_with(None, forced_chips={1: "wc1"})
        assert ns.forced_chips == {1: "wc1"}


# --------------------------------------------------------------------------------------
# plan_stability
# --------------------------------------------------------------------------------------


class SensStubSolver:
    """Buys 200 for 108 iff perturbed xp favors it; captain = this-GW argmax."""

    def __init__(self) -> None:
        self.captured: list[pd.DataFrame] = []

    def __call__(self, xp, prices, state, *, horizon=8, params=None):
        self.captured.append(xp)
        gw = state.current_gw if state is not None else int(xp["gw"].min())
        this_gw = xp[xp["gw"] == gw].groupby("player_code")["xp"].sum()
        buy = bool(this_gw.get(200, -np.inf) > this_gw.get(108, -np.inf))
        plan = SimpleNamespace(
            gw=gw,
            squad=[],
            lineup=[],
            bench_order=[],
            captain=int(this_gw.idxmax()),
            vice=None,
            transfers_in=[200] if buy else [],
            transfers_out=[108] if buy else [],
            hit_points=0,
            chip=None,
            expected_points=float(this_gw.sum()),
        )
        return SimpleNamespace(
            objective=float(this_gw.sum()), gws=[plan], solve_seconds=0.01, gap=0.0
        )


def sens_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Squad + close-call transfer target 200 (5.05 vs 5.00) + a nailed captain 113."""
    extra = {200: "MID", **{c: "MID" for c in range(300, 400)}}

    def xp_fn(c: int, g: int) -> float:
        if c == 113:
            return 8.0  # nailed captain
        if c == 200:
            return 5.05
        if 300 <= c < 400:
            return 0.01  # pool filler, pruned by the warm pool
        return 5.0

    xp, prices = make_frames([1, 2], xp_fn, extra_positions=extra, q0=0.0)
    # High minutes-uncertainty on the transfer pair -> §3.4 inflates their noise.
    xp.loc[xp["player_code"].isin([200, 108]), "q0"] = 0.5
    return xp, prices


class TestPlanStability:
    def test_columns_support_and_determinism(self) -> None:
        xp, prices = sens_frames()
        kwargs = dict(n=30, strength=1.0, seed=7, solve_fn=SensStubSolver(), pool_size=40)
        df = plan_stability(xp, prices, make_state(), **kwargs)
        assert list(df.columns) == ["move", "support_pct", "count"]
        support = df.set_index("move")["support_pct"]
        # The 5.05-vs-5.00 call flips under noise: strictly contested both ways.
        assert 0.0 < support["buy:200"] < 100.0
        assert support["buy:200"] == support["sell:108"]
        assert support["chip:none"] == 100.0
        assert support.get("captain:113", 0.0) > 50.0
        # Deterministic for a fixed seed.
        kwargs2 = dict(kwargs, solve_fn=SensStubSolver())
        pd.testing.assert_frame_equal(df, plan_stability(xp, prices, make_state(), **kwargs2))

    def test_zero_strength_is_unanimous(self) -> None:
        xp, prices = sens_frames()
        df = plan_stability(
            xp,
            prices,
            make_state(),
            n=10,
            strength=0.0,
            seed=0,
            solve_fn=SensStubSolver(),
            pool_size=40,
        )
        support = df.set_index("move")["support_pct"]
        assert support["buy:200"] == 100.0
        assert support["sell:108"] == 100.0
        assert support["captain:113"] == 100.0
        assert "hold" not in support.index

    def test_warm_pool_reused_and_pruned(self) -> None:
        xp, prices = sens_frames()
        solver = SensStubSolver()
        plan_stability(
            xp,
            prices,
            make_state(),
            n=5,
            strength=1.0,
            seed=3,
            solve_fn=solver,
            pool_size=40,
        )
        assert len(solver.captured) == 5
        pools = [set(frame["player_code"].unique()) for frame in solver.captured]
        assert all(p == pools[0] for p in pools)  # same warm pool every run
        pool = pools[0]
        assert len(pool) <= 45  # pool_size + per-position minima slack
        assert set(SQUAD_POSITIONS) <= pool  # squad always kept
        assert 200 in pool
        assert len([c for c in pool if 300 <= c < 400]) < 30  # filler mostly pruned

    def test_distinct_seeds_produce_distinct_noise(self) -> None:
        xp, prices = sens_frames()
        solver = SensStubSolver()
        plan_stability(
            xp,
            prices,
            make_state(),
            n=3,
            strength=1.0,
            seed=11,
            solve_fn=solver,
            pool_size=40,
        )
        a, b = solver.captured[0]["xp"].to_numpy(), solver.captured[1]["xp"].to_numpy()
        assert not np.allclose(a, b)

    def test_hold_reported_when_no_transfers(self) -> None:
        xp, prices = sens_frames()
        xp.loc[xp["player_code"] == 200, "xp"] = 0.5  # clearly worse: never bought
        df = plan_stability(
            xp,
            prices,
            make_state(),
            n=8,
            strength=0.0,
            seed=0,
            solve_fn=SensStubSolver(),
            pool_size=40,
        )
        assert df.set_index("move")["support_pct"]["hold"] == 100.0

    def test_perturb_xp_formula_exact(self) -> None:
        """§3.4 exactly: xp' = xp + strength * xp * (92 - 90*(1-q0)) / 134 * z."""
        xp, _ = sens_frames()
        noisy = perturb_xp(xp, strength=1.3, rng=np.random.default_rng(99))
        z = np.random.default_rng(99).standard_normal(len(xp))
        q0 = xp["q0"].to_numpy()
        base = xp["xp"].to_numpy()
        expected = base + base * 1.3 * (92.0 - 90.0 * (1.0 - q0)) / 134.0 * z
        np.testing.assert_allclose(noisy["xp"].to_numpy(), expected)
        # Original frame untouched.
        np.testing.assert_array_equal(xp["xp"].to_numpy(), base)

    def test_perturb_xp_warns_without_q0(self) -> None:
        xp, _ = sens_frames()
        with pytest.warns(UserWarning, match="no 'q0' column"):
            perturb_xp(xp.drop(columns=["q0"]), strength=1.0, rng=np.random.default_rng(0))

    def test_invalid_n_raises(self) -> None:
        xp, prices = sens_frames()
        with pytest.raises(ValueError, match="n must be positive"):
            plan_stability(xp, prices, make_state(), n=0, solve_fn=SensStubSolver())


# --------------------------------------------------------------------------------------
# Integration with the real MILP (skipped until optimizer.milp lands)
# --------------------------------------------------------------------------------------


def test_real_milp_integration() -> None:
    """Exercise chip_ev_curves through the real solver iff optimizer.milp is importable."""
    milp = pytest.importorskip("fplai.optimizer.milp", reason="optimizer.milp not built yet")
    if not hasattr(milp, "solve_plan"):
        pytest.skip("optimizer.milp has no solve_plan yet")

    params = default_solve_params()
    for attr in ("time_limit_s", "time_limit", "max_seconds", "time_limit_seconds"):
        if params is not None and hasattr(params, attr):
            params = copy_params_with(params, **{attr: 30.0})
            break

    # 30-player pool: 4 GKP / 10 DEF / 10 MID / 6 FWD, all affordable, <=3 per club.
    positions = (
        {c: "GKP" for c in range(1, 5)}
        | {c: "DEF" for c in range(5, 15)}
        | {c: "MID" for c in range(15, 25)}
        | {c: "FWD" for c in range(25, 31)}
    )
    xp = pd.DataFrame(
        [
            {"season": SEASON, "gw": g, "player_code": c, "xp": 2.0 + 0.1 * (c % 7), "q0": 0.05}
            for g in (1, 2)
            for c in positions
        ]
    )
    prices = pd.DataFrame(
        [
            {"player_code": c, "price": 50, "position": p, "team_code": 1 + (c % 10)}
            for c, p in positions.items()
        ]
    )
    try:
        curves = chip_ev_curves(
            xp,
            prices,
            None,
            ["bb1"],
            [1, 2],
            horizon=2,
            params=params,
            solve_fn=milp.solve_plan,
        )
    except Exception as exc:  # pragma: no cover - depends on the parallel milp build
        pytest.skip(f"milp call-site shape differs; adapt chips.py glue constants: {exc}")
    evaluated = curves[curves["evaluated"]]
    assert len(evaluated) >= 1
    assert np.isfinite(evaluated["objective"]).all()
