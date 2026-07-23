"""Tests for optimizer.season_sim — the full-window Monte Carlo chip simulator.

The MILP and the PointsSampler are injected through the ``solve_fn`` / ``sampler`` seams:
most tests run against a deterministic stub solver and hand-planted sampled
distributions, so every expectation below is computable by hand. One smoke test runs the
real MILP on a tiny pool, and one integration test uses the real
``models.sampler.PointsSampler`` iff it is importable (it is built in parallel).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from fplai import config
from fplai.optimizer.chips import season_chip_ids
from fplai.optimizer.milp import GwPlan, PlanResult
from fplai.optimizer.season_sim import (
    ChipSimReport,
    simulate_chip_plans,
)
from fplai.optimizer.state import OwnedPlayer, SquadState

SEASON = 2026

# --------------------------------------------------------------------------------------
# Synthetic world: 15 players (codes 1-15), one club each, fixed backbone plan
# --------------------------------------------------------------------------------------

POSITIONS: dict[int, str] = {
    **{c: "GKP" for c in (1, 2)},
    **{c: "DEF" for c in range(3, 8)},
    **{c: "MID" for c in range(8, 13)},
    **{c: "FWD" for c in range(13, 16)},
}
LINEUP = [1, 3, 4, 5, 6, 8, 9, 10, 11, 13, 14]  # 1-4-4-2
BENCH = [2, 7, 12, 15]  # slot 0 = GK
CAPTAIN, VICE = 13, 8
BACKBONE_OBJECTIVE = 200.0


def xp_val(code: int, gw: int) -> float:  # noqa: ARG001 - gw-flat by design
    return code * 0.3


def make_frames(gws: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    xp = pd.DataFrame(
        [
            {"season": SEASON, "gw": g, "player_code": c, "xp": xp_val(c, g), "q0": 0.1}
            for g in gws
            for c in POSITIONS
        ]
    )
    prices = pd.DataFrame(
        [
            {"player_code": c, "price": 50, "position": p, "team_code": c}
            for c, p in POSITIONS.items()
        ]
    )
    return xp, prices


def make_state(current_gw: int = 1) -> SquadState:
    return SquadState(
        season=SEASON,
        current_gw=current_gw,
        squad=[
            OwnedPlayer(player_code=c, purchase_price=50, current_price=50) for c in POSITIONS
        ],
        bank=0,
        free_transfers=1,
        chips_available=season_chip_ids(SEASON),
        active_chip=None,
    )


def make_plan(
    weeks: list[int], objective: float, variant_gw: int | None = None, chip: str | None = None
) -> PlanResult:
    """A canned PlanResult; ``variant_gw`` swaps starter 11 for bench MID 12 that week."""
    gws = []
    for w in weeks:
        if variant_gw == w:
            lineup = [1, 3, 4, 5, 6, 8, 9, 10, 12, 13, 14]
            bench = [2, 7, 11, 15]
            chip_id = chip
        else:
            lineup, bench, chip_id = LINEUP, BENCH, None
        gws.append(
            GwPlan(
                gw=w,
                squad=sorted(POSITIONS),
                lineup=lineup,
                bench_order=bench,
                captain=CAPTAIN,
                vice=VICE,
                transfers_in=[],
                transfers_out=[],
                hit_points=0,
                chip=chip_id,
                expected_points=30.0,
                bank=0,
                free_transfers=1,
            )
        )
    return PlanResult(
        objective=objective, gws=gws, solve_seconds=1.0, gap=0.01, status="time_limit"
    )


class StubSolver:
    """solve_plan-shaped stub: canned backbone + per-(chip, gw) planted segment deltas."""

    def __init__(
        self,
        *,
        window_start: int,
        full_horizon: int,
        seg_deltas: dict[tuple[str, int], float] | None = None,
    ) -> None:
        self.window_start = window_start
        self.full_horizon = full_horizon
        self.seg_deltas = seg_deltas or {}
        self.calls: list[SimpleNamespace] = []

    def __call__(
        self,
        xp: pd.DataFrame,
        prices: pd.DataFrame,
        state: Any,
        *,
        horizon: int,
        params: Any,
    ) -> PlanResult:
        forced = dict(getattr(params, "forced_chips", {}) or {})
        self.calls.append(
            SimpleNamespace(state=state, horizon=horizon, params=params, forced=forced)
        )
        start = int(state.current_gw) if state is not None else int(xp["gw"].min())
        weeks = list(range(start, start + horizon))
        if not forced:
            if start == self.window_start and horizon == self.full_horizon:
                return make_plan(weeks, BACKBONE_OBJECTIVE)  # the backbone call
            return make_plan(weeks, 50.0 + start)  # a no-chip segment baseline
        ((g, chip),) = forced.items()
        return make_plan(
            weeks, 50.0 + start + self.seg_deltas.get((chip, g), 0.0), variant_gw=g, chip=chip
        )


class PlantedSampler:
    """Planted per-(player_code, gw) point distributions; deterministic under seed.

    Table values: a scalar (constant points), ``("choice", values, probs)``, or a
    callable ``(rng, n) -> array``. Rows not in the table draw ``default``.
    """

    def __init__(
        self, table: dict[tuple[int, int], Any] | None = None, default: Any = 1.0
    ) -> None:
        self.table = table or {}
        self.default = default

    def _draw(self, spec: Any, rng: np.random.Generator, n: int) -> np.ndarray:
        if callable(spec):
            return np.asarray(spec(rng, n), dtype=float)
        if isinstance(spec, tuple) and spec and spec[0] == "choice":
            _, values, probs = spec
            return rng.choice(np.asarray(values, dtype=float), size=n, p=list(probs))
        return np.full(n, float(spec))

    def sample(self, predictions: pd.DataFrame, n: int, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        cols = [
            self._draw(self.table.get((int(c), int(g)), self.default), rng, n)
            for c, g in zip(predictions["player_code"], predictions["gw"], strict=True)
        ]
        return np.column_stack(cols)


class MinutesSampler(PlantedSampler):
    """PlantedSampler that also exposes the duck-typed ``sample_minutes`` extension."""

    def __init__(
        self,
        table: dict[tuple[int, int], Any] | None = None,
        default: Any = 1.0,
        minutes_default: float = 90.0,
        minutes_table: dict[tuple[int, int], float] | None = None,
    ) -> None:
        super().__init__(table, default)
        self.minutes_default = minutes_default
        self.minutes_table = minutes_table or {}

    def sample_minutes(self, predictions: pd.DataFrame, n: int, seed: int) -> np.ndarray:
        del seed
        cols = [
            np.full(n, self.minutes_table.get((int(c), int(g)), self.minutes_default))
            for c, g in zip(predictions["player_code"], predictions["gw"], strict=True)
        ]
        return np.column_stack(cols)


def stat_of(report: ChipSimReport, chip: str, gw: int):
    return next(s for s in report.stats if s.chip == chip and s.gw == gw)


def verdict_of(report: ChipSimReport, chip: str):
    return next(v for v in report.verdicts if v.chip == chip)


# --------------------------------------------------------------------------------------
# BB: planted bench distribution -> the best week is knowably GW3
# --------------------------------------------------------------------------------------


class TestBenchBoost:
    def run(self, n_rollouts: int = 8) -> ChipSimReport:
        gws = [1, 2, 3, 4]
        xp, prices = make_frames(gws)
        table = {(c, g): (3.0 if g == 3 else 0.5) for c in BENCH for g in gws}
        return simulate_chip_plans(
            xp,
            prices,
            None,
            window=range(1, 5),
            n_rollouts=n_rollouts,
            seed=0,
            sampler=PlantedSampler(table),
            solve_fn=StubSolver(window_start=1, full_horizon=4),
            chips=["bb1"],
        )

    def test_best_week_is_gw3(self):
        report = self.run()
        verdict = verdict_of(report, "bb1")
        assert verdict.recommended_gw == 3
        assert verdict.e_gain == pytest.approx(12.0)
        assert verdict.confidence == pytest.approx(1.0)
        assert verdict.sd == pytest.approx(0.0)

    def test_per_gw_stats(self):
        report = self.run()
        for g in (1, 2, 4):
            s = stat_of(report, "bb1", g)
            assert s.evaluated
            assert s.e_gain == pytest.approx(2.0)
            assert s.p_best_week == pytest.approx(0.0)
        # deterministic gains: sd is exactly 0, p_best_week sums to 1
        assert stat_of(report, "bb1", 3).sd == pytest.approx(0.0)
        total = sum(s.p_best_week for s in report.stats if s.evaluated)
        assert total == pytest.approx(1.0)

    def test_p_beats_hold(self):
        report = self.run()
        # GW1: 2 points now vs 12 later -> playing now never beats holding
        assert stat_of(report, "bb1", 1).p_beats_hold == pytest.approx(0.0)
        # GW3: 12 now vs 2 later -> always beats holding
        assert stat_of(report, "bb1", 3).p_beats_hold == pytest.approx(1.0)
        # GW4 (last candidate): holding forfeits the chip -> any gain >= 0 wins
        assert stat_of(report, "bb1", 4).p_beats_hold == pytest.approx(1.0)

    def test_deterministic_xp_gain_is_bench_xp_sum(self):
        report = self.run()
        expected = sum(xp_val(c, 3) for c in BENCH)
        assert stat_of(report, "bb1", 3).xp_gain == pytest.approx(expected)


# --------------------------------------------------------------------------------------
# TC: a fat-tailed captain week wins on E but with honest sub-1 confidence
# --------------------------------------------------------------------------------------


class TestTripleCaptain:
    def run(self) -> ChipSimReport:
        gws = [1, 2, 3, 4]
        xp, prices = make_frames(gws)
        table = {
            (CAPTAIN, 1): 1.0,
            (CAPTAIN, 2): 7.0,
            (CAPTAIN, 3): 1.0,
            (CAPTAIN, 4): ("choice", [4.0, 30.0], [0.5, 0.5]),  # the fat tail
        }
        return simulate_chip_plans(
            xp,
            prices,
            None,
            window=range(1, 5),
            n_rollouts=600,
            seed=3,
            sampler=PlantedSampler(table),
            solve_fn=StubSolver(window_start=1, full_horizon=4),
            chips=["tc1"],
        )

    def test_fat_tail_week_recommended_with_split_confidence(self):
        report = self.run()
        verdict = verdict_of(report, "tc1")
        assert verdict.recommended_gw == 4  # E = 17 beats the safe 7 at GW2
        assert verdict.e_gain == pytest.approx(17.0, abs=1.5)
        # ...but the tail only lands half the time, and p_best_week says so
        assert 0.4 < stat_of(report, "tc1", 4).p_best_week < 0.6
        assert 0.4 < stat_of(report, "tc1", 2).p_best_week < 0.6
        assert stat_of(report, "tc1", 1).p_best_week == pytest.approx(0.0)
        assert stat_of(report, "tc1", 4).sd > 5.0

    def test_p_beats_hold_quantifies_the_gamble(self):
        report = self.run()
        # GW2: 7 now beats holding iff the GW4 draw comes up 4 (p = 0.5)
        assert 0.4 < stat_of(report, "tc1", 2).p_beats_hold < 0.6
        assert stat_of(report, "tc1", 4).p_beats_hold == pytest.approx(1.0)

    def test_vice_fallback_when_captain_absent(self):
        gws = [1, 2, 3]
        xp, prices = make_frames(gws)
        # captain never appears (0 points = absent under the inference proxy); vice scores 4
        table = {(CAPTAIN, g): 0.0 for g in gws} | {(VICE, g): 4.0 for g in gws}
        report = simulate_chip_plans(
            xp,
            prices,
            None,
            window=range(1, 4),
            n_rollouts=8,
            seed=0,
            sampler=PlantedSampler(table),
            solve_fn=StubSolver(window_start=1, full_horizon=3),
            chips=["tc1"],
        )
        assert report.appearance_source == "points_nonzero"
        for g in gws:
            assert stat_of(report, "tc1", g).e_gain == pytest.approx(4.0)

    def test_sample_minutes_overrides_points_inference(self):
        gws = [1, 2, 3]
        xp, prices = make_frames(gws)
        # captain nets exactly 0 but PLAYED 90 minutes -> no vice fallback
        table = {(CAPTAIN, g): 0.0 for g in gws} | {(VICE, g): 4.0 for g in gws}
        report = simulate_chip_plans(
            xp,
            prices,
            None,
            window=range(1, 4),
            n_rollouts=8,
            seed=0,
            sampler=MinutesSampler(table),
            solve_fn=StubSolver(window_start=1, full_horizon=3),
            chips=["tc1"],
        )
        assert report.appearance_source == "sample_minutes"
        for g in gws:
            assert stat_of(report, "tc1", g).e_gain == pytest.approx(0.0)


# --------------------------------------------------------------------------------------
# WC/FH: segment re-solves, realized deltas with autosubs, backbone-fixed states
# --------------------------------------------------------------------------------------


class TestSegmentChips:
    SEG_DELTAS = {("wc1", 2): 1.5, ("wc1", 3): 2.5, ("wc1", 4): 0.5}

    def run(self) -> tuple[ChipSimReport, StubSolver]:
        gws = [1, 2, 3, 4]
        xp, prices = make_frames(gws)
        solver = StubSolver(window_start=1, full_horizon=4, seg_deltas=self.SEG_DELTAS)
        # starter 11 never appears; his replacement-by-transfer (12) scores 3; the
        # autosub who covers him on no-chip weeks (bench slot 1 = player 7) scores 1
        table = {(11, g): 0.0 for g in gws} | {(12, g): 3.0 for g in gws}
        report = simulate_chip_plans(
            xp,
            prices,
            make_state(),
            window=range(1, 5),
            n_rollouts=16,
            seed=0,
            sampler=PlantedSampler(table),
            solve_fn=solver,
            chips=["wc1", "fh1"],
        )
        return report, solver

    def test_wc1_not_playable_at_gw1(self):
        report, _ = self.run()
        for chip in ("wc1", "fh1"):
            s = stat_of(report, chip, 1)
            assert not s.evaluated
            assert s.skip_reason == "outside_window"

    def test_deterministic_gain_is_segment_objective_delta(self):
        report, _ = self.run()
        for (_chip, g), delta in self.SEG_DELTAS.items():
            assert stat_of(report, "wc1", g).xp_gain == pytest.approx(delta)

    def test_realized_gain_accounts_for_autosub_counterfactual(self):
        report, _ = self.run()
        # chip week: 12 starts and scores 3; no-chip week: 11 blanks and the autosub
        # (player 7, 1 pt) covers him -> realized delta is 3 - 1 = 2, every rollout
        for g in (2, 3, 4):
            s = stat_of(report, "wc1", g)
            assert s.e_gain == pytest.approx(2.0)
            assert s.sd == pytest.approx(0.0)
            assert s.p_best_week == pytest.approx(1 / 3)  # exact ties split evenly
            assert s.p_beats_hold == pytest.approx(1.0)

    def test_segment_states_are_fixed_from_the_backbone(self):
        _, solver = self.run()
        seg_calls = [c for c in solver.calls if c.state is not None and c.state.current_gw > 1]
        assert seg_calls, "expected segment re-solves"
        for call in seg_calls:
            assert [p.player_code for p in call.state.squad] == sorted(POSITIONS)
            assert call.state.bank == 0
            assert call.state.free_transfers == 1
            if call.forced:
                ((g, chip),) = call.forced.items()
                assert g == call.state.current_gw
                assert call.state.chips_available == [chip]
            else:
                assert call.state.chips_available == []
        # the shared no-chip segment baseline is solved once per GW, not once per chip
        baselines = [c for c in seg_calls if not c.forced]
        assert len(baselines) == len({c.state.current_gw for c in baselines})

    def test_backbone_call_bans_all_chips_over_full_window(self):
        _, solver = self.run()
        backbone_call = solver.calls[0]
        assert backbone_call.horizon == 4
        assert backbone_call.forced == {}
        assert set(season_chip_ids(SEASON)) <= set(backbone_call.params.banned_chips)
        assert backbone_call.params.time_limit_s == pytest.approx(120.0)
        assert backbone_call.params.prune_keep_frac == pytest.approx(0.06)


# --------------------------------------------------------------------------------------
# Windows and expiry
# --------------------------------------------------------------------------------------


class TestWindowsAndExpiry:
    def test_nothing_past_gw19_for_set1_chips(self):
        gws = list(range(16, 22))  # crosses the GW19 set-1 expiry
        xp, prices = make_frames(gws)
        report = simulate_chip_plans(
            xp,
            prices,
            make_state(current_gw=16),
            window=range(16, 22),
            n_rollouts=8,
            seed=0,
            sampler=PlantedSampler(),
            solve_fn=StubSolver(window_start=16, full_horizon=6),
            chips=["bb1", "tc1"],
        )
        for chip in ("bb1", "tc1"):
            for g in (20, 21):
                s = stat_of(report, chip, g)
                assert not s.evaluated
                assert s.skip_reason == "outside_window"
            for g in range(16, 20):
                assert stat_of(report, chip, g).evaluated
        evaluated_gws = {s.gw for s in report.stats if s.evaluated}
        assert max(evaluated_gws) <= 19
        frame = report.to_frame()
        assert not frame.loc[frame["gw"] > 19, "evaluated"].any()

    def test_used_chip_reported_not_available(self):
        gws = [1, 2, 3]
        xp, prices = make_frames(gws)
        state = make_state()
        state = state.model_copy(
            update={"chips_available": [c for c in state.chips_available if c != "bb1"]}
        )
        report = simulate_chip_plans(
            xp,
            prices,
            state,
            window=range(1, 4),
            n_rollouts=4,
            seed=0,
            sampler=PlantedSampler(),
            solve_fn=StubSolver(window_start=1, full_horizon=3),
            chips=["bb1", "tc1"],
        )
        assert all(
            s.skip_reason == "not_available" for s in report.stats if s.chip == "bb1"
        )
        assert verdict_of(report, "bb1").recommended_gw is None
        assert verdict_of(report, "tc1").recommended_gw is not None


# --------------------------------------------------------------------------------------
# Joint schedules
# --------------------------------------------------------------------------------------


class TestJointSchedules:
    def run(self) -> ChipSimReport:
        gws = [1, 2, 3, 4]
        xp, prices = make_frames(gws)
        # both chips peak at GW3: BB bench sum 12 there (else 2), TC captain 9 (else 1)
        table = {(c, g): (3.0 if g == 3 else 0.5) for c in BENCH for g in gws}
        table |= {(CAPTAIN, g): (9.0 if g == 3 else 1.0) for g in gws}
        return simulate_chip_plans(
            xp,
            prices,
            None,
            window=range(1, 5),
            n_rollouts=8,
            seed=0,
            sampler=PlantedSampler(table),
            solve_fn=StubSolver(window_start=1, full_horizon=4),
            chips=["bb1", "tc1"],
            joint_gws_per_chip=2,
        )

    def test_no_schedule_places_two_chips_in_one_gw(self):
        report = self.run()
        assert report.schedules
        for schedule in report.schedules:
            placed = list(schedule.placements.values())
            assert len(placed) == len(set(placed)), schedule

    def test_conflict_resolved_toward_the_bigger_gain(self):
        report = self.run()
        top = report.schedules[0]
        # BB@3 is worth 12, TC@3 only 9 -> BB takes GW3, TC settles for its next-best GW
        assert top.placements["bb1"] == 3
        assert top.placements.get("tc1") != 3
        assert top.e_total_gain == pytest.approx(12.0 + 1.0)
        assert top.p_best == pytest.approx(1.0)

    def test_schedules_ranked_by_expected_total(self):
        report = self.run()
        totals = [s.e_total_gain for s in report.schedules]
        assert totals == sorted(totals, reverse=True)


# --------------------------------------------------------------------------------------
# Determinism, schema, honesty, input validation
# --------------------------------------------------------------------------------------


class TestReportContract:
    def run(self, seed: int) -> ChipSimReport:
        gws = [1, 2, 3, 4]
        xp, prices = make_frames(gws)
        table = {(CAPTAIN, 4): ("choice", [0.0, 20.0], [0.5, 0.5])}
        table |= {(c, 2): ("choice", [1.0, 5.0], [0.5, 0.5]) for c in BENCH}
        return simulate_chip_plans(
            xp,
            prices,
            None,
            window=range(1, 5),
            n_rollouts=300,
            seed=seed,
            sampler=PlantedSampler(table),
            solve_fn=StubSolver(window_start=1, full_horizon=4),
            chips=["bb1", "tc1"],
        )

    def test_deterministic_under_seed(self):
        a, b = self.run(seed=11), self.run(seed=11)
        # sim_seconds is wall-clock metadata; everything statistical must be identical
        assert a.model_dump(exclude={"sim_seconds"}) == b.model_dump(exclude={"sim_seconds"})
        pd.testing.assert_frame_equal(a.to_frame(), b.to_frame())

    def test_seed_changes_the_draws(self):
        # compare the full statistical payload: a collision across every planted
        # distribution at once is astronomically unlikely, so this cannot flake
        a, c = self.run(seed=11), self.run(seed=12)
        assert a.model_dump(exclude={"sim_seconds"}) != c.model_dump(exclude={"sim_seconds"})

    def test_to_frame_is_additive_chip_curves_v2(self):
        report = self.run(seed=0)
        frame = report.to_frame()
        v1_columns = [
            "chip",
            "gw",
            "objective",
            "delta_vs_no_chip",
            "evaluated",
            "skip_reason",
            "window_last_gw",
            "urgency",
        ]
        assert list(frame.columns) == v1_columns + ["sd", "p_best_week", "p_beats_hold", "n_rollouts"]
        assert len(frame) == 2 * 4  # 2 chips x 4 window GWs
        assert (frame["n_rollouts"] == 300).all()
        assert frame.attrs["baseline_objective"] == pytest.approx(BACKBONE_OBJECTIVE)
        evaluated = frame[frame["evaluated"]]
        assert np.allclose(
            evaluated["objective"], BACKBONE_OBJECTIVE + evaluated["delta_vs_no_chip"]
        )
        assert (frame["window_last_gw"] == 19).all()

    def test_backbone_quality_and_assumptions_surface(self):
        report = self.run(seed=0)
        assert report.backbone.gap == pytest.approx(0.01)
        assert report.backbone.solve_seconds == pytest.approx(1.0)
        assert report.backbone.status == "time_limit"
        assert report.backbone.horizon == 4
        joined = "\n".join(report.assumptions)
        for marker in ("fixed_backbone", "segment_approximation", "independence"):
            assert marker in joined
        assert report.n_rollouts == 300
        assert report.appearance_source in ("points_nonzero", "sample_minutes")

    def test_missing_gw_coverage_raises(self):
        xp, prices = make_frames([1, 2, 4])
        with pytest.raises(ValueError, match=r"\[3\]"):
            simulate_chip_plans(
                xp,
                prices,
                None,
                window=range(1, 5),
                n_rollouts=4,
                sampler=PlantedSampler(),
                solve_fn=StubSolver(window_start=1, full_horizon=4),
            )

    def test_window_must_start_at_current_gw(self):
        xp, prices = make_frames([1, 2, 3])
        with pytest.raises(ValueError, match="current_gw"):
            simulate_chip_plans(
                xp,
                prices,
                make_state(current_gw=2),
                window=range(1, 4),
                n_rollouts=4,
                sampler=PlantedSampler(),
                solve_fn=StubSolver(window_start=1, full_horizon=3),
            )

    def test_bad_window_and_rollouts_raise(self):
        xp, prices = make_frames([1, 2, 3])
        stub = StubSolver(window_start=1, full_horizon=3)
        with pytest.raises(ValueError, match="contiguous"):
            simulate_chip_plans(
                xp, prices, None, window=[1, 3], n_rollouts=4,  # type: ignore[arg-type]
                sampler=PlantedSampler(), solve_fn=stub,
            )
        with pytest.raises(ValueError, match="n_rollouts"):
            simulate_chip_plans(
                xp, prices, None, window=range(1, 4), n_rollouts=0,
                sampler=PlantedSampler(), solve_fn=stub,
            )

    def test_sampler_shape_mismatch_raises(self):
        xp, prices = make_frames([1, 2, 3])

        class BadSampler:
            def sample(self, predictions: pd.DataFrame, n: int, seed: int) -> np.ndarray:
                del seed
                return np.zeros((len(predictions), n))  # transposed

        with pytest.raises(ValueError, match="shape"):
            simulate_chip_plans(
                xp,
                prices,
                None,
                window=range(1, 4),
                n_rollouts=4,
                sampler=BadSampler(),
                solve_fn=StubSolver(window_start=1, full_horizon=3),
            )


# --------------------------------------------------------------------------------------
# Real-MILP smoke test (tiny pool, real backbone + segment re-solves)
# --------------------------------------------------------------------------------------


class TestRealMilp:
    def test_end_to_end_with_real_solver(self):
        codes = {
            **{c: "GKP" for c in range(201, 204)},
            **{c: "DEF" for c in range(204, 210)},
            **{c: "MID" for c in range(210, 216)},
            **{c: "FWD" for c in range(216, 220)},
        }
        gws = [1, 2, 3, 4]
        xp = pd.DataFrame(
            [
                {
                    "season": SEASON,
                    "gw": g,
                    "player_code": c,
                    "xp": ((c * 7 + g * 13) % 40) / 10 + 0.5,
                    "q0": 0.1,
                }
                for g in gws
                for c in codes
            ]
        )
        prices = pd.DataFrame(
            [
                {"player_code": c, "price": 45 + (c % 7), "position": p, "team_code": 1 + c % 10}
                for c, p in codes.items()
            ]
        )
        report = simulate_chip_plans(
            xp,
            prices,
            None,
            window=range(1, 5),
            n_rollouts=32,
            seed=5,
            sampler=PlantedSampler(default=1.0),
            backbone_time_limit_s=20.0,
            segment_time_limit_s=8.0,
        )
        assert sorted(report.chips) == ["bb1", "fh1", "tc1", "wc1"]
        assert report.backbone.gap >= 0.0
        assert np.isfinite(report.backbone.objective)
        # transfer chips can't be placed in the initial-build GW1 (window opens GW2)
        for chip in ("wc1", "fh1"):
            assert stat_of(report, chip, 1).skip_reason == "outside_window"
        # BB/TC are analytic on the backbone and evaluated at every window GW
        for chip in ("bb1", "tc1"):
            for g in gws:
                s = stat_of(report, chip, g)
                assert s.evaluated
                assert s.e_gain is not None and np.isfinite(s.e_gain)
        for schedule in report.schedules:
            placed = list(schedule.placements.values())
            assert len(placed) == len(set(placed))
        frame = report.to_frame()
        assert len(frame) == 4 * len(gws)
        assert report.n_segment_solves > 0


# --------------------------------------------------------------------------------------
# Integration with the real PointsSampler (built in parallel; skip when absent)
# --------------------------------------------------------------------------------------


PREDICTIONS_PATH = config.PROCESSED_DIR / "predictions.parquet"


@pytest.mark.skipif(not PREDICTIONS_PATH.exists(), reason="predictions.parquet not built")
def test_real_sampler_integration():
    sampler_mod = pytest.importorskip("fplai.models.sampler")
    sampler_cls = getattr(sampler_mod, "PointsSampler", None)
    if sampler_cls is None:
        pytest.skip("models.sampler exists but PointsSampler is not defined yet")
    try:
        sampler = sampler_cls()
    except TypeError as exc:  # ctor contract drifted in the parallel build
        pytest.skip(f"PointsSampler() could not be constructed: {exc}")

    pred = pd.read_parquet(PREDICTIONS_PATH)
    pred = pred[pred["gw"].isin([1, 2, 3])]
    if pred.empty:
        pytest.skip("predictions.parquet has no rows for GWs 1-3")
    top_teams = pred["team_code"].value_counts().index[:8]
    pred = pred[pred["team_code"].isin(top_teams)].copy()
    prices = (
        pred.drop_duplicates("player_code")[["player_code", "price", "position", "team_code"]]
        .reset_index(drop=True)
    )
    report = simulate_chip_plans(
        pred,
        prices,
        None,
        window=range(1, 4),
        n_rollouts=40,
        seed=1,
        sampler=sampler,
        backbone_time_limit_s=10.0,
        segment_time_limit_s=5.0,
    )
    assert report.appearance_source in ("sample_minutes", "points_nonzero")
    for g in (1, 2, 3):
        s = stat_of(report, "bb1", g)
        assert s.evaluated
        assert s.e_gain is not None and s.e_gain >= 0.0
        assert s.sd is not None and s.sd >= 0.0
    assert stat_of(report, "wc1", 1).skip_reason == "outside_window"
    frame = report.to_frame()
    assert (frame["n_rollouts"] == 40).all()
