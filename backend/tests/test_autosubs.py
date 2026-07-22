"""Tests for optimizer.autosubs — MC autosub weights vs hand-computed exact probabilities.

All tests are offline and deterministic (fixed seeds). Exact reference values are
enumerated by hand in comments per FPL_KNOWLEDGE §1.5.
"""

from __future__ import annotations

import numpy as np
import pytest

from fplai.optimizer.autosubs import bench_weights_mc, expected_autosub_points

# Player-code layout used throughout: 1 = starting GK, then DEF, MID, FWD blocks;
# bench codes 12 (GK), 13-15 (outfield, priority order = insertion order).


def build_xi(
    n_def: int, n_mid: int, n_fwd: int, q0_overrides: dict[int, float] | None = None
) -> tuple[dict[int, float], dict[int, str], dict[str, int]]:
    """Build (lineup_q0, positions, formation) for a GK + n_def/n_mid/n_fwd XI."""
    positions: dict[int, str] = {1: "GKP"}
    code = 2
    for pos, count in (("DEF", n_def), ("MID", n_mid), ("FWD", n_fwd)):
        for _ in range(count):
            positions[code] = pos
            code += 1
    lineup_q0 = {c: 0.0 for c in positions}
    for c, q in (q0_overrides or {}).items():
        lineup_q0[c] = q
    formation = {"GKP": 1, "DEF": n_def, "MID": n_mid, "FWD": n_fwd}
    return lineup_q0, positions, formation


def add_bench(positions: dict[int, str], bench_spec: list[tuple[str, float]]) -> dict[int, float]:
    """Register bench players (GK first) into ``positions``; return their q0 mapping."""
    bench_q0: dict[int, float] = {}
    for i, (pos, q0) in enumerate(bench_spec):
        code = 12 + i
        positions[code] = pos
        bench_q0[code] = q0
    return bench_q0


class TestTwoAbsenceExact:
    """4-4-2 with DEF q0=0.5 and MID q0=0.3; bench [GK, MID, DEF, FWD] all certain.

    Enumeration (DEF codes 2-5, MID 6-9; absents: DEF=2 p=0.5, MID=6 p=0.3):
      neither absent (0.35): no subs.
      DEF only (0.5*0.7=0.35): bench MID (slot 1) replaces the DEF -> 3-5-2 legal.
      MID only (0.5*0.3=0.15): bench MID (slot 1) replaces the MID.
      both (0.15): slot 1 covers the DEF (team-sheet order), then bench DEF (slot 2)
        covers the MID (4-4-2 -> 4-4-2 via 3-5-2 -> legal at each step).
    Exact: P(slot1)=0.65, P(slot2)=0.15, P(slot3)=0, P(GK slot)=0.
    """

    def test_matches_enumeration(self) -> None:
        lineup_q0, positions, formation = build_xi(4, 4, 2, {2: 0.5, 6: 0.3})
        bench_q0 = add_bench(positions, [("GKP", 0.0), ("MID", 0.0), ("DEF", 0.0), ("FWD", 0.0)])
        probs = bench_weights_mc(
            lineup_q0, bench_q0, formation, positions=positions, n=20_000, seed=42
        )
        exact = np.array([0.0, 0.65, 0.15, 0.0])
        np.testing.assert_allclose(probs, exact, atol=0.02)


class TestFormationLegality:
    def test_illegal_formation_blocks_sub(self) -> None:
        """3-4-3 with a DEF certainly absent and no DEF on the bench: DEF would drop to
        2 (< 3), so no bench player may enter — all slots stay at zero."""
        lineup_q0, positions, formation = build_xi(3, 4, 3, {2: 1.0})
        bench_q0 = add_bench(positions, [("GKP", 0.0), ("MID", 0.0), ("FWD", 0.0), ("MID", 0.0)])
        probs = bench_weights_mc(
            lineup_q0, bench_q0, formation, positions=positions, n=2000, seed=0
        )
        np.testing.assert_array_equal(probs, np.zeros(4))

    def test_same_position_sub_always_legal(self) -> None:
        """3-4-3 DEF absence IS covered when the bench has a DEF (3-4-3 preserved)."""
        lineup_q0, positions, formation = build_xi(3, 4, 3, {2: 1.0})
        bench_q0 = add_bench(positions, [("GKP", 0.0), ("DEF", 0.0), ("MID", 0.0), ("FWD", 0.0)])
        probs = bench_weights_mc(
            lineup_q0, bench_q0, formation, positions=positions, n=2000, seed=0
        )
        np.testing.assert_array_equal(probs, np.array([0.0, 1.0, 0.0, 0.0]))


class TestGkSwapOnly:
    def test_gk_replaced_only_by_bench_gk(self) -> None:
        lineup_q0, positions, formation = build_xi(4, 4, 2, {1: 1.0})
        bench_q0 = add_bench(positions, [("GKP", 0.0), ("MID", 0.0), ("DEF", 0.0), ("FWD", 0.0)])
        probs = bench_weights_mc(
            lineup_q0, bench_q0, formation, positions=positions, n=2000, seed=1
        )
        np.testing.assert_array_equal(probs, np.array([1.0, 0.0, 0.0, 0.0]))

    def test_bench_gk_never_covers_outfield(self) -> None:
        """MID certainly absent, outfield bench never plays: the playing bench GK must
        NOT come in — everyone stays at zero."""
        lineup_q0, positions, formation = build_xi(4, 4, 2, {6: 1.0})
        bench_q0 = add_bench(positions, [("GKP", 0.0), ("MID", 1.0), ("DEF", 1.0), ("FWD", 1.0)])
        probs = bench_weights_mc(
            lineup_q0, bench_q0, formation, positions=positions, n=2000, seed=2
        )
        np.testing.assert_array_equal(probs, np.zeros(4))

    def test_absent_bench_gk_cannot_sub(self) -> None:
        lineup_q0, positions, formation = build_xi(4, 4, 2, {1: 1.0})
        bench_q0 = add_bench(positions, [("GKP", 1.0), ("MID", 0.0), ("DEF", 0.0), ("FWD", 0.0)])
        probs = bench_weights_mc(
            lineup_q0, bench_q0, formation, positions=positions, n=2000, seed=3
        )
        np.testing.assert_array_equal(probs, np.zeros(4))


class TestBenchBoost:
    def test_all_slots_score(self) -> None:
        lineup_q0, positions, formation = build_xi(4, 4, 2, {2: 0.9, 6: 0.9})
        bench_q0 = add_bench(positions, [("GKP", 0.5), ("MID", 0.5), ("DEF", 0.5), ("FWD", 0.5)])
        probs = bench_weights_mc(
            lineup_q0,
            bench_q0,
            formation,
            positions=positions,
            n=100,
            seed=0,
            bench_boost=True,
        )
        np.testing.assert_array_equal(probs, np.ones(4))


class TestNineManScenarios:
    def test_two_def_absent_covered_by_def_and_mid(self) -> None:
        """4-4-2 with two DEF certainly out: bench DEF covers one, bench MID the other
        (4-4-2 -> 3-5-2, legal); the bench FWD is never needed."""
        lineup_q0, positions, formation = build_xi(4, 4, 2, {2: 1.0, 3: 1.0})
        bench_q0 = add_bench(positions, [("GKP", 0.0), ("DEF", 0.0), ("MID", 0.0), ("FWD", 0.0)])
        probs = bench_weights_mc(
            lineup_q0, bench_q0, formation, positions=positions, n=2000, seed=4
        )
        np.testing.assert_array_equal(probs, np.array([0.0, 1.0, 1.0, 0.0]))

    def test_three_def_absent_no_def_cover_plays_with_eight(self) -> None:
        """3-5-2 with all three DEF out and no bench DEF: every replacement would leave
        DEF < 3, so nobody enters — the team just plays with 8."""
        lineup_q0, positions, formation = build_xi(3, 5, 2, {2: 1.0, 3: 1.0, 4: 1.0})
        bench_q0 = add_bench(positions, [("GKP", 0.0), ("MID", 0.0), ("MID", 0.0), ("FWD", 0.0)])
        probs = bench_weights_mc(
            lineup_q0, bench_q0, formation, positions=positions, n=2000, seed=5
        )
        np.testing.assert_array_equal(probs, np.zeros(4))


class TestConditionalWeights:
    def test_weights_condition_on_bench_appearance(self) -> None:
        """MID certainly absent; bench slot 1 (MID) plays only 50% of the time.

        Exact: slot 1 is subbed in whenever it plays -> weight = P(sub | plays) = 1.0
        (unconditional 0.5); slot 2 (MID, always plays) covers the other half -> 0.5.
        """
        lineup_q0, positions, formation = build_xi(4, 4, 2, {6: 1.0})
        bench_q0 = add_bench(positions, [("GKP", 0.0), ("MID", 0.5), ("MID", 0.0), ("FWD", 0.0)])
        probs = bench_weights_mc(
            lineup_q0, bench_q0, formation, positions=positions, n=20_000, seed=6
        )
        np.testing.assert_allclose(probs, np.array([0.0, 1.0, 0.5, 0.0]), atol=0.03)

    def test_never_playing_bench_player_gets_zero_weight(self) -> None:
        lineup_q0, positions, formation = build_xi(4, 4, 2, {6: 1.0})
        bench_q0 = add_bench(positions, [("GKP", 0.0), ("MID", 1.0), ("MID", 0.0), ("FWD", 0.0)])
        probs = bench_weights_mc(
            lineup_q0, bench_q0, formation, positions=positions, n=2000, seed=7
        )
        assert probs[1] == 0.0
        assert probs[2] == 1.0


class TestApi:
    def test_deterministic_for_fixed_seed(self) -> None:
        lineup_q0, positions, formation = build_xi(4, 4, 2, {2: 0.4, 6: 0.2, 10: 0.3})
        bench_q0 = add_bench(positions, [("GKP", 0.1), ("MID", 0.2), ("DEF", 0.1), ("FWD", 0.3)])
        kwargs = dict(positions=positions, n=2000, seed=123)
        a = bench_weights_mc(lineup_q0, bench_q0, formation, **kwargs)
        b = bench_weights_mc(lineup_q0, bench_q0, formation, **kwargs)
        np.testing.assert_array_equal(a, b)

    def test_bench_order_controls_priority(self) -> None:
        """Promoting the DEF to first priority moves the sub from slot 2 to slot 1."""
        lineup_q0, positions, formation = build_xi(4, 4, 2, {2: 1.0})  # a DEF is out
        # Insertion order: 13=MID (never plays), 14=DEF, 15=FWD.
        bench_q0 = add_bench(positions, [("GKP", 0.0), ("MID", 1.0), ("DEF", 0.0), ("FWD", 0.0)])
        default = bench_weights_mc(
            lineup_q0, bench_q0, formation, positions=positions, n=500, seed=0
        )
        # Slot 1 (MID) never appears, so the DEF in slot 2 covers.
        np.testing.assert_array_equal(default, np.array([0.0, 0.0, 1.0, 0.0]))
        promoted = bench_weights_mc(
            lineup_q0,
            bench_q0,
            formation,
            positions=positions,
            bench_order=[14, 13, 15],  # DEF now first priority
            n=500,
            seed=0,
        )
        np.testing.assert_array_equal(promoted, np.array([0.0, 1.0, 0.0, 0.0]))

    def test_expected_autosub_points(self) -> None:
        assert expected_autosub_points([1.0, 2.0, 3.0, 4.0], [1.0, 0.5, 0.25, 0.0]) == (
            pytest.approx(2.75)
        )

    def test_validation_errors(self) -> None:
        lineup_q0, positions, formation = build_xi(4, 4, 2)
        bench_q0 = add_bench(positions, [("GKP", 0.0), ("MID", 0.0), ("DEF", 0.0), ("FWD", 0.0)])
        with pytest.raises(ValueError, match="does not match"):
            bench_weights_mc(
                lineup_q0,
                bench_q0,
                {"GKP": 1, "DEF": 3, "MID": 5, "FWD": 2},
                positions=positions,
            )
        with pytest.raises(ValueError, match="probability"):
            bad = dict(lineup_q0)
            bad[2] = 1.5
            bench_weights_mc(bad, bench_q0, formation, positions=positions)
        with pytest.raises(ValueError, match="4 bench"):
            bench_weights_mc(lineup_q0, {12: 0.0, 13: 0.0}, formation, positions=positions)
        with pytest.raises(ValueError, match="shape mismatch"):
            expected_autosub_points([1.0, 2.0], [1.0, 0.5, 0.25])
