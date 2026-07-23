"""Offline tests for fplai.optimizer.plans — the weekly-verdict orchestrator.

The solver seams (solve_plan / chip_ev_curves / plan_stability) are monkeypatched with
deterministic fakes, so these tests run without the sibling milp/chips/sensitivity
modules (built in parallel). A final integration class exercises the real MILP on a
small synthetic instance and skips cleanly while those modules are absent.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd
import pytest
from pydantic import BaseModel

from fplai.optimizer import plans
from fplai.optimizer.plans import (
    ChipAdvice,
    DreamTeam,
    Recommendation,
    StabilityEntry,
    TransferPair,
    build_chip_advice_from_sim,
    build_recommendation,
    confidence_from_p_beats_hold,
    dream_team,
    merge_chip_advice,
)

SEASON = 2026
GW = 5
AS_OF = datetime(2026, 8, 20, 12, 0, tzinfo=UTC)

# Player universe: user squad 101-115, transfer target 201, dream squad 301-315.
USER_SQUAD = list(range(101, 116))
DREAM_SQUAD = list(range(301, 316))

_POS = {}
for base in (101, 301):
    _POS.update({base: "GKP", base + 1: "GKP"})
    _POS.update({base + i: "DEF" for i in range(2, 7)})
    _POS.update({base + i: "MID" for i in range(7, 12)})
    _POS.update({base + i: "FWD" for i in range(12, 15)})
_POS[201] = "MID"

_NAMES = {201: "Nova", 109: "Oldman", 108: "Skipper", 113: "Vicente", 308: "Dreamcap"}


def make_prices() -> pd.DataFrame:
    codes = USER_SQUAD + [201] + DREAM_SQUAD
    return pd.DataFrame(
        {
            "player_code": codes,
            "price": [50 + (c % 7) * 5 for c in codes],
            "position": [_POS[c] for c in codes],
            "team_code": [1 + (c % 14) for c in codes],
            "web_name": [_NAMES.get(c, f"P{c}") for c in codes],
        }
    )


def make_xp() -> pd.DataFrame:
    """xp frame with planted numbers: 201 sums 21.3 (avg 7.1), 109 sums 11.9, q0 0.82."""
    rows: list[dict[str, Any]] = []
    planted_xp = {201: [7.1, 7.3, 6.9], 109: [4.0, 3.9, 4.0], 108: [8.6, 7.0, 7.2]}
    planted_q0 = {109: 0.82, 108: 0.05}
    for code in USER_SQUAD + [201] + DREAM_SQUAD:
        for i, gw in enumerate((GW, GW + 1, GW + 2)):
            xp_val = planted_xp.get(code, [3.0, 3.0, 3.0])[i]
            rows.append(
                {
                    "season": SEASON,
                    "gw": gw,
                    "player_code": code,
                    "xp": xp_val,
                    "q0": planted_q0.get(code, 0.02),
                }
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Stub solver components (deterministic fakes matching the ARCHITECTURE contracts)
# --------------------------------------------------------------------------------------


class StubGwPlan(BaseModel):
    gw: int
    squad: list[int]
    lineup: list[int]
    bench_order: list[int]
    captain: int
    vice: int
    transfers_in: list[int] = []
    transfers_out: list[int] = []
    hit_points: int = 0
    chip: str | None = None
    expected_points: float = 0.0


class StubPlanResult(BaseModel):
    objective: float
    gws: list[StubGwPlan]
    solve_seconds: float = 0.01
    gap: float = 0.0


@dataclass
class FakeState:
    """Duck-typed SquadState (the real pydantic model is built in parallel)."""

    season: int = SEASON
    current_gw: int = GW
    squad: list[Any] = field(default_factory=list)
    bank: int = 15
    free_transfers: int = 2
    chips_available: list[str] = field(default_factory=lambda: ["wc1", "fh1", "bb1", "tc1"])
    active_chip: str | None = None


def _user_gw(gw: int, **over: Any) -> StubGwPlan:
    squad = [c for c in USER_SQUAD if c != 109] + [201]
    kwargs: dict[str, Any] = {
        "gw": gw,
        "squad": squad,
        "lineup": [101, 103, 104, 105, 108, 110, 111, 201, 113, 114, 115],
        "bench_order": [102, 106, 107, 112],
        "captain": 108,
        "vice": 113,
        "expected_points": 57.3,
    }
    kwargs.update(over)
    return StubGwPlan(**kwargs)


def _hold_gw(gw: int, **over: Any) -> StubGwPlan:
    kwargs: dict[str, Any] = {
        "gw": gw,
        "squad": list(USER_SQUAD),
        "lineup": [101, 103, 104, 105, 108, 109, 110, 111, 113, 114, 115],
        "bench_order": [102, 106, 107, 112],
        "captain": 108,
        "vice": 113,
        "expected_points": 55.0,
    }
    kwargs.update(over)
    return StubGwPlan(**kwargs)


DREAM_PLAN = StubPlanResult(
    objective=130.0,
    gws=[
        StubGwPlan(
            gw=GW,
            squad=list(DREAM_SQUAD),
            lineup=[301, 303, 304, 305, 308, 309, 310, 311, 313, 314, 315],
            bench_order=[302, 306, 307, 312],
            captain=308,
            vice=313,
            expected_points=65.0,
        )
    ],
    solve_seconds=0.02,
)

TRANSFER_PLAN = StubPlanResult(
    objective=123.4,
    gws=[
        _user_gw(GW, transfers_in=[201], transfers_out=[109]),
        _user_gw(GW + 1),
        _user_gw(GW + 2),
    ],
)

HOLD_PLAN = StubPlanResult(objective=118.0, gws=[_hold_gw(GW), _hold_gw(GW + 1)])

CHIP_PLAN = StubPlanResult(objective=125.0, gws=[_hold_gw(GW, chip="bb1"), _hold_gw(GW + 1)])

CURVES = pd.DataFrame(
    {
        "chip": ["bb1", "bb1", "tc1"],
        "gw": [GW, GW + 2, GW],
        "objective": [119.2, 124.8, 122.2],
        "delta_vs_no_chip": [1.2, 6.8, 4.2],
    }
)

STABILITY = pd.DataFrame(
    {"move": ["109 -> 201", "109 -> 202"], "support_pct": [74.0, 12.0]}
)


def _patch(monkeypatch: pytest.MonkeyPatch, user_plan: StubPlanResult) -> list[Any]:
    """Patch the three solver seams; return the list of states solve_plan was called with."""
    calls: list[Any] = []

    def stub_solve(xp: Any, prices: Any, state: Any, **kwargs: Any) -> StubPlanResult:
        calls.append(state)
        return DREAM_PLAN if state is None else user_plan

    monkeypatch.setattr(plans, "solve_plan", stub_solve)
    monkeypatch.setattr(plans, "chip_ev_curves", lambda *a, **k: CURVES.copy())
    monkeypatch.setattr(plans, "plan_stability", lambda *a, **k: STABILITY.copy())
    return calls


# --------------------------------------------------------------------------------------
# Action selection
# --------------------------------------------------------------------------------------


class TestActionSelection:
    def test_transfer_action(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch(monkeypatch, TRANSFER_PLAN)
        rec = build_recommendation(
            FakeState(), make_xp(), make_prices(), as_of=AS_OF, horizon=3
        )
        assert rec.action == "transfer"
        assert rec.gw == GW
        assert rec.season == SEASON
        assert rec.hits == 0

    def test_hold_action(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch(monkeypatch, HOLD_PLAN)
        rec = build_recommendation(
            FakeState(), make_xp(), make_prices(), as_of=AS_OF, horizon=3
        )
        assert rec.action == "hold"
        assert rec.transfers == []

    def test_chip_action_takes_precedence(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch(monkeypatch, CHIP_PLAN)
        rec = build_recommendation(
            FakeState(), make_xp(), make_prices(), as_of=AS_OF, horizon=3
        )
        assert rec.action == "chip:bb1"
        bb1 = next(a for a in rec.chip_advice if a.chip == "bb1")
        assert bb1.verdict == "play"
        assert bb1.planned_gw == GW
        assert bb1.ev_now == pytest.approx(1.2)

    def test_initial_squad_action(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = _patch(monkeypatch, TRANSFER_PLAN)
        rec = build_recommendation(None, make_xp(), make_prices(), as_of=AS_OF, horizon=3)
        assert rec.action == "initial-squad"
        assert sorted(rec.squad) == DREAM_SQUAD
        assert len(rec.squad) == 15
        assert len(rec.lineup) == 11
        assert len(rec.bench_order) == 4
        assert rec.captain == 308
        assert rec.vice == 313
        assert rec.formation == "3-4-3"
        # the None-state plan doubles as the dream team: exactly one solve call
        assert calls == [None]
        assert rec.dream_team is not None
        assert rec.dream_team.expected_points == pytest.approx(65.0)


# --------------------------------------------------------------------------------------
# Transfers, deltas and stability support
# --------------------------------------------------------------------------------------


class TestTransferDistillation:
    def test_pair_and_deltas(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch(monkeypatch, TRANSFER_PLAN)
        rec = build_recommendation(
            FakeState(), make_xp(), make_prices(), as_of=AS_OF, horizon=3
        )
        assert len(rec.transfers) == 1
        pair = rec.transfers[0]
        assert pair.player_in == 201
        assert pair.player_out == 109
        assert pair.player_in_name == "Nova"
        assert pair.player_out_name == "Oldman"
        assert pair.position == "MID"
        assert pair.xp_in == pytest.approx(21.3)
        assert pair.xp_out == pytest.approx(11.9)
        assert pair.xp_delta == pytest.approx(9.4)
        assert pair.out_q0 == pytest.approx(0.82)
        assert pair.support_pct == pytest.approx(74.0)

    def test_stability_entries_present(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch(monkeypatch, TRANSFER_PLAN)
        rec = build_recommendation(
            FakeState(), make_xp(), make_prices(), as_of=AS_OF, horizon=3
        )
        assert [e.move for e in rec.stability] == ["109 -> 201", "109 -> 202"]
        assert rec.stability[0].support_pct == pytest.approx(74.0)

    def test_dream_team_benchmark_present(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = _patch(monkeypatch, TRANSFER_PLAN)
        rec = build_recommendation(
            FakeState(), make_xp(), make_prices(), as_of=AS_OF, horizon=3
        )
        assert rec.dream_team is not None
        assert rec.dream_team.expected_points == pytest.approx(65.0)
        assert sorted(rec.dream_team.squad) == DREAM_SQUAD
        # solved twice: once with the user state, once with None for the benchmark
        assert FakeState() in [c for c in calls if c is not None] or len(calls) == 2
        assert None in calls

    def test_chip_advice_hold_values(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch(monkeypatch, TRANSFER_PLAN)
        rec = build_recommendation(
            FakeState(), make_xp(), make_prices(), as_of=AS_OF, horizon=3
        )
        assert {a.chip for a in rec.chip_advice} == {"wc1", "fh1", "bb1", "tc1"}
        bb1 = next(a for a in rec.chip_advice if a.chip == "bb1")
        assert bb1.verdict == "hold"
        assert bb1.best_gw == GW + 2
        assert bb1.best_ev == pytest.approx(6.8)
        assert bb1.ev_now == pytest.approx(1.2)
        tc1 = next(a for a in rec.chip_advice if a.chip == "tc1")
        assert tc1.best_gw == GW
        assert tc1.best_ev == pytest.approx(4.2)


# --------------------------------------------------------------------------------------
# Chip advice v2: simulation overlay (duck-typed ChipSimReport — built in parallel)
# --------------------------------------------------------------------------------------


class StubSimReport:
    """Duck-typed ChipSimReport per the ARCHITECTURE stage-6 contract.

    ``to_frame()`` yields the chip_curves.parquet v2 shape (chip, gw,
    delta_vs_no_chip + sd, p_best_week, p_beats_hold, n_rollouts); report-level
    attributes carry current_gw / n_rollouts / assumptions.
    """

    def __init__(
        self,
        frame: pd.DataFrame,
        *,
        current_gw: int = GW,
        n_rollouts: int = 1000,
        assumptions: list[str] | None = None,
    ) -> None:
        self._frame = frame
        self.current_gw = current_gw
        self.n_rollouts = n_rollouts
        self.assumptions = assumptions or [
            "Predictions frozen at the last refresh",
            "Autosubs and vice resolved per sampled season",
        ]

    def to_frame(self) -> pd.DataFrame:
        return self._frame.copy()


def make_sim_frame() -> pd.DataFrame:
    """v2 frame over GWs 5-7: bb1 (hold, best window GW7) and tc1 (play now)."""
    return pd.DataFrame(
        {
            "chip": ["bb1", "bb1", "bb1", "tc1", "tc1", "tc1"],
            "gw": [GW, GW + 1, GW + 2, GW, GW + 1, GW + 2],
            "delta_vs_no_chip": [1.2, 3.0, 6.8, 4.2, 3.1, 2.6],
            "sd": [2.0, 2.2, 3.1, 1.8, 1.7, 1.6],
            "p_best_week": [0.10, 0.20, 0.28, 0.44, 0.22, 0.18],
            "p_beats_hold": [0.34, 0.48, 0.62, 0.72, 0.51, 0.40],
            "n_rollouts": [1000] * 6,
        }
    )


class TestChipAdviceFromSim:
    def test_hold_chip_fields(self) -> None:
        advice = build_chip_advice_from_sim(StubSimReport(make_sim_frame()))
        assert [a.chip for a in advice] == ["bb1", "tc1"]
        bb1 = advice[0]
        assert bb1.verdict == "hold"  # p_beats_hold now = 0.34 <= 0.5
        assert bb1.e_gain_now == pytest.approx(1.2)
        assert bb1.ev_now == pytest.approx(1.2)  # pre-v2 field kept in sync
        assert bb1.sd == pytest.approx(2.0)
        assert bb1.p_best_week_now == pytest.approx(0.10)
        assert bb1.p_beats_hold == pytest.approx(0.34)
        assert bb1.recommended_gw == GW + 2  # argmax p_best_week
        assert bb1.p_best_week_reco == pytest.approx(0.28)
        assert bb1.confidence == "medium"  # |0.34 - 0.5| = 0.16
        assert bb1.n_rollouts == 1000
        assert bb1.assumptions is not None and len(bb1.assumptions) == 2
        assert bb1.best_gw == GW + 2
        assert bb1.best_ev == pytest.approx(6.8)
        assert bb1.planned_gw is None

    def test_play_chip_verdict(self) -> None:
        advice = build_chip_advice_from_sim(StubSimReport(make_sim_frame()))
        tc1 = advice[1]
        assert tc1.verdict == "play"  # p_beats_hold now = 0.72 > 0.5
        assert tc1.confidence == "high"  # |0.72 - 0.5| = 0.22
        assert tc1.recommended_gw == GW  # argmax p_best_week is now
        assert tc1.p_best_week_reco == pytest.approx(0.44)

    def test_bare_frame_accepted(self) -> None:
        """A raw v2 DataFrame works too — current GW defaults to the min GW."""
        advice = build_chip_advice_from_sim(make_sim_frame())
        bb1 = advice[0]
        assert bb1.p_beats_hold == pytest.approx(0.34)
        assert bb1.n_rollouts == 1000  # picked up from the frame column

    def test_recommended_falls_back_to_gain(self) -> None:
        frame = make_sim_frame().drop(columns=["p_best_week"])
        advice = build_chip_advice_from_sim(StubSimReport(frame))
        bb1 = advice[0]
        assert bb1.recommended_gw == GW + 2  # argmax delta_vs_no_chip
        assert bb1.p_best_week_reco is None

    def test_missing_columns_raise(self) -> None:
        with pytest.raises(ValueError, match="missing required column"):
            build_chip_advice_from_sim(pd.DataFrame({"chip": ["bb1"]}))
        with pytest.raises(TypeError, match="to_frame"):
            build_chip_advice_from_sim(object())

    def test_empty_frame_gives_no_advice(self) -> None:
        frame = make_sim_frame().iloc[0:0]
        assert build_chip_advice_from_sim(StubSimReport(frame)) == []

    def test_confidence_bands(self) -> None:
        assert confidence_from_p_beats_hold(0.72) == "high"
        assert confidence_from_p_beats_hold(0.30) == "high"
        assert confidence_from_p_beats_hold(0.34) == "medium"
        assert confidence_from_p_beats_hold(0.62) == "medium"
        assert confidence_from_p_beats_hold(0.55) == "low"
        assert confidence_from_p_beats_hold(0.50) == "low"

    def test_sim_advice_round_trips(self) -> None:
        for advice in build_chip_advice_from_sim(StubSimReport(make_sim_frame())):
            assert ChipAdvice.model_validate_json(advice.model_dump_json()) == advice

    def test_report_verdicts_and_window_first_gw_honored(self) -> None:
        """The real ChipSimReport surface: window_first_gw + per-chip verdict objects
        (recommended_gw; float confidence = p_best_week at that GW) take precedence."""

        @dataclass
        class FakeVerdict:
            chip: str
            recommended_gw: int | None
            confidence: float | None = None

        @dataclass
        class FakeReport:
            frame: pd.DataFrame
            window_first_gw: int = GW
            n_rollouts: int = 500
            assumptions: tuple[str, ...] = ("Backbone squad frozen over the window",)
            verdicts: tuple[FakeVerdict, ...] = ()

            def to_frame(self) -> pd.DataFrame:
                return self.frame

        frame = make_sim_frame().drop(columns=["n_rollouts"])
        report = FakeReport(
            frame,
            verdicts=(FakeVerdict("bb1", GW + 1, 0.20), FakeVerdict("tc1", None)),
        )
        advice = build_chip_advice_from_sim(report)
        bb1, tc1 = advice
        assert bb1.recommended_gw == GW + 1  # report verdict beats the argmax (GW+2)
        assert bb1.p_best_week_reco == pytest.approx(0.20)  # frame value at GW+1
        assert bb1.n_rollouts == 500  # report attr when the column is absent
        assert bb1.assumptions == ["Backbone squad frozen over the window"]
        assert bb1.p_beats_hold == pytest.approx(0.34)  # at window_first_gw
        assert tc1.recommended_gw == GW  # None in the verdict -> argmax p_best_week


class TestMergeChipAdvice:
    def _base(self) -> list[ChipAdvice]:
        return [
            ChipAdvice(chip="wc1", verdict="hold", best_gw=GW + 1, best_ev=4.1),
            ChipAdvice(chip="bb1", verdict="play", planned_gw=GW, ev_now=1.2),
        ]

    def test_overlay_and_verdict_override(self) -> None:
        sim = build_chip_advice_from_sim(StubSimReport(make_sim_frame()))
        merged = merge_chip_advice(self._base(), sim)
        assert [a.chip for a in merged] == ["wc1", "bb1"]  # base order + membership
        wc1, bb1 = merged
        # wc1 has no sim entry: untouched
        assert wc1.p_beats_hold is None
        assert wc1.best_ev == pytest.approx(4.1)
        # bb1: sim verdict wins (greedy plan said play, sim says hold), fields overlaid
        assert bb1.verdict == "hold"
        assert bb1.planned_gw == GW  # the plan's schedule is preserved
        assert bb1.p_beats_hold == pytest.approx(0.34)
        assert bb1.recommended_gw == GW + 2
        assert bb1.confidence == "medium"
        assert bb1.ev_now == pytest.approx(1.2)  # sim e_gain matches; still filled
        assert bb1.best_gw == GW + 2

    def test_sim_only_chips_are_not_added(self) -> None:
        sim = build_chip_advice_from_sim(StubSimReport(make_sim_frame()))
        merged = merge_chip_advice([self._base()[0]], sim)  # wc1 only
        assert [a.chip for a in merged] == ["wc1"]


class TestBuildRecommendationWithSim:
    def test_chip_sim_hook_folds_into_advice_and_rationale(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch(monkeypatch, TRANSFER_PLAN)
        rec = build_recommendation(
            FakeState(),
            make_xp(),
            make_prices(),
            as_of=AS_OF,
            horizon=3,
            chip_sim=StubSimReport(make_sim_frame()),
        )
        bb1 = next(a for a in rec.chip_advice if a.chip == "bb1")
        assert bb1.p_beats_hold == pytest.approx(0.34)
        assert bb1.confidence == "medium"
        assert bb1.n_rollouts == 1000
        tc1 = next(a for a in rec.chip_advice if a.chip == "tc1")
        assert tc1.verdict == "play"
        # chips without sim rows keep their EV-only advice
        wc1 = next(a for a in rec.chip_advice if a.chip == "wc1")
        assert wc1.p_beats_hold is None
        text = " | ".join(rec.rationale)
        assert (
            "Bench Boost (bb1) now beats holding in 34% of 1,000 simulated seasons "
            f"— hold; best window GW{GW + 2} (P(best)=0.28)." in text
        )
        assert (
            "Triple Captain (tc1) now beats holding in 72% of 1,000 simulated "
            f"seasons — play; best window GW{GW} (P(best)=0.44)." in text
        )
        # the whole thing still serializes and round-trips
        restored = Recommendation.model_validate_json(rec.model_dump_json())
        assert restored == rec

    def test_without_sim_nothing_changes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch(monkeypatch, TRANSFER_PLAN)
        rec = build_recommendation(
            FakeState(), make_xp(), make_prices(), as_of=AS_OF, horizon=3
        )
        assert all(a.p_beats_hold is None for a in rec.chip_advice)
        assert all(a.assumptions is None for a in rec.chip_advice)
        assert "simulated seasons" not in " | ".join(rec.rationale)


# --------------------------------------------------------------------------------------
# Rationale bullets contain the planted numbers
# --------------------------------------------------------------------------------------


class TestRationale:
    def _text(self, rec: Recommendation) -> str:
        return " | ".join(rec.rationale)

    def test_transfer_rationale_numbers(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch(monkeypatch, TRANSFER_PLAN)
        rec = build_recommendation(
            FakeState(), make_xp(), make_prices(), as_of=AS_OF, horizon=3
        )
        text = self._text(rec)
        assert "Nova" in text
        assert "Oldman" in text
        assert "7.1 xP/GW next 3" in text  # fixture run of the incoming player
        assert "+9.4 xP over the horizon" in text  # horizon delta
        assert "q0 82%" in text  # availability flag from the xp input's q0
        assert "74% of noise re-solves agree" in text  # stability support
        assert "Captain Skipper (8.6 xP this GW)" in text
        assert "captain q0 5%" in text
        # dream-team benchmark comparison: 65.0 vs 57.3 = -7.7
        assert "65.0" in text and "57.3" in text and "-7.7" in text

    def test_hold_rationale_ft_bank(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch(monkeypatch, HOLD_PLAN)
        rec = build_recommendation(
            FakeState(free_transfers=2), make_xp(), make_prices(), as_of=AS_OF, horizon=3
        )
        text = self._text(rec)
        assert "0 transfers" in text
        assert "(2 → 3)" in text  # rules.next_free_transfers(2, 0, 2026, False) == 3

    def test_chip_rationale(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch(monkeypatch, CHIP_PLAN)
        rec = build_recommendation(
            FakeState(), make_xp(), make_prices(), as_of=AS_OF, horizon=3
        )
        text = self._text(rec)
        assert "Play Bench Boost (bb1) this GW" in text
        assert "+1.2 xP vs the best no-chip plan" in text
        assert f"GW{GW + 2}: +6.8" in text  # best future window comparison

    def test_hold_chip_window_bullet(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch(monkeypatch, TRANSFER_PLAN)
        rec = build_recommendation(
            FakeState(), make_xp(), make_prices(), as_of=AS_OF, horizon=3
        )
        text = self._text(rec)
        assert f"Bench Boost (bb1) in GW{GW + 2} (+6.8 xP vs no chip)" in text

    def test_initial_squad_rationale(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch(monkeypatch, TRANSFER_PLAN)
        prices = make_prices()
        rec = build_recommendation(None, make_xp(), prices, as_of=AS_OF, horizon=3)
        text = self._text(rec)
        cost = int(
            prices.set_index("player_code").loc[DREAM_SQUAD, "price"].sum()
        )
        assert f"Initial squad for GW{GW}" in text
        assert f"£{cost / 10:.1f}m spent" in text
        assert "65.0 xP" in text


# --------------------------------------------------------------------------------------
# Standalone dream team
# --------------------------------------------------------------------------------------


class TestDreamTeam:
    def test_standalone(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch(monkeypatch, TRANSFER_PLAN)
        prices = make_prices()
        dt = dream_team(make_xp(), prices, GW, SEASON)
        assert isinstance(dt, DreamTeam)
        assert dt.gw == GW
        assert dt.season == SEASON
        assert sorted(dt.squad) == DREAM_SQUAD
        assert dt.captain == 308
        assert dt.formation == "3-4-3"
        assert dt.expected_points == pytest.approx(65.0)
        assert dt.objective == pytest.approx(130.0)
        expected_cost = int(prices.set_index("player_code").loc[DREAM_SQUAD, "price"].sum())
        assert dt.total_cost == expected_cost

    def test_no_rows_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch(monkeypatch, TRANSFER_PLAN)
        with pytest.raises(ValueError, match="no xp rows"):
            dream_team(make_xp(), make_prices(), 99, SEASON)


# --------------------------------------------------------------------------------------
# JSON serialization
# --------------------------------------------------------------------------------------


class TestSerialization:
    def test_model_dump_json_clean(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch(monkeypatch, TRANSFER_PLAN)
        rec = build_recommendation(
            FakeState(), make_xp(), make_prices(), as_of=AS_OF, horizon=3
        )
        payload = rec.model_dump(mode="json")
        text = json.dumps(payload)  # must not raise
        parsed = json.loads(text)
        assert parsed["action"] == "transfer"
        assert parsed["as_of"].startswith("2026-08-20T12:00:00")
        assert parsed["plan"]["objective"] == pytest.approx(123.4)
        assert parsed["plan"]["gws"][0]["transfers_in"] == [201]
        assert parsed["dream_team"]["expected_points"] == pytest.approx(65.0)

    def test_round_trip(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch(monkeypatch, TRANSFER_PLAN)
        rec = build_recommendation(
            FakeState(), make_xp(), make_prices(), as_of=AS_OF, horizon=3
        )
        restored = Recommendation.model_validate_json(rec.model_dump_json())
        assert restored == rec

    def test_submodels_round_trip(self) -> None:
        models: list[Any] = [
            TransferPair(player_in=1, player_out=2, xp_delta=1.5),
            ChipAdvice(chip="wc1", verdict="hold", best_gw=7, best_ev=3.2),
            StabilityEntry(move="1 -> 2", support_pct=50.0),
            DreamTeam(season=2026, gw=1, squad=[], lineup=[], bench_order=[]),
        ]
        for model in models:
            cls = type(model)
            assert cls.model_validate_json(model.model_dump_json()) == model


# --------------------------------------------------------------------------------------
# Integration with the real MILP (skips while sibling modules are absent)
# --------------------------------------------------------------------------------------


def _small_instance() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Deterministic 40-player, 3-GW synthetic instance (feasible under £100m)."""
    rng = np.random.default_rng(42)
    positions = ["GKP"] * 4 + ["DEF"] * 12 + ["MID"] * 14 + ["FWD"] * 10
    codes = list(range(1001, 1001 + len(positions)))
    teams = [1 + (i % 14) for i in range(len(codes))]
    base_price = {"GKP": 45, "DEF": 45, "MID": 55, "FWD": 55}
    prices = pd.DataFrame(
        {
            "player_code": codes,
            "price": [
                base_price[pos] + int(rng.integers(0, 40)) for pos in positions
            ],
            "position": positions,
            "team_code": teams,
            "web_name": [f"Syn{c}" for c in codes],
        }
    )
    price_of = dict(zip(codes, prices["price"], strict=True))
    rows = []
    for code, pos in zip(codes, positions, strict=True):
        for gw in (1, 2, 3):
            rows.append(
                {
                    "season": 2026,
                    "gw": gw,
                    "player_code": code,
                    "xp": round(price_of[code] / 20 + float(rng.uniform(0, 1.5)), 2),
                    "q0": 0.05,
                }
            )
    return pd.DataFrame(rows), prices


class TestRealMilpIntegration:
    def test_dream_team_real_milp(self) -> None:
        milp = pytest.importorskip("fplai.optimizer.milp")
        xp, prices = _small_instance()
        params = None
        try:  # best-effort short time limit; field name is owned by the milp task
            params = milp.SolveParams()
            for name in ("time_limit_s", "time_limit", "time_limit_seconds", "max_seconds"):
                if hasattr(params, name):
                    try:
                        setattr(params, name, 60.0)
                    except Exception:
                        pass
                    break
        except Exception:
            params = None
        dt = dream_team(xp, prices, 1, 2026, horizon=2, params=params)
        assert len(dt.squad) == 15
        assert len(dt.lineup) == 11
        assert len(dt.bench_order) == 4
        assert dt.captain in dt.lineup
        assert dt.total_cost <= 1000
        pos = dict(zip(prices["player_code"], prices["position"], strict=True))
        from collections import Counter

        quota = Counter(pos[c] for c in dt.squad)
        assert quota == Counter({"GKP": 2, "DEF": 5, "MID": 5, "FWD": 3})
        json.dumps(dt.model_dump(mode="json"))

    def test_build_recommendation_initial_squad_real_milp(self) -> None:
        pytest.importorskip("fplai.optimizer.milp")
        xp, prices = _small_instance()
        rec = build_recommendation(
            None,
            xp,
            prices,
            as_of=AS_OF,
            horizon=2,
            run_chips=False,
            run_stability=False,
        )
        assert rec.action == "initial-squad"
        assert len(rec.squad) == 15
        assert rec.captain is not None
        assert rec.rationale  # bullets generated
        json.dumps(rec.model_dump(mode="json"))

    def test_build_recommendation_full_stack(self) -> None:
        pytest.importorskip("fplai.optimizer.milp")
        pytest.importorskip("fplai.optimizer.chips")
        pytest.importorskip("fplai.optimizer.sensitivity")
        xp, prices = _small_instance()
        rec = build_recommendation(
            None,
            xp,
            prices,
            as_of=AS_OF,
            horizon=2,
            stability_n=3,
            stability_seed=7,
        )
        assert rec.action == "initial-squad"
        assert isinstance(rec.chip_advice, list)
        assert isinstance(rec.stability, list)
        json.dumps(rec.model_dump(mode="json"))
