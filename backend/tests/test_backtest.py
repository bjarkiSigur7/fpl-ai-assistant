"""Tests for the walk-forward backtest harness (fplai.backtest).

Offline tests replay a synthetic 3-GW mini-season (22 players, planted realized
points/minutes/prices) through the REAL MILP solver and assert the state-rolling
arithmetic (bank/FT/hits/sell prices), the §1.5 autosub rules on planted 0-minute
starters, ledger totals, chip-trigger behaviour and the result round-trip. One
``@pytest.mark.live`` test replays real GWs 30-32 of 2025-26 with the on-disk
artifacts (slow; excluded by default).
"""

from __future__ import annotations

import json
import math

import pandas as pd
import pytest

from fplai import rules
from fplai.backtest import (
    BacktestParams,
    BacktestResult,
    apply_autosubs,
    compute_xp_metrics,
    last5_xp_frame,
    replay,
    run_backtest,
    score_gw,
)
from fplai.backtest.harness import _apply_transfers, _normalize_gws
from fplai.optimizer.state import OwnedPlayer

SEASON = 2025

# --------------------------------------------------------------------------------------
# Synthetic universe: 22 players, prices planted, one price riser (code 911)
# --------------------------------------------------------------------------------------

#: (player_code, position, team_code) — quotas satisfiable, club limit trivially met.
PLAYERS: list[tuple[int, str, int]] = [
    (901, "GKP", 1), (902, "GKP", 2), (903, "GKP", 3),
    (911, "DEF", 4), (912, "DEF", 5), (913, "DEF", 6), (914, "DEF", 7),
    (915, "DEF", 8), (916, "DEF", 9), (917, "DEF", 10),
    (921, "MID", 11), (922, "MID", 12), (923, "MID", 13), (924, "MID", 14),
    (925, "MID", 15), (926, "MID", 16), (927, "MID", 17),
    (931, "FWD", 18), (932, "FWD", 19), (933, "FWD", 20), (934, "FWD", 1),
    (935, "FWD", 2),
]
CODES = [c for c, _, _ in PLAYERS]
POSITIONS = {c: p for c, p, _ in PLAYERS}

#: Planted per-GW prices: everyone 50, except 911 rises 50 -> 54 -> 56.
PRICE_PLAN: dict[int, dict[int, int]] = {
    1: {c: 50 for c in CODES},
    2: {c: 50 for c in CODES} | {911: 54},
    3: {c: 50 for c in CODES} | {911: 56},
}

#: Planted realized minutes: everyone 90, except 921 (dominant xP -> always starts)
#: sits out GW2 entirely — the autosub test target.
MINUTES_PLAN: dict[int, dict[int, int]] = {
    1: {c: 90 for c in CODES},
    2: {c: 90 for c in CODES} | {921: 0},
    3: {c: 90 for c in CODES},
}

#: Planted realized points: base 2, stars 921/931 return more (except 921's GW2 blank).
POINTS_PLAN: dict[int, dict[int, int]] = {
    1: {c: 2 for c in CODES} | {921: 12, 931: 8},
    2: {c: 2 for c in CODES} | {921: 0, 931: 6},
    3: {c: 2 for c in CODES} | {921: 9, 931: 7},
}


def make_player_gw(
    gws: list[int],
    *,
    points: dict[int, dict[int, int]] | None = None,
    minutes: dict[int, dict[int, int]] | None = None,
    prices: dict[int, dict[int, int]] | None = None,
) -> pd.DataFrame:
    """Synthetic player_gw rows (season fixed) from the planted per-GW dicts."""
    points = points if points is not None else POINTS_PLAN
    minutes = minutes if minutes is not None else MINUTES_PLAN
    prices = prices if prices is not None else PRICE_PLAN
    rows = [
        {
            "season": SEASON,
            "gw": gw,
            "player_code": code,
            "position": pos,
            "team_code": team,
            "minutes": minutes[gw][code],
            "total_points": points[gw][code],
            "value": prices[gw][code],
            "n_fixtures": 1,
        }
        for gw in gws
        for code, pos, team in PLAYERS
    ]
    return pd.DataFrame(rows)


def make_predictions(xp_by_gw: dict[int, dict[int, float]], q0: float = 0.05) -> pd.DataFrame:
    """Synthetic model-xP frame: season, gw, player_code, xp, q0, n_fixtures."""
    rows = [
        {"season": SEASON, "gw": gw, "player_code": code, "xp": xp, "q0": q0,
         "n_fixtures": 1}
        for gw, per_player in xp_by_gw.items()
        for code, xp in per_player.items()
    ]
    return pd.DataFrame(rows)


def default_xp() -> pd.DataFrame:
    """xP for GWs 1-3: flat 2.0, with 921 (12.0) and 931 (6.0) dominant every GW."""
    per_gw = {c: 2.0 for c in CODES} | {921: 12.0, 931: 6.0}
    return make_predictions({1: dict(per_gw), 2: dict(per_gw), 3: dict(per_gw)})


FAST = BacktestParams(
    horizon=3,
    time_limit_s=10.0,
    mip_rel_gap=0.01,
    evaluate_chips=False,
    run_baselines=True,
)


@pytest.fixture(scope="module")
def mini_result() -> BacktestResult:
    """The 3-GW mini-replay through the real MILP (shared across assertions)."""
    return replay(
        season=SEASON,
        gws=[1, 2, 3],
        predictions=default_xp(),
        player_gw=make_player_gw([1, 2, 3]),
        params=FAST,
    )


# --------------------------------------------------------------------------------------
# Autosubs (FPL_KNOWLEDGE §1.5)
# --------------------------------------------------------------------------------------


class TestAutosubs:
    LINEUP = [901, 911, 912, 913, 914, 921, 922, 923, 924, 931, 932]  # 4-4-2
    BENCH = [902, 915, 925, 933]  # GK, DEF, MID, FWD

    @staticmethod
    def minutes(zeros: set[int]) -> dict[int, int]:
        return {c: 0 if c in zeros else 90 for c in CODES}

    def test_zero_min_starter_replaced_by_first_playing_bench(self) -> None:
        xi, subs = apply_autosubs(
            self.LINEUP, self.BENCH, POSITIONS, self.minutes({921})
        )
        assert subs == [(921, 915)]  # first outfield bench slot covers the absent MID
        assert 921 not in xi and 915 in xi

    def test_gk_replaced_only_by_bench_gk(self) -> None:
        xi, subs = apply_autosubs(self.LINEUP, self.BENCH, POSITIONS, self.minutes({901}))
        assert subs == [(901, 902)]
        assert 901 not in xi and 902 in xi
        # bench GK absent too -> nobody covers the GK slot
        xi2, subs2 = apply_autosubs(
            self.LINEUP, self.BENCH, POSITIONS, self.minutes({901, 902})
        )
        assert subs2 == [] and 901 in xi2

    def test_formation_guard_blocks_illegal_entry(self) -> None:
        # 3-5-2 lineup; the absent starter is a DEF -> a bench FWD may not replace him
        lineup = [901, 911, 912, 913, 921, 922, 923, 924, 925, 931, 932]
        bench = [902, 933, 915, 926]  # outfield priority: FWD, DEF, MID
        xi, subs = apply_autosubs(lineup, bench, POSITIONS, self.minutes({911}))
        # FWD 933 would make DEF count 2 < 3 (illegal) — the bench DEF steps in instead
        assert subs == [(911, 915)]
        assert 933 not in xi

    def test_played_starter_never_replaced(self) -> None:
        minutes = self.minutes(set())  # everyone played (even a 1-pointer stays)
        xi, subs = apply_autosubs(self.LINEUP, self.BENCH, POSITIONS, minutes)
        assert subs == [] and xi == list(self.LINEUP)

    def test_bench_boost_empty_bench_no_subs(self) -> None:
        xi, subs = apply_autosubs(self.LINEUP, [], POSITIONS, self.minutes({921}))
        assert subs == [] and xi == list(self.LINEUP)


class TestScoreGw:
    SQUAD = TestAutosubs.LINEUP + TestAutosubs.BENCH

    def test_captain_doubles_and_vice_fallback(self) -> None:
        points = {c: 2 for c in CODES} | {921: 10, 931: 6}
        minutes = {c: 90 for c in CODES}
        scored = score_gw(
            squad=self.SQUAD, lineup=TestAutosubs.LINEUP, bench_order=TestAutosubs.BENCH,
            captain=921, vice=931, chip=None, points=points, minutes=minutes,
            positions=POSITIONS,
        )
        assert scored.gross_points == (9 * 2 + 10 + 6) + 10  # XI sum + captain extra
        assert scored.effective_captain == 921
        assert scored.captain_points == 20 and scored.captain_success
        # captain blanks -> vice doubled instead; captain autosubbed out
        minutes2 = minutes | {921: 0}
        scored2 = score_gw(
            squad=self.SQUAD, lineup=TestAutosubs.LINEUP, bench_order=TestAutosubs.BENCH,
            captain=921, vice=931, chip=None, points=points | {921: 0},
            minutes=minutes2, positions=POSITIONS,
        )
        assert scored2.effective_captain == 931
        assert scored2.autosubs == [(921, 915)]
        assert scored2.gross_points == (9 * 2 + 6 + 2) + 6  # subbed XI + vice extra

    def test_triple_captain_and_bench_boost(self) -> None:
        points = {c: 2 for c in CODES} | {921: 10}
        minutes = {c: 90 for c in CODES}
        tc = score_gw(
            squad=self.SQUAD, lineup=TestAutosubs.LINEUP, bench_order=TestAutosubs.BENCH,
            captain=921, vice=931, chip="tc1", points=points, minutes=minutes,
            positions=POSITIONS,
        )
        assert tc.gross_points == (10 * 2 + 10) + 20  # XI + 2x extra
        assert tc.captain_points == 30
        bb = score_gw(
            squad=self.SQUAD, lineup=TestAutosubs.LINEUP, bench_order=[],
            captain=921, vice=931, chip="bb1", points=points, minutes=minutes,
            positions=POSITIONS,
        )
        assert bb.gross_points == (14 * 2 + 10) + 10  # all 15 + captain extra
        assert bb.bench_points == 0


# --------------------------------------------------------------------------------------
# Transfer / sell-price arithmetic (FPL_KNOWLEDGE §1.8)
# --------------------------------------------------------------------------------------


def test_apply_transfers_sell_price_and_bank() -> None:
    squad = [
        OwnedPlayer(player_code=911, purchase_price=50, current_price=56),
        OwnedPlayer(player_code=912, purchase_price=50, current_price=46),
    ]
    values = {911: 56, 912: 46, 921: 60}
    new_squad, bank = _apply_transfers(squad, 10, [921], [911, 912], values.get)
    # 911: bought 50 now 56 -> sell 53 (50% fee); 912: full fall borne -> 46
    assert bank == 10 + rules.sell_price(50, 56) + 46 - 60 == 10 + 53 + 46 - 60
    codes = [p.player_code for p in new_squad]
    assert codes == [921] and new_squad[0].purchase_price == 60


def test_apply_transfers_rejects_unknown_sale() -> None:
    with pytest.raises(RuntimeError, match="not owned"):
        _apply_transfers([], 0, [], [999], lambda _c: 50)


# --------------------------------------------------------------------------------------
# Last-5 baseline frame
# --------------------------------------------------------------------------------------


def test_last5_frame_mean_and_no_same_gw_leak() -> None:
    pg = make_player_gw([1, 2, 3])
    frame = last5_xp_frame(pg, SEASON, [1, 2, 3])
    row = frame.set_index(["gw", "player_code"])
    # GW1: no history -> xp 0, q0 1
    assert row.loc[(1, 921), "xp"] == 0.0 and row.loc[(1, 921), "q0"] == 1.0
    # GW2: exactly GW1's points (12 for 921) — GW2's own 0 not leaked
    assert row.loc[(2, 921), "xp"] == 12.0
    assert row.loc[(2, 921), "q0"] == 0.0
    # GW3: mean of GW1+GW2 (12 + 0) / 2; q0 = one blank of two
    assert row.loc[(3, 921), "xp"] == 6.0
    assert row.loc[(3, 921), "q0"] == 0.5


# --------------------------------------------------------------------------------------
# The 3-GW mini-replay: state rolling, autosubs, ledger totals, baselines, metrics
# --------------------------------------------------------------------------------------


class TestMiniReplay:
    def test_ledger_shape_and_labels(self, mini_result: BacktestResult) -> None:
        assert mini_result.season == SEASON and mini_result.gws == [1, 2, 3]
        assert set(mini_result.policies) == {"model", "last5", "set_and_forget"}
        for ledger in mini_result.policies.values():
            assert [r.gw for r in ledger.rows] == [1, 2, 3]
        assert mini_result.references["average_manager"] == 2250
        assert mini_result.references["top_10k"] == 2625

    def test_initial_build_bank_and_unlimited_transfers(
        self, mini_result: BacktestResult
    ) -> None:
        row1 = mini_result.policies["model"].rows[0]
        assert row1.free_transfers is None  # unlimited initial build
        assert row1.hit_points == 0 and row1.n_transfers == 0
        assert len(row1.squad) == 15 and len(set(row1.squad)) == 15
        cost = sum(PRICE_PLAN[1][c] for c in row1.squad)
        assert row1.bank == 1000 - cost
        assert row1.squad_value == cost
        assert row1.team_value == 1000  # purchase == current at build -> sell == price

    def test_free_transfer_rolling(self, mini_result: BacktestResult) -> None:
        rows = mini_result.policies["model"].rows
        assert rows[1].free_transfers == 1  # 1 FT after the unlimited build week
        expected = rules.next_free_transfers(
            rows[1].free_transfers, rows[1].n_transfers, SEASON, False
        )
        assert rows[2].free_transfers == expected

    def test_bank_and_team_value_arithmetic(self, mini_result: BacktestResult) -> None:
        rows = mini_result.policies["model"].rows
        purchase = {c: PRICE_PLAN[1][c] for c in rows[0].squad}
        bank = rows[0].bank
        for row in rows[1:]:
            values = PRICE_PLAN[row.gw]
            for out in row.transfers_out:
                bank += rules.sell_price(purchase.pop(out), values[out])
            for code in row.transfers_in:
                bank -= values[code]
                purchase[code] = values[code]
            assert row.bank == bank
            assert sorted(row.squad) == sorted(purchase)
            assert row.squad_value == sum(values[c] for c in row.squad)
            assert row.team_value == bank + sum(
                rules.sell_price(purchase[c], values[c]) for c in row.squad
            )

    def test_points_recompute_from_planted_data(self, mini_result: BacktestResult) -> None:
        for ledger in mini_result.policies.values():
            for row in ledger.rows:
                points = POINTS_PLAN[row.gw]
                base = sum(points[c] for c in row.lineup)
                extra = (
                    points[row.effective_captain]
                    if row.effective_captain is not None
                    else 0
                )
                assert row.gross_points == base + extra
                assert row.points == row.gross_points - row.hit_points
            assert ledger.total_points == sum(r.points for r in ledger.rows)

    def test_autosub_applied_to_planted_zero_min_starter(
        self, mini_result: BacktestResult
    ) -> None:
        # 921 has dominant xP (always fielded) but 0 planted minutes in GW2
        row2 = mini_result.policies["model"].rows[1]
        assert 921 in [out for out, _in in row2.autosubs]
        assert 921 not in row2.lineup
        sub_in = dict(row2.autosubs)[921]
        assert sub_in in row2.lineup

    def test_captain_fields_consistent(self, mini_result: BacktestResult) -> None:
        for row in mini_result.policies["model"].rows:
            points = POINTS_PLAN[row.gw]
            minutes = MINUTES_PLAN[row.gw]
            if minutes[row.captain] > 0:
                assert row.effective_captain == row.captain
            assert row.captain_points == (
                2 * points[row.effective_captain]
                if row.effective_captain is not None
                else 0
            )
            top = max(points[c] for c in row.lineup)
            assert row.captain_success == (
                row.effective_captain is not None
                and points[row.effective_captain] >= top
            )

    def test_set_and_forget_never_moves(self, mini_result: BacktestResult) -> None:
        ledger = mini_result.policies["set_and_forget"]
        squads = {tuple(sorted(r.squad)) for r in ledger.rows}
        assert len(squads) == 1
        for row in ledger.rows:
            assert row.n_transfers == 0 and row.hit_points == 0 and row.chip is None
            assert row.bank == ledger.rows[0].bank
        # autocaptain = highest-xP XI member = 921 (xp 12) whenever fielded
        assert ledger.rows[0].captain == 921

    def test_metrics_shapes(self, mini_result: BacktestResult) -> None:
        metrics = mini_result.metrics
        assert set(metrics["by_category"]) == {"zeros", "blanks", "tickers", "haulers"}
        assert metrics["overall"]["n"] == 66  # 22 players x 3 GWs
        assert metrics["overall"]["rmse"] >= 0.0
        assert metrics["by_position"] == {}  # synthetic predictions carry no position
        # spearman defined (planted variation exists)
        assert metrics["spearman_within_gw"] is None or -1 <= metrics["spearman_within_gw"] <= 1


def test_metrics_by_category_planted_values() -> None:
    preds = make_predictions({1: {901: 0.0, 902: 2.0, 903: 4.0, 911: 8.0}})
    pg = make_player_gw([1])
    # realized: 901..903 = 2 pts (blanks), 911 = 2 -> plant custom values instead
    pg.loc[pg.player_code == 901, "total_points"] = 0
    pg.loc[pg.player_code == 902, "total_points"] = 2
    pg.loc[pg.player_code == 903, "total_points"] = 4
    pg.loc[pg.player_code == 911, "total_points"] = 8
    m = compute_xp_metrics(preds, pg, [1])
    assert m["by_category"]["zeros"] == {"rmse": 0.0, "mae": 0.0, "n": 1}
    assert m["by_category"]["blanks"]["n"] == 1 and m["by_category"]["blanks"]["rmse"] == 0.0
    assert m["by_category"]["tickers"]["n"] == 1
    assert m["by_category"]["haulers"]["n"] == 1
    assert m["overall"]["n"] == 4 and math.isclose(m["overall"]["rmse"], 0.0)


# --------------------------------------------------------------------------------------
# Chip policy: BB trigger fires and the forced-now solve is adopted
# --------------------------------------------------------------------------------------


def test_bb_chip_trigger_fires_and_scores_bench() -> None:
    # GW1: flat 1.0 xP (bench EV 4 < threshold -> no chip solve);
    # GW2: 6.0 xP for everyone (bench EV 24 >= 10 -> BB fires; delta >> margin).
    xp = make_predictions(
        {1: {c: 1.0 for c in CODES}, 2: {c: 6.0 for c in CODES}}
    )
    points = {1: {c: 2 for c in CODES}, 2: {c: 2 for c in CODES}}
    minutes = {1: {c: 90 for c in CODES}, 2: {c: 90 for c in CODES}}
    prices = {1: {c: 50 for c in CODES}, 2: {c: 50 for c in CODES}}
    result = replay(
        season=SEASON,
        gws=[1, 2],
        predictions=xp,
        player_gw=make_player_gw([1, 2], points=points, minutes=minutes, prices=prices),
        params=BacktestParams(
            horizon=2,
            time_limit_s=10.0,
            evaluate_chips=True,
            tc_captain_xp_threshold=99.0,  # keep TC quiet — this test isolates BB
            run_baselines=False,
        ),
    )
    rows = result.policies["model"].rows
    assert rows[0].chip is None
    assert rows[1].chip == "bb1"
    # BB week: all 15 score (2 each) + captain extra
    assert rows[1].gross_points == 15 * 2 + 2
    assert rows[1].bench_points == 0
    assert result.policies["model"].chips_played == ["bb1"]


# --------------------------------------------------------------------------------------
# Result round-trip + validation
# --------------------------------------------------------------------------------------


def test_to_frame_and_save_roundtrip(mini_result: BacktestResult, tmp_path) -> None:
    frame = mini_result.to_frame()
    assert len(frame) == 3 * 3  # 3 policies x 3 GWs
    assert {"policy", "gw", "points", "bank", "team_value", "chip"} <= set(frame.columns)
    paths = mini_result.save(tmp_path)
    assert paths["result"].exists() and paths["ledger"].exists()
    payload = json.loads(paths["result"].read_text())
    assert payload["season"] == SEASON
    assert payload["policies"]["model"]["total_points"] == (
        mini_result.policies["model"].total_points
    )
    assert len(pd.read_parquet(paths["ledger"])) == len(frame)


def test_normalize_gws_rejects_gaps() -> None:
    with pytest.raises(ValueError, match="contiguous"):
        _normalize_gws([1, 3], [1, 2, 3])
    with pytest.raises(ValueError, match="no player_gw rows"):
        _normalize_gws([4], [1, 2, 3])
    assert _normalize_gws(None, [5, 6, 7]) == [5, 6, 7]
    assert _normalize_gws(range(6, 8), [5, 6, 7]) == [6, 7]


# --------------------------------------------------------------------------------------
# Real-data smoke test (slow; needs data/processed + data/models on disk)
# --------------------------------------------------------------------------------------


@pytest.mark.live
def test_real_backtest_gws_30_32() -> None:
    """3-GW real replay of 2025-26 with the existing artifacts: sane, positive totals."""
    result = run_backtest(
        2025,
        gws=range(30, 33),
        policy_params=BacktestParams(horizon=3, time_limit_s=10.0),
    )
    assert result.mode == "coarse-pretrained"
    assert result.gws == [30, 31, 32]
    for name, ledger in result.policies.items():
        assert len(ledger.rows) == 3, name
        assert ledger.total_points > 0, name
        for row in ledger.rows:
            is_bb = row.chip is not None and row.chip.startswith("bb")
            assert len(row.lineup) == (15 if is_bb else 11)
            assert row.bank >= 0
    # existing artifacts were trained on all seasons incl. 2025 -> loud leakage warning
    assert any("LEAKY" in w or "train" in w for w in result.leakage_warnings)
    assert result.metrics["overall"]["n"] > 1000
