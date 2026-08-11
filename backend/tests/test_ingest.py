"""Offline tests for played-GW outcome ingestion (fplai.data.ingest).

Builds a live snapshot in tmp_path from the trimmed real day-1 fixture
(``tests/fixtures/live_snapshot_2026``) with GW1 fixtures flipped to finished,
plus synthesized element-summary payloads, and verifies the canonical
player_match/player_gw splice, freeze bookkeeping and idempotency.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from fplai.data import ingest, live

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "live_snapshot_2026"

# From the trimmed snapshot: fixture 1 is ARS (fpl_team_id 1) v COV (7) in GW1.
RAYA = {"element_id": 1, "code": 154561, "team_id": 1}
SAKA = {"element_id": 12, "code": 223340, "team_id": 1}
WRIGHT = {"element_id": 193, "code": 176412, "team_id": 7}


def _snapshot(
    tmp_path: Path,
    *,
    finish_gw1: bool = True,
    data_checked: bool = False,
    only_fixture: int | None = None,
) -> Path:
    """Write a live snapshot dir with GW1 (or one GW1 fixture) finished."""
    snap = tmp_path / "snap"
    snap.mkdir(parents=True, exist_ok=True)
    bootstrap = json.loads((FIXTURE_DIR / "bootstrap.json").read_text())
    fixtures = json.loads((FIXTURE_DIR / "fixtures.json").read_text())
    if finish_gw1:
        bootstrap["events"][0]["finished"] = only_fixture is None
        bootstrap["events"][0]["data_checked"] = data_checked
        for f in fixtures:
            if f.get("event") == 1 and (only_fixture is None or f.get("id") == only_fixture):
                f["finished"] = True
    (snap / "bootstrap.json").write_text(json.dumps(bootstrap))
    (snap / "fixtures.json").write_text(json.dumps(fixtures))
    return snap


def _ctx(tmp_path: Path, **kwargs: Any) -> live.LiveContext:
    loaded = live.load_live_context(snap_dir=_snapshot(tmp_path, **kwargs))
    assert loaded is not None
    return loaded


def _history_row(
    *,
    fixture: int = 1,
    opponent_team: int = 7,
    round_: int = 1,
    was_home: bool = True,
    minutes: int = 90,
    total_points: int = 6,
    value: int = 60,
    **extra: Any,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "fixture": fixture,
        "opponent_team": opponent_team,
        "round": round_,
        "was_home": was_home,
        "minutes": minutes,
        "total_points": total_points,
        "value": value,
        "goals_scored": 0,
        "assists": 0,
        "clean_sheets": 1,
        "goals_conceded": 0,
        "saves": 3,
        "penalties_saved": 0,
        "penalties_missed": 0,
        "yellow_cards": 0,
        "red_cards": 0,
        "own_goals": 0,
        "bonus": 1,
        "bps": 29,
        "starts": 1,
        "defensive_contribution": 0,
        "tackles": 1,
        "recoveries": 6,
        "clearances_blocks_interceptions": 2,
        "expected_goals": "0.05",
        "expected_assists": "0.02",
        "expected_goals_conceded": "0.61",
        "transfers_in": 1000,
        "transfers_out": 50,
    }
    row.update(extra)
    return row


def _base_processed(tmp_path: Path, ctx: live.LiveContext) -> Path:
    """Minimal processed dir with one historical (2025) row in each table."""
    processed = tmp_path / "processed"
    processed.mkdir(exist_ok=True)
    pm_new, gw_new = ingest.build_played_tables(
        ctx, {RAYA["element_id"]: {"history": [_history_row()]}}, [1]
    )
    old_pm = pm_new.assign(season=2025, gw=38)
    old_gw = gw_new.assign(season=2025, gw=38)
    old_pm.to_parquet(processed / "player_match.parquet", index=False)
    old_gw.to_parquet(processed / "player_gw.parquet", index=False)
    return processed


class FakeClient:
    """element_summary_sweep stand-in writing payload JSONs like the real one."""

    def __init__(self, raw_dir: Path, payloads: dict[int, dict[str, Any]]) -> None:
        self.raw_dir = raw_dir
        self.payloads = payloads
        self.swept: list[list[int]] = []

    def element_summary_sweep(self, ids: Any, on_progress: Any = None) -> list[Path]:
        id_list = [int(i) for i in ids]
        self.swept.append(id_list)
        out_dir = self.raw_dir / "element_summary" / "2026"
        out_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        for element_id in id_list:
            path = out_dir / f"{element_id}.json"
            path.write_text(json.dumps(self.payloads.get(element_id, {"history": []})))
            paths.append(path)
        return paths


# ---------------------------------------------------------------------------
# GW selection
# ---------------------------------------------------------------------------


def test_ingestable_gws_preseason_is_empty(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path, finish_gw1=False)
    todo, checked = ingest.ingestable_gws(ctx, {})
    assert todo == []
    assert checked == set()


def test_ingestable_gws_finished_and_frozen(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path, data_checked=True)
    todo, checked = ingest.ingestable_gws(ctx, {})
    assert todo == [1]
    assert 1 in checked
    # Frozen GWs are excluded.
    todo2, _ = ingest.ingestable_gws(ctx, {"2026": {"frozen_gws": [1]}})
    assert todo2 == []


# ---------------------------------------------------------------------------
# Row building
# ---------------------------------------------------------------------------


def test_build_played_tables_canonical_row(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    payloads = {
        RAYA["element_id"]: {"history": [_history_row()]},
        # A history row outside the target GWs must be dropped.
        SAKA["element_id"]: {"history": [_history_row(round_=2, fixture=11)]},
    }
    pm, gw = ingest.build_played_tables(ctx, payloads, [1])
    assert len(pm) == 1 and len(gw) == 1
    row = pm.iloc[0]
    assert row["season"] == 2026 and row["gw"] == 1 and row["fpl_fixture_id"] == 1
    assert row["player_code"] == RAYA["code"]
    assert row["position"] == "GKP" and row["was_home"]
    # opponent_team 7 (COV) maps to the stable team_code, not the per-season id.
    cov_code = int(
        ctx.teams.loc[ctx.teams["fpl_team_id"] == 7, "team_code"].iloc[0]
    )
    assert row["opponent_code"] == cov_code
    assert row["price"] == 60 and row["saves"] == 3 and row["bps"] == 29
    assert row["xg"] == pytest.approx(0.05) and row["xgc"] == pytest.approx(0.61)
    assert pd.isna(row["us_xg"])  # Understat arrives with the season backfill
    g = gw.iloc[0]
    assert g["n_fixtures"] == 1 and g["value"] == 60
    assert g["transfers_in_event"] == 1000 and pd.isna(g["selected_by_percent"])


def test_build_played_tables_dgw_sums_per_gw(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    payloads = {
        RAYA["element_id"]: {
            "history": [
                _history_row(total_points=6, minutes=90),
                _history_row(fixture=2, opponent_team=15, was_home=False,
                             total_points=2, minutes=45),
            ]
        }
    }
    pm, gw = ingest.build_played_tables(ctx, payloads, [1])
    assert len(pm) == 2
    assert len(gw) == 1
    g = gw.iloc[0]
    assert g["n_fixtures"] == 2
    assert g["minutes"] == 135 and g["total_points"] == 8
    # GW-level transfer counts must not double on DGWs.
    assert g["transfers_in_event"] == 1000


def test_build_played_tables_unknown_element_skipped(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    pm, _ = ingest.build_played_tables(ctx, {99999: {"history": [_history_row()]}}, [1])
    assert len(pm) == 0


# ---------------------------------------------------------------------------
# End-to-end splice + freeze + idempotency
# ---------------------------------------------------------------------------


def test_ingest_played_splices_and_freezes(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path, data_checked=True)
    processed = _base_processed(tmp_path, ctx)
    payloads = {
        RAYA["element_id"]: {"history": [_history_row()]},
        WRIGHT["element_id"]: {
            "history": [_history_row(was_home=False, opponent_team=1, minutes=13,
                                     total_points=1, value=55, clean_sheets=0, saves=0)]
        },
    }
    client = FakeClient(tmp_path / "raw", payloads)
    report = ingest.ingest_played(client, ctx, processed_dir=processed)
    assert report is not None
    assert report.gws == [1] and report.frozen_gws == [1]
    assert report.n_rows == 2 and report.n_players == 2

    pm = pd.read_parquet(processed / "player_match.parquet")
    assert len(pm[pm["season"] == 2025]) == 1  # history untouched
    assert len(pm[(pm["season"] == 2026) & (pm["gw"] == 1)]) == 2
    state = ingest.load_state(processed)
    assert state["2026"]["frozen_gws"] == [1]

    # Second run: GW1 frozen -> no-op, zero requests.
    report2 = ingest.ingest_played(client, ctx, processed_dir=processed)
    assert report2 is None
    assert len(client.swept) == 1


def test_ingest_sweeps_only_teams_with_finished_fixtures(tmp_path: Path) -> None:
    # Mid-weekend partial state: only fixture 1 (ARS v COV) has finished.
    ctx = _ctx(tmp_path, only_fixture=1)
    processed = _base_processed(tmp_path, ctx)
    payloads = {RAYA["element_id"]: {"history": [_history_row()]}}
    client = FakeClient(tmp_path / "raw", payloads)
    report = ingest.ingest_played(client, ctx, processed_dir=processed)
    assert report is not None and report.frozen_gws == []
    swept = set(client.swept[0])
    assert RAYA["element_id"] in swept and WRIGHT["element_id"] in swept
    assert 411 not in swept  # Haaland (MCI): no finished MCI fixture yet


def test_ingest_played_reruns_replace_until_frozen(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path, data_checked=False)  # played but bonus not settled
    processed = _base_processed(tmp_path, ctx)
    payloads = {RAYA["element_id"]: {"history": [_history_row(bonus=0, total_points=5)]}}
    client = FakeClient(tmp_path / "raw", payloads)
    report = ingest.ingest_played(client, ctx, processed_dir=processed)
    assert report is not None and report.frozen_gws == []

    # Bonus lands; the re-run must REPLACE the GW1 rows, not duplicate them.
    payloads[RAYA["element_id"]] = {"history": [_history_row(bonus=1, total_points=6)]}
    report2 = ingest.ingest_played(client, ctx, processed_dir=processed)
    assert report2 is not None
    pm = pd.read_parquet(processed / "player_match.parquet")
    rows = pm[(pm["season"] == 2026) & (pm["gw"] == 1)]
    assert len(rows) == 1
    assert int(rows.iloc[0]["bonus"]) == 1


def test_ingest_played_preseason_noop(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path, finish_gw1=False)
    client = FakeClient(tmp_path / "raw", {})
    assert ingest.ingest_played(client, ctx, processed_dir=tmp_path) is None
    assert client.swept == []
