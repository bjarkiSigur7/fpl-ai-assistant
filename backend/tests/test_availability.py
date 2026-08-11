"""Offline tests for availability v2 — dated news gates (fplai.data.live).

The trimmed live snapshot fixture carries the real day-1 news grammar:
J.Timber "Expected back 21 Aug" (i), Fofana "Suspended until 6 Sep" (s),
Andersen "Suspended until 29 Aug" (s), Saliba "Unknown return date" (i).
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pandas as pd
import pytest

from fplai.data import live

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "live_snapshot_2026"

TIMBER_CODE = 445122  # ARS, "Groin injury - Expected back 21 Aug"
FOFANA_CODE = 444463  # CHE, "Suspended until 6 Sep"
SALIBA_CODE = 462424  # ARS, "Back injury - Unknown return date"


@pytest.fixture(scope="module")
def ctx() -> live.LiveContext:
    loaded = live.load_live_context(snap_dir=FIXTURE_DIR)
    assert loaded is not None
    return loaded


# ---------------------------------------------------------------------------
# parse_news_return
# ---------------------------------------------------------------------------


def test_parse_expected_back() -> None:
    parsed = live.parse_news_return(
        "Ankle injury - Expected back 23 Aug", "2026-08-06T16:00:10.997714Z"
    )
    assert parsed == (dt.date(2026, 8, 23), "injury")


def test_parse_suspended_until() -> None:
    parsed = live.parse_news_return("Suspended until 6 Sep", "2026-07-23T12:01:23Z")
    assert parsed == (dt.date(2026, 9, 6), "suspension")


def test_parse_year_wraps_over_new_year() -> None:
    # "Expected back 15 Jan" announced in December -> January of the NEXT year.
    parsed = live.parse_news_return(
        "Hamstring injury - Expected back 15 Jan", "2026-12-20T09:00:00Z"
    )
    assert parsed == (dt.date(2027, 1, 15), "injury")


def test_parse_recent_past_date_stays_in_year() -> None:
    # A stale return date a few weeks back must NOT roll a year forward.
    parsed = live.parse_news_return(
        "Knock - Expected back 10 Jul", "2026-08-01T09:00:00Z"
    )
    assert parsed == (dt.date(2026, 7, 10), "injury")


@pytest.mark.parametrize(
    "news",
    [
        "",
        "Knee injury - Unknown return date",
        "Knee injury - 75% chance of playing",
        "Has joined Paris Saint-Germain permanently",
    ],
)
def test_parse_undated_news_is_none(news: str) -> None:
    assert live.parse_news_return(news, "2026-08-01T09:00:00Z") is None


def test_parse_fallback_date_used_when_added_missing() -> None:
    parsed = live.parse_news_return(
        "Groin injury - Expected back 21 Aug", None, fallback=dt.date(2026, 7, 23)
    )
    assert parsed == (dt.date(2026, 8, 21), "injury")


# ---------------------------------------------------------------------------
# availability_overrides (fixture snapshot: GW1 kickoffs 2026-08-21/22)
# ---------------------------------------------------------------------------


def test_overrides_suspension_hard_zero_before_return(ctx: live.LiveContext) -> None:
    overrides, report = live.availability_overrides(ctx)
    # Fofana suspended until 6 Sep: GW1 (21 Aug) is a hard 0.0 — a ban is not a
    # fitness state that "recovers".
    assert overrides[(FOFANA_CODE, 1)] == 0.0
    assert report[FOFANA_CODE]["kind"] == "suspension"
    assert report[FOFANA_CODE]["return_date"] == "2026-09-06"


def test_overrides_injury_return_gw_ramp(ctx: live.LiveContext) -> None:
    overrides, report = live.availability_overrides(ctx)
    # Timber expected back 21 Aug; ARS kick off GW1 on the 21st -> that IS the
    # return GW: comeback factor, not a hard out.
    assert overrides[(TIMBER_CODE, 1)] == pytest.approx(0.65)
    assert report[TIMBER_CODE]["return_gw"] == 1


def test_overrides_skip_undated_news(ctx: live.LiveContext) -> None:
    overrides, report = live.availability_overrides(ctx)
    assert SALIBA_CODE not in report
    assert not any(code == SALIBA_CODE for code, _ in overrides)


def test_factors_apply_overrides_end_to_end(ctx: live.LiveContext, tmp_path: Path) -> None:
    pm = pd.DataFrame(
        {
            "season": [2020],
            "gw": [1],
            "player_code": [1],
            "minutes": [90],
            "position": ["MID"],
            "price": [50],
            "goals_scored": [0],
            "assists": [0],
            "saves": [0],
            "tackles": [0],
            "recoveries": [0],
            "clearances_blocks_interceptions": [0],
        }
    )
    adj = live.build_adjustments(ctx, pm, processed_dir=tmp_path)
    features = pd.DataFrame(
        {
            "season": [2026, 2026, 2026],
            "gw": [1, 1, 3],
            "player_code": [FOFANA_CODE, SALIBA_CODE, FOFANA_CODE],
            "fpl_fixture_id": [5, 1, 30],
            "f_status_s": [1.0, 0.0, 1.0],
            "f_status_i": [0.0, 1.0, 0.0],
            "f_status_u": [0.0, 0.0, 0.0],
            "f_status_d": [0.0, 0.0, 0.0],
            "f_chance_of_playing": [0.0, float("nan"), 0.0],
        }
    )
    factors = live._availability_factors(features, adj)
    assert factors[0] == 0.0  # suspended, dated: hard zero (was 0.05 + recovery)
    assert factors[1] == pytest.approx(0.05)  # undated injury, no chance %: heuristic floor
    # GW3 is outside the fixture snapshot's calendar (GW1 + partial GW2 only),
    # so no override exists and the linear heuristic applies.
    assert 0.0 < factors[2] <= 1.0

    payload = json.loads((tmp_path / "availability_2026.json").read_text())
    assert payload["returns"][str(FOFANA_CODE)]["kind"] == "suspension"
