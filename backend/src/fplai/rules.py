"""Season constants for FPL: scoring tables, feature flags, chip windows, BPS, prices.

Machine encoding of ``docs/FPL_KNOWLEDGE.md`` Parts 1 and 2 (the authoritative rules
document). Everything here is pure data + pure functions — no I/O, no network, no state.

Conventions
-----------
* ``season`` is always the start year as an int (2016 … 2026); use :func:`season_label`
  for the "2016-17" display form.
* All prices are in £0.1m units (API units), e.g. £5.5m == 55.
* Positions are the FPL short codes ``GKP`` / ``DEF`` / ``MID`` / ``FWD``.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from datetime import UTC, datetime

# --------------------------------------------------------------------------------------
# Seasons
# --------------------------------------------------------------------------------------

#: All seasons covered by this knowledge base (start years 2016-17 .. 2026-27).
SEASONS: range = range(2016, 2027)

#: Canonical position codes, in element_type order.
POSITIONS: tuple[str, ...] = ("GKP", "DEF", "MID", "FWD")


def season_label(season: int) -> str:
    """Return the display label for a season start year, e.g. ``2016 -> "2016-17"``."""
    return f"{season}-{(season + 1) % 100:02d}"


def _validate_season(season: int) -> None:
    if season not in SEASONS:
        raise ValueError(f"season must be in {SEASONS.start}..{SEASONS.stop - 1}, got {season!r}")


# --------------------------------------------------------------------------------------
# Scoring tables (FPL_KNOWLEDGE §1.1 + Part 2 changelog)
# --------------------------------------------------------------------------------------

#: One save point per this many shot saves (points = floor(saves / 3)).
SAVES_PER_POINT: int = 3

#: One conceded-goal deduction per this many goals conceded (points = -floor(conceded / 2)).
CONCEDED_PER_POINT: int = 2

#: Defensive-contribution thresholds (2025-26+). DEF: CBIT >= 10; MID/FWD: CBIRT >= 12.
#: GKPs are ineligible (hence absent). Award is +2, capped at once per player per match.
DEFCON_THRESHOLDS: dict[str, int] = {"DEF": 10, "MID": 12, "FWD": 12}

# Per-unit values, matching API `game_config.scoring` semantics: divisors (per 3 saves,
# per 2 conceded) and DefCon thresholds are NOT baked in here — they live in the
# constants above. `bonus` is per bonus point awarded (1-3 per match for top-3 BPS).
_SCORING_BASE: dict[str, dict[str, int]] = {
    "short_play": {"GKP": 1, "DEF": 1, "MID": 1, "FWD": 1},  # played 1-59 min
    "long_play": {"GKP": 2, "DEF": 2, "MID": 2, "FWD": 2},  # played >= 60 min
    "goals_scored": {"GKP": 6, "DEF": 6, "MID": 5, "FWD": 4},  # GKP -> 10 from 2024-25
    "assists": {"GKP": 3, "DEF": 3, "MID": 3, "FWD": 3},
    "clean_sheets": {"GKP": 4, "DEF": 4, "MID": 1, "FWD": 0},  # requires >= 60 min on pitch
    "goals_conceded": {"GKP": -1, "DEF": -1, "MID": 0, "FWD": 0},  # per CONCEDED_PER_POINT
    "saves": {"GKP": 1, "DEF": 0, "MID": 0, "FWD": 0},  # per SAVES_PER_POINT
    "penalties_saved": {"GKP": 5, "DEF": 0, "MID": 0, "FWD": 0},
    "penalties_missed": {"GKP": -2, "DEF": -2, "MID": -2, "FWD": -2},
    "yellow_cards": {"GKP": -1, "DEF": -1, "MID": -1, "FWD": -1},
    "red_cards": {"GKP": -3, "DEF": -3, "MID": -3, "FWD": -3},  # includes any YC deduction
    "own_goals": {"GKP": -2, "DEF": -2, "MID": -2, "FWD": -2},
    "bonus": {"GKP": 1, "DEF": 1, "MID": 1, "FWD": 1},  # per bonus point (1-3 per match)
}

# +2 once per match when DEFCON_THRESHOLDS met; GKPs ineligible. 2025-26 onwards only.
_DEFCON_SCORING: dict[str, int] = {"GKP": 0, "DEF": 2, "MID": 2, "FWD": 2}


def get_scoring(season: int) -> dict[str, dict[str, int]]:
    """Return the full per-position scoring table for ``season``: action -> position -> points.

    Values are per-unit (API ``game_config.scoring`` semantics); apply
    :data:`SAVES_PER_POINT`, :data:`CONCEDED_PER_POINT` and :data:`DEFCON_THRESHOLDS`
    for divisor/threshold actions. The changelog applied (FPL_KNOWLEDGE Part 2):

    * GKP goal: 6 points through 2023-24, 10 from 2024-25.
    * ``defensive_contribution`` exists only from 2025-26.

    Returns a fresh deep copy — callers may mutate it freely.
    """
    _validate_season(season)
    table = copy.deepcopy(_SCORING_BASE)
    if season >= 2024:
        table["goals_scored"]["GKP"] = 10
    if season >= 2025:
        table["defensive_contribution"] = dict(_DEFCON_SCORING)
    return table


# --------------------------------------------------------------------------------------
# Season feature flags (FPL_KNOWLEDGE §2.2)
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SeasonFlags:
    """Per-season rule/data-regime flags for the training pipeline (FPL_KNOWLEDGE §2.2)."""

    season: int
    assist_definition: str  # "v1" (<=2016) | "v2" (2017-2024) | "v3" (2025+)
    bps_version: str  # "v1" (<=2023) | "v2" (2024) | "v3" (2025) | "v4" (2026+)
    gk_goal_points: int  # 6 (<=2023) | 10 (2024+)
    defcon: bool  # defensive_contribution points exist (2025+)
    max_free_transfers: int  # 2 (<=2023) | 5 (2024+)
    chips_wipe_banked_fts: bool  # True (<=2023) | False (2024+)
    deadline_minutes: int  # minutes before first kickoff: 60 (<=2019) | 90 (2020+)
    chip_inventory: tuple[str, ...]  # chips available that season, one entry per instance
    amnesty: str | None  # free-transfer amnesty to exclude from transfer-cost modeling
    void_events: tuple[int, ...]  # canonical GWs voided (2019: 30-38 post-remap; 2022: (7,))
    empty_stadiums: str | None  # None | "events_39_47" (2019 restart) | "all" (2020-21)
    subs_regime: int  # 3 (<=2021) | 5 (2022+) substitutes allowed in PL matches
    xg_in_api: bool  # FPL API expected_* fields present (2022+); Understat needed before
    manager_elements_in_api: bool  # 2024-25 only: element_type 5 managers pollute pools
    detailed_defensive_stats: str  # "full" (2016-18) | "none" (2019-24) | "cbit_recoveries" (2025+)
    lockdown: str  # "legacy" | "0900_next_day" (2026+: GW final at 09:00 UK next day)


_CHIP_INVENTORY: dict[int, tuple[str, ...]] = {
    2016: ("WC", "WC", "BB", "TC", "AOA"),  # last season of All Out Attack
    2021: ("WC", "WC", "BB", "TC", "FH", "FH_extra_from_GW20"),  # Omicron extra FH
    2024: ("WC", "WC", "BB", "TC", "FH", "AM_from_GW24"),  # Assistant Manager one-off
    2025: ("WC", "WC", "FH", "FH", "BB", "BB", "TC", "TC"),  # two sets; set 1 -> GW19
    2026: ("WC", "WC", "FH", "FH", "BB", "BB", "TC", "TC"),
}
_CHIP_INVENTORY_DEFAULT: tuple[str, ...] = ("WC", "WC", "BB", "TC", "FH")  # 2017-2023 ex-2021

_AMNESTIES: dict[int, str] = {
    2019: "unlimited_before_event_39",  # COVID restart
    2022: "unlimited_between_GW16_and_GW17",  # Qatar World Cup break
    2025: "topped_to_5_at_GW16",  # AFCON top-up (one-off; none in 2026)
}


def _build_flags(season: int) -> SeasonFlags:
    if season <= 2016:
        assist = "v1"
    elif season <= 2024:
        assist = "v2"
    else:
        assist = "v3"
    if season <= 2023:
        bps = "v1"
    elif season == 2024:
        bps = "v2"
    elif season == 2025:
        bps = "v3"
    else:
        bps = "v4"
    if season == 2019:
        void: tuple[int, ...] = tuple(range(30, 39))  # canonical numbering post-remap
    elif season == 2022:
        void = (7,)  # Queen Elizabeth II — rolled over, 0 pts for all
    else:
        void = ()
    empty = {2019: "events_39_47", 2020: "all"}.get(season)
    if season <= 2018:
        defensive = "full"
    elif season <= 2024:
        defensive = "none"
    else:
        defensive = "cbit_recoveries"
    return SeasonFlags(
        season=season,
        assist_definition=assist,
        bps_version=bps,
        gk_goal_points=6 if season <= 2023 else 10,
        defcon=season >= 2025,
        max_free_transfers=2 if season <= 2023 else 5,
        chips_wipe_banked_fts=season <= 2023,
        deadline_minutes=60 if season <= 2019 else 90,
        chip_inventory=_CHIP_INVENTORY.get(season, _CHIP_INVENTORY_DEFAULT),
        amnesty=_AMNESTIES.get(season),
        void_events=void,
        empty_stadiums=empty,
        subs_regime=3 if season <= 2021 else 5,
        xg_in_api=season >= 2022,
        manager_elements_in_api=season == 2024,
        detailed_defensive_stats=defensive,
        lockdown="0900_next_day" if season >= 2026 else "legacy",
    )


#: Feature flags for every season in :data:`SEASONS` (FPL_KNOWLEDGE §2.2).
SEASON_FLAGS: dict[int, SeasonFlags] = {s: _build_flags(s) for s in SEASONS}


# --------------------------------------------------------------------------------------
# Chip windows (FPL_KNOWLEDGE §1.7 and §2.1)
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ChipWindow:
    """One playable instance of a chip: valid from ``first_gw`` to ``last_gw`` inclusive."""

    name: str  # "WC" | "FH" | "BB" | "TC" | "AOA" | "AM"
    set: int  # 1 for the first (or only) instance of a chip, 2 for the second
    first_gw: int
    last_gw: int


def chip_windows(season: int) -> tuple[ChipWindow, ...]:
    """Return every chip instance playable in ``season`` with its GW validity window.

    2025-26 / 2026-27: two of each chip; set 1 expires at the GW19 deadline (unused
    chips lost, no carry-over), set 2 covers GW20-38. WC/FH open at GW2 (transfers are
    unlimited before the GW1 deadline; FH is never playable in GW1), BB/TC at GW1.
    CONFIRMED AT LAUNCH for 2026-27: the day-1 bootstrap ``chips[]`` (snapshot
    2026-07-23) carries exactly these 8 instances — wildcard/freehit start_event 2,
    bboost/3xc start_event 1, all stop_event 19; set 2 all 20-38. GW19 deadline
    2027-01-02T13:30:00Z per ``events[]``.

    Earlier seasons: two Wildcards (canonical half-season split GW2-19 / GW20-38 —
    exact historical WC1 expiry dates varied by calendar and are UNCERTAIN at the
    margin, except 2022-23 where the doc pins WC1 to GW2-16 and WC2 to GW18+ around
    the Qatar World Cup break) plus single BB/TC (GW1-38) and FH (GW2-38; AOA in
    2016-17 instead). 2021-22 adds the Omicron extra FH from GW20 (the two FHs were
    not playable in consecutive GWs — not expressible here; handle in the planner).
    2024-25 adds the Assistant Manager chip, playable GW24+ for 3 consecutive GWs.
    At most one chip may be active per GW in all seasons.
    """
    _validate_season(season)
    if season >= 2025:
        return (
            ChipWindow("WC", 1, 2, 19),
            ChipWindow("WC", 2, 20, 38),
            ChipWindow("FH", 1, 2, 19),
            ChipWindow("FH", 2, 20, 38),
            ChipWindow("BB", 1, 1, 19),
            ChipWindow("BB", 2, 20, 38),
            ChipWindow("TC", 1, 1, 19),
            ChipWindow("TC", 2, 20, 38),
        )
    if season == 2022:  # WC1 expired at the GW16 deadline (12 Nov); WC2 from GW18
        wc1, wc2 = ChipWindow("WC", 1, 2, 16), ChipWindow("WC", 2, 18, 38)
    else:
        wc1, wc2 = ChipWindow("WC", 1, 2, 19), ChipWindow("WC", 2, 20, 38)
    windows = [wc1, wc2, ChipWindow("BB", 1, 1, 38), ChipWindow("TC", 1, 1, 38)]
    if season == 2016:
        windows.append(ChipWindow("AOA", 1, 1, 38))
    else:
        windows.append(ChipWindow("FH", 1, 2, 38))
    if season == 2021:
        windows.append(ChipWindow("FH", 2, 20, 38))  # Omicron extra FH from GW20
    if season == 2024:
        windows.append(ChipWindow("AM", 1, 24, 38))  # Assistant Manager, GW24+
    return tuple(windows)


#: GW at whose deadline chip set 1 expires in the two-set era (2025-26+).
SET1_EXPIRY_GW: int = 19


# --------------------------------------------------------------------------------------
# Selling price (FPL_KNOWLEDGE §1.8)
# --------------------------------------------------------------------------------------


def sell_price(purchase: int, now: int) -> int:
    """Return the selling price in £0.1m units given purchase and current prices.

    50% sell-on fee on profit, floored to £0.1m: ``purchase + floor((now - purchase)/2)``
    when in profit; the full price fall is borne otherwise (sell at ``now``).
    E.g. bought 55, now 58 -> 56; bought 55, now 52 -> 52.
    """
    if purchase < 0 or now < 0:
        raise ValueError(f"prices must be non-negative, got purchase={purchase}, now={now}")
    if now <= purchase:
        return now
    return purchase + (now - purchase) // 2


# --------------------------------------------------------------------------------------
# BPS (FPL_KNOWLEDGE §1.2)
# --------------------------------------------------------------------------------------

#: Full 2026-27 (v4) BPS matrix. Community-standard reconstruction with the four
#: official 2026-27 deltas applied — FPL has never published a canonical
#: machine-readable BPS table, so treat as best-available (UNCERTAIN at the margin).
BPS_2026: dict[str, float] = {
    "minutes_1_60": 3,  # playing 1-60 min
    "minutes_over_60": 6,
    "goal_gkp_def": 12,
    "goal_mid": 18,
    "goal_fwd": 24,
    "penalty_goal": 12,  # any penalty goal, all positions (since v3)
    "assist": 9,
    "clean_sheet_gkp_def": 12,
    "match_winning_goal": 3,
    "gk_save": 2,  # any save (v4 restructure)
    "gk_save_inside_box_extra": 1,  # additional, net 3 for an in-box save
    "gk_big_chance_save_extra": 1,  # additional (new in v4)
    "gk_penalty_save": 7,  # big-chance +1 usually applies -> 8 net
    "goal_line_clearance": 9,
    "tackle_won": 2,  # tackles won (not net, since v3)
    "cbi_per_3": 1,  # per 3 clearances+blocks+interceptions (per 2 before v4)
    "recoveries_per_3": 1,
    "key_pass": 1,
    "big_chance_created": 3,
    "open_play_cross": 1,  # successful open-play cross
    "successful_dribble": 1,
    "foul_won": 1,
    "shot_on_target": 2,
    "pass_completion_70": 2,  # 70-79% completion, >= 30 passes attempted
    "pass_completion_80": 4,  # 80-89%
    "pass_completion_90": 6,  # 90%+
    "goal_conceded_gkp_def": -4,  # each
    "penalty_conceded": -3,
    "penalty_miss": -6,
    "yellow_card": -3,
    "red_card": -9,
    "own_goal": -6,
    "big_chance_missed": -3,
    "error_leading_to_goal": -3,
    "error_leading_to_attempt": -1,
    "foul_conceded": -1,
    "caught_offside": -1,
    "shot_off_target": -1,
}

#: Changed entries per BPS version, relative to the PREVIOUS version, keyed like
#: :data:`BPS_2026`. ``None`` means the item was removed in that version. To
#: reconstruct season s's matrix: start from BPS_2026, undo deltas back to the
#: season's ``bps_version`` (or apply forward from a v1 base). v1 covers <=2023-24,
#: v2 = 2024-25, v3 = 2025-26, v4 = 2026-27.
BPS_DELTAS: dict[str, dict[str, float | None]] = {
    "v2": {  # 2024-25 vs v1
        "gk_penalty_save": 9,  # was 15
        "goal_conceded_gkp_def": -4,  # new
        "goal_line_clearance": 3,  # new
        "foul_won": 1,  # new
        "shot_on_target": 2,  # new
    },
    "v3": {  # 2025-26 vs v2
        "penalty_goal": 12,  # penalty goals score 12 BPS for all positions
        "gk_save_inside_box": 3,  # save categories split in/out of box
        "gk_save_outside_box": 2,
        "gk_penalty_save": 8,  # was 9
        "goal_line_clearance": 9,  # was 3
        "tackle_won": 2,  # tackles won, no longer net of tackles lost
    },
    "v4": {  # 2026-27 vs v3 (the four official changes)
        "tackled_against": None,  # "being tackled" -1 each -> removed
        "cbi_per_2": None,  # +1 per 2 CBI -> replaced by per 3
        "cbi_per_3": 1,
        "gk_save_inside_box": None,  # category removed ...
        "gk_save_outside_box": None,  # ... category removed
        "gk_save": 2,  # +2 any save
        "gk_save_inside_box_extra": 1,  # +1 extra in-box (net 3, as before)
        "gk_big_chance_save_extra": 1,  # new
        "gk_penalty_save": 7,  # was 8
    },
}


# --------------------------------------------------------------------------------------
# 2026-27 position reclassifications (FPL_KNOWLEDGE §1.10)
# --------------------------------------------------------------------------------------

#: The 11 players reclassified for 2026-27 as (web_name, club_short, old_pos, new_pos,
#: player_code). Verified against the 2026-27 day-1 bootstrap (snapshot 2026-07-23) by
#: ``element.code``: 10 of 11 CONFIRMED holding new_pos; web_name is the launch
#: spelling. Eric da Silva Moreira (code 569014) was ABSENT from the day-1 elements[]
#: entirely — his announced MID -> DEF change is retained (official announcement) but is
#: not API-verified; his web_name below is the 2025-26 form. See
#: :data:`RECLASSIFIED_ABSENT_AT_LAUNCH_2026`.
#: Historical per-90 features for these players were earned under the OLD position.
POSITION_RECLASSIFICATIONS_2026: tuple[tuple[str, str, str, str, int], ...] = (
    ("Lewis-Skelly", "ARS", "DEF", "MID", 499169),  # Myles Lewis-Skelly, Arsenal
    ("Bogarde", "AVL", "DEF", "MID", 515597),  # Lamare Bogarde, Aston Villa
    ("Kroupi.Jr", "BOU", "FWD", "MID", 560262),  # Junior Kroupi, Bournemouth
    ("Lewis-Potter", "BRE", "DEF", "MID", 249231),  # Keane Lewis-Potter, Brentford
    ("Wieffer", "BHA", "MID", "DEF", 467779),  # Mats Wieffer, Brighton
    ("Georginio", "BHA", "MID", "FWD", 463067),  # Georginio Rutter, Brighton
    ("Cardines", "CRY", "MID", "DEF", 590014),  # Rio Cardines, Crystal Palace
    ("Sessegnon", "FUL", "MID", "DEF", 184349),  # Ryan Sessegnon, Fulham
    ("Marmoush", "MCI", "MID", "FWD", 438234),  # Omar Marmoush, Man City
    ("Dorgu", "MUN", "DEF", "MID", 596777),  # Patrick Dorgu, Man Utd
    ("Da Silva Moreira", "NFO", "MID", "DEF", 569014),  # Eric da Silva Moreira, Nott'm Forest
)

#: player_codes from the announced 2026-27 reclassifications with NO element in the
#: day-1 2026-27 bootstrap (snapshot 2026-07-23). Treat these reclassifications as
#: announcement-only until the player (re)appears in the API.
RECLASSIFIED_ABSENT_AT_LAUNCH_2026: frozenset[int] = frozenset({569014})


# --------------------------------------------------------------------------------------
# Transfer state machine (FPL_KNOWLEDGE §1.6, §2.2)
# --------------------------------------------------------------------------------------

#: Hard cap on transfers in a single GW (API ``transfers_cap``); does not apply in
#: WC/FH weeks. CONFIRMED AT LAUNCH for 2026-27: ``transfers_cap: 20`` in the day-1
#: bootstrap ``game_settings`` (snapshot 2026-07-23).
TRANSFERS_CAP_PER_GW: int = 20


def next_free_transfers(current: int, used: int, season: int, wildcard_or_fh: bool) -> int:
    """Return the free-transfer count going into the NEXT gameweek.

    ``current`` is the FT count going into this GW, ``used`` the number of transfers
    made this GW (transfers beyond ``current`` are -4 hits and never push the next
    balance below the floor of 1). ``wildcard_or_fh`` marks a WC/FH active this GW
    (transfers then cost nothing and consume no FTs).

    Season regimes: from 2024-25, ``FT_next = min(current - used + 1, 5)`` floored at
    1, and WC/FH pass the bank through with the +1 accrual (``min(current + 1, 5)``).
    Through 2023-24 the bank cap is 2 and playing WC/FH WIPES banked FTs (next GW
    starts on exactly 1). GW1 pre-deadline unlimited transfers are out of scope.
    """
    _validate_season(season)
    if current < 0 or used < 0:
        raise ValueError(f"current and used must be >= 0, got current={current}, used={used}")
    flags = SEASON_FLAGS[season]
    cap = flags.max_free_transfers
    if wildcard_or_fh:
        if flags.chips_wipe_banked_fts:
            return 1
        return min(current + 1, cap)
    return max(1, min(current - used + 1, cap))


# --------------------------------------------------------------------------------------
# Fixed dates and remaps
# --------------------------------------------------------------------------------------

#: 2026-27 GW1 deadline: Friday 21 Aug 2026, 18:30 BST = 17:30 UTC.
#: CONFIRMED AT LAUNCH — ``events[0].deadline_time == "2026-08-21T17:30:00Z"`` in the
#: 2026-27 day-1 bootstrap (snapshot 2026-07-23, static_content_url .../2026_27/).
GW1_DEADLINE_2026: datetime = datetime(2026, 8, 21, 17, 30, tzinfo=UTC)

#: 2019-20 COVID restart: the API renumbered the restarted rounds as events 39-47.
#: Canonical mapping API event id -> canonical GW (the void original GWs 30-38 are
#: replaced by the restarted rounds; see FPL_KNOWLEDGE §2.4).
VOID_EVENT_REMAP: dict[int, int] = {api_event: api_event - 9 for api_event in range(39, 48)}
