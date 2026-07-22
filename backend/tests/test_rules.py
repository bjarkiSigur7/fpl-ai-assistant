"""Offline tests for fplai.rules — the FPL_KNOWLEDGE Parts 1+2 machine encoding.

Pure-module tests: no network, no fixtures, no I/O.
"""

from datetime import UTC, datetime

import pytest

from fplai import rules
from fplai.rules import (
    BPS_2026,
    BPS_DELTAS,
    CONCEDED_PER_POINT,
    DEFCON_THRESHOLDS,
    GW1_DEADLINE_2026,
    POSITION_RECLASSIFICATIONS_2026,
    POSITIONS,
    SAVES_PER_POINT,
    SEASON_FLAGS,
    SEASONS,
    SET1_EXPIRY_GW,
    VOID_EVENT_REMAP,
    ChipWindow,
    chip_windows,
    get_scoring,
    next_free_transfers,
    season_label,
    sell_price,
)


class TestSeasonBasics:
    def test_seasons_range(self) -> None:
        assert list(SEASONS) == list(range(2016, 2027))
        assert len(SEASONS) == 11

    def test_season_label(self) -> None:
        assert season_label(2016) == "2016-17"
        assert season_label(2019) == "2019-20"  # zero-padded wrap into the 20s
        assert season_label(2026) == "2026-27"

    def test_positions(self) -> None:
        assert POSITIONS == ("GKP", "DEF", "MID", "FWD")


class TestScoring:
    def test_gk_goal_6_through_2023(self) -> None:
        for season in range(2016, 2024):
            assert get_scoring(season)["goals_scored"]["GKP"] == 6, season

    def test_gk_goal_10_from_2024(self) -> None:
        for season in (2024, 2025, 2026):
            assert get_scoring(season)["goals_scored"]["GKP"] == 10, season

    def test_outfield_goals_stable_all_seasons(self) -> None:
        for season in SEASONS:
            table = get_scoring(season)
            assert table["goals_scored"]["DEF"] == 6
            assert table["goals_scored"]["MID"] == 5
            assert table["goals_scored"]["FWD"] == 4

    def test_defcon_only_from_2025(self) -> None:
        for season in range(2016, 2025):
            assert "defensive_contribution" not in get_scoring(season), season
        for season in (2025, 2026):
            defcon = get_scoring(season)["defensive_contribution"]
            assert defcon == {"GKP": 0, "DEF": 2, "MID": 2, "FWD": 2}

    def test_clean_sheets_and_conceded(self) -> None:
        table = get_scoring(2026)
        assert table["clean_sheets"] == {"GKP": 4, "DEF": 4, "MID": 1, "FWD": 0}
        assert table["goals_conceded"] == {"GKP": -1, "DEF": -1, "MID": 0, "FWD": 0}

    def test_flat_actions(self) -> None:
        table = get_scoring(2026)
        assert table["assists"] == {p: 3 for p in POSITIONS}
        assert table["penalties_missed"] == {p: -2 for p in POSITIONS}
        assert table["yellow_cards"] == {p: -1 for p in POSITIONS}
        assert table["red_cards"] == {p: -3 for p in POSITIONS}
        assert table["own_goals"] == {p: -2 for p in POSITIONS}
        assert table["short_play"] == {p: 1 for p in POSITIONS}
        assert table["long_play"] == {p: 2 for p in POSITIONS}

    def test_gk_only_actions(self) -> None:
        table = get_scoring(2026)
        assert table["saves"] == {"GKP": 1, "DEF": 0, "MID": 0, "FWD": 0}
        assert table["penalties_saved"]["GKP"] == 5

    def test_divisors_and_thresholds(self) -> None:
        assert SAVES_PER_POINT == 3
        assert CONCEDED_PER_POINT == 2
        assert DEFCON_THRESHOLDS == {"DEF": 10, "MID": 12, "FWD": 12}
        assert "GKP" not in DEFCON_THRESHOLDS  # GKs ineligible

    def test_every_action_covers_all_positions(self) -> None:
        for season in SEASONS:
            for action, per_pos in get_scoring(season).items():
                assert set(per_pos) == set(POSITIONS), (season, action)

    def test_invalid_season_raises(self) -> None:
        with pytest.raises(ValueError):
            get_scoring(2015)
        with pytest.raises(ValueError):
            get_scoring(2027)

    def test_returns_independent_copy(self) -> None:
        table = get_scoring(2026)
        table["goals_scored"]["GKP"] = 999
        assert get_scoring(2026)["goals_scored"]["GKP"] == 10


class TestSeasonFlags:
    def test_all_seasons_present(self) -> None:
        assert set(SEASON_FLAGS) == set(SEASONS)
        for season, flags in SEASON_FLAGS.items():
            assert flags.season == season

    def test_aoa_only_2016(self) -> None:
        assert "AOA" in SEASON_FLAGS[2016].chip_inventory
        for season in range(2017, 2027):
            assert "AOA" not in SEASON_FLAGS[season].chip_inventory, season

    def test_fh_from_2017(self) -> None:
        assert "FH" not in SEASON_FLAGS[2016].chip_inventory
        for season in range(2017, 2027):
            assert "FH" in SEASON_FLAGS[season].chip_inventory, season

    def test_two_of_each_chip_2025_onwards(self) -> None:
        for season in (2025, 2026):
            inv = SEASON_FLAGS[season].chip_inventory
            assert sorted(inv) == ["BB", "BB", "FH", "FH", "TC", "TC", "WC", "WC"]

    def test_special_inventories(self) -> None:
        assert "FH_extra_from_GW20" in SEASON_FLAGS[2021].chip_inventory
        assert "AM_from_GW24" in SEASON_FLAGS[2024].chip_inventory
        assert SEASON_FLAGS[2022].chip_inventory == ("WC", "WC", "BB", "TC", "FH")

    def test_assist_definition_versions(self) -> None:
        assert SEASON_FLAGS[2016].assist_definition == "v1"
        assert SEASON_FLAGS[2017].assist_definition == "v2"
        assert SEASON_FLAGS[2024].assist_definition == "v2"
        assert SEASON_FLAGS[2025].assist_definition == "v3"
        assert SEASON_FLAGS[2026].assist_definition == "v3"

    def test_bps_versions(self) -> None:
        assert SEASON_FLAGS[2016].bps_version == "v1"
        assert SEASON_FLAGS[2023].bps_version == "v1"
        assert SEASON_FLAGS[2024].bps_version == "v2"
        assert SEASON_FLAGS[2025].bps_version == "v3"
        assert SEASON_FLAGS[2026].bps_version == "v4"

    def test_gk_goal_points_flag(self) -> None:
        assert SEASON_FLAGS[2023].gk_goal_points == 6
        assert SEASON_FLAGS[2024].gk_goal_points == 10

    def test_defcon_flag(self) -> None:
        assert not SEASON_FLAGS[2024].defcon
        assert SEASON_FLAGS[2025].defcon
        assert SEASON_FLAGS[2026].defcon

    def test_transfer_regime_flags(self) -> None:
        assert SEASON_FLAGS[2023].max_free_transfers == 2
        assert SEASON_FLAGS[2024].max_free_transfers == 5
        assert SEASON_FLAGS[2023].chips_wipe_banked_fts is True
        assert SEASON_FLAGS[2024].chips_wipe_banked_fts is False

    def test_deadline_minutes(self) -> None:
        assert SEASON_FLAGS[2019].deadline_minutes == 60
        assert SEASON_FLAGS[2020].deadline_minutes == 90
        assert SEASON_FLAGS[2026].deadline_minutes == 90

    def test_amnesties(self) -> None:
        assert SEASON_FLAGS[2019].amnesty == "unlimited_before_event_39"
        assert SEASON_FLAGS[2022].amnesty == "unlimited_between_GW16_and_GW17"
        assert SEASON_FLAGS[2025].amnesty == "topped_to_5_at_GW16"
        assert SEASON_FLAGS[2026].amnesty is None  # no AFCON top-up in 2026-27

    def test_void_events(self) -> None:
        assert SEASON_FLAGS[2019].void_events == tuple(range(30, 39))
        assert SEASON_FLAGS[2022].void_events == (7,)
        assert SEASON_FLAGS[2020].void_events == ()

    def test_empty_stadium_regime(self) -> None:
        assert SEASON_FLAGS[2019].empty_stadiums == "events_39_47"
        assert SEASON_FLAGS[2020].empty_stadiums == "all"
        assert SEASON_FLAGS[2021].empty_stadiums is None

    def test_subs_regime(self) -> None:
        assert SEASON_FLAGS[2021].subs_regime == 3
        assert SEASON_FLAGS[2022].subs_regime == 5

    def test_data_availability_flags(self) -> None:
        assert not SEASON_FLAGS[2021].xg_in_api
        assert SEASON_FLAGS[2022].xg_in_api
        assert SEASON_FLAGS[2024].manager_elements_in_api
        assert not SEASON_FLAGS[2023].manager_elements_in_api
        assert not SEASON_FLAGS[2025].manager_elements_in_api
        assert SEASON_FLAGS[2018].detailed_defensive_stats == "full"
        assert SEASON_FLAGS[2019].detailed_defensive_stats == "none"
        assert SEASON_FLAGS[2025].detailed_defensive_stats == "cbit_recoveries"

    def test_lockdown(self) -> None:
        assert SEASON_FLAGS[2025].lockdown == "legacy"
        assert SEASON_FLAGS[2026].lockdown == "0900_next_day"


class TestChipWindows:
    def test_2025_two_set_structure(self) -> None:
        for season in (2025, 2026):
            windows = chip_windows(season)
            assert len(windows) == 8
            by_key = {(w.name, w.set): w for w in windows}
            for name in ("WC", "FH", "BB", "TC"):
                assert by_key[(name, 1)].last_gw == SET1_EXPIRY_GW  # set-1 expiry at GW19
                assert by_key[(name, 2)].first_gw == 20
                assert by_key[(name, 2)].last_gw == 38
            # WC/FH open GW2 (FH never playable GW1); BB/TC open GW1
            assert by_key[("WC", 1)].first_gw == 2
            assert by_key[("FH", 1)].first_gw == 2
            assert by_key[("BB", 1)].first_gw == 1
            assert by_key[("TC", 1)].first_gw == 1

    def test_2016_has_aoa_no_fh(self) -> None:
        windows = chip_windows(2016)
        names = [w.name for w in windows]
        assert "AOA" in names
        assert "FH" not in names
        assert names.count("WC") == 2

    def test_2017_fh_replaces_aoa(self) -> None:
        windows = chip_windows(2017)
        names = [w.name for w in windows]
        assert "FH" in names
        assert "AOA" not in names
        fh = next(w for w in windows if w.name == "FH")
        assert (fh.first_gw, fh.last_gw) == (2, 38)

    def test_2021_extra_fh(self) -> None:
        fh_windows = [w for w in chip_windows(2021) if w.name == "FH"]
        assert len(fh_windows) == 2
        extra = next(w for w in fh_windows if w.set == 2)
        assert extra.first_gw == 20  # Omicron grant from GW20

    def test_2022_world_cup_wildcard_split(self) -> None:
        by_key = {(w.name, w.set): w for w in chip_windows(2022)}
        assert by_key[("WC", 1)].last_gw == 16  # expired at the GW16 deadline
        assert by_key[("WC", 2)].first_gw == 18  # second WC from GW18

    def test_2024_assistant_manager(self) -> None:
        am = next(w for w in chip_windows(2024) if w.name == "AM")
        assert (am.first_gw, am.last_gw) == (24, 38)

    def test_single_chip_seasons_bb_tc_full_season(self) -> None:
        for season in range(2016, 2025):
            by_key = {(w.name, w.set): w for w in chip_windows(season)}
            assert (by_key[("BB", 1)].first_gw, by_key[("BB", 1)].last_gw) == (1, 38)
            assert (by_key[("TC", 1)].first_gw, by_key[("TC", 1)].last_gw) == (1, 38)

    def test_windows_are_frozen(self) -> None:
        window = chip_windows(2026)[0]
        assert isinstance(window, ChipWindow)
        with pytest.raises(AttributeError):
            window.first_gw = 5  # type: ignore[misc]


class TestSellPrice:
    def test_no_change(self) -> None:
        assert sell_price(50, 50) == 50

    def test_loss_borne_in_full(self) -> None:
        assert sell_price(50, 44) == 44
        assert sell_price(50, 49) == 49

    def test_even_profit_split(self) -> None:
        assert sell_price(50, 56) == 53
        assert sell_price(55, 58) == 56  # docs example: bought 5.5, now 5.8 -> sell 5.6

    def test_odd_profit_floors(self) -> None:
        assert sell_price(50, 55) == 52  # floor(5/2) = 2
        assert sell_price(50, 51) == 50  # a single 0.1 rise yields nothing
        assert sell_price(50, 53) == 51

    def test_negative_raises(self) -> None:
        with pytest.raises(ValueError):
            sell_price(-1, 50)
        with pytest.raises(ValueError):
            sell_price(50, -1)


class TestTransferStateMachine:
    def test_modern_accrual_and_cap(self) -> None:
        assert next_free_transfers(1, 0, 2025, False) == 2
        assert next_free_transfers(2, 1, 2025, False) == 2
        assert next_free_transfers(4, 0, 2025, False) == 5
        assert next_free_transfers(5, 0, 2025, False) == 5  # capped at 5
        assert next_free_transfers(5, 5, 2025, False) == 1

    def test_hits_never_push_below_floor(self) -> None:
        # 1 FT, 4 transfers -> 3 hits (-12 pts) but next GW still starts on 1
        assert next_free_transfers(1, 4, 2025, False) == 1
        assert next_free_transfers(2, 5, 2026, False) == 1

    def test_pre_2024_cap_of_2(self) -> None:
        assert next_free_transfers(1, 0, 2023, False) == 2
        assert next_free_transfers(2, 0, 2023, False) == 2  # capped at 2, not 3
        assert next_free_transfers(2, 1, 2023, False) == 2
        assert next_free_transfers(1, 1, 2023, False) == 1
        assert next_free_transfers(1, 3, 2016, False) == 1  # 2 hits, floor 1

    def test_chips_wipe_bank_pre_2024(self) -> None:
        assert next_free_transfers(2, 0, 2023, True) == 1
        assert next_free_transfers(2, 7, 2016, True) == 1  # used is irrelevant on WC/FH

    def test_chips_pass_bank_through_from_2024(self) -> None:
        assert next_free_transfers(3, 15, 2024, True) == 4  # +1 accrual, no consumption
        assert next_free_transfers(5, 0, 2025, True) == 5  # still capped
        assert next_free_transfers(1, 11, 2026, True) == 2

    def test_invalid_inputs_raise(self) -> None:
        with pytest.raises(ValueError):
            next_free_transfers(-1, 0, 2025, False)
        with pytest.raises(ValueError):
            next_free_transfers(1, -2, 2025, False)
        with pytest.raises(ValueError):
            next_free_transfers(1, 0, 2015, False)


class TestBps:
    def test_v4_headline_values(self) -> None:
        assert BPS_2026["gk_penalty_save"] == 7
        assert BPS_2026["cbi_per_3"] == 1
        assert BPS_2026["gk_save"] == 2
        assert BPS_2026["gk_save_inside_box_extra"] == 1  # net 3 for an in-box save
        assert BPS_2026["gk_big_chance_save_extra"] == 1
        assert "tackled_against" not in BPS_2026  # removed in v4
        assert "cbi_per_2" not in BPS_2026
        assert "gk_save_inside_box" not in BPS_2026  # v3 categories dropped
        assert "gk_save_outside_box" not in BPS_2026

    def test_goal_and_position_values(self) -> None:
        assert BPS_2026["goal_gkp_def"] == 12
        assert BPS_2026["goal_mid"] == 18
        assert BPS_2026["goal_fwd"] == 24
        assert BPS_2026["penalty_goal"] == 12
        assert BPS_2026["assist"] == 9
        assert BPS_2026["clean_sheet_gkp_def"] == 12
        assert BPS_2026["goal_conceded_gkp_def"] == -4
        assert BPS_2026["red_card"] == -9
        assert BPS_2026["pass_completion_90"] == 6

    def test_deltas_document_version_history(self) -> None:
        assert set(BPS_DELTAS) == {"v2", "v3", "v4"}
        assert BPS_DELTAS["v2"]["gk_penalty_save"] == 9  # 15 -> 9
        assert BPS_DELTAS["v3"]["gk_penalty_save"] == 8
        assert BPS_DELTAS["v4"]["gk_penalty_save"] == 7
        assert BPS_DELTAS["v4"]["tackled_against"] is None  # removal marker
        assert BPS_DELTAS["v4"]["cbi_per_2"] is None
        assert BPS_DELTAS["v3"]["goal_line_clearance"] == 9
        assert BPS_DELTAS["v2"]["goal_line_clearance"] == 3

    def test_v4_deltas_are_consistent_with_bps_2026(self) -> None:
        for key, value in BPS_DELTAS["v4"].items():
            if value is None:
                assert key not in BPS_2026, key
            else:
                assert BPS_2026[key] == value, key


class TestReclassifications:
    def test_eleven_players(self) -> None:
        assert len(POSITION_RECLASSIFICATIONS_2026) == 11
        assert len({(n, c) for n, c, _, _ in POSITION_RECLASSIFICATIONS_2026}) == 11

    def test_known_entries(self) -> None:
        assert ("Wieffer", "BHA", "MID", "DEF") in POSITION_RECLASSIFICATIONS_2026
        assert ("Marmoush", "MCI", "MID", "FWD") in POSITION_RECLASSIFICATIONS_2026
        assert ("Lewis-Skelly", "ARS", "DEF", "MID") in POSITION_RECLASSIFICATIONS_2026
        assert ("Kroupi", "BOU", "FWD", "MID") in POSITION_RECLASSIFICATIONS_2026

    def test_positions_valid_and_changed(self) -> None:
        for name, _club, old, new in POSITION_RECLASSIFICATIONS_2026:
            assert old in POSITIONS and new in POSITIONS, name
            assert old != new, name
            assert "GKP" not in (old, new), name  # no GK reclassifications

    def test_direction_counts(self) -> None:
        to_mid = [r for r in POSITION_RECLASSIFICATIONS_2026 if r[3] == "MID"]
        to_def = [r for r in POSITION_RECLASSIFICATIONS_2026 if r[3] == "DEF"]
        to_fwd = [r for r in POSITION_RECLASSIFICATIONS_2026 if r[3] == "FWD"]
        assert (len(to_mid), len(to_def), len(to_fwd)) == (5, 4, 2)


class TestDatesAndRemaps:
    def test_gw1_deadline_2026(self) -> None:
        assert GW1_DEADLINE_2026 == datetime(2026, 8, 21, 17, 30, tzinfo=UTC)
        assert GW1_DEADLINE_2026.tzinfo is UTC  # 18:30 BST stored as UTC

    def test_void_event_remap(self) -> None:
        assert VOID_EVENT_REMAP == {
            39: 30,
            40: 31,
            41: 32,
            42: 33,
            43: 34,
            44: 35,
            45: 36,
            46: 37,
            47: 38,
        }
        assert set(VOID_EVENT_REMAP.values()) == set(SEASON_FLAGS[2019].void_events)

    def test_module_is_pure(self) -> None:
        # No I/O machinery should leak into this module's namespace.
        for banned in ("os", "requests", "httpx", "open_file", "Path"):
            assert not hasattr(rules, banned)
