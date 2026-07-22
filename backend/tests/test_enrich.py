"""Offline tests for fplai.data.enrich and the crosswalk fuzzy matcher.

Everything is synthetic and generated in-code: a tiny processed-table set (teams,
players, fixtures, player_match, odds) plus a raw tree with an Understat league.json
and vaastav-style per-player understat CSVs.  No network, no real-data dependence.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from fplai.data import build, crosswalk, enrich

# --- name normalization / similarity ------------------------------------------------


def test_name_tokens_unescapes_html_and_strips_accents() -> None:
    assert crosswalk.name_tokens("N&#039;Golo Kanté") == frozenset({"n", "golo", "kante"})
    assert crosswalk.name_tokens("Sadio Mané") == crosswalk.name_tokens("Sadio Mane")


@pytest.mark.parametrize(
    ("variants", "understat_name"),
    [
        (["Sadio Mané"], "Sadio Mane"),  # accents
        (["Heung-min Son", "Son"], "Son Heung-Min"),  # hyphen + token order
        (["Son Heung-min"], "Heung-Min Son"),  # full reversal
        (["Bruno Miguel Borges Fernandes", "B.Fernandes"], "Bruno Fernandes"),  # subset
        (["Diogo Teixeira da Silva", "Diogo J."], "Diogo Jota"),  # web-name initial
        (["Benjamin White", "White"], "Ben White"),  # diminutive prefix
        (["Matty Cash"], "Matthew Cash"),  # shared 4-letter stem
        (["Nayef Aguerd", "N.Aguerd"], "Naif Aguerd"),  # transliteration
        (["Ederson Santana de Moraes", "Ederson M."], "Ederson"),  # mononym
    ],
)
def test_name_similarity_accepts_known_renderings(
    variants: list[str], understat_name: str
) -> None:
    assert crosswalk.name_similarity(variants, understat_name) >= crosswalk.TEAM_FUZZY_THRESHOLD


def test_name_similarity_rejects_different_players() -> None:
    # Same club, different people — must stay below the same-team threshold.
    assert (
        crosswalk.name_similarity(["Gabriel Fernando de Jesus", "G.Jesus"], "Gabriel Magalhaes")
        < crosswalk.TEAM_FUZZY_THRESHOLD
    )
    assert crosswalk.name_similarity(["Kevin De Bruyne"], "John Stones") < 0.5


# --- match_players ------------------------------------------------------------------


def _fpl_frame(rows: list[tuple[int, list[str], set[str]]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "player_code": [r[0] for r in rows],
            "variants": [r[1] for r in rows],
            "teams": [frozenset(r[2]) for r in rows],
        }
    )


def _us_frame(rows: list[tuple[int, str, set[str]]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "understat_id": [r[0] for r in rows],
            "player_name": [r[1] for r in rows],
            "teams": [frozenset(r[2]) for r in rows],
        }
    )


def test_match_players_exact_fuzzy_and_surname_passes() -> None:
    fpl = _fpl_frame(
        [
            (1, ["Son Heung-min", "Son"], {"Tottenham"}),
            (2, ["Ederson Santana de Moraes", "Ederson M."], {"Manchester City"}),
            (3, ["Matty Cash"], {"Aston Villa"}),
            (4, ["Joseph Gomez", "Joe Gomez"], {"Liverpool"}),
            (5, ["Kevin De Bruyne"], {"Manchester City"}),
        ]
    )
    us = _us_frame(
        [
            (453, "Heung-Min Son", {"Tottenham"}),
            (6054, "Ederson", {"Manchester City"}),
            (8864, "Matthew Cash", {"Aston Villa"}),
            (987, "Joe Gomez", {"Liverpool"}),
            (447, "Kevin De Bruyne", {"Manchester City"}),
        ]
    )
    matches = {m.player_code: m for m in crosswalk.match_players(fpl, us)}
    assert matches[1].understat_id == 453
    assert matches[2].understat_id == 6054
    assert matches[3].understat_id == 8864
    assert matches[4].understat_id == 987  # exact via the "Joe Gomez" variant
    assert matches[5].understat_id == 447 and matches[5].method == "exact"


def test_match_players_same_name_disambiguated_by_team() -> None:
    fpl = _fpl_frame(
        [
            (10, ["Danilo dos Santos de Oliveira", "Danilo"], {"Nottingham Forest"}),
            (11, ["Danilo Luiz da Silva", "Danilo"], {"Manchester City"}),
        ]
    )
    us = _us_frame(
        [
            (900, "Danilo", {"Nottingham Forest"}),
            (901, "Danilo", {"Manchester City"}),
        ]
    )
    matches = {m.player_code: m.understat_id for m in crosswalk.match_players(fpl, us)}
    assert matches == {10: 900, 11: 901}


def test_match_players_seeds_win_and_block_reuse() -> None:
    fpl = _fpl_frame([(1, ["Rodrigo Moreno Machado", "Rodrigo"], {"Leeds"})])
    us = _us_frame([(2381, "Rodrigo", {"Leeds"}), (999, "Rodrigo Moreno", {"Leeds"})])
    matches = crosswalk.match_players(fpl, us, seeds={1: 999})
    assert len(matches) == 1
    assert matches[0].understat_id == 999 and matches[0].method == "seed"


def test_match_players_is_one_to_one_and_deterministic() -> None:
    fpl = _fpl_frame(
        [(i, [f"Player Alpha{i}"], {"Arsenal"}) for i in range(1, 4)]
    )
    us = _us_frame([(100 + i, f"Player Alpha{i}", {"Arsenal"}) for i in range(1, 4)])
    first = crosswalk.match_players(fpl, us)
    second = crosswalk.match_players(fpl, us)
    assert first == second
    assert len({m.understat_id for m in first}) == len(first) == 3


# --- team name maps -----------------------------------------------------------------


def test_apply_team_name_maps_fills_all_known_clubs() -> None:
    teams = pd.DataFrame(
        {
            "season": [2024, 2024, 2024, 2016],
            "fpl_team_id": [1, 2, 3, 4],
            "team_code": [43, 1, 17, 999],
            "name": pd.array(["Man City", "Man Utd", "Nott'm Forest", "Unknown FC"],
                             dtype="string"),
            "short_name": pd.array(["MCI", "MUN", "NFO", "UNK"], dtype="string"),
            "understat_name": pd.array([pd.NA] * 4, dtype="string"),
            "clubelo_name": pd.array([pd.NA] * 4, dtype="string"),
            "footballdata_name": pd.array([pd.NA] * 4, dtype="string"),
        }
    )
    out = crosswalk.apply_team_name_maps(teams)
    city = out[out["name"] == "Man City"].iloc[0]
    assert city["understat_name"] == "Manchester City"
    assert city["footballdata_name"] == "Man City"
    assert city["clubelo_name"] == "Man City"
    utd = out[out["name"] == "Man Utd"].iloc[0]
    assert utd["footballdata_name"] == "Man United"
    assert utd["understat_name"] == "Manchester United"
    forest = out[out["name"] == "Nott'm Forest"].iloc[0]
    assert forest["understat_name"] == "Nottingham Forest"
    assert forest["clubelo_name"] == "Forest"
    unknown = out[out["name"] == "Unknown FC"].iloc[0]
    assert pd.isna(unknown["understat_name"])  # unmapped clubs stay NA, never guessed


def test_team_name_maps_cover_every_fpl_club_2016_2025() -> None:
    fpl_names = set(crosswalk.FOOTBALLDATA_TEAM_NAMES)
    assert fpl_names == set(crosswalk.UNDERSTAT_TEAM_NAMES)
    assert fpl_names <= set(crosswalk.CLUBELO_TEAM_NAMES)
    assert len(fpl_names) == 34  # distinct PL clubs across 2016-17..2025-26


# --- synthetic processed + raw tree -------------------------------------------------

SEASON = 2024
KICKOFFS = ["2024-08-17T14:00:00Z", "2024-08-17T16:30:00Z", "2024-08-24T14:00:00Z"]


def _write_processed(processed: Path) -> None:
    processed.mkdir(parents=True, exist_ok=True)
    teams = pd.DataFrame(
        {
            "season": SEASON,
            "fpl_team_id": [1, 2],
            "team_code": [3, 43],
            "name": pd.array(["Arsenal", "Man City"], dtype="string"),
            "short_name": pd.array(["ARS", "MCI"], dtype="string"),
            "understat_name": pd.array([pd.NA, pd.NA], dtype="string"),
            "clubelo_name": pd.array([pd.NA, pd.NA], dtype="string"),
            "footballdata_name": pd.array([pd.NA, pd.NA], dtype="string"),
        }
    )
    teams.to_parquet(processed / "teams.parquet", index=False)

    players = pd.DataFrame(
        {
            "player_code": [111, 222, 333],
            "web_name": pd.array(["Saka", "Ederson M.", "Rice"], dtype="string"),
            "first_name": pd.array(["Bukayo", "Ederson", "Declan"], dtype="string"),
            "second_name": pd.array(
                ["Saka", "Santana de Moraes", "Rice"], dtype="string"
            ),
            "understat_id": pd.array([pd.NA, pd.NA, pd.NA], dtype="Int64"),
            "opta_code": pd.array([pd.NA] * 3, dtype="string"),
        }
    )
    players.to_parquet(processed / "players.parquet", index=False)

    fixtures = pd.DataFrame(
        {
            "season": SEASON,
            "gw": [1, 1, 2],
            "fpl_fixture_id": [10, 11, 20],
            "kickoff_utc": pd.to_datetime(KICKOFFS, utc=True).as_unit("ns"),
            "home_team_code": [3, 43, 43],
            "away_team_code": [43, 3, 3],
            "home_goals": pd.array([2, 1, 0], dtype="Int64"),
            "away_goals": pd.array([1, 1, 0], dtype="Int64"),
            "finished": True,
            "void": False,
        }
    )
    fixtures.to_parquet(processed / "fixtures.parquet", index=False)

    pm = pd.DataFrame(
        {
            "season": SEASON,
            "gw": [1, 1, 2],
            "fpl_fixture_id": [10, 11, 20],
            "player_code": [111, 222, 111],
            "fpl_element_id": [7, 8, 7],
            "team_code": [3, 43, 3],
            "opponent_code": [43, 3, 43],
            "was_home": [True, True, False],
            "position": pd.array(["MID", "GKP", "MID"], dtype="string"),
            "price": [90, 55, 90],
        }
    )
    for col in build.CORE_OUTCOME_COLS:
        pm[col] = 0
    pm["minutes"] = [90, 90, 88]
    for col in build.NULLABLE_INT_OUTCOME_COLS:
        pm[col] = pd.array([pd.NA] * 3, dtype="Int64")
    for col in build.XG_COLS.values():
        pm[col] = pd.array([pd.NA] * 3, dtype="Float64")
    for col in build.UNDERSTAT_COLS:
        pm[col] = pd.array([pd.NA] * 3, dtype="Float64")
    pm["empty_stadium"] = False
    pm["void_gw"] = False
    pm["subs_regime"] = 5
    pm["stint_id"] = 0
    pm[list(build.PLAYER_MATCH_COLS)].to_parquet(processed / "player_match.parquet", index=False)

    odds = pd.DataFrame(
        {
            "season": SEASON,
            # Second row is one day off the kickoff date: exercises the ±1-day pass.
            "date": pd.to_datetime(["2024-08-17", "2024-08-18", "2024-08-24"]).as_unit("ns"),
            "home_footballdata_name": pd.array(
                ["Arsenal", "Man City", "Man City"], dtype="string"
            ),
            "away_footballdata_name": pd.array(
                ["Man City", "Arsenal", "Arsenal"], dtype="string"
            ),
            "odds_h": pd.array([2.0, 2.1, 2.2], dtype="Float64"),
            "odds_d": pd.array([3.4, 3.5, 3.6], dtype="Float64"),
            "odds_a": pd.array([3.6, 3.4, 3.2], dtype="Float64"),
            "odds_over25": pd.array([1.8, 1.9, 2.0], dtype="Float64"),
            "odds_under25": pd.array([2.0, 1.9, 1.8], dtype="Float64"),
            "fpl_fixture_id": pd.array([pd.NA] * 3, dtype="Int64"),
        }
    )
    odds.to_parquet(processed / "odds.parquet", index=False)


def _write_raw(raw: Path) -> None:
    us_dir = raw / "understat" / str(SEASON)
    us_dir.mkdir(parents=True, exist_ok=True)
    league = {
        "players": [
            {"id": "5555", "player_name": "Bukayo Saka", "team_title": "Arsenal",
             "time": "2500"},
            {"id": "6054", "player_name": "Ederson", "team_title": "Manchester City",
             "time": "2400"},
        ],
        "teams": {},
        "dates": [],
    }
    (us_dir / "league.json").write_text(json.dumps(league), encoding="utf-8")

    # vaastav-style per-player CSV for Saka: one row per match over his career;
    # the 2023-season row and the non-PL row must both be filtered out.
    va_dir = raw / "vaastav" / "2024-25" / "understat"
    va_dir.mkdir(parents=True, exist_ok=True)
    saka = pd.DataFrame(
        {
            "goals": [1, 0, 2],
            "shots": [4, 3, 5],
            "xG": [0.7, 0.4, 1.2],
            "time": [90, 88, 90],
            "position": ["AMR", "AMR", "AMR"],
            "h_team": ["Arsenal", "Bayern Munich", "Manchester City"],
            "a_team": ["Manchester City", "Arsenal", "Arsenal"],
            "h_goals": [2, 1, 0],
            "a_goals": [1, 1, 0],
            "date": ["2024-08-17", "2024-08-20", "2024-08-24"],
            "id": [30001, 39999, 30002],
            "season": [2024, 2024, 2024],
            "roster_id": [1, 2, 3],
            "xA": [0.3, 0.1, 0.5],
            "assists": [0, 0, 1],
            "key_passes": [2, 1, 3],
            "npg": [1, 0, 2],
            "npxG": [0.7, 0.4, 0.9],
            "xGChain": [1.0, 0.5, 1.5],
            "xGBuildup": [0.2, 0.1, 0.3],
        }
    )
    saka.to_csv(va_dir / "Bukayo_Saka_5555.csv", index=False)
    # Ederson comes from the JSON-endpoint parquet fallback instead.
    ederson = pd.DataFrame(
        {
            "season": [SEASON],
            "match_id": [30003],
            "understat_id": [6054],
            "date": pd.to_datetime(["2024-08-17"]).as_unit("ns"),
            "h_team": ["Manchester City"],
            "a_team": ["Arsenal"],
            "us_xg": [0.0],
            "us_xa": [0.1],
            "us_npxg": [0.0],
            "us_shots": [0],
            "us_key_passes": [1],
            "minutes": [90],
        }
    )
    ederson.to_parquet(us_dir / "player_matches.parquet", index=False)


@pytest.fixture()
def synth_dirs(tmp_path: Path) -> tuple[Path, Path]:
    processed, raw = tmp_path / "processed", tmp_path / "raw"
    _write_processed(processed)
    _write_raw(raw)
    return processed, raw


# --- enrich_all ---------------------------------------------------------------------


def test_enrich_all_fills_everything(synth_dirs: tuple[Path, Path]) -> None:
    processed, raw = synth_dirs
    stats = enrich.enrich_all(processed_dir=processed, raw_root=raw)

    teams = pd.read_parquet(processed / "teams.parquet")
    assert teams["understat_name"].tolist() == ["Arsenal", "Manchester City"]
    assert teams["footballdata_name"].notna().all()
    assert teams["clubelo_name"].notna().all()

    players = pd.read_parquet(processed / "players.parquet")
    by_code = players.set_index("player_code")["understat_id"]
    assert by_code[111] == 5555  # exact name match
    assert by_code[222] == 6054  # Ederson mononym via containment rule
    assert pd.isna(by_code[333])  # Rice not in the understat list

    pm = pd.read_parquet(processed / "player_match.parquet")
    assert list(pm.columns) == list(build.PLAYER_MATCH_COLS)
    saka_gw1 = pm[(pm["player_code"] == 111) & (pm["gw"] == 1)].iloc[0]
    assert saka_gw1["us_xg"] == pytest.approx(0.7)
    assert saka_gw1["us_key_passes"] == 2
    saka_gw2 = pm[(pm["player_code"] == 111) & (pm["gw"] == 2)].iloc[0]
    assert saka_gw2["us_xg"] == pytest.approx(1.2)  # PL row kept, Bayern row dropped
    ederson_row = pm[pm["player_code"] == 222].iloc[0]
    assert ederson_row["us_xa"] == pytest.approx(0.1)  # via the parquet fallback

    odds = pd.read_parquet(processed / "odds.parquet")
    assert odds["fpl_fixture_id"].tolist() == [10, 11, 20]  # incl. the ±1-day row
    assert stats["odds"]["odds_join_rate_by_season"][SEASON] == 1.0
    assert stats["understat_join"]["us_coverage_by_season"][SEASON] == 1.0

    report = processed / "unmatched_understat.csv"
    assert report.exists()


def test_enrich_all_is_idempotent(synth_dirs: tuple[Path, Path]) -> None:
    processed, raw = synth_dirs
    enrich.enrich_all(processed_dir=processed, raw_root=raw)
    first = {
        p.name: p.read_bytes()
        for p in sorted(processed.iterdir())
        if p.suffix in {".parquet", ".csv"}
    }
    stats = enrich.enrich_all(processed_dir=processed, raw_root=raw)
    second = {
        p.name: p.read_bytes()
        for p in sorted(processed.iterdir())
        if p.suffix in {".parquet", ".csv"}
    }
    assert first.keys() == second.keys()
    for name in first:
        assert first[name] == second[name], f"{name} changed on re-run"
    assert stats["understat_join"]["us_coverage_by_season"][SEASON] == 1.0


def test_enrich_all_tolerates_empty_raw(tmp_path: Path) -> None:
    processed, raw = tmp_path / "processed", tmp_path / "raw"
    _write_processed(processed)
    raw.mkdir()
    stats = enrich.enrich_all(processed_dir=processed, raw_root=raw)
    pm = pd.read_parquet(processed / "player_match.parquet")
    assert pm["us_xg"].isna().all()  # no understat raw -> untouched placeholders
    assert stats["understat_join"]["seasons_with_data"] == []
    odds = pd.read_parquet(processed / "odds.parquet")
    assert odds["fpl_fixture_id"].notna().all()  # odds join needs no understat data


# --- vaastav per-player CSV downloader ----------------------------------------------


def test_download_vaastav_understat_players_offline(tmp_path: Path) -> None:
    us_dir = tmp_path / "vaastav" / "2021-22" / "understat"
    us_dir.mkdir(parents=True)
    pd.DataFrame(
        {"id": [751, 453], "player_name": ["N&#039;Golo Kanté", "Son Heung-Min"]}
    ).to_csv(us_dir / "understat_player.csv", index=False)
    (us_dir / "Son_Heung-Min_453.csv").write_text("already here", encoding="utf-8")
    # players_raw.csv so resolve_seasons discovers the season dir.
    (tmp_path / "vaastav" / "2021-22" / "players_raw.csv").write_text("id,code\n",
                                                                      encoding="utf-8")

    requested: list[str] = []

    def fake_fetch(url: str) -> bytes:
        requested.append(url)
        return b"goals,shots\n1,2\n"

    got = enrich.download_vaastav_understat_players(
        [2021], raw_root=tmp_path, fetch=fake_fetch
    )
    assert got == {2021: 1}  # Son skipped (exists), Kanté fetched
    assert len(requested) == 1
    # HTML entity and accent must survive into the URL, percent-encoded.
    assert "N%26%23039%3BGolo_Kant%C3%A9_751.csv" in requested[0]
    assert (us_dir / "N&#039;Golo_Kanté_751.csv").read_bytes() == b"goals,shots\n1,2\n"

    def missing_fetch(url: str) -> bytes:
        raise FileNotFoundError(url)

    (us_dir / "N&#039;Golo_Kanté_751.csv").unlink()
    got = enrich.download_vaastav_understat_players(
        [2021], raw_root=tmp_path, fetch=missing_fetch
    )
    assert got == {2021: 0}  # 404s skipped, not raised


# --- odds join specifics ------------------------------------------------------------


def test_join_odds_synthetic_e0_names_and_dates(synth_dirs: tuple[Path, Path]) -> None:
    processed, _ = synth_dirs
    teams = crosswalk.apply_team_name_maps(pd.read_parquet(processed / "teams.parquet"))
    fixtures = pd.read_parquet(processed / "fixtures.parquet")
    odds = pd.read_parquet(processed / "odds.parquet")
    joined, stats = enrich.join_odds(odds, fixtures, teams)
    assert joined["fpl_fixture_id"].tolist() == [10, 11, 20]
    assert stats["odds_join_rate_by_season"] == {SEASON: 1.0}

    # An unknown club name must not join (and must not raise).
    bad = odds.copy()
    bad.loc[0, "home_footballdata_name"] = "Narnia Rovers"
    joined_bad, _ = enrich.join_odds(bad, fixtures, teams)
    assert pd.isna(joined_bad.loc[0, "fpl_fixture_id"])
    assert joined_bad.loc[1:, "fpl_fixture_id"].notna().all()


def test_join_odds_reports_only_seasons_with_fixtures(
    synth_dirs: tuple[Path, Path]
) -> None:
    processed, _ = synth_dirs
    teams = crosswalk.apply_team_name_maps(pd.read_parquet(processed / "teams.parquet"))
    fixtures = pd.read_parquet(processed / "fixtures.parquet")
    odds = pd.read_parquet(processed / "odds.parquet")
    other = odds.copy()
    other["season"] = 1999  # odds season with no fixtures on disk
    both = pd.concat([odds, other], ignore_index=True)
    joined, stats = enrich.join_odds(both, fixtures, teams)
    assert set(stats["odds_join_rate_by_season"]) == {SEASON}
    assert joined.loc[joined["season"] == 1999, "fpl_fixture_id"].isna().all()
