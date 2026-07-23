"""Offline tests for the bootstrap-grounded live layer (fplai.data.live).

Fixtures under ``tests/fixtures/live_snapshot_2026/`` are a trimmed copy of the
REAL day-1 2026-27 snapshot (2026-07-23): all 20 teams, GW1+partial GW2
fixtures and 22 real elements chosen to cover the interesting cases — Haaland's
launch price, two 2026-27 position reclassifications (Marmoush FWD, Wieffer
DEF), promoted-team players (Coventry/Hull/Ipswich), WC2026 fatigue names and
every availability status (a/d/i/s/u).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from fplai.data import build, live
from fplai.models.minutes import HeuristicMinutesModel

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "live_snapshot_2026"


@pytest.fixture(scope="module")
def ctx() -> live.LiveContext:
    loaded = live.load_live_context(snap_dir=FIXTURE_DIR)
    assert loaded is not None
    return loaded


# ---------------------------------------------------------------------------
# Snapshot discovery + parsing
# ---------------------------------------------------------------------------


def test_latest_live_snapshot_dir_ignores_pre_relaunch(tmp_path: Path) -> None:
    root = tmp_path / "snapshots"
    old = root / "2026-07-01"
    old.mkdir(parents=True)
    bootstrap = json.loads((FIXTURE_DIR / "bootstrap.json").read_text())
    bootstrap["game_config"]["settings"]["static_content_url"] = (
        "https://fantasy.premierleague.com/gcs/x/2025_26/"
    )
    (old / "bootstrap.json").write_text(json.dumps(bootstrap))
    (old / "fixtures.json").write_text((FIXTURE_DIR / "fixtures.json").read_text())
    assert live.latest_live_snapshot_dir(root) is None

    new = root / "2026-07-23"
    new.mkdir()
    (new / "bootstrap.json").write_text((FIXTURE_DIR / "bootstrap.json").read_text())
    (new / "fixtures.json").write_text((FIXTURE_DIR / "fixtures.json").read_text())
    assert live.latest_live_snapshot_dir(root) == new


def test_load_live_context_season_state(ctx: live.LiveContext) -> None:
    assert ctx.season == 2026
    assert ctx.next_gw == 1


def test_roster_parses_real_prices_and_reclassified_positions(ctx: live.LiveContext) -> None:
    r = ctx.roster.set_index("player_code")
    haaland = r.loc[223094]
    assert haaland["web_name"] == "Haaland"
    assert haaland["position"] == "FWD"
    assert haaland["price"] == 155  # £15.5m launch price
    assert haaland["team_code"] == 43
    # 2026-27 reclassifications come free via element_type:
    assert r.loc[438234]["position"] == "FWD"  # Marmoush, was MID
    assert r.loc[467779]["position"] == "DEF"  # Wieffer, was MID


def test_roster_includes_promoted_team_players(ctx: live.LiveContext) -> None:
    r = ctx.roster
    cov = r[r["team_code"] == 9]  # Coventry City
    assert set(cov["web_name"]) >= {"Wright", "Wilson", "Bidwell"}
    wright = cov[cov["web_name"] == "Wright"].iloc[0]
    assert wright["position"] == "FWD" and wright["price"] == 55
    assert (r["team_code"] == 88).any()  # Hull City
    assert (r["team_code"] == 40).any()  # Ipswich Town


def test_roster_availability_columns(ctx: live.LiveContext) -> None:
    r = ctx.roster.set_index("player_code")
    saliba = r.loc[462424]
    assert saliba["status"] == "i" and saliba["chance_of_playing"] == 0
    kamara = r.loc[226944]
    assert kamara["status"] == "d" and kamara["chance_of_playing"] == 75
    uche = r.loc[660392]
    assert uche["status"] == "u"


def test_live_fixtures_conform_to_schema(ctx: live.LiveContext) -> None:
    fx = ctx.fixtures
    assert list(fx.columns) == list(build.FIXTURES_COLS)
    assert (fx["season"] == 2026).all()
    assert set(fx["gw"].unique()) == {1, 2}
    assert not fx["finished"].any()
    assert fx["home_goals"].isna().all()
    assert str(fx["kickoff_utc"].dt.tz) == "UTC"
    opener = fx[fx["fpl_fixture_id"] == 1].iloc[0]
    # Arsenal (code 3) vs Coventry City (code 9), Fri 21 Aug 2026 19:00 UTC.
    assert opener["home_team_code"] == 3 and opener["away_team_code"] == 9
    assert opener["kickoff_utc"] == pd.Timestamp("2026-08-21T19:00:00Z")


def test_live_teams_have_aux_names_for_promoted_clubs(ctx: live.LiveContext) -> None:
    t = ctx.teams.set_index("name")
    assert len(ctx.teams) == 20
    for name, expected in (
        ("Coventry City", "Coventry"),
        ("Hull City", "Hull"),
        ("Ipswich Town", "Ipswich"),
    ):
        row = t.loc[name]
        assert row["understat_name"] == expected
        assert row["clubelo_name"] == expected
        assert row["footballdata_name"] == expected
    assert t["footballdata_name"].notna().all()


# ---------------------------------------------------------------------------
# Roster-grounded synthesis + availability frame
# ---------------------------------------------------------------------------


def _pm_template() -> pd.DataFrame:
    return pd.DataFrame(columns=list(build.PLAYER_MATCH_COLS))


def test_bootstrap_player_match_synthesizes_both_sides(ctx: live.LiveContext) -> None:
    target = ctx.fixtures[ctx.fixtures["fpl_fixture_id"] == 1]  # ARS vs COV
    synth = live.bootstrap_player_match(ctx, target, _pm_template())
    assert list(synth.columns) == list(build.PLAYER_MATCH_COLS)
    assert (synth["season"] == 2026).all() and (synth["gw"] == 1).all()
    cov = synth[synth["team_code"] == 9]
    assert set(cov["player_code"]) >= {176412, 110690, 80178}  # Wright, Wilson, Bidwell
    wright = cov[cov["player_code"] == 176412].iloc[0]
    assert wright["position"] == "FWD" and wright["price"] == 55
    assert wright["opponent_code"] == 3 and not bool(wright["was_home"])
    assert pd.isna(wright["minutes"])  # outcomes unknown
    ars = synth[synth["team_code"] == 3]
    assert 223340 in set(ars["player_code"])  # Saka at home
    assert bool(ars[ars["player_code"] == 223340].iloc[0]["was_home"])


def test_bootstrap_player_match_excludes_departed_players(ctx: live.LiveContext) -> None:
    # Uche (status "u", returned to Getafe) plays for Crystal Palace (code 31).
    target = ctx.fixtures[
        (ctx.fixtures["home_team_code"] == 31) | (ctx.fixtures["away_team_code"] == 31)
    ].head(1)
    synth = live.bootstrap_player_match(ctx, target, _pm_template())
    assert 660392 not in set(synth["player_code"])


def test_availability_frame_shape_and_values(ctx: live.LiveContext) -> None:
    avail = live.availability_frame(ctx, [1, 2])
    assert set(avail.columns) == {"season", "gw", "player_code", "status", "chance_of_playing"}
    assert len(avail) == 2 * len(ctx.roster)
    kamara = avail[(avail["player_code"] == 226944) & (avail["gw"] == 1)].iloc[0]
    assert kamara["status"] == "d" and kamara["chance_of_playing"] == 75


# ---------------------------------------------------------------------------
# Fatigue watchlist (FPL_KNOWLEDGE §3.3)
# ---------------------------------------------------------------------------


def test_match_fatigue_watchlist_on_real_names(ctx: live.LiveContext) -> None:
    damp, report = live.match_fatigue_watchlist(ctx.roster, ctx.teams)
    # Rodri's 2026-27 web_name is "Rodrigo" — the fuzzy matcher must bridge it.
    assert damp[220566] == live.FATIGUE_HIGH
    assert damp[441164] == live.FATIGUE_HIGH  # Pedro Porro
    assert damp[154561] == live.FATIGUE_LIGHT  # Raya (lighter Spain load)
    assert damp[223340] == live.FATIGUE_HIGH  # Saka
    rodri = next(m for m in report["matched"] if m["name"] == "Rodri")
    assert rodri["web_name"] == "Rodrigo" and rodri["player_code"] == 220566
    # The trimmed roster misses most watchlist names — they must be *reported*,
    # never silently dropped or spuriously matched.
    assert "Watkins" in report["unmatched"]
    assert report["n_watchlist"] == len(live.WC2026_FATIGUE_WATCHLIST)


def test_watchlist_does_not_match_across_clubs(ctx: live.LiveContext) -> None:
    damp, _ = live.match_fatigue_watchlist(ctx.roster, ctx.teams)
    # McBurnie (Hull) must not be claimed by the "Burn" (NEW) entry.
    assert 169432 not in damp


# ---------------------------------------------------------------------------
# Cold-start priors
# ---------------------------------------------------------------------------


def _history_pm() -> pd.DataFrame:
    """Synthetic played history: cheap FWDs never score, premium FWDs always do."""
    n = 200
    rows = {
        "season": [2024] * (2 * n) + [2024] * n,
        "gw": list(range(1, 2 * n + 1)) + list(range(1, n + 1)),
        "player_code": [1] * n + [2] * n + [3] * n,
        "position": ["FWD"] * n + ["FWD"] * n + ["GKP"] * n,
        "price": [45] * n + [120] * n + [50] * n,
        "minutes": [90] * (3 * n),
        "goals_scored": [0] * n + [1] * n + [0] * n,
        "assists": [0] * (3 * n),
        "saves": [0] * (2 * n) + [3] * n,
        "defensive_contribution": [2] * (2 * n) + [0] * n,
    }
    return pd.DataFrame(rows)


def test_rate_priors_split_by_position_and_price(loaded_priors: None = None) -> None:
    priors = live.RatePriors(min_season=2024).fit(_history_pm())
    out = priors.lookup(
        np.array(["FWD", "FWD", "GKP"]), np.array([45.0, 120.0, 50.0])
    )
    assert out["lam_goal"][1] > out["lam_goal"][0]  # premium FWD >> budget FWD
    assert out["lam_goal"][1] > 0.5
    assert out["lam_saves"][2] > 2.0  # GKP saves prior from history
    assert out["lam_saves"][0] == 0.0  # outfield rows never get a saves rate
    assert out["lam_defcon"][2] == 0.0  # GKP DefCon is structurally 0


def _adjust(
    *,
    fatigue: dict[int, float] | None = None,
    cold: frozenset[int] = frozenset(),
) -> live.LiveAdjustments:
    priors = live.RatePriors(min_season=2024).fit(_history_pm())
    return live.LiveAdjustments(
        season=2026,
        first_gw=1,
        fatigue=fatigue or {},
        cold_codes=cold,
        rate_priors=priors,
    )


def _features(rows: list[dict[str, Any]]) -> pd.DataFrame:
    base = {
        "season": 2026,
        "gw": 1,
        "fpl_fixture_id": 1,
        "player_code": 0,
        "position": "MID",
        "price": 50,
        "f_price": 50.0,
        "f_status_i": np.nan,
        "f_status_s": np.nan,
        "f_status_u": np.nan,
        "f_status_d": np.nan,
        "f_chance_of_playing": np.nan,
        "minutes": np.nan,
    }
    return pd.DataFrame([{**base, **r} for r in rows])


def test_live_minutes_cold_start_uses_position_price_band_priors() -> None:
    feats = _features(
        [
            {"player_code": 10, "position": "GKP", "price": 40, "f_price": 40.0},
            {"player_code": 11, "position": "MID", "price": 85, "f_price": 85.0},
        ]
    )
    adj = _adjust(cold=frozenset({10, 11}))
    pred = live.live_minutes_predict(HeuristicMinutesModel(), feats, adj)
    gkp = pred[pred["player_code"] == 10].iloc[0]
    mid = pred[pred["player_code"] == 11].iloc[0]
    # DEFAULT_SHARES: a £4.0m GKP barely starts; a £8.5m MID mostly does.
    assert gkp["q2"] < 0.2
    assert mid["q2"] > 0.4
    assert abs(gkp[["q0", "q1", "q2"]].sum() - 1.0) < 1e-9


def test_live_minutes_fatigue_dampens_gw1_to_4_only() -> None:
    feats = _features(
        [
            {"player_code": 20, "gw": 1},
            {"player_code": 20, "gw": 5, "fpl_fixture_id": 50},
        ]
    )
    adj = _adjust(fatigue={20: 0.5})
    pred = live.live_minutes_predict(HeuristicMinutesModel(), feats, adj)
    gw1 = pred[pred["gw"] == 1].iloc[0]
    gw5 = pred[pred["gw"] == 5].iloc[0]
    assert gw1["q2"] < 0.6 * gw5["q2"]  # dampened in the window
    assert gw1["q1"] > gw5["q1"]  # removed mass shifts mostly to cameos


def test_live_minutes_availability_gating_recovers_over_gws() -> None:
    rows = [
        {"player_code": 30, "gw": g, "fpl_fixture_id": g, "f_status_i": 1.0,
         "f_chance_of_playing": 0.0}
        for g in (1, 3, 5, 9)
    ]
    feats = _features(rows)
    adj = _adjust()
    pred = live.live_minutes_predict(HeuristicMinutesModel(), feats, adj).set_index("gw")
    assert pred.loc[1, "q0"] > 0.95  # flagged out at the deadline
    assert pred.loc[3, "q0"] < pred.loc[1, "q0"]  # recovering
    assert pred.loc[9, "q0"] == pytest.approx(pred.loc[5, "q0"], abs=1e-9)  # fully recovered


def test_apply_cold_start_rates_overrides_only_cold_rows() -> None:
    feats = _features(
        [
            {"player_code": 40, "position": "FWD", "price": 120},
            {"player_code": 41, "position": "FWD", "price": 120},
        ]
    )
    rates = feats[["season", "gw", "player_code", "fpl_fixture_id"]].copy()
    for col in ("lam_goal", "lam_assist", "lam_saves", "lam_defcon"):
        rates[col] = 0.123
    adj = _adjust(cold=frozenset({40}))
    out = live.apply_cold_start_rates(rates, feats, adj)
    cold = out[out["player_code"] == 40].iloc[0]
    warm = out[out["player_code"] == 41].iloc[0]
    assert cold["lam_goal"] > 0.5  # premium-FWD decile prior
    assert warm["lam_goal"] == 0.123  # untouched


# ---------------------------------------------------------------------------
# Team model seeding (§3.5 / §3.6)
# ---------------------------------------------------------------------------


def test_seed_team_model_promoted_and_new_manager(ctx: live.LiveContext) -> None:
    from fplai.models.team import TeamModel

    model = TeamModel()
    # Hand-built fitted state: Arsenal 3, Man City 43, Hull 88 (stale 2016-17-era
    # rating), Liverpool 14.
    model.team_codes_ = [3, 14, 43, 88]
    model.log_a_ = np.array([0.5, 0.4, 0.6, -0.8])
    model.log_b_ = np.array([-0.2, -0.1, -0.3, 0.6])
    model.promoted_prior_ = (-0.3, 0.3)
    report = live.seed_team_model(model, ctx)

    # Hull's stale rating is dropped -> routes through the promoted prior hook.
    assert 88 not in model.team_codes_
    assert model.promoted_strength[88] == live.PROMOTED_STRENGTH_2026["Hull City"]
    assert model.promoted_strength[9] == live.PROMOTED_STRENGTH_2026["Coventry City"]
    assert model.promoted_strength[40] == live.PROMOTED_STRENGTH_2026["Ipswich Town"]
    hull = next(p for p in report["promoted"] if p["name"] == "Hull City")
    assert hull["dropped_stale_rating"] is True

    # Man City (new manager) shrinks toward the mean; Arsenal (continuity) not.
    i_mci = model.team_codes_.index(43)
    i_ars = model.team_codes_.index(3)
    assert model.log_a_[i_mci] == pytest.approx(0.6 * (1 - live.NEW_MANAGER_SHRINK))
    assert model.log_a_[i_ars] == pytest.approx(0.5)
    # Ratings-for path: promoted codes now resolve via prior + multiplier.
    la, lb = model._ratings_for(pd.Series([9, 88]))
    shift_cov = np.log(live.PROMOTED_STRENGTH_2026["Coventry City"])
    assert la[0] == pytest.approx(-0.3 + shift_cov)
    assert lb[0] == pytest.approx(0.3 - shift_cov)
    assert la[1] < la[0]  # Hull seeded weaker than Coventry


# ---------------------------------------------------------------------------
# Live odds
# ---------------------------------------------------------------------------


def _sample_events() -> list[dict[str, Any]]:
    return json.loads(
        (Path(__file__).parent / "fixtures" / "odds_api_epl_sample.json").read_text()
    )


def test_odds_frame_from_events_maps_names_and_medians(ctx: live.LiveContext) -> None:
    frame = live.odds_frame_from_events(_sample_events(), ctx)
    assert len(frame) >= 1
    row = frame[frame["home_footballdata_name"] == "Arsenal"].iloc[0]
    assert row["away_footballdata_name"] == "Chelsea"
    assert row["season"] == 2026
    assert row["odds_h"] == pytest.approx(np.median([1.7, 1.73]))
    assert row["odds_over25"] == pytest.approx(1.83)


def test_odds_frame_resolves_odds_api_long_names(ctx: live.LiveContext) -> None:
    events = [
        {
            "home_team": "Manchester City",
            "away_team": "Coventry City",
            "commence_time": "2026-08-23T15:00:00Z",
            "bookmakers": [
                {
                    "markets": [
                        {
                            "key": "h2h",
                            "outcomes": [
                                {"name": "Manchester City", "price": 1.2},
                                {"name": "Coventry City", "price": 12.0},
                                {"name": "Draw", "price": 7.0},
                            ],
                        }
                    ]
                }
            ],
        }
    ]
    frame = live.odds_frame_from_events(events, ctx)
    assert frame.iloc[0]["home_footballdata_name"] == "Man City"
    assert frame.iloc[0]["away_footballdata_name"] == "Coventry City"


def test_fetch_live_odds_prefers_disk_cache(ctx: live.LiveContext, tmp_path: Path) -> None:
    import datetime as dt

    today = dt.datetime.now(dt.UTC).date().isoformat()
    (tmp_path / f"epl_odds_{today}.json").write_text(json.dumps(_sample_events()))
    frame = live.fetch_live_odds(ctx, raw_dir=tmp_path, use_network=False)
    assert frame is not None and len(frame) >= 1
    # No cache + no network -> graceful None.
    assert live.fetch_live_odds(ctx, raw_dir=tmp_path / "empty", use_network=False) is None


# ---------------------------------------------------------------------------
# Table augmentation + persistence + adjustments bundle
# ---------------------------------------------------------------------------


def _mini_tables() -> dict[str, pd.DataFrame]:
    fixtures = pd.DataFrame(
        {
            "season": [2025],
            "gw": pd.array([38], dtype="Int64"),
            "fpl_fixture_id": [380],
            "kickoff_utc": pd.to_datetime(["2026-05-24T15:00:00Z"]).as_unit("ns"),
            "home_team_code": [3],
            "away_team_code": [43],
            "home_goals": pd.array([1], dtype="Int64"),
            "away_goals": pd.array([1], dtype="Int64"),
            "finished": [True],
            "void": [False],
        }
    )
    teams = pd.DataFrame(
        {
            "season": [2025],
            "fpl_team_id": [1],
            "team_code": [3],
            "name": pd.array(["Arsenal"], dtype="string"),
            "short_name": pd.array(["ARS"], dtype="string"),
            "understat_name": pd.array(["Arsenal"], dtype="string"),
            "clubelo_name": pd.array(["Arsenal"], dtype="string"),
            "footballdata_name": pd.array(["Arsenal"], dtype="string"),
        }
    )
    players = pd.DataFrame(
        {
            "player_code": [223340],
            "web_name": pd.array(["Saka"], dtype="string"),
            "first_name": pd.array(["Bukayo"], dtype="string"),
            "second_name": pd.array(["Saka"], dtype="string"),
            "understat_id": pd.array([7322], dtype="Int64"),
            "opta_code": pd.array(["p223340"], dtype="string"),
        }
    )
    return {"fixtures": fixtures, "teams": teams, "players": players}


def test_augment_tables_splices_live_season(ctx: live.LiveContext) -> None:
    tables = _mini_tables()
    out = live.augment_tables(tables, ctx)
    assert set(out["fixtures"]["season"].unique()) == {2025, 2026}
    assert len(out["fixtures"]) == 1 + len(ctx.fixtures)
    assert len(out["teams"][out["teams"]["season"] == 2026]) == 20
    # New roster identities appended; existing rows keep their understat_id.
    players = out["players"].set_index("player_code")
    assert 176412 in players.index  # Haji Wright (new code)
    assert players.loc[223340, "understat_id"] == 7322
    # Idempotent: augmenting the augmented tables changes nothing.
    again = live.augment_tables(out, ctx)
    assert len(again["fixtures"]) == len(out["fixtures"])
    assert len(again["players"]) == len(out["players"])


def test_persist_live_tables_writes_roster_snapshot(
    ctx: live.LiveContext, tmp_path: Path
) -> None:
    tables = live.augment_tables(_mini_tables(), ctx)
    paths = live.persist_live_tables(ctx, tables, processed_dir=tmp_path)
    assert set(paths) == {"fixtures", "teams", "players", "live_roster"}
    roster = pd.read_parquet(paths["live_roster"])
    assert (roster["season"] == 2026).all()
    assert set(roster.columns) >= {"player_code", "price", "position", "team_code", "status"}
    fx = pd.read_parquet(paths["fixtures"])
    assert (fx["season"] == 2026).sum() == len(ctx.fixtures)


def test_build_adjustments_cold_codes_and_fatigue_report(
    ctx: live.LiveContext, tmp_path: Path
) -> None:
    hist = _history_pm()
    hist.loc[hist.index[:5], "player_code"] = 223340  # Saka has history
    adj = live.build_adjustments(ctx, hist, processed_dir=tmp_path)
    assert 223340 not in adj.cold_codes
    assert 176412 in adj.cold_codes  # Haji Wright: promoted, no PL history
    assert adj.first_gw == 1 and adj.season == 2026
    payload = json.loads((tmp_path / "fatigue_2026.json").read_text())
    assert payload["gws"] == [1, 2, 3, 4]
    assert payload["dampening"][str(220566)] == live.FATIGUE_HIGH  # Rodri
    assert "unmatched" in payload and "matched" in payload


# ---------------------------------------------------------------------------
# Optimizer price source (live roster wins over stale player_gw)
# ---------------------------------------------------------------------------


def test_optimizer_prices_prefer_live_roster_over_player_gw(tmp_path: Path) -> None:
    from fplai import pipeline

    xp = pd.DataFrame(
        {
            "season": [2026, 2026],
            "gw": [1, 1],
            "player_code": [223094, 176412],
            "xp": [8.0, 4.0],
        }
    )
    # Stale 2025 context: Haaland at an old price; Wright absent entirely.
    pd.DataFrame(
        {
            "season": [2025],
            "gw": [38],
            "player_code": [223094],
            "team_code": [43],
            "position": ["FWD"],
            "value": [147],
        }
    ).to_parquet(tmp_path / "player_gw.parquet", index=False)
    pd.DataFrame(
        {
            "season": [2026, 2026],
            "player_code": [223094, 176412],
            "price": [155, 55],
            "position": ["FWD", "FWD"],
            "team_code": [43, 9],
            "web_name": ["Haaland", "Wright"],
        }
    ).to_parquet(tmp_path / "live_roster.parquet", index=False)

    prices = pipeline._optimizer_prices(xp, tmp_path, 2026, 1).set_index("player_code")
    assert prices.loc[223094, "price"] == 155  # live launch price, not 2025's 147
    assert prices.loc[176412, "price"] == 55  # promoted player resolvable at all
    assert prices.loc[176412, "team_code"] == 9
