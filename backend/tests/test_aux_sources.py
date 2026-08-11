"""Tests for the auxiliary data sources: football-data.co.uk, ClubElo, the-odds-api.

Offline tests run against trimmed real payloads in tests/fixtures/ and never
touch the network. Live tests are marked ``@pytest.mark.live`` and excluded by
default (run with ``-m live``); they install a minimal ``polite_get`` shim if
the shared ``fplai.data.fpl_api`` helper has not been implemented yet.
"""

from __future__ import annotations

import json
import sys
import time
import types
from pathlib import Path

import pandas as pd
import pytest

from fplai.config import settings
from fplai.data import clubelo, football_data, odds_api

FIXTURES = Path(__file__).parent / "fixtures"

E0_2425 = (FIXTURES / "E0_2425_sample.csv").read_bytes()
E0_1011 = (FIXTURES / "E0_1011_sample.csv").read_bytes()
CLUBELO_HISTORY = (FIXTURES / "clubelo_history_sample.csv").read_bytes()
CLUBELO_SNAPSHOT = (FIXTURES / "clubelo_snapshot_sample.csv").read_bytes()
ODDS_API_EPL = (FIXTURES / "odds_api_epl_sample.json").read_bytes()


# ---------------------------------------------------------------------------
# football_data
# ---------------------------------------------------------------------------


def _stage_e0(raw_dir: Path, season: int, payload: bytes) -> None:
    dest = football_data.e0_path(season, raw_dir)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(payload)


class TestFootballData:
    def test_season_code(self) -> None:
        assert football_data.season_code(2025) == "2526"
        assert football_data.season_code(1993) == "9394"
        assert football_data.season_code(1999) == "9900"
        assert football_data.season_code(2009) == "0910"

    def test_e0_url(self) -> None:
        assert football_data.e0_url(2025) == "https://www.football-data.co.uk/mmz4281/2526/E0.csv"

    def test_season_validation(self) -> None:
        with pytest.raises(ValueError):
            football_data.season_code(1992)
        with pytest.raises(ValueError):
            football_data.download_e0(2027)
        with pytest.raises(ValueError):
            football_data.build_odds_table([2016, 1888])

    def test_download_e0_writes_and_is_idempotent(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[str] = []

        def fake_fetch(url: str, **_: object) -> bytes:
            calls.append(url)
            return E0_2425

        monkeypatch.setattr(football_data, "_fetch", fake_fetch)
        dest = football_data.download_e0(2024, raw_dir=tmp_path)
        assert dest == tmp_path / "2024" / "E0.csv"
        assert dest.read_bytes() == E0_2425
        assert calls == ["https://www.football-data.co.uk/mmz4281/2425/E0.csv"]
        # Second call must not re-fetch.
        football_data.download_e0(2024, raw_dir=tmp_path)
        assert len(calls) == 1

    def test_download_e0_rejects_garbage(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(football_data, "_fetch", lambda url, **_: b"<html>404</html>")
        with pytest.raises(ValueError, match="unexpected E0.csv payload"):
            football_data.download_e0(2024, raw_dir=tmp_path)

    def test_download_e0_rejects_redirected_wrong_division(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Pre-publication, football-data 301s E0.csv to another division's file
        # (observed live 2026-08-11: 2627/E0.csv -> EC.csv). Same "Div," header,
        # wrong rows — must raise, never be written to disk as E0.
        ec = b"Div,Date,HomeTeam,AwayTeam,FTHG,FTAG\nEC,08/08/2026,Foo,Bar,1,0\n"
        monkeypatch.setattr(football_data, "_fetch", lambda url, **_: ec)
        with pytest.raises(ValueError, match="not division E0"):
            football_data.download_e0(2026, raw_dir=tmp_path)
        assert not (tmp_path / "2026" / "E0.csv").exists()

    def test_parse_odds_modern_closing_pinnacle(self, tmp_path: Path) -> None:
        _stage_e0(tmp_path, 2024, E0_2425)
        df = football_data.parse_odds(2024, raw_dir=tmp_path)
        assert list(df.columns) == list(football_data.ODDS_COLUMNS)
        row = df.iloc[0]  # Man United v Fulham, 16/08/2024
        assert row["home_footballdata_name"] == "Man United"
        assert row["away_footballdata_name"] == "Fulham"
        assert row["date"] == pd.Timestamp("2024-08-16")
        # Pinnacle closing (PSCH/PSCD/PSCA) preferred over B365CH etc.
        assert row["odds_h"] == pytest.approx(1.65)
        assert row["odds_d"] == pytest.approx(4.23)
        assert row["odds_a"] == pytest.approx(5.28)
        # Closing O/U 2.5 from PC>2.5 / PC<2.5.
        assert row["odds_over25"] == pytest.approx(1.63)
        assert row["odds_under25"] == pytest.approx(2.38)

    def test_parse_odds_2010_fallbacks(self, tmp_path: Path) -> None:
        _stage_e0(tmp_path, 2010, E0_1011)
        df = football_data.parse_odds(2010, raw_dir=tmp_path)
        row = df.iloc[0]  # Aston Villa v West Ham, 14/08/10 (dd/mm/yy format)
        assert row["date"] == pd.Timestamp("2010-08-14")
        # No Pinnacle/closing columns in 2010-11 -> B365H/B365D/B365A.
        assert row["odds_h"] == pytest.approx(2.0)
        assert row["odds_d"] == pytest.approx(3.3)
        assert row["odds_a"] == pytest.approx(4.0)
        # O/U only via Betbrain averages that season.
        assert row["odds_over25"] == pytest.approx(2.01)
        assert row["odds_under25"] == pytest.approx(1.75)

    def test_demargin_proportional(self) -> None:
        df = pd.DataFrame({"odds_h": [2.0], "odds_d": [3.6], "odds_a": [3.6]}).astype("Float64")
        out = football_data.demargin(df, ["odds_h", "odds_d", "odds_a"])
        total = 1 / 2.0 + 1 / 3.6 + 1 / 3.6
        assert out["p_h"].iloc[0] == pytest.approx((1 / 2.0) / total)
        assert out["p_d"].iloc[0] == pytest.approx((1 / 3.6) / total)
        assert (out["p_h"] + out["p_d"] + out["p_a"]).iloc[0] == pytest.approx(1.0)

    def test_build_odds_table(self, tmp_path: Path) -> None:
        raw = tmp_path / "raw"
        out = tmp_path / "odds.parquet"
        _stage_e0(raw, 2010, E0_1011)
        _stage_e0(raw, 2024, E0_2425)
        dest = football_data.build_odds_table([2010, 2024], raw_dir=raw, out_path=out)
        table = pd.read_parquet(dest)
        assert set(table["season"]) == {2010, 2024}
        for col in (*football_data.ODDS_COLUMNS, "p_h", "p_d", "p_a",
                    "p_over25", "p_under25", "fpl_fixture_id"):
            assert col in table.columns, col
        assert table["fpl_fixture_id"].isna().all()
        complete = table.dropna(subset=["p_h", "p_d", "p_a"])
        assert len(complete) == len(table)  # both sample seasons carry full 1X2
        sums = complete["p_h"] + complete["p_d"] + complete["p_a"]
        assert ((sums - 1.0).abs() < 1e-9).all()
        # Implied probabilities are strictly inside (0, 1).
        assert ((complete["p_h"] > 0) & (complete["p_h"] < 1)).all()


# ---------------------------------------------------------------------------
# clubelo
# ---------------------------------------------------------------------------


class TestClubElo:
    def test_name_mapping(self) -> None:
        assert clubelo.fpl_to_clubelo("Spurs") == "Tottenham"
        assert clubelo.fpl_to_clubelo("Nott'm Forest") == "Forest"
        assert clubelo.fpl_to_clubelo("Man Utd") == "Man United"
        assert clubelo.fpl_to_clubelo("Wolves") == "Wolves"
        assert clubelo.fpl_to_clubelo("Unknown FC") == "Unknown FC"  # passthrough

    def test_url_name(self) -> None:
        assert clubelo.clubelo_url_name("Man City") == "ManCity"
        assert clubelo.clubelo_url_name("Sheffield Weds") == "SheffieldWeds"
        assert clubelo.clubelo_url_name("Forest") == "Forest"

    def test_mapping_covers_snapshot_clubs(self) -> None:
        """Every PL club in the real 2026-07-01 snapshot is a known mapping value."""
        snap = pd.read_csv(FIXTURES / "clubelo_snapshot_sample.csv")
        known = set(clubelo.PL_CLUBELO_NAMES.values())
        assert set(snap["Club"]) <= known

    def test_team_history(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        urls: list[str] = []

        def fake_fetch(url: str, **_: object) -> bytes:
            urls.append(url)
            return CLUBELO_HISTORY

        monkeypatch.setattr(clubelo, "_fetch", fake_fetch)
        client = clubelo.EloClient(raw_dir=tmp_path)
        df = client.team_history("Man City")
        assert urls == ["http://api.clubelo.com/ManCity"]
        assert (tmp_path / "ManCity.csv").read_bytes() == CLUBELO_HISTORY
        assert (df["Club"] == "Man City").all()
        assert df["Elo"].dtype == "float64"
        assert df["From"].iloc[0] == pd.Timestamp("1946-07-07")
        assert df["Rank"].isna().all()  # 'None' strings -> NA (nullable Int64)
        assert str(df["Rank"].dtype) == "Int64"

    def test_snapshot_and_pl_filter(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(clubelo, "_fetch", lambda url, **_: CLUBELO_SNAPSHOT)
        client = clubelo.EloClient(raw_dir=tmp_path)
        snap = client.snapshot("2026-07-01")
        assert (tmp_path / "2026-07-01.csv").exists()
        assert len(snap) == 20
        pl = client.pl_snapshot("2026-07-01")
        assert len(pl) == 20
        assert {"Man City", "Forest", "Tottenham", "Sunderland"} <= set(pl["Club"])
        assert pl["Elo"].between(1500, 2200).all()

    def test_snapshot_rejects_bad_date(self) -> None:
        with pytest.raises(ValueError):
            clubelo.EloClient().snapshot("01/07/2026")

    def test_bad_csv_raises(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(clubelo, "_fetch", lambda url, **_: b"nope,nope\n1,2\n")
        with pytest.raises(ValueError, match="missing columns"):
            clubelo.EloClient(raw_dir=tmp_path).team_history("Arsenal")


# ---------------------------------------------------------------------------
# odds_api
# ---------------------------------------------------------------------------


class TestOddsApi:
    def test_disabled_returns_none(self) -> None:
        client = odds_api.TheOddsApiClient(api_key="")
        assert not client.enabled
        assert client.sports() is None
        assert client.epl_odds() is None

    def test_low_level_get_without_key_raises(self) -> None:
        client = odds_api.TheOddsApiClient(api_key="")
        with pytest.raises(odds_api.ConfigurationError):
            client._get("sports", {}, estimated_cost=0.0)

    def test_estimate_cost(self) -> None:
        assert odds_api.estimate_cost("h2h,totals", "uk") == 2
        assert odds_api.estimate_cost("h2h", "uk") == 1
        assert odds_api.estimate_cost("h2h,totals", "uk,eu,us") == 6

    def test_epl_odds_parses_and_tracks_quota(self, monkeypatch: pytest.MonkeyPatch) -> None:
        urls: list[str] = []

        def fake_fetch(url: str, **_: object) -> tuple[bytes, dict[str, str]]:
            urls.append(url)
            headers = {
                "x-requests-remaining": "496",
                "x-requests-used": "4",
                "x-requests-last": "2",
            }
            return ODDS_API_EPL, headers

        monkeypatch.setattr(odds_api, "_fetch", fake_fetch)
        client = odds_api.TheOddsApiClient(api_key="SECRET")
        events = client.epl_odds()
        assert events is not None and len(events) == 2
        assert events[0]["home_team"] == "Arsenal"
        markets = {m["key"] for b in events[0]["bookmakers"] for m in b["markets"]}
        assert markets == {"h2h", "totals"}
        (url,) = urls
        assert url.startswith("https://api.the-odds-api.com/v4/sports/soccer_epl/odds?")
        assert "apiKey=SECRET" in url
        assert "regions=uk" in url
        assert "markets=h2h%2Ctotals" in url
        assert client.quota.remaining == 496.0
        assert client.quota.used == 4.0
        assert client.quota.last_cost == 2.0
        assert client.quota.spent_estimate == 2.0

    def test_sports_free_and_headerless_cache(self, monkeypatch: pytest.MonkeyPatch) -> None:
        payload = json.dumps(
            [{"key": "soccer_epl", "group": "Soccer", "title": "EPL", "active": True}]
        ).encode()
        # Bytes-only return (e.g. served from the shared helper's disk cache):
        # body still parses, quota headers simply unavailable.
        monkeypatch.setattr(odds_api, "_fetch", lambda url, **_: (payload, None))
        client = odds_api.TheOddsApiClient(api_key="SECRET")
        sports = client.sports()
        assert sports is not None and sports[0]["key"] == "soccer_epl"
        assert client.quota.remaining is None
        assert client.quota.spent_estimate == 0.0  # /sports is free


# ---------------------------------------------------------------------------
# live tests (excluded by default; run with -m live)
# ---------------------------------------------------------------------------

_BROWSER_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
)


def _ensure_polite_get() -> None:
    """Install a minimal polite_get shim if fplai.data.fpl_api is not built yet.

    The real module is owned by another agent; its contract is
    ``polite_get(url, *, min_interval_s, cache_ttl_s)``. The shim honours the
    throttle and returns an httpx.Response, matching what our ``_fetch``
    normalisers accept.
    """
    try:
        from fplai.data import fpl_api  # noqa: F401

        if hasattr(fpl_api, "polite_get"):
            return
    except Exception:
        pass

    import httpx

    shim = types.ModuleType("fplai.data.fpl_api")
    state = {"t": 0.0}

    def polite_get(url: str, *, min_interval_s: float = 1.0, cache_ttl_s: float = 0.0):
        wait = state["t"] + min_interval_s - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        resp = httpx.get(
            url, headers={"User-Agent": _BROWSER_UA}, timeout=30.0, follow_redirects=True
        )
        state["t"] = time.monotonic()
        resp.raise_for_status()
        return resp

    shim.polite_get = polite_get  # type: ignore[attr-defined]
    sys.modules["fplai.data.fpl_api"] = shim


@pytest.mark.live
class TestLive:
    def test_football_data_2024_download_and_parse(self, tmp_path: Path) -> None:
        _ensure_polite_get()
        football_data.download_e0(2024, raw_dir=tmp_path)
        df = football_data.parse_odds(2024, raw_dir=tmp_path)
        assert len(df) == 380
        assert df["odds_h"].notna().all()
        assert df["odds_over25"].notna().sum() >= 370
        assert df["date"].min() >= pd.Timestamp("2024-08-01")
        assert df["date"].max() <= pd.Timestamp("2025-06-01")
        assert "Man United" in set(df["home_footballdata_name"])

    def test_football_data_1993_exists(self) -> None:
        """The 1993..2016 archive exists (single small probe, no bulk download)."""
        _ensure_polite_get()
        payload = football_data._fetch(football_data.e0_url(1993))
        assert payload.lstrip(b"\xef\xbb\xbf").startswith(b"Div,Date")
        assert b"Arsenal" in payload

    def test_clubelo_snapshot(self, tmp_path: Path) -> None:
        _ensure_polite_get()
        client = clubelo.EloClient(raw_dir=tmp_path)
        pl = client.pl_snapshot("2026-07-01")
        assert len(pl) >= 18
        assert {"Man City", "Forest", "Liverpool", "Arsenal"} <= set(pl["Club"])
        assert pl["Elo"].between(1400, 2300).all()

    def test_odds_api_sports(self) -> None:
        if not settings.odds_api_key:
            pytest.skip("FPLAI_ODDS_API_KEY not configured (optional source)")
        _ensure_polite_get()
        client = odds_api.TheOddsApiClient()
        sports = client.sports()
        assert sports is not None
        assert any(s.get("key") == "soccer_epl" for s in sports)
