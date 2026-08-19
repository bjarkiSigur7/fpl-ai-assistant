"""Offline tests for the screenshot scanner: name matching, Gemini response parsing
and the /api/scan-team endpoint (Gemini call monkeypatched — no network)."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from fplai import config
from fplai.api.app import app
from fplai.api.cache import cache as file_cache
from fplai.data import gemini
from fplai.data.gemini import RecognitionError, SeenPlayer
from fplai.scan import match_squad, normalize_name

SEASON = 2026

#: (player_code, web_name, first_name, second_name, position, team_code, price)
ROSTER_ROWS = [
    (1001, "M.Salah", "Mohamed", "Salah", "MID", 1, 127),
    (1002, "Van Dijk", "Virgil", "van Dijk", "DEF", 1, 60),
    (1003, "Ødegaard", "Martin", "Ødegaard", "MID", 3, 82),
    (1004, "Haaland", "Erling", "Haaland", "FWD", 2, 151),
    # Same web_name at two clubs/prices — club + price must disambiguate.
    (1005, "Ward", "Danny", "Ward", "GKP", 1, 40),
    (1006, "Ward", "Joel", "Ward", "DEF", 2, 43),
]


def make_roster() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "season": SEASON,
                "player_code": code,
                "web_name": web,
                "first_name": first,
                "second_name": second,
                "position": pos,
                "team_code": team,
                "price": price,
            }
            for code, web, first, second, pos, team, price in ROSTER_ROWS
        ]
    )


SHORTS = {1: "LIV", 2: "MCI", 3: "ARS"}


# --------------------------------------------------------------------------------------
# normalize_name
# --------------------------------------------------------------------------------------


class TestNormalizeName:
    def test_diacritics_and_punctuation(self) -> None:
        assert normalize_name("Ødegaard") == "odegaard"
        assert normalize_name("M.Salah") == "m salah"
        assert normalize_name("  Van  Dijk ") == "van dijk"
        assert normalize_name("Gvardiol") == "gvardiol"

    def test_hand_mapped_chars(self) -> None:
        assert normalize_name("Włodarczyk") == "wlodarczyk"
        assert normalize_name("Sørloth") == "sorloth"


# --------------------------------------------------------------------------------------
# match_squad
# --------------------------------------------------------------------------------------


class TestMatchSquad:
    def test_exact_and_ocr_wobble(self) -> None:
        seen = [
            SeenPlayer(name="M.Salah", position="MID"),
            SeenPlayer(name="Odegaard", position="MID"),  # OCR dropped the Ø
            SeenPlayer(name="Van Dijk", position="DEF"),
        ]
        got = match_squad(seen, make_roster(), SHORTS)
        assert [m.player_code for m in got] == [1001, 1003, 1002]
        assert got[0].score == 1.0

    def test_full_name_containment(self) -> None:
        # Screenshot printed the surname; roster web_name is dotted.
        got = match_squad([SeenPlayer(name="Salah")], make_roster(), SHORTS)
        assert got[0].player_code == 1001

    def test_price_and_club_disambiguate_duplicates(self) -> None:
        seen = [
            SeenPlayer(name="Ward", position="DEF", price=4.3, club="MCI"),
            SeenPlayer(name="Ward", position="GKP", price=4.0, club="LIV"),
        ]
        got = match_squad(seen, make_roster(), SHORTS)
        assert got[0].player_code == 1006
        assert got[1].player_code == 1005

    def test_duplicate_reads_claim_one_code(self) -> None:
        seen = [SeenPlayer(name="Haaland"), SeenPlayer(name="Haaland")]
        got = match_squad(seen, make_roster(), SHORTS)
        codes = [m.player_code for m in got]
        assert codes.count(1004) == 1
        assert codes.count(None) == 1

    def test_garbage_stays_unmatched(self) -> None:
        got = match_squad([SeenPlayer(name="Zzyzx Qwerty")], make_roster(), SHORTS)
        assert got[0].player_code is None
        assert got[0].score == 0.0

    def test_position_mismatch_blocks_weak_match(self) -> None:
        # A fuzzy-only name with the wrong row should not clear the bar.
        got = match_squad([SeenPlayer(name="Haland", position="GKP")], make_roster(), SHORTS)
        assert got[0].player_code is None


# --------------------------------------------------------------------------------------
# Gemini response parsing
# --------------------------------------------------------------------------------------


PLAYERS_JSON = json.dumps(
    {"players": [{"name": "M.Salah", "position": "MID", "price": 12.7, "club": "LIV"}]}
)


class TestGeminiParsing:
    def test_output_text(self) -> None:
        assert gemini._extract_text({"output_text": PLAYERS_JSON}) == PLAYERS_JSON

    def test_steps_fallback(self) -> None:
        data = {"steps": [{"type": "model", "content": [{"text": PLAYERS_JSON}]}]}
        assert gemini._extract_text(data) == PLAYERS_JSON

    def test_legacy_candidates_fallback(self) -> None:
        data = {"candidates": [{"content": {"parts": [{"text": PLAYERS_JSON}]}}]}
        assert gemini._extract_text(data) == PLAYERS_JSON

    def test_no_text_raises(self) -> None:
        with pytest.raises(RecognitionError):
            gemini._extract_text({"steps": []})

    def test_parse_players_with_fence(self) -> None:
        got = gemini._parse_players(f"```json\n{PLAYERS_JSON}\n```")
        assert got == [SeenPlayer(name="M.Salah", club="LIV", price=12.7, position="MID")]

    def test_parse_players_aliases_and_bounds(self) -> None:
        text = json.dumps(
            {
                "players": [
                    {"name": "Raya", "position": "GK", "price": 127},  # 127 = misread 0.1m units
                    {"name": "", "position": "DEF"},  # dropped: no name
                ]
            }
        )
        got = gemini._parse_players(text)
        assert got == [SeenPlayer(name="Raya", club=None, price=None, position="GKP")]

    def test_parse_non_json_raises(self) -> None:
        with pytest.raises(RecognitionError):
            gemini._parse_players("sorry, I cannot see any players")

    def test_no_cards_raises(self) -> None:
        with pytest.raises(RecognitionError):
            gemini._parse_players(json.dumps({"players": []}))


# --------------------------------------------------------------------------------------
# POST /api/scan-team
# --------------------------------------------------------------------------------------


IMAGE_B64 = base64.b64encode(b"fake-png-bytes").decode()


@pytest.fixture()
def scan_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    processed = tmp_path / "processed"
    processed.mkdir(parents=True)
    monkeypatch.setattr(config, "PROCESSED_DIR", processed)
    file_cache.clear()
    make_roster().to_parquet(processed / "live_roster.parquet", index=False)
    pd.DataFrame(
        [
            {
                "season": SEASON,
                "fpl_team_id": tid,
                "team_code": code,
                "name": short,
                "short_name": short,
            }
            for tid, (code, short) in enumerate(SHORTS.items(), start=1)
        ]
    ).to_parquet(processed / "teams.parquet", index=False)
    yield SimpleNamespace(processed=processed)
    file_cache.clear()


@pytest.fixture()
def client(scan_env: SimpleNamespace) -> TestClient:
    return TestClient(app)


class TestScanTeamEndpoint:
    def test_ok(self, client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
        def fake_recognize(image_base64: str, mime_type: str, **_: object) -> list[SeenPlayer]:
            assert image_base64 == IMAGE_B64
            assert mime_type == "image/png"
            return [
                SeenPlayer(name="M.Salah", position="MID", price=12.7),
                SeenPlayer(name="Nobody Real", position="FWD"),
            ]

        monkeypatch.setattr(gemini, "recognize_squad", fake_recognize)
        resp = client.post(
            "/api/scan-team", json={"image_base64": IMAGE_B64, "mime_type": "image/png"}
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["codes"] == [1001]
        assert body["unmatched"] == ["Nobody Real"]
        assert body["model"] == gemini.MODEL
        assert body["players"][0]["web_name"] == "M.Salah"
        assert body["players"][1]["player_code"] is None

    def test_bad_base64_422(self, client: TestClient) -> None:
        resp = client.post("/api/scan-team", json={"image_base64": "not@base64!!"})
        assert resp.status_code == 422

    def test_no_key_503(self, client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(config.settings, "gemini_api_key", "")
        resp = client.post("/api/scan-team", json={"image_base64": IMAGE_B64})
        assert resp.status_code == 503
        assert "FPLAI_GEMINI_API_KEY" in resp.json()["detail"]

    def test_gemini_down_502(self, client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
        def boom(*_: object, **__: object) -> list[SeenPlayer]:
            raise RecognitionError("Gemini unreachable: timeout")

        monkeypatch.setattr(gemini, "recognize_squad", boom)
        resp = client.post("/api/scan-team", json={"image_base64": IMAGE_B64})
        assert resp.status_code == 502

    def test_roster_missing_404(
        self, client: TestClient, scan_env: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (scan_env.processed / "live_roster.parquet").unlink()
        file_cache.clear()
        resp = client.post("/api/scan-team", json={"image_base64": IMAGE_B64})
        assert resp.status_code == 404
        assert "live_roster.parquet" in resp.json()["detail"]
