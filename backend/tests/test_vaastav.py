"""Tests for fplai.data.vaastav (offline fixture-based + one live smoke)."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import httpx
import pytest

import fplai.data
from fplai.data import vaastav

CSV = b"a,b\n1,2\n"


class RecordingFetch:
    """Fake fetch: records URLs; per-URL payloads; None payload -> 404."""

    def __init__(
        self,
        payloads: dict[str, bytes | None] | None = None,
        default: bytes | None = CSV,
    ) -> None:
        self.calls: list[str] = []
        self.payloads = payloads or {}
        self.default = default

    def __call__(self, url: str) -> bytes:
        self.calls.append(url)
        payload = self.payloads.get(url, self.default)
        if payload is None:
            raise FileNotFoundError(url)
        return payload


def test_season_label_fallback() -> None:
    assert vaastav.season_label(2016) == "2016-17"
    assert vaastav.season_label(2025) == "2025-26"
    assert vaastav.season_label(1999) == "1999-00"


def test_download_season_urls_and_dests(tmp_path: Path) -> None:
    fetch = RecordingFetch()
    paths = vaastav.download_season(2022, raw_root=tmp_path, fetch=fetch)

    base = f"{vaastav.VAASTAV_BASE_URL}/2022-23"
    assert f"{base}/gws/merged_gw.csv" in fetch.calls
    assert f"{base}/players_raw.csv" in fetch.calls
    assert f"{base}/understat/understat_player.csv" in fetch.calls

    season = tmp_path / "vaastav" / "2022-23"
    # gws/merged_gw.csv lands at the season root per the ARCHITECTURE raw layout.
    assert (season / "merged_gw.csv").read_bytes() == CSV
    assert (season / "players_raw.csv").exists()
    assert (season / "understat" / "understat_player.csv").exists()
    assert all(p.exists() for p in paths)


def test_download_season_skips_existing_and_force(tmp_path: Path) -> None:
    fetch = RecordingFetch()
    files = ["players_raw.csv"]
    vaastav.download_season(2020, files, raw_root=tmp_path, fetch=fetch)
    assert len(fetch.calls) == 1
    vaastav.download_season(2020, files, raw_root=tmp_path, fetch=fetch)
    assert len(fetch.calls) == 1  # skip-if-exists
    vaastav.download_season(2020, files, raw_root=tmp_path, fetch=fetch, force=True)
    assert len(fetch.calls) == 2  # force re-downloads


def test_missing_optional_file_is_skipped(tmp_path: Path) -> None:
    url_404 = f"{vaastav.VAASTAV_BASE_URL}/2016-17/teams.csv"
    fetch = RecordingFetch(payloads={url_404: None})
    paths = vaastav.download_season(2016, raw_root=tmp_path, fetch=fetch)
    assert not (tmp_path / "vaastav" / "2016-17" / "teams.csv").exists()
    assert (tmp_path / "vaastav" / "2016-17" / "merged_gw.csv").exists()
    assert all(p.name != "teams.csv" for p in paths)


def test_missing_required_file_raises(tmp_path: Path) -> None:
    fetch = RecordingFetch(default=None)
    with pytest.raises(FileNotFoundError, match="merged_gw"):
        vaastav.download_season(2016, raw_root=tmp_path, fetch=fetch)


def test_string_manifest_entries_are_required(tmp_path: Path) -> None:
    fetch = RecordingFetch(default=None)
    with pytest.raises(FileNotFoundError):
        vaastav.download_season(2018, ["gws/merged_gw.csv"], raw_root=tmp_path, fetch=fetch)
    fetch = RecordingFetch()
    vaastav.download_season(2018, ["gws/merged_gw.csv"], raw_root=tmp_path, fetch=fetch)
    assert (tmp_path / "vaastav" / "2018-19" / "merged_gw.csv").exists()


def test_download_all_and_master_team_list(tmp_path: Path) -> None:
    fetch = RecordingFetch()
    out = vaastav.download_all([2016, 2017], ["players_raw.csv"], raw_root=tmp_path, fetch=fetch)
    assert set(out) == {2016, 2017}
    assert (tmp_path / "vaastav" / "master_team_list.csv").exists()
    assert f"{vaastav.VAASTAV_BASE_URL}/master_team_list.csv" in fetch.calls


def test_payload_bytes_variants() -> None:
    url = "http://x/y.csv"
    assert vaastav._payload_bytes(b"abc", url) == b"abc"
    assert vaastav._payload_bytes("abc", url) == b"abc"
    resp = types.SimpleNamespace(status_code=200, content=b"ok")
    assert vaastav._payload_bytes(resp, url) == b"ok"
    with pytest.raises(FileNotFoundError):
        vaastav._payload_bytes(types.SimpleNamespace(status_code=404, content=b""), url)
    with pytest.raises(RuntimeError):
        vaastav._payload_bytes(types.SimpleNamespace(status_code=500, content=b""), url)
    # raw.githubusercontent's "404: Not Found" body without a surfaced status code
    with pytest.raises(FileNotFoundError):
        vaastav._payload_bytes(b"404: Not Found", url)


def test_default_fetch_uses_polite_get(monkeypatch: pytest.MonkeyPatch) -> None:
    """The default fetch adapts the fpl_api.polite_get contract (bytes or Response)."""
    seen: dict[str, object] = {}

    def polite_get(url: str, *, min_interval_s: float, cache_ttl_s: int) -> bytes:
        seen.update(url=url, min_interval_s=min_interval_s, cache_ttl_s=cache_ttl_s)
        return b"payload"

    stub = types.ModuleType("fplai.data.fpl_api")
    stub.polite_get = polite_get  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "fplai.data.fpl_api", stub)
    monkeypatch.setattr(fplai.data, "fpl_api", stub, raising=False)

    assert vaastav._polite_fetch("http://x/y.csv") == b"payload"
    assert seen["url"] == "http://x/y.csv"
    assert seen["min_interval_s"] == vaastav.MIN_INTERVAL_S
    assert seen["cache_ttl_s"] == vaastav.CACHE_TTL_S


def test_read_csv_tolerant_latin1(tmp_path: Path) -> None:
    """2018-19 quirk: files contain latin-1 bytes that are invalid UTF-8."""
    path = tmp_path / "players_raw.csv"
    path.write_bytes(b"first_name,code\nJo\xe3o,123\n")  # 0xE3 = latin-1 a-tilde
    df = vaastav.read_csv_tolerant(path)
    assert df.loc[0, "first_name"] == "João"
    utf8 = tmp_path / "utf8.csv"
    utf8.write_text("first_name,code\nÖzil,9\n", encoding="utf-8")
    assert vaastav.read_csv_tolerant(utf8).loc[0, "first_name"] == "Özil"


# --- live -------------------------------------------------------------------------


def _httpx_fetch(url: str) -> bytes:
    resp = httpx.get(
        url, timeout=30.0, follow_redirects=True, headers={"User-Agent": "fplai-tests"}
    )
    if resp.status_code == 404:
        raise FileNotFoundError(url)
    resp.raise_for_status()
    return resp.content


@pytest.mark.live
def test_live_download_2025_26_small_files(tmp_path: Path) -> None:
    """Hit the real GitHub raw host for two small 2025-26 files."""
    paths = vaastav.download_season(
        2025, ["teams.csv", "players_raw.csv"], raw_root=tmp_path, fetch=_httpx_fetch
    )
    assert len(paths) == 2
    teams = vaastav.read_csv_tolerant(tmp_path / "vaastav" / "2025-26" / "teams.csv")
    assert len(teams) == 20
    assert "Arsenal" in set(teams["name"])
    players = vaastav.read_csv_tolerant(tmp_path / "vaastav" / "2025-26" / "players_raw.csv")
    assert {"id", "code", "element_type", "team", "team_code"} <= set(players.columns)
