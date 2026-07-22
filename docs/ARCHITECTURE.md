# ARCHITECTURE.md — Module map and data contracts

Companion to `FPL_KNOWLEDGE.md` (game rules — authoritative) and `MODEL_DESIGN_INPUTS.md`
(model/pipeline spec — authoritative). This file fixes the *code* structure: who owns which
files, the canonical data schemas, and the interfaces between modules. Implementation agents
must not edit files outside their ownership without coordination.

## Module map (backend/src/fplai/)

```
rules.py              Season constants: scoring tables per season, feature flags, chip windows,
                      BPS matrix, sell-price formula, position reclassifications. Pure data+fns.
config.py             Paths & settings (exists — do not restructure).
data/
  fpl_api.py          FPL API client: all endpoints, 1 req/s throttle, browser UA, retries,
                      on-disk HTTP cache. Snapshot archiver + 2026-27 launch detector.
  vaastav.py          Historical backfill: download per-season files from the
                      vaastav/Fantasy-Premier-League GitHub repo into data/raw/vaastav/.
  understat.py        Understat client (post-Dec-2025 JSON endpoints: /getLeagueData/EPL/{year},
                      /getPlayerData/{id}/) + vaastav's bundled understat extracts for backfill.
  football_data.py    football-data.co.uk E0.csv odds downloader (1993->present).
  clubelo.py          ClubElo API client (CSV; team Elo history + snapshots).
  odds_api.py         the-odds-api.com client (optional; FPLAI_ODDS_API_KEY; degrade gracefully).
  build.py            Raw -> processed: builds the canonical parquet tables below, applying all
                      FPL_KNOWLEDGE Part 2 remaps (2019-20 events, void GWs, stint splits).
  crosswalk.py        Identity resolution: player_code <-> per-season element ids <-> understat
                      ids; canonical team table across FPL/Understat/ClubElo/football-data.
features/             Feature engineering (multi-horizon windows per OpenFPL) — stage 2.
models/               minutes.py, team.py (Dixon-Coles), rates.py, defcon.py, bonus.py,
                      assemble.py (xP decomposition) — stage 2.
optimizer/            milp.py (multi-GW MILP on HiGHS), chips.py, sensitivity.py — stage 3.
backtest/             Walk-forward harness — stage 4.
api/                  FastAPI app — stage 5.
pipeline.py           Orchestration only. cli.py wraps it. (Wired by the integrator, not agents.)
```

## Data directory layout (data/, gitignored)

```
raw/fpl_api/snapshots/{YYYY-MM-DD}/bootstrap.json, fixtures.json   # daily archive, append-only
raw/fpl_api/element_summary/{season}/{element_id}.json
raw/vaastav/{season}/merged_gw.csv, players_raw.csv, teams.csv, understat/...
raw/understat/{season}/league.json, players/{understat_id}.json
raw/football_data/{season}/E0.csv
raw/clubelo/{ClubName}.csv
processed/player_match.parquet   processed/player_gw.parquet   processed/fixtures.parquet
processed/teams.parquet          processed/players.parquet     processed/odds.parquet
models/{component}/...           cache/http/...
```

`season` is always the start year as int (2016 … 2026). Never the "2016-17" string in code;
use `fplai.rules.season_label(2016) == "2016-17"` for display.

## Canonical processed tables (parquet; pandas; column names are the contract)

### players.parquet — one row per player (cross-season identity)
`player_code` (int, FPL `element.code` — THE stable key), `web_name`, `first_name`,
`second_name`, `understat_id` (nullable Int64), `opta_code` (nullable str).

### teams.parquet — one row per team-season
`season`, `fpl_team_id` (per-season!), `team_code` (int, stable), `name`, `short_name`,
`understat_name` (nullable), `clubelo_name` (nullable), `footballdata_name` (nullable).

### fixtures.parquet — one row per fixture
`season`, `gw` (canonical, remapped per FPL_KNOWLEDGE §2.4: 2019-20 API events 39-47 -> 30-38;
2022-23 GW7 kept but flagged), `fpl_fixture_id`, `kickoff_utc` (datetime), `home_team_code`,
`away_team_code`, `home_goals`, `away_goals` (nullable), `finished` (bool),
`void` (bool — 0-pts anomaly rows).

### player_match.parquet — one row per player per fixture appearance-opportunity
Keys: `season`, `gw`, `fpl_fixture_id`, `player_code`. Identity: `fpl_element_id` (per-season),
`team_code`, `opponent_code`, `was_home` (bool), `position` (str GKP/DEF/MID/FWD — the FPL
position that season), `price` (int, 0.1m units at that GW).
Outcomes: `minutes`, `total_points`, `goals_scored`, `assists`, `clean_sheets`, `goals_conceded`,
`saves`, `penalties_saved`, `penalties_missed`, `yellow_cards`, `red_cards`, `own_goals`,
`bonus`, `bps`, `starts` (nullable), `defensive_contribution`, `tackles`, `recoveries`,
`clearances_blocks_interceptions` (nullable — 2025+ per §2.3; also 2016-18 rich stats where available),
`xg`, `xa`, `xgc` (nullable — FPL API 2022+, Understat-sourced `us_xg`, `us_xa`, `us_npxg`,
`us_shots`, `us_key_passes` for all seasons where matched).
Flags: `empty_stadium` (bool), `void_gw` (bool), `subs_regime` (int 3|5), `stint_id` (int,
increments when a player changes club mid-season).

### player_gw.parquet — per player per GW (sum over the GW's fixtures; the FPL-points view)
Same identity columns; `n_fixtures`, summed outcomes, `value` (price that GW),
`selected_by_percent` (nullable), `transfers_in_event`/`transfers_out_event` (nullable).

### odds.parquet — one row per fixture with market data
`season`, `date`, `home_footballdata_name`, `away_footballdata_name`, closing 1X2
(`odds_h`, `odds_d`, `odds_a`), over/under 2.5 (`odds_over25`, `odds_under25`) where present,
joined to `fpl_fixture_id` when resolvable.

## Cross-module interface contracts

- `rules.get_scoring(season: int) -> dict[str, dict[str, int]]` — action -> position -> points.
- `rules.SEASON_FLAGS: dict[int, SeasonFlags]` — dataclass mirroring FPL_KNOWLEDGE §2.2.
- `rules.BPS_2026: dict[str, float]`, `rules.chip_windows(season) -> ...`, `rules.sell_price(purchase: int, now: int) -> int`.
- `fpl_api.FplApiClient` — `.bootstrap()`, `.fixtures()`, `.element_summary(id)`,
  `.event_live(gw)`, `.entry(id)`, `.entry_picks(id, gw)`, `.entry_history(id)`,
  `.entry_transfers(id)`; `.take_snapshot() -> Path` (writes today's snapshot dir, idempotent);
  `.season_state() -> SeasonState` (dataclass: `season`, `is_live_2026_27`, `next_gw`,
  `next_deadline_utc`, from static_content_url + events).
- `build.build_all()` — reads raw/, writes every processed table. Individual `build_*()` fns
  callable separately. Deterministic, idempotent, safe to re-run.
- All HTTP goes through a shared throttled fetch helper in `fpl_api.py`
  (`polite_get(url, *, min_interval_s, cache_ttl_s)`) — reused by every data client.

## Conventions

- Python 3.12, type hints everywhere, dataclasses/pydantic at module boundaries.
- No global mutable state; clients take explicit paths from `fplai.config`.
- Every data module gets a `tests/test_<module>.py` with offline fixture-based tests
  (record one small real payload into `tests/fixtures/`); live-network tests marked
  `@pytest.mark.live` (excluded by default; run in CI/verification explicitly).
- `ruff check` clean; line length 100.
- Errors: raise, don't silently continue; downloaders log and retry with tenacity.
