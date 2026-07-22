# STATUS.md — what works today

Last updated: 2026-07-22 (integrator pass, stage 1 complete).
Companions: `ARCHITECTURE.md` (module map + schemas), `FPL_KNOWLEDGE.md` (game rules),
`MODEL_DESIGN_INPUTS.md` (model spec).

## TL;DR

Stage 1 (data layer) is **built, integrated, and proven end-to-end on real data**.
All CLI data commands work. Model stages (train/predict/optimize) are stubs.

```
157 offline tests pass (uv run pytest -q -m "not live")
8 live network tests pass, 1 skips without FPLAI_ODDS_API_KEY (-m live)
ruff clean (uv run ruff check src tests)
```

## Commands (run from `backend/`)

| Command | What it does | Status |
|---|---|---|
| `uv run fplai snapshot` | Archives today's bootstrap+fixtures to `data/raw/fpl_api/snapshots/{date}/`, prints season state incl. the 2026-27 launch check | works |
| `uv run fplai backfill [--seasons 2016..2025]` | Downloads raw history: vaastav CSVs, football-data E0.csv, Understat league JSON, ClubElo snapshot + current-PL club histories. Idempotent (skip-if-exists) | works |
| `uv run fplai build [--seasons ...]` | Builds all processed parquet tables from whatever raw seasons are on disk (partial-season safe); odds.parquet only over seasons with a raw E0.csv. Offline, deterministic, idempotent | works |
| `uv run fplai refresh` | snapshot -> best-effort incremental pulls for the current season -> build -> model stages print "not yet implemented". **Exits 0** as long as the data portion succeeds | works |
| `uv run fplai train/predict/optimize` | Stage 2/3 stubs | print + exit 1 |

`--seasons` accepts single years and inclusive ranges: `--seasons 2024,2025`, `--seasons 2016..2025`.

## Proven end-to-end (real runs, 2026-07-22)

`snapshot` -> `backfill --seasons 2024,2025` -> `build` -> `refresh` all ran against the
real services. Raw on disk: vaastav 2018-19, 2019-20, 2024-25, 2025-26 (21 MB);
football-data + Understat 2024-25/2025-26; ClubElo snapshot + 20 club histories;
FPL API snapshot for 2026-07-22 (season=2025, GW38 done, 13,107,732 managers, 2026-27 not live).

Processed tables (built over seasons 2018, 2019, 2024, 2025):

| table | rows | notes |
|---|---|---|
| fixtures.parquet | 1,520 | 380 x 4 seasons; 2019-20 events 39-47 remapped to GWs 30-38 |
| player_match.parquet | 101,380 | per player per fixture; stints, empty_stadium, subs_regime flags |
| player_gw.parquet | 99,704 | per player per GW; DGWs summed with n_fixtures |
| players.parquet | 1,723 | one row per stable player_code |
| teams.parquet | 80 | 20 per season, all 4 seasons |
| odds.parquet | 760 | 2024+2025 only (seasons with raw E0.csv); de-margined probs sum to 1.0 |

Sanity checks verified: Haaland 2025-26 total_points = 239 (matches FPL_KNOWLEDGE §3.7);
2024-25 DGWs present (364 player-GW rows with n_fixtures >= 2 in GWs 24/25/32/33, e.g.
Tarkowski GW24 2 fixtures 17 pts); 20 teams per season in the crosswalk; 2018-19 builds
without teams.csv (derived from players_raw + master_team_list) with latin-1 names intact;
2019-20 restart rows flagged empty_stadium.

## Data coverage

- **vaastav** (FPL historical CSVs): downloader covers 2016-17..2025-26; currently on disk
  2018-19, 2019-20, 2024-25, 2025-26. Full backfill = `uv run fplai backfill` (~200 MB).
- **FPL API**: live client with 1 req/s throttle, browser UA, retries, on-disk HTTP cache
  (`data/cache/http/`). Daily snapshot archive works; element-summary sweep implemented.
- **Understat**: post-Dec-2025 JSON endpoints (the `X-Requested-With: XMLHttpRequest` header
  is the load-bearing requirement). League JSON on disk for 2024, 2025; history served back
  to 2016. Per-player fetch (`fetch_season`/`fetch_history(include_players=True)`) works but
  is slow by design (~500 req/season at 1 req/2s).
- **football-data.co.uk**: E0.csv 1993..present; closing 1X2 (Pinnacle PSC*, fallback chain)
  and O/U 2.5 parsed and de-margined. On disk: 2024, 2025.
- **ClubElo**: snapshot + per-club history archiving works (HTTP only — the API has no TLS).
  On disk: 2026-07-22 snapshot + the 20 current PL clubs.
- **the-odds-api**: client ready, degrades gracefully; **inactive** until `FPLAI_ODDS_API_KEY`
  is set in `.env`.
- **rules.py**: scoring tables 2016..2026, season flags, chip windows, BPS v4 matrix,
  sell-price, free-transfer arithmetic, 2026-27 position reclassifications — pure data, tested.

## Known gaps / next steps

1. **Model stages are stubs** — features/, models/, optimizer/, backtest/, api/ are stage 2+.
2. **odds.parquet `fpl_fixture_id` is all-NA** — the football-data -> FPL fixture join
   (via the teams crosswalk `footballdata_name` column, also still NA) is not wired yet.
3. **teams.parquet aux name columns (`understat_name`, `clubelo_name`, `footballdata_name`)
   are NA placeholders** — identity resolution against those sources is pending.
4. **Understat per-match xG is not joined into player_match yet** — `us_xg/us_xa/us_npxg/
   us_shots/us_key_passes` are all-NA placeholders; raw league JSONs are on disk and
   `understat.to_player_match_frame()` produces the tidy frame, but the crosswalk join
   (players.understat_id is only filled from vaastav id_dict for ~2021-23) needs the fuzzy
   name matcher.
5. **`player_gw.selected_by_percent` is all-NA for history** (vaastav ships raw counts, not
   percent); our own 2026-27 snapshots will carry it.
6. **ClubElo backfill only covers current PL clubs** — clubs relegated before 2025-26 need a
   pass driven by the teams crosswalk once clubelo_name is resolved.
7. **2026-27 launch**: `fplai snapshot` is the detector. When it reports LIVE, re-verify the
   UNCERTAIN items in FPL_KNOWLEDGE (GW1 deadline, BPS values, reclassification web_names,
   chip windows) against the new bootstrap.
8. **Full historical backfill not yet run** — only 4 seasons on disk. `uv run fplai backfill`
   pulls all 10 vaastav seasons + odds + Understat leagues in one go; build handles whatever
   subset is present.
9. **Refresh aux pulls are best-effort** (warn-and-continue on upstream outage); the FPL
   snapshot and the build remain hard failures by design.
