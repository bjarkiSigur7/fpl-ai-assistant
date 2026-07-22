# STATUS.md — what works today

Last updated: 2026-07-22 (integrator pass, stages 4-6 complete: backtest harness, CLI
wiring, FastAPI layer, Next.js dashboard — the full product is live in pre-launch demo
mode). Companions: `ARCHITECTURE.md` (module map + schemas), `FPL_KNOWLEDGE.md` (game
rules), `MODEL_DESIGN_INPUTS.md` (model spec).

## TL;DR

The whole chain works end to end: **data -> features -> models -> predictions ->
MILP optimizer -> recommendation artifacts -> FastAPI -> dashboard**, plus a
walk-forward **policy backtester** that replays a season GW-by-GW. The 2026-27 FPL
game has **not launched yet** (the API still serves finished 2025-26), so the product
runs in **pre-launch demo mode**: predictions/optimization run on a 2025-26 backtest
window (default GW34+, chosen from strictly-prior information) and the dashboard
counts down to the announced GW1 deadline (2026-08-21 17:30 UTC). When the FPL API
flips to 2026-27 the same commands serve the live season.

```
709+ offline tests pass   (backend: uv run pytest -q -m "not live")
ruff clean                (uv run ruff check src tests)
eslint clean + prod build (frontend: npm run lint && npm run build)
```

## Commands

From the repo root (Makefile):

| Command | What it does |
|---|---|
| `make dev` | API (uvicorn :8000) + dashboard (next dev :3000) together |
| `make api` / `make web` | Either half on its own |
| `make refresh` | Full daily cycle: snapshot -> pulls -> build -> predict -> optimize (exit-0) |
| `make train` | Retrain all model components (~70 s) |
| `make demo [SEASON=2025 GW=34]` | Pre-launch demo chain: predict + optimize a backtest window |
| `make predict` / `make optimize` | Live-mode stages (degrade gracefully pre-launch) |
| `make backtest [SEASON=2025] [GWS=30..38]` | Walk-forward policy backtest |
| `make test` / `make test-live` / `make lint` | Offline suite / live-marked tests / linters |

From `backend/` (full flag surface):

| Command | What it does | Status |
|---|---|---|
| `uv run fplai snapshot` | Archives today's bootstrap+fixtures, prints season state incl. the 2026-27 launch check | works |
| `uv run fplai backfill [--seasons 2016..2025]` | Downloads raw history: vaastav, football-data E0, Understat, ClubElo. Idempotent | works |
| `uv run fplai build [--seasons ...]` | Builds all processed parquet tables. Offline, deterministic, idempotent | works |
| `uv run fplai train [--seasons ...] [--before-season S --before-gw G]` | Trains minutes/team/rates/bonus; artifacts + manifest to `data/models/` | works (~70 s) |
| `uv run fplai predict [--season S --gw G] [--horizon N] [--no-odds]` | Live or backtest-window predictions -> `predictions(.gw).parquet` | works |
| `uv run fplai optimize [--entry-id E] [--season S --gw G] [--horizon N] [--no-chips] [--no-stability] [--stability-n N]` | MILP plan + weekly verdict -> `recommendation.json`, `dream_team.json`, `chip_curves.parquet` | works (~15 min full; ~1 min with `--no-chips --no-stability`) |
| `uv run fplai backtest --season 2025 [--gws 30..38]` | Stage-4 walk-forward policy replay -> `data/backtests/<season>-gwA-B/` | works |
| `uv run fplai refresh` | snapshot -> pulls -> build -> predict -> optimize, ends with the launch-watch panel. **Exits 0** as long as the data portion succeeds | works |
| `uv run uvicorn fplai.api.app:app --port 8000` | The stage-5 API (all routes under `/api`) | works |

`--seasons`/`--gws` accept single values and inclusive ranges (`2016..2025`, `30..38,38`).

## Pre-launch demo mode (what the dashboard shows today)

2026-27 has not launched, so there are no upcoming fixtures. The demo pipeline is:

```
uv run fplai predict  --season 2025 --gw 34   # walk-forward predictions, GWs 34-38
uv run fplai optimize --season 2025 --gw 34   # None-state initial-squad build over that window
```

`optimize` auto-detects demo mode (no upcoming fixtures on disk) and says so; the API
exposes `pre_launch: true` plus the GW1 countdown in `GET /api/state`, and the
dashboard ticker shows "PRE-SEASON — 2026/27 LAUNCHES SOON" with the provisional
deadline. With `FPLAI_ENTRY_ID`/`--entry-id` set, `optimize` builds the squad state
from the FPL API; if the entry's GW isn't covered by the predictions window (always
true pre-launch) it falls back loudly to the initial-squad demo.

### Current demo verdict (2025-26 GW34, horizon 5, full chips+stability)

Initial-squad build: 15 players, £99.9m spent, expected **78.8 xP in GW34**
(Bench Boost played), captain M.Salah. Top draft bullets: Virgil (+18.1 xP over the
horizon, 100% stability), B.Fernandes (+31.0, 93%), Bowen (+24.6, 100%). Chip
verdicts: bb2 play GW34 (sweep best GW35 +9.3), tc2 hold for GW36 (+12.0), wc2 hold
for GW35 (+11.3), fh2 hold for GW36 (+11.5). Artifacts: `recommendation.json`,
`dream_team.json`, `chip_curves.parquet` in `data/processed/`. (End-of-season demo
window — chips cluster because they expire at GW38; a real GW1 plan spreads them.)

## Real end-to-end verification (2026-07-22, retrained artifacts)

- Offline test suite: **714 passed** (709 + 5 chip-GW-ban tests), ruff clean, frontend
  `npm run build` + eslint clean, zero browser-console errors across all four pages.
- Walk-forward xP accuracy (train < GW30, eval GWs 30-38 of 2025-26): Zeros RMSE/MAE
  **0.798/0.318** (beats OpenFPL's published 0.818/0.427), Blanks 1.464/1.109,
  Tickers 1.554/1.269, Haulers 5.368/4.543; within-GW Spearman 0.726. Beats the
  last-5 baseline in every category on identical rows.
- Policy backtest (GWs 33-35, coarse pretrained mode): model policy **249 pts** vs
  last-5 baseline 223 vs set-and-forget 187. Caveat auto-reported by the harness:
  artifacts overlapped the window — use `fplai train --before-season 2025
  --before-gw 33` for honest season-scale numbers.
- Team model: 1X2 log-loss within 0.011 of de-margined closing odds on the 2025-26
  holdout; minutes model bucket log-loss 0.437 on the eval window.

## API (stage 5)

FastAPI under `/api`, pydantic response models mirroring the optimizer contracts
(`Recommendation`/`DreamTeam`/`SquadState` imported verbatim, never redefined):

| Route | Serves |
|---|---|
| `GET /api/health` | liveness |
| `GET /api/state` | season state (parsed offline from the latest snapshot), `pre_launch` + GW1 countdown, data freshness (incl. predictions season/GWs), model manifest |
| `GET /api/predictions?season&gw&limit&offset&position&sort&order` | paginated per-player xP + components + q0, name/team/price joins, per-fixture DGW breakdown with kickoffs |
| `GET /api/dream-team` | `dream_team.json` (404 when missing/stale vs predictions) |
| `GET /api/recommendation` | `recommendation.json` (bare or entry-wrapped shape) |
| `GET /api/chip-curves` | `chip_curves.parquet` incl. evaluated/skip_reason/urgency columns |
| `GET /api/players/{player_code}` | identity + per-GW season history + upcoming per-fixture xP |
| `GET /api/my-team/{entry_id}` (+`/status`) | cached recommendation (6 h TTL: memory -> per-entry cache -> shared file) or **202** + background single-flight MILP solve |
| `POST /api/refresh` (+`GET /api/refresh/status`) | single-flight background `run_refresh` with captured log tail (409 while running) |

Every parquet/JSON read goes through an mtime-invalidated cache, so the API picks up
refresh output without restarting. CORS allows `http://localhost:3000`.

## Frontend (PITCHSIDE dashboard)

Next.js App Router + TS strict + Tailwind v4 in `frontend/`; only added dep is `swr`.
Pages: `/` (ticker, VERDICT card, dream-team + my-team SVG pitches, top movers, xP
curve), `/players` (sortable predictions explorer), `/players/[code]` (history chart +
per-fixture xP breakdown with kickoffs), `/planner` (multi-GW plan timeline, chip EV
small multiples, stability bars), `/settings` (entry id, wiring readout, refresh
trigger with live log tail). `src/lib/types.ts` carries `Api*` wire types mirroring
`fplai/api/schemas.py` field-for-field; `src/lib/api.ts` adapts them to the view
models (and pages the per-GW `/api/predictions` into the horizon-wide view).
`NEXT_PUBLIC_MOCK=1` runs the whole UI standalone on deterministic 2025-26 mocks.

## Backtester (stage 4)

`fplai.backtest.run_backtest(season, gws=..., policy_params=..., retrain_every=...)`
replays a season GW-by-GW: predictions from artifacts (coarse mode with a manifest
leakage check, or retrain-every-N blocks), weekly MILP re-solves on rolled state
(bank/sell-prices/FTs/chips incl. the 2025 AFCON top-up), deterministic §1.5 autosubs,
captain->vice fallback, TC/BB/FH/WC semantics, hit subtraction, and a light forced
chip sweep. Baselines in the same loop: last-5-average xP through the same solver and
a set-and-forget template squad; §7.3 reference constants (avg manager 2250, top-10k
2625). Results (per-GW ledgers + xP-accuracy metrics + leakage warnings) save under
`data/backtests/<season>-gwA-B/` as `result.json` + `ledger.parquet`.

## Model artifacts (data/models/)

Retrained 2026-07-22 15:02 UTC on the fully-enriched corpus (all 10 seasons,
253,568 player-fixture rows, 3,800 matches): minutes LgbMinutesModel (bucket log-loss
0.445 in-sample 2025), team Dixon-Coles (gamma 1.19, rho −0.08), rates per-position
LGBM Poisson, bonus calibration on 2025-26 residuals. `manifest.json` records the
train window — the backtester checks it and warns loudly when artifacts postdate the
replayed GWs (LEAKY-ARTIFACTS).

## Walk-forward sanity eval (honest numbers, trained before 2025-26 GW30)

Protocol: `fplai train --before-season 2025 --before-gw 30`, then 1-GW-ahead
predictions for GWs 30-38 (7,074 player-GW rows). RMSE/MAE by realized-points
category vs a last-5-average baseline on identical rows:

| category | ours | last-5 (same rows) | OpenFPL (ref) | FPL Review (ref) |
|---|---|---|---|---|
| Zeros (0 pts) | **0.798 / 0.318** | 1.013 / 0.347 | 0.818 / 0.427 | 0.689 / 0.237 |
| Blanks (1-2) | **1.464 / 1.109** | 1.916 / 1.418 | 1.291 / 0.749 | 1.189 / 0.597 |
| Tickers (3-4) | **1.554 / 1.269** | 1.986 / 1.667 | 1.517 / 1.127 | 1.594 / 1.227 |
| Haulers (>=5) | **5.368 / 4.543** | 5.646 / 4.751 | 5.142 / 4.317 | 5.172 / 4.381 |
| All | 1.870 / 0.904 | 2.106 / 1.029 | — | — |

Beats last-5 in every category; Zeros beat OpenFPL's reference; within-GW Spearman
0.726. (Numbers predate the enriched-corpus retrain; the harness reproduces the same
protocol via `fplai train --before-season/--before-gw` + `fplai backtest`.)

## Data coverage

- Processed tables: all 10 seasons 2016-17..2025-26 (fixtures 3,800; player_match
  253,568; player_gw 244,425; odds 3,800 with 100% fixture-id join).
- Understat xG joined into player_match for all matched seasons (per-player JSON
  backfill for the remaining 2016-18/2024-25 gaps runs in the background; re-run
  `fplai build` after it completes to lift coverage further).
- rules.py: scoring 2016..2026, season flags, chip windows, BPS v4, sell-price, FT
  arithmetic, 2026-27 reclassifications.

## Launch-week checklist (2026-27 — from FPL_KNOWLEDGE §3.8 + re-verification protocol)

`fplai snapshot` (inside `fplai refresh`) is the detector: it polls
`bootstrap-static` and flags when `static_content_url` flips to `2026_27/`. When it
does:

1. **Re-verify rules**: diff `game_settings`, `game_config.rules`/`scoring`,
   `chips[]`, `element_types[]`, `events[]` (GW1 deadline), `elements[]` against
   `FPL_KNOWLEDGE.md`'s UNCERTAIN items (DefCon thresholds, chip inventory incl.
   AOA/AM, BPS v4 values, 5-FT cap). All element IDs reset at relaunch —
   `element.code` is the stable key (already our canonical id everywhere).
2. **Ingest launch data**: daily `fplai refresh` now captures prices, availability
   (`status`/`chance_of_playing` — our first deadline-timestamped snapshots) and
   `selected_by_percent`; rebuild the GW map.
3. **Rosters for GW1**: switch the prediction pool to the bootstrap roster (promoted
   teams + summer movers have no player_match history; the roll-forward synthesizer
   misses them until then — known gap 5 below). Split 2025-26 per-player data by club
   stint; apply the 11 announced position changes (already in `rules.py`).
4. **Retrain + re-point**: `fplai train` on all data; `fplai refresh` then serves
   live predictions/recommendations automatically; the dashboard ticker flips to
   "LIVE 2026/27" with real event deadlines.
5. **Projection adjustments**: WC-fatigue dampening (§3.3) and new-manager style
   uncertainty (§3.5); re-pull transfer trackers weekly until 1 Sep; treat GW scores
   as provisional until next-day 09:00 UK bonus lockdown.
6. **Odds**: snapshot pre-deadline odds so live decisions never use closing lines
   (backtest caveat 3 below).

## Known gaps / caveats

1. **Optimize latency**: full verdict (chip EV curves + 30 stability re-solves) is
   ~15 min of MILP; `--no-chips --no-stability` gives a ~1 min verdict. `refresh`
   uses full defaults — fine for a daily cron, mind it for interactive runs.
2. **Coarse backtest look-ahead**: within a decision at GW t, xP for t+1..t+H-1 uses
   matches before *their own* GW (planning-only peek; the scored GW is strictly
   pre-deadline). Recorded in `BacktestResult.leakage_warnings`. Use
   `retrain_every` mode + `--before-*` artifacts for per-decision-clean numbers.
3. **Odds leakage in backtests**: the blend uses closing odds; for strict
   deadline-time evals run `predict --no-odds` (2026-27 fixes this with pre-deadline
   odds snapshots).
4. **No availability/news features pre-2026** — `f_status_*` NaN in history; our own
   snapshot archive fixes this from 2026-27 GW1.
5. **Live-mode rosters roll forward from last appearances** — GW1 2026-27 needs the
   bootstrap-based roster (launch checklist item 3).
6. **Chip policy in the backtester is greedy within the horizon** (no opportunity
   cost beyond it) — raise `chip_play_margin` or `evaluate_chips=False` to tame it.
7. **`/api/my-team/{entry_id}` pre-launch**: a live entry's GW is never covered by
   the demo predictions, so the background solve falls back to the initial-squad
   demo verdict (loudly). Squad-aware verdicts become real at launch.
8. **Blanks (1-2 pts)** remain our weakest xP category vs OpenFPL; revisit with
   team-partitioned CV.
9. **Understat per-player JSON backfill** for 2016-18/2024/2025 still filling in the
   background (throttled); models tolerate the NaNs; re-run `fplai build` to ingest.
