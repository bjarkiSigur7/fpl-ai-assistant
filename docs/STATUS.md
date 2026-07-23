# STATUS.md — what works today

Last updated: 2026-07-23 (launch-day integrator pass: 2026-27 went LIVE, the full
chain ran against the real day-1 API and the product now serves the real GW1
build). Companions: `ARCHITECTURE.md` (module map + schemas), `FPL_KNOWLEDGE.md` (game
rules — §1.12 uncertainty register fully resolved at launch), `MODEL_DESIGN_INPUTS.md`
(model spec).

## TL;DR

**THE 2026-27 GAME IS LIVE.** The FPL API flipped to `2026_27/` on 2026-07-23
(next_gw=1, GW1 deadline **2026-08-21 17:30 UTC**, 555 elements, 380 fixtures).
The whole chain now runs in live mode end to end: **day-1 bootstrap -> live roster
layer -> features -> models -> predictions (2026 GWs 1-8) -> MILP optimizer ->
real GW1 recommendation -> FastAPI -> dashboard**. Every rules assumption was
re-verified against the launch bootstrap (scoring identical, chips BB/TC GW1-19 +
WC/FH GW2-19 + set 2 GW20-38, 5-FT bank, transfers cap 20, 10/11 position
reclassifications confirmed). The dashboard ticker shows LIVE 2026/27 with the
real deadline.

```
742 offline tests pass    (backend: uv run pytest -q -m "not live")
ruff clean                (uv run ruff check src tests)
eslint clean + prod build (frontend: npm run lint && npm run build)
```

## Commands

From the repo root (Makefile):

| Command | What it does |
|---|---|
| `make dev` | API (uvicorn :8000) + dashboard (next dev :3000) together |
| `make api` / `make web` | Either half on its own |
| `make refresh` | Full daily cycle: snapshot -> pulls -> build -> **live predict (2026 GWs 1-8)** -> **live optimize** (exit-0) |
| `make train` | Retrain all model components (~70 s) |
| `make predict` / `make optimize` | Live-mode stages (now the real thing) |
| `make backtest [SEASON=2025] [GWS=30..38]` | Walk-forward policy backtest |
| `make demo [SEASON=2025 GW=34]` | Backtest-window chain (still available for evals) |
| `make test` / `make test-live` / `make lint` | Offline suite / live-marked tests / linters |

From `backend/` the full flag surface is unchanged (`uv run fplai
snapshot|backfill|build|train|predict|optimize|backtest|refresh`); see
`fplai <cmd> --help`. `predict`/`optimize` with no `--season/--gw` are live mode.

## LIVE mode (what ran on launch day, 2026-07-23)

`uv run fplai refresh` end to end against the real API:

- **snapshot**: day-1 archive `data/raw/fpl_api/snapshots/2026-07-23/` —
  season 2026, next_gw 1, deadline 2026-08-21 17:30 UTC, 223,802 managers.
- **predict (live)**: bootstrap-grounded rosters via `fplai.data.live` — 555
  rostered players, 80 upcoming fixtures over GWs 1-8, 4,432 player-fixture rows.
  Applied: 3 promoted clubs seeded (COV 1.08 > IPS 1.00 > HUL 0.92), 9
  new-manager clubs shrunk toward the mean, 63 cold-start players on
  position×price priors, WC2026 fatigue dampening for 36 matched players (GWs
  1-4; Stones/Konaté not in the day-1 pool — auto-match when they appear),
  availability gating from the snapshot flags, 10 GW1 fixtures blended with a
  cached the-odds-api snapshot (pre-deadline odds, archived under
  `raw/odds_api/`). Top GW1 xP: Haaland 8.14, Thiago 7.62, B.Fernandes 6.22,
  Gibbs-White 6.16, Watkins 5.86, Gabriel 5.68.
- **optimize (live)**: None-state initial-squad build, season 2026 GW1, horizon
  8, 554-player pool at real launch prices (live_roster.parquet is the price
  source), full chips + 30 stability re-solves. Main solve 50 s, gap 4.1%.

### The real GW1 verdict (2026-27, fresh £100.0m)

**80.1 xP in GW1** (Bench Boost played — all 15 count), £99.5m spent, £0.5m ITB,
formation BB · 5-5-3, captain **Thiago** (BRE, home v TOT), vice Gibbs-White:

- GKP: Raya (ARS 6.0), Lammens (MUN 5.0)
- DEF: Gabriel (ARS 8.0), Tarkowski (EVE 6.0), Mukiele (SUN 5.5),
  N.Williams (NFO 5.0), Shaw (MUN 4.5)
- MID: Gibbs-White (NFO 8.0, V), Mbeumo (MUN 8.0), Szoboszlai (LIV 7.0),
  Gakpo (LIV 7.0), E.Le Fée (SUN 6.0)
- FWD: Thiago (BRE 8.0, C), Watkins (AVL 8.0), João Pedro (CHE 7.5)

The draft leans into promoted-opponent fixtures (MUN trio at HUL, SUN pair at
IPS, ARS pair v COV, NFO pair v LEE) and skips £15.5m Haaland at GW1 in favour
of spread + a planned GW6 restructure. 8-GW plan: **BB GW1 -> WC GW2 -> TC GW3
(Watkins) -> FH GW5 -> banked 5 FTs buy Haaland at GW6** — zero hits, objective
253.2. Chip EVs: bb1 GW1 +11.1, tc1 +7.6, wc1 best GW2 +4.1, fh1 best GW2 +2.0
(WC/FH GW1 correctly banned by `rules.chip_windows(2026)`). Artifacts:
`recommendation.json`, `dream_team.json`, `chip_curves.parquet` (32 rows,
4 chips × GWs 1-8) in `data/processed/`, all season-2026-GW1 stamped.
(Greedy-chip caveat applies — the horizon ends at GW8, so chips cluster early;
re-run weekly as prices/news move. A human may prefer holding BB/TC.)

## API + dashboard (verified live 2026-07-23)

- `GET /api/state`: `pre_launch: false`, season 2026, next_gw 1, real deadline
  countdown, predictions_season 2026 GWs [1..8], fresh artifact mtimes.
- `GET /api/predictions?season=2026&gw=1`: 554 players at launch prices.
- `GET /api/recommendation` + `GET /api/dream-team`: the GW1 build above.
- `GET /api/chip-curves`: the 32-row live curve set.
- Dashboard ticker flips to LIVE 2026/27 automatically (state-driven); the
  mtime-invalidated caches picked the new artifacts up without a restart.

## Launch-week checklist — DONE 2026-07-23 (was FPL_KNOWLEDGE §3.8)

1. ~~Re-verify rules~~ **DONE** — full §1.12 register resolved against the day-1
   bootstrap; scoring dict identical to 2025-26; chips exactly as presumed
   (`chip_windows(2026)` needed no change); GW1 deadline confirmed; transfers
   cap 20 confirmed (`rules.TRANSFERS_CAP_PER_GW`); 10/11 reclassifications
   confirmed by `element.code` (Eric da Silva Moreira absent from the day-1
   pool — `rules.RECLASSIFIED_ABSENT_AT_LAUNCH_2026`). Haaland launched £15.5m
   (above the expected band), Bruno £12.0m; **Salah is not in the 2026-27 game**.
2. ~~Ingest launch data~~ **DONE** — daily `fplai refresh` archives prices,
   status/news/chance and selected_by; day-1 snapshot is the baseline.
3. ~~Rosters for GW1~~ **DONE** — `fplai.data.live` grounds the prediction pool
   in the bootstrap roster (movers keep history by `player_code`, departed
   players never resurrected, cold-start priors for the 63 unseen codes).
4. ~~Re-point~~ **DONE** — models trained on seasons ≤2025 predict 2026 (exactly
   right); `fplai refresh` serves live predictions/recommendations; ticker live.
   Optional: retrain after a few 2026 GWs of data exist.
5. ~~Projection adjustments~~ **DONE** — WC-fatigue dampening (36 players, GWs
   1-4) and new-manager shrink applied; re-pull transfer trackers weekly until
   1 Sep (squads move — re-run `fplai refresh` daily).
6. ~~Odds~~ **DONE** — pre-deadline odds snapshots cached daily under
   `raw/odds_api/` (~2 credits/day); GW1 blended from today's snapshot.

## Real end-to-end verification

- 2026-07-23 (launch day): offline suite **742 passed** (+25 live-layer, +3
  rules-launch pins), ruff clean; real live chain ran to completion (above);
  API routes verified against the running server on :8010.
- 2026-07-22 (pre-launch): frontend `npm run build` + eslint clean, zero
  browser-console errors across all four pages; walk-forward xP accuracy
  (train < GW30, eval GWs 30-38 of 2025-26): Zeros RMSE/MAE **0.798/0.318**
  (beats OpenFPL's published 0.818/0.427), within-GW Spearman 0.726; policy
  backtest GWs 33-35: model 249 pts vs last-5 223 vs set-and-forget 187.

## Model artifacts (data/models/)

Trained 2026-07-22 15:02 UTC on all 10 seasons (253,568 player-fixture rows,
3,800 matches) — i.e. strictly pre-2026 data, which is exactly the right
train/deploy split for predicting 2026-27. Live-layer adjustments (promoted
seeding, new-manager shrink, fatigue, cold-start priors) are applied at predict
time in memory; artifacts on disk are untouched.

## Data coverage

- Processed tables: 10 historical seasons 2016-17..2025-26 plus the live 2026
  splice (fixtures 3,800 + 380; teams 200 + 20 rows; players extended with the
  63 new codes). `player_match`/`player_gw` stay history-only; future 2026 rows
  are synthesized per-predict from the bootstrap roster.
- rules.py: scoring 2016..2026 (launch-confirmed), season flags, chip windows
  (launch-confirmed), BPS v4, sell-price, FT arithmetic, transfers cap,
  reclassifications with player codes.
- Understat per-player JSON backfill for 2016-18/2024/2025 still filling in the
  background; re-run `fplai build` to ingest.

## Known gaps / caveats

1. **Optimize latency**: full verdict (32 chip re-solves + 30 stability
   re-solves) took ~35 min on the 554-player × 8-GW live pool;
   `--no-chips --no-stability` gives a ~1 min verdict. `refresh` uses full
   defaults — fine for a daily cron, mind it for interactive runs.
2. **Squads move until 1 Sep**: Salah/Stones/Konaté/Gordon/Cucurella absent
   from the day-1 pool may yet appear (FPL adds players mid-window); Eric da
   Silva Moreira's reclassification is announcement-only until he has an
   element. Daily `fplai refresh` picks all of this up automatically — re-check
   before the GW1 deadline.
3. **Cold-start xP is deliberately conservative** (promoted starters ~1-3 xP):
   position×price-decile priors carry no Championship data; promoted-strength
   multipliers (COV 1.08/IPS 1.00/HUL 0.92) and NEW_MANAGER_SHRINK=0.15 are
   documented judgment constants.
4. **Availability gating is a linear 4-GW recovery heuristic** from the
   snapshot flag; news return-date strings are not parsed — flagged players'
   GW2+ xP may be over/under-stated (Saliba and Rodri are status-i today and
   correctly zeroed for GW1).
5. **`fplai build` alone temporarily drops the 2026 splice** (it rebuilds from
   raw, which has no 2026 rows) — the next `fplai predict` re-splices;
   `refresh`'s build->predict->optimize order handles it.
6. **Chip policy is greedy within the horizon** (no opportunity cost beyond
   GW8) — hence BB/WC/TC/FH all inside GWs 1-5; raise `chip_play_margin` or
   `evaluate_chips=False` to tame it.
7. **Odds cover only the events bookmakers have posted** (GW1 today); later GWs
   are pure Dixon-Coles until lines appear — by design, and the pre-deadline
   snapshot archive fixes the old closing-line leakage caveat from GW1 onward.
8. **`/api/my-team/{entry_id}`** becomes fully real once entries exist for
   2026-27 (post-launch squads); pre-deadline picks are not exposed by the FPL
   API — the endpoint falls back loudly to the initial-squad verdict until then.
9. **Blanks (1-2 pts)** remain our weakest xP category vs OpenFPL; revisit with
   team-partitioned CV. Understat backfill (data coverage note) still running.
