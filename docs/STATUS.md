# STATUS.md — what works today

Last updated: 2026-08-11 (pre-GW1 research + upgrade pass — see "August 2026
upgrade pass" below; research synthesis in `research/aug-2026-update.md`).
Previous pass 2026-07-24 (public-release integrator pass: static publisher +
dual-mode frontend + GitHub Actions release engineering verified end to end —
see "Public release engineering" below). Before that 2026-07-23 (stage-6:
the Monte Carlo season-simulation
chip planner is live — predictions extend through GW19, `fplai simulate` prices every
chip week over the FULL set-1 window with rollout-level uncertainty, and the old
greedy chip cascade is re-verdicted with probabilities; earlier the same day the
2026-27 game went LIVE and the full chain ran against the real day-1 API).
Companions: `ARCHITECTURE.md` (module map + schemas incl. the stage-6 contracts),
`FPL_KNOWLEDGE.md` (game rules — §1.12 uncertainty register fully resolved at
launch), `MODEL_DESIGN_INPUTS.md` (model spec), `research/chip-strategy-verdict.md`
(evidence review of the launch-day chip cascade — its recommendations #1/#2 are what
`fplai simulate` implements).

## August 2026 upgrade pass — COMPLETE (2026-08-11, 10 days before GW1)

Deep research (4 parallel agents: season state / modeling SOTA / community-tool
feature mining / data-source audit — synthesis in `research/aug-2026-update.md`)
followed by an implementation pass. Verified: **886 offline tests pass** (was
855; +ingest/availability/captaincy/publish/API suites), ruff clean, eslint
clean, both frontend build flavours green + rating parity suite, real live
chain re-run against the 2026-08-11 snapshot (577 elements) and
`publish-static` produced the extended 11-file bundle at **556,344 B (28% of
budget)**.

1. **In-season outcome ingestion** (`fplai.data.ingest`, `fplai ingest`,
   wired into refresh + model-run.yml between build and predict): played
   live-season GWs are spliced into player_match/player_gw from
   `element-summary` history rows (DGW-safe, carries per-GW price, BPS, starts,
   DefCon components). Freeze bookkeeping via bootstrap `data_checked` +
   `ingest_state.json` — zero API requests between GWs. **This closed a
   season-critical gap**: build is vaastav-only and vaastav ended weekly
   updates after 2024-25, so form features would have starved from GW2.
2. **Availability v2** (`fplai.data.live`): FPL news strings parsed for dated
   returns ("Expected back 23 Aug" / "Suspended until 6 Sep" — the live grammar
   is a small closed vocabulary, 63 flagged players on 2026-08-11, 10 dated).
   Per-(player, GW) gates from the player's OWN team's kickoffs: suspensions
   hard-zero until the return GW then instantly 1.0 (a ban is not a fitness
   state); dated injuries floored until the return GW with a 0.65 comeback-GW
   ramp; undated news keeps the linear 4-GW heuristic. Persisted to
   `availability_{season}.json`; return GW published per player.
3. **Set-piece duties**: `penalties_order`/`direct_freekicks_order`/
   `corners_and_indirect_freekicks_order` (+ notes, + `news_added`,
   + `price_change_percent`) parsed from the bootstrap into live_roster —
   130 duty holders published in `set_pieces.json`; P-badges on the players
   table.
4. **Team fixture outlook**: `team_fixtures.parquet` persisted from the
   TeamModel's per-fixture predictions (λ for/against, CS%, 1X2, odds-blend
   flag) → `fixtures.json` → new **Fixtures page**: model-based difficulty
   ticker (attack/defence/overall lenses, GW-range filter, sort-by-best-run,
   CVD-safe diverging ramp) — our own Dixon-Coles difficulty, not official FDR.
5. **Distributional captaincy** (`pipeline._captaincy_frame` → 
   `captaincy.parquet` → `captaincy.json`): 2,000 joint sampler rollouts for
   the top-8 candidates — P(haul ≥10), P(blank ≤2), P(best, ties split),
   P(beats top pick) — dashboard card. Real GW1 output: Haaland 8.22 xP,
   P(haul) 0.33, P(best) 0.23; Thiago beats him in 49.6% of rollouts with a
   quarter of the blank risk.
6. **Ownership surfaces**: `selected_by_percent` → players.json → OWN% column
   + differential quick-filter (<15% owned, fit, ranked by xP per ownership
   point) on the players page.
7. **Solver tail guard**: `SolveParams.no_transfer_last_gws` (chip-exempt MILP
   constraint; research follow-up #1) — live optimize runs with tail=2 when
   horizon ≥ 4.
8. **Latent bug fixed**: football-data's pre-publication 301 (2627/E0.csv →
   EC.csv, the National League) would have been silently ingested as PL odds;
   downloads now assert division E0 on data rows.
9. **API parity**: GET /api/fixtures-outlook, /api/set-pieces, /api/captaincy
   serve the bundle shapes in local mode (publisher record-builders reused).

Season facts locked by the research pass (see `research/aug-2026-update.md`):
no rule/API changes since launch; Salah/Konaté/Cucurella/Gordon/Stones all left
the PL permanently (none will be added); pool 555→577; merged 3-week
international break GW5→GW6 (26 Sep–10 Oct) confirmed; no AFCON, no announced
FT top-up event; BB/TC playable GW1, WC/FH from GW2; official app now ships a
price predictor + live ranks + projected bonus (deliberately NOT rebuilt here).

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

The stage-6 layer on top: **predictions now cover GWs 1-19** (the set-1 chip
window), `fplai simulate` runs a 1,000-rollout Monte Carlo chip-timing
simulation (112 s), and the recommendation's chip advice speaks probability
("BB now beats holding in 12% of 1,000 simulated seasons — hold") instead of
greedy in-horizon deltas. See "Season simulation" below.

```
886 offline tests pass    (backend: uv run pytest -q -m "not live")
ruff clean                (uv run ruff check src tests)
eslint clean + prod build (frontend: npm run lint && npm run build — default AND
                           NEXT_PUBLIC_STATIC=1 static-export flavours)
```

## Public release engineering — COMPLETE (2026-07-24)

The $0/month all-GitHub release architecture is built and verified end to end
(nothing has been pushed — the maintainer publishes):

- **Static publisher**: `fplai publish-static [--out DIR]` writes the 8-file
  contract bundle to `site-data/` (gitignored) — meta / players / xp /
  predictions_gw1 / recommendation / dream_team / chip_curves / rating JSON +
  empty `history/`. Real-data bundle: **440,991 B total (22% of the 2 MB
  budget)**, byte-identical across repeat builds, rating anchors
  (floor 109.708 / optimal 571.491, GW1..19) exactly match the Python engine.
- **Dual-mode frontend**: default local mode untouched; `NEXT_PUBLIC_STATIC=1`
  + `NEXT_PUBLIC_BASE_PATH` produce a 562-page static export whose AI-Rating
  runs client-side (`lib/rating.ts`, TS port of `fplai/optimizer/rating.py`;
  parity suite `npm run test:unit`: 25/25 real-data squads match the Python
  engine to 1e-13). Public build hides entry/refresh/my-team surfaces and
  carries the unofficial-tool disclaimer + MIT/GitHub footer.
- **Workflows** (`.github/workflows/`, all actionlint-clean; helper scripts
  shellcheck-clean): `ci.yml`; `model-run.yml` (daily 05:30 UTC cron +
  dispatch; snapshot -> build -> predict -> optimize -> simulate ->
  publish-static; data/ via Actions cache with marker-file self-heal);
  `deploy-pages.yml` (deploys the export + bundle to GitHub Pages on
  model-run success or push to main); `deadline-watch.yml` (hourly, triggers
  an extra full run ~3 h before each GW deadline).
- **Verified this pass**: 855 offline tests + ruff clean; eslint clean; both
  build flavours green; served the basePath export GH-Pages-style and proved
  every page (dashboard/players/player-detail/planner/rating/settings) and all
  8 bundle files load with zero console errors, the disclaimer footer is in
  every page's HTML, and local-only surfaces (entry card, wiring, refresh) are
  absent from the static build's rendered output; no secrets in tracked files
  (the odds key lives only in gitignored `.env`).

**Maintainer's next steps (in order):**

1. Create the **public** GitHub repo `bjarkiSigur7/fpl-ai-assistant` (public =
   free Actions minutes) and push `main`.
2. Repo *Settings -> Secrets and variables -> Actions*: add secret
   `FPLAI_ODDS_API_KEY` (optional but recommended; odds degrade gracefully
   without it). Optional repo **variables**: `FPLAI_SEASON` (default 2026),
   `FPLAI_DATA_CACHE_VERSION` (bump to force a data re-bootstrap),
   `PAGES_BASE_PATH` (set to `/` at domain cutover; default
   `/fpl-ai-assistant`).
3. *Settings -> Pages*: set **Source = GitHub Actions** (enables the
   deploy-pages workflow).
4. Trigger the first `model-run` (*Actions -> model-run -> Run workflow*) or
   wait for the 05:30 UTC cron. **First run hits a cold Actions cache: the
   marker-file self-heal bootstraps backfill -> build -> train, adding ~40 min
   on top of the normal batch — expect it and don't cancel.** Subsequent runs
   restore the cache and skip the bootstrap.
5. On model-run success, deploy-pages publishes automatically to
   `bjarkisigur7.github.io/fpl-ai-assistant`. (A push to `main` before any
   successful model-run deploys the site data-less with loud empty states —
   intentional first-boot behaviour.)
6. Add real screenshots at `docs/screenshots/{dashboard,planner,rating}.png`
   (the README references them; they render as broken images until added).

Operational runbook: `docs/PUBLIC_RELEASE.md` (cadence, cache self-heal,
archive branch, key rotation, domain cutover, incident playbook).

## Commands

From the repo root (Makefile):

| Command | What it does |
|---|---|
| `make dev` | API (uvicorn :8000) + dashboard (next dev :3000) together |
| `make api` / `make web` | Either half on its own |
| `make refresh` | Full daily cycle: snapshot -> pulls -> build -> **live predict (through the chip-window end, GW19)** -> **live optimize** -> **simulate** (exit-0) |
| `make train` | Retrain all model components (~70 s) |
| `make predict` / `make optimize` | Live-mode stages (now the real thing) |
| `make simulate` | Monte Carlo chip-timing simulation over the full chip window (~2 min) |
| `make backtest [SEASON=2025] [GWS=30..38]` | Walk-forward policy backtest |
| `make demo [SEASON=2025 GW=34]` | Backtest-window chain (still available for evals) |
| `make test` / `make test-live` / `make lint` | Offline suite / live-marked tests / linters |

From `backend/` the full flag surface (`uv run fplai
snapshot|backfill|build|train|predict|optimize|simulate|backtest|refresh`); see
`fplai <cmd> --help`. `predict`/`optimize` with no `--season/--gw` are live mode.
New stage-6 surface: `fplai predict --through-gw 19` (extend live predictions to
the chip-window end; `refresh` does this automatically in live mode) and
`fplai simulate [--rollouts N] [--seed S] [--through-gw N] [--entry-id E]`.

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
253.2. The plan's greedy chip EVs (bb1 GW1 +11.1, tc1 +7.6, wc1 GW2 +4.1, fh1
GW2 +2.0, decayed and horizon-truncated) are now **superseded by the season
simulation below**, which re-verdicts every set-1 chip to HOLD at GW1.
Artifacts: `recommendation.json` (chip advice sim-folded), `dream_team.json`,
`chip_curves.parquet` (v2: 76 rows, 4 chips × GWs 1-19 with MC probability
columns), `chip_sim.parquet` + `chip_sim_report.json` in `data/processed/`,
all season-2026-GW1 stamped. Re-run weekly as prices/news move.

## Season simulation (stage 6) — verified live 2026-07-23

`fplai simulate --rollouts 1000` on the real GW1 pool (554 players, window GWs
1-19, seed 0): **112 s wall** — no-chip backbone MILP over all 19 GWs solved to
gap 0.10% in 80 s (status optimal, objective 277.6), 54 bounded WC/FH segment
re-solves, `PointsSampler` rollouts with appearance from jointly-sampled
minutes (`sample_draws`). Deterministic under seed.

**The simulation's chip verdicts vs the greedy cascade** (E[gain] is undecayed
realized points over the full window; the greedy numbers were decayed deltas
inside GWs 1-8):

| chip | greedy plan | sim verdict | best window | E[gain] ± sd | P(best week) | P(play GW1 beats holding) |
|---|---|---|---|---|---|---|
| bb1 | play GW1 (+11.1) | **hold** | GW1 (flat: GW2/GW3 within 0.2) | +6.6 ± 4.5 | 0.10 | 0.12 |
| tc1 | play GW3 (+7.6) | **hold** | GW1 (flat) | +7.8 ± 4.6 | 0.10 | 0.12 |
| wc1 | play GW2 (+4.1) | **hold** | **GW5** | +6.9 ± 22.2 | 0.14 | n/a (GW1 outside window) |
| fh1 | play GW5 (+2.0) | **hold** | **GW10** | +6.0 ± 14.3 | 0.12 | n/a (GW1 outside window) |

Top joint schedule: tc1@GW1, bb1@GW3, wc1@GW5, fh1@GW10 (E +27.1, P(best)
0.02 — the timing surface is genuinely flat, exactly what the evidence review
predicted for a DGW-less half). The launch-day cascade (all four chips by GW5)
is not the sim's answer: WC moves to the GW5/6 international-break slot and FH
deep into the window as insurance, matching `research/chip-timing-evidence.md`
doctrine. BB/TC GW1 remain the E[gain]-argmax weeks — the research's "BB GW1 is
CONTESTED, not artifact" nuance survives — but with P(best week) ≈ 0.10 and
neighbouring weeks within a fraction of a point, playing them is a coin-flip
convenience, not an edge; the +11.1 greedy BB number was decay-inflated (the
undecayed MC estimate is +6.6, below the community's 8-10 prior).

Honest semantics + assumptions (all surfaced in
`ChipSimReport.assumptions` and the API): `p_beats_hold` compares playing now
against the *realized best later week* — an option-value upper bound for
holding (a real manager cannot time the future max), so it is biased toward
"hold"; the E[gain] column is the fair point estimate. WC/FH gains are bounded
segment re-solves (squad value carried past the segment is not credited —
lower bounds). One fixed no-chip backbone; chips are scored independently
(joint schedules sum per-chip gains); prices/news/predictions are frozen at
today's snapshot; GW9-19 fixtures are pure Dixon-Coles (no odds posted yet).
The backbone honesty block (`backbone_quality`) reports objective/gap/status.

## API + dashboard (verified live 2026-07-23)

- `GET /api/state`: `pre_launch: false`, season 2026, next_gw 1, real deadline
  countdown, predictions_season 2026 GWs [1..8], fresh artifact mtimes.
- `GET /api/predictions?season=2026&gw=1`: 554 players at launch prices.
- `GET /api/recommendation` + `GET /api/dream-team`: the GW1 build above, chip
  advice carrying the sim fields (verdict/p_beats_hold/recommended_gw/
  confidence/n_rollouts) and probability rationale bullets.
- `GET /api/chip-curves`: the 76-row v2 curve set (4 chips × GWs 1-19 with
  sd/p_best_week/p_beats_hold/n_rollouts) — the dashboard planner's whisker +
  P(best)-marker source. Both verified against the running server post-sim
  (the pre-stage-6 uvicorn had to be restarted to pick up the v2 schema).
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

- 2026-07-23 (stage 6): offline suite **813 passed** (742 launch baseline +
  17 sampler + 28 season_sim + 15 sim-advice/API + 11 pipeline-wiring), ruff
  clean; `fplai predict --through-gw 19` produced 10,526 player-fixture rows
  (GWs 1-19, 190 fixtures) in ~3 s; `fplai simulate --rollouts 1000` completed
  in 112 s and folded its verdicts into recommendation.json; `fplai optimize`
  re-ran with the sim report folded at build time; API v2 fields verified by
  curl against the restarted server on :8000.
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
   defaults — fine for a daily cron, mind it for interactive runs. The
   simulate stage adds only ~2 min on top.
2. **Squads move until 1 Sep**: Salah/Stones/Konaté/Gordon/Cucurella absent
   from the day-1 pool may yet appear (FPL adds players mid-window); Eric da
   Silva Moreira's reclassification is announcement-only until he has an
   element. Daily `fplai refresh` picks all of this up automatically — re-check
   before the GW1 deadline.
3. **Cold-start xP is deliberately conservative** (promoted starters ~1-3 xP):
   position×price-decile priors carry no Championship data; promoted-strength
   multipliers (COV 1.08/IPS 1.00/HUL 0.92) and NEW_MANAGER_SHRINK=0.15 are
   documented judgment constants.
4. **Availability gating v2 (2026-08-11)**: dated news ("Expected back D Mon",
   "Suspended until D Mon") now gates per (player, GW) against the player's
   own team's kickoffs — suspensions hard-zero then instantly back, dated
   injuries floored with a comeback ramp. Remaining heuristic: UNDATED news
   ("Unknown return date", e.g. Rodri/Saliba today) still uses the linear 4-GW
   recovery — a status-transition hazard model (FPL-Core-Insights git history
   is a free training corpus) is the documented next step.
5. **`fplai build` alone temporarily drops the 2026 splice** (it rebuilds from
   raw, which has no 2026 rows) — the next `fplai predict` re-splices;
   `refresh`'s build->predict->optimize order handles it.
6. **Greedy-chip-within-horizon: FIXED at the advice layer** — `fplai simulate`
   prices every chip week over the full GW1-19 window undecayed with MC
   uncertainty, and its verdicts override the plan's chip EVs in
   `recommendation.json`/the dashboard. Remaining honesty notes: the MILP
   *plan* itself still schedules chips greedily inside its 8-GW horizon (the
   plan trajectory shows BB GW1 etc.; read the chip_advice verdicts, not the
   plan, for timing), WC/FH sim gains are segment-local lower bounds, and
   `p_beats_hold` is an option-value upper bound for holding (see the Season
   simulation section). Follow-ups from `research/chip-strategy-verdict.md`
   not yet implemented: transfer-depth tail (`no_transfer_last_gws≈2` needs a
   new MILP constraint), decay 0.84→0.90 sensitivity run, chip-timing
   guardrail warnings, EO/field-risk annotation, feeding sim-chosen chip weeks
   back into the transfer solve as constraints.
7. **Odds cover only the events bookmakers have posted** (GW1 today); later GWs
   are pure Dixon-Coles until lines appear — by design, and the pre-deadline
   snapshot archive fixes the old closing-line leakage caveat from GW1 onward.
8. **`/api/my-team/{entry_id}`** becomes fully real once entries exist for
   2026-27 (post-launch squads); pre-deadline picks are not exposed by the FPL
   API — the endpoint falls back loudly to the initial-squad verdict until then.
9. **Blanks (1-2 pts)** remain our weakest xP category vs OpenFPL; revisit with
   team-partitioned CV. Understat backfill (data coverage note) still running.
