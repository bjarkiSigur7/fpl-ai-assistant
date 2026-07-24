# PUBLIC_RELEASE.md — operating the public PITCHSIDE site

The public site is a static Next.js export on **GitHub Pages**, fed by a JSON
bundle that scheduled **GitHub Actions** runs regenerate. There is no server,
no database, and no paid service anywhere in the loop. This file is the
operator's runbook. The workflow YAML under `.github/workflows/` is the
authoritative source for schedules, job names, and branch names — this document
explains the design and what to do when something breaks.

## 1. What runs where

| Piece | Where | Cost |
|---|---|---|
| Model batch (`fplai refresh` + `fplai publish-static`) | GitHub Actions, scheduled | $0 (public repo = free standard-runner minutes) |
| Static site (`next build` export) | GitHub Actions → GitHub Pages | $0 |
| Data bundle `site-data/` (<2 MB) | Baked into the Pages deploy; archived on the archive branch | $0 |
| Bookmaker odds | the-odds-api.com free tier via `FPLAI_ODDS_API_KEY` secret | $0 (free tier) |
| Everything personalized (my-team plans, refresh button, FastAPI) | **Local only** — never deployed | n/a |

## 2. Actions cadence

Two schedules:

1. **Daily full run.** Executes the same chain as `make refresh` — snapshot →
   pulls → build → predict (through the chip-window end, currently GW19) →
   optimize → simulate — then `fplai publish-static --out site-data`, then the
   static site build and Pages deploy. Budget ~40-60 minutes per run: the full
   optimize verdict (chip re-solves + stability re-solves) took ~35 min on the
   554-player live pool (STATUS.md), simulate adds ~2 min. `refresh` is exit-0
   data-safe by design: a failed pull degrades gracefully rather than
   publishing a broken bundle.
2. **Hourly deadline watch.** A cheap job (one FPL API request: read
   `next_deadline_utc` from bootstrap) that triggers an *extra* full run when
   the next GW deadline is ~3 hours away, so the published verdict reflects
   the final press-conference news and pre-deadline odds snapshot. It must
   stay idempotent: if a triggered run already happened inside the window, it
   exits 0 without dispatching another.

Retraining (`fplai train`, ~70 s) is **not** part of refresh and not scheduled;
the maintainer retrains deliberately after enough 2026-27 GWs exist and commits
to a fresh cache (see §3). Models trained on seasons ≤2025 predicting 2026 is
the correct deploy split at launch.

## 3. Cache self-heal

The batch job restores `data/` (raw snapshots, processed parquet, model
artifacts, HTTP cache) from `actions/cache` between runs. GitHub evicts caches
(unused-7-days rule, 10 GB per-repo cap), so the pipeline is designed to
survive a cold start:

- **Cache hit**: refresh is incremental — one snapshot, delta pulls, fast run.
- **Cache miss**: the job re-bootstraps from public sources — `fplai backfill`
  (10 seasons of history), `fplai build`, `fplai train` — then proceeds as
  normal and re-primes the cache. The run is slower and heavier on upstream
  APIs (the clients keep their 1 req/s throttles regardless), but it succeeds
  with **zero operator action**. Nothing lives in the cache that cannot be
  regenerated from public sources.
- **Manual heal**: if the cache is corrupted (e.g. a half-written parquet from
  a cancelled run), delete the cache entries in *Actions → Caches* and re-run
  the daily workflow. That's the whole procedure.

One genuine loss on a cold start: the day-by-day `raw/fpl_api/snapshots/`
archive accumulated in the cache. That is why snapshots are also pushed to the
archive branch (§4) — the cache is a performance layer, the branch is the
record.

## 4. Archive branch

Each daily run commits its outputs to a dedicated archive branch (kept out of
`main` so the code history stays clean; see the workflow YAML for the branch
name). It carries the dated `site-data/` bundles (the bundle's optional
`history/` directory) and the day's API snapshot. Purpose:

- **Time series** of model outputs for later evaluation (what did we predict
  before GW n?) — this is the walk-forward audit trail.
- **Disaster recovery** for the snapshot archive when the Actions cache is
  evicted (§3).

The branch grows by a few hundred KB per day at 3-decimal float rounding.
Prune policy: none needed short-term; if it ever bothers you, squash history
older than a season into a single commit.

## 5. Secrets and odds-key rotation

Exactly one secret exists: `FPLAI_ODDS_API_KEY` (the-odds-api.com). Everything
else the pipeline touches is keyless and public.

Rotation (also the compromise-response procedure):

1. Log in at the-odds-api.com → regenerate/obtain a new key (free tier: 500
   credits/month; our usage is a daily `h2h+totals` snapshot ≈ 2 credits/day,
   ≤150 credits/month even with the optional deadline-morning anytime-scorer
   sweep — see MODEL_DESIGN_INPUTS.md §5.2).
2. Repo → *Settings → Secrets and variables → Actions* → update
   `FPLAI_ODDS_API_KEY`. No code change, no redeploy needed; the next
   scheduled run picks it up.
3. Locally, update `.env` (gitignored).

Failure mode is soft everywhere: with no key (or exhausted credits) the odds
client logs a skip and the team model runs on pure Dixon-Coles + cached
football-data closing odds. Never put the key in workflow YAML, bundle output,
or frontend env — the bundle and the site are public.

## 6. Custom-domain cutover (config-only, by design)

The site lives at `bjarkisigur7.github.io/fpl-ai-assistant` until a domain is
bought. The Next.js `basePath` is driven by `NEXT_PUBLIC_BASE_PATH`, so the
cutover touches zero code:

1. **DNS** (at the registrar): for `www.<domain>` add a `CNAME` record →
   `bjarkisigur7.github.io`. For an apex domain add `A`/`AAAA` records to
   GitHub Pages' published IPs (see GitHub's Pages custom-domain docs for the
   current list).
2. **Repo**: *Settings → Pages → Custom domain* → enter the domain. Because we
   deploy via Actions, also ensure a `CNAME` file containing the domain ends
   up in the exported site root on every deploy (add it to `frontend/public/`)
   — otherwise each deploy drops the domain binding.
3. **Base path**: set the repo Actions variable `NEXT_PUBLIC_BASE_PATH=""`
   (empty — the site now lives at the domain root instead of
   `/fpl-ai-assistant`). This is the only build-config change.
4. Wait for the certificate to provision, then tick **Enforce HTTPS**.

Rollback is the mirror image: remove the custom domain and set
`NEXT_PUBLIC_BASE_PATH="/fpl-ai-assistant"` back.

## 7. Local-only vs public feature matrix

The public build consumes the static `site-data/` bundle; the local build talks
to the FastAPI backend. Features:

| Feature | Public site | Local (`make dev`) |
|---|---|---|
| Weekly verdict, dream team, plan timeline | Yes (daily bundle) | Yes (live artifacts) |
| Player explorer + next-GW xP component breakdown | Yes (`players.json`, `xp.json`, `predictions_gw1.json`) | Yes (API) |
| Chip planner with Monte Carlo curves (sd, P(best week), P(beats hold)) | Yes (`chip_curves.json`) | Yes (API) |
| AI-Rating squad rater (0-100 vs floor/optimal benchmarks) | Yes — **computed client-side**, same formula as the Python engine, constants from `rating.json` | Yes (`POST /api/rate-team`) |
| Deadline countdown / freshness | Yes (`meta.json`: `generated_utc`, `next_deadline_utc`) | Yes (live `/api/state`) |
| Personalized my-team plan (`entry_id`) | **No — hidden** | Yes (`/api/my-team/{entry_id}`) |
| Refresh button (re-run the pipeline on demand) | **No — hidden** | Yes (`POST /api/refresh`) |
| FastAPI / uvicorn | Not deployed | :8000 |

The rule of thumb: anything requiring compute-on-request or a user identity is
local-only; anything that is a pure function of the daily bundle is public.

## 8. Cost model

Total: **$0/month** (plus the optional domain, ~$10-15/year, when bought).

| Service | Free-tier limit | Our usage | Headroom |
|---|---|---|---|
| GitHub Actions | Free unlimited standard-runner minutes for public repos (per GitHub's published billing docs); 20 concurrent jobs | ~1 daily run of ~40-60 min + hourly seconds-long watch + deadline-window extras | Effectively unbounded; watch job concurrency, not minutes |
| GitHub Pages | 1 GB site, 100 GB/month soft bandwidth, ~10 builds/hour soft limit (GitHub's published Pages limits) | Site + bundle ≈ a few MB; ≤ ~2 deploys/day | Vast |
| Actions cache | 10 GB per repo, 7-day unused eviction | `data/` raw + processed + models (single-digit GB) | §3 makes eviction a non-event |
| the-odds-api.com | 500 credits/month | ≈60/month daily snapshots; ≤150 with deadline sweeps (MODEL_DESIGN_INPUTS §5.2, STATUS launch log ~2/day) | ≥3× |
| FPL API, football-data.co.uk, Understat, ClubElo | No keys; unofficial etiquette limits (≤1 req/s FPL, ~1 req/2 s Understat — docs/research/data-sources.md) | One snapshot/delta pull per day | Fine while throttles are respected |

If GitHub ever changes public-repo Actions pricing, the entire batch also runs
on any machine with `make refresh` — the site is just files.

## 9. Things that page a human

- **Daily run red for >24 h**: the site silently serves yesterday's bundle
  (it keeps working — `meta.json.generated_utc` is the tell, and the frontend
  surfaces staleness). Check the Actions log; the usual suspects are an
  upstream API schema change or an evicted-cache rebuild hitting a flaky
  source.
- **FPL API schema change**: the offline test suite pins the contracts;
  reproduce locally with `make refresh`, fix, and the next scheduled run
  self-heals.
- **Squad-moving window (until 1 Sep)**: launch-absent players (see STATUS.md
  known gaps) appear via the daily refresh automatically — no action, just
  awareness.
- **Season rollover**: chip windows, scoring, and rules constants live in
  `backend/src/fplai/rules.py` and are launch-verified each season per the
  FPL_KNOWLEDGE checklist.
