# PITCHSIDE — the FPL quant-desk dashboard

Next.js (App Router) + TypeScript + Tailwind v4 frontend for the fpl-ai-assistant
backend. Dark trading-desk aesthetic; hand-rolled SVG charts (no chart libs); SWR
for data fetching. The only added npm dependency is `swr`.

## Dev commands (run from `frontend/`)

| Command | What it does |
|---|---|
| `npm run dev` | Dev server on http://localhost:3000 (expects the API on :8000) |
| `NEXT_PUBLIC_MOCK=1 npm run dev` | **Standalone demo mode** — all endpoints served from deterministic in-browser mocks (`src/mocks/`), no backend needed |
| `npm run build` | Production build (type-checks; must pass clean) |
| `NEXT_PUBLIC_STATIC=1 npm run build` | **Static public build** — `output: "export"` to `out/`, all data from the bundle in `public/data/` (see below) |
| `npm run lint` | ESLint (`eslint-config-next` + TS) |
| `npm run test:unit` | `node --test` unit suite — incl. the TS↔Python rating-engine parity fixtures (`tests/`) |
| `npm start` | Serve the production build |

## Environment

| Variable | Default | Meaning |
|---|---|---|
| `NEXT_PUBLIC_API_URL` | `http://localhost:8000` | FastAPI base URL (local mode) |
| `NEXT_PUBLIC_MOCK` | unset | `1` = serve every request from `src/mocks/` (ignored when STATIC) |
| `NEXT_PUBLIC_STATIC` | unset | `1` = public GitHub Pages mode: static export, all data from `{basePath}/data/*.json`, AI-Rating computed client-side, local-only surfaces (my-team, refresh) hidden |
| `NEXT_PUBLIC_BASE_PATH` | `""` | Next `basePath`, env-driven so the domain cutover is config-only. GitHub Actions passes `/fpl-ai-assistant` for the project-pages URL |

## Static public mode (GitHub Pages)

`NEXT_PUBLIC_STATIC=1` flips `src/lib/api.ts` onto `src/lib/staticBundle.ts`: every
hook reads the daily model bundle from `{basePath}/data/<file>.json` and adapts it
onto the same view models, so pages never branch on the mode. **The GitHub Actions
workflow must copy the `fplai publish-static --out site-data` bundle into
`frontend/public/data/` BEFORE `next build`** — the export both ships the JSON and
uses `players.json` at build time to `generateStaticParams` one page per player
(`/players/[code]`). With **no bundle at all** the build still succeeds as a
dataless site shell (deploy-pages' intentional first-boot path: loud warning, one
placeholder player page, empty states); a **partial** bundle (`meta.json` without
`players.json`) fails the build loudly. `public/data/` is gitignored; for a local
rehearsal:

```sh
cd backend && uv run fplai publish-static --out ../site-data
mkdir -p ../frontend/public/data && cp ../site-data/*.json ../frontend/public/data/
cd ../frontend && NEXT_PUBLIC_STATIC=1 npm run build   # -> out/
```

The AI-Rating page scores squads **client-side** in this mode via
`src/lib/rating.ts`, an exact pure-TS port of the Python metric
(`backend/src/fplai/optimizer/rating.py`): greedy best-XI per GW with formation
minimums then fill, captain bonus = XI max,
`score = clamp((team − floor)/(optimal − floor)) · 100`, bands
ELITE/STRONG/SOLID/ROUGH/FODDER. The floor/optimal anchors ship precomputed in
`rating.json`. Parity with Python is enforced by `npm run test:unit` against
`tests/fixtures/rating-parity.json` (25 seeded real-data squads scored by the actual
Python engine; tolerance 0.05).

## Pages

- `/` — ticker (deadline countdown, freshness, model, season state), THE VERDICT card
  (`/api/recommendation`, or `/api/my-team/{entry}` once an entry id is saved), dream-team
  and my-team pitches, top movers + xP trend rail.
- `/players` — sortable/filterable predictions explorer (position tabs, club filter,
  search, component sparkbars, availability risk). Row click → `/players/[code]`.
- `/players/[code]` — identity header, season history chart (points bars + xG line),
  upcoming per-fixture xP breakdown bars.
- `/planner` — multi-GW plan timeline, chip EV small multiples (`/api/chip-curves`),
  plan-stability support bars.
- `/settings` — entry-id persistence (localStorage), API wiring readout, refresh
  trigger (`POST /api/refresh`) with 2s status polling and a live log tail.

## API contract

`src/lib/types.ts` has two layers:

- **Wire types** (`Api*`-prefixed) — field-for-field mirrors of
  `backend/src/fplai/api/schemas.py` (snake_case, no remapping). The optimizer
  contracts (`Recommendation`/`TransferPair`/`ChipAdvice`/`DreamTeam`/`StabilityEntry`
  from plans.py, `PlanResult`/`GwPlan` from milp.py, `SquadState`/`OwnedPlayer` from
  state.py) are shared verbatim between both layers.
- **View models** — the flattened shapes the pages consume. `src/lib/api.ts` adapts
  wire → view on the network path (e.g. it pages through the per-GW paginated
  `/api/predictions` and assembles the horizon-wide view; it flattens
  `/api/state`'s `season_state`/`freshness` nesting; it maps `/api/refresh/status`'s
  `state`/`log_tail` to `running`/`log`). Mock mode serves the view models directly.

`GET /api/my-team/{entry}` answers **202** (job status) while the background MILP
solve runs; the client then falls back to the shared `/api/recommendation` until a
later poll returns the squad-aware result. Squads travel as `player_code` lists; the
UI joins names/prices/xP client-side via `src/lib/playerIndex.ts` from
`/api/predictions`.

## Mock scenario

Pre-season 2026-07-22: the 2026-27 game has not launched (ticker counts down to the
provisional GW1 deadline, 2026-08-21 17:30 UTC); predictions are a 2025-26 GW34
backtest with a 5-GW horizon. GW34 top-10 xP values are the real walk-forward numbers
from `docs/STATUS.md`. Without an entry id the verdict is the pre-season
`initial-squad` build ("ASSEMBLE."); save any entry id in settings to flip the desk to
the squad-aware `transfer` verdict, plan and stability views. GW37 contains a LIV/MCI
double gameweek; `POST /api/refresh` runs a ~9 s simulated pipeline with a scripted
log for the settings page demo.
