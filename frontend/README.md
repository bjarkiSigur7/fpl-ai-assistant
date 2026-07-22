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
| `npm run lint` | ESLint (`eslint-config-next` + TS) |
| `npm start` | Serve the production build |

## Environment

| Variable | Default | Meaning |
|---|---|---|
| `NEXT_PUBLIC_API_URL` | `http://localhost:8000` | FastAPI base URL |
| `NEXT_PUBLIC_MOCK` | unset | `1` = serve every request from `src/mocks/` |

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
