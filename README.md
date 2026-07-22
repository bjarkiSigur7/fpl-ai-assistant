# FPL AI Assistant

A state-of-the-art Fantasy Premier League assistant for the 2026-27 season. It predicts
expected points for every player, computes the best legal £100m squad each gameweek, and
tells you exactly what to do with *your* team — transfers, hits, captaincy, and chip timing
(Wildcard / Free Hit / Bench Boost / Triple Captain) — all optimized to maximize total
season points.

## What it does

- **Expected-points engine** — an ensemble of gradient-boosted trees, neural sequence
  models, and a Bayesian minutes model, decomposing points into
  P(minutes) × (appearance + attacking + clean-sheet + bonus + …) per the modern
  FPL-modeling consensus. Trained on a decade of per-gameweek history, xG data,
  bookmaker odds, and injury news.
- **Perfect-squad benchmark** — the highest-EV legal 15-man squad (£100m, max 3 per club)
  with optimal XI, captain, and bench order, recomputed every gameweek.
- **Your team's plan** — a multi-gameweek MILP planner (6-8 GW horizon, discounted EV,
  free-transfer state machine) that decides: make these transfers / take a hit / do
  nothing, plus a full-season chip plan that re-plans as fixtures and injuries evolve.
- **Initial squad builder** — before GW1, builds your optimal starting squad from scratch.
- **Dashboard** — a Next.js web app to see all of it at a glance.

## Architecture

```
backend/    Python: data ingestion, features, models, MILP optimizer, FastAPI
frontend/   Next.js dashboard
docs/       Research corpus, canonical rules, model design docs
data/       (gitignored) raw pulls, processed features, trained model artifacts
```

Everything runs locally. Refresh is manual: `make refresh` (or the refresh button in the
dashboard) pulls fresh data, re-predicts, and re-optimizes.

## Status

🚧 Under construction — research phase complete artifacts land in `docs/`.
