# STATUS.md — what works today

Last updated: 2026-07-22 (integrator pass, stage 2 complete: models trained + walk-forward eval).
Companions: `ARCHITECTURE.md` (module map + schemas), `FPL_KNOWLEDGE.md` (game rules),
`MODEL_DESIGN_INPUTS.md` (model spec).

## TL;DR

Stage 1 (data layer) and stage 2 (features + minutes/team/rates/bonus/assemble models) are
**built, integrated, trained on all 10 seasons, and evaluated walk-forward on real data**.
Stage 3 (optimizer: MILP/autosubs/chips/sensitivity/plans) is implemented and tested but not
yet wired into `refresh`/CLI. Stage 4+ (backtest harness, API, frontend) not started.

```
~430 offline tests pass (uv run pytest -q -m "not live")
ruff clean (uv run ruff check src tests)
```

## Commands (run from `backend/`)

| Command | What it does | Status |
|---|---|---|
| `uv run fplai snapshot` | Archives today's bootstrap+fixtures, prints season state incl. the 2026-27 launch check | works |
| `uv run fplai backfill [--seasons 2016..2025]` | Downloads raw history: vaastav, football-data E0, Understat league JSON, ClubElo. Idempotent | works |
| `uv run fplai build [--seasons ...]` | Builds all processed parquet tables (incl. enrichment: Understat xG join, odds fixture ids). Offline, deterministic, idempotent | works |
| `uv run fplai train [--seasons ...] [--before-season S --before-gw G]` | Trains minutes (LightGBM v1), team (Dixon-Coles), rates (per-90 LightGBM), bonus calibration; saves artifacts + `manifest.json` to `data/models/`. The `--before-*` cutoff is the walk-forward guard | works (~70 s full history) |
| `uv run fplai predict [--season S --gw G] [--horizon N] [--no-odds]` | Live mode: predicts upcoming fixtures (synthesizing rosters); degrades gracefully to a message when none exist (now, between seasons). Backtest mode (`--season/--gw`): predicts historical GWs from strictly-prior info; writes `data/processed/predictions.parquet` + `predictions_gw.parquet` | works |
| `uv run fplai refresh` | snapshot -> best-effort pulls -> build -> predict (if artifacts exist). **Exits 0** as long as the data portion succeeds | works |
| `uv run fplai optimize` | Stage 3 wiring pending (optimizer modules themselves are implemented + tested) | print + exit 1 |

`--seasons` accepts single years and inclusive ranges: `--seasons 2024,2025`, `--seasons 2016..2025`.

## Model artifacts (data/models/, ~16 MB total)

Trained 2026-07-22 on all 10 seasons (253,568 player-fixture feature rows, 3,800 matches):

| component | class | size | fit time | headline |
|---|---|---|---|---|
| minutes | LgbMinutesModel (two-stage LGBM + isotonic) | 10 MB | 47 s | bucket log-loss 0.437 on walk-forward eval window |
| team | TeamModel (Dixon-Coles, time decay, empty-stadium regime) | 4 KB | 0.1 s | gamma=1.19, rho=-0.08 |
| rates | RatesModel (per-position LGBM Poisson, per-90) | 5.4 MB | 22 s | beats xG-only baseline on goals/assists |
| bonus | BonusCalibration (per-position bias + sigma line) | 4 KB | 0.02 s | fitted on 2025-26 BPS residuals |

Feature frame: `features/windows.py` builds 130 leakage-safe `f_*` columns over all 10
seasons in ~0.6 s (`assert_no_leakage` verified on real samples).

## Walk-forward sanity eval (honest numbers)

Protocol: single train strictly before 2025-26 GW30 (`fplai train --before-season 2025
--before-gw 30`), then 1-GW-ahead predictions for GWs 30-38 (7,074 player-GW rows; features
per target GW use only matches before that GW; closing-odds blend active — see caveat 4).
RMSE/MAE by realized-points category vs a last-5-average baseline computed on the identical
rows; reference numbers (OpenFPL / FPL Review, GW32-38 2024-25 — different window) alongside:

| category | ours | last-5 (same rows) | OpenFPL (ref) | FPL Review (ref) |
|---|---|---|---|---|
| Zeros (0 pts) | **0.798 / 0.318** | 1.013 / 0.347 | 0.818 / 0.427 | 0.689 / 0.237 |
| Blanks (1-2) | **1.464 / 1.109** | 1.916 / 1.418 | 1.291 / 0.749 | 1.189 / 0.597 |
| Tickers (3-4) | **1.554 / 1.269** | 1.986 / 1.667 | 1.517 / 1.127 | 1.594 / 1.227 |
| Haulers (>=5) | **5.368 / 4.543** | 5.646 / 4.751 | 5.142 / 4.317 | 5.172 / 4.381 |
| All | 1.870 / 0.904 | 2.106 / 1.029 | — | — |

- Beats the last-5 baseline in **every** category (both RMSE and MAE, same population).
- Zeros beat OpenFPL's reference numbers (the minutes model working as intended, even with
  no availability/news data for history).
- Tickers/Haulers sit right at OpenFPL level; Blanks lag OpenFPL — first-pass target met
  ("between last-5 and OpenFPL"), Blanks are the weakest category.
- Within-GW Spearman: 0.726 (last-5: 0.728 — rank correlation is dominated by the
  plays/doesn't-play axis; improving in-GW ranking among starters is a next step).
- Minutes component on the eval window: bucket log-loss 0.437, Brier P(0) 0.083.
- Calibration: mean xp 1.095 vs mean realized 1.138.
- Plausibility (top of GW34 2025-26 predictions): M.Salah 6.53, B.Fernandes 6.40, Rice 5.72,
  Solanke 5.67, Gabriel 5.65, Cunha 5.26, Rogers 5.24, Gakpo 5.24, Szoboszlai 4.97, Xavi 4.96
  — no bench fodder at the top.

## Data coverage

- **Processed tables**: all 10 seasons 2016-17..2025-26 on disk (fixtures 3,800; player_match
  253,568; player_gw 244,425; odds 3,800 rows with 100% `fpl_fixture_id` join rate).
- **Understat xG joined into player_match**: 100% of minutes>0 rows for 2019-2023, 79% for
  2024 (vaastav upstream stops at GW30; JSON fallback queued), 0% for 2016-2018 and 2025
  (per-player JSON downloads incomplete — see gaps).
- **rules.py**: scoring tables 2016..2026, season flags, chip windows, BPS v4 matrix,
  sell-price, FT arithmetic, 2026-27 reclassifications — pure data, tested.
- **FPL API / football-data / ClubElo / the-odds-api**: as before (throttled clients, caching,
  snapshot archive, graceful degradation without API keys).

## Known gaps / next steps

1. **Optimizer not wired into the CLI/refresh** — `optimizer/` (MILP on HiGHS, autosubs MC,
   chip EV curves, plan stability, recommendations) is implemented and tested; `fplai optimize`
   + `refresh` integration and `predictions_gw` -> `solve_plan` plumbing is the next task.
2. **No availability/news features in history** — `f_status_*`/`f_chance_of_playing` are NaN
   pre-2026 (no deadline-timestamped snapshots exist). Zeros MAE (0.318 vs FPL Review's 0.237)
   is the visible cost. Our own snapshot archive fixes this from 2026-27 GW1 onward.
3. **Understat us_* gaps**: 2016-2018 and 2025 per-player JSONs not yet downloaded (throttled
   1 req/2s, ~500/season); 2024 GW31-38 missing upstream. Re-running the queued downloader +
   `fplai build` lifts coverage automatically; models tolerate the NaNs.
4. **Odds leakage caveat in backtests**: the odds blend uses *closing* odds, which for
   matches later in a GW post-date the GW deadline. Fine for team-strength eval; for strict
   deadline-time backtests run `predict --no-odds` or snapshot pre-deadline odds (2026-27).
5. **Live-mode rosters roll forward from last appearances** — before a new season's GW1,
   promoted-team players and transfers-in have no player_match rows, so the synthesized pool
   misses them until vaastav/our snapshots produce the new season's rows. Good enough for
   in-season use; GW1 needs the bootstrap-based roster (todo at launch).
6. **Blanks (1-2 pts) is our weakest category vs OpenFPL** — likely wants better appearance
   -points/CS interplay for rotation players; revisit with team-partitioned CV.
7. **Backtest harness (stage 4)** — the per-GW walk-forward loop with state rolling (prices,
   FTs, chips) and season-points policy eval is not built; `fplai predict --season --gw` is
   the primitive it will drive.
8. **2026-27 launch**: `fplai snapshot` is the detector. When LIVE: re-verify FPL_KNOWLEDGE
   UNCERTAIN items, start daily snapshots (availability + prices), retrain with
   `fplai train`, and `fplai refresh` begins producing live predictions automatically.
9. **`player_gw.selected_by_percent` all-NA for history** (vaastav ships counts); our own
   2026-27 snapshots will carry it.
