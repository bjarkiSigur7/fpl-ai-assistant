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

## Model layer contracts (stage 2)

All models follow one lifecycle: `Model.fit(...) -> self`, `.predict(...) -> DataFrame`,
`.save(dir: Path)`, `Model.load(dir) -> Model`. Artifacts under `MODELS_DIR/<component>/`.
Prediction frames always carry the keys `season, gw, player_code` (player-level) or
`season, fpl_fixture_id` (fixture-level). All probabilities are proper (sum to 1 where
applicable); all rates are per-90.

### features/windows.py
`build_feature_frame(tables: dict[str, DataFrame], *, target: Literal["match","gw"]) -> DataFrame`
— one row per (player_code, season, gw[, fpl_fixture_id]) with `f_*` feature columns computed
STRICTLY from information available before that GW's deadline (rows for GW g may only use
matches with gw < g, plus static/pre-season info). Multi-horizon means/sums over the previous
1/3/5/10/38 matches per the OpenFPL template (player, team, opponent categories), plus
days-rest, congestion, venue, promoted-team flags, season-phase, position, price,
availability (`status`/`chance_of_playing` — nullable pre-2026 where snapshots don't exist).
Includes the label columns (`minutes`, `total_points`, component outcomes) for training joins.
Must expose `FEATURE_PREFIX = "f_"` and never leak label-time data into `f_*` columns.

### models/minutes.py
`MinutesModel.predict(features) -> DataFrame[keys..., q0, q1, q2, mu1, mu2]` where
q0=P(0 min), q1=P(1-59), q2=P(60+), q0+q1+q2=1; mu1/mu2 = E[minutes | bucket].
v0 = availability+start-share heuristic (no fit needed); v1 = two-stage LightGBM
(P(start), P(60+|start), cameo mixture). `.evaluate(features) -> dict` with bucket log-loss.

### models/team.py
`TeamModel.fit(fixtures, odds)` — Dixon-Coles with exponential time decay (MLE, scipy),
promoted-team priors seeded per FPL_KNOWLEDGE §3.6, empty-stadium regime flags respected.
`.predict_fixtures(fixtures) -> DataFrame[season, fpl_fixture_id, home_lambda, away_lambda,
p_cs_home, p_cs_away, p_home_win, p_draw, p_away_win]` from the scoreline grid (τ-corrected,
grid to 10 goals). `.blend_odds(pred, odds, weight)` — de-margined odds override/blend where
markets exist; Dixon-Coles fills the rest.

### models/rates.py
`RatesModel.predict(features) -> DataFrame[keys..., lam_goal, lam_assist, lam_saves,
lam_defcon, p_yellow, p_red, lam_og]` — per-90 intensities per player-fixture, LightGBM per
position for goals/assists (anchored on xG/xA blends × fixture multiplier from TeamModel),
negative-binomial dispersion parameter `defcon_disp` alongside `lam_defcon`.

### models/bonus.py
`expected_bonus(event_profile: DataFrame, season: int) -> Series` — E[bonus] per player-fixture
from expected BPS under that season's BPS version (rules.BPS_2026 / deltas), using an
empirical mapping E[bonus | expected-BPS rank context within fixture]. Trained on
rule-adjusted historical BPS (reconstruct v4 BPS for old seasons from raw counts where
available; document approximations).

### models/assemble.py
`assemble_xp(minutes, team, rates, features, season) -> DataFrame[season, gw, player_code,
fpl_fixture_id, xp, xp_appearance, xp_goals, xp_assists, xp_cs, xp_concede, xp_saves,
xp_defcon, xp_bonus, xp_cards, xp_other]` per FPL_KNOWLEDGE §1.1 maths exactly (position
scoring from rules.get_scoring, ⌊S/3⌋ and −⌊C/2⌋ via the Poisson grids, DefCon via NB
threshold, CS conditioned on 60+ bucket). `aggregate_gw(xp) -> DataFrame` sums a player's
fixtures within a GW (DGWs fall out automatically). This is THE number the optimizer consumes.

## Optimizer contracts (stage 3)

### optimizer/state.py
`SquadState` (pydantic): `season`, `current_gw` (the next GW to be played), `squad:
list[OwnedPlayer]` (player_code, purchase_price, current_price — 0.1m units), `bank` (0.1m
units), `free_transfers` (1-5), `chips_available: list[ChipId]` (e.g. "wc1","fh1","bb1","tc1",
"wc2",...), `active_chip: ChipId | None` (a WC already active this GW). `None`-state = initial
squad build: GW1, £1000, unlimited transfers, all 8 chips.
`from_entry(entry_id)` builds it from the FPL API (public endpoints, post-deadline picks +
transfers + chips history; document the pre-deadline limitation).

### optimizer/milp.py
`solve_plan(xp: DataFrame, prices: DataFrame, state: SquadState | None, *, horizon=8,
params: SolveParams) -> PlanResult` — the multi-GW MILP per MODEL_DESIGN_INPUTS §3
(HiGHS via highspy; FT big-M state machine per rules; chip binaries respecting
rules.chip_windows; per-player TC; separate FH squad; sell-price arithmetic; player-pool
pruning per §3.5; decay/bench-weight/FT-value/ITB-value in `SolveParams` with §3.3 defaults).
`PlanResult` (pydantic): `objective`, `gws: list[GwPlan]` where GwPlan = gw, squad (codes),
lineup, bench_order, captain, vice, transfers_in/out, hit_points, chip (ChipId|None),
expected_points; plus `solve_seconds`, `gap`.
xp input = assemble contract's `predictions_gw` (season, gw, player_code, xp) + per-player q0
column for vice weighting; prices input = player_code -> current price + position + team_code
(club-limit + quota constraints need them).

### optimizer/autosubs.py
`bench_weights_mc(lineup_q0: Mapping[player_code, float], bench_q0, formation, n=2000, seed)`
-> per-bench-slot autosub score probabilities via Monte Carlo per MODEL_DESIGN_INPUTS §4;
used by milp for bench weighting and by plans for reporting.

### optimizer/chips.py + optimizer/sensitivity.py
`chip_ev_curves(xp, prices, state, chips, gw_range) -> DataFrame[chip, gw, objective, delta_vs_no_chip]`
(forced-chip re-solves); `plan_stability(xp, prices, state, n=30, strength=1.0, seed) ->
DataFrame[move, support_pct]` (noise re-solves per §3.4 noise model over this-GW moves).

### optimizer/plans.py
`build_recommendation(state, xp, prices) -> Recommendation` (pydantic): the weekly verdict —
`action` ("hold" | "transfer" | "chip:<id>"), `transfers` (in/out pairs with xp deltas),
`hits`, `captain`, `vice`, `lineup`, `bench_order`, `chip_advice` (play/hold per available
chip with EV deltas), `dream_team` (the fresh-£100m benchmark squad from a None-state solve),
`plan` (the full PlanResult), `stability` (from sensitivity), `rationale` (human-readable
bullet strings, generated from the numbers — no LLM calls).

## Season-simulation contracts (stage 6 — Monte Carlo chip planner)

### models/sampler.py
`PointsSampler.sample(predictions: DataFrame, n: int, seed: int) -> ndarray[n, rows]` —
vectorized draws of realized FPL points per prediction row (player-fixture), sampling the
decomposition jointly: minutes bucket ~ Categorical(q0,q1,q2); goals/assists ~ Poisson(λ·E[min]/90)
conditional on bucket; CS/concede from team scoreline draws SHARED within a fixture (all players
in one fixture-draw see the same scoreline — correlation matters for BB/DEF stacks); saves,
DefCon ~ NB, bonus from the BPS rank machinery (cheap approximation acceptable, document it);
cards/OG. Deterministic under seed. `sample_gw(predictions_gw, ...)` aggregates fixtures.

### optimizer/season_sim.py
`simulate_chip_plans(xp, prices, state, *, window: range, n_rollouts, seed, params) ->
ChipSimReport` — (1) one no-chip backbone MILP over `window` (aggressive pruning, capped
time); (2) candidate chip placements on the backbone: BB/TC gains analytic per GW (bench xP /
best-captain xP from the backbone squad), WC/FH via bounded segment re-solves on a pruned
pool; (3) PointsSampler rollouts scoring every candidate schedule per sampled season —
autosubs/vice resolved per-sample from sampled minutes. Output per chip: E[gain] by GW over
the FULL window, sd, P(this GW is the best week), P(playing now beats holding), recommended
GW + confidence; plus joint-schedule ranking of the top-k chip schedules.
`ChipSimReport.to_frame()` feeds chip_curves.parquet v2 (adds columns: sd, p_best_week,
p_beats_hold, n_rollouts) — additive, the API/UI schema extends, never breaks.

### Pipeline integration
`fplai predict` gains `--through-gw` (default: end of current chip window, e.g. 19);
`fplai simulate` runs simulate_chip_plans and writes chip_sim.parquet + folds the verdicts
into recommendation.json's chip_advice (hold now carries a quantified value). Recommendation
rationale cites simulation confidence, not just point deltas.

## API contracts (stage 5 — frontend depends on these)

FastAPI under /api: `GET /health`; `GET /state` (season state, next deadline, data freshness,
model manifest); `GET /predictions?gw&horizon` (per-player xP + components + q's); `GET
/dream-team?gw` (benchmark squad); `GET /my-team/{entry_id}` (squad state + Recommendation);
`GET /players/{player_code}` (identity, history, upcoming fixture xP breakdown); `POST
/refresh` (runs the refresh pipeline; SSE/polled status at GET /refresh/status); `GET
/chip-curves?entry_id`. All responses are pydantic models mirroring the optimizer/model
contracts; frontend consumes these shapes verbatim.

## Conventions

- Python 3.12, type hints everywhere, dataclasses/pydantic at module boundaries.
- No global mutable state; clients take explicit paths from `fplai.config`.
- Every data module gets a `tests/test_<module>.py` with offline fixture-based tests
  (record one small real payload into `tests/fixtures/`); live-network tests marked
  `@pytest.mark.live` (excluded by default; run in CI/verification explicitly).
- `ruff check` clean; line length 100.
- Errors: raise, don't silently continue; downloaders log and retry with tenacity.
