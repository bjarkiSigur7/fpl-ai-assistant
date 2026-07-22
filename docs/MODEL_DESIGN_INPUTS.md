# MODEL_DESIGN_INPUTS.md — Implementation Brief for fpl-ai-assistant

**Synthesized 2026-07-22** from `docs/research/sota-methods.md` and `docs/research/data-sources.md`. Companion to `docs/FPL_KNOWLEDGE.md` (rules, scoring, season flags — that document is authoritative for all game constants; this one specifies the model and pipeline). Scoring values used below reflect the verified 2026-27 table (note: GK goal = 10 pts, in force since 2024-25).

---

## 1. Recommended architecture: decomposed expected points (xP)

**Verdict from the SOTA survey:** component decomposition beats monolithic regression at the architecture level because it (a) isolates the minutes problem, (b) lets us splice in bookmaker/market data where it is strongest (team goals, clean sheets), and (c) survives rule changes by retraining only the affected component (DefCon, BPS). But OpenFPL (arXiv:2508.09992) proves a well-engineered gradient-boosted monolith gets within ~2-8% RMSE of a commercial decomposed model *except on Zeros* (the minutes problem). **Therefore: hybrid — ML-estimated per-90 rates inside an explicit probabilistic decomposition, with dedicated minutes / CS / DefCon / bonus layers.**

Every serious model (AIrsenal, FPL Review, theFPLkiwi) computes, per player p, per match m:

```
xP(p,m) = Σ_buckets P(minutes bucket) × [appearance + attack + clean_sheet + conceded + saves + defcon + bonus + cards_og]
```

Minutes distribution collapsed to FPL-relevant buckets {0, 1-59, 60+} with probabilities (q0, q1, q2) and conditional expected minutes (mu1, mu2) — supplied by the minutes model (§2).

### 1.1 Component maths

1. **Appearance:** `E = 1·q1 + 2·q2`.

2. **Attacking returns.** Estimate per-90 intensities λ_g(p,m), λ_a(p,m). Rate route (recommended default):
   - `λ_g = f(npxG/90 blend, penalty share, shot volume, finishing prior) × F_att(m)` where the fixture multiplier `F_att(m) = E[team goals in m] / E[team goals vs average opponent]`, with E[team goals] from the Dixon-Coles team model (§1.2) or de-margined bookmaker over/unders.
   - `E[goal pts] = pts_goal(pos) × λ_g × (E[min]/90)` with pts_goal = {GKP 10, DEF 6, MID 5, FWD 4}; assists analogous at 3 pts.
   - Alternative conditional route (AIrsenal): Σ over scorelines P(score) × E[involvements | team goals, θ_p, minutes], where θ_p = (p_score, p_assist, p_neither) ~ Dirichlet with position-level empirical-Bayes prior, exponential time-decay ε=0.2, prior strength ~35 goals. Cleaner uncertainty propagation, costlier; use for validation of the rate route.

3. **Clean sheets.** `P(CS_team) = P(goals_conceded = 0)` from the Dixon-Coles scoreline grid (or bookmaker-derived via totals/btts — CS is not a standard odds market). Player-level: `E = pts_CS(pos) × q2 × P(CS | on-pitch window)` with pts_CS = {GKP 4, DEF 4, MID 1, FWD 0}. Conceded-goals penalty for GKP/DEF: `E = −E[⌊C/2⌋]` computed over the same Poisson grid.

4. **Saves (GKP).** Shots-on-target-against ~ Poisson(λ_SoT(opponent, venue)); saves = SoT − goals. `E[save pts] = E[⌊saves/3⌋]` over the joint grid (≈ E[saves]/3 with a small convexity correction).

5. **Defensive contribution (must-have for 2026-27).** DEF: `E = 2 × P(CBIT ≥ 10 | minutes)`; MID/FWD: `E = 2 × P(CBIRT ≥ 12 | minutes)`. Model the per-90 count rate ν(p) (API provides `defensive_contribution_per_90` and raw `tackles`/`recoveries`/`clearances_blocks_interceptions`), scale by expected minutes and opponent-possession/game-state factors (counts rise when the team defends more), and use a **negative binomial** for the threshold probability (counts are overdispersed). One full season (2025-26) of realized data exists for calibration.

6. **Bonus.** BPS is a *rank within the match*, so model it relative to other players' expected BPS: simulate the match's BPS ranking from event-count distributions, or learn an empirical mapping E[bonus | position, event profile]. **The 2026-27 BPS rebalance (being-tackled penalty removed; CBI 1 per 3; GK save restructure) breaks naive historical mappings — reconstruct rule-adjusted BPS from raw event counts and retrain.** Expect fewer CB bonuses, more GK/attacker bonuses.

7. **Cards / own goals:** small negative expectations from per-player yellow (−1) / red (−3) rates and OG priors.

8. **Multi-match GWs:** `xP(GW) = Σ over the GW's fixtures` — DGW value and blank worthlessness fall out automatically; the optimizer sees them for free.

### 1.2 Team model (feeds components 2, 3, 4)

Dixon-Coles (1997) variant à la `bpl-next` (NumPyro, used by AIrsenal):

```
p(y_h, y_a) = τ(y_h, y_a) · Poisson(y_h | a_h·b_a·γ_h) · Poisson(y_a | a_a·b_h)
```

with per-team latent attack a_i, defence b_i, per-team home advantage γ_i (LogNormal prior), DC low-score correlation τ, bivariate hierarchical prior on (log a_i, log b_i) with correlation ρ, optional team covariates. Fit MCMC/NUTS (or fit our own Dixon-Coles by MLE on football-data.co.uk history for speed). Output: full scoreline probability grid per fixture.

Blending priority for team goals/CS where available: **de-margined bookmaker odds > Dixon-Coles > FPL FDR (display only, never model input)**. Odds are the strongest single fixture-strength signal; Elo (ClubElo) covers fixtures beyond bookmaker horizons. Seed 2026-27 promoted clubs (Coventry, Ipswich, Hull) from Championship xG with a promotion discount (see FPL_KNOWLEDGE §3.6); apply home-advantage regime flags per FPL_KNOWLEDGE §2.4 when fitting on history.

### 1.3 Per-90 rate models (component 2 inputs)

OpenFPL is the public benchmark and feature-engineering template: per-position ensembles (XGBoost + Random Forest, median of 50 models), ~200 features per position, every feature **averaged over 1-, 3-, 5-, 10- and 38-match windows** (multi-horizon form) — feature categories: player (points, minutes, ICT, penalties, saves, BPS, shots, xG, xA…), team (goals, xG, deep completions, PPDA), opponent (xGA, PPDA att/def), match status (availability). Train per forecast horizon (1/2/3 GWs ahead; 1-GW-ahead cuts Zeros RMSE 15-25% vs 3-GW). Use OpenFPL's published results as the accuracy bar (see §7.3).

---

## 2. Minutes model — the highest-leverage component

**Why first:** OpenFPL's only decisive loss to commercial FPL Review is the Zeros category (RMSE 0.818 vs 0.689; MAE 0.427 vs 0.237 — a 1.8× MAE gap), explicitly attributed to lacking expected-minutes inputs. Every downstream number (xP, autosubs, captaincy, vice weight, solver plans) is linear-or-worse in minutes error. Commercial SOTA (FPL Review xMins) is 1,000 simulations/player/match mixing start/cameo branches with rotation events and injury-proneness decay, user-overridable.

**Formalization (build this):** per player-match, predict a categorical **(start, bench-cameo, bench-unused, out)** plus a start-minutes distribution conditional on start (mixture: mass at 90 + early-hook component). This yields exactly the (q0, q1, q2, mu) the decomposition needs, plus P(plays 0) for autosub/vice math.

**Features:** recent start share (1/3/5/10-match windows), minutes trajectory, days rest, competition congestion (UCL/UEL/UECL/cup midweeks — FPL-Core-Insights supplies cup fixtures), manager rotation history at position, new-signing/returning-from-injury flags, opponent strength, season phase, FPL API availability (`status` a/d/i/s/u, `chance_of_playing_this/next_round` 0/25/50/75/100, `news` + `news_added` timestamp). Beware price/ownership features (leakage, §7.2).

**Staged plan:**
1. v0: availability flags + recent-start-share heuristic (AIrsenal level).
2. v1: two-stage supervised — P(start) classifier, then P(60+|start) and cameo-minutes models (community-craft level; no strong academic precedent).
3. v2: hierarchical Bayesian start propensity nested in team-level rotation regimes, posterior-updated on each team sheet (no public reference implementation — build-it-ourselves area).
4. v2+: structured news extraction (press conferences → injury/fitness-doubt/rotation-hint flags). Warning from arXiv:2405.02412: naive news-text embeddings *underperformed* — extract structured facts, not embeddings.

**Expose user overrides** (FPL Review does; theFPLkiwi enters minutes manually — evidence the signal is news-shaped). 2026-27 specifics: World Cup fatigue dampening GW1-4 for the ~40 deep-run players; new-manager rotation uncertainty at 10 clubs (FPL_KNOWLEDGE §3.3/§3.5).

---

## 3. The multi-GW MILP (implement from scratch)

Transcribed from the community solver (`solioanalytics/open-fpl-solver`, `dev/solver.py`, read in full by the research pass). **Licensing:** Apache-2.0 for personal use only; commercial use needs a license — so we implement our own code from this maths (maths is not copyrightable; the code is).

### 3.1 Sets and decision variables

Sets: players p; gameweeks w ∈ {next … next+H−1} (horizon H = 5-8 typical); bench slots o ∈ {0,1,2,3} (0 = GK); FT states s ∈ {0..5}.

Binary unless noted:
- `squad[p,w]`, `squad_fh[p,w]` (separate Free-Hit squad), `lineup[p,w]`, `captain[p,w]`, `vicecap[p,w]`, `bench[p,w,o]`
- `transfer_in[p,w]`, `transfer_out_regular[p,w]`, `transfer_out_first[p,w]` (price-modified players: first sale at selling price, later at buy price — the multiple-sell fix)
- `in_the_bank[w]` continuous ≥0; `fts[w]` integer 0-5 with one-hot `fts_state[w,s]`; big-M indicator pair `ft_above_ub[w]`, `ft_below_lb[w]`; `penalized_transfers[w]` integer ≥0; `transfer_count[w]`
- Chips: `use_wc[w]`, `use_bb[w]`, `use_fh[w]`, `use_tc[p,w]` (**TC is per-player**)

### 3.2 Constraints

- **Squad composition:** Σ_p squad[p,w] = 15; per-position quotas 2/5/5/3; ≤3 per club; budget; the FH squad mirrors all composition constraints × `use_fh[w]`.
- **Lineup:** Σ lineup = 11 + 4·use_bb[w]; formation bounds `squad_min_play ≤ Σ_pos lineup ≤ squad_max_play (+ use_bb adjustments)`; lineup ⊆ squad (FH squad in FH weeks); bench: exactly one GK in slot 0, one player per outfield slot, lineup+bench ≤ 1 per player; captain, vice ∈ lineup; captain ≠ vice.
- **Continuity & budget:** `squad[p,w] = squad[p,w−1] + transfer_in[p,w] − transfer_out[p,w]`; no transfers in FH weeks; `itb[w] = itb[w−1] + sold[w] − bought[w]`; FH affordability against previous squad value. Selling prices per FPL_KNOWLEDGE §1.8.
- **FT state machine** (2024-25+ rules): `raw_ft[w] = fts[w] − transfer_count[w] + 1 − use_wc[w] − use_fh[w]`, then big-M (M=20) clamps: raw > 5 ⇒ fts[w+1] = 5; raw ≤ 0 ⇒ fts[w+1] = 1; else fts[w+1] = raw. Hits: `penalized_transfers[w] ≥ transfer_count[w] − fts[w] − 15·use_wc[w]`. (No AFCON special case in 2026-27; keep the hook configurable — 2025-26 had a `+5` top-up at GW16.) Cap Σ transfers ≤ 20 per non-chip GW.
- **Chips:** `use_wc + use_fh + use_bb + Σ_p use_tc ≤ 1` per GW; per-chip count limits over the horizon respecting the two-set windows (set 1: WC/FH GW2-19, BB/TC GW1-19, all expire at GW19 deadline; set 2: GW20-38 — FPL_KNOWLEDGE §1.7); `use_tc[p,w] ≤ captain[p,w]`; support forced/allowed/banned chip-GW options; detect an already-active WC from user API state.
- **Optional constraints worth carrying over:** banned/locked players (global or per GW), booked transfers, no-transfer GWs, per-position transfer bans, max defenders per opposing team (attack-covariance control), "no opposing play" (hard or penalty-weighted with linearized products), ITB buffer, weekly/horizon hit limits.

### 3.3 Objective

With decay base β (community default **0.84**; FPL Review recommends 0.80-0.95, default 0.85; β=1 for "regular" mode):

```
max Σ_w β^(w−next) · [ gw_xp(w) − 4·penalized_transfers(w) + ft_gain(w) − ft_penalty(w) + 0.08·itb(w) − opposing_penalty(w) ]

gw_xp(w) = Σ_p xPts[p,w] · ( lineup[p,w] + captain[p,w] + 0.1·vicecap[p,w] + use_tc[p,w] + Σ_o benchweight_o·bench[p,w,o] )
```

- Captain adds +1×xP (doubling); TC adds a further +1×xP; vice weight 0.1 ≈ assumed P(captain plays 0). **Upgrade: set vice weight to the captain's actual q0 from the minutes model.**
- Bench weights (static defaults): {GK: 0.03, slot1: 0.21, slot2: 0.06, slot3: 0.002}. **Upgrade: replace with true autosub probabilities via Monte Carlo (§4).** (AIrsenal's (0.65, 0.3, 0.1) are *relative* outfield weights — different semantics; do not mix.)
- **FT continuation value:** each saved FT worth `ft_value` (default **1.5 pts**, configurable per state) implemented as a state-value function `ft_state_value[s] = ft_state_value[s−1] + ft_value(s)`, objective earns the increment — this stops the solver burning the FT stack.
- **ITB value:** 0.08 pts per £1m per GW.
- Decay rationale: projection error grows with horizon and plans are re-solved weekly. Set transfer depth a couple of GWs short of the projection horizon to avoid end-of-horizon artifacts.

### 3.4 Uncertainty handling

Re-solve with perturbed projections and report **how often each move is optimal** (sensitivity analysis). Community noise model:

```
Pts'[p,w] = Pts[p,w] + strength · Pts[p,w] · (92 − xMins[p,w]) / 134 · N(0,1)
```

(relative noise, inflated for low-minutes players). Enumerate alternative plans via iterative no-good cuts on this-GW transfers/lineup.

### 3.5 Solver engineering

- Build the model with any MILP layer; solve with **HiGHS via highspy** (presolve on, parallel on, `mip_rel_gap`, time limit ~20 min default); Gurobi optional (~1 order of magnitude faster, not required — FPL-sized instances solve in seconds-to-minutes on HiGHS after pruning).
- **Player-pool pruning is essential and standard:** keep top-EV percentile (default top 10%), total xMins lower bound (default ≥100 over horizon), EV-per-price quantile cut; always keep current squad + locked/booked players.

---

## 4. Captaincy, EO, autosubs, chips

- **Effective ownership:** `EO_p = (owners + captainers + triple-captainers) / teams` within a reference sample (overall, top-10k, elite-1k — LiveFPL elite page). Expected rank movement vs a field: `Σ_p (m_p − EO_p)·xP_p` with your multiplier m_p ∈ {0,1,2,3}. A 100%-EO captain is rank-neutral; captaining a >100%-EO player you merely own loses rank when he hauls. **Ship two objective modes:** max EV (season points) and max E[rank move] (EO-weighted field) — switchable by user goal (overall rank vs mini-league), echoing the variance-seeking DFS literature (Hunter/Vielma/Zaman, arXiv:1604.01455).
- **Vice EV:** `q0(captain) × xP_vice` — use the real q0, not the 0.1 default.
- **Autosubs (exact):** bench slot o scores iff some starters miss, higher-priority bench used or formation-infeasible, and the resulting formation is legal. Compute by **Monte Carlo over independent Bernoulli(q0) starter absences (11 draws)** — cheap, and precisely what FPL Review does ("full probability based calculations… rather than fixed factors"). Static weights understate bench value for rotation-risky squads. Good bench ordering ≈ 5-10 pts/season.
- **Chip heuristics (converged community consensus):** WC before the biggest fixture swing; BB on a DGW after a WC with 15 playing assets (typically 6-20+ pts); TC on the premium captain's best fixture, mildly favoring DGWs; FH on big blanks (or an un-wildcardable monster DGW). Each half-season needs its own plan (GW19 chip cliff creates forced-usage dynamics). Order-of-magnitude value of good timing: ~+49 pts avg vs no chips (unaudited single-tool claim).
- **Chip planning practice:** chips are binaries in the MILP, but don't trust one joint solve — run `forced/allowed chip-GW` sweeps and compare objective values ("chip EV curves"), because chip EV 4-8 GWs out is dominated by projection uncertainty and unannounced BGW/DGWs. Target end-state: season-long simulation (sample outcomes from predictive distributions, run the weekly re-solve policy inside each rollout, pick chip policy maximizing E[season points] or P(beat rival)) — no public tool documents a full implementation; this is a differentiator.

---

## 5. Data ingestion plan

### 5.1 Pipeline-critical sources

| Source | What | Cadence | Notes / fallbacks |
|---|---|---|---|
| FPL API `bootstrap-static/` | prices, ownership, status/news, scoring rules, events, chips | 2-4×/day (hourly near deadline); **snapshot to disk daily from launch day 1** | ~2 MB; no key; browser UA required; diff for price/status changes. Our own snapshots are the archive going forward |
| FPL API `fixtures/` | schedule, kickoffs, FDR, DGW/BGW detection | daily | per-fixture stats incl. `defensive_contribution` |
| FPL API `element-summary/{id}/` | per-GW player history + remaining fixtures | nightly full sweep after match days (~850 req @ 1 req/s ≈ 15 min) | 40 columns incl. xG family, DefCon inputs, `value`, `selected` |
| FPL API `event/{gw}/live/` | live points + `explain` breakdown; post-GW ground truth | every 60-90 s during matches; once post-GW | scores provisional until 09:00-next-day lockdown |
| FPL API `entry/*` (public) | user team post-deadline, rivals, mini-league EO | on demand | picks public only after deadline; pre-deadline squad needs auth `my-team/{id}/` (403 without cookies; automated login UNCERTAIN — plan for pasted session cookie or manual team entry) |
| vaastav/Fantasy-Premier-League `merged_gw.csv` | training corpus 2016-17 → 2025-26 | one-time backfill + verify at season boundaries | last commit 2026-07-20, alive; xG cols only from 2022-23; encoding quirks (2018-19 latin-1); watch for `data/2026-27/` in Aug |
| Understat | player/team xG, npxG, shots, xGChain; pre-2022 xG | league call nightly post-matches; player calls weekly/on-demand | **site re-architected Dec 2025**: use JSON endpoints `GET /getLeagueData/EPL/{year}` and `GET /getPlayerData/{id}/` (trailing slash required) with `--compressed` + browser UA; ~1 req/2s. Fallback: understat/understatAPI pip packages (pin + integration-test), worst case Playwright |
| ClubElo `api.clubelo.com` | team strength prior/validation | weekly + after each round | free CSV, no key: `/{ClubName}` history, `/{YYYY-MM-DD}` snapshot. Or fit own Dixon-Coles on football-data history |
| football-data.co.uk `E0.csv` | historical odds (Bet365/Pinnacle/closing) + match stats, 1993-94 → present | one-time backfill + weekly in season | URL `mmz4281/{yy}{yy}/E0.csv`; backbone for odds→goals/CS calibration without burning API credits |

### 5.2 Enrichment (degrade gracefully)

- **the-odds-api.com** (free 500 credits/mo; cost = markets × regions/call): daily `h2h+totals` (uk) snapshot + one `player_goal_scorer_anytime` per-event sweep (US bookies only) on deadline morning ≈ **150 credits/mo**. CS odds not a standard market — derive from totals/btts or Poisson. Fallback: football-data closing odds + Elo probabilities.
- **olbauday/FPL-Core-Insights** (renamed from FPL-Elo-Insights): cup/European fixtures + ClubElo pre-joined to FPL ids; weekly git pull. Covers 2024-25/2025-26 only.
- **premierleague.com/en/latest-player-injuries**: plain-curl scrapeable (tested 200); daily as second injury signal. premierinjuries.com now behind Cloudflare — manual check only. FPL API `status`/`news`/`chance_of_playing` alone are sufficient for a functional model.
- **fplform.com CSV export**: weekly, external xP benchmark (not ground truth).
- **FBref via `soccerdata`** (cloudscraper, ≤10 req/min hard cap): **deferred to v2** — since 2025-26 the FPL API itself carries the DefCon inputs; FBref only adds progressive passes/SCA/PSxG enrichment. Direct scraping 403s (Cloudflare).
- **Rotowire predicted lineups**: UNCERTAIN scrapeability; predicted-XI probability is the biggest external gap — fallback is our own minutes model.

### 5.3 Etiquette and identity

- FPL API: ≤1 req/s sustained, exponential backoff on 429, browser UA, cache aggressively. Understat: ~1 req/2s. FBref: ≤10 req/min or banned ~24h.
- **Identity crosswalk (build day 1 of 2026-27):** FPL `element.id` (per-season) ↔ `element.code`/`opta_code` (cross-season stable); Understat id ↔ FPL via fuzzy name+team seeded from vaastav's `understat/` maps; one canonical team table mapping FPL `team.code` ↔ Understat ↔ ClubElo ↔ football-data names. Team ids are per-season.

---

## 6. Benchmarks to beat

OpenFPL (public, code at github.com/daniegr/OpenFPL) vs FPL Review (commercial) vs last-5-average baseline — 1-GW horizon, RMSE/MAE by return category (prospective test GW32-38 of 2024-25):

| Category | OpenFPL | FPL Review | Last-5 baseline |
|---|---|---|---|
| Zeros (0 pts) | 0.818 / 0.427 | **0.689 / 0.237** | 0.791 / 0.270 |
| Blanks (≤2) | 1.291 / 0.749 | **1.189 / 0.597** | 1.400 / 0.652 |
| Tickers (3-4) | **1.517 / 1.127** | 1.594 / 1.227 | 2.136 / 1.645 |
| Haulers (≥5) | **5.142 / 4.317** | 5.172 / 4.381 | 5.613 / 4.709 |

Interpretation: match OpenFPL everywhere, beat it on Zeros via the minutes model; FPL Review's Zeros numbers are the commercial bar. Hauler RMSE ≈ 5 for everyone — hauls are irreducible variance; predict hazard rates, not hauls.

---

## 7. Backtesting protocol

### 7.1 Walk-forward by GW (the only valid protocol)

For each historical GW t: build features strictly from data available **before the deadline of t**; predict; optimize; record realized points; roll all state forward (prices, FT bank, chips). Model development on team-partitioned CV (OpenFPL: 5 folds split by team allocation, 16 team-seasons/fold — prevents same-team leakage), plus a **prospective holdout never touched during development** (OpenFPL used GW32-38 2024-25; we should reserve the equivalent tail of 2025-26). Every feature joins on an *availability timestamp*, not just (team, season); calibrators fitted in-history and frozen.

### 7.2 FPL-specific leakage traps (all must be engineered against)

1. **Price encodes future information** — `now_cost`, `transfers_in_event`, `selected_by_percent` from end-of-season dumps leak crowd knowledge of news that postdates the prediction time. Use GW-snapshot rows (vaastav) and our own daily snapshots.
2. **News timing** — `chance_of_playing`/`news` mutate continuously; only deadline-timestamped snapshots are safe.
3. **Retrospective revisions** — Opta corrections, bonus recalcs, fixture rescheduling flow backwards into scraped data (and the 2026-27 next-morning lockdown makes late corrections *more* common).
4. **Rule non-stationarity** — DefCon (2025-26) and BPS v4 (2026-27): reconstruct rule-adjusted targets from raw event counts; never train those components on mixed-rule seasons naively (feature flags in FPL_KNOWLEDGE §2.2).
5. **Own-model feedback** — never evaluate a solver policy with projections fitted on the evaluation window; keep projection-fitting and policy-evaluation windows disjoint.

### 7.3 Metrics (in priority order)

1. **Component calibration:** minutes-bucket log-loss; CS Brier; per-90 rate calibration curves.
2. **xP accuracy:** RMSE/MAE per position × return category (Zeros/Blanks/Tickers/Haulers — overall RMSE hides the minutes problem) + within-GW Spearman rank correlation (what selection actually uses).
3. **Decision quality:** season points of the full weekly re-solve policy vs (a) last-5-average + same solver, (b) average human ≈ 2,200-2,300 pts, top-10k ≈ 2,550-2,700 (varies by season), (c) hindsight optimum ≈ 4,984 (2019-20 perfect-foresight solve) — enormous headroom, mostly unreachable variance.
4. **Known anchors:** Matthews et al. (AAAI 2012) agent ≈ top 1% of 2.5M; AIrsenal "well inside top 30%"; solver + elite projections anecdotally top-10k (survivorship-biased, no audited study).

---

## 8. Build order

1. **Data layer first** (§5): daily API snapshotting must start at 2026-27 launch (this week) — the snapshot archive is unrecoverable if missed. Backfill vaastav + Understat + football-data; build identity crosswalks.
2. **Minutes model v0→v1** (§2) — highest leverage.
3. **Team model** (Dixon-Coles + odds blend, §1.2) and **per-90 rate models** (§1.3).
4. **xP assembly** (§1.1) with DefCon negative-binomial and simulated-BPS bonus under 2026-27 rules.
5. **MILP** (§3) with true autosub Monte Carlo (§4), FT state machine, chip windows; HiGHS.
6. **Backtest harness** (§7) against OpenFPL numbers (§6); walk-forward on 2023-24/2024-25 with 2025-26 tail as prospective holdout.
7. **Chip EV curves and EO/rank-mode objectives** (§4); season simulation as the stretch goal.

Key references: OpenFPL arXiv:2508.09992 (+ github.com/daniegr/OpenFPL) · community solver maths github.com/solioanalytics/open-fpl-solver (license: reimplement) · AIrsenal github.com/alan-turing-institute/AIrsenal + bpl-next · FPL Review docs docs.fplreview.com (xMins, solver settings) · AlpsCode alpscode.com/blog/intro-to-fpl-analytics/ + hindsight-optimization · Ramezani & Dinh arXiv:2505.02170 · Frees et al. arXiv:2405.02412 · Matthews et al. AAAI 2012.
