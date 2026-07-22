# State of the Art in FPL Points Prediction and Squad Optimization

**Research date:** 2026-07-22. Compiled as the model-design blueprint for `fpl-ai-assistant` (2026-27 season).
**Status legend:** ✅ CONFIRMED (≥1 primary source read directly, cross-checked where noted) · ⚠️ UNCERTAIN / secondary-source only / unannounced.

---

## 0. Season context (2026-27) that constrains our model design

- ✅ As of 2026-07-22 the official FPL API (`https://fantasy.premierleague.com/api/bootstrap-static/`, verified live via curl) **still serves 2025-26 data** (GW1 deadline `2025-08-15`, 38 events, 841 elements). The 2026-27 game has **not launched yet**; launch is expected "by the weekend or the start of the week commencing 27 July" ([Fantasy Football Scout, 2026-07-20](https://www.fantasyfootballscout.co.uk/2026/07/20/when-will-fpl-go-live-for-2026-27), [Fantasy Football Fix](https://www.fantasyfootballfix.com/blog-index/fpl-2026-27-game-launch-announced-late-launch/)).
- ✅ **Chips 2025-26 (and confirmed unchanged for 2026-27):** two full sets of Wildcard / Free Hit / Bench Boost / Triple Captain, one set playable in each half of the season (first set expires at the GW19 deadline). Verified in the live API `chips` array: `wildcard(2-19), wildcard(20-38), freehit(2-19), freehit(20-38), bboost(1-19), bboost(20-38), 3xc(1-19), 3xc(20-38)` and in [FFS 2026/27 rule-change coverage](https://www.fantasyfootballscout.co.uk/2026/07/20/fpl-2026-27-5-rule-changes-new-features-announced).
- ✅ **Defensive contribution (DefCon) points continue unchanged in 2026-27**: +2 pts (capped) for DEF reaching **10 CBIT** (clearances+blocks+interceptions+tackles) in a match; +2 pts for MID/FWD reaching **12 CBIRT** (CBIT + ball recoveries) ([premierleague.com](https://www.premierleague.com/en/news/4361991/whats-new-in-202526-fantasy-defensive-contributions), cross-checked vs [FFS explainer](https://www.fantasyfootballscout.co.uk/2025/07/18/fpl-2025-26-defensive-contributions-what-is-a-tackle) and the API's `defensive_contribution` / `defensive_contribution_per_90` element fields).
- ✅ **2026-27 announced changes** ([FFS](https://www.fantasyfootballscout.co.uk/2026/07/20/fpl-2026-27-5-rule-changes-new-features-announced), [allaboutfpl](https://allaboutfpl.com/2026/07/2026-27-fpl-season-new-rules-changes-whats-new-in-fpl/)): live-updating official ranks/mini-leagues; projected bonus added to scores after 20' of each match; BPS rebalance (1 BPS per **3** CBI instead of per 2; the −1 BPS "was tackled" penalty removed; GK +1 BPS for big-chance saves, 2 BPS per save + 1 extra for in-box saves). An official **price-change predictor** is coming ([FFS](https://www.fantasyfootballscout.co.uk/2026/07/21/fpl-2026-27-price-change-predictions)). ⚠️ Full 2026-27 scoring table not verifiable until the API flips over.
- ✅ **Free-transfer rules since 2024-25:** FTs accumulate up to a maximum of **5**; wildcard/free-hit weeks roll the FT balance (do not reset it). In 2025-26 FPL granted everyone 5 FTs at the AFCON gameweek (GW16) — encoded as `AFCON_GW = 16` with a special-case rule in the community solver ([solioanalytics/open-fpl-solver `dev/solver.py`](https://github.com/solioanalytics/open-fpl-solver), read directly). ⚠️ Whether 2026-27 has a similar mid-season FT top-up event is unannounced.
- ✅ Squad rules (from live API `element_types` + `game_settings`): 15 players = 2 GKP / 5 DEF / 5 MID / 3 FWD; XI must field 1 GKP, ≥3 DEF, ≥2 MID(*min play*), ≥1 FWD; max 3 per club; £100.0m budget; sell price = purchase + ⌊profit/2⌋.

---

## 1. Academic literature

### 1.1 Matthews, Ramchurn & Chalkiadakis (AAAI 2012) — the founding paper

"Competing with Humans at Fantasy Football: Team Formation in Large Partially-Observable Domains" ([AAAI proceedings](https://ojs.aaai.org/index.php/AAAI/article/view/8259), [PDF](https://eprints.soton.ac.uk/340382/1/fantasyFootball2012cr.pdf)). ✅ Models FPL as **sequential team formation under partial observability**: belief distributions over player abilities, **Bayesian Q-learning** with **Value of Perfect Information (VPI)** to trade exploration/exploitation over the season. Their agent ranked **around the top percentile against ~2.5M human players** — still the canonical demonstration that a fully automated agent can compete at elite level. Uses a points model decomposed per player and a knapsack-style team formation step each week.

### 1.2 NTNU MSc thesis (Kristiansen et al., 2018) — the first serious multi-period MILP

"Developing a Forecast-Based Optimization Model for Fantasy Premier League" ([NTNU Open, handle 11250/2577003](https://ntnuopen.ntnu.no/ntnu-xmlui/handle/11250/2577003)). ✅ Contributions (per NTNU record and citing papers, e.g. [ORiON 2024](https://journals.co.za/doi/abs/10.5784/40-1-004)):
- A complete **mathematical model of the FPL rules** (squad/lineup/captain/transfers/chips as integer constraints).
- Solved with a **rolling-horizon heuristic** (re-solve each GW with a truncated look-ahead) because the full 38-GW problem is intractable.
- Three forecast generators feeding the optimizer: (i) recency-weighted average points per player, (ii) multivariate regression on explanatory variables, (iii) a combined variant.
⚠️ The exact author list is muddled in secondary sources ("Kristiansen, Gupta, Eilertsen" per one citation); I could not confirm a co-author named "Gunnes", and the NTNU/NVA record page would not render. Treat "Kristiansen/Gunnes" as an unverified attribution; the thesis itself is confirmed. ⚠️ Reported final points/rank numbers were not retrievable.

### 1.3 OpenFPL — Groos (2025), the strongest public prediction benchmark ✅

"OpenFPL: An open-source forecasting method rivaling state-of-the-art Fantasy Premier League services", Daniel Groos, [arXiv:2508.09992](https://arxiv.org/abs/2508.09992) (submitted 2025-07-29; [HTML](https://arxiv.org/html/2508.09992v1), code: [github.com/daniegr/OpenFPL](https://github.com/daniegr/OpenFPL)). Read in full; all numbers below verified against the paper.

**Architecture.** One ensemble **per FPL position** (GKP, DEF, MID, FWD, plus the 2024-25 "assistant manager" position). Base learners: **XGBoost and Random Forest regressors**. Model selection via "K-Best Search" (K=10) per cross-validation fold; the final prediction is the **median of 50 individual models** (top-K models × 5 folds). Targets and features MinMax-scaled to [0,1]. Sample weighting by entropy-based discretization of the target (2/3/4/3/5 bins for GK/DEF/MID/FWD/AM).

**Features.** Public data only (official FPL API + Understat): GK 196 features, DEF/MID/FWD 206, AM 122. Categories: player-specific (FPL points, minutes, ICT, penalties, saves, BPS, bonus, shots, xG, xA, cards…), team-specific (goals, xG, deep completions, PPDA), opponent-specific (xGA, deep allowed, PPDA att/def), match status (availability %). Every feature is **averaged over 1-, 3-, 5-, 10- and 38-match windows** — the paper's key feature-engineering idea (multi-horizon form).

**Hyperparameter search spaces** (exact): RF `n_estimators∈{200,400,800}`, `max_depth∈{10,20,None}`, `min_samples_split∈{2,5}`, `min_samples_leaf∈{1,2,5}`, `max_features∈{'sqrt',0.2,1.0}`, `bootstrap∈{T,F}`; XGB `n_estimators∈{300,600,1200}`, `max_depth∈{3,5,7}`, `lr∈{0.01,0.05,0.1}`, `subsample∈{0.5,0.75,1.0}`, `colsample_bytree∈{0.5,0.75,1.0}`, `min_child_weight∈{1,5}`, `gamma∈{0,0.1}`, `reg_lambda∈{1,5}`.

**Evaluation protocol (a template for our backtests).** Train on 2020-21→2023-24 with 5-fold CV **split by team allocation** (16 team-seasons per fold — prevents same-team leakage across folds); **prospective test on GW32-38 of 2024-25**. Errors reported per return category, not just overall: *Zeros* (0 pts), *Blanks* (≤2), *Tickers* (3-4), *Haulers* (≥5).

**Results (1-GW horizon, RMSE/MAE), OpenFPL vs FPL Review (commercial benchmark) vs last-5-average baseline:**

| Category | OpenFPL | FPL Review | Last-5 baseline |
|---|---|---|---|
| Zeros | 0.818 / 0.427 | **0.689 / 0.237** | 0.791 / 0.270 |
| Blanks (≤2) | 1.291 / 0.749 | **1.189 / 0.597** | 1.400 / 0.652 |
| Tickers (3-4) | **1.517 / 1.127** | 1.594 / 1.227 | 2.136 / 1.645 |
| Haulers (≥5) | **5.142 / 4.317** | 5.172 / 4.381 | 5.613 / 4.709 |

Hauler RMSE by position (OpenFPL vs FPL Review): GK 5.678/5.040, DEF 5.062/5.016, MID **5.274/5.559**, FWD 5.235/**4.621**, AM **4.598**/4.906. Separate models per forecast horizon (1/2/3 GWs ahead); 1-GW-ahead lowers Zeros RMSE by **15-25%** vs 3-GW-ahead.

**The paper's headline diagnosis (load-bearing for us):** OpenFPL matches or beats a leading commercial service for players who *return* (>2 pts), but **loses badly on Zeros because it lacks a minutes model** — it only uses FPL API availability tags, not proprietary "expected minutes". Minutes prediction is where the remaining commercial edge lives (§4).

### 1.4 Ramezani & Dinh (2025) — MILP + classical time-series ✅

"A data-driven framework for team selection in Fantasy Premier League", [arXiv:2505.02170](https://arxiv.org/abs/2505.02170) (May 2025, rev. Jan 2026; earlier title "Data-Driven Team Selection in FPL Using Integer Programming and Predictive Modeling Approach"). Deterministic and **robust** MILPs choosing XI + bench + captain under budget/formation/club-quota constraints; player value estimators compared: linear regression on match-performance features, **ARIMA**, exponential smoothing, Monte Carlo simulation, and a hybrid of realized points + model predictions. Tested on 2023-24 with rolling windows; **ARIMA with a constrained budget and rolling window was most consistent out-of-sample**; robust variants and hybrid scoring "mixed". Single-GW focus; chips/multi-week planning left as future work. Useful mainly as evidence that *classical* forecasting + exact optimization is a workable but not SOTA baseline.

### 1.5 Frees, Ravella & Zhang (2024) — deep learning & transfer learning ✅

"Deep Learning and Transfer Learning Architectures for English Premier League Player Performance Forecasting", [arXiv:2405.02412](https://arxiv.org/abs/2405.02412). Ridge and LightGBM baselines vs a **CNN over recent-gameweek feature windows**; CNN wins with fewer inputs and "very strong Spearman correlation with player rankings"; claims SOTA over prior EPL-forecasting literature at time of writing. **Transfer learning from news text (The Guardian) underperformed** the pure statistical models — a caution against naive news-NLP features. Top features: recent FPL points, influence, creativity, threat, playtime.

### 1.6 Other 2024-2026 academic items (lower priority)

- ✅ "An optimisation approach towards soccer Fantasy Premier League team selection", ORiON 40(1), 2024 ([journals.co.za](https://journals.co.za/doi/abs/10.5784/40-1-004)) — cites the NTNU thesis; standard IP selection model.
- ⚠️ "Optimizing Fantasy Premier League lineups using machine learning and linear programming" ([jsju.org](https://www.jsju.org/index.php/journal/article/view/2160)) — ML + LP pipeline, details not extracted.
- ⚠️ Uppsala MSc thesis 2025 "Enhancing Fantasy Premier League Strategies…" ([diva-portal PDF](https://uu.diva-portal.org/smash/get/diva2:1972615/FULLTEXT02.pdf)) — not read in full.
- ⚠️ A 2025 IJCSS article combining transformer-based news-sentiment + injury info with boosting/NN for FPL ([DOAJ record](https://doaj.org/article/4f0e8be4a94f4f69b14c9058df7a46ce)) — surfaced in search, not read.
- ✅ Adjacent classic: Hunter, Vielma & Zaman, "Picking Winners in Daily Fantasy Sports Using Integer Programming" ([arXiv:1604.01455](https://arxiv.org/pdf/1604.01455)) — variance-seeking lineup construction for DFS; relevant to our mini-league/rank-chasing mode where we may want *max P(exceed threshold)* rather than max EV.

---

## 2. Serious open-source projects

### 2.1 AIrsenal (Alan Turing Institute) ✅ — the reference Bayesian pipeline

[github.com/alan-turing-institute/AIrsenal](https://github.com/alan-turing-institute/AIrsenal) (active for 2025-26; README and source read directly).

**Team model** — `bpl-next` ([github.com/anguswilliams91/bpl-next](https://github.com/anguswilliams91/bpl-next)), a NumPyro **Dixon-Coles (1997) variant**:

  p(y_h, y_a) = τ(y_h, y_a) · Poisson(y_h | a_h·b_a·γ_h) · Poisson(y_a | a_a·b_h)

with per-team latent **attack a_i**, **defence b_i**, per-team home advantage γ_i (LogNormal prior), Dixon-Coles low-score correlation τ, and a **bivariate hierarchical prior** on (log a_i, log b_i) with correlation ρ (Beta-transformed prior) and optional team covariates X_i. Hyperpriors N(0,1)/half-normal. Fit by MCMC (NUTS). Output: full scoreline probability grid per fixture.

**Player model** — `airsenal/framework/player_model.py` (read in full): a **trinomial Dirichlet-Multinomial model of goal involvement**. For each goal a player's team scores while he's on the pitch, he either scored it, assisted it, or neither: θ_p = (p_score, p_assist, p_neither) ~ Dirichlet(α), with position-level empirical-Bayes α. Minutes enter the likelihood: for a match with m minutes played, P(score) = θ₁·(m/90), P(neither) = θ₃·(m/90) + (90−m)/90. Two implementations: NUTS MCMC (NumPyro) and an exact **conjugate version** (posterior = Dirichlet(α + weighted involvement counts)), with **exponential time-decay weighting of past matches (ε = 0.2 per unit time-diff)** and prior strength `n_goals_prior = 35` (~a team's goals in 10 matches).

**Expected points** = appearance pts (from a minutes estimate) + attacking pts (Σ over scorelines: P(scoreline) × Σ over goal-involvement permutations: P(perm|θ_p) × pts, position-scaled) + defending pts (P(CS) × position CS value, minus conceded-goal penalties from the scoreline grid). ([NOTES.md](https://github.com/alan-turing-institute/AIrsenal/blob/main/NOTES.md))

**Optimization** — *not* a MILP: an enumerative/heuristic **tree search over transfer strategies** for N weeks ahead (`optimization_utils.py`, read directly): strategies are sequences like (0, 1, 2, "W", "F", "Bx", "Tx") per GW; expected squad points computed with future-GW discounting relative to the root GW; **bench weighted by fixed sub-appearance probabilities `{"GK": 0.03, "Outfield": (0.65, 0.3, 0.1)}`**; FT logic updated for the ≥2024-25 rules (max 5, WC/FH preserve balance). Historic performance: "well inside the top 30%" in early seasons ([Turing blog](https://www.turing.ac.uk/news/airsenal)); ⚠️ recent-season finishing ranks not confirmed (Turing pages 403'd).

**Takeaways for us:** the cleanest open probabilistic decomposition; slow (MCMC) and its fixed bench weights + heuristic search are inferior to the community MILP (§5); its minutes handling is simplistic (recent-appearance heuristics).

### 2.2 The community multi-period MILP solver ✅ — sertalpbilal/FPL-Optimization-Tools → solioanalytics/open-fpl-solver

The de-facto standard "solver" used by the elite community. **The repo moved**: `sertalpbilal/FPL-Optimization-Tools` now redirects to [github.com/solioanalytics/open-fpl-solver](https://github.com/solioanalytics/open-fpl-solver) (176 stars, last pushed 2026-03-19; Apache-2.0 for personal use, **commercial use requires a license from info@fploptimized.com**). I read `dev/solver.py` (1,244 lines) in full; the complete formulation is transcribed in §5. Companion material: Sertalp B. Çay's [AlpsCode blog](https://alpscode.com/blog/intro-to-fpl-analytics/) and [hindsight-optimization post](https://alpscode.com/blog/hindsight-optimization/), plus a YouTube series. Inputs are CSV projections with `{gw}_Pts` and `{gw}_xMins` columns — **compatible with FPL Review and Mikkel Tokvam data exports** ([Çay on X](https://x.com/sertalpbilal/status/1819967159306498488)).

### 2.3 OpenFPL code ✅

[github.com/daniegr/OpenFPL](https://github.com/daniegr/OpenFPL) — trained models + inference for the §1.3 paper (paper says models freely available; abstract page mentions MIT/CC-BY licensing ⚠️ license detail unverified). This is the best drop-in public xPts baseline to benchmark our own model against.

### 2.4 theFPLkiwi ✅ — open component-decomposition projections

[github.com/theFPLkiwi/theFPLkiwi](https://github.com/theFPLkiwi/theFPLkiwi), [projections site](https://thefplkiwi.github.io/webpage/). Free weekly projections; xPts = **xMinPts + xGoalAssists + xCleanSheetPts + xDefConPts + xDiscPts** (explicit component sum, including a DefCon component since 2025-26); **minutes are manually entered** by the author from lineups/formation reading — i.e., even good free models outsource the minutes problem to a human. Named alongside FPL Review and Tokvam as one of the community's reference models ([fplform resources](https://fplform.com/fpl-resources)).

### 2.5 fpl-prediction (Paul Solomon / @solpaul7) ✅

Medium series "How to win at FPL using data" ([Part 1: deep learning forecasts](https://medium.com/@sol.paul/how-to-win-at-fantasy-premier-league-using-data-part-1-forecasting-with-deep-learning-bf121f38643a), [Part 2: team selection](https://medium.com/@sol.paul/how-to-win-at-fantasy-premier-league-using-data-part-2-picking-your-team-d68759498fe3)). Deep-learning forecasts trained on the public historical FPL dataset, team optimization via (initially) Excel Solver; philosophy: maximize expected points across simulated seasons. Historically used the **vaastav/Fantasy-Premier-League** GitHub data dump (the community's standard historical dataset).

### 2.6 Other

- `lazyfpl` (PyPI) — small optimizer, not SOTA.
- KnightAdz/Premier-League-Fantasy-Football and many student repos — not load-bearing.

---

## 3. Elite/commercial tools and the consensus xP architecture

### 3.1 The tool landscape as of mid-2026 ✅

- **FPL Review** ([fplreview.com](https://fplreview.com/)) — still operating; toolset relaunched for 2025-26 with "upgraded modeling and extended planning horizons" ([@fplreview on X](https://x.com/fplreview)). Products: Massive Data projections (premium), free model, Transfer Solver + Linear Optimiser, docs at [docs.fplreview.com](https://docs.fplreview.com/). ⚠️ Rumors of shutdown were **not** confirmed — treat FPL Review as alive for 2026-27.
- **Mikkel Tokvam's "Transfer Algorithm"** — Patreon-distributed projections used by top managers; "over a dozen key statistics… minutes played, big chances, goal attempts, chances created, shots, key passes, touches" ([Trademate/FFS coverage](https://tradematesports.medium.com/trademate-sports-introduction-to-the-fantasy-premier-league-transfer-algorithm-9ab7c2f868b8)). Solver-compatible export format.
- **FPL Optimized** ([fploptimized.com](https://fploptimized.com/)) — Sertalp Çay's site: EV comparisons, live GW EV, season highlights, hindsight/foresight transfer reviews; the commercial arm of the open solver.
- **LiveFPL** ([livefpl.net](https://www.livefpl.net/) / [plan.livefpl.net](https://plan.livefpl.net/)) — created by "Ragabolly", an MIT physics instructor, running since 2018 ([FFS interview](https://www.fantasyfootballscout.co.uk/2025/10/13/meet-the-manager-ragaboly-creator-of-live-fpl)). Live rank engine (now partly obsoleted by official live ranks in 2026-27), **Elite page: live teams/EO of the best 1,000 managers of all time** ([plan.livefpl.net/elite](https://plan.livefpl.net/elite)), ownership-combination stats, price-change predictions updated every 10 minutes. ⚠️ No public methodology doc for its models.
- **FPL Form** ([fplform.com](https://fplform.com/)) — Nicholas Hope, since 2020; free predicted points + single/double transfer optimizers + wildcard solver; xG-based, methodology not fully disclosed.
- **Fantasy Football Fix, FPL Team ([fpl.team](https://fpl.team/)), Premier Fantasy Tools, FPL Copilot, FPL Oracle, fplwatch, fplpilot** — second tier of tools with predicted points + optimizers; several (Copilot, Pilot) now market **chip-strategy optimizers** that enumerate chip placements against projections.

### 3.2 FPL Review "Massive Data" — what is publicly documented ✅

From [docs.fplreview.com/the-model/projections/massive-data-model/](https://docs.fplreview.com/the-model/projections/massive-data-model/):
- Combines "traditional statistical methods and machine learning" over historical performance, **market (bookmaker) odds**, tactical analysis; accounts for team strength/style, player roles, penalty takers, rotation, data recency.
- Projects "all point scoring events **weighted by their respective probability**" — i.e., a probabilistic component decomposition, not a single regressor.
- Horizon: **up to 14 GWs ahead**. Updates: hourly (market data, news, xMins), overnight post-match, pre-deadline after press conferences.
- Claims lowest RMSE/MAE, highest R² among compared FPL models, and that model-based projections beat market-only baselines (their ["Ultimate Truth"](https://fplreview.com/ultimate-truth-how-fpl-models-perform-relative-to-a-perfect-model/) article compares models to a theoretical "perfect model"; ⚠️ page 403'd — exact numbers unverified; OpenFPL's Table (§1.3) independently confirms FPL Review's accuracy tier).

From [docs.fplreview.com/the-model/projections/xmins/](https://docs.fplreview.com/the-model/projections/xmins/): **xMins = mean of 1,000 simulations** per player per match mixing start/cameo scenarios (51 xMins ≡ a blend, not a literal 51-minute start), with rotation events, injury proneness decay, role security; user-editable; **EV is deliberately non-linear in xMins**.

From [docs.fplreview.com/the-model/solvers/settings/](https://docs.fplreview.com/the-model/solvers/settings/) (their solver = same family as §5): time decay recommended **0.80-0.95, default 0.85**; FT value ~0-2 pts; bank value 0.00-0.20 per £1m; **autosubs via "full probability based calculations (determined by xMins, availability and usage characteristics)"** rather than fixed bench-slot factors; vice-captain and sub weights exposed; "Solve Lines" returns top-N alternative plans.

### 3.3 The consensus architecture: decomposed expected points

Every serious model (AIrsenal, FPL Review, Kiwi, and what we should build) computes, per player p, per match m:

**xP(p,m) = Σ_buckets P(minutes bucket) × [ appearance + attack + clean sheet + conceded + saves + DefCon + bonus + cards/OG ]**

Formally, with minutes distribution collapsed to the FPL-relevant buckets {0, 1-59, 60+} (probabilities q₀, q₁, q₂; and conditional expected minutes μ₁, μ₂):

1. **Appearance:** E = 1·q₁ + 2·q₂.
2. **Attacking returns.** Estimate per-90 scoring/assisting intensities λ_g(p,m), λ_a(p,m). Two equivalent routes:
   - *Rate route (Kiwi/most ML models):* λ_g = f(npxG/90 blend, penalty share, shot volume, finishing prior) × **fixture multiplier** F_att(m) = E[team goals in m]/E[team goals vs average opponent], with E[team goals] from Dixon-Coles (§2.1) or bookmaker over/unders. Then E[goal pts] = pts_goal(pos) × λ_g × (E[min]/90); assists analogous (pts=3). 2025-26/26-27 goal points: GKP/DEF 6, MID 5, FWD 4.
   - *Conditional route (AIrsenal):* Σ_scorelines P(score) × E[involvements | team goals, θ_p, minutes] — cleaner propagation of uncertainty, costlier to compute.
3. **Clean sheets.** P(CS_team) = Σ_y P(y_conceded = 0) from the DC grid (or bookmaker CS odds, de-margined). Player CS points require the 60+ bucket: E = pts_CS(pos) × q₂ × P(CS | on-pitch window) with pts_CS = 4 (GKP/DEF), 1 (MID). Conceded penalty for GKP/DEF: −1 per 2 goals conceded while on pitch: E = −Σ_k P(concede 2k or 2k+1 …) computed from the same Poisson grid (use ⌊y/2⌋ expectation).
4. **Saves (GKP).** 1 pt per 3 saves. Model shots-on-target-against ~ Poisson(λ_SoT(opponent, venue)), saves = SoT − goals: E[save pts] = E[⌊saves/3⌋] over the joint grid, or approximately E[saves]/3 with a small convexity correction.
5. **Defensive contribution (new, must-have for 2026-27).** For DEF: E = 2 × P(CBIT ≥ 10 | minutes); MID/FWD: 2 × P(CBIRT ≥ 12 | minutes). Model per-90 count rate ν(p) (API gives `defensive_contribution_per_90`), scale by expected minutes and opponent-possession/game-state factors (defensive counts rise when the team defends more), and use a **negative binomial** (counts are overdispersed) for the threshold probability. Kiwi adds it as `xDefConPts`; OpenFPL's 2025-26 refresh treats it via features. One 2025-26 season of realized data exists for calibration.
6. **Bonus.** Either (a) empirical mapping: E[bonus | position, event profile] learned from history (bonus is a *rank* within the match by BPS, so it must be modeled relative to other players' expected BPS — simulate the match's BPS ranking), or (b) regression on the same features as xP (what tree-ensemble models like OpenFPL implicitly do). **The 2026-27 BPS rebalance (CBI 1/3, no tackled-penalty, GK save BPS up) breaks naive historical bonus mappings — retrain/recalibrate on rule-adjusted BPS.**
7. **Cards/OG:** small negative expectations from per-player card rates (−1 yellow, −3 red) and OG priors.
8. **Multi-match GWs:** xP(GW) = Σ over that GW's fixtures (this is what makes DGWs valuable and blanks worthless — the optimizer sees it automatically).

The FPL Copilot explainer independently documents the identical decomposition ("xG, xA, clean sheet probability, minutes likelihood, bonus patterns… 50+ data points per player per GW") ([fplcopilot.com](https://fplcopilot.com/blog/expected-points-explained)); Marcus Leadboot's public build logs ([v1](https://medium.com/@marcusleadboot/modelling-xpts-in-fpl-gameweek-1-01fd2179eac6), [v2](https://medium.com/@marcusleadboot/modelling-xpts-in-fpl-version-2-0-e7d8cd738e75)) are worked examples.

**Design verdict:** decomposition (probabilistic components) beats monolithic regression at the *architecture* level because it (a) isolates the minutes problem, (b) lets us splice in odds/market data where it's strongest (team goals, CS), (c) survives rule changes by retraining only the affected component (e.g. DefCon, BPS). But OpenFPL proves a well-engineered gradient-boosted monolith gets within ~2-8% RMSE of a commercial decomposed model *except on Zeros* — so a pragmatic hybrid is: **ML for per-90 rates + explicit probabilistic minutes/CS/DefCon/bonus layers.**

---

## 4. Minutes prediction — the hardest sub-problem

**Why it dominates:** OpenFPL's only decisive loss to FPL Review is the Zeros category (RMSE 0.818 vs 0.689; MAE 0.427 vs 0.237 — a 1.8× MAE gap), explicitly attributed to lacking expected-minutes inputs ([arXiv:2508.09992](https://arxiv.org/html/2508.09992v1)). A 6-pt-EV player with 60% start probability is a 3.6-pt asset; every downstream number (xP, autosubs, captaincy, solver plans) is linear-or-worse in minutes error.

**Known approaches, in increasing sophistication:**
1. **Availability flags only** (OpenFPL): FPL API `status`, `chance_of_playing_next_round` (25/50/75/100 news buckets), `news` text. Necessary, insufficient — the API does not distinguish starter vs bench.
2. **Manual expert entry** (theFPLkiwi; FFS predicted line-ups): human reads press conferences, rotation patterns, formations. Still competitive — evidence the signal is largely *news-shaped*, not stats-shaped.
3. **Simulation with structured factors** (FPL Review xMins): 1,000 sims per player mixing start/cameo branches with rotation events, injury-proneness decay, "role ownership" security ([docs](https://docs.fplreview.com/the-model/projections/xmins/)). This is the commercial SOTA interface: output q₀/q₁/q₂ + μ, user-overridable.
4. **Supervised classifiers:** predict P(start) with features: recent start share (1/3/5/10-match windows), minutes trajectory, days rest, competition congestion (UCL/cups), manager rotation history at position, price/ownership (⚠️ leakage-prone, §8), new-signing/returning-from-injury flags, opponent strength (rotation correlates with easy cup ties), season phase. Then P(60+|start) and E[cameo minutes] as second-stage models. Academic support is thin — closest is lineup prediction from physiological data (79% accuracy with XGBoost, [MDPI Computers 11(3):40](https://mdpi.com/2073-431X/11/3/40/htm)) — so this is mostly community craft.
5. **Hierarchical Bayesian:** player-level start propensity nested in team-level rotation regimes; posterior updates from each team-sheet observation; naturally handles small samples after transfers/injuries. No public reference implementation (AIrsenal's is simpler); ⚠️ this is a build-it-ourselves area.
6. **Crowd/news fusion:** the OpenFPL paper explicitly names "crowd-sourced minutes forecasts and web-scraping AI agents" as the future direction. Press-conference scraping + LLM extraction into (injury, fitness-doubt, rotation-hint) features is the obvious 2026 upgrade — note §1.5's warning that *naive* news-text transfer learning underperformed; extract structured facts, not embeddings.

**Recommended formalization for our model:** per player-match, predict a categorical (start, bench-cameo, bench-unused, out) with a start-minutes distribution conditional on start (mixture at 90 + early-hook mass) — this yields q₀,q₁,q₂,μ exactly as §3.3 needs, plus P(plays 0) for autosub/vice math (§6).

---

## 5. Multi-GW squad optimization — the exact community MILP ✅

Transcribed from [`dev/solver.py`](https://github.com/solioanalytics/open-fpl-solver) (read in full). Sets: players p, gameweeks w ∈ {next, …, next+H−1} (horizon H, typical 3-8), bench slots o ∈ {0,1,2,3} (0 = GK), FT states s ∈ {0..5}.

**Decision variables (all binary unless noted):**
- `squad[p,w]`, `squad_fh[p,w]` (separate free-hit squad), `lineup[p,w]`, `captain[p,w]`, `vicecap[p,w]`, `bench[p,w,o]`
- `transfer_in[p,w]`, `transfer_out_regular[p,w]`, `transfer_out_first[p,w]` (price-modified players: first sale at selling price, later at buy price — the "multiple-sell fix")
- `in_the_bank[w]` (continuous ≥0), `fts[w]` (integer 0-5) + `fts_state[w,s]` one-hot, `ft_above_ub[w]`, `ft_below_lb[w]` (big-M indicator pair), `penalized_transfers[w]` (integer ≥0), `transfer_count[w]`
- Chips: `use_wc[w]`, `use_bb[w]`, `use_fh[w]`, `use_tc[p,w]` (TC is per-player!)

**Core constraints:**
- Squad composition: Σ_p squad = 15; per-position `squad_select` (2/5/5/3); ≤3 per club; FH squad mirrors all of these ×`use_fh[w]`.
- Lineup: Σ lineup = 11 + 4·use_bb[w]; formation bounds `squad_min_play ≤ lineup_type_count ≤ squad_max_play (+use_bb)`; lineup ⊆ squad (or FH squad in FH weeks); bench: exactly one GK in slot 0, one player per outfield slot, lineup+bench ≤ 1 per player; captain, vice ∈ lineup, captain≠vice.
- Squad continuity: `squad[p,w] = squad[p,w−1] + transfer_in[p,w] − transfer_out[p,w]`; no transfers during FH week; buy/sell budget flow `itb[w] = itb[w−1] + sold − bought`; FH affordability constraint against previous squad value.
- **FT state machine** (2024-25+ rules): `raw_ft[w] = fts[w] − transfer_count[w] + 1 − use_wc[w] − use_fh[w]` (the `+1` becomes `+5` in the 2025-26 AFCON GW16 special case), then big-M (M=20) logic: raw > 5 ⇒ fts[w+1]=5; raw ≤ 0 ⇒ fts[w+1]=1; else fts[w+1]=raw. Hits: `penalized_transfers[w] ≥ num_transfers[w] − fts[w] − 15·use_wc[w]`.
- Chips: ≤1 chip per GW (`use_wc+use_fh+use_bb+Σ_p use_tc ≤ 1`); per-chip count limits over the horizon; `use_tc[p,w] ≤ captain[p,w]`; forced/allowed/banned chip-GW options; WC auto-detected as active from the user's API data.
- Rich optional constraints: banned/locked players (globally or per GW), booked transfers, no-transfer GWs, per-position transfer bans, max defenders per team (≤3 attack-covariance control), "no opposing play" (never field players facing each other — hard or penalty-weighted with linearized product variables), double-defense pick, ITB buffer, price-point picks, no GK rotation, weekly/horizon hit limits.

**Objective** (minimized as negative): with `decay_base` β (default **0.84**; "regular" mode β=1):

  max Σ_w β^(w−next) · [ gw_xp(w) − 4·penalized_transfers(w) + ft_gain(w) − ft_penalty(w) + 0.08·itb(w) − opposing_penalty(w) ]

where

  gw_xp(w) = Σ_p xPts[p,w] · ( lineup + captain + **0.1**·vicecap + use_tc + Σ_o benchweight_o·bench[p,w,o] )

- Default **bench weights {GK: 0.03, slot1: 0.21, slot2: 0.06, slot3: 0.002}** — i.e., static autosub probabilities (FPL Review's solver upgrades these to full per-player autosub probability calcs; AIrsenal uses (0.65,0.3,0.1) *relative* outfield weights — note these encode different things: probability the sub *comes on and the slot is reached*).
- Captain adds +1×xP (doubling), TC adds a further +1×xP, vice adds 0.1×xP (≈ P(captain doesn't play) — should equal q₀ of the captain; 0.1 is the community default).
- **FT continuation value:** each saved FT is worth `ft_value` (default **1.5 pts**, configurable per state via `ft_value_list`) — implemented as a state-value function `ft_state_value[s] = ft_state_value[s−1] + ft_value(s)` with the objective earning the *increment* `gw_ft_gain[w]`; this is what stops the solver burning the FT stack.
- **ITB value 0.08 pts/£1m/GW** — mild incentive to hold cash.
- Decay rationale (per [AlpsCode](https://alpscode.com/blog/intro-to-fpl-analytics/) and FPL Review docs): future xP is uncertain (projection error grows with horizon) and plans are re-solved every week, so future GWs are discounted; community default 0.84, FPL Review recommends 0.80-0.95 (default 0.85). Horizon: 5-8 GWs typical (their docs advise transfer depth "a couple of GWs short of the loaded projection horizon" to avoid end-of-horizon artifacts).

**Uncertainty handling ("sensitivity analysis"):** re-solve the MILP many times with projections perturbed by noise, then report *how often each move is optimal* ([AlpsCode](https://alpscode.com/blog/intro-to-fpl-analytics/); `run/sensitivity.py`, `run/simulations.py`). The noise model in `solver.py` is:

  Pts'[p,w] = Pts[p,w] + strength · Pts[p,w] · (92 − xMins[p,w])/134 · N(0,1)

i.e., relative noise scaled up for low-minutes players. Alternative-solution enumeration via iterative no-good cuts on this-GW transfers/lineup (`iteration_criteria`).

**Solvers & performance:** model built in `sasoptpy`, exported to MPS, solved with **HiGHS via highspy** (options: presolve on, parallel on, `mip_rel_gap`, time limit default 20 min); **Gurobi** optional. Older versions used CBC; the ecosystem migrated to HiGHS. Benchmarks: open-source MIP solvers (HiGHS/CBC/SCIP) are roughly **1-2 orders of magnitude slower than Gurobi/CPLEX**, with HiGHS ≈ 1 order behind Gurobi on Mittelmann benchmarks ([HiGHS discussion #1683](https://github.com/ERGO-Code/HiGHS/discussions/1683)); for FPL-sized instances (≈700 players × 8 GWs after pool filtering) HiGHS solves in seconds-to-minutes. **Player-pool pruning is essential and standard**: keep top-EV percentile (default top 10%), xMins lower bound (default total ≥100), EV-per-price quantile cut, always keeping current squad/locked/booked players.

⚠️ Note the licensing change: the community solver is no longer freely usable commercially (Apache-2.0 personal/educational; commercial license required). For `fpl-ai-assistant` we should implement our own formulation (the maths above is not copyrightable; the code is).

---

## 6. Captaincy, effective ownership, bench-order/autosub maths

**Effective ownership (EO):** EO_p = (teams selecting p + teams captaining p + teams triple-captaining p) / total teams, within any reference sample (overall, top-10k, elite-1k). E.g. 70% owned + 40% captained ⇒ EO 110% ([fantasyfootballpundit](https://www.fantasyfootballpundit.com/fpl-effective-ownership/), [FPL Oracle](https://fploracle.team/blog/effective-ownership-fpl)). Live EO by rank-sample: [LiveFPL](https://plan.livefpl.net/elite), [FotPrem](https://fotprem.com/fpl-effective-ownership).

**Rank-relative expected gain.** Your expected score movement vs a reference field is Σ_p (m_p − EO_p)·xP_p where m_p ∈ {0,1,2,3} is your multiplier on p. Consequences: a 100%-EO captain is rank-neutral however he scores; captaining a >100%-EO player *loses* rank when he hauls if you merely own him. Community decision heuristics quantified by [FPL Oracle](https://fploracle.team/blog/fpl-captaincy-strategy): differential captaincy (EO<40%) is for rank-chasing; above ~70% EO, matching the field is rank-protecting unless EV gap ≥ ~1.5-2 pts. **Design note:** our assistant should support both objectives — max EV (season points) and max E[rank move] = EV measured against an EO-weighted field, switchable by user goal (mini-league vs overall rank), echoing the variance-seeking logic of the DFS "picking winners" literature (§1.6).

**Vice-captain EV:** captain multiplier falls to the vice iff the captain plays 0 minutes: E[bonus from vice] = q₀(captain)·xP_vice. The community solver's fixed 0.1 weight ≈ assuming q₀(captain)=10%; a proper implementation uses the captain's actual P(0 min).

**Autosubs / bench order.** A bench player in slot o scores iff (a) some starter plays 0 minutes, (b) higher-priority bench players are either used already or formation-infeasible, (c) the resulting formation is legal (≥3 DEF, ≥2 MID, ≥1 FWD; GK swaps only GK). Exact EV requires summing over starter-absence subsets:
 E[autosub pts, slot o] = Σ_{S ⊆ XI} P(exactly S miss) · 1[slot o reached & legal | S] · xP_o
In practice: Monte Carlo over independent Bernoulli(q₀) absences (11 draws) — cheap and exact enough; this is precisely what FPL Review does ("full probability based calculations… rather than fixed factors per sub slot", [solver docs](https://docs.fplreview.com/the-model/solvers/settings/)). Static approximations (bench weights ≈ 0.21/0.06/0.002 for a nailed XI) understate bench value for rotation-risky squads. Community estimate: good bench ordering is worth **5-10 pts/season** ([FPL Copilot](https://fplcopilot.com/blog/expected-points-explained)). Bench Boost converts bench weights to 1.0, which is why BB EV = Σ_bench xP is solver-visible (§5, §7).

---

## 7. Chip timing — heuristics, solver practice, simulation

**Consensus heuristics** (converged across [FFS ultimate chip guide](https://www.fantasyfootballscout.co.uk/2026/04/09/the-ultimate-fpl-chip-strategy-guide-for-all-16-scenarios), [FPL Copilot](https://fplcopilot.com/blog/chip-strategy-guide), [fplpilot 2026-27 guide](https://www.fplpilot.com/blog/fpl-chip-strategy-2026-27), [FPL360](https://fpl360.com/2026/03/13/fpl-chip-strategy-when-to-use-bench-boost-triple-captain-free-hit-and-wildcard/)):
- **Wildcard** immediately before the biggest fixture swing (restructure at an inflection point, often also to set up BB/TC);
- **Bench Boost** on the double gameweek following a wildcard, with 15 playing assets (EV = Σ_{bench} xP, typically 6-20+ pts on a good DGW);
- **Triple Captain** on the premium captain's best fixture; the maths mildly favor a **DGW** (two draws from the goals distribution: TC EV = xP_cap over both fixtures) over a single juicy fixture, all else equal;
- **Free Hit** on big blank GWs (or a monster DGW you can't wildcard into).
- With the two-half chip structure (since 2025-26), each half needs its own plan and "chip deadlines" (GW19) create forced-usage dynamics late in each window.
- Quantified value (one tool's season analysis, method opaque ⚠️): optimal chip timing averaged **+49 pts** vs no chips, best case +73, worst +2 ([search-surfaced claim](https://www.nevermanagealone.com/playerpicks/14801/fpl-chip-strategy-the-optimal-time-to-play-wildcard-bench-boost-triple-captain-and-free-hit); treat as order-of-magnitude).

**Solver practice** (§5): chips are just binaries with limits, so the standard workflow is *scenario comparison*: run the MILP with `forced_chip_gws`/`allowed_chip_gws` sweeps and compare objective values per placement ("chip EV curves"), rather than trusting one joint solve — because chip EV is dominated by projection uncertainty 4-8 GWs out and by DGW/blank announcements (cup rounds) that aren't in the fixture file yet. FPL Copilot's "chip strategy optimizer tests every valid chip combination across remaining gameweeks" — same enumeration idea productized.

**Season-long simulation** (the more rigorous approach, our target): simulate seasons by sampling match outcomes + haul variance from the model's predictive distributions, run the weekly re-solve policy inside each rollout, and pick the chip policy maximizing E[season points] (or P(beat rival), for mini-leagues). No public tool documents a full implementation (⚠️); Solomon's "thousands of simulated seasons" phrasing (§2.5) and Ramezani & Dinh's Monte Carlo estimator are partial precedents. Hindsight ceiling for calibration: a perfect-foresight 2019-20 season scores **4,984 points** ([AlpsCode hindsight optimization](https://alpscode.com/blog/hindsight-optimization/)) vs ~2,200-2,500 for the average-to-good human — enormous headroom, most of it unreachable variance.

---

## 8. Backtesting methodology and achievable performance

**Walk-forward by GW is the only valid protocol.** For each historical GW t: build features strictly from data available before the deadline of t; predict; optimize; record realized points; roll forward (including price/FT/chip state). OpenFPL's design is the academic reference: team-partitioned CV for model development + a **prospective holdout (GW32-38 2024-25)** never touched during development (§1.3). The football-modeling literature adds: every feature must join on an *availability timestamp*, not just (team, season), because corrected stats and rescheduled fixtures flow backwards; calibrators must be fitted in-history and frozen ([footballproofai](https://footballproofai.com/research/walk-forward-football-model-validation)).

**Leakage traps specific to FPL:**
1. **Price encodes future information.** Price changes are driven by net transfers, which react to news (injuries, form, press conferences) *after* the data snapshot you may be training on. Using `now_cost`, `transfers_in_event`, `selected_by_percent` from an end-of-season dump as features for mid-season predictions leaks crowd knowledge. Mitigation: use GW-snapshot data (the vaastav dataset stores per-GW rows; ideally archive our own daily API snapshots — note prices are frozen pre-GW1, [FFS](https://www.fantasyfootballscout.co.uk/2026/07/21/fpl-2026-27-price-change-predictions)).
2. **News timing.** `chance_of_playing`/`news` fields mutate continuously; the value at scrape time ≠ value at the historical deadline. Only deadline-timestamped snapshots are safe.
3. **Retrospective stats revisions** (Opta corrections, bonus recalcs) and **fixture rescheduling** (a "GW25" fixture list scraped in May differs from what was known in February).
4. **Rule non-stationarity:** DefCon (2025-26) and the BPS rebalance (2026-27) mean pre-2025 bonus/DefCon targets need rule-adjusted reconstruction from raw event counts — never train those components on mixed-rule seasons naively.
5. **Own-model feedback:** evaluating a solver policy with the same projections that were fitted on the evaluation window flatters it; keep projection-fitting and policy-evaluation windows disjoint.

**Metrics that matter (in order):**
- Component calibration (minutes bucket log-loss; CS Brier; per-90 rate calibration).
- xP accuracy: RMSE/MAE **per position × return-category** (OpenFPL's Zeros/Blanks/Tickers/Haulers split exposes exactly where models differ; overall RMSE hides the minutes problem) + Spearman rank correlation within GW (what selection actually uses).
- Decision quality: season points of the full re-solve policy vs (a) last-5-average baseline + same solver, (b) average human (~2,200-2,300 pts in a typical season; top-10k finishes are typically ~2,550-2,700 ⚠️ varies by season), (c) hindsight optimum (~4,900+, §7).
- **Known anchors:** Matthews et al. agent ≈ top 1% of 2.5M (2012 rules); AIrsenal "well inside the top 30%" in its early public seasons ([Turing](https://www.turing.ac.uk/news/airsenal)); community solver users with elite projections routinely report top-10k trajectories (⚠️ anecdotal, survivorship-biased — no audited public study found). FPL Review's "Ultimate Truth" frames the ceiling: even a *perfect* probability model retains large irreducible RMSE because single-match FPL points are dominated by variance (⚠️ article inaccessible; framing corroborated by OpenFPL's Hauler RMSE ≈ 5 — you cannot predict hauls, only hazard rates).

---

## 9. Blueprint implications for `fpl-ai-assistant`

1. **Architecture:** decomposed xP (§3.3) with ML-estimated per-90 rates; Dixon-Coles/bpl-style Bayesian team model (or de-margined bookmaker odds where available) for team goals & CS; explicit DefCon (negative-binomial threshold) and simulated-BPS bonus components under 2026-27 rules.
2. **Minutes first:** invest most modeling effort in the start/cameo/absence model (§4); expose user overrides like FPL Review; output (q₀,q₁,q₂,μ) per player-match; consider LLM extraction from press-conference reports as a structured-feature source.
3. **Optimizer:** implement the §5 MILP from scratch (HiGHS/highspy; big-M FT state machine, chip binaries, decay β≈0.84-0.87, bench weights replaced by true autosub probabilities, vice weight = captain's q₀); horizon 5-8; pool pruning; sensitivity via the noise-resolve loop.
4. **Validation:** OpenFPL as the public benchmark to beat (their exact RMSE table, §1.3); walk-forward with deadline-snapshotted data; per-category metrics; policy backtest vs last-5 baseline and hindsight ceiling.
5. **Rank-awareness:** EO-adjusted objective mode for mini-league/rank goals (§6).
6. **Data:** official API (snapshot daily from launch week — expected w/c 2026-07-27), Understat xG, vaastav historical dump, odds feed if licensable.

---

## Appendix: primary sources read directly

- OpenFPL paper: https://arxiv.org/abs/2508.09992 (+ HTML full text) and repo https://github.com/daniegr/OpenFPL
- Community solver source: https://github.com/solioanalytics/open-fpl-solver (`dev/solver.py`, settings, repo metadata via GitHub API)
- AIrsenal: https://github.com/alan-turing-institute/AIrsenal (README, NOTES.md, `player_model.py`, `optimization_utils.py`); bpl-next: https://github.com/anguswilliams91/bpl-next
- FPL Review docs: https://docs.fplreview.com/the-model/projections/massive-data-model/ · https://docs.fplreview.com/the-model/projections/xmins/ · https://docs.fplreview.com/the-model/solvers/settings/
- Live FPL API: https://fantasy.premierleague.com/api/bootstrap-static/ (curl, 2026-07-22)
- arXiv:2505.02170 (Ramezani & Dinh), arXiv:2405.02412 (Frees et al.), AAAI 2012 Matthews et al.
- AlpsCode: https://alpscode.com/blog/intro-to-fpl-analytics/ · https://alpscode.com/blog/hindsight-optimization/
- premierleague.com DefCon 2026-27 confirmation: https://www.premierleague.com/en/news/4361991/whats-new-in-202526-fantasy-defensive-contributions
