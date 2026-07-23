# Horizon Length and Decay in Serious FPL Optimization — Practice Review

**Research date:** 2026-07-23 (2026-27 game launched ~1 day ago; GW1 deadline 2026-08-21 17:30 UTC).
**Question:** Is our multi-GW MILP setting of **horizon = 8 GWs, decay = 0.84** in line with serious practice, and what do practitioners do about the **greedy-chip-within-horizon** problem (chip EV evaluated only inside GWs 1–8 while set-1 chips run to GW19)?

**Evidence labels:** `CONFIRMED` = directly sourced and cited. `ANECDOTAL` = community practice / single-voice / inferred from structure. 2025-26 evidence is treated as directly comparable (first two-set chip season; 2026-27 confirmed to repeat the format — see §2.4).

---

## 1. Executive summary

- Our **8-GW horizon is exactly the mainstream default**: the sertalpbilal/open-fpl-solver ships `horizon: 8`, and FPL Review's ecosystem is built around ~8-GW working horizons (projections up to 14 GWs, planner display default 10, solver transfer depth default 6).
- Our **0.84 decay is literally the code-fallback default in the reference community solver** (`decay_base = options.get("decay_base", 0.84)` in `dev/solver.py`), but note the *shipped settings file* uses **0.9** and FPL Review's default is **0.85** (recommended range 0.80–0.95). 0.84 sits at the aggressive/short-term end of accepted practice.
- The user's intuition is arithmetically correct: **8 GWs at 0.84 decay ≈ a flat 4.7-GW horizon** (sum of weights = 4.70; GW8 carries weight 0.84⁷ ≈ 0.295; 77% of all objective weight sits in GWs 1–5).
- **No serious practitioner lets a decayed, within-horizon objective place chips freely.** The reference open-source solver ships with all chips **disabled by default** (`chip_limits: {bb:0, wc:0, fh:0, tc:0}`); chip decisions are made by **forced-scenario comparison**. FPL Review's docs prescribe a special **"literal" evaluation mode — decay set to ~1.0 and all auxiliary values zeroed — specifically because "different solver settings/frameworks/chip plans create incomparable evaluation scores."** AIrsenal exposes a chip sweep (`--wildcard_week 0` = "try playing the chip in all gameweeks").
- Our GW1 chip cascade (**BB GW1 → WC GW2 → TC GW3 → FH GW5; all four set-1 chips burned by GW5**) is exactly the failure mode this practice exists to prevent, and it contradicts 2025-26 comparable-season guidance, which spread first-set chips across roughly **GW5–GW17**. The solver author's own first-season retrospective: *"adding a premature chip policy logic forced me to use my WC by GW6. I received only 31 points in the GW that I used my first WC in."*
- **Nobody we could find implements a chip terminal-value/hold-value function** (negative finding after repeated searches). But the community *does* implement terminal values for the two other inter-temporal resources — free transfers (`ft_value_list: {2:2.0, 3:1.6, 4:1.3, 5:1.1}`) and bank (`itb_value: 0.08/£1m`) — so a chip hold value is a natural, precedented-in-spirit extension. The de-facto standard alternative is simpler: **evaluate chips by forced re-solves swept over the full chip window (here GW1–19), compared in undecayed xP.**

---

## 2. What horizons do practitioners actually run?

### 2.1 FPL Review (the commercial reference) — `CONFIRMED`

From [docs.fplreview.com — Solver Settings](https://docs.fplreview.com/the-model/solvers/settings/):

| Setting | Default | Notes (quoted) |
|---|---|---|
| Transfer Depth | **6** | "The number of future Gameweeks transfers can be considered in when running a solve." "GWs beyond this depth will contribute to the plans total 'Evaluation Score' however the solver will not consider making transfers in those GWs." |
| Time Decay | **0.85** | "A weighting factor that progressively discounts points in later GWs to account for uncertainty." Range: **0.80–0.95**. |
| FT Value | 1.75 | value of a saved transfer |
| Bank Value | 0.10 /£1m | |

- Guidance from the same docs (via search snippet of the settings/glossary): with an **8 GW projection horizon, "set a depth of 6–7"** so the solver doesn't over-plan the final weeks; and: **"Leaving a buffer of a week or two, from the loaded projection horizon, can prevent unrealistic 'dead-end' moves in the final weeks."**
- Their [Massive Data Model](https://docs.fplreview.com/the-model/projections/massive-data-model/) projects **"up to 14 gameweeks ahead"**; the site's [User Settings](https://docs.fplreview.com/the-site/user-settings/) default the planning display to **10 GWs (typical range 5–14)**.
- Net effect: *evaluation* horizon ~8–14 GWs, *decision* (transfer) horizon 6–7, future GWs decayed at 0.85.

### 2.2 sertalpbilal / solioanalytics `open-fpl-solver` (the open-source reference) — `CONFIRMED` (read from the repo itself)

Cloned and inspected 2026-07-23 ([github.com/solioanalytics/open-fpl-solver](https://github.com/solioanalytics/open-fpl-solver), same repo as the old sertalpbilal/FPL-Optimization-Tools):

| Setting | Shipped default (`data/user_settings.json` + `comprehensive_settings.json`) |
|---|---|
| `horizon` | **8** |
| `decay_base` | **0.9** (settings files) — but the **code fallback is 0.84**: `dev/solver.py:289 → decay_base = options.get("decay_base", 0.84)`; objective: `gw_total[w] * pow(decay_base, w - next_gw)` |
| `no_transfer_last_gws` | **2** → transfers banned in the last 2 GWs of the horizon (decision depth 6 of 8 — same pattern as FPL Review) |
| `ft_value_list` | {2FT: 2.0, 3FT: 1.6, 4FT: 1.3, 5FT: 1.1} — explicit **terminal/carry value for free transfers** |
| `itb_value` | 0.08 pts/£1m — terminal value for bank |
| `chip_limits` | **{bb: 0, wc: 0, fh: 0, tc: 0} — all chips OFF by default** |
| `use_wc/use_bb/use_fh/use_tc` | `[]` — chips are used by **forcing** them into specific GWs |
| `report_decay_base` | `[0.85, 1.0, 1.017]` — solutions are *re-scored* under several decays, including **undecayed (1.0)**, for comparison |

Notes:
- Our 0.84 is not exotic — it is the long-standing hardcoded fallback and has propagated into forks (e.g., [dbozbay/FPL-Optimization](https://github.com/dbozbay/FPL-Optimization) documents `decay_base: 0.84`). But the maintained user-facing default moved to **0.9**.
- The `report_decay_base` mechanism is direct evidence that practitioners **do not trust a single decayed number** when comparing plans; they re-score under decay 1.0.

### 2.3 Working practice, podcasts, AIrsenal, academia

- **FPL Optimized podcast** (sertalp + Bas): weekly planning content is framed over a **"next 6 GW horizon"** (e.g., [Episode 121, "GW29: Optimal FH and WC Plans"](https://fploptimized.transistor.fm/) — "They cover the usual content for the next 6 GW horizon"; Episode 87 "look at GW32-37 horizon"). Episode 73 (Pre-GW21 Q&A) explicitly lists **"Ideal horizon length"** and **"Ideal chip times"** as discussed audience questions — i.e., horizon choice is a live tuning question, not settled science. `CONFIRMED` (titles/descriptions; audio content not transcribed).
- **AIrsenal** (Alan Turing Institute): default lookahead is **3 gameweeks** — "by default predictions and transfers are calculated for the next 3 gameweeks" ([README](https://github.com/alan-turing-institute/AIrsenal)); the project rationale (per project docs/search snippet) is that 3 weeks is computationally achievable for its brute-force strategy search *"but also because things like form and injuries can change a lot in 3 weeks."* Chips are **not chosen by the optimizer freely**: you pass `--wildcard_week <GW>` etc., **"or use 0 to try playing the chip in all gameweeks"** — i.e., a sweep. `CONFIRMED`
- **Academic MILP work is behind community practice on this specific question:**
  - NTNU MSc thesis (Kristiansen, Gupta & Eilertsen 2018, ["Developing a Forecast-Based Optimization Model for Fantasy Premier League"](https://ntnuopen.ntnu.no/ntnu-xmlui/handle/11250/2577003)): season simulated with a **rolling-horizon heuristic** over GWs 1–35 of 2017-18; chips ("gamechips") modeled as a feature of the game. The archive is now JS-only; the exact lookahead length could not be re-verified today — cite as rolling-horizon precedent only. `CONFIRMED (approach), UNVERIFIED (exact K)`
  - Ramezani & Dinh 2025/2026 ([arXiv:2505.02170](https://arxiv.org/abs/2505.02170)): deterministic + robust MILPs for weekly line-ups; **chips excluded** ("All plots report cumulative team points (no chips)"), no decay, and multi-week rolling-horizon planning + chips are explicitly listed as **future work**. `CONFIRMED`
  - Uppsala thesis 2025 (["Enhancing Fantasy Premier League Strategies…"](https://uu.diva-portal.org/smash/get/diva2:1972615/FULLTEXT02.pdf), text-extracted): prototype **"does not account for the chips"**, deferring chip timing to the human; suggests dynamic programming / rolling-horizon planning as future work. `CONFIRMED`
  - Matthews, Ramchurn & Chalkiadakis (AAAI 2012) treated the season as a belief-state MDP (top ~1% offline vs 2010-11) — sequential-decision framing exists in the literature, but no usable horizon/decay guidance for the modern chip game. `CONFIRMED (existence)`
- **OpenFPL** forecasting paper ([arXiv:2508.09992](https://arxiv.org/html/2508.09992v1)): open-source projections are published for **1–3 GW horizons** only — relevant to how fast projection quality decays (see §5).

**Bottom line:** serious practice clusters at **evaluation horizons of 5–8 GWs (up to 14 for display), decision/transfer depth 1–2 GWs shorter, decay 0.80–0.95 with 0.85–0.9 the norm.** Our 8/0.84 is inside the envelope, at the myopic end. Nobody serious runs 15+ GW decision horizons; nobody serious runs undecayed long horizons for weekly decisions.

### 2.4 Context: the 2026-27 chip window we must evaluate over — `CONFIRMED`

[Premier League official — "What's happening with FPL chips in 2026/27"](https://www.premierleague.com/en/news/4679879/whats-happening-with-fpl-chips-in-202627): two sets of chips again; **"The first set of chips must be played before the Gameweek 19 deadline… at 13:30 GMT on Saturday 2 January"**; no carry-over; one chip per GW. So the true option window for set-1 chips is **GW1–19 (WC/FH from GW2)** — our chip evaluation currently sees only GWs 1–8 of it, i.e., ~40%.

---

## 3. Known horizon artifacts

### 3.1 End-of-horizon transfer artifacts — documented and standard — `CONFIRMED`

Both reference implementations independently keep **transfer depth strictly shorter than the evaluation horizon**:

- FPL Review: depth 6 default vs 8–14 GW evaluation; *"Leaving a buffer of a week or two… can prevent unrealistic 'dead-end' moves in the final weeks."* ([Solver Settings](https://docs.fplreview.com/the-model/solvers/settings/))
- open-fpl-solver: `no_transfer_last_gws: 2` shipped default with `horizon: 8`.

The artifact being prevented: with no future beyond GW_H, the solver happily makes moves whose payback lies outside the window (or fails to make moves whose payback lies outside it), because the terminal squad has zero continuation value.

### 3.2 The chip version of the artifact — how practice handles it

The chip artifact is worse than the transfer artifact because a chip is a **one-shot option whose exercise window (GW1–19) is much longer than any sane projection horizon**. Within-horizon chip optimization assigns **zero opportunity cost** to exercising the option now. Three independent lines of evidence on how practitioners handle this:

1. **Structural (open-fpl-solver): chips are off by default and forced by scenario.** `chip_limits` all zero; the workflow is: run a no-chip solve, then run solves with `use_wc: [gw]` / `use_bb: [gw]` / etc. for candidate GWs and compare (there is a whole tutorial video, ["Wildcard Optimization for FPL — FPL Optimization with Python #9"](https://www.youtube.com/watch?v=4PNcDUoRQOE)). If you *do* hand the solver a chip budget via `chip_limits`, it will place the chip inside the horizon — which is precisely why the default is 0. `CONFIRMED (defaults/code), ANECDOTAL (workflow intent)`
2. **FPL Review's sanctioned method: "literal" evaluation mode for chip comparisons.** From [Understanding Evaluation Score](https://docs.fplreview.com/the-model/solvers/evaluation-score/): to compare chip plans, switch to literal settings — **"FT Value, Burn FT Value, Future Info Value, Bank Value ⇒ set to 0.00"**, **"Time Decay set to 1.01 (cancelling out xMin decay effects)"**, transfer depth maximized — because **"Different solver settings/frameworks/chip plans create incomparable evaluation scores."** I.e., the commercial reference explicitly tells users that chip decisions must be scored in (approximately) **undecayed xP over a common window via separate solves per scenario**, not read off a decayed optimizer objective. `CONFIRMED`
3. **AIrsenal: chip timing by exhaustive sweep** — `--wildcard_week 0` tries the chip in **all** gameweeks. `CONFIRMED`

And one first-person cautionary tale from the reference-solver author (alpscode blog, ["Reflections from my first FPL season"](https://alpscode.com/blog/fpl-reflections/)): *"adding a premature chip policy logic forced me to use my WC (wildcard) chip by GW6. I received only 31 points in the GW that I used my first WC in."* `CONFIRMED (self-reported)`

### 3.3 Does anyone implement a chip hold-value / terminal value function? — negative finding

Repeated targeted searches (GitHub code/issues, docs, blogs, "terminal value", "opportunity cost", "chip hold") found **no public FPL solver with a chip terminal value**. What exists instead:

- **Terminal values for the *other* carried resources are standard:** `ft_value` / `ft_value_list` (value of rolling FTs: 2.0/1.6/1.3/1.1 pts) and `itb_value` (0.08 pts/£1m) in open-fpl-solver; FT Value 1.75 and Bank Value 0.10 in FPL Review; FPL Review even has a "Future Info Value" knob (value of *keeping* a transfer until more information arrives — an information-option value). `CONFIRMED`
- Chips are instead handled **outside the objective** by full-window scenario sweeps (§3.2).

Implication: a `chip_hold_value` term in our objective would be *novel but philosophically consistent* with existing FT/bank terminal values — however the evidence-backed route is the sweep, not the terminal value (see §7).

### 3.4 The decay-on-chips bias (analytic, applies directly to our numbers)

If chip deltas are computed by forced re-solves **under the decayed objective** (ours are), an identical chip payoff is mechanically inflated when placed early in the window: a +11 xP Bench Boost counts as **+11.0 in GW1 but +11 × 0.84⁷ ≈ +3.2 in GW8** — a 3.4× distortion *within* our own horizon, before even considering GW9–19, where the chip's value is implicitly **zero**. Combined effect: the solver is structurally guaranteed to cascade chips into the first few GWs. Our observed recommendation (BB GW1, WC GW2, TC GW3, FH GW5 — all four by GW5) is the textbook signature of this bias, not evidence that GW1–5 is genuinely the best chip real estate. `ANALYTIC — follows from our own model definition`

Corroborating 2025-26 comparable-season practice (first two-set season, chips also expiring GW19):

- [AllAboutFPL first-half chip guide](https://allaboutfpl.com/2025/09/2025-26-fpl-chip-strategy-guide-first-half-of-the-season/) recommended paths spread chips over **GW5–GW17** (e.g., BB GW5 → FH GW6 → WC GW8/13 → TC GW13/15/16), with WC at GW7/8 because that's when there's *"enough data to judge who's real."* `CONFIRMED (guidance)`
- [Fantasy Football Pundit GW1–19 chip guide](https://www.fantasyfootballpundit.com/best-fpl-chip-strategy-2025-26-gw1-to-gw19/) is the *most* aggressive mainstream take and still only endorses **BB in GW1 "with caveats"** (predictable line-ups + unlimited pre-season transfers to build for it), TC held for mid-window premium fixtures (GW6–GW15 targets), FH GW4+. `CONFIRMED (guidance)`
- [FPL Copilot chip data](https://fplcopilot.com/blog/chip-strategy-guide): single-GW BB averages **8–12 pts** (DGW BB 15–25); optimal-vs-random chip timing worth **20–30 pts**; "97% of optimized squads use Wildcard on fixture swings and 79% use Free Hit on blanks." Note in the set-1 window there are normally **no DGWs/BGWs** (first 2025-26 DGW was GW24 — FPL Optimized ep. 117), so set-1 BB/TC are single-GW plays and the *between-GW* differences are small — which cuts both ways: burning early costs less than it would in a DGW season, but a +2.0 FH (our GW5 number) is far below any sensible exercise threshold for a chip with 15 GWs of insurance/option value left. `CONFIRMED (stats are the site's own solver-derived claims), ANECDOTAL (threshold judgement)`

---

## 4. Decay × horizon arithmetic — is our effective horizon really ~5 GWs?

Yes. Weight table w_k = d^(k) for GW offset k = 0…7 (computed):

| d | GW1 | GW2 | GW3 | GW4 | GW5 | GW6 | GW7 | GW8 | Σ (8 GWs) | share in GW1–5 |
|---|---|---|---|---|---|---|---|---|---|---|
| **0.84 (ours)** | 1.000 | 0.840 | 0.706 | 0.593 | 0.498 | 0.418 | 0.351 | **0.295** | **4.70** | **77.4%** |
| 0.85 (FPL Review default) | 1.000 | 0.850 | 0.722 | 0.614 | 0.522 | 0.444 | 0.377 | 0.321 | 4.85 | 76.5% |
| 0.90 (open-fpl-solver shipped) | 1.000 | 0.900 | 0.810 | 0.729 | 0.656 | 0.590 | 0.531 | 0.478 | 5.70 | 71.9% |
| 0.95 | 1.000 | 0.950 | 0.902 | 0.857 | 0.815 | 0.774 | 0.735 | 0.698 | 6.73 | 67.2% |

- **Effective horizon (sum of weights): 8 GWs @ 0.84 ≈ a flat 4.7-GW solve.** The user's "~5 GWs" intuition is confirmed; GW8 is weighted 0.84⁷ = **0.295**.
- Deep-horizon + strong decay ≈ shallow horizon: 8 @ 0.84 (4.70) ≈ 6 flat GWs at d≈0.92, and is *more myopic* than the podcast's flat "6 GW horizon" working practice.
- The infinite-horizon weight budget at d=0.84 is 1/(1−0.84) = 6.25 GW-equivalents, so extending our horizon beyond 8 adds at most 6.25 − 4.70 ≈ 1.55 GW-equivalents — i.e., **at 0.84, lengthening the horizon changes transfer decisions only marginally; it is the chip evaluation that materially needs the longer window** (undecayed).
- Moving to d=0.90 at horizon 8 raises effective depth to 5.7 GWs and GW8 weight to 0.48 — the cheapest way to "lengthen" planning without touching horizon.

---

## 5. Are longer horizons actively harmful? (projection noise)

Evidence that far-out projections are materially worse / that decay exists precisely to encode this:

- FPL Review defines Time Decay as discounting later GWs **"to account for uncertainty"** ([Solver Settings](https://docs.fplreview.com/the-model/solvers/settings/)). `CONFIRMED`
- **OpenFPL** ([arXiv:2508.09992](https://arxiv.org/html/2508.09992v1)): *"For all methods, shorter horizon improves forecasting for low-return categories, especially for Zeros where one-gameweek-ahead predictions lower RMSE by 15-25% relative to forecasts three gameweeks ahead"*, while *"No systematic horizon effect is observed for the high-return categories."* The degradation is driven mostly by **minutes/availability information**, not by talent estimates — exactly the thing that changes with news. It also notes FPL Review shows a *stronger* horizon effect on low-return predictions (their minute-projection edge is a near-term edge). `CONFIRMED`
- AIrsenal deliberately caps at 3 GWs partly because *"form and injuries can change a lot in 3 weeks."* `CONFIRMED (project rationale)`
- alpscode: *"FPL is a noisy problem with huge variance"*; schedule chaos (postponements) wrecked far-out plans ([Reflections](https://alpscode.com/blog/fpl-reflections/)). `CONFIRMED (experience report)`
- "Fixture-run mirage": we found no rigorous study showing long horizons *systematically* mislead solvers into fixture-run traps, but the community norm of wildcarding **into** swings only once ~4–8 GWs of data exist (AllAboutFPL: WC GW7/8, *"enough data to judge who's real"*) is the practical hedge. `ANECDOTAL`

**The standard compromise, as practiced:** medium decayed horizon (6–8 GWs) for *transfers*, with a 1–2 GW no-transfer tail; **full-window, undecayed, forced-scenario sweeps for *chips***; re-scoring of candidate plans under multiple decays (`report_decay_base` including 1.0); optional sensitivity/randomized re-solves (open-fpl-solver ships `run/sensitivity.py` and `run/simulations.py`) to check that the recommended first move and chip week are stable under projection noise. No practitioner we found responds to noise by *lengthening* the decision horizon.

---

## 6. Scorecard: our current settings vs practice

| Ours | Practice | Verdict |
|---|---|---|
| Horizon 8 | 8 is the reference default; pods work at 6; AIrsenal 3; FPL Review evaluates 8–14 | ✅ Keep |
| Decay 0.84 | 0.85 (FPL Review default), 0.9 (open-fpl-solver shipped), range 0.80–0.95 | ⚠️ Acceptable but myopic end; effective horizon 4.7 GWs |
| Transfer depth = full 8 (implied) | Depth 6/8 (both references leave a 2-GW tail) | ❌ Add `no_transfer_last_gws ≈ 2` |
| Chips optimized within GW1–8, decayed objective, no hold value | Chips forced-scenario-swept over the full chip window, compared undecayed ("literal mode"); chips off by default in the reference solver | ❌ Biggest gap — this alone plausibly explains the BB1/WC2/TC3/FH5 cascade |
| FT banking logic (banks FTs for GW6 Haaland) | ft_value_list terminal values standard | ✅ Consistent in spirit |

---

## 7. Recommendations menu (ranked by evidence strength)

1. **[STRONG — mirrors FPL Review "literal" mode + open-fpl-solver workflow] Decouple chip valuation from the transfer solve.** For each chip c and each feasible GW g in the chip's true window (BB/TC: GW1–19; WC/FH: GW2–19), run a forced-chip re-solve and score **Δ(c,g) = undecayed xP(plan with chip at g) − undecayed xP(no-chip plan)** over a **common evaluation window covering GW1–19** (coarser projections for GW9–19 are fine — fixture-strength-level projections beat an implicit zero). Choose chip weeks from the Δ(c,g) curves, then **feed the chosen weeks into the 8-GW transfer solve as constraints** (`use_bb=[g]`-style), exactly like the reference tooling.
2. **[STRONG — FPL Review docs, `report_decay_base` including 1.0] Never compare chip scenarios on the decayed objective.** At minimum, re-score our existing GW1–8 forced-chip solves in undecayed xP before believing any "+11.1 BB GW1" style delta; our current deltas conflate chip value with position-in-window weight (GW1 chip weight 1.00 vs GW8 0.295).
3. **[STRONG — both references' shipped defaults] Add a transfer-depth tail:** ban transfers in the last 1–2 GWs of the horizon (`no_transfer_last_gws: 2`) to kill dead-end moves; keep evaluating all 8 GWs in the objective.
4. **[MEDIUM] Decay: either keep 0.84 knowingly (effective 4.7-GW planner) or move to 0.85–0.90 to match FPL Review/open-fpl-solver shipped defaults.** If GW6-Haaland-style multi-week plans matter, 0.90 (effective 5.7 GWs, GW8 weight 0.48) is the evidence-backed nudge. Do not "fix" chip myopia via decay — fix it via #1/#2.
5. **[MEDIUM — corroborated by 2025-26 guidance, FPL Copilot data] Chip-timing sanity guardrails as validation (not constraints):** flag any plan that burns ≥2 set-1 chips before GW5; expect WC ~GW6–13 (information + fixture swing), TC on a premium vs promoted sides mid-window, BB only where the sweep shows the week is near the GW1–19 max (single-GW BB ≈ 8–12 pts; between-week spread is small in a no-DGW window, so option value usually dominates small early edges — our FH GW5 "+2.0" is far below any sensible threshold).
6. **[MEDIUM-LOW — novel, but consistent with ft_value/itb_value precedent] Chip hold value as an interim objective patch** if full GW1–19 sweeps are too expensive at solve time: give each unused chip a terminal value at GW8 equal to (an estimate of) max over g∈9–19 of Δ(c,g) — even a crude constant (e.g., single-GW BB ≈ 10, TC ≈ 6–8, FH ≈ 6, WC ≈ 8–12, from FPL Copilot averages/2025-26 guides) converts "free" chips into "costly" ones and kills the cascade. No public precedent for chips specifically; treat as our own engineering, validated against #1.
7. **[MEDIUM — open-fpl-solver `run/sensitivity.py`, `randomized` options] Stability testing:** re-solve with noise on projections (and with horizon 6 vs 8, decay 0.84 vs 0.90) and only trust chip weeks / first transfers that survive across runs.
8. **[LOW — AIrsenal pattern] Cheap fallback if #1 is too heavy:** AIrsenal-style exhaustive chip sweep at lower fidelity (fixed squad or 1-transfer-per-week heuristic) across GW1–19 to get order-of-magnitude Δ(c,g) curves.

---

## 8. Sources

**Primary tool docs / code (highest quality)**
- FPL Review docs — [Solver Settings](https://docs.fplreview.com/the-model/solvers/settings/) (depth 6, decay 0.85, 0.80–0.95, buffer advice); [Understanding Evaluation Score](https://docs.fplreview.com/the-model/solvers/evaluation-score/) (literal mode: decay 1.01, values zeroed; "incomparable evaluation scores" warning); [User Settings](https://docs.fplreview.com/the-site/user-settings/) (10-GW default display, 5–14 range); [Massive Data Model](https://docs.fplreview.com/the-model/projections/massive-data-model/) (14-GW projections); [Intro to FPL Solvers](https://docs.fplreview.com/the-model/solvers/into-to-solvers/); [Solver comparison](https://docs.fplreview.com/the-model/solvers/solver-comparison/)
- [solioanalytics/open-fpl-solver](https://github.com/solioanalytics/open-fpl-solver) (cloned 2026-07-23): `data/user_settings.json`, `data/comprehensive_settings.json`, `data/README.md`, `dev/solver.py` (decay fallback 0.84 at line 289; chip constraint block; `report_decay_base`)
- [alan-turing-institute/AIrsenal README](https://github.com/alan-turing-institute/AIrsenal) (3-GW default; chip sweep flags)

**Practitioner voice (high quality, first-person)**
- alpscode (Sertalp Çay) — [Reflections from my first FPL season](https://alpscode.com/blog/fpl-reflections/) (premature WC-by-GW6 policy failure; noise); [On the meaning of optimal in FPL](https://alpscode.com/blog/optimal-in-fpl/) (8-GW planning horizon on FPL Review data)
- FPL Optimized podcast [feed](https://feeds.transistor.fm/fpl-optimized) — ep. 121 (6-GW horizon framing), ep. 87 (GW32–37 horizon), ep. 73 (Q&A: "Ideal horizon length", "Ideal chip times"), ep. 117 (first 2025-26 DGW = GW24), ep. 80 (optimizing chips with solvers)
- [Wildcard Optimization for FPL — FPL Optimization with Python #9 (YouTube)](https://www.youtube.com/watch?v=4PNcDUoRQOE)

**Academic (moderate quality for this question)**
- [NTNU thesis, Kristiansen/Gupta/Eilertsen 2018](https://ntnuopen.ntnu.no/ntnu-xmlui/handle/11250/2577003) (rolling-horizon heuristic, GWs 1–35 of 2017-18)
- [Ramezani & Dinh, arXiv:2505.02170](https://arxiv.org/abs/2505.02170) (chips/rolling horizon = future work)
- Uppsala thesis 2025, [diva2:1972615](https://uu.diva-portal.org/smash/get/diva2:1972615/FULLTEXT02.pdf) (chips out of scope, deferred to human)
- [OpenFPL, arXiv:2508.09992](https://arxiv.org/html/2508.09992v1) (1–3 GW horizons; 15–25% RMSE degradation for zeros at 3 GW vs 1 GW)
- [Matthews, Ramchurn & Chalkiadakis, AAAI 2012](https://ojs.aaai.org/index.php/AAAI/article/view/8259)

**Rules / season context**
- [Premier League — What's happening with FPL chips in 2026/27](https://www.premierleague.com/en/news/4679879/whats-happening-with-fpl-chips-in-202627) (two sets; set 1 dies at GW19 deadline, 2 Jan 2027)
- [What's new in 2025/26 Fantasy: two sets of chips](https://www.premierleague.com/en/news/4362027/whats-new-in-202526-fantasy-two-sets-of-chips)

**Community chip-strategy guidance (lower quality; comparable 2025-26 season)**
- [AllAboutFPL 2025/26 first-half chip guide](https://allaboutfpl.com/2025/09/2025-26-fpl-chip-strategy-guide-first-half-of-the-season/) (paths spreading chips GW5–17)
- [Fantasy Football Pundit GW1–19 chip guide](https://www.fantasyfootballpundit.com/best-fpl-chip-strategy-2025-26-gw1-to-gw19/) (BB GW1 "with caveats"; TC mid-window)
- [FPL Copilot chip strategy guide](https://fplcopilot.com/blog/chip-strategy-guide) (solver-derived chip stats: single-GW BB 8–12; optimal-vs-random 20–30 pts; enumerates chip combos "across your remaining gameweeks")
