# Chip Strategy Verdict — Model Decisions vs Evidence

**Synthesis date:** 2026-07-23 (launch day +0/+1; GW1 deadline 2026-08-21 17:30 UTC).
**Inputs:** `gw1-consensus-2026.md` (community/ownership landscape), `chip-timing-evidence.md` (realized 2025-26 two-set-season data + doctrine), `solver-horizon-practice.md` (reference-solver methodology), `../STATUS.md` (model output + self-documented caveats).
**Model under review:** multi-GW MILP, horizon 8, decay 0.84, chip EV via forced-chip re-solves within GWs 1-8 only (known greedy-within-horizon limitation; set-1 chips actually run to GW19).

**Verdict scale:**
- **SUPPORTED** — independent sources land in the same place as the model.
- **CONTESTED** — defensible with real mechanism and/or named backers, but most elite practice differs.
- **LIKELY_ARTIFACT** — better explained by the greedy-horizon/decay limitation than by genuine EV.
- **INSUFFICIENT_EVIDENCE** — cannot be adjudicated from the collected evidence.

## Scoreboard

| # | Decision | Verdict |
|---|---|---|
| 1 | No Haaland GW1 (buy GW6) | **CONTESTED** |
| 2 | Thiago captain GW1 | **CONTESTED** |
| 3 | Promoted-opponent loading | **SUPPORTED** |
| 4 | Bench Boost GW1 (+11.1) | **CONTESTED** |
| 5 | Wildcard GW2 (+4.1) | **LIKELY_ARTIFACT** |
| 6 | Triple Captain GW3 Watkins (+7.6) | **LIKELY_ARTIFACT** |
| 7 | Free Hit GW5 (+2.0) | **LIKELY_ARTIFACT** |
| 8 | All four set-1 chips by GW5 (package) | **LIKELY_ARTIFACT** |
| 9 | Horizon 8 @ decay 0.84 (as transfer-planning settings) | **SUPPORTED** (with caveats) |

The clean split: the *squad-construction* decisions are defensible-to-endorsed; the *chip cascade* is where the known solver limitation shows through. The two structural outliers (no Haaland, four-chip burn) "both trace to the same solver artifact" (gw1-consensus §8), but they do not deserve the same verdict — the Haaland call has a real mechanism behind it; the cascade mostly does not.

---

## 1. No Haaland at GW1, buy at GW6 — CONTESTED

**Against (strong, near-unanimous):**
- 66.0% selected on day 2 (live API) — 1.5x the next player; with majority captaincy share, GW1 effective ownership plausibly >100%. A Haaland brace at home to Bournemouth costs a non-owner ~15-25 net pts vs the field in one week — the same order as the entire +11.1 BB GW1 edge (gw1-consensus §2).
- The Scout's mechanism argument: top-scoring player after six GWs in **all four** of his City seasons (~9 goals over GW1-6), with home fixtures v promoted COV (GW3) and IPS (GW7) inside the window (premierleague.com "MUST-HAVES").
- Every concrete recovered draft starts him (fpl.page, FFScout-associated, Onside); no prominent public defence of a Haaland-less GW1 was found anywhere (gw1-consensus §2, §4).
- The specific entry week is self-inflicted damage: GW6 is **MCI away at Liverpool**, City's hardest fixture of the opening 8; GW5 (SUN h) or GW7 (IPS h) strictly dominate as entry points (gw1-consensus §2). The GW6 choice smells of the FT-banking arithmetic (5 FTs ready at GW6) rather than fixture logic — an in-horizon artifact riding on a defensible structure.

**For (real, but thin on endorsement):**
- £15.5m is the highest price in FPL history and partly reflects "the lack of other premiums" (only forward above £9.0m) — the one structural counter-argument found in launch content, though its author still picked him (gw1-consensus §2).
- A spread squad can genuinely out-xP a Haaland squad on raw sum, and our model beat OpenFPL's published accuracy (STATUS: Zeros RMSE 0.798 vs 0.818) — the model is entitled to a contrarian xP view.
- Pep is gone at City (REPORTED) — the only structural uncertainty the community's own sceptics cite.

**Why CONTESTED, not LIKELY_ARTIFACT:** the no-Haaland structure is a coherent xP-maximizing position that a full-season solver could also reach; it is not manufactured by the horizon bug. But it is entangled with two artifact-tinged choices — the 15-playing-bodies build only makes sense because of BB GW1, and the GW6 entry week is the worst of the plausible windows. And the model carries no EO/variance term, so it is blind to the ~15-25 pt one-week downside tail it is accepting. If the chip cascade falls (below), the case for this structure weakens with it.

---

## 2. Captain Thiago (BRE h TOT) — CONTESTED

**For:**
- Internally coherent: Thiago is #2 GW1 xP in our model (7.62, behind only Haaland 8.14), and external xP orderings agree — Attacking Football has Haaland/Bruno above Thiago over the opening run, same order as ours once Haaland is out of the squad (gw1-consensus §3).
- The pick itself is community-endorsed: #3 most-selected forward (22.9%), "the dictator pick in the cheap-forward bracket", Brentford's GW1-5 FDR second only to Liverpool, opening home v Spurs (gw1-consensus §3).

**Against:**
- **No source found proposes him as GW1 captain** — he is on the radar as a pick, not an armband (gw1-consensus §3). Consensus captain is Haaland, with Bruno Fernandes (a HUL) the non-Haaland alternative — whom our plan does not buy until GW2.
- 9 of his 22 goals in 2025-26 were penalties (41%) — streaky output carrying our armband (gw1-consensus §3).
- Vs the field it is a stacked double differential (no Haaland + differential captain), maximizing variance exactly where the model has no variance term.

**Why CONTESTED:** conditional on the no-Haaland squad, captaining the squad's top xP player is the mechanically correct move and externally order-consistent — the model is not wrong *given its structure*. The contest is inherited from decision #1, plus the note that a plan wanting a promoted-stack captain had the option to buy Bruno for GW1 and chose not to.

---

## 3. Promoted-opponent loading — SUPPORTED

- "Promoted-opponent targeting is fully consensus — we are not contrarian here" (gw1-consensus §5). AllAboutFPL names MUN the standout opening run (back-to-back promoted opponents); premierleague.com pushes Bruno and Haaland on the same logic; Sunderland's IPS(a)→FUL(h) start makes their assets "viable differential options". Our MUN trio, SUN pair, and ARS pair all sit inside consensus.
- 11 of our 15 are top-45 most-selected; 13/15 "defensible or outright template" (gw1-consensus §8).
- **One labeling correction, no squad change implied:** NFO v LEE is *not* a promoted-opponent fixture — Leeds survived 2025-26. Re-label the NFO pair as "favourable home fixture" (gw1-consensus §5).

---

## 4. Bench Boost GW1 (+11.1) — CONTESTED

**For:**
- A real minority line with named backers and official acknowledgement: Nick Harris on the PL expert panel; Full90's structural case (no DGWs in the half, GW1 is the only week the whole 15 is built free, delaying costs transfers to rebuild a boostable bench); PL.com's 2026-27 chips article explicitly lists BB GW1 as legitimate (gw1-consensus §6; chip-timing §3.1).
- In a DGW-less half the BB timing surface is flat — elite 2025-26 BB usage was "far more scattered" with no consensus week — so BB GW1 "loses little *if* the bench is truly nailed; its risk is lineup uncertainty, not timing" (chip-timing §5).

**Against:**
- Majority-sceptical: ~1 for / 6 against / 3 conditional on the 2025-26 expert panel; 4.7% realized GW1 adoption; FFScout's core con: "Gameweek 1 is arguably the most challenging to predict" — lineup risk lands on exactly the £4.5-5.0m players a BB needs (chip-timing §3.1).
- Our +11.1 exceeds the community's 8-10 pt prior for a purpose-built GW1 BB (chip-timing §3.1), and the delta is measured under the decayed objective, where a GW1 placement enjoys weight 1.00 vs 0.295 at GW8 — a 3.4x position-in-window inflation (solver-horizon §3.4). The number needs an undecayed re-score before it is believed.
- Structural side-effect realized by design: FFScout warned a BB-GW1 build "potentially forc[es] an early Wildcard" — our model then wildcards in GW2, "the con realised by design" (gw1-consensus §5).

**Why CONTESTED, not LIKELY_ARTIFACT:** unlike the rest of the cascade, BB GW1 has independent backers, a genuine structural mechanism, and a flat opportunity-cost surface. The artifact contaminates the *magnitude* (+11.1) more than the *placement*.

---

## 5. Wildcard GW2 (+4.1) — LIKELY_ARTIFACT

**Against (mechanism + testimony):**
- A wildcard's payback extends over every remaining GW; truncating evaluation at GW8 systematically favours early placement (a GW2 WC improves 7 in-horizon weeks, a GW6 WC only 3 — the GW9-19 payback of a later WC is priced at zero). On top of that, the decayed objective halves an identical payoff placed at GW6 (weight 0.418 vs 0.840). This is the transfer-artifact both reference tools guard against, applied to the biggest structural move (solver-horizon §3.1-3.2).
- The reigning champion did exactly this — WC GW2 in 2025-26 — and publicly counts it as the thing to fix: "the way I did it the second half season is the way I should approach it from now on" (chip-timing §2.4). Doctrine names early WC the "common mistake"; elite pre-planning clustered WC GW6-7; the 2026-27 natural slot is the merged 3-week break between GW5 and GW6 (chip-timing §3.2).
- +4.1 is within the ~3 pt resolution at which chip plans are distinguishable (FPL Copilot: best vs second-best plan ≈ 3 pts; chip-timing §5).
- Internal tension: "a GW1 squad good enough to Bench Boost should not need tearing up seven days later — if WC2 is right, the GW1 build was wrong" (chip-timing §3.2).

**Steelman:** an early WC harvests price rises and structurally funds B. Fernandes one week after his best fixture. But price rises are not in the model's objective — so the model's *own stated case* (+4.1 decayed, in-window) is artifact-dominated even where an out-of-model justification exists. The mildest of the three artifact calls, but the mechanism fits the placement better than genuine EV does.

---

## 6. Triple Captain GW3 on Watkins (+7.6) — LIKELY_ARTIFACT

**Against:**
- "The clearest anti-pattern in our cascade" (chip-timing §3.3). Realized elite set-1 TC in 2025-26 was ~exclusively top-premium at a picked peak: Haaland GW6 (home v promoted Burnley) and GW17 — "a classic premium captaincy fixture" (Fix Top-50 post-mortem, chip-timing §2.2). The named 2026-27 windows are Haaland v COV (GW3) or IPS (GW7), or Bruno v IPS (GW2). "Nobody mentions Watkins TC" (gw1-consensus §6).
- The smoking gun is internal: **our own plan acquires Haaland in GW6, and his home-promoted fixture v IPS is GW7 — inside our horizon** — yet the solver still spends TC in GW3 on a 7.6 xP mid-premium. Under decay 0.84, GW7 carries weight 0.351: a TC Haaland GW7 worth ~8.5-9 raw scores ~3.0 in the objective vs Watkins GW3's 7.6 × 0.706 ≈ 5.4. The preference for GW3-Watkins over GW7-Haaland is manufactured by the discount, not by xP (solver-horizon §3.4).
- TC in a single GW ≈ +1x the captain's score, so TC-ing ~7.6 instead of a later ~8-9+ Haaland fixture gives up EV *and* the option value of waiting for form confirmation (chip-timing §3.3).

**Steelman:** none found — no external source, no realized-elite precedent, and the in-model comparison is decay-distorted. This is the decision the artifact explains most completely.

---

## 7. Free Hit GW5 (+2.0) — LIKELY_ARTIFACT

**Against:**
- +2.0 is below the ~3 pt plan-resolution noise floor (chip-timing §5) — "far below any sensible exercise threshold for a chip with 15 GWs of insurance/option value left" (solver-horizon §3.4).
- Doctrine is unanimous that FH is insurance/problem-solving ("Save for awkward weeks… Do not waste it on normal gameweeks" — FPL Pilot; the champion: "more about being risk-averse than being aggressive"). Even the blank-free 2025-26 first half saw the elite hold FH 13 weeks for a picked fixture-wave peak (GW13, triple City) (chip-timing §2.2, §3.4).
- Burning it GW5 liquidates the only insurance for GWs 6-19 immediately before the 3-week international break — which itself provides free replanning time (chip-timing §3.4).
- "A +2 in-horizon delta is exactly what a greedy solver produces when it cannot see the GW6-19 alternatives" (chip-timing §3.4). Any positive in-horizon delta gets burned when the alternative use is priced at zero.

**Steelman:** none. This is the textbook symptom, self-diagnosed in STATUS caveat #6.

---

## 8. All four set-1 chips by GW5 (package) — LIKELY_ARTIFACT

**Against:**
- "No elite manager, expert panel, or solver-community source endorses burning all four set-1 chips by GW5" (chip-timing, bottom line). Realized elite set-1 chips clustered **GW5-17**; the reference-solver author's own cautionary tale is a premature chip policy forcing a WC by GW6 for 31 points (chip-timing §2.2; solver-horizon §3.2).
- The failure mode the two-set system actually punished in 2025-26 was the **opposite** — drifting into the GW19 cliff (10% of the game BB'd GW19; ~half of all managers let set-2 chips die). "Our model over-corrects to the opposite extreme, which nobody exhibited" (chip-timing §2.3).
- Estimated cost of the cascade vs an evidence-based spread: **−3 to −18 xP** (labelled inference, chip-timing §5), before counting the variance-insurance value of a held FH.
- Mechanism fully accounts for the pattern: chips evaluated only inside GW1-8 (~40% of the true window) under a decay that inflates GW1 placements 3.4x over GW8 and prices GW9-19 alternatives at zero — "structurally guaranteed to cascade chips into the first few GWs… the textbook signature of this bias, not evidence that GW1-5 is genuinely the best chip real estate" (solver-horizon §3.4).

**Steelman (honest):** the two-set system genuinely lowers the penalty for early chips — no DGWs before GW19, flat peaks, chips can't be hoarded — and a model that schedules all chips deliberately inside the window is directionally right vs the mass market that lets chips die (chip-timing §6, steelman section). But "use them all by GW19 at chosen peaks" ≠ "use them all by GW5".

---

## 9. Horizon 8 @ decay 0.84 — SUPPORTED (with caveats)

**For:**
- Horizon 8 is **exactly the mainstream default**: open-fpl-solver ships `horizon: 8`; FPL Review evaluates 8-14 GWs with transfer depth 6; the FPL Optimized podcast works at 6; AIrsenal at 3 (solver-horizon §2).
- 0.84 is literally the reference solver's code fallback (`decay_base = options.get("decay_base", 0.84)`) and sits inside FPL Review's accepted 0.80-0.95 range (solver-horizon §1, §2.2).

**Caveats (real but do not overturn the verdict):**
- 0.84 is the myopic end: effective horizon ≈ 4.7 flat GWs, 77% of objective weight in GWs 1-5, GW8 weighted 0.295. The maintained defaults moved to 0.85 (FPL Review) / 0.90 (open-fpl-solver shipped). If multi-week plans like the GW6 Haaland entry matter, 0.90 (effective 5.7 GWs) is the evidence-backed nudge (solver-horizon §4).
- Missing standard guard: both references keep transfer depth 1-2 GWs shorter than the evaluation horizon (`no_transfer_last_gws: 2` / depth 6-of-8) to kill dead-end moves; we run full-depth (solver-horizon §3.1, §6).
- **The settings are fine for transfers and wrong for chips.** No serious practitioner lets a decayed within-horizon objective place chips: chips are off by default in the reference solver and chosen by forced sweeps over the full window, compared undecayed (FPL Review "literal" mode: decay→~1.0, values zeroed, because "different solver settings/frameworks/chip plans create incomparable evaluation scores") (solver-horizon §3.2). Do not "fix" chip myopia via decay retuning — fix it via the chip-evaluation pipeline (solver-horizon §7.4).

---

## Recommended engineering changes (ranked by evidence strength x effort; not implemented)

1. **Re-score the existing GW1-8 forced-chip solves in undecayed xP** before believing any delta (the +11.1/+7.6/+4.1/+2.0 all conflate chip value with position-in-window weight: GW1 weight 1.00 vs GW8 0.295). *Evidence: STRONG (FPL Review literal mode; `report_decay_base` incl. 1.0). Effort: SMALL — pure re-scoring of already-computed solves. Best ratio in the list.*
2. **Full-window chip sweeps: decouple chip valuation from the transfer solve.** For each chip and each feasible GW in the true window (BB/TC GW1-19; WC/FH GW2-19), forced-chip re-solve scored as undecayed Δ over a common GW1-19 evaluation window (coarse fixture-level projections for GW9-19 beat the current implicit zero), then feed chosen weeks into the 8-GW transfer solve as constraints. *Evidence: STRONG (uniform reference practice: open-fpl-solver forced-scenario workflow, FPL Copilot full-remaining-window MIP, AIrsenal `--wildcard_week 0` sweep). Effort: MEDIUM (projection extension + sweep loop; mind the 35-min solve budget — a lower-fidelity sweep per solver-horizon §7.8 is an acceptable first cut).*
3. **Add the transfer-depth tail** (`no_transfer_last_gws ≈ 2`): ban transfers in the final 1-2 horizon GWs while still scoring them. *Evidence: STRONG (both references' shipped defaults). Effort: SMALL.*
4. **Chip hold-value as an interim objective patch** if full sweeps are too slow at refresh time: terminal value per unused chip at GW8 ≈ estimated max Δ over GW9-19 (crude constants fine: single-GW BB ≈ 10, TC ≈ 6-8, FH ≈ 6, WC ≈ 8-12). Novel — no public solver implements chip terminal values — but precedented in spirit by `ft_value_list`/`itb_value`, and our own FT-banking logic already follows that pattern. Validate against #2. *Evidence: MEDIUM-LOW (negative finding on precedent). Effort: SMALL-MEDIUM.*
5. **Chip-timing guardrails as validation flags, not constraints:** warn on ≥2 set-1 chips before GW5; expect WC ~GW5-13 (the GW5→6 break is the 2026-27 natural slot), TC premium-only vs promoted at a swept peak, FH held unless Δ ≥ ~6 or a broken week; keep the GW19 hard stop so nothing drifts past the cliff (the mistake the mass market actually made). *Evidence: MEDIUM (doctrine + Top-50 realized timing). Effort: SMALL.*
6. **Decay sensitivity 0.84 vs 0.90 (and horizon 6 vs 8) + stability re-solves under projection noise;** only trust chip weeks and first transfers that survive across runs. Consider moving the default to 0.90 if GW6+ plans (Haaland entry) keep mattering. *Evidence: MEDIUM (`report_decay_base`, `run/sensitivity.py` patterns). Effort: SMALL-MEDIUM.*
7. **BB GW1 bench-nailedness check:** the chip's entire value sits on four ~£4.5-5.0m players in the least predictable GW of the season (Lammens must start); validate +11.1 against the 8-10 pt community prior after the undecayed re-score. *Evidence: MEDIUM (expert-panel conditionals keyed exactly on this). Effort: SMALL.*
8. **Surface an EO/field-risk annotation for anti-template positions** (e.g. "not owning Haaland at ~66%/EO>100% risks ~15-25 net pts on a brace week") — a report-level annotation, not an objective term, so the human sees the variance the xP-max objective ignores. *Evidence: MEDIUM (ownership math is confirmed; whether to price EO is philosophy). Effort: SMALL.*
9. **Plan-level fix regardless of the above:** if the no-Haaland line survives re-evaluation, move the entry week off GW6 (MCI at LIV) to GW5 or GW7, which dominate (gw1-consensus §2). *Evidence: STRONG for the fixture fact; SMALL effort (likely falls out of #2 automatically).*

---

## Honest-uncertainty register

- All quantified chip-cost figures (−3 to −18 xP for the cascade) are labelled inference from solver-based community estimates, not realized-outcome studies; "no published study quantifies 'all four chips by GW5 vs spread to GW19'" (chip-timing §5).
- Day-2 ownership is early-adopter-skewed; treat ranks, not levels, as signal (gw1-consensus §7). 2026-27 content is one day old; solver houses (FPL Review, planfpl) have published nothing for 2026-27 yet — re-scan in 1-2 weeks (gw1-consensus §9).
- The model beat OpenFPL's published accuracy on the walk-forward eval and its backtest beat both baselines (STATUS) — its *projections* have earned trust; the artifact findings here concern the *chip-decision layer* on top of those projections, and the two should not be conflated in either direction.
- No source states "chip horizon must be ≥ chip window" as an explicit rule — the practice is uniform but the principle is inferred (chip-timing §4); FPL Review's solver docs were 403-blocked, so their horizon guidance is uncited (chip-timing §7).
