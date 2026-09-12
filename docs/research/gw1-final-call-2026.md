# GW1 Final Call — 2026-08-19 pass (deadline 2026-08-21 17:30 UTC)

**What this is:** the pre-deadline adjudication pass — fresh `fplai refresh` on the
2026-08-19 snapshot (595 elements), retrain (byte-identical: no new training rows),
six comparison MILP solves, a 6-agent research sweep (news delta, GW1 consensus,
SOTA methods, winning doctrine, season landscape + adversarial API verification),
and a starter fact-check on every low-ownership model pick. Companion files:
`aug-2026-update.md` (Aug 11 pass), `chip-strategy-verdict.md` (doctrine).

## The verdict the model shipped vs the call we make

`fplai optimize` (free solve, 2026-08-19): no-Haaland 3-4-3, captain Thiago,
65.2 GW1 xP, objective 237.2 — with three broken picks: **Rulli** (MCI £5.0 GK,
0.0% owned — verified BACKUP to Donnarumma; cold-start position×price prior
misreads him as a starter), **Chavarria** (CHE, verified rotation/bench),
**Khalaili** (CRY, signed Aug 15, clearance pending, verified backup).

**Comparison solves** (no-chips, tail=2, gap ≈ 0):

| scenario | objective (8 GW) | GW1 xP | note |
|---|---|---|---|
| free (d=0.84) | 237.18 | 65.16 | ships Rulli/Chavarria/Khalaili |
| Haaland locked | 232.74 | 65.60 | C: Haaland |
| free (d=0.90) | 279.77 | 65.16 | same GW1 squad — decay is a non-issue for the build |
| clean pool (3 banned) | 234.77 | 65.62 | |
| **clean + Haaland locked** | **230.06** | **65.60** | **the adopted structure** |

**Locking Haaland costs 4.4–4.7 objective pts over 8 GWs (~0.55/GW) and GW1 xP
is actually higher** (Haaland-c 8.19×2 beats Thiago-c 7.74×2). Against that:
Haaland is 69.9% owned with 62.7% of surveyed captaincy (Plan FPL n=3,708) —
GW1 EO ≈ 130%. One haul week costs a non-owner ~15–25 net pts vs the field.
The model has no EO term (known gap, `chip-strategy-verdict.md` §1); this pass
adjudicates it: **the EV cost of owning him is an order of magnitude smaller
than the field-risk of shorting him.** The 2025-26 champion's documented
formula (Haaland 27 GWs, ~zero hits, template captaincy) points the same way.

## The final GW1 squad (£100.0m, £0.0 ITB, 3-4-3, 65.6 GW1 xP)

XI: **Lammens** (MUN 5.0, verified No.1) · **Tarkowski** (EVE 6.0) · **Shaw**
(MUN 4.5) · **N.Williams** (NFO 5.0) · **B.Fernandes** (MUN 12.0) ·
**Szoboszlai** (LIV 7.0) · **E.Le Fée** (SUN 6.0, verified starter) · **Ampadu**
(LEE 5.5) · **Haaland** (MCI 15.5, **C**) · **Thiago** (BRE 8.0, **V**) ·
**Watkins** (AVL 8.0).

Bench (solver fodder Button/Jacquet/Dowell/Bidwell → same-£17.5 verified
starters): **Leno** (FUL 4.5, 9.5 xP/8GW) → **Hume** (SUN 4.5, 12.9) →
**O'Shea** (IPS 4.0, 7.9) → **Hughes** (CRY 4.5, 4.9). Club limits: MUN 3/3,
SUN 2, all others ≤2. Positions 2-5-5-3 ✓.

Chips GW1: **hold all** (sim 2026-08-19: BB1 P(beats hold) 0.17, TC1 0.10,
WC1 best GW13 +16.5±20.3, FH1 best GW14). Field: 67% of planned first-half BBs
go GW1-2 — letting the field burn BB on a flat week is fine by us.

## Decision-critical facts (all API-verified 2026-08-19)

- GW1 fixtures confirmed: ARS-COV (Fri), HUL-MUN, EVE-CRY, IPS-SUN, NFO-LEE,
  BRE-TOT (Sat), BHA-AVL, MCI-BOU, NEW-LIV (Sun), FUL-CHE (Mon).
- Avoided flags: Šeško d-75 (no pre-season minutes), Mukiele d-75 (Le Bris:
  bench), Doku d-75 (reported out weeks), Mount d-50, Saliba/Ekitiké out.
- Spurs sold Romero, Spence AND Vicario (loan) in 72h pre-GW1 — strengthens
  the Thiago (BRE v TOT) vice slot; new TOT GK undecided (Kinsky/Dubravka).
- Rodri left in July (API news_added 07-23), not this week; Grealish and
  L.Martínez cleared to 100%.
- Watkins carries a "short of pre-season minutes post-WC" flag (FFS, not API):
  Friday-presser checkpoint — fallback Igor Jesus (NFO 6.0, 19.3 xP/8GW) or
  Calvert-Lewin (LEE 6.0, 20.9).

## Season doctrine locked by the research pass

1. **Transfers:** ~zero hits; bank toward 5 FTs (champion + rank-math evidence:
   hits ≈ EV-neutral at best; the 5-FT bank is "most of a wildcard, free").
2. **Chips set 1 (expire GW19):** engineered single-GW peaks, not GW1 burns.
   Working plan: WC1 in the GW5→GW6 international break (3 weeks to plan; 5 GWs
   of new-manager data) unless the sim's GW13 keeps winning; TC1 on Haaland
   home v promoted — GW3 COV / GW7 IPS / GW16 HUL (kiwi-consensus ceiling week
   GW16, Hull weakest); BB1 on a WC-engineered bench week; FH1 held as
   insurance (sim: GW13-14) for injury clusters/fixture chaos.
3. **Chips set 2:** the structural block — Carabao-final BGW30 (2 clubs),
   FA-Cup-SF BGW33, catch-up DGWs ~GW34-36. WC2 ~GW31-32 to build into them.
4. **Captaincy:** highest-xP default (= template most weeks); differential
   strikes only when model xP gap ≤ ~1 AND rank calls for variance.
5. **EO posture:** template core + 2-4 held sub-15%-EO differentials (currently
   Tarkowski/Ampadu/Le Fée); climb mode GW1-30, protect only if inside top-50k
   from ~GW31.
6. **Weekly ops:** trust the 3×-daily CI verdicts; act after final pressers
   (evening before deadline); price-predictor flags are advisory only.

## Model-improvement backlog (from the SOTA sweep, ranked impact/effort)

1. ~~Bonus/BPS v4~~ already correct (`rules.BPS_2026`); recalibrate
   `BonusCalibration` once ≥2000 played 2026 rows exist (~GW7).
2. **FT/ITB objective terms** (`ft_value_list` concave {2:2.0,3:1.6,4:1.3,5:1.1},
   `ft_use_penalty` ≈0.2): +3-8 pts/season, effort S.
3. **Status-transition hazard model** for undated news + post-return minutes
   ramp (FPL-Core-Insights git history as corpus): +5-15 pts/season, effort M —
   the single biggest modeling gap vs FPL Review (their edge is xMins ops).
4. **Scenario-averaged EV (SAA)** for captain/chip legs from the existing 1000
   rollouts: +3-8 concentrated in chip weeks, effort S-M.
5. **Cold-start ownership guard**: flag (never auto-ban) any solver pick with
   ownership <1% at price ≤£5.5 whose minutes come from priors — this pass
   caught Rulli/Chavarria/Khalaili/Button/Jacquet/Dowell/Bidwell by hand.
6. **Binary fixture scenarios** (open-fpl-solver recipe) for the BGW30/33
   rearrangement block, effort S-M — needed from ~GW25.
7. **Price-change ingestion** (official predictor → transfer-timing advice):
   +2-6 pts/season via team value, effort S.
8. **Rank-objective/differential mode** (Picking-Winners candidate enumeration
   + EO field model): decisive for endgame rank pushes, effort M — build by
   mid-season. **No EO/rank objective exists in open source — genuinely open ground.**
9. Deprioritized: box-robust MILP (arXiv:2505.02170's own results are negative);
   referee effects (~1-3 pts/season — fold into next card-model refit).

Full agent reports: workflow `fpl-deep-research-aug19` (session transcripts);
key sources: FPL API (primary), FFS team-news 08-17, Plan FPL survey n=3,708
(08-19), FPL Oracle rank-math series, PMC7928501 skill study, champion
interviews (Ibsen/Budisin/Jahangirov), open-fpl-solver repo, FPL Review docs.
