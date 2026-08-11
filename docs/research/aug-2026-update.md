# August 2026 research pass — season state, SOTA delta, feature mining, upgrade plan

**Research date: 2026-08-11** (GW1 deadline 2026-08-21 17:30 UTC). Four parallel research
agents: season state, modeling SOTA delta since 2026-07-22, community-tool feature mining,
data-source audit. This file is the condensed synthesis; the upgrade plan at the bottom is
what the 2026-08-11 implementation pass built. Flags: CONFIRMED = primary source read.

## 1. Season state (as of 2026-08-11)

- **No rule/scoring/chip/API changes since launch** (CONFIRMED, live API diff). Chips:
  BB/TC playable GW1; WC1/FH1 from GW2 (`start_event: 2`); set 1 expires at GW19 deadline
  2027-01-02T13:30Z. `price_change_percent` exists per element (official predictor surface,
  all "0" until GW1 deadline — prices frozen).
- **All five launch absentees left the PL permanently**: Salah → Trabzonspor,
  Konaté/Cucurella/Bernardo Silva → Real Madrid, Gordon → Barcelona, Stones → Inter.
  None will be added. Vinícius Jr → Arsenal died ~10 Aug (renewed at Madrid).
- **Pool 555 → 577 elements** (22 added, 0 removed). Notables: Tzolis (ARS £6.5m),
  Gonzalo García (FUL £6.0m), Maeda (IPS £5.5m), Antonio Silva (BOU £5.0m). Intra-PL:
  **Bruno Guimarães NEW→ARS 8 Aug (£7.0m)**, Lacroix/Welbeck/Henderson → CHE,
  Nørgaard → EVE, Trafford → LEE, Lukić → IPS. Rogers → CHE, Semenyo → MCI, Eze → ARS,
  Guéhi → MCI (already reflected in API team codes).
- **GW1 injury landscape**: out unknown-return — Saliba, J.Timber, Rodri, Ekitiké,
  Kroupi Jr, Onana, Mitoma, Kulusevski, Xavi Simons, De Ligt, Ugarte, Gomez, Bradley.
  Doubtful 75% — Šeško, Grealish, Kudus, L.Martínez, Mukiele, Livramento. Suspended with
  dates — Christie until 29 Aug, Andersen until 30 Aug, Fofana until 6 Sep.
- **News grammar confirmed** (63 live samples): `"<Type> injury - Expected back <D Mon>"`,
  `"<Type> injury - Unknown return date"`, `"<Type> injury - <NN>% chance of playing"`,
  `"Suspended until <D Mon>"`, `"Has joined <club> …"`. Machine-parseable.
- **Template/EO**: Haaland £15.5m at 73.8% owned + dominant armband share → GW1 EO >100%
  near-certain. Bruno F. 48.3%, João Pedro 55.4%, Szoboszlai 44.5%, Gabriel 26.7%,
  Raya 31.0%. Scout must-haves: Haaland, Bruno F., Gabriel, João Pedro.
- **Chip consensus**: TC1 = Haaland home v promoted (GW3 COV h favourite, GW7 IPS h
  fallback); BB1 = GW1 or GW7-post-WC; WC1 = GW4 or GW6 (FFS leans GW6, after the break);
  FH1 = hold as insurance. Matches our sim's "flat surface, hold" verdicts.
- **Calendar**: merged 3-week international break **GW5 (18 Sep) → GW6 (10 Oct)**
  (CONFIRMED — resolves the earlier single-source uncertainty; break 26 Sep–10 Oct).
  Nov break GW10→GW11; long March gap GW30 (20 Mar) → GW31 (10 Apr). Midweek GWs 13, 18,
  20, 25, 28; GWs 17-20 in 12 days. No AFCON this season; **no FT top-up event announced**;
  no visible Feb winter break. No blanks/doubles scheduled yet (expect FA Cup R5/QF + EFL
  final as usual).
- **Promoted clubs**: every COV/HUL/IPS defender is £4.0m. Most-backed: Diop (IPS 18.8%),
  van Ewijk (COV 15.6%). DefCon-per-90 leaders from Championship: Hughes (HUL 11.8),
  Egan (HUL 10.1), Greaves (IPS 9.6). Haji Wright (COV, Championship top scorer) 1.8%.
- Total players 3.83m on 11 Aug (13.1m final last season — late surge expected).

## 2. Modeling SOTA delta (since 2026-07-22 pass)

- **OpenFPL still v1** (arXiv:2508.09992, no update, ~6 commits) — our benchmark stands.
  No new FPL prediction/optimization papers since the July pass (arXiv sweep).
- Possibly-missed: arXiv:2505.02170 (robust MILP formulation vs EV uncertainty sets,
  v3 Jan 2026); rank-math foundations arXiv:1604.01455 "Picking Winners" (maximize
  P(exceed threshold), not E[points]) + arXiv:2211.02417 "Diversity is Key".
- **vaastav/Fantasy-Premier-League ENDED weekly updates after 2024-25** (CONFIRMED,
  README) — season-start/January/season-end snapshots only. Our build is vaastav-only ⇒
  played-2026-GW ingestion must come from the FPL API itself (event/{gw}/live).
- **Active replacement dataset**: olbauday/FPL-Core-Insights — twice-daily refresh,
  2026-27 from day 1, CBIT components, ClubElo, status/news snapshots in git history
  (a free status-transition corpus for hazard modeling later).
- **sertalpbilal/FPL-Optimization-Tools superseded by solioanalytics/open-fpl-solver**
  (active, HiGHS). Its knobs we lack: transfer-tail control, binary fixture scenarios,
  opposing-play penalty, simulated price changes. **No EO/rank objective anywhere in
  open source** — genuinely open ground.
- FPL Review beats OpenFPL on Zeros/Blanks specifically via xMins quality (team news +
  congestion + odds), not architecture — validates our news-parsing priority.
- Yellow-card refinement idea: referee strictness random effect (assignments public
  pre-GW). Not implemented this pass.
- **2026-27 BPS v4 confirmed official** (tackled −1 removed; CBI 1-per-3; GK save 2 +1
  in-box +1 big-chance; pen save 8→7). `rules.BPS_2026` already encodes v4 — no change
  needed.
- theFPLkiwi publishes free weekly xPts/xMins — usable as an external calibration
  benchmark.

## 3. Feature mining (community tools → static-feasible ideas)

- **Official FPL 2026-27 now ships free**: price-change predictor (15-min updates), live
  ranks/mini-leagues, projected bonus after 20', squad assistant, FDR/ownership squad
  toggles. ⇒ do NOT build price predictors or live-rank tools.
- **CORS verified blocked** on all fantasy.premierleague.com endpoints (no ACAO header,
  `cross-origin-resource-policy: same-origin`) ⇒ client-side entry import needs paste-JSON
  flow or CI-side fetch; no direct browser fetch from GitHub Pages.
- Top static-feasible ideas (ranked): model-based fixture-difficulty ticker (from our own
  Dixon-Coles, beats the official 1-5 FDR); set-piece/penalty-taker tables (already in
  bootstrap: `penalties_order`, `direct_freekicks_order`,
  `corners_and_indirect_freekicks_order` + `*_text` — 63 players carry pen orders today);
  differential finder (xP vs ownership quadrant); MC captaincy comparison (reuse sampler:
  P(outscores), P(haul), ceiling/floor); EO/template tracker by rank tier (post-deadline
  CI sampling job — EO is deadline-frozen, needs no server); what-if transfer sandbox;
  rotation-pair finder; plan-robustness display (we already compute stability);
  hindsight/accuracy pages; projections CSV export + deadline RSS.
- LiveFPL's strategically valuable features (EO tables, template, chip usage by tier) are
  all deadline-frozen snapshots ⇒ static-feasible via a once-per-GW Actions job (only
  possible after GW1 — picks endpoints are empty pre-season).

## 4. What the 2026-08-11 implementation pass built (see STATUS.md for verification)

1. **In-season outcome ingestion** (`fplai.data.event_live`): archive + parse
   `event/{gw}/live` for finished GWs of the live season into canonical
   player_match/player_gw rows, spliced + persisted idempotently (refreeze until
   bootstrap `data_checked`); wired into `fplai refresh`. Closes the vaastav gap —
   form features stay alive from GW2.
2. **Availability v2** (`fplai.data.live`): parse news return dates/suspensions,
   per-(player, GW) gates from the player's team's actual kickoffs — suspensions hard-zero
   until the return GW then instantly available; dated injuries floor until return GW with
   a one-GW sharpness ramp; unparsed news keeps the linear-recovery fallback. Return info
   published per player.
3. **Set-piece orders**: penalties/DFK/corner orders + notes parsed from bootstrap into
   the live roster and published (`set_pieces.json`); P-badges + duty tables in the UI.
   (Day-over-day change flags: future work — daily snapshots already archive the raw
   fields, so the history exists.)
4. **Team fixture outlook**: `team_fixtures.parquet` persisted from the TeamModel's
   per-fixture predictions (λ for/against, CS%, win%, odds-blended flag) and published as
   `fixtures.json` → frontend Fixture Ticker (attack/defence/overall difficulty from OUR
   model, sortable, GW range slider).
5. **Ownership surfaces**: `selected_by_percent` through to `players.json`; players-page
   ownership column + differential view; EO context in the captain card.
6. **Captaincy comparison**: sampler-driven distributional comparison of top captain
   candidates published as `captaincy.json` (EV, P(haul ≥10), P(blank ≤2), P(best))
   → dashboard card.
7. **Solver**: `no_transfer_last_gws` tail constraint (horizon-edge churn guard,
   research follow-up #1) in SolveParams.
8. Research archive: this file.
