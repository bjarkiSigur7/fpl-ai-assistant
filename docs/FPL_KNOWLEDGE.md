# FPL_KNOWLEDGE.md — Canonical Knowledge Base for fpl-ai-assistant

**Status:** synthesized 2026-07-22 from the five research documents in `docs/research/` (rules-2026-27.md was additionally adversarially fact-checked against the live FPL API, five official premierleague.com articles, Wayback API snapshots and secondary sources: 19/20 claims CONFIRMED, 1 corrected). This file is the single source of truth the codebase is built against. Where documents disagreed, the verified rules doc wins; conflicts resolved are noted inline.

**Game status as of 2026-07-22: the 2026-27 FPL game has NOT launched.** The API (`https://fantasy.premierleague.com/api/bootstrap-static/`) still serves completed 2025-26 data (`static_content_url` ends `/2025_26/`, all 38 events `finished: true`, 841 elements, 13,107,732 total players). Launch is imminent (rule changes announced 2026-07-20, price reveals began 2026-07-22).

**Re-verification protocol (mandatory at launch):** poll `bootstrap-static` daily; when `static_content_url` flips to `2026_27/`, re-extract and diff against this document: `game_settings`, `game_config.rules`, `game_config.scoring`, `chips[]`, `element_types[]`, `events[]` (GW1 deadline), `elements[]` (launch prices, new fields). **All element IDs reset at relaunch — `element.code` and `opta_code` are the stable cross-season identifiers.**

---

## Part 1 — 2026-27 Game Rules (machine-encodable)

Everything below is CONFIRMED (≥2 independent sources, official article, or direct API observation) unless explicitly marked UNCERTAIN. Baseline values come from the live 2025-26 API; the official 2026-27 announcements changed nothing outside BPS internals, live-scoring UX, and lockdown timing.

Primary sources:
- Changes overview: https://www.premierleague.com/en/news/4679873/all-you-need-to-know-about-changes-to-fpl-for-202627
- BPS changes: https://www.premierleague.com/en/news/4679946/whats-new-in-202627-fantasy-changes-to-bonus-points-system
- Position changes: https://www.premierleague.com/en/news/4679886/position-changes-for-202627-fantasy-premier-league
- Chips: https://www.premierleague.com/en/news/4679879/whats-happening-with-fpl-chips-in-202627
- Price Change Predictor (also confirms GW1 deadline + daily 00:00 price changes): https://www.premierleague.com/en/news/4680462/whats-new-in-202627-fantasy-price-change-predictor
- Live API: https://fantasy.premierleague.com/api/bootstrap-static/ , https://fantasy.premierleague.com/api/fixtures/

### 1.1 Scoring table (encode exactly; per-position)

No changes to base scoring were announced for 2026-27; these are the live 2025-26 `game_config.scoring` values.

| Action (API key) | GKP | DEF | MID | FWD |
|---|---|---|---|---|
| Played 1–59 min (`short_play`) | 1 | 1 | 1 | 1 |
| Played ≥60 min (`long_play`) | 2 | 2 | 2 | 2 |
| Goal scored (`goals_scored`) | **10** | 6 | 5 | 4 |
| Assist (`assists`) | 3 | 3 | 3 | 3 |
| Clean sheet (`clean_sheets`; requires ≥60 min on pitch, no goal conceded while on) | 4 | 4 | 1 | 0 |
| Per 2 goals conceded while on pitch (`goals_conceded`; −⌊C/2⌋) | −1 | −1 | 0 | 0 |
| Per 3 shot saves (`saves`; +⌊S/3⌋) | 1 | — | — | — |
| Penalty save (`penalties_saved`) | 5 | — | — | — |
| Penalty miss (`penalties_missed`) | −2 | −2 | −2 | −2 |
| Defensive contribution (`defensive_contribution`; once per match max) | 0 | **+2 at CBIT ≥ 10** | **+2 at CBIRT ≥ 12** | **+2 at CBIRT ≥ 12** |
| Yellow card (`yellow_cards`) | −1 | −1 | −1 | −1 |
| Red card (`red_cards`; includes any YC deduction — 2nd-yellow red = −3 total) | −3 | −3 | −3 | −3 |
| Own goal (`own_goals`) | −2 | −2 | −2 | −2 |
| Bonus (top-3 BPS in each match) | 1–3 | 1–3 | 1–3 | 1–3 |

Notes:
- **GK goal = 10 pts since 2024-25** ("increased this season from six to 10", https://www.premierleague.com/en/news/4058895; archived 2024-25 API confirms `GKP: 10`). Verifier correction: this was previously mis-dated to 2025-26. Value for 2026-27: 10.
- **DefCon:** CBIT = clearances + blocks + interceptions + tackles; CBIRT = CBIT + ball recoveries. GKs ineligible. Cap +2/player/match. Introduced 2025-26, retained unchanged for 2026-27 (a third-party "recalibration" claim was NOT corroborated by any primary source — treat thresholds as unchanged; re-verify at launch).
- **Assist definition v3** (simplified, 2025-26): assist stands if scorer receives with ≤1 defensive touch in between; handball-intent test removed. Carries into 2026-27.
- Red-carded players keep accruing goals-conceded deductions until full time.
- The API scoring dict stores per-unit values (`saves: 1`, `goals_conceded: -1`); divisors (per 3 saves, per 2 conceded) come from the rules text, not the JSON.
- Dead fields scoring 0: `starts`, ICT components, `expected_*`, all `mng_*` (Assistant Manager remnants).

### 1.2 Bonus points and the 2026-27 BPS matrix

- Top-3 BPS per **match** get 3/2/1. Ties: tie for 1st → 3,3,1; tie for 2nd → 3,2,2; tie for 3rd → 3,2,1,1. (https://www.fantasyfootballscout.co.uk/2025/07/21/what-are-fpl-bonus-points)
- New 2026-27: **projected** bonus shown live after 20' of each fixture; final bonus settles at lockdown (09:00 next day).

**The four 2026-27 BPS changes** (official + FFS impact analysis https://www.fantasyfootballscout.co.uk/2026/07/21/how-fpls-new-bonus-points-system-tweaks-would-have-affected-2025-26):

| BPS item | 2025-26 | 2026-27 |
|---|---|---|
| Being tackled | −1 each | **removed** |
| Clearances/blocks/interceptions | +1 per 2 CBI | +1 per **3** CBI |
| GK save (inside box) | +3 | +2 (any save) + 1 (in-box) = 3 net |
| GK save (outside box) | +2 | category removed (still +2 as "any save") |
| GK save of a "big chance" | — | **+1** (new) |
| GK penalty save | +8 | **+7** (big-chance +1 usually → 8 net) |

**Full BPS matrix, 2026-27** (2025-26 community-standard reconstruction with deltas applied — FPL has never published a canonical machine-readable BPS table; treat as best-available, UNCERTAIN at the margin):

| Action | BPS |
|---|---|
| Playing 1–60 min | 3 |
| Playing >60 min | 6 |
| Goal: GKP/DEF (and any penalty goal, all positions) | 12 |
| Goal: MID | 18 |
| Goal: FWD | 24 |
| Assist | 9 |
| GKP/DEF clean sheet | 12 |
| Match-winning goal | 3 |
| GK: any save | 2 |
| GK: save from inside box | +1 (net 3) |
| GK: big-chance save | +1 |
| GK: penalty save | 7 |
| Goal-line clearance | 9 |
| Tackle won | 2 |
| Per 3 CBI | 1 |
| Per 3 recoveries | 1 |
| Key pass | 1 |
| Big chance created | 3 |
| Successful open-play cross | 1 |
| Successful dribble | 1 |
| Foul won | 1 |
| Shot on target | 2 |
| Pass completion 70–79% (≥30 passes) | 2 |
| Pass completion 80–89% | 4 |
| Pass completion 90%+ | 6 |
| GKP/DEF conceding a goal | −4 each |
| Conceding a penalty | −3 |
| Penalty miss | −6 |
| Yellow card | −3 |
| Red card | −9 |
| Own goal | −6 |
| Big chance missed | −3 |
| Error leading to goal | −3 |
| Error leading to attempt | −1 |
| Conceding a foul | −1 |
| Caught offside | −1 |
| Shot off target | −1 |

Expected net effect: centre-backs lose bonus (Gabriel 30 → 20 in FFS's 2025-26 resimulation); GKs, dribblers and attackers gain. **The bonus model must be retrained/recalibrated on rule-adjusted BPS — naive historical bonus mappings are broken by these changes.**

### 1.3 Squad construction constraints (API `game_settings` / `element_types`)

- Squad size: **15** (`squad_squadsize: 15`).
- Budget: **£100.0m** = 1000 API units (all prices in £0.1m units; `ui_currency_multiplier: 10`).
- Position quotas (exact, not min/max): **2 GKP / 5 DEF / 5 MID / 3 FWD** (`squad_select`).
- Club limit: **≤3 players per PL club** (`squad_team_limit: 3`).
- Starting XI: 11 of 15 (`squad_squadplay: 11`); 4 bench slots in priority order; bench slot 1 = reserve GK always.
- Formation bounds (`squad_min_play`/`squad_max_play`): GKP exactly 1; DEF 3–5; MID 2–5; FWD 1–3. Valid formations: 3-4-3, 3-5-2, 4-3-3, 4-4-2, 4-5-1, 5-2-3, 5-3-2, 5-4-1.

### 1.4 Captaincy

- Captain ×2. If captain plays 0 minutes in the GW, vice-captain gets ×2 instead. If neither plays, no multiplier applied.
- Triple Captain chip: ×3 (transfers to vice on captain's non-appearance).
- Captain ≠ vice; both must be in the starting XI.

### 1.5 Automatic substitutions (processed at lockdown)

- A starter with **0 minutes** in the GW is replaced by the highest-priority bench player whose entry keeps the formation legal (GKP exactly 1, DEF ≥3, MID ≥2, FWD ≥1).
- Starting GK is only ever replaced by the bench GK.
- Players who played but scored ≤0 points are NOT substituted — only non-appearance triggers.
- Postponed-fixture players count as non-appearing once the GW completes.

### 1.6 Transfer state machine (2024-25 rules, confirmed retained)

- **Unlimited free transfers before the GW1 deadline.**
- +1 FT credited at each deadline; bank up to **5** total (`max_extra_free_transfers: 4` = 1 base + 4 extra).
- Formal: `FT_{t+1} = min(FT_t − used_t + 1, 5)`, floor 1 (cannot go below 1 going into a GW; extra transfers beyond FT are hits).
- Each extra transfer: **−4 points** from that GW's score.
- **Hard cap: 20 transfers per GW** (`transfers_cap: 20`); does not apply in WC/FH weeks. PRESUMED retained for 2026-27 (API-evidenced since ≥2023-24; no change announced) — re-verify value at launch.
- Playing WC/FH does **not** consume or reset banked FTs (bank passes through, +1 accrual continues).
- **No AFCON FT top-up in 2026-27** (AFCON is June/July 2027; the 2025-26 GW16 top-up-to-5 was a one-off).

### 1.7 Chips (structure identical to 2025-26; verified in API `chips[]`)

**8 chips: 2× each of Wildcard, Free Hit, Bench Boost, Triple Captain — one set per half-season. No Assistant Manager chip.**

| Chip (API name) | Set 1 window | Set 2 window | Type |
|---|---|---|---|
| Wildcard (`wildcard`) | GW2–GW19 | GW20–GW38 | transfer |
| Free Hit (`freehit`) | GW2–GW19 | GW20–GW38 | transfer |
| Bench Boost (`bboost`) | GW1–GW19 | GW20–GW38 | team |
| Triple Captain (`3xc`) | GW1–GW19 | GW20–GW38 | team |

- **Set-1 expiry: the GW19 deadline, 13:30 GMT, Saturday 2 January 2027.** Unused chips are lost, no carry-over.
- **≤1 chip active per GW** (official). FH unavailable GW1 (transfers unlimited pre-GW1 anyway).
- Effects: WC = unlimited transfers that GW, permanent; FH = unlimited transfers for one GW, squad reverts; BB = all 15 players score; TC = captain ×3. WC/FH irreversible once confirmed; neither wipes the FT bank.
- UNCERTAIN (minor): whether set-1 BB/TC again open at GW1 with WC/FH from GW2 exactly as 2025-26 — the official article only confirms "two sets, first until GW19"; the table above is the 2025-26 API pattern. Verify `chips[]` at launch.

### 1.8 Pricing mechanics

- 2026-27 starting prices: **not yet released** (reveals began 2026-07-22; full list at launch).
- **Prices locked until the GW1 deadline (18:30 BST, Fri 21 Aug 2026); in-season changes happen daily at 00:00 UK time** (official, new for 2026-27 — previously ~01:30–02:30 processing). Steps of ±£0.1m driven by net transfer activity; at most one change per player per day (community knowledge).
- Algorithm/thresholds: proprietary, **UNCERTAIN** (community consensus: ownership-scaled net-transfer thresholds, max ±£0.3m/GW, wildcarder moves capped, flag suppression). New official **Price Change Predictor** shows % progress toward a change (>100% = expected).
- **Selling price:** `transfers_sell_on_fee: 0.5`, `element_sell_at_purchase_price: false` → sell = purchase + ⌊(now − purchase)/2⌋ floored to £0.1m when in profit; full price fall borne otherwise. Encode: `sell(p) = purchase + max(0, floor((now − purchase)/2))` in 0.1m units, else `now` when now ≤ purchase.

### 1.9 Deadlines, lockdown, leagues

- **Deadlines: 90 minutes before each GW's first kickoff** (standard since 2020-21).
- **GW1 deadline: Friday 21 Aug 2026, 18:30 BST = 17:30 UTC** (officially stated in the Price Change Predictor article; re-verify `events[0].deadline_time` at launch).
- **NEW lockdown: GW scores final at 09:00 UK the day after the GW's last match.** Treat all GW points as provisional until then (late Opta corrections to bonus/DefCon now expected).
- H2H: win 3 / draw 1 / loss 0; tiebreak +goals_scored, −goals_conceded. Private classic leagues ≤30 joined. FPL Cup format TBD at launch.

### 1.10 The 11 position reclassifications for 2026-27

(https://www.premierleague.com/en/news/4679886/position-changes-for-202627-fantasy-premier-league)

| Player | Club | 2025-26 | 2026-27 |
|---|---|---|---|
| Myles Lewis-Skelly | Arsenal | DEF | MID |
| Lamare Bogarde | Aston Villa | DEF | MID |
| Junior Kroupi | Bournemouth | FWD | MID |
| Keane Lewis-Potter | Brentford | DEF | MID |
| Mats Wieffer | Brighton | MID | DEF |
| Georginio Rutter | Brighton | MID | FWD |
| Rio Cardines | Crystal Palace | MID | DEF |
| Ryan Sessegnon | Fulham | MID | DEF |
| Omar Marmoush | Man City | MID | FWD |
| Patrick Dorgu | Man Utd | DEF | MID |
| Eric Moreira | Nott'm Forest | MID | DEF |

Modeling note: historical per-90 features for these players were earned under a different scoring position — re-map their training rows or flag the position change.

### 1.11 Optimizer encoding checklist

1. Squad MILP: 15 players; 2/5/5/3; ≤3/club; Σ price ≤ 1000 units; XI = 11 with GK=1, DEF≥3, MID≥2, FWD≥1.
2. Transfers: `FT_{t+1} = min(FT_t − used + 1, 5)`; hits −4; ≤20/GW; WC/FH unlimited with FT pass-through.
3. Chips: 8 chips, ≤1/GW; set-1 expiry at GW19 deadline 2027-01-02 13:30 GMT; set-2 GW20–38.
4. Scoring: §1.1 exactly (GK goal 10; DefCon 10/12 thresholds; ⌊S/3⌋ saves; −⌊C/2⌋ conceded).
5. Bonus: retrain BPS→bonus projection for the §1.2 deltas.
6. Prices: 0.1m units; sell-price formula §1.8.
7. Data: re-diff everything at the `2026_27` API flip.

### 1.12 Uncertainty register (as of 2026-07-22)

| Item | Status |
|---|---|
| Exact launch date/time of 2026-27 game | UNKNOWN — poll daily |
| GW1 deadline 2026-08-21T17:30:00Z | Officially stated; pending API confirmation |
| Starting prices | Pending (reveals started 2026-07-22) |
| Set-1 BB/TC from GW1, WC/FH from GW2 | PRESUMED (2025-26 pattern) |
| 20-transfer cap value | PRESUMED 20 |
| GK goal 10 / assist v3 carry-over | PRESUMED (no change announced) |
| DefCon threshold "recalibration" rumor | REJECTED (uncorroborated) — assume unchanged |
| FPL Cup format | TBD at launch |
| Price-change algorithm internals | Proprietary, permanently UNCERTAIN |
| Canonical machine-readable BPS table | Does not exist; §1.2 is sourced reconstruction |

---

## Part 2 — Season Changelog for Training Data (2016-17 → 2025-26)

### 2.1 Per-season rule/anomaly table

| Season | Scoring changes | Chip changes | Transfer rules | Structural anomalies | New data fields |
|---|---|---|---|---|---|
| 2016-17 | ICT introduced (display only) | WC×2, BB, TC, **All Out Attack** (last season of AOA) | 1 FT/wk, max 1 banked (2 usable) | normal | rich per-match stats in API (key passes, big chances, CBI, tackles, …) |
| 2017-18 | **Assist definition v2** ("significant touch" doctrine) | **Free Hit replaces AOA** | unchanged | normal | unchanged |
| 2018-19 | none | none | unchanged | BGW31, DGW32 (10 teams), BGW33 (4 fixtures), DGW35 | last season of rich per-match stats |
| 2019-20 | none | none | unchanged | **COVID: 47 API events, 30–38 void, restart = events 39–47**; unlimited FTs at restart; BGW18→DGW24; GW26 split over 2 weekends; BGW28; restart behind closed doors; VAR introduced | **rich per-match stats dropped from API** |
| 2020-21 | none | none | deadline 60→**90 min** pre-kickoff | **whole season ~closed doors — away wins (40%) > home (38%), first ever**; GW1 only 8 fixtures; BGW18/DGW19; doubles GW24–26; BGW29 | (vaastav adds position/team/xP columns dataset-side) |
| 2021-22 | none | **extra FH granted mid-season from GW20** (Omicron); the two FHs not playable in consecutive GWs | unchanged | Omicron postponements → big spring doubles/blanks | none |
| 2022-23 | none | WC1 expired at GW16 deadline (12 Nov); WC2 from GW18 | **unlimited FTs during WC break (12 Nov–26 Dec)**; 20/GW cap documented | **Qatar WC break GW16→GW17 (6 weeks)**; prices frozen in break; **GW7 void (0 pts for all)** — Queen Elizabeth II; PL moves to 5 subs | **`starts` + xG family (`expected_*`) added** |
| 2023-24 | none | none | unchanged | BGW29 (4 fixtures), spring doubles | none |
| 2024-25 | **GK goal 6→10**; **BPS v2** (pen save 15→9; new −4 concede, +3 goal-line clearance, +1 foul won, +2 SoT) | **Assistant Manager chip** (GW24+, 3 consecutive GWs, manager £0.5–1.5m, W6/D3/goal1/CS2, underdog +10W/+5D) | **bank up to 5 FTs; chips no longer wipe bank** | managers = 5th element type in API | `mng_*` fields; manager elements (filter element_type 5) |
| 2025-26 | **DefCon points** (DEF 10 CBIT / MID+FWD 12 CBIRT, +2 cap); **BPS v3** (pen goal 12 all; saves in/out-box 3/2, pen save 8; goal-line clearance 9; tackles-won not net); **assist v3** (simplified; +41 assists retro-applied to 2024-25) | **two of each chip** (set 1 → GW19 deadline 30 Dec 2025, set 2 GW20–38); no AM chip | 5-FT banking; **AFCON top-up to 5 FTs at GW16** | AFCON absences Dec–Jan | **`defensive_contribution`, `clearances_blocks_interceptions`, `recoveries`, `tackles` added** |

2026-27 deltas on top of 2025-26 (from Part 1): BPS v4; live projected bonus; 09:00-next-day lockdown; no AFCON top-up; 11 position changes; everything else unchanged.

### 2.2 Feature flags for the training pipeline (season key = start year)

```yaml
assist_definition: {v1: [..2016], v2: [2017..2024], v3: [2025..]}
bps_version:       {v1: [..2023], v2: [2024], v3: [2025], v4: [2026..]}
gk_goal_points:    {6: [..2023], 10: [2024..]}
defensive_contribution_points: {absent: [..2024], present: [2025..]}  # DEF 10 CBIT / MID+FWD 12 CBIRT, +2 cap
max_free_transfers: {2: [..2023], 5: [2024..]}
chips_wipe_banked_fts: {true: [..2023], false: [2024..]}
deadline_minutes_before_kickoff: {60: [..2019], 90: [2020..]}
chip_inventory:
  2016: [WC, WC, BB, TC, AOA]
  2017-2020: [WC, WC, BB, TC, FH]
  2021: [WC, WC, BB, TC, FH, FH_extra_from_GW20]
  2022-2023: [WC, WC, BB, TC, FH]            # 2022: WC1 expires GW16, WC2 from GW18
  2024: [WC, WC, BB, TC, FH, AM_from_GW24]
  2025-2026: [2xWC, 2xFH, 2xBB, 2xTC]        # set1 expires GW19 deadline, set2 GW20+
free_transfer_amnesties:                      # exclude from transfer-cost modeling
  2019: unlimited_before_event_39             # COVID restart ("GW30+")
  2022: unlimited_between_GW16_and_GW17       # World Cup break
  2025: topped_to_5_at_GW16                   # AFCON
void_events:
  2019: [30, 31, 32, 33, 34, 35, 36, 37, 38]  # remap API events 39-47 -> "30+".."38+"
  2022: [7]                                   # rolled over, 0 pts for all
home_advantage_regime:
  empty_stadiums: {2019: events 39-47, 2020: all}   # away wins > home wins in 2020-21
mid_season_breaks:
  2019: GW26 split over two weekends (winter break)
  2022: 6 weeks between GW16 (12 Nov) and GW17 (26 Dec)
  2026: merged international break ~21 Sep - 6 Oct between GW5/GW6 (REPORTED, single source)
minutes_regime:
  three_subs: [..2021]
  five_subs: [2022..]
var_in_pl: [2019..]
detailed_defensive_stats_in_api: {full: [2016..2018], none: [2019..2024], cbit_recoveries: [2025..]}
xg_in_api: [2022..]                           # Understat needed for xG pre-2022
manager_elements_in_api: [2024]               # filter element_type 5 from 2024-25 pools
afcon_topup: [2025]                           # none in 2026
gw_lockdown_0900_next_day: [2026..]
```

### 2.3 API data availability matrix (verified from vaastav headers + live API)

| Fields | 16-17 | 17-18 | 18-19 | 19-20 | 20-21 | 21-22 | 22-23 | 23-24 | 24-25 | 25-26 |
|---|---|---|---|---|---|---|---|---|---|---|
| Core scoring stats + BPS + ICT | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| Rich per-match stats (key passes, big chances, tackles, CBI, …) | Y | Y | Y | — | — | — | — | — | — | CBI/recoveries/tackles only |
| `starts` | — | — | — | — | — | — | Y | Y | Y | Y |
| xG family (`expected_*`) | — | — | — | — | — | — | Y | Y | Y | Y |
| `defensive_contribution` | — | — | — | — | — | — | — | — | — | Y |
| Manager (MNG) elements | — | — | — | — | — | — | — | — | Y | — |
| Events in season | 38 | 38 | 38 | **47** (30–38 void) | 38 | 38 | 38 (GW7 void) | 38 | 38 | 38 |

### 2.4 Hard rules for the training pipeline

1. **Never train bonus/DefCon components on mixed-rule seasons naively** — reconstruct rule-adjusted targets from raw event counts per `bps_version`/`defensive_contribution_points` flags.
2. **2019-20:** join on remapped events (39→"30+" … 47→"38+"); events 30–38 are header-only stubs in vaastav.
3. **2022-23 GW7:** rows exist with 0 minutes for all players — exclude from minutes/appearance models.
4. **Home advantage:** regime flag on `is_home` for 2019-20 events 39–47 + all 2020-21.
5. **Minutes models:** 3-subs (≤2021) vs 5-subs (2022+) regime shifts cameo rates and starter minutes.
6. **xG pre-2022-23:** source from Understat (vaastav ships per-season `understat/` extracts with ID maps).
7. **Prices:** historical price series are regime-shifting (silent algorithm re-tunes); prefer observed `value` columns over simulated prices.
8. **Assists:** counts across v1/v2/v3 boundaries are not perfectly comparable (v3 would have added 41 assists to 2024-25).
9. Team IDs are per-season everywhere; `element.code`/`opta_code` are the only stable player keys.

---

## Part 3 — 2026-27 Season Context

### 3.1 The 20 clubs

**Promoted:** Coventry City (Championship champions, +11 pts, 97 goals — Frank Lampard), Ipswich Town (runners-up — Gary O'Neil, new), Hull City (play-off winners — Sergej Jakirović).
**Relegated out:** West Ham, Burnley, Wolves.

2025-26 final table (context for strength priors): 1 Arsenal 85 (champions), 2 Man City 78 (FA Cup + EFL Cup double, Guardiola's farewell), 3 Man Utd 71, 4 Aston Villa 65 (won Europa League), 5 Liverpool 60, 6 Bournemouth 57, 7 Sunderland 54, 8 Brighton 53, 9 Brentford 53, 10 Chelsea 52, 11 Fulham 52, 12 Newcastle 49, 13 Everton 49, 14 Leeds 47, 15 Crystal Palace 45, 16 Nott'm Forest 44, 17 Tottenham 41, 18 West Ham 39, 19 Burnley 22, 20 Wolves 20.
(https://en.wikipedia.org/wiki/2026%E2%80%9327_Premier_League)

**European contingent (midweek congestion / rotation risk):** UCL — Arsenal, Man City, Man Utd, Aston Villa, Liverpool. UEL — Bournemouth, Sunderland (first-ever/rare European Thursdays: classic UEL-hangover rotation risk). UECL — Brighton. Villa play PSG in the UEFA Super Cup 12 Aug 2026.

### 3.2 Schedule

- Season: **Fri 21 Aug 2026 → Sun 30 May 2027** (delayed a week by WC2026; final was 19 Jul 2026). 38 GWs, 380 fixtures; 33 weekend + 5 midweek rounds. Fixtures released 19 Jun 2026.
- **GW1:** Arsenal vs Coventry (Fri 21 Aug, 20:00 BST); Hull v Man Utd, Everton v Palace, Ipswich v Sunderland, Forest v Leeds, Brentford v Spurs (Sat); Brighton v Villa, Man City v Bournemouth, Newcastle v Liverpool (Sun); Fulham v Chelsea (Mon).
- **GW1 deadline: Fri 21 Aug 2026, 18:30 BST (17:30 UTC).**
- **Transfer window closes Tue 1 Sep 2026, 23:00 BST — after GW2**; late moves will disrupt early projections.
- REPORTED (single source): merged three-week international break 21 Sep – 6 Oct 2026 between GW5 and GW6. Nov/Mar breaks normal. GW numbering provisional until API is live.
- Boxing Day (Sat 26 Dec): full 10-match round. New player-welfare rule: no club plays twice within 60 hours over the festive period.
- Winter break: UNCERTAIN — none confirmed; compressed calendar suggests little slack.
- BGW/DGWs: unknown until cup draws — planner must treat spring blanks/doubles probabilistically (EFL Cup final ~late Mar 2027; FA Cup R5/QF/SF).
- **No AFCON clash** (June/July 2027) — no December exodus, no FT compensation.

### 3.3 World Cup 2026 fatigue watchlist (GW1–4 minutes-model dampening)

WC2026 final 19 Jul 2026: **Spain 1-0 Argentina aet** (Ferran Torres). England 3rd (6-4 vs France). Golden Ball: Rodri (MCI). ~40 PL players played until 14–19 July → only 2–3 weeks' pre-season; historical pattern: benched/early-hooked GW1–4, elevated injury hazard ~2 months. The one-week-later season start partially mitigates.

**Highest-load flags:** Rodri (MCI — started all 8, Golden Ball), Pedro Porro (TOT), Mikel Merino (ARS); Argentina core: E. Martínez (AVL), Romero (TOT), L. Martínez (MUN), Enzo Fernández (CHE), Mac Allister (LIV); England deep-runners: Rice, Saka, Eze, Madueke (ARS), Guéhi, Stones, O'Reilly, Trafford, Anderson (MCI), Pickford (EVE), Rashford, Mainoo (MUN), Watkins, Konsa (AVL), Burn (NEW), Chalobah, James (CHE), Spence (TOT), Henderson (CRY), J. Henderson (BRE), Rogers (AVL/CHE — see §3.4); France: Konaté (LIV), Digne (AVL), Gusto (CHE), Lacroix + Mateta (CRY), Cherki (MCI).
**Lighter Spain loads:** Zubimendi, Raya (ARS), Muñoz (LIV).
**Injury carry-overs:** Yeremy Pino (CRY, collarbone), Saliba (ARS, back).

### 3.4 Key summer 2026 transfers (as of 22 Jul; window open until 1 Sep)

CONFIRMED (≥2 sources): Elliot Anderson NFO→MCI £116m; Anthony Gordon NEW→Barcelona £69m; van Hecke BHA→TOT £52m; Andrey Santos CHE→MUN £48m; Cucurella CHE→Real Madrid £47.5m; Palestra Atalanta→CHE £47m; Vuskovic TOT→BHA £46m; Quenda Sporting→CHE £44m; Senesi BOU→TOT; Muñoz →LIV.
REPORTED (single source — verify): **Morgan Rogers AVL→CHE £117m**; Tonali NEW→TOT £92.5m; Mateus Fernandes WHU→TOT £85m; Jacquet Rennes→LIV £55m; Manzambi Freiburg→AVL £51m; Touré Hoffenheim→NEW £40m; Højlund MUN→Napoli £38m; Tielemans AVL→MUN £35m.

Already in 2025-26 data (Jan 2026, **split club stints before projecting**): Semenyo BOU→MCI (~£64m), Guéhi CRY→MCI (~£20m).
PL exits to model out: Gordon, Cucurella, Højlund (+ earlier: Toney, Quansah; Kane still at Bayern).

Narrative reads: Tottenham £280m+ rebuild under De Zerbi (high variance); Newcastle weakened (Tonali/Gordon out); Chelsea reload under Alonso; Man City post-Pep succession (Maresca) with Anderson/Semenyo/Guéhi.

### 3.5 Manager changes (10 clubs vs start of 2025-26 — style-uncertainty priors)

| Club | 2026-27 manager | From |
|---|---|---|
| Chelsea | Xabi Alonso | Real Madrid (Jul 2026) |
| Man City | Enzo Maresca | Chelsea (succeeds Guardiola) |
| Liverpool | Andoni Iraola | Bournemouth (Slot departed) |
| Nott'm Forest | Oliver Glasner | Crystal Palace |
| Tottenham | Roberto De Zerbi | Marseille (in-season, Mar 2026) |
| Crystal Palace | Pierre Sage | RC Lens |
| Bournemouth | Marco Rose | RB Leipzig |
| Fulham | Álvaro Arbeloa | ex-Real Madrid |
| Man Utd | Michael Carrick | Middlesbrough (in-season, Jan 2026; led Utd to 3rd) |
| Ipswich | Gary O'Neil | Strasbourg (McKenna stepped down) |

Continuity: Arteta (ARS), Emery (AVL), Andrews (BRE), Hürzeler (BHA), Moyes (EVE), Farke (LEE), Howe (NEW), Le Bris (SUN), Lampard (COV), Jakirović (HUL).

### 3.6 Promoted-team priors and assets

- **Coventry** (strongest prior): Lampard, possession + wide play; **Haji Wright** ~17-18 goals (premier promoted asset); five double-digit scorers.
- **Ipswich:** **Jack Clarke** 16 goals; Azor Matusiwa (POTY); manager change (McKenna→O'Neil) = continuity downgrade.
- **Hull** (weakest prior): pragmatic counter 4-2-3-1/5-ATB; **Oli McBurnie** 18 goals; Ryan Giles (elite crosser); Millar/Belloumi returning from ACLs (flag).
- Historical promoted prior is bimodal: 2023-24 and 2024-25 all three straight back down; 2022-23 all survived; 2025-26 massively over-performed (Sunderland 7th). Seed promoted-club team strength from Championship xG with a promotion discount (weight Coventry > Ipswich > Hull), not a flat relegation-fodder constant. Promoted DEF/GK usually £4.0–4.5m; premium promoted attackers £5.5–7.0m.

### 3.7 2025-26 reference points (pricing/projection priors)

Top FPL scorers 2025-26: Haaland (MCI) 239 @ £14.7m end price; Bruno Fernandes (MUN) 235 @ £10.4m; Gabriel (ARS) 209 @ £7.3m; Semenyo (BOU→MCI) 202; Gibbs-White (NFO) 188; Bowen (WHU, relegated) 187; Rice (ARS) 184; Thiago (BRE) 181; Elliot Anderson (NFO→MCI) 180 @ £5.7m; Guéhi 179; João Pedro (CHE) 177.
Down years: Salah 123 @ £14.0m (age 34, new manager); Palmer 114; Eze 113; **Isak (LIV) 41 pts — reason unconfirmed (presumed long injury; verify status/price at launch)**.
Price expectations: Haaland ~£14.0–15.0m; Bruno expected 2nd-most-expensive; Salah expected to fall.

### 3.8 Launch-week action items

1. Poll `bootstrap-static` daily until the `2026_27` flip; then ingest prices, re-diff API fields (live-bonus/price-predictor may add fields), rebuild the GW map.
2. Split 2025-26 per-player data by club stint (Semenyo, Guéhi minimum); re-map all summer movers; apply the 11 position changes.
3. Apply WC-fatigue dampening (§3.3) and new-manager style uncertainty (§3.5) to projections.
4. Re-pull transfer trackers weekly until the window closes 1 Sep.
5. Treat GW scores as provisional until 09:00 UK next-day lockdown.
