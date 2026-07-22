# FPL Rule Changes, Season by Season: 2016-17 → 2025-26

**Purpose:** normalize historical training data for the fpl-ai-assistant expected-points model, MILP planner and chip planner. Every season below lists scoring changes, chip changes, transfer-rule changes, structural anomalies (blanks/doubles/COVID/World Cup), and per-season API data availability, ending in a feature-flag table for the training pipeline.

**Research date:** 2026-07-22. As of this date, `https://fantasy.premierleague.com/api/bootstrap-static/` still serves the **2025-26** season (GW1 deadline `2025-08-15T17:30:00Z`); the 2026-27 game has not yet relaunched. All "current API" observations below are first-hand from that endpoint. Confidence labels: **CONFIRMED** (2+ sources or observed directly in API/data), **LIKELY** (single good source or strong community consensus), **UNCERTAIN**.

---

## 1. Quick changelog table

| Season | Scoring changes | Chip changes | Transfer rules | Structural anomalies | New API/data fields |
|---|---|---|---|---|---|
| **2016-17** | ICT Index introduced (display stat, not points) | Chips: 2×Wildcard, Bench Boost, Triple Captain, **All Out Attack** (last season of AOA) | 1 FT/wk, bank max 1 (2 usable) | Normal 38-GW season; usual cup-driven BGWs/DGWs | ICT (influence/creativity/threat); rich per-match stats in API |
| **2017-18** | **Assist definition rewritten** ("significant touch" rules; rebounds off keeper/woodwork/blocks; own-goal-forcing counts) | **Free Hit introduced, All Out Attack removed** | unchanged | Normal | unchanged |
| **2018-19** | none | none | unchanged | BGW31, DGW32 (10 teams), **BGW33 (only 4 fixtures)**, DGW35 | unchanged (last season of rich per-match stats) |
| **2019-20** | none | none | unchanged | **COVID**: paused 13 Mar 2020 after GW29; restart 17 Jun as "GW30+" = **API events 39-47** (47 events total); unlimited FTs at restart; BGW18 (Liverpool, Club World Cup) → DGW24; winter-break **GW26 split over 2 weekends**; BGW28 (4 clubs); restart behind closed doors. VAR introduced in PL. | **Detailed per-match stats dropped from API** (tackles, key passes, big chances etc. gone) |
| **2020-21** | none | none | unchanged; **deadline moved 60→90 min before kickoff** | **Whole season ~behind closed doors — home advantage vanished** (away wins 40% > home 38%, first time ever); compressed calendar (started 12 Sep); GW1 only 8 fixtures; BGW18, DGW19, doubles GW24-26, BGW29 | vaastav adds `position`/`team`/`xP` columns (dataset-side, not API) |
| **2021-22** | none | **Extra (2nd) Free Hit granted mid-season from GW20** (Omicron postponements); the two FHs couldn't be played in consecutive GWs | unchanged | Dec 2021 Omicron wave of postponements (~20+ matches) → big doubles/blanks in spring | none |
| **2022-23** | none | **WC1 expired at GW16 deadline (12 Nov)**; **unlimited free transfers during World Cup break** (12 Nov–26 Dec); WC2 available after GW17 deadline (26 Dec), i.e. from GW18 | 20-transfers-per-GW cap documented | **Qatar World Cup mid-season break** between GW16 (12-13 Nov) and GW17 (26 Dec); prices frozen during break; **GW7 postponed entirely** (death of Queen Elizabeth II) and "rolled over" with 0 points; PL moves to 5 substitutions | **`starts`, `expected_goals`, `expected_assists`, `expected_goal_involvements`, `expected_goals_conceded` added** |
| **2023-24** | none | none | unchanged | BGW29 (only 4 fixtures), spring doubles; normal 38 events | none |
| **2024-25** | **GK goal 6→10 pts; BPS overhaul v2** (pen save 15→9 BPS; new: −4 conceding, +3 goal-line clearance, +1 foul won, +2 shot on target) | **Mystery → Assistant Manager chip** (playable GW24 on, 3 consecutive GWs, manager £0.5-1.5m from budget, W6/D3/goal 1/CS 2, underdog table bonus +10 W/+5 D) | **Bank up to 5 FTs (was 2 usable/1 banked); chips no longer wipe banked FTs** | Managers added as a 5th element type in API (for AM chip) | `mng_*` scoring fields; manager elements |
| **2025-26** | **Defensive contribution points**: DEF +2 at 10 CBIT; MID/FWD +2 at 12 CBIRT (cap +2); **BPS v3** (pen goal 12 for all; saves in-box 3/out-box 2/pen save 8; goal-line clearance 3→9; tackles won not net); **assist definition simplified** (single defensive touch in box OK; handball intent removed) | **Two of every chip**: set 1 usable GW1/2-19 (expired at GW19 deadline 30 Dec 2025), set 2 GW20-38; no AM chip | 5-FT banking retained; **AFCON top-up: everyone topped to 5 FTs at GW16** | AFCON absences Dec-Jan | **`defensive_contribution`, `clearances_blocks_interceptions`, `recoveries`, `tackles` added to API** |

---

## 2. Constants across the whole window (2016-17 → 2025-26)

All **CONFIRMED** via current `game_settings`/`game_config` and rule archives; none of these changed in the window:

- **Budget £100.0m** (`squad_total_spend: 1000`), squad of 15: 2 GKP / 5 DEF / 5 MID / 3 FWD (`element_types` squad_select), max **3 per club** (`squad_team_limit: 3`).
- Formation constraints: 1 GKP, ≥3 DEF, ≥2 MID (min-play numbers per current API), ≥1 FWD, 11 starters.
- **Sell-on fee 50%** of profit, rounded down to nearest £0.1m (`transfers_sell_on_fee: 0.5`, `element_sell_at_purchase_price: false`).
- Base scoring (unchanged except where noted in §3): appearance 1 (<60 min) / 2 (60+); goals GKP/DEF 6, MID 5, FWD 4 (GKP → 10 from 2024-25); assist 3; clean sheet GKP/DEF 4, MID 1; 1 pt per 3 saves; penalty save +5; penalty miss −2; −1 per 2 goals conceded (GKP/DEF); YC −1; RC −3; OG −2; bonus 3/2/1 from BPS. Source: current `game_config.scoring` + [PL scoring explainer](https://www.premierleague.com/en/news/2174909).
- −4 points transfer hit per extra transfer; 1 free transfer added per GW.
- Price changes ±£0.1m steps driven by net transfers; algorithm **unofficial/undocumented** throughout (see §4 Price mechanics).

---

## 3. Season-by-season detail

### 2016-17 — baseline season

- **Chips (5):** Wildcard ×2, Bench Boost, Triple Captain, **All Out Attack**. AOA let you field a **2-5-3 formation for one GW** (one fewer defender). AOA was introduced summer 2015 and **removed summer 2017** — so 2016-17 is its last season. CONFIRMED: [FFS retrospective "All Out Attack, Ultimate FPL…"](https://www.fantasyfootballscout.co.uk/2025/07/07/all-out-attack-ultimate-fpl-man-in-the-stand-long-lost-bits-of-fpl), [JOE.co.uk on its replacement](https://www.joe.co.uk/sport/all-out-attack-has-been-replaced-on-fantasy-football-with-an-absolute-doozy-of-a-feature-133319).
- **ICT Index introduced in the 2016-17 game** (Influence/Creativity/Threat; display metric only, no points). CONFIRMED: [PL ICT explainer](https://www.premierleague.com/en/news/65567), [fantasyfootballreports](https://www.fantasyfootballreports.com/fpl-ict-index-explained/); `ict_index` present in 2016-17 data.
- Historical context pre-window (for completeness): chips (BB/TC/AOA) date from 2015-16; before that the second wildcard was a "**Winter Wildcard**" usable only in the January window (2010-2015) ([FFS retrospective](https://www.fantasyfootballscout.co.uk/2025/07/07/all-out-attack-ultimate-fpl-man-in-the-stand-long-lost-bits-of-fpl)).
- **Transfers:** 1 FT/week, max 1 banked (2 usable) — the regime that held until 2024-25.
- **Deadline:** 60 minutes before first kickoff (held until 2020-21 change).
- **Data:** FPL API of this era exposed **rich per-match stats**: `attempted_passes, big_chances_created, big_chances_missed, clearances_blocks_interceptions, completed_passes, dribbles, errors_leading_to_goal, fouls, key_passes, offside, open_play_crosses, penalties_conceded, recoveries, tackled, tackles, target_missed, winning_goals, ea_index…` — CONFIRMED first-hand from [vaastav dataset 2016-17 headers](https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/2016-17/gws/gw1.csv). Machine-readable history starts here: **vaastav/Fantasy-Premier-League covers 2016-17 → 2025-26**.

### 2017-18 — Free Hit; assist definition rewritten

- **Free Hit chip introduced, replacing All Out Attack**: unlimited transfers for one GW, squad reverts afterwards. CONFIRMED: [FFS Free Hit guide](https://www.fantasyfootballscout.co.uk/2024/03/14/what-is-the-fpl-free-hit-chip-and-when-should-it-be-used/) ("Free Hit arrived for 2017/18"), [fantasyfootball247 launch piece](https://fantasyfootball247.co.uk/fpl-launch-201718-rules-changes-player-price-list/), [PL explainer](https://www.premierleague.com/en/news/816951). Chip set from 2017-18 to 2020-21: WC×2, FH, BB, TC.
- **Assist definition v2** (in force 2017-18 → 2024-25): shots **saved/blocked/hitting the woodwork** that rebound to a scoring teammate (or force an own goal) earn the shooter an assist; winning a penalty/free-kick that is scored earns the fouled player an assist; **opponent touches only void an assist if they "significantly alter" the ball's trajectory/destination** (the "significant touch" doctrine that produced a decade of Dubious Points Panel controversies). CONFIRMED: the PL's own Oct-2017 explainer series on the then-new rules — [Final pass](https://www.premierleague.com/en/news/825262), [Rebounds](https://www.premierleague.com/en/news/827841), [Foul play](https://www.premierleague.com/en/news/827842).
- **Training note:** assist counts before/after 2017-18 are not perfectly comparable (slightly more generous from 2017-18).

### 2018-19 — no rule changes

- No changes to scoring, chips or transfers (season launch coverage lists only price/position reclassifications). CONFIRMED (absence of change): [PL 2018-19 launch](https://www.premierleague.com/en/news/775534), [Goal 2018-19 rules guide](https://www.goal.com/en-us/news/goal-premier-league-fantasy-football-how-to-play-prizes-and-new-rules-for-the-2018-19-season/7yz06ivv81hw13zt199a5ek9n).
- **Structural:** the most extreme cup-driven blank/double pattern of the pre-COVID era — BGW31, **DGW32 (10 teams)**, **BGW33 with only 4 fixtures**, DGW35 ([Goal DGW schedule 2018-19](https://www.goal.com/en/news/fantasy-football-double-gameweeks-premier-league-2018-19-match-schedule-full/chsh1x1nrx3v1ix5eetshha8e), [FFS DGW history](https://www.fantasyfootballscout.co.uk/2026/04/04/the-good-the-bad-and-the-duffy-a-history-of-fpl-double-gameweeks)).
- **Data:** last season with the rich per-match stat block in the API (see 2016-17). Also circa summer 2018 the API base path moved from `/drf/` to `/api/` (community-documented; LIKELY).

### 2019-20 — COVID season: 47 events, restart anomalies

- **No scoring/chip/transfer rule changes at launch** ([PL 2019-20 launch](https://www.premierleague.com/en/news/1245426)).
- **VAR introduced in the Premier League** — affects goal/assist/penalty distributions in the underlying data (context, not an FPL rule).
- **Structural anomalies (all CONFIRMED):**
  - **BGW18:** West Ham v Liverpool postponed for Liverpool's FIFA Club World Cup → both blank GW18, rescheduled into **DGW24** ([FFS](https://www.fantasyfootballscout.co.uk/2019/09/04/the-first-blank-gameweek-of-2019-20-is-confirmed/), [FFS Jan 2020](https://www.fantasyfootballscout.co.uk/2020/01/16/when-the-blank-gameweek-and-double-gameweek-fixtures-could-take-place/)).
  - **Winter break: GW26 split across two weekends** (~10 days, some teams playing a week after the deadline) — one FPL event spanning two football weekends ([FFS](https://www.fantasyfootballscout.co.uk/2020/01/16/when-the-blank-gameweek-and-double-gameweek-fixtures-could-take-place/), [Goal explainer](https://www.goal.com/en-gb/news/why-fpl-gameweeks-18-19-split-premier-league-fantasy-football-break-explained/q2xnt2e957471ep3la5iwmvvi)).
  - **BGW28** for Man City, Arsenal, Aston Villa, Sheffield United (Carabao Cup final); the two postponed matches (City v Arsenal, Villa v Sheff Utd) became the first restart fixtures.
  - **COVID suspension:** league paused 13 March 2020 after GW29 completed; **restarted 17 June 2020** behind closed doors. The remaining rounds were branded "**GW30+ … GW38+**" but were created as **new API events 39-47** — the original events 30-38 exist but are empty. **CONFIRMED first-hand:** vaastav 2019-20 has `gw1..gw47.csv`; `gw30-38.csv` are 339-byte header-only stubs, real data resumes in `gw39-47.csv`. **So 2019-20 has 47 `events`, with 30-38 void — any per-GW join must remap 39-47 → "30+"-"38+".**
  - **Restart concessions:** all managers received **unlimited free transfers ahead of GW30+** (no wildcard needed); unused chips stayed available; GW30+ was a double for Arsenal, Villa, City, Sheff Utd ([PL restart guide](https://www.premierleague.com/news/1678559), [NMA chip strategies](https://www.nevermanagealone.com/2020/6/11/21288765/fpl-chip-strategies-for-project-restart)). Prices were frozen during the suspension (LIKELY; community-documented).
  - **All 92 restart matches behind closed doors** — home-advantage regime break starts here, mid-season.
- **Data availability break (CONFIRMED first-hand):** from 2019-20 the API's per-match stat set is slimmed to the scoring stats only (`minutes, goals, assists, CS, conceded, OG, pens, cards, saves, bonus, bps, ICT`) — **tackles, key passes, big chances, recoveries, CBI etc. disappear** until 2025-26 partially restores the defensive trio. Any feature built on those stats must switch source (Opta/Understat/FBref) for 2019-20 → 2024-25.

### 2020-21 — the empty-stadium season; 90-minute deadline

- **Rule change:** GW deadline moved from **60 to 90 minutes before the first kickoff** (anti-team-leak measure). CONFIRMED: [FFS relaunch article](https://www.fantasyfootballscout.co.uk/2020/08/15/fpl-relaunches-for-2020-21-as-gameweek-deadlines-move-back-half-an-hour/). No scoring/chip/transfer changes.
- **Structural anomalies:**
  - Compressed calendar: started 12 Sep 2020, ended 23 May 2021; **GW1 had only 8 fixtures** (Man Utd, Burnley, Man City, Aston Villa started late, fixtures redistributed).
  - COVID postponements produced **BGW18 (six fixtures) / DGW19**, a run of doubles GW24-GW27, and a **four-fixture BGW29** ([fantasyfootballgeek](https://www.fantasyfootballgeek.co.uk/fpl-double-gameweeks-bgw18-dgw19/), [Fantasy Football Hub 20/21 guide](https://www.fantasyfootballhub.co.uk/fpl-double-gameweek-guide-tips-20-21)).
  - **Home advantage vanished: away win rate 40% vs home 38% — the first season in English top-flight history with more away than home wins**; home wins <40% for the first time on record; effect attributed to empty stadiums. CONFIRMED: [PL season trends](https://www.premierleague.com/news/2165807), [Significance magazine analysis](https://significancemagazine.com/home-advantage-whats-changed-since-covid/) (home/away win ratio 0.94 in 2020-21), [Sky Sports](https://www.skysports.com/football/news/11095/13511444/home-advantage-is-on-the-wane-in-the-premier-league-between-the-lines). ~2,000 fans were allowed briefly in Dec 2020 in some grounds and ~10,000 in the final rounds of May 2021 (LIKELY).
  - **Training note:** any `is_home` feature needs a regime flag covering **2019-20 events 39-47 + all of 2020-21**.

### 2021-22 — mid-season extra Free Hit

- **No launch changes.** Mid-season: because of the December Omicron postponement wave, **all managers were given an extra Free Hit chip, usable from GW20** (announced January 2022); a manager holding two FHs could not play them in consecutive GWs; the second FH remained usable "from Gameweek 22" onwards if the first was played in GW20. CONFIRMED: [PL announcement](https://www.premierleague.com/en/news/2425494). → 2021-22 is the only season in the window with **6 chips** without a mid-season game reset.
- **Structural:** ~20+ matches postponed Dec 2021-Jan 2022 (Omicron), producing large spring doubles (e.g. DGW22 with 8 teams, DGW26, DGW33, DGW36/37) and blanks (e.g. BGW30) — derive exact team-level patterns from the fixtures data rather than trusting lists (individual DGWs confirmed piecemeal: [fantasyfootballgeek](https://www.fantasyfootballgeek.co.uk/fpl-blank-and-double-gameweek-guide-29126/), [FFS DGW history](https://www.fantasyfootballscout.co.uk/2026/04/04/the-good-the-bad-and-the-duffy-a-history-of-fpl-double-gameweeks)).

### 2022-23 — Qatar World Cup season; xG lands in the API

- **Chips/transfers rearranged around the World Cup (all CONFIRMED, [PL "What's new in 2022/23"](https://www.premierleague.com/en/news/2667633), [PL WC-break guide](https://www.premierleague.com/en/news/2890870), [sportbible](https://www.sportbible.com/football/fantasy-premier-league-wildcard-deadline-gw16-fpl-20221107)):**
  - **First wildcard usable only until the GW16 deadline (Sat 12 Nov 2022, 13:30)** — much earlier than normal; if unused, lost.
  - **Unlimited free transfers for everyone between the GW16 deadline and the GW17 deadline (Mon 26 Dec, 13:30)** — a de-facto free wildcard over the 6-week break. Wildcard itself not playable during the break; BB/TC/FH technically playable.
  - **Second wildcard available after the GW17 deadline ("ahead of Gameweek 18")** through season end.
  - **Player prices frozen 13 Nov – 26 Dec** ([FFS FAQ](https://www.fantasyfootballscout.co.uk/2022/11/13/fpl-faq-will-price-rises-happen-during-the-world-cup/), [fantasyfootballcommunity](https://fantasyfootballcommunity.com/fpl-price-changes-during-the-world-cup/)).
  - A **20-transfers-per-GW cap** (excl. WC/FH) is documented this season and still exists (`transfers_cap: 20`); introduction date UNCERTAIN (possibly earlier).
- **Structural anomalies:**
  - **GW7 postponed in full** (death of Queen Elizabeth II, 8 Sep 2022). FPL **rolled GW7 over with 0 points for everyone**: the deadline stood, saved FTs and transfer hits stood (hits kept their −4), but activated TC/BB/FH chips were **reinstated**; the postponed fixtures scored in the GWs they were eventually played in. CONFIRMED: [FFS](https://www.fantasyfootballscout.co.uk/2022/09/09/gameweek-7-fixtures-postponed/), [PL](https://www.premierleague.com/en/news/2786896). → 2022-23 effectively has **37 scoring events out of 38**; GW7 rows exist with zero minutes for all players.
  - A few extra postponements around the funeral/rail strikes (GW8) plus the usual cup blanks: BGW28, DGW29, BGW32, DGW34 etc. (derive from fixtures).
  - **Mid-season 6-week break** GW16 → GW17 (12 Nov – 26 Dec): form/minutes features must not treat this as a normal 1-week gap; players returned with World Cup fatigue or 6 weeks' rust.
  - **Real-football change: 5 substitutions allowed from 2022-23** — structurally lowers minutes-per-appearance for starters and raises cameo counts (affects minutes models across the boundary).
- **Data (CONFIRMED first-hand from vaastav 2022-23 headers):** API adds **`starts`**, **`expected_goals`, `expected_assists`, `expected_goal_involvements`, `expected_goals_conceded`** (Opta xG family) — first season of official xG in FPL data.

### 2023-24 — quiet season

- **No material changes** to scoring, chips or transfers; launch coverage lists only position reclassifications (11 players) and app/UI features. CONFIRMED by absence in launch coverage ([Dexerto position changes](https://www.dexerto.com/gaming/all-fpl-position-changes-for-2023-24-season-jota-gakpo-more-2202112/), [FFS launch dates](https://www.fantasyfootballscout.co.uk/2023/05/31/when-will-fpl-launch-for-2023-24-key-dates-for-fantasy-managers)) and by the 2024-25 announcement billing the 5-FT change as "the biggest alteration since chips" ([FourFourTwo](https://www.fourfourtwo.com/news/fantasy-premier-league-will-launch-a-new-format-for-next-season)).
- **Structural:** standard 38 events; largest blank (BGW29) had only four fixtures; spring doubles (DGW34, DGW37) — derive from fixtures.

### 2024-25 — five free transfers; BPS v2; Assistant Manager chip

All CONFIRMED: [PL "BIG changes announced"](https://www.premierleague.com/en/news/4058895), [FFS "Significant changes announced"](https://www.fantasyfootballscout.co.uk/2024/07/16/fpl-2024-25-significant-changes-announced/), [PL points-scoring changes](https://www.premierleague.com/en/news/4059044).

- **Transfers: bank up to 5 free transfers** (previously 2 usable / 1 banked — a limit that had held since the early 2010s). Current API: `max_extra_free_transfers: 4` (= 1 weekly + 4 banked). **Chips no longer wipe banked FTs**: play a WC/FH with 4 banked and you still have them afterwards (pre-2024-25, playing WC/FH reset you to 1 FT).
- **Scoring: goalkeeper goals 6 → 10 points** (all other goal values unchanged: DEF 6, MID 5, FWD 4).
- **BPS v2:** penalty save 15 → 9 BPS; **new:** −4 BPS for GK/DEF conceding a goal, +3 goal-line clearance, +1 foul won, +2 shot on target. (Winners/losers analysis: [FFS](https://www.fantasyfootballscout.co.uk/2025/03/14/the-winners-and-losers-from-the-fpl-2024-25-bonus-points-changes).) → **bonus points before/after 2024-25 are not comparable** for keepers and shot-heavy attackers.
- **Assistant Manager chip** (the pre-season "Mystery Chip", revealed 24 Jan 2025): [PL reveal](https://www.premierleague.com/en/news/4193484), [PL full guide](https://www.premierleague.com/en/news/4192707), [FFS](https://www.fantasyfootballscout.co.uk/2025/01/24/fpl-mystery-chip-revealed-what-is-it-when-can-it-be-played):
  - Playable **from GW24** (Feb 2025) at any point over the rest of the season; once activated it runs for **three consecutive gameweeks**.
  - You buy one of the 20 real head coaches for a **fixed £0.5m-£1.5m out of your existing squad budget** (money returns when the chip ends); the manager **counts against the 3-per-club limit**; swapping manager mid-chip costs a transfer; cannot be captained.
  - Manager scoring: **win +6, draw +3, +1 per team goal, +2 clean sheet**; **table bonus** if the opponent started the GW ≥5 places higher: **+10 extra for a win (16 total), +5 for a draw (8 total)**.
  - One-chip-per-GW rule applied while active.
  - **API side-effect:** managers were added as a 5th element type ("MNG") with their own elements and `mng_*` scoring keys — those keys still exist (zeroed) in the 2025-26 `game_config.scoring` (observed first-hand). Filter element_type 5 out of 2024-25 player pools.
- **Structural:** normal season; cup-driven **DGW33 / BGW34** ([PL](https://www.premierleague.com/en/news/4624106/which-clubs-have-a-confirmed-dgw33-and-bgw34-after-fa-cup-ties)).

### 2025-26 — defensive contributions; double chips; assist v3; BPS v3

All CONFIRMED: [PL "All you need to know about changes for 2025/26"](https://www.premierleague.com/en/news/4362211/all-you-need-to-know-about-changes-to-fantasy-for-202526), [PL defensive-contribution explainer](https://www.premierleague.com/en/news/4361991/whats-new-in-202526-fantasy-defensive-contributions), [PL BPS changes](https://www.premierleague.com/en/news/4362127/whats-new-in-202526-fantasy-changes-to-bonus-points-system), [PL assist changes](https://www.premierleague.com/en/news/4362187/whats-new-in-202526-fantasy-changes-to-assists-rules), [FFS BPS explainer](https://www.fantasyfootballscout.co.uk/2025/07/19/fpl-2025-26-all-the-bonus-points-changes-explained), plus the live 2025-26 API.

- **Defensive contribution points (new scoring category, "DefCon"):**
  - **Defenders: +2 points for 10+ combined clearances, blocks, interceptions, tackles (CBIT) in a match.**
  - **Midfielders & forwards: +2 points for 12+ combined CBIT + ball recoveries (CBIRT).**
  - Goalkeepers: not eligible (`defensive_contribution: {GKP: 0, DEF: 2, MID: 2, FWD: 2}` in `game_config.scoring`, observed).
  - **Capped at +2 per player per match** regardless of volume; contributions also feed BPS.
- **BPS v3:** penalty goals 12 BPS for all positions (was 24 FWD / 18 MID / 12 DEF); GK saves split — 3 BPS in-box, 2 BPS out-of-box, penalty save 8 (+3 for the save itself = 11); goal-line clearance 3 → 9; tackle BPS switches from net (won−lost) to **2 BPS per tackle won**.
- **Assist definition v3 ("simplified"):** intended target/destination of a pass no longer matters — an assist stands if the scorer receives it in the box with **at most one defensive touch** in between; the "forced handball" intent test removed (any attacking action leading directly to a scored penalty/FK from a handball = assist). Retro-applied to 2024-25 it would have added **41 assists** (~small but non-zero label shift).
- **Chips: two of each** (2×WC, 2×FH, 2×BB, 2×TC = 8):
  - **Set 1:** BB/TC usable GW1-19, WC/FH GW2-19; **expired unused at the GW19 deadline (18:30 GMT, Tue 30 Dec 2025)**.
  - **Set 2:** all usable GW20-38.
  - Exactly as in the live API `chips` array (observed: `stop_event: 19` / `start_event: 20` pairs). **No Assistant Manager chip.**
- **Transfers:** 5-FT banking retained; **AFCON accommodation: every manager topped up to the maximum 5 FTs at GW16** (Dec 2025) instead of extra chips.
- **Data (observed first-hand):** API `element_stats` and vaastav 2025-26 headers add **`clearances_blocks_interceptions`, `recoveries`, `tackles`, `defensive_contribution`** — the first partial return of defensive per-match stats since 2018-19.

---

## 4. Cross-cutting mechanics notes

### Price-change mechanics
- Never officially documented in the whole window ([PL's own explainer](https://www.premierleague.com/en/news/2858775) confirms only ±£0.1m steps from transfer activity). Community reverse-engineering ([LiveFPL](https://livefpl.com/blog/fpl-price-changes), [Fantasy Football Hub](https://www.fantasyfootballhub.co.uk/fpl-price-changes), [fplform](https://fplform.com/fpl-price-change)) agrees on: net-transfer thresholds scaled by ownership, overnight processing (~02:30 UK), max ±£0.3m/GW, wildcarders' moves counted (capped per manager/day), flag-related suppression. Thresholds have been silently re-tuned several times (notably ~2019-20 "flattening") — **UNCERTAIN exact dates; treat historical price series as regime-shifting and prefer observed `value` columns over simulated prices.**
- Price freezes: during the 2019-20 COVID suspension (LIKELY) and the 2022-23 World Cup break, 13 Nov – 26 Dec (CONFIRMED, [FFS](https://www.fantasyfootballscout.co.uk/2022/11/13/fpl-faq-will-price-rises-happen-during-the-world-cup/)).

### Free-transfer banking timeline
- ≤2023-24: 1 FT/week, max 1 banked (2 usable). Playing WC/FH wiped the bank (back to 1).
- 2024-25 onward: max 5 usable (`max_extra_free_transfers: 4`); bank survives chip play.
- One-off top-ups/amnesties: 2019-20 GW30+ unlimited; 2022-23 WC break unlimited; 2025-26 GW16 top-up to 5.

### Deadlines
- 60 min before first kickoff through 2019-20 → **90 min from 2020-21** ([FFS](https://www.fantasyfootballscout.co.uk/2020/08/15/fpl-relaunches-for-2020-21-as-gameweek-deadlines-move-back-half-an-hour/)). Verified against live API (2025-26 GW1: deadline 17:30 UTC vs 19:00 UTC kickoff).

### Chip windows (wildcard split)
- WC1/WC2 split sits at late December in normal seasons; exact boundary varies by season and is **best read from each season's archived `chips`/`events` API data**. Confirmed anchors: 2022-23 split at GW16/GW18 (World Cup); 2024-25 WC1 until GW19 deadline, WC2 from GW20; 2025-26 whole chip sets split GW19/GW20.

### API data availability summary (verified from vaastav headers + live API)

| Fields | 16-17 | 17-18 | 18-19 | 19-20 | 20-21 | 21-22 | 22-23 | 23-24 | 24-25 | 25-26 |
|---|---|---|---|---|---|---|---|---|---|---|
| Core scoring stats + BPS + ICT | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Rich per-match stats (key passes, big chances, tackles, recoveries, CBI, fouls…) | ✓ | ✓ | ✓ | — | — | — | — | — | — | CBI/recoveries/tackles only |
| `starts` | — | — | — | — | — | — | ✓ | ✓ | ✓ | ✓ |
| xG family (`expected_*`) | — | — | — | — | — | — | ✓ | ✓ | ✓ | ✓ |
| `defensive_contribution` | — | — | — | — | — | — | — | — | — | ✓ |
| Manager (MNG) elements | — | — | — | — | — | — | — | — | ✓ | — |
| Events in season | 38 | 38 | 38 | **47** (30-38 void) | 38 | 38 | 38 (GW7 void) | 38 | 38 | 38 |

---

## 5. Suggested feature flags for the training pipeline

```yaml
# season keys use start year: 2016 == 2016-17
assist_definition: {v1: [..2016], v2: [2017..2024], v3: [2025..]}
bps_version:       {v1: [..2023], v2: [2024], v3: [2025], v4: [2026..]}  # v4 announced for 2026-27
gk_goal_points:    {6: [..2023], 10: [2024..]}
defensive_contribution_points: {absent: [..2024], present: [2025..]}   # DEF 10 CBIT / MID+FWD 12 CBIRT, +2 cap
max_free_transfers: {2: [..2023], 5: [2024..]}
chips_wipe_banked_fts: {true: [..2023], false: [2024..]}
deadline_minutes_before_kickoff: {60: [..2019], 90: [2020..]}
chip_inventory:
  2016: [WC, WC, BB, TC, AOA]
  2017-2020: [WC, WC, BB, TC, FH]
  2021: [WC, WC, BB, TC, FH, FH_extra_from_GW20]
  2022-2023: [WC, WC, BB, TC, FH]           # 2022: WC1 expires GW16, WC2 from GW18
  2024: [WC, WC, BB, TC, FH, AM_from_GW24]
  2025: [2xWC, 2xFH, 2xBB, 2xTC]            # set1 expires GW19 deadline, set2 GW20+
free_transfer_amnesties:                     # exclude these GWs from transfer-cost modeling
  2019: unlimited_before_event_39            # "GW30+"
  2022: unlimited_between_GW16_and_GW17      # World Cup break
  2025: topped_to_5_at_GW16                  # AFCON
void_events:
  2019: [30, 31, 32, 33, 34, 35, 36, 37, 38] # renumber 39-47 -> "30+".."38+"
  2022: [7]                                  # rolled over, 0 pts for all
home_advantage_regime:
  empty_stadiums: [2019: events 39-47, 2020: all]   # away wins > home wins in 2020-21
mid_season_breaks:
  2019: GW26 split over 2 weekends (winter break)
  2022: 6 weeks between GW16 (12 Nov) and GW17 (26 Dec)  # World Cup
minutes_regime:
  three_subs: [..2021]
  five_subs: [2022..]                         # plus the 2019-20 restart mini-regime
var_in_pl: [2019..]
detailed_defensive_stats_in_api: {full: [2016..2018], none: [2019..2024], cbit_recoveries: [2025..]}
xg_in_api: [2022..]
manager_elements_in_api: [2024]              # filter element_type 5
```

---

## 6. 2026-27 preview (announced as of 2026-07-22)

For the season this assistant targets — announced but the game is not yet live ([PL "changes for 2026/27"](https://www.premierleague.com/en/news/4679873/all-you-need-to-know-about-changes-to-fpl-for-202627), [FFS "5 rule changes"](https://www.fantasyfootballscout.co.uk/2026/07/20/fpl-2026-27-5-rule-changes-new-features-announced), [PL 2026-27 BPS changes](https://www.premierleague.com/en/news/4679946/whats-new-in-202627-fantasy-changes-to-bonus-points-system)):

- **BPS v4** (de-duplicating vs. DefCon): no more −1 for being tackled; CBI earn 1 BPS per **3** (was per 2); GK saves restructured (in-box 3, out-of-box save category removed, other saves 2, big-chance save +1).
- Projected bonus shown live from 20 minutes into each fixture; live rank/mini-league updates.
- GW "lockdown" (points finalization) moves to 09:00 UK the day after the final match.
- **No AFCON free-transfer top-up** (AFCON is June/July 2027).
- Defensive contribution rules reportedly carry over; watch for a possible threshold tweak announcement at launch (UNCERTAIN until the API is live).
- Season history pages add percentage ranks (cosmetic).

---

## 7. Open questions / unconfirmed items

1. Exact FPL chip-window boundaries (WC1/WC2 split GW) for 2016-17 → 2021-22 and 2023-24 — read from archived season API data or wayback captures rather than news.
2. When the 20-transfers-per-GW cap was introduced (documented 2022-23; may predate).
3. Exact date banking a free transfer was first allowed (pre-window, early 2010s).
4. Whether prices were formally frozen during the 2019-20 suspension (strong community consensus, no surviving official statement found).
5. Price-algorithm re-tunings: dates/magnitudes are community folklore (2019-20 "flattening" etc.) — unverifiable officially.
6. Assistant Manager chip: latest GW it could be activated in 2024-25 (GW36 to fit 3 GWs, or truncated if later) — not confirmed.
7. Whether any 2017-18 BPS tweaks accompanied the assist rewrite (none found; treat BPS as v1 constant 2016-17 → 2023-24).
8. Exact fan-attendance numbers/dates for late-2020 partial reopenings (minor for modeling).
9. 2026-27: whether DefCon thresholds change, chip arrangement (single or double sets), and launch date — not yet announced as of 2026-07-22.
