# Data Sources Audit — fpl-ai-assistant

**Date of audit: 2026-07-22.** Every endpoint marked *tested* below was hit live with `curl` during this audit (HTTP status codes reported are from those tests). Facts are labeled **CONFIRMED** (verified directly or via ≥2 sources) or **UNCERTAIN**.

**Season context (CONFIRMED):** As of 2026-07-22 the official FPL API still serves the **completed 2025-26 season** (GW38 `finished: true`, deadline 2026-05-24). The **2026-27 game has not launched yet**. GW1 of 2026-27 kicks off ~Friday 21/22 August 2026, and launch is expected imminently — the 2026/27 rule-change announcement went out 20 July 2026 ([premierleague.com](https://www.premierleague.com/en/news/4679873/all-you-need-to-know-about-changes-to-fpl-for-202627), [fantasyfootballscout.co.uk](https://www.fantasyfootballscout.co.uk/2026/07/20/when-will-fpl-go-live-for-2026-27)). **When the game relaunches, all element IDs reset**; the `code` and `opta_code` fields are the stable cross-season player identifiers.

---

## 1. Official FPL API (primary source)

Base URL: `https://fantasy.premierleague.com/api/`. No API key. No official documentation or published rate limits. Send a browser-like `User-Agent` header (default python-requests UA has historically been blocked intermittently). All endpoints below were **tested live 2026-07-22**.

### 1.1 Endpoint inventory (all tested)

| Endpoint | Auth? | Tested result | What it returns |
|---|---|---|---|
| `bootstrap-static/` | No | 200, ~2.0 MB | The master blob: `events` (38 GWs w/ deadlines, finished flags), `teams` (20, with strength ratings), `elements` (841 players in 2025-26), `element_types`, `element_stats`, `chips`, `phases`, `game_settings`, `game_config`, `total_players` (13,107,732 for 2025-26) |
| `fixtures/` | No | 200, ~974 KB | All 380 fixtures: `event`, `kickoff_time`, `team_h/a`, `team_h_difficulty`/`team_a_difficulty` (FDR 1–5), per-fixture `stats` (goals, assists, bonus, bps, **defensive_contribution**, …) |
| `fixtures/?event={gw}` | No | 200 | Same, filtered to one GW |
| `element-summary/{player_id}/` | No | 200, ~32 KB | Per-player: `fixtures` (remaining), `history` (per-GW rows this season, 40 columns incl. xG/xA/xGC, `defensive_contribution`, `tackles`, `recoveries`, `clearances_blocks_interceptions`, `transfers_in/out`, `value`, `selected`), `history_past` (season-total aggregates for prior seasons) |
| `event/{gw}/live/` | No | 200, ~590 KB | Every player's stats for that GW + `explain` breakdown (per-fixture points by identifier, incl. `points_modification` field) |
| `event-status/` | No | 200 | Bonus-added / league-update status for current GW day(s) |
| `entry/{id}/` | No | 200 | Public manager profile: team name, overall points/rank, leagues joined, favourite club |
| `entry/{id}/history/` | No | 200 | `current` (per-GW points, rank, bank, team value, transfers), `past` (all prior seasons' totals), **`chips` (chips played + which GW)** |
| `entry/{id}/event/{gw}/picks/` | **No** | **200** | **Full 15-man picks** incl. captain/vice, multipliers, `active_chip`, `automatic_subs`, `entry_history` (points, bank, value, transfers cost) |
| `entry/{id}/transfers/` | **No** | **200** | Full season transfer log (element in/out, prices, timestamp) — tested entry 5000: 113 transfers returned |
| `my-team/{id}/` | **Yes** | **403** | Current team *before* deadline, chip availability, saved transfers — requires login cookies (`pl_profile` session) |
| `entry/{id}/transfers-latest/` | Yes | 403 | Latest (pre-deadline) transfers — auth only |
| `leagues-classic/{id}/standings/` | No | 200 | Classic league standings, paginated (`?page_standings=N`); league 314 = overall |
| `leagues-h2h/{id}/standings/` | No | 404 (invalid id) | H2H standings for valid H2H league ids |
| `dream-team/{gw}/` | No | 200 | GW dream team |
| `team/set-piece-notes/` | No | 200 | Curated penalty/set-piece taker notes per team with `last_updated` |
| `stats/most-valuable-teams/` | No | 200 | Leaderboard extras |

### 1.2 What you can see about a user's team WITHOUT logging in (CONFIRMED by direct test)

- **Public (no auth): picks for any completed/locked GW** (`entry/{id}/event/{gw}/picks/` returned 200 unauthenticated), full GW-by-GW history, chips played, all past transfers, profile, leagues. During a live season, the current GW's picks become publicly visible **only after the GW deadline** (long-standing behavior, corroborated by every community API guide, e.g. [Frenzel Timothy's endpoint guide](https://medium.com/@frenzelts/fantasy-premier-league-api-endpoints-a-detailed-guide-acbd5598eb19)); before the deadline they are only reachable via the authenticated `my-team/{id}/`.
- **Auth-only: `my-team/{id}/`** (403 without login) — the *current* pre-deadline squad, saved-transfer count, and chip availability. Auth is cookie-based (login via `users/login` on `account.premierleague.com` has become harder to automate since ~2024 due to a new login flow; **UNCERTAIN** whether headless login is currently reliable — plan for the user to either paste session cookies or tell the assistant their team manually).
- **New for 2026-27 (CONFIRMED, official):** "Gameweek lockdown" extends until **09:00 UK the day after the final match of a GW** ([premierleague.com](https://www.premierleague.com/en/news/4679873/all-you-need-to-know-about-changes-to-fpl-for-202627)) — expect rival-picks/EO data to be unavailable slightly longer than the old "1 hour after final whistle."

### 1.3 Rate-limit etiquette (no official policy)

No documented limits; the API is Cloudflare-fronted and will 429/403 on abusive bursts. Community practice that works: **≤1 req/s sustained, exponential backoff on 429, cache aggressively**. A full nightly `element-summary` sweep is ~850 requests ≈ 15 min at 1 req/s. Never poll `bootstrap-static` more than ~once a minute even on deadline day; during matches poll `event/{gw}/live/` once every 60–90 s.

### 1.4 2025-26/2026-27 scoring fields that matter for the model (CONFIRMED)

- `elements` and per-GW histories now carry **`defensive_contribution`** (+ inputs `clearances_blocks_interceptions`, `recoveries`, `tackles`) — the DC scoring introduced in 2025-26 (DEF: 10+ CBIT → 2 pts; MID/FWD: 12+ CBIRT → 2 pts) continues in 2026-27. The FPL API itself is therefore a sufficient source for DC modeling targets.
- Also present: `expected_goals`, `expected_assists`, `expected_goal_involvements`, `expected_goals_conceded` (Opta xG, per GW and season), `chance_of_playing_this/next_round`, `news`, `news_added`, `status` (a/d/i/s/u), `birth_date`, `opta_code`, `price_change_percent`, `scout_news_link`/`scout_risks` (Fantasy Football Scout integration fields).
- **2026-27 announced changes** ([premierleague.com](https://www.premierleague.com/en/news/4679873/all-you-need-to-know-about-changes-to-fpl-for-202627), [FFS](https://www.fantasyfootballscout.co.uk/2026/07/20/fpl-2026-27-5-rule-changes-new-features-announced)): live-updating ranks/mini-leagues with projected bonus after 20', **BPS tweaks reducing overlap with DC points** (boosts GK/full-back/attacker bonus prospects; being tackled no longer penalized), official Price Change Predictor (price changes at 00:00 UK), two full chip sets (first set expires at GW19 deadline 13:30 GMT Sat 2 Jan 2027), 5-FT rollover cap unchanged. **Exact BPS coefficient changes not yet published** (UNCERTAIN until launch — re-scrape `bootstrap-static.game_settings`/`scoring` at launch).

---

## 2. Historical bulk data (model training, 2016-17 → 2025-26)

### 2.1 vaastav/Fantasy-Premier-League — still the canonical archive (CONFIRMED alive)

[github.com/vaastav/Fantasy-Premier-League](https://github.com/vaastav/Fantasy-Premier-League) — checked via GitHub API 2026-07-22:

- Last commit **2026-07-20** ("Update 2026 world cup fantasy data"); "**Final 2025-26 update**" committed 2026-06-17. Actively maintained.
- `data/` has complete seasons **2016-17 through 2025-26**, plus `cleaned_merged_seasons.csv` (all seasons concatenated) and `master_team_list.csv`.
- Per season: `gws/gw1..38.csv` + **`gws/merged_gw.csv`** (per-player-per-GW), `players_raw.csv` (bootstrap snapshot), `cleaned_players.csv`, `player_idlist.csv`, `fixtures.csv`, `teams.csv`, per-player folders with full `gw.csv`/`history.csv`.
- **Verified**: `data/2025-26/gws/merged_gw.csv` header includes `xP, expected_assists, expected_goal_involvements, expected_goals, expected_goals_conceded, clearances_blocks_interceptions, defensive_contribution, recoveries, tackles` — so the 2025-26 file already carries the DC columns. `data/2016-17/gws/merged_gw.csv` also confirmed present (HTTP 200). Note: xG columns only exist from 2022-23 onward (FPL added them then); Understat is the xG source for earlier seasons.
- Caveats: historical seasons have known encoding quirks (2018-19 merged_gw latin-1), position column naming drifts across seasons, and team ids are per-season. The repo also ships Understat extracts per season (`data/{season}/understat/`).

**Risk**: the repo is one volunteer's side project; each season's scraper needs a manual kickoff after game launch. Watch the repo in Aug 2026 for a `data/2026-27/` directory.

### 2.2 Alternatives / successors

- **olbauday/FPL-Core-Insights** (recently **renamed from FPL-Elo-Insights** — old name 404s, rename confirmed by the repo's own discussion thread): [github.com/olbauday/FPL-Core-Insights](https://github.com/olbauday/FPL-Core-Insights). Pushed **2026-07-02**. Covers **2024-2025 and 2025-2026** with `players.csv`, `playerstats.csv`, `teams.csv`, `gameweek_summaries.csv`, and per-GW / per-tournament folders (`player_gameweek_stats.csv`, match stats) including **FA Cup, League Cup, European competitions and friendlies, all keyed to FPL player ids, plus ClubElo team ratings**. Best modern enrichment source; too short a history to replace vaastav for training.
- **fplform.com** — free **CSV export of its predicted points (xFPL) and probability-of-playing per fixture** ([fplform.com/export-fpl-form-data](https://fplform.com/export-fpl-form-data)); useful as a *benchmark* to validate our own xP model, not as ground truth.
- **FPLYogi/FPL-Data**, **Ayanab01/FPL_Stats** — smaller mirrors found via GitHub topic search; unvetted, use only as tertiary fallbacks.
- **ffdataviz** — **UNCERTAIN**: could not verify an active site/archive under this name in 2026; do not depend on it.

**Verdict**: vaastav `merged_gw.csv` (2016-17→2025-26) is the primary training corpus; FPL-Core-Insights for Elo + cup/rotation context; our own nightly API snapshots become the primary source going forward (never depend on a volunteer repo mid-season).

---

## 3. Understat — xG/xA per player per match (CONFIRMED alive, free; **site re-architected Dec 2025**)

[understat.com](https://understat.com) is alive and fully updated (league page title: "EPL xG Table and Scorers for the **2025/2026** season"). EPL coverage from season 2014 onward. **Critical change discovered in this audit**: around **7–9 Dec 2025** (per JS asset timestamps) Understat stopped embedding `playersData/teamsData/datesData = JSON.parse(...)` blobs in page HTML (league pages are now ~19 KB shells). Any scraper that regexes embedded JSON is now broken. Data now comes from **plain JSON endpoints** (all tested 2026-07-22, no auth, gzip — use `curl --compressed`, send a browser UA; `X-Requested-With: XMLHttpRequest` header recommended):

- `GET https://understat.com/getLeagueData/EPL/2025` → 200. Returns `{dates, players, teams}`: full season player table (games, minutes, goals, xG, npg, npxG, assists, xA, shots, key_passes, xGChain, xGBuildup), team objects with **38-match per-game history** (team xG/xGA/ppda etc.), and the fixture list. Season `2025` = 2025-26.
- `GET https://understat.com/getPlayerData/{player_id}/` (trailing slash required; without it → 404) → 200. Returns `{player, groups (per-season splits), matches (entire career per-match: goals, shots, xG, xA, npxG, key_passes, time, roster_id), shots (every career shot w/ x,y,xG,situation,shotType), minMaxPlayerStats, positionsList}`. Haaland (id 8260): 199 EPL/Bundesliga matches returned in one call.
- `POST https://understat.com/main/getPlayersStats/` (form: `league=EPL&season=2025`) → 200 (players table; the league.js also references `main/getPlayerMatches/`, `main/getPlayersName/`).
- Match pages (`/match/{id}`) load via analogous endpoints; per-match rosters give per-player per-match xG/xA.

**Python packages**: [amosbastian/understat](https://github.com/amosbastian/understat) (async, 184★, last pushed **2025-12-16** — i.e. right after the site change, likely a compatibility fix) and [collinb9/understatAPI](https://github.com/collinb9/understatAPI) (last pushed **2026-02-21**). Both alive, but given the Dec 2025 re-architecture, **pin and integration-test whichever you adopt, or just call the two JSON endpoints above directly — they are simpler than any wrapper** (UNCERTAIN: current pip releases of these packages may or may not handle the new endpoints).

Etiquette: no published limits; it's an ad-funded hobby site — keep to ~1 req/2s, cache hard (one `getLeagueData` call per day covers the whole league). Name-matching Understat→FPL ids requires a manual mapping table (vaastav's repo ships `understat/` id mappings per season; FPL-Core-Insights also aligns ids).

---

## 4. FBref — deeper Opta stats (defensive contributions, progressive passes)

**What it adds over Understat**: Opta (StatsPerform) event data — tackles, interceptions, blocks, clearances, ball recoveries (the exact inputs to FPL's DC points), progressive passes/carries, SCA/GCA, touches by third, aerials, GK advanced (PSxG±), plus npxG per shot. Understat has none of the defensive/possession detail.

**Scraping viability in 2026 (tested + corroborated): poor for plain HTTP.**
- Direct `curl` of `https://fbref.com/en/comps/9/Premier-League-Stats` with a full browser UA → **403 Cloudflare "Just a moment…" challenge** (tested 2026-07-22). Same for `sports-reference.com/bot-traffic.html`.
- Official policy ([sports-reference.com/bot-traffic.html](https://www.sports-reference.com/bot-traffic.html), as reported by third parties since the page itself now sits behind the challenge): **hard cap ~10 requests/min** (≈1 req/6s, tightened in 2024 from the older 1 req/3s), violators blocked for ~24h. Community reports of worldfootballR/plain-requests 403s throughout late 2025 ([Posit forum, Nov 2025](https://forum.posit.co/t/http-403-error-worldfootballr-scraping/209090)).
- Workarounds that still function: **cloudscraper/curl-cffi TLS-fingerprint clients at ≤10 req/min** (the `soccerdata` python package wraps FBref this way), real-browser automation (Playwright) using FBref's per-table "Share & Export → Get table as CSV" trick, or paid scraping proxies. The underlying Opta feed is otherwise only available via commercial StatsPerform licensing — not realistic.

**Recommendation**: **deprioritize FBref for v1**. Since 2025-26 the FPL API itself publishes per-GW `tackles`, `recoveries`, `clearances_blocks_interceptions`, `defensive_contribution` — the DC signal no longer requires FBref. Add FBref later (via `soccerdata`, one slow weekly sweep) only for enrichment features (progressive passes, SCA, PSxG for keepers).

---

## 5. Bookmaker odds

### 5.1 the-odds-api.com (forward-looking odds; CONFIRMED specifics)

- **Free tier: 500 credits/month**, no card ([the-odds-api.com](https://the-odds-api.com/)). **Cost per call = #markets × #regions** (historical endpoints cost 10×). E.g. EPL `h2h,totals` for region `uk` = 2 credits → a daily pull all month ≈ 60 credits. Corroborated by [oddspapi.io's 2026 comparison](https://oddspapi.io/blog/the-odds-api-free-tier-limits/).
- Featured soccer markets: `h2h` (includes draw), `totals`, `spreads`, `btts`, `draw_no_bet` ([betting-markets docs](https://the-odds-api.com/sports-odds-data/betting-markets.html)).
- **Player props incl. `player_goal_scorer_anytime`** (also first/last scorer, shots on target, assists, cards): available for the EPL **but only from US bookmakers**, and only via the per-event endpoint `/events/{eventId}/odds` — i.e. ~10 events × (markets×regions) credits per GW sweep. Budget: one anytime-scorer sweep per GW on deadline morning ≈ 10–20 credits. Realistic free-tier plan: daily `h2h+totals` league snapshot + one deadline-day player-props sweep ≈ **~150 credits/month — comfortably inside 500**.
- Clean-sheet odds are not a standard market; derive CS probability from `totals`/`btts` (team-conceding models) or Poisson from h2h+totals.

### 5.2 football-data.co.uk (historical odds for backtesting; CONFIRMED)

- Free CSVs per season, EPL (`E0`) verified from **1993-94** (HTTP 200) through **2025-26** (380 rows, tested): match stats (shots, SoT, corners, cards) + odds from Bet365, Betfair, Pinnacle (PSH/PSD/PSA), William Hill etc., **Max/Avg market odds, over/under 2.5, Asian handicap, and closing odds** (column glossary in [notes.txt](https://www.football-data.co.uk/notes.txt)). URL pattern: `https://www.football-data.co.uk/mmz4281/{yy}{yy}/E0.csv`. Updated ~twice weekly in season. This is the backbone for calibrating an odds→goals/CS model without burning API credits.

### 5.3 OddsPortal

JS-rendered, aggressive anti-bot; community scrapers break every few months. **Not viable as a dependency** — skip (use only ad-hoc via browser automation if a one-off historical props sample is ever needed).

---

## 6. Injury & availability news

1. **FPL API itself (primary)**: `status` (a=available, d=doubtful, i=injured, s=suspended, u=unavailable), `news` (free text, e.g. "Knee injury - Expected back 15 Mar"), `news_added` (timestamp), `chance_of_playing_this_round`/`next_round` (0/25/50/75/100). Updated by FPL's team typically **within hours of official club news/press conferences** — reliable but sometimes a day behind Twitter/X beat reporters (no formal SLA; treat `news_added` as the freshness signal). Free, structured, zero extra work — this drives availability in the model.
2. **premierleague.com/en/latest-player-injuries** — official club-by-club injury page, **tested: plain curl 200** (82 KB HTML, scrapeable). Good secondary confirmation.
3. **premierinjuries.com** (Ben Dinnery) — the classic injury-table source, but **tested: now behind Cloudflare (curl 403 "Just a moment")**. Scraping requires cloudscraper/browser automation; treat as manual-check enrichment, not pipeline.
4. **Ben Dinnery / @PremierInjuries on X** — fastest human source, but X API access is paid and scraping X is impractical in 2026. Out of pipeline scope.
5. **Fantasy Football Scout** — free: news articles, team-news previews (RSS/HTML scrapeable); **premium (£2.99/mo billed annually, [pricing](https://www.fantasyfootballscout.co.uk/pricing))**: projections, Opta tables, predicted lineups. Note the FPL API's `scout_news_link`/`scout_risks` fields already surface FFS flags. Premium data is behind login and its ToS won't permit re-scraping — use as a human benchmark only.
6. **Rotowire soccer lineups** (rotowire.com/soccer/lineups.php) — free predicted XIs, historically scrapeable; **UNCERTAIN** current anti-bot posture (untested here). Predicted-lineup probability is the biggest gap left by the sources above; fallback is modeling minutes from FPL `starts` history + `chance_of_playing`.

---

## 7. Fixture difficulty: FPL FDR vs our own Elo

- **FPL FDR** (`team_h_difficulty`/`team_a_difficulty`, 1–5, in `fixtures/`): opaque in-house rating, coarse (mostly 2–5), updated rarely. Fine for UI display; too crude for xP modeling.
- **ClubElo (build our own — CONFIRMED free API)**: [api.clubelo.com](http://api.clubelo.com) plain-CSV API, tested 2026-07-22: `GET api.clubelo.com/{ClubName}` → full Elo history since 1946 (e.g. ManCity, 343 KB); `GET api.clubelo.com/{YYYY-MM-DD}` → snapshot of all clubs (tested 2026-07-20: Arsenal #1 at 2063.8). No key, no documented limit (be polite: it's one request per day per need; ratings update after each match day). Feed Elo diff + home advantage into expected team goals, or fit our own Elo/Dixon-Coles on football-data.co.uk history and use ClubElo as prior/validation. FPL-Core-Insights already bundles ClubElo ratings keyed to fixtures if we want it pre-joined.
- Odds-implied difficulty (from §5) is the strongest single fixture-strength signal when available; Elo covers fixtures further out than bookmaker markets.

---

## 8. Recommended ingestion plan

### Primary sources (pipeline-critical)
| Source | Use | Cadence | Notes |
|---|---|---|---|
| FPL API `bootstrap-static` | prices, ownership, status/news, season scoring rules | 2–4×/day (hourly near deadline); snapshot to disk daily (build our own 2026-27 archive from day 1) | ~2 MB; diff for price/status changes |
| FPL API `fixtures` | schedule, FDR, DGW/BGW detection | daily | |
| FPL API `element-summary/{id}` | per-GW player history + upcoming fixtures | nightly full sweep after match days (~850 req @ 1 req/s ≈ 15 min) | |
| FPL API `event/{gw}/live` | live points, post-GW ground truth | during matches: every 60–90 s; else once post-GW | |
| FPL API `entry/*` (public) | user's team after deadline, rivals, EO in mini-league | on demand | pre-deadline squad needs auth (`my-team`) or user input; note new 09:00-next-day lockdown |
| vaastav merged_gw.csv 2016-17→2025-26 | training corpus | one-time backfill + verify at season boundaries | xG cols only from 2022-23; join Understat for earlier |
| Understat `getLeagueData/EPL/{year}`, `getPlayerData/{id}/` | player/team xG, npxG, shot data, xGChain | league call nightly after matches; player calls weekly or on demand | new Dec-2025 JSON endpoints; `--compressed`; ~1 req/2s |
| ClubElo API | team strength for fixture difficulty | weekly + after each match round | free CSV; also fit own Dixon-Coles on football-data history |
| football-data.co.uk E0.csv | historical odds + match stats for backtests, odds-model calibration | one-time backfill (1993→) + weekly in season | |

### Enrichment (nice-to-have, degrade gracefully)
- **the-odds-api.com free tier**: daily `h2h+totals` (uk region) snapshot; one `player_goal_scorer_anytime` per-event sweep (US region) on deadline morning. Budget ≈150/500 credits/mo.
- **FPL-Core-Insights**: cup/European fixture congestion + pre-joined Elo; weekly git pull.
- **premierleague.com injury page**: daily scrape as second injury signal.
- **fplform.com CSV export**: weekly, as external xP benchmark for model eval.
- **FBref via `soccerdata`** (cloudscraper, ≤10 req/min): deferred to v2 for progressive-pass/PSxG features.

### Fallbacks
- vaastav stale at 2026-27 rollover → our own daily API snapshots are the archive; FPL-Core-Insights as secondary.
- Understat endpoint changes again → understatAPI/understat packages (both maintained into 2026); worst case Playwright.
- Odds credits exhausted → football-data.co.uk closing odds (post-hoc) + Elo-derived probabilities.
- Injury feeds blocked → FPL `status`/`news` fields alone are sufficient for a functional model.

### Identity/joining keys
- FPL `element.id` (per-season) ↔ `element.code` / `opta_code` (cross-season) — build the crosswalk on day 1 of 2026-27.
- Understat player id ↔ FPL: fuzzy name+team match seeded from vaastav's historical `understat/` id maps.
- Team names: maintain one canonical team table mapping FPL `team.code` ↔ Understat ↔ ClubElo ↔ football-data.co.uk names.

---

## 9. Open questions (UNCERTAIN as of 2026-07-22)

1. Exact 2026-27 launch day and GW1 deadline; player prices; any API schema changes (new BPS coefficients, live-rank endpoints). Re-audit `bootstrap-static` at launch.
2. Exact new BPS coefficients ("reduced DC overlap", "tackled no longer penalized") — details unpublished.
3. Whether automated login for `my-team/` still works with the account.premierleague.com flow.
4. Whether current pip releases of `understat`/`understatAPI` support Understat's Dec-2025 endpoints.
5. Rotowire predicted-lineups scrapeability; any free machine-readable predicted-XI source.
6. FPL API formal rate limits (none published; 1 req/s etiquette assumed).
7. ffdataviz status — could not verify; likely defunct or misremembered name.
