# GW1 Consensus Research — FPL 2026/27 (launch week)

**Research date:** 2026-07-23 (day 2 of the 2026/27 game; GW1 deadline 2026-08-21 17:30 UTC, confirmed from the live API).
**Purpose:** Compare our model's GW1 recommendation (squad, captain, chip cascade) against what other models, tools and elite FPL thinkers are recommending, and against day-2 ownership data from `bootstrap-static`.

**Evidence labels used below:**
- **CONFIRMED** — sourced from a fetched page or the live FPL API, cited inline.
- **REPORTED** — appeared in search snippets / summarised fetches from one source; plausible but not independently double-checked.
- **ANECDOTAL/STALE** — community comment level, or from a page with internally inconsistent (pre-launch/last-season) data.

---

## 1. The landscape 24–48h after launch

The game went live 22–23 July 2026 ("Fantasy Premier League (FPL) 2026/27 is live!", [FFScout, 23 Jul](https://www.fantasyfootballscout.co.uk/2026/07/23/fantasy-premier-league-fpl-2026-27-is-live); "FPL is live: Pick your 2026/27 squad NOW", [premierleague.com](https://www.premierleague.com/en/news/4680722/fpl-is-live-pick-your-202627-squad-now)). Content is thin and mostly price-reaction: full first-draft reveals from Fantasy Football Scout's Scout Squad are announced but paywalled/"coming soon" ([FFScout live post](https://www.fantasyfootballscout.co.uk/2026/07/23/fantasy-premier-league-fpl-2026-27-is-live)); Fantasy Football Hub's "Ultimate Guide to Gameweek 1" (by Oli Poole, dated 23 Jul 2026) is members-only ([FFH](https://www.fantasyfootballhub.co.uk/fantasy-premier-league-ultimate-guide-fpl-tips)); r/FantasyPL launch threads were not surfaced by search indexing yet and reddit's JSON endpoints blocked scraping. Solver-house content (FPL Review, planfpl, fplform) for 2026/27 has not surfaced publicly at all yet — the closest is Fantasy Football Fix's pre-launch price/draft tooling ([Fix price predictions](https://www.fantasyfootballfix.com/blog-index/fpl-2026-27-player-price-predictions/)).

What *is* abundant: launch-day editorial from premierleague.com (The Scout) and FFScout, one full independent blogger draft (fpl.page), one launch-day "hidden gems" listicle (SportBible), the 2025/26 chip-season retrospectives (directly comparable — first two-set season), and the live ownership numbers themselves.

**Confirmed 2025/26 final-scores context** (cross-checked across FFScout price-reveal coverage, premierleague.com and Onside's season review):
- Haaland (MCI) 239 pts, 27 goals, Golden Boot — [FFScout price reveals](https://www.fantasyfootballscout.co.uk/2026/07/22/fpl-2026-27-price-reveals-live-haaland-rises-to-record-high), [Onside season review](https://onsidearena.com/season/2025-26)
- Bruno Fernandes (MUN) 235 pts, 9 goals, 24 assists (PL assists record) — [FFScout](https://www.fantasyfootballscout.co.uk/2026/07/22/fpl-2026-27-price-reveals-live-haaland-rises-to-record-high), [Fix](https://www.fantasyfootballfix.com/blog-index/fpl-2026-27-player-price-predictions/)
- Gabriel (ARS) 209 pts — top defender, "third-best all-time for defenders" — [premierleague.com Scout's 15](https://www.premierleague.com/en/news/4680821/the-scouts-analysis-of-15-key-player-prices-in-202627-fantasy)
- **Igor Thiago (BRE) 181 pts, 22 league goals (9 pens), 41 big chances (19 more than anyone at BRE)** — [premierleague.com Brentford guide](https://www.premierleague.com/en/news/4676527/best-teams-to-invest-in-for-202627-fantasy-brentford), cross-checked with [Wikipedia/LiveScore player pages](https://en.wikipedia.org/wiki/Igor_Thiago)
- João Pedro (CHE) 177 pts — price **unchanged** at £7.5m, flagged by FFScout as "a notable anomaly" given Chelsea have no European football — [FFScout 9 first impressions](https://www.fantasyfootballscout.co.uk/2026/07/23/fpl-2026-27-9-first-impressions-of-the-player-prices)
- Salah is **not in the 2026/27 game** (left the PL after a down 2025/26); his exit is repeatedly cited as the reason Haaland was priced to a record and Bruno to £12.0m — [FFScout](https://www.fantasyfootballscout.co.uk/2026/07/22/fpl-2026-27-price-reveals-live-haaland-rises-to-record-high)

REPORTED managerial context that colours picks this summer (each seen in one fetched source, not independently verified): no Pep at Man City ([expert price-prediction panel](https://www.premierleague.com/en/news/4672877/fpl-experts-price-predictions-for-202627)); Michael Carrick at Man Utd (same panel + [Scout's 15](https://www.premierleague.com/en/news/4680821/the-scouts-analysis-of-15-key-player-prices-in-202627-fantasy)); Xabi Alonso at Chelsea (Scout's 15 + SportBible snippet); Oliver Glasner at Forest ([FFScout pre-season notes](https://www.fantasyfootballscout.co.uk/2026/07/20/fpl-pre-season-glasners-tactics-osula-pen-andrey-debut-sesko-injury)); Keith Andrews took Brentford to 9th in his first season ([PL.com Brentford guide](https://www.premierleague.com/en/news/4676527/best-teams-to-invest-in-for-202627-fantasy-brentford)).

---

## 2. The Haaland question at £15.5m

**Consensus: overwhelming "start with him." Our no-Haaland-until-GW6 line is a genuine outlier, and we found no prominent public defence of it.**

- CONFIRMED: Haaland is the most expensive player in FPL history at £15.5m and is **66.0% selected on day 2** (live API) — nearly 1.5x the next player (João Pedro 44.6%). Owning him leaves £84.5m for 14 players ([premierleague.com record-price piece](https://www.premierleague.com/en/news/4680490)).
- CONFIRMED: The Scout's explicit argument: going without Haaland "would be a very risky tactic" because he has been **the top-scoring player after the first six Gameweeks in all four of his Man City seasons**, averaging ~9 goals over GW1–6, and City host promoted Coventry (GW3) and Ipswich (GW7) — [Why Fernandes and Haaland look like MUST-HAVES](https://www.premierleague.com/en/news/4675553/why-fernandes-and-haaland-look-like-must-haves-to-start-202627-fpl).
- CONFIRMED: Fantasy Football Fix's expert panel treated him as essential pre-launch: "He needs to be at least £15.0m unless some other superstar comes to the Premier League" — [Fix price predictions](https://www.fantasyfootballfix.com/blog-index/fpl-2026-27-player-price-predictions/).
- CONFIRMED: Every concrete draft we could recover starts him (fpl.page draft, the FFScout-associated early draft, Onside's cheat sheet — see §4).
- ANECDOTAL: FFScout's price-reveal comments show a live debate about "essentiality at £15.5m", with the sceptic case resting on the managerial change at City — but the summarised community verdict was "the man is still essential at £15.5m". The only structural counter-argument found in any launch content is Oscar (fpl.page) *noting* the price partly reflects "the lack of other premiums" (only Haaland is above £9.0m among forwards, [FFScout 9 impressions](https://www.fantasyfootballscout.co.uk/2026/07/23/fpl-2026-27-9-first-impressions-of-the-player-prices)) — and he still picked him.
- Directly relevant prior-season evidence (CONFIRMED, 2025/26 comparable): AllAboutFPL's initial-drafts format listed as a standing con of Haaland-less builds "the difficulty in getting back Haaland if he goes on to have a brilliant start" — [AllAboutFPL 25/26 drafts](https://allaboutfpl.com/2025/07/initial-fpl-gw1-drafts-for-the-25-26-fpl-season-with-pros-cons/).

**Effective-ownership math (our own, from API data):** at 66% selected and a likely majority captaincy share, Haaland's GW1 effective ownership will plausibly exceed 100%. A Haaland brace+ in GW1 (home v Bournemouth) costs a non-owner roughly 15–25 net points vs the field in one week. Our model's +11.1 xP BB GW1 edge is of the same order as a single bad Haaland variance draw.

**Timing critique of our own plan:** our plan buys him for GW6 — which the fixture grid (live API) shows is **MCI away at Liverpool**, City's hardest fixture of the opening 8. The community's TC/entry windows are GW3 (COV h) and GW7 (IPS h) ([PL.com chips article](https://www.premierleague.com/en/news/4679879/whats-happening-with-fpl-chips-in-202627)). If we hold the no-Haaland line at all, entering before GW5 (SUN h) or GW7 (IPS h) dominates a GW6 entry.

---

## 3. Captaincy: the consensus and where Thiago sits

**Consensus GW1 captain: Haaland (MCI h BOU), with Bruno Fernandes (MUN a HUL) the clear alternative/vice.**

- CONFIRMED: premierleague.com's launch framing makes Haaland and Bruno the two GW1 armband candidates: Haaland the fast starter; Bruno opens HUL (a) then IPS (h) and was **the top scorer over the final 17 GWs of 2025/26 (129 pts, 7.6/match)** — [MUST-HAVES article](https://www.premierleague.com/en/news/4675553/why-fernandes-and-haaland-look-like-must-haves-to-start-202627-fpl).
- REPORTED: Onside's cheat sheet projects Haaland at "56% top-1k captaincy… the unanimous Tier-1 default", Bruno vice ([Onside cheat sheet](https://onsidearena.com/tips/fpl-cheat-sheet-2026-27) — caution: page still carries pre-launch prices, see §9).
- CONFIRMED: The reigning champion's philosophy pushes the same way: 2025/26 world #1 Erik Ibsen captained Haaland 22/38 weeks — "a bad captaincy can ruin a Gameweek" — [FPL champion interview](https://www.premierleague.com/en/news/4672128/fpl-champion-how-to-pick-your-captain-and-maximise-your-chips).

**Igor Thiago (our captain):** he is firmly **on the community radar as a pick, not as a captain**. He is the #3 most-selected forward (22.9%, live API) and launch content is glowing — "the dictator pick in the cheap-forward bracket" with 28.7 xP over the opening run and 86/90 expected minutes ([Attacking Football predicted points](https://www.attackingfootball.com/fantasy-premier-league/best-fpl-players-by-predicted-points-2026-27/), via search snippets — site 403s direct fetch); Pranil Sheth: "on penalties and the talisman for Brentford" ([PL.com expert panel](https://www.premierleague.com/en/news/4672877/fpl-experts-price-predictions-for-202627)); Brentford's GW1–5 FDR (2.8) is second only to Liverpool and they avoid all of last season's top five, opening **home v Tottenham** ([PL.com Brentford guide](https://www.premierleague.com/en/news/4676527/best-teams-to-invest-in-for-202627-fantasy-brentford)). But **no source we found proposes him as GW1 captain**. Attacking Football's xP list has Haaland 35.5 xP (81/100) and Bruno 35.8 xP over the run, both above Thiago's 28.7 — order-consistent with our model's within-GW1 ranking (Haaland 8.14 > Thiago 7.62) once Haaland is excluded from the squad. Captaining Thiago is therefore a *consequence* of our no-Haaland structure, not a community-endorsed armband; vs the field it is a double differential (no Haaland + differential captain).

Also note the caveat baked into his 181 pts: 9 of 22 goals were penalties ([PL.com Brentford guide](https://www.premierleague.com/en/news/4676527/best-teams-to-invest-in-for-202627-fantasy-brentford)) — pen-dependent xP is streaky, which matters when he's carrying our armband.

---

## 4. Concrete early drafts collected (all start Haaland)

Only a handful of complete or near-complete GW1 drafts are public this early:

**(a) fpl.page "Oscar" first draft** ([source](https://fpl.page/article/fpl-gw1-team-selection-first-draft-2627), CONFIRMED fetch): Verbruggen; Guéhi, Jacquet, Mosquera; Mbeumo, Rogers, Palmer, Szoboszlai, Anderson; João Félix, **Haaland**; bench Steele, Cash, Kayode, Kusi-Asare ("dead spot last on the bench"). 3-5-2, premium-heavy, deliberately cheap dead bench — the opposite of a BB-GW1 structure. Names Palmer (v HUL, GW4) as an early captain window.

**(b) FFScout-associated early draft** (recovered from FFScout search snippets across their launch articles, REPORTED — assembled list: Raya, Roefs; Gabriel, O'Reilly, Senesi, Mukiele, Guéhi; B.Fernandes, Semenyo, Rice, Anderson, Gibbs-White; **Haaland**, Thiago, Kroupi). Notable: it contains **four of our players** (Raya, Gabriel, Gibbs-White, Thiago) *plus* Mukiele — and still fits Haaland + Bruno by using £4.0–6.5m defenders and no second premium mid beyond Bruno.

**(c) Onside cheat sheet XI** ([source](https://onsidearena.com/tips/fpl-cheat-sheet-2026-27), ANECDOTAL/STALE — pre-launch predicted prices): Roefs; Gabriel, Robinson, Tarkowski, Murillo; Saka, Bruno (VC), Semenyo, Rogers; **Haaland (C)**, Watkins. Haaland captain, Isak flagged as the differential, Palmer flagged as the "trap".

**(d) SportBible launch "hidden gems"** ([source](https://www.sportbible.com/football/football-news/fpl-hidden-gems-season-launch-486194-20260723), CONFIRMED fetch): Shaw £4.5m (ours), Bobby Thomas £4.0m (COV), Kinsky £4.5m, Szoboszlai £7.0m (ours — "arguably the best value player in FPL", set pieces + possible pens post-Salah), Mosquera £5.5m (starts while "Saliba remains sidelined with a back injury").

**Template read from day-2 API (CONFIRMED):** the proto-template is Haaland (66%) + João Pedro (44.6%) + Bruno (42.6%) + Rogers (38.3%) + Szoboszlai (33%) + Raya (27.3%) + a City defender (Guéhi 26 / O'Reilly 24.7) + Thiago/Kroupi (~23%) + £4.0–4.5m TOT/IPS enablers (Dubravka 21.4, Diop 19.7, Pedro Porro 21.5, Senesi 18.9).

---

## 5. Structure themes: promoted-opponent stacking, DefCon, bench philosophy

**Promoted-opponent targeting is fully consensus — we are not contrarian here, except in one detail.** AllAboutFPL's fixture analysis names Man Utd the standout opening run ("back-to-back fixtures against promoted opposition", appealing to GW8) with MCI/LIV/ARS the elite tier ([AllAboutFPL fixtures](https://allaboutfpl.com/2026/07/fpl-fixture-analysis-for-the-2026-27-fpl-season-pl-fixtures/)); premierleague.com pushes Bruno (HUL a, IPS h) and Haaland (COV h GW3, IPS h GW7) for the same reason, plus value angles Dewsbury-Hall £6.5m and Bruno Guimarães £7.0m v promoted sides GW5–6 ([MUST-HAVES](https://www.premierleague.com/en/news/4675553/why-fernandes-and-haaland-look-like-must-haves-to-start-202627-fpl)); Sunderland's IPS (a) → FUL (h) start makes their assets "viable differential options" ([chaseyoursport](https://www.chaseyoursport.com/football/sunderland-best-players-this-season-2025-26/11854), REPORTED). One nuance: **NFO v LEE is not a promoted-opponent fixture** — Leeds survived 2025/26 (Calvert-Lewin is at Leeds at £6.0m per [Scout's 15](https://www.premierleague.com/en/news/4680821/the-scouts-analysis-of-15-key-player-prices-in-202627-fantasy)); our internal framing of "promoted targeting" should re-label the NFO pair as merely "favourable home fixture".

**DefCon second season:** last season's DefCon winners were repriced (Senesi £6.0m +£1.5m, Tarkowski £6.0m, Lacroix £6.0m; every 20+ DefCon defender now ≥£5.0m), while all COV/HUL/IPS defenders are ≤£4.0m — the community bargain pool ([FFScout 9 impressions](https://www.fantasyfootballscout.co.uk/2026/07/23/fpl-2026-27-9-first-impressions-of-the-player-prices)). Ipswich's Diop (£4.0m, 19.7%) is the enabler of choice despite Ipswich's hard opening — a pick bought to sit on benches, exactly what BB GW1 squads avoid.

**Bench philosophy:** the loudest public drafts run **premium concentration + dead bench** (Oscar's Kusi-Asare "dead spot"), funded by £4.0–4.5m GKs (eight starting GKs under £5.0m this season, [FFScout](https://www.fantasyfootballscout.co.uk/2026/07/23/fpl-2026-27-9-first-impressions-of-the-player-prices)). Our 15-playing-bodies structure (£99.5m spent, cheapest player £4.5m Shaw who scored 113 pts and started all 38 last season, [SportBible](https://www.sportbible.com/football/football-news/fpl-hidden-gems-season-launch-486194-20260723)) only makes sense *because* of BB GW1 — which is precisely the structure FFScout's 2025/26 BB-GW1 article warned "is likely to impact the balance of your squad, potentially forcing an early Wildcard" ([FFScout BB GW1 pros/cons](https://www.fantasyfootballscout.co.uk/2025/08/04/gameweek-1-bench-boost-is-it-a-good-idea/)). Our model indeed forces that Wildcard in GW2 — the con realised by design.

---

## 6. Chip strategy: our GW1–5 burn vs the field

2026/27 chip rules CONFIRMED ([premierleague.com chips](https://www.premierleague.com/en/news/4679879/whats-happening-with-fpl-chips-in-202627)): two sets of four; set 1 expires at the GW19 deadline, 13:30 GMT Sat 2 Jan 2027; one chip per GW; Free Hit not playable in GW1.

Where the public conversation sits:

| Chip | Our model | Community consensus |
|---|---|---|
| Bench Boost | **GW1** (+11.1 xP) | Minority-viable, majority-sceptical. In the identical 2025/26 debate, 1 of ~10 PL.com experts (Nick Harris) backed BB GW1; most preferred GW2–8+ or waiting for data/AFCON value ([experts on early BB](https://www.premierleague.com/en/news/4364931/fpl-experts-is-it-a-good-idea-to-bench-boost-in-gameweek-1)). FFScout's dedicated piece was deliberately split — pros: predictable line-ups, only week you can pair unlimited pre-season transfers with BB, few early DGWs; cons: GW1 is "arguably the most challenging to predict" ([FFScout](https://www.fantasyfootballscout.co.uk/2025/08/04/gameweek-1-bench-boost-is-it-a-good-idea/)). PL.com's 2026/27 chips article explicitly lists BB GW1 as a legitimate line. |
| Wildcard | **GW2** | Against. FPL Pilot: hold "until the fixtures settle, usually a few gameweeks in" ([FPL Pilot](https://www.fplpilot.com/blog/fpl-chip-strategy-2026-27)). Champion Ibsen used a GW2 WC in 2025/26 and **called it suboptimal in hindsight** ([champion interview](https://www.premierleague.com/en/news/4672128/fpl-champion-how-to-pick-your-captain-and-maximise-your-chips)). |
| Triple Captain | **GW3, Watkins (AVL a HUL)** (+7.6) | The named community windows are **Haaland home v COV (GW3) or IPS (GW7)**, or Bruno v IPS (GW2) ([PL.com chips](https://www.premierleague.com/en/news/4679879/whats-happening-with-fpl-chips-in-202627)). Nobody mentions Watkins TC. FPL Pilot: "Do not spend it on a single gameweek out of impatience" — though with no early DGWs in set 1, a single-GW TC before GW19 is structurally forced for everyone. |
| Free Hit | **GW5** (+2.0) | Hold for blanks/emergencies ([FPL Pilot](https://www.fplpilot.com/blog/fpl-chip-strategy-2026-27)). +2.0 xP is a thin edge to burn insurance for. |

2025/26 ground truth (CONFIRMED): elite/expert first-set chips clustered mid-to-late in the half, and the champion's big hauls came from second-half chips timed to DGWs (BB 112 pts GW33, TC 92 pts GW36) — but note set-2 chips can exploit DGWs while **set-1 chips (GW1–19) almost never see a double**, so the 2025/26 second-half evidence overstates the cost of early set-1 usage. The honest read: our BB GW1 is defensible and has named public backers; our WC GW2 + TC GW3 + FH GW5 cascade has none, and its known cause — the solver's greedy-within-horizon chip EV with no opportunity cost beyond GW8 — matches exactly the failure mode FPL Pilot warns about ("treating chips as point generators rather than multipliers… lose rank by playing the right chip in the wrong week").

---

## 7. Our 15, player by player, vs day-2 ownership (live API, 2026-07-23)

Sum of selected_by% across our 15 = **267.4%** (avg 17.8%/player). Eleven of fifteen sit inside the overall top-45 most-selected. We are a *template-adjacent* squad with four genuine differentials — and two enormous template holes (Haaland 66%, Bruno 42.6%).

| Player | Pos | £ | Sel% | Pos-rank | Verdict vs field |
|---|---|---|---|---|---|
| Raya (ARS) | GK | 6.0 | 27.3% | #1 GK | Full template — most-selected GK despite being the priciest since 2021/22 ([FFScout](https://www.fantasyfootballscout.co.uk/2026/07/23/fpl-2026-27-9-first-impressions-of-the-player-prices)). Value-critics prefer £4.0–4.5m TOT keepers. |
| Lammens (MUN) | GK | 5.0 | 17.2% | #3 GK | Template-ish; MUN@HUL GW1 fits the consensus MUN-stack. As our BB GW1 second keeper he must *start* to pay off — verify no rotation. |
| Gabriel (ARS) | DEF | 8.0 | 22.2% | #3 DEF | Template. Top defender of 2025/26 (209 pts); universally endorsed. £8.0m price is the only debate. |
| Tarkowski (EVE) | DEF | 6.0 | 16.0% | #8 DEF | Template DefCon pick, named in FFScout's repricing analysis; EVE h CRY GW1 is fine. |
| Mukiele (SUN) | DEF | 5.5 | 10.0% | #16 | Mild differential with expert backing (appears in the FFScout-associated draft; SUN@IPS consensus-approved). |
| N.Williams (NFO) | DEF | 5.0 | 8.2% | #20 | Differential. Forest wing-back under Glasner (REPORTED); v LEE (h) is favourable but not a promoted opponent. |
| Shaw (MUN) | DEF | 4.5 | 11.9% | #13 | Value consensus pick — SportBible hidden gem (113 pts, 38/38 starts). Template-adjacent. |
| Gibbs-White (NFO) | MID | 8.0 | 14.1% | #8 MID | Semi-template with strong editorial tailwind (15g+4a in 25/26, PL Player-of-Matchweek coverage, in the FFScout draft). |
| Mbeumo (MUN) | MID | 8.0 | 13.3% | #11 MID | Semi-template; in Oscar's draft; the cheaper way into the consensus MUN@HUL stack (the field's way is Bruno at 42.6%). |
| Szoboszlai (LIV) | MID | 7.0 | 33.0% | #3 MID | Full template — "arguably the best value player in FPL" (SportBible), possible pens post-Salah (FFScout). Note GW1 is NEW (a), a hard fixture — the field owns him for the season value, not GW1. |
| Gakpo (LIV) | MID | 7.0 | 5.2% | #25 | Big differential; no launch coverage found at all; NEW (a) GW1. Weakest external validation in our XI. |
| E.Le Fée (SUN) | MID | 6.0 | 3.3% | #31 | Biggest differential in our squad. Indirect support only (Sunderland's 25/26 creator, 4g+4a; SUN@IPS narrative). The field's SUN exposure is Brobbey (17.4%). |
| Thiago (BRE) | FWD | 8.0 | 22.9% | #3 FWD | Template as a *pick* (top-10 overall selected); radical as *captain* (see §3). |
| Watkins (AVL) | FWD | 8.0 | 18.2% | #4 FWD | Template-ish pick; but AVL open BHA (a) — nobody flags Watkins as a GW1 points target, and our GW3 TC on him (at HUL) is unsupported externally. |
| João Pedro (CHE) | FWD | 7.5 | 44.6% | #2 FWD | Maximum template — #2 selected player in the game; FFScout calls his unchanged £7.5m an outright pricing error in owners' favour. FUL (a) GW1. |
| — Haaland (not owned) | FWD | 15.5 | 66.0% | #1 | The defining anti-template position of our squad (§2). |
| — B.Fernandes (not owned until WC GW2) | MID | 12.0 | 42.6% | #1 MID | Second-biggest hole GW1; the field's preferred promoted-stack captain-alternative. Our WC GW2 buys him one week after his HUL (a) opener. |

Field picks we *don't* own that a consensus-checker would raise: Rogers (CHE £7.5m, 38.3% — highest-owned player not in our squad after Haaland/Bruno), Guéhi/O'Reilly (MCI defence 26/24.7%), Rice (24.9%), Kroupi Jr (22.8%), Pedro Porro/Dubravka/Diop (cheap TOT/IPS enablers ~20%), Semenyo (19.5%), Gyökeres (18.1%), Brobbey (17.4%).

**Ownership caveat:** day-2 selection percentages are early-adopter-skewed and will move a lot before 21 Aug; treat ranks, not levels, as signal. (Effective ownership at the deadline is historically higher than day-2 for premiums like Haaland, not lower.)

---

## 8. Bottom line — where we sit vs consensus

1. **13/15 of our picks are defensible or outright template**; the squad's per-player quality is not in question anywhere consensus has an opinion.
2. **The two structural bets nobody else is making: no Haaland GW1, and the GW1–5 four-chip burn.** Both trace to the same solver artifact (chip EV computed only within the GW1–8 horizon; no holding-value for chips through GW19, no EO/variance term for Haaland).
3. If we keep the no-Haaland line, the GW6 entry (at Liverpool) is the worst of the plausible entry weeks — GW5 (SUN h) or GW7 (IPS h) dominate.
4. If we keep BB GW1, we're in a real but minority camp with named backers (Nick Harris et al.) and official acknowledgement of the line; WC GW2 has an anti-testimonial from the reigning champion; TC-on-Watkins-GW3 and FH GW5 have zero external support.
5. Captain Thiago is a stacked differential (non-Haaland squad + non-consensus armband) on a player whose 2025/26 output was 41% penalties. The community's non-Haaland captain of choice for GW1 is Bruno Fernandes at Hull — whom we don't buy until GW2.

---

## 9. Source reliability notes

- **premierleague.com (The Scout)** — first-party, launch-current, prices/fixtures verified against the live API. Highest confidence.
- **fantasyfootballscout.co.uk** — launch-current (22–23 Jul articles), detailed and price-accurate; first-draft reveals still paywalled at research time.
- **Live FPL API** (`bootstrap-static`, `fixtures`) — ground truth for prices, ownership, deadline (2026-08-21T17:30:00Z), GW1 fixtures; snapshot taken 2026-07-23.
- **fpl.page, SportBible, AllAboutFPL, chaseyoursport** — launch-window editorial, single-source; treated as REPORTED.
- **Fantasy Football Fix** — pre-launch price-prediction content only; strategic stance (Haaland essential) clear, no post-launch draft yet.
- **onsidearena.com** — internally inconsistent (cheat sheet uses pre-launch prices, e.g. Haaland £14.0m; FDR page mixes in 2025/26 promoted teams). Used only for direction-of-consensus and 2025/26 season-review numbers that cross-check elsewhere. Low confidence otherwise.
- **attackingfootball.com** — 403 to direct fetch; xP figures (Bruno 35.8, Haaland 35.5, Thiago 28.7, Mbeumo 27.9) recovered via search snippets only.
- **Not reachable this cycle:** FPL Review / planfpl / fplform 2026/27 output, r/FantasyPL launch threads, X/Twitter drafts, YouTube first-draft videos (titles confirmed to exist — e.g. "FIRST FPL DRAFT 2026/27 | Is This The Perfect GW1 Team?" — but descriptions/transcripts not extractable). Re-run this scan in 1–2 weeks; draft density will be 10x by early August.
