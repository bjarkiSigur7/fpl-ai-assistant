/**
 * Mock fixture data for standalone UI development (NEXT_PUBLIC_MOCK=1).
 *
 * Scenario: 2026-07-22 pre-season. The 2026-27 game has not launched; the model desk is
 * running a 2025-26 backtest demo at GW34 with a 5-GW horizon (GW34-38), matching the
 * walk-forward eval documented in docs/STATUS.md (the GW34 top-10 xP values are the real
 * ones from that eval). Everything is generated deterministically from a seeded PRNG so
 * every reload shows identical numbers. Shapes mirror the backend pydantic contracts in
 * src/lib/types.ts verbatim.
 */

import type {
  ChipAdvice,
  ChipCurvePoint,
  ChipCurvesResponse,
  DreamTeam,
  FixturePrediction,
  GwPlan,
  GwPrediction,
  MyTeamResponse,
  PlanResult,
  PlayerDetail,
  PlayerGwHistory,
  Position,
  PredictionsResponse,
  Recommendation,
  SquadState,
  StabilityEntry,
  StateResponse,
  TeamInfo,
} from "@/lib/types";

// ---------------------------------------------------------------------------
// deterministic PRNG
// ---------------------------------------------------------------------------

/** mulberry32 — tiny deterministic PRNG; same seed, same stream, every load. */
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** One deterministic uniform draw keyed by (a, b, c). */
function rand(a: number, b: number, c = 0): number {
  return mulberry32(a * 2654435761 + b * 40503 + c * 9973 + 7)();
}

const round2 = (v: number) => Math.round(v * 100) / 100;

// ---------------------------------------------------------------------------
// clubs (2025-26 Premier League)
// ---------------------------------------------------------------------------

interface Club extends TeamInfo {
  strength: number;
}

export const CLUBS: Club[] = [
  { team_code: 3, name: "Arsenal", short_name: "ARS", strength: 4.7 },
  { team_code: 7, name: "Aston Villa", short_name: "AVL", strength: 3.9 },
  { team_code: 91, name: "Bournemouth", short_name: "BOU", strength: 3.5 },
  { team_code: 94, name: "Brentford", short_name: "BRE", strength: 3.4 },
  { team_code: 36, name: "Brighton", short_name: "BHA", strength: 3.7 },
  { team_code: 90, name: "Burnley", short_name: "BUR", strength: 2.6 },
  { team_code: 8, name: "Chelsea", short_name: "CHE", strength: 4.2 },
  { team_code: 31, name: "Crystal Palace", short_name: "CRY", strength: 3.5 },
  { team_code: 11, name: "Everton", short_name: "EVE", strength: 3.2 },
  { team_code: 54, name: "Fulham", short_name: "FUL", strength: 3.3 },
  { team_code: 2, name: "Leeds", short_name: "LEE", strength: 2.9 },
  { team_code: 14, name: "Liverpool", short_name: "LIV", strength: 4.9 },
  { team_code: 43, name: "Man City", short_name: "MCI", strength: 4.8 },
  { team_code: 1, name: "Man Utd", short_name: "MUN", strength: 4.0 },
  { team_code: 4, name: "Newcastle", short_name: "NEW", strength: 4.1 },
  { team_code: 17, name: "Nott'm Forest", short_name: "NFO", strength: 3.4 },
  { team_code: 56, name: "Sunderland", short_name: "SUN", strength: 2.7 },
  { team_code: 6, name: "Spurs", short_name: "TOT", strength: 4.0 },
  { team_code: 21, name: "West Ham", short_name: "WHU", strength: 3.1 },
  { team_code: 39, name: "Wolves", short_name: "WOL", strength: 3.0 },
];

const clubByShort = new Map(CLUBS.map((c) => [c.short_name, c]));

export const TEAMS: TeamInfo[] = CLUBS.map(({ team_code, name, short_name }) => ({
  team_code,
  name,
  short_name,
}));

// ---------------------------------------------------------------------------
// fixtures GW34-38 (home short, away short); GW37 has the LIV/MCI double
// ---------------------------------------------------------------------------

const PAIRINGS: Record<number, [string, string][]> = {
  34: [
    ["LIV", "BHA"], ["ARS", "AVL"], ["MCI", "WOL"], ["MUN", "EVE"], ["TOT", "FUL"],
    ["NEW", "BUR"], ["CHE", "CRY"], ["BOU", "LEE"], ["NFO", "SUN"], ["WHU", "BRE"],
  ],
  35: [
    ["BHA", "MUN"], ["AVL", "MCI"], ["WOL", "TOT"], ["EVE", "ARS"], ["FUL", "LIV"],
    ["BUR", "CHE"], ["CRY", "NFO"], ["LEE", "WHU"], ["SUN", "NEW"], ["BRE", "BOU"],
  ],
  36: [
    ["LIV", "ARS"], ["MCI", "CHE"], ["MUN", "TOT"], ["NEW", "AVL"], ["BHA", "CRY"],
    ["EVE", "FUL"], ["WOL", "BUR"], ["LEE", "NFO"], ["SUN", "WHU"], ["BOU", "BRE"],
  ],
  37: [
    ["WHU", "LIV"], ["MCI", "EVE"], ["ARS", "TOT"], ["AVL", "WOL"], ["BOU", "CHE"],
    ["BRE", "LEE"], ["BHA", "NFO"], ["BUR", "MUN"], ["CRY", "SUN"], ["FUL", "NEW"],
    ["MCI", "LIV"], // rescheduled derby — the DGW
  ],
  38: [
    ["LIV", "WOL"], ["ARS", "SUN"], ["MCI", "NFO"], ["CHE", "MUN"], ["TOT", "BHA"],
    ["NEW", "LEE"], ["AVL", "BUR"], ["EVE", "BOU"], ["CRY", "FUL"], ["WHU", "BRE"],
  ],
};

const GW_DATES: Record<number, string> = {
  34: "2026-04-25",
  35: "2026-05-02",
  36: "2026-05-09",
  37: "2026-05-16",
  38: "2026-05-24",
};

export const DEMO_SEASON = 2025;
export const DEMO_GWS = [34, 35, 36, 37, 38];
export const DEMO_GW = 34;
const AS_OF = "2026-07-22T06:30:00Z";

interface MockFixture {
  gw: number;
  fpl_fixture_id: number;
  home: Club;
  away: Club;
  kickoff_utc: string;
}

const FIXTURES: MockFixture[] = (() => {
  const out: MockFixture[] = [];
  let id = 20340;
  for (const gw of DEMO_GWS) {
    PAIRINGS[gw].forEach(([h, a], i) => {
      const slot = i % 3;
      const hour = slot === 0 ? "12:30" : slot === 1 ? "15:00" : "17:30";
      out.push({
        gw,
        fpl_fixture_id: id++,
        home: clubByShort.get(h)!,
        away: clubByShort.get(a)!,
        kickoff_utc: `${GW_DATES[gw]}T${hour}:00Z`,
      });
    });
  }
  return out;
})();

function fixturesFor(teamCode: number, gw: number): MockFixture[] {
  return FIXTURES.filter(
    (f) => f.gw === gw && (f.home.team_code === teamCode || f.away.team_code === teamCode),
  );
}

/** Fixture attractiveness multiplier for a team's players. */
function fixtureMult(opponent: Club, home: boolean): number {
  return 1.0 + (3.55 - opponent.strength) * 0.09 + (home ? 0.05 : -0.05);
}

// ---------------------------------------------------------------------------
// player catalog (web_name, first, second, pos, club, price 0.1m, base xP/GW, q0)
// ---------------------------------------------------------------------------

type CatalogRow = [string, string, string, Position, string, number, number, number, number?];

// A few real FPL element codes for verisimilitude; the rest are auto-assigned.
const CATALOG: CatalogRow[] = [
  // Arsenal
  ["Raya", "David", "Raya Martín", "GKP", "ARS", 56, 3.8, 0.05],
  ["Gabriel", "Gabriel", "dos Santos Magalhães", "DEF", "ARS", 63, 4.4, 0.07, 226597],
  ["Saliba", "William", "Saliba", "DEF", "ARS", 61, 3.9, 0.06],
  ["Timber", "Jurriën", "Timber", "DEF", "ARS", 58, 3.6, 0.08],
  ["Rice", "Declan", "Rice", "MID", "ARS", 68, 4.6, 0.05, 204480],
  ["Saka", "Bukayo", "Saka", "MID", "ARS", 102, 5.5, 0.1, 223340],
  ["Ødegaard", "Martin", "Ødegaard", "MID", "ARS", 82, 4.6, 0.09],
  ["Eze", "Eberechi", "Eze", "MID", "ARS", 78, 4.3, 0.11],
  ["Gyökeres", "Viktor", "Gyökeres", "FWD", "ARS", 95, 5.2, 0.08],
  // Aston Villa
  ["Watkins", "Ollie", "Watkins", "FWD", "AVL", 88, 4.4, 0.09, 178301],
  ["Rogers", "Morgan", "Rogers", "MID", "AVL", 70, 4.5, 0.07],
  ["Konsa", "Ezri", "Konsa Ngoyo", "DEF", "AVL", 45, 3.0, 0.08],
  ["Martínez", "Emiliano", "Martínez Romero", "GKP", "AVL", 50, 3.4, 0.06],
  // Bournemouth
  ["Semenyo", "Antoine", "Semenyo", "MID", "BOU", 70, 4.3, 0.07],
  ["Evanilson", "Evanilson", "de Lima Batista", "FWD", "BOU", 70, 3.9, 0.55],
  ["Petrović", "Đorđe", "Petrović", "GKP", "BOU", 45, 3.3, 0.06],
  ["Senesi", "Marcos", "Senesi", "DEF", "BOU", 46, 3.2, 0.08],
  ["Kluivert", "Justin", "Kluivert", "MID", "BOU", 60, 3.8, 0.14],
  ["Tavernier", "Marcus", "Tavernier", "MID", "BOU", 55, 3.3, 0.12],
  // Brentford
  ["Thiago", "Igor", "Thiago", "FWD", "BRE", 60, 3.7, 0.12],
  ["Damsgaard", "Mikkel", "Damsgaard", "MID", "BRE", 55, 3.4, 0.1],
  ["Kelleher", "Caoimhín", "Kelleher", "GKP", "BRE", 45, 3.2, 0.07],
  ["Van den Berg", "Sepp", "van den Berg", "DEF", "BRE", 45, 3.0, 0.09],
  ["Schade", "Kevin", "Schade", "MID", "BRE", 58, 3.6, 0.11],
  // Brighton
  ["Mitoma", "Kaoru", "Mitoma", "MID", "BHA", 65, 3.9, 0.12],
  ["Welbeck", "Danny", "Welbeck", "FWD", "BHA", 60, 3.7, 0.15],
  ["Verbruggen", "Bart", "Verbruggen", "GKP", "BHA", 45, 3.2, 0.06],
  ["Van Hecke", "Jan Paul", "van Hecke", "DEF", "BHA", 45, 3.1, 0.08],
  ["Baleba", "Carlos", "Baleba", "MID", "BHA", 50, 3.0, 0.1],
  // Burnley
  ["Dúbravka", "Martin", "Dúbravka", "GKP", "BUR", 40, 3.0, 0.08],
  ["Flemming", "Zian", "Flemming", "FWD", "BUR", 55, 3.0, 0.15],
  ["Estève", "Maxime", "Estève", "DEF", "BUR", 40, 2.7, 0.1],
  ["Anthony", "Jaidon", "Anthony", "MID", "BUR", 50, 3.0, 0.12],
  // Chelsea
  ["Palmer", "Cole", "Palmer", "MID", "CHE", 105, 5.3, 0.06, 244851],
  ["Sánchez", "Robert", "Sánchez", "GKP", "CHE", 50, 3.4, 0.07],
  ["Cucurella", "Marc", "Cucurella Saseta", "DEF", "CHE", 58, 3.7, 0.06],
  ["Colwill", "Levi", "Colwill", "DEF", "CHE", 52, 3.3, 0.1],
  ["João Pedro", "João Pedro", "Junqueira de Jesus", "FWD", "CHE", 78, 4.3, 0.1],
  ["Neto", "Pedro", "Lomba Neto", "MID", "CHE", 65, 3.8, 0.15],
  ["Caicedo", "Moisés", "Caicedo Corozo", "MID", "CHE", 55, 3.2, 0.07],
  // Crystal Palace
  ["Muñoz", "Daniel", "Muñoz Mejía", "DEF", "CRY", 55, 3.8, 0.06],
  ["Mateta", "Jean-Philippe", "Mateta", "FWD", "CRY", 75, 4.1, 0.09],
  ["Guéhi", "Marc", "Guéhi", "DEF", "CRY", 48, 3.2, 0.07],
  ["Henderson", "Dean", "Henderson", "GKP", "CRY", 45, 3.3, 0.06],
  ["Sarr", "Ismaïla", "Sarr", "MID", "CRY", 62, 3.7, 0.1],
  // Everton
  ["Grealish", "Jack", "Grealish", "MID", "EVE", 68, 4.0, 0.12],
  ["Pickford", "Jordan", "Pickford", "GKP", "EVE", 50, 3.6, 0.04],
  ["Ndiaye", "Iliman", "Ndiaye", "MID", "EVE", 60, 3.6, 0.1],
  ["Tarkowski", "James", "Tarkowski", "DEF", "EVE", 45, 3.1, 0.07],
  ["Barry", "Thierno", "Barry", "FWD", "EVE", 55, 3.2, 0.14],
  // Fulham
  ["B.Muniz", "Rodrigo", "Muniz Carvalho", "FWD", "FUL", 60, 3.5, 0.15],
  ["Iwobi", "Alex", "Iwobi", "MID", "FUL", 58, 3.5, 0.09],
  ["Leno", "Bernd", "Leno", "GKP", "FUL", 50, 3.4, 0.05],
  ["Bassey", "Calvin", "Bassey", "DEF", "FUL", 45, 3.0, 0.08],
  // Leeds
  ["Piroe", "Joël", "Piroe", "FWD", "LEE", 55, 3.1, 0.13],
  ["Aaronson", "Brenden", "Aaronson", "MID", "LEE", 55, 3.2, 0.11],
  ["Perri", "Lucas", "Perri", "GKP", "LEE", 45, 3.1, 0.07],
  ["Struijk", "Pascal", "Struijk", "DEF", "LEE", 42, 2.9, 0.09],
  ["Stach", "Anton", "Stach", "MID", "LEE", 50, 3.0, 0.1],
  // Liverpool
  ["M.Salah", "Mohamed", "Salah", "MID", "LIV", 131, 6.0, 0.04, 118748],
  ["Wirtz", "Florian", "Wirtz", "MID", "LIV", 88, 4.8, 0.08],
  ["Gakpo", "Cody", "Gakpo", "MID", "LIV", 76, 4.4, 0.09],
  ["Szoboszlai", "Dominik", "Szoboszlai", "MID", "LIV", 68, 4.2, 0.08],
  ["Isak", "Alexander", "Isak", "FWD", "LIV", 105, 5.0, 0.12, 219168],
  ["Ekitiké", "Hugo", "Ekitiké", "FWD", "LIV", 86, 4.6, 0.1],
  ["Van Dijk", "Virgil", "van Dijk", "DEF", "LIV", 60, 4.0, 0.05, 97032],
  ["Frimpong", "Jeremie", "Frimpong", "DEF", "LIV", 60, 3.8, 0.1],
  ["Alisson", "Alisson", "Ramses Becker", "GKP", "LIV", 55, 3.6, 0.1],
  // Man City
  ["Haaland", "Erling", "Haaland", "FWD", "MCI", 148, 5.8, 0.06, 223094],
  ["Foden", "Phil", "Foden", "MID", "MCI", 92, 4.7, 0.12, 209244],
  ["Marmoush", "Omar", "Marmoush", "FWD", "MCI", 72, 4.1, 0.14],
  ["Doku", "Jérémy", "Doku", "MID", "MCI", 68, 3.9, 0.18],
  ["Gvardiol", "Joško", "Gvardiol", "DEF", "MCI", 60, 3.8, 0.08],
  ["Cherki", "Rayan", "Cherki", "MID", "MCI", 65, 3.8, 0.2],
  ["Donnarumma", "Gianluigi", "Donnarumma", "GKP", "MCI", 55, 3.7, 0.06],
  // Man Utd
  ["B.Fernandes", "Bruno", "Borges Fernandes", "MID", "MUN", 90, 5.0, 0.04, 141746],
  ["Cunha", "Matheus", "Santos Carneiro da Cunha", "FWD", "MUN", 78, 4.5, 0.09],
  ["Mbeumo", "Bryan", "Mbeumo", "MID", "MUN", 82, 4.7, 0.07],
  ["Šeško", "Benjamin", "Šeško", "FWD", "MUN", 76, 4.2, 0.12],
  ["Dalot", "Diogo", "Dalot Teixeira", "DEF", "MUN", 50, 3.1, 0.12],
  ["Onana", "André", "Onana", "GKP", "MUN", 50, 3.2, 0.1],
  // Newcastle
  ["Gordon", "Anthony", "Gordon", "MID", "NEW", 72, 4.1, 0.1],
  ["Woltemade", "Nick", "Woltemade", "FWD", "NEW", 70, 4.0, 0.1],
  ["Tonali", "Sandro", "Tonali", "MID", "NEW", 58, 3.4, 0.08],
  ["Hall", "Lewis", "Hall", "DEF", "NEW", 52, 3.5, 0.09],
  ["Pope", "Nick", "Pope", "GKP", "NEW", 50, 3.4, 0.08],
  ["Burn", "Dan", "Burn", "DEF", "NEW", 46, 3.0, 0.1],
  // Nott'm Forest
  ["Gibbs-White", "Morgan", "Gibbs-White", "MID", "NFO", 75, 4.2, 0.08],
  ["Wood", "Chris", "Wood", "FWD", "NFO", 72, 4.0, 0.09],
  ["Sels", "Matz", "Sels", "GKP", "NFO", 50, 3.5, 0.05],
  ["Aina", "Ola", "Aina", "DEF", "NFO", 50, 3.5, 0.06],
  ["Milenković", "Nikola", "Milenković", "DEF", "NFO", 48, 3.3, 0.07],
  ["Hudson-Odoi", "Callum", "Hudson-Odoi", "MID", "NFO", 55, 3.4, 0.12],
  // Sunderland
  ["Le Fée", "Enzo", "Le Fée", "MID", "SUN", 52, 3.1, 0.1],
  ["Isidor", "Wilson", "Isidor", "FWD", "SUN", 52, 3.0, 0.13],
  ["Roefs", "Robin", "Roefs", "GKP", "SUN", 40, 3.0, 0.08],
  ["Ballard", "Daniel", "Ballard", "DEF", "SUN", 42, 3.0, 0.09],
  ["Xhaka", "Granit", "Xhaka", "MID", "SUN", 50, 3.1, 0.06],
  // Spurs
  ["Solanke", "Dominic", "Solanke-Mitchell", "FWD", "TOT", 76, 4.8, 0.08],
  ["Xavi", "Xavi", "Simons", "MID", "TOT", 76, 4.6, 0.1],
  ["Kudus", "Mohammed", "Kudus", "MID", "TOT", 68, 4.0, 0.12],
  ["Romero", "Cristian", "Romero", "DEF", "TOT", 52, 3.4, 0.1],
  ["Vicario", "Guglielmo", "Vicario", "GKP", "TOT", 50, 3.5, 0.07],
  ["Porro", "Pedro", "Porro", "DEF", "TOT", 55, 3.6, 0.08],
  // West Ham
  ["Bowen", "Jarrod", "Bowen", "MID", "WHU", 78, 4.4, 0.06, 178186],
  ["Füllkrug", "Niclas", "Füllkrug", "FWD", "WHU", 60, 3.4, 0.18],
  ["Areola", "Alphonse", "Areola", "GKP", "WHU", 45, 3.2, 0.1],
  ["Wan-Bissaka", "Aaron", "Wan-Bissaka", "DEF", "WHU", 45, 3.1, 0.08],
  ["Paquetá", "Lucas", "Tolentino Coelho de Lima", "MID", "WHU", 60, 3.6, 0.12],
  ["Summerville", "Crysencio", "Summerville", "MID", "WHU", 55, 3.4, 0.14],
  // Wolves
  ["Strand Larsen", "Jørgen", "Strand Larsen", "FWD", "WOL", 65, 3.9, 0.09],
  ["André", "André", "Trindade da Costa Neto", "MID", "WOL", 50, 3.0, 0.09],
  ["Sá", "José", "Malheiro de Sá", "GKP", "WOL", 45, 3.3, 0.06],
  ["Agbadou", "Emmanuel", "Agbadou", "DEF", "WOL", 42, 3.0, 0.08],
  ["Arias", "Santiago", "Arias", "MID", "WOL", 55, 3.4, 0.12],
];

export interface MockPlayer {
  player_code: number;
  web_name: string;
  first_name: string;
  second_name: string;
  position: Position;
  club: Club;
  price: number;
  base: number;
  q0: number;
}

export const PLAYERS: MockPlayer[] = CATALOG.map((row, i) => {
  const [web, first, second, pos, team, price, base, q0, code] = row;
  return {
    player_code: code ?? 150000 + i * 211,
    web_name: web,
    first_name: first,
    second_name: second,
    position: pos,
    club: clubByShort.get(team)!,
    price,
    base,
    q0,
  };
});

const byName = new Map(PLAYERS.map((p) => [p.web_name, p]));
const byCode = new Map(PLAYERS.map((p) => [p.player_code, p]));

/** Named lookup that throws loudly on catalog typos. */
function P(web: string): MockPlayer {
  const p = byName.get(web);
  if (!p) throw new Error(`mock catalog: unknown player ${web}`);
  return p;
}

// STATUS.md walk-forward eval: real GW34 2025-26 top-10 xP — anchored exactly.
const GW34_ANCHORS: Record<string, number> = {
  "M.Salah": 6.53,
  "B.Fernandes": 6.4,
  Rice: 5.72,
  Solanke: 5.67,
  Gabriel: 5.65,
  Cunha: 5.26,
  Rogers: 5.24,
  Gakpo: 5.24,
  Szoboszlai: 4.97,
  Xavi: 4.96,
};

// ---------------------------------------------------------------------------
// xP generation (per-fixture, then per-GW aggregation)
// ---------------------------------------------------------------------------

/** Component fractions of non-appearance xP per position (sum to 1, incl. negatives). */
const SPLITS: Record<Position, Record<string, number>> = {
  GKP: { goals: 0, assists: 0.02, cs: 0.46, saves: 0.36, defcon: 0.08, bonus: 0.1, cards: -0.02, concede: -0.1, other: 0.1 },
  DEF: { goals: 0.14, assists: 0.1, cs: 0.44, saves: 0, defcon: 0.22, bonus: 0.12, cards: -0.03, concede: -0.09, other: 0.1 },
  MID: { goals: 0.4, assists: 0.28, cs: 0.08, saves: 0, defcon: 0.08, bonus: 0.12, cards: -0.02, concede: 0, other: 0.06 },
  FWD: { goals: 0.58, assists: 0.18, cs: 0, saves: 0, defcon: 0.02, bonus: 0.16, cards: -0.02, concede: 0, other: 0.08 },
};

interface RawFixtureXp {
  fixture: MockFixture;
  opponent: Club;
  was_home: boolean;
  xp: number;
}

/** Per-fixture xP for one player and GW (before anchoring). */
function rawFixtureXp(p: MockPlayer, gw: number): RawFixtureXp[] {
  return fixturesFor(p.club.team_code, gw).map((f, i) => {
    const was_home = f.home.team_code === p.club.team_code;
    const opponent = was_home ? f.away : f.home;
    const jitter = 0.92 + 0.16 * rand(p.player_code, gw, i);
    const avail = 1 - p.q0 * 0.85;
    return { fixture: f, opponent, was_home, xp: p.base * fixtureMult(opponent, was_home) * jitter * avail };
  });
}

function anchorScale(p: MockPlayer, gw: number, rows: RawFixtureXp[]): number {
  const anchor = gw === 34 ? GW34_ANCHORS[p.web_name] : undefined;
  if (anchor === undefined) return 1;
  const total = rows.reduce((s, r) => s + r.xp, 0);
  return total > 0 ? anchor / total : 1;
}

function componentsOf(p: MockPlayer, xp: number, q0: number) {
  const q1 = (1 - q0) * 0.12;
  const q2 = (1 - q0) * 0.88;
  const appearance = q1 + 2 * q2;
  const attack = Math.max(xp - appearance, 0.12);
  const s = SPLITS[p.position];
  return {
    xp_appearance: round2(appearance),
    xp_goals: round2(attack * s.goals),
    xp_assists: round2(attack * s.assists),
    xp_cs: round2(attack * s.cs),
    xp_concede: round2(attack * s.concede),
    xp_saves: round2(attack * s.saves),
    xp_defcon: round2(attack * s.defcon),
    xp_bonus: round2(attack * s.bonus),
    xp_cards: round2(attack * s.cards),
    xp_other: round2(attack * s.other),
    q1: round2(q1),
    q2: round2(q2),
  };
}

/** Per-GW xP for one player (fixture sum, anchored). 0 rows on blank GWs. */
function gwXp(p: MockPlayer, gw: number): number {
  const rows = rawFixtureXp(p, gw);
  const scale = anchorScale(p, gw, rows);
  return round2(rows.reduce((s, r) => s + r.xp, 0) * scale);
}

function gwPredictionRow(p: MockPlayer, gw: number): GwPrediction | null {
  const rows = rawFixtureXp(p, gw);
  if (rows.length === 0) return null;
  const scale = anchorScale(p, gw, rows);
  const xp = round2(rows.reduce((s, r) => s + r.xp, 0) * scale);
  const q0 = round2(Math.min(0.97, p.q0 * (0.9 + 0.2 * rand(p.player_code, gw, 9))));
  const c = componentsOf(p, xp, q0);
  return {
    season: DEMO_SEASON,
    gw,
    player_code: p.player_code,
    n_fixtures: rows.length,
    xp,
    xp_appearance: c.xp_appearance,
    xp_goals: c.xp_goals,
    xp_assists: c.xp_assists,
    xp_cs: c.xp_cs,
    xp_concede: c.xp_concede,
    xp_saves: c.xp_saves,
    xp_defcon: c.xp_defcon,
    xp_bonus: c.xp_bonus,
    xp_cards: c.xp_cards,
    xp_other: c.xp_other,
    q0,
    team_code: p.club.team_code,
    position: p.position,
    price: p.price,
    web_name: p.web_name,
  };
}

let _predictions: PredictionsResponse | null = null;

export function predictionsResponse(): PredictionsResponse {
  if (_predictions) return _predictions;
  const rows: GwPrediction[] = [];
  for (const gw of DEMO_GWS) {
    for (const p of PLAYERS) {
      const row = gwPredictionRow(p, gw);
      if (row) rows.push(row);
    }
  }
  _predictions = { season: DEMO_SEASON, gws: DEMO_GWS, as_of: AS_OF, teams: TEAMS, rows };
  return _predictions;
}

// ---------------------------------------------------------------------------
// player detail: identity + season history + upcoming per-fixture breakdown
// ---------------------------------------------------------------------------

function historyFor(p: MockPlayer): PlayerGwHistory[] {
  const out: PlayerGwHistory[] = [];
  const returnPts = p.position === "FWD" ? 4 : p.position === "MID" ? 5 : 6;
  for (let gw = 1; gw <= 33; gw++) {
    const r = (k: number) => rand(p.player_code, gw, 100 + k);
    const missed = r(0) < p.q0 * 0.85;
    const cameo = !missed && r(1) < 0.12;
    const minutes = missed ? 0 : cameo ? Math.round(8 + r(2) * 40) : Math.round(62 + r(2) * 28);
    let points = 0;
    let goals = 0;
    let assists = 0;
    let cs = 0;
    let bonus = 0;
    if (minutes > 0) {
      points = minutes >= 60 ? 2 : 1;
      const pReturn = Math.min(0.62, Math.max(0.08, (p.base - 2.1) / 4.2));
      if (r(3) < pReturn * (minutes >= 60 ? 1 : 0.35)) {
        goals = r(4) < (p.position === "FWD" ? 0.62 : 0.45) ? (r(5) < 0.18 ? 2 : 1) : 0;
        assists = goals === 0 ? 1 : r(6) < 0.15 ? 1 : 0;
        points += goals * returnPts + assists * 3;
        if (r(7) < 0.42) {
          bonus = 1 + Math.floor(r(8) * 3);
          points += bonus;
        }
      }
      if ((p.position === "DEF" || p.position === "GKP") && minutes >= 60 && r(9) < 0.34) {
        cs = 1;
        points += 4;
      }
      if (p.position === "GKP") points += Math.floor(r(10) * 3); // save points
    }
    const xg =
      p.position === "GKP"
        ? 0
        : round2(Math.max(0, (p.base - 2) * 0.16 * (minutes / 90) * (0.4 + 1.2 * r(11))));
    const xa = p.position === "GKP" ? 0 : round2(Math.max(0, xg * 0.6 * r(12) + 0.03));
    out.push({
      season: DEMO_SEASON,
      gw,
      n_fixtures: 1,
      minutes,
      total_points: points,
      goals_scored: goals,
      assists,
      clean_sheets: cs,
      bonus,
      xg,
      xa,
      value: p.price - Math.round((1 - gw / 38) * (2 + 4 * rand(p.player_code, 0, 500))),
    });
  }
  return out;
}

function upcomingFor(p: MockPlayer): FixturePrediction[] {
  const out: FixturePrediction[] = [];
  for (const gw of DEMO_GWS) {
    const rows = rawFixtureXp(p, gw);
    const scale = anchorScale(p, gw, rows);
    const q0gw = Math.min(0.97, p.q0 * (0.9 + 0.2 * rand(p.player_code, gw, 9)));
    for (const r of rows) {
      const xp = round2(r.xp * scale);
      const q0 = round2(Math.min(0.97, 1 - (1 - q0gw) ** (1 / rows.length)));
      const c = componentsOf(p, xp, q0);
      out.push({
        season: DEMO_SEASON,
        gw,
        player_code: p.player_code,
        fpl_fixture_id: r.fixture.fpl_fixture_id,
        xp,
        xp_appearance: c.xp_appearance,
        xp_goals: c.xp_goals,
        xp_assists: c.xp_assists,
        xp_cs: c.xp_cs,
        xp_concede: c.xp_concede,
        xp_saves: c.xp_saves,
        xp_defcon: c.xp_defcon,
        xp_bonus: c.xp_bonus,
        xp_cards: c.xp_cards,
        xp_other: c.xp_other,
        team_code: p.club.team_code,
        opponent_code: r.opponent.team_code,
        was_home: r.was_home,
        position: p.position,
        price: p.price,
        q0,
        q1: c.q1,
        q2: c.q2,
        mu1: 28.0,
        mu2: 84.0,
        kickoff_utc: r.fixture.kickoff_utc,
      });
    }
  }
  return out;
}

export function playerDetail(code: number): PlayerDetail | null {
  const p = byCode.get(code);
  if (!p) return null;
  return {
    player_code: p.player_code,
    web_name: p.web_name,
    first_name: p.first_name,
    second_name: p.second_name,
    position: p.position,
    team_code: p.club.team_code,
    team_name: p.club.name,
    team_short_name: p.club.short_name,
    price: p.price,
    season: DEMO_SEASON,
    history: historyFor(p),
    upcoming: upcomingFor(p),
  };
}

// ---------------------------------------------------------------------------
// dream team: greedy None-state solve stand-in (valid squad rules, <= £100m)
// ---------------------------------------------------------------------------

const FORMATIONS: [number, number, number][] = [
  [3, 4, 3], [3, 5, 2], [4, 4, 2], [4, 3, 3], [4, 5, 1], [5, 4, 1], [5, 3, 2],
];

interface Solved {
  squad: MockPlayer[];
  lineup: MockPlayer[];
  bench: MockPlayer[];
  xpOf: Map<number, number>;
}

/**
 * Greedy stand-in for the None-state MILP: for each formation, buy the
 * quota-completing bench (cheapest 15-minus-XI per position, so the squad is
 * always exactly 2 GKP / 5 DEF / 5 MID / 3 FWD), then fill the XI greedily by
 * xP under the remaining budget with club-limit and fill-feasibility checks.
 */
function solveDreamSquad(gw: number): Solved {
  const xpOf = new Map(PLAYERS.map((p) => [p.player_code, gwXp(p, gw)]));
  const sorted = [...PLAYERS].sort(
    (a, b) => xpOf.get(b.player_code)! - xpOf.get(a.player_code)!,
  );
  const cheapestN = (pos: Position, n: number, exclude: Set<number>) =>
    PLAYERS.filter((p) => p.position === pos && !exclude.has(p.player_code))
      .sort((a, b) => a.price - b.price)
      .slice(0, n);

  let best: Solved & { total: number } | null = null;
  for (const [d, m, f] of FORMATIONS) {
    // bench completes the 2/5/5/3 squad quota for this formation
    const picked = new Set<number>();
    const bench: MockPlayer[] = [];
    for (const [pos, n] of [
      ["GKP", 1],
      ["DEF", 5 - d],
      ["MID", 5 - m],
      ["FWD", 3 - f],
    ] as [Position, number][]) {
      for (const b of cheapestN(pos, n, picked)) {
        bench.push(b);
        picked.add(b.player_code);
      }
    }
    const benchCost = bench.reduce((s, p) => s + p.price, 0);

    const need: Record<Position, number> = { GKP: 1, DEF: d, MID: m, FWD: f };
    const clubCount = new Map<number, number>();
    for (const b of bench) {
      clubCount.set(b.club.team_code, (clubCount.get(b.club.team_code) ?? 0) + 1);
    }
    let budget = 1000 - benchCost;
    const xi: MockPlayer[] = [];
    const taken = new Set(picked);
    // Reserve a mid-tier floor per unfilled XI slot so early premium buys don't
    // force pure fodder into the XI (a crude proxy for the MILP's balance).
    const RESERVE_FLOOR = 48;
    const minPrice = (pos: Position, excl: Set<number>) =>
      Math.max(
        RESERVE_FLOOR,
        Math.min(
          ...PLAYERS.filter((p) => p.position === pos && !excl.has(p.player_code)).map(
            (p) => p.price,
          ),
        ),
      );
    for (const p of sorted) {
      if (xi.length === 11) break;
      if (taken.has(p.player_code) || need[p.position] === 0) continue;
      if ((clubCount.get(p.club.team_code) ?? 0) >= 3) continue;
      // feasibility: after buying p, the remaining slots must be fillable at the floor
      const exclNext = new Set(taken);
      exclNext.add(p.player_code);
      let reserve = 0;
      for (const pos of ["GKP", "DEF", "MID", "FWD"] as Position[]) {
        const remaining = need[pos] - (pos === p.position ? 1 : 0);
        if (remaining > 0) reserve += remaining * minPrice(pos, exclNext);
      }
      if (p.price + reserve > budget) continue;
      xi.push(p);
      taken.add(p.player_code);
      need[p.position] -= 1;
      clubCount.set(p.club.team_code, (clubCount.get(p.club.team_code) ?? 0) + 1);
      budget -= p.price;
    }
    if (xi.length === 11) {
      const total = xi.reduce((s, p) => s + xpOf.get(p.player_code)!, 0);
      if (!best || total > best.total) {
        best = { squad: [...xi, ...bench], lineup: xi, bench, xpOf, total };
      }
    }
  }
  if (!best) throw new Error("mock dream team: no feasible XI");
  return { squad: best.squad, lineup: best.lineup, bench: best.bench, xpOf };
}

function formationOf(lineup: MockPlayer[]): string {
  const n = (pos: Position) => lineup.filter((p) => p.position === pos).length;
  return `${n("DEF")}-${n("MID")}-${n("FWD")}`;
}

let _dream: DreamTeam | null = null;

export function dreamTeamResponse(): DreamTeam {
  if (_dream) return _dream;
  const { squad, lineup, bench, xpOf } = solveDreamSquad(DEMO_GW);
  const ranked = [...lineup].sort((a, b) => xpOf.get(b.player_code)! - xpOf.get(a.player_code)!);
  const captain = ranked[0];
  const vice = ranked[1];
  const expected = round2(
    lineup.reduce((s, p) => s + xpOf.get(p.player_code)!, 0) + xpOf.get(captain.player_code)!,
  );
  _dream = {
    season: DEMO_SEASON,
    gw: DEMO_GW,
    squad: squad.map((p) => p.player_code),
    lineup: lineup.map((p) => p.player_code),
    bench_order: bench.map((p) => p.player_code),
    captain: captain.player_code,
    vice: vice.player_code,
    formation: formationOf(lineup),
    expected_points: expected,
    objective: expected,
    total_cost: squad.reduce((s, p) => s + p.price, 0),
    solve_seconds: 1.84,
  };
  return _dream;
}

// ---------------------------------------------------------------------------
// my team: squad state + 5-GW plan + weekly verdict
// ---------------------------------------------------------------------------

const OWNED = [
  "Raya", "Dúbravka",
  "Gabriel", "Van Dijk", "Muñoz", "Aina", "Konsa",
  "M.Salah", "B.Fernandes", "Rice", "Rogers", "Gakpo",
  "Haaland", "Evanilson", "Watkins",
] as const;

/** Lineup after the recommended Evanilson -> Solanke move (3-5-2). */
const LINEUP_34 = [
  "Raya",
  "Gabriel", "Van Dijk", "Muñoz",
  "M.Salah", "B.Fernandes", "Rice", "Rogers", "Gakpo",
  "Haaland", "Solanke",
] as const;
const BENCH_34 = ["Dúbravka", "Watkins", "Aina", "Konsa"] as const;

function squadState(): SquadState {
  return {
    season: DEMO_SEASON,
    current_gw: DEMO_GW,
    squad: OWNED.map((w) => {
      const p = P(w);
      const drift = Math.round(rand(p.player_code, 3, 77) * 4) - 1;
      return {
        player_code: p.player_code,
        purchase_price: Math.max(35, p.price - Math.max(0, drift)),
        current_price: p.price,
      };
    }),
    bank: 15,
    free_transfers: 2,
    chips_available: ["wc2", "bb2", "tc2"],
    active_chip: null,
  };
}

function lineupXp(names: readonly string[], gw: number): number {
  return round2(names.reduce((s, w) => s + gwXp(P(w), gw), 0));
}

function buildGwPlan(
  gw: number,
  lineup: readonly string[],
  bench: readonly string[],
  opts: {
    captain: string;
    vice: string;
    transfers_in?: string[];
    transfers_out?: string[];
    chip?: string | null;
    bank: number;
    free_transfers: number;
  },
): GwPlan {
  const capXp = gwXp(P(opts.captain), gw);
  const benchXp = opts.chip === "bb2" ? lineupXp(bench, gw) : 0;
  return {
    gw,
    squad: [...lineup, ...bench].map((w) => P(w).player_code),
    lineup: lineup.map((w) => P(w).player_code),
    bench_order: opts.chip === "bb2" ? [] : bench.map((w) => P(w).player_code),
    captain: P(opts.captain).player_code,
    vice: P(opts.vice).player_code,
    transfers_in: (opts.transfers_in ?? []).map((w) => P(w).player_code),
    transfers_out: (opts.transfers_out ?? []).map((w) => P(w).player_code),
    hit_points: 0,
    chip: opts.chip ?? null,
    expected_points: round2(lineupXp(lineup, gw) + capXp + benchXp),
    bank: opts.bank,
    free_transfers: opts.free_transfers,
  };
}

let _plan: PlanResult | null = null;

function planResult(): PlanResult {
  if (_plan) return _plan;
  const l34 = LINEUP_34;
  const b34 = BENCH_34;
  // GW36: Rogers -> Semenyo (BOU home run-in), like-for-like price.
  const l36 = l34.map((w) => (w === "Rogers" ? "Semenyo" : w));
  const gws: GwPlan[] = [
    buildGwPlan(34, l34, b34, {
      captain: "M.Salah", vice: "B.Fernandes",
      transfers_in: ["Solanke"], transfers_out: ["Evanilson"],
      bank: 8, free_transfers: 2,
    }),
    buildGwPlan(35, l34, b34, {
      captain: "Haaland", vice: "M.Salah", bank: 8, free_transfers: 2,
    }),
    buildGwPlan(36, l36, b34, {
      captain: "M.Salah", vice: "Haaland",
      transfers_in: ["Semenyo"], transfers_out: ["Rogers"],
      bank: 8, free_transfers: 3,
    }),
    buildGwPlan(37, l36, b34, {
      captain: "M.Salah", vice: "Haaland", chip: "bb2", bank: 8, free_transfers: 3,
    }),
    buildGwPlan(38, l36, b34, {
      captain: "M.Salah", vice: "Haaland", bank: 8, free_transfers: 4,
    }),
  ];
  _plan = {
    objective: round2(gws.reduce((s, g) => s + g.expected_points, 0)),
    gws,
    solve_seconds: 12.41,
    gap: 0.0,
    status: "Optimal",
    n_variables: 18240,
    n_constraints: 9612,
  };
  return _plan;
}

function horizonXp(w: string): number {
  return round2(DEMO_GWS.reduce((s, gw) => s + gwXp(P(w), gw), 0));
}

const STABILITY: StabilityEntry[] = [
  { move: `captain:${P("M.Salah").player_code}`, support_pct: 96.7 },
  { move: `chip:none`, support_pct: 93.3 },
  { move: `sell:${P("Evanilson").player_code}`, support_pct: 90.0 },
  { move: `buy:${P("Solanke").player_code}`, support_pct: 86.7 },
  { move: `buy:${P("Mateta").player_code}`, support_pct: 10.0 },
];

function chipAdviceList(): ChipAdvice[] {
  return [
    { chip: "wc2", verdict: "hold", planned_gw: null, ev_now: 2.2, best_gw: 35, best_ev: 3.1 },
    { chip: "bb2", verdict: "hold", planned_gw: 37, ev_now: 2.1, best_gw: 37, best_ev: 8.2 },
    { chip: "tc2", verdict: "hold", planned_gw: null, ev_now: 3.1, best_gw: 37, best_ev: 4.4 },
  ];
}

let _reco: Recommendation | null = null;

/** The my-team weekly verdict (entry configured). */
export function myTeamRecommendation(): Recommendation {
  if (_reco) return _reco;
  const plan = planResult();
  const g34 = plan.gws[0];
  const dream = dreamTeamResponse();
  const inXp = horizonXp("Solanke");
  const outXp = horizonXp("Evanilson");
  const delta = round2(inXp - outXp);
  const capXp = gwXp(P("M.Salah"), 34);
  const gap = round2(g34.expected_points - dream.expected_points);
  const in3 = round2(
    DEMO_GWS.slice(0, 3).reduce((s, gw) => s + gwXp(P("Solanke"), gw), 0) / 3,
  );
  _reco = {
    season: DEMO_SEASON,
    gw: DEMO_GW,
    as_of: AS_OF,
    action: "transfer",
    transfers: [
      {
        player_in: P("Solanke").player_code,
        player_in_name: "Solanke",
        player_out: P("Evanilson").player_code,
        player_out_name: "Evanilson",
        position: "FWD",
        xp_in: inXp,
        xp_out: outXp,
        xp_delta: delta,
        out_q0: 0.55,
        support_pct: 86.7,
      },
    ],
    hits: 0,
    captain: P("M.Salah").player_code,
    vice: P("B.Fernandes").player_code,
    lineup: g34.lineup,
    bench_order: g34.bench_order,
    squad: g34.squad,
    formation: "3-5-2",
    expected_points: g34.expected_points,
    objective: plan.objective,
    chip_advice: chipAdviceList(),
    dream_team: dream,
    plan,
    stability: STABILITY,
    rationale: [
      `Bring in Solanke (${in3.toFixed(1)} xP/GW next 3) for Evanilson (q0 55%, availability risk): +${delta.toFixed(1)} xP over the horizon, 87% of noise re-solves agree.`,
      `Captain M.Salah (${capXp.toFixed(1)} xP this GW); vice B.Fernandes (captain q0 4%).`,
      `Best future chip window: Bench Boost (bb2) in GW37 (+8.2 xP vs no chip) — hold for now.`,
      `Fresh-£100m benchmark: ${dream.expected_points.toFixed(1)} xP this GW vs your ${g34.expected_points.toFixed(1)} (${gap >= 0 ? "+" : ""}${gap.toFixed(1)}).`,
      `Stability: 88% of noise re-solves agree with this GW's moves.`,
    ],
  };
  return _reco;
}

let _initial: Recommendation | null = null;

/** The no-entry pre-season product: the initial-squad build (state=None solve). */
export function initialSquadRecommendation(): Recommendation {
  if (_initial) return _initial;
  const dream = dreamTeamResponse();
  const { lineup, xpOf } = solveDreamSquad(DEMO_GW);
  const capName = byCode.get(dream.captain!)!.web_name;
  const spent = dream.total_cost;
  const bank = 1000 - spent;
  const gwsPlan: GwPlan[] = DEMO_GWS.map((gw, i) => ({
    gw,
    squad: dream.squad,
    lineup: dream.lineup,
    bench_order: dream.bench_order,
    captain: dream.captain!,
    vice: dream.vice!,
    transfers_in: [],
    transfers_out: [],
    hit_points: 0,
    chip: null,
    expected_points:
      i === 0
        ? dream.expected_points
        : round2(
            lineup.reduce((s, p) => s + gwXp(p, gw), 0) +
              gwXp(byCode.get(dream.captain!)!, gw),
          ),
    bank,
    free_transfers: i === 0 ? null : 1 + i,
  }));
  _initial = {
    season: DEMO_SEASON,
    gw: DEMO_GW,
    as_of: AS_OF,
    action: "initial-squad",
    transfers: [],
    hits: 0,
    captain: dream.captain,
    vice: dream.vice,
    lineup: dream.lineup,
    bench_order: dream.bench_order,
    squad: dream.squad,
    formation: dream.formation,
    expected_points: dream.expected_points,
    objective: round2(gwsPlan.reduce((s, g) => s + g.expected_points, 0)),
    chip_advice: chipAdviceList().map((a) => ({ ...a, planned_gw: null })),
    dream_team: dream,
    plan: {
      objective: round2(gwsPlan.reduce((s, g) => s + g.expected_points, 0)),
      gws: gwsPlan,
      solve_seconds: 8.02,
      gap: 0.0,
      status: "Optimal",
      n_variables: 15110,
      n_constraints: 8244,
    },
    stability: [],
    rationale: [
      `Initial squad for GW${DEMO_GW}: 15 players, £${(spent / 10).toFixed(1)}m spent, £${(bank / 10).toFixed(1)}m in the bank; expected ${dream.expected_points.toFixed(1)} xP in GW${DEMO_GW}.`,
      `Captain ${capName} (${xpOf.get(dream.captain!)!.toFixed(1)} xP this GW); vice ${byCode.get(dream.vice!)!.web_name}.`,
      `Best future chip window: Bench Boost (bb2) in GW37 (+8.2 xP vs no chip) — hold for now.`,
      `Demo mode: 2025-26 GW${DEMO_GW} backtest data — the 2026-27 game has not launched yet.`,
    ],
  };
  return _initial;
}

export function myTeamResponse(entryId: number): MyTeamResponse {
  return { entry_id: entryId, state: squadState(), recommendation: myTeamRecommendation() };
}

// ---------------------------------------------------------------------------
// chip EV curves (chips.chip_ev_curves rows)
// ---------------------------------------------------------------------------

const CHIP_DELTAS: Record<string, number[]> = {
  wc2: [2.2, 3.1, 2.4, 2.9, 1.8],
  bb2: [2.1, 2.8, 3.4, 8.2, 4.6],
  tc2: [3.1, 2.4, 2.9, 4.4, 3.3],
};

export function chipCurvesResponse(entryId: number | null): ChipCurvesResponse {
  const baseline = planResult().objective;
  const curves: ChipCurvePoint[] = [];
  for (const [chip, deltas] of Object.entries(CHIP_DELTAS)) {
    DEMO_GWS.forEach((gw, i) => {
      curves.push({
        chip,
        gw,
        objective: round2(baseline + deltas[i]),
        delta_vs_no_chip: deltas[i],
        evaluated: true,
        skip_reason: null,
        window_last_gw: 38,
        urgency: round2((i + 1) / DEMO_GWS.length),
      });
    });
  }
  return { season: DEMO_SEASON, entry_id: entryId, curves };
}

// ---------------------------------------------------------------------------
// state
// ---------------------------------------------------------------------------

export function stateResponse(): StateResponse {
  return {
    season: 2025,
    is_live_2026_27: false,
    current_gw: null,
    next_gw: null,
    next_deadline_utc: null,
    total_players: 11_412_508,
    data_freshness: {
      snapshot_utc: "2026-07-22T06:00:12Z",
      processed_utc: "2026-07-22T06:14:47Z",
      predictions_utc: AS_OF,
      predictions_season: DEMO_SEASON,
      predictions_gws: DEMO_GWS,
    },
    model: {
      schema: 1,
      created_utc: "2026-07-22T05:58:41Z",
      train_window: {
        seasons: [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025],
        before_season: 2025,
        before_gw: 34,
        n_feature_rows: 253_568,
        n_team_matches: 3_800,
      },
      components: {
        minutes: { class: "LgbMinutesModel", version: "v1" },
        team: { class: "TeamModel", version: "v1" },
        rates: { class: "RatesModel", version: "v1" },
        bonus: { class: "BonusCalibration", version: "v1" },
      },
    },
  };
}
