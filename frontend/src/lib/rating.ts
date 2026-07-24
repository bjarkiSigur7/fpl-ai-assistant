/**
 * Client-side AI-Rating engine — an EXACT pure-TS port of the Python metric in
 * backend/src/fplai/optimizer/rating.py (rate_team + best_xi + squad_metric +
 * validate_squad). Static (public) builds score squads in the browser from the
 * published bundle (players.json + xp.json + rating.json) with this engine;
 * local builds keep POST /api/rate-team.
 *
 *     metric(squad) = sum over the window's GWs of (best-legal-XI xP sum + captain bonus)
 *     score = clamp01((metric(team) - floor) / (optimal - floor)) * 100
 *
 * The best XI for a GW is chosen greedily — formation minimums (1 GKP/3 DEF/
 * 2 MID/1 FWD) by xP first, then filled to 11 by xP respecting the maxes
 * (1 GKP/5 DEF/5 MID/3 FWD) — and the captain bonus is the highest xP within
 * that XI. `floor`/`optimal` anchors arrive precomputed in rating.json (same
 * metric, evaluated by the Python engine over the same window). Every tie
 * breaks by player_code, mirroring the Python engine bit for bit.
 *
 * This module is dependency-free and framework-free on purpose: `node --test`
 * runs the parity suite (tests/rating.parity.test.ts) against fixtures exported
 * from the real Python engine.
 */

// --------------------------------------------------------------------------------------
// Constants — mirrors of fplai.rules.POSITIONS + fplai.optimizer.milp squad constants
// --------------------------------------------------------------------------------------

export const RATING_POSITIONS = ["GKP", "DEF", "MID", "FWD"] as const;
export type RatingPosition = (typeof RATING_POSITIONS)[number];

export const SQUAD_SIZE = 15;
export const XI_SIZE = 11;
export const POSITION_QUOTA: Record<RatingPosition, number> = { GKP: 2, DEF: 5, MID: 5, FWD: 3 };
export const FORMATION_MIN: Record<RatingPosition, number> = { GKP: 1, DEF: 3, MID: 2, FWD: 1 };
export const FORMATION_MAX: Record<RatingPosition, number> = { GKP: 1, DEF: 5, MID: 5, FWD: 3 };
export const CLUB_LIMIT = 3;
export const INITIAL_BUDGET = 1000; // £100.0m in £0.1m units

/** Score bands, checked in order: first threshold the score reaches wins. */
export const VERDICT_BANDS: readonly (readonly [number, RatingVerdict])[] = [
  [95.0, "ELITE"],
  [85.0, "STRONG"],
  [70.0, "SOLID"],
  [50.0, "ROUGH"],
];

export type RatingVerdict = "ELITE" | "STRONG" | "SOLID" | "ROUGH" | "FODDER";

/** Thrown by rateTeam on an illegal squad; `problems` lists EVERY violated rule. */
export class RatingValidationError extends Error {
  readonly problems: string[];

  constructor(problems: string[]) {
    super(problems.join("; "));
    this.name = "RatingValidationError";
    this.problems = problems;
  }
}

// --------------------------------------------------------------------------------------
// Inputs — the bundle shapes the engine consumes (players.json / xp.json / rating.json)
// --------------------------------------------------------------------------------------

/** One players.json row (the legality pool). */
export interface RatingPoolPlayer {
  player_code: number;
  web_name: string;
  position: string;
  team_short?: string | null;
  team_code: number;
  price: number; // £0.1m units
  q0_gw1?: number | null;
}

/** xp.json: compact per-GW horizon table; players["<code>"][i] pairs with gws[i]. */
export interface RatingXpTable {
  gws: number[];
  players: Record<string, number[]>;
}

/** rating.json: the precomputed anchors + window (same Python metric). */
export interface RatingAnchors {
  floor_metric: number;
  optimal_metric: number;
  window_gws: number[];
  captain_bonus?: boolean;
}

export interface RatingInputs {
  season: number;
  pool: RatingPoolPlayer[];
  xp: RatingXpTable;
  anchors: RatingAnchors;
}

// --------------------------------------------------------------------------------------
// Result — structurally identical to the POST /api/rate-team 200 body
// --------------------------------------------------------------------------------------

export interface RatingPlayerRow {
  player_code: number;
  web_name: string;
  position: string;
  team_short: string | null;
  price: number;
  xp_gw1: number;
  xp_horizon: number;
  q0: number | null;
  in_best_xi_gw1: boolean;
}

export interface RatingWeakestRow {
  player_code: number;
  web_name: string;
  xp_horizon: number;
}

export interface RatingScore {
  score: number;
  verdict: RatingVerdict;
  season: number;
  from_gw: number;
  horizon: number;
  team_xp_gw1: number;
  team_xp_horizon: number;
  optimal_xp_horizon: number;
  floor_xp_horizon: number;
  best_xi_gw1: number[];
  formation_gw1: string;
  suggested_captain: number;
  player_ratings: RatingPlayerRow[];
  weakest: RatingWeakestRow[];
}

export interface XiSelection {
  xi: number[];
  formation: string;
  xi_xp: number;
  captain: number;
  captain_xp: number;
}

// --------------------------------------------------------------------------------------
// Squad legality — validate_squad port (message strings verbatim)
// --------------------------------------------------------------------------------------

/** Every squad-legality rule the submitted codes violate (empty = legal). */
export function validateSquad(playerCodes: number[], pool: RatingPoolPlayer[]): string[] {
  const problems: string[] = [];
  const codes = playerCodes.map((c) => Math.trunc(c));
  if (codes.length !== SQUAD_SIZE) {
    problems.push(`need exactly ${SQUAD_SIZE} players, got ${codes.length}`);
  }
  const seen = new Map<number, number>();
  for (const code of codes) seen.set(code, (seen.get(code) ?? 0) + 1);
  const dupes = [...seen.entries()].filter(([, n]) => n > 1).map(([c]) => c);
  for (const code of dupes.sort((a, b) => a - b)) {
    problems.push(`duplicate player_code ${code}`);
  }

  const info = new Map<number, { position: string; team_code: number; price: number }>();
  for (const p of pool) {
    info.set(p.player_code, {
      position: p.position,
      team_code: p.team_code,
      price: p.price,
    });
  }
  // NB: `known` keeps duplicates, exactly like the Python list comprehension.
  const known = codes.filter((c) => info.has(c));
  const unknown = [...new Set(codes.filter((c) => !info.has(c)))];
  for (const code of unknown.sort((a, b) => a - b)) {
    problems.push(`unknown player_code ${code}`);
  }

  const counts = new Map<string, number>();
  for (const c of known) {
    const pos = info.get(c)!.position;
    counts.set(pos, (counts.get(pos) ?? 0) + 1);
  }
  for (const pos of RATING_POSITIONS) {
    if ((counts.get(pos) ?? 0) !== POSITION_QUOTA[pos]) {
      problems.push(
        `position quota: need ${POSITION_QUOTA[pos]} ${pos}, got ${counts.get(pos) ?? 0}`,
      );
    }
  }
  const clubs = new Map<number, number>();
  for (const c of known) {
    const club = info.get(c)!.team_code;
    clubs.set(club, (clubs.get(club) ?? 0) + 1);
  }
  for (const [club, n] of [...clubs.entries()].sort((a, b) => a[0] - b[0])) {
    if (n > CLUB_LIMIT) {
      problems.push(
        `club limit: max ${CLUB_LIMIT} players per club, got ${n} from team_code ${club}`,
      );
    }
  }
  let totalPrice = 0;
  for (const c of known) totalPrice += info.get(c)!.price;
  if (totalPrice > INITIAL_BUDGET) {
    problems.push(`total price ${totalPrice} exceeds budget ${INITIAL_BUDGET}`);
  }
  return problems;
}

// --------------------------------------------------------------------------------------
// The metric: greedy best XI + captain bonus, summed over the window
// --------------------------------------------------------------------------------------

/**
 * Greedy best legal XI from `[player_code, position, xp]` entries.
 *
 * Formation minimums by xP first, then filled to 11 by xP respecting the maxes.
 * Every tie breaks by player_code; the captain is the XI's highest-xP player
 * (ties -> lowest code). Throws when the entries cannot field a legal XI.
 */
export function bestXi(entries: Iterable<readonly [number, string, number]>): XiSelection {
  const byPos = new Map<string, [number, number][]>(RATING_POSITIONS.map((p) => [p, []]));
  for (const [code, pos, xp] of entries) {
    const group = byPos.get(pos);
    if (group === undefined) throw new Error(`unknown position '${pos}' for player ${code}`);
    group.push([Math.trunc(code), xp]);
  }
  for (const group of byPos.values()) {
    group.sort((a, b) => b[1] - a[1] || a[0] - b[0]); // (-xp, code)
  }

  const chosen: [number, RatingPosition, number][] = [];
  const counts: Record<string, number> = {};
  for (const pos of RATING_POSITIONS) {
    const need = FORMATION_MIN[pos];
    const group = byPos.get(pos)!;
    if (group.length < need) {
      throw new Error(`cannot field a legal XI: need ${need} ${pos}, have ${group.length}`);
    }
    for (const [code, xp] of group.slice(0, need)) chosen.push([code, pos, xp]);
    counts[pos] = need;
  }
  const rest: [number, RatingPosition, number][] = [];
  for (const pos of RATING_POSITIONS) {
    for (const [code, xp] of byPos.get(pos)!.slice(FORMATION_MIN[pos])) {
      rest.push([code, pos, xp]);
    }
  }
  rest.sort((a, b) => b[2] - a[2] || a[0] - b[0]); // (-xp, code)
  for (const [code, pos, xp] of rest) {
    if (chosen.length === XI_SIZE) break;
    if (counts[pos] >= FORMATION_MAX[pos]) continue;
    chosen.push([code, pos, xp]);
    counts[pos] += 1;
  }
  if (chosen.length < XI_SIZE) {
    throw new Error(`cannot field a legal XI: only ${chosen.length} eligible players`);
  }

  let captain = chosen[0][0];
  let captainXp = chosen[0][2];
  for (const [code, , xp] of chosen) {
    // Python max(key=(xp, -code)): higher xp wins; ties -> lower code.
    if (xp > captainXp || (xp === captainXp && code < captain)) {
      captain = code;
      captainXp = xp;
    }
  }
  const order = new Map<string, number>(RATING_POSITIONS.map((p, i) => [p, i]));
  const xi = [...chosen]
    .sort((a, b) => order.get(a[1])! - order.get(b[1])! || b[2] - a[2] || a[0] - b[0])
    .map(([code]) => code);
  let xiXp = 0;
  for (const [, , xp] of chosen) xiXp += xp;
  return {
    xi,
    formation: `${counts.DEF}-${counts.MID}-${counts.FWD}`,
    xi_xp: xiXp,
    captain,
    captain_xp: captainXp,
  };
}

/** `sum over window GWs of (best-XI xP + captain bonus)` for one 15-man squad. */
export function squadMetric(
  codes: number[],
  posOf: Map<number, string>,
  xpOf: (gw: number, code: number) => number,
  windowGws: number[],
): number {
  let total = 0;
  for (const gw of windowGws) {
    const sel = bestXi(codes.map((c) => [c, posOf.get(c)!, xpOf(gw, c)] as const));
    total += sel.xi_xp + sel.captain_xp;
  }
  return total;
}

// --------------------------------------------------------------------------------------
// rateTeam
// --------------------------------------------------------------------------------------

function clamp01(x: number): number {
  return Math.max(0, Math.min(1, x));
}

export function verdictOf(score: number): RatingVerdict {
  for (const [threshold, verdict] of VERDICT_BANDS) {
    if (score >= threshold) return verdict;
  }
  return "FODDER";
}

/**
 * Score a 15-man squad 0-100 between the floor and dream-team anchors — the
 * Python rate_team, fed from the static bundle. Pure and deterministic; players
 * with no xp.json row contribute 0 xP but remain legal picks. Throws
 * RatingValidationError with EVERY violated legality rule for an illegal squad.
 */
export function rateTeam(playerCodes: number[], inputs: RatingInputs): RatingScore {
  const { pool, xp, anchors } = inputs;
  const problems = validateSquad(playerCodes, pool);
  if (problems.length > 0) throw new RatingValidationError(problems);

  const codes = playerCodes.map((c) => Math.trunc(c));
  const windowGws = [...anchors.window_gws].sort((a, b) => a - b);
  const fromGw = windowGws[0];
  const horizon = windowGws.length > 0 ? windowGws[windowGws.length - 1] - fromGw + 1 : 0;

  const gwIndex = new Map<number, number>(xp.gws.map((g, i) => [g, i]));
  const xpOf = (gw: number, code: number): number => {
    const i = gwIndex.get(gw);
    if (i === undefined) return 0;
    return xp.players[String(code)]?.[i] ?? 0;
  };

  const posOf = new Map<number, string>();
  const byCode = new Map<number, RatingPoolPlayer>();
  for (const p of pool) {
    posOf.set(p.player_code, p.position);
    byCode.set(p.player_code, p);
  }

  const teamMetric = squadMetric(codes, posOf, xpOf, windowGws);
  const optimalMetric = anchors.optimal_metric;
  const floorMetric = anchors.floor_metric;

  const denom = optimalMetric - floorMetric;
  let score: number;
  if (denom > 0) {
    score = clamp01((teamMetric - floorMetric) / denom) * 100;
  } else {
    // Degenerate anchors (e.g. empty window): pass/fail split, as in Python.
    score = teamMetric >= optimalMetric ? 100 : 0;
  }

  const selGw1 = bestXi(codes.map((c) => [c, posOf.get(c)!, xpOf(fromGw, c)] as const));
  const xiGw1 = new Set(selGw1.xi);

  const xpHorizonOf = new Map<number, number>();
  for (const c of codes) {
    let sum = 0;
    for (const gw of windowGws) sum += xpOf(gw, c);
    xpHorizonOf.set(c, sum);
  }

  const ratings: RatingPlayerRow[] = codes.map((c) => {
    const p = byCode.get(c)!;
    return {
      player_code: c,
      web_name: p.web_name,
      position: p.position,
      team_short: p.team_short ?? null,
      price: p.price,
      xp_gw1: xpOf(fromGw, c),
      xp_horizon: xpHorizonOf.get(c)!,
      q0: p.q0_gw1 ?? null,
      in_best_xi_gw1: xiGw1.has(c),
    };
  });
  const weakest: RatingWeakestRow[] = [...codes]
    .sort((a, b) => xpHorizonOf.get(a)! - xpHorizonOf.get(b)! || a - b)
    .slice(0, 3)
    .map((c) => ({
      player_code: c,
      web_name: byCode.get(c)!.web_name,
      xp_horizon: xpHorizonOf.get(c)!,
    }));

  return {
    score,
    verdict: verdictOf(score),
    season: inputs.season,
    from_gw: fromGw,
    horizon,
    team_xp_gw1: selGw1.xi_xp + selGw1.captain_xp,
    team_xp_horizon: teamMetric,
    optimal_xp_horizon: optimalMetric,
    floor_xp_horizon: floorMetric,
    best_xi_gw1: selGw1.xi,
    formation_gw1: selGw1.formation,
    suggested_captain: selGw1.captain,
    player_ratings: ratings,
    weakest,
  };
}
