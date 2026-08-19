/**
 * Resolve scan-proxy-recognized squad cards onto player codes — the pure-TS port
 * of backend fplai/scan.py for the static build, where matching runs client-side
 * against the player bundle the page already loaded.
 *
 * Matching is fuzzy by necessity — screenshots print FPL web_name ("M.Salah",
 * "Van Dijk"), sometimes with OCR wobble ("Odegaard" for "Ødegaard") — so each
 * card/pool pair is scored on normalized-name similarity with small nudges from
 * the position row (reliable in screenshots), the printed price and the shirt
 * club. Codes are assigned greedily best-score-first, one card per code, and a
 * pair below ACCEPT_SCORE stays unmatched rather than guessing.
 *
 * One deliberate difference from the Python matcher: the public bundle carries
 * no first/second names, so similarity runs on web_name alone — which is what
 * FPL screenshots print anyway.
 */

/** One player card as read off the screenshot by the scan proxy (pre-matching). */
export interface SeenCard {
  name: string;
  club: string | null;
  /** £m as printed, e.g. 12.7. */
  price: number | null;
  position: string | null;
}

/** The pool fields matching needs — satisfied by the predictions view models. */
export interface ScanPoolPlayer {
  player_code: number;
  web_name: string;
  position: string;
  /** 0.1m units, e.g. 127 = £12.7m. */
  price: number;
  team_short: string | null;
}

export interface CardMatch {
  seen: SeenCard;
  player_code: number | null;
  web_name: string | null;
  team_short: string | null;
  score: number;
}

/** A pair must reach this score to claim a player code. */
export const ACCEPT_SCORE = 0.72;
/** Name similarity below this never enters the candidate pool. */
export const MIN_NAME_SCORE = 0.55;

/** Characters NFKD won't decompose to ASCII — mapped by hand before stripping marks. */
const CHAR_MAP: Record<string, string> = {
  ø: "o",
  Ø: "O",
  đ: "d",
  Đ: "D",
  ł: "l",
  Ł: "L",
  ß: "ss",
  æ: "ae",
  Æ: "AE",
  œ: "oe",
  Œ: "OE",
};

/** Lowercase ASCII words: strip accents, drop punctuation, collapse spaces. */
export function normalizeName(name: string): string {
  const mapped = [...name].map((c) => CHAR_MAP[c] ?? c).join("");
  const stripped = mapped.normalize("NFKD").replace(/[\u0300-\u036f]/g, "");
  return stripped
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, " ")
    .trim();
}

/** Levenshtein-based similarity in [0, 1] — the TS stand-in for difflib.ratio. */
function ratio(a: string, b: string): number {
  if (a === b) return 1;
  const la = a.length;
  const lb = b.length;
  if (la === 0 || lb === 0) return 0;
  let prev = Array.from({ length: lb + 1 }, (_, j) => j);
  for (let i = 1; i <= la; i++) {
    const cur = [i];
    for (let j = 1; j <= lb; j++) {
      cur[j] = Math.min(
        prev[j] + 1,
        cur[j - 1] + 1,
        prev[j - 1] + (a[i - 1] === b[j - 1] ? 0 : 1),
      );
    }
    prev = cur;
  }
  return 1 - prev[lb] / Math.max(la, lb);
}

function nameScore(seenNorm: string, webNorm: string): number {
  if (!seenNorm) return 0;
  if (seenNorm === webNorm) return 1;
  let best = ratio(seenNorm, webNorm);
  // Whole-token containment ("Salah" ⊂ "m salah") beats raw edit distance.
  if (` ${webNorm} `.includes(` ${seenNorm} `) || ` ${seenNorm} `.includes(` ${webNorm} `)) {
    best = Math.max(best, 0.92);
  }
  return best;
}

function pairScore(seen: SeenCard, name: number, p: ScanPoolPlayer): number {
  let score = name;
  if (seen.position !== null) {
    score += seen.position === p.position ? 0.08 : -0.25;
  }
  if (seen.price !== null) {
    const diff = Math.abs(p.price / 10 - seen.price);
    if (diff <= 0.2) score += 0.08;
    else if (diff > 1.5) score -= 0.15;
  }
  if (seen.club !== null && p.team_short !== null) {
    score += seen.club.trim().toUpperCase() === p.team_short.toUpperCase() ? 0.08 : -0.05;
  }
  return score;
}

/** Assign each seen card the best-scoring unclaimed pool code (greedy, unique). */
export function matchSquad(seen: SeenCard[], pool: ScanPoolPlayer[]): CardMatch[] {
  const rows = pool.map((p) => ({ p, webNorm: normalizeName(p.web_name) }));

  const candidates: { score: number; i: number; j: number }[] = [];
  seen.forEach((card, i) => {
    const seenNorm = normalizeName(card.name);
    rows.forEach((row, j) => {
      const name = nameScore(seenNorm, row.webNorm);
      if (name < MIN_NAME_SCORE) return;
      const score = pairScore(card, name, row.p);
      if (score >= ACCEPT_SCORE) candidates.push({ score, i, j });
    });
  });

  const assigned = new Map<number, { score: number; p: ScanPoolPlayer }>();
  const usedCodes = new Set<number>();
  for (const { score, i, j } of candidates.sort((a, b) => b.score - a.score)) {
    const { p } = rows[j];
    if (assigned.has(i) || usedCodes.has(p.player_code)) continue;
    assigned.set(i, { score, p });
    usedCodes.add(p.player_code);
  }

  return seen.map((card, i) => {
    const hit = assigned.get(i);
    if (!hit) {
      return { seen: card, player_code: null, web_name: null, team_short: null, score: 0 };
    }
    return {
      seen: card,
      player_code: hit.p.player_code,
      web_name: hit.p.web_name,
      team_short: hit.p.team_short,
      score: Math.round(Math.min(hit.score, 1) * 1000) / 1000,
    };
  });
}
