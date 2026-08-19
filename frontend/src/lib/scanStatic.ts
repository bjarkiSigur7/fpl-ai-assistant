"use client";

/**
 * Static-build scan path: the public GitHub Pages desk has no backend, so the
 * screenshot goes to the scan proxy (a tiny Vercel function holding the Gemini
 * key, NEXT_PUBLIC_SCAN_PROXY_URL), which answers the recognized cards; matching
 * onto player codes then runs right here in the browser (lib/scanMatch.ts)
 * against the bundle pool the page already loaded. The result is the same
 * ScanTeamResponse the local FastAPI endpoint serves — pages never notice the
 * mode.
 */

import { matchSquad, type SeenCard } from "./scanMatch";
import { staticPredictions } from "./staticBundle";
import { ScanTeamError, type ScanTeamRequest, type ScanTeamResponse } from "./types";

/** Production scan proxy; override per-build with NEXT_PUBLIC_SCAN_PROXY_URL. */
export const SCAN_PROXY_URL: string = process.env.NEXT_PUBLIC_SCAN_PROXY_URL ?? "";

interface ProxyResponse {
  cards: SeenCard[];
  model: string;
}

export async function staticScanTeam(req: ScanTeamRequest): Promise<ScanTeamResponse> {
  if (!SCAN_PROXY_URL) throw new ScanTeamError(405, "Scanning is not configured on this build.");
  const res = await fetch(`${SCAN_PROXY_URL}/api/scan-team`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "application/json" },
    body: JSON.stringify(req),
  });
  if (!res.ok) {
    const body = (await res.json().catch(() => null)) as { detail?: unknown } | null;
    const detail =
      body !== null && typeof body.detail === "string"
        ? body.detail
        : `Scan failed (HTTP ${res.status}).`;
    throw new ScanTeamError(res.status, detail);
  }
  const proxy = (await res.json()) as ProxyResponse;

  const preds = await staticPredictions();
  const shorts = new Map(preds.teams.map((t) => [t.team_code, t.short_name]));
  const pool = preds.rows.map((p) => ({
    player_code: p.player_code,
    web_name: p.web_name,
    position: p.position,
    price: p.price,
    team_short: shorts.get(p.team_code) ?? null,
  }));

  const matches = matchSquad(proxy.cards, pool);
  const codes: number[] = [];
  for (const m of matches) {
    if (m.player_code !== null && !codes.includes(m.player_code) && codes.length < 15) {
      codes.push(m.player_code);
    }
  }
  return {
    players: matches.map((m) => ({
      seen_name: m.seen.name,
      seen_club: m.seen.club,
      seen_price: m.seen.price,
      seen_position: m.seen.position,
      player_code: m.player_code,
      web_name: m.web_name,
      team_short: m.team_short,
      score: m.score,
    })),
    codes,
    unmatched: matches.filter((m) => m.player_code === null).map((m) => m.seen.name),
    model: proxy.model,
  };
}
