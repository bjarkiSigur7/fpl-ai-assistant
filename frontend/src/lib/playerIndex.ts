"use client";

/**
 * Client-side player index: joins prediction rows (which carry web_name, position,
 * price, team_code per (gw, player)) into one lookup keyed by player_code, so squads
 * expressed as code lists (DreamTeam, Recommendation, GwPlan) can be rendered with
 * names, prices and xP without extra endpoints.
 */

import { useMemo } from "react";
import { usePredictions } from "./api";
import type { GwPrediction, Position, PredictionsResponse, TeamInfo } from "./types";

export interface IndexedPlayer {
  player_code: number;
  web_name: string;
  position: Position;
  price: number;
  team_code: number;
  team_short: string;
  /** gw -> summed xp for that GW. */
  xpByGw: Map<number, number>;
  /** gw -> q0 for that GW. */
  q0ByGw: Map<number, number>;
  /** Latest-GW row (components etc.) for the first horizon GW it appears in. */
  firstRow: GwPrediction;
  /** Total xP over the whole horizon. */
  xpHorizon: number;
}

export interface PlayerIndex {
  season: number;
  gws: number[];
  nextGw: number;
  byCode: Map<number, IndexedPlayer>;
  teams: Map<number, TeamInfo>;
  get(code: number): IndexedPlayer | undefined;
  name(code: number): string;
  /** Humanize a sensitivity move string like "buy:118748" -> "Buy M.Salah". */
  humanizeMove(move: string): string;
}

export function buildIndex(data: PredictionsResponse): PlayerIndex {
  const teams = new Map(data.teams.map((t) => [t.team_code, t]));
  const byCode = new Map<number, IndexedPlayer>();
  const gws = [...data.gws].sort((a, b) => a - b);
  for (const row of data.rows) {
    let entry = byCode.get(row.player_code);
    if (!entry) {
      entry = {
        player_code: row.player_code,
        web_name: row.web_name,
        position: row.position,
        price: row.price,
        team_code: row.team_code,
        team_short: teams.get(row.team_code)?.short_name ?? String(row.team_code),
        xpByGw: new Map(),
        q0ByGw: new Map(),
        firstRow: row,
        xpHorizon: 0,
      };
      byCode.set(row.player_code, entry);
    }
    entry.xpByGw.set(row.gw, (entry.xpByGw.get(row.gw) ?? 0) + row.xp);
    entry.q0ByGw.set(row.gw, row.q0);
    entry.xpHorizon += row.xp;
    if (row.gw < entry.firstRow.gw) entry.firstRow = row;
  }
  const nextGw = gws[0] ?? 1;
  const index: PlayerIndex = {
    season: data.season,
    gws,
    nextGw,
    byCode,
    teams,
    get: (code) => byCode.get(code),
    name: (code) => byCode.get(code)?.web_name ?? `#${code}`,
    humanizeMove: (move) => {
      const m = /^(buy|sell|captain|chip):(.+)$/.exec(move);
      if (!m) return move === "hold" ? "Hold" : move;
      const [, verb, arg] = m;
      const code = Number(arg);
      const label = Number.isFinite(code) ? index.name(code) : arg;
      if (verb === "chip") return arg === "none" ? "No chip" : `Chip ${arg}`;
      return `${verb === "buy" ? "Buy" : verb === "sell" ? "Sell" : "Captain"} ${label}`;
    },
  };
  return index;
}

/** SWR-backed hook returning the built index (undefined while loading). */
export function usePlayerIndex(): {
  index: PlayerIndex | undefined;
  isLoading: boolean;
  error: unknown;
} {
  const { data, isLoading, error } = usePredictions();
  const index = useMemo(() => (data ? buildIndex(data) : undefined), [data]);
  return { index, isLoading, error };
}
