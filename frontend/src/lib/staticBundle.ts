"use client";

/**
 * Static-mode data source (NEXT_PUBLIC_STATIC=1): the public GitHub Pages build.
 *
 * Reads the daily model bundle from {NEXT_PUBLIC_BASE_PATH}/data/<file>.json —
 * the GitHub Actions batch runs `fplai publish-static --out site-data` and copies
 * the bundle into public/data/ before `next build`, so the export ships its own
 * data — and adapts the bundle shapes onto the SAME view models the pages
 * consume in local mode. Pages never know which mode they run in.
 *
 * Bundle contract (docs/ARCHITECTURE.md release plan):
 *   meta.json             generated_utc, season, next_gw, next_deadline_utc,
 *                         model_version, window {from_gw, through_gw}
 *   players.json          the pool (identity + price + status + q0_gw1)
 *   xp.json               {gws, players: {"<code>": [xp per gw]}}
 *   predictions_gw1.json  per-player component rows for the window's first GW
 *   recommendation.json   Recommendation, verbatim
 *   dream_team.json       DreamTeam, verbatim
 *   chip_curves.json      chip-curve rows (v2 incl. sim columns), as records
 *   rating.json           client-side AI-Rating anchors (floor/optimal metric)
 *
 * The AI-Rating is computed CLIENT-SIDE here via lib/rating.ts (exact port of
 * the Python metric); local mode keeps POST /api/rate-team.
 */

import {
  RateTeamValidationError,
  type CaptaincyRow,
  type ChipCurvePoint,
  type ChipCurvesResponse,
  type DreamTeam,
  type FixtureOutlook,
  type FixturePrediction,
  type GwPrediction,
  type PlayerDetail,
  type Position,
  type PredictionsResponse,
  type RateTeamRequest,
  type RateTeamResponse,
  type Recommendation,
  type RefreshStatus,
  type SetPieceDuty,
  type StateResponse,
  type TeamInfo,
} from "./types";
import {
  RatingValidationError,
  rateTeam as rateTeamEngine,
  type RatingAnchors,
  type RatingPoolPlayer,
  type RatingXpTable,
} from "./rating";

/** Where the bundle lives; basePath-aware for GitHub Pages project sites. */
export const DATA_BASE: string = `${process.env.NEXT_PUBLIC_BASE_PATH ?? ""}/data`;

export class BundleError extends Error {
  constructor(
    public readonly status: number,
    public readonly file: string,
  ) {
    super(`static bundle ${status} on ${file}`);
    this.name = "BundleError";
  }
}

// ---------------------------------------------------------------------------
// bundle wire shapes
// ---------------------------------------------------------------------------

interface BundleMeta {
  generated_utc: string;
  season: number;
  next_gw: number | null;
  next_deadline_utc: string | null;
  model_version: string | null;
  window: { from_gw: number; through_gw: number };
}

interface BundlePlayer extends RatingPoolPlayer {
  team_short: string | null;
  status: string;
  q0_gw1: number | null;
  ownership?: number | null;
  news?: string | null;
  return_gw?: number | null;
  pen_order?: number | null;
}

/** Per-fixture detail nested in a predictions_gw1.json row (publisher v1+). */
interface BundleGw1Fixture {
  fpl_fixture_id: number;
  xp: number;
  opponent_code?: number | null;
  opponent_short?: string | null;
  was_home?: boolean | null;
  q0?: number | null;
  q1?: number | null;
  q2?: number | null;
  mu1?: number | null;
  mu2?: number | null;
}

/** predictions_gw1.json row — xp components for the window's first GW. */
interface BundleGw1Row {
  player_code: number;
  n_fixtures?: number | null;
  xp: number;
  xp_appearance?: number | null;
  xp_goals?: number | null;
  xp_assists?: number | null;
  xp_cs?: number | null;
  xp_concede?: number | null;
  xp_saves?: number | null;
  xp_defcon?: number | null;
  xp_bonus?: number | null;
  xp_cards?: number | null;
  xp_other?: number | null;
  q0?: number | null;
  fixtures?: BundleGw1Fixture[] | null;
}

// ---------------------------------------------------------------------------
// fetch + cache (the bundle is immutable per deploy; meta.json stays fresh
// so the ticker notices a new daily publish without a reload)
// ---------------------------------------------------------------------------

const cache = new Map<string, Promise<unknown>>();

async function fetchBundle<T>(file: string): Promise<T> {
  const res = await fetch(`${DATA_BASE}/${file}`, { headers: { Accept: "application/json" } });
  if (!res.ok) throw new BundleError(res.status, file);
  return (await res.json()) as T;
}

function loadCached<T>(file: string): Promise<T> {
  let p = cache.get(file) as Promise<T> | undefined;
  if (!p) {
    p = fetchBundle<T>(file).catch((err: unknown) => {
      cache.delete(file); // don't cache failures
      throw err;
    });
    cache.set(file, p);
  }
  return p;
}

const loadMeta = () => fetchBundle<BundleMeta>("meta.json");
const loadPlayers = () => loadCached<BundlePlayer[]>("players.json");
const loadXp = () => loadCached<RatingXpTable>("xp.json");
const loadGw1 = () => loadCached<BundleGw1Row[]>("predictions_gw1.json");
const loadAnchors = () => loadCached<RatingAnchors>("rating.json");

/** Some processed artifacts ship wrapped ({"dream_team": {...}}); accept both. */
function unwrap<T>(data: unknown, key: string): T {
  if (data !== null && typeof data === "object" && key in (data as Record<string, unknown>)) {
    return (data as Record<string, T>)[key];
  }
  return data as T;
}

// ---------------------------------------------------------------------------
// adapters: bundle -> view models
// ---------------------------------------------------------------------------

/** "2026-07-22T15:02:36+00:00" -> "v2026-07-22"; short tags pass through. */
function compactModelVersion(raw: string | null): string | null {
  if (raw === null) return null;
  const m = /^(\d{4}-\d{2}-\d{2})T/.exec(raw);
  return m ? `v${m[1]}` : raw;
}

export async function staticState(): Promise<StateResponse> {
  const meta = await loadMeta();
  const gws: number[] = [];
  for (let g = meta.window.from_gw; g <= meta.window.through_gw; g++) gws.push(g);
  return {
    season: meta.season,
    is_live_2026_27: meta.season >= 2026,
    current_gw: null,
    next_gw: meta.next_gw,
    next_deadline_utc: meta.next_deadline_utc,
    total_players: 0,
    data_freshness: {
      snapshot_utc: meta.generated_utc,
      processed_utc: meta.generated_utc,
      predictions_utc: meta.generated_utc,
      predictions_season: meta.season,
      predictions_gws: gws,
    },
    model: null,
    model_version: compactModelVersion(meta.model_version),
  };
}

const ZERO_COMPONENTS = {
  xp_appearance: 0,
  xp_goals: 0,
  xp_assists: 0,
  xp_cs: 0,
  xp_concede: 0,
  xp_saves: 0,
  xp_defcon: 0,
  xp_bonus: 0,
  xp_cards: 0,
  xp_other: 0,
} as const;

function componentsOf(row: BundleGw1Row | undefined) {
  if (!row) return ZERO_COMPONENTS;
  return {
    xp_appearance: row.xp_appearance ?? 0,
    xp_goals: row.xp_goals ?? 0,
    xp_assists: row.xp_assists ?? 0,
    xp_cs: row.xp_cs ?? 0,
    xp_concede: row.xp_concede ?? 0,
    xp_saves: row.xp_saves ?? 0,
    xp_defcon: row.xp_defcon ?? 0,
    xp_bonus: row.xp_bonus ?? 0,
    xp_cards: row.xp_cards ?? 0,
    xp_other: row.xp_other ?? 0,
  };
}

function teamsOf(players: BundlePlayer[]): Map<number, TeamInfo> {
  const teams = new Map<number, TeamInfo>();
  for (const p of players) {
    if (!teams.has(p.team_code)) {
      const short = p.team_short ?? String(p.team_code);
      teams.set(p.team_code, { team_code: p.team_code, name: short, short_name: short });
    }
  }
  return teams;
}

export async function staticPredictions(): Promise<PredictionsResponse> {
  const [meta, players, xp, gw1] = await Promise.all([
    loadMeta(),
    loadPlayers(),
    loadXp(),
    loadGw1(),
  ]);
  const gw1ByCode = new Map(gw1.map((r) => [r.player_code, r]));
  const firstGw = xp.gws[0];

  const rows: GwPrediction[] = [];
  for (const p of players) {
    const series = xp.players[String(p.player_code)];
    const gw1Row = gw1ByCode.get(p.player_code);
    const gws = series ? xp.gws : [firstGw]; // poolable but unscored -> one 0-xP row
    for (let i = 0; i < gws.length; i++) {
      const gw = gws[i];
      rows.push({
        season: meta.season,
        gw,
        player_code: p.player_code,
        n_fixtures: gw === firstGw ? (gw1Row?.n_fixtures ?? 1) : 1,
        xp: series?.[i] ?? 0,
        ...(gw === firstGw ? componentsOf(gw1Row) : ZERO_COMPONENTS),
        q0: gw1Row?.q0 ?? p.q0_gw1 ?? 0,
        team_code: p.team_code,
        position: p.position as Position,
        price: p.price,
        web_name: p.web_name,
        ownership: p.ownership ?? null,
        news: p.news ?? null,
        return_gw: p.return_gw ?? null,
        pen_order: p.pen_order ?? null,
      });
    }
  }
  return {
    season: meta.season,
    gws: [...xp.gws].sort((a, b) => a - b),
    as_of: meta.generated_utc,
    teams: [...teamsOf(players).values()],
    rows,
  };
}

export async function staticPlayerDetail(code: number): Promise<PlayerDetail> {
  const [meta, players, xp, gw1] = await Promise.all([
    loadMeta(),
    loadPlayers(),
    loadXp(),
    loadGw1(),
  ]);
  const p = players.find((x) => x.player_code === code);
  if (!p) throw new BundleError(404, `players.json#${code}`);
  const gw1Row = gw1.find((r) => r.player_code === code);
  const series = xp.players[String(code)];
  const firstGw = xp.gws[0];

  const upcoming: FixturePrediction[] = [];
  for (let i = 0; i < (series ? xp.gws.length : 0); i++) {
    const gw = xp.gws[i];
    const base = {
      season: meta.season,
      gw,
      player_code: code,
      team_code: p.team_code,
      position: p.position as Position,
      price: p.price,
      kickoff_utc: null,
    };
    const fixtures = gw === firstGw ? (gw1Row?.fixtures ?? []) : [];
    if (fixtures.length > 0) {
      // First GW: the bundle carries real per-fixture rows (opponent, venue, q's).
      for (const [j, fx] of fixtures.entries()) {
        upcoming.push({
          ...base,
          fpl_fixture_id: fx.fpl_fixture_id,
          xp: fx.xp,
          ...(j === 0 ? componentsOf(gw1Row) : ZERO_COMPONENTS),
          opponent_code: fx.opponent_code ?? 0,
          was_home: fx.was_home ?? false,
          q0: fx.q0 ?? gw1Row?.q0 ?? p.q0_gw1 ?? 0,
          q1: fx.q1 ?? 0,
          q2: fx.q2 ?? 0,
          mu1: fx.mu1 ?? null,
          mu2: fx.mu2 ?? null,
          opponent_short_name: fx.opponent_short ?? null,
        });
      }
    } else {
      // Later GWs are per-GW xp only: no opponent/fixture detail in the bundle.
      upcoming.push({
        ...base,
        fpl_fixture_id: gw, // synthetic id: one row per GW
        xp: series?.[i] ?? 0,
        ...(gw === firstGw ? componentsOf(gw1Row) : ZERO_COMPONENTS),
        opponent_code: 0,
        was_home: false,
        q0: gw1Row?.q0 ?? p.q0_gw1 ?? 0,
        q1: 0,
        q2: 0,
        mu1: null,
        mu2: null,
        opponent_short_name: null,
      });
    }
  }

  return {
    player_code: code,
    web_name: p.web_name,
    first_name: "",
    second_name: "",
    position: p.position as Position,
    team_code: p.team_code,
    team_name: p.team_short ?? null,
    team_short_name: p.team_short ?? null,
    price: p.price,
    season: meta.season,
    history: [], // the bundle ships no per-GW history
    upcoming,
  };
}

export async function staticRecommendation(): Promise<Recommendation> {
  const data = await loadCached<unknown>("recommendation.json");
  return unwrap<Recommendation>(data, "recommendation");
}

export async function staticDreamTeam(): Promise<DreamTeam> {
  const data = await loadCached<unknown>("dream_team.json");
  return unwrap<DreamTeam>(data, "dream_team");
}

export async function staticChipCurves(): Promise<ChipCurvesResponse> {
  const data = await loadCached<unknown>("chip_curves.json");
  const curves = Array.isArray(data)
    ? (data as ChipCurvePoint[])
    : unwrap<ChipCurvePoint[]>(data, "curves");
  return { curves };
}

export async function staticFixturesOutlook(): Promise<FixtureOutlook[]> {
  return loadCached<FixtureOutlook[]>("fixtures.json");
}

export async function staticSetPieces(): Promise<SetPieceDuty[]> {
  return loadCached<SetPieceDuty[]>("set_pieces.json");
}

export async function staticCaptaincy(): Promise<CaptaincyRow[]> {
  return loadCached<CaptaincyRow[]>("captaincy.json");
}

/** Static mode never runs the pipeline; the batch is GitHub Actions' job. */
export function staticRefreshStatus(): RefreshStatus {
  return { running: false, stage: null, started_utc: null, finished_utc: null, ok: null, log: [] };
}

// ---------------------------------------------------------------------------
// client-side AI-Rating (lib/rating.ts — exact port of the Python metric)
// ---------------------------------------------------------------------------

export async function staticRateTeam(req: RateTeamRequest): Promise<RateTeamResponse> {
  const [meta, players, xp, anchors] = await Promise.all([
    loadMeta(),
    loadPlayers(),
    loadXp(),
    loadAnchors(),
  ]);
  try {
    const result = rateTeamEngine(req.player_codes, {
      season: req.season ?? meta.season,
      pool: players,
      xp,
      anchors,
    });
    return result as RateTeamResponse;
  } catch (err) {
    if (err instanceof RatingValidationError) {
      throw new RateTeamValidationError(err.problems);
    }
    throw err;
  }
}
