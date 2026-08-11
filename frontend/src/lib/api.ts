"use client";

/**
 * Typed API client + SWR hooks — the desk's ONLY data layer. Three modes:
 *
 * - LOCAL (default): the real FastAPI backend at NEXT_PUBLIC_API_URL serves the
 *   `Api*` wire shapes in ./types (field names verbatim from
 *   backend/src/fplai/api/schemas.py); the adapters below map them into the
 *   flattened view models the pages consume.
 * - STATIC (NEXT_PUBLIC_STATIC=1): the public GitHub Pages build. Everything is
 *   served from the daily static bundle at {NEXT_PUBLIC_BASE_PATH}/data/ (the
 *   GitHub Actions workflow copies `fplai publish-static` output into
 *   public/data/ before `next build`); ./staticBundle adapts the bundle onto
 *   the same view models, and the AI-Rating runs client-side via ./rating.
 *   Local-only surfaces (my-team, refresh) are inert in this mode.
 * - MOCK (NEXT_PUBLIC_MOCK=1, ignored when STATIC): served from src/mocks/ so
 *   the UI runs standalone against deterministic demo data.
 *
 * Pages never branch on the mode — they consume the view models and the
 * mode-aware copy exported here (OFFLINE_HINT).
 */

import useSWR, { type SWRConfiguration, type SWRResponse } from "swr";
import type {
  ApiChipCurvesResponse,
  ApiFixturePrediction,
  ApiMyTeamResponse,
  ApiPlayerDetail,
  ApiPlayerPrediction,
  ApiPredictionsResponse,
  ApiRefreshStatus,
  ApiStateResponse,
  CaptaincyRow,
  ChipCurvesResponse,
  DreamTeam,
  FixtureOutlook,
  FixturePrediction,
  GwPrediction,
  ModelManifest,
  MyTeamResponse,
  PlayerDetail,
  Position,
  PredictionsResponse,
  RateTeamRequest,
  RateTeamResponse,
  Recommendation,
  RefreshStatus,
  SetPieceDuty,
  StateResponse,
  TeamInfo,
} from "./types";
import { RateTeamValidationError } from "./types";

export const API_URL: string =
  process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

/** Public GitHub Pages build: all data from the static bundle, no backend. */
export const STATIC: boolean = process.env.NEXT_PUBLIC_STATIC === "1";

export const MOCK: boolean = !STATIC && process.env.NEXT_PUBLIC_MOCK === "1";

/** Mode-aware "feed is down" copy for the pages' empty states. */
export const OFFLINE_HINT: string = STATIC
  ? "The published data bundle failed to load. The site refreshes daily on GitHub Actions — try again in a moment."
  : "The model API is not answering. Start the backend (uvicorn fplai.api.app:app) or run the frontend with NEXT_PUBLIC_MOCK=1 for the standalone demo.";

/** Page size for /api/predictions (backend caps limit at 1000). */
const PREDICTIONS_PAGE = 1000;

export class ApiError extends Error {
  constructor(
    public readonly status: number,
    public readonly path: string,
  ) {
    super(`API ${status} on ${path}`);
  }
}

async function realFetch(path: string, init?: RequestInit): Promise<Response> {
  const res = await fetch(`${API_URL}${path}`, {
    headers: { Accept: "application/json" },
    ...init,
  });
  if (!res.ok) throw new ApiError(res.status, path);
  return res;
}

export async function fetchJson<T>(path: string): Promise<T> {
  if (MOCK) {
    const { mockFetch } = await import("@/mocks");
    return mockFetch<T>(path);
  }
  const res = await realFetch(path);
  return (await res.json()) as T;
}

export async function postJson<T>(path: string): Promise<T> {
  if (MOCK) {
    const { mockPost } = await import("@/mocks");
    return mockPost<T>(path);
  }
  const res = await realFetch(path, { method: "POST" });
  return (await res.json()) as T;
}

// ---------------------------------------------------------------------------
// wire -> view adapters
// ---------------------------------------------------------------------------

function adaptState(api: ApiStateResponse): StateResponse {
  const season = api.season_state;
  const f = api.freshness;
  const processedTimes = Object.entries(f.processed ?? {})
    .filter(([name]) => name.endsWith(".parquet"))
    .map(([, iso]) => iso)
    .sort();
  return {
    season: season?.season ?? f.predictions_season ?? 2025,
    is_live_2026_27: season?.is_live_2026_27 ?? false,
    current_gw: season?.current_gw ?? null,
    next_gw: season?.next_gw ?? null,
    // Real event deadline only when the game is live; pre-launch the ticker
    // falls back to the provisional GW1 deadline on its own.
    next_deadline_utc: api.pre_launch ? null : (season?.next_deadline_utc ?? api.gw1_deadline_utc),
    total_players: season?.total_players ?? 0,
    data_freshness: {
      snapshot_utc:
        f.latest_snapshot?.bootstrap_mtime_utc ?? f.latest_snapshot?.fixtures_mtime_utc ?? null,
      processed_utc: processedTimes.length ? processedTimes[processedTimes.length - 1] : null,
      predictions_utc: f.processed?.["predictions_gw.parquet"] ?? null,
      predictions_season: f.predictions_season,
      predictions_gws: f.predictions_gws ?? [],
    },
    model: (api.model_manifest as unknown as ModelManifest | null) ?? null,
  };
}

function adaptGwRow(p: ApiPlayerPrediction, season: number, gw: number): GwPrediction {
  return {
    season,
    gw,
    player_code: p.player_code,
    n_fixtures: p.n_fixtures,
    xp: p.xp,
    xp_appearance: p.xp_appearance ?? 0,
    xp_goals: p.xp_goals ?? 0,
    xp_assists: p.xp_assists ?? 0,
    xp_cs: p.xp_cs ?? 0,
    xp_concede: p.xp_concede ?? 0,
    xp_saves: p.xp_saves ?? 0,
    xp_defcon: p.xp_defcon ?? 0,
    xp_bonus: p.xp_bonus ?? 0,
    xp_cards: p.xp_cards ?? 0,
    xp_other: p.xp_other ?? 0,
    q0: p.q0 ?? 0,
    team_code: p.team_code ?? 0,
    position: (p.position as Position | null) ?? "MID",
    price: p.price ?? 0,
    web_name: p.web_name ?? `#${p.player_code}`,
  };
}

function adaptFixture(fx: ApiFixturePrediction, playerCode: number): FixturePrediction {
  return {
    season: fx.season,
    gw: fx.gw,
    player_code: playerCode,
    fpl_fixture_id: fx.fpl_fixture_id,
    xp: fx.xp,
    xp_appearance: fx.xp_appearance ?? 0,
    xp_goals: fx.xp_goals ?? 0,
    xp_assists: fx.xp_assists ?? 0,
    xp_cs: fx.xp_cs ?? 0,
    xp_concede: fx.xp_concede ?? 0,
    xp_saves: fx.xp_saves ?? 0,
    xp_defcon: fx.xp_defcon ?? 0,
    xp_bonus: fx.xp_bonus ?? 0,
    xp_cards: fx.xp_cards ?? 0,
    xp_other: fx.xp_other ?? 0,
    team_code: fx.team_code ?? 0,
    opponent_code: fx.opponent_code ?? 0,
    was_home: fx.was_home ?? false,
    position: (fx.position as Position | null) ?? "MID",
    price: fx.price ?? 0,
    q0: fx.q0 ?? 0,
    q1: fx.q1 ?? 0,
    q2: fx.q2 ?? 0,
    mu1: fx.mu1,
    mu2: fx.mu2,
    kickoff_utc: fx.kickoff_utc,
    opponent_short_name: fx.opponent_short_name,
  };
}

/**
 * Fetch every player of one (season, gw), paging past the backend limit.
 * Without season/gw the server picks the latest season + earliest GW on disk.
 */
async function fetchPredictionsGw(
  season?: number,
  gw?: number,
): Promise<ApiPredictionsResponse> {
  const scope = season !== undefined && gw !== undefined ? `season=${season}&gw=${gw}&` : "";
  const first = await fetchJson<ApiPredictionsResponse>(
    `/api/predictions?${scope}limit=${PREDICTIONS_PAGE}`,
  );
  const players = [...first.players];
  while (players.length < first.total) {
    const page = await fetchJson<ApiPredictionsResponse>(
      `/api/predictions?season=${first.season}&gw=${first.gw}&limit=${PREDICTIONS_PAGE}&offset=${players.length}`,
    );
    if (page.players.length === 0) break;
    players.push(...page.players);
  }
  return { ...first, players };
}

/** Assemble the horizon-wide predictions view from the per-GW paginated endpoint. */
async function fetchPredictionsView(): Promise<PredictionsResponse> {
  if (STATIC) {
    const { staticPredictions } = await import("./staticBundle");
    return staticPredictions();
  }
  if (MOCK) {
    const { mockFetch } = await import("@/mocks");
    return mockFetch<PredictionsResponse>("/api/predictions");
  }
  const first = await fetchPredictionsGw();
  const gws = first.gws.length ? first.gws : [first.gw];
  const rest = await Promise.all(
    gws.filter((g) => g !== first.gw).map((g) => fetchPredictionsGw(first.season, g)),
  );
  const pages = [first, ...rest];

  const teams = new Map<number, TeamInfo>();
  const rows: GwPrediction[] = [];
  for (const page of pages) {
    for (const p of page.players) {
      rows.push(adaptGwRow(p, page.season, page.gw));
      if (p.team_code !== null && !teams.has(p.team_code)) {
        teams.set(p.team_code, {
          team_code: p.team_code,
          name: p.team_name ?? p.team_short_name ?? String(p.team_code),
          short_name: p.team_short_name ?? String(p.team_code),
        });
      }
    }
  }
  return {
    season: first.season,
    gws: [...gws].sort((a, b) => a - b),
    as_of: null,
    teams: [...teams.values()],
    rows,
  };
}

function adaptPlayerDetail(api: ApiPlayerDetail): PlayerDetail {
  const code = api.player.player_code;
  return {
    player_code: code,
    web_name: api.player.web_name ?? `#${code}`,
    first_name: api.player.first_name ?? "",
    second_name: api.player.second_name ?? "",
    position: (api.position as Position | null) ?? "MID",
    team_code: api.team_code ?? 0,
    team_name: api.team_name,
    team_short_name: api.team_short_name,
    price: api.price ?? 0,
    season: api.season ?? 0,
    history: api.history.map((h) => ({
      season: h.season,
      gw: h.gw,
      n_fixtures: h.n_fixtures,
      minutes: h.minutes,
      total_points: h.total_points,
      goals_scored: h.goals_scored,
      assists: h.assists,
      clean_sheets: h.clean_sheets,
      bonus: h.bonus,
      xg: h.xg,
      xa: h.xa,
      value: h.value,
    })),
    upcoming: api.upcoming.map((fx) => adaptFixture(fx, code)),
  };
}

/** Derive a coarse pipeline stage label from the captured refresh log tail. */
function refreshStage(log: string[]): string | null {
  for (let i = log.length - 1; i >= 0; i--) {
    const m = /^([a-z][a-z-]*):/.exec(log[i]);
    if (m) return m[1];
  }
  return null;
}

function adaptRefresh(api: ApiRefreshStatus): RefreshStatus {
  return {
    running: api.state === "running",
    stage: api.state === "running" ? refreshStage(api.log_tail) : null,
    started_utc: api.started_at,
    finished_utc: api.finished_at,
    ok: api.state === "done" ? true : api.state === "error" ? false : null,
    log: api.log_tail,
  };
}

// ---------------------------------------------------------------------------
// SWR hooks
// ---------------------------------------------------------------------------

const BASE: SWRConfiguration = {
  revalidateOnFocus: false,
  shouldRetryOnError: false,
};

/** Season state + freshness + model manifest: the ticker's feed (30s). */
export function useDeskState(): SWRResponse<StateResponse> {
  return useSWR<StateResponse>(
    "/api/state",
    async (path: string) => {
      if (STATIC) {
        const { staticState } = await import("./staticBundle");
        return staticState();
      }
      if (MOCK) return fetchJson<StateResponse>(path);
      return adaptState(await fetchJson<ApiStateResponse>(path));
    },
    { ...BASE, refreshInterval: 30_000 },
  );
}

/** Per-player xP over the whole prediction horizon (60s). */
export function usePredictions(): SWRResponse<PredictionsResponse> {
  return useSWR<PredictionsResponse>("predictions-view", fetchPredictionsView, {
    ...BASE,
    refreshInterval: 60_000,
  });
}

/**
 * The weekly verdict. With an entry id -> /api/my-team/{id} (squad-aware verdict;
 * while the background solve is still running the backend answers 202 and we fall
 * back to the shared /api/recommendation until the next poll); without an entry id
 * -> /api/recommendation (the initial-squad / benchmark product).
 */
export function useRecommendation(
  entryId: number | null,
): SWRResponse<Recommendation> & { entryScoped: boolean } {
  // Static mode has no my-team endpoints: everyone gets the shared verdict.
  const scoped = !STATIC && entryId !== null;
  const key = scoped ? `/api/my-team/${entryId}` : "/api/recommendation";
  const swr = useSWR<Recommendation>(
    key,
    async (path: string) => {
      if (STATIC) {
        const { staticRecommendation } = await import("./staticBundle");
        return staticRecommendation();
      }
      if (entryId !== null) {
        if (MOCK) {
          const res = await fetchJson<MyTeamResponse>(path);
          return res.recommendation;
        }
        // 200 -> ApiMyTeamResponse; 202 -> job status (solve still running).
        const body = await fetchJson<Partial<ApiMyTeamResponse>>(path);
        if (body.recommendation) return body.recommendation;
        return fetchJson<Recommendation>("/api/recommendation");
      }
      return fetchJson<Recommendation>(path);
    },
    { ...BASE, refreshInterval: 60_000 },
  );
  return { ...swr, entryScoped: scoped };
}

export function usePlayerDetail(code: number | null): SWRResponse<PlayerDetail> {
  return useSWR<PlayerDetail>(
    code !== null ? `/api/players/${code}` : null,
    async (path: string) => {
      if (STATIC) {
        const { staticPlayerDetail } = await import("./staticBundle");
        return staticPlayerDetail(code!);
      }
      if (MOCK) return fetchJson<PlayerDetail>(path);
      return adaptPlayerDetail(await fetchJson<ApiPlayerDetail>(path));
    },
    BASE,
  );
}

export function useChipCurves(entryId: number | null): SWRResponse<ChipCurvesResponse> {
  const key =
    !STATIC && entryId !== null ? `/api/chip-curves?entry_id=${entryId}` : "/api/chip-curves";
  return useSWR<ChipCurvesResponse>(
    key,
    async (path: string) => {
      if (STATIC) {
        const { staticChipCurves } = await import("./staticBundle");
        return staticChipCurves();
      }
      if (MOCK) return fetchJson<ChipCurvesResponse>(path);
      const api = await fetchJson<ApiChipCurvesResponse>("/api/chip-curves");
      return { curves: api.curves };
    },
    BASE,
  );
}

/** Team-level per-fixture model outlook (fixture ticker source). */
export function useFixturesOutlook(): SWRResponse<FixtureOutlook[]> {
  return useSWR<FixtureOutlook[]>(
    "/api/fixtures-outlook",
    async (path: string) => {
      if (STATIC) {
        const { staticFixturesOutlook } = await import("./staticBundle");
        return staticFixturesOutlook();
      }
      return fetchJson<FixtureOutlook[]>(path);
    },
    BASE,
  );
}

/** Set-piece duty holders (pens / direct FKs / corners). */
export function useSetPieces(): SWRResponse<SetPieceDuty[]> {
  return useSWR<SetPieceDuty[]>(
    "/api/set-pieces",
    async (path: string) => {
      if (STATIC) {
        const { staticSetPieces } = await import("./staticBundle");
        return staticSetPieces();
      }
      return fetchJson<SetPieceDuty[]>(path);
    },
    BASE,
  );
}

/** Distributional captaincy comparison for the next GW. */
export function useCaptaincy(): SWRResponse<CaptaincyRow[]> {
  return useSWR<CaptaincyRow[]>(
    "/api/captaincy",
    async (path: string) => {
      if (STATIC) {
        const { staticCaptaincy } = await import("./staticBundle");
        return staticCaptaincy();
      }
      return fetchJson<CaptaincyRow[]>(path);
    },
    BASE,
  );
}

/** Refresh status; polls every 2s while `poll` is true. Inert in static mode. */
export function useRefreshStatus(poll: boolean): SWRResponse<RefreshStatus> {
  return useSWR<RefreshStatus>(
    "/api/refresh/status",
    async (path: string) => {
      if (STATIC) {
        const { staticRefreshStatus } = await import("./staticBundle");
        return staticRefreshStatus();
      }
      if (MOCK) return fetchJson<RefreshStatus>(path);
      return adaptRefresh(await fetchJson<ApiRefreshStatus>(`${path}?tail=200`));
    },
    { ...BASE, refreshInterval: !STATIC && poll ? 2_000 : 0 },
  );
}

/** GET /api/dream-team — the fresh-£100m benchmark squad on disk (DreamTeam verbatim). */
export async function fetchDreamTeam(): Promise<DreamTeam> {
  if (STATIC) {
    const { staticDreamTeam } = await import("./staticBundle");
    return staticDreamTeam();
  }
  return fetchJson<DreamTeam>("/api/dream-team");
}

/**
 * Score any 15-player squad against the model optimum. Local mode POSTs to
 * /api/rate-team; static mode computes the identical metric CLIENT-SIDE via
 * lib/rating.ts from the bundle (players + xp + rating anchors). Throws
 * RateTeamValidationError with every violated rule for an illegal squad and
 * ApiError on any other failure. Mock mode scores against the demo pool.
 */
export async function rateTeam(req: RateTeamRequest): Promise<RateTeamResponse> {
  if (STATIC) {
    const { staticRateTeam } = await import("./staticBundle");
    return staticRateTeam(req);
  }
  if (MOCK) {
    const { mockPost } = await import("@/mocks");
    return mockPost<RateTeamResponse>("/api/rate-team", req);
  }
  const path = "/api/rate-team";
  const res = await fetch(`${API_URL}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "application/json" },
    body: JSON.stringify(req),
  });
  if (res.status === 422) {
    const body = (await res.json().catch(() => null)) as { detail?: unknown } | null;
    const detail =
      body !== null && Array.isArray(body.detail)
        ? (body.detail as unknown[]).map(String)
        : ["Squad rejected by the rating engine."];
    throw new RateTeamValidationError(detail);
  }
  if (!res.ok) throw new ApiError(res.status, path);
  return (await res.json()) as RateTeamResponse;
}

export async function startRefresh(): Promise<RefreshStatus> {
  if (STATIC) {
    // Local-only feature: the public build's data is refreshed by the daily
    // GitHub Actions batch, never from the browser.
    throw new ApiError(405, "/api/refresh");
  }
  if (MOCK) return postJson<RefreshStatus>("/api/refresh");
  try {
    return adaptRefresh(await postJson<ApiRefreshStatus>("/api/refresh"));
  } catch (err) {
    // 409 = a refresh is already running — report its status instead of failing.
    if (err instanceof ApiError && err.status === 409) {
      return adaptRefresh(await fetchJson<ApiRefreshStatus>("/api/refresh/status"));
    }
    throw err;
  }
}
