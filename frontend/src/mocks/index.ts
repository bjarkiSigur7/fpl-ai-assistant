/**
 * Mock API router (NEXT_PUBLIC_MOCK=1): resolves the same paths the real FastAPI
 * backend serves, from the deterministic fixtures in ./data. A tiny in-memory state
 * machine simulates POST /api/refresh so the settings page can demo live polling.
 */

import {
  chipCurvesResponse,
  initialSquadRecommendation,
  myTeamResponse,
  playerDetail,
  predictionsResponse,
  stateResponse,
} from "./data";
import type { RefreshStatus } from "@/lib/types";

/** Simulated latency so skeletons are visible in mock mode (ms). */
const MOCK_DELAY_MS = 350;

const sleep = (ms: number) => new Promise<void>((resolve) => setTimeout(resolve, ms));

export class MockNotFound extends Error {
  status = 404;
  constructor(path: string) {
    super(`mock: no route for ${path}`);
  }
}

// ---------------------------------------------------------------------------
// refresh simulation
// ---------------------------------------------------------------------------

const REFRESH_SCRIPT: [number, string][] = [
  [0, "refresh: snapshot"],
  [300, "snapshot: archived to data/raw/fpl_api/snapshots/2026-07-22"],
  [600, "  season=2025 (2025-26)  next_gw=None  players=11,412,508"],
  [900, "  2026-27 launch check: not live yet (API still serving 2025)"],
  [1200, "refresh: incremental pulls"],
  [1700, "  vaastav 2025 (fill-if-missing) ..."],
  [2400, "  football-data E0 2025 ..."],
  [3100, "  Understat league 2025 ..."],
  [3800, "  ClubElo snapshot ..."],
  [4300, "build: seasons 2016..2025 (from raw vaastav dirs)"],
  [5600, "  player_match.parquet: 253,568 rows"],
  [5900, "  player_gw.parquet: 244,425 rows"],
  [6100, "  fixtures.parquet: 3,800 rows"],
  [6400, "train: using existing artifacts (retrain with `fplai train`)"],
  [6800, "predict: backtest 2025 GW34+ (horizon 5) — 5,830 player-fixture rows"],
  [8200, "  top-10 xP for GW34: M.Salah (MID, 13.1m, 6.53), B.Fernandes (MID, 9.0m, 6.40), ..."],
  [8600, "predict: wrote data/processed/predictions.parquet and predictions_gw.parquet"],
  [8900, "refresh: data refresh complete"],
];

const REFRESH_TOTAL_MS = 9200;

let refreshStartedAt: number | null = null;
let lastFinished: { started: string; finished: string } | null = null;

function refreshStatus(): RefreshStatus {
  if (refreshStartedAt !== null) {
    const elapsed = Date.now() - refreshStartedAt;
    if (elapsed < REFRESH_TOTAL_MS) {
      const log = REFRESH_SCRIPT.filter(([t]) => t <= elapsed).map(([, line]) => line);
      const stage =
        elapsed < 1200 ? "snapshot" : elapsed < 4300 ? "pulls" : elapsed < 6400 ? "build" : "predict";
      return {
        running: true,
        stage,
        started_utc: new Date(refreshStartedAt).toISOString(),
        finished_utc: null,
        ok: null,
        log,
      };
    }
    lastFinished = {
      started: new Date(refreshStartedAt).toISOString(),
      finished: new Date(refreshStartedAt + REFRESH_TOTAL_MS).toISOString(),
    };
    refreshStartedAt = null;
  }
  return {
    running: false,
    stage: null,
    started_utc: lastFinished?.started ?? "2026-07-22T06:00:04Z",
    finished_utc: lastFinished?.finished ?? "2026-07-22T06:30:11Z",
    ok: true,
    log: lastFinished ? REFRESH_SCRIPT.map(([, line]) => line) : [],
  };
}

// ---------------------------------------------------------------------------
// router
// ---------------------------------------------------------------------------

export async function mockFetch<T>(path: string): Promise<T> {
  await sleep(MOCK_DELAY_MS);
  const [pathname, query] = path.split("?");
  const params = new URLSearchParams(query ?? "");

  if (pathname === "/api/health") return { status: "ok" } as T;
  if (pathname === "/api/state") return stateResponse() as T;
  if (pathname === "/api/predictions") return predictionsResponse() as T;
  if (pathname === "/api/dream-team") {
    const { dreamTeamResponse } = await import("./data");
    return dreamTeamResponse() as T;
  }
  if (pathname === "/api/recommendation") return initialSquadRecommendation() as T;
  if (pathname === "/api/chip-curves") {
    const entry = params.get("entry_id");
    return chipCurvesResponse(entry !== null ? Number(entry) : null) as T;
  }
  if (pathname === "/api/refresh/status") return refreshStatus() as T;

  const myTeam = /^\/api\/my-team\/(\d+)$/.exec(pathname);
  if (myTeam) return myTeamResponse(Number(myTeam[1])) as T;

  const player = /^\/api\/players\/(\d+)$/.exec(pathname);
  if (player) {
    const detail = playerDetail(Number(player[1]));
    if (!detail) throw new MockNotFound(path);
    return detail as T;
  }

  throw new MockNotFound(path);
}

export async function mockPost<T>(path: string): Promise<T> {
  await sleep(150);
  if (path === "/api/refresh") {
    if (refreshStartedAt === null) refreshStartedAt = Date.now();
    return refreshStatus() as T;
  }
  throw new MockNotFound(path);
}
