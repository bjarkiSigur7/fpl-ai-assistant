/**
 * TypeScript mirrors of the backend contracts.
 *
 * Field names are VERBATIM copies of the backend pydantic models / parquet columns
 * (snake_case in transit, no remapping):
 *   - backend/src/fplai/optimizer/plans.py  (Recommendation, TransferPair, ChipAdvice,
 *     DreamTeam, StabilityEntry)
 *   - backend/src/fplai/optimizer/milp.py   (PlanResult, GwPlan)
 *   - backend/src/fplai/optimizer/state.py  (SquadState, OwnedPlayer)
 *   - backend/src/fplai/data/fpl_api.py     (SeasonState)
 *   - data/processed/predictions_gw.parquet / predictions.parquet columns
 *     (fplai/models/assemble.py + fplai/pipeline.py)
 *
 * All prices are integers in 0.1m units. `season` is the start year (2025 = 2025-26).
 * Datetimes are ISO-8601 strings in transit.
 */

export type Position = "GKP" | "DEF" | "MID" | "FWD";

export const POSITIONS: readonly Position[] = ["GKP", "DEF", "MID", "FWD"];

/** Chip instance id, e.g. "wc1", "bb2" (state.py ChipId). */
export type ChipId = string;

// ---------------------------------------------------------------------------
// optimizer/state.py
// ---------------------------------------------------------------------------

export interface OwnedPlayer {
  player_code: number;
  purchase_price: number;
  current_price: number;
}

export interface SquadState {
  season: number;
  current_gw: number;
  squad: OwnedPlayer[];
  bank: number;
  free_transfers: number;
  chips_available: ChipId[];
  active_chip: ChipId | null;
}

// ---------------------------------------------------------------------------
// optimizer/milp.py
// ---------------------------------------------------------------------------

export interface GwPlan {
  gw: number;
  squad: number[];
  lineup: number[];
  bench_order: number[];
  captain: number;
  vice: number;
  transfers_in: number[];
  transfers_out: number[];
  hit_points: number;
  chip: ChipId | null;
  expected_points: number;
  bank: number;
  free_transfers: number | null;
}

export interface PlanResult {
  objective: number;
  gws: GwPlan[];
  solve_seconds: number;
  gap: number;
  status: string;
  n_variables: number;
  n_constraints: number;
}

// ---------------------------------------------------------------------------
// optimizer/plans.py
// ---------------------------------------------------------------------------

export interface TransferPair {
  player_in: number | null;
  player_in_name: string | null;
  player_out: number | null;
  player_out_name: string | null;
  position: string | null;
  xp_in: number;
  xp_out: number;
  xp_delta: number;
  out_q0: number | null;
  support_pct: number | null;
}

export interface ChipAdvice {
  chip: ChipId;
  verdict: "play" | "hold";
  planned_gw: number | null;
  ev_now: number | null;
  best_gw: number | null;
  best_ev: number | null;
  /** Season-simulation fields (plans.py v2) — null until `fplai simulate` has run. */
  e_gain_now?: number | null;
  sd?: number | null;
  p_best_week_now?: number | null;
  p_beats_hold?: number | null;
  recommended_gw?: number | null;
  p_best_week_reco?: number | null;
  /** "high" | "medium" | "low", derived from p_beats_hold bands. */
  confidence?: string | null;
  n_rollouts?: number | null;
  assumptions?: string[] | null;
}

export interface DreamTeam {
  season: number;
  gw: number;
  squad: number[];
  lineup: number[];
  bench_order: number[];
  captain: number | null;
  vice: number | null;
  formation: string | null;
  expected_points: number;
  objective: number;
  total_cost: number;
  solve_seconds: number | null;
}

export interface StabilityEntry {
  move: string;
  support_pct: number;
}

/** The weekly verdict (plans.py Recommendation). `plan` is the PlanResult as a dict. */
export interface Recommendation {
  season: number;
  gw: number;
  as_of: string;
  /** "hold" | "transfer" | "chip:<id>" | "initial-squad" */
  action: string;
  transfers: TransferPair[];
  hits: number;
  captain: number | null;
  vice: number | null;
  lineup: number[];
  bench_order: number[];
  squad: number[];
  formation: string | null;
  expected_points: number;
  objective: number;
  chip_advice: ChipAdvice[];
  dream_team: DreamTeam | null;
  plan: Partial<PlanResult>;
  stability: StabilityEntry[];
  rationale: string[];
}

// ---------------------------------------------------------------------------
// predictions (predictions_gw.parquet / predictions.parquet rows)
// ---------------------------------------------------------------------------

/** One (season, gw, player) row of predictions_gw.parquet. Components sum to xp. */
export interface GwPrediction {
  season: number;
  gw: number;
  player_code: number;
  n_fixtures: number;
  xp: number;
  xp_appearance: number;
  xp_goals: number;
  xp_assists: number;
  xp_cs: number;
  xp_concede: number;
  xp_saves: number;
  xp_defcon: number;
  xp_bonus: number;
  xp_cards: number;
  xp_other: number;
  /** P(0 minutes) across the whole GW. */
  q0: number;
  team_code: number;
  position: Position;
  price: number;
  web_name: string;
}

/** One per-fixture row of predictions.parquet (player detail upcoming view). */
export interface FixturePrediction {
  season: number;
  gw: number;
  player_code: number;
  fpl_fixture_id: number;
  xp: number;
  xp_appearance: number;
  xp_goals: number;
  xp_assists: number;
  xp_cs: number;
  xp_concede: number;
  xp_saves: number;
  xp_defcon: number;
  xp_bonus: number;
  xp_cards: number;
  xp_other: number;
  team_code: number;
  opponent_code: number;
  was_home: boolean;
  position: Position;
  price: number;
  q0: number;
  q1: number;
  q2: number;
  mu1: number | null;
  mu2: number | null;
  kickoff_utc: string | null;
  /** Joined server-side by the real API; mocks may omit it. */
  opponent_short_name?: string | null;
}

/** One row of teams.parquet (identity subset). */
export interface TeamInfo {
  team_code: number;
  name: string;
  short_name: string;
}

// ---------------------------------------------------------------------------
// Wire types — field-for-field mirrors of backend/src/fplai/api/schemas.py.
// These are what the real FastAPI backend sends; api.ts adapts them into the
// view models below (mock mode serves the view models directly).
// ---------------------------------------------------------------------------

/** schemas.SeasonStateModel (mirror of data/fpl_api.py SeasonState). */
export interface ApiSeasonState {
  season: number;
  is_live_2026_27: boolean;
  current_gw: number | null;
  next_gw: number | null;
  next_deadline_utc: string | null;
  total_players: number;
  static_content_url: string;
}

/** schemas.SnapshotFreshness. */
export interface ApiSnapshotFreshness {
  date: string;
  bootstrap_mtime_utc: string | null;
  fixtures_mtime_utc: string | null;
}

/** schemas.DataFreshness. */
export interface ApiDataFreshness {
  latest_snapshot: ApiSnapshotFreshness | null;
  /** processed/ artifact file name -> mtime (UTC ISO). */
  processed: Record<string, string>;
  models_trained_utc: string | null;
  predictions_season: number | null;
  predictions_gws: number[];
}

/** GET /api/state (schemas.StateResponse). */
export interface ApiStateResponse {
  server_time_utc: string;
  season_state: ApiSeasonState | null;
  pre_launch: boolean;
  gw1_deadline_utc: string | null;
  seconds_to_gw1_deadline: number | null;
  days_to_gw1_deadline: number | null;
  freshness: ApiDataFreshness;
  model_manifest: Record<string, unknown> | null;
}

/** schemas.FixturePrediction — one player-fixture row (predictions.parquet names). */
export interface ApiFixturePrediction {
  season: number;
  gw: number;
  fpl_fixture_id: number;
  kickoff_utc: string | null;
  team_code: number | null;
  opponent_code: number | null;
  opponent_short_name: string | null;
  was_home: boolean | null;
  position: string | null;
  price: number | null;
  xp: number;
  xp_appearance: number | null;
  xp_goals: number | null;
  xp_assists: number | null;
  xp_cs: number | null;
  xp_concede: number | null;
  xp_saves: number | null;
  xp_defcon: number | null;
  xp_bonus: number | null;
  xp_cards: number | null;
  xp_other: number | null;
  q0: number | null;
  q1: number | null;
  q2: number | null;
  mu1: number | null;
  mu2: number | null;
}

/** schemas.PlayerPrediction — one player-GW row + identity joins. */
export interface ApiPlayerPrediction {
  player_code: number;
  web_name: string | null;
  position: string | null;
  team_code: number | null;
  team_name: string | null;
  team_short_name: string | null;
  price: number | null;
  n_fixtures: number;
  xp: number;
  xp_appearance: number | null;
  xp_goals: number | null;
  xp_assists: number | null;
  xp_cs: number | null;
  xp_concede: number | null;
  xp_saves: number | null;
  xp_defcon: number | null;
  xp_bonus: number | null;
  xp_cards: number | null;
  xp_other: number | null;
  q0: number | null;
  fixtures: ApiFixturePrediction[];
}

/** GET /api/predictions (schemas.PredictionsResponse — paginated, one GW per call). */
export interface ApiPredictionsResponse {
  season: number;
  gw: number;
  /** Every GW of this season available in predictions_gw.parquet. */
  gws: number[];
  total: number;
  limit: number;
  offset: number;
  sort: string;
  order: "asc" | "desc";
  players: ApiPlayerPrediction[];
}

/** schemas.PlayerIdentity (players.parquet row). */
export interface ApiPlayerIdentity {
  player_code: number;
  web_name: string | null;
  first_name: string | null;
  second_name: string | null;
  understat_id: number | null;
  opta_code: string | null;
}

/** schemas.PlayerGwHistoryRow (player_gw.parquet names, verbatim). */
export interface ApiPlayerGwHistoryRow {
  season: number;
  gw: number;
  n_fixtures: number;
  minutes: number;
  total_points: number;
  goals_scored: number;
  assists: number;
  clean_sheets: number;
  goals_conceded: number;
  saves: number;
  bonus: number;
  bps: number;
  yellow_cards: number;
  red_cards: number;
  starts: number | null;
  defensive_contribution: number | null;
  xg: number | null;
  xa: number | null;
  xgc: number | null;
  value: number;
  selected_by_percent: number | null;
}

/** GET /api/players/{code} (schemas.PlayerDetail — identity nested under `player`). */
export interface ApiPlayerDetail {
  player: ApiPlayerIdentity;
  seasons: number[];
  season: number | null;
  team_code: number | null;
  team_name: string | null;
  team_short_name: string | null;
  position: string | null;
  price: number | null;
  history: ApiPlayerGwHistoryRow[];
  upcoming: ApiFixturePrediction[];
}

/** GET /api/chip-curves (schemas.ChipCurvesResponse). */
export interface ApiChipCurvesResponse {
  generated_at: string;
  curves: ChipCurvePoint[];
}

/** GET /api/my-team/{entry_id} 200 body (schemas.MyTeamResponse). */
export interface ApiMyTeamResponse {
  entry_id: number;
  generated_at: string;
  source: "memory" | "cache" | "shared";
  squad_state: SquadState | null;
  recommendation: Recommendation;
}

/** 202 body of /api/my-team/{id} + its /status endpoint (schemas.JobStatusResponse). */
export interface ApiJobStatus {
  job: string;
  state: "idle" | "running" | "done" | "error";
  started_at: string | null;
  finished_at: string | null;
  error: string | null;
  detail: string | null;
  status_url: string | null;
}

/** POST /api/refresh + GET /api/refresh/status (schemas.RefreshStatusResponse). */
export interface ApiRefreshStatus {
  state: "idle" | "running" | "done" | "error";
  started_at: string | null;
  finished_at: string | null;
  error: string | null;
  log_tail: string[];
}

// ---------------------------------------------------------------------------
// View models — the flattened shapes the pages consume. api.ts adapts the wire
// types above into these; mock mode (src/mocks) produces them directly.
// ---------------------------------------------------------------------------

export interface DataFreshness {
  snapshot_utc: string | null;
  processed_utc: string | null;
  predictions_utc: string | null;
  predictions_season: number | null;
  predictions_gws: number[];
}

export interface ModelManifest {
  schema: number;
  created_utc: string;
  train_window: {
    seasons: number[];
    before_season: number | null;
    before_gw: number | null;
    n_feature_rows: number;
    n_team_matches: number;
  };
  components: Record<string, { class?: string; version?: string }>;
}

/** Desk state view (adapted from ApiStateResponse by api.ts). */
export interface StateResponse {
  season: number;
  is_live_2026_27: boolean;
  current_gw: number | null;
  next_gw: number | null;
  next_deadline_utc: string | null;
  total_players: number;
  data_freshness: DataFreshness | null;
  model: ModelManifest | null;
  /** Compact model tag when no full manifest is available (static bundle meta). */
  model_version?: string | null;
}

/** Horizon-wide predictions view (assembled from per-GW ApiPredictionsResponse pages). */
export interface PredictionsResponse {
  season: number;
  gws: number[];
  as_of: string | null;
  teams: TeamInfo[];
  rows: GwPrediction[];
}

/** GET /api/players/{player_code} */
export interface PlayerGwHistory {
  season: number;
  gw: number;
  n_fixtures: number;
  minutes: number;
  total_points: number;
  goals_scored: number;
  assists: number;
  clean_sheets: number;
  bonus: number;
  xg: number | null;
  xa: number | null;
  value: number;
}

export interface PlayerDetail {
  player_code: number;
  web_name: string;
  first_name: string;
  second_name: string;
  position: Position;
  team_code: number;
  team_name: string | null;
  team_short_name: string | null;
  price: number;
  season: number;
  history: PlayerGwHistory[];
  upcoming: FixturePrediction[];
}

/** GET /api/my-team/{entry_id} */
export interface MyTeamResponse {
  entry_id: number;
  state: SquadState;
  recommendation: Recommendation;
}

/** chips.chip_ev_curves rows verbatim (schemas.ChipCurvePoint, v1 + sim v2). */
export interface ChipCurvePoint {
  chip: ChipId;
  gw: number;
  objective: number | null;
  delta_vs_no_chip: number | null;
  evaluated?: boolean | null;
  skip_reason?: string | null;
  window_last_gw?: number | null;
  urgency?: number | null;
  /** Season-simulation v2 columns — null when chip_curves.parquet predates the sim. */
  sd?: number | null;
  p_best_week?: number | null;
  p_beats_hold?: number | null;
  n_rollouts?: number | null;
}

export interface ChipCurvesResponse {
  season?: number;
  entry_id?: number | null;
  curves: ChipCurvePoint[];
}

/** GET /api/refresh/status (polled after POST /api/refresh). */
export interface RefreshStatus {
  running: boolean;
  stage: string | null;
  started_utc: string | null;
  finished_utc: string | null;
  ok: boolean | null;
  log: string[];
}

export interface HealthResponse {
  status: string;
}

// ---------------------------------------------------------------------------
// POST /api/rate-team — field-for-field mirror of the rate-team contract.
// Request/response snake_case verbatim; 422 bodies are {"detail": [str, ...]}.
// ---------------------------------------------------------------------------

export interface RateTeamRequest {
  /** Exactly 15 player codes (2 GKP / 5 DEF / 5 MID / 3 FWD, <=3 per club, <=£100.0m). */
  player_codes: number[];
  /** null -> the live prediction window's season. */
  season: number | null;
  /** null -> the live prediction window's first GW. */
  gw: number | null;
}

/** Score bands: >=95 ELITE, >=85 STRONG, >=70 SOLID, >=50 ROUGH, else FODDER. */
export type RateVerdict = "ELITE" | "STRONG" | "SOLID" | "ROUGH" | "FODDER";

export interface RateTeamPlayerRating {
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

export interface RateTeamWeakest {
  player_code: number;
  web_name: string;
  xp_horizon: number;
}

/** 200 body of POST /api/rate-team. */
export interface RateTeamResponse {
  /** clamp01((team - floor) / (optimal - floor)) * 100. */
  score: number;
  verdict: RateVerdict;
  season: number;
  from_gw: number;
  horizon: number;
  team_xp_gw1: number;
  team_xp_horizon: number;
  optimal_xp_horizon: number;
  floor_xp_horizon: number;
  best_xi_gw1: number[];
  /** "D-M-F". */
  formation_gw1: string;
  suggested_captain: number;
  player_ratings: RateTeamPlayerRating[];
  weakest: RateTeamWeakest[];
}

/** Thrown by api.rateTeam on a 422 — `detail` lists every violated squad rule. */
export class RateTeamValidationError extends Error {
  readonly status = 422;
  constructor(public readonly detail: string[]) {
    super(detail.join("; "));
    this.name = "RateTeamValidationError";
  }
}

// ---------------------------------------------------------------------------
// display helpers shared across the app
// ---------------------------------------------------------------------------

/** Human names for chip-id prefixes (plans.py CHIP_NAMES). */
export const CHIP_NAMES: Record<string, string> = {
  wc: "Wildcard",
  fh: "Free Hit",
  bb: "Bench Boost",
  tc: "Triple Captain",
  aoa: "All Out Attack",
  am: "Assistant Manager",
};

export function chipName(chipId: ChipId): string {
  const m = /^([a-z]+?)(\d+)$/.exec(chipId);
  const prefix = m ? m[1] : chipId;
  return CHIP_NAMES[prefix] ?? chipId.toUpperCase();
}
