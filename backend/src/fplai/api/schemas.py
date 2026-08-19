"""Pydantic response models for the FastAPI layer.

These are the shapes the frontend consumes verbatim (its TS types are generated from
this app's OpenAPI schema), so field names must match the optimizer/model contracts
exactly.  Models that already exist as pydantic contracts are **imported and re-used**
— :class:`~fplai.optimizer.plans.Recommendation`, :class:`~fplai.optimizer.plans.DreamTeam`
and :class:`~fplai.optimizer.state.SquadState` are returned as-is, never redefined.
Frames from ``predictions(.gw).parquet`` keep their column names verbatim
(``xp_appearance`` … ``xp_other``, ``q0``/``q1``/``q2``/``mu1``/``mu2``).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from fplai.optimizer.plans import Recommendation
from fplai.optimizer.state import SquadState

__all__ = [
    "ChipCurvePoint",
    "ChipCurvesResponse",
    "DataFreshness",
    "FixturePrediction",
    "HealthResponse",
    "JobStatusResponse",
    "MyTeamResponse",
    "PlayerDetail",
    "PlayerGwHistoryRow",
    "PlayerIdentity",
    "PlayerPrediction",
    "PredictionsResponse",
    "RateTeamRequest",
    "RefreshStatusResponse",
    "SeasonStateModel",
    "SnapshotFreshness",
    "StateResponse",
]


class HealthResponse(BaseModel):
    """Liveness probe payload."""

    status: Literal["ok"] = "ok"
    service: str = "fplai-api"
    version: str
    time_utc: datetime


# --------------------------------------------------------------------------------------
# /state
# --------------------------------------------------------------------------------------


class SeasonStateModel(BaseModel):
    """Mirror of :class:`fplai.data.fpl_api.SeasonState` (a dataclass, mirrored 1:1)."""

    season: int = Field(description="Season start year (2025 = the 2025-26 season)")
    is_live_2026_27: bool
    current_gw: int | None
    next_gw: int | None
    next_deadline_utc: datetime | None
    total_players: int
    static_content_url: str


class SnapshotFreshness(BaseModel):
    """The latest daily FPL API snapshot on disk."""

    date: str = Field(description="Snapshot directory name, YYYY-MM-DD")
    bootstrap_mtime_utc: datetime | None = None
    fixtures_mtime_utc: datetime | None = None


class DataFreshness(BaseModel):
    """When each data artifact was last written."""

    latest_snapshot: SnapshotFreshness | None = None
    processed: dict[str, datetime] = Field(
        default_factory=dict, description="processed/ artifact file name -> mtime (UTC)"
    )
    models_trained_utc: datetime | None = Field(
        default=None, description="manifest.json created_utc, when model artifacts exist"
    )
    predictions_season: int | None = Field(
        default=None, description="Season covered by predictions_gw.parquet, when present"
    )
    predictions_gws: list[int] = Field(
        default_factory=list,
        description="GWs covered by predictions_gw.parquet for that season",
    )


class StateResponse(BaseModel):
    """Season state + data freshness + model manifest + launch countdown."""

    server_time_utc: datetime
    season_state: SeasonStateModel | None = Field(
        default=None, description="Parsed from the latest on-disk snapshot (no network)"
    )
    pre_launch: bool = Field(
        description="True while the FPL API still serves the finished 2025-26 season"
    )
    gw1_deadline_utc: datetime | None = Field(
        default=None,
        description="2026-27 GW1 deadline (announced 2026-08-21 17:30 UTC) or the "
        "API's next deadline once live",
    )
    seconds_to_gw1_deadline: float | None = None
    days_to_gw1_deadline: float | None = None
    freshness: DataFreshness
    model_manifest: dict[str, Any] | None = Field(
        default=None, description="data/models/manifest.json verbatim, when present"
    )


# --------------------------------------------------------------------------------------
# /predictions and /players
# --------------------------------------------------------------------------------------


class FixturePrediction(BaseModel):
    """One player-fixture xP row from ``predictions.parquet`` (assemble contract names)."""

    season: int
    gw: int
    fpl_fixture_id: int
    kickoff_utc: datetime | None = Field(
        default=None, description="Joined from fixtures.parquet (best-effort)"
    )
    team_code: int | None = None
    opponent_code: int | None = None
    opponent_short_name: str | None = None
    was_home: bool | None = None
    position: str | None = None
    price: int | None = None
    xp: float
    xp_appearance: float | None = None
    xp_goals: float | None = None
    xp_assists: float | None = None
    xp_cs: float | None = None
    xp_concede: float | None = None
    xp_saves: float | None = None
    xp_defcon: float | None = None
    xp_bonus: float | None = None
    xp_cards: float | None = None
    xp_other: float | None = None
    q0: float | None = None
    q1: float | None = None
    q2: float | None = None
    mu1: float | None = None
    mu2: float | None = None


class PlayerPrediction(BaseModel):
    """One player-GW xP row from ``predictions_gw.parquet`` + identity joins."""

    player_code: int
    web_name: str | None = None
    position: str | None = None
    team_code: int | None = None
    team_name: str | None = None
    team_short_name: str | None = None
    price: int | None = Field(default=None, description="£0.1m units")
    n_fixtures: int = 1
    xp: float
    xp_appearance: float | None = None
    xp_goals: float | None = None
    xp_assists: float | None = None
    xp_cs: float | None = None
    xp_concede: float | None = None
    xp_saves: float | None = None
    xp_defcon: float | None = None
    xp_bonus: float | None = None
    xp_cards: float | None = None
    xp_other: float | None = None
    q0: float | None = Field(default=None, description="P(0 minutes across the whole GW)")
    fixtures: list[FixturePrediction] = Field(
        default_factory=list, description="Per-fixture breakdown (2 entries on DGWs)"
    )


class PredictionsResponse(BaseModel):
    """Paginated per-player predictions for one (season, gw)."""

    season: int
    gw: int
    gws: list[int] = Field(
        default_factory=list,
        description="Every GW of this season available in predictions_gw.parquet",
    )
    total: int = Field(description="Matching players before pagination")
    limit: int
    offset: int
    sort: str
    order: Literal["asc", "desc"]
    players: list[PlayerPrediction]


class PlayerIdentity(BaseModel):
    """Cross-season identity row from ``players.parquet``."""

    player_code: int
    web_name: str | None = None
    first_name: str | None = None
    second_name: str | None = None
    understat_id: int | None = None
    opta_code: str | None = None


class PlayerGwHistoryRow(BaseModel):
    """One realized player-GW from ``player_gw.parquet`` (column names verbatim)."""

    season: int
    gw: int
    n_fixtures: int
    minutes: int
    total_points: int
    goals_scored: int
    assists: int
    clean_sheets: int
    goals_conceded: int
    saves: int
    bonus: int
    bps: int
    yellow_cards: int
    red_cards: int
    starts: int | None = None
    defensive_contribution: int | None = None
    xg: float | None = None
    xa: float | None = None
    xgc: float | None = None
    value: int = Field(description="Price that GW, £0.1m units")
    selected_by_percent: float | None = None


class PlayerDetail(BaseModel):
    """Identity + season history + upcoming predictions for one player."""

    player: PlayerIdentity
    seasons: list[int] = Field(description="Seasons with player_gw history rows")
    season: int | None = Field(default=None, description="Season the history covers")
    team_code: int | None = None
    team_name: str | None = None
    team_short_name: str | None = None
    position: str | None = None
    price: int | None = Field(default=None, description="Latest known price, £0.1m units")
    history: list[PlayerGwHistoryRow] = Field(default_factory=list)
    upcoming: list[FixturePrediction] = Field(
        default_factory=list, description="Per-fixture xP breakdown from predictions.parquet"
    )


# --------------------------------------------------------------------------------------
# /rate-team
# --------------------------------------------------------------------------------------


class RateTeamRequest(BaseModel):
    """POST /api/rate-team body: a full 15-man squad by player code.

    ``season``/``gw`` default to the live prediction window's first GW.  The list
    length (and every other legality rule) is validated by the endpoint, not this
    schema, so an illegal squad always gets the contract's 422
    ``{"detail": [rule, ...]}`` shape rather than FastAPI's loc/msg records.  The
    response model is :class:`fplai.optimizer.rating.RatingResult`, served verbatim.
    """

    player_codes: list[int] = Field(
        description="Exactly 15 FPL player codes (element.code, the stable key)"
    )
    season: int | None = Field(
        default=None, description="Season start year; None = the live prediction season"
    )
    gw: int | None = Field(
        default=None, description="First GW of the rating window; None = earliest on disk"
    )


# --------------------------------------------------------------------------------------
# /scan-team
# --------------------------------------------------------------------------------------


class ScanTeamRequest(BaseModel):
    """POST /api/scan-team body: a squad screenshot for Gemini recognition.

    The frontend downscales/re-encodes before posting, so the payload stays small;
    the endpoint still hard-caps the decoded size. Response codes: 503 when
    ``FPLAI_GEMINI_API_KEY`` is not configured, 502 when Gemini fails, 404 when the
    live roster is missing on disk.
    """

    image_base64: str = Field(description="Base64 screenshot payload, no data: URI prefix")
    mime_type: str = Field(
        default="image/jpeg", description="Image MIME type (image/jpeg, image/png, image/webp)"
    )


class ScannedPlayer(BaseModel):
    """One recognized card: what Gemini saw plus the roster resolution (if any)."""

    seen_name: str
    seen_club: str | None = None
    seen_price: float | None = Field(default=None, description="£m as printed on the card")
    seen_position: str | None = None
    player_code: int | None = Field(default=None, description="None when no roster match")
    web_name: str | None = None
    team_short: str | None = None
    score: float = Field(description="Match confidence 0-1; 0 when unmatched")


class ScanTeamResponse(BaseModel):
    """200 body of POST /api/scan-team."""

    players: list[ScannedPlayer]
    codes: list[int] = Field(description="Matched player codes, unique, capped at 15")
    unmatched: list[str] = Field(description="Seen names that resolved to no roster player")
    model: str = Field(description="Gemini model id that read the screenshot")


# --------------------------------------------------------------------------------------
# /chip-curves
# --------------------------------------------------------------------------------------


class ChipCurvePoint(BaseModel):
    """One chip-curve point (``optimizer.chips.chip_ev_curves`` contract, v1 + v2).

    The v2 columns (``sd`` … ``n_rollouts``) come from the Monte Carlo season
    simulation (``optimizer.season_sim`` — ``ChipSimReport.to_frame()`` feeds
    ``chip_curves.parquet`` v2). They are additive and nullable: pre-simulation
    files simply serve ``None`` for all of them.
    """

    chip: str
    gw: int
    objective: float | None = None
    delta_vs_no_chip: float | None = None
    evaluated: bool | None = None
    skip_reason: str | None = None
    window_last_gw: int | None = None
    urgency: float | None = None
    # --- season-simulation v2 columns (nullable, additive) ---------------------------
    sd: float | None = None
    p_best_week: float | None = None
    p_beats_hold: float | None = None
    n_rollouts: int | None = None


class ChipCurvesResponse(BaseModel):
    """Chip EV curves from ``data/processed/chip_curves.parquet``."""

    generated_at: datetime = Field(description="File mtime (UTC)")
    curves: list[ChipCurvePoint]


# --------------------------------------------------------------------------------------
# /my-team and /refresh (background jobs)
# --------------------------------------------------------------------------------------


class MyTeamResponse(BaseModel):
    """Squad state + the weekly Recommendation for one FPL entry."""

    entry_id: int
    generated_at: datetime
    source: Literal["memory", "cache", "shared"] = Field(
        description="memory = this process solved it; cache = per-entry cache file; "
        "shared = data/processed/recommendation.json"
    )
    squad_state: SquadState | None = Field(
        default=None, description="None when only the shared recommendation is available"
    )
    recommendation: Recommendation


class JobStatusResponse(BaseModel):
    """Status of one background job (my-team solve); also the 202 body."""

    job: str
    state: Literal["idle", "running", "done", "error"]
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error: str | None = None
    detail: str | None = None
    status_url: str | None = None


class RefreshStatusResponse(BaseModel):
    """Status of the single-flight refresh pipeline job, with a captured log tail."""

    state: Literal["idle", "running", "done", "error"]
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error: str | None = None
    log_tail: list[str] = Field(default_factory=list)
